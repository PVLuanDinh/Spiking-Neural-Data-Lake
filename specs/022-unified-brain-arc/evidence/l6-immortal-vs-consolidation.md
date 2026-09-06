# L6 (T091) — the immortal class and the sleep tick disagree

*Opened 2026-09-06 by L5's own instrument, which is the reason the instrument exists. Red L-R7 (Part F).
Pre-registration written BEFORE the measurement; §3 records what was measured, §4 the disposition applied.*

## 1. The contradiction, as measured

`substrate/immortal_retention.py` (with the rewrite arm the panel restored) reports exactly one durable
reclaimer that does not consult the retention label:

```
reclaimers of DURABLE records that do not consult the label: 1 [('substrate/home.py', 'consolidate')]
```

Both halves are on the live path:

- `lake_engine.RETENTION[0x01] = ("IMMORTAL", False, "_T_FACT: learned knowledge is never reclaimed")`.
  `reclaimable=False` is the registry's own word for *no reclaimer may drop this*.
- `cortex/brain.py`'s sleep tick, every eighth night when `_persist`, runs
  `LakeConsolidation().consolidate(records, trace=RecallTrace())` and then
  `KnowledgeLog.consolidate(survivors)`, which rewrites the one lake to the survivor set.
  `LakeConsolidation` splits records by `_src_class` into *re-derivable* and *learned*, caps the first at
  `rederivable_budget=3000` and the second at `learned_cap=250000`, and — with a trace — drops the
  never-recalled learned records first. Its own docstring says **USE IT OR LOSE IT**.

Exactly one of those two may survive (P2). The row does not pre-judge which.

## 2. Pre-registration

### 2.1 The measurement that decides it

The live lake is `C:\Users\vinhl\.snn_data_lake\brain_lake.bin` and the daemon is running on it, so it is
read **without a writer**: `BinLog.__init__` calls `_repair_torn_tail()`, which truncates. `BinLog.replay`
uses only `self.path` and opens the file read-only, so the census calls the real parser bound to a bare
path holder — the tree's own framing code, never a re-implementation (P2), and no write path is entered.
A record appended by the daemon mid-read ends the replay cleanly at the torn frame, so the census may
undercount by at most one record; that is disclosed rather than prevented.

Counted: total records, the per-tag census, the `_T_FACT` count, and that count split by `_src_class` into
learned and re-derivable.

### 2.2 The dispositions, fixed in advance

| What the measurement shows | Disposition |
|---|---|
| learned count **below** `learned_cap` | The cap has never bound. Removing it changes nothing observable today, and the label becomes true at zero behavioural cost. Apply the fix in §2.3 and say plainly that no record is known to have been lost. |
| learned count **at or above** `learned_cap` | The brain has already forgotten learned records. Apply the same fix, and additionally record how many records the cap discarded and that the loss is not recoverable — a correction of the "never forgets" claim, not only of the code. |
| the learned class turns out to be re-derivable in fact | The row DELETES rather than re-labels: `0x01` becomes COMPACTABLE with the re-derivation as its reconstruction proof, and L5's "never forgets" sentence narrows to say what it covers. |

A number that lands between these is not anticipated; if one does, it is recorded before it is interpreted.

### 2.3 The fix, named before the number is seen

The label is at TAG granularity and the reclamation decision is at RECORD granularity inside that tag, which
is how the two drifted apart. Both halves are made honest:

1. `LakeConsolidation.consolidate` consults `lake_engine.retention_of` and **never drops a record of a
   `reclaimable=False` class**. The re-derivable budget stays — those records reconstruct at boot from the
   same source, which is the COMPACTABLE contract applied per record — and its exemption is written into the
   `0x01` reason so the registry states what is true rather than what was intended.
2. `KnowledgeLog.consolidate` consults `retention_of` in the same function, which is what makes the label
   load-bearing at the drop site and what retires the `KNOWN_UNLABELED` entry.

Bounding does not disappear; it moves from *silently forget* to `AppendBudget`'s bytes/tick assert, which is
the plan's declared instrument for it (L4 gives it its first live caller). That is a real trade and it is
stated here: **this row makes the lake grow rather than let it forget.**

### 2.4 Red and control (L-R7)

- **Red:** a constructed reclaimer that drops a record of a `reclaimable=False` class must be refused by the
  policy, and the refusal must be reachable — a record that the old policy would have dropped is kept by the
  new one, demonstrated in the same command.
- **Control (not the method):** a COMPACTABLE tag reclaimed in the same pass must STILL be reclaimed. A fix
  that simply stops consolidating passes the red and fails this control, and is refused.
- **Second control:** the re-derivable budget must still bind, or the change has disabled compaction rather
  than scoped it.

### 2.5 What this row does not do

It does not touch the safety envelope, the daemon, or the live lake's contents. It does not add a second
retention registry. It does not claim recall: a record kept is not a record findable, which stays Phase 4's
business.

## 3. Measured

### 3.1 The census (live lake, read-only, 2026-09-06)

The positive control ran in the same command and fired first: the same census over a constructed log of 7
learned facts, 3 re-derivable facts and 1 foreign record returned exactly `(11, 7, 3)`. Only then was the
live lake read.

`C:\Users\vinhl\.snn_data_lake\brain_lake.bin` — 2.456 GB, **4,753,532 records**.

| tag | class | records | share |
|---|---|---|---|
| `0x0a` | `_T_COALITION` (IMMORTAL) | 3,812,834 | 80.21% |
| `0x04` | `_T_EPISODE` (IMMORTAL) | 538,705 | 11.33% |
| `0x08` | `_T_GROWTH_DELTA` (COMPACTABLE) | 257,401 | 5.41% |
| `0x03` | `_T_GROWTH` (COMPACTABLE) | 86,095 | 1.81% |
| `0x05` | `_T_SELFBUILD` (IMMORTAL) | 32,426 | 0.68% |
| `0x01` | `_T_FACT` (IMMORTAL) | 26,037 | 0.55% |
| `0x07` | `_T_STATE` (IMMORTAL) | 34 | 0.00% |

**The number the row turns on:**

| | |
|---|---|
| `_T_FACT` learned | **26,037** |
| `_T_FACT` re-derivable | **0** |
| `_T_FACT` unparsable | 0 |
| `learned_cap` | 250,000 — **has never bound** |
| `rederivable_budget` | 3,000 — **has never bound** |

The learned count is 10.4% of the cap. Every fact in the lake is learned; the re-derivable class, which the
budget exists for, holds nothing at all. **`LakeConsolidation` has therefore never dropped a single record.**

### 3.2 Two things the census found that the row was not looking for

**The growth pathology has MOVED, and the registry still records the old one.** `lake_engine`'s module
docstring says "97.64% of the live lake is repeated whole-substrate `_T_GROWTH` snapshots (83,405 records)",
measured 2026-08-05. Today `_T_GROWTH` is 86,095 records and **1.81%**; the lake is now 80.21%
`_T_COALITION`. Both numbers were true when taken. The stale one is recorded here rather than silently
overwritten, and the note in `lake_engine` is dated so a reader cannot mistake it for current.

**Four of the eleven declared classes have never held a record.** `_T_SPIKE` (0x02), `_T_CALIB` (0x06),
`_T_SYNDELTA` (0x09) and `_T_GOAL` (0x0b) are all absent from the live lake. A class labelled IMMORTAL that
has never been written is a label with no referent — retention's version of *reachability is not charge*.
`0x09` and `0x0b`'s own reasons already say they await a live writer, so two of the four are disclosed; the
other two are not, and this is where that is recorded. It is not fixed here: it is a census of the live lake,
which is too slow to run in a battery, so it belongs in a record rather than in the gate.

## 4. Disposition applied

The pre-registered branch is the first row of §2.2: **the cap has never bound.** Removing it changes nothing
observable on today's brain, and it makes the registry's sentence true at zero behavioural cost. **No learned
record is known to have been lost.**

The fix landed as pre-registered in §2.3, in three places:

1. `RETENTION[0x01]`'s reason now STATES the per-record exemption that was previously only implied: a fact
   whose `src` names a bulk re-derivable ingest may be reclaimed, because it reconstructs at boot from that
   same source — the COMPACTABLE contract applied per record. Everything else under the tag is learned and is
   never dropped. The measured 26,037 / 0 split is written into the reason so the claim carries its date.
2. `LakeConsolidation.consolidate` reads `retention_of(_T_FACT)` and refuses the learned cap while the class
   is `reclaimable=False`. The cap stays a field on purpose: **the registry decides, not the policy.** Flip
   `0x01` to reclaimable and the cap takes effect again without touching that file.
3. `KnowledgeLog.consolidate` reads the label in the same function and REFUSES a rewrite that would lose a
   learned fact. That is the guard at the drop site, and it is what retires the debt entry.

### 4.1 The reds and controls, run

| | Result |
|---|---|
| **L-R7 red** — a cap of 1 over three learned records | all three kept; the block that asserted the old capping is the red, and reverting either half of the fix fires it |
| **Control 1 (not the method)** — flip `0x01` to reclaimable, same call | caps again, with the use-it-or-lose-it recall ranking intact; the registry is restored in a `finally` and the restoration is asserted |
| **Control 2 (not the method)** — re-derivable bulk in the same pass | still reclaimed, 5 → 2; a change that merely stopped consolidating would pass the red and fail here |
| **Guard red** — a survivor set with one learned fact removed | `ValueError` naming the LEARNED loss |
| **Guard control** — a survivor set dropping only a re-derivable fact | permitted, `dropped == 1` |

### 4.2 The debt entry, paid out the day it was opened

`immortal_retention.KNOWN_UNLABELED` is empty. It was not emptied by hand: with the fix in place the census
stopped finding the reclaimer, and `stale_known()` — the arm that fires on a listed entry that no longer
reclaims — went red and demanded the payout. The emptied table keeps a dated note of what it held, because a
table that arrives empty teaches nothing.

### 4.3 The trade, stated plainly

**This row makes the lake grow rather than let it forget.** Bounding does not disappear; it moves from
*silently discard the least-recalled learned records* to `AppendBudget`'s bytes/tick assert, which is the
plan's declared instrument for growth and which L4 gives its first live caller. On today's numbers the trade
costs nothing — the cap was 10× above the live count — but it is a real commitment for a never-off brain, and
the class that dominates the lake (`_T_COALITION`, 80.21%) is IMMORTAL too and already carries a baselined
flood assert.

### 4.4 What this row did NOT do

It did not touch the safety envelope, the daemon, or the live lake's contents — the census opened the file
read-only through `BinLog.replay` bound to a bare path holder, never through a constructor that repairs. It
did not add a second retention registry. It did not make a kept record a findable one: recall stays Phase 4's
business.

### 4.5 A P3 event the row did not plan, and the gate caught it the same run

Putting `retention_of` on the sleep tick made `lake_engine` reachable from a live path, and `lake_engine`
imports `lake_migrate.TAG_DISPOSITIONS` at module top, which carried `lake_bridge` and `lake_segments` in
behind it. `dormancy_gate`'s stale-exemption arm went RED on all four `wire` rows in the same run that L6's
own gates went green — an exemption whose stated reason no longer holds is a defect, and this is that arm
working.

All four rows are retired. **Only one of the four is CHARGED.** `lake_engine`'s `retention_of` decides
whether a learned fact survives the night, which is an outcome, and L-R7 is the constructed red that proves
it. `lake_migrate`, `lake_bridge` and `lake_segments` are reachable through an import and nothing more:
their rows promised a live consumer at the US3 kernel paging tier (T064/T066), that consumer has not
arrived, and *BOUND EXPIRED AT T039 UNMET* is still true of all three. Retiring the rows removed their
standing-TODO lines, so the debt is recorded in a comment where the rows used to sit — **a retired exemption
must not be read as a satisfied promise.** Reachability is not charge.

The standing-TODO list falls from 9 open `wire` exemptions to 5.
