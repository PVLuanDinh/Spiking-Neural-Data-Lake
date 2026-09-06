# L4 — persist what makes a relaunch resume instead of replay, within a ceiling

*Measured 2026-09-06 at `a3117b2`. The row has two halves and they came out very differently: the first is
**refused on measurement**, the second is built.*

## 1. The row, and what it assumed

> "The perception pointers and RNG stream positions, the affect SOM, the lifelong learner, the arm's Q-table
> — the same `_T_STATE` family, each with the cross-process red 2b built. … L4 gives `AppendBudget` its first
> live caller: a bytes/tick ceiling on this record class with a constructed-overload red (FR-003), derived
> from the measured night cadence."

The first half assumes a relaunch REPLAYS: that a restarted brain has lost those carriers and must re-learn
them. That assumption had never been measured across a real process boundary, so it was measured before
anything was built.

## 2. Measured: the relaunch already resumes

Three arms, each a SEPARATE interpreter, all under a throwaway `NORD_HOME` (the live lake was never opened):

| arm | what it is |
|---|---|
| **A** | a first life on a fresh lake, 120 ticks, persisting |
| **B** | a SECOND life — new process, same lake — 120 ticks. The relaunch under test. |
| **C** | a first life on an EMPTY lake, 120 ticks. The no-lake control, which separates "re-learned from the lake" from "re-learned from nothing". |

Over 5 seeds, comparing `state()` after EQUAL driving:

| | |
|---|---|
| keys IDENTICAL in both lives on every seed | **57** |
| keys differing on any seed | 33, nearly all on 1–3 seeds of 5 and by a few percent |
| the one large, consistent difference | `episode_retained` **+42.8 %, higher in the second life on 5 of 5 seeds** |

**The row's premise does not hold.** A relaunched brain resumes for the large majority of its reported state,
and where the lake changes the outcome it more often helps than hurts. Building the five-carrier persistence
the row names would be building against a premise the measurement refutes.

The first probe looked much worse — 63 keys "replayed" — because it compared a driven brain against a
**freshly constructed** one. That difference is mostly per-tick READINGS that refill on the next tick, not
lost learned state. Driving both lives equally is what separates the two, and it is the whole instrument.

## 3. The one apparent regression was two near-zero numbers

`access_index` read 0.000797 in the first life and **0.0** in the second on 4 of 5 seeds, with the no-lake
control matching the first life — which looks exactly like "restoring from the lake destroys a live reading",
scored at −64 % on the key's own scale.

It is not. Sampling `ignited` directly on every tick:

| arm | access_index | ignitions in 120 ticks |
|---|---|---|
| A first life | 0.000797 | **0** |
| B second life (120 and 240 ticks) | 0.0 | **0** |
| C no-lake | 0.000797 | **0** |

**The workspace never ignites in ANY arm**, so the whole comparison was between two essentially-zero
quantities, and the percentage came from dividing by a magnitude of 1e-3. That is a relative bar with no
absolute bar, and this project has the scar for it already. There is no relaunch regression here.

**Recorded separately, because it is a real fact this probe found and nothing else was looking for:** over
120 and 240 ticks, at three seeds, with or without a lake, **`ignited` never fires once.** `access_index` is
therefore reporting an EMA of a gate that does not open. That is not L4's row to fix, and it is not claimed
as a defect here — it is measured, dated, and left where the next reader of that surface will find it.

## 4. Disposition on the first half: REFUSED, and why that is the honest outcome

No carrier is persisted by this row. The measurement says the relaunch resumes; the row's premise was
untested when written; and persisting state to fix a replay that does not occur would add lake traffic, add
restore paths, and buy nothing measurable. If a future measurement exhibits a carrier that genuinely fails to
resume — the shape this probe was built to find and did not — it re-opens with that carrier named and its
cross-process red written first.

What the probe leaves behind is the instrument: the three-arm equal-driving comparison, which is what any
re-proposal has to run.

## 5. The second half: `AppendBudget`'s first live caller

Unaffected by any of the above, still owed, and it closes part of what L6 disclosed. See §6 for the
pre-registration and §7 for what it measured.

## 6. Pre-registration: `AppendBudget`'s first live caller

### 6.1 What is wrong today

`lake_engine.AppendBudget` has existed since T010-i with a measured per-tag baseline in
`tools/lake_append_baseline.json` and **no caller anywhere in the tree** — reachable and uncharged, P3 in its
plainest shape. It is also the only instrument this project has for bounding lake growth, which is what makes
its dormancy expensive rather than merely untidy: L6 removed the learned class's cap on the argument that
bounding belongs here, and the L2 panel then measured that this had no live caller at all.

### 6.2 The wire

- `home.set_append_observer(fn)` — a module-level hook called `(tag, n_bytes)` at each site where `home`
  knows the tag of what it is writing. It lives in `home` rather than in `BinLog` because a tag is a home
  concept and `BinLog` carries opaque payloads by design. It is module-global because this module's log
  cache already is, and threading a budget object through every writer would couple far more than it bounds.
  **Disclosed:** one observer per process, so two brains in one process would share it — which P4 forbids for
  other reasons. A raising observer must never break a durable write, so the call is guarded.
- `Brain` owns an `AppendBudget`, installs the observer at construction (persisting brains only — a
  non-persisting brain writes nothing to bound), and **closes the tick** in `tick()`.
- Violations are COUNTED and reported in `state()` as `append_floods` / `append_flood_last`, with their
  PROVENANCE rows. **Never raised:** a flood is a fact about the lake, and killing the tick over it would
  lose the brain to protect the disk.

### 6.3 Red and controls

- **RED (FR-003), on the live path:** a burst strictly above `FLOOD_FACTOR ×` the RECORDED p95 for `0x07`,
  written between two ticks through `home`'s own writer, must raise `append_floods` on the next tick. The
  overload is sized FROM the measured baseline, not from a number chosen for the test.
- **CONTROL, same test:** a calm tick with a small write must NOT trip it. A budget that fires on everything
  passes the red alone.
- **CONTROL 2:** an unbaselined tag far above any plausible ceiling must still never trip, because a budget
  nobody measured is not a budget. This is what stops the wire from inventing a bound for a class nobody
  baselined.
- **CONTROL 3:** a raising observer must not lose a write.

### 6.4 What this does NOT do

**It does not restore the bound L6 removed.** `0x01` has no baseline row, so by `AppendBudget`'s own contract
it can never trip for the learned class, wired or not. This caller bounds `0x03`, `0x04`, `0x07`, `0x08` and
`0x0A` — the five tags that carry a recorded p95 — and the learned class stays unbounded until someone
measures it. Stated here because the tempting sentence is that L4 closed L6's disclosure, and it does not.

## 7. Measured

| | |
|---|---|
| observer fires with tag and byte count | `(0x07, 111)` and `(0x01, 22)` on a two-write probe |
| a persisting brain owns the budget | yes; a non-persisting one does not |
| calm brain, 30 ticks | `append_floods = 0`, `append_flood_last = ""` |
| constructed overload above 8 × the 12,381 B p95 for `0x07` | **trips**, message names `FLOOD` and `0x07` |
| an unbaselined tag at 10 MB in one tick | does not trip |
| a raising observer | write succeeds and reads back |

Four tests in `tests/test_append_budget_wire.py`, plus an arm in `home`'s own self-check so `run_selfchecks`
covers the observer. The self-check arm writes to its OWN store rather than the shared self-check lake —
the first version of it wrote a probe into that lake and broke an assertion twelve lines below, which is a
mistake this file has now made twice and now carries a comment about.

### 7.1 Disposition

The wire LANDS. `AppendBudget` has a live caller, its red is reachable on the production path, and its two
controls hold. What is claimed is exactly that: the budget is now charged for the five baselined tags. What
is not claimed is any bound on the learned class, which remains L6's open disclosure and needs a measurement,
not a wire.
