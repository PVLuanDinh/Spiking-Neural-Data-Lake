# L4 — persist what makes a relaunch resume instead of replay, within a ceiling

*Measured 2026-09-06 at `a3117b2`. READ SECTION 8 FIRST: an adversarial panel retracted this record's central
conclusion. Sections 2-4 are kept unaltered so the correction is legible, and each is marked with the section
that replaces it. The persistence half is RE-OPENED, not refused; the append half stands with its coverage
claim corrected from five tags to the tags actually observed.*

## 1. The row, and what it assumed

> "The perception pointers and RNG stream positions, the affect SOM, the lifelong learner, the arm's Q-table
> — the same `_T_STATE` family, each with the cross-process red 2b built. … L4 gives `AppendBudget` its first
> live caller: a bytes/tick ceiling on this record class with a constructed-overload red (FR-003), derived
> from the measured night cadence."

The first half assumes a relaunch REPLAYS: that a restarted brain has lost those carriers and must re-learn
them. That assumption had never been measured across a real process boundary, so it was measured before
anything was built.

## 2. **RETRACTED BY SECTION 8.1** -- Measured: the relaunch already resumes

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

## 3. **RETRACTED BY SECTION 8.2** -- The one apparent regression was two near-zero numbers

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

## 4. **REVERSED BY SECTION 8.3** -- Disposition on the first half: REFUSED, and why that is the honest outcome

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

---

## 8. The adversarial panel, and the RETRACTION it forced (2026-09-06)

Three lenses read commit `994f812`; twelve findings verified by separate refuting agents, **ten survived, two
were refuted.** One of them overturns this record's central conclusion, so §2, §3 and §4 are RETRACTED rather
than annotated. What follows replaces them.

### 8.1 RETRACTED: "the relaunch already resumes"

The claim rested on "57 of ~90 keys identical in both lives". The probe now ships
(`python -m snn_data_lake.cortex.brain --resume-probe`), states its universe, and separates keys that MOVE
during driving from keys that never move:

| | |
|---|---|
| numeric `state()` keys present in both lives | **93** |
| of those, keys that MOVE at all | **52** |
| identical in both lives on every seed, **among moving keys** | **17 of 52 (33 %)** |
| identical among keys that never move | 41 of 41 — uninformative by construction |

**Two thirds of the state that can differ does differ.** The 57 was inflated by 41 keys that could not have
differed, which is the same objection §2 raised against its own first probe and then committed itself.

### 8.2 **ITSELF RETRACTED -- see the block at the end of this section.** RETRACTED, and this is the serious one: "the workspace never ignites in ANY arm"

§3 reported 0 ignitions everywhere and concluded the `access_index` difference was two near-zero numbers.
**That was a broken counter** — it sampled `state()["ignited"]`, which does not read the flag the gate sets.
Counting `_access_state["ignited"]` directly, with arm A as the counter's own positive control:

| seed | ignitions, arm A (first life) | arm B (SECOND life) | arm C (no lake) |
|---|---|---|---|
| 0 | **1** | 0 | 1 |
| 1 | 0 | 0 | 0 |
| 2 | **1** | 0 | 1 |
| 3 | **1** | 0 | 1 |
| 4 | **1** | 0 | 1 |
| total | **4** | **0** | 4 |

**A first life ignites on 4 of 5 seeds. A second life on the same lake ignites on none.** The `access_index`
difference was never a comparison of two zeros — it was the visible edge of a real behavioural loss, and I
argued myself out of it with an instrument that could not count the thing it was counting.

That is the THIRD instrument defect inside this row's own apparatus, after the first probe's
freshly-constructed comparator and the relative-scale reading. All three pointed the same way: toward
concluding that nothing was wrong.

> **RETRACTION, 2026-09-06 -- see `evidence/l4c-ignition-is-coalition-size.md`.** The counter above was
> the THIRD wrong one on this quantity and a FOURTH followed. Its NUMBERS are right (two independent
> instruments, required to agree, reproduce them exactly), but the CONCLUSION drawn from them is not.
> Ignition is governed by the bound assembly's SIZE, not by the relaunch: pinning the assembly small in a
> SECOND life makes it ignite three times -- more than a first life ever does -- and pinning it large in a
> FIRST life removes ignition entirely. A first life ignites once while k is still ramping and never again
> in 236 wake ticks. **This is not a resume defect**, and ignition is struck from L4's target list. What
> survives of section 8.1 -- 17 of 52 moving keys returning identical -- is unaffected.

### 8.3 The disposition is REVERSED: L4's persistence half RE-OPENS

§4 refused the row on the strength of §2 and §3. Both are retracted, so the refusal goes with them. The
corrected measurement **supports** the row: a relaunched brain loses two thirds of its moving state and does
not ignite at all.

The row is re-opened rather than built here, because the build is its own arc and this one has already spent
its budget correcting itself. What it inherits that it did not have before:

1. **A committed instrument** — the `--resume-probe` arm, with its universe stated and its ignition counter
   carrying its own positive control.
2. **A named target** — ignition, not the five carriers the row originally guessed at. `_access_state` is the
   first thing to follow, and the question is what a first life accumulates before tick ~26 that a second
   life does not.
3. **A red that already fires** — arm A ignites, arm B does not, on 4 of 5 seeds. Any fix has to close that.

### 8.4 The append half stands, with its claim corrected

The panel measured that the wire covered **one** of the five baselined tags, not five: the observer sat at
three sites covering `0x01` (unbaselined) and `0x07`, while a live brain wrote `0x04` and `0x0A` that the
budget never saw, and constructed floods on both read CALM. The commit's "bounds the five tags that carry a
recorded p95" was false by four of five.

Fixed, and re-measured on a live 60-tick brain: **written `{0x04, 0x07}`, observed `{0x04, 0x07}`, nothing
unobserved.** `_observe_append` now sits at every site in `home` that builds a tag — `0x01`, `0x03`, `0x04`,
`0x05`, `0x07`, `0x08`. **`0x0A` is still NOT observed and is now named rather than implied:** coalitions
frame onto `BinLog` through the coalition journaller and never enter `home`, and `0x0A` is 80.21 % of the
production lake. `0x0B` likewise writes through its own store.

Three more, all real:

- **The observer was installed and never removed.** A second brain in one process silently deafened the
  first — measured: brain A read 0 floods for a flood on its own lake, under a PROVENANCE row saying 0 is
  expected — and a discarded brain's budget kept accumulating bytes nothing would close. `Brain.close()` now
  restores what it replaced, which is why `set_append_observer` returns it.
- **`AppendBudget` was not thread-safe on this project's free-threading interpreter.** 8 threads × 20,000
  appends counted 36,431 — **78 % lost** — and `end_tick` could raise dict-changed-size out of the caller's
  tick. It now holds a lock and swaps the dict out under it; the tick's close is guarded as well.
- **`extend` reported payloads that dedup dropped**, counting traffic that never reached the disk. It now
  reports only what was written, attributing bytes to the largest written payloads — an over-attribution
  bounded by the size difference between duplicates, disclosed rather than exact.

### 8.5 An open finding the panel measured and this row does not fix

**An ordinary brain tick reaches 79–83 % of the `0x07` flood ceiling and is still climbing at 1250 ticks**,
carried by whole-frame state names. The `0x07` p95 was recorded on a 120-tick fresh brain, so the only live
bound this project has is on a path to firing on a matured brain in normal operation. Re-baselining from a
matured brain's nightly persist frame is the fix, and it is not this row's: it needs a measurement run on the
pinned host, which is what `tools/lake_append_baseline.json` records. **Disclosed here so the first false
flood is not a surprise.**

### 8.6 What was refuted

Two findings did not survive. A claim that a disarmed budget is indistinguishable from a calm one was refuted
by running it: the shipped test suite fails loudly if the baseline file is missing or the budget fails to
construct. A claim that "0x01 can never trip" is pinned only against a fabricated dict was refuted because
`immortal_retention.budget_coverage()` reads the shipped file and prints the absent-key set on every run, and
because the sentence sits under a heading that says what the row does NOT do.

### 8.7 The honest caveat

**This fold is itself unreviewed code.** It was written in response to the panel and has not been through a
panel of its own. It has: the re-measured coverage on a live brain, the committed resume probe with its
positive control, the module self-checks, the tests, and the arc's battery. That is battery-verified, not
panel-reviewed.

---

## 9. The disclosed risk arrived within minutes, on the live daemon

§8.5 disclosed that an ordinary tick reaches 79–83 % of the `0x07` ceiling and is still climbing, so the only
live bound this project has was on a path to firing in normal operation. It did not take a matured brain.

The daemon relaunched on this arc's push and, at `nights_elapsed = 8`, reports:

```
"append_floods": 7
"append_flood_last": "FLOOD: tag 0x07 (IMMORTAL) appended 160470 B this tick > 8.0x its recorded p95 (12381 B)"
```

**Seven floods in eight nights, all on `0x07`, at 160 KB against a 12 KB p95.** Two things follow, and they
point in opposite directions:

1. **The wire works.** It was dormant for the life of the instrument and it fired on the real brain within
   minutes of being connected. The constructed red predicted a live one.
2. **The baseline is wrong for the brain that exists now.** 12,381 B was recorded on a 120-tick fresh brain
   in August; the live brain writes an order of magnitude more per persisting tick, carried by whole-frame
   state names. These seven are FALSE floods — the brain is not flooding, the ceiling is stale.

`FLOOD_FACTOR = 8` is not the problem and moving it would be tuning a constant to silence a measurement.
What is owed is a re-recorded `0x07` p95 from a MATURED brain's nightly persist frame, on the pinned host,
by the same instrument that recorded the original — which is a measurement run, not an edit, and is why it
is not done here. Until then `append_floods` on the live cockpit reads a stale ceiling and not a lake
problem, and this section is what a reader of that number should find.

Recorded rather than smoothed: the first live reading of a newly-charged instrument is evidence about the
instrument, and this one says the ceiling needs re-measuring before the count means anything.
