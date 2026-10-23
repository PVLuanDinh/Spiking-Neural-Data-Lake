# Insula P8 question: does its charge reach the engine?

**Status: FROZEN 2026-10-23 on the owner's approval in chat ("approve, run the census"), unchanged from the draft
reviewed (1e2961c8) except this line. This commit precedes every arm; nothing is measured yet.** Origin: the owner's pick "Open a P8 question"
(`insula-level-arms-production.md` section 3): the insula is not P3 shape 5, and whether its winner-only charge is
load-bearing for the engine's function is open.

## 0. What orientation found (static reading at 2ff11373; Stage 0 confirms or refutes it at runtime)

1. **One entry point.** The insula's output, `_feeling_salience`, has one reader: the ignition-threshold bias
   (`engine/brain.py:2877`). Its measured effect on the production tier is the workspace WINNER only; the ignition
   sequence is identical across every arm (`insula-level-arms-production.md`, "What every count is").
2. **Where winner identity goes.** (a) The action re-route reads `max(broadcast_to("action").values())`, the bias
   MAGNITUDE, not which coalition won (`brain.py:2198-2199`); `tools/charge_probe.py` measured that re-route changing
   34 of 40 internal reads and 0 of 40 act/hold decisions or trajectories. (b) `report_use("action", content)`, an
   association inside the workspace. (c) The insula's own drain of the bus (T039h), which is circular.
3. **The engine-function reads that exist do not tick the brain.** G2's taught read is `route_text` on the learner,
   pure inference (`readout/interaction_learning.py:698`), and its teaching is `learn_turn`. G9's posed actor reads
   through the Part D reader on the same learner (`brain.py:5363-5398`). The insula lives in `tick()`. A G2 or G9
   score with insula arms therefore sees the insula only if the tick path changes the learner, or the lake the
   learner reads. That is a reach question, and it comes first.
4. **Not re-proposed:** the 2026-09-30 reservoir relational probe (its competence gate failed three times; the
   re-proposal guard covers it). The engine's held-out readout of record is G-R9, counted by G9, whose deciding
   status is REFUSED today (not built).
5. **Owner-drawn check.** The insula is not on the owner's 2026-09-25 diagram (`engine-structure-2026-09-25.md`
   section 1: Brain, new neurons, reservoir, readout neurons, the lake). It is an invented region (T039f, T039h), so
   P8 item 5, deletion on a null, applies, not P4d's revert-and-do-not-claim rule for drawn parts.

So Stage 0, a reach census runnable now, decides whether the question has a functional stage at all.

## 1. Stage 0: reach census (pre-registered by this file)

**Question.** Does any change the insula makes reach an engine-function consumer?

**Harness.** The insula-lvl-1022 harness (`arm.py`, the semantics of `_run` in `tests/test_interoceptive_ignition.py`
with the seed as a parameter): `make_brain(seed, persist=False)` under `throwaway_home()`, the production tier pinned
by absolute path to the four `SNN_*` variables, the tier gate as before (a run that is not
("eprop-snn", True, device) is REFUSED), one arm per subprocess, OMP_NUM_THREADS=4.

**Length.** T = 1000 ticks, five times the level arms. The level arms' injected effects ended by tick 61 while removal
persisted to tick 199 on one seed; 1000 ticks tells a startup transient from a steady state.

**Seeds.** 0-4, the level-arm seeds (this measures reach on known cases, not a fresh-sample estimate).

**Arms.** live, live2, neutered, const, shuf, defined as in the level arms (const and shuf from live's consulted trace
over the 1000 ticks), plus **forced-winner** (diagnostic): on every ignited tick the reentrant winner is replaced by a
fixed coalition other than live's, an upper bound on what ANY winner change can reach.

**Deciding columns** (engine-function consumers, compared with live):
- **E1 decision:** `arm_stopped` and `map_position` per tick (`tools/charge_probe.py`'s decision read).
- **E2 lake:** the throwaway lake's records as a (kind, sha256 of payload) list, at ticks 100, 200, ..., 1000.
- **E3 growth:** grown-unit birth records, count and ids, per tick.
- **E4 learner:** the sha256 of the learner's (`_ixn`) state at ticks 100, ..., 1000, and its `route_text` answers at
  tick 1000 on a fixed probe set (G2 family (i), delayed-cue, validation split, seed 0); read only, never taught.

Sleep ENTRY is not a column: `clock.consolidation_gate()` schedules it, which the insula cannot reach. What a sleep
writes or changes lands in E2 and E4.

**Diagnostics** (reported, never deciding): winner, ignited, live_thr, `_reroute_bias`, the action drive vector, and
the workspace's report_use association.

**Controls.**
1. **Repeat pair.** live2 equals live on every column on every seed; otherwise the census is INVALID.
2. **Positive control per deciding column** (seed 0, a planted change at that column's own input on tick 500): E1
   forces one action's drive above the rest; E2 writes one extra lake record; E3 forces one birth through the
   grower's own API; E4 runs one `learn_turn` on a probe. Each must register in its column. A column whose control
   does not register is UNPLUGGED, and any unplugged deciding column makes the census INVALID.
3. **Tier gate red**, as in the level arms.

**Readings** (applied literally to neutered, const and shuf; "differs" means on any tick or checkpoint, since
determinism makes any difference causal):
- **INVALID:** control 1 or 2 fails. Nothing is read.
- **REACH-NULL:** no deciding column differs from live for neutered, const or shuf on any of the 5 seeds. The
  insula's charge ends inside the workspace. The disposition is a causal claim, and control 2 is its positive
  control: P8 deletion on a null is PROPOSED to the owner, with the section 3 disclosures. Stage 1 is not run.
- **REACHES:** some deciding column differs for at least one of neutered, const or shuf on at least one seed. Each
  reached column is reported with its seeds (k of 5) and its first and last differing tick (transient or steady
  state). Stage 1 is owed against the reached columns.
- **Forced-winner**, reported whatever the reading: if it reaches no deciding column, no winner change can (stated);
  if it reaches and the insula arms do not, the insula's winner changes are too small or too brief to reach (stated).

**Prior, written before data so the result can contradict it:** static reading predicts REACH-NULL on E1-E3, and is
unsure on E4, where sleep consolidation of the learner's memory is the one plausible path.

**Cost.** About 15 s per 200-tick arm (measured 2026-10-22); 30 census arms and 4 control arms at 1000 ticks is about
one hour on one lane. A docs-only arc plus a row folder: no `src/`, `tests/` or `tools/` change. If a column needs a
code change to be read, the lane stops and reports.

## 2. Stage 1: functional (named here, designed only after Stage 0)

Stage 1 runs only on REACHES. Its pre-registration is written after Stage 0, against the reached columns, and is
committed before any Stage 1 arm.
- **E4 or E2 reached:** the G2 contract (`taught_problems.Problem`, the solved predicate on held-out splits) with the
  insula arms, the tick path run between teaching episodes so it can act. Competence gate first: a known path-breaker
  (learning OFF) must lose at the bar before any insula reading counts.
- **E1 or E3 only:** the engine's endpoint is G9's posed-problem solve. The insula arms join G-R9 (ii)'s per-part
  attribution, under G9's own competence controls, and wait for G9's deciding campaign (REFUSED today).
- **Null at Stage 1:** deletion proposed (P8 item 5). **Load-bearing:** claimed only for the endpoint and tier measured.

## 3. Disclosures (P8 item 4)

- Deleting the insula would move counts it carries: FR-015 region charge 8 to 7 of 16 (T039f), and SC-001
  faculty-to-faculty reads 1 to 0 (T039h; the insula is the second specialist to drain the bus). Whether those counts
  were functional is exactly this question; the deletion is the owner's call.
- REACH-NULL covers these columns over 1000 ticks of a fresh brain on the default schedule, on cuda. Stage 0 drives no
  chat and no teaching, so a path that opens only under chat or posed problems would be missed: named, not measured.
- E4's probe set is read, never taught. A learner that answers identically after different tick histories is
  reach-null on E4 for that probe set only; the state digest is the wider check.
- Unmeasured: cpu; seeds beyond 4; runs longer than 1000 ticks; chat-driven runs.
