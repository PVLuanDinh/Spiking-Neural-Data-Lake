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

## 4. Results (2026-10-23)

Sections 0-3 above are unchanged (frozen at 37384760). Lane `insula-reach`, worktree `lanes/insula-reach-1023`, branch
`claude/insula-reach-1023` off 37384760. Harness, raw JSON, logs and the full report are in the row folder
`D:\AI-project\lanes\insula-reach-1023-row` (`arm.py`, `driver.py`, `results/<seed>-<arm>.json`, `table.json`, `REPORT.md`).
Every arm: T = 1000 ticks, one arm per subprocess, OMP_NUM_THREADS=4, `make_brain(seed, persist=False)` under
`throwaway_home()`, tier ("eprop-snn", True, "cuda") on all 35 arms [34 measured arms plus the tier-red run: section 5, item 12] (all arms of a seed share the device).

**Reading: REACHES** (applied literally). Deciding columns E1 (`map_position`) and E2 (lake records) differ from live for
neutered, const and shuf on at least one seed. E3 and E4 differ for no arm on any seed. Stage 1 is owed against E1 and E2.
The static prior (REACH-NULL on E1-E3) is contradicted on E1 and E2 and holds on E3; the prior's unsure E4 came out null [corrected: not eligible, section 5, item 6].

### Controls (run before the census)
1. **Repeat pair: PASS.** live2 equals live on E1, E2, E3, E4 and every diagnostic, on all 5 seeds.
2. **Positive control per deciding column, seed 0, plant on tick 500, each against live:** all four register.
   E1: the least-preferred action's score in `Autonomous._route_signals` forced +100 (the column's own input); 370 ticks of
   `map_position` differ, first 502. E2: one extra record through `KnowledgeLog.append`; 6 of 10 checkpoints differ (500..1000).
   E3: one birth through `Ledger.record_units` with `make_admission` labels; `e3` count differs on 501 ticks (and E2 on 6
   checkpoints, as expected, since the birth row is a lake record). E4: one `learn_turn(..., feedback=...)`; the learner digest
   differs at 6 checkpoints and the tick-1000 answers differ. **No column is unplugged.**
   *Disclosed:* my first E1 plant fired at tick 500 only, and the arm steps on about 493 of 1000 ticks, so tick 500 had no arm
   step and the plant never fired (0 differences). It was a harness defect, not an unplugged column. The plant now fires on the
   first arm decision at or after tick 500 (it fired on tick 502). Only the ctl-e1 path of `arm.py` changed; live and live2 ran
   on the earlier file and contain no plant code path [inaccurate: section 5, item 9].
3. **Tier gate red:** `SNN_EPROP_CKPT` set to a nonexistent absolute path gives tier ("stdlib", True, "cuda"), REFUSED, rc 3
   (`results/tier-gate-red.log`).

### Table: which deciding columns differ from live (k of 5 seeds; first and last differing tick or checkpoint)
| arm | column | k of 5 | per seed |
|---|---|---|---|
| neutered | E1 | 2 of 5 [1, 2] | seed 1: 527 ticks, first 474, last 1000; seed 2: 944 ticks, first 57, last 1000 |
| neutered | E2 | 2 of 5 [1, 2] | seed 1: 6 reads, first 500, last 1000; seed 2: 10 reads, first 100, last 1000 |
| neutered | E3 | 0 of 5  | - |
| neutered | E4 | 0 of 5  | - |
| const | E1 | 2 of 5 [3, 4] | seed 3: 378 ticks, first 623, last 1000; seed 4: 485 ticks, first 516, last 1000 |
| const | E2 | 2 of 5 [3, 4] | seed 3: 4 reads, first 700, last 1000; seed 4: 5 reads, first 600, last 1000 |
| const | E3 | 0 of 5  | - |
| const | E4 | 0 of 5  | - |
| shuf | E1 | 1 of 5 [4] | seed 4: 485 ticks, first 516, last 1000 |
| shuf | E2 | 1 of 5 [4] | seed 4: 5 reads, first 600, last 1000 |
| shuf | E3 | 0 of 5  | - |
| shuf | E4 | 0 of 5  | - |
| forced | E1 | 3 of 5 [1, 3, 4] | seed 1: 305 ticks, first 696, last 1000; seed 3: 379 ticks, first 622, last 1000; seed 4: 485 ticks, first 516, last 1000 |
| forced | E2 | 3 of 5 [1, 3, 4] | seed 1: 4 reads, first 700, last 1000; seed 3: 4 reads, first 700, last 1000; seed 4: 5 reads, first 600, last 1000 |
| forced | E3 | 0 of 5  | - |
| forced | E4 | 0 of 5  | - |

E1 means `map_position`: `arm_stopped` differs on 0 ticks in every arm. Every E1 difference that starts persists to tick 1000 (last =
1000 in all 8 cells), i.e. a divergence that does not return, not a startup transient [corrected: section 5, item 5]. E2 differences are all `0a` (fact-type) [mislabelled: section 5, item 2]
records. In all 8 cells the first differing E2 checkpoint is at or after the first differing E1 tick, so E2 never differs on a seed
and arm where E1 does not; the data do not show the path from one to the other [corrected: section 5, item 3]. E4 "reads" include the tick-1000 `route_text`
answers (a 32-presentation read of the G2 family (i) validation probe texts for `i.S0C0`, seed 0, read only).

By seed, the union of neutered, const and shuf reaches E1 and E2 on seeds 1, 2, 3, 4 (4 of 5) and neither on seed 0.

### Forced-winner (diagnostic, reported whatever the reading)
Session call, as the brief fixed it: on every ignition the reentrant winner is replaced by ONE fixed coalition, the first
coalition name sorted by `str()` among the names submitted to the workspace in live that is not live's most frequent winner.
The pool in live on every seed was [0..9, novel, self]; live's most frequent winner was `self` on every seed; the forced name was
the digit coalition `0` on every seed. Implementation: the workspace's `_simulate` return is patched in the subprocess so
the fixed coalition takes the sustained-firing max on every tick that ignited (490-498 ticks per seed); no `src/` edit.
Forced-winner reaches E1 and E2 on seeds 1, 3, 4 and neither on seeds 0 and 2; E3 and E4 on none.
Stated per the rule: it reaches E1 and E2, so a winner change CAN reach the engine's decision and the lake. It does NOT bound the
insula: on seed 2 neutered reaches E1 at tick 57 and forced reaches nothing, and on seed 3 const reaches where neutered does not.
Forced keeps the ignition threshold path unchanged [false: section 5, item 1] while neutering changes it (`live_thr` differs on 995+ of 1000 ticks in every
insula arm), so the insula arms are not a subset of what a winner change does.

### What surprised me
- The insula-removed arm reaches E1 within 57 ticks on seed 2 while the winner-only forced arm does not reach on that seed at all.
- E3 and E4 are structurally constant, not merely unmoved by the insula: a default 1000-tick brain writes no birth row in any
  arm (max births 0) and its learner digest is identical at all ten checkpoints in every arm, including live. Their positive
  controls register, so the null is real for this schedule, but they could not have registered any insula effect on it.
- The reentrant workspace and the insula observe on about 493 of 1000 ticks, so half the ticks carry no insula input.
- The tick-1000 `route_text` answers differ across seeds (the learner is seed-specific) but not across arms of one seed.

### Scope, unmeasured, and what REACHES does not say
Differences are exact (determinism makes any difference causal under the pre-registered rule). A difference in `map_position` or in
fact-record bytes is reach, not function: this census does not say the insula's charge helps or harms the engine. cuda only; no chat,
no teaching; seeds 0-4; 1000 ticks (section 3 disclosures stand). The E1 read is the arm's `map_position` from `state()`; the
decision-argmax inside the arm is not recorded separately. The E3 read is the immortal `unit`/`born` rows in the throwaway lake
(the G9 `_g9_birth_records` read), so growth that leaves no such row is not seen. The raw JSON is in the row folder for the lead's
own re-derivation. Stage 1 is owed against E1 and E2 and is not designed here.

## 5. Closing review and corrections (2026-10-23)

Opus closing review (read-only and independent; it rebuilt every cell of the section 4 table from the raw JSON, and every cell
matched): **APPROVE-WITH-FOLDS**. The pre-registered reading **REACHES stands**: section 1's rules, applied literally, select it.
The lead re-derived every fact below from the raw JSON and the source with its own code. This section is the fold. It is not
re-reviewed; it is checked only by the lead's re-derivation and the docs-only battery. Section 4 stays as the lane wrote it,
with bracketed pointers to the items here.

**Corrections and additions (MAJOR)**
1. Forced-winner DOES change the threshold path: its `live_thr` differs from live on 995-999 ticks per seed, because the
   insula's top-down term reads the winner's broadcast (`faculties/interoceptive_integration.py`). The conclusion that forced
   does not bound the insula still stands on the seed data alone (seed 2: neutered reaches and forced does not; seed 3: const
   reaches and neutered does not).
2. The `0a` records are `_T_COALITION`, the workspace's coalition journal (`substrate/home.py`), not facts (`_T_FACT` is
   `0x01`). They are the WHOLE lake in every run (3204 of 3204 live records at tick 1000 over the 5 seeds), because
   `make_brain(seed, persist=False)` gates off the episode write, the nightly facts, skill formation and the growth flush
   (`engine/brain.py` lines 2395, 3610, 3693, 3708 at 2ff11373). On this harness E2 is the coalition journal and nothing else.
   Section 3 does not name `persist=False`; it is named here (P8 item 4).
3. The E1 to E2 path IS shown. ctl-e1 plants only on the arm (winner and reroute identical to live on all 1000 ticks), and E2
   still differs at checkpoints 700-1000. In every census cell the first differing coalition record is appended 1-26 ticks
   after the first E1 tick (seed 1 neutered 474 to 480; seed 2 neutered 57 to 58; seed 3 const 623 to 642; seed 4
   const/shuf 516 to 542; forced likewise). E2 is the trajectory's echo: one reach, not two.
4. Removal is not winner-only: it changes whether the reentrant workspace ignites. Under neutered, `rw_ignited` differs from
   live on 114 / 333 / 517 / 1 / 0 ticks (seeds 0-4), first at ticks 62 / 49 / 29 / 649, and the winner is None on
   117 / 333 / 517 / 1 / 0 ticks against 3 / 0 / 0 / 0 / 0 in live. const and shuf never change `rw_ignited`. The
   pre-registered `ignited` diagnostic (`_access_state["ignited"]`) is a different, near-constant bit. So removal darkens the
   workspace on seeds 1-2 (a level effect), while the injected arms change only which coalition wins. The same holds in the
   level arms (addendum, `insula-level-arms-production.md` section 4).
5. Reach is a fork, not a graded effect. const and shuf differ from live only early (last winner difference at tick 62 or
   earlier, last reroute difference at 129 or earlier, every seed): the start-up ramp, where live salience is 0.0 on its first
   consulted ticks while const and shuf inject the mean from tick 1. A residue in the drive vector survives, and on some seeds
   the trajectory forks hundreds of ticks later (516, 623). Which seeds fork is not ordered by perturbation size: on seed 0
   nothing forks, forced and neutered included; on seed 4 const, shuf and forced give byte-identical positions while differing
   from each other in winner. Exact inequality cannot tell a transient from a steady state: on seed 4 the const and shuf
   positions come back within 1e-6 of live on 26 of the last 100 ticks. Removal's own effect is sustained (winner differences
   to tick 970-1000 on seeds 0-2). The k-of-5 counts measure how often a seed hits a fork, not how much charge the insula
   carries.
6. E3 and E4 were NOT ELIGIBLE, not null. No run writes a birth row (max births 0), and the learner digest is constant at all
   ten checkpoints in every non-control run, live included. The learner updates only on chat paths, Stage 0 drives no chat,
   and `persist=False` closes the growth flush. The positive controls register, so the columns work; this schedule cannot move
   them.
7. `arm_stopped` is False on every tick of every run, live included: the act/hold half of E1 is a constant here. E1 is
   `map_position` alone.

**Minor**
8. The ctl-e1 re-run is a deviation from section 1 (a control that does not register makes the census INVALID). The first
   ctl-e1 plant missed (tick 500 is not an arm step), `arm.py` was edited, and ctl-e1 re-ran with the plant at tick 502. The
   lane disclosed it; it is named here as a deviation for the owner to ratify. The first ctl-e1 JSON was overwritten.
9. live and live2 ran on the earlier `arm.py` and the census arms on the edited one; live's JSON shows it passed through the
   `_route` wrapper. Evidence that the edit is behaviour-neutral: seed-4 neutered (edited file) matches live (earlier file) on
   winner, reroute, drive, position and E2 on all 1000 ticks.
10. Neutered removes two things, the threshold term and the insula's bus drain (`report_use("interoception", ...)`); const
    and shuf keep the drain. They are not separated here (P8 item 3).
11. Forced injects coalition `0` with no bottom-up drive behind it (no re-entry recharge, zero drive, a Hebbian update with an
    empty source set): a structural perturbation, not the "upper bound on any winner change" section 1 describes.
12. Nits: 34 measured arms plus the tier-red run (35 runs); the tick-1000 `route_text` answers are near-constant per seed (1-2
    distinct labels over 32 probes); ctl-e1 also moved E2 (item 3).

**What this changes for Stage 1** (the owner decides; nothing here is a new rule)
- Section 2's literal routing ("E4 or E2 reached: the G2 contract") fires on E2, which item 3 shows is the trajectory's echo,
  while the G2 learner is the column this schedule cannot move (item 6). The reach that exists is E1, the arm's position,
  which section 2 routes to G9's attribution.
- A functional stage must beat a size-matched sham perturbation on a non-insula path across many seeds (item 5), keep live's
  start-up ramp in the injected arms, and run with `persist=True` inside `throwaway_home()` so the episode, fact and growth
  paths exist (items 2 and 6).
- Removal's sustained effect is a darker workspace on some seeds (item 4); attribution between the threshold term and the bus
  drain is owed (item 10).
