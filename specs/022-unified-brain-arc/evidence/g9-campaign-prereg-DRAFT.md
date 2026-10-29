# G9 / T117 -- the engine, end to end: campaign pre-registration (DRAFT pre-registration)

> **Status: DRAFT. Nothing in this file is frozen, and nothing has been run.**
> - Every number marked OWED below waits on a run (the pilot) or on an owner answer. No seed block is claimed.
> - Written statically by lane g9-prereg (Opus, docs only) on branch `claude/g9-prereg-1029`, parent `d8b3ef73`
>   (= `claude/g9-gaps-1029`). Policy and code anchors are read at `c4c10b28`; the arc's code equals that commit's
>   except `infra/_mathutils.rsa_drift` (added by `f0f50385`, test first in `fd987f3f`).
> - No test, brain, battery or gate was run (campaign rules 1-3, `lanes/engine-1024/LANE-RULES-CAMPAIGN-1029.md`).
> - It had no panel and no policy-enforcer read when written.
> - **Task:** T117 [G9] (`tasks.md:257`). The row's wording binds: this draft claims nothing T117 does not name.
> - **Anchors.** PLAN = `specs/022-unified-brain-arc/PLAN-FINAL-2026-09-02.md`; TASKS = `specs/022-unified-brain-arc/tasks.md`;
>   POL = `docs/POLICIES.md` at `c4c10b28`; DI = `evidence/g9-campaign-design-input-DRAFT.md` at `d8b3ef73`
>   (its item 1-2 text `:19-86`, its g9-gaps addendum `:129-229`); FOLD1 = DI's "Panel fold 1" addendum, commit
>   `097944a5` on `claude/hw4-g9-1029` (board #6316). **FOLD1 is NOT an ancestor of this branch** (`git merge-base
>   --is-ancestor 097944a5 HEAD` = false), so its text was read with `git diff 7c1eebe2 097944a5 -- <DI>`; it is cited
>   as "FOLD1 item n". Where FOLD1 and the g9-gaps addendum disagree, this draft follows FOLD1 and says so (section 5).

## 0. Raising-ambition record (self-apply, owner 2026-10-25)

- **Axis:** Depth. G9 is the engine's deciding campaign; the move is to make every arm, control and red one a careful
  experimentalist would accept, not to add questions.
- **Ladder.**
  - **R1:** gather PLAN's arms and G-R9's reds into one file with a P1 section that says "OWED".
  - **R2:** R1, plus the owner's four causal-organization arms mapped onto the arms that already serve them, the two
    missing arms (sleep-off, frozen floor) each with a non-method control and a red, endpoints read against the
    ENGINE's function with path-breaking controls, the drift items and the RSA red, and the full sequential design
    written down with every number either derived from a named source or marked OWED by the pilot that will measure it.
  - **R3:** freeze it: derive each seed count from a measured paired SD, run the boundary simulation, and edit the arm
    lists in PLAN:2036-2059 and TASKS:257.
- **PICKED:** R2.
- **The rung above, ruled out:** R3. POL:172-207 (P1 amendment item 1) requires the target to come from a measured SD,
  and none exists: the pilot needs the built G9 engine (G9 runs last, PLAN:2057) and campaign rule 1 forbids any run.
  The lane rules (`LANE-RULES-1029.md`, "Never touch ... any file your brief does not name") forbid editing PLAN and
  TASKS. POL:927 (P8 item 1) also bars claiming a red that has not been counted.
- **Outside judge:** the policy-enforcer's static read against POLICIES.md and the integrator's panel, each re-opening
  the file:line anchors. Later, mechanically: the P1 gate's field check (POL:203-207, when built), the boundary
  simulation (section 4.4) and the pilot. No statement here is a measurement.

## 1. What G9 is, and the engine sentence it serves

PLAN:2036-2059 and TASKS:257, condensed. In scratch brains (throwaway homes, never the live lake), on G2's problems
taught through G0's channel: the brain attempts; on failure G3's loop grows column-sized batches on the map and FEEDS
them into the reservoir; the brain transports weights (both forms as arms); G5's readout neurons answer and feed back
(the return path); the loop repeats until the solved predicate holds or the problem pauses at its budget; sleep prunes
and saves (G6); a FRESH PROCESS rebuilds the reservoir from lake records and re-solves with no regrowth and no teaching.

**Engine sentence (owner 2026-09-30, memory `feedback-evaluate-against-engine-function.md`):** fed-in neurons pass
through the reservoir to acquire relational properties that connect them to the existing brain. So the PRIMARY
endpoint is functional: a held-out relational readout, against controls that BREAK the path. Structural statistics
(in-degree, rates, drift) are secondary readings, never pass/fail bars unless tied to that endpoint.

G9 is also the owner's condition on transport ("as long as the engine can work", POL:595-600): transport lapses only
on its own measured failure, the whole engine failing G-R9 (i) or G-R9 (ii)'s transport neuter changing nothing.

## 2. Arms

Every arm runs on the same seeds, the same problems and taught inputs, the same number of Brain ticks and teaching
exchanges (equal compute), and the same probe set. Each mechanism is credited only after it beats its simpler
equal-capacity AND equal-compute arm (PLAN:1857). Each arm names its control: one that is not the method (POL:930).

### 2.1 PLAN's arms (PLAN:2043-2053; TASKS:257)

| Id | Arm | What it isolates | Control (not the method) |
|---|---|---|---|
| A0 | the engine | -- | -- |
| A1 | reservoir removed (new neurons read directly) | the reservoir | A2 (the reservoir kept, the feed cut) so A1 is not the only cut of connection 2 |
| A2 | feed neutered: reservoir driven by norm-matched random units (G3's feed control) | the feed (connection 2) | norm matching: the random units' drive norm equals the fed units' |
| A3c | transport neutered, copy form: a norm-matched random matrix | transport (copy) | norm matching |
| A3a | transport neutered, activity form: the brain's drive time-shuffled | transport (activity) | the shuffle keeps the drive's marginal statistics |
| A4 | return path neutered | the return path | a time-shuffled return (T113's P6 test, TASKS:253) keeps its marginal statistics |
| A5 | growth OFF at the engine's final size (presized) | growth | SC-012b's equal-count silencing (`spec.md:202`): silencing grown units must drop the score; silencing an equal number of presized units must not |
| A6 | random growth, matched budget | targeted growth | budget matched in units, synapses and growth events |
| A7 | G7's T1 at engine level: column batches vs single neurons | the organizing unit | single-neuron growth at the same unit budget |
| A8 | G7's T4 task-score half: rule-grown wiring vs imposed columns vs random attachment | the wiring rule | imposed columns and random attachment at equal unit and synapse counts |

P2 keeps one transport form after G9 (PLAN:2055): A3c and A3a decide which.

### 2.2 The owner's four causal-organization arms (mechanism evidence only)

Owner, 2026-10-13 (memory `project-snn-owner-consciousness-mechanism-tests-1013.md`, verbatim core): "Compare trained
and untrained populations at the same size. Compare targeted growth with random growth at matched budgets. Disrupt
specific feedback or information-sharing paths and test which abilities selectively disappear. Check whether those
effects survive restoration from the lake." The owner scoped them as mechanism tests that "would still require a
separate argument connecting those mechanisms to subjective experience". So every O-reading below is reported as
**mechanism evidence only**: a dissociation, never a marker, and never a verdict about experience (POL:1014;
`docs/CONSCIOUSNESS_METHODS.md:101`).

| Id | Owner's arm | Arm(s) it runs on | Control (not the method) |
|---|---|---|---|
| O1 | trained vs untrained at the same size | NEW arm A9: the engine's final topology and unit count, weights at their seeded initial values (no learning applied) | a weight-shuffled twin of A0 (A0's trained weights permuted within each projection): it keeps the weight statistics and destroys the learned organization, so "trained beats untrained" is not only "trained weights are larger" |
| O2 | targeted vs random growth, matched budgets | A0 vs A6 (reuse) | A6's budget match (units, synapses, growth events) |
| O3 | disrupt specific paths; which abilities selectively disappear | the neuter arms A1, A2, A3c, A3a, A4, read as a PATH x ABILITY matrix (rows: paths; columns: each G2 problem class's held-out score). Candidate information-sharing paths add the two workspaces (`faculties/global_workspace.py`, `faculties/reentrant_workspace.py`; P2 standing adjudication, POL:252-256) as two more rows, if both are on the engine's path at the built G9 (OWED, section 6) | a SHAM neuter per path: the same hook installed, passing the signal through unchanged. A path effect is read only against its sham. "Selective" means a double dissociation: path p costs ability x and not y, and some path q costs y and not x |
| O4 | do the effects survive restoration from the lake | each O1-O3 contrast re-read after the fresh-process rebuild (G-R9 iv, vi) | the uninterrupted twin (G-R9's control list, PLAN:2331). Read only after #6231's persistence build is GREEN (FOLD1 item 2), or the gap reads as a P4b red, not as an effect that did not survive |

### 2.3 The two arms the design input found missing (DI :129-212, designed by g9-gaps)

| Id | Arm | How, at the tree | Control (not the method) | Red (refuses the arm) |
|---|---|---|---|---|
| A10 | sleep OFF: no sleep bout starts; growth ON, everything else as A0 | `gate_from(brain, lambda: False)` (`instruments/sleep_pressure.py:305-320`), REUSED; not `Brain.neuter_sleep` (that is the structural part only, `engine/brain.py:3623-3638`) | the TRANSPARENCY twin: the same override replaying the clock's own scheduled gate sequence (sleep stays ON; the harness is the only change) | `_enter_sleep` count over the run: > 0 on A10 exits 3; the same count must be > 0 on A0 and the transparency twin (positive control) |
| A11 | frozen-engine FLOOR: the same events (growth slot, sleep slot, restart), nothing written | COMPOSED from existing switches (DI :187-197); no whole-engine freeze exists | a PLANTED-CHANGE twin: one known weight-row scaling between readings; each drift metric must move off the floor in the right direction | `state_digest` (`instruments/sleep_plasticity_ab.py:221`) of weights + rng equal before and after; a freeze with one write path left open moves it and exits 3 |

Two carried limits. (1) Growth is sleep-gated on the incumbent (`engine/brain.py:3352-3354`), so A10 also stops that
growth; a sleep-only claim is read against A5 as well, and is not made if G3's growth is also sleep-gated (unknown
until G3 is built). (2) If the lake records nothing outside a sleep bout, A10's restart leg is NOT READABLE, never
drift 0. If the panel does not adopt A10: "no claim about sleep's share of drift is made; drift after a sleep bout is
read, and not compared with a no-sleep twin" (FOLD1 item 6).

## 3. Endpoints

### 3.1 Primary: the engine works (G-R9, PLAN:2331), read on the engine's function

- **E1 (solve, relational, held out).** Per problem: the solved predicate on the normalized HELD-OUT score of G5's
  brain-tuned readout (P4d, POL:573), on items the loop never taught. Statistic per contrast: the per-seed PAIRED
  difference (A0 minus arm) of the held-out score, one-sided.
- **Contrasts and directions** (registered now; the smallest effect of interest is OWED, section 6 item 4):
  - G-R9 (i): A0 > A5 and A0 > A6 on a problem A5 fails (prospective: the problem is registered before the run).
  - G-R9 (ii), each attributed separately: A0 > A1, A0 > A2, A0 > A3c, A0 > A3a, A0 > A4 (the path-breaking controls).
  - G-R9 (vii): A8 rule-grown >= imposed columns (a red if below, under P4d's null rule). A7: column batches vs single
    neurons, reported beside G-R3's T1 red (PLAN:2331).
- **Reds that are counts, not contrasts:** G-R9 (iii) the loop reports solved on a constructed unsolvable problem, or
  does not pause at its budget; (iv) a fresh process does not re-solve the saved problem without regrowth or teaching;
  (v) an earlier solved problem regresses past its bound; (vi) the reservoir rebuilt from lake records does not
  reproduce the pre-restart answers. Each is constructed and counted RED before any campaign seed (section 3.4).

### 3.2 Secondary: the owner's four (mechanism evidence only)

O1: A0 > A9 on E1, and A0 > its weight-shuffled twin. O2 = G-R9 (i)'s A0 vs A6 contrast (no second test, one
endpoint). O3: the path x ability matrix of E1 differences, each cell against its sham; reported as a matrix with the
double-dissociation cells marked, never collapsed to one number. O4: O1-O3 re-read after the rebuild. None of these
gates T117 beyond what G-R9 already gates.

### 3.3 Secondary: representational drift (HW4 items 1-2; owner 2026-10-25 fold, TASKS:257)

- **D1 (item 1).** A fixed probe set presented before and after (a) a growth step, (b) a sleep bout, (c) a
  fresh-process restart. Reservoir spikes scored by population-vector, ensemble-rate and tuning-curve correlation
  against elapsed events, and RSA-matrix stability (`rsa_drift`). Each reading is taken over the INTERSECTION unit set
  and reports `n_common`, `n_pruned`, `n_born` beside it (FOLD1 item 3). Every reading is reported against A11's floor.
  Drift is not a pass or fail on the engine; the biology (Deitch, Rubin & Ziv 2021) is a reference SHAPE, not a target.
- **Event (c) is not neutral** (FOLD1 item 2): restart drift above the floor files a P4b red first. It is read as drift
  only for state the pre-registration NAMES as deliberately not persisted, shown by a control not to be learned state.
  A bit-identical rebuild reads zero by construction, and the restart reading is then NOT READ.
- **D2 (item 2, attribution).** The paired difference (A0 minus arm) of each D1 metric at the event the arm's part acts
  on: A5 at (a), A10 at (b), A3c/A3a and A4 at all three. A1 has no reservoir spikes: NOT READABLE, never zero drift.
- Direction registered now: an arm that removes a writing part (A5, A10) drifts LESS than A0 at that part's event;
  a neutered path (A3c, A3a, A4) is two-sided (no prior direction), so its alpha is halved per side.

### 3.4 Reds counted before any campaign seed (each must fire; one that does not fire halts the campaign)

| Red | Shape | State at `d8b3ef73` |
|---|---|---|
| RSA reader | `tests/test_rsa_drift.py` (`fd987f3f`): rotated cells read 1.0, identical read 1.0, shuffled probes read low, a raw-response mutant fails the rotated case, a flat probe row and a bad shape REFUSE | written, red/green QUEUED (campaign rule 3), never run |
| pv / rate / tuning reader | FOLD1 item 6's set: identity 1.0; a planted permutation or rate change reads lower; a flat response raises the flag (not `ZeroDivisionError`, not 0.0) | OWED: no shared drift reader exists (DI :101) |
| A10 sleep-off | `_enter_sleep` count, section 2.3 | OWED with the arm's build |
| A11 freeze | `state_digest` equality, section 2.3 | OWED with the arm's build |
| A11 planted change | the floor reader moves by the planted amount | OWED with the arm's build |
| sham neuters (O3) | a sham that changes E1 means the hook, not the path, moved it: exit 3 | OWED with the hooks |
| G-R9 (iii)-(vi) | constructed per PLAN:2331 | OWED; no counted G-R9 red was found by this lane (static read, not a census) |
| A9 untrained | the twin's weight digest equals the seeded init digest | OWED with the arm's build |

## 4. Sample size (P1)

This section carries the P1 amendment's fields (POL:183-198). No count in it is a decision. TASKS:257's "S = 8 under
Part F's rule" is NOT carried as a sample size (FOLD1 item 5; POL:180-182, the E1 incident).

### 4.1 Fields

| Field | Value |
|---|---|
| Measured SD | **OWED: pilot P0.** Per contrast, the SD of the per-seed PAIRED difference (arm minus A0) on shared seeds. A11's floor is NOT the variance source (FOLD1 item 1; a frozen arm can read SD near 0 and size a tiny n) |
| Pilot P0 | the full engine A0 and EVERY contrast arm (A1-A11, A9's shuffled twin, the O3 shams) on the same pilot seeds, a block disjoint from the campaign's; its seed count is a one-shot whose only job is an SD estimate, so it reports the SD with a bootstrap 95% CI and sizes the campaign at the CI's 80% upper SD (the E1d precedent, `e1d-plan-w-pc2.md:213-218`) |
| Claimed effect (delta) | **OWED before the pilot** (FOLD1 item 7): one smallest effect of interest per contrast, on the normalized held-out score for E1 and per metric for D2. Chosen by the pre-registration author from the engine's purpose, never from pilot data |
| Power and test | 0.8; one-sided alpha .05 for directional contrasts; two-sided (.025 per side) for D2's neutered-path contrasts |
| Multiplicity | primary family = G-R9 (i) 2 contrasts + (ii) 5 contrasts = 7, Holm across the family; each contrast is SIZED at alpha .05/7 (the most demanding Holm step), so the smallest-p step is powered. Secondary families (O, D) are sized and reported separately, each with its own Holm correction, and never enter the primary family |
| Fixed-sample target | per contrast n0 = ((z_{1-a'} + z_{0.8}) x SD_upper / delta)^2, rounded up, a' = .05/7; the campaign's n0 is the MAX over the primary contrasts (every arm runs on every seed) |
| Design | sequential, named looks L1-L4 at 25/50/75/100% of the cap; Lan-DeMets O'Brien-Fleming spending per contrast at a' |
| Futility | non-binding; it opens only at a look whose mde, computed from the pre-registered SD at a', is at or below delta (POL:191-194); before such a look the campaign continues |
| Hard cap | ceil(n0 x the design's inflation factor), power at the cap >= .8. For O'Brien-Fleming at 4 looks the factor is small (about 1.02-1.03); the value used is the one the simulation measures, not this note |
| Stop rule | stop only at a named look, only on a written boundary (crossed for every primary contrast, or futility on all of them); seeds added after an unregistered look void the reading |
| Simulated before data | yes, OWED (section 4.4) |

### 4.2 Cost disclosure

One seed of G9 is a whole engine run per arm (A0, A1-A11, A9's twin, the shams: about 17 arm runs per seed before the
O4 rebuild leg). If the cap is beyond what the host can run, the answer is NOT a smaller habitual S: the campaign is
expected to read INCONCLUSIVE at that S, and the owner decides (section 6 item 7). A fixed one-shot count is allowed
only with its power stated before data (POL:195-196).

### 4.3 P8 null-scale axis (POL:944-994)

Each null writes one `NULL-RECORD:` line (units the method acted on, ticks per run, seeds, mde, re-test state).
The axis each contrast scales with, named here (the author names it; a disputed axis is decided by the author with a panel review, POL:955-957):

| Contrast | Axis | Why |
|---|---|---|
| A5, A6, A7, A8, O1, O2 (growth, organizing unit, wiring, training at size) | grown units | capacity mechanisms |
| A1, A2 (reservoir, feed) | grown units fed | the path's capacity is the fed population |
| A3c, A3a (transport), A4 (return) | ticks per run | the effect is learned through the path over time |
| A10 (sleep), D1/D2 drift | sleep bouts (ticks) | a consolidation mechanism scales with bouts |

Owner-drawn parts (the two networks, the reservoir, the return path) are never deleted on a null: implementation
reverted, not claimed, owner decides, NULL-RECORD still written (POL:598). A decision-C part (A7, A8) is deleted only
after the one 10x re-test (POL:598, 955). An underpowered result is INCONCLUSIVE, never a null (POL:971-974).

### 4.4 The simulation check (before any campaign seed)

On the pilot's paired-difference distributions (resampled, centered for the null): a planted null keeps the
false-positive rate of each primary contrast at or below a' and the family's at or below .05; a planted effect of size
delta crosses with power >= .8 by the cap; the probability of every outcome (MET at each look, futility, uncrossed cap,
INCONCLUSIVE) is reported. A design that fails the simulation is redesigned before data, not run.

## 5. Disposition, fixed in advance, and what "engine finished" means

Per primary contrast, and then for the campaign:

| Outcome | Rule | What it means for T117 |
|---|---|---|
| **MET** | every G-R9 (i)-(ii) contrast crosses its boundary in the registered direction, no G-R9 (iii)-(vii) red fires, every section 3.4 red fired before data | T117's condition holds: "the engine can work" (P4d, POL:595) is met at the measured scale, and transport stands. The box may be ticked with the closing record. **Nothing beyond T117's wording is claimed**: no mechanism is credited outside its own contrast, the O-readings are mechanism evidence only, and no claim about experience is made (POL:1014) |
| **NOT MET, powered** | a contrast's boundary is not crossed (or futility taken) at a look whose mde <= delta, or a G-R9 count red fires | T117 closes as a measured failure. Per part: an owner-drawn part is reverted, not claimed, owner decides; a decision-C part whose deciding test reds is deleted after its 10x re-test and the owner told; transport lapses only if G-R9 (i) fails or A3c/A3a changes nothing (POL:595-600). Every null writes its NULL-RECORD |
| **INCONCLUSIVE (underpowered)** | the cap is reached, or futility read, with mde > delta | NOT a null (POL:971-974). The engine is NOT finished: T117 stays open, nothing is deleted, the mde is reported, and the owner decides whether to fund the powered count |
| **VOID** | a section 3.4 red did not fire, a sham moved E1, the A11 digest moved, A10's sleep count > 0, a look was taken off-plan, or #6231's persistence gap unresolved for event (c) | no verdict. A defect found before any campaign seed is exposed is repaired by a dated addendum on the same seeds; after any exposed seed, only a new pre-registration on a fresh block the integrator claims |

The owner's standing order counts the engine done as "all rows + G9 on origin/main". This draft reads "G9" there as
**a G9 disposition of record on origin/main**; whether an INCONCLUSIVE or NOT MET disposition counts as "done", or only
MET does, is the owner's to say (section 6 item 8). Under no outcome does this campaign promote anything past T117.

Conflict disclosed: DI's g9-gaps addendum B (`DI:207-210`) says "the floor arm IS the pilot"; FOLD1 item 1 supersedes
DI's earlier lines with "the frozen-engine arm is the noise floor only, never the variance source". This draft follows
FOLD1 (section 4.1). The integrator decides at merge, since FOLD1 is on another branch.

## 6. Owed before freeze

| # | Item | Needs a run? |
|---|---|---|
| 1 | G9's engine built: G1(c), G3, G5, G6 landed (PLAN:2057), then every arm A1-A11, A9's shuffled twin and the O3 sham hooks built, each with graph-exists dispositions (P2: reuse `gate_from`, `state_digest`, `rsa_drift`) | build, no campaign run |
| 2 | Section 3.4's reds counted RED (including the queued `tests/test_rsa_drift.py`), logs with the command on line 1 | YES (tests; after the campaign holds the host) |
| 3 | The shared drift reader (pv / rate / tuning + `rsa_drift` moved in, one unit-set rule agreed with #6244; FOLD1 items 3, 5) | build + its reds (YES) |
| 4 | Per-contrast smallest effect of interest (delta), registered before the pilot (FOLD1 item 7) | no (author choice) |
| 5 | Pilot P0 on a disjoint seed block: paired-difference SD per contrast, bootstrap CI | YES |
| 6 | n0, the cap, the look sizes, then the section 4.4 simulation | YES (simulation; no brain) |
| 7 | Owner: if the cap exceeds host capacity, accept an expected INCONCLUSIVE, fund the runs, or narrow the primary family | no (owner answer) |
| 8 | Owner: does a NOT MET or INCONCLUSIVE G9 disposition count as "G9 on origin/main" for "engine done"? | no (owner answer) |
| 9 | Panel: adopt A10 (sleep-off) and A11 (floor) into PLAN:2043-2053 and TASKS:257, and correct TASKS:257's self-contradicting "sleep OFF" note (FOLD1 item 10); adopt A9 (O1's untrained twin) and the O3 shams likewise | no (panel + integrator edit) |
| 10 | O3's workspace rows: are both workspaces on the engine's path at the built G9? (read the built tree) | no (static read after the build) |
| 11 | #6231's persistence build GREEN before event (c) and O4 are read, or E.2's named-not-persisted list written | build (YES for its tests) |
| 12 | Seed blocks claimed on the board by the integrator (pilot block and campaign block, disjoint) | no |
| 13 | The policy-enforcer and the panel review this draft; folds append below a dated heading | no |
