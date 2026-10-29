# G9 / T117 -- the engine, end to end: campaign pre-registration (DRAFT pre-registration)

> **Status: DRAFT. Nothing in this file is frozen, and nothing has been run.**
> - Every number marked OWED below waits on a run (the pilot) or on an owner answer. No seed block is claimed.
> - Written statically by lane g9-prereg (Opus, docs only) on branch `claude/g9-prereg-1029`, parent `d8b3ef73`
>   (= `claude/g9-gaps-1029`). Policy and code anchors are read at `c4c10b28`; the arc's code equals that commit's
>   except `infra/_mathutils.rsa_drift` (added by `f0f50385`, test first in `fd987f3f`).
> - No test, brain, battery or gate was run (campaign rules 1-3, `lanes/engine-1024/LANE-RULES-CAMPAIGN-1029.md`).
> - It had no panel and no policy-enforcer read when written. **Fold 1 (2026-10-29, lane fold-g9)** folds the 2-lens panel
>   (lens A #6524, lens C #6528) and the launcher's ROUND 1b targets; the change list is the "Fold 1" section at the END,
>   the per-finding dispositions are `lanes/engine-1024/fold-g9-row/FOLD.md`. The policy lens has not read it yet.
> - **FOLD1 citations** read "FOLD1 finding n" (its findings list 1-11) or "FOLD1 owed n" (its owed list 1-8).
> - **Task:** T117 [G9] (`tasks.md:257`). The row's wording binds: this draft claims nothing T117 does not name.
> - **Anchors.** PLAN = `specs/022-unified-brain-arc/PLAN-FINAL-2026-09-02.md`; TASKS = `specs/022-unified-brain-arc/tasks.md`;
>   POL = `docs/POLICIES.md` at `c4c10b28`; DI = `evidence/g9-campaign-design-input-DRAFT.md` at `d8b3ef73`
>   (its item 1-2 text `:19-86`, its g9-gaps addendum `:129-229`); FOLD1 = DI's "Panel fold 1" addendum, commit
>   `097944a5` on `claude/hw4-g9-1029` (board #6316). **FOLD1 is NOT an ancestor of this branch** (`git merge-base
>   --is-ancestor 097944a5 HEAD` = false), so its text was read with `git diff 7c1eebe2 097944a5 -- <DI>`; it is cited
>   as "FOLD1 finding n" (Fold 1 renamed; see below). Where FOLD1 and the g9-gaps addendum disagree, this draft follows FOLD1 and says so (section 5).

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

Every arm runs on the same seeds, the same problems and taught inputs, and the same probe set. Each mechanism is
credited only after it beats its simpler equal-capacity AND equal-compute arm (PLAN:1857). Each arm names its control:
one that is not the method (POL:930).

**Equal compute (Fold 1, lens A M2).** The loop stops at solve or at budget, so "the same number of ticks" cannot hold
under adaptive stopping. Rule: every arm is ALLOTTED one registered budget B (Brain ticks and teaching exchanges per
problem; B's value OWED, section 6 item 4). An arm whose validation predicate holds before B stops growing and teaching
(the engine's own rule) and idles, untaught, to B. E1 is read once per arm, at the end of B. Compute USED (ticks and
exchanges to solve) is reported per arm beside E1 as a descriptive reading, never a pass/fail bar.

**Splits (Fold 1, lens A M1).** Each registered problem's items are split, before any seed, into TRAIN (taught), VALIDATION
(the loop's solved predicate and its stop read only these) and TEST (E1 reads only these; the loop, the grower and the
readout's tuning never query them). The split rule and sizes are registered with the problems (section 6 item 4).

### 2.1 PLAN's arms (PLAN:2043-2053; TASKS:257)

| Id | Arm | What it isolates | Control (not the method) |
|---|---|---|---|
| A0c | the engine, copy-form transport only | -- | -- |
| A0a | the engine, activity-form transport only | -- | -- |
| A1 | reservoir removed (new neurons read directly) | the reservoir | A2 (the reservoir kept, the feed cut) so A1 is not the only cut of connection 2 |
| A2 | feed neutered: reservoir driven by norm-matched random units (G3's feed control) | the feed (connection 2) | norm matching: the random units' drive norm equals the fed units' |
| A3c | A0c with its transport neutered: a norm-matched random matrix | transport (copy) | norm matching |
| A3a | A0a with its transport neutered: the brain's drive time-shuffled (not norm-matched; TASKS:257's "norm-matched" fits the copy form only) | transport (activity) | the shuffle keeps the drive's marginal statistics |
| A4 | return path neutered | the return path | a time-shuffled return (T113's P6 test, TASKS:253) keeps its marginal statistics |
| A5 | growth OFF at the engine's final size (presized) | growth | SC-012b's equal-count silencing (`spec.md:202`): silencing grown units must drop the score; silencing an equal number of presized units must not |
| A6 | random growth, matched budget | targeted growth | budget matched in units, synapses and growth events |
| A7 | G7's T1 at engine level: column batches vs single neurons | the organizing unit | single-neuron growth at the same unit budget |
| A8 | G7's T4 task-score half: rule-grown wiring vs imposed columns vs random attachment | the wiring rule | imposed columns and random attachment at equal unit and synapse counts |

**Transport forms (Fold 1, lens A M3).** PLAN:2039 runs "both forms as arms" and PLAN:2055 keeps one, so the forms are
rivals: one engine per form (A0c, A0a), each against its OWN neuter twin (A3c vs A0c, A3a vs A0a). An engine carrying
both forms would let one form mask the other's neuter, and transport would lapse while it works. "A0" below means the
REFERENCE engine for every non-transport arm. Which form the non-transport arms (A1, A2, A4-A11, shams) carry is OWED to
the integrator (section 6 item 14): (alpha) the full arm set once per form (about 2x the cost); (beta) the copy form as
the registered reference (PLAN lists it first), A0a/A3a as the second form's pair only. Sizing below assumes (beta).

**P2 keep rule (written before data).** Per form x, the transport leg is the paired contrast A0x minus A3x (section 4.1).
- One form crosses its boundary and its engine passes the absolute gate (section 3.1): P2 keeps that form.
- Both cross: P2 keeps the form with the larger paired mean difference at the stop; if the two forms' paired E1 scores are
  equivalent (TOST at delta), the owner picks (no data-driven tie-break is invented here).
- Neither crosses: transport "changes nothing" ONLY if, for BOTH forms, A0x minus A3x is shown equivalent to 0 by TOST at
  delta (one-sided alpha .05 each side); then transport lapses (POL:599-600). A non-crossing that is not equivalent is
  INCONCLUSIVE for transport, never a lapse.

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
| O1 | trained vs untrained at the same size | NEW arm A9: the engine's final topology and unit count, weights at their seeded initial values (no learning applied) | A0-shuf, a weight-shuffled twin of A0 (A0's trained weights permuted within each projection): it keeps the weight statistics and destroys the learned organization, so "trained beats untrained" is not only "trained weights are larger" |
| O2 | targeted vs random growth, matched budgets | A0 vs A6 (reuse) | A6's budget match (units, synapses, growth events) |
| O3 | disrupt specific paths; which abilities selectively disappear | the neuter arms A1, A2, A3c, A3a, A4, read as a PATH x ABILITY matrix (rows: paths; columns: each G2 problem class's held-out score). Candidate information-sharing paths add the two workspaces (`faculties/global_workspace.py`, `faculties/reentrant_workspace.py`; P2 standing adjudication, POL:252-256) as two more rows, if both are on the engine's path at the built G9 (OWED, section 6) | a SHAM neuter per path: the same hook installed, passing the signal through unchanged. A path effect is read only against its sham. "Selective" means a double dissociation, read as an INTERACTION contrast (Fold 1, lens A M5): (p:x minus p:y) minus (q:x minus q:y), one test per path pair, Holm across the path pairs; "costs x and not y" is never read from one significant and one non-significant cell, and "not y" is claimed only by equivalence (TOST) at a registered margin (OWED, section 6 item 4) |
| O4 | do the effects survive restoration from the lake | each O1-O3 contrast re-read after the fresh-process rebuild (G-R9 iv, vi) | the uninterrupted twin (G-R9's control list, PLAN:2331). Read only after #6231's persistence build is GREEN (FOLD1 finding 2), or the gap reads as a P4b red, not as an effect that did not survive |

### 2.3 The two arms the design input found missing (DI :129-212, designed by g9-gaps)

| Id | Arm | How, at the tree | Control (not the method) | Red (refuses the arm) |
|---|---|---|---|---|
| A10 | sleep OFF: no sleep bout starts; growth ON, everything else as A0 | `gate_from(brain, lambda: False)` (`instruments/sleep_pressure.py:305-320`), REUSED; not `Brain.neuter_sleep` (that is the structural part only, `engine/brain.py:3623-3638`) | the TRANSPARENCY twin: the same override replaying the clock's own scheduled gate sequence (sleep stays ON; the harness is the only change) | `_enter_sleep` count over the run: > 0 on A10 exits 3; the same count must be > 0 on A0 and the transparency twin (positive control) |
| A11 | frozen-engine FLOOR: the same events (growth slot, sleep slot, restart), nothing written | COMPOSED from existing switches (DI :187-197); no whole-engine freeze exists | a PLANTED-CHANGE twin: one known weight-row scaling between readings; each drift metric must move off the floor in the right direction | `state_digest` (`instruments/sleep_plasticity_ab.py:221`) of weights + rng equal before and after; a freeze with one write path left open moves it and exits 3 |

A11's floor readings use the same intersection unit-set rule as every engine arm (section 3.3); for a frozen engine
that is trivially the full set, with `n_pruned = n_born = 0` reported. This supersedes DI:211's "full unit set" wording
(Fold 1, lens C C-m4).

Two carried limits. (1) Growth is sleep-gated on the incumbent (`engine/brain.py:3352-3354`), so A10 also stops that
growth; a sleep-only claim is read against A5 as well, and is not made if G3's growth is also sleep-gated (unknown
until G3 is built). (2) If the lake records nothing outside a sleep bout, A10's restart leg is NOT READABLE, never
drift 0. If the panel does not adopt A10: "no claim about sleep's share of drift is made; drift after a sleep bout is
read, and not compared with a no-sleep twin" (FOLD1 finding 6).

## 3. Endpoints

### 3.1 Primary: the engine works (G-R9, PLAN:2331), read on the engine's function

- **E1 (solve, relational, held out).** Per problem: the solved predicate on the normalized held-out score of G5's
  brain-tuned readout (P4d, POL:573), read on the TEST split only (section 2). Statistic per contrast: the per-seed
  PAIRED difference (A0 minus arm) of ONE registered aggregate, one-sided (Fold 1, lens A M4): the mean over the
  contrast's registered problem set of the per-problem normalized test-split score. For G-R9 (i) that set is the
  problem(s) A5 fails, registered before the run; for every other contrast it is all registered G2 problems. So the
  family counts contrasts, not contrasts x problems; per-problem differences are reported descriptively, never tested.
- **The absolute gate, before any arm contrast counts (Fold 1, lens A B1; launcher ROUND 1b (a)).** G-R9 (i) asks the
  engine to solve AT THE BAR and (ii) asks its SOLVE to survive (PLAN:2331); a relative difference alone is reachable by
  an engine that solves nothing, because a neuter that injects random drive (A1, A2, A3c) can push its arm BELOW the
  do-nothing level. So, in order:
  - **G-a (absolute solve).** A0 meets the solved predicate on the registered problem(s), counted per seed: a one-sided
    exact binomial test that A0's per-seed solve rate exceeds a registered floor rate pi0 (OWED, section 6 item 4), its
    power at a registered pi1 included in section 4.4's joint simulation. G-a is tested at the same named looks K1-K4 as
    the legs (section 4.1), with its own Lan-DeMets O'Brien-Fleming spending of alpha .05 on the exact binomial; its
    power at pi1 at each stop is computed before data and stated beside its reading (Fold 2, #6589 MINOR-1).
  - **G-b (beats the inert baseline).** A0 > A9 on E1 (paired, one-sided). The inert baseline is A9 (the untrained engine
    at the same size, already an arm): **RULED (integrator D-72, 2026-10-29; section 6 item 14 (b))**; the no-feed
    reservoir is not an arm of this campaign (Fold 2, #6589 MINOR-4).
  - **Damage check, per neuter arm (A1, A2, A3c, A3a, A4).** The arm's E1 must not fall below A9's by more than delta:
    a one-sided non-inferiority test (arm minus A9 > -delta). It is read ONLY at a stop, never at an interim look, and
    "damage" is recorded only on (a) a POWERED failure at a stop (non-inferiority not shown at a stop whose mde for the
    check, at its alpha and power .8, is at or below delta), or (b) a SHOWN inferiority (arm minus A9 < -delta, one-sided
    alpha .05). A non-inferiority not shown at an underpowered stop is not damage: the check reads INCONCLUSIVE and the
    contrast is read as registered, its damage check reported beside it (Fold 2, #6589 MINOR-3; launcher D-72a). A neuter
    recorded as damage reads "damage, not attribution" and its G-R9 (ii) contrast is **VOID (contrast)** (section 5):
    a validity failure of THAT contrast only, never of the campaign (enforcer ruling #6589, ratified D-72a). Each neuter
    is thus read against the inert floor, not only against A0.
  - **If the absolute gate fails** (G-a or L1 A0 > A9 not rejected at a POWERED STOP: a stop at which G-a's power at
    pi1 is >= .8 and L1's mde is at or below delta): the disposition is **NOT MET (ENGINE FLOOR)**, a finding about the
    engine, and every mechanism contrast L2-L8 is NOT TESTED (not tested against a working engine; section 4.1b's NOT
    TESTED protections apply). A non-rejection at an underpowered cap or at an interim look is **INCONCLUSIVE**, never
    ENGINE FLOOR (launcher D-72a, amending D-72; Fold 2, #6589 MINOR-1). It is never VOID: VOID is for validity
    failures only (launcher D-72, 2026-10-29).
  - **Precedence (launcher D-72a).** ENGINE FLOOR is read FIRST and stands regardless of any later neuter's outcome: a
    damage VOID on a later contrast neither replaces nor erases it, and the analyst never chooses between them after
    seeing the data.
- **Contrasts and directions** (registered now; delta per contrast is OWED, section 6 item 4), in the fixed testing
  order of section 4.1:
  - G-R9 (i): A0 > A5 on the problem(s) A5 fails (prospective). A0 > A6 is the P4d random-growth CONTROL contrast
    (POL:596-597; PLAN:2331 lists random growth in G-R9's control column), not G-R9 (i)'s own wording (Fold 1, C-m1).
  - G-R9 (ii), each attributed separately: A0 > A1, A0 > A2, A0 > A4, and the transport leg A0c > A3c / A0a > A3a
    (section 2.1's keep rule).
  - G-R9 (vii) (Fold 1, lens C C-M1, lens A M4): rule-grown (A0) vs imposed columns (A8-imp), sized like the others.
    For MET: non-inferiority, A0 minus A8-imp > -delta_vii, one-sided. The decision-C red ("scores below", PLAN:2331)
    fires only on a powered one-sided superiority test of A8-imp over A0 at the stop; a result that is neither
    non-inferior nor below reads INCONCLUSIVE for (vii), never a deletion. A8's random-attachment arm (A8-rand) and A7
    (column batches vs single neurons, beside G-R3's T1 red) are reported, not MET legs.
- **Reds that are counts, not contrasts:** G-R9 (iii) the loop reports solved on a constructed unsolvable problem, or
  does not pause at its budget; (iv) a fresh process does not re-solve the saved problem without regrowth or teaching;
  (v) an earlier solved problem regresses past its bound; (vi) the reservoir rebuilt from lake records does not
  reproduce the pre-restart answers. Each is constructed and counted RED before any campaign seed (section 3.4).

### 3.2 Secondary: the owner's four (mechanism evidence only)

O1: A0 > A9 on E1 (= gate G-b, section 3.1; no second test) and A0 > A0-shuf. O2 = the A0 vs A6 control contrast
(no second test, one endpoint). O3: the path x ability matrix of E1 differences, each cell against its sham; reported
as a matrix with the double-dissociation interaction contrasts marked (section 2.2), never collapsed to one number. O4: O1-O3 re-read after the rebuild. None of these
gates T117 beyond what G-R9 already gates.

### 3.3 Secondary: representational drift (HW4 items 1-2; owner 2026-10-25 fold, TASKS:257)

- **D1 (item 1).** A fixed probe set presented before and after (a) a growth step, (b) a sleep bout, (c) a
  fresh-process restart. Reservoir spikes scored by population-vector, ensemble-rate and tuning-curve correlation
  against elapsed events, and RSA-matrix stability (`rsa_drift`). Each reading is taken over the INTERSECTION unit set
  and reports `n_common`, `n_pruned`, `n_born` beside it (FOLD1 finding 3). Every reading is reported against A11's floor.
  Drift is not a pass or fail on the engine; the biology (Deitch, Rubin & Ziv 2021) is a reference SHAPE, not a target.
- **Event (c) is not neutral** (FOLD1 finding 2): restart drift above the floor files a P4b red first. It is read as drift
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
| pv / rate / tuning reader | FOLD1 owed 6's set: identity 1.0; a planted permutation or rate change reads lower; a flat response raises the flag (not `ZeroDivisionError`, not 0.0) | OWED: no shared drift reader exists (DI :101) |
| A10 sleep-off | `_enter_sleep` count, section 2.3 | OWED with the arm's build |
| A11 freeze | `state_digest` equality, section 2.3 | OWED with the arm's build |
| A11 planted change | the floor reader moves by the planted amount | OWED with the arm's build |
| sham neuters (O3) | per path, at the stop: the sham's E1 must be shown equivalent to A0's by TOST at margin delta (one-sided alpha .05 each side); a sham shown different moves VOID for that path (the hook, not the path, moved it); a sham neither equivalent nor different leaves that path's O3 cells INCONCLUSIVE (Fold 1, lens A m2). Before data, a planted non-transparent hook (one known scaling) must read different: that is the red that must fire | OWED with the hooks |
| G-R9 (iii)-(vi) | constructed per PLAN:2331 | OWED; no counted G-R9 red was found by this lane (static read, not a census) |
| A9 untrained | the twin's weight digest equals the seeded init digest | OWED with the arm's build |
| damage check (B1) | a planted damaging neuter (drive injected at a known norm that lowers E1 below A9) must read "damage" at a stop, and its contrast VOID (contrast) | OWED with the hooks |

## 4. Sample size (P1)

This section carries the P1 amendment's fields (POL:183-198). No count in it is a decision. TASKS:257's "S = 8 under
Part F's rule" is NOT carried as a sample size (FOLD1 finding 5; POL:180-182, the E1 incident).

### 4.1 Fields

| Field | Value |
|---|---|
| Measured SD | **OWED: pilot P0.** Per contrast, the SD of the per-seed PAIRED difference (arm minus A0) on shared seeds. A11's floor is NOT the variance source (FOLD1 finding 1; a frozen arm can read SD near 0 and size a tiny n) |
| Pilot P0 | A0c, A0a and EVERY contrast arm (A1-A11, A8-imp, A8-rand, A0-shuf, the O3 shams) on the same pilot seeds, a block disjoint from the campaign's; its seed count is a one-shot whose only job is an SD estimate, so it reports the SD with a bootstrap 95% CI and sizes the campaign at the SD's 80% upper confidence bound (the E1d precedent, `e1d-plan-w-pc2.md:213-218`: SD 0.3097, 95% CI 0.206-0.400, sized at the 80% upper SD 0.345). **Pilot count rule** (Fold 2, #6589 MINOR-4): the pilot's seed count n_p is the smallest n with sqrt((n - 1) / chi2_{.20; n-1}) <= r, the ratio of the SD's 80% upper bound to the point SD, r registered before the pilot (section 6 item 4) |
| Claimed effect (delta) | **OWED before the pilot** (FOLD1 owed 7): one smallest effect of interest per contrast (delta_vii for G-R9 (vii)'s non-inferiority margin included), on the normalized test-split score for E1 and per metric for D2. Chosen by the pre-registration author from the engine's purpose, never from pilot data |
| Power and test | joint P(MET \| delta on every leg) >= .8 (section 4.1b); one-sided alpha .05 per leg; two-sided (.025 per side) for D2's neutered-path contrasts |
| Multiplicity | the MET legs are a FIXED-SEQUENCE (gatekeeping) family, section 4.1b; each leg at full alpha .05, the transport leg's two forms at .025 each. Secondary families (O, D) are sized and reported separately, each with its own Holm correction, and never enter the primary family |
| Fixed-sample target | per leg n0 = ((z_{1-a_leg} + z_{p*}) x SD_upper / delta)^2, rounded up, p* = .8^(1/K) the per-leg power (section 4.1b); the campaign's n0 is the MAX over the legs (every arm runs on every seed) |
| Design | sequential, named looks K1-K4 (renamed from L1-L4 so "L1" names only the leg; Fold 2, #6589 MINOR-2) at 25/50/75/100% of the cap; Lan-DeMets O'Brien-Fleming spending per leg at that leg's alpha; a leg is tested at a look only once every earlier leg has crossed at that look or before |
| Futility | non-binding; it opens only at a look whose mde, computed from the pre-registered SD at the leg's alpha, is at or below delta (POL:191-194); before such a look the campaign continues. The mde is ALSO reported at the campaign's observed SD; if that mde exceeds delta the read is INCONCLUSIVE, not powered (Fold 1, lens A m1) |
| Hard cap | ceil(n0 x the design's inflation factor), JOINT power at the cap >= .8 (section 4.1b). For O'Brien-Fleming at 4 looks the factor is small (about 1.02-1.03); the value used is the one the simulation measures, not this note |
| Stop rule | stop only at a named look, only on a written boundary: every MET leg crossed, or futility on a leg (non-binding), or the cap. A NOT MET or INCONCLUSIVE disposition is decided only at a stop, never at an interim look (Fold 1, lens A m3); seeds added after an unregistered look void the reading |

### 4.1b The conjunction: why fixed-sequence, and the seed cap (Fold 1, lens A B2; launcher ROUND 1b (b))

MET needs EVERY leg to cross. Sized at .8 each, 7 legs give P(MET) near .8^7 = .21 under independence (higher with
the shared A0, still far under .8), and the leg that misses sits at the cap with mde <= delta, so a working engine would
be filed a powered failure more often than a success. Two designs fix this; this draft picks **(ii), fixed-sequence**.

- **The legs, in testing order** (K = 8 E1 legs; G-a and the damage checks are conditions, inside the simulation):
  G-a (absolute solve, binomial) -> L1 A0 > A9 (G-b) -> **L2 A0 > A5, the named PRIMARY contrast** (G-R9 (i), the
  owner's condition) -> L3 A0 > A6 -> L4 A0 > A1 -> L5 A0 > A2 -> L6 A0 > A4 -> L7 (vii) A0 vs A8-imp non-inferiority ->
  L8 transport, {A0c > A3c, A0a > A3a} at alpha .025 each, passing if either crosses (last, so its split passes no
  alpha onward). The order runs from the absolute gate, through the claim the engine exists for, to the attributions.
- **Why (ii) over (i) (joint power with Holm).** MET is an intersection-union rule: it is claimed only if every leg
  rejects, and an intersection-union test at level alpha needs no split of alpha across its legs (each leg at alpha
  bounds the error of the MET claim at alpha). Holm at alpha/K controls an error MET never makes (claiming "at least
  one leg") and pays for it in seeds. Fixed-sequence keeps each leg at full alpha and STILL controls the family-wise
  error of every individual attribution in the strong sense, because a leg is tested only after all earlier ones
  reject, which is the order MET reads anyway. Per (SD_upper/delta)^2 at joint power .8: Holm at alpha/K needs about
  19.5 (K = 8), fixed-sequence about 12.7 per leg and 15.0 for the transport leg (alpha .025), so about 0.77x the seeds.
- **What (ii) gives up.** A leg after the first failure is NOT TESTED: it is reported descriptively, it is not a null,
  it writes no NULL-RECORD, it deletes nothing, and it cannot lapse transport. A failure early in the order therefore
  leaves the later attributions unread; this is the price of the cheaper cap, and it is stated here before data.
- **A VOID (contrast) leg in the sequence** (Fold 2, #6589 MAJOR-1 (ii)-(iii); enforcer rule adopted, ratified by the
  launcher D-72a). Fixed before data: the VOID leg counts as NOT CROSSED; every later leg is NOT TESTED (with the
  protections above); every earlier leg keeps its verdict, NOT MET (ENGINE FLOOR) included. The VOID leg itself carries
  no verdict and writes no NULL-RECORD. T117 reading: if no powered failure precedes the VOID leg, the campaign reads
  **INCONCLUSIVE** (not VOID, not NOT MET); if a powered failure precedes it, that earlier verdict stands.
- **Per-leg power.** p* = .8^(1/K) = .9725 for K = 8 (z = 1.919). The legs share A0, so they are positively correlated
  and the product is conservative; section 4.4's simulation measures the joint P(MET | delta), with G-a and the five
  damage checks inside it, and raises the cap until it is >= .8. If G-a and the damage checks are counted as legs
  (K = 14, p* = .9842), the multiplier for the transport leg becomes about 16.9.
- **The seed cap that results** (it feeds owner question #6514). Analytic starting point, before the simulation:
  cap = ceil(1.03 x 15.04 x max_leg (SD_upper / delta)^2), the transport leg setting the max. At SD_upper/delta = 1,
  1.5, 2: **16, 35, 62 seeds** (K = 14 sensitivity: 18, 40, 70; Holm alpha/8 for comparison: 21, 46, 81). The ratio is
  unknown until the pilot measures SD (section 6 item 5), so no number here is a decision. At about 23-25 arm runs per
  seed (section 4.2), 16 seeds is about 370-400 engine runs and 62 seeds about 1,430-1,550. The 16 / 35 / 62 cap and
  the 23-25 runs per seed assume design (beta) (section 2.1); under (alpha) both are re-derived (about 15 more runs per
  seed, section 4.2) before freeze (section 6 item 14 (a); Fold 2, #6589 MINOR-4).
| Simulated before data | yes, OWED (section 4.4) |

### 4.2 Cost disclosure

One seed of G9 is a whole engine run per arm. Under design (beta) (section 2.1), counted (Fold 1, lens C C-m6): A0c,
A0a, A1, A2, A3c, A3a, A4, A5, A6, A7, A8-imp, A8-rand, A9, A0-shuf, A10, A10's transparency twin, A11, A11's
planted-change twin (18), plus 5-7 O3 shams: **about 23-25 arm runs per seed** before the O4 rebuild leg. Design
(alpha) adds about 15 more. If the cap is beyond what the host can run, the answer is NOT a smaller habitual S: the campaign is
expected to read INCONCLUSIVE at that S, and the owner decides (section 6 item 7). A fixed one-shot count is allowed
only with its power stated before data (POL:195-196).

### 4.3 P8 null-scale axis (POL:944-994)

Each null writes one `NULL-RECORD:` line (units the method acted on, ticks per run, seeds, mde, re-test state).
The axis each contrast scales with, named here (the author names it; a disputed axis is decided by the author with a panel review, POL:955-957):

| Contrast | Axis | Why |
|---|---|---|
| G-a, L1 (A0 vs A9): NOT MET (ENGINE FLOOR) (Fold 2, #6589 MINOR-2) | grown units (the engine as a whole) | the floor is a finding about the whole engine, whose capacity is its grown population |
| A5, A6, A7, A8, O1, O2 (growth, organizing unit, wiring, training at size) | grown units | capacity mechanisms |
| A1, A2 (reservoir, feed) | grown units fed | the path's capacity is the fed population |
| A3c, A3a (transport), A4 (return) | ticks per run | the effect is learned through the path over time |
| A10 (sleep), D1/D2 drift | sleep bouts (ticks) | a consolidation mechanism scales with bouts |

Owner-drawn parts (the two networks, the reservoir, the return path) are never deleted on a null: implementation
reverted, not claimed, owner decides, NULL-RECORD still written (POL:598). A decision-C part (A7, A8) is deleted only
after the one 10x re-test (POL:598, 955). An underpowered result is INCONCLUSIVE, never a null (POL:971-974).

### 4.4 The simulation check (before any campaign seed)

On the pilot's paired-difference distributions (resampled, centered for the null), with the legs' observed
correlation kept: a planted null on any subset of legs keeps each leg's false-positive rate at or below its alpha and
the family-wise rate at or below .05 under the fixed-sequence group-sequential order (section 4.1b); a planted effect of
size delta on EVERY leg (G-a at pi1, every damage check at its null) gives **joint P(MET | delta) >= .8** by the cap,
reported as the design's power (Fold 1, lens A B2); the probability of every outcome (MET at each look, futility,
uncrossed cap, NOT TESTED legs, INCONCLUSIVE, NOT MET (ENGINE FLOOR), VOID (contrast) by damage) is reported, and
P(VOID (contrast)) per neuter under the planted effect is reported beside the design's power (Fold 2, #6589 MAJOR-1 (v)). A design that fails the simulation is redesigned before data, not run.

## 5. Disposition, fixed in advance, and what "engine finished" means

Per primary contrast, and then for the campaign:

| Outcome | Rule | What it means for T117 |
|---|---|---|
| **MET** | the absolute gate holds (G-a and L1, section 3.1), no damage check records damage (section 3.1), every MET leg L2-L8 crosses its boundary in the registered direction (L7 = G-R9 (vii) non-inferiority; L8 = at least one transport form), no G-R9 (iii)-(vi) count red fires, every section 3.4 red fired before data | T117's condition holds: "the engine can work" (P4d, POL:595) is met at the measured scale, and transport stands. The box may be ticked with the closing record. **Nothing beyond T117's wording is claimed**: no mechanism is credited outside its own contrast, the O-readings are mechanism evidence only, and no claim about experience is made (POL:1014) |
| **NOT MET (ENGINE FLOOR)** (own row: Fold 2, #6589 MINOR-2) | read FIRST (precedence, D-72a): G-a or L1 (A0 > A9) not rejected at a POWERED STOP (G-a's power at pi1 >= .8, L1's mde <= delta, section 3.1); a non-rejection at an underpowered cap or an interim look is INCONCLUSIVE, not this row (D-72 amended by D-72a). It stands whatever any later neuter's outcome | T117 closes as a measured failure of the engine as a whole. L2-L8 are NOT TESTED: section 4.1b's protections cover them (no NULL-RECORD, nothing deleted, transport not lapsed). One NULL-RECORD is written on the engine-floor axis (section 4.3) |
| **NOT MET, powered** | decided only at a stop: a tested leg's boundary is not crossed (or futility taken) at a look whose mde <= delta at BOTH the pre-registered and the observed SD, or a G-R9 (iii)-(vi) count red fires, or (vii)'s powered superiority test of A8-imp over A0 fires. Legs after the failing one are NOT TESTED (section 4.1b) and carry no verdict | T117 closes as a measured failure. Per part: an owner-drawn part is reverted, not claimed, owner decides; a decision-C part whose deciding test reds is deleted after its 10x re-test and the owner told; transport lapses only if G-R9 (i) fails or both forms' neuters are shown to change nothing by TOST at delta (section 2.1's keep rule; POL:595-600); a decision-C part is deleted only on its own TESTED, powered red, never on a NOT TESTED leg. Every null writes its NULL-RECORD |
| **INCONCLUSIVE (underpowered)** | the cap is reached, or futility read, with mde > delta | NOT a null (POL:971-974). The engine is NOT finished: T117 stays open, nothing is deleted, the mde is reported, and the owner decides whether to fund the powered count |
| **VOID (contrast)** (Fold 2, #6589 MAJOR-1; ruled by the enforcer, ratified D-72a) | a neuter recorded as damage at a stop (section 3.1: a powered failure of its non-inferiority check, or a shown inferiority): "damage, not attribution", a validity failure of THAT contrast only | that leg carries no verdict and writes no NULL-RECORD; in the fixed sequence it counts as not crossed, later legs are NOT TESTED, earlier legs keep their verdicts, NOT MET (ENGINE FLOOR) included (section 4.1b). T117 reads INCONCLUSIVE when no powered failure precedes the VOID leg. Repair: a rebuilt neuter on a fresh seed block the integrator claims, paired with A0 re-run on that block |
| **VOID** (campaign) | validity failures only (D-72): a section 3.4 red did not fire, a sham was shown to move E1, the A11 digest moved, A10's sleep count > 0, a look was taken off-plan, or #6231's persistence gap unresolved for event (c) | no verdict. A defect found before any campaign seed is exposed is repaired by a dated addendum on the same seeds; after any exposed seed, only a new pre-registration on a fresh block the integrator claims |

The owner's standing order counts the engine done as "all rows + G9 on origin/main". This draft reads "G9" there as
**a G9 disposition of record on origin/main**; whether an INCONCLUSIVE or NOT MET disposition counts as "done", or only
MET does, is the owner's to say (section 6 item 8). **ANSWERED by the owner 2026-10-29 (#6513; see the last section): only
a MET disposition counts.** A NOT MET or INCONCLUSIVE G9 means more engine work, not a close. Under no outcome does this campaign
promote anything past T117.

Conflict disclosed (Fold 1, lens C C-m3: three sites, not one). DI's g9-gaps addenda at `d8b3ef73` make the frozen
floor the variance source at THREE places: `DI:172` (addendum A: sleep-off SD "from the frozen-engine pilot (section
B)"), `DI:185-186` ("the only arm that yields the PILOT SD") and `DI:207-210` (addendum B: "the floor arm IS the
pilot"). FOLD1 finding 1 (`097944a5`) supersedes DI's earlier lines with "the frozen-engine arm is the noise floor
only, never the variance source"; FOLD1 never saw the addenda (sibling branches). DI:211's full-unit-set floor is also
superseded here (section 2.3). This draft follows FOLD1 (section 4.1). **RULED (integrator D-72, 2026-10-29; section 6
item 14 (c)):** the floor arm is not the pilot; FOLD1 finding 1 governs the three addendum sites at the merge, since
FOLD1 is not on this base (Fold 2, #6589 MINOR-4).

## 6. Owed before freeze

| # | Item | Needs a run? |
|---|---|---|
| 1 | G9's engine built: G1(c), G3, G5, G6 landed (PLAN:2057), then A0c, A0a, every arm A1-A11, A8-imp, A8-rand, A0-shuf and the O3 sham hooks built, each with graph-exists dispositions (P2: reuse `gate_from`, `state_digest`, `rsa_drift`) | build, no campaign run |
| 2 | Section 3.4's reds counted RED (including the queued `tests/test_rsa_drift.py`), logs with the command on line 1 | YES (tests; after the campaign holds the host) |
| 3 | The shared drift reader (pv / rate / tuning + `rsa_drift` moved in, one unit-set rule agreed with #6244; FOLD1 finding 3, FOLD1 owed 5) | build + its reds (YES) |
| 4 | Registered before the pilot (author choice): per-contrast delta and delta_vii (FOLD1 owed 7); G-a's floor rate pi0 and its planned rate pi1; the pilot's SD-bound ratio r (section 4.1's pilot count rule); the budget B (section 2); the train/validation/test split rule and sizes; O3's equivalence margin | no (author choice) |
| 5 | Pilot P0 on a disjoint seed block: paired-difference SD per contrast, bootstrap CI | YES |
| 6 | n0, the cap, the look sizes, then the section 4.4 simulation (joint P(MET \| delta) >= .8) | YES (simulation; no brain) |
| 7 | Owner (#6514): if the cap (section 4.1b: 16 / 35 / 62 seeds at SD_upper/delta = 1 / 1.5 / 2, about 23-25 arm runs per seed) exceeds host capacity, accept an expected INCONCLUSIVE, fund the runs, or narrow the primary family. **ANSWERED 2026-10-29: fund the runs** (last section) | no (owner answer) |
| 8 | Owner: does a NOT MET or INCONCLUSIVE G9 disposition count as "G9 on origin/main" for "engine done"? **ANSWERED 2026-10-29: no, only MET** (last section) | no (owner answer) |
| 9 | Panel: adopt A10 (sleep-off) and A11 (floor) into PLAN:2043-2053 and TASKS:257, and correct TASKS:257's self-contradicting "sleep OFF" note (FOLD1 finding 10); adopt A9 (O1's untrained twin and the inert baseline), A0-shuf, A0c/A0a and the O3 shams likewise; remove "S = 8 under Part F's rule" from PLAN:2055 and TASKS:257 (Fold 1, lens C C-m8; POL:180-182) | no (panel + integrator edit) |
| 10 | O3's workspace rows: are both workspaces on the engine's path at the built G9? (read the built tree) | no (static read after the build) |
| 11 | #6231's persistence build GREEN before event (c) and O4 are read, or E.2's named-not-persisted list written | build (YES for its tests) |
| 12 | Seed blocks claimed on the board by the integrator (pilot block and campaign block, disjoint) | no |
| 13 | The policy-enforcer and the panel review this draft; folds append below a dated heading | no |
| 14 | Integrator rulings (Fold 1): (a) design (alpha) or (beta) for the non-transport arms' transport form (section 2.1): **OWED**, decided before the freeze from pilot cost only; the cap and runs per seed are re-derived if (alpha) (section 4.1b); (b) **RULED (D-72)**: A9 is the inert baseline (section 3.1); (c) **RULED (D-72)**: the floor arm is not the pilot, and FOLD1 finding 1 governs DI:172, :185-186, :207-210 and DI:211 at merge (section 5) | (a) no (integrator ruling); (b), (c) done |
| 15 | Owner: confirm the four causal-organization arms (O1-O4) as secondary, non-gating mechanism evidence; the owner's memory calls them arms "for G9's deciding campaign" (lens C note) | no (owner answer) |

## Fold 1, 2026-10-29 (lane fold-g9; panel lens A #6524, lens C #6528; launcher ROUND 1b)

Docs only: nothing run. Per-finding dispositions: `lanes/engine-1024/fold-g9-row/FOLD.md`. Changes:

1. Header: fold note; FOLD1 citations read "finding n" / "owed n" (C-m2), corrected at sections 2.3, 3.4, 4.1, 6.
2. Section 2: equal compute = one registered budget B per arm, compute used reported (A-M2); train/validation/test
   splits, the loop never queries the test split (A-M1).
3. Section 2.1: A0 split into A0c / A0a, each with its own neuter twin; written P2 keep rule; "changes nothing" =
   TOST at delta for both forms (A-M3). A3a disclosed as not norm-matched (C note). Design (alpha)/(beta) OWED (item 14).
4. Section 2.2: A0-shuf named (C-m5); O3 double dissociation as an interaction contrast, Holm over path pairs, "not y"
   by TOST (A-M5).
5. Section 2.3: A11's floor uses the intersection rule (trivially full set); supersedes DI:211 (C-m4).
6. Section 3.1: E1 read on the test split, one registered aggregate per contrast (A-M4); the absolute gate G-a / G-b
   and a per-neuter damage check against A9 before any contrast counts (A-B1, ROUND 1b (a)); A0 > A6 labelled the P4d
   control contrast (C-m1); G-R9 (vii) sized as non-inferiority with a powered superiority red (C-M1, A-M4).
7. Section 3.2: O1 reuses G-b; O2 relabelled; O3 interaction contrasts.
8. Section 3.4: sham red given a TOST margin and a planted positive (A-m2); damage-check red added (A-B1).
9. Section 4.1: pilot arm list and the e1d precedent wording corrected (C note); joint power; fixed-sequence; mde at the
   observed SD too (A-m1); NOT MET only at a stop (A-m3).
10. New section 4.1b: fixed-sequence gatekeeping chosen over Holm joint power, with the reason, the cost (NOT TESTED
    legs), per-leg power .9725, and the seed cap 16 / 35 / 62 at SD_upper/delta = 1 / 1.5 / 2 (A-B2, ROUND 1b (b)).
11. Section 4.2: arm runs per seed counted, 23-25 (C-m6). Section 4.4: joint P(MET | delta) >= .8 reported (A-B2).
12. Section 5: MET / NOT MET / VOID rows rewritten for the gate, damage, NOT TESTED legs and the keep rule; the
    conflict disclosure names three DI sites and is OWED to the integrator (C-m3, brief step 4).
13. Section 6: items 1, 3, 4, 6, 7, 9 extended (C-m8: remove "S = 8"); items 14 (integrator rulings) and 15 (owner) added.
14. Declined: C-m7 (`brain.py:3352-3354` holds at `c4c10b28`: `def _sleep_tick` is :3352, "Growth is thus sleep-gated"
    is :3354), see FOLD.md.

## D-72 (integrator, 2026-10-29; commit fbd29d67, #6553)

Recorded here after the fact (Fold 2, #6589 NIT: the edit was inline with no dated entry). Changes made by the integrator:

1. Section 3.1: a failed absolute gate (G-a, or L1 A0 > A9) is NOT MET (ENGINE FLOOR), L2-L8 uninterpretable, never VOID;
   VOID covers validity failures only.
2. Section 5: the NOT MET row carries ENGINE FLOOR; the VOID row is restricted to validity failures.

## Fold 2, 2026-10-29 (lane fold-g9 round 2; enforcer policy lens #6589; launcher D-72a)

Docs only: nothing run. Per-finding dispositions: `lanes/engine-1024/fold-g9-row/FOLD.md` (Fold 2 table). Changes:

1. Section 3.1, G-a: tested at looks K1-K4 with its own O'Brien-Fleming spending; power at pi1 stated per stop (MINOR-1).
2. Section 3.1, G-b: A9 RULED the inert baseline (D-72, item 14 (b)); the no-feed alternative removed (MINOR-4).
3. Section 3.1, damage check: read only at a stop; damage = a powered failure or a shown inferiority, else INCONCLUSIVE
   (MINOR-3, D-72a); its consequence is VOID (contrast), never campaign-wide (MAJOR-1).
4. Section 3.1, absolute gate: NOT MET (ENGINE FLOOR) only at a POWERED STOP, else INCONCLUSIVE (MINOR-1; D-72 amended by
   D-72a); "uninterpretable" relabelled NOT TESTED (MINOR-2); precedence: ENGINE FLOOR read first and stands (D-72a).
5. Section 3.4: damage red reads at a stop and yields VOID (contrast).
6. Section 4.1: pilot count rule (ratio r of the SD's 80% upper bound to the point SD; MINOR-4); looks renamed K1-K4
   (MINOR-2).
7. Section 4.1b: a VOID (contrast) leg = not crossed, later legs NOT TESTED, earlier legs keep verdicts, T117 reads
   INCONCLUSIVE when no powered failure precedes it (MAJOR-1 (ii)-(iii)); cap and runs per seed assume (beta),
   re-derived under (alpha) (MINOR-4).
8. Section 4.3: engine-floor axis row (grown units, the engine as a whole) (MINOR-2).
9. Section 4.4: P(VOID (contrast)) per neuter reported under the planted effect (MAJOR-1 (v)).
10. Section 5: own NOT MET (ENGINE FLOOR) row with NOT TESTED protections for L2-L8 (MINOR-2); VOID split into VOID
    (contrast), with its sequence effect, T117 reading and repair (MAJOR-1 (i)-(iv)), and campaign VOID; the conflict
    disclosure's item 14 (c) marked RULED (MINOR-4).
11. Section 6: item 4 adds the pilot ratio r; item 14 (b), (c) RULED, (a) OWED from pilot cost only (MINOR-4).
12. This file: the D-72 entry above (NIT).

## Owner answers, 2026-10-29 (relayed by the launcher from the owner's chat, transcript e49a47ce; integrator fold, pre-data)

- **#6513 (section 6 item 8), verbatim:** "a NOT MET or INCONCLUSIVE G9 does not count towards Engine done, you should start a
  research and debate loop right using /webgrab right away if you run into any novel problems." So "engine done" needs a G9
  **MET** disposition of record on origin/main. NOT MET (ENGINE FLOOR included) and INCONCLUSIVE route to more engine work.
  The second clause is a standing order for every novel problem this campaign meets: a research lane using webgrab plus a
  multi-lens debate, BEFORE a fix is built.
- **#6514 (section 6 item 7), verbatim:** "Fund more runs". If the powered cap of section 4.1b exceeds what this host runs in a
  working session, the campaign runs to that cap on this host with a longer wall-clock (approved). It does NOT narrow the
  primary family or accept an expected INCONCLUSIVE in place of the cap. Any PAID cloud compute still needs the owner's
  confirmation of the exact spend first (launcher relay). Section 4.2's "the owner decides" is thereby decided.
- These answers change no bar, no seed block and no decision rule of this draft: they settle what a disposition MEANS for the
  engine and how its cap is paid for. The draft stays pre-freeze (section 6 items 1-6 and 9-14 are unchanged).
