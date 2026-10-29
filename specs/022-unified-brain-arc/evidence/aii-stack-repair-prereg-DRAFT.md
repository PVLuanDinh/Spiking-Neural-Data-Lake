# A_ii STACK REPAIR -- make the CANONICAL brain learn A_ii (DRAFT pre-registration)

> **Status: DRAFT, pre-data, NOT frozen.** Lane aii-prereg (thread AII-PREREG), 2026-10-29, written on
> `claude/aii-prereg-1029` off `c4c10b28`. Docs only: no brain, no battery, no run of any kind went into this file.
> Every number below is quoted from a cited record; none was measured here. The only arithmetic computed here is the
> sample-size design in section 4 (stdlib python, no brain).
> - **Why it exists:** owner #2135 (2026-10-13, answering Q-AII #2134), verbatim selection: *"Fix the stack first"*.
>   Ruling as relayed: G2(ii) and A_ii stay parked; *"first build and measure a repair of how E/I balance, Benna-Fusi
>   consolidation and homeostatic scaling interact so the CANONICAL brain learns A_ii (neuter A/B, reachable red, no
>   safety relaxation); then run (ii) on the repaired brain. G2(ii), G3 C11 on S0C1 and G9 wait on that arc."*
>   The dated note in `g2-taught-problems.md:647-652` (at `c4c10b28`) records the same answer.
> - **The gap it closes:** G2 2.18 F7 (on `claude/g2-coverage-1029@4b6bc9ea`, `g2-taught-problems.md:670` and `:681`):
>   A_ii's A4 reading FAILS its registered requirement and "freezes only after that repair"; and 2.18.2 row P1 (`:684`):
>   A_ii owes a "Sample size (P1)" section. Section 4 below is that section for the repair's confirmatory read.
> - **Freezes when:** every item in section 6 is closed, the 3-lens panel and the policy-enforcer AGREE, and the
>   freezing commit drops "DRAFT" from the title.

## 0. Raising-ambition record (self-apply)

- **Axis:** Depth. The question is fixed by the owner (make the canonical brain learn A_ii without relaxing safety).
  Six rounds failed at it by guessing a mechanism and measuring it. A top practitioner localizes the loss first, then
  edits the one part that causes it.
- **Ladder.**
  - **R1:** freeze one more candidate (round 7's unbuilt H7, extending C6's shared reference to single-row turns) and
    measure it against the old conjunction. This is what rounds 1-7 did, six times, to six nulls.
  - **R2:** first measure, per stabilizer and per turn type, where the taught contrast is lost. The measurement carries
    a bookkeeping-identity red. A rule written here, before data, then picks the ONE mechanism to edit. The candidate
    is frozen in a dated pre-data addendum and confirmed on a fresh seed block, with a P1-derived design.
  - **R3:** build a reduced model of the stack's per-deposit map that predicts each candidate's contrast ratio R and its
    stable-10 outcome before any brain run. Backtest it on the six recorded nulls C1-C6, then build only the candidate
    it predicts will pass.
- **Pick: R2.**
  - R3 is ruled out by **P2**: a model that predicts `stabilize`'s output re-implements `stabilize`'s dynamics, a second
    mechanism for one job. Its output would also be load-bearing (it chooses the repair), so P8 "What an invented
    method owes" would require its own red, control and attribution. That is a separate arc, not this one.
- **Range.**
  - Control engineering: when several regulators act on one variable, open each loop separately and measure what each
    one removes before retuning any of them (section 2.3, stage A).
  - Adaptive clinical trials (seamless phase II/III): the selection stage freezes its selection rule before data, and
    its patients never count toward the confirmatory analysis (stage A and the dev seeds never count, section 4).
  - Adversarial auditing: construct the "repair" that passes A_ii by switching safety off (the three-off arm solves
    A_ii, `g2:558-564`), and require the bar to reject it (red RD-3, section 3.3).
- **Outside judge** (none of these is the method itself):
  - G2's registered `solved` / stable-10 reading through `taught_campaign.attempt` (`g2:174-178`), written before this arc;
  - the original stack's reproduced failure on the same seeds;
  - the bare learner and the three-off arm, which do learn A_ii through the same harness;
  - B5's constructed non-learners (lr 0, shuffled labels; `g2:211`);
  - the stack module's own self-checks, unchanged;
  - the 3-lens panel and the policy-enforcer.

## 1. What is claimed, and what is not

- **Claim (if MET):** a single edit to one existing mechanism of `PlasticityRuleStack` makes the canonical incumbent
  (`taught_problems.canonical_incumbent`, `taught_problems.py:386-398`: the class the live brain builds plus the stack
  `Brain.__init__` attaches, `engine/brain.py:375-380`; B8 proves them equal, `taught_campaign.py:343-360`) solve A_ii
  by G2's registered reading on every confirmatory seed. E/I balance, Benna-Fusi consolidation and homeostatic scaling
  all stay live, and every existing safety bound is unchanged.
- **Not claimed:** anything about G2(ii), G3 C11 or G9 (each runs under its own registration after a MET); recognition
  quality on live chat; any change to the five `brainstem/` envelope files (not touched, not needed).
- **Lands only on a measured win.** The repair changes what the live brain learns (brain.py:353-363). Only MET licenses
  a landing, and then only through the normal arc: panel, battery, push. Every other disposition leaves the live brain
  byte-identical.

## 2. The stack, what is broken, and the candidate repair

### 2.1 What "the stack" is (code at `c4c10b28`)

`SpikingInteractionLearner._deposit` (`readout/interaction_learning.py:512-565`) runs one R-STDP deposit:
`pre_gain` (PRE, `:535`), the reward-gated eligibility delta (`:542`), then `stack.stabilize` on the whole row
(POST, `:558-560`). `interact` (`:628-650`) deposits +1 on the taught target on EVERY teaching turn, and -1 on the
learner's own wrong answer only when `pred != tgt` (`:645-648`). A correct turn is therefore a single-row deposit; an
error turn is a pair.

`PlasticityRuleStack.stabilize` (`plasticity/plasticity_rule_stack.py:318-329`) composes, in this fixed order:

| Order | Mechanism | Code | What it does to the row |
|---|---|---|---|
| 1 | base: homeostatic scaling | `homeostatic_plasticity_stack.py:106-139` (`_HomeostaticScaling`) | a soft ceiling. The set-point is calibrated ONCE to `ceiling_factor` (3.0) x the first-sight drive (`:132-133`); scaling acts only when the drive exceeds it (`:135-138`), multiplicatively |
| 1b | base: intrinsic plasticity | `homeostatic_plasticity_stack.py:142-166` | rate set-point (not in the owner's three; does not block A_ii, `g2:563`) |
| 2 | E/I balance | `plasticity_rule_stack.py:150-191` (`_EIBalance`) | an iSTDP controller; a multiplicative trim `f = 1 - min(max_trim, k*inhibition)` on the WHOLE row at every deposit (`:189-191`) |
| 3 | Benna-Fusi consolidation | `plasticity_rule_stack.py:194-256` (`_BennaFusiConsolidation`) | deposits `weights - prev` into each synapse's fast beaker, diffuses one step toward the deep beakers (initialized to the first-sight weights), and returns the fast beaker (`:226-240`) |
| 4 | BCM metaplastic gate | `plasticity_rule_stack.py:87-148` | observes the drive for the NEXT `pre_gain`; does not block A_ii (`g2:563`) |

graph-exists (run in this lane, `--root` this worktree) finds exactly these as the live implementations: `_EIBalance`
(also `inhibitory_stdp.py`, the reused controller), `_BennaFusiConsolidation` (also `benna_fusi.py`, the reused
synapse), and `_HomeostaticScaling` (also `homeostatic_scaling.py`, the reused neuron). A repair edits one of these,
or the composing method `stabilize`. It never adds a parallel stabilizer (P2).

### 2.2 What is broken (each claim cited; all dev seeds 8000-8003, E 400, unless named)

1. **The canonical incumbent never stable-solves A_ii.** It fails on dev seeds 8000-8003 (`g2:534`; per-epoch reads
   `g2:566-571`: mean accuracy 0.491-0.497, an isolated 1.0 read on 3 seeds) and on the confirmatory block 8200-8207
   (`g2:640-645`: `inc.aii 0/8`, `a_fails` true).
2. **The construction is learnable; the stack blocks it.** The bare learner through the same harness and criterion
   stabilises A_ii at epochs 118/133/140/141 (`g2:560-561`). E/I balance, Benna-Fusi and homeostatic scaling EACH
   block alone (only-one-on); no single switch-off or pair switch-off restores learning; all three off restores it
   (174/156/179/141). The metaplastic gate and intrinsic plasticity do not block (`g2:561-564`).
3. **The four pre-registered repair nulls of 2026-10-13** (row `aii-stack-repair-2026-10-13`, `REPORT.md` at
   `C:/Users/vinhl/.config/opencode/sndl-runs/2026-10-09-step5/rows/aii-stack-repair-2026-10-13/`; board results):
   - C1/C2: a shared clock dt on all three stabilizers (REPORT:38-53). Stable-10 0/4. Both fail the old retention
     self-check (#2252).
   - C3: C1's clock plus a lagged absolute-drive baseline for the scaling ceiling (REPORT:301-310). First stable-10
     windows, but tail/future 0/4 (#2313).
   - C4: cascade coordinates, with E/I as a fresh readout gain (REPORT:524-535). First/tail 0/4 (#2367).
   - C5: a dual fast/slow store (REPORT:824-857). 0/4 (#2406).
   - Two more nulls followed on 2026-10-14: C6, a joint target/loser pair stabilization (REPORT:1532-1595, #2520), and
     the norm60 projection arm of round 6 (REPORT:1372-1384). That makes six recorded candidate nulls in all
     (handoff-1014:37).
4. **Round 6 (H6 SUPPORTED 4/4, descriptive, REPORT:1293-1444).** The raw teaching deposit moves the task contrast
   b = (g - o)/2 the useful way on all 28 arms. In the original stack, the post-deposit stabilization nearly cancels it:
   raw gain about +0.00094-0.00098 per epoch, post-stack contribution about -0.00094 to -0.00097 (REPORT:1400-1404).
   The primary R = net/raw contrast ratio is 0.0016/0.0070/0.0075/-0.0012 for the original against 1.0 for the bare
   learner and the three-off arm (REPORT:1374-1382).
5. **Round 7 localization (descriptive, post hoc, REPORT:1579-1587).** C6 restores the contrast on pair (error) turns
   exactly (net = raw). On CORRECT single-row turns, about 79 per epoch, the per-row stack still turns a raw +0.0113
   into a net -0.0351: the stabilization removes about four times the deposit. Same shape on all four seeds.
6. **Scaling is inert on A_ii inside the full stack.** max drive/ceiling is 0.42-0.45 < 1; `C6_no_homeostatic` and
   `C5_no_homeostatic` are bit-identical to their candidates (REPORT:1551, #2406). Item 2 nevertheless reports that
   scaling blocks ALONE. The ceiling is first-sight calibrated (`homeostatic_plasticity_stack.py:132-133`), so with
   E/I's trim removed the drive can reach it. This is an interaction: E/I's trim keeps the drive under the scaling
   ceiling. It is not proof that scaling is harmless.
7. **Not yet measured:** which of the three stabilizers takes the -0.0464 on correct single-row turns. Rounds 6-7 logged
   only PRE / RAW / POST per deposit (REPORT:1336-1340), not per mechanism. Reading the code gives two candidates:
   E/I's whole-row multiplicative trim (it shrinks the accumulated contrast at every deposit, `:189-191`), and
   Benna-Fusi's pull of the fast beaker toward a deep store anchored at the first-sight weights (`:226-240`). This
   is a static reading, not evidence. P2's "edit the one mechanism" needs this measurement, and stage A provides it.

### 2.3 The repair, in two stages

**Stage A -- per-mechanism loss decomposition (diagnostic; decides only which mechanism stage B edits).**
- **Instrument:** round 6's coordinate (REPORT:1325-1346), unchanged: the fixed learning-OFF panel captured before
  teaching, class-mean drives g/o, contrast b, bias q. It adds one capture point AFTER EACH stage of
  `stabilize` (`:318-329`): base scaling, base intrinsic, E/I, Benna-Fusi. Per deposit, per turn type
  (correct-single, error-pair target, error-pair loser), it records `db_stage` for every stage.
- **Seeds:** dev 8000-8003 (exposed) plus 8004-8007 (G2's further-development block, `g2:267-268`; each use disclosed).
  Canonical incumbent, E 400, A_ii.
- **Red RA-1 (bookkeeping identity):** on every deposit, the sum of the per-stage `db_stage` equals the measured
  stabilization_db within 1e-12. A failure is VOID for stage A.
- **Red RA-2 (planted stage):** a fixture stage that zeroes the row's contrast, inserted in each of the four positions
  in turn, must be attributed 100% of the loss at its position. If it is not, the instrument is blind there: VOID.
- **Red RA-3 (identity):** with the capture points present and no stage changed, the final weights are bit-identical
  to the uninstrumented original on 4/4 dev seeds.
- **Selection rule (frozen here, before data).** Over the last 50 epochs of correct-single turns, take the median per
  seed of each stage's share of the total b loss.
  - If ONE of {E/I, Benna-Fusi, scaling} has the largest share on EVERY one of the 8 dev seeds, that mechanism is
    stage B's edit site.
  - Otherwise the edit site is the composing method `PlasticityRuleStack.stabilize`: the loss is an interaction, and
    the owner's wording ("how they interact") names it.
  - Base intrinsic or BCM taking the largest share on any seed is a finding against `g2:563`. It halts at stage A and
    goes to the integrator.

**Stage B -- one frozen candidate, edited at the stage-A site.**
- It is frozen in a dated PRE-DATA addendum to this file, before any candidate run on any seed. The addendum carries
  the exact diff, the predicted sign of R on each dev seed, and an identity sham (section 3.2).
- **Constraints, each checked by the panel on the diff:**
  - one existing method is edited, and no new class, flag or parallel path is added (P2);
  - no new numeric constant, and no retuning of `max_trim`, `k`, `rho0`, `ceiling_factor`, `scale_rate`, `bf_w_max`
    or `n_levels` (P1; the refuted list's "one-constant fix" guard, section 2.4);
  - the absolute bounds are unchanged: cascade +/-4, E/I can only reduce drive, the scaling ceiling only trims;
  - local plasticity only (no gradient, no feedback matrix);
  - one-lake checkpoint continuation is equal, and a missing or corrupt state is refused, never silently migrated
    (P4b; REPORT:540-574's checks reused).
- **Candidate menu named against the code.** Stage A picks the site; the shape below is a starting point, not a freeze.
  - **K-EI** (`_EIBalance.apply`, `:177-191`): a trim that regulates the row's drive without shrinking the contrast
    already learned, e.g. applied to the common-mode component of the row. It must still pass the stack's runaway
    self-check (`_stability_selfcheck`, `:557+`), so no safety is relaxed.
  - **K-BF** (`_BennaFusiConsolidation.consolidate`, `:226-240`): deposit the plasticity event, not the other
    stabilizers' corrections, into the fast beaker. Today `weights - prev` includes the scaling and E/I corrections, so
    the cascade consolidates the trims too.
  - **K-COMP** (`PlasticityRuleStack.stabilize`, `:318-329`): change the composition, meaning the order of the
    stages or the reference each stage corrects against, without changing any stage's math. This is the "interaction"
    edit the owner named. Round 7's H7 (REPORT:1589-1594) is a K-COMP shape for single-row turns only if it stays inside
    `stabilize`. C6 added a second method `stabilize_pair` beside `stabilize` and a learner-side join, which is a P2
    hazard (two stabilize paths). H7 inherits that hazard unless it is re-expressed as one path.

### 2.4 Re-proposal check (memory `project-snn-refuted-and-negatives.md`, and the six A_ii nulls)

The refuted-list file has no A_ii entry; the six nulls live in the row REPORT and in handoff-1014:35-49 (gap: section 6,
item O-9). Each candidate must state why it differs:

| Refuted | What it changed | Why stage B differs, or the guard it must pass |
|---|---|---|
| C1/C2 (#2252) | a shared clock dt on all three stabilizers | stage B edits ONE site, chosen by measurement; it has no clock constant. A clock rescaling is excluded outright |
| C3 (#2313) | a lagged baseline for the scaling ceiling | scaling is inert on A_ii in the full stack (2.2 item 6), so a scaling-ceiling edit is admitted ONLY if stage A gives scaling the largest share on every seed |
| C4 (#2367) | cascade coordinates, scaling every beaker, E/I as a readout gain (several changes at once) | K-EI and K-BF each touch one stage; C4's E/I part was never measured alone, so a K-EI that resembles it must state the difference in the addendum and carry its own sham |
| C5 (#2406) | a dual fast/slow store with a new state | no new persistent state is admitted unless P4b checkpoint equality and refusal are shown first. A second store for one job is excluded (P2) |
| C6 (#2520) | `stabilize_pair` beside `stabilize`, plus a learner-side join | excluded as shaped: two stabilize paths (P2). It fixed only pair turns, and stage A targets the single-row loss it could not reach |
| norm60 (round 6) | a fixed-norm projection | a projection is a new mechanism with a constant (60); excluded |
| refuted file, "accuracy lever is scale" (v3.43-v3.48) | -- | not the lever here: the bare learner at the SAME scale solves A_ii (2.2 item 2). No capacity change is proposed |
| refuted file, "do not re-propose a one-constant fix" (JEPA entry) | -- | no constant may move (stage B constraints) |

## 3. Endpoint, controls and reds

### 3.1 Endpoint: G2's registered reading, unchanged

- **A_ii** (`g2:99-102`): (ii)'s word-order construction on `ram` 15, `cup` 16, `emu` 20, `cod` 26, intents GREET / GOAL.
- **Solved** (`g2:174-178`; `taught_problems.py:186-200`): every probe of the validation read is answered correctly
  through the real read (16 presentations). **Stable** = `solved` at each of the last STABLE = 10 epoch-end
  validation reads (`taught_problems.py:49`, `:197`) of an E = 400 attempt (`g2:529`, 2.17 R2), through
  `taught_campaign.attempt` (`taught_campaign.py:211+`) and `make_reader` (`:165`), with the canonical incumbent
  (`taught_campaign.py:61-63`).
- **The per-seed predicate** (G2 2.9, `g2:225-231`, B5's form): the canonical incumbent solves A_ii on EVERY seed.
  One failure is a counterexample, not a statistic (A3 / A4's rule).
- **Per-seed conjunction for the repaired stack (MET needs all of them on a seed):**
  - **E1:** A_ii stable-10 at E 400 (above).
  - **E2 (no regression, paired):** each of G2's 12 (i) arm-cells and A_c1d reads the same `met` as the original stack
    on the same seed, or better where "better" is the registered expectation (`g2:640-642`). A pre-existing original
    failure is not charged to the repair (round 7 found A_c1d's strict conjunct failing for the original on 8000/8001,
    REPORT:1552).
  - **E3 (each stabilizer stays live; no safety relaxation):**
    - E/I and Benna-Fusi: own-neuter on the repaired stack changes the final weight energy by more than 1e-12 on A_ii
      on every seed (REPORT:87-106's statistic).
    - Scaling: the original stack is itself inert on A_ii (2.2 item 6), so the A_ii own-neuter cannot read it. Scaling
      is read on its designed input instead: the stack's runaway self-check (`_stability_selfcheck`, `:557+`) and the
      scaling-active A->B->rest fixture (REPORT:1550), with a nonzero scaling effect no smaller than the original's.
      **This substitution is a session call, OWED a ruling (section 6, O-4).** Rounds 2-7 asked for scaling's A_ii
      own-neuter, a conjunct the original stack fails by construction.
  - **E4:** every `plasticity_rule_stack` and `homeostatic_plasticity_stack` self-check stays green, unchanged in text
    (the retention, adaptive-lr, stability, equivalence and tick-charge self-checks, `:464-790`). This includes the rate
    homeostasis that C5 broke (#2406).
- **Secondary (decides nothing):** round 6's R per seed, predicted > 0 and > R_original; future-split accuracy; epochs
  to the first stable window.

### 3.2 Controls (P8 "What an invented method owes", item 2: not the method)

| Control | Expected | Read on |
|---|---|---|
| original stack, same seeds (paired) | fails E1 on every seed | dev and confirmatory |
| bare learner (`im._incumbent`) | solves E1 (bare 4/4, first windows 108-131, REPORT:1384-1385) | dev and confirmatory |
| three-off arm (E/I, BF, scaling off) | solves E1 (4/4, REPORT:1384-1385) | dev and confirmatory |
| identity sham: the repaired code path present, its edit neutered | final weights and reads bit-identical to the original on every seed | dev and confirmatory |
| B8 twin with the repaired stack (`taught_campaign.py:343+`) | Brain learner == factory learner at every epoch end | confirmatory, before the arms |

### 3.3 Reds (each must fire before the confirmatory arms run; a red that does not fire halts the campaign)

- **RD-1 (non-learner, lr 0):** the repaired incumbent with lr 0 does NOT solve A_ii (B5's red, `g2:211`; "not solved",
  never "unchanged", because `stabilize` can still move weights).
- **RD-2 (non-learner, shuffled labels):** the repaired incumbent taught `_shuffled` labels does NOT solve A_ii.
- **RD-3 (a repair that cannot fail):** the three-off arm, PRESENTED AS a candidate, must be REJECTED by the
  per-seed conjunction. It passes E1 and must fail E3. If the conjunction accepts it, the bar rewards switching safety
  off: VOID.
- **RD-4 (reachable E1 red):** the original stack fails E1 on the confirmatory seeds. If it solves on any seed, the bar
  cannot separate the repair from no repair on that block: VOID.
- **RD-5 (harness sensitivity):** the bare learner solves E1 on the confirmatory seeds. If it fails, the harness or the
  block cannot show learning: VOID.
- **RD-6 (identity sham):** bit-identity to the original. A mismatch means the edit leaks outside its neuter: VOID.

## 4. Sample size (P1)

| Field | Value |
|---|---|
| Endpoint type | binary per seed: the E1-E4 conjunction holds or not. MET = every confirmatory seed (G2's registered all-seeds reading) |
| Variance source | the binary endpoint's variance is p(1-p); the measured rates are original 0/4 dev + 0/8 confirmatory (`g2:534`, `:640`; one-sided 95% upper bound 0.221 on 0/12), bare 4/4 and three-off 4/4 dev (REPORT:1376-1385). Continuous secondary: bare first-stable epoch SD 10.6 (n 4), three-off 17.4 (n 4) |
| Claimed effect (derived, not habitual) | G2's (ii) campaign needs the incumbent to solve A_ii on all S = 8 seeds (`g2:225-231`). For that campaign to pass with probability 0.8, the per-seed rate must be p1 = 0.8^(1/8) = **0.9725** |
| Test | exact one-sided test of H0: p <= p0 against p >= p1; reject H0 (MET) iff all n confirmatory seeds pass |
| Fixed-sample target | the largest n with p1^n >= 0.8 is **n = 8**; at n = 8, alpha = 0.05 fixes **p0 = 0.05^(1/8) = 0.688**. So MET at 8/8 rejects "the repaired brain solves at a rate <= 0.69" at one-sided .05 with power 0.80 at p1 |
| Multiplicity | none needed: E1-E4 are an intersection-union conjunction per seed (every part must hold), so the test is level alpha without correction |
| Sequential shape (evaluated, then rejected by its own arithmetic) | looks at 4 and 8 under Lan-DeMets O'Brien-Fleming: alpha spent at t = 0.5 is 0.0056, while an all-4 pass has P = 0.224 under p0, so look 1 can never stop for efficacy. A futility stop at look 1 would not read powered: a failure after 0-3 passes leaves the one-sided 95% upper bound on p at 0.950-0.987, above or near p1, so P1 item 2 forbids taking it. The looks add nothing |
| Design | a **fixed one-shot n = 8** (P1 item 3), power 0.80 stated here, before data. Inflation 1.0, hard cap 8, no futility bound. A disclosed session call, OWED the policy-enforcer's AGREE (section 6, O-5) |
| Simulated before data (stdlib, 200,000 reps per p, this lane) | P(MET): p 0.50 -> 0.0039; p0 0.688 -> 0.0505; p 0.90 -> 0.429; p1 0.9725 -> 0.800; p 0.99 -> 0.923; p 1.0 -> 1.000. The exact p^8 values agree to 4 decimals. The freeze re-runs this as a committed script (O-6) |
| Not counted | stage A and the dev reads (8000-8007): exposed, development only |

**What a non-MET reads as (P8 null-scale item 4):** with x passes in 8, the NULL-RECORD states the exact one-sided 95%
upper bound on p. NOT MET is **powered** only when that bound is below p1 = 0.9725. One failure (7/8) leaves the bound at
0.994, so it is NOT MET for G2's reading but INCONCLUSIVE as a null on the mechanism. Two or more failures read as powered.

## 5. Disposition (fixed in advance; first row that applies decides)

| Reading | When | Consequence |
|---|---|---|
| **VOID** (validity only) | RA-1, RA-2 or RA-3 fails (stage A); or any of RD-1 to RD-6 or the B8 twin fails (confirmatory); or the block is exposed before the run | no verdict. Before any confirmatory run is exposed: repair by a dated addendum on the same seeds. After: a new pre-registration on a fresh block the integrator claims |
| **MET** | the E1-E4 conjunction holds on all 8 confirmatory seeds | the candidate is the measured win. It may land through the normal arc (3-lens panel on the build, the battery, the push), and only then do G2(ii), G3 C11 on S0C1 and G9 unpark under their own registrations |
| **NOT MET, powered** | two or more seeds fail (upper bound < p1) | the candidate is rejected evidence and never lands. A NULL-RECORD with scale: units = the 9 response neurons x H inputs, ticks = E 400 epochs, seeds 8, the mde (P8 null-scale item 1). It was never on the live path, so nothing is deleted or exempted |
| **NOT MET for G2, INCONCLUSIVE as a null** | exactly one seed fails | it does not land, since there is no measured win. The NULL-RECORD says INCONCLUSIVE (P8 item 4); a further run needs its own pre-registration on a fresh block |

The live brain stays byte-identical under every reading except MET, and under MET until the arc lands.

## 6. Owed before freeze

- **O-1 Seeds.** The integrator assigns and claims the confirmatory block on the board. Proposed: 8100-8107 (G2's
  "8100-8199 held, unassigned", `g2:268`); stage A's 8004-8007 from G2's further-development block. 8200-8207 is
  exposed for A_ii (5.1) and is never reused.
- **O-2 Held paths.** Confirm whether `readout/interaction_learning.py` and `plasticity/plasticity_rule_stack.py` are
  held by another arc (C4-C6 needed held-path patches for G9). Stage A adds capture points inside `stabilize`, and
  stage B edits one of the cited methods.
- **O-3 Instrument location.** Stage A's per-stage capture reuses round 6's instrument. graph-exists must be run for it
  under `src/` before any build. The round-6 script lives outside the repo, and P4c says it is not a build until it
  lands.
- **O-4 E3 scaling substitution** (section 3.1). The owner or integrator rules whether scaling's liveness is read on
  its designed input, not on A_ii where the ORIGINAL stack is already inert. This is the one place the draft departs
  from rounds 2-7's bar. It relaxes no safety bound: the runaway self-check and the scaling-active fixture still bind.
- **O-5 P1 design.** The policy-enforcer reviews the fixed one-shot n = 8 derivation (section 4), and whether G2's
  all-seeds reading may set the claimed effect.
- **O-6 Simulation script.** Commit section 4's stdlib simulation with a planted-null and a planted-effect red, before
  data.
- **O-7 Perception tier.** The confirmatory record states `state()['perception_tier']` and `perceiving_audio` for the B8
  twin brain. A 'stdlib' tier halts.
- **O-8 G2 F7(a).** A_ii's A1 (generator check) reading on dev seeds 8000-8003 (`g2:681` on `4b6bc9ea`). It is G2's to
  write, and it precedes the confirmatory read: a generator that fails A1 voids A_ii as a problem.
- **O-9 Records gap.** The six A_ii repair nulls (C1-C6 and norm60) are absent from `project-snn-refuted-and-negatives.md`,
  and their NULL-RECORDs are an open owner question (G2 Q3; memory `project-snn-owner-e1b-production-tier-g2-twin-1023`
  :25). This draft neither writes nor re-tests them.
- **O-10 Review.** The 3-lens panel and the policy-enforcer review this text. Their challenges are folded, and the
  freeze follows their AGREEs.
