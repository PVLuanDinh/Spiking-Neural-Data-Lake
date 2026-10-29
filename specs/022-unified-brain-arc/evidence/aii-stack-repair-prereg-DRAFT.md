# A_ii STACK REPAIR -- make the CANONICAL brain learn A_ii (DRAFT pre-registration)

> **Status: DRAFT, pre-data, NOT frozen.** Lane aii-prereg (thread AII-PREREG), 2026-10-29, written on
> `claude/aii-prereg-1029` off `c4c10b28`. Docs only: no brain, no battery, no run of any kind went into this file.
> Every number below is quoted from a cited record; none was measured here. The only arithmetic computed here is the
> sample-size design in section 4 (stdlib python, exact binomial lattice plus a Monte Carlo check, no brain).
> Fold 1 (2026-10-29, the end of this file) folds the 2-lens panel (#6792, #6831) and the policy-enforcer (#6817).
> - **Why it exists:** owner #2135 (2026-10-13, answering Q-AII #2134), verbatim selection: *"Fix the stack first"*.
>   Ruling as relayed: G2(ii) and A_ii stay parked; *"first build and measure a repair of how E/I balance, Benna-Fusi
>   consolidation and homeostatic scaling interact so the CANONICAL brain learns A_ii (neuter A/B, reachable red, no
>   safety relaxation); then run (ii) on the repaired brain. G2(ii), G3 C11 on S0C1 and G9 wait on that arc."*
>   The dated note in `g2-taught-problems.md:647-652` (at `c4c10b28`) records the same answer.
> - **Later owner rulings that bind this arc** (handoff-1014:36, :45-49; memory `project-snn-session-handoff-1014.md`):
>   the path ran #2135 -> #2316 -> #2369 -> #2410 -> #2527 -> #2564/#2565. #2527 (verbatim): *"A_ii returns only on a
>   stack that retains"*. #2565 (verbatim selection *"Bouts scale with labels"*) closed retention as "retains within a
>   label-scaled night" (bouts = 7 x label rows) and recorded the A/B shared-weight conflict as the stack's real
>   trade-off; A_ii's blocker is LEARNING. Section 3.1's E5 carries #2527 into the bar.
> - **The gap it closes:** G2 2.18 F7 (on `claude/g2-coverage-1029@4b6bc9ea`, `g2-taught-problems.md:670` and `:681`):
>   A_ii's A4 reading FAILS its registered requirement and "freezes only after that repair". Section 4 below is the
>   REPAIR ARC's own Sample size (P1) section. It is NOT A_ii's or G2(ii)'s P1 section: D-67 (`4b6bc9ea g2:702`) leaves
>   each of those owing its own section in G2 (2.18.2 row P1, `:684`), and they stay owed (section 6, O-11).
> - **Freezes when:** every item in section 6 is closed, the 2-lens panel (lenses A and C) and the policy-enforcer
>   AGREE, and the freezing commit drops "DRAFT" from the title.

## 0. Raising-ambition record (self-apply)

- **Axis:** Depth. The question is fixed by the owner (make the canonical brain learn A_ii without relaxing safety).
  Six candidate rounds failed at it by guessing a mechanism and measuring it. A top practitioner localizes the loss
  first, then edits the one part that causes it.
- **Ladder.**
  - **R1:** freeze one more candidate (round 7's unbuilt H7, extending C6's shared reference to single-row turns) and
    measure it against the old conjunction. This is what rounds 1-7 did, to six candidate nulls.
  - **R2:** first measure, per stabilizer and per turn type, where the taught contrast is lost. The measurement carries
    a planted-stage red. A rule written here, before data, then picks ONE of four candidate diffs that are all frozen
    before stage A runs. The candidate is confirmed on a fresh seed block under a P1-derived sequential design.
  - **R3:** build a reduced model of the stack's per-deposit map that predicts each candidate's contrast ratio R and its
    stable-10 outcome before any brain run. Backtest it on the six recorded nulls C1-C6, then build only the candidate
    it predicts will pass.
- **Pick: R2.**
  - R3 is ruled out by **P2**: a model that predicts `stabilize`'s output re-implements `stabilize`'s dynamics, a second
    mechanism for one job. Its output would also be load-bearing (it chooses the repair), so P8 "What an invented
    method owes" would require its own red, control and attribution. That is a separate arc, not this one.
- **Range.**
  - Control engineering: when several regulators act on one variable, open each loop separately and measure what each
    one removes before retuning any of them (section 2.3, stage A, including the on/off-subset attribution).
  - Adaptive clinical trials (seamless phase II/III): the selection stage freezes its selection rule AND its candidate
    arms before data, and its patients never count toward the confirmatory analysis (stage A and the dev seeds never
    count, section 4).
  - Adversarial auditing: construct the "repair" that passes A_ii by switching safety off (the three-off arm solves
    A_ii, `g2:558-564`), or by turning it down (the attenuated arm), and require the bar to reject both (RD-3, RD-3b).
- **Outside judge** (none of these is the method itself):
  - G2's registered `solved` / stable-10 reading through `taught_campaign.attempt` (`g2:174-178`), written before this arc;
  - the original stack's reproduced failure on the same seeds;
  - the bare learner and the three-off arm, which do learn A_ii through the same harness;
  - B5's constructed non-learners (lr 0, shuffled labels; `g2:211`);
  - the retention instrument of #2565 (`rows/retention-2026-10-14`), proven on retainers 1.0 / overwriters 0.0;
  - the stack module's own self-checks, unchanged;
  - the 2-lens panel and the policy-enforcer.

## 1. What is claimed, and what is not

- **Claim (if MET):** a single edit to one existing mechanism of `PlasticityRuleStack` raises the canonical incumbent's
  per-seed rate of solving A_ii by G2's registered reading (with E2-E5 holding on the same seed) ABOVE the original
  stack's measured rate: it rejects H0 "rate <= 0.221" (the exact one-sided 95% upper bound on the original's 0/12) at
  one-sided alpha 0.05. The incumbent is `taught_problems.canonical_incumbent` (`taught_problems.py:386-398`: the class
  the live brain builds plus the stack `Brain.__init__` attaches, `engine/brain.py:375-380`; B8 proves them equal,
  `taught_campaign.py:343-360`). E/I balance, Benna-Fusi consolidation and homeostatic scaling all stay live, and every
  existing safety bound is unchanged.
- **Not claimed:** that the repaired brain is reliable enough to pass G2(ii)'s all-seeds reading. That reading needs a
  per-seed rate near 1 (0.9725 for an 0.8 chance of 8/8), and no affordable confirmatory n can certify it (section 4,
  "Claim choice"). G2(ii) is the test of that, under its own registration. Also not claimed: anything about G3 C11 or
  G9; recognition quality on live chat; any change to the five `brainstem/` envelope files (not touched, not needed).
- **Lands only on a measured win.** The repair changes what the live brain learns (brain.py:353-363). Only MET licenses
  a landing, and then only through the normal arc: panel, battery, push, after the ruling O-12. Every other disposition
  leaves the live brain byte-identical.

## 2. The stack, what is broken, and the candidate repair

### 2.1 What "the stack" is (code at `c4c10b28`)

`SpikingInteractionLearner._deposit` (`readout/interaction_learning.py:512-565`) runs one R-STDP deposit on ONE
response neuron's input-weight row: `pre_gain` (PRE, `:535`), the reward-gated eligibility delta `n.reward` (`:538`),
then `stack.stabilize` on that whole row (POST, `:558-560`). `interact` (`:628-650`) deposits +1 on the taught target
on EVERY teaching turn (`:647`), and -1 on the learner's own wrong answer only when `pred != tgt` (`:648-649`). A
correct turn is therefore a single-row deposit; an error turn is a pair.

`PlasticityRuleStack.stabilize` (`plasticity/plasticity_rule_stack.py:318-329`) composes, in this fixed order:

| Order | Mechanism | Code | What it does to the row |
|---|---|---|---|
| 1 | base: homeostatic scaling | `homeostatic_plasticity_stack.py:106-139` (`_HomeostaticScaling`) | a soft ceiling. The set-point is calibrated ONCE to `ceiling_factor` (3.0) x the first-sight drive (`:132-133`); scaling acts only when the drive exceeds it (`:135-138`), multiplicatively |
| 1b | base: intrinsic plasticity | `homeostatic_plasticity_stack.py:142-166` | tunes the cell's gain and bias; returns the weights UNCHANGED (`:162`). Not in the owner's three; does not block A_ii (`g2:563`) |
| 2 | E/I balance | `plasticity_rule_stack.py:150-191` (`_EIBalance`) | an iSTDP controller; a multiplicative trim `f = 1 - min(max_trim, k*inhibition)` on the WHOLE row at every deposit (`:187-191`) |
| 3 | Benna-Fusi consolidation | `plasticity_rule_stack.py:194-256` (`_BennaFusiConsolidation`) | deposits `weights - prev` into each synapse's fast beaker, diffuses one step toward the deep beakers, and returns the fast beaker (`:226-240`). The deep beakers start at the weights of the FIRST `consolidate` call (`:217-224`), which are already post-deposit and post-trim, not first-sight weights |
| 4 | BCM metaplastic gate | `plasticity_rule_stack.py:87-148` | only observes the drive (`:326-327`) for the NEXT `pre_gain`; does not block A_ii (`g2:563`) |

The constructor parameters are `ei_max_trim`, `ei_k`, `ei_rho0`, `bf_n_levels` (and the base's `ceiling_factor`,
`scale_rate`, `bf_w_max`). graph-exists (run in this lane, `--root` this worktree) finds exactly these as the live
implementations: `_EIBalance` (also `inhibitory_stdp.py`, the reused controller), `_BennaFusiConsolidation` (also
`benna_fusi.py`, the reused synapse), and `_HomeostaticScaling` (also `homeostatic_scaling.py`, the reused neuron). A
repair edits one of these, or the composing method `stabilize`. It never adds a parallel stabilizer (P2).

### 2.2 What is broken (each claim cited; all dev seeds 8000-8003, E 400, unless named)

1. **The canonical incumbent never stable-solves A_ii.** It fails on dev seeds 8000-8003 (`g2:534`; per-epoch reads
   `g2:566-571`: mean accuracy 0.491-0.497, an isolated 1.0 read on 3 seeds) and on G2 5.1's confirmatory block
   8200-8207 (`g2:638` `a_fails` true; `g2:640-645` `inc.aii 0/8`).
2. **The construction is learnable; the stack blocks it.** The bare learner through the same harness and criterion
   stabilises A_ii at epochs 118/133/140/141 (`g2:560-561`). E/I balance, Benna-Fusi and homeostatic scaling EACH
   block alone (only-one-on); no single switch-off or pair switch-off restores learning; all three off restores it
   (174/156/179/141). The metaplastic gate and intrinsic plasticity do not block (`g2:561-564`).
3. **Six pre-registered candidate nulls, C1-C6** (handoff-1014:37; row `aii-stack-repair-2026-10-13`, `REPORT.md` at
   `C:/Users/vinhl/.config/opencode/sndl-runs/2026-10-09-step5/rows/aii-stack-repair-2026-10-13/`; board results):
   - C1/C2: a shared clock dt on all three stabilizers (REPORT:38-53). Stable-10 0/4. Both fail the old retention
     self-check (#2252).
   - C3: C1's clock plus a lagged absolute-drive baseline for the scaling ceiling (REPORT:301-310). First stable-10
     windows, but tail/future 0/4 (#2313).
   - C4: cascade coordinates, with E/I as a fresh readout gain (REPORT:524-535). First/tail 0/4 (#2367).
   - C5: a dual fast/slow store (REPORT:824-857). 0/4 (#2406).
   - C6 (2026-10-14): a joint target/loser pair stabilization (REPORT:1532-1595, #2520).
   - Plus one diagnostic arm, not a candidate: round 6's norm60 projection (REPORT:1372-1384).
4. **Round 6 (H6 SUPPORTED 4/4, descriptive, REPORT:1293-1444).** The raw teaching deposit moves the task contrast
   b = (g - o)/2 the useful way on all 28 arms. In the original stack, the post-deposit stabilization nearly cancels it:
   raw gain about +0.00094-0.00098 per epoch, post-stack contribution about -0.00094 to -0.00097 (REPORT:1400-1404).
   The primary R = net/raw contrast ratio is 0.0016/0.0070/0.0075/-0.0012 for the original against 1.0 for the bare
   learner and the three-off arm (REPORT:1374-1382).
5. **Round 7 localization (descriptive, post hoc, REPORT:1579-1587).** C6 restores the contrast on pair (error) turns
   exactly (net = raw). On CORRECT single-row turns, about 79 per epoch, the per-row stack still turns a raw +0.0113
   into a net -0.0351: the stabilization removes about four times the deposit. Same shape on all four seeds.
6. **Scaling is inert on A_ii inside the full stack.** max drive/ceiling is 0.43-0.46 for the original stack (0.42-0.45
   for C6; REPORT:1577) < 1; `C6_no_homeostatic` and `C5_no_homeostatic` are bit-identical to their candidates
   (REPORT:1551, #2406). Item 2 nevertheless reports that scaling blocks ALONE. The ceiling is first-sight calibrated
   (`homeostatic_plasticity_stack.py:132-133`), so with E/I's trim removed the drive can reach it. This is an
   interaction: E/I's trim keeps the drive under the scaling ceiling. It is not proof that scaling is harmless.
7. **Not yet measured:** which stabilizer takes the -0.0464 on correct single-row turns. Rounds 6-7 logged only
   PRE / RAW / POST per deposit (REPORT:1336-1340), not per mechanism. Reading the code gives two candidates: E/I's
   whole-row multiplicative trim (it shrinks the accumulated contrast at every deposit, `:187-191`), and Benna-Fusi's
   pull of the fast beaker toward a deep store anchored at the first `consolidate` call's weights (`:217-240`). This is
   a static reading, not evidence. P2's "edit the one mechanism" needs this measurement, and stage A provides it.

### 2.3 The repair, in two stages

**Addendum 0 (before ANY stage-A run): the four candidate diffs, frozen.** Stage A's data only SELECTS among four
candidates; it never shapes one. Before the first stage-A run, a dated pre-data addendum to this file commits the
exact diff of each of the four candidates below, with the predicted sign of R on each dev seed and an identity sham
for each. The shapes are fixed here with no free parameter; the addendum may only write them as code.
- **K-EI** (`_EIBalance.apply`, `:177-191`): the trim removes the SAME drive as today (the row's drive after the trim
  equals `f * drive`, exactly), but as the minimum-norm change along the row's activity vector,
  `w_k <- w_k - (1 - f) * drive * a_k / sum_j a_j^2`, instead of scaling every weight by `f`. The drive bound and
  "E/I only reduces drive" are unchanged; the per-synapse differences the deposit wrote are not scaled away.
- **K-HS** (`_HomeostaticScaling.stabilize`, `:125-139`): the same substitution for the scaling step: when `y > target`,
  the drive is brought to the value the reused `HomeostaticNeuron.stabilize` produces today, by the minimum-norm change
  along the activity vector, instead of the multiplicative rescale. Admitted only under C3's guard (section 2.4).
- **K-BF** (`_BennaFusiConsolidation.consolidate`, `:226-240`): the fast beaker receives only the R-STDP event, i.e.
  the row as it ENTERED `stabilize` minus the reference `prev`, not the base and E/I corrections; the corrections are
  applied to the returned weights and carried into `prev`. The addendum writes this as one diff inside `consolidate`
  and `stabilize`'s call to it (the row at entry is passed, no new state).
- **K-COMP** (`PlasticityRuleStack.stabilize`, `:318-329`): one reordering, no stage's math changed: Benna-Fusi
  consolidates the deposited row FIRST, then the base sandwich, then E/I. Round 7's H7 (REPORT:1589-1594) is NOT this
  candidate: C6 added a second method `stabilize_pair` and a learner-side join, a P2 hazard (two stabilize paths).

**Stage A -- per-mechanism loss decomposition (diagnostic; decides only which frozen candidate stage B confirms).**
- **Instrument:** round 6's coordinate (REPORT:1325-1346), unchanged: the fixed learning-OFF panel captured before
  teaching, class-mean drives g/o, contrast b, bias q. It adds one POST capture point after each stage of `stabilize`
  (`:318-329`): base scaling, base intrinsic, E/I, Benna-Fusi. Per deposit and per turn type (correct-single,
  error-pair target, error-pair loser) it records `db_stage` for every stage. The capture is POST-only: intrinsic
  plasticity returns the weights unchanged (`homeostatic_plasticity_stack.py:162`) and BCM only observes
  (`plasticity_rule_stack.py:326-327`), so their POST `db_stage` is 0 by construction. Their PRE effect (and short-term
  gain's, `homeostatic_plasticity_stack.py:213`) acts through the NEXT `pre_gain`; stage A records `pre_gain(li)` per
  deposit descriptively. `g2:563`'s "does not block" rests on the switch-off evidence, which stands.
- **On/off-subset attribution (order-free).** Because a share inside a sequential composition depends on stage order
  (BF consolidates `weights - prev`, which includes E/I's trim), stage A also runs the 8 on/off subsets of
  {E/I, BF, scaling} (the bare learner is the empty subset, the original the full one) on the same seeds. For each
  subset S, V(S) = 1 - R_single(S), where R_single = sum of net db / sum of raw db over correct-single deposits in the
  last 50 epochs. The Shapley share of mechanism m is phi_m / sum over the three of max(phi, 0).
- **Seeds:** dev 8000-8003 (exposed) plus 8004-8007 (G2's further-development block, `g2:267-268`; each use disclosed).
  Canonical incumbent, E 400, A_ii.
- **RA-1 (logger consistency; NOT a red):** on every deposit, the sum of the per-stage `db_stage` equals round 6's
  PRE/RAW/POST logger's stabilization_db within 1e-12. Each `db_stage` is a difference of consecutive captures, so the
  sum telescopes; RA-1 only checks the new captures against the old logger's code path. A failure is VOID for stage A.
- **Red RA-2 (planted stage; the red that can fire):** a fixture stage that zeroes the row's contrast, inserted in each
  of the four positions in turn, must be attributed 100% of the loss at its position. If it is not, the instrument is
  blind there: VOID.
- **Red RA-3 (identity):** with the capture points present and no stage changed, the final weights are bit-identical
  to the uninstrumented original on 8/8 stage-A seeds.
- **Selection rule (frozen here, before data).** Per seed, over the correct-single deposits of the last 50 epochs:
  - L_s = -(sum over those deposits of `db_stage` for stage s), in contrast units (a sum, not a median).
  - A stage with L_s <= 0 (it gains contrast or is neutral) gets sequential share 0 and is reported by value.
  - Sequential share_s = L_s / (sum of L_t over stages with L_t > 0).
  - **Near-zero / contradiction guard:** if V(full) <= 1 on any seed (the stack removes no more than the raw deposit,
    against round 7's ~4x), or the sum of raw db over those deposits is <= 0, stage A contradicts round 7: halt, to the
    integrator. No candidate is selected.
  - **Pick:** mechanism m in {E/I, BF, scaling} is the site iff BOTH its sequential share AND its Shapley share exceed
    0.5 on EVERY one of the 8 seeds (a strict majority, so two mechanisms can never tie). E/I -> K-EI, BF -> K-BF,
    scaling -> K-HS.
  - **Otherwise** -> K-COMP: the loss is an interaction, and the owner's wording ("how they interact") names it.

**Stage B -- the one selected candidate, confirmed.** Exactly ONE candidate (the stage-A pick, already frozen in
Addendum 0) is confirmed under this registration (k = 1; section 4, "Arc budget").
- **Constraints, each checked by the panel on the diff:**
  - one existing method is edited, and no new class, flag or parallel path is added (P2);
  - no new numeric constant, and no retuning of `ei_max_trim`, `ei_k`, `ei_rho0`, `ceiling_factor`, `scale_rate`,
    `bf_w_max` or `bf_n_levels` (P1; the refuted list's "one-constant fix" guard, section 2.4);
  - the absolute bounds are unchanged: cascade +/-4, E/I can only reduce drive, the scaling ceiling only trims;
  - local plasticity only (no gradient, no feedback matrix);
  - one-lake checkpoint continuation is equal, and a missing or corrupt state is refused, never silently migrated
    (P4b; REPORT:540-574's checks reused).

### 2.4 Re-proposal check (memory `project-snn-refuted-and-negatives.md`, and the six A_ii nulls)

The refuted-list file has no A_ii entry; the six candidate nulls live in the row REPORT and in handoff-1014:35-49 (gap:
section 6, item O-9). Each candidate must state why it differs:

| Refuted | What it changed | Why stage B differs, or the guard it must pass |
|---|---|---|
| C1/C2 (#2252) | a shared clock dt on all three stabilizers | stage B edits ONE site, chosen by measurement; it has no clock constant. A clock rescaling is excluded outright |
| C3 (#2313) | a lagged baseline for the scaling ceiling | scaling is inert on A_ii in the full stack (2.2 item 6), so K-HS is admitted ONLY if stage A gives scaling the majority share on every seed. K-HS changes the trim's direction, never the ceiling or its baseline |
| C4 (#2367) | cascade coordinates, scaling every beaker, E/I as a readout gain (several changes at once) | each frozen candidate touches one stage; C4's E/I part was never measured alone. K-EI keeps E/I a weight trim (not a readout gain) and changes only its direction; its sham is in Addendum 0 |
| C5 (#2406) | a dual fast/slow store with a new state | no new persistent state is admitted (K-BF adds none) unless P4b checkpoint equality and refusal are shown first. A second store for one job is excluded (P2) |
| C6 (#2520) | `stabilize_pair` beside `stabilize`, plus a learner-side join | excluded as shaped: two stabilize paths (P2). It fixed only pair turns, and stage A targets the single-row loss it could not reach |
| norm60 (round 6 diagnostic arm, not a candidate) | a fixed-norm projection | a projection is a new mechanism with a constant (60); excluded |
| refuted file, "accuracy lever is scale" (v3.43-v3.48, `:40`) | -- | not the lever here: the bare learner at the SAME scale solves A_ii (2.2 item 2). No capacity change is proposed |
| refuted file, "do not re-propose a one-constant fix" (JEPA entry, `:282`) | -- | no constant may move (stage B constraints) |

## 3. Endpoint, controls and reds

### 3.1 Endpoint: G2's registered reading, unchanged

- **A_ii** (`g2:99-102`): (ii)'s word-order construction on `ram` 15, `cup` 16, `emu` 20, `cod` 26, intents GREET / GOAL.
- **Solved** (`g2:174-178`; `taught_problems.py:186-200`): every probe of the validation read is answered correctly
  through the real read (16 presentations). **Stable** = `solved` at each of the last STABLE = 10 epoch-end
  validation reads (`taught_problems.py:49`, `:197`) of an E = 400 attempt (`g2:529`, 2.17 R2), through
  `taught_campaign.attempt` (`taught_campaign.py:211+`) and `make_reader` (`:165`), with the canonical incumbent
  (`taught_campaign.py:63-65`).
- **A seed PASSES** iff the conjunction E1-E5 below holds on it. The confirmatory statistic is the count of passing
  seeds (section 4). G2's own per-seed predicate (G2 2.9, `g2:225-231`: one failure is a counterexample) is G2(ii)'s
  reading, not this arc's; this arc reports the count of E1 passes beside it, descriptively.
- **Per-seed conjunction for the repaired stack:**
  - **E1:** A_ii stable-10 at E 400 (above).
  - **E2 (no regression, paired):** the arms of G2 5.1 (`g2:640-642`) whose learner carries the stack read the same
    `met` as the original stack on the same seed: `inc.S0C0`, `inc.S1C0`, `inc.S0C1`, `inc.S1C1`, `inc.ac1d`. The
    non-stack arms (`hist.S0C0`, `hist.S1C0`, `hist.S1C1`, `pre.S0C1`, `pre.S1C1h`, `pre.S1C0`) are read as harness
    controls (same `met` as the original). `pre.c1d` (0/8 in 5.1, a presized control the stack does not drive) and
    `inc.c1d` (C1-d's REPRESENTABLE, NOT SHOWN LEARNABLE reading, owner O11) are reported, not read for regression. A
    pre-existing original failure is not charged to the repair (round 7: A_c1d's strict conjunct fails for the
    original on 8000/8001, REPORT:1552).
  - **E3 (each stabilizer stays live at full strength; no safety relaxation).** Liveness is read on each stabilizer's
    DESIGNED input, not on A_ii (where the original stack's scaling is already inert, 2.2 item 6): on the stack's
    runaway self-check fixture (`_stability_selfcheck`, `:557+`), each of E/I, BF and scaling, own-neutered on the
    repaired stack, changes the fixture's bounded quantity by NO LESS than the same own-neuter changes it on the
    original stack. On A_ii, each own-neuter's weight-energy effect (REPORT:87-106's statistic) is reported per seed,
    beside the original's (round 7: E/I +15.7 to +20.8, BF -0.92 to -1.21, REPORT:1550), and decides nothing.
    **This substitution is a session call, OWED a ruling (O-4).** Rounds 2-7 asked for each own-neuter on A_ii, a
    conjunct the original stack fails by construction for scaling.
  - **E4:** every `plasticity_rule_stack` and `homeostatic_plasticity_stack` self-check stays green, unchanged in text
    (the retention, adaptive-lr, stability, equivalence and tick-charge self-checks, `:464-790`). This includes the rate
    homeostasis that C5 broke (#2406).
  - **E5 (retains; owner #2527, #2565):** the repaired stack retains A within a label-scaled night (bouts = 7 x label
    rows), read with #2565's instrument (`rows/retention-2026-10-14`, the redesigned fixture: both tasks scored,
    relax-capable negatives, the B-loss red N5; handoff-1014:45-47), on every confirmatory seed, paired against the
    original stack, which retains there (#2565). The old A->B->rest-400 fixture (REPORT:1550) is NOT used: the retention
    arc found its rest-400 reading a knife edge (A recovers at ~402). Its scaling-effect number is reported only.
- **Secondary (decides nothing):** round 6's R per seed, predicted > 0 and > R_original; future-split accuracy; epochs
  to the first stable window; the E1-only pass count (G2's per-seed predicate, for G2(ii)'s planning).

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
- **RD-2 (non-learner, shuffled labels):** the repaired incumbent taught `taught_campaign._shuffled_labels` (`:195`;
  each epoch a re-arrangement never equal to the truth) does NOT solve A_ii.
- **RD-3 (a repair that cannot fail):** the three-off arm, PRESENTED AS a candidate, must be REJECTED by the per-seed
  conjunction. It passes E1 and must fail E3. If the conjunction accepts it, the bar rewards switching safety off: VOID.
- **RD-3b (a repair that turns safety down):** an attenuated-off fixture, each of E/I's, BF's and scaling's correction
  scaled by 1e-3 (a test fixture; no stack constant moved), PRESENTED AS a candidate, must be REJECTED by E3 on the
  designed input. This is P3's "live but inert" shape. If E3 accepts it: VOID.
- **RD-4 (reachable E1 red):** the original stack fails E1 on the confirmatory seeds read so far. If it solves on any
  seed, the bar cannot separate the repair from no repair on that block: VOID.
- **RD-5 (harness sensitivity):** the bare learner solves E1 on the confirmatory seeds. If it fails, the harness or the
  block cannot show learning: VOID.
- **RD-6 (identity sham):** bit-identity to the original. A mismatch means the edit leaks outside its neuter: VOID.

## 4. Sample size (P1) -- the REPAIR ARC's section (not A_ii's or G2(ii)'s; O-11)

**Claim choice (the claim first, then n).** Two claims were sized:
- (a) "the repair beats the original" -- H0 is the measured original rate. Sizable: below.
- (b) "the repair is reliable enough for G2(ii)" -- G2(ii) passes all S = 8 seeds with probability 0.8 only if the
  per-seed rate is >= 0.8^(1/8) = 0.9725. Rejecting "rate <= 0.917" (the rate at which G2(ii) passes at most half the
  time) with an all-pass rule needs n >= ln 0.05 / ln 0.917 = 35 seeds, where the power at 0.9725 is 0.9725^35 = 0.38;
  showing rate >= 0.9725 itself needs n >= 108. **Claim (b) cannot be powered at an affordable n.** This arc registers
  claim (a); G2(ii)'s own registration is the test of (b). The round-1 text's n = 8 came from p1 := 0.8^(1/8), for
  which "largest n with p1^n >= 0.8" returns 8 by identity: that was G2's S = 8 (`g2:225`, itself a habit count,
  `4b6bc9ea g2:684`), not a derivation. It is withdrawn.

| Field | Value |
|---|---|
| Endpoint type | binary per seed: the E1-E5 conjunction passes or not. Statistic: x = number of passing seeds among those read |
| Variance source (measured) | binomial, var p(1-p). **Null rate:** the canonical incumbent (original stack) solves A_ii on 0/12 seeds: dev 0/4 (`g2:534`) + G2 5.1 confirmatory 0/8 (`g2:638`, `:640-645`); exact one-sided 95% upper bound **p0 = 0.2209**. **Alternative rate:** the unstacked learners solve A_ii on every seed: bare 4/4 and three-off 4/4 (`g2:560-561`; REPORT:1384-1385). Both arms ran on the SAME 4 seeds 8000-8003, so they count as n = 4 independent seeds, not 8; exact one-sided 95% lower bound on 4/4 **p1 = 0.4729** |
| Effect the claim needs | the repair restores learning to at least the smallest rate the unstacked learners are consistent with: p1 = 0.4729 against p0 = 0.2209. Disclosed assumption: p1 is an E1-only rate of arms that fail E3 by construction; the repaired conjunction's rate is assumed to reach it. A larger true rate only raises power (table below) |
| Fixed-sample target | exact one-sided binomial test, alpha 0.05, power 0.80: smallest n with a critical count c such that P(X >= c \| p0) <= 0.05 and P(X >= c \| p1) >= 0.80 is **n = 23, c = 9** (alpha 0.0491, power 0.8394). (Normal-approximation formula n = [(z.95 sqrt(p0 q0) + z.80 sqrt(p1 q1)) / (p1 - p0)]^2 = 19.2, rounded up by the exact lattice to 23) |
| Multiplicity | none inside a seed: E1-E5 are an intersection-union conjunction (every part must hold), so the per-seed pass is level-alpha without correction. Across candidates: see "Arc budget" |
| Sequential design | three named looks on the confirmatory block, seeds read in ascending order: **L1** after 8 seeds, **L2** after 15, **L3** (cap) after 23. Efficacy boundaries from the Lan-DeMets O'Brien-Fleming spending function alpha(t) = 2 - 2 Phi(z_0.975 / sqrt t), computed exactly on the binomial lattice under p0: **MET iff x >= 7 at L1, x >= 8 at L2, x >= 9 at L3**. Spend allowed 0.00089 / 0.01523 / 0.05; realised cumulative alpha 0.00017 / 0.00806 / **0.04997** |
| Futility (non-binding) | stop for futility iff the exact one-sided 95% upper bound on p is below p1 at the look: **x <= 1 at L1** (bound 0.471), **x <= 3 at L2** (bound 0.440). Non-binding: the efficacy boundaries were computed ignoring it, so alpha holds whether or not it is obeyed |
| Cap and inflation | hard cap 23 = the fixed-sample target 23 x inflation factor 1.00. On this lattice the cap keeps power >= 0.80: **0.8405** with futility ignored, **0.8199** with futility obeyed. Disclosed: the continuous LD-OBF 3-look factor 1.017 would give 23.4 -> 24, but at 24 the lattice boundaries (7, 8, 10) drop power to 0.785, so 24 is rejected and 23 is kept |
| Operating characteristics (exact) | P(MET): p 0 -> 0.000; p0 0.2209 -> 0.047; p 0.35 -> 0.398; p1 0.4729 -> 0.820; p 0.6 -> 0.980; p 0.8 -> 0.9999; p 0.9 -> 1.000; p 1.0 -> 1.000. Expected seeds read: 14.8 at p0, 18.7 at p1, 9.3 at p 0.9, 8.0 at p 1.0 |
| Simulated before data (stdlib, 200,000 reps per p, RNG seed 20261029, this lane) | P(MET) at p0 = 0.0466, at p1 = 0.8200 (futility obeyed); both match the exact values to 3 decimals. The freeze commits this script with a planted-null and a planted-effect red (O-6) |
| Arc budget (k = 1) | this registration confirms exactly ONE candidate on ONE block, at family alpha 0.05. A NOT MET or INCONCLUSIVE closes it; no second candidate runs under it, and the same candidate is never re-run on a fresh block under it. A further candidate is a new pre-registration that must count this block as a spent look of the arc's family (the integrator rules the family correction, O-13) |
| Not counted | stage A, the dev reads (8000-8007) and Addendum 0's sham checks: exposed, development only |
| Cost (disclosed) | each confirmatory seed runs the candidate, the original, the bare learner, the three-off arm, the sham and the RD-3b fixture at E 400, plus E2's arms and E5's retention read; at most 23 seeds |

**What a non-MET reads as (P8 null-scale item 4).** Every NOT MET states x, the look, and the exact one-sided 95% upper
bound on p.
- A futility stop at L1 or L2 has that bound below p1 by construction: **NOT MET, powered** (mde: the design's power
  reaches 0.80 at p = 0.465 with futility obeyed).
- At the cap with x <= 8 (the boundary not crossed): powered only when the bound is below p1, i.e. x <= 6 of 23. For
  x = 7 or 8 the bound is 0.496 or 0.540: **INCONCLUSIVE** with that mde stated.

## 5. Disposition (fixed in advance; first row that applies decides)

| Reading | When | Consequence |
|---|---|---|
| **VOID** (validity only) | RA-1, RA-2 or RA-3 fails (stage A); or any of RD-1 to RD-6, RD-3b or the B8 twin fails (confirmatory); or the block is exposed before the run | no verdict. Before any candidate outcome on the block is read: repair by a dated addendum on the same seeds. After a candidate outcome is read: the block is spent (it counts as a used look of the family, O-13), and any further read is a new pre-registration on a fresh block the integrator claims |
| **MET** | the efficacy boundary is crossed at L1, L2 or L3 | the inference licensed, and only this: the repaired stack's per-seed rate of passing E1-E5 exceeds 0.2209 (the original's bound) at one-sided alpha 0.05; the record states x, the look and the exact lower bound (e.g. 7/8 -> 0.529; 9/23 -> 0.222). It is the measured win. It does NOT certify G2(ii) readiness (section 4, claim (b)). It may land through the normal arc (2-lens panel on the build, the battery, the push) after ruling O-12; then G2(ii), G3 C11 on S0C1 and G9 unpark under their own registrations, and a G2(ii) failure re-parks them |
| **NOT MET, powered** | a futility stop (x <= 1 at L1, x <= 3 at L2), or x <= 6 at the cap | the candidate is rejected evidence and never lands. A NULL-RECORD with scale: units = the 9 response neurons x H inputs, ticks = E 400 epochs, seeds = those read, the mde (P8 null-scale item 1). It was never on the live path, so nothing is deleted or exempted |
| **INCONCLUSIVE** | x = 7 or 8 at the cap | it does not land, since there is no measured win. The NULL-RECORD says INCONCLUSIVE with its bound and mde (P8 item 4); a further run needs its own pre-registration (Arc budget) |

The live brain stays byte-identical under every reading except MET, and under MET until the arc lands.

## 6. Owed before freeze

- **O-1 Seeds.** The integrator assigns and claims the confirmatory block on the board. Proposed: 8100-8122 (23 seeds
  from G2's "8100-8199 held, unassigned", `g2:268`), read in ascending order; stage A's 8004-8007 from G2's
  further-development block. 8200-8207 is exposed for A_ii (G2 5.1) and is never reused.
- **O-2 Held paths.** Confirm whether `readout/interaction_learning.py`, `plasticity/plasticity_rule_stack.py` and
  `plasticity/homeostatic_plasticity_stack.py` are held by another arc (C4-C6 needed held-path patches for G9). Stage A
  adds capture points inside `stabilize`, and stage B edits one of the cited methods.
- **O-3 Instrument location.** Stage A's per-stage capture reuses round 6's instrument. graph-exists must be run for it
  under `src/` before any build. The round-6 script lives outside the repo, and P4c says it is not a build until it
  lands.
- **O-4 E3 liveness on the designed input** (section 3.1). The owner or integrator rules whether each stabilizer's
  liveness is read on its designed input (the runaway self-check fixture, effect no smaller than the original's), not
  on A_ii, where the ORIGINAL stack's scaling is already inert. Disclosed: the old A->B->rest-400 fixture
  (REPORT:1550) binds NOTHING here. Its conjuncts were acquisitions > .9, A recovery >= .80, no worse than the
  original, and a nonzero scaling effect; the original itself fails A recovery (0.0, "FAIL, INHERITED"), and the
  retention arc found its rest-400 reading a knife edge. Retention is read by E5 on #2565's redesigned instrument
  instead.
- **O-5 P1 design.** The policy-enforcer reviews section 4: claim (a) and its measured p0/p1, the 3-look LD-OBF design
  with non-binding futility and cap 23, and claim (b) recorded as unpowerable. The sequential shape is used, so P1
  item 3's one-shot clause is NOT invoked.
- **O-6 Simulation script.** Commit section 4's stdlib exact-lattice computation and Monte Carlo (RNG seed 20261029)
  with a planted-null and a planted-effect red, before data.
- **O-7 Perception tier.** The confirmatory record states `state()['perception_tier']` and `perceiving_audio` for the B8
  twin brain. A 'stdlib' tier halts.
- **O-8 G2 F7(a).** A_ii's A1 (generator check) reading on dev seeds 8000-8003 (`g2:681` on `4b6bc9ea`). It is G2's to
  write, and it precedes the confirmatory read: a generator that fails A1 voids A_ii as a problem.
- **O-9 Records gap.** The six A_ii candidate nulls C1-C6 (and the norm60 diagnostic arm) are absent from
  `project-snn-refuted-and-negatives.md`, and their NULL-RECORDs are an open owner question (G2 Q3; memory
  `project-snn-owner-e1b-production-tier-g2-twin-1023` :25). This draft neither writes nor re-tests them.
- **O-10 Review.** The 2-lens panel and the policy-enforcer review this text. Their challenges are folded, and the
  freeze follows their AGREEs.
- **O-11 A_ii's and G2(ii)'s own P1 sections** (D-67, `4b6bc9ea g2:702`; 2.18.2 row P1, `:684`) remain owed in G2.
  Section 4 does not supply them; G2(ii)'s S must be derived there (claim (b) above is input to that derivation).
- **O-12 Landing licence.** The owner or integrator rules whether claim (a)'s MET (rate above the original's bound)
  suffices to land the repair on the live brain, given that G2(ii) needs a rate near 1 that this arc cannot certify.
  If not, MET is recorded and the landing waits on G2(ii) run against the candidate branch.
- **O-13 Family budget.** The integrator rules how a second candidate (a new pre-registration) corrects for this
  block as a spent look of the A_ii-repair family (e.g. alpha split across the family's confirmatory blocks).
- **O-14 Addendum 0.** The four candidate diffs (2.3) are committed as a dated pre-data addendum before the first
  stage-A run, each with its predicted R sign and its identity sham.

## Fold 1 (2026-10-29) -- changelog

Folds lens A #6792, lens C #6831 and the policy-enforcer #6817 on `a7ae6781`. One row per finding, with dedupes, in
`lanes/engine-1024/aii-prereg-row/FOLD.md`.
- **Section 4 rewritten (KNOWN ITEM, all three reviewers).** n = 8 withdrawn: it was G2's S returned by an identity.
  New claim (a) "beats the original" with measured p0 = 0.2209 (0/12) and p1 = 0.4729 (4/4, 4 independent seeds);
  fixed-sample target n = 23 (exact); 3-look LD-OBF sequential design (7/8, 8/15, 9/23), non-binding futility (<= 1/8,
  <= 3/15), cap 23, inflation 1.00, power 0.82-0.84, exact and simulated. Claim (b) "G2-ready" recorded as unpowerable
  (35 seeds at power 0.38; 108 to show 0.9725). Section retitled as the repair arc's P1 section; O-11 added.
- **Arc budget k = 1** and VOID-after-exposure spends the block (lens A F3); O-13 added.
- **Stage A frozen harder (lens A F4, lens C F8):** Addendum 0 freezes all four candidate diffs before any stage-A run
  (O-14); share defined as a sum with a sign rule, a contradiction guard and a strict-majority pick; on/off-subset
  Shapley attribution added beside the sequential share; K-HS added as the scaling branch; RA-3 on 8/8 seeds (lens C
  F9); RA-1 relabelled a logger-consistency check, RA-2 the red; the inert intrinsic/BCM halt branch struck and the
  capture stated POST-only (lens C F4).
- **E5 retention added** with owner #2527/#2565 cited; the old rest-400 fixture dropped from E3 and disclosed in O-4
  (lens C F1, F2). E3 liveness moved to the designed input with an effect no smaller than the original's, and RD-3b
  added (lens A F5).
- **MET row states its exact inference** and O-12 added (lens A F6; policy-enforcer consequence).
- **One-failure inconsistency removed** (policy-enforcer NIT, lens C KNOWN ITEM (2)): the new non-MET rules follow the
  look and the bound.
- **Consistency fixes (lens C F5, F6, F7, F10):** six nulls = C1-C6 plus the norm60 diagnostic arm; line cites
  `:538`, `:647-649`, `taught_campaign.py:63-65`, `g2:638`; BF deep beakers start at the first `consolidate` call's
  weights; constructor parameter names; "G2 5.1"; `_shuffled_labels`; E2's arms named; the original's max
  drive/ceiling 0.43-0.46; the "agree to 4 decimals" sentence removed with the n = 8 table; "3-lens" corrected to
  the 2-lens panel plus the policy-enforcer.
