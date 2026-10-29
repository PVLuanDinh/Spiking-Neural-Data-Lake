# A_ii STACK REPAIR -- make the CANONICAL brain learn A_ii (DRAFT pre-registration)

> **Status: DRAFT, pre-data, NOT frozen.** Lane aii-prereg (thread AII-PREREG), 2026-10-29, written on
> `claude/aii-prereg-1029` off `c4c10b28`. Docs only: no brain, no battery, no run of any kind went into this file.
> Every number below is quoted from a cited record; none was measured here. The only arithmetic computed here is the
> sample-size design in section 4 (stdlib python, exact binomial lattice plus a Monte Carlo check, no brain).
> Fold 1 (2026-10-29, the end of this file) folds the 2-lens panel (#6792, #6831) and the policy-enforcer (#6817).
> Fold 2 (2026-10-29, the end of this file) folds the policy-enforcer's #6893 and its addendum #6904.
> Fold 3 (2026-10-29, the end of this file) re-derives section 4 at alpha_1 = 0.0303964 (launcher D-82) with `tools/aii_o6_design.py`.
> Fold 4 (2026-10-29, Round 5, the end of this file) folds the enforcer's #7019 (2 MAJOR, 4 MINOR, 1 NIT).
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
  one-sided alpha_1 = 0.0304 (section 4, "Arc budget"). The incumbent is `taught_problems.canonical_incumbent` (`taught_problems.py:386-398`: the class
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
| original stack, same seeds (paired) | E1 count below RD-4's tolerance; its per-seed E1 is also MET's paired comparator (section 4) | dev and confirmatory |
| bare learner (`im._incumbent`) | E1 count at or above RD-5's tolerance (bare 4/4 measured, first windows 108-131, REPORT:1384-1385) | dev and confirmatory |
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
- **RD-4 (reachable E1 red; the historical null still holds):** the original stack runs paired on every confirmatory
  seed. At each look, VOID iff its E1 count on the seeds read so far is **>= 7 of 10, >= 10 of 20, or >= 14 of 30**. Each
  threshold is the smallest count with P(count >= t | rate p0 = 0.2209) <= 0.02/3. So RD-4 VOIDs only when the block
  itself rejects "the original's rate <= p0", the assumption claim (a)'s H0 rests on. Own error rate: P(RD-4 VOID over
  the 3 looks | original at p0) = 0.0077 (exact). Round 2's "any original success VOIDs" is withdrawn: at a true rate
  of p0 it VOIDed with probability 0.997 over 23 seeds (#6904).
- **RD-5 (harness sensitivity):** the bare learner runs on every confirmatory seed. At each look, VOID iff its E1 count
  is **< 1 of 10, < 4 of 20, or < 8 of 30**. Each threshold is the largest count with P(count < l | rate p1 = 0.4729)
  <= 0.02/3. So RD-5 VOIDs only when the block rejects "the bare learner reaches p1", the rate section 4's effect
  assumes. Own error rate: P(RD-5 VOID over the 3 looks | bare at p1) = 0.0088 (exact). Round 2's "any bare failure
  VOIDs" is withdrawn (P(bare solves all 23) = 0.307 at a rate of 0.95, #6904).
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
| Fixed-sample target | exact one-sided binomial test at the block's alpha_1 = 0.0303964 (series rule, "Arc budget"), power 0.80: smallest n with a critical count c such that P(X >= c \| p0) <= alpha_1 and P(X >= c \| p1) >= 0.80 is **n = 27, c = 11** (alpha 0.0227, power 0.8085). At the former alpha 0.05 it was n = 23, c = 9 (alpha 0.0491, power 0.8394). Produced by `tools/aii_o6_design.py` (its default alpha is alpha_1 = 0.0303964; `--alpha 0.05` is the Fold 2 reproduction) |
| Multiplicity | none inside a seed: E1-E5 are an intersection-union conjunction (every part must hold), so the per-seed pass is level-alpha without correction. Across candidates: see "Arc budget" |
| Sequential design | three named looks, the same count and the same boundary family as before (Lan-DeMets O'Brien-Fleming, alpha(t) = 2 - 2 Phi(z_{1-alpha_1/2} / sqrt t), exact on the binomial lattice under p0); only the look sizes move with the cap: **L1** after 10 seeds, **L2** after 20, **L3** (cap) after 30, seeds read in ascending order. **MET iff x >= 8 at L1, x >= 10 at L2, x >= 12 at L3.** Spend allowed 0.00018 / 0.00802 / 0.03040; realised cumulative alpha 0.00017 / 0.00563 / **0.02214**. (Former design at alpha 0.05: looks 8/15/23, counts 7/8/9.) **Paired guard (Fold 2):** MET also needs x > x_orig, the original's E1 count on the same seeds. This is an intersection-union conjunct, so it can only lower alpha. If the boundary is crossed but the guard fails, the block continues to the next look; at the cap it reads INCONCLUSIVE (section 5) |
| Method (disclosed) | claim (a)'s H0 is the HISTORICAL bound p0 (the original's 0/12), not a concurrent two-arm comparison. The original IS re-run, paired, on every confirmatory seed: as RD-4's validity control (VOID when the block rejects "rate <= p0") and as the paired guard. A drift of the original's true rate up to RD-4's tolerance is not excluded by the H0; the guard keeps MET from being read when the concurrent original does as well. **The guard does NOT control drift (enforcer #6990 NIT, Fold 3).** If the original's true rate has drifted to q and the candidate is the original (no repair effect, bare 1), P(MET) / P(VOID) at alpha_1 are **0.0212 / 0.0053 at q = p0, 0.0500 / 0.0139 at q = 0.25, 0.1431 / 0.0542 at q = 0.30, 0.2668 / 0.1483 at q = 0.35** (at the former alpha 0.05: 0.0454 / 0.0044, 0.0862 / 0.0113, 0.1882 / 0.0431, 0.3003 / 0.1166, reproducing #6990). RD-4's tolerance is too loose to catch that range, and the guard only compares against the concurrent original, which has drifted with the candidate. Claim (a) is a one-sample test against the historical p0 and is stated so; a reader must not take the guard or RD-4 as a drift control. `tools/aii_o6_design.py` prints this row. |
| Validity tolerances (Fold 2) | RD-4 and RD-5 (section 3.3) VOID only when the block itself rejects the rate the design assumes for that arm, each at an own error rate <= 0.02 over the 3 looks (Bonferroni 0.02/3 per look): RD-4 thresholds 7/10/14 of 10/20/30, error 0.0077 at p0; RD-5 thresholds 1/4/8, error 0.0088 at p1. Assumption (disclosed): the three arms' per-seed outcomes are independent across arms |
| Futility (non-binding) | stop for futility iff the exact one-sided 95% upper bound on p is below p1 at the look: **x <= 1 at L1** (bound 0.394), **x <= 5 at L2** (bound 0.456). Non-binding: the efficacy boundaries were computed ignoring it, so alpha holds whether or not it is obeyed |
| Cap and inflation | **hard cap 30 = the fixed-sample target 27 x inflation factor 1.11** (raised from 23 / 1.00 so that power stays >= 0.80 at the smaller alpha_1; the owner approved more runs, D-82). The cap is the SMALLEST n >= 27 with the same 3-look family (looks ceil(n/3), round(2n/3), n) whose worst-case P(MET and no VOID) at p1 is >= 0.80: caps 27 / 28 / 29 give 0.7869 / 0.7365 / 0.7822 and fail; **30 gives 0.8198**. Power at the cap: **0.8400** with futility ignored, **0.8320** with futility obeyed (no VOID); counting VOID and the paired guard, P(MET and no VOID) at p1 is **0.8198** in the worst case the measured counts allow (original at p0 AND bare at p1), and 0.8320 at the measured point rates (original 0, bare 1). Cap 31 gives 0.8384 and cap 32 gives 0.7980: power is not monotone in n on a lattice, so the smallest passing cap is the registered one |
| Operating characteristics (exact) | P(MET), no VOID, guard idle: p 0 -> 0.000; p0 0.2209 -> 0.022 (0.0215); p 0.35 -> 0.344; p1 0.4729 -> 0.832; p 0.6 -> 0.990; p 0.8 -> 1.000; p 0.9 -> 1.000; p 1.0 -> 1.000. Expected seeds read: 19.4 at p0, 24.1 at p1, 10.7 at p 0.9, 10.0 at p 1.0. mde (power 0.80 at the cap, futility obeyed): p = 0.462 |
| P(MET and no VOID) beside P(VOID) (Fold 2; exact joint lattice over the three arms, guard included) | measured point rates (original 0, bare 1): P(VOID) = 0 and P(MET) as in the row above. Worst case (original p0, bare p1): candidate at p0 -> MET 0.0210, VOID 0.0097; at 0.35 -> 0.3377, 0.0135; at p1 -> **0.8198, 0.0128**; at 0.6 -> 0.9801, 0.0100; at 0.9 -> 0.9962, 0.0038. Round 2's VOID rules gave P(VOID) 0.997 at the original's p0 (#6904); these give <= 0.015 |
| Simulated before data (stdlib, 200,000 reps per p, RNG seed 20261029; committed: `tools/aii_o6_design.py`) | candidate arm alone (guard idle, no VOID): P(MET) at p0 = 0.0212, at p1 = 0.8314 (futility obeyed), against the exact 0.0215 and 0.8320. JOINT lattice (Round 5; candidate p1, original p0, bare p1, VOID and the paired guard on): P(MET) 0.8191 and P(VOID) 0.0131 against the exact 0.8198 and 0.0128. All within Monte Carlo error (SE about 0.0003 to 0.0009). Reds: `tests/test_aii_o6_design.py` (planted null, planted effect, an always-solving original, the measured point rates; Round 5: one pin per registered number at alpha_1, the smallest-cap power red, a failing bare arm, VOID-before-MET ordering, the joint Monte Carlo, input validation, the bound lines), queued to run after the campaign |
| Arc budget (series rule, D-82) | **Open-ended alpha-spending series over the A_ii-repair family (launcher D-82, replacing option (a)):** candidate i is confirmatory at alpha_i = 0.05 x 6 / (pi^2 i^2) = 0.0304, 0.0076, 0.0034, ... (sums to 0.05 for any number of candidates). It is fixed in this record BEFORE block 1's data. This registration is candidate i = 1: ONE candidate on ONE block, at alpha_1 = **0.0303964**. A NOT MET or INCONCLUSIVE closes it; no second candidate runs under it, and the same candidate is never re-run on a fresh block under it. A further candidate i claims at alpha_i in its own pre-registration, with its own P1 section re-derived at alpha_i (the script takes `--alpha`); a VOID after a candidate outcome is read spends candidate 1's block, and any further read is a later candidate in the same series at its alpha_i (one family, D-82). Every candidate serves one decision (the G2(ii) unpark) |
| Not counted | stage A, the dev reads (8000-8007) and Addendum 0's sham checks: exposed, development only |
| Cost (disclosed) | each confirmatory seed runs the candidate, the original, the bare learner, the three-off arm, the sham and the RD-3b fixture at E 400, plus E2's arms and E5's retention read; at most 30 seeds |

**What a non-MET reads as (P8 null-scale item 4).** Every NOT MET states x, the look, and the exact one-sided 95% upper
bound on p.
- A futility stop at L1 or L2 has that bound below p1 by construction: **NOT MET, powered** (mde: the design's power
  reaches 0.80 at p = 0.462 with futility obeyed).
- At the cap with x <= 11 (the boundary not crossed): powered only when the bound is below p1, i.e. x <= 9 of 30. For
  x = 10 or 11 the bound is 0.499 or 0.533: **INCONCLUSIVE** with that mde stated.
- At the cap with x >= 12 but x <= x_orig (the paired guard fails): **INCONCLUSIVE**. The candidate clears the historical
  bound but does not beat the concurrent original; the record states x, x_orig and both bounds.

## 5. Disposition (fixed in advance; first row that applies decides)

| Reading | When | Consequence |
|---|---|---|
| **VOID** (validity only) | RA-1, RA-2 or RA-3 fails (stage A); or any of RD-1 to RD-6, RD-3b or the B8 twin fails (confirmatory); or the block is exposed before the run | no verdict. Before any candidate outcome on the block is read: repair by a dated addendum on the same seeds. After a candidate outcome is read: the block is spent, and any further read is a later candidate in the same series at alpha_i (section 4 "Arc budget"; one family, D-82), in its own pre-registration on a fresh block the integrator claims. RD-4 and RD-5 fire only past their tolerances (section 3.3); P(VOID) is stated beside P(MET) in section 4 |
| **MET** | the efficacy boundary is crossed at L1, L2 or L3 AND x > x_orig on the same seeds (the paired guard) | the inference licensed, and only this: the repaired stack's per-seed rate of passing E1-E5 exceeds 0.2209 (the original's bound) at one-sided alpha_1 = 0.0304; the record states x, the look and the exact one-sided lower bound at confidence 1 - alpha_1 = 0.9696 (8/10 -> 0.457; 10/20 -> 0.280; 12/30 -> 0.233; the 95% values 0.493 / 0.302 / 0.250 are shown for reference). These bounds ignore the sequential stop; every one still exceeds p0 = 0.2209, so no verdict depends on the label. It is the measured win. It does NOT certify G2(ii) readiness (section 4, claim (b)). It may land through the normal arc (2-lens panel on the build, the battery, the push) after ruling O-12; then G2(ii), G3 C11 on S0C1 and G9 unpark under their own registrations, and a G2(ii) failure re-parks them |
| **NOT MET, powered** | a futility stop (x <= 1 at L1, x <= 5 at L2), or x <= 9 at the cap | the candidate is rejected evidence and never lands. A NULL-RECORD with scale: units = the 9 response neurons x H inputs, ticks = E 400 epochs, seeds = those read, the mde (P8 null-scale item 1). It was never on the live path, so nothing is deleted or exempted |
| **INCONCLUSIVE** | x = 10 or 11 at the cap; or x >= 12 at the cap with x <= x_orig | it does not land, since there is no measured win. The NULL-RECORD says INCONCLUSIVE with its bound and mde (P8 item 4); a further run needs its own pre-registration (Arc budget) |

The live brain stays byte-identical under every reading except MET, and under MET until the arc lands.

## 6. Owed before freeze

- **O-1 Seeds.** The integrator assigns and claims the confirmatory block on the board. Proposed: 8100-8129 (30 seeds
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
  with non-binding futility and cap 30 (alpha_1, series rule), and claim (b) recorded as unpowerable. The sequential shape is used, so P1
  item 3's one-shot clause is NOT invoked.
- **O-6 Simulation script.** Commit section 4's stdlib exact-lattice computation and Monte Carlo (RNG seed 20261029)
  with a planted-null and a planted-effect red, before data. It includes Fold 2's exact joint lattice over the
  candidate, original and bare arms (RD-4/RD-5 tolerances, paired guard, P(VOID)), with two more reds: an original
  that always solves VOIDs at L1 with certainty, and at the measured point rates P(VOID) = 0 and P(MET) at p1 = 0.8199.
  **BUILT (2026-10-29, Round 4; Round 5 fold):** `tools/aii_o6_design.py` @ `92b7e3db`, Round 5 @ `d8072c69`; reds `tests/test_aii_o6_design.py` @ `3b6a9197`, Round 5 reds @ `6a8a7912`
  on `claude/aii-prereg-1029`; alpha and looks are parameters. Run line and printed numbers: row REPORT.md. Reds queued
  to run after the campaign.
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
- **O-13 Family budget -- ANSWERED (launcher D-82, 2026-10-29).** Neither (a) nor (b): an open-ended alpha-spending series over the
  A_ii-repair family, alpha_i = 0.05 x 6 / (pi^2 i^2); this block is candidate 1 at alpha_1 = 0.0303964, and its n cap is raised to
  30 so power stays >= 0.80 (section 4, Fold 3). The integrator writes the O-13 RULED (D-82) entry in the D-78 section.
- **O-15 D-78 disclosure text -- DONE (integrator, @6af4d1fa).** The D-78 method disclosure now says the original IS re-run paired on
  every seed (RD-4 control and the paired guard); nothing is owed.
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

## Integrator rulings and method disclosure (2026-10-29, `claude-engine-1024`; launcher D-78; pre-data)

- **Method disclosure (D-78):** claim (a) is a ONE-SAMPLE test of the repaired stack's per-seed pass rate against a HISTORICAL
  control's bound (the original stack's 0/12, Clopper-Pearson upper p0 = 1 - 0.05^(1/12) = .2209), not a concurrent two-arm
  comparison. The original IS re-run, paired, on every confirmatory seed, as RD-4's validity control and the paired guard
  (x > x_orig); the test's H0 is still the historical bound p0, not a concurrent two-arm comparison. (Corrected 2026-10-29 by the
  integrator, O-15, enforcer #6893 MINOR-1: this sentence said the original is not re-run, which Fold 2 made false.)
- **O-12 RULED (integrator, accepted by launcher D-78):** claim (a)'s MET licenses landing the repair on the live brain only through
  the normal arc (panel, then the full battery including no regression on the existing learner checks). G2 (ii) stays PARKED until
  G2's own registered reading runs on the landed brain (the owner's Q-AII unpark condition, unchanged); (a) does not certify it.
- **O-13 RULED (integrator, accepted by launcher D-78):** this block keeps k = 1 at alpha .05 as registered. Any later candidate's
  pre-registration applies Holm across the A_ii-repair family's confirmatory blocks, counting this block as spent, and claims only at
  the adjusted level.
- **O-13 RE-RULED (launcher D-82, 2026-10-29, before any data; supersedes the Holm clause above, which #6893 MINOR-2 showed cannot
  follow a block that claims at the full .05):** an open-ended alpha-spending series over the A_ii-repair family. Candidate i is
  confirmatory at alpha_i = .05 x 6/(pi^2 i^2) (.0304, .0076, .0034, ...), which sums to .05 for any number of candidates. This
  block is candidate 1, at alpha_1 = .0303964; section 4 is re-derived at that alpha with its n cap raised to keep power >= .80
  (Fold 3, @08c27533; the owner approved more runs, "Fund more runs"). Launcher's reason: every candidate serves one decision
  (G2 (ii) unpark), so a new family per candidate would leave that decision's false-MET rate uncontrolled (1 - .95^k), and
  pre-allocation needs a k_max guess.

## Fold 2 (2026-10-29) -- changelog

Folds the policy-enforcer's #6893 (2 MINOR) and its addendum #6904 (1 MAJOR) on `bc7c85ea` + `2d7054d1`. The D-78 section
above is the integrator's and is unchanged; where a finding lands in it, the fix is written in this lane's text and the
D-78 edit is listed as owed (O-13, O-15). Numbers come from an exact stdlib joint binomial lattice (no brain, no run).
- **MAJOR-1 (#6904, validity reds vs the design's own rates): FOLDED.** RD-4 and RD-5 now VOID only when the block rejects
  the rate the design assumes for that arm. RD-4: original >= 6/8, 9/15, 11/23 (error 0.0078 at p0). RD-5: bare < 1/8,
  2/15, 5/23 (error 0.0086 at p1). Each uses Bonferroni 0.02/3 per look. Section 4 now states P(MET and no VOID) beside
  P(VOID): worst case 0.8032 / 0.0133 at p1, so power stays >= 0.80 with VOID counted. The round-2 rules gave
  P(VOID) 0.997. Controls table and section 5's VOID row updated.
- **Paired guard added (from #6904's alternative and MINOR-1's paired original):** MET also needs x > x_orig. It is an
  IUT conjunct, so alpha can only fall; it costs 0.006 of worst-case power. A guard failure at the cap reads INCONCLUSIVE.
- **MINOR-1 (#6893, the D-78 disclosure says the original is not re-run): FOLDED in section 4's new "Method" row, OWED in
  D-78 (O-15).** The H0 is the historical p0; the original IS run paired, as RD-4's control and the guard.
- **MINOR-2 (#6893, Holm cannot follow a block that claimed at 0.05): FOLDED as option (a) in section 4's "Arc budget"
  and section 5; O-13 REOPENED to the integrator** (confirm (a), or pre-allocate 0.05/k_max and re-derive section 4).
- **O-6 extended** to commit the joint-lattice script with two more reds.

## Fold 3 (2026-10-29) -- changelog

Round 4: O-6 built, and section 4 re-derived at the D-82 alpha. The D-78 section is the integrator's and is unchanged.
- **O-6 BUILT:** `tools/aii_o6_design.py` (exact lattice, joint three-arm lattice, seeded Monte Carlo; alpha and looks are
  parameters), reds `tests/test_aii_o6_design.py`. At alpha 0.05 every printed number reproduces Fold 2's (REPORT.md lists them).
- **O-13 ANSWERED (D-82): series rule** alpha_i = 0.05 x 6/(pi^2 i^2) replaces option (a) in "Arc budget"; block 1 at alpha_1 = 0.0303964.
- **Section 4 re-derived at alpha_1 (same 3 looks and LD-OBF family; only the look sizes move):** fixed-sample n 23 -> 27 (c 9 -> 11);
  cap 23 -> 30 (inflation 1.00 -> 1.11; the smallest cap with worst-case P(MET and no VOID) at p1 >= 0.80: 0.8198 at 30, caps 27-29 fail);
  looks 8/15/23 -> 10/20/30; MET counts 7/8/9 -> 8/10/12; futility counts 1/3 -> 1/5; RD-4 6/9/11 -> 7/10/14; RD-5 1/2/5 -> 1/4/8;
  power at the cap 0.8405/0.8199 -> 0.8400/0.8320 (futility ignored/obeyed); seeds 8100-8122 -> 8100-8129 (O-1); non-MET and
  disposition counts updated (section 4 reading list, section 5).
- **Enforcer #6990 NIT folded:** the drift row is printed in section 4 "Method" beside the paired guard, with the plain statement that
  the guard does not control drift and RD-4's tolerance is too loose to catch that range.

## Fold 4 (2026-10-29, Round 5) -- changelog

Folds the policy-enforcer's #7019 on `c275b299`. One row per item in `lanes/engine-1024/aii-prereg-row/FOLD.md`; the D-78 section is the integrator's and is unchanged.
- **MAJOR-1:** one red per registered number at alpha_1 = 0.0303964 (n 27, c 11, cap 30, looks 10/20/30, MET 8/10/12, futility 1/5 and 9, RD-4 7/10/14, RD-5 1/4/8, 0.8198 / 0.8320, joint rows, drift row, planted cap 31 / eff [8,10,13] / lax boundary). The alpha-0.05 tests stay as Fold 2's regression test, labelled so.
- **MAJOR-2:** a red that joint power is >= 0.80 at the returned cap and < 0.80 at cap - 1 (and the five neighbouring caps' values); `derive` now RAISES when no cap reaches power (`cap_max`).
- **MINOR-1:** the script's bound lines are derived from the design (MET lower bounds per look, INCONCLUSIVE upper bounds at the cap); the false "nothing hard-coded" docstring is rewritten.
- **MINOR-2:** the VOID rows read "a later candidate in the same series at alpha_i" (one family, D-82); the stale O-15 sentence in Method and the open O-15 entry are removed.
- **MINOR-3:** `--alpha` defaults to alpha_1; the 0.05 run is "Fold 2 reproduction".
- **MINOR-4:** the MET row's lower bounds are labelled and printed at confidence 1 - alpha_1 (0.457 / 0.280 / 0.233), 95% values kept for reference.
- **NIT:** vacuous `joint(0) == 0` replaced by a lax-boundary red on the joint lattice; a failing-bare-arm (RD-5) red and a VOID-before-MET red added; the Monte Carlo now covers the joint lattice; `_fixed_sample` and `--looks` raise ValueError (CLI exits with a usage error).
