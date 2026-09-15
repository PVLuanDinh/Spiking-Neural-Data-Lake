# T063 — the three-arm drift harness: is the H01 prior beatable, and does learning move the brain off it?

DRAFT pre-registration, written 2026-09-09 while T062 builds, from the read-only orientation
`scratchpad/t063/design.md` (66 facts, 12 open questions), then REVISED before commit after the arc's one
Codex read-only adversarial read (archived beside it as `codex_defects.md`; its twelve defects and the
correction each produced are listed in §2.11 — Codex's output is data, the decisions are this session's).
It is committed only after T062's arc is pushed, as
`specs/022-unified-brain-arc/evidence/t063-prior-drift-harness.md`; until then every number in §1 is a
citation and every rule in §2 is a decision this session makes under the 2026-09-05 authorization and
discloses as its own. Governing text: `tasks.md::T063`, `spec.md::SC-006`/`FR-022`, specs/011 (SC-004,
SC-005, FR-008/009/010, C5, D1, D4, Scenario 2, T003, T004), PLAN-FINAL Part F's preamble (S = 8, paired
one-sided t > 1.895 with 7 d.f., ties counted, permutation false-positive rate printed, no majority-of-
seeds rule), POLICIES.md P2/P3/P8 and the e-prop invariant's SCOPE clause.

## 1. Territory (measured 2026-09-09; each fact is cited in design.md)

**1.1 Two premises of the row are FALSE in the tree, each with a pinned measurement.** (a) On the OBJECT
brain the prior has no reader: `tests/test_placement_charge_null.py` pins that a brain whose placement is
replaced by constant tuples is value-identical over 300 ticks; the live learner is a fully-connected
`n_in x n_labels` R-STDP layer with no topology. Three arms seeded through placement would produce three
identical outcomes (F60): direction 2 could never fire. (b) On the KERNEL population (T062's deliverable)
NOTHING can learn: `Synapses` compiles `(pre, post, weight, delay)` into quantized bundles once and has no
write path; `KernelTick.tick` writes only v/refrac/last_spike; no rule in `plasticity/*` (22 enumerated),
`cortex/eprop_kernel` (torch, offline encoders) or the 0x09 writer (NOT_A_RULE, not live, dense-only) can
touch a bundle table (F16–F27). "Learning moves the brain off the seed" has NO MECHANISM on either route,
so T063 carries the one thing the row cannot run without: a learner on the kernel's synapses.

**1.2 What specs/011 fixed and what it left open.** Fixed: `DriftRecord` = (statistic, seeded, current,
distance, direction) per seeded statistic; `distance == 0` across all statistics is a FAILURE of the
seeding approach, its message naming pinning (FR-009, C5, T012); sign is EXCLUDED from drift (Dale) and
SC-004 asserts it survives learning; the prior is read ONCE at construction; the seeded quantity is a
PROPORTION (D1); the instrument is shown to work BEFORE any seeding is blessed (D4); FR-010 requires the
random and the wrong arms; SC-005 requires the wrong prior to UNDERPERFORM random. OPEN and unmade: the
downstream TASK (T003; `decisions.md` does not exist), the WRONG prior (T004), the distance formula, the
width, tick count and seed count, degree preservation in the random arm, wrong-arm-as-bar vs as-red,
T071's "prior separation", a Part F row, and the P2 adjudication against `DriftMonitor` ([B10.13]).

**1.3 What exists to adopt.** `DriftMonitor.psi` (mean per-feature PSI with eps) — a distance for
distributional statistics. `RSTDPNeuron` (r_stdp.py) — the tree's live three-factor rule: pair-based
eligibility with exponential kernels, slow decay, reward gating `w += lr * d * e`, and a module docstring
that itself rejects "mere weight movement" as a charge criterion. `ng_snn.marginal_holdout` — the
marginal-preserving split. `gradient_claim_gate` dispositions: LOCAL / APPROXIMATES_GRADIENT /
EXACT_GRADIENT / CONTROL / NOT_A_RULE / DIRECT_SOLVE. The kernel's `wire` exemptions name a consumer ON
THE LIVE TICK; an instrument that ticks the population is not that consumer and does not discharge them.

## 2. Pre-registration

**2.1 Deliverables.**
- `basal_ganglia/r_stdp.py` — P2 MERGE FIRST: the rule's three operations (eligibility on a causal pair,
  eligibility on an anti-causal pair, decay; reward gating) are lifted into ONE authoritative
  implementation — module-level functions over arrays — that `RSTDPNeuron` calls for its per-object lists
  AND the kernel learner calls over bundle arrays. Identical-input equivalence is a test: the per-object
  neuron and the kernel form, fed one spike/reward sequence, produce the same weights to within the
  kernel's quantization step; r_stdp's existing reds stay green. One mechanism survives; if the merge
  cannot be made behaviour-equivalent, the debt is RECORDED as unresolved P2 debt, never claimed closed.
- `substrate/kernel_plasticity.py` — the kernel-side driver of that one rule: per-bundle eligibility
  arrays beside `qweight`, updated from the tick's spike sets, decayed per tick, gated by a broadcast
  scalar reward into `qweight` with re-quantization at `frac_bits` (same seed + same reward sequence =
  byte-equal weights; the determinism gate's contract holds). SIGN IS PRESERVED THROUGH LEARNING: a
  weight moves toward zero and may reach zero but never crosses it (SC-004 as a checked invariant; the
  red: remove the clamp, a flip occurs, the check fires). `GRADIENT_CLAIM`: `LOCAL` (mirroring r_stdp's
  own claim for `reward`; no gradient propagated; no cosine owed under the SCOPE clause — the gate's
  census re-run in the record). `method_dormancy_gate`: the learner's learning-verb methods have their
  non-`__main__` caller in the harness.
- `substrate/drift_monitor.py` — P2 MERGE: the PSI formula lifted into a module-level `psi(actual,
  reference, eps)` that `DriftMonitor.psi` calls and the harness imports (the formula lives ONCE).
  `DriftMonitor` (an online sensor-window monitor whose zero is a pass) is NOT the drift instrument and
  is not retired. This is the ADOPT/MERGE decision [B10.13] demanded.
- `cortex/prior_drift_harness.py` — the ONE harness: `DriftRecord` (specs/011 §4 fields), the arms, the
  task, the statistics, the diagnostics, both reds, the positive control, the verdict printer.
  Registered REGIONS / `run_selfchecks` CORE / `dormancy_gate.EXEMPT` (an instrument over a scratch
  population, the `amdahl_gate` shape). Bare invocation = an offline self-check on a 64-unit synthetic
  population (both reds fire, the positive control fires, the controls hold; < 60 s); the campaign runs
  behind `--campaign`.
- `specs/011-h01-neurogenesis-prior/decisions.md` — CREATED with T003 and T004 decided (2.3, 2.2), each
  with the rejected candidates and the side on which each failed, plus the measurement T003 requires.
- A dated Part F row `H-R1` appended to PLAN-FINAL (SC-006's reds, controls, widths).
- `tests/test_prior_drift_harness.py` and `tests/test_kernel_plasticity.py` — both reds, the positive
  control, determinism, sign preservation, the read-once rule, the equivalence test, the invalid-control
  refusals (2.2), and the diagnostics of 2.5.

**2.2 The arms (same units, same drive, same learner, same readout, same NUISANCE properties — only
the wiring differs).** Nuisance properties held constant across every arm, by construction: the unit
set and each unit's cell sign; every unit's OUTGOING edges carry its own sign (Dale); every unit's
total absolute fan-in is 1 (the fan-in-share normalisation re-applied after any rewiring); the edge
count; the weight multiset. An arm violating any of these is refused before it runs.
- SEEDED: T062's population at width W (2.7).
- RANDOM (the control that is not the method): the DEGREE-PRESERVING rewire of the seeded table,
  performed WITHIN EACH SOURCE-SIGN CLASS (excitatory sources' edges rewired among excitatory sources,
  inhibitory among inhibitory) so every unit keeps its in-degree, its out-degree and its polarity; the
  weight multiset reassigned uniformly at random within each class; no self-edges or multi-edges;
  fan-in re-normalised. WHAT THIS NULL TESTS, stated narrowly: wiring information BEYOND the retained
  H01 marginals (degrees, signs, weight magnitudes) — which the two arms SHARE. It cannot tell "H01
  contributes nothing" from "H01's useful information is what both arms share"; every claim and the
  null disposition (2.9) are restricted to that component.
- WRONG (T004 decided): the DIRECTION-REVERSED laminar prior — every INTER-band edge of the seeded
  table has its pre and post swapped; intra-band edges are unchanged; the weight multiset and the edge
  count are preserved; polarity is re-applied from the new source's sign; fan-in re-normalised. This
  transposes the realised band-pair matrix exactly and swaps each unit's inter-band in- and out-degree
  (the degree DISTRIBUTION is preserved, per-unit degrees are not — disclosed). VALIDITY PRECONDITION
  (measured before the campaign, on the realised table): the reversed table's band-pair weight-share
  matrix must be farther from SEEDED's (PSI, 2.4) than RANDOM's realised matrix is from SEEDED's; if not
  — a near-symmetric laminar matrix — the WRONG arm is INVALID, red 2 is not read, and the record says
  which asymmetry it measured. Rejected: INVERTED E/I (sign is excluded from drift and the invariant
  SC-004 protects; a flipped population changes dynamics for a reason unrelated to the prior's content);
  SHUFFLED LAYER LABELS (no edge changes; dynamics identical to SEEDED; red 2 vacuous);
  UNIFORM-ACROSS-LAYERS (destroys the degree sequence, inseparable from RANDOM's question).
- PLANTED (the harness's POSITIVE CONTROL; never an arm of the verdict): a table wired to make the task
  easy — each drive class's input units densely connected to a disjoint block of non-driven units —
  under the same nuisance constraints. ONE predicate, the verdict's own: PLANTED beats RANDOM by the
  paired one-sided t > 1.895 over the 8 seeds. If not, the harness is BLIND and no verdict is read.
- FROZEN (the learner's own control): the SEEDED table with `lr = 0` — used by red 1 and by the
  learner's retention criterion (2.9).

**2.3 The task (T003 decided), the shortcut removed, the protocol frozen.** K drive classes (K fixed at
pilot time, 2.7), each a fixed 5 % subset of units drawn per seed (disjoint); a trial drives one class's
subset for T_trial ticks with a deterministic per-seed Poisson-shaped external current. THE READOUT
NEVER SEES A DRIVEN UNIT: the feature vector is the spike counts of the NON-driven units only (every
class's subset excluded), so the class code cannot pass through the drive itself. The readout is a
least-squares linear map. T003's required measurement, taken at pilot time and recorded in
decisions.md: a DRIVE-ONLY population (the same units, NO edges) under the same protocol must score at
chance on the held-out trials — the shortcut is measured absent, not assumed. Protocol per trial:
reset membranes, refractory state and eligibility → drive → collect spikes → PREDICT with the readout
fit on all PREVIOUS blocks (block 0 uses a readout fit on a warm-up block during which the learner is
OFF) → reward = +1 if correct else −1, delivered to THAT arm's learner → weights update. Reward VALUES
are per arm (each arm's own correctness); what the arms share is the seed's drive, classes, trial
order, split and reward TIMING. Held-out trials: learner OFF, no eligibility, no reward, no state
carried; the split is `marginal_holdout` per class within each block. Acceptance clause disclosed:
connectivity plausibly affects non-driven units' responses (the only path from drive to feature is
the graph), and the task is defined by drive + readout, not by H01. Rejected: next-category prediction
on the live percept stream (unreachable from a kernel population); C0's battery (not in `src`);
SC-013's criterion (a different mechanism).

**2.4 The seeded statistics — NEWLY CHOSEN graph-weight statistics, and what they do not measure.**
Two, both proportions BY WEIGHT: S1 `band_pair_weight_share`, the 7x7 matrix Σ|w| over edges a→b /
Σ|w| over all edges (49 cells; distance = `psi(current, seeded)`); S2 `band_fanin_e_fraction`, per band
b, Σ w⁺ into b / Σ|w| into b (seven scalars; distance = |current − seeded| per band, mean reported).
DISCLOSED: specs/011's original seeded statistics — layer occupancy, layer geometry, the excitatory
CELL fraction — are placement-side quantities that this row does not measure (they are a pinned null
on the object brain, 1.1a); no obligation on them is claimed closed. Both S1 and S2 are invariant to a
uniform positive rescaling of all weights and to redistribution within a band pair and sign class —
therefore drift is never read from them alone (2.5). `seeded` from the table at construction (read
once); `current` after learning; `direction` = sign of distance(end) − distance(mid-campaign). Sign is
never a statistic. A DriftRecord is produced for EVERY arm (C5).

**2.5 Diagnostics, the two reds, their fire conditions and controls (same command).** Every arm
reports, separately: learner EXECUTED (update calls), ATTEMPTED updates (nonzero eligibility × reward
events), EFFECTIVE weight change (count of `qweight` entries changed and the L1 of the change), and
the DriftRecords. A zero anywhere is read from these four lines, never inferred from one.
- RED 1 (never moves off the seed): the FROZEN arm (`lr = 0`, the constructed pin). The verdict FAILS
  naming the pin the harness itself applied: "seeding approach failed: PINNED (lr = 0 by construction):
  executed N, attempted M, effective 0, drift 0". Pinning is attributed ONLY to that verified
  intervention. On a learning arm, effective 0 with attempted > 0 is reported as "updates cancelled or
  below quantization", and attempted 0 as "no eligible spike pairs" — neither is called pinning.
  CONTROL: the SEEDED arm at the declared `lr` reads effective > 0 and distance > 0 on both statistics.
- RED 2 (worse than random): WRONG must UNDERPERFORM RANDOM — paired one-sided t over
  acc_random − acc_wrong > 1.895 — with the PLANTED predicate holding. Then the criterion "seeded beats
  random" re-run with WRONG in the seeded slot FAILS naming the prior. A non-significant difference is
  reported as "underperformance NOT ESTABLISHED" with no attribution, and red 2 is NOT FIRED (the
  verdict is then NOT MET on conjunct (d)). Q7 decided: SC-006's "beats random AND wrong" is TWO direct
  contrasts under the same rule (conjuncts (b) and (e)); SC-005's red is conjunct (d).

**2.6 The verdict rule (before any number).** Per seed s = 0..7, per arm, accuracy on the final block's
held-out trials. d_rs = acc_seeded − acc_random; d_ws = acc_seeded − acc_wrong; d_ps = acc_planted −
acc_random; d_fs = acc_seeded − acc_frozen. The statistic everywhere is the paired one-sided t over the
8 seeds against 1.895 (7 d.f.), ties counted as non-positive, the permutation false-positive rate over
the 2^8 sign flips printed beside each; a zero-variance set of differences is MET iff its common value
is > 0 (recorded as degenerate) and NOT MET otherwise. SC-006 MET iff ALL of:
(a) DRIFT — on the SEEDED arm, effective weight change > 0 and both statistics' distance > 0 on every
    seed, and red 1 fired;
(b) BEATS RANDOM — t(d_r) > 1.895;
(c) NOT BLIND — t(d_p) > 1.895 (else no verdict is read at all);
(d) RED 2 FIRED — t(acc_random − acc_wrong) > 1.895 and the WRONG arm was VALID (2.2);
(e) BEATS WRONG — t(d_w) > 1.895.
Each conjunct is printed MET / NOT MET / NOT READ with its numbers; no majority-of-seeds rule; the
per-seed table is printed. PRIOR SEPARATION (T071's item, defined here for the first time) = (b), (d)
and (e) together: the harness separates the true prior from its marginal-matched null and from its
direction-reversed corruption on one task.

**2.7 Constants and the width rule.** CHOSEN before the pilot, disclosed as P1 debt: T_trial = 50
ticks; 4 trials per class per block; C = 5 learning blocks after 1 warm-up block; held-out = 1 trial
per class per block via `marginal_holdout`; K = the smallest power of two ≥ 8 at which the no-learning
RANDOM arm scores below 0.9 held-out on the pilot seed (a ceiling margin; CHOSEN). The pilot (seed 0,
one warm-up + one learning block, the learner ON so its cost is in the basis) measures the tick at the
full ingested width and at the largest band-balanced induced subgraphs of 1/2, 1/4 and 1/8 of the
units (units kept by highest total degree within each band). W = the LARGEST of those widths whose
measured cost projects the full campaign (5 tables × 8 seeds × (1 + C) blocks × K × 4 trials × T_trial
ticks, plus the reds) under 6 hours on this host; the projection, the pilot's numbers and W are recorded
before the campaign. Host condition: the daemon UP and its cockpit LISTENING; no battery, no panel.

**2.8 What this row does NOT claim.** Nothing about the LIVE brain (the population is off the live
tick; the kernel's `wire` exemptions are NOT discharged by an instrument — recorded). Nothing about the
connectome beyond the c3 soma-to-soma subgraph T062 measured. Nothing about R-STDP being the brain's
learner — it is the minimal in-tree rule, merged and ported so the row can be measured. Nothing about
specs/011's placement-side statistics (2.4).

**2.9 Null clauses (fixed in advance, exhaustive over the verdict).**
- (c) NOT MET → INSTRUMENT BLIND: no verdict; the harness is repaired or deleted; nothing else moves.
- (c) MET, (b) NOT MET → the H01 WIRING BEYOND ITS MARGINALS does not beat its null on this task:
  US3's "beatable prior" recorded NOT MET for that component; `cortical_prior` and T062's artifact
  remain reference data; the seeded population constructor loses its "prior" role and is retained
  only as the harness's fixture.
- (c) MET, (b) MET, (d) or (e) NOT MET → separation INCOMPLETE: the conjunct named; no prior claim
  beyond (b) is made; T071 reads it as such.
- The LEARNER's own disposition, independent of the prior's: KEPT iff its attributed benefit holds —
  t(d_f) > 1.895 (learning beats the frozen twin on the seeded table) AND (a)'s effective change > 0;
  DELETED if t(d_f) ≤ 1.895 (a learner that moves weights without benefit is a null under P8 clauses
  3 and 5, and r_stdp's own docstring says weight movement is not charge); if effective change is 0 on
  every arm the learner is DELETED and red 1's reading is a harness defect, not a finding.
- Any combination not listed is recorded as such, conjunct by conjunct, and no verdict word is used.

**2.10 Order.** decisions.md + the Part F row → reds committed RED → the r_stdp merge with its
equivalence test → `kernel_plasticity` with its determinism, sign and LOCAL-claim tests → the harness
with its offline self-check (both reds, PLANTED, the invalid-control refusals) → the pilot (K, W, the
drive-only measurement, the WRONG validity precondition) → the campaign → the record → panel → fold →
ONE battery → closing record → push.

**2.11 The Codex read (2026-09-09, before commit) and what each defect changed.** (1) red 2 required
demonstrated underperformance, not a failure to establish superiority → 2.5; (2) SC-006's seeded-vs-
wrong contrast was silently dropped → conjunct (e); (3) the rewire changed source polarity and fan-in
totals → within-sign rewire, fan-in re-normalisation, sign-preserving learning, nuisance list in 2.2;
(4) red 1's diagnostic could not distinguish pinning from cancellation, quantization or absent pairs,
and both statistics are scale-invariant → the four diagnostic lines and attribution to the verified pin
only; (5) the transposed prior was unrealised and possibly negligible → the edge-reversal construction,
its preserved properties, and the validity precondition; (6) the drive was itself a class code →
non-driven-unit features, the drive-only measurement for T003; (7) the null tests wiring beyond the
marginals only → the claim and disposition narrowed; (8) two R-STDP implementations were not a merge →
one authoritative implementation with an equivalence test, else recorded debt; (9) the learner was
retained for moving weights → retention tied to t(d_f), disposition made exhaustive; (10) the reward
protocol was inconsistent → per-arm reward values, shared timing, the frozen trial order, held-out
isolation; (11) the pilot under-determined the allocation and PLANTED had two bars → constants fixed
before the pilot, one predicate, degenerate cases defined; (12) "proportions" over-claimed D1 → the
statistics named as newly chosen, the unmeasured originals disclosed.

### 2.12 Amendment BEFORE the campaign (2026-09-09; the pilot's readings are quoted, and none is discarded)

**What the pilot measured (`tools/t063_pilot.json`, harness sha256 `3d9a8ee5…8820da`, artifact
`a1d0491a…4a0549`, 14,215 units / 70,001 edges, W = 1.0 at 0.229 h projected against the 6 h ceiling).**
(1) 2.7's K rule selects NOTHING: the no-learning RANDOM arm scores 1.0000 held-out at K = 8 and K = 16, and
K = 32 does not fit (32 × 5 % > 100 %). The ceiling is structural, not the readout's: a 13-cell sweep over
class fraction 0.05 / 0.02 / 0.01 / 0.005 × K = 8 / 16 / 32 / 64 reads 1.0000 in every cell for SEEDED and for
RANDOM, and a second instrument that fits no readout — nearest-centroid over raw spike counts — reads 1.0000
for both arms while its own controls sit exactly at chance (drive-only 0.1250 from 0 feature spikes;
label-shuffled 0.1250). T003's drive-only measurement passed cleanly (0.0625 = chance at K = 16; 1.0000 with
edges): the class reaches the readout only through the graph, and over ~14k sparsely wired units a class's
REACHABLE SET is a unique signature that every wiring produces. (2) 2.2's validity precondition FAILS for the
direction-reversed table: PSI(WRONG, SEEDED) = 0.032389 against PSI(RANDOM, SEEDED) = 2.281271; the band-pair
weight-share matrix carries 0.726520 of its mass on the diagonal (intra-band edges, untouched by a reversal by
construction), 29.6 % of edges are inter-band, and `Σ|M − Mᵀ|/2 = 0.041883` of a matrix summing to 1. No
transpose-based corruption can be far from SEEDED under S1 on this connectome; T004's decided candidate is
structurally near-null, not unlucky. (3) The host clause's "two consecutive 90 s windows within 6 %" cannot be
met by this daemon: five consecutive windows read 112.0 / 122.9 / 106.3 / 133.6 / 117.8 CPU-s per 90 s, the
ordinary wake/sleep duty cycle, no monotone trend. (4) The self-check fixture predicted the saturation
(`out_degree` 2 gave every arm a perfect class code) and the builder read it as a fixture setting; recorded.

**Replacement task instrument (T003 re-decided; the task's DEFINITION is unchanged: K drive classes on fixed
5 % subsets, features from NON-driven units only, a least-squares readout, `marginal_holdout` per class,
the frozen trial protocol of 2.3).** What changes is the readout's CAPACITY, chosen by a measured rule instead
of the class count: the feature vector is a FIXED per-seed random projection (Gaussian, orthonormalised) of the
non-driven units' spike counts to D dimensions; K = 8; D is the LARGEST of {4, 8, 16, 32, 64, 128} at which the
no-learning RANDOM arm scores below 0.9 held-out on the pilot seed (a capacity guard on the null, applied
before any arm is compared; the ceiling margin 0.9 is CHOSEN, as before). If NO D in that ladder puts RANDOM
under 0.9, the row is REFUSED ON MEASUREMENT with the finding that this population admits no readout capacity
at which its wiring can be separated from a marginal-matched null by this protocol — recorded, not softened.
The drive-only control is re-measured at the chosen D and must read chance. Rejected: overlapping class
subsets (leaves the reachable-set signature intact where the subsets differ; the same saturation one level
down); a graded regression target (changes the task's kind after the numbers, not its capacity).

**Replacement wrong prior (T004 re-decided).** WRONG = the BAND-PERMUTED-TARGETS table: every edge a→b keeps
its pre and its synapse count and gets a post drawn from band π(band(b)) with the same within-band degree
rank, where π reverses cortical depth (Layer 1↔Layer 6, Layer 2↔Layer 5, Layer 3↔Layer 4, White matter fixed);
polarity re-applied from the pre; fan-in re-normalised; the synapse-count multiset and the edge count
preserved. It corrupts what the diagonal carries — the intra-band mass moves to the mirrored band — so it is
far from SEEDED under S1 wherever band masses differ (Layer 2 4,709 units against Layer 6 1,040). It falsifies
the claim red 2 is FOR: that the laminar IDENTITY of a synapse's target carries information. The same validity
precondition applies unchanged (its PSI from SEEDED must exceed RANDOM's) and is measured before the campaign;
the reversed-laminar reading above stays in the record as the candidate this connectome could not distinguish.
T004's three earlier rejections stand.

**Three readings of frozen text, disclosed.** The settling clause of 2.7 is read as the ABSENCE of a
monotone trend across at least three 90 s windows, the windows recorded beside every timing number and every
timing taken as the minimum of repeats (the daemon also restarts between readings; the PID beside a number
is recorded). "The weight multiset" of 2.2 is realised as the per-edge SYNAPSE-COUNT multiset, the quantity the
weight is computed from, with the fan-in normalisation re-applied on top (a weight multiset and a re-applied
normalisation cannot both be preserved). PLANTED may repeat a (pre, post) pair: "no self-edges or
multi-edges" is a property 2.2 states for the RANDOM rewire, and the unique-pair form leaves the positive
control unbuildable wherever the edge count exceeds the class×block product.

**What is unchanged.** 2.1, 2.4, 2.5, 2.6, 2.8, 2.9, 2.10, 2.11; every gate, red and control; the learner's own
retention criterion (the 64-unit fixture read the FROZEN arm ahead of SEEDED at t ≈ −6.0 — a fixture
observation the campaign will read evidentially). Authorized by this amendment: the pilot re-run under the D
rule and the new WRONG arm, then the campaign iff the guard selects a D and the WRONG arm is valid; the pilot
readings above stay in the record beside the re-run's.

## 3. Results (written after the runs; nothing above this line was edited after §2.12)

### 3.1 The reds, RUN RED, the merges, the learner, the harness, and the pilot that stopped the first campaign

The reds were committed before any mechanism existed (`e63cb98`: a collection error on the harness, 16 failing tests
on the learner and the merge, one deliberate PASS pinning the incumbent R-STDP trajectory) and again before each later
mechanism (`91882fa` for §2.12's instrument; `0c5c636` and `b327e1c` for the host probe's two defects). `decisions.md`
was CREATED with T003 and T004 decided and the Part F row `H-R1` appended (`3c79573`). The R-STDP rule was MERGED into
one authoritative implementation in `basal_ganglia/r_stdp.py` (`bccb89a`): the per-object neuron calls the module-level
operations (a spy test proves it goes through them), the incumbent's trajectory is byte-identical before and after
(`_run(7)` = 0.5587212259170016 / 0.5453367971886981 / 0.4783055031868412), and its own self-check reproduces every
number it printed before the merge. `substrate/kernel_plasticity.py` (`b4e981b`) drove that one rule over the kernel's
bundle table: equivalence with the per-object form on one spike/reward sequence within one quantization step
(max |Δw| 7.104e-06 against a step of 1.526e-05, clamp not fired; the test is not vacuous — doubling `a_plus` breaks
it); sign preserved through learning (the clamp acted on 44 entries and removing it produced 10 flips — the invariant
is reachable); byte-equal weights for the same seed and reward sequence (sha256 `c66b0f0f…`, a flipped sequence
`53e6c943…`); the four diagnostic lines separated (lr = 0: executed 2, attempted 48, effective 0; lr = 1e-12:
attempted 24, effective 0, reported as below quantization, NOT pinning). `DriftMonitor`'s PSI was lifted to module
scope (`d0d4c53`). `cortex/prior_drift_harness.py` (`548a17a`) carried both reds, PLANTED, the WRONG validity
precondition and the nuisance refusals in its offline self-check (64 units, < 4 s): red 1 fired ("PINNED (lr = 0 by
construction): executed 48, attempted 151, effective 0, drift 0"), PLANTED beat RANDOM (t +3.53, FPR 0.0625), WRONG
valid (PSI 8.86 vs 4.74), SEEDED moved on every seed, and the three nuisance refusals fired. The census gates caught the
harness's `_learner` symbol without a declaration (`dc6d52e`; obs #2164's shape a second time).

**The first pilot STOPPED the campaign, as 2.7 wrote it to.** At the full ingested width (14,215 units, 70,001 edges;
W = 1.0 projected 0.229 h against the 6 h ceiling) the K rule selected nothing: the no-learning RANDOM arm decoded
K = 8 and K = 16 disjoint drive classes at 1.0000 held-out (K = 32 does not fit), a 13-cell sweep over class fraction
0.05 / 0.02 / 0.01 / 0.005 × K = 8 / 16 / 32 / 64 read 1.0000 in every cell on both arms, and a nearest-centroid
instrument with no readout agreed (1.0000 on both arms; its own controls exactly at chance). T003's drive-only
measurement passed cleanly (0.0625 = chance with no edges; 1.0000 with edges): the class reaches the readout only
through the graph, and over ~14k sparse units a class's reachable set is a signature every wiring produces. The
direction-reversed WRONG arm failed 2.2's validity precondition (PSI 0.032389 against RANDOM's 2.281271; 0.726520 of
the band-pair weight mass is intra-band and a transpose cannot move it). No campaign ran; no known-vacuous criterion
was executed. The fixture had predicted this — `out_degree` 2 gave every arm a perfect class code and was tuned to 4
so PLANTED could fire — and the builder read it as a fixture setting (obs #2165). Amendment 2.12 (`aba9a71`) replaced
the readout's CAPACITY rule and the wrong prior before any campaign.

**Three defects in the host instrument, each caught red-first (`0c5c636`/`9e92768`, `b327e1c`/`7caea77`):** the
settling probe passed on a DEAD reading (`[-515.156, 0.0, 0.0]` → settled; the daemon had restarted under it and the
cached handle differenced a corpse; now every window must be positive, the PID is recorded per window, the listener is
re-resolved at both ends); three windows rejected a settled host one time in three (exactly 2 of n! orderings are
monotone; runs now take 5 windows, the amendment's floor of 3 unchanged); and the campaign's host clause is NOT MET
on the pinned run — `learn_daemon.bat`'s supervisor restarts the daemon on a cycle (~15 min) shorter than the probe
plus the campaign, four PIDs held 8137 in the session, `daemon_restarted = True`. Measured, not argued: the campaign
was executed three times across four daemon PIDs and two settling outcomes and every per-seed accuracy, every
executed / attempted / effective / L1 count and all five t-values came back identical; the third run is pinned.

### 3.2 The pilot under 2.12 and the campaign (`tools/t063_pilot.json`, `tools/t063_campaign.json`)

Pilot (git `7caea77`, harness sha256 `93f6fa58…89fa095`, artifact `a1d0491a…4a0549`): the D ladder on the no-learning
RANDOM arm at K = 8 (8,535 non-driven features projected): D = 4 → 0.7500; D = 8 / 16 / 32 / 64 / 128 → 1.0000. D = 4
selected — the largest under the 0.9 ceiling and the ladder's bottom rung. Drive-only at D = 4: 0.1250 = chance (0.6250
with edges). WRONG (band-permuted targets) VALID: PSI 6.175870 against RANDOM's 2.281271 (0 off-band fallbacks). Tick
cost with the learner ON: 1.0789 / 0.5874 / 0.2969 / 0.1564 ms per tick at W = 1.0 / 0.5 / 0.25 / 0.125 → 0.115 h
projected; W = 1.0. Host windows 82.2 / 84.3 / 86.5 / 81.9 / 69.3 CPU-s on one PID, settled.

Campaign (git `f37fe90`; K = 8, D = 4, W = 1.0; 5 tables × 8 seeds; wall 589.7 s):

| conjunct | reading | numbers |
|---|---|---|
| (a) DRIFT | MET | red 1 fired on the pin (executed 120, effective 0, drift 0 on every seed; attempted 174,781–184,936); SEEDED S1 distance 0.107–0.172, effective weight change 435k–654k entries (L1 232k–398k) on every seed |
| (b) BEATS RANDOM | NOT MET | paired one-sided t = −0.2277 (FPR 0.0781) |
| (c) NOT BLIND | MET | PLANTED − RANDOM t = +9.2601 (FPR 0.0508) — the verdict is read |
| (d) RED 2 FIRED | NOT MET | t(random − wrong) = −2.5459 (FPR 0.0391); WRONG VALID on all 8 seeds (PSI 6.175870 vs 2.2711–2.3029) — the corrupted prior OUTPERFORMS the null |
| (e) BEATS WRONG | NOT MET | t = −5.2271 (FPR 0.0312) |
| the learner's own criterion | NOT MET | t(d_f) = −13.0154 (FPR 0.0430); per-seed differences −0.75 / −0.375 / −0.625 / −0.75 / −0.875 / −0.625 / −0.75 / −0.75 — the FROZEN twin beats the learning arm on every seed |

DriftRecords (S1, PSI to the seeded table): SEEDED 0.107–0.172, RANDOM 0.029–0.086, WRONG 0.142–0.252, PLANTED
0.179–0.297, FROZEN exactly 0 with direction 0 on all 8 seeds. The per-seed table, every contrast's differences and
every diagnostic line are in the pin with the harness sha256s, the artifact sha256, the git head and the host windows.

**APPENDED 2026-09-10 (T063 panel of 2026-09-09) — the grid these t-values sit on, and the campaign's own host
reading.** Every arm accuracy above is k/8. Per 2.6 and 2.7 the score is the FINAL block's held-out trials alone — 1 per
class, K = 8, so 8 trials per seed; the other five blocks' 40 held-out trials are scored and overwritten, and never
enter the readout's training pool. In TRIALS OUT OF 8 the mean differences are: (b) seeded − random −0.125,
(c) planted − random +3.5, (d) random − wrong −1.25, (e) seeded − wrong −1.375, and the learner's seeded − frozen −5.5;
the per-seed diff vectors for all five contrasts are in `tools/t063_campaign.json` under `verdict.contrasts[*].diffs`.
This is a REPORTING correction, not a deviation: 2.6 pre-registers "accuracy on the final block's held-out trials" and
2.7 pre-registers 1 held-out trial per class per block, so the harness implements the frozen text literally. And the
campaign's own host windows, which this section quoted for the pilot only: **0.0 / 113.25 / 106.281 / 122.344 / 123.984
CPU-s on PIDs none / 34212 / 34212 / 34212 / 34212, daemon_restarted = True, NOT settled; host 29380 → host_end
34212** — the first window's owner vanished under the probe, which the repaired rule reports as `none 0` and rejects.
3.5(4)'s attribution of the clause failure to the restart is complete as it stands and is not widened here.

### 3.3 The verdict, by the rule of record

SC-006 is NOT MET: conjuncts (a) and (c) MET, (b), (d) and (e) NOT MET. By 2.9's enumeration: (c) MET and (b) NOT MET →
the H01 WIRING BEYOND ITS MARGINALS does not beat its degree-, sign- and weight-matched null on this task; US3's
"beatable prior" is recorded NOT MET for that component; `cortical_prior` and T062's artifact remain reference data; the
seeded population constructor loses its "prior" role and is retained only as the harness's fixture. The learner: t(d_f)
≤ 1.895 → DELETED (a learner that moves weights — at scale — and makes the readout worse on every seed is a null under
P8 clauses 3 and 5, and r_stdp's own docstring says weight movement is not charge). PRIOR SEPARATION (T071's item, as
2.6 defined it) is NOT MET.

**A finding 2.9 had no clause for, recorded as its own:** (d) and (e) fail in the OPPOSITE direction — the
depth-reversed corrupted prior decodes BETTER than both the true prior (t −5.23) and the marginal-matched null
(t −2.55) at the readout capacity the guard selected (D = 4). On this task the H01 laminar targets are not merely
uninformative; the wiring that sends each band's synapses to the mirrored band is more decodable. Nothing here says
why; the record names it and does not explain it.

**APPENDED 2026-09-10 (T063 panel of 2026-09-09) — two withdrawals from the paragraph above, which is left readable
exactly as written so the audit trail survives.** FIRST: contrast (e) differs from RANDOM along TWO axes — laminar
target identity AND fan-in concentration (3.5 item 12) — so the inference "the H01 laminar targets are not merely
uninformative" is WITHDRAWN. What the contrast carries is only this: at D = 4 the band-permuted table is more decodable
than both SEEDED and RANDOM, and this record does not isolate which of the two axes carries the sign. SECOND: the
clause "the wiring that sends each band's synapses to the mirrored band is more decodable" over-reaches as a claim
about the TABLE and is withdrawn too. Every accuracy behind (d) and (e) was taken with the learner ON, and this arc has
no reading that separates the band-permuted table's decodability from its interaction with the learner. What WAS
measured is the sentence before it: the band-permuted arm SCORED higher at D = 4 under the learner (t −5.23 against
SEEDED, −2.55 against RANDOM). An off-pin no-learner reading of all four arms at D = 4 over the same 8 seeds cannot
decide it either way — its positive control reads PLANTED − RANDOM t = +0.8143 against the 1.895 bar, so by 2.9 that
configuration is INSTRUMENT BLIND and no verdict is read from it, consistent with 3.4's disclosure that the learner had
supplied part of PLANTED's margin. Recorded here so the next reader does not re-take that reading as the panel did.
**"Nothing here says why" STANDS**, and no mechanism is put in its place.

### 3.4 The disposition applied

Applied as §2.9 prescribes, in seven commits (`1fb01ba`..`594b95d`), the harness's no-learner form landing BEFORE the
deletion so no commit imports a module that no longer exists: (1) `cortex/prior_drift_harness.py` kept as a NO-LEARNER
instrument (`1fb01ba` the reds RED, `4c6c4cf`, `077d647` the census gate's catch of `no_learner_holds`): every arm runs as
the frozen form; the drift half returns the harness-defect refusal §2.9 pre-registered ("if effective change is 0 on
every arm the learner is DELETED and red 1's reading is a harness defect, not a finding"), AUTHORED BY THE DISPOSITION
to stand in red 1's §2.5 slot — it carries §2.5's four diagnostic lines, never says PINNED, and closes with the module's
`NO_LEARNER` constant, quoted here exactly: "no learner in the tree -- substrate/kernel_plasticity DELETED on its own
criterion 2026-09-09 (T063's record, section 3): drift is not measurable; FR-008/FR-009 (specs/011) remain OPEN" — and
reads no drift verdict; the contrasts (b)(c)(d)(e), the D guard, the WRONG validity precondition, the
nuisance refusals and PLANTED all stand; `--campaign` still runs the no-learning contrasts; the pinned campaign is the
record of the learner's run and is neither re-run nor edited. (2) `substrate/kernel_plasticity.py` (474 lines) and its
tests DELETED with its REGIONS / CORE / dormancy registrations (`2da3bb5`); `tests/test_r_stdp_kernel_equivalence.py`
keeps the incumbent's byte-exact trajectory pin, the module-level and spy tests, and gains a test that the deleted
kernel form cannot be imported (a second substrate cannot return quietly); `tools/gradient_claim_baseline.json` untouched
(governed rules_seen 174 → 172 across `2da3bb5` itself — the two keys removed are
`substrate/kernel_plasticity.py::KernelPlasticity.reward` and `::update_eligibility`; the disposition's span reads 173 at
`963799e`, the commit before `1fb01ba`, to 172 at the head, its +1 being `cortex/prior_drift_harness.py::no_learner_holds`,
which ENTERS the governed set at `4c6c4cf` (UNDETERMINABLE, gate RED) and is declared NOT_A_RULE at `077d647` with the
count already 174; floor 149). (3) The R-STDP merge KEPT as the one implementation; its two
container branches written for the kernel form measured ORPHANED by two instruments with a positive control taken
before the deletion (decay numpy 38,400 / deltas numpy 1,920 entries with the kernel form; numpy 0 and tripwire 0
after, list 297 / 4,476) and deleted (`96082cc`). (4) `h01_ingest`'s dormancy row re-dated by appended succession
(`371d607`): the population is the harness's FIXTURE; the "prior" role withdrawn on T063's reading; the kernel_tick /
kernel_state `wire` rows untouched. (5) The dated annotations (`594b95d`): PLAN-FINAL's T062/T063 bullet, Part F row
H-R1 (FIRED, RESOLVED — its reds are not re-runnable as written without a learner), the T071 exit line ("prior
separation" NOT MET, a measured negative); tasks.md T063 `[X]` with its DONE note and T071's note; spec.md SC-006 and
US3; specs/011 tasks.md T010–T014, T031, T039 (what T063 closes and what it leaves OPEN: FR-008/FR-009 without a
learner); `decisions.md`'s OUTCOME section.

Two readings the disposition took, disclosed: after the learner's removal the fixture's PLANTED control at
`out_degree = 4` no longer cleared the bar (t +0.7977 — the learner had supplied part of PLANTED's margin); the
docstring's own rule (the smallest density that clears while the base task stays hard) re-applied to a fresh sweep
chose 8 (t +4.4063, RANDOM 0.5469; 12 and 16 rejected with both base arms at chance); both sweeps stay in the
docstring, their difference being the finding. And the criterion as a whole gained a third reading, NOT READ, for
"every conjunct this instrument can read held and drift could not be measured" — §2.6 defines the readings per
conjunct, not for the whole; recorded as the builder's judgment call. One stale index entry surfaced: the graphify
guard's inventory cites `substrate/lake_engine.py L769` for a symbol that is not in the tree (the file is 561 lines) —
a carried opening, not this arc's.

**APPENDED 2026-09-10 (T063 panel of 2026-09-09) — WHAT "STAND" AND "SURVIVES" MEAN, MEASURED ON THE ARTIFACT.** On
`data/h01/h01_c3_graph.npz` at the pinned K = 8, D = 4, W = 1.0 over seeds 0–7 the surviving no-learner harness reads
**(c) NOT MET at t = +0.8143** (permutation FPR 0.0469) — per-arm means seeded 0.8750, random 0.8281, wrong 0.8281,
planted 0.8750, with FROZEN identical to SEEDED and d_f = 0 on all 8 seeds, so the identity control holds and the
reading is on the pinned path. `verdict()` therefore prints SC-006 = **NO VERDICT** with (a), (b), (d) and (e) NOT
READ: "no verdict is read: the harness is BLIND". The other contrasts at lr = 0 read (b) +1.0000, (d) 0.0000,
(e) +1.0000, none clearing 1.895, with every arm between 0.75 and 1.00 — at CEILING, which adjudicates nothing in
either direction, and (c)'s pinned margin is NOT called damage on the strength of it. So what SURVIVES the deletion is
RETAINED AS A MECHANISM, and on this population at this capacity the instrument REFUSES rather than reads — §2.2's
pre-registered disposition FIRING, not a new defect. The consequence a successor inherits: (b)(c)(d)(e) CANNOT be
re-read on H01 with this instrument as-is, and re-reading them needs a NEW pre-registration of the capacity guard. The
bare self-check's green is the 64-unit fixture at `out_degree = 8`, which does not cover the artifact. REPRODUCTION,
and it is deliberately not the CLI: direct calls to `band_balanced_subgraph` / `drive_classes` / `build_arms` /
`wrong_validity` / `run_arm` / `verdict` at `_pin_stages()[-1]`'s constants with `NORD_HOME` on a scratch dir — and
NEVER `--campaign`, because `run_campaign()` writes `tools/t063_campaign.json` unconditionally and would destroy the
record of the learner's run. The pinned campaign's own (c) = +9.2601 was read WITH the learner and STANDS.

**APPENDED 2026-09-10 (T063 panel of 2026-09-09) — the mirror of the H-R1 correction.** Item (5) above records
PLAN-FINAL's H-R1 cell as saying its reds are "not re-runnable as written without a learner". That over-reaches and is
corrected there by an appended sentence rather than removed: only RED 1 lost its subject — its pin stood against a
learning arm and there is no learning arm (the harness's own self-check docstring reads "WHAT IS GONE: red 1"). RED 2's
contrast, the paired one-sided t over acc_random − acc_wrong against the 1.895 bar, needs NO learner, is conjunct (d),
is named among what survives in the module docstring, and `--campaign` still computes it. Two qualifications ride with
that: red 2 as literally written names the REVERSED-LAMINAR table for its validity precondition, and 2.12 retired that
table on measurement in favour of the BAND-PERMUTED-TARGETS arm, so what remains runnable is red 2's PREDICATE under
2.12's arm; and an evidential re-reading requires `--campaign`, which must not be taken against the pin. No self-check
number enters either site — 3.5(9) already fixes that the offline red-2 reading is a predicate on a constructed input,
not an evidential reading.

### 3.5 Deviations from the pre-registration, each with its reason (dated 2026-09-09)

1. 2.7's K rule had no satisfying K; 2.2's reversed-laminar WRONG arm failed its own precondition; both replaced by
   the dated amendment 2.12 BEFORE any campaign (the numbers seen are quoted there).
2. The D ladder is read on ONE seed with 8 held-out trials, so its grid is eighths: 0.7500 is 6/8 against a ceiling
   admitting 7/8, and one trial flip changes the selection. Disclosed; a successor reads the ladder over ≥ 3 seeds.
3. D = 4 is the ladder's bottom rung: "the largest D under the ceiling" degenerated to "the only one". Had D = 4 also
   read ≥ 0.9 the row would have been refused; the selected capacity sits at the floor.
4. The host clause is NOT MET on the pinned campaign (the daemon's supervisor restarts it on a cycle shorter than the
   measurement); the verdict is host-independent by the three identical executions of 3.1, and the timing numbers are
   declared host-contended by construction.
5. The settling clause's reading (2.12) was itself defective twice — a dead reading passed; three windows rejected a
   settled host — and was repaired red-first before the pinned run.
6. `e63cb98`'s trailer reads "the arc battery" (missing the possessive); amending is forbidden; recorded.
7. Two assertions in the first red file were wrong about their own subject (an empty DriftRecord tuple; the "does not
   say pinning" check rejecting the disclaimer inside the refusing message) and were corrected in `548a17a`, stated in
   that commit.
8. The self-check fixture's `out_degree = 4` was chosen by measurement so PLANTED could fire; it was the campaign's
   blocker in miniature and was not read as such until the pilot (obs #2165).
9. Red 2 in the offline self-check is a predicate on a constructed input, not an evidential reading (the evidential
   reading is the campaign's).
10. "The weight multiset", "no multi-edges" and the settling clause each admitted two readings of frozen text; the
    readings taken are in 2.12.
11. The kernel's `wire` exemptions are NOT discharged by this instrument, and the harness's fixture (T062's population)
    is not a live reader: `tests/test_placement_charge_null.py` stays green.

*Items 12–17 are APPENDED 2026-09-10 by the T063 panel of 2026-09-09. Items 1–11 above are untouched.*

12. (2026-09-10) **2.12's replacement WRONG arm is not degree-matched to RANDOM, and 2.12 made no degree disclosure
    where 2.2 had made one for the retired arm.** RANDOM preserves every unit's in-degree exactly (14,215/14,215
    unchanged); the band-permuted arm's quantile rank map is many-to-one into every smaller mirrored band, leaving only
    3,701/14,215 in-degrees unchanged — max in-degree 67 → 190, receiving units 11,227 → 8,222, top-1% fan-in share
    0.0126 → 0.0173, out-degree untouched. The many-to-one map is disclosed in `decisions.md`'s T004 reading (1) and in
    `band_rank_map`'s docstring, but only as the reason multi-edges are permitted; its degree consequence was not
    stated. Degrees are not among 2.2's enumerated nuisance invariants, so `check_nuisance` is correct not to test them:
    this is a DISCLOSURE GAP and an interpretive confound, not a nuisance violation. What a successor owes to separate
    the two axes: a degree-matched laminar corruption, or a degree-only control alongside the laminar corruption, read
    in a NON-saturated regime — the separating control attempted during the panel (WRONG's in-degree sequence with
    laminar identity destroyed) was uninformative here because every arm saturated, which is the separate blocker of
    item 15's neighbour, 3.4's appended paragraph. SC-006's verdict is unaffected: it rests on SEEDED-vs-RANDOM, and
    both arms preserve in-degree exactly.
13. (2026-09-10) **Two of §2.12's quoted pilot readings are NOT in `tools/t063_pilot.json` and no committed code
    produces them.** The 13-cell class-fraction × K sweep and the nearest-centroid second instrument were taken by
    uncommitted out-of-harness probes and are not reproducible from the tree. The amendment's DECISION rests only on
    pinned stage-0 fields: `K_rule.satisfied = false` (K = 8 and K = 16 both 1.0000, K = 32 does not fit), `drive_only`
    (0.0625 = chance with no edges, 1.0000 seeded) and `wrong_validity` (PSI 0.032389 vs 2.281271). The two unpinned
    readings corroborate the "structural, not the readout's" explanation and carry no decision on their own. §2.12 and
    §3.1 are left byte-identical; the same disclosure is appended to `specs/011`'s decisions.md, naming its lines 55–60.
14. (2026-09-10) **`594b95d`'s subject claims the dated annotations landed "at every site the verdict touches". It did
    not:** the census enumerated in §3.4(5) missed `quickstart.md` §S2 step 2 and `tasks.md:140`, both found by the
    T063 panel and annotated at `6d2a309`. The subject line is left as written — amending a pushed commit is forbidden
    (item 6 is the precedent) — and this line is the correction. The miss was an OMISSION, not a declared scope.
15. (2026-09-10) **CORRECTION to §3.1 and §3.5(4): the "~15 min supervisor cycle" is not a property of this host.**
    `learn_daemon.bat` has no timer — it relaunches only when the daemon exits, 8 s later — and
    `substrate/learn_daemon.py` has exactly one `os._exit(0)`, reachable only from `_head() != start_head`. The log tail
    holds 254 clean auto-update exits against 255 "new commit detected" lines, and all 30 relaunches in
    `1d4602c..a4c2504` land 1–27 s after one of this arc's own 30 commits, with gaps from 44.7 s to 22 m 16 s. The
    pinned campaign's `daemon_restarted = True` was produced by THIS SESSION committing `f37fe90` at 21:58:40, seconds
    before the probe: `host.daemon_pid` 29380 exited cleanly at 21:58:45.59, settling window 1 reads pid "none", windows
    2–5 pid 34212. What is NOT MET is 2.12(c)'s ONE-LIVE-DAEMON condition, and it IS meetable — the run needs a
    commit-free window of 450 s probe + 589.66 s campaign ≈ 17.4 min, and the session controls the trigger. 2.7's
    literal "two consecutive 90 s windows within 6 %" remains NOT MEETABLE for the separate, measured reason of the
    duty cycle (106.3–133.6 CPU-s per 90 s, ±11.5 %). `f37fe90`'s message and `tools/t063_campaign.json`'s `host_note`
    carry the superseded attribution and are deliberately NOT edited: the commit is pushed and amending is forbidden,
    and the JSON is the pin. Item 4's text stands as written; the `restarted` guard and its red are untouched, because
    a probe whose subject changed is not a reading whatever moved the subject.
16. (2026-09-10) **§2.9 and §3.3 label the learner's outcome "a null under P8 clauses 3 and 5"; clause 5 is
    mis-cited.** Clause 3 (attribution — no gain to attribute against 435,119–654,321 moved entries, L1
    231,865.22–397,915.81) applies as cited. Clause 5's words are "an invented method that measures nothing", and the
    outcome was not nothing: seeded mean 0.1875 against frozen 0.8750, t(d_f) = −13.0154, the frozen twin ahead on 8 of
    8 seeds. The P8 line that governs is the transferable one — "it was measured and it did not work". The disposition
    is UNCHANGED: DELETED on 2.9's pre-registered t(d_f) ≤ 1.895 criterion, and §3.6 already states it in these terms.
    2.9:207 is frozen and untouched; §3.3's sentence, the pin's `verdict.learner.note`, `r_stdp.py`'s appended
    paragraph and the `NO_LEARNER` constant with its tests are all left exactly as they are.
17. (2026-09-10) **SEEDED, FROZEN and WRONG are deterministic constructions on the single W-table** (`arm_seeded(art)`
    and `arm_wrong(art)` take no seed); only RANDOM and PLANTED are re-drawn per seed. The band-permuted corruption
    therefore has exactly ONE realisation across the campaign — psi_wrong = 6.175869708803301 and the whole
    `band_masses` block are identical on all 8 seeds, against RANDOM's 2.2711–2.3029. So (e)'s eight paired differences
    vary only in the per-seed drive / class-subset / projection draw, and (d)'s vary in that PLUS the RANDOM draw;
    neither replicates the corruption. The reading of §3.3's negative-direction finding is about THIS band-permuted
    table on those eight task draws, and generalises over neither corruptions of this kind nor other choices of π.
    §2.2 and §2.12 are left byte-unchanged.

### 3.6 What this row does NOT claim

Not that the H01 prior is useless: that it does not beat its marginal-matched null on ONE task (K = 8 drive classes
decoded from a 4-dimensional projection of non-driven units' spike counts) at the one capacity the guard selected. Not
that R-STDP cannot learn on a kernel: that this reward (per-trial readout correctness, ±1) on this population made the
readout worse than a frozen twin on every seed, and the pre-registered criterion deletes it. Not that drift was measured
in specs/011's sense: the drift-eligible statistics here were newly chosen graph-weight statistics (2.4) and, with the
learner deleted, drift on the kernel has no mechanism again — FR-008 / FR-009 remain OPEN. Not a claim about the live
brain. Not a claim about why the corrupted prior decodes better.

*Four limitations APPENDED 2026-09-10 by the T063 panel of 2026-09-09. Nothing above this line is edited.*

- (2026-09-10) **Both pins carry a `host_note` module constant whose CPU windows and clause-reading are the 2.7
  host-clause measurement of 2026-09-09 on daemon PID 27228, not the pinned run's.** Each run's settling reading is its
  in-band `settling` block — settled = false, daemon_restarted = true on the campaign; settled = true on pilot stage 1 —
  and where the two differ the IN-BAND BLOCK GOVERNS. Neither pin is touched or regenerated; the constant now carries a
  static prefix saying the same thing (`ad98d47`), with no per-run outcome written into it.
- (2026-09-10) **The two arms of (b) sit at or beside the chance floor in the pinned campaign.** The per-arm reading
  against chance = 1/K, and what it does and does not license, is §3.7's opening block; it is a POST-HOC reading, not a
  pre-registered conjunct, since 2.6 defines only arm-vs-arm differences. Attribution only where it is controlled:
  SEEDED's collapse is attributed to its FROZEN twin (t(d_f) = −13.0154, the twin ahead on every seed); the pin holds
  NO frozen twin for RANDOM, whose only no-learning reading is the pilot's single-seed D-ladder (0.7500 at D = 4), so
  RANDOM's cause is NOT CONTROLLED and "the learner drove both arms to chance" is not written anywhere in this record.
- (2026-09-10) **2.12's D guard is a ceiling with no floor, and it is read in a configuration the pinned verdict did
  not run.** The ladder calls the arm at lr = 0.0 over ONE block, while the pinned campaign's non-FROZEN arms ran at
  lr = LR over five. 2.12's own wording ("the no-learning RANDOM arm") discloses the learner half; the one-block half
  was not disclosed until this line. Filed as a LIMITATION and NOT a deviation — 2.12 was implemented literally. The
  selection is corroborated inside the pin: at the selected D = 4 the guard read no-learner RANDOM 0.7500 at one block,
  and the pin's only no-learner five-block arm, FROZEN (the seeded table at lr = 0), reads 0.8750 — under the same 0.9
  ceiling. The learning arms' 0.1875–0.3594 at that rung is the LEARNER's effect, already read by the learner's own
  retention criterion (t(d_f) = −13.0154, DELETED), not evidence of a mis-selected capacity. A successor's guard is
  read in the configuration the verdict runs, and carries a floor as well as a ceiling.
- (2026-09-10) **This row does not claim any of its magnitudes is large.** |t| > 5 on (e) is one and a bit held-out
  trials out of eight. A larger per-seed held-out sample needs its own pre-registration, because 2.6 is frozen; and if
  a successor pools, it pools only the C = 5 SCORED learning blocks, NEVER the warm-up block, where `predict(None, …)`
  returns class 0 and every arm scores a deterministic 1/K.
- (2026-09-14, Codex closing report 2026-09-10, reproduced 2026-09-10) The line above reading 'The learning
  arms' 0.1875-0.3594 at that rung is the LEARNER's effect' is licensed for SEEDED alone (its FROZEN twin, t(d_f) -13.0154, 8
  of 8 seeds); RANDOM (0.2031) and WRONG (0.3594) have no frozen twin in the pin and their low scores are attributed to nothing
  here. The sentence stands as the record of the fold's wording; this line corrects its scope.

### 3.7 The panel fold (appended 2026-09-10)

The panel ran 2026-09-09 against head `a4c2504`: three lenses and 28 refuters. **22 findings STOOD and 6 were
REFUTED — and every one of the 22 filed fixes was judged WRONG by its own refuter.** So this fold adopts the
REFUTERS' minimal fixes and not the panel's (obs #2146: a confirmed finding can carry a fix that introduces a defect).
Nothing below re-runs the campaign or the pilot; `tools/t063_campaign.json` and `tools/t063_pilot.json` are
byte-untouched, and sections 1–2.12 above are byte-untouched.

**WHERE THE ARMS WERE WHEN THE VERDICT WAS READ.** Per-arm MEAN HELD-OUT ACCURACY in the pinned campaign, against
chance = 1/K = 0.1250 on the 1/8 grid:

| arm | mean accuracy | held-out trials above chance | t vs chance | FPR | above chance |
|---|---|---|---|---|---|
| SEEDED | 0.1875 | +0.50 | +1.5275 | 0.0312 | **no** |
| RANDOM | 0.2031 | +0.62 | +1.9296 | 0.0938 | yes (barely) |
| WRONG | 0.3594 | +1.88 | +6.3549 | 0.0391 | yes |
| PLANTED | 0.6406 | +4.12 | +18.2054 | 0.0352 | yes |
| FROZEN | 0.8750 | +6.00 | +22.4499 | 0.0352 | yes |

FROZEN **is the SEEDED table at lr = 0**, so the wiring the verdict scored at 0.1875 decodes at 0.8750 in the same
pin, the same seeds and the same D. Conjunct (b) therefore contrasts two arms 0.50 and 0.62 held-out trials above
chance on a mean gap of ONE EIGHTH OF ONE TRIAL; WRONG, the arm (d) and (e) fail toward, sits 1.88 trials above chance
— it is NOT at the floor, and this record does not say it is. The subject §2.9's clause was read on is **the arm under
THIS task's learner**, not the wiring alone. §2.9 is the clause that FIRED and it fired as written; §3.3's quotation of
it is not re-scoped, and the disposition is HELD, not withdrawn — US3's "beatable prior" stays NOT MET and
`cortical_prior` stays the harness's fixture. What is recorded beside it is that its WARRANT is an OPENING: (c), the
control that licenses the whole verdict, was read against a RANDOM arm at 1.6× chance, and the tree's own floor rule
rejects that shape in the fixture ("a positive control that beats a floor is a formality, not a reading" — accepted
cell RANDOM 2.2× chance, rejected cells 1.1×). The row is re-read at a capacity where the base arms are above the
floor before any of the three retirements is treated as settled. The lr = 0 re-reading is an OPENING TOO and not a
conclusion: every arm there sits between 0.75 and 1.00, at CEILING, adjudicating nothing in either direction, and
(c)'s pinned margin is NOT called learner damage on the strength of it.

**THE ONE CODE CHANGE, red first.** `verdict()` now carries `arms_vs_chance` — every arm's mean, chance = 1/K, and the
harness's OWN `contrast` on (accuracy − chance) over the seeds — printed beside the t of every contrast; and every
contrast carries `wiring_contrast`, which reads NOT INTERPRETABLE AS A WIRING CONTRAST when BOTH of its arms fail the
above-chance check. The rule is ONE module-level `above_chance(accs, k)`, LIFTED out of `synthetic_artifact`'s
docstring so one rule governs the fixture and the campaign (P2); that docstring now cites it. **It REPORTS and it never
refuses:** no conjunct state, no verdict word and no bar depends on it. A gate here would have been worse than the
omission — had one existed, the pinned campaign would have refused, taking with it t(d_f) = −13.0154, the number that
deleted the learner. Applied offline to the pin's own per-seed accuracies (read, not re-run), **the mark fires on NO
pinned contrast**: RANDOM clears the bar, barely, so (b)'s near-floor reading is carried by the per-arm line above and
not by the mark. Red at `da0139f`, green at `82eafb1`; the bare self-check stays green with PLANTED still firing.

**THE 22 STANDING FINDINGS, and the form adopted for each.**

| # | sev | the finding | form adopted |
|---|---|---|---|
| F1 | MAJOR | WRONG is not degree-matched to RANDOM; 2.12 made no degree disclosure | §3.5 item 12 + the §3.3 withdrawal + a decisions.md OUTCOME line. NOT an edit of §2.12 (frozen). |
| F2 | MINOR | `h01_ingest`:1 still reads "a beatable prior"; no T063 succession in the module | appended dated block at the END of the module docstring (`2c2fea9`); :1/:817/:962 untouched. |
| F3 | MINOR | `HOST_NOTE` contradicts the machine `settling` block, and the pin field is undisclosed | §3.6 line + a STATIC prefix on the constant (`ad98d47`). No per-run outcome in the constant; pins untouched. |
| F4 | MAJOR | §2.12 cites two pilot readings that are not in the pin and no code produces | §3.5 item 13 + a decisions.md OUTCOME line naming its lines 55–60. §2.12 / §3.1 byte-identical; stage 0 never backfilled. |
| F5 | MAJOR | `quickstart.md`:32 and `tasks.md`:140 un-annotated; `594b95d`'s "every site" claim false | dated notes at both sites (`6d2a309`) + §3.5 item 14. The pushed subject line is NOT amended. |
| F6 | MINOR | the SEEDED-vs-chance t (+1.5275) is undisclosed | §3.6 line pointing at this section + a decisions.md OUTCOME line. RANDOM's collapse marked NOT CONTROLLED. |
| F7 | MINOR | the D guard is read at lr = 0 / one block, and is a ceiling with no floor | §3.6 line, filed as a LIMITATION not a deviation, carrying FROZEN 0.8750 as the corroborating no-learner five-block arm. |
| F8 | MAJOR | H-R1's "reds not re-runnable as written" over-reaches — red 2 needs no learner | appended sentence on the H-R1 cell (`6d2a309`) + its §3.4 mirror. No self-check number enters the register. |
| F9 | MAJOR | the "~15 min supervisor cycle" is false; the daemon restarts on every NEW COMMIT | §3.5 item 15 + appended succession in `settling_report`'s docstring (`ad98d47`). `f37fe90` and the pin's `host_note` deliberately not edited. |
| F10 | MINOR | "a null under P8 clauses 3 and 5" mis-cites clause 5 | §3.5 item 16. Clause 3 stands as cited; the disposition unchanged; no re-wording anywhere. |
| F11 | MINOR | "the §2.5 no-learner refusal … verbatim" mis-cites §2.5 and is not verbatim | IN-PLACE replacement of the §3.4(1) sentence, quoting the `NO_LEARNER` constant exactly. |
| F12 | MINOR | "attempted 179,571" is seed 0's, not an every-seed constant | prose sites rewritten to the range; QUOTE sites (specs/011 T012/T013) keep the message BYTE-IDENTICAL with the attribution corrected; decisions.md:271's sibling corrected by an appended line. |
| F13 | MINOR | `rules_seen` 173 → 172 does not name its span | IN-PLACE replacement of the §3.4(2) parenthetical with the span-named form. `1d4602c` is NOT cited as the 173 anchor — it reads 169. |
| F14 | MINOR | `tasks.md`'s "38,504" is a cross-driver sum of one counter | appended dated correction INSIDE the T063 DONE note (`6d2a309`), quoting only per-driver numbers already in `r_stdp.py`. No new aggregate. |
| F15 | MAJOR | the surviving harness reads NO VERDICT on H01 at the pinned capacity | the §3.4 appended paragraph (merged with F21 and F22c) + the module docstring's DISPOSITION block + `synthetic_artifact`'s appended succession (`ad98d47`). The harness is KEPT. |
| F16 | MINOR | the accuracies are k/8 and the verdict's t-values rest on 8 trials per seed | §3.2 appended paragraph with the trial-unit differences, + a §3.6 clause. No pooling recipe is prescribed. |
| F17 | MINOR | WRONG is seed-independent, so the corruption has ONE realisation | §3.5 item 17, stating it correctly for BOTH (d) and (e). §2.2 / §2.12 byte-unchanged. |
| F18 | MINOR | decisions.md's "Nothing is at ceiling" is false over its own table | one dated line appended at the end of that file's OUTCOME. "No arm is PINNED at chance" STANDS; (c) explicitly ring-fenced. |
| F19 | MINOR | §3.3's "the wiring … is more decodable" over-reaches | merged into the §3.3 appended paragraph. "Nothing here says why" STANDS; no decisions.md change; no mechanism claim. |
| F20 | MINOR | the campaign's settling windows are absent from §3.2 | the first half only: the windows quoted in the §3.2 appended paragraph. §3.5(4)'s attribution is NOT widened to a second cause. |
| F21 | MAJOR | the module docstring asserts the contrasts survive in a stronger form than §3.4 | merged into the same §3.4 paragraph and into the docstring's DISPOSITION block, with the reproduction recipe and the explicit "never `--campaign`". |
| F22 | BLOCKER | (b) was read near the floor; FROZEN decodes the same wiring at 0.8750 | this section's opening block, the §3.4 paragraph, the disposition HELD, the warrant recorded as an opening, and the ONE code change above. |

**THE 6 REFUTED FINDINGS, one line each, disclosed because a refuted finding is part of the record.**

1. [MAJOR] *"The D guard is a one-seed reading whose other side is REFUSAL OF THE ROW, not a different D."* — every
   number reproduced on an independent probe, but the finding's consequence did not follow: the ladder's blocks = 1
   protocol is what 2.7's frozen text defines, and the alternative reading it proposed is a different pre-registration.
2. [MAJOR] *"t(d_f) = −13 is a property of the unbaselined reward as much as of the rule, and the mechanism is a
   positive-feedback collapse the record does not name."* — the numbers reproduce byte-for-byte against the pin, but
   the mechanism claim is not established by them, and the disposition rests on the pre-registered criterion.
3. [MAJOR] *"Conjunct (b) was read between two arms at/near chance, under a learner the same campaign deleted;
   decisions.md's licensing sentence says otherwise."* — the arithmetic reproduces exactly (seeded +1.5275 below the
   bar, random +1.9296 above it), but the two load-bearing assertions fail: the licensing sentence is MISREAD (the
   licence is attached to t(planted − random) = +9.2601, not to the descriptive sentence beside it), and "pinned at
   chance" is true in this record's own sense of PINNED. The measured half survives as the §3.6 line and this
   section's opening block.
4. [MINOR] *"The whole-criterion NOT READ is a category 2.6 does not define and 2.9's closing clause forbids."* — the
   mechanical facts reproduce, but the finding is disclosed-half plus misread-half, and its own reachability
   sub-claim is false. §3.4 already records the third reading as the builder's judgment call.
5. [MINOR] *"The re-reading of the frozen host/settling clause was applied in code and used by a pinned pilot before
   the amendment that registers it."* — the timestamps reproduce exactly, and they refute the ordering claim: `aba9a71`
   appends §2.12 with no in-place edit of frozen text.
6. [MINOR] *"The repair that keeps a verdict whose host clause is NOT MET rests on two executions that left no
   artifact."* — the filed command reproduces byte-for-byte, and the conclusion does not follow from it.

**EVERY IN-PLACE CORRECTION THIS FOLD MADE** (everything else is appended dated succession):

| site | commit |
|---|---|
| `specs/022 tasks.md` T063 DONE note — "attempted 179,571" → the range | `6d2a309` |
| `specs/022 spec.md` SC-006 note — the same quartet | `6d2a309` |
| `PLAN-FINAL` :681 (T063 bullet) — the same quartet | `6d2a309` |
| `PLAN-FINAL` :864 (H-R1 cell) — the same quartet, inside 594b95d's own resolution text | `6d2a309` |
| `specs/011 tasks.md` T012 / T013 — the ATTRIBUTION around the quoted message (the message itself byte-identical) | `d560a95` |
| `prior_drift_harness.py` `HOST_NOTE` — a static prefix, body verbatim | `ad98d47` |
| evidence §3.2 (a) DRIFT cell — the same quartet | this record commit |
| evidence §3.4(1) — the no-learner refusal sentence (F11) | this record commit |
| evidence §3.4(2) — the `rules_seen` parenthetical (F13) | this record commit |
| `specs/011 tasks.md` T013 — the SEEDED control's seed-0 sibling ("528,383 / 231,865.22 … on every seed" → the ranges), a site no refuter named; the fold builder reported it instead of widening its scope, the orchestrator folded it | the commit after this record commit |

**THE FOLD'S COMMITS.** `6d2a309` (specs/022 annotation sites) · `d560a95` (specs/011 OUTCOME + T012/T013) ·
`2c2fea9` (h01_ingest appended succession) · `ad98d47` (the harness's DISPOSITION block, HOST_NOTE,
`settling_report`, `synthetic_artifact`) · `da0139f` (the RED) · `82eafb1` (the green: `above_chance`,
`arms_vs_chance`, `wiring_contrast`) · this record commit.

**This fold is battery-verified only (obs #35).** It was not itself panelled: the fold of review findings is unreviewed
code, and this sentence is the disclosure rather than a smoothing of it.
