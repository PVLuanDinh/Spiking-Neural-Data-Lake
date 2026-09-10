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
