# G4 / T112 -- the anatomical wiring prior: the H01 distance kernel first (G7's T3), then the attachment rule (pre-registration, P0 freeze text)

Plan row: PLAN-FINAL G workstream, G4 / T112 (its bullets and the 2026-09-25 dated note), G7's deciding tests T2, T3 and
T4's wiring half (`evidence/g7-organizing-unit.md` sections 3, 5, 6), Part F's G-R4, and H-R1 (the T063 negative, cited,
never reused). Owner rules binding here: POLICIES P2, P3 (+ amendment), P4d (decision C: "wiring follows rules (layer
target x distance kernel x shared-partner bonus) ... anatomical data is a fitted prior that must beat a shuffled copy"),
P8 and the cross-cutting invariants. The owner's work loop (2026-10-03) puts T3 first.

**Status: P0 FREEZE TEXT (2026-10-03), lead read + advisor + Codex critique folded; the lead commits it.** Section 2.14 governs 2.2-2.13, 3.1b, 3.5 and section 5 where they disagree. Section 2
(T3) is written to be frozen as is; sections 3.1-3.5 (Parts B-E) are outlines, each owing its own dated pre-registration
amendment before its first run. Every number marked (projection) is unmeasured; every other number was measured on the
date given, at `e1b3ae3c`, in the worktree `claude/g4`.

## 1. Territory (measured 2026-10-03 at `e1b3ae3c`)

1. **The plan's test.** T3 (g7 section 6): connection probability against distance on H01 -- Gaussian, exponential,
   both, or flat -- scored on held-out neurons, after a planted Gaussian (sigma = 100 um) is recovered within 10%, else
   REFUSED as a blind fit; red: the Gaussian loses to the exponential or the flat model. G-R4's first clause adds: "the
   fitted spatial model fails held-out neurons or regions". G4's first bullet: fitted "by ELIGIBLE pair counts over
   distance"; a global width (the baseline) against partially pooled pathway widths and a distance-free model.
2. **The data.** `data/h01/h01_c3_graph.npz` (gitignored; sha256 `a1d0491a...0549`, equal to
   `tools/h01_c3_graph_pin.json` stage `166-shards...#2`, re-hashed today): 14,215 units, all signed (9,793 excitatory,
   4,422 inhibitory -- the CELL's Dale sign from cell type; T062 recorded that the export's per-synapse type agrees with
   it on only 0.3014 of 202,232 synapses and retired that gate as an instrument defect, replacing it by (e'), r 0.4527);
   70,001 directed aggregated pairs (`pre_idx`, `post_idx`; `n_syn` 1-72, 104,959 synapses); `xyz_nm` int64. Units per
   band: L1 527, L2 4,709, L3 2,946, L4 2,509, L5 2,015, L6 1,040, white matter 469. The node set is NOT selected on the
   outcome: T062 recorded 2,988 units with no in-edge and 3,609 with no out-edge (read from the T062 record, not
   recomputed here). Edge density 70,001 / (14,215 x 14,214) = 3.46e-4.
3. **The geometry.** Extents 3,172.0 x 1,949.9 x 167.6 um (x, y, z; largest pair distance 3,727 um): a slab whose third
   axis is thinner than one Gaussian of sigma 100 um is wide at 2 sigma. The radial (pia -> white matter) axis lies in the
   x-y plane (`placement.layer_of(x, y)`, H01's `fix_layer_mem` circle rule), not along z. T062's unit gate (c) holds the
   frame (declared 8 x 8 x 33 nm, ratios 0.860 / 0.852 / 1.017).
4. **Incumbents (P2).** No module fits or applies a distance-dependent connection kernel. Searched: a regex over `src/`
   for distance kernels / `exp(-d` / connection-probability names (it FIRES on `growth/arealization.py::
   OrganizerField.sample`, `np.exp(-d / self.lam)` -- a 1-D positional-information field, a different job), `graphify
   query` over the main graph (nearest nodes: `basic_components.Connection`, `prior_drift_harness`; neither is a spatial
   kernel), and `SRC_INVENTORY.md`. Adopted, never re-derived: `growth/h01_ingest.py` (the artifact, its pin and gates;
   `DATA_DIR = REPO / "data" / "h01"`, absent in this worktree), `growth/placement.py` (`layer_of`, `LAYER_NAMES`,
   `assign_sign`, `place`, `place_grown`, `region_prior`), `reservoir/cortical_microcircuit.py::CONN_PROB` (driverless;
   Part C makes placement its first reader), `instruments/paired_contrast.py` (`contrast`, `permutation_fpr`, the seed-
   paired t at `T_BAR = 1.895`). Distinct, recorded: `sleep/structural_rewiring.py` forms synapses uniformly (no kernel);
   `growth/ng_snn.py::NGSNN` grows dense input rows with no position.
5. **Exposure.** No committed record fits connection probability against distance on H01: a grep for
   distance/sigma/gaussian/exponential over the T062, T063 and S1 evidence, specs/011, `h01_ingest.py`, `placement.py` and
   `prior_drift_harness.py` finds only T063's PSI "distance" and a random projection (the grep's positive control: it
   fires on those). This draft's pilot read POSITIONS only (item 6); the H01 edges' relation to distance is unread.
6. **Design pilot (out of tree: `scratchpad/g4/g4_t3_pilot.py`; development seeds 10000-10003, spent; outcome-blind).**
   Synthetic edges planted on H01's own 14,215 soma positions at H01's edge count, K = 5 neuron folds, the four models
   below, held-out log-likelihood (LL) on every pair touching the held-out fold, summed over folds:

   | seed | planted truth | edges | width fitted on all pairs | held-out LL vs flat (nats): G / E / both | best |
   |---|---|---|---|---|---|
   | 10000 | Gaussian, sigma 100 | 70,174 | sigma 100.15 (folds 99.96-100.34) | +384,828 / +378,438 / +384,828 | G |
   | 10001 | exponential, lambda 100 | 70,216 | lambda 100.16 (folds 99.70-100.45) | +268,189 / +277,303 / +277,299 | E |
   | 10002 | flat | 69,743 | sigma 4.4e8 (degenerate) | -0.0 / -0.1 / -2.0 | flat |
   | 10003 | G 100 + 0.2 x E 300 | 70,044 | -- | +136,272 / +165,459 / +177,836 | both |

   No probe built a Brain or touched the lake home (PROTOCOL 4b; `~/.snn_data_lake` mtime 2026-09-24, unchanged).
   Cost of one unit (one seed, K = 5): pair histogram 6.7-7.9 s (2.02e8 ordered pairs, numpy, one process), planted draw
   8.1-11.8 s, all 20 fits 0.2-0.4 s. **Incident kept as a constructed red:** the first run read sigma 9,122 um and "both"
   +2,991 nats on the planted Gaussian -- a bin-layout bug (counts laid out fold-major, reshaped bin-major) scrambled every
   bin. Nothing in the fit refused it; only the planted recovery did. Check 2.11 (c) makes that the REFUSED red.
   Two facts the pilot surfaced that bind the design: (i) flat is NESTED in the Gaussian (sigma -> infinity) and the
   pilot's planted-flat Gaussian sat 0.0 nats from flat, so "the Gaussian LOSES to flat" cannot fire (section 2.7);
   (ii) in the planted mixture the exponential beats the Gaussian and the mixture beats both (section 2.8).
7. **Seed block.** 10000-10999 (SEEDS.md). A seed-context regex (`Random(` / `seed` / `default_rng(` / `manual_seed(`
   within 30 non-digits of the number) fires on the known seed 4242 (22 hits in 14 files) and finds in this block only
   two DERIVED-stream constants, `faculties/synfire_chain.py` (`Random(self.seed + 10007)`) and
   `instruments/ignition_instrument.py` (`seed * 10007 + ...`); neither is a recorded seed set. Reported to "main";
   10007 is excluded from every allocation below.

## 2. Pre-registration: T3 -- does H01 favour a Gaussian distance kernel?

### 2.1 Question, and what a result licenses

- **Question.** On H01's c3 graph as released, is the probability that an eligible ordered pair is connected better
  described, on held-out neurons and held-out regions, by a Gaussian of soma distance than by an exponential or a
  distance-free (flat) model?
- **A PASS licenses only** "on this graph, the Gaussian is not refused by the exponential or the flat model at soma-
  distance grain", and, if its contrast is met, "the graph favours the Gaussian over the exponential". It is a property
  of ONE dataset: one human, one ~1 mm^3 temporal-cortex fragment, an automated reconstruction whose observed density
  (3.5e-4) is two orders below biological short-range connection probability. FR-007 binds: H01 "is a measurement of a
  HUMAN brain that a microscope saw, NOT a measurement of this brain -- any quantity derived from it is a prior". A PASS
  never licenses a benefit to the engine (that is T2 and Part D), a biological law, or the attachment rule's charge.
- **Not separable by any control here (disclosed, not handled):** the observed kernel is anatomy times reconstruction
  completeness. Synapses on axon fragments not agglomerated to their soma are lost, and that loss grows with path
  length from the soma; slab truncation at the z faces loses paths that leave the 168 um volume. The planted stage proves
  the FITTER is not blind; it cannot prove the DATA's kernel is anatomical. The z-interior reading (2.4) bounds the
  truncation half only.

### 2.2 Data, eligible pairs, distance (frozen)

- Source: the artifact at a path passed in (the instrument never resolves `h01_ingest.DATA_DIR` silently); refused
  unless its sha256 equals the pin's `artifact_sha256` and `n_nodes == 14215`, `n_edges == 70001`. The campaign reads
  THIS worktree's gitignored `data/h01/h01_c3_graph.npz` (copied by the orchestrator 2026-10-03, sha256 re-verified
  `a1d0491a...0549`), never the main checkout's `data/` (main's ruling on Q5). The pilot (1.6) read the main checkout's
  copy read-only, positions only, before that ruling; disclosed.
- Eligible pairs: every ORDERED pair (i, j), i != j, of the 14,215 units (202,052,010 pairs). Connected: (i, j) is one of
  the 70,001 (`pre_idx`, `post_idx`) rows, whatever its `n_syn`. Self-pairs excluded; reciprocal pairs count twice (once
  per direction). No unit is filtered by degree, layer or sign in T3.
- Distance: 3-D Euclidean soma-to-soma, `xyz_nm / 1000` in um, float64. Binned at 1 um (3,729 bins; the last bin closes
  at the largest distance); a bin is scored at its centre. (The pilot's recovery error at 1 um bins was 0.15%.)

### 2.3 Models and the fit (frozen)

- flat p(d) = p0; Gaussian p0 exp(-d^2 / 2 sigma^2); exponential p0 exp(-d / lambda); both (mixture) a exp(-d^2 / 2
  sigma^2) + c exp(-d / lambda), a, c >= 0. All parameters fitted on the log scale.
- Fit: maximum binomial likelihood over the training bins, sum_b k_b log p(d_b) + (n_b - k_b) log(1 - p(d_b)), with p
  clipped to [1e-300, 1 - 1e-15]. Start: a 40-point log grid of the width over [10, 3162] um with the amplitude profiled
  under the Poisson approximation; then Nelder-Mead (xatol = fatol = 1e-6, 4,000 iterations). The mixture starts from the
  fitted Gaussian and exponential. Flat is closed-form (k / n). A width above the largest pair distance is DEGENERATE: the
  record names it, and the model is scored as fitted.

### 2.4 Held-out scoring (frozen)

- **Neuron folds (the primary).** Per partition seed s, `np.random.default_rng(s).permutation(14215) % 5` assigns units to
  K = 5 folds. For fold f: train = pairs with BOTH units outside f; test = every pair touching f. Each model is fitted on
  train and scored on test; per seed the score is the sum over the 5 folds. The SAME partition serves every model
  (the pairing).
- **Region blocks (G-R4's "held-out regions"; the second instrument).** Four blocks by the x quartiles of soma position
  (tangential to the radial x-y circles; every block spans every layer): train = pairs with both units outside block r;
  test = pairs touching block r. Deterministic (no seed); four per-block readings.
- **z-interior robustness reading.** The same neuron-fold scoring restricted to units whose z lies in the central half of
  the slab's z extent ([z_min + 41.9, z_max - 41.9] um, a quarter of the thickness trimmed from each face -- a disclosed
  session choice). It exists so that a ranking produced by truncation at the faces cannot be read as anatomy.

### 2.5 Statistic, bar and inference (frozen)

- Per seed s, a contrast's value is d_s = LL_held(A) - LL_held(B) in nats (summed over folds). Decision by
  `paired_contrast.contrast(name, diffs, T_BAR)` on exactly 8 values, one-sided, `T_BAR = 1.895` (Part F's rule at S = 8);
  the exact sign-flip false-positive rate (`permutation_fpr`, 2^8 masks) is printed beside it. A contrast is MET only if
  it is non-degenerate, its t exceeds T_BAR, AND its mean exceeds the null band B (below); a degenerate row reads NOT MET.
- **The null band B (derived, not picked; P1 case (a)).** Measured in the planted stage BEFORE H01's edges are read: for
  each contrast and each scoring (neuron folds; each region block; z-interior), B = the largest |d| that contrast takes
  over the 8 planted-flat seeds under that same scoring, where every model reduces to flat. (Pilot, neuron folds: about
  0.1 nats for E - G, 0.0 for G - flat, 2.0 for both - G.) Every planted row of 2.6 is scored through all three
  scorings, so the region blocks and the z-interior reading carry their own positive controls.
- **What carries the inference, disclosed.** The 8 partition seeds re-partition ONE fixed dataset: they share almost every
  pair, so the t measures stability across partitions, not sampling uncertainty, and any systematic difference reaches
  an enormous t. The project rule is kept as the decision; the region blocks are the second instrument that must AGREE
  (2.7); effect size is reported in nats per held-out edge-instance with the planted-truth discrimination beside it for
  scale (pilot: about 0.05 nats per edge for G vs E at the planted Gaussian, 0.07 at the planted exponential). A met
  contrast whose mean is below one tenth of the matching planted discrimination reads "met, resolution-flagged" (a flag
  routed to the owner, never a change of verdict; the precedent is G1(c)'s skew flag). The population uncertainty (one
  specimen) is not measured by anything here and nothing claims it.

### 2.6 The planted stage (the precondition; G-R4's REFUSED clause, strengthened)

Run FIRST, on H01's real positions, at H01's edge count (p0 set so the expected edge count is 70,001), each truth on its
own 8 seeds, every reading through the SAME code path H01 will take. All must hold before H01's edges are read:

| planted truth (seeds) | must read | role |
|---|---|---|
| Gaussian sigma 100 um (10120-10127) | sigma fitted on all pairs within [90, 110] um on EVERY seed; G - E and G - flat MET | the plan's REFUSED clause (strengthened: the selection, not only the width) |
| exponential lambda 100 um (10130-10137) | E - G MET | the red "Gaussian loses to the exponential" is reachable; an E-win on H01 is believable |
| flat (10140-10147) | G - flat NOT MET; reading RED-flat | the false-positive control (slab geometry and folds manufacture no distance dependence) and the reachable form of the flat red; defines B |
| mixture G 100 + 0.2 x E 300 (10150-10157) | both - G and both - E MET | the mixture reading is sighted |
| Gaussian and exponential at 50, 200, 400 um (G50 10160-10167, G200 10170-10177, E50 10180-10187, E200 10190-10197, G400 10200-10207, E400 10210-10217) | the matching single kernel MET over the other | the selection is not tuned to one scale (strengthening) |
| Gaussian and exponential 100 um, degree-corrected: p_ij proportional to out_i x in_j x g(d) with H01's own out- and in-degree sequences (10220-10227 G, 10230-10237 E) | the matching kernel MET | realistic heterogeneity does not flip the selection (strengthening; reads degree MARGINALS only, which T063 already exposed) |
| z-interior subset: Gaussian, exponential, flat at 100 um (10240-10247, 10250-10257, 10260-10267) | as the rows above | the robustness reading has its own positive control |

Any row failing -> **REFUSED (blind fit)**: the H01 reading is not taken, nothing is deleted, the fitter is repaired and
the stage re-run on the reserve seeds (10270-10299) under a dated amendment. (Seeds, validation set and reserve mapping superseded by 2.14 A5-A6.)

### 2.7 The reading (G-R4's T3 clause and first clause; a pure function, rows read top to bottom, first that applies) -- table and flat paragraph superseded by 2.14 A1-A3 (heading per L3: neuron folds decide, the REGION-BLOCK AGREEMENT is the real second instrument)

| inputs | T3 reads |
|---|---|
| any planted row in 2.6 fails | REFUSED (blind fit) |
| the z-interior ranking disagrees in sign with the full set on G - flat or on E - G | NOT READ (boundary-sensitive) |
| G - flat NOT MET on the neuron folds | RED-flat ("the fitted spatial model fails held-out neurons"; G-R4 first clause) |
| G - flat not above B on any one of the four region blocks | RED-flat-region ("fails held-out regions") |
| E - G MET on the neuron folds | RED-exp ("the Gaussian loses to the exponential") |
| the region blocks' summed E - G has the opposite sign to the neuron folds' mean | NOT READ (instruments disagree) |
| otherwise | PASS, worded "favours the Gaussian" if G - E is MET, else "Gaussian not refused; G vs E unresolved" |

**The flat clause is reworded from the plan, and the rewording is a P8 obligation, not a choice.** The plan writes "the
Gaussian loses to ... the flat model"; flat is the Gaussian's sigma -> infinity limit, so on held-out data the Gaussian
can trail flat only by the parameter noise the pilot measured at 0.0 nats: that red is unreachable and P8 / F11 deletes
it rather than weakening it. The reachable form is G-R4's own first-clause wording, "fails to beat the distance-free
model", and planted-flat makes it fire. Routed to "main" (PLAN-FINAL is orchestrator-only).

### 2.8 The kernel adopted, and its width

- Nested comparisons are one-sided: the mixture is adopted only if both - W is MET, W the single kernel the reading leaves
  standing (G unless RED-exp, then E). Otherwise W alone.
- **Disclosed consequence:** in a mixture world the pilot read E above G (seed 10003), so T3 can read RED-exp while the
  adopted kernel still carries a Gaussian core. The literature starting kernel in g7 section 3 ("a Gaussian of about
  85-114 um locally, falling off about 4.6 per mm") is itself of that shape. Whether a Gaussian-core mixture counts as
  "the Gaussian" for P4d's deletion is the owner's call (section 6, Q2); this file reports the literal clause.
- The width(s) are fitted once on ALL eligible pairs after the reading and recorded in `tools/g4_t3_campaign.json` with
  the 40 fold fits' range. Running-baseline-relative until the owner's P1 call (g7 section 3): consumers read the
  record, never a constant; refitting on the pinned data reproduces it (a check, 2.11 (f)).
- **The 3-D width is NOT the attachment rule's width** (debate D1 with G7). T3's 3-D reading decides the FORM and P4d's
  deletion; the rule measures distance in G7's tangential sheet frame (`x_um, y_um`, depth carried only by the layer
  target), so Part B (3.1) refits form and width in a tangential frame with layer-pair amplitudes, and the rule reads
  THAT record. A form disagreement between T3's 3-D reading and Part B's tangential one is disclosed and routed to the
  owner; it does not change T3's verdict.

### 2.9 Null dispositions are causal claims, each with its positive control run first

- **RED-flat** claims "this graph carries no usable soma-distance dependence". Its instrument's positive controls: planted
  Gaussian and exponential each beat flat; planted flat reads RED-flat (2.6, run before H01).
- **RED-exp** claims "the exponential describes this graph better than the Gaussian". Positive controls: planted
  exponential reads RED-exp, planted Gaussian reads "favours the Gaussian", at four scales and under degree heterogeneity.
- **NOT READ (boundary-sensitive)** claims "the ranking depends on the slab faces". Positive control: the z-interior
  planted rows hold, so a disagreement is the data's, not the subset's lost power.
- The cause NO control here can name: reconstruction completeness (2.1). Every disposition's wording carries "on the c3
  graph as released".

### 2.10 What each reading deletes, and what it does not

- **RED-exp:** the Gaussian as the attachment rule's kernel form (decision C's "distance kernel"; g7 section 6
  dispositions, P4d: "a part of decision C whose deciding test reds ... is deleted and the owner told"). The exponential,
  or the mixture by 2.8, becomes the kernel. The owner is told.
- **RED-flat / RED-flat-region:** the distance-kernel FACTOR of the attachment rule (it becomes layer target x shared-
  partner bonus); the owner is told.
- **Never deleted by T3:** the anatomical map and grown positions (T2's question), `h01_ingest` and its artifact (reference
  data), `placement`, `CONN_PROB`, G5's Gaussian-neighbourhood topographic map (a different owner-drawn component with its
  own control), and any owner-drawn part (the two networks, the reservoir, the return path; P4d).
- **REFUSED / NOT READ** delete nothing; the kernel stays the literature starting kernel, refused for use by any arm
  ("a hand-picked width is refused", g7 section 3) until a READ T3 exists.

### 2.11 Checks, each with a constructed red (the build arc owes every one; a red that does not fire halts the arc)

(a) Artifact: a copy with one byte changed is refused by the sha check; a wrong `n_edges` is refused. (b) Units: distances
fed in nm (the unit trap) make the planted sigma-100 recovery fail -> REFUSED fires. (c) Layout: the pilot's incident --
bins reshaped bin-major from fold-major counts -> REFUSED fires (planted recovery fails). (d) Split: a mutant that leaves
test pairs in training is caught by the disjointness assertion (train + test == all, train touches no unit of fold f).
(e) Determinism: the same seed gives byte-equal pair and edge histograms and equal fitted parameters; a different seed
gives different fold counts. (f) Width record: a refit on the pinned data reproduces the recorded width to 1e-6
relative; a perturbed record is refused. (g) The reading function: one constructed input per row of 2.7, each reaching
its row, and each planted truth of 2.6 reaching its expected row on a small synthetic population (no H01, no network) in
the self-check. (h) Degenerate width: a planted-flat fit is named DEGENERATE in the record. (i) The null band: a planted
row whose |d_s| exceeds B by construction is MET; at B exactly, NOT MET (the boundary is strict).

### 2.12 Seeds (block 10000-10999; 10007 excluded; paired-seed rule: every model on one partition per seed)

- 10000-10003 development, SPENT (pilot); 10004-10006, 10008-10099 development (build and tests).
- 10100-10107 **confirmatory** (H01 neuron-fold partitions; read ONCE, after 2.6 holds); 10108-10115 reserve.
- 10120-10267 the planted stage (2.6); 10270-10299 planted reserve. (Superseded by 2.14 A6: 10240-10267 retired unused, 10270-10277 flat validation, reserve 10600-10703.)
- 10300-10349 Part B; 10350-10399 Part B2 (beta); 10400-10407 T2 confirmatory (paired across T2's arms), 10410-10417
  T2's positive control, 10418-10429 its decoy draws, 10430-10499 Part D and reserves; 10500-10599 Part E generator development and reds (G7's
  13000s carry T4's scorer and arms); 10600-10999 unallocated.
- Before the freeze the seed-context grep (item 1.7) is re-run on the frozen head with its 4242 positive control: the LEAD runs it at the commit head (host-quiet phase); G4 re-grepped only the new ranges (2.14 A6).

### 2.13 Cost at real scale

Measured (item 1.6): one H01 unit (histogram only, the edge distances are 70,001 rows) about 7 s; one planted unit about
17-20 s. **(projection)** The planted stage: 12 full-set truths x 8 seeds at about 18 s (about 29 minutes) plus the 3
z-interior truths x 8 seeds at about a quarter of that (about 2 minutes), in one process; the H01 reading (8 seeds, 4 blocks, the z-interior) about 2 minutes; well under
an hour in total, no GPU, CPU-only like CI. It can run beside the G1(c) campaign (one process, about 1 GB peak for a
256-row chunk). (Superseded by 2.14 A11 below.)

### 2.14 Amendments at the P0 freeze (lead read, advisor read and Codex critique folded) -- these GOVERN 2.2-2.13, 3.1b, 3.5 and section 5 wherever they disagree

Sources: the lead read (board `p0_reports\G4.md`, L1-L6), DEPENDENCIES.md rulings 9-10, and the Codex critique
(findings 1-7; Review record at the end of this file). Each item names the text it replaces.

- **A1 -- the record's T3 red wording (replaces the plan's flat clause and the flat paragraph under 2.7; L1).**
  Verbatim: "red: the Gaussian loses to the exponential on held-out neurons; or the fitted spatial model (the kernel the
  reading leaves standing) fails to beat the distance-free model on held-out neurons or on any held-out region. Planted
  flat must read this red; planted Gaussian must not." The lead records it as a dated PLAN-FINAL note in the integration
  arc. "The kernel the reading leaves standing" is K of A2, selected before any verdict is read.
- **A2 -- contrast states and kernel selection, verdict-independent (replaces 2.7's table, 2.8's first bullet and W;
  Codex #1, #3).** (i) *States.* Every contrast X - Y under every scoring has a tri-state: +1 if MET (non-degenerate,
  t > T_BAR and mean > its band); -1 if the reversed contrast Y - X is MET under the same rule; 0 (unresolved)
  otherwise. Neuron folds and the z-interior use t over the 8 partition seeds and their own bands; a region block has
  one deterministic value d_r, state +1 if d_r > B_r, -1 if d_r < -B_r, else 0; the region AGGREGATE is sum_r d_r against
  sum_r B_r. An unresolved (0) state never vetoes and never establishes disagreement. (ii) *The candidate set*
  C = {k in (G, E, mixture) : k - flat is +1 on the neuron folds}. (iii) *The agreed E - G state* a = s_n if the neuron-
  fold state s_n of E - G is non-zero AND equals the region-aggregate state s_r; else 0. (iv) *Selection of K* (a fixed
  function of the states, computed before any verdict): if C is empty, K = none. Else, among the single kernels in C:
  both G and E -> K = E if a = +1, G if a = -1, and by the PRE-REGISTERED tie order G if a = 0 (the plan's named form);
  one of them -> that one; neither (the mixture alone in C) -> K = mixture. Then a single K is upgraded to the mixture
  iff mixture - K is +1 on the neuron folds AND its region-aggregate state is +1 (the nested comparison, one-sided).
- **A3 -- the reading (replaces the 2.7 table; L3: the heading of 2.7 now reads "neuron folds decide, the REGION-BLOCK
  AGREEMENT is the decision's real second instrument").** A pure function of the states and K; rows top to bottom, the
  first that applies:

  | # | inputs | T3 reads |
  |---|---|---|
  | 1 | any planted row of 2.6 / A6 fails its expectation | REFUSED (blind fit) |
  | 2 | check 2.11 (j) refuses a region block (A7) | NOT READ (region instrument invalid) |
  | 3 | K = none | RED-flat (no kernel beats the distance-free model on held-out neurons) |
  | 4 | the z-interior state is the OPPOSITE non-zero state of the full set on K - flat or on E - G | NOT READ (boundary-sensitive) |
  | 5 | s_n and s_r of E - G are opposite non-zero states | NOT READ (instruments disagree) |
  | 6 | K - flat is not +1 on any one of the four region blocks | RED-flat-region (fails a held-out region) |
  | 7 | K = E, or K = mixture upgraded from E, or (G not in C and E in C) | RED-exp (the Gaussian loses to the exponential) |
  | 8 | otherwise | PASS: K = G and a = -1 -> "favours the Gaussian"; K = G and a = 0 -> "Gaussian not refused; G vs E unresolved (tie order)"; K = mixture -> the mixture's wording, Q2 routes |

  Disclosed: K - flat is nested (flat is every kernel's limit), so its -1 state is near-unreachable and row 4 bites in
  practice on E - G; the z-interior reading of K - flat is printed regardless. Row 3 precedes rows 4-5 because with
  K = none there is no form to compare. 2.5's disclosure stands (L3): the 8 partition seeds re-partition one dataset,
  their t measures partition stability, and the region-block agreement (rows 5-6 and a in A2) is the instrument that
  carries sampling-like evidence. 2.11 (g) gains one constructed input per row above, plus: E-only over flat (C = {E},
  reads RED-exp), mixture-only over flat (C = {mixture}, K = mixture, reads PASS-mixture), G and E both in C with a = 0
  (tie order picks G), a raw-sign disagreement inside the bands (states 0, no veto), and an opposite resolved pair
  (reads NOT READ).
- **A4 -- degenerate contrasts (replaces the last sentence of 2.5's first bullet; ruling 9 supersedes L2).** A
  zero-variance (degenerate) paired contrast reads NOT MET in every row, whatever its common value (state 0 in A2).
  Disclosed as STRICTER than `paired_contrast.contrast`, which reads a zero-variance set MET iff its common value > 0
  (G1(c) 2.3 is the pushed precedent; one convention for every row in the integration). Band B stays on top: a
  non-degenerate contrast is MET only if t > T_BAR AND its mean > B. The record prints `paired_contrast.contrast`'s own
  `degenerate` flag and `permutation_fpr` (an FPR diagnostic at the fixed bar, never a p-value) beside each row.
  2.11 (i) gains: a zero-variance set with common value above B reads NOT MET.
- **A5 -- the null bands: calibration separated from validation (replaces 2.5's band bullet and 2.6's flat row;
  Codex #2).** The bands are CALIBRATED on planted flat seeds 10140-10147 only: for each contrast and each scoring
  (neuron folds, each region block, z-interior), B = the largest |d| over those 8 seeds (the z-interior and region
  readings of each seed come from the same draw, A8). The bands are then LOCKED (written to the record with their
  sha256) before any other planted row is scored. An UNTOUCHED planted-flat VALIDATION set, seeds 10270-10277, is drawn
  and scored after the lock and must read RED-flat (A3 row 3) through the COMPLETE reading function (one reading over its
  8 seeds, A6), with no kernel - flat state +1 under any scoring. The calibration set no longer refutes itself. An
  artefact that varies across draws is caught by the validation set; a CONSISTENT common-path artefact shared by
  calibration and validation draws is absorbed into B, so validation still reads RED-flat -- that case is caught only by
  the positive rows (recovery of the planted width, the kernel selections of A6), disclosed. The calibration set reads
  nothing and is a band source only (disclosed: it cannot fail by construction).
- **A6 -- the planted stage, its expectations and its seeds (replaces 2.6's table and closing paragraph; Codex #7).**
  Every truth is drawn on H01's real positions with p0 solved so the expected edge count over ALL eligible pairs is
  70,001; each truth has its own 8 recording seeds (never shared across truths), each seed spawning its own streams (A9).
  Every row is scored through the complete reading function (A2-A3) with the locked bands, ONE reading per truth over
  its 8 seeds (the sigma-recovery check alone is per seed). Planted inputs to A2, registered: neuron folds and z-interior
  take t and mean over the 8 seeds exactly as for H01; a region block's value for a planted truth is the 8-seed MEAN
  d_r against B_r (the aggregate is the sum of those means against sum_r B_r). Disclosed asymmetry: a planted truth's 8
  values span independent draws AND partitions, H01's span re-partitions of one dataset only, so planted t's are not
  calibrated to H01's (2.5 already says H01's t measures partition stability). Each row must read:

  | truth (seeds) | must read |
  |---|---|
  | flat CALIBRATION (10140-10147) | nothing; band source (A5) |
  | flat VALIDATION (10270-10277) | RED-flat on every seed |
  | Gaussian sigma 100 (10120-10127) | PASS "favours the Gaussian", K = G, on every seed; sigma fitted on all pairs within [90, 110] um |
  | exponential lambda 100 (10130-10137) | RED-exp with K = E |
  | mixture G 100 + 0.2 x E 300 (10150-10157) | K = mixture |
  | G50, G200, G400 (10160-10167, 10170-10177, 10200-10207) | PASS "favours the Gaussian", K = G |
  | E50, E200, E400 (10180-10187, 10190-10197, 10210-10217) | RED-exp with K = E |
  | degree-corrected G 100 / E 100 (10220-10227 / 10230-10237; A10) | as G 100 / E 100 |

  The three z-interior truths of the draft (10240-10267) are RETIRED unused: every row's z-interior reading is the same
  draw restricted to interior pairs (A8), so each truth carries the robustness reading's positive control, and the
  validation set carries its false-positive control. **Reserve (repair re-run):** any failure is repaired once in the
  common code path and then EVERY row above re-runs -- never a selected row -- on the reserve, mapped truth by truth in
  the table's order: flat calibration 10600-10607, flat validation 10608-10615, G100 10616-10623, E100 10624-10631,
  mixture 10632-10639, G50 10640-10647, G200 10648-10655, G400 10656-10663, E50 10664-10671, E200 10672-10679, E400
  10680-10687, degree-corrected G 10688-10695, degree-corrected E 10696-10703, under a dated amendment; a second failure
  is REFUSED with no further re-run. 10278-10299 stay unallocated. Grep 2026-10-03 (Grep tool, worktree `claude/g4`): the
  pattern for 10270-10289 and 10600-10799 finds nothing in `src/`, `tests/` or `specs/` beyond this file and one
  decimal fragment `0.10632` in `evidence/jepa-live-floor.md` (a measured value, not a seed); the same search's positive control, 4242,
  fires in 10 `src/` files.
- **A14 -- section 5's data sentence (replaces "Data: the H01 artifact lives only in the main checkout's
  gitignored `data/h01/` (section 6, Q5)").** "Data: the campaign reads THIS worktree's gitignored
  `data/h01/h01_c3_graph.npz` (copied by the orchestrator 2026-10-03, sha256 `a1d0491a...0549`), never the main
  checkout's `data/` (2.2; Q5 ANSWERED)."
- **A7 -- region blocks cover every layer (added to 2.4 and 2.11).** "Every block spans every layer" is UNMEASURED and
  in doubt (3.1: the layer circles are non-concentric, the radial axis rotates across x). Registered as check 2.11 (j),
  positions only, run before any edge is read: the 4 x 7 table of unit counts per x-quartile block per band (L1-L6, white
  matter), its minimum recorded; a block with an empty band is REFUSED as a held-out-region instrument and the reading
  takes NOT READ (region instrument invalid, A3 row 2) rather than substituting another split. Measured only after
  the host is released (A11); no number is claimed here.
- **A8 -- the z-interior subset, positions-only (replaces 2.4's third bullet's population and the draft's interior planted count;
  Codex #4).** Interior units = z in [z_min + 41.9, z_max - 41.9] um. Eligible pairs = ordered pairs with BOTH units
  interior; connected = edges with both endpoints interior. Folds (the same partition restricted) and fits run on that
  subset alone, so its edge set is self-contained. Planted rows are NOT matched to H01's interior edge count (that count
  needs an edge-to-geometry join, which is an outcome read): each planted truth is drawn once over all pairs at the
  full-set p0 (A6) and its interior reading is that draw restricted to interior pairs -- a positions-only planting
  density. Disclosed: the planted interior edge count may differ from H01's.
  **Exposure before the stage completes (whitelist).** Permitted: the artifact's sha256, `n_nodes`, `n_edges` (the pin
  check, 2.2), unit positions and layer bands, and the per-unit out- and in-degree SEQUENCES (degree marginals, already
  exposed by T063; A10 needs them). Prohibited until every row of A6 holds: any per-edge distance, any join of edges to
  geometry (including interior or per-block edge counts), any fit or score on H01 edges. The instrument enforces the
  order: its H01-scoring entry point refuses unless the locked planted record exists and every row holds; a constructed
  red (calling it first) must refuse (2.11 (k)).
- **A9 -- RNG streams (added to 2.4, 2.6 and 2.11 (e)).** Each recording seed s spawns independent streams:
  `np.random.SeedSequence(s).spawn(3)` -> stream 0 the unit fold partition (`default_rng(stream0).permutation(14215) %
  5`), stream 1 the planted edge draw, stream 2 the PAIR fold partition used only by Part B2 (A12). The H01 rows use
  stream 0 (and B2 stream 2) only. Check 2.11 (e) gains: changing stream 1 alone leaves
  the partition byte-equal; changing stream 0 alone leaves the planted edges byte-equal.
- **A10 -- degree-corrected planted rows (added to 2.6's degree-corrected row).** p_ij = min(1, c x out_i x in_j x
  g(d_ij)), with c solved so the expected edge count is 70,001 AFTER clipping; the record carries the fraction of pairs
  clipped at 1 and their share of expected edges. Units with zero out-degree (as source) or zero in-degree (as target)
  get p = 0 by construction: disclosed; they still enter every fit's pair counts as zeros, exactly as on H01.
- **A11 -- when it runs (replaces 2.13's "It can run beside the G1(c) campaign"; L4).** Nothing in 2.6 or 2.4 runs
  while the G1(c) campaign records. The planted stage (projection: 13 truths x 8 seeds at about 18 s, about 31 minutes,
  one process) is a heavy job: it starts only after the lead messages "campaign DONE" AND releases the host for it; the
  H01 reading follows the planted stage in the same released window, never in parallel with another row's battery or
  campaign.
- **A12 -- Part B2's hold-out contract (replaces 3.1b's "scored by EDGE-level hold-out" sentence; Codex #5).** The
  split is over ELIGIBLE PAIRS, not edges: every ordered pair (connected or not) is assigned to one of 5 folds by the
  PAIR partition stream (A9 stream 2) of seed s. For fold f: test = every pair in f, connected or not (the denominator includes
  non-connections); train = every pair outside f. All covariates are computed from the TRAINING GRAPH (the connections
  among pairs outside f) and used unchanged for train and test pairs: out_i, in_j and m_ij (shared partners). So a held-
  out connection never increments its own degree or partner predictors. Thinning: the training graph holds about 4/5 of
  the edges; covariates are NOT rescaled, and beta is reported as fitted on the thinned graph (disclosed; the planted
  controls run through the same split, so recovery is judged on the same footing). The Part B2 amendment freezes the
  rest before its first run.
- **A13 -- run gates downstream (replaces 3.5; Codex #6).** Fixture BUILDS (code and tests on synthetic populations, no
  deciding number) may proceed now. DECIDING runs wait on named records:
  - Part B: T3 READ (not REFUSED, not NOT READ) with K != none.
  - Part B2: T3 READ; its planted controls first (3.1b).
  - Part C used in any deciding test, and every T4 kernel arm: T3 READ with K != none, AND Part B's frozen tangential
    record. T3 = RED-flat removes the kernel factor: the kernel arms do not run and the rule is layer target x bonus.
  - T4's beta arm: Part B2's result, or its registered beta = 0 fallback recorded as such.
  - T2 and Part D: G3's grown populations, T2's positive control (3.3) holding first, AND ruling 8's gate (G2's
    `taught_problems.py` / `delayed_cue.py` / `taught_campaign.py` integrated on main before G4's T2 confirmatory run).
  A deciding run that finds a gate record absent refuses; no fixture or provisional width ever feeds one.

## 3. Parts B-E (outlines; each owes a dated pre-registration amendment before its first run)

### 3.1 Part B -- pathway widths (G4 bullet 1; G-R4 first clause) -- data-only, after T3 reads

A global width (the T3 kernel, the baseline) against widths partially pooled by pathway (pre layer x post layer, using
`placement.layer_of`; pre sign x post sign, the cell's Dale sign, with T062's 0.30 export-type disagreement disclosed),
against the distance-free model with the same pathway amplitudes. Pooling by a penalized likelihood shrinking each
pathway's log-width toward the global one, the shrinkage chosen on the training folds only. Scored on held-out neurons
and region blocks under 2.4-2.5. Red: the pooled widths do not beat the global width (then the global width alone is
kept), or the pathway model with the kernel does not beat its own distance-free twin. Planted control: pathway-specific
widths planted on H01 positions must be recovered. Seeds 10300-10349.

**The tangential frame (debate D1).** Part B fits in the frame the rule uses. H01's layer circles
(`cortical_bounds_circles.json`, measured 2026-10-03) are three NON-concentric centre pairs, (462, 2806), (1150, 2458)
and (1158, 2412) um, radii 1,416-2,479 um, every centre outside the slab (soma y 340-2,290 um): the radial direction
lies in the x-y plane and rotates across x; z is tangential. Per pair, about one reference centre (a disclosed
approximation; its validity check: `layer_of` must be monotone in the radial coordinate r on the 14,215 somata, the
violation count recorded), d_rad = |r_i - r_j| and d_tan = sqrt((r_mean * dtheta)^2 + dz^2). The rule's kernel is a
function of d_tan; depth enters only through the layer target, so no cortical thickness is needed. The same pass
measures H01's TANGENTIAL soma density (per mm^2 of sheet, pooled over layers and per layer) and records it beside the
width; G7's T4 substrate and G3's batch placement read both from the record (D6). (Projection: about 2.7e4 somata per
mm^2 pooled, from 14,215 / (3.172 mm x 0.168 mm); G7's first figure of 2,300 divided by the x-y cross-section, which
contains the radial axis.)

### 3.1b Part B2 -- the shared-partner bonus beta, fitted on H01 (debate D3 with G7) -- data-only, after T3 reads -- hold-out contract superseded by 2.14 A12

The rule's (1 + beta x m_uv) term (m = shared partners) needs a beta that is NEVER fitted on modularity (T4 would be
circular). Fitted on H01 as P(connect) against m at matched distance, DEGREE-CORRECTED (p_ij proportional to out_i x
in_j x g(d_ij) x (1 + beta m_ij); without the degree terms beta absorbs degree heterogeneity), scored by EDGE-level
hold-out: m counted from training edges only (neuron folds leak here, since a test neuron's partners are test edges).
Planted controls first, on H01 positions with H01's degree sequences: a kernel-only world must read beta not above its
null (the confound control: distance and degree alone create shared partners), and a planted beta > 0 world must
recover beta. Either failing -> REFUSED, and T4 runs at beta = 0 with the shared-partner clause reported untested (G7's
fallback). Power is unknown: mean in+out degree is about 9.8 and most pairs share no partner (projection). Seeds
10350-10399.

### 3.2 Part C -- the attachment rule as code (G4 bullet 2), the interface G3 consumes

`growth/wiring_prior.py` (claimed): p(attach u -> v) = layer_target(layer_u, sign_u, layer_v, sign_v) x kernel(|x_u - x_v|)
x shared_partner_bonus(u, v), where layer_target reads `cortical_microcircuit.CONN_PROB` THROUGH `placement` (its first
reader; P3), with the L2/3 aggregation (H01's and placement's separate L2 and L3 bands map onto CONN_PROB's single L2/3
population) and L1's inhibitory-only disposition made explicit and tested; the kernel is Part B's TANGENTIAL record
(form and width; distance = |(x_um, y_um)_u - (x_um, y_um)_v| within one map cell, never across cells); the bonus's beta
is Part B2's record or 0. The decoder `beta` of the G4 bullet (a readout's output weights) stays a decoder, never a Dale
efferent -- a different symbol from the shared-partner beta. It does not ship a kernel before T3 reads (no hand width).
Interface (agreed with G3 and G7, D4): `attach_probabilities(new_units, existing_units, record) -> array`, units being
G7's labeled admission records (`place_labeled`; read through `labels_by`) plus partner sets; wiring_prior holds NO
per-unit field; deterministic for a given RNG stream. Its callers are G3's column batches (grown units only).

### 3.3 Part D -- downstream on G3's grown populations (WAITS for G3's grown populations; not before)

The fitted prior wires G3's grown neurons; arms: the fitted prior; a degree- and pathway-preserving distance shuffle
(each unit keeps its degree and pathway mix, partners re-drawn ignoring distance); the layer x sign prior against its
degree-, distance- and source-sign-matched twin with layer and target-sign preferences removed. Measures: silence,
saturation, temporal discrimination, cost (G-R4). **T2** runs here: growth with the H01 prior (position, layer, sign,
distance rule) vs the same prior with positions shuffled among units vs a generic template (CONN_PROB without distance),
scored on G3's growth outcome; red: H01 does not beat shuffled H01 -> the H01 prior is reverted and the map part of C is
deleted, the owner told (g7 section 6). T063 is cited (its null kept degrees, signs and magnitudes, never position) and
its pin is not reused (H-R1). Seeds 10400-10499.

- **T2's endpoint (agreed with G3, D4):** U = admitted grown units (E and the shared I) at which G2's solved predicate
  first holds on validation and then on future presentations, learning OFF, censored at budget + batch; solve-within-
  budget reported beside it. Run through G3's `instruments/growth_campaign.py` with the attachment prior as an arm
  argument (G3's file; G4 never edits it); confirmatory seeds 10400-10407, paired across T2's arms only.
- **T2's positive control, run FIRST (G3's refutation, D5, CONCEDED).** On G3's taught-learner problems every grown unit
  reads the frozen 52-rank code and writes to the label neurons, so unit-to-unit wiring may barely move U: a T2 null
  there could be the instrument's insensitivity, and T2's red DELETES the map part of C. So T2's null disposition owes
  its positive control before T2 runs: a constructed problem whose solving feature needs specific earlier grown units
  that are SPATIALLY CLUSTERED to combine; on it the generator's own (correct) kernel must beat its position-shuffled
  twin on U by the bar (S = 8, paired_contrast, seeds 10410-10417). If it does not separate, T2 reads **REFUSED
  (insensitive instrument)** -- nothing is deleted and the owner is told -- never RED. The same precondition binds Part
  D's downstream readings (a planted wiring that should matter must move silence / saturation / temporal
  discrimination before a null on H01's prior is read). Shape (D5, D9): B = an EXISTING capacity cell (G2's XOR in
  `inseparable_mapping` or delayed-cue S0C1), whose current weights provably cannot solve it (G2's A3 / B4); the planted
  prerequisite units = G2's `witness_units(problem)` (two seed-free threshold units over the frozen code whose
  conjunction separates B, checked by G2's A5 / B2), placed in a spatial cluster at the growth site by G3's site rule;
  G2's `decoy_units(problem, n, seed)` (random threshold units over the same code) fill the rest of the sheet at the
  same density; correct kernel vs its position-shuffled twin (witness and decoy positions permuted together), U at
  S = 8. Beside it, reported only: a dilution series with decoys inside the cluster (1:1, 1:4, 1:16) locating the
  detection floor. Precondition owned by G3's checks: the learner solves B once its input port reads the witnesses.
  Decoy seeds 10418-10429. Exact numbers are frozen in the Part D amendment.

### 3.4 Part E -- T4's wiring half (split agreed with G7, D2)

**G7 owns** the modularity scorer (`instruments/unit_organization.py`: Louvain Q, maximum over 10 restarts, its
instrument reds IR1-IR3) and the decision's pre-registration (`evidence/g7-unit-label-gate.md` section 2.7: RULE vs
RANDOM at exactly matched N and M, S = 8, the plan's literal red). **G4 supplies** the RULE generator
(`growth/wiring_prior.py`: layer target x the Part B tangential kernel x (1 + beta x shared partners), beta from Part
B2 or 0) and runs the wiring half. Every kernel arm waits on T3 (and Part B's tangential record); the beta arm waits on
B2. **Attribution (G4's challenge, conceded by G7):** RULE > RANDOM is near-guaranteed on spatially placed units by the
kernel alone (random attachment has no spatial structure; a degree-preserving swap destroys space too, so G7 dropped
it), so a PASS of the literal red licenses only "rule wiring is more modular than random attachment"; each term is
credited only by its own registered contrast (shared partners: RULE > distance-only at beta = 0; distance: RULE > flat
kernel; layer: RULE > no layer target; matched N and M, same positions, S = 8). The reachable construction of the
literal red is the rule at kernel width -> infinity with beta = 0 and no layer target, which IS random attachment.
Substrate (G7's CH2 answer): `place_labeled` units in one map cell at H01's TANGENTIAL density read from the Part B
record (not the x-y cross-section density, D6), N the smallest of {1000, 2000, 4000} whose sheet side is at least 6 x
the tangential sigma; G3's grown population is a secondary, reported substrate. Seeds: G7's (13000s) for the scorer and
its arms; G4's 10500-10599 only for the generator's own development and reds. The task-score half runs in G9.

### 3.5 What waits -- superseded by 2.14 A13 (run gates)

T3 and Part B wait on nothing (DEPENDENCIES.md). Part C's code may be built on fixtures after T3 reads; its use in a
deciding test waits for G3. Part D and T2 wait for G3's grown populations (which wait for G2's capacity cell). Part E's
data-only form waits on nothing but the ownership answer; its grown-map form waits for G3.

## 4. The arc split and the files each part touches

- **P0 (docs):** this file, frozen by the orchestrator.
- **P1 (build: the T3 instrument):** `src/snn_data_lake/instruments/h01_distance_kernel.py` (new; artifact path + sha pin,
  pair and edge histograms, the four fits, neuron-fold / region / z-interior scoring, the planted generator, the 2.7
  reading as a pure function, a synthetic self-check), `tests/test_h01_distance_kernel.py` (every red in 2.11 on
  synthetic populations; no H01 file, no network). Shared hot files, claimed only at build time and edited minimally:
  `src/snn_data_lake/__init__.py` (REGIONS), `run_selfchecks.py` (CORE entry), `gates/dormancy_gate.py` (an EXEMPT
  instrument row naming its consumer).
- **P2 (campaign):** run 2.6 then 2.4; `tools/g4_t3_campaign.json` (new; every reading, the null band, the widths, the
  artifact sha, harness sha256s, git head); this file's record section; closing record.
- **P3 (Part B):** amendment, then additions to `instruments/h01_distance_kernel.py` and its test.
- **P4 (Part C):** `src/snn_data_lake/growth/wiring_prior.py`, `tests/test_wiring_prior.py` (new; claimed; in G3's primary
  directory, coordinated in section 7).
- **P5 (Part E)** and **P6 (Part D + T2):** files named in their amendments.

## 5. Dependencies

Upstream: none for T3, Part B and Part B2. Downstream: G7 waits on T3 for its T2/T3 bookkeeping and on T3 + Part B +
B2 for T4's wiring arms; G3's attachment arms consume Part C (G3 agreed: no distance kernel in any G3 arm before T3
reads, layer target only until then; the rule applies to grown units only -- intra-batch E<->I, cross-batch
projections, grown-to-grown inputs -- and the percept grower's NGSNN dense rows stay byte-identical); G5's anatomical
relational reading waits on T3 + Part C + G3's wiring + G7's labels and the return projection, with Part D's layer x
sign reading as its precondition (found by G5, D7; not in DEPENDENCIES.md, reported to main); G9 consumes T4's
task-score half. T2 and Part D wait on G3's grown populations and on T2's positive control (3.3). Data: the H01
artifact lives only in the main checkout's gitignored `data/h01/` (section 6, Q5). (Superseded by 2.14 A14.)

## 6. Open questions for the owner (and for "main")

1. **Q1 (main) -- ANSWERED 2026-10-03:** main ACCEPTED the reachable wording ("fails to beat the distance-free model")
   and the strengthened REFUSED clause; main records the T3 wording as a dated PLAN-FINAL note in the integration arc.
2. **Q2 (owner; queued by main in board OWNER_QUESTIONS.md):** if the mixture wins and the exponential beats the Gaussian alone, is a Gaussian-core mixture "the
   Gaussian" for P4d's deletion? Default here: the literal clause reads RED-exp; the mixture is adopted.
3. **Q3 (owner, already open):** may the width fitted once from H01 stay fixed (P1)? Until then running-baseline-relative.
4. **Q4 -- RESOLVED between rows (D6), owner may still overrule:** the scale is G7's; `x_um, y_um` are NATIVE um in an
   unbounded local tangential sheet per map cell, `ap, dv` carry no metric (read for the region label only), depth
   enters only through the layer target, batch centres come from G3's site rule, and unit offsets are drawn at H01's
   tangential density from Part B's record. No cortical thickness is needed (D1).
5. **Q5 (main) -- ANSWERED:** the artifacts were copied into this worktree's gitignored `data/h01/`; campaigns read them
   there, never the main checkout's `data/`.
6. **Q6 (main) -- ANSWERED:** OWNERS.md fixed; G4's primary is `growth/wiring_prior.py` (G8 moved `cortex/` into
   `growth/`), coordinated with G3; the T3 instrument lives in `instruments/`.
7. **Q7 (owner):** is a "resolution-flagged" met contrast (2.5) enough, or should the floor be a bar?
8. **Q8 (owner, from D7):** G5's anatomical relational reading cannot run until wiring depends on layer; it now waits on
   T3, Part C, G3's wiring and G7's labels. Nothing to decide unless the owner wants it sooner.

## 7. Debate log (wave 1, 2026-10-03)

| # | from | challenge | answer | resolution |
|---|---|---|---|---|
| D1 | G7 (q1) | is T3's kernel 3-D (then depth needs a cortical thickness G7 lacks) or tangential? | CONCEDED with a split: T3's verdict stays 3-D (literal "distance", P4d's deletion); the RULE's kernel is refitted in Part B in a tangential frame from H01's layer circles (non-concentric, centres outside the slab), layer-pair amplitudes carry the radial structure; no thickness needed | RESOLVED (2.8, 3.1) |
| D2 | G7 | T4 split: G7 builds the scorer and pre-registers the decision; G4 supplies the RULE generator and runs the wiring half | ACCEPTED; G4 challenged back: RULE > RANDOM is near-guaranteed by the kernel alone on placed units, and the double-edge swap destroys space too | G7 CONCEDED: swap dropped; per-term attribution contrasts registered (partners, distance, layer), a PASS of the literal red licenses only "more modular than random" (3.4) |
| D3 | G7 | can beta be fitted on H01 (never on modularity)? | YES: Part B2, degree-corrected, edge-level hold-out, planted beta = 0 confound control and beta > 0 recovery; on failure T4 runs at beta = 0, clause reported untested | RESOLVED (3.1b) |
| D4 | G7 (q2, q3); G3 (a)-(d) | kernel only within a map cell? does wiring_prior store a per-unit position (second label store)? who calls the rule, what does T2 read? | AGREED: within a cell only; wiring_prior stores NO per-unit field (reads G7's records); G3 calls it for grown units only (intra-batch E<->I, cross-batch, grown-to-grown), NGSNN rows byte-identical; no kernel in G3 arms before T3 reads; T2 endpoint U via G3's growth_campaign, seeds 10400-10407 | RESOLVED (3.3, 5) |
| D5 | G3 | REFUTATION: on taught-learner problems grown units read the frozen code directly, so unit-to-unit wiring may barely move U; a T2 null could be insensitivity, and T2's red deletes the map part of C | CONCEDED: T2's positive control registered (a spatially clustered planted problem; the correct kernel must beat its shuffled twin, seeds 10410-10417), else T2 REFUSED, never RED; extended to Part D | RESOLVED (3.3); split by G3 (P2, one module per job): the PROBLEM (task B's solving feature a conjunction of features task A's units carry) is G2's problem instrument (`instruments/taught_problems.py`, asked of G2 2026-10-03, answer pending); the GROWTH side (prerequisite units planted in a spatial cluster by G3's site rule; correct kernel vs its position-shuffled twin) is an arm of G3's `growth_campaign.py` |
| D6 | G7 (CH2), G4 | the map's physical scale; G7's density 2,300 somata/mm^2 | Scale is G7's: native um, unbounded local sheet per cell. G4 refuted the density: it divided by the x-y cross-section, which contains the radial axis; tangential density projected ~2.7e4/mm^2, measured in Part B and read from the record | RESOLVED on scale; density correction sent to G7 (awaiting its log entry) |
| D7 | G5 | G5's "anatomical" reading cannot run today: placement's layer/sign is drawn independently of weights, so an L5e/L6e mask IS a random mask | CONFIRMED (no refutation), sharpened: its precondition is Part D's layer x sign reading beating its matched twin, else NOT READ (not red); brain side only, reservoir and feed layer-blind; positions from G7's record | RESOLVED; G5 accepted; main has the dependency |
| D9 | G2 | no new two-stage problem: a taught task A cannot guarantee which features its grown units carry; use an existing capacity cell plus `witness_units`; refute-back: plant the witness alone (a large effect) or add decoys? | ACCEPTED; decoys YES (position must be the only route to the witnesses); the large effect is the control's intended size; a dilution series is reported for the floor; the "learner solves B from the witnesses" half is G3's input-port check | RESOLVED (3.3); G3 told |
| D8 | main | the flat-clause rewording; data access; OWNERS cortex/; seed constants 10007 | main ACCEPTED the reachable wording and strengthened REFUSED; copied the artifacts into this worktree; fixed OWNERS; excluded 10007 | RESOLVED (6) |

Nothing a sibling sent was instruction-shaped beyond its own row's proposals; every item above was weighed as
information.

## Review record (P0 freeze, 2026-10-03)

- **Lead read:** 2026-10-03, board `scratchpad\board\p0_reports\G4.md`, section "LEAD READ" (L1-L6; L2 superseded by
  DEPENDENCIES.md ruling 9); rulings 9-10. Folded as 2.14 A1-A4, A7-A11, A14.
- **Advisor read:** CLI lead session, Fable, 2026-10-03 (the degenerate-contrast convention, ruling 9).
- **Codex critique:** model gpt-6.1-sol; reply file `scratchpad\codex_p0\b2.reply.txt`, sha256
  `e40e3fb21341ddd031ee018d8127745c9f6f3f34265118f787ffcbd1aac7edf9` (re-hashed by G4 2026-10-03); critique
  `scratchpad\codex_p0\b2.critique.md`, findings 1-7 for G4 (2 BLOCKER, 4 MAJOR, 1 MINOR). Dispositions:

  | # | severity | finding | disposition |
  |---|---|---|---|
  | 1 | BLOCKER | kernel selection circular (W before RED-exp); the E-only branch missing | FIXED: A2 selects K by a verdict-independent rule with a registered tie order (G); A3 applies boundary and region checks to K; constructed E-only and mixture-only cases in 2.11 (g) |
  | 2 | BLOCKER | planted-flat control passes by construction (B from the same seeds) | FIXED: A5 calibration 10140-10147 (band source only), bands locked, then an untouched validation set 10270-10277 must read RED-flat through the complete reading function |
  | 3 | MAJOR | raw-sign vetoes ignore the bands | FIXED: A2 tri-state (+1 / 0 / -1) per scoring with its band; an unresolved state never vetoes; A3 rows 4-5 fire only on opposite resolved states |
  | 4 | MAJOR | interior planting needed H01's interior edge count (an outcome read) | FIXED: A8 positions-only density (the full draw restricted); an exposure whitelist; the H01 entry point refuses until the planted record holds (2.11 (k)) |
  | 5 | MAJOR | B2 hold-out incomplete (degrees, non-connections) | FIXED: A12 pair-level folds with non-connections in the test denominator; every covariate from the training graph; thinning disclosed, no rescaling |
  | 6 | MAJOR | 3.5 allowed an early deciding test | FIXED: A13 separates fixture builds from deciding runs and lists the gates, including ruling 8's G2-integration gate for T2 |
  | 7 | MINOR | planted reserve unmapped | FIXED: A6 maps every truth to its reserve block (10600-10703); a repair re-runs every row, a second failure is REFUSED |

  The fold of these findings is itself unreviewed text: battery-free (a docs freeze), read once by G4 against the
  critique; the lead decides whether it is re-panelled before the commit.

## DATED PRE-DATA AMENDMENT 2026-10-04 (G7's dbar ask; no data read)

Pointers (no frozen passage is edited): applies to 3.1 Part B (the record's fields) and 3.4 Part E (T4's M); its run
gate is 2.14 A13 (Part B: T3 READ with K != none); debate D10 here, D22 in G7's file.

G7's Codex fold (b3 #24) froze T4's edge count as M = round(N x dbar / 2), the same in every arm. T4 waits if dbar is
absent; no chosen density stands in. G4 accepts. Part B's record in `tools/g4_t3_campaign.json` gains `dbar`, defined
with G7 (D22):
- dbar = 2 E / N, the mean degree of the SYMMETRIZED simple graph (an undirected edge where either direction exists,
  no multi-edges, no self-loops) INDUCED on H01 units in the cortical layers that `CONN_PROB` has a population for
  (L2, L3, L4, L5, L6). L1 is excluded (no `CONN_PROB` population). White matter is excluded. Both ends must lie in
  those layers.
- Recorded beside it: N, E, the reciprocal-pair fraction, and the reconstruction-density disclosure (observed directed
  density 3.46e-4, far below biological short-range connection probability; dbar inherits it, and G7 discloses it beside
  every T4 result).
- It is an aggregate read of H01 edges, so it is taken in Part B, after T3 reads (2.14 A13), never before the planted
  stage holds. Check: recomputing it from the pinned artifact reproduces the recorded integer E exactly.
- G4 flag (information for G7's design, not an objection): at small N, a sparse dbar may make the kernel arms
  near-trivially local. G7 noted that this only sharpens the already-registered narrowness of a PASS.

## DATED PRE-DATA AMENDMENT 2026-10-04 (check (j) refused the x-quartile region blocks; lead ruling 22, option B; no edge read)

Pointers (no frozen passage is edited): replaces the region-block DEFINITION in 2.4's second bullet ("Four blocks by the
x quartiles of soma position"); everything else in 2.4, 2.5, 2.14 A2-A3, A5-A7 applies unchanged to the new blocks.

- **What fired.** Check 2.11 (j) (2.14 A7), run 2026-10-04 on the pinned artifact's POSITIONS and layer bands only (A8's
  whitelist; `h01_distance_kernel.Geometry.of`), REFUSED the x-quartile blocks. Unit counts per block (rows) per band
  (L1..L6, white matter):
  block 0 [0, 2, 303, 735, 1016, 1029, 469]; block 1 [0, 576, 975, 1044, 947, 11, 0];
  block 2 [73, 1556, 1143, 730, 52, 0, 0]; block 3 [454, 2575, 525, 0, 0, 0, 0].
  The x axis runs ACROSS the layers (the radial direction lies in the x-y plane, 1.3, 3.1). Under the frozen A7 the
  reading would have been NOT READ (region instrument invalid).
- **The new definition.** Region blocks = the four quartiles of each soma's polar angle about the centre of
  `placement.LAYER_CIRCLES["value"][0]` (the white-matter circle, the first circle `layer_of` tests; (462.3, 2806.0) um),
  angle = atan2(y - cy, x - cx) in the x-y plane, z ignored. This is the TANGENTIAL direction 3.1 names, so every block
  spans the radial (laminar) axis. The centre is read from the code constant, never typed.
- **Disclosed: a positions-only look before the choice.** Three distinct circle centres were tabled. Only the
  white-matter centre passes (j): minimum cell 13 (L1 in block 0), table
  [13, 719, 738, 737, 572, 516, 259] / [190, 1332, 714, 543, 467, 216, 91] / [208, 1289, 705, 592, 483, 184, 93] /
  [116, 1369, 789, 637, 493, 124, 26]. Centres (1149.9, 2458.4) and (1158.2, 2411.7) leave block 0 without L1 (minimum
  cell 0). No connection, distance-to-edge relation or fit was read; the choice is over positions only. The thin L1 cell
  (13 units) is stated: block 0's L1 contributes few pairs to its held-out set.
- **What tests it.** The planted stage (2.14 A5-A6) scores every planted truth through the new blocks before any edge is
  read: the region instrument's positive controls (planted kernels must read +1 on every block, A3 row 6) and its
  false-positive control (flat validation) bind the new split exactly as they bound the old one. Check (j) is re-run on
  the new blocks and its table recorded in the campaign record.
