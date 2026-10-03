# G1(c) / T109 part (c) -- carried state through the selected pool (pre-registration and record)

Plan row: PLAN-FINAL G workstream, G1 / T109 (the row, its dated notes items (1)-(5), and Part F's G-R1 and G-R2). Inherited
and frozen: `evidence/g1a-causal-contract.md` (the causal contract, the recorder, the no-pool arms) and
`evidence/g1b-pool-selection.md` (the pool, the reference readout, the Settled twin, section 2.6's instrument, decision rule and
power). G1B section 2.6 says this pre-registration "may only strengthen" that text; every item below that is not a restatement
is marked **(strengthening)**, and none relaxes a frozen sentence. The owner's work loop (memory, 2026-10-02 and 2026-10-03)
orders G1(c) as a single-head campaign after the seam arc's push, with the arc split in section 2.14.

## 1. Territory, re-taken at the base `e1b3ae3c` (the pushed seam head) before the design

Every fact here was re-taken at `08616d88` by six read-only readers and spot-checked by the orchestrator (the design-map
record is the operator's scratch note, out of tree). `git diff --stat 08616d88 e1b3ae3c` touches only the seam's record
(one evidence file and the two battery JSON files), so no code fact moved; the four module sizes below were re-counted at
`e1b3ae3c`.
1. G8-3 has landed: no anatomy package remains (`faculties/` 92 modules); the G1 code lives at
   `instruments/memory_diagnostic.py` (1,730 lines), `instruments/pool_selection.py` (1,567), `reservoir/recurrent_pool.py`
   (932), `instruments/paired_contrast.py` (112). The trainer retirement touched none of them.
2. Present: the pool (g1 LSM, `CANDIDATES == ("lsm",)`), the harness, the causal contract, the reference readout
   (`ReferenceReadout`, refit every 20 pairs, lambda 1.0, bias unpenalized), the Settled twin, the power rule
   (`power_s(0.0436) == {"S": 31, ...}` asserted in `pool_selection.self_check`), and the critical value
   `t_quantile(0.95, 30) = 1.6972609` (pinned to 1e-7 by `tests/test_pool_selection.py::test_t_quantiles_match_the_printed_tables`).
3. Absent (each built by the arcs in section 2.14): an exact sign-flip false-positive rate at S = 31
   (`paired_contrast.permutation_fpr` enumerates 2^S masks; 2^31 = 2,147,483,648), the R-b6 seed as a parameter
   (`pool_selection.RB6_SEED = 4242` is a constant), the oracle-beats-(a) test through the decoder, the G2(ii) generator and
   G-R2's construction check, the within-block time shuffle, a window-0 pool (`_Pool.__init__` refuses `window < 1`), the
   transport arms, a KernelTick-hosted g1, and a G1(c) campaign driver. No committed record names a seed in 300-361 or the
   synthetic seed 6000 (scanned with positive controls on 4242, 203 and 0-7).
4. The tuned configuration {k 8, w 0.25, lambda 0.9, g 4, T_in 1} is not a code constant (it is text in G1B 3.4 and the
   plan); it lies in `recurrent_pool.GRIDS` and passes `canonical`.
5. The live stream carries little usable history for a competent reader: G1B 1.10 found the reference readout's
   (e1) - (e1-0) near +0.0006 (sd 0.0067) on 100-103, and G1B 3.4 found live D near -0.006 for every candidate (g1's
   per-seed D -0.0063, +0.0082, +0.0007, -0.0276) while the synthetic D_task was 0.1266. A primary null is the likelier
   outcome; section 2.6's stream-history reading exists so that a null can be worded.
6. G1B 3.4's learner-side term: with Settled read from the first percept the "window term" averaged -0.00906 on 12 exposed live
   recordings (sd 0.01569, t -2.00) and +0.00103 on 32 synthetic seeds. Its sign convention is not stated there; section 2.6
   defines it.
7. **Design rehearsal at `e1b3ae3c` (2026-10-03, out of tree; exposed seeds only).** G1(b)'s built steps `record` (12
   recordings, 511 s at 4 workers), `mechanism` (4 s) and `instrument --pilot` (19 s) all COMPLETED at the pushed head: R-b8's
   seed-100 source digest equals the G1(a) pilot's (1880143915), both halves hold on all eight scored arms and both reds
   fired, so the stream did not change across G8 and the seam; R-b6 on 4242 reproduces its pinned values (readout (e1) -
   (e1-0) +0.12394; history-free red -0.00424). Descriptive dev-seed readings through the reference readout, seeds 100-103:
   (f64) - (a) 0.000, +0.102, +0.027, +0.054 (mean +0.046); (e1) - (e1-0) -0.001, -0.000, -0.006, +0.010 (mean +0.0006).
   They decide nothing (2.4 (ii) and 2.6 are frozen on 300-330); they say the oracle clause is likely to hold and the
   stream-history reading likely not, so a primary null would likely be worded "stream-null".

## 2. Pre-registration (frozen at the commit that lands this text; later changes are dated amendments made before any
## G1(c) recording, each a strengthening or a specification of an item this file leaves to a named design measurement)

### 2.1 What is measured and what a result licenses

- **Primary endpoint.** D = the reference readout's accuracy(Carried) minus accuracy(Settled) on the same live recording, a
  None prediction a miss, the mean over the recording's available scored segments (repeated-canonical, unseen-order,
  repeated-permuted); a recording with none halts the campaign as a harness defect (G1B 2.5). The grower's D is reported
  beside it, never decisive (G1B 2.1, 2.6).
- **A PASS licenses only** "carried state improves prospective prediction through this fixed decoder on this stream": never a
  benefit to the incumbent grower, never adoption (adoption is a later confirmatory family with its corrected bar stated
  first), and never a claim about the engine's reservoir, which only G-R9 can claim (plan, G1 dated note 2026-09-25).
- **A null** refuses "carried state helps" on this stream and nothing else: the pool is kept, G4 does not wait on it, and it is
  not a stop of the work loop (it is not a null on an owner-drawn part; POLICIES P4d).

### 2.2 Inherited and frozen

- Pool: g1 LSM at {k 8, w 0.25, lambda 0.9, g 4, T_in 1}, N 256, window 256 (the adaptation window), S_settle 64, W_in
  N(0,1)/8 seeded `recording_seed*31+7`, realization seed `recording_seed*31+11+INDEX`. The configuration becomes one frozen
  constant in the campaign module with a test equal to this text; nothing re-tunes it.
- Forms: Carried (never reset), Reset, Settled (reset, then 64 uncounted steps on the current percept, then T_in counted),
  Front-end; Reset and Front-end are reported.
- The causal contract (G1A 2.1), the views and their channels, the binding (R-b7) and the revision binding (a replay at a
  head other than its recording's is refused).

### 2.3 Statistic, bar and inference

- **Decision.** One-sided paired t over the S = 31 per-seed differences against the alpha = 0.05 critical value at 30 d.f.
  The bar is `T_BAR_S31 = 1.6973`: `t_quantile(0.95, 30) = 1.6972609` rounded UP at the fourth place (3.9e-5 stricter), as
  `T_BAR = 1.895` rounds 1.8945786 up at the third; a build assertion ties the two. The null boundary is 0; Delta = 0.02 is the
  design effect, not an acceptance threshold; the estimate and its one-sided 95% lower bound (mean - T_BAR_S31 * sd / sqrt(31))
  are reported against Delta (G1B 2.6).
- **Exact false-positive rate (strengthening of Part F's preamble at S = 31).** Beside every contrast, the fraction of the 2^31
  sign flips whose t exceeds the bar, computed by meet-in-the-middle inside `paired_contrast.permutation_fpr`, one mechanism for
  every n. The definition stays `paired_t(flipped) > bar` with `paired_t` unchanged. The fast path decides a flip by its
  RECONSTRUCTED total S' = s_left + s_right against the threshold thr = n * bar * sqrt(Q / (n (n - 1 + bar^2))) (Q = sum d^2,
  invariant under flips; t is increasing in S' for S' > 0), and every flip whose total lies within a certified band
  |S' - thr| <= eps of the threshold is RE-DECIDED by `paired_t` on its reconstructed vector, as are the two flips whose
  variance is zero. eps is derived in the build as a conservative bound on the combined floating error of the half-sum
  totals, the threshold and `paired_t`'s own mean and centred-deviation arithmetic, and it is never below 1e-9 * sqrt(n * Q)
  (about 10^5 times that error at n 31); the band's population is printed. Brute force survives only as the test oracle; n
  above 40 is refused, never approximated. Its build owes equality with brute force on at least 160 random samples plus
  adversarial fixtures (cancellation: large and tiny magnitudes together; a dynamic range of 1e-8 to 1; samples constructed
  with a flip exactly at the threshold), every existing n = 8 recorded value unchanged, and the reds in 2.10.
- **Exactness conditions, disclosed (research, 2.16).** The sign-flip null is exact under independence across seeds and a
  symmetric per-seed difference; each seed varies both the stream and the pool realization. **Skew disclosure
  (strengthening):** the sample skewness and the sign count of the 31 differences are printed beside every contrast; a met
  contrast whose skewness is below -1.0 (about 2.3 standard errors of skewness at n 31, sqrt(6/31) = 0.44) reads
  "met, skew-flagged". Left skew inflates the one-sided size (a mean-zero left-skewed difference read 0.0965 at S 31 in the
  research reproduction).
- Ties are counted; no statistic is normalized by a quantity a control destroys; no majority-of-seeds rule.
- Every contrast is computed by `paired_contrast.contrast(name, diffs, T_BAR_S31)` on exactly 31 values; a shorter list (a
  TIMEOUT, a missing cell) halts, never averages over what completed. **For every decision in this file a contrast is met only
  if it is non-degenerate AND its t exceeds T_BAR_S31** (G1B 2.6: t above the critical value); `contrast`'s rule that a
  zero-variance positive sample is met (kept for its existing n = 8 callers) does not apply here: a degenerate row is reported
  as degenerate and read NOT MET, and its false-positive rate is still printed through `permutation_fpr`.
- **Reported beside every contrast (G1B 2.5; plan row "repeated-pack and unseen-order reported separately", "accuracy
  beside prediction entropy"):** per seed, per form and per scored segment, the accuracy through BOTH learners, prediction
  entropy, coverage, the prior-fallback rate, the degeneracy and startup diagnostics, and the input-response statistics
  (driven-unit mask, the mean and sd of per-unit drive); the report stage writes these tables beside the contrast rows.

### 2.4 The blind harness (G-R1's REFUSED clause), read before any pool replay of a live seed

- **(i) R-b6 on synthetic seed 6000**: the reference readout's (e1) - (e1-0) on `rb6_stream(6000)` >= `RB6_BAR` = 0.05, and on its
  history-free twin < 0.05.
- **(ii) The oracle through the decoder**: per seed s in 300-330, accuracy(f64) - accuracy(a), both through the reference
  readout, contrast met at T_BAR_S31. The grower's reading of the same difference is reported beside it. (G1A's pilot read
  (f64) above (a) in 12 of 12 cells through the GROWER; this clause has no prior reading through the decoder.)
- Either clause failing makes G-R1 read **REFUSED (blind harness)**. The campaign still runs every later stage and reports
  each reading descriptively under that verdict; no later contrast is read as a verdict.

### 2.5 The primary and the G-R1 reading

| inputs (in this order) | G-R1 reads |
|---|---|
| 2.4 (i) or (ii) fails | REFUSED (blind harness) |
| primary not met | RED (carried state not shown), worded "pool-null" if 2.6's stream-history reading is met, else "stream-null" |
| primary met, and the shuffle's D_shuf contrast met (2.6) | RED ("its advantage survives a within-block time shuffle": not within-block order) |
| primary met, D_shuf not met, the shuffle's positive control blind (2.6) | NOT READ (the shuffle clause has no sighted instrument) |
| primary met, D_shuf not met, positive control sighted on the surviving host | PASS (2.1's license only) |

The rows are read top to bottom and the first that applies decides: a met D_shuf reads RED whether or not the shuffle's
positive control is sighted, and no combination with a blind control reads PASS. A met primary with skewness below -1.0
carries "skew-flagged" in every row. The reading is computed by a pure function with a constructed red for every row and for
each blind-control combination (2.10).

### 2.6 Controls beside the primary

- **Stream-history reading (strengthening; carried opening 2026-09-25).** Per seed, accuracy(e1) - accuracy(e1-0) through the
  reference readout on the same 31 recordings, contrast at T_BAR_S31. It only words a primary null (2.5); it is not a
  precondition. Its instrument's positive control is 2.4 (i).
- **Within-block time shuffle (G-R1's red clause; definitions are strengthenings, the plan leaves them open).**
  - A block is 256 consecutive events by index within one segment (the packs are 256-periodic); a final partial block is its
    own block; x_1 (in no segment) never moves; every block is shuffled, warm-up included, so the readout trains and is tested
    on the same surrogate.
  - Within a block the percept order is permuted with the percept's fields moving together (x, c, source, attended,
    prediction, fallback); the slot fields (t, tick, segment), each event's causal |LP| (a property of the time axis read by
    the grower's gate) and every sleep row stay in place. An event key in neither set refuses the transform.
  - K = 4 shuffles per recording, permutation seeds from `SeedSequence([recording_seed, TAG, k])`; a block of two or more
    events moves at least half its events (asserted on the drawn seeds before the first recording; no re-draw).
  - Per seed, D_shuf = the mean over the K shuffles of accuracy(Carried) - accuracy(Settled) through the decoder, on the same
    pool realization. **"Survives" means contrast(D_shuf) is met at T_BAR_S31.** contrast(D - D_shuf) is reported as the
    plan's "incremental benefit". A survival reads "not within-block order", never "not history" (Schreiber and Schmitz 2000,
    s.6.1), and no magnitude guard is added (a guard would make the RED harder to reach).
  - **Positive control (sighted shuffle):** on `rb6_stream(6000)`, (e1) - (e1-0) through the decoder >= 0.05 unshuffled and
    < 0.05 shuffled (pool-free; run in the instrument stage), AND, on the host that survives 2.9 and after the migrate stage,
    the pool's D_task (Carried - Settled) >= 0.05 unshuffled and < 0.05 shuffled (the mean of the same K shuffles). Either half
    failing makes the shuffle blind (2.5's NOT READ row). **A/A:** on the history-free twin both stay < 0.05 shuffled and
    unshuffled, the pool half on the surviving host. **Settled invariance (surviving host):** in every block lying wholly after
    the adaptation boundary (pool delivery count > 256), the multiset of Settled phi rows is equal before and after the shuffle
    while their sequence differs; warm-up blocks are excluded, because every form runs as Carried through the window (G1B 2.2).
    Its red: the same check on Carried, on a fixture that exercises history, must find the multisets different. The binding
    of every pool-dependent control names the surviving host, and a PASS requires it to match the primary's.
- **Settled-from-first-percept (strengthening; G1B 3.4's opening).** The pool admits `window = 0` (no adaptation window; the
  boundary fingerprint is taken at construction). Settled@0 replays every recording; D0 = accuracy(Carried) -
  accuracy(Settled@0) and the window term W = D0 - D = accuracy(Settled@256) - accuracy(Settled@0) are reported per seed with
  their contrasts, never decisive. Carried@0 is replayed on every seed and must be byte-identical to Carried@256 (phi, state
  chain, both learners' predictions and fingerprints). W is the COMBINED window intervention, not an isolated training-history
  term: window 0 changes the Settled features of the first 256 deliveries (the readout's training rows), the feature the
  readout holds for its next prediction at the delivery boundary (it predicts from the previous phi), and any scored row that
  falls at or before delivery 256 (segments follow acquisition pointers, the window follows the pool's delivery count). The
  report counts the scored rows at or before the boundary per seed.
- **Startup flag** per seed, beside D (G1B 2.2, 3.4: g1 flagged on 4 of 4 selection recordings).
- **Stream identity = R-b8 as built** (first job of the campaign): seed 100 re-recorded at the campaign head; its
  `_source_digest` compared with the G1(a) pilot recording (`archive/g1a-2026-09-24/pilot3_full.tar.gz`), then both halves of
  R-b8 with their reds (G1B 2.7). A differing digest or any failed half HALTS the campaign as a shared-instrument failure, as
  `pool_selection.step_instrument` does today. The difference is then attributed to a commit before anything re-runs; if it is
  an intended behaviour change, the owner is briefed before G1(c) continues. The design rehearsal at `e1b3ae3c` (section 1
  item 7) ran R-b8 as built on the head G1(c) branches from, and it held. Disclosed: the loop's prep skeleton (2026-10-02)
  had made a changed stream "disclosed, not a stop"; the Codex design read restored G1B's halt.

### 2.7 The transport family (plan item (1)-(2); secondary; bar frozen here)

- **Source (the brain's weights at boundary b).** b = 256, the adaptation-window boundary; the copy is taken at the START of
  delivery 257, after every refit through the sleep ticks before it and before any later pair, from an arm-(a) replay of the
  same recording (the pool arm's own grower reads pool phi and is not the incumbent; a transport read from it would be
  circular). Copied: the grower's input weights W (L x 64, rows fixed from birth). Not copied: its bias and its category-space
  readout beta (no 64-column meaning; beta moves on every pair). Disclosed.
- **Copy form ("w@b").** From percept 257 the pool's encoder is the random draw with rows 0..L-1 replaced by W's rows, each
  rescaled to the mean row norm of the random draw; rows L..255 stay random; the padding fraction (256 - L)/256 and the raw row
  norms are recorded (the grower held 2-4 units at the end of every pilot replay: padding 98.4-99.2%). Refused if L == 0 at b
  (never padded from nothing) or L > 256.
- **Activity form.** The pool is driven by the grower's hidden rates for the CURRENT percept, computed before c_t is
  delivered (G1A 2.1's prediction-then-update), zero-padded to the encoder's 64 inputs; events before the grower's first model
  drive zeros (counted; none is expected in the scored segments); refused if L_t > 64.
- **Declared channels.** pool-copy reads ("x", "w@b") and nothing else; w@b is None before b; pool-act reads ("grower_activity",)
  only. A transport that reads any pair at or after b, or the live weights after b, must RAISE (built like R-a2b's forbidden
  map, G1A 2.2): the red mutant keeps pool-copy's permitted declaration ("x", "w@b") and REQUESTS "w_live" while that value
  is present in the replay's values; the view must raise "forbidden feature channel". A second check validates every
  transport arm's declaration against the permitted channel set, with a red arm that declares "w_live".
- **Readings (secondary; bar T_BAR_S31; never decisive for G-R1):** per form, D_t = Carried - Settled through the decoder;
  contrast(D_t - D) against the random-encoder form, and contrast(D_t - D_swap), where the **swap control** swaps the encoder
  at b to a fresh random draw of equal row norms (so the encoder discontinuity at b is not mistaken for transported content).

### 2.8 The separation reading (plan item (3)) and G-R2's construction check

- **Generator** (the plan's G2(ii)): texts "cat dog sun sky", "cat dog sky sun", "dog cat sun sky", "dog cat sky sun" with u =
  "dog precedes cat", v = "sky precedes sun"; XOR labels GREET iff u == v, else GOAL (the plan's A/B, as two real intents of
  `_IXN_INTENTS`). **Separable twin (strengthening; the plan names it, never defines it):** the same texts and code, labelled
  GREET iff "cat precedes dog", else GOAL. Home: a new module `instruments/inseparable_mapping.py` (G2, G3 and G9 import it).
- **Construction check (G-R2 clauses 1-2, run first, at the campaign head):** A1 the four words occupy four distinct buckets of
  52 (asserting the plan's 28, 41, 31, 51 and re-derived independently in the test), none in buckets 0-1; A2 phi00 + phi11 =
  phi01 + phi10 in the learner's PSP-of-rank code, exactly on the integer ranks and within 1e-12 in float (the plan read
  1.1e-16); A3 the bare incumbent learner (`SpikingInteractionLearner(n_in=52, t_post=6)`) under the pre-registered teaching
  schedule never answers all four XOR texts correctly at any step on any of 8 check seeds, and 0 of 2,000 random weight
  settings do; A4 it learns the twin (4 of 4 at the last 10 epoch ends on every check seed), with a witness weight setting that
  reads 4 of 4 through the real `route_text`. A5 (witness only) two hand threshold units over the code separate XOR. The
  learner-level presized nonlinear control and the delayed-cue control are G2's (T110), not run here.
- **Design numbers set before the freeze of the check, on development seeds only** (learner 7100-7103, pool 100-103): the
  teaching schedule TEACH_EPOCHS = 2 x the largest dev epochs-to-stable, rounded up to a multiple of 50; the arm-(a) floor on
  the twin stream and the pool-readability floor on the twin. They enter this file as a dated amendment before any check seed
  (7000-7007) or campaign seed is run.
  *DATED AMENDMENT 2026-10-03 (pre-data: no check seed 7000-7007 and no campaign seed has been run; it specifies the numbers
  this bullet left to the development measurement, each a strengthening):*
  - **TEACH_EPOCHS = 400.** On learner seeds 7100-7103 the first run of 10 consecutive 4-of-4 epoch ends began after 122,
    170, 112 and 124 epochs, and the last epoch end NOT reading 4 of 4 within 4,000 epochs was 179, 170, 162 and 161 (both
    reproduced by the orchestrator). The rule reads the stronger of the two: 2 x 179 = 358, rounded up to 400 (the weaker
    reading gave 350). A full construction check on the four development seeds took 3.01 s at 350 epochs.
  - **Floors, both 0.75**, the affine ceiling on the XOR set, which a decoder can exceed on the linearly separable twin: arm
    (a)'s mean decoder accuracy on the twin streams must reach 0.75, else the stream is wrong; a pool form's XOR reading is
    "readable" only if that form's mean on the twin reaches 0.75, else it is reported as "pool blind to the code". Means are
    over the seeds read (300-330 in the campaign).
  - **Development readings** (decoder accuracy, pool seeds 100-103, streams built and replayed out of tree at `deb2be25`):
    arm (a) XOR 0.485, 0.429, 0.573, 0.510 and twin 1.0 on all four; Settled XOR 0.992, 0.989, 0.991, 0.988 and twin 1.000,
    0.997, 1.000, 0.993; Carried XOR 0.555, 0.682, 0.621, 0.680 and twin 0.784, 0.823, 0.884, 0.897. Disclosed: Settled's
    near-perfect XOR reading is the pool's threshold nonlinearity separating four fixed codes (a Settled pool is a function
    of x alone), which this section's reading anticipated; Carried mixes the previous presentation into the state.
- **Pool reading (reported for G9, no bar):** a synthetic stream per live seed 300-330 (so the pool realizations are the
  primary's): 1,200 events, texts i.i.d. uniform, x = the 52-dim PSP-of-rank code scaled to norm 1.73 and zero-padded to 64,
  c_t = label(text_{t-1}) (the decoder predicts c_t from phi_{t-1}, so each label answers the previous presentation; disclosed),
  warm-up 256 then "unseen-order" (the harness's tag). Replays: {XOR, twin} x {arm (a), Carried, Settled}. Reported: decoder
  accuracy and per-text accuracy, distinct Settled phi rows. Reading rules: arm (a) on XOR must stay below 0.75 + 4 binomial
  sd (the affine ceiling) or the stream is wrong; an XOR reading is "readable" only if the same pool form clears the twin floor,
  else it is reported as "pool blind to the code".
- **If the construction check reds,** item (3) is reported NOT RUN and G1(c) continues (owner, work loop 2026-10-03).

### 2.9 The migration gate (plan item (5); G1B 2.9)

- **Hosted form:** `KernelLsmPool`, the same network as the port (same draws, a separate host dimension, never a second
  candidate) run on `KernelTick`: fixed-point weights and drive at the lineage's binding `frac_bits` (12), integer accumulation,
  float32 membranes, decay lambda, threshold 1.0, reset 0, no refraction, delay 1; one host tick per port step.
- **Prediction, frozen as a prediction:** whole-trajectory bit parity with the float64 port is NOT expected at frac_bits 12
  (weights and drive round at 2^-12; the port moves under a 1e-4 threshold nudge, G1B 3.6); parity is expected on lattice
  fixtures only. The parity horizon (first diverging tick; equal-activity fraction) is reported for frac_bits 4-32.
- **Gate:** (i) parity fixtures in the host's arithmetic: Class A exact (the hand fixture through all four forms, delay 1, the
  threshold boundary, refraction, lattice quantization and its rounding convention, duplicate edges, the mid-flight lesion,
  identical structure) and Class B declared divergences (a weight below half an LSB, a float32 membrane rounding, a float32
  decay product found by scan) asserting the host's arithmetic; (ii) at the campaign head, |mean D_hosted - mean D_port| <= 0.005
  on 204-207 re-recorded there **AND (strengthening) the same on D_task over the synthetic 5005-5008**: G1B 3.4's candidate
  comparison (four mechanisms within 0.0008 on live D) did not demonstrate that live D is sensitive to the targeted carrying
  defect (substituting Settled for Carried moves g1's mean D only from -0.0063 to 0, a margin of 0.0013 over the tolerance),
  while D_task (0.1266) falls toward 0 when carrying is destroyed; the gate's machinery red is a hosted Reset standing in for
  Carried, which must fail the D_task conjunct; (iii) R-b3 and R-b4 on the hosted form. R-b3's red is re-founded because the
  port's permuted-float-sum red cannot fire on integer accumulation, and because lattice weights at 2^-12 sum exactly in float32
  in any order: the red is a float-accumulation twin of the hosted pool fed RAW (unquantized) weights with a declared
  magnitude spread (within each unit's incoming set, weights spanning at least 2^12 in magnitude), its fixture asserted to
  exercise the spread (at least 100 unit-steps with three or more simultaneous presynaptic spikes) and the twin's continuation
  chain asserted to differ under an unseeded delivery order through the hosted adapter's own comparator;
  `determinism_gate.broken_control` runs beside it as a wiring control only.
- **Disposition:** all three hold -> the hosted form runs every later stage; any fails -> the port runs them and the migration
  is an opening (G1B 2.9), unconditionally. Disclosed: P(PASS) for a faithful host is not 1. The D_hosted values across
  frac_bits 4-32 are reported as quantization sensitivity, never used to explain a FAIL away. The loser and the host switch
  are deleted after the results record (G1B 3.5 precedent) in a commit whose survivor is checked by golden fixture fingerprints
  equal before and after (2.11), and a test then counts one reachable pool implementation.

### 2.10 Checks, each with a constructed red (the build arcs owe every one; a red that does not fire halts its arc)

- **Exact FPR (P1):** equality with brute force on >= 160 samples (n 8-14, ties included) plus lattice samples, n 18 and 20;
  invariance to reversal, shuffle, negation, scaling and |d| at n 31; reds: a `>=` boundary mutant on the exact-boundary sample
  [3.0, 0.0] at bar 1.0, the bar dropped from the threshold, the var' = 0 correction removed, bar <= 0 and n > 40 refused;
  every existing n = 8 recorded value unchanged.
- **Bar tie (P1):** `t_quantile(0.95, 30) <= T_BAR_S31 < t_quantile(0.95, 30) + 1e-4`; the same predicate is False for 1.895 at
  30 d.f. and for 1.6973 at 7, 29 and 31.
- **R-b6 seed (P1):** default 4242 byte-identical to G1(b) (file names, calls); 6000 carries 6000 everywhere.
- **Oracle through the decoder (P1):** hand digests where the grower reads f64 > a on every seed but the decoder does not must
  read NOT MET; a missing or TIMEOUT cell halts; a named symmetric zero-mean fixture (15 pairs of +-0.02 and one 0.0) and an
  all-zero fixture are not met (sign balance is never a decision rule; Part F forbids a majority-of-seeds rule); a constant
  positive fixture is degenerate and NOT MET under 2.3's rule.
- **R-b1 (inherited causality, G1B 2.7):** G1(a)'s R-a2 on the surviving host's Carried and Settled forms at the tuned
  configuration, both learners, at the three cuts, on seed 203 re-recorded at the campaign head; the (leak) red must change
  predictions at every cut. A failure halts as a shared-instrument failure.
- **Seed sets (P1):** live and reserve disjoint from each other and from every exposed set; a reserve seed in the primary table
  is refused by R-b7.
- **Construction (P2):** A1 reds (a duplicated word; n_in 3; swapped bucket table); A2 red (a text breaking additivity); A3 red
  (the twin labelling through the XOR counter reaches 4 of 4); A4 reds (lr 0, shuffled teaching labels, the XOR labels in the
  twin slot); stream reds (lag 0, a stream inside the window, permutation-block texts, a twin stream drawn from another RNG state).
- **Window 0 (P3):** Settled@0's first phi equals an independent reset-settle derivation; a window silently clamped to 1 must
  fail it; Carried@0 == Carried@256 with a red (a Carried whose boundary resets); a window-0 replay filed as window 256 is
  refused by R-b7; R-b0..R-b5 at the default window byte-identical before and after.
- **Shuffle (P3):** the transform's reds (cross-block, c-only, identity, a slot field moved, an unknown key, a non-contiguous
  live block); Settled invariance with its Carried red and a skip-a-carrier red; the "survives" decision function returns True
  for a planted order-independent gain (a decoder fed the current category, synthetic seeds 6100-6107 at the Part F bar) and
  False for the order-dependent R-b6 family.
- **Transport (P3):** channel reds (forbidden-copy raises with the value present; pool-act cannot read x); timing by two
  instruments that must agree (the tap equals the last refit at or before b; a truncated-stream replay reproduces it) with a
  late-tap red; causality at three cuts with a final-model-copy red; the tap is read-only (a writing tap must change R-b8's
  equality).
- **Migration (P4):** each Class A fixture with its wrong-host twin (reset value, refraction 1, delay 2, `>` for `>=`, a
  truncating quantizer, de-duplicated edges, a lesion without zeroing the in-flight charge, a second-candidate realization);
  Class B fixtures assert the divergence itself; the gate's machinery red (a hosted Reset standing in for Carried must fail the
  D_task conjunct); `within()` boundary and NaN reds; wrong-host binding reds.
- **Driver (P5):** the seed-set assert, preflight refusals (wrong cwd, OUT inside the repo, an asset override set), the stage
  chain and single-head refusals, a recording TIMEOUT halts, the stream-identity red (one perturbed x), the G-R1 reading table
  (one red per row), the non-halting construct stage, R-b4 twins and the pool-noise red, the variant bindings (window, host,
  encoder, shuffled stream), and a mini-campaign test on the synthetic fixture with a tampered pilot that must halt R-b8.

### 2.11 Failures, repairs and reserve seeds

- A shared-instrument failure (a red that does not fire, R-b1, R-b2, R-b4 learner difference, R-b7, R-b8 in either half, a
  harness defect) halts the campaign. A FAIL of a job is a defect: fix, new head, every recording and replay re-run. No retries.
- **After the first recording, a change to code the campaign executed** (a repair, a panel fold, a closing-report fold, a
  battery fix, the deletion of 2.9's non-surviving host) is tested by the **golden check**, run in-process at the campaign
  head and at the new head: the pool's hand fixtures and R-b3's fixture chain on the surviving host, a full replay of the
  48-event synthetic fixture stream through every arm and form (predictions, phi, both learners' fingerprints, pool state
  chains), and the report stage's pure functions recomputed over the campaign's saved summaries. Every output must be
  byte-identical; any difference restarts the campaign at a new clean head on the reserve seeds (below), and disclosure never
  substitutes. The earlier results stay attributed to their own head. The golden check is built in P5 with its red (a one-ulp
  change to the pool's leak must move it).
- **Before the first recording,** every non-pytest stage of the battery (ruff, ruff format, `run_selfchecks` in full, the 16
  gates `--assert`, the census) runs on the stacked head, and each arc's touched test files run per commit, so that a battery
  red after the campaign is unlikely to touch campaign code.
- **Reserve seeds (strengthening):** live 331-361. Inspection starts when any replay of a live seed 300-330 has been written. A
  repair after inspection exposes 300-330 and re-runs the whole campaign on 331-361 in a new OUT; a second repair halts G1(c)
  and is brought to the owner. Synthetic reserve: 6001 for R-b6, 5009-5012 and 208-211 for the gate (G1B 2.8's reserve).

### 2.12 Seeds

Live 300-330 (S = 31, asserted equal to `power_s(0.0436)["S"]`); reserve 331-361; identity 100; R-b1 203 (a G1(b) tuning
seed, read here only by the causality check); gate 204-207 (live) and
5005-5008 (synthetic), already exposed by G1(b) and read here only by the gate; R-b6 6000; shuffle-red synthetic 6100-6107;
G2(ii) learner check 7000-7007, learner dev 7100-7103, pool dev 100-103. Exposed, never confirmatory: 0-7, 100-103, 200-211,
4242-4247, 5001-5012. Seeds 303 and 321 appear as unrelated self-check RNG seeds in three modules; they are not recordings.

### 2.13 The campaign (one clean head; nothing committed from the first recording to the results record)

Stages, each refusing a dirty tree, a predecessor at another head, and a re-run in the same OUT: record (seed 100 alone and
timed first, the one-seed cost check, and its R-b8 digest comparison; then 203, 204-207 and 300-330, 36 more recordings) ->
mechanism (R-b0x, R-b3/R-b5 on the tuned configuration) -> instrument (R-b2, R-b7's reds, R-b6 on 6000 with its red and the
shuffle's pool-free positive control, R-b8's two halves) -> references (a, f64, e1, e1-0 on 300-330: 2.4 (ii) and the
stream-history reading) -> construct (2.8, non-halting) -> migrate (2.9) -> causality (R-b1 on seed 203, surviving host) ->
primary (four forms on 300-330 on the surviving host, R-b4 twins on every seed, the pool-noise red) -> controls (the shuffle's
pool positive control, A/A and Settled invariance on the surviving host; the shuffle on 300-330; window 0) -> transport ->
separation -> report (one machine-readable verdict: every contrast row with name, n, mean, sd, t, bar, lower bound, exact FPR,
band population, skewness, signs, met; and 2.3's per-seed, per-form, per-segment tables). Launched detached from the repo
root with OUT outside it, the main checkout's `.venv-lake`, no asset or device override. Cost, recorded figures (not yet
measured at this head): 37 recordings about 27 min at 4 workers (the rehearsal recorded 12 in 511 s at `e1b3ae3c`), about
650-800 replays about 15-25 min; the seed-100 timing is printed with the projection before the batch starts.

### 2.14 Arc split, commits, review and battery

One arc per part, stacked on one branch off `e1b3ae3c`: P0 this file and the plan's dated notes; P1 instruments (the exact FPR
and the bar in `paired_contrast`; the R-b6 seed parameter in `pool_selection`; and a new module
`instruments/carried_state_campaign.py` holding G1(c)'s frozen constants, the seed sets, 2.3's contrast row and the oracle
check, registered in the commit that adds it); P2 the G2(ii) generator and construction check (the named pre-campaign
commit); P3 the pool-side arms (window 0, then the shuffle, then the tap and transports); P4 the hosted form and its fixtures;
P5 the campaign driver (a default-preserving refactor of `pool_selection`'s step, job and table machinery first, then the
stages added to the P1 module, which imports that machinery). Each arc: its registrations in the commit that adds a module, the worktree census after every
symbol-adding commit, one Codex gpt-6.1-sol panel (three lenses) before the campaign, a fold stated as re-panelled or
battery-verified only. Then the campaign, the results record (the first commit after the first recording), the exit duties
(the dormancy rows that say "until G1(c)'s disposition", the loser of 2.9, the plan's T109 note), the Codex closing report,
ONE battery on the stacked head, the closing record LAST, push. A post-recording change to executed code restarts the campaign
on the reserve seeds when 2.11's golden check moves; a text-only change does not.

### 2.14a This file's own review (2026-10-03, before the freeze)

A Codex gpt-6.1-sol design read of the draft (read-only) filed 15 findings (4 BLOCKER, 8 MAJOR, 3 MINOR); each was checked
against the cited source and all 15 were folded into this text before the freeze: R-b8's halt restored (it had been made a
disclosure); the degenerate-contrast rule and the t > bar requirement; post-recording code changes restart on the reserve
seeds; R-b1 restored on seed 203; Settled shuffle invariance limited to post-window blocks; the pool-dependent shuffle controls
moved to the surviving host; the exact-FPR boundary defined on the reconstructed total with a derived band; the hosted R-b3 red
given a magnitude-spread witness; the sign-balanced red replaced by a named symmetric fixture; the forbidden-map red made a
request for an undeclared channel; window 0 described as a combined intervention; the per-segment and entropy tables made
mandatory; the shuffle-blind prose aligned with 2.5's precedence; the frac_bits spread renamed quantization sensitivity; the
live-D sensitivity claim corrected.

### 2.15 Scope

Built: the items of 2.14. Not built: live wiring of the pool or the transports; any change to the reference readout, the
recorder or the causal contract; G2's presized nonlinear and delayed-cue controls; a growth target inside the reservoir
(owner-held, engine-structure section 5); T096-T100.

### 2.16 Prior work (cited summary; raw notes out of tree; web numbers are context, never bars)

- Sign-flip tests are exact under independence and symmetry, not normality (Winkler et al. 2014, NeuroImage 92, "Permutations
  and exchangeability", Table 1); a permutation p-value includes the observed assignment and is never zero (Phipson and Smyth
  2010, s.5). The sum of a sign-flipped sample with fixed sum of squares orders t monotonically, so counting sums past the
  threshold is the exact rate (the derivation is the build's, checked against brute force).
- Reservoir practice resets state per sequence for sample independence and discards an initial transient (Lukosevicius 2012,
  s.4.5; Jaeger 2001/2010, s.3.2); Maass et al. 2002 randomize the initial state per trial and use the state difference under
  different initial conditions as a noise level (s.5). Settled is the field's default; Carried is the departure under test.
- A shuffle keeps a series' values and destroys its serial order; a within-block shuffle keeps block composition, block order
  and slow drift, so a surviving effect is "not within-block order" (Schreiber and Schmitz 2000, s.3.1, s.6.1), and a surrogate
  test also rejects on non-stationarity (s.6.1; Theiler, Linsay and Rubin 1993). Block resampling is the complement (Kunsch 1989).

## 3. Results

(Empty until the campaign's results record.)
