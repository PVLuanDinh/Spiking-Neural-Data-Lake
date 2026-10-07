# G5 / T113 -- readout variants against the incumbent `OnlineReadout` (pre-registration DRAFT)

**Status: P0 pre-registration, 2026-10-03, wave 1. Folded after the lead read, the advisor read and the Codex critique
(Review record at the end). The lead commits it on `claude/g5`. Every number not marked "measured" is a projection.** Row agent a2a188231e63cd173. Branch `claude/g5`, base `e1b3ae3c`. Seeds come
from the pre-assigned block 11000-11999 (board SEEDS.md).

## 0. The engine sentence this row serves (owner memory 2026-09-30: evaluate against the ENGINE's function)

P4d's engine: the reservoir drives READOUT NEURONS (connection 4). The brain tunes them by a learning rule whose teaching
or modulatory signal is a brain output (connection 5). They return relational information to the brain (connection 6:
"THE NEURONS GAIN RELATIONAL PROPERTIES AND INFORMATION TO CONNECT TO THE BRAIN"). G5 owns 4-6 on the readout side and
G9 charges the whole engine.

The engine-functional test is Part D: after a neuron is fed in, a brain-tuned readout recovers its relation to the brain
on held-out data, against path-breaking controls. **Part D waits for G1(c).** Parts A and B run now. They are P2 contests
on today's substrate, which is the incumbent's own home stream. They decide which tuning rule (A) and which relational
property (B) Part D carries. By themselves they do not test the engine's function, and this file never reads them as if
they did.

| part | what | engine connection | runs |
|---|---|---|---|
| A | trace-delta challenger vs `OnlineReadout` (delta) | 5 (tuning rule) | now |
| B | categorical relational reading: lateral coupling learned from the confusion matrix | 6 (property), read side first | now |
| C | anatomical relational reading: read L5/L6 units with sign +1 | 6 (property of the return projection) | WAITS on G4 + G7 + the return path (section 5, a dependency DEPENDENCIES.md does not list) |
| D | spiking readout neurons on the reservoir, brain-tuned (STDP / predictive coding / topographic), and the return path | 4, 5, 6 | WAITS on G1(c)'s disposition, plus G2, G7 and G3's feed for named controls |

## 1. Territory (read at `e1b3ae3c`; runtime evidence marked "measured")

- **The incumbent.** `readout/online_readout.py::OnlineReadout("delta")`:
  `beta -= lr * s^T (softmax(s @ beta) - onehot(cat))`, `lr = MU / (||s||^2 + 1)`, `MU = 1.0`. MU is a P1 exception (a)
  carried from laminar, an owed step sweep. It is called from `growth/neurogenesis_engine.py::NeurogenesisSelfImprove.observe`
  right after the live instrument scores the pair (s = the rates of the PREVIOUS percept, cat = the category that
  followed), so it learns test-then-train. The brain constructs it (`engine/brain.py`, `online_readout="delta"`). It
  landed in L1c (`evidence/l1c-online-readout.md` 4b) with t 4.132, mean +0.143 over the batch-only twin, 8 seeds, in-brain.
  It sits below the live persistence floor on 8 of 8 seeds. **It was never held to a label null.** L1c 3.3 records that
  "DELTA and SIGN were never held to a label null" and that a permutation of a one-category window is the identity. Its
  charge on the deployed brain is 0.0 (one silent unit). That is an open P3 item handed to L2 / SC-012a.
- **Where the row comes from.** The 2026-09-22 grow-reservoir debate, round 3 (memory
  `archive/grow-reservoir-eval-2026-09-23/synthesis.md` #3). It specifies: keep the next-category alignment; "Eligibility
  x error is a trace-delta variant, not spike-timing dependence"; "MERGE-OR-DELETE as a variant of `OnlineReadout`". Its red:
  "it loses to delta; OR its gain survives a shuffled teaching signal or neutered coupling".
- **The retained model's lifecycle (read in `NeurogenesisSelfImprove._refit`).** Under a CLOSED gate, `_refit` scores the
  held-out slice with the batch instrument and returns early: the same model object, the same units. Only at the end of
  an OPEN-gate refit, after `prune_unearned` and the placement reindex by `last_keep`, does it assign
  `self._model = model` and re-clone `_batch_beta`. A model-object change is therefore exactly the event at which the
  units can change. 2.2's trace lifecycle keys on it.
- **The substrate is stateless.** `growth/ng_snn.py::NGSNN._rates` sets `V = torch.zeros(...)` for every batch of percepts
  and has no recurrent term (read at `e1b3ae3c`). Inference reads only
  s(t-1), so a presynaptic trace can act only through LEARNING: it spreads credit over earlier presynaptic patterns.
  Whatever a trace can gain on this substrate comes from presynaptic autocorrelation. A problem whose answer lives only
  in history (the S1 cells of G2(i)'s delayed cue) is unsolvable by ANY readout here. That is why the engine form of a
  trace belongs to Part D (a stateful reservoir). G2's problems are taught to the chat learner, never to NGSNN or
  `OnlineReadout`, so Part A never touches them.
- **Placement carries no function.** `growth/placement.py::place` draws each grown unit's layer from H01 occupancy and its
  sign from the layer's E-fraction. The RNG is seeded `(engine seed * 1_000_003) + unit index`
  (`NeurogenesisSelfImprove`'s refit), which is independent of the unit's weights. No reader exists
  (`evidence/s1-placement-null.md`; `tests/test_placement_charge_null.py`). **Measured 2026-10-03 (this row):**
  P(a unit is L5 or L6 with sign +1) = 0.1706 from `cortical_prior.LAYER_OCCUPANCY` x `LAYER_E_FRACTION`. At 2-7 units
  the L5e/L6e mask is empty with probability 0.69-0.27. The seed-11000 cell grew 5 units, 0 of them L5e/L6e.
- **Incumbent search (P2).**
  - Categorical coupling: none. `grep -i confusion` over `src/` fires on a known positive
    (`faculties/affect_core.py::_confusion_section`'s confusion matrix), and no readout couples categories.
    `graphify query` (the main checkout's graph, read-only) was also run and FAILED its positive control: "affect core
    confusion matrix section" started at `instruments/neuro_chip.py::Core` and never reached `_confusion_section`. Its
    "readout that couples category outputs through a learned confusion matrix" query returned `learned_evc` and
    `world_red_probe` nodes only. The no-incumbent claim therefore rests on grep, the instrument whose positive control
    fired, and not on graphify.
  - Trace incumbents: `plasticity/r_stdp.py` (a per-synapse spike-timing eligibility trace times reward: a different
    object, pre-post timing rather than presynaptic rate history) and `plasticity/eligibility_algebra.py` (the declared
    sum-of-products authoring contract for three-factor rules, specs/022 T050). Section 2.2 says how the trace uses them.
  - Part D's named rules: `readout/predictive_coding.py` (adjudicated 2026-08-14, NOT a violation; open exact-symmetry
    scope), `reasoning/spiking_som.py` (G x G Kohonen map, BMU plus a decaying Gaussian neighbourhood),
    `plasticity/r_stdp.py`.
- **Engine inventory (memory, 2026-10-03).** The reservoir is instrument-only. Spiking readout neurons and the return path
  are spec-only. The manifest has no engine ports.
- **Cost, measured 2026-10-03.** One in-brain cell in the L1c harness shape: `make_brain(seed=11000, persist=False)` in
  `throwaway_home()`, 800 ticks, deployed wiring (delta), torch threads 1, one process. G1(c)-era host load was present.
  Wall time 59.8 s (make_brain 1.5 s, 800 ticks 57.9 s). Real perception packs present in this worktree
  (`_perception_unified` True). 399 wake percepts, 20 refits, 5 units, 5 categories. Live accuracy 0.912 against live
  persistence 0.944. The readout made 376 calls and 371 non-zero steps, sum |dbeta| 148.4.

## 2. Part A -- the trace-delta challenger (runs now)

### 2.1 Question

On the incumbent's own measurement, does a presynaptic-eligibility delta rule beat the delta rule? The comparison is
live next-category accuracy on the wake stream, scored before learning, in-brain, on the same rows. If it wins, is the
win attributable to the trace: the teaching pairing, rather than a smaller effective step or label-free drift?

### 2.2 Mechanism (nested: lambda = 0 IS the incumbent)

**Where the code lives until it earns a place.** The candidate is first a subclass of `OnlineReadout` INSIDE the
instrument (`instruments/g5_readout_cells.py`). The brain never constructs it. Only if 2.7a's gate is met AND Part A
lands does the rule move into `OnlineReadout` itself (the P2 merge, 2.8). On deletion only the instrument's candidate
and its arms go, and `readout/online_readout.py` is never touched. The rule, with the same s and cat the incumbent
uses:

    e   <- lambda * e + (1 - lambda) * s          # presynaptic eligibility: an EMA of the readout's input rates
    p    = softmax(s @ beta);  p[cat] -= 1         # the output layer's OWN error on the CURRENT prediction
    beta -= MU / (||e||^2 + 1) * outer(e, p)

- **Next-category alignment kept.** The error is formed on the prediction the instrument just scored. Credit goes to the
  presynaptic history e.
- **The EMA form is chosen so that ||e|| is near ||s|| within a run.** A raw sum would shrink the effective step by
  about (1 - lambda) and confound "trace" with "smaller step". The flipped delta already read ABOVE the incumbent at
  mu 0.05-0.3 (L1c 2.0c), so a step-size confound is real here. The A-R1b arm below still controls for it.
- **Reach.** At lambda = 0, e = s and the update is byte-identical to today's `learn`.
- **Lifecycle (disclosed).**
  - e resets to zero whenever the retained model OBJECT changes. That happens at an open-gate refit, where the batch
    solve re-initializes beta; the trace restarts with it.
  - e continues across a closed-gate refit (same object, same units).
  - A widen adds beta columns, not rows, so e is untouched.
  - A row-count mismatch between e and beta RAISES. It never broadcasts.
- **GRADIENT_CLAIM.** It stays LOCAL. The read set is s, its own EMA, beta's own columns and cat; nothing is transposed and
  no hidden error is formed. The `rule_composer` census row for `OnlineReadout.learn` is re-worded in the P4a merge
  commit (section 8), only if the rule merges.
- **P2 against the trace incumbents.** The update is ONE term, dbeta = c * e (x) err, and the build declares it through
  `eligibility_algebra`'s contract if that contract admits a vector outer-product term. If it does not, the module states
  why (the contract's operand shape) and the open item goes to the panel. `r_stdp`'s trace is a spike-timing coincidence
  trace and is not reused. A `distinct-from` note records that.
- **lambda is a P1 debt (CHOSEN on dev seeds, never typed).**
  - The grid is lambda in {0, exp(-1), exp(-1/3), lambda_run}. lambda_run = exp(-1/tau_run), where tau_run is the mean
    category run length measured on the dev seeds' recorded sequences. Projection: about 18 percepts at persistence
    0.944, so lambda_run is about 0.95.
  - The argmax of the mean paired dev gain over DELTA is frozen with its value printed.
  - **Stop rule:** if the argmax is lambda = 0, or its mean dev gain is <= 0, Part A reads NULL AT DEV. No confirmatory
    cell runs, and 2.8 applies.
- **The step-matched MU (for A-R1b).** On the dev seeds, MU_step = MU x mean over seeds of
  (TRACE's sum|dbeta| per non-zero step / DELTA's). It is frozen with lambda*.

### 2.3 Arms (in-brain, the L1c harness: `make_brain(seed)`, `throwaway_home()`, 800 ticks, threads pinned)

| arm | readout | role |
|---|---|---|
| BATCH | none (beta only at open-gate refits) | the NEUTERED TUNING PORT: the brain's teaching signal does not reach the readout between refits |
| DELTA | `OnlineReadout("delta")` | the incumbent (deployed wiring) |
| TRACE | lambda* | the challenger |
| TRACE0 | lambda = 0 | reach control: must equal DELTA bit for bit; it doubles as the determinism A/A |
| DELTA_STEP | delta at MU_step | attribution: the trace vs a smaller CONSTANT step |
| DELTA_SCHED | delta DIRECTION outer(s, p), at the trace's EVENTWISE update scale | attribution: the trace's historical direction vs its adaptive step schedule (Codex #10) |
| TRACE_SHUF | lambda*, taught the SHIFTED category sequence | shuffled teaching signal (G-R5) |
| DELTA_SHUF | delta, taught the SHIFTED sequence | the incumbent's first label null (a finding about `OnlineReadout`, never a G5 red) |
| TRACE_FLIP | lambda*, sign -1 | the control that is not the method (directional) |
| DELTA_FLIP | delta, sign -1, incumbent MU | the incumbent's own flip, so A-R4 reads the trace's direction and not an incumbent property (D3) |

- **DELTA_SCHED, defined (Codex #10).** It runs the trace's EMA e in shadow, with the same lambda* and the same
  lifecycle, and never lets e set the direction. At each event the trace would apply
  dbeta_T = -c_e * outer(e, p) with c_e = MU / (||e||^2 + 1). DELTA_SCHED applies
  dbeta_S = -a_t * outer(s, p) with a_t = c_e * ||e|| / ||s|| (a_t = 0 when ||s|| = 0). That is the delta direction at
  the trace's SCALAR STEP c_e * ||e||, so the per-event step scale matches.
  - The full update norm is NOT matched: ||outer(x, p)||_F = ||x|| ||p||, and p comes from each arm's own beta, so
    ||p|| differs per arm.
  - The matching check is IN-ARM. In DELTA_SCHED, at every event, a_t * ||s|| must equal c_e * ||e|| from its own
    shadow trace, to within relative 1e-5 (float32). The maximum relative error is printed per seed, and a larger error
    halts the cell as an instrument failure. No cross-arm norm comparison is made.
  - DELTA_STEP is kept: it is the constant-step control. DELTA_SCHED is the eventwise one.

- **The shuffled teaching signal, constructed to be non-vacuous.** It is a CIRCULAR SHIFT of the seed's recorded
  category sequence by k = floor(n/2) transitions. The shift keeps the marginal and the burst autocorrelation and breaks
  the s-to-label alignment. The live instrument still scores the TRUE category; only the label passed to `learn` moves.
  The sequence is recorded from the DELTA arm of the same seed. Rows are proven identical across arms by the same-rows
  check below.
  - Printed per seed: the fraction f of transitions whose teaching label changed.
  - **VACUOUS if f < 0.20** (CHOSEN; projection f of about 0.4-0.6, from 1 - sum p_c^2 at majority 0.5-0.75). A vacuous
    seed is reported, and the A-R3 conjuncts read REFUSED rather than held.
- **Common vocabulary initialization (Codex #8, the BLOCKER).** Read at `readout/online_readout.py::learn`: a teaching
  label at or beyond beta's column count WIDENS beta. A shifted sequence teaches a later category EARLY, which adds
  columns early and changes softmax competition. TRACE could then beat TRACE_SHUF through that structural change
  rather than through the alignment.
  - The fix: every arm's readout starts with the seed's FULL recorded category set allocated. M_seed = 1 + the largest
    category index in the seed's recorded DELTA-stage sequence. That is the column count `learn` would reach by the
    end, and it does not assume the indices are contiguous.
  - At every model-object creation (the start and every open-gate refit), in EVERY arm (BATCH, DELTA, TRACE, the
    shuffled, flipped and step arms, and Part B's arms), the harness calls `OnlineReadout.widen(model, M_seed)` before
    the first scoring. The initialization is identical across arms: the batch solve's beta, zero-padded to M_seed.
  - The live scoring then sees the same M_seed columns in every arm. Shifting labels moves only the alignment, never
    the category availability, column allocation or initialization.
  - `learn`'s own widen never fires inside a cell. The `widened` counter is printed and must be 0 in every arm.
  - Part B's count matrix C is M_seed x M_seed from the first event in every coupled arm.
  - Disclosed: the common init changes DELTA from the deployed wiring, which widens lazily. A-R0 is therefore read
    under the common init. If A-R0 fails here although L1c held, Part A reads REFUSED (instrument), never a challenger
    null.
- **Same rows (halts on a mismatch).** Per seed, every arm must have identical category-sequence hashes and identical
  majority, persistence and live-persistence floors to 1e-6. This is L1c 2.3's check plus the sequence hash.
  - Added for Codex #8: the hash of the per-event column-count schedule (beta.shape[1] at every scoring) must also match,
    and the `widened` counter must be 0.

### 2.4 Statistic (shared by A and B)

- `instruments/paired_contrast.contrast(name, diffs, T_BAR=1.895)` over S = 8 paired seed differences. The bar is
  `T_BAR = 1.895`; no new statistic.
- A contrast is MET only if it is non-degenerate AND t > 1.895 AND at least 5 seeds are non-tied. Ties are
  |d| <= 1e-6, the harness's `TIE_TOL`.
- **The FPR is a diagnostic, not a p-value (Codex #9; read in `instruments/paired_contrast.py`).**
  `permutation_fpr(diffs, bar)` counts the sign flips whose t exceeds the FIXED BAR, not the flips at or beyond the
  observed statistic. It is the exact false-positive rate of the bar on this sample, printed beside every contrast as
  Part F's preamble requires, and it gates nothing.
  - The draft's earlier "exact sign-flip p <= 0.05" conjunct (from L1c 2.4) is DROPPED. As a gate it would refuse the
    module's own MET all-positive self-check example, whose FPR is 13/256 = 0.051.
  - Disclosed: dropping it makes this rule less strict than L1c 2.4's on that one conjunct. No permutation p-value is
    registered.
- **The five-non-tied restriction is kept as a separate, named restriction.** Its purpose is to refuse a contrast
  carried by three or fewer moving seeds while the others tie. On this stream a readout change often leaves a seed's
  accuracy unmoved, and a t over mostly-tied differences is fragile.
- A degenerate (zero-variance) contrast reads NOT MET, under board DEPENDENCIES.md ruling 9: one convention for every
  G/R row in one integration. That is stricter than `paired_contrast.contrast`'s zero-variance rule, and the pushed
  precedent is G1(c) 2.3. It applies to every contrast in Parts A and B, the brain-free planted checks included.
- **Pairing is asserted (Codex #13).** Every paired difference is built from two cells that carry the same seed key. The
  harness asserts it per difference and halts on a mismatch.
- Printed beside every contrast: the tie pattern (+/-/0), the exact FPR, the sample skewness, and per arm
  (live acc - live persistence).
- There is no majority-of-seeds decision rule.
- Two families (A, B) share the BATCH and DELTA cells. No multiplicity correction is applied, as in house practice.
  Disclosed: each part's decision stands alone.

### 2.5 The reds (each MET / NOT MET by its letter)

- **A-R0 (instrument positive control, read FIRST): contrast(DELTA - BATCH) MET.** The incumbent's L1c charge must reappear
  on these seeds. If it does not, the regime differs and Part A reads REFUSED (instrument). It is never read as a
  challenger null.
- **A-R1a (the challenger beats the incumbent): contrast(TRACE - DELTA) MET.**
- **A-R1b (not a constant step-size effect): contrast(TRACE - DELTA_STEP) MET.**
- **A-R1c (not the adaptive step schedule; Codex #10): contrast(TRACE - DELTA_SCHED) MET.** This credits the trace's
  HISTORICAL DIRECTION e. If A-R1a and A-R1b are met but A-R1c is not, the gain is attributable only to the complete EMA
  update, including its eventwise step schedule. That is reported in those words, and the trace does NOT land: the
  registered claim is the presynaptic history, and a schedule-only gain is a different claim needing its own amendment.
- **A-R2 (reach): TRACE0 == DELTA on 8 of 8** for the W hash, beta hash, live accuracy, units and trajectory. Otherwise no
  verdict is read.
- **A-R3 (the gain depends on the teaching alignment; D1).** All three conjuncts must hold:
  1. the shift is non-vacuous (f >= 0.20) on 8 of 8 seeds;
  2. contrast(TRACE - TRACE_SHUF) is MET. This is the deciding conjunct, and it is a reachable red: a gain that
     survives the shift fails it;
  3. contrast(TRACE_SHUF - BATCH) is NOT met. This secondary conjunct can only add a refusal, never rescue a pass.
  - Reported, not a bar: contrast(TRACE_SHUF - DELTA_SHUF).
  - Why the old form was replaced: it passed when a contrast was NOT met, which a noisy or underpowered shuffled arm
    satisfies trivially.
- **A-R4 (the direction matters for the trace, read against the incumbent's own flip; D3).**
  contrast(DELTA_FLIP - TRACE_FLIP) MET decides.
  - Printed beside it: contrast(BATCH - TRACE_FLIP), and contrast(DELTA_FLIP - BATCH), which re-reads L1c 2.0c's
    flipped-above-delta finding on these seeds.
  - If DELTA_FLIP reads above BATCH, that is reported to the lead as a property of the incumbent (P8 clause 4), never
    as a G5 red.
  - Disclosed: L1c 2.0c measured the flip at mu 0.05-0.3, so its sign at MU = 1.0 is unmeasured. Comparing with BATCH
    alone could have red for that incumbent property.
- **A-R5 (the growth gate's evidence is untouched):** refit accuracy and the whole trajectory equal DELTA's on 8 of 8.
  This holds by construction under the decoupled instrument (L1c 3.3) and is verified, not assumed.
- **Reported, not a bar:** contrast(DELTA_SHUF - BATCH). If the incumbent's own gain survives the shift, that finding is
  reported to the orchestrator as P8 clause 4: the incumbent's landing had no label null. It limits what A-R1 can
  attribute, and it is not a G5 red.

**Decision:** the trace LANDS iff A-R0, A-R1a, A-R1b, A-R1c, A-R2, A-R3, A-R4 and A-R5 all hold. Otherwise it is
deleted (2.8). Part A has 10 arms (2.3).

### 2.6 Brain-free checks, each with a constructed red (the P1 build owes every one; a red that does not fire halts the arc)

All run on the real engine (`NeurogenesisSelfImprove`) under `ClosingGate`, fed by ONE frozen generator (below). The
planted seeds are listed in section 4.

- **The frozen generator (Codex #11): `planted_chain(seed, kind, noise, n=600, k=4, run=20, d=64)`.** One
  `torch.Generator` is seeded once per call, and draws happen in this fixed order:
  1. Prototypes P_0..P_{k-1}, i.i.d. N(0, I_d), each scaled to unit norm.
  2. The category sequence.
     - For kind in {burst, drift}: runs of FIXED length `run`. Each new run's category is uniform over the k - 1 other
       categories. The first category is uniform over k.
     - For kind = iid: c_t is i.i.d. uniform over k for every t. Successive (category, feature) pairs are then
       independent, so there is no history to exploit beyond chance repeats at rate 1/k.
  3. The features, with eps_t i.i.d. N(0, I_d) per event.
     - burst: x_t = P_{c_t} + noise * eps_t.
     - drift: x_t = (1 - w_t) * P_{c_prev} + w_t * P_{c_t} + noise * eps_t, where w_t ramps linearly from 0 to 1 over
       the first floor(run/2) events of a run and is 1 after. The run's first category has c_prev = c_t.
     - iid: x_t = P_{c_t} + noise * eps_t.
  - The engine observes (x_t, c_t) in order.
  - The 2.6a pilot used out-of-tree scratch generators whose "i.i.d.-switching" chain matches kind = iid. Pilot
    numbers are never re-read as PA results.
- **PA1 (the trace's positive control; reachable red for A's claim).** The 2.7a family (kind in {burst, drift}),
  discovery on 11900-11907 and replication on 11936-11943. contrast(TRACE - DELTA) is MET by 2.7a's replication rule.
  contrast(TRACE - DELTA_SCHED) is printed beside it.
  - If NO configuration is met and replicated, Part A has no reachable red for its claim. It is DELETED, not weakened
    (P8 F11), before any in-brain cell.
- **PA2 (the trace's negative).** It runs only if 2.7a's PA1 is met, at the replicated configuration's noise and lambda,
  with kind = iid. contrast(TRACE - DELTA) must NOT be met on 11908-11915.
  - Why the old wording was replaced: "an i.i.d.-within-run chain" kept the shared run prototype that the EMA can
    average (Codex #11). kind = iid removes the run structure itself.
- **PA3 (the shuffle's positive control; D1).** An oracle readout whose input is one-hot(previous category), on kind =
  burst at noise 2.0, under delta and the common vocabulary init (2.3). Seeds 11916-11923.
  - PA3 is MET iff contrast(ORACLE - BATCH) is MET AND contrast(ORACLE - ORACLE_SHUF) is MET. The second contrast is the
    deciding conjunct.
  - contrast(ORACLE_SHUF - BATCH) NOT met is a secondary conjunct that can only add a refusal.
- **PA4 (the vacuity detector fires on a known positive).** A one-category sequence must read f = 0, i.e. VACUOUS.
- **PA5 (reach).** lambda = 0 is bit-identical to today's `learn`, brain-free and in-brain.
- **PA6 (lifecycle).**
  - A model swap must zero e. Red: a trace that survives the swap fails it.
  - A row mismatch must raise. Red: a broadcasting trace fails it.
- **PA7 (determinism).** The same seed twice gives identical beta.
- **PA8 (same rows).** A planted one-label perturbation in one arm's sequence must halt the same-rows check.
- **PA9 (home guard; board PROTOCOL 4b; Codex #14).** A directory-name prefix is not isolation, because a
  `selfcheck_home_*` path can be a junction or symlink to the live lake. So the harness ADOPTS ONE SHARED GUARD, common
  infrastructure shared with R1's I7 (agreed with R1, section 10 row 8), and never writes its own. Its contract:
  1. create a fresh temporary directory;
  2. resolve it (`Path.resolve(strict=True)`). Refuse if the resolved path equals the RESOLVED live home, or if either
     path contains the other;
  3. set NORD_HOME BEFORE any lake-binding import or Brain construction.
  - The live home is a parameter, so the red runs on a decoy and never touches C:\Users\vinhl\.snn_data_lake.
  - Reds:
    - a constructed prefixed alias (junction or symlink) to a decoy "live" home must refuse;
    - NORD_HOME pointing anywhere other than the guard's own fresh dir must refuse before any Brain is built.
  - The guard lives outside Part A's code, so it SURVIVES a Part A deletion: Part B still builds brains.
  - After every campaign stage the live home's mtime is checked unchanged.
- **PA10 (common vocabulary; Codex #8's red).** An early-category mutation must refuse. Construct a shifted sequence in
  which a category whose first appearance is late is taught early, and run that arm WITHOUT the common init. `learn`'s
  widen then fires early (`widened` > 0), and the column-count schedule hash differs from DELTA's. The same-rows check
  must HALT.
  - The same construction run WITH the common init must pass the check (widened = 0, identical schedule). That is the
    fix's own reach.

### 2.6a Design pilot, measured 2026-10-03 (brain-free; exploratory; seeds 11932-11935 now EXPOSED)

The question was whether a planted chain exists that lets the learning-only EMA trace beat delta. The pilot ran
2.2's rule as a scratch subclass, out of tree, on the real `NeurogenesisSelfImprove` under `ClosingGate(3)`: 600
observations, k = 4 categories, runs of 20, 64-d prototypes plus Gaussian noise, 4 seeds. Two chain shapes were used:
"burst" (a fixed prototype per run) and "drift" (the percept moves from the previous prototype to the new one over half
a run). Live persistence was 0.955-0.966 and the units grown were 2-9.

Mean paired (TRACE - DELTA) live accuracy:

| chain | noise 0.2 | noise 0.8 | noise 2.0 |
|---|---|---|---|
| burst, lambda 0.368 / 0.717 | -0.004 / -0.023 | -0.009 / -0.035 | -0.006 / -0.020 |
| drift, lambda 0.368 / 0.717 | -0.007 / -0.048 | -0.007 / -0.029 | **+0.006 / +0.002** (signs mixed, 3 of 4 and 2 of 4) |

A first run with an i.i.d.-switching chain (PA2's shape) read -0.031 / -0.078 / -0.112 at lambda 0.368 / 0.717 / 0.95.

**What this says (the session's reading, a projection for the build).** On a stateless substrate the learning-only
trace credits presynaptic history that inference never reads, so it biases beta. It can help only through variance
reduction, and here only at the highest noise with smooth drift, by about +0.006. PA1 is therefore at risk of having no
reachable positive at the bar. Section 2.7a registers what follows from that.

**The carry, verbatim in the record (lead read L5).** The trace-delta LOST on every planted stream except one marginal
cell. That cell is drift at noise 2.0: +0.006 at lambda 0.368 and +0.002 at lambda 0.717, with mixed signs (3 of 4 and
2 of 4 seeds). The i.i.d.-switching stream lost by -0.031 / -0.078 / -0.112 at lambda 0.368 / 0.717 / 0.95. This
sentence is carried unchanged into the results record, whatever 2.7a reads.

### 2.7a The trace's reachability gate (registered on the pilot; runs FIRST in P1, before any in-brain cell)

- **The frozen family.** 12 configurations: kind in {drift, burst} x noise {2.0, 3.0, 4.0} x lambda {0.368, 0.717}.
  Run length 20, on 2.6's frozen generator. The family is never widened after a look.
- **Discovery and replication (lead ruling L2, replacing "ANY configuration MET").**
  - Each configuration's contrast(TRACE - DELTA) is first read on 11900-11907 (discovery).
  - A configuration counts as MET only if it REPLICATES: the same contrast must be MET at the same bar on the planted
    set 11936-11943, with the same configuration and lambda and nothing re-tuned.
  - The replication set is run only for configurations met on discovery. It is never looked at otherwise and never
    re-drawn.
  - PA1 is MET iff at least one configuration is met on discovery AND replicates. If several replicate, the frozen
    tie-break is the largest replication mean gain, and that configuration's lambda joins the dev grid.
- **Printed beside the reading: false-pass references (Codex #15; references, not guarantees).** Per contrast, the
  nominal one-sided alpha is 0.05, because T_BAR 1.895 is the df-7 one-sided 0.05 point under the t model.
  - "Any of 12" without replication:
    - INDEPENDENCE REFERENCE: 1 - 0.95^12 = 0.46. It holds only if the 12 configurations are independent, and they
      share seeds, so it is not a bound.
    - General nominal union bound: 12 x 0.05 = 0.60, with no independence assumed.
  - With replication: at most 12 x 0.05^2 = 0.03. This bound additionally REQUIRES the discovery and replication data
    to be independent (disjoint planted seed sets drawn by the frozen generator), and each stage's per-contrast error
    to be at most its nominal alpha. The t model is approximate at n = 8, so this is also a nominal figure.
  - No claim is made that correlation lowers the unreplicated rate.
- **Why replication rather than Bonferroni (ruling L2).** It keeps the project's bar and attacks the false pass directly.
  Given the pilot's sign (the carry above), a chance pass is the dominant risk.
- **If PA1 is NOT met, Part A is DELETED before any in-brain cell.** Its claim has no reachable red on this substrate,
  and P8 F11 says such a criterion is "deleted, not weakened". The record states the pilot and the gate. The trace's
  engine form moves to Part D, where the substrate carries state, by its own amendment.
- **If PA1 is met,** the replicated configuration's lambda joins the dev grid and Part A proceeds as written.
- **Cost.** 12 x 8 x 2 discovery runs, plus at most 12 x 8 x 2 replication runs, about 0.6 s each (measured in 2.6a).

### 2.7 Null disposition (a causal claim; its instrument's positive control comes first)

A null reads: "the presynaptic trace adds nothing the delta rule lacks on the stateless substrate". Its named cause is
that inference reads only s(t-1), so the trace can act only through presynaptic autocorrelation. The instruments owing
positive controls BEFORE the null is read are:

- PA1: the instrument CAN see a trace benefit when one is planted;
- PA3: the shuffle can remove a real gain;
- A-R0: the incumbent's charge is present on these seeds.

These outcomes differ:

- PA1 NOT met means the claim has no reachable red. That is DELETION under 2.7a (P8 F11), not a refusal.
- PA3 or A-R0 failing is an instrument failure. Part A then reads REFUSED (instrument), not a null, and the instrument
  is repaired under a dated amendment before any re-read.

### 2.8 What a null deletes and what it does not (P4d, P8 clause 5)

- **A null deletes** the trace (lambda, e, its lifecycle code and its harness arms) from `OnlineReadout`.
  `OnlineReadout("delta")` stays the incumbent and the record stands.
- **A null does not delete** the readout neurons, the brain's tuning connection or the return path (owner-drawn parts).
  Part D also stays.
  - The null is scoped to the stateless substrate. Part D carries a trace only if Part A lands, or if a new amendment
    registers it for the stateful substrate.
- **A win** MERGES. `OnlineReadout`'s default becomes lambda*, as one rule; lambda = 0 stays reachable only as the reach
  control, like `mu=0` and `sign=-1` today. The brain picks it up through the module default, with no `engine/brain.py`
  edit. The strict in-tree test in `tests/test_real_percept_stream.py` carries the reading.
  - Disclosed: on the deployed brain's silent unit the trace inherits the incumbent's 0.0 charge. The P3 item stays open.

## 3. Part B -- the categorical relational reading (runs now)

### 3.1 Question

Does a lateral coupling between the readout's category outputs, learned online from the readout's own confusion matrix,
improve the live answer? This is the necessary condition, on the read side, for "readouts that couple" being a relational
property worth carrying into the return projection (Part D; G7 section 4).

**Part B named for what it is (D6).**

- Part B is a CALIBRATION LAYER on the readout's output distribution: a learned map from predicted category to true
  category.
- It is NOT the engine-functional test. P4d's connection 6, "the neurons gain relational properties and information to
  connect to the brain", concerns readout neurons on the reservoir projecting back into labeled brain units. Part B grows
  no neuron, reads no reservoir and projects nowhere.
- A Part B win is never cited as evidence for connection 6 or for the owner's 09-30 relational endpoint. It decides one
  thing only: whether a which-confuses-which coupling is a property worth carrying into Part D's return projection,
  where it is re-tested as an ablation of that projection.
- A Part B null deletes the module (3.6) and says nothing about relational properties in the engine.

### 3.2 Mechanism (new module `readout/categorical_coupling.py`; no incumbent, section 1)

- **Counts.** C[i, j] counts scored transitions where the readout predicted i and the truth was j. It is updated AFTER
  the live instrument scores, from the same pair the online rule learns from (prequential).
- **The coupling.** K[i, j] = (C[i, j] + alpha * [i == j]) / (sum_j C[i, j] + alpha), with alpha = 1 (CHOSEN, one
  pseudo-count of "the readout was right"; P1 debt). With no counts, K = I and the decision is the incumbent's exactly.
- **The coupled decision.** argmax_j sum_i q_i K[i, j], with q = softmax(s @ beta). Evidence for category i is shared with
  the categories i's predictions turn out to be.
- **Abstention.** When `decide` abstains (no unit fires), its fallback is kept untouched.
- **Scope.** Learning is untouched: `OnlineReadout.learn` reads the uncoupled softmax. The growth gate's batch instrument
  is untouched.
- **Measurement path.** The harness wraps `NeurogenesisSelfImprove.predict_next` and leaves the engine unedited until a
  win. This follows the house precedent: `_l_cell` wraps `eng.observe`.

### 3.3 Arms (same seeds and cells as A; DELTA and BATCH shared)

| arm | role |
|---|---|
| COUPLED | the reading, on the incumbent readout (DELTA) |
| COUPLED0 | counts frozen at zero, so K = I: reach control, bit-identical to DELTA |
| COUPLED_MARG | a diagonal prior (label-shift) correction from the same counts' MARGINALS only: the control that is not the method (D2) |
| COUPLED_SHUF | counts built from the circularly shifted truth sequence (as A's shift): shuffled teaching |
| COUPLED_PERM | REPORTED only, never a bar: the truth axis of C relabelled by a fixed random category permutation |

- All Part B arms run under 2.3's common vocabulary init. C is M_seed x M_seed from the first event, so the shifted
  counts never see a different category set (Codex #8).
- **COUPLED_MARG, defined (D2).** It is built from the same counts, prequential and in the same order, but keeps only
  their marginals:

      r_j = sum_k C[j, k]   (how often j was predicted)        m_j = sum_i C[i, j]   (how often j was the truth)
      K_marg = diag((m_j + alpha) / (r_j + alpha))            off-diagonal 0, alpha = 1 as in K
      decision = argmax_j q_j * K_marg[j, j]

  - It is a diagonal label-shift correction. With no counts, K_marg = I and the decision is the incumbent's exactly.
  - It cannot move evidence from one category to another; only K's off-diagonal terms can.
- **Why COUPLED_PERM is no longer the control.** On a stream with a 0.90-0.99 persistence floor, C is
  diagonal-dominant. Permuting its truth axis moves most of the mass off the diagonal, so COUPLED_PERM is worse BY
  CONSTRUCTION: a strawman. The informative question is whether WHICH category is confused with which carries anything
  beyond a plain prior correction.

### 3.4 Reds

- **B-R0:** A-R0 (the shared positive control).
- **B-R1:** contrast(COUPLED - DELTA) MET.
- **B-R2 (reach):** COUPLED0 == DELTA on 8 of 8.
- **B-R3 (the control that is not the method, and the alignment; D2 + D1).** All four conjuncts must hold:
  1. contrast(COUPLED - COUPLED_MARG) MET: which-confuses-which beats prior correction;
  2. contrast(COUPLED - COUPLED_SHUF) MET: the gain depends on the true teaching alignment;
  3. the shift is non-vacuous on 8 of 8;
  4. contrast(COUPLED_SHUF - DELTA) NOT met. This secondary conjunct can only add a refusal.
- **B-R4:** refit accuracy and the trajectory are identical to DELTA's on 8 of 8.
- **Reported:** (live acc - live persistence) per arm. The confusion matrix on this stream is persistence-shaped (the live
  floor is 0.90-0.99). A coupling that only approaches that floor is read against it and never as a relational
  capability.

**Decision:** the reading LANDS iff B-R0 through B-R4 hold. Then `categorical_coupling` is wired into the engine's
decision with its consumer. That is a hook in `NeurogenesisSelfImprove.predict_next`, G3's directory, claimed then and
messaged first. Otherwise the module is deleted.

### 3.5 Brain-free checks with constructed reds (planted seeds 11924-11931 and 11944-11951)

- **PB1 (positive).** A readout with a planted systematic relabelling: the stream's class a is always predicted as b.
  Both conjuncts must hold:
  - contrast(COUPLED - uncoupled) MET: the coupling recovers the relabelling;
  - contrast(COUPLED - COUPLED_MARG) MET: a diagonal rescale cannot move the b-predictions that are really a, so this
    red is reachable by construction.
- **PB5 (the marginal control's own positive control; a control that detects nothing is not a control).** A planted
  pure base-rate miscalibration: the readout's predictions are biased toward one class, with no systematic
  relabelling. Seeds 11944-11951.
  - contrast(COUPLED_MARG - uncoupled) MET decides.
  - contrast(COUPLED - COUPLED_MARG) NOT met is a secondary conjunct: on pure prior miscalibration the off-diagonal terms
    must add nothing.
- **PB2 (negative).** Non-systematic (i.i.d.) readout errors. It must NOT clear.
- **PB3 (reach).** Frozen counts are bit-identical to the uncoupled decision.
- **PB4 (prequential).** A planted leak, counts updated BEFORE scoring, must raise in the harness's order assert.

### 3.6 Null disposition and deletion

A null reads: "the readout's errors on this stream are not a consistent relabelling". The instrument owing its positive
control first is PB1, plus A-R0. A null deletes `categorical_coupling.py` and its arms. The owner's "relational
properties" requirement is untouched and stays build work on the return projection (Part D; P0, P4d).

### 3.7 Composition (only if A and B both land)

A composed check, TRACE+COUPLED vs TRACE on seeds 11200-11207, runs before the composed wire. The wire lands only if
contrast(TRACE+COUPLED - TRACE) is MET. Otherwise the later-landing part is re-read against the earlier one.

## 4. Seeds (block 11000-11999; grep-checked 2026-10-03)

| set | seeds | use |
|---|---|---|
| dev | 11000-11007 | lambda grid, MU_step, tau_run; B's instrument dry-run (vacuity, reach). EXPOSED after use |
| confirmatory | 11100-11107 | A and B, read once on the frozen lambda* / MU_step |
| reserve | 11108-11115 | replaces a confirmatory SEED, never a single cell, ONLY on an instrument failure (crash, TIMEOUT, `unified` False, same-rows halt), never on a result (Codex #13; rule below) |
| composition | 11200-11207 | 3.7, only if A and B both land |
| Part C / D amendments | 11400-11479 | held; assigned by their own amendments |
| planted (brain-free) | 11900-11931 | PA1 (the 2.7a gate) 11900-11907, PA2 11908-11915, PA3 11916-11923, PB1/PB2 11924-11931 |
| design pilot | 11932-11935 | 2.6a, run 2026-10-03; EXPOSED, never reused for a registered check |
| PA1 replication (lead ruling L2 / L6) | 11936-11943 | 2.7a's replication set. Never run before; run only for configurations met on discovery (11900-11907) |
| PB5 planted (D2) | 11944-11951 | the COUPLED_MARG control's own positive control |

- **The reserve rule (Codex #13).** A replacement takes the WHOLE seed block:
  - every participating arm of Parts A and B, both stages (the DELTA recording stage that produces the shifted sequence,
    then the arms), and every dependent recorded sequence, all rerun on the replacement seed;
  - reserve seeds are taken in order (11108 first), one per failed confirmatory seed. The failed seed's cells are
    discarded from every contrast, and the replacement is printed beside the seed it replaces;
  - no contrast ever pairs cells from two different seeds. 2.4's per-difference seed-key assert enforces it.
  - A repair of the instrument itself, rather than a crash, is a dated amendment, and every affected seed reruns.
- **Grep for the two added sets, run 2026-10-03 with the read-only Grep tool.**
  - Positive control: `\b4242\b` over specs/, src/ and tests/ fired, with 36 occurrences in 16 files. The "14 files"
    below counted specs/ and src/ only.
  - `\b(1193[6-9]|1194[0-9]|1195[01])\b` over the same trees hit only this file's own lines. 11936-11951 is clear.

- **Grep check.** The positive control first: `grep -rlE '\b4242\b' specs src` hit 14 files. Over specs/, src/ and tests/
  the block's hits were 11167, 11257, 11317, 11346, 11484, 11696, 11786, 11788, 11883 and 11977. Each was a DOI, an arXiv
  id, a calibration value, a rho bound or a count, never a seed. None is in a set above. The 11300-11399 range was skipped
  because it contains 11317 and 11346.
- **Never run here:** 300-361, 6000, 6001, 6100-6107, 7000-7007.

## 5. Part C -- the anatomical relational reading (WAITS; a dependency DEPENDENCIES.md does not list)

**Why it does not run now: its control would be the method itself.**

1. Section 1 shows a unit's layer and sign are drawn from an RNG independent of its weights and read by nothing. A
   readout over "L5/L6 units with sign +1" is therefore a label-selected subset whose distribution EQUALS a size-matched
   random mask's. The null is entailed, and a red that cannot fire is not a red (P8).
2. At the incumbent's 2-7 units, the mask is empty 27-69% of the time (measured).

**What it waits on.**

- Layers that carry function: G4's attachment rule (layer target x distance kernel, after T3).
- G7's labeled grown populations and their label gate.
- The return projection itself. G7 section 4: "G5's 'anatomical' relational reading becomes which layers and labels the
  projection reaches".

**Precondition (G4's sharpening, section 10 row 4).** Layer labels carry function only if (a) wiring differs by layer
AND (b) that wiring reaches the readout. (a) is G-R4's layer x sign clause: G4's layer x sign prior must beat its
degree-, distance- and source-sign-matched twin with the layer and target-sign preferences removed. If it does not,
layer is still a random label and Part C is **NOT READ**, which is not a red.

The reading is brain side only. Grown units attach to labeled grown units through G3's column batches; the reservoir
and the feed are layer-blind (G7 section 4). Positions are read from G7's record (`labels_by`); G4's
`growth/wiring_prior.py` stores no per-unit field.

**Registered now (numbers frozen at its own amendment, before its first run).**

- The return projection restricted to L5e/L6e targets vs:
  - all labeled grown units;
  - a size- and degree-matched random-label mask;
  - layer labels shuffled within the population.
- Red: no benefit over the size-matched random-label mask, or a benefit that survives the layer shuffle.
- A null deletes the reading's implementation. The return path is owner-drawn and stays (P4d).

## 6. Part D -- readout neurons on the reservoir and the return path (WAITS on G1(c)'s disposition)

Outline only. Bars are registered now; numbers and seeds are frozen at its own amendment before its first run.

- **Substrate.** G1(b)'s selected spiking pool (g1 LSM, N = 256; `KernelLsmPool` on `claude/g1c`), in the form G1(c)'s
  disposition leaves. Per the 10-02 work loop, a G1(c) null refuses only "carried state helps" and keeps the pool.
- **Readout neurons: ADOPT, never build a second spiking readout (P2; G3's debate challenge, row 2 of section 10).** The
  posed-problem job already has a spiking, brain-tuned readout: `readout/interaction_learning.py::SpikingInteractionLearner`
  as `engine/live_brain.py` builds it. It has one R-STDP neuron per label, and teaching enters through G0's
  `learn_turn(..., feedback=label)`. G2's problems are posed to it, and G3's feed arms read the reservoir through it (a
  separate first-spike latency segment, G3 draft 2.3). Part D's readout neurons ARE those label neurons, reading the
  reservoir segment.
  - **Two readout jobs exist today:** next category (`OnlineReadout` on NGSNN) and the taught problem (the label
    neurons). P4d allows one tuning rule per readout job. Open question 5 asks the owner whether they stay two jobs.
- **Each tuning rule is one arm against R-STDP, the incumbent of the posed-problem job.** P2: one tuning rule survives
  per readout job. Each rule's teaching or modulatory signal enters through one named argument.
- **The tuning signal's producer is UNRESOLVED (Codex #12; owner question 7).**
  - What the drawing requires: P4d's connection 5 says the readout neurons are "tuned by the brain through a learning
    rule".
  - What the draft has today: the signal is the teacher's external reward d, as G0's
    `learn_turn(..., feedback=label)` delivers it. A TypedPort declaration around an external signal does not make the
    brain its producer.
  - **The claim is limited until the owner answers.** While the signal is the teacher's d, a Part D win is licensed
    only as "readout neurons on the reservoir, tuned by an external teaching signal, solve the posed problem and return
    to the brain". It is NEVER credited as demonstrating connection 5, the brain's tuning connection. The results record
    states that limit in those words.
  - **Owed before Part D's amendment freezes (a freeze precondition, not build work for now).** Either the owner accepts
    external feedback for connection 5 (and the limit above is lifted by the owner's words, quoted), or the amendment
    names three things:
    - (a) the brain PRODUCER of the tuning signal: named brain neurons, for example a neuromodulatory or error output
      that receives the teacher's d and the readout's answer;
    - (b) its transformation into the modulatory factor;
    - (c) the receiving port on the label neurons.
    It then registers an ablation of that producer as a path-breaking control: with the producer neutered and d still
    delivered to the brain, the readout neurons must lose the gain.
  - delta, `OnlineReadout`'s rule ported to the label neurons;
  - trace-delta, only per 2.8;
  - R-STDP as the learner runs it today (the incumbent arm; `plasticity/r_stdp.py`'s family). Any UNSUPERVISED STDP arm
    carries L1b's co-adaptation controls: the re-solve-the-readout-on-every-refit control arm runs, and the
    mu-derivation debt (P1) stays open until paid;
  - predictive coding by ADOPTING `readout/predictive_coding.py`, citing its 2026-08-14 adjudication and open
    exact-symmetry scope. `eprop_transport_gate` must be green, and the new site is judged at its own readout by the
    distinguishing test;
  - topographic clustering by ADOPTING `reasoning/spiking_som.py`, promoted out of `reasoning/` (P2, P4c). Its control is
    the neighbourhood width at zero.
- **The posed problems, per cell (G2's correction, section 10 row 5).** G2's (i) is a 2x2 of labellings over the same
  8 three-turn episodes.
  - Only the S1 cells (S1C0 = cue, S1C1 = cue XOR p) are unsolvable by ANY readout of the current turn. This is exact:
    the route is a pure function of (w, probe text), and the probe text is identical across cue values. The S1 cells
    test the reservoir as STATE.
  - S0C1 (p XOR q) is solvable by a nonlinear readout of the current turn. It tests the reservoir as CAPACITY: there the
    reservoir arm must beat a frozen random expansion of matched size. That control is G2's presized expansion,
    REUSED (P2): `instruments/taught_campaign.py::presized_expansion(problem, H, seed).features(text)`, with H set to
    the reservoir's unit count. Without it, "the reservoir helps" reduces to "any expansion helps". G2's disclosures,
    carried:
    - (a) its per-unit threshold is a label-free median over the PROBLEM's stimulus texts, so it is defined per
      problem, not per stream;
    - (b) it is a positive control only at the H G2's dev seeds certify (projection 64; marginal at 16). The H-matched
      comparison is read only beside the expansion solving S0C1 at G2's certified H. Otherwise, an H-matched
      expansion failure is not "the reservoir beats any expansion";
    - (c) on the S1 cells it cannot help (no state, exact bound 0.5), so it is a control for S0C1 and (ii) only.
  - G5's own disclosure: that median threshold is stimulus-set-adapted, and the reservoir is not. A reservoir that loses
    to the expansion at matched H is read with that asymmetry stated. G2 added `texts=None` (row 7), so a non-adapted
    expansion arm (medians over a stream-level text set) runs beside it.
  - If G2 adopts NGSNN's feature mode as the expansion (row 7), the control becomes a STATELESS spiking LIF expansion
    of matched size. That sharpens the S0C1 reading: a reservoir win over it is attributable to recurrence and state,
    not to "spiking" or "random features". The call signature does not change.
- **Turns to ticks (section 10, row 7).** The Problem stays turn-level. Part D's harness owns the turn-to-tick mapping,
  using the reservoir encoder G1(c)'s disposition leaves. Reset-vs-carry at an episode boundary is an arm, not a
  constant. Memory span is reported as G2's `delay_turns` (D = 1) times the harness's ticks per turn.
  - S0C0 is solved by the incumbent and is the sanity cell.
  - (ii), the inseparable set, is a capacity problem like S0C1.
- **G2's interface.** `instruments/taught_problems.py::Problem` exposes `episodes(seed, split)` with split in
  teach / validation / future, `solved(reads)` and budget fields. `teaching_sequence(problem, seed, epochs)` is the
  recorded per-episode (index, probe text, label) list that the time-shuffled tuning-port control shifts.
  - Disclosed: held-out = never-taught PRESENTATIONS of the same finite items (8 episodes for (i), 4 texts for (ii)),
    not novel items.
  - Presentations read the route directly (`human=False`). MEMORY, EXPLORE and GOAL are `_ROUTE_CUES` keys that a
    human=True `Brain.respond` turns into G0(b)'s clarify turn.
- **No double credit.** G-R2 certifies G2's PROBLEMS on G2's seeds (8200-8207) and G1(c)'s seeds (7000-7007). Part D
  runs its own incumbent and reservoir-neutered controls on 11400-11479 and never cites G2's runs as its control.
- **Primary endpoint (engine-functional, the 09-30 memory).** Held-out accuracy on the posed problem by the brain-tuned
  readout neurons, read per cell as above, against path-breaking controls:
  - the reservoir neutered (the readout reads direct input / brain units only);
  - the tuning port neutered (untuned readout neurons);
  - the tuning port time-shuffled;
  - the feed neutered (grown neurons not fed), when G3's feed exists.
  - Secondary performance numbers sit beside the endpoint. Structural statistics are diagnostics only.
- **The return path (connection 6).** Readout neurons project through a declared port (FR-012), with plastic synapses,
  into the labeled grown population that posed the problem (G7). They never project into the action gate (P6).
  - Targets are resolved through G7's readers on the one extended placement record: `labels_by(records, key)` and
    `label_of(records, unit_id)`. There is no second label store.
  - At every build or rebuild of the projection, never per tick, the port calls
    `gates/unit_label_gate.py::require_labeled_targets(target_ids, records)`, which raises `UnitLabelRefused` on an
    unknown id, an unlabeled legacy unit or a failed record. G7 owns it, with constructed red R13; section 10, row 3.
  - **"The population that posed the problem" is G5's selection rule, registered explicitly.** G7's gate does not own it:
    problem_id is LINEAGE, not ownership, and Astra's reusable-batch contract lets a batch grown for P1 serve P2.
    - PRIMARY: strict, `labels_by(records, 'problem_id')[P]`, the units grown because P was posed.
    - SECONDARY, once G3's reuse record exists: lineage-inclusive, strict plus the batches G3's record shows serving P.
    - Both are reported. The primary decides. The choice is open question 6 for the owner. Red (P6's
  test): neutering the return path leaves the brain's own next answer and next attempt unchanged against the
  time-shuffled return. B's and C's readings become properties of this projection, each ablated separately.
- **Waits on:**
  - G1(c) (the substrate);
  - G2 (the problems, the solved predicate and held-out split);
  - G7 (the return target's labels and gate);
  - G3 (the feed, for the feed-neuter control only; and its default-off grown/reservoir segment port on the label
    neurons).
- **What G5 asked of G3's port (section 10, row 2).** All default-off and byte-identical when unused:
  - segment-tagged synapses, with ablation as a mask;
  - per-synapse source unit ids resolvable to G7's label record;
  - the reward d as one named argument;
  - a read-only accessor for the label neurons' per-episode answer, so the return path can project from them;
  - segment order fixed by unit id.
- **Null disposition (P4d).** A null reverts the implementation and claims nothing. Readout neurons, brain tuning and the
  return path are owner-drawn and are never deleted on a null; the owner decides on the evidence. Each named rule is an
  owner-named EXAMPLE: its implementation is reverted on its own null and the owner is told.

## 7. Cost at real scale (projections from the measured cell: 59.8 s per 800-tick cell, one process, threads 1)

| stage | cells | projected wall at 4 workers |
|---|---|---|
| dev (A grid: BATCH, DELTA, TRACE x 3 lambda; B: COUPLED, COUPLED0, COUPLED_MARG) | 8 x 8 = 64 | about 16-22 min |
| confirmatory A (10 arms: 2.3) + B (5 more arms: COUPLED, COUPLED0, COUPLED_MARG, COUPLED_SHUF, COUPLED_PERM reported) | 15 x 8 = 120 | about 30-40 min |
| the 2.7a reachability gate (runs first) | 12 configurations x 8 seeds x (TRACE + DELTA) discovery, plus at most the same again for replication; brain-free | about 2-4 min (measured: 0.6 s per 600-observation engine run in 2.6a) |
| other brain-free planted checks | 40 seeds (PA2, PA3, PB1/PB2, PB5), synthetic | seconds each (projection) |
| composition (if both land) | 2 x 8 = 16 | about 5 min |

- The shifted arms need the DELTA arm's recorded sequence, so each seed runs two stages.
- Host-load caveat: L1c read 54-114 s per cell at 4 parallel with the daemon up. Batteries and campaigns are serialized by
  the orchestrator. No pytest suite, recording or live seed was run for this draft.
- Part D's cost is unmeasured. It projects from G1(c)'s campaign once that runs.

## 8. Arc split (P0 docs, then builds) and the files each part touches

- **P0 (docs).** This file. The orchestrator owns PLAN-FINAL and tasks.md. Dated notes to propose:
  - Part C's new dependency (G4/G7/return path);
  - the shuffled teaching signal applied to the incumbent;
  - the EMA form of the trace and the step-matched control.
- **P1 (instruments, brain-free first).** The 2.7a reachability gate runs before anything in-brain. Its NOT-met outcome
  deletes Part A and leaves P1 serving Part B only.
  - NEW `src/snn_data_lake/instruments/g5_readout_cells.py`. It holds:
    - the cell runner and arms, with the common vocabulary init (2.3);
    - the recorded-sequence shift with the vacuity floor;
    - the same-rows check, including the column schedule;
    - the per-difference seed-key assert and the reserve rule;
    - the frozen `planted_chain` generator, PA1-PA3 and PB1/PB2/PB5;
    - the contrasts via `paired_contrast`, with the FPR printed as a diagnostic.
  - The home guard is ADOPTED from the shared module agreed with R1 (2.6 PA9), never re-implemented here.
  - NEW `tests/test_g5_readout_cells.py`.
  - Registration in `src/snn_data_lake/__init__.py` (REGIONS) and a `gates/dormancy_gate.py` EXEMPT row (instrument).
    Both are HOT, claimed at the build.
- **P2 (Part A, the candidate inside the instrument).** In `instruments/g5_readout_cells.py`: the subclass, its lifecycle
  DELTA_SCHED's shadow trace, and the reds PA1-PA10. The lambda = 0 bit-identity is checked against today's `OnlineReadout.learn`.
  `readout/online_readout.py` is NOT edited in this part.
- **P4a (only if 2.7a is met AND Part A lands): the merge.** `src/snn_data_lake/readout/online_readout.py` gets lambda, e,
  the lifecycle and the GRADIENT_CLAIM text, and `tests/test_online_readout.py` gets the reach and lifecycle reds.
  `plasticity/rule_composer.py`'s census row is re-worded (claimed then). The growth record carries e at beta's
  boundary (`growth/neurogenesis_engine.py`; claimed then, with R2 and G3 messaged; section 10 row 6).
  `plasticity/eligibility_algebra.py` is read only, unless the contract needs a declared factor; its owner is messaged
  first in that case.
- **P3 (Part B).** NEW `src/snn_data_lake/readout/categorical_coupling.py` and NEW `tests/test_categorical_coupling.py`, with
  their registrations.
- **Campaign.** Dev, then freeze lambda* and MU_step (a dated amendment), then confirmatory, read once. A results record
  follows. Nothing is committed between the first confirmatory recording and the results record.
- **P4 (on outcomes).**
  - A wins: P4a above, plus the strict test in `tests/test_real_percept_stream.py` (claimed then).
  - B wins: the decision hook in `growth/neurogenesis_engine.py::predict_next` (G3's directory; claimed and messaged
    then).
  - A null in either part: its deletion commit.
- **Parts C and D.** Their own amendments and arcs after their upstream results.
- **Every arc:** registrations in the commit that adds a module, the census, the Codex gpt-6.1-sol panel (the orchestrator
  sends it), the fold stated as re-panelled or battery-verified only, and ONE battery on the stacked head (the
  orchestrator).

Claimed on the board 2026-10-03:

- the evidence file;
- `instruments/g5_readout_cells.py` and its test;
- `readout/online_readout.py` and its test;
- `readout/categorical_coupling.py` and its test.

## 9. Open questions for the owner

1. **Part C waits** on G4's attachment rule, G7's labels and the return path, because today its null is entailed (section
   5). Accept the wait, or measure now with the entailed null disclosed?
2. **The shifted teaching signal also hits the incumbent** (DELTA_SHUF). If `OnlineReadout`'s gain survives it, does that
   reopen L1c's landing? Its landing never had a label null.
3. **lambda, MU_step and alpha are P1 debts** CHOSEN on dev, like MU. Accept them as P1 exception (a) debts, or demand
   derivations first?
4. **"Trace" is registered as a LEARNING-side presynaptic eligibility across percepts** (the debate's "eligibility x
   error"). The 2.6a pilot suggests it has no reachable positive on a stateless substrate. Two readings are NOT
   registered:
   - a within-percept spike-timing trace;
   - a readout that READS the filtered input. Inference and learning would both use e, a synaptic filter, which gives
     the readout fading memory and stays consistent rather than biased. It overlaps G1(c)'s cheap explicit-history
     control and the question of where state should live (P2), so it is the owner's or the panel's call, not this
     row's.

   Which reading did the owner mean? Or should the trace be tested only on the reservoir (Part D)?
5. **Readout jobs.** `OnlineReadout` reads the brain's grown units and predicts the next category. Part D's readout
   neurons are the taught learner's label neurons; they read the reservoir and answer posed problems. P4d allows "one
   tuning rule per readout job". Are these two jobs, or must one survivor serve both?
6. **The return path's target.** Should it reach only the units grown because the problem was posed (strict problem_id,
   registered as primary), or also reused batches that served it (lineage-inclusive, registered as secondary)?
7. **Who tunes the readout neurons (connection 5)?** P4d says "tuned by the brain". Part D's signal today is the
   teacher's external reward d. Should the signal route through named brain neurons (a neuromodulatory or error output,
   with its own ablation) before it tunes? Or is external feedback acceptable for connection 5? Until the owner answers,
   Part D's claim is limited as section 6 states (Codex #12).

The lead read (L4) batches Q1-Q7 for one owner brief.

## 10. Debate log

| # | partner | challenge or exchange | answer | resolution |
|---|---|---|---|---|
| 1 | R2 (a97af13910daffbad) | G5 asked whether R2's replay calls `OnlineReadout.learn` or reads the grower's beta, or edits `online_readout.py`, `rule_composer.py` or the GRADIENT_CLAIM gate | R2: no. Hydration feeds only `SpikingInteractionLearner` (`interaction_learning.py:320`, `learn_turn`/`interact`/`replay` at :554/:587/:621), with zero references to OnlineReadout/predict_next; R2 edits none of those files. R2's seeds are 15000-15999 | No collision. **Watch item:** if replay into the GROWER (R3's question to R2) is ever assigned, it would move the online readout; R2 will message G5 first |
| 2 | G3 (aab6003013a8b76f6) | G3 asked: (a) does G5 collide with its default-off grown-segment port on `interaction_learning.py`; (b) is the taught learner's label neurons as the readout in G3's feed arms a problem for G5; (c) what interface G5 needs | (a) No: G5 never edits `interaction_learning.py`. R2 holds that file's claim, so G3 needs R2's agreement and G0(c)'s schema owner. (b) **CONCEDED, with a change:** the label neurons are the incumbent spiking, brain-tuned readout of the posed-problem job. Part D now ADOPTS them, and its rules run as arms against R-STDP. G5 asked G3 to reword "G5's readout neurons replace it" accordingly. (c) The five port asks in section 6 | Part D amended (section 6; open question 5 reworded). **Closed:** G3 accepted the reword ("G5's tuning-rule arms run on the taught learner's label neurons; a rule replaces R-STDP only by winning G-R5") and port asks (1) segment tags with a mask, (2) unit ids that are G7's, (5) unit-id order. Asks (3) the reward argument and (4) the answer accessor touch `learn_turn`/`interact`/`_argmax` on R2's file, so they become a G5 default-off step on R2's single upcaster chain at Part D's build, sequenced by main. G5 accepted |
| 3 | G7 (a80b14a934d639732) | G5 asked G7 to refute Part C's wait and to supply an id-addressable label record plus return-path target coverage | G7 AGREED that Part C waits and cites this row's measurement (placement independent of weights; P(L5e/L6e) 0.1706). Record: G7's extension of `growth/placement.py`, with fields unit_id, batch_id, position (ap, dv, x_um, y_um, depth), layer, sign, region, problem_id and origin. Readers: `labels_by`, `label_of`. Target check: `gates/unit_label_gate.py::require_labeled_targets`, called at projection build and not per tick (0.44 us/record measured; a per-tick call is the observer-becomes-the-load shape). No file overlap | Closed. Section 6 amended; G5 accepts the build-time call. **Round 2:** G5 asked G7's R13 to also refuse a target with a different problem_id. G7 REFUSED, for a policy reason: problem_id is lineage, not ownership, and a gate enforcing ownership would break G3's reusable-batch arm. G5 CONCEDED: the problem filter is G5's own selection rule, registered in section 6 as strict primary plus lineage-inclusive secondary (open question 6). G7 logged it as its D5, resolved |
| 4 | G4 (a088e6b57aed001c4) | G5 asked G4 to refute Part C's wait and asked: does the attachment rule act on NGSNN or the reservoir; is there a position record; is there file or seed overlap | G4 CONFIRMED the dependency, with a sharpening: layer carries function only if wiring differs by layer AND reaches the readout. Precondition = G-R4's layer x sign reading beating its matched twin; else Part C is NOT READ, not red. Brain side only; the reservoir and feed are layer-blind; positions come from G7's record; `growth/wiring_prior.py` stores no per-unit field. No readout/ files; G4's seeds are 10000-10999 | Sharpening ACCEPTED (section 5). Closed |
| 5 | G2 (ab92cc5f5070123ac) | G5 asked for G2's problem interface, claimed (i) is unsolvable by any readout on a stateless substrate, and asked about double credit | **PARTLY REFUTED:** only the S1 cells need state; S0C1 is solvable by a nonlinear readout of the current turn. G2's problems never reach NGSNN/`OnlineReadout`, so Part A does not touch them. Interface: `taught_problems.Problem`, `teaching_sequence`. G-R2 certifies problems, never a readout: no double credit if Part D runs its own controls on its own seeds. G2 asked G5 to refute (a) its frozen random expansion control (instruments/) and (b) human=False presentations | G5 CONCEDED (2) and amended Part D per cell (state vs capacity). G5 REUSES G2's expansion as Part D's matched-capacity control on S0C1, and asked G2 for a callable taking (H, seed). (a) no collision; (b) no objection, Part D reads the route directly. **Closed:** G2 ACCEPTED one shared expansion, `taught_campaign.presized_expansion(problem, H, seed).features(text)`, which returns the stimulus vector; the reader rank-codes it. Its disclosures are carried into section 6: a per-problem threshold; a positive control only at G2's certified H (projection 64), so an H-matched failure is read only beside a certified-H solve; no use on S1 cells |
| 7 | G2 -> G5 (ab92cc5f5070123ac), G2's draft `g2-taught-problems.md` | G2 asked G5 to refute: (a) must G2's episodes carry a ticks-per-turn field for Part D's ticked reservoir; (b) any red of G2's that cannot fire, or a control that is the method itself. G2 also moved (i) to GREET / GOAL; GOAL stays covered by the human=False rule | (a) NO tick field: the turn-to-tick mapping and reset-vs-carry at episode boundaries are Part D's harness's (and G9's) arms. G5 asked instead for a `delay_turns` field (D = 1) and the episode tuple as the boundary unit. (b1) CHALLENGE: B6's "explicit history must not solve S1C1-with-history" rests on a capacity bound that G2's own 2.3 calls a float MEASUREMENT, yet G2's table reads a B6 solve as "the control fails". Proposed: route it like B4's float exceedance, to the construction rows. (b2) B5's lr-0 red is reachable (init w = 0.3 +- 0.05; 4 of 6 probe ranks are shared fillers); no control is the method. (b3) Disclosed: the expansion's per-problem median threshold is stimulus-set-adapted; G5 carries that asymmetry; optional `texts=` argument | **Closed (G2's log rows 14-16).** (a) G2 CONCEDED: no ticks in the Problem; `delay_turns` added (D = 1 for (i), 0 for (ii)); the episode tuple is the boundary unit. (b1) G2 CONCEDED: a capacity-cell solve, by the incumbent or by explicit history on S1C1, reads "construction wrong" with B2(c)'s witness margin. Clause 4 keeps only "history fails S1C0". A solve of an exact-bound STATE cell reads HALT. (b3) ACCEPTED: `presized_expansion(problem, H, seed, texts=None)`, with the default byte-identical. G2's P2 change: NGSNN's feature mode (`growth/ng_snn.py`) is tried as the expansion on G2's dev seeds; if it meets B7 it is adopted and `presized_expansion` becomes an adapter with the same signature |
| 6 | R2, round 2 (a97af13910daffbad) | R2 (now assigned the grower's percept-transition record) asked: is `OnlineReadout.learn` a pure function with no RNG; does the trace carry state; will it go into `growth_state_bytes`, or should R2's mid-wake grower kill (K2g) stay report-only | G5: the incumbent has no RNG, and beta's update is a pure function of (beta, m, S, cat, mu, sign); the counters never feed back. A mid-wake redo is exact iff the blob carries the online beta at the watermark. The trace carries e plus its model key; if Part A lands, e goes into `growth_state_bytes` at beta's boundary with a restart-equality red, claimed then with R2 and G3 messaged. Until then K2g report-only is correct for any trace arm | Agreed. Closed |
| 8 | R1 (lead's instruction on Codex #14; R1's I7) | G5 proposed ONE shared live-home guard: a fresh temp dir; a resolved-path check against the resolved live home (equality or containment refuses); NORD_HOME set before any lake-binding import; the live home as a parameter so the alias red runs on a decoy; home next to `throwaway_home()` in substrate/home.py, or R1's I7 module if it exists | R1 AGREED: one guard inside `substrate/home.py`, which R1 builds. (1) `home.assert_not_live_home(path, live_home=None) -> Path`: it resolves `path` strictly and `live_home` non-strictly (the live home may not exist on CI); it refuses equality or containment in either direction (`commonpath` after `normcase`) and an empty or None path; the default live home is `~/.snn_data_lake`; it returns the resolved path. (2) `throwaway_home(live_home=None)` calls it and puts the RESOLVED path in NORD_HOME, so every caller is guarded (one mechanism). (3) `home.require_isolated_home(live_home=None)` refuses an unset or empty NORD_HOME and runs (1); harnesses call it before any lake-binding import. Reds on a decoy: a prefixed junction, a subdir, the parent, an unset NORD_HOME; positive: a fresh throwaway. It lands in R1's P1 commit, after R2's new kinds (home.py order) | CLOSED. G5's harness calls `require_isolated_home()` before any lake-binding import and checks the live home's mtime after each stage. G5 writes no guard of its own (P2). R1 named the refusal type: `home.LiveHomeRefused(ValueError)`, whose message names the resolved path and the rule that fired (equal, contains, contained or unset). G5's harness catches it and prints it as a refusal. It is recorded in R1's draft as I7 and debate row D7. G5's in-brain cells therefore also wait on R1's P1 landing (or run on a branch carrying it) |

## Review record

- **Lead read.** 2026-10-03, board `p0_reports\G5.md`, "=== LEAD READ ===" items L1-L6, and board DEPENDENCIES.md
  rulings 9-11. Folded:
  - L1: D1, D2, D3 and D6;
  - L2: D4, replication on 11936-11943;
  - L3: task #19 re-wired to #29 under ruling 11; Parts A and B wait only on campaign DONE and their own dev freeze;
  - L4: Q1-Q7 batched for the owner;
  - L5: the pilot carry, verbatim in 2.6a;
  - L6: 11936-11943 registered in section 4.
- **Advisor read.** The CLI lead session, Fable, 2026-10-03.
- **Codex critique.** Model gpt-6.1-sol. Reply file
  `C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\161935bb-36f0-4e8d-8a38-0411dcc64d0f\scratchpad\codex_p0\b2.reply.txt`,
  sha256 `e40e3fb21341ddd031ee018d8127745c9f6f3f34265118f787ffcbd1aac7edf9` (re-hashed by G5 at the fold: matches).
  G5's findings are #8-#15: 1 BLOCKER, 6 MAJOR, 1 MINOR. Dispositions:
  - **#8 BLOCKER, FIXED.** A common vocabulary init: every arm widens to M_seed at every model creation, so the shifted
    labels move only the alignment. The same-rows check now also matches the column schedule and `widened == 0`. New red
    PA10: an early-category mutation must halt. Verified in `online_readout.learn`, which widens when
    cat >= beta.shape[1].
  - **#9 MAJOR, FIXED.** `permutation_fpr` is printed as the bar's exact FPR diagnostic and gates nothing. The
    "p <= 0.05" conjunct is dropped; it would refuse the module's own 13/256 MET example. The five-non-tied rule is
    kept as a named, separate restriction with its purpose stated.
  - **#10 MAJOR, FIXED.** A new arm, DELTA_SCHED: the delta direction at the trace's eventwise scalar step scale
    c_e * ||e||, with an in-arm matching assert, the error printed and a 1e-5 halt. New bar A-R1c, contrast(TRACE - DELTA_SCHED) MET. A schedule-only gain is
    reported in those words and does not land.
  - **#11 MAJOR, FIXED.** One frozen `planted_chain` generator, with kind in {burst, drift, iid}. PA2 uses kind = iid
    (successive pairs independent). The pilot's numbers are never re-read as PA results.
  - **#12 MAJOR, FIXED as a limit on the claim plus a freeze precondition; it stays owner question 7.** While the
    signal is the teacher's d, Part D is never credited with connection 5. Before Part D's amendment freezes, it must
    name the brain producer, the transformation and the receiving port, and register the producer's ablation, unless the
    owner accepts external feedback.
  - **#13 MAJOR, FIXED.** The reserve replaces a whole seed block across every arm and both stages, with a
    per-difference seed-key assert.
  - **#14 MAJOR, FIXED; R1's `home.require_isolated_home` / `assert_not_live_home` is agreed (debate row 8, CLOSED).** A resolved-path guard against
    the resolved live home, set before lake-binding imports, with an alias red on a decoy. It is common infrastructure
    shared with R1's I7 and survives a Part A deletion.
  - **#15 MINOR, FIXED.** 0.46 is labelled an independence reference. The general union bound 12 x 0.05 is printed. The
    0.03 replicated bound states its independence and per-stage-error preconditions. The claim that correlation lowers
    the rate is withdrawn.
- **The fold itself was not re-panelled.** It is design text only; nothing was run during the G1(c) campaign. Its only
  tool evidence:
  - the read-only Grep for 11936-11951 (section 4);
  - reads of `online_readout.py` and `paired_contrast.py`, which confirmed #8's widen and #9's FPR definition;
  - `sha256sum` of the reply file (one small file);
  - `wc` and `grep` over this draft.

## Results: 2.7a, the trace's reachability gate (2026-10-04; appended below the frozen text, which is unchanged)

**Verdict: PA1 NOT MET.** 1 of 12 configurations was met on discovery, and 0 replicated. **Part A is DELETED before any
in-brain cell.** This is 2.7a's deletion under P8 F11 ("deleted, not weakened"). Section 2.7 states that it is not a
refusal. Section 2.8 applies:
- `OnlineReadout("delta")` stays the incumbent and the record stands.
- The owner-drawn parts are untouched: the readout neurons, the brain's tuning connection and the return path.
- Part D carries a trace only by its own amendment for the stateful substrate.

The replication rule (lead ruling L2) did the job it was adopted for. The one discovery pass read t +4.378 (7 of 8
seeds positive, 1 tie) and then t +0.153 on the fresh replication seeds. On that same discovery cell, DELTA_SCHED also
cleared the bar (t +3.780, printed, not a bar). Even the discovery gain was therefore at least partly the trace's step
schedule rather than its history.

**Run 2 (cited), the 12 discovery contrasts, the replication and the verdict, verbatim:**

    TRACE - DELTA  [drift noise 2.0 lambda 0.368] disc mean -0.0011  t -0.236  -+-+---+  FPR(diag) 0.0625  skew +0.21  NOT MET
    TRACE - DELTA  [drift noise 2.0 lambda 0.717] disc mean -0.0127  t -1.651  +--+----  FPR(diag) 0.0469  skew -0.10  NOT MET
    TRACE - DELTA  [drift noise 3.0 lambda 0.368] disc mean +0.0009  t +0.220  +-++---+  FPR(diag) 0.0352  skew +0.61  NOT MET
    TRACE - DELTA  [drift noise 3.0 lambda 0.717] disc mean -0.0114  t -2.244  ---+----  FPR(diag) 0.0508  skew +1.24  NOT MET
    TRACE - DELTA  [drift noise 4.0 lambda 0.368] disc mean +0.0058  t +1.208  -+--+-++  FPR(diag) 0.0430  skew +0.67  NOT MET
    TRACE - DELTA  [drift noise 4.0 lambda 0.717] disc mean -0.0025  t -0.351  -+--+---  FPR(diag) 0.0430  skew +1.49  NOT MET
    TRACE - DELTA  [burst noise 2.0 lambda 0.368] disc mean +0.0176  t +4.378  ++++++0+  FPR(diag) 0.0625  skew +0.41  MET
    TRACE - DELTA  [burst noise 2.0 lambda 0.717] disc mean +0.0027  t +0.523  ++-+-+-+  FPR(diag) 0.0586  skew -0.86  NOT MET
    TRACE - DELTA  [burst noise 3.0 lambda 0.368] disc mean +0.0058  t +1.868  ++++--++  FPR(diag) 0.0430  skew +0.05  NOT MET
    TRACE - DELTA  [burst noise 3.0 lambda 0.717] disc mean -0.0071  t -1.042  -+-+---+  FPR(diag) 0.0469  skew +0.82  NOT MET
    TRACE - DELTA  [burst noise 4.0 lambda 0.368] disc mean +0.0056  t +1.011  +-+++0-+  FPR(diag) 0.0391  skew -0.38  NOT MET
    TRACE - DELTA  [burst noise 4.0 lambda 0.717] disc mean -0.0020  t -0.204  +-+++---  FPR(diag) 0.0547  skew -0.15  NOT MET
    TRACE - DELTA  [burst noise 2.0 lambda 0.368] REPL mean +0.0009  t +0.153  +-+-+-+-  FPR(diag) 0.0469  skew -0.08  NOT MET
    TRACE - DELTA_SCHED  [burst noise 2.0 lambda 0.368] REPL mean -0.0018  t -0.661  0-+-+-+-  FPR(diag) 0.0469  skew +0.32  NOT MET  (printed, not a bar)
    false-pass references (record 2.7a; references, not guarantees): independence reference 1-(1-0.05)^12 = 0.46; union bound 12x0.05 = 0.60; replicated <= 12x0.05^2 = 0.030 (needs independent discovery/replication data)
    PA1 NOT MET: 1/12 met on discovery, 0 replicated -- Part A is DELETED before any in-brain cell (record 2.7a, P8 F11)

The 12 discovery TRACE - DELTA_SCHED lines are in the run log; they are printed and not a bar.

**Reproduction.**
- Run 1 at 07:43:54Z and run 2 at 07:49:33Z gave byte-identical discovery lines (all 24, `diff`).
- Between the runs one change was made: the replication branch now also prints its DELTA_SCHED contrast, which 2.7a
  registered as "printed beside". It changes no computed TRACE or DELTA number.
- The one discovery MET and its failed replication were also reproduced alone, in a third process.

**Instrument sensitivity: two measured positives, so a null here is not a deaf instrument.**
1. The reading cleared the bar on a +0.018 paired difference (the discovery pass above).
2. **DIAGNOSTIC, unregistered.** contrast(DELTA - BATCH) on the burst / noise-2.0 chains read mean +0.3259, t +15.203,
   ++++++++, on discovery seeds 11900-11907 (already exposed). The harness therefore resolves a known readout effect on
   these chains. It is disclosed as an extra look, run after the verdict, and it alters nothing.

**Run conditions.**
- Commit 500d3f58 plus the uncommitted instrument `src/snn_data_lake/instruments/g5_readout_cells.py`, sha256
  c2e2e762...cd0128 at run 2.
  - The COMMITTED file is the same code after `ruff format`, sha256 0658b781...8573aa. Run 3 (formatted) printed
    every contrast, reference and verdict line byte-identical to run 2 (`diff`).
- `D:\AI-project\spiking-neural-data-lake\.venv-lake` (python 3.14t), with PYTHONPATH set to this worktree's src.
- `NeurogenesisSelfImprove(refit_every=1, device="cpu")` under `ClosingGate(3)`, ticking every 20 observations.
- `planted_chain` with n 600, k 4, run 20, d 64. torch threads 1, one process, wall 2m41s (run 1) and 3m37s (run 2).
- Brain-free: no Brain and no lake. The live lake mtime was 1790231317 before and after every run.
- Logs, outside the repo, in the G5 scratchpad: `g5_gate_2.7a.log`, `g5_gate_2.7a_run2.log`,
  `g5_gate_2.7a_repro.log` and `g5_gate_2.7a_run3_formatted.log`.

**What the deletion takes and what stays (Part B still needs the shared machinery).**
- GOES: `TraceDelta`, `DeltaSched`, `GATE_FAMILY`, `LAMBDAS` and `reachability_gate`; the trace-specific checks PA1,
  PA2, PA5, PA6 and PA7; and the arms TRACE, TRACE0, TRACE_SHUF, TRACE_FLIP, DELTA_STEP and DELTA_SCHED, together with
  the lambda / MU_step dev grid.
- STAYS for Part B:
  - `planted_chain`, `reading` and `run_planted`;
  - PA3, the shift's positive control, which B-R3 uses;
  - PA4 (vacuity), PA8 (same rows), PA9 (the home guard) and PA10 (the common vocabulary);
  - A-R0, contrast(DELTA - BATCH), which B-R0 reads.
- DELTA_SHUF and DELTA_FLIP were incumbent findings, not G5 bars. They lapse with Part A, and the label-null question
  about `OnlineReadout` (open question 2) stays open for the lead.
- **For the owner (question 4, now sharpened by measurement):** the learning-side presynaptic trace has no
  replicable positive on the stateless substrate, even on a planted chain. A trace stays plausible only where the
  substrate carries state (Part D, by its own amendment). This stays batched.

## Results: Part B's brain-free planted checks PB1 / PB2 / PB5 (2026-10-04)

**All three are MET.** These checks run on a SYNTHETIC readout (`g5_readout_cells.planted_readout`): class
probabilities from a planted error process, with no engine. They test the coupling and its control, not the brain.

**The fixture**, fixed in code BEFORE the registered seeds ran (once) and never tuned after: n = 400 events, k = 4
classes, logit strength 2.0 on the readout's named class over N(0, 1) logit noise, and a hit rate of 0.85.
- PB1 (relabel): truths 0-2 are named (t + 1) % 3, class 3 is answered right, and the truths are uniform, so both
  marginals are equal.
- PB2 (iid_errors): the readout is right at 0.85, else names a uniform OTHER class.
- PB5 (prior): the truths are drawn from (0.55, 0.15, 0.15, 0.15), with a +1.0 logit bias on class 3.

**Output, verbatim** (`--planted`; seeds 11924-11931 for PB1/PB2 and 11944-11951 for PB5):

    relabel: full - uncoupled                    mean +0.4584  t +47.407  ++++++++  FPR(diag) 0.0352  skew +0.96  MET
    relabel: full - marg                         mean +0.4509  t +40.015  ++++++++  FPR(diag) 0.0352  skew +0.73  MET
    iid_errors: full - uncoupled                 mean -0.0100  t -3.480  ----0---  FPR(diag) 0.0625  skew -0.21  NOT MET
    prior: marg - uncoupled                      mean +0.0897  t +13.055  ++++++++  FPR(diag) 0.0469  skew +0.38  MET
    prior: full - marg                           mean -0.0338  t -2.555  -+-----+  FPR(diag) 0.0625  skew -0.65  NOT MET
    PB1 MET; PB2 MET; PB5 MET

**What this establishes.**
- The coupling can recover a which-confuses-which structure (PB1).
- The marginal control is alive (PB5): it corrects a pure base-rate miscalibration, so B-R3's
  contrast(COUPLED - COUPLED_MARG) cannot pass because the control is dead.
- The two readings differ in the direction 3.3 assumes: on equal marginals only the off-diagonal moves evidence.

**Disclosed, unprompted (P8 clause 4).** On NON-systematic errors (PB2) the full coupling did not merely fail to
help: it read measurably WORSE than the uncoupled argmax (mean -0.0100, t -3.480, 7 of 8 negative). PB2 asks only
that it "must NOT clear", so this is not a red. It does mean the coupling has a real cost where the confusions are
noise: its counts re-route some correct decisions. That cost is part of what the in-brain B-R1 contrast must
overcome on the wake stream.

**Run conditions.**
- Commit d446d9f0 plus the uncommitted `categorical_coupling.py` (sha256 78a18eb8...270526) and instrument (sha256
  6be99321...51de49e5).
- Re-run on the committed instrument (sha256 e2568787...27760, which adds only self-check lines): byte-identical
  output.
- Brain-free. Logs in the G5 scratchpad: `g5_planted_PB.log` and `g5_planted_PB_run2.log`.

## Results: the cell machinery's brain-free reds PA3 / PA4 / PA8 / PA10 (2026-10-04)

**All fire as registered.** The in-brain runner is built (`brain_cell`, `run_cells`, `instrument_engine`) and has
NOT been run on any Brain. Dev cells wait for the lead's go.

**PA3, the shift's positive control** (`--pa3`, seeds 11916-11923, oracle input one-hot(c_{t-1}) on the frozen burst
chain at noise 2.0, delta under the common vocabulary init), verbatim:

    PA3 ORACLE - BATCH                           mean +0.0810  t +3.234  ++++++++  FPR(diag) 0.0391  skew +1.34  MET
    PA3 ORACLE - ORACLE_SHUF                     mean +0.5339  t +15.140  ++++++++  FPR(diag) 0.0391  skew +0.52  MET
    PA3 ORACLE_SHUF - BATCH                      mean -0.4529  t -15.051  --------  FPR(diag) 0.0469  skew -0.25  NOT MET
    PA3 MET

The circular shift therefore removes a real teaching gain, and does so decisively (+0.534). A-R3 / B-R3's
X - X_SHUF contrast can fire.

**PA4, PA8 and PA10** run in the bare self-check:
- PA4: a one-category sequence reads f = 0, i.e. VACUOUS.
- PA8: a one-label perturbation halts the same-rows check.
- PA10 (Codex #8): the late-category fixture taught its circular shift WITHOUT the common init widened beta (2 vs 3
  widens), changed the column schedule hash and HALTED the same-rows check. The same construction WITH the common
  init passes.

**Wiring check (brain-free, on fixture seed 20261003 only, not a registered reading).**
- COUPLED0 equals DELTA bit for bit (live accuracy and beta hash): B-R2's reach.
- Every coupled arm leaves beta and the refit trajectory identical to DELTA's: B-R4's construction.
- The coupling re-reads only the decision. Pinned in tests/test_g5_readout_cells.py.

**Run conditions.** Commit d34993a0 plus the instrument at sha256 8e773656...ab620cf. Brain-free, 23 s. Log in the G5
scratchpad: `g5_pa3.log`.

## Amendment 2026-10-06 (pre-data: no in-brain cell has run; challenger G6's plan-level refutation, three points)

Dated pre-data amendment to 3.4, applied before any in-brain Part B cell. It adds a reading and an ordering and removes
one word; no threshold, arm, seed or bar changes.

1. **B-R1 reports its sign.** B-R1 stays "contrast(COUPLED - DELTA) MET". The printed line carries the contrast's mean
   sign. A contrast that is significantly NEGATIVE (the paired t at the bar, wrong direction) reads HARMFUL, not null:
   B-R1 is NOT met either way, `categorical_coupling` is deleted (3.6), and the evidence file records HARMFUL beside
   the number. Reason: PB2 measured the coupling worse than the uncoupled argmax on non-systematic errors (mean
   -0.0100, t -3.480, 7 of 8 negative), so a negative read is a reachable outcome and must not collapse into "null".
2. **B-R3 conjunct 4: the word "secondary" is struck; all four conjuncts bind (line 467).** The frozen text above stays byte-intact; this amendment overrides its label. The conjunct reads: "contrast(COUPLED_SHUF - DELTA) NOT met.
   This conjunct can only add a refusal." All four conjuncts must hold (hard rule 5: no relabelling a failed conjunct).
3. **Order of evaluation.** B-R2 (COUPLED0 == DELTA on 8 of 8) and B-R4 are evaluated and printed BEFORE B-R1 is read.
   A failed B-R2 or B-R4 HALTS the read: no B-R1 verdict is printed, so a zero-count arm cannot read MET by noise.

## Provenance correction 2026-10-07 (Codex cp4 #16, MINOR; append-only; no frozen text, threshold or result changes)

Frozen 2.2's P2 clause required that the build either declare the trace update through `eligibility_algebra` or state
why it cannot, and record a `distinct-from` note against `r_stdp`. The measurement commit 4ecc90d9 carried neither. Both
are recorded here, after the fact, for the citable record:

- **eligibility_algebra not used, and why.** `plasticity/eligibility_algebra.py` evaluates SCALAR terms only: its header
  says "evaluate is SCALAR-only: float(tensor) raises", `AuthoredRule.evaluate` converts every binding with `float()`,
  and `author` accepts (coefficient, factors) pairs of floats. The trace update is a vector outer product
  `dbeta = c * e (x) err` over a feature vector e and an error vector p, which the contract cannot express. This is an
  operand-shape incompatibility, not a preference. Section 2.2 sent that case to the panel; no algebra extension was built, because
  the 2.7a gate deleted the module (d446d9f0) before any merge.
- **distinct-from `r_stdp`.** The candidate's e is an EMA of the readout's input RATES (presynaptic, readout-local).
  `r_stdp`'s trace is a spike-timing coincidence trace. They share a word, not a mechanism, and neither reuses the other.

This correction documents a gap in what 4ecc90d9 disclosed. It does not change its measurement (PA1 NOT MET) or the deletion.
