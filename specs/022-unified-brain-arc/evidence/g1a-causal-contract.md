# G1(a) / T109 part (a) -- the causal information-and-feedback contract, the recorded live stream, and the arms that need no pool

Row: plan Part D "G workstream", G1 / T109 ("FIRST freeze a causal information-and-feedback contract"); red G-R1
(Part F); FR-041, SC-019. Part (b), the pool-selection benchmark, and part (c), the confirmatory diagnostic, are later
arcs with their own pre-registrations. Status: RESULTS RECORDED (2026-09-24). Two Codex design reads (23 and
10 findings) were folded before the freeze (section 2.10). Section 2 is frozen at the commit that landed this file
(`51b2b98`); section 3 records the runs and every deviation from section 2.

## 1. Territory, read at HEAD `71e2e4c` before the design

The probes named here, their outputs and the recorded stream are archived in the memory store under
`archive/g1a-2026-09-24/` (`probe_stream.py`, `probe_closed_loop.py`, `probe_replay_cost.py`, `probe_prepost.py`).

1. **The grower's live stream.** When the arm step yields a category, `Brain._wake_tick` hands
   `cortex/neurogenesis_engine.py::NeurogenesisSelfImprove.observe` the attention-gained percept (`res["percept"]`,
   64 floats) and the brain's own category of it (the `GrowingLearner` prototype id, `res["unit"]`); the reflex path
   feeds nothing. `observe` appends the pair (percept_{t-1} -> category_t) to the buffer, scores the retained model's
   prediction of this category from the previous percept (the live instrument `live_n`, `live_correct`,
   `live_persist`), and then lets the online readout learn from the scored pair. The model itself changes only at
   refits: `NeurogenesisSelfImprove.tick` refits every `refit_every` = 20 sleep ticks, and a refit may skip when data
   are insufficient.
2. **Measured on seed 0, 800 ticks** (`probe_stream.py`; `throwaway_home()`, `persist=False`; 44 s, 55 ms per tick):
   393 wake percepts and 407 sleep ticks; width 64; 6 categories (counts 1 / 270 / 94 / 11 / 6 / 11, majority 0.687);
   persistence floor 0.929 with 28 category changes (runs: mean 13.6, median 3, max 146); H(next | previous, a change
   happened) = 0.86 bits. The incumbent's live instrument reads 0.759 against a live persistence of 0.932. No percept
   vector repeats exactly at any lag; category agreement is 0.53 at lag 256 and 0.38 at lag 128. **Seed 0 is
   therefore exposed**: this probe, the open-loop probe (item 4) and a cost probe (item 10, which printed a
   sensory-history input's live accuracy, 0.867 against 0.759 with the growth gate off) all read it.
3. **Acquisition order.** `Brain._wake_tick` builds its context from `Brain._safe_ctx()`, which uses the latents cached
   by the PREVIOUS wake tick's `_perceive_real` and `_perceive_audio`; it steps the arm, delivers
   `neuro.observe(x_t, c_t)`, and only then calls `_perceive_real()` and `_perceive_audio()`, acquiring the frames
   that x_{t+1} will carry. `_perceive_real` feeds the |LP| facade (`LearningProgress.consume_surprise`) that frame's
   surprise. **A sleep tick after that reads an |LP| carrying the next frame's surprise.** `_perceive_real` indexes
   `_real_frames` by `_real_ptr % len(_real_frames)`, advances the pointer before transduction, and can return None.
   `_wake_tick` stores each acquisition's result unconditionally, so a failed acquisition leaves a None latent, and
   `_safe_ctx` then substitutes a fallback for that modality while still setting `_perception_unified` when the other
   modality succeeded. `_perceiving_real` and `_perceiving_audio` only say that the packs have rows. The first
   delivery precedes every acquisition, so x_1 carries no frame; under uninterrupted acquisition x_2 carries frame 0
   and x_257 frame 255 (both packs hold 256 frames).
4. **The loop is open for the grower's feed, in the one intervention measured.** `Brain` reads the grower's
   `predict_accuracy` and `grown_units` (`_observe_self_wake`, `_observe_self_sleep`, `_update_access`). Feeding the
   grower the oracle's input (a 64-wide one-hot of the category) for 800 ticks on seed 0 (`probe_closed_loop.py`)
   left the stream it was handed identical: 393 of 393 percepts and categories equal. The comparator caught planted
   differences at exactly the planted indices (17 and 40). The claim is limited to that intervention; section 2.6's
   R-a1b re-checks it on every pilot seed with the complete event sequence.
5. **Randomness and execution.** The grower's candidate RNG is instance-owned (`torch.Generator(seed)` in every NGSNN
   fit; placement seeded from `self.seed`). That does not fix execution: `NeurogenesisSelfImprove._refit` routes each
   refit through `substrate/device.py::select`, which honours `NORD_DEVICE` / `NORD_GPU`, then a crossover measured
   once per process (the module records 262,144 elements on this machine), unless the engine was built with
   `device="cpu"` (the `pin`). A 1,000-row buffer is 64,000 elements at width 64 but 256,000 at width 256, next to
   that crossover. `hippocampus/ng_snn.py::NGSNN.decide` writes `last_abstentions_` and `last_unanswerable_` on
   every call.
6. **The growth gate.** `Brain._growth_gate_verdict` feeds the engine's own trajectory into
   `cortex/growth_stop.py::GrowthStop` and returns `should_grow("cortex")`. `GrowthStop.observe_refit` reads |LP|
   through `_read_lp`, which returns None for an unfed channel AND for any exception its reader raises.
   `LearningProgress.lp` is a pure getter.
7. **The schedule is exogenous.** `Brain.tick` steps the circadian clock with `light = self._schedule(self._step)`, a
   function of the step count only, and sleeps while `consolidation_gate()` is open. Sleep positions carry no stimulus
   content.
8. **The incumbent arm harness and the assets.** `tests/test_real_percept_stream.py::_l_cell` runs one arm per brain
   with the grower's `observe` replaced (LIVE, ORACLE_PREVCAT, NEUTER_PERCEPT, SYNTH, CAP-N, the L1C arms) and serves
   the pinned L-row tests. No in-tree mechanism records the stream and replays arms offline (`graph-exists "percept
   stream replay arms harness"` lists only `percept_surprise.py`'s replay-twice A/B, a charge attribution). The live
   packs are required (`_require_the_live_packs` skips without them): sha256 prefixes `9863a50ead6190ae`
   (`models/nmnist_pack.npz`, 754,456 bytes), `295eef1755e2993a` (`models/shd_pack.npz`), `9b471619e39acfa0`
   (`models/eprop_nmnist.pt`), `6b32f9eb38beeca3` (`models/eprop_shd.pt`).
9. **Prior results, bounded.** The historical oracle (`evidence/l-parallel-arms.md`, ORACLE_PREVCAT: a 64-wide
   one-hot, online readout removed, refit-held-out scoring, 800 ticks, seeds 0-7) beat LIVE on 8 of 8 under L1a
   (t 2.97) and sat at its own persistence floor on that stream; on a synthetic structured chain it exceeded
   persistence (its section 3.4(c)). Every live arm then sat below its live persistence floor
   (`evidence/l1c-online-readout.md`). These are historical observations under other configurations; nothing here
   assumes they transfer. The refuted "LSM as fluid intelligence" entry stands, so G1 claims temporal memory only.
10. **Cost** (`probe_replay_cost.py`, seed 0, 800 events, gate off): a replay of the recorded stream through a fresh
    grower takes 0.3 s on CPU at width 64 and 0.4 s at width 256. Recording costs 55 ms per tick.
11. **The refactor reference** (`probe_prepost.py`): the deployed brain at `71e2e4c`, seeds 900 and 901, 800 ticks,
    dumped the grower's live counters, trajectory, baselines, gate decisions and refusals, and CRCs of its model
    tensors (seed 900: live 374 / 354 / 367, 20 refits, 19 gate decisions). A rerun of seed 900 was byte-identical.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.1 The causal contract

- **Index.** The grower's wake percepts are t = 1..n in delivery order, with the sleep ticks between them. Step t
  delivers x_t (the percept) and c_t (the brain's own category of x_t). The acquisition record of x_t (section 2.3)
  goes to the scorer only.
- **Prediction, then update.** Before step t delivers anything, the arm's prediction y^_t of c_t is computed and
  cached. Then the harness hands the arm (x_t, c_t), and only then may it update: the learner buffers
  (phi_{t-1} -> c_t) and its online readout learns, exactly as live. One computation per event: `predict_next()`
  computes the pending event's rate row and decision and caches them against the model and the last percept;
  `observe` reuses that cached pair, so scoring and learning share one computation and `decide` runs once per event.
- **Channels.** The learner, for every arm, receives the supervision targets (the c_s) and its growth gate's |LP|.
  Each feature map receives ONLY the channels section 2.2 declares for it, through a view that raises on any other
  channel. Categories are never a channel of a sensory or pool feature map; (d) receives its own cached predictions
  and nothing else of the learner; no arm reads the acquisition record.
- **CAUSAL |LP| (every reading of an arm).** At a sleep tick between t and t+1, the gate reads the |LP| value recorded
  at the delivery of x_t, before that wake tick's acquisition. The live |LP| at the sleep tick carries the next
  frame's surprise (section 1.3) and is used only in LIVE mode, by the equivalence checks R-a1 and R-a1b, which
  reproduce the deployed engine and license nothing about prediction. LIVE and CAUSAL are different experiments and
  are never mixed within a reading.
- **Widths.** Every contrast compares arms of equal width (section 2.2's table); the deployed incumbent (a) stays
  unpadded at 64 and is compared only with width-64 arms. Zero padding changes the learner (NGSNN draws its random
  candidates over the input width), so a padded control is part of every contrast that pads.

### 2.2 The arms built here (the pool arms (b) and (c) arrive with G1(b))

| arm | phi_t | width | channels | role |
|---|---|---|---|---|
| (a) | x_t | 64 | x | the deployed incumbent; never padded |
| (f64) | onehot_32(c_t) (+) 0_32 | 64 | c | the persistence oracle: the positive control against (a); handed the category (disclosed) |
| (d) | x_t (+) onehot_32(y^_t) | 96 | x, own predictions | readout self-feedback, version 2: the arm's cached prediction of c_t as a detached input (a zero block when y^_t is None) |
| (d0) | x_t (+) 0_32 | 96 | x | (d)'s disconnected twin |
| (e1) | x_t (+) x_{t-1} (+) x_{t-2} (+) x_{t-3} | 256 | x | sensory history (zeros before the stream starts) |
| (e1-0) | x_t (+) 0_192 | 256 | x | (e1)'s null-history control |
| (e2) | x_t (+) onehot_32(c_t) (+) ... (+) onehot_32(c_{t-3}) | 192 | x, c | category history; holds the category advantage (disclosed) |
| (e2-0) | x_t (+) onehot_32(c_t) (+) 0_96 | 192 | x, c | (e2)'s current-category-only control |

- **The learner** for every arm is `NeurogenesisSelfImprove` built with KEYWORDS from the recorded constructor
  arguments (`seed=`, `refit_every=`, `buffer_cap=`, `l_max=`, `eps_tol=`, `T=`), `device="cpu"`,
  `growth_gate=<the replay gate>` and `online_readout="delta"`: the deployed constructor (`Brain._make_neuro`), with
  the device pinned to CPU (section 2.3) and the online readout the deployed `OnlineReadout("delta")` at its defaults
  (mu 1.0, sign +1). (`seed` is not the first positional parameter.)
- **(f64) is not the historical ORACLE_PREVCAT** (a 64-wide one-hot, online readout removed); its readings here are
  new and owe their own numbers.
- **k = 4 and N = 256** are declared settings, not a capacity equivalence or a memory horizon. N = 256 is also the
  width G1(b)'s pools must use, so that G1(b) cannot tune it.
- **Red-only constructions, never scored:** (leak) a predictor that answers c_t read from the stream, bypassing the
  learner (R-a2's positive control); (forbidden) a sensory feature map that asks its view for c (R-a2b's); (noise) (a)
  with unseeded `os.urandom` noise added to phi (R-a4's).
- `onehot_32` refuses a category of 32 or more. A stream that exceeds it is recorded as such, and the arm set is
  re-registered before it is read.

### 2.3 Recording

- **Execution policy (frozen).** Every process sets torch to 1 thread. The grower is pinned to CPU: the recorder sets
  the in-brain engine's `device` to `"cpu"` before its first refit, and every replay builds its learner with
  `device="cpu"`. The artifact records `NORD_DEVICE`, `NORD_GPU`, the torch, numpy and Python versions and the
  platform; a recording or replay under another policy is refused.
- **Prerequisites.** The brain must hold the packs and checkpoints of section 1.8 (by hash), and every delivery after
  the bootstrap delivery x_1 must have been built from a SUCCESSFUL acquisition of BOTH modalities at the previous
  wake tick (section 1.3's flags are not enough). Zero grower feed refusals (`_neuro_feed_refusals`). Any failed
  acquisition or fallback context refuses the recording; a degraded stream is never accepted in its place.
- **Per wake tick, the acquisition attempt:** for each modality, the pointer value used, success, and the frame's
  ORIGINAL pack index (the slot through the inverse of the permutation after the switch).
- **Per wake percept:** x_t, c_t; the acquisition attempt that built x_t's context (the attempt at the immediately
  preceding wake tick), with its success flags, or "bootstrap, no frame" for x_1; the CAUSAL |LP| (read at this
  delivery, before the tick's acquisition); and the in-brain engine's prediction for this event (`predict_next()`
  before the delivery), with its prior-fallback flag.
- **Per sleep tick:** the LIVE |LP| (read before the tick runs); the gate's effective verdict whenever it is
  consulted; and, after each COMPLETED refit (the trajectory grew; a skipped refit is recorded as a skip), the model
  fingerprints keyed by event index and engine tick: crc32 of W, b, beta, prior and the batch beta.
- **At the end:** the in-brain engine's live counters, trajectory, baselines, gate decisions and final fingerprints,
  plus `head`, `dirty`, the policy and the asset hashes.
- **The unseen-order segment.** 2,400 ticks per seed. After the delivery of wake percept 720 returns and before that
  wake tick's acquisition, the recorder permutes the frame order of the vision and audio packs, frames and labels
  together (a seeded permutation, seed x 7919 + 1, used nowhere else), so percept 721 is the first percept from the
  new order. The segments are derived from the acquisition records, never from fixed indices: x_1 (bootstrap)
  belongs to none; percepts whose frame lies on the canonical order's first pass are the unscored warm-up;
  repeated-canonical is the canonical order after its first pass; unseen-order is the new order's first pass (256
  acquisitions from the switch); repeated-permuted follows. Under uninterrupted acquisition that is warm-up x_2-x_257,
  repeated-canonical x_258-x_720, unseen-order x_721-x_976, repeated-permuted x_977-x_n. Denominators are reported,
  and an empty segment is reported as unavailable.

### 2.4 Replay

- **Two modes.** LIVE (the sleep tick's |LP|) serves only R-a1 and R-a1b. CAUSAL (section 2.1) serves every reading.
- **The |LP| ledger.** The replay's |LP| reader logs every read (engine tick, mode, value, and whether a recorded
  value existed) in a ledger the harness holds OUTSIDE `GrowthStop._read_lp`'s catch. A replay is refused if any read
  was unrecorded, OR if the ledger does not hold exactly one read for every trajectory row the gate fed
  (`observe_refit`), so a disconnected reader cannot pass on an empty ledger. A recorded None (an unfed channel) is
  accepted; a missing record is not. Reds: a replay whose gate has no reader, and a replay with one recorded |LP|
  deleted, must both be refused.
- **Fingerprints.** crc32 of every phi_t, and of the learner's W, b, beta, prior and batch beta after every refit and
  at the end.
- **The shared gate.** `Brain._growth_gate_verdict`'s loop and verdict move into `cortex/growth_stop.py` as one
  function that the brain and the replay both call.

### 2.5 Scoring

- **Rows.** The segments of section 2.3, identical for every arm.
- **Per arm, seed and segment:** accuracy (a None prediction is a miss); coverage (the fraction of rows with a
  prediction); the marginal prediction entropy (bits, over the rows with a prediction) beside the true categories'
  entropy; the prior-fallback frequency (predictions `decide` answered from its prior); and, on the same rows, the
  persistence floor and the causal majority (the running mode of c_1..c_{t-1}).
- **Nothing measured in this arc is confirmatory.** Pilot numbers license no claim; they feed G1(b) and G1(c).

### 2.6 Checks, each with a constructed red

- **R-a0, the refactor is behavior-identical.** After the build, `probe_prepost.py` on seeds 900 and 901 prints JSON
  identical to section 1.11's references, and the pinned tests stay green (`tests/test_growth_stop_wire.py`,
  `tests/test_neurogenesis_live_instrument.py`, the fast tests of `tests/test_real_percept_stream.py`). Red: a unit
  test pins the old loop's outputs (cursor, refusals, verdict) on a trajectory with a poisoned row, and a variant of
  the shared function that drops one row must fail it.
- **R-a1, replay equivalence (LIVE mode).** On each pilot seed, the replayed incumbent reproduces the in-brain engine
  event for event: every per-event prediction, every gate verdict, the fingerprints after every refit, the live
  counters, the trajectory and the total event counts. It is asserted that scoring rows exist and that at least one
  refit completed. Reds, each witnessed by a divergence of learner fingerprints or predictions (a changed event count
  or timestamp is never a sufficient witness): (i) count-preserving: one wake percept's category changed before the
  first completed refit; (ii) LP-sensitive: on a synthetic fixture where the gate's verdict depends on |LP|, one
  recorded |LP| changed so that a verdict flips.
- **R-a1b, in-brain equivalence of a fed arm.** On each pilot seed, (f64) runs in-brain through this harness's own
  recorder, feeding the grower as `_l_cell` does (the feed idiom; `_l_cell` itself is not called: it runs 800 ticks,
  removes the online readout and returns aggregates). Its complete event sequence (percepts, categories, sleep
  positions, LIVE and CAUSAL |LP|, acquisition records) equals the incumbent recording, and its in-brain predictions
  and per-refit fingerprints equal the LIVE-mode replay of (f64). Reds: as R-a1.
- **R-a2, causality (CAUSAL mode).** For every scored arm, a second replay mutates the suffix from a cut t* =
  floor(n/4), floor(n/2) and floor(3n/4): x_s for s >= t* by seeded uniform noise; c_{t*} by the smallest seen
  category other than c_{t*}; later c_s by seeded random seen categories; the CAUSAL |LP| of deliveries >= t* by
  seeded values. Every cached prediction y^_t for t <= t* must equal the unmutated replay's. Red: (leak) must change
  at every cut.
- **R-a2b, channels.** Every feature map runs through its declared view. Red: (forbidden) must raise.
- **R-a3, the scorer is calibrated.** On each pilot stream and segment, planted predictors score exactly their known
  values: persistence (y^_t = c_{t-1}) scores the independently computed persistence floor, the future oracle
  (y^_t = c_t) scores 1.0, and a constant predictor scores its frequency. Red: an off-by-one scorer (comparing y^_t
  with c_{t-1}) must fail on a fixed synthetic fixture with category transitions. On real segments the count of rows
  where c_t differs from c_{t-1} is reported, and a segment with none is marked non-discriminating for this red
  rather than required to fail. The contrast (f64) vs (a) is REPORTED per pilot seed as the early reading of G-R1's
  REFUSED clause; it is not presented as a proof about the harness.
- **R-a4, determinism.** Two fresh-process CAUSAL replays of every scored arm on seed 100 give identical phi
  fingerprints, learner fingerprints and predictions. Red: (noise) must differ in its phi fingerprints.

### 2.7 Seeds, the run matrix and the budget

- **Seeds.** Pilot 100-103; G1(b)'s selection seeds 200-207; G1(c)'s confirmatory seeds from 300; the refactor
  reference 900-901. Exposed and never used for a G1 confirmatory reading: seed 0 (section 1.2) and the L-row seeds
  0-7. Each recording's unseen-order permutation seed is seed x 7919 + 1.
- **The pilot matrix.** 4 recordings of 2,400 ticks (about 130 s each); 4 in-brain (f64) runs (about 130 s each);
  about 110 replays: 8 scored arms on 4 seeds (CAUSAL), (a) and (f64) in LIVE mode, R-a2's 3 cuts on seed 100, and
  R-a4's twins on seed 100. Each replay measured under 2 s at 800 ticks, CPU, width 256.
- **Limits.** 4 worker processes; torch 1 thread; a 900 s timeout per process (a timeout is recorded as TIMEOUT, never
  as a result); temporary homes removed. The module's self-check uses synthetic fixtures only and runs in under 20 s.

### 2.8 Disposition on a failed check

- **R-a0 fails:** the refactor changed behavior; it is reverted, and nothing else runs.
- **R-a1 or R-a1b fails:** the replay is not the live stream. No arm is read; the harness is repaired, or G1 returns to
  in-brain arms under a new pre-registration.
- **R-a2 or R-a2b fails on a real arm:** that arm leaks or reads a forbidden channel; it is fixed before any run. A red
  that does not fire means the check is blind, and nothing is read.
- **R-a3's calibration fails:** the scorer is wrong; it is fixed before anything is read. If (f64) does not beat (a) on
  the pilot streams, that is recorded as the early G-R1 REFUSED indication for G1(c) to confirm.
- **R-a4 fails:** the learner or an arm is nondeterministic under the frozen policy; it is fixed before G1(b).

### 2.9 Scope, consumers and dispositions

- **Built here:** the recorder; the replay in its two modes with the |LP| ledger; the arms of section 2.2 and the three
  red-only constructions; the scorer; checks R-a0 to R-a4; the engine's `predict_next()` with the cached per-event
  decision; the shared growth-gate function; a governed module registered in `REGIONS` and `run_selfchecks`, whose
  self-check constructs every red on synthetic fixtures; and tests.
- **Consumers (P3).** The module's campaign entrypoints (record, pilot, check), run in this arc, and its tests. Its
  dormancy disposition is an explicit EXEMPT row as a governed benchmark harness, naming those consumers.
- **Deferred:** the within-block time shuffle is built with its consumer in G1(c); pools and the selection benchmark
  in G1(b).
- **`_l_cell` is retained** as the independent in-brain reference that serves the L-row tests' protocols. This harness
  does not supersede it. R-a1b checks this harness against an in-brain run that uses the same feed idiom; it does not
  call `_l_cell`.
- **Carriers.** All harness state (history buffers, cached feedback, replay learners, recordings) is benchmark-only,
  non-resumable, and never part of the live brain or its lake (P4b). The recordings are experiment artifacts written
  outside the repo.

### 2.10 The design read and its fold

Codex (`gpt-6-astra`, high effort, read-only, dispatched through claude-octopus on the Codex desktop app's engine) read
the draft at `71e2e4c` and returned 23 findings: 2 BLOCKER, 20 MAJOR, 1 MINOR. The orchestrator reproduced the
decisive code facts before folding: the acquisition order (section 1.3), `_read_lp`'s catch (1.6), the device pin
(1.5), `decide`'s writes (1.5) and the exogenous schedule (1.7). All 23 are folded:
F1 -> section 1 and the archive; F2 -> 1.4 and R-a1b; F3 -> 1.3 and 2.3; F4 (BLOCKER) -> 2.1's CAUSAL |LP|; F5 -> 2.1's
channels and R-a2b; F6 -> 1.5 and 2.3's policy; F7 (BLOCKER) -> 2.4's ledger; F8 -> 1.9 and 2.2's (f64) note;
F9 and F10 -> 2.2's widths and controls; F11 -> 2.3's prerequisites; F12 -> 2.3's segments; F13 -> 2.7's seeds;
F14 -> R-a1; F15 -> R-a2; F16 -> R-a3; F17 -> R-a4; F18 -> 2.5; F19 -> 2.1 and R-a0; F20, F21 and F22 -> 2.9;
F23 -> 2.7. The review is archived as `archive/g1a-2026-09-24/design_read.txt`.

A second read (the same channel, on the folded text) returned 10 findings (7 MAJOR, 3 MINOR) and confirmed that the
CAUSAL |LP| repair, the CPU pin for a fresh non-restored engine, the cached decision and the permutation hook hold. The
orchestrator reproduced its code facts (the constructor's parameter order, `_perception_unified`'s either-modality
rule, the `consume_surprise` site). All 10 are folded: S1 -> 2.2's keyword constructor; S2 -> 2.3's prerequisites
(both modalities, every delivery); S3 -> 2.3's acquisition attempts with original frame identity; S4 -> 2.3's
per-refit fingerprints; S5 -> 2.4's ledger count and its two reds; S6 -> R-a1's witnessed reds; S7 -> R-a3's fixture
and its per-segment disclosure; S8 -> 1.3 and 2.3's record-derived segments; S9 -> R-a1b and 2.9; S10 -> 1.3. The
second review is archived as `archive/g1a-2026-09-24/design_read2.txt`.

## 3. Results (2026-09-24, every artifact at head `520eff9`, `dirty` False)

### 3.1 The build and its verification

- **Built** by Codex `gpt-6-sol` through claude-octopus on the desktop app's engine, from a bounded brief. The first
  dispatch wrote nothing: octopus's default curated persona is read-only and its injected tool policy forbade edits
  inside a `workspace-write` sandbox. The launcher now drops that text for writing tasks. Commits: `516ad40` (the shared
  gate verdict), `f29f9f5` (`predict_next`), `8fa5b0d` (the harness), `520eff9` (the campaign fix below).
- **The orchestrator's review fixed three deviations before the pilot:** segments now come from the frames' pointer
  passes, as section 2.3 says (the build counted deliveries, and a reflex tick acquires without delivering); the
  CAUSAL |LP| before the first delivery is a recorded None, as section 2.1 implies (the build treated it as missing);
  and campaign children now run the module by its own name (the first pilot's 8 recordings died in 2 s on
  `python -m __main__`).
- **R-a0 PASS.** `probe_prepost.py` on seeds 900 and 901 printed JSON byte-identical to the references taken at
  `71e2e4c`. The pinning test (the shared function against the old inline loop on a poisoned row, with a dropped row
  caught), 16 fast tests, ruff and the module's self-check are green.

### 3.2 The pilot (seeds 100-103, 2,400 ticks each)

- **Campaign.** 160 of 160 jobs OK in 453 s: 4 incumbent recordings, 4 in-brain (f64) recordings and 152 replays. The
  checks took 126 s. Each seed gave 1,183-1,190 wake percepts, 60 completed refits and 4-7 categories (`onehot_32`
  never refused). The segments, derived from the acquisition records, are warm-up 256, repeated-canonical 463,
  unseen-order 256 and repeated-permuted 207-214 on every seed. Every delivery after x_1 came from a successful
  acquisition of both modalities.
- **Checks, on every seed:**
  - R-a1 PASS: the per-event predictions (1,183-1,190), the 60 per-refit fingerprints, the gate verdicts, the counters
    and the trajectory were identical. Both reds diverged: the count-preserving category change, and the LP-sensitive
    fixture, which flipped a verdict.
  - R-a1b PASS: the in-brain (f64) run's complete event sequence equals the incumbent recording, and its predictions
    and per-refit fingerprints equal the LIVE replay of (f64). Its reds fired.
  - R-a2 PASS: every scored arm's predictions up to each of the three cuts were unchanged by the suffix mutation, and
    (leak) changed at every cut.
  - R-a2b PASS: (forbidden) raised.
  - R-a3 PASS: the planted predictors scored exactly, and every scored segment had transitions.
  - R-a4 PASS: fresh-process twins on seed 100 were identical in phi, learner fingerprints and predictions (seeds
    101-103 were also checked with in-process twins), and (noise) differed in phi.
- **The second instrument.** A re-scorer written from sections 2.3 and 2.5, not from the module
  (`archive/g1a-2026-09-24/pilot/recheck_g1a.py`), reproduced the segments and every n, accuracy and persistence
  value: 0 disagreements over 4 seeds x 8 arms x 3 segments. It also reproduced R-a1's raw equality.

### 3.3 Pilot readings (exploratory; they license no claim)

Accuracy on repeated-canonical / unseen-order / repeated-permuted:

| arm | seed 100 | seed 101 | seed 102 | seed 103 |
|---|---|---|---|---|
| (a) | .970 / .977 / .967 | .778 / .926 / .519 | .903 / .934 / .884 | .890 / .965 / .919 |
| (f64) | .983 / .980 / .986 | .987 / .957 / .822 | .970 / .949 / .942 | .898 / .984 / .962 |
| (d) | .965 / .945 / .976 | .924 / .820 / .584 | .652 / .855 / .821 | .879 / .965 / .962 |
| (d0) | .909 / .734 / .766 | .940 / .910 / .519 | .758 / .922 / .923 | .927 / .961 / .962 |
| (e1) | .903 / .863 / .957 | .890 / .754 / .776 | .933 / .906 / .768 | .898 / .957 / .856 |
| (e1-0) | .976 / .980 / .976 | .596 / .703 / .491 | .937 / .934 / .928 | .937 / .969 / .933 |
| (e2) | .888 / .727 / .976 | .948 / .641 / .598 | .890 / .945 / .884 | .918 / .973 / .962 |
| (e2-0) | .894 / .828 / .986 | .857 / .957 / .808 | .940 / .941 / .913 | .879 / .973 / .967 |
| persistence | .987 / .984 / .986 | .996 / .973 / .907 | .974 / .969 / .961 | .970 / .984 / .981 |

- **The oracle (f64) exceeds the incumbent (a) in 12 of 12 seed x segment cells.** G-R1's early REFUSED indication
  does not fire on the pilot streams.
- **Coverage** is 1.000 for every arm and segment. The prior-fallback rate reaches 0.43 for (e1-0) and 0.34 for (a),
  and is 0 for (f64). Mean prediction entropy is 1.17-1.29 bits, against 1.26 for the true categories.
- **Observed for G1(b) and G1(c)'s design (not findings):**
  - no arm reaches persistence consistently;
  - the learner's seed-to-seed variance is large: (a) reads 0.519 against a persistence of 0.907 in one of seed
    101's segments;
  - zero padding alone moves the learner: (a) 0.778 against (e1-0) 0.596 on seed 101, so every contrast needs its
    width-matched control, as pre-registered;
  - the history arms against their null controls are mixed in sign across seeds.

### 3.4 Deviations from section 2

1. R-a2 ran on all four pilot seeds. Section 2.6 names every scored arm; section 2.7's matrix named seed 100 only.
   A superset.
2. R-a4's fresh-process twins ran on seed 100, as registered; seeds 101-103 were additionally checked in-process.
3. R-a3's off-by-one red is built as persistence-shifted predictions under the correct scorer, on the fixture, which
   is equivalent to shifting the scorer by one. R-a0's red is a dropped-row input in the pinning test; a function that
   dropped a row would fail the same equality.
4. Section 3.1's three orchestrator fixes. Two implement sections 2.3 and 2.1 as written; the third was a bug that
   blocked the campaign.
5. Disposition: every check passed, so G1(b), the pool-selection benchmark, is next. The pilot recordings validate the
   harness; they are neither selection data nor confirmatory data.
