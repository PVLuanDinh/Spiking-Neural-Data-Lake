# Verified engine data and comparison notes

Prepared November 2, 2026, America/Chicago. These notes support the presentation plan, not a new experiment.

Repository inspected: `c4c10b285f1741aa798d5ee9c5296ae371fabecf`. Local saved-data example: [SHOWCASE_DATA_EXAMPLE.json](SHOWCASE_DATA_EXAMPLE.json).

## Evidence labels

- **Recorded local experiment:** values and states occur in the saved component experiments.
- **Implemented interface:** source code defines and consumes these fields, but the local recovery experiment did not execute this path.
- **Research objective:** the complete engine's causal benefit or recursive self-improvement is not established by these records.

Do not combine fields from separate learners into a fictional single execution trace. The local delta-readout experiment and the posed grown-neuron/return path are separate paths.

## Concrete data at each processing stage

The identifiers below are existing source identifiers or expressions. Shapes describe the current implementation. They are not proposed future schemas.

| Stage | Consumes | Produces or changes | Evidence and boundary |
|---|---|---|---|
| Task episode | `Problem.teach_epochs(...)` or `Problem.episodes(...)`; each source `Turn` has `text`, `role`, `label` | An ordered episode. Local records store `texts: list[str]`, `roles: list[str]`, `target: str`, and later `x: list[float]`. | Recorded locally. During evaluation the learner receives label-free inputs; the scorer retains `target`. |
| Text encoding | `text` and the configured `n_in` | `text_features(text, n_in)` produces a weighted CRC32 bucket vector. `im.ranks(text)` produces integer spike ranks. `im.code(text)` produces PSP-of-rank values. `pool_features` creates its local zero-padded input `x`. | Actual pipeline, not a pretrained language embedding. In the recorded configuration the code has 52 entries and the reservoir input has 64. These are reproducibility details, not the engine's central idea. |
| Reservoir processing | `pool_features.x`, `pool.w_in`, recurrent `pool.pre`/`pool.w`, and activity carriers | `percept_drive(u)` returns mean activity `total / self.t_in`. `_Drive.walk` stores this as `Turn.x`; the local records store it as `row['x']`. `KernelLsmPool.v`, `.s`, and `.acc` change as the kernel steps. | The local pilot records feature vectors. `v` is the kernel membrane array, `s` is a Boolean spike array, and `acc` is an integer accumulator view. Do not invent physical voltage units or imply that these carriers were saved in the local checkpoint. |
| Task prediction | The current feature vector and readout `unit.beta`; `unit.labels` maps output indices to classes | The delta adaptor evaluates `s @ self.beta`, selects `argmax`, and returns a string from `self.labels`. | Recorded locally. `s` here is the adaptor's input tensor, not the reservoir's Boolean spike array. Output scores are not calibrated probabilities. |
| Online learning | `unit.learn(row['x'], row['target'])`, current `beta`, and the existing delta rule | Updated `unit.beta`, readiness, and rule fields including `steps`. | Recorded locally. The teacher supplies the training target. During this experiment the reservoir topology and parameters stay fixed. |
| Growth decision | `learner.diagnose()`, `ProblemRecord.budget`, `spend_since_teach`, `spend_units`, and attempts | A source batch is admitted, or the ledger reports `SOLVED`, `UNSOLVED`, `CONTRADICTORY`, `LEARNING_RULE_DEFICIT`, or `UNRESOLVED`, as determined by the actual loop. | Implemented interface. These fields are not present in the local recovery trajectory. `UNSOLVED` does not close the problem. |
| Structural growth | The batch returned by `source(self.batch)` plus admission labels/ticket | New `GrownUnit` entries with `unit_id`, `w_in`, `threshold`, `window`, `sign`, `out`, and `active`; journaled birth rows add `problem_id`, `batch_id`, position/region/layer metadata, and a digest. | Implemented interface with a separately reported upstream growth baseline. The input weights of an admitted `GrownUnit` are frozen in this representation; its output weights can learn. Adding units does not demonstrate that the growth improves performance. |
| Grown-unit feed | In `PosedActor._effect`, `ranks`, grown `units`, prior return `afferents`, `tau`, and `t_post_g` | `phi = segment_psp(...)`; `feed` is the matrix-vector sum over `self.feed` and `phi`; `features = pool_features(pool, text, feed=feed).tolist()`. | Implemented posed-problem interface. `feed` contains one finite drive per reservoir unit. The local delta recovery pilot uses the feature path without this grown-unit feed. |
| Readout return | The posed reader's selected `answer`, its emission time, target crossings, masks, and reward | `packet = (answer, self.clock + float(neuron.t_post))`; `port.apply(...)` receives that packet and the target/learning arguments. Return context affects later processing through `port.context(...)`. | Implemented for the posed label-neuron path. Do not splice this packet into the local delta experiment or claim an end-to-end causal result from source availability. |
| Offline replay/consolidation | Saved training `replay` rows, their `x` and `target`, and the selected checkpoint | Replay calls `unit.learn(...)`; the batch arm calls the existing solver and replaces `unit.beta`. It reports `presentations` and `solves`. | Recorded locally. Only training examples enter the update. The experiment does not validate biological sleep stages or pruning of this readout. |
| Local save | `schema`, `head`, `seed`, `phase`, `pool`, and `state = {'readout': ..., 'replay': ...}` | An envelope containing `body` and `sha256`, saved by `save_named_state` into `brain_lake.bin`. The function returns a receipt digest. Phases include `after_a`, `b30`, `b100`, `b300`, and `result`. | Recorded locally. The readout state includes `beta`, `dtype`, `labels`, `n_in`, `ready`, `buffer`, and `rule`. The `pool` field stores recipe/configuration, fingerprints, shapes and hashes, not all numerical weight arrays. |
| Local restore | The expected receipt, named-state bytes, and label-free `texts`/`roles` for evaluation | `load_named_state` verifies the envelope; `restore_unit` reconstructs the readout and replay; a seed/configuration rebuild of the pool regenerates the recorded features. | Recorded locally at episode boundaries. Full restored readout/replay and predictions match. Mid-episode carrier continuation and whole-Brain crash recovery are not tested. |

Use `H = pool.n` only as a notation for the configured reservoir width. In the reviewed local experiment H stays fixed. Changing notation must not imply that this particular reservoir expanded during the experiment. The active grown population has its own concrete unit list, recorded by the growth/posed interfaces.

## Exact example from saved data

Development seed 8000, development row 1:

- Input `texts = ['fog gin keg mod']`.
- Input `roles = ['probe']`.
- Scorer-only `target = 'GREET'`.
- Recorded `row['x']` is the full activity feature vector included in the JSON example. No vector values have been synthesized.

| Recorded phase | Prediction for this probe | `readout.rule.steps` | Replay rows |
|---|---|---:|---:|
| After A | `GREET` | 480 | 16 |
| After B30 | `GOAL` | 960 | 32 |
| After replay | `GREET` | 1280 | 32 |
| New-process restore | `GREET` | 1280 | 32 |

This is one illustrative probe, not an additional statistical result. The complete A/B results remain the evidence for recovery. The example file includes exact source paths, source-file SHA256 values, saved state hashes, and the full feature vector. `after_replay` and the restored full state are equal.

## Source map

All paths are under the pinned repository or the saved local experiment.

| Contract | Source and symbol |
|---|---|
| Input episode | [taught_problems.py](../repo/src/snn_data_lake/instruments/taught_problems.py), `Turn`, `Problem._episode`, `teach_epochs`, `episodes` |
| Text features | [interaction_learning.py](../repo/src/snn_data_lake/readout/interaction_learning.py), `text_features`; [inseparable_mapping.py](../repo/src/snn_data_lake/instruments/inseparable_mapping.py), `ranks`, `code` |
| Pool input/output | [g5_readout_cells.py](../repo/src/snn_data_lake/instruments/g5_readout_cells.py), `pool_features`; [g5_part_d_cells.py](../repo/src/snn_data_lake/instruments/g5_part_d_cells.py), `Turn`, `_Drive.walk`; [recurrent_pool.py](../repo/src/snn_data_lake/reservoir/recurrent_pool.py), `_Pool.percept_drive`, `percept_fed`, `KernelLsmPool` carrier properties/step |
| Delta prediction and update | [g5_part_d_adaptors.py](../repo/src/snn_data_lake/instruments/g5_part_d_adaptors.py), `DeltaAdaptor.learn`, `_answer` |
| Growth | [growth_loop.py](../repo/src/snn_data_lake/growth/growth_loop.py), `ProblemRecord`, `Ledger._birth_row`, `GrowthLoop._grow`, `run`; [grown_code.py](../repo/src/snn_data_lake/growth/grown_code.py), `GrownUnit` |
| Feed/return | [posed_problem.py](../repo/src/snn_data_lake/readout/posed_problem.py), `PosedActor._effect`, `state_dict` |
| Offline processing/save/restore | [class_block.py](reports/05-artifacts/class_block.py), `prepare`, `teach`, `offline`, `checkpoint`, `restore`; [evaluate.py](reports/05-artifacts/evaluate.py), `snapshot`, `batch_update`, `restore_unit`; [stage-04 state/read helpers](reports/04-artifacts/evaluate.py), `state`, `answer`, `param_state` |
| Original sample and saved outputs | [inputs](reports/05-artifacts/class-pilot-3/inputs-8000.json), [discovery](reports/05-artifacts/class-pilot-3/discovery-8000.json), [replay](reports/05-artifacts/class-pilot-3/replay-8000.json), [restore](reports/05-artifacts/class-pilot-3/restore-replay-8000.json) |

## Comparison after the Objectives slide

Use the title **Adaptation and Self-Improvement Across Neural Systems**. Follow this comparison immediately with the concrete data slide.

A checkpoint is a stored model state. It is not the output type of a transformer or SNN. A standard frozen-weight transformer generates output distributions/tokens while its context/cache evolves. An SNN produces spikes or a decoded task output and can have changing neural activity even with fixed weights. Snapshot storage and persistent adaptation are separate properties.

Do not use the blanket statement that current transformers or SNNs only produce snapshots or cannot self-improve. Current primary research includes online neural memory, expanding SNNs, and self-modifying foundation-model agents.

| Approach | What can adapt | What the comparison may claim |
|---|---|---|
| Standard frozen-weight transformer deployment | Context and runtime/cache state change. Parameter updates require an additional training/adaptation procedure. | A useful comparison for inference with frozen parameters. Do not generalize it to all transformer-based systems. |
| Adaptive attention/neural-memory research | Titans introduces a neural memory that learns historical context at test time alongside attention. | Online memory adaptation already exists in attention-based research. It does not itself demonstrate recursive improvement of the learning mechanism. |
| Current adaptive SNN research | PS-SNN combines incremental learning with dynamically expandable network structure and pattern-separation targets. | Continual learning and neurogenesis-inspired expansion are existing SNN research directions. SNDL cannot claim them as unique by themselves. |
| Foundation-model self-improving agents | DGM modifies and evaluates agent code; Hyperagents also makes the mechanism that proposes modifications editable. | There are current system-level self-improvement comparators. Code/agent-level changes are distinct from changes to a spiking substrate. |
| SNDL Engine | The research design connects local learning, budgeted structural growth, reservoir/readout feedback, offline processing, and reconstruction from a single durable lake. | This is the integration under investigation. Component evidence is available. End-to-end performance benefit and recursive self-improvement are not established by the experiments reviewed here. |

For the slide, compress the table into a few comparison axes: what changes during use, where structural changes occur, what persists, and what level of self-improvement is actually evaluated. Keep the named research exceptions in speaker notes or a readable reference line. This is a conceptual comparison, not a head-to-head benchmark or a claim of state-of-the-art superiority.

Primary references, checked November 2, 2026:

1. [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663). Describes test-time neural memory integrated with attention.
2. [PS-SNN: pattern separation learning for expandable spiking neural networks in class-incremental learning](https://www.nature.com/articles/s41598-026-42970-6). Published March 9, 2026. Describes continual learning with dynamic expansion.
3. [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954). Latest inspected revision March 12, 2026. Describes agent code self-modification with empirical evaluation.
4. [Hyperagents](https://arxiv.org/abs/2603.19461). Submitted March 19, 2026, preprint. Describes editable task and meta agents, including modifications to the improvement mechanism.

## How to discuss recursive self-improvement

For the talk, distinguish:

- **Continual adaptation:** parameters, activity or structure change with additional experience.
- **Recursive self-improvement:** changes improve the mechanism or capability used to generate subsequent improvements. This is a working distinction for this presentation, not a claim that the field has one universally settled definition.

Growth, a repeated learning loop, and restart persistence do not by themselves establish the second claim. A future SNDL RSI experiment must identify what improvement mechanism changes, how the modified system generates subsequent improvements, and which held-out measurements and controls establish that benefit. The reviewed local records contain no validated `RSI_score`, and no such variable should be invented for the data slide.

Suggested positioning: **SNDL investigates whether local learning, structural growth, feedback, and durable memory can support one continuing self-improvement process. Our current evidence tests components of that process.**

Keep RSI as an explicit future research goal if the team wants that emphasis. Do not present it as a demonstrated capability or a capability absent from every current ANN/SNN system.
