# SNDL large scale evaluation plan

Prepared November 3, 2026, America/Chicago. Status: proposed research protocol, not launch ready.

The next study should measure how learning, retention, active structure, and resource use change during a sustained stream of real examples. The current binary text experiments remain useful apparatus checks. They are too small to represent the complete engine's performance. Repeating their examples for longer would not fix that limitation.

Target local execution on the existing Mac, subject to the runtime and resource checks below. Begin with a real-data learning benchmark and progress to the integrated engine when its execution gate passes. Report each route under its actual scope.

## Research questions and scope

Primary integrated-engine question: **At the same declared resource budget, does the complete SNDL Engine acquire and retain real classification tasks better than a comparable fixed-capacity learner?**

Secondary questions concern the causes and cost of any difference: earned versus random growth, feed and return, offline replay, representation drift, and restoration after restart. The study should also expose when growth allocates capacity without improving held-out behavior.

The real-data component benchmark asks how the existing event-audio learning path performs across a substantial sequence of examples and classes. It can establish learning and forgetting in that component while engine integration proceeds. It cannot establish G9 or recursive self-improvement.

Credibility depends on task construction, data independence, fair controls, and traceable measurements. The protocol must retain ceiling results, chance-level failures, and negative findings. Do not select tasks or runs because they yield attractive intermediate percentages.

## Current readiness and required repairs

The reviewed source is the clean detached checkout at `c4c10b285f1741aa798d5ee9c5296ae371fabecf`.

| Area | Verified state | Required action |
|---|---|---|
| Local hardware | Earlier environment report identifies Apple M2, arm64, 8 GB RAM. Local component experiments ran on CPU. | Recheck available memory and disk before each campaign block. Use one worker initially. |
| Runtime today | The 45-second import/tensor preflight timed out. A second 30-second diagnosis started Python 3.14.6t and imported NumPy in about 0.47 seconds, then timed out during Torch import. The cause is unresolved. | Restore a reliably importing, version-pinned runtime and measure warm and cold startup. Do not interpret import failure as a learning result. |
| Full-engine readiness | `CanonicalGrowthAdapter.require_ready` currently raises `G9_A_UNREADY`. Its bound path lists retention selection, grown-source/admission binding, the G6 mask/progress carrier, and durable consumption of the final retention reading as debts. Sequence budgets can also require a source-owned BETA baseline. | Resolve and test these interfaces before any run is called a complete-engine campaign. Do not bypass the refusal or substitute scripted success. |
| Real audio learning | `EpropAudioNet`, `EpropAudioEncoder`, an SHD loader, logits, and local e-prop updates already exist. | Validate their current runtime and real-data behavior at useful scale. An encoder benchmark alone is a component result. |
| Continual-learning runner | The stock SHD training entry evaluates the official test loader each epoch. It does not provide the required independent development split and complete continual-engine comparison. | Build a wrapper with a separate development split and a sealed final-test path. Reuse the existing model/update code. |
| Numeric/event task bridge | The reviewed taught-problem/growth path is built around text codes and label-neuron interfaces. Existence of an audio encoder does not establish an integrated event-audio growth experiment. | Specify and verify an event/numerical Problem adapter and its consumers. Never encode the true class into an input token or latent feature. |
| Historical seeds | Many seed blocks are already used or reserved. | Reconcile the seed ledger and register separate development and confirmation cohorts. Do not assume an unused-looking integer is available. |

Source evidence: [environment report](reports/01-environment.md), [current preflight](reports/scale-preflight.json), [import diagnosis](reports/scale-import-diagnosis.json), [growth adapter](../repo/src/snn_data_lake/growth/growth_loop.py), [audio encoder and trainer](../repo/src/snn_data_lake/sensorimotor/eprop_shd.py), and [upstream tasks](../repo/specs/022-unified-brain-arc/tasks.md).

## Work packages and execution gates

| Stage | Work | Exit condition | Result if the condition fails |
|---|---|---|---|
| 1. Runtime and throughput | Repair imports, check free-threading after imports, validate a real-data loader, and measure a bounded training/evaluation segment. | Correct predictions and updates can be recorded within memory and time limits; repeated CPU runs reproduce within a declared numerical tolerance. | Environment or throughput report, with no scientific conclusion about the mechanism. |
| 2. Real-data component runner | Add train/development/test boundaries, sequential class blocks, controls, and full logging around existing learning code. | A positive control beats its stated floor on development data, and planted scoring/leakage faults fail. | A measured representation, learning-rule, or apparatus limitation. Larger runs wait for a diagnosis. |
| 3. Development pilot | Run a small registered set of paired streams to estimate variability, run time, and task difficulty. | Freeze configuration, practical effect size, statistical design, and feasible run count using pilot evidence. | An exploratory pilot with an explicit reason confirmation is not ready. |
| 4. Integrated engine | Complete growth/feed/return/retention/save bindings and run an isolated end-to-end readiness trajectory. | Actual inputs reach admitted units, their feed reaches the reservoir, return changes downstream behavior, budgets hold, and restart/retention checks pass. | Continue the component study and report the integration gap. Do not label it a full-engine result. |
| 5. Confirmatory comparison | Run the frozen primary comparison on untouched streams and test data at scheduled looks. | A complete per-run record and a valid statistical reading, whether positive, harmful, limited, or inconclusive. | Technical failures remain visible and follow the predeclared failure policy. |
| 6. Attribution and extension | Execute predeclared ablations and the larger dataset extension as compute permits. | Each claimed mechanism has its own measured contrast and resource accounting. | Restrict the conclusion to the demonstrated contrast and scale. |

Stages 2–3 can proceed for components independently of Stage 4. Integrated confirmation requires Stage 4. Integration completion has no defensible fixed duration yet.

## Data and workload scale

Use real event-audio first because the repository already has its learning/encoder implementation. It does not require new camera hardware or changes to the robot envelope.

| Dataset | Official classes | Official train / validation / test | Proposed use |
|---|---:|---|---|
| Spiking Heidelberg Digits, SHD | 20 | 8,156 / no supplied validation split / 2,264 | First real-data learning and continual-retention study |
| Spiking Speech Commands, SSC | 35 | 75,466 / 9,981 / 20,382 | Larger extension after the dataset/class-head adapter passes validation |

Both provide spike times and channel IDs over 700 input channels. The [dataset authors](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) document these specifications and official splits. Verify the downloaded manifests and checksums before using the counts as run metadata. Stream examples from disk rather than materializing the full dense dataset in RAM.

For SHD, create development data from the official training pool, preferably by speaker grouping after checking class coverage. The official test pool remains sealed. Report test-only speakers separately from seen speakers. For SSC, preserve the official splits and speaker grouping metadata where available. Record any missing metadata and its effect on the generalization claim.

N-MNIST may remain an installation or vision-adapter diagnostic. It is not the primary credibility benchmark: its [creator explicitly recommends newer data for more serious work](https://www.garrickorchard.com/datasets/n-mnist).

### Distinguish the quantities being scaled

Track unique recordings, repeated presentations, spike events, simulation ticks, parameter-update steps, task blocks, and independent paired runs separately. A million spikes from a few recordings are not a million independent observations.

Cost calibration uses training-only sample counts of 256, 1,024, and 4,096, stopping earlier on a resource limit. These are proposed throughput workloads, not statistical sample sizes. The principal real-data run uses the full eligible training split. A secondary multi-pass schedule may use 1, 3, and 10 presentations of that split, frozen on development evidence. SSC supplies a larger pool of distinct recordings; repeated SHD epochs do not become distinct data.

For the integrated engine, vary the declared growth budget after measuring feasibility. Compare initial, intermediate, and larger capacity budgets, with their exact values frozen before confirmation. Record the realized active-unit and synapse trajectory instead of assuming that reaching a larger allocation means useful growth.

## Sequential task construction

The proposed SHD scenario uses ten class blocks, with two new classes introduced per block. SSC uses seven blocks of five classes if its adapter is eligible. Register class grouping, order generation, exposure counts, and temporal preprocessing before the confirmatory cohort. The primary schedule is one pass through each block, with no reset of learned weights between blocks.

All methods receive the same examples in the same order within a paired run. Class-block order and model initialization vary across independent runs. Sample orders from a registered distribution, with a balanced order design where practical; repeating one stream with different initialization alone does not test order robustness. The scorer knows the task blocks; the learner does not receive a test-time task identity that gives away the class subset.

Use one declared global class vocabulary and the same output-head policy across methods. Any future-class mask or head expansion must be explicitly shared across the comparison. Reset transient neural activity between independent utterances according to the frozen protocol, while weights, structure, replay, and durable state continue. Test longer contextual carry in a separate temporal diagnostic with known dependencies rather than inventing a causal link between unrelated recordings.

After every block, evaluate held-out examples for earlier and current classes, producing a retention matrix. Any future-class evaluation is scorer-only and cannot affect updates, growth, hyperparameters, stopping, or task selection. Include a recurrent-class revisit schedule as a registered secondary condition, rather than choosing it after seeing favorable recovery.

## Representation and label boundaries

Do not pretrain a supervised encoder on all future benchmark classes and then describe the downstream study as acquiring those classes from scratch. The first component study starts the existing e-prop audio learner from its declared initialization and updates only from the available training stream.

For the full engine, decide before confirmation whether the sensory encoder learns online or stays frozen. Online encoder changes are part of the learning and cost account; readout co-adaptation and representation drift must be measured. A frozen encoder must have declared provenance. A separately pretrained representation benchmark is a valid narrower experiment if its pretraining data and costs are reported.

Cache raw recordings or outputs of a strictly frozen, shared preprocessing stage. Do not share cached reservoir/readout features across arms whose growth, feedback, or encoder changes make those features different. Labels enter teaching and the scorer, never the inference feature path.

## Comparison arms and fair budgets

The primary confirmatory comparison has two arms at one frozen dataset and resource budget:

1. The complete eligible SNDL configuration.
2. A fixed-capacity twin using the same input information, learning rule, replay allowance, and declared maximum capacity budget.

A smaller initial-size fixed twin provides a secondary width comparison. The larger fixed twin may use its capacity from the start, making it a strong comparator. If a separate actual-final-size matched twin is used, specify its construction in advance and derive its size from training/development information only. Never choose it using final-test outcomes.

Attribution arms are staged after the primary design is frozen, with their multiplicity and sample sizes specified:

| Arm | Question |
|---|---|
| Matched random growth | Does earned admission help beyond allocating the same number of units on the same growth schedule? |
| No offline updates | How much retention/recovery comes from replay or consolidation? Keep logging/checkpoint costs comparable. |
| Feed disabled | Does the grown-unit feed change task performance? |
| Return disabled | Does the readout return change later task behavior? |
| Reservoir removed/reset, transport disabled | Which particular reservoir/transport claim survives its own control? Use only well-defined eligible paths. |

The matched random-growth schedule is derived from the training trajectory of its paired full arm and must not inspect test results. Equalize its admissions and relevant search/learning budgets. A run with fewer resources is not silently promoted as a fair architectural comparison.

Add a simple numerical classifier and a conventional continual learner with replay as practical reference points. Give them equivalent temporal information, data, memory limits, and tuning effort. Their status as external benchmark controls must be explicit. A small transformer/RNN baseline is useful only when the exact input/history and budget comparison is fair; a generic transformer-versus-SNN headline is not an experimental result.

Run the two-arm primary first. Do not launch a full Cartesian product of datasets, widths, algorithms, and seeds on an 8 GB laptop. Ablations and SSC extension form separate registered work packages.

## Measurements and reported figures

Primary endpoint: **final held-out macro accuracy across the complete class vocabulary at the declared training/resource budget**, compared per independent paired run. Raw overall accuracy accompanies macro accuracy. The final checkpoint and budget follow the frozen rule rather than the best test score.

Secondary measurements describe the ongoing process:

| Question | Measurement and figure |
|---|---|
| How does learning develop? | Accuracy versus unique recordings, repeated exposures, update steps, and elapsed CPU/wall time. Predict-before-update accuracy on the training stream is clearly labeled and separate from held-out accuracy. |
| What is retained? | Retention matrix R[i,j], average forgetting, backward transfer, and old/new class scores over time. Include per-class confusion matrices. |
| Does additional structure help? | Active units, synapses, admissions, rejected candidates, and budget spending over time, aligned with held-out learning curves. Show quality/resource frontiers across registered budgets. |
| What changes during offline processing? | Scores immediately before/after, amount of replay and solve/search work, retained new-task performance, and cost. |
| What survives restart? | New-process prediction agreement, score difference, restored carrier/state checks, and restart latency. Separate episode-boundary restart from a later deliberately interrupted mid-trajectory test. |
| What does it cost? | End-to-end wall time, function CPU time, startup, data transformation, training, growth search, evaluation, persistence/rebuild, peak RSS, replay bytes, and lake growth. Report latency distributions rather than only the fastest query. |
| Where does it fail? | Errors by class/speaker, unsolved problems, budget pauses, failed controls, exceptions, and completion rate. |

Report prediction loss or calibration only for a defined probabilistic output. A raw delta readout score is not a calibrated probability. Do not turn spike counts, FLOP proxies, or CPU time into measured energy without an actual energy measurement.

These choices follow the scope of standard [continual-learning evaluation metrics](https://avalanche.continualai.org/from-zero-to-hero-tutorial/05_evaluation). No Avalanche installation is required merely to compute the defined metrics. The [online continual-learning comparison](https://arxiv.org/abs/2308.10328) also motivates including a properly implemented replay baseline and measuring behavior during the stream.

## Statistical design and independence

Allocate a registered development pilot of up to eight paired independent streams, within a four-hour pilot compute cap after runtime readiness. This is a variance/runtime estimation budget, not a powered scientific sample size or a source of confirmatory headlines. If it cannot estimate variability adequately within that budget, report that limitation before freezing confirmation.

Choose the practically relevant accuracy difference before viewing confirmatory outcomes. A provisional planning value is **3 percentage points**, subject to an explicit research justification and pilot-only revision. The measured paired variability determines the confirmatory run count. Use an upper uncertainty bound on pilot variability when sizing the run; zero pilot SD does not imply zero future variation or zero required runs.

Register a sequential design consistent with P1: the independent-run unit, separate development and confirmation cohorts, scheduled looks, significance spending, cap, effect/power calculation, and futility rule. A concrete conservative option is four planned looks with a total two-sided alpha of 0.05 allocated as 0.0125 per look. Derive the fixed target and inflated cap for at least 80% power at the chosen effect, and validate the complete stopping design in simulation before use. Report intervals valid for the planned sequential analysis, rather than ordinary unadjusted 95% intervals at a selected stopping look.

The sampling unit for method comparison is the independent paired run/stream. Thousands of examples within one stream, repeated epochs, different arms, and restart checks do not each count as independent model runs. Speaker and dataset generalization are separate sampling questions and must keep their own scope.

Classify the outcome using the interval and effect scale: benefit supported, harm supported, practically important gain ruled out at this scale, or inconclusive. Reaching a resource cap or a nonsignificant p-value alone does not prove no effect. Additional seeds after an unregistered look require a new exploratory reading, not a retroactive confirmatory claim. Secondary comparisons use a predeclared hierarchical or multiplicity-adjusted procedure.

This plan creates no new seed allocation and launches no historical frozen cohort. Scientific seeds must be reconciled and reserved before execution. P8 scale-retest rules govern mechanism disposition; this study alone does not authorize deleting a mechanism.

## Evaluation integrity and failure policy

Freeze code, dataset manifests, task order generation, encoders, budgets, and analysis before opening confirmation. Growth decisions may use training and development evidence only. An evaluation process receives a copied checkpoint and label-free inputs. Learning and replay insertion are disabled; transient episode state follows the frozen reset protocol. Evaluation cannot write back into the training state or lake.

Save every prediction and target in scorer-owned records, together with enough metadata for independent recomputation. Seal confirmatory test feedback from development/tuning until the registered analysis look. Retain failed and partial runs. Retry an infrastructure failure only under a predefined rule with the original run/seed identified; do not replace a technically valid low-scoring run.

An algorithm timeout, memory breach, or exhausted growth budget is part of operational performance. Report completion rate and completed-run quality together. State before the campaign whether the primary score uses the last valid checkpoint or a fixed failure penalty. Publish sensitivity to that choice; do not hide noncompletion by analyzing only successful runs.

## Logs and implementation deliverables

Reuse `taught_problems`/`taught_campaign`, `CanonicalGrowthAdapter`, `PosedActor`, the lake save/restore paths, and the existing paired-contrast machinery where their contracts apply. They do not eliminate the need for a numerical/event task adapter and a continual runner.

Proposed outputs under `planning/experiments/scale/` are a dataset manifest/split builder, a bounded runner, a restart worker, an independent analyzer, and a frozen configuration. These are planned implementation deliverables, not existing runnable commands.

The runner's planned record schema includes run ID, source SHA, library versions, dataset/split hashes, seed and task-order ID, arm, sample/exposure/update counters, prediction/target records, train/development/test role, resource limits, failures, and all checkpoint receipts. Engine telemetry should use actual fields such as `ProblemRecord.state`, `budget`, `spend_units`, `spend_evals`, admitted unit IDs, feed/return call counts, and saved state hashes. New schema fields must be implemented and tested before they are called measured data.

The analyzer produces learning curves with individual runs and valid uncertainty, a retention matrix, growth/resource trajectories, a per-run table, and a failure ledger. Figures read saved records. No hand-written smooth curves or replacement performance values enter the report.

## Mac compute envelope and scheduling

Start with CPU, one experiment worker, and zero DataLoader subprocess workers. The reviewed audio CLI supports CPU/CUDA; MPS is not assumed to work. Start with a small batch, then choose the final batch using measured memory. Use a provisional worker RSS cap of 3 GB, a 15-minute calibration timeout, and a disk budget checked against actual dataset sizes and available space. These are proposed operational limits, not claims of achieved throughput.

Measure a representative complete trajectory before committing to an overnight block. Include feature extraction, learning, evaluation, and persistence, not just a cached inner update loop. Use the median and a slow-run percentile to estimate the matrix with headroom. With one worker:

`total wall time = data/runtime preparation + sum(arm × independent run × measured trajectory time) + analysis`.

The following examples illustrate arithmetic only. Fifty runs is not the selected scientific sample size.

| Measured trajectory time | Two arms × 50 runs | Six arms × 50 runs |
|---:|---:|---:|
| 1 minute | 1 h 40 min | 5 h |
| 10 minutes | 16 h 40 min | 50 h |
| 60 minutes | 100 h | 300 h |

Use an initial four-hour development block, then checkpointed blocks of up to eight hours if the measured workload fits. If the powered primary comparison is too expensive, reduce secondary arms or the declared workload before confirmation, or report the available study as exploratory. Do not preserve the original claim by quietly reducing required independent runs.

For a showcase in a few days, aim first for a verified real-data pilot with full learning/resource curves, clear failures, and a precise engine-readiness status. A complete, powered full-engine study cannot be promised until integration and measured throughput make its schedule credible.

## First execution sequence

1. Repair the stalled Torch import and repeat the bounded runtime check. Verify versions, CPU operation, and free-threading after imports.
2. Validate the official SHD download and split manifest; create an independent development split and keep official test data sealed.
3. Implement the bounded continual component runner around the existing e-prop audio path. Validate scoring, label boundaries, checkpoint reconstruction, and planted failures.
4. Profile 256, 1,024, and 4,096 training examples under the Mac limits. Freeze a feasible development-pilot workload from those timings.
5. Run the registered pilot and produce actual learning, forgetting, confusion, and cost plots. Use that evidence to freeze the confirmatory design and to prioritize the full-engine integration work.

The user's conditional authorization is to run locally if the system works normally on the Mac. Current runtime readiness has not passed. There is no request here to spend money, acquire a server, change a safety envelope, or publish a scientific success claim.

## Execution record — November 4, 2026

The authorized local component pilot is complete: Torch imports/tensors verified with the GIL off; official SHD training data prepared with exact spike-count conservation and a speaker-disjoint development split; 16 runs over four exposed development seeds completed. All 144 prediction checkpoints were independently rescored. One infrastructure interruption was retained; its retry reproduced all nine shared checkpoints bit for bit. Final mean macro accuracies were 41.16% for joint IID, 6.22% sequential, 7.21% repeated-current updates, and 12.68% replay. Replay versus its update-matched control differed by +5.47 pp on average across four paired seeds, an exploratory component result. The official test remained sealed.

The full engine remains blocked at G9_A_UNREADY. See the [execution ledger](experiments/scale/LEDGER.md), [report and figures](experiments/scale/reports/pilot-report.html), [canonical results](experiments/scale/reports/pilot-results.json), [deviation record](experiments/scale/DEVIATIONS.md), and [reproduction instructions](experiments/scale/RUNBOOK.md). The canonical analysis corrects a secondary forgetting-summary clipping error; primary accuracy measurements are unaffected.
