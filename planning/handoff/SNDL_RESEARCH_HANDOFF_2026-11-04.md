# SNDL Research Status and Decision Handoff

Prepared November 4, 2026, America/Chicago, for the project team and a Claude-assisted technical review.

**Current assessment:** We have an integrated prototype that completes a constrained growth, retention, and restart cycle. We do **not** yet have evidence of effective continual learning on a real dataset, a performance advantage from growth, or recursive self-improvement. The immediate research decision is whether to diagnose the underlying learner before investing in a larger full-Engine experiment. The recommendation is to diagnose it first.

This handoff includes both successful engineering checks and negative research outcomes. The accompanying packet contains the actual protocols, scripts, machine-readable results, code, and local changes. Archive member names below identify evidence within that packet; they are not public download links.

# Project goal and showcase context

Spiking Neural Data Lake, or SNDL, investigates a neural system whose learned state and active structure can develop with experience and survive process restarts. The research question is whether local learning, structural growth, feedback, retention mechanisms, and durable state can support continued acquisition of capabilities without destroying earlier ones.

The intended architecture connects a spiking substrate and readout with newly admitted neural units, feedback paths, bounded search, retention checks, and a persistent event/state lake. The size of one configured reservoir is an experimental setting, not the central research contribution. Growth of active structure and adaptation of learned state are the intended focus.

Distinguish three different claims:

1. **Engineering integration:** The implemented learning, growth, feedback, retention, and persistence paths execute together and reconstruct their state correctly.
2. **Continual-learning efficacy:** The system learns useful new tasks, retains old performance, and compares favorably with appropriate controls on genuinely held-out data.
3. **Recursive self-improvement:** Changes improve the process that produces subsequent improvements. Adding units, repeating training, and restoring a checkpoint do not establish this claim by themselves.

Current evidence supports a bounded instance of the first claim. It reveals substantial problems for the second. The third remains a long-term objective requiring its own operational definition and experiment.

The showcase is in a few days. The presentation is approximately ten minutes, for a technical audience that may know little about this topic or AI. The requested flow is: title and hook; background and problems; how the Engine works; objectives; comparison with Transformer-based ANNs and SNNs; concrete data consumed and produced; methods; results and discussion; shortcomings and future goals.

The team explicitly wants real measurements, disclosed denominators, and no invented data or exaggerated success story. Avoid the blanket claim that other ANNs or SNNs only produce snapshots or cannot adapt. Any novelty or current-literature comparison needs specific primary-source support and should remain separate from our experimental results.

# Two experiments with different scopes

| Evidence stream | What actually ran | What it can answer |
| --- | --- | --- |
| SHD real-data pilot | Fixed-width event-audio SNN component using the existing e-prop learner | Whether this component learns from the chosen representation and how sequential training, current-data repetition, and replay affect development accuracy |
| Full Engine integration | Actual persisted Brain on two very small registered text tasks, including growth, protected-core retention, feedback, and fresh-process recovery | Whether these implemented paths can work together under the disclosed engineering controls |

**There is no full-Engine SHD experiment in the current evidence.** Do not join these two streams into a fictional end-to-end run. The full Engine's tiny-task accuracy cannot compensate for the weak real-data component results.

# SHD real-data component pilot

## Data and input representation

The pilot used the official SHD training archive, divided into training and development by speaker. The official test set was not used.

| Item | Recorded value |
| --- | --- |
| Total recordings | 8,156 |
| Training recordings | 6,369 |
| Development recordings | 1,787 |
| Held-out development speakers | IDs 10 and 11 |
| Output classes | 20 |
| Original spike events | 63,993,588 |
| Network input per recording | 100 time bins by 700 channels, spike counts converted to float32 |
| Hidden width | 64, fixed throughout the experiment |

Each recording's own observed minimum-to-maximum spike-time interval was mapped into 100 equal-width bins. All events were conserved; the largest stored bin count was six. This removes absolute duration and is not claimed equivalent to every published SHD preprocessing method. Its effect on learning remains a diagnostic question.

Labels provided training feedback and scoring targets; speaker IDs determined the split and supported auditing. Neither was an inference input feature. Evaluation used all 20 outputs without a task-identity mask. All classes were present in both splits. The publisher MD5, archive SHA256, array hashes, split IDs, and event-conservation checks are retained. [S1, S2]

## Model and experimental protocol

The runner imported the existing `EpropAudioNet`, `eprop_grads`, and shared `_apply_grads`; it did not replace them with a separately implemented learner. CPU execution used one numerical thread. Recorded settings were Adam learning rate 0.002, batch size 16, alpha 0.9, threshold 0.6, gamma 0.5, damp 0.3, and kappa 0.9. The source and runner hashes are preserved. [S1, S3]

Four already-exposed development seeds, 8000 through 8003, were used for each of four arms. Within a seed, the arms shared the initial weights and the incremental class order. The 20 classes were shuffled and divided into ten two-class blocks.

| Arm | Training schedule | Presentations per seed | Updates per seed |
| --- | --- | ---: | ---: |
| Joint | Two shuffled passes over all training classes | 12,738 | 798 |
| Sequential | One update per current batch, one pass through the class blocks | 6,369 | 403 |
| Repeat | Current update followed by a second update on the same batch | 12,738 | 806 |
| Replay | Current update followed by an equally sized update sampled from earlier training examples | 12,738 | 806 |

Replay used a label-blind reservoir holding up to 400 previous training recordings. The reservoir was updated after the two updates, preventing access to future-stream data. The first batch repeated current examples because no history existed. Each replay run had 6,353 historical-example presentations and 16 repeated-current presentations in addition to its 6,369 current presentations.

Replay versus repeat is the most direct control for the benefit of historical examples at matched updates and batch sizes. Sequential has half as many presentations. Joint matches aggregate presentations, but differs in data order and update count because its batching differs. It is a limited learnability reference, not a converged upper bound or a competitive benchmark.

The prospective apparatus gate was development macro accuracy above 10% for the first two-pass joint run. That threshold meant the apparatus learned enough above the 5% uniform-chance expectation to continue a feasibility pilot. It was never a criterion for useful continual learning. [S1]

## Final accuracy results

The primary figures below are **macro accuracy**, meaning the average of the 20 per-class accuracies. The plus/minus term is the sample standard deviation across four seeds, in percentage points. It is not a confidence interval. Overall example-weighted accuracy is a separate quantity in the machine-readable report.

| Arm | Final macro accuracy | Mean macro accuracy on the old 18 classes | Mean macro accuracy on the latest 2 classes |
| --- | ---: | ---: | ---: |
| Joint | 41.16% ± 3.20 pp | Not applicable | Not applicable |
| Sequential | 6.22% ± 1.83 pp | 2.48% | 39.90% |
| Repeat | 7.21% ± 2.30 pp | 0.00% | 72.09% |
| Replay | 12.68% ± 2.57 pp | 6.03% | 72.50% |

The old/latest grouping follows each seed's shuffled class order. All predictions still choose among the full 20 classes.

| Seed | Joint | Sequential | Repeat | Replay | Replay minus repeat |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8000 | 38.09% | 5.00% | 5.73% | 11.58% | +5.85 pp |
| 8001 | 45.50% | 5.25% | 8.44% | 13.86% | +5.42 pp |
| 8002 | 41.44% | 8.93% | 4.87% | 9.70% | +4.83 pp |
| 8003 | 39.62% | 5.71% | 9.80% | 15.58% | +5.78 pp |

The mean paired replay-minus-repeat difference was **+5.4694 pp**, with between-seed SD **0.4635 pp**. All four observed differences were positive. These are descriptive results from four exposed seeds on one speaker split, not an independent confirmatory study. [S3]

## Interpretation and the forgetting metric

The real-data accuracy is too low to call this an effective continual learner. Sequential performance is near the uniform-chance expectation. Replay improves final accuracy over matched current-data repetition, but the old/latest split shows that most useful performance remains concentrated on the latest classes.

The registered mean forgetting metric, calculated from each earlier task's best earlier accuracy minus its final accuracy, was **78.4206 pp for replay** and **73.7808 pp for repeat**. Consequently, **the experiment does not support the general statement that replay reduced measured forgetting**. Replay improved final macro accuracy and final old-class accuracy relative to repeat; it did not improve this forgetting endpoint. Different earlier maxima can make these measures move in different directions.

Independent analysis found and corrected a secondary summary error: the original collector included the final checkpoint when finding the best earlier score, clipping later improvements. The final analyzer recomputed the registered metric from the raw task matrices and allowed negative values for improvement. Original collector records remain intact. Classification accuracy, model training, and the paired accuracy comparisons were unaffected. [S3, S4]

Joint accuracy of 41.16% after only two passes establishes some learning, but leaves basic learnability and convergence unresolved. The present evidence does not identify whether the main limitation is training duration, representation, optimization, implementation behavior, capacity, or a combination. These are hypotheses to test, not findings to assert.

## Reliability and operational history

There were 16 completed runs and 17 initialized training attempts. One sequential seed-8003 attempt was interrupted during storage pressure and apparent temporary-cache loss; the exact reset cause was not established. The interrupted attempt was retained rather than silently discarded. Recovery rebuilt the same data and imported source files against their original hashes, then restarted the same seed. All nine shared prediction checkpoints were bitwise equal between the interrupted attempt and retry.

Independent analysis rescored 144 completed-run prediction checkpoints. Evaluation checked that weights were unchanged, recomputed predictions from saved logits, checked sample identities, and verified paired initial states, streams, and exposure accounting. The completed cohort contained 178,332 training presentations; its summed training-run wall time was about 34.1 minutes. This excludes preparation, debugging, interruptions, source verification, and recovery, and is not an end-to-end project runtime. [S3, S5]

# Full Engine integration

## What initially blocked the cycle

The canonical `GrowthLoop` initially refused the relevant configuration with `G9_A_UNREADY`. Existing manually admitted-population tests did not establish that automatic candidate growth and the actual final Brain learner could work together. The Part D reader and canonical chat learner were different consumers, and capacity diagnosis did not account for all grown/return context.

The full runtime also required dependencies and the compiled native extension absent from the audio-only environment. A separate complete free-threaded runtime was recovered outside the source checkout.

Further integration work addressed writer-bound callbacks copied into a lock during clone diagnosis, durable selection and final-data consumption, candidate-source binding and accounting, and exact continuation after restart. A real plasticity initialization defect was repaired so existing Benna–Fusi/homeostatic baselines were captured before the first update. That fix alone did not solve stable learning of the initial task. [E1, E2, E3]

## Successful configuration and its conditions

The integrated run used the actual persisted `make_brain(9900)`, common event writer, candidate lifecycle, G7 admission, grown population, Part D/feed/return path, canonical final learner, and checkpoint/journal recovery.

The curriculum was registered task `aii` followed by `ii.xor`, with explicit task identity selecting the posed context. Old-task words were `ram`, `cup`, `emu`, and `cod`; new-task words were `cat`, `dog`, `sun`, and `sky`. Each task used four phrase-order variants, not four isolated words. A's target depended on whether `ram` preceded `cup`. B was the following word-order XOR fixture:

| B input text | Target intent |
| --- | --- |
| `cat dog sun sky` | `GREET` |
| `cat dog sky sun` | `GOAL` |
| `dog cat sun sky` | `GOAL` |
| `dog cat sky sun` | `GREET` |

**There were only four unique constructed inputs per task.** Each 16-presentation evaluation repeated those four inputs four times. The final or future set changed their order, not their semantic content. These are controlled order-discrimination tasks, not demonstrations of natural-language semantic understanding. The actual canonical output vocabulary contains nine intents, although these tasks use only `GREET` and `GOAL` as targets.

Two explicit engineering controls were essential:

- **Corrective feedback:** The shared learner skipped a reward-driven readout update when its actual answer was already correct. The common event and other required execution paths still ran.
- **Protected old core:** During B, the learned core and its full plasticity-stack state were frozen; newly grown output connections remained plastic. Preserving a frozen core is a legitimate control, but does not solve retention while that core continues learning.

The selection recipe was all-active/no-prune. The successful run used one already-exposed engineering seed, 9900. Protocol amendments responding to observed engineering failures are dated and retained; this was not a fresh, preregistered scientific efficacy cohort. [E1, E2]

## Measured outcomes and interventions

| Event | Recorded outcome |
| --- | --- |
| Acquire old task A | 22 epochs, 88 teaching presentations, 66 base-stack deposits; final 10 epoch ends all correct |
| Grow for B | 1,024 candidate evaluations; one admitted batch of 18 units, IDs 0–17 |
| Acquire B | 107 epochs, 428 teaching presentations; epoch ends 98–107 all correct |
| Validation and once-only final scores | A 16/16 and B 16/16, each based on four unique inputs repeated four times |
| Old core during B | Entire saved canonical neuron and stack state exactly equal to A endpoint; zero additional core deposits |
| Grown output learning | 36 nonzero final output weights |
| Silence grown units | B falls to 8/16; A stays 16/16 |
| Zero actual return conductances | B falls to 8/16 |
| Zero actual feed conductances | B stays 16/16; internal pool features change, L2 difference 32.6803 |
| Damage old-task feature weights | A falls to 0/16 while B stays unchanged; retention feasibility rejects the damaged state and full state is restored |
| Fresh-process checkpoint recovery | Exact model digest, actor carriers, and predictions recovered without teaching or regrowth |
| Journal-only recovery | Exact continuation; boot does not re-read held-out sets |
| Attempt final judgment again | Explicit refusal, `G9_RETENTION_ALREADY_CONSUMED` |

The 18-unit batch size and 1,024-candidate pool are disclosed engineering settings. This run does not show that the system discovered an optimal size. Unit and return interventions establish that this trained fixture depends on those components. They do not establish that growing outperforms an equally sized network initialized with the same capacity. Feed changed internal features but showed no answer-accuracy benefit in this fixture. [E4, E5]

The selection and both final-data consumption claims were durably recorded before reading final data. New adapters, process restarts, and rollback cannot re-arm them. An independent read-only audit checked the raw journal and rescored all 129 epoch-end rows, source accounting, validation/final answers, and interventions. It did not rerun training. [E4]

## Negative results that must remain visible

The genuine full-Engine all-plastic control acquired B at 16/16 but left old A at **0/16**. Retention selection correctly rejected the state before final-data consumption. This is direct evidence that the current all-plastic configuration did not preserve the old skill.

Other retained development diagnostics include unsuccessful always-positive acquisition of A after 1,200 epochs, initialization repairs that were insufficient on their own, and a rehearsal control that failed to produce joint stability. Brain-free candidate and policy probes are development diagnostics and must not be pooled with the actual full-Brain trajectory.

One checkpoint verification initially failed because the controller compared tuples to JSON lists. Serialized predictions were identical; the comparator was corrected and a new process passed. Neural values were not adjusted to repair that check. Earlier failed controllers and source overlays are retained. [E1, E2]

## Implementation state and verification

| Item | Current recorded state |
| --- | --- |
| Repository | `https://github.com/PVLuanDinh/Spiking-Neural-Data-Lake.git` |
| Base revision for this work | `c4c10b285f1741aa798d5ee9c5296ae371fabecf` |
| Implementation branch | `engine-cycle-integration` |
| Implementation commit | `26501ed3` |
| Source working tree at handoff check | Clean |
| Focused tests | 215 passed: 120 pure/growth-related plus 95 Brain/protocol-related |
| Additional checks | Five bounded module selfchecks, pinned Ruff 0.15.20 checks, numerical-source equivalence after declaration-only metadata changes |
| Push or main merge | Neither performed |
| Full operator battery | Not performed |

The commit message explicitly says the change is not verified by a battery. Focused checks support implementation reliability in their tested scope; they are not evidence of research efficacy or a broader release. The implementation commit is local and should not be assumed fetchable from GitHub. The packet includes the local commits and a patch. [E6, E7]

# Concrete data and code orientation

These are two separate pipelines. Use each one's own recorded variables when explaining it.

| Pipeline stage | Actual inputs or state | Actual outputs or changes |
| --- | --- | --- |
| SHD preparation | Spike times and channel IDs; split/scoring labels and speaker metadata | Count frames with shape 100 by 700, sample/split IDs, hashes, event statistics |
| SHD learning and scoring | Float32 frames; model weights; training targets only during updates | 20 logits, predicted class, hidden activity, losses, updated weights, saved per-example predictions and confusion/task matrices |
| Engine task interface | Registered problem identity; text/role episode inputs; target as teaching/scoring metadata | Actual canonical answer, diagnosis, per-epoch outcomes, task ledger state |
| Engine growth | Current consumer state; candidate recipe and stream; budget and admission facts | Admitted `GrownUnit` state including `unit_id`, `w_in`, `threshold`, `window`, `sign`, `out`, and `active`; actual births and costs |
| Engine continuation | Canonical neuron/stack state, actor carriers, selected state, journal claims and lake | Checkpoint or journal reconstruction, matching predictions/digest, retained final-consumption guards |

For exact field ownership, inspect the source and raw receipts rather than synthesizing a single universal vector or an invented RSI score. Labels must not become inference features. Readout scores are not automatically calibrated probabilities.

For the integrated cycle, the common event contains `op` (`engine_teach` or `engine_read`), `problem_id`, `turns`, `shifted_label`, `neuron_mask`, `return_mask`, and `return_neutered`. Each recorded turn has the form `[text, "probe", target]`; a learning-OFF read sets the target to `None`. The actual final answer is the canonical `ixn.route_text(text)` intent string. Grown input parameters are fixed at admission while `out` can remain locally plastic. Shared context includes `units`, `afferents`, `t_post_g`, `out_eligible`, and optional `core_eligible`. Durable actor state contains the factory binding, units, feed, reader, return port, clock/event carriers, selection, and retention claims. These are concrete implementation fields, not a proposed future schema.

The eight modified production files are identified in `completion.json`: `engine/brain.py`, `growth/growth_loop.py`, `growth/capacity_certificate.py`, `instruments/joint_mask_search.py`, `readout/posed_problem.py`, `readout/interaction_learning.py`, `plasticity/homeostatic_plasticity_stack.py`, and `plasticity/plasticity_rule_stack.py`, under the source snapshot's `src/snn_data_lake` package.

Additional entry points are `sensorimotor/eprop_shd.py`, `plasticity/eprop_kernel.py`, `growth/grown_code.py`, the six new focused test files listed by the patch, and the experiment scripts in the evidence directories.

# Open questions and current blockers

| Question or blocker | Current evidence | Consequence |
| --- | --- | --- |
| Can the underlying SHD component learn the representation well? | Two-pass joint reference is 41.16%; convergence and tiny-set fitting are unresolved | Diagnose basic learning before attributing failure entirely to continual learning |
| Can an adapting shared core retain old skills? | All-plastic Engine control forgot A completely; protected-core control passed | Protected retention must be disclosed; general plastic-core retention is open |
| Does structural growth improve performance or efficiency? | Current fixture depends on grown units, but no capacity-matched efficacy comparison was run | Need a pre-sized equal-capacity control and charged search/training costs |
| Can the full Engine process SHD end to end? | Not tested | Requires a real adapter and an integrated experimental protocol |
| Can it continue through further costly growth tasks? | Source-owned BETA for a later positive-spend task remains unset | The two-task cycle does not release the later budget gate; calibration is still required |
| Does feedback help accuracy generally? | Return matters on the fixture; feed has no observed accuracy effect there | Do not attribute a general benefit to both channels |
| Does the system work without supplied task identity? | Current Engine context is explicitly task-conditioned | Uncued composition/context selection remains open |
| Are pruning, sleep/consolidation, or RSI effective? | No deciding experiment in the present evidence | Keep these as future goals |

The broader enabled-stack scientific repair/release issue remains open. The successful controls narrowed a workable configuration; they did not establish that every original configuration now learns reliably.

# Recommended next research sequence

These experiments are proposed, not executed. The sequence is intended to identify the dominant failure before a larger campaign.

1. **Audit and fit a small fixed SHD subset.** Freeze balanced sample IDs and an explicit stopping budget before training. Record training loss and accuracy, class predictions, spike activity, update norms, and nonfinite checks. Establish whether the learner can fit a small sample at all. High training accuracy here is an implementation diagnostic, not a showcase efficacy result. Agree a numeric criterion and its justification prospectively rather than choosing a threshold after viewing the curve.
2. **Measure joint-training convergence.** The current reference has only two passes. Extend a declared training budget while recording both training and development curves. Check the same input/label alignment and evaluation invariance. Distinguish failure to fit training data from failure to generalize to held-out speakers. A low development score alone does not identify the cause.
3. **Run a simple reference learner on the identical split and input representation.** Use the same access to labels, development data, and preprocessing. A simple linear or small neural reference can test how much information is recoverable from the representation. A BPTT-trained spiking reference could separately diagnose the local learning rule if resources allow. These references are proposed diagnostics, not existing benchmark scores.
4. **Investigate incremental collapse.** Reuse saved logits and task matrices to examine prediction concentration, old/latest classes, per-speaker errors, loss of earlier task performance, and reservoir class/age coverage. Avoid tuning on the official test set. Any newly tuned configuration remains development work.
5. **Only then design the integrated growth comparison.** Include growth-on, growth-off, and an equally sized pre-initialized network; match or explicitly account for training examples, updates, historical data access, search cost, task-context information, feedback gating, and protected-core eligibility. Separate the protected-core and all-plastic questions. Include held-out inputs that are actually new, multiple task sequences, and prospectively chosen seeds.

Keep acquisition, final old-task accuracy, backward transfer/forgetting, total capacity, candidate evaluations, presentations, wall time, and memory as separate reported quantities. Do not compress them into an invented improvement score. Freeze the analysis and a substantively justified primary decision criterion before a confirmatory cohort; use these pilot results to plan that cohort, not to portray it as already completed.

# Decisions for the team

| Option | What we can deliver honestly | Main tradeoff |
| --- | --- | --- |
| Present current evidence | Integrated prototype, constrained recovery/growth demonstration, and negative real-data findings | Credible work in progress, with no claim of successful scalable continual learning |
| Add bounded learning diagnostics before the showcase | Same story plus a clearer diagnosis of the weak SHD result | Best immediate research value if time permits; may still not yield a stronger efficacy result |
| Start a large full-Engine real-data campaign now | Additional integration work and exploratory measurements | High risk of spending the remaining time measuring a learner whose basic limitations are not understood |

Recommended choice: prepare the honest work-in-progress showcase in parallel with bounded component diagnostics. Do not make completion of a major new architecture campaign a prerequisite for having a presentable talk. If the showcase requires a positive efficacy result, the present evidence does not meet that requirement; the team should narrow the claim or reconsider the submission framing.

Questions to resolve together are the primary near-term contribution, the next experiment's measurable decision criterion, whether protected-core retention is an acceptable initial research direction, and the compute/time budget available before the talk. These are team choices, not results that have already been obtained.

# Presentation implications

The existing beta is **SNDL Research Showcase Beta**, at `https://chatgpt.com/space/page_25fc2658473c819194876bbbe931bd18`. It contains 11 slides and predates the SHD pilot and complete Engine cycle. Some slides still say full-cycle integration/recovery is untested, while older tiny component examples show perfect recovery. That combination is outdated and can mislead unless revised.

The deck has not been changed as part of this handoff. After the team chooses its framing, revise it to distinguish aspiration, implemented mechanism, engineering evidence, real-data performance, and next experiments. Show the SHD seed variation and old/latest-class gap. Describe the full Engine task as four unique inputs per task, rather than a headline 100% result. Include the all-plastic failure and explain why the frozen-core control matters.

A defensible central message is:

> SNDL now supports a constrained integrated cycle of growth, protected retention, and exact restart. Real-data component testing exposes severe forgetting and limited accuracy. The next research step is to establish strong basic learning and then measure whether growth improves continual learning under matched controls.

The concrete data slide must use a real trace from the selected pipeline. Conclusions should state the shortcomings and future goals. Literature comparisons should describe specific mechanisms and evidence rather than imply that all competing systems are static or that SNDL has already demonstrated RSI.

# Evidence packet and access

The local workspace is `/Users/arthurkhvan/Documents/ChatGPT/Spiking-Neural-Data-Lake 2`; the Git checkout is its `repo` subdirectory. Planning, scripts, and local results outside `repo` are not automatically present in a teammate's Git clone. A paid Claude subscription does not give access to this machine, private Pages, unpublished commits, or the persistent lake. Transfer the packet explicitly.

The packet includes:

- This report, a short starting prompt, compact machine-readable context, per-seed metrics, and a SHA256 manifest.
- The exact tracked source snapshot at the implementation commit, excluding the `research` literature directory to avoid bundling third-party papers. The source snapshot includes repository instructions, policy/cold-start documents, tests, native build code, and specifications.
- An incremental Git bundle of the local implementation commits and a binary-capable patch against the base revision. These are alternatives to an external push.
- The SHD protocols, scripts, analysis, completed/interrupted prediction records, small scoring metadata, and reports. The large raw/prepared dataset is excluded and must be restored or downloaded for training.
- The Engine experiment scripts, exact phase source overlays, result receipts, failed controls, dependency lock, and verification records. The persistent runtime, installed dependencies, compiled wheel, and actual lake are excluded; result receipts alone do not permit replaying the original lake.
- The original project proposal and earlier presentation planning notes as historical design context. Older statements about integration status are superseded by the current evidence, not silently treated as current results.

`MANIFEST.json` identifies every included file and its SHA256. The compact context does not replace the raw evidence. For a browser-only Claude session, extract the packet and upload the report, starting prompt, compact context, and selected evidence documents first. A Claude coding environment with filesystem access can inspect the unpacked code and records directly. Do not assume an uploaded ZIP will be expanded or every file will fit into one context window.

# Evidence index

The prefixes below are archive member names. `evidence` preserves local experiment directory contents; `source/repo` is the tracked source snapshot.

| Key | Packet member or members | Purpose |
| --- | --- | --- |
| S1 | `evidence/scale/PROTOCOL.md` | Prospective SHD pilot scope, preprocessing, arms, gate, endpoints |
| S2 | `evidence/scale/reports/data-manifest.json` | Actual data counts, split, events, hashes |
| S3 | `evidence/scale/reports/pilot-results.json` and `pilot-report.html` | Canonical final metrics, per-seed results, diagnostics |
| S4 | `evidence/scale/DEVIATIONS.md` and `analyze.py` | Operational deviations and corrected forgetting analysis |
| S5 | `evidence/scale/reports/closing-verification.json` and retry/recovery audits | Independent rescoring and recovery checks |
| E1 | `source/repo/specs/022-unified-brain-arc/evidence/g9-engine-cycle-integration.md` | Integration protocol, execution record, dated amendments, limitations |
| E2 | `evidence/engine-integration/LEDGER.md` | Actual implementation sequence and negative controls |
| E3 | `evidence/engine-integration/RUNTIME.md` and `core-lock.json` | Full runtime requirements and isolated execution |
| E4 | `evidence/engine-integration/cycle-04-independent-audit.json` | Independent rescoring, raw-journal and source-accounting checks |
| E5 | `evidence/engine-integration/cycle-04-old.json`, `cycle-04-grow.json`, and restore receipts | Actual states, outcomes, interventions, continuation |
| E6 | `evidence/engine-integration/completion.json` and `final-source-equivalence.json` | Commit, focused checks, source fingerprints |
| E7 | `evidence/engine-integration/final-pure-regression.*` and `final-brain-regression.*` | Focused test receipts |

# Operational notes for continuing implementation

Read the source snapshot's `AGENTS.md`, `docs/AGENT_COLD_START.md`, `docs/POLICIES.md`, and `SECURITY.md` before executing Brain code. The repository intentionally requires a free-threaded CPython build and a compiled native extension. This work used CPython 3.14.6t with the GIL off, Torch 2.15.0 CPU, NumPy 2.5.4, and an Apple Silicon native build. Ordinary Python 3.12 can perform standard-library analysis but is not a substitute runtime for importing the Brain.

The local `run-engine.sh` defaults to the immutable baseline source, **not the repaired working source**. Modified-code runs require an explicit `ENGINE_SOURCE_ROOT` pointing at the verified working snapshot. The audio `runtime.sh` is a partial source environment and must not be used for the full Brain. Both wrappers contain host-specific paths and are not portable setup scripts.

Use isolated homes, one Brain worker at a time, bounded parent/child deadlines, and the documented test working directory. The main checkout lacks the external compiled extension used for acceptance. Do not overwrite original run IDs, receipts, source overlays, or once-consumed final datasets. A repeat of exposed seed 9900 is a reproduction, not a new independent cohort.

No live home, physical-arm path, main merge, push, or full operator battery was part of this work. Continue within the selected experiment's scope. Preserve negative outcomes and distinguish a broader release check from the focused tests already completed. Git branches must have meaningful names without `codex`, `codex/`, or `codex-` prefixes, per the owner's permanent preference.
