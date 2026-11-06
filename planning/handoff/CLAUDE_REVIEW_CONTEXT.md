# SNDL Review Context for Claude

This is a single text upload for an evidence-grounded review of SNDL as of November 4, 2026. It combines the review request, full English report, compact context, and selected primary evidence. Read the request first, then use the evidence to challenge the report. The full ZIP also contains the actual source, local Git commits, phase receipts, negative controls, and prediction files.

This file is a review subset, not a runnable source checkout or an original persistent lake. Packaging descriptions in the report refer to the full SNDL_CONTEXT_PACKET_2026-11-04.zip. Repeated values in the context and evidence are the same measurements, not additional experiments. Proposed future tests have not been run.



# Starting request

Original workspace member: `planning/handoff/CLAUDE_START_HERE.md`

# SNDL Starting Context for Claude

We are preparing a research showcase in a few days: approximately ten minutes for a technical audience unfamiliar with our specific topic and possibly AI. Please help us evaluate the current research state and choose the next experiments. We need a critical technical collaborator, not a marketing rewrite.

Read `SNDL_RESEARCH_HANDOFF_2026-11-04.md` first, then `CONTEXT.json`. These names refer to files in the supplied packet. If you cannot access a file, say so and distinguish the report's description from evidence you have actually inspected.

Our current position is that a constrained integrated Engine prototype works, but effective real-data continual learning, the benefit of growth, and recursive self-improvement have not been demonstrated.

Two evidence streams must remain separate:

- Fixed-width SHD audio component: joint macro accuracy 41.16%; sequential 6.22%; repeat 7.21%; replay 12.68%, over four already-exposed development seeds. The official test set was untouched. Replay improved final accuracy over matched repeat, but the registered forgetting metric was worse. Joint training was only two passes.
- Full persisted Brain: a task-conditioned growth/retention/restart cycle passed on four unique inputs per task, repeated during evaluation. Success required corrective feedback and a frozen old core while grown outputs learned. The all-plastic control learned B but forgot A completely. This was not SHD, unseen-input generalization, or a capacity-matched growth comparison.

Please produce the following, in this order:

1. Audit the main conclusions against the supplied records. Identify unsupported claims, methodological confounds, measurement errors, and important missing evidence. Separate findings from hypotheses.
2. Give a short independent assessment of what contribution is defensible now and whether the work should be framed primarily as an engineering prototype, a continual-learning method, or an architecture proposal.
3. Rank the plausible causes of weak SHD performance as hypotheses, with the cheapest discriminating test for each. Inspect the existing learner/preprocessing/analysis code if available. Do not infer a cause from low accuracy alone.
4. Propose the smallest useful next experiment, with fixed sample IDs/splits, controls, metrics, a justified prospective decision criterion, an initial profiling step, and a bounded compute budget suitable for an M2 Mac with 8 GB RAM. Start by testing basic learnability rather than assuming a larger Engine campaign is the right next move.
5. Define a later fair growth comparison: growth-on/off, equal-capacity pre-sized model, protected versus plastic core, task-context access, replay access, feedback, search cost, training exposures, and genuinely held-out examples.
6. Recommend an honest ten-minute showcase story and explain what additional evidence would change that recommendation.

Treat a dataset/model/runtime change as a new development configuration; retain the original results. Do not cherry-pick seeds/checkpoints or tune on the official test set. A repeated exposed seed is not independent confirmation. Passing a tiny engineering fixture is not a benchmark result. Do not invent variables, an RSI score, calibrated probabilities, or measurements.

The source commit is local, `26501ed3` on `engine-cycle-integration`, based on `c4c10b285f1741aa798d5ee9c5296ae371fabecf`. The packet includes the source snapshot, incremental Git bundle, patch, and experiment records; a GitHub clone alone may not contain these commits or local evidence. Consult `PACKET_README.md` for access and recovery details.

Before executing or changing Brain code, read the repository instructions and environment notes. Use a new isolated experimental home and new run IDs; preserve the original lake and final-consumption records. Branch names must not use a `codex` prefix. Do not edit the presentation, publish, merge, or push merely because you have reviewed this packet; first return the assessment and proposed next experiment for the team to choose its direction.


# Research status and decisions

Original workspace member: `planning/handoff/SNDL_RESEARCH_HANDOFF_2026-11-04.md`

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


# Compact machine readable context

Original workspace member: `planning/handoff/CONTEXT.json`

```json
{
  "schema": "sndl.research.handoff.v1",
  "as_of_date": "2026-11-04",
  "timezone": "America/Chicago",
  "purpose": "Evidence-grounded teammate and Claude review before further research decisions",
  "source": {
    "origin": "https://github.com/PVLuanDinh/Spiking-Neural-Data-Lake.git",
    "base": "c4c10b285f1741aa798d5ee9c5296ae371fabecf",
    "branch": "engine-cycle-integration",
    "commit": "26501ed3d0ac118c227b0f1919f520844e564948",
    "source_working_tree_clean": true,
    "implementation_pushed": false,
    "main_merged": false,
    "full_operator_battery_performed": false
  },
  "showcase": {
    "timing": "in a few days; exact date not supplied",
    "duration_minutes_approx": 10,
    "audience": "technical, potentially unfamiliar with this topic or AI",
    "beta_page_id": "page_25fc2658473c819194876bbbe931bd18",
    "beta_url": "https://chatgpt.com/space/page_25fc2658473c819194876bbbe931bd18",
    "beta_status": "outdated; predates the SHD pilot and complete Engine cycle",
    "presentation_edited_in_this_handoff": false
  },
  "assessment": {
    "bounded_full_engine_integration_demonstrated": true,
    "effective_real_data_continual_learning_demonstrated": false,
    "growth_superiority_demonstrated": false,
    "recursive_self_improvement_demonstrated": false,
    "recommended_next_step": "Diagnose basic SHD learnability before a larger integrated efficacy campaign"
  },
  "shd": {
    "scope": "exploratory fixed-width SHD audio component pilot; no full-engine inference",
    "metric_units": "accuracy in percent; variability and differences in percentage points",
    "variability": "sample standard deviation across four exposed development seeds, not confidence intervals",
    "seeds": [
      8000,
      8001,
      8002,
      8003
    ],
    "data": {
      "dataset": "SHD official training set only",
      "url": "https://zenkelab.org/datasets/shd/shd_train.h5.gz",
      "publisher_md5": "d47c9825dee33347913e8ce0f2be08b0",
      "archive_sha256": "e95bdc00c03c36537ff587d85d14efeee14a01b3846e14a70defc7e8a14e387c",
      "bins": 100,
      "channels": 700,
      "classes": 20,
      "framing": "per-recording observed min/max equal-width bins, last endpoint clamped; counts preserved",
      "recordings": 8156,
      "train_recordings": 6369,
      "dev_recordings": 1787,
      "held_speakers": [
        10,
        11
      ],
      "train_speakers": [
        0,
        1,
        2,
        3,
        6,
        7,
        8,
        9
      ],
      "total_events": 63993588,
      "largest_bin_count": 6,
      "all_events_conserved": true,
      "split_disjoint": true,
      "all_classes_each_split": true,
      "official_test_read": false
    },
    "completed_runs": 16,
    "initialized_training_attempts": 17,
    "prediction_checkpoints_independently_rescored": 144,
    "infrastructure_interruptions": [
      {
        "status": "infrastructure_interruption",
        "cause": "processes disappeared and all /private/tmp/sndl-* caches were absent after storage pressure; reset cause unconfirmed",
        "last_recorded_exposures": 5736,
        "last_recorded_updates": 363,
        "last_recorded_elapsed_s": 480.3300643750117,
        "completed_prediction_checkpoints": [
          "initial",
          "block-1",
          "block-2",
          "block-3",
          "block-4",
          "block-5",
          "block-6",
          "block-7",
          "block-8"
        ],
        "retry": "same seed 8003, same code, original byte-identical data manifest, full restart; retain original attempt"
      }
    ],
    "frozen_training_fingerprint": {
      "runner_sha256": "5cdb06d80c56f83261af335687123a15ccb2f96a50fdeac5e6f10e99faec493a",
      "protocol_sha256": "b6407d8bfd47585d1fa6a244fc0390b61afbf2f66a34f2498ca8522cf09f6aa1",
      "dataset_manifest_sha256": "abb54567b576e6b66c8de2c8e4dddf2c30bce20d5e548cc0b9bc543fb69ddace",
      "config": {
        "channels": 700,
        "num_classes": 20,
        "hidden": 64,
        "alpha": 0.9,
        "v_th": 0.6,
        "gamma": 0.5,
        "damp": 0.3,
        "kappa": 0.9
      },
      "lr": 0.002,
      "batch": 16
    },
    "endpoints": [
      {
        "arm": "joint",
        "n": 4,
        "macro_accuracy_mean_pct": 41.16210750824956,
        "overall_accuracy_mean_pct": 41.46614437604924,
        "macro_accuracy_sd_pp": 3.1963200418470654,
        "min_pct": 38.09483561605395,
        "max_pct": 45.495070869162575,
        "mean_wall_seconds": 148.3926364272338,
        "max_peak_rss_bytes": 514064384
      },
      {
        "arm": "sequential",
        "n": 4,
        "macro_accuracy_mean_pct": 6.221827905939617,
        "overall_accuracy_mean_pct": 5.609960828203693,
        "macro_accuracy_sd_pp": 1.8270665273314324,
        "min_pct": 5.0,
        "max_pct": 8.927076359725735,
        "mean_wall_seconds": 87.87215957300708,
        "max_peak_rss_bytes": 642678784,
        "mean_forgetting_pp": 68.83865505129573,
        "final_old_class_macro_mean_pct": 2.479743433257149,
        "final_latest_class_macro_mean_pct": 39.90058816008183
      },
      {
        "arm": "repeat",
        "n": 4,
        "macro_accuracy_mean_pct": 7.208917620705595,
        "overall_accuracy_mean_pct": 6.869054280917739,
        "macro_accuracy_sd_pp": 2.2988328039377555,
        "min_pct": 4.871794871794872,
        "max_pct": 9.797979797979798,
        "mean_wall_seconds": 137.38546959374779,
        "max_peak_rss_bytes": 742653952,
        "mean_forgetting_pp": 73.78076634057246,
        "final_old_class_macro_mean_pct": 0.0,
        "final_latest_class_macro_mean_pct": 72.08917620705596
      },
      {
        "arm": "replay",
        "n": 4,
        "macro_accuracy_mean_pct": 12.67832533297165,
        "overall_accuracy_mean_pct": 11.66759932848349,
        "macro_accuracy_sd_pp": 2.5719725434089993,
        "min_pct": 9.703817573546488,
        "max_pct": 15.57506954891707,
        "mean_wall_seconds": 138.41684523974624,
        "max_peak_rss_bytes": 875003904,
        "mean_forgetting_pp": 78.42061794517596,
        "final_old_class_macro_mean_pct": 6.032026485281534,
        "final_latest_class_macro_mean_pct": 72.4950149621827
      }
    ],
    "paired_results": [
      {
        "seed": 8000,
        "replay_minus_repeat_macro_pp": 5.845543215284389,
        "repeat_forgetting_pp": 77.9078204893817,
        "replay_forgetting_pp": 79.93228717264215
      },
      {
        "seed": 8001,
        "replay_minus_repeat_macro_pp": 5.422975181090943,
        "repeat_forgetting_pp": 72.61117279860834,
        "replay_forgetting_pp": 75.89397732284256
      },
      {
        "seed": 8002,
        "replay_minus_repeat_macro_pp": 4.832022701751615,
        "repeat_forgetting_pp": 64.7853667102366,
        "replay_forgetting_pp": 75.62437528658059
      },
      {
        "seed": 8003,
        "replay_minus_repeat_macro_pp": 5.7770897509372725,
        "repeat_forgetting_pp": 79.8187053640632,
        "replay_forgetting_pp": 82.23183199863851
      }
    ],
    "paired_mean_pp": 5.4694077122660545,
    "paired_sd_pp": 0.46352338493023615,
    "total_completed_cohort_training_exposures": 178332,
    "summed_training_run_wall_seconds_excludes_setup_and_recovery": 2048.2684433349395,
    "cautions": [
      "Fixed width 64; no full Brain, structural growth, or persistence experiment",
      "Joint reference was only two passes, not a converged upper bound",
      "Replay and repeat match updates and batch sizes; sequential has half the exposures",
      "Replay improves final accuracy but not the registered mean forgetting endpoint",
      "Development-only split and exposed seeds; no confirmatory significance claim",
      "Per-recording time normalization removes absolute duration",
      "Secondary forgetting summary corrected from raw matrices; training and accuracy unchanged"
    ]
  },
  "engine": {
    "scope": "single exposed fixture, actual persisted full Brain, task-conditioned, no-prune",
    "seed": 9900,
    "curriculum": [
      "aii",
      "ii.xor"
    ],
    "unique_inputs_per_task": 4,
    "presentations_per_validation_or_final_task": 16,
    "final_set_uses_unseen_semantic_inputs": false,
    "feedback_gate": "corrective",
    "old_core_during_B": "frozen; grown outputs remain plastic",
    "candidate_pool": 1024,
    "admitted_batch_size": 18,
    "batch_size_is_optimal_or_autonomously_discovered": false,
    "audit": {
      "scope": "independent artifact and raw-journal audit, no Brain or learning run",
      "seed": 9900,
      "checks": {
        "lake_rb_only_no_change": true,
        "raw_console_rows_equal_journal": true,
        "canonical_neurons_and_complete_stack_exact_old_to_grow": true,
        "source_18_of_1024_one_batch_stream9900": true
      },
      "scores": {
        "old.validation.aii": {
          "correct": 16,
          "presentations": 16
        },
        "grow.validation.ii.xor": {
          "correct": 16,
          "presentations": 16
        },
        "grow.validation.aii": {
          "correct": 16,
          "presentations": 16
        },
        "grow.final.ii.xor": {
          "correct": 16,
          "presentations": 16
        },
        "grow.final.aii": {
          "correct": 16,
          "presentations": 16
        },
        "grow.grown_off.ii.xor": {
          "correct": 8,
          "presentations": 16
        },
        "grow.grown_off.aii": {
          "correct": 16,
          "presentations": 16
        },
        "prior_allplastic.aii": {
          "correct": 0,
          "presentations": 16
        },
        "prior_allplastic.ii.xor": {
          "correct": 16,
          "presentations": 16
        },
        "grow.channel.baseline": {
          "correct": 16,
          "presentations": 16
        },
        "grow.channel.feed_off": {
          "correct": 16,
          "presentations": 16
        },
        "grow.channel.return_off": {
          "correct": 8,
          "presentations": 16
        },
        "grow.channel.both_off": {
          "correct": 8,
          "presentations": 16
        },
        "cycle-04-grow-restore-2.json.ii.xor": {
          "correct": 16,
          "presentations": 16
        },
        "cycle-04-grow-restore-2.json.aii": {
          "correct": 16,
          "presentations": 16
        },
        "cycle-04-grow-restore-journal.json.ii.xor": {
          "correct": 16,
          "presentations": 16
        },
        "cycle-04-grow-restore-journal.json.aii": {
          "correct": 16,
          "presentations": 16
        }
      },
      "actual_epoch_history": {
        "aii": {
          "epochs": 22,
          "teach_presentations": 88,
          "validation_presentations": 352,
          "shared_plasticity_deposits": 66,
          "first_all_correct_epoch": 13,
          "final_stable_run": 10,
          "all_epoch_rows_rescored": 22
        },
        "ii.xor": {
          "epochs": 107,
          "teach_presentations": 428,
          "validation_presentations": 1712,
          "shared_plasticity_deposits": 0,
          "first_all_correct_epoch": 98,
          "final_stable_run": 10,
          "all_epoch_rows_rescored": 107
        }
      },
      "schedule_summary": {
        "aii": {
          "unique_teach_items": 4,
          "teach_items_per_epoch": 4,
          "validation_unique_texts": 4,
          "validation_presentations": 16,
          "repetitions_per_unique_item": 4,
          "actor_teaches": 88
        },
        "ii.xor": {
          "unique_teach_items": 4,
          "teach_items_per_epoch": 4,
          "validation_unique_texts": 4,
          "validation_presentations": 16,
          "repetitions_per_unique_item": 4,
          "actor_teaches": 428
        }
      },
      "source": {
        "batch_id": 0,
        "stream_seed": 9900,
        "evaluations": 1024,
        "admitted_ids": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
          17
        ],
        "selected_source_indices": [
          737,
          678,
          328,
          22,
          471,
          505,
          87,
          407,
          762,
          390,
          140,
          114,
          387,
          111,
          816,
          17,
          711,
          973
        ]
      },
      "channel_interventions": {
        "baseline": {
          "score": {
            "correct": 16,
            "presentations": 16
          },
          "pool_feature_l2_delta": 0.0,
          "feed_l2_minmax": [
            2.567633719219973,
            2.976195381808685
          ],
          "return_weights_l2": 4.913636133957407,
          "afferent_l1_minmax": [
            7946.315312388685,
            8230.419010981283
          ],
          "changed_latency_entries": 0,
          "grown_drive_l2_delta": 0.0
        },
        "feed_off": {
          "score": {
            "correct": 16,
            "presentations": 16
          },
          "pool_feature_l2_delta": 32.68026927673638,
          "feed_l2_minmax": [
            0.0,
            0.0
          ],
          "return_weights_l2": 4.913636133957407,
          "afferent_l1_minmax": [
            7946.315312388685,
            8230.419010981283
          ],
          "changed_latency_entries": 0,
          "grown_drive_l2_delta": 0.0
        },
        "return_off": {
          "score": {
            "correct": 8,
            "presentations": 16
          },
          "pool_feature_l2_delta": 21.166010488516726,
          "feed_l2_minmax": [
            3.760032649929224,
            3.969203856319631
          ],
          "return_weights_l2": 0.0,
          "afferent_l1_minmax": [
            0.0,
            0.0
          ],
          "changed_latency_entries": 220,
          "grown_drive_l2_delta": 0.536067283836626
        },
        "both_off": {
          "score": {
            "correct": 8,
            "presentations": 16
          },
          "pool_feature_l2_delta": 32.68026927673638,
          "feed_l2_minmax": [
            0.0,
            0.0
          ],
          "return_weights_l2": 0.0,
          "afferent_l1_minmax": [
            0.0,
            0.0
          ],
          "changed_latency_entries": 220,
          "grown_drive_l2_delta": 0.536067283836626
        }
      },
      "lesion_arithmetic_rescore": {
        "aii": {
          "correct": 0,
          "presentations": 16,
          "predictions_unchanged": false
        },
        "ii.xor": {
          "correct": 16,
          "presentations": 16,
          "predictions_unchanged": true
        }
      },
      "restores": {
        "cycle-04-grow-restore-2.json": {
          "journal_only": false,
          "exact": {
            "model_digest": true,
            "core_state": true,
            "actors": true,
            "validation": true,
            "source_events": true
          },
          "final_repeat_refused": true,
          "reason": "refusal asserted by recorded controller"
        },
        "cycle-04-grow-restore-journal.json": {
          "journal_only": true,
          "exact": {
            "model_digest": true,
            "core_state": true,
            "actors": true,
            "validation": true,
            "source_events": true
          },
          "final_repeat_refused": true,
          "reason": "G9_RETENTION_ALREADY_CONSUMED"
        }
      },
      "provenance": {
        "old": {
          "exit_code": 0,
          "controller_seconds": 9.676446291000502,
          "source_hashes_verified": 8
        },
        "grow": {
          "exit_code": 0,
          "controller_seconds": 135.82026391600084,
          "source_hashes_verified": 8
        },
        "restore-2": {
          "exit_code": 0,
          "controller_seconds": 5.082242708000194,
          "source_hashes_verified": 8
        },
        "restore_journal": {
          "exit_code": 0,
          "controller_seconds": 31.41120899999987,
          "source_hashes_verified": 8
        }
      },
      "lake": {
        "sha256": "dda9da7974984173ccba86d74fcc2b2755bb64ab3133628f370e5c0b4dcc5f7a",
        "frames": 685,
        "goal_events": 159
      },
      "retained_negative_control": {
        "allplastic_A": {
          "correct": 0,
          "presentations": 16
        },
        "allplastic_B": {
          "correct": 16,
          "presentations": 16
        },
        "no_frozen_selection": true,
        "no_final_claims": true
      },
      "limitations": [
        "Single already exposed engineering seed; no independent statistical cohort or efficacy comparison.",
        "Four unique constructed text/code items per task, repeated four times per read set; not 16 independent cases.",
        "Future/final is a different presentation order of those same four items, not a new semantic generalization set.",
        "Task-conditioned context, corrective feedback, explicit protected core, one admitted batch of18, no pruning; no autonomous RSI claim.",
        "Feed ablation changes Part D features but retains16/16; no demonstrated final-answer feed benefit.",
        "Return ablation reduces B to8/16 on this fixture; not a general causal/statistical effect estimate.",
        "Canonical neurons+complete stack are independently equal; full shared-state digest is compared as recorded, not recomputed from an unavailable full shared snapshot.",
        "Lesion rescore uses saved core weights plus actual captured grown-drive continuation; controller independently asserted real lesion refusal.",
        "First checkpoint restore controller retained an assertion failure caused by tuple/list normalization; corrected checkpoint and journal-only controllers pass.",
        "No SHD result or full operator battery is implied."
      ]
    },
    "completion": {
      "commit": "26501ed3",
      "branch": "engine-cycle-integration",
      "focused_tests": 215,
      "checkpoint_recovery": true,
      "journal_only_recovery": true,
      "final_repeat_reason": "G9_RETENTION_ALREADY_CONSUMED",
      "model_digest": "ff48914c33b40f4c87eb4bbf1de6b8f3414f810249eb0a6567dd7b45ed516a1c",
      "source_files": [
        {
          "path": "src/snn_data_lake/growth/growth_loop.py",
          "acceptance_sha256": "23d36ccf5cb5d8eed418678150be92a26300cdaf9c90e94413cd54a24ff4546f",
          "final_sha256": "23d36ccf5cb5d8eed418678150be92a26300cdaf9c90e94413cd54a24ff4546f",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        },
        {
          "path": "src/snn_data_lake/growth/capacity_certificate.py",
          "acceptance_sha256": "e002742ffe9245e70ef4d6dcb5fec32bb0c1d4f54580b23e6e7351d250af1814",
          "final_sha256": "0c715d32840e55b21f463798f0b5811d5c490ac0de05a64982a6201679e92fe8",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        },
        {
          "path": "src/snn_data_lake/engine/brain.py",
          "acceptance_sha256": "5007170644a026a337b50053ea05c2962d308de7553eeed998cba0e1c8790c0f",
          "final_sha256": "6b07551d4b9a808d3cc9a82c722a68750a91260b11de3723de580017b818b9fc",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        },
        {
          "path": "src/snn_data_lake/readout/posed_problem.py",
          "acceptance_sha256": "0e6e90143a0bc586c574a9e3bb8cdc56a1892e03346f5f5fc03ebcfea7f8d86d",
          "final_sha256": "0e6e90143a0bc586c574a9e3bb8cdc56a1892e03346f5f5fc03ebcfea7f8d86d",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        },
        {
          "path": "src/snn_data_lake/instruments/joint_mask_search.py",
          "acceptance_sha256": "2771ab09eac8e4e34dfa0d82b752b4a1a839a2c5420eaa06bcf55eee61b8b0dc",
          "final_sha256": "2771ab09eac8e4e34dfa0d82b752b4a1a839a2c5420eaa06bcf55eee61b8b0dc",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        },
        {
          "path": "src/snn_data_lake/readout/interaction_learning.py",
          "acceptance_sha256": "f1d9a1785e66921c6f4bb2fa608f4cca0e8080b317bcf68243a5078cbbac14b9",
          "final_sha256": "f1d9a1785e66921c6f4bb2fa608f4cca0e8080b317bcf68243a5078cbbac14b9",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        },
        {
          "path": "src/snn_data_lake/plasticity/plasticity_rule_stack.py",
          "acceptance_sha256": "07f27c5b21becb63d95fba41f60d1aeb108c9bb32365c6321a50e26121022a88",
          "final_sha256": "32487f79b4e13edfa7413eec391d425e5fc5d3889097576eba25b88b84c98d65",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        },
        {
          "path": "src/snn_data_lake/plasticity/homeostatic_plasticity_stack.py",
          "acceptance_sha256": "99fe23a9df1cda8286f30479a873655f68586af82f0a1e9606df3fcaa8f5a637",
          "final_sha256": "4d1175fa1a3e1871c8d333ced5259e07a9a2ba15d40844ee6ecf6ba15b543dda",
          "numerical_ast_identical_excluding_claim_declaration": true,
          "missing_claims": [],
          "phantom_claims": []
        }
      ]
    },
    "cautions": [
      "All-plastic full-Engine control: B16/16, A0/16; final selection refused",
      "Grown/return ablations show dependence, not superiority over equal-capacity controls",
      "Feed changes internal features but no answer-accuracy effect established",
      "No full-Engine SHD, independent efficacy cohort, uncued composition, pruning benefit, or RSI",
      "Source-owned BETA for a later positive-spend task remains unset",
      "Original persistent lake and host runtime are not included in this packet"
    ]
  },
  "next_experiments_proposed_not_executed": [
    "Small fixed balanced subset fitting and learner diagnostics",
    "Longer joint training with training/development convergence curves",
    "Simple reference learner on identical split and representation",
    "Analysis of incremental collapse, confusion, reservoir coverage, and per-speaker errors",
    "Later growth-on/off and equal-capacity full-Engine efficacy comparison"
  ],
  "operational_context": {
    "host_used": "Apple M2 Mac, 8 GB RAM, CPU",
    "python_used": "CPython 3.14.6 free-threaded with GIL off",
    "torch_used": "2.15.0 CPU",
    "numpy_used": "2.5.4",
    "native_build": "external Apple Silicon cp314t extension; not included",
    "branch_preference": "Never use a codex prefix",
    "wrapper_caveat": "run-engine.sh defaults to baseline; set ENGINE_SOURCE_ROOT for repaired source",
    "do_not_use_audio_wrapper_for_full_Brain": true
  }
}
```


# SHD prospective protocol

Original workspace member: `planning/experiments/scale/PROTOCOL.md`

# SHD component pilot protocol, revision 1

Registered before SHD model outcomes on 2026-11-03. Exploratory feasibility study, not a powered confirmation. Previously exposed development seeds 8000–8003 are reused; they are not a new independent confirmation cohort. The official SHD test archive is not downloaded or read.

## Data and measured variables

Use the official SHD training archive, verified against the publisher's MD5 `d47c9825dee33347913e8ce0f2be08b0`; also record SHA256. Every recording has raw spike times, channel IDs, a spoken-digit class and speaker metadata. Labels and speaker IDs are scorer/split metadata, never network inputs.

Operational cache location: `/private/tmp/sndl-scale-data-20261103`, because the first Documents archive became a macOS dataless file and timed out during reading. Portable manifests, scripts and run results stay in the workspace; cache is regenerable and may be removed by the OS. This relocation precedes all model outcomes.

Inputs: per-recording 100-bin by 700-channel spike **counts**. Map each recording's observed [minimum time, maximum time] linearly to 100 bins, clamp the last event into bin 99, and preserve every event exactly. No clipping, normalization by label, synthetic augmentation or fabricated samples. Variable-duration normalization removes absolute duration; retain duration/event statistics in the preparation report. This explicit adapter is not claimed bit-identical to Tonic. UInt16 storage must be checked for overflow. Float32 counts enter the network.

Hold out the two numerically highest training-speaker IDs, chosen by metadata only; all other speakers train. Stop if either side lacks any of the 20 classes, and register a corrected split before training. No utterance or speaker crosses this boundary. Save sample IDs and split/array hashes. Evaluation uses all held-out recordings, all 20 output classes and no task-identity mask.

Outputs: 20 logits, predicted class, summed hidden spike activity, per-example target/prediction records, per-class confusion counts, cross-entropy, macro and overall accuracy, actual exposures/updates, wall time and peak RSS. Initial and every-block/epoch evaluation cannot modify weights or optimizer state. Hash weights before and after evaluation. This component has fixed width; no growth is measured or implied.

## Existing implementation and pilot budget

Import the pinned source's `EpropAudioNet`, `eprop_grads`, and shared `_apply_grads` directly; no copied or substituted learning kernel. CPU, one Torch thread, no DataLoader workers. Initial calibration: 100 time bins, hidden width 64, batch 16, Adam learning rate 0.002 and source dynamics/initialization defaults. H=64 is a laptop feasibility setting, not an estimate of engine capacity. Profile cumulative 256, 1,024 and 4,096 training exposures. Log actual seconds, RSS, loss and spiking. Stop on nonfinite outputs/gradients, RSS above 3 GiB, or four hours total training time. Do not drop low-scoring runs.

First run a joint IID positive control on seed 8000 for two full passes through training data. Apparatus gate: final held-out macro accuracy must exceed 10% (5% uniform chance plus 5 percentage points). This is a feasibility threshold, not a meaningful scientific effect size. If it fails, stop the larger comparison and report the diagnosis with all real measurements. Any later hyperparameter revision gets a new protocol and visible exploratory status.

If feasible, run up to four paired seeds (8000–8003), limited by measured runtime and the four-hour cap. Each seed uses the same initial weights and class order across arms. Shuffle 20 classes with the seed and divide into ten disjoint two-class blocks. Each training recording appears once as a current-stream example. Shuffle within each block; batch size 16 (final partial batch allowed). Evaluate every block on the complete held-out vocabulary and record a 10-by-10 task retention matrix.

Arms: (1) sequential one update per current batch; (2) sequential two updates on the same current batch; (3) sequential one current update plus one replay update of the same batch size, sampled without replacement where possible from previously seen **training-only** examples, up to a uniform 400-recording reservoir. Reservoir selection is label-blind and seeded. Update the reservoir after both updates, so replay comes only from earlier samples. On the first batch use a second current update because no history exists. Replay and repeated-current arms have exactly matched update counts and batch sizes. They differ in historical input access; report that distinction and actual unique/exposure counts. An IID two-pass reference has the same aggregate training exposure count but a different order; it is a learnability reference, not an architectural win.

Final endpoints: overall/macro held-out accuracy; mean seen-task accuracy after each block; forgetting (best earlier task accuracy minus final, excluding tasks not yet learned); replay-minus-repeat paired differences; exposures, time, RSS and memory footprint. Report every seed and descriptive variability without p-values or confirmatory confidence claims from four exposed seeds. Speaker holdout uncertainty and single-split limits remain explicit. No choosing the best checkpoint. Retain raw predictions and failures. No saved-lake persistence claim in this pilot; do not start a Brain or live lake.

## Required checks before interpreting results

Verify raw event conservation and ranges, deterministic binning, train/development sample and speaker disjointness, class coverage, hashes and source identity. Construct a known confusion matrix and a deliberately permuted target fault that changes its score. Verify evaluation leaves weights unchanged. Check repeated identical updates yield identical CPU weights. Replay must contain training IDs only, never a future stream sample or development ID. Keep bounded source self-check results as apparatus evidence, separate from real-data scores.

This pilot tests the existing local e-prop event-audio learner. It does not test G9, growing/adapting engine state, an integrated reservoir, recursive self-improvement, SSC, transformer superiority or the official SHD test benchmark.


# SHD actual data manifest

Original workspace member: `planning/experiments/scale/reports/data-manifest.json`

```json
{
  "dataset": "SHD official training set only",
  "url": "https://zenkelab.org/datasets/shd/shd_train.h5.gz",
  "license": "CC BY 4.0",
  "publisher_md5": "d47c9825dee33347913e8ce0f2be08b0",
  "archive_sha256": "e95bdc00c03c36537ff587d85d14efeee14a01b3846e14a70defc7e8a14e387c",
  "raw_sha256": "2bddb4bd46732f09982b7d1631b7c29c19853c73d3d240e3eb32bba909bdd6c1",
  "schema": {
    "extra/keys": {
      "shape": [
        20
      ],
      "dtype": "|S6"
    },
    "extra/meta_info/age": {
      "shape": [
        12
      ],
      "dtype": "uint16"
    },
    "extra/meta_info/body_height": {
      "shape": [
        12
      ],
      "dtype": "uint16"
    },
    "extra/meta_info/gender": {
      "shape": [
        12
      ],
      "dtype": "|S6"
    },
    "extra/speaker": {
      "shape": [
        8156
      ],
      "dtype": "uint16"
    },
    "labels": {
      "shape": [
        8156
      ],
      "dtype": "uint16"
    },
    "spikes/times": {
      "shape": [
        8156
      ],
      "dtype": "object"
    },
    "spikes/units": {
      "shape": [
        8156
      ],
      "dtype": "object"
    }
  },
  "bins": 100,
  "channels": 700,
  "classes": 20,
  "framing": "per-recording observed min/max equal-width bins, last endpoint clamped; counts preserved",
  "recordings": 8156,
  "train_recordings": 6369,
  "dev_recordings": 1787,
  "held_speakers": [
    10,
    11
  ],
  "train_speakers": [
    0,
    1,
    2,
    3,
    6,
    7,
    8,
    9
  ],
  "train_class_counts": [
    326,
    322,
    323,
    329,
    313,
    319,
    323,
    318,
    317,
    323,
    315,
    313,
    318,
    313,
    317,
    320,
    313,
    315,
    311,
    321
  ],
  "dev_class_counts": [
    80,
    79,
    81,
    78,
    80,
    80,
    77,
    82,
    78,
    83,
    96,
    99,
    94,
    100,
    101,
    101,
    102,
    97,
    99,
    100
  ],
  "total_events": 63993588,
  "events_min_median_max": [
    2410.0,
    7605.0,
    14917.0
  ],
  "duration_min_median_max_seconds": [
    0.2333984375,
    0.70458984375,
    1.369140625
  ],
  "largest_bin_count": 6,
  "all_events_conserved": true,
  "split_disjoint": true,
  "all_classes_each_split": true,
  "official_test_read": false,
  "files": {
    "dev_ids.npy": {
      "bytes": 14424,
      "sha256": "edb791c147b0924bc9db2a325b20a67a4875a8da8b7c4d7ac885494c7fafc4f8"
    },
    "event_counts.npy": {
      "bytes": 65376,
      "sha256": "cacd71654dc0c14796ce2bc045ff59afd63c2c2780a53dd293f77768cf3ad926"
    },
    "frames.npy": {
      "bytes": 1141840128,
      "sha256": "99dcf4c2afaf3bba7767b3ddde8520ed21e74bf0cce03f81bcdfb3f009957ea2"
    },
    "labels.npy": {
      "bytes": 65376,
      "sha256": "79c1773981b65a29dd269af5eab37cdcbc6de6390c2534b28806ee6449dec9b9"
    },
    "speakers.npy": {
      "bytes": 65376,
      "sha256": "1305c3a0e299d3eb4e1dbd9acb537f766eab24307789e33e817549e4e9231d4d"
    },
    "time_limits.npy": {
      "bytes": 130624,
      "sha256": "c6af64927c2cc49468a8dbc108bde20d3ab7d3da10b5d78f3e39284e5ed5e331"
    },
    "train_ids.npy": {
      "bytes": 51080,
      "sha256": "2c5894e5f354d9973088c465f7430edce6e1f599e4eedefaa825817103b3feba"
    }
  },
  "prepare_script_sha256": "7e710a3d99d61eda535b1fdc35d3bde2f3d5782a08b1359b05e7d392f9dbc3e2",
  "numpy": "2.5.4",
  "h5py": "3.16.0",
  "wall_seconds": 19.380558333999943
}
```


# SHD canonical measured results

Original workspace member: `planning/experiments/scale/reports/pilot-results.json`

```json
{
  "scope": "exploratory fixed-width SHD audio component pilot; no full-engine inference",
  "completed_runs": 16,
  "failures": [],
  "independently_verified_prediction_checkpoints": 144,
  "infrastructure_interruptions": [
    {
      "status": "infrastructure_interruption",
      "cause": "processes disappeared and all /private/tmp/sndl-* caches were absent after storage pressure; reset cause unconfirmed",
      "last_recorded_exposures": 5736,
      "last_recorded_updates": 363,
      "last_recorded_elapsed_s": 480.3300643750117,
      "completed_prediction_checkpoints": [
        "initial",
        "block-1",
        "block-2",
        "block-3",
        "block-4",
        "block-5",
        "block-6",
        "block-7",
        "block-8"
      ],
      "retry": "same seed 8003, same code, original byte-identical data manifest, full restart; retain original attempt"
    }
  ],
  "initialized_training_attempts": 17,
  "dataset_manifest_sha256": "abb54567b576e6b66c8de2c8e4dddf2c30bce20d5e548cc0b9bc543fb69ddace",
  "official_test_read": false,
  "scoring_metadata_source": "/Users/arthurkhvan/Documents/ChatGPT/Spiking-Neural-Data-Lake 2/planning/experiments/scale/reports/evaluation-metadata",
  "endpoints": [
    {
      "arm": "joint",
      "n": 4,
      "macro_accuracy_mean_pct": 41.16210750824956,
      "overall_accuracy_mean_pct": 41.46614437604924,
      "macro_accuracy_sd_pp": 3.1963200418470654,
      "min_pct": 38.09483561605395,
      "max_pct": 45.495070869162575,
      "mean_wall_seconds": 148.3926364272338,
      "max_peak_rss_bytes": 514064384
    },
    {
      "arm": "sequential",
      "n": 4,
      "macro_accuracy_mean_pct": 6.221827905939617,
      "overall_accuracy_mean_pct": 5.609960828203693,
      "macro_accuracy_sd_pp": 1.8270665273314324,
      "min_pct": 5.0,
      "max_pct": 8.927076359725735,
      "mean_wall_seconds": 87.87215957300708,
      "max_peak_rss_bytes": 642678784,
      "mean_forgetting_pp": 68.83865505129573,
      "final_old_class_macro_mean_pct": 2.479743433257149,
      "final_latest_class_macro_mean_pct": 39.90058816008183
    },
    {
      "arm": "repeat",
      "n": 4,
      "macro_accuracy_mean_pct": 7.208917620705595,
      "overall_accuracy_mean_pct": 6.869054280917739,
      "macro_accuracy_sd_pp": 2.2988328039377555,
      "min_pct": 4.871794871794872,
      "max_pct": 9.797979797979798,
      "mean_wall_seconds": 137.38546959374779,
      "max_peak_rss_bytes": 742653952,
      "mean_forgetting_pp": 73.78076634057246,
      "final_old_class_macro_mean_pct": 0.0,
      "final_latest_class_macro_mean_pct": 72.08917620705596
    },
    {
      "arm": "replay",
      "n": 4,
      "macro_accuracy_mean_pct": 12.67832533297165,
      "overall_accuracy_mean_pct": 11.66759932848349,
      "macro_accuracy_sd_pp": 2.5719725434089993,
      "min_pct": 9.703817573546488,
      "max_pct": 15.57506954891707,
      "mean_wall_seconds": 138.41684523974624,
      "max_peak_rss_bytes": 875003904,
      "mean_forgetting_pp": 78.42061794517596,
      "final_old_class_macro_mean_pct": 6.032026485281534,
      "final_latest_class_macro_mean_pct": 72.4950149621827
    }
  ],
  "paired_results": [
    {
      "seed": 8000,
      "replay_minus_repeat_macro_pp": 5.845543215284389,
      "repeat_forgetting_pp": 77.9078204893817,
      "replay_forgetting_pp": 79.93228717264215
    },
    {
      "seed": 8001,
      "replay_minus_repeat_macro_pp": 5.422975181090943,
      "repeat_forgetting_pp": 72.61117279860834,
      "replay_forgetting_pp": 75.89397732284256
    },
    {
      "seed": 8002,
      "replay_minus_repeat_macro_pp": 4.832022701751615,
      "repeat_forgetting_pp": 64.7853667102366,
      "replay_forgetting_pp": 75.62437528658059
    },
    {
      "seed": 8003,
      "replay_minus_repeat_macro_pp": 5.7770897509372725,
      "repeat_forgetting_pp": 79.8187053640632,
      "replay_forgetting_pp": 82.23183199863851
    }
  ],
  "source_files": {
    "snn_data_lake/__init__.py": "d682c8dffa9519486740d75e6bfe886febca62645ce21add3961b24e341ef6ae",
    "snn_data_lake/sensorimotor/__init__.py": "14f44e2c16c3aee993cc6af227715c1af849de59b9bd036cb01ebddba548679a",
    "snn_data_lake/plasticity/__init__.py": "e597c53ba470dc8eddd185582d873f1daf6033ca4b400982b8a4ca5814fc8614",
    "snn_data_lake/plasticity/eprop_kernel.py": "f10cca303e28ba2084dd38fae1b5e47b6edaa9aa3e443aad805310c270dbf5d9",
    "snn_data_lake/sensorimotor/eprop_shd.py": "0ddc1301ffa73cc7ba109b2a8824628f2ade706506f4eef0e82e911e07fb3ba6"
  },
  "frozen_training_fingerprint": {
    "runner_sha256": "5cdb06d80c56f83261af335687123a15ccb2f96a50fdeac5e6f10e99faec493a",
    "protocol_sha256": "b6407d8bfd47585d1fa6a244fc0390b61afbf2f66a34f2498ca8522cf09f6aa1",
    "dataset_manifest_sha256": "abb54567b576e6b66c8de2c8e4dddf2c30bce20d5e548cc0b9bc543fb69ddace",
    "config": {
      "channels": 700,
      "num_classes": 20,
      "hidden": 64,
      "alpha": 0.9,
      "v_th": 0.6,
      "gamma": 0.5,
      "damp": 0.3,
      "kappa": 0.9
    },
    "lr": 0.002,
    "batch": 16
  },
  "heldout_hidden_activity": [
    {
      "run_id": "shd-joint-s8000",
      "arm": "joint",
      "seed": 8000,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19854198396205902
    },
    {
      "run_id": "shd-joint-s8000",
      "arm": "joint",
      "seed": 8000,
      "checkpoint": "epoch-1",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.15799105167388916
    },
    {
      "run_id": "shd-joint-s8000",
      "arm": "joint",
      "seed": 8000,
      "checkpoint": "epoch-2",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.17186661064624786
    },
    {
      "run_id": "shd-joint-s8001",
      "arm": "joint",
      "seed": 8001,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19202756881713867
    },
    {
      "run_id": "shd-joint-s8001",
      "arm": "joint",
      "seed": 8001,
      "checkpoint": "epoch-1",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.18366719782352448
    },
    {
      "run_id": "shd-joint-s8001",
      "arm": "joint",
      "seed": 8001,
      "checkpoint": "epoch-2",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.17328903079032898
    },
    {
      "run_id": "shd-joint-s8002",
      "arm": "joint",
      "seed": 8002,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18266673386096954
    },
    {
      "run_id": "shd-joint-s8002",
      "arm": "joint",
      "seed": 8002,
      "checkpoint": "epoch-1",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.13981463015079498
    },
    {
      "run_id": "shd-joint-s8002",
      "arm": "joint",
      "seed": 8002,
      "checkpoint": "epoch-2",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.13992103934288025
    },
    {
      "run_id": "shd-joint-s8003",
      "arm": "joint",
      "seed": 8003,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18768815696239471
    },
    {
      "run_id": "shd-joint-s8003",
      "arm": "joint",
      "seed": 8003,
      "checkpoint": "epoch-1",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.18484750390052795
    },
    {
      "run_id": "shd-joint-s8003",
      "arm": "joint",
      "seed": 8003,
      "checkpoint": "epoch-2",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.18544015288352966
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19854198396205902
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.33139392733573914
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.19603762030601501
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.1482185572385788
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.11827459931373596
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.07845664769411087
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.0875486209988594
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.07886698842048645
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.09707190841436386
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.10285884141921997
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "seed": 8000,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.09357836097478867
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19202756881713867
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.4002998471260071
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.2983323037624359
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.2612527012825012
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.12116561830043793
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.08564694225788116
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.06343374401330948
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.07224556058645248
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.07349450886249542
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.06756959855556488
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "seed": 8001,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.0459345281124115
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18266673386096954
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.3559449315071106
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.21357381343841553
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.13056650757789612
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.13858160376548767
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.09863851219415665
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.05521824210882187
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.0842059776186943
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.08681563287973404
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.08859419077634811
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "seed": 8002,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.05788559839129448
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18768815696239471
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.3657997250556946
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.2127949297428131
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.17281101644039154
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.14586089551448822
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.09618765860795975
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.1037960797548294
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.07042389363050461
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.08211257308721542
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.09438365697860718
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "seed": 8003,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.09092745184898376
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19854198396205902
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.3367234766483307
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.2082323580980301
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.1615770161151886
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.1474425494670868
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.1115257740020752
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.07823482155799866
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.0655406266450882
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.03026222437620163
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.06851015985012054
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "seed": 8000,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.06495549529790878
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19202756881713867
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.40694963932037354
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.2770286202430725
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.23291681706905365
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.16857311129570007
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.1319846510887146
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.10382825881242752
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.08637109398841858
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.08731096237897873
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.08834420144557953
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "seed": 8001,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.071088507771492
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18266673386096954
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.35812222957611084
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.24049900472164154
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.14673711359500885
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.0919005274772644
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.05760885775089264
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.027947939932346344
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.03680059313774109
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.021031932905316353
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.030178632587194443
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "seed": 8002,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.017375051975250244
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18768815696239471
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.36082592606544495
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.22756357491016388
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.17491640150547028
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.15061748027801514
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.0982922613620758
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.07727187871932983
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.049654360860586166
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.033487603068351746
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.052383970469236374
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "seed": 8003,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.0632328987121582
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19854198396205902
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.3333333730697632
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.19638876616954803
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.13867226243019104
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.10448009520769119
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.09891245514154434
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.05021107196807861
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.045658402144908905
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.047627922147512436
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.06577538698911667
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "seed": 8000,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.05164521560072899
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.19202756881713867
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.3959640860557556
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.24493905901908875
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.18318147957324982
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.09646265208721161
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.07856131345033646
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.06963687390089035
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.07569582760334015
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.08803484588861465
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.08718164265155792
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "seed": 8001,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.12031145393848419
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18266673386096954
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.3889632523059845
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.25958484411239624
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.15473943948745728
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.11601287126541138
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.06764260679483414
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.06599818170070648
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.06849660724401474
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.018331700935959816
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.015474345535039902
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "seed": 8002,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.0023621993605047464
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "initial",
      "learned_tasks": 0,
      "mean_heldout_hidden_spike_fraction": 0.18768815696239471
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-1",
      "learned_tasks": 1,
      "mean_heldout_hidden_spike_fraction": 0.3933295011520386
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-2",
      "learned_tasks": 2,
      "mean_heldout_hidden_spike_fraction": 0.21384574472904205
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-3",
      "learned_tasks": 3,
      "mean_heldout_hidden_spike_fraction": 0.12472326308488846
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-4",
      "learned_tasks": 4,
      "mean_heldout_hidden_spike_fraction": 0.10808399319648743
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-5",
      "learned_tasks": 5,
      "mean_heldout_hidden_spike_fraction": 0.05888631194829941
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-6",
      "learned_tasks": 6,
      "mean_heldout_hidden_spike_fraction": 0.051428984850645065
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-7",
      "learned_tasks": 7,
      "mean_heldout_hidden_spike_fraction": 0.051894411444664
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-8",
      "learned_tasks": 8,
      "mean_heldout_hidden_spike_fraction": 0.05280227214097977
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-9",
      "learned_tasks": 9,
      "mean_heldout_hidden_spike_fraction": 0.0459577850997448
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "seed": 8003,
      "checkpoint": "block-10",
      "learned_tasks": 10,
      "mean_heldout_hidden_spike_fraction": 0.07900872826576233
    }
  ],
  "final_speaker_metrics": [
    {
      "run_id": "shd-joint-s8000",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.47782002534854245,
      "macro_accuracy": 0.47971786555097184
    },
    {
      "run_id": "shd-joint-s8000",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.3076152304609218,
      "macro_accuracy": 0.3018147270755366
    },
    {
      "run_id": "shd-joint-s8001",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.5475285171102662,
      "macro_accuracy": 0.5464457759615907
    },
    {
      "run_id": "shd-joint-s8001",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.39579158316633267,
      "macro_accuracy": 0.3721561646989009
    },
    {
      "run_id": "shd-joint-s8002",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.5057034220532319,
      "macro_accuracy": 0.5045355120467259
    },
    {
      "run_id": "shd-joint-s8002",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.3527054108216433,
      "macro_accuracy": 0.33138141614236827
    },
    {
      "run_id": "shd-joint-s8003",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.4435994930291508,
      "macro_accuracy": 0.43303109929650174
    },
    {
      "run_id": "shd-joint-s8003",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.3527054108216433,
      "macro_accuracy": 0.34386285738354233
    },
    {
      "run_id": "shd-repeat-s8000",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.034220532319391636,
      "macro_accuracy": 0.03461538461538462
    },
    {
      "run_id": "shd-repeat-s8000",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.08016032064128256,
      "macro_accuracy": 0.07992910943730616
    },
    {
      "run_id": "shd-repeat-s8001",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.08998732572877059,
      "macro_accuracy": 0.08771341463414635
    },
    {
      "run_id": "shd-repeat-s8001",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.06312625250501001,
      "macro_accuracy": 0.08076923076923077
    },
    {
      "run_id": "shd-repeat-s8002",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.053231939163498096,
      "macro_accuracy": 0.04883720930232558
    },
    {
      "run_id": "shd-repeat-s8002",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.03406813627254509,
      "macro_accuracy": 0.04857142857142857
    },
    {
      "run_id": "shd-repeat-s8003",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.09505703422053231,
      "macro_accuracy": 0.09861111111111112
    },
    {
      "run_id": "shd-repeat-s8003",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.09919839679358718,
      "macro_accuracy": 0.09761904761904762
    },
    {
      "run_id": "shd-replay-s8000",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.0697084917617237,
      "macro_accuracy": 0.06739926739926741
    },
    {
      "run_id": "shd-replay-s8000",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.1402805611222445,
      "macro_accuracy": 0.15735270553432762
    },
    {
      "run_id": "shd-replay-s8001",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.10899873257287707,
      "macro_accuracy": 0.10393265672689948
    },
    {
      "run_id": "shd-replay-s8001",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.13627254509018036,
      "macro_accuracy": 0.17395278737384
    },
    {
      "run_id": "shd-replay-s8002",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.10519645120405577,
      "macro_accuracy": 0.09802555168408827
    },
    {
      "run_id": "shd-replay-s8002",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.06913827655310621,
      "macro_accuracy": 0.09701864458443406
    },
    {
      "run_id": "shd-replay-s8003",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.1520912547528517,
      "macro_accuracy": 0.153125
    },
    {
      "run_id": "shd-replay-s8003",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.14529058116232466,
      "macro_accuracy": 0.15896512113617378
    },
    {
      "run_id": "shd-sequential-s8000",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.051964512040557666,
      "macro_accuracy": 0.05
    },
    {
      "run_id": "shd-sequential-s8000",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.03707414829659319,
      "macro_accuracy": 0.05
    },
    {
      "run_id": "shd-sequential-s8001",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.051964512040557666,
      "macro_accuracy": 0.05
    },
    {
      "run_id": "shd-sequential-s8001",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.04308617234468938,
      "macro_accuracy": 0.055128205128205134
    },
    {
      "run_id": "shd-sequential-s8002",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.10266159695817491,
      "macro_accuracy": 0.09727296474311133
    },
    {
      "run_id": "shd-sequential-s8002",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.06513026052104208,
      "macro_accuracy": 0.08115384615384616
    },
    {
      "run_id": "shd-sequential-s8003",
      "speaker": 10,
      "n": 789,
      "accuracy": 0.050697084917617236,
      "macro_accuracy": 0.05
    },
    {
      "run_id": "shd-sequential-s8003",
      "speaker": 11,
      "n": 998,
      "accuracy": 0.0531062124248497,
      "macro_accuracy": 0.061111111111111116
    }
  ],
  "final_prediction_diagnostics": [
    {
      "run_id": "shd-joint-s8000",
      "arm": "joint",
      "predicted_class_counts": [
        278,
        43,
        225,
        52,
        5,
        0,
        131,
        68,
        53,
        3,
        48,
        101,
        30,
        39,
        193,
        123,
        36,
        65,
        62,
        232
      ],
      "number_of_predicted_classes": 19,
      "dominant_prediction_fraction": 0.15556799104644656
    },
    {
      "run_id": "shd-joint-s8001",
      "arm": "joint",
      "predicted_class_counts": [
        2,
        120,
        55,
        150,
        3,
        3,
        79,
        127,
        62,
        22,
        0,
        111,
        5,
        17,
        265,
        207,
        125,
        185,
        57,
        192
      ],
      "number_of_predicted_classes": 19,
      "dominant_prediction_fraction": 0.14829322887520985
    },
    {
      "run_id": "shd-joint-s8002",
      "arm": "joint",
      "predicted_class_counts": [
        72,
        49,
        57,
        158,
        7,
        0,
        90,
        70,
        65,
        16,
        66,
        93,
        178,
        1,
        31,
        159,
        64,
        285,
        9,
        317
      ],
      "number_of_predicted_classes": 19,
      "dominant_prediction_fraction": 0.17739227756015669
    },
    {
      "run_id": "shd-joint-s8003",
      "arm": "joint",
      "predicted_class_counts": [
        194,
        112,
        239,
        10,
        3,
        2,
        152,
        158,
        40,
        47,
        150,
        128,
        20,
        9,
        1,
        292,
        20,
        90,
        31,
        89
      ],
      "number_of_predicted_classes": 20,
      "dominant_prediction_fraction": 0.1634023503077784
    },
    {
      "run_id": "shd-repeat-s8000",
      "arm": "repeat",
      "predicted_class_counts": [
        0,
        0,
        0,
        883,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        904,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "number_of_predicted_classes": 2,
      "dominant_prediction_fraction": 0.5058757694459989
    },
    {
      "run_id": "shd-repeat-s8001",
      "arm": "repeat",
      "predicted_class_counts": [
        0,
        447,
        0,
        0,
        1340,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "number_of_predicted_classes": 2,
      "dominant_prediction_fraction": 0.7498601007274762
    },
    {
      "run_id": "shd-repeat-s8002",
      "arm": "repeat",
      "predicted_class_counts": [
        0,
        0,
        0,
        0,
        0,
        45,
        0,
        0,
        1742,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "number_of_predicted_classes": 2,
      "dominant_prediction_fraction": 0.9748181309457191
    },
    {
      "run_id": "shd-repeat-s8003",
      "arm": "repeat",
      "predicted_class_counts": [
        0,
        1501,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        286,
        0
      ],
      "number_of_predicted_classes": 2,
      "dominant_prediction_fraction": 0.8399552322327923
    },
    {
      "run_id": "shd-replay-s8000",
      "arm": "replay",
      "predicted_class_counts": [
        0,
        0,
        1,
        1341,
        0,
        0,
        77,
        0,
        0,
        0,
        0,
        0,
        72,
        276,
        0,
        0,
        0,
        20,
        0,
        0
      ],
      "number_of_predicted_classes": 6,
      "dominant_prediction_fraction": 0.7504196978175713
    },
    {
      "run_id": "shd-replay-s8001",
      "arm": "replay",
      "predicted_class_counts": [
        239,
        428,
        17,
        0,
        981,
        0,
        48,
        32,
        0,
        0,
        0,
        0,
        23,
        0,
        0,
        0,
        6,
        13,
        0,
        0
      ],
      "number_of_predicted_classes": 9,
      "dominant_prediction_fraction": 0.548964745383324
    },
    {
      "run_id": "shd-replay-s8002",
      "arm": "replay",
      "predicted_class_counts": [
        0,
        0,
        13,
        14,
        0,
        187,
        241,
        0,
        1310,
        22,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "number_of_predicted_classes": 6,
      "dominant_prediction_fraction": 0.7330721880246223
    },
    {
      "run_id": "shd-replay-s8003",
      "arm": "replay",
      "predicted_class_counts": [
        0,
        891,
        7,
        0,
        0,
        0,
        582,
        0,
        0,
        0,
        10,
        0,
        4,
        0,
        0,
        0,
        0,
        0,
        293,
        0
      ],
      "number_of_predicted_classes": 6,
      "dominant_prediction_fraction": 0.4986010072747622
    },
    {
      "run_id": "shd-sequential-s8000",
      "arm": "sequential",
      "predicted_class_counts": [
        0,
        0,
        0,
        1774,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        13,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "number_of_predicted_classes": 2,
      "dominant_prediction_fraction": 0.9927252378287633
    },
    {
      "run_id": "shd-sequential-s8001",
      "arm": "sequential",
      "predicted_class_counts": [
        0,
        61,
        0,
        0,
        1726,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "number_of_predicted_classes": 2,
      "dominant_prediction_fraction": 0.965864577504197
    },
    {
      "run_id": "shd-sequential-s8002",
      "arm": "sequential",
      "predicted_class_counts": [
        0,
        31,
        0,
        0,
        434,
        0,
        0,
        818,
        0,
        346,
        3,
        0,
        0,
        0,
        0,
        0,
        12,
        135,
        0,
        8
      ],
      "number_of_predicted_classes": 8,
      "dominant_prediction_fraction": 0.45775041969781755
    },
    {
      "run_id": "shd-sequential-s8003",
      "arm": "sequential",
      "predicted_class_counts": [
        0,
        1543,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        244,
        0
      ],
      "number_of_predicted_classes": 2,
      "dominant_prediction_fraction": 0.863458310016788
    }
  ],
  "total_training_run_wall_seconds": 2048.2684433349395,
  "total_exposures": 178332,
  "logical_replay_frame_bytes": 56000000,
  "source_verification_note": "All 483 original export blobs matched the pinned GitHub tree before cache loss; all five imported files were reverified after recovery. Every run's imported hashes match this audit. Immutable start logs precede the fresh audit.",
  "runs": [
    {
      "status": "complete",
      "seed": 8000,
      "arm": "joint",
      "final_accuracy": 0.38276440962506997,
      "final_macro_accuracy": 0.3809483561605395,
      "positive_control_gate_pass": true,
      "wall_seconds": 165.68107412499376,
      "peak_rss_bytes": 418414592,
      "updates": 798,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 12738,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "run_id": "shd-joint-s8000"
    },
    {
      "status": "complete",
      "seed": 8001,
      "arm": "joint",
      "final_accuracy": 0.46278679350867374,
      "final_macro_accuracy": 0.45495070869162574,
      "positive_control_gate_pass": true,
      "wall_seconds": 152.21713004197227,
      "peak_rss_bytes": 493961216,
      "updates": 798,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 12738,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "run_id": "shd-joint-s8001"
    },
    {
      "status": "complete",
      "seed": 8002,
      "arm": "joint",
      "final_accuracy": 0.4202574146614438,
      "final_macro_accuracy": 0.4144185961387293,
      "positive_control_gate_pass": true,
      "wall_seconds": 132.90552500000922,
      "peak_rss_bytes": 470302720,
      "updates": 798,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 12738,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "run_id": "shd-joint-s8002"
    },
    {
      "status": "complete",
      "seed": 8003,
      "arm": "joint",
      "final_accuracy": 0.3928371572467823,
      "final_macro_accuracy": 0.3961666393390881,
      "positive_control_gate_pass": true,
      "wall_seconds": 142.76681654196,
      "peak_rss_bytes": 514064384,
      "updates": 798,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 12738,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "run_id": "shd-joint-s8003"
    },
    {
      "status": "complete",
      "seed": 8000,
      "arm": "repeat",
      "final_accuracy": 0.05987688864017907,
      "final_macro_accuracy": 0.0573076923076923,
      "positive_control_gate_pass": null,
      "wall_seconds": 147.69313966599293,
      "peak_rss_bytes": 484278272,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 6369,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        0.8,
        0.7839506172839507,
        0.5642458100558659,
        0.546448087431694,
        0.5480225988700564,
        0.9894736842105263,
        0.98,
        0.8388888888888889,
        0.9606741573033708
      ],
      "mean_forgetting": 0.7790782048938171,
      "run_id": "shd-repeat-s8000",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0,
      "final_latest_class_macro_accuracy": 0.573076923076923
    },
    {
      "status": "complete",
      "seed": 8001,
      "arm": "repeat",
      "final_accuracy": 0.07498601007274762,
      "final_macro_accuracy": 0.08435126582278481,
      "positive_control_gate_pass": null,
      "wall_seconds": 134.9932218340109,
      "peak_rss_bytes": 683327488,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 6369,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        1.0,
        0.49238578680203043,
        0.45054945054945056,
        0.49122807017543857,
        0.8465909090909091,
        0.8846153846153846,
        0.43575418994413406,
        0.945054945054945,
        0.9888268156424581
      ],
      "mean_forgetting": 0.7261117279860834,
      "run_id": "shd-repeat-s8001",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0,
      "final_latest_class_macro_accuracy": 0.843512658227848
    },
    {
      "status": "complete",
      "seed": 8002,
      "arm": "repeat",
      "final_accuracy": 0.042529378847229994,
      "final_macro_accuracy": 0.04871794871794872,
      "positive_control_gate_pass": null,
      "wall_seconds": 167.71105049998732,
      "peak_rss_bytes": 449773568,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 6369,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        0.43575418994413406,
        0.989010989010989,
        0.4666666666666667,
        0.6022099447513812,
        1.0,
        0.4444444444444444,
        0.8070175438596491,
        0.5604395604395604,
        0.5251396648044693
      ],
      "mean_forgetting": 0.647853667102366,
      "run_id": "shd-repeat-s8002",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0,
      "final_latest_class_macro_accuracy": 0.48717948717948717
    },
    {
      "status": "complete",
      "seed": 8003,
      "arm": "repeat",
      "final_accuracy": 0.09736989367655288,
      "final_macro_accuracy": 0.09797979797979797,
      "positive_control_gate_pass": null,
      "wall_seconds": 99.14446637499998,
      "peak_rss_bytes": 742653952,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 6369,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        1.0,
        0.5681818181818182,
        0.9253731343283582,
        0.6722222222222223,
        0.5024875621890548,
        0.7758620689655172,
        0.7653631284916201,
        1.0,
        0.9741935483870968
      ],
      "mean_forgetting": 0.798187053640632,
      "run_id": "shd-repeat-s8003",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0,
      "final_latest_class_macro_accuracy": 0.9797979797979798
    },
    {
      "status": "complete",
      "seed": 8000,
      "arm": "replay",
      "final_accuracy": 0.10912143256855064,
      "final_macro_accuracy": 0.11576312446053619,
      "positive_control_gate_pass": null,
      "wall_seconds": 166.56652508297702,
      "peak_rss_bytes": 509460480,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 6353,
      "repeat_exposures": 16,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        1.0,
        0.6296296296296297,
        0.9608938547486033,
        0.546448087431694,
        0.7457627118644068,
        0.7894736842105263,
        0.98,
        0.7888888888888889,
        0.7528089887640449
      ],
      "mean_forgetting": 0.7993228717264215,
      "run_id": "shd-replay-s8000",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0547368049561513,
      "final_latest_class_macro_accuracy": 0.665
    },
    {
      "status": "complete",
      "seed": 8001,
      "arm": "replay",
      "final_accuracy": 0.1242305540011192,
      "final_macro_accuracy": 0.13858101763369424,
      "positive_control_gate_pass": null,
      "wall_seconds": 147.73538941703737,
      "peak_rss_bytes": 703004672,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 6353,
      "repeat_exposures": 16,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        1.0,
        0.7969543147208122,
        0.7857142857142858,
        0.7426900584795322,
        0.5625,
        0.7967032967032968,
        0.43575418994413406,
        0.9615384615384616,
        0.7486033519553073
      ],
      "mean_forgetting": 0.7589397732284255,
      "run_id": "shd-replay-s8001",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.07991069469847883,
      "final_latest_class_macro_accuracy": 0.6666139240506329
    },
    {
      "status": "complete",
      "seed": 8002,
      "arm": "replay",
      "final_accuracy": 0.08505875769445999,
      "final_macro_accuracy": 0.09703817573546487,
      "positive_control_gate_pass": null,
      "wall_seconds": 154.21222979197046,
      "peak_rss_bytes": 495501312,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 6353,
      "repeat_exposures": 16,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        0.7430167597765364,
        0.9780219780219781,
        0.8333333333333334,
        0.9060773480662982,
        0.9384615384615385,
        0.4777777777777778,
        0.7660818713450293,
        0.5824175824175825,
        0.5810055865921788
      ],
      "mean_forgetting": 0.7562437528658059,
      "run_id": "shd-replay-s8002",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.04412717531860771,
      "final_latest_class_macro_accuracy": 0.5732371794871794
    },
    {
      "status": "complete",
      "seed": 8003,
      "arm": "replay",
      "final_accuracy": 0.14829322887520985,
      "final_macro_accuracy": 0.1557506954891707,
      "positive_control_gate_pass": null,
      "wall_seconds": 85.1532366670001,
      "peak_rss_bytes": 875003904,
      "updates": 806,
      "exposures": 12738,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 6353,
      "repeat_exposures": 16,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        1.0,
        0.7727272727272727,
        0.8009950248756219,
        0.9611111111111111,
        0.8308457711442786,
        0.8390804597701149,
        0.7653631284916201,
        0.9726775956284153,
        0.45806451612903226
      ],
      "mean_forgetting": 0.8223183199863852,
      "run_id": "shd-replay-s8003",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.06250638443802353,
      "final_latest_class_macro_accuracy": 0.994949494949495
    },
    {
      "status": "complete",
      "seed": 8000,
      "arm": "sequential",
      "final_accuracy": 0.04364857302742026,
      "final_macro_accuracy": 0.05,
      "positive_control_gate_pass": null,
      "wall_seconds": 123.49545545800356,
      "peak_rss_bytes": 459915264,
      "updates": 403,
      "exposures": 6369,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        0.79375,
        0.7283950617283951,
        0.5363128491620112,
        0.546448087431694,
        0.6271186440677966,
        0.9473684210526315,
        0.505,
        0.7944444444444444,
        1.0
      ],
      "mean_forgetting": 0.7198708342096637,
      "run_id": "shd-sequential-s8000",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0,
      "final_latest_class_macro_accuracy": 0.5
    },
    {
      "status": "complete",
      "seed": 8001,
      "arm": "sequential",
      "final_accuracy": 0.047006155567991044,
      "final_macro_accuracy": 0.052531645569620256,
      "positive_control_gate_pass": null,
      "wall_seconds": 79.26216045900946,
      "peak_rss_bytes": 524648448,
      "updates": 403,
      "exposures": 6369,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        1.0,
        0.8527918781725888,
        0.945054945054945,
        0.7719298245614035,
        0.7840909090909091,
        0.44505494505494503,
        0.43575418994413406,
        0.8791208791208791,
        0.994413407821229
      ],
      "mean_forgetting": 0.7898012198690036,
      "run_id": "shd-sequential-s8001",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0,
      "final_latest_class_macro_accuracy": 0.5253164556962026
    },
    {
      "status": "complete",
      "seed": 8002,
      "arm": "sequential",
      "final_accuracy": 0.0817011751538892,
      "final_macro_accuracy": 0.08927076359725736,
      "positive_control_gate_pass": null,
      "wall_seconds": 86.21134129201528,
      "peak_rss_bytes": 506347520,
      "updates": 403,
      "exposures": 6369,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        0.5586592178770949,
        0.8406593406593407,
        0.5388888888888889,
        0.9337016574585636,
        0.5025641025641026,
        0.2666666666666666,
        0.8128654970760234,
        0.5604395604395604,
        -0.3687150837988827
      ],
      "mean_forgetting": 0.5161922053145953,
      "run_id": "shd-sequential-s8002",
      "forgetting_summary_clipping_difference": 0.3687150837988827,
      "final_old_class_macro_accuracy": 0.09918973733028595,
      "final_latest_class_macro_accuracy": 0.0
    },
    {
      "status": "complete",
      "seed": 8003,
      "arm": "sequential",
      "final_accuracy": 0.05204252937884723,
      "final_macro_accuracy": 0.05707070707070707,
      "positive_control_gate_pass": null,
      "wall_seconds": 62.51968108300002,
      "peak_rss_bytes": 642678784,
      "updates": 403,
      "exposures": 6369,
      "unique_train_examples": 6369,
      "current_exposures": 6369,
      "replay_exposures": 0,
      "repeat_exposures": 0,
      "official_test_read": false,
      "scope": "fixed-width audio component, not full engine",
      "forgetting_by_task": [
        1.0,
        0.7556818181818182,
        0.681592039800995,
        0.5388888888888889,
        0.5024875621890548,
        0.6666666666666666,
        0.5586592178770949,
        1.0,
        0.8451612903225807
      ],
      "mean_forgetting": 0.7276819426585666,
      "run_id": "shd-sequential-s8003",
      "forgetting_summary_clipping_difference": 0.0,
      "final_old_class_macro_accuracy": 0.0,
      "final_latest_class_macro_accuracy": 0.5707070707070707
    }
  ],
  "frozen_initialization_macro_mean_pct": 5.4443775281501186,
  "frozen_initialization_macro_sd_pp": 0.8887550563002371,
  "paired_mean_pp": 5.4694077122660545,
  "paired_sd_pp": 0.46352338493023615
}
```


# SHD operational deviations and metric correction

Original workspace member: `planning/experiments/scale/DEVIATIONS.md`

# Operational deviations

All entries concern storage/runtime, not model selection.

1. The first archive in Documents became `compressed,dataless` and a hash read timed out. The failed preparation log is retained. A recovered local copy matched the official MD5 exactly. Dataset cache now lives in `/private/tmp/sndl-scale-data-20261103`.
2. The first training launch stalled before model initialization on Documents/Git pack reads and was terminated without model exposures. It created no scientific run record.
3. Relaunch uses `/private/tmp/sndl-class-source-tlefts_z`, the unchanged export previously verified against all 483 Git blobs at commit `c4c10b285f1741aa798d5ee9c5296ae371fabecf` (prior receipt: `planning/reports/05-artifacts/class-source-verification.json`). A fresh Git check is blocked by dataless pack reads. The current runner asserts the export's provenance and logs SHA256 of every imported package source. It explicitly records `fresh_git_verification: false`. This is source relocation, not replacement of the learning rule. No upstream source files were edited.
4. Local bytecode cache avoids the earlier Python module-loading timeout. Runtime receipts show three successful fresh-process CPU tensor checks with the GIL off.

5. Source-verification blocker resolved without changing the checkout: queried GitHub's recursive tree for the exact pinned commit through the read-only GitHub connector, saved its 483 source/tool blob IDs, and freshly verified the complete local export. All 483 match. `reports/source-audit.json` and `verify_source.py` reproduce the audit. Start logs correctly retain `fresh_git_verification: false` because those logs precede this audit; analysis verifies every run's imported hashes against the fresh audit. Local Git pack reads remain unreliable, but source provenance is now freshly verified.

6. During the final seed, available disk fell to 640 MiB, then worker processes disappeared and every `/private/tmp/sndl-*` directory was absent. The cause of the apparent local reset is unconfirmed. Thirteen completed runs survived in the workspace. The partial sequential seed-8003 attempt is retained under `runs/interrupted-shd-sequential-s8003` with its last recorded counters and prediction files. Before retry, all exact wheel archives were hash-verified and reinstalled; the five imported source files were recovered through authenticated GitHub blob reads and verified against the same Git blobs. Every regenerated data-array hash and raw/archive hash matched the original. The original manifest bytes were restored. Same seed, code, hyperparameters, and stream will be restarted because no learned-model checkpoint was claimed or persisted. Retry is for external infrastructure loss, not a low result. Shared completed prediction checkpoints will be compared after retry. This interrupted attempt is not an extra independent seed.

7. Independent final analysis found a secondary summary-metric error: the collector included the final checkpoint when finding a task's best earlier accuracy, clipping positive backward transfer to zero. The analyzer recomputes the registered best-earlier-minus-final forgetting value directly from the raw task matrices, allowing negative values for improvement. Original collector summaries and frozen runner code remain unchanged for provenance. Classification accuracy, prediction records, all matched comparisons and training are unaffected. The canonical HTML/JSON report uses the corrected secondary values and records the difference per run.
8. Matplotlib's system-font scan stalled. Plotting now uses its documented `MPL_IGNORE_SYSTEM_FONTS=1` option and bundled DejaVu fonts. Final post-recovery Torch checks passed three fresh processes in 0.76–1.11 seconds with the GIL off. The working bytecode cache and exact wheel/source/archive recovery inputs are retained outside `/private/tmp`; no learning code or numerical hyperparameter changed.


# SHD closing verification

Original workspace member: `planning/experiments/scale/reports/closing-verification.json`

```json
{
  "completed_runs": 16,
  "independently_rescored_checkpoints": 144,
  "complete_cohort_exposures": 178332,
  "independent_seeds": 4,
  "infrastructure_interruptions_retained": 1,
  "offline_scoring_metadata_verified": true,
  "no_official_test_used": true,
  "retry_shared_checkpoints_bitwise_equal": 9,
  "upstream_git_status": {
    "status": "timeout",
    "seconds": 10,
    "source_integrity": "Fresh pinned GitHub blob audit and run hashes used instead; no source edits performed"
  },
  "root_shims": [],
  "cache_recovery_helper_verified": true,
  "prepared_cache_restore_helper_verified": true,
  "persistent_bytecode_cache": true,
  "no_active_experiment_workers": true
}
```


# Engine integration protocol and dated amendments

Original workspace member: `repo/specs/022-unified-brain-arc/evidence/g9-engine-cycle-integration.md`

# G9 canonical engine cycle integration

## 1. Scope and state found

Prepared 2026-11-04, America/Chicago, on c4c10b285f1741aa798d5ee9c5296ae371fabecf.
The owner explicitly requested the first actual canonical grow/retain/restore cycle after the
SHD audio-component pilot. This is an integration build and engineering acceptance protocol,
not a confirmatory G3/G9 campaign or a replacement of their frozen primary endpoints.

The existing `GrowthLoop` refuses a canonical adapter unconditionally. Ordinary G9 admission,
Part D readout/return and restart carriers exist. Their manually supplied populations do not
establish automatic earned growth or improvement. The Part D label reader and the canonical
Brain chat learner are different consumers; a capacity certificate for one cannot license a
claim about the other. An initial brain-free architectural diagnostic of the Part D code was
performed during design; it is a diagnostic witness, not trained performance or a campaign.

Incumbents adopted: `candidate_lifecycle` scoring/selection, `growth_loop.trace_train`,
`grown_code` first-spike units and local return rule, `capacity_certificate`, the canonical
`SpikingInteractionLearner`, `PosedActor`, the common R2 posed event writer/replay, G7 admission,
`RetentionSets`/`Meter`, the problem ledger and `make_brain`. No second Brain, learning rule,
private learned-state file, or scripted solve outcome is permitted.

## 2. Pre-implementation protocol

### 2.1 Actual consumer and curriculum

The final decision for this integration is the actual shared canonical Brain chat learner.
The real grown-to-pool/readout/return path must execute, and joint teaching must update the
canonical learner through its incumbent local rule, under the same journal event used for
live effects and replay. The old Part D comparator remains unchanged.

Use the already registered disjoint-word `aii` as old task A and `ii.xor` as task B, with their
registered label order and episode generators. `ii.twin` relabels the same words as `ii.xor`
and is not an old-skill retention curriculum. Problem identity selects the existing posed
context; this first trajectory is task-conditioned and does not claim uncued composition.

Only training data may fit candidate inputs/readouts. Selection reads validation and old-task
retention-select sets, with full reversible state. Untouched final sets are judged once after
the chosen state is durably frozen. Labels are feedback/judging metadata, never input features.

### 2.2 Growth, readiness and budgets

The diagnostic must read the actual current canonical code, including grown units and causal
return afferents. A separator is accepted only through the real cloned reader. Candidate
training adopts the existing local trace rule and residual-scoring lifecycle. New units require
actual nonconstant contributions, G7 labels, write-ahead births, charged search cost, and real
feed/return descriptors. Do not hand-select an oracle unit or use zero feed to claim a benefit.

The generic readiness refusal remains for unbound/incomplete configurations. A supported
integration configuration may open only when its actual owner, source, selection/retention
consumer and durable continuation are bound and checked. Unsupported problem families,
stateful-growth claims and general pruning/search modes remain closed until implemented.

The initial first-budget rule and finite pause behavior remain unchanged. A sequence requiring
an unset source-owned BETA refuses; no arbitrary global BETA is installed. A zero-spend old
task may precede a first positive-spend task under the existing rule. Any later positive-spend
sequence needs its separately dated development calibration.

### 2.3 Retention and durable final judgment

Reuse `RetentionSets` for learning-OFF selection/final reads and complete state restoration.
The frozen winner has a source-owned journal identity and actual-state fingerprint. Foreign
meters, changed winners/models and pre-freeze judgment refuse before consuming data. Final
old-task consumption is reserved durably BEFORE the first read. A partial read, failed read,
ordinary rollback, same-Brain rebind, checkpoint or new process cannot re-arm it. Replaying a
reservation restores protocol state and never re-reads held-out examples.

The first all-active/no-prune selection is an explicit integration control. It does not claim
a G6 pruning benefit, an all-plastic-circuit search, or close the registered G6 campaign. Any
supported mask mutation must act on the actual circuits it names and retain old tasks as a
hard feasibility constraint.

### 2.4 Acceptance conjuncts and reachable failures

1. Actual canonical grow loop: start from the registered initial state, draw/train/select units
   through the incumbent lifecycle, admit exactly those IDs and run actual teaching attempts.
   A fake source, wrong owner, missing projection, unearned/constant contribution, missing birth,
   bad ticket or exceeded budget must refuse. Log real units, exposures, attempts and cost.
2. Actual behavior: score the same final canonical reader before/after. A no-growth/refit control
   and grown/feed/return interventions identify which channel changes which answer. An absent
   effect is recorded as absent; no generic architecture benefit is claimed from construction.
3. Old skill: A is learned before B, evaluated learning OFF, and preserved within its registered
   retention bound after B. A planted damaging state/mask must be rejected by retention-select;
   otherwise the retention apparatus is insensitive and no retention claim is read.
4. Stable acquisition and accounting: use the existing stable-epoch solved contract, record actual
   examples/epochs, and do not rename one final correct read as stable learning. A paused task
   remains OPEN/UNSOLVED with its budget and cursor.
5. Restart: exit the writer, build a new canonical Brain over the same isolated lake, re-read A/B
   without teaching or regrowth, and check state/continuation and durable one-shot judgment.
   A missing/corrupt carrier and re-seeded continuation must fail the corresponding checks.

### 2.5 Host, seeds and analysis scope

Local Apple Silicon CPU, CPython 3.14 free-threaded with the GIL off. Native and dependency
builds, runtime caches and outputs stay outside the checkout. All Brain tests use isolated
homes, one Brain worker at a time, named tests, no coverage and bounded parent/child deadlines.
No live home, daemon, arm, safety envelope or policy certification is touched.

Sample size (P1): this is deterministic engineering acceptance using the already exposed
fixture seed 9900, not an independent statistical cohort, superiority/null reading or powered
claim. Existing registered campaign seeds are not opened. Any scientific comparison beyond
these constructed controls requires a new pilot/power/sequential design under P1.

### 2.6 Implementation and verification order

1. Verify the original isolated runtime and focused ordinary G9 controls.
2. Extend actual-code diagnosis and joint canonical learning/read/replay.
3. Add durable freeze/retention consumption with malformed-state and crash/rebind controls.
4. Bind actual candidate/source and selection consumers; retain unsupported readiness refusals.
5. Run the complete A-to-B-to-fresh-process trajectory, instrument/fault controls, focused old
   regressions, lint and per-module checks; independently review before stating a result.

Changes remain on the implementation branch, not on main. No operator battery or merge is claimed.
Amendments are appended and dated; historical evidence and the SHD pilot remain unchanged.

## 3. Execution record

Bounded engineering acceptance04 passed on2026-11-04, using the actual isolated persisted
`make_brain(9900)`, the common writer and real PartD/feed/return/canonical readout path.
Raw commands, exact source overlays/hashes, frozen recipes, lake and result receipts are kept
under the local `planning/experiments/engine-integration` workspace directory. The operator
battery and a merge to main were not performed.

| Measured event | Actual result |
| --- | --- |
| Acquire old Aii with enabled canonical stack and explicit corrective feedback | 22epochs,88teaching presentations,66base-stack deposits; stable10 |
| Diagnose B and grow via the actual source/G7 writer | CAPACITY;1024candidate evaluations; one admitted18-unit batch, IDs0–17 |
| Learn ii.xor with the learned core protected and grown outputs plastic | 107epochs,428teaching presentations; stable10 at epoch ends98–107;36nonzero grown-output weights |
| Old/new validation and once-only future judgments | A16/16 and B16/16; frozen selection and both consumption claims journaled before final reads |
| Protected core | Entire saved canonical neuron/stack state exactly equal to A's endpoint; zero B base-stack deposits |
| Planted old-feature lesion | A0/16 while B's decisions unchanged; feasibility refused; full state restored |
| Actual grown units silenced | B8/16; A16/16 |
| Actual return conductances zeroed | B8/16 |
| Actual feed conductances zeroed | B16/16; pool features change, L2difference32.68026927673638; no final-answer benefit established for feed |
| Fresh-process checkpoint and full-journal recovery | Exact model digest, all actor carriers and A/B predictions; repeated final read refused; boot heldout reads forbidden |

Each task contains FOUR unique constructed texts;16presentations are four repetitions in a
frozen order. The future sets reorder the SAME items. These counts are not16independent examples
or unseen-semantic generalization. This is one already exposed fixture, explicit task identity,
corrective feedback, protected core and all-active/no-prune selection. It is not a SHD result,
powered architecture comparison, entirely plastic-core solution or general recursive improvement.

An independent reader rescored all129recorded epoch ends, all validation/final/intervention
answers and the source accounting from raw receipts/lake bytes. It confirmed these results.
The genuine all-plastic negative control acquired B but forgot A (0/16); final selection refused
and no final set was consumed. That negative and the failed initialization/intrinsic/rehearsal
controls remain in the record. One checkpoint-controller assertion failed only because it
compared tuples with JSON lists; the exact serialized predictions matched, and the corrected
comparison passed in a new process. No neural values were adjusted.

Final focused checks:120growth/budget/certificate/codec/retention-set/PRE tests plus95Brain/
posed/admission/eligibility/protocol/mask-search regressions passed. Five bounded module
selfchecks passed; Ruff0.15.20 and targeted gradient-claim declaration checks passed. No new
gradient algorithm or baseline relaxation was introduced. Final declaration-only additions
are checked against acceptance source with exact AST equality after removing only the
`GRADIENT_CLAIM` constants; numerical bodies are identical.

Remaining debts: source-owned BETA for a later positive-spend task, broader problem families,
uncued composition, learned-core plasticity without forgetting, competitive pruning/sleep
claims, full-Engine SHD, independent-seed efficacy and the full operator battery. Their gates
are not released by this integration result.

## 4. Engineering amendment: canonical stack blockers (2026-11-04)

This amendment follows exposed-fixture diagnostics; it is not a pre-data scientific efficacy
registration. The owner authorized fixing full-Engine blockers, including the actual canonical
consumer and the ability to acquire/retain its first skill. Existing #2135's requirement to repair
the enabled E/I/Benna-Fusi/homeostatic stack remains applicable; no release is inferred.

Initial Aii on fixture9900 fails even over the existing three400-epoch attempts: best/final8/16,
no all-correct epoch. Removing one faculty or preinitializing its state is diagnostic only and
does not supply a successful canonical curriculum. B's separate learned-growth feasibility does
not establish old-skill retention. All attempts and raw traces are retained.

Confirmed source defect: cold BF initialization occurs after the first synaptic update, so that
update becomes the baseline of every deep beaker. The homeostatic ceiling is likewise described
as a baseline but calibrated after the first update. Repair the PRE baseline boundary in the
existing `plasticity_rule_stack`, `homeostatic_plasticity_stack`, and `interaction_learning`
consumer, without resetting any warm/restored state, deleting a faculty, or inventing a new
learning rule. Tests must show a first deposit reaches fast state and diffuses normally instead
of instantly populating all slow levels, and a warm continuation is unchanged. This defect fix
alone has already failed to establish stable Aii in exposed diagnostic controls and is not
represented as a sufficient learning repair.

Further diagnosis may test the existing intrinsic-plasticity participation/clock seam:
currently its learned gain/bias is not consumed by the final reader, and activity faculties see
reward-deposited rows rather than all responses once per presentation. Freeze each disclosed
engineering variant before its outcomes, compare raw/unconsumed and label-independent all-row
observation plus actual transfer consumption, retain all failures, and keep validation pure.
Do not silently insert a sigmoid into only trained rows, hand-install separating weights,
change the old task, or relabel a transient correct read as stable acquisition. Promote any
additional source repair only after a causal control, restore proof and explicit evidence of
its behavior; broad G2/G3/G9 efficacy and the full operator battery remain unverified.

### 4.1 Explicit corrective-feedback engineering mode

The exposed teaching-policy diagnostic found Aii stable at22epochs/88presentations with33actual
teaching calls when already-correct answers received no further synaptic reward. The original
always-positive control remained unstable at1200epochs. This is post-diagnostic motivation,
not a primary-endpoint or untouched-data finding. Pure cache clearing and snapshot restore did
not change that diagnostic's decisions.

Add an explicitly bound optional `feedback_gate="corrective"` to the engineering cycle recipe;
omission preserves the existing always-positive path. The common joint event first executes
the real Part D/readout/return path, then reads the actual shared canonical decision. If that
decision already matches the teaching feedback, its shared readout eligibility is zero for
that correction; otherwise the unchanged incumbent local R-STDP update applies. Every input,
feedback/cache event, pool/readout/return effect, epoch and actual shared plasticity deposit
is still accounted for and replays through the same event body. No answer is looked up from
teaching authority, no fitted weights are installed, and no unseen label enters inference.

This is a disclosed feedback policy, not the original always-positive G3 campaign. Evaluate
A-to-B and fresh-process recovery under this frozen mode, retaining both positive and failed
trajectories. All-active selection must still require actual stable acquisition and old-task
feasibility; changing the feedback policy does not waive either requirement. Do not promote
experimental intrinsic-transfer wiring, which failed its own stability diagnostic.

### 4.2 Separate eligibility for a protected core and learned growth

The actual full-Brain corrective trajectory acquired A in22epochs and B in88epochs after one
18-unit admission, but A then scored0/16. The retention feasibility gate refused selection;
no untouched final set was consumed. A separate prospective fixed rehearsal control also failed
to produce simultaneous A/B stability. These negative trajectories remain part of the record.

The existing whole-response neuron mask returns before the grown-output deposit, so it cannot
protect an old shared core while learning new grown synapses. Add explicit optional
`core_plasticity="frozen"` to the B engineering recipe and derive a typed `core_eligible` response
mask in its actual grown context. The ordinary neuron mask continues to govern the WHOLE
response. A false core-only eligibility skips the old52-synapse row and its stack state while
permitting the existing local `chat_grown_deposit` on admitted output synapses, with its own
causal eligibility and raw scalar feedback. No new synaptic rule is introduced. A's original
all-faculty learning remains enabled; only the already learned core is protected during B.
The capacity LP must hold ineligible core rows at their actual weights as well as honoring
grown-output eligibility; acceptance still requires real cloned-reader answers.

The disclosed exposed brain-free eligibility control retained A throughout B and reached
joint stable10 at127Bepochs with18grown units, while the whole core/stack stayed exactly equal
to A's endpoint. This motivates the prospective full-Brain test; it is not that test's result.
Run the actual candidate/PartD/feed/return/journal path, old-skill and planted-damage feasibility,
and a fresh process before claiming this bounded grow/retain/restore cycle. Record zero core
deposits separately from actual grown synaptic changes. No claim is made about an entirely
plastic shared core, uncued task composition, optimal size, pruning, or general RSI.


# Engine actual execution ledger

Original workspace member: `planning/experiments/engine-integration/LEDGER.md`

# Engine cycle integration ledger

Started November 4, 2026, America/Chicago.

Owner request: begin fixing the previously identified full-Engine blockers; first complete a canonical task cycle with real growth, preservation of an earlier skill, and recovery in a fresh process. This authorizes the concrete source integration described in the preceding explanation (growth_loop, actual Brain/posed consumers, retention selection, task/budget interfaces) and necessary focused tests. No live lake, envelope, policy certification, publishing, or main merge is in scope.

Source found: clean detached c4c10b285f1741aa798d5ee9c5296ae371fabecf. Work branch: engine-cycle-integration (owner preference: no assistant prefix).

Mode S: local shell and bounded tests. Read AGENTS, SECURITY, POLICIES in full and cold-start protocol. New code must adopt existing mechanisms; preserve actual full-state replay, local learning, finite budgets and guards. No synthetic solve outcomes.

Initial findings:
- Full canonical GrowthLoop refuses via unconditional G9_A_UNREADY.
- Existing ordinary adapter tests exercise manual admission/Part D state; the actual canonical final answer requires shared chat learner participation as well as reservoir/readout/return.
- Canonical constructor only accepts ii.xor/ii.twin; those share texts with different labels. For retention use already registered aii (disjoint words), then ii.xor. Do not treat relabelling the same words as forgotten skill.
- Current capacity diagnosis ignores grown/return context. Its target must be the actual final decision circuit.
- Full Brain dependencies and native module were absent from audio runtime. Runtime audit recovered isolated exact dependencies and built native outside the checkout; baseline G9 tests must be independently reproduced before code claims.

Ownership:
- Root: growth_loop, canonical task scope, source/search binding, end-to-end acceptance, registration/evidence/closing.
- Retention worker: posed_problem/Brain joint actual consumer, durable selection/once-only retention; focused tests.
- Growth worker: capacity_certificate full current code + real-reader separator verification; focused tests.
- Runtime worker: outside-source environment/build/diagnostics; no production code.

Next action: freeze integration protocol, then implement each missing port and run failing/positive controls on the actual isolated Brain. One Brain worker at a time. Record incomplete modes rather than opening readiness globally.

Implementation/checkpoints:
- Registered source protocol in d31e68c3 before implementation. New source work stays on engine-cycle-integration.
- Isolated complete free-threaded runtime rebuilt; root independently reproduced 3 named original canonical admission/return/restart controls (7.21 s pytest). Runtime worker's original 31-test baseline also retained.
- Joint engine_teach/engine_read now use the actual shared canonical learner and existing Part D pool/return under one common event. New strict v2 selection/claim carriers reserve each current/old untouched read before execution; checkpoint/full-prefix replay restores authority/ring/protocol facts without re-reading heldout data.
- Retention worker bounded tests: 8 new controls passed, 14 original adapter controls passed. Root: 69 original growth/budget controls passed. These are engineering tests, not growth efficacy results.
- Candidate feasibility revisions, all retained outside source: revision 1 used identical random feedback vector across candidates and was negative; revision 2 corrected to independent derived feedback seeds and actual lifecycle select_block. Width9 did not attain stable10 over1200 epochs. Exposed brain-free width18/pool1024 first-growth-only probe attained stable10 at704 epochs, with grown-off8/16 versus grown-on16/16. This is source feasibility without the full Brain/return or old skill, not a full-Engine result.
- First actual full-Brain A phase (cycle-old-controller.json) stopped before teaching: capacity separator deep-copied writer-bound callbacks into Brain's RLock. Retained raw failure and isolated home; clone fix assigned to certificate owner. No solve or retention claim was made.
- Separate initial Aii diagnosis at400 epochs failed to learn the old task. Extend to the registered three400-epoch attempts and diagnose enabled canonical stack before proceeding. An unlearned task cannot establish retention. Existing #2135 stack-repair debt remains open.

## Completed bounded cycle04

The actual persisted full Brain passed the first grow/retain/restart integration on the registered Aii→ii.xor curriculum, exposed seed9900. This required explicitly declared corrective feedback and separate eligibility for the learned core and grown outputs. The all-plastic control is retained as a genuine failure: B16/16, old A0/16, selection refused before final data.

| Stage/control | Observed result |
| --- | --- |
| Old A acquisition | 22epochs,88teaching presentations; stable10; zero grown units |
| B source/admission | 1024candidate evaluations,18actual G7-admitted units, IDs0–17, within52first budget |
| B acquisition | 107epochs,428teaching presentations; stable10 at98–107 |
| Validation and untouched once-only final | Both A and B16/16 |
| Protected core | Exact old neuron/stack state, zero additional base-stack deposits;36learned nonzero grown output weights |
| Old-feature lesion | Old A0/16 while B unchanged; feasibility refused; full snapshot restored |
| Grown outputs silenced | B8/16, A16/16 |
| Return conductances zero | B8/16 |
| Feed conductances zero | B16/16; real pool features change (L2Δ32.68026927673638); answer benefit not demonstrated |
| Fresh checkpoint and journal-only processes | Exact states/digest/predictions; final consumption cannot be re-armed; boot cannot read heldout examples |

Each task has only FOUR unique constructed texts, repeated four times per evaluation. Future order differs, items do not. This is an engineering acceptance case, not a large-data accuracy claim, SHD/full-Engine result, statistical comparison, general RSI result or all-plastic-core solution. The task context is supplied explicitly. No pruning benefit is claimed.

Evidence:
- `cycle-04-old.json`, `cycle-04-grow.json`: raw model/actor states, responses, costs and interventions.
- `cycle-04-grow-restore-2.json`, `cycle-04-grow-restore-journal.json`: fresh-process receipts.
- `cycle-04-independent-audit.json`: independent rescoring of all129epoch rows, all responses and raw lake accounting; no Brain construction or lake mutation.
- `final-pure-regression.{json,log}`:120passed; `final-brain-regression.{json,log,xml}`:95passed.
- `final-source-equivalence.json`:8numerical ASTs equal to acceptance pins, excluding only final gradient-claim declaration constants; no missing/phantom declarations.
- Every controller keeps its source SHA256 map, exact script, isolated home and module overlay. Earlier failures and engineering recipe revisions remain available.

The initial checkpoint verification failed on the controller's tuple-versus-JSON-list comparison; serialized predictions were identical. The comparator was corrected and rerun in a new process. Production/numerical values were unchanged. The earlier always-positive full run timed out at240s and its raw log/source overlay were retained; this is not a completed efficacy run.

Remaining research gates stay closed: BETA for later positive-spend tasks, general families/context selection, continued shared-core plasticity without forgetting, competitive pruning/sleep, independent-seed and real-data full-Engine efficacy. No main merge, push, live lake or full operator battery was performed.

Closed implementation on branch `engine-cycle-integration`, commit `26501ed3`; source working tree is clean. Final-source journal-only replay was repeated after the five declaration-only metadata additions: exact same model digest `ff48914c33b40f4c87eb4bbf1de6b8f3414f810249eb0a6567dd7b45ed516a1c`, both skills unchanged, explicit `G9_RETENTION_ALREADY_CONSUMED` refusal. Receipt `cycle-04-grow-restore-journal-2.json`; consolidated source/test completion receipt `completion.json`.


# Engine independent raw journal audit

Original workspace member: `planning/experiments/engine-integration/cycle-04-independent-audit.json`

```json
{
  "scope": "independent artifact and raw-journal audit, no Brain or learning run",
  "seed": 9900,
  "checks": {
    "lake_rb_only_no_change": true,
    "raw_console_rows_equal_journal": true,
    "canonical_neurons_and_complete_stack_exact_old_to_grow": true,
    "source_18_of_1024_one_batch_stream9900": true
  },
  "scores": {
    "old.validation.aii": {
      "correct": 16,
      "presentations": 16
    },
    "grow.validation.ii.xor": {
      "correct": 16,
      "presentations": 16
    },
    "grow.validation.aii": {
      "correct": 16,
      "presentations": 16
    },
    "grow.final.ii.xor": {
      "correct": 16,
      "presentations": 16
    },
    "grow.final.aii": {
      "correct": 16,
      "presentations": 16
    },
    "grow.grown_off.ii.xor": {
      "correct": 8,
      "presentations": 16
    },
    "grow.grown_off.aii": {
      "correct": 16,
      "presentations": 16
    },
    "prior_allplastic.aii": {
      "correct": 0,
      "presentations": 16
    },
    "prior_allplastic.ii.xor": {
      "correct": 16,
      "presentations": 16
    },
    "grow.channel.baseline": {
      "correct": 16,
      "presentations": 16
    },
    "grow.channel.feed_off": {
      "correct": 16,
      "presentations": 16
    },
    "grow.channel.return_off": {
      "correct": 8,
      "presentations": 16
    },
    "grow.channel.both_off": {
      "correct": 8,
      "presentations": 16
    },
    "cycle-04-grow-restore-2.json.ii.xor": {
      "correct": 16,
      "presentations": 16
    },
    "cycle-04-grow-restore-2.json.aii": {
      "correct": 16,
      "presentations": 16
    },
    "cycle-04-grow-restore-journal.json.ii.xor": {
      "correct": 16,
      "presentations": 16
    },
    "cycle-04-grow-restore-journal.json.aii": {
      "correct": 16,
      "presentations": 16
    }
  },
  "actual_epoch_history": {
    "aii": {
      "epochs": 22,
      "teach_presentations": 88,
      "validation_presentations": 352,
      "shared_plasticity_deposits": 66,
      "first_all_correct_epoch": 13,
      "final_stable_run": 10,
      "all_epoch_rows_rescored": 22
    },
    "ii.xor": {
      "epochs": 107,
      "teach_presentations": 428,
      "validation_presentations": 1712,
      "shared_plasticity_deposits": 0,
      "first_all_correct_epoch": 98,
      "final_stable_run": 10,
      "all_epoch_rows_rescored": 107
    }
  },
  "schedule_summary": {
    "aii": {
      "unique_teach_items": 4,
      "teach_items_per_epoch": 4,
      "validation_unique_texts": 4,
      "validation_presentations": 16,
      "repetitions_per_unique_item": 4,
      "actor_teaches": 88
    },
    "ii.xor": {
      "unique_teach_items": 4,
      "teach_items_per_epoch": 4,
      "validation_unique_texts": 4,
      "validation_presentations": 16,
      "repetitions_per_unique_item": 4,
      "actor_teaches": 428
    }
  },
  "source": {
    "batch_id": 0,
    "stream_seed": 9900,
    "evaluations": 1024,
    "admitted_ids": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17
    ],
    "selected_source_indices": [
      737,
      678,
      328,
      22,
      471,
      505,
      87,
      407,
      762,
      390,
      140,
      114,
      387,
      111,
      816,
      17,
      711,
      973
    ]
  },
  "channel_interventions": {
    "baseline": {
      "score": {
        "correct": 16,
        "presentations": 16
      },
      "pool_feature_l2_delta": 0.0,
      "feed_l2_minmax": [
        2.567633719219973,
        2.976195381808685
      ],
      "return_weights_l2": 4.913636133957407,
      "afferent_l1_minmax": [
        7946.315312388685,
        8230.419010981283
      ],
      "changed_latency_entries": 0,
      "grown_drive_l2_delta": 0.0
    },
    "feed_off": {
      "score": {
        "correct": 16,
        "presentations": 16
      },
      "pool_feature_l2_delta": 32.68026927673638,
      "feed_l2_minmax": [
        0.0,
        0.0
      ],
      "return_weights_l2": 4.913636133957407,
      "afferent_l1_minmax": [
        7946.315312388685,
        8230.419010981283
      ],
      "changed_latency_entries": 0,
      "grown_drive_l2_delta": 0.0
    },
    "return_off": {
      "score": {
        "correct": 8,
        "presentations": 16
      },
      "pool_feature_l2_delta": 21.166010488516726,
      "feed_l2_minmax": [
        3.760032649929224,
        3.969203856319631
      ],
      "return_weights_l2": 0.0,
      "afferent_l1_minmax": [
        0.0,
        0.0
      ],
      "changed_latency_entries": 220,
      "grown_drive_l2_delta": 0.536067283836626
    },
    "both_off": {
      "score": {
        "correct": 8,
        "presentations": 16
      },
      "pool_feature_l2_delta": 32.68026927673638,
      "feed_l2_minmax": [
        0.0,
        0.0
      ],
      "return_weights_l2": 0.0,
      "afferent_l1_minmax": [
        0.0,
        0.0
      ],
      "changed_latency_entries": 220,
      "grown_drive_l2_delta": 0.536067283836626
    }
  },
  "lesion_arithmetic_rescore": {
    "aii": {
      "correct": 0,
      "presentations": 16,
      "predictions_unchanged": false
    },
    "ii.xor": {
      "correct": 16,
      "presentations": 16,
      "predictions_unchanged": true
    }
  },
  "restores": {
    "cycle-04-grow-restore-2.json": {
      "journal_only": false,
      "exact": {
        "model_digest": true,
        "core_state": true,
        "actors": true,
        "validation": true,
        "source_events": true
      },
      "final_repeat_refused": true,
      "reason": "refusal asserted by recorded controller"
    },
    "cycle-04-grow-restore-journal.json": {
      "journal_only": true,
      "exact": {
        "model_digest": true,
        "core_state": true,
        "actors": true,
        "validation": true,
        "source_events": true
      },
      "final_repeat_refused": true,
      "reason": "G9_RETENTION_ALREADY_CONSUMED"
    }
  },
  "provenance": {
    "old": {
      "exit_code": 0,
      "controller_seconds": 9.676446291000502,
      "source_hashes_verified": 8
    },
    "grow": {
      "exit_code": 0,
      "controller_seconds": 135.82026391600084,
      "source_hashes_verified": 8
    },
    "restore-2": {
      "exit_code": 0,
      "controller_seconds": 5.082242708000194,
      "source_hashes_verified": 8
    },
    "restore_journal": {
      "exit_code": 0,
      "controller_seconds": 31.41120899999987,
      "source_hashes_verified": 8
    }
  },
  "lake": {
    "sha256": "dda9da7974984173ccba86d74fcc2b2755bb64ab3133628f370e5c0b4dcc5f7a",
    "frames": 685,
    "goal_events": 159
  },
  "retained_negative_control": {
    "allplastic_A": {
      "correct": 0,
      "presentations": 16
    },
    "allplastic_B": {
      "correct": 16,
      "presentations": 16
    },
    "no_frozen_selection": true,
    "no_final_claims": true
  },
  "limitations": [
    "Single already exposed engineering seed; no independent statistical cohort or efficacy comparison.",
    "Four unique constructed text/code items per task, repeated four times per read set; not 16 independent cases.",
    "Future/final is a different presentation order of those same four items, not a new semantic generalization set.",
    "Task-conditioned context, corrective feedback, explicit protected core, one admitted batch of18, no pruning; no autonomous RSI claim.",
    "Feed ablation changes Part D features but retains16/16; no demonstrated final-answer feed benefit.",
    "Return ablation reduces B to8/16 on this fixture; not a general causal/statistical effect estimate.",
    "Canonical neurons+complete stack are independently equal; full shared-state digest is compared as recorded, not recomputed from an unavailable full shared snapshot.",
    "Lesion rescore uses saved core weights plus actual captured grown-drive continuation; controller independently asserted real lesion refusal.",
    "First checkpoint restore controller retained an assertion failure caused by tuple/list normalization; corrected checkpoint and journal-only controllers pass.",
    "No SHD result or full operator battery is implied."
  ]
}
```


# Engine source and implementation completion

Original workspace member: `planning/experiments/engine-integration/completion.json`

```json
{
  "commit": "26501ed3",
  "branch": "engine-cycle-integration",
  "focused_tests": 215,
  "checkpoint_recovery": true,
  "journal_only_recovery": true,
  "final_repeat_reason": "G9_RETENTION_ALREADY_CONSUMED",
  "model_digest": "ff48914c33b40f4c87eb4bbf1de6b8f3414f810249eb0a6567dd7b45ed516a1c",
  "source_files": [
    {
      "path": "src/snn_data_lake/growth/growth_loop.py",
      "acceptance_sha256": "23d36ccf5cb5d8eed418678150be92a26300cdaf9c90e94413cd54a24ff4546f",
      "final_sha256": "23d36ccf5cb5d8eed418678150be92a26300cdaf9c90e94413cd54a24ff4546f",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    },
    {
      "path": "src/snn_data_lake/growth/capacity_certificate.py",
      "acceptance_sha256": "e002742ffe9245e70ef4d6dcb5fec32bb0c1d4f54580b23e6e7351d250af1814",
      "final_sha256": "0c715d32840e55b21f463798f0b5811d5c490ac0de05a64982a6201679e92fe8",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    },
    {
      "path": "src/snn_data_lake/engine/brain.py",
      "acceptance_sha256": "5007170644a026a337b50053ea05c2962d308de7553eeed998cba0e1c8790c0f",
      "final_sha256": "6b07551d4b9a808d3cc9a82c722a68750a91260b11de3723de580017b818b9fc",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    },
    {
      "path": "src/snn_data_lake/readout/posed_problem.py",
      "acceptance_sha256": "0e6e90143a0bc586c574a9e3bb8cdc56a1892e03346f5f5fc03ebcfea7f8d86d",
      "final_sha256": "0e6e90143a0bc586c574a9e3bb8cdc56a1892e03346f5f5fc03ebcfea7f8d86d",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    },
    {
      "path": "src/snn_data_lake/instruments/joint_mask_search.py",
      "acceptance_sha256": "2771ab09eac8e4e34dfa0d82b752b4a1a839a2c5420eaa06bcf55eee61b8b0dc",
      "final_sha256": "2771ab09eac8e4e34dfa0d82b752b4a1a839a2c5420eaa06bcf55eee61b8b0dc",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    },
    {
      "path": "src/snn_data_lake/readout/interaction_learning.py",
      "acceptance_sha256": "f1d9a1785e66921c6f4bb2fa608f4cca0e8080b317bcf68243a5078cbbac14b9",
      "final_sha256": "f1d9a1785e66921c6f4bb2fa608f4cca0e8080b317bcf68243a5078cbbac14b9",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    },
    {
      "path": "src/snn_data_lake/plasticity/plasticity_rule_stack.py",
      "acceptance_sha256": "07f27c5b21becb63d95fba41f60d1aeb108c9bb32365c6321a50e26121022a88",
      "final_sha256": "32487f79b4e13edfa7413eec391d425e5fc5d3889097576eba25b88b84c98d65",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    },
    {
      "path": "src/snn_data_lake/plasticity/homeostatic_plasticity_stack.py",
      "acceptance_sha256": "99fe23a9df1cda8286f30479a873655f68586af82f0a1e9606df3fcaa8f5a637",
      "final_sha256": "4d1175fa1a3e1871c8d333ced5259e07a9a2ba15d40844ee6ecf6ba15b543dda",
      "numerical_ast_identical_excluding_claim_declaration": true,
      "missing_claims": [],
      "phantom_claims": []
    }
  ]
}
```


# Engine runtime and continuation requirements

Original workspace member: `planning/experiments/engine-integration/RUNTIME.md`

# Full Engine runtime recovery

Recovered in an isolated directory outside the repository. No repository source, safety-envelope file, policy certification, or live lake was modified by this runtime work.

- CPython 3.14.6 free-threaded; Torch 2.15.0 CPU; NumPy 2.5.4; strict free-threaded Brain import passed.
- Core packages + test/build tools: exact wheel versions/hashes in `core-lock.json` (31 packages); pip receipt `core-install.json`.
- Rust 1.96.0 installed with isolated CARGO_HOME and RUSTUP_HOME under the visualization workspace's `engine-runtime`. Native kernel built by the incumbent `snn_data_lake.infra.native_build` on a full source snapshot at commit c4c10b285f1741aa798d5ee9c5296ae371fabecf.
- Native wheel: `sndl_native-0.1.0-cp314-cp314t-macosx_11_0_arm64.whl`; SHA256 fbb553c6b25b4e84a0ee54185a02af8d84601eca790e9c63305616d2947b4feb. Native extension is present only in the external snapshot's package, not the main repo/src.
- Full `engine.brain` import now succeeds with the GIL off. Original import failure was missing mujoco; `_native`, scipy, pyarrow, psutil, Brian2 and pytest were also absent before recovery.
- Baseline tests `tests/test_g9_posed_actor.py` and `tests/test_g9_growth_adapter.py`: 31 passed in 50.07 seconds, one serial worker, 600-second child deadline. These are existing software controls; they do not demonstrate the blocked deciding growth campaign.

## Running modified Python source

`run-engine.sh` defaults to the immutable `engine-runtime/baseline-source`. A separate `engine-runtime/working-source` copy includes the same compiled native extension. Copy only the changed src/tests files from the working repository into that snapshot, leaving the baseline intact; then set `ENGINE_SOURCE_ROOT` to `working-source` before invoking the wrapper. Main source shadows a separately installed package, so the native wheel installed in a second site alone would not make `_native` available under main repo/src. Do not use `maturin develop` or an editable .pth as a shortcut.

For tests, run from the chosen snapshot root with a bounded subprocess controller. Set NORD_HOME to an isolated temporary directory and SNDL_R2_CHILD_DEADLINE_NS to time.time_ns()+budget_s*10**9 at launch. The tests' conftest provides per-test and per-module isolated homes and respects a shorter caller deadline. Use `-p no:cacheprovider --no-cov`, no xdist/parallel Brain workers initially. Example test arguments: `-m pytest -p no:cacheprovider --no-cov -x -vv tests/test_g9_posed_actor.py tests/test_g9_growth_adapter.py`.

Brian2 2.10.1 is installed but reference simulations must remain in their existing isolated subprocess path because its import enables the GIL. No live home, daemon or physical arm path was exercised.
