# Research showcase presentation plan

Draft for approval. Prepared November 2, 2026, America/Chicago. Revised for concrete source-traced variables and a current-research comparison after the objectives.

## Brief

- Showcase: in a few days. Exact date remains unspecified.
- Talk: approximately 10 minutes.
- Audience: technical, with limited assumed knowledge of AI or spiking neural networks.
- Current deliverable: the research story and presentation plan. Slide authoring follows approval.

## Recommended story

SNDL investigates a spiking learning system that can add capacity when existing weights cannot solve a problem, use a separate recurrent reservoir, and preserve learned state in one durable data lake. The proposal's long-term deciding test compares the complete engine with matched controls and ablations.

For this showcase, center the presentation on the engine as a growing, adapting system. Explain how it works immediately after the background and problem slides, then state the research objectives. Immediately after the objectives, compare the engine with conventional transformer inference and current adaptive ANN/SNN research. Follow the comparison with the concrete data contract. Use the existing learning, forgetting, recovery, and persistence experiments as evidence for parts of this larger design.

Working title: **Spiking Neural Data Lake: Toward a Growing, Adaptive Neural Engine**

Opening hook: **How can a neural system expand what it can learn while retaining what it already knows?**

Primary research question: **Can an engine that adapts its weights and structure improve problem solving while preserving earlier learning and its state across restarts?**

Deciding future test: **Does the complete growth–reservoir–readout engine improve held-out performance against an equal-capacity growth-off system, with an advantage attributable to its components?** This remains a future evaluation, not a result of the local pilots.

Completed pilot subquestion: **Can offline replay recover an earlier class after sequential learning, preserve the newer class, and retain the recovered state after restart?**

Long-term ambition: **recursive self-improvement**, with an experiment that identifies and measures improvement of the mechanism used to generate subsequent improvements. This is a future research goal, not a demonstrated result of neural growth, replay, or persistence alone.

## Engine explanation and adapting state

Explain this as the target engine cycle, with the implemented and experimentally tested parts identified:

1. Receive an encoded problem or experience and attempt an answer using the current state and learned weights.
2. Learn from teaching feedback and evaluate whether the problem meets its declared solved criterion. Distinguish insufficient information or contradictory data from a capacity problem.
3. When additional capacity is justified, add and connect groups of neurons, then attempt the problem again. Repeat within a growth budget. At the limit, keep the problem open and report UNSOLVED.
4. Feed the new neural activity and permitted transported weights into the separate spiking reservoir. Train readout neurons to extract task-relevant output and return information to the brain.
5. Use offline processing to replay experience, adapt useful weights, and evaluate which active units or connections to retain. Preserve the required state and experience in the lake for reconstruction at restart.

The explanation should distinguish four aspects of state:

| Aspect | What changes | Role |
|---|---|---|
| Neural activity | Spikes, membrane variables, and recent context | Represents what is happening now and recent input history |
| Learned parameters | Synaptic/readout weights | Changes how the system responds to experience |
| Active structure | Neurons and their connections | Adds capacity and reorganizes the active network |
| Durable state | Saved parameters, structural records, and experience | Preserves acquired state between processes |

The whole engine's evolving state is the focus. Avoid depicting the reservoir as a permanently fixed-size container. Do not imply that every kind of structural growth inside the reservoir is already implemented or validated. The exact reservoir size in the component experiments is a reproducibility parameter for notes or backup, not the presentation's organizing idea.

Architecture sources: [engine directive](../repo/docs/POLICIES.md), section P4d, and [recorded engine structure](../repo/specs/022-unified-brain-arc/evidence/engine-structure-2026-09-25.md). Current [Grower implementation](../repo/src/snn_data_lake/growth/grower.py) is still an initial wrapper, so the complete cycle must be presented as the design under investigation.

## Comparison immediately after the objectives

Use the title **Adaptation and Self-Improvement Across Neural Systems**. Compare what changes during use, structural adaptation, persistence, and the demonstrated level of self-improvement.

The phrase "current transformer-based ANNs and SNNs only produce snapshots" cannot be used as a factual claim. A checkpoint is stored state, not an output type. Current research includes test-time neural memory ([Titans](https://arxiv.org/abs/2501.00663)), expandable continual-learning SNNs ([PS-SNN](https://www.nature.com/articles/s41598-026-42970-6)), and foundation-model agents that modify their own code and, in newer work, their improvement procedure ([DGM](https://arxiv.org/abs/2505.22954), [Hyperagents](https://arxiv.org/abs/2603.19461)).

The intended distinction is specific: SNDL investigates local learning, budgeted growth, reservoir/readout feedback, offline consolidation, and a single durable lake as one continuing process. The current evidence does not prove an exclusive capability, superiority to these systems, or recursive self-improvement. Treat the comparison as research positioning, not a head-to-head benchmark.

The detailed comparison and primary-source references are in [verified data and comparison notes](SHOWCASE_DATA_AND_COMPARISON.md).

## Concrete data consumed and produced

Place this slide after the comparison. Show a compact selection of actual variables, with their human-readable meanings. The full stage-by-stage source map, types, expressions, and implementation/test boundaries are in [verified data and comparison notes](SHOWCASE_DATA_AND_COMPARISON.md). Do not invent a generic engine-wide state structure or splice separate learner paths into one execution trace.

| Processing stage | Concrete consumed data | Concrete produced/changed data |
|---|---|---|
| Input episode | `Turn.text`, `Turn.role`, and `Turn.label` as learner feedback only during teaching | Saved records `texts`, `roles`, `target`; evaluation targets stay with the scorer |
| Encoding and reservoir | `text`, `im.code(text)`, zero-padded `pool_features.x`, existing `pool.v/s/acc`, weights/connectivity | Updated `pool.v/s/acc` and activity vector `Turn.x`, saved as `row['x']` |
| Prediction and learning | `row['x']`, `unit.beta`, `unit.labels`, and training `row['target']` | The label selected by `argmax(s @ beta)`; learning updates `unit.beta` and `rule.steps` |
| Growth and feedback interfaces | Diagnosis, `ProblemRecord.budget`, `spend_since_teach`, source units, grown-unit `phi`, feed matrix | New `GrownUnit`/birth records, ledger status, reservoir `feed`, and return `packet = (answer, self.clock + float(neuron.t_post))` |
| Offline processing | Training `replay` rows and saved readout state | Updated `beta`, `presentations`, and `solves`; no biological stage result is inferred |
| Save and restore | `schema`, `head`, `seed`, `phase`, `pool`, `state['readout']`, `state['replay']`, and expected receipt | Named lake-state bytes and checksum; reconstructed readout/replay and regenerated feature vectors |

Rows covering growth/feed/return refer to implemented interfaces not executed in the local recovery pilot. The local checkpoint saves the readout/replay and pool reconstruction recipe/receipts. It does not save a mid-episode `v/s/acc` continuation. Output scores are not calibrated probabilities.

Use a real saved probe: **`fog gin keg mod`**, role **`probe`**, scorer target **`GREET`**. For development seed 8000, row 1, the saved predictions are **GREET after A, GOAL after B30, GREET after replay, GREET after restart**. All values, the full feature vector, state hashes, and original source-file hashes are preserved in [the exact data example](SHOWCASE_DATA_EXAMPLE.json). This is an illustration from an existing experiment, not a new measurement.

## Source and version check

- Original proposal: [October 19 proposal](/Users/arthurkhvan/Downloads/SNDL-Engine-Project-Proposal-APA.pdf), all seven pages inspected through text extraction.
- Public upstream `HEAD` and `main` checked directly with `git ls-remote` during this review. Both point to `c4c10b285f1741aa798d5ee9c5296ae371fabecf`, dated October 29. The local `repo/` checkout matches and is clean.
- Reviewed the changes from the earlier reviewed revision `c1df388` to the current head. Recent work includes experiment instrumentation, readout-staleness design, and follow-up reward-rule tests. The latest record is an execution record, not evidence that the full engine has passed its scientific evaluation.
- Current [upstream task record](../repo/specs/022-unified-brain-arc/tasks.md): full G5/G6/G9 remain open. Earlier blanket status sentences and checkbox counts require dated evidence to interpret.
- Local reports from November 1–2 contain newer component experiments outside the upstream checkout. `planning/STATUS.md` has stale coordination rows, so it is not the evidence source for completion.
- Fresh offline analyses of the saved readout, class-order recovery, and duration/pruning records all reproduced the corresponding saved summary files byte for byte. No new training was run for this presentation review.

## Evidence available now

| Finding | Exact scope | Presentation use |
|---|---|---|
| The upstream G3 baseline records 2, 2, and 4 grown units after 300 ticks for its three live seeds, versus 0 with growth disabled. | Historical operator capture, October 23, with an October 25 telemetry-schema recapture. This verifies observable structural growth and a sensitive capture, not a task-performance advantage from growth or a local reproduction. | A concrete supporting example of changing structure. Use the reported endpoint comparison. A continuous growth curve requires the actual saved trajectory, not interpolation from these counts. |
| Delta readout solves the tested tasks on a fixed 256-neuron spiking reservoir. On the delayed-cue diagnostic it scores 32/32 with carried state and 16/32 with per-turn reset on each of four development seeds. | The memory diagnostic repeats known episodes. It demonstrates information carried in this tested state, not broad generalization or an attribution solely to recurrent synapses. Reservoir and growth remain fixed. Keep the exact size in notes/backup. | Supporting evidence that evolving activity state carries task-relevant context. This is not a growth experiment. |
| Class A development accuracy is 100% after A training and 0% after B training. Offline replay and batch consolidation both recover A while preserving B, with final A=24/24 and B=24/24 on every seed. | A=GREET and B=GOAL are class groups of one text task. Final contains six unique texts per class, each presented four times. The splits use different combinations of familiar words. | Main result: forgetting and recovery. |
| Reloaded readout/replay state and answers exactly match the saved state in new processes. | Explicit component save/load through the lake at episode boundaries. Production Brain flush and crash recovery are not tested. | Main result: persistence. |
| One mixed replay pass, 32 updates, already recovers both class groups. Tested budgets of 2, 5, and 10 passes remain at the same accuracy ceiling. | The smallest successful tested budget, not an optimal universal sleep duration. This is offline replay, not a biological sleep-stage result. | A brief efficiency observation accompanying the main result. |
| A separate prototype learner can prune 12 active prototypes to 6 while retaining all three synthetic tasks and halving search comparisons. | Different learner and fixture. Removing high-utility prototypes also retains accuracy, so the experiment does not establish the specific utility ranking as the cause. | Backup only, unless there is extra time. |

Sources: [upstream growth baseline](../repo/specs/022-unified-brain-arc/evidence/g3-t0-baseline.md), [readout report](reports/04-readout.md), [recovery and persistence report](reports/05-sleep.md), [duration and pruning report](reports/06-pruning.md).

All main local experiments use four already exposed development seeds, 8000–8003. Report them as exploratory component results. Repeated examples and repeated runs do not increase the independent sample size. Batch has no observed final-accuracy advantage over replay in the tested budget. NREM/REM comparison remains unperformed because the delta baseline lacks the required interface.

The local delayed-cue success does not overturn the historical G1c stream result: the task, stream, and endpoint differ. Avoid a blanket claim that the reservoir is now validated for the full engine.

## Proposed eleven-slide structure

Target speaking time: 9 minutes 30 seconds, leaving 30 seconds for transitions.

| Slide | Requested section | Purpose and content | Time |
|---:|---|---|---:|
| 1 | Title & Hook | Introduce the growing, adapting engine and opening research question. | 0:20 |
| 2 | Background | Explain the relevant spiking-network idea and the difference between neural activity, learned weights, and structure. | 0:35 |
| 3 | Problem | Explain capacity limits, interference with earlier learning, and preserving acquired state across restarts. | 0:40 |
| 4 | How the Engine Works | Explain the attempt, learning, growth, reservoir/readout feedback, offline processing, and save/rebuild cycle. Use one architecture diagram with the current implementation/test scope visible. | 1:20 |
| 5 | Objectives / Research Question | State the adaptive-engine question, the matched-control test it ultimately owes, and the completed pilot's narrower subquestion. Identify RSI as a long-term goal. | 0:35 |
| 6 | Comparison with Current ANNs and SNNs | Compare concrete adaptation mechanisms and scope. Distinguish frozen-weight inference from adaptive ANN/SNN work and agent-level self-modification. | 0:55 |
| 7 | Concrete Data Consumed and Produced | Use actual variables and the saved probe through processing, learning, growth interfaces, offline processing, and persistence. Distinguish recorded phases from implemented interfaces. | 1:10 |
| 8 | Methods | Show the actual A-training, B-training, offline-processing, and restart experiment. Explain no-update, replay, batch, and shuffled-label controls. State fixed topology and four development seeds as experimental scope. | 0:45 |
| 9 | Results & Discussion | Show evidence of adapting state: the reported upstream structural-growth endpoints and the local readout/context result. Label them as separate component experiments with different provenance. Exact configuration details stay in notes/backup. | 0:40 |
| 10 | Results & Discussion | Show forgetting, recovery, and persistence for both classes. Mention replay/batch equality and the one-pass observation within the tested setup. | 1:30 |
| 11 | Conclusion: Shortcomings and Future Goals | State the demonstrated component outcome, remaining integration and evaluation gaps, the next growth-integrated tests, and the evidence still needed for RSI. | 1:00 |

For the main forgetting plot, use development accuracy consistently across phases. Do not join an early development measurement to a later final measurement as though they were one evaluation series. Display final A/B results separately. Clearly label replay as training on stored training examples and distinguish restart from additional learning.

## Production plan after approval

1. **Freeze claims and evidence.** Map each factual slide statement to its source record. Separate measured findings, implementation status, and future experiments. Retain the exact seed count and unique-example counts in notes or readable figure captions.
2. **Write the talk before styling.** Draft eleven slide purposes, concise slide text, and speaker notes for the 9:30 schedule. Explain the engine after the problem. Place the comparison immediately after the objectives, then the concrete data contract. Use the saved probe and explain an unfamiliar term at first use.
3. **Prepare the figures and slides.** Use one architecture figure and a small number of data figures generated from saved records. Show how activity, learned weights, active structure, and durable state evolve. Look for the archived growth trajectory before proposing a growth-over-time plot. If only the documented endpoint counts are available, show those endpoints and their growth-off control. Distinguish the full design from the tested subset. Keep the fixed reservoir count, method parameters, and pruning details in notes or backup.
4. **Rehearse and verify.** Check each claim and figure against the frozen evidence, rehearse to ten minutes, and prepare backup answers on controls, persistence, pruning, and the difference between component success and complete-engine validation.

The talk can be completed from existing evidence. Additional research should not be on the critical path. If a growth-integrated result becomes available before the final content freeze, include it only after its controls and restoration check pass. Otherwise retain it as planned work, with no forecasted performance numbers.

## Shortcomings and future goals for the closing slide

Current shortcomings:

- The complete adaptive-growth, feed, return, and offline-processing loop still requires integration and causal evaluation.
- The presented local experiments isolate components with fixed topology and four exposed development seeds. They do not establish the benefit of structural growth or broad task generalization.
- Offline replay results do not validate biological sleep stages. Pruning was tested in a separate prototype learner, not in the presented reservoir/readout path.
- Component persistence does not establish production Brain crash recovery or arbitrary interruption/resume.
- The reviewed experiments do not establish recursive improvement of the mechanism that generates future improvements. Current adaptive ANN/SNN and self-modifying-agent research must be acknowledged.

Future goals:

- Integrate the chosen readout and persistence path with budgeted adaptive growth and the return signal.
- Compare the complete engine with an equal-capacity growth-off baseline and per-component ablations.
- Test retention through structural changes, offline processing, and restart on broader tasks under a fresh confirmatory design with justified sample size.
- Study compatible sleep/plasticity/pruning mechanisms and measure their accuracy/resource tradeoffs.
- Define and test the specific modification to the improvement mechanism needed for an RSI claim, using held-out measurements and matched controls.

Potential impact: a controlled experimental platform for investigating how structural adaptation, learning, and durable memory can work together. Evidence for the complete engine remains the next research goal.

The approval decision is the engine-centered story and revised eleven-slide structure, including the verified data contract and qualified current-research comparison. No presentation artifact has been authored yet.
