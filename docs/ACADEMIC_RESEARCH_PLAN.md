# From SNDL to a paper on affordable edge learning

Prepared 2026-09-20 for a professor in AI and robotics. Assumes a 15-minute presentation.
This is a research proposal, not a report of completed edge-hardware experiments. See the
[assessment](RESEARCH_ASSESSMENT.md) for the evidence available today.

## 1. The proposition to take into the meeting

**Study when a locally learning spiking agent can add useful capacity and retain experience under
fixed resource budgets, and whether it improves on compact conventional controllers.**

Suggested working title:

> Budgeted Local Adaptation with Spiking Computation and Persistent Event Memory

Use "a brain that grows on a local machine" to explain the motivation. Define it experimentally:
the system changes learned state or capacity from local experience, improves held-out task
performance, retains earlier skills, and pays for its learning within measured resource limits.
Growth without those outcomes is allocation, not demonstrated intelligence.

The current project supplies an apparatus: spiking modules, learning and growth machinery,
persistent storage, an arm simulation, and extensive checks. The proposed contribution is a
**measured quality-cost frontier and a tested growth decision rule**. Novelty of that combination
must be established through the literature review; the existence of spikes, a data lake, or
anatomical module names is not itself a novel result.

### A 60-second opening

> I want to study affordable robots that adapt where they are used, without needing a cloud
> training service for each change in their environment. I have built a local research platform
> combining spiking computation, persistent event memory, growth mechanisms, and a simulated
> robot arm. Some storage and control components work, but I have not yet established an
> end-to-end efficiency advantage. In fact, our current energy model shows that a fair compact
> dense baseline can win. I would like to turn the platform into a controlled study: under the
> same memory, energy, and latency budget, when does local growth help a controller adapt and
> retain skills? I am seeking your help with the experimental design, strong baselines, and the
> scope of a first paper.

## 2. What to claim now

| Defensible statement | Evidence or qualification |
|---|---|
| There is a runnable local simulation and event-memory platform. | Demo, storage, query, and scorer checks rerun in the assessment. |
| Sparse storage saves bytes on the measured sparse workload. | 417,976 B versus 3,200,000 B bit-packed dense storage; 7.7x on that example. The 61x figure uses a byte raster. |
| The code has mechanisms intended to support local growth and persistence. | Growth gates, neurogenesis, widening, restart tests; usefulness must be measured on the selected task. |
| Width improved one recorded next-category task over a one-unit baseline. | The corrected SC-012a mean gain is 15.1 percentage points across eight seeds; it does not beat the incumbent admission policy or previous-category predictor, and review found a non-discriminating null veto. See the [assessment](RESEARCH_ASSESSMENT.md). |
| The project records unsuccessful experiments. | Neuron paging was deleted after a latency-bound failure despite a memory improvement. |
| Cheap, capable edge learning is the intended application. | It remains a hypothesis requiring on-device resource and task measurements. |

Do not claim a proven low-power brain, unbounded growth on finite hardware, human-level reasoning,
consciousness, an all-spiking system, or MCU deployment. The current runtime includes CPython
3.14t, PyTorch, NumPy, PyArrow, MuJoCo, and snnTorch. A small model does not erase that footprint.
Keeping a fact on disk also does not prove that a learned skill remains usable.

## 3. Choose one first-paper question

**Primary question:** on a CPU-only local machine, does evidence-gated growth with persistent
experience improve adaptation and retention per unit of compute, relative to fixed-size spiking
and compact dense controllers?

Use continual reaching in the existing SO-101 MuJoCo body as the main task. Start with a small,
well-defined observation/action interface. Change target distributions and selected simulated
dynamics between task blocks. Evaluate the same agent on earlier conditions after each block.
The task adapter and complete comparison harness are work to build, not a finished benchmark.

Use streaming gesture recognition as a lower-cost diagnostic and possible first pilot. It tests
the sensing and temporal-learning path, but cannot stand in for manipulation performance.

### Candidate hypotheses, to ratify before the confirmatory run

| Hypothesis | Proposed success rule | Outcome that rejects or limits it |
|---|---|---|
| H1: the adaptive system improves resource efficiency at useful quality | At a frozen quality target, at least 2x lower measured compute-system energy to adapt than the strongest eligible fixed baseline, while staying within RAM and latency limits | No energy advantage, quality target missed, or any binding resource limit exceeded |
| H2: growth buys useful capacity | Better held-out adaptation than the same initial network with growth disabled, and competitive quality-cost tradeoffs against fixed larger networks | Added units fail to improve held-out behavior, or fixed networks dominate the frontier |
| H3: persistent experience improves retention | Better old-task performance than an equal-budget no-replay/no-retrieval arm after matched updates, including after restart | Retention does not improve, or benefit vanishes after charging retrieval and replay costs |

The **2x threshold is a proposed practical bar**, not a derived law or an achieved result. Choose
it, the quality target, retention margin, and hardware budgets with the professor after a separate
pilot. Freeze them before collecting confirmatory outcomes. H1 is the primary endpoint; H2 and H3
are mechanism tests. Do not rename a failed primary endpoint as secondary after seeing results.

## 4. Build the experiment

### Task and split

1. Define reaching success using a fixed end-effector distance tolerance, dwell time, episode
   deadline, and failure rule. Select these from task needs before the final run. Report continuous
   distance error alongside binary success so a threshold cannot conceal poor behavior.
2. Train sequentially on at least three conditions, such as changed goal regions and simulated
   payload/friction regimes. Randomize and counterbalance their order across independent runs.
3. Hold out target positions, initial states, and disturbance realizations. After each block,
   freeze learning during evaluation on both new and previous conditions.
4. Separate training, validation, and final test seeds. Growth decisions and hyperparameter
   selection may use training/validation evidence only. Never feed final test scores back into
   the growth gate, even though the existing gate accepts an accuracy trajectory.
5. For sensor recordings, split by recording session and, if applicable, participant or device.
   Adjacent windows from one capture must not straddle training and test sets.

All agents receive the same observation information and action opportunities. Give dense baselines
equivalent temporal history; comparing an SNN with a memoryless model on a temporal task is unfair.
Freeze observation dimensions, encoding precision, action rate, and simulator settings.

### Baselines and ablations

| Arm | Purpose |
|---|---|
| Classical PID or inverse-kinematics controller, with a suitable adaptive variant | Establish whether learning is needed for the chosen task at all. |
| Small MLP or GRU controller | Strong conventional learned baseline; include a practical optimized inference path and charge online update costs separately. |
| Fixed-size SNN at the initial width | Isolate whether growth improves on the starting system. |
| Fixed-size SNNs across a capacity sweep | Compare with a Pareto frontier, including a model at the adaptive system's final width. |
| SNDL learner with evidence-gated growth and persistent experience | Proposed full experimental condition. |
| Same learner, growth disabled | Isolate growth with initialization, inputs, updates, and memory held constant. |
| Same learner, replay/retrieval disabled but logging retained | Isolate the learning value of stored experience without giving the control an unfair logging-cost reduction. |
| Growth events with learning disabled, or matched random growth | Test whether changes merely allocate capacity or alter randomness. |

Use the same environmental samples and equal tuning effort. Report both sample-matched learning
curves and resource-matched outcomes. A dense controller may use its strongest practical training
method; it need not be artificially restricted to the SNN's update rule. Record offline pretraining,
including any GPU time, instead of hiding it outside the "local" cost.

Use an INT8 baseline only where its actual implementation and update workflow support it. The
current energy model's INT8 comparison is a warning about fairness, not a deployed INT8 controller.

### Growth rule to investigate

Start by tracing the selected live growth path and its consumer. Reuse the existing growth gate
where applicable. Check that its measured signal reaches the controller's action outputs; adding
neurons or changing a counter alone is not evidence of a functional change.

A proposed resource-aware extension would admit a growth event only when validation performance
is stalled, the event fits the remaining resource budget, and subsequent validation shows useful
improvement. Keep the update if it passes the frozen rule; otherwise reject or roll it back with
the event logged. Specify the exact observation window, noise treatment, allocation cost, and
rollback semantics before implementing this extension. It is **proposed work**, not a claim about
the existing gate. Discard it if a simpler fixed-capacity method works as well.

Do not revive the deleted neuron pager as if its previous failure had never happened. A new paging
study needs a changed design or workload, a stated reason it could overcome the measured overhead,
and a control reproducing the old failure. Persistent event storage can be studied independently.

### Hardware and affordability

Begin with equipment already available. Use a CPU-only configuration on the development computer
for the pilot, with GPU use explicitly disabled and checked at runtime. That is a CPU measurement
on a workstation, not proof of inexpensive-edge deployment.

For the confirmatory study, borrow an ordinary laptop or low-power mini PC and verify the entire
3.14t dependency stack. An ARM board is a later portability test if compatible builds are available.
Do not buy a board based on model size alone. ESP32 hardware can serve as a sensor/actuator
interface; it does not currently run the full Python brain.

Suggested discussion budgets are **1 GiB whole-process peak memory, 20 Hz controller updates
(50 ms period), and a USD 300 compute-device acquisition cap**. These are proposed constraints,
not measured requirements or current market-price claims. If runtime overhead alone exceeds the
memory budget, report that and reconsider scope before freezing the protocol. Keep the budget
fixed for the final campaign. List robot, sensors, storage, and meter costs separately from compute.

Record actual device model, CPU, RAM, storage, OS, interpreter, dependency versions, thread counts,
power mode, and cooling. Include dated purchase quotes or receipts only once the device is chosen.
Cap cores as a scaling experiment, but do not equate a core-limited workstation with a small board.

Treat the memory budget as a whole-system sum:

`M_total = M_runtime + M_weights + M_neuron_state + M_learning_state + M_replay + M_indexes + M_transient`

Measure each term where possible, including eligibility traces and temporary allocation during
growth. A per-neuron estimate alone cannot establish how many units fit. Persistent storage also
has a finite budget: define what happens when it fills, preserve the project's retention contracts,
and refuse further allocation when required. "Grows locally" means useful development within a
declared operating envelope, not infinite growth on fixed hardware.

### Measurements

| Quantity | Measurement boundary |
|---|---|
| Task quality | Success rate, distance error, adaptation curves, retained old-task scores, and safety violations. |
| Adaptation cost | Samples, updates, elapsed time, and joules needed to reach the frozen validation target; report failures to reach it. |
| Latency | Observation-ready to action-ready p50/p95 and deadline misses, including encoding, retrieval, growth, and learning interference. Report cold start separately. |
| Memory | Peak process-tree RSS/working set from process start, including construction and transient growth; model-only bytes as a separate diagnostic. Include GPU memory if used. |
| Storage | Lake/index size, bytes written per episode, replay reads, checkpoint size, and restart time. |
| Compute | CPU time, threads, spike/event counts, SynOps and MACs using explicit counting conventions; counters are not joules. |
| Energy | External power measurement where available, with idle baseline, sample rate, integration interval, uncertainty, and a sensor boundary stated. |
| Cost | Compute purchase cost, electricity over a stated workload/duty cycle, and any paid training or cloud costs. |

For energy, integrate measured power over a fixed interval: `E = sum(P_i * delta_t_i)`. Report total
energy and idle-subtracted dynamic energy separately. If only package counters are available,
label them CPU-package energy and do not present them as wall-plug or whole-device energy. Run long
enough for the meter's resolution, calibrate with an idle and a loaded condition, and report error.
If no suitable measurement is available, H1 cannot be evaluated as an energy claim.

For simulation, report both the agent-only measurements and the full agent-plus-MuJoCo workload.
For a physical robot, separate compute power from motors and sensors, and also report total-system
power when claiming robot-level efficiency. Include background consolidation and replay over the
deployment duty cycle. A quiet inference step can conceal expensive learning between decisions.

### Statistical protocol and failure controls

- Start with 3-5 pilot seeds for variance and feasibility; exclude these from confirmatory tests.
  Plan at least 10 independent training seeds, then justify the final number using pilot variance,
  the practical effect threshold, and available compute. Multiple episodes under one trained seed
  are nested observations, not independent training runs.
- Pair seeds and task schedules across methods. Plot every seed and learning curve. Report effect
  sizes and 95% uncertainty intervals; resample at the independent-seed level, preserving paired
  comparisons. For several tasks, explain the aggregation and include per-task results.
- Freeze a primary comparison and endpoint. Treat other comparisons as secondary/exploratory and
  address multiplicity. Select the best fixed baseline using validation, not final test results.
- Randomize execution order; use comparable warm-up, power modes, thermal conditions, and background
  load. Record all failed, timed-out, or out-of-memory runs. Do not silently drop them.
- Demonstrate positive learning on a solvable control task. Shuffle supervision or disable updates
  to ensure the scorer detects failure. Check a known allocation spike and known delay to validate
  the memory and timing instruments themselves.
- For restart claims, save and resume into isolated state, then compare behavior with a matched
  uninterrupted run. Check deliberately altered or missing state is detected.

## 5. Presentation: ten slides in fifteen minutes

| Slide | Time | Content and visual | Speaker's point |
|---|---:|---|---|
| 1. The application | 1:00 | A local robot adapting after its environment changes | "Can useful adaptation fit the computer its owner already has?" |
| 2. Define the constraint | 1:30 | Quality, latency, RAM, energy, and purchase cost | "We need a measured budget, not a neuron-count promise." |
| 3. The platform | 2:00 | README architecture diagram; identify the actual experimental path | "These components exist; the task-level benefit is the research question." |
| 4. What runs today | 1:30 | Short recorded simulation demo and storage example | "This is a reproducible starting apparatus, not a trained manipulation result." |
| 5. What the evidence corrected | 1:30 | Bit-packed vs byte raster; fair dense energy comparison; failed paging result | "The existing negatives tell us where a fair experiment must be careful." |
| 6. Proposed contribution | 2:00 | H1/H2/H3 with the primary endpoint highlighted | "A bounded growth rule and measured quality-cost frontier." |
| 7. Experiment design | 2:00 | Task sequence, held-out evaluation, baselines, and ablations | "We will isolate growth and memory while matching data and budgets." |
| 8. Hardware and measurement | 1:00 | Local CPU device, measurement boundaries, power meter | "No edge-energy claim until we measure on the target device." |
| 9. Deliverables | 1:30 | Eight-week milestones and planned paper figures | "Each phase has a concrete artifact and a stop-or-revise condition." |
| 10. The collaboration request | 1:00 | Three decisions requested from the professor | "Help choose the task, practical success bar, and strongest comparison." |

For slide 4, rehearse `snn-demo` and the telemetry module in an isolated environment. Prepare a
short screen recording or saved terminal output as a fallback. Label these as component demos;
do not stage a live adaptive-learning result that has not been reproduced. The September 17
handoff recorded the daemon as stopped then; this documentation task does not restart it.

Keep the C0 reasoning result and detailed runtime requirements in backup slides. If asked, show
0/20 honestly and explain that the proposed paper studies a bounded control task. Do not interpret
that score as either a robotics result or a representative estimate of all reasoning ability.

### Questions the professor is likely to ask

| Question | Answer to prepare |
|---|---|
| Why spikes instead of a small neural network? | We have not established they are better; a compact dense controller is a required baseline. Temporal sparsity is the hypothesis. |
| What does "grows" mean? | Measured capacity changes that improve validation/test behavior under explicit budgets; show allocation, update, and retention curves separately. |
| Where is the novelty? | Potentially the growth decision rule and its measured local-adaptation tradeoff. The literature comparison must establish what is new. |
| Can it run on a cheap device today? | Some components run locally; full-stack footprint, portability, energy, and task quality on a chosen inexpensive device are still to be measured. |
| How do you know memory helps? | Compare replay/retrieval on and off at equal budgets and evaluate old tasks after new training and restart. |
| What if the dense baseline wins? | Report the crossover conditions and negative result; revise the algorithmic claim instead of changing the baseline after seeing the outcome. |
| Is this a biological brain or consciousness model? | Anatomical organization is a design aid; neither equivalence nor subjective experience is established by this study. |
| Is the code ready for another lab? | The demo is runnable and local validation is recorded in the assessment. Independent clean-install and cross-platform reproduction remain necessary. |

## 6. Eight-week sequence

This schedule assumes access to a compatible machine and a working task adapter. It is an estimate,
not a commitment that an accepted paper will result in eight weeks.

| Week | Work | Concrete exit artifact |
|---|---|---|
| 1 | Agree on paper scope; verify current tree and finish release prerequisites through the normal project workflow; review closest work | One-page proposal, immutable baseline commit, environment manifest, claim/evidence table |
| 2 | Build task adapter, splits, score definitions, and a useful classical/compact-dense baseline; establish CPU-only execution | A clean-run baseline report with positive and negative scoring controls |
| 3 | Measure process footprint, adaptation cost, latency and power; run pilot seeds; choose hardware and budgets | Instrument calibration, pilot variance, frozen confirmatory protocol and compute estimate |
| 4 | Connect and test the selected growth and memory path; implement only the missing experimental controls | All ablations runnable; task-level causal-path checks; rollback/restart behavior verified |
| 5 | Run the frozen confirmatory campaign, preserve raw logs and failed runs | Complete seed-by-method dataset and immutable configuration bundle |
| 6 | Replicate on the inexpensive target device; test sustained operation and restart; optional physical-arm validation if calibrated | Hardware table, thermal/duty-cycle results, device replication report |
| 7 | Analyze quality-cost frontiers and ablations; test robustness to task order and selected disturbances | Figures, uncertainty intervals, limitations, and a second-person reproduction |
| 8 | Draft and revise the paper; package source, configs, plotting commands, and data provenance | Submission-ready draft and reproducibility archive, subject to professor review |

Decision points: if useful learning is absent in week 2, focus on that before growth. If the Python
stack cannot fit the hardware in week 3, narrow the deployment claim or make runtime reduction an
explicit separate contribution. If growth has no benefit, retain the negative result and do not
expand into more cognitive faculties to avoid the finding. If energy measurement is unavailable,
write a latency/memory study with an energy question left open.

## 7. Paper and artifact plan

Suggested sections: introduction and deployment problem; related work; system and growth method;
experimental protocol; adaptation and efficiency results; ablations and failure analysis;
limitations and conclusion.

Prepare these figures before writing conclusions:

1. Actual experimental data/control path with measurement boundaries.
2. Adaptation curves across sequential task changes, with uncertainty and individual seeds.
3. Old-task retention matrix before and after new learning and restart.
4. Task quality versus measured joules and peak memory, showing all baseline operating points.
5. Capacity and storage growth alongside latency and memory over time.
6. Ablation effects and a failure/crossover plot, including regimes where dense control wins.

Release the exact commit, dependency manifest, configurations, splits, seed list, hardware metadata,
raw per-run observations, unsuccessful runs, and scripts that regenerate the tables and figures.
Record data licenses and permissions for any human sensor recordings. No pretrained weight or
private local memory should be required for the published experiment unless explicitly distributed
under suitable terms. Include an honest scope statement for simulation-only results.

Discuss a robotics or efficient-learning workshop as an initial venue if the study is narrow, or
a fuller robotics/neuromorphic paper once target-device and task evidence is strong. Select an
actual venue with the professor after scope and results are clear; no acceptance or deadline is
assumed here.

## 8. Literature to position against

These sources are starting points checked during this assessment, not an exhaustive novelty search.

| Primary source | Relevance to this proposal |
|---|---|
| [Yik et al., NeuroBench, Nature Communications, 2025](https://www.nature.com/articles/s41467-025-56739-4) | Distinguishes algorithm-level measures from hardware-system evaluation; guides efficiency reporting. |
| [Kwon et al., TinyTrain, ICML 2024](https://proceedings.mlr.press/v235/kwon24c.html) | Shows that resource-aware local learning is also pursued with conventional DNNs; local adaptation alone is not a novelty claim. |
| [Cai et al., TinyTL, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/81f7acabd411274fcf65ce2070ed568a-Abstract.html) | Motivates measuring training activations and whole learning memory, not just parameter counts. |
| [Bellec et al., e-prop, Nature Communications, 2020](https://doi.org/10.1038/s41467-020-17236-y) | Foundational recurrent-spiking learning method; identify the exact rule and scope used in SNDL. |
| [Chen et al., Net2Net, ICLR 2016](https://arxiv.org/abs/1511.05641) | Function-preserving widening has established precedent; SNDL must credit and distinguish its adaptation. |
| [Zeng et al., BrainCog](https://arxiv.org/abs/2207.08533) | A closely related broad brain-inspired SNN platform; compare concrete capabilities and experiments rather than module names. |
| [Agarwal et al., Statistical Precipice, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | Supports careful uncertainty and aggregate reporting across training runs and tasks. |

Expand the review with recent budgeted dynamic networks, online robot adaptation, continual-learning
replay, compressed event storage, and actual CPU/neuromorphic deployments. For each close paper,
record hardware, task, update rule, growth policy, replay budget, measured energy boundary, and code
availability. Do not compare published headline numbers across incompatible tasks or hardware.

## 9. The request to the professor

Ask for agreement on **one task, one practical resource budget, and one primary comparison**.
Request methodological supervision, access to a modest local device and suitable power measurement,
and a review of the novelty matrix. Physical-arm access is useful only after the simulator study
and calibration requirements are satisfied.

Bring the rewritten README, the current-state assessment, the rehearsed component demo, and this
plan. The immediate next step is to freeze the first experiment with the professor, not to promise
an inexpensive general-purpose brain before measuring one.
