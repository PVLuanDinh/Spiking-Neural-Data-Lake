# From SNDL to a paper on general-purpose continual learning

Prepared 2026-09-20 for a professor in AI and robotics; revised 2026-09-22 so the study treats SNDL
as a general-purpose learner rather than a robot controller. Robotics remains one task family among
several. Assumes a 15-minute presentation. This is a research proposal, not a report of completed
experiments. See the [assessment](RESEARCH_ASSESSMENT.md) for the evidence available today, and
section 10 for the project's plan of record as of 2026-09-22.

## 1. The proposition to take into the meeting

**Study whether one persistent spiking learner can acquire and retain competence across different
kinds of tasks (event vision, event audio, symbol sequences, and sensorimotor control) within a
fixed resource budget, and whether earned growth and durable memory are what make the difference.**

Suggested working title:

> Earned Growth and Durable Event Memory for General-Purpose Continual Learning with Spiking Networks

The project's long-term goals are general intelligence in a brain that builds itself and, as an
openly pursued research question, consciousness. This study claims neither. Use "one brain that
keeps learning whatever it is given" to explain the motivation, and define **general-purpose**
experimentally, as a property of the learner rather than of its results:

1. **One learner, one memory.** A single learner instance and a single durable lake serve every task
   family. There are no per-task model copies and no per-task memory stores.
2. **One configuration.** Learning rules, the growth rule, and hyperparameters are frozen before the
   confirmatory run and shared across families. Modality-specific input encoders and any per-task
   readouts are declared in advance and counted in the resource budget.
3. **Breadth.** The task families differ in modality and structure. Each is scored against its own
   trivial floor: chance, majority, persistence, or an n-gram, whichever is strongest.
4. **Retention.** After each new task, the learner is re-scored on held-out data from every earlier
   task, including after a process restart.
5. **A declared budget.** Memory, compute, and latency limits are fixed before the confirmatory run.

A learner can be general-purpose in this sense and still perform poorly; measuring how poorly, and
why, is part of the result. Growth without improved held-out behavior is allocation, not
demonstrated intelligence, and breadth of scope is not evidence of general intelligence. Any
comparison with human performance needs a human number collected under the same protocol; without
one the comparison is refused, as the project's
[capability bar](../src/snn_data_lake/substrate/capability_bar.py) already requires.

The current project supplies an apparatus: spiking modules with local (e-prop) learning, growth
machinery whose units must earn their place on held-out data, one append-only durable lake, input
paths for event vision, event audio, text, and a simulated arm, and extensive checks. The proposed
contribution is a **measured cross-task acquisition and retention result, a quality-cost frontier,
and a tested earned-growth rule**. Novelty of that combination must be established through the
literature review; the existence of spikes, a data lake, or anatomical module names is not itself a
novel result.

### A 60-second opening

> I want to study a learner that keeps learning whatever it is given (images, sound, text, and
> control) on one machine, without retraining from scratch for each new task and without
> forgetting the old ones. I have built a research platform: a spiking network with local learning
> rules, capacity that must earn its place on held-out data, and one durable event memory that
> survives restarts, with input paths for event cameras, audio, text, and a simulated robot arm.
> Several components work, but I have not shown a general-purpose learning advantage. Our records
> include negatives: a fair dense baseline wins our energy comparison, an everyday reasoning battery
> scores 0 of 20, and a neuron-paging design was deleted after missing its latency bound. I would
> like to turn the platform into a controlled study: across a sequence of different tasks under one
> fixed budget, do earned growth and durable memory improve acquisition and retention, and how far
> is a local spiking learner from strong conventional continual learners? I am seeking your help
> with the task suite, the baselines, and the scope of a first paper.

## 2. What to claim now

| Defensible statement | Evidence or qualification |
|---|---|
| There is a runnable local simulation and event-memory platform. | Demo, storage, query, and scorer checks rerun in the assessment. |
| Sparse storage saves bytes on the measured sparse workload. | 417,976 B versus 3,200,000 B bit-packed dense storage; 7.7x on that example. The 61x figure uses a byte raster. |
| The learner has input paths for several modalities. | E-prop encoders for [event vision](../src/snn_data_lake/cortex/eprop_nmnist.py) and [event audio](../src/snn_data_lake/cortex/eprop_shd.py), an [online next-character learner](../src/snn_data_lake/cortex/language_acquisition.py), and the [SO-101 body](../src/snn_data_lake/brainstem/so101_arm.py). An input path is not task competence, and small-scale accuracies are not benchmark results. |
| Learning on the governed path is local. | The e-prop encoders' self-checks compare their gradient with an independent BPTT gradient (cosine above 0.5). Two gates partially mechanize the rule; review covers the rest. The language learner uses a local delta rule and produces sub-word babble, not fluent text. |
| The code has mechanisms intended to support growth and persistence. | Growth gates, neurogenesis, widening, restart tests; usefulness must be measured on the selected tasks. |
| Width improved one recorded next-category task over a one-unit baseline. | The corrected SC-012a mean gain is 15.1 percentage points across eight seeds; it does not beat the incumbent admission policy or previous-category predictor, and review found a non-discriminating null veto. See the [assessment](RESEARCH_ASSESSMENT.md). |
| The brain's own live stream is a weak curriculum. | The live learner sits below its persistence floor on 8 of 8 seeds ([L1b record](../specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md)), and the recorded percept packs carry no first-order label succession ([W2 record](../specs/022-unified-brain-arc/evidence/w2-context-latents.md)). Benchmarks with known structure are needed. |
| Everyday reasoning is weak on the recorded items. | C0 scores 0 of 20 on each of five seeds while its controls pass ([record](../specs/022-unified-brain-arc/evidence/c0-everyday-battery.md)). Twenty items are not a representative sample of reasoning. |
| The project records unsuccessful experiments. | Neuron paging was deleted after a latency-bound failure despite a memory improvement; a goal curriculum was retired on its own measurement. |
| General-purpose learning and inexpensive local operation are the intended direction. | Both remain hypotheses requiring the task suite, baselines, and resource measurements below. |

Do not claim general intelligence, human-level or superhuman performance, a proven low-power brain,
unbounded growth on finite hardware, consciousness, an all-spiking system, or MCU deployment. The
current runtime includes CPython 3.14t, PyTorch, NumPy, PyArrow, MuJoCo, and Brian2, which serves as
the equation reference while a Torch bridge executes it. A small model does not erase that
footprint. Keeping a fact on disk does not prove that a learned skill remains usable. "Never
forgets" means in this project only that a declared
[immortal record class](../src/snn_data_lake/substrate/immortal_retention.py) is never evicted, not
that every observation is kept or every skill retained.

## 3. Choose one first-paper question

**Primary question:** in one persistent spiking learner, do earned growth and durable replay improve
acquisition and retention across a sequence of heterogeneous tasks, compared with the same learner
with both disabled, at a matched memory and compute budget?

### The task suite

The suite is chosen from inputs the brain already has, never invented for the comparison. The task
adapters and the complete comparison harness are work to build, not a finished benchmark.

| Family | Candidate task | Existing path | Main open work |
|---|---|---|---|
| Event vision | N-MNIST digits as sequential class groups, for example five two-class tasks | [e-prop N-MNIST encoder](../src/snn_data_lake/cortex/eprop_nmnist.py); Tonic via the `nmnist` extra | Continual splits, readout policy, accuracy at useful scale |
| Event audio | SHD spoken digits (0-9 in English and German, 20 classes), English then German as a natural domain shift | [e-prop SHD encoder](../src/snn_data_lake/cortex/eprop_shd.py) | The official test split holds out two of twelve speakers but also contains seen speakers, so report unseen speakers separately; SHD is hard at the current small scale |
| Symbol sequences | Next-character prediction on a public-domain corpus, plus a planted-grammar stream with known structure | [online language learner](../src/snn_data_lake/cortex/language_acquisition.py) (local delta rule) | Task adapter; bits-per-character scoring against n-gram floors |
| Sensorimotor control | Continual reaching in the SO-101 MuJoCo body under changed goal regions and simulated payload or friction | [SO-101 body](../src/snn_data_lake/brainstem/so101_arm.py), used through its existing gate | Observation/action adapter, success rule, classical controller baseline |
| Reasoning probe (diagnostic only) | The 20-item everyday battery | [C0 scorer](../src/snn_data_lake/cortex/everyday_battery.py) | None to start; always reported, never an endpoint |

Recommended first paper: event vision, event audio, and symbol sequences. Their encoders or
learners already exist and their datasets are public. Add sensorimotor control once its adapter
passes the same controls; it is the one family that acts on the world. Start with a two-family
pilot (event vision, then event audio), because a failure there is cheap and informative. The
SO-101 files belong to the project's safety envelope: the study uses the simulator through its
existing gate and never modifies them.

**The world before the wire.** Before any growth or memory arm runs, show on each task's exact
stimulus that it carries structure the scorer can detect: a simple baseline must beat the family's
trivial floor, and a label-shuffled copy must not. This is why the study uses benchmark tasks with
known structure rather than the brain's persistence-shaped live stream.

### Candidate hypotheses, to ratify before the confirmatory run

| Hypothesis | Proposed success rule | Outcome that rejects or limits it |
|---|---|---|
| H1 (primary): earned growth and durable replay together improve cross-task learning | After the full sequence, a higher mean normalized held-out score across all tasks (ACC) than the same learner with growth and replay disabled, by a pre-registered practical margin, at matched updates and budget | No reliable difference, or the gain disappears once growth and replay costs are charged against the budget |
| H2: one configuration is broad | The frozen configuration clears each family's trivial floor on held-out data, with the paired interval excluding the floor | Any family stays at its floor; the result is then reported as a narrower, per-family claim |
| H3: growth buys useful capacity | Better held-out acquisition than growth disabled, and no worse than a fixed network pre-sized to the grown network's final width | Added units fail to change held-out behavior, or the pre-sized twin matches the grown network, making it a width effect rather than a growth effect |
| H4: durable memory improves retention | Better backward transfer (BWT) than an equal-budget arm without replay or retrieval, at the pre-registered replay budget and after a restart from the lake | No retention gain, a gain that vanishes after charging retrieval and replay costs, or one that comes from overfitting replayed examples |
| H5: the learner is competitive on a quality-cost frontier | At a frozen quality floor on every task, at least 2x lower cost than the strongest eligible conventional continual learner on one pre-registered axis: measured energy if a meter is available, otherwise peak memory | Conventional baselines dominate the frontier on that axis |

The **2x threshold is a proposed practical bar**, not a derived law or an achieved result; the H1
margin, floors, and budgets are proposals too. Choose them with the professor after a separate
pilot and freeze them before collecting confirmatory outcomes. H1 is the primary endpoint; H2-H5
are secondary. Do not rename a failed primary endpoint as secondary after seeing results.

Normalize each task's score as `(score - floor) / (reference - floor)`, where the floor is the
strongest trivial baseline and the reference is a specialist trained on that task alone. Report raw
scores beside normalized ones; normalization must never hide a family sitting at its floor.

The prior expectation is not favorable, and the proposal should say so. Conventional replay-based
continual learners will probably beat a local spiking learner on accuracy. The study remains
publishable if it measures that gap honestly and attributes whatever growth and memory do buy.

## 4. Build the experiment

### Task sequence and splits

1. Define each task's score before the final run: accuracy for classification, bits per character
   for sequences, and success rate with continuous distance error for reaching, so a threshold
   cannot conceal poor behavior. Define the floor and the specialist reference for each task.
2. Train sequentially on at least four tasks drawn from at least three families. Randomize and
   counterbalance task order across independent runs, so an order effect cannot pass as a method
   effect.
3. Hold out test data for every task. After each task block, freeze learning and evaluate every task
   seen so far, filling the retention matrix `R[i][j]`: the score on task j after training through
   task i.
4. Separate training, validation, and final test data and seeds. Growth decisions and
   hyperparameter selection may use training/validation evidence only. Never feed final test scores
   back into the growth gate, even though the existing gate accepts an accuracy trajectory.
5. Split by recording session, speaker, device, or document, as each dataset allows. Adjacent
   windows from one recording must not straddle training and test sets.
6. Declare the incremental-learning scenario (task-, domain-, or class-incremental) and the task
   identity available at test time. Recommended default: the modality is known from the input
   channel, and no task label is given within a modality.

All agents receive the same information and the same number of updates. Give dense baselines
equivalent temporal history; comparing an SNN with a memoryless model on a temporal task is unfair.
Freeze encodings, input rates, and simulator settings.

### Baselines and ablations

| Arm | Purpose |
|---|---|
| Trivial floors: chance, majority, persistence, n-gram | Define each score's zero and show the task is learnable at all. |
| Specialists: one model per task, trained on that task alone | Upper reference for normalization; not a win condition. |
| Dense network trained sequentially without protection | Show that forgetting occurs. |
| Dense network with experience replay or DER++ at the same memory budget | Strongest conventional continual learner. |
| Dense network with EWC or synaptic intelligence | Continual learning without stored examples. |
| Fixed-size SNN at the initial width, same local rule | Isolate growth from the starting system. |
| Fixed-size SNNs across a capacity sweep, including a twin pre-sized to the grown width | Separate growth from width. |
| SNDL learner with earned growth and durable replay | Proposed full condition. |
| Same learner, growth disabled | Isolate growth with initialization, inputs, updates, and memory held constant. |
| Same learner, replay and retrieval disabled but logging retained | Isolate the learning value of stored experience without an unfair logging-cost reduction. |
| Matched random growth, or growth events with learning disabled | Test whether changes merely allocate capacity or alter randomness. |
| Full learner restarted from the lake between tasks | Test that retention survives a new process. |
| Classical PID or inverse-kinematics controller, control family only | Establish whether learning is needed for the reaching task at all. |

Use the same data and equal tuning effort for every arm. Report sample-matched learning curves and
resource-matched outcomes. Record offline pretraining, including any GPU time, instead of hiding it
outside the "local" cost.

**Learning-rule boundary.** The SNDL arms use the project's local rules: e-prop and local
plasticity, with no backpropagation through time on the brain's live path. The
[gradient-claim](../src/snn_data_lake/substrate/gradient_claim_gate.py) and
[weight-transport](../src/snn_data_lake/substrate/eprop_transport_gate.py) gates check part of that
rule. Conventional baselines may use backpropagation: they are comparators, not the proposed
method, and need not be restricted to the SNN's rule. Keep them in an evaluation harness that the
brain's live path cannot reach; otherwise the project's e-prop policy brings them into scope. The
repository's surrogate-gradient spiking language-model trainer is a baseline for the same reason.

Use an INT8 baseline only where its actual implementation and update workflow support it. The
current energy model's INT8 comparison is a warning about fairness, not a deployed INT8 model.

### Growth rule to investigate

Start by tracing the selected live growth path and its consumer. Reuse the existing growth gate
where applicable. Check with a neuter test that its signal reaches the task outputs; adding neurons
or changing a counter alone is not evidence of a functional change.

The plan of record's rule is **synaptic first, growth on plateau**: existing units learn online, a
new unit is admitted only after synaptic learning stalls on validation error, and it stays only if
it lowers held-out error. A proposed resource-aware extension would also require the unit to fit
the remaining budget. Keep the update if it passes the frozen rule; otherwise reject or roll it
back with the event logged. Specify the observation window, noise treatment, allocation cost, and
rollback semantics before implementing the extension. It is **proposed work**, not a claim about the
existing gate. Discard it if a simpler fixed-capacity method works as well.

Two recorded negatives shape the design. Hebbian updates to grown units' input weights harmed the
readout when the readout could not co-adapt, so moving input weights need a co-adapting readout
([L1b](../specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md),
[L1c](../specs/022-unified-brain-arc/evidence/l1c-online-readout.md)). An earlier growable readout
overfit its finite replay buffer, and more replay made it worse (recorded in the changelog); H4's
replay budget and its held-out evaluation exist to detect that failure.

Do not revive the deleted neuron pager as if its previous failure had never happened. A new paging
study needs a changed design or workload, a stated reason it could overcome the measured overhead,
and a control reproducing the old failure. Persistent event storage can be studied independently.

### Durable memory path

Durable replay is the least finished part of the apparatus. Today the interaction learner's replay
ring is process-local and is not rebuilt on restart, and deep whole-lake recall is a scan rather
than bounded-cost retrieval (R workstream of the
[plan of record](../specs/022-unified-brain-arc/PLAN-FINAL-2026-09-02.md)). The restart arm and H4's
restart condition therefore depend on restart-safe typed replay and its evaluation (plan tasks
T103-T104) passing their own restart proofs. If they have not landed, run the in-process replay arm
and state that the restart claim is out of scope.

### Resource budgets and hardware

General-purpose learning is measured under a declared budget. Inexpensive-device deployment is a
separate, secondary claim.

Begin with equipment already available. Run the pilot CPU-only on the development computer, with
GPU use explicitly disabled and checked at runtime. That is a CPU measurement on a workstation, not
proof of inexpensive-edge deployment. If the paper keeps a deployment claim, replicate on an
ordinary laptop or low-power mini PC after verifying the entire 3.14t dependency stack there; an ARM
board is a later portability test if compatible builds are available. Do not buy a board based on
model size alone. ESP32 hardware can serve as a sensor/actuator interface; it does not run the full
Python brain.

Suggested discussion budgets for the deployment variant are **1 GiB whole-process peak memory, 20 Hz
controller updates (50 ms period) for the control family, and a USD 300 compute-device acquisition
cap**. These are proposed constraints, not measured requirements or current market-price claims. If
runtime overhead alone exceeds the memory budget, report that and reconsider scope before freezing
the protocol. Keep the budget fixed for the final campaign. List robot, sensors, storage, and meter
costs separately from compute.

Record actual device model, CPU, RAM, storage, OS, interpreter, dependency versions, thread counts,
power mode, and cooling. Include dated purchase quotes or receipts only once a device is chosen. Cap
cores as a scaling experiment, but do not equate a core-limited workstation with a small board.

Treat the memory budget as a whole-system sum:

`M_total = M_runtime + M_weights + M_neuron_state + M_learning_state + M_replay + M_indexes + M_transient`

Measure each term where possible, including eligibility traces and temporary allocation during
growth. A per-neuron estimate alone cannot establish how many units fit. The conventional
baselines' replay buffers count against the same budget as the lake's replay. Persistent storage
also has a finite budget: define what happens when it fills, preserve the project's retention
contracts, and refuse further allocation when required. "Grows locally" means useful development
within a declared operating envelope, not infinite growth on fixed hardware.

### Measurements

| Quantity | Measurement boundary |
|---|---|
| Task quality | Per task: accuracy, bits per character, or reaching success and distance error, raw and normalized; safety violations for the control family. |
| Retention and transfer | The full retention matrix; average final score (ACC), backward transfer (BWT), and forward transfer (FWT), per family and overall. |
| Acquisition cost | Samples, updates, elapsed time, and joules needed to reach each task's frozen validation target; report failures to reach it. |
| Latency | Input-ready to output-ready p50/p95 per family and deadline misses, including encoding, retrieval, growth, and learning interference. Report cold start separately. |
| Memory | Peak process-tree RSS/working set from process start, including construction and transient growth; model-only bytes as a separate diagnostic. Include GPU memory if used. |
| Storage | Lake/index size, bytes written per task, replay reads, checkpoint size, and restart time. |
| Compute | CPU time, threads, spike/event counts, SynOps and MACs using explicit counting conventions; counters are not joules. |
| Energy | External power measurement where available, with idle baseline, sample rate, integration interval, uncertainty, and a sensor boundary stated. |
| Cost | Compute purchase cost, electricity over a stated workload/duty cycle, and any paid training or cloud costs. |

For energy, integrate measured power over a fixed interval: `E = sum(P_i * delta_t_i)`. Report total
energy and idle-subtracted dynamic energy separately. If only package counters are available, label
them CPU-package energy and do not present them as wall-plug or whole-device energy. Run long enough
for the meter's resolution, calibrate with an idle and a loaded condition, and report error. If no
suitable measurement is available, H5 falls back to peak memory and makes no energy claim.

For the control family, report agent-only measurements and the full agent-plus-MuJoCo workload
separately. For a physical robot, separate compute power from motors and sensors. Include
background consolidation and replay over the whole duty cycle: a quiet inference step can conceal
expensive learning between decisions.

### Statistical protocol and failure controls

- Start with 3-5 pilot seeds for variance and feasibility; exclude these from confirmatory tests.
  Plan at least 10 independent training seeds, then justify the final number using pilot variance,
  the practical effect threshold, and available compute. Multiple evaluations under one trained
  seed are nested observations, not independent runs.
- Pair seeds and task orders across methods. Plot every seed and learning curve. Report effect sizes
  and 95% uncertainty intervals, resampling at the independent-seed level while preserving pairs.
  Report every family separately; an aggregate score never replaces per-family results.
- Freeze H1 as the single primary comparison. Correct the secondary hypotheses for multiplicity,
  for example with Holm-Bonferroni. Select the best fixed baseline using validation, not final test
  results.
- Randomize execution order; use comparable warm-up, power modes, thermal conditions, and background
  load. Record all failed, timed-out, or out-of-memory runs. Do not silently drop them.
- Give every family a reachable failure: show positive learning on a solvable variant, and show
  that a label-shuffled or update-disabled run scores at the floor, so the scorer detects failure.
  Check a known allocation spike and a known delay to validate the memory and timing instruments
  themselves.
- For restart claims, save and resume into isolated state, then compare behavior with a matched
  uninterrupted run. Check that deliberately altered or missing state is detected. Never use the
  owner's live brain or its lake as an experiment subject; every run gets a scratch home.

### Threats to validity

| Threat | Mitigation |
|---|---|
| Construct: "general-purpose" is defined by this protocol | State the definition up front, report per-family results, and never generalize beyond the tested families. |
| Construct: small benchmarks chosen from existing encoders | Name the selection rule; treat results as evidence about these tasks, not about intelligence in general. |
| Internal: unequal tuning effort favors one arm | Fix a tuning budget per arm, select on validation only, and publish every configuration. |
| Internal: task order drives the result | Counterbalance orders and report order-by-method interactions. |
| Conclusion: many comparisons | One primary endpoint, multiplicity correction, and secondary results labeled as such. |
| External: workstation results may not transfer to inexpensive devices | Make no deployment claim without replication on the target device. |

## 5. Presentation: ten slides in fifteen minutes

| Slide | Time | Content and visual | Speaker's point |
|---|---:|---|---|
| 1. The question | 1:00 | One learner receiving vision, audio, text, and control tasks in sequence | "Can one learner keep acquiring new kinds of tasks without forgetting the old ones?" |
| 2. Define general-purpose | 1:30 | One learner, one memory, one configuration; breadth, retention, budget | "General-purpose is a property we test, not a claim of intelligence." |
| 3. The platform | 2:00 | README architecture diagram; identify the actual experimental path | "These components exist; the cross-task benefit is the research question." |
| 4. What runs today | 1:30 | Short recorded component demos: storage, temporal query, an e-prop encoder self-check | "This is a reproducible starting apparatus, not a trained multi-task result." |
| 5. What the evidence corrected | 1:30 | Fair dense energy comparison; failed paging result; C0 at 0 of 20; persistence-shaped live stream | "The existing negatives tell us where a fair experiment must be careful." |
| 6. Proposed contribution | 2:00 | H1-H5 with the primary endpoint highlighted | "Earned growth and durable memory, tested across modalities under one budget." |
| 7. Experiment design | 2:00 | Task sequence, retention matrix, baselines, and ablations | "We isolate growth and memory while matching data, updates, and budgets." |
| 8. Measurement and hardware | 1:00 | Resource boundaries, CPU-only pilot, power meter | "No efficiency claim until we measure it." |
| 9. Deliverables | 1:30 | Eight-week milestones and planned paper figures | "Each phase has a concrete artifact and a stop-or-revise condition." |
| 10. The collaboration request | 1:00 | Three decisions requested from the professor | "Help choose the task suite, the budget, and the strongest comparison." |

For slide 4, rehearse the [telemetry storage example](../src/snn_data_lake/substrate/spike_telemetry_hub.py),
the [temporal query example](../src/snn_data_lake/cortex/paradigm_b_engine.py), and one e-prop encoder
self-check in an isolated environment. Prepare a short screen recording or saved terminal output as
a fallback. `snn-demo` can illustrate the control family. Label all of these as component demos; do
not stage a live multi-task learning result that has not been reproduced, and do not demonstrate on
the owner's live brain.

Keep detailed runtime requirements in backup slides, but show the C0 result on slide 5: a
general-purpose pitch that hides its reasoning score invites the wrong question. Explain that 0 of 20
on twenty everyday items is neither a representative estimate of all reasoning ability nor a result
this study sets out to change.

### Questions the professor is likely to ask

| Question | Answer to prepare |
|---|---|
| What does "general-purpose" mean here? | One learner, one memory, and one frozen configuration across task families, scored for breadth and retention under a budget. It is not a claim of general intelligence. |
| Why spikes instead of a small neural network? | We have not established they are better; conventional continual learners are required baselines. Temporal sparsity and local learning are the hypotheses. |
| How does this differ from generalist models such as Gato? | Those are trained offline at large scale with backpropagation. This study asks what local, online learning with growth and a durable memory buys and costs at small scale; it does not compete on scale. |
| What does "grows" mean? | Measured capacity changes that improve held-out behavior under explicit budgets, compared with a network pre-sized to the same width. |
| Where is the novelty? | Potentially the earned-growth rule with durable replay, measured across modalities under a local-learning constraint. Growth-based continual learning in spiking networks already exists (DSD-SNN, section 8), so the literature comparison must establish what is new. |
| How do you know memory helps? | Replay and retrieval on and off at equal budgets, with old tasks evaluated after new training and after restart. |
| Does it reason? | Not on the recorded items: C0 reads 0 of 20 on five seeds. Reasoning is a diagnostic probe here, not an endpoint. |
| What if the dense baseline wins? | It probably will on accuracy. Report the gap, the crossover conditions, and the negative result; revise the claim instead of changing the baseline after seeing the outcome. |
| Can it run on a cheap device today? | Some components run locally; full-stack footprint, portability, energy, and task quality on a chosen inexpensive device are still to be measured. |
| Is this a biological brain or a consciousness model? | Anatomical organization is a design aid, not a claim of biological equivalence. Consciousness is a goal the project pursues openly; whether the system has experience is an open question that this study neither tests nor settles. |
| Is the code ready for another lab? | The demo is runnable and local validation is recorded in the assessment. Independent clean-install and cross-platform reproduction remain necessary. |

## 6. Eight-week sequence

This schedule assumes access to a compatible machine and working task adapters. It is an estimate,
not a commitment that an accepted paper will result in eight weeks. Week 4's restart arm depends on
plan rows R1-R2 (section 10); if they have not landed, the restart claim is scoped out.

| Week | Work | Concrete exit artifact |
|---|---|---|
| 1 | Agree on paper scope and task suite; verify current tree and finish release prerequisites through the normal project workflow; review continual-learning and SNN work | One-page proposal, immutable baseline commit, environment manifest, claim/evidence table |
| 2 | Build the two-family pilot (event vision, event audio): adapters, splits, scores, floors, and dense baselines; establish CPU-only execution | A clean-run baseline report with positive and negative controls per task |
| 3 | Add the symbol-sequence family; measure footprint and costs; run pilot seeds; choose budgets and hardware | Instrument calibration, pilot variance, frozen confirmatory protocol and compute estimate |
| 4 | Connect the selected growth and replay path; implement only the missing experimental controls | All ablations runnable; neuter checks that growth and replay reach task outputs; restart behavior verified or scoped out |
| 5 | Run the frozen confirmatory campaign, preserving raw logs and failed runs | Complete seed-by-method dataset and immutable configuration bundle |
| 6 | Robustness: task orders, restarts, sustained operation; optional control family and device replication | Order and restart results; hardware table if a device is used |
| 7 | Analyze retention matrices, frontiers, and ablations | Figures, uncertainty intervals, limitations, and a second-person reproduction |
| 8 | Draft and revise the paper; package source, configs, plotting commands, and data provenance | Submission-ready draft and reproducibility archive, subject to professor review |

Decision points: if the SNDL learner clears no family's floor in week 2, focus on learning before
growth. If it clears only one family, narrow the paper to that family and report the breadth
failure as a result. If growth has no benefit, retain the negative result and do not add cognitive
faculties to avoid the finding. If the Python stack cannot fit the budget, narrow the deployment
claim or make runtime reduction an explicit separate contribution. If energy measurement is
unavailable, write a memory and latency study with the energy question left open.

## 7. Paper and artifact plan

Suggested sections: introduction and the general-purpose continual-learning problem; related work;
the learner, growth rule, and durable memory; experimental protocol; acquisition and retention
results; resource frontier; ablations and failure analysis; limitations and conclusion.

Prepare these figures before writing conclusions:

1. The actual data and learning path, with measurement boundaries.
2. Retention matrices for each method, with individual seeds available.
3. ACC, BWT, and FWT per family and overall, with uncertainty intervals.
4. Normalized quality versus measured memory, compute, and joules, showing every baseline's operating points.
5. Capacity and storage growth across the task sequence, alongside latency and memory.
6. Ablation effects and a failure/crossover plot, including regimes where dense learners win.

Release the exact commit, dependency manifest, configurations, splits, seed list, task orders,
hardware metadata, raw per-run observations, unsuccessful runs, and scripts that regenerate the
tables and figures. Record dataset licenses and permissions for any human recordings. No pretrained
weight or private local memory should be required for the published experiment unless explicitly
distributed under suitable terms. State plainly which results are simulation-only.

Discuss a continual-learning, efficient-ML, or neuromorphic workshop as an initial venue if the
study is narrow, and a fuller paper once cross-family evidence is strong. A robotics venue fits only
if the control family carries the result. Select an actual venue with the professor after scope and
results are clear; no acceptance or deadline is assumed here.

## 8. Literature to position against

The first seven sources were checked during the 2026-09-20 assessment. The rest were added with this
revision and checked against their publisher, proceedings, or arXiv pages on 2026-09-22. None of
this is an exhaustive novelty search.

| Primary source | Relevance to this proposal |
|---|---|
| [Yik et al., NeuroBench, Nature Communications, 2025](https://www.nature.com/articles/s41467-025-56739-4) | Distinguishes algorithm-level measures from hardware-system evaluation; guides efficiency reporting. |
| [Kwon et al., TinyTrain, ICML 2024](https://proceedings.mlr.press/v235/kwon24c.html) | Shows that resource-aware local learning is also pursued with conventional DNNs; local adaptation alone is not a novelty claim. |
| [Cai et al., TinyTL, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/81f7acabd411274fcf65ce2070ed568a-Abstract.html) | Motivates measuring training activations and whole learning memory, not just parameter counts. |
| [Bellec et al., e-prop, Nature Communications, 2020](https://doi.org/10.1038/s41467-020-17236-y) | Foundational recurrent-spiking learning method; identify the exact rule and scope used in SNDL. |
| [Chen et al., Net2Net, ICLR 2016](https://arxiv.org/abs/1511.05641) | Function-preserving widening has established precedent; SNDL must credit and distinguish its adaptation. |
| [Zeng et al., BrainCog](https://arxiv.org/abs/2207.08533) | A closely related broad brain-inspired SNN platform; compare concrete capabilities and experiments rather than module names. |
| [Agarwal et al., Statistical Precipice, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | Supports careful uncertainty and aggregate reporting across training runs and tasks. |
| [van de Ven, Tuytelaars and Tolias, Three types of incremental learning, Nature Machine Intelligence, 2022](https://doi.org/10.1038/s42256-022-00568-3) | Scenario definitions (task-, domain-, class-incremental) the protocol must declare. |
| [Lopez-Paz and Ranzato, Gradient Episodic Memory for Continual Learning, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) | Defines the ACC, BWT, and FWT metrics used here. |
| [Kirkpatrick et al., EWC, PNAS, 2017](https://doi.org/10.1073/pnas.1611835114); [Zenke, Poole and Ganguli, Synaptic Intelligence, ICML 2017](https://proceedings.mlr.press/v70/zenke17a.html) | Regularization baselines that store no examples. |
| [Rolnick et al., Experience Replay for Continual Learning, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html); [Buzzega et al., Dark Experience for General Continual Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) | Strong replay baselines at matched memory. |
| [Rusu et al., Progressive Neural Networks, 2016 (preprint)](https://arxiv.org/abs/1606.04671); [Yoon et al., Dynamically Expandable Networks, ICLR 2018](https://openreview.net/forum?id=Sk7KsfW0-) | Growth-based continual learning that earned growth must be distinguished from. |
| [Han et al., Dynamic Structure Development of Spiking Neural Networks (DSD-SNN), IJCAI 2023](https://doi.org/10.24963/ijcai.2023/334) | The closest spiking precedent: grows neurons for new tasks and prunes redundant ones. Earned growth must be compared with it directly. |
| [Xiao et al., Hebbian Learning based Orthogonal Projection for Continual Learning of Spiking Neural Networks, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46b1be2b90c6addc84efdf5d7e90eebc-Abstract-Conference.html) | Continual learning in spiking networks through local Hebbian plasticity; a local-learning comparison point. |
| [Dohare et al., Loss of plasticity in deep continual learning, Nature, 2024](https://doi.org/10.1038/s41586-024-07711-7) | Plasticity loss as a failure mode to measure over a long task sequence. |
| [Reed et al., A Generalist Agent, TMLR, 2022](https://openreview.net/forum?id=1ikK0kHjvj) | A generalist model from large-scale offline training; the contrast case for "general-purpose". |
| [Zhu et al., SpikeGPT, TMLR, 2024](https://openreview.net/forum?id=gcf1anBL9e) | Spiking language modeling trained with surrogate gradients; context for the symbol-sequence family's baselines. |
| [Chollet, On the Measure of Intelligence, 2019 (preprint)](https://arxiv.org/abs/1911.01547) | Separates skill from skill-acquisition efficiency; why breadth of scope is not intelligence. |
| [Orchard et al., N-MNIST, Frontiers in Neuroscience, 2015](https://doi.org/10.3389/fnins.2015.00437); [Cramer et al., Heidelberg Spiking Data Sets, IEEE TNNLS, 2022](https://doi.org/10.1109/TNNLS.2020.3044364) | Dataset sources, recording conditions, and official splits. |

Expand the review further with continual learning in spiking networks, budgeted dynamic networks,
compressed event storage, and actual CPU/neuromorphic deployments. For each close paper, record tasks and
modalities, scenario, update rule, growth policy, replay budget, measured energy boundary, and code
availability. Do not compare published headline numbers across incompatible tasks or hardware.

## 9. The request to the professor

Ask for agreement on **one task suite, one practical resource budget, and one primary comparison**.
Request methodological supervision, a review of the novelty matrix, and access to suitable power
measurement and a modest local device if a deployment claim is kept. Physical-arm access is useful
only after the simulated control family passes its controls and calibration requirements are
satisfied.

Bring the README, the current-state assessment, the rehearsed component demos, and this plan. The
immediate next step is to freeze the first experiment with the professor, not to promise a
general-purpose brain before measuring one.

## 10. The project's plan of record (recorded 2026-09-22)

This study sits inside one ordered program, the
[plan of record](../specs/022-unified-brain-arc/PLAN-FINAL-2026-09-02.md): ratified 2026-09-02 and
extended by a combined amendment on 2026-09-22. This section records that plan as it stands at
commit `9976af7`, so the whole program and the paper's place in it can be discussed together. The
plan owns order, gates, and pre-registered failure conditions; [`tasks.md`](../specs/022-unified-brain-arc/tasks.md)
owns task status. Where either disagrees with this section, they win.

### How work lands

Every plan row follows the same rhythm, and the study's own experiment inherits it:

1. Pre-register the protocol and a reachable red, an input that must make the success criterion fail,
   before the first run.
2. Build, then measure in isolated processes, with a control that is not the method itself.
3. Review with a three-lens adversarial panel; refuters judge each finding and its proposed fix
   separately.
4. Send a closing report to an independent reviewer for refutation.
5. Run one full test battery on the stacked commits, record its result as the last commit, and push.

A null result is applied rather than defended, and a mechanism that measures nothing is deleted. Of
the plan's first two completed phases, more mechanism was removed than kept.

### The architecture today (measured at `9976af7`)

| Layer | What exists | Measured size or status |
|---|---|---|
| Anatomy | Modules grouped by brain region, joined by a typed, signed, delayed connectome | 411 source modules (excluding `__init__.py`), 283 registered across 18 regions; 43 connectome nodes, 91 edges, 14 named loops |
| Spiking substrate | LIF, ALIF, AdEx, Izhikevich, SRM, and dendritic neuron models; a partitioned kernel tick; an e-prop kernel | Spike trains byte-identical across thread counts at a fixed partition count, checked by the determinism gate |
| Framework contract | Brian2 2.10.1 as the equation and event reference; a Torch bridge executes it while the long-lived process stays free-threaded | Accepted for its amended contract on 27 parameter arms and five seeds (plan row B1 below) |
| Learning | Local rules only on the live path: e-prop encoders for event vision and audio, an online readout rule, local plasticity | Gradient-claim and weight-transport gates check part of the rule; backpropagation stays in training experiments |
| Memory | One append-only lake, `brain_lake.bin`, with tagged record classes and an immortal class that no eviction path may remove | 11 declared record tags; growth state survives a restart; the interaction replay ring does not yet (rows R2 and T093) |
| Drive and surface | An always-on daemon running the wake/sleep cycle, and one cockpit for chat, state, arm controls, and the E-STOP | The daemon relaunches on each new commit |
| Safety envelope | Five `brainstem/` files; every motor command passes one gate | Hash-checked on every battery; changed only with the owner's approval |
| Instruments | Policy gates, module self-checks, and unit tests, run as one battery per pushed change set | Last battery (2026-09-22): 361 of 361 self-checks, 1,450 tests passed, 88.48% coverage, all 16 substrate gates green. GitHub CI has not run since 2026-08-20, so that local battery is the only full gate |

### The planned data flow (combined amendment, 2026-09-22)

```text
observations / interactions / outcomes
    -> versioned records in brain_lake.bin
    -> verified recall + bounded sleep replay
    -> incumbent local synaptic learning -> held-out earned growth
    -> persisted learned state -> fresh-process continuation

Brian2 equation/event contract -> spiking substrate -> learned spiking readout
                                     ^       |
                                     |       v
                              bounded delayed NS feedback (N, experimental)

measured lake workload -> offline annealing -> validated derived cache/index plan
                                         -> guarded activation or keep incumbent
```

The lake is memory, not a learner: retrieval supplies evidence or a replayable example, and the
neural learner changes weights. A new unit must earn retention on untouched data, never on a
replayed training example. The fluid reservoir supplies candidate temporal features, not answers.
Annealing searches storage and access configurations, never truth, labels, or safety rules.

### Program status at `9976af7`

| Part | Goal | Status |
|---|---|---|
| Phase 0 | Owner unblocks | Done 2026-09-02 |
| Phase 1: the world | Show that the brain's stimulus carries learnable structure before wiring anything to it | Closed 2026-09-05 ([exit](../specs/022-unified-brain-arc/evidence/w-exit.md)): the world-red instrument landed; a goal curriculum was retired on its own measurement; context latents were refused on measurement |
| Phase 2: learning in the substrate | Synaptic learning first, growth only on plateau | Closed 2026-09-07 ([exit](../specs/022-unified-brain-arc/evidence/l-exit.md)): seven rows landed, two were deleted on their nulls, one was refused on measurement. The surviving learning mechanism is an online readout rule; nothing at any width reaches the persistence floor |
| Phase 3: a substrate that scales | Lake-resident sparse units, the H01 connectome prior, paging | Exit certified 2026-09-10 ([exit](../specs/022-unified-brain-arc/evidence/s2-exit.md)) with all three named items negative: no prior separation, paging deleted after missing its latency bound, region charge at 8 of 16. Unit blocks (T064) and the width derivation (T065) remain open |
| Phase 4: cognition on the substrate | A reasoning battery and consciousness-science instruments, each with a dissociation control | Open. Landed: the everyday battery (0 of 20), metacognition, agency, temporal-binding, report-order, and social-cognition instruments, and the conclusion protocol that fixes what each instrument may and may not claim. Remaining: C1a-C1c, C3, C2, the attention schema (T078) and the phase exit |
| Phase 5: self-improvement | An internal proposer, an evaluator cascade, and an archive of variants | Open and waiting on Phase 4 (T072-T076, T083); the self-build route fires and has kept zero modifications |
| Phase 6: polish | Cockpit sweep, human-scale derivation, consistency sweep, final run-through | Open (T084-T087) |
| Scope rows (2026-09-06) | X1 capability target, L5 immortal class, C5 conclusion protocol | C5 done; X1 (T088) and L5 (T089) open, with their gates already in the battery |
| B: Brian2 framework (T101) | Brian2 as the equation and event reference | Accepted 2026-09-22 for the amended contract; the original experiment missed its precision bar, and the precision guarantee covers the registered traces only. The migration has no before-deletion comparison against the retired snntorch layer (an open item) |
| R: durable recall becomes retained learning (T102-T104) | Bounded-cost recall, restart-safe typed replay, replay composed with earned growth | Planned, open |
| A: lake-access annealing (T105-T107) | Offline search over derived cache and index configurations, never canonical records | Planned, open |
| N: Navier-Stokes coupled to a liquid state machine (T096-T100) | An experimental fluid reservoir feeding spiking readouts | Planned, unbuilt |

### Execution order

1. B1 accepts the Brian2 contract (done); R1 audits and bounds historical recall and may run alongside it.
2. R2 adds restart-safe typed sleep replay; R3 tests its contribution to the incumbent learner and
   grower. No end-to-end continuity claim precedes both restart proofs.
3. N1-N5 keep their Phase 2 and Phase 4 prerequisites; B1 precedes N2, and an N4 recall arm waits
   for R1-R3.
4. A1 follows R1; A2 evaluates copies offline; A3 activates only a proven candidate after the
   affected R2/R3 continuation tests.

The historical critical path continues through Phase 4's reasoning chain (next: C1a), then Phase 5,
then polish. None of the new workstreams bypasses Phase 4's exit or the evaluator gates for
autonomous promotion.

### Where this study sits in the plan

| What the study needs | Plan rows it depends on | Status |
|---|---|---|
| A local spiking learner with a pinned neuron contract | B1 (T101) | Accepted for the amended contract |
| Earned growth (H1, H3) | Phase 2's growth gate; R3 (T104) composes replay with earned growth | The gate exists; R3 is open. Phase 2 found width nearly exhausted where the brain currently sits |
| Durable replay and the restart arm (H4) | R1-R2 (T102-T103); the continuity census (T093) | Open. Until they land, run the in-process replay arm and scope the restart claim out (section 4) |
| Human comparisons | X1 (T088), the capability registry | Gate exists; registry row open |
| The meaning of "never forgets" | L5 (T089), the immortal class | Gate exists; row open |
| The reasoning probe | C0, the everyday battery | Landed; 0 of 20 on five seeds |
| Task adapters, splits, floors, dense baselines, and the comparison harness | None yet | New work: it becomes its own pre-registered plan row, lands inside the governed tree like any other build, and keeps its backpropagation baselines unreachable from the live path |

The study is therefore a new row in this program rather than a separate project: it inherits the
rhythm above, adds the benchmark harness the plan does not yet have, and depends on the R workstream
for its restart and durable-replay claims.
