# BrainCog vs. the Spiking Neural Data Lake brain

A capability comparison between BrainCog (the Institute of Automation, CAS
brain-inspired SNN engine; Zeng et al.) and this project's zero-dependency
neuromorphic brain, plus a ranked, honest, buildable improvement list.

Sources (fetched 2026-07-14): BrainCog GitHub `github.com/BrainCog-X/Brain-Cog`,
homepage `brain-cog.network`, and the BrainCog paper (Zeng et al., *Patterns*
2023, PMC10435966). Claims below are grounded in those pages, not memory.

---

## TL;DR

BrainCog is a large **PyTorch** research framework: ~50 functional SNN modules,
models of ~28 brain areas, deep SNNs trained on real vision/event datasets
(MNIST/CIFAR10/ImageNet/DVS-CIFAR10/DVS-Gesture), ANN-SNN conversion, a
biologically-plausible feedforward+feedback training rule, FPGA co-design, and a
roadmap AGI engine (BORN). Its value is overwhelmingly in **torch-trained SNNs
and datasets we deliberately cannot replicate zero-dep**, and we should not
pretend otherwise.

Our brain is a **zero-dependency, pure-stdlib** neuromorphic organism: 126
modules across 17 anatomical regions, a unified always-on circadian Brain, and a
built-out **self-improvement ladder (L0-L13)** with a hard safety envelope.
BrainCog has richer *trained* perception and more explicitly-modeled brain
regions; we are clearly ahead on **integrated self-improvement, the unified
circadian organism, determinism, and safety**.

The genuinely adoptable gaps are small and specific (social cognition /
theory-of-mind, a feedback-based credit-assignment rule, deeper multisensory
integration) plus one honest torch-tier item (surrogate-gradient deep-SNN on a
DVS dataset). Most of BrainCog's bulk is either torch-tier-only or redundant
with what we already have.

---

## 1. Capabilities matrix

Legend: **Have** = shipped in the zero-dep core (or torch tier where noted);
**Partial** = a related/limited form exists; **Absent** = not present.

| Mechanism | BrainCog | Our brain |
| --- | --- | --- |
| **Neuron models** IF / LIF | yes | Have (`lif_neuron`, `alif_neuron`, `adaptive_spiking`) |
| Izhikevich | yes | Have (`izhikevich_neuron`) |
| Hodgkin-Huxley | yes | Have (`hodgkin_huxley`) |
| AdEx / resonate-and-fire / SRM / dendritic / autapse | partial (H-H, Izh focus) | Have (AdEx+R&F+SRM in `research/`; `dendritic_neuron`, `autapse_neuron` in core) |
| **Plasticity** Hebbian / STDP | yes | Have (`valence_stdp`, `r_stdp`, `inhibitory_stdp`, `competitive_learning`) |
| Short-term plasticity (STP) | yes | Have (`short_term_plasticity`) |
| BCM / homeostatic / intrinsic / metaplasticity | partial | Have (`bcm_plasticity`, `homeostatic_scaling`, `intrinsic_plasticity`, `meta_plasticity`) |
| Three-factor / eligibility / delay learning | partial | Have (`r_stdp` distal reward; `delay_learning`, `eprop` in research) |
| Complex-synapse consolidation (Benna-Fusi) | absent | Have (`benna_fusi`) |
| **Deep-SNN training** surrogate-gradient BPTT | yes (core strength) | Absent (no surrogate-grad trainer) |
| ANN-SNN conversion (near-lossless, fast) | yes (core strength) | Absent |
| Feedforward+feedback global-error + local rule (BackEISNN / spatio-temporal adjustment) | yes | Partial (local rules + `eprop` research; no wired global-feedback trainer) |
| Unsupervised STDP classifier (Diehl-Cook) | yes | Partial (torch tier `training/mnist.py`; best 90% at 1600 neurons) |
| **Encoding** rate / temporal / population | yes | Have (`rank_order_coding`, `population_encoding`, `population_vector`, latency, Van Rossum) |
| **Connectivity** lateral inhibition / feedforward / feedback / local | yes | Have (WTA, divisive-norm, TRN gate; `connectome.py` signed loops) |
| **Brain-region SNN models** PFC / hippocampus / basal ganglia / thalamus | yes (28 areas) | Have (`prefrontal_executive`, trisynaptic hippocampus, CBGTC, thalamic workspace+TRN) |
| Visual pathway (retina->V1->...) | yes (trained) | Absent (no deep visual hierarchy) |
| Motor cortex | yes | Partial (arm loop + cerebellum, not a cortical motor map) |
| Superior colliculus / orienting / habenula / claustrum / olfactory | partial | Have (all present as closed loops) |
| **Cognition** decision-making (attractor / RL) | yes (Drosophila, deep-RL SNN) | Have (Wang-2002 evidence accumulation, spiking PFC WTA) |
| Working + long-term memory | yes | Have (PFC WM attractor; hippocampus->cortex consolidation) |
| Knowledge representation & reasoning | yes (SNN KG, music) | Partial (`vsa_reasoning`, transitive inference, `predictive_coding`, RBM) |
| **Social cognition / theory-of-mind / moral** | yes | **Absent** |
| Attention & consciousness (access) | partial (BORN roadmap) | Have (`access_consciousness` GNW ignition, `global_workspace`) |
| Emotion / affect | yes (BORN) | Have (amygdala threat, cortisol, dopamine/5-HT neuromodulators) |
| **Multi-sensory integration** | yes | Partial (`cue_integration`, `multimodal_encoder`, `population_vector`) |
| **Developmental plasticity** (pruning/neurogenesis) | yes | Have (`developmental_pruning`, `neurogenesis`, `lifelong_consolidation`) |
| **Evolutionary plasticity** (neuron/architecture evolution across species) | yes | Partial (`quality_diversity` MAP-Elites over policies; no cross-species neuron evolution) |
| **Self-improvement loop** (curiosity->goals->metacog->criticality->QD, integrated) | Absent (BORN is a roadmap) | **Have (L0-L13 ladder, `wholistic_loop`)** |
| **Unified always-on organism** (circadian wake/sleep, self-model) | Absent (toolkit of modules) | **Have (`cortex/brain.py`)** |
| Sleep-gated neurogenesis + sleep-replay consolidation | absent | **Have** |
| Metacognitive self-monitoring / confidence-gated abstain | absent | **Have (`metacognitive_monitor`)** |
| Closed criticality / homeostasis regulator | absent | **Have (`criticality_regulator`)** |
| **Safety envelope** (reflex E-STOP, gated actuator, firewall, fail-closed) | Absent | **Have** |
| **Zero-dependency determinism** (stdlib, crc32-reproducible, self-checks) | No (heavy torch) | **Have** |
| **Neuromorphic hardware co-design** (FPGA / FireFly) | yes | Absent (out of scope) |
| Real datasets (ImageNet, DVS-Gesture, DVS-CIFAR10) | yes | Absent (a few small torch experiments only) |

---

## 2. What BrainCog has that we don't (or do worse) -- ranked by value

Ranked by value *to this project specifically*, given our zero-dep charter.

1. **Trained deep SNNs on real vision/event datasets** (surrogate-gradient BPTT +
   ANN-SNN conversion on MNIST/CIFAR10/ImageNet/DVS-CIFAR10/DVS-Gesture).
   BrainCog's headline result -- ~3% of ANN energy, near-lossless conversion at
   1/10-1/50 sim time. This is the single biggest capability gap and the single
   least replicable one for us.
2. **Social cognition / theory-of-mind circuit.** BrainCog models a whole
   cognitive category we have *zero* coverage of: false-belief reasoning,
   affective/cognitive empathy, brain-inspired moral decision-making. We have a
   `self_model` but no *other*-model.
3. **A biologically-plausible deep-training rule that isn't pure BPTT**
   (BackEISNN: feedforward + feedback connections carrying a global error signal
   fused with local plasticity; the "spatio-temporal adjustment" algorithm).
   This is a real learning-rule gap -- we have local rules and research e-prop
   but nothing that does global credit assignment through feedback connections.
4. **Explicitly-modeled sensory pathways** (a trained visual hierarchy; a motor
   cortex map). We have subcortical orienting (superior colliculus) but no deep
   retina->cortex visual pathway and no cortical motor map.
5. **Deeper multi-sensory integration** as a first-class binding mechanism.
   Ours is partial (`cue_integration`, `multimodal_encoder`) -- functional but
   not a principled multisensory-enhancement / inverse-effectiveness model.
6. **Evolutionary plasticity as an explicit axis** (BrainCog's finding that
   swapping rodent neurons for human neurons in a fixed PFC circuit improves a
   memory task -- i.e. evolution acts on the *unit*, not just the wiring; plus
   evolutionary search over SNN architectures).
7. **Knowledge-graph / commonsense reasoning at scale** and creative generation
   (music composition in the BORN engine). Ours (`vsa_reasoning`, transitive
   inference) is toy-scale by comparison.
8. **Neuromorphic hardware co-design** (FPGA / FireFly accelerator). Genuine
   BrainCog capability; entirely outside our software-only scope.

### Buildability verdict for each gap

Verdict key: **(a)** buildable pure-stdlib now; **(b)** buildable only in the
optional torch tier; **(c)** not worth it / redundant / out of scope.

1. **Trained deep SNNs + datasets -> (b), and mostly (c) beyond a benchmark.**
   Surrogate-gradient BPTT and ImageNet-scale training are torch-and-dataset
   bound by definition. A single honest torch-tier benchmark (e.g. DVS-Gesture,
   which aligns with the existing NG-SNN reproduction goal) is worth doing; a
   full ANN-SNN conversion + ImageNet pipeline is **not** -- it would bloat the
   torch tier for a capability that isn't the product's purpose (an arm brain).
   Do **not** attempt any of this zero-dep.
2. **Social cognition / theory-of-mind -> (a) buildable pure-stdlib now.** This
   is the best-value pure-stdlib build available. We already ship `self_model`;
   a second "other-agent" model instance whose belief state can diverge from the
   world (a Sally-Anne false-belief primitive) is small, deterministic, novel
   for us, and closes an entire missing category. See top-3 below.
3. **Feedback-based global+local credit assignment -> (a) as a PoC, (b) at
   scale.** Random feedback alignment (Lillicrap) and a feedforward+feedback
   local rule are just matmuls -- a toy stdlib demo on a small task is feasible
   and honest as a *mechanism* PoC. Deep-network competitiveness needs torch.
   Ship the stdlib PoC; label it a mechanism, not a trained result.
4. **Sensory pathways -> (b)/(c).** A trained visual hierarchy is torch-tier and
   redundant with our stated scope (we are not a vision benchmark). A *small*
   stdlib retinotopic edge->orientation front-end feeding the existing colliculus
   is (a) but low-value. Skip unless a concrete task needs vision.
5. **Multisensory integration -> (a) buildable pure-stdlib now.** A principled
   superior-colliculus multisensory-enhancement primitive (super-additive when
   cues are weak/coincident; inverse effectiveness) deepens an existing partial
   capability with stdlib LIF. Good value. See top-3.
6. **Evolutionary plasticity -> (a) small PoC, (c) at scale.** A cross-species
   "neuron swap" experiment (replace LIF with a richer neuron in a fixed circuit,
   measure task gain) is a small, novel, pure-stdlib mirror of BrainCog's human-
   vs-rodent-neuron finding. Full evolutionary NAS over SNNs is heavy and largely
   redundant with our `quality_diversity` MAP-Elites -- skip that part.
7. **KG/commonsense reasoning + creative generation -> (b)/(c).** Scale needs
   embeddings + data. Our VSA/transitive-inference primitives already cover the
   mechanism honestly; scaling is not the product's purpose. Redundant.
8. **FPGA hardware co-design -> (c).** Out of scope; we are software-only,
   deliberately.

---

## 3. Where our brain is actually AHEAD

These are real advantages, not consolation prizes:

- **Integrated self-improvement ladder (L0-L13).** BrainCog has powerful trained
  modules but *no integrated developmental self-improvement loop*. BORN (their
  AGI engine) is an explicit **roadmap**, not a shipped system. We have a built,
  self-checked ladder: curiosity/learning-progress, self-paced curriculum,
  meta-plasticity, closed criticality regulation, sleep consolidation with
  non-regression, quality-diversity repertoire, self-generated drive-grounded
  goals, metacognitive HOLD, and a `wholistic_loop` that measures whole > sum.
- **A unified, always-on organism.** `cortex/brain.py` runs one brain on a
  circadian clock: wake ticks (curiosity-driven, safe), sleep ticks (neurogenesis
  gated to sleep + noise-driven replay consolidation), a self-model narrator that
  speaks freshly from real signals, and GNW ignition. BrainCog is a *toolkit* you
  assemble per experiment; it has no single integrated circadian creature.
- **Zero-dependency determinism.** Pure stdlib, `zlib.crc32`-hashed reproducible
  demos, a `run_selfchecks.py` runner (~91 modules). BrainCog needs a full torch
  stack and its results are stochastic GPU training runs. Ours is auditable and
  bit-reproducible.
- **A hard safety envelope.** Reflex E-STOP (NaN fails closed), a gated
  `ArmSim.apply`, a command firewall, and fail-closed comparisons wrap every
  actuator path. BrainCog is a research framework with no safety layer -- because
  it isn't driving a physical arm.
- **Homeostasis / criticality as a closed control loop**, and **honest-negative
  methodology** (refutation controls, de-scoped overclaims) baked into the build
  process. BrainCog reports wins; we report the wins *and* the ceilings.

Framed plainly: BrainCog is a bigger, better-*trained* pile of brain modules; our
brain is a smaller, *integrated, self-improving, safe, deterministic* one.

---

## 4. Top improvements actually worth adopting (ranked)

Each is honest about tier and scope. The first three are pure-stdlib core wins;
the fourth is the one place BrainCog is genuinely ahead and we adopt it honestly
in the optional torch tier.

1. **Theory-of-Mind / social-cognition primitive (pure-stdlib core).**
   *Sketch:* instantiate a second spiking `self_model` as an "other-agent" model
   whose belief population is updated only from what the other observed, so it can
   diverge from world state. Score a Sally-Anne false-belief task: predict the
   other's action from its (possibly stale) belief, not from ground truth; the
   control (predict from ground truth) must fail. Closes a whole missing cognitive
   category with a small, deterministic, novel-for-us module. Highest value.

2. **Feedback-connection credit-assignment learning rule (pure-stdlib PoC).**
   *Sketch:* a feedforward+feedback two-layer spiking net trained with random
   feedback alignment (fixed random feedback matrix carries a global error) fused
   with a local STDP/BCM update -- BrainCog's BackEISNN idea in miniature. Prove
   on a toy task that feedback-global + local beats local-only and approaches the
   BPTT reference; ship it labelled a *mechanism* PoC, not a trained benchmark.
   Closes a real learning-rule gap without adding a dependency.

3. **Multisensory-integration primitive (pure-stdlib core).**
   *Sketch:* a superior-colliculus multisensory-enhancement cell over the existing
   `cue_integration` / `multimodal_encoder` -- super-additive spiking response
   when two weak, coincident cross-modal cues arrive (inverse effectiveness),
   sub-additive when either is strong. Deterministic, testable, deepens a partial
   capability into a principled one.

4. **Surrogate-gradient deep-SNN on a DVS event dataset (torch `[train]` tier).**
   *Sketch:* add a surrogate-gradient BPTT trainer and run DVS-Gesture (event
   data, small, aligns with the existing NG-SNN reproduction goal) in the optional
   torch tier -- never in core. Gives us one real trained-SNN benchmark to anchor
   energy/accuracy claims against BrainCog's, without pretending zero-dep. Adopt
   *only* this slice; skip ANN-SNN conversion + ImageNet as bloat.

5. **(Optional, small) Cross-species "neuron-swap" evolutionary-plasticity demo
   (pure-stdlib).** *Sketch:* hold a circuit fixed, swap LIF for a richer neuron
   (Izhikevich / dendritic / AdEx we already have), measure the task-accuracy
   delta -- a tiny, honest mirror of BrainCog's human-vs-rodent-neuron PFC result,
   showing "evolution acts on the unit, not just the wiring." Low effort, novel
   framing, reuses existing neuron models.

---

## 5. Honest bottom line

BrainCog and this project optimize different things. BrainCog maximizes *trained
breadth* -- many brain areas, real datasets, deep-SNN training, hardware -- at the
cost of a heavy torch dependency and no integration/safety story. We maximize
*integrated depth* -- one deterministic, self-improving, circadian, safety-gated
organism -- at the cost of trained perception we deliberately don't chase.

We should **not** try to become BrainCog. Copying its torch training stack and
datasets would dilute the zero-dep charter for capabilities that aren't the
product's purpose. The right moves are the three small pure-stdlib primitives
above (theory-of-mind, feedback credit-assignment, multisensory integration),
which genuinely extend our brain in its own idiom, plus one honest torch-tier
benchmark (DVS-Gesture) to keep our claims measurable against theirs.
