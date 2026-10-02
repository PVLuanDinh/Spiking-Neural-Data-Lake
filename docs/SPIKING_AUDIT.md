# Spiking-purity audit -- did the project stray from spike signals + SNNs?

The Spiking Neural Data Lake's stated purpose is **spike signals and spiking neural networks**. This is
an honest assessment of how faithful the codebase is to that purpose, done 2026-07-10.

## Verdict

**The foundation is genuinely spiking; the recent higher-cognition / learning tier (v3.11-v3.20) drifted
to rate-level ANN / linear methods.** The spiking identity holds where the project touches neurons, spikes,
and plasticity; it strayed where it added learning/reasoning/value machinery on top.

## Genuinely spiking (the foundation -- the bulk of the core)

- **Neuron models** -- `cortex/` LIF, Izhikevich, Hodgkin-Huxley, ALIF, dendritic, adaptive: real
  membrane / threshold / reset / spike dynamics.
- **Spike encoders** -- rank-order (Thorpe latency), population, temporal, theta-gamma: turn input into
  spike trains / discrete spike times.
- **Spiking classifiers / learners** -- tempotron, `snn_classifier`, paradigm-B: spike-timing readouts.
- **Plasticity** -- the STDP family (`r_stdp` reward-modulated, `valence_stdp`, `inhibitory_stdp`,
  triplet), BCM, homeostatic: spike-timing learning.
- **Circuits** -- hippocampus (DG->CA3, replay, ring attractor), cerebellum (spiking PID, synfire,
  Kuramoto), brainstem CPG, amygdala fear conditioning.
- **The data lake itself** stores and streams **spike events**. `research/` adds polychronization, SRM,
  resonate-and-fire, e-prop, delay learning -- all spiking.

## The drift -- rate-level, not spiking (mostly the v3.11-v3.20 learning arc)

These use continuous activations + gradient / least-squares / EMA, **not spikes or spike-timing learning**.
They ride *on* the spiking lake's stored `[0,1]^64` pattern (a rate/analog vector) and do rate-level
function approximation on it:

| module (version) | what it actually is | spiking? |
|---|---|---|
| `faculties/jepa_world_model` (v3.11) | self-describedly a "rate-level, LINEAR" JEPA: `z=unit(tanh(We.x+be))` | no |
| `faculties/mlp_q` (v3.18) | one-hidden-layer MLP, hand-**backprop** | no |
| `faculties/latent_q` (v3.17) | linear `W[a].x`, Widrow-Hoff | no |
| reward-shaping (v3.19-v3.20) | linear reward head + gradient | no |
| `faculties/intrinsic_motivation` (v3.15) | EMA of prediction error | no |
| `neuromodulation/curriculum` (v3.16) | a counter / threshold | not even neural |
| `faculties/hjepa_serve` + torch H-JEPA (v3.13-14) | dense MLPs | no |
| `sensorimotor/nl_interface` + `LiveBrain.converse` | keyword-set routing | not neural |
| language backend | a dense transformer -- code admits "NOT a spiking LLM" | no |

Several of these docstrings admit the rate-level nature honestly, but it *is* a drift from
"depends entirely on spike signals + SNNs."

## Correcting course (v3.21)

Rather than deepen the drift, new learning added from here should be **spiking**. v3.21 does exactly that:
`readout/interaction_learning.py` (`SpikingInteractionLearner`) improves the live brain **by being talked
to** through a genuinely spiking path -- rank-order spike encoding + reward-modulated spike-timing
plasticity (R-STDP), reusing the real spiking primitives, with a self-check that structurally proves it is
spike-timing (not a gradient) and a shuffled-teaching control that stays at chance. See `LiveBrain(
interaction_learning=True).teach()/route()`.

## Open work to further reduce the drift (honest)

The value / goal / planning substrate is still rate-level. Spiking replacements exist in-repo to build on:
a spiking action-value via `r_stdp` (instead of `mlp_q`/`latent_q`), a spike-timing world model, and a
spike-coded curiosity signal. These are follow-on primitives, not yet built.
