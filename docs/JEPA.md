# JEPA -- a latent-space world model for the live brain

A **Joint-Embedding Predictive Architecture** (JEPA -- LeCun, "A Path Towards Autonomous Machine
Intelligence" 2022; I-JEPA, Assran et al. 2023; V-JEPA 2024; VICReg, Bardes et al. 2022) added as a
learning method for the live brain (v3.11).

## The idea

JEPA predicts in a learned **representation space**, not the raw input/pixel space. Pixel-space
prediction wastes capacity modelling unpredictable detail; JEPA instead learns to predict an abstract
latent of the future. Four parts:

- **Encoder** `enc(x) -> z` -- maps a context to a latent.
- **Predictor** `pred(z, action) -> z'` -- action-conditioned latent dynamics.
- **Target encoder** -- an EMA / stop-gradient copy of the encoder; its latent of the *next*
  observation is the prediction target.
- **Anti-collapse** -- without it the encoder can map everything to a constant. Two guards, both
  present here: the predictor + stop-gradient **asymmetry** (as in BYOL/SimSiam) and **VICReg**
  (a per-dimension variance hinge + off-diagonal covariance decorrelation).

The training objective is the prediction error **in embedding space**:
`|| pred(enc(x_t), a_t) - stopgrad(target_enc(x_{t+1})) ||^2 + VICReg`.

## In this repo

`src/snn_data_lake/cerebellum/jepa_world_model.py` (CORE, zero-dep, `JepaWorldModel`) is a **minimal
deterministic linear JEPA**: a single linear encoder `z = unit(tanh(We@x + be))`, a residual linear
predictor `z' = clamp(z + Wz@z + Wa@onehot(a) + b)` (spectral-bounded so long rollouts stay finite), an
EMA target encoder, and a hand-rolled VICReg with closed-form gradients. `latent_dim = 16`.

**Honest framing.** This is a *minimal* JEPA -- a single linear layer, no latent-variable `z` for
multimodality, no transformer, no image-patch masking. It demonstrates the JEPA *mechanism* and is
interface-compatible with the planner; it is **not** a frontier ViT world model and makes no such
claim. The scale path is the torch scaffold below.

### Why it beats the sibling `cerebellum/forward_model.py`

`ForwardModel` predicts the next sensory state in the **raw input space** (a scalar reafference
comparator). `JepaWorldModel` predicts in a **learned latent** space. The self-check builds a 2-factor
world where two actions move the first hidden factor identically but the second oppositely: a 1-D
scalar model is *provably indifferent* between them, while the 16-D latent separates them -- so
`PrefrontalExecutive.plan()` driven by the JEPA reaches a goal the scalar model cannot.

### Wired into the live brain (live-reactive)

With `AutonomousArm(jepa=True)` / `LiveBrain(jepa=True)`:

- The model **self-trains online** from the live `(z_t, a_t, z_{t+1})` tick stream -- the live stream is
  the self-supervised corpus. This runs *after* the gated action, off the safety path.
- The prefrontal executive **plans over it** each tick: `z = enc(context)`, then commit the action
  whose bounded latent rollout best reaches a **reward-derived latent goal** (an EMA of latents at
  positive-reward ticks). The result is the same top-down bias (bounded by `pfc_gain`) into the same
  gated selection. `LiveBrain.plan_latent(context, goal_context)` exposes the same faculty
  deliberatively.

**Safety.** JEPA only biases the executive's goal; it drives no motion and adds no path to the
actuator. The reflex E-STOP runs first and still preempts every tick, and every motor command flows
through the gated `ArmSim`. None of the 5 brainstem safety-envelope files changed. All of this is
asserted with the flag ON in `autonomous_loop.py` and `live_brain.py` self-checks.

## The torch scale scaffold (research, unverified)

`research/jepa_scaffold_torch.py` is a torch I-JEPA-style scaffold (real EMA target + VICReg,
standard autograd) that **exports** a trained latent predictor into the core `snn_jepa/1` JSON format,
so the zero-dep core serves torch-trained latents with no torch (the same core+torch export bridge as
`trained_brain` / `snn_classifier`). It is **UNVERIFIED ON HARDWARE** (no GPU in CI), ruff-excluded,
not in CI, and behind the `[gpu]` / `[train]` extra; without torch installed it prints a guidance
message and exits cleanly.

## Run it

```bash
# core self-check (collapse control, identity + ForwardModel baselines, bounded rollout, latent>scalar, determinism)
PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.jepa_world_model

# JEPA wired live into the whole brain (safety + determinism with the flag ON)
PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.live_brain

# the torch scale scaffold (prints a guidance banner without torch)
python research/jepa_scaffold_torch.py
```
