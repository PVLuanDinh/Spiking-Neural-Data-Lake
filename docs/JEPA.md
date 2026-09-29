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
through the gated `So101Arm`. None of the 5 brainstem safety-envelope files changed. All of this is
asserted with the flag ON in `autonomous_loop.py` and `live_brain.py` self-checks.

## Scaling: a deeper encoder + the torch trainer (v3.12)

A single linear layer is a toy. To scale, the core gains an optional **hidden-layer encoder**
(`z = unit(tanh(We @ tanh(W1@x + b1) + be))`), persisted as **`snn_jepa/2`** (the encoder as an ordered
list of `[W, b]` layers). A freshly-built core model stays linear; a **deep** encoder arrives via
`load_weights(snn_jepa/2)` and is **served + frozen** -- only the predictor keeps adapting online. The
core self-check proves the hidden layer is load-bearing: a 2-layer encoder makes XOR separable (4/4)
while the single-layer control (same output layer, no hidden pre-layer) cannot (1/4).

`research/jepa_scaffold_torch.py` is the **scale tier** -- a real PyTorch trainer for the RTX 5070: a
deeper MLP-encoder latent JEPA (encoder + residual predictor + EMA target + VICReg), device selection
(`NORD_GPU=1` -> cuda), AMP, checkpointing, and `export_jepa` -> `snn_jepa/2`, plus `verify_transfer`
which loads the export into the zero-dep core and confirms the served `encode`/`predict` match the torch
forward. Verified on the actual RTX 5070 GPU (v3.13.2; torch 2.11.0+cu128): the deep encoder trains on the
GPU and the **export -> core bridge is bit-identical** (max |diff| 3.33e-16 over 5 probes). It is
ruff-excluded, not in CI, behind the `[gpu]` / `[train]` extra; without torch it prints a guidance message
and exits cleanly. Train + serve:

```bash
pip install ".[gpu]"
NORD_GPU=1 python -m snn_data_lake.reasoning.jepa_scaffold_torch   # trains on the 5070, exports snn_jepa/2, verify_transfer
# then in the zero-dep core: JepaWorldModel(...).load_weights("<export>.json") serves the deep encoder, no torch
```

> Path corrected 2026-07-30: `research/` became `src/snn_data_lake/reasoning/` in v3.78 (`627b2e2b`),
> so the runnable form is `python -m snn_data_lake.reasoning.<module>`. The files are the same files;
> only the import path moved. `tools/citation_guard.py` found the three stale commands on this page.

A latent-variable `z` for multimodality is a remaining depth axis.

## H-JEPA (hierarchical JEPA) -- an honest negative + the torch path (v3.13)

H-JEPA stacks JEPA levels at increasing abstraction and coarser timescale (LeCun's hierarchical
world model): Level 0 models fast per-tick detail; Level 1 abstracts a window of Level-0 latents and
predicts the slow/abstract structure over a long horizon.

**A minimal LINEAR H-JEPA does not beat flat baselines** (measured, not assumed). Two failure modes:
1. **Rollout saturation** -- both levels used spectrally-contractive linear predictors, so multi-step
   rollout collapses to a fixed point and neither extrapolates; L0 fast-roll actually beat L1
   coarse-roll (MSE 0.35 vs 0.48).
2. **Trivial abstraction** -- the slow regime was already a linear band in the observation, so raw
   windowed *averaging* (no learning) decoded it (MSE 0.0005) better than the learned L1 abstraction
   (0.0163). The learned hierarchy added nothing.

The fix is nonlinearity on both axes: `research/hjepa_torch.py` (research, GPU, **UNVERIFIED ON
HARDWARE**) is a 2-level torch H-JEPA with **nonlinear MLP encoders** at both levels, a **non-contractive
MLP coarse predictor** (bounded by normalization, not clamped to a contraction), and a **nonlinear
abstraction**: the slow regime is `XOR(b0, b1)` of two factors rendered in disjoint bands, so a flat
linear readout and raw windowed averaging are *provably floored* at MSE 0.25 while a nonlinear
hierarchical encoder has the capacity to fold XOR into a linearly-decodable latent. It measures the two
claims the linear version failed (`[GPU-VERIFY]` lines).

**Verified on the actual RTX 5070 GPU (v3.13.2; torch 2.11.0+cu128, sm_120, AMP): both claims HOLD.**
(a) the learned nonlinear L1 abstraction decodes the XOR regime at MSE 0.0000 while a flat linear readout
(0.260) and raw windowed averaging (0.250) sit at the provable 0.25 XOR floor; (b) L1 coarse-roll (0.054)
beats L0 fine-roll (0.706). The nonlinear hierarchy genuinely beats flat where the linear version could
not. (Also confirmed on CPU torch; the results are device-agnostic.)

```bash
pip install ".[gpu]"   # a cu128 / Blackwell-sm_120 build for the 5070
NORD_GPU=1 python -m snn_data_lake.reasoning.hjepa_torch   # trains the nonlinear H-JEPA, prints [GPU-VERIFY] hierarchy-vs-flat
```

**AirLLM does not apply here.** AirLLM is disk-streaming *inference* for a transformer too big for
VRAM; it has no training path and would only slow a compute-bound H-JEPA that fits the 5070 many times
over. The real levers are CUDA + AMP, and (for a model that ever exceeds VRAM) torch
gradient-checkpointing / offload -- not AirLLM. See `docs/AIRLLM_ANALYSIS.md`.

## Run it

```bash
# core self-check (collapse control, identity + ForwardModel baselines, bounded rollout, latent>scalar, determinism)
PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.jepa_world_model

# JEPA wired live into the whole brain (safety + determinism with the flag ON)
PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.engine.live_brain

# the torch scale scaffold (prints a guidance banner without torch)
PYTHONPATH=src python -m snn_data_lake.reasoning.jepa_scaffold_torch
```

## H-JEPA in the live brain (v3.14)

The nonlinear H-JEPA can now drive the live executive's reasoning. Two pieces:

- **`cerebellum/hjepa_serve.py` (`HJepaServe`)** -- a zero-dep pure-Python SERVER for a torch-trained
  2-level H-JEPA exported as `snn_hjepa/1`. It replicates the torch forward exactly: `encode0` (deep
  ReLU-MLP), `encode1` (mean-pool a K-window of L0 latents -> abstract latent), and residual/pure
  predictors, with `predict(state, command)` = the abstract `predict1` so it drops into
  `PrefrontalExecutive.plan()`. Forward-serve only (torch trains the depth; the core serves it). The
  core serve == the torch forward, **max |diff| 2.50e-16** (verified on the RTX 5070).
- **Live wiring (`AutonomousArm(hjepa_serve=...)`, opt-in)** -- each tick accumulates `encode0(pattern)`
  into a rolling K-window; once full, the executive PLANS over the abstract `encode1(window)` latent
  (rolling the L1 predictor) toward a reward-EMA abstract goal. The abstract latent summarises the recent
  K-tick *situation*, so the goal tracks the slow regime rather than the instantaneous frame. Still a
  gated top-down bias -- reflex E-STOP first, no actuator route, the 5 envelope files untouched.

`research/hjepa_arm_torch.py` trains an arm-native H-JEPA (arm `[0,1]^64` observations + the 3 arm
actions + a slow two-timescale regime) on the GPU and exports `snn_hjepa/1`. **GPU-verified: the abstract
z1 decodes the slow regime far better than a flat single frame (acc 1.000 vs 0.820)** -- the temporal
hierarchy genuinely helps vs instantaneous perception. **Honest limit:** z1 does *not* beat raw windowed
averaging on this arm task (both ~1.0), because the arm regime is band-linear and trivial pooling already
recovers it -- unlike the XOR world, there is no non-linear floor for the learned abstraction to exploit.
So on the current arm task the hierarchy's value is "pool beats one frame," not "learned beats trivial."
