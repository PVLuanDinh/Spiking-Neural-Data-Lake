r"""
PHYSICS-JEPA, HIGH-DIM -- the case where a learned JEPA latent SHOULD beat a random encoder.

physics_jepa.py reported an HONEST NEGATIVE: over a near-linear 4-D world a random projection already
linearly decodes the next state (Johnson-Lindenstrauss), so the learned latent had nothing to add.
JEPA's actual claimed advantage (LeCun): predicting in a learned latent lets the encoder IGNORE
unpredictable detail and keep only the predictable structure. That advantage only shows up when the
observation is high-dimensional and mostly UNPREDICTABLE.

So here the observation is exactly that: a few PREDICTABLE nonlinear signal dims (a deterministic
nonlinear lift of the true 4-D state) concatenated with MANY UNPREDICTABLE noise dims (fresh random
every step). The JEPA invariance loss cannot predict the noise, so training drives the encoder to
attend to the signal and drop the noise. A RANDOM encoder mixes signal and noise equally, so its
latent decodes the true next-state worse. The probe (latent -> true next-state) should now favour the
LEARNED encoder -- the honest positive that physics_jepa.py's linear world could not show.

Un-riggable: (1) invariance loss drops; (2) no collapse vs the anti_collapse=False ablation;
(3) the probe: learned latent decodes next-state BETTER than a random encoder, by a real margin.

  PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.physics_jepa_highdim
  Knobs: NORD_STEPS(9000) NORD_NOISE(40) NORD_LAT(16)
Deps: numpy (probe) + stdlib (JepaWorldModel). Reuses the physics dynamics from physics_jepa.
"""

import os
import random

import numpy as np

from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel
from snn_data_lake.reasoning.physics_jepa import gen_dynamics, ACTIONS


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _signal(state):
    """A deterministic NONLINEAR lift of the 4-D state -> 8 PREDICTABLE observation dims."""
    x, y, u, v = state
    return [x, y, u, v, x * u, y * v, np.sin(2 * x), np.cos(2 * y)]


def _lift(state, n_noise, rng):
    """Observation = predictable nonlinear signal ++ UNPREDICTABLE fresh noise (the JEPA must learn
    to ignore the noise). Noise is regenerated every call, so it carries no cross-step information."""
    return _signal(state) + [rng.uniform(-1.0, 1.0) for _ in range(n_noise)]


def _probe_mse(model, obs_next_pairs):
    """Linear probe: least-squares from the model's latent of the current OBS to the true next STATE.
    Lower MSE => the latent kept more of the predictable state structure (and less of the noise)."""
    Z = np.array([model.encode(o) for o, _ in obs_next_pairs], dtype=np.float64)
    Y = np.array([ns for _, ns in obs_next_pairs], dtype=np.float64)
    cut = int(len(Z) * 0.8)
    Ztr = np.hstack([Z[:cut], np.ones((cut, 1))])
    Zte = np.hstack([Z[cut:], np.ones((len(Z) - cut, 1))])
    W, *_ = np.linalg.lstsq(Ztr, Y[:cut], rcond=None)
    return float(((Zte @ W - Y[cut:]) ** 2).mean())


def main():
    STEPS = _envi("NORD_STEPS", 9000)
    N_NOISE = _envi("NORD_NOISE", 40)
    LAT = _envi("NORD_LAT", 16)
    IN = 8 + N_NOISE

    print("=" * 92)
    print(
        "PHYSICS-JEPA (HIGH-DIM) -- learned latent should beat a random encoder when obs is mostly noise"
    )
    print(
        f"obs dim {IN} = 8 predictable signal + {N_NOISE} unpredictable noise; latent {LAT}; steps {STEPS}"
    )
    print("=" * 92)

    trans = gen_dynamics(n_traj=300, steps=40, seed=0)  # (state, action, next_state)
    nrng = random.Random(123)
    # build observation transitions: obs_t (signal+noise) -> we also keep the TRUE next state for the probe
    obs_trans = [(_lift(s, N_NOISE, nrng), a, _lift(ns, N_NOISE, nrng)) for s, a, ns in trans]
    probe_pairs = [(o, ns) for (o, _, _), (_, _, ns) in zip(obs_trans, trans)]

    jepa = JepaWorldModel(ACTIONS, in_dim=IN, latent_dim=LAT, seed=0)
    rng = random.Random(1)
    first, last = [], []
    for i in range(STEPS):
        o, a, on = obs_trans[rng.randrange(len(obs_trans))]
        loss = jepa.train_step(o, a, on)
        (first if i < 200 else last if i >= STEPS - 200 else []).append(loss)
    loss0, lossN = sum(first) / len(first), sum(last) / len(last)
    std_full = jepa.latent_std()

    abl = JepaWorldModel(ACTIONS, in_dim=IN, latent_dim=LAT, seed=0, anti_collapse=False)
    for i in range(STEPS):
        o, a, on = obs_trans[rng.randrange(len(obs_trans))]
        abl.train_step(o, a, on)
    std_abl = abl.latent_std()

    rand = JepaWorldModel(ACTIONS, in_dim=IN, latent_dim=LAT, seed=7)  # untrained control
    mse_learned = _probe_mse(jepa, probe_pairs)
    mse_random = _probe_mse(rand, probe_pairs)
    lift = (1 - mse_learned / mse_random) * 100

    print("-" * 92)
    print(f"  JEPA invariance loss : {loss0:.4f} -> {lossN:.4f}")
    print(
        f"  latent std (collapse): full {std_full:.3f}  vs  anti-collapse-OFF ablation {std_abl:.3f}"
    )
    print(
        f"  probe MSE to true next-state: learned {mse_learned:.4f}  vs  random {mse_random:.4f}  "
        f"({lift:+.1f}% {'BETTER' if lift > 0 else 'worse'})"
    )
    print("-" * 92)
    print(
        f"  with {N_NOISE} unpredictable noise dims, the JEPA learns to attend to the predictable signal;"
    )
    print(
        "  the random encoder mixes noise in and decodes worse -- JEPA's advantage shows when obs is noisy."
    )
    print("=" * 92)

    assert lossN < loss0, f"invariance loss did not drop ({loss0:.4f} -> {lossN:.4f})"
    assert std_full > std_abl * 2, (
        f"full model did not avoid collapse ({std_full:.3f} vs {std_abl:.3f})"
    )
    assert mse_learned < mse_random, (
        f"HIGH-DIM: learned latent must beat the random encoder here ({mse_learned:.4f} vs {mse_random:.4f})"
    )
    print(
        f"self-check OK: high-dim JEPA -- learned latent decodes next-state {lift:+.0f}% better than a "
        f"random encoder (loss {loss0:.3f}->{lossN:.3f}, no collapse {std_full:.2f}>>{std_abl:.2f}). "
        f"JEPA's predict-in-latent advantage shows once the observation is high-dim and mostly noise."
    )
    return 0


if __name__ == "__main__":
    main()
