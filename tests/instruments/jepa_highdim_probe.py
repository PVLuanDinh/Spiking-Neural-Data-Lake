r"""The physics-JEPA high-dim probe (2026-08-11, #10 lane 3): the MERGE half of the trio's
settlement. physics_jepa_highdim.py was a BENCHMARK over the governed
faculties/jepa_world_model.JepaWorldModel -- it held no mechanism of its own -- so its merge
target is the instrument category (the gwt_deep_split_ab / rankingdomain_nightkb_ab pattern),
and the retiring physics_jepa's dynamics (gen_dynamics/ACTIONS/_THRUST, ~40 self-contained
lines) are ABSORBED verbatim so the licence survives both deletions.

What it measures (JEPA's actual claimed advantage, LeCun): with an observation that is a few
predictable nonlinear signal dims plus MANY unpredictable noise dims, the invariance loss
drives the learned encoder to attend to the signal -- so a linear probe from the learned
latent to the true next state beats the same probe from an untrained (random) encoder. The
trio's recorded verdicts: physics_jepa = honest negative on a near-linear world (a random
projection already decodes it, Johnson-Lindenstrauss); THIS = the honest positive;
double_pendulum_jepa = +60% on chaotic dynamics (CHANGELOG). Three asserts, unchanged from the
retired module: invariance loss drops; no collapse vs the anti_collapse=False ablation;
learned probe beats random.

  PYTHONPATH=src python tests/instruments/jepa_highdim_probe.py
  Knobs: NORD_STEPS(9000) NORD_NOISE(40) NORD_LAT(16)
"""

import os
import random
import sys

import numpy as np

from snn_data_lake.faculties.jepa_world_model import JepaWorldModel

ACTIONS = ("coast", "up", "down", "left", "right")
_THRUST = {
    "coast": (0.0, 0.0),
    "up": (0.0, 3.0),
    "down": (0.0, -3.0),
    "left": (-3.0, 0.0),
    "right": (3.0, 0.0),
}


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def gen_dynamics(n_traj=300, steps=40, seed=0):
    """Real controlled 2-D point-mass dynamics under gravity + discrete thrust (absorbed
    verbatim from the retired physics_jepa). Returns (state, action, next_state) transitions."""
    rng = random.Random(seed)
    g, dt = 9.8, 0.05
    tr = []
    for _ in range(n_traj):
        x, y = rng.uniform(-4, 4), rng.uniform(2, 10)
        vx, vy = rng.uniform(-2, 2), rng.uniform(-2, 2)
        for _ in range(steps):
            a = rng.choice(ACTIONS)
            ax, ay = _THRUST[a]
            nvx = vx + ax * dt
            nvy = vy + (ay - g) * dt
            nx = x + nvx * dt
            ny = y + nvy * dt
            if ny <= 0.0:
                ny = 0.0
                nvy = -nvy * 0.7
            st = [x / 10.0, y / 12.0, vx / 6.0, vy / 6.0]
            nst = [nx / 10.0, ny / 12.0, nvx / 6.0, nvy / 6.0]
            tr.append((st, a, nst))
            x, y, vx, vy = nx, ny, nvx, nvy
    return tr


def _signal(state):
    """A deterministic NONLINEAR lift of the 4-D state -> 8 PREDICTABLE observation dims."""
    x, y, u, v = state
    return [x, y, u, v, x * u, y * v, np.sin(2 * x), np.cos(2 * y)]


def _lift(state, n_noise, rng):
    """Observation = predictable nonlinear signal ++ UNPREDICTABLE fresh noise."""
    return _signal(state) + [rng.uniform(-1.0, 1.0) for _ in range(n_noise)]


def _probe_mse(model, obs_next_pairs):
    """Linear probe: least-squares from the model's latent of the current OBS to the true next
    STATE. Lower MSE => the latent kept more predictable structure (and less noise)."""
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
    n_in = 8 + N_NOISE

    print("=" * 92)
    print("JEPA HIGH-DIM PROBE -- learned latent vs random encoder when obs is mostly noise")
    print(f"obs {n_in} = 8 signal + {N_NOISE} noise; latent {LAT}; steps {STEPS}")
    print("=" * 92)

    trans = gen_dynamics(n_traj=300, steps=40, seed=0)
    nrng = random.Random(123)
    obs_trans = [(_lift(s, N_NOISE, nrng), a, _lift(ns, N_NOISE, nrng)) for s, a, ns in trans]
    probe_pairs = [(o, ns) for (o, _, _), (_, _, ns) in zip(obs_trans, trans, strict=True)]

    jepa = JepaWorldModel(ACTIONS, in_dim=n_in, latent_dim=LAT, seed=0)
    rng = random.Random(1)
    first, last = [], []
    for i in range(STEPS):
        o, a, on = obs_trans[rng.randrange(len(obs_trans))]
        loss = jepa.train_step(o, a, on)
        (first if i < 200 else last if i >= STEPS - 200 else []).append(loss)
    loss0, loss_n = sum(first) / len(first), sum(last) / len(last)
    std_full = jepa.latent_std()

    abl = JepaWorldModel(ACTIONS, in_dim=n_in, latent_dim=LAT, seed=0, anti_collapse=False)
    for _i in range(STEPS):
        o, a, on = obs_trans[rng.randrange(len(obs_trans))]
        abl.train_step(o, a, on)
    std_abl = abl.latent_std()

    rand = JepaWorldModel(ACTIONS, in_dim=n_in, latent_dim=LAT, seed=7)  # untrained control
    mse_learned = _probe_mse(jepa, probe_pairs)
    mse_random = _probe_mse(rand, probe_pairs)
    lift = (1 - mse_learned / mse_random) * 100

    print(f"  invariance loss : {loss0:.4f} -> {loss_n:.4f}")
    print(f"  latent std      : full {std_full:.3f} vs anti-collapse-OFF {std_abl:.3f}")
    print(
        f"  probe MSE       : learned {mse_learned:.4f} vs random {mse_random:.4f} ({lift:+.1f}%)"
    )
    assert loss_n < loss0, f"invariance loss did not drop ({loss0:.4f} -> {loss_n:.4f})"
    assert std_full > std_abl * 2, f"collapse not avoided ({std_full:.3f} vs {std_abl:.3f})"
    assert mse_learned < mse_random, (
        f"learned latent must beat the random encoder here ({mse_learned:.4f} vs {mse_random:.4f})"
    )
    print(f"PROBE OK: learned latent {lift:+.0f}% better; loss drops; no collapse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
