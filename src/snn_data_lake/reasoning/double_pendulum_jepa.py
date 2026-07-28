r"""
DOUBLE-PENDULUM JEPA -- action-conditioned latent prediction over a GENUINELY CHAOTIC system.

physics_jepa.py (near-linear point mass) was an honest negative; physics_jepa_highdim.py won by
adding unpredictable noise dims. This file uses the real thing: a double pendulum, the textbook
chaotic system (sensitive dependence on initial conditions, no closed-form solution). Trajectories
are RK4-integrated; a small torque on the first joint is the discrete action. The observation is the
natural nonlinear lift of the 4-D state (sin/cos of both angles + the two angular velocities +
sin/cos of their difference) plus a few unpredictable noise dims.

Chaos IS the point: over any horizon the fine detail of a double pendulum is unpredictable, so a JEPA
that predicts in latent space should learn to keep the SHORT-horizon predictable structure and drop
the rest, beating a random encoder at decoding the next state. The three un-riggable checks are the
same as the other JEPA files: (1) invariance loss drops, (2) no collapse vs the ablation, (3) the
learned latent decodes the true next state better than a random encoder.

HONEST SCOPE: this is 1-step latent prediction on a chaotic system, not long-horizon rollout (chaos
makes accurate rollout impossible for anyone). The claim is that JEPA's predict-in-latent extracts
the predictable part of a genuinely chaotic world; it is not a claim to have tamed chaos.

  PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python research/double_pendulum_jepa.py
  Knobs: NORD_STEPS(9000) NORD_NOISE(24) NORD_LAT(16) NORD_TRAJ(300)
Deps: numpy (probe) + stdlib (JepaWorldModel).
"""

import math
import os
import random

import numpy as np

from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel

ACTIONS = ("coast", "torque_pos", "torque_neg")
_TORQUE = {"coast": 0.0, "torque_pos": 4.0, "torque_neg": -4.0}


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _deriv(s, tau, g=9.8):
    """Double pendulum (m1=m2=1, l1=l2=1) derivatives, with torque tau on joint 1. Standard equations."""
    th1, th2, w1, w2 = s
    d = th1 - th2
    den = 3.0 - math.cos(2.0 * d)
    dw1 = (
        -3.0 * g * math.sin(th1)
        - g * math.sin(th1 - 2.0 * th2)
        - 2.0 * math.sin(d) * (w2 * w2 + w1 * w1 * math.cos(d))
    ) / den + tau
    dw2 = (
        2.0 * math.sin(d) * (2.0 * w1 * w1 + 2.0 * g * math.cos(th1) + w2 * w2 * math.cos(d))
    ) / den
    return [w1, w2, dw1, dw2]


def _rk4(s, tau, dt):
    k1 = _deriv(s, tau)
    k2 = _deriv([s[i] + 0.5 * dt * k1[i] for i in range(4)], tau)
    k3 = _deriv([s[i] + 0.5 * dt * k2[i] for i in range(4)], tau)
    k4 = _deriv([s[i] + dt * k3[i] for i in range(4)], tau)
    return [s[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]


def gen_chaos(n_traj=300, steps=40, dt=0.02, seed=0):
    """RK4 double-pendulum transitions (state, action, next_state), started near the unstable top for
    real chaos. State normalized: angles wrapped to [-pi,pi], velocities scaled."""
    rng = random.Random(seed)
    tr = []
    for _ in range(n_traj):
        s = [
            math.pi + rng.uniform(-0.4, 0.4),
            math.pi + rng.uniform(-0.4, 0.4),
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
        ]
        for _ in range(steps):
            a = rng.choice(ACTIONS)
            ns = _rk4(s, _TORQUE[a], dt)
            tr.append((_norm(s), a, _norm(ns)))
            s = ns
    return tr


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _norm(s):
    return [_wrap(s[0]), _wrap(s[1]), s[2] / 8.0, s[3] / 8.0]


def _signal(state):
    th1, th2, w1, w2 = state
    return [
        math.sin(th1),
        math.cos(th1),
        math.sin(th2),
        math.cos(th2),
        w1,
        w2,
        math.sin(th1 - th2),
        math.cos(th1 - th2),
    ]


def _lift(state, n_noise, rng):
    return _signal(state) + [rng.uniform(-1.0, 1.0) for _ in range(n_noise)]


def _probe_mse(model, pairs):
    Z = np.array([model.encode(o) for o, _ in pairs], dtype=np.float64)
    Y = np.array([ns for _, ns in pairs], dtype=np.float64)
    cut = int(len(Z) * 0.8)
    Ztr = np.hstack([Z[:cut], np.ones((cut, 1))])
    Zte = np.hstack([Z[cut:], np.ones((len(Z) - cut, 1))])
    W, *_ = np.linalg.lstsq(Ztr, Y[:cut], rcond=None)
    return float(((Zte @ W - Y[cut:]) ** 2).mean())


def _lyapunov_check(seed=0):
    """Confirm the system is CHAOTIC: two trajectories 1e-6 apart diverge exponentially."""
    dt = 0.02
    a = [math.pi + 0.1, math.pi - 0.1, 0.0, 0.0]
    b = [a[0] + 1e-6, a[1], a[2], a[3]]
    for _ in range(300):
        a = _rk4(a, 0.0, dt)
        b = _rk4(b, 0.0, dt)
    return math.dist(a, b)


def main():
    STEPS = _envi("NORD_STEPS", 9000)
    N_NOISE = _envi("NORD_NOISE", 24)
    LAT = _envi("NORD_LAT", 16)
    IN = 8 + N_NOISE

    sep = _lyapunov_check()
    print("=" * 92)
    print(
        "DOUBLE-PENDULUM JEPA -- action-conditioned latent prediction over a genuinely CHAOTIC system"
    )
    print(
        f"obs dim {IN} = 8 nonlinear state features + {N_NOISE} noise; latent {LAT}; steps {STEPS}"
    )
    print(
        f"chaos check: two trajectories 1e-6 apart diverge to {sep:.2e} after 300 steps (>> 1e-6 => chaotic)"
    )
    print("=" * 92)

    trans = gen_chaos(n_traj=_envi("NORD_TRAJ", 300), steps=40, seed=0)
    nrng = random.Random(123)
    obs = [(_lift(s, N_NOISE, nrng), a, _lift(ns, N_NOISE, nrng)) for s, a, ns in trans]
    probe = [(o, ns) for (o, _, _), (_, _, ns) in zip(obs, trans)]

    jepa = JepaWorldModel(ACTIONS, in_dim=IN, latent_dim=LAT, seed=0)
    rng = random.Random(1)
    first, last = [], []
    for i in range(STEPS):
        o, a, on = obs[rng.randrange(len(obs))]
        loss = jepa.train_step(o, a, on)
        (first if i < 200 else last if i >= STEPS - 200 else []).append(loss)
    loss0, lossN = sum(first) / len(first), sum(last) / len(last)
    std_full = jepa.latent_std()

    abl = JepaWorldModel(ACTIONS, in_dim=IN, latent_dim=LAT, seed=0, anti_collapse=False)
    for i in range(STEPS):
        o, a, on = obs[rng.randrange(len(obs))]
        abl.train_step(o, a, on)
    std_abl = abl.latent_std()

    rand = JepaWorldModel(ACTIONS, in_dim=IN, latent_dim=LAT, seed=7)
    mse_learned = _probe_mse(jepa, probe)
    mse_random = _probe_mse(rand, probe)
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
    print("  on a genuinely chaotic double pendulum, the JEPA keeps the short-horizon predictable")
    print(
        "  structure and drops the rest, beating a random encoder. Not a claim to have tamed chaos."
    )
    print("=" * 92)

    assert sep > 1e-3, f"system is not chaotic enough (divergence {sep:.2e})"
    assert lossN < loss0, f"invariance loss did not drop ({loss0:.4f} -> {lossN:.4f})"
    assert std_full > std_abl * 2, (
        f"full model did not avoid collapse ({std_full:.3f} vs {std_abl:.3f})"
    )
    assert mse_learned < mse_random, (
        f"learned latent must beat the random encoder on the chaotic system ({mse_learned:.4f} vs {mse_random:.4f})"
    )
    print(
        f"self-check OK: double-pendulum (chaotic, divergence {sep:.1e}) JEPA -- learned latent decodes "
        f"next-state {lift:+.0f}% better than random (loss {loss0:.3f}->{lossN:.3f}, no collapse "
        f"{std_full:.2f}>>{std_abl:.2f}). Predict-in-latent extracts the predictable part of chaos."
    )
    return 0


if __name__ == "__main__":
    main()
