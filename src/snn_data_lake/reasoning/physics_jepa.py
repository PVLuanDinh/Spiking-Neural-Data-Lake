r"""
PHYSICS-JEPA -- the physics adapter for the autonomous skill builder (target-encoder C: predict the
LATENT of the next state, action-conditioned). This is the honest answer for CONTINUOUS dynamics,
where predicting the raw next state is ill-posed and discretizing loses information: learn to predict
in a LEARNED representation space instead (LeCun's JEPA; V-JEPA-2-AC action-conditioning).

Reuses the repo's `cerebellum.jepa_world_model.JepaWorldModel` (a minimal deterministic linear JEPA:
online encoder + EMA/stop-grad target + action-conditioned residual predictor + hand-rolled VICReg
anti-collapse) as the latent producer. The world is a REAL controlled 2-D point mass: state (x, y,
vx, vy), gravity, and 5 discrete thrust actions; trajectories are generated with random actions and
streamed as (state_t, action_t, state_{t+1}) -- self-supervised, NO labels (the JEPA invariance loss
is the signal).

Un-riggable controls (this is where honesty lives for a JEPA):
  1. the invariance (prediction) loss must DROP over training (it learns to predict in latent space);
  2. the full model's latent must NOT collapse, vs the module's own anti_collapse=False ablation
     which is GUARANTEED able to collapse (latent std -> ~0);
  3. a linear PROBE from the learned latent to the true next-state, reported vs the SAME probe from a
     RANDOM (untrained) encoder. HONEST NEGATIVE observed: over this near-LINEAR 4-D system a random
     projection ALREADY linearly decodes the next state (Johnson-Lindenstrauss: random features
     preserve linear structure), so the learned latent does NOT beat random on a linear probe -- it is
     a few percent WORSE. We print this rather than hide it. The JEPA's real, un-riggable win is (1)
     and (2): it learns to predict in latent space and does not collapse. Beating a random encoder on
     the probe would need a genuinely NONLINEAR world (or a nonlinear probe); this matches the
     project's finding that JEPA representation wins are hard to show at toy scale.

HONEST SCOPE: this produces a lake-indexable latent-dynamics SKILL with measurable learning curves.
It is NOT an accurate simulator and will not roll out precise trajectories or match a numerical PDE
solver. Physics discretized-next-state (target-encoder A) already runs in meta_skill_loop; this is the
latent variant for the continuous case.

  PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.physics_jepa
Deps: numpy (probe) + stdlib (JepaWorldModel is stdlib). Reuses lake_memory + skill_builder.signature.
"""

import os
import random

import numpy as np

from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel
from snn_data_lake.reasoning.brain_adapters.lake_memory import LakeMemory
from snn_data_lake.reasoning.skill_builder import signature

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
    """Real controlled 2-D point-mass dynamics under gravity + discrete thrust. Returns a list of
    (state[x,y,vx,vy] normalized, action_name, next_state) transitions."""
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


def _probe_mse(model, trans):
    """Linear probe: least-squares map from the model's latent of the CURRENT state to the TRUE next
    state. Lower MSE => the latent carries more next-state (dynamics) information."""
    Z = np.array([model.encode(s) for s, _, _ in trans], dtype=np.float64)
    Y = np.array([ns for _, _, ns in trans], dtype=np.float64)
    cut = int(len(Z) * 0.8)
    Ztr, Ytr, Zte, Yte = Z[:cut], Y[:cut], Z[cut:], Y[cut:]
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])  # bias
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
    W, *_ = np.linalg.lstsq(Ztr, Ytr, rcond=None)
    return float(((Zte @ W - Yte) ** 2).mean())


def main():
    STEPS = _envi("NORD_JEPA_STEPS", 6000)
    lake_path = os.path.join(os.environ.get("TEMP", "."), "sndl_physics_jepa.spkl")
    if os.path.exists(lake_path):
        os.remove(lake_path)

    print("=" * 92)
    print(
        "PHYSICS-JEPA -- action-conditioned LATENT next-state prediction over a real 2-D ODE (no labels)"
    )
    print("=" * 92)
    trans = gen_dynamics(n_traj=300, steps=40, seed=0)
    print(
        f"[data] {len(trans):,} (state, action, next-state) transitions; actions {ACTIONS}; state dim 4"
    )

    # ---- train the JEPA (self-supervised invariance loss), track the loss curve ----
    jepa = JepaWorldModel(ACTIONS, in_dim=4, latent_dim=16, seed=0)
    rng = random.Random(1)
    order = list(range(len(trans)))
    first_losses, last_losses = [], []
    for i in range(STEPS):
        s, a, ns = trans[rng.choice(order)]
        loss = jepa.train_step(s, a, ns)
        if i < 200:
            first_losses.append(loss)
        elif i >= STEPS - 200:
            last_losses.append(loss)
    loss0 = sum(first_losses) / len(first_losses)
    lossN = sum(last_losses) / len(last_losses)
    std_full = jepa.latent_std()

    # ---- control 1: the anti_collapse=False ablation is GUARANTEED able to collapse ----
    abl = JepaWorldModel(ACTIONS, in_dim=4, latent_dim=16, seed=0, anti_collapse=False)
    for i in range(STEPS):
        s, a, ns = trans[rng.choice(order)]
        abl.train_step(s, a, ns)
    std_abl = abl.latent_std()

    # ---- control 2: probe from learned latent vs a RANDOM (untrained) encoder ----
    rand = JepaWorldModel(ACTIONS, in_dim=4, latent_dim=16, seed=7)
    mse_learned = _probe_mse(jepa, trans)
    mse_random = _probe_mse(rand, trans)

    # ---- index the skill in the durable lake (state signatures -> skill id) ----
    lake = LakeMemory(path=lake_path, seed=0)

    def sig(s):  # discretize state -> tokens -> 64-d signature
        return signature([f"{k}:{int(v * 6)}" for k, v in zip("xyuv", s)])

    lake.consolidate(
        [sig(s) for s, _, _ in trans[::40][:150]], payloads=[0] * len(trans[::40][:150])
    )
    dur = lake.durability_check()

    print("-" * 92)
    print(
        f"  JEPA invariance loss : {loss0:.4f} -> {lossN:.4f}  (drops as it learns to predict in latent space)"
    )
    print(
        f"  latent std (collapse): full {std_full:.3f}  vs  anti-collapse-OFF ablation {std_abl:.3f}"
    )
    print(
        f"  probe MSE to true next-state: learned {mse_learned:.4f}  vs  random encoder {mse_random:.4f}  "
        f"(HONEST NEGATIVE: {(1 - mse_learned / mse_random) * 100:+.1f}%, random projection already decodes "
        f"a linear world)"
    )
    print(f"  lake: {lake.n_stored} state signatures indexed; durable across restart = {dur['ok']}")
    print("-" * 92)
    print(
        "  honest scope: a lake-indexable latent-dynamics SKILL with learning curves, NOT an accurate"
    )
    print(
        "  simulator. Real wins = latent-space prediction + no collapse; the probe does NOT beat a random"
    )
    print(
        "  encoder over this near-linear system (JL). Rate JEPA + spiking JEPA both hard to show at toy scale."
    )
    print("=" * 92)

    assert lossN < loss0, f"JEPA invariance loss did not drop ({loss0:.4f} -> {lossN:.4f})"
    assert std_full > std_abl * 2, (
        f"full model did not clearly avoid collapse vs the ablation ({std_full:.3f} vs {std_abl:.3f})"
    )
    assert dur["ok"], "physics skill signature did not survive the lake restart"
    # NOTE: the probe control is REPORTED, not asserted -- it is an honest negative (see above), and
    # forcing it would be dishonest on a near-linear world where random features already decode well.
    print(
        f"self-check OK: JEPA learns to predict in latent space (loss {loss0:.3f}->{lossN:.3f}) without "
        f"collapse ({std_full:.2f} >> ablation {std_abl:.2f}); physics latent-skill indexed + durable. "
        f"HONEST: probe does not beat a random encoder here ({(1 - mse_learned / mse_random) * 100:+.0f}%)."
    )
    lake.close()
    return 0


if __name__ == "__main__":
    main()
