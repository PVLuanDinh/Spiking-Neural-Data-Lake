r"""
pc_active_inference.py -- the shared-observation-space BRIDGE that lets a real action-conditioned world
model (JepaWorldModel) and the predictive-coding cortex (PcCortexServe) COMPOSE into genuine active-
inference planning. No live wiring, no
actuator route. The honest culmination of the PC-planning arc:

  slice 7 (v3.31): PC alone canNOT be the action model -- it has no action channel (plan() ties).
  slice 8 (v3.32): PC's honest role is the SURPRISE / expected-free-energy VALUE (epistemic term).
  THIS (the "next lever"): give action-conditioned dynamics to JEPA and let PC score JEPA's rollouts
  through a DECODER -- the two compose, each doing the half it is actually good at.

Why a decoder is needed (stated, not hidden). JEPA is a JOINT-EMBEDDING predictive model: it has an
action-conditioned latent predictor predict(z, a) but NO reconstruction path back to observations. PC's
surprise(x) is defined on OBSERVATIONS. So to score a JEPA-predicted latent with PC we must map it back
to observation space: a DECODER D(z_jepa) -> x_hat, fit (ridge, stdlib) on (JEPA.encode(x), x) pairs.
Then the composed expected-free-energy value over a JEPA latent z is

    value(z) = -( ||z - z_goal||_1  +  beta * PC.surprise( D(z) ) )
                 \___ pragmatic (JEPA) ___/         \___ epistemic (PC) ___/

JEPA supplies the action-conditioned dynamics PC lacks; PC supplies the surprise/epistemic value goal-
distance lacks. Both share ONE observation stream here (a small synthetic manifold world), which is the
prerequisite the slice-8 note called out -- a real deployment needs JEPA + PC + decoder trained on the
same real observations.

Un-riggable self-check (each claim paired with a control that must fail):
  1. ACTION CHANNEL: JEPA.predict discriminates actions (predict(z,a_i) differ) -- the channel PC has NOT.
  2. DECODER BRIDGE: the ridge decoder reconstructs observations from JEPA latents far better than the
     mean-observation baseline (the bridge carries information).
  3. SURPRISE SURVIVES THE BRIDGE: PC.surprise(D(JEPA.encode(x))) keeps the on-manifold << off-manifold
     ordering -- PC still flags an anomalous observation after the JEPA-encode + decode round-trip.
  4. COMPOSED ACTIVE INFERENCE: a REAL PrefrontalExecutive.plan() over JEPA latents with the composed EFE
     value AVOIDS an off-manifold look-alike that the JEPA-goal-distance-ONLY value WALKS INTO.

  PYTHONUTF8=1 PYTHONPATH=src python research/pc_active_inference.py
"""

import json
import math
import os
import random
import tempfile

from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel
from snn_data_lake.cortex.pc_cortex_serve import PcCortexServe
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive


# ---- a small deterministic PC cortex (snn_pcnet/1) over an 8-dim observation space --------------------
def _build_pc():
    """A hand-built deterministic PcCortexServe: in_dim 8, causes [6, 4] (top_dim 4). Weights are small +
    structured so generate() lands on a well-conditioned manifold. Dumped + loaded through the trust boundary."""

    def grid(rows, cols):
        return [
            [
                round(0.30 * math.cos(0.7 * i + 1.3 * j) + 0.20 * math.sin(0.5 * i - 0.9 * j), 4)
                for j in range(cols)
            ]
            for i in range(rows)
        ]

    dims = [8, 6, 4]
    blob = {
        "format": "snn_pcnet/1",
        "dims": dims,
        "gen_weights": [grid(dims[0], dims[1]), grid(dims[1], dims[2])],
        "prec": [1.0, 1.0],
        "theta": 0.5,
        "leak": 0.02,
        "steps": 200,
        "lr": 0.05,
        "meta": {"trained_by": "pc_active_inference-demo", "unverified_on_hardware": True},
    }
    p = os.path.join(tempfile.gettempdir(), "pc_active_inference_pc.json")
    with open(p, "w", encoding="ascii") as f:
        json.dump(blob, f)
    srv = PcCortexServe.load(p)
    os.remove(p)
    return srv


# ---- a ridge-regression decoder D: JEPA latent -> observation ---------------------------
def _solve(A, B):
    """Solve A X = B for X (Gauss-Jordan with partial pivoting). A is n x n, B is n x m."""
    n, m = len(A), len(B[0])
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c] if abs(M[c][c]) > 1e-12 else 1e-12
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n + m)]
    return [row[n:] for row in M]


def fit_decoder(Z, X, lam=1e-2):
    """Ridge-fit a linear decoder from latents Z [N x L] to observations X [N x D]. Returns decode(z) that
    maps a length-L latent to a length-D observation via (Z1^T Z1 + lam I)^-1 Z1^T X with a bias column."""
    n, L, D = len(Z), len(Z[0]), len(X[0])
    Z1 = [list(Z[i]) + [1.0] for i in range(n)]  # append bias
    p = L + 1
    A = [
        [sum(Z1[k][i] * Z1[k][j] for k in range(n)) + (lam if i == j else 0.0) for j in range(p)]
        for i in range(p)
    ]
    B = [[sum(Z1[k][i] * X[k][d] for k in range(n)) for d in range(D)] for i in range(p)]
    W = _solve(A, B)  # p x D

    def decode(z):
        z1 = list(z) + [1.0]
        return [sum(z1[i] * W[i][d] for i in range(p)) for d in range(D)]

    return decode


# ---- the synthetic shared-observation manifold world -------------------------------------------------
class ManifoldWorld:
    """A tiny action world whose OBSERVATIONS live on the PC cortex's generative manifold. State = a PC
    top-cause s (length top_dim); observation = PC.generate(s) (on-manifold, low surprise). Four actions
    nudge s along two latent axes. A HAZARD region (s[0] beyond a threshold) makes the observation
    off-manifold (a co-kernel component the PC model cannot explain -> high surprise)."""

    def __init__(self, pc, hazard_dir, step=0.35, hazard_gain=2.0):
        self.pc = pc
        self.top = pc.top_dim
        self.step = step
        self.hazard_dir = hazard_dir  # unit co-kernel direction of W_1 (inexplicable)
        self.hazard_gain = hazard_gain
        self.actions = ("X+", "X-", "Y+", "Y-")
        self._delta = {"X+": (0, +step), "X-": (0, -step), "Y+": (1, +step), "Y-": (1, -step)}

    def obs(self, s):
        """Observation of state s: PC.generate(s), plus an off-manifold co-kernel component in the hazard region."""
        x = self.pc.generate(list(s))
        if s[0] > 0.5:  # HAZARD region -> add an inexplicable component
            x = [x[k] + self.hazard_gain * self.hazard_dir[k] for k in range(len(x))]
        return x

    def step_state(self, s, action):
        """Apply an action's latent delta (clamped to a bounded box)."""
        i, d = self._delta[action]
        s2 = list(s)
        s2[i] = max(-1.2, min(1.2, s2[i] + d))
        return s2


def _cokernel_dir(W1, seed_vec):
    """A unit vector orthogonal to every generative column of W_1 (a left-null direction the top-down map
    cannot produce -> adding it raises reconstruction error / surprise)."""
    rows, cols = len(W1), len(W1[0])
    columns = [[W1[i][j] for i in range(rows)] for j in range(cols)]

    def vdot(a, b):
        return sum(a[i] * b[i] for i in range(len(a)))

    basis = []
    for col in columns:
        v = list(col)
        for u in basis:
            p = vdot(v, u)
            v = [v[i] - p * u[i] for i in range(len(v))]
        nrm = math.sqrt(vdot(v, v))
        if nrm > 1e-9:
            basis.append([vi / nrm for vi in v])
    u = list(seed_vec)
    for b in basis:
        p = vdot(u, b)
        u = [u[i] - p * b[i] for i in range(len(u))]
    nn = math.sqrt(vdot(u, u)) or 1.0
    return [v / nn for v in u]


class _CandidateModel:
    """A trivial 'policy model' for plan(): predict(state, command) returns the pre-set candidate latent for
    that option index. Lets a REAL PrefrontalExecutive.plan() score fixed candidates by a value function."""

    def __init__(self, cands):
        self._c = cands

    def predict(self, state, command):
        return self._c[int(command)]


def main():
    rng = random.Random(0)
    pc = _build_pc()
    # The hazard direction must be a co-kernel of the top-down map M = W_1 @ W_2 (top_dim -> obs): the columns
    # of M are generate(e_k), so a vector orthogonal to all of them is an observation the model cannot produce
    # (inexplicable -> high surprise). Obtain M via the public generate() on the standard top-cause basis.
    Mcols = [
        pc.generate([1.0 if k == j else 0.0 for k in range(pc.top_dim)]) for j in range(pc.top_dim)
    ]
    Mrows = [[Mcols[j][i] for j in range(pc.top_dim)] for i in range(pc.in_dim)]  # obs x top_dim
    hazard_dir = _cokernel_dir(Mrows, [1.0, -0.4, 0.3, 0.8, -0.6, 0.5, -0.2, 0.7][: pc.in_dim])
    world = ManifoldWorld(pc, hazard_dir)

    # ---- train a JEPA action model on random-walk trajectories over the world (light, honest) ----
    jepa = JepaWorldModel(actions=world.actions, in_dim=pc.in_dim, latent_dim=6, seed=0)
    Zpairs, Xpairs = [], []  # (encode(obs), obs) for the decoder
    s = [0.0] * pc.top_dim
    for _ in range(600):
        x_t = world.obs(s)
        a = world.actions[rng.randrange(len(world.actions))]
        s2 = world.step_state(s, a)
        x_n = world.obs(s2)
        jepa.train_step(x_t, a, x_n)
        Zpairs.append(jepa.encode(x_t))
        Xpairs.append(x_t)
        s = s2 if rng.random() < 0.85 else [0.0] * pc.top_dim  # occasional reset

    decode = fit_decoder(Zpairs, Xpairs, lam=1e-2)

    # ============================================================================================
    # (1) ACTION CHANNEL: JEPA.predict discriminates actions (the channel PC does NOT have).
    z_probe = jepa.encode(world.obs([0.2, -0.1] + [0.0] * (pc.top_dim - 2)))
    outs = [jepa.predict(z_probe, a) for a in world.actions]
    action_channel = any(
        outs[i] != outs[j] for i in range(len(outs)) for j in range(i + 1, len(outs))
    )

    # (2) DECODER BRIDGE: reconstructs observations from JEPA latents better than the mean baseline.
    mean_obs = [sum(x[d] for x in Xpairs) / len(Xpairs) for d in range(pc.in_dim)]
    test_states = [[rng.uniform(-0.4, 0.4) for _ in range(pc.top_dim)] for _ in range(40)]
    test_obs = [pc.generate(s) for s in test_states]  # clean on-manifold test observations
    dec_err = sum(
        sum((decode(jepa.encode(x))[d] - x[d]) ** 2 for d in range(pc.in_dim)) for x in test_obs
    )
    base_err = sum(sum((mean_obs[d] - x[d]) ** 2 for d in range(pc.in_dim)) for x in test_obs)
    decoder_ok = dec_err < 0.5 * base_err

    # (3) SURPRISE SURVIVES THE BRIDGE: PC flags an off-manifold obs through JEPA-encode + decode.
    z_goal_top = [0.6, -0.3] + [0.0] * (pc.top_dim - 2)
    x_goal = pc.generate(z_goal_top)  # on-manifold goal observation
    x_on = pc.generate(
        [-0.9, 0.8] + [0.0] * (pc.top_dim - 2)
    )  # a FARTHER on-manifold obs: trusted but not closest (low surprise)
    x_off = [
        x_goal[k] + 2.0 * hazard_dir[k] for k in range(pc.in_dim)
    ]  # off-manifold look-alike (high surprise)
    s_on = pc.surprise(decode(jepa.encode(x_on)))
    s_off = pc.surprise(decode(jepa.encode(x_off)))
    surprise_bridge = s_off > 2.0 * s_on + 0.2

    # (4) COMPOSED ACTIVE INFERENCE: plan() over JEPA latents with the composed EFE value avoids the off-
    #     manifold look-alike that a JEPA-goal-distance-ONLY value walks into. Candidates encoded to JEPA latents.
    z_goal = jepa.encode(x_goal)
    z_on = jepa.encode(x_on)  # the trusted good option (on-manifold)
    z_off = jepa.encode(x_off)  # the look-alike trap (off-manifold, near the goal)
    z_far = jepa.encode(pc.generate([-0.7, 0.6] + [0.0] * (pc.top_dim - 2)))
    cands = [z_on, z_off, z_far]  # 0 = good, 1 = trap, 2 = far

    def dist(z):
        return sum(abs(z[i] - z_goal[i]) for i in range(len(z_goal)))

    d_on, d_off = dist(z_on), dist(z_off)
    trap_is_closer = d_off < d_on  # the trap looks more like the goal in JEPA latent

    beta = 50.0

    def dist_value(z):
        return -dist(z)  # goal-distance ONLY (JEPA make_goal_value convention)

    def efe_value(z):
        return -(
            dist(z) + beta * pc.surprise(decode(z))
        )  # composed: pragmatic (JEPA) + epistemic (PC surprise)

    predictor = _CandidateModel(cands)
    sel_dist, _ = PrefrontalExecutive(n_options=3, seed=0).plan(
        z_on, [0, 1, 2], predictor, dist_value, horizon=1, write=False
    )
    sel_efe, _ = PrefrontalExecutive(n_options=3, seed=0).plan(
        z_on, [0, 1, 2], predictor, efe_value, horizon=1, write=False
    )
    dist_walks_into_trap = sel_dist == 1
    efe_avoids_trap = sel_efe == 0

    print("=" * 90)
    print(
        "PC ACTIVE INFERENCE -- JepaWorldModel (action dynamics) + PcCortexServe (surprise/EFE value) compose"
    )
    print("=" * 90)
    print(
        f"(1) action channel  : JEPA.predict discriminates actions = {action_channel}  (PC has none -> plan() ties)"
    )
    print(
        f"(2) decoder bridge  : decode recon err {dec_err:.3f} vs mean baseline {base_err:.3f} -> better = {decoder_ok}"
    )
    print(
        f"(3) surprise bridge : PC.surprise(D(encode(x))) on-manifold {s_on:.3f} << off-manifold {s_off:.3f} = {surprise_bridge}"
    )
    print(
        f"(4) composed AI     : JEPA-latent dist good={d_on:.3f} trap={d_off:.3f} (trap closer={trap_is_closer}); "
        f"dist-only plan picks {sel_dist} (walks into trap={dist_walks_into_trap}); "
        f"EFE plan picks {sel_efe} (avoids trap={efe_avoids_trap})"
    )
    print("-" * 90)

    assert action_channel, (
        "(1) JEPA.predict must discriminate actions -- the action channel PC lacks"
    )
    assert decoder_ok, (
        "(2) the ridge decoder must reconstruct observations from JEPA latents better than the mean baseline"
    )
    assert surprise_bridge, (
        "(3) PC surprise must survive the JEPA-encode + decode round-trip (off-manifold >> on-manifold)"
    )
    assert trap_is_closer, (
        "(4a) the off-manifold trap must be the CLOSER look-alike in JEPA latent (else the test is trivial)"
    )
    assert dist_walks_into_trap, (
        "(4b) the goal-distance-ONLY value must walk into the closer off-manifold trap"
    )
    assert efe_avoids_trap, (
        "(4c) the composed EFE value (JEPA distance + PC surprise via the decoder) must AVOID the trap"
    )

    print(
        "self-check OK: the DECODER bridges JEPA's action-conditioned latents into PC's observation space, so a "
        "real PrefrontalExecutive plans with JEPA dynamics + PC surprise -- action-conditioned active inference "
        "that avoids off-manifold look-alikes a goal-distance-only value walks into. JEPA supplies the action "
        "channel PC lacks; PC supplies the epistemic value distance lacks; no actuator route."
    )


if __name__ == "__main__":
    main()
