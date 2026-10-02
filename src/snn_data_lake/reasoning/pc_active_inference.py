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

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.pc_active_inference
"""

import json
import math
import os
import random
import tempfile

import numpy as np

from snn_data_lake.faculties.jepa_world_model import JepaWorldModel
from snn_data_lake.sensorimotor.pc_cortex_serve import PcCortexServe
from snn_data_lake.faculties.prefrontal_executive import PrefrontalExecutive


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
# P7 (2026-08-04) -- this is the incident `docs/POLICIES.md` names BY LINE in its P7 table:
# "Three independent hand-rolled linear solvers, none importing numpy | reasoning/nef_compute.py:32,
# reasoning/pc_active_inference.py:85, reasoning/sorn.py:263". nef_compute's copy was already
# deleted; THIS one was still live. A `_solve` doing Gauss-Jordan with partial pivoting, and an
# O(N*p^2) Python triple-comprehension building the Gram matrix, both existed only because the
# zero-dependency rule (abolished 2026-07-20) forbade numpy. Note what the fix actually required,
# because the first draft of this comment got it wrong and the run caught it: this module did NOT
# import numpy, which is precisely the incident ("none importing numpy"). The `import numpy as np`
# at the top of this file is NEW, and adding it is what P7 authorizes -- numpy is an UNCONDITIONAL
# core dependency (pyproject `[project] dependencies`), so it costs nothing that was not already
# installed and already imported by this module's own transitive imports.
#
# A/B'd on the REAL workload this module runs (the 5 bridges x 600 (latent, observation) pairs,
# L=6 -> D=8), median of 20 calls, .venv-lake, no GPU:
#     hand-rolled Gauss-Jordan   7.232 ms per fit
#     np.linalg.solve            0.593 ms per fit      -> 12.2x, x5 fits per run
# Numerics moved only at float64 round-off: over 200 latents from the same bridge the two decoders
# agree to a max RELATIVE difference of 4.6e-10, and np.linalg.solve is additionally better
# conditioned on the normal equations than a hand-rolled elimination. The decode() contract is
# UNCHANGED and deliberately so -- it still returns a plain Python list, because its consumer is
# `PcCortexServe.surprise(x)`, which takes an observation as a list; returning an ndarray here
# would be a silent interface change dressed as a speedup.
def fit_decoder(Z, X, lam=1e-2):
    """Ridge-fit a linear decoder from latents Z [N x L] to observations X [N x D]. Returns decode(z) that
    maps a length-L latent to a length-D observation via (Z1^T Z1 + lam I)^-1 Z1^T X with a bias column."""
    Za = np.asarray(Z, dtype=np.float64)
    Xa = np.asarray(X, dtype=np.float64)
    Z1 = np.hstack([Za, np.ones((len(Za), 1))])  # append bias column
    p = Z1.shape[1]
    W = np.linalg.solve(Z1.T @ Z1 + lam * np.eye(p), Z1.T @ Xa)  # p x D

    def decode(z):
        return (np.concatenate([np.asarray(z, dtype=np.float64), [1.0]]) @ W).tolist()

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


def _bridge(seed, pc, world, n_train=600, n_hold=200):
    """ONE independently seeded JEPA+decoder bridge, plus a HELD-OUT slice of the observation stream it was
    fit on. Trains the JEPA on a random walk through `world`, ridge-fits the decoder on the (encode(x), x)
    pairs of that walk, then walks a FRESH trajectory the decoder never saw for held-out scoring. Returns
    (jepa, decode, held_obs, mean_obs) where mean_obs is the training-mean baseline the decoder is judged
    against. Seed 0 reproduces the single bridge this module used before, byte for byte."""
    rng = random.Random(seed)
    jepa = JepaWorldModel(actions=world.actions, in_dim=pc.in_dim, latent_dim=6, seed=seed)
    Zpairs, Xpairs = [], []  # (encode(obs), obs) for the decoder
    s = [0.0] * pc.top_dim
    for _ in range(n_train):
        x_t = world.obs(s)
        a = world.actions[rng.randrange(len(world.actions))]
        s2 = world.step_state(s, a)
        x_n = world.obs(s2)
        jepa.train_step(x_t, a, x_n)
        Zpairs.append(jepa.encode(x_t))
        Xpairs.append(x_t)
        s = s2 if rng.random() < 0.85 else [0.0] * pc.top_dim  # occasional reset
    held, s = [], [0.0] * pc.top_dim
    for _ in range(n_hold):
        held.append(world.obs(s))
        s = world.step_state(s, world.actions[rng.randrange(len(world.actions))])
        if rng.random() >= 0.85:
            s = [0.0] * pc.top_dim
    mean_obs = [sum(x[d] for x in Xpairs) / len(Xpairs) for d in range(pc.in_dim)]
    return jepa, fit_decoder(Zpairs, Xpairs, lam=1e-2), held, mean_obs


def main():
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

    # ---- train N_BRIDGES JEPA action models on random-walk trajectories over the world (light, honest) ----
    # Bridge 0 IS the single bridge this module used before; the extra four exist only to average the two
    # claims below over independently seeded bridges instead of reading them off one draw. Cost ~0.8s.
    N_BRIDGES = 5
    bridges = [_bridge(k, pc, world) for k in range(N_BRIDGES)]
    jepa, decode, _held0, _mean0 = bridges[0]

    # ============================================================================================
    # (1) ACTION CHANNEL: JEPA.predict discriminates actions (the channel PC does NOT have).
    z_probe = jepa.encode(world.obs([0.2, -0.1] + [0.0] * (pc.top_dim - 2)))
    outs = [jepa.predict(z_probe, a) for a in world.actions]
    action_channel = any(
        outs[i] != outs[j] for i in range(len(outs)) for j in range(i + 1, len(outs))
    )

    # (2) DECODER BRIDGE: reconstructs observations from JEPA latents better than the mean baseline.
    # Scored on a HELD-OUT slice of the same observation stream both the decoder and its mean-observation
    # baseline come from, pooled over the N_BRIDGES bridges: 5 x 200 held-out observations, vs the 40 this
    # check used to draw from pc.generate(uniform(-0.4, 0.4)). That old probe was a narrow clean sliver of
    # the state box with the hazard region excluded, while mean_obs is the mean of the FULL stream -- so
    # base_err there measured how far the sliver sits from the global mean rather than what a mean
    # predictor costs on the data, and the verdict was a coin flip on the seed: over 15 independently
    # seeded bridges the old ratio ran 0.07-4.36 and cleared the 0.5x bar on only 9 of them (with every
    # pinned seed shifted by 0..6 it held on 4 of 7; the shifts that failed read 0.87, 4.48, 0.64). Same
    # decoder, same baseline, same 0.5x bar, matched distribution, 25x the samples: the pooled ratio reads
    # 0.15-0.33 across shifts 0..6, i.e. the bridge explains 67-85% of the observation variance.
    dec_err = sum(
        sum((dc(jp.encode(x))[d] - x[d]) ** 2 for d in range(pc.in_dim))
        for jp, dc, held, _m in bridges
        for x in held
    )
    base_err = sum(
        sum((m[d] - x[d]) ** 2 for d in range(pc.in_dim))
        for _jp, _dc, held, m in bridges
        for x in held
    )
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
    # Run on EVERY bridge, because where the trap lands in latent is a property of the individual randomly
    # initialised encoder, not of the composition under test: over 15 independently seeded bridges the trap
    # was the closer look-alike on 13 (on the 2 stragglers it sat at 1.0x and 1.05x the good option's
    # distance, so the dist-only planner was not fooled either), and with every pinned seed shifted by 0..6
    # the single-bridge check held on 6 of 7. Averaged over N_BRIDGES the mean distances separate cleanly
    # on all 7 shifts (trap 0.31-1.41 vs good 2.44-2.94), and the dist-only planner is fooled on 4-5 of 5.
    beta = 50.0
    d_ons, d_offs, sels_dist, sels_efe = [], [], [], []
    for jp, dc, _held, _m in bridges:
        z_goal = jp.encode(x_goal)
        z_on = jp.encode(x_on)  # the trusted good option (on-manifold)
        z_off = jp.encode(x_off)  # the look-alike trap (off-manifold, near the goal)
        z_far = jp.encode(pc.generate([-0.7, 0.6] + [0.0] * (pc.top_dim - 2)))
        cands = [z_on, z_off, z_far]  # 0 = good, 1 = trap, 2 = far

        def dist(z, zg=z_goal):
            return sum(abs(z[i] - zg[i]) for i in range(len(zg)))

        def dist_value(z, dist=dist):
            return -dist(z)  # goal-distance ONLY (JEPA make_goal_value convention)

        def efe_value(z, dist=dist, dc=dc):
            return -(
                dist(z) + beta * pc.surprise(dc(z))
            )  # composed: pragmatic (JEPA) + epistemic (PC surprise)

        predictor = _CandidateModel(cands)
        sd, _ = PrefrontalExecutive(n_options=3, seed=0).plan(
            z_on, [0, 1, 2], predictor, dist_value, horizon=1, write=False
        )
        se, _ = PrefrontalExecutive(n_options=3, seed=0).plan(
            z_on, [0, 1, 2], predictor, efe_value, horizon=1, write=False
        )
        d_ons.append(dist(z_on))
        d_offs.append(dist(z_off))
        sels_dist.append(sd)
        sels_efe.append(se)

    d_on = sum(d_ons) / N_BRIDGES
    d_off = sum(d_offs) / N_BRIDGES
    trap_is_closer = d_off < d_on  # the trap looks more like the goal in JEPA latent, on average
    n_dist_trap = sum(s == 1 for s in sels_dist)
    n_efe_trap = sum(s == 1 for s in sels_efe)
    dist_walks_into_trap = n_dist_trap > N_BRIDGES // 2
    # HONEST RESTATEMENT: this used to demand sel_efe == 0, i.e. that the EFE plan pick the ONE nominated
    # good candidate. That is not what the claim is -- candidate 2 is on-manifold too, so choosing it also
    # avoids the trap, and which of the two low-surprise options wins is decided by whichever happens to
    # sit nearer the goal in that encoder's latent, an accident of the seed. Measured: sel_efe was 0 on 2
    # of the 7 shifted seeds and 2 on the other 5 (0 and 2 on 15 independently seeded bridges), so the
    # "== 0" form was false more often than true. What IS true on every seed measured -- 7 of 7 shifts,
    # 15 of 15 bridges -- is the claim the assert always stated in words: the composed EFE value NEVER
    # walks into the off-manifold trap, while the goal-distance-only value does on 13 of those 15.
    efe_avoids_trap = n_efe_trap == 0

    print("=" * 90)
    print(
        "PC ACTIVE INFERENCE -- JepaWorldModel (action dynamics) + PcCortexServe (surprise/EFE value) compose"
    )
    print("=" * 90)
    print(
        f"(1) action channel  : JEPA.predict discriminates actions = {action_channel}  (PC has none -> plan() ties)"
    )
    print(
        f"(2) decoder bridge  : decode recon err {dec_err:.3f} vs mean baseline {base_err:.3f} "
        f"(held-out, pooled over {N_BRIDGES} bridges) -> better = {decoder_ok}"
    )
    print(
        f"(3) surprise bridge : PC.surprise(D(encode(x))) on-manifold {s_on:.3f} << off-manifold {s_off:.3f} = {surprise_bridge}"
    )
    print(
        f"(4) composed AI     : mean JEPA-latent dist over {N_BRIDGES} bridges good={d_on:.3f} "
        f"trap={d_off:.3f} (trap closer={trap_is_closer}); dist-only plan walks into the trap on "
        f"{n_dist_trap}/{N_BRIDGES} bridges; EFE plan on {n_efe_trap}/{N_BRIDGES} (avoids it={efe_avoids_trap})"
    )
    print("-" * 90)

    assert action_channel, (
        "(1) JEPA.predict must discriminate actions -- the action channel PC lacks"
    )
    assert decoder_ok, (
        "(2) the ridge decoder must reconstruct held-out observations from JEPA latents better than the "
        "mean baseline, pooled over the independently seeded bridges"
    )
    assert surprise_bridge, (
        "(3) PC surprise must survive the JEPA-encode + decode round-trip (off-manifold >> on-manifold)"
    )
    assert trap_is_closer, (
        "(4a) the off-manifold trap must be the CLOSER look-alike in JEPA latent, averaged over the "
        "bridges (else the test is trivial)"
    )
    assert dist_walks_into_trap, (
        "(4b) the goal-distance-ONLY value must walk into the closer off-manifold trap on most bridges"
    )
    assert efe_avoids_trap, (
        "(4c) the composed EFE value (JEPA distance + PC surprise via the decoder) must AVOID the trap "
        "on EVERY bridge"
    )

    print(
        "self-check OK: the DECODER bridges JEPA's action-conditioned latents into PC's observation space, so a "
        "real PrefrontalExecutive plans with JEPA dynamics + PC surprise -- action-conditioned active inference "
        "that avoids off-manifold look-alikes a goal-distance-only value walks into. JEPA supplies the action "
        "channel PC lacks; PC supplies the epistemic value distance lacks; no actuator route."
    )


if __name__ == "__main__":
    main()
