"""
counterfactual_sim -- brain-native counterfactual "what if I did X" simulation (imagination / prospection).

THE GAP THIS FILLS. The brain's existing mental imagery (cortex/brain.py `imagine`) is TOP-DOWN
SYNTHESIS: it clamps a random-ish top cause and reconstructs a percept -- a "picture this" mechanism
with no notion of ACTION and no scoring. There is no prospective simulation that asks "what if I did
X", rolls the consequence forward, and RANKS candidate actions by their imagined outcome. This module
is that missing piece: a small LEARNED forward model of the latent dynamics, z_next ~ f(z, action),
plus a do()-operator that rolls each candidate action out and scores the imagined result. It is the
counterfactual / planning side of imagination (prospection), complementing the generative side.

LOCAL LEARNING, NO BACKPROP (reuses the e-prop pattern from reasoning/eprop.py). The forward model
is trained by the same three-factor, forward-only rule e-prop uses: a synapse-LOCAL ELIGIBILITY
trace (pre-synaptic activity) times a BROADCAST ERROR (the per-output prediction error), with NO
backpropagation-through-time and NO stored history. For a ONE-STEP feed-forward predictor the
e-prop eligibility e_ji(t) = (pseudo-derivative) x (filtered pre-synaptic activity) reduces exactly
to the INSTANTANEOUS pre-synaptic value (the readout is linear so its pseudo-derivative is 1, and
there is no membrane history to low-pass because the target z_next is supervised one step ahead).
The weight update dW_oi = lr * err_o * z_i is therefore e-prop's `err[t] * ebar` with the trace
ebar collapsed to the raw pre-synaptic vector -- eligibility x broadcast error, no BPTT. Carrying a
leaky trace ACROSS independent (z, action, z_next) samples would be wrong here (it would credit this
sample's error to earlier samples' pre-synaptics), so the trace is per-transition, which is both the
faithful reduction and the correct estimator. Bias synapses use a constant eligibility of 1.

MODEL FORM. A PERSISTENCE prior: the model predicts z_next = z + (W_a . z + b_a), so at
initialisation (W~0, b=0) it predicts "nothing changes" -- a sensible brain-native default -- and
learning ADDS the action-specific effect. One (W_a, b_a) pair per discrete action a.

HONEST CEILING. This is a tiny linear-residual latent forward model, not a deep
world model; the "imagined surprise" it surfaces is a heuristic dispersion of the rolled-out step
sizes, not a calibrated posterior uncertainty. It is representation-only: it scores and RANKS
imagined actions, it never emits a motor command -- there is no actuator route and the arm
E-STOP/safety path is never touched.

Uses `random` seeded per-instance and `zlib.crc32` for the determinism fingerprint;
ASCII output; deterministic.

  Run: python -m snn_data_lake.cortex.counterfactual_sim   # (self-check; also runnable standalone)
"""

import math
import random
import zlib

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "CounterfactualSim.learn": (
        "LOCAL",
        "one local three-factor update -- eligibility (pre-synaptic activity) times a broadcast "
        "per-output error. That is the e-prop SHAPE without the gradient claim: nothing is "
        "back-propagated, so there is no gradient for a cosine to compare against",
    ),
}

# ---- defaults (small, deterministic; the self-check tunes nothing at runtime) ----
W_INIT_SCALE = 0.05  # init half-range of the residual weight W (near-identity persistence prior)
LR_DEFAULT = 0.05  # local three-factor learning rate (eligibility x broadcast error)


class CounterfactualSim:
    """A learned latent forward model z_next ~ f(z, action) with a counterfactual do()-operator.

    Per action a it holds W[a] (dim x dim residual weights) and b[a] (dim bias). The prediction is a
    PERSISTENCE residual  z_next = z + W[a].z + b[a].  Training is local (e-prop-style eligibility x
    broadcast error; see module docstring), so the model improves online with every learn() call.
    """

    def __init__(self, dim, n_actions, seed=0, lr=LR_DEFAULT):
        self.dim = dim
        self.n_actions = n_actions
        self.lr = lr
        self.rng = random.Random(seed)  # per-instance seeding (never module-level)
        self.W = [
            [
                [self.rng.uniform(-W_INIT_SCALE, W_INIT_SCALE) for _ in range(dim)]
                for _ in range(dim)
            ]
            for _ in range(n_actions)
        ]
        self.b = [[0.0] * dim for _ in range(n_actions)]
        self.last_surprise = {}  # action -> imagined surprise from the last counterfactual()

    # ---- forward model -------------------------------------------------------------------------
    def _predict(self, z, action):
        """One-step prediction z_next = z + W[a].z + b[a]. Returns a FRESH list (never aliases z)."""
        W = self.W[action]
        b = self.b[action]
        out = [0.0] * self.dim
        for o in range(self.dim):
            acc = z[o] + b[o]  # persistence prior + learned bias
            Wo = W[o]
            for i in range(self.dim):
                acc += Wo[i] * z[i]
            out[o] = acc
        return out

    def learn(self, z, action, z_next):
        """Apply ONE local three-factor update so the forward model improves online.

        eligibility (pre-synaptic activity) x broadcast error (per-output prediction error), no BPTT.
        Returns the pre-update mean-squared prediction error (for monitoring)."""
        pred = self._predict(z, action)
        err = [z_next[o] - pred[o] for o in range(self.dim)]  # broadcast error (third factor)
        W = self.W[action]
        b = self.b[action]
        lr = self.lr
        for o in range(self.dim):
            eo = err[o]
            Wo = W[o]
            for i in range(self.dim):
                Wo[i] += lr * eo * z[i]  # eligibility_i = pre-synaptic activity z_i
            b[o] += lr * eo  # bias eligibility = 1
        return sum(e * e for e in err) / self.dim

    # ---- imagination / prospection -------------------------------------------------------------
    def imagine(self, z, action, horizon):
        """Roll the forward model `horizon` steps under the COUNTERFACTUAL action (held fixed).

        Returns the imagined latent trajectory [z1, z2, ..., z_horizon] (fresh lists), the mental
        simulation of "what if I keep doing `action` from here". Does not mutate z."""
        traj = []
        cur = list(z)
        for _ in range(horizon):
            cur = self._predict(cur, action)
            traj.append(list(cur))
        return traj

    def _imagined_surprise(self, z0, traj):
        """Heuristic 'imagined surprise' = spread of the imagined per-step change magnitudes.

        Small when the model foresees steady, confident drift; large when the rolled-out step sizes
        disperse. Deterministic; a dispersion proxy, NOT a calibrated uncertainty."""
        mags = []
        prev = z0
        for zt in traj:
            mags.append(math.sqrt(sum((zt[i] - prev[i]) ** 2 for i in range(self.dim))))
            prev = zt
        if len(mags) < 2:
            return 0.0
        mean = sum(mags) / len(mags)
        var = sum((m - mean) ** 2 for m in mags) / len(mags)
        return math.sqrt(var)

    def counterfactual(self, z, candidate_actions, horizon, value_fn, with_surprise=False):
        """The do()-operator: for each candidate action, imagine the rollout and SCORE the outcome.

        value_fn maps an imagined latent -> scalar (higher = better). Each action's score is
        value_fn(final imagined latent); its imagined surprise (model prediction spread) is also
        computed and cached in self.last_surprise. Returns the candidates RANKED best-first
        [(action, score), ...]; with_surprise=True returns [(action, score, surprise), ...].
        Ties keep input order (stable sort) so the ranking is deterministic."""
        records = []
        for a in candidate_actions:
            traj = self.imagine(z, a, horizon)
            score = float(value_fn(traj[-1]))
            surprise = self._imagined_surprise(z, traj)
            records.append((a, score, surprise))
        records.sort(key=lambda r: r[1], reverse=True)  # best-first = highest value
        self.last_surprise = {a: s for (a, _sc, s) in records}
        if with_surprise:
            return [(a, sc, s) for (a, sc, s) in records]
        return [(a, sc) for (a, sc, _s) in records]

    # ---- determinism fingerprint ---------------------------------------------------------------
    def fingerprint(self):
        """crc32 over the (rounded) weights -- a deterministic identity for the learned model
        (zlib.crc32, not builtin hash, per the repo's persisted-value rule)."""
        parts = []
        for a in range(self.n_actions):
            for o in range(self.dim):
                parts.append(",".join(f"{self.W[a][o][i]:.6f}" for i in range(self.dim)))
                parts.append(f"{self.b[a][o]:.6f}")
        return zlib.crc32(";".join(parts).encode("ascii"))


# ---- toy experiment shared by the self-check and the tests -------------------------------------
_DIM, _N_ACTIONS, _HORIZON = 6, 4, 3
_SHIFT = 0.4  # per-action latent-shift magnitude of the KNOWN toy dynamics
_N_TRAIN = 4000  # local-update transitions
_BEST_ACTION = 2  # the action the counterfactual target is reachable by


def _toy_experiment(seed=0):
    """Train on a KNOWN toy dynamics (action a shifts the latent by a fixed action-specific vector
    delta_a) and return the measured numbers used by BOTH the __main__ self-check and the pytest
    suite. Deterministic: every rng is seeded off `seed`."""
    world = random.Random(seed + 101)
    deltas = [[world.uniform(-_SHIFT, _SHIFT) for _ in range(_DIM)] for _ in range(_N_ACTIONS)]

    def true_step(z, a):
        d = deltas[a]
        return [z[i] + d[i] for i in range(_DIM)]

    def rollout_true(z, a, h):
        cur = list(z)
        for _ in range(h):
            cur = true_step(cur, a)
        return cur

    probe_rng = random.Random(seed + 202)
    probes = [
        ([probe_rng.uniform(-0.5, 0.5) for _ in range(_DIM)], probe_rng.randrange(_N_ACTIONS))
        for _ in range(24)
    ]

    def probe_error(sim):
        tot, cnt = 0.0, 0
        for z, a in probes:
            pred = sim.imagine(z, a, _HORIZON)[-1]  # imagined rollout endpoint
            tru = rollout_true(z, a, _HORIZON)
            tot += sum((pred[i] - tru[i]) ** 2 for i in range(_DIM))
            cnt += _DIM
        return tot / cnt

    sim = CounterfactualSim(_DIM, _N_ACTIONS, seed=seed)
    early = probe_error(sim)  # (a) BEFORE training
    train_rng = random.Random(seed + 303)
    for _ in range(_N_TRAIN):
        z = [train_rng.uniform(-0.5, 0.5) for _ in range(_DIM)]
        a = train_rng.randrange(_N_ACTIONS)
        sim.learn(z, a, true_step(z, a))  # local plasticity, online
    late = probe_error(sim)  # (a) AFTER training

    # (b) counterfactual: which action's imagined rollout best REACHES a target?
    z0 = [0.1 * i - 0.2 for i in range(_DIM)]  # fixed start latent
    target = rollout_true(z0, _BEST_ACTION, _HORIZON)  # only _BEST_ACTION lands here

    def value_fn(zi):  # target-distance value: closer = higher
        return -math.sqrt(sum((zi[i] - target[i]) ** 2 for i in range(_DIM)))

    ranked = sim.counterfactual(z0, list(range(_N_ACTIONS)), _HORIZON, value_fn, with_surprise=True)
    return {
        "early": early,
        "late": late,
        "best_action": _BEST_ACTION,
        "ranked": ranked,
        "surprises": sim.last_surprise,
        "fingerprint": sim.fingerprint(),
    }


def main():
    print("=" * 74)
    print("counterfactual_sim -- brain-native 'what if I did X' simulation (imagination)")
    print("learned latent forward model z_next~f(z,a); local e-prop-style update; do()-operator")
    print("=" * 74)

    r = _toy_experiment(0)
    ratio = r["late"] / r["early"] if r["early"] > 0 else 0.0
    ranked_actions = [a for (a, _sc, _s) in r["ranked"]]
    print(
        f"model          : dim={_DIM}, actions={_N_ACTIONS}, horizon={_HORIZON}, "
        f"train transitions={_N_TRAIN}"
    )
    print(
        f"a learns       : imagine() rollout MSE {r['early']:.5f} -> {r['late']:.5f}  "
        f"(ratio {ratio:.4f}; drops with local training)"
    )
    print(
        f"b do()-operator: target-distance value ranks actions {ranked_actions} best-first; "
        f"best={r['best_action']} -> ranked first = {ranked_actions[0] == r['best_action']}"
    )
    print(
        f"               : imagined surprise (prediction spread) per action = "
        f"{ {a: round(s, 4) for a, s in r['surprises'].items()} }"
    )

    # (c) determinism: same seed -> identical numbers, ranking, and weight fingerprint.
    r2 = _toy_experiment(0)
    identical = (
        r["early"] == r2["early"]
        and r["late"] == r2["late"]
        and r["ranked"] == r2["ranked"]
        and r["fingerprint"] == r2["fingerprint"]
    )
    print(
        f"c determinism  : same seed -> identical run + weight crc32 {r['fingerprint']} = {identical}"
    )
    print("=" * 74)

    # ---- self-check (deterministic asserts) ----
    assert r["late"] < 0.1 * r["early"], (
        f"(a) imagine() error must DROP with training: late {r['late']:.5f} !< 0.1*early {r['early']:.5f}"
    )
    assert ranked_actions[0] == r["best_action"], (
        f"(b) counterfactual must rank the target-reaching action first: got {ranked_actions[0]}"
    )
    assert set(ranked_actions) == set(range(_N_ACTIONS)), (
        "(b) ranking must cover every candidate action once"
    )
    assert identical, (
        "(c) same seed must reproduce the run and the weight fingerprint byte-identically"
    )
    print(
        f"self-check OK: forward model LEARNS (imagine MSE {r['early']:.4f}->{r['late']:.4f}, "
        f"ratio {ratio:.3f}); do()-operator RANKS target-reaching action {r['best_action']} first "
        f"of {ranked_actions}; local eligibility x broadcast error, no BPTT; "
        f"deterministic (crc32 {r['fingerprint']})"
    )


if __name__ == "__main__":
    main()
