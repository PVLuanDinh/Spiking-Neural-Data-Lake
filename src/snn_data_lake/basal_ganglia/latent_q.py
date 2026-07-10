r"""
Shared LINEAR action-value over the JEPA latent -- a TRANSFER substrate that replaces the
arm's TABULAR per-context action-value q[unit][action].

Honest framing (read this first). The arm currently learns a tabular action-value: one row
q[unit] per discrete context (a GrowingLearner unit), updated bootstrap-free by q += lr*(r - q).
A table has ZERO transfer -- mastering one context teaches nothing about any other context,
because every context is an independent row. So a curriculum's later stages get no benefit from
mastery of the earlier ones. This module replaces that table with ONE weight set shared across
ALL contexts, evaluated over the JEPA latent z (cerebellum/jepa_world_model.encode, a unit vector
in R^latent_dim):

    Q(z, a) = dot(W[a], z) + b[a]          # ONE W (n_actions x latent_dim), b (n_actions), shared

This is textbook LINEAR value-function approximation (Sutton & Barto, ch. 9-10; semi-gradient TD).
Because similar contexts have similar latents z, an update from one context GENERALIZES to nearby
contexts -- so mastery TRANSFERS along the latent geometry instead of being trapped in one row.

The learning rule is the semi-gradient / Widrow-Hoff (LMS) step toward a scalar `target`. The arm
passes the observed reward as the target, matching its current bootstrap-free tabular update
q += lr*(r - q); the linear analogue is

    err        = target - Q(z, a)
    W[a][i]   += lr * err * z[i]           # gradient of 0.5*err^2 w.r.t. W[a] is -err*z
    b[a]      += lr * err

update() returns err. It is ONE primitive -- a linear map plus a delta rule -- nothing more.

The SAME shared W that gives transfer also means the update can INTERFERE: when two contexts have
nearby latents but CONFLICTING optimal actions, an update that helps one nudges the other the wrong
way, and neither can be fit as well as a dedicated table row would fit it in isolation. Transfer and
interference are two faces of the one shared representation; this is honest, not a bug. Linear
approximation also cannot represent an action-value that is a non-linear function of z that no linear
map separates -- for that the latent itself (a deeper JEPA encoder) must carry the non-linearity.

Distinctness. This is the VALUE table (which action is worth most in a context), not the SELECTION
circuit (action_gating.py's Go/NoGo disinhibition that releases exactly one action) nor the
DECISION-by-integration model (evidence_accumulation.py). It is the drop-in replacement for the
scalar q the reward-learning side already keeps; values(z) returns a dict keyed by action name with
the same shape as the old q[unit] row, so it wires in without changing the selection stage.

Pure stdlib (random; dot/unit reused from substrate._mathutils), ASCII-only prints, deterministic
(random.Random(seed) per instance -- identical seed => byte-identical weights).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.basal_ganglia.latent_q
"""
import random

from snn_data_lake.substrate._mathutils import dot, unit


class LatentQ:
    """Shared linear action-value Q(z, a) = dot(W[a], z) + b[a] over the JEPA latent z.

    ONE weight set W (n_actions x latent_dim) and bias b (n_actions) are shared across every context,
    so a semi-gradient update from one latent z transfers to all nearby latents. values(z) returns the
    all-action value row as a dict keyed by action name (same shape as the tabular q[unit] it replaces).
    """

    def __init__(self, actions, latent_dim, lr=0.1, seed=0):
        self.actions = tuple(actions)                  # ordered action names
        self.n = len(self.actions)
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.seed = seed
        self.rng = random.Random(seed)                 # per-instance rng; same seed => same weights
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        # small random W (breaks ties / symmetry), zero bias -- matches the tabular q's zero init
        scale = 0.01
        self.W = [[self.rng.uniform(-1.0, 1.0) * scale for _ in range(self.latent_dim)]
                  for _ in range(self.n)]
        self.b = [0.0] * self.n

    def _idx(self, a):
        """Map an action name or an integer index to an action index (duck-typed, like the JEPA)."""
        if isinstance(a, str):
            return self._aidx[a]
        return int(a)

    def values(self, z):
        """All-action value row over latent z as a dict {action_name: Q(z, a)} (shape of old q[unit])."""
        return {self.actions[i]: dot(self.W[i], z) + self.b[i] for i in range(self.n)}

    def value(self, z, a):
        """Q(z, a) for one action (a = action name or integer index)."""
        i = self._idx(a)
        return dot(self.W[i], z) + self.b[i]

    def greedy(self, z):
        """argmax action name over Q(z, .); deterministic tie-break to the LOWEST index."""
        best_i = 0
        best_v = dot(self.W[0], z) + self.b[0]
        for i in range(1, self.n):
            v = dot(self.W[i], z) + self.b[i]
            if v > best_v:                             # strict > keeps the lowest index on ties
                best_v, best_i = v, i
        return self.actions[best_i]

    def update(self, z, a, target):
        """Semi-gradient / Widrow-Hoff step of Q(z, a) toward scalar `target`; returns err.

        err = target - Q(z, a); W[a][i] += lr*err*z[i]; b[a] += lr*err. Because W is SHARED, this one
        step shifts Q(., a) for ALL nearby-z contexts -- that is the transfer (and the interference)."""
        i = self._idx(a)
        err = target - (dot(self.W[i], z) + self.b[i])
        lr = self.lr
        wi = self.W[i]
        for k in range(self.latent_dim):
            wi[k] += lr * err * z[k]
        self.b[i] += lr * err
        return err


# =========================================================================================================
# Self-check. The whole point is TRANSFER / GENERALIZATION, so the anchors compare LatentQ (shared linear
# over the latent) against a FAIR tabular control (a dict keyed by a quantization of z) on a DISJOINT
# held-out split of latents neither model ever saw during training.
#
# Rule (a function OF the latent): split R^latent_dim into n_actions disjoint contiguous groups; the
# optimal action for a latent z is the group whose coordinates have the largest sum. This is linearly
# realizable (each group-sum is a linear functional of z), so a shared linear Q CAN transfer to new z;
# a table keyed by z CANNOT -- an unseen key has no learned value.
# =========================================================================================================


def _rule_action(z, n_actions):
    """Optimal action = index of the contiguous latent group with the largest coordinate sum."""
    g = len(z) // n_actions
    best_i, best_s = 0, sum(z[0:g])
    for j in range(1, n_actions):
        s = sum(z[j * g:(j + 1) * g])
        if s > best_s:
            best_s, best_i = s, j
    return best_i


def _make_contexts(n, latent_dim, n_actions, rng):
    """n (z, optimal_action) pairs; z is a unit vector in R^latent_dim (matches JEPA encode())."""
    out = []
    for _ in range(n):
        z = unit([rng.gauss(0.0, 1.0) for _ in range(latent_dim)])
        out.append((z, _rule_action(z, n_actions)))
    return out


def _quant_key(z, bins=10):
    """Quantize a latent to a hashable table key -- the tabular control's discretization of z."""
    return tuple(round(v * bins) for v in z)


class _TabularQ:
    """Fair control: the arm's tabular action-value, keyed by a quantization of z instead of by a
    GrowingLearner unit. SAME data, SAME bootstrap-free q += lr*(r - q) update. It cannot transfer:
    a held-out (unseen-key) latent has no learned row, so greedy falls back to the lowest index."""

    def __init__(self, n_actions, lr=0.1, bins=10):
        self.n = n_actions
        self.lr = lr
        self.bins = bins
        self.q = {}                                    # (quant_key, action) -> value

    def update(self, z, a, target):
        k = (_quant_key(z, self.bins), a)
        cur = self.q.get(k, 0.0)
        self.q[k] = cur + self.lr * (target - cur)     # q += lr*(r - q): identical to the arm's rule

    def greedy(self, z):
        key = _quant_key(z, self.bins)
        best_i, best_v = 0, self.q.get((key, 0), 0.0)
        for i in range(1, self.n):
            v = self.q.get((key, i), 0.0)
            if v > best_v:                             # unseen key => all zeros => lowest index
                best_v, best_i = v, i
        return best_i


def _held_out_acc(model, contexts, name_of=None):
    """Fraction of held-out contexts whose greedy() equals the optimal action."""
    ok = 0
    for z, opt in contexts:
        g = model.greedy(z)
        gi = name_of[g] if name_of is not None else g
        ok += (gi == opt)
    return ok / len(contexts)


def main():
    LATENT, N_ACT = 6, 3
    ACTIONS = ("A0", "A1", "A2")
    idx_of = {a: i for i, a in enumerate(ACTIONS)}
    N_TRAIN, N_TEST, PASSES, LR = 300, 200, 8, 0.1

    # DISJOINT train / test latents (different rng streams -> the test z's are never seen in training).
    train = _make_contexts(N_TRAIN, LATENT, N_ACT, random.Random(1))
    test = _make_contexts(N_TEST, LATENT, N_ACT, random.Random(2))

    # ---- (1)+(2) train LatentQ and the tabular control on the SAME train split, reward +1/-1 ----
    lq = LatentQ(ACTIONS, LATENT, lr=LR, seed=0)
    tab = _TabularQ(N_ACT, lr=LR, bins=10)
    for _ in range(PASSES):
        for z, opt in train:
            for i, a in enumerate(ACTIONS):
                r = 1.0 if i == opt else -1.0          # reward = target (bootstrap-free), for both models
                lq.update(z, a, r)
                tab.update(z, i, r)
    lq_holdout = _held_out_acc(lq, test, name_of=idx_of)
    tab_holdout = _held_out_acc(tab, test)
    lq_train = _held_out_acc(lq, train, name_of=idx_of)

    # ---- (3) online learning: on a small fixed context set, update() drives greedy() to optimal ----
    fixed = _make_contexts(4, LATENT, N_ACT, random.Random(7))
    online = LatentQ(ACTIONS, LATENT, lr=0.2, seed=3)
    v0 = online.value(fixed[0][0], fixed[0][1])        # optimal-action value BEFORE any reward
    for _ in range(60):
        for z, opt in fixed:
            for i, a in enumerate(ACTIONS):
                online.update(z, a, 1.0 if i == opt else -1.0)
    online_ok = all(idx_of[online.greedy(z)] == opt for z, opt in fixed)
    z0, opt0 = fixed[0]
    v1 = online.value(z0, opt0)                        # optimal-action value AFTER reward
    others_below = all(online.value(z0, i) < v1 for i in range(N_ACT) if i != opt0)

    # ---- (4) determinism: two LatentQ(seed=s) on the same stream -> identical weights + greedy ----
    stream = [(z, opt % N_ACT, 1.0 if k % 2 else -1.0) for k, (z, opt) in enumerate(train[:120])]
    da = LatentQ(ACTIONS, LATENT, lr=LR, seed=11)
    db = LatentQ(ACTIONS, LATENT, lr=LR, seed=11)
    for z, a, r in stream:
        da.update(z, a, r)
        db.update(z, a, r)
    same_w = (da.W == db.W and da.b == db.b)
    same_greedy = all(da.greedy(z) == db.greedy(z) for z, _o in test[:50])

    print("=" * 74)
    print("LATENT Q -- shared linear action-value over the JEPA latent (transfer vs tabular)")
    print("=" * 74)
    print(f"latent_dim={LATENT} | actions={ACTIONS} | train={N_TRAIN} test={N_TEST} "
          f"| passes={PASSES} lr={LR}")
    print(f"rule: optimal action = latent group ([{LATENT // N_ACT} dims each]) with the largest sum")
    print(f"(1) LatentQ  held-out accuracy : {lq_holdout:.3f}   (train {lq_train:.3f}) -> GENERALIZES")
    print(f"(2) tabular  held-out accuracy : {tab_holdout:.3f}   (unseen z-keys -> ~chance {1.0 / N_ACT:.3f})")
    print(f"    => shared latent transfers where the table cannot ({lq_holdout:.3f} >> {tab_holdout:.3f})")
    print(f"(3) online   : greedy reaches optimal on {len(fixed)} fixed contexts = {online_ok}; "
          f"rewarded value {v0:+.3f} -> {v1:+.3f} (rose above the others: {others_below})")
    print(f"(4) determin : two seed=11 models identical weights={same_w}, identical greedy={same_greedy}")
    print("=" * 74)

    # ---- un-riggable anchors (deterministic asserts) --------------------------------------------
    # (1) TRANSFER: LatentQ generalizes to held-out latents it never trained on.
    assert lq_holdout > 0.85, f"LatentQ must generalize to held-out latents (>0.85), got {lq_holdout:.3f}"
    # (2) FAIR CONTROL: the tabular Q cannot transfer -- unseen keys leave it near chance.
    assert tab_holdout < 0.45, f"tabular Q must NOT transfer (<0.45), got {tab_holdout:.3f}"
    assert lq_holdout > tab_holdout + 0.30, (
        f"shared latent must beat tabular by a wide margin ({lq_holdout:.3f} vs {tab_holdout:.3f})")
    # (3) ONLINE: reward drives greedy to the optimal action and lifts its value above the others.
    assert online_ok, "online update() must drive greedy() to the optimal action on the fixed set"
    assert v1 > v0, f"rewarding the optimal action must raise its value ({v0:.3f} -> {v1:.3f})"
    assert others_below, "the rewarded action's value must rise above every competing action"
    # (4) DETERMINISM: identical seed + stream => byte-identical weights and identical greedy choices.
    assert same_w, "two LatentQ(seed=11) on the same stream must have byte-identical weights"
    assert same_greedy, "two LatentQ(seed=11) must make identical greedy choices"

    print(f"self-check OK: LatentQ generalizes to held-out latents ({lq_holdout:.3f}) while the tabular "
          f"control cannot ({tab_holdout:.3f} ~ chance);")
    print(f"  online reward drives greedy to optimal (value {v0:+.3f} -> {v1:+.3f}); "
          "identical seed => byte-identical weights + greedy")


if __name__ == "__main__":
    main()
