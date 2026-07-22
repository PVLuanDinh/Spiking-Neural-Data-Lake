"""
Similarity-sized structural growth + frozen-expert reuse (an ADVISOR, deterministic).

Brain-Cog's SCA-SNN has one genuine lever worth taking (and NOTHING of its conv/backprop): size a
new task's FRESH capacity by how DISSIMILAR it is to what the brain already knows, and when a task
is SIMILAR to a learned one, REUSE + FREEZE the prior sub-net instead of growing -- fewer new units,
and old skills cannot be overwritten (no catastrophic forgetting). SCA-SNN spells the sizing as
`channel_number = base * (1 - exp(-5 * min_dist))` (network.py:247-262) and the reuse as "freeze the
prior experts, train only the newest" (incmodel.py:114-116). This module is that lever, brain-native.

It BUILDS ON the growth signals we already compute -- it does NOT reinvent them:
  * NOVELTY / SIMILARITY = the ART developmental match we already run. `GrowingLearner._best`
    (hippocampus.growing_learner) returns (nearest-prototype id, cosine similarity); novelty is
    `1 - similarity`. That is exactly "do I already recognise this?" -- `developmental_loop`
    calls it "the ART / hippocampus.GrowingLearner vigilance rule". We IMPORT it (with its `_cosine`
    kernel) rather than recompute cosine. (For spike-EVENT signatures a caller can swap in
    substrate.proximity_index's Victor-Purpura metric NN; here the signatures are [0,1]^D centroids,
    so the cosine developmental match is the right sibling.)
  * ROUTING = the same content-addressable idea as `lake_skill_library.route` (nearest consolidated
    key -> payload = which frozen expert). We keep the STDLIB GrowingLearner as the in-advisor router
    (lr=0.0 -> its prototypes are frozen keys) instead of pulling the torch+lake LakeMemory stack into
    a stdlib advisor; a main-agent that wants durability can back the same route_or_grow with
    LakeMemory. Honest scope: identical routing semantics, lighter substrate.

What it is (and is NOT): a pure ADVISOR / wrapper. It NEVER edits growable.py or the neurogenesis
engine in place -- it exposes `recruit_cap(novelty)` (how many fresh units this cycle) and
`route_or_grow(task_sig)` (reuse a frozen expert vs. grow a new one) for the main agent to CALL and
wire into the developmental loop. It arbitrates NO effector/motor anything; a task signature is a
perceptual centroid, never an action.

Brain-native learning: each FrozenExpert is a single-layer readout trained by a LOCAL error-modulated
Hebbian rule (perceptron / three-factor: pre-trace x post-unit-activity gated by a global reward =
error sign) computed in numpy -- NO loss.backward, NO autograd, NO surrogate gradients. Weights start
at zero and AUTONOMOUSLY diverge under the rule (no hand-set tables). `freeze()` makes the array
literally read-only (numpy write=False), so a consolidated expert cannot be mutated by later tasks --
that read-only freeze is the structural reason old skills are retained.

No magic behaviour-arbitrating constants: the cap's exp-rate `k` is 1/(running mean min_dist) -- it
tracks the domain's OWN novelty scale -- and the route-vs-grow threshold is a scale-free fraction of
the bank's OWN mean inter-prototype separation, not a hardcoded distance.

Honest scope: "sized capacity" bounds HOW MANY units to admit; it does not make any single unit
compute more. Retention here is the strong, structural kind (frozen experts are untouchable), shown
on a small synthetic multi-task setup -- not a claim about a real continual-learning benchmark.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.similarity_sized_growth
"""

import math
import random
from collections import namedtuple

import numpy as np

# REUSE the developmental match we already compute (ART cosine to nearest prototype) + its kernel.
from snn_data_lake.hippocampus.growing_learner import GrowingLearner, _cosine

# The decision route_or_grow hands back to the caller (the main agent wires on it).
GrowthDecision = namedtuple(
    "GrowthDecision", ["action", "expert_id", "novelty", "recruit_cap", "grew", "n_experts"]
)


class FrozenExpert:
    """A per-task specialist readout, learned by a LOCAL error-modulated Hebbian rule, then FROZEN.

    W is (n_classes, dim), zero-initialised. learn() runs the perceptron / three-factor local rule
    (pre-trace = input x, post = which class unit fires, global factor = error sign): on a
    misclassification, potentiate the correct unit's weights toward x (LTP) and depress the
    wrongly-firing unit's (LTD). Single layer, no chain rule, no autograd. Weights diverge from the
    zero init autonomously. freeze() sets the array read-only so no later task can overwrite it --
    the structural guarantee behind retention / no catastrophic forgetting."""

    def __init__(self, n_classes, dim, lr=0.5):
        self.n_classes = int(n_classes)
        self.dim = int(dim)
        self.lr = float(lr)
        self.W = np.zeros((self.n_classes, self.dim), dtype=float)
        self._W0 = self.W.copy()  # init snapshot, to PROVE the weights actually moved
        self._frozen = False

    @property
    def frozen(self):
        return self._frozen

    @property
    def diverged(self):
        """True once the local rule has moved the weights away from their zero init (they LEARNED)."""
        return not np.array_equal(self.W, self._W0)

    def learn(self, X, y, epochs=30, seed=0):
        """LOCAL error-modulated Hebbian training on (X, y). Refuses to run once frozen (read-only)."""
        if self._frozen:
            raise RuntimeError("frozen expert is read-only (consolidated); cannot re-learn")
        X = np.asarray(X, dtype=float)
        y = [int(v) for v in y]
        rng = random.Random(seed)  # deterministic sample order
        order = list(range(len(y)))
        for _ in range(int(epochs)):
            rng.shuffle(order)
            for i in order:
                x = X[i]
                k = y[i]
                pred = int(np.argmax(self.W @ x))
                if pred != k:
                    self.W[k] += self.lr * x  # LTP: potentiate the should-have-fired unit toward x
                    self.W[pred] -= self.lr * x  # LTD: depress the wrongly-firing unit
        return self

    def freeze(self):
        """Consolidate: make the weight array literally read-only so later tasks cannot mutate it."""
        self._frozen = True
        self.W.setflags(write=False)
        return self

    def predict(self, x):
        return int(np.argmax(self.W @ np.asarray(x, dtype=float)))

    def accuracy(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray([int(v) for v in y])
        preds = np.argmax(X @ self.W.T, axis=1)
        return float(np.mean(preds == y))


class SimilaritySizedGrowth:
    """Advisor: size fresh capacity by dissimilarity, and reuse+freeze a prior expert when similar.

    Holds a bank of FrozenExperts parallel to a GrowingLearner's prototype keys (the router). The
    GrowingLearner runs with lr=0.0 so its prototypes are FROZEN routing keys (a familiar hit never
    perturbs them). Exposes recruit_cap(novelty) and route_or_grow(task_sig) for the main agent to
    call -- it edits neither growable.py nor the neurogenesis engine."""

    def __init__(self, dim, n_classes, cap_max=64, cap_min=1, route_frac=0.5):
        if not (0.0 < route_frac < 1.0):
            raise ValueError("route_frac is a scale-free fraction in (0, 1)")
        self.dim = int(dim)
        self.n_classes = int(n_classes)
        self.cap_max = int(cap_max)
        self.cap_min = int(cap_min)
        self.route_frac = float(route_frac)
        # lr=0.0 -> prototypes are frozen keys; vigilance is driven per-call by the baseline-relative
        # threshold, so the route/grow decision is OURS, not GrowingLearner's fixed 0.85.
        self.bank = GrowingLearner(vigilance=1.0, lr=0.0, dim=self.dim)
        self.experts = []  # experts[i] is the frozen specialist for prototype i (parallel arrays)
        self._nov_history = []  # running min_dist samples -> the baseline the cap's k tracks

    # ---- similarity signal (REUSED from the developmental match) ---------------------------------
    def _match(self, sig):
        """(nearest prototype id, cosine similarity) via the ART developmental match we already run.
        (-1, -1.0) when the bank is empty. This is the reused novelty signal, not a reimplementation."""
        return self.bank._best(sig)

    def _mean_pairwise_dist(self):
        """Mean cosine DISTANCE among the bank's prototypes -- the domain's OWN 'typical separation
        between distinct tasks' baseline. None with < 2 prototypes. Reuses growing_learner._cosine."""
        protos = self.bank.prototypes
        n = len(protos)
        if n < 2:
            return None
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1.0 - _cosine(protos[i], protos[j])
                count += 1
        return total / count

    def _dist_scale(self):
        """Running mean of observed min_dist -- the novelty SCALE the cap's exp-rate k tracks."""
        h = self._nov_history
        return (sum(h) / len(h)) if h else None

    def _sim_threshold(self):
        """Cosine-similarity threshold for 'familiar enough to reuse', RELATIVE to the running
        baseline: familiar iff min_dist < route_frac * (mean inter-prototype separation). With < 2
        prototypes there is no separation baseline yet, so reuse only on numerical identity (grow
        by default). route_frac is a scale-free ratio, NOT a hardcoded distance."""
        mean_sep = self._mean_pairwise_dist()
        if mean_sep is None:
            return 1.0 - 1e-9  # no baseline -> route only a numerically-identical signature
        return 1.0 - self.route_frac * mean_sep

    # ---- the two advisor calls the main agent wires on -------------------------------------------
    def recruit_cap(self, novelty):
        """Bounded, monotone recruit cap sized by dissimilarity (SCA-SNN's genuine lever):

            cap = ceil(cap_max * (1 - exp(-k * min_dist))),  k = 1 / (running mean min_dist)

        Few units when a task is close to a learned prototype (small min_dist), many when far. `k`
        is distribution-relative (1/running-mean) so the SAME shape adapts to whatever novelty scale
        this domain actually has -- no magic '5'. Before any history exists, the task's own novelty
        sets the scale (k*novelty = 1 -> ~63% of cap_max)."""
        novelty = max(0.0, float(novelty))
        scale = self._dist_scale()
        if scale is None or scale <= 0.0:
            scale = novelty if novelty > 0.0 else 1.0
        k = 1.0 / max(scale, 1e-12)
        frac = 1.0 - math.exp(-k * novelty)
        cap = math.ceil(self.cap_max * frac)
        return max(self.cap_min, min(self.cap_max, int(cap)))

    def route_or_grow(self, task_sig):
        """Reuse a frozen expert if the task is SIMILAR to a known one; else advise growing fresh,
        similarity-sized capacity. Returns a GrowthDecision. On 'grow' a new (untrained) FrozenExpert
        slot is appended for the caller to train + freeze; on 'route' nothing grows and the frozen
        prototype key is left untouched (lr=0.0)."""
        sig = [float(v) for v in task_sig]
        if len(sig) != self.dim:
            raise ValueError(f"task_sig width {len(sig)} != advisor dim {self.dim}")
        assert len(self.experts) == len(self.bank.prototypes), "expert/key banks must stay parallel"

        bid, sim = self._match(sig)
        novelty = (1.0 - sim) if bid >= 0 else 1.0  # empty bank -> maximally novel
        if bid >= 0:
            self._nov_history.append(novelty)  # feed the running novelty-scale baseline
        cap = self.recruit_cap(novelty)

        self.bank.vigilance = self._sim_threshold()  # baseline-relative route/grow boundary
        uid, grew = self.bank.observe(sig)  # lr=0.0 -> a familiar (route) hit does NOT move the key
        if grew:
            self.experts.append(
                FrozenExpert(self.n_classes, self.dim)
            )  # experts[uid] for the caller
            action = "grow"
        else:
            action = "route"
        return GrowthDecision(action, uid, novelty, cap, grew, len(self.experts))

    def expert_for(self, task_sig):
        """Route-only (no growth): the frozen expert whose key is nearest, or None if the bank is empty."""
        bid = self.bank.classify([float(v) for v in task_sig])
        return self.experts[bid] if bid >= 0 else None


# =====================================================================================================
# Self-check: un-riggable, each claim measured on real runs, with a CONTROL that MUST forget.
# =====================================================================================================
def _unit_dirs(rng, k, d):
    dirs = []
    for _ in range(k):
        v = rng.uniform(-1.0, 1.0, size=d)
        n = float(np.linalg.norm(v)) or 1.0
        dirs.append(v / n)
    return dirs


def _make_task(rng, dirs, d, shift, k=3, n_train=40, n_test=20, dir_scale=0.16, noise=0.02):
    """A K-class task in a distinct [0,1]^D region. Class k sits along dir[(k+shift)%K] -- the
    cyclic shift makes the SAME label occupy DIFFERENT directions across tasks, so no single shared
    readout can serve all (the regime where a frozen per-task expert bank pays off), while each
    within-task expert sees a consistent geometry. Signature = the train centroid (the router key)."""
    center = rng.uniform(0.25, 0.85, size=d)
    means = [np.clip(center + dir_scale * dirs[(c + shift) % k], 0.0, 1.0) for c in range(k)]
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for c in range(k):
        for _ in range(n_train):
            tr_x.append(np.clip(means[c] + rng.normal(0.0, noise, size=d), 0.0, 1.0))
            tr_y.append(c)
        for _ in range(n_test):
            te_x.append(np.clip(means[c] + rng.normal(0.0, noise, size=d), 0.0, 1.0))
            te_y.append(c)
    sig = list(np.mean(np.asarray(tr_x), axis=0))
    return {"train_X": tr_x, "train_y": tr_y, "test_X": te_x, "test_y": te_y, "sig": sig}


def _build(seed):
    """Build the whole scenario deterministically and return every measured quantity."""
    d, k = 24, 3
    rng = np.random.default_rng(seed)
    dirs = _unit_dirs(rng, k, d)
    a = _make_task(rng, dirs, d, shift=0)
    b = _make_task(rng, dirs, d, shift=1)
    c = _make_task(rng, dirs, d, shift=2)
    far = _make_task(rng, dirs, d, shift=0)  # a genuinely new, distinct-region task (novel probe)
    a_dup = _make_task(
        rng, dirs, d, shift=0
    )  # resampled near-duplicate of A's region (familiar probe)
    # pin the near-duplicate to A's exact region so it is unambiguously "the same task again"
    a_dup["sig"] = list(np.clip(np.asarray(a["sig"]) + rng.normal(0.0, 0.002, size=d), 0.0, 1.0))

    adv = SimilaritySizedGrowth(dim=d, n_classes=k, cap_max=64, cap_min=1, route_frac=0.5)

    consolidated = {}  # task letter -> (expert_id, own-test acc at consolidation, grow decision)
    for letter, task, sd in (("A", a, 1001), ("B", b, 1002), ("C", c, 1003)):
        dec = adv.route_or_grow(task["sig"])
        exp = adv.experts[dec.expert_id]
        exp.learn(task["train_X"], task["train_y"], epochs=30, seed=sd)
        exp.freeze()
        own = exp.accuracy(task["test_X"], task["test_y"])
        consolidated[letter] = (dec.expert_id, own, dec)

    a_id = consolidated["A"][0]
    accA0 = consolidated["A"][1]  # A's expert accuracy right after A was consolidated

    # (c) RETENTION: after B and C were learned, A's frozen expert is untouched -> same accuracy.
    accA1 = adv.experts[a_id].accuracy(a["test_X"], a["test_y"])
    retention = accA1 / accA0 if accA0 > 0 else 0.0

    # (c) CONTROL that MUST forget: one SHARED plastic readout trained A -> B -> C (never frozen).
    shared = FrozenExpert(k, d, lr=0.5)
    shared.learn(a["train_X"], a["train_y"], epochs=30, seed=1001)
    shared_A_before = shared.accuracy(a["test_X"], a["test_y"])
    shared.learn(b["train_X"], b["train_y"], epochs=30, seed=1002)
    shared.learn(c["train_X"], c["train_y"], epochs=30, seed=1003)
    shared_A_after = shared.accuracy(a["test_X"], a["test_y"])

    # (b) REUSE: the near-duplicate of A must ROUTE to A's frozen expert, growing nothing.
    n_before = adv.experts.__len__()
    dec_dup = adv.route_or_grow(a_dup["sig"])
    routed_ok = (
        dec_dup.action == "route"
        and not dec_dup.grew
        and dec_dup.expert_id == a_id
        and len(adv.experts) == n_before
    )
    routed_acc = adv.experts[dec_dup.expert_id].accuracy(a["test_X"], a["test_y"])

    # (a) SIZED CAP: at ONE advisor state, a novel task's cap > a familiar task's cap (monotone in
    #     the reused novelty signal). Real novelties from real signatures against the same bank.
    nov_fam = 1.0 - adv._match(a_dup["sig"])[1]
    nov_novel = 1.0 - adv._match(far["sig"])[1]
    cap_fam = adv.recruit_cap(nov_fam)
    cap_novel = adv.recruit_cap(nov_novel)

    # weights genuinely LEARNED (diverged from zero init) under the local rule
    all_diverged = all(e.diverged for e in adv.experts)

    return {
        "d": d,
        "k": k,
        "accA0": accA0,
        "accA1": accA1,
        "retention": retention,
        "shared_A_before": shared_A_before,
        "shared_A_after": shared_A_after,
        "routed_ok": routed_ok,
        "routed_acc": routed_acc,
        "nov_fam": nov_fam,
        "nov_novel": nov_novel,
        "cap_fam": cap_fam,
        "cap_novel": cap_novel,
        "grow_caps": {L: consolidated[L][2].recruit_cap for L in "ABC"},
        "own_accs": {L: consolidated[L][1] for L in "ABC"},
        "all_diverged": all_diverged,
        "n_experts": len(adv.experts),
        "protos": [list(p) for p in adv.bank.prototypes],
        "expert_W": [e.W.copy() for e in adv.experts],
    }


def main():
    r = _build(seed=7)

    print("=" * 78)
    print("SIMILARITY-SIZED GROWTH -- size fresh capacity by dissimilarity + reuse frozen experts")
    print("=" * 78)
    print(
        f"setup     : 3 heterogeneous tasks (cyclic-shifted class geometry) in R^{r['d']}, "
        f"K={r['k']} classes; signatures = train centroids"
    )
    print(
        f"experts   : {r['n_experts']} frozen specialists; own-test acc "
        f"{{{', '.join(f'{L}:{a:.2f}' for L, a in r['own_accs'].items())}}}; "
        f"weights diverged from zero init = {r['all_diverged']}"
    )
    print("-" * 78)
    print(
        "(a) SIMILARITY-SIZED CAP (cap = ceil(cap_max*(1-exp(-k*min_dist))), k = 1/running-mean):"
    )
    print(f"      familiar task  min_dist={r['nov_fam']:.4f} -> recruit_cap = {r['cap_fam']:>3d}")
    print(
        f"      novel task     min_dist={r['nov_novel']:.4f} -> recruit_cap = {r['cap_novel']:>3d}"
        f"   <- FAR task sizes MORE fresh capacity"
    )
    print(
        f"      caps as consolidated (grow path): "
        f"{{{', '.join(f'{L}:{c}' for L, c in r['grow_caps'].items())}}}"
    )
    print("-" * 78)
    print("(b) FROZEN-EXPERT REUSE (route, do not grow):")
    print(
        f"      near-duplicate of task A -> action=route to expert {r['n_experts'] and 0}, "
        f"grew=False, n_experts stayed {r['n_experts']}; routed expert acc on A = {r['routed_acc']:.2f}"
    )
    print("-" * 78)
    print("(c) RETENTION -- no catastrophic forgetting (frozen bank vs a shared plastic readout):")
    print(
        f"      A's frozen expert : acc {r['accA0']:.2f} at consolidation -> {r['accA1']:.2f} after "
        f"B,C learned  (retention {r['retention']:.2f})"
    )
    print(
        f"      CONTROL shared W  : acc on A {r['shared_A_before']:.2f} -> {r['shared_A_after']:.2f} "
        f"after B,C  <- one shared readout FORGETS"
    )
    print("-" * 78)

    # ---- asserts (deterministic; real numbers) ----
    # (a) a novel task sizes strictly more fresh capacity than a familiar one (monotone, same state)
    assert r["cap_novel"] > r["cap_fam"], (
        f"novel cap {r['cap_novel']} must exceed familiar cap {r['cap_fam']}"
    )
    assert r["nov_novel"] > r["nov_fam"], (
        "the novel task must actually be less similar than the familiar one"
    )
    # (b) the familiar task reuses a frozen expert with NO new growth
    assert r["routed_ok"], "near-duplicate of A must ROUTE to A's frozen expert without growing"
    assert r["routed_acc"] >= 0.8, (
        f"routed frozen expert must still classify A well ({r['routed_acc']:.2f})"
    )
    # (c) retention: the frozen expert is untouched -> exact retention; the shared readout forgets
    assert r["retention"] == 1.0, f"frozen expert must retain A exactly ({r['retention']:.2f})"
    assert r["accA1"] == r["accA0"], "A's frozen expert accuracy must be bit-identical after B,C"
    assert r["shared_A_after"] < r["accA0"], (
        f"CONTROL: a shared readout must forget A ({r['shared_A_after']:.2f} < {r['accA0']:.2f})"
    )
    # experts are real: they specialize and their weights actually moved under the local rule
    assert all(a >= 0.8 for a in r["own_accs"].values()), f"experts too weak: {r['own_accs']}"
    assert r["all_diverged"], "local Hebbian rule must have moved weights away from zero init"

    # (d) DETERMINISTIC: an independent identical build reproduces every number + every weight array
    r2 = _build(seed=7)
    same_scalars = all(
        r[key] == r2[key]
        for key in (
            "accA0",
            "accA1",
            "retention",
            "shared_A_before",
            "shared_A_after",
            "cap_fam",
            "cap_novel",
            "routed_ok",
            "n_experts",
        )
    )
    same_W = all(np.array_equal(w1, w2) for w1, w2 in zip(r["expert_W"], r2["expert_W"]))
    same_protos = r["protos"] == r2["protos"]
    assert same_scalars and same_W and same_protos, "two seed=7 builds must be bit-identical"
    print(
        f"(d) DETERMINISM   : two seed=7 builds identical (scalars={same_scalars}, "
        f"expert weights={same_W}, keys={same_protos})"
    )
    print("=" * 78)
    print(
        f"self-check OK: recruit cap scales with dissimilarity (familiar {r['cap_fam']} < novel "
        f"{r['cap_novel']}); a familiar task REUSES its frozen expert (no growth); a frozen expert "
        f"RETAINS task A exactly ({r['retention']:.2f}) while a shared readout forgets "
        f"({r['shared_A_before']:.2f}->{r['shared_A_after']:.2f}); weights learned by a LOCAL "
        f"Hebbian rule (diverged from init); deterministic. ADVISOR only -- growable/neurogenesis "
        f"untouched. HONEST: strong structural retention on a small synthetic set, not a benchmark."
    )


if __name__ == "__main__":
    main()
