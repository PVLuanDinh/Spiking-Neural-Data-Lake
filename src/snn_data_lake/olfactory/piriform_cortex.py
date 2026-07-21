"""
Piriform cortex -- sparse associative expansion with pattern COMPLETION (olfactory memory).

Piriform (primary olfactory) cortex reads the bulb's odor code into a SPARSE, distributed,
non-topographic representation and stores odors in its recurrent excitatory network as
auto-associative attractors. That gives it two functions this module makes explicit:
  * COMPLETION  -- a degraded / partial cue (only a subset of an odor's features present)
                   re-activates the FULL stored odor code via the recurrent weights.
  * NOVELTY GATE -- a cue that resembles NO stored odor is rejected, so completion is genuine
                   memory recall, not an always-on attractor that snaps anything to a pattern.

This is the olfactory analogue of hippocampal CA3 completion, but it is implemented STANDALONE
here (it does NOT import the hippocampus) over a piriform-style sparse code.

Mechanism (deterministic):
  1. EXPANSION + k-WTA encode: each of m cortical units sums a fixed random subset ("tap set") of
     receptor features; the top-k units fire -> a sparse binary code (frozenset of unit ids).
     Random expansion makes codes for dissimilar odors near-orthogonal (Dasgupta et al. 2017).
  2. STORE (Hebbian auto-association): units co-active in a stored code wire together, W[i][j] += 1.
  3. RECALL: encode the cue. A familiarity/NOVELTY gate first checks that the cue code already
     overlaps some stored code above `gate` -- an unfamiliar cue is REJECTED here (novelty signal).
     A familiar cue is then SETTLED: repeatedly drive units by the recurrent field of the active
     set and re-apply k-WTA until the state stops changing -- it falls into the stored attractor,
     filling in the missing units (completion). Deterministic; ties break to the lower unit id.

Fair control (un-riggable -- ONE fixed gate, applied identically to every cue):
  (a) LEARNED  : a partial cue (mask_frac of features zeroed) of a stored odor must COMPLETE to
     that odor -- overlap with the original code rises above the accept level, and completion
     recovers MORE of the original code than the raw partial cue did (post_overlap > pre_overlap).
  (b) NOVEL    : a fresh odor never stored must be REJECTED by the gate -- no false completion
     (the same machinery does not manufacture a stored memory from an unfamiliar cue).
  (c) DISCRIMINATION : each stored odor's partial cue completes to its OWN attractor, never a
     different stored odor -- no cross-talk between the stored memories.

Honest framing: a reduced attractor model of olfactory associative memory on synthetic odors. It
models a store/complete/reject FUNCTION, not smell, recognition, or cognition. Ceiling: a fixed-
sparsity k-WTA Hopfield with a familiarity gate -- no spike timing, no plasticity schedule, no
temporal sequence; the odors are SYNTHETIC. Additive / opt-in: a clean-up + novelty stage you
route noisy odor codes through. Credit: Haberly 2001, Wilson & Sullivan 2011 (piriform associative
cortex); Hopfield 1982 (auto-association); Dasgupta, Stevens & Navlakha 2017 (expansion coding).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.olfactory.piriform_cortex
"""

import random

N_INPUT = 48  # receptor / bulb-output features feeding the cortex
N_UNITS = 200  # piriform expansion units
K_ACTIVE = 12  # winners kept by k-WTA (sparse code size)
TAPS = 8  # receptor features each unit sums
GATE = 0.30  # familiarity threshold: min cue-vs-stored overlap to attempt recall
ACCEPT = 0.60  # min completed overlap to count as a genuine recall
ACTIVE_FEATURES = 18  # receptors an odor activates (sparse odor input)


class PiriformCortex:
    """Sparse expansion code + Hebbian recurrent attractors: store, complete, and reject odors."""

    def __init__(
        self,
        n_input=N_INPUT,
        n_units=N_UNITS,
        k_active=K_ACTIVE,
        taps=TAPS,
        gate=GATE,
        accept=ACCEPT,
        seed=0,
    ):
        self.n_input = n_input
        self.m = n_units
        self.k = k_active
        self.gate = gate
        self.accept = accept
        rng = random.Random(seed)  # all RNG seeded -> deterministic
        self.proj = [tuple(rng.sample(range(n_input), taps)) for _ in range(n_units)]
        self.W = [[0] * n_units for _ in range(n_units)]  # recurrent Hebbian weights
        self.codes = []  # stored sparse codes (frozensets)
        self.origs = []  # stored original [0,1] vectors

    def _encode(self, x):
        """Feature vector -> sparse binary code (top-k expansion units). Ties break to lower id."""
        act = [sum(x[i] for i in taps) for taps in self.proj]
        order = sorted(range(self.m), key=lambda u: (act[u], -u), reverse=True)
        return frozenset(order[: self.k])

    def _match(self, s):
        """Best (overlap_fraction, stored_id) between code s and any stored code (0.0, -1 if none)."""
        best, bid = 0.0, -1
        for pid, code in enumerate(self.codes):
            ov = len(s & code) / self.k
            if ov > best:
                best, bid = ov, pid
        return best, bid

    def store(self, odor):
        """Encode the odor and Hebbian-wire its co-active units into a recurrent attractor."""
        code = self._encode(odor)
        pid = len(self.codes)
        self.codes.append(code)
        self.origs.append(list(odor))
        for i in code:
            row = self.W[i]
            for j in code:
                if i != j:
                    row[j] += 1
        return pid

    def _settle(self, s, max_iter=8):
        """Recurrent attractor dynamics: drive units by the field of the active set, re-apply k-WTA,
        repeat to a fixed point. Fills in the missing units of a partial cue (pattern completion)."""
        for _ in range(max_iter):
            field = [0] * self.m
            for i in s:
                row = self.W[i]
                for j in range(self.m):
                    field[j] += row[j]
            order = sorted(range(self.m), key=lambda u: (field[u], -u), reverse=True)
            s_new = frozenset(order[: self.k])
            if s_new == s:
                break
            s = s_new
        return s

    def recall(self, cue):
        """Recall an odor from a (partial) cue. Returns (stored_id, completed_overlap, cue_overlap):
        stored_id = -1 if the NOVELTY gate rejects the cue as unfamiliar (no false completion)."""
        s0 = self._encode(cue)
        pre_best, _ = self._match(s0)
        if pre_best < self.gate:  # familiarity / novelty gate
            return -1, 0.0, pre_best
        settled = self._settle(s0)
        post_best, post_id = self._match(settled)
        return post_id, post_best, pre_best


def _sparse_odor(rng, n_input, active):
    """A sparse odor: `active` random receptors driven in [0.5, 1.0], the rest silent."""
    x = [0.0] * n_input
    for g in rng.sample(range(n_input), active):
        x[g] = rng.uniform(0.5, 1.0)
    return x


def _mask(x, rng, frac):
    """Degrade a cue to a partial subset: drop each present feature with probability `frac`
    (a silent feature stays silent, so this keeps only a random subset of the odor's features)."""
    return [0.0 if rng.random() < frac else v for v in x]


def main():
    K = 5
    MASK = 0.25
    ctx = PiriformCortex(seed=0)

    # 5 well-separated sparse odors; store each as a recurrent attractor.
    orng = random.Random(3)
    odors = [_sparse_odor(orng, ctx.n_input, ACTIVE_FEATURES) for _ in range(K)]
    ids = [ctx.store(v) for v in odors]

    # (a) LEARNED: partial cues of stored odors must complete to their own attractor.
    crng = random.Random(42)
    T = 300
    comp_hits = 0
    pre_sum = post_sum = 0.0
    cross = 0
    for _ in range(T):
        pid = crng.randrange(K)
        cue = _mask(odors[pid], crng, MASK)
        rid, post_ov, pre_ov = ctx.recall(cue)
        pre_sum += pre_ov
        post_sum += post_ov
        comp_hits += rid == ids[pid] and post_ov >= ctx.accept
        cross += rid != -1 and rid != ids[pid]  # completed to a DIFFERENT stored odor
    comp_acc = comp_hits / T
    pre_overlap = pre_sum / T
    post_overlap = post_sum / T
    cross_rate = cross / T

    # (b) NOVEL: fresh odors never stored must be rejected by the gate (no false completion).
    nrng = random.Random(99)
    rejects = 0
    nov_pre = 0.0
    for _ in range(T):
        novel = _sparse_odor(nrng, ctx.n_input, ACTIVE_FEATURES)  # same statistics as stored odors
        rid, _, pre_ov = ctx.recall(novel)
        nov_pre += pre_ov
        rejects += rid == -1
    reject_rate = rejects / T
    novel_pre = nov_pre / T

    print("=" * 66)
    print("PIRIFORM CORTEX -- sparse associative expansion + pattern completion")
    print("=" * 66)
    print(
        f"features = {ctx.n_input}, units = {ctx.m}, k-active = {ctx.k}, stored = {K}, mask = {MASK:.0%}"
    )
    print(f"gate (familiarity) = {ctx.gate:.2f}, accept (completed overlap) = {ctx.accept:.2f}")
    print("-" * 66)
    print(
        f"(a) LEARNED : partial-cue completion acc = {comp_acc:.3f}  "
        f"[overlap {pre_overlap:.3f} (cue) -> {post_overlap:.3f} (completed)]"
    )
    print(
        f"(b) NOVEL   : rejection rate = {reject_rate:.3f}  [mean novel cue overlap = {novel_pre:.3f} < gate]"
    )
    print(
        f"(c) DISCRIM : cross-completion rate = {cross_rate:.3f}  [cue never completes to a WRONG odor]"
    )
    print("=" * 66)

    # ---- self-check: deterministic asserts, ASCII-only ----
    assert comp_acc >= 0.90, (
        f"learned partial cues must complete to their odor (acc {comp_acc:.3f})"
    )
    assert post_overlap > pre_overlap + 0.03, (
        f"completion must recover MORE than the raw cue ({post_overlap:.3f} <= {pre_overlap:.3f})"
    )
    assert reject_rate >= 0.90, (
        f"novel cues must be rejected, not falsely completed (rate {reject_rate:.3f})"
    )
    assert novel_pre < ctx.gate, (
        f"novel cues must be unfamiliar (mean overlap {novel_pre:.3f} < gate {ctx.gate})"
    )
    assert cross_rate <= 0.02, f"stored odors must not cross-complete (cross rate {cross_rate:.3f})"

    d1 = PiriformCortex(seed=0)
    d2 = PiriformCortex(seed=0)
    for v in odors:
        d1.store(v)
        d2.store(v)
    drng = random.Random(7)
    det = True
    for _ in range(60):
        pid = drng.randrange(K)
        cue = _mask(odors[pid], drng, MASK)
        if d1.recall(cue) != d2.recall(cue):
            det = False
            break
    assert det and d1.codes == d2.codes, "same seed must give identical codes and recalls"

    print(
        f"self-check OK: learned completion {comp_acc:.3f} (overlap {pre_overlap:.3f} -> {post_overlap:.3f}), "
        f"novel reject {reject_rate:.3f}, cross-talk {cross_rate:.3f}, deterministic"
    )


if __name__ == "__main__":
    main()
