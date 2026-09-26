"""
Sleep replay prevents catastrophic forgetting in a spiking network (Golden et al. 2022).

Golden, Delanois, Sanda & Bazhenov, "Sleep prevents catastrophic forgetting in spiking
neural networks by forming a joint synaptic weight representation", PLOS Computational
Biology 18(11):e1010628 (2022), DOI 10.1371/journal.pcbi.1010628.

THE PROBLEM. A spiking net that learns task A then task B tends to learn B by OVERWRITING the
synapses A relied on -> catastrophic forgetting of A. Here A and B are stored as overlapping
cell assemblies in the RECURRENT weights of one spiking network, and each neuron's incoming
weight is homeostatically normalized to a fixed budget (synaptic scaling). Storing B pulls that
budget onto B's assemblies and starves A's shared neurons. The forgetting is severe: after B,
cueing a task-A pattern no longer completes to A -- it completes to the B assembly that
overwrote it, so A recall drops BELOW chance (active interference, not mere randomness).

THE MECHANISM (this module). Between/after tasks we insert a "sleep" phase: the network is
driven only by NOISE (no task input, no labels, no teaching) with STDP still ON. Because the
memories are stored as attractors, noise reactivation is not random -- a noise fluctuation is
amplified by the recurrent weights into whichever stored assembly it most resembles, and (with
spike-frequency reset between events) sleep spontaneously REPLAYS A's and B's assemblies in
turn. STDP with the same covariance rule used for storage then re-consolidates each replayed
assembly, and the homeostatic budget lets A and B COEXIST -- a JOINT weight representation.
Sleep before B entrenches A so B cannot fully evict it; sleep after B replays both and rebinds
the degraded A assembly. Net effect: A recall is restored WELL ABOVE the no-sleep level (from
far-below-chance toward/above chance) while B stays learned.

WHY THIS IS AN HONEST TEST, NOT A RIG (the whole point):
  * FAIR CONTROL. Identical architecture, seed, initial weights, task patterns, and (fully
    deterministic) task-storage in every condition. The ONLY difference between the no-sleep and
    sleep conditions is whether the sleep phase runs. Sleep sees NO task-A data -- only noise --
    so any A-retention is genuine replay/consolidation, never re-training on A.
  * ANCHOR (a) REPLAY IS REAL. During sleep we record the spontaneous network states and measure
    their overlap with the STORED assemblies. On the trained net that overlap is far above the
    overlap the SAME states have with random assemblies, and far above what an UNTRAINED
    (random-weight) net produces under identical noise: the spontaneous activity is genuinely
    replaying the stored memories, not generic noise. (Reported as `replay_stored` vs
    `replay_random` vs `replay_untrained`.)
  * ANCHOR (b) IT IS THE STDP, NOT THE NOISE. A "sleep with STDP OFF" condition runs the exact
    same noise episodes with plasticity disabled. Because it changes no weights, A recall after B
    is identical to no-sleep -- proving the benefit is the consolidation (weight change), not the
    stimulation. (Reported as `a_stdpoff`, asserted no better than `a_nosleep`.)

FRAGILITY / SPECIFICITY (disclosed honestly). This is a small, hand-tuned demonstration: 24 LIF
neurons, K=6 per assembly, two 2-way tasks with deliberately overlapping assemblies, homeostatic
budget normalization as the forgetting substrate, spontaneous replay via noise + per-event reset.
What is ROBUST (verified across seeds 0-11): no-sleep always forgets A far below chance (~0.12);
sleep always beats no-sleep on A by a wide margin (the sleep-minus-no-sleep gap was >= +0.33 on
every one of the 12 seeds); the replay anchor and the STDP-off control always hold. What is
SEED-DEPENDENT (fragile in magnitude): the ABSOLUTE restored A accuracy (0.46-0.79 across seeds,
i.e. around chance -- above chance on some seeds, at chance on others; recall is near-binary so it
does not land at a smooth 0.8), and how much sleep costs B (B_sleep ranges 0.50-1.00 -- sleep
occasionally rebalances toward A at B's expense, down to chance). The effect also needs a HARD
partial cue (CUE_FRAC ~ 1/3) to expose the forgetting, and depends on the assembly overlap, the
budget, and the STDP rate. It demonstrates the Golden et al. mechanism on a toy task; it is NOT a
general continual-learning solver, and A is only PARTIALLY restored, not returned to its original
1.00. The deterministic self-check below runs on seed 0 (representative: gap +0.42, sleep 0.54 >
chance, B_sleep 0.88) and asserts only the seed-robust claims (the gap, forgetting, the anchors),
not the seed-dependent absolute levels.

Seeded per-instance (`random` + `copy`), ASCII output, deterministic
(no global-RNG side effects, no wall-clock). Reuses the core spiking primitive
snn_data_lake.cortex.lif_neuron.LIFNeuron for every neuron. Additive core primitive (a reusable
spiking attractor associative memory with sleep consolidation); touches no safety-envelope file.

  Run: python -m snn_data_lake.hippocampus.sleep_replay        # (also run by run_selfchecks.py)
"""

import copy
import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "AttractorNet.store": (
        "LOCAL",
        "stores an assembly by clamping it ON and applying covariance Hebbian plasticity to co-active "
        "pairs. Pairwise and local",
    ),
    "AttractorNet.sleep": (
        "LOCAL",
        "noise-driven spontaneous replay: seed a few neurons, run the attractor, let the same Hebbian "
        "rule consolidate. Offline application of a local rule",
    ),
    "AttractorNet._normalize": (
        "LOCAL",
        "synaptic scaling: rescales each neuron's incoming weights to a fixed budget. Homeostatic "
        "plasticity computed from that neuron's own weights",
    ),
}

# ---- network ----
N = 24  # LIF neurons (one recurrent layer)
K = 6  # neurons per stored assembly (pattern size)

# two sequential 2-way tasks; each pattern is a cell assembly (set of neuron indices).
# The assemblies OVERLAP across tasks -> storing B competes for A's shared neurons (interference).
P_A0 = frozenset(range(0, 6))  # task A, class 0
P_A1 = frozenset(range(6, 12))  # task A, class 1
P_B0 = frozenset({1, 2, 3, 7, 8, 9})  # task B, class 0 (overlaps A0 and A1)
P_B1 = frozenset({4, 5, 10, 11, 12, 13})  # task B, class 1

# ---- dynamics (recurrent LIF attractor with global inhibition) ----
V_TH = 1.0  # LIF threshold
TAU_M = 8.0  # LIF membrane time constant
G_INH = 0.10  # global inhibition per active neuron -> k-WTA-like competition (bounds activity)
EXT_CLAMP = 2.5  # external current that clamps a cued/taught neuron ON
BUDGET = 2.5  # homeostatic L1 budget of each neuron's incoming weights (synaptic scaling)
A_STORE = 0.30  # Hebbian potentiation increment during (clamped) storage

# ---- storage (supervised clamp) ----
STORE_REPS_A = 5  # storage repetitions per task-A assembly
STORE_REPS_B = 4  # storage repetitions per task-B assembly (enough to overwrite A)

# ---- recall / test ----
T_CUE = 5  # steps the partial cue is clamped
T_FREE = 15  # free-run steps (the attractor completes the pattern)
CUE_FRAC = 0.34  # fraction of an assembly given as the cue (HARD partial cue -> exposes forgetting)
TEST_TRIALS = 12  # cues per pattern (different random halves) for an accuracy estimate

# ---- sleep (noise-driven spontaneous replay + covariance STDP) ----
SLEEP_EPISODES = 500  # replay events per sleep phase
EP_STEPS = 12  # steps per replay event (reset between events -> different basins visited)
NOISE_AMP = 0.9  # amplitude of the per-neuron noise current
NOISE_P = 0.20  # per-step probability each neuron gets a noise kick
KICK_SEEDS = 2  # neurons given an onset kick to seed each event
LR_SLEEP = 0.10  # covariance-STDP rate during sleep
ANCHOR_EPISODES = 200  # sleep episodes recorded for the replay-reality anchor

PATTERNS = [P_A0, P_A1, P_B0, P_B1]


def _neurons():
    return [LIFNeuron(tau_m=TAU_M, v_threshold=V_TH, refractory=0) for _ in range(N)]


def _overlap(state, pattern):
    """Fraction of the active neurons that fall inside `pattern` (0..1)."""
    tot = sum(state)
    return sum(state[i] for i in pattern) / tot if tot else 0.0


class AttractorNet:
    """One recurrent layer of N LIF neurons with plastic, homeostatically-normalized weights.
    Assemblies are stored as attractors; a partial cue completes to the nearest attractor; noise
    drives spontaneous replay. Deterministic: every stochastic draw comes from a caller-supplied
    random.Random (no global RNG, no wall-clock)."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.W = [[0.0 if i == k else rng.uniform(0.02, 0.06) for k in range(N)] for i in range(N)]
        self._normalize()

    def clone(self):
        c = AttractorNet.__new__(AttractorNet)
        c.W = copy.deepcopy(self.W)
        return c

    def _normalize(self):
        """Synaptic scaling: rescale each neuron's incoming weights to the fixed BUDGET. This is
        the forgetting substrate -- growing one assembly's weights necessarily shrinks others'."""
        for i in range(N):
            s = sum(self.W[i])
            if s > 0:
                f = BUDGET / s
                for k in range(N):
                    self.W[i][k] *= f

    def downscale_prune(self, eps=1e-3):
        """SHY (Synaptic Homeostasis Hypothesis, Tononi-Cirelli): the distinctive offline limb -- a net
        synaptic DOWNSCALE + weak-synapse HARD-PRUNE. Zero every WEAK synapse (0 < W[i][k] < eps), then
        renormalize so the freed budget redistributes to the survivors. The hard-prune is the load-
        bearing, energy-saving effect: removed edges cost nothing to evaluate and stay gone (unlike the
        multiplicative rescale, which _normalize refills). Returns a telemetry readout. Deterministic."""
        n_zeroed, removed = 0, 0.0
        for i in range(N):
            wi = self.W[i]
            for k in range(N):
                w = wi[k]
                if 0.0 < w < eps:
                    wi[k] = 0.0
                    n_zeroed += 1
                    removed += w
        self._normalize()  # redistribute the freed budget to survivors
        total = N * (N - 1)  # off-diagonal synapses (self-weights are 0)
        return {
            "pruned": n_zeroed,
            "total": total,
            "weight_removed": removed,
            "frac_pruned": n_zeroed / total if total else 0.0,
        }

    def _recurrent_step(self, neurons, z_prev, ext):
        """One synchronous LIF update: current = recurrent excitation - global inhibition + ext."""
        active_prev = sum(z_prev)
        z = [0] * N
        for i in range(N):
            rec = 0.0
            wi = self.W[i]
            for k in range(N):
                if z_prev[k]:
                    rec += wi[k]
            z[i] = neurons[i].step(rec - G_INH * active_prev + ext[i])
        return z

    def store(self, pattern, reps):
        """Store an assembly by clamping it ON and applying covariance Hebbian plasticity:
        co-active pair -> potentiate; active-with-silent -> depress; then renormalize. Supervised
        (the pattern is the label). Fully deterministic (no RNG)."""
        on = [1 if i in pattern else 0 for i in range(N)]
        for _ in range(reps):
            for i in range(N):
                if not on[i]:
                    continue
                for k in range(N):
                    if i != k and on[k]:
                        self.W[i][k] += A_STORE
            self._normalize()

    def recall(self, cue):
        """Clamp a partial cue for T_CUE steps, then free-run; return the final firing state."""
        neurons = _neurons()
        z = [0] * N
        for t in range(T_CUE + T_FREE):
            ext = [EXT_CLAMP if (t < T_CUE and i in cue) else 0.0 for i in range(N)]
            z = self._recurrent_step(neurons, z, ext)
        return z

    def classify(self, pattern, patterns, seed):
        """Cue a random CUE_FRAC subset of `pattern`; predict the stored pattern the completed
        state most resembles (a preference score = mean-inside minus mean-outside firing)."""
        rng = random.Random(seed)
        ncue = max(1, int(K * CUE_FRAC))
        cue = set(rng.sample(sorted(pattern), ncue))
        st = self.recall(cue)
        scores = []
        for q in patterns:
            inside = sum(st[i] for i in q) / len(q)
            outside = sum(st[i] for i in range(N) if i not in q) / (N - len(q))
            scores.append(inside - outside)
        return scores.index(max(scores))

    def _trials(self, patterns, seed_base):
        """The (class index, cue set) trial list, in classify()'s own draw order.

        ONE definition, shared by the batched accuracy() below and by the equivalence check in
        main(). Extracted 2026-08-04 so the two cannot silently disagree about WHICH cues they are
        comparing -- a check that draws its own cues would pass whatever the fused path did."""
        ncue = max(1, int(K * CUE_FRAC))
        trials = []
        for ci, p in enumerate(patterns):
            for tr in range(TEST_TRIALS):
                rng = random.Random(seed_base + ci * 97 + tr)
                trials.append((ci, set(rng.sample(sorted(p), ncue))))
        return trials

    def _batch_states(self, trials):
        """Run every trial's cue -> free-run in ONE array pass; return the (B, N) final firing state.

        This is the fused twin of `recall()`, extracted from accuracy() so the equivalence between
        them is CHECKABLE rather than merely asserted in prose (see main()). Every statement is the
        transcription it always was; nothing about the arithmetic or its order changed."""
        import numpy as np

        B = len(trials)
        W = np.array(self.W, dtype=np.float64)  # W[i][k] = weight into i from k
        cue = np.zeros((B, N), dtype=bool)
        for b, (_, c) in enumerate(trials):
            for i in c:
                cue[b, i] = True

        v = np.zeros((B, N), dtype=np.float64)
        z = np.zeros((B, N), dtype=np.int64)
        rec = np.empty((B, N), dtype=np.float64)
        zero = np.zeros((B, N), dtype=np.float64)
        decay = math.exp(-1.0 / TAU_M)  # == LIFNeuron.__init__; refractory is 0 here, so that
        # branch of step() is dead and the update is v*decay + current.

        for t in range(T_CUE + T_FREE):
            ext = (cue * EXT_CLAMP) if t < T_CUE else zero
            active_prev = z.sum(axis=1)  # integer sum -> exact
            rec.fill(0.0)
            for k in range(N):  # k-outer: same order as `for k: if z_prev[k]: rec += wi[k]`
                np.add(rec, W[:, k][None, :], out=rec, where=z[:, k : k + 1].astype(bool))
            cur = rec - G_INH * active_prev[:, None] + ext
            v = 0.0 + (v - 0.0) * decay + cur  # literal transcription of LIFNeuron.step
            sp = v >= V_TH
            v = np.where(sp, 0.0, v)
            z = sp.astype(np.int64)
        return z

    def accuracy(self, patterns, seed_base):
        """Recall accuracy over TEST_TRIALS partial cues per pattern -- all trials in ONE array pass.

        WHY THIS IS BATCHED AND recall()/classify() ARE NOT
            The trials are mutually independent (each gets a fresh neuron set and its own cue), so they
            are the only genuine batch dimension in the brain. Measured: this is 720 of the 756
            LIF steps on a sleep tick, ~49% of it. recall() and classify() stay scalar -- they are the
            readable reference and the B=1 public API, and nothing calls them in a hot loop.

        EQUIVALENCE WITH THE SCALAR PATH IS NOW CHECKED, NOT ONLY CLAIMED
            This docstring used to assert "BIT-EXACTNESS IS THE CONTRACT" and cite a one-time
            elementwise verification over 432 trials -- with no assertion anywhere in the tree
            comparing the fused branch against the scalar one. That is the shape P8's incident table
            names for `access_consciousness._coherence`: a bit-exactness claim resting on a prose
            argument, in a module the sleep tick depends on. main() now compares
            `_batch_states(...)` against `recall()` CELL BY CELL and records the input that makes it
            fail.

            Three things make the equivalence hold, and each will silently break it if edited
            carelessly:

            1. The cue draws happen in the same order, with the same per-trial random.Random(seed).
            2. The accumulation is k-OUTER, not `Z @ W.T`. It adds the same values in the same
               ascending-k order the scalar loop does, so it is exact BY CONSTRUCTION. (Measured
               aside: matmul happens to agree bit-for-bit here too, because the mean active count is
               ~2 and there is nothing for reassociation to reorder -- but that is a property of the
               current sparsity, not of the code. This form does not depend on it.)
            3. The scoring stays in Python, including `scores.index(max(scores))`, so ties break
               toward the first pattern exactly as classify() does.

            WHAT THE CHECK CERTIFIES, AND WHAT IT DOES NOT -- measured 2026-08-04, because a check
            whose resolution is unstated is the next thing to go stale. At STATE resolution it goes
            red on a transposed cue-seed formula (92 of 1,152 cells differ) and on a halved TAU_M
            (45 of 1,152), so it genuinely certifies points 1 and 3 and the transcription of the
            dynamics. It does NOT resolve a last bit: a 1e-9 relative perturbation of the decay, and
            swapping the k-outer accumulation for `Z @ W.T`, both leave the states IDENTICAL. And
            the same comparison at ACCURACY resolution is worthless -- the returned float is
            unmoved by the transposed cue order AND by the halved TAU_M, so an
            `accuracy() == scalar_accuracy()` assert would have had no reachable red at all and was
            deliberately NOT written (P8/F11: a criterion that cannot go red is deleted, not
            shipped). Point 2 therefore remains a structural argument, and is labelled as one.
        """
        trials = self._trials(patterns, seed_base)
        z = self._batch_states(trials)
        correct = 0
        for b, (ci, _) in enumerate(trials):
            st = z[b]
            scores = []
            for q in patterns:
                inside = sum(int(st[i]) for i in q) / len(q)
                outside = sum(int(st[i]) for i in range(N) if i not in q) / (N - len(q))
                scores.append(inside - outside)
            correct += 1 if scores.index(max(scores)) == ci else 0
        return correct / len(trials)

    def sleep(self, seed, episodes=SLEEP_EPISODES, learn=True, record=None):
        """Noise-driven spontaneous replay. Each episode: reset, seed a couple of neurons, run
        EP_STEPS with per-neuron noise; the recurrent weights amplify the fluctuation into a
        stored assembly. If learn, apply the SAME covariance STDP used for storage to the replayed
        state (potentiate co-active, depress active-with-silent) + renormalize -> consolidation.
        No task input, no labels. Deterministic given `seed`."""
        rng = random.Random(seed)
        neurons = _neurons()
        for _ in range(episodes):
            kick = set(rng.sample(range(N), KICK_SEEDS))
            z = [0] * N
            for t in range(EP_STEPS):
                ext = [
                    NOISE_AMP if (rng.random() < NOISE_P or (t < 2 and i in kick)) else 0.0
                    for i in range(N)
                ]
                z = self._recurrent_step(neurons, z, ext)
            if record is not None:
                record.append(z[:])
            if learn and K - 2 <= sum(z) <= K + 2:  # only consolidate clean, full-ish replays
                active = set(i for i in range(N) if z[i])
                for i in active:
                    for k in range(N):
                        if i == k:
                            continue
                        if k in active:
                            self.W[i][k] += LR_SLEEP
                        else:
                            self.W[i][k] = max(0.0, self.W[i][k] - LR_SLEEP)
                self._normalize()


def _replay_index(net, seed, ref_patterns):
    """Anchor (a): run a no-learning sleep, return the mean (over full replay events) of the best
    overlap between each spontaneous state and the reference assemblies. High => the spontaneous
    activity is replaying those assemblies."""
    rec = []
    net.clone().sleep(seed, episodes=ANCHOR_EPISODES, learn=False, record=rec)
    full = [s for s in rec if sum(s) >= K - 2]
    if not full:
        return 0.0
    return sum(max(_overlap(s, p) for p in ref_patterns) for s in full) / len(full)


def run(seed):
    """One full deterministic experiment. Returns every measured number so main() can print a
    table, assert the a-priori self-checks, and confirm a repeat run is byte-identical."""
    A = [P_A0, P_A1]
    B = [P_B0, P_B1]

    # ---- shared prefix: store task A (identical for all conditions) ----
    base = AttractorNet(seed)
    for p in A:
        base.store(p, STORE_REPS_A)
    a_after_a = base.accuracy(A, 100)

    # ---- anchor (a): replay is real ----
    # trained net: spontaneous states vs the STORED assemblies, vs RANDOM assemblies of equal size.
    rng_r = random.Random(seed + 555)
    rand_pats = [set(rng_r.sample(range(N), K)) for _ in range(len(PATTERNS))]
    replay_stored = _replay_index(base, seed * 99 + 1, PATTERNS)
    replay_random = _replay_index(base, seed * 99 + 1, rand_pats)
    # untrained net (random weights, no task): same noise -> no stored-assembly structure.
    replay_untrained = _replay_index(AttractorNet(seed), seed * 99 + 1, PATTERNS)

    # ---- condition 1: NO-SLEEP control -- A -> B ----
    ns = base.clone()
    for p in B:
        ns.store(p, STORE_REPS_B)
    a_nosleep = ns.accuracy(A, 100)
    b_nosleep = ns.accuracy(B, 100)

    # ---- condition 2: SLEEP -- A -> sleep -> B -> sleep ----
    sl = base.clone()
    sl.sleep(seed * 7 + 1)
    for p in B:
        sl.store(p, STORE_REPS_B)
    sl.sleep(seed * 7 + 2)
    a_sleep = sl.accuracy(A, 100)
    b_sleep = sl.accuracy(B, 100)

    # ---- condition 3: STDP-OFF SLEEP -- identical noise episodes, plasticity disabled ----
    off = base.clone()
    off.sleep(seed * 7 + 1, learn=False)
    for p in B:
        off.store(p, STORE_REPS_B)
    off.sleep(seed * 7 + 2, learn=False)
    a_stdpoff = off.accuracy(A, 100)
    b_stdpoff = off.accuracy(B, 100)

    return {
        "a_after_a": a_after_a,
        "a_nosleep": a_nosleep,
        "b_nosleep": b_nosleep,
        "a_sleep": a_sleep,
        "b_sleep": b_sleep,
        "a_stdpoff": a_stdpoff,
        "b_stdpoff": b_stdpoff,
        "replay_stored": replay_stored,
        "replay_random": replay_random,
        "replay_untrained": replay_untrained,
    }


# ---- a-priori thresholds (fixed BEFORE measuring; these are the SEED-ROBUST claims -- each holds
#      on all of seeds 0-11, cleared with margin, not tuned to barely pass. The seed-DEPENDENT
#      absolute levels (exact restored A, exact B cost) are reported but NOT asserted.) ----
CHANCE = 0.50
FORGET_MARGIN = 0.30  # no-sleep must forget A by at least this vs right-after-A (obs: ~0.88 drop)
RETAIN_MARGIN = (
    0.20  # sleep must beat no-sleep on A by at least this (obs: gap >= +0.33 every seed)
)
REPLAY_MIN = 0.60  # trained-net spontaneous states must overlap the stored assemblies this much
REPLAY_GAP = 0.10  # ... and clearly more than random / untrained baselines
OFF_TOL = 0.03  # STDP-off sleep must be no better than no-sleep (within this tolerance)


def main():
    print("=" * 74)
    print("SLEEP REPLAY prevents catastrophic forgetting (Golden et al. 2022, PLOS Comp Biol)")
    print("recurrent spiking attractor net; sleep = noise-driven replay + STDP consolidation")
    print("=" * 74)

    r = run(0)
    print(
        f"task            : store A (2 assemblies) then B (2 assemblies) in {N} LIF neurons, "
        f"K={K}/assembly; chance {CHANCE:.0%}"
    )
    print("")
    print("                          task A acc      task B acc")
    print(f"  after storing A       :   {r['a_after_a']:.2f}            -")
    print(
        f"  no-sleep  (A->B)      :   {r['a_nosleep']:.2f}          {r['b_nosleep']:.2f}"
        f"     <- A forgotten (BELOW chance: cue completes to B)"
    )
    print(
        f"  SLEEP (A->s->B->s)    :   {r['a_sleep']:.2f}          {r['b_sleep']:.2f}"
        f"     <- A partially restored (>> no-sleep), B still learned"
    )
    print(
        f"  STDP-OFF sleep        :   {r['a_stdpoff']:.2f}          {r['b_stdpoff']:.2f}"
        f"     <- no benefit (same noise, plasticity off)"
    )
    print("")
    print(
        f"  A retention gain (sleep - no-sleep) : {r['a_sleep'] - r['a_nosleep']:+.2f}  "
        f"(threshold >= +{RETAIN_MARGIN}; gap >= +0.33 on all of seeds 0-11)"
    )
    print(
        f"  replay overlap: stored {r['replay_stored']:.2f}  vs random {r['replay_random']:.2f}"
        f"  vs untrained {r['replay_untrained']:.2f}   (spontaneous activity replays the "
        f"STORED assemblies)"
    )
    print("")
    print("  NOTE: small hand-tuned demo (24 LIF neurons, K=6, overlapping assemblies, homeostatic")
    print("  budget as the forgetting substrate). ROBUST across seeds 0-11: no-sleep forgets A far")
    print("  below chance; sleep beats no-sleep on A by >= +0.33 every seed; anchors hold. SEED-")
    print(
        "  DEPENDENT (reported, not asserted): the ABSOLUTE restored A (0.46-0.79, around chance --"
    )
    print(
        "  A is only PARTIALLY restored, never back to 1.00) and B's cost (B_sleep 0.50-1.00). Needs"
    )
    print(
        "  the hard partial cue. Demonstrates the mechanism, not a general continual-learning solver."
    )

    # determinism: same seed -> identical result dict.
    r2 = run(0)
    identical = r == r2
    print(f"  determinism  : same seed -> identical result = {identical}")
    print("=" * 74)

    # ---- self-check (deterministic asserts; all thresholds fixed a priori above) ----
    assert r["a_nosleep"] <= r["a_after_a"] - FORGET_MARGIN, (
        f"no-sleep must show REAL forgetting: A {r['a_after_a']:.2f} -> {r['a_nosleep']:.2f} "
        f"(need drop >= {FORGET_MARGIN})"
    )
    assert r["a_sleep"] >= r["a_nosleep"] + RETAIN_MARGIN, (
        f"THE RESULT: sleep must retain A markedly better than no-sleep: {r['a_sleep']:.2f} "
        f"!>= {r['a_nosleep']:.2f} + {RETAIN_MARGIN}"
    )
    assert r["b_nosleep"] >= CHANCE and r["b_sleep"] >= CHANCE, (
        f"B must stay learned -- sleep must recover A WITHOUT destroying B (both >= chance "
        f"{CHANCE}): no-sleep {r['b_nosleep']:.2f}, sleep {r['b_sleep']:.2f}"
    )
    assert r["replay_stored"] > REPLAY_MIN, (
        f"anchor (a): trained-net spontaneous activity must replay the stored assemblies "
        f"({r['replay_stored']:.2f} !> {REPLAY_MIN})"
    )
    assert r["replay_stored"] > r["replay_random"] + REPLAY_GAP, (
        f"anchor (a): replay must match STORED assemblies more than random ones "
        f"({r['replay_stored']:.2f} vs {r['replay_random']:.2f})"
    )
    assert r["replay_stored"] > r["replay_untrained"] + REPLAY_GAP, (
        f"anchor (a): the replay structure comes from the LEARNING, not the noise "
        f"(trained {r['replay_stored']:.2f} vs untrained {r['replay_untrained']:.2f})"
    )
    assert r["a_stdpoff"] <= r["a_nosleep"] + OFF_TOL, (
        f"anchor (b): STDP-off sleep must give NO benefit ({r['a_stdpoff']:.2f} vs "
        f"no-sleep {r['a_nosleep']:.2f}); the benefit is consolidation, not the noise"
    )
    # ---- SHY downscale + hard-prune of the WEAKEST synapses (offline energy-saving limb) ----
    shy = AttractorNet(seed=0)
    asm = list(range(K))  # one assembly = the first K neurons
    shy.store(asm, reps=6)
    cue = asm[: max(1, K // 2)]  # partial cue for pattern completion

    def _ov(state):
        return sum(state[i] for i in asm) / len(asm)  # fraction of the assembly that fired

    pre = _ov(shy.recall(cue))
    weak_before = sum(1 for i in range(N) for k in range(N) if 0.0 < shy.W[i][k] < 5e-2)
    tel = shy.downscale_prune(eps=5e-2)
    post = _ov(shy.recall(cue))
    print(
        f"  SHY prune    : zeroed {tel['pruned']}/{tel['total']} weak synapses (frac "
        f"{tel['frac_pruned']:.1%}, weight removed {tel['weight_removed']:.3f}); assembly recall "
        f"{pre:.2f} -> {post:.2f}"
    )
    assert tel["pruned"] == weak_before and tel["pruned"] > 0, (
        "SHY must hard-prune EXACTLY the sub-eps synapses (and some weak ones exist to prune)"
    )
    assert post >= pre - 0.10, (
        "pruning the WEAKEST synapses must not degrade recall of a strong assembly (energy saved, memory kept)"
    )

    # ---- FUSED vs SCALAR: the equivalence accuracy() rests on, CHECKED (P8) --------------------
    # accuracy() runs every trial in one array pass; recall() is the readable scalar reference the
    # whole module is written against. Until 2026-08-04 the claim that they agree lived only in a
    # docstring. Compared here CELL BY CELL over 48 trials x 24 neurons, with the red recorded.
    # BOTH weight regimes, because the retired docstring's one-time verification covered "trained +
    # random weights x 6 seeds" and a check that only ever sees a trained net silently narrows that:
    # the k-outer-vs-matmul argument (property 2) is about SPARSITY, and an untrained net is the
    # denser regime. Measured 2026-08-04b over 12 seeds x {trained, random} = 24 configs, 27,648
    # cells: zero mismatches, so the two nets kept here are a representative pair, not a lucky one.
    eq = AttractorNet(0)
    for _p in (P_A0, P_A1):
        eq.store(_p, STORE_REPS_A)
    untrained = AttractorNet(1)  # random weights, never stored into -> the denser activity regime
    eq_trials = eq._trials(PATTERNS, 100)
    fused_rows = [[int(x) for x in row] for row in eq._batch_states(eq_trials)]
    scalar_rows = [eq.recall(c) for _ci, c in eq_trials]
    un_trials = untrained._trials(PATTERNS, 314)
    un_fused = [[int(x) for x in row] for row in untrained._batch_states(un_trials)]
    un_scalar = [untrained.recall(c) for _ci, c in un_trials]
    assert fused_rows == scalar_rows, (
        "the batched accuracy() path must land on the SAME final firing state as the scalar recall() "
        "it replaces, for every trial"
    )
    assert un_fused == un_scalar, (
        "fused == scalar must also hold on UNTRAINED (random) weights -- the denser regime, where the "
        "k-outer accumulation order is the thing actually being relied on"
    )
    # REACHABLE RED, constructed and run: transpose the per-trial cue-seed formula (property 1 in
    # accuracy()'s docstring) and the states diverge. Measured: 92 of 1,152 cells. This is what makes
    # the assert above a measurement rather than a restatement.
    _ncue = max(1, int(K * CUE_FRAC))
    red_trials = [
        (ci, set(random.Random(100 + tr * 97 + ci).sample(sorted(p), _ncue)))
        for ci, p in enumerate(PATTERNS)
        for tr in range(TEST_TRIALS)
    ]
    red_rows = [[int(x) for x in row] for row in eq._batch_states(red_trials)]
    n_red = sum(1 for a, b in zip(red_rows, scalar_rows) for x, y in zip(a, b) if x != y)
    assert n_red > 0, (
        "the fused-vs-scalar check must be able to FAIL: a wrong cue-seed order left the states "
        "identical, so the assert above certifies nothing"
    )
    print(
        f"  fused==scalar: {len(fused_rows) + len(un_fused)} trials x {N} cells identical between "
        f"the batched accuracy() path and scalar recall(), on TRAINED and on UNTRAINED weights; "
        f"red reachable ({n_red} cells differ under a transposed cue-seed order)"
    )

    assert identical, "determinism: same seed must reproduce the result byte-identically"
    print(
        f"self-check OK: no-sleep forgot A {r['a_after_a']:.2f}->{r['a_nosleep']:.2f} (below "
        f"chance); sleep recovered it to {r['a_sleep']:.2f} (+{r['a_sleep'] - r['a_nosleep']:.2f} "
        f"vs no-sleep, partial); B kept {r['b_sleep']:.2f}/{r['b_nosleep']:.2f}; replay overlaps stored "
        f"{r['replay_stored']:.2f}>random {r['replay_random']:.2f}/untrained {r['replay_untrained']:.2f}; "
        f"STDP-off no benefit; deterministic"
    )


if __name__ == "__main__":
    main()
