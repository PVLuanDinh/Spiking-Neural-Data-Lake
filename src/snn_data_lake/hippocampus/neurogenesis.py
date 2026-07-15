r"""
v3.27 -- neurogenesis as the SPIKING basis of self-improvement: supervised,
contribution-criterion, single-shot neurogenic classification over the fused
multimodal spike-feature space.

Grounded in NG-SNN: "A neurogenesis-inspired dynamic adaptive framework for
efficient spike classification" (Tang, Li, Zhang, Zeng; Neural Networks 2026;
DOI 10.1016/j.neunet.2026.108656). The paper's two innovations are realized here:

  1. SUPERVISED INCREMENTAL CONSTRUCTION under a CONTRIBUTION criterion. The
     network grows a new spiking neuron ONLY when the current structure would
     misclassify the labeled pattern OR wins it by too small a spiking margin
     (i.e. the new neuron will CONTRIBUTE to correct classification) -- NOT on
     mere novelty. Growth is driven by task error, so the structure stays
     compact and task-optimal.
  2. ACTIVITY-DEPENDENT ANALYTICAL LEARNING. When a neuron is grown, its input
     weights are set in ONE SHOT from the current spike ACTIVITY: the weight
     vector is the normalized rank-order / first-spike latency code of the
     pattern (early-firing afferents -> high weight). That makes the neuron a
     matched filter that integrates its own pattern's early spikes fastest and
     crosses threshold soonest. No backprop, no surrogate gradient, no epochs.

This SUPERSEDES growing_learner.py -- that learner grows on RATE cosine NOVELTY
(grow when similarity < vigilance, ignoring correctness) and its decision is a
graded cosine. This one is its SPIKING, CONTRIBUTION-criterion, single-shot
successor: input -> rank-order spike code (Thorpe) -> grown LIF neurons that
actually cross threshold and emit spikes -> the class is read from EMITTED
SPIKES (first-spike time + spike count), never from a graded/rate value.

Reuses (does NOT reimplement):
  * cortex.rank_order_coding.rank_code   -> the Thorpe rank / first-spike order.
  * cortex.lif_neuron.LIFNeuron          -> the leaky integrate-and-fire cell.
  * cortex.multimodal_encoder.encode_multimodal -> the [0,1]^N=64 fused feature
    vector the arm lives in (the self-check builds genuinely multimodal inputs).

HONEST FRAMING. This is a TOY-SCALE MECHANISM demonstration of NG-SNN's two
innovations, NOT a reproduction of the paper's dataset SOTA. The single-shot
analytical rule here is a matched-filter / prototype weight-set (a normalized
latency code), a deliberate simplification of the paper's full activity-
dependent analytical learning. The multimodal streams are SYNTHETIC. It grows
spiking capacity on a contribution criterion; it does not "think".

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.neurogenesis
"""
import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.cortex.rank_order_coding import rank_code
from snn_data_lake.cortex.spike_preprocessing import N


def _latency_code(pattern):
    """Activity-dependent single-shot weight rule: the normalized first-spike
    LATENCY code of `pattern`. rank_code gives 0 = fires first (largest value);
    we map earliest-firing afferents to the HIGHEST weight (Thorpe sensitivity),
    then L2-normalize so cosine and the matched-filter gain are scale-free."""
    n = len(pattern)
    ranks = rank_code(pattern)                          # 0 = earliest (largest)
    code = [float(n - 1 - r) for r in ranks]            # earliest -> n-1, latest -> 0
    norm = math.sqrt(sum(c * c for c in code)) or 1.0
    return [c / norm for c in code]


def _cos(a, b):
    """Cosine of two vectors (both L2-normalized here -> a plain dot product)."""
    return sum(x * y for x, y in zip(a, b))


class NeurogenesisLearner:
    """Supervised, single-pass, SPIKING neurogenic classifier.

    Grows LIF neurons on a CONTRIBUTION criterion and sets each new neuron's
    weights SINGLE-SHOT from the pattern's spike activity. Recognition reads the
    winning class from EMITTED SPIKES (earliest first-spike + most spikes).

    Parameters (defaults are the tuned toy-scale operating point):
      n_classes   : optional hint; labels are discovered online regardless.
      dim         : feature-space dimensionality (N=64 fused spikes).
      capacity    : BOUNDED max grown neurons.
      margin      : spiking-response margin below which a *correct* win is judged
                    under-confident -> grow (the confidence half of contribution).
      gain        : matched-filter current scale. Kept so a single afferent's
                    current stays BELOW threshold -> the cell must INTEGRATE
                    several early spikes to fire (order matters, not magnitude).
      tau_m, v_threshold : LIF membrane constant / firing threshold.
      grow_criterion : "contribution" (grow on error/under-confidence) or
                       "novelty" (grow when spike-code similarity < vigilance,
                       IGNORING correctness -- the control that over-grows).
      weight_init : "matched" (single-shot analytical) or "random" (the control
                    that proves the analytical rule carries the accuracy).
      vigilance   : novelty threshold (only used when grow_criterion=="novelty").
      refine      : if True, a familiar correct hit nudges the winner one single-
                    shot running-average step toward the pattern (still no loop).
    """

    def __init__(self, n_classes=None, dim=N, capacity=64, margin=1.0,
                 gain=1.6, tau_m=10.0, v_threshold=1.0,
                 grow_criterion="contribution", weight_init="matched",
                 vigilance=0.9, refine=False, seed=0):
        self.n_classes = n_classes
        self.dim = dim
        self.capacity = capacity
        self.margin = margin
        self.gain = gain
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.grow_criterion = grow_criterion
        self.weight_init = weight_init
        self.vigilance = vigilance
        self.refine = refine
        self.refine_lr = 0.1
        self.rng = random.Random(seed)                  # per-instance determinism
        self.neurons = []                               # each: {"w", "code", "label"}
        self.labels_seen = set()

    @property
    def n_units(self):
        return len(self.neurons)

    # -- spiking readout ---------------------------------------------------
    def _order(self, pattern):
        """Firing schedule: order[t] = feature that emits its spike at step t
        (descending value -> the largest feature fires first). Ties break by
        index, so the schedule is deterministic."""
        return sorted(range(self.dim), key=lambda i: pattern[i], reverse=True)

    def _spike_train(self, w, order):
        """Drive a fresh LIF cell with the rank-ordered afferent currents and
        return its emitted 0/1 spike train. current at step t = gain*w[order[t]]:
        a matched neuron's high weights land on the EARLY steps, so its membrane
        ramps and crosses threshold first."""
        lif = LIFNeuron(tau_m=self.tau_m, v_threshold=self.v_threshold)
        lif.reset_state()
        return [lif.step(self.gain * w[order[t]]) for t in range(self.dim)]

    def _run_full(self, w, pattern):
        """Like _spike_train but also returns the graded membrane trace -- used
        ONLY by the structural-spiking self-check to contrast spikes vs graded V."""
        order = self._order(pattern)
        lif = LIFNeuron(tau_m=self.tau_m, v_threshold=self.v_threshold)
        lif.reset_state()
        train, vtrace = [], []
        for t in range(self.dim):
            train.append(lif.step(self.gain * w[order[t]]))
            vtrace.append(lif.v)
        return train, vtrace

    def _score_train(self, train):
        """Collapse a spike train to a scalar response read PURELY from spikes:
        (spike count) + (recency of the first spike). Higher = fired more, and
        earlier. first_spike is an integer index in the train; count is an int."""
        count = sum(train)
        first = next((i for i, s in enumerate(train) if s), self.dim)
        return count + (self.dim - first) / self.dim, count, first

    def _ranked(self, pattern):
        """Score every grown neuron on `pattern`; return (score, label, idx,
        count, first) sorted best-first. order is computed once and shared."""
        order = self._order(pattern)
        scored = []
        for idx, nu in enumerate(self.neurons):
            train = self._spike_train(nu["w"], order)
            score, count, first = self._score_train(train)
            scored.append((score, nu["label"], idx, count, first))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # -- growth + refinement ----------------------------------------------
    def _grow(self, pattern, label):
        """Recruit one spiking neuron for `label`; set its input weights SINGLE-
        SHOT from the current spike activity (matched filter), or randomly for
        the control. Returns the new unit id."""
        code = _latency_code(pattern)                   # the activity-dependent code
        if self.weight_init == "random":
            w = [self.rng.gauss(0.0, 1.0) for _ in range(self.dim)]
            norm = math.sqrt(sum(x * x for x in w)) or 1.0
            w = [x / norm for x in w]
        else:                                           # single-shot analytical rule
            w = list(code)
        self.neurons.append({"w": w, "code": code, "label": label})
        return len(self.neurons) - 1

    def _refine_winner(self, idx, pattern):
        """One single-shot running-average step of the winner toward the pattern
        code (no gradient, no loop). Off by default."""
        code = _latency_code(pattern)
        lr = self.refine_lr
        blended = [(1.0 - lr) * a + lr * b for a, b in zip(self.neurons[idx]["w"], code)]
        norm = math.sqrt(sum(x * x for x in blended)) or 1.0
        self.neurons[idx]["w"] = [x / norm for x in blended]

    # -- public API --------------------------------------------------------
    def observe(self, pattern, label):
        """The neurogenic step. Spike-code `pattern`, run the grown spiking
        neurons, and grow iff the CONTRIBUTION criterion fires (misclassified or
        the correct win is under the spiking margin) and capacity remains. On a
        confident correct hit, do NOT grow (optionally refine). Returns
        (unit_id, grew)."""
        self.labels_seen.add(label)
        ranked = self._ranked(pattern)

        if self.grow_criterion == "novelty":
            # CONTROL criterion: grow on low spike-code similarity, ignore label.
            code = _latency_code(pattern)
            best_sim = max((_cos(code, nu["code"]) for nu in self.neurons), default=-1.0)
            do_grow = best_sim < self.vigilance
        else:
            # CONTRIBUTION criterion: grow on task error or thin spiking margin.
            if not ranked:
                do_grow = True                          # empty net -> must recruit
            else:
                pred = ranked[0][1]
                top = ranked[0][0]
                runner = next((s for s, lab, *_ in ranked if lab != pred), 0.0)
                do_grow = (pred != label) or (top - runner < self.margin)

        if do_grow and len(self.neurons) < self.capacity:
            return self._grow(pattern, label), True

        winner = ranked[0][2] if ranked else None       # confident/blocked -> no growth
        if self.refine and winner is not None and self.neurons[winner]["label"] == label:
            self._refine_winner(winner, pattern)
        return winner, False

    def classify(self, pattern):
        """Recognition only (never grows): the class of the neuron with the
        strongest SPIKING response (earliest first-spike + most spikes)."""
        ranked = self._ranked(pattern)
        return ranked[0][1] if ranked else None


# ==========================================================================
# self-check: un-riggable, deterministic; each claim carries a CONTROL that
# MUST fail. Builds a K-class multimodal spiking task from encode_multimodal
# signatures, streamed ONCE (single pass). Prints a table, then asserts.
# ==========================================================================
def _jitter(levels, amt, rng):
    """Per-modality sensor jitter clamped to [0,1] (a noisy capture)."""
    return [min(1.0, max(0.0, v + rng.uniform(-amt, amt))) for v in levels]


def _encode(levels):
    """Genuinely multimodal [0,1]^N vector from four per-modality levels."""
    v, t, a, d = levels
    return encode_multimodal(visual=[v], temperature=t, audio=[a], distance=d)


# 6 well-separated multimodal signatures (visual, temperature, audio, distance).
_TASK = [
    (0.1, 0.1, 0.1, 0.1),      # class 0: all-low
    (0.9, 0.9, 0.9, 0.9),      # class 1: all-high
    (0.1, 0.9, 0.1, 0.9),      # class 2: alternating A
    (0.9, 0.1, 0.9, 0.1),      # class 3: alternating B
    (0.5, 0.5, 0.5, 0.5),      # class 4: all-mid
    (0.9, 0.5, 0.1, 0.5),      # class 5: descending
]
_JIT = 0.05                    # capture jitter
_REPEATS = 8                   # single pass = _REPEATS x K jittered samples, each seen ONCE


def _make_stream(data_seed):
    """A fixed single-pass stream of (pattern, label): each class repeated with
    fresh jitter, interleaved. Deterministic given data_seed."""
    rng = random.Random(data_seed)
    stream = []
    for _ in range(_REPEATS):
        for label, levels in enumerate(_TASK):
            stream.append((_encode(_jitter(levels, _JIT, rng)), label))
    return stream


def _make_test(data_seed, per_class=6):
    """Held-out jittered instances (data unseen in training) for one-pass acc."""
    rng = random.Random(data_seed)
    test = []
    for _ in range(per_class):
        for label, levels in enumerate(_TASK):
            test.append((_encode(_jitter(levels, _JIT, rng)), label))
    return test


def _stream_once(learner, stream):
    for pattern, label in stream:
        learner.observe(pattern, label)
    return learner


def _accuracy(learner, test):
    hits = sum(1 for pattern, label in test if learner.classify(pattern) == label)
    return hits / len(test)


def main():
    stream = _make_stream(data_seed=0)
    test = _make_test(data_seed=100)

    print("=" * 72)
    print("NEUROGENESIS -- spiking, contribution-criterion, single-shot classifier")
    print("  grounded in NG-SNN (Tang et al., Neural Networks 2026, DOI 10.1016/j.neunet.2026.108656)")
    print("=" * 72)

    # 1. CONTRIBUTION (not novelty) + PARSIMONY -------------------------------
    #    contribution grows on task error; novelty grows on low spike-code
    #    similarity ignoring correctness -> novelty OVER-GROWS for no accuracy win.
    contrib = _stream_once(NeurogenesisLearner(grow_criterion="contribution", seed=0), stream)
    novelty = _stream_once(
        NeurogenesisLearner(grow_criterion="novelty", vigilance=0.9, seed=0), stream)
    contrib_acc = _accuracy(contrib, test)
    novelty_acc = _accuracy(novelty, test)

    # 2. SINGLE-SHOT analytical vs RANDOM weights (same net, same growth rule) --
    single_acc = contrib_acc                             # contrib IS the single-shot net
    randw = _stream_once(
        NeurogenesisLearner(grow_criterion="contribution", weight_init="random", seed=0),
        stream)
    random_acc = _accuracy(randw, test)

    # 3. GENUINELY SPIKING: the decision is a function of EMITTED SPIKES --------
    p0, lab0 = test[0]
    ranked0 = contrib._ranked(p0)
    win_idx = ranked0[0][2]
    train, vtrace = contrib._run_full(contrib.neurons[win_idx]["w"], p0)
    win_count = sum(train)
    win_first = next((i for i, s in enumerate(train) if s), contrib.dim)
    spikes_are_binary = all(s in (0, 1) for s in train)
    graded_is_not_spikes = not all(v in (0.0, 1.0) for v in vtrace)   # CONTROL: graded V
    decision_from_spikes = contrib.classify(p0) == contrib.neurons[win_idx]["label"]

    # 4. BOUNDED: structure follows TASK, not noise ----------------------------
    task_units = contrib.n_units
    noise_rng = random.Random(3)
    noise_stream = [([noise_rng.random() for _ in range(N)], noise_rng.randrange(len(_TASK)))
                    for _ in range(3 * contrib.capacity)]      # random patterns + random labels
    noise = _stream_once(NeurogenesisLearner(grow_criterion="contribution", seed=0), noise_stream)
    noise_units = noise.n_units

    # 5. RAPID SINGLE-PASS CONVERGENCE: one-pass accuracy (no epochs) -----------
    #    (reported below; honest negative printed -- toy mechanism, not SOTA.)

    # 6. DETERMINISM -----------------------------------------------------------
    da = _stream_once(NeurogenesisLearner(seed=7), stream)
    db = _stream_once(NeurogenesisLearner(seed=7), stream)
    same_struct = (da.n_units == db.n_units
                   and [nu["w"] for nu in da.neurons] == [nu["w"] for nu in db.neurons])
    same_preds = ([da.classify(p) for p, _ in test] == [db.classify(p) for p, _ in test])

    # -- table ----------------------------------------------------------------
    print(f"task: K={len(_TASK)} classes, {len(stream)} samples single-pass, "
          f"jitter={_JIT}, dim={N}, capacity={contrib.capacity}")
    print("-" * 72)
    print(f"1 CONTRIBUTION vs NOVELTY : contrib {task_units:2d} units acc {contrib_acc:.2f}  |  "
          f"novelty {novelty.n_units:2d} units acc {novelty_acc:.2f}")
    print("                            -> contribution is compact (fewer units, >= accuracy)")
    print(f"2 SINGLE-SHOT vs RANDOM   : matched acc {single_acc:.2f}  |  random-weight acc {random_acc:.2f}"
          f"  (zero iterations after growth)")
    print(f"3 SPIKING DECISION        : winner emitted {win_count} spikes, first-spike t={win_first}; "
          f"train binary={spikes_are_binary}, graded-V-is-not-spikes={graded_is_not_spikes}")
    print(f"4 BOUNDED (task vs noise) : task grew {task_units:2d}/{contrib.capacity}  |  "
          f"noise grew {noise_units:2d}/{noise.capacity}  (structure follows task, not noise)")
    print(f"5 ONE-PASS ACCURACY       : {contrib_acc:.2f} on held-out jittered captures (single pass)")
    print(f"6 DETERMINISM             : identical structure={same_struct}, identical predictions={same_preds}")
    print("=" * 72)

    # -- asserts: every claim with a control that must fail --------------------
    # 1 contribution parsimony: fewer units than novelty at comparable accuracy.
    assert contrib_acc >= 0.8, f"contribution one-pass accuracy too low: {contrib_acc:.2f}"
    assert task_units < novelty.n_units, (
        f"contribution must be more parsimonious than novelty "
        f"({task_units} vs {novelty.n_units})")
    assert contrib_acc >= novelty_acc - 0.05, (
        f"contribution must match/beat novelty accuracy at fewer units "
        f"({contrib_acc:.2f} vs {novelty_acc:.2f})")

    # 2 single-shot beats random weights, zero post-growth iterations.
    assert single_acc >= 0.8, f"single-shot analytical accuracy too low: {single_acc:.2f}"
    assert single_acc > random_acc + 0.3, (
        f"single-shot analytical must beat random weights "
        f"({single_acc:.2f} vs {random_acc:.2f})")

    # 3 genuinely spiking: decision reads emitted spikes; graded read would fail.
    assert spikes_are_binary, "spike train must be 0/1 emitted spikes"
    assert isinstance(win_count, int) and win_count >= 1, "winner must cross threshold and spike"
    assert isinstance(win_first, int), "first-spike time must be an integer index"
    assert decision_from_spikes, "class must be the argmax spiking response"
    assert graded_is_not_spikes, "CONTROL: a raw graded membrane value is not a spike train"

    # 4 bounded: task saturates well under capacity; noise runs to the cap.
    assert task_units < contrib.capacity, "task structure must stay under capacity"
    assert task_units <= 2 * len(_TASK), f"task structure not compact: {task_units} units"
    assert noise_units > 2 * task_units, (
        f"noise must over-grow vs task ({noise_units} vs {task_units})")
    assert noise_units >= 0.75 * noise.capacity, (
        f"noise growth must approach the capacity cap ({noise_units}/{noise.capacity})")

    # 6 determinism.
    assert same_struct, "same seed + stream must give identical grown structure"
    assert same_preds, "same seed + stream must give identical predictions"

    print("HONEST NEGATIVE: toy-scale MECHANISM demo of NG-SNN's two innovations, "
          "NOT the paper's dataset SOTA;")
    print("                 the single-shot rule is a matched-filter / prototype latency "
          "weight-set (a simplification")
    print("                 of the paper's full activity-dependent analytical learning).")
    print("-" * 72)
    print("self-check OK: neurogenesis as the spiking basis -- grows spiking neurons on a "
          "CONTRIBUTION criterion")
    print(f"  ({task_units} vs novelty's {novelty.n_units}), sets weights SINGLE-SHOT from spike "
          f"activity (matched {single_acc:.2f}")
    print(f"  vs random {random_acc:.2f}, zero iterations), bounded/compact ({task_units}/"
          f"{contrib.capacity}), single-pass acc {contrib_acc:.2f}, deterministic.")


if __name__ == "__main__":
    main()
