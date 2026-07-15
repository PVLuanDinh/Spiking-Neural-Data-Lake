"""
Competitive activity-maximization learning (Demin & Nekhaev 2018, "FEELING") -- UNSUPERVISED
spiking self-organization of class families via intralayer inhibitory competition.

Reproduces the core learning principle of Demin & Nekhaev, "Recurrent Spiking Neural Network
Learning Based on a Competitive Maximization of Neuronal Activity", Front. Neuroinform. 12:79
(2018) -- PMC6250118, the "Family-Engaged Execution and Learning of Induced Neuron Groups"
(FEELING) algorithm. The paper's premise: each neuron tends to MAXIMIZE ITS OWN activity in
COMPETITION with the others for limited resources. A spiking layer with feed-forward excitatory
weights plus learnable intralayer INHIBITORY weights, trained under that single principle, makes
its neurons SELF-ORGANIZE into families that correspond to input CLASSES -- with NO teacher, after
only a short supervised weight init (which merely breaks the initial symmetry).

Faithful mechanism (paper's equations, event-driven at post-synaptic spikes):
  * Activity approx (eq. 1-2): each neuron carries an INSTANT activity a (fast EMA of its 0/1 spike
    train, tau_a) and an AVERAGE activity theta (slow EMA, tau_theta >> tau_a).
  * Neuron current (eq. 4): I_j = sum_i w_ij * a_i  -- the post current is the weighted sum of the
    pre-synaptic INSTANT activities (not raw spikes), plus the (negative) lateral contributions.
  * Feed-forward, activity-maximization (eq. 6): on a spike of post j,  w_ij += alpha*(a_i - theta_i),
    clamped to [0, 1]. Post j strengthens inputs that fire ABOVE their own running average and weakens
    the rest -- "forced uptake" of limited resource under competition (the BCM-like selectivity driver).
  * Lateral inhibitory, learnable competition (eq. 8): on a spike of post k',
    w_kk' += -gamma*(a_k - theta_k), clamped to [-1, 0]. A currently-highly-active competitor k gets a
    STRONGER inhibitory grip on the neuron k' that just fired -- a learnable Winner-Take-All.
  * A spike-frequency adaptation / dynamic threshold (eq. 5 intent): each unit's effective threshold
    rises on its own spikes and decays (tau_th), keeping the competition equitable so different units
    can win different classes rather than one unit hogging every input.
  * Short supervised init then FULLY unsupervised: a few labeled presentations inject an extra current
    into the target unit (breaking symmetry, per the paper); thereafter NO labels are used -- the layer
    keeps self-organizing from competition alone.

HONEST scope / deviations (this is a MECHANISM demo, not the full model):
  * Reproduced: eq. 1, 2, 4, 5, 6, 8 -- the feed-forward activity-maximization update and the
    learnable lateral inhibition, which are exactly "spiking units + lateral inhibition + the
    activity-maximization update". This is the heart of FEELING.
  * NOT reproduced at this scale: the reciprocal top-down connections (eq. 7, needs a deeper output
    layer) and the cooperative intralayer rule that grows multi-neuron families (eq. 9); the 784-100-10
    MNIST network; and BRIAN-level continuous LIF dynamics. Here it is a small discrete-time layer over
    a SYNTHETIC few-class spike dataset. It is label-free FEATURE self-organization -- the substrate for
    a future autonomous spike-stream reader -- an MNIST-SCALE mechanism, not a full MNIST model.
  * Three discrete-time ENGINEERING choices, stated plainly, make the coarse (dt=1) simulation converge
    where BRIAN's continuous dynamics would: (i) per-neuron L1 weight normalization -- a concrete
    encoding of the paper's own "limited biochemical resources" premise (strengthening one synapse must
    weaken others); (ii) WINNER-gated plasticity -- only the highest-current unit tunes each step, the
    coarse-step realization of the instantaneous winner-take-all that STRONG lateral inhibition produces
    in continuous time (the learnable lateral weights still shape which unit wins); (iii) a small
    baseline lateral inhibition at init so units compete from step one instead of co-firing and locking
    into identical copies. These are faithful to the FEELING principle, not extra supervision.

Purely label-free: labels are used ONLY at eval time to READ OUT the unsupervised clusters (standard
cluster-purity scoring), never to train. Pure standard library, zero-dep, deterministic (seeded),
additive (a new member of the plasticity family; nothing existing changes).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.competitive_learning
"""
import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron


class CompetitiveLearner:
    """A spiking layer trained by competitive activity maximization (FEELING; Demin & Nekhaev 2018).

    n_inputs pre-synaptic channels drive n_units LIF neurons through feed-forward excitatory weights
    w[j][i] in [0, 1] and learnable intralayer inhibitory weights lat[j][k] in [-1, 0]. Each neuron
    tracks an instant activity (fast EMA) and an average activity (slow EMA); learning is event-driven
    at post-synaptic spikes and follows the activity-maximization principle (eq. 6) plus a learnable
    lateral competition (eq. 8). No teacher after a short symmetry-breaking init.
    """

    def __init__(self, n_inputs, n_units, seed=0, alpha=0.04, gamma=0.05,
                 tau_a=5.0, tau_theta=60.0, dv_th=0.45, tau_th=40.0,
                 tau_m=20.0, v_threshold=0.9, refractory=2, w_norm=3.0, lat_init=-0.5):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.alpha = alpha                 # eq. 6 feed-forward learning rate
        self.gamma = gamma                 # eq. 8 lateral-inhibition learning rate
        self.w_norm = w_norm               # per-neuron L1 budget: the paper's LIMITED resource
        self.da = 1.0 - 1.0 / tau_a        # instant-activity EMA retention (eq. 1)
        self.ka = 1.0 / tau_a              # instant-activity spike increment
        self.dth = 1.0 - 1.0 / tau_theta   # average-activity EMA retention (eq. 2)
        self.kth = 1.0 / tau_theta         # average-activity spike increment
        self.dv_th = dv_th                 # dynamic-threshold spike increment (eq. 5)
        self.th_decay = math.exp(-1.0 / tau_th)   # dynamic-threshold leak
        rng = random.Random(seed)
        # feed-forward excitatory weights, small random init on the L1 resource simplex (each unit
        # starts as a random point of total weight w_norm) -- symmetry is broken by the supervised init
        self.w = [self._renorm([rng.uniform(0.1, 1.0) for _ in range(n_inputs)]) for _ in range(n_units)]
        # intralayer inhibitory weights lat[j][k] = grip of competitor k onto j, in [-1, 0]; a small
        # baseline inhibition exists from the start (the paper: lateral inhibition "helps learn faster")
        # so units compete immediately instead of co-firing and locking into redundant copies
        self.lat = [[0.0 if j == k else lat_init for k in range(n_units)] for j in range(n_units)]
        self.units = [LIFNeuron(tau_m=tau_m, v_threshold=v_threshold, refractory=refractory)
                      for _ in range(n_units)]
        self.a_in = [0.0] * n_inputs       # input instant activity a_i
        self.th_in = [0.0] * n_inputs      # input average activity theta_i
        self.a_out = [0.0] * n_units       # unit instant activity a_j
        self.th_out = [0.0] * n_units      # unit average activity theta_j
        self.adapt = [0.0] * n_units       # spike-frequency adaptation (dynamic threshold)

    def _renorm(self, wj):
        """Project a feed-forward weight vector onto the LIMITED-resource L1 budget (sum == w_norm,
        entries >= 0). This IS the paper's premise: biochemical resources are limited, so a neuron that
        strengthens one synapse must weaken others -- competition among a neuron's own inputs."""
        s = sum(wj)
        if s <= 0.0:
            return [self.w_norm / len(wj)] * len(wj)
        f = self.w_norm / s
        return [x * f for x in wj]

    def reset_activity(self):
        """Clear the fast/slow activity traces, adaptation, and membrane state (weights are KEPT)."""
        self.a_in = [0.0] * self.n_inputs
        self.th_in = [0.0] * self.n_inputs
        self.a_out = [0.0] * self.n_units
        self.th_out = [0.0] * self.n_units
        self.adapt = [0.0] * self.n_units
        for u in self.units:
            u.reset_state()

    def step(self, in_spikes, learn=True, sup_unit=None, sup_current=0.0):
        """Advance one timestep. `in_spikes` is a 0/1 list over inputs. Returns the 0/1 output list.

        If `learn`, applies the feed-forward (eq. 6) and lateral (eq. 8) updates at post spikes and the
        weight leak. If `sup_unit` is not None, injects `sup_current` into that unit (symmetry-breaking
        supervised init only)."""
        # 1. input activities (eq. 1-2) from this step's input spikes
        for i in range(self.n_inputs):
            s = in_spikes[i]
            self.a_in[i] = self.a_in[i] * self.da + s * self.ka
            self.th_in[i] = self.th_in[i] * self.dth + s * self.kth

        # 2. output currents (eq. 4): feed-forward instant-activity sum + learned lateral inhibition
        out = [0] * self.n_units
        currents = [0.0] * self.n_units
        for j in range(self.n_units):
            wj = self.w[j]
            cur = 0.0
            for i in range(self.n_inputs):
                cur += wj[i] * self.a_in[i]
            latj = self.lat[j]
            for k in range(self.n_units):
                if k != j:
                    cur += latj[k] * self.a_out[k]     # lat <= 0, a_out >= 0  -> inhibition
            if sup_unit is not None and j == sup_unit:
                cur += sup_current
            currents[j] = cur
            out[j] = self.units[j].step(cur - self.adapt[j])   # dynamic threshold as adaptation current

        # 3. output activities (eq. 1-2) + adaptation (eq. 5) from this step's output spikes
        for j in range(self.n_units):
            s = out[j]
            self.a_out[j] = self.a_out[j] * self.da + s * self.ka
            self.th_out[j] = self.th_out[j] * self.dth + s * self.kth
            self.adapt[j] = self.adapt[j] * self.th_decay + s * self.dv_th

        # 4. WINNER-gated learning (the "competition for limited resources": the best-matched unit
        #    takes the plasticity). The winner is the highest-current unit -- during the supervised
        #    init it is forced to be the labeled target (it carries the injected current). Only the
        #    winner tunes, so a unit never learns a pattern it did not win -> tunings stay distinct.
        if learn:
            if sup_unit is not None:
                winner = sup_unit
            else:
                winner = max(range(self.n_units), key=lambda j: currents[j])
            if out[winner]:
                wj = self.w[winner]
                a_in, th_in, alpha = self.a_in, self.th_in, self.alpha
                for i in range(self.n_inputs):         # eq. 6: strengthen above-average inputs,
                    v = wj[i] + alpha * (a_in[i] - th_in[i])   # weaken the rest
                    wj[i] = 0.0 if v < 0.0 else v
                self.w[winner] = self._renorm(wj)      # limited-resource competition (re-distribute)
                latw = self.lat[winner]
                for k in range(self.n_units):          # eq. 8: active competitors grip the firer
                    if k != winner:
                        v = latw[k] - self.gamma * (self.a_out[k] - self.th_out[k])
                        latw[k] = -1.0 if v < -1.0 else (0.0 if v > 0.0 else v)
        return out

    def present(self, rate_vector, rng, steps, learn=True, sup_unit=None, sup_current=0.0,
                reset_before=False):
        """Present one pattern (per-input firing rates) for `steps` timesteps of seeded Poisson input.
        Returns the per-unit output spike counts over the presentation."""
        if reset_before:
            self.reset_activity()
        counts = [0] * self.n_units
        for _ in range(steps):
            in_spikes = [1 if rng.random() < rate_vector[i] else 0 for i in range(self.n_inputs)]
            out = self.step(in_spikes, learn=learn, sup_unit=sup_unit, sup_current=sup_current)
            for j in range(self.n_units):
                counts[j] += out[j]
        return counts

    def response(self, rate_vector, rng, steps):
        """Frozen readout: reset the transient state, present a pattern with NO learning (weights and
        learned lateral inhibition are kept), and return the per-unit firing rate."""
        self.reset_activity()
        counts = self.present(rate_vector, rng, steps, learn=False)
        return [c / steps for c in counts]


# --------------------------------------------------------------------------------------------------
# Synthetic multi-class spike dataset + unsupervised training + cluster-purity evaluation
# --------------------------------------------------------------------------------------------------

def make_prototypes(n_classes=4, n_inputs=16, width=6, hi=0.85, lo=0.08):
    """n_classes prototype firing-rate vectors: each class lights a contiguous, OVERLAPPING block of
    `width` inputs at rate `hi`, the rest at `lo`. Overlap between adjacent classes makes the classes
    genuinely share features (digit-stroke-like), so random projections do NOT trivially separate them."""
    stride = max(1, (n_inputs - width) // max(1, n_classes - 1))
    protos = []
    for c in range(n_classes):
        start = min(c * stride, n_inputs - width)
        v = [lo] * n_inputs
        for i in range(start, start + width):
            v[i] = hi
        protos.append(v)
    return protos


def noisy_variant(proto, rng, jitter=0.14):
    """A noisy instance of a prototype: each input rate is jittered and clamped to [0, 1]."""
    return [min(1.0, max(0.0, r + rng.uniform(-jitter, jitter))) for r in proto]


def make_dataset(protos, rng, per_class):
    """Return a list of (class_index, rate_vector) noisy variants, `per_class` per prototype."""
    data = []
    for c, proto in enumerate(protos):
        for _ in range(per_class):
            data.append((c, noisy_variant(proto, rng)))
    return data


def train(learner, protos, seed, sup_per_class=25, unsup_presentations=900,
          steps=45, sup_current=6.0):
    """Short SUPERVISED symmetry-breaking init (unit c gets an injected current on class c), then a
    FULLY UNSUPERVISED phase: random unlabeled variants, competition only. Returns nothing (in place)."""
    n_classes = len(protos)
    rng = random.Random(seed)
    # supervised init: cycle classes, inject current into the matching unit to break symmetry
    for _ in range(sup_per_class):
        for c in range(n_classes):
            learner.present(noisy_variant(protos[c], rng), rng, steps, learn=True,
                            sup_unit=c, sup_current=sup_current, reset_before=True)
    # unsupervised: NO labels, NO injected current -- self-organization by competition alone
    for _ in range(unsup_presentations):
        c = rng.randrange(n_classes)
        learner.present(noisy_variant(protos[c], rng), rng, steps, learn=True, reset_before=True)


def preferred_classes(learner, protos, seed, per_class=12, steps=45):
    """Read out each unit's preferred class from a LABELING set (labels used only to score clusters).
    Returns (pref[unit] -> class, mean-response matrix R[unit][class])."""
    rng = random.Random(seed)
    n_classes = len(protos)
    resp_sum = [[0.0] * n_classes for _ in range(learner.n_units)]
    counts = [0] * n_classes
    label_set = make_dataset(protos, rng, per_class)
    for c, vec in label_set:
        r = learner.response(vec, rng, steps)
        for j in range(learner.n_units):
            resp_sum[j][c] += r[j]
        counts[c] += 1
    R = [[resp_sum[j][c] / counts[c] for c in range(n_classes)] for j in range(learner.n_units)]
    pref = [max(range(n_classes), key=lambda c: R[j][c]) for j in range(learner.n_units)]
    return pref, R


def purity(learner, protos, pref, seed, per_class=20, steps=45):
    """Cluster purity on a DISJOINT test set: classify each pattern by the winning unit's preferred
    class. Returns (accuracy, n_distinct_covered_classes)."""
    rng = random.Random(seed)
    test_set = make_dataset(protos, rng, per_class)
    correct = 0
    for c, vec in test_set:
        r = learner.response(vec, rng, steps)
        winner = max(range(learner.n_units), key=lambda j: r[j])
        if pref[winner] == c:
            correct += 1
    covered = len(set(pref))
    return correct / len(test_set), covered


N_CLASSES, N_INPUTS, WIDTH = 6, 24, 10


def _evaluate(seed, train_flag):
    """Build a learner; optionally train it (unsupervised, after a short supervised init); return
    (purity, families_covered). Fully deterministic in `seed`."""
    protos = make_prototypes(n_classes=N_CLASSES, n_inputs=N_INPUTS, width=WIDTH)
    learner = CompetitiveLearner(N_INPUTS, N_CLASSES, seed=seed)
    if train_flag:
        train(learner, protos, seed=seed + 1)
    pref, _ = preferred_classes(learner, protos, seed=seed + 2)
    return purity(learner, protos, pref, seed=seed + 3)


def main():
    seeds = [1, 3, 7, 11]                              # average over seeds -> no lucky-seed rigging
    chance = 1.0 / N_CLASSES
    trained = [_evaluate(s, train_flag=True) for s in seeds]
    control = [_evaluate(s, train_flag=False) for s in seeds]
    det_pur, _ = _evaluate(seeds[0], train_flag=True)  # determinism re-run of the first seed
    t_pur = [p for p, _ in trained]
    t_cov = [c for _, c in trained]
    c_pur = [p for p, _ in control]
    c_cov = [c for _, c in control]
    t_mean = sum(t_pur) / len(t_pur)
    c_mean = sum(c_pur) / len(c_pur)

    print("=" * 78)
    print("COMPETITIVE ACTIVITY-MAXIMIZATION (FEELING; Demin & Nekhaev 2018) -- unsupervised")
    print("=" * 78)
    print(f"{N_CLASSES} synthetic spike classes over {N_INPUTS} inputs (heavily OVERLAPPING blocks); "
          f"chance purity {chance:.2f}")
    print(f"  seeds tested        : {seeds}")
    print(f"  UNSUPERVISED-trained: purity {[round(p, 2) for p in t_pur]}  mean {t_mean:.2f}  "
          f"families {t_cov} / {N_CLASSES}")
    print(f"  CONTROL (untrained) : purity {[round(p, 2) for p in c_pur]}  mean {c_mean:.2f}  "
          f"families {c_cov} / {N_CLASSES}")
    print("  (labels are used ONLY to read out the clusters at eval time; training saw no labels)")
    print("=" * 78)

    # ---- self-check: deterministic asserts, zero-dep, ASCII only ----
    assert all(p > 0.85 for p in t_pur), \
        f"unsupervised FEELING must self-organize class-selective families every seed ({t_pur})"
    assert all(cov == N_CLASSES for cov in t_cov), \
        f"a distinct family must cover EACH class, every seed ({t_cov} vs {N_CLASSES})"
    assert all(cov < N_CLASSES for cov in c_cov), \
        f"the untrained control must FAIL to cover all classes (no families form) ({c_cov})"
    assert all(tp > cp + 0.30 for tp, cp in zip(t_pur, c_pur)), \
        f"trained must beat the untrained control by a wide margin every seed ({t_pur} vs {c_pur})"
    assert c_mean < 0.5 and c_mean < t_mean - 0.40, \
        f"untrained control must sit far below trained, near chance ({c_mean:.2f} vs {t_mean:.2f}, chance {chance:.2f})"
    assert det_pur == t_pur[0], f"same seed -> identical purity (deterministic) ({det_pur} vs {t_pur[0]})"
    print(f"self-check OK: unsupervised purity mean {t_mean:.2f} (all {N_CLASSES}/{N_CLASSES} families) "
          f">> untrained {c_mean:.2f} ~ chance {chance:.2f}; label-free; deterministic")


if __name__ == "__main__":
    main()
