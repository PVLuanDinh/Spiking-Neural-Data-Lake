r"""
Metacognitive monitor -- the LESIONABLE SPIKING reliability monitor that turns the
brain's self-model from PRODUCED into CONSUMED. Closes ladder gap L10 / SI-10 in
docs/SELF_IMPROVEMENT_RESEARCH.md.

Honest framing (read this first). Across the atlas the brain already PRODUCES
metacognitive signals -- decision confidence, prediction-error surprise, learning
progress (neuromodulation/intrinsic_motivation.py), novelty, drift. Until now they
were SILOED: computed and then dropped on the floor. Nothing fused them, and nothing
let them CONTROL behavior (self-improvement gap #7: "METACOGNITION PRODUCED BUT NOT
CONSUMED"). This module is the consumer. It fuses the signals into ONE reliability
state via a dedicated, LESIONABLE LIF population and emits control GATES that decide
whether the brain should LEARN, EXPLORE, GROW, or HOLD.

What it is -- and is not. This is a bounded metacognitive-CONTROL primitive: a spiking
reliability monitor that gates learning (focus where learnable, abstain when
unreliable, grow only on learnable-novelty), per ladder rung L10 / increment SI-10.
It is NOT AGI and NOT consciousness. The "reliability" it computes is a FUNCTIONAL
signal -- the windowed spike-count firing rate of a LIF pool -- not a felt state, not
an introspective report, not a claim of a self. No aleatoric-vs-epistemic split is
claimed; it reads total reliability only.

The spiking mechanism (why a LIF pool, not a scalar). update() drives a population of
LIF neurons with a current that is EXCITED by confidence and by rising learning
progress, and INHIBITED by persistent-surprise-WITHOUT-progress -- the noisy-TV
signature (high surprise while learning progress ~ 0). Reliability is then read from
the EMITTED SPIKES: the windowed spike-count firing rate = spikes / (pop_size * steps)
over the update's integration window, NOT the subthreshold membrane. Reading spikes
(the neuron's actual output), not the membrane (a would-be intention that may never
fire), is the whole point -- a neuron can hold a high sub-threshold voltage yet emit
nothing, and such a silent "almost" must NOT read as reliable. Per-neuron bias
heterogeneity (fixed per instance) turns the all-or-none LIF into a graded population
rate.

The gates (metacognition made into control). All are pure functions of the reliability
plus the current inputs:
  * learn_gate()   -> [0,1] learning-rate multiplier: HIGH when reliable AND learnable
                      (LP>0), LOW on the unreliable / noisy.
  * explore()      -> bool: explore when reliability is LOW or learnable-novelty is
                      high; exploit when reliability is high and the context familiar.
  * grow_trigger() -> bool: recruit a unit ONLY on novelty that is HIGH-but-LEARNABLE
                      (novel AND LP>0); NEVER on pure noise (high surprise, LP ~ 0).
  * hold()/abstain()-> bool: reliability in the bottom `hold_p` of this pool's OWN recent output
                      -> HOLD, do not commit a decision. The bar is a running quantile, not a
                      value: see __init__ for the measurement that retired the previous
                      "no-evidence" floor (a structural zero on 197/197 live ticks).
  * reliability()  -> [0,1] the fused spike-count firing rate.

Lesionability (the un-riggable control). Constructed with lesion=True every gate
becomes a no-op that reproduces the UNGATED, pre-metacognition behavior exactly:
learn_gate == 1.0, explore == False, hold == False, and grow_trigger falls back to the
old novelty-threshold-only policy. A lesioned monitor is byte-identical downstream to
having no monitor at all -- so the self-check can prove the GATES (not some incidental
plumbing) are what improve learning.

Reuses (does NOT reimplement): cortex/lif_neuron.LIFNeuron (the spiking pool),
substrate/_mathutils.{mean,sigmoid}, substrate/signal_calibrator.SignalCalibrator (the O(1)
running quantile that is the HOLD bar). The self-check additionally drives
hippocampus/growing_learner.GrowingLearner (a real neurogenesis learner whose capacity
the monitor gates) and neuromodulation/intrinsic_motivation.CuriosityDrive (the honest
learning-progress / novelty source), keyed off a coarse, label-free input bucket.

Pure standard library (math, random, zlib, collections). Per-instance random.Random,
zlib.crc32 for bucketing (never hash()), ASCII-only prints, deterministic.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.metacognitive_monitor
"""

import math

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate._mathutils import sigmoid
from snn_data_lake.substrate.signal_calibrator import SignalCalibrator


def _clamp01(x):
    """Clamp a scalar into [0,1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class MetacognitiveMonitor:
    """A lesionable spiking reliability monitor that GATES learning.

    update(confidence, surprise, learning_progress, novelty) drives a LIF population and
    reads a reliability from its EMITTED spikes; learn_gate / explore / grow_trigger /
    hold / abstain are pure control gates over that reliability plus the inputs. With
    lesion=True the gates become no-ops == the ungated control. Deterministic;
    per-instance random.Random(seed) fixes the population's bias heterogeneity."""

    def __init__(
        self,
        pop_size=24,
        steps=16,
        tau_m=10.0,
        w_conf=0.6,
        w_lp=0.5,
        w_surprise=0.9,
        base=0.0,
        bias_spread=0.4,
        r_explore=0.30,
        nov_hi=0.4,
        lp_eps=1e-6,
        hold_p=0.10,
        lesion=False,
        seed=0,
    ):
        import random

        self.rng = random.Random(seed)  # per-instance rng (fixes bias heterogeneity)
        self.pop_size = int(pop_size)
        self.steps = int(steps)
        self.w_conf = float(w_conf)  # confidence -> excitation
        self.w_lp = float(w_lp)  # rising learning-progress -> excitation
        self.w_surprise = float(w_surprise)  # persistent-surprise-without-progress -> inhibition
        self.base = float(base)
        # (r_hold retired: the HOLD bar is now `_rel_floor`, a running quantile of this pool's rate)
        self.r_explore = float(r_explore)  # reliability below this -> explore
        self.nov_hi = float(nov_hi)  # novelty at/above this counts as "novel"
        self.lp_eps = float(lp_eps)  # learning-progress above this counts as "learnable"
        self.lesion = bool(lesion)
        # the dedicated LIF reliability pool (fixed per-neuron bias -> graded population rate)
        self.pool = [LIFNeuron(tau_m=tau_m) for _ in range(self.pop_size)]
        self.bias = [self.rng.uniform(-bias_spread, bias_spread) for _ in range(self.pop_size)]
        # last-update cache (all gates are pure functions of these)
        self._rel = 0.0  # windowed spike-count firing rate in [0,1]
        self._spikes = 0  # integer spike count over the window
        self._mem = 0.0  # mean final membrane (CONTROL comparison ONLY)
        self._conf = 0.0
        self._surprise = 0.0
        self._lp = 0.0
        self._nov = 0.0
        # DERIVED HOLD FLOOR -- a running LOW QUANTILE of the rate this pool has actually emitted.
        #
        # This replaces a one-shot "no-evidence" probe (update(confidence=0, surprise=1.0, lp=0) at
        # construction) that was advertised as self-recalibrating and was in fact a STRUCTURAL ZERO.
        # Measured over 197 live wake ticks it printed 0.0000 on 197/197, so hold() was not
        # "reliability no better than no-evidence" but the much narrower "the pool emitted literally
        # zero spikes". The cause is arithmetic, not tuning: that probe drives every neuron with
        # base - w_surprise (0.4 - 0.9 = -0.5), and the widest bias is +bias_spread (+0.4), so the
        # most excited neuron still sees -0.1. The drive is strictly negative for ANY
        # base + bias_spread <= w_surprise, the LIF cannot fire, and the floor is 0 by construction.
        #
        # Re-probing at the LIVE operating point does not fix it either, and the reason is worth
        # recording so it is not retried: the pool's rate is monotone in `drive`, so rel <= floor
        # would reduce to w_conf*c + w_lp*lp + w_surprise*s*min(1,lp) <= 0, which for non-negative
        # inputs is exactly (c == 0 and lp == 0). A live counterfactual floor is degenerate too,
        # just in a different place.
        #
        # So the floor becomes DISTRIBUTION-relative instead of value-relative: abstain when this
        # tick's reliability sits in the bottom `hold_p` of what this pool has recently emitted.
        # That is the repo's own running-baseline idiom (cf. brain.py `_margin_base` / `_lg_base`)
        # in its O(1)-memory quantile form, and it is scale-FREE -- rescaling the upstream
        # confidence moves the samples and the quantile together, which is the property the old
        # comment claimed and the old mechanism did not have.
        self.hold_p = float(hold_p)
        self._rel_q = SignalCalibrator("metacog_reliability", p=self.hold_p)

    def update(self, confidence, surprise, learning_progress, novelty=0.0):
        """Fuse one set of reliability signals into a spike-count reliability state.

        Drives every LIF neuron for `steps` timesteps with a current that is EXCITED by
        confidence + learning progress and INHIBITED by surprise that comes WITHOUT
        progress (surprise * (1 - lp) -- learnable surprise is not penalized, noisy
        surprise is). Reliability is the windowed spike-count firing rate read from the
        EMITTED spikes. Returns the reliability."""
        c = _clamp01(confidence)
        s = _clamp01(surprise)
        lp = max(0.0, float(learning_progress))
        nov = _clamp01(novelty)
        self._conf, self._surprise, self._lp, self._nov = c, s, lp, nov
        # excite by confidence + rising LP; inhibit by persistent-surprise-WITHOUT-progress
        drive = (
            self.base
            + self.w_conf * c
            + self.w_lp * lp
            - self.w_surprise * s * (1.0 - min(1.0, lp))
        )
        spikes = 0
        memsum = 0.0
        for neuron, b in zip(self.pool, self.bias):
            neuron.reset_state()  # fresh integration window per update (no cross-leak)
            current = drive + b
            for _ in range(self.steps):
                spikes += neuron.step(current)
            memsum += neuron.v  # final subthreshold membrane (for the control read only)
        self._spikes = spikes
        self._rel = spikes / float(
            self.pop_size * self.steps
        )  # SPIKE-COUNT firing rate (not membrane)
        self._mem = memsum / float(self.pop_size)
        self._rel_q.observe(self._rel)  # fold into the pool's own rate distribution (the HOLD bar)
        return self._rel

    def reliability(self):
        """Fused reliability in [0,1] -- the windowed spike-count firing rate of the LIF pool."""
        return self._rel

    def _reliability_from_membrane(self):
        """CONTROL ONLY -- a (worse) reliability read from the subthreshold MEMBRANE, not
        the spikes. Used by the self-check to prove reliability is spike-derived: this is
        a continuous voltage read (a squashed mean membrane), so it is NOT an integer
        spike-count fraction, and it reads a silent-but-charged pool as 'reliable' when
        the true output (spikes) is zero. Never used for gating."""
        return sigmoid(self._mem)

    def learn_gate(self):
        """Learning-rate multiplier in [0,1]: HIGH when reliable AND learnable (LP>0), LOW
        on the unreliable / noisy. lesion -> 1.0 (ungated)."""
        if self.lesion:
            return 1.0
        return _clamp01(self._rel * (1.0 + self._lp))

    def explore(self):
        """Explore when reliability is LOW or learnable-novelty is high; exploit (False)
        when reliability is high and the context is familiar. lesion -> False (ungated)."""
        if self.lesion:
            return False
        if self._rel < self.r_explore:
            return True
        return self._nov >= self.nov_hi and self._lp > self.lp_eps

    def grow_trigger(self):
        """Recruit a unit ONLY on novelty that is HIGH-but-LEARNABLE (novel AND LP>0);
        NEVER on pure noise (high surprise, LP ~ 0). lesion -> the ungated
        novelty-threshold-only policy (grows on ANY novelty, including noise)."""
        if self.lesion:
            return self._nov >= self.nov_hi
        return self._nov >= self.nov_hi and self._lp > self.lp_eps

    @property
    def _rel_floor(self):
        """The live HOLD bar: the running `hold_p` quantile of this pool's own emitted rate, or None
        before two samples exist. Kept under the original name so existing readers keep working."""
        r = self._rel_q.result()
        return r["value"] if r["n"] >= 2 else None

    def hold(self):
        """HOLD / abstain: reliability in the BOTTOM `hold_p` of this pool's own recent output ->
        do not commit a decision.

        The bar is `_rel_floor`, a running low quantile of the rates this very pool has emitted (see
        __init__ for why the previous no-evidence probe was a structural zero and why re-probing at
        the live operating point is degenerate too). Distribution-relative, so it survives any
        rescaling of the upstream confidence rather than encoding the scale it was picked under.
        Requires >=2 samples -- a single observation is trivially its own quantile, which would make
        the first update always hold. lesion -> False (ungated: always commit)."""
        if self.lesion:
            return False
        floor = self._rel_floor
        return False if floor is None else self._rel <= floor

    def abstain(self):
        """Alias of hold() -- do not commit a decision when reliability is very low."""
        return self.hold()


# =====================================================================================
# Un-riggable self-check. Deterministic (seed=0), ASCII-only. Each claim is
# paired with a CONTROL that FAILS. The monitor GATES a real neurogenesis learner over a
# stream that mixes LEARNABLE classes with an UNLEARNABLE noise context.
# =====================================================================================

DIM = 64  # feature-space dimensionality
K = 4  # number of learnable classes
ROUNDS = 12  # stream repeats per class
NOISE_PER_ROUND = 12  # unlearnable noise samples interleaved per round
HELDOUT_PER_CLASS = 10  # held-out class samples for the accuracy test
VIG = 0.80  # grower vigilance (classes recognized above it; fresh noise falls below it)
ENGAGE = 0.12  # learn_gate floor to refine a recognized context
JIT = 0.15  # class-sample input jitter (kept within a quantization bin)


def _quantize(vec):
    """Coarse, label-free bucket code: threshold each feature at 0.5 -> a DIM-bit signature.
    Class samples (jittered around well-separated levels) collapse to a STABLE code; fresh
    noise lands on an essentially unique code every time (2**DIM space)."""
    return bytes(1 if x >= 0.5 else 0 for x in vec)


def _bucket_key(vec):
    """Deterministic bucket id from the coarse code (zlib.crc32, never hash())."""
    import zlib

    return zlib.crc32(_quantize(vec))


def _cosine(a, b):
    """Cosine similarity of two non-negative vectors; 0 if either has zero norm."""
    d = na = nb = 0.0
    for x, y in zip(a, b):
        d += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return d / math.sqrt(na * nb)


def _make_classes(seed):
    """K well-separated sparse templates: class c is HIGH (0.75) on its own disjoint
    feature block and LOW (0.25) elsewhere -> classes separable, noise (uniform) overlaps."""
    import random

    rng = random.Random(1000 + seed)
    block = DIM // K
    templates = []
    for c in range(K):
        lo, hi = c * block, (c + 1) * block
        templates.append([0.75 if lo <= i < hi else 0.25 for i in range(DIM)])
    # tiny per-template perturbation so templates are not perfectly structured (still separable)
    for t in templates:
        for i in range(DIM):
            t[i] = _clamp01(t[i] + rng.uniform(-0.02, 0.02))
    return templates


def _class_sample(template, rng):
    """A jittered draw of a class template, clamped to [0,1]."""
    return [_clamp01(v + rng.uniform(-JIT, JIT)) for v in template]


def _noise_sample(rng):
    """A fresh, unlearnable uniform-random sample (a new one every call -> LP stays ~0)."""
    return [rng.random() for _ in range(DIM)]


class _World:
    """Produces the four reliability signals per sample WITHOUT using any ground-truth
    label. confidence/surprise come from an always-on per-bucket familiarity running mean;
    learning_progress/novelty come from a CuriosityDrive keyed by the same coarse bucket.
    Always-on and cheap -- separate from the gated, capacity-committing grower."""

    def __init__(self, seed):
        from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive

        self.fam = {}  # bucket -> running-mean feature vector
        self.cur = CuriosityDrive(novelty_decay=0.8, seed=seed)
        self.beta = 0.3  # familiarity running-mean rate

    def signals(self, vec):
        """Return (confidence, surprise, learning_progress, novelty) for one input.
        confidence = recognition vs the bucket's prior running mean (0 for a novel bucket);
        surprise = 1 - confidence (the prediction error fed to the curiosity drive);
        learning_progress, novelty = the anti-noisy-TV curiosity signals for the bucket."""
        key = _bucket_key(vec)
        st = self.fam.get(key)
        conf = _clamp01(_cosine(vec, st)) if st is not None else 0.0  # recognize BEFORE updating
        surprise = 1.0 - conf
        if st is None:  # then fold the sample into the running mean
            self.fam[key] = list(vec)
        else:
            self.fam[key] = [(1.0 - self.beta) * m + self.beta * x for m, x in zip(st, vec)]
        self.cur.observe(key, surprise)  # honest LP: falls only for a RECURRING bucket
        return conf, surprise, self.cur.learning_progress(key), self.cur.novelty(key)


def _build_stream(seed):
    """A shuffled stream interleaving K jittered class samples per round with fresh noise."""
    import random

    rng = random.Random(seed)
    templates = _make_classes(seed)
    plan = []
    for _ in range(ROUNDS):
        for ci in range(K):
            plan.append(("class", ci))
        for _ in range(NOISE_PER_ROUND):
            plan.append(("noise", -1))
    rng.shuffle(plan)
    stream = []
    for kind, ci in plan:
        vec = _class_sample(templates[ci], rng) if kind == "class" else _noise_sample(rng)
        stream.append((ci if kind == "class" else -1, vec))
    return stream


def _run_gated(stream, world, monitor, grower):
    """Drive the grower through the monitor's gates. Returns (proto_label, abstains).

    hold() -> abstain entirely (the unreliable noise); on a recognized context refine iff
    learn_gate clears ENGAGE; on an unrecognized context recruit iff grow_trigger fires
    (learnable-novelty, never pure noise). proto_label[i] is the true class that recruited
    prototype i (or -1 for a noise-born unit) -- harness bookkeeping only, never fed in."""
    proto_label = []
    abstains = 0
    for ci, vec in stream:
        conf, surp, lp, nov = world.signals(vec)
        monitor.update(conf, surp, lp, nov)
        if monitor.hold():
            abstains += 1
            continue
        _, best_sim = grower._best(vec)
        recognized = best_sim >= grower.vigilance
        engage = (recognized and monitor.learn_gate() >= ENGAGE) or (
            not recognized and monitor.grow_trigger()
        )
        if not engage:
            continue
        uid, grew = grower.observe(vec)
        if grew:
            if uid != len(proto_label):
                raise AssertionError("prototype-label alignment broken")
            proto_label.append(ci)
    return proto_label, abstains


def _run_plain(stream, grower):
    """No-monitor baseline: observe EVERY sample (the ungated control)."""
    proto_label = []
    for ci, vec in stream:
        uid, grew = grower.observe(vec)
        if grew:
            proto_label.append(ci)
    return proto_label


def _heldout_acc(grower, proto_label, seed):
    """Held-out accuracy on fresh class samples: correct iff the nearest prototype was
    recruited by the SAME class (a noise-born or wrong-class prototype winning = wrong)."""
    import random

    rng = random.Random(9000 + seed)
    templates = _make_classes(seed)
    correct = total = 0
    for ci in range(K):
        for _ in range(HELDOUT_PER_CLASS):
            vec = _class_sample(templates[ci], rng)
            pid = grower.classify(vec)
            total += 1
            if 0 <= pid < len(proto_label) and proto_label[pid] == ci:
                correct += 1
    return correct / total


def _experiment(seed):
    """Run the gated vs lesioned vs plain comparison and the abstain test. Returns a dict
    of pure numbers/bools (so two runs can be compared byte-for-byte for determinism)."""
    from snn_data_lake.hippocampus.growing_learner import GrowingLearner

    stream = _build_stream(seed)

    # ---- (1) GATED: monitor gates learning; abstains on noise ----
    g_world = _World(seed)
    g_mon = MetacognitiveMonitor(lesion=False, seed=seed)
    g_grower = GrowingLearner(vigilance=VIG, lr=0.2, dim=DIM)
    g_label, g_abstains = _run_gated(stream, g_world, g_mon, g_grower)
    g_units = g_grower.n_units
    g_acc = _heldout_acc(g_grower, g_label, seed)

    # ---- LESIONED: same harness, gates OFF -> learns + grows on everything (incl. noise) ----
    l_world = _World(seed)
    l_mon = MetacognitiveMonitor(lesion=True, seed=seed)
    l_grower = GrowingLearner(vigilance=VIG, lr=0.2, dim=DIM)
    l_label, l_abstains = _run_gated(stream, l_world, l_mon, l_grower)
    l_units = l_grower.n_units
    l_acc = _heldout_acc(l_grower, l_label, seed)

    # ---- PLAIN no-monitor baseline (for the lesion byte-identical proof) ----
    p_grower = GrowingLearner(vigilance=VIG, lr=0.2, dim=DIM)
    p_label = _run_plain(stream, p_grower)
    lesion_byte_identical = l_grower.prototypes == p_grower.prototypes and l_label == p_label

    # ---- (2) CONFIDENCE-ABSTAIN vs always-commit, on a RELIABLE + UNRELIABLE query mix ----
    import random

    qrng = random.Random(5000 + seed)
    templates = _make_classes(seed)
    queries = []  # (true_label, vec); -1 == OOD noise
    for _ in range(HELDOUT_PER_CLASS):
        for ci in range(K):
            queries.append((ci, _class_sample(templates[ci], qrng)))
    ood_total = K * HELDOUT_PER_CLASS
    for _ in range(ood_total):
        queries.append((-1, _noise_sample(qrng)))
    qrng.shuffle(queries)

    a_correct = a_total = 0  # always-commit
    c_correct = c_committed = ood_abstained = 0  # monitor: commit only when reliable
    for true_label, vec in queries:
        conf, surp, lp, nov = g_world.signals(vec)
        g_mon.update(conf, surp, lp, nov)
        pid = g_grower.classify(vec)
        pred = g_label[pid] if 0 <= pid < len(g_label) else -2
        hit = 1 if pred == true_label else 0
        a_correct += hit
        a_total += 1
        if g_mon.hold():  # abstain on the unreliable
            if true_label == -1:
                ood_abstained += 1
            continue
        c_committed += 1
        c_correct += hit
    always_acc = a_correct / a_total
    commit_acc = c_correct / c_committed if c_committed else 0.0

    # ---- (3) SPIKING structural reads: reliable vs noise vs sub-threshold ----
    probe = MetacognitiveMonitor(lesion=False, seed=seed)
    probe.update(confidence=0.99, surprise=0.02, learning_progress=0.2, novelty=0.5)  # reliable
    rel_reliable = probe.reliability()
    spike_count_is_integer = abs(rel_reliable * probe.pop_size * probe.steps - probe._spikes) < 1e-9
    mem_reliable = probe._reliability_from_membrane()
    mem_prod = mem_reliable * probe.pop_size * probe.steps
    mem_is_integer_spikecount = abs(mem_prod - round(mem_prod)) < 1e-9
    probe.update(confidence=0.0, surprise=1.0, learning_progress=0.0, novelty=1.0)  # noise
    rel_noise = probe.reliability()

    # sub-threshold trap: a pool charged to just below threshold emits NOTHING, yet its
    # membrane sits high -> the membrane read falsely says "reliable", spikes correctly 0.
    trap = MetacognitiveMonitor(lesion=False, seed=seed, bias_spread=0.0)
    decay = math.exp(-1.0 / 10.0)
    sub = 0.99 * (1.0 - decay)  # steady-state V* ~ 0.99 < threshold 1.0
    n = LIFNeuron(tau_m=10.0)
    trap_spikes = sum(n.step(sub) for _ in range(trap.steps))
    trap_membrane_rel = sigmoid(n.v)
    trap_spike_rel = trap_spikes / float(trap.steps)

    return {
        "g_acc": g_acc,
        "l_acc": l_acc,
        "g_units": g_units,
        "l_units": l_units,
        "g_abstains": g_abstains,
        "l_abstains": l_abstains,
        "lesion_byte_identical": lesion_byte_identical,
        "always_acc": always_acc,
        "commit_acc": commit_acc,
        "ood_abstained": ood_abstained,
        "committed": c_committed,
        "rel_reliable": rel_reliable,
        "rel_noise": rel_noise,
        "spike_count_is_integer": spike_count_is_integer,
        "mem_is_integer_spikecount": mem_is_integer_spikecount,
        "mem_reliable": mem_reliable,
        "trap_spikes": trap_spikes,
        "trap_membrane_rel": trap_membrane_rel,
        "trap_spike_rel": trap_spike_rel,
    }


def _p(lines, s):
    """Append an ASCII-only line (Windows cp1252 safety)."""
    if any(ord(ch) > 127 for ch in s):
        raise AssertionError("non-ASCII output")
    lines.append(s)


def main():
    r = _experiment(0)
    lines = []
    _p(lines, "=" * 78)
    _p(lines, "METACOGNITIVE MONITOR -- lesionable spiking reliability gate (L10 / SI-10)")
    _p(lines, "=" * 78)
    _p(lines, "metacognition CONSUMED: a LIF reliability pool gates LEARN / EXPLORE / GROW / HOLD")
    _p(lines, "-" * 78)
    _p(
        lines,
        f"stream: {K} learnable classes x {ROUNDS} rounds + {NOISE_PER_ROUND} noise/round "
        f"(dim={DIM}, vigilance={VIG})",
    )
    _p(lines, "-" * 78)
    _p(lines, "                       held-out acc   units    abstains")
    _p(
        lines,
        f"  GATED (monitor on) :   {r['g_acc']:.3f}       {r['g_units']:>3}      {r['g_abstains']:>3}",
    )
    _p(
        lines,
        f"  LESIONED (control) :   {r['l_acc']:.3f}       {r['l_units']:>3}      {r['l_abstains']:>3}",
    )
    _p(lines, "-" * 78)
    _p(
        lines,
        f"(1) gating helps    : gated acc {r['g_acc']:.3f} >= lesioned {r['l_acc']:.3f} + margin, "
        f"and gated units {r['g_units']} < lesioned {r['l_units']}",
    )
    _p(
        lines,
        f"(2) confidence-hold : committed acc {r['commit_acc']:.3f} > always-commit {r['always_acc']:.3f}; "
        f"abstained on {r['ood_abstained']} OOD queries ({r['committed']} committed)",
    )
    _p(
        lines,
        f"(3) spiking read    : reliable rate {r['rel_reliable']:.3f} vs noise {r['rel_noise']:.3f}; "
        f"spike-count integer={r['spike_count_is_integer']} membrane-integer={r['mem_is_integer_spikecount']}",
    )
    _p(
        lines,
        f"    sub-thr trap    : spikes={r['trap_spikes']} spike_rel={r['trap_spike_rel']:.3f} "
        f"but membrane_rel={r['trap_membrane_rel']:.3f} (membrane falsely 'reliable')",
    )
    _p(lines, f"    lesion==ungated : byte-identical downstream={r['lesion_byte_identical']}")
    _p(lines, f"(4) determinism     : two seed=0 runs identical={_experiment(0) == r}")
    _p(lines, "=" * 78)
    for line in lines:
        print(line)

    # ---- (1) GATING HELPS LEARNING (headline) ----
    assert r["g_acc"] >= r["l_acc"] + 0.05, (
        f"gated held-out {r['g_acc']:.3f} must beat lesioned {r['l_acc']:.3f} by a margin"
    )
    assert r["g_units"] < r["l_units"], (
        f"gated must recruit FEWER units ({r['g_units']}) than lesioned ({r['l_units']}) -- no noise memorizing"
    )
    assert r["g_abstains"] > 0, "the gated monitor must ABSTAIN on the unlearnable noise"

    # ---- (2) CONFIDENCE-ABSTAIN beats always-commit ----
    assert r["commit_acc"] > r["always_acc"], (
        f"committed-only accuracy {r['commit_acc']:.3f} must beat always-commit {r['always_acc']:.3f}"
    )
    assert r["ood_abstained"] > 0, "the monitor must HOLD on some OOD / unreliable queries"

    # ---- (3) SPIKING + LESIONABLE ----
    assert r["rel_reliable"] > r["rel_noise"], (
        "the spike-count reliability must separate reliable from noise"
    )
    assert r["spike_count_is_integer"], (
        "reliability * pop_size * steps must be an exact SPIKE COUNT (integer)"
    )
    assert not r["mem_is_integer_spikecount"], (
        "the membrane read is NOT a spike-count fraction -- structural proof reliability is spike-derived"
    )
    assert r["trap_spikes"] == 0 and r["trap_spike_rel"] == 0.0, (
        "a sub-threshold pool emits NO spikes -> spike reliability is correctly 0 (abstain)"
    )
    assert r["trap_membrane_rel"] > 0.5, (
        "the membrane read falsely calls the silent sub-threshold pool 'reliable' -- why we read spikes"
    )
    assert r["lesion_byte_identical"], "lesion=True must be byte-identical downstream to no monitor"

    # ---- (4) DETERMINISM ----
    assert _experiment(0) == r, "two seed=0 runs must be byte-for-byte identical"

    print(
        f"self-check OK: GATING lifts held-out {r['l_acc']:.3f}->{r['g_acc']:.3f} while cutting units "
        f"{r['l_units']}->{r['g_units']} (abstained on {r['g_abstains']} noise);"
    )
    print(
        f"  confidence-HOLD raises accuracy {r['always_acc']:.3f}->{r['commit_acc']:.3f} by abstaining on "
        f"{r['ood_abstained']} OOD; reliability is SPIKE-count ({r['rel_reliable']:.2f} vs noise "
        f"{r['rel_noise']:.2f}),"
    )
    print(
        "  membrane-read control FAILS the structural spike-count assert and mis-reads a silent pool; "
        "lesion==ungated; deterministic"
    )


if __name__ == "__main__":
    main()
