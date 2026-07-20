"""
NACA -- Neuromodulation-Assisted Credit Assignment (a diffuse global broadcast that RESHAPES local
plasticity, additive and opt-in; changes no existing behaviour).

The paper's idea (Science Advances, Zhang/Zeng et al., adi2947): a GLOBAL neuromodulatory signal --
computed from an INPUT-TYPE EXPECTATION, not from a backprop gradient -- gates how a LOCAL plasticity
rule writes to the weights. There is NO error gradient and NO weight transport: credit is assigned by
a chemical broadcast lambda passing through a nonlinear metaplasticity gate f_local(lambda) that can
scale AND FLIP the sign of the local LTP/LTD update. Used to gate a CONFLICTING new task away from an
already-consolidated one, it protects the old task -- lower catastrophic interference.

WHAT IS AND IS NOT HERE (honest):
  * NACABroadcast is a MODULATOR, not a plasticity primitive. It never computes an eligibility trace or
    a weight update of its own. It takes an EXISTING local delta (here, the R-STDP eligibility gate from
    basal_ganglia.r_stdp.RSTDPNeuron, REUSED verbatim -- no new learning rule is invented) and returns an
    ADDITIVE correction  -eta * f_local(lambda) * delta  to lay ON TOP of it. lambda=0 (or any lambda that
    zeroes the gate) -> the correction is all zeros -> the update reduces EXACTLY to the unmodulated local
    rule. That off path is proven byte-identical in the self-check.
  * N_in is an INPUT-TYPE EXPECTATION matrix: one random-uniform lambda in [0,1] per (input type, synapse),
    built with random.Random(seed). An expectation vector E (from the current input-type signature) gates
    N_in; the per-type scalar neuromodulator lambda_l is the mean of that gated row. Each input TYPE thus
    gets its own fixed lambda -> its own metaplasticity regime.
  * f_local (paper Eq. 6) is the nonlinear metaplasticity gate: as lambda rises it climbs toward theta_max,
    then FALLS back to zero at lambda_inv, then INVERTS SIGN above lambda_inv -- i.e. the same neuromodulator
    reshapes both the AMPLITUDE and the POLARITY of local LTP/LTD. Below lambda_inv the correction damps the
    local update (stability); above it, the correction flips it (the metaplastic switch).

HONEST SCOPE of the continual-learning self-check. This is credit assignment by neuromodulation, a
stability/plasticity KNOB -- not backprop, not a gradient, not cognition/AGI. The self-check runs the
standard catastrophic-INTERFERENCE probe: consolidate task 1 (discriminate P0 from P1) at full strength,
then train the DIRECTLY CONFLICTING task 2 (the reversal: P1 over P0) on the SAME synapses. Plain R-STDP
fully overwrites task 1 (forgets). With NACA gating the conflicting task, task 1's consolidated weights
are protected -- sparse weight change, task 1 retained -- AT THE COST of not learning the reversal (the
stability/plasticity trade-off, reported not hidden). Uniform damping of BOTH tasks cannot help (it scales
task 1's margin and task 2's perturbation equally); the gain requires gating the NEW task specifically,
which is exactly what an input-type-keyed neuromodulator does.
Refs: Zhang, Zeng et al., Sci. Adv. 2023 (adi2947); Izhikevich 2007 (the R-STDP eligibility reused here).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.naca
"""
import random

ETA = 0.5              # neuromodulatory correction gain (how strongly the broadcast reshapes the local delta)
LAM_INV = 0.5          # inversion threshold: below it the gate is positive (damp), above it it flips sign
THETA_MAX = 1.2        # peak height of the metaplasticity gate


class NACABroadcast:
    """A diffuse neuromodulatory broadcast that reshapes a LOCAL plasticity delta (never replaces it).

    Holds an input-type EXPECTATION matrix N_in (n_types x n_syn, random-uniform in [0,1]). For a given
    input-type signature it forms an expectation vector E, gates N_in by E, and averages to a scalar
    neuromodulator lambda_l in [0,1]. lambda_l passes through the nonlinear metaplasticity gate f_local
    to yield an ADDITIVE correction on top of the local delta. Pure stdlib, deterministic per seed."""

    def __init__(self, n_syn, n_types, seed=0, eta=ETA, lam_inv=LAM_INV, theta_max=THETA_MAX):
        rng = random.Random(seed)
        self.n_syn = int(n_syn)
        self.n_types = int(n_types)
        self.eta = float(eta)
        self.lam_inv = float(lam_inv)
        self.theta_max = float(theta_max)
        # N_in: one random-uniform expectation lambda in [0,1] per (input type, synapse).
        self.N_in = [[rng.uniform(0.0, 1.0) for _ in range(self.n_syn)] for _ in range(self.n_types)]

    def _expectation(self, signature):
        """Turn an input-type signature into a convex expectation vector E over the input types. An int is a
        one-hot (select that type); a vector is normalised to sum 1; an empty/zero signature -> all zeros
        (which yields lambda_l = 0, i.e. the gate is off)."""
        if isinstance(signature, int):
            e = [0.0] * self.n_types
            e[signature] = 1.0
            return e
        s = list(signature)
        tot = sum(s)
        if tot <= 0.0:
            return [0.0] * self.n_types
        return [v / tot for v in s]

    def lam(self, signature):
        """The per-type neuromodulator lambda_l: gate the input-type expectation matrix N_in by E, then
        average over the synapses -> a scalar in [0,1]. (a mean/select of N_in gated by E.)"""
        if self.n_syn == 0:
            return 0.0
        e = self._expectation(signature)
        acc = 0.0
        for j in range(self.n_syn):
            g = 0.0
            for t in range(self.n_types):
                g += e[t] * self.N_in[t][j]
            acc += g
        return acc / self.n_syn

    @staticmethod
    def f_local(lam, lam_inv=LAM_INV, theta_max=THETA_MAX):
        """Nonlinear metaplasticity gate (paper Eq. 6). As lambda rises the gate climbs to theta_max, then
        FALLS back to 0 at lambda_inv, then INVERTS SIGN above lambda_inv -- one neuromodulator reshaping the
        AMPLITUDE and the POLARITY of the local LTP/LTD delta. f_local(0)=0 and f_local(lam_inv)=0, so the
        endpoints leave the local rule untouched; the peak lies at lambda_inv/2."""
        if lam <= 0.0:
            return 0.0
        if lam <= lam_inv:
            peak = 0.5 * lam_inv
            if lam <= peak:
                return theta_max * (lam / peak) if peak > 0.0 else theta_max
            return theta_max * (lam_inv - lam) / (lam_inv - peak)
        span = 1.0 - lam_inv
        return -theta_max * (lam - lam_inv) / span if span > 0.0 else 0.0

    def modulate(self, local_delta, lam):
        """The additive neuromodulatory correction to lay ON TOP of an existing local delta:
        -eta * f_local(lambda) * delta, per synapse. It does NOT replace local plasticity -- add it to the
        local update. When the gate is off (f_local(lambda) = 0, e.g. lambda = 0) every entry is 0, so the
        combined update is byte-identical to the unmodulated local rule."""
        g = self.f_local(lam, self.lam_inv, self.theta_max)
        return [-self.eta * g * d for d in local_delta]


# --------------------------------------------------------------------------------------------------
# self-check (deterministic, zero-dep, ASCII)
# --------------------------------------------------------------------------------------------------
_N_IN = 20
_TAU = 8.0
_T_POST = 20.0
_LR = 0.06


def _psp(dt):
    return 2.718281828 ** (-dt / _TAU) if dt > 0.0 else 0.0


def _drive(neuron, code):
    """Peak spike-timing membrane response of the neuron to a spike train (tempotron-style)."""
    return sum(neuron.w[i] * _psp(_T_POST - code[i]) for i in range(_N_IN))


def _deposit(neuron, code, d, naca=None, ptype=0):
    """One R-STDP deposit (REUSED RSTDPNeuron eligibility as the local delta source): present the stimulus
    spike train, fire the post spike after it, gate the eligibility into the weights by reward d. When
    `naca` is given, add its neuromodulatory correction ON TOP of the local delta (additive)."""
    neuron.e = [0.0] * _N_IN
    neuron.t_pre = [None] * _N_IN
    neuron.t_post = None
    for i, t in enumerate(code):
        neuron.pre_spike(i, float(t))
    neuron.post_spike(_T_POST)
    deltas = neuron.reward(float(d))                      # local delta = lr * reward * eligibility (applied)
    if naca is not None:
        mod = naca.modulate(deltas, naca.lam(ptype))     # additive correction ON TOP (0 when the gate is off)
        for i in range(_N_IN):
            neuron.w[i] += mod[i]
    return deltas


def _gate_shape_check():
    """f_local rises to theta_max, falls back to 0 at lambda_inv, then inverts sign above it."""
    f = NACABroadcast.f_local
    return f(0.10), f(0.25), f(0.40), f(0.0), f(0.5), f(0.75), f(1.0)


def _off_path_byte_identity():
    """Prove the off path: an R-STDP local delta with NACA at lambda=0 reduces EXACTLY to the unmodulated
    local rule. Two identical RSTDPNeurons run the same deposit; one adds the NACA correction at lambda=0."""
    from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron

    naca = NACABroadcast(n_syn=_N_IN, n_types=2, seed=1)
    code = list(range(_N_IN))                             # a fixed rank-order spike train

    plain = RSTDPNeuron(_N_IN, seed=0)
    _deposit(plain, code, +1.0, naca=None)
    w_plain = list(plain.w)

    gated = RSTDPNeuron(_N_IN, seed=0)
    local = _deposit(gated, code, +1.0, naca=None)        # apply the local rule first (same as plain)
    mod = naca.modulate(local, naca.lam([0.0, 0.0]))      # empty expectation -> lambda=0 -> gate off
    for i in range(_N_IN):
        gated.w[i] += mod[i]

    return all(m == 0.0 for m in mod), gated.w == w_plain


def _consolidation_reversal_check():
    """The standard catastrophic-INTERFERENCE probe on ONE shared-weight R-STDP discriminator. Consolidate
    task 1 (P0 over P1) at full strength (PLAIN), then train the DIRECTLY CONFLICTING reversal task 2 (P1
    over P0) on the SAME synapses. PLAIN overwrites task 1 (forgets); NACA gating the conflicting task keeps
    task 1's consolidated weights (sparse change, task 1 retained) at the cost of not learning task 2."""
    from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron
    from snn_data_lake.cortex.rank_order_coding import rank_code

    seed, noise, e1, e2 = 7, 0.05, 60, 60
    proto_rng = random.Random(seed)
    protos = [[proto_rng.random() for _ in range(_N_IN)] for _ in range(2)]   # P0, P1

    def code_of(li, rng):
        return rank_code([min(1.0, max(0.0, protos[li][j] + rng.uniform(-noise, noise))) for j in range(_N_IN)])

    def acc(neuron, pos, neg, rng):                       # P(drive(pos) > drive(neg)) over noisy samples
        return sum(_drive(neuron, code_of(pos, rng)) > _drive(neuron, code_of(neg, rng)) for _ in range(50)) / 50.0

    def run(naca_phase2):
        neuron = RSTDPNeuron(_N_IN, lr=_LR, tau_plus=_TAU, tau_minus=_TAU, w0=0.3, seed=seed)
        tr, ev = random.Random(seed + 1), random.Random(seed + 2)
        for _ in range(e1):                               # task 1 (PLAIN, ungated): consolidate P0 over P1
            _deposit(neuron, code_of(0, tr), +1.0, None)
            _deposit(neuron, code_of(1, tr), -1.0, None)
        w1, a1 = list(neuron.w), acc(neuron, 0, 1, ev)
        for _ in range(e2):                               # task 2 (reversal): P1 over P0 -- conflicts with task 1
            _deposit(neuron, code_of(1, tr), +1.0, naca_phase2, 1)
            _deposit(neuron, code_of(0, tr), -1.0, naca_phase2, 0)
        a1_after, a2_rev = acc(neuron, 0, 1, ev), acc(neuron, 1, 0, ev)
        dw = [abs(neuron.w[i] - w1[i]) for i in range(_N_IN)]
        return a1, a1_after, a2_rev, dw

    naca = NACABroadcast(n_syn=_N_IN, n_types=2, seed=seed + 50, eta=0.82, lam_inv=1.0, theta_max=1.2)
    p_a1, p_after, p_rev, p_dw = run(None)
    n_a1, n_after, n_rev, n_dw = run(naca)

    eps = 0.2
    total = _N_IN
    return {
        "p_a1": p_a1, "p_after": p_after, "p_rev": p_rev, "p_sparse": sum(d < eps for d in p_dw) / total,
        "n_a1": n_a1, "n_after": n_after, "n_rev": n_rev, "n_sparse": sum(d < eps for d in n_dw) / total,
        "p_forget": p_a1 - p_after, "n_forget": n_a1 - n_after,
    }


def _wire_engages_check():
    """The opt-in wire in cortex.interaction_learning: with a NACABroadcast passed, the learner's weights
    DIFFER from the naca=None run on identical training (the modulation is live). The naca=None OFF path is
    byte-identical -- proven by the existing interaction_learning self-check, which is unchanged."""
    from snn_data_lake.cortex.interaction_learning import SpikingInteractionLearner

    responses, n = ["a", "b"], 12
    rng = random.Random(1)
    protos = [[rng.random() for _ in range(n)] for _ in responses]
    data = [([min(1.0, max(0.0, v + rng.uniform(-0.05, 0.05))) for v in protos[li]], responses[li])
            for li in range(2) for _ in range(6)]
    plain = SpikingInteractionLearner(responses, n, seed=3)
    gated = SpikingInteractionLearner(responses, n, seed=3,
                                      naca=NACABroadcast(n, 2, seed=9, eta=0.5, lam_inv=1.0))
    for _ in range(10):
        for stim, lab in data:
            plain.interact(stim, lab)
            gated.interact(stim, lab)
    return plain.weights() != gated.weights()


def main():
    print("=" * 78)
    print("NACA -- Neuromodulation-Assisted Credit Assignment (global gate reshaping a LOCAL delta)")
    print("=" * 78)

    rise, peak, fall, zero_lo, zero_hi, flipped, end = _gate_shape_check()
    print("metaplasticity gate f_local(lambda) (lambda_inv=0.5, theta_max=1.2):")
    print(f"  f(0.0)={zero_lo:+.3f}  f(0.10)={rise:+.3f}  f(0.25)={peak:+.3f}  f(0.40)={fall:+.3f}  "
          f"f(0.50)={zero_hi:+.3f}  f(0.75)={flipped:+.3f}  f(1.00)={end:+.3f}")
    print("  -> rises toward theta_max, falls back to 0 at lambda_inv, then INVERTS SIGN above it")

    off_zero, identical = _off_path_byte_identity()
    print("-" * 78)
    print("off path (lambda=0): NACA correction is all zeros -> combined update == unmodulated local rule")
    print(f"  correction all-zero: {off_zero}   weights byte-identical to plain R-STDP: {identical}")

    r = _consolidation_reversal_check()
    print("-" * 78)
    print("catastrophic-interference probe: consolidate task1 (P0>P1), then the CONFLICTING reversal task2 (P1>P0):")
    print(f"  PLAIN local rule : task1 {r['p_a1']:.2f} -> {r['p_after']:.2f}  (forgetting {r['p_forget']:+.2f})"
          f"   reversal learned {r['p_rev']:.2f}   synapses left at ~0: {r['p_sparse']:.2f}")
    print(f"  NACA-gated task2 : task1 {r['n_a1']:.2f} -> {r['n_after']:.2f}  (forgetting {r['n_forget']:+.2f})"
          f"   reversal learned {r['n_rev']:.2f}   synapses left at ~0: {r['n_sparse']:.2f}")
    print("  honest trade-off: NACA protects task1 by NOT learning the conflicting task2 (stability over plasticity)")

    wire_live = _wire_engages_check()
    print("-" * 78)
    print(f"opt-in wire (cortex.interaction_learning): naca set changes the learner's weights: {wire_live}")
    print("  (naca=None is byte-identical -- proven by the unchanged interaction_learning self-check)")
    print("=" * 78)

    # ---- un-riggable asserts ----
    # metaplasticity gate: rises to a positive crest, falls to 0 at the endpoints, inverts sign above lambda_inv
    assert zero_lo == 0.0 and abs(zero_hi) < 1e-12, "f_local must be 0 at lambda=0 and at lambda_inv"
    assert 0.0 < rise < peak, "f_local must RISE from 0 toward the crest"
    assert peak > fall > 0.0, "f_local must FALL back after the crest while staying positive below lambda_inv"
    assert abs(peak - THETA_MAX) < 1e-12, "the crest must reach theta_max"
    assert flipped < 0.0 and end < 0.0, "f_local must INVERT SIGN above lambda_inv (polarity switch)"
    # off path: lambda=0 reduces EXACTLY to the unmodulated local rule (byte-identical)
    assert off_zero, "at lambda=0 the NACA correction must be exactly zero on every synapse"
    assert identical, "off path must be byte-identical to the plain local R-STDP update"
    # phase 1 (consolidation) is identical for both runs -> a fair comparison
    assert r["p_a1"] == r["n_a1"] and r["p_a1"] > 0.85, "both must consolidate task 1 identically before task 2"
    # headline: NACA yields SPARSE weight changes and LOWER forgetting of task 1
    assert r["p_forget"] > 0.5, f"the plain local rule must catastrophically forget task 1 ({r['p_forget']:+.2f})"
    assert r["n_forget"] < r["p_forget"] - 0.3, \
        f"NACA must forget task 1 LESS than the plain local rule ({r['n_forget']:+.2f} vs {r['p_forget']:+.2f})"
    assert r["n_sparse"] > 0.5, f"NACA must leave a visible peak of synapses at ~0 ({r['n_sparse']:.2f})"
    assert r["n_sparse"] > r["p_sparse"] + 0.2, \
        f"NACA weight changes must be sparser than the plain local rule ({r['n_sparse']:.2f} vs {r['p_sparse']:.2f})"
    # honest trade-off: protection costs plasticity -- NACA does NOT learn the conflicting reversal
    assert r["n_rev"] < r["p_rev"] - 0.3, \
        f"NACA must trade plasticity for stability: it should NOT learn the conflicting task ({r['n_rev']:.2f})"
    # the opt-in wire is live (naca=None off path is byte-identical, via the existing self-check)
    assert wire_live, "the interaction_learning wire must actually apply the modulation when naca is set"

    print(f"self-check OK: metaplasticity gate rises to {THETA_MAX:.1f}, falls, and inverts sign; the off path "
          f"(lambda=0) is byte-identical to plain R-STDP; consolidating task1 then training the CONFLICTING "
          f"task2, plain forgets {r['p_forget']:+.2f} while NACA holds task1 ({r['n_forget']:+.2f}) with "
          f"{r['n_sparse']:.2f} of synapses left at ~0 (vs {r['p_sparse']:.2f}) -- at the honest cost of not "
          f"learning the reversal ({r['n_rev']:.2f}); additive, opt-in, deterministic, zero-dep")


if __name__ == "__main__":
    main()
