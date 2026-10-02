"""
acetylcholine (ACh) neuromodulation -- the encode-vs-retrieve mode switch.

Headline: ACh is a novelty-driven MODE SWITCH between feedforward ENCODING (high ACh: afferent
input enhanced, recurrent path suppressed, plasticity ON -> write NEW memories from external
input) and recurrent RETRIEVAL/consolidation (low ACh: feedforward attenuated, recurrent
connections dominate -> RECALL stored associations, plasticity reduced). ACh rises on NOVELTY /
mismatch and decays with familiarity / quiescence. This completes the repo's neuromodulator set.

Genuinely DISTINCT from the other two neuromodulators here:
  - cortisol.py (TONIC STRESS): a slow `stress` scalar rising on AVERSIVE/danger events; it biases
    toward the safe action (lowers reflex threshold, amplifies aversive learning, adds caution).
    ACh is NOT stress and NOT about danger -- it sets the ENCODE-vs-RETRIEVE dynamical mode
    (feedforward-vs-recurrent balance + plasticity gate), driven by NOVELTY.
  - valence_stdp.py (PHASIC DOPAMINE): a reward-PREDICTION-ERROR that drives value learning
    (what is good/bad). ACh is not reward and carries no valence -- it gates HOW the network
    processes input: write new (encode) vs recall old (retrieve).

Demo store: a tiny self-contained Hopfield-ish Hebbian recurrent associative memory over N bipolar
units. Under HIGH ACh feedforward clamps a pattern and plasticity writes it while recurrent gain is
low (clean, interference-free write); under LOW ACh feedforward is attenuated and the recurrent
weights (scaled by recurrent_gain) complete a PARTIAL cue back to the stored pattern (recall).

The ceiling is a single scalar ACh level (a leaky EMA of a novelty signal) gating a tiny
toy recurrent store on synthetic bipolar patterns -- a reduced model of cholinergic encode/retrieve
dynamics, NOT a cortical circuit, NOT cognition/AGI. Additive/opt-in: a new neuromodulator that
changes no existing behaviour. Credit Hasselmo (Neuromodulation: acetylcholine and memory
consolidation, TiCS 3:351, 1999; The role of acetylcholine in learning and memory, Curr Opin
Neurobiol 2006; encode/retrieve scheduling, J Neurosci 33:8689, 2013).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.acetylcholine
"""

import math
import random

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "RecurrentStore.encode": (
        "LOCAL",
        "the HIGH-ACh write: feedforward clamps the pattern while the recurrent path is suppressed, "
        "then the local Hebbian write lands. ACh gates WHEN it writes, not what error it follows",
    ),
    "Acetylcholine.plasticity_scale": (
        "NOT_A_RULE",
        "returns a learning GATE that rises with ACh. It is a scalar a rule may multiply by -- a "
        "modulator, not a rule; nothing here updates a weight",
    ),
    "_learn_expect": (
        "NOT_A_RULE",
        "EMAs an expectation vector toward each observation. A running average over observations, "
        "caught on the name",
    ),
}

TAU = 10.0  # ACh time constant (steps) -- how fast it tracks the novelty drive
MODE_TH = 0.5  # ACh >= this -> ENCODE, else RETRIEVE
FF_MIN, FF_MAX = 0.2, 1.0  # feedforward (afferent) gain range: attenuated when ACh is low
REC_MAX = 1.0  # recurrent (feedback) gain when ACh is fully low
PLAST_MAX = 1.0  # plasticity scale when ACh is fully high


class Acetylcholine:
    """Tonic ACh level in [0,1]: a leaky EMA that rises on novelty and decays toward baseline.
    High ACh -> ENCODE (feedforward + plasticity up, recurrent down); low ACh -> RETRIEVE."""

    def __init__(self, tau=TAU, baseline=0.0):
        self.decay = math.exp(-1.0 / tau)
        self.baseline = baseline
        self.level = baseline

    def step(self, novelty):
        """Advance one tick. `novelty` in [0,1] (mismatch / arousal) drives ACh up; with novelty
        near baseline the level decays back down (leaky EMA). Returns the new ACh level."""
        drive = max(self.baseline, min(1.0, novelty))
        self.level = self.decay * self.level + (1.0 - self.decay) * drive
        return self.level

    # ---- how ACh reshapes the encode/retrieve balance -----------------------
    def feedforward_gain(self):
        """Afferent/input gain -- INCREASES with ACh (encode: external input enhanced)."""
        return FF_MIN + (FF_MAX - FF_MIN) * self.level

    def recurrent_gain(self):
        """Feedback/recurrent gain -- INCREASES as ACh DROPS (retrieve: reactivate stored
        associations); suppressed toward 0 when ACh is high (clean write)."""
        return REC_MAX * (1.0 - self.level)

    def plasticity_scale(self):
        """Learning gate -- INCREASES with ACh (encode: write new; low in retrieve)."""
        return PLAST_MAX * self.level

    def mode(self):
        return "ENCODE" if self.level >= MODE_TH else "RETRIEVE"


def _sign(x):
    return 1 if x >= 0.0 else -1


class RecurrentStore:
    """Tiny Hopfield-ish Hebbian recurrent associative memory over `n` bipolar (+1/-1) units.
    encode() writes a clamped pattern into a recurrent weight matrix; retrieve() lets the recurrent
    weights complete a partial cue. ACh gates both: high ACh -> clean write, low ACh -> recall."""

    def __init__(self, n):
        self.n = n
        self.w = [[0.0] * n for _ in range(n)]

    def encode(self, pattern, ach):
        """HIGH-ACh write: feedforward clamps `pattern` while the recurrent path is suppressed
        (recurrent_gain low), so the clamped state is the clean input -- writing it Hebbianly does
        not bind in reactivated old memories. Returns the clamped state that was written."""
        g, ff, lr = ach.recurrent_gain(), ach.feedforward_gain(), ach.plasticity_scale()
        clamped = []
        for i in range(self.n):
            rec = sum(self.w[i][j] * pattern[j] for j in range(self.n) if j != i)
            clamped.append(_sign(ff * pattern[i] + g * rec))  # recurrent suppressed when ACh high
        for i in range(self.n):
            ci = clamped[i]
            for j in range(self.n):
                if i != j:
                    self.w[i][j] += lr * ci * clamped[j]
        return clamped

    def retrieve(self, cue, ach, steps=6):
        """LOW-ACh recall: feedforward attenuated (ff low), recurrent dominates (recurrent_gain
        high) -> the stored weights complete `cue` back to the nearest stored pattern."""
        g, ff = ach.recurrent_gain(), ach.feedforward_gain()
        state = list(cue)
        for _ in range(steps):
            nxt = []
            for i in range(self.n):
                rec = sum(self.w[i][j] * state[j] for j in range(self.n) if j != i)
                nxt.append(_sign(g * rec + ff * cue[i]))
            state = nxt
        return state


def _accuracy(a, b):
    return sum(1 for i in range(len(a)) if a[i] == b[i]) / len(a)


def _pattern(n, rng):
    return [1 if rng.random() < 0.5 else -1 for _ in range(n)]


def _corrupt(pattern, rng, frac=0.28):
    """Flip a fraction of bits -> a partial/degraded cue."""
    p = list(pattern)
    idx = list(range(len(p)))
    rng.shuffle(idx)
    for k in idx[: int(frac * len(p))]:
        p[k] = -p[k]
    return p


def _novelty(expect, x):
    """Mismatch signal: high (~1) when the input is unpredicted, decays to ~0 as `expect` learns it.
    conf = how trained the expectation is; match = sign agreement -> novelty = 1 - conf*match."""
    conf = sum(abs(v) for v in expect) / len(expect)
    match = sum(1 for i in range(len(x)) if _sign(expect[i]) == x[i]) / len(x)
    return max(0.0, 1.0 - conf * match)


def _learn_expect(expect, x, rate=0.2):
    for i in range(len(x)):
        expect[i] = (1.0 - rate) * expect[i] + rate * x[i]


def demo_encode_retrieve(seed):
    """Headline: drive ACh HIGH and write pattern A; drop ACh LOW and recall A from a partial cue."""
    rng = random.Random(seed)
    n = 32
    a = _pattern(n, rng)
    store = RecurrentStore(n)
    ach = Acetylcholine(tau=8.0)

    for _ in range(40):  # sustained novelty -> ACh climbs into ENCODE mode
        ach.step(1.0)
    ach_hi = ach.level
    store.encode(a, ach)  # clean, recurrent-suppressed write of A

    for _ in range(60):  # sustained quiescence -> ACh decays into RETRIEVE mode
        ach.step(0.0)
    ach_lo = ach.level
    cue = _corrupt(a, rng)
    recalled = store.retrieve(cue, ach)
    return {
        "ach_hi": ach_hi,
        "ach_lo": ach_lo,
        "cue_acc": _accuracy(cue, a),
        "recall_acc": _accuracy(recalled, a),
    }


def demo_interference(seed):
    """Encode A then B, both under HIGH ACh (clean writes, recurrent suppressed) -> writing B does
    not corrupt recall of A."""
    rng = random.Random(seed)
    n = 32
    a, b = _pattern(n, rng), _pattern(n, rng)
    store = RecurrentStore(n)
    ach = Acetylcholine(tau=8.0)
    for _ in range(40):
        ach.step(1.0)
    store.encode(a, ach)
    store.encode(b, ach)
    for _ in range(60):
        ach.step(0.0)
    rec_a = store.retrieve(_corrupt(a, rng), ach)
    rec_b = store.retrieve(_corrupt(b, rng), ach)
    return {"a_acc": _accuracy(rec_a, a), "b_acc": _accuracy(rec_b, b)}


def demo_habituation(seed):
    """NOVELTY dynamics: present the SAME pattern repeatedly. ACh is HIGH on the novel first
    exposures and decays as the expectation learns the pattern (habituation)."""
    rng = random.Random(seed)
    n = 32
    x = _pattern(n, rng)
    expect = [0.0] * n
    ach = Acetylcholine(tau=6.0)
    trace, nov = [], []
    for _ in range(40):
        v = _novelty(expect, x)  # perceive
        nov.append(v)
        trace.append(ach.step(v))
        _learn_expect(expect, x)  # then habituate
    ach_novel = max(trace[:10])  # peak while still novel
    ach_familiar = trace[-1]  # after habituation
    return {
        "ach_novel": ach_novel,
        "ach_familiar": ach_familiar,
        "nov_first": nov[0],
        "nov_last": nov[-1],
    }


def _mode_switch():
    """ACh high vs low -> the encode/retrieve gains flip."""
    hi = Acetylcholine(tau=8.0)
    for _ in range(40):
        hi.step(1.0)
    lo = Acetylcholine(tau=8.0)
    for _ in range(60):
        lo.step(0.0)
    return hi, lo


def main():
    hi, lo = _mode_switch()
    er = demo_encode_retrieve(0)
    hab = demo_habituation(1)
    intf = demo_interference(2)

    print("=" * 66)
    print("ACETYLCHOLINE -- novelty-driven ENCODE-vs-RETRIEVE mode switch")
    print("=" * 66)
    print(
        f"high ACh={hi.level:.2f} -> {hi.mode():8s} ff={hi.feedforward_gain():.2f} "
        f"rec={hi.recurrent_gain():.2f} plast={hi.plasticity_scale():.2f}"
    )
    print(
        f"low  ACh={lo.level:.2f} -> {lo.mode():8s} ff={lo.feedforward_gain():.2f} "
        f"rec={lo.recurrent_gain():.2f} plast={lo.plasticity_scale():.2f}"
    )
    print(
        f"encode-then-retrieve: wrote A at ACh={er['ach_hi']:.2f}, recalled at ACh={er['ach_lo']:.2f}"
    )
    print(f"  partial cue acc={er['cue_acc']:.2f} -> recall acc={er['recall_acc']:.2f}")
    print(
        f"novelty: ACh(novel)={hab['ach_novel']:.2f} > ACh(familiar)={hab['ach_familiar']:.2f} "
        f"(novelty {hab['nov_first']:.2f} -> {hab['nov_last']:.2f})"
    )
    print(
        f"interference: recall A={intf['a_acc']:.2f}, recall B={intf['b_acc']:.2f} after writing both"
    )
    print("=" * 66)

    # ---- self-checks (deterministic, ASCII-only) ----------------------------
    # 1. mode switch: gains flip with ACh
    assert hi.mode() == "ENCODE", "sustained novelty should put ACh in ENCODE mode"
    assert lo.mode() == "RETRIEVE", "sustained quiescence should put ACh in RETRIEVE mode"
    assert hi.feedforward_gain() > lo.feedforward_gain(), (
        "feedforward gain must be higher in encode"
    )
    assert hi.recurrent_gain() < lo.recurrent_gain(), "recurrent gain must be higher in retrieve"
    assert hi.plasticity_scale() > lo.plasticity_scale(), "plasticity must be higher in encode"
    # 2. encode-then-retrieve (headline): write in encode mode, recall in retrieve mode
    assert er["ach_hi"] >= MODE_TH, "ACh must be high (ENCODE) at write time"
    assert er["ach_lo"] < MODE_TH, "ACh must be low (RETRIEVE) at recall time"
    assert er["recall_acc"] >= 0.8, (
        f"recall of A from a partial cue must be high ({er['recall_acc']})"
    )
    assert er["recall_acc"] > er["cue_acc"], "retrieval must improve on the degraded cue"
    # 3. novelty dynamics: habituation
    assert hab["ach_novel"] > hab["ach_familiar"], "ACh must be higher on novel than familiar input"
    assert hab["nov_first"] > hab["nov_last"], "novelty signal must habituate on repetition"
    # 4. interference: high-ACh writes stay clean -> A survives writing B
    assert intf["a_acc"] >= 0.8, f"recall of A must survive encoding B ({intf['a_acc']})"
    assert intf["b_acc"] >= 0.8, f"recall of B must be high ({intf['b_acc']})"
    # 5. determinism: same seed/input -> identical results across fresh instances
    assert demo_encode_retrieve(0) == demo_encode_retrieve(0), (
        "encode/retrieve must be deterministic"
    )
    assert demo_habituation(1) == demo_habituation(1), "habituation trace must be deterministic"

    print(
        f"self-check OK: ACh novel={hab['ach_novel']:.2f} vs familiar={hab['ach_familiar']:.2f}; "
        f"mode flips ENCODE ff={hi.feedforward_gain():.2f}/rec={hi.recurrent_gain():.2f} -> "
        f"RETRIEVE ff={lo.feedforward_gain():.2f}/rec={lo.recurrent_gain():.2f}; "
        f"encode-then-retrieve recall acc={er['recall_acc']:.2f}"
    )


if __name__ == "__main__":
    main()
