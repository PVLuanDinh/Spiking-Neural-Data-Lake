"""
BCM plasticity (Bienenstock-Cooper-Munro) — competitive selectivity on the classifier's shape inputs.

PROMOTED from research/ into the product (v0.87, additive). BCM's sliding modification threshold makes
a rate neuron self-organize to respond to ONE input pattern among several competing, DISTINCT ones
(the classic ocular-dominance / orientation-selectivity mechanism). It is genuinely UNSUPERVISED — a
single repeated pattern gets depressed, and near-identical patterns collapse it — so it needs
separable inputs. `develop_selectivity` presents several patterns to one BCM unit and returns the one
it became selective to; here it runs on `snn_classifier`'s distinct 8x8 shape templates (the trainable
classifier's real inputs). Refs:
Bienenstock, Cooper & Munro, J. Neurosci. 2 (1982); Cooper & Bear, Nat. Rev. Neurosci. 13 (2012).

WIRING (corrected 2026-08-04 -- "Additive/opt-in: nothing in the recognition path changes" was
stale). `BCMNeuron`'s sliding threshold theta is the metaplastic variable of
`plasticity_rule_stack._BCMMetaGate`, which gates the effective learning rate on the canonical
brain's interaction learner (`engine/brain.py:273-278`). The RECOGNITION path is indeed untouched;
the LEARNING path is not. Measured charge is in `homeostatic_scaling.py`'s wiring note.

  python -m snn_data_lake.plasticity.bcm_plasticity        # (also run by run_selfchecks.py)
"""

import random

import snn_data_lake.readout.snn_classifier as C

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "BCMNeuron.update": (
        "LOCAL",
        "BCM: a sliding-threshold Hebbian rule on this neuron's own pre/post activity. The threshold "
        "is the neuron's OWN running average, which is what makes it metaplastic rather than "
        "gradient-driven",
    ),
}


class BCMNeuron:
    """Rate neuron with the BCM rule: dw = lr * x * y * (y - theta), theta tracks <y^2>."""

    def __init__(self, d, lr=0.5, tau_theta=40.0, seed=0):
        rng = random.Random(seed)
        self.w = [rng.uniform(0.05, 0.15) for _ in range(d)]
        self.theta = 0.05
        self.lr, self.tau_theta = lr, tau_theta

    def response(self, x):
        return max(0.0, sum(w * xi for w, xi in zip(self.w, x)))

    def update(self, x):
        y = self.response(x)
        phi = y * (y - self.theta)
        for i in range(len(self.w)):
            self.w[i] = min(2.0, max(0.0, self.w[i] + self.lr * x[i] * phi))
        self.theta += (y * y - self.theta) / self.tau_theta
        return y


def _selectivity(responses):
    """1 = responds to exactly one pattern, 0 = responds equally (winner vs mean)."""
    top = max(responses)
    return (top - sum(responses) / len(responses)) / top if top > 0 else 0.0


def develop_selectivity(patterns, presentations=3000, seed=0):
    """Present several DISTINCT patterns (in random order) to one BCM unit; via the sliding threshold
    it self-organizes to respond to a SINGLE one. `patterns` is a dict name->vector (magnitude-normalized
    so BCM's threshold stays well-scaled). Returns (neuron, responses_by_name, winner)."""
    names = list(patterns)
    d = len(patterns[names[0]])
    neuron = BCMNeuron(d, seed=seed)
    rng = random.Random(seed)
    for _ in range(presentations):
        base = patterns[names[rng.randrange(len(names))]]
        jit = [min(1.0, max(0.0, v + rng.uniform(-0.02, 0.02))) for v in base]
        neuron.update(jit)
    responses = {nm: neuron.response(patterns[nm]) for nm in names}
    return neuron, responses, max(responses, key=responses.get)


def _shape_patterns():
    """The classifier's four distinct 8x8 shapes (vbar / hbar / diag / box), magnitude-normalized
    (L1=1) so the BCM sliding threshold stays well-scaled and develops healthy selectivity."""
    out = {}
    for i, t in enumerate(C.templates()):
        s = sum(t) or 1.0
        out[f"shape{i}"] = [float(v) / s for v in t]
    return out


def main():
    patterns = _shape_patterns()
    names = list(patterns)
    n0 = BCMNeuron(len(patterns[names[0]]), seed=0)
    resp_before = [n0.response(patterns[nm]) for nm in names]
    sel_before = _selectivity(resp_before)

    neuron, responses, winner = develop_selectivity(patterns)
    resp_after = [responses[nm] for nm in names]
    sel_after = _selectivity(resp_after)

    print("=" * 60)
    print("BCM SELECTIVITY -- one unit self-organizes to a single shape input")
    print("=" * 60)
    print(f"{len(names)} distinct classifier shapes")
    print(f"responses before : {[round(r, 2) for r in resp_before]}  selectivity {sel_before:.2f}")
    print(
        f"responses after  : {[round(responses[n], 2) for n in names]}  selectivity {sel_after:.2f}"
    )
    print(f"unit became selective to '{winner}'; theta settled at {neuron.theta:.2f}")
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert sel_after > 0.5, f"BCM must develop selectivity to one shape ({sel_after:.2f})"
    assert sel_after > sel_before + 0.2, (
        f"selectivity must sharpen with exposure ({sel_before:.2f} -> {sel_after:.2f})"
    )
    assert responses[winner] > 1.5 * max(responses[n] for n in names if n != winner), (
        "the selected shape must dominate the others"
    )
    assert 0 < neuron.theta < 100 and all(0 <= w <= 2 for w in neuron.w), (
        "theta + weights stay bounded"
    )
    assert develop_selectivity(patterns)[2] == winner, (
        "same seed -> same selected shape (deterministic)"
    )
    print(
        f"self-check OK: selectivity {sel_before:.2f}->{sel_after:.2f}, unit selective to '{winner}'; "
        f"BCM competitive selectivity on the classifier shapes; additive; deterministic"
    )


if __name__ == "__main__":
    main()
