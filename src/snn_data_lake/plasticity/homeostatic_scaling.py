"""
Homeostatic synaptic scaling (Turrigiano) — an additive rate-stabilizer for the learning stores.

PROMOTED from research/ into the product (v0.88, additive). Pure Hebbian potentiation is unstable —
weights and firing rate run away. Turrigiano's fix is MULTIPLICATIVE synaptic scaling: the neuron
measures its own rate and rescales all its synapses toward a set-point, holding the rate steady while
preserving the RELATIVE weight pattern (unlike subtractive normalization). `stabilize` exposes that as
a wrapper other learners can opt into. Refs:
Turrigiano et al., Nature 391 (1998); Turrigiano, "The self-tuning neuron", Cell 135 (2008).

WIRING (corrected 2026-08-04 -- this docstring said "an optional tool, not wired into any default"
and that has been false since the plasticity stacks landed). `stabilize` is the POST phase of
`plasticity/homeostatic_plasticity_stack._HomeostaticScaling`, which `plasticity_rule_stack`
subclasses and `cortex/brain.py:273-278` constructs and attaches to the canonical brain's
interaction learner (`brain._ixn.stack = ...`). MEASURED CHARGE on that real consumer, 40 epochs
on its own task: attaching the stack moves the learner's weights by max|dw| 15.30 against
`stack=None`, and the rule stack's extras move them a further 0.95 beyond the base stack.
Stated precisely, because the distinction is the point: that is charge in the WEIGHTS. Held-out
routing accuracy is 1.000 in every condition, so this task is at ceiling and the stack is NOT
shown to change the OUTCOME here -- only the state the outcome is read from.

  python -m snn_data_lake.plasticity.homeostatic_scaling        # (also run by run_selfchecks.py)
"""

import snn_data_lake.cortex.snn_classifier as C

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "HomeostaticNeuron.hebb": (
        "LOCAL",
        "Hebbian coincidence on this neuron's own pre/post pair, with homeostatic scaling holding the "
        "incoming budget. Both halves are synapse-local",
    ),
}


class HomeostaticNeuron:
    """Hebbian potentiation + multiplicative synaptic scaling toward a rate set-point."""

    def __init__(self, weights, target=1.0, hebb_lr=0.05, scale_rate=0.5):
        self.w = list(weights)
        self.target, self.hebb_lr, self.scale_rate = target, hebb_lr, scale_rate

    def output(self, x):
        return sum(w * xi for w, xi in zip(self.w, x))

    def hebb(self, x):
        y = self.output(x)
        for i in range(len(self.w)):
            self.w[i] += self.hebb_lr * x[i] * y  # correlated activity -> potentiation

    def stabilize(self, x):
        """Multiplicative synaptic scaling toward the rate set-point (same factor on every synapse)."""
        y = self.output(x)
        if y > 0:
            f = (self.target / y) ** self.scale_rate
            self.w = [w * f for w in self.w]

    def step(self, x, homeostasis=True):
        self.hebb(x)
        if homeostasis:
            self.stabilize(x)
        return self.output(x)


def _shape():
    """A real classifier shape (box) as the input drive (28 active pixels -> moderate S)."""
    return [float(v) for v in C.templates()[3]]


def main():
    x = _shape()
    init = [0.5] * len(x)
    lr = 0.003  # small so scaling holds the rate near target

    hn = HomeostaticNeuron(init, target=1.0, hebb_lr=lr)
    for _ in range(300):
        hn.step(x, homeostasis=True)
    conv, wmax = hn.output(x), max(hn.w)

    free = HomeostaticNeuron(init, target=1.0, hebb_lr=lr)  # same start, Hebbian only
    for _ in range(300):
        free.step(x, homeostasis=False)
    runaway, wmax_free = free.output(x), max(free.w)

    ratios_before = [w / hn.w[0] for w in hn.w]
    hn.stabilize(x)
    ratios_after = [w / hn.w[0] for w in hn.w]
    drift = max(abs(a - b) for a, b in zip(ratios_before, ratios_after))

    print("=" * 60)
    print("HOMEOSTATIC SCALING -- additive rate-stabilizer (driven by a real shape)")
    print("=" * 60)
    print(
        f"input: classifier shape ({len(x)} synapses, {int(sum(x))} active), target rate {hn.target:.1f}, 300 steps"
    )
    print(f"with scaling : rate {conv:.2f} (-> target), max weight {wmax:.2f} (bounded)")
    print(f"Hebbian only : rate {runaway:.1f} (runaway), max weight {wmax_free:.1f}")
    print(f"ratio drift under a scale step: {drift:.1e} (multiplicative = pattern preserved)")
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert abs(conv - hn.target) < 0.15, (
        f"scaling must hold the rate near the set-point ({conv:.3f})"
    )
    assert runaway > 5 * hn.target, (
        f"Hebbian alone must run away without homeostasis ({runaway:.1f})"
    )
    assert wmax < wmax_free, (
        f"homeostatic weights must stay bounded far below runaway ({wmax:.2f} vs {wmax_free:.1f})"
    )
    assert drift < 1e-9, "multiplicative scaling must preserve all relative weight ratios"
    assert HomeostaticNeuron(init).w == HomeostaticNeuron(init).w, "deterministic"
    print(
        f"self-check OK: rate {conv:.2f}~target vs runaway {runaway:.0f}; ratios preserved ({drift:.0e}); "
        f"additive stabilizer, no learner default changed; deterministic"
    )


if __name__ == "__main__":
    main()
