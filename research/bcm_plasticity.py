"""
BCM plasticity (Bienenstock-Cooper-Munro) — selectivity development via a sliding threshold
(learning axis). Pure-stdlib.

The repo's learners are STDP (timing), tempotron (supervised timing), delay/homeostatic. Missing is
the classic RATE-based rule that explains how a neuron becomes SELECTIVE to one input pattern —
BCM (1982), the model of ocular dominance and orientation selectivity. Its trick is a sliding
modification threshold theta that tracks the neuron's own recent activity <y^2>: when the response
exceeds theta the synapse potentiates, below it depresses, and because theta chases activity the
neuron self-organizes toward responding to a SINGLE pattern while suppressing the rest — stable,
competitive feature learning without a supervisor. Refs: Bienenstock, Cooper & Munro, "Theory for
the development of neuron selectivity", J. Neurosci. 2 (1982); Cooper & Bear, Nat. Rev. Neurosci. 13
(2012).

  Run: python bcm_plasticity.py        # (also run by run_selfchecks.py)
"""
import random


class BCMNeuron:
    """Rate neuron with the BCM rule: dw = lr * x * y * (y - theta), theta tracks <y^2>."""

    def __init__(self, d, lr=0.02, tau_theta=40.0, seed=0):
        rng = random.Random(seed)
        self.w = [rng.uniform(0.05, 0.15) for _ in range(d)]
        self.theta = 0.05
        self.lr, self.tau_theta = lr, tau_theta

    def response(self, x):
        return max(0.0, sum(w * xi for w, xi in zip(self.w, x)))

    def update(self, x):
        y = self.response(x)
        phi = y * (y - self.theta)                     # BCM plasticity function (LTP above theta, LTD below)
        for i in range(len(self.w)):
            self.w[i] = min(2.0, max(0.0, self.w[i] + self.lr * x[i] * phi))
        self.theta += (y * y - self.theta) / self.tau_theta   # sliding threshold chases recent activity
        return y


def _patterns(d, k):
    """k block-patterns: pattern j drives inputs [2j, 2j+1] strongly, the rest weakly."""
    pats = []
    for j in range(k):
        p = [0.1] * d
        p[2 * j] = p[2 * j + 1] = 1.0
        pats.append(p)
    return pats


def _selectivity(resp):
    """1 = responds to exactly one pattern, 0 = responds equally to all (winner vs mean)."""
    top = max(resp)
    return (top - sum(resp) / len(resp)) / top if top > 0 else 0.0


def main():
    k, d = 4, 8
    pats = _patterns(d, k)
    neuron = BCMNeuron(d, seed=0)

    resp0 = [neuron.response(p) for p in pats]
    sel0 = _selectivity(resp0)

    rng = random.Random(1)
    for _ in range(4000):
        neuron.update(pats[rng.randrange(k)])          # random pattern presentation

    resp1 = [neuron.response(p) for p in pats]
    sel1 = _selectivity(resp1)
    winner = max(range(k), key=lambda j: resp1[j])

    print("=" * 60)
    print("BCM PLASTICITY (Bienenstock-Cooper-Munro) — selectivity via sliding threshold")
    print("=" * 60)
    print(f"{k} input patterns, d={d}, 4000 presentations")
    print(f"responses before : {[round(r, 2) for r in resp0]}  selectivity {sel0:.2f}")
    print(f"responses after  : {[round(r, 2) for r in resp1]}  selectivity {sel1:.2f}")
    print(f"neuron became selective to pattern {winner}; theta settled at {neuron.theta:.2f}")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert sel1 > 0.6, f"BCM must develop selectivity to one pattern ({sel1:.2f})"
    assert sel1 > sel0 + 0.2, f"selectivity must sharpen with training ({sel0:.2f} -> {sel1:.2f})"
    assert resp1[winner] > 2 * max(r for j, r in enumerate(resp1) if j != winner), \
        "the winning pattern must dominate the others (>2x)"
    assert 0 < neuron.theta < 100 and all(0 <= w <= 2 for w in neuron.w), "theta + weights stay bounded"
    assert BCMNeuron(d, seed=0).w == BCMNeuron(d, seed=0).w, "same seed -> identical init (deterministic)"
    print(f"self-check OK: selectivity {sel0:.2f}->{sel1:.2f}, winner pattern {winner} dominates; "
          f"theta bounded {neuron.theta:.2f}; deterministic")


if __name__ == "__main__":
    main()
