"""
Spiking Neural P system (SN P system) -- membrane computing meets spiking neurons.

Ionescu, Paun & Yokomori (2006) placed the P-systems (membrane-computing) formalism onto
SPIKING neurons: each neuron (a membrane) holds a single nonnegative INTEGER -- a count of
identical spikes `a` -- and carries firing rules of the form

    E / a^c -> a^p ; d

read as: when the neuron's spike count matches the regular-expression predicate E, the rule
CONSUMES c spikes, PRODUCES p spikes onto every outgoing synapse, and imposes a refractory
DELAY d during which the neuron is CLOSED (steps t+1..t+d): it fires nothing and DROPS every
spike that arrives. Computation is MAXIMALLY PARALLEL across neurons but at most one rule may
fire per neuron per step; the machine's output is the spike train (the emission times) of a
distinguished output neuron. The answer is carried by WHEN the output neuron fires -- on
thesis with this repo's "reason in spike timing" line.

Determinism choice (why this port differs from the textbook machine): the canonical SN P
example in the paper is a NONDETERMINISTIC generator -- two rules sharing the SAME predicate
compete inside one neuron, so the machine may emit ANY natural number / an arbitrary interval
train. That nondeterminism is the entire point there, and it has no single "correct" result to
assert against. This repo's contract is deterministic, zero-RNG, emergent-property self-checks,
so the nondeterministic "generate all naturals" generator is DELIBERATELY NOT PORTED. Instead
the same engine (transcribed verbatim below) drives two DETERMINISTIC closed-form constructions
whose spike trains are provable:
  * build_adder(n, m)       -- an unary ADDER: the output neuron emits exactly n+m spikes.
  * build_delay_gen(dA, dB) -- a two-neuron A<->B loop whose output train is perfectly periodic
                               with interval dA+dB+2.
The engine enforces determinism STRUCTURALLY: step() asserts at most ONE rule is applicable per
neuron, so a nondeterministic configuration fails loudly instead of silently picking one.

Pure standard library (collections only), no randomness, ASCII-only output.

  PYTHONPATH=src python -m snn_data_lake.reasoning.sn_p_system   # (also run by run_selfchecks.py)

Paper: Ionescu, M., Paun, Gh., Yokomori, T. (2006), "Spiking Neural P Systems",
Fundamenta Informaticae 71(2-3):279-308.
"""

from collections import namedtuple, defaultdict

Rule = namedtuple("Rule", "pred consume produce delay")


def at_least(k):
    return lambda n: n >= k  # E = a^k (a)*


def exactly(k):
    return lambda n: n == k  # E = a^k


class SNPSystem:
    def __init__(self, counts, rules, synapses, out):
        self.count = list(counts)
        self.rules = rules  # rules[i] = list[Rule]
        self.succ = synapses  # {src: [dst, ...]}
        self.out = out
        self.open_at = [0] * len(counts)
        self.pending = defaultdict(lambda: defaultdict(int))
        self.t = 0
        self.train = []

    def step(self):
        t = self.t
        for i, s in self.pending.pop(t, {}).items():
            if t >= self.open_at[i]:
                self.count[i] += s
        firing = []
        for i, rs in enumerate(self.rules):
            if t < self.open_at[i]:
                continue
            hits = [r for r in rs if r.pred(self.count[i])]
            assert len(hits) <= 1, (
                f"nondeterministic: {len(hits)} rules applicable in neuron {i} at t={t}"
            )
            if hits:
                firing.append((i, hits[0]))
        for i, r in firing:
            self.count[i] -= r.consume
            self.open_at[i] = t + r.delay + 1
            if i == self.out:
                self.train.append(t + r.delay)
            for j in self.succ.get(i, ()):
                self.pending[t + r.delay + 1][j] += r.produce
        self.t += 1
        return bool(firing)

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.train


def build_adder(n, m):
    """Unary adder. in1=neuron 0 holds n spikes, in2=neuron 1 holds m; both drain into out=2,
    which forwards each received spike. out therefore emits exactly n+m spikes."""
    r = Rule(at_least(1), 1, 1, 0)  # a+/a -> a ; 0  (drain one spike per step)
    return SNPSystem([n, m, 0], [[r], [r], [r]], {0: [2], 1: [2]}, out=2)


def build_delay_gen(dA, dB):
    """Two-neuron oscillator. A=0 seeded with one spike fires into B=1, B fires back into A;
    the closed-neuron delay windows dA,dB set the loop period. out=A emits a periodic train
    whose consecutive intervals are exactly dA+dB+2."""
    rules = [[Rule(exactly(1), 1, 1, dA)], [Rule(exactly(1), 1, 1, dB)]]
    return SNPSystem([1, 0], rules, {0: [1], 1: [0]}, out=0)


def main():
    print("=" * 64)
    print("SPIKING NEURAL P SYSTEM -- membrane computing on spiking neurons")
    print("=" * 64)
    add = build_adder(2, 3).run(2 + 3 + 3)
    print(f"adder(2,3): out train {add} -> {len(add)} spikes == 2+3")
    e = build_delay_gen(1, 0).run(5 * (1 + 0 + 2))
    d = [b - a for a, b in zip(e, e[1:])]
    print(f"delay_gen(1,0): emissions {e}, intervals {d} == dA+dB+2 == 3")
    print("=" * 64)

    # ---- self-check (deterministic asserts, emergent -- nothing hardcoded) ----
    # 1. unary adder: the output neuron emits exactly n+m spikes for every operand pair
    for n, m in [(0, 0), (1, 0), (0, 3), (2, 3), (4, 1), (3, 3)]:
        assert len(build_adder(n, m).run(n + m + 3)) == n + m, f"adder {n}+{m}"
    # 2. delay generator: the output train is perfectly periodic with interval dA+dB+2
    for dA, dB in [(0, 0), (1, 0), (0, 2), (2, 1), (3, 3)]:
        period = dA + dB + 2
        emits = build_delay_gen(dA, dB).run(5 * period)
        diffs = [b - a for a, b in zip(emits, emits[1:])]
        assert diffs and all(x == period for x in diffs), (
            f"delay_gen({dA},{dB}) -> {emits} diffs {diffs}"
        )
    # 3. deterministic: no RNG anywhere -> identical result on a repeat run
    assert build_adder(2, 3).run(8) == build_adder(2, 3).run(8), "nondeterministic adder"
    print(
        "self-check OK: unary adder out == n+m for 6 operand pairs; delay-gen train "
        "periodic at dA+dB+2 for 5 delay pairs; deterministic (zero RNG)"
    )


if __name__ == "__main__":
    main()
