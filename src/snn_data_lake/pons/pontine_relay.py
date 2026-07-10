"""
Pontine nuclei -- the cortico-ponto-cerebellar mossy-fibre EXPANSION relay.

Motor cortex does not wire straight to the cerebellum: its efference copy is relayed through the
PONTINE NUCLEI, which fan a low-dimensional cortical command out onto a vast sheet of mossy fibres
that drive the cerebellar granule cells. Marr-Albus and the modern expansion-coding view read this
as a fixed random EXPANSION: a small cortical vector is projected up into a much higher-dimensional,
SPARSE code (few active mossy/granule lines), which makes downstream (Purkinje) learning easy while
KEEPING the inputs separable. The relay's job is to be INFORMATION-PRESERVING -- the cerebellum must
still be able to tell one cortical command from another after the expansion.

Mechanism (rate-level, deterministic):
  1. A d-dim cortical motor-efference vector arrives at the pons.
  2. A FIXED random projection expands it to m = d * expansion mossy lines; each line reads a sparse
     random sample of the cortical dims and sums them (sparse divergent pontocerebellar connectivity).
  3. Winner-take-all thresholding keeps the top-k most active lines -> a sparse binary mossy code
     (the set of active line indices). This is the granule-layer sparse expansion code.
  4. The cerebellum reads the code by nearest-code match against the enrolled command codes.

Fair control (un-riggable -- SAME cortical inputs, SAME expansion size and sparsity in both relays):
  (a) FAITHFUL relay : the random-expansion code above. N distinct cortical commands map to N
      pairwise-DISTINGUISHABLE mossy codes (every pair of codes stays apart by a distance margin),
      and a plain nearest-code readout recovers the correct command from a jittered probe.
  (b) COLLAPSED relay: a degenerate expansion in which every mossy line reads the SAME cortical
      sample, so all lines share one activation and WTA returns ONE constant code for EVERY input
      (all-inputs-to-one-code). Pairwise code distance collapses to zero and the readout falls to
      chance. Same expansion budget, same sparsity -- only the projection is collapsed.
  Faithful must PRESERVE distinctions (high accuracy, codes apart); collapsed must LOSE them
  (chance accuracy, codes identical) -- proving the win is a faithful expansion, not just "more units".

Honest framing: a reduced, rate-level model of pontine mossy-fibre expansion coding on synthetic
cortical vectors. It is a feed-forward relay/encoder FUNCTION -- it commands no motion, is not in the
arm loop, and models information relay, not cognition/awareness.
Credit: Marr 1969; Albus 1971; Cayco-Gajic & Silver 2019 (expansion recoding);
Dasgupta, Stevens & Navlakha 2017 (random-expansion sparse tags).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.pons.pontine_relay
"""
import random

D_CORTEX = 24        # cortical motor-efference dimensionality
EXPANSION = 16       # mossy-fibre fan-out factor (m = D_CORTEX * EXPANSION)
WTA = 0.05           # fraction of mossy lines kept active (sparse granule code)
SAMPLES = 6          # cortical dims each mossy line reads (sparse divergent projection)
N_CLASSES = 10       # distinct cortical commands to relay
PROBES = 30          # jittered probes per command for the nearest-code readout
JITTER = 0.10        # per-dim cortical noise on a probe


class PontineRelay:
    """Fixed random EXPANSION + WTA -> a sparse binary mossy-fibre code (set of active line indices).

    With collapse=True the projection is degenerate (every line reads one shared cortical sample),
    so every input maps to a single constant code -- the un-riggable collapsed control.
    """

    def __init__(self, d=D_CORTEX, expansion=EXPANSION, wta=WTA, samples=SAMPLES,
                 collapse=False, seed=0):
        self.d = d
        self.m = d * expansion
        self.k = max(1, round(self.m * wta))
        rng = random.Random(seed)
        s = min(samples, d)
        if collapse:
            shared = rng.sample(range(d), s)               # one shared sample for EVERY line
            self.proj = [shared for _ in range(self.m)]     # degenerate: all lines read the same dims
        else:
            self.proj = [rng.sample(range(d), s) for _ in range(self.m)]

    def encode(self, x):
        """Cortical vector -> sparse mossy code. Each line sums its sampled cortical dims; the
        top-k lines win. Deterministic tie-break by lowest index (matches fly_hash / hippocampus)."""
        act = [sum(x[i] for i in idx) for idx in self.proj]
        winners = sorted(range(self.m), key=lambda j: (act[j], -j), reverse=True)[:self.k]
        return frozenset(winners)


def code_distance(a, b):
    """Symmetric-difference (Hamming) distance between two sparse binary codes, in active lines."""
    return len(a ^ b)


def min_pairwise_distance(codes):
    """Smallest distance between any two enrolled command codes -- the separability margin."""
    return min(code_distance(codes[i], codes[j])
               for i in range(len(codes)) for j in range(i + 1, len(codes)))


def make_commands(n, d, seed):
    """n distinct cortical command prototypes (each a d-dim vector in [0, 1])."""
    rng = random.Random(seed)
    return [[rng.random() for _ in range(d)] for _ in range(n)]


def jitter(vec, amount, rng):
    """A noisy cortical probe of a command: each dim perturbed, clamped to [0, 1]."""
    return [min(1.0, max(0.0, v + rng.uniform(-amount, amount))) for v in vec]


def readout_accuracy(relay, commands, probes, jitter_amt, seed):
    """Nearest-code readout: enroll one code per command, then classify jittered probes by the
    closest (max code-overlap) enrolled code. Returns (accuracy, enrolled_codes)."""
    enrolled = [relay.encode(c) for c in commands]
    rng = random.Random(seed)
    hits = total = 0
    for label, cmd in enumerate(commands):
        for _ in range(probes):
            code = relay.encode(jitter(cmd, jitter_amt, rng))
            pred = max(range(len(enrolled)), key=lambda j: len(code & enrolled[j]))
            hits += pred == label
            total += 1
    return hits / total, enrolled


def main():
    commands = make_commands(N_CLASSES, D_CORTEX, seed=1)

    faithful = PontineRelay(seed=0)
    collapsed = PontineRelay(collapse=True, seed=0)

    acc_f, codes_f = readout_accuracy(faithful, commands, PROBES, JITTER, seed=2)
    acc_c, codes_c = readout_accuracy(collapsed, commands, PROBES, JITTER, seed=2)

    dmin_f = min_pairwise_distance(codes_f)
    dmin_c = min_pairwise_distance(codes_c)
    chance = 1.0 / N_CLASSES
    margin = faithful.k                                    # require class codes apart by >= k lines

    print("=" * 68)
    print("PONTINE RELAY -- cortico-ponto-cerebellar mossy-fibre expansion")
    print("=" * 68)
    print(f"cortical dim = {faithful.d}, mossy-fibre dim = {faithful.m} (x{EXPANSION} expansion), "
          f"code = {faithful.k}/{faithful.m} active lines")
    print(f"commands = {N_CLASSES}, probes/command = {PROBES}, jitter = {JITTER:.2f}, "
          f"chance = {chance:.0%}")
    print("-" * 68)
    print(f"(a) FAITHFUL relay  : nearest-code accuracy = {acc_f:.0%}   "
          f"[{N_CLASSES}-way, chance {chance:.0%}]")
    print(f"    min pairwise code distance between commands = {dmin_f} lines  (margin {margin})")
    print(f"(b) COLLAPSED relay : nearest-code accuracy = {acc_c:.0%}   "
          "[all inputs -> one code]")
    print(f"    min pairwise code distance between commands = {dmin_c} lines  (indistinguishable)")
    print("=" * 68)

    # ---- self-check: deterministic asserts, zero-dep, ASCII-only ----
    # (1) faithful expansion PRESERVES the commands: probes decode far above chance.
    assert acc_f >= 0.9, f"faithful relay must decode commands ({acc_f:.2f} vs chance {chance:.2f})"
    # (2) faithful codes stay pairwise separable by the margin.
    assert dmin_f >= margin, f"command codes must stay apart by >= {margin} lines (min {dmin_f})"
    # (3) COLLAPSED control loses the distinctions: every input -> one constant code.
    assert dmin_c == 0, f"collapsed relay must map all commands to one code (min dist {dmin_c})"
    assert acc_c <= 1.5 * chance, f"collapsed relay must fall to chance ({acc_c:.2f} vs {chance:.2f})"
    assert acc_f > 3.0 * acc_c, f"faithful must markedly beat collapsed ({acc_f:.2f} vs {acc_c:.2f})"

    # (4) determinism: same seed -> identical projection, codes, data and accuracy.
    r1, r2 = PontineRelay(seed=0), PontineRelay(seed=0)
    assert r1.encode(commands[0]) == r2.encode(commands[0]), "same seed -> identical mossy code"
    assert make_commands(N_CLASSES, D_CORTEX, seed=1) == commands, "same seed -> identical commands"
    acc_f2, _ = readout_accuracy(PontineRelay(seed=0), commands, PROBES, JITTER, seed=2)
    assert acc_f2 == acc_f, "same seed -> identical readout accuracy"

    print(f"self-check OK: faithful expansion decodes {acc_f:.0%} ({N_CLASSES}-way, chance "
          f"{chance:.0%}), codes apart >= {dmin_f} lines; collapsed relay -> one code "
          f"({acc_c:.0%}, dist 0); expansion preserves the commands; deterministic")


if __name__ == "__main__":
    main()
