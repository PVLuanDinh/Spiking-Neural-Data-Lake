"""
Three-factor delay learning — learn the TIMING, not the weight (scan #3).

The thesis is "reason in spike TIMING", yet every trainable piece in the repo (LSM, deep-SSM, gesture
recognizer) learns WEIGHTS. Making a synaptic DELAY a first-class learnable parameter is the most
on-thesis training upgrade: a coincidence neuron with a learnable per-input delay can classify temporal
ORDER (ch0-before-ch1 vs ch1-before-ch0) — exactly the primitive Paradigm B's temporal-order query
exploits — which a zero-delay coincidence detector cannot. The delay is trained by a THREE-FACTOR rule:
Δd = lr · eligibility(pre−post timing mismatch) · global reward. Refs: arXiv:2601.00668 (2026, "Three
factor delay learning rules for spiking neural networks"; +20% over weights-only, SHD BPTT-parity).

  Run: python delay_learning.py       # (also run by run_selfchecks.py)
"""

import random

random.seed(0)

T = 50  # time window
GAP = 8  # true lead/lag between the two channels
W = 4.0  # coincidence window (triangular kernel half-width)
LR = 0.3  # delay learning rate
EPOCHS = 15


def sample(cls, rng):
    """Two channels; class 0 = ch0 leads ch1 by GAP, class 1 = ch1 leads. Returns (t_ch0, t_ch1)."""
    t = rng.randrange(6, T - GAP - 6)
    j = rng.uniform(-1.0, 1.0)  # jitter on the gap
    return (float(t), t + GAP + j) if cls == 0 else (t + GAP + j, float(t))


def coincidence(s, d0, d1):
    """Triangular coincidence of the DELAYED spikes: 1 when (t0+d0)==(t1+d1), 0 past ±W."""
    return max(0.0, 1.0 - abs((s[0] + d0) - (s[1] + d1)) / W)


def balanced(n, seed):
    rng = random.Random(seed)
    return [(c, sample(c, rng)) for _ in range(n // 2) for c in (0, 1)]


def train_delay(train):
    """Learn d0 (fix d1=0) via the three-factor rule on the target (class-0) samples only."""
    d0 = 0.0
    for _ in range(EPOCHS):
        for cls, s in train:
            if cls == 0:  # reward class: drive the delayed spikes to coincide
                elig = s[1] - (s[0] + d0)  # pre−post timing mismatch (eligibility)
                d0 += LR * elig * 1.0  # × global reward (+1)
    return d0


def accuracy(data, d0):
    ok = 0
    for cls, s in data:
        pred = 0 if coincidence(s, d0, 0.0) > 0.5 else 1
        ok += pred == cls
    return ok / len(data)


def main():
    train, test = balanced(80, 1), balanced(60, 2)
    pre = accuracy(test, 0.0)  # no delay learned yet
    d0 = train_delay(train)
    post = accuracy(test, d0)

    print("=" * 58)
    print("THREE-FACTOR DELAY LEARNING (temporal-order classification)")
    print("=" * 58)
    print(f"task: ch0-leads vs ch1-leads by GAP={GAP} (jittered), coincidence window W={W}")
    print(f"learned delay d0 = {d0:.2f}  (target ~ GAP={GAP} to align the leading channel)")
    print(f"order-classification accuracy : pre {pre:.0%} -> post {post:.0%}  (chance 50%)")
    print("delay is a LEARNED parameter (three-factor: eligibility × reward) — timing, not weight.")
    print("=" * 58)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert pre <= 0.6, f"a zero-delay coincidence detector can't classify order: pre={pre:.2f}"
    assert post >= 0.9, f"learned delay should classify temporal order: post={post:.2f}"
    assert abs(d0 - GAP) <= 1.5, f"learned delay {d0:.2f} should converge near GAP={GAP}"
    print(
        f"self-check OK: delay learned to {d0:.1f}~{GAP}; order accuracy {pre:.0%}->{post:.0%} "
        "(timing learned, not weight)"
    )


if __name__ == "__main__":
    main()
