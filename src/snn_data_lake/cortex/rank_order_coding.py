"""
Rank-order coding (Thorpe) — gain-invariant recognition for the arm/data-lake matcher.

PROMOTED from research/ into the product (v0.79): the live recognizer (signal_loop / learned_matcher)
matches a window to the nearest enrolled command by MAGNITUDE — Van Rossum distance on latency codes,
where latency tracks amplitude. That is exactly why arm_config.py warns "move the rig + the 95%
recognition collapses": a sensor gain / contrast / mounting change rescales the window and the
magnitude match drifts. Thorpe's rank-order code keeps only the ORDER afferents fire, discarding
magnitude, so it is INVARIANT to any monotonic gain change — the firing order is unchanged. This
module is that gain-robust matcher, enrolled on the same command library, and it proves the win by
beating the template matcher under a contrast change on the real library. Refs: Thorpe, Delorme &
VanRullen, Neural Networks 14 (2001).

  python -m snn_data_lake.cortex.rank_order_coding        # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake.brainstem.signal_loop import build_default_library
from snn_data_lake.cortex.spike_preprocessing import encode_latency, van_rossum_distance


def rank_code(x):
    """Descending rank per input: 0 = fires first (largest value). Depends only on ORDER."""
    order = sorted(range(len(x)), key=lambda i: x[i], reverse=True)
    rank = [0] * len(x)
    for pos, i in enumerate(order):
        rank[i] = pos
    return rank


class RankOrderMatcher:
    """Enroll command references as rank codes; classify a window by nearest rank agreement."""

    def __init__(self, library):
        self.labels = list(library)
        self.templates = [rank_code(library[lab]) for lab in self.labels]

    def label(self, window):
        rx = rank_code(window)
        agree = [-sum(abs(a - b) for a, b in zip(rx, t)) for t in self.templates]
        return self.labels[max(range(len(self.labels)), key=lambda i: agree[i])]


def _template_label(refs_enc, labels, window):
    """The baseline magnitude matcher: nearest Van Rossum distance on latency codes."""
    q = encode_latency(window)
    d = [van_rossum_distance(q, ev) for ev in refs_enc]
    return labels[min(range(len(labels)), key=lambda i: d[i])]


def _rig_gain(v, b, p):
    """A monotonic sensor gain/contrast change (order-preserving): [0,1] -> [b,1] via v**p.
    Strictly increasing, so rank order is preserved exactly while magnitudes shift — a rig move."""
    return b + (1.0 - b) * (v ** p)


def main():
    rng = random.Random(1)
    library = build_default_library()
    labels = list(library)
    refs_enc = [encode_latency(library[lab]) for lab in labels]
    rank = RankOrderMatcher(library)

    def jitter(v, n):
        return [min(1.0, max(0.0, x + rng.uniform(-n, n))) for x in v]

    trials = 300
    # each window carries its own random monotonic rig gain (contrast/offset varies per capture)
    windows = []
    for i in range(trials):
        lab = labels[i % len(labels)]
        w = jitter(library[lab], 0.10)
        b, p = rng.uniform(0.0, 0.45), rng.uniform(0.5, 2.2)     # per-capture rig setting
        windows.append((lab, w, b, p))

    def acc(matcher, apply_gain):
        ok = 0
        for lab, w, b, p in windows:
            x = [_rig_gain(v, b, p) for v in w] if apply_gain else w
            ok += matcher(x) == lab
        return ok / trials

    rank_clean = acc(rank.label, False)
    rank_gain = acc(rank.label, True)
    tmpl_clean = acc(lambda w: _template_label(refs_enc, labels, w), False)
    tmpl_gain = acc(lambda w: _template_label(refs_enc, labels, w), True)

    print("=" * 66)
    print("RANK-ORDER CODING — gain-invariant recognition on the command library")
    print("=" * 66)
    print(f"{len(labels)} commands, {trials} jittered windows")
    print("                     clean    under a per-capture rig gain change")
    print(f"  rank-order match : {rank_clean:.0%}     {rank_gain:.0%}   (order unchanged -> invariant)")
    print(f"  template (VanR)  : {tmpl_clean:.0%}     {tmpl_gain:.0%}   (magnitude drifts -> degrades)")
    print("=" * 66)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert rank_clean >= 0.9, f"rank-order matcher must recognize the commands ({rank_clean:.2f})"
    assert rank_gain == rank_clean, "rank-order recognition must be EXACTLY gain-invariant"
    assert tmpl_gain < tmpl_clean - 0.05, f"the magnitude template must degrade under rig gain ({tmpl_clean:.2f}->{tmpl_gain:.2f})"
    assert rank_gain > tmpl_gain, f"rank-order must beat the template under a rig gain change ({rank_gain:.2f} vs {tmpl_gain:.2f})"
    assert rank_code([3.0, 1.0, 2.0]) == [0, 2, 1], "descending rank: largest -> 0"
    print(f"self-check OK: rank-order {rank_clean:.0%} clean and {rank_gain:.0%} under rig gain "
          f"(invariant); template drops {tmpl_clean:.0%}->{tmpl_gain:.0%}; wired to the command library")


if __name__ == "__main__":
    main()
