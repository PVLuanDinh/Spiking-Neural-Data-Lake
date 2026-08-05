"""
Bayesian spiking inference — calibrated uncertainty on top of a recall (scan #7).

Nothing in store/search/reason emits a CONFIDENCE: the KG query and the novelty gate return an answer
with no "how sure". A Bernoulli weight mask sampled per forward pass turns the deterministic readout
into an ensemble; the disagreement across N spike-count passes is a cheap, calibrated uncertainty —
low for a clean in-distribution cue, high for an out-of-distribution / ambiguous one. Composes
with the drift monitor and the KG. Refs: arXiv:2401.15453 (Bayesian Inference Accelerator for
SNNs, DATE 2024).

  Run: python bayesian_snn.py       # (also run by run_selfchecks.py)
"""

import random

# random.seed(0) DELETED 2026-08-04 (specs/010, C33): it was an IMPORT-TIME global reseed that
# clobbered any caller's seed, and it was VESTIGIAL here -- every RNG use in this module already
# goes through a per-instance `random.Random(seed)`, so nothing read the global stream it set.
# Verified by AST: no module-level statement and no function in this file touches the global
# random API. Deterministic behaviour is unchanged; the side effect on importers is gone.

FEAT, CLASSES, PASSES, KEEP = 24, 3, 25, 0.6


def prototypes(rng):
    return [[rng.choice((-1, 1)) for _ in range(FEAT)] for _ in range(CLASSES)]


def forward(x, protos, mask):
    """Masked spike-count readout: argmax over class prototypes with a Bernoulli weight mask."""
    scores = [sum(protos[c][i] * x[i] for i in range(FEAT) if mask[i]) for c in range(CLASSES)]
    return max(range(CLASSES), key=lambda c: scores[c])


def bayesian_predict(x, protos, rng):
    """N masked passes -> (modal prediction, uncertainty = disagreement rate across passes)."""
    votes = [forward(x, protos, [rng.random() < KEEP for _ in range(FEAT)]) for _ in range(PASSES)]
    modal = max(set(votes), key=votes.count)
    return modal, 1.0 - votes.count(modal) / PASSES


def main():
    rng = random.Random(0)
    protos = prototypes(rng)

    # in-distribution: a class prototype with a few bits flipped (should be confident + correct)
    acc, unc_in = 0, []
    for _ in range(60):
        c = rng.randrange(CLASSES)
        x = list(protos[c])
        for i in rng.sample(range(FEAT), 4):
            x[i] = -x[i]
        pred, u = bayesian_predict(x, protos, rng)
        acc += pred == c
        unc_in.append(u)

    # out-of-distribution: random cues far from every prototype (should be UNcertain)
    unc_ood = [
        bayesian_predict([rng.choice((-1, 1)) for _ in range(FEAT)], protos, rng)[1]
        for _ in range(60)
    ]

    m_in = sum(unc_in) / len(unc_in)
    m_ood = sum(unc_ood) / len(unc_ood)
    accuracy = acc / 60

    print("=" * 58)
    print("BAYESIAN SPIKING INFERENCE (uncertainty via masked passes)")
    print("=" * 58)
    print(f"feat={FEAT} classes={CLASSES} passes={PASSES} keep-prob={KEEP}")
    print(f"in-distribution : accuracy {accuracy:.0%}  mean uncertainty {m_in:.2f}")
    print(
        f"out-of-dist     : mean uncertainty {m_ood:.2f}  ({m_ood / max(m_in, 1e-6):.1f}x higher)"
    )
    print(
        "attaches 'answer + how sure' to KG queries / the novelty gate; composes with drift_monitor."
    )
    print("=" * 58)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert accuracy >= 0.95, f"in-distribution recall should stay accurate: {accuracy:.2f}"
    assert m_in < 0.2, f"clean cues should be confident (low uncertainty): {m_in:.2f}"
    assert m_ood > m_in + 0.2, (
        f"OOD cues should be measurably less certain: ood={m_ood:.2f} in={m_in:.2f}"
    )
    print(
        f"self-check OK: {accuracy:.0%} accurate + confident in-dist ({m_in:.2f}); "
        f"OOD uncertainty {m_ood:.2f} flags 'don't trust this answer'"
    )


if __name__ == "__main__":
    main()
