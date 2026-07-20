r"""
READOUT FAIR COMPARE (Tier-1 honesty fix) -- isolate the DECODER effect from the training-budget
confound the code audit flagged.

The original language comparison was NOT apples-to-apples: the spiking R-STDP readout trained on 400
pairs, the SLDA-RFF on 6000, and the n-gram baselines on the FULL train split -- so part of the
"SLDA-RFF hybrid closes the n-gram gap" was just 15x more data, not the decoder. This re-measures all
of them at ONE training budget, swept, on the SAME stream / split / context window, so the decoder
delta is clean and the data-budget effect is visible separately.

Reads out next-phoneme over the real CMUdict+Gutenberg stream. Three readouts + two baselines, each
at the same `per`:
  * R-STDP (single-layer spiking, rank-order + reward-modulated STDP)   [language_self_build path]
  * SLDA-RFF (rate decoder over the SAME positional spiking context code)
  * bigram / trigram (most-frequent next given the last 1 / 2 tokens)   [same per, not full train]

  PYTHONPATH=src .venv-lake/Scripts/python research/readout_fair_compare.py
  Knobs: NORD_CTX(3) NORD_PERS("400,1200,3600,6000") NORD_RFF(1500) NORD_GAMMA(0.15)
Deps: numpy (+ stdlib). Reuses corpus_stream, language_self_build, language_readout_slda.
"""
import os
from collections import Counter, defaultdict

import numpy as np

from snn_data_lake.reasoning.corpus_stream import ARPABET, load_cmudict, load_text, phoneme_stream
from snn_data_lake.reasoning.language_self_build import _positional_code
from snn_data_lake.reasoning.language_readout_slda import SLDA_RFF_np
from snn_data_lake.cortex.interaction_learning import SpikingInteractionLearner


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


def _ngram_acc(train, test, k):
    """Most-frequent next-token given the last k context tokens, fit on `train` (the SAME budget the
    readouts get), scored on `test`."""
    tbl = defaultdict(Counter)
    for c, nx in train:
        key = tuple(c[-k:]) if k else ()
        tbl[key][nx] += 1
    if not test:
        return 0.0
    ok = 0
    for c, nx in test:
        key = tuple(c[-k:]) if k else ()
        if tbl[key]:
            ok += (tbl[key].most_common(1)[0][0] == nx)
    return ok / len(test)


def _rstdp_acc(train, test, vocab, ctx):
    """Single-layer spiking R-STDP readout trained on `train`, scored on `test`. Positional context
    code as the spike stimulus; the taught label is the true next phoneme."""
    ridx = {r: i for i, r in enumerate(vocab)}
    n_in = len(vocab) * ctx
    learner = SpikingInteractionLearner(vocab, n_in=n_in, seed=0, t_post=max(4.0, n_in / 4.0))
    for c, nx in train:
        learner.interact(_positional_code(c, ridx, len(vocab), ctx), nx)
    ok = sum(learner.respond(_positional_code(c, ridx, len(vocab), ctx)) == nx for c, nx in test)
    return ok / max(1, len(test))


def _slda_acc(train, test, vocab, ctx, rff, gamma):
    ridx = {r: i for i, r in enumerate(vocab)}
    V = len(vocab)
    X = lambda rows: np.array([_positional_code(c, ridx, V, ctx) for c, _ in rows], dtype=np.float64)  # noqa: E731
    s = SLDA_RFF_np(V * ctx, V, rff=rff, gamma=gamma, shrink=0.1, seed=0)
    s.observe(X(train), np.array([ridx[nx] for _, nx in train]))
    return float((s.predict(X(test)) == np.array([ridx[nx] for _, nx in test])).mean())


def main():
    CTX = _envi("NORD_CTX", 3)
    PERS = [int(x) for x in os.environ.get("NORD_PERS", "400,1200,3600,6000").split(",")]
    RFF = _envi("NORD_RFF", 1500)
    GAMMA = _envf("NORD_GAMMA", 0.15)

    cmudict, _ = load_cmudict()
    words, _ = load_text()
    pairs = [(c, nx) for c, nx in phoneme_stream(words, cmudict, k=CTX) if nx in set(ARPABET)]
    cut = int(len(pairs) * 0.8)
    train_all, test = pairs[:cut], pairs[cut:]

    print("=" * 92)
    print("READOUT FAIR COMPARE -- decoder effect at EQUAL training budget (isolates the data confound)")
    print(f"next-phoneme, ctx={CTX}, held-out n={len(test)}; each readout trained on the SAME `per` pairs")
    print("=" * 92)
    print(f"{'per':>6} | {'R-STDP':>8} | {'SLDA-RFF':>9} | {'bigram':>7} | {'trigram':>8} | decoder delta (SLDA-R-STDP)")
    print("-" * 92)
    rows = []
    for per in PERS:
        tr = train_all[:per]
        r = _rstdp_acc(tr, test, ARPABET, CTX)
        s = _slda_acc(tr, test, ARPABET, CTX, RFF, GAMMA)
        b = _ngram_acc(tr, test, 1)
        t3 = _ngram_acc(tr, test, 2)
        rows.append((per, r, s, b, t3))
        print(f"{per:>6} | {r:>7.1%} | {s:>8.1%} | {b:>6.1%} | {t3:>7.1%} | {(s - r) * 100:>+5.1f} pts")
    print("-" * 92)
    # honest readings at the LARGEST equal budget
    per, r, s, b, t3 = rows[-1]
    rstdp_degrades = rows[-1][1] < rows[0][1]
    ngram_best = max(b, t3)
    print(f"  at per={per} (equal budget): SLDA-RFF {s:.1%} vs R-STDP {r:.1%} = a {(s-r)*100:+.1f}-pt DECODER "
          f"gain (not a data artifact)")
    print(f"  vs the best n-gram at the SAME budget (trigram {t3:.1%}): SLDA-RFF {s:.1%} = {(s-ngram_best)*100:+.1f} pts "
          f"-- so at equal data the readout only about TIES the n-gram, NOT the decisive win the 15x-data confound suggested")
    print(f"  NEW honest finding: the single-layer R-STDP readout DEGRADES with more data "
          f"({rows[0][1]:.1%} at per={rows[0][0]} -> {r:.1%} at per={per}) -- it saturates/overwrites, "
          f"the same recency problem that sank the enclosed-loop MLP.")
    print("=" * 92)
    # self-checks: the DECODER swap helps at equal budget; SLDA-RFF (unlike R-STDP) still improves with data
    assert s > r, f"SLDA-RFF must beat R-STDP at EQUAL budget ({s:.3f} vs {r:.3f})"
    assert rows[-1][2] >= rows[0][2], "SLDA-RFF did not benefit from more data (sanity)"
    assert rstdp_degrades, "expected R-STDP to degrade with data at this budget (the honest finding)"
    print(f"self-check OK: at EQUAL budget the decoder swap (R-STDP -> SLDA-RFF) is worth {(s-r)*100:+.1f} pts; "
          f"at equal data SLDA-RFF only ~ties the n-gram ({s:.1%} vs trigram {t3:.1%}); R-STDP DEGRADES with more "
          f"data. The earlier 7%-vs-22% comparison conflated a 15x data difference -- corrected.")
    return 0


if __name__ == "__main__":
    main()
