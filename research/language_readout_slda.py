r"""
LANGUAGE READOUT (STRONG) -- close the n-gram gap the pure-spiking readout could not.

research/language_self_build.py honestly showed the single-layer spike-timing R-STDP readout forms
separated assemblies but scores BELOW n-gram counts (phonetics ~7% vs bigram ~20%) -- the
readout-is-the-lever ceiling seen all across this project. This file applies the SAME fix that won
everywhere else in the recovered loop: an SLDA-on-Random-Fourier-Features decoder (order-invariant,
RBF-kernel-approx) over the SPIKING context code. Same spiking representation (the positional
rank-order context from language_self_build), stronger decoder.

This is the HYBRID, stated honestly: the assemblies / context code stay spiking (representation);
competence rides a strong linear decoder (SLDA-RFF). It is NOT a claim that spikes alone reach this
-- it is the deliberate division of labour this project keeps finding: spiking for representation +
purity, a strong decoder for accuracy. The SLDA-RFF readout is order-invariant continual == joint
(no forgetting), so it doubles as the GENERAL competence head for any autonomously-built skill.

The torch SLDA_RFF in developmental_loop_slda is hardcoded to 10 classes; language needs 39 phones /
K words, so this ports the identical math to numpy with a general class count (research tier -- numpy
is allowed; CORE stays zero-dep and untouched).

  PYTHONPATH=src python research/language_readout_slda.py
  Knobs: NORD_CTX(3) NORD_PER(6000) NORD_VOCAB(60) NORD_RFF(1500) NORD_GAMMA(0.15) NORD_SHRINK(0.1)
Deps: numpy (+ stdlib). Reuses corpus_stream + language_self_build's positional code + baselines.
"""
import os

import numpy as np

from corpus_stream import ARPABET, load_cmudict, load_text, phoneme_stream, word_stream
from language_self_build import _positional_code, _frequent, _bigram_baseline, _unigram_baseline


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


class SLDA_RFF_np:
    """Streaming LDA on Random Fourier Features, general n_classes (numpy port of the torch SLDA_RFF
    in developmental_loop_slda). z = sqrt(2/M) cos(F W + b) approximates an RBF kernel; classify by
    the shared-covariance linear discriminant over ORDER-INVARIANT sufficient statistics."""

    def __init__(self, in_dim, n_classes, rff=1500, gamma=0.15, shrink=0.1, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(in_dim, rff).astype(np.float64) * gamma
        self.b = rng.rand(rff).astype(np.float64) * (2.0 * np.pi)
        self.scale = (2.0 / rff) ** 0.5
        self.M, self.C, self.shrink = rff, n_classes, float(shrink)
        self.sum_c = np.zeros((n_classes, rff))
        self.cnt_c = np.zeros(n_classes)
        self.ZZt = np.zeros((rff, rff))

    def _rff(self, F):
        return self.scale * np.cos(F @ self.W + self.b)

    def observe(self, F, y):
        Z = self._rff(F)
        self.ZZt += Z.T @ Z
        for c in range(self.C):
            m = y == c
            if m.any():
                self.sum_c[c] += Z[m].sum(0)
                self.cnt_c[c] += int(m.sum())

    def predict(self, F):
        Z = self._rff(F)
        cnt = np.clip(self.cnt_c, 1e-6, None)
        mu = self.sum_c / cnt[:, None]
        n = max(1.0, self.cnt_c.sum())
        S = self.ZZt - (self.cnt_c[:, None] * mu).T @ mu
        cov = S / n
        cov += self.shrink * np.eye(self.M) * np.trace(cov) / self.M
        cinv = np.linalg.pinv(cov)
        W = mu @ cinv
        bias = -0.5 * (W * mu).sum(1)
        seen = self.cnt_c > 0
        scores = Z @ W.T + bias
        scores[:, ~seen] = -1e30
        return scores.argmax(1)


def _slda_acc(pairs, responses, ctx, per, rff, gamma, shrink, seed=0):
    """Train SLDA-RFF on (positional context code, next-token index); held-out accuracy on the tail."""
    ridx = {r: i for i, r in enumerate(responses)}
    V = len(responses)
    usable = [(c, nx) for c, nx in pairs if nx in ridx]
    if len(usable) < 60:
        return 0.0, 0
    cut = int(len(usable) * 0.8)
    train, test = usable[:cut], usable[cut:]
    train = train[:per] if per else train

    def X(rows):
        return np.array([_positional_code(c, ridx, V, ctx) for c, _ in rows], dtype=np.float64)

    ytr = np.array([ridx[nx] for _, nx in train])
    yte = np.array([ridx[nx] for _, nx in test])
    s = SLDA_RFF_np(V * ctx, V, rff=rff, gamma=gamma, shrink=shrink, seed=seed)
    s.observe(X(train), ytr)
    pred = s.predict(X(test))
    return float((pred == yte).mean()), len(test)


def main():
    CTX = _envi("NORD_CTX", 3)
    PER = _envi("NORD_PER", 6000)
    VOCAB = _envi("NORD_VOCAB", 60)
    RFF = _envi("NORD_RFF", 1500)
    GAMMA = _envf("NORD_GAMMA", 0.15)
    SHRINK = _envf("NORD_SHRINK", 0.1)

    print("=" * 92)
    print("LANGUAGE READOUT (STRONG) -- SLDA-RFF decoder over the spiking context code vs n-gram")
    print(f"ctx={CTX}  per={PER}  vocab={VOCAB}  rff={RFF}  gamma={GAMMA}  (hybrid: spiking repr + strong decoder)")
    print("=" * 92)
    cmudict, dict_off = load_cmudict()
    words, text_off = load_text()
    print(f"[data] {len(words):,} tokens ({'offline' if text_off else 'real corpus'}); "
          f"cmudict {len(cmudict):,} ({'offline' if dict_off else 'real'})")

    ph_pairs = list(phoneme_stream(words, cmudict, k=CTX))
    wd_pairs = list(word_stream(words, k=CTX))
    freq_words = _frequent([nx for _, nx in wd_pairs], VOCAB)

    ph_slda, ph_n = _slda_acc(ph_pairs, ARPABET, CTX, PER, RFF, GAMMA, SHRINK, seed=0)
    ph_bigram = _bigram_baseline(ph_pairs)
    wd_slda, wd_n = _slda_acc(wd_pairs, freq_words, CTX, PER, RFF, GAMMA, SHRINK, seed=1)
    wd_uni = _unigram_baseline(wd_pairs, freq_words)

    print("\n" + "-" * 92)
    print("RESULT -- strong SLDA-RFF decoder over the SAME spiking positional context code")
    print("-" * 92)
    print(f"  PHONETICS next-phoneme: SLDA-RFF {ph_slda:.1%}  vs  bigram {ph_bigram:.1%}  "
          f"vs  R-STDP ~7.1% (from language_self_build)   [held-out n={ph_n}]")
    print(f"  SEMANTICS next-word   : SLDA-RFF {wd_slda:.1%}  vs  unigram {wd_uni:.1%}  "
          f"vs  R-STDP ~2.1%   [held-out n={wd_n}]")
    print("-" * 92)
    print(f"  phonetics gap vs bigram: {(ph_slda - ph_bigram) * 100:+.1f} pts")
    print("  the assemblies/representation stay SPIKING; competence rides the strong decoder (the hybrid).")
    print("  SLDA-RFF is order-invariant (continual==joint, no forgetting) -> the GENERAL skill-competence head.")
    print("=" * 92)

    assert ph_slda > ph_bigram, (
        f"strong readout must CLOSE the n-gram gap: SLDA-RFF {ph_slda:.3f} vs bigram {ph_bigram:.3f}")
    assert wd_slda >= wd_uni, (
        f"strong readout must at least match the word unigram: {wd_slda:.3f} vs {wd_uni:.3f}")
    print(f"self-check OK: SLDA-RFF decoder over the spiking context CLOSES the n-gram gap "
          f"(phonetics {ph_slda:.1%} > bigram {ph_bigram:.1%}); the hybrid (spiking repr + strong "
          f"order-invariant decoder) is the general competence head for autonomously-built skills.")
    return 0


if __name__ == "__main__":
    main()
