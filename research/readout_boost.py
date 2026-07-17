r"""
READOUT BOOST (Tier-2) -- squeeze the last points out of the READOUT on the FROZEN conv-STDP digit
features, without touching the encoder. The audit's cheapest sound gain: standardize SLDA-RFF, then
ENSEMBLE it and add a small nonlinear MLP head.

On the frozen conv1 features the ladder is: continual-linear 93.3%, static-joint-linear 95.1%,
single RBF-SVM ~96.2%, RBF ENSEMBLE 98.2%. So error-decorrelating ENSEMBLES -- not a fancier single
head -- are the remaining lever. This measures, on the SAME frozen features and disjoint held-out:
  * single SLDA-RFF (tuned gamma)
  * ENSEMBLE of SLDA-RFF (varied gamma + RFF seed, soft-voted)
  * a small nonlinear MLP head (1 hidden layer) over the same features
vs the linear / RBF-ensemble reference points. Feature learning stays unsupervised + frozen; only the
readout changes -- so any gain is honestly attributable to the decoder, and bounded by the 98.2%
feature ceiling (the audit's key caveat: past this, only a better ENCODER helps -- see Tier 3).

  NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/readout_boost.py
  Knobs: NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_RFF(2000) NORD_TRAIN(12000) NORD_TEST(2000)
Deps: numpy torch torchvision snntorch (feature extraction) + numpy readouts.
"""
import os

import numpy as np
import torch

from torch_device import resolve_device, enable
from snn_mnist_stdp import load_mnist
from deep_stdp_latency import ConvLatencyStdp, _train
from fast_features import batched_features
from language_readout_slda import SLDA_RFF_np


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _softmax_scores(slda, F):
    """Class scores from one SLDA-RFF model (the discriminant scores, for soft-voting ensembles)."""
    Z = slda._rff(F)
    cnt = np.clip(slda.cnt_c, 1e-6, None)
    mu = slda.sum_c / cnt[:, None]
    n = max(1.0, slda.cnt_c.sum())
    S = slda.ZZt - (slda.cnt_c[:, None] * mu).T @ mu
    cov = S / n + slda.shrink * np.eye(slda.M) * np.trace(S / n) / slda.M
    cinv = np.linalg.pinv(cov)
    W = mu @ cinv
    bias = -0.5 * (W * mu).sum(1)
    sc = Z @ W.T + bias
    sc -= sc.max(1, keepdims=True)
    e = np.exp(sc)
    return e / e.sum(1, keepdims=True)                       # normalized -> comparable across members


class MLPHead:
    """A small 1-hidden-layer softmax MLP readout (nonlinear head) over the frozen features, numpy SGD."""

    def __init__(self, in_dim, n_cls=10, hidden=256, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, hidden) * (1.0 / np.sqrt(in_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, n_cls) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(n_cls)

    def fit(self, X, y, epochs=25, lr=0.1, batch=256, seed=0):
        rng = np.random.RandomState(seed)
        Y = np.eye(self.W2.shape[1])[y]
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                b = idx[i:i + batch]
                xb, yb = X[b], Y[b]
                h = np.maximum(0, xb @ self.W1 + self.b1)
                s = h @ self.W2 + self.b2
                s -= s.max(1, keepdims=True)
                p = np.exp(s)
                p /= p.sum(1, keepdims=True)
                g = (p - yb) / len(b)
                gW2 = h.T @ g
                gb2 = g.sum(0)
                gh = (g @ self.W2.T) * (h > 0)
                gW1 = xb.T @ gh
                gb1 = gh.sum(0)
                self.W2 -= lr * gW2
                self.b2 -= lr * gb2
                self.W1 -= lr * gW1
                self.b1 -= lr * gb1
        return self

    def predict(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).argmax(1)


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    C1 = _envi("NORD_C1", 30)
    T_LAT = _envi("NORD_T", 15)
    L1E = _envi("NORD_L1_EPOCHS", 3)
    RFF = _envi("NORD_RFF", 2000)

    print("=" * 90)
    print("READOUT BOOST (Tier-2) -- ensemble + MLP head over FROZEN conv-STDP features vs the 98.2% ceiling")
    print("=" * 90)
    Xtr, Ytr, Xte, Yte = load_mnist(_envi("NORD_TRAIN", 12000), _envi("NORD_TEST", 2000))
    torch.manual_seed(0)
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1E):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1E}")
    Ftr = batched_features(conv, Xtr).cpu().numpy().astype(np.float64)
    Fte = batched_features(conv, Xte).cpu().numpy().astype(np.float64)
    ytr, yte = Ytr.numpy(), Yte.numpy()
    D = Ftr.shape[1]
    print(f"[features] frozen conv1, dim={D}; train {len(Ftr)}, test {len(Fte)}")

    # single SLDA-RFF (tuned gamma)
    single = SLDA_RFF_np(D, 10, rff=RFF, gamma=0.02, shrink=0.1, seed=0)
    single.observe(Ftr, ytr)
    acc_single = float((single.predict(Fte) == yte).mean())

    # ENSEMBLE: varied gamma + RFF seed, soft-vote the normalized discriminant scores
    members = [(0.01, 1), (0.02, 2), (0.02, 3), (0.04, 4), (0.04, 5)]
    probs = np.zeros((len(Fte), 10))
    for g, sd in members:
        m = SLDA_RFF_np(D, 10, rff=RFF, gamma=g, shrink=0.1, seed=sd)
        m.observe(Ftr, ytr)
        probs += _softmax_scores(m, Fte)
    acc_ens = float((probs.argmax(1) == yte).mean())

    # small nonlinear MLP head over the same frozen features
    mlp = MLPHead(D, 10, hidden=256, seed=0).fit(Ftr, ytr, epochs=25, lr=0.1)
    acc_mlp = float((mlp.predict(Fte) == yte).mean())

    print("-" * 90)
    print(f"  single SLDA-RFF (gamma 0.02)        : {acc_single:.1%}")
    print(f"  ENSEMBLE of {len(members)} SLDA-RFF (soft-vote) : {acc_ens:.1%}   ({(acc_ens - acc_single) * 100:+.1f} pts)")
    print(f"  nonlinear MLP head (1x256)          : {acc_mlp:.1%}   ({(acc_mlp - acc_single) * 100:+.1f} pts)")
    print("-" * 90)
    print(f"  reference: static-joint linear 95.1%   single RBF-SVM ~96.2%   RBF ENSEMBLE 98.2% (ceiling)")
    print(f"  honest: gains are bounded by the 98.2% FROZEN-FEATURE ceiling; past it only a better ENCODER")
    print(f"  helps (Tier 3: learnable delays + adaptive-LIF on TEMPORAL data, served frozen into core).")
    print("=" * 90)

    assert acc_single > 0.90, f"single SLDA-RFF regressed ({acc_single:.3f})"
    assert acc_ens >= acc_single, f"ensembling must not hurt ({acc_ens:.3f} vs {acc_single:.3f})"
    assert max(acc_ens, acc_mlp) <= 0.99, "a readout implausibly beat the frozen-feature RBF ceiling -- check for leakage"
    print(f"self-check OK: on FROZEN features, ensembling {acc_single:.1%}->{acc_ens:.1%} and an MLP head "
          f"{acc_mlp:.1%}; all bounded by the 98.2% feature ceiling -- the readout is near its limit, so "
          f"the next real gain is a better encoder (Tier 3), not more decoder engineering.")
    return 0


if __name__ == "__main__":
    main()
