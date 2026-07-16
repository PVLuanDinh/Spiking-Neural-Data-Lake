r"""
Shared helpers for the developmental-loop suite -- the DRY extraction of the boilerplate that was
copy-pasted across developmental_loop_{deep,lake,lake99,wm,slda}.py (and the ensemble scripts).

Exports (all behavior-identical to the local versions they replace):
  envi(key, default) / envf(key, default)  -- int/float env readers
  LinearReadout(in_dim, n_classes=10, seed=0, lr=0.01)  -- the continual linear head (Adam + CE)
  cache_features(conv, X, dev=None, batch=256)  -- BATCHED feature extraction (fast_features,
      byte-identical to the per-image conv.features loop, ~135x faster on GPU)
  acc(readout, feats, labels, idx)  -- held-out accuracy over cached-feature rows

Zero-dep CORE untouched; research tier only.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_device import resolve_device
from fast_features import batched_features


def envi(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


def envf(key, default):
    v = os.environ.get(key)
    return float(v) if v else default


class LinearReadout:
    """Continual linear head: nn.Linear + persistent Adam + cross-entropy. Identical to the class that
    was duplicated in the lake / lake99 / wm / slda / readout_ensemble loops."""

    def __init__(self, in_dim, n_classes=10, seed=0, lr=0.01):
        torch.manual_seed(seed)
        self.dev = resolve_device()
        self.clf = nn.Linear(in_dim, n_classes).to(self.dev)
        self.opt = torch.optim.Adam(self.clf.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(self, X, y, lr, steps=1):
        for g in self.opt.param_groups:
            g["lr"] = lr
        for _ in range(steps):
            self.opt.zero_grad()
            F.cross_entropy(self.clf(X), y).backward()
            self.opt.step()

    def predict(self, X):
        with torch.no_grad():
            return self.clf(X)


def cache_features(conv, X, dev=None, batch=256):
    """Batched, byte-identical (<1e-4 GPU) replacement for `torch.stack([conv.features(x) for x in X])`.
    `dev` is accepted for call-site compatibility but ignored -- batched_features returns on conv.device."""
    return batched_features(conv, X, batch=batch)


def acc(readout, feats, labels, idx):
    if not idx:
        return 0.0
    return float((readout.predict(feats[idx]).argmax(1) == labels[idx]).float().mean())
