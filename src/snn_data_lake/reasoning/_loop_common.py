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

from snn_data_lake.substrate.device import resolve_device
from snn_data_lake.reasoning.fast_features import batched_features

#: Which symbols here OWE a gradient cosine, read statically by
#: `substrate/gradient_claim_gate.py`; never imported by it.
#:
#: ADDED 2026-08-18 by the specs/020 audit. `docs/POLICIES.md` already NAMED this symbol as one of
#: the two genuine autograd copies "declared APPROXIMATES_GRADIENT" -- and it carried no declaration
#: at all, so the governing document asserted a state the tree did not have. It slipped because
#: `reasoning/` is UNGOVERNED: the census reports MISSING_DECLARATION there and never asserts on it,
#: which is exactly the tier where a claim can outlive what it describes.
GRADIENT_CLAIM = {
    "LinearReadout.train_step": (
        "APPROXIMATES_GRADIENT",
        "explicit autograd: `F.cross_entropy(self.clf(X), y).backward()` then `self.opt.step()` on "
        "an Adam optimizer. The M21 duplicate of "
        "`hippocampus/developmental_loop_brain::_LinearReadout.train_step` and carries the SAME "
        "disposition by the same criterion -- LIVE-PATH REACHABILITY, per the e-prop invariant's "
        "scope clause. PROVISIONAL and deliberately OVER-flagged: it is train-tier offline code no "
        "live path reaches today, which is what permits the Adam path, but nothing in the tree "
        "announces the day that changes, and Standing Invariant 3 makes the confident negative the "
        "dangerous verdict",
    ),
}


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
        # A constructor may not re-seed the process-global RNG -- see the identical note on
        # hippocampus/developmental_loop_brain._LinearReadout, which is a verbatim copy of this
        # class (AUDIT_BACKLOG M21). The duplication is NOT collapsed here: adopting either into
        # the other needs the promotion lane M21 is owed by, and this fix must not create a new
        # governed->reasoning import. Fixed twice, on purpose, and recorded rather than laundered.
        # `nn.Linear` takes no `generator=`, so the RNG is forked instead: identical weights, no
        # global side effect. `devices=[]` forks CPU only; the draw below is a CPU draw.
        with torch.random.fork_rng(devices=[]):
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
