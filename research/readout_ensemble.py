r"""
FEATURE SCALING + READOUT ENSEMBLE experiment -- can we raise BOTH the linear and the
progressive-columns readout, and does COMBINING them give a fully accurate loop?

The v3.53/60 finding: both readouts hit the SAME wall because they share the frozen unsupervised
conv1 features (C1=30 -> ~95% static ceiling; 93.3% linear / 90.5% columns continual). Two levers:
  1. SCALE the shared features: more conv1 maps (C1, default 100) -> richer basis -> higher ceiling
     for BOTH readouts (research: more STDP maps => more extractable features).
  2. ENSEMBLE the two readouts (average softmax). Ensemble gain REQUIRES decorrelated errors; these
     two readouts read the SAME features -> correlated errors -> expect only a small gain, NOT full
     accuracy. This run measures exactly how small.

Both readouts are trained CONTINUALLY on an IDENTICAL developmental schedule (same batches, same
replay) so their comparison + ensemble is fair; features are frozen + unsupervised (labels only in
the supervised readouts). No leakage: readouts train only on train-cache rows, measured only on the
held-out test cache; replay pools are held-out train rows.

Run (from repo root; GPU; C1=100 conv is slower to pre-train):
    NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/readout_ensemble.py
Knobs: NORD_C1(100) NORD_CYCLES(15) NORD_PER(300) NORD_REPLAY(200) NORD_SGD(300) NORD_HIDDEN(32)
       NORD_T(15) NORD_L1_EPOCHS(3) NORD_TRAIN(8000) NORD_TEST(2000)
Deps: numpy torch torchvision snntorch   (research/GPU tier; core stays zero-dep)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_device import resolve_device, enable
from snn_mnist_stdp import load_mnist
from deep_stdp_latency import ConvLatencyStdp, _train
from brain_adapters.laminar_readout import LaminarColumns


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


class LinearReadout:
    """Plain continual nn.Linear (the v3.54 winner) -- matches the columns readout interface."""

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

    def grow(self, k=1, freeze_old=False):
        return

    @property
    def n_hidden(self):
        return 0


def _cache(conv, X, dev, tag):
    rows = []
    for i, img in enumerate(X):
        rows.append(conv.features(img).reshape(-1))
        if (i + 1) % 2000 == 0:
            print(f"  cache {tag}: {i + 1}/{len(X)}")
    return torch.stack(rows, 0).to(dev)


def _acc(logits, y, idx):
    if not idx:
        return 0.0
    pred = logits[idx].argmax(1)
    return (pred == y[idx]).float().mean().item()


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    C1 = _envi("NORD_C1", 100)
    CYCLES = _envi("NORD_CYCLES", 15)
    PER = _envi("NORD_PER", 300)
    REPLAY = _envi("NORD_REPLAY", 200)
    SGD = _envi("NORD_SGD", 300)
    HIDDEN = _envi("NORD_HIDDEN", 32)
    T_LAT = _envi("NORD_T", 15)
    L1_EPOCHS = _envi("NORD_L1_EPOCHS", 3)
    TRAIN_N = _envi("NORD_TRAIN", 8000)
    TEST_N = _envi("NORD_TEST", 2000)

    print("=" * 78)
    print(f"FEATURE SCALING (C1={C1}) + READOUT ENSEMBLE  (linear vs columns vs combined)")
    print("=" * 78)
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    print(f"[perception] frozen; feat_dim={conv.feat_dim()}; diversity={conv.kernel_diversity():.3f}")

    print("\n[cache] conv1 features (train+test)...")
    Ftr, Fte = _cache(conv, Xtr, dev, "train"), _cache(conv, Xte, dev, "test")
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    # two readouts on the SAME features, trained on the SAME schedule/batches
    lin = LinearReadout(D, seed=0)
    col = LaminarColumns(D, n_classes=10, n23=HIDDEN, seed=0)
    memory = {}                                        # held-out replay pool per learned digit
    learned = []
    comp = {d: 0.0 for d in range(10)}

    for cyc in range(1, CYCLES + 1):
        # deterministic developmental schedule: phase 1 learn 0..9 in order; phase 2 revisit the
        # weakest digit by the LINEAR readout's competence (shared schedule -> fair comparison).
        if cyc <= 10:
            obj = cyc - 1
        else:
            obj = min(learned, key=lambda d: comp[d])
        is_new = obj not in learned
        n_train = PER if is_new else max(REPLAY, PER // 4)
        obj_rows = tr_by[obj][:n_train]
        replay_rows = [r for d in learned if d != obj for r in memory.get(d, [])]
        idx = torch.tensor(obj_rows + replay_rows, dtype=torch.long, device=dev)
        Xb, yb = Ftr[idx], ytr[idx]

        # columns grows+freezes a fresh column per NEW digit (progressive neurogenesis)
        if is_new and len(learned) >= 1:
            col.grow(1, freeze_old=True)
        lin.train_step(Xb, yb, lr=0.01, steps=SGD)
        col.train_step(Xb, yb, lr=0.01, steps=SGD)
        if is_new:
            memory[obj] = tr_by[obj][n_train:n_train + REPLAY]
            learned.append(obj)

        # competence (linear) drives the phase-2 schedule
        with torch.no_grad():
            lin_log = lin.predict(Fte)
        for d in learned:
            comp[d] = _acc(lin_log, yte, te_by[d])
        pooled = [r for d in learned for r in te_by[d]]
        print(f"CYCLE {cyc:2d} obj={obj} {'new' if is_new else 'rev'}  "
              f"linear over {sorted(learned)} = {_acc(lin_log, yte, pooled):.1%}")

    # ---- final: linear, columns, and their ENSEMBLE on the held-out test set ----
    pooled = [r for d in learned for r in te_by[d]]
    with torch.no_grad():
        lin_log = lin.predict(Fte)
        col_log = col.predict(Fte)
    acc_lin = _acc(lin_log, yte, pooled)
    acc_col = _acc(col_log, yte, pooled)
    # ensemble = average of the two softmax distributions (probability-space vote)
    ens_prob = 0.5 * (F.softmax(lin_log, dim=1) + F.softmax(col_log, dim=1))
    acc_ens = (ens_prob[pooled].argmax(1) == yte[pooled]).float().mean().item()
    # error correlation: fraction of test points where BOTH readouts are wrong (correlated failure)
    lin_wrong = lin_log[pooled].argmax(1) != yte[pooled]
    col_wrong = col_log[pooled].argmax(1) != yte[pooled]
    both_wrong = (lin_wrong & col_wrong).float().mean().item()
    either_wrong = (lin_wrong | col_wrong).float().mean().item()

    print("\n" + "=" * 78)
    print(f"RESULT  (C1={C1} features; readouts on the SAME frozen features; chance=10%)")
    print("-" * 78)
    print(f"  linear readout             : {acc_lin:.1%}   (C1=30 baseline was 93.3%)")
    print(f"  progressive columns        : {acc_col:.1%}   (C1=30 baseline was 90.5%)")
    print(f"  ENSEMBLE (avg softmax)     : {acc_ens:.1%}")
    print("-" * 78)
    print(f"  ensemble gain vs best single : {acc_ens - max(acc_lin, acc_col):+.1%}")
    print(f"  BOTH wrong (correlated fail) : {both_wrong:.1%}   EITHER wrong (union) : {either_wrong:.1%}")
    print(f"  -> ensemble ceiling = 1 - (both wrong) = {1 - both_wrong:.1%} "
          f"(the fraction at least one gets right)")
    print("=" * 78)

    # self-check: features + readouts sane; report the honest ensemble gap (do NOT assert a big gain).
    assert acc_lin > 0.5 and acc_col > 0.5, "a readout failed to beat chance"
    print(f"self-check OK: linear {acc_lin:.1%}, columns {acc_col:.1%}, ensemble {acc_ens:.1%} "
          f"(gain {acc_ens - max(acc_lin, acc_col):+.1%}); both-wrong {both_wrong:.1%} = correlated-error floor")


if __name__ == "__main__":
    main()
