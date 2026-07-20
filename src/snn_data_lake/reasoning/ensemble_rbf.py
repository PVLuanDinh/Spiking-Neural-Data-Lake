r"""
ENSEMBLE WITH THE RBF-SVM -- does adding the strong NONLINEAR RBF-SVM to the earlier
(linear + columns) ensemble finally give a gain?

Background: v3.61 ensembled the two LINEAR-boundary readouts (linear readout + progressive
columns) that share the same conv1 features -> CORRELATED errors -> the ensemble HURT.
v3.63 found the RBF-SVM (nonlinear) hits 96.2% on the same conv1 spatial features. A nonlinear
member makes DIFFERENT errors than the linear ones, so it is the decorrelated member an
ensemble actually needs. This measures every single readout + every ensemble combination on
the SAME frozen unsupervised conv1 features (C1=30, spatial), plus pairwise error-correlation.

Feature learning stays UNSUPERVISED (conv1 latency-STDP, frozen); only the readouts see labels.
No leakage: readouts train/fit on the train cache, scored on the held-out test cache.

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/ensemble_rbf.py
Knobs: NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(4) NORD_TRAIN(10000) NORD_TEST(2000)
       NORD_HIDDEN(32) NORD_CYCLES(15) NORD_SGD(300) NORD_PER(300) NORD_REPLAY(200)
Deps: numpy torch torchvision snntorch scikit-learn
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from snn_data_lake.reasoning.torch_device import resolve_device, enable
from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.reasoning.brain_adapters.laminar_readout import LaminarColumns


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _cache(conv, X, dev):
    return torch.stack([conv.features(img).reshape(-1) for img in X], 0).to(dev)


def _softmax_np(logits):
    return F.softmax(logits, dim=1).detach().cpu().numpy()


def _acc(P, y):
    return float((P.argmax(1) == y).mean())


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    C1 = _envi("NORD_C1", 30)
    T_LAT = _envi("NORD_T", 15)
    L1_EPOCHS = _envi("NORD_L1_EPOCHS", 4)
    TRAIN_N = _envi("NORD_TRAIN", 10000)
    TEST_N = _envi("NORD_TEST", 2000)
    HIDDEN = _envi("NORD_HIDDEN", 32)
    CYCLES = _envi("NORD_CYCLES", 15)
    SGD = _envi("NORD_SGD", 300)
    PER = _envi("NORD_PER", 300)
    REPLAY = _envi("NORD_REPLAY", 200)

    print("=" * 76)
    print("ENSEMBLE WITH THE RBF-SVM  (linear + columns + RBF on the same conv1 features)")
    print("=" * 76)
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    print(f"[perception] frozen; feat_dim={conv.feat_dim()}; diversity={conv.kernel_diversity():.3f}")

    print("[cache] conv1 spatial features (train+test)...")
    Ftr, Fte = _cache(conv, Xtr, dev), _cache(conv, Xte, dev)
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    yte_np = Yte.numpy()
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    # ---- member 1: LINEAR readout (static joint, torch nn.Linear) ----
    torch.manual_seed(0)
    lin = nn.Linear(D, 10).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy(lin(Ftr), ytr).backward()
        opt.step()
    with torch.no_grad():
        P_lin = _softmax_np(lin(Fte))

    # ---- member 2: progressive COLUMNS (continual developmental schedule) ----
    col = LaminarColumns(D, n_classes=10, n23=HIDDEN, seed=0)
    memory, learned, comp = {}, [], {d: 0.0 for d in range(10)}
    for cyc in range(1, CYCLES + 1):
        obj = (cyc - 1) if cyc <= 10 else min(learned, key=lambda d: comp[d])
        is_new = obj not in learned
        n_train = PER if is_new else max(REPLAY, PER // 4)
        rows = tr_by[obj][:n_train] + [r for d in learned if d != obj for r in memory.get(d, [])]
        idx = torch.tensor(rows, dtype=torch.long, device=dev)
        if is_new and len(learned) >= 1:
            col.grow(1, freeze_old=True)
        col.train_step(Ftr[idx], ytr[idx], lr=0.01, steps=SGD)
        if is_new:
            memory[obj] = tr_by[obj][n_train:n_train + REPLAY]
            learned.append(obj)
        with torch.no_grad():
            cl = col.predict(Fte)
        for d in learned:
            m = (Yte == d)
            comp[d] = float((cl[m.to(dev)].argmax(1) == yte[m.to(dev)]).float().mean())
    with torch.no_grad():
        P_col = _softmax_np(col.predict(Fte))

    # ---- member 3: RBF-SVM (nonlinear, sklearn, probability) ----
    print("[svm] fitting RBF-SVC on conv1 spatial features...")
    Xtr_np, Xte_np = Ftr.cpu().numpy(), Fte.cpu().numpy()
    sc = StandardScaler().fit(Xtr_np)
    # decision_function + softmax (an ensemble-usable probability surrogate) instead of predict_proba --
    # skips the slow Platt CV so RBF-SVC scales to larger train sets (30k+) in reasonable time.
    rbf = SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=0)
    rbf.fit(sc.transform(Xtr_np), Ytr.numpy())
    dec = rbf.decision_function(sc.transform(Xte_np))               # [N,10] one-vs-rest scores
    dec = dec - dec.max(axis=1, keepdims=True)
    P_rbf = np.exp(dec) / np.exp(dec).sum(axis=1, keepdims=True)     # softmax -> probability surrogate

    # ---- singles ----
    a_lin, a_col, a_rbf = _acc(P_lin, yte_np), _acc(P_col, yte_np), _acc(P_rbf, yte_np)

    # ---- ensembles (probability average) ----
    def ens(*Ps):
        return _acc(sum(Ps) / len(Ps), yte_np)

    e_old = ens(P_lin, P_col)                 # the earlier ensemble (linear + columns)
    e_new = ens(P_lin, P_col, P_rbf)          # + RBF
    e_lr = ens(P_lin, P_rbf)                  # linear + RBF
    e_cr = ens(P_col, P_rbf)                  # columns + RBF
    e_all = e_new

    # ---- error correlation (both-wrong) for the informative pairs ----
    def both_wrong(P, Q):
        return float(((P.argmax(1) != yte_np) & (Q.argmax(1) != yte_np)).mean())

    print("\n" + "=" * 76)
    print("RESULT  (all on the SAME frozen conv1 spatial features; chance = 10%)")
    print("-" * 76)
    print("  SINGLES:")
    print(f"    linear readout            : {a_lin:.1%}")
    print(f"    progressive columns       : {a_col:.1%}")
    print(f"    RBF-SVM (nonlinear)       : {a_rbf:.1%}")
    print("  ENSEMBLES (probability average):")
    print(f"    linear + columns  (OLD)   : {e_old:.1%}   ({e_old - max(a_lin, a_col):+.1%} vs best member)")
    print(f"    linear + RBF              : {e_lr:.1%}   ({e_lr - max(a_lin, a_rbf):+.1%} vs best member)")
    print(f"    columns + RBF             : {e_cr:.1%}   ({e_cr - max(a_col, a_rbf):+.1%} vs best member)")
    print(f"    linear + columns + RBF    : {e_new:.1%}   ({e_new - a_rbf:+.1%} vs RBF alone)")
    print("  ERROR CORRELATION (both wrong, lower = more decorrelated):")
    print(f"    linear vs columns         : {both_wrong(P_lin, P_col):.1%}")
    print(f"    linear vs RBF             : {both_wrong(P_lin, P_rbf):.1%}")
    print(f"    columns vs RBF            : {both_wrong(P_col, P_rbf):.1%}")
    print("=" * 76)
    best = max(a_lin, a_col, a_rbf, e_old, e_new, e_lr, e_cr)
    print(f"self-check OK: best single RBF {a_rbf:.1%}; best ensemble {max(e_old, e_new, e_lr, e_cr):.1%}; "
          f"overall best {best:.1%} (honest -- ensemble helps only if it beats {a_rbf:.1%})")
    assert a_rbf > 0.5, "RBF-SVM failed to beat chance"


if __name__ == "__main__":
    main()
