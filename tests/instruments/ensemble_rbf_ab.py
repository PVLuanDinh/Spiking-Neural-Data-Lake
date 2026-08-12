r"""The readout-ensemble A/B (2026-08-12, #10 ensemble_rbf settlement): the harness that
produced the tree's 98.2% Kheradpisheh-class headline, settled as an INSTRUMENT over
governed subjects (the jepa_highdim_probe pattern -- it holds no mechanism of its own).

Every subject is governed: `plasticity.snn_mnist_stdp.load_mnist`,
`cortex.deep_stdp_latency.ConvLatencyStdp/_train` (the unsupervised conv1 perception),
`hippocampus.growable_readout.LaminarColumns` (governed at the 2026-08-12 laminar merge),
`substrate.device`. The linear member is stock nn.Linear+Adam, the RBF member is stock
sklearn SVC, the fusion is an unweighted probability average -- measurements, not
mechanisms.

THE CREDITED VERDICT (durable in CHANGELOG v3.65/v3.64, which this instrument reproduces
rather than owns): on MNIST 30000/10000, linear 97.8% / columns 93.1% / RBF-SVM 98.0%;
linear+RBF = 98.2%, the NEW BEST -- because linear-vs-RBF is the most DECORRELATED pair
(both-wrong 1.3%), while the columns member "drags every ensemble it is in, so it must be
EXCLUDED" (v3.64). The columns member is kept HERE as that recorded negative's live
contrast, not as a contributor to the headline.

SCALE-DEPENDENCE, disclosed rather than smoothed: the ensemble-beats-best-member gain is
MEASURED FALSE below ~10k labelled examples (at 1200/400: linear+RBF 79.5% LOSES to RBF
alone 81.2%) -- that threshold is itself the "data lever" finding v3.65 is named for. So
the bare run asserts the DECORRELATION ORDERING (which holds at BOTH scales: both-wrong
lin-vs-RBF 14.8% < lin-vs-col 19.5% reduced; 1.3% < 1.7% credited) plus pipeline
integrity, and the beats-best-member assert lives behind ENSEMBLE_RUN=1 only.

Run shapes (the deep_stdp DEEPSTDP_RUN precedent):
  PYTHONPATH=src python tests/instruments/ensemble_rbf_ab.py
      -> bounded mechanics+structure check, ~35-60s GPU (train 1200 / test 400, 1 epoch)
  ENSEMBLE_RUN=1 PYTHONPATH=src python tests/instruments/ensemble_rbf_ab.py
      -> the full default-knob measurement (~10 min GPU; 10000/2000)
  ENSEMBLE_RUN=1 NORD_TRAIN=30000 NORD_TEST=10000 ...  -> the ~31-min credited-headline
      reproduction (figures already durable in CHANGELOG v3.65)

Honest tier scope: tests/instruments/ is collected by NO automated gate (pytest's
python_files pattern skips it; verified: --collect-only exits 5) -- instruments are run
by hand and checked statically by tools/instrument_guard.py, same as the five incumbents.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from snn_data_lake.cortex.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.hippocampus.growable_readout import LaminarColumns
from snn_data_lake.plasticity.snn_mnist_stdp import _env, load_mnist
from snn_data_lake.substrate.device import enable, resolve_device


def _cache(conv, X, dev):
    return torch.stack([conv.features(img).reshape(-1) for img in X], 0).to(dev)


def _softmax_np(logits):
    return F.softmax(logits, dim=1).detach().cpu().numpy()


def _acc(P, y):
    return float((P.argmax(1) == y).mean())


def main():
    FULL = os.environ.get("ENSEMBLE_RUN") == "1"
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    # Bounded mechanics knobs by default; ENSEMBLE_RUN=1 restores the credited defaults.
    C1 = _env("NORD_C1", 30)
    T_LAT = _env("NORD_T", 15)
    L1_EPOCHS = _env("NORD_L1_EPOCHS", 4 if FULL else 1)
    TRAIN_N = _env("NORD_TRAIN", 10000 if FULL else 1200)
    TEST_N = _env("NORD_TEST", 2000 if FULL else 400)
    HIDDEN = _env("NORD_HIDDEN", 32)
    CYCLES = _env("NORD_CYCLES", 15 if FULL else 12)
    SGD = _env("NORD_SGD", 300 if FULL else 150)
    PER = _env("NORD_PER", 300 if FULL else 100)
    REPLAY = _env("NORD_REPLAY", 200 if FULL else 60)

    print("=" * 76)
    print("ENSEMBLE WITH THE RBF-SVM  (linear + columns + RBF on the same conv1 features)")
    shape = "FULL measurement (ENSEMBLE_RUN=1)" if FULL else "bounded mechanics check"
    print(f"{shape}: train {TRAIN_N} / test {TEST_N}")
    print("=" * 76)
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    print(
        f"[perception] frozen; feat_dim={conv.feat_dim()}; diversity={conv.kernel_diversity():.3f}"
    )

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
    # Kept as the recorded NEGATIVE's live contrast (v3.64: it drags every ensemble it is
    # in), and as the decorrelation criterion's operand. Governed home since 2026-08-12.
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
            memory[obj] = tr_by[obj][n_train : n_train + REPLAY]
            learned.append(obj)
        with torch.no_grad():
            cl = col.predict(Fte)
        for d in learned:
            m = Yte == d
            comp[d] = float((cl[m.to(dev)].argmax(1) == yte[m.to(dev)]).float().mean())
    with torch.no_grad():
        P_col = _softmax_np(col.predict(Fte))

    # ---- member 3: RBF-SVM (nonlinear, sklearn, probability) ----
    print("[svm] fitting RBF-SVC on conv1 spatial features...")
    Xtr_np, Xte_np = Ftr.cpu().numpy(), Fte.cpu().numpy()
    sc = StandardScaler().fit(Xtr_np)
    # decision_function + softmax (an ensemble-usable probability surrogate) instead of
    # predict_proba -- skips the slow Platt CV so RBF-SVC scales to larger train sets (30k+).
    rbf = SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=0)
    rbf.fit(sc.transform(Xtr_np), Ytr.numpy())
    dec = rbf.decision_function(sc.transform(Xte_np))  # [N,10] one-vs-rest scores
    dec = dec - dec.max(axis=1, keepdims=True)
    P_rbf = np.exp(dec) / np.exp(dec).sum(axis=1, keepdims=True)  # softmax -> prob surrogate

    # ---- singles ----
    a_lin, a_col, a_rbf = _acc(P_lin, yte_np), _acc(P_col, yte_np), _acc(P_rbf, yte_np)

    # ---- ensembles (probability average) ----
    def ens(*Ps):
        return _acc(sum(Ps) / len(Ps), yte_np)

    e_old = ens(P_lin, P_col)  # the earlier ensemble (linear + columns)
    e_new = ens(P_lin, P_col, P_rbf)  # + RBF
    e_lr = ens(P_lin, P_rbf)  # linear + RBF
    e_cr = ens(P_col, P_rbf)  # columns + RBF

    # ---- error correlation (both-wrong) for the informative pairs ----
    def both_wrong(P, Q):
        return float(((P.argmax(1) != yte_np) & (Q.argmax(1) != yte_np)).mean())

    bw_lc = both_wrong(P_lin, P_col)
    bw_lr = both_wrong(P_lin, P_rbf)
    bw_cr = both_wrong(P_col, P_rbf)

    print("\n" + "=" * 76)
    print("RESULT  (all on the SAME frozen conv1 spatial features; chance = 10%)")
    print("-" * 76)
    print("  SINGLES:")
    print(f"    linear readout            : {a_lin:.1%}")
    print(f"    progressive columns       : {a_col:.1%}")
    print(f"    RBF-SVM (nonlinear)       : {a_rbf:.1%}")
    print("  ENSEMBLES (probability average):")
    print(
        f"    linear + columns  (OLD)   : {e_old:.1%}   "
        f"({e_old - max(a_lin, a_col):+.1%} vs best member)"
    )
    print(
        f"    linear + RBF              : {e_lr:.1%}   "
        f"({e_lr - max(a_lin, a_rbf):+.1%} vs best member)"
    )
    print(
        f"    columns + RBF             : {e_cr:.1%}   "
        f"({e_cr - max(a_col, a_rbf):+.1%} vs best member)"
    )
    print(f"    linear + columns + RBF    : {e_new:.1%}   ({e_new - a_rbf:+.1%} vs RBF alone)")
    print("  ERROR CORRELATION (both wrong, lower = more decorrelated):")
    print(f"    linear vs columns         : {bw_lc:.1%}")
    print(f"    linear vs RBF             : {bw_lr:.1%}")
    print(f"    columns vs RBF            : {bw_cr:.1%}")
    print("=" * 76)
    best = max(a_lin, a_col, a_rbf, e_old, e_new, e_lr, e_cr)

    # (1) PIPELINE-INTEGRITY red: neuter conv.features and this collapses to ~chance.
    assert a_rbf > 0.5, f"RBF-SVM failed to beat chance ({a_rbf:.3f} <= 0.5, chance is 0.10)"

    # (2) THE DECORRELATION ORDERING -- the mechanism the credited 98.2% actually rests on,
    # and the criterion that survives shrinking (measured: 14.8% < 19.5% at 1200/400;
    # 1.3% < 1.7% credited at 30k/10k). Reachable red: swap the RBF member for a second
    # linear readout and the ordering collapses (two linear members correlate).
    assert bw_lr < bw_lc, (
        f"linear-vs-RBF must be MORE decorrelated than linear-vs-columns "
        f"(both-wrong {bw_lr:.3f} vs {bw_lc:.3f}) -- the nonlinear member is the point"
    )

    if FULL:
        # (3) The credited claim itself -- FULL RUN ONLY, because it is MEASURED FALSE
        # below ~10k labelled examples (79.5% < 81.2% at 1200/400): the data-lever finding.
        assert e_lr > max(a_lin, a_rbf), (
            f"at full scale linear+RBF must beat its best member "
            f"({e_lr:.4f} vs {max(a_lin, a_rbf):.4f}) -- the v3.64/v3.65 credited gain"
        )
        print(
            f"self-check OK (FULL): linear+RBF {e_lr:.1%} beats best member "
            f"{max(a_lin, a_rbf):.1%}; decorrelation ordering holds; overall best {best:.1%}"
        )
    else:
        print(
            f"self-check OK (bounded): pipeline live (RBF {a_rbf:.1%}), decorrelation "
            f"ordering holds ({bw_lr:.1%} < {bw_lc:.1%}); the beats-best-member claim needs "
            f">=10k train and lives behind ENSEMBLE_RUN=1 (measured: at this scale the "
            f"ensemble {e_lr:.1%} does NOT beat RBF alone {a_rbf:.1%} -- the data lever)"
        )


if __name__ == "__main__":
    main()
