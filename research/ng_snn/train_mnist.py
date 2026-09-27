"""
research/ng_snn/train_mnist.py -- attach the validated NG-SNN head to the FROZEN SoftHebb-CNN
features and measure MNIST vs the paper's 99.19% (Table 2, NG-SNN + unsupervised front-end).

Pipeline (the front-end + features are produced separately by softhebb.py):
  frozen SoftHebb features (data/ng_snn/mnist_feats_{train,test}.npy)
    -> per-feature min-max to [0,1] (fit on train)
    -> NG-SNN grown-LIF head in CONSTANT-CURRENT feature mode (model.py, feature input), 1 pass
    -> argmax over 10 outputs.

High-dim note: candidate weights are drawn from U(-Lam,Lam); at d_FE ~ 18k the pre-activation
A = W f grows with d, so we scale the nominal lambdas by 1/sqrt(d_FE) to keep the LIF operating
point (A vs v_th) dimension-invariant -- the same nominal {0.1,1,5} that worked on UCI then apply.

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/train_mnist.py [T ...]
"""
import sys
import time

import numpy as np
import torch

from snn_data_lake.growth.ng_snn import NGSNN

FDIR = "../../data/ng_snn"


def load_feats():
    tr = np.load(f"{FDIR}/mnist_feats_train.npy")
    ytr = np.load(f"{FDIR}/mnist_feats_train_labels.npy")
    te = np.load(f"{FDIR}/mnist_feats_test.npy")
    yte = np.load(f"{FDIR}/mnist_feats_test_labels.npy")
    return tr, ytr, te, yte


def main(Ts):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Ftr, ytr, Fte, yte = load_feats()
    d = Ftr.shape[1]
    lo = Ftr.min(0, keepdims=True)                              # per-feature min-max on TRAIN -> [0,1]
    hi = Ftr.max(0, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    Ftr = np.clip((Ftr - lo) / rng, 0, 1).astype(np.float32)
    Fte = np.clip((Fte - lo) / rng, 0, 1).astype(np.float32)
    Ftr = torch.tensor(Ftr, device=dev)
    Fte = torch.tensor(Fte, device=dev)
    ytr = torch.tensor(ytr.astype(np.int64), device=dev)
    yte = torch.tensor(yte.astype(np.int64), device=dev)

    # Candidate scales tuned to the LIF operating point (diagnosed): features min-max'd to [0,1] with
    # mean ~0.2 over 18.8k dims; scale ~0.1-0.3 puts the pre-activation A=W f astride v_th so firing
    # rates are GRADED/discriminative (5-14% dead). Smaller -> all neurons silent (the 93.8% failure).
    lambdas = (0.05, 0.1, 0.3)
    print("=" * 78)
    print(f"NG-SNN head on SoftHebb features | d_FE={d}  train={Ftr.shape[0]}  test={Fte.shape[0]}")
    print(f"lambdas(/sqrt d)={[f'{x:.2e}' for x in lambdas]}  device={dev}")
    print("-" * 78)
    best = 0.0
    for T in Ts:
        t0 = time.perf_counter()
        clf = NGSNN(lambdas=lambdas, r=0.9, t_cand=50, l_max=200, T=T, device=dev, seed=0)
        clf.fit(Ftr, ytr)
        acc = 100.0 * (clf.predict(Fte) == yte).float().mean().item()
        best = max(best, acc)
        print(f"T={T:2d}: MNIST test {acc:.2f}%  (grew {clf.n_units} LIF, 1 pass, "
              f"{time.perf_counter() - t0:.0f}s)  |  paper 99.19%  gap {acc - 99.19:+.2f}")
    print("=" * 78)
    print(f"best {best:.2f}% vs paper 99.19% (Tang et al., NG-SNN, Neural Networks 2026, Table 2)")


if __name__ == "__main__":
    Ts = [int(a) for a in sys.argv[1:]] or [10, 15, 20]
    main(Ts)
