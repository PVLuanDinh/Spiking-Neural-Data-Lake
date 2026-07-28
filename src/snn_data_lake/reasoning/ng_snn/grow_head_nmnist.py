"""
research/ng_snn/grow_head_nmnist.py -- grow the NG-SNN CONTRIBUTION head on the 3-conv SoftHebb
mean+max N-MNIST features and test whether block-neurogenesis + subsampled scoring beats the
random-LIF head (98.84%) toward the linear feature ceiling (99.42%).

Features: data/ng_snn/nmnist_st3_feats_{train,test}.npy (60000x49152 / 10000x49152 float32) +
_labels.npy. Per-feature min-max to [0,1] on TRAIN stats. Xtr stays a CPU torch tensor (NGSNN._rates
batches N to the GPU); the grown S is (60000, L<=2000) < 0.5 GB.

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/grow_head_nmnist.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model import NGSNN

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

DATA = Path(__file__).resolve().parents[2] / "data" / "ng_snn"
RANDOM_LIF = 98.84  # prior random-LIF head on the same features
LINEAR_CEIL = 99.42  # linear readout ceiling on the same features


def load_split(split):
    fx = DATA / f"nmnist_st3_feats_{split}.npy"
    fy = DATA / f"nmnist_st3_feats_{split}_labels.npy"
    for p in (fx, fy):
        if not p.exists():
            raise FileNotFoundError(f"missing feature file: {p}")
    return np.load(fx), np.load(fy)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("WARNING: CUDA not available -- NGSNN will fall back to CPU (very slow for N-MNIST).")

    print("loading 3-conv mean+max N-MNIST features ...")
    Xtr_np, ytr_np = load_split("train")  # (60000, 49152) float32, (60000,)
    Xte_np, yte_np = load_split("test")  # (10000, 49152) float32, (10000,)

    # per-feature min-max to [0,1] on TRAIN stats (in-place -> no 11.8 GB copy)
    lo = Xtr_np.min(axis=0)
    hi = Xtr_np.max(axis=0)
    rng = np.where(hi > lo, hi - lo, np.float32(1.0)).astype(np.float32)
    for A in (Xtr_np, Xte_np):
        A -= lo
        A /= rng
        np.clip(A, 0.0, 1.0, out=A)

    Xtr = torch.from_numpy(Xtr_np)  # CPU float32 tensor (kept on CPU; _rates batches to GPU)
    ytr = torch.from_numpy(ytr_np).long()
    Xte = torch.from_numpy(Xte_np)
    yte = torch.from_numpy(yte_np).long()

    clf = NGSNN(
        lambdas=(0.05, 0.1, 0.3, 0.5), r=0.9, t_cand=200, l_max=2000, T=15, device="cuda", seed=0
    )

    print("growing NG-SNN head (delta=64 block-neurogenesis, score_subsample=8000) ...")
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr, delta=64, score_subsample=8000, verbose=True)
    dt = time.perf_counter() - t0

    # free the big train tensors before predict
    del Xtr, Xtr_np
    if dev == "cuda":
        torch.cuda.empty_cache()

    preds = clf.predict(Xte).cpu()  # predict moves the 10000-row test set to GPU (fits)
    acc = (preds == yte).float().mean().item()

    del Xte, Xte_np
    if dev == "cuda":
        torch.cuda.empty_cache()

    a = 100.0 * acc
    print("=" * 68)
    print("NG-SNN CONTRIBUTION-GROWTH head | N-MNIST (3-conv mean+max feats)")
    print(f"grown units L        : {clf.n_units}")
    print(f"N-MNIST test acc     : {a:.2f} %")
    print(f"wall-time (fit)      : {dt:.1f} s")
    print(f"vs random-LIF 98.84  : {a - RANDOM_LIF:+.2f}")
    print(f"vs linear ceil 99.42 : {a - LINEAR_CEIL:+.2f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
