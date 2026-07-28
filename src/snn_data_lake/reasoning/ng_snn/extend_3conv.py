"""
Extend the verified 2-conv SoftHebb front-end to 3 conv blocks (99.35% linear ceiling, needed to
reproduce NG-SNN's MNIST 99.19%). Reuses the FROZEN blocks 0/1 from softhebb_mnist.pt; trains ONLY
block 2 (1536ch) unsupervised (1 epoch, greedy); extracts the 13824-d frozen features + linear sanity.

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/extend_3conv.py
"""

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from softhebb_mnist import SoftHebbCNN

DATA = "../../data"
OUT = "../../data/ng_snn"


def extract(model, ds, dev):
    ld = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    feats, labs = [], []
    with torch.no_grad():
        for x, y in ld:
            feats.append(model.features(x.to(dev)).cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.ToTensor()
    tr = datasets.MNIST(DATA, train=True, download=False, transform=tf)
    te = datasets.MNIST(DATA, train=False, download=False, transform=tf)

    model = SoftHebbCNN(n_blocks=3, seed=0).to(dev)
    sd = torch.load("softhebb_mnist.pt", map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(
        sd, strict=False
    )  # block 2 absent from ckpt -> stays at init
    print(f"loaded 2-conv ckpt | missing(block2)={len(missing)} unexpected={len(unexpected)}")

    # GREEDY: freeze blocks 0/1 (eval BN + no plasticity); train ONLY block 2 (train BN + plasticity on).
    model.eval()
    for blk in model.blocks:
        blk.conv.learning = False
    model.blocks[2].train()
    model.blocks[2].conv.learning = True

    ld = DataLoader(tr, batch_size=32, shuffle=True, num_workers=4, drop_last=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, _ in ld:
            model(x.to(dev))  # block 2 learns locally per batch
    frac, rmean, rstd = model.blocks[2].conv.convergence()
    print(
        f"block2 trained 1 epoch [{time.perf_counter() - t0:.0f}s] | ||w||_p converge frac={frac:.2f} "
        f"r_mean={rmean:.3f} r_std={rstd:.3f}"
    )

    # FREEZE all + extract features
    model.eval()
    for blk in model.blocks:
        blk.conv.learning = False
    Ftr, ytr = extract(model, tr, dev)
    Fte, yte = extract(model, te, dev)
    d = Ftr.shape[1]
    print(f"d_FE={d} | train {Ftr.shape} test {Fte.shape}")
    assert d == 13824, f"expected 13824, got {d}"

    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/mnist_feats3_train.npy", Ftr.astype(np.float32))
    np.save(f"{OUT}/mnist_feats3_train_labels.npy", ytr)
    np.save(f"{OUT}/mnist_feats3_test.npy", Fte.astype(np.float32))
    np.save(f"{OUT}/mnist_feats3_test_labels.npy", yte)
    torch.save(model.state_dict(), "softhebb_mnist_3conv.pt")

    # LINEAR SANITY (ridge readout, fast lower bound)
    lo, hi = Ftr.min(0, keepdims=True), Ftr.max(0, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    Xtr = torch.tensor(np.clip((Ftr - lo) / rng, 0, 1), dtype=torch.float32, device=dev)
    Xte = torch.tensor(np.clip((Fte - lo) / rng, 0, 1), dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr.astype(np.int64), device=dev)
    ye = torch.tensor(yte.astype(np.int64), device=dev)
    Y = torch.zeros(Xtr.shape[0], 10, device=dev)
    Y[torch.arange(Xtr.shape[0]), yt] = 1.0
    W = torch.linalg.solve(Xtr.t() @ Xtr + 1e2 * torch.eye(d, device=dev), Xtr.t() @ Y)
    acc = 100.0 * ((Xte @ W).argmax(1) == ye).float().mean().item()
    print(
        f"3-conv LINEAR-PROBE (ridge) MNIST test = {acc:.2f}%  (target ~99.35%; 2-conv was 99.01%)"
    )


if __name__ == "__main__":
    main()
