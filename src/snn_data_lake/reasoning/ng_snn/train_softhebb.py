"""
Train the unsupervised SoftHebb-CNN front-end on MNIST (greedy layer-wise,
1 epoch/block, no labels), run the throwaway Linear sanity head, then extract
and save the frozen d_FE features for train (60k) and test (10k).

Run with the cu128 GPU venv:
  PYTHONUTF8=1 .venv-cu128/Scripts/python.exe research/ng_snn/train_softhebb.py

Optional args (env):
  NG_SNN_BLOCKS = 2 (default) or 3
  NG_SNN_UNSUP_BS = 32 (unsupervised mini-batch)
ASCII-only output (cp1252-safe).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

pass  # migrated: sys.path bootstrap removed (now an importable package)
from softhebb_mnist import SoftHebbCNN  # noqa: E402

# ---- paths (relative to project root = two levels up from this file) --------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(DATA_ROOT, "ng_snn")
CKPT = os.path.join(HERE, "softhebb_mnist.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BLOCKS = int(os.environ.get("NG_SNN_BLOCKS", "2"))
UNSUP_BS = int(os.environ.get("NG_SNN_UNSUP_BS", "32"))
SEED = 0


def log(msg):
    print(msg, flush=True)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_datasets():
    tf = transforms.ToTensor()  # -> 1x28x28 in [0,1]; NO normalize, NO aug
    train = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    return train, test


# ----------------------------------------------------------------------------
# Greedy layer-wise unsupervised training
# ----------------------------------------------------------------------------
def train_block(model, k, train_ds):
    """Train block k's HebbConv while upstream blocks 0..k-1 stay frozen (eval)."""
    g = torch.Generator().manual_seed(SEED + k)
    loader = DataLoader(train_ds, batch_size=UNSUP_BS, shuffle=True,
                        num_workers=0, generator=g)
    model.eval()                       # all BN use running stats, no updates
    blk = model.blocks[k]
    blk.train()                        # only this block's BN accumulates stats
    blk.conv.learning = True
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            for j in range(k):
                x = model.blocks[j](x)  # frozen upstream (eval, no learning)
            blk(x)                      # local plasticity update happens here
            n += 1
    blk.conv.learning = False
    model.eval()
    frac, rmean, rstd = blk.conv.convergence(tol=5e-3)
    log("  block %d trained: %d batches, %.1fs | ||w||_p mean=%.4f std=%.4f | "
        "frac|r-1|<5e-3 = %.3f" % (k, n, time.time() - t0, rmean, rstd, frac))
    return frac, rmean, rstd


def infer_d_fe(model, test_ds):
    model.eval()
    x = torch.stack([test_ds[i][0] for i in range(2)]).to(DEVICE)
    with torch.no_grad():
        f = model.features(x)
    return f.shape[1]


# ----------------------------------------------------------------------------
# Feature extraction + save (memmap -> .npy on disk)
# ----------------------------------------------------------------------------
def extract_and_save(model, ds, d_fe, feat_path, label_path, bs=500):
    model.eval()
    n = len(ds)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    feats = np.lib.format.open_memmap(
        feat_path, mode="w+", dtype=np.float32, shape=(n, d_fe)
    )
    labels = np.empty((n,), dtype=np.int64)
    i = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            f = model.features(x).float().cpu().numpy()
            b = f.shape[0]
            feats[i:i + b] = f
            labels[i:i + b] = y.numpy()
            i += b
    feats.flush()
    del feats
    np.save(label_path, labels)
    return n


# ----------------------------------------------------------------------------
# Throwaway supervised Linear sanity head on FROZEN features
# ----------------------------------------------------------------------------
def sanity_linear(feat_train, lab_train, feat_test, lab_test, d_fe,
                  max_epochs=30, bs=64, patience=6):
    # Load features onto GPU once (fits: ~4.5GB train + ~0.75GB test for 2-conv).
    Xtr = torch.from_numpy(np.load(feat_train)).to(DEVICE)
    ytr = torch.from_numpy(np.load(lab_train)).to(DEVICE)
    Xte = torch.from_numpy(np.load(feat_test)).to(DEVICE)
    yte = torch.from_numpy(np.load(lab_test)).to(DEVICE)

    head = nn.Sequential(nn.Dropout(0.5), nn.Linear(d_fe, 10)).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    ntr = Xtr.shape[0]

    best = 0.0
    best_ep = -1
    since = 0
    for ep in range(max_epochs):
        head.train()
        perm = torch.randperm(ntr, device=DEVICE)
        for s in range(0, ntr, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            out = head(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
        # eval on test
        head.eval()
        with torch.no_grad():
            correct = 0
            for s in range(0, Xte.shape[0], 1000):
                out = head(Xte[s:s + 1000])
                correct += (out.argmax(1) == yte[s:s + 1000]).sum().item()
            acc = 100.0 * correct / Xte.shape[0]
        if acc > best:
            best, best_ep, since = acc, ep, 0
        else:
            since += 1
        log("  head epoch %2d: test acc = %.2f%% (best %.2f%% @ep%d)"
            % (ep, acc, best, best_ep))
        if since >= patience:
            log("  early stop (no test improvement for %d epochs)" % patience)
            break
    return best


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUT_DIR, exist_ok=True)
    log("device=%s | blocks=%d | unsup_bs=%d | seed=%d" %
        (DEVICE, N_BLOCKS, UNSUP_BS, SEED))
    if DEVICE == "cuda":
        log("gpu=%s | torch=%s" % (torch.cuda.get_device_name(0), torch.__version__))

    train_ds, test_ds = get_datasets()
    log("MNIST: train=%d test=%d (ToTensor, no normalize)" %
        (len(train_ds), len(test_ds)))

    model = SoftHebbCNN(n_blocks=N_BLOCKS, seed=SEED).to(DEVICE)

    # ---- greedy layer-wise unsupervised training (NO labels) ----
    log("--- unsupervised greedy layer-wise training (1 epoch/block) ---")
    conv_report = []
    t0 = time.time()
    for k in range(N_BLOCKS):
        conv_report.append(train_block(model, k, train_ds))
    log("total unsupervised train time: %.1fs" % (time.time() - t0))

    # ---- freeze whole extractor + save state_dict ----
    for blk in model.blocks:
        blk.conv.learning = False
    model.eval()
    torch.save(model.state_dict(), CKPT)
    log("saved frozen extractor state_dict -> %s" % CKPT)

    # ---- assert d_FE from a real forward pass ----
    d_fe = infer_d_fe(model, test_ds)
    log("measured d_FE (flattened feature dim) = %d" % d_fe)

    # ---- extract + save frozen features ----
    ftr = os.path.join(OUT_DIR, "mnist_feats_train.npy")
    ltr = os.path.join(OUT_DIR, "mnist_feats_train_labels.npy")
    fte = os.path.join(OUT_DIR, "mnist_feats_test.npy")
    lte = os.path.join(OUT_DIR, "mnist_feats_test_labels.npy")
    log("--- extracting + saving frozen features ---")
    t0 = time.time()
    ntr = extract_and_save(model, train_ds, d_fe, ftr, ltr)
    nte = extract_and_save(model, test_ds, d_fe, fte, lte)
    log("extracted train=%d test=%d in %.1fs" % (ntr, nte, time.time() - t0))

    # ---- throwaway Linear sanity head (must reach >= 98.8%) ----
    log("--- Linear sanity head on FROZEN features (Adam 1e-3, dropout 0.5, bs64) ---")
    best = sanity_linear(ftr, ltr, fte, lte, d_fe)

    # ---- final report ----
    log("=================== SUMMARY ===================")
    log("conv blocks used                : %d" % N_BLOCKS)
    log("d_FE                            : %d" % d_fe)
    for k, (frac, rmean, rstd) in enumerate(conv_report):
        log("block %d ||w||_p converge frac    : %.3f (mean r=%.4f, std=%.4f)"
            % (k, frac, rmean, rstd))
    log("LINEAR SANITY MNIST TEST ACC    : %.2f%%  (target >= 98.8%%)" % best)
    log("features (train)                : %s  shape=(%d, %d) float32" % (ftr, ntr, d_fe))
    log("labels   (train)                : %s  shape=(%d,) int64" % (ltr, ntr))
    log("features (test)                 : %s  shape=(%d, %d) float32" % (fte, nte, d_fe))
    log("labels   (test)                 : %s  shape=(%d,) int64" % (lte, nte))
    log("PASS" if best >= 98.8 else "FAIL - front-end needs debugging")


if __name__ == "__main__":
    main()
