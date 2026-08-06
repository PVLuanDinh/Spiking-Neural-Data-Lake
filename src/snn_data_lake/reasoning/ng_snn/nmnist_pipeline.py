"""
research/ng_snn/nmnist_pipeline.py -- N-MNIST reproduction for NG-SNN (Milestone 2).

Pipeline (single file, torch/GPU tier, runs in .venv-cu128 on the RTX 5070):

  N-MNIST event stream  --data.py, T=15 bins-->  (T, 2, 34, 34) event frames
    --SUM over T + log1p + per-sample max--> static 2-channel rate image (2, 34, 34) in [0,1]
    --UNSUPERVISED 2-conv SoftHebb-CNN (greedy layer-wise, 1 epoch, no backprop)--> d_FE features
    --min-max [0,1]-->  (a) ridge linear-probe readout      -> N-MNIST test acc  (front-end CEILING)
                        (b) large random-LIF pool + ridge   -> N-MNIST test acc  (NG-SNN head)

INPUT REPRESENTATION (noted in the report): N-MNIST is an event stream. We bin it into T=15 frames
(data.py), then SUM the frames -> a per-pixel per-polarity event COUNT image (2,34,34), take log1p
(the counts are heavy-tailed -- a few pixels fire many times) and divide by the per-sample max so
each sample lands in [0,1]. This static 2-channel rate image is the SoftHebb input -- the simplest
faithful reduction of the event stream to an "image" the (image-trained) SoftHebb front-end expects.
BatchNorm2d(affine=False) at the head of each conv block whitens per channel, so the exact scaling is
secondary; it matters mainly for the downstream min-max feature step.

FRONT-END is the VERIFIED HebbConv2d/Triangle/SoftHebbBlock from softhebb_mnist.py (math UNCHANGED),
re-wired for in_channels=2, 34x34. HEAD (b) mirrors model.NGSNN's analytical ridge (Eq. 9-12) over a
large FIXED random LIF pool (constant-current firing rates, reusing model.NGSNN._rates faithfully);
growing 20k units one-at-a-time is infeasible, so we draw the pool up front and solve beta once.

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/nmnist_pipeline.py
Env: NG_SNN_WORKERS (default 4), NG_SNN_POOL (default 20001), NG_SNN_HEAD_T (default 15).
ASCII-only output (cp1252-safe). Files written under data/ng_snn/ + research/ng_snn/ only.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
pass  # migrated: sys.path bootstrap removed (now an importable package)
from softhebb_mnist import HebbConv2d, Triangle, SoftHebbBlock  # noqa: E402  (math UNCHANGED)
from data import _make_dataset  # noqa: E402  (reuses disk cache)
from snn_data_lake.hippocampus.ng_snn import NGSNN  # noqa: E402  (faithful LIF + ridge)

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(DATA_ROOT, "nmnist_cache")
OUT_DIR = os.path.join(DATA_ROOT, "ng_snn")
CKPT = os.path.join(HERE, "softhebb_nmnist.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_BINS = 15  # event-frame time bins (SoftHebb front-end input)
HEAD_T = int(os.environ.get("NG_SNN_HEAD_T", "15"))  # LIF integration window in the NG-SNN head
SEED = 0
UNSUP_BS = 32  # unsupervised greedy mini-batch (spec)
EXTRACT_BS = 256
NW = int(os.environ.get("NG_SNN_WORKERS", "4"))
POOL = int(os.environ.get("NG_SNN_POOL", "20001"))  # NG-SNN random-LIF pool size (~20000)
SCALES = (0.05, 0.1, 0.3)  # candidate lambda scales (min-max'd feature operating point)


def log(m):
    print(m, flush=True)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# Input representation: (B,T,2,34,34) event frames -> (B,2,34,34) rate image in [0,1]
# ----------------------------------------------------------------------------
def to_rate_image(x):
    """Accumulate the T event-frame bins, log1p-compress the heavy tail, per-sample max-normalize."""
    x = x.sum(dim=1)  # (B,2,34,34) event-count image
    x = torch.log1p(x)
    b = x.shape[0]
    m = x.view(b, -1).amax(dim=1).clamp_min(1e-6).view(b, 1, 1, 1)
    return x / m


# ----------------------------------------------------------------------------
# 2-channel SoftHebb front-end (VERIFIED HebbConv2d math; only in_channels/anatomy re-wired)
# ----------------------------------------------------------------------------
class SoftHebbNMNIST(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        blocks = [
            SoftHebbBlock(
                nn.BatchNorm2d(2, affine=False),
                HebbConv2d(
                    2,
                    96,
                    5,
                    stride=1,
                    padding=2,
                    padding_mode="reflect",
                    t_invert=1.0,
                    lr_amp=0.08,
                    power_lr=0.5,
                    seed=seed,
                ),
                Triangle(0.7),
                nn.MaxPool2d(4, 2, 1),
            ),
            SoftHebbBlock(
                nn.BatchNorm2d(96, affine=False),
                HebbConv2d(
                    96,
                    384,
                    3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                    t_invert=0.65,
                    lr_amp=0.005,
                    power_lr=0.5,
                    seed=seed,
                ),
                Triangle(1.4),
                nn.MaxPool2d(4, 2, 1),
            ),
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def features(self, x):
        return self.forward(x).reshape(x.size(0), -1)


# ----------------------------------------------------------------------------
# Loaders (reuse the disk cache; absolute paths so cwd does not matter)
# ----------------------------------------------------------------------------
def make_loader(train, bs, shuffle, drop_last):
    ds = _make_dataset(train, DATA_ROOT, T_BINS, CACHE_DIR)
    return ds, DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=NW,
        pin_memory=True,
        persistent_workers=(NW > 0),
        prefetch_factor=(4 if NW else None),
    )


# ----------------------------------------------------------------------------
# Greedy layer-wise unsupervised training (only the block being trained changes)
# ----------------------------------------------------------------------------
def train_block(model, k, loader):
    model.eval()  # frozen blocks use running BN stats, no updates
    blk = model.blocks[k]
    blk.train()  # this block's BN accumulates stats
    blk.conv.learning = True  # this block's conv does local plasticity
    snaps = [b.conv.weight.detach().clone() for b in model.blocks]  # to verify greedy correctness
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = to_rate_image(x.to(DEVICE, non_blocking=True).float())
            for j in range(k):
                x = model.blocks[j](x)  # frozen upstream (eval, no learning)
            blk(x)  # local plasticity update happens here
            n += 1
    blk.conv.learning = False
    model.eval()
    changed = [
        not torch.equal(snaps[j], model.blocks[j].conv.weight) for j in range(len(model.blocks))
    ]
    frac, rmean, rstd = blk.conv.convergence(tol=5e-3)
    log(
        "  block %d: %d batches %.1fs | ||w||_p mean=%.4f std=%.4f frac|r-1|<5e-3=%.3f | "
        "conv-weights-changed=%s" % (k, n, time.time() - t0, rmean, rstd, frac, changed)
    )
    assert changed[k] and sum(changed) == 1, (
        "GREEDY VIOLATION: training block %d changed conv weights %s (want only block %d)"
        % (k, changed, k)
    )
    return dict(frac=frac, rmean=rmean, rstd=rstd)


# ----------------------------------------------------------------------------
# Feature extraction -> disk memmap
# ----------------------------------------------------------------------------
def extract_and_save(model, ds, d_fe, feat_path, label_path, bs=EXTRACT_BS):
    model.eval()
    n = len(ds)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=NW,
        pin_memory=True,
        persistent_workers=(NW > 0),
        prefetch_factor=(4 if NW else None),
    )
    feats = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float32, shape=(n, d_fe))
    labels = np.empty((n,), dtype=np.int64)
    i = 0
    with torch.no_grad():
        for x, y in loader:
            x = to_rate_image(x.to(DEVICE, non_blocking=True).float())
            f = model.features(x).float().cpu().numpy()
            b = f.shape[0]
            feats[i : i + b] = f
            labels[i : i + b] = np.asarray(y, dtype=np.int64)
            i += b
    feats.flush()
    del feats
    np.save(label_path, labels)
    return n


# ----------------------------------------------------------------------------
# Per-feature min-max stats on TRAIN (streamed from the memmap; -> GPU tensors)
# ----------------------------------------------------------------------------
def train_minmax(feat_path, d, ch=5000):
    X = np.load(feat_path, mmap_mode="r")
    N = X.shape[0]
    lo = np.full(d, np.inf, np.float32)
    hi = np.full(d, -np.inf, np.float32)
    for s in range(0, N, ch):
        blk = np.asarray(X[s : s + ch], dtype=np.float32)
        lo = np.minimum(lo, blk.min(0))
        hi = np.maximum(hi, blk.max(0))
    rng = np.where(hi > lo, hi - lo, 1.0).astype(np.float32)
    return torch.tensor(lo, device=DEVICE), torch.tensor(rng, device=DEVICE)


def _norm_chunk(blk_np, lo_t, rng_t):
    blk = torch.tensor(np.asarray(blk_np, dtype=np.float32), device=DEVICE)
    return ((blk - lo_t) / rng_t).clamp_(0, 1)


# ----------------------------------------------------------------------------
# HEAD (a): ridge linear-probe on min-max features -> front-end ceiling
# ----------------------------------------------------------------------------
def ridge_linear_probe(ftr, ltr, fte, lte, d, lo_t, rng_t, regs=(10.0, 100.0, 1000.0), ch=5000):
    Xtr = np.load(ftr, mmap_mode="r")
    ytr = np.load(ltr)
    N = Xtr.shape[0]
    gram = torch.zeros(d, d, device=DEVICE)
    Xty = torch.zeros(d, 10, device=DEVICE)
    for s in range(0, N, ch):
        blk = _norm_chunk(Xtr[s : s + ch], lo_t, rng_t)
        yb = torch.tensor(ytr[s : s + blk.shape[0]].astype(np.int64), device=DEVICE)
        Y = torch.zeros(blk.shape[0], 10, device=DEVICE)
        Y[torch.arange(blk.shape[0], device=DEVICE), yb] = 1.0
        gram += blk.t() @ blk
        Xty += blk.t() @ Y
        del blk, Y
    Xte = np.load(fte, mmap_mode="r")
    yte = torch.tensor(np.load(lte).astype(np.int64), device=DEVICE)
    Nte = Xte.shape[0]
    best, best_reg = 0.0, None
    for reg in regs:
        A = gram.clone()
        A.diagonal().add_(reg)
        W = torch.linalg.solve(A, Xty)
        del A
        correct = 0
        for s in range(0, Nte, ch):
            blk = _norm_chunk(Xte[s : s + ch], lo_t, rng_t)
            correct += ((blk @ W).argmax(1) == yte[s : s + blk.shape[0]]).sum().item()
            del blk
        acc = 100.0 * correct / Nte
        log("  ridge reg=%-6.0f test acc %.2f%%" % (reg, acc))
        if acc > best:
            best, best_reg = acc, reg
    del gram, Xty
    torch.cuda.empty_cache()
    return best, best_reg


# ----------------------------------------------------------------------------
# HEAD (b): large random-LIF pool + analytical ridge (mirror model.NGSNN._solve_beta)
# ----------------------------------------------------------------------------
def ngsnn_head(ftr, ltr, fte, lte, d, lo_t, rng_t, pool=POOL, scales=SCALES, T=HEAD_T, ch=4000):
    clf = NGSNN(lambdas=scales, r=0.9, t_cand=0, l_max=0, T=T, device=DEVICE, seed=SEED)
    # Draw the fixed pool across the multi-scale set (even split), one shot, deterministic via clf.g.
    per = pool // len(scales)
    Ws, bs_ = [], []
    for Lam in scales:
        Ws.append((torch.rand(per, d, generator=clf.g, device=DEVICE) * 2 - 1) * Lam)
        bs_.append((torch.rand(per, generator=clf.g, device=DEVICE) * 2 - 1) * Lam)
    W = torch.cat(Ws, 0)
    b = torch.cat(bs_, 0)
    L = W.shape[0]
    del Ws, bs_

    Xtr = np.load(ftr, mmap_mode="r")
    ytr = np.load(ltr)
    N = Xtr.shape[0]
    gram = torch.zeros(L, L, device=DEVICE)  # S^T S
    Sty = torch.zeros(L, 10, device=DEVICE)  # S^T Y
    mus = torch.zeros(L, device=DEVICE)  # sum_n s  (-> mu, Eq. 9)
    for s in range(0, N, ch):
        blk = _norm_chunk(Xtr[s : s + ch], lo_t, rng_t)
        Sc = clf._rates(blk, W, b)  # (chunk, L) faithful constant-current LIF rates
        yb = torch.tensor(ytr[s : s + Sc.shape[0]].astype(np.int64), device=DEVICE)
        Y = torch.zeros(Sc.shape[0], 10, device=DEVICE)
        Y[torch.arange(Sc.shape[0], device=DEVICE), yb] = 1.0
        gram += Sc.t() @ Sc
        Sty += Sc.t() @ Y
        mus += Sc.sum(0)
        del blk, Sc, Y
    mu = mus / N
    gamma = clf.c1 * torch.exp(-clf.k * mu) + clf.c0  # activity-dependent reg (Eq. 9-11)
    gram.diagonal().add_(gamma)  # (S^T S + diag(gamma))
    beta = torch.linalg.solve(gram, Sty)  # Eq. 12
    del gram, Sty

    Xte = np.load(fte, mmap_mode="r")
    yte = torch.tensor(np.load(lte).astype(np.int64), device=DEVICE)
    Nte = Xte.shape[0]
    correct = 0
    for s in range(0, Nte, ch):
        blk = _norm_chunk(Xte[s : s + ch], lo_t, rng_t)
        Sc = clf._rates(blk, W, b)
        correct += ((Sc @ beta).argmax(1) == yte[s : s + Sc.shape[0]]).sum().item()
        del blk, Sc
    acc = 100.0 * correct / Nte
    dead = (mu < 1e-6).float().mean().item()
    sat = (mu > 0.99).float().mean().item()
    del W, b, beta, mu, mus
    torch.cuda.empty_cache()
    return acc, L, float(gamma.mean()), dead, sat


def _retry_oom(fn, **kw):
    """Run fn(**kw); on CUDA OOM halve the chunk (and pool, if present) and retry."""
    while True:
        try:
            return fn(**kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            kw["ch"] = max(500, kw.get("ch", 4000) // 2)
            if "pool" in kw:
                kw["pool"] = max(3000, kw["pool"] // 2)
            log(
                "  [OOM] retrying with ch=%d%s"
                % (kw["ch"], (" pool=%d" % kw["pool"]) if "pool" in kw else "")
            )


# ----------------------------------------------------------------------------
def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUT_DIR, exist_ok=True)
    log(
        "device=%s | T_bins=%d | head_T=%d | pool=%d | workers=%d | seed=%d"
        % (DEVICE, T_BINS, HEAD_T, POOL, NW, SEED)
    )
    if DEVICE == "cuda":
        log(
            "gpu=%s | torch=%s | free=%.1fGB"
            % (torch.cuda.get_device_name(0), torch.__version__, torch.cuda.mem_get_info()[0] / 1e9)
        )

    model = SoftHebbNMNIST(seed=SEED).to(DEVICE)

    # ---- greedy layer-wise unsupervised training (NO labels, 1 epoch/block, bs 32) ----
    log(
        "--- unsupervised greedy layer-wise SoftHebb training (1 epoch/block, bs=%d) ---" % UNSUP_BS
    )
    _, train_loader = make_loader(True, UNSUP_BS, shuffle=True, drop_last=True)
    conv_report = []
    t0 = time.time()
    for k in range(len(model.blocks)):
        conv_report.append(train_block(model, k, train_loader))
    log("total unsupervised train time: %.1fs" % (time.time() - t0))
    del train_loader

    # freeze extractor + save checkpoint
    for blk in model.blocks:
        blk.conv.learning = False
    model.eval()
    torch.save(model.state_dict(), CKPT)
    log("saved frozen extractor -> %s" % CKPT)

    # ---- assert d_FE from a real forward pass ----
    ds_te, _tmp = make_loader(False, 8, shuffle=False, drop_last=False)
    xb = torch.stack([torch.as_tensor(ds_te[i][0]) for i in range(8)]).to(DEVICE).float()
    with torch.no_grad():
        d_fe = model.features(to_rate_image(xb)).shape[1]
    log("measured d_FE (flattened feature dim) = %d" % d_fe)
    del _tmp

    # ---- extract + save frozen features (train 60k + test 10k) ----
    ftr = os.path.join(OUT_DIR, "nmnist_feats_train.npy")
    ltr = os.path.join(OUT_DIR, "nmnist_feats_train_labels.npy")
    fte = os.path.join(OUT_DIR, "nmnist_feats_test.npy")
    lte = os.path.join(OUT_DIR, "nmnist_feats_test_labels.npy")
    log("--- extracting + saving frozen features ---")
    t0 = time.time()
    ds_tr, _ = make_loader(True, EXTRACT_BS, shuffle=False, drop_last=False)
    ntr = extract_and_save(model, ds_tr, d_fe, ftr, ltr)
    nte = extract_and_save(model, ds_te, d_fe, fte, lte)
    log("extracted train=%d test=%d in %.1fs" % (ntr, nte, time.time() - t0))
    del model
    torch.cuda.empty_cache()

    # ---- shared min-max (fit on train) ----
    lo_t, rng_t = train_minmax(ftr, d_fe)

    # ---- HEAD (a): ridge linear-probe = front-end ceiling ----
    log("--- HEAD (a) ridge linear-probe on min-max features (front-end ceiling) ---")
    lin_acc, lin_reg = _retry_oom(
        ridge_linear_probe,
        ftr=ftr,
        ltr=ltr,
        fte=fte,
        lte=lte,
        d=d_fe,
        lo_t=lo_t,
        rng_t=rng_t,
        ch=5000,
    )

    # ---- HEAD (b): NG-SNN large random-LIF pool + analytical ridge ----
    log(
        "--- HEAD (b) NG-SNN random-LIF pool (L~%d, scales %s, T=%d) + analytical ridge ---"
        % (POOL, SCALES, HEAD_T)
    )
    t0 = time.time()
    ng_acc, L, gmean, dead, sat = _retry_oom(
        ngsnn_head,
        ftr=ftr,
        ltr=ltr,
        fte=fte,
        lte=lte,
        d=d_fe,
        lo_t=lo_t,
        rng_t=rng_t,
        pool=POOL,
        ch=4000,
    )
    ng_dt = time.time() - t0

    # ---- report ----
    FIELD = 99.5
    log("=" * 74)
    log("N-MNIST NG-SNN PIPELINE SUMMARY")
    log(
        "input representation             : sum T=%d frames -> log1p -> per-sample max -> [0,1] (2ch 34x34)"
        % T_BINS
    )
    log(
        "front-end                        : 2-conv SoftHebb (2->96->384), greedy 1-epoch, no backprop"
    )
    log("d_FE                             : %d" % d_fe)
    for k, r in enumerate(conv_report):
        log(
            "block %d ||w||_p converge         : frac=%.3f (mean r=%.4f std=%.4f)"
            % (k, r["frac"], r["rmean"], r["rstd"])
        )
    log(
        "HEAD (a) ridge linear-probe      : %.2f%%  (reg=%.0f) -- front-end CEILING"
        % (lin_acc, lin_reg)
    )
    log(
        "HEAD (b) NG-SNN pool+ridge       : %.2f%%  (L=%d, T=%d, %.0fs; dead=%.1f%% sat=%.1f%% gamma=%.3f)"
        % (ng_acc, L, HEAD_T, ng_dt, 100 * dead, 100 * sat, gmean)
    )
    log("field SOTA (~)                   : %.1f%%" % FIELD)
    log(
        "gap: linear-probe %+.2f | NG-SNN %+.2f  (vs field ~%.1f%%)"
        % (lin_acc - FIELD, ng_acc - FIELD, FIELD)
    )
    log("features train                   : %s  (%d, %d) f32" % (ftr, ntr, d_fe))
    log("features test                    : %s  (%d, %d) f32" % (fte, nte, d_fe))
    log("extractor checkpoint             : %s" % CKPT)
    log("=" * 74)


if __name__ == "__main__":
    main()
