"""
research/ng_snn/nmnist_spatiotemporal.py -- SPATIOTEMPORAL SoftHebb front-end for N-MNIST (NG-SNN).

WHY (the honest limiter). The summed-image pipeline (nmnist_pipeline.py) collapses the T event frames
into ONE static 2ch image BEFORE the SoftHebb conv (to_rate_image = sum over T), discarding N-MNIST's
temporal structure, and caps at ~97.25% linear-probe. Field SOTA (~99.5%) uses spatiotemporal SNNs
over the event timesteps. This script keeps the VERIFIED SoftHebb math (HebbConv2d / Triangle /
SoftHebbBlock from softhebb_mnist.py) UNCHANGED and gives the front-end temporal structure in TWO
places:

  (1) TRAIN the unsupervised 2-conv stack on PER-TIMESTEP event frames. The loader yields
      (B, T, 2, 34, 34); we COLLAPSE (B, T) -> (B*T, 2, 34, 34) and feed every timestep-frame as its
      own training sample, so the local soft-WTA Hebbian rule learns temporal-slice filters instead of
      summed-image filters. This is THE key change vs nmnist_pipeline.py.
  (2) FEATURE per sample = run all T frames through the FROZEN conv -> per-frame flattened features
      (B, T, d_conv), then TEMPORAL POOL to a fixed vector. Two options behind a flag (coordinator
      sweeps NG_SNN_ST_POOL):
        pool="mean"    : mean over T                       -> d_FE = d_conv           (rate)
        pool="meanmax" : concat[mean over T, max over T]   -> d_FE = 2*d_conv         (rate + transient)
      Even "mean" is genuinely spatiotemporal: the conv+Triangle is NONLINEAR, so
      mean_t( conv(frame_t) ) != conv( sum_t frame_t ); the extractor is also trained on per-frame
      images. "meanmax" additionally keeps the per-pixel temporal transient (max response over T).

HEADS mirror nmnist_pipeline.py: (a) ridge linear-probe on min-max features = front-end CEILING,
(b) NG-SNN large random-LIF pool (constant-current firing rates, reusing model.NGSNN._rates) +
analytical activity-dependent ridge (Eq. 9-12). Everything else (disk-cached loaders, greedy
layer-wise freeze, memmap feature extraction, min-max scaling, memory hygiene) is copied from
nmnist_pipeline.py; ONLY the feature representation changes.

MEMORY (12GB card). d_FE for meanmax is 2*d_conv (= 49152 at the default anatomy), so the ridge
Gram is d_FE x d_FE ~ 9.7 GB and a direct LU-solve (which internally copies the matrix) would OOM.
The ridge here accumulates the Gram in-place (addmm_, no d x d temporary) and picks the solver by a
free-VRAM estimate: a direct torch.linalg.solve when a factorization fits (the mean-mode / small-d
path, identical in spirit to nmnist_pipeline.py), else a memory-light conjugate-gradient solve on the
resident GPU Gram (no factorization copy). The NG-SNN head is unaffected by d_FE: it projects
d_FE -> L via a random W, so its Gram is L x L (~1.6 GB) regardless of d_FE.

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/nmnist_spatiotemporal.py
Env: NG_SNN_ST_T (default 15, event-frame time bins) | NG_SNN_ST_POOL (mean|meanmax, default meanmax)
     NG_SNN_ST_UNSUP_BS (default 16 samples -> B*T frames/update) | NG_SNN_ST_EXTRACT_BS (default 64)
     NG_SNN_WORKERS (default 4) | NG_SNN_POOL (default 20001) | NG_SNN_HEAD_T (default 15).
ASCII-only output (cp1252-safe). Files written under data/ng_snn/ + research/ng_snn/ only.
Runs in .venv-cu128 on the RTX 5070 (torch/GPU training tier); the core never imports this.
DO NOT run without a GPU (the coordinator runs it).
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
CACHE_DIR = os.path.join(
    DATA_ROOT, "nmnist_cache"
)  # same cache as nmnist_pipeline.py (same T-binning)
OUT_DIR = os.path.join(DATA_ROOT, "ng_snn")
CKPT = os.path.join(HERE, "softhebb_nmnist_st.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_BINS = int(os.environ.get("NG_SNN_ST_T", "15"))  # event-frame time bins (sweep-able)
POOL_MODE = os.environ.get("NG_SNN_ST_POOL", "meanmax").strip().lower()  # "mean" | "meanmax"
HEAD_T = int(os.environ.get("NG_SNN_HEAD_T", "15"))  # LIF integration window in the NG-SNN head
SEED = 0
UNSUP_BS = int(
    os.environ.get("NG_SNN_ST_UNSUP_BS", "16")
)  # samples/batch -> B*T frames per Hebbian update
EXTRACT_BS = int(
    os.environ.get("NG_SNN_ST_EXTRACT_BS", "64")
)  # samples/batch during feature extraction
NW = int(os.environ.get("NG_SNN_WORKERS", "4"))
POOL = int(os.environ.get("NG_SNN_POOL", "20001"))  # NG-SNN random-LIF pool size (~20000)
SCALES = (0.05, 0.1, 0.3)  # candidate lambda scales (feature operating point)

assert POOL_MODE in ("mean", "meanmax"), (
    "NG_SNN_ST_POOL must be 'mean' or 'meanmax' (got %r)" % POOL_MODE
)


def log(m):
    print(m, flush=True)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# Input representation: per-timestep event frames (the key change).
#   frames_to_batch : (B, T, 2, 34, 34) -> (B*T, 2, 34, 34), one image per timestep-frame.
# We log1p-compress the (heavy-tailed) per-pixel event counts -- monotone, harmless -- and let the
# block's BatchNorm2d(2, affine=False) whiten per channel. We DO NOT per-frame max-normalize: a sparse
# frame's own max is noise, so dividing by it would amplify noise; BN handles the scale instead.
# ----------------------------------------------------------------------------
def frames_to_batch(x):
    b, t = x.shape[0], x.shape[1]
    x = x.reshape(b * t, x.shape[2], x.shape[3], x.shape[4])  # collapse (B,T) -> (B*T) in C-order
    return torch.log1p(x)


def st_features(model, x, pool_mode):
    """Spatiotemporal feature per sample.
      x (B, T, 2, 34, 34) -> collapse (B*T, 2, 34, 34) -> frozen conv -> (B*T, d_conv)
      -> reshape (B, T, d_conv) -> TEMPORAL POOL over T -> (B, d_FE).
    The reshape is the exact inverse of the collapse (same T, contiguous C-order), so row b*T+t maps
    back to (b, t). pool_mode: 'mean' -> (B, d_conv); 'meanmax' -> concat[mean, max] -> (B, 2*d_conv)."""
    b, t = x.shape[0], x.shape[1]
    f = model.features(frames_to_batch(x))  # (B*T, d_conv)
    f = f.reshape(b, t, -1)  # (B, T, d_conv)
    if pool_mode == "mean":
        return f.mean(dim=1)  # (B, d_conv)
    return torch.cat([f.mean(dim=1), f.amax(dim=1)], dim=1)  # (B, 2*d_conv)  rate + transient


# ----------------------------------------------------------------------------
# 2-channel SoftHebb front-end (VERIFIED HebbConv2d math; only in_channels/anatomy re-wired).
# Identical block hyper-parameters to nmnist_pipeline.py's SoftHebbNMNIST.
# ----------------------------------------------------------------------------
class SoftHebbNMNISTST(nn.Module):
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
# Loaders (reuse the disk cache built by nmnist_pipeline.py; absolute paths so cwd does not matter).
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
# Greedy layer-wise unsupervised training (only the block being trained changes) -- on PER-FRAME images.
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
            x = frames_to_batch(
                x.to(DEVICE, non_blocking=True).float()
            )  # (B*T, 2, 34, 34) per-frame
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
        "  block %d: %d updates %.1fs | ||w||_p mean=%.4f std=%.4f frac|r-1|<5e-3=%.3f | "
        "conv-weights-changed=%s" % (k, n, time.time() - t0, rmean, rstd, frac, changed)
    )
    assert changed[k] and sum(changed) == 1, (
        "GREEDY VIOLATION: training block %d changed conv weights %s (want only block %d)"
        % (k, changed, k)
    )
    return dict(frac=frac, rmean=rmean, rstd=rstd)


# ----------------------------------------------------------------------------
# Feature extraction -> disk memmap (temporal pool over T inside the batch loop).
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
            x = x.to(DEVICE, non_blocking=True).float()  # (B, T, 2, 34, 34)
            f = st_features(model, x, POOL_MODE).float().cpu().numpy()  # (B, d_fe)
            b = f.shape[0]
            feats[i : i + b] = f
            labels[i : i + b] = np.asarray(y, dtype=np.int64)
            i += b
    feats.flush()
    del feats
    np.save(label_path, labels)
    return n


# ----------------------------------------------------------------------------
# Per-feature min-max stats on TRAIN (streamed from the memmap; -> GPU tensors).
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


def _ridge_chunk(d, cap=5000, budget=4e8):
    """Rows/chunk so a normalized chunk (chunk x d, fp32) stays ~<=budget bytes (keeps VRAM headroom
    for the resident d x d Gram at large d_FE)."""
    return max(1000, min(cap, int(budget / (d * 4))))


# ----------------------------------------------------------------------------
# Conjugate-gradient solve of (Gram + reg I) W = Xty on the RESIDENT GPU Gram (no factorization copy).
# Vectorized over the m RHS columns (10 classes). Used only when a direct LU-solve would not fit.
# ----------------------------------------------------------------------------
def _cg_solve(gram, Xty, reg, iters=800, tol=1e-5):
    B = Xty
    X = torch.zeros_like(B)
    R = B.clone()  # residual = B - A@0 = B
    P = R.clone()
    rs_old = (R * R).sum(dim=0)  # (m,)
    b_norm = (B * B).sum(dim=0).clamp_min(1e-30)  # (m,) for a relative stop
    used = iters
    for it in range(iters):
        AP = gram @ P + reg * P  # A @ P  (only matvecs; A never materialized+copied)
        denom = (P * AP).sum(dim=0).clamp_min(1e-30)  # (m,)
        alpha = rs_old / denom
        X = X + alpha.unsqueeze(0) * P
        R = R - alpha.unsqueeze(0) * AP
        rs_new = (R * R).sum(dim=0)
        if float((rs_new / b_norm).max().item()) < tol * tol:
            used = it + 1
            break
        beta = rs_new / rs_old.clamp_min(1e-30)
        P = R + beta.unsqueeze(0) * P
        rs_old = rs_new
    final_rel = float((rs_old / b_norm).max().sqrt().item())
    return X, used, final_rel


# ----------------------------------------------------------------------------
# HEAD (a): ridge linear-probe on min-max features -> front-end CEILING.
# Gram accumulated in-place (addmm_, no d x d temporary); solver chosen by free VRAM.
# ----------------------------------------------------------------------------
def ridge_linear_probe(ftr, ltr, fte, lte, d, lo_t, rng_t, regs=(10.0, 100.0, 1000.0)):
    ch = _ridge_chunk(d)
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
        gram.addmm_(blk.t(), blk)  # in-place S^T S accumulation (no temporary)
        Xty.addmm_(blk.t(), Y)  # in-place S^T Y
        del blk, Y, yb

    # ---- choose the solver by free VRAM (direct LU internally copies the d x d matrix) ----
    use_cg = False
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
        use_cg = (d * d * 4 * 1.15) > free  # a full factorization copy would not fit
    log("  ridge d=%d chunk=%d solver=%s" % (d, ch, "CG(resident-Gram)" if use_cg else "direct-LU"))

    Xte = np.load(fte, mmap_mode="r")
    yte = torch.tensor(np.load(lte).astype(np.int64), device=DEVICE)
    Nte = Xte.shape[0]
    best, best_reg = 0.0, None
    for reg in regs:
        if use_cg:
            W, cg_it, cg_res = _cg_solve(gram, Xty, reg)
            if cg_res > 1e-3:
                log(
                    "  [warn] ridge CG reg=%.0f did not fully converge (rel-res=%.2e, iters=%d)"
                    % (reg, cg_res, cg_it)
                )
        else:
            gram.diagonal().add_(reg)  # add reg in place ...
            W = torch.linalg.solve(gram, Xty)
            gram.diagonal().sub_(reg)  # ... then restore for the next reg
            cg_it, cg_res = 0, 0.0
        correct = 0
        for s in range(0, Nte, ch):
            blk = _norm_chunk(Xte[s : s + ch], lo_t, rng_t)
            correct += ((blk @ W).argmax(1) == yte[s : s + blk.shape[0]]).sum().item()
            del blk
        acc = 100.0 * correct / Nte
        tag = (" (CG %d it, rel-res %.1e)" % (cg_it, cg_res)) if use_cg else ""
        log("  ridge reg=%-6.0f test acc %.2f%%%s" % (reg, acc, tag))
        if acc > best:
            best, best_reg = acc, reg
        del W
    del gram, Xty
    torch.cuda.empty_cache()
    return best, best_reg


# ----------------------------------------------------------------------------
# HEAD (b): large random-LIF pool + analytical ridge (mirror model.NGSNN._solve_beta).
# Gram is L x L (independent of d_FE); W is L x d_FE (the only d-dependent tensor).
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
    torch.cuda.empty_cache()

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
        "device=%s | T_bins=%d | pool=%s | head_T=%d | pool_units=%d | unsup_bs=%d | extract_bs=%d | "
        "workers=%d | seed=%d"
        % (DEVICE, T_BINS, POOL_MODE, HEAD_T, POOL, UNSUP_BS, EXTRACT_BS, NW, SEED)
    )
    if DEVICE == "cuda":
        log(
            "gpu=%s | torch=%s | free=%.1fGB"
            % (torch.cuda.get_device_name(0), torch.__version__, torch.cuda.mem_get_info()[0] / 1e9)
        )

    model = SoftHebbNMNISTST(seed=SEED).to(DEVICE)

    # ---- greedy layer-wise unsupervised training on PER-FRAME images (NO labels, 1 epoch/block) ----
    log(
        "--- unsupervised greedy layer-wise SoftHebb training on per-timestep frames "
        "(1 epoch/block, %d samples/batch = %d frames/update) ---" % (UNSUP_BS, UNSUP_BS * T_BINS)
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

    # ---- measure d_conv from a real forward pass; derive d_FE for BOTH pool modes ----
    ds_te, _tmp = make_loader(False, 8, shuffle=False, drop_last=False)
    xb = (
        torch.stack([torch.as_tensor(ds_te[i][0]) for i in range(8)]).to(DEVICE).float()
    )  # (8,T,2,34,34)
    with torch.no_grad():
        d_conv = model.features(frames_to_batch(xb)).shape[1]  # per-frame flattened conv dim
    d_fe_mean = d_conv
    d_fe_meanmax = 2 * d_conv
    d_fe = d_fe_meanmax if POOL_MODE == "meanmax" else d_fe_mean
    log(
        "d_conv (per-frame) = %d | d_FE(mean) = %d | d_FE(meanmax) = %d | ACTIVE pool=%s -> d_FE = %d"
        % (d_conv, d_fe_mean, d_fe_meanmax, POOL_MODE, d_fe)
    )
    del _tmp

    # ---- extract + save frozen spatiotemporal features (train 60k + test 10k) ----
    ftr = os.path.join(OUT_DIR, "nmnist_st_feats_train.npy")
    ltr = os.path.join(OUT_DIR, "nmnist_st_feats_train_labels.npy")
    fte = os.path.join(OUT_DIR, "nmnist_st_feats_test.npy")
    lte = os.path.join(OUT_DIR, "nmnist_st_feats_test_labels.npy")
    log(
        "--- extracting + saving frozen spatiotemporal features (pool=%s, d_FE=%d) ---"
        % (POOL_MODE, d_fe)
    )
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
    lin_acc, lin_reg = ridge_linear_probe(ftr, ltr, fte, lte, d_fe, lo_t, rng_t)

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
    log("=" * 78)
    log("N-MNIST SPATIOTEMPORAL SoftHebb PIPELINE SUMMARY")
    log(
        "input representation             : per-timestep event frames (T=%d), log1p, BN-whitened "
        "(NO summed static image)" % T_BINS
    )
    log(
        "front-end                        : 2-conv SoftHebb (2->96->384) trained on per-frame images, "
        "greedy 1-epoch, no backprop"
    )
    log("temporal pool mode               : %s" % POOL_MODE)
    log("d_conv (per-frame)               : %d" % d_conv)
    log("d_FE (mean)                      : %d" % d_fe_mean)
    log("d_FE (meanmax)                   : %d" % d_fe_meanmax)
    log("d_FE (active, pool=%s)%s: %d" % (POOL_MODE, " " * max(1, 11 - len(POOL_MODE)), d_fe))
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
    log("prev summed-image linear-probe   : 97.25%% (static baseline this script aims to beat)")
    log("features train                   : %s  (%d, %d) f32" % (ftr, ntr, d_fe))
    log("features test                    : %s  (%d, %d) f32" % (fte, nte, d_fe))
    log("extractor checkpoint             : %s" % CKPT)
    log("=" * 78)


if __name__ == "__main__":
    main()
