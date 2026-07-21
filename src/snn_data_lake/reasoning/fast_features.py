"""
Batched conv1 feature extractor -- a drop-in, numerically-equivalent, far faster
replacement for the per-image `torch.stack([conv.features(img) for img in X])`
loop that dominates the SNDL developmental-loop runtime.  (torch/GPU tier,
ruff-excluded; this file is ADDITIVE and imports the
frozen `ConvLatencyStdp` from research/deep_stdp_latency.py, editing nothing.)

WHY THIS FILE EXISTS
--------------------
Every developmental loop caches layer-1 features by calling `conv.features(img)`
once per image over 10k-30k images. Each call runs a FULL T-step conv-IF forward
with a SINGLE-image `F.conv2d` at every time step -- so the loop launches
N * (1 DoG conv + T volley convs) tiny GPU kernels, each on a batch of ONE. GPUs
are starved by that. Because `ConvLatencyStdp.features` is FROZEN inference (no
STDP write, no learning, no cross-image state), the whole thing vectorises over
the batch dimension: the same convolutions, run on [B,...] tensors, produce the
same numbers with a fraction of the kernel-launch overhead.

WHAT IS BATCHED (byte-for-byte the same math, one batch axis added)
-------------------------------------------------------------------
`_dog_latency` (per image [2,28,28])  -> `_dog_latency_batch` ([B,2,28,28]):
  one `F.conv2d` of the whole [B,1,28,28] image batch through the DoG kernel;
  ON=relu(resp), OFF=relu(-resp); per-image vmax via amax over (C,H,W) with
  keepdim so the threshold / normalisation broadcast per image; identical
  round((1-norm)*(t_lat-1)) latency spike-times; never-fired stays t_lat.

`_forward` (per image, T single-image convs) -> `_forward_batch` ([B,...]):
  cumulative-volley integrate over the SAME T steps, but each step is ONE
  `F.conv2d` on the [B,2,28,28] cumulative volley -> [B,C,hout,hout] potentials.
  FIRST-SPIKE bookkeeping across the batch: `fired` is [B,C,hout,hout] init
  t_lat; each step `newly = (pot >= cthresh) & (fired == t_lat)` is a per-element
  AND over the whole batch (a neuron in image b fires the first step its own
  potential crosses threshold AND it has not fired yet), then
  `fired = where(newly, t, fired)` stamps t only into those still-unfired
  neurons. Non-leaky IF => cumulative potential is monotonic, so the first
  crossing is well-defined and independent per (b, map, y, x) -- exactly the
  per-image semantics, just with a leading batch axis.

`features` tail: val = (t_lat - fired)/t_lat where fired<t_lat else 0;
  `F.max_pool2d` over [B,C,hout,hout]; flatten to [B, feat_dim]; per-image /max
  via amax(dim=1, keepdim=True).clamp_min(1e-6). Row r == the old per-image feat.

Result: `batched_features(conv, X)` returns a [N, feat_dim] tensor identical (to
within GPU conv float noise, < 1e-4) to `torch.stack([conv.features(x) for x in X])`.

Run the correctness + speed self-check:
    NORD_GPU=1 PYTHONPATH=src ../.venv-cu128/Scripts/python research/fast_features.py
Knobs: NORD_C NORD_T NORD_TRAIN NORD_CORRECT NORD_SPEED NORD_BATCH NORD_GPU
Deps: numpy torch torchvision snntorch   (research/GPU tier)
"""
import os
import time

import torch
import torch.nn.functional as F

from snn_data_lake.reasoning.torch_device import resolve_device, enable
from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp, DOG_SIZE, DOG_THRESH


def _env(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


def _sync(device):
    """Make GPU timing honest -- wait for queued kernels before reading the clock."""
    if torch.is_tensor(device):
        device = device
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _as_batch(xb, device):
    """Coerce a slice of X (tensor row-slice or list of images) to [B,1,28,28]."""
    if not torch.is_tensor(xb):
        xb = torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x) for x in xb])
    b = xb.shape[0]
    return xb.to(device).reshape(b, 1, 28, 28).float()


def _dog_latency_batch(conv, imgs):
    """Vectorised ConvLatencyStdp._dog_latency over a batch.

    imgs: [B,1,28,28] on conv.device -> spike-times [B,2,28,28] (long), value in
    [0, t_lat-1] where a neuron fires (stronger contrast -> earlier), t_lat where
    it never fires. Byte-identical to stacking the per-image results.
    """
    t_lat = conv.t_lat
    resp = F.conv2d(imgs, conv.dog, padding=DOG_SIZE // 2)[:, 0]        # [B,28,28]
    on = F.relu(resp)
    off = F.relu(-resp)
    combined = torch.stack((on, off), dim=1)                           # [B,2,28,28]
    vmax = combined.amax(dim=(1, 2, 3), keepdim=True)                  # [B,1,1,1]
    thr = DOG_THRESH * vmax
    active = combined > thr                                            # per-image sparse mask
    # divide by vmax exactly where vmax>0 (matches per-image `if vmax>0`); where
    # vmax==0 the mask is all-False so the chosen denom is irrelevant -> use 1.
    denom = torch.where(vmax > 0, vmax, torch.ones_like(vmax))
    norm = (combined / denom).clamp(0.0, 1.0)
    times = torch.round((1.0 - norm) * (t_lat - 1)).long()
    never = torch.full_like(times, t_lat)
    return torch.where(active, times, never)                          # [B,2,28,28] long


def _forward_batch(conv, spike_time):
    """Vectorised ConvLatencyStdp._forward over a batch.

    spike_time: [B,2,28,28] long -> fired [B,C,hout,hout] long (t_lat == never).
    The T-step loop stays (T is small), but each step is ONE conv2d over the whole
    [B,2,28,28] cumulative volley instead of B separate single-image conv2ds.
    """
    B = spike_time.shape[0]
    t_lat = conv.t_lat
    device = conv.device
    cum = torch.zeros(B, 2, 28, 28, device=device)
    fired = torch.full((B, conv.C, conv.hout, conv.hout), t_lat,
                       dtype=torch.long, device=device)
    for t in range(t_lat):
        vol = (spike_time == t).float()                               # [B,2,28,28]
        cum = cum + vol
        pot = F.conv2d(cum, conv.Wc)                                  # [B,C,hout,hout]
        newly = (pot >= conv.cthresh) & (fired == t_lat)             # first crossing, per neuron
        fired = torch.where(newly, torch.full_like(fired, t), fired)
    return fired


def batched_features(conv, X, batch=256):
    """Drop-in replacement for `torch.stack([conv.features(x) for x in X])`.

    Returns a [N, conv.feat_dim()] tensor on conv.device: first-spike-latency map
    -> spatial max-pool -> flatten -> per-image rescale to [0,1], identical to the
    per-image loop but computed one image-batch at a time through batched conv2d.
    """
    device = conv.device
    t_lat = conv.t_lat
    N = len(X)
    out = []
    with torch.no_grad():
        for i in range(0, N, batch):
            xb = _as_batch(X[i:i + batch], device)
            st = _dog_latency_batch(conv, xb)                         # [B,2,28,28]
            fired = _forward_batch(conv, st)                          # [B,C,hout,hout]
            val = torch.where(fired < t_lat,
                              (t_lat - fired).float() / t_lat,
                              torch.zeros_like(fired, dtype=torch.float32))
            pooled = F.max_pool2d(val, conv.pool)                     # [B,C,ph,ph]
            feat = pooled.reshape(val.shape[0], -1)                   # [B, feat_dim]
            feat = feat / feat.amax(dim=1, keepdim=True).clamp_min(1e-6)
            out.append(feat)
    return torch.cat(out, dim=0)


def batched_features_and_keys(conv, X, batch=256):
    """One batched forward -> BOTH the spatial feature AND the compact global KEY.

    Replaces the SNDL lake loop's redundant DOUBLE conv-forward: `batched_features`
    (spatial [N,feat_dim]) and `torch.stack([_global_key(conv, img) for img in X])`
    (key [N,C1]) each ran `_dog_latency`+`_forward` over the same images, so the
    second forward was pure waste -- both outputs are functions of the SAME first-
    spike map `fired`. Here `fired` is computed ONCE per image-batch and both come
    off the ONE latency map `val`:

      val    = where(fired < t_lat, (t_lat - fired)/t_lat, 0)   # [B,C,hout,hout]
      spatial = (max_pool2d(val, conv.pool) -> flatten -> /per-image-max)
                -- byte-identical tail to batched_features(conv, X)  -> [N, feat_dim]
      keys    = val.amax(dim=(2, 3))                              # per-map global-max
                -- byte-identical to _global_key's val.amax(dim=(1,2)) per image
                                                                   -> [N, C1]

    Returns (spatial [N, conv.feat_dim()], keys [N, conv.C]), both on conv.device.
    """
    device = conv.device
    t_lat = conv.t_lat
    N = len(X)
    spat = []
    keys = []
    with torch.no_grad():
        for i in range(0, N, batch):
            xb = _as_batch(X[i:i + batch], device)
            st = _dog_latency_batch(conv, xb)                         # [B,2,28,28]
            fired = _forward_batch(conv, st)                          # [B,C,hout,hout]
            val = torch.where(fired < t_lat,
                              (t_lat - fired).float() / t_lat,
                              torch.zeros_like(fired, dtype=torch.float32))
            pooled = F.max_pool2d(val, conv.pool)                     # [B,C,ph,ph]
            feat = pooled.reshape(val.shape[0], -1)                   # [B, feat_dim]
            feat = feat / feat.amax(dim=1, keepdim=True).clamp_min(1e-6)
            spat.append(feat)
            keys.append(val.amax(dim=(2, 3)))                         # [B, C] global key
    return torch.cat(spat, dim=0), torch.cat(keys, dim=0)


def main():
    device = resolve_device()
    enable(device)
    torch.manual_seed(0)

    C = _env("NORD_C", 30)
    T_LAT = _env("NORD_T", 15)
    TRAIN_N = _env("NORD_TRAIN", 2000)
    CORRECT_N = _env("NORD_CORRECT", 300)
    SPEED_N = _env("NORD_SPEED", 2000)
    BATCH = _env("NORD_BATCH", 256)
    EPOCHS = _env("NORD_L1_EPOCHS", 2)

    print("=" * 66)
    print("BATCHED conv1 FEATURE EXTRACTOR  (correctness + speedup self-check)")
    print("=" * 66)
    print(f"device={device}  C={C}  T={T_LAT}  train={TRAIN_N}  correct-N={CORRECT_N}  "
          f"speed-N={SPEED_N}  batch={BATCH}")
    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, max(CORRECT_N, SPEED_N))

    print(f"training a small ConvLatencyStdp layer 1 ({EPOCHS} epochs, {TRAIN_N} imgs)...")
    conv = ConvLatencyStdp(C, T_LAT)
    for ep in range(EPOCHS):
        for img in Xtr:
            conv.train_image(img)
        print(f"  epoch {ep + 1}/{EPOCHS} kernel diversity (cos-sim) = "
              f"{conv.kernel_diversity():.3f}")
    print(f"layer-1 frozen; feat_dim = {conv.feat_dim()}")

    # -------- CORRECTNESS (the non-negotiable gate) --------
    Xc = Xte[:CORRECT_N]
    ref = torch.stack([conv.features(x) for x in Xc])                 # per-image loop
    fast = batched_features(conv, Xc, batch=BATCH)                    # batched
    assert fast.shape == ref.shape, f"shape mismatch: {fast.shape} vs {ref.shape}"
    max_diff = float((fast - ref).abs().max().item())
    print("\n" + "-" * 66)
    print(f"CORRECTNESS  ({CORRECT_N} images, shape {tuple(ref.shape)})")
    print(f"  max abs diff (batched vs per-image loop) = {max_diff:.3e}")
    assert max_diff < 1e-4, (
        f"FEATURES DIFFER: max abs diff {max_diff:.3e} >= 1e-4 -- the batched "
        "extractor is NOT numerically equivalent; the optimization is invalid.")
    print("  PASS: batched features match the per-image loop (< 1e-4).")

    # -------- ONE-PASS spatial+keys (byte-equivalence gate) --------
    def _ref_global_key(c, img):
        """Reference per-image key -- exact copy of developmental_loop_lake._global_key."""
        st = c._dog_latency(img)
        fired, _ = c._forward(st)
        val = torch.where(fired < c.t_lat, (c.t_lat - fired).float() / c.t_lat,
                          torch.zeros_like(fired, dtype=torch.float32))
        return val.amax(dim=(1, 2))                                   # [C1]

    spatial_1p, keys_1p = batched_features_and_keys(conv, Xc, batch=BATCH)
    key_ref = torch.stack([_ref_global_key(conv, x) for x in Xc])     # [N, C1] per-image loop
    assert spatial_1p.shape == ref.shape, f"spatial shape mismatch: {spatial_1p.shape} vs {ref.shape}"
    assert keys_1p.shape == key_ref.shape, f"keys shape mismatch: {keys_1p.shape} vs {key_ref.shape}"
    spatial_diff = float((spatial_1p - ref).abs().max().item())
    key_diff = float((keys_1p - key_ref).abs().max().item())
    print("\n" + "-" * 66)
    print(f"ONE-PASS batched_features_and_keys  ({CORRECT_N} images)")
    print(f"  spatial vs batched_features    max abs diff = {spatial_diff:.3e}")
    print(f"  keys    vs _global_key stack   max abs diff = {key_diff:.3e}")
    assert spatial_diff < 1e-4, (
        f"ONE-PASS SPATIAL DIFFERS: {spatial_diff:.3e} >= 1e-4 -- not equivalent to "
        "batched_features; the single-pass refactor is invalid.")
    assert key_diff < 1e-4, (
        f"ONE-PASS KEYS DIFFER: {key_diff:.3e} >= 1e-4 -- not equivalent to the "
        "per-image _global_key stack; the single-pass refactor is invalid.")
    print("  PASS: one-pass spatial AND keys both match their two-pass references (< 1e-4).")

    # -------- SPEEDUP (same images, wall-clock) --------
    Xs = Xte[:SPEED_N]
    _sync(device)
    t0 = time.perf_counter()
    ref_speed = torch.stack([conv.features(x) for x in Xs])
    _sync(device)
    t_loop = time.perf_counter() - t0

    _sync(device)
    t0 = time.perf_counter()
    fast_speed = batched_features(conv, Xs, batch=BATCH)
    _sync(device)
    t_batch = time.perf_counter() - t0

    speed_diff = float((fast_speed - ref_speed).abs().max().item())
    speedup = t_loop / max(t_batch, 1e-9)
    print("-" * 66)
    print(f"SPEEDUP  ({SPEED_N} images)")
    print(f"  per-image loop     : {t_loop:8.3f} s")
    print(f"  batched_features   : {t_batch:8.3f} s")
    print(f"  speedup            : {speedup:8.2f}x  (batched vs per-image loop)")
    print(f"  max abs diff       : {speed_diff:.3e}")
    print("=" * 66)

    assert speed_diff < 1e-4, f"speed-run features diverged: {speed_diff:.3e}"
    assert t_batch < t_loop, (
        f"batched not faster: batched {t_batch:.3f}s vs loop {t_loop:.3f}s")
    print(f"self-check OK: features MATCH (max diff {max_diff:.3e} < 1e-4) AND "
          f"batched is {speedup:.1f}x FASTER ({t_batch:.3f}s vs {t_loop:.3f}s).")


if __name__ == "__main__":
    main()
