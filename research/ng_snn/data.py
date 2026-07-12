"""
research/ng_snn/data.py -- disk-streaming N-MNIST loader for the NG-SNN reproduction (torch/GPU tier).

Streams the on-disk N-MNIST event dataset (data/NMNIST/, 60k train / 10k test, 34x34x2, events
x/y/t/p) as binned spike-frame tensors [B, T, 2, 34, 34]. A DiskCachedDataset computes the
(expensive) event->frame binning ONCE and every later epoch STREAMS the compact cache from disk --
bounded RAM, high throughput -- which is the "disk streaming for performance" this pipeline needs.

The number of time bins T is the ONE method-dependent knob: SET IT FROM THE NG-SNN PAPER. Everything
else here (loading, disk caching, streaming, pinned GPU transfer, worker prefetch) is
method-independent and stays as the paper fixes the model.

NOT zero-dep -- torch/GPU TRAINING TIER only (needs tonic + torch). Runs in .venv-cu128 on the
RTX 5070; the zero-dep core NEVER imports this. Windows note: num_workers>0 spawns processes, so the
frame->tensor transform is a module-level CLASS (picklable, not a lambda) and the smoke is guarded by
__main__.

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/data.py        # disk-streaming smoke
"""
import time

import numpy as np
import tonic
import tonic.transforms as TT
import torch
from torch.utils.data import DataLoader

SENSOR = tonic.datasets.NMNIST.sensor_size   # (34, 34, 2) -> frames are (T, 2, 34, 34)
DEFAULT_T = 15                               # PLACEHOLDER time-bin count -- SET FROM THE PAPER (N-MNIST common T ~ 10-25)


class ToFloatTensor:
    """Picklable frame->tensor transform (module-level class, so Windows num_workers>0 can spawn it;
    a lambda would not pickle). tonic ToFrame emits a (T, 2, H, W) numpy stack -> float32 tensor."""

    def __call__(self, frames):
        return torch.from_numpy(np.asarray(frames, dtype=np.float32))


def _make_dataset(train, save_to, n_time_bins, cache_dir):
    """N-MNIST with event->T-frame binning, wrapped in a DiskCachedDataset (bin once, then stream the
    compact per-sample cache from disk on every later epoch). Separate cache per split + per T."""
    to_frame = TT.ToFrame(sensor_size=SENSOR, n_time_bins=n_time_bins)
    ds = tonic.datasets.NMNIST(save_to=save_to, train=train,
                               transform=TT.Compose([to_frame, ToFloatTensor()]))
    split = "train" if train else "test"
    return tonic.DiskCachedDataset(ds, cache_path=f"{cache_dir}/{split}_T{n_time_bins}")


def make_nmnist_loaders(save_to="data", n_time_bins=DEFAULT_T, batch_size=128,
                        num_workers=4, cache_dir="data/nmnist_cache"):
    """(train_loader, test_loader) that STREAM [B, T, 2, 34, 34] float frames + int labels from disk.
    Disk-cached (bin once, stream after) + pinned memory + worker prefetch for GPU throughput. Set
    n_time_bins from the paper; batch_size / num_workers tune to the RTX 5070."""
    tr = _make_dataset(True, save_to, n_time_bins, cache_dir)
    te = _make_dataset(False, save_to, n_time_bins, cache_dir)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=(num_workers > 0),
                  prefetch_factor=(4 if num_workers else None))
    return (DataLoader(tr, shuffle=True, drop_last=True, **common),
            DataLoader(te, shuffle=False, **common))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bs = 128
    tr, te = make_nmnist_loaders(n_time_bins=DEFAULT_T, batch_size=bs, num_workers=4)
    print(f"device={dev}  T={DEFAULT_T}  sensor={SENSOR}  train_batches={len(tr)}  test_batches={len(te)}")

    # first pass caches transformed frames to disk; time a handful of STREAMED batches -> GPU.
    t0 = time.perf_counter()
    n = 0
    for x, y in tr:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        n += 1
        if n == 1:
            print(f"batch shape {tuple(x.shape)}  labels {tuple(y.shape)}  dtype {x.dtype}  "
                  f"nonzero-frac {float((x > 0).float().mean()):.4f}")
        if n >= 20:
            break
    dt = time.perf_counter() - t0
    print(f"streamed {n} batches ({n * bs} samples) in {dt:.2f}s -> {n * bs / max(dt, 1e-9):.0f} samples/s on {dev}")
    print("data OK: N-MNIST disk-streaming loader ready (set T from the paper, then build the model)")


if __name__ == "__main__":
    main()
