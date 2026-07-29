"""Reproduce the NG-SNN capacity numbers in docs/PERFORMANCE.md.

A table nobody can re-derive is a claim, not a measurement -- and this repo has already been caught
once citing an instrument that was gitignored. Every number in the "Neural capacity scales with real
data" section comes from here, so it can be checked rather than believed.

Needs the torch tier and the [hf] extra (streaming is live):

  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python tools/capacity_curve.py control
  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python tools/capacity_curve.py grid
  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python tools/capacity_curve.py saturation

`control` is the one that matters most and is the only offline mode: it is the check that the
earn-your-place stop can tell signal from noise at all. Before the held-out split was made real, it
grew to the hard bound on RANDOM labels -- so this is the regression test for that bug.
"""

import sys
import time

import torch

from snn_data_lake.hippocampus.ng_snn import NGSNN

REPO = "ylecun/mnist"


def _load(split, n):
    from snn_data_lake.substrate.hf_stream import stream_features

    X, y, _ = stream_features(REPO, split=split, n_rows=n)
    return torch.tensor(X), torch.tensor(y)


def control():
    """Random labels vs learnable data of identical shape. The gap IS the criterion working."""
    n, d, m = 600, 64, 10
    g = torch.Generator().manual_seed(7)
    y = torch.randint(0, m, (n,), generator=g)

    noise = NGSNN()
    noise.fit(torch.rand(n, d, generator=g), y, delta=8)

    proto = torch.rand(m, d, generator=g)
    X = (proto[y] + 0.15 * torch.randn(n, d, generator=g)).clamp(0, 1)
    real = NGSNN()
    real.fit(X, y, delta=8)

    ratio = real.n_units / max(noise.n_units, 1)
    print(f"random labels L={noise.n_units:4d}   learnable L={real.n_units:4d}   gap {ratio:.1f}x")
    assert noise.n_units < n // 4, (
        f"grew {noise.n_units} units to fit {n} RANDOM labels -- the held-out stop is not "
        f"out-of-sample (this is exactly the v4.80 bug)"
    )
    assert ratio > 3.0, f"signal/noise gap {ratio:.1f}x is too small to call the stop selective"
    print("OK: capacity is spent on structure, not on memorising noise")


def grid():
    """The patience column of the PERFORMANCE.md table, scored on the real test split."""
    Xte, yte = _load("test", 2000)
    floor = yte.bincount().max().item() / len(yte)
    print(f"test split {len(yte)} unseen rows, majority-class floor {floor:.3f}")
    print(f"{'rows':>7} {'setting':>15} {'L':>7} {'test acc':>9}")
    for n in (2000, 10000):
        Xtr, ytr = _load("train", n)
        for name, kw in (
            ("patience=16", {"patience": 16}),
            ("patience=48", {}),  # the default
            ("fixed cap 200", {"l_max": 200}),
        ):
            net = NGSNN(**kw)
            net.fit(Xtr, ytr, delta=8)
            acc = (net.predict(Xte).cpu() == yte).float().mean().item()
            print(f"{n:>7} {name:>15} {net.n_units:>7} {acc:>9.3f}")


def saturation():
    """Does grown width reach the N~4,096 CUDA crossover? (P5 rests on the answer.)"""
    Xte, yte = _load("test", 2000)
    print(f"{'rows':>7} {'grown L':>8} {'test acc':>9} {'fit s':>8}")
    for n in (10000, 16000, 24000):
        Xtr, ytr = _load("train", n)
        net = NGSNN()
        t0 = time.perf_counter()
        net.fit(Xtr, ytr, delta=8)
        dt = time.perf_counter() - t0
        acc = (net.predict(Xte).cpu() == yte).float().mean().item()
        print(f"{n:>7} {net.n_units:>8} {acc:>9.3f} {dt:>8.0f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "control"
    {"control": control, "grid": grid, "saturation": saturation}[mode]()
