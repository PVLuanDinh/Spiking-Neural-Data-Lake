"""
Spike pooling — custom max-pooling for spike feature maps (pure Python, no tensors).

Downsampling spike maps with AVERAGE pooling acts as a low-pass filter: a sparse, high-frequency spike (one
hot cell in a quiet window) gets diluted by 1/k (1/k^2 in 2-D) and washes out. MAX pooling is the right
operator for spikes -- it keeps the strongest event in each window, so a rare but salient spike SURVIVES the
downsample at full amplitude. This is the anti-low-pass counterpart to the torch Max-Former pooling in
research/neuromorphic_torch. 1-D and 2-D, arbitrary kernel/stride. Pure standard library, deterministic.

  python -m snn_data_lake.cortex.spike_pooling        # (also run by run_selfchecks.py)
"""


def max_pool_1d(x, k=2, stride=None):
    """Max-pool a 1-D spike map; keeps the peak of each window (preserves sparse high-frequency spikes)."""
    stride = stride or k
    return [max(x[i:i + k]) for i in range(0, len(x) - k + 1, stride)]


def avg_pool_1d(x, k=2, stride=None):
    """Average-pool a 1-D map (the low-pass control that smears sparse spikes)."""
    stride = stride or k
    return [sum(x[i:i + k]) / k for i in range(0, len(x) - k + 1, stride)]


def max_pool_2d(x, k=2, stride=None):
    """Max-pool a 2-D spike map (list of equal-length rows) over k x k windows."""
    stride = stride or k
    rows, cols = len(x), len(x[0])
    out = []
    for r in range(0, rows - k + 1, stride):
        out.append([max(x[r + dr][c + dc] for dr in range(k) for dc in range(k))
                    for c in range(0, cols - k + 1, stride)])
    return out


def main():
    # 1-D: a lone high-frequency spike in a quiet stream
    n, k = 16, 2
    x = [0.0] * n
    x[7] = 1.0                                   # sparse salient spike
    mx = max_pool_1d(x, k)
    av = avg_pool_1d(x, k)
    peak_max, peak_avg = max(mx), max(av)

    # 2-D: a lone hot pixel in a sparse map
    grid = [[0.0] * 8 for _ in range(8)]
    grid[3][5] = 1.0
    g2 = max_pool_2d(grid, 2)
    peak_grid = max(v for row in g2 for v in row)

    exp_len = (n - k) // k + 1                    # expected downsampled length

    print("=" * 60)
    print("SPIKE POOLING — max-pool preserves sparse high-frequency spikes")
    print("=" * 60)
    print(f"1-D {n}->{len(mx)} (k={k}); lone spike -> max peak {peak_max}, avg peak {peak_avg}")
    print(f"2-D 8x8->{len(g2)}x{len(g2[0])}; hot pixel -> max peak {peak_grid}")
    print(f"anti-low-pass: max keeps {peak_max} where avg smears to {peak_avg}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert peak_max == 1.0, f"max-pool must preserve the sparse spike at full amplitude ({peak_max})"
    assert peak_avg < peak_max, f"avg-pool must smear (low-pass) the sparse spike ({peak_avg} vs {peak_max})"
    assert len(mx) == exp_len, f"1-D downsample shape wrong ({len(mx)} != {exp_len})"
    assert len(g2) == 4 and len(g2[0]) == 4 and peak_grid == 1.0, "2-D max-pool must downsample + keep the peak"
    assert max_pool_1d(x, k) == max_pool_1d(x, k), "same input -> identical pooled output"
    print(f"self-check OK: max-pool keeps peak {peak_max} (avg smears to {peak_avg}); 1-D {n}->{len(mx)}, "
          f"2-D 8x8->{len(g2)}x{len(g2[0])}; deterministic")


if __name__ == "__main__":
    main()
