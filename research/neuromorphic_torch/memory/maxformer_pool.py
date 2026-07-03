"""
Phase 4b — Max-Former pooling: max-pool spatial downsampling to counter the SNN low-pass bias.

Spiking neurons integrate and average, which acts as a low-pass filter and smears high-frequency
spatial detail (sharp edges, sparse hot pixels). Following the Max-Former architecture, MAX-pooling is
used for spatial downsampling instead of average pooling so that high-frequency peaks survive the
resolution reduction. Multi-step over a spiking feature map (T, B, C, H, W). Refs: Max-Former
(max-pool downsampling to preserve high-frequency features in spiking transformers).

Quarantined: needs torch + SpikingJelly (own venv). NOT part of the zero-dep core / CI.

  python memory/maxformer_pool.py        # CPU smoke: max-pool retains high-freq peaks vs avg-pool
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer


class MaxFormerPool(nn.Module):
    """Max-pool downsampling over a spiking feature map (T, B, C, H, W)."""

    def __init__(self, kernel=2):
        super().__init__()
        self.pool = layer.MaxPool2d(kernel)
        functional.set_step_mode(self, "m")

    def forward(self, x):
        functional.reset_net(self)
        return self.pool(x)


def smoke():
    torch.manual_seed(0)
    T, B, C, H, W = 6, 2, 3, 8, 8
    x = torch.full((T, B, C, H, W), 0.1)          # low background
    x[:, :, :, 2, 2] = 1.0                         # sparse high-frequency peaks (single hot pixels)
    x[:, :, :, 5, 6] = 1.0

    maxed = MaxFormerPool(2)(x)                                       # max-pool downsample
    avg = F.avg_pool2d(x.flatten(0, 1), 2).view(T, B, C, H // 2, W // 2)   # avg-pool baseline

    max_peak = maxed.max().item()
    avg_peak = avg.max().item()

    assert maxed.shape[-2:] == (H // 2, W // 2), "must downsample the spatial dims"
    assert max_peak > 1.5 * avg_peak, \
        f"max-pool must preserve the high-freq peak vs avg-pool ({max_peak:.2f} vs {avg_peak:.2f})"
    print(f"Max-Former smoke OK: peak retained {max_peak:.2f} (max-pool) vs {avg_peak:.2f} (avg-pool), "
          f"{max_peak / avg_peak:.1f}x; downsample {H}x{W} -> {maxed.shape[-2]}x{maxed.shape[-1]} "
          f"(high-frequency features preserved)")


if __name__ == "__main__":
    smoke()
