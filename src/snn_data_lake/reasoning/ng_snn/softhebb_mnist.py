"""
Unsupervised SoftHebb-CNN feature extractor for MNIST (torch / GPU tier).

Faithful port of the official SoftHebb front-end
  Journe, Rodriguez, Guo, Moraitis - "Hebbian deep learning without feedback",
  ICLR 2023 (arXiv:2209.11883); code: github.com/NeuromorphicComputing/SoftHebb
specifically the HebbSoftKrotovConv2d layer running in its DEFAULT mode (mode 0):
  the winner output-channel keeps +softmax_responsibility, every OTHER channel
  gets -softmax_responsibility (this is the plain "soft" negate-all-but-winner
  form; ranking_param / delta are NOT used in the default conv mode).

No backprop anywhere in the conv stack: each HebbConv2d learns fully locally,
per mini-batch, from its own (input, current, soft-WTA) triplet.

Part of reproducing NG-SNN (Neural Networks 2026): this module produces FROZEN
d_FE-dim features that a downstream NG-SNN head consumes. The Linear sanity head
here is throwaway (used only to prove the front-end is correct).

Run with the cu128 GPU venv:
  PYTHONUTF8=1 .venv-cu128/Scripts/python.exe research/ng_snn/softhebb_mnist.py
ASCII-only output (cp1252-safe).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


# ----------------------------------------------------------------------------
# Activation
# ----------------------------------------------------------------------------
class Triangle(nn.Module):
    """SoftHebb Triangle activation: relu(x - mean_over_channels(x)) ** power."""

    def __init__(self, power: float = 1.0):
        super().__init__()
        self.power = float(power)

    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        return F.relu(x) ** self.power

    def extra_repr(self):
        return "power=%s" % self.power


# ----------------------------------------------------------------------------
# SoftHebb soft-Krotov conv layer (default mode 0), fully local, no backprop
# ----------------------------------------------------------------------------
class HebbConv2d(nn.Module):
    """
    Local soft-WTA Hebbian conv layer. Weights are a registered buffer (never a
    Parameter): no gradient flows through them. When `learning` is True the
    forward pass also applies one local plasticity update per mini-batch.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        padding_mode="zeros",
        lebesgue_p=2,
        t_invert=1.0,
        lr_amp=0.08,
        power_lr=0.5,
        weight_range=2.0,
        weight_offset=0.0,
        seed=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.lebesgue_p = lebesgue_p
        self.t_invert = float(t_invert)
        self.lr_amp = float(lr_amp)
        self.power_lr = float(power_lr)
        self.padding_mode = padding_mode

        # Reflect-style padding is applied via F.pad in forward (F_pad); the
        # internal conv then uses zero internal padding. Matches the official
        # code so the plasticity conv sees the same padded input.
        self.F_pad = padding_mode != "zeros"
        if self.F_pad:
            self.padding = _pair(0)
            self.F_padding = (padding, padding, padding, padding)
        else:
            self.padding = _pair(padding)

        # weight init: 'positive' -> U(0, weight_range)
        g = torch.Generator().manual_seed(seed)
        w = (
            weight_range * torch.rand((out_channels, in_channels, *self.kernel_size), generator=g)
            + weight_offset
        )
        self.register_buffer("weight", w)
        # per-neuron adaptive learning rate (recomputed from the weight radius)
        self.register_buffer("lr_pc", torch.ones(out_channels))
        self._update_lr()

        self.learning = False

    # --- helpers ---------------------------------------------------------
    def _radius(self):
        w = self.weight.view(self.out_channels, -1)
        return torch.linalg.norm(w, dim=1, ord=self.lebesgue_p)

    def _update_lr(self):
        r = self._radius()
        # eta_i = lr_amp * (|r_i - 1| + 1e-10) ** power_lr  ; -> 0 as r_i -> 1
        self.lr_pc = self.lr_amp * torch.pow(torch.abs(r - 1.0) + 1e-10, self.power_lr)

    def _current(self, x):
        # lebesgue_p pre-transform: sign(W)*|W|^(p-1)  (p=2 -> == W)
        wp = torch.sign(self.weight) * torch.abs(self.weight) ** (self.lebesgue_p - 1)
        return F.conv2d(x, wp, None, self.stride, self.padding, self.dilation)

    def _get_wta(self, pre_x):
        # soft-Krotov, default mode 0: softmax responsibilities over OUTPUT
        # channels per spatial location; winner keeps +y, all others get -y.
        b, c, ho, wo = pre_x.shape
        flat = pre_x.transpose(0, 1).reshape(c, -1)  # (C, B*Ho*Wo)
        wta = torch.softmax(self.t_invert * flat, dim=0)  # over channels
        wta = -wta
        winners = torch.argmax(flat, dim=0)  # (B*Ho*Wo,)
        idx = torch.arange(flat.size(1), device=flat.device)
        wta[winners, idx] = -wta[winners, idx]  # flip winner back to +
        return wta.view(c, b, ho, wo).transpose(0, 1)  # (B, C, Ho, Wo)

    def _delta_w(self, x, pre_x, wta):
        # yx: correlation of soft-WTA with input patches (conv trick, summed
        # over batch AND spatial positions). x is the (possibly F-padded) input.
        yx = F.conv2d(
            x.transpose(0, 1),
            wta.transpose(0, 1),
            padding=self.padding,
            stride=self.dilation,
            dilation=self.stride,
        ).transpose(0, 1)
        yx = yx[:, :, : self.kernel_size[0], : self.kernel_size[1]]
        # yu: sum over (B,Ho,Wo) of (wta * current) -> (out_ch,)
        yu = torch.sum(wta * pre_x, dim=(0, 2, 3))
        dw = yx - yu.view(-1, 1, 1, 1) * self.weight  # Oja-style decay
        dw = dw / (torch.abs(dw).amax() + 1e-30)  # global max-abs norm
        return dw

    def forward(self, x):
        if self.F_pad:
            x = F.pad(x, self.F_padding, self.padding_mode)
        pre_x = self._current(x)
        if self.learning:
            with torch.no_grad():
                wta = self._get_wta(pre_x)
                dw = self._delta_w(x, pre_x, wta)
                self.weight.add_(self.lr_pc.view(-1, 1, 1, 1) * dw)  # ASCENT (+)
                self._update_lr()
        return pre_x

    def convergence(self, tol=5e-3):
        """Fraction of channels with ||w||_p within tol of 1 (+mean, std)."""
        r = self._radius()
        frac = (torch.abs(r - 1.0) < tol).float().mean().item()
        return frac, float(r.mean()), float(r.std())

    def extra_repr(self):
        return "in=%d out=%d k=%s s=%s p_mode=%s t_invert=%s lr_amp=%s power_lr=%s" % (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding_mode,
            self.t_invert,
            self.lr_amp,
            self.power_lr,
        )


# ----------------------------------------------------------------------------
# Block: BatchNorm2d(affine=False) -> HebbConv2d -> Triangle -> Pool
# ----------------------------------------------------------------------------
class SoftHebbBlock(nn.Module):
    def __init__(self, bn, conv, act, pool):
        super().__init__()
        self.bn = bn
        self.conv = conv
        self.act = act
        self.pool = pool

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SoftHebbCNN(nn.Module):
    """
    2- or 3-conv SoftHebb feature extractor. Hyper-parameters follow the task
    spec (which mirrors the official SoftHebb wide conv net).
    """

    def __init__(self, n_blocks=2, seed=0):
        super().__init__()
        assert n_blocks in (2, 3)
        self.n_blocks = n_blocks
        blocks = []

        # Block 0: 1 -> 96, k5 s1 p2 reflect ; t_invert 1.0 ; Triangle(0.7) ; MaxPool(4,2,1)
        blocks.append(
            SoftHebbBlock(
                nn.BatchNorm2d(1, affine=False),
                HebbConv2d(
                    1,
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
            )
        )
        # Block 1: 96 -> 384, k3 s1 p1 ; t_invert 0.65 ; Triangle(1.4) ; MaxPool(4,2,1)
        blocks.append(
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
            )
        )
        if n_blocks == 3:
            # Block 2: 384 -> 1536, k3 s1 p1 ; t_invert 0.25 ; Triangle(1.0) ; AvgPool(2,2,0)
            blocks.append(
                SoftHebbBlock(
                    nn.BatchNorm2d(384, affine=False),
                    HebbConv2d(
                        384,
                        1536,
                        3,
                        stride=1,
                        padding=1,
                        padding_mode="zeros",
                        t_invert=0.25,
                        lr_amp=0.01,
                        power_lr=0.5,
                        seed=seed,
                    ),
                    Triangle(1.0),
                    nn.AvgPool2d(2, 2, 0),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def features(self, x):
        return self.forward(x).reshape(x.size(0), -1)
