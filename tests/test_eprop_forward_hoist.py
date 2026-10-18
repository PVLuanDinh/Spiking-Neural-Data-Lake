"""EpropLIFBase.forward with the input projection hoisted out of the timestep loop is
bit-identical to the per-step original (kept here as the reference, not in src), and the
parity check rejects wrong hoists (P8 red)."""

import pytest
import torch

from snn_data_lake.plasticity.eprop_kernel import _TinyNet

SEEDS = range(5)
SHAPES = {"vision": (3, 6, 2, 4, 4), "audio": (3, 6, 16)}
N_IN = {"vision": 32, "audio": 16}


def old_forward(net, x):
    """The pre-hoist forward, verbatim."""
    x = net.flatten_frames(x.float())
    b, t = x.shape[0], x.shape[1]
    dev = x.device
    v = torch.zeros(b, net.hidden, device=dev)
    z = torch.zeros(b, net.hidden, device=dev)
    y = torch.zeros(b, net.num_classes, device=dev)
    logits_sum = torch.zeros(b, net.num_classes, device=dev)
    latent_sum = torch.zeros(b, net.hidden, device=dev)
    w_rec = net.W_rec * net.offdiag_mask(dev)
    for step in range(t):
        cur = x[:, step] @ net.W_in.t() + z @ w_rec.t()
        v = net.alpha * v - net.v_th * z + cur
        z = (v >= net.v_th).float()
        y = net.kappa * y + z @ net.W_out.t()
        logits_sum = logits_sum + (y + net.b_out)
        latent_sum = latent_sum + z
    return logits_sum, latent_sum


def wrong_forward(net, x, pick):
    """Old forward with the hoisted projection read through `pick(x_proj, step)` (a bad hoist)."""
    x = net.flatten_frames(x.float())
    b, t = x.shape[0], x.shape[1]
    x_proj = x @ net.W_in.t()
    v = torch.zeros(b, net.hidden)
    z = torch.zeros(b, net.hidden)
    y = torch.zeros(b, net.num_classes)
    logits_sum = torch.zeros(b, net.num_classes)
    latent_sum = torch.zeros(b, net.hidden)
    w_rec = net.W_rec * net.offdiag_mask(x.device)
    for step in range(t):
        cur = pick(x_proj, step) + z @ w_rec.t()
        v = net.alpha * v - net.v_th * z + cur
        z = (v >= net.v_th).float()
        y = net.kappa * y + z @ net.W_out.t()
        logits_sum = logits_sum + (y + net.b_out)
        latent_sum = latent_sum + z
    return logits_sum, latent_sum


def make(kind, seed):
    net = _TinyNet(n_in=N_IN[kind], hidden=8, num_classes=3, seed=seed)
    g = torch.Generator().manual_seed(100 + seed)
    return net, (torch.rand(*SHAPES[kind], generator=g) < 0.4).float()


def identical(fn_a, fn_b, kind, seed):
    net, x = make(kind, seed)
    (la, za), (lb, zb) = fn_a(net, x), fn_b(net, x)
    return torch.equal(la, lb) and torch.equal(za, zb)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", SHAPES)
def test_control_old_vs_old(kind, seed):
    assert identical(old_forward, old_forward, kind, seed)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", SHAPES)
def test_hoisted_forward_bit_identical(kind, seed):
    assert identical(old_forward, lambda net, x: net(x), kind, seed)


RED = {
    "shifted_step": lambda p, s: p[:, s - 1],
    "scaled_w_in": lambda p, s: p[:, s] * 1.05,
    "reversed_time": lambda p, s: p[:, -1 - s],
}


def test_red_wrong_hoists_fail_parity():
    caught = {
        name: not all(
            identical(old_forward, lambda net, x, pick=pick: wrong_forward(net, x, pick), k, s)
            for k in SHAPES
            for s in SEEDS
        )
        for name, pick in RED.items()
    }
    assert all(caught.values()), caught


def test_red_w_rec_projection_cannot_pass():
    net, x = make("audio", 0)
    with pytest.raises(RuntimeError):
        flat = net.flatten_frames(x)
        flat @ net.W_rec.t()
