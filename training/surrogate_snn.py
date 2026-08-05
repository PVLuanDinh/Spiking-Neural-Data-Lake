"""surrogate_snn.py -- surrogate-gradient deep spiking classifier on N-MNIST event data, and
`SurrogateEncoder`, the FALLBACK perception encoder the Brain's torch perception tier loads.

WHAT THIS IS (honest framing). A small spiking conv net (LIF layers, fast-sigmoid surrogate
derivative) trained by BPTT-through-time on N-MNIST event frames with a spike-count (rate)
cross-entropy readout, plus `SurrogateEncoder`, which turns an event frame into a latent vector (the
penultimate spiking layer's spike counts) and into a predicted digit. Any accuracy printed is the
number ACTUALLY obtained at the scale it was run at; nothing here is extrapolated to full-dataset or
long-training numbers.

E-PROP INVARIANT -- WHICH HALF IS LIVE, AND WHICH IS NOT. `loss.backward()` (BPTT through the T
unrolled time-bins) appears in exactly two functions, `train_and_eval` and `_self_check`, and BOTH
are OFFLINE: their only call sites are in this module's own `main()`, reached only from
`__main__`. The half the Brain imports is `SurrogateEncoder.load` / `.encode` / `.classify`, and all
three are `@torch.no_grad()` -- no gradient, no optimizer, no weight update. This module is
therefore promotable under the e-prop invariant because BPTT here is NOT on a live learning path;
it is an offline trainer plus a frozen serving wrapper. The brain-native e-prop counterpart
(`training/eprop_nmnist.py`, `EpropEncoder`, which carries its own independent BPTT cosine control)
is PREFERRED by `cortex/brain.py`; this surrogate encoder is only the fallback when no e-prop
checkpoint is on disk.

DEPENDENCIES (P7; the zero-dependency rule was abolished 2026-07-20). The former docstring here
claimed this file was "the OPTIONAL torch [train]/[gpu] tier", that "the zero-dep stdlib safety-core
NEVER imports this module", and that "nothing in src/snn_data_lake/ or brain.py imports back". All
three are retired or simply false: torch is an unconditional core dependency, there is no zero-dep
core to protect, and `cortex/brain.py` DOES import `SurrogateEncoder` from here. What remains TRUE
is the weights gate -- with no checkpoint on disk the Brain never constructs an encoder and uses its
stdlib perception path. That is the honest claim, and it is the only one kept.

  snntorch was DROPPED, and this is a P2 adopt-then-retire, NOT a P7 dependency-avoidance argument.
  What it provided -- `snn.Leaky` and `surrogate.fast_sigmoid` -- is the ~25 lines this file now
  carries as `LeakyLIF` and `_FastSigmoid`. The reason is structural and specific to this
  architecture, which is what P7 demands of any rejection: snntorch is declared ONLY in the
  `[train]`/`[gpu]` extras, while a module living under `src/snn_data_lake/` is gated by
  `run_selfchecks.TORCH_GATED`, which probes `import torch` and nothing else, and CI installs
  `-e .` plus `[dev]` only (`.github/workflows/ci.yml:42`, `:85`). An extra imported from core makes
  that gate RED, not skipped. Precedent: `hippocampus/ng_snn.py` was promoted out of research with
  a hand-rolled LIF and torch as its only numeric dependency, for the same reason.

  MEASURED, not assumed. `LeakyLIF` reproduces `snntorch.Leaky(beta, spike_grad=fast_sigmoid(slope),
  reset_mechanism="subtract")` semantics exactly -- same reset-delay ordering, same detached reset,
  same `beta.clamp(0, 1)` -- and it registers the SAME four buffers (`threshold`, `beta`,
  `graded_spikes_factor`, `reset_mechanism_val`), so checkpoints trained by the snntorch version
  load unchanged. Verified against snntorch 1.0.0 / torch 2.11.0+cu128 on the real checkpoint
  `models/surrogate_nmnist.pth` (hidden=128, beta=0.9, slope=25.0), 32 N-MNIST frames from
  `models/nmnist_pack.npz`: max |latent_new - latent_snntorch| = 0.0 over 4096 values, max |logit
  delta| = 0.0 over 320 values, class predictions identical 32/32, and the conv1 BPTT gradient
  matched to max |delta| = 0.0 over 600 values. BIT-EXACT, at full precision, not to 4 decimals.

  Checkpoints are loaded with `weights_only=True`. The path can come from the `SNN_PERCEPTION_CKPT`
  environment variable on the Brain's construction path, so a checkpoint must be data, never code.

DATA. The N-MNIST disk-streaming loader is REUSED, not reinvented: `research/ng_snn/data.py`
(tonic DiskCachedDataset -> [B, T, 2, 34, 34] frames). It is resolved by walking UP from `__file__`
to the repo root rather than by counting `..` hops, and loaded BY FILE PATH under a private module
name -- see `_repo_root` and `_load_data_module` for why both of those matter for a file that moves.

  .venv-lake/Scripts/python training/surrogate_snn.py
      # DEFAULT: the self-check. One tiny synthetic batch, CPU, no dataset, no network, no GPU.
  .venv-lake/Scripts/python training/surrogate_snn.py --train --epochs 1 --limit 2000
      # short real N-MNIST train on a small subset -> prints test accuracy (needs tonic + dataset)

  (Both lines read `python -m snn_data_lake.cortex.surrogate_snn` until 2026-08-04. That module does
  not exist: this file is still `training/surrogate_snn.py` and the promotion the paths anticipated
  has not happened. RUN, not read -- both exited 1 with "No module named
  snn_data_lake.cortex.surrogate_snn". `_repo_root` and `_load_data_module` below are still written
  to survive that promotion, which is the correct thing for them to be; the commands were not.)

ASCII-only output (Windows cp1252-safe): the self-check runs in-process under pytest after
promotion, so a non-ASCII character in a print() is a crash, not a cosmetic issue.
"""

import argparse
import importlib.util
import os
import tempfile

import torch
import torch.nn as nn

# N-MNIST event geometry (matches research/ng_snn/data.py SENSOR = (34, 34, 2)).
SENSOR_H = 34
SENSOR_W = 34
IN_CH = 2
NUM_CLASSES = 10

# P1 (brain-native / no-hardcode) LABEL for every default below, stated once rather than per line.
# These are OFFLINE-TRAINER hyperparameters and self-check fixtures. Every one is overridable on the
# CLI, and every one sits on a path with ZERO live callers -- `train_and_eval` and `_self_check` are
# reached only from this module's `__main__`. That is P1 honesty-guard case (c): the constant is
# inert (default-off, no live caller), so a learned or running-baseline form would never charge.
#
# They are explicitly NOT the values the LIVE path uses. `SurrogateEncoder.load` restores `beta`,
# `threshold` and `graded_spikes_factor` from the checkpoint's own buffers and the architecture from
# its saved `config`, so at serve time these module-level numbers arbitrate nothing at all. The one
# value that would arbitrate live behaviour if it were hardcoded -- the reset mechanism -- is read
# from the checkpoint and CHECKED rather than assumed (see `SpikingConvNet.assert_supported_reset`).
DEFAULT_T = 15  # event time-bins; matches research/ng_snn/data.py DEFAULT_T (paper-set)
DEFAULT_HIDDEN = 128  # penultimate spiking-layer width == the latent dimension the encoder emits
DEFAULT_BETA = 0.9  # LIF membrane decay, training-time initial value only
DEFAULT_SLOPE = 25.0  # fast-sigmoid surrogate steepness (Zenke & Ganguli SuperSpike default)
DEFAULT_THRESHOLD = 1.0  # LIF firing threshold; a fixed point on the membrane's own unit scale


# ----------------------------------------------------------------------------
# The spiking primitives. Formerly snntorch.surrogate.fast_sigmoid + snntorch.Leaky; see the
# docstring for why they were adopted in-file and for the bit-exactness measurement.
# ----------------------------------------------------------------------------
class _FastSigmoid(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward: d/du = 1 / (1 + slope*|u|)^2.

    Zenke & Ganguli (2018), SuperSpike. Identical to snntorch.surrogate.FastSigmoid.
    """

    @staticmethod
    def forward(ctx, input_, slope):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad = grad_output.clone() / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None


def _fast_sigmoid(slope=DEFAULT_SLOPE):
    def inner(x):
        return _FastSigmoid.apply(x, slope)

    return inner


class LeakyLIF(nn.Module):
    """First-order leaky integrate-and-fire layer with reset-by-subtraction.

    Recurrence, in the exact order snntorch.Leaky(reset_mechanism="subtract", reset_delay=True) uses
    -- the reset is computed from the PREVIOUS membrane, which is what makes the reset one step
    delayed, and it is detached so no gradient flows through it:

        reset = H(mem - threshold).detach()
        mem   = clamp(beta, 0, 1) * mem + input + (-reset * threshold)
        spk   = H(mem - threshold) * graded_spikes_factor

    `threshold`, `beta`, `graded_spikes_factor` and `reset_mechanism_val` are registered buffers
    with the same names snntorch uses, so a checkpoint saved by the snntorch build loads into this
    class unchanged, and the LIF parameters at serve time come from the CHECKPOINT, not this file.
    """

    def __init__(self, beta, slope=DEFAULT_SLOPE, threshold=DEFAULT_THRESHOLD):
        super().__init__()
        self.register_buffer("threshold", torch.as_tensor(float(threshold)))
        self.register_buffer("beta", torch.as_tensor(float(beta)))
        self.register_buffer("graded_spikes_factor", torch.as_tensor(1.0))
        # 0 == reset by subtraction, mirroring snntorch's encoding. Kept as a buffer so a checkpoint
        # trained with a DIFFERENT reset mechanism is detectable rather than silently mis-served.
        self.register_buffer("reset_mechanism_val", torch.as_tensor(0, dtype=torch.int64))
        self.spike_grad = _fast_sigmoid(slope)

    def init_leaky(self):
        """Membrane initial state. None means 'allocate zeros_like(input) on the first step'."""
        return None

    def forward(self, input_, mem=None):
        if mem is None or mem.shape != input_.shape:
            mem = torch.zeros_like(input_)
        reset = self.spike_grad(mem - self.threshold).detach()
        mem = self.beta.clamp(0, 1) * mem + input_ - reset * self.threshold
        spk = self.spike_grad(mem - self.threshold) * self.graded_spikes_factor
        return spk, mem


# ----------------------------------------------------------------------------
# The spiking classifier: 2 spiking conv blocks -> spiking FC (latent) -> spiking readout.
# Membrane state is threaded explicitly so the encoder can read the penultimate layer's spike
# activity cleanly. Readout = spike COUNT per class over T (rate-coded), fed to cross-entropy.
# ----------------------------------------------------------------------------
class SpikingConvNet(nn.Module):
    def __init__(
        self,
        in_ch=IN_CH,
        num_classes=NUM_CLASSES,
        hidden=DEFAULT_HIDDEN,
        beta=DEFAULT_BETA,
        slope=DEFAULT_SLOPE,
        sensor_h=SENSOR_H,
        sensor_w=SENSOR_W,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.beta = float(beta)
        self.slope = float(slope)
        self.sensor_h = int(sensor_h)
        self.sensor_w = int(sensor_w)

        self.conv1 = nn.Conv2d(self.in_ch, 12, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = LeakyLIF(beta, slope)
        self.conv2 = nn.Conv2d(12, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = LeakyLIF(beta, slope)

        flat = self._flat_dim()  # derive FC input from a shape-only pass
        self.fc1 = nn.Linear(flat, self.hidden)
        self.lif3 = LeakyLIF(beta, slope)  # PENULTIMATE spiking layer == the latent
        self.fc2 = nn.Linear(self.hidden, self.num_classes)
        self.lif4 = LeakyLIF(beta, slope)  # spiking readout

    def _flat_dim(self):
        """Flatten width after the two conv+pool blocks (shape-only, no LIF, no grad)."""
        with torch.no_grad():
            d = torch.zeros(1, self.in_ch, self.sensor_h, self.sensor_w)
            d = self.pool1(self.conv1(d))
            d = self.pool2(self.conv2(d))
            return int(d.flatten(1).shape[1])

    def lif_layers(self):
        return (self.lif1, self.lif2, self.lif3, self.lif4)

    def assert_supported_reset(self):
        """Refuse a checkpoint whose LIFs were trained with a reset mechanism this class does not
        implement. snntorch encodes 0=subtract, 1=zero, 2=none; `LeakyLIF` implements SUBTRACT only.
        Loading a reset-to-zero checkpoint would run silently and serve wrong perception values, so
        the mismatch is raised at load time instead of being assumed away."""
        for i, lif in enumerate(self.lif_layers(), start=1):
            val = int(lif.reset_mechanism_val)
            if val != 0:
                raise ValueError(
                    f"lif{i}: checkpoint reset_mechanism_val={val}, but LeakyLIF implements "
                    f"reset-by-subtraction (0) only -- refusing to serve it as if it matched"
                )

    def forward(self, x):
        """x: [B, T, C, H, W] event frames. Returns (spk_out_sum [B, num_classes],
        latent_sum [B, hidden]) -- both spike COUNTS summed over the T time-bins. spk_out_sum are
        the class logits (rate readout); latent_sum is the penultimate spiking-layer activity."""
        b, t = x.shape[0], x.shape[1]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk_out_sum = torch.zeros(b, self.num_classes, device=x.device)
        latent_sum = torch.zeros(b, self.hidden, device=x.device)
        for step in range(t):
            xt = x[:, step]
            cur1 = self.pool1(self.conv1(xt))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)  # penultimate spikes -> latent
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk_out_sum = spk_out_sum + spk4
            latent_sum = latent_sum + spk3
        return spk_out_sum, latent_sum

    def config(self):
        return dict(
            in_ch=self.in_ch,
            num_classes=self.num_classes,
            hidden=self.hidden,
            beta=self.beta,
            slope=self.slope,
            sensor_h=self.sensor_h,
            sensor_w=self.sensor_w,
        )


# ----------------------------------------------------------------------------
# The perception interface the Brain's hook calls. Lazy: constructing or loading an encoder does NOT
# train and does NOT require tonic or the dataset -- only torch and the saved weights.
# ----------------------------------------------------------------------------
class SurrogateEncoder:
    """Wraps a trained SpikingConvNet as a frozen perception encoder.

    encode(event_frame) -> latent vector (the penultimate spiking layer's spike-count activity):
      * event_frame [T, C, H, W]      -> latent [hidden]        (single sample; brain-hook path)
      * event_frame [B, T, C, H, W]   -> latent [B, hidden]     (batched)
    Accepts a numpy array or a torch tensor. Values are spike counts over the T window (>= 0).

    load(path, device=...) rebuilds the net from the checkpoint's saved config and weights -- no
    training, no dataset needed. This is the interface `cortex/brain.py` imports and calls; when the
    checkpoint is absent the Brain uses its stdlib perception path instead. NOTHING on this class
    computes a gradient: every entry point is `@torch.no_grad()`.
    """

    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        # weights_only=True: the path reaches this from the SNN_PERCEPTION_CKPT environment variable
        # on the Brain's construction path, so a checkpoint is data and must never be executable.
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        net = SpikingConvNet(**cfg)
        net.load_state_dict(ckpt["model"])
        net.assert_supported_reset()
        return cls(net, device=device)

    def _prepare(self, event_frame, what):
        x = event_frame
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float().to(self.device)
        single = x.ndim == 4  # [T, C, H, W] -> add a batch dim
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"{what} expects [T,C,H,W] or [B,T,C,H,W]; got shape {tuple(x.shape)}")
        return x, single

    @torch.no_grad()
    def encode(self, event_frame):
        x, single = self._prepare(event_frame, "encode")
        _, latent = self.net(x)
        return latent[0] if single else latent

    @torch.no_grad()
    def classify(self, event_frame):
        """Predicted class label(s) from the rate readout -- shows the encoder PERCEIVES real
        content (not noise) on the live path. [T,C,H,W] -> int; [B,T,C,H,W] -> list[int]."""
        x, single = self._prepare(event_frame, "classify")
        logits, _ = self.net(x)
        pred = logits.argmax(dim=1)
        return int(pred[0]) if single else [int(v) for v in pred]


# ----------------------------------------------------------------------------
# Data (reused from research/ng_snn/data.py). Both helpers below exist because this file MOVES.
# ----------------------------------------------------------------------------
def _repo_root():
    """Locate the repo root by walking UP from this file to a marker, not by counting `..` hops.

    SELF-CONTAINED PATH FIREWALL, and the highest-value edit in this file. The previous form was
    `os.path.join(os.path.dirname(__file__), "..", "research", "ng_snn")`, which encodes this
    module's own directory DEPTH. Promoting the file (training/ -> src/snn_data_lake/cortex/)
    silently retargets it at `src/snn_data_lake/research/ng_snn`, which does not exist -- and since
    the only caller sits inside `train_and_eval`, which needs tonic plus the dataset, the breakage
    would never surface at the gate. It would FAIL GREEN. Walking up to the marker makes the
    resolution depth-independent, and `_self_check` asserts the result exists, so a move that breaks
    it is RED at the gate instead of silent.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "pyproject.toml")) and os.path.isdir(
            os.path.join(d, "research")
        ):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                f"repo root not found above {os.path.abspath(__file__)} "
                f"(looked for a directory holding both pyproject.toml and research/)"
            )
        d = parent


def ng_snn_data_file():
    """Absolute path to research/ng_snn/data.py, the N-MNIST disk-streaming loader."""
    return os.path.join(_repo_root(), "research", "ng_snn", "data.py")


def _load_data_module():
    """Import research/ng_snn/data.py BY FILE PATH, under a private module name.

    The previous form inserted that directory onto sys.path and did a bare `import data`, which
    binds the global module name "data" process-wide and leaves the sys.path entry behind for
    everything imported afterwards (docs/AUDIT_BACKLOG.md M37). Loading by path keeps the import
    local to this call and cannot shadow another module named `data`.
    """
    path = ng_snn_data_file()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"N-MNIST loader not found at {path}")
    spec = importlib.util.spec_from_file_location("_snn_ng_snn_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# Train / eval. OFFLINE ONLY -- `loss.backward()` below is BPTT and has no live call site.
# ----------------------------------------------------------------------------
def evaluate(net, loader, device, max_samples=None):
    """Test accuracy = argmax of the spike-count readout. Caps at max_samples for quick runs.

    Returns (accuracy_pct, n_seen). n_seen is returned rather than folded away so a caller can tell
    "0% accuracy" apart from "the loader yielded nothing", which are not the same result.
    """
    net.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()
            spk_out, _ = net(x)
            correct += int((spk_out.argmax(1) == y).sum().item())
            seen += int(y.shape[0])
            if max_samples is not None and seen >= max_samples:
                break
    return (100.0 * correct / seen) if seen else 0.0, seen


def train_and_eval(args, device):
    torch.manual_seed(args.seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = _load_data_module()
    print(
        f"device={device}  T={args.T}  batch={args.batch_size}  hidden={args.hidden}  "
        f"limit={args.limit}  epochs={args.epochs}"
    )
    if device == "cuda" and torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    root = _repo_root()  # cwd-independent, for the same reason the loader path is
    train_loader, test_loader = data.make_nmnist_loaders(
        save_to=os.path.join(root, "data"),
        n_time_bins=args.T,
        batch_size=args.batch_size,
        num_workers=args.workers,
        cache_dir=os.path.join(root, "data", "nmnist_cache"),
    )
    print(f"train_batches={len(train_loader)}  test_batches={len(test_loader)}")

    net = SpikingConvNet(hidden=args.hidden, beta=args.beta, slope=args.slope).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"model: 2 spiking-conv blocks -> spiking FC({args.hidden}) latent -> spiking readout | "
        f"params={n_params}"
    )
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        net.train()
        seen = 0
        running = 0.0
        nb = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()  # tonic labels are int32; CE needs int64
            spk_out, _ = net(x)
            loss = loss_fn(spk_out, y)  # CE on spike-count logits (rate readout)
            opt.zero_grad()
            loss.backward()  # OFFLINE BPTT through the T unrolled time-bins; no live call site
            opt.step()
            running += float(loss.item())
            nb += 1
            seen += int(y.shape[0])
            if nb % 20 == 0:
                print(f"  epoch {epoch}  batch {nb}  seen {seen}  loss {running / nb:.4f}")
            if args.limit is not None and seen >= args.limit:
                break
        acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
        print(
            f"epoch {epoch} done: train_samples={seen}  avg_loss={running / max(nb, 1):.4f}  "
            f"test_acc={acc:.2f}% (on {ntest} test samples)"
        )

    acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
    if not ntest:
        # A 0.00% line that actually means "the test loader was empty" is the silent-degradation
        # shape this repo keeps catching. Say which one happened.
        raise RuntimeError("test loader yielded 0 samples -- accuracy is undefined, not 0%")
    print("=" * 72)
    print(f"SURROGATE-SNN (torch perception tier, BPTT-trained) | device={device}")
    print(
        f"test accuracy {acc:.2f}% on {ntest} N-MNIST test samples "
        f"(T={args.T}, {min(args.limit or 10**12, 60000)} train samples, {args.epochs} epoch(s))"
    )
    print("HONEST: small-scale number, not SOTA and not extrapolated. The brain-native e-prop")
    print("encoder (training/eprop_nmnist.py) is PREFERRED; this is the fallback tier, and the")
    print(f"brain uses stdlib perception when no checkpoint exists. Latent dim = {args.hidden}.")
    print("=" * 72)

    if args.save:
        torch.save({"model": net.state_dict(), "config": net.config()}, args.save)
        print(f"saved encoder checkpoint -> {args.save}")
    return acc


# ----------------------------------------------------------------------------
# The self-check. No dataset, no network, no GPU, CPU-deterministic. This is what runs when the
# module is invoked with no arguments, which is how run_selfchecks.py invokes it (both the
# subprocess path and run_one_in_process, which forces sys.argv to [module_name]).
#
# P8: every criterion below has a REACHABLE RED, and three of them carry that red INSIDE the check
# as a constructed control -- C2's true-Heaviside swap, C4's lif1.beta neuter, C5's wrong-reset
# checkpoint. Deleted from the previous version rather than weakened: `assert spk_out.shape ==
# (b, NUM_CLASSES)` and
# `assert latent.shape == (b, 32)`. Both were structurally unfalsifiable: forward() allocates those
# accumulators as `torch.zeros(b, self.num_classes)` and `torch.zeros(b, self.hidden)` and adds
# tensors of exactly those shapes to them, so no input to this module can make either shape differ.
# ----------------------------------------------------------------------------
def _heaviside_no_surrogate(x):
    """A TRUE Heaviside: no custom backward, so it detaches the graph. Negative control only."""
    return (x > 0).float()


def _first_layer_grad(net):
    g = net.conv1.weight.grad
    return None if g is None else float(g.abs().sum())


def _fwd_bwd(net, x, y):
    """One forward + backward. Returns the summed |grad| at the FIRST conv layer, or None.

    `loss.requires_grad` is checked rather than assumed: with a true Heaviside spike function the
    whole graph is detached and `loss.backward()` would raise instead of reporting no gradient. The
    negative control below needs "no gradient" as a MEASUREMENT, not as an exception.
    """
    net.zero_grad(set_to_none=True)
    spk_out, latent = net(x)
    loss = nn.CrossEntropyLoss()(spk_out, y)
    if loss.requires_grad:
        loss.backward()
    return spk_out, latent, float(loss.item()), _first_layer_grad(net)


def _synthetic_frames(b, t_steps, device):
    """A synthetic event batch with the STATISTICS of a real N-MNIST frame, not a token one.

    Measured from models/nmnist_pack.npz (256 x 15 x 2 x 34 x 34, the pack every perception number
    in this repo was taken against): 6.0% of cells nonzero, mean nonzero value 1.98, max 9.
    Frames are event COUNTS per (bin, polarity, pixel), NOT binary.

    This matters, and it is the disclosure P8 asks for rather than a tidied-up detail. The previous
    smoke used T=4 and a BINARY `rand < 0.1` mask, ~1/30th of the real drive. Measured: with that
    fixture the whole network is SILENT -- latent.sum() == 0.0 and logits.sum() == 0.0 at every
    seed tried. So the old `assert latent.shape == (b, 32)` was passing over an all-zero tensor, the
    argmax readout was arbitrary, and no criterion in that smoke ever saw a spike. With this fixture
    (T=15, the real geometry) latent.sum() == 11.0 and logits.sum() == 4.0 on an untrained net, so
    C2a below can assert activity and mean it.
    """
    shape = (b, t_steps, IN_CH, SENSOR_H, SENSOR_W)
    mask = (torch.rand(shape, device=device) < 0.06).float()
    return mask * (1.0 + torch.poisson(torch.ones(shape, device=device)))


def _self_check(device="cpu"):
    # DETERMINISM WITHOUT THE BLAST RADIUS (2026-08-05). This was a bare `torch.manual_seed(0)`,
    # which mutates the PROCESS-GLOBAL torch RNG. Both sibling perception trainers
    # (training/eprop_nmnist.py and training/eprop_shd.py) removed exactly that call, each with a
    # written rationale -- "run_selfchecks drives CORE + TORCH_GATED through runpy in ONE process;
    # tests/test_selfchecks.py does the same under pytest, so a global reseed here silently
    # re-rolls the RNG for every module parametrized after it" -- and this file kept it while its
    # OWN docstring says "the self-check runs in-process under pytest after promotion". One of
    # three is not a convention.
    # It is NOT simply deleted, because unlike the siblings this check has no per-instance
    # generator to fall back on: `_synthetic_frames` uses torch.rand/torch.poisson, the labels use
    # torch.randint, and nn.Conv2d/nn.Linear initialise from the global RNG. Deleting the seed
    # would make the check non-deterministic; leaving it reaches into every other module. So the
    # global stream is seeded and then RESTORED, which is both properties at once. MEASURED: the
    # printed numbers are byte-identical to the previous version, and the process RNG is
    # bit-identical before and after the call (asserted in `main`).
    _rng_state = torch.random.get_rng_state()
    try:
        return _self_check_body(device)
    finally:
        torch.random.set_rng_state(_rng_state)


def _self_check_body(device="cpu"):
    torch.manual_seed(0)
    t_steps, b, hidden = DEFAULT_T, 2, 32
    print(
        f"self-check: device={device}  synthetic batch=[B={b}, T={t_steps}, C={IN_CH}, "
        f"{SENSOR_H}x{SENSOR_W}] (no dataset, no network)"
    )

    # --- C1: the path firewall. RED is reachable by MOVING this file without fixing _repo_root,
    # or by renaming research/ng_snn/data.py. That is the exact failure this check exists to make
    # loud, because every other caller of it is behind tonic + a dataset.
    loader_py = ng_snn_data_file()
    assert os.path.isfile(loader_py), f"N-MNIST loader path went stale: {loader_py}"
    print(
        f"  C1 loader path resolves (depth-independent): {os.path.relpath(loader_py, _repo_root())}"
    )

    net = SpikingConvNet(hidden=hidden).to(device)
    x = _synthetic_frames(b, t_steps, device)
    y = torch.randint(0, NUM_CLASSES, (b,), device=device)

    # --- C2: the surrogate derivative reaches the FIRST layer. A spiking net whose gradient dies at
    # the last linear layer trains nothing, and `any(p.grad is not None ...)` -- the previous
    # criterion -- passes in exactly that case. conv1 is four LIF layers deep, so a nonzero gradient
    # there is only possible if the surrogate propagates through all of them.
    spk_out, latent, loss, g_live = _fwd_bwd(net, x, y)

    # --- C2a: the network actually SPIKES on this fixture. A silent net satisfies every shape and
    # dispatch criterion below while measuring nothing at all -- that is exactly what the previous
    # smoke did (see _synthetic_frames). RED is reachable and was measured: swap the fixture back to
    # T=4 with a binary `rand < 0.1` mask and both sums go to 0.0.
    n_lat = float(latent.sum().item())
    n_out = float(spk_out.sum().item())
    assert n_lat > 0.0, "penultimate layer emitted ZERO spikes -- the latent is an all-zero vector"
    assert n_out > 0.0, "readout layer emitted ZERO spikes -- the class logits are all-zero"
    print(f"  C2a network is active: latent spikes={n_lat:.0f}  readout spikes={n_out:.0f}")

    # NEGATIVE CONTROL, run here rather than asserted from a comment: swap the surrogate for a true
    # Heaviside and the same measurement must collapse. If it does not, C2 is not measuring the
    # surrogate and the check is worthless.
    for lif in net.lif_layers():
        lif.spike_grad = _heaviside_no_surrogate
    _, _, _, g_dead = _fwd_bwd(net, x, y)
    for lif in net.lif_layers():
        lif.spike_grad = _fast_sigmoid(net.slope)
    # 0 < g_live < inf catches all three ways this fails: no gradient (None), a dead gradient (0.0),
    # and a diverged one (NaN compares False, inf fails the upper bound).
    assert g_live is not None and 0.0 < g_live < float("inf"), (
        f"surrogate gradient did not reach conv1 finite and nonzero (g={g_live})"
    )
    assert not g_dead, f"negative control did NOT collapse (g={g_dead}) -- C2 measures nothing"
    print(
        f"  C2 surrogate reaches conv1: |grad|={g_live:.6f} with fast-sigmoid, "
        f"{g_dead} with true Heaviside (negative control)  loss={loss:.4f}"
    )

    with tempfile.TemporaryDirectory() as enc_dir:
        ckpt = os.path.join(enc_dir, "self_check_encoder.pt")
        torch.save({"model": net.state_dict(), "config": net.config()}, ckpt)
        enc = SurrogateEncoder.load(ckpt, device=device)

        # --- C3: the encode() rank dispatch. RED: invert the `x.ndim == 4` test and both shapes
        # come back wrong. This is real branching logic, not an allocation shape.
        single = enc.encode(x[0])
        batched = enc.encode(x)
        assert tuple(single.shape) == (hidden,), (
            f"[T,C,H,W] must give [hidden]; got {tuple(single.shape)}"
        )
        assert tuple(batched.shape) == (b, hidden), (
            f"[B,...] must give [B,hidden]; got {tuple(batched.shape)}"
        )
        assert torch.equal(single, batched[0]), "single-sample path disagrees with the batched path"

        # --- C4: load() round-trips the net EXACTLY. Not "close" -- the serving encoder must be the
        # trained net, and both run float32 on the same device, so bit-equality is the honest bar.
        with torch.no_grad():
            _, latent_direct = net(x)
        assert torch.equal(batched, latent_direct), "loaded encoder does not reproduce the net"

        # ATTRIBUTION for C4 (P8: per-item neuter, not "the total matched"). Perturb ONE restored
        # buffer -- lif1.beta -- and the equality must break. If it does not, C4 is passing because
        # the buffers are irrelevant, not because they round-tripped.
        sd = dict(net.state_dict())
        sd["lif1.beta"] = torch.as_tensor(0.1)
        bad = os.path.join(enc_dir, "perturbed.pt")
        torch.save({"model": sd, "config": net.config()}, bad)
        latent_perturbed = SurrogateEncoder.load(bad, device=device).encode(x)
        assert not torch.equal(latent_perturbed, latent_direct), (
            "perturbing lif1.beta changed nothing -- the LIF buffers are not load-bearing, so C4 "
            "would pass even if load() dropped them"
        )
        print("  C3/C4 encode dispatch + exact load round-trip (beta-neuter control breaks it)")

        # --- C5: a checkpoint whose reset mechanism this class does not implement must RAISE, not
        # be served. RED is constructed right here: snntorch encodes reset-to-zero as 1.
        sd_reset = dict(net.state_dict())
        sd_reset["lif1.reset_mechanism_val"] = torch.as_tensor(1, dtype=torch.int64)
        wrong = os.path.join(enc_dir, "wrong_reset.pt")
        torch.save({"model": sd_reset, "config": net.config()}, wrong)
        raised = False
        try:
            SurrogateEncoder.load(wrong, device=device)
        except ValueError:
            raised = True
        assert raised, "a reset-to-zero checkpoint loaded silently -- it would serve wrong values"

        # --- C6: classify() rank dispatch and types. RED: an off-by-one in the batched branch, or
        # returning a tensor where the Brain's hook does int(...) on the result.
        one = enc.classify(x[0])
        many = enc.classify(x)
        assert isinstance(one, int), f"classify([T,C,H,W]) must give int; got {type(one).__name__}"
        assert isinstance(many, list) and len(many) == b, f"classify([B,...]) must give {b} ints"
        assert one == many[0], "classify single/batched disagree"
        print(
            f"  C5/C6 wrong-reset checkpoint refused; classify -> {one} (single) / {many} (batched)"
        )

    print(f"SELF-CHECK OK: 7 criteria, 3 with a constructed control (no dataset, {device}).")
    return True


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Surrogate-gradient spiking classifier on N-MNIST + the SurrogateEncoder "
        "perception interface. With no arguments it runs its self-check."
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="run the OFFLINE N-MNIST trainer (needs tonic + dataset); default is the self-check",
    )
    p.add_argument("--smoke", action="store_true", help="alias for the default self-check")
    p.add_argument("--epochs", type=int, default=1, help="training epochs")
    p.add_argument(
        "--limit", type=int, default=None, help="cap training samples per epoch (quick subset run)"
    )
    p.add_argument(
        "--eval-limit",
        type=int,
        default=2000,
        dest="eval_limit",
        help="cap test samples during eval (quick runs); a large value for the full test set",
    )
    p.add_argument("--T", type=int, default=DEFAULT_T, help="event time-bins per sample")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="penultimate/latent width")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA, help="LIF membrane decay")
    p.add_argument(
        "--slope", type=float, default=DEFAULT_SLOPE, help="fast-sigmoid surrogate slope"
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2, help="dataloader workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None, help="path to save the encoder checkpoint")
    # No default computed from torch.cuda.is_available(): the SELF-CHECK must be CPU-deterministic
    # and must run where CI has no card (P5), while the trainer should default to the GPU it needs.
    # Resolved per-mode in main() so neither choice is imposed on the other.
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    return p


def main():
    args = build_parser().parse_args()
    train = args.train and not args.smoke
    if args.device:
        device = args.device
    else:
        device = "cuda" if (train and torch.cuda.is_available()) else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[note] cuda requested but unavailable -- falling back to cpu")
        device = "cpu"
    if not train:
        # C7: the self-check must leave the PROCESS-GLOBAL torch RNG exactly as it found it.
        # run_selfchecks drives part of the suite through runpy in ONE process (and
        # tests/test_selfchecks.py under pytest), so a self-check that reseeds the global stream
        # silently re-rolls every module that runs after it. RED CONSTRUCTED AND MEASURED
        # 2026-08-05: drop the get_rng_state/set_rng_state pair around `_self_check_body` and this
        # assert fires. It is asserted here rather than inside _self_check because the property
        # being measured is about the state _self_check leaves behind.
        _before = torch.random.get_rng_state()
        _self_check(device)
        assert torch.equal(torch.random.get_rng_state(), _before), (
            "the self-check moved the process-global torch RNG -- run_selfchecks runs part of the "
            "suite in ONE process, so this re-rolls every module parametrized after it"
        )
        print("  C7 process-global torch RNG unchanged by the self-check (8th criterion; it lives")
        print("     here rather than in the body because it measures what the body LEAVES BEHIND)")
        return
    train_and_eval(args, device)


if __name__ == "__main__":
    main()
