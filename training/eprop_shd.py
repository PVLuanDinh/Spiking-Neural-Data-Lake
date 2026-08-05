"""
eprop_shd -- BRAIN-NATIVE (e-prop) spiking SHD event-AUDIO classifier / encoder for the Spiking
Neural Data Lake. This is the AUDITORY counterpart of the N-MNIST vision encoder (eprop_nmnist):
same perception-tier ROLE (an event stream -> latent + class) and the SAME learning rule (e-prop),
but on a second real sensory modality -- SHD (Spiking Heidelberg Digits, 20-class spoken digits, 700
cochlear event channels). It gives the brain a brain-native SENSE OF HEARING alongside vision.

NO SELF-PATH IN THIS DOCSTRING, deliberately, and not even as an example. This file is being
promoted out of the ungoverned `training/` tier into the governed package, and a literal
`training/<this file>` written into a Usage line would go stale on the move while every gate stayed
GREEN: `tools/citation_guard.py` matches any `dir/name.py`-shaped token, but only its STRICT scope
(.md runnable lines) asserts -- a stale self-path in a .py docstring is report-only, so nothing
fails. So the module's own location is DERIVED at runtime instead -- see `_self_ref()` -- and every
run prints the invocation that is actually true for wherever the file currently sits. That is also
why the prose below never spells the old path out: doing so would re-introduce the stale token this
paragraph exists to remove.

WHAT THIS IS (honest framing). A REAL recurrent spiking network -- a 700-channel event-audio frame per
timestep -> input weights -> a RECURRENT spiking (LIF) hidden layer -> a 20-class leaky readout -- trained
by e-prop (Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, Nature Communications 11:3625,
2020), NOT by backpropagation-through-time. The hidden-weight gradient is computed ONLINE, forward-in-time,
from a per-synapse ELIGIBILITY TRACE (filtered pre-synaptic activity gated by the post-synaptic surrogate
pseudo-derivative) times a BROADCAST learning signal (the readout error routed back through the readout
weights). No BPTT history is unrolled or stored -- e-prop's memory is O(neurons + synapses), independent of
the sequence length T. `loss.backward()` is NEVER called for the hidden weights; the factored gradient is
built by hand in torch tensors, vectorized over batch and units. It is the SAME FACTORIZATION as
src/snn_data_lake/reasoning/eprop.py, which is written in stdlib scalars -- that is a readability
property of a scalar single-neuron reference, NOT a purity requirement (P7: the zero-dependency rule
was abolished 2026-07-20). It is not the same LOSS: eprop.py regresses (err = y - target) where this
classifies (err = softmax(logits) - onehot). Any accuracy printed is the number
ACTUALLY obtained at the small scale it was run at; nothing here is extrapolated to full-dataset numbers,
and where e-prop trails BPTT it is said so. SHD is HARD: the repo's backprop spiking-SSM reached only ~32%
(20-class, chance 5%) on a capped run, so a small-scale e-prop run may sit well below that -- the REAL
number is reported, never faked.

NON-RIGGABLE GOLD STANDARD (self-check). --smoke computes the e-prop hidden-weight gradient AND an
INDEPENDENT reverse-mode BPTT gradient of the SAME loss through the SAME unrolled recurrent net using
the SAME surrogate pseudo-derivative -- and the two share NO gradient code (the BPTT reference is torch
autograd flowing through a differentiable rebuild of the forward, with the surrogate injected in a custom
autograd Function; the e-prop path is a hand-written eligibility loop under no_grad). We assert their
COSINE SIMILARITY over the hidden weights is clearly positive (> 0.5). It is NOT 1.0 -- e-prop truncates
the recurrent-error and reset paths BPTT keeps -- but a positive cosine proves the eligibility trace really
IS the factored gradient (not backprop mislabeled). --smoke also asserts a tiny CPU e-prop train REDUCES
the loss and beats its OWN UNTRAINED TWIN (the identical weights with zero updates applied), on
audio-shaped synthetic data (no dataset needed). It used to assert "accuracy above chance + 15 points"
instead; that criterion was DELETED under P8 because it has no reachable red -- the UNTRAINED net
already scores 73.33% against a 48.33% threshold, so it passed while measuring the task's
separability rather than the learning. A SECOND criterion went the same way on 2026-08-04:
`cos_readout > 0.98` is DELETED, not weakened, because the readout gradient IS the reverse-mode
gradient and no INPUT separates them -- swept over 21 configurations (T, readout scale, and
training depth to loss 0.02) the worst relative error was 8.9e-07. It is still printed, as a
report. The reachable-red probe behind every surviving assert is recorded at the assert block in
`run_smoke`.

IMPORT CONTRACT (this is what makes the module safe to register in TORCH_TIER). Importing it costs
exactly `torch` and has NO side effects: no data loading, no training, no global RNG mutation, no
network or disk access. `tonic` and `torch.utils.data` are imported LAZILY inside the SHD loader, so
this module and EpropAudioEncoder stay importable with torch alone -- which is required, because
`brain_atlas` walks TORCH_TIER and imports every member whenever plain torch imports, in a CI job
that has neither tonic nor the dataset. It imports nothing from the safety envelope (brainstem/).
Runs on the RTX 5070 under .venv-cu128; GPU kernels are not bit-exact (expected for the torch tier).

  # (1) self-check -- tiny synthetic: e-prop-vs-BPTT cosine gold standard + a tiny e-prop train.
  #     CPU-ok, no dataset, no tonic. This is what NO ARGUMENTS runs, which is what makes the module
  #     safe for run_selfchecks (it invokes every module with an empty argv).
  #         python <this module>            # or:  python <this module> --smoke
  # (2) the real SHD train -- needs tonic + data/SHD/*.h5, and must be asked for explicitly:
  #         python <this module> --epochs 30 --hidden 128 --device cuda --save models/eprop_shd.pt
  # The exact, correct invocation for this file's CURRENT location is printed by every run.

ASCII-only output (Windows cp1252-safe): promotion puts this self-check in-process under pytest, and
Windows cp1252 raises on a non-ASCII print. Seeding is PER-INSTANCE (torch.Generator), never the
process-global torch RNG -- see `train_and_eval`. GPU is not bit-exact.
"""

import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


def _self_ref():
    """This module's own identity, DERIVED -- never a hardcoded self-path.

    The promotion moves this file into the governed package. A literal "training/eprop_shd.py" in a
    print or a Usage line would then be a stale path that no gate catches (citation_guard's
    asserting STRICT scope is .md runnable lines; .py citations are broad-scope, report-only), so
    the move would break the documentation silently. Deriving from __spec__/__file__ cannot go
    stale: under `-m` this is the dotted name, otherwise the path the interpreter actually loaded.
    """
    dotted = getattr(__spec__, "name", None) if __spec__ is not None else None
    if dotted and dotted != "__main__":
        return "-m " + dotted
    try:
        return os.path.relpath(os.path.abspath(__file__)).replace(os.sep, "/")
    except ValueError:  # different drive on Windows -> no relative path exists
        return os.path.abspath(__file__).replace(os.sep, "/")


# SHD event-audio geometry (700 cochlear channels, no C/H/W split; matches tonic SHD sensor_size (700,1,1)).
CHANNELS = 700  # input event-audio channels per timestep (SHD)
NUM_CLASSES = 20  # SHD spoken-digit classes
DEFAULT_T = 100  # event time-bins per sample (audio resolution; e-prop memory is O(1) in T)
DEFAULT_HIDDEN = 128  # recurrent LIF width == latent dim the encoder emits (700-dim audio input is
# wider than the 64-latent vision default, so pick a wider latent)

# ---- LIF / e-prop dynamics constants (the PROVEN values from reasoning/eprop.py + eprop_nmnist.py) ----
#
# P1 DISPOSITION for every constant in this file, stated once, here, because promotion carries them
# across the tier boundary and P1 forbids doing that silently.
#
# (i) DEFAULTS OF A LEARNED MODEL, not arbiters of behaviour. ALPHA/V_TH/GAMMA/DAMP/KAPPA are the LIF
#     time constants and the surrogate shape -- the model's own hyperparameters, the direct analogue
#     of a layer width. They do not decide an outcome; they define the dynamical system whose WEIGHTS
#     are then learned by e-prop, which is exactly the brain-native form P1 asks for. Every one is an
#     `__init__` argument, so a caller (or a checkpoint `config`) overrides them without editing code,
#     and `EpropAudioEncoder.load` restores whatever the trained net used. P1's own reuse anchors are
#     learning rules with fixed structural parameters (`r_stdp.py`: `w += lr*d*e`) -- same shape.
# (ii) g_in / g_rec / g_out (EpropAudioNet.__init__) are weight-INIT scales, i.e. the prior on the
#     weights the learning then moves. Also constructor arguments. Same disposition as (i).
# (iii) density / noise in `_make_synthetic` are the generator of a SYNTHETIC TEST FIXTURE. They
#     arbitrate nothing about the brain; they define the self-check's input. Function arguments.
# (iv) THRESHOLDS THAT ARBITRATE A GATE VERDICT -- the only ones P1 actually bites on -- are the
#     self-check asserts in `run_smoke`. They are dispositioned individually at their assert, and
#     THREE have now been DELETED rather than weakened, each for having no reachable red:
#     `final_acc > chance + 15.0` and `cos_readout > 0.98` (both 2026-08-04), and
#     `final_acc > chance` (2026-08-05, whose recorded red was re-run and did not reproduce).
#     See the block at the end of `run_smoke` for the measurement behind each.
DEFAULT_ALPHA = 0.90  # membrane decay exp(-dt/tau_m): the diagonal of the recurrent Jacobian
DEFAULT_V_TH = 0.60  # spike threshold (soft reset subtracts V_TH)
DEFAULT_GAMMA = 0.50  # pseudo-derivative half-width: psi = DAMP*max(0, 1 - |v-V_TH|/GAMMA)
DEFAULT_DAMP = 0.30  # pseudo-derivative dampening factor
DEFAULT_KAPPA = 0.90  # leaky-readout decay (also the eligibility-trace output filter)


# ----------------------------------------------------------------------------
# The recurrent spiking network: audio frame -> input weights -> recurrent LIF hidden -> leaky readout.
# Weights mirror eprop.py's Win / Wrec / Wout (Wrec has NO self-loop: zero diagonal, kept zero). Only the
# INPUT geometry differs from eprop_nmnist.py: the audio frame is already a flat [channels] vector per
# timestep (no C/H/W to flatten), so n_in == channels == 700.
# nn.Module purely to hold parameters + a serving forward; TRAINING gradients are computed by hand
# (eprop_grads), NOT by autograd -- see the module docstring.
# ----------------------------------------------------------------------------
class EpropAudioNet(nn.Module):
    def __init__(
        self,
        channels=CHANNELS,
        num_classes=NUM_CLASSES,
        hidden=DEFAULT_HIDDEN,
        alpha=DEFAULT_ALPHA,
        v_th=DEFAULT_V_TH,
        gamma=DEFAULT_GAMMA,
        damp=DEFAULT_DAMP,
        kappa=DEFAULT_KAPPA,
        seed=0,
        g_in=4.0,
        g_rec=0.6,
        g_out=0.6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.alpha = float(alpha)
        self.v_th = float(v_th)
        self.gamma = float(gamma)
        self.damp = float(damp)
        self.kappa = float(kappa)
        self.n_in = self.channels  # audio frame is already a flat channel vector (no C*H*W)

        gen = torch.Generator().manual_seed(int(seed))  # per-instance seeded init (deterministic)
        h, n_in, k = self.hidden, self.n_in, self.num_classes
        w_in = (torch.rand(h, n_in, generator=gen) * 2 - 1) * (g_in / (n_in**0.5))
        w_rec = (torch.rand(h, h, generator=gen) * 2 - 1) * (g_rec / (h**0.5))
        w_rec.fill_diagonal_(0.0)  # NO self-loop (eprop.py Wrec convention)
        w_out = (torch.rand(k, h, generator=gen) * 2 - 1) * (g_out / (h**0.5))
        self.W_in = nn.Parameter(w_in)
        self.W_rec = nn.Parameter(w_rec)
        self.W_out = nn.Parameter(w_out)
        self.b_out = nn.Parameter(torch.zeros(k))

    # -- shared helpers (the ONE forward math; the three code paths -- serving, e-prop, BPTT -- each
    #    re-run these equations in their own loop, sharing constants but not gradient logic) --
    def psi(self, v):
        """Surrogate pseudo-derivative of the spike: DAMP * max(0, 1 - |v - V_TH| / GAMMA)."""
        return self.damp * torch.clamp(1.0 - (v - self.v_th).abs() / self.gamma, min=0.0)

    def flatten_frames(self, x):
        """[B, T, C(, ...)] audio frames -> [B, T, n_in] per-timestep input vectors. The SHD frame is
        already a flat channel vector, so this only collapses any trailing singleton dims (e.g. a raw
        tonic [B, T, 700, 1, 1] -> [B, T, 700]); the common [B, T, 700] case is passed through."""
        return x.reshape(x.shape[0], x.shape[1], self.n_in)

    def offdiag_mask(self, device):
        return 1.0 - torch.eye(self.hidden, device=device)

    @torch.no_grad()
    def forward(self, x):
        """Serve: x [B, T, C] audio frames -> (logits_sum [B, num_classes], latent_sum [B, hidden]).
        logits_sum is the time-summed leaky readout (the class scores); latent_sum is the hidden
        spike-COUNT activity over the T window (the encoder latent). Manual LIF recurrence, no autograd,
        no dataset/tonic needed -- torch-only-to-serve."""
        x = self.flatten_frames(x.float())  # [B, T, n_in]
        b, t = x.shape[0], x.shape[1]
        dev = x.device
        v = torch.zeros(b, self.hidden, device=dev)
        z = torch.zeros(b, self.hidden, device=dev)
        y = torch.zeros(b, self.num_classes, device=dev)  # leaky readout integrator
        logits_sum = torch.zeros(b, self.num_classes, device=dev)
        latent_sum = torch.zeros(b, self.hidden, device=dev)
        w_rec = self.W_rec * self.offdiag_mask(dev)
        for step in range(t):
            cur = x[:, step] @ self.W_in.t() + z @ w_rec.t()
            v = self.alpha * v - self.v_th * z + cur  # soft reset: subtract V_TH on last spike
            z = (v >= self.v_th).float()
            y = self.kappa * y + z @ self.W_out.t()
            logits_sum = logits_sum + (y + self.b_out)
            latent_sum = latent_sum + z
        return logits_sum, latent_sum

    def config(self):
        return dict(
            channels=self.channels,
            num_classes=self.num_classes,
            hidden=self.hidden,
            alpha=self.alpha,
            v_th=self.v_th,
            gamma=self.gamma,
            damp=self.damp,
            kappa=self.kappa,
        )


# ----------------------------------------------------------------------------
# e-prop gradient -- ONLINE, forward-only, hand-built (NO autograd for the hidden weights).
# Factors dLoss/dw EXACTLY as reasoning/eprop.py + eprop_nmnist.py do:
#   eligibility VECTOR   eps = ALPHA*eps + pre_activity            (per pre-synaptic unit, shared over post)
#   filtered TRACE       ebar = KAPPA*ebar + psi_post * eps        (KAPPA is the readout filter)
#   learning signal      L_j(t) = sum_k W_out[k,j] * err_k(t)      (broadcast error through readout weights)
#   hidden update        dW_ji = sum_t L_j(t) * ebar_ji(t)
# Readout uses its EXACT (non-recurrent) gradient: dW_out = sum_t err(t) (x) zbar(t), zbar = KAPPA*zbar + z.
# Loss = per-timestep softmax cross-entropy against the constant label, summed over t and batch (so err(t)
# = softmax(logits(t)) - onehot, the direct analogue of eprop.py's regression err(t) = y(t) - target(t)).
# ----------------------------------------------------------------------------
@torch.no_grad()
def eprop_grads(net, x, y):
    """x [B, T, C], y [B] int labels. Returns (grads, loss, spike_rate) where grads is a dict of
    dLoss/dw for W_in, W_rec, W_out, b_out (the e-prop-factored hidden grads + exact readout grads). No
    per-timestep history is stored: eps/ebar/zbar are updated in place -> O(batch * synapses) memory,
    independent of T.

    P5: `loss_sum` and `spikes` accumulate IN TORCH, on device, with a single `.item()` after the
    loop. They used to be Python floats accumulated via a per-timestep `.item()` -- two host-device
    syncs on EVERY step, which serialises the whole T-step loop against the GPU.

    RE-MEASURED 2026-08-05 on the RTX 5070 (.venv-cu128, torch 2.11.0+cu128) at this module's real
    training shape (B=64, T=100, C=700, H=128, K=20), median of 5 after a warmup, FIXTURE STATED so
    the number can be re-derived rather than inherited -- x = (rand(B,T,C, gen=Generator(3)) < 0.05),
    y = randint(0,K,(B,), gen=Generator(3)), net seed 0:
        CUDA  123.91 ms/batch -> 99.51 ms/batch   1.245x, 24.40 ms saved
        CPU   995.58 ms/batch -> 777.57 ms/batch  1.280x, 218.02 ms saved
    THE PREVIOUS RECORD DID NOT REPRODUCE and is replaced above: it read "84.13 -> 56.34 ms/batch,
    a 1.493x speedup, 27.79 ms saved" on the same card, and stated no fixture, so the gap cannot be
    attributed. The DIRECTION and the reason hold on re-run; the magnitude was 1.2x optimistic.

    What DID reproduce exactly, and is the load-bearing half: the four returned GRADIENTS are
    BIT-IDENTICAL across the change -- re-verified 2026-08-05, torch.equal True for all of
    W_in/W_rec/W_out/b_out, on BOTH CPU and CUDA. Only the two reported diagnostics move, because
    they now sum in float32 rather than float64. At the SMOKE shape that pair re-measures EXACTLY as
    recorded: 21.77506184577942 -> 21.775062561035156 (3.285e-08 relative, re-run 2026-08-05). The
    training-shape pair previously recorded here (203.13387298583984 -> 203.1338671875) is DROPPED
    rather than restated: it is a function of a training fixture that was never written down, so it
    is not re-derivable. On the stated fixture above the same effect measures
    207.7863456726074 -> 207.78634765625. Nothing consumes either value except a print and the
    smoke's own loss-must-fall assert, whose margin is four orders of magnitude larger."""
    x = net.flatten_frames(x.float())  # [B, T, n_in]
    b, t = x.shape[0], x.shape[1]
    dev = x.device
    h, n_in, k = net.hidden, net.n_in, net.num_classes
    onehot = F.one_hot(y.long(), k).float().to(dev)  # [B, K]
    mask = net.offdiag_mask(dev)
    w_rec = net.W_rec * mask

    v = torch.zeros(b, h, device=dev)
    z = torch.zeros(b, h, device=dev)
    yint = torch.zeros(b, k, device=dev)  # leaky readout integrator
    eps_in = torch.zeros(b, n_in, device=dev)  # alpha-filtered input (shared over post j)
    eps_rec = torch.zeros(b, h, device=dev)  # alpha-filtered z_prev (shared over post j)
    ebar_in = torch.zeros(b, h, n_in, device=dev)  # kappa-filtered eligibility trace [B, post, pre]
    ebar_rec = torch.zeros(b, h, h, device=dev)
    zbar = torch.zeros(b, h, device=dev)  # kappa-filtered spikes (exact readout eligibility)

    dW_in = torch.zeros(h, n_in, device=dev)
    dW_rec = torch.zeros(h, h, device=dev)
    dW_out = torch.zeros(k, h, device=dev)
    db_out = torch.zeros(k, device=dev)
    loss_sum = torch.zeros((), device=dev)  # on-device: no per-step host sync (P5)
    spikes = torch.zeros((), device=dev)

    for step in range(t):
        z_prev = z
        xt = x[:, step]
        cur = xt @ net.W_in.t() + z_prev @ w_rec.t()
        v = net.alpha * v - net.v_th * z_prev + cur
        z = (v >= net.v_th).float()
        pj = net.psi(v)  # [B, H]
        spikes = spikes + z.mean()

        yint = net.kappa * yint + z @ net.W_out.t()
        logits = yint + net.b_out  # [B, K]
        err = torch.softmax(logits, dim=1) - onehot  # [B, K] softmax-CE error (per sample)
        loss_sum = loss_sum + F.cross_entropy(logits, y.long(), reduction="sum")

        # eligibility: eps = ALPHA*eps + pre ; ebar = KAPPA*ebar + psi_post * eps  (mirrors eprop.py)
        eps_in = net.alpha * eps_in + xt
        eps_rec = net.alpha * eps_rec + z_prev
        ebar_in = net.kappa * ebar_in + pj.unsqueeze(2) * eps_in.unsqueeze(1)  # [B, H, n_in]
        ebar_rec = net.kappa * ebar_rec + pj.unsqueeze(2) * eps_rec.unsqueeze(1)  # [B, H, H]

        lsig = err @ net.W_out  # broadcast learning signal L_j(t) = [B, H]
        dW_in += torch.einsum("bj,bji->ji", lsig, ebar_in)
        dW_rec += torch.einsum("bj,bjk->jk", lsig, ebar_rec)

        zbar = net.kappa * zbar + z  # readout weights: exact (non-recurrent)
        dW_out += torch.einsum("bk,bj->kj", err, zbar)
        db_out += err.sum(0)

    dW_rec = dW_rec * mask  # no self-loop gradient
    grads = {"W_in": dW_in, "W_rec": dW_rec, "W_out": dW_out, "b_out": db_out}
    # the ONLY host-device sync in this function, and it is outside the loop.
    return grads, float(loss_sum.item()) / max(t, 1), float(spikes.item()) / max(t, 1)


# ----------------------------------------------------------------------------
# Independent reverse-mode BPTT reference (the gold standard). Shares NO gradient code with eprop_grads:
# a differentiable rebuild of the SAME forward with the SAME surrogate (injected via _SpikeFn.backward),
# then torch autograd. Autograd keeps the recurrent off-diagonal and soft-reset (-V_TH*z_prev) paths that
# e-prop truncates -> this is the TRUE gradient the eligibility gradient is measured against.
# ----------------------------------------------------------------------------
class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_th, gamma, damp):
        ctx.save_for_backward(v)
        ctx.v_th, ctx.gamma, ctx.damp = v_th, gamma, damp
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        psi = ctx.damp * torch.clamp(1.0 - (v - ctx.v_th).abs() / ctx.gamma, min=0.0)
        return grad_out * psi, None, None, None  # SAME surrogate as net.psi / eprop.py


def bptt_grads(net, x, y):
    """True reverse-mode BPTT of the SAME summed-per-timestep softmax-CE loss through a differentiable
    rebuild of the SAME unrolled net, via autograd. Returns a dict of grads for W_in, W_rec, W_out,
    b_out. Does NOT touch net.*.grad (uses torch.autograd.grad)."""
    x = net.flatten_frames(x.float())
    b, t = x.shape[0], x.shape[1]
    dev = x.device
    mask = net.offdiag_mask(dev)
    w_rec = net.W_rec * mask
    v = torch.zeros(b, net.hidden, device=dev)
    z = torch.zeros(b, net.hidden, device=dev)
    yint = torch.zeros(b, net.num_classes, device=dev)
    loss = torch.zeros((), device=dev)
    for step in range(t):
        cur = x[:, step] @ net.W_in.t() + z @ w_rec.t()
        v = net.alpha * v - net.v_th * z + cur
        z = _SpikeFn.apply(v, net.v_th, net.gamma, net.damp)
        yint = net.kappa * yint + z @ net.W_out.t()
        logits = yint + net.b_out
        loss = loss + F.cross_entropy(logits, y.long(), reduction="sum")
    g_in, g_rec, g_out, g_b = torch.autograd.grad(loss, (net.W_in, net.W_rec, net.W_out, net.b_out))
    return {"W_in": g_in, "W_rec": g_rec * mask, "W_out": g_out, "b_out": g_b}


def _cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))


# ----------------------------------------------------------------------------
# The perception interface the brain's hook calls -- mirrors training/eprop_nmnist.EpropEncoder, but on
# AUDIO frames ([T, channels], no C/H/W). Lazy: constructing/loading an encoder does NOT train and needs
# only torch + the saved weights.
# ----------------------------------------------------------------------------
class EpropAudioEncoder:
    """Wraps a trained EpropAudioNet as an AUDITORY perception encoder (mirrors EpropEncoder).

    encode(audio_frame) -> latent vector (the recurrent hidden layer's spike-COUNT activity over T):
      * audio_frame [T, C]      -> latent [hidden]        (single sample; brain-hook path)
      * audio_frame [B, T, C]   -> latent [B, hidden]     (batched)
    classify(audio_frame) -> predicted class label(s) in 0..19: [T,C] -> int ; [B,T,C] -> list[int].
    Accepts a numpy array or a torch tensor. C is the 700 SHD channels. Latent values are spike counts
    over the T window (>= 0).

    load(path, device=...) rebuilds the net from the checkpoint's saved config + weights -- no training,
    no dataset needed. Checkpoint format matches eprop_nmnist: {"model": state_dict, "config": cfg}."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        # weights_only=True, stated explicitly rather than inherited from the torch default: this
        # path reaches here from the SNN_EPROP_AUDIO_CKPT environment variable on the BRAIN's
        # construction path (cortex/brain.py, the auditory tier), so a checkpoint is DATA and must
        # never be executable. The sibling training/surrogate_snn.py already pins this and says so;
        # this loader and eprop_nmnist's -- the two the brain prefers FIRST -- did not, which left
        # the hardened path on the fallback and the unhardened one on the default.
        # The flag is a no-op on torch >= 2.6 (verified: torch 2.13.0+cpu already defaults True and
        # models/eprop_shd.pt loads unchanged either way), and it stops the guarantee from silently
        # depending on which torch happens to be installed.
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        net = EpropAudioNet(**cfg)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    def _prep(self, audio_frame, what):
        x = audio_frame
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float().to(self.device)
        # collapse any trailing singleton feature dims into the channel axis (raw tonic frame safety):
        # [T, C, 1, 1] -> [T, C] ; [B, T, C, 1, 1] -> [B, T, C].
        if x.ndim == 4 and x.shape[2:] != torch.Size([]):  # [T, C, *] single -> [T, C]
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 5:  # [B, T, C, *] batched -> [B, T, C]
            x = x.reshape(x.shape[0], x.shape[1], -1)
        single = x.ndim == 2  # [T, C] -> add batch dim
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"{what} expects [T,C] or [B,T,C]; got shape {tuple(x.shape)}")
        return x, single

    @torch.no_grad()
    def encode(self, audio_frame):
        x, single = self._prep(audio_frame, "encode")
        _, latent = self.net(x)
        return latent[0] if single else latent

    @torch.no_grad()
    def classify(self, audio_frame):
        x, single = self._prep(audio_frame, "classify")
        logits, _ = self.net(x)
        pred = logits.argmax(dim=1)
        return int(pred[0]) if single else [int(v) for v in pred]


# ----------------------------------------------------------------------------
# Data -- real SHD via Tonic, imported LAZILY so this module stays torch-only-importable. Mirrors
# reasoning/snn_ssm_shd.load_shd: ToFrame(sensor_size=SHD.sensor_size, n_time_bins=T) -> frames per sample,
# reshaped to [T, channels]. Streamed through a torch DataLoader (no in-memory cap; the full dataset flows).
# ----------------------------------------------------------------------------
def _shd_collate(batch):
    """Stack a list of (frame, label) into (x [B, T, channels], y [B] long). Each tonic frame comes back
    as [T, 700, 1, 1]; reshape to [T, 700]."""
    xs, ys = [], []
    for frame, lab in batch:
        f = torch.as_tensor(frame, dtype=torch.float32)
        f = f.reshape(f.shape[0], -1)  # [T, channels]
        xs.append(f)
        ys.append(int(lab))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def _make_shd_loaders(n_time_bins, batch_size, num_workers, seed=0, data_root=None):
    """Build streaming train/test DataLoaders over the real SHD dataset (data/SHD/*.h5) via Tonic. tonic
    is imported here (lazily) so importing this module never requires tonic or the dataset.

    `seed` seeds the train sampler through a PER-INSTANCE torch.Generator. That generator is the
    reason `train_and_eval` no longer calls `torch.manual_seed`: a shuffling DataLoader is the only
    consumer of the global torch RNG in this file, so passing it its own generator buys back the
    shuffle determinism without mutating process-global state that every other module shares."""
    import tonic  # noqa: PLC0415 -- lazy: keep the module importable without tonic
    from torch.utils.data import DataLoader  # noqa: PLC0415

    sensor_size = tonic.datasets.SHD.sensor_size  # (700, 1, 1)
    transform = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)
    root = data_root or os.environ.get("SNN_SHD_DATA_ROOT") or "./data"
    train_ds = tonic.datasets.SHD(save_to=root, train=True, transform=transform)
    test_ds = tonic.datasets.SHD(save_to=root, train=False, transform=transform)
    gen = torch.Generator().manual_seed(int(seed))  # per-instance, never the global RNG
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_shd_collate,
        drop_last=False,
        generator=gen,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_shd_collate,
        drop_last=False,
    )
    return train_loader, test_loader


def _apply_grads(net, grads, opt):
    """Load the e-prop-computed grads into .grad and take one optimizer step (Adam on the e-prop grads).
    No loss.backward() anywhere -- the hidden grads are the eligibility factorization."""
    opt.zero_grad(set_to_none=True)
    net.W_in.grad = grads["W_in"]
    net.W_rec.grad = grads["W_rec"]
    net.W_out.grad = grads["W_out"]
    net.b_out.grad = grads["b_out"]
    opt.step()
    with torch.no_grad():
        net.W_rec.copy_(net.W_rec * net.offdiag_mask(net.W_rec.device))  # keep the diagonal at zero


def evaluate(net, loader, device, max_samples=None):
    """Test accuracy = argmax of the time-summed leaky readout. Caps at max_samples for quick runs."""
    net.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for x, yb in loader:
            x = x.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            logits, _ = net(x)
            correct += int((logits.argmax(1) == yb).sum().item())
            seen += int(yb.shape[0])
            if max_samples is not None and seen >= max_samples:
                break
    return (100.0 * correct / seen) if seen else 0.0, seen


def train_and_eval(args):
    device = args.device
    # NO torch.manual_seed / torch.cuda.manual_seed_all here, deliberately. Those mutate the
    # PROCESS-GLOBAL torch RNG, and promotion puts this module's self-check in the same process as
    # every other module's (run_selfchecks drives CORE + TORCH_GATED through runpy in ONE process;
    # tests/test_selfchecks.py does the same under pytest), so a global reseed here silently
    # re-rolls the RNG for every module parametrized after it. Nothing in this file needs it:
    # EpropAudioNet seeds a per-instance torch.Generator, _make_synthetic seeds its own two, and the
    # one real global-RNG consumer -- the shuffling train DataLoader -- is now handed an explicit
    # generator by _make_shd_loaders(seed=...). Determinism kept, blast radius removed.
    print(f"run: python {_self_ref()}")
    print(
        f"device={device}  T={args.T}  batch={args.batch_size}  hidden={args.hidden}  "
        f"limit={args.limit}  epochs={args.epochs}  lr={args.lr:.4g}"
    )
    if device == "cuda" and torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    train_loader, test_loader = _make_shd_loaders(
        n_time_bins=args.T, batch_size=args.batch_size, num_workers=args.workers, seed=args.seed
    )
    print(f"train_batches={len(train_loader)}  test_batches={len(test_loader)}")

    net = EpropAudioNet(
        channels=CHANNELS, num_classes=NUM_CLASSES, hidden=args.hidden, seed=args.seed
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"model: audio({CHANNELS}) -> recurrent LIF({args.hidden}) -> leaky readout({NUM_CLASSES})"
        f" | e-prop (no BPTT) | params={n_params}"
    )
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        net.train()
        seen = 0
        running = 0.0
        nb = 0
        for x, yb in train_loader:
            x = x.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()  # tonic labels are int -> int64
            grads, loss, rate = eprop_grads(net, x, yb)  # e-prop factored gradient (no backward)
            _apply_grads(net, grads, opt)
            running += loss
            nb += 1
            seen += int(yb.shape[0])
            if nb % 20 == 0:
                print(
                    f"  epoch {epoch}  batch {nb}  seen {seen}  loss {running / nb:.4f}  "
                    f"spike_rate {rate:.3f}"
                )
            if args.limit is not None and seen >= args.limit:
                break
        acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
        print(
            f"epoch {epoch} done: train_samples={seen}  avg_loss={running / max(nb, 1):.4f}  "
            f"test_acc={acc:.2f}% (on {ntest} test samples)"
        )

    acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
    cap = str(args.limit) if args.limit else "full"
    print("=" * 74)
    print(f"EPROP-SNN (brain-native AUDITORY perception tier, e-prop -- NO BPTT) | device={device}")
    print(
        f"test accuracy {acc:.2f}% on {ntest} SHD test samples "
        f"(T={args.T}, {cap} train cap, {args.epochs} epoch(s))"
    )
    print(
        "HONEST: small-scale foundation number, not SOTA and not extrapolated. SHD is hard (the repo's"
    )
    print(
        "backprop spiking-SSM reached ~32% on a capped run); e-prop may sit below that at small scale."
    )
    print(
        "Hidden weights learned by e-prop eligibility traces + broadcast error (loss.backward NEVER"
    )
    print(
        "called for them); readout uses its exact gradient. TORCH_TIER: this module needs torch, so"
    )
    print("it is import-verified only where torch is present -- NOT because a zero-dependency core")
    print("exists (that rule was abolished 2026-07-20; P7), but because CI must stay GPU-free.")
    print(f"Encoder latent dim = {args.hidden} (recurrent hidden spike-count activity).")
    print("=" * 74)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save({"model": net.state_dict(), "config": net.config()}, args.save)
        print(f"saved encoder checkpoint -> {args.save}")
    return acc


# ----------------------------------------------------------------------------
# Smoke: synthetic class-separable AUDIO frames (no dataset). Proves (b) the e-prop-vs-BPTT gold-standard
# cosine and (a) that a tiny e-prop train reduces loss and beats its OWN untrained twin, then round-trips
# the encoder through the real load(path) interface. CPU-ok. Audio-shaped: frames are [N, T, channels].
# ----------------------------------------------------------------------------
def _make_synthetic(n_per_class, n_classes, t_steps, channels, seed, density=0.28, noise=0.03):
    """Class-separable audio frames [N, T, channels] + labels [N]. Each class has a FIXED channel
    template (SHARED across train/test so the task is generalizable), persistent over time, plus a little
    per-timestep flip noise. The `seed` varies only the noise + sample order, NOT the templates."""
    tgen = torch.Generator().manual_seed(
        7000 + n_classes
    )  # template seed FIXED (shared train<->test)
    templates = (torch.rand(n_classes, channels, generator=tgen) < density).float()
    gen = torch.Generator().manual_seed(int(seed))  # per-split noise + shuffle only
    xs, ys = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            base = templates[c].unsqueeze(0).repeat(t_steps, 1)  # [T, channels] constant pattern
            flip = (torch.rand(t_steps, channels, generator=gen) < noise).float()
            frame = (base + flip - 2 * base * flip).clamp(0, 1)  # XOR-style noise flips
            xs.append(frame)  # [T, channels]
            ys.append(c)
    x = torch.stack(xs)  # [N, T, channels]
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


def run_smoke(device):
    # No torch.manual_seed here: every generator this smoke depends on is per-instance
    # (EpropAudioNet's own, and _make_synthetic's two), so the global call seeded nothing and only
    # re-rolled the process-global RNG for whichever module runs after this one in run_selfchecks.
    # Verified by removal: the smoke's printed numbers are byte-identical without it.
    n_classes, t_steps, channels = 3, 8, 24  # tiny + fast on CPU
    hidden = 12
    x_tr, y_tr = _make_synthetic(10, n_classes, t_steps, channels, seed=1)
    x_te, y_te = _make_synthetic(5, n_classes, t_steps, channels, seed=2)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    print(f"run: python {_self_ref()}")
    print(
        f"smoke: device={device}  synthetic audio [N={x_tr.shape[0]}, T={t_steps}, "
        f"C={channels}] {n_classes} classes  hidden={hidden} (no dataset)"
    )

    net = EpropAudioNet(channels=channels, num_classes=n_classes, hidden=hidden, seed=0).to(device)

    # (b) GOLD STANDARD: e-prop hidden grad vs an INDEPENDENT reverse-mode BPTT grad, SAME loss/net/psi.
    ge, _, rate = eprop_grads(net, x_tr, y_tr)
    gb = bptt_grads(net, x_tr, y_tr)
    mask = net.offdiag_mask(device).bool()
    e_hidden = torch.cat([ge["W_in"].reshape(-1), ge["W_rec"][mask].reshape(-1)])
    b_hidden = torch.cat([gb["W_in"].reshape(-1), gb["W_rec"][mask].reshape(-1)])
    cos_hidden = _cosine(e_hidden, b_hidden)
    cos_readout = _cosine(ge["W_out"], gb["W_out"])
    print(f"  hidden spike_rate at init = {rate:.3f} (must be > 0 for a meaningful gradient)")
    print(
        f"  e-prop vs BPTT cosine (HIDDEN W_in+W_rec) = {cos_hidden:.4f}"
        "   [gold standard, must be > 0.5]"
    )
    # REPORTED, NOT ASSERTED (P8 / falsifier F11). See the criterion block below for the sweep.
    print(
        f"  e-prop vs BPTT cosine (readout W_out) = {cos_readout:.6f}"
        "   [report only -- exact by construction, no reachable red, so NOT a criterion]"
    )

    # (a) tiny e-prop train: loss must fall, and the trained net must beat its OWN UNTRAINED TWIN.
    # init_acc is that twin -- the same weights, same data, zero updates -- and it is the control,
    # not a decoration. See the P1/P8 disposition block below for why "above chance" is not.
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    _, init_loss, _ = eprop_grads(net, x_tr, y_tr)
    init_acc, _ = evaluate(net, [(x_te, y_te)], device)
    for _ in range(80):
        grads, _, _ = eprop_grads(net, x_tr, y_tr)
        _apply_grads(net, grads, opt)
    _, final_loss, _ = eprop_grads(net, x_tr, y_tr)
    final_acc, _ = evaluate(net, [(x_te, y_te)], device)
    chance = 100.0 / n_classes  # DERIVED from the task, not a constant
    print(
        f"  e-prop train: loss {init_loss:.4f} -> {final_loss:.4f} ; "
        f"test acc {init_acc:.1f}% -> {final_acc:.1f}% (untrained twin {init_acc:.1f}%, "
        f"chance {chance:.1f}%)"
    )

    # encoder round-trip via the real load(path) interface (tempfile, then deleted).
    enc_dir = tempfile.mkdtemp()
    ckpt = os.path.join(enc_dir, "smoke_eprop_audio_encoder.pt")
    torch.save({"model": net.state_dict(), "config": net.config()}, ckpt)
    enc = EpropAudioEncoder.load(ckpt, device=device)
    single = enc.encode(x_te[0])  # [T,C] -> [hidden]
    batched = enc.encode(x_te)  # [B,T,C] -> [B,hidden]
    label = enc.classify(x_te[0])
    labels = enc.classify(x_te)
    os.remove(ckpt)
    os.rmdir(enc_dir)
    print(
        f"  encoder.encode([T,C]) -> {tuple(single.shape)} ; "
        f"encode([B,T,C]) -> {tuple(batched.shape)} ; "
        f"classify -> int {label}, list len {len(labels)}"
    )

    # ---- self-check asserts. P8: EVERY criterion below was probed for a REACHABLE RED by
    # constructing the mutation that makes it fail and RUNNING it. Reds RE-MEASURED 2026-08-05 in
    # `.venv-lake`; two entries in the previous version of this block did not survive that re-run
    # and are corrected rather than restated.
    #   rate > 0.0             <- all-zero INPUT frames (an input, not a weight edit)
    #                             -> rate 0.000000. Also reds at zero W_in/W_rec -> 0.000000.
    #   cos_hidden > 0.5       <- alpha=0.99, g_rec=30.0, g_in=0.05, T=96
    #                             -> cosine +0.1648 at a HEALTHY rate of 0.240, with both gradient
    #                             norms finite and nonzero (|e-prop|=1.08e4, |BPTT|=1.24e12).
    #                             Swept into the corner: g_rec 0.6 -> 12 -> 30 -> 50 gives
    #                             +0.9992 -> +0.6293 -> +0.1648 -> +0.4735, so the bar sits on a
    #                             real edge of the CONFIGURATION space.
    #                             CORRECTED 2026-08-05: this line recorded "sign-flip e-prop grad
    #                             -> 0.9992 becomes -0.9992". That reproduces exactly, but it
    #                             mutates the INSTRUMENT, which is the very objection this block
    #                             uses two paragraphs down to DELETE `cos_readout > 0.98`. A block
    #                             that applies a standard to one criterion and not to its neighbour
    #                             is not applying a standard. The sibling training/eprop_nmnist.py
    #                             already carried a configuration red for this same criterion; this
    #                             file did not, and now does.
    #   final_loss < init_loss <- lr=0 -> 21.7751 !< 21.7751 (re-measured, exact)
    #   final_acc > init_acc   <- lr=0 -> init 73.33, final 73.33 (re-measured, exact)
    #   the two shape asserts  <- any width/batch mismatch
    #
    # DELETED 2026-08-05 (P8, falsifier F11 -- deleted, NOT weakened): `final_acc > chance`.
    # Two independent findings, both measured:
    #   (1) ITS RECORDED RED WAS FALSE. This block claimed "constant-class readout -> 33.33 !>
    #       33.33". RUN: zeroing W_out and b_out gives init_acc 33.33 and final_acc 100.00 -- the
    #       assert does NOT fire, and `final_acc > init_acc` one line above fires first under every
    #       mutation that does reach it. A red that does not reproduce is worse than no red,
    #       because it is read as evidence the criterion was checked.
    #   (2) IT IS IMPLIED BY THE ASSERT DIRECTLY ABOVE IT. The smoke is fully deterministic
    #       (net seed 0, fixture seeds 1 and 2), and MEASURED the untrained twin scores
    #       init_acc = 73.33% against chance = 33.33%. So whenever `final_acc > init_acc` passes,
    #       `final_acc > chance` is already true; and when it fails, this line is never reached.
    #       It is an assert restating the line above it -- a defect shape this tree has recorded
    #       before -- and it contributed no bar the untrained control had not already cleared.
    # `chance` is still COMPUTED from n_classes and still PRINTED, as a report. The surviving
    # accuracy criterion is the UNTRAINED-TWIN control, which is the stronger of the two and the
    # one P8 names (item 2, "a control that is not the method itself").
    #
    # DELETED 2026-08-04 (P8, falsifier F11 -- deleted, NOT weakened): `cos_readout > 0.98`.
    # The red recorded for it above ("random readout grad -> 0.1007") mutated the INSTRUMENT, not
    # an INPUT, which is not what P8 item (1) asks for: dW_out = sum_t err(t) (x) zbar(t) IS what
    # reverse-mode autograd computes for a non-recurrent readout, so no input can separate the two.
    # SWEPT for an input red across 21 non-degenerate configurations -- T = 2/8/64/256 x
    # g_out = 0.01/0.6/50/500, plus the trained regime at 1/200/1000/2000 Adam steps (loss
    # 21.7751 -> 0.0229) and a single-class batch.
    # RE-RUN 2026-08-05, all 21, and the CONCLUSION stands while two of the numbers did not:
    #   * worst relative error MEASURED 3.98e-07 (this said 8.9e-07 -- 2.2x high, and the sweep
    #     protocol above is stated in full precisely so the next reader can re-derive rather than
    #     inherit it; the 0.0229 trained-loss endpoint DID reproduce exactly);
    #   * "every cosine printed 1.0000000 to seven places" is not what the sweep prints. MEASURED
    #     range 0.9999999 to 1.0000001 -- float32 round-off either side of 1, which is the same
    #     finding stated honestly.
    # Either way the closest any configuration came to the 0.98 bar was 0.9999999, four orders of
    # magnitude clear of it, so the criterion had no reachable red and its deletion stands.
    # The ONE configuration that goes below it is an ALL-ZERO input, where BOTH gradients are
    # identically zero (MEASURED: both norms 0.000e+00) and `_cosine`'s zero-norm guard returns
    # 0.0 -- a FALSE red on a case where the method is exactly right, and unreachable here anyway
    # because `rate > 0.0` rejects that input first (MEASURED: rate 0.000000 on it). The sibling
    # training/eprop_nmnist.py deleted the same criterion on the same finding; this file had kept
    # it. The number is still MEASURED and PRINTED above -- as a report, not a bar.
    #
    # DELETED (P8, and this is the P1 disposition the promotion owed): `final_acc > chance + 15.0`.
    # It had NO reachable red under the canonical "the learning is broken" mutation. Measured: with
    # lr=0 -- no weight ever updated -- the UNTRAINED net scores 73.33% against a threshold of
    # 48.33%, so the assert passes. It was presented as evidence that e-prop learns and was in fact
    # evidence about the synthetic task's separability: a random 12-unit recurrent projection
    # already separates three fixed channel templates. Across 8 net seeds the untrained accuracy
    # ranges 0.00%-73.33%, so the threshold was also seed-luck. Weakening or re-tuning the 15.0
    # would have kept a criterion that cannot fail for the reason it claims to test.
    # Replaced by ONE criterion that DOES go red and is not a magic constant:
    #   final_acc > init_acc -- an UNTRAINED TWIN control (same weights, same data, zero updates).
    #   This is baseline-relative in P1's sense and is P8's named control type (item 2).
    #   (A second replacement, `final_acc > chance`, was added here and has since been DELETED --
    #   see the block above the asserts for the measurement that removed it.)
    # A permutation null was built and REJECTED, and the honest version of that rejection is
    # narrower than the one this comment used to make. RE-RUN 2026-08-05 with the protocol stated
    # so it is reproducible -- shuffle y_tr with torch.Generator().manual_seed(1000+s) for s in
    # 0..7, net seed 0, same 80 Adam steps at lr 5e-3, scored on the untouched test split:
    #   MEASURED  26.67 20.00 46.67 40.00 13.33 53.33 6.67 53.33  ->  range 6.67%-53.33%.
    # This comment previously recorded "26.67%-80.00% over 8 seeds" and concluded "the null still
    # learns". Neither reproduces: the measured mean is 32.5% against a chance of 33.33%, i.e. the
    # permutation null lands ON chance, which is what a clean null is supposed to do. The seeding
    # protocol behind the 80.00% was never written down, so it cannot be re-derived -- which is the
    # actual defect here and the reason the protocol is spelled out above.
    # The null is still NOT adopted, on the surviving half of the argument, which is structural
    # rather than numeric: shuffling the labels of a balanced 30-sample, 3-class train set leaves
    # ~1/3 of them correct BY CONSTRUCTION, so the "null" is a 1/3-contaminated training signal
    # rather than an absence of one. The untrained twin has no such contamination and is the
    # stronger control, so it is the one that ships.
    assert rate > 0.0, "hidden units must spike at init (else psi=0 and the gradient is undefined)"
    assert cos_hidden > 0.5, (
        f"(b) e-prop gradient must ALIGN with the true BPTT gradient: cosine {cos_hidden:.4f} !> 0.5"
    )
    assert final_loss < init_loss, (
        f"(a) e-prop must reduce loss: {final_loss:.4f} !< {init_loss:.4f}"
    )
    assert final_acc > init_acc, (
        f"(a) the trained net must beat its own UNTRAINED TWIN: {final_acc:.1f}% !> {init_acc:.1f}%"
    )
    assert tuple(single.shape) == (hidden,), "single-sample latent must be [hidden]"
    assert tuple(batched.shape) == (x_te.shape[0], hidden), "batched latent must be [B, hidden]"
    assert isinstance(label, int) and isinstance(labels, list), "classify types"
    print(
        f"SMOKE OK: e-prop~BPTT hidden cosine {cos_hidden:.4f} (>0.5, factored gradient confirmed "
        "vs an INDEPENDENT"
    )
    print(
        f"reverse-mode reference); e-prop LEARNS loss {init_loss:.4f}->{final_loss:.4f} "
        f"(acc {init_acc:.1f}%->{final_acc:.1f}%, ASSERTED to beat its own untrained twin; "
        f"chance {chance:.1f}% is reported, not a bar -- see run_smoke);"
    )
    print(
        f"encoder load/encode/classify path verified (no dataset, {device}). loss.backward NEVER "
        "called for"
    )
    print("the hidden weights -- they are trained purely by the online eligibility factorization.")


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Brain-native (e-prop) recurrent spiking classifier / encoder "
        "on SHD event-AUDIO -- online eligibility traces, NO BPTT."
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny synthetic audio: e-prop-vs-BPTT cosine gold standard + tiny train "
        "(CPU-ok, no dataset). This is ALSO what no arguments runs.",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="run the real SHD train (needs tonic + data/SHD/*.h5). Required for a train "
        "run that passes no other flag, so that bare argv can never start one.",
    )
    p.add_argument("--epochs", type=int, default=1, help="training epochs")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap training samples per epoch (small-subset quick run)",
    )
    p.add_argument(
        "--eval-limit",
        type=int,
        default=2000,
        dest="eval_limit",
        help="cap test samples during eval (quick runs)",
    )
    p.add_argument("--T", type=int, default=DEFAULT_T, help="event time-bins per sample")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument(
        "--hidden", type=int, default=DEFAULT_HIDDEN, help="recurrent LIF width == latent dim"
    )
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=2, help="dataloader workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None, help="path to save the encoder checkpoint")
    # "auto" resolves PER PATH, in main(): cuda-if-available for a --train run (P5 -- the training
    # tier belongs on the card), ALWAYS cpu for the self-check. The previous default was
    # `cuda if torch.cuda.is_available() else cpu` for BOTH, so the no-argument self-check -- the
    # thing run_selfchecks invokes -- ran on non-bit-exact GPU kernels on any machine with a card
    # and on CPU in CI: two different gates wearing one name, and the cosine/loss/accuracy numbers
    # it prints are not comparable across them. The sibling training/eprop_nmnist.py already
    # carries exactly this fix (`--device auto`, resolved in main); this file did not.
    # At the smoke's width of 12 units a GPU buys nothing anyway; `--device cuda` still forces it.
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p


def main():
    # NO ARGUMENTS RUNS THE SELF-CHECK, NOT A DATASET TRAIN. This inversion is what makes the module
    # safe to register: run_selfchecks invokes every module as `python -m snn_data_lake.<region>.<m>`
    # with an EMPTY argv (and its in-process path pins sys.argv = [mod]), so the previous default --
    # fall through to train_and_eval -> tonic + the on-disk SHD dataset -- would have turned the
    # smoke gate into a full training run that fails on any machine without the dataset.
    # len(sys.argv) == 1 is the test for "invoked with no arguments at all", which is exactly the
    # runner's shape and is not expressible as an argparse default.
    args = build_parser().parse_args()
    smoke = args.smoke or len(sys.argv) == 1
    if args.device == "auto":
        args.device = "cuda" if (not smoke and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[note] cuda requested but unavailable -- falling back to cpu")
        args.device = "cpu"
    if smoke:
        run_smoke(args.device)
        return
    train_and_eval(args)


if __name__ == "__main__":
    main()
