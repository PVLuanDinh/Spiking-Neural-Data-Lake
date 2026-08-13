"""
eprop_nmnist -- BRAIN-NATIVE (e-prop) spiking N-MNIST classifier / encoder for the Spiking Neural
Data Lake, and the brain's FIRST-CHOICE vision transducer: `cortex/brain.py` loads `EpropEncoder`
from `models/eprop_nmnist.pt` and sets `_perception_tier = "eprop-snn"` when that checkpoint exists
(measured live 2026-08-04: tier=eprop-snn, encoder loaded, device=cpu). It is the e-prop counterpart
of the RETIRED training/surrogate_snn.py (deleted 2026-08-13, specs/013 -- this module is now the
ONE vision perception encoder): same perception-tier ROLE (event frame -> latent + class), same encoder
interface, but a fundamentally different LEARNING RULE.

WHAT THIS IS (honest framing). A REAL recurrent spiking network -- input frame -> input weights ->
a RECURRENT spiking (LIF) hidden layer -> a 10-class leaky readout -- trained by e-prop (Bellec,
Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, Nature Communications 11:3625, 2020), NOT by
backpropagation-through-time. The hidden-weight gradient is computed ONLINE, forward-in-time, from a
per-synapse ELIGIBILITY TRACE (filtered pre-synaptic activity gated by the post-synaptic surrogate
pseudo-derivative) times a BROADCAST learning signal (the readout error routed back through the
readout weights). No BPTT history is unrolled or stored -- e-prop's memory is O(neurons + synapses),
independent of the sequence length T. `loss.backward()` is NEVER called for the hidden weights; the
factored gradient is built by hand in torch tensors, vectorized over batch and units, EXACTLY the way
src/snn_data_lake/reasoning/eprop.py factors it (that pure-stdlib module is the proven reference).
Any accuracy printed is the number ACTUALLY obtained at the small scale it was run at; nothing here
is extrapolated to full-dataset numbers, and where e-prop trails BPTT it is said so.

NON-RIGGABLE GOLD STANDARD (the self-check, run by BARE invocation -- no arguments). It computes the
e-prop hidden-weight gradient AND an INDEPENDENT reverse-mode BPTT gradient of the SAME loss through
the SAME unrolled recurrent net using the SAME surrogate pseudo-derivative -- and the two share NO
gradient code (the BPTT reference is torch autograd flowing through a differentiable rebuild of the
forward, with the surrogate injected in a custom autograd Function; the e-prop path is a hand-written
eligibility loop under no_grad). We assert their COSINE SIMILARITY over the hidden weights is clearly
positive (> 0.5). It is NOT 1.0 -- e-prop truncates the recurrent-error and reset paths BPTT keeps --
but a positive cosine proves the eligibility trace really IS the factored gradient (not backprop
mislabeled). This mirrors reasoning/eprop.py's grad_eprop-vs-grad_bptt gold standard.

P8 -- EVERY SURVIVING CRITERION HAS A CONSTRUCTED, MEASURED RED. Every number in this block was
RE-RUN 2026-08-05 in .venv-lake on CPU; where the re-run disagreed with what was written here, the
MEASURED value replaced it and the disagreement is recorded rather than quietly corrected.
  * `rate > 0`          RED at g_in=0.0 -> rate = 0.0 exactly (re-measured 2026-08-05: 0.000000).
  * `cos_hidden > 0.5`  RED at alpha=0.99, g_rec=100, g_in=0.05, T=96 -> cosine -0.2930 at a HEALTHY
                        rate of 0.332 (not a dead-network artifact; both gradient norms finite and
                        nonzero, |e-prop| 1.95e4, |BPTT| 4.46e18). Strong recurrence + weak drive +
                        long T is exactly the regime where the recurrent-error path e-prop truncates
                        dominates the one it keeps.
                        RE-MEASURED, AND THE RECORDED PAIR DID NOT REPRODUCE: this said "-0.2667
                        with a HEALTHY rate of 0.44". Neither number came back -- the configuration
                        as written gives -0.2930 / 0.332, and running it with the self-check's own
                        non-default dynamics (v_th .55 / gamma .44 / damp .27 / kappa .86) gives
                        -0.3972 / 0.348, so no reading of the recorded configuration produces
                        -0.2667 / 0.44. The RED IS REAL AND REACHABLE either way, which is what the
                        criterion needs; the numbers beside it were not reproducible and are now
                        stated with the exact configuration that produces them.
                        The monotone sweep this block claimed ("0.9992 -> 0.9017 -> 0.8038 ->
                        -0.2667") is DROPPED rather than restated: its intermediate configurations
                        were never written down, so it cannot be re-derived, and an unreproducible
                        sweep is not evidence that the bar sits on an edge.
  * accuracy bar        RED is measured INSIDE the run: the untrained twin (init_acc) is the control
                        and it FAILS the same bar the trained net must pass; both are asserted.
  * round-trip (x3)     RED by dropping ONE dynamics field from config(): shapes still match and
                        load_state_dict still passes, but the reload is a different network.
                        MEASURED 2026-08-05, all five fields dropped one at a time:
                          drop alpha -> cfg differs, latents differ, logits differ
                          drop v_th  -> cfg differs, latents differ, logits differ
                          drop kappa -> cfg differs, latents BIT-IDENTICAL, logits differ
                          drop gamma -> cfg differs, latents and logits BOTH BIT-IDENTICAL
                          drop damp  -> cfg differs, latents and logits BOTH BIT-IDENTICAL
                        so three probes (config / latent / logit) are each load-bearing and no two
                        of them cover all five fields. That is also why the self-check net is built
                        with NON-DEFAULT alpha/v_th/gamma/damp/kappa: with module defaults a lost
                        field reloads to the same value and every probe goes vacuous.
                        THIS SAID "x4". The fourth probe, `class_exact`, is DELETED -- see below.
  DELETED, not weakened: `cos_readout > 0.98`. The readout gradient is the EXACT reverse-mode
  gradient (dW_out = sum_t err(t) (x) zbar(t) is what autograd computes, not an approximation of it),
  so no input can separate them. Searched for a red across T = 8/64/256 x g_out = 1/50/500 and the
  trained regime at 2000 Adam steps. RE-RUN 2026-08-05: relative error stayed in [4.3e-8, 4.6e-7],
  i.e. float32 round-off, never within four orders of magnitude of the 0.98 bar (the upper end read
  3.8e-7 here and measures 4.6e-7; the trained endpoint read "loss 36.5 -> 0.0036" and measures
  36.4759 -> 0.0741, which is 20x off and is corrected because the 0.0036 cannot be reproduced by
  running the sweep this block describes). A criterion with no reachable red is a gate that cannot
  fail; per P8 it goes. The number is still MEASURED and PRINTED -- as a report, not a bar.
  DELETED 2026-08-05, not weakened: `class_exact` (the fourth round-trip probe). Two measurements
  killed it. (1) Its stated rationale -- "a logit shift too small to flip an argmax is caught by the
  logit probe; this one says the CONSUMED label survived" -- has the implication backwards: class
  predictions are the ARGMAX OF THE LOGITS, so `logit_exact` (asserted immediately above it) implies
  `class_exact` and never the reverse. (2) Run against the mutation family it documents, all five
  field-drops leave the class predictions IDENTICAL, including the two (alpha, v_th) that move both
  the latents and the logits. So no member of its own documented red family reddens it, and no
  member ever can while the assert above it holds. The label is still COMPUTED and PRINTED in the
  round-trip line -- as a report, not a bar.

DEPENDENCIES (P7, zero-dependency rule abolished 2026-07-20). torch is an unconditional core
dependency and this module uses it as the core tier, not as an apologetic extra; the previous header
("OPTIONAL torch [train]/[gpu] tier", "the zero-dep core NEVER imports this", "Imports NOTHING from
src/snn_data_lake/") was retired-rule residue and is now also FALSE -- cortex/brain.py imports this
module on the live construction path. Nothing here was hand-rolled to avoid a dependency: the LIF /
e-prop loop is already vectorized torch over batch and units, and reasoning/eprop.py stays the
stdlib SCALAR reference it is correct to be (importing it here would buy nothing -- different tier,
different width). GPU kernels are not bit-exact; the self-check therefore runs on CPU (P5: the suite
must stay runnable without a card, and at the self-check's width of 12 units a GPU buys nothing).

SELF-PATH INDEPENDENCE (promotion). This module hardcodes NO path of its own. The only external file
it reaches for is research/ng_snn/data.py, located by walking UP from __file__ until it is found
(`_ng_snn_dir`), and the module actually bound is verified to BE that file -- so relocating this file
between tiers cannot silently repoint or mis-bind it. That reach happens only on --train.

  <no arguments>                       SELF-CHECK: tiny synthetic, CPU, no dataset, no tonic.
  --train --epochs 1 --limit 4000 --save models/eprop_nmnist.pt
                                       short real N-MNIST e-prop train -> test accuracy (needs tonic
                                       + the dataset). Add --device cuda for the RTX 5070.
  e.g.  PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.cortex.eprop_nmnist
  (This line has now come TRUE twice over: it read `python -m snn_data_lake.cortex.eprop_nmnist`
  until 2026-08-04, when that module did not exist and the line was corrected to the training/
  path with a note that the promotion it anticipated had not happened. On 2026-08-12 the PROMOTE
  lane executed exactly that promotion -- this file IS `cortex/eprop_nmnist.py`, registered in the
  TORCH_TIER annex (torch at module top) and run_selfchecks TORCH_GATED. RUN, not read, both
  times: the pre-promotion exit-1 was executed then, and the post-promotion run is the gate now.)

ASCII-only output (Windows cp1252-safe -- the promoted self-check runs in-process under pytest).
Every RNG is per-instance seeded; this module sets no global torch seed (verified not load-bearing:
the self-check is byte-identical under global seeds 0 and 999).
"""

import argparse
import importlib
import math
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.cortex.eprop_kernel import (  # the ONE e-prop engine (specs/013 C4)
    DEFAULT_ALPHA,
    DEFAULT_DAMP,
    DEFAULT_GAMMA,
    DEFAULT_KAPPA,
    DEFAULT_V_TH,
    EPROP_BPTT_COSINE_FLOOR,
    EpropEncoderBase,
    EpropLIFBase,
    _apply_grads,
    _cosine,
    bptt_grads,
    evaluate,
)

# N-MNIST event geometry (matches research/ng_snn/data.py SENSOR = (34, 34, 2)).
SENSOR_H = 34
SENSOR_W = 34
IN_CH = 2
NUM_CLASSES = 10
DEFAULT_T = 15  # event time-bins; matches data.py DEFAULT_T
DEFAULT_HIDDEN = 64  # recurrent LIF width == latent dim the encoder emits (matches existing wiring)

# ---- LIF / e-prop dynamics constants (the PROVEN values from reasoning/eprop.py) ----
# P1 LABEL -- documented exception (a), "a fixed point on a NORMALIZED output". All five are
# DIMENSIONLESS constants of a normalized dynamical system: alpha/kappa are decay factors
# exp(-dt/tau) in [0,1), damp is a gain, and v_th/gamma are levels on a membrane whose input current
# is normalized to O(1) by the 1/sqrt(fan_in) weight scaling below. None is a decision threshold on a
# raw measured signal, which is what P1's "never keep a constant fixed" is aimed at.
# They are also NOT frozen at these values in practice: they are per-INSTANCE constructor arguments
# that `config()` PERSISTS into every checkpoint and `EpropEncoder.load` restores, so a trained
# encoder carries the values it was trained under rather than inheriting whatever this module's
# defaults happen to be today. The self-check builds its net with values DIFFERENT from all five, so
# the checkpoint round-trip criterion has a reachable red (see the P8 block in the module docstring).

# The project's e-prop invariant, not a tunable of this module: "verified cosine > 0.5 against an
# INDEPENDENT BPTT gradient" is the cross-cutting rule every learning path in this tree owes
# (docs/POLICIES.md). Named here so the self-check reads the bar from the policy, not from a literal.

# Significance level for the accuracy bar: how many standard deviations above chance a result must
# sit before it counts as learning. Dimensionless (P1 exception (a)) -- the BAR ITSELF is not a
# constant, it is recomputed from the actual test-set size and class count at run time, so it moves
# with the data instead of staying pinned. See `_chance_bound`.
CHANCE_SIGMAS = 3.0


# ----------------------------------------------------------------------------
# The recurrent spiking network: input frame -> input weights -> recurrent LIF hidden -> leaky readout.
# Weights mirror eprop.py's Win / Wrec / Wout (Wrec has NO self-loop: zero diagonal, kept zero).
# nn.Module purely to hold parameters + a serving forward; TRAINING gradients are computed by hand
# (eprop_grads), NOT by autograd -- see the module docstring.
# ----------------------------------------------------------------------------
class EpropNet(EpropLIFBase):
    def __init__(
        self,
        in_ch=IN_CH,
        num_classes=NUM_CLASSES,
        hidden=DEFAULT_HIDDEN,
        sensor_h=SENSOR_H,
        sensor_w=SENSOR_W,
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
        self.in_ch = int(in_ch)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.sensor_h = int(sensor_h)
        self.sensor_w = int(sensor_w)
        self.alpha = float(alpha)
        self.v_th = float(v_th)
        self.gamma = float(gamma)
        self.damp = float(damp)
        self.kappa = float(kappa)
        self.n_in = self.in_ch * self.sensor_h * self.sensor_w

        # P1 LABEL for g_in / g_rec / g_out: INITIALIZATION SCALE ONLY. They arbitrate nothing after
        # step 0 -- every weight they set is immediately overwritten by learning (train) or by
        # load_state_dict (serve), and they are deliberately NOT in config() because a loaded
        # checkpoint has no use for them. They stay arguments rather than literals so the self-check
        # can construct the g_in=0.0 dead-network RED for `rate > 0`.
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

    def config(self):
        return dict(
            in_ch=self.in_ch,
            num_classes=self.num_classes,
            hidden=self.hidden,
            sensor_h=self.sensor_h,
            sensor_w=self.sensor_w,
            alpha=self.alpha,
            v_th=self.v_th,
            gamma=self.gamma,
            damp=self.damp,
            kappa=self.kappa,
        )


# ----------------------------------------------------------------------------
# e-prop gradient -- ONLINE, forward-only, hand-built (NO autograd for the hidden weights).
# Factors dLoss/dw EXACTLY as reasoning/eprop.py does:
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
    """x [B, T, C, H, W], y [B] int labels. Returns (grads, loss, spike_rate) where grads is a dict of
    dLoss/dw for W_in, W_rec, W_out, b_out (the e-prop-factored hidden grads + exact readout grads). No
    per-timestep history is stored: eps/ebar/zbar are updated in place -> O(batch * synapses) memory,
    independent of T."""
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
    loss_sum = 0.0
    spikes = 0.0

    for step in range(t):
        z_prev = z
        xt = x[:, step]
        cur = xt @ net.W_in.t() + z_prev @ w_rec.t()
        v = net.alpha * v - net.v_th * z_prev + cur
        z = (v >= net.v_th).float()
        pj = net.psi(v)  # [B, H]
        spikes += float(z.mean().item())

        yint = net.kappa * yint + z @ net.W_out.t()
        logits = yint + net.b_out  # [B, K]
        err = torch.softmax(logits, dim=1) - onehot  # [B, K] softmax-CE error (per sample)
        loss_sum += float(F.cross_entropy(logits, y.long(), reduction="sum").item())

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
    return grads, loss_sum / max(t, 1), spikes / max(t, 1)


# ----------------------------------------------------------------------------
# Independent reverse-mode BPTT reference (the gold standard). Shares NO gradient code with eprop_grads:
# a differentiable rebuild of the SAME forward with the SAME surrogate (injected via _SpikeFn.backward),
# then torch autograd. Autograd keeps the recurrent off-diagonal and soft-reset (-V_TH*z_prev) paths that
# e-prop truncates -> this is the TRUE gradient the eligibility gradient is measured against.
# ----------------------------------------------------------------------------


def _chance_bound(n_classes, n_test, sigmas=CHANCE_SIGMAS):
    """Accuracy (%) an unbiased chance classifier exceeds only `sigmas` standard deviations out.

    P1: this REPLACES a hardcoded `chance + 15.0` margin. The bar is DERIVED from the two things that
    actually determine how noisy the estimate is -- the class count and how many test samples were
    scored -- so it tightens automatically as the evaluation grows instead of staying pinned at a
    number chosen once. At the self-check's 3 classes x 15 test samples it computes 69.85%, which is
    STRICTER than the 48.33% the old literal produced, and the untrained control sits at 33.3%."""
    p = 1.0 / float(n_classes)
    return 100.0 * (p + sigmas * math.sqrt(p * (1.0 - p) / max(int(n_test), 1)))


# ----------------------------------------------------------------------------
# The perception interface the brain's hook calls. It once mirrored SurrogateEncoder; that module
# was retired 2026-08-13 (specs/013) and this is now the only implementation of the interface.
# Lazy: constructing/loading an encoder does NOT train and needs only torch + the saved weights.
# ----------------------------------------------------------------------------
class EpropEncoder(EpropEncoderBase):
    """Wraps a trained EpropNet as a perception encoder (`load` / `encode` / `classify`).

    encode(event_frame) -> latent vector (the recurrent hidden layer's spike-COUNT activity over T):
      * event_frame [T, C, H, W]      -> latent [hidden]        (single sample; brain-hook path)
      * event_frame [B, T, C, H, W]   -> latent [B, hidden]     (batched)
    classify(event_frame) -> predicted class label(s): [T,C,H,W] -> int ; [B,T,C,H,W] -> list[int].
    Accepts a numpy array or a torch tensor. Latent values are spike counts over the T window (>= 0).

    load(path, device=...) rebuilds the net from the checkpoint's saved config + weights -- no training,
    no dataset needed. Checkpoint format: {"model": state_dict, "config": cfg}."""

    @classmethod
    def load(cls, path, device="cpu"):
        # weights_only=True, stated explicitly rather than inherited from the torch default: this
        # path reaches here from the SNN_EPROP_CKPT environment variable on the BRAIN's construction
        # path, so a checkpoint is DATA and must never be executable. The now-retired surrogate_snn --
        # the FALLBACK tier -- already pins this and explains why; this loader, which the brain
        # prefers FIRST, did not, so the hardened loader sat on the path the brain takes second.
        # The flag is a no-op on torch >= 2.6 (verified: torch 2.13.0+cpu already defaults True and
        # models/eprop_nmnist.pt loads bit-identically either way -- config equal, all 4 weight
        # tensors torch.equal), and it stops the guarantee from silently depending on which torch
        # version happens to be installed.
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        net = EpropNet(**cfg)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    def _prep(self, event_frame, what):
        x = event_frame
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float().to(self.device)
        single = x.ndim == 4  # [T, C, H, W] -> add batch dim
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"{what} expects [T,C,H,W] or [B,T,C,H,W]; got shape {tuple(x.shape)}")
        return x, single

    @torch.no_grad()
    def encode(self, event_frame):
        x, single = self._prep(event_frame, "encode")
        _, latent = self.net(x)
        return latent[0] if single else latent

    @torch.no_grad()
    def classify(self, event_frame):
        x, single = self._prep(event_frame, "classify")
        logits, _ = self.net(x)
        pred = logits.argmax(dim=1)
        return int(pred[0]) if single else [int(v) for v in pred]


# ----------------------------------------------------------------------------
# Data (reused from research/ng_snn/data.py, imported lazily so this module -- and EpropEncoder, the
# surface the brain consumes -- stay importable with no tonic and no dataset on disk).
# ----------------------------------------------------------------------------
def _ng_snn_dir():
    """Locate research/ng_snn/ by walking UP from THIS FILE until data.py is found.

    MOVE-SAFE BY CONSTRUCTION. The previous form was one fixed hop --
    `join(dirname(__file__), "..", "research", "ng_snn")` -- which is correct only while this file
    sits in training/. Relocating it to src/snn_data_lake/<region>/ made that resolve to
    src/snn_data_lake/research/ng_snn, a directory that does not exist; `sys.path.insert` of a
    non-existent directory raises NOTHING, and the bare `import data` that followed then binds by
    NAME, so any other importable top-level `data` module silently takes its place. A stale path that
    FAILS GREEN (backlog M37). The walk has no hop count to get wrong, and the caller verifies the
    module it actually got is the file resolved here."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, "research", "ng_snn", "data.py")
        if os.path.isfile(cand):
            return os.path.dirname(cand)
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError(
                "research/ng_snn/data.py (the disk-streaming N-MNIST loader) was not found in any "
                f"ancestor of {os.path.abspath(__file__)}. --train needs the repo checkout; the self-check does not."
            )
        d = parent


def _load_data_module():
    """research/ng_snn/data.py -- disk-streaming N-MNIST loader (tonic). --train ONLY.

    sys.path is still extended (NOT replaced by a spec_from_file_location import) because the
    DataLoader spawns worker processes on Windows and those children re-import the module by NAME to
    unpickle data.ToFloatTensor -- a private file-path import would break num_workers > 0. What is
    fixed is the silent part: the directory is derived by the upward walk, and the module that
    actually binds is asserted to BE the file that walk resolved, so a wrong `data` on the path is a
    loud failure instead of a wrong dataset. Not reached by the self-check, so the promoted
    in-process pytest run never mutates sys.path."""
    ng = _ng_snn_dir()
    if ng not in sys.path:
        sys.path.insert(0, ng)
    nmnist_data = importlib.import_module("data")
    got = os.path.abspath(getattr(nmnist_data, "__file__", "") or "")
    want = os.path.join(ng, "data.py")
    if got != os.path.abspath(want):
        raise ImportError(
            f"top-level module 'data' resolved to {got!r}, not the N-MNIST loader at {want!r} -- refusing to "
            "train against an unknown dataset module"
        )
    return nmnist_data


def train_and_eval(args):
    device = args.device
    torch.manual_seed(args.seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = _load_data_module()
    print(
        f"device={device}  T={args.T}  batch={args.batch_size}  hidden={args.hidden}  "
        f"limit={args.limit}  epochs={args.epochs}  lr={args.lr:.4g}"
    )
    if device == "cuda" and torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    train_loader, test_loader = data.make_nmnist_loaders(
        n_time_bins=args.T, batch_size=args.batch_size, num_workers=args.workers
    )
    print(f"train_batches={len(train_loader)}  test_batches={len(test_loader)}")

    net = EpropNet(hidden=args.hidden, seed=args.seed).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"model: input -> recurrent LIF({args.hidden}) -> leaky readout({NUM_CLASSES}) | "
        f"e-prop (no BPTT) | params={n_params}"
    )
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        net.train()
        seen = 0
        running = 0.0
        nb = 0
        for x, yb in train_loader:
            x = x.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()  # tonic labels are int32 -> int64
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
    print("=" * 74)
    print(f"EPROP-SNN (brain-native perception tier, e-prop -- NO BPTT) | device={device}")
    n_train = min(args.limit or 10**12, 60000)
    print(
        f"test accuracy {acc:.2f}% on {ntest} N-MNIST test samples "
        f"(T={args.T}, {n_train} train samples, {args.epochs} epoch(s))"
    )
    print(
        "HONEST: small-scale foundation number, not SOTA and not extrapolated. Hidden weights learned"
    )
    print(
        "by e-prop eligibility traces + broadcast error (loss.backward NEVER called for them); readout"
    )
    print(
        "uses its exact gradient. The checkpoint this writes is what cortex/brain.py loads as its"
    )
    print("first-choice vision tier (perception_tier=eprop-snn).")
    print(f"Encoder latent dim = {args.hidden} (recurrent hidden spike-count activity).")
    print("=" * 74)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save({"model": net.state_dict(), "config": net.config()}, args.save)
        print(f"saved encoder checkpoint -> {args.save}")
    return acc


# ----------------------------------------------------------------------------
# Smoke: synthetic class-separable events (no dataset). Proves (b) the e-prop-vs-BPTT gold-standard
# cosine and (a) that a tiny e-prop train reduces loss + raises accuracy above chance, then round-trips
# the encoder through the real load(path) interface. CPU-ok.
# ----------------------------------------------------------------------------
def _make_synthetic(n_per_class, n_classes, t_steps, in_ch, h, w, seed, density=0.28, noise=0.03):
    """Class-separable event frames [N, T, C, H, W] + labels [N]. Each class has a FIXED spatial
    template (SHARED across train/test so the task is generalizable), persistent over time, plus a
    little per-timestep flip noise. The `seed` varies only the noise + sample order, NOT the templates."""
    n_in = in_ch * h * w
    tgen = torch.Generator().manual_seed(
        7000 + n_classes
    )  # template seed FIXED (shared train<->test)
    templates = (torch.rand(n_classes, n_in, generator=tgen) < density).float()
    gen = torch.Generator().manual_seed(int(seed))  # per-split noise + shuffle only
    xs, ys = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            base = templates[c].unsqueeze(0).repeat(t_steps, 1)  # [T, n_in] constant class pattern
            flip = (torch.rand(t_steps, n_in, generator=gen) < noise).float()
            frame = (base + flip - 2 * base * flip).clamp(0, 1)  # XOR-style noise flips
            xs.append(frame.reshape(t_steps, in_ch, h, w))
            ys.append(c)
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


def run_smoke(device):
    # No global torch.manual_seed here: every RNG on this path is per-instance (EpropNet's own
    # Generator, _make_synthetic's two Generators) and Adam is deterministic, so the global seed the
    # old code set was dead -- MEASURED, not assumed: the whole self-check is byte-identical under
    # global seeds 0 and 999. Removing it matters because the promoted self-check runs IN-PROCESS
    # under pytest, where seeding the global torch RNG silently reaches into every other test.
    n_classes, t_steps, h, w = 3, 8, 8, 8  # tiny + fast on CPU
    hidden = 12
    x_tr, y_tr = _make_synthetic(10, n_classes, t_steps, IN_CH, h, w, seed=1)
    x_te, y_te = _make_synthetic(5, n_classes, t_steps, IN_CH, h, w, seed=2)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    print(
        f"smoke: device={device}  synthetic [N={x_tr.shape[0]}, T={t_steps}, C={IN_CH}, {h}x{w}] "
        f"{n_classes} classes  hidden={hidden} (no dataset)"
    )

    # NON-DEFAULT dynamics on purpose. With DEFAULT_* values a config() that silently stopped
    # persisting a field would rebuild the identical net on load and the round-trip criterion below
    # would be vacuous; with these, dropping any one of the five changes the latents. This is what
    # gives that criterion its reachable red (verified: dropping `alpha` -> shapes still match,
    # load_state_dict still passes, latents differ).
    dyn = dict(alpha=0.83, v_th=0.55, gamma=0.44, damp=0.27, kappa=0.86)
    net = EpropNet(
        in_ch=IN_CH, num_classes=n_classes, hidden=hidden, sensor_h=h, sensor_w=w, seed=0, **dyn
    ).to(device)

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
        f"  e-prop vs BPTT cosine (HIDDEN W_in+W_rec) = {cos_hidden:.4f}   [gold standard, must be > {EPROP_BPTT_COSINE_FLOOR:.2f}]"
    )
    # REPORTED, NOT ASSERTED (P8): the readout gradient IS the reverse-mode gradient, so no input can
    # separate them and a bar over it is a gate that cannot fail. Printed because the number is worth
    # seeing; the criterion it used to carry was deleted rather than weakened.
    print(
        f"  e-prop vs BPTT cosine (readout W_out) = {cos_readout:.6f}   [report only -- exact by construction,"
        " no reachable red, so NOT a criterion]"
    )

    # (a) tiny e-prop train: loss must fall, accuracy must rise above chance (= 1/n_classes).
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    _, init_loss, _ = eprop_grads(net, x_tr, y_tr)
    init_acc, _ = evaluate(net, [(x_te, y_te)], device)
    for _ in range(80):
        grads, _, _ = eprop_grads(net, x_tr, y_tr)
        _apply_grads(net, grads, opt)
    _, final_loss, _ = eprop_grads(net, x_tr, y_tr)
    final_acc, _ = evaluate(net, [(x_te, y_te)], device)
    chance = 100.0 / n_classes
    bar = _chance_bound(n_classes, int(x_te.shape[0]))
    print(
        f"  e-prop train: loss {init_loss:.4f} -> {final_loss:.4f} ; test acc {init_acc:.1f}% -> "
        f"{final_acc:.1f}% (chance {chance:.1f}%, bar {bar:.2f}% = chance + {CHANCE_SIGMAS:.0f} "
        f"sigma on {x_te.shape[0]} test samples)"
    )

    # ROUND-TRIP through the real load(path) interface -- the exact surface cortex/brain.py consumes.
    # The criterion is NOT "the shapes look right": nothing reachable can make that fail (a config
    # that loses a SHAPE field is caught earlier and louder by load_state_dict, and int()/[...]
    # cannot return a non-int/non-list). It is "the reloaded encoder IS this net" -- and it takes
    # THREE probes, because MEASURED (2026-08-05, all five dynamics fields dropped one at a time) no
    # two of them cover all five:
    #   config equality  -- the only probe that sees gamma/damp, which forward() never touches (they
    #                       live in psi(), i.e. in TRAINING); losing them silently breaks any further
    #                       training of a reloaded encoder while every output stays identical.
    #                       MEASURED: dropping gamma or damp leaves latents AND logits bit-identical.
    #   latent equality  -- alpha and v_th, the hidden dynamics behind every latent brain.py consumes.
    #   logit equality   -- kappa, which is READOUT-only: it moves the logits and leaves the hidden
    #                       spike counts untouched, so a latent-only check does not see it (MEASURED:
    #                       dropping kappa from config left the latents bit-identical).
    # A FOURTH probe, `classify equality`, stood here and is DELETED (P8 falsifier F11 -- deleted,
    # not weakened). Its rationale ran "a logit shift too small to flip an argmax is caught by the
    # logit probe; this one says the CONSUMED label survived", which has the implication backwards:
    # the class predictions ARE the argmax of the logits, so the logit probe asserted below implies
    # it and it can never fire while that assert holds. MEASURED against its own documented red
    # family: all five field-drops leave the class predictions identical, including alpha and v_th
    # which move the latents and the logits. The label is still computed and PRINTED below.
    # (This comment ALSO said "FOUR probes ... no single one covers all TEN persisted fields", which
    # over-counted: config() persists ten keys but five of them are SHAPE fields that
    # load_state_dict catches first and louder. The five DYNAMICS fields are what these probes are
    # for, and that is the count now stated.)
    # TemporaryDirectory so an assert below cannot leak the checkpoint.
    ref_logits, ref_latent = net(x_te)
    ref_cfg = net.config()
    with tempfile.TemporaryDirectory() as enc_dir:
        ckpt = os.path.join(enc_dir, "smoke_eprop_encoder.pt")
        torch.save({"model": net.state_dict(), "config": ref_cfg}, ckpt)
        enc = EpropEncoder.load(ckpt, device=device)
        single = enc.encode(x_te[0])  # [T,C,H,W] -> [hidden]
        batched = enc.encode(x_te)  # [B,T,C,H,W] -> [B,hidden]
        label = enc.classify(x_te[0])
        labels = enc.classify(x_te)
        enc_logits, _ = enc.net(x_te)
        enc_cfg = enc.net.config()
    ref_pred = [int(v) for v in ref_logits.argmax(dim=1)]
    cfg_exact = enc_cfg == ref_cfg
    latent_exact = bool(torch.equal(batched, ref_latent)) and bool(
        torch.equal(single, ref_latent[0])
    )
    logit_exact = bool(torch.equal(enc_logits, ref_logits))
    class_exact = (labels == ref_pred) and (label == ref_pred[0])
    print(
        f"  encoder round-trip: encode -> {tuple(single.shape)} / {tuple(batched.shape)} ; "
        f"config identical = {cfg_exact} ; latents bit-equal = {latent_exact} ; "
        f"logits bit-equal = {logit_exact} ; classify agrees = {class_exact}"
    )

    # ---- criteria. Every one has a RED constructed and measured; see the P8 block in the docstring.
    assert rate > 0.0, "hidden units must spike at init (else psi=0 and the gradient is undefined)"
    assert cos_hidden > EPROP_BPTT_COSINE_FLOOR, (
        f"(b) e-prop gradient must ALIGN with the true BPTT gradient: cosine {cos_hidden:.4f} !> {EPROP_BPTT_COSINE_FLOOR:.2f}"
    )
    assert final_loss < init_loss, (
        f"(a) e-prop must reduce loss: {final_loss:.4f} !< {init_loss:.4f}"
    )
    # CONTROL (P8 item 2): the untrained twin is the same net before any e-prop step. It must FAIL
    # the same bar the trained net must pass -- otherwise the task is winnable without learning and
    # the accuracy result says nothing. This is the red for the bar, measured inside the run.
    assert init_acc <= bar, (
        f"control broken: the UNTRAINED net already clears the bar ({init_acc:.1f}% vs {bar:.2f}%)"
        " -- the task is too easy to be evidence of learning"
    )
    assert final_acc > bar, (
        f"(a) accuracy must clear chance + {CHANCE_SIGMAS:.0f} sigma: {final_acc:.1f}% !> {bar:.2f}%"
    )
    assert cfg_exact, (
        f"checkpoint round-trip must restore every dynamics field: {enc_cfg} != {ref_cfg} -- a field"
        " config() stops persisting is silent in the OUTPUTS when it only affects psi() (gamma/damp)"
    )
    assert latent_exact, (
        "checkpoint round-trip must reproduce the source net's latents EXACTLY -- a config()/"
        "state_dict field that stops persisting changes every latent cortex/brain.py consumes,"
        " with the shapes still matching"
    )
    assert logit_exact, (
        "checkpoint round-trip must reproduce the source net's logits EXACTLY -- kappa is"
        " readout-only, so it moves these and NOT the latents"
    )
    # `assert class_exact` stood here. DELETED 2026-08-05 -- it is implied by the assert directly
    # above (class predictions are the argmax of those logits) and no member of its documented
    # field-drop red family reddens it. `class_exact` is still computed and printed above, as a
    # report. See the P8 block in the module docstring for the measurement.
    print(
        f"SMOKE OK: e-prop~BPTT hidden cosine {cos_hidden:.4f} (>{EPROP_BPTT_COSINE_FLOOR:.2f}, factored gradient confirmed vs an INDEPENDENT"
    )
    print(
        f"reverse-mode reference); e-prop LEARNS loss {init_loss:.4f}->{final_loss:.4f} (acc {init_acc:.1f}%->{final_acc:.1f}%; the UNTRAINED"
    )
    print(
        f"control fails the same {bar:.2f}% bar). Encoder load/encode/classify reproduces the source net"
    )
    print(
        f"bit-exactly (no dataset, {device}). loss.backward NEVER called for the hidden weights -- they are"
    )
    print("trained purely by the online eligibility factorization.")


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Brain-native (e-prop) recurrent spiking classifier / "
        "encoder on N-MNIST -- online eligibility traces, NO BPTT."
    )
    # BARE INVOCATION RUNS THE SELF-CHECK. run_selfchecks.py executes every registered module as
    # `python -m <dotted.module>` with NO arguments, so a promoted module whose no-arg path needed
    # tonic + the 60k-sample N-MNIST dataset could never be a gate entry. Training is now opt-in
    # behind --train. --smoke is kept as an accepted alias so existing invocations keep working.
    p.add_argument(
        "--train",
        action="store_true",
        help="run the real N-MNIST e-prop training (needs tonic + the dataset)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="alias for the default no-argument self-check (kept for existing callers)",
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
    # "auto" resolves PER PATH, in main(): cuda-if-available for --train (P5 -- the training tier
    # belongs on the card), always cpu for the self-check. The old default was cuda-if-available for
    # BOTH, which would have put a promoted gate entry on non-bit-exact GPU kernels on any machine
    # that has one and on CPU in CI -- two different gates wearing one name. At the self-check's
    # width of 12 units a GPU buys nothing anyway; --device cuda still forces it if asked.
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p


def main():
    args = build_parser().parse_args()
    train = args.train and not args.smoke
    if args.device == "auto":
        args.device = "cuda" if (train and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[note] cuda requested but unavailable -- falling back to cpu")
        args.device = "cpu"
    if not train:
        run_smoke(args.device)
        return
    train_and_eval(args)


if __name__ == "__main__":
    main()
