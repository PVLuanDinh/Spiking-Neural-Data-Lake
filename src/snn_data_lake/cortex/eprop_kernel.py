"""eprop_kernel.py -- the ONE e-prop training kernel for this project's torch spiking encoders.

WHAT THIS IS. `sensorimotor/eprop_nmnist.py` (vision, N-MNIST) and `sensorimotor/eprop_shd.py` (audio, SHD) held
NINE byte-identical definitions between them -- the surrogate spike function, the independent BPTT
reference, the cosine comparator, the Adam step, the evaluation loop, and four LIF pieces. Two
copies of one engine is the P2 debt this module discharges: P2 asks that exactly ONE mechanism
survive, and a MERGE is "build the one mechanism that supersedes both, then retire both originals".
Both consumers now import from here and their copies are gone.

Recorded because a move is not a merge: the 2026-08-13 promotion of eprop_shd into cortex/ put both
twins in the governed tree, which made the duplication VISIBLE but no smaller. This module is what
actually reduces it.

# consolidates: src/snn_data_lake/sensorimotor/eprop_nmnist.py -- its 9 shared definitions are deleted
#   there and imported from here in the same change (specs/013 C4)
# consolidates: src/snn_data_lake/sensorimotor/eprop_shd.py -- likewise; it also loses its hand-written
#   bare 0.5 in favour of EPROP_BPTT_COSINE_FLOOR below
# distinct-from: src/snn_data_lake/reasoning/eprop.py -- NOT a third copy and deliberately left
#   alone. That module is a pure-STDLIB e-prop reference (math/zip, no torch, no autograd) whose
#   `_cosine` is a different implementation for a different substrate; POLICIES.md cites it as the
#   verification anchor for the whole e-prop invariant. Same algorithm, different substrate, so
#   merging them would mean deleting the stdlib reference, which is not this lane's business.
# distinct-from: src/snn_data_lake/reasoning/eprop_lm_serve.py, reservoir_eprop.py -- language-model
#   and reservoir e-prop variants with their own forward contracts and their own thresholds; the
#   SCOPE LIMIT below is exactly why they are not folded in here.

SCOPE LIMIT -- TWO CONSUMERS, NOT FOUR. `_SpikeFn.backward` returns `grad_out * psi, None, None,
None`: NO gradient reaches the threshold argument. That is correct for both consumers here, whose
`v_th` is a float constant. It is WRONG for a consumer whose threshold is an adapted tensor --
measured on `training/eprop_lm_alif.py`, the 1-arg surrogate form gives ||d spikes/d a|| = 1.0727
where this 4-arg form gives None. So this module must NOT be advertised as "the one surrogate for
every e-prop trainer": adopting it there would silently zero the ALIF adaptation gradient WHILE
REDDENING NOTHING, because that module's own cosine assert stays above its floor. Widening the
contract requires a backward that returns a real `v_th` gradient, and that is its own lane with its
own A/B.

WHAT THE COSINE CHECK CANNOT SEE, disclosed here rather than left to be discovered. `bptt_grads` is
a genuinely independent GRADIENT computation -- it back-propagates adjoints including the recurrent
off-diagonal and reset terms e-prop deliberately drops. It is NOT independent of the FORWARD path:
it consumes the same `forward`, hence the same `flatten_frames`. A defect upstream of that fork
displaces BOTH sides together and the cosine is unchanged. MEASURED at specs/013: permuting
`flatten_frames`' feature axis moved all 12 pinned tensors (max |delta| 538.071) while the cosine
went 0.995375 -> 0.997168 on the vision net -- it ROSE. Independence is a property of a control
WITH RESPECT TO A STAGE, never a property of the control alone. The verifier for anything upstream
of that fork is the bit-exact pin at `tests/pin/`, not this assert.

IMPORT CONTRACT. This module imports nothing from `eprop_nmnist` or `eprop_shd` -- the dependency
runs consumer -> kernel only, and a back-import would be a cycle. Its self-check builds a minimal
concrete subclass inline for the same reason.

  Self-check:  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.cortex.eprop_kernel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Which symbols here OWE a gradient cosine, read statically by
#: `gates/gradient_claim_gate.py` (never imported by it). Per-SYMBOL rather than per-module
#: because this one file holds the e-prop rule, its independent BPTT reference, a surrogate
#: autograd.Function and a diagnostic -- four different answers that a per-module verdict cannot give.
#:
#: `_first_layer_grad` is now declared NOT_A_RULE. It was left UNDECLARED at first, because the four
#: original dispositions had no value meaning "not a learning rule" -- it is a DIAGNOSTIC ("summed
#: |grad| at the input weights, or None when the graph was detached"). Leaving it undeclared reported
#: it as UNDETERMINABLE, which was true but conflated "I cannot decide" with "there is nothing here
#: to decide", and UNDETERMINABLE is ratcheted DOWN as debt this is not. The fifth value closed that.
GRADIENT_CLAIM = {
    "bptt_grads": (
        "EXACT_GRADIENT",
        "true reverse-mode BPTT of the same loss through a differentiable rebuild of the same "
        "unrolled net, via autograd -- this IS the independent reference the cosine clause compares "
        "against, so it owes no cosine of its own",
    ),
    "_TinyNet.eprop_grads": (
        "APPROXIMATES_GRADIENT",
        "eligibility factorization, symmetric e-prop: local trace x broadcast learning signal. Owes "
        "the cosine, and is the inline fixture FR-011 reconciles to the kappa-filtered form both "
        "shipped encoders run",
    ),
    "_apply_grads": (
        "APPROXIMATES_GRADIENT",
        "applies the e-prop-computed grads via one optimizer step with no loss.backward() anywhere; "
        "the update half of the same rule, so it carries the same claim",
    ),
    "_first_layer_grad": (
        "NOT_A_RULE",
        "a DIAGNOSTIC: summed |grad| at the input weights, or None when the graph was detached. It "
        "reads a gradient to report a magnitude and updates no weight, so it makes no claim a cosine "
        "could check. Enumerated only because the recall-biased walk keys on 'grad'",
    ),
}

# ---- LIF / e-prop constants. Byte-identical AND same-valued in both consumers before the merge;
# ---- the per-modality GEOMETRY (num_classes, T, hidden, sensor dims, channels) deliberately does
# ---- NOT live here -- those differ between vision and audio and hoisting them would hardcode one
# ---- modality's shape into the other's path.
DEFAULT_ALPHA = 0.90  # membrane decay exp(-dt/tau_m): the diagonal of the recurrent Jacobian
DEFAULT_V_TH = 0.60  # spike threshold (soft reset subtracts V_TH)
DEFAULT_GAMMA = 0.50  # pseudo-derivative half-width: psi = DAMP*max(0, 1 - |v-V_TH|/GAMMA)
DEFAULT_DAMP = 0.30  # pseudo-derivative dampening factor
DEFAULT_KAPPA = 0.90  # leaky-readout decay (also the eligibility-trace output filter)

# The e-prop invariant's bar (docs/POLICIES.md cross-cutting: "verified cosine > 0.5 against an
# independent reverse-mode BPTT gradient"). It was a NAMED constant in eprop_nmnist and a bare
# hand-written 0.5 in eprop_shd; one definition now, imported by both.
EPROP_BPTT_COSINE_FLOOR = 0.5


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


class EpropLIFBase(nn.Module):
    """The recurrent-LIF forward shared by both perception encoders.

    A subclass MUST set, in its own __init__: `n_in`, `hidden`, `num_classes`, the parameters
    `W_in`/`W_rec`/`W_out`/`b_out`, and the dynamics constants `alpha`/`v_th`/`gamma`/`damp`/
    `kappa`. A subclass MUST also define `eprop_grads` -- the eligibility factorization is the
    per-modality half and does not live here.
    """

    def psi(self, v):
        """Surrogate pseudo-derivative of the spike: DAMP * max(0, 1 - |v - V_TH| / GAMMA)."""
        return self.damp * torch.clamp(1.0 - (v - self.v_th).abs() / self.gamma, min=0.0)

    def flatten_frames(self, x):
        """[B, T, ...] frames -> [B, T, n_in] per-timestep input vectors.

        ONE body for both modalities. The vision consumer previously used
        `x.reshape(*x.shape[:-3], self.n_in)`, which is bit-identical to this on every shape it
        receives (verified 9/9 across B in {1,2,5} x T in {1,3,15}, and end-to-end: forward,
        eprop_grads and bptt_grads all bit-identical after the swap) but RAISES on the audio
        consumer's 3-D input. This body subsumes it, so keeping two would be additive with extra
        steps. The genuine per-modality seam is `self.n_in`, already a per-subclass attribute.
        """
        return x.reshape(x.shape[0], x.shape[1], self.n_in)

    def offdiag_mask(self, device):
        return 1.0 - torch.eye(self.hidden, device=device)

    @torch.no_grad()
    def forward(self, x):
        """Serve: x -> (logits_sum [B, num_classes], latent_sum [B, hidden]).

        Input is [B, T, C, H, W] event frames for the VISION consumer and [B, T, C] audio frames for
        the AUDIO consumer; `flatten_frames` reduces both to [B, T, n_in]. logits_sum is the
        time-summed leaky readout (the class scores); latent_sum is the hidden spike-COUNT activity
        over the T window (the encoder latent). Manual LIF recurrence, no autograd, no
        dataset/tonic needed -- torch-only-to-serve.
        """
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


class EpropEncoderBase:
    """Wraps a trained net as a perception encoder. Subclasses define load/encode/classify."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device


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


# ---------------------------------------------------------------------------------------------
# THE NEGATIVE CONTROL, ported here 2026-08-13 from the retired training/surrogate_snn.py.
#
# It was the only surrogate-OFF control in the tree, and the FR-018 capability audit ruled it a
# PORT rather than a DROP for that reason: P8 guard (2) asks an invented method for "a control that
# is not the method itself", and every other Heaviside in this tree is Heaviside-forward-with-
# surrogate-backward -- that is the METHOD, not a control on it. It reddens on a different axis
# from the cosine check, which is exactly why it is worth carrying.
# ---------------------------------------------------------------------------------------------
def _heaviside_no_surrogate(v, v_th, gamma, damp):
    """A TRUE Heaviside: no custom backward, so it DETACHES the graph. Negative control only.

    Signature matches `_SpikeFn.apply` so it can be swapped in directly.
    """
    return (v >= v_th).float()


def _first_layer_grad(net):
    """Summed |grad| at the input weights, or None when the graph was detached."""
    g = net.W_in.grad
    return None if g is None else float(g.abs().sum())


def _fwd_bwd(net, x, y, spike_fn):
    """One differentiable forward+backward through the unrolled net using `spike_fn`.

    `loss.requires_grad` is CHECKED rather than assumed: with a true Heaviside the whole graph is
    detached and `loss.backward()` would RAISE instead of reporting no gradient. The control needs
    "no gradient" as a MEASUREMENT, not as an exception. (This reasoning is inherited verbatim from
    surrogate_snn._fwd_bwd, which is the file this control came from.)
    """
    net.zero_grad(set_to_none=True)
    xf = net.flatten_frames(x.float())
    b, t = xf.shape[0], xf.shape[1]
    dev = xf.device
    mask = net.offdiag_mask(dev)
    w_rec = net.W_rec * mask
    v = torch.zeros(b, net.hidden, device=dev)
    z = torch.zeros(b, net.hidden, device=dev)
    yint = torch.zeros(b, net.num_classes, device=dev)
    loss = torch.zeros((), device=dev)
    for step in range(t):
        cur = xf[:, step] @ net.W_in.t() + z @ w_rec.t()
        v = net.alpha * v - net.v_th * z + cur
        z = spike_fn(v, net.v_th, net.gamma, net.damp)
        yint = net.kappa * yint + z @ net.W_out.t()
        loss = loss + F.cross_entropy(yint + net.b_out, y.long(), reduction="sum")
    if loss.requires_grad:
        loss.backward()
    return float(loss.item()), _first_layer_grad(net)


# ---------------------------------------------------------------------------------------------


class _TinyNet(EpropLIFBase):
    """Minimal concrete subclass, built INLINE for the self-check.

    It exists because the kernel must not import its consumers (cycle), so the self-check cannot
    reach for EpropNet or EpropAudioNet. Deliberately tiny and seeded.
    """

    def __init__(self, n_in=12, hidden=8, num_classes=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_in, self.hidden, self.num_classes = n_in, hidden, num_classes
        self.alpha, self.v_th = DEFAULT_ALPHA, DEFAULT_V_TH
        self.gamma, self.damp, self.kappa = DEFAULT_GAMMA, DEFAULT_DAMP, DEFAULT_KAPPA
        self.W_in = nn.Parameter(torch.randn(hidden, n_in, generator=g) * 0.3)
        self.W_rec = nn.Parameter(torch.randn(hidden, hidden, generator=g) * 0.2)
        self.W_out = nn.Parameter(torch.randn(num_classes, hidden, generator=g) * 0.3)
        self.b_out = nn.Parameter(torch.zeros(num_classes))

    def eprop_grads(self, x, y):
        """Eligibility factorization, symmetric e-prop. Local trace x broadcast learning signal."""
        xf = self.flatten_frames(x.float())
        b, t = xf.shape[0], xf.shape[1]
        dev = xf.device
        mask = self.offdiag_mask(dev)
        w_rec = self.W_rec * mask
        v = torch.zeros(b, self.hidden, device=dev)
        z = torch.zeros(b, self.hidden, device=dev)
        yint = torch.zeros(b, self.num_classes, device=dev)
        # RECONCILED to the kappa-filtered form both shipped encoders run (specs/020 T048, FR-010's
        # single named exception). `eps` is the ALPHA-filtered presynaptic trace, shared across post
        # units; `ebar` is the KAPPA-filtered eligibility the readout's leaky integrator actually
        # sees. The unfiltered form previously here applied the learning signal straight to `eps`,
        # which is a DIFFERENT factorization from the one that ships -- so the criterion was
        # measuring a fixture nothing runs.
        eps_in = torch.zeros(b, self.n_in, device=dev)
        eps_rec = torch.zeros(b, self.hidden, device=dev)
        ebar_in = torch.zeros(b, self.hidden, self.n_in, device=dev)
        ebar_rec = torch.zeros(b, self.hidden, self.hidden, device=dev)
        zbar = torch.zeros(b, self.hidden, device=dev)
        g_in = torch.zeros_like(self.W_in)
        g_rec = torch.zeros_like(self.W_rec)
        g_out = torch.zeros_like(self.W_out)
        g_b = torch.zeros_like(self.b_out)
        loss_sum = 0.0
        with torch.no_grad():
            for step in range(t):
                xt, z_prev = xf[:, step], z
                cur = xt @ self.W_in.t() + z_prev @ w_rec.t()
                v = self.alpha * v - self.v_th * z + cur
                p = self.psi(v)
                z = (v >= self.v_th).float()
                yint = self.kappa * yint + z @ self.W_out.t()
                logits = yint + self.b_out
                err = torch.softmax(logits, dim=1)
                err[torch.arange(b), y.long()] -= 1.0
                loss_sum += float(F.cross_entropy(logits, y.long(), reduction="sum").item())
                # eps = ALPHA*eps + pre ; ebar = KAPPA*ebar + psi_post * eps  (mirrors eprop_shd)
                eps_in = self.alpha * eps_in + xt
                eps_rec = self.alpha * eps_rec + z_prev
                ebar_in = self.kappa * ebar_in + p.unsqueeze(2) * eps_in.unsqueeze(1)
                ebar_rec = self.kappa * ebar_rec + p.unsqueeze(2) * eps_rec.unsqueeze(1)
                lsig = err @ self.W_out  # broadcast learning signal
                g_in += torch.einsum("bj,bji->ji", lsig, ebar_in)
                g_rec += torch.einsum("bj,bjk->jk", lsig, ebar_rec)
                # The readout is non-recurrent, so its gradient is EXACT -- but only against the
                # KAPPA-filtered spike train, because `yint = kappa*yint + z @ W_out` is what the
                # loss actually sees. The raw `z` used here before was not exact and not identical
                # to BPTT: measured cosine 0.986830, relative L2 7.4233e-01, allclose False.
                zbar = self.kappa * zbar + z
                g_out += torch.einsum("bk,bj->kj", err, zbar)
                g_b += err.sum(0)
        return {"W_in": g_in, "W_rec": g_rec * mask, "W_out": g_out, "b_out": g_b}, loss_sum


def _self_check():
    torch.manual_seed(7)
    net = _TinyNet(seed=7)
    x = (torch.rand(6, 9, 12, generator=torch.Generator().manual_seed(11)) < 0.3).float()
    y = torch.tensor([0, 1, 2, 0, 1, 2])

    logits, latent = net.forward(x)
    assert logits.shape == (6, 3), logits.shape
    assert latent.shape == (6, 8), latent.shape

    # (a) THE CLAIM: the eligibility factorization really is the factored gradient.
    #
    # ASSERTED ON THE HIDDEN WEIGHTS ONLY -- W_in and the off-diagonal W_rec -- which is what both
    # consumers assert (`cos_hidden`) and what the claim is actually about. The readout gradients
    # never touch the eligibility machinery, so they can only dilute.
    #
    # THE OLD WORDING HERE WAS MEASURABLY FALSE AND IS CORRECTED (specs/020 T052). It said the
    # readout gradients "are IDENTICAL between e-prop and BPTT by construction", computed as
    # `err.t() @ z`. They were not: the loss sees `yint = kappa*yint + z @ W_out`, so the exact
    # readout gradient is `err (x) zbar` against the KAPPA-FILTERED spike train. Measured on this
    # fixture with the raw-`z` form: W_out cosine 0.986830, relative L2 7.4233e-01,
    # `torch.allclose` FALSE. Only `b_out` was ever identical (1.000000). Under the reconciled
    # kappa-filtered form the claim is now TRUE rather than asserted: W_out cosine 1.000000,
    # relative L2 8.16e-08, allclose True. That is what the reconciliation actually buys.
    #
    # This was measured, not assumed. The first version of this check asserted the cosine over all
    # four gradient keys, and W_out+b_out carry 81.2% of that vector's squared norm -- so a defect in
    # the hidden path barely moves it. Three seeded mechanism defects, all-4 assert vs hidden-only,
    # ALL RE-DERIVED under the reconciled form (the pre-reconciliation values are in parentheses):
    #     sign-flipped learning signal   all4 0.6504 GREEN | hidden -0.9993 RED  (was 0.5859 | -0.9714)
    #     zeroed input eligibility trace all4 0.9214 GREEN | hidden  0.2588 RED  (was 0.7465 |  0.2563)
    #     zeroed surrogate derivative    all4 0.9155 GREEN | hidden  0.0000 RED  (was 0.7398 |  0.0000)
    # The reconciliation STRENGTHENS this argument rather than weakening it: making the readout
    # exact moved every all-4 value UP (0.59->0.65, 0.75->0.92, 0.74->0.92), so the diluting share
    # is now a perfectly-aligned block and an all-4 criterion would be even easier to pass while the
    # mechanism is broken. The hidden-only reds are unchanged in kind.
    # THREE OF FOUR real defects passed the all-4 assert while reddening both consumers. A criterion
    # that is 90% a quantity which cannot differ is not measuring the mechanism it names.
    #
    # The all-4 value is still PRINTED, as a report. It is not a bar (P8: a number reported is not a
    # number asserted), and it uniquely catches a CE-target sign flip, which is worth seeing.
    ge, _ = net.eprop_grads(x, y)
    gb = bptt_grads(net, x, y)
    hid = net.offdiag_mask(ge["W_rec"].device).bool()
    cos = _cosine(
        torch.cat([ge["W_in"].reshape(-1), ge["W_rec"][hid].reshape(-1)]),
        torch.cat([gb["W_in"].reshape(-1), gb["W_rec"][hid].reshape(-1)]),
    )
    cos_all = _cosine(
        torch.cat([ge[k].reshape(-1) for k in sorted(ge)]),
        torch.cat([gb[k].reshape(-1) for k in sorted(gb)]),
    )
    print(
        f"e-prop~BPTT cosine_hidden {cos:.4f} (floor {EPROP_BPTT_COSINE_FLOOR}); "
        f"all-parameter {cos_all:.4f} (report only, NOT a criterion)"
    )
    # SC-008: every cosine the specs/020 T048 reconciliation MOVED, printed with both values, so no
    # movement is silently absorbed. This fixture ran an UNFILTERED factorization; the shipped
    # encoders run the kappa-filtered one, so the criterion was attached to a form nothing runs.
    #     cosine_hidden   0.9714 -> 0.9993      W_out cosine  0.986830 -> 1.000000
    #     all-parameter   0.8236 -> 0.9993      W_out rel L2  7.4233e-01 -> 8.16e-08
    #     RED case (b)   -0.2540 -> -0.1782     readout share    90.7% -> 81.2%
    # WHAT THIS DID NOT FIX, stated so the scope is not overread (specs/020 T054): this is an inline
    # TEST FIXTURE, not the guard protecting the shipped encoders. All three mechanizations already
    # run in the CI torch job and each encoder guards itself on its OWN kappa-filtered form -- none
    # of them was ever measured against the unfiltered version. Nothing shipped was mis-trained; what
    # was wrong is that the REFERENCE this module offers was not the rule the tree runs.
    print(
        "  reconciled to the shipped kappa-filtered form (specs/020 T048): cosine_hidden "
        "0.9714 -> 0.9993, all-parameter 0.8236 -> 0.9993, readout now EXACT (0.986830 -> 1.000000)"
    )
    assert cos > EPROP_BPTT_COSINE_FLOOR, f"cos_hidden {cos} <= {EPROP_BPTT_COSINE_FLOOR}"

    # (b) REACHABLE RED #1 (FR-019): a CONFIGURATION in which the eligibility factorization genuinely
    # stops tracking the true gradient. Strong recurrence + weak drive + long T is the regime where
    # the recurrent-error path e-prop TRUNCATES dominates the one it keeps, so the cosine collapses
    # while the network stays healthily active -- not a dead-network artifact.
    #
    # THIS REPLACED A VACUOUS ASSERT, and the replacement is the point rather than an embarrassment
    # to bury. The first version negated the already-computed gradient dict
    # (`flipped = {k: -v for k, v in ge.items()}`). `_cosine` is ODD in its first argument and its
    # norm is negation-invariant, so cos_bad == -cos BIT-EXACTLY (measured: worst
    # |cos_bad - (-cos)| = 0.000e+00 over 12 seeds). With `assert cos > FLOOR` on the line above,
    # `cos_bad <= FLOOR` was ENTAILED -- an assert that could not fail, which is exactly what P8
    # guard (1) says must be deleted rather than weakened. Worse, the mutation that version NAMED
    # (negating the broadcast learning signal INSIDE eprop_grads) measures cosine 0.6504 and does
    # NOT redden the check at all, because b_out and W_out do not depend on it. (0.5859 before the
    # specs/020 T048 reconciliation; the value ROSE, so the defect is even better hidden now.)
    # `sensorimotor/eprop_shd.py` had already made this exact correction for this exact defect on
    # 2026-08-05; the kernel now follows the precedent instead of re-introducing what it rejected.
    red_net = _TinyNet(n_in=12, hidden=8, num_classes=3, seed=3)
    red_net.alpha = 0.99
    with torch.no_grad():
        red_net.W_rec.mul_(30.0)  # strong recurrence: the path e-prop truncates
        red_net.W_in.mul_(0.05)  # weak drive
    rx = (torch.rand(4, 96, 12, generator=torch.Generator().manual_seed(5)) < 0.3).float()
    ry = torch.tensor([0, 1, 2, 0])
    ge_bad, _ = red_net.eprop_grads(rx, ry)
    gb_bad = bptt_grads(red_net, rx, ry)
    hid_bad = red_net.offdiag_mask(ge_bad["W_rec"].device).bool()
    cos_bad = _cosine(  # same scope as the assert above -- a red must redden the ACTUAL criterion
        torch.cat([ge_bad["W_in"].reshape(-1), ge_bad["W_rec"][hid_bad].reshape(-1)]),
        torch.cat([gb_bad["W_in"].reshape(-1), gb_bad["W_rec"][hid_bad].reshape(-1)]),
    )
    rate = float((red_net.forward(rx)[1].sum() / (4 * 96 * 8)).item())
    print(
        f"RED  alpha=0.99 g_rec=30 g_in=0.05 T=96 -> cosine {cos_bad:.4f} "
        f"(<= floor: {cos_bad <= EPROP_BPTT_COSINE_FLOOR}) at a HEALTHY spike rate {rate:.3f}"
    )
    assert cos_bad <= EPROP_BPTT_COSINE_FLOOR, (
        "no reachable red -- DELETE the criterion, do not weaken it"
    )
    assert rate > 0.05, (
        f"the red must not be a dead-network artifact: spike rate {rate:.3f} -- a silent net "
        "reddens the cosine for the wrong reason and would make this red meaningless"
    )

    # (c) REACHABLE RED #2, on a DIFFERENT axis: the surrogate-OFF control ported from the retired
    # surrogate_snn. With a TRUE Heaviside the graph detaches and there is no gradient at all; with
    # the surrogate there is one. If both behaved alike the surrogate would not be doing the work
    # this module claims for it.
    loss_s, grad_s = _fwd_bwd(net, x, y, _SpikeFn.apply)
    loss_h, grad_h = _fwd_bwd(net, x, y, _heaviside_no_surrogate)
    print(f"RED  control: surrogate grad={grad_s}  true-Heaviside grad={grad_h}")
    assert grad_s is not None and grad_s > 0.0, "surrogate produced no gradient"
    assert grad_h is None, "true Heaviside produced a gradient -- the control is not a control"
    assert abs(loss_s - loss_h) < 1e-6, "forward must be identical; only the BACKWARD differs"

    print(
        "SMOKE OK: one governed e-prop kernel; cosine asserted against an INDEPENDENT reverse-mode\n"
        "reference, with TWO demonstrated reds (a truncation-regime configuration, and surrogate-off).\n"
        "The cosine CANNOT see a defect upstream of the forward/gradient fork -- see the module\n"
        "docstring; the bit-exact pin at tests/pin/ is the verifier for that."
    )


if __name__ == "__main__":
    _self_check()
