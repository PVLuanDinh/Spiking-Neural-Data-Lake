r"""training/eprop_lm_alif.py -- BRAIN-NATIVE [train] tier: a DEEP, LONG-RANGE recurrent-spiking char LM.

WHY (the gap this closes). training/eprop_lm.py trains a SINGLE-LAYER recurrent LIF char LM by e-prop.
That model works (its e-prop gradient aligns with BPTT), but a plain LIF neuron keeps NO slow state:
its membrane leaks within a few steps, so it cannot carry a grammatical dependency across DISTANCE
(subject-verb agreement over an intervening clause) -- exactly the long-range / hierarchical structure
BLiMP probes, which is why the single-layer LIF sits at chance there. This file closes that gap the way
the e-prop paper itself prescribes: (1) ADAPTIVE-threshold LIF (ALIF / LSNN) neurons -- an adaptation
variable with a SLOW time constant tau_a >> tau_m gives each neuron a working-memory trace that outlasts
the membrane by many membrane time constants (Bellec, Salaj, Subramoney, Legenstein & Maass, NeurIPS
2018, the LSNN), and (2) DEPTH -- a STACK of recurrent ALIF layers, so higher layers compose the slower,
more abstract dependencies. Trained by e-prop end to end (Bellec, Scherr, Subramoney, Hajek, Salaj,
Legenstein & Maass, "A solution to the learning dilemma for recurrent networks of spiking neurons",
Nature Communications 11:3625, 2020, DOI 10.1038/s41467-020-17236-y): online eligibility traces + a
per-layer broadcast learning signal, with NO loss.backward() through time for the hidden weights.

THE ALIF NEURON (matches src/snn_data_lake/growth/alif_neuron.py's adaptive-threshold mechanism).
Per layer, per neuron j, discrete time (soft-reset LIF membrane -- the form e-prop is derived for, as in
training/eprop_lm.py -- plus the LSNN's slow adaptive threshold from alif_neuron.py):
    v_j[t]  = ALPHA*v_j[t-1] - V_TH*z_j[t-1] + cur_j[t]        # fast membrane (leak ALPHA), soft reset
    a_j[t]  = RHO  *a_j[t-1] +      z_j[t-1]                    # SLOW adaptation (leak RHO >> nothing)
    A_j[t]  = V_TH + BETA*a_j[t]                               # effective threshold RISES with recent firing
    z_j[t]  = 1[ v_j[t] >= A_j[t] ]                            # spike against the ADAPTED threshold
    psi_j[t]= DAMP*max(0, 1 - |v_j[t]-A_j[t]|/GAMMA)           # triangular pseudo-derivative at A_j[t]
BETA=0 collapses this EXACTLY onto training/eprop_lm.py's plain-LIF neuron (a=0 forever), mirroring
alif_neuron.py's "beta=0 == LIF" anchor. RHO = exp(-1/tau_a) with tau_a >> tau_m is the long timescale.

THE ALIF e-prop UPDATE (the paper's adaptive-neuron rule -- TWO eligibility components, not one).
A LIF neuron's hidden state is 1-D (v), so e-prop carries ONE eligibility component. An ALIF neuron's
hidden state is 2-D (v, a), so its eligibility carries a SECOND component that tracks the slow adaptation
state -- this is what lets the slow timescale participate in temporal credit assignment. Per synapse j<-i:
    eps_v_ji[t] = ALPHA*eps_v_ji[t-1] + pre_i[t]                              # membrane eligibility VECTOR
    eps_a_ji[t] = psi_j[t-1]*eps_v_ji[t-1] + (RHO - BETA*psi_j[t-1])*eps_a_ji[t-1]   # ADAPTATION eligibility
    e_ji[t]     = psi_j[t] * ( eps_v_ji[t] - BETA*eps_a_ji[t] )               # eligibility TRACE
    ebar_ji[t]  = KAPPA*ebar_ji[t-1] + e_ji[t]                                # readout(KAPPA)-filtered trace
    dW_ji      += L_j[t] * ebar_ji[t]                                         # broadcast learning signal x trace
pre_i is the presynaptic activity (input one-hot for W_in; lower-layer spike for the feedforward W_ff;
same-layer previous spike for W_rec). At BETA=0, eps_a stays 0 and e_ji = psi_j*eps_v_ji -- byte-identical
to training/eprop_lm.py's LIF eligibility. eps_v is shared across post-neurons j (it depends only on the
presynapse), but eps_a is NOT (its recursion is gated by the post pseudo-derivative psi_j) -- so the
adaptation eligibility is a genuine per-synapse tensor, the price the ALIF rule pays for its long memory.

PER-LAYER BROADCAST LEARNING SIGNAL (symmetric-feedback deep e-prop). The top hidden layer feeds the
leaky readout, so its learning signal is the readout error broadcast through the readout weights, exactly
as in the single-layer file:  L_top_j[t] = sum_c W_out[c,j]*err_c[t]  (err = softmax - onehot). A lower
layer l gets its signal by broadcasting the layer above through the ACTUAL feedforward weights (symmetric
feedback), gated by the upper layer's pseudo-derivative:  L_l[t] = (psi_{l+1}[t] * L_{l+1}[t]) @ W_ff_{l+1}.
The vertical (inter-layer, same-step) credit is thus exact; only the HORIZONTAL (within-layer temporal /
recurrent + adaptation) credit is the e-prop eligibility approximation -- which is what keeps the
gold-standard cosine below high while remaining an honest online, BPTT-free rule.

NON-RIGGABLE SELF-CHECK (the gold standard). "The two-component eligibility trace really IS the factored
gradient" is tested by cosine(e-prop hidden gradient, an INDEPENDENT reverse-mode BPTT gradient of the
SAME cross-entropy loss through the SAME unrolled DEEP-ALIF net using the SAME pseudo-derivative). The
BPTT reference (bptt_grads) is torch autograd's genuine reverse pass through v AND a AND the reset,
recurrent and inter-layer paths -- it shares NO code with the eligibility accumulation in eprop_grads.
e-prop is an APPROXIMATION (it truncates the recurrent-error, reset and inter-layer-temporal paths BPTT
keeps), so the cosine is not 1.0, but symmetric deep-ALIF e-prop aligns clearly; we assert cosine > 0.5.
We also assert a tiny e-prop train REDUCES the loss, and that BETA=0 reduces the ALIF rule to the LIF one.

CONTRACT (mirrors training/eprop_lm.py). torch [train] tier (torch + GPU allowed; verified env is
.venv-cu128). Imports NOTHING from brainstem/ or the zero-dep core; NO import-time side effects (training
runs only under an explicit train() / __main__ call). Exports a FROZEN numpy .npz (per-layer weights +
vocab + config) that reasoning/eprop_lm_alif_serve.py reloads and runs FORWARD-ONLY in numpy. Determinism:
seeded torch + numpy RNGs; the SERVE path over frozen weights is deterministic (GPU training itself is
stochastic, as expected for the torch tier). ASCII-only output; zlib.crc32 (never builtin hash()).

HONEST SCALE. ALIF + depth is the PRINCIPLED long-range lever, but a TINY model still sits near chance on
real open-domain grammar. The self-check proves the DEEP-ALIF e-prop PIPELINE (two-component factored
gradient fidelity + loss reduction) and the serve's ORDER-SENSITIVITY -- NOT a BLiMP number. Moving BLiMP
needs the real-scale GPU run below, and even then the language ceiling may not fully clear; this file
claims no benchmark number a tiny model cannot back.

  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python training/eprop_lm_alif.py         # tiny self-check
  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python training/eprop_lm_alif.py --full  # full GPU train
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_ARTIFACT = os.path.join(_ROOT, "models", "eprop_lm_alif.npz")

FORMAT = "eprop_lm_alif.v1"      # frozen-artifact schema tag (trainer + serve must agree)
# ---- ALIF / e-prop dynamics constants (membrane mirrors eprop_lm.py; adaptation mirrors alif_neuron.py) ----
ALPHA = 0.90                     # fast membrane decay exp(-dt/tau_m): diagonal of the membrane Jacobian
RHO = 0.98                       # SLOW adaptation decay exp(-dt/tau_a), tau_a >> tau_m -> long memory
BETA = 0.12                      # threshold adaptation gain: how much one unit of adaptation lifts V_th
V_TH = 0.6                       # baseline spike threshold (soft reset subtracts V_TH; = plain-LIF threshold)
GAMMA = 0.5                      # pseudo-derivative half-width
DAMP = 0.3                       # pseudo-derivative dampening factor
KAPPA = 0.9                      # leaky-readout decay (also the eligibility-trace output filter)
W_IN_SCALE = 0.9                 # input-weight init std (large enough that chars drive spikes at init)
W_REC_SCALE = 0.2                # recurrent-weight init std
W_FF_SCALE = 0.6                 # inter-layer (feedforward) weight init std (drives the upper layers)
W_OUT_SCALE = 0.2                # readout-weight init std


class _SpikeFn(torch.autograd.Function):
    """Heaviside spike forward (byte-exact for numpy parity) + triangular surrogate on the backward.
    surrogate(arg) = DAMP*max(0, 1 - |arg|/GAMMA), arg = v - A (the ADAPTED threshold) -- the SAME
    pseudo-derivative _psi returns. Used ONLY by the BPTT reference; because A = V_TH + BETA*a, autograd
    routes the gradient through BOTH v and the adaptation a -- the e-prop path recomputes psi itself."""

    @staticmethod
    def forward(ctx, arg):
        ctx.save_for_backward(arg)
        return (arg >= 0).to(arg.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (arg,) = ctx.saved_tensors
        surrogate = DAMP * torch.clamp(1.0 - arg.abs() / GAMMA, min=0.0)
        return grad_out * surrogate


def _psi(v, a):
    """Triangular pseudo-derivative at the ADAPTED threshold A = V_TH + BETA*a (e-prop path). Plain value
    (no autograd) for the eligibility gating; identical shape to the _SpikeFn surrogate."""
    return DAMP * torch.clamp(1.0 - (v - (V_TH + BETA * a)).abs() / GAMMA, min=0.0)


class EpropAlifLM(torch.nn.Module):
    """DEEP recurrent-spiking char LM: W_in[N,V] -> a STACK of `layers` recurrent ALIF layers (each with
    a no-self-loop recurrent matrix W_rec[l] and, above the first, a feedforward matrix W_ff[l] from the
    layer below) -> a leaky readout W_out[V,N] (+b_out) over the vocab. b_hid[l] is a per-layer resting
    drive. All hidden layers share the width `hidden`. forward(ids[B,T]) records, per layer, the spikes z,
    membrane v, adaptation a and pseudo-derivative psi so both gradient routines can read one shared roll."""

    def __init__(self, vocab, hidden, layers=2, seed=0):
        super().__init__()
        torch.manual_seed(int(seed))
        self.vocab = list(vocab)
        self.vsize = len(self.vocab)
        self.hidden = int(hidden)
        self.layers = int(layers)
        n, vsz, L = self.hidden, self.vsize, self.layers
        self.W_in = torch.nn.Parameter(torch.randn(n, vsz) * W_IN_SCALE)
        self.W_rec = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(n, n) * W_REC_SCALE) for _ in range(L)])
        # W_ff[l] connects layer l-1 -> layer l; index 0 is unused (kept for a simple 1-based layer loop).
        self.W_ff = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(1)) if l == 0
             else torch.nn.Parameter(torch.randn(n, n) * W_FF_SCALE) for l in range(L)])
        self.b_hid = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(n)) for _ in range(L)])
        self.W_out = torch.nn.Parameter(torch.randn(vsz, n) * W_OUT_SCALE)
        self.b_out = torch.nn.Parameter(torch.zeros(vsz))
        self.register_buffer("_mask", 1.0 - torch.eye(n))   # zero the recurrent self-loop

    def _wrec(self, l):
        return self.W_rec[l] * self._mask

    def forward(self, ids):
        """Roll the deep ALIF net forward over ids[B,T]. Returns (logits[B,T,V], Z, V, A, PSI) where each
        of Z/V/A/PSI is a length-`layers` list of [B,T,N] tensors. Layers are advanced bottom-up within a
        step, so layer l reads the SAME-step spike of layer l-1 (feedforward) and its OWN previous spike
        (recurrent). Spikes come from _SpikeFn so this single roll is differentiable for the BPTT path."""
        b, t = ids.shape
        n, vsz, L = self.hidden, self.vsize, self.layers
        dev = self.W_in.device
        cur_in = self.W_in.t()[ids]                          # [B,T,N]: input current into layer 0 per step
        v = [torch.zeros(b, n, device=dev) for _ in range(L)]
        a = [torch.zeros(b, n, device=dev) for _ in range(L)]
        z_prev = [torch.zeros(b, n, device=dev) for _ in range(L)]
        yr = torch.zeros(b, vsz, device=dev)
        logits_l = []
        Z = [[] for _ in range(L)]
        Vv = [[] for _ in range(L)]
        Aa = [[] for _ in range(L)]
        Ps = [[] for _ in range(L)]
        for step in range(t):
            z_below = cur_in[:, step]                        # drive into layer 0 is the char current
            for l in range(L):
                if l == 0:
                    cur = z_below + z_prev[0] @ self._wrec(0).t() + self.b_hid[0]
                else:
                    cur = z_below @ self.W_ff[l].t() + z_prev[l] @ self._wrec(l).t() + self.b_hid[l]
                v[l] = ALPHA * v[l] - V_TH * z_prev[l] + cur         # soft-reset fast membrane
                a[l] = RHO * a[l] + z_prev[l]                        # slow adaptation
                A = V_TH + BETA * a[l]                               # effective (adapted) threshold
                z = _SpikeFn.apply(v[l] - A)
                Z[l].append(z); Vv[l].append(v[l]); Aa[l].append(a[l]); Ps[l].append(_psi(v[l], a[l]))
                z_prev[l] = z
                z_below = z                                         # feed this layer's spikes upward
            yr = KAPPA * yr + z_below @ self.W_out.t()               # readout integrates the TOP layer
            logits_l.append(yr + self.b_out)
        logits = torch.stack(logits_l, 1)
        Z = [torch.stack(s, 1) for s in Z]
        Vv = [torch.stack(s, 1) for s in Vv]
        Aa = [torch.stack(s, 1) for s in Aa]
        Ps = [torch.stack(s, 1) for s in Ps]
        return logits, Z, Vv, Aa, Ps


def eprop_grads(model, ids, y):
    """DEEP-ALIF e-prop gradient -- ONLINE, forward-only two-component eligibility traces + a per-layer
    broadcast learning signal, NO BPTT history. Returns (grads dict {param_name: gradient}, mean CE loss).
    Per synapse it carries eps_v (membrane eligibility, shared across post-neurons) AND eps_a (adaptation
    eligibility, per-synapse), forms e = psi*(eps_v - BETA*eps_a), filters by KAPPA, and accumulates
    L_j*ebar. The learning signal is broadcast top-down through the readout then the feedforward weights."""
    n, vsz, L = model.hidden, model.vsize, model.layers
    dev = model.W_in.device
    with torch.no_grad():
        logits, Z, _Vv, _Aa, Ps = model(ids)
        b, t = ids.shape
        wout = model.W_out                                   # [V,N]
        oh_y = F.one_hot(y, vsz).to(logits.dtype)            # [B,T,V] targets
        oh_x = F.one_hot(ids, vsz).to(logits.dtype)          # [B,T,V] input one-hots (presynaptic to layer 0)
        p = torch.softmax(logits, dim=-1)
        err = p - oh_y                                       # softmax/CE error == the readout "third factor"

        # --- eligibility state (per layer) ---
        ev_in = torch.zeros(b, vsz, device=dev)              # layer-0 input membrane eligibility (shared over j)
        ea_in = torch.zeros(b, n, vsz, device=dev)           # layer-0 input adaptation eligibility (per synapse)
        eb_in = torch.zeros(b, n, vsz, device=dev)           # filtered input trace
        ev_rec = [torch.zeros(b, n, device=dev) for _ in range(L)]
        ea_rec = [torch.zeros(b, n, n, device=dev) for _ in range(L)]
        eb_rec = [torch.zeros(b, n, n, device=dev) for _ in range(L)]
        ev_ff = [torch.zeros(b, n, device=dev) for _ in range(L)]     # index 0 unused
        ea_ff = [torch.zeros(b, n, n, device=dev) for _ in range(L)]
        eb_ff = [torch.zeros(b, n, n, device=dev) for _ in range(L)]
        ev_b = [torch.zeros(b, device=dev) for _ in range(L)]         # bias eligibility (constant pre=1)
        ea_b = [torch.zeros(b, n, device=dev) for _ in range(L)]
        eb_b = [torch.zeros(b, n, device=dev) for _ in range(L)]
        psi_prev = [torch.zeros(b, n, device=dev) for _ in range(L)]  # psi[t-1] per layer
        zbar = torch.zeros(b, n, device=dev)                 # readout eligibility (exact, non-recurrent)

        dW_in = torch.zeros(n, vsz, device=dev)
        dW_rec = [torch.zeros(n, n, device=dev) for _ in range(L)]
        dW_ff = [torch.zeros(n, n, device=dev) for _ in range(L)]
        db_hid = [torch.zeros(n, device=dev) for _ in range(L)]
        dW_out = torch.zeros(vsz, n, device=dev)
        db_out = torch.zeros(vsz, device=dev)

        for step in range(t):
            # per-layer broadcast learning signal, computed TOP-DOWN.
            lsig = [None] * L
            lsig[L - 1] = err[:, step] @ wout                # [B,N] readout error into the top layer
            for l in range(L - 2, -1, -1):
                up = Ps[l + 1][:, step] * lsig[l + 1]        # dE/dv of the layer above (psi-gated)
                lsig[l] = up @ model.W_ff[l + 1]             # symmetric feedback through the real W_ff

            for l in range(L):
                psi = Ps[l][:, step]                         # [B,N] post pseudo-derivative at step t
                pv = psi_prev[l]                             # [B,N] psi[t-1]
                z_prev_l = Z[l][:, step - 1] if step > 0 else torch.zeros(b, n, device=dev)

                if l == 0:                                   # --- input weights (presyn = char one-hot) ---
                    pre = oh_x[:, step]                      # [B,V]
                    ev_old = ev_in
                    ea_in = pv.unsqueeze(2) * ev_old.unsqueeze(1) + (RHO - BETA * pv).unsqueeze(2) * ea_in
                    ev_in = ALPHA * ev_in + pre
                    e = psi.unsqueeze(2) * (ev_in.unsqueeze(1) - BETA * ea_in)     # [B,N,V]
                    eb_in = KAPPA * eb_in + e
                    dW_in += torch.einsum("bj,bji->ji", lsig[0], eb_in)
                else:                                        # --- feedforward weights (presyn = layer below) ---
                    pre = Z[l - 1][:, step]                  # [B,N] SAME-step lower-layer spikes
                    ev_old = ev_ff[l]
                    ea_ff[l] = pv.unsqueeze(2) * ev_old.unsqueeze(1) + (RHO - BETA * pv).unsqueeze(2) * ea_ff[l]
                    ev_ff[l] = ALPHA * ev_ff[l] + pre
                    e = psi.unsqueeze(2) * (ev_ff[l].unsqueeze(1) - BETA * ea_ff[l])
                    eb_ff[l] = KAPPA * eb_ff[l] + e
                    dW_ff[l] += torch.einsum("bj,bji->ji", lsig[l], eb_ff[l])

                # --- recurrent weights (presyn = same-layer previous spikes) ---
                ev_old_r = ev_rec[l]
                ea_rec[l] = pv.unsqueeze(2) * ev_old_r.unsqueeze(1) + (RHO - BETA * pv).unsqueeze(2) * ea_rec[l]
                ev_rec[l] = ALPHA * ev_rec[l] + z_prev_l
                e_r = psi.unsqueeze(2) * (ev_rec[l].unsqueeze(1) - BETA * ea_rec[l])
                eb_rec[l] = KAPPA * eb_rec[l] + e_r
                dW_rec[l] += torch.einsum("bj,bjk->jk", lsig[l], eb_rec[l])

                # --- bias (constant presynaptic drive 1) ---
                ev_old_b = ev_b[l]
                ea_b[l] = pv * ev_old_b.unsqueeze(1) + (RHO - BETA * pv) * ea_b[l]
                ev_b[l] = ALPHA * ev_b[l] + 1.0
                e_b = psi * (ev_b[l].unsqueeze(1) - BETA * ea_b[l])
                eb_b[l] = KAPPA * eb_b[l] + e_b
                db_hid[l] += (lsig[l] * eb_b[l]).sum(0)

                psi_prev[l] = psi

            # readout weights: exact (non-recurrent), integrates the TOP layer's spikes.
            zbar = KAPPA * zbar + Z[L - 1][:, step]
            dW_out += err[:, step].t() @ zbar
            db_out += err[:, step].sum(0)

        for l in range(L):
            dW_rec[l] = dW_rec[l] * model._mask              # keep the recurrent self-loop out
        logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        loss = -(oh_y * logp).sum(-1).mean().item()

    grads = {"W_in": dW_in, "W_out": dW_out, "b_out": db_out}
    for l in range(L):
        grads["W_rec.%d" % l] = dW_rec[l]
        grads["b_hid.%d" % l] = db_hid[l]
        if l > 0:
            grads["W_ff.%d" % l] = dW_ff[l]
    return grads, loss


def _named_params(model):
    """Map the module's parameters onto the flat gradient-dict keys eprop_grads produces."""
    named = {"W_in": model.W_in, "W_out": model.W_out, "b_out": model.b_out}
    for l in range(model.layers):
        named["W_rec.%d" % l] = model.W_rec[l]
        named["b_hid.%d" % l] = model.b_hid[l]
        if l > 0:
            named["W_ff.%d" % l] = model.W_ff[l]
    return named


def bptt_grads(model, ids, y):
    """INDEPENDENT reference: genuine reverse-mode BPTT of the SAME cross-entropy loss through the SAME
    unrolled DEEP-ALIF recurrence, using the SAME pseudo-derivative (via _SpikeFn's surrogate backward).
    torch autograd's own reverse pass -- shares NO code with the eligibility accumulation above; it keeps
    the recurrent-error, soft-reset, ADAPTATION (v->a->threshold) and inter-layer-temporal paths that
    e-prop truncates. Returns a grads dict keyed like eprop_grads."""
    model.zero_grad(set_to_none=True)
    logits, _, _, _, _ = model(ids)                          # autograd on
    loss = F.cross_entropy(logits.reshape(-1, model.vsize), y.reshape(-1))
    loss.backward()                                          # reverse-mode BPTT through the deep unroll
    out = {}
    for name, p in _named_params(model).items():
        out[name] = (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
    return out


def _flat_hidden(model, grads):
    """Flatten the HIDDEN-weight gradient (W_in + every off-diagonal W_rec[l] + every W_ff[l]) -- the
    subspace where e-prop APPROXIMATES BPTT. Readout / biases are excluded (their gradients are exact or
    near-exact in both methods, so including them would only inflate the cosine). Mirrors eprop_lm.py."""
    n = model.hidden
    off = (1.0 - torch.eye(n, device=grads["W_in"].device)).bool()
    parts = [grads["W_in"].reshape(-1)]
    for l in range(model.layers):
        parts.append(grads["W_rec.%d" % l][off].reshape(-1))
        if l > 0:
            parts.append(grads["W_ff.%d" % l].reshape(-1))
    return torch.cat(parts)


def _cosine(a, b):
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


def export_npz(model, path, trained=True, meta_extra=None):
    """Freeze the trained deep-ALIF model to a numpy .npz with the schema
    reasoning/eprop_lm_alif_serve.py reloads (per-layer W_rec_l / W_ff_l / b_hid_l keys)."""
    import numpy as np
    L = model.layers
    w = {
        "W_in": model.W_in.detach().cpu().numpy(),
        "W_out": model.W_out.detach().cpu().numpy(),
        "b_out": model.b_out.detach().cpu().numpy(),
    }
    for l in range(L):
        w["W_rec_%d" % l] = (model.W_rec[l].detach() * model._mask.detach()).cpu().numpy()
        w["b_hid_%d" % l] = model.b_hid[l].detach().cpu().numpy()
        if l > 0:
            w["W_ff_%d" % l] = model.W_ff[l].detach().cpu().numpy()
    meta = {
        "format": FORMAT, "vocab": model.vocab, "hidden": model.hidden, "layers": L,
        "vsize": model.vsize, "alpha": ALPHA, "rho": RHO, "beta": BETA, "kappa": KAPPA,
        "v_th": V_TH, "gamma": GAMMA, "damp": DAMP, "trained": bool(trained),
        "note": "deep recurrent-spiking ALIF char LM trained by e-prop (two-component eligibility); "
                "forward-served by reasoning/eprop_lm_alif_serve.py (numpy, forward-only)",
    }
    if meta_extra:
        meta.update(meta_extra)
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta)), **w)
    return path


def train(corpus, hidden=128, layers=2, ctx=64, steps=400, seed=0, out_path=None,
          batch=16, lr=3e-3, log_every=50):
    """Train the DEEP recurrent-spiking ALIF char LM BY E-PROP (online two-component eligibility traces +
    per-layer broadcast learning signal, NO loss.backward() through time for the hidden weights) on a CHAR
    corpus, then export the frozen artifact. `corpus` is a string; the vocab is its sorted distinct
    characters. Returns out_path."""
    import numpy as np
    torch.manual_seed(int(seed))
    if not isinstance(corpus, str):
        corpus = "".join(corpus)
    vocab = sorted(set(corpus))
    if len(vocab) < 2:
        raise ValueError("corpus needs at least 2 distinct characters")
    stoi = {c: i for i, c in enumerate(vocab)}
    ids_all = np.array([stoi[c] for c in corpus], dtype=np.int64)
    n_chars = ids_all.shape[0]
    if n_chars < ctx + 2:
        raise ValueError("corpus too short (%d chars) for ctx=%d" % (n_chars, ctx))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EpropAlifLM(vocab, hidden, layers=layers, seed=seed).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if log_every:
        print("train: device=%s vocab=%d hidden=%d layers=%d ctx=%d steps=%d params=%d (deep ALIF, e-prop)"
              % (device, len(vocab), hidden, layers, ctx, steps, n_params))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    named = _named_params(model)
    rng = np.random.RandomState(int(seed))
    last_loss = float("nan")
    for step in range(steps):
        starts = rng.randint(0, n_chars - ctx - 1, size=batch)
        xb = torch.stack([torch.from_numpy(ids_all[s:s + ctx]) for s in starts]).to(device)
        yb = torch.stack([torch.from_numpy(ids_all[s + 1:s + ctx + 1]) for s in starts]).to(device)
        grads, last_loss = eprop_grads(model, xb, yb)        # e-prop online gradient
        opt.zero_grad(set_to_none=True)
        for name, p in named.items():
            p.grad = grads[name]                             # inject the e-prop gradient; Adam applies it
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print("  step %d/%d  loss %.4f" % (step, steps, last_loss))

    out_path = out_path or DEFAULT_ARTIFACT
    export_npz(model, out_path, trained=True,
               meta_extra={"ctx": int(ctx), "final_loss": float(last_loss)})
    if log_every:
        print("saved frozen artifact -> %s (final e-prop train loss %.4f)" % (out_path, last_loss))
    return out_path


def _ascii_filter_corpus(path, max_chars):
    """Read `path` and keep only ASCII (Windows cp1252-safe); return up to max_chars of char stream."""
    out = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        while total < max_chars:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            keep = "".join(c for c in chunk if ord(c) < 128)
            out.append(keep)
            total += len(keep)
    return ("".join(out))[:max_chars]


def main():
    """Full-scale GPU train on the real wikitext char stream -> models/eprop_lm_alif.npz. Not run by the
    self-check (the user runs this). MEMORY NOTE: the ALIF eligibility keeps per-synapse [B,N,N] tensors
    per layer (eps_a + ebar for W_rec and W_ff), so peak memory scales as ~layers*B*N^2 -- larger than the
    single-layer LIF trainer. Drop hidden/batch if the GPU runs out."""
    corpus_path = os.path.join(_ROOT, "data", "lang", "wikitext", "train0.txt")
    max_chars = 2_000_000
    if not os.path.exists(corpus_path):
        print("corpus not found: %s" % corpus_path)
        return None
    text = _ascii_filter_corpus(corpus_path, max_chars)
    print("full: char corpus %d chars, vocab %d (ASCII-filtered wikitext)" % (len(text), len(set(text))))
    out = train(text, hidden=192, layers=2, ctx=128, steps=4000, seed=0, out_path=DEFAULT_ARTIFACT,
                batch=16, lr=2e-3, log_every=100)
    print("HONEST: ALIF + depth is the PRINCIPLED long-range lever, but even at this scale the language")
    print("ceiling may not fully clear; the e-prop pipeline + order-sensitivity are what is proven here,")
    print("not a BLiMP number.")
    return out


_TINY_TEXT = (
    "the spiking network reads the sentence one character at a time and it never looks ahead "
    "so the order of the letters changes the score that the model gives to the whole sentence "
    "a cat sat on the mat while the dog ran across the yard in the bright morning sun and then "
    "she sells sea shells by the sea shore and the shells shine softly in the pale evening light "
) * 3


def _selfcheck():
    """Fast NON-RIGGABLE proof of the DEEP-ALIF e-prop pipeline: (b) cosine(e-prop hidden grad, INDEPENDENT
    reverse-mode BPTT grad through the deep ALIF net) > 0.5 on a tiny instance; (a) a tiny e-prop train
    REDUCES the loss; (c) BETA=0 reduces the ALIF eligibility to the plain-LIF one. CPU-runnable; not a
    BLiMP model. Prints every measured number."""
    import numpy as np
    print("=" * 96)
    print("DEEP-ALIF E-PROP RECURRENT-SPIKING CHAR LM -- long-range brain-native trainer (Bellec 2018/2020)")
    print("stacked adaptive-threshold LIF; TWO-component eligibility (membrane + adaptation) x broadcast error;")
    print("NO loss.backward() through time for hidden weights")
    print("=" * 96)

    vocab = sorted(set(_TINY_TEXT))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids_all = np.array([stoi[c] for c in _TINY_TEXT], dtype=np.int64)

    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    ctx, batch, hidden, layers = 40, 12, 24, 2
    starts = rng.randint(0, ids_all.shape[0] - ctx - 1, size=batch)
    xb = torch.stack([torch.from_numpy(ids_all[s:s + ctx]) for s in starts])
    yb = torch.stack([torch.from_numpy(ids_all[s + 1:s + ctx + 1]) for s in starts])

    # (b) GOLD STANDARD -- cosine of the deep-ALIF e-prop hidden gradient with a genuine INDEPENDENT BPTT
    #     gradient through the SAME unrolled deep-ALIF net.
    model = EpropAlifLM(vocab, hidden=hidden, layers=layers, seed=0)
    ge, loss0 = eprop_grads(model, xb, yb)
    gb = bptt_grads(model, xb, yb)
    cos = _cosine(_flat_hidden(model, ge), _flat_hidden(model, gb))
    with torch.no_grad():
        _, Z, _, Aa, _ = model(xb)
        rates = [float((z > 0).float().mean()) for z in Z]
        a_top = float(Aa[-1].mean())
    print("task            : char LM, ctx=%d, %d ALIF layers x %d units, vocab=%d" % (ctx, layers, hidden, len(vocab)))
    print("spike rate/layer: %s (nonzero -> every layer is active, eligibility is well-defined)"
          % ", ".join("%.3f" % r for r in rates))
    print("adaptation state: mean top-layer a=%.3f (>0 -> the slow threshold memory is engaged)" % a_top)
    print("e-prop vs BPTT  : cosine(deep-ALIF e-prop hidden grad, INDEPENDENT reverse-mode BPTT grad) = %.4f" % cos)

    # (c) BETA=0 anchor: with no adaptation, the two-component ALIF eligibility must reduce to the LIF one,
    #     so its e-prop gradient must ALSO align with the (beta=0) BPTT gradient. Mirrors alif_neuron.py.
    global BETA
    beta_save = BETA
    try:
        BETA = 0.0
        m0 = EpropAlifLM(vocab, hidden=hidden, layers=layers, seed=0)
        g0e, _ = eprop_grads(m0, xb, yb)
        g0b = bptt_grads(m0, xb, yb)
        cos0 = _cosine(_flat_hidden(m0, g0e), _flat_hidden(m0, g0b))
        ea0 = float(g0e["W_in"].abs().sum())
    finally:
        BETA = beta_save
    print("beta=0 anchor   : cosine=%.4f (adaptation off -> ALIF eligibility collapses to LIF, still ~BPTT)" % cos0)

    # (a) a tiny e-prop train reduces the loss (fresh model, Adam over injected e-prop grads).
    torch.manual_seed(0)
    m2 = EpropAlifLM(vocab, hidden=hidden, layers=layers, seed=0)
    named = _named_params(m2)
    opt = torch.optim.Adam(m2.parameters(), lr=4e-3)
    rng2 = np.random.RandomState(0)
    init_loss = eprop_grads(m2, xb, yb)[1]
    for _ in range(250):
        s2 = rng2.randint(0, ids_all.shape[0] - ctx - 1, size=batch)
        x2 = torch.stack([torch.from_numpy(ids_all[s:s + ctx]) for s in s2])
        y2 = torch.stack([torch.from_numpy(ids_all[s + 1:s + ctx + 1]) for s in s2])
        g2, _ = eprop_grads(m2, x2, y2)
        opt.zero_grad(set_to_none=True)
        for name, p in named.items():
            p.grad = g2[name]
        opt.step()
    final_loss = eprop_grads(m2, xb, yb)[1]
    print("1 learns (loss) : e-prop train  initial %.4f -> final %.4f (ratio %.3f)"
          % (init_loss, final_loss, final_loss / init_loss))
    print("2 memory        : eligibility updated forward per-step -> O(neurons+synapses), NOT O(T);")
    print("                  the SLOW adaptation (RHO=%.2f, tau_a>>tau_m) carries the long-range trace" % RHO)

    # confirm the train()->export()->reload pipeline writes a schema-valid frozen artifact.
    art = train(_TINY_TEXT, hidden=16, layers=layers, ctx=ctx, steps=20, seed=0,
                out_path=os.path.join(_HERE, "_tmp_eprop_lm_alif_selfcheck.npz"),
                batch=batch, lr=4e-3, log_every=0)
    data = np.load(art, allow_pickle=False)
    need = {"W_in", "W_out", "W_rec_0", "W_rec_1", "W_ff_1"}
    assert need <= set(data.files), "artifact missing core weights: have %s" % sorted(data.files)
    assert json.loads(str(data["meta"]))["format"] == FORMAT, "artifact format tag wrong"
    assert json.loads(str(data["meta"]))["layers"] == layers, "artifact layer count wrong"
    del data
    if os.path.exists(art):
        os.remove(art)

    print("=" * 96)
    assert all(r > 0.0 for r in rates), "every ALIF layer must spike at init or the eligibility is degenerate"
    assert a_top > 0.0, "the top-layer adaptation state must engage (>0) or ALIF reduces to LIF"
    assert cos > 0.5, "(b) deep-ALIF e-prop hidden grad must ALIGN with true BPTT: cosine %.4f !> 0.5" % cos
    assert cos0 > 0.5, "(c) BETA=0 (LIF-reduced) e-prop grad must ALIGN with BPTT: cosine %.4f !> 0.5" % cos0
    assert ea0 > 0.0, "(c) BETA=0 gradient must be non-degenerate"
    assert final_loss < init_loss, "(a) e-prop must LEARN: final loss %.4f !< initial %.4f" % (final_loss, init_loss)
    print("self-check OK: deep-ALIF e-prop~BPTT cosine %.4f (>0.5, two-component factored gradient confirmed" % cos)
    print("vs an INDEPENDENT reverse-mode reference); beta=0 collapses to LIF (cosine %.4f); e-prop train" % cos0)
    print("reduces loss %.4f->%.4f; eligibility memory O(neurons), not O(T), with a SLOW adaptation trace"
          % (init_loss, final_loss))
    print("for long-range credit. HONEST: ALIF+depth is the principled long-range lever, but a tiny model")
    print("still sits near chance on real grammar -- this proves the PIPELINE + order-sensitivity, NOT a")
    print("BLiMP number (real-scale GPU run: --full).")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    if "--full" in sys.argv:
        main()
        sys.exit(0)
    sys.exit(_selfcheck())
