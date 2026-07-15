r"""
Zero-dep forward SERVER for a torch-trained DEEP spiking predictive-coding cortex (`snn_pcnet/1`).

Honest framing (read this first). The 2-level seed cortex/predictive_coding.py is a small Rao-Ballard
SPIKING coder (signed ON/OFF LIF error populations, graded cause readouts, spiking settling, Hebbian-
on-spikes). This module is the matching PURE-STDLIB core SERVER for the DEEP version of that same
network: an L-level (L >= 2) generative hierarchy TRAINED in the torch tier (research/pc_cortex_torch.py:
GPU-batched local predictive-coding -- top-down generative weights, bottom-up precision-weighted error,
local Hebbian weight updates, NO global backprop) and EXPORTED as an `snn_pcnet/1` JSON artifact. Here
we LOAD that artifact and replay the SPIKING SETTLING forward with NO torch, NO numpy -- forward-serve
only, no training. It is the predictive_coding analogue of cerebellum/hjepa_serve.py one architecture-
class over: a spiking generative hierarchy instead of a feedforward H-JEPA.

The forward replicated here is the seed's spiking settling made deep. With L cause levels r_1..r_L and
L generative matrices W_1..W_L (W_1: d_in x d_1 predicts the input x from r_1; W_l (l>=2): d_{l-1} x d_l
predicts r_{l-1} from r_l), one settling step is:

    e_0 = x    - W_1 @ r_1                                  (input prediction error)
    e_l = r_l  - W_{l+1} @ r_{l+1}   for l = 1 .. L-1        (level-l prediction error; r_L has none above)

each e_l encoded by a signed ON/OFF LIF pair (membrane leak, threshold THETA, reset-by-subtraction ->
discrete spike in {-1, 0, +1}); then the causes integrate the error SPIKES:

    r_1 += lr * ( W_1^T @ s_0 - s_1 )
    r_j += lr * ( W_j^T @ s_{j-1} - s_j )   for j = 2 .. L-1
    r_L += lr * ( W_L^T @ s_{L-1} )                          (top cause: driven ONLY by the level-(L-1) error
                                                             spikes -- the deep hierarchical message-passing
                                                             signature, generalizing the seed's r2 rule)

As the causes climb to explain the input the mismatch shrinks, the error neurons fall silent, and the
prediction-error SPIKE ACTIVITY DECAYS -- the network settles. infer(x) returns the settled top cause
r_L (the latent percept); classify(x) runs an optional linear readout (W r_L + b) argmax over r_L. This
is quantized/approximate spiking inference (residual reported), not exact biophysics, not cognition.

Trust boundary. load() NEVER trusts an unvalidated weight file: it checks the `snn_pcnet/1` format tag,
the positive-integer geometry (dims), that gen_weights has exactly L matrices each shaped
[dims[l-1]] x [dims[l]] and chaining through the hierarchy, that the precision vector has L entries, and
(if present) that the readout W is [n_classes] x [dims[L]] with a matching bias -- raising ValueError on
any mismatch. save()/round-trip is intentionally NOT provided -- this is a serve-only loader.

Pure stdlib (math, json + shared _mathutils.unit is NOT needed here; raw matvec inline), ASCII-only
prints, deterministic.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.pc_cortex_serve
"""
import json

PCNET_FORMAT = "snn_pcnet/1"


# ---- forward primitives (match research/pc_cortex_torch.py + the seed predictive_coding.py) ----------
def _matvec(m, v):
    """M @ v : m is rows x cols, v has length cols -> result length rows."""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def _matvec_t(m, v):
    """M^T @ v : m is rows x cols, v has length rows -> result length cols."""
    rows, cols = len(m), len(m[0])
    return [sum(m[i][j] * v[i] for i in range(rows)) for j in range(cols)]


# ---- weight-file validation (the trust boundary; never serve an unvalidated file) ------------------
def _pos_int(v, name):
    """A JSON value must be a positive integer (reject bool, which subclasses int)."""
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"bad {PCNET_FORMAT} file: {name} must be a positive integer (got {v!r})")
    return v


def _validate_matrix(name, m, rows, cols):
    """Validate that m is a row-major rows x cols matrix of numbers, returning it. Raises ValueError on
    any structural or shape mismatch."""
    if not (isinstance(m, list) and len(m) == rows):
        raise ValueError(f"bad {PCNET_FORMAT} file: {name} must have {rows} rows (got {len(m) if isinstance(m, list) else type(m).__name__})")
    for i, row in enumerate(m):
        if not (isinstance(row, list) and len(row) == cols):
            raise ValueError(f"bad {PCNET_FORMAT} file: {name}[{i}] must be a length-{cols} row")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in row):
            raise ValueError(f"bad {PCNET_FORMAT} file: {name}[{i}] must be all numbers")
    return m


class PcCortexServe:
    """A pure-stdlib forward server for a torch-trained deep spiking predictive-coding cortex
    (`snn_pcnet/1`).

    load() reads + trust-validates the artifact. infer(x) settles the spiking hierarchy on input x and
    returns the top cause r_L (the latent percept); classify(x) argmaxes the optional linear readout over
    r_L. dims / n_levels / theta / leak expose the config. Construct via PcCortexServe.load(path); the
    __init__ trusts already-validated components."""

    def __init__(self, *, dims, gen_weights, theta, leak, prec, steps, lr,
                 readout=None, classes=None):
        self._dims = [int(d) for d in dims]          # [d_in, d_1, ..., d_L]
        self._L = len(self._dims) - 1                 # number of cause levels
        self._W = gen_weights                         # [W_1, ..., W_L]; W_l is dims[l-1] x dims[l]
        self.theta = float(theta)
        self.leak = float(leak)
        self._prec = [float(p) for p in prec]         # L precisions, one per error level e_0..e_{L-1}
        self.steps = int(steps)
        self.lr = float(lr)
        self._readout = readout                       # (W, b) or None; W is n_classes x dims[L]
        self._classes = tuple(classes) if classes is not None else None

    # ---- construction / trust boundary -------------------------------------------------------------
    @classmethod
    def load(cls, path):
        """Load + validate an `snn_pcnet/1` JSON artifact, returning a ready PcCortexServe. Validates the
        format tag, the dims geometry (>= 2 cause levels, all positive), that gen_weights has exactly L
        matrices each shaped [dims[l-1]] x [dims[l]], that the precision vector has L entries, and (if
        present) the readout shape. Raises ValueError on any mismatch."""
        with open(path, encoding="ascii") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or obj.get("format") != PCNET_FORMAT:
            raise ValueError(f"bad {PCNET_FORMAT} file: missing/incorrect 'format' tag (expected {PCNET_FORMAT})")

        dims = obj.get("dims")
        if not (isinstance(dims, list) and len(dims) >= 3):
            raise ValueError(f"bad {PCNET_FORMAT} file: dims must be a list [d_in, d_1, ..., d_L] with L >= 2 cause levels")
        for i, d in enumerate(dims):
            _pos_int(d, f"dims[{i}]")
        L = len(dims) - 1

        gw = obj.get("gen_weights")
        if not (isinstance(gw, list) and len(gw) == L):
            raise ValueError(f"bad {PCNET_FORMAT} file: gen_weights must be a list of exactly {L} matrices (got {len(gw) if isinstance(gw, list) else type(gw).__name__})")
        gen = [_validate_matrix(f"gen_weights[{l}]", gw[l], dims[l], dims[l + 1]) for l in range(L)]

        prec = obj.get("prec")
        if prec is None:
            prec = [1.0] * L
        if not (isinstance(prec, list) and len(prec) == L and
                all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in prec)):
            raise ValueError(f"bad {PCNET_FORMAT} file: prec must be a list of {L} numbers (one per error level)")

        theta = obj.get("theta", 0.5)
        leak = obj.get("leak", 0.02)
        if not (isinstance(theta, (int, float)) and not isinstance(theta, bool) and theta > 0):
            raise ValueError(f"bad {PCNET_FORMAT} file: theta must be a positive number")
        if not (isinstance(leak, (int, float)) and not isinstance(leak, bool) and 0.0 <= leak < 1.0):
            raise ValueError(f"bad {PCNET_FORMAT} file: leak must be a number in [0, 1)")

        readout = None
        classes = obj.get("classes")
        ro = obj.get("readout")
        if ro is not None:
            if not (isinstance(ro, list) and len(ro) == 2):
                raise ValueError(f"bad {PCNET_FORMAT} file: readout must be a [W, b] pair")
            W, b = ro
            if not (isinstance(W, list) and len(W) >= 1 and isinstance(b, list) and len(b) == len(W)):
                raise ValueError(f"bad {PCNET_FORMAT} file: readout W/b malformed or mismatched")
            _validate_matrix("readout.W", W, len(W), dims[L])
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b):
                raise ValueError(f"bad {PCNET_FORMAT} file: readout bias must be all numbers")
            if classes is not None and not (isinstance(classes, list) and len(classes) == len(W)):
                raise ValueError(f"bad {PCNET_FORMAT} file: classes length must equal readout rows ({len(W)})")
            readout = (W, b)

        return cls(dims=dims, gen_weights=gen, theta=theta, leak=leak, prec=prec,
                   steps=int(obj.get("steps", 400)), lr=float(obj.get("lr", 0.05)),
                   readout=readout, classes=classes)

    # ---- config (read-only) ------------------------------------------------------------------------
    @property
    def dims(self):
        return list(self._dims)

    @property
    def n_levels(self):
        return self._L

    @property
    def in_dim(self):
        return self._dims[0]

    @property
    def top_dim(self):
        return self._dims[-1]

    @property
    def classes(self):
        return self._classes

    # ---- the spiking settling forward --------------------------------------------------------------
    def _lif_step(self, v_on, v_off, drive, k, theta, leak):
        """One LIF step for the signed ON/OFF neuron pair encoding one error component. ON integrates the
        positive part of the drive, OFF the negative; each leaks, crosses theta -> emits a discrete spike
        and resets by subtraction. Returns the signed spike in {-1, 0, +1}. Mutates v_on/v_off in place."""
        pos = drive if drive > 0.0 else 0.0
        neg = -drive if drive < 0.0 else 0.0
        v_on[k] = (1.0 - leak) * v_on[k] + pos
        s_on = 1 if v_on[k] >= theta else 0
        if s_on:
            v_on[k] -= theta
        v_off[k] = (1.0 - leak) * v_off[k] + neg
        s_off = 1 if v_off[k] >= theta else 0
        if s_off:
            v_off[k] -= theta
        return s_on - s_off

    def settle(self, x, steps=None, lr=None, trace=False):
        """Settle the causes r_1..r_L for input x by running the spiking network. Returns the settled
        causes [r_1, ..., r_L]; if trace=True, returns (causes, activity_trace) where activity_trace is
        the per-step total error-spike count (it falls as the network settles). Passing lr=0.0 reads the
        error spikes WITHOUT moving the causes (a settled/ablated readout)."""
        if len(x) != self._dims[0]:
            raise ValueError(f"settle: x length {len(x)} != in_dim {self._dims[0]}")
        steps = self.steps if steps is None else int(steps)
        lr = self.lr if lr is None else float(lr)
        L, dims, W, prec, theta, leak = self._L, self._dims, self._W, self._prec, self.theta, self.leak

        x = list(x)
        r = [[0.0] * dims[j + 1] for j in range(L)]          # r[0]=r_1 ... r[L-1]=r_L
        v_on = [[0.0] * dims[i] for i in range(L)]           # error membranes e_0..e_{L-1}
        v_off = [[0.0] * dims[i] for i in range(L)]
        act_trace = []

        for _ in range(steps + 1):                            # +1: final pass is the settled readout below
            # top-down predictions -> analog errors e_0..e_{L-1}
            errs = []
            pred0 = _matvec(W[0], r[0])                        # W_1 @ r_1 predicts x
            errs.append([x[k] - pred0[k] for k in range(dims[0])])
            for lvl in range(1, L):                           # e_l = r_l - W_{l+1} @ r_{l+1}
                pred = _matvec(W[lvl], r[lvl])
                errs.append([r[lvl - 1][k] - pred[k] for k in range(dims[lvl])])
            # encode each error into signed ON/OFF LIF spikes
            spikes = []
            for i in range(L):
                si = [self._lif_step(v_on[i], v_off[i], prec[i] * errs[i][k], k, theta, leak)
                      for k in range(dims[i])]
                spikes.append(si)
            act_trace.append(sum(1 for si in spikes for s in si if s))
            # integrate error spikes into the causes: r_j += lr * (W_j^T @ s_{j-1} - s_j), top has no s_L
            for j in range(L):
                up = _matvec_t(W[j], spikes[j])               # length dims[j+1]
                if j < L - 1:
                    s_above = spikes[j + 1]
                    r[j] = [r[j][m] + lr * (up[m] - s_above[m]) for m in range(dims[j + 1])]
                else:
                    r[j] = [r[j][m] + lr * up[m] for m in range(dims[j + 1])]
        return (r, act_trace) if trace else r

    def infer(self, x, steps=None, lr=None):
        """Settle the hierarchy on x and return the top cause r_L (the latent percept)."""
        return self.settle(x, steps=steps, lr=lr)[-1]

    def classify(self, x, steps=None, lr=None):
        """Argmax of the linear readout (W r_L + b) over the settled top cause. Returns the class name if
        `classes` was provided, else the integer index. Raises if the artifact carried no readout."""
        if self._readout is None:
            raise ValueError("this snn_pcnet/1 artifact has no readout head; classify() is unavailable")
        rL = self.infer(x, steps=steps, lr=lr)
        W, b = self._readout
        logits = [sum(W[c][m] * rL[m] for m in range(len(rL))) + b[c] for c in range(len(W))]
        idx = max(range(len(logits)), key=lambda c: logits[c])
        return self._classes[idx] if self._classes is not None else idx

    def reconstruct(self, causes):
        """Top-down reconstruction of the input from settled causes: x_hat = W_1 @ r_1."""
        return _matvec(self._W[0], causes[0])


# =========================================================================================================
# Self-check: build a TINY snn_pcnet/1 (L=3) by hand, serve it, and assert the settling forward matches an
# INDEPENDENT hand-rolled reference (raw arithmetic, not the served code path); the spiking settling is
# genuinely PREDICTIVE (error-spike activity + recon residual FALL vs an lr=0 frozen-cause control); spikes
# are discrete in {-1,0,1} with a real leak and the core imports NO torch/numpy; and the loader rejects
# bad-format and wrong-shape files. Deterministic, ASCII-only.
# =========================================================================================================
def _build_tiny_blob():
    """A hand-set snn_pcnet/1 dict: dims [5, 4, 3, 2] (L=3 cause levels -> exercises the MIDDLE-level cause
    update r_2 += lr*(W_2^T s_1 - s_2), the deep generalization the seed's L=2 never hits), a 2-class
    readout over the top cause. Weights small so the hierarchy is settleable."""
    def grid(rows, cols, base):
        return [[round(base * (0.35 * (i + 1) - 0.25 * (j + 1) + 0.2), 4) for j in range(cols)] for i in range(rows)]
    dims = [5, 4, 3, 2]
    gen_weights = [grid(dims[l], dims[l + 1], 1.0) for l in range(3)]   # W_1 5x4, W_2 4x3, W_3 3x2 (O(1) entries)
    readout = [[[0.3, -0.2], [-0.1, 0.4]], [0.0, 0.0]]                  # 2 classes x top_dim 2
    return {
        "format": PCNET_FORMAT,
        "dims": dims,
        "gen_weights": gen_weights,
        "prec": [1.0, 1.0, 1.0],
        "theta": 0.5, "leak": 0.02,
        "steps": 300, "lr": 0.05,
        "classes": ["A", "B"], "readout": readout,
        "meta": {"trained_by": "self-check", "unverified_on_hardware": True},
    }


def main():
    import os
    import sys
    import tempfile

    blob = _build_tiny_blob()
    dims = blob["dims"]
    W = blob["gen_weights"]
    THETA, LEAK = blob["theta"], blob["leak"]
    prec = blob["prec"]
    L = len(dims) - 1

    # ---- independent reference settling (raw arithmetic; NOT the served code path) ----
    def r_mv(m, v):
        return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]

    def r_mvt(m, v):
        rows, cols = len(m), len(m[0])
        return [sum(m[i][j] * v[i] for i in range(rows)) for j in range(cols)]

    def r_settle(x, steps, lr):
        r = [[0.0] * dims[j + 1] for j in range(L)]
        von = [[0.0] * dims[i] for i in range(L)]
        voff = [[0.0] * dims[i] for i in range(L)]
        trace = []
        for _ in range(steps + 1):
            errs = [[x[k] - r_mv(W[0], r[0])[k] for k in range(dims[0])]]
            for lvl in range(1, L):
                pred = r_mv(W[lvl], r[lvl])
                errs.append([r[lvl - 1][k] - pred[k] for k in range(dims[lvl])])
            spikes = []
            for i in range(L):
                si = []
                for k in range(dims[i]):
                    drive = prec[i] * errs[i][k]
                    pos = drive if drive > 0 else 0.0
                    neg = -drive if drive < 0 else 0.0
                    von[i][k] = (1 - LEAK) * von[i][k] + pos
                    son = 1 if von[i][k] >= THETA else 0
                    if son:
                        von[i][k] -= THETA
                    voff[i][k] = (1 - LEAK) * voff[i][k] + neg
                    soff = 1 if voff[i][k] >= THETA else 0
                    if soff:
                        voff[i][k] -= THETA
                    si.append(son - soff)
                spikes.append(si)
            trace.append(sum(1 for si in spikes for s in si if s))
            for j in range(L):
                up = r_mvt(W[j], spikes[j])
                if j < L - 1:
                    r[j] = [r[j][m] + lr * (up[m] - spikes[j + 1][m]) for m in range(dims[j + 1])]
                else:
                    r[j] = [r[j][m] + lr * up[m] for m in range(dims[j + 1])]
        return r, trace, spikes

    def close(a, b, tol=1e-9):
        return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))

    # a representable input the hierarchy can explain: x = W_1 @ (W_2 @ (W_3 @ r_top))
    r_top = [0.8, -0.5]
    tmp_up = list(r_top)
    for lvl in range(L - 1, -1, -1):
        tmp_up = r_mv(W[lvl], tmp_up)
    x = tmp_up

    # ---- write + load the artifact through the trust boundary ----
    tmp = os.path.join(tempfile.gettempdir(), "snn_pcnet1_selfcheck.json")
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(blob, f)
    srv = PcCortexServe.load(tmp)
    os.remove(tmp)

    steps, lr = 300, 0.05

    # ---- (1) served settling matches the independent reference for the causes AND the activity trace ----
    causes_srv, act_srv = srv.settle(x, steps=steps, lr=lr, trace=True)
    causes_ref, act_ref, last_spikes = r_settle(x, steps, lr)
    causes_ok = all(close(causes_srv[j], causes_ref[j]) for j in range(L))
    trace_ok = act_srv == act_ref
    infer_ok = close(srv.infer(x, steps=steps, lr=lr), causes_ref[-1])

    # ---- (2) genuinely predictive: error-spike activity + recon residual FALL vs an lr=0 frozen control ----
    def win(t, lo, hi):
        seg = t[lo:hi]
        return sum(seg) / len(seg)
    act_early = win(act_srv, 0, 40)
    act_late = win(act_srv, len(act_srv) - 40, len(act_srv))
    xhat = srv.reconstruct(causes_srv)
    recon = sum((x[k] - xhat[k]) ** 2 for k in range(dims[0])) / dims[0]
    causes_ab, act_ab = srv.settle(x, steps=steps, lr=0.0, trace=True)     # causes frozen at 0
    act_late_ab = win(act_ab, len(act_ab) - 40, len(act_ab))
    xhat_ab = srv.reconstruct(causes_ab)
    recon_ab = sum((x[k] - xhat_ab[k]) ** 2 for k in range(dims[0])) / dims[0]
    settles_ok = act_late < 0.5 * act_early and act_late < act_late_ab and recon < 0.5 * recon_ab

    # ---- (3) genuinely spiking: last spikes discrete in {-1,0,1}, real leak, no torch/numpy in core ----
    spikes_discrete = all(s in (-1, 0, 1) for si in last_spikes for s in si)
    leak_ok = 0.0 < srv.leak < 1.0
    pure_ok = "torch" not in sys.modules and "numpy" not in sys.modules

    # ---- (4) classify + config exposure ----
    label = srv.classify(x, steps=steps, lr=lr)
    classify_ok = label in ("A", "B")
    config_ok = (srv.dims == [5, 4, 3, 2] and srv.n_levels == 3 and srv.in_dim == 5 and srv.top_dim == 2
                 and srv.classes == ("A", "B"))

    # ---- (5) the loader rejects bad-format and wrong-shape files ----
    def _expect_valueerror(mutate):
        bad = json.loads(json.dumps(blob))
        mutate(bad)
        p = os.path.join(tempfile.gettempdir(), "snn_pcnet1_bad.json")
        with open(p, "w", encoding="ascii") as fh:
            json.dump(bad, fh)
        try:
            PcCortexServe.load(p)
            return False
        except ValueError:
            return True
        finally:
            os.remove(p)

    reject_format_ok = _expect_valueerror(lambda b: b.update(format="snn_pcnet/2"))
    reject_shape_ok = _expect_valueerror(lambda b: b["gen_weights"][1].append([0.0, 0.0, 0.0]))  # W_2 gains a bad row
    reject_count_ok = _expect_valueerror(lambda b: b["gen_weights"].pop())                        # too few matrices
    reject_readout_ok = _expect_valueerror(lambda b: b["readout"][0].append([0.0, 0.0]))          # 3-class W vs 2-class bias

    print("=" * 78)
    print("PC-CORTEX SERVE -- zero-dep forward server for a torch-trained deep spiking snn_pcnet/1")
    print("=" * 78)
    print(f"config      : dims={srv.dims} levels={srv.n_levels} in={srv.in_dim} top={srv.top_dim} classes={list(srv.classes)}")
    print(f"settling    : served causes+trace == independent reference: causes={causes_ok} trace={trace_ok} infer={infer_ok}")
    print(f"predictive  : error-spike act {act_early:.2f}->{act_late:.2f} (frozen {act_late_ab:.2f}); "
          f"recon {recon:.4f} vs frozen {recon_ab:.4f}: {settles_ok}")
    print(f"spiking     : spikes in {{-1,0,1}}={spikes_discrete} leak={srv.leak} pure(no torch/numpy)={pure_ok}")
    print(f"readout     : classify(x)={label!r} in classes: {classify_ok}")
    print(f"trust       : reject bad-format={reject_format_ok} wrong-shape={reject_shape_ok} "
          f"wrong-count={reject_count_ok} bad-readout={reject_readout_ok}")
    print("-" * 78)

    assert causes_ok, "served settling causes must match the independent hand-rolled reference"
    assert trace_ok, "served error-spike activity trace must match the independent reference"
    assert infer_ok, "infer(x) must equal the reference top cause r_L"
    assert settles_ok, "spiking settling must DROP error-spike activity and recon residual vs a frozen-cause control"
    assert spikes_discrete, "error spikes must be discrete in {-1, 0, +1}"
    assert leak_ok, "the LIF membrane must have a genuine leak in (0, 1)"
    assert pure_ok, "the core server must import NO torch and NO numpy"
    assert classify_ok, "classify(x) must return one of the readout classes"
    assert config_ok, "dims / n_levels / in_dim / top_dim / classes must be exposed"
    assert reject_format_ok, "load must raise ValueError on a bad format tag"
    assert reject_shape_ok, "load must raise ValueError on a wrong-shape generative matrix"
    assert reject_count_ok, "load must raise ValueError on the wrong number of generative matrices"
    assert reject_readout_ok, "load must raise ValueError on a readout W/bias mismatch"

    print("self-check OK: snn_pcnet/1 deep spiking predictive-coding forward served pure-stdlib, matches an "
          "independent reference settling, is genuinely predictive (activity+residual fall) and spiking, with a "
          "validating loader")


if __name__ == "__main__":
    main()
