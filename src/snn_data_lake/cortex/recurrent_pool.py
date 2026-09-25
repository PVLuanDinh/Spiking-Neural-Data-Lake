"""G1(b) recurrent pools: the four research-tier reservoirs as governed streaming pools.

The frozen text is specs/022-unified-brain-arc/evidence/g1b-pool-selection.md section 2 (commit
23c7e97). This module holds the pool contract of section 2.2 (the shared encoder, the adaptation
window, the four forms, the startup diagnostic), the candidates of section 2.3 (g1 LSM, g2 ESN-LSM,
g3 SORN, g4 RBN, each with its source's exact draws), the manifest and cost of section 2.4, and the
machinery of R-b0 (parity with each source), R-b3 (determinism by construction), R-b4's pool-noise
red and R-b5 (the forms do what they claim). Every red is constructed by `main()`. The reference
readout, the harness arms and the campaign are cortex/memory_diagnostic.py's. Nothing here builds a
brain or touches the lake.

Arithmetic (section 2.2): float64 numpy; no BLAS product in any per-step or per-percept pool
computation; every reduction is a sequential sum, taken as np.cumsum's last element along a declared
axis. Where R-b0 claims bit parity the order is the source's. SORN's synaptic normalization calls
the source's builtin sum(), which CPython has computed with Neumaier compensation since 3.12, so
`_builtin_sum_rows` reproduces that algorithm rather than a plain running sum (a plain sum differs
from it on about half of all rows). Construction-time eigvals runs inside threadpool_limits(1).

One module on purpose: section 2.11 lands one governed pool component, and its size is the size of
that contract.
"""

# Ports, not consolidations: reasoning/spiking_ssm.py, reservoir_eprop.py, sorn.py and
# random_boolean_network.py stay untouched in this arc (section 2.9). None of them can stream (a
# scalar or symbolic drive, a reset per sequence, sizes fixed at module scope, no input port on the
# RBN), so each is ported to the pool contract and R-b0 holds every port to its source. The losing
# ports are deleted after the selection record; each source's retire/promote/exempt row is its own
# P2 opening (section 2.13).
# distinct-from: src/snn_data_lake/cortex/arealization.py -- a GRN consensus network settled once
#   into developmental region labels; nothing is driven by a percept or carried between percepts
# distinct-from: src/snn_data_lake/cortex/prefrontal_executive.py -- its LIF pool holds a goal with
#   NO external input (section 1.2 names that a different job); no stimulus drive, no forms
# distinct-from: src/snn_data_lake/cortex/recursive_tom.py -- a theory-of-mind reading over recorded
#   rows; no recurrent state at all
# distinct-from: src/snn_data_lake/cortex/report_order_probe.py -- a report-order probe reading over
#   constructed rows; no recurrent state at all
# distinct-from: src/snn_data_lake/reasoning/cellular_neural_network.py -- a 3x3 cellular-network
#   convolution on a grid; no streaming pool, no percept drive
# distinct-from: src/snn_data_lake/reasoning/rsnn_delays_torch.py -- its verify_transfer checks an
#   exported delay-RSNN against its own torch model; R-b0 here holds a pool port to its source
# distinct-from: src/snn_data_lake/reasoning/hjepa_arm_torch.py -- an H-JEPA export->core bridge
#   check; no recurrent pool
# distinct-from: src/snn_data_lake/reasoning/jepa_scaffold_torch.py -- a JEPA export->core bridge
#   check; no recurrent pool
# distinct-from: src/snn_data_lake/substrate/eprop_transport_gate.py -- AmbiguousDisposition is a
#   gate's refusal type; nothing here steps a network
# distinct-from: src/snn_data_lake/reasoning/reservoir_eprop.py -- `_run` is g2's parity SOURCE,
#   read (never edited) by check_r_b0_esn; the port streams where the source runs whole batches

import copy
import inspect
import itertools
import json
import os
import random
import statistics
import sys
import time
import zlib

import numpy as np

N = 256
N_IN = 64
WINDOW = 256
S_SETTLE = 64
FORMS = ("carried", "reset", "settled", "front_end")
CANDIDATES = ("lsm", "esn", "sorn", "rbn")
INDEX = {name: i + 1 for i, name in enumerate(CANDIDATES)}
LABEL = {"lsm": "g1", "esn": "g2", "sorn": "g3", "rbn": "g4"}

# Section 2.3's grids, factor order and levels exactly as written (the last factor varies fastest).
GRIDS = {
    "lsm": (
        ("k", (8, 32)),
        ("w", (0.25, 0.5, 1.0)),
        ("lambda", (0.5, 0.7, 0.9)),
        ("g", (1, 2, 4, 8)),
        ("T_in", (1, 2, 4, 8)),
    ),
    "esn": (("rho", (0.5, 0.9, 1.2)), ("g", (1, 2, 4, 8)), ("T_in", (1, 2, 4, 8))),
    "sorn": (
        ("plasticity", ("off", (0.001, 0.001), (0.001, 0.01), (0.004, 0.001), (0.004, 0.01))),
        ("H_IP", (0.05, 0.1, 0.2)),
        ("g", (1, 2, 4, 8)),
        ("T_in", (1, 2, 4, 8)),
    ),
    "rbn": (
        ("K", (1, 2, 3)),
        ("p", (0.5, 0.7)),
        ("f", (0.1, 0.25, 0.5)),
        ("mode", ("xor", "clamp")),
        ("tau", (-0.1, 0, 0.1)),
        ("T_in", (1, 2, 4, 8)),
    ),
}
MANIFEST_SIZE = 32
COST_BOUND_MS = 25.0
DEV_SEED = 100  # section 2.10's development seeds: the mechanism checks' W_in and realizations

LSM_THRESH = 1.0
SOURCE_W = 0.5  # spiking_ssm's recurrent weight bound, which g2's drawing keeps
ESN_K = 8
ESN_ALPHA = 0.90
ESN_V_TH = 0.6
SORN_N_E = 213
SORN_N_I = 43
SORN_P_EE = 0.10
SORN_P_IE = 0.20
SORN_P_EI = 0.30
SORN_T_E_MAX = 0.5
SORN_T_I_MAX = 0.8
SORN_IN_PER_SYM = 8
SORN_N_SYMBOLS = 8  # the source's alphabet at SHARED_LEN 4; it shapes only the consumed shuffle

DIAG_SEED = 777
DIAG_FIXTURES = 16
DIAG_BURN = 256
DIAG_MEAN = 64
DIAG_SETTLES = (0, 16, 64)
DIAG_FLAG = 0.05

R_B3_PERCEPTS = 512
R_B3_BLAS = (1, 24)
R_B3_TORCH = (1, 4)
R_B3_MULTI = 100
# g1 and g2 are section 2.7's witnesses; g3 and g4's are declared here (the text names none).
R_B3_WITNESS = {
    "lsm": {"k": 32, "w": 1.0, "lambda": 0.9, "g": 8, "T_in": 8},
    "esn": {"rho": 1.2, "g": 8, "T_in": 8},
    "sorn": {"plasticity": (0.004, 0.01), "H_IP": 0.1, "g": 2, "T_in": 1},
    "rbn": {"K": 2, "p": 0.5, "f": 0.25, "mode": "xor", "tau": 0, "T_in": 1},
}
R_B5_PERSISTENCE_WITNESS = {"k": 8, "w": 0.5, "lambda": 0.9, "g": 2, "T_in": 1}
# R-b5 (i)'s skip-a-carrier red: per candidate, a declared configuration and the carrier whose
# skipped reset must leak prefix history through both Reset and Settled's 64 uncounted steps.
R_B5_ERASURE_WITNESS = {
    "lsm": ({"k": 32, "w": 1.0, "lambda": 0.9, "g": 8, "T_in": 8}, "v"),
    "esn": ({"rho": 1.2, "g": 8, "T_in": 8}, "v"),
    "sorn": ({"plasticity": "off", "H_IP": 0.1, "g": 1, "T_in": 1}, "x_e"),
    "rbn": ({"K": 3, "p": 0.5, "f": 0.1, "mode": "xor", "tau": 0, "T_in": 1}, "state"),
}
R_B5_FREEZE_WITNESS = {"plasticity": (0.004, 0.01), "H_IP": 0.1, "g": 2, "T_in": 1}
G2_TOLERANCE = 1e-12
G2_MARGIN = 1e-4


# ---- the shared encoder and the seeds ------------------------------------------------------------


def w_in(recording_seed):
    """W_in (256 x 64), entries N(0, 1) / 8 from default_rng(recording_seed * 31 + 7)."""
    return np.random.default_rng(recording_seed * 31 + 7).standard_normal((N, N_IN)) / 8


def drive(weights, x):
    """u = W x as a sequential sum over the inputs, left to right from 0.0 (section 2.2)."""
    weights = np.asarray(weights, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != x.shape[0]:
        raise ValueError(f"REFUSED: drive shapes {weights.shape} and {x.shape}")
    terms = np.concatenate([np.zeros((weights.shape[0], 1)), weights * x], axis=1)
    return np.cumsum(terms, axis=1)[:, -1]


def realization_seed(candidate, recording_seed):
    return recording_seed * 31 + 11 + INDEX[candidate]


# ---- the grids and the manifest (section 2.4) ----------------------------------------------------


def grid(candidate):
    factors = GRIDS[candidate]
    names = [name for name, _ in factors]
    return [dict(zip(names, levels)) for levels in itertools.product(*(v for _, v in factors))]


def canonical(candidate, config):
    """The grid's own entry equal to `config` (JSON lists read as tuples), else refused."""
    wanted = {key: tuple(v) if isinstance(v, list) else v for key, v in dict(config).items()}
    for entry in grid(candidate):
        if entry == wanted:
            return entry
    raise ValueError(f"REFUSED: {candidate} configuration outside section 2.3's grid: {config}")


def draw_order(candidate):
    configs = grid(candidate)
    return random.Random(2026 + INDEX[candidate]).sample(configs, len(configs))


def crc_json(value):
    return zlib.crc32(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii"))


def draw_order_crc32():
    return {candidate: crc_json(draw_order(candidate)) for candidate in CANDIDATES}


def screen(candidate, cost_ms, *, size=MANIFEST_SIZE, bound=COST_BOUND_MS):
    """Walk the draw order; a configuration whose cost exceeds `bound` is dropped and disclosed,
    and the first `size` survivors are kept (all of them, flagged short, if fewer survive)."""
    kept, costs, dropped = [], [], []
    for config in draw_order(candidate):
        if len(kept) == size:
            break
        value = float(cost_ms(config))
        if value > bound:
            dropped.append({"config": config, "cost_ms": value})
        else:
            kept.append(config)
            costs.append(value)
    return {
        "candidate": candidate,
        "configs": kept,
        "costs_ms": costs,
        "dropped": dropped,
        "short": len(kept) < size,
    }


def manifest(screens):
    body = {candidate: screens[candidate]["configs"] for candidate in CANDIDATES}
    return {
        "configs": body,
        "crc32": crc_json(body),
        "short": [c for c in CANDIDATES if screens[c]["short"]],
    }


# ---- the candidates' drawings (section 2.3) ------------------------------------------------------


def _lsm_draw(seed, n, k, w):
    """spiking_ssm.build_reservoir's call order at size n: n uniform(-1, 1) input draws consumed
    first, then per unit k pairs (randrange(n), uniform(-w, w))."""
    rng = random.Random(seed)
    consumed = [rng.uniform(-1, 1) for _ in range(n)]
    rows = [[(rng.randrange(n), rng.uniform(-w, w)) for _ in range(k)] for _ in range(n)]
    pre = np.array([[j for j, _ in row] for row in rows], dtype=np.int64)
    weights = np.array([[x for _, x in row] for row in rows], dtype=np.float64)
    return consumed, pre, weights


# distinct-from: cortex/everyday_battery.py -- sites read (a NamedTuple, prior_drift_harness's host recorder, episodic_recall's unbind); none limits BLAS threads
def _blas_threads(limit):
    """threadpoolctl's limit on every loaded BLAS: section 2.2's eigvals, R-b3 and the cost."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=limit)


def _esn_w_rec(seed, n, rho):
    """g1's drawing at k 8, w 0.5, densified [pre, post] in list order, zero diagonal, rescaled
    to spectral radius rho (reservoir_eprop.build_reservoir's recipe)."""
    _, pre, weights = _lsm_draw(seed, n, ESN_K, SOURCE_W)
    dense = np.zeros((n, n))
    for i in range(n):
        for p in range(ESN_K):
            dense[pre[i, p], i] += weights[i, p]
    np.fill_diagonal(dense, 0.0)
    with _blas_threads(1):
        raw = float(np.max(np.abs(np.linalg.eigvals(dense))))
    return dense * (rho / raw) if raw > 0 else dense


def _sorn_draw(rng, n_e, n_i, n_symbols=SORN_N_SYMBOLS):
    """sorn.build_net's exact call order at sizes (n_e, n_i), returning its lists; its builtin
    sum() calls stay builtin so the normalized rows are the source's to the bit."""
    ee_pre, ee_w = [], []
    for i in range(n_e):
        pre, w = [], []
        for j in range(n_e):
            if j != i and rng.random() < SORN_P_EE:
                pre.append(j)
                w.append(rng.random())
        if not pre:
            j = rng.randrange(n_e - 1)
            j = j + 1 if j >= i else j
            pre.append(j)
            w.append(rng.random())
        s = sum(w)
        ee_pre.append(pre)
        ee_w.append([x / s for x in w])
    ei = []
    for _ in range(n_e):
        pre = [k for k in range(n_i) if rng.random() < SORN_P_EI]
        wk = [rng.random() for _ in pre]
        s = sum(wk) or 1.0
        ei.append(list(zip(pre, [x / s for x in wk])))
    ie = []
    for _ in range(n_i):
        pre = [j for j in range(n_e) if rng.random() < SORN_P_IE]
        if not pre:
            pre = [rng.randrange(n_e)]
        wj = [rng.random() for _ in pre]
        s = sum(wj) or 1.0
        ie.append(list(zip(pre, [x / s for x in wj])))
    t_e = [rng.uniform(0, SORN_T_E_MAX) for _ in range(n_e)]
    t_i = [rng.uniform(0, SORN_T_I_MAX) for _ in range(n_i)]
    per = min(SORN_IN_PER_SYM, max(2, n_e // n_symbols))
    units = list(range(n_e))
    rng.shuffle(units)  # the source's input-block shuffle: consumed, unused by the pool
    in_units = [units[s * per : (s + 1) * per] for s in range(n_symbols)]
    return {
        "ee_pre": ee_pre,
        "ee_w": ee_w,
        "ei": ei,
        "ie": ie,
        "t_e": t_e,
        "t_i": t_i,
        "in_units": in_units,
    }


def _sorn_arrays(net, n_e, n_i):
    """Pack the source's lists into padded arrays, one row per post unit, in list order. An E row
    is [drive - T_E, W_EE terms..., -W_EI terms...]; an I row is [-T_I, W_IE terms...]."""
    me = max(len(pre) for pre in net["ee_pre"])
    mi = max((len(row) for row in net["ei"]), default=0)
    mj = max(len(row) for row in net["ie"])
    e_idx = np.zeros((n_e, 1 + me + mi), dtype=np.int64)
    e_w = np.zeros((n_e, 1 + me + mi))
    e_valid = np.zeros((n_e, 1 + me + mi), dtype=bool)
    for i in range(n_e):
        pre = net["ee_pre"][i]
        e_idx[i, 1 : 1 + len(pre)] = pre
        e_w[i, 1 : 1 + len(pre)] = net["ee_w"][i]
        e_valid[i, 1 : 1 + len(pre)] = True
        for q, (k, wk) in enumerate(net["ei"][i]):
            e_idx[i, 1 + me + q] = n_e + k
            e_w[i, 1 + me + q] = -wk  # r - w == r + (-w) exactly in IEEE arithmetic
            e_valid[i, 1 + me + q] = True
    i_idx = np.zeros((n_i, 1 + mj), dtype=np.int64)
    i_w = np.zeros((n_i, 1 + mj))
    i_valid = np.zeros((n_i, 1 + mj), dtype=bool)
    for k in range(n_i):
        for q, (j, wj) in enumerate(net["ie"][k]):
            i_idx[k, 1 + q] = j
            i_w[k, 1 + q] = wj
            i_valid[k, 1 + q] = True
    return {
        "e_idx": e_idx,
        "e_w": e_w,
        "e_valid": e_valid,
        "ee_len": np.array([len(pre) for pre in net["ee_pre"]], dtype=np.int64),
        "me": me,
        "i_idx": i_idx,
        "i_w": i_w,
        "i_valid": i_valid,
        "t_e": np.array(net["t_e"], dtype=np.float64),
        "t_i": np.array(net["t_i"], dtype=np.float64),
    }


def _sorn_reset_state(seed, h_ip, n_e, n_i):
    """One binary state from random.Random(seed + 2000), each unit active with probability H_IP,
    drawn E units first, then I."""
    rng = random.Random(seed + 2000)
    bits = np.array([rng.random() < h_ip for _ in range(n_e + n_i)], dtype=bool)
    return bits[:n_e], bits[n_e:]


def _rbn_draw(seed, n, k, p):
    """random_boolean_network.build_rbn's call order: wiring, then truth tables."""
    rng = random.Random(seed)
    inputs = [rng.sample(range(n), k) for _ in range(n)]
    tables = [[1 if rng.random() < p else 0 for _ in range(1 << k)] for _ in range(n)]
    return np.array(inputs, dtype=np.int64), np.array(tables, dtype=np.int64)


def _rbn_input_nodes(seed, n, f):
    return np.array(random.Random(seed + 1000).sample(range(n), n)[: round(f * n)], dtype=np.int64)


def _rbn_reset_state(seed, n):
    rng = random.Random(seed + 2000)
    return np.array([rng.randrange(2) for _ in range(n)], dtype=np.int64)


def _builtin_sum_rows(values, lengths):
    """CPython's builtin sum() over each row's first lengths[i] floats, left to right, vectorized
    across rows: the running total starts at the first term (0 + x), each later term updates a
    Neumaier compensation, and a nonzero finite compensation is added once at the end."""
    total = values[:, 0].copy()
    comp = np.zeros(len(values))
    for p in range(1, values.shape[1]):
        live = p < lengths
        x = values[:, p]
        t = total + x
        term = np.where(np.abs(total) >= np.abs(x), (total - t) + x, (x - t) + total)
        comp = np.where(live, comp + term, comp)
        total = np.where(live, t, total)
    return np.where((comp != 0.0) & np.isfinite(comp), total + comp, total)


# ---- the pools (section 2.2's contract, section 2.3's rules) -------------------------------------


def _crc_arrays(named, crc=0):
    for key in sorted(named):
        a = np.ascontiguousarray(named[key])
        crc = zlib.crc32(f"{key}|{a.dtype.str}|{a.shape}".encode("ascii"), crc)
        crc = zlib.crc32(a.tobytes(), crc)
    return crc


def _urandom_below(n):
    return int.from_bytes(os.urandom(8), "little") % n


def _urandom_kick(scale):
    """Unseeded noise of magnitude in [1, 2) x scale and random sign."""
    raw = int.from_bytes(os.urandom(8), "little")
    magnitude = (1.0 + (raw >> 11) / 2.0**53) * scale
    return magnitude if raw & 1 else -magnitude


def _unseeded_permutation(n):
    return np.argsort(np.frombuffer(os.urandom(8 * n), dtype=np.uint64), kind="stable")


class _Pool:
    """One streaming pool behind section 2.2's contract.

    `step(u)` is one internal step on a per-unit drive vector and returns that step's activity;
    `percept(x)` is one wake percept: u = W_in x, the pool's own percept count, Carried for the
    first `window` percepts (SORN's plasticity only there), a boundary fingerprint of every
    parameter at the last window percept, then the form, then T_in counted steps; phi is each
    unit's mean activity over the counted steps.
    """

    name = ""
    CARRIERS = ()

    def __init__(self, config, form, w_in, *, window=WINDOW, s_settle=S_SETTLE):
        if form not in FORMS:
            raise ValueError(f"REFUSED: unknown form {form!r}")
        if int(window) < 1 or int(s_settle) < 0 or int(config["T_in"]) < 1:
            raise ValueError("REFUSED: window and T_in must be positive, s_settle non-negative")
        self.config = dict(config)
        self.form = form
        self.w_in = None if w_in is None else np.array(w_in, dtype=np.float64)
        self.t_in = int(config["T_in"])
        self.window = int(window)
        self.s_settle = int(s_settle)
        self.count = 0
        self.boundary = None
        self.lesion_record = None
        self.observer = None

    def carriers(self):
        return {name: getattr(self, name).copy() for name in self.CARRIERS}

    def reset_carriers(self):
        for name, value in self._reset_values().items():
            setattr(self, name, value.copy())

    def params_fingerprint(self):
        return _crc_arrays(self.params())

    def fingerprint(self):
        """crc32 of the complete continuation state: the percept count, carriers, parameters."""
        crc = zlib.crc32(f"count|{self.count}".encode("ascii"))
        return _crc_arrays(self.params(), _crc_arrays(self.carriers(), crc))

    def frozen(self):
        """R-b5 (iii): the parameters equal their boundary fingerprint (Front-end: its lesion's)."""
        if self.boundary is None:
            return None
        expected = self.lesion_record["params"] if self.lesion_record else self.boundary
        return self.params_fingerprint() == expected

    def percept(self, x):
        if self.w_in is None:
            raise ValueError("REFUSED: this pool was built without an encoder")
        return self.percept_drive(drive(self.w_in, x))

    def percept_drive(self, u):
        u = np.asarray(u, dtype=np.float64)
        self.count += 1
        adapting = self.count <= self.window
        if not adapting and self.form in ("reset", "settled"):
            self.reset_carriers()
            for _ in range(self.s_settle if self.form == "settled" else 0):
                self._tick(u, False)
        total = np.zeros(self.n)
        for _ in range(self.t_in):
            total = total + self._tick(u, adapting)
        if self.count == self.window:
            self._at_boundary()
        return total / self.t_in

    def _tick(self, u, adapting):
        activity = self._window_step(u) if adapting else self.step(u)
        if self.observer is not None:
            self.observer(self, activity)
        return activity

    def _window_step(self, u):
        return self.step(u)

    def _at_boundary(self):
        self.boundary = self.params_fingerprint()
        if self.form == "front_end":
            before = {key: value.copy() for key, value in self.params().items()}
            self.lesion()
            after = self.params()
            self.lesion_record = {
                "params": self.params_fingerprint(),
                "changed": sorted(k for k in after if not np.array_equal(before[k], after[k])),
            }


class LsmPool(_Pool):
    """g1: spiking_ssm's fixed sparse LIF reservoir, hard reset to 0 at threshold 1."""

    name = "lsm"
    CARRIERS = ("v", "s")

    def __init__(self, pre, w, config, form="carried", w_in=None, **kwargs):
        super().__init__(config, form, w_in, **kwargs)
        self.pre = np.array(pre, dtype=np.int64)
        self.w = np.array(w, dtype=np.float64)
        self.n = self.pre.shape[0]
        self.lam = float(config["lambda"])
        self.g = config["g"]
        self.reset_carriers()

    def _reset_values(self):
        return {"v": np.zeros(self.n), "s": np.zeros(self.n, dtype=bool)}

    def _rec(self):
        # the source's `acc = 0.0; acc += w if s[j]` over each unit's stored list: a silent entry
        # adds +0.0, which leaves every partial sum as it was, so the cumsum is the same sequence
        return np.cumsum(np.where(self.s[self.pre], self.w, 0.0), axis=1)[:, -1]

    def step(self, u):
        u = np.asarray(u, dtype=np.float64)
        v = (self.lam * self.v + self.g * u) + self._rec()
        s = v >= LSM_THRESH
        self.v = np.where(s, 0.0, v)
        self.s = s
        return s.astype(np.float64)

    def lesion(self):
        self.w = np.zeros_like(self.w)

    def params(self):
        return {"pre": self.pre, "w": self.w}

    def perturb(self):
        unit, noise = _urandom_below(self.n), _urandom_kick(LSM_THRESH)
        v = self.v.copy()
        v[unit] += noise
        self.v = v
        return {"unit": unit, "noise": noise}


class EsnPool(_Pool):
    """g2: reservoir_eprop's fixed reservoir on EpropLIFBase's equation, soft reset only."""

    name = "esn"
    CARRIERS = ("v", "z")

    def __init__(self, w_rec, config, form="carried", w_in=None, *, dtype=np.float64, **kwargs):
        super().__init__(config, form, w_in, **kwargs)
        self.dtype = np.dtype(dtype)
        self.w_rec = np.array(w_rec, dtype=self.dtype)
        self.n = self.w_rec.shape[0]
        self.alpha = self.dtype.type(ESN_ALPHA)
        self.v_th = self.dtype.type(ESN_V_TH)
        self.g = config["g"]
        self.reset_carriers()

    def _reset_values(self):
        return {"v": np.zeros(self.n, self.dtype), "z": np.zeros(self.n, self.dtype)}

    def _rec(self):
        # sum over pre in index order of z_prev[pre] * W[pre, i]: a silent pre adds +-0.0, which
        # changes no partial sum, so the cumsum over the spiking rows alone is the same sequence
        rows = np.flatnonzero(self.z)
        if rows.size == 0:
            return np.zeros(self.n, self.dtype)
        return np.cumsum(self.w_rec[rows], axis=0)[-1]

    def step(self, u):
        cur = self._rec() + self.g * np.asarray(u, dtype=self.dtype)
        v = (self.alpha * self.v - self.v_th * self.z) + cur
        self.z = (v >= self.v_th).astype(self.dtype)
        self.v = v
        return self.z.astype(np.float64)

    def lesion(self):
        self.w_rec = np.zeros_like(self.w_rec)

    def params(self):
        return {"w_rec": self.w_rec}

    def perturb(self):
        unit, noise = _urandom_below(self.n), _urandom_kick(float(self.v_th))
        v = self.v.copy()
        v[unit] += noise
        self.v = v
        return {"unit": unit, "noise": noise}


class SornPool(_Pool):
    """g3: sorn's binary E/I threshold net; STDP, normalization and IP only inside the window."""

    name = "sorn"
    CARRIERS = ("x_e", "x_i")
    _ARRAYS = ("e_idx", "e_w", "e_valid", "ee_len", "i_idx", "i_w", "i_valid", "t_e", "t_i")

    def __init__(self, arrays, reset_state, config, form="carried", w_in=None, **kwargs):
        super().__init__(config, form, w_in, **kwargs)
        for key in self._ARRAYS:
            setattr(self, key, np.array(arrays[key]))
        self.me = int(arrays["me"])
        self.n_e, self.n_i = len(self.t_e), len(self.t_i)
        self.n = self.n_e + self.n_i
        plasticity = config["plasticity"]
        self.eta = None if plasticity == "off" else (float(plasticity[0]), float(plasticity[1]))
        self.h_ip = float(config["H_IP"])
        self.g = config["g"]
        self._reset = (np.array(reset_state[0], dtype=bool), np.array(reset_state[1], dtype=bool))
        self.x_e = np.zeros(self.n_e, dtype=bool)  # the source's self_organize starts from zeros
        self.x_i = np.zeros(self.n_i, dtype=bool)
        self.stdp_events = 0

    def _reset_values(self):
        return {"x_e": self._reset[0], "x_i": self._reset[1]}

    def _activity(self):
        return np.concatenate([self.x_e, self.x_i]).astype(np.float64)

    def _advance(self, u):
        """sorn._step: r = drive - T_E, + W_EE in list order where x_E, - W_EI where x_I; x = r >= 0."""
        xs = np.concatenate([self.x_e, self.x_i])
        e_terms = np.where(xs[self.e_idx] & self.e_valid, self.e_w, 0.0)
        e_terms[:, 0] = self.g * u[: self.n_e] - self.t_e
        i_terms = np.where(xs[self.i_idx] & self.i_valid, self.i_w, 0.0)
        i_terms[:, 0] = -self.t_i
        return (
            np.cumsum(e_terms, axis=1)[:, -1] >= 0.0,
            np.cumsum(i_terms, axis=1)[:, -1] >= 0.0,
        )

    def step(self, u):
        self.x_e, self.x_i = self._advance(np.asarray(u, dtype=np.float64))
        return self._activity()

    def _window_step(self, u):
        if self.eta is None:
            return self.step(u)
        old = self.x_e
        new_e, new_i = self._advance(np.asarray(u, dtype=np.float64))
        self._stdp(old, new_e)
        self.t_e = self.t_e + self.eta[1] * (new_e.astype(np.float64) - self.h_ip)
        self.x_e, self.x_i = new_e, new_i
        return self._activity()

    def _stdp(self, old, new):
        """sorn.self_organize's STDP (clipped at 0) and its row normalization by builtin sum()."""
        cols = slice(1, 1 + self.me)
        pre = self.e_idx[:, cols]
        valid = self.e_valid[:, cols]
        post_new, post_old = new.astype(np.int64), old.astype(np.int64)
        sign = post_new[:, None] * post_old[pre] - post_old[:, None] * post_new[pre]
        delta = self.eta[0] * np.where(valid, sign, 0)
        moved = delta != 0.0
        weights = self.e_w[:, cols]
        stepped = weights + delta
        weights = np.where(moved, np.where(stepped > 0.0, stepped, 0.0), weights)
        total = _builtin_sum_rows(weights, self.ee_len)
        scale = moved.any(axis=1) & (total > 0.0)
        divided = weights / np.where(scale, total, 1.0)[:, None]
        e_w = self.e_w.copy()
        e_w[:, cols] = np.where(scale[:, None] & valid, divided, weights)
        self.e_w = e_w
        self.stdp_events += int(np.count_nonzero(moved))

    def ee_rows(self):
        return [
            (self.e_idx[i, 1 : 1 + n].tolist(), self.e_w[i, 1 : 1 + n].tolist())
            for i, n in enumerate(self.ee_len.tolist())
        ]

    def lesion(self):
        self.e_w = np.zeros_like(self.e_w)  # W_EE and W_EI
        self.i_w = np.zeros_like(self.i_w)  # W_IE

    def params(self):
        return {key: getattr(self, key) for key in self._ARRAYS}

    def perturb(self):
        unit = _urandom_below(self.n)
        x_e, x_i = self.x_e.copy(), self.x_i.copy()
        if unit < self.n_e:
            x_e[unit] = not x_e[unit]
        else:
            x_i[unit - self.n_e] = not x_i[unit - self.n_e]
        self.x_e, self.x_i = x_e, x_i
        return {"unit": unit}


class RbnPool(_Pool):
    """g4: build_rbn's synchronous network; input bits [u > tau] act on the input nodes after it."""

    name = "rbn"
    CARRIERS = ("state",)

    def __init__(
        self, inputs, tables, in_nodes, reset_state, config, form="carried", w_in=None, **kwargs
    ):
        super().__init__(config, form, w_in, **kwargs)
        self.inputs = np.array(inputs, dtype=np.int64).reshape(len(inputs), -1)
        self.tables = np.array(tables, dtype=np.int64)
        self.in_nodes = np.array(in_nodes, dtype=np.int64)
        self.n = self.inputs.shape[0]
        if config["mode"] not in ("xor", "clamp"):
            raise ValueError(f"REFUSED: unknown input mode {config['mode']!r}")
        self.mode = config["mode"]
        self.tau = config["tau"]
        self.hold = False
        self._reset = np.array(reset_state, dtype=np.int64)
        self._rows = np.arange(self.n)
        self.reset_carriers()  # the initial state is the reset state

    def _reset_values(self):
        return {"state": self._reset}

    def _next(self):
        """random_boolean_network.step: every node reads its K inputs' previous values."""
        addr = np.zeros(self.n, dtype=np.int64)
        for bit in range(self.inputs.shape[1]):
            addr = addr | (self.state[self.inputs[:, bit]] << bit)
        return self.tables[self._rows, addr]

    def step(self, u):
        u = np.asarray(u, dtype=np.float64)
        nxt = self.state.copy() if self.hold else self._next()
        if self.in_nodes.size:
            bits = (u[self.in_nodes] > self.tau).astype(np.int64)
            nxt[self.in_nodes] = nxt[self.in_nodes] ^ bits if self.mode == "xor" else bits
        self.state = nxt
        return nxt.astype(np.float64)

    def lesion(self):
        self.hold = True

    def params(self):
        return {
            "inputs": self.inputs,
            "tables": self.tables,
            "in_nodes": self.in_nodes,
            "hold": np.array([self.hold]),
        }

    def perturb(self):
        unit = _urandom_below(self.n)
        state = self.state.copy()
        state[unit] ^= 1
        self.state = state
        return {"unit": unit}


# distinct-from: neuromodulation/spiking_curiosity.py -- its _make_pool builds a LIF pool that turns a prediction error into a count for curiosity; no recurrence, no percept stream (site read)
POOLS = {"lsm": LsmPool, "esn": EsnPool, "sorn": SornPool, "rbn": RbnPool}


def make_pool(
    candidate, config, recording_seed, form="carried", *, window=WINDOW, s_settle=S_SETTLE
):
    """A pool from (candidate, config, recording seed): W_in from the recording seed, the random
    structure from realization_seed, each drawn in its source's exact call order."""
    config = canonical(candidate, config)
    seed = realization_seed(candidate, recording_seed)
    encoder = w_in(recording_seed)
    kwargs = {"window": window, "s_settle": s_settle}
    if candidate == "lsm":
        _, pre, weights = _lsm_draw(seed, N, config["k"], config["w"])
        return LsmPool(pre, weights, config, form, encoder, **kwargs)
    if candidate == "esn":
        return EsnPool(_esn_w_rec(seed, N, config["rho"]), config, form, encoder, **kwargs)
    if candidate == "sorn":
        arrays = _sorn_arrays(
            _sorn_draw(random.Random(seed), SORN_N_E, SORN_N_I), SORN_N_E, SORN_N_I
        )
        state = _sorn_reset_state(seed, config["H_IP"], SORN_N_E, SORN_N_I)
        return SornPool(arrays, state, config, form, encoder, **kwargs)
    inputs, tables = _rbn_draw(seed, N, config["K"], config["p"])
    in_nodes = _rbn_input_nodes(seed, N, config["f"])
    return RbnPool(
        inputs, tables, in_nodes, _rbn_reset_state(seed, N), config, form, encoder, **kwargs
    )


# ---- fixture percepts and the startup diagnostic (section 2.2) -----------------------------------


def _percept(r):
    v = r**4
    norm = float(np.sqrt(np.cumsum(v * v)[-1]))
    return np.clip(v * 1.73 / norm, 0.0, 1.0)


def fixture_percepts():
    """The 16 fixtures: successive random(64) draws r of default_rng(777), clip(v 1.73 / |v|, 0, 1)
    with v = r ** 4."""
    rng = np.random.default_rng(DIAG_SEED)
    return [_percept(rng.random(N_IN)) for _ in range(DIAG_FIXTURES)]


def cycled_fixture(count):
    fixtures = fixture_percepts()
    return [fixtures[i % len(fixtures)] for i in range(count)]


def random_percepts(seed, count):
    rng = np.random.default_rng(seed)
    return [_percept(rng.random(N_IN)) for _ in range(count)]


def startup_diagnostic(pool):
    """Each fixture from reset: 320 steps on its drive, r_j = the mean over steps 257-320; the
    transient at S is the max over fixtures of the mean over units of |phi_{j,S} - r_j|, phi_{j,S}
    being steps S+1..S+T_in. Runs on copies with the pool's current parameters, plasticity off."""
    base = copy.deepcopy(pool)
    base.observer = None
    transient = dict.fromkeys(DIAG_SETTLES, 0.0)
    rows = []
    for x in fixture_percepts():
        u = drive(base.w_in, x)
        run = copy.deepcopy(base)
        run.reset_carriers()
        acts = np.stack([run.step(u) for _ in range(DIAG_BURN + DIAG_MEAN)])
        long_run = np.cumsum(acts[DIAG_BURN:], axis=0)[-1] / DIAG_MEAN
        row = {}
        for settle in DIAG_SETTLES:
            phi = np.cumsum(acts[settle : settle + run.t_in], axis=0)[-1] / run.t_in
            row[settle] = float(np.cumsum(np.abs(phi - long_run))[-1] / run.n)
            transient[settle] = max(transient[settle], row[settle])
        rows.append(row)
    return {"transient": transient, "flagged": transient[64] > DIAG_FLAG, "per_fixture": rows}


# ---- the cost scalar (section 2.4) ---------------------------------------------------------------


def cost(candidate, config, recording_seed, percepts, *, reps=3, settled=False):
    """Median over `reps` of the Carried form's per-percept wall time over percepts 257-end, pool
    alone with the projection, at one BLAS thread; initialization and the window timed apart."""
    xs = [np.asarray(x, dtype=np.float64) for x in percepts]
    if len(xs) <= WINDOW:
        raise ValueError("REFUSED: the cost is timed over percepts 257-end; pass more than 256")
    rest = len(xs) - WINDOW
    carried, init, adapt, settle = [], [], [], []
    with _blas_threads(1):
        for _ in range(reps):
            t0 = time.perf_counter()
            pool = make_pool(candidate, config, recording_seed, "carried")
            t1 = time.perf_counter()
            for x in xs[:WINDOW]:
                pool.percept(x)
            t2 = time.perf_counter()
            for x in xs[WINDOW:]:
                pool.percept(x)
            t3 = time.perf_counter()
            init.append(t1 - t0)
            adapt.append(t2 - t1)
            carried.append(1000.0 * (t3 - t2) / rest)
            if settled:
                twin = make_pool(candidate, config, recording_seed, "settled")
                for x in xs[:WINDOW]:
                    twin.percept(x)
                t4 = time.perf_counter()
                for x in xs[WINDOW:]:
                    twin.percept(x)
                settle.append(1000.0 * (time.perf_counter() - t4) / rest)
    result = {
        "cost_ms": statistics.median(carried),
        "carried_ms": carried,
        "init_s": init,
        "window_s": adapt,
    }
    if settled:
        result["settled_ms"] = statistics.median(settle)
        result["settled_reps_ms"] = settle
    return result


# ---- R-b0: parity with the sources (section 2.7) -------------------------------------------------


def _snapshot(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.array(value, copy=True)


def _trace_locals(call, code, watch, names):
    """Run `call()` and copy the named locals of the frame running `code` each time `watch` is
    rebound: the source's own per-step state, read without editing the source."""
    frames, last = [], [None]

    def local(frame, event, arg):
        if event in ("line", "return"):
            current = frame.f_locals.get(watch)
            if current is not None and current is not last[0]:
                last[0] = current
                frames.append(tuple(_snapshot(frame.f_locals[n]) for n in names))
        return local

    def outer(frame, event, arg):
        return local if frame.f_code is code else None

    previous = sys.gettrace()
    sys.settrace(outer)
    try:
        result = call()
    finally:
        sys.settrace(previous)
    return result, frames


def check_r_b0_lsm(seed=0, sequences=6, red_leak=0.9):
    """g1 against spiking_ssm.run_reservoir's rates and SynOps, bit for bit (N 100, scalar drive,
    reset between sequences). Red: the leak moved to `red_leak` on a fixture that carries membranes."""
    from snn_data_lake.reasoning import spiking_ssm as src

    w_src, rec = src.build_reservoir(seed)
    consumed, pre, weights = _lsm_draw(seed, src.N, src.FANIN, SOURCE_W)
    structure = (
        consumed == w_src
        and pre.tolist() == [[j for j, _ in row] for row in rec]
        and weights.tolist() == [[x for _, x in row] for row in rec]
        and LSM_THRESH == src.THRESH
    )
    rng = random.Random(seed + 1)
    protos = src.make_sequences(rng)
    seqs = [
        [
            min(1.0, max(0.0, protos[i % src.CLASSES][t] + rng.uniform(-src.NOISE, src.NOISE)))
            for t in range(src.T)
        ]
        for i in range(sequences)
    ]
    projection = np.array(consumed)

    def port(leak):
        pool = LsmPool(pre, weights, {"lambda": leak, "g": 1, "T_in": 1})
        out, carried = [], 0
        for seq in seqs:
            pool.reset_carriers()
            counts = np.zeros(src.N, dtype=np.int64)
            synops = 0
            for x in seq:
                synops += int(np.count_nonzero(pool.s[pool.pre]))
                carried += int(np.count_nonzero(pool.v))
                counts += pool.step(projection * x).astype(np.int64)
            out.append(((counts / len(seq)).tolist(), synops))
        return out, carried

    source = [
        (list(rates), synops) for rates, synops in (src.run_reservoir(s, w_src, rec) for s in seqs)
    ]
    mine, carried = port(src.LEAK)
    red, _ = port(red_leak)
    parity = mine == source
    broke = red != source
    return {
        "verdict": bool(structure and parity and broke and carried > 0),
        "numbers": {
            "structure": bool(structure),
            "parity": bool(parity),
            "sequences": len(seqs),
            "synops": [s for _, s in source],
            "carried_unit_steps": carried,
        },
        "red": bool(broke),
    }


# distinct-from: reasoning/rsnn_delays_torch.py -- an export bridge for another model; all five sites read, reasons at the module head
def _esn_torch_leg(recording_seed=DEV_SEED, rho=0.9, g=4, hold=8, percepts=4, red_alpha=0.95):
    """The port in float32 against EpropLIFBase.forward (W_out zero) on a declared fixture: the
    first `percepts` fixture percepts, each held `hold` steps, W_rec at rho, drive g W_in x. The
    drive enters through an identity W_in so both sides step the same float32 drive. The fixture
    was declared from a build-time scan of 18 (g, percepts, hold) choices at rho 0.9: all 18 gave
    identical trains, 15 kept every membrane at least 1e-4 from V_TH, and this one keeps 4.4e-4."""
    import torch

    from snn_data_lake.cortex.eprop_kernel import (
        DEFAULT_DAMP,
        DEFAULT_GAMMA,
        DEFAULT_KAPPA,
        EpropLIFBase,
    )

    class _Probe(EpropLIFBase):
        """EpropLIFBase's own forward on the port's weights; forward only, readout inert."""

        def __init__(self, w_rec):
            super().__init__()
            n = w_rec.shape[0]
            self.n_in = self.hidden = n
            self.num_classes = 1
            self.alpha, self.v_th = ESN_ALPHA, ESN_V_TH
            self.gamma, self.damp, self.kappa = DEFAULT_GAMMA, DEFAULT_DAMP, DEFAULT_KAPPA
            self.W_in = torch.nn.Parameter(torch.eye(n), requires_grad=False)
            self.W_rec = torch.nn.Parameter(
                torch.from_numpy(np.ascontiguousarray(w_rec.T)), requires_grad=False
            )
            self.W_out = torch.nn.Parameter(torch.zeros(1, n), requires_grad=False)
            self.b_out = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    w32 = _esn_w_rec(realization_seed("esn", recording_seed), N, rho).astype(np.float32)
    encoder = w_in(recording_seed)
    us = []
    for x in fixture_percepts()[:percepts]:
        us.extend([(g * drive(encoder, x)).astype(np.float32)] * hold)
    net = _Probe(w32)
    x = torch.from_numpy(np.stack(us)[None])
    _, frames = _trace_locals(
        lambda: net(x), inspect.unwrap(EpropLIFBase.forward).__code__, "z", ("v", "z")
    )
    torch_v = [f[0][0] for f in frames[1:]]
    torch_z = [f[1][0] for f in frames[1:]]

    def port(alpha):
        pool = EsnPool(w32, {"g": 1, "T_in": 1}, dtype=np.float32)
        pool.alpha = np.float32(alpha)
        vs, zs = [], []
        for u in us:
            pool.step(u)
            vs.append(pool.v.copy())
            zs.append(pool.z.copy())
        return vs, zs

    vs, zs = port(ESN_ALPHA)
    same = len(zs) == len(torch_z) and all(np.array_equal(a, b) for a, b in zip(zs, torch_z))
    threshold = np.float64(np.float32(ESN_V_TH))
    margin = min(float(np.min(np.abs(v.astype(np.float64) - threshold))) for v in vs + torch_v)
    _, red_z = port(red_alpha)
    broke = not all(np.array_equal(a, b) for a, b in zip(red_z, torch_z))
    spikes = int(sum(int(np.count_nonzero(z)) for z in zs))
    return {
        "verdict": bool(same and margin >= G2_MARGIN and broke and 0 < spikes < len(zs) * N),
        "numbers": {"steps": len(us), "spikes": spikes, "margin": margin, "identical": bool(same)},
        "red": bool(broke),
    }


def check_r_b0_esn(seed=0, steps=24, n_in=16, red_alpha=0.95):
    """g2 against reservoir_eprop's reservoir forward (readout inert): identical spike trains and
    membranes within 1e-12 in float64; plus the float32 leg against EpropLIFBase.forward. Red: the
    leak moved to `red_alpha` in both legs, on fixtures that carry membranes."""
    from snn_data_lake.reasoning import reservoir_eprop as src

    with _blas_threads(1):  # the source's own eigvals, at the port's one thread
        win, w_src, _, _ = src.build_reservoir(n_in, seed)
    w_port = _esn_w_rec(seed, src.H, src.RHO)
    w_gap = float(np.max(np.abs(w_port - w_src)))
    x = np.random.default_rng(seed + 5).random((1, steps, n_in)) * 2.0
    readout = (np.zeros((src.H, 1)), np.zeros(1))
    _, frames = _trace_locals(
        lambda: src._run(x, win, w_src, *readout, None, False), src._run.__code__, "z", ("v", "z")
    )
    src_v = [f[0][0] for f in frames[1:]]
    src_z = [f[1][0] for f in frames[1:]]
    drives = [drive(win.T, x[0, t]) for t in range(steps)]

    def port(alpha):
        pool = EsnPool(w_port, {"g": 1, "T_in": 1})
        pool.alpha = np.float64(alpha)
        vs, zs, carried = [], [], 0
        for u in drives:
            carried += int(np.count_nonzero(pool.v))
            pool.step(u)
            vs.append(pool.v.copy())
            zs.append(pool.z.copy())
        return vs, zs, carried

    def matches(vs, zs):
        spikes = len(zs) == len(src_z) and all(np.array_equal(a, b) for a, b in zip(zs, src_z))
        gap = max(float(np.max(np.abs(a - b))) for a, b in zip(vs, src_v))
        return spikes and gap <= G2_TOLERANCE, gap

    vs, zs, carried = port(ESN_ALPHA)
    parity, gap = matches(vs, zs)
    broke = not matches(*port(red_alpha)[:2])[0]
    spikes = int(sum(int(np.count_nonzero(z)) for z in src_z))
    constants = (ESN_ALPHA, ESN_V_TH) == (src.ALPHA, src.V_TH)
    torch_leg = _esn_torch_leg(red_alpha=red_alpha)
    exercised = carried > 0 and 0 < spikes < steps * src.H
    return {
        "verdict": bool(
            constants
            and w_gap <= G2_TOLERANCE
            and parity
            and broke
            and exercised
            and torch_leg["verdict"]
        ),
        "numbers": {
            "w_gap": w_gap,
            "membrane_gap": gap,
            "spikes": spikes,
            "margin": min(float(np.min(np.abs(v - src.V_TH))) for v in src_v),
            "carried_unit_steps": carried,
            "torch": torch_leg["numbers"],
        },
        "red": bool(broke and torch_leg["red"]),
    }


def check_r_b0_sorn(seed=7, words=40, red_eta=0.005):
    """g3 against sorn at its own sizes: build_net's structure, _step's states and self_organize's
    weights and thresholds on the source's stream (noise included for this fixture only), bit for
    bit. Red: the STDP rate moved to `red_eta` on a stream where STDP events occur."""
    from snn_data_lake.reasoning import sorn as src

    constants = (SORN_P_EE, SORN_P_IE, SORN_P_EI, SORN_T_E_MAX, SORN_T_I_MAX, SORN_IN_PER_SYM) == (
        src.P_EE,
        src.P_IE,
        src.P_EI,
        src.T_E_MAX,
        src.T_I_MAX,
        src.IN_PER_SYM,
    )
    net = src.build_net(random.Random(seed))
    mine = _sorn_draw(random.Random(seed), src.N_E, src.N_I, src.N_SYMBOLS)
    structure = constants and mine == net
    stream = src.make_stream(words, random.Random(seed + 1))
    drives = []
    for sym, noise in stream:
        d = np.zeros(src.N_E + src.N_I)
        d[net["in_units"][sym]] = src.I_IN
        d[list(noise)] = src.I_IN
        drives.append(d)
    zeros = (np.zeros(src.N_E, dtype=bool), np.zeros(src.N_I, dtype=bool))
    pool = SornPool(
        _sorn_arrays(mine, src.N_E, src.N_I),
        zeros,
        {"plasticity": "off", "H_IP": src.H_IP, "g": 1, "T_in": 1},
    )
    x_e, x_i = [0] * src.N_E, [0] * src.N_I
    steps = True
    for (sym, noise), d in zip(stream, drives):
        x_e, x_i = src._step(net, x_e, x_i, sym, noise)
        pool.step(d)
        steps &= pool.x_e.astype(int).tolist() == x_e and pool.x_i.astype(int).tolist() == x_i
    fresh = src.build_net(random.Random(seed))
    src.self_organize(fresh, stream)

    def plastic(eta_stdp):
        config = {"plasticity": (eta_stdp, src.ETA_IP), "H_IP": src.H_IP, "g": 1, "T_in": 1}
        tuned = SornPool(_sorn_arrays(mine, src.N_E, src.N_I), zeros, config)
        for d in drives:
            tuned._window_step(d)
        return tuned

    tuned = plastic(src.ETA_STDP)
    learned = [w for _, w in tuned.ee_rows()] == fresh["ee_w"] and tuned.t_e.tolist() == fresh[
        "t_e"
    ]
    broke = [w for _, w in plastic(red_eta).ee_rows()] != fresh["ee_w"]
    return {
        "verdict": bool(structure and steps and learned and broke and tuned.stdp_events > 0),
        "numbers": {
            "structure": bool(structure),
            "steps": len(stream),
            "states_identical": bool(steps),
            "learned_identical": bool(learned),
            "stdp_events": tuned.stdp_events,
        },
        "red": bool(broke),
    }


def check_r_b0_rbn(seed=3, steps=32):
    """g4 against random_boolean_network.step with no input, bit for bit, at N 256 for every K and
    p of the grid. Red: flip the truth-table entry node 0 addresses at the first step."""
    from snn_data_lake.reasoning import random_boolean_network as src

    runs = {}
    for k in (1, 2, 3):
        for p in (0.5, 0.7):
            net = src.build_rbn(N, k, seed, p)
            inputs, tables = _rbn_draw(seed, N, k, p)
            start = src.random_state(N, seed + 1)
            pool = RbnPool(inputs, tables, [], start, {"mode": "xor", "tau": 0, "T_in": 1})
            mine = [pool.step(np.zeros(N)).astype(np.int64).tolist() for _ in range(steps)]
            structure = inputs.tolist() == net.inputs and tables.tolist() == net.tables
            runs[f"K{k}/p{p}"] = bool(structure and mine == src.run(net, start, steps)[1:])
    net = src.build_rbn(N, 2, seed, 0.5)
    inputs, tables = _rbn_draw(seed, N, 2, 0.5)
    start = src.random_state(N, seed + 1)
    addr = 0
    for bit, j in enumerate(net.inputs[0]):
        addr |= start[j] << bit
    flipped = tables.copy()
    flipped[0, addr] ^= 1
    pool = RbnPool(inputs, flipped, [], start, {"mode": "xor", "tau": 0, "T_in": 1})
    red = [pool.step(np.zeros(N)).astype(np.int64).tolist() for _ in range(steps)]
    broke = red != src.run(net, start, steps)[1:]
    return {"verdict": bool(all(runs.values()) and broke), "numbers": runs, "red": bool(broke)}


def check_r_b0():
    return {
        "lsm": check_r_b0_lsm(),
        "esn": check_r_b0_esn(),
        "sorn": check_r_b0_sorn(),
        "rbn": check_r_b0_rbn(),
    }


# ---- the constructed faults (every red below uses one) -------------------------------------------


def _fault(pool, label, **methods):
    pool.__class__ = type(f"{type(pool).__name__}_{label}", (type(pool),), methods)
    return pool


def _skip_carrier(pool, carrier):
    """R-b5 (i)'s red: a reset that leaves one carrier as it was."""
    base = type(pool)

    def reset_carriers(self):
        kept = getattr(self, carrier).copy()
        base.reset_carriers(self)
        setattr(self, carrier, kept)

    return _fault(pool, f"skip_{carrier}", reset_carriers=reset_carriers)


def _plastic_forever(pool):
    """R-b5 (iii)'s red: SORN's plasticity left on after the window."""
    base = type(pool)
    return _fault(
        pool, "plastic_forever", _tick=lambda self, u, adapting: base._tick(self, u, True)
    )


def _count_multi(pool):
    """Count unit-steps whose recurrent sum has three or more spiking presynaptic terms (the only
    sums whose association can change the float result). The arithmetic is untouched."""
    base = type(pool)
    pool.multi = 0

    def _rec(self):
        if isinstance(self, LsmPool):
            active = np.count_nonzero(self.s[self.pre], axis=1)
        else:
            rows = np.flatnonzero(self.z)
            active = np.count_nonzero(self.w_rec[rows] != 0.0, axis=0) if rows.size else np.zeros(1)
        self.multi += int(np.count_nonzero(active >= 3))
        return base._rec(self)

    return _fault(pool, "count_multi", _rec=_rec)


def _permuted_sum(pool):
    """R-b3's g1/g2 red: the recurrent sum taken in an order permuted by an unseeded source."""
    if isinstance(pool, LsmPool):

        def _rec(self):
            terms = np.where(self.s[self.pre], self.w, 0.0)
            return np.cumsum(terms[:, _unseeded_permutation(terms.shape[1])], axis=1)[:, -1]

    else:

        def _rec(self):
            rows = np.flatnonzero(self.z)
            if rows.size == 0:
                return np.zeros(self.n, self.dtype)
            return np.cumsum(self.w_rec[rows[_unseeded_permutation(rows.size)]], axis=0)[-1]

    return _fault(pool, "permuted_sum", _rec=_rec)


def _asynchronous(pool):
    """R-b3's g3/g4 red: an asynchronous in-place update in an unseeded order. `async_reads`
    counts updates that read a neighbour already changed within the same step."""
    pool.async_reads = 0
    if isinstance(pool, SornPool):

        def _advance(self, u):
            xs = np.concatenate([self.x_e, self.x_i])
            old, fresh = xs.copy(), np.zeros(self.n, dtype=bool)
            for i in _unseeded_permutation(self.n_e):
                cols, valid = self.e_idx[i], self.e_valid[i]
                self.async_reads += bool(np.any(fresh[cols] & valid & (xs[cols] != old[cols])))
                terms = np.where(xs[cols] & valid, self.e_w[i], 0.0)
                terms[0] = self.g * u[i] - self.t_e[i]
                xs[i], fresh[i] = np.cumsum(terms)[-1] >= 0.0, True
            for k in _unseeded_permutation(self.n_i):
                cols, valid = self.i_idx[k], self.i_valid[k]
                self.async_reads += bool(np.any(fresh[cols] & valid & (xs[cols] != old[cols])))
                terms = np.where(xs[cols] & valid, self.i_w[k], 0.0)
                terms[0] = -self.t_i[k]
                xs[self.n_e + k] = np.cumsum(terms)[-1] >= 0.0
            return xs[: self.n_e].copy(), xs[self.n_e :].copy()

        return _fault(pool, "asynchronous", _advance=_advance)

    def _next(self):
        state, old = self.state.copy(), self.state
        fresh = np.zeros(self.n, dtype=bool)
        for i in _unseeded_permutation(self.n):
            cols = self.inputs[i]
            self.async_reads += bool(np.any(fresh[cols] & (state[cols] != old[cols])))
            addr = 0
            for bit, j in enumerate(cols.tolist()):
                addr |= int(state[j]) << bit
            state[i], fresh[i] = self.tables[i, addr], True
        return state

    return _fault(pool, "asynchronous", _next=_next)


# ---- R-b3: determinism by construction (section 2.7) ---------------------------------------------


def _record(pool, percepts):
    tally = {"crc": 0, "steps": 0}

    def observe(_, activity):
        tally["crc"] = zlib.crc32(activity.tobytes(), tally["crc"])
        tally["steps"] += 1

    pool.observer = observe
    phi = 0
    for x in percepts:
        phi = zlib.crc32(pool.percept(x).tobytes(), phi)
    pool.observer = None
    return {
        "activity_crc": tally["crc"],
        "steps": tally["steps"],
        "phi_crc": phi,
        "state": pool.fingerprint(),
    }


def check_r_b3(
    candidate,
    config,
    recording_seed=DEV_SEED,
    *,
    percepts=R_B3_PERCEPTS,
    window=WINDOW,
    forms=("carried", "settled"),
    blas=R_B3_BLAS,
    torch_threads=R_B3_TORCH,
):
    """The pool alone on the 16 fixtures cycled to `percepts`, in each form, under every BLAS and
    torch thread setting: every step's activity and the full continuation state must agree."""
    import torch
    from threadpoolctl import threadpool_info

    seq = cycled_fixture(percepts)
    runs, settings = {}, []
    saved = torch.get_num_threads()
    try:
        for b in blas:
            for t in torch_threads:
                torch.set_num_threads(t)
                with _blas_threads(b):
                    backends = [
                        {
                            key: info.get(key)
                            for key in ("internal_api", "prefix", "version", "num_threads")
                        }
                        for info in threadpool_info()
                    ]
                    settings.append(
                        {
                            "blas": b,
                            "torch": t,
                            "torch_effective": torch.get_num_threads(),
                            "backends": backends,
                        }
                    )
                    for form in forms:
                        pool = make_pool(candidate, config, recording_seed, form, window=window)
                        runs[f"{b}/{t}/{form}"] = _record(pool, seq)
    finally:
        torch.set_num_threads(saved)
    equal = {}
    for form in forms:
        seen = {
            json.dumps(run, sort_keys=True) for key, run in runs.items() if key.endswith("/" + form)
        }
        equal[form] = len(seen) == 1
    return {
        "verdict": all(equal.values()),
        "numbers": {"equal": equal, "settings": settings, "runs": runs},
    }


def check_r_b3_red(candidate, recording_seed=DEV_SEED, *, percepts=R_B3_PERCEPTS, window=WINDOW):
    """On the candidate's declared witness, the order fault must change the continuation state,
    and the fixture must exercise it: at least 100 unit-steps with three or more simultaneous
    presynaptic spikes (g1, g2), or in-place reads of neighbours already changed (g3, g4)."""
    config = R_B3_WITNESS[candidate]
    seq = cycled_fixture(percepts)
    base = make_pool(candidate, config, recording_seed, "carried", window=window)
    if candidate in ("lsm", "esn"):
        base = _count_multi(base)
        faulted = _permuted_sum(
            make_pool(candidate, config, recording_seed, "carried", window=window)
        )
    else:
        faulted = _asynchronous(
            make_pool(candidate, config, recording_seed, "carried", window=window)
        )
    canonical_run, faulted_run = _record(base, seq), _record(faulted, seq)
    differs = canonical_run["state"] != faulted_run["state"]
    if candidate in ("lsm", "esn"):
        exercised, count = base.multi >= R_B3_MULTI, base.multi
    else:
        exercised, count = faulted.async_reads >= 1, faulted.async_reads
    return {
        "verdict": bool(differs and exercised),
        "numbers": {"config": config, "exercise_count": count, "differs": bool(differs)},
        "red": bool(differs),
    }


# ---- R-b4's pool-noise red (the pool's half; the fresh-process replays are the harness's) --------


def pool_noise_red(candidate, config, recording_seed=DEV_SEED, *, window=WINDOW, after=16):
    """`perturb()` before every post-window percept of one of two otherwise identical Carried runs:
    the continuation state and the phi sequence must both change."""
    seq = cycled_fixture(window + after)
    plain = make_pool(candidate, config, recording_seed, "carried", window=window)
    noisy = make_pool(candidate, config, recording_seed, "carried", window=window)
    phi_plain = phi_noisy = 0
    for i, x in enumerate(seq):
        if i >= window:
            noisy.perturb()
        phi_plain = zlib.crc32(plain.percept(x).tobytes(), phi_plain)
        phi_noisy = zlib.crc32(noisy.percept(x).tobytes(), phi_noisy)
    state = plain.fingerprint() != noisy.fingerprint()
    phi = phi_plain != phi_noisy
    return {
        "verdict": bool(state and phi),
        "numbers": {"kicks": after, "state": bool(state), "phi": bool(phi)},
    }


# ---- R-b5: the forms do what they claim (section 2.7) --------------------------------------------


def _two_prefixes(pool, window, prefix):
    for x in cycled_fixture(window):
        pool.percept(x)
    left, right = copy.deepcopy(pool), copy.deepcopy(pool)
    for a, b in zip(random_percepts(1, prefix), random_percepts(2, prefix), strict=True):
        left.percept(a)
        right.percept(b)
    diverged = left.fingerprint() != right.fingerprint()  # else erasure would pass vacuously
    probe = fixture_percepts()[0]
    return left, right, left.percept(probe), right.percept(probe), diverged


def check_r_b5_erasure(
    candidate, config, form, recording_seed=DEV_SEED, *, window=WINDOW, prefix=8, skip=None
):
    """(i) After the window, two different random prefixes, then the same Reset or Settled percept:
    identical phi and continuation state. `skip` builds the red: a reset that misses one carrier."""
    if form not in ("reset", "settled"):
        raise ValueError(f"REFUSED: erasure is a property of reset and settled, not {form}")
    pool = make_pool(candidate, config, recording_seed, form, window=window)
    if skip is not None:
        pool = _skip_carrier(pool, skip)
    left, right, phi_left, phi_right, diverged = _two_prefixes(pool, window, prefix)
    same_phi = bool(np.array_equal(phi_left, phi_right))
    same_state = left.fingerprint() == right.fingerprint()
    return {
        "verdict": bool(same_phi and same_state),
        "numbers": {
            "form": form,
            "skip": skip,
            "prefixes_diverged": bool(diverged),
            "same_phi": same_phi,
            "same_state": same_state,
        },
    }


def check_r_b5_erasure_red(candidate, form, recording_seed=DEV_SEED, *, window=WINDOW):
    """The declared witness with its declared carrier skipped must FAIL erasure; every other
    carrier's skip is reported beside it."""
    config, carrier = R_B5_ERASURE_WITNESS[candidate]
    skips = {
        name: not check_r_b5_erasure(
            candidate, config, form, recording_seed, window=window, skip=name
        )["verdict"]
        for name in POOLS[candidate].CARRIERS
    }
    return {
        "verdict": bool(skips[carrier]),
        "numbers": {"config": config, "carrier": carrier, "fired": skips},
    }


def check_r_b5_persistence(
    candidate="lsm",
    config=None,
    recording_seed=DEV_SEED,
    *,
    window=WINDOW,
    prefix=8,
    carried_form="carried",
):
    """(ii) Carried after two different prefixes must end in different continuation states.
    `carried_form="reset"` builds the red: a "carried" form that resets every percept."""
    config = R_B5_PERSISTENCE_WITNESS if config is None else config
    pool = make_pool(candidate, config, recording_seed, carried_form, window=window)
    left, right, _, _, _ = _two_prefixes(pool, window, prefix)
    differs = left.fingerprint() != right.fingerprint()
    return {
        "verdict": bool(differs),
        "numbers": {"config": config, "form": carried_form, "persists": bool(differs)},
    }


def check_r_b5_freeze(
    candidate, config, recording_seed=DEV_SEED, *, window=WINDOW, after=16, plastic_forever=False
):
    """(iii) All four forms reach the same boundary fingerprint and every parameter still equals
    it at the end (Front-end: its recorded lesion). `plastic_forever` builds SORN's red."""
    boundary, frozen, lesions = {}, {}, {}
    for form in FORMS:
        pool = make_pool(candidate, config, recording_seed, form, window=window)
        if plastic_forever:
            pool = _plastic_forever(pool)
        for x in cycled_fixture(window + after):
            pool.percept(x)
        boundary[form], frozen[form] = pool.boundary, bool(pool.frozen())
        lesions[form] = pool.lesion_record and pool.lesion_record["changed"]
    agree = len(set(boundary.values())) == 1
    return {
        "verdict": bool(agree and all(frozen.values())),
        "numbers": {
            "boundary_agree": agree,
            "frozen": frozen,
            "front_end_changed": lesions["front_end"],
        },
    }


# ---- the self-check ------------------------------------------------------------------------------


def main():
    t0 = time.perf_counter()
    crcs = draw_order_crc32()
    assert crcs == draw_order_crc32(), "the draw order is not deterministic"
    assert [len(grid(c)) for c in CANDIDATES] == [288, 48, 240, 432]
    print("draw-order crc32: " + ", ".join(f"{c}={crcs[c]}" for c in CANDIDATES))
    for name, result in check_r_b0().items():
        assert result["verdict"] and result["red"], (name, result["numbers"])
        print(f"R-b0 {LABEL[name]} {name}: parity and its red OK")
    small = {"percepts": 64, "window": 32}
    for candidate in CANDIDATES:
        red = check_r_b3_red(candidate, **small)
        assert red["verdict"], (candidate, red["numbers"])
        equal = check_r_b3(candidate, R_B3_WITNESS[candidate], **small)
        assert equal["verdict"], (candidate, equal["numbers"]["equal"])
        print(
            f"R-b3 {LABEL[candidate]}: thread-invariant; red fired ({red['numbers']['exercise_count']} exercising)"
        )
    for candidate in CANDIDATES:
        config, _ = R_B5_ERASURE_WITNESS[candidate]
        for form in ("reset", "settled"):
            assert check_r_b5_erasure(candidate, config, form, window=32)["verdict"], (
                candidate,
                form,
            )
            red = check_r_b5_erasure_red(candidate, form, window=32)
            assert red["verdict"], (candidate, form, red["numbers"])
        assert check_r_b5_freeze(candidate, config, window=32, after=4)["verdict"], candidate
        assert pool_noise_red(candidate, config, window=32)["verdict"], candidate
        print(
            f"R-b5 {LABEL[candidate]}: erasure, skip-a-carrier red, freeze; R-b4 pool-noise red OK"
        )
    assert check_r_b5_persistence(window=32)["verdict"]
    assert not check_r_b5_persistence(window=32, carried_form="reset")["verdict"]
    assert check_r_b5_freeze("sorn", R_B5_FREEZE_WITNESS, window=32)["verdict"]
    assert not check_r_b5_freeze("sorn", R_B5_FREEZE_WITNESS, window=32, plastic_forever=True)[
        "verdict"
    ]
    print(
        "R-b5 persistence and its reset-every-percept red; freeze and SORN's plasticity-left-on red OK"
    )
    diagnostic = startup_diagnostic(make_pool("lsm", R_B5_PERSISTENCE_WITNESS, DEV_SEED))
    print(
        "startup diagnostic g1 witness: transients "
        + ", ".join(f"S{s}={v:.4f}" for s, v in diagnostic["transient"].items())
    )
    print(
        f"self-check OK: R-b0, R-b3, R-b4 pool noise, R-b5 and their reds ({time.perf_counter() - t0:.1f} s)"
    )


if __name__ == "__main__":
    main()
