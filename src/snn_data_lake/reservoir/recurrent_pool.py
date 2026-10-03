"""G1(b) recurrent pool: the selected research-tier reservoir, g1 LSM, as a governed streaming pool.

The frozen text is specs/022-unified-brain-arc/evidence/g1b-pool-selection.md section 2 (commit
23c7e97). This module holds the pool contract of section 2.2 (the shared encoder, the adaptation
window, the four forms, the startup diagnostic), section 2.3's g1 LSM with its source's exact draws,
the manifest and cost of section 2.4, and the machinery of R-b0 (parity with its source), R-b3
(determinism by construction), R-b4's pool-noise red and R-b5 (the forms do what they claim); G1(c)
adds window 0 (g1c-carried-state.md 2.6, 2.10: no adaptation window, Settled from the first
percept) and its check, and 2.7's transported encoder (the grower's weights copied into W_in at
the random draw's mean row norm, and the swap control's fresh draw) with its refusals. Every red
is constructed by `main()`. The reference readout, the harness arms and the campaign are
instruments/memory_diagnostic.py's. Nothing here builds a brain or touches the lake.

The selection record (section 3.4, commit 0bd1de7) named g1 the winner, and section 2.9 then deleted
the losing ports g2 (ESN-LSM), g3 (SORN) and g4 (RBN) from this module. Commit f27ae70 holds them,
and the campaign that selected among all four ran at it.

Arithmetic (section 2.2): float64 numpy; no BLAS product in any per-step or per-percept pool
computation; every reduction is a sequential sum, taken as np.cumsum's last element along a declared
axis. Where R-b0 claims bit parity the order is the source's.

One module on purpose: section 2.11 lands one governed pool component, and its size is the size of
that contract.
"""

# A port, not a consolidation: reasoning/spiking_ssm.py cannot stream (a scalar drive, a reset per
# sequence, sizes fixed at module scope), so g1 is ported to the pool contract and R-b0 holds the
# port to its source. The four research sources (spiking_ssm.py, reservoir_eprop.py, sorn.py and
# random_boolean_network.py) stay untouched in this arc (section 2.9); each one's
# retire/promote/exempt row is its own P2 opening (section 2.13).
# distinct-from: src/snn_data_lake/growth/arealization.py -- a GRN consensus network settled once
#   into developmental region labels; nothing is driven by a percept or carried between percepts
# distinct-from: src/snn_data_lake/faculties/prefrontal_executive.py -- its LIF pool holds a goal with
#   NO external input (section 1.2 names that a different job); no stimulus drive, no forms
# distinct-from: src/snn_data_lake/faculties/recursive_tom.py -- a theory-of-mind reading over recorded
#   rows; no recurrent state at all
# distinct-from: src/snn_data_lake/instruments/report_order_probe.py -- a report-order probe reading over
#   constructed rows; no recurrent state at all
# distinct-from: src/snn_data_lake/reasoning/cellular_neural_network.py -- a 3x3 cellular-network
#   convolution on a grid; no streaming pool, no percept drive
# distinct-from: src/snn_data_lake/reasoning/rsnn_delays_torch.py -- its verify_transfer checks an
#   exported delay-RSNN against its own torch model; R-b0 here holds a pool port to its source
# distinct-from: src/snn_data_lake/reasoning/hjepa_arm_torch.py -- an H-JEPA export->core bridge
#   check; no recurrent pool
# distinct-from: src/snn_data_lake/reasoning/jepa_scaffold_torch.py -- a JEPA export->core bridge
#   check; no recurrent pool
# distinct-from: src/snn_data_lake/gates/eprop_transport_gate.py -- AmbiguousDisposition is a
#   gate's refusal type; nothing here steps a network

import copy
import itertools
import json
import os
import random
import statistics
import time
import zlib

import numpy as np

N = 256
N_IN = 64
WINDOW = 256
S_SETTLE = 64
FORMS = ("carried", "reset", "settled", "front_end")
# Section 2.9: the winner alone (section 3.4). g1 keeps section 2.3's index 1, so its realization
# seeds and draw order are the campaign's; g2-g4 (indices 2-4) were deleted with their ports.
CANDIDATES = ("lsm",)
INDEX = {name: i + 1 for i, name in enumerate(CANDIDATES)}
LABEL = {"lsm": "g1"}

# Section 2.3's grid, factor order and levels exactly as written (the last factor varies fastest).
GRIDS = {
    "lsm": (
        ("k", (8, 32)),
        ("w", (0.25, 0.5, 1.0)),
        ("lambda", (0.5, 0.7, 0.9)),
        ("g", (1, 2, 4, 8)),
        ("T_in", (1, 2, 4, 8)),
    ),
}
MANIFEST_SIZE = 32
COST_BOUND_MS = 25.0
DEV_SEED = 100  # section 2.10's development seeds: the mechanism checks' W_in and realizations

LSM_THRESH = 1.0
SOURCE_W = 0.5  # spiking_ssm's recurrent weight bound, which R-b0's g1 fixture draws at

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
# Section 2.7's R-b3 witness for g1.
R_B3_WITNESS = {
    "lsm": {"k": 32, "w": 1.0, "lambda": 0.9, "g": 8, "T_in": 8},
}
R_B5_PERSISTENCE_WITNESS = {"k": 8, "w": 0.5, "lambda": 0.9, "g": 2, "T_in": 1}
# R-b5 (i)'s skip-a-carrier red: per candidate, a declared configuration and the carrier whose
# skipped reset must leak prefix history through both Reset and Settled's 64 uncounted steps.
R_B5_ERASURE_WITNESS = {
    "lsm": ({"k": 32, "w": 1.0, "lambda": 0.9, "g": 8, "T_in": 8}, "v"),
}
# G1(c) 2.10's window-0 check: more percepts than the default window so percepts 257-end can be
# compared, and the short run each form's boundary and freeze are read after
W0_PERCEPTS = 320
W0_AFTER = 16
# each constructed fault and the one clause it must make fail
W0_FAULTS = {
    "clamp": "i_first_phi",
    "write_after": "ii_boundary",
    "boundary_resets": "iv_carried_identity",
}


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


# ---- the transported encoder and the swap control's fresh draw (G1(c) 2.7) ----------------------

# The fixed middle word of the swap control's SeedSequence([recording_seed, SWAP_TAG]): the ASCII
# bytes of "SWAP". W_in itself is seeded by the integer recording_seed * 31 + 7, and the within-block
# shuffle's own tag is "SHUF" (instruments/carried_state_campaign.SHUFFLE_TAG), so no draw shares one.
SWAP_TAG = 0x53574150
NORM_TOL = 1e-12  # a rescaled row's norm equals the random draw's mean row norm within this


# Sites the duplicate-capability hook flagged, read; none computes an encoder's row norms:
# distinct-from: instruments/everyday_battery.py::conjunct_c -- compares a seed's campaign rows with a sequential run's
# distinct-from: engine/brain.py::resume_probe -- returns a probe's rows and summary from child interpreters
# distinct-from: faculties/homeostatic_drives.py::_train_values -- a value rule's exploration run
# distinct-from: gates/safety_envelope.py::_enumerated -- the paths a doc's table rows name
# distinct-from: instruments/attention_schema.py::t078_planted_rows -- n rows of noise features with a planted label
def row_norms(w):
    """Each row's L2 norm: the square root of a sequential sum of squares (section 2.2's rule)."""
    w = np.asarray(w, dtype=np.float64)
    return np.sqrt(np.cumsum(w * w, axis=1)[:, -1])


def mean_row_norm(w):
    norms = row_norms(w)
    return float(np.cumsum(norms)[-1] / norms.shape[0])


def _encoder_shaped(w, what):
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (N, N_IN):
        raise ValueError(f"REFUSED: {what} is not an encoder shape ({N} x {N_IN}): {w.shape}")
    return w


def rescale_rows(rows, target):
    """`rows` with every row scaled to L2 norm `target`; a row of zero or non-finite norm cannot be."""
    rows = np.asarray(rows, dtype=np.float64)
    norms = row_norms(rows)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("REFUSED: a row of zero or non-finite norm cannot be rescaled")
    return rows * (target / norms)[:, None]


def transported_encoder(w_random, w_b):
    """G1(c) 2.7's copy form: the random encoder with rows 0..L-1 replaced by the L x 64 block `w_b`
    (cast to float64), each row rescaled to the mean row norm of the random draw; rows L..255 are
    the draw's, bit for bit. Refused: no unit (L == 0, never padded from nothing), a width other
    than 64, L above 256, a zero or non-finite row. Neither input is written."""
    random = _encoder_shaped(w_random, "the random draw")
    block = np.asarray(w_b, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError(f"REFUSED: the transported block's width is not 64 inputs: {block.shape}")
    units = block.shape[0]
    if units == 0:
        raise ValueError("REFUSED: the transported block holds no unit (L == 0)")
    if block.shape[1] != N_IN:
        raise ValueError(f"REFUSED: the transported block's width {block.shape[1]} is not {N_IN}")
    if units > N:
        raise ValueError(
            f"REFUSED: the transported block holds {units} units, more than the encoder's {N}"
        )
    out = random.copy()
    out[:units] = rescale_rows(block, mean_row_norm(random))
    return out


def fresh_encoder(recording_seed, w_random):
    """G1(c) 2.7's swap control: a fresh N(0, 1) / 8 draw from SeedSequence([recording_seed,
    SWAP_TAG]) with EVERY row rescaled to the mean row norm of `w_random` (the copy form's own
    target), so nothing is transported and only the encoder's discontinuity at b remains."""
    random = _encoder_shaped(w_random, "the random draw")
    stream = np.random.default_rng(np.random.SeedSequence([recording_seed, SWAP_TAG]))
    return rescale_rows(stream.standard_normal((N, N_IN)) / 8, mean_row_norm(random))


# ---- the grids and the manifest (section 2.4) ----------------------------------------------------


def grid(candidate):
    factors = GRIDS[candidate]
    names = [name for name, _ in factors]
    return [dict(zip(names, levels)) for levels in itertools.product(*(v for _, v in factors))]


def canonical(candidate, config):
    """The grid's own entry equal to `config`, else refused."""
    wanted = dict(config)
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


# distinct-from: instruments/everyday_battery.py -- sites read (a NamedTuple, prior_drift_harness's host recorder, episodic_recall's unbind); none limits BLAS threads
# distinct-from: instruments/memory_diagnostic.py -- its L102 reports the loaded pools (threadpool_info), never limits them; pool_selection.py L684 CALLS this limiter
def _blas_threads(limit):
    """threadpoolctl's limit on every loaded BLAS: R-b3, the cost and the harness's one-thread pin.
    BLAS only (G1(b) panel, 2026-09-25): an unscoped limit also caps torch's OpenMP runtime, so R-b3's
    torch-at-4 setting silently ran at the BLAS count once torch had loaded OpenMP."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=limit, user_api="blas")


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
    first `window` percepts, a boundary fingerprint of every parameter at the last window
    percept, then the form, then T_in counted steps; phi is each unit's mean activity over the
    counted steps. Window 0 (G1(c) 2.6) is no window at all: every percept runs the form's own
    regime from percept 1, and the boundary fingerprint is taken at construction (the subclass
    takes it once its parameters exist).
    """

    name = ""
    CARRIERS = ()

    def __init__(self, config, form, w_in, *, window=WINDOW, s_settle=S_SETTLE):
        if form not in FORMS:
            raise ValueError(f"REFUSED: unknown form {form!r}")
        if int(window) < 0:
            raise ValueError("REFUSED: the window is a non-negative count of percepts")
        if int(s_settle) < 0 or int(config["T_in"]) < 1:
            raise ValueError("REFUSED: T_in must be positive, s_settle non-negative")
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

    # Sites the duplicate-capability hook flagged, read; none replaces a pool's input matrix:
    # distinct-from: engine/brain.py::_attend_real_stream -- pools an encoder latent into a 4-d sensory channel by salience
    # distinct-from: faculties/locus_coeruleus.py::_population_spikes -- integrates one LIF pool under a constant drive
    # distinct-from: gates/selfcheck_cache.py::closure_key -- a sha256 carry key for a module closure
    # distinct-from: plasticity/eprop_kernel.py::EpropLIFBase -- the recurrent-LIF forward of the two trained perception encoders
    def replace_encoder(self, w):
        """Swap the input matrix W_in for `w` (256 x 64, finite). W_in is no part of the pool's
        state: its count, carriers, parameters and fingerprints are as they were, which is why a
        replay's binding records the swapped matrix's own crc."""
        w = _encoder_shaped(w, "the encoder")
        if not np.all(np.isfinite(w)):
            raise ValueError("REFUSED: the encoder holds a non-finite entry")
        self.w_in = np.array(w, dtype=np.float64)

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
                self._tick(u)
        total = np.zeros(self.n)
        for _ in range(self.t_in):
            total = total + self._tick(u)
        if self.count == self.window:
            self._at_boundary()
        return total / self.t_in

    def _tick(self, u):
        activity = self.step(u)
        if self.observer is not None:
            self.observer(self, activity)
        return activity

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
        if self.window == 0:
            # no window: the parameters are final from construction, so the boundary fingerprint
            # (and Front-end's lesion) exist now; percept_drive never reaches count == window
            self._at_boundary()

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


# distinct-from: faculties/spiking_curiosity.py -- its _make_pool builds a LIF pool that turns a prediction error into a count for curiosity; no recurrence, no percept stream (site read)
POOLS = {"lsm": LsmPool}


def make_pool(
    candidate, config, recording_seed, form="carried", *, window=WINDOW, s_settle=S_SETTLE
):
    """A pool from (candidate, config, recording seed): W_in from the recording seed, the random
    structure from realization_seed, drawn in spiking_ssm's exact call order. `canonical` raises for
    any candidate or configuration outside section 2.3's g1 grid."""
    config = canonical(candidate, config)
    seed = realization_seed(candidate, recording_seed)
    encoder = w_in(recording_seed)
    kwargs = {"window": window, "s_settle": s_settle}
    _, pre, weights = _lsm_draw(seed, N, config["k"], config["w"])
    return LsmPool(pre, weights, config, form, encoder, **kwargs)


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
    being steps S+1..S+T_in. Runs on copies with the pool's current parameters."""
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


# ---- R-b0: parity with the source (section 2.7) --------------------------------------------------


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


def check_r_b0():
    return {"lsm": check_r_b0_lsm()}


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


# The fault below writes a weight only so R-b5 (iii) can be seen to fail (sites read; none is a
# planted fault or a freeze red):
# distinct-from: readout/predictive_coding.py -- learn_step is a real local weight update
# distinct-from: growth/h01_ingest.py -- decode_shard counts a shard's joins in one pass
# distinct-from: faculties/vsa_reasoning.py -- coincidence_many counts spike coincidences
# distinct-from: plasticity/naca.py -- its reversal check is a real R-STDP experiment
# distinct-from: reasoning/spike_chat_lake.py -- _kb_reply retrieves stored triples
def _write_after_boundary(pool):
    """R-b5 (iii)'s red: one weight moved by one ulp right after the boundary fingerprint."""
    base = type(pool)

    def _at_boundary(self):
        base._at_boundary(self)
        w = self.w.copy()
        w[0, 0] = np.nextafter(w[0, 0], np.inf)
        self.w = w

    return _fault(pool, "write_after_boundary", _at_boundary=_at_boundary)


def _count_multi(pool):
    """Count unit-steps whose recurrent sum has three or more spiking presynaptic terms (the only
    sums whose association can change the float result). The arithmetic is untouched."""
    base = type(pool)
    pool.multi = 0

    def _rec(self):
        active = np.count_nonzero(self.s[self.pre], axis=1)
        self.multi += int(np.count_nonzero(active >= 3))
        return base._rec(self)

    return _fault(pool, "count_multi", _rec=_rec)


def _permuted_sum(pool):
    """R-b3's g1 red: the recurrent sum taken in an order permuted by an unseeded source."""

    def _rec(self):
        terms = np.where(self.s[self.pre], self.w, 0.0)
        return np.cumsum(terms[:, _unseeded_permutation(terms.shape[1])], axis=1)[:, -1]

    return _fault(pool, "permuted_sum", _rec=_rec)


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
    # every requested torch setting must be the one that ran (it silently was not before 2026-09-25)
    applied = all(s["torch_effective"] == s["torch"] for s in settings)
    return {
        "verdict": all(equal.values()) and applied,
        "numbers": {"equal": equal, "applied": applied, "settings": settings, "runs": runs},
    }


def check_r_b3_red(candidate, recording_seed=DEV_SEED, *, percepts=R_B3_PERCEPTS, window=WINDOW):
    """On the candidate's declared witness, the order fault must change the continuation state,
    and the fixture must exercise it: at least 100 unit-steps with three or more simultaneous
    presynaptic spikes."""
    config = R_B3_WITNESS[candidate]
    seq = cycled_fixture(percepts)
    base = _count_multi(make_pool(candidate, config, recording_seed, "carried", window=window))
    faulted = _permuted_sum(make_pool(candidate, config, recording_seed, "carried", window=window))
    canonical_run, faulted_run = _record(base, seq), _record(faulted, seq)
    differs = canonical_run["state"] != faulted_run["state"]
    exercised, count = base.multi >= R_B3_MULTI, base.multi
    return {
        "verdict": bool(differs and exercised),
        "numbers": {"config": config, "exercise_count": count, "differs": bool(differs)},
        "red": bool(differs),
    }


# ---- R-b4's pool-noise red (the pool's half; the fresh-process replays are the harness's) --------


def pool_noise_red(candidate, config, recording_seed=DEV_SEED, *, window=WINDOW, after=16):
    """`perturb()` before every post-window percept of one of two otherwise identical Carried runs:
    the continuation-state CHAIN (`fingerprint()` at every percept boundary) and the phi sequence
    must both change. Amended 2026-09-25: the final state alone is one sample, and a pool that forgets
    a kick within one percept (g3 under strong drive, campaign 1 on seed 203) can leave it unchanged."""
    seq = cycled_fixture(window + after)
    plain = make_pool(candidate, config, recording_seed, "carried", window=window)
    noisy = make_pool(candidate, config, recording_seed, "carried", window=window)
    phi_plain = phi_noisy = chain_plain = chain_noisy = 0
    for i, x in enumerate(seq):
        if i >= window:
            noisy.perturb()
        phi_plain = zlib.crc32(plain.percept(x).tobytes(), phi_plain)
        phi_noisy = zlib.crc32(noisy.percept(x).tobytes(), phi_noisy)
        chain_plain = zlib.crc32(plain.fingerprint().to_bytes(4, "little"), chain_plain)
        chain_noisy = zlib.crc32(noisy.fingerprint().to_bytes(4, "little"), chain_noisy)
    state = chain_plain != chain_noisy
    phi = phi_plain != phi_noisy
    final = plain.fingerprint() != noisy.fingerprint()
    return {
        "verdict": bool(state and phi),
        "numbers": {"kicks": after, "state": bool(state), "phi": bool(phi), "final": bool(final)},
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
    candidate, config, recording_seed=DEV_SEED, *, window=WINDOW, after=16, write=None
):
    """(iii) All four forms reach the same boundary fingerprint and every parameter still equals
    it at the end (Front-end: its recorded lesion). `write` names the form whose pool has one weight
    written after its boundary fingerprint: the red."""
    boundary, frozen, lesions = {}, {}, {}
    for form in FORMS:
        pool = make_pool(candidate, config, recording_seed, form, window=window)
        if form == write:
            pool = _write_after_boundary(pool)
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


# Sites read; none checks a pool's parameters against a boundary fingerprint:
# distinct-from: faculties/recursive_tom.py -- _t082_s2 checks one seed of a theory-of-mind control
# distinct-from: instruments/deep_stdp.py -- _mechanics_check asserts an STDP library's mechanics
# distinct-from: plasticity/inhibitory_stdp.py -- istdp_synapse_rule authors a synapse rule
# distinct-from: gates/amdahl_gate.py -- within_noise_of_zero is a sham timing criterion
def check_r_b5_freeze_red(candidate, config, recording_seed=DEV_SEED, *, window=WINDOW, after=16):
    """(iii)'s red on g1, which learns nothing (the pre-registered red, SORN's plasticity left on,
    went with g3): the Carried form's pool has one weight written after its boundary fingerprint.
    Its freeze must fail on that form alone while the unfaulted twin holds on all four."""
    twin = check_r_b5_freeze(candidate, config, recording_seed, window=window, after=after)
    red = check_r_b5_freeze(
        candidate, config, recording_seed, window=window, after=after, write="carried"
    )
    fired = sorted(form for form, ok in red["numbers"]["frozen"].items() if not ok)
    return {
        "verdict": bool(
            twin["verdict"]
            and not red["verdict"]
            and fired == ["carried"]
            and red["numbers"]["boundary_agree"]
        ),
        "numbers": {
            "config": config,
            "twin_frozen": twin["numbers"]["frozen"],
            "faulted_frozen": red["numbers"]["frozen"],
            "fired": fired,
            "boundary_agree": red["numbers"]["boundary_agree"],
        },
    }


# ---- window 0: Settled from the first percept (G1(c) 2.6, 2.10) ---------------------------------


def _settled_first_phi(config, recording_seed, x):
    """Reset, s_settle uncounted steps, then t_in counted steps summed left to right and divided
    by t_in, written from step() and reset_carriers() alone (never percept_drive or the window)."""
    stepper = make_pool("lsm", config, recording_seed, "carried")
    u = drive(stepper.w_in, x)
    stepper.reset_carriers()
    for _ in range(stepper.s_settle):
        stepper.step(u)
    total = np.zeros(stepper.n)
    for _ in range(stepper.t_in):
        total = total + stepper.step(u)
    return total / stepper.t_in


def _boundary_resets(pool):
    """(iv)'s red: the window boundary also resets the carriers of a Carried pool."""
    base = type(pool)

    def _at_boundary(self):
        base._at_boundary(self)
        self.reset_carriers()

    return _fault(pool, "boundary_resets", _at_boundary=_at_boundary)


# distinct-from: instruments/pool_selection.py::mechanism_checks -- G1(b)'s campaign step CALLS this module's R-b0..R-b5 checks at the default window; this is G1(c)'s own window-0 check, which that step does not run
def check_window_zero(
    config=R_B5_PERSISTENCE_WITNESS, recording_seed=DEV_SEED, *, percepts=W0_PERCEPTS, fault=None
):
    """G1(c) 2.10's window-0 check: four clauses on one realization and one percept sequence.
    (i) Settled@0's first phi equals an independent derivation (reset, S_SETTLE uncounted steps,
    T_in counted steps, their mean) bit for bit. (ii) Every form's boundary fingerprint exists at
    count 0, the four agree, Front-end's lesion is recorded, and frozen() holds after a short run.
    (iii) Settled@0's phi equals Settled@256's on percepts 257-end and differs on at least one of
    percepts 1-256. (iv) Carried@0 and Carried@256 give identical phi sequences, state chains and
    final fingerprints. `fault` builds a red and must fail exactly its own clause (W0_FAULTS):
    "clamp" silently raises window 0 to 1 after construction, "write_after" writes one weight
    after the construction-time boundary, "boundary_resets" gives Carried a boundary that resets
    its carriers."""
    if fault is not None and fault not in W0_FAULTS:
        raise ValueError(f"REFUSED: unknown window-0 fault {fault!r}")
    if percepts <= WINDOW:
        raise ValueError(
            "REFUSED: window 0 meets window 256 on percepts 257-end; pass more than 256"
        )
    sequence = random_percepts(recording_seed, percepts)

    def build(form, window):
        return make_pool("lsm", config, recording_seed, form, window=window)

    settled = build("settled", 0)
    if fault == "clamp":
        settled.window = max(1, settled.window)
    phi_zero = [settled.percept(x) for x in sequence]
    first = _settled_first_phi(config, recording_seed, sequence[0])
    first_equal = bool(np.array_equal(phi_zero[0], first))

    settled_default = build("settled", WINDOW)
    phi_default = [settled_default.percept(x) for x in sequence]
    after_equal = all(
        np.array_equal(a, b) for a, b in zip(phi_zero[WINDOW:], phi_default[WINDOW:], strict=True)
    )
    inside_differ = sum(
        not np.array_equal(a, b)
        for a, b in zip(phi_zero[:WINDOW], phi_default[:WINDOW], strict=True)
    )

    boundary, exists, frozen, lesions = {}, {}, {}, {}
    for form in FORMS:
        pool = build(form, 0)
        exists[form] = pool.count == 0 and pool.boundary is not None
        boundary[form] = pool.boundary
        if fault == "write_after" and form == "carried":
            w = pool.w.copy()
            w[0, 0] = np.nextafter(w[0, 0], np.inf)
            pool.w = w
        for x in sequence[:W0_AFTER]:
            pool.percept(x)
        frozen[form] = bool(pool.frozen())
        lesions[form] = pool.lesion_record and pool.lesion_record["changed"]
    agree = len(set(boundary.values())) == 1

    carried = {}
    for window in (0, WINDOW):
        pool = build("carried", window)
        if fault == "boundary_resets":
            pool = _boundary_resets(pool)
        rows, crc = [], 0
        for x in sequence:
            rows.append(pool.percept(x))
            crc = zlib.crc32(pool.fingerprint().to_bytes(4, "little"), crc)
        carried[window] = (rows, crc, pool.fingerprint())
    zero, default = carried[0], carried[WINDOW]
    phi_identical = all(np.array_equal(a, b) for a, b in zip(zero[0], default[0], strict=True))

    clauses = {
        "i_first_phi": first_equal,
        "ii_boundary": bool(
            all(exists.values())
            and agree
            and all(frozen.values())
            and lesions["front_end"] == ["w"]
        ),
        "iii_settled_invariance": bool(after_equal and inside_differ >= 1),
        "iv_carried_identity": bool(
            phi_identical and zero[1] == default[1] and zero[2] == default[2]
        ),
    }
    return {
        "verdict": all(clauses.values()),
        "numbers": {
            "config": config,
            "percepts": percepts,
            "fault": fault,
            "clauses": clauses,
            "first_phi_crc": zlib.crc32(phi_zero[0].tobytes()),
            "after_boundary_equal": bool(after_equal),
            "inside_window_differing": int(inside_differ),
            "boundary_at_count_0": exists,
            "boundary_agree": agree,
            "frozen": frozen,
            "front_end_changed": lesions["front_end"],
            "carried_phi_identical": bool(phi_identical),
            "carried_chain": [zero[1], default[1]],
            "carried_final": [zero[2], default[2]],
        },
    }


# Sites read; none runs constructed window-0 faults against check_window_zero's clauses:
# distinct-from: instruments/attention_schema.py::main -- its own self-check, six reds of the attention-schema scorer
# distinct-from: instruments/paired_contrast.py::main -- its own self-check of the paired t statistic and its exact FPR
# distinct-from: instruments/pool_selection.py::step_mechanism -- G1(b)'s campaign step that runs the default-window pool checks in a child process
# distinct-from: readout/interaction_learning.py::g0_campaign -- a seed/arm campaign driver that writes a label and routing verdict
# distinct-from: reasoning/jepa_scaffold_torch.py::verify_transfer -- an export-to-core bridge check on a torch model
def check_window_zero_faults(
    config=R_B5_PERSISTENCE_WITNESS, recording_seed=DEV_SEED, *, percepts=W0_PERCEPTS, twin=None
):
    """The window-0 faults of W0_FAULTS: each must fail its own clause of check_window_zero and no
    other, while the unfaulted twin (`twin`, a result already in hand) holds all four."""
    twin = check_window_zero(config, recording_seed, percepts=percepts) if twin is None else twin
    failed = {}
    for fault in W0_FAULTS:
        broken = check_window_zero(config, recording_seed, percepts=percepts, fault=fault)
        failed[fault] = sorted(key for key, ok in broken["numbers"]["clauses"].items() if not ok)
    expected = {fault: [clause] for fault, clause in W0_FAULTS.items()}
    return {
        "verdict": bool(twin["verdict"] and failed == expected),
        "numbers": {"twin": twin["numbers"]["clauses"], "failed": failed},
    }


# ---- the transported encoder (G1(c) 2.7, 2.10 "Transport") ---------------------------------------


def _norm_matched(rows, target):
    return bool(np.max(np.abs(row_norms(rows) - target)) < NORM_TOL)


def _is_refused(build):
    try:
        build()
    except ValueError as error:
        return str(error).startswith("REFUSED")
    return False


# Sites the duplicate-capability hook flagged, read; none holds a copied weight block to a norm contract:
# distinct-from: gates/conclusion_protocol.py::mechanism_findings -- reads what a served page renders for a charged row
# distinct-from: gates/safety_envelope.py::_enumerated -- the paths a doc's table rows name
# distinct-from: gates/selfcheck_cache.py::_global_files -- the file rows of the global envelope
# distinct-from: growth/basic_components.py -- the (encoder, neuron, connection, rule) composition contract
# distinct-from: growth/function_preserving_growth.py::_eligibility_seed -- unit-normalizes the mean input into one NEW unit's in-weights
def check_transported_encoder(recording_seed=DEV_SEED, units=3):
    """G1(c) 2.10's encoder clause. The copy's rows 0..L-1 sit at the random draw's mean row norm
    within NORM_TOL and rows L..255 are the draw's, bit for bit; the swap draw's 256 rows all sit at
    that norm. Each refusal fires (L == 0, width 63, L 257, a zero-norm row), and a copy that skips
    the rescale fails the norm match (the red of the match itself)."""
    w_random = w_in(recording_seed)
    target = mean_row_norm(w_random)
    block = np.random.default_rng(recording_seed + 1).standard_normal((units, N_IN))
    block = block.astype(np.float32)
    copied = transported_encoder(w_random, block)
    swapped = fresh_encoder(recording_seed, w_random)
    copy_error = float(np.max(np.abs(row_norms(copied[:units]) - target)))
    swap_error = float(np.max(np.abs(row_norms(swapped) - target)))
    kept = bool(np.array_equal(copied[units:], w_random[units:]))
    unscaled = w_random.copy()
    unscaled[:units] = block
    red = {
        "L == 0": _is_refused(lambda: transported_encoder(w_random, np.zeros((0, N_IN)))),
        "width 63": _is_refused(lambda: transported_encoder(w_random, np.ones((units, N_IN - 1)))),
        "L 257": _is_refused(lambda: transported_encoder(w_random, np.ones((N + 1, N_IN)))),
        "zero-norm row": _is_refused(
            lambda: transported_encoder(w_random, np.zeros((units, N_IN)))
        ),
        "an unscaled copy": not _norm_matched(unscaled[:units], target),
    }
    return {
        "verdict": bool(
            _norm_matched(copied[:units], target)
            and _norm_matched(swapped, target)
            and kept
            and all(red.values())
        ),
        "numbers": {
            "units": units,
            "mean_row_norm": target,
            "norm_error": max(copy_error, swap_error),
            "rows_kept": kept,
        },
        "red": red,
    }


# ---- the self-check ------------------------------------------------------------------------------


def main():
    t0 = time.perf_counter()
    crcs = draw_order_crc32()
    assert crcs == draw_order_crc32(), "the draw order is not deterministic"
    assert [len(grid(c)) for c in CANDIDATES] == [288]
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
        red = check_r_b5_freeze_red(candidate, config, window=32, after=4)
        assert red["verdict"], (candidate, red["numbers"])
        assert pool_noise_red(candidate, config, window=32)["verdict"], candidate
        print(
            f"R-b5 {LABEL[candidate]}: erasure, skip-a-carrier red, freeze, write-after-boundary red "
            f"(fired on {red['numbers']['fired']}); R-b4 pool-noise red OK"
        )
    assert check_r_b5_persistence(window=32)["verdict"]
    assert not check_r_b5_persistence(window=32, carried_form="reset")["verdict"]
    print("R-b5 persistence and its reset-every-percept red OK")
    zero = check_window_zero()
    assert zero["verdict"], zero["numbers"]
    faults = check_window_zero_faults(twin=zero)
    assert faults["verdict"], faults["numbers"]
    print(
        "window 0: Settled@0's first phi, boundary at count 0, Settled@0 = Settled@256 after 256, "
        "Carried@0 = Carried@256; faults fired on "
        + ", ".join(f"{fault}->{clause}" for fault, clause in W0_FAULTS.items())
    )
    encoder = check_transported_encoder()
    assert encoder["verdict"], encoder
    print(
        "G1(c) transport: copied rows and the swap draw at the random draw's mean row norm "
        f"(error {encoder['numbers']['norm_error']:.1e}); refusals fired on "
        + ", ".join(encoder["red"])
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
