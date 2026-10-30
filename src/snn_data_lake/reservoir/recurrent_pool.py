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

G1(c) 2.9 hosts the pool: `KernelLsmPool` runs g1's network (the same draws, realization seed and
index) on reservoir/kernel_tick.py's KernelTick at the lineage's binding frac_bits. G1(c)'s migration
gate chose the hosted form (g1c-carried-state.md 3.3), and 2.9 then deleted the loser, the float64
port (LsmPool's own step, R-b0's parity with spiking_ssm, the port's R-b3 and freeze reds), and the
host switch (HOSTS, `make_pool(host=...)`, the parity fixtures that ran both hosts side by side).
Commit 6dbe61e2 holds them; the campaign ran at 5761f3a7. `make_pool` builds the one pool implementation, and R-b3
(equality across (P, threads); its red a float-accumulation twin fed raw weights of a declared
magnitude spread in an unseeded order, `check_r_b3_hosted_red`), R-b4 and R-b5 run on it.

The selection record (section 3.4, commit 0bd1de7) named g1 the winner, and section 2.9 then deleted
the losing ports g2 (ESN-LSM), g3 (SORN) and g4 (RBN) from this module. Commit f27ae70 holds them,
and the campaign that selected among all four ran at it.

Arithmetic: the external drive is float64 numpy, a sequential sum taken as np.cumsum's last element
(section 2.2); the pool steps in KernelTick's fixed point (integer accumulation, a float32 membrane).
No BLAS product in any per-step or per-percept pool computation.

One module on purpose: section 2.11 lands one governed pool component, and its size is the size of
that contract.
"""

# A port, not a consolidation: reasoning/spiking_ssm.py cannot stream (a scalar drive, a reset per
# sequence, sizes fixed at module scope), so g1 is ported to the pool contract (R-b0 held the float64
# port to its source until G1(c) 2.9 deleted the port; the draws are still the source's). The four research sources (spiking_ssm.py, reservoir_eprop.py, sorn.py and
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
#   exported delay-RSNN against its own torch model; R-b0 here held a pool port to its source
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

from snn_data_lake.growth.growable import grow_rows, grow_vector
from snn_data_lake.reservoir.kernel_state import KernelState
from snn_data_lake.reservoir.kernel_tick import KernelTick, Synapses

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
R_B3_MULTI = 100
# G1(c) 2.9: the hosted pool's R-b3 settings, (P partitions, threads), and the declared magnitude
# spread of its float-accumulation red: within each unit's incoming set, weights spanning at least
# 2**12 in magnitude (one entry scaled up by it, one down by it)
R_B3_HOSTED = ((1, 1), (8, 1), (8, 4))
R_B3_SPREAD = 2**12
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


def w_in(recording_seed, n=N):
    """W_in (n x 64, n = 256 at birth), entries N(0, 1) / 8 from default_rng(recording_seed * 31 + 7).
    The stream fills rows in order, so a grown pool's encoder is prefix-stable: rows 0..255 of the
    n-row draw are the born draw, bit for bit, and row j is a joined unit's row by the same law."""
    return np.random.default_rng(recording_seed * 31 + 7).standard_normal((n, N_IN)) / 8


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


def _encoder_shaped(w, what, rows=None):
    """`w` as float64 if it is an encoder: `rows` x 64 (the pool's own n), or any positive row count
    x 64 when `rows` is None (a random draw sets the row count of what is built from it)."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != N_IN or w.shape[0] < 1 or rows not in (None, w.shape[0]):
        raise ValueError(
            f"REFUSED: {what} is not an encoder shape ({rows or 'n'} x {N_IN}): {w.shape}"
        )
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
    than 64, L above the draw's row count, a zero or non-finite row. Neither input is written."""
    random = _encoder_shaped(w_random, "the random draw")
    block = np.asarray(w_b, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError(f"REFUSED: the transported block's width is not 64 inputs: {block.shape}")
    units = block.shape[0]
    if units == 0:
        raise ValueError("REFUSED: the transported block holds no unit (L == 0)")
    if block.shape[1] != N_IN:
        raise ValueError(f"REFUSED: the transported block's width {block.shape[1]} is not {N_IN}")
    if units > random.shape[0]:
        raise ValueError(
            f"REFUSED: the transported block holds {units} units, more than the encoder's "
            f"{random.shape[0]}"
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
    return rescale_rows(stream.standard_normal((random.shape[0], N_IN)) / 8, mean_row_norm(random))


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
# ---- a fed unit joins the pool as unit N (G3 grow round 2, RULING.md items 1-6, AMENDMENT 1) ------

JOIN_TAG = "g3-join"


def join_stream(join_seed, n):
    """The declared join seed stream of the join that makes unit n: random.Random over the string
    "g3-join|<join_seed>|<n>" (a str seed is hashed by sha512, the same on every run and platform),
    so no join shares a stream with _lsm_draw's integer seeds or w_in's numpy stream."""
    return random.Random(f"{JOIN_TAG}|{int(join_seed)}|{int(n)}")


def _dale(sign):
    """A joined unit's write-once Dale sign (grown_code.GrownUnit.sign): +1 or -1, nothing else."""
    if isinstance(sign, bool) or sign not in (1, -1):
        raise ValueError(f"REFUSED: a joined unit's Dale sign is +1 or -1, got {sign!r}")
    return int(sign)


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
        """Swap the input matrix W_in for `w` (n x 64, finite). W_in is no part of the pool's
        state: its count, carriers, parameters and fingerprints are as they were, which is why a
        replay's binding records the swapped matrix's own crc."""
        w = _encoder_shaped(w, "the encoder", self.n)
        if not np.all(np.isfinite(w)):
            raise ValueError("REFUSED: the encoder holds a non-finite entry")
        self.w_in = np.array(w, dtype=np.float64)

    def percept(self, x):
        if self.w_in is None:
            raise ValueError("REFUSED: this pool was built without an encoder")
        return self.percept_drive(drive(self.w_in, x))

    # distinct-from: faculties/affect_integrator.py -- a jittered LIF unit pool, no external reservoir drive
    # distinct-from: faculties/task_set.py -- advances a working-memory pool with no external input
    # distinct-from: faculties/vital_nuclei.py -- a respiratory drive from CO2, not a reservoir's input
    # distinct-from: growth/neurogenesis_engine.py -- feeds an engine percepts; no pool drive is added
    # distinct-from: instruments/carried_state_driver.py -- runs stages under pool_selection's protocol
    # (all five read 2026-10-07; no caller of percept_drive or percept_fed exists outside this module)
    def percept_fed(self, x, feed):
        """G3 P5 feed port: percept with `feed` (one external drive per pool unit) added to W_in x.

        A zero feed is byte-identical to percept; a wrong-length or non-finite feed is REFUSED.
        """
        feed = np.asarray(feed)
        if (
            feed.dtype.kind not in "fiu"
        ):  # never cast a complex/object feed (1j -> 0, complex NaN hidden)
            raise ValueError(f"REFUSED: the feed must be real numbers, got dtype {feed.dtype}")
        feed = feed.astype(np.float64)
        if feed.shape != (self.n,) or not np.all(np.isfinite(feed)):
            raise ValueError(f"REFUSED: the feed must be {self.n} finite drives, got {feed.shape}")
        if self.w_in is None:
            raise ValueError("REFUSED: this pool was built without an encoder")
        return self.percept_drive(drive(self.w_in, x) + feed)

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

    # The network, not an arithmetic: stored lists, leak, gain, the window-0 boundary, the lesion's
    # rule, the parameters and R-b4's kick. Its float64 step (the port) went with the host switch
    # after G1(c)'s migration gate (2.9, 3.3); KernelLsmPool, the one pool implementation, supplies
    # the carriers, reset_carriers and step.
    name = "lsm"

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


# ---- g1 hosted on KernelTick (G1(c) 2.9) ---------------------------------------------------------


def _lineage_host(frac_bits, p_parts):
    """The hosted pool's fixed-point width and partition count: each as given, else the lineage's
    recorded value (the determinism gate's binding frac_bits, the partition baseline's P). A value
    that is not given and not recorded is refused: the host takes a measurement, never a pick."""
    from snn_data_lake.gates.determinism_gate import binding_frac_bits
    from snn_data_lake.reservoir.kernel_tick import PARTITION_BASELINE

    if frac_bits is None:
        frac_bits = binding_frac_bits()
        if frac_bits is None:
            raise ValueError(
                "REFUSED: no binding frac_bits is recorded (tools/determinism_baseline.json)"
            )
    if p_parts is None:
        if not PARTITION_BASELINE.exists():
            raise ValueError(
                "REFUSED: no lineage partition baseline is recorded (tools/kernel_partition_baseline.json)"
            )
        p_parts = json.loads(PARTITION_BASELINE.read_text())["P"]
    if int(frac_bits) < 1 or int(p_parts) < 1:
        raise ValueError("REFUSED: frac_bits and p_parts are positive counts")
    return int(frac_bits), int(p_parts)


class KernelLsmPool(LsmPool):
    """g1 hosted on KernelTick: LsmPool's network (the same draws, realization seed and index; never
    a second candidate), its recurrent weights compiled once into fixed-point bundles at the
    lineage's binding frac_bits (post = each unit's own index repeated k times, delay 1) and its
    external drive quantized per step as rint(g * u * 2**frac_bits). One KernelTick.tick is one pool
    step, and the activity is the fired mask as float64, so percept_drive, T_in, S_settle, the four
    forms, the window and its boundary are inherited unchanged. The arithmetic is the host's own:
    integer accumulation, a float32 membrane, the decay applied in float32, threshold 1.0, reset 0,
    no refraction (the weights and the drive round at 2**-frac_bits).

    Carriers: `v` (a float32 view of the state's membranes), `s` (the last tick's fired mask) and
    `acc` (the partitions' integer accumulators summed: the charge in flight, which at delay 1 is
    all of it between ticks). `reset_carriers` builds a fresh KernelState and KernelTick over the
    SAME compiled table. `frac_bits`, `p_parts` and `threads` are constructor options, defaulting to
    the lineage's record; a non-binding frac_bits is for fixtures and the quantization-sensitivity
    reading, and is named by host_policy().

    The class attributes are the host's policy and the two hooks below its quantization and edge
    list. G1(c) 2.9 kept this host and deleted the port and the host switch after the results record
    (3.3): this is the one pool implementation.
    """

    CARRIERS = ("v", "s", "acc")
    THR = LSM_THRESH
    V_RESET = 0.0
    REFRAC_TICKS = 0
    DELAY = 1

    def __init__(
        self,
        pre,
        w,
        config,
        form="carried",
        w_in=None,
        *,
        frac_bits=None,
        p_parts=None,
        threads=1,
        **kwargs,
    ):
        self.frac_bits, self.p_parts = _lineage_host(frac_bits, p_parts)
        self.threads = int(threads)
        self._syn = None  # compiled by the first reset_carriers, from self.pre and self.w
        self._kt = None
        self._s = None
        self._signs = None  # every born unit is unsigned; join() writes each joined unit's sign
        super().__init__(pre, w, config, form, w_in, **kwargs)

    # -- the host's tables -------------------------------------------------------------------------

    def _edges(self):
        """(pre, post, weight, delay) of the synapse table: row i of the stored lists is unit i's
        incoming entries, so post is i repeated k times; a duplicate (pre, post) pair is kept."""
        units, fan_in = self.pre.shape
        post = np.repeat(np.arange(units, dtype=np.int64), fan_in)
        return self.pre.ravel(), post, self.w.ravel(), np.full(units * fan_in, self.DELAY, np.int64)

    def _compile(self):
        pre, post, weight, delay = self._edges()
        return Synapses(pre, post, weight, delay, self.n, self.p_parts, self.frac_bits)

    def _quantize_drive(self, drive_vector):
        """The external drive in the host's fixed point: the same rint convention as the weights."""
        return np.rint(drive_vector * 2**self.frac_bits).astype(np.int64)

    def _start(self):
        state = KernelState([("pool", self.n)])
        state.arrays["thr"][:] = self.THR
        if self._signs is not None:  # the network's write-once signs, never a carrier a reset wipes
            state.arrays["sign"][:] = self._signs
        self._kt = KernelTick(
            state,
            self._syn,
            decay=self.lam,
            v_reset=self.V_RESET,
            refrac_ticks=self.REFRAC_TICKS,
            threads=self.threads,
        )
        self._s = np.zeros(self.n, dtype=bool)

    def _live_edges(self):
        """The live bundles' (post, pre, q) triples as three int64 arrays sorted by (post, pre, q):
        independent of P and of bundle order, a duplicate edge kept."""
        keys = sorted(self._syn.bundles)
        if not keys:
            empty = np.zeros(0, dtype=np.int64)
            return {"edge_post": empty, "edge_pre": empty, "edge_q": empty}
        posts = np.concatenate([self._syn.bundles[key][0] for key in keys])
        pres = np.concatenate(
            [np.full(len(self._syn.bundles[key][0]), key[0], np.int64) for key in keys]
        )
        qs = np.concatenate([self._syn.bundles[key][1] for key in keys])
        order = np.lexsort((qs, pres, posts))
        return {"edge_post": posts[order], "edge_pre": pres[order], "edge_q": qs[order]}

    # -- the carriers ------------------------------------------------------------------------------

    @property
    def v(self):
        return self._kt.state.arrays["v"]

    @v.setter
    def v(self, value):
        self._kt.state.arrays["v"][:] = value

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, value):
        self._s = np.array(value, dtype=bool)

    @property
    def acc(self):
        return np.sum(np.stack(self._kt._acc), axis=0)

    @acc.setter
    def acc(self, value):
        value = np.asarray(value, dtype=np.int64)
        for part, members in enumerate(self._kt._members):
            self._kt._acc[part][:] = 0
            self._kt._acc[part][members] = value[members]

    def reset_carriers(self):
        if self._syn is None:
            self._syn = self._compile()
        self._start()

    def step(self, u):
        u = np.asarray(u, dtype=np.float64)
        fired = self._kt.tick(self._quantize_drive(self.g * u))
        s = np.zeros(self.n, dtype=bool)
        s[fired] = True
        self._s = s
        return s.astype(np.float64)

    # -- the front-end lesion, the fingerprints, the policy ----------------------------------------

    def _zero_in_flight(self):
        for part in range(self.p_parts):
            self._kt._acc[part][:] = 0

    def lesion(self):
        """Every recurrent weight zero (the port's rule), the table recompiled, and every in-flight
        accumulator zeroed: charge already delivered into acc was computed from the old weights,
        which the port's zero-weight sum would never add at its next step."""
        super().lesion()
        self._syn = self._compile()
        self._kt.syn = self._syn
        self._zero_in_flight()

    def params_fingerprint(self):
        return _crc_arrays(self._live_edges(), _crc_arrays(self.params()))

    # -- a fed unit joins as unit N (RULING.md items 1-6, AMENDMENT 1) -----------------------------

    @property
    def signs(self):
        """Each unit's write-once Dale sign: 0 for a born (unsigned) unit, s_j for a joined one."""
        return np.zeros(self.n, dtype=np.int8) if self._signs is None else self._signs

    def _join_stream(self, join_seed):
        return join_stream(join_seed, self.n)

    def _retarget_p(self):
        """A4's coin: 2/(N+1), so with half the entries sign-matched each ends at N w.p. 1/(N+1)."""
        return 2 / (self.n + 1)

    def join(self, sign, join_seed, encoder_row=None):
        """Draw the join that makes unit N = self.n and apply it; returns the applied journal row."""
        return self.apply_join(self.draw_join(sign, join_seed, encoder_row))

    def draw_join(self, sign, join_seed, encoder_row=None):
        """The join row that makes unit N = self.n, drawn from the declared join stream and never
        applied (apply_join's input; a caller journals it first). Row N first (B1): k pairs
        (randrange(N+1), uniform(-w, w)), an entry whose pre is a signed unit (self included)
        weighted s_pre * |u| (AMENDMENT 1). Then A4's coin, sign-matched: every existing entry whose weight sign is
        `sign`, rows then slots in order, moves to pre = N with probability _retarget_p(), keeping
        its weight. A row with no entry of that sign takes no edge from N: counted, never forced.
        No cap: N is any size; the pace is the caller's (GrowthLoop's admission, P4d budget)."""
        sign = _dale(sign)
        bound = self.config.get("w")
        if bound is None:
            raise ValueError("REFUSED: a join draws by the pool's weight bound w; none is set")
        n, k = self.n, self.pre.shape[1]
        rng = self._join_stream(join_seed)
        signs = np.append(self.signs, np.int8(sign))
        pre_row, w_row = [], []
        for _ in range(k):
            p, u = rng.randrange(n + 1), rng.uniform(-float(bound), float(bound))
            pre_row.append(p)
            w_row.append(u if signs[p] == 0 else int(signs[p]) * abs(u))
        matched = np.sign(self.w) == sign
        chance = self._retarget_p()
        retargeted = [[int(i), int(c)] for i, c in zip(*np.nonzero(matched), strict=True)]
        retargeted = [cell for cell in retargeted if rng.random() < chance]
        record = {
            "n": n,
            "sign": sign,
            "seed": int(join_seed),
            "retargeted": retargeted,
            "pre_row": pre_row,
            "w_row": w_row,
            "encoder_row": None if encoder_row is None else [float(x) for x in encoder_row],
            "rows_without_sign": int(np.count_nonzero(~matched.any(axis=1))),
        }
        return record

    def apply_join(self, record, *, dry=False):
        """Apply one join row: the live join and a restart's rebuild (P4b) are this one path, and
        neither draws. Unit N takes row N, its sign and its encoder row; each listed (row, slot)
        entry points at N with its weight kept; the carriers, the accumulators and the tick index
        are carried bytewise (unit N starts at rest); the table is recompiled and the boundary
        fingerprint re-anchored (a journaled re-boundary per join). Returns the record with the
        new boundary. REFUSED, before any write: a record for another N, a lesioned pool, a bundle
        in flight, a malformed row, a signed pre whose weight breaks its sign, a retargeted entry
        whose weight is not the unit's sign (C17), an encoder row that is not 64 finite values.
        `dry` runs every refusal and writes nothing (a write-ahead journal checks before it writes)."""
        n, k = self.n, self.pre.shape[1]
        sign = _dale(record["sign"])
        if int(record["n"]) != n:
            raise ValueError(f"REFUSED: the join row makes unit {record['n']}, this pool's is {n}")
        if self.lesion_record is not None:
            raise ValueError("REFUSED: a lesioned front_end pool takes no join")
        if any(slot for ring in self._kt._rings for slot in ring):
            raise ValueError("REFUSED: a delay ring holds an in-flight bundle at a join")
        pre_row = np.asarray(record["pre_row"], dtype=np.int64)
        w_row = np.asarray(record["w_row"], dtype=np.float64)
        if pre_row.shape != (k,) or w_row.shape != (k,) or not np.all(np.isfinite(w_row)):
            raise ValueError(f"REFUSED: row N is k = {k} finite entries")
        if np.any(pre_row < 0) or np.any(pre_row > n):
            raise ValueError(f"REFUSED: row N's pre indices lie outside 0..{n}")
        signs = np.append(self.signs, np.int8(sign))
        if np.any((signs[pre_row] != 0) & (np.sign(w_row) * signs[pre_row] < 0)):
            raise ValueError("REFUSED: an entry from a signed unit breaks its Dale sign (C17)")
        cells = np.asarray(record["retargeted"], dtype=np.int64).reshape(-1, 2)
        if np.any(cells < 0) or np.any(cells[:, 0] >= n) or np.any(cells[:, 1] >= k):
            raise ValueError("REFUSED: a retargeted entry lies outside the pool's rows and slots")
        if np.any(np.sign(self.w[cells[:, 0], cells[:, 1]]) != sign):
            raise ValueError("REFUSED: a retargeted entry's weight is not the unit's sign (C17)")
        row = record.get("encoder_row")
        if (row is None) != (self.w_in is None):
            raise ValueError("REFUSED: an encoder row joins a pool with an encoder, and only one")
        if row is not None:
            row = np.asarray(row, dtype=np.float64)
            if row.shape != (N_IN,) or not np.all(np.isfinite(row)):
                raise ValueError(f"REFUSED: the encoder row is not {N_IN} finite values")
        if dry:
            return None
        pre = np.array(grow_rows(self.pre.tolist(), n + 1), dtype=np.int64)
        weights = np.array(grow_rows(self.w.tolist(), n + 1), dtype=np.float64)
        pre[n], weights[n] = pre_row, w_row
        pre[cells[:, 0], cells[:, 1]] = n
        old_kt, old_s = self._kt, self._s
        self.pre, self.w, self.n, self._signs = pre, weights, n + 1, signs
        if row is not None:
            self.w_in = np.array(grow_rows(self.w_in.tolist(), n + 1), dtype=np.float64)
            self.w_in[n] = row
        self._syn = self._compile()
        self._start()
        for key, old in old_kt.state.arrays.items():
            fresh = self._kt.state.arrays[key]
            fresh[:] = grow_vector(old.tolist(), n + 1, fill=fresh[n].item())
        for part in range(self.p_parts):
            self._kt._acc[part][:] = grow_vector(old_kt._acc[part].tolist(), n + 1)
        self._kt.tick_index = old_kt.tick_index
        self._s = np.array(grow_vector(old_s.tolist(), n + 1, fill=False), dtype=bool)
        if self.boundary is not None:
            self.boundary = self.params_fingerprint()
        return dict(record, n=n, sign=sign, boundary=self.boundary)

    def fingerprint(self):
        """crc32 of the complete continuation state, independent of P: the percept count and the
        tick index, the carriers (v float32, the last fired mask, the summed accumulators), EVERY
        array of the host's KernelState (v, thr, refrac, last_spike, sign: the refractory counter
        and the adaptive threshold are state the carriers do not name, and a host that differs only
        in them continues differently), the parameters and the live edges. Every delay is 1, so every
        ring slot is drained within its tick; a fingerprint taken with a bundle in flight is
        refused."""
        if any(slot for ring in self._kt._rings for slot in ring):
            raise ValueError("REFUSED: a delay ring holds an in-flight bundle at a fingerprint")
        crc = zlib.crc32(f"count|{self.count}|tick|{self._kt.tick_index}".encode("ascii"))
        crc = _crc_arrays(self.carriers(), crc)
        crc = _crc_arrays(self._kt.state.arrays, crc)
        return _crc_arrays(self._live_edges(), _crc_arrays(self.params(), crc))

    @classmethod
    def policy_of(cls, frac_bits, p_parts, lam):
        """Kernel host policy dict.

        The host's arithmetic, built in this one place: the fixed-point width,
        the partition count and the decay as float32 sees it, then this class's reset, refraction,
        delay and threshold. host_policy() reads it off an instance and lineage_policy() off the
        lineage's records; neither restates a constant."""
        return {
            "host": "kernel",
            "frac_bits": int(frac_bits),
            "p_parts": int(p_parts),
            "decay_f32": float(np.float32(lam)),
            "v_reset": float(cls.V_RESET),
            "refrac_ticks": int(cls.REFRAC_TICKS),
            "delay": int(cls.DELAY),
            "thr": float(np.float32(cls.THR)),
        }

    @classmethod
    def lineage_policy(cls, config):
        """Lineage-recorded policy of a configuration.

        policy_of at the lineage's recorded frac_bits and partition count (the options a campaign
        replay runs at) and the configuration's leak: what R-b7 holds a hosted binding's whole
        policy equal to. An absent lineage record raises, as building the host does."""
        frac_bits, p_parts = _lineage_host(None, None)
        return cls.policy_of(frac_bits, p_parts, float(config["lambda"]))

    def host_policy(self):
        """Instance policy dict.

        What a replay's binding records of this host: policy_of at its own options and leak."""
        return self.policy_of(self.frac_bits, self.p_parts, self.lam)

    def __deepcopy__(self, memo):
        """copy.deepcopy fails on a KernelState and on a KernelTick (each holds the numpy module as an
        attribute: TypeError, cannot pickle 'module' object), so the copy is rebuilt: the table is
        deep-copied (its bundles are not shared), a fresh KernelState and KernelTick are built over
        the copy, and the runtime (the state arrays, the tick index, the integer accumulators and the
        ring slots) is carried over by the replay gate's own capture and restore, the one
        serialization of a kernel's runtime."""
        from snn_data_lake.gates.replay_gate import capture, restore

        dup = self.__class__.__new__(self.__class__)
        memo[id(self)] = dup
        host = ("_kt", "_syn", "_s")
        for key, value in self.__dict__.items():
            if key not in host:
                setattr(dup, key, copy.deepcopy(value, memo))
        dup._syn = copy.deepcopy(self._syn, memo)
        dup._start()
        restore(capture(self._kt, {}), dup._kt)
        dup._s = self._s.copy()
        return dup


# distinct-from: faculties/spiking_curiosity.py -- its _make_pool builds a LIF pool that turns a prediction error into a count for curiosity; no recurrence, no percept stream (site read)
# G1(c) 2.9: the migration gate chose the hosted form (3.3), and its loser (the port) and the host
# switch are deleted; tests/test_recurrent_pool.py counts one reachable pool implementation.
POOLS = {"lsm": KernelLsmPool}


# Sites the duplicate-capability hook flagged, read; neither is a second pool builder:
# distinct-from: faculties/spiking_curiosity.py::_make_pool -- builds a log-spaced-tau LIF pool for a curiosity count; no recurrence, no percept stream
# distinct-from: instruments/memory_diagnostic.py::_PoolArm -- the CALLER of this make_pool (its docstring names it); it builds no pool itself
def make_pool(
    candidate,
    config,
    recording_seed,
    form="carried",
    *,
    window=WINDOW,
    s_settle=S_SETTLE,
    **host_options,
):
    """A pool from (candidate, config, recording seed): W_in from the recording seed, the random
    structure from realization_seed, drawn in spiking_ssm's exact call order. `canonical` raises for
    any candidate or configuration outside section 2.3's g1 grid. The pool is KernelLsmPool, whose
    options (frac_bits, p_parts, threads) default to the lineage's record."""
    config = canonical(candidate, config)
    seed = realization_seed(candidate, recording_seed)
    encoder = w_in(recording_seed)
    kwargs = {"window": window, "s_settle": s_settle}
    _, pre, weights = _lsm_draw(seed, N, config["k"], config["w"])
    return POOLS[candidate](pre, weights, config, form, encoder, **kwargs, **host_options)


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


# ---- R-b0: the draws against the source (section 2.7) --------------------------------------------


def check_r_b0_lsm(seed=0, red_shift=1):
    """R-b0's structure clause: g1's draw (N 100: the n uniform(-1, 1) input draws consumed first,
    then each unit's k (pre, weight) pairs) equals spiking_ssm.build_reservoir's, and the threshold is
    the source's. Its parity clause (the float64 port's rates and SynOps against run_reservoir, bit for
    bit) went with the port (G1(c) 2.9; commit 6dbe61e2 holds it). Red: the draw at seed + red_shift
    must not match."""
    from snn_data_lake.reasoning import spiking_ssm as src

    w_src, rec = src.build_reservoir(seed)

    def matches(draw_seed):
        consumed, pre, weights = _lsm_draw(draw_seed, src.N, src.FANIN, SOURCE_W)
        return (
            consumed == w_src
            and pre.tolist() == [[j for j, _ in row] for row in rec]
            and weights.tolist() == [[x for _, x in row] for row in rec]
            and LSM_THRESH == src.THRESH
        )

    structure, broke = matches(seed), not matches(seed + red_shift)
    return {
        "verdict": bool(structure and broke),
        "numbers": {"structure": bool(structure), "units": src.N, "fan_in": src.FANIN},
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


# R-b5 (iii)'s red on the hosted pool (G1(c) 2.9): the stored `w` is only the source the table was
# compiled from; the charge is delivered from the LIVE bundles, so the red writes q + 1 into one live
# bundle weight right after the boundary and leaves `w` as it was.
# A plain assignment, not an in-place add: the write is the fault's whole body. Sites the hook flagged, read:
# distinct-from: reservoir/kernel_tick.py::compile_bundles -- COMPILES the bundles once; this writes one compiled q after the boundary, only so the freeze red can fail
# distinct-from: engine/live_brain.py::jepa_on -- a predicate on a wired world model (docstring: "after >= 1 live tick"); no write
# distinct-from: gates/selfcheck_cache.py::SelfcheckCache -- consult and record of a cached self-check row after a live PASS
# distinct-from: sensorimotor/afferent_boundary.py::proprioception_deltas -- a per-joint angle difference, theta_after - theta_before
# distinct-from: sleep/experience_replay.py::add -- records one transition into a replay buffer
# distinct-from: substrate/proximity_replay.py::cross_process_durability_check -- reads a lake back from a fresh process
def _write_live_after_boundary(pool):
    base = type(pool)

    def _at_boundary(self):
        base._at_boundary(self)
        quantized = self._syn.bundles[sorted(self._syn.bundles)[0]][1]
        quantized[0] = quantized[0] + 1

    return _fault(pool, "write_live_after_boundary", _at_boundary=_at_boundary)


# ---- R-b3: determinism by construction (section 2.7) ---------------------------------------------


def _record(pool, percepts, *, chain=False):
    tally = {"crc": 0, "steps": 0}

    def observe(_, activity):
        tally["crc"] = zlib.crc32(activity.tobytes(), tally["crc"])
        tally["steps"] += 1

    pool.observer = observe
    phi = linked = 0
    for x in percepts:
        phi = zlib.crc32(pool.percept(x).tobytes(), phi)
        # the hosted R-b3 (G1(c) 2.9): the state fingerprint at EVERY percept boundary, chained
        if chain:
            linked = zlib.crc32(pool.fingerprint().to_bytes(4, "little"), linked)
    pool.observer = None
    found = {
        "activity_crc": tally["crc"],
        "steps": tally["steps"],
        "phi_crc": phi,
        "state": pool.fingerprint(),
    }
    return dict(found, state_chain=linked) if chain else found


# R-b3's comparator, which the equality and its red both read: every run the same
def _runs_agree(runs):
    return len({json.dumps(run, sort_keys=True) for run in runs}) == 1


# G1(c) 2.9 re-founded R-b3's red: a permuted-float-sum red cannot fire on integer delivery, and
# lattice weights at 2**-12 add exactly in float32 in any order, so the red is a float-accumulation
# twin of the hosted pool, fed RAW weights with a declared magnitude spread, summed in an order
# nothing seeds.


class _FloatTwin(KernelLsmPool):
    """A hosted pool whose recurrent delivery sums RAW float32 weights, in the order ORDER names:
    "unseeded" draws a fresh os.urandom permutation of each unit's incoming entries every tick,
    "forward" keeps the stored order (the control: deterministic). The network is the hosted pool's
    (same draws, edges and quantized drive); entry 0 of every unit is scaled UP by R_B3_SPREAD and
    entry 1 DOWN by it, the declared spread. `multi` counts unit-steps with three or more spiking
    inputs and `spread` those among them where both scaled entries spike. Reached only by the red."""

    ORDER = "unseeded"

    def __init__(self, pre, w, config, form="carried", w_in=None, **kwargs):
        raw = np.array(w, dtype=np.float64)
        raw[:, 0] = raw[:, 0] * R_B3_SPREAD
        raw[:, 1] = raw[:, 1] / R_B3_SPREAD
        super().__init__(pre, raw, config, form, w_in, **kwargs)
        self.multi = self.spread = 0

    def step(self, u):
        u = np.asarray(u, dtype=np.float64)
        spiking = self.s[self.pre]  # the inputs delivered this tick (every delay is 1)
        crowded = np.count_nonzero(spiking, axis=1) >= 3
        self.multi += int(np.count_nonzero(crowded))
        self.spread += int(np.count_nonzero(crowded & spiking[:, 0] & spiking[:, 1]))
        terms = np.where(spiking, self.w.astype(np.float32), np.float32(0.0))
        width = terms.shape[1]
        order = _unseeded_permutation(width) if self.ORDER == "unseeded" else np.arange(width)
        delivered = np.cumsum(terms[:, order], axis=1, dtype=np.float32)[:, -1]
        injected = self._quantize_drive(self.g * u) / float(2**self.frac_bits) + delivered
        v = (self.v * self.lam + injected).astype(np.float32)  # the host's dtype path
        fired = v >= self.THR
        self.v = np.where(fired, np.float32(self.V_RESET), v)
        self._s = fired
        self._kt.tick_index += 1
        return fired.astype(np.float64)


class _ForwardTwin(_FloatTwin):
    ORDER = "forward"


def _float_twin(candidate, config, recording_seed, form="carried", *, window, order):
    base = make_pool(candidate, config, recording_seed, form, window=window)
    cls = {"unseeded": _FloatTwin, "forward": _ForwardTwin}[order]
    return cls(
        base.pre, base.w, base.config, form, base.w_in, window=window, s_settle=base.s_settle
    )


def _unit_spread(weights):
    """The smallest over units of (largest |weight| / smallest |weight|) in a unit's incoming set."""
    magnitude = np.abs(np.asarray(weights, dtype=np.float64))
    with np.errstate(divide="ignore"):
        return float(np.min(np.max(magnitude, axis=1) / np.min(magnitude, axis=1)))


def check_r_b3(
    candidate,
    config,
    recording_seed=DEV_SEED,
    *,
    percepts=R_B3_PERCEPTS,
    window=WINDOW,
    forms=("carried", "settled"),
):
    """R-b3 on the pool alone, on the 16 fixtures cycled to `percepts`: every step's activity, the phi,
    the final state and the CHAIN of the state at every percept boundary equal across (P, threads) in
    R_B3_HOSTED, per form. The BLAS and torch axes are vacuous here (the pool's step and drive use
    np.cumsum and np.add.at only)."""
    seq = cycled_fixture(percepts)
    runs = {}
    for p_parts, threads in R_B3_HOSTED:
        for form in forms:
            pool = make_pool(
                candidate,
                config,
                recording_seed,
                form,
                window=window,
                p_parts=p_parts,
                threads=threads,
            )
            runs[f"P{p_parts}/T{threads}/{form}"] = _record(pool, seq, chain=True)
    equal = {
        form: _runs_agree(run for key, run in runs.items() if key.endswith("/" + form))
        for form in forms
    }
    return {
        "verdict": bool(all(equal.values())),
        "numbers": {
            "equal": equal,
            "settings": [{"P": p, "threads": t} for p, t in R_B3_HOSTED],
            "blas": "vacuous: no BLAS product in the hosted pool",
            "torch": "vacuous: the hosted pool never loads torch",
            "runs": runs,
        },
    }


def check_r_b3_hosted_red(
    candidate="lsm", recording_seed=DEV_SEED, *, percepts=R_B3_PERCEPTS, window=WINDOW
):
    """The hosted R-b3's red on the declared witness: the float-accumulation twin summed in an
    unseeded order must change the continuation-state CHAIN against the same twin in the stored
    order, through the hosted equality's own comparator (`_runs_agree` over `_record(chain=True)`);
    the stored order twice must agree (the divergence is the order and nothing else). The fixture
    must exercise it: at least R_B3_MULTI unit-steps with three or more simultaneous presynaptic
    spikes in each run, as many among them with both scaled entries spiking, and every unit's
    incoming weights spanning at least R_B3_SPREAD. `determinism_gate.broken_control()` runs beside
    it as a wiring control only: its output is reported, and it decides nothing here."""
    from snn_data_lake.gates.determinism_gate import broken_control

    config = R_B3_WITNESS[candidate]
    seq = cycled_fixture(percepts)
    pools = {
        label: _float_twin(candidate, config, recording_seed, window=window, order=order)
        for label, order in (("forward", "forward"), ("again", "forward"), ("unseeded", "unseeded"))
    }
    runs = {label: _record(pool, seq, chain=True) for label, pool in pools.items()}
    control = _runs_agree([runs["forward"], runs["again"]])
    differs = not _runs_agree([runs["forward"], runs["unseeded"]])
    chain_changed = runs["forward"]["state_chain"] != runs["unseeded"]["state_chain"]
    counts = {
        label: {"multi": pools[label].multi, "spread": pools[label].spread}
        for label in ("forward", "unseeded")
    }
    exercised = all(
        row["multi"] >= R_B3_MULTI and row["spread"] >= R_B3_MULTI for row in counts.values()
    )
    span = _unit_spread(pools["forward"].w)
    float_diverged, int_agreed = broken_control()
    return {
        "verdict": bool(
            differs and chain_changed and control and exercised and span >= R_B3_SPREAD
        ),
        "numbers": {
            "config": config,
            "exercise_count": counts["forward"]["multi"],
            "counts": counts,
            "unit_spread_min": span,
            "control_agrees": bool(control),
            # which observable the unseeded order moved (the activity trains may agree while v's bits do not)
            "changed": {
                key: runs["forward"][key] != runs["unseeded"][key]
                for key in ("activity_crc", "phi_crc", "state", "state_chain")
            },
            "differs": bool(differs),
            "chain_changed": bool(chain_changed),
            "wiring": {"float_diverged": bool(float_diverged), "int_agreed": bool(int_agreed)},
        },
        "red": bool(differs and chain_changed),
    }


# ---- R-b4's pool-noise red (the pool's half; the fresh-process replays are the harness's) --------


# Sites the duplicate-capability hook flagged for a pool-noise red's name, read; none kicks a pool's membrane:
# distinct-from: instruments/pool_selection.py::noise_red_fires -- COMPARES two finished replays' state chains and phi (R-b4's harness half); this builds the pool-only twins and kicks one
# distinct-from: faculties/drive_reduction.py::NeedPool -- a drive's LIF need pool read as a rate; no kick, no equality test
# distinct-from: faculties/global_workspace.py::_fold_parts -- the ordered shared-inhibition sum of a workspace settle; no noise
# distinct-from: faculties/time_cells.py::tick -- advances a seeded-Gaussian ramp as a timing model; the noise here is unseeded and what must FAIL is an equality
# distinct-from: growth/growing_learner.py::_jitter -- seeded jitter on an input vector, clamped to [0, 1]
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
    candidate,
    config,
    form,
    recording_seed=DEV_SEED,
    *,
    window=WINDOW,
    prefix=8,
    skip=None,
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


# Sites the duplicate-capability hook flagged for an erasure red's name, read; none skips a pool's carrier:
# distinct-from: faculties/agency_monitor.py::t079_reds -- six reds of a world-model instrument gathered into one self-check line
# distinct-from: faculties/metacognitive_monitor.py::t077_reds -- six reds of a metacognition instrument as one self-check line
# distinct-from: faculties/recursive_tom.py::_t082_s2 -- a per-seed theory-of-mind control check, an exact count
# distinct-from: faculties/spiking_storage_prototype.py::check_reds -- runs a storage prototype's RED_CONFIGS against its criteria
# distinct-from: gates/amdahl_gate.py::self_check_reds -- the Amdahl gate's three pre-registered timing controls
def check_r_b5_erasure_red(candidate, form, recording_seed=DEV_SEED, *, window=WINDOW):
    """The declared witness with its declared carrier skipped must FAIL erasure; every other
    carrier's skip is reported beside it."""
    config, carrier = R_B5_ERASURE_WITNESS[candidate]
    found = {
        name: check_r_b5_erasure(candidate, config, form, recording_seed, window=window, skip=name)
        for name in POOLS[candidate].CARRIERS
    }
    skips = {name: not run["verdict"] for name, run in found.items()}
    numbers = {"config": config, "carrier": carrier, "fired": skips}
    # G1(c) 2.9: HOW each skipped carrier leaked -- through phi, the continuation state, or neither
    # (the hosted `s` is a recorded mask the delay rings never read, so it can only reach the state)
    numbers["by"] = {
        name: (
            "+".join(
                label
                for label, same in (
                    ("phi", run["numbers"]["same_phi"]),
                    ("state", run["numbers"]["same_state"]),
                )
                if not same
            )
            or "none"
        )
        for name, run in found.items()
    }
    return {"verdict": bool(skips[carrier]), "numbers": numbers}


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
            # the hosted form's freeze is re-based on its LIVE bundles: the red writes one of them
            pool = _write_live_after_boundary(pool)
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
# and the five the erasure red's marker above names, read for this name too, none a freeze red:
# distinct-from: faculties/agency_monitor.py::t079_reds -- six reds of a world-model instrument gathered into one self-check line
# distinct-from: faculties/metacognitive_monitor.py::t077_reds -- six reds of a metacognition instrument as one self-check line
# distinct-from: faculties/spiking_storage_prototype.py::check_reds -- runs a storage prototype's RED_CONFIGS against its criteria
# distinct-from: gates/amdahl_gate.py::self_check_reds -- the Amdahl gate's three pre-registered timing controls
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


# ---- the hand fixture (the golden check's, G1(c) 2.11) -------------------------------------------

# G1(c) 2.9's parity fixtures stepped the port and the host side by side (the migration gate's part
# (i)); they went with the port. The hand fixture stays: carried_state_report's golden check runs it
# through the hosted pool's four forms (`_hosted`, `_percepted`).

HAND_PRE = [[1, 2], [0, 0], [1, 0]]  # the hand fixture of tests/test_recurrent_pool.py (N 3, k 2)
HAND_W = [[0.5, 0.25], [1.0, -0.5], [0.5, 0.5]]
HAND_CFG = {"lambda": 0.5, "g": 2, "T_in": 2}
HAND_U = ([0.25, 0.5, 0.375], [0.0, 0.25, 0.0], [0.0, 0.3125, 0.0])


def _hosted(host_cls, bits, pre, w, config, form="carried", **kwargs):
    return host_cls(pre, w, config, form, frac_bits=bits, p_parts=1, **kwargs)


def _row(pool, activity):
    return (
        activity.tobytes(),
        np.asarray(pool.v, dtype=np.float64).tobytes(),
        np.asarray(pool.s, dtype=bool).tobytes(),
    )


def _percepted(pool, drives):
    return [(*_row(pool, pool.percept_drive(u)), pool.count) for u in drives]


# ---- the self-check ------------------------------------------------------------------------------


def main():
    t0 = time.perf_counter()
    crcs = draw_order_crc32()
    assert crcs == draw_order_crc32(), "the draw order is not deterministic"
    assert [len(grid(c)) for c in CANDIDATES] == [288]
    print("draw-order crc32: " + ", ".join(f"{c}={crcs[c]}" for c in CANDIDATES))
    for name, result in check_r_b0().items():
        assert result["verdict"] and result["red"], (name, result["numbers"])
        print(
            f"R-b0 {LABEL[name]} {name}: the draws equal the source's and the shifted draw's red OK"
        )
    small = {"percepts": 64, "window": 32}
    # the one pool implementation, each check beside its constructed red
    for candidate in CANDIDATES:
        config, _ = R_B5_ERASURE_WITNESS[candidate]
        equal = check_r_b3(candidate, R_B3_WITNESS[candidate], **small)
        assert equal["verdict"], (candidate, equal["numbers"]["equal"])
        red = check_r_b3_hosted_red(candidate, **small)
        assert red["verdict"], (candidate, red["numbers"])
        wired = red["numbers"]["wiring"]
        assert wired["float_diverged"] and wired["int_agreed"], wired
        # R-b4's kicks are unseeded: on the erasure witness (saturated drive) 6 of 150 hosted trials
        # left phi unmoved at 16 kicks, so this red runs on the persistence witness (0 of 150)
        noise = pool_noise_red(candidate, R_B5_PERSISTENCE_WITNESS, window=32)
        assert noise["verdict"], (candidate, noise["numbers"])
        by = {}
        for form in ("reset", "settled"):
            assert check_r_b5_erasure(candidate, config, form, window=32)["verdict"]
            erasure = check_r_b5_erasure_red(candidate, form, window=32)
            assert erasure["verdict"], (candidate, form, erasure["numbers"])
            by[form] = erasure["numbers"]["by"]
        assert check_r_b5_freeze(candidate, config, window=32, after=4)["verdict"]
        freeze = check_r_b5_freeze_red(candidate, config, window=32, after=4)
        assert freeze["verdict"], (candidate, freeze["numbers"])
        print(
            f"R-b3 {LABEL[candidate]}: equal across (P, threads) {list(R_B3_HOSTED)} on the "
            f"witness, BLAS and torch axes vacuous; float-accumulation red fired "
            f"({red['numbers']['exercise_count']} unit-steps with 3+ spikes, "
            f"{red['numbers']['counts']['forward']['spread']} with both scaled entries, per-unit "
            f"spread min {red['numbers']['unit_spread_min']:.0f} >= {R_B3_SPREAD}); broken_control "
            f"(float diverged {wired['float_diverged']}, int agreed {wired['int_agreed']}); R-b4 "
            f"pool-noise red fired; R-b5 erasure skip-a-carrier fired by {by['settled']}; freeze "
            f"red on the live bundles fired on {freeze['numbers']['fired']}"
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
        f"self-check OK: R-b0's draws, R-b3, R-b4 pool noise, R-b5 and their reds on the one pool implementation "
        f"({time.perf_counter() - t0:.1f} s)"
    )


if __name__ == "__main__":
    main()
