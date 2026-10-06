r"""T3 (G4 / T112): does H01 favour a Gaussian distance kernel? -- the pre-registered instrument.

The pre-registration is `specs/022-unified-brain-arc/evidence/g4-wiring-prior.md` section 2 with its
section 2.14 amendments (A1-A14), FROZEN at 857163d4 before a line of this module existed. This file
implements it and adds nothing to it; where the code needed a choice the text did not make, the
choice is named in a comment with the words "NOT IN THE TEXT".

WHAT IT MEASURES. On H01's c3 graph (14,215 units, 70,001 directed connected pairs; the artifact
pinned in `tools/h01_c3_graph_pin.json`), the probability that an ordered pair is connected, as a
function of 3-D soma distance, binned at 1 um. Four models are fitted by maximum binomial likelihood
(2.3): flat p0; Gaussian p0 exp(-d^2 / 2 sigma^2); exponential p0 exp(-d / lambda); the mixture
a exp(-d^2 / 2 sigma^2) + c exp(-d / lambda). Each is scored by held-out log-likelihood under three
scorings (2.4, A8): K = 5 NEURON folds per partition seed (the primary), four x-quartile REGION
blocks by polar angle about the white-matter circle centre (the second instrument whose agreement
carries the decision, L3; x quartiles until the 2026-10-04 amendment), and the Z-INTERIOR subset
(both units in the central half of the slab's z extent).

THE READING (A2-A3) is a pure function of tri-state contrasts (+1 / 0 / -1 against the null bands):
the kernel K is selected first and verdict-independently, then the rows are read top to bottom:
REFUSED, NOT_READ_REGION_INVALID, RED_FLAT, NOT_READ_BOUNDARY, NOT_READ_DISAGREE, RED_FLAT_REGION,
RED_EXP, PASS_* . A degenerate (zero-variance) contrast is NOT MET (ruling 9, disclosed as stricter
than `paired_contrast.contrast`); band B sits on top of the paired t.

THE ORDER IS ENFORCED, NOT ADVISED (A5, A6, A8). `run_planted_stage` reads POSITIONS, layer bands
and degree MARGINALS only: the flat CALIBRATION set defines the bands, they are locked (sha256),
then every other planted truth -- the untouched flat VALIDATION set among them -- must read its
expected row through this same reading function. `score_h01` is the only entry point that reads
H01's edges, and it REFUSES unless it is handed a planted record whose rows all hold and whose bands
still hash to the locked value. `final_width_record` refuses the same way.

DISCLOSED, NOT HANDLED (2.1): the observed kernel is anatomy times reconstruction completeness; the
planted stage proves the FITTER is not blind, never that the DATA's kernel is anatomical. FR-007:
H01 is a measurement of a human brain a microscope saw, not of this brain; anything here is a prior.

No scipy: it is not a dependency of this package (pyproject `dependencies`), so the pre-registered
Nelder-Mead (2.3: xatol = fatol = 1e-6, 4,000 iterations) is implemented here with scipy's default
coefficients and initial simplex; the test cross-checks it against scipy when scipy is installed.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.h01_distance_kernel        (self-check)
  ... h01_distance_kernel planted --artifact data/h01/h01_c3_graph.npz --out <planted.json>   (CAMPAIGN)
  ... h01_distance_kernel h01 --artifact data/h01/h01_c3_graph.npz --planted <planted.json> --out <h01.json>
"""

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast

REPO = Path(__file__).resolve().parents[3]
GRAPH_PIN = REPO / "tools" / "h01_c3_graph_pin.json"

N_NODES = 14215
N_EDGES = 70001
K_FOLDS = 5  # 2.4
N_BLOCKS = 4  # 2.4: angle quartiles (amendment 2026-10-04; x quartiles before)
N_BANDS = 7  # L1-L6 + white matter, the artifact's `layer` codes 0..6
N_GROUPS = K_FOLDS * N_BLOCKS  # a unit's histogram group: fold x block
FOLD_OF = np.arange(N_GROUPS) // N_BLOCKS
BLOCK_OF = np.arange(N_GROUPS) % N_BLOCKS
BIN_UM = 1.0  # 2.2
CHUNK = 256
Z_TRIM = 0.25  # 2.4 / A8: a quarter of the z thickness trimmed from each face (41.9 um on H01)
# NOT IN THE TEXT: a units guard. H01's slab diagonal is 3,727 um; no c3-class volume spans a
# centimetre, so a larger extent is positions left in nm (2.11 (b)) and is refused before any bin.
MAX_EXTENT_UM = 10_000.0
WIDTH_GRID = np.logspace(1.0, 3.5, 40)  # 2.3: 40-point log grid over [10, 3162] um
NM_TOL = 1e-6
NM_MAXITER = 4000
P_LO, P_HI = 1e-300, 1.0 - 1e-15  # 2.3's clip
RECOVERY_TOL = 0.10  # A6: sigma within [90, 110] um for the planted 100
WIDTH_RTOL = 1e-6  # 2.11 (f)

MODELS = ("flat", "gauss", "exp", "mix")
KERNELS = ("gauss", "exp", "mix")
CONTRASTS = (("gauss", "flat"), ("exp", "flat"), ("mix", "flat"), ("exp", "gauss"), ("mix", "gauss"), ("mix", "exp"))
CONTRAST_NAMES = tuple(f"{a}-{b}" for a, b in CONTRASTS)

CONFIRMATORY_SEEDS = tuple(range(10100, 10108))  # 2.12, read ONCE
CONFIRMATORY_RESERVE = tuple(range(10108, 10116))


class ArtifactRefused(ValueError):
    """The artifact is not the pinned one, or its units are wrong."""


class ExposureRefused(RuntimeError):
    """An edge read was attempted before the planted record holds (A8)."""


class RecordRefused(ValueError):
    """A width record does not reproduce on refit (2.11 (f))."""


class PlanRow(NamedTuple):
    name: str
    truth: tuple
    seeds: tuple
    expect: object  # None | "red_flat" | "gauss" | "exp" | "mix"
    width_check: bool


def _rows(spec):
    return tuple(PlanRow(n, t, tuple(range(s, s + 8)), e, w) for n, t, s, e, w in spec)


# A6, frozen: the table's order is the reserve's order.
_PLAN_SPEC = (
    ("flat_calibration", ("flat",), 10140, None, False),
    ("flat_validation", ("flat",), 10270, "red_flat", False),
    ("gauss100", ("gauss", 100.0), 10120, "gauss", True),
    ("exp100", ("exp", 100.0), 10130, "exp", False),
    ("mixture", ("mix", 100.0, 300.0, 0.2), 10150, "mix", False),
    ("gauss50", ("gauss", 50.0), 10160, "gauss", False),
    ("gauss200", ("gauss", 200.0), 10170, "gauss", False),
    ("gauss400", ("gauss", 400.0), 10200, "gauss", False),
    ("exp50", ("exp", 50.0), 10180, "exp", False),
    ("exp200", ("exp", 200.0), 10190, "exp", False),
    ("exp400", ("exp", 400.0), 10210, "exp", False),
    ("dc_gauss100", ("dc_gauss", 100.0), 10220, "gauss", True),  # A6: "as G 100"
    ("dc_exp100", ("dc_exp", 100.0), 10230, "exp", False),
)
PLANTED = _rows(_PLAN_SPEC)


def reserve_plan():
    """A6: the repair re-run -- EVERY row, mapped truth by truth to 10600-10703 in the table's order."""
    return _rows(tuple((n, t, 10600 + 8 * i, e, w) for i, (n, t, _, e, w) in enumerate(_PLAN_SPEC)))


# ---- the artifact ------------------------------------------------------------------------------
def pinned_artifact():
    """(sha256, n_nodes, n_edges) of the pin file's latest stage -- never a constant typed here."""
    pin = json.loads(GRAPH_PIN.read_text(encoding="utf-8"))
    st = pin["stages"][pin["latest"]]
    return st["artifact_sha256"], int(st["n_nodes"]), int(st["n_edges"])


def _verified(path, pin):
    path = Path(path)
    sha, n_nodes, n_edges = pin or pinned_artifact()
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise ArtifactRefused(f"artifact sha256 {got} != pin {sha}")
    z = np.load(path, allow_pickle=False)
    if z["xyz_nm"].shape[0] != n_nodes or z["pre_idx"].shape[0] != n_edges:
        raise ArtifactRefused(
            f"artifact holds {z['xyz_nm'].shape[0]} nodes / {z['pre_idx'].shape[0]} edges, pin {n_nodes} / {n_edges}"
        )
    return z, got


@dataclass(frozen=True)
class Population:
    """Positions (um), the artifact's layer band codes, and the degree MARGINALS -- never an edge."""

    xyz_um: np.ndarray
    layer: np.ndarray
    out_deg: np.ndarray
    in_deg: np.ndarray


def xyz_um(xyz_nm):
    return np.asarray(xyz_nm, dtype=np.float64) / 1000.0  # the units trap (2.11 (b))


def load_population(path, pin=None):
    """A8's whitelist: sha, counts, positions, layer bands, per-unit out/in-degree SEQUENCES."""
    z, _ = _verified(path, pin)
    n = z["xyz_nm"].shape[0]
    return Population(
        xyz_um=xyz_um(z["xyz_nm"]),
        layer=np.asarray(z["layer"], dtype=np.int64),
        out_deg=np.bincount(z["pre_idx"], minlength=n).astype(np.int64),
        in_deg=np.bincount(z["post_idx"], minlength=n).astype(np.int64),
    )


def _fingerprint(pop):
    h = hashlib.sha256()
    for a in (pop.xyz_um, pop.layer, pop.out_deg, pop.in_deg):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ---- geometry: bins, region blocks, the z-interior, check (j) ----------------------------------
# distinct-from: growth/h01_ingest.py -- population() builds the kernel population by band; nothing in
#   src/ partitions somata by polar angle (no arctan2 in growth/), and placement.layer_of classifies the
#   RADIAL band. The circle data itself is REUSED from growth/cortical_prior.LAYER_CIRCLES, never retyped.
# distinct-from: connectome.py -- a schematic inter-region projection graph ("white-matter WIRING"
#   means long-range tracts there); this is a held-out split of H01 somata, no graph and no tract.
def region_centre():
    """The white-matter circle's centre (um), LAYER_CIRCLES[0] (amendment 2026-10-04)."""
    from snn_data_lake.growth.cortical_prior import LAYER_CIRCLES

    (cx, cy), _ = LAYER_CIRCLES["value"][0]
    return float(cx), float(cy)


def region_blocks(xyz):
    """Four blocks by the quartiles of each soma's polar angle about the white-matter circle's centre
    (the TANGENTIAL direction, so each block spans the laminar axis). DATED PRE-DATA AMENDMENT
    2026-10-04, lead ruling 22: replaces 2.4's x quartiles, which check (j) refused on H01's positions."""
    cx, cy = region_centre()
    theta = np.arctan2(xyz[:, 1] - cy, xyz[:, 0] - cx)
    q = np.quantile(theta, [0.25, 0.5, 0.75])
    return np.searchsorted(q, theta, side="right").astype(np.int64)


def interior_mask(xyz):
    """A8: z in [z_min + Z_TRIM t, z_max - Z_TRIM t], t the z thickness."""
    z = xyz[:, 2]
    lo, hi = float(z.min()), float(z.max())
    t = hi - lo
    return (z >= lo + Z_TRIM * t) & (z <= hi - Z_TRIM * t)


def block_layer_table(blocks, layer):
    """2.11 (j): unit counts per region block per band; a block with an empty band is refused."""
    table = np.zeros((N_BLOCKS, N_BANDS), dtype=np.int64)
    np.add.at(table, (blocks, layer), 1)
    return table


@dataclass(frozen=True)
class Geometry:
    nb: int
    d_c: np.ndarray
    dmax: float  # the bounding-box diagonal: sizes the bins
    dpair: float  # the largest pair distance: 2.3's DEGENERATE bound
    blocks: np.ndarray
    interior: np.ndarray
    table: np.ndarray
    region_valid: bool
    cache: dict = field(default_factory=dict, compare=False)

    @classmethod
    def of(cls, pop):
        xyz = pop.xyz_um
        ext = xyz.max(0) - xyz.min(0)
        dmax = float(np.sqrt((ext**2).sum()))
        if dmax > MAX_EXTENT_UM:
            raise ArtifactRefused(f"units: extent diagonal {dmax:.4g} um > {MAX_EXTENT_UM} um (positions in nm?)")
        nb = int(math.ceil(dmax / BIN_UM)) + 1
        dpair = max(float(np.sqrt(((xyz[s:e, None, :] - xyz[None, :, :]) ** 2).sum(-1)).max()) for s, e in _chunks(len(xyz)))
        blocks = region_blocks(xyz)
        table = block_layer_table(blocks, pop.layer)
        return cls(
            nb=nb,
            d_c=(np.arange(nb) + 0.5) * BIN_UM,
            dmax=dmax,
            dpair=dpair,
            blocks=blocks,
            interior=interior_mask(xyz),
            table=table,
            region_valid=bool((table > 0).all()),
        )


# ---- seeds: one recording seed, three independent streams (A9) ---------------------------------
def seed_streams(seed):
    """stream 0: the unit fold partition; stream 1: the planted edge draw; stream 2: Part B2's pairs."""
    return tuple(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))


def unit_folds(rng, n):
    return rng.permutation(n) % K_FOLDS


# ---- planted truths ----------------------------------------------------------------------------
def kernel_shape(truth, d):
    kind = truth[0].removeprefix("dc_")
    if kind == "flat":
        return np.ones_like(d)
    if kind == "gauss":
        return np.exp(-(d**2) / (2.0 * truth[1] ** 2))
    if kind == "exp":
        return np.exp(-d / truth[1])
    if kind == "mix":
        return np.exp(-(d**2) / (2.0 * truth[1] ** 2)) + truth[3] * np.exp(-d / truth[2])
    raise ValueError(f"unknown truth {truth!r}")


def _is_dc(truth):
    return truth[0].startswith("dc_")


def _chunks(n):
    for s in range(0, n, CHUNK):
        yield s, min(s + CHUNK, n)


def _distances(xyz, s, e):
    d = np.sqrt(((xyz[s:e, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    mask = np.ones(d.shape, dtype=bool)
    rows = np.arange(s, e)
    mask[rows - s, rows] = False
    return d, mask, rows


def _n_all(pop, geo):
    """Positions only: ordered-pair counts per distance bin (cached per geometry)."""
    if "n_all" not in geo.cache:
        n = np.zeros(geo.nb, dtype=np.int64)
        for s, e in _chunks(len(pop.xyz_um)):
            d, mask, _ = _distances(pop.xyz_um, s, e)
            n += np.bincount(np.minimum((d[mask] / BIN_UM).astype(np.int64), geo.nb - 1), minlength=geo.nb)
        geo.cache["n_all"] = n
    return geo.cache["n_all"]


_LOGW_LO, _LOGW_HI, _LOGW_STEP = -40.0, 6.0, 1e-3
EXPECT_RTOL = 1e-9  # A10: the planted expectation is exact to this, not the bucket approximation


def _weight_buckets(pop, geo, truth):
    """Buckets (count, sum of w) with w the planted pair weight before the scale c."""
    if not _is_dc(truth):
        n = _n_all(pop, geo).astype(np.float64)
        return n, n * kernel_shape(truth, geo.d_c)
    nbk = int(round((_LOGW_HI - _LOGW_LO) / _LOGW_STEP)) + 2
    cnt = np.zeros(nbk)
    sw = np.zeros(nbk)
    for s, e in _chunks(len(pop.xyz_um)):
        d, mask, rows = _distances(pop.xyz_um, s, e)
        w = (pop.out_deg[rows][:, None] * pop.in_deg[None, :]) * kernel_shape(truth, d)
        w = w[mask]
        w = w[w > 0]
        idx = np.clip(((np.log10(w) - _LOGW_LO) / _LOGW_STEP).astype(np.int64) + 1, 0, nbk - 1)
        cnt += np.bincount(idx, minlength=nbk)
        sw += np.bincount(idx, weights=w, minlength=nbk)
    return cnt, sw


def solve_scale(pop, geo, truth, n_edges):
    """c such that sum over pairs of min(1, c w) = n_edges (A10: AFTER clipping). Distance-only truths
    solve on the bin histogram (exact at bin centres); degree-corrected truths on 1e-3-dex weight
    buckets (NOT IN THE TEXT: the bucket width; the realized edge count is recorded beside it)."""
    key = ("scale", truth, n_edges)
    if key in geo.cache:
        return geo.cache[key]
    cnt, sw = _weight_buckets(pop, geo, truth)
    pos = sw > 0
    if cnt[pos].sum() < n_edges:
        raise ValueError(f"truth {truth!r}: only {cnt[pos].sum():.0f} pairs carry weight, < {n_edges} edges")

    def expected(c):
        return float(np.minimum(cnt[pos], c * sw[pos]).sum())

    lo, hi = 0.0, 1.0
    while expected(hi) < n_edges:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if expected(mid) < n_edges else (lo, mid)
    c = _refine_exact(pop, truth, hi, n_edges)  # the buckets only start it: they clip per bucket, not per pair
    geo.cache[key] = (c, *_exact_pass(pop, truth, c)[::2])
    return geo.cache[key]


def _exact_pass(pop, truth, c):
    """The actual pairwise clipped expectation at scale c: (E = sum min(1, c w), slope dE/dc, clipped share of E)."""
    e_sum = slope = n_clip = 0.0
    for s, e in _chunks(len(pop.xyz_um)):
        d, mask, rows = _distances(pop.xyz_um, s, e)
        w = kernel_shape(truth, d)
        if _is_dc(truth):
            w = w * (pop.out_deg[rows][:, None] * pop.in_deg[None, :])
        w = w[mask]
        raw = c * w
        e_sum += float(np.minimum(raw, 1.0).sum())
        slope += float(w[raw < 1.0].sum())
        n_clip += float((raw >= 1.0).sum())
    return e_sum, slope, n_clip / e_sum


def _refine_exact(pop, truth, c, n_edges):
    """Newton on the concave piecewise-linear E(c): monotone from below, finite."""
    for _ in range(30):
        e_sum, slope, _ = _exact_pass(pop, truth, c)
        if abs(e_sum - n_edges) <= EXPECT_RTOL * n_edges:
            return c
        c += (n_edges - e_sum) / slope
    raise ValueError(f"truth {truth!r}: the exact expected edge count did not reach {n_edges}")


# ---- the histograms: one pass over all ordered pairs -------------------------------------------
class Hist(NamedTuple):
    n: np.ndarray  # [g_i, g_j, bin], g = fold * N_BLOCKS + block
    k: np.ndarray
    n_int: np.ndarray  # [fold_i, fold_j, bin], both units interior
    k_int: np.ndarray
    planted: dict


def pair_histograms(pop, geo, fold, truth=None, n_edges=None, rng=None, edges=None):
    """Pair counts n (and connected counts k) by distance bin and group. With `truth`, edges are
    DRAWN from stream `rng` chunk by chunk (the draw never reads the fold labels, so A9's streams are
    independent); with `edges` = (pre, post), k is counted from the given edges."""
    xyz, nb, G = pop.xyz_um, geo.nb, N_GROUPS
    grp = fold * N_BLOCKS + geo.blocks
    inner = geo.interior
    n = np.zeros(nb * G * G, dtype=np.int64)
    k = np.zeros_like(n)
    ni = np.zeros(nb * K_FOLDS * K_FOLDS, dtype=np.int64)
    ki = np.zeros_like(ni)
    planted = {}
    if truth is not None:
        c, expected, share = solve_scale(pop, geo, truth, n_edges)
        planted = {"truth": list(truth), "scale": c, "expected_edges": expected, "clipped_share_of_edges": share,
                   "clipped_pairs": 0, "expected_edges_drawn": 0.0}
    for s, e in _chunks(len(xyz)):
        d, mask, rows = _distances(xyz, s, e)
        b = np.minimum((d / BIN_UM).astype(np.int64), nb - 1)
        code = b + nb * (grp[rows][:, None] * G + grp[None, :])
        both_in = mask & inner[rows][:, None] & inner[None, :]
        code_i = b + nb * (fold[rows][:, None] * K_FOLDS + fold[None, :])
        n += np.bincount(code[mask], minlength=n.size)
        ni += np.bincount(code_i[both_in], minlength=ni.size)
        if truth is not None:
            w = kernel_shape(truth, d)
            if _is_dc(truth):
                w = w * (pop.out_deg[rows][:, None] * pop.in_deg[None, :])
            raw = c * w
            planted["clipped_pairs"] += int((raw[mask] > 1.0).sum())
            planted["expected_edges_drawn"] += float(np.minimum(raw[mask], 1.0).sum())  # independent of the solver
            hit = mask & (rng.random(d.shape) < np.minimum(raw, 1.0))
            k += np.bincount(code[hit], minlength=k.size)
            ki += np.bincount(code_i[hit & both_in], minlength=ki.size)
    if edges is not None:
        pre, post = (np.asarray(a, dtype=np.int64) for a in edges)
        de = np.sqrt(((xyz[pre] - xyz[post]) ** 2).sum(-1))
        be = np.minimum((de / BIN_UM).astype(np.int64), nb - 1)
        k += np.bincount(be + nb * (grp[pre] * G + grp[post]), minlength=k.size)
        ein = inner[pre] & inner[post]
        ki += np.bincount((be + nb * (fold[pre] * K_FOLDS + fold[post]))[ein], minlength=ki.size)
    if truth is not None:
        planted["edges"] = int(k.sum())
        planted["clipped_fraction"] = planted["clipped_pairs"] / float(len(xyz) * (len(xyz) - 1))
    return Hist(
        n.reshape(G, G, nb), k.reshape(G, G, nb), ni.reshape(K_FOLDS, K_FOLDS, nb), ki.reshape(K_FOLDS, K_FOLDS, nb), planted
    )


def _hist_sha(h):
    m = hashlib.sha256()
    for a in (h.n, h.k, h.n_int, h.k_int):
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


# ---- the split (2.4, 2.11 (d)) -----------------------------------------------------------------
def check_split(train, labels, f):
    """Train must touch no group of the held-out label f."""
    assert not np.any(train & (labels == f)), f"train touches held-out {f}"


def split(n, k, labels, f):
    """train = both units outside f; test = every pair touching f. Returns per-bin ntr, ktr, nte, kte."""
    train = labels != f
    check_split(train, labels, f)
    ix = np.ix_(train, train)
    ntr, ktr = n[ix].sum((0, 1)), k[ix].sum((0, 1))
    nte, kte = n.sum((0, 1)) - ntr, k.sum((0, 1)) - ktr
    return ntr, ktr, nte, kte


def _exp(x):
    """A width from its log; a runaway (degenerate) width reads inf rather than raising."""
    with np.errstate(over="ignore"):
        return float(np.exp(x))


# ---- the models and the fit (2.3) --------------------------------------------------------------
@np.errstate(over="ignore", under="ignore")
def model_p(model, th, d):
    if model == "flat":
        return np.full_like(d, np.exp(th[0]))
    if model == "gauss":
        return np.exp(th[0] - d**2 / (2.0 * np.exp(2.0 * th[1])))
    if model == "exp":
        return np.exp(th[0] - d / np.exp(th[1]))
    if model == "mix":
        return np.exp(th[0] - d**2 / (2.0 * np.exp(2.0 * th[1]))) + np.exp(th[2] - d / np.exp(th[3]))
    raise ValueError(model)


@np.errstate(divide="ignore", invalid="ignore")
def loglik(p, n, k):
    p = np.clip(p, P_LO, P_HI)
    return float((k * np.log(p) + (n - k) * np.log1p(-p)).sum())


def nelder_mead(f, x0, xatol=NM_TOL, fatol=NM_TOL, maxiter=NM_MAXITER):
    """Nelder-Mead with scipy's default coefficients (rho 1, chi 2, psi 0.5, sigma 0.5), its initial
    simplex (5% per coordinate, 0.00025 for a zero) and its stopping rule."""
    x0 = np.asarray(x0, dtype=np.float64)
    n = len(x0)
    sim = np.empty((n + 1, n))
    sim[0] = x0
    for j in range(n):
        y = x0.copy()
        y[j] = 1.05 * y[j] if y[j] != 0 else 0.00025
        sim[j + 1] = y
    fsim = np.array([f(x) for x in sim])
    it = 0
    while it < maxiter:
        order = np.argsort(fsim, kind="stable")
        sim, fsim = sim[order], fsim[order]
        if np.max(np.abs(sim[1:] - sim[0])) <= xatol and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol:
            break
        xbar = sim[:-1].mean(0)
        xr = 2.0 * xbar - sim[-1]
        fxr = f(xr)
        shrink = False
        if fxr < fsim[0]:
            xe = 3.0 * xbar - 2.0 * sim[-1]
            fxe = f(xe)
            sim[-1], fsim[-1] = (xe, fxe) if fxe < fxr else (xr, fxr)
        elif fxr < fsim[-2]:
            sim[-1], fsim[-1] = xr, fxr
        elif fxr < fsim[-1]:
            xc = 1.5 * xbar - 0.5 * sim[-1]
            fxc = f(xc)
            if fxc <= fxr:
                sim[-1], fsim[-1] = xc, fxc
            else:
                shrink = True
        else:
            xcc = 0.5 * xbar + 0.5 * sim[-1]
            fxcc = f(xcc)
            if fxcc < fsim[-1]:
                sim[-1], fsim[-1] = xcc, fxcc
            else:
                shrink = True
        if shrink:
            for j in range(1, n + 1):
                sim[j] = sim[0] + 0.5 * (sim[j] - sim[0])
                fsim[j] = f(sim[j])
        it += 1
    best = int(np.argmin(fsim))
    return sim[best], float(fsim[best]), it


def fit_model(model, d, n, k, init=None):
    nz = n > 0
    d, n, k = d[nz], n[nz], k[nz]
    if model == "flat":
        return np.array([math.log(max(k.sum(), 1e-300) / n.sum())])

    def obj(th):
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            v = -loglik(model_p(model, th, d), n, k)
        return v if math.isfinite(v) else math.inf

    if init is None:
        best = None
        for w in WIDTH_GRID:  # the width grid with the amplitude profiled under the Poisson approximation
            g = model_p(model, np.array([0.0, math.log(w)]), d)
            a = k.sum() / max(float((n * g).sum()), 1e-300)
            th = np.array([math.log(max(a, 1e-300)), math.log(w)])
            v = obj(th)
            if best is None or v < best[0]:
                best = (v, th)
        init = best[1]
    return nelder_mead(obj, init)[0]


def fit_all(d, n, k):
    th = {m: fit_model(m, d, n, k) for m in ("flat", "gauss", "exp")}
    th["mix"] = fit_model("mix", d, n, k, init=np.r_[th["gauss"], th["exp"][0], th["exp"][1]])
    return th


def _scored(d, n, k, labels, n_labels):
    """Fit on train, score on test, per held-out label. Returns per-label LL dicts, params and test edges."""
    out = []
    for f in range(n_labels):
        ntr, ktr, nte, kte = split(n, k, labels, f)
        th = fit_all(d, ntr, ktr)
        ll = {m: loglik(model_p(m, th[m], d), nte, kte) for m in MODELS}
        out.append((ll, {m: [float(x) for x in th[m]] for m in MODELS}, int(kte.sum())))
    return out


def _diffs(ll):
    return {f"{a}-{b}": ll[a] - ll[b] for a, b in CONTRASTS}


def score_unit(hist, geo):
    """One recording unit (one seed): neuron-fold, z-interior and region-block held-out scores."""
    d = geo.d_c
    folds = _scored(d, hist.n, hist.k, FOLD_OF, K_FOLDS)
    inner = _scored(d, hist.n_int, hist.k_int, np.arange(K_FOLDS), K_FOLDS)
    region = _scored(d, hist.n, hist.k, BLOCK_OF, N_BLOCKS)
    neuron_ll = {m: math.fsum(r[0][m] for r in folds) for m in MODELS}
    inner_ll = {m: math.fsum(r[0][m] for r in inner) for m in MODELS}
    n_all, k_all = hist.n.sum((0, 1)), hist.k.sum((0, 1))
    th_all = {m: fit_model(m, d, n_all, k_all) for m in ("gauss", "exp")}
    width_all = {m: _exp(th_all[m][1]) for m in th_all}
    return {
        "neuron": _diffs(neuron_ll),
        "interior": _diffs(inner_ll),
        "region": [_diffs(r[0]) for r in region],
        "ll": {"neuron": neuron_ll, "interior": inner_ll},
        "edge_instances": sum(r[2] for r in folds),
        "width_all": width_all,
        "degenerate": {m: width_all[m] > geo.dpair for m in width_all},
        "params": {"folds": [r[1] for r in folds], "all": {m: [float(x) for x in th_all[m]] for m in th_all}},
        "hist_sha256": _hist_sha(hist),
    }


# ---- bands and states (A2, A4, A5) -------------------------------------------------------------
def calibrate_bands(units):
    """A5: B = the largest |d| over the CALIBRATION seeds, per contrast and per scoring."""
    return {
        "neuron": {c: max(abs(u["neuron"][c]) for u in units) for c in CONTRAST_NAMES},
        "interior": {c: max(abs(u["interior"][c]) for u in units) for c in CONTRAST_NAMES},
        "region": [{c: max(abs(u["region"][r][c]) for u in units) for c in CONTRAST_NAMES} for r in range(N_BLOCKS)],
    }


def bands_sha256(bands):
    return hashlib.sha256(json.dumps(bands, sort_keys=True).encode("ascii")).hexdigest()


def seed_state(diffs, band):
    """+1 MET, -1 reverse MET, 0 unresolved; degenerate is NOT MET (ruling 9); mean must exceed band."""
    if len(diffs) != len(CONFIRMATORY_SEEDS):
        raise ValueError(f"the statistic is over {len(CONFIRMATORY_SEEDS)} seeds, got {len(diffs)}")
    mean = math.fsum(diffs) / len(diffs)
    up = contrast("up", diffs)
    if not up.degenerate and up.met and mean > band:
        return 1
    down = contrast("down", [-x for x in diffs])
    if not down.degenerate and down.met and -mean > band:
        return -1
    return 0


def value_state(v, band):
    return 1 if v > band else -1 if v < -band else 0


def _opposite(x, y):
    return x != 0 and y != 0 and x != y


def read_t3(neuron, interior, region, bands, planted_ok=True, region_valid=True):
    """A2-A3, a pure function. `neuron`/`interior`: contrast -> 8 per-seed values; `region`: per block,
    contrast -> one value; `bands` as calibrate_bands returns."""
    sn = {c: seed_state(neuron[c], bands["neuron"][c]) for c in CONTRAST_NAMES}
    sz = {c: seed_state(interior[c], bands["interior"][c]) for c in CONTRAST_NAMES}
    sb = [{c: value_state(region[r][c], bands["region"][r][c]) for c in CONTRAST_NAMES} for r in range(N_BLOCKS)]
    sr = {
        c: value_state(
            math.fsum(region[r][c] for r in range(N_BLOCKS)), math.fsum(bands["region"][r][c] for r in range(N_BLOCKS))
        )
        for c in CONTRAST_NAMES
    }
    cands = [k for k in KERNELS if sn[f"{k}-flat"] == 1]
    eg = "exp-gauss"
    a = sn[eg] if sn[eg] != 0 and sn[eg] == sr[eg] else 0
    singles = [k for k in ("gauss", "exp") if k in cands]
    if not cands:
        K = None
    elif len(singles) == 2:
        K = "exp" if a == 1 else "gauss"  # a = -1 -> G; a = 0 -> G by the registered tie order
    elif singles:
        K = singles[0]
    else:
        K = "mix"
    upgraded_from = None
    if K in ("gauss", "exp") and sn[f"mix-{K}"] == 1 and sr[f"mix-{K}"] == 1:
        upgraded_from, K = K, "mix"

    def verdict():
        if not planted_ok:
            return "REFUSED"
        if not region_valid:
            return "NOT_READ_REGION_INVALID"
        if K is None:
            return "RED_FLAT"
        kf = f"{K}-flat"
        if _opposite(sn[kf], sz[kf]) or _opposite(sn[eg], sz[eg]):
            return "NOT_READ_BOUNDARY"
        if _opposite(sn[eg], sr[eg]):
            return "NOT_READ_DISAGREE"
        if any(sb[r][kf] != 1 for r in range(N_BLOCKS)):
            return "RED_FLAT_REGION"
        if K == "exp" or upgraded_from == "exp" or ("gauss" not in cands and "exp" in cands):
            return "RED_EXP"
        if K == "gauss":
            # a = +1 with K = G (E beats G yet fails to beat flat) is not a row of A3; it reads "not refused".
            return "PASS_FAVOURS_GAUSSIAN" if a == -1 else "PASS_GAUSSIAN_NOT_REFUSED"
        return "PASS_MIXTURE"

    detail = {}
    for name, vals in (("neuron", neuron), ("interior", interior)):
        detail[name] = {}
        for c in CONTRAST_NAMES:
            ct = contrast(c, vals[c])
            detail[name][c] = {
                "mean": math.fsum(vals[c]) / len(vals[c]),
                "t": ct.t,
                "fpr": ct.fpr,  # an FPR diagnostic at the fixed bar, never a p-value (A4)
                "degenerate": ct.degenerate,
            }
    return {
        "verdict": verdict(),
        "K": K,
        "C": cands,
        "a": a,
        "upgraded_from": upgraded_from,
        "states": {"neuron": sn, "interior": sz, "region_blocks": sb, "region_aggregate": sr},
        "contrasts": detail,
        "t_bar": T_BAR,
    }


# ---- the planted stage (A5, A6) ----------------------------------------------------------------
def planted_units(pop, row, n_edges, geo=None, log=None):
    geo = geo or Geometry.of(pop)
    units = []
    for seed in row.seeds:
        t0 = time.perf_counter()
        rng_fold, rng_draw, _ = seed_streams(seed)
        fold = unit_folds(rng_fold, len(pop.xyz_um))
        hist = pair_histograms(pop, geo, fold, truth=row.truth, n_edges=n_edges, rng=rng_draw)
        u = score_unit(hist, geo)
        u["seed"] = seed
        u["planted"] = hist.planted
        u["seconds"] = round(time.perf_counter() - t0, 2)
        units.append(u)
        if log:
            log(f"  {row.name} seed {seed}: edges {hist.planted['edges']} widths {u['width_all']} ({u['seconds']} s)")
    return units


def width_recovered(unit, truth):
    kind = truth[0].removeprefix("dc_")
    fitted = unit["width_all"]["exp" if kind == "exp" else "gauss"]
    return abs(fitted / truth[1] - 1.0) <= RECOVERY_TOL


def _gather(units):
    neuron = {c: [u["neuron"][c] for u in units] for c in CONTRAST_NAMES}
    interior = {c: [u["interior"][c] for u in units] for c in CONTRAST_NAMES}
    # A6: a planted truth's region value is the 8-seed MEAN per block
    region = [{c: math.fsum(u["region"][r][c] for u in units) / len(units) for c in CONTRAST_NAMES} for r in range(N_BLOCKS)]
    return neuron, interior, region


def _holds(row, reading, units):
    ok = True
    if row.expect == "red_flat":
        st = reading["states"]
        plus = [st["neuron"], st["interior"], *st["region_blocks"]]
        ok = reading["verdict"] == "RED_FLAT" and not any(s[f"{k}-flat"] == 1 for s in plus for k in KERNELS)
    elif row.expect == "gauss":
        ok = reading["verdict"] == "PASS_FAVOURS_GAUSSIAN" and reading["K"] == "gauss"
    elif row.expect == "exp":
        ok = reading["verdict"] == "RED_EXP" and reading["K"] == "exp"
    elif row.expect == "mix":
        ok = reading["K"] == "mix"
    if row.width_check:
        ok = ok and all(width_recovered(u, row.truth) for u in units)
    return bool(ok)


def run_planted_stage(pop, plan=PLANTED, n_edges=N_EDGES, log=None):
    """A5-A6: calibrate the bands on the first row, LOCK them, then read every other row."""
    geo = Geometry.of(pop)
    rec = {
        "population_sha256": _fingerprint(pop),
        "n_edges_target": n_edges,
        "region_table": geo.table.tolist(),
        "region_valid": geo.region_valid,
        "plan": [r._asdict() for r in plan],
        "rows": {},
    }
    if not geo.region_valid:  # A7: positions only, before any planted row or edge
        rec["all_rows_hold"] = False
        rec["stopped"] = "NOT_READ_REGION_INVALID: a region block lacks a band (2.11 (j))"
        return rec
    cal = plan[0]
    assert cal.expect is None and cal.truth == ("flat",), "the first plan row is the flat calibration set"
    units = planted_units(pop, cal, n_edges, geo, log)
    bands = calibrate_bands(units)
    rec["bands"] = bands
    rec["bands_sha256"] = bands_sha256(bands)  # LOCKED before any other row is drawn
    rec["rows"][cal.name] = {"units": units, "reading": None, "holds": None}
    for row in plan[1:]:
        units = planted_units(pop, row, n_edges, geo, log)
        reading = read_t3(*_gather(units), bands)
        rec["rows"][row.name] = {"units": units, "reading": reading, "holds": _holds(row, reading, units)}
        if log:
            log(f"{row.name}: {reading['verdict']} K={reading['K']} holds={rec['rows'][row.name]['holds']}")
    rec["all_rows_hold"] = all(v["holds"] for k, v in rec["rows"].items() if k != cal.name)
    return rec


# ---- H01 (the only edge reader) ----------------------------------------------------------------
def _plan_rows(plan):
    """A plan in comparable form (tuples and JSON lists read alike)."""
    out = []
    for r in plan:
        d = r._asdict() if isinstance(r, PlanRow) else r
        out.append([d["name"], list(d["truth"]), [int(x) for x in d["seeds"]], d["expect"], bool(d["width_check"])])
    return out


def _is_production(z, sha):
    """H01 is recognised by CONTENT (node and edge counts) as well as by sha: a re-saved copy of the edges
    under a caller-supplied pin is still H01 (G7 H1)."""
    psha, n_nodes, n_edges = pinned_artifact()
    return sha == psha or (z["xyz_nm"].shape[0], z["pre_idx"].shape[0]) == (n_nodes, n_edges)


def _authorized_plan(prod, planted_record, plan, n_edges):
    """Which planted plan may authorize an edge read. On the PRODUCTION artifact (H01, by `_is_production`)
    only the registered plan -- A6's table or its full reserve re-run -- at H01's edge count, whatever
    the caller passes: a reduced (rehearsal) plan can never authorize production. On any other
    artifact (a synthetic fixture) the caller names the plan, which must still be complete."""
    if prod:
        rec = _plan_rows((planted_record or {}).get("plan", []))
        for registered in (PLANTED, reserve_plan()):
            if rec == _plan_rows(registered):
                return registered, N_EDGES
        raise ExposureRefused("the planted record's plan is not the registered plan (A6) or its reserve")
    if plan is None or n_edges is None:
        raise ExposureRefused("a fixture artifact needs its plan and edge count named explicitly")
    return plan, n_edges


def validate_planted(planted_record, pop, plan, n_edges):
    """THE ONE validator every edge-reading entry point calls (score_h01, final_width_record,
    verify_width_record). It never trusts a stored verdict: it checks the record against `plan` row by
    row and seed by seed, recomputes the calibration bands from the calibration units, and re-reads
    every other row through read_t3 and its expectation. Returns the recomputed bands."""
    if not planted_record:
        raise ExposureRefused("no planted record: H01's edges are not read (A8)")
    if len(plan) < 2 or plan[0].expect is not None or tuple(plan[0].truth) != ("flat",):
        raise ExposureRefused("a plan needs its flat calibration row and at least one row that can fail")
    if _plan_rows(planted_record.get("plan", [])) != _plan_rows(plan):
        raise ExposureRefused("the planted record's plan differs from the authorized plan")
    if planted_record.get("n_edges_target") != n_edges:
        raise ExposureRefused(f"planted density {planted_record.get('n_edges_target')} != {n_edges} edges")
    if planted_record.get("population_sha256") != _fingerprint(pop):
        raise ExposureRefused("the planted record was drawn on a different population")
    rows = planted_record.get("rows") or {}
    if set(rows) != {r.name for r in plan}:
        raise ExposureRefused(f"planted rows {sorted(rows)} != the plan's {sorted(r.name for r in plan)}")
    for r in plan:
        if [int(u["seed"]) for u in rows[r.name]["units"]] != list(r.seeds):
            raise ExposureRefused(f"row {r.name}: its units are not the plan's seeds")
    bands = calibrate_bands(rows[plan[0].name]["units"])
    if bands != planted_record.get("bands") or bands_sha256(bands) != planted_record.get("bands_sha256"):
        raise ExposureRefused("the planted bands do not recompute from the calibration units")
    for r in plan[1:]:
        units = rows[r.name]["units"]
        if not _holds(r, read_t3(*_gather(units), bands), units):
            raise ExposureRefused(f"planted row {r.name} does not hold on recomputation")
    return bands


def _load_gated(path, planted_record, pin, plan=None, n_edges=None, seeds=None):
    z, sha = _verified(path, pin)
    pop = load_population(path, pin)
    prod = _is_production(z, sha)
    if seeds is not None:
        _check_seeds(prod, seeds)
    plan, n_edges = _authorized_plan(prod, planted_record, plan, n_edges)
    bands = validate_planted(planted_record, pop, plan, n_edges)
    return pop, (np.asarray(z["pre_idx"]), np.asarray(z["post_idx"])), sha, bands, plan


# 2.5's resolution flag: the matching planted discrimination (row, sign) for each contrast
_REFERENCE = {
    "gauss-flat": ("gauss100", 1),
    "exp-flat": ("exp100", 1),
    "mix-flat": ("mixture", 1),
    "exp-gauss": ("exp100", 1),
    "mix-gauss": ("mixture", 1),
    "mix-exp": ("mixture", 1),
}


def _resolution_flags(reading, units, planted_record):
    flags = {}
    per_edge = math.fsum(u["edge_instances"] for u in units) / len(units)
    for c in CONTRAST_NAMES:
        if reading["states"]["neuron"][c] == 0:
            continue
        row, sign = _REFERENCE[c]
        if reading["states"]["neuron"][c] == -1 and c == "exp-gauss":
            row = "gauss100"
        ref = planted_record["rows"].get(row)
        if not ref:
            flags[c] = None
            continue
        ref_units = ref["units"]
        ref_edge = math.fsum(u["edge_instances"] for u in ref_units) / len(ref_units)
        ref_eff = abs(math.fsum(u["neuron"][c] for u in ref_units) / len(ref_units)) / ref_edge
        eff = abs(reading["contrasts"]["neuron"][c]["mean"]) / per_edge
        flags[c] = {"nats_per_edge": eff, "planted_nats_per_edge": ref_eff, "flagged": eff < 0.1 * ref_eff}
    return flags


def _check_seeds(prod, seeds):
    """The registered statistic is over EIGHT distinct recording seeds: on the production artifact exactly
    the confirmatory set or its reserve (2.12); a fixture names any eight distinct integers."""
    seeds = tuple(seeds)
    if prod:
        if seeds not in (CONFIRMATORY_SEEDS, CONFIRMATORY_RESERVE):
            raise ExposureRefused("H01's edges are read on the registered confirmatory seeds or their reserve only (2.12)")
    elif len(seeds) != len(CONFIRMATORY_SEEDS) or len(set(seeds)) != len(seeds) or not all(type(x) is int for x in seeds):
        raise ExposureRefused(f"the statistic needs {len(CONFIRMATORY_SEEDS)} distinct integer seeds, got {seeds!r}")


def score_h01(path, planted_record, seeds=CONFIRMATORY_SEEDS, pin=None, plan=None, n_edges=None, log=None):
    """2.4-2.8 on H01's edges, read ONCE, only after validate_planted accepts the planted record. A
    record that STOPPED at check (j) yields A3 row 2's reading, NOT_READ_REGION_INVALID, and no edge
    distance is read."""
    if planted_record and str(planted_record.get("stopped", "")).startswith("NOT_READ_REGION_INVALID"):
        z, sha = _verified(path, pin)
        pop = load_population(path, pin)
        geo = Geometry.of(pop)
        plan, n_edges = _authorized_plan(_is_production(z, sha), planted_record, plan, n_edges)
        if (
            planted_record.get("population_sha256") != _fingerprint(pop)
            or geo.region_valid
            or _plan_rows(planted_record.get("plan", [])) != _plan_rows(plan)
        ):
            raise ExposureRefused("the stopped planted record does not describe this population and plan")
        return {
            "artifact_sha256": sha,
            "seeds": [],
            "units": [],
            "region_table": geo.table.tolist(),
            "reading": {"verdict": "NOT_READ_REGION_INVALID", "K": None, "C": [], "a": 0},
        }
    pop, edges, sha, bands, plan = _load_gated(path, planted_record, pin, plan, n_edges, seeds)
    geo = Geometry.of(pop)
    units = []
    for seed in seeds:
        fold = unit_folds(seed_streams(seed)[0], len(pop.xyz_um))
        u = score_unit(pair_histograms(pop, geo, fold, edges=edges), geo)
        u["seed"] = seed
        units.append(u)
        if log:
            log(f"  H01 seed {seed}: widths {u['width_all']}")
    region0 = units[0]["region"]
    identical = all(u["region"] == region0 for u in units)  # region blocks sum over folds: seed-free
    neuron, interior, _ = _gather(units)
    reading = read_t3(neuron, interior, region0, bands, planted_ok=True, region_valid=geo.region_valid)
    out = {
        "artifact_sha256": sha,
        "seeds": list(seeds),
        "units": units,
        "region_identical_across_seeds": identical,
        "reading": reading,
        "resolution_flags": _resolution_flags(reading, units, planted_record),
    }
    if reading["verdict"].startswith("PASS") or reading["verdict"] == "RED_EXP":
        out["final_width"] = _final_width(pop, geo, edges, reading["K"], sha)
        out["final_width"]["fold_width_range"] = _fold_range(units, reading["K"])
    return out


def _fold_range(units, model):
    idx = (1,) if model in ("gauss", "exp") else (1, 3)
    vals = [[_exp(f[model][i]) for f in u["params"]["folds"]] for u in units for i in idx]
    flat = [v for row in vals for v in row]
    return [min(flat), max(flat)]


def _width_um(model, th):
    if model == "gauss":
        return {"sigma": _exp(th[1])}
    if model == "exp":
        return {"lambda": _exp(th[1])}
    return {"sigma": _exp(th[1]), "lambda": _exp(th[3])}


def _final_width(pop, geo, edges, model, sha):
    fold = np.zeros(len(pop.xyz_um), dtype=np.int64)
    h = pair_histograms(pop, geo, fold, edges=edges)
    n_all, k_all = h.n.sum((0, 1)), h.k.sum((0, 1))
    if model == "mix":
        th = fit_all(geo.d_c, n_all, k_all)["mix"]
    else:
        th = fit_model(model, geo.d_c, n_all, k_all)
    wu = _width_um(model, th)
    return {
        "model": model,
        "params": [float(x) for x in th],
        "width_um": wu,
        "degenerate": any(v > geo.dpair for v in wu.values()),
        "artifact_sha256": sha,
        "n_pairs": int(n_all.sum()),
        "n_edges": int(k_all.sum()),
    }


def final_width_record(path, model, planted_record, pin=None, plan=None, n_edges=None):
    """2.8: the width(s) fitted once on ALL eligible pairs -- an edge read, gated by validate_planted."""
    pop, edges, sha, _, _ = _load_gated(path, planted_record, pin, plan, n_edges)
    return _final_width(pop, Geometry.of(pop), edges, model, sha)


def verify_width_record(record, path, planted_record, pin=None, plan=None, n_edges=None):
    """2.11 (f): a refit on the pinned data reproduces the record -- an edge read, gated by validate_planted."""
    pop, edges, sha, _, _ = _load_gated(path, planted_record, pin, plan, n_edges)
    if record.get("artifact_sha256") != sha:
        raise RecordRefused("the record names a different artifact")
    fresh = _final_width(pop, Geometry.of(pop), edges, record["model"], sha)
    if not np.allclose(fresh["params"], record["params"], rtol=WIDTH_RTOL, atol=0.0):
        raise RecordRefused(f"refit {fresh['params']} != record {record['params']}")
    # the widths consumers read are physical (um): the record's own, and consistent with its params
    wu = record.get("width_um") or {}
    if set(wu) != set(fresh["width_um"]) or any(
        not np.isclose(wu[k], v, rtol=WIDTH_RTOL, atol=0.0) or not np.isclose(wu[k], _width_um(record["model"], record["params"])[k], rtol=WIDTH_RTOL, atol=0.0)
        for k, v in fresh["width_um"].items()
    ):
        raise RecordRefused(f"refit widths {fresh['width_um']} != record widths {wu}")
    return fresh


# ---- CLI and the offline self-check ------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _selfcheck():
    """Offline, synthetic, a few seconds: the reading rows, the optimizer, a planted recovery, a
    degenerate flat width, the units guard. No H01 file, no Brain, no lake."""
    x, fx, _ = nelder_mead(lambda v: float(((v - np.array([1.0, -2.0])) ** 2).sum()), np.array([3.0, 3.0]))
    assert np.allclose(x, [1.0, -2.0], atol=1e-5) and fx < 1e-9, (x, fx)

    jit = [0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 0.98, 1.02]

    def inputs(n_vals, r_vals):
        n = {c: [j * n_vals.get(c, 0.0) for j in jit] for c in CONTRAST_NAMES}
        r = [{c: r_vals.get(c, 0.0) for c in CONTRAST_NAMES} for _ in range(N_BLOCKS)]
        bands = {
            "neuron": dict.fromkeys(CONTRAST_NAMES, 1.0),
            "interior": dict.fromkeys(CONTRAST_NAMES, 1.0),
            "region": [dict.fromkeys(CONTRAST_NAMES, 1.0) for _ in range(N_BLOCKS)],
        }
        return n, n, r, bands

    g_world = ({"gauss-flat": 50.0, "exp-flat": 40.0, "exp-gauss": -10.0}, {"gauss-flat": 20.0, "exp-flat": 15.0, "exp-gauss": -5.0})
    e_world = ({"gauss-flat": 40.0, "exp-flat": 50.0, "exp-gauss": 10.0}, {"gauss-flat": 15.0, "exp-flat": 20.0, "exp-gauss": 5.0})
    assert read_t3(*inputs(*g_world))["verdict"] == "PASS_FAVOURS_GAUSSIAN"
    assert read_t3(*inputs(*e_world))["verdict"] == "RED_EXP"
    assert read_t3(*inputs({}, {}))["verdict"] == "RED_FLAT"
    assert read_t3(*inputs(*g_world), planted_ok=False)["verdict"] == "REFUSED"
    assert seed_state([5.0] * 8, 1.0) == 0, "a degenerate contrast is NOT MET (ruling 9)"

    rng = np.random.default_rng(20261003)
    pop = Population(
        xyz_um=rng.random((240, 3)) * np.array([300.0, 200.0, 40.0]),
        layer=rng.integers(0, N_BANDS, size=240),
        out_deg=np.ones(240, dtype=np.int64),
        in_deg=np.ones(240, dtype=np.int64),
    )
    g = planted_units(pop, PlanRow("g", ("gauss", 25.0), (10004,), "gauss", True), 1200)[0]
    f = planted_units(pop, PlanRow("f", ("flat",), (10005,), None, False), 1200)[0]
    assert width_recovered(g, ("gauss", 25.0)), g["width_all"]
    assert g["neuron"]["gauss-flat"] > 0 and g["neuron"]["gauss-flat"] > g["neuron"]["exp-flat"], g["neuron"]
    nm = Population(xyz_um=pop.xyz_um * 1000.0, layer=pop.layer, out_deg=pop.out_deg, in_deg=pop.in_deg)
    try:
        Geometry.of(nm)
        raise AssertionError("positions in nm were not refused")
    except ArtifactRefused:
        pass
    print(
        f"self-check OK: reading rows PASS/RED_EXP/RED_FLAT/REFUSED reached; degenerate NOT MET; planted sigma 25 "
        f"recovered at {g['width_all']['gauss']:.2f} um (G - flat {g['neuron']['gauss-flat']:+.1f} nats, "
        f"G - E {g['neuron']['gauss-flat'] - g['neuron']['exp-flat']:+.1f}); flat world width "
        f"{f['width_all']['gauss']:.3g} um degenerate={f['degenerate']['gauss']}; nm positions refused"
    )


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("planted", help="A5-A6: the planted stage on the pinned artifact's positions (a CAMPAIGN)")
    p.add_argument("--artifact", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reserve", action="store_true", help="A6's repair re-run on 10600-10703")
    h = sub.add_parser("h01", help="2.4-2.8 on H01's edges (a CAMPAIGN; refuses without a holding planted record)")
    h.add_argument("--artifact", required=True)
    h.add_argument("--planted", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--reserve", action="store_true", help="the confirmatory reserve 10108-10115")
    a = ap.parse_args(argv)
    if a.cmd is None:
        _selfcheck()
        return 0
    t0 = time.perf_counter()
    if a.cmd == "planted":
        pop = load_population(a.artifact)
        rec = run_planted_stage(pop, plan=reserve_plan() if a.reserve else PLANTED, log=print)
    else:
        planted = json.loads(Path(a.planted).read_text(encoding="utf-8"))
        rec = score_h01(a.artifact, planted, seeds=CONFIRMATORY_RESERVE if a.reserve else CONFIRMATORY_SEEDS, log=print)
        print(f"T3 reads: {rec['reading']['verdict']} (K = {rec['reading']['K']})")
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(_jsonable(rec), indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {a.out} in {rec['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
