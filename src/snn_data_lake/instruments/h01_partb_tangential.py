"""G4 Part B (g4-wiring-prior.md 3.1 and the DATED PRE-DATA AMENDMENT 2026-10-07): the TANGENTIAL kernel
with pathway structure, fitted on H01 and read by held-out scoring -- the record `growth/wiring_prior.py` reads.

Frame: r = |(x, y) - the white-matter circle's centre|, theta = atan2; per pair d_tan = sqrt((r_mean dtheta)^2
+ dz^2). A pathway = (pre layer x sign, post layer x sign), 14 classes -> 196 cells. Arms, all scored by
held-out binomial log-likelihood on neuron folds (decide), polar-angle region blocks (agree) and the z-interior:
  F  distance-free twin: one flat amplitude per cell
  G  one global kernel (the mixture, `kind` selects gauss / exp for the form disclosure) x a per-cell amplitude
  P  G with a per-cell log-width OFFSET delta_p, partially pooled: penalised NLL + 0.5 pen sum(delta^2), pen
     chosen on the TRAINING folds only by nested folds (A12-style).
Contrasts: G-F (does the kernel beat its distance-free twin) and P-G (does pooling beat the global width).

ORDER IS ENFORCED (amendment 4, A8): `run_planted` reads positions and layer / sign only; `score_h01` refuses
unless a planted record is handed in whose bands recompute and whose every row recomputes to its expectation.
SIMPLIFICATIONS, DISCLOSED in the amendment's dated addendum: P fits delta_p given the global shape (no joint
refit); the penalty search reuses the outer-training global shape; a cell amplitude carries a pseudo-count of
0.5 so a cell with no training edge keeps a positive probability; the planted draw is edge-level, like T3's.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.h01_partb_tangential        (self-check)
"""

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from snn_data_lake.instruments import h01_distance_kernel as t3
from snn_data_lake.instruments.paired_contrast import contrast
from snn_data_lake.reservoir.cortical_microcircuit import CONN_PROB

N_CLASS = 14  # 7 bands x 2 signs
N_CELL = N_CLASS * N_CLASS
K = t3.K_FOLDS
NB_REG = t3.N_BLOCKS
BIN = t3.BIN_UM
PSEUDO = 0.5
FRAME_VIOLATION_MAX = 0.05  # amendment 1: a session choice
PATHWAY_RECOVERY_TOL = 0.15  # amendment 4 (iii): a session choice
PENALTIES = (1.0, 4.0, 16.0, 64.0)  # nats per unit delta^2: a session choice
DELTAS = np.linspace(-1.5, 1.5, 31)
SEEDS = tuple(range(10350, 10358))  # the confirmatory H01 partitions (amendment 5)
CONTRASTS = ("G-F", "P-G")
KINDS = ("gauss", "exp", "mix")


class ExposureRefused(RuntimeError):
    """An edge read was attempted before the planted record holds (amendment 4)."""


class PlanRow(NamedTuple):
    name: str
    truth: dict
    seeds: tuple
    expect: object  # None (band source) | "red_flat" | "pass" | "red_pool"


def _row(name, truth, first, expect):
    return PlanRow(name, truth, tuple(range(first, first + 8)), expect)


# amendment 4's seed map: calibration 10300-07, validation 10308-15, (i) 10316-31, (iii) 10332-47
PLAN = (
    _row("flat_calibration", {"form": "flat"}, 10300, None),
    _row("flat_validation", {"form": "flat"}, 10308, "red_flat"),
    _row("gauss100", {"form": "gauss", "width": 100.0}, 10316, "red_pool"),
    _row("exp100", {"form": "exp", "width": 100.0}, 10324, "red_pool"),
    _row(
        "pathway_widths",
        {"form": "gauss", "width": 100.0, "pathway": (60.0, 160.0)},
        10332,
        "pooling_gate",
    ),
    _row("pathway_flat", {"form": "gauss", "width": 100.0}, 10340, "red_pool"),
)


# ---- the frame ---------------------------------------------------------------------------------
def frame(xyz):
    cx, cy = t3.region_centre()
    dx, dy = xyz[:, 0] - cx, xyz[:, 1] - cy
    return np.hypot(dx, dy), np.arctan2(dy, dx), xyz[:, 2]


def d_tan(fr, i, j):
    r, th, z = fr
    dth = (th[i] - th[j] + math.pi) % (
        2.0 * math.pi
    ) - math.pi  # wrapped (check (a): the theta-wrap mutant)
    return np.sqrt((0.5 * (r[i] + r[j]) * dth) ** 2 + (z[i] - z[j]) ** 2)


FRAME_WINDOW = 5  # neighbours each side in the r-order: a session choice (amendment 1 names the statistic, not W)


def frame_violation(layer, r):
    """Soma-level, amendment 1: somata ordered by r (pia outward of the white matter), the band index must be
    non-decreasing along it. A soma violates when one of its FRAME_WINDOW r-order neighbours on either side
    disagrees with the order (an earlier neighbour in a deeper band, or a later one in a shallower band).
    Returns (fraction of somata, count): 0 = r is monotone in the band; the bar is on the fraction."""
    b = np.asarray(layer)[np.argsort(-np.asarray(r), kind="stable")]
    bad = np.zeros(len(b), dtype=bool)
    for w in range(1, FRAME_WINDOW + 1):
        bad[w:] |= b[:-w] > b[w:]  # an earlier (further-out) neighbour carries a deeper band
        bad[:-w] |= b[w:] < b[:-w]  # a later neighbour carries a shallower band
    return float(bad.mean()), int(bad.sum())


def unit_class(layer, sign):
    return np.asarray(layer, dtype=np.int64) * 2 + (np.asarray(sign) > 0)


# ---- the population (positions, band, sign, degree marginals -- never an edge) -----------------
@dataclass(frozen=True)
class Pop:
    xyz: np.ndarray
    layer: np.ndarray
    sign: np.ndarray
    cls: np.ndarray
    fr: tuple
    nb: int


def make_pop(xyz, layer, sign):
    xyz = np.asarray(xyz, dtype=np.float64)
    diag = float(np.sqrt(((xyz.max(0) - xyz.min(0)) ** 2).sum()))
    if diag > t3.MAX_EXTENT_UM:  # the unit trap (check (a)): nm fed as um
        raise t3.ArtifactRefused(
            f"units: extent diagonal {diag:.4g} um > {t3.MAX_EXTENT_UM} um (nm?)"
        )
    fr = frame(xyz)
    r, th, z = fr
    arc = float(r.max() * (th.max() - th.min()))
    nb = int(math.ceil(math.hypot(arc, float(z.max() - z.min())) / BIN)) + 2
    return Pop(xyz, np.asarray(layer), np.asarray(sign), unit_class(layer, sign), fr, nb)


def load_pop(path, pin=None):
    z, _ = t3._verified(path, pin)
    return make_pop(t3.xyz_um(z["xyz_nm"]), z["layer"], z["sign"])


def fingerprint(pop):
    h = hashlib.sha256()
    for a in (pop.xyz, pop.layer, pop.sign):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ---- histograms: n (all eligible pairs) and k (edges) by [group, group, cell, bin] -------------
@dataclass
class Hists:
    n_fold: np.ndarray
    k_fold: np.ndarray
    n_block: np.ndarray
    k_block: np.ndarray
    n_int: np.ndarray
    k_int: np.ndarray


def _zeros(g, nb):
    return np.zeros((g, g, N_CELL, nb), dtype=np.int64)


def _add(pop, parts, i, j, which):
    folds, blocks, interior = parts
    b = np.minimum((d_tan(pop.fr, i, j) / BIN).astype(np.int64), pop.nb - 1)
    cell = pop.cls[i] * N_CLASS + pop.cls[j]
    for arr, ga, gb, g, sel in (
        (which[0], folds[i], folds[j], K, None),
        (which[1], blocks[i], blocks[j], NB_REG, None),
        (which[2], folds[i], folds[j], K, interior[i] & interior[j]),
    ):
        key = ((ga * g + gb) * N_CELL + cell) * pop.nb + b
        if sel is not None:
            key = key[sel]
        arr += np.bincount(key, minlength=arr.size).reshape(arr.shape)


def _parts(pop, seed):
    s = t3.seed_streams(seed)
    folds = t3.unit_folds(s[0], len(pop.xyz))
    return (folds, t3.region_blocks(pop.xyz), t3.interior_mask(pop.xyz)), s


def _all_pairs_n(pop):
    """Positions only: ordered pairs per [cell, bin], seed-independent (the planted scale needs it)."""
    n = np.zeros((N_CELL, pop.nb), dtype=np.int64)
    for s, e in t3._chunks(len(pop.xyz)):
        i = np.repeat(np.arange(s, e), len(pop.xyz))
        j = np.tile(np.arange(len(pop.xyz)), e - s)
        keep = i != j
        i, j = i[keep], j[keep]
        b = np.minimum((d_tan(pop.fr, i, j) / BIN).astype(np.int64), pop.nb - 1)
        n += np.bincount(
            (pop.cls[i] * N_CLASS + pop.cls[j]) * pop.nb + b, minlength=n.size
        ).reshape(n.shape)
    return n


def cell_amplitude():
    """Planted pathway amplitudes: CONN_PROB's (target, source) for the modelled layers L2-L6, a small
    constant for L1 / white matter; per cell = (pre class) * 14 + (post class)."""
    amp = np.full(N_CELL, 0.02)
    for a in range(N_CLASS):
        for b in range(N_CLASS):
            la, lb = a // 2, b // 2
            if 1 <= la <= 5 and 1 <= lb <= 5:
                pa = {1: 0, 2: 0, 3: 2, 4: 4, 5: 6}[la] + (0 if a % 2 else 1)
                pb = {1: 0, 2: 0, 3: 2, 4: 4, 5: 6}[lb] + (0 if b % 2 else 1)
                amp[a * N_CLASS + b] = max(CONN_PROB[pb][pa], 1e-3)
    return amp


def planted_prob(truth, d, cell, scale):
    form = truth["form"]
    if form == "flat":
        return np.ones_like(d)
    w = truth["width"] * scale[cell]
    return np.exp(-(d**2) / (2.0 * w**2)) if form == "gauss" else np.exp(-d / w)


def pathway_scales(truth, pop):
    """Per-cell width multiplier: the two most populous cells carry the planted pathway widths."""
    scale = np.ones(N_CELL)
    if truth.get("pathway"):
        n = np.bincount(pop.cls, minlength=N_CLASS)
        pair = np.outer(n, n).astype(float)
        np.fill_diagonal(pair, 0.0)
        pair = pair.ravel()
        for cell, w in zip(np.argsort(-pair)[:2], truth["pathway"]):
            scale[cell] = w / truth["width"]
    return scale


def planted_scale(truth, pop, n_all, n_edges, scale):
    """c with sum n_all * c * amp * shape = n_edges, exact (linear); refuses a clipped world."""
    d = (np.arange(pop.nb) + 0.5) * BIN
    amp = cell_amplitude()
    cells = np.repeat(np.arange(N_CELL), pop.nb)
    shp = planted_prob(truth, np.tile(d, N_CELL), cells, scale).reshape(N_CELL, pop.nb)
    c = n_edges / float((n_all * amp[:, None] * shp).sum())
    if (c * amp[:, None] * shp).max() >= 1.0:
        raise ValueError("planted world clips at 1: the expected-count scale is not linear")
    return c


def build_hists(pop, seed, n_all=None, truth=None, n_edges=None, edges=None):
    """One recording seed: n (positions) and k (planted edges, or H01's `edges` = (pre, post)) hists."""
    parts, streams = _parts(pop, seed)
    n_h = [_zeros(K, pop.nb), _zeros(NB_REG, pop.nb), _zeros(K, pop.nb)]
    k_h = [np.zeros_like(a) for a in n_h]
    if edges is not None:
        _add(pop, parts, edges[0].astype(np.int64), edges[1].astype(np.int64), k_h)
    scale = c = None
    if truth is not None:
        scale = pathway_scales(truth, pop)
        c = planted_scale(truth, pop, n_all, n_edges, scale)
    amp, rng, n = cell_amplitude(), streams[1], len(pop.xyz)
    for s, e in t3._chunks(n):
        i = np.repeat(np.arange(s, e), n)
        j = np.tile(np.arange(n), e - s)
        keep = i != j
        i, j = i[keep], j[keep]
        _add(pop, parts, i, j, n_h)
        if truth is not None:
            cell = pop.cls[i] * N_CLASS + pop.cls[j]
            p = c * amp[cell] * planted_prob(truth, d_tan(pop.fr, i, j), cell, scale)
            hit = rng.random(len(p)) < p
            _add(pop, parts, i[hit], j[hit], k_h)
    return Hists(*(a for pair in zip(n_h, k_h) for a in pair)), parts


# ---- the fits ----------------------------------------------------------------------------------
def shape(d, th, kind="mix"):
    s, lam, rho = (
        math.exp(min(max(float(x), -30.0), 30.0)) for x in th
    )  # a runaway width stays finite
    g = np.exp(-(d**2) / (2.0 * s**2))
    e = np.exp(-d / lam)
    return g if kind == "gauss" else e if kind == "exp" else g + rho * e


def _profile(n, k, g):
    a = (k.sum(1) + PSEUDO) / np.maximum((n * g).sum(1), 1e-300)
    p = np.clip(a[:, None] * g, t3.P_LO, t3.P_HI)
    return a, (k * np.log(p) + (n - k) * np.log1p(-p)).sum(1)


def fit_global(d, n, k, kind="mix"):
    keep = n.sum(1) > 0
    n, k = n[keep], k[keep]

    def nll(th):
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            v = -_profile(n, k, shape(d[None, :], th, kind))[1].sum()
        return v if math.isfinite(v) else math.inf

    grid = t3.WIDTH_GRID[::5]
    best = min(
        (
            (nll(np.array([math.log(s), math.log(lam), math.log(r)])), s, lam, r)
            for s in grid
            for lam in grid
            for r in (0.03, 0.3, 3.0)
        ),
        key=lambda t: t[0],
    )
    x0 = np.array([math.log(best[1]), math.log(best[2]), math.log(best[3])])
    return t3.nelder_mead(nll, x0)[0]


def flat_ll(ntr, ktr, nte, kte):
    a = (ktr.sum(1) + PSEUDO) / np.maximum(ntr.sum(1), 1.0)
    return _ll(np.broadcast_to(a[:, None], nte.shape), nte, kte)


def _ll(p, n, k):
    p = np.clip(p, t3.P_LO, t3.P_HI)
    return float((k * np.log(p) + (n - k) * np.log1p(-p)).sum())


def shaped_p(d, ntr, ktr, th, kind, delta=None):
    """p[cell, bin] for the amplitude profiled on the training counts (delta = per-cell log-width offset)."""
    dd = d[None, :] if delta is None else d[None, :] * np.exp(-delta)[:, None]
    g = shape(dd, th, kind)
    a, _ = _profile(ntr, ktr, g)
    return a[:, None] * g


def fit_delta(d, n, k, th, kind, pen):
    """Per-cell offset by a 1-D grid on the penalised cell NLL, the shape fixed."""
    nll = np.stack([-_profile(n, k, shape(d[None, :] * math.exp(-x), th, kind))[1] for x in DELTAS])
    return DELTAS[np.argmin(nll + 0.5 * pen * DELTAS[:, None] ** 2, axis=0)]


def leak_free(train_folds, test_folds, f, g):
    """The nested split's disjointness (check (b)): no inner-train unit group is the outer-held-out f or the
    inner-held-out g, and the inner test folds contain g and avoid f."""
    return f not in train_folds and g not in train_folds and g in test_folds and f not in test_folds


def inner_split(hist_n, hist_k, f, g):
    """Outer fold f held out; inner fold g held out of the rest. Returns ntr, ktr, nte, kte."""
    groups = range(hist_n.shape[0])  # K unit folds or the four region blocks
    tr = [h for h in groups if h != f]
    ok = [h for h in tr if h != g]
    assert leak_free(ok, tr, f, g), f"inner split leaks held-out folds {f}, {g}"
    ntr, ktr = (a[np.ix_(ok, ok)].sum((0, 1)) for a in (hist_n, hist_k))
    nin, kin = (a[np.ix_(tr, tr)].sum((0, 1)) for a in (hist_n, hist_k))
    return ntr, ktr, nin - ntr, kin - ktr


def choose_penalty(d, hist_n, hist_k, f, th, kind):
    """Nested folds on the TRAINING folds only: the penalty with the best summed inner held-out LL."""
    score = dict.fromkeys(PENALTIES, 0.0)
    for g in (h for h in range(hist_n.shape[0]) if h != f):
        ntr, ktr, nte, kte = inner_split(hist_n, hist_k, f, g)
        for pen in PENALTIES:
            delta = fit_delta(d, ntr, ktr, th, kind, pen)
            score[pen] += _ll(shaped_p(d, ntr, ktr, th, kind, delta), nte, kte)
    return max(PENALTIES, key=lambda p: score[p])


def score_labels(d, n, k, labels_n, kind="mix", forms=False):
    """Held-out per label: LL of F, G, P (and the three forms if `forms`), fit on train, scored on test."""
    out = []
    for f in range(labels_n):
        ntr, ktr, nte, kte = t3.split(n, k, np.arange(n.shape[0]), f)
        th = fit_global(d, ntr, ktr, kind)
        pen = choose_penalty(d, n, k, f, th, kind)
        delta = fit_delta(d, ntr, ktr, th, kind, pen)
        rec = {
            "F": flat_ll(ntr, ktr, nte, kte),
            "G": _ll(shaped_p(d, ntr, ktr, th, kind), nte, kte),
            "P": _ll(shaped_p(d, ntr, ktr, th, kind, delta), nte, kte),
            "pen": pen,
            "th": [float(x) for x in th],
            "edges": int(kte.sum()),
        }
        if forms:
            for fm in KINDS:
                thf = th if fm == kind else fit_global(d, ntr, ktr, fm)
                rec[f"form_{fm}"] = _ll(shaped_p(d, ntr, ktr, thf, fm), nte, kte)
                rec[f"th_{fm}"] = [float(x) for x in thf]
        out.append(rec)
    return out


def _diffs(rows):
    s = {m: math.fsum(r[m] for r in rows) for m in ("F", "G", "P")}
    return {"G-F": s["G"] - s["F"], "P-G": s["P"] - s["G"]}


def score_unit(pop, h, kind="mix"):
    d = (np.arange(pop.nb) + 0.5) * BIN
    neuron = score_labels(d, h.n_fold, h.k_fold, K, kind, forms=True)
    interior = score_labels(d, h.n_int, h.k_int, K, kind)
    region = score_labels(d, h.n_block, h.k_block, NB_REG, kind)
    return {
        "neuron": _diffs(neuron),
        "interior": _diffs(interior),
        "region": [_diffs([r]) for r in region],
        "forms": {fm: math.fsum(r[f"form_{fm}"] for r in neuron) for fm in KINDS},
        "pen": [r["pen"] for r in neuron],
        "edge_instances": sum(r["edges"] for r in neuron),
        "th_forms": {fm: [r[f"th_{fm}"] for r in neuron] for fm in KINDS},
    }


# ---- bands and the reading (the T3 tri-state machinery, reused) --------------------------------
def form_gap(u):
    """The mixture's held-out advantage over the better single form, one seed (neuron folds)."""
    return u["forms"]["mix"] - max(u["forms"]["gauss"], u["forms"]["exp"])


def calibrate_bands(units):
    return {
        "form_gap": max(abs(form_gap(u)) for u in units),
        "neuron": {c: max(abs(u["neuron"][c]) for u in units) for c in CONTRASTS},
        "interior": {c: max(abs(u["interior"][c]) for u in units) for c in CONTRASTS},
        "region": [
            {c: max(abs(u["region"][r][c]) for u in units) for c in CONTRASTS}
            for r in range(NB_REG)
        ],
    }


def _state(vals, band):
    mean = math.fsum(vals) / len(vals)
    up = contrast("up", vals)
    if not up.degenerate and up.met and mean > band:
        return 1
    dn = contrast("down", [-x for x in vals])
    return -1 if (not dn.degenerate and dn.met and -mean > band) else 0


def states(units, bands):
    out = {"neuron": {}, "interior": {}, "region_blocks": [], "region_aggregate": {}}
    for sc in ("neuron", "interior"):
        for c in CONTRASTS:
            out[sc][c] = _state([u[sc][c] for u in units], bands[sc][c])
    for r in range(NB_REG):
        out["region_blocks"].append(
            {
                c: t3.value_state(
                    math.fsum(u["region"][r][c] for u in units) / len(units), bands["region"][r][c]
                )
                for c in CONTRASTS
            }
        )
    for c in CONTRASTS:
        tot = math.fsum(
            math.fsum(u["region"][r][c] for u in units) / len(units) for r in range(NB_REG)
        )
        out["region_aggregate"][c] = t3.value_state(
            tot, math.fsum(bands["region"][r][c] for r in range(NB_REG))
        )
    return out


def read_partb(st, frame_ok=True, planted_ok=True, pooling_sighted=True):
    """A pure function of the states, rows top to bottom. Returns (verdict, kernel kept). `pooling_sighted`: the
    planted pathway-width row fired P-G on H01 positions; without it a P-G reading is uninterpretable."""
    if not frame_ok:
        return "REFUSED_FRAME", None
    if not planted_ok:
        return "REFUSED", None
    sn, sz, sr = st["neuron"], st["interior"], st["region_aggregate"]
    if sn["G-F"] != 1:
        return "RED_PATHWAY_FLAT", None  # the kernel factor leaves the rule (the owner told)
    for c in CONTRASTS:
        if t3._opposite(sn[c], sz[c]):
            return "NOT_READ_BOUNDARY", None
        if t3._opposite(sn[c], sr[c]):
            return "NOT_READ_DISAGREE", None
    if any(blk["G-F"] != 1 for blk in st["region_blocks"]):
        return (
            "RED_PATHWAY_FLAT_REGION",
            None,
        )  # the kernel fails a held-out region (A3 row 6): the factor leaves
    if not pooling_sighted:
        return "NOT_READ_POOLING_UNSIGHTED", None
    if sn["P-G"] != 1:
        return "RED_POOL", "global"  # the global width alone is kept
    return "PASS", "pooled"


# ---- the planted stage and the gate ------------------------------------------------------------
def _unit(pop, seed, n_all, truth, n_edges):
    h, _ = build_hists(pop, seed, n_all=n_all, truth=truth, n_edges=n_edges)
    u = score_unit(pop, h)
    if truth.get("pathway"):
        u["pathway_check"] = pathway_check(pop, h, truth, u)
    u["seed"] = int(seed)
    u["edges_drawn"] = int(h.k_fold.sum())
    return u


def pathway_check(pop, h, truth, unit):
    """B1: P's fitted per-pathway widths against the planted ones. The single-kernel (Gaussian) shape and the
    per-cell offsets are fitted on ALL pairs with the penalty this seed's nested folds chose most often."""
    d = (np.arange(pop.nb) + 0.5) * BIN
    n, k = h.n_fold.sum((0, 1)), h.k_fold.sum((0, 1))
    th = fit_global(d, n, k, "gauss")
    pens = unit["pen"]
    delta = fit_delta(d, n, k, th, "gauss", max(set(pens), key=pens.count))
    scale = pathway_scales(truth, pop)
    cells = [int(c) for c in np.flatnonzero(scale != 1.0)]
    return {
        "cells": cells,
        "planted": [float(truth["width"] * scale[c]) for c in cells],
        "fitted": [float(math.exp(th[0]) * math.exp(delta[c])) for c in cells],
    }


def pathway_recovered(units):
    """Every planted pathway's fitted width within PATHWAY_RECOVERY_TOL of the planted one, on EVERY seed."""
    for u in units:
        chk = u.get("pathway_check")
        if not chk or len(chk["cells"]) != 2 or len(chk["fitted"]) != 2:
            return False
        if any(
            abs(f - p) > PATHWAY_RECOVERY_TOL * p for f, p in zip(chk["fitted"], chk["planted"])
        ):
            return False
    return True


def recovered(truth, units):
    """Amendment 4 (i): on a single-width kernel row the width is recovered within 10% on every seed (the
    fold-mean of the fitted width) and the matching form beats the other single form on held-out LL."""
    form = truth["form"]
    if truth.get("pathway"):
        return pathway_recovered(units)
    if form not in ("gauss", "exp"):
        return True
    col = 0 if form == "gauss" else 1
    other = "exp" if form == "gauss" else "gauss"
    for u in units:
        ths = u["th_forms"][form]
        w = math.fsum(math.exp(th[col]) for th in ths) / len(ths)
        if abs(w - truth["width"]) > 0.10 * truth["width"] or u["forms"][form] <= u["forms"][other]:
            return False
    return True


def no_positive_kernel(st):
    """A5's validation bar: no kernel-over-flat state is +1 under ANY scoring (neuron, interior, every region
    block, the region aggregate)."""
    states_ = [st["neuron"]["G-F"], st["interior"]["G-F"], st["region_aggregate"]["G-F"]]
    return all(x != 1 for x in states_ + [blk["G-F"] for blk in st["region_blocks"]])


def _holds(row, units, bands):
    st = states(units, bands)
    verdict, _ = read_partb(st)
    want = {"red_flat": "RED_PATHWAY_FLAT", "red_pool": "RED_POOL"}[row.expect]
    if row.expect == "red_flat" and not no_positive_kernel(st):
        return False
    return verdict == want and recovered(row.truth, units)


def pooling_sighted(plan, rows, bands):
    """The explicit gate: the planted pathway-width row's neuron-fold P-G state is +1 (P sees pathway widths on
    these positions). Not part of `all_rows_hold`: its failure turns every P-G reading into NOT_READ."""
    gate = [r for r in plan if r.expect == "pooling_gate"]
    return bool(gate) and all(
        states(rows[r.name]["units"], bands)["neuron"]["P-G"] == 1
        and recovered(r.truth, rows[r.name]["units"])
        for r in gate
    )


def plan_dicts(plan):
    return [r._asdict() | {"truth": dict(r.truth)} for r in plan]


def _plan_json(plan_like):
    """Names, truths, seeds, expectations and ORDER, normalised through JSON (tuples read back as lists)."""
    return json.loads(json.dumps(plan_like))


def run_planted(pop, plan=PLAN, n_edges=t3.N_EDGES, log=None):
    """Positions + layer/sign only. Calibration -> bands (locked by hash) -> every other row."""
    n_all = _all_pairs_n(pop)
    rows, bands = {}, None
    for row in plan:
        units = [_unit(pop, s, n_all, row.truth, n_edges) for s in row.seeds]
        rows[row.name] = {"units": units}
        if row.expect is None:
            bands = calibrate_bands(units)
        elif row.expect == "pooling_gate":
            rows[row.name]["sighted"] = pooling_sighted((row,), rows, bands)
        else:
            rows[row.name]["reading"] = read_partb(states(units, bands))[0]
            rows[row.name]["holds"] = _holds(row, units, bands)
        if log:
            log(row.name, rows[row.name].get("reading"))
    return {
        "plan": plan_dicts(plan),
        "population_sha256": fingerprint(pop),
        "n_edges_target": n_edges,
        "bands": bands,
        "bands_sha256": hashlib.sha256(json.dumps(bands, sort_keys=True).encode()).hexdigest(),
        "rows": rows,
        "all_rows_hold": all(r.get("holds", True) for r in rows.values()),
        "pooling_sighted": pooling_sighted(plan, rows, bands),
    }


def validate_planted(record, pop, plan=PLAN, n_edges=t3.N_EDGES):
    """Recomputed from the record, never a stored verdict (T3's 2.15 rule): plan, population, seeds, bands and
    every row's reading. An honest-record gate, not an anti-forgery one."""
    if not record:
        raise ExposureRefused("no planted record: H01's edges are not read")
    if (
        record.get("population_sha256") != fingerprint(pop)
        or record.get("n_edges_target") != n_edges
    ):
        raise ExposureRefused("the planted record was drawn on a different population or density")
    if _plan_json(record["plan"]) != _plan_json(plan_dicts(plan)) or set(record["rows"]) != {
        r.name for r in plan
    }:
        raise ExposureRefused("the planted record's plan differs from the authorized plan")
    for r in plan:
        if [int(u["seed"]) for u in record["rows"][r.name]["units"]] != list(r.seeds):
            raise ExposureRefused(f"row {r.name}: its units are not the plan's seeds")
    bands = calibrate_bands(record["rows"][plan[0].name]["units"])
    if bands != record["bands"]:
        raise ExposureRefused("the planted bands do not recompute from the calibration units")
    for r in plan[1:]:
        if r.expect != "pooling_gate" and not _holds(r, record["rows"][r.name]["units"], bands):
            raise ExposureRefused(f"planted row {r.name} does not hold on recomputation")
    return bands


def score_h01(path, planted_record, seeds=SEEDS, pin=None, t3_ref=None):
    """The only edge read. Refuses unless the planted record validates (and the seeds are the registered 8)."""
    if not t3_ref:
        raise ExposureRefused(
            "Part B reads only after a READ T3 record with a kernel (A13): none given"
        )
    z, sha = t3._verified(path, pin)
    pop = make_pop(t3.xyz_um(z["xyz_nm"]), z["layer"], z["sign"])
    if tuple(seeds) != SEEDS:
        raise ExposureRefused("H01's edges are read on the registered confirmatory seeds only")
    bands = validate_planted(planted_record, pop)
    if frame_violation(pop.layer, pop.fr[0])[0] > FRAME_VIOLATION_MAX:
        return {"reading": "REFUSED_FRAME", "artifact_sha256": sha}
    edges = (np.asarray(z["pre_idx"]), np.asarray(z["post_idx"]))
    units, first = [], None
    for s in seeds:
        h, _ = build_hists(pop, s, edges=edges)
        first = first or h
        u = score_unit(pop, h)
        u["seed"] = int(s)
        units.append(u)
    st = states(units, bands)
    sighted = pooling_sighted(PLAN, planted_record["rows"], bands)
    verdict, keep = read_partb(st, pooling_sighted=sighted)
    out = {
        "reading": verdict,
        "keep": keep,
        "pooling_sighted": sighted,
        "states": st,
        "units": units,
        "artifact_sha256": sha,
    }
    if verdict in ("PASS", "RED_POOL"):
        out["record"] = build_record(pop, first, units, verdict, keep, edges, sha, t3_ref, bands)
    return out


def select_form(units, bands):
    """The form the record exports (frozen before any H01 read; T3's A2 shape on the Part B forms): the better
    single kernel by summed held-out LL (tie order: Gaussian), upgraded to the mixture iff the mixture's advantage
    over that single form is a MET paired contrast over the 8 seeds above the calibrated `form_gap` band."""
    ll = {f: math.fsum(u["forms"][f] for u in units) for f in KINDS}
    single = "gauss" if ll["gauss"] >= ll["exp"] else "exp"
    gaps = [u["forms"]["mix"] - u["forms"][single] for u in units]
    return "mix" if _state(gaps, bands["form_gap"]) == 1 else single


def export_params(kind, th):
    """(a, c, sigma_um, lambda_um) of the exported kernel: an unused component carries weight 0 (a Gaussian-only
    kernel c = 0 with lambda_um a placeholder = sigma; an exponential-only kernel a = 0 with sigma_um = 1)."""
    sigma, lam, rho = (math.exp(float(x)) for x in th)
    if kind == "gauss":
        return 1.0, 0.0, sigma, sigma
    if kind == "exp":
        return 0.0, 1.0, 1.0, lam
    return 1.0, rho, sigma, lam


def build_record(pop, h, units, verdict, keep, edges, sha, t3_ref=None, bands=None):
    """The record growth/wiring_prior.py reads (schema agreed with G7). The exported kernel is the SELECTED form
    (select_form) fitted on ALL pairs and edges, so a tangential disagreement with T3's form changes what is
    delivered; the disagreement is recorded. Per-cell amplitudes, per-cell widths if pooled, beta = 0 until B2."""
    d = (np.arange(pop.nb) + 0.5) * BIN
    n, k = h.n_fold.sum((0, 1)), h.k_fold.sum((0, 1))
    kind = select_form(units, bands)
    th = fit_global(d, n, k, kind)
    a_w, c_w, sigma, lam = export_params(kind, th)
    live = np.flatnonzero(n.sum(1) > 0)
    t3_k = (t3_ref or {}).get("K")
    pens = [p for u in units for p in u["pen"]]
    pen = max(set(pens), key=pens.count)
    delta = fit_delta(d, n, k, th, kind, pen) if keep == "pooled" else np.zeros(N_CELL)
    # N1: amplitudes are profiled AFTER the final offsets, so they belong to the widths the record delivers
    amp, _ = _profile(n, k, shape(d[None, :] * np.exp(-delta)[:, None], th, kind))
    rec = {
        "source": "part_b",
        "frame": "tangential",
        "reading": verdict,
        "keep": keep,
        "form": kind,
        "form_t3": t3_k,
        "form_disagrees_t3": t3_k is not None and t3_k != kind,
        "a": a_w,
        "c": c_w,
        "sigma_um": sigma,
        "lambda_um": lam,
        "pathway_amplitudes": {int(c): float(amp[c]) for c in live},
        "beta": 0.0,
        "beta_fallback": True,
        "dbar": dbar(pop.layer, edges[0], edges[1]),
        "density_per_mm2": tangential_density(pop),
        "frame_violation": dict(zip(("fraction", "count"), frame_violation(pop.layer, pop.fr[0]))),
        "artifact_sha256": sha,
        "t3_record": t3_ref,
    }
    if keep == "pooled":
        rec["penalty"] = pen
        rec["pathway_widths"] = {
            int(c): [
                sigma * math.exp(float(delta[c])),
                lam * math.exp(float(delta[c])),
            ]  # (sigma_um, lambda_um)
            for c in live
            if delta[c] != 0
        }
    return rec


def dbar(layer, pre, post):
    """Mean degree 2E/N of the symmetrised simple graph induced on layers L2-L6 (codes 1..5): amendment 2026-10-04."""
    ok = (
        (layer[pre] >= 1)
        & (layer[pre] <= 5)
        & (layer[post] >= 1)
        & (layer[post] <= 5)
        & (pre != post)
    )
    e = {(min(a, b), max(a, b)) for a, b in zip(pre[ok].tolist(), post[ok].tolist())}
    nodes = np.flatnonzero((layer >= 1) & (layer <= 5))
    recip = len({(a, b) for a, b in zip(pre[ok].tolist(), post[ok].tolist())}) - len(e)
    return {
        "N": int(len(nodes)),
        "E": len(e),
        "dbar": 2.0 * len(e) / len(nodes),
        "reciprocal_pairs": recip,
    }


def tangential_density(pop):
    """Somata per mm^2 of sheet, pooled and per band: N / ((r_mean x theta range) x z range), um^2 -> mm^2."""
    r, th, z = pop.fr
    out = {}
    for b in [None, *np.unique(pop.layer).tolist()]:
        m = np.ones(len(r), bool) if b is None else pop.layer == b
        area = float(r[m].mean() * (th[m].max() - th[m].min()) * (z[m].max() - z[m].min())) / 1e6
        out["pooled" if b is None else int(b)] = float(m.sum()) / area
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


CRLF, LF = bytes([13, 10]), bytes([10])


T3_READ = ("RED_EXP", "PASS_FAVOURS_GAUSSIAN", "PASS_GAUSSIAN_NOT_REFUSED", "PASS_MIXTURE")


def check_t3_record(path, pin=None):
    """A13's gate on the T3 record Part B cites: its artifact is the pinned one, its reading is a READ with a
    kernel (not REFUSED, NOT READ, RED-flat), and it carries its git head and harness hashes. Returns the ref the
    record serialises: the committed (LF) blob hash, the verdict, K, the head."""
    raw = Path(path).read_bytes()
    rec = json.loads(raw)
    sha = (pin or t3.pinned_artifact())[0]
    reading = (rec.get("h01") or {}).get("reading") or {}
    if rec.get("artifact_sha256") != sha:
        raise ExposureRefused("the T3 record names a different artifact than the pin")
    if reading.get("verdict") not in T3_READ or reading.get("K") not in ("gauss", "exp", "mix"):
        raise ExposureRefused(
            f"T3 is not a READ with a kernel: {reading.get('verdict')!r} K={reading.get('K')!r}"
        )
    if not rec.get("git_head") or not rec.get("harness_sha256"):
        raise ExposureRefused("the T3 record carries no git head or harness hashes")
    return {
        "path": str(path),
        "blob_sha256": hashlib.sha256(raw.replace(CRLF, LF)).hexdigest(),
        "verdict": reading["verdict"],
        "K": reading["K"],
        "git_head": rec["git_head"],
    }


HARNESS = (
    "src/snn_data_lake/instruments/h01_partb_tangential.py",
    "src/snn_data_lake/instruments/h01_distance_kernel.py",
    "src/snn_data_lake/instruments/paired_contrast.py",
    "src/snn_data_lake/growth/wiring_prior.py",
    "tests/test_h01_partb_tangential.py",
    "tests/test_wiring_prior.py",
)


def provenance():
    """The commit and the harness files' hashes (LF-normalised) a campaign record must carry (A8's ledger)."""
    import subprocess

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=t3.REPO, text=True).strip()
    sha = {
        f: hashlib.sha256((t3.REPO / f).read_bytes().replace(CRLF, LF)).hexdigest() for f in HARNESS
    }
    return {"git_head": head, "harness_sha256": sha}


def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("planted", help="the planted stage (positions, layer, sign only)")
    h = sub.add_parser("h01", help="the H01 edge read; refuses without a valid planted record")
    for q in (p, h):
        q.add_argument("--artifact", required=True)
        q.add_argument("--out", required=True)
    h.add_argument("--planted", required=True)
    h.add_argument("--t3", required=True, help="tools/g4_t3_campaign.json, cited in the record")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "planted":
        out = run_planted(load_pop(a.artifact), log=lambda n, r: print(n, r, flush=True))
    else:
        out = score_h01(
            a.artifact,
            json.loads(Path(a.planted).read_text(encoding="utf-8")),
            t3_ref=check_t3_record(a.t3),
        )
        print("Part B reads:", out["reading"], out.get("keep"))
    out.update(provenance())
    if "record" in out:
        out["record"]["provenance"] = provenance()
    out["wall_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(_jsonable(out), indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
