r"""D-R0's drift instrument: the Marks & Goard (2021) stability index, transcribed from the authors'
published computation (pre-registration specs/022-unified-brain-arc/evidence/d-r0-representational-drift-
instrument.md section 1.4), plus the four synthetic criteria its section 2.2 registers.

  stability_index(s1, s2, rng) -- one neuron, two sessions (trials x frames each): random half-splits with the
                                  SAME indices in both sessions, within-session (ws) and between-session (bs)
                                  Pearson, each pair averaged and rectified per iteration, 50 iterations at 8
                                  trials (100 if more), CC_ws <= 0 -> 0.001, RDI = (ws - bs) / (ws + bs)
  mov_index(s1, s2, rng)       -- the natural-movie path: 8 of N trials (the same indices in both sessions),
                                  10 times, each through stability_index, averaged
  DriftReading                 -- (cc_ws, cc_bs, rdi)

MERGED 2026-11-15 (P2, final-project M1): the MAP D0 readout-geometry half from branch
origin/cloud/drift-d0-instrument-1102 (ddaaf233) now lives here too, so ONE module serves both datasets:
  pv_correlation / drift_index   population-vector correlation per stimulus, averaged (A/A = 1, 0)
  tuning_correlation             per-neuron correlation of the response profile across stimuli
  preferred_shift                circular difference of preferred orientation (0/180 wrap)
  readout_null_split             the fraction of a population change in null(W) of a readout W
  random_readout_null_fractions  the same fraction for random readouts of matched rank (the control)
  decoder_accuracy               the closed-form readout, CAPACITY-GUARDED (refuses to interpolate)
  procrustes_realign / recovered the best orthogonal re-alignment of one session onto another
  read_drift -> ReadoutDrift     the above on a two-session labelled pair (renamed from the branch's
                                 DriftReading, which collided with the RDI tuple above)
RSA geometry stays in ONE home, `infra/_mathutils.rsa_drift` (tests/test_rsa_drift.py); not copied here.

Correlations are Pearson, vectorised across iterations (`_rowcorr`); the self-check ties it to
`numpy.corrcoef` to 1e-12. Distinct-from `faculties/grid_place_map._pearson` and `infra/drift_monitor.psi`.

Criteria (each named `*_verdict`; each has its constructed red in `_selfcheck`): aa_stability_verdict,
planted_drift_dose_verdict, mov_vs_mov_control_verdict, author_agreement_verdict. The synthetic half only:
the real-data halves belong to `instruments/d_r0_part_a.py`. Pure numpy + scipy.stats; deterministic; ASCII prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.representational_drift
  ... --pin-ref   (writes tools/d_r0_synthetic_ref.json; refuses to overwrite an existing pin)
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr

# distinct-from: instruments/prior_drift_harness.py, report_order_probe.py, carried_state_report.py, binding_instrument.py, episode_driver.py -- the five sites the write hook listed, each read: a ridge solve, per-seed readings, a trial stream; none computes a split-half within/between-session stability index (graph-exists on five phrasings: no drift estimator; d-r0-build-row/graph-exists.log)

REF_PATH = Path(__file__).resolve().parents[3] / "tools" / "d_r0_synthetic_ref.json"

N_FRAMES = 300
SIGMA = 1.0
LENGTH_SCALE = 10.0
SEED = 20261103
SEED_REF = 20261104
SEED_BOOT = 20261105
N_REF = 2000
DOSES = (0.4, 0.55, 0.7, 0.85, 1.0)
REF_DOSES = (0.0, 0.4, 0.55, 0.6, 0.7, 0.85, 1.0)
AA_BAR = 0.01
MOV_FLOOR = 0.05
MOV_CAP = 0.10
N_BOOT = 1000


class DriftReading(NamedTuple):
    cc_ws: float
    cc_bs: float
    rdi: float


def _rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    with np.errstate(
        invalid="ignore", divide="ignore"
    ):  # constant row: NaN as in MATLAB; fmax reads 0
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def stability_index(s1, s2, rng):
    """One neuron: `s1`, `s2` are (trials, frames), session 1 the reference; `rng` a numpy Generator."""
    n = s1.shape[0]
    iters = 50 if n <= 8 else 100
    idx = np.stack([rng.permutation(n) for _ in range(iters)])
    h1, h2 = idx[:, : n // 2], idx[:, n // 2 :]
    a1, a2 = s1[h1].mean(1), s1[h2].mean(1)
    b1, b2 = s2[h1].mean(1), s2[h2].mean(1)
    ws = np.fmax((_rowcorr(a1, a2) + _rowcorr(b1, b2)) / 2, 0.0).mean()
    bs = np.fmax((_rowcorr(a1, b2) + _rowcorr(a2, b1)) / 2, 0.0).mean()
    if ws <= 0:
        ws = 0.001
    return DriftReading(float(ws), float(bs), float((ws - bs) / (ws + bs)))


def mov_index(s1, s2, rng, pick=8, reps=10):
    rs = []
    for _ in range(reps):
        idx = rng.choice(s1.shape[0], pick, replace=False)
        rs.append(stability_index(s1[idx], s2[idx], rng))
    return DriftReading(*(float(np.mean(col)) for col in zip(*rs)))


def population_rdi(S1, S2, fn, seed):
    """RDI of every neuron (rows of the (neurons, trials, frames) arrays) through `fn`, one rng stream."""
    rng = np.random.default_rng(seed)
    return np.array([fn(a, b, rng).rdi for a, b in zip(S1, S2)])


# =============================================================================================
# Readout geometry half (MAP D0, merged from origin/cloud/drift-d0-instrument-1102 ddaaf233)
# distinct-from: instruments/prior_drift_harness.py -- WEIGHT drift off the H01 prior (DriftRecord); its
#   closed-form readout is REUSED below (fit_readout / predict), not copied (P2).
# =============================================================================================
#: P1, a NUMERICAL tolerance (not a behavioural bar): an orthogonal planted drift is undone by
#: orthogonal Procrustes to float64 round-off; 1e-8 leaves six orders of headroom.
RECOVERY_TOL = 1e-8


class CapacityRefusal(ValueError):
    """A decoder with at least as many free parameters as training samples interpolates any labelling."""


def _corr(x, y):
    """Pearson of two vectors as num / sqrt(sxx * syy), so identical inputs read exactly 1.0."""
    xc = np.asarray(x, dtype=np.float64) - np.mean(x)
    yc = np.asarray(y, dtype=np.float64) - np.mean(y)
    sxx, syy = float(xc @ xc), float(yc @ yc)
    if sxx == 0.0 or syy == 0.0:
        raise ValueError("a constant vector has no correlation")
    return float(xc @ yc) / math.sqrt(sxx * syy)


def _same_shape(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"sessions differ in shape: {a.shape} vs {b.shape}")
    return a, b


def pv_correlation(a, b) -> float:
    """Mean over stimuli (columns) of the correlation between two sessions' population vectors.
    `a`, `b`: (n_neurons, n_stimuli) trial-averaged responses of the SAME neurons."""
    a, b = _same_shape(a, b)
    return float(np.mean([_corr(a[:, j], b[:, j]) for j in range(a.shape[1])]))


def drift_index(a, b) -> float:
    """1 - pv_correlation: 0 for identical sessions, larger as the population code moves."""
    return 1.0 - pv_correlation(a, b)


def tuning_correlation(a, b):
    """Per neuron (row), the correlation of its response profile across stimuli between sessions."""
    a, b = _same_shape(a, b)
    return np.array([_corr(a[i], b[i]) for i in range(a.shape[0])])


def preferred_shift(pref_a, pref_b, period=180.0) -> float:
    """Signed circular change of a preferred orientation (degrees) in [-period/2, period/2)."""
    half = period / 2.0
    return float((pref_b - pref_a + half) % period - half)


def readout_null_split(delta, w) -> float:
    """Fraction of the squared norm of a change `delta` ((n,) or (n, m)) in the null space of readout
    `w` (k, n). 1.0 = invisible to the readout, 0.0 = entirely in its row space."""
    d = np.asarray(delta, dtype=np.float64)
    d = d.reshape(d.shape[0], -1)
    total = float(np.sum(d * d))
    if total == 0.0:
        raise ValueError("a zero change has no null fraction")
    _, s, vt = np.linalg.svd(np.asarray(w, dtype=np.float64), full_matrices=False)
    rank = int(np.sum(s > s.max() * max(np.asarray(w).shape) * np.finfo(np.float64).eps))
    row = vt[:rank] @ d
    return 1.0 - float(np.sum(row * row)) / total


def random_readout_null_fractions(delta, rank, n_draws, seed):
    """The control that is not the method: the null split for `n_draws` Gaussian readouts of the same
    rank. For isotropic change its mean is about (n - rank) / n."""
    rng = np.random.default_rng(seed)
    n = np.asarray(delta).shape[0]
    return np.array(
        [readout_null_split(delta, rng.normal(size=(rank, n))) for _ in range(int(n_draws))]
    )


def decoder_accuracy(x_train, y_train, x_test, y_test, k) -> float:
    """Train prior_drift_harness's closed-form readout and return its test accuracy; refuses interpolation."""
    from snn_data_lake.instruments.prior_drift_harness import fit_readout, predict

    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.shape[1] + 1 >= x_train.shape[0]:
        raise CapacityRefusal(
            f"{x_train.shape[1]} features + bias > {x_train.shape[0]} training samples: the readout "
            "would interpolate any labelling"
        )
    coef = fit_readout(x_train, y_train, k)
    hits = [predict(coef, row, k) == int(label) for row, label in zip(x_test, y_test, strict=True)]
    return sum(hits) / len(hits)


def procrustes_realign(source, target):
    """The orthogonal R minimising ||source @ R - target||, and the relative residual."""
    from scipy.linalg import orthogonal_procrustes

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    r, _ = orthogonal_procrustes(source, target)
    return r, float(np.linalg.norm(source @ r - target) / np.linalg.norm(target))


def recovered(rel_residual, tol=RECOVERY_TOL) -> bool:
    """True iff an orthogonal re-alignment undid the change to numerical precision."""
    return rel_residual <= tol


@dataclass(frozen=True)
class ReadoutDrift:
    """One two-session readout reading (was the branch's DriftReading; renamed, P2 merge)."""

    drift_index: float
    null_fraction: float | None
    null_fraction_random: float | None
    frozen_acc_day0: float
    frozen_acc_day1: float
    realigned_acc_day1: float


def _class_means(x, y, k):
    return np.stack([x[y == c].mean(axis=0) for c in range(k)])


def read_drift(x0, y0, x1, y1, k, null_draws=200, seed=0) -> ReadoutDrift:
    """Rule et al. (2020) aims on one labelled session pair (rows = trials, columns = the SAME neurons).
    Even trials train/calibrate, odd trials are judged; re-alignment is fitted on even-trial class means
    only; the null fraction splits the class-mean change against the frozen readout, beside the
    matched-rank random control."""
    from snn_data_lake.instruments.prior_drift_harness import fit_readout

    x0, x1 = np.asarray(x0, dtype=np.float64), np.asarray(x1, dtype=np.float64)
    y0, y1 = np.asarray(y0), np.asarray(y1)
    tr0, te0 = np.arange(0, len(y0), 2), np.arange(1, len(y0), 2)
    tr1, te1 = np.arange(0, len(y1), 2), np.arange(1, len(y1), 2)
    m0, m1 = _class_means(x0, y0, k), _class_means(x1, y1, k)
    acc0 = decoder_accuracy(x0[tr0], y0[tr0], x0[te0], y0[te0], k)
    acc1 = decoder_accuracy(x0[tr0], y0[tr0], x1[te1], y1[te1], k)
    r, _ = procrustes_realign(_class_means(x1[tr1], y1[tr1], k), _class_means(x0[tr0], y0[tr0], k))
    acc1r = decoder_accuracy(x0[tr0], y0[tr0], x1[te1] @ r, y1[te1], k)
    delta = (m1 - m0).T
    if np.any(delta):
        w = fit_readout(x0[tr0], y0[tr0], k)[:-1].T  # (k, n): weights, bias row dropped
        null = readout_null_split(delta, w)
        rand = float(
            np.mean(
                random_readout_null_fractions(
                    delta, int(np.linalg.matrix_rank(w)), null_draws, seed
                )
            )
        )
    else:
        null = rand = None
    return ReadoutDrift(drift_index(m0.T, m1.T), null, rand, acc0, acc1, acc1r)


def _rotation(n, dose, seed=0):
    from scipy.linalg import expm

    a = np.random.default_rng(seed).normal(size=(n, n))
    return expm(dose * (a - a.T) / np.sqrt(n))


def _readout_reds():
    """The six constructed reds of the merged half; returns the list of failures (empty = all fire)."""
    rng = np.random.default_rng(0)
    fails = []
    a = rng.normal(size=(40, 12))
    if drift_index(a, a.copy()) != 0.0:
        fails.append("red1: A/A did not read exactly zero")
    x = a.T
    drifts = []
    for dose in (0.05, 0.2, 0.5, 1.0):
        r = _rotation(x.shape[1], dose)
        _, rel = procrustes_realign(x @ r, x)
        if not recovered(rel):
            fails.append(f"red2: rotation at dose {dose} not recovered (rel {rel:.2e})")
        drifts.append(drift_index(x.T, (x @ r).T))
    if drifts != sorted(drifts) or drifts[0] <= 0.0:
        fails.append(f"red2: drift does not grow with the dose {drifts}")
    w = rng.normal(size=(4, 30))
    _, _, vt = np.linalg.svd(w)
    if abs(readout_null_split(vt[4:].T @ rng.normal(size=(26, 5)), w) - 1.0) > 1e-12:
        fails.append("red3: null-confined drift did not read 1")
    if abs(readout_null_split(vt[:4].T @ rng.normal(size=(4, 5)), w)) > 1e-12:
        fails.append("red3: row-space drift did not read 0")
    if abs(preferred_shift(170.0, 10.0) - 20.0) > 1e-12:
        fails.append("red4: the 0/180 wrap misread")
    try:
        decoder_accuracy(rng.normal(size=(10, 10)), np.arange(10) % 2, np.zeros((1, 10)), [0], 2)
        fails.append("red5: an over-parameterised decoder was not refused")
    except CapacityRefusal:
        pass
    try:  # the boundary: 9 features + bias = 10 parameters on 10 samples also interpolates (review fold)
        decoder_accuracy(rng.normal(size=(10, 9)), np.arange(10) % 2, np.zeros((1, 9)), [0], 2)
        fails.append("red5: the d + 1 == n boundary was not refused")
    except CapacityRefusal:
        pass
    shear = np.eye(x.shape[1])
    shear[: x.shape[1] // 2] *= 1.5
    if recovered(procrustes_realign(x @ shear, x)[1]):
        fails.append("red6: a non-orthogonal drift read as recovered")
    return fails


# =============================================================================================
# Synthetic generator (pre-registration 2.2): smooth GP mean per neuron, i.i.d. Gaussian noise
# =============================================================================================
def _patterns(rng, n):
    t = np.arange(N_FRAMES)
    k = np.exp(-0.5 * ((t[:, None] - t[None, :]) / LENGTH_SCALE) ** 2) + 1e-6 * np.eye(N_FRAMES)
    return (np.linalg.cholesky(k) @ rng.standard_normal((N_FRAMES, n))).T


def _sessions(rng, p1, p2, dose, trials):
    n = p1.shape[0]
    m2 = np.sqrt(1 - dose**2) * p1 + dose * p2
    s1 = p1[:, None, :] + SIGMA * rng.standard_normal((n, trials, N_FRAMES))
    s2 = m2[:, None, :] + SIGMA * rng.standard_normal((n, trials, N_FRAMES))
    return s1, s2


def synth_pair(n, dose, seed, trials=8):
    rng = np.random.default_rng(seed)
    p1, p2 = _patterns(rng, n), _patterns(rng, n)
    return _sessions(rng, p1, p2, dose, trials)


def reference(n=N_REF, seed=SEED_REF):
    """The pinned expectation: the built estimator on `n` neurons, per dose (mean and per-neuron SD)."""
    out = {}
    for d in REF_DOSES:
        v = population_rdi(*synth_pair(n, d, seed), stability_index, [seed, 1])
        out[repr(d)] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    return {"n": n, "seed": seed, "sigma": SIGMA, "doses": out}


def load_reference(path=REF_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def se_at(ref, dose, n=200):
    return ref["doses"][repr(dose)]["sd"] / np.sqrt(n)


def se_triggers(ref):
    """Section 2.2 per-bar trigger: a 200-neuron reference SE above bar / 3 means the bar must be amended."""
    hits = []
    if se_at(ref, 0.0) > AA_BAR / 3:
        hits.append(f"aa_stability: SE200 {se_at(ref, 0.0):.5f} > {AA_BAR / 3:.5f}")
    worst = max(se_at(ref, 0.0), se_at(ref, 0.6))
    if worst > MOV_FLOOR / 3:
        hits.append(f"mov_vs_mov floor: SE200 {worst:.5f} > {MOV_FLOOR / 3:.5f}")
    return hits


# =============================================================================================
# The four criteria (synthetic halves)
# =============================================================================================
def aa_stability_verdict(fn, dose=0.0, n=200):
    """MET iff two sessions of ONE generative model read mean RDI within +-0.01 of 0. `dose` > 0 is the red."""
    m = float(population_rdi(*synth_pair(n, dose, SEED), fn, SEED).mean())
    return abs(m) <= AA_BAR, m


def planted_drift_dose_verdict(fn, ref, n=200):
    """MET iff the five doses read monotone increasing, each within max(10% of the reference, 3 x its SE)."""
    means, within = [], []
    for d in DOSES:
        m = float(population_rdi(*synth_pair(n, d, SEED), fn, SEED).mean())
        r = ref["doses"][repr(d)]["mean"]
        within.append(abs(m - r) <= max(0.1 * r, 3 * se_at(ref, d, n)))
        means.append(m)
    return bool(all(within) and all(np.diff(means) > 0)), means


def mov_vs_mov_control_verdict(n, drift_a=0.0, seed=SEED):
    """MET iff |D_null| <= max(0.05, 3 x neuron-bootstrap SE) with the bar <= 0.10. Arm A may carry planted drift."""
    rng = np.random.default_rng(seed)
    p1, p2 = _patterns(rng, n), _patterns(rng, n)
    arms = [_sessions(rng, p1, p2, d, 15) for d in (drift_a, 0.0)]
    rdi = [population_rdi(*a, mov_index, [seed, 2, k]) for k, a in enumerate(arms)]
    diff = rdi[0] - rdi[1]
    boot = np.random.default_rng(SEED_BOOT)
    se = float(boot.choice(diff, (N_BOOT, n)).mean(1).std(ddof=1))
    bar = max(MOV_FLOOR, 3 * se)
    d_null = float(diff.mean())
    return bool(abs(d_null) <= bar and bar <= MOV_CAP), d_null, bar


def rank_corr(x, y):
    """Spearman rank correlation: the ONE scipy entry point for the drift instruments (d_r0_part_a reuses it, so
    instruments/ keeps one scipy importer for this job; native_ratchet)."""
    return float(spearmanr(x, y)[0])


def author_agreement_verdict(ceil, self_mad, ours, theirs):
    """Pass: ceil >= 0.9, Spearman(ours, theirs) >= max(0.9 ceil, 0.8), median |ours - theirs| <= min(2 self_mad,
    0.05), at least 200 neuron-sessions."""
    rho = float(spearmanr(ours, theirs)[0])
    mad = float(np.median(np.abs(ours - theirs)))
    ok = (
        ceil >= 0.9
        and rho >= max(0.9 * ceil, 0.8)
        and mad <= min(2 * self_mad, 0.05)
        and len(ours) >= 200
    )
    return bool(ok), rho, mad


def _agreement_inputs(fn, s1, s2):
    r1, r2, theirs = (population_rdi(s1, s2, fn, [SEED, 3, k]) for k in (1, 2, 3))
    return float(spearmanr(r1, r2)[0]), float(np.median(np.abs(r1 - r2))), r1, theirs


def _ws_for_bs(s1, s2, rng):
    """The dose red's broken estimator (pre-registration 2.2): within-session CC used in place of between-session."""
    r = stability_index(s1, s2, rng)
    return DriftReading(r.cc_ws, r.cc_ws, (r.cc_ws - r.cc_ws) / (r.cc_ws + r.cc_ws))


def _shuffle_labels(s1, s2, seed):
    """Per neuron, pool both sessions' trials and re-split them at random."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([s1, s2], axis=1)
    out = np.stack([p[rng.permutation(p.shape[0])] for p in pooled])
    return out[:, : s1.shape[1]], out[:, s1.shape[1] :]


# =============================================================================================
# E-PC: drift in the engine's live predictive-coding layer, LEARN vs a frozen twin (MOVED 2026-11-15 from
# instruments/drift_studies.py; pre-registration research/representational-drift/ENGINE-PC-DRIFT-PREREG.md).
# `python -m snn_data_lake.instruments.representational_drift engine-pc`
# =============================================================================================
SEEDS = range(10)
N_PROBES, REPEATS, SESSIONS, TICKS = 30, 10, 4, 300
NOISE = 0.05
R1, R2 = 16, 8  # the live LiveBrain dims; engine-pc --scale k multiplies both (amendment S10)


def raw_context(rng):
    return {
        "visual": rng.uniform(0, 1, 8),
        "temperature": rng.uniform(0, 1, 1),
        "audio": rng.uniform(0, 1, 8),
        "distance": rng.uniform(0, 1, 1),
    }


def encode(ctx, rng=None):
    from snn_data_lake.sensorimotor.multimodal_encoder import encode_multimodal

    c = {k: (v + rng.normal(0, NOISE, v.shape) if rng is not None else v) for k, v in ctx.items()}
    return encode_multimodal(c["visual"], c["temperature"], c["audio"], float(c["distance"][0]))


def settle(net, x):
    """Exactly LiveBrain.tick: reset, infer(steps=60, lr=0.05). Returns the 24 causes."""
    net.reset()
    net.infer(x, steps=60, lr=0.05)
    return np.concatenate([np.asarray(net.r1, dtype=float), np.asarray(net.r2, dtype=float)])


def imaging(net, probes, rng):
    """(24, REPEATS, N_PROBES): measurement only; the network is copied so no state leaks back."""
    probe_net = copy.deepcopy(net)
    out = np.zeros((R1 + R2, REPEATS, len(probes)))
    for r in range(REPEATS):
        for j, p in enumerate(probes):
            out[:, r, j] = settle(probe_net, encode(p, rng))
    return out


def run_arm(seed, learn):
    from snn_data_lake.readout.predictive_coding import PredictiveCodingNetwork

    rng_p = np.random.default_rng([seed, 1])
    probes = [raw_context(rng_p) for _ in range(N_PROBES)]
    x0 = encode(probes[0])
    net = PredictiveCodingNetwork(dims=(len(x0), R1, R2), seed=seed)
    stream = np.random.default_rng([seed, 2])  # identical stream in both arms
    noise = np.random.default_rng([seed, 3])  # identical trial noise in both arms
    sessions = []
    for s in range(SESSIONS):
        sessions.append(imaging(net, probes, noise))
        if s < SESSIONS - 1:
            for _ in range(TICKS):
                settle(net, encode(raw_context(stream)))
                if learn:
                    net.learn_step()
    return sessions


def stats(sessions):
    half = [(s[:, 0::2].mean(1), s[:, 1::2].mean(1)) for s in sessions]
    prof = [s.mean(1) for s in sessions]
    within = float(np.mean([pv_correlation(*h) for h in half]))
    rows = []
    for a, b in itertools.combinations(range(len(sessions)), 2):
        between = np.mean(
            [pv_correlation(half[a][0], half[b][1]), pv_correlation(half[a][1], half[b][0])]
        )
        iu = np.triu_indices(len(prof[a][0]), 1)
        geo = np.corrcoef(np.corrcoef(prof[a].T)[iu], np.corrcoef(prof[b].T)[iu])[0, 1]
        tun = float(np.nanmedian(tuning_correlation(prof[a], prof[b])))
        rows.append(
            {
                "pair": (a, b),
                "ticks": (b - a) * TICKS,
                "between": float(between),
                "geo": float(geo),
                "tun": tun,
            }
        )
    btw = [r["between"] for r in rows]
    return {
        "d1": within - float(np.mean(btw)),
        "d2": float(np.mean([r["geo"] - r["tun"] for r in rows])),
        "d3": float(np.polyfit([r["ticks"] for r in rows], btw, 1)[0]),
        "within": within,
        "pairs": rows,
    }


def readout(sessions):
    """read_drift on session 0 vs the last: trials = repeats x probes, labels = probe identity."""

    def flat(s):
        # rows probe-major (index = probe * repeats + repeat), so read_drift's even/odd row split is a
        # split over repeats and every probe appears on both sides
        x = s.transpose(2, 1, 0).reshape(-1, s.shape[0])  # (probes * repeats, 24)
        y = np.repeat(np.arange(s.shape[2]), s.shape[1])
        return x, y

    x0, y0 = flat(sessions[0])
    x1, y1 = flat(sessions[-1])
    return asdict(read_drift(x0, y0, x1, y1, k=N_PROBES))


def seed_sign_test(v, pos=True):
    v = np.asarray(v)
    v = v[np.isfinite(v) & (v != 0)]
    k = int(np.sum(v > 0 if pos else v < 0))
    n = v.size
    return k, n, sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def engine_pc_main(scale=1):
    """The registered run (scale 1) or amendment S10's 10x re-test (scale 10: 240 neurons, 20 repeats)."""
    global R1, R2, REPEATS
    saved = (R1, R2, REPEATS)
    if scale != 1:
        R1, R2, REPEATS = 16 * scale, 8 * scale, 20
    try:
        return _engine_pc_run("" if scale == 1 else f"_scale{scale}")
    finally:
        R1, R2, REPEATS = saved


def _engine_pc_run(suffix):
    per = []
    for seed in SEEDS:
        row = {"seed": seed}
        for arm, learn in (("learn", True), ("frozen", False)):
            sess = run_arm(seed, learn)
            row[arm] = stats(sess)
            row[arm + "_readout"] = readout(sess)
        per.append(row)
        print(
            f"seed {seed}: learn d1 {row['learn']['d1']:+.4f} d2 {row['learn']['d2']:+.4f} d3 {row['learn']['d3']:+.2e} | "
            f"frozen d1 {row['frozen']['d1']:+.4f} | acc day0 {row['learn_readout']['frozen_acc_day0']:.3f} "
            f"day3 {row['learn_readout']['frozen_acc_day1']:.3f} realigned {row['learn_readout']['realigned_acc_day1']:.3f}",
            flush=True,
        )
    tests = {
        "E-H1 d1(learn)-d1(frozen)>0": seed_sign_test(
            [r["learn"]["d1"] - r["frozen"]["d1"] for r in per]
        ),
        "E-H2 d2(learn)>0": seed_sign_test([r["learn"]["d2"] for r in per]),
        "E-H3 d3(learn)<0": seed_sign_test([r["learn"]["d3"] for r in per], pos=False),
        "RED d1(frozen)>0 (must NOT be met)": seed_sign_test([r["frozen"]["d1"] for r in per]),
    }
    for k, (hits, n, p) in tests.items():
        print(f"{k}: {hits}/{n} p={p:.4g} {'MET' if hits >= 9 and n == 10 else 'NOT MET'}")
    for key in ("d1", "d2", "d3"):
        for arm in ("learn", "frozen"):
            v = [r[arm][key] for r in per]
            print(f"{arm} {key}: mean {np.mean(v):+.4g} sd {np.std(v, ddof=1):.3g}")
    for k in (
        "frozen_acc_day0",
        "frozen_acc_day1",
        "realigned_acc_day1",
        "null_fraction",
        "null_fraction_random",
    ):
        for arm in ("learn", "frozen"):
            v = [r[arm + "_readout"][k] for r in per if r[arm + "_readout"][k] is not None]
            print(f"{arm} readout {k}: mean {np.mean(v):.4f}" if v else f"{arm} readout {k}: n/a")
    out = {"per_seed": per, "tests": {k: list(v) for k, v in tests.items()}}
    with open(
        f"research/representational-drift/engine_pc_drift{suffix}_result.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(out, fh, indent=1, default=str)
    return 0


# =============================================================================================
# Self-check
# =============================================================================================
def _selfcheck():
    failures = []

    def expect(label, got, want):
        print(("ok   " if got == want else "FAIL ") + label)
        if got != want:
            failures.append(f"{label}: got {got}, want {want}")

    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((8, 40)), rng.standard_normal((8, 40))
    ref_corr = np.corrcoef(a.mean(0), b.mean(0))[0, 1]
    if abs(_rowcorr(a.mean(0)[None], b.mean(0)[None])[0] - ref_corr) > 1e-12:
        failures.append("rowcorr != numpy.corrcoef")
    flat = stability_index(np.zeros((8, 30)), np.zeros((8, 30)), rng)
    if flat.cc_ws != 0.001:
        failures.append(f"floor: constant-zero sessions gave cc_ws {flat.cc_ws}")

    ref = load_reference() if REF_PATH.exists() else reference()
    expect("se triggers", se_triggers(ref), [])

    ok, m = aa_stability_verdict(stability_index)
    expect(f"aa green (mean {m:+.4f})", ok, True)
    ok, m = aa_stability_verdict(stability_index, dose=0.55)
    expect(f"aa red d=0.55 (mean {m:+.4f})", ok, False)
    ok, means = planted_drift_dose_verdict(stability_index, ref)
    expect(f"dose green ({[round(x, 3) for x in means]})", ok, True)
    ok, means = planted_drift_dose_verdict(_ws_for_bs, ref)
    expect(f"dose red, ws-for-bs ({[round(x, 3) for x in means]})", ok, False)
    for n in (200, 65):
        ok, d, bar = mov_vs_mov_control_verdict(n)
        expect(f"mov_vs_mov green n={n} (D {d:+.4f}, bar {bar:.4f})", ok, True)
        ok, d, bar = mov_vs_mov_control_verdict(n, drift_a=0.6)
        expect(f"mov_vs_mov red n={n} (D {d:+.4f}, bar {bar:.4f})", ok, False)
    s1, s2 = synth_pair(200, 0.7, SEED)
    ceil, self_mad, ours, theirs = _agreement_inputs(stability_index, s1, s2)
    ok, rho, mad = author_agreement_verdict(ceil, self_mad, ours, theirs)
    expect(
        f"agreement green (ceil {ceil:.3f}, rho {rho:.3f}, mad {mad:.4f}, self_mad {self_mad:.4f})",
        ok,
        True,
    )
    sh1, sh2 = _shuffle_labels(s1, s2, SEED)
    shuf = population_rdi(sh1, sh2, stability_index, [SEED, 3, 1])
    ok, rho, mad = author_agreement_verdict(ceil, self_mad, shuf, theirs)
    expect(f"agreement red, labels shuffled (rho {rho:.3f}, mad {mad:.4f})", ok, False)
    readout_fails = _readout_reds()
    saved_ticks = TICKS
    globals()["TICKS"] = 60
    try:
        learn, frozen = stats(run_arm(0, True))["d1"], stats(run_arm(0, False))["d1"]
    finally:
        globals()["TICKS"] = saved_ticks
    expect(
        f"E-PC: PC learning drifts, frozen twin does not (learn {learn:+.4f}, frozen {frozen:+.4f})",
        learn > 0.01 and abs(frozen) < 0.01,
        True,
    )
    expect(f"readout half: six reds fire ({readout_fails or 'all'})", readout_fails, [])
    if failures:
        return 1
    print(
        "representational_drift self-check OK: floor, rowcorr == corrcoef, SE triggers clear, aa / dose / "
        "mov_vs_mov (200 and 65) / agreement each green and its constructed red fires; readout half: "
        "A/A 0, rotation recovered at 4 doses, null 1 / row 0, 0/180 wrap, interpolating decoder "
        "refused, shear not recovered"
    )
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["engine-pc"]:
        return engine_pc_main(int(argv[2]) if argv[1:2] == ["--scale"] else 1)
    if argv == ["--pin-ref"]:
        if REF_PATH.exists():
            print(f"refused: {REF_PATH} exists; a pin is never overwritten")
            return 1
        REF_PATH.write_text(json.dumps(reference(), indent=1) + "\n", encoding="utf-8")
        print(f"pinned {REF_PATH}")
        return 0
    return _selfcheck()


if __name__ == "__main__":
    sys.exit(main())
