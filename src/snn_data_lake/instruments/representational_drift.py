"""MAP D0: the representational-drift instrument -- pure measurement, validated by constructed reds.

research/representational-drift/MAP.md section 4, D0 (on main since int3), and the owner's HW5 plan
(Luan Dinh, "Research idea refinement", 2026-11-02): measure how a population's responses to the SAME
stimuli change between two sessions, whether that change avoids the directions a linear readout
uses, and how much of a frozen readout a re-alignment of the population space rescues. The same
functions read mouse data (Part A) and, later, the engine's own populations (Part B), so the two
are judged by one instrument.

What it measures, each from arrays the caller supplies (it opens no lake and builds no brain):
  pv_correlation / drift_index   population-vector correlation per stimulus, averaged (A/A = 1, 0)
  tuning_correlation             per-neuron correlation of the response profile across stimuli
  preferred_shift                circular difference of preferred orientation (0/180 wrap)
  readout_null_split             the fraction of a population change in null(W) of a readout W
  random_readout_null_fractions  the same fraction for random readouts of matched rank (the control)
  decoder_accuracy               the closed-form readout, CAPACITY-GUARDED (refuses to interpolate)
  procrustes_realign / recovered the best orthogonal re-alignment of one session onto another
  read_drift -> DriftReading     the above on a two-session labelled pair, the HW5 aims in one call

Constructed reds (`main()` runs them; tests/test_representational_drift.py pins them): A/A reads
exactly zero; a planted rotation is recovered at graded doses; null-confined drift reads 1 and
row-space drift 0, with a matched-rank random control near (n - rank) / n; a 0/180-wrap orientation
shift reads its true size; an over-parameterised decoder is refused; a non-orthogonal drift is not
"recovered". A reading this instrument cannot reproduce on constructed data is never read on data.

  python -m snn_data_lake.instruments.representational_drift     # the constructed reds
"""

# distinct-from: infra/drift_monitor.py -- PSI of the INPUT feature distribution against an enrolled
#   reference (a rig/lighting shift detector); it never compares a population's responses to the same
#   stimuli across sessions. Site read 2026-11-02.
# distinct-from: instruments/prior_drift_harness.py -- WEIGHT drift off the H01 prior (DriftRecord); this
#   reads REPRESENTATIONS. Its closed-form readout is REUSED here (fit_readout / predict), not copied (P2).
# distinct-from: evidence/d1-readout-staleness.md's proposed instruments/readout_staleness.py (L1b-STALE,
#   T121, parked, not built) -- that row owns L1b's staleness control in the brain; this module builds no
#   L1b arm, reads no L1b state and is not that instrument.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: P1, a NUMERICAL tolerance (not a behavioural bar): an orthogonal planted drift is undone by
#: orthogonal Procrustes to float64 round-off (~1e-14 relative on the reds' well-conditioned data);
#: 1e-8 leaves six orders of headroom, and any non-orthogonal component above it is "not recovered".
RECOVERY_TOL = 1e-8


class CapacityRefusal(ValueError):
    """A decoder with at least as many free parameters per class as training samples interpolates any
    labelling (T063's over-parameterised readout read 1.0000 on every wiring), so it is refused."""


def _corr(x, y):
    """Pearson correlation of two equal-length vectors. Written as num / sqrt(sxx * syy) so that
    identical inputs read exactly 1.0: sxx == syy == num, and sqrt(fl(num * num)) == num in IEEE."""
    xc = np.asarray(x, dtype=np.float64) - np.mean(x)
    yc = np.asarray(y, dtype=np.float64) - np.mean(y)
    sxx, syy = float(xc @ xc), float(yc @ yc)
    if sxx == 0.0 or syy == 0.0:
        raise ValueError("a constant vector has no correlation")
    return float(xc @ yc) / math.sqrt(sxx * syy)


def pv_correlation(a, b) -> float:
    """Mean over stimuli (columns) of the correlation between the two sessions' population vectors.
    `a`, `b`: (n_neurons, n_stimuli) trial-averaged responses of the SAME neurons."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"sessions differ in shape: {a.shape} vs {b.shape}")
    return float(np.mean([_corr(a[:, j], b[:, j]) for j in range(a.shape[1])]))


def drift_index(a, b) -> float:
    """1 - pv_correlation: 0 for identical sessions, larger as the population code moves."""
    return 1.0 - pv_correlation(a, b)


def tuning_correlation(a, b):
    """Per neuron (row), the correlation of its response profile across stimuli between sessions."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"sessions differ in shape: {a.shape} vs {b.shape}")
    return np.array([_corr(a[i], b[i]) for i in range(a.shape[0])])


def preferred_shift(pref_a, pref_b, period=180.0) -> float:
    """Signed circular change of a preferred orientation from `pref_a` to `pref_b` (degrees), in
    [-period/2, period/2): 170 -> 10 reads +20 on the 180-degree orientation circle, never -160."""
    half = period / 2.0
    return float((pref_b - pref_a + half) % period - half)


def readout_null_split(delta, w) -> float:
    """Fraction of the squared norm of a population change `delta` ((n,) or (n, m)) that lies in the
    null space of a readout `w` (k, n). 1.0 = invisible to the readout, 0.0 = entirely in its row space."""
    d = np.asarray(delta, dtype=np.float64)
    d = d.reshape(d.shape[0], -1)
    total = float(np.sum(d * d))
    if total == 0.0:
        raise ValueError("a zero change has no null fraction")
    u, s, vt = np.linalg.svd(np.asarray(w, dtype=np.float64), full_matrices=False)
    rank = int(np.sum(s > s.max() * max(np.asarray(w).shape) * np.finfo(np.float64).eps))
    q = vt[:rank]
    row = q @ d
    return 1.0 - float(np.sum(row * row)) / total


def random_readout_null_fractions(delta, rank, n_draws, seed):
    """The control that is not the method: `readout_null_split` for `n_draws` Gaussian readouts of the
    same rank. For isotropic change its mean is about (n - rank) / n."""
    rng = np.random.default_rng(seed)
    n = np.asarray(delta).shape[0]
    return np.array(
        [readout_null_split(delta, rng.normal(size=(rank, n))) for _ in range(int(n_draws))]
    )


def decoder_accuracy(x_train, y_train, x_test, y_test, k) -> float:
    """Train the closed-form readout (prior_drift_harness.fit_readout: lstsq, DIRECT_SOLVE) on
    (x_train, y_train) and return its accuracy on (x_test, y_test). Refuses an interpolating fit."""
    from snn_data_lake.instruments.prior_drift_harness import fit_readout, predict

    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.shape[1] + 1 > x_train.shape[0]:
        raise CapacityRefusal(
            f"{x_train.shape[1]} features + bias > {x_train.shape[0]} training samples: the readout "
            "would interpolate any labelling"
        )
    coef = fit_readout(x_train, y_train, k)
    hits = [predict(coef, row, k) == int(label) for row, label in zip(x_test, y_test, strict=True)]
    return sum(hits) / len(hits)


def procrustes_realign(source, target):
    """The orthogonal R minimising ||source @ R - target|| (rows = samples, columns = neurons), and the
    relative residual ||source @ R - target|| / ||target||."""
    from scipy.linalg import orthogonal_procrustes

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    r, _ = orthogonal_procrustes(source, target)
    rel = float(np.linalg.norm(source @ r - target) / np.linalg.norm(target))
    return r, rel


def recovered(rel_residual, tol=RECOVERY_TOL) -> bool:
    """True iff an orthogonal re-alignment undid the change to numerical precision."""
    return rel_residual <= tol


@dataclass(frozen=True)
class DriftReading:
    """One two-session reading. Deliberately not `DriftRecord` (prior_drift_harness's weight record)."""

    drift_index: float
    null_fraction: float | None
    null_fraction_random: float | None
    frozen_acc_day0: float
    frozen_acc_day1: float
    realigned_acc_day1: float


def _class_means(x, y, k):
    return np.stack([x[y == c].mean(axis=0) for c in range(k)])


def read_drift(x0, y0, x1, y1, k, null_draws=200, seed=0) -> DriftReading:
    """The HW5 aims on one labelled pair of sessions (rows = trials, columns = the SAME neurons).

    Trials split by parity: even trials train and calibrate, odd trials are judged. The drift index
    is on the class-mean (neuron x class) profiles. The frozen readout is trained on day 0's even
    trials and judged on each day's odd trials. The re-alignment is fitted ONLY on class means of the
    even trials (a small labelled calibration, never the judged trials) and applied to day 1's odd
    trials. The null fraction splits the class-mean change against the frozen readout's weights,
    beside the matched-rank random control."""
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
        w = fit_readout(x0[tr0], y0[tr0], k)[
            :-1
        ].T  # (k, n): the readout's weights, bias row dropped
        null = readout_null_split(delta, w)
        rand = float(np.mean(random_readout_null_fractions(delta, w.shape[0], null_draws, seed)))
    else:
        null = rand = None
    return DriftReading(drift_index(m0.T, m1.T), null, rand, acc0, acc1, acc1r)


# ---- the constructed reds ------------------------------------------------------------------------------
def _rotation(n, dose, seed=0):
    from scipy.linalg import expm

    a = np.random.default_rng(seed).normal(size=(n, n))
    return expm(dose * (a - a.T) / np.sqrt(n))


def main() -> int:
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
    shear = np.eye(x.shape[1])
    shear[: x.shape[1] // 2] *= 1.5
    if recovered(procrustes_realign(x @ shear, x)[1]):
        fails.append("red6: a non-orthogonal drift read as recovered")
    if fails:
        print("representational_drift self-check FAILED: " + "; ".join(fails))
        return 1
    print(
        "representational_drift self-check OK: A/A reads 0; a planted rotation is recovered at 4 doses "
        "and drift grows with the dose; null-confined drift reads 1 and row-space 0; the 0/180 wrap "
        "reads its true size; an interpolating decoder is refused; a non-orthogonal drift is not "
        "recovered"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
