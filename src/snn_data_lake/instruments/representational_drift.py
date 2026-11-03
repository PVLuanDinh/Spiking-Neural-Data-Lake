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

Correlations are Pearson, vectorised across iterations (`_rowcorr`); the self-check ties it to
`numpy.corrcoef` to 1e-12. Distinct-from `faculties/grid_place_map._pearson` and `infra/drift_monitor.psi`.

Criteria (each named `*_verdict`; each has its constructed red in `_selfcheck`): aa_stability_verdict,
planted_drift_dose_verdict, mov_vs_mov_control_verdict, author_agreement_verdict. The synthetic half only:
the real-data halves belong to `tools/d_r0_partA.py`. Pure numpy + scipy.stats; deterministic; ASCII prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.representational_drift
  ... --pin-ref   (writes tools/d_r0_synthetic_ref.json; refuses to overwrite an existing pin)
"""

from __future__ import annotations

import json
import sys
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
    with np.errstate(invalid="ignore", divide="ignore"):  # constant row: NaN as in MATLAB; fmax reads 0
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


def author_agreement_verdict(ceil, self_mad, ours, theirs):
    """Pass: ceil >= 0.9, Spearman(ours, theirs) >= max(0.9 ceil, 0.8), median |ours - theirs| <= min(2 self_mad,
    0.05), at least 200 neuron-sessions."""
    rho = float(spearmanr(ours, theirs)[0])
    mad = float(np.median(np.abs(ours - theirs)))
    ok = ceil >= 0.9 and rho >= max(0.9 * ceil, 0.8) and mad <= min(2 * self_mad, 0.05) and len(ours) >= 200
    return bool(ok), rho, mad


def _agreement_inputs(fn, s1, s2):
    r1, r2, theirs = (population_rdi(s1, s2, fn, [SEED, 3, k]) for k in (1, 2, 3))
    return float(spearmanr(r1, r2)[0]), float(np.median(np.abs(r1 - r2))), r1, theirs


def _shuffle_labels(s1, s2, seed):
    """Per neuron, pool both sessions' trials and re-split them at random."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([s1, s2], axis=1)
    out = np.stack([p[rng.permutation(p.shape[0])] for p in pooled])
    return out[:, : s1.shape[1]], out[:, s1.shape[1] :]


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
    broken = lambda s1, s2, r: stability_index(s1, s1, r)  # noqa: E731 -- the red: no between-session pairing

    ok, m = aa_stability_verdict(stability_index)
    expect(f"aa green (mean {m:+.4f})", ok, True)
    ok, m = aa_stability_verdict(stability_index, dose=0.55)
    expect(f"aa red d=0.55 (mean {m:+.4f})", ok, False)
    ok, means = planted_drift_dose_verdict(stability_index, ref)
    expect(f"dose green ({[round(x, 3) for x in means]})", ok, True)
    ok, means = planted_drift_dose_verdict(broken, ref)
    expect(f"dose red, ws-for-bs ({[round(x, 3) for x in means]})", ok, False)
    for n in (200, 65):
        ok, d, bar = mov_vs_mov_control_verdict(n)
        expect(f"mov_vs_mov green n={n} (D {d:+.4f}, bar {bar:.4f})", ok, True)
        ok, d, bar = mov_vs_mov_control_verdict(n, drift_a=0.6)
        expect(f"mov_vs_mov red n={n} (D {d:+.4f}, bar {bar:.4f})", ok, False)
    s1, s2 = synth_pair(200, 0.7, SEED)
    ceil, self_mad, ours, theirs = _agreement_inputs(stability_index, s1, s2)
    ok, rho, mad = author_agreement_verdict(ceil, self_mad, ours, theirs)
    expect(f"agreement green (ceil {ceil:.3f}, rho {rho:.3f}, mad {mad:.4f}, self_mad {self_mad:.4f})", ok, True)
    sh1, sh2 = _shuffle_labels(s1, s2, SEED)
    shuf = population_rdi(sh1, sh2, stability_index, [SEED, 3, 1])
    ok, rho, mad = author_agreement_verdict(ceil, self_mad, shuf, theirs)
    expect(f"agreement red, labels shuffled (rho {rho:.3f}, mad {mad:.4f})", ok, False)
    if failures:
        return 1
    print("representational_drift self-check OK: floor, rowcorr == corrcoef, SE triggers clear, aa / dose / "
          "mov_vs_mov (200 and 65) / agreement each green and its constructed red fires")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
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
