"""Which author-agreement rule is better? A synthetic test with a known answer (final project, owner 2026-11-15:
"test both to find the best method"). No mouse data is read.

Ground truth: the "authors' RDI" is our CORRECT estimator (mov_index) run with an independent seed, so a correct
implementation should pass and each planted bug should fail. Two rules from the D-R0 pre-registration section 2.4:
  doc rule:  rho >= max(0.9 x ceil, 0.8) and mad <= min(2 x self_mad, 0.05) and n >= 200
  D-R0 rule: the same AND ceil >= 0.9
The noise level is swept so the correct estimator's self-agreement ceiling spans the real DEV value (0.863).

  PYTHONPATH=src python research/representational-drift/agreement_rule_sim.py
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr

from snn_data_lake.instruments import representational_drift as rd

N_NEURONS, TRIALS, REPS = 300, 30, 6
SIGMAS = tuple(float(x) for x in sys.argv[1:]) or (1.0, 1.5, 2.0, 2.5)


def _halves(s1, s2, rng):
    n = s1.shape[0]
    iters = 50 if n <= 8 else 100
    idx = np.stack([rng.permutation(n) for _ in range(iters)])
    h1, h2 = idx[:, : n // 2], idx[:, n // 2 :]
    return s1[h1].mean(1), s1[h2].mean(1), s2[h1].mean(1), s2[h2].mean(1)


def _rdi(ws, bs):
    ws = 0.001 if ws <= 0 else ws
    return rd.DriftReading(float(ws), float(bs), float((ws - bs) / (ws + bs)))


def bug_same_half(s1, s2, rng):
    """Between-session pairs the SAME half (a1-b1, a2-b2) instead of opposite halves."""
    a1, a2, b1, b2 = _halves(s1, s2, rng)
    c = rd._rowcorr
    ws = np.fmax((c(a1, a2) + c(b1, b2)) / 2, 0).mean()
    bs = np.fmax((c(a1, b1) + c(a2, b2)) / 2, 0).mean()
    return _rdi(ws, bs)


def bug_no_rectify(s1, s2, rng):
    """Drops the per-iteration rectification (np.fmax(., 0))."""
    a1, a2, b1, b2 = _halves(s1, s2, rng)
    c = rd._rowcorr
    return _rdi(((c(a1, a2) + c(b1, b2)) / 2).mean(), ((c(a1, b2) + c(a2, b1)) / 2).mean())


def bug_ws_session1(s1, s2, rng):
    """Within-session CC from session 1 only (the authors average both sessions)."""
    a1, a2, b1, b2 = _halves(s1, s2, rng)
    c = rd._rowcorr
    return _rdi(np.fmax(c(a1, a2), 0).mean(), np.fmax((c(a1, b2) + c(a2, b1)) / 2, 0).mean())


def bug_diff_index(s1, s2, rng):
    """Reports ws - bs instead of the normalised (ws - bs) / (ws + bs)."""
    r = rd.stability_index(s1, s2, rng)
    return rd.DriftReading(r.cc_ws, r.cc_bs, r.cc_ws - r.cc_bs)


def mov(fn):
    def f(s1, s2, rng):
        return rd.mov_index(s1, s2, rng) if fn is None else _mov_with(fn, s1, s2, rng)

    return f


def _mov_with(fn, s1, s2, rng, pick=8, reps=10):
    rs = []
    for _ in range(reps):
        idx = rng.choice(s1.shape[0], pick, replace=False)
        rs.append(fn(s1[idx], s2[idx], rng))
    return rd.DriftReading(*(float(np.mean(col)) for col in zip(*rs)))


def bug_pick15(s1, s2, rng):
    """Subsamples 15 of 30 trials instead of the authors' 8."""
    return _mov_with(rd.stability_index, s1, s2, rng, pick=15)


IMPLS = {
    "correct": mov(None),
    "bug_same_half": mov(bug_same_half),
    "bug_no_rectify": mov(bug_no_rectify),
    "bug_ws_session1": mov(bug_ws_session1),
    "bug_diff_index": mov(bug_diff_index),
    "bug_pick15": bug_pick15,
}


def dataset(sigma, seed):
    rng = np.random.default_rng(seed)
    p1, p2 = rd._patterns(rng, N_NEURONS), rd._patterns(rng, N_NEURONS)
    amp = rng.lognormal(0.0, 0.5, N_NEURONS)[:, None]  # heterogeneous reliability, as in real neurons
    dose = rng.uniform(0.0, 1.0, N_NEURONS)[:, None]
    m2 = np.sqrt(1 - dose**2) * p1 + dose * p2
    s1 = amp[:, None] * p1[:, None, :] + sigma * rng.standard_normal((N_NEURONS, TRIALS, rd.N_FRAMES))
    s2 = amp[:, None] * m2[:, None, :] + sigma * rng.standard_normal((N_NEURONS, TRIALS, rd.N_FRAMES))
    return s1, s2


def pop(s1, s2, fn, seed):
    """(ws, bs, rdi) per neuron, one rng stream (population_rdi returns rdi only)."""
    rng = np.random.default_rng(seed)
    return np.array([tuple(fn(a, b, rng)) for a, b in zip(s1, s2)])


def _rel(x1, x2, th):
    ceil = float(spearmanr(x1, x2)[0])
    self_mad = float(np.median(np.abs(x1 - x2)))
    rho = float(spearmanr(x1, th)[0])
    mad = float(np.median(np.abs(x1 - th)))
    return rho >= max(0.9 * ceil, 0.8) and mad <= min(2 * self_mad, 0.05) and len(x1) >= 200


def verdicts(ceil, self_mad, ours, theirs):
    rho = float(spearmanr(ours, theirs)[0])
    mad = float(np.median(np.abs(ours - theirs)))
    rel = rho >= max(0.9 * ceil, 0.8) and mad <= min(2 * self_mad, 0.05) and len(ours) >= 200
    return {"doc": bool(rel), "dr0": bool(rel and ceil >= 0.9), "rho": rho, "mad": mad}


def main():
    rows = []
    for sigma in SIGMAS:
        for rep in range(REPS):
            s1, s2 = dataset(sigma, [7, rep, int(sigma * 10)])
            th = pop(s1, s2, IMPLS["correct"], [99, rep])
            for name, fn in IMPLS.items():
                a, b = pop(s1, s2, fn, [1, rep]), pop(s1, s2, fn, [2, rep])
                r1, r2, theirs = a[:, 2], b[:, 2], th[:, 2]
                ceil = float(spearmanr(r1, r2)[0])
                self_mad = float(np.median(np.abs(r1 - r2)))
                v = verdicts(ceil, self_mad, r1, theirs)
                v["comp"] = bool(v["doc"] and _rel(a[:, 0], b[:, 0], th[:, 0]) and _rel(a[:, 1], b[:, 1], th[:, 1]))
                rows.append({"sigma": sigma, "rep": rep, "impl": name, "ceil": ceil, **v})
                print(f"sigma {sigma} rep {rep} {name:16s} ceil {ceil:.3f} rho {v['rho']:.3f} mad {v['mad']:.4f} "
                      f"doc {v['doc']} dr0 {v['dr0']} comp {v['comp']}", flush=True)
    summary = {}
    for sigma in SIGMAS:
        for name in IMPLS:
            sel = [r for r in rows if r["sigma"] == sigma and r["impl"] == name]
            summary[f"{sigma}|{name}"] = {
                "mean_ceil": round(float(np.mean([r["ceil"] for r in sel])), 3),
                "doc_pass": sum(r["doc"] for r in sel),
                "dr0_pass": sum(r["dr0"] for r in sel),
                "comp_pass": sum(r["comp"] for r in sel),
                "n": len(sel),
            }
    out = {"rows": rows, "summary": summary}
    path = f"research/representational-drift/agreement_rule_sim_{'_'.join(map(str, SIGMAS))}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nsigma | impl | mean ceil | doc pass | D-R0 pass")
    for k, v in summary.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
