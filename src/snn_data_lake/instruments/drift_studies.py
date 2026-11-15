r"""Two drift studies for the final project, as ONE governed module (MOVED 2026-11-15, P4c, from
research/representational-drift/{agreement_rule_sim,engine_pc_drift}.py; results stay in that folder).

  rule-sim [sigma ...]  which author-agreement rule is better: a synthetic test with a known answer and five planted
                        bugs (owner 2026-11-15 "test both to find the best method"); D-R0 amendment 13.
  engine-pc             E-PC: drift in the engine's live predictive-coding layer, LEARN vs a frozen twin, 10 seeds
                        (research/representational-drift/ENGINE-PC-DRIFT-PREREG.md).
  (no argument)         the self-check: small versions of both, each with its constructed red.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.drift_studies [rule-sim 2.0 3.0 3.5 | engine-pc]
"""

# distinct-from: instruments/representational_drift.py -- the metrics; this module only RUNS two studies on them.
# distinct-from: instruments/allen_drift.py, instruments/d_r0_part_a.py -- mouse-data drivers; these read no data.

import copy
import dataclasses
import itertools
import json
import math
import sys

import numpy as np
from scipy.stats import spearmanr

from snn_data_lake.instruments import representational_drift as rd
from snn_data_lake.readout.predictive_coding import PredictiveCodingNetwork
from snn_data_lake.sensorimotor.multimodal_encoder import encode_multimodal

# ---- rule-sim ---------------------------------------------------------------------------------------------
N_NEURONS, TRIALS, REPS = 300, 30, 6
SIGMAS = (1.0, 1.5, 2.0, 2.5)


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
    amp = rng.lognormal(0.0, 0.5, N_NEURONS)[
        :, None
    ]  # heterogeneous reliability, as in real neurons
    dose = rng.uniform(0.0, 1.0, N_NEURONS)[:, None]
    m2 = np.sqrt(1 - dose**2) * p1 + dose * p2
    s1 = amp[:, None] * p1[:, None, :] + sigma * rng.standard_normal(
        (N_NEURONS, TRIALS, rd.N_FRAMES)
    )
    s2 = amp[:, None] * m2[:, None, :] + sigma * rng.standard_normal(
        (N_NEURONS, TRIALS, rd.N_FRAMES)
    )
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


def rule_sim_main(sigmas=SIGMAS):
    rows = []
    for sigma in sigmas:
        for rep in range(REPS):
            s1, s2 = dataset(sigma, [7, rep, int(sigma * 10)])
            th = pop(s1, s2, IMPLS["correct"], [99, rep])
            for name, fn in IMPLS.items():
                a, b = pop(s1, s2, fn, [1, rep]), pop(s1, s2, fn, [2, rep])
                r1, r2, theirs = a[:, 2], b[:, 2], th[:, 2]
                ceil = float(spearmanr(r1, r2)[0])
                self_mad = float(np.median(np.abs(r1 - r2)))
                v = verdicts(ceil, self_mad, r1, theirs)
                v["comp"] = bool(
                    v["doc"]
                    and _rel(a[:, 0], b[:, 0], th[:, 0])
                    and _rel(a[:, 1], b[:, 1], th[:, 1])
                )
                rows.append({"sigma": sigma, "rep": rep, "impl": name, "ceil": ceil, **v})
                print(
                    f"sigma {sigma} rep {rep} {name:16s} ceil {ceil:.3f} rho {v['rho']:.3f} mad {v['mad']:.4f} "
                    f"doc {v['doc']} dr0 {v['dr0']} comp {v['comp']}",
                    flush=True,
                )
    summary = {}
    for sigma in sigmas:
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
    path = f"research/representational-drift/agreement_rule_sim_{'_'.join(map(str, sigmas))}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nsigma | impl | mean ceil | doc pass | D-R0 pass")
    for k, v in summary.items():
        print(k, v)
    return 0


# ---- engine-pc -------------------------------------------------------------------------------------------
SEEDS = range(10)
N_PROBES, REPEATS, SESSIONS, TICKS = 30, 10, 4, 300
NOISE = 0.05


def raw_context(rng):
    return {
        "visual": rng.uniform(0, 1, 8),
        "temperature": rng.uniform(0, 1, 1),
        "audio": rng.uniform(0, 1, 8),
        "distance": rng.uniform(0, 1, 1),
    }


def encode(ctx, rng=None):
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
    out = np.zeros((24, REPEATS, len(probes)))
    for r in range(REPEATS):
        for j, p in enumerate(probes):
            out[:, r, j] = settle(probe_net, encode(p, rng))
    return out


def run_arm(seed, learn):
    rng_p = np.random.default_rng([seed, 1])
    probes = [raw_context(rng_p) for _ in range(N_PROBES)]
    x0 = encode(probes[0])
    net = PredictiveCodingNetwork(dims=(len(x0), 16, 8), seed=seed)
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
    within = float(np.mean([rd.pv_correlation(*h) for h in half]))
    rows = []
    for a, b in itertools.combinations(range(len(sessions)), 2):
        between = np.mean(
            [rd.pv_correlation(half[a][0], half[b][1]), rd.pv_correlation(half[a][1], half[b][0])]
        )
        iu = np.triu_indices(len(prof[a][0]), 1)
        geo = np.corrcoef(np.corrcoef(prof[a].T)[iu], np.corrcoef(prof[b].T)[iu])[0, 1]
        tun = float(np.nanmedian(rd.tuning_correlation(prof[a], prof[b])))
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
    return dataclasses.asdict(rd.read_drift(x0, y0, x1, y1, k=N_PROBES))


def seed_sign_test(v, pos=True):
    v = np.asarray(v)
    v = v[v != 0]
    k = int(np.sum(v > 0 if pos else v < 0))
    n = v.size
    return k, n, sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def engine_pc_main():
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
        "research/representational-drift/engine_pc_drift_result.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(out, fh, indent=1, default=str)
    return 0


def _selfcheck():
    """Small versions of both studies, each with its red: correct code passes the components rule and the
    15-trial bug fails it; live PC learning drifts (d1 > 0) while the frozen twin does not."""
    global N_NEURONS, TICKS
    fails = []
    n_saved, t_saved = N_NEURONS, TICKS
    N_NEURONS, TICKS = 240, 60
    try:
        s1, s2 = dataset(2.0, [7, 0, 20])
        th = pop(s1, s2, IMPLS["correct"], [99, 0])
        for name, want in (("correct", True), ("bug_pick15", False)):
            a, b = pop(s1, s2, IMPLS[name], [1, 0]), pop(s1, s2, IMPLS[name], [2, 0])
            comp = all(_rel(a[:, j], b[:, j], th[:, j]) for j in range(3))
            print(
                ("ok   " if comp == want else "FAIL ")
                + f"rule-sim components rule on {name}: {comp}"
            )
            if comp != want:
                fails.append(f"rule-sim {name}")
        learn, frozen = stats(run_arm(0, True))["d1"], stats(run_arm(0, False))["d1"]
        ok = learn > 0.01 and abs(frozen) < 0.01
        print(
            ("ok   " if ok else "FAIL ")
            + f"engine-pc: learn d1 {learn:+.4f}, frozen d1 {frozen:+.4f}"
        )
        if not ok:
            fails.append("engine-pc")
    finally:
        N_NEURONS, TICKS = n_saved, t_saved
    if fails:
        print("drift_studies self-check FAILED: " + "; ".join(fails))
        return 1
    print(
        "drift_studies self-check OK: components rule passes correct code and catches the 15-trial bug; "
        "PC learning drifts, its frozen twin does not"
    )
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _selfcheck()
    if argv[0] == "rule-sim":
        return rule_sim_main(tuple(float(x) for x in argv[1:]) or SIGMAS)
    if argv[0] == "engine-pc":
        return engine_pc_main()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
