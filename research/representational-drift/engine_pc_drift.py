"""E-PC: representational drift in the SNDL engine's live predictive-coding layer (ENGINE-PC-DRIFT-PREREG.md).
Same configuration as engine/live_brain.LiveBrain.tick; same statistics as the Allen held-out analysis.

  PYTHONPATH=src python research/representational-drift/engine_pc_drift.py
"""

import copy
import dataclasses
import itertools
import json
import math
import sys

import numpy as np

from snn_data_lake.instruments import representational_drift as rd
from snn_data_lake.readout.predictive_coding import PredictiveCodingNetwork
from snn_data_lake.sensorimotor.multimodal_encoder import encode_multimodal

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
        rows.append({"pair": (a, b), "ticks": (b - a) * TICKS, "between": float(between), "geo": float(geo), "tun": tun})
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


def sign_test(v, pos=True):
    v = np.asarray(v)
    v = v[v != 0]
    k = int(np.sum(v > 0 if pos else v < 0))
    n = v.size
    return k, n, sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def main():
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
        "E-H1 d1(learn)-d1(frozen)>0": sign_test([r["learn"]["d1"] - r["frozen"]["d1"] for r in per]),
        "E-H2 d2(learn)>0": sign_test([r["learn"]["d2"] for r in per]),
        "E-H3 d3(learn)<0": sign_test([r["learn"]["d3"] for r in per], pos=False),
        "RED d1(frozen)>0 (must NOT be met)": sign_test([r["frozen"]["d1"] for r in per]),
    }
    for k, (hits, n, p) in tests.items():
        print(f"{k}: {hits}/{n} p={p:.4g} {'MET' if hits >= 9 and n == 10 else 'NOT MET'}")
    for key in ("d1", "d2", "d3"):
        for arm in ("learn", "frozen"):
            v = [r[arm][key] for r in per]
            print(f"{arm} {key}: mean {np.mean(v):+.4g} sd {np.std(v, ddof=1):.3g}")
    for k in ("frozen_acc_day0", "frozen_acc_day1", "realigned_acc_day1", "null_fraction", "null_fraction_random"):
        for arm in ("learn", "frozen"):
            v = [r[arm + "_readout"][k] for r in per if r[arm + "_readout"][k] is not None]
            print(f"{arm} readout {k}: mean {np.mean(v):.4f}" if v else f"{arm} readout {k}: n/a")
    out = {"per_seed": per, "tests": {k: list(v) for k, v in tests.items()}}
    with open("research/representational-drift/engine_pc_drift_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
