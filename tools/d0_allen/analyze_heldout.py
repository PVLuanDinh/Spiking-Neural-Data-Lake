"""D0 Part A held-out analysis (prereg d0-partA-heldout-prereg.md sections 3-5, owner decisions section 10):
per-unit d1-d4 on Allen V1 Natural Movie One, exact one-sided sign tests (Bonferroni 0.0125), bootstrap CIs,
and the pre-read checks R-SIGN, R-B, R-C, R-D, R-E. Reuses residualize/rsa/upper from the frozen analyze_dev.py.

  python analyze_heldout.py checks [--after-freeze]   # pre-read checks; exit 1 on any FAIL
  python analyze_heldout.py dev                       # DEV per-unit table (descriptive, no test)
  python analyze_heldout.py sealed                    # refused unless tools/d0_allen/FROZEN exists
"""

import ast
import hashlib
import itertools
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from snn_data_lake.instruments import representational_drift as rd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "drift" / "allen"
ALPHA = 0.05 / 4
PAIRS = ("AB", "AC", "BC")
MODES = ("raw", "run+pupil", "run")
MIN_COMPLETE = 150  # prereg 3: run-only fallback below half the 300 samples
HYP = (("H1", "d1", "pos"), ("H2", "d2", "pos"), ("H3", "d3", "neg"), ("H4", "d4", "pos"))


def _frozen_functions():
    """residualize, rsa, upper from the frozen analyze_dev.py, verified by sha256, without running its loop."""
    pin = json.loads(
        (ROOT / "tools" / "d0_partA_heldout_units_pin.json").read_text(encoding="utf-8")
    )
    src = (HERE / "analyze_dev.py").read_bytes()
    if hashlib.sha256(src).hexdigest() != pin["frozen_pipeline_sha256"]["analyze_dev.py"]:
        raise SystemExit("refused: analyze_dev.py differs from its pinned sha256")
    tree = ast.parse(src.decode("utf-8"))
    keep = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("residualize", "rsa", "upper")
    ]
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "analyze_dev.py", "exec"), ns)
    return ns["residualize"], ns["rsa"], ns["upper"]


residualize, rsa, upper = _frozen_functions()


def _pins(split):
    if split == "dev":
        return json.loads((ROOT / "tools" / "d0_partA_split_pin.json").read_text(encoding="utf-8"))[
            "dev"
        ]
    return json.loads(
        (ROOT / "tools" / "d0_partA_heldout_units_pin.json").read_text(encoding="utf-8")
    )["units"]


def _key(session_type):
    return "C" if session_type.endswith("C2") else session_type[-1]


def load_unit(unit, split):
    """{A,B,C: npz dict} restricted to cells common to all three sessions, and {A,B,C: date}."""
    data, dates = {}, {}
    for st, exp_id, d in unit["sessions"]:
        z = np.load(DATA / split / f"{exp_id}.npz")
        data[_key(st)] = {k: z[k] for k in ("resp", "cells", "run", "pupil")}
        dates[_key(st)] = date.fromisoformat(d)
    common = sorted(set.intersection(*(set(v["cells"].tolist()) for v in data.values())))
    for v in data.values():
        pos = {c: i for i, c in enumerate(v["cells"].tolist())}
        v["resp"] = v["resp"][[pos[c] for c in common]].astype(np.float64)
    return data, dates


def _clean(v, mode):
    """Response for one session under `mode`; returns (resp, fell_back_to_run_only)."""
    if mode == "raw":
        return v["resp"], False
    if mode == "run":
        return residualize(v["resp"], v["run"], v["pupil"], False)[0], False
    complete = int(np.sum(np.isfinite(v["run"]) & np.isfinite(v["pupil"])))
    use_pupil = complete >= MIN_COMPLETE
    return residualize(v["resp"], v["run"], v["pupil"], use_pupil)[0], not use_pupil


def mode_reading(data, dates, mode):
    """within, and per pair: between, median tuning r, RSA geometry r, days. Plus fallback count."""
    prof, half, fb = {}, {}, 0
    for k, v in data.items():
        resp, fell = _clean(v, mode)
        fb += fell
        prof[k] = resp.mean(axis=1)
        half[k] = (resp[:, 0::2].mean(axis=1), resp[:, 1::2].mean(axis=1))
    within = float(np.mean([rd.pv_correlation(*half[k]) for k in "ABC"]))
    pairs = {}
    for a, b in itertools.combinations("ABC", 2):
        between = np.mean(
            [rd.pv_correlation(half[a][0], half[b][1]), rd.pv_correlation(half[a][1], half[b][0])]
        )
        geo = np.corrcoef(upper(rsa(prof[a])), upper(rsa(prof[b])))[0, 1]
        pairs[a + b] = {
            "between": float(between),
            "tuning": float(np.nanmedian(rd.tuning_correlation(prof[a], prof[b]))),
            "geo": float(geo),
            "days": abs((dates[b] - dates[a]).days),
        }
    return {"within": within, "pairs": pairs, "fallbacks": fb}


def stats_from(data, dates):
    m = {mode: mode_reading(data, dates, mode) for mode in MODES}
    raw = m["raw"]["pairs"]
    days = [raw[p]["days"] for p in PAIRS]
    btw = [raw[p]["between"] for p in PAIRS]
    d3 = float("nan") if len(set(days)) == 1 else float(np.polyfit(days, btw, 1)[0])
    return {
        "d1": m["raw"]["within"] - float(np.mean(btw)),
        "d2": float(np.mean([raw[p]["geo"] - raw[p]["tuning"] for p in PAIRS])),
        "d3": d3,
        "d4": float(
            np.mean([m["run+pupil"]["pairs"][p]["between"] - raw[p]["between"] for p in PAIRS])
        ),
        "d4_run": float(
            np.mean([m["run"]["pairs"][p]["between"] - raw[p]["between"] for p in PAIRS])
        ),
        "fallbacks": m["run+pupil"]["fallbacks"],
        "modes": m,
    }


def unit_stats(unit, split):
    data, dates = load_unit(unit, split)
    out = stats_from(data, dates)
    out.update(
        container=unit["container"], cre=unit["cre"], n_cells=int(data["A"]["resp"].shape[0])
    )
    return out


# ---- the test --------------------------------------------------------------------------------------------
def binom_tail(n, k):
    """Exact P[Binomial(n, 1/2) >= k]."""
    return sum(math.comb(n, i) for i in range(max(k, 0), n + 1)) / 2**n


def crit(n):
    return next((k for k in range(n + 1) if binom_tail(n, k) <= ALPHA), n + 1)


def sign_test(values, predicted):
    v = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(v) & (v != 0)
    v = v[keep]
    k = int(np.sum(v > 0 if predicted == "pos" else v < 0))
    return k, int(v.size), binom_tail(int(v.size), k), int((~keep).sum())


def boot_ci(values, n=10000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    means = np.random.default_rng(seed).choice(v, (n, v.size)).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hypothesis_table(rows):
    out = {}
    for h, stat, pred in HYP:
        vals = [r[stat] for r in rows]
        k, n, p, dropped = sign_test(vals, pred)
        fin = np.asarray(vals, dtype=np.float64)
        fin = fin[np.isfinite(fin)]
        met = k >= crit(n)
        out[h] = {
            "stat": stat,
            "predicted": pred,
            "k": k,
            "n": n,
            "crit": crit(n),
            "p": p,
            "dropped": dropped,
            "median": float(np.median(fin)),
            "mean": float(np.mean(fin)),
            "ci95": boot_ci(vals),
            "verdict": "MET"
            if met
            else "NOT MET (INCONCLUSIVE for true share in (0.5, 0.75); mde p+ 0.75)",
        }
    d1 = np.nanmean([r["d1"] for r in rows])
    out["H4"]["share_d4_over_d1"] = float(np.nanmean([r["d4"] for r in rows]) / d1)
    return out


# ---- pre-read checks -------------------------------------------------------------------------------------
def r_sign():
    for n in range(1, 21):
        counts = [0] * (n + 1)
        for bits in itertools.product((0, 1), repeat=n):
            counts[sum(bits)] += 1
        for k in range(n + 2):
            brute = sum(counts[k:]) / 2**n
            if abs(brute - binom_tail(n, k)) > 1e-12:
                return False, f"n={n} k={k}"
    return True, "exact tail == enumeration, n 1..20"


def r_b():
    worst = 0.0
    for unit in _pins("dev"):
        data, _ = load_unit(unit, "dev")
        for k in "ABC":
            same = {x: data[k] for x in "ABC"}
            dates = {x: date(2020, 1, 1) + timedelta(days=i + 1) for i, x in enumerate("ABC")}
            worst = max(worst, abs(stats_from(same, dates)["d1"]))
    return worst <= 1e-12, f"max |d1| on 12 self-paired pseudo-units = {worst:.2e}"


def r_c():
    t = np.random.default_rng(1).normal(size=(60, 30))
    g = np.random.default_rng(2).normal(size=60)
    data, dates = {}, {}
    for j, k in enumerate("ABC"):
        run = np.random.default_rng(10 + j).normal(size=30)[None, :] * 2.0
        run = run + 0.5 * np.random.default_rng(30 + j).normal(size=(10, 30))
        run = run - run.mean()
        pupil = np.random.default_rng(20 + j).normal(size=(10, 30))
        data[k] = {"resp": t[:, None, :] + g[:, None, None] * run[None], "run": run, "pupil": pupil}
        dates[k] = date(2020, 1, 1) + timedelta(days=j)
    raw = mode_reading(data, dates, "raw")
    clean = mode_reading(data, dates, "run")
    d1_raw = raw["within"] - np.mean([raw["pairs"][p]["between"] for p in PAIRS])
    ok_pairs = all(clean["pairs"][p]["between"] >= 0.999 * clean["within"] for p in PAIRS)
    return bool(
        d1_raw > 0 and ok_pairs
    ), f"raw d1 {d1_raw:.4f}; residualized between/within >= 0.999: {ok_pairs}"


def r_d():
    res = []
    for unit in _pins("dev"):
        data, dates = load_unit(unit, "dev")
        base = mode_reading(data, dates, "raw")["pairs"]["AB"]["geo"]
        shuf = {k: dict(v) for k, v in data.items()}
        shuf["B"]["resp"] = data["B"]["resp"][:, :, np.random.default_rng(3).permutation(30)]
        res.append((base, mode_reading(shuf, dates, "raw")["pairs"]["AB"]["geo"]))
    return all(s < b for b, s in res), " ".join(f"{b:.3f}->{s:.3f}" for b, s in res)


def r_e():
    ids = {str(e) for u in _pins("sealed") for _, e, _ in u["sessions"]}
    hits = [p for p in (DATA / "sealed").glob("*")] if (DATA / "sealed").exists() else []
    for p in ROOT.rglob("*"):
        if ".git" in p.parts:
            continue
        if any(i in p.name for i in ids):
            hits.append(p)
    return not hits, f"{len(hits)} sealed-derived file(s)" + (f": {hits[:3]}" if hits else "")


def checks(after_freeze):
    todo = [("R-SIGN", r_sign), ("R-B", r_b), ("R-C", r_c), ("R-D", r_d)]
    if not after_freeze:
        todo.append(("R-E", r_e))
    fails = 0
    for name, fn in todo:
        ok, msg = fn()
        print(("ok   " if ok else "FAIL ") + f"{name}: {msg}")
        fails += not ok
    return 1 if fails else 0


def run_split(split):
    rows, missing = [], []
    for unit in _pins(split):
        try:
            rows.append(unit_stats(unit, split))
        except FileNotFoundError as e:
            missing.append({"container": unit["container"], "error": str(e)})
    for r in rows:
        print(
            f"{r['container']} {r['cre'][:10]:10s} cells {r['n_cells']:4d}  d1 {r['d1']:+.3f}  d2 {r['d2']:+.3f}  "
            f"d3 {r['d3']:+.4f}  d4 {r['d4']:+.4f}  d4_run {r['d4_run']:+.4f}  fb {r['fallbacks']}"
        )
    return rows, missing


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    if argv[0] == "checks":
        return checks("--after-freeze" in argv)
    if argv[0] == "dev":
        run_split("dev")
        return 0
    if argv[0] == "sealed":
        if not (HERE / "FROZEN").exists():
            raise SystemExit(
                "refused: sealed read before the freeze (tools/d0_allen/FROZEN absent)"
            )
        rows, missing = run_split("sealed")
        table = hypothesis_table(rows)
        for h, t in table.items():
            print(
                f"{h} {t['stat']} {t['k']}/{t['n']} (crit {t['crit']}) p={t['p']:.3g} mean {t['mean']:+.4f} "
                f"CI [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] median {t['median']:+.4f} dropped {t['dropped']}: "
                f"{t['verdict']}"
            )
        print(f"missing units (excluded, never imputed): {len(missing)}")
        out = {"units": rows, "missing": missing, "tests": table}
        (DATA / "heldout_result.json").write_text(
            json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8"
        )
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
