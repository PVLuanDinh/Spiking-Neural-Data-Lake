r"""D0 Part A on Allen Brain Observatory 2-photon V1 (final project RQ1-RQ3): range-read extraction, the
held-out analysis (d1-d4, exact sign tests, bootstrap CIs) and its pre-read checks, as ONE governed module.

MOVED 2026-11-15 (P4c) from tools/d0_allen/{rangefile,extract,analyze_dev,fetch,analyze_heldout}.py. The frozen
code (sha256-pinned in tools/d0_partA_heldout_units_pin.json and tools/d0_allen/FROZEN) is kept byte-exact as
records in tools/d0_allen/frozen-code/*.py.txt; the functions marked FROZEN below are copied from those records
mechanically, and the self-check proves each one AST-identical to its record and every record's sha256 equal to
its pin. The sealed read already ran once (prereg section 12); `sealed` refuses to run again.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.allen_drift      # self-check (no network)
  ... allen_drift fetch dev|sealed     # range-read sessions into data/drift/allen/<split>/ (gitignored)
  ... allen_drift checks [--after-freeze] | dev | sealed
"""

# distinct-from: instruments/d_r0_part_a.py -- the Marks & Goard .mat driver (RQ4); this reads Allen NWB (RQ1-3).
# distinct-from: sensorimotor/organoid_ingest.py -- DANDI NWB ingest into the lake; this writes only
#   data/drift/allen/ npz arrays, never the lake (P4b), and reuses representational_drift's metrics.

import ast
import hashlib
import io
import itertools
import json
import math
import re
import sys
import urllib.request
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from snn_data_lake.instruments import representational_drift as rd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "drift" / "allen"
RECORDS = ROOT / "tools" / "d0_allen" / "frozen-code"
FREEZE_MARK = ROOT / "tools" / "d0_allen" / "FROZEN"
UNITS_PIN = ROOT / "tools" / "d0_partA_heldout_units_pin.json"
SPLIT_PIN = ROOT / "tools" / "d0_partA_split_pin.json"
BUCKET = "https://allen-brain-observatory.s3.amazonaws.com"
EYE_PREFIX = "visual-coding-2p/ophys_eye_gaze_mapping/"
BLOCK = 1 << 20  # FROZEN constant (rangefile)
SEG = 30  # FROZEN constant (extract): 30 frames = 1 s at 30 Hz
ALPHA = 0.05 / 4
PAIRS = ("AB", "AC", "BC")
MODES = ("raw", "run+pupil", "run")
MIN_COMPLETE = 150  # prereg 3: run-only fallback below half the 300 samples
HYP = (("H1", "d1", "pos"), ("H2", "d2", "pos"), ("H3", "d3", "neg"), ("H4", "d4", "pos"))
FROZEN_NAMES = {
    "rangefile.py": ("resolve", "RangeFile"),
    "extract.py": ("nwb_link", "windows", "bin_mean", "extract"),
    "analyze_dev.py": ("residualize", "rsa", "upper"),
}
h5py = (
    None  # bound lazily by _bind_h5py() (the [nir] extra); the FROZEN extract() reads this global
)


def _bind_h5py():
    global h5py
    import h5py as _h5py

    h5py = _h5py


# ---- FROZEN (copied mechanically from the records; the self-check proves them AST-identical) -----------
def resolve(url):
    """Follow redirects once (DANDI download -> S3) and return the final URL and total size."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.geturl(), int(r.headers["Content-Range"].rsplit("/", 1)[1])


class RangeFile(io.RawIOBase):
    def __init__(self, url, cache_blocks=512):
        self.url, self.size = resolve(url)
        self.pos = 0
        self.cache = OrderedDict()
        self.cache_blocks = cache_blocks
        self.fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        lo, hi = i * BLOCK, min(self.size, (i + 1) * BLOCK) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 3:
                    raise
        self.fetched += len(data)
        self.cache[i] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def readinto(self, b):
        n = min(len(b), max(0, self.size - self.pos))
        out, got = memoryview(b), 0
        while got < n:
            i, off = divmod(self.pos + got, BLOCK)
            blk = self._block(i)
            take = min(n - got, len(blk) - off)
            out[got : got + take] = blk[off : off + take]
            got += take
        self.pos += n
        return n


def nwb_link(exp_id):
    q = (
        "https://api.brain-map.org/api/v2/data/query.json?criteria=model::WellKnownFile,rma::criteria,"
        f"well_known_file_type[id%24eq514173063],[attachable_id%24eq{exp_id}]"
    )
    w = json.load(urllib.request.urlopen(q, timeout=120))["msg"]
    assert len(w) == 1, (exp_id, w)
    return "https://api.brain-map.org" + w[0]["download_link"]


def windows(stim_ts, stim_frames):
    """(repeat, segment) -> (t_start, t_end) of each 30-frame movie segment."""
    starts = np.flatnonzero(stim_frames == 0)
    out = []
    for s in starts:
        rep = []
        for k in range(900 // SEG):
            i0, i1 = s + k * SEG, s + (k + 1) * SEG - 1
            rep.append((stim_ts[i0], stim_ts[i1] + 1 / 30))
        out.append(rep)
    return np.array(out)


def bin_mean(t, x, win):
    """Mean of samples x at times t inside each window; x may be (n, T) or (T,)."""
    lo = np.searchsorted(t, win[..., 0])
    hi = np.searchsorted(t, win[..., 1])
    if x.ndim == 1:
        return np.array(
            [
                [np.nanmean(x[a:b]) if b > a else np.nan for a, b in zip(r0, r1)]
                for r0, r1 in zip(lo, hi)
            ]
        )
    return np.stack(
        [
            np.stack([x[:, a:b].mean(axis=1) for a, b in zip(r0, r1)], axis=1)
            for r0, r1 in zip(lo, hi)
        ],
        axis=1,
    )


def extract(exp_id, eye_key, out):
    f = RangeFile(nwb_link(exp_id))
    h = h5py.File(f, "r")
    p = "processing/brain_observatory_pipeline/"
    st = h["stimulus/presentation/natural_movie_one_stimulus"]
    stim_ts, stim_fr = st["timestamps"][:], st["data"][:]
    win = windows(stim_ts, stim_fr)
    t2p = h[p + "DfOverF/imaging_plane_1/timestamps"][:]
    i0 = max(0, int(np.searchsorted(t2p, win.min())) - 2)
    i1 = int(np.searchsorted(t2p, win.max())) + 2
    dff = h[p + "DfOverF/imaging_plane_1/data"][:, i0:i1]
    resp = bin_mean(t2p[i0:i1], dff, win)  # (cells, repeats, segments)
    cells = h[p + "ImageSegmentation/cell_specimen_ids"][:]
    rs_t = h[p + "BehavioralTimeSeries/running_speed/timestamps"][:]
    rs = h[p + "BehavioralTimeSeries/running_speed/data"][:]
    run = bin_mean(rs_t, rs, win)
    meta = {
        k: h["general/" + k][()].decode()
        for k in ("session_type", "targeted_structure", "experiment_container_id")
    }
    nwb_mb = f.fetched / 1e6
    e = RangeFile("https://allen-brain-observatory.s3.amazonaws.com/" + eye_key)
    eh = h5py.File(e, "r")
    et = eh["synced_frame_timestamps/values"][:]
    pa = eh["new_pupil_areas/values"][:]
    pupil = bin_mean(et, pa, win)
    np.savez_compressed(
        out, resp=resp.astype(np.float32), cells=cells, run=run, pupil=pupil, win=win, **meta
    )
    return (
        resp.shape,
        meta,
        round(nwb_mb, 1),
        round(e.fetched / 1e6, 1),
        float(np.mean(np.isnan(pupil))),
    )


def residualize(resp, run, pupil, use_pupil):
    """Remove each cell's linear dependence on running (and pupil) across repeat x segment samples, keeping the
    segment means: fit resp = segment effect + b * behaviour by least squares, return the segment-effect part."""
    c, r, s = resp.shape
    seg = np.tile(np.eye(s), (r, 1))  # (r*s, s) one-hot segment design
    beh = [run.reshape(-1)]
    if use_pupil:
        beh.append(pupil.reshape(-1))
    b = np.column_stack(beh)
    ok = np.all(np.isfinite(b), axis=1)
    b = (b - np.nanmean(b[ok], axis=0)) / (np.nanstd(b[ok], axis=0) + 1e-12)
    x = np.hstack([seg, b])[ok]
    y = resp.reshape(c, -1)[:, ok].T  # (samples, cells)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    clean = resp.copy()
    full_b = np.where(ok[:, None], b, 0.0)
    clean = (resp.reshape(c, -1).T - full_b @ coef[s:]).T.reshape(c, r, s)
    return clean, int(ok.sum())


def rsa(profile):
    """Segment x segment correlation of population vectors (the representational geometry)."""
    return np.corrcoef(profile.T)


def upper(m):
    i, j = np.triu_indices(m.shape[0], 1)
    return m[i, j]


# ---- records and fetch (fetch absorbed from tools/d0_allen/fetch.py) -----------------------------------
def check_records():
    """Every frozen record's sha256 equals its pin (units pin for the three pipeline blocks, FROZEN for the
    held-out analysis). Raises SystemExit on a mismatch."""
    want = dict(json.loads(UNITS_PIN.read_text(encoding="utf-8"))["frozen_pipeline_sha256"])
    m = re.search(
        r"analyze_heldout\.py sha256 ([0-9a-f]{64})", FREEZE_MARK.read_text(encoding="utf-8")
    )
    want["analyze_heldout.py"] = m.group(1)
    for name, sha in want.items():
        got = hashlib.sha256((RECORDS / f"{name}.txt").read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"refused: record {name} sha256 {got[:12]} != pinned {sha[:12]}")
    return want


def frozen_identical():
    """Names whose definition here is NOT AST-identical to its frozen record (empty = all identical)."""
    here = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    mine = {
        n.name: ast.dump(n) for n in here.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    bad = []
    for rec, names in FROZEN_NAMES.items():
        tree = ast.parse((RECORDS / f"{rec}.txt").read_text(encoding="utf-8"))
        theirs = {
            n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        bad += [n for n in names if mine.get(n) != theirs.get(n)]
    return bad


def fetch(split):
    """Run the FROZEN extract() for every session of `split` into data/drift/allen/<split>/; idempotent."""
    check_records()
    if split == "sealed" and not FREEZE_MARK.exists():
        raise SystemExit("refused: sealed read before the freeze (tools/d0_allen/FROZEN absent)")
    _bind_h5py()
    out = DATA / split
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for u in _pins(split):
        for _, exp_id, _ in u["sessions"]:
            dst = out / f"{exp_id}.npz"
            if dst.exists():
                continue
            try:
                print(u["container"], *extract(str(exp_id), eye_key(exp_id), str(dst)), flush=True)
            except Exception as e:  # reported, never imputed
                print(f"FAILED {u['container']} {exp_id}: {type(e).__name__}: {e}", flush=True)
                failed += 1
    return failed


def eye_key(exp_id):
    url = f"{BUCKET}/?list-type=2&prefix={EYE_PREFIX}{exp_id}_"
    xml = urllib.request.urlopen(url, timeout=120).read().decode()
    keys = [
        k
        for k in re.findall(r"<Key>([^<]+)</Key>", xml)
        if k.endswith("_eyetracking_dlc_to_screen_mapping.h5")
    ]
    if len(keys) != 1:
        raise RuntimeError(f"{exp_id}: expected 1 eye file, found {keys}")
    return keys[0]


def _pins(split):
    if split == "dev":
        return json.loads(SPLIT_PIN.read_text(encoding="utf-8"))["dev"]
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


def analyze_main(argv):
    if not argv:
        raise SystemExit(__doc__)
    if argv[0] == "checks":
        return checks("--after-freeze" in argv)
    if argv[0] == "dev":
        run_split("dev")
        return 0
    if argv[0] == "sealed":
        if not FREEZE_MARK.exists():
            raise SystemExit(
                "refused: sealed read before the freeze (tools/d0_allen/FROZEN absent)"
            )
        rows, missing = run_split("sealed")
        if not rows:  # never spend the one sealed read on an empty set (review fold)
            raise SystemExit(f"refused: no sealed unit could be scored ({len(missing)} missing)")
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


def _selfcheck():
    """No network, no mouse data: records hash to their pins, FROZEN copies are AST-identical, the sign test
    equals enumeration, planted behaviour is removed, and the prereg's critical count holds."""
    fails = []
    check_records()
    bad = frozen_identical()
    if bad:
        fails.append(f"frozen copies differ from their records: {bad}")
    for name, fn in (("R-SIGN", r_sign), ("R-C", r_c)):
        ok, msg = fn()
        print(("ok   " if ok else "FAIL ") + f"{name}: {msg}")
        if not ok:
            fails.append(name)
    if sign_test([1, 2, -1, 0, float("nan")], "pos") != (2, 3, binom_tail(3, 2), 2):
        fails.append("sign_test drops zeros/NaN wrongly")
    if crit(44) != 30:
        fails.append("crit(44) must be 30 (prereg section 4)")
    if fails:
        print("allen_drift self-check FAILED: " + "; ".join(fails))
        return 1
    print(
        "allen_drift self-check OK: records == pins, FROZEN copies AST-identical, R-SIGN, R-C, crit(44) = 30"
    )
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _selfcheck()
    if argv[0] == "fetch":
        return 1 if fetch(argv[1]) else 0  # any failed session -> non-zero (review fold)
    if argv[0] == "sealed" and (DATA / "heldout_result.json").exists():
        raise SystemExit(
            "refused: the sealed read already ran once (data/drift/allen/heldout_result.json)"
        )
    return analyze_main(argv)


if __name__ == "__main__":
    sys.exit(main())
