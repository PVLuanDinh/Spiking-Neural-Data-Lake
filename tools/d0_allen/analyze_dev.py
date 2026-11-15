"""DEV reading (4 mice, descriptive only): does the D0 instrument reproduce the published directions on Allen V1
Natural Movie One, with and without the behavioural control? Uses snn_data_lake.instruments.representational_drift."""
import json
import sys
from datetime import date
from itertools import combinations

import numpy as np

from snn_data_lake.instruments import representational_drift as rd

split = json.load(open("split.json"))


def load(exp_id):
    z = np.load(f"dev/{exp_id}.npz")
    return {k: z[k] for k in z.files}


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


rows = []
for c in split["dev"]:
    sess = {s[0][-1] if s[0] != "three_session_C2" else "C": (s[1], date.fromisoformat(s[2])) for s in c["sessions"]}
    data = {k: load(v[0]) for k, v in sess.items()}
    common = set(data["A"]["cells"])
    for k in "BC":
        common &= set(data[k]["cells"])
    common = sorted(common)
    for mode in ("raw", "run", "run+pupil"):
        prof, half = {}, {}
        nbeh = {}
        for k, d in data.items():
            idx = [list(d["cells"]).index(x) for x in common]
            resp = d["resp"][idx]
            if mode != "raw":
                resp, nbeh[k] = residualize(resp, d["run"], d["pupil"], mode == "run+pupil")
            prof[k] = resp.mean(axis=1)  # (cells, segments)
            half[k] = (resp[:, 0::2].mean(axis=1), resp[:, 1::2].mean(axis=1))
        within = np.mean([rd.pv_correlation(*half[k]) for k in "ABC"])
        for a, b in combinations("ABC", 2):
            days = abs((sess[b][1] - sess[a][1]).days)
            between_half = np.mean([rd.pv_correlation(half[a][0], half[b][1]), rd.pv_correlation(half[a][1], half[b][0])])
            tun = rd.tuning_correlation(prof[a], prof[b])
            geo = float(np.corrcoef(upper(rsa(prof[a])), upper(rsa(prof[b])))[0, 1])
            rows.append({"container": c["container"], "cre": c["cre"], "mode": mode, "pair": a + b, "days": days,
                         "cells": len(common), "within_half_pv": round(float(within), 4),
                         "between_half_pv": round(float(between_half), 4),
                         "pv_full": round(rd.pv_correlation(prof[a], prof[b]), 4),
                         "tuning_median": round(float(np.nanmedian(tun)), 4), "rsa_geometry": round(geo, 4),
                         "beh_samples": {k: nbeh.get(k) for k in "ABC"} if mode != "raw" else None})

json.dump(rows, open("dev_reading.json", "w"), indent=1)
for mode in ("raw", "run", "run+pupil"):
    print(f"== {mode}")
    print("container  cre         pair days cells  within  between  pv_full  tuning_med  rsa_geo")
    for r in rows:
        if r["mode"] == mode:
            print(f"{r['container']} {r['cre'][:10]:10s} {r['pair']}   {r['days']:>2}  {r['cells']:>4}  {r['within_half_pv']:.3f}  "
                  f"{r['between_half_pv']:.3f}    {r['pv_full']:.3f}   {r['tuning_median']:.3f}      {r['rsa_geometry']:.3f}")
