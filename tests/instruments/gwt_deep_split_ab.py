"""A/B instrument for the gwt_deep MERGE surgery (2026-08-11): the CREDITED binding
benchmark, in the exact form that retired the workspace bind_* family and then credited
gwt_deep's binding half on 2026-08-07 (ledger row / CHANGELOG at 66b6168a) — Jaccard,
120 trials/K, seed 0, all-fired-pairs strawman, rate control, baseline-relative margin.

Run BEFORE the surgery: must reproduce the credited numbers (validates this harness).
Run AFTER: the numbers must be UNCHANGED (the surgery touched ignition/re-entry, never
binding). Tolerates both the pre-surgery (ignited, raster) and post-surgery (raster)
return shapes so the same instrument runs on both sides of the diff.

  PYTHONPATH=src python tests/instruments/gwt_deep_split_ab.py
(Lived at docs/coverage/artifacts/ until 2026-08-11; moved because a docs .py is a build the
p4c ratchet counts, while tests/** is the instrument category it excludes.)
"""

import random
import statistics
import sys

from snn_data_lake.faculties.gwt_deep import (
    _objects,
    rate_broadcast,
    rate_decode,
    spiking_broadcast,
    spiking_decode,
)

TRIALS = 120


def _raster(out):
    return out[1] if isinstance(out, tuple) else out


def _acts(out):
    return out[1:] if isinstance(out, tuple) and len(out) == 3 else out


def _jaccard(t, g):
    t, g = set(t), set(g)
    if not t and not g:
        return 1.0
    u = t | g
    return len(t & g) / len(u) if u else 0.0


def main():
    n_color = n_shape = 8
    print("gwt_deep binding benchmark (credited form: Jaccard, 120 trials/K, seed 0)")
    print(f"{'K':>3} | {'spiking':>8} | {'strawman':>8} | {'rate':>8} | {'rate sd':>8}")
    ok = True
    for k in (1, 2, 4):
        rng = random.Random(0)
        sp = st = rt = []
        sp, st, rt = [], [], []
        for _ in range(TRIALS):
            objs = _objects(n_color, n_shape, k, rng)
            r = _raster(spiking_broadcast(objs, n_color, n_shape, drive=1.0))
            got = spiking_decode(r)
            sp.append(_jaccard(objs, got))
            fired_c = sorted({c for fc, _ in r for c in fc})
            fired_s = sorted({s for _, fs in r for s in fs})
            straw = {(c, s) for c in fired_c for s in fired_s}
            st.append(_jaccard(objs, straw))
            ca, sa = _acts(rate_broadcast(objs, n_color, n_shape, drive=1.0))
            rt.append(_jaccard(objs, rate_decode(ca, sa, k, rng)))
        m_sp = sum(sp) / TRIALS
        m_st = sum(st) / TRIALS
        m_rt = sum(rt) / TRIALS
        sd_rt = statistics.pstdev(rt)
        print(f"{k:>3} | {m_sp:>8.4f} | {m_st:>8.4f} | {m_rt:>8.4f} | {sd_rt:>8.4f}")
        if k >= 2 and not (m_sp > m_st):
            print(f"  FAIL: spiking must beat the all-fired-pairs strawman at K={k}")
            ok = False
        if m_sp < 0.999:
            print(f"  FAIL: spiking Jaccard must stay 1.0000 at K={k} (credited figure)")
            ok = False
    print("A/B " + ("OK: credited binding figures reproduced" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
