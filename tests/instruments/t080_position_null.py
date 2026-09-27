"""T080 (b2)'s POSITION-PERMUTATION null per seed, from the committed pin alone (the panel fold,
2026-09-16, discharging 2.11's stated consideration). THE MAPPING: the pack position of the k-th
SCORED tick of an arm is (start + k) mod 256 -- the pointers advance once per awake tick, so ~294
scored ticks walk ~1.15 laps of the 256-frame pack; the recorder's vptr / aptr columns are not
pinned, so the position CLASS is the scored-tick ordinal mod 256, never the recorded `t` (which
counts asleep ticks too). The in-vivo NULL arm is one draw of a position permutation, so this null
pairs each scored tick's vision object with the audio object of its permuted class. Prints per
seed: the class count and within-class inconsistencies (0 = the same frame reads the same object),
the position null's per-seed 99 % band and family-wise band, its sd against the pin's draws_sd,
the seed's one-sided p; then the family-wise refusal rate of a per-seed 99 % band simulated from
these nulls. Writes nothing.

  PYTHONUTF8=1 PYTHONPATH=src python tests/instruments/t080_position_null.py [pin.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

from snn_data_lake.instruments.binding_instrument import T080_BAND_FW, T080_PIN, _t080_order_stat

N_POS = 256
N_DRAWS = 4000


def _cols(recs):
    return (
        np.array([r.get("obj_v") or "" for r in recs]),
        np.array([r.get("obj_a") or "" for r in recs]),
    )


def position_null(recs, rng, n_draws=N_DRAWS):
    ov, oa = _cols(recs)
    n = len(recs)
    cls = np.arange(n) % N_POS
    aud_at, inconsistent = {}, 0
    for k in range(n):
        c = int(cls[k])
        if c in aud_at and aud_at[c] != oa[k]:
            inconsistent += 1
        aud_at.setdefault(c, oa[k])
    classes = np.array(sorted(aud_at))
    aud_vec = np.array([aud_at[c] for c in classes])
    idx = {c: i for i, c in enumerate(classes)}
    pos = np.array([idx[int(c)] for c in cls])
    draws = np.empty(n_draws)
    for i in range(n_draws):
        perm = rng.permutation(len(classes))
        draws[i] = np.mean((ov != "") & (ov == aud_vec[perm][pos]))
    return np.sort(draws), len(classes), inconsistent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pin = json.loads((Path(argv[0]) if argv else T080_PIN).read_text(encoding="utf-8"))
    print(
        f"T080 position null -- pin {pin['head'][:9]} verdict {pin['verdict']}; {N_DRAWS} draws per seed"  # noqa: E501 -- an instrument report line
    )
    print(
        "| seed | in-vivo NULL | classes / inconsistent | position 99 % band | position family-wise band | in? | sd position / pin draws_sd (ratio) | p (one-sided high) |"  # noqa: E501 -- an instrument report line
    )
    print("|---|---|---|---|---|---|---|---|")
    per_seed = []
    for s in pin["per_seed"]:
        seed = s["seed"]
        d, ncls, incons = position_null(
            s["arms"]["aligned"]["recs"], np.random.default_rng(9000 + seed)
        )
        x = float(s["null_in_vivo"])
        lo99, hi99 = _t080_order_stat(list(d), 0.005), _t080_order_stat(list(d), 0.995)
        lo_fw, hi_fw = (
            _t080_order_stat(list(d), T080_BAND_FW[0]),
            _t080_order_stat(list(d), T080_BAND_FW[1]),
        )
        sd = float(np.std(d))
        ratio = sd / float(s["null"]["draws_sd"])
        p = float(np.mean(d >= x))
        per_seed.append(d)
        print(
            f"| {seed} | {x:.4f} | {ncls} / {incons} | [{lo99:.4f}, {hi99:.4f}] | [{lo_fw:.4f}, {hi_fw:.4f}] | {lo_fw <= x <= hi_fw} | {sd:.4f} / {float(s['null']['draws_sd']):.4f} ({ratio:.3f}) | {p:.4f} |"  # noqa: E501 -- an instrument report line
        )
    rng = np.random.default_rng(31337)
    bands = [(np.quantile(d, 0.005), np.quantile(d, 0.995)) for d in per_seed]
    trials, misses = 20000, 0
    for _ in range(trials):
        for d, (lo, hi) in zip(per_seed, bands, strict=True):
            v = d[rng.integers(len(d))]
            if v < lo or v > hi:
                misses += 1
                break
    print(
        f"family-wise refusal rate of a per-seed 99 % band over {len(per_seed)} seeds, simulated from these nulls: {misses / trials:.4f} (analytic 1 - 0.99^8 = {1 - 0.99**8:.4f})"  # noqa: E501 -- an instrument report line
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
