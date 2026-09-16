"""T080 (b2) re-read from the committed pin alone, no brain (the panel fold, 2026-09-16): the
family- wise band's endpoints as pinned are order statistics 1 and 1998 of 2,000 draws at per-
side alpha 0.000625 (1.25 expected exceedances a side), whose own Monte-Carlo sd is an order of
magnitude above q95's; with the campaign's recordings FIXED the pinned clause refuses on a
fraction of null RNG seeds. This instrument prints, per seed: the in-vivo NULL fraction; q_hi's
sd over 200 independent 2,000-draw nulls and how many of them would refuse; and the RESOLVED
reading at 200,000 draws under the pre-registered RNG (7000 + seed) with the seed's one-sided
tail p. It writes nothing and changes no pin; (b1) is printed at 200,000 draws beside the pinned
t so the barred claim's robustness to the draw count is on the record.

  PYTHONUTF8=1 PYTHONPATH=src python tests/instruments/t080_b2_reread.py [pin.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

from snn_data_lake.substrate.paired_contrast import contrast
from snn_data_lake.thalamus.binding_instrument import (
    T080_BAND_FW,
    T080_N_DRAWS,
    T080_NULL_RNG_BASE,
    T080_PIN,
    _t080_order_stat,
)

REREAD_DRAWS = 200_000
NULL_SEEDS = 200


def _cols(recs):
    return (
        np.array([r.get("obj_v") or "" for r in recs]),
        np.array([r.get("obj_a") or "" for r in recs]),
    )


def _draws(recs, rng, n_draws):
    ov, oa = _cols(recs)
    n = len(recs)
    out = np.empty(n_draws)
    for i in range(n_draws):
        out[i] = np.mean((ov != "") & (ov == oa[rng.permutation(n)]))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pin_path = Path(argv[0]) if argv else T080_PIN
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    print(
        f"T080 (b2) re-read -- pin {pin['head'][:9]} verdict {pin['verdict']}; {NULL_SEEDS} nulls x {T080_N_DRAWS} draws, then {REREAD_DRAWS} draws"  # noqa: E501 -- an instrument report line
    )
    refusing_seeds = 0
    b1_diffs = []
    for s in pin["per_seed"]:
        seed = s["seed"]
        recs = s["arms"]["aligned"]["recs"]
        x = float(s["null_in_vivo"])
        his, outs = [], 0
        for i in range(NULL_SEEDS):
            d = np.sort(
                _draws(recs, np.random.default_rng(500_000 + 1_000 * i + seed), T080_N_DRAWS)
            )
            lo, hi = (
                _t080_order_stat(list(d), T080_BAND_FW[0]),
                _t080_order_stat(list(d), T080_BAND_FW[1]),
            )
            his.append(hi)
            outs += int(not (lo <= x <= hi))
        big = np.sort(_draws(recs, np.random.default_rng(T080_NULL_RNG_BASE + seed), REREAD_DRAWS))
        lo, hi = (
            _t080_order_stat(list(big), T080_BAND_FW[0]),
            _t080_order_stat(list(big), T080_BAND_FW[1]),
        )
        q95_big = _t080_order_stat(list(big), 0.95)
        p_tail = float(np.mean(big >= x))
        b1_diffs.append(float(s["within"]) - q95_big)
        refusing_seeds += int(outs > 0)
        print(
            f"  seed {seed}: in-vivo {x:.4f} | pinned fw [{s['q_lo_null']:.4f}, {s['q_hi_null']:.4f}] | q_hi sd over "  # noqa: E501 -- an instrument report line
            f"{NULL_SEEDS} nulls {float(np.std(his)):.4f}, refusing {outs}/{NULL_SEEDS} | {REREAD_DRAWS}-draw fw "  # noqa: E501 -- an instrument report line
            f"[{lo:.4f}, {hi:.4f}] inside {lo <= x <= hi}, tail p {p_tail:.5f} (alpha/side {T080_BAND_FW[0]:.6f}) | "  # noqa: E501 -- an instrument report line
            f"q95 {REREAD_DRAWS}-draw {q95_big:.4f} vs pinned {s['q95_null']:.4f}"
        )
    b1 = contrast("(b1) at 200,000 draws", b1_diffs)
    print(
        f"  seeds where the pinned 2,000-draw clause refuses on some null RNG seed: {refusing_seeds} of {len(pin['per_seed'])}"  # noqa: E501 -- an instrument report line
    )
    print(
        f"  (b1) re-read at {REREAD_DRAWS} draws: t {b1.t:+.4f} FPR {b1.fpr} met {b1.met} | pinned t {pin['contrasts']['b1']['t']:+.4f}"  # noqa: E501 -- an instrument report line
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
