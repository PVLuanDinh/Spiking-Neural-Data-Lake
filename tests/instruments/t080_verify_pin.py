"""Re-derive every barred number of the T080 pin from the artifacts alone -- the orchestrator's own
probe, committed at the panel fold (2026-09-16) because the record cited it under an untracked
scratchpad (obs #2180 / #2185: a builder's number is data until reproduced). Reads
tools/t080_temporal_binding.json and tools/t080_latents.npz; constructs NO brain. Per seed: (1)
the decomposition of the companion's latents through a fresh CrossModalBinder taught the pinned
prototypes must equal the pinned recs and the recorded handles; (2) the bound / congruent
fractions from the recs; (3) the offline membership-matched null with an INDEPENDENT permutation
loop (not t080_null), its q95 / band; (4) the contrasts through instruments.paired_contrast; (5) the
window. Prints every disagreement with the pin beyond 1e-9 and exits 1 on any.

  PYTHONUTF8=1 PYTHONPATH=src python tests/instruments/t080_verify_pin.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from snn_data_lake.cortex.crossmodal_binder import CrossModalBinder
from snn_data_lake.infra._mathutils import q95
from snn_data_lake.instruments.paired_contrast import contrast

REPO = Path(__file__).resolve().parents[2]
PIN = REPO / "tools" / "t080_temporal_binding.json"
NPZ = REPO / "tools" / "t080_latents.npz"

pin = json.loads(PIN.read_text(encoding="utf-8"))
meta = pin["latents_npz"]
sha = hashlib.sha256(NPZ.read_bytes()).hexdigest()
bad = []
if sha != meta["sha256"]:
    bad.append(f"companion sha256 {sha[:12]} != pinned {meta['sha256'][:12]}")
z = np.load(NPZ)
print(
    f"pin head {pin['head'][:9]} verdict {pin['verdict']} refusals {pin['refusals']} elapsed {pin['elapsed_s']} s; companion sha ok {sha == meta['sha256']}"  # noqa: E501 -- an instrument report line
)

per_seed_rows = []
for s in pin["per_seed"]:
    seed = s["seed"]
    binder = CrossModalBinder(seed=seed)
    for h, p in s["prototypes"].items():
        binder.register(h, p["vis"], p["aud"])
    taught_ok = sorted(binder.objects) == sorted(s["taught_end"]) if "taught_end" in s else True
    row = {"seed": seed, "taught_ok": taught_ok}
    for arm, counts in s["arms"].items():
        vis, aud = z[f"s{seed}_{arm}_vis"], z[f"s{seed}_{arm}_aud"]
        recs = counts["recs"]
        if len(recs) != len(vis):
            bad.append(f"seed {seed} {arm}: {len(recs)} recs vs {len(vis)} latent rows")
            continue
        dis = 0
        bound = 0
        for r, v, a in zip(recs, vis, aud, strict=True):
            ov, sv = binder._recognize(binder._project([float(x) for x in v], "vision"), "vision")
            oa, sa = binder._recognize(binder._project([float(x) for x in a], "audio"), "audio")
            b = ov is not None and ov == oa
            bound += int(b)
            if b != bool(r["bound"]) or ov != r.get("obj_v") or oa != r.get("obj_a"):
                dis += 1
        frac = bound / len(recs)
        if dis:
            bad.append(f"seed {seed} {arm}: decomposition disagrees on {dis}/{len(recs)} rows")
        if abs(frac - counts["bound_frac"]) > 1e-9:
            bad.append(
                f"seed {seed} {arm}: bound_frac {frac:.6f} vs pinned {counts['bound_frac']:.6f}"
            )
        row[arm] = frac
    # the offline null, independently: permute the audio objects across the ALIGNED ticks
    recs = s["arms"]["aligned"]["recs"]
    ov = np.array([r.get("obj_v") or "" for r in recs])
    oa = np.array([r.get("obj_a") or "" for r in recs])
    rng = np.random.default_rng(pin["constants"].get("T080_NULL_RNG_BASE", 7000) + seed)
    n = len(recs)
    draws = []
    for _ in range(int(pin["constants"].get("T080_N_DRAWS", 2000))):
        perm = rng.permutation(n)
        draws.append(float(np.mean((ov != "") & (ov == oa[perm]))))
    my_q95 = q95(draws)
    my_mean = float(np.mean(draws))
    row["null_q95"] = my_q95
    row["null_mean"] = my_mean
    if abs(my_q95 - s["q95_null"]) > 1e-9:
        bad.append(
            f"seed {seed}: null q95 {my_q95:.6f} vs pinned {s['q95_null']:.6f} (the RNG stream may differ; mean {my_mean:.4f} vs {s['null']['mean']:.4f})"  # noqa: E501 -- an instrument report line
        )
    lo, hi = s["q_lo_null"], s["q_hi_null"]  # the family-wise refusal band (2.11)
    row["b2"] = lo <= s["null_in_vivo"] <= hi
    row["within"] = s["within"]
    row["cong_aligned"] = s["congruent_aligned"]
    row["cong_null"] = s["congruent_null"]
    lag_rates = {int(k): v for k, v in s["lag_rates"].items()}
    win = next((d for d in sorted(lag_rates) if lag_rates[d] <= s["q95_null"]), "none")
    if str(win) != str(s["window"]):
        bad.append(f"seed {seed}: window {win} vs pinned {s['window']}")
    row["window"] = win
    per_seed_rows.append(row)
    print(
        f"  seed {seed}: within {row['within']:.4f} q95 {my_q95:.4f} (pin {s['q95_null']:.4f}) null_in_vivo {s['null_in_vivo']:.4f} b2 {row['b2']} "  # noqa: E501 -- an instrument report line
        f"congruent {row['cong_aligned']:.4f}/{row['cong_null']:.4f} window {win} bound {{{', '.join(f'{a}: {row[a]:.4f}' for a in s['arms'])}}}"  # noqa: E501 -- an instrument report line
    )

g0 = contrast("g0", [r["cong_aligned"] - r["cong_null"] for r in per_seed_rows])
b1 = contrast("b1", [r["within"] - r["null_q95"] for r in per_seed_rows])
b2 = all(r["b2"] for r in per_seed_rows)
print(
    f"MY (g0): t {g0.t:+.4f} FPR {g0.fpr} met {g0.met} | pin t {pin['contrasts']['g0']['t']:+.4f} met {pin['contrasts']['g0']['met']}"  # noqa: E501 -- an instrument report line
)
print(
    f"MY (b1): t {b1.t:+.4f} FPR {b1.fpr} met {b1.met} | pin t {pin['contrasts']['b1']['t']:+.4f} met {pin['contrasts']['b1']['met']}"  # noqa: E501 -- an instrument report line
)
print(f"MY (b2): {b2} | pin {pin['contrasts']['b2']}")
for k, mine, theirs in (
    ("g0", g0.t, pin["contrasts"]["g0"]["t"]),
    ("b1", b1.t, pin["contrasts"]["b1"]["t"]),
):
    if abs(mine - theirs) > 1e-6:
        bad.append(f"{k}: my t {mine:.6f} vs pinned {theirs:.6f}")
verdict = (
    "REFUSED" if (pin["refusals"] or not g0.met or not b2) else ("PASS" if b1.met else "NOT MET")
)
print(f"MY VERDICT {verdict} | pin {pin['verdict']}")
if verdict != pin["verdict"]:
    bad.append(f"verdict {verdict} vs pinned {pin['verdict']}")
print("windows", [r["window"] for r in per_seed_rows])
if bad:
    print("DISAGREEMENTS:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("OK: every barred number re-derived from the pin and the companion with no brain")
