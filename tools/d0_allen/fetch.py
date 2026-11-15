"""Fetch runner for D0 Part A (final project M2/M4): for each container in a pin, resolve each session's
eye-gaze file on the public Allen bucket and run the FROZEN extraction (extract.py, sha256 pinned) into
data/drift/allen/<split>/<exp_id>.npz (gitignored; never the repo or the lake).

  python fetch.py dev      # the 4 DEV containers (tools/d0_partA_split_pin.json "dev")
  python fetch.py sealed   # the 44 sealed units; REFUSED unless tools/d0_allen/FROZEN exists (prereg sec. 5, R-E)

Idempotent: an existing npz is skipped. One request stream at a time (polite to the Allen servers).
"""

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))  # extract.py imports `rangefile` as a top-level module (frozen code)
BUCKET = "https://allen-brain-observatory.s3.amazonaws.com"
EYE_PREFIX = "visual-coding-2p/ophys_eye_gaze_mapping/"
OUT = ROOT / "data" / "drift" / "allen"


def check_frozen_code():
    pin = json.loads(
        (ROOT / "tools" / "d0_partA_heldout_units_pin.json").read_text(encoding="utf-8")
    )
    for name, want in pin["frozen_pipeline_sha256"].items():
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(
                f"refused: {name} sha256 {got[:12]} != pinned {want[:12]} (frozen code changed)"
            )


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


def units(split):
    if split == "dev":
        return json.loads((ROOT / "tools" / "d0_partA_split_pin.json").read_text(encoding="utf-8"))[
            "dev"
        ]
    if split == "sealed":
        if not (HERE / "FROZEN").exists():
            raise SystemExit(
                "refused: sealed read before the freeze (tools/d0_allen/FROZEN absent)"
            )
        return json.loads(
            (ROOT / "tools" / "d0_partA_heldout_units_pin.json").read_text(encoding="utf-8")
        )["units"]
    raise SystemExit(f"unknown split {split!r}")


def main(split):
    check_frozen_code()
    from extract import extract  # frozen block 2

    out = OUT / split
    out.mkdir(parents=True, exist_ok=True)
    for u in units(split):
        for _, exp_id, _ in u["sessions"]:
            dst = out / f"{exp_id}.npz"
            if dst.exists():
                continue
            try:
                print(u["container"], *extract(str(exp_id), eye_key(exp_id), str(dst)), flush=True)
            except (
                Exception
            ) as e:  # report and continue; a missing session is reported, never imputed
                print(f"FAILED {u['container']} {exp_id}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
