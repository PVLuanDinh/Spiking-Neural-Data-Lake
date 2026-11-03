r"""D-R0 layout dump: the deposit's variable NAMES, shapes and dtypes for one DEV mouse, no values (pre-reg 2.3-2.4).

`read_field` in tools/d_r0_partA.py needs the RespData / RoiINFO / StabilityData layout, which round 1 could not read
(no mouse data). This prints structure only, through `ophys_ingest.read_response(..., split="dev")`, so every file it
opens is logged in tools/d_r0_dev_log.jsonl and a held-out file is refused. Run once, then commit the dev log.

  python tools/d_r0_layout.py --mouse 10 [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_data_lake.sensorimotor.ophys_ingest import DATA_DIR, read_response  # noqa: E402

PIN_PATH = ROOT / "tools" / "marks_goard_pin.json"
DEV = (2, 3, 10, 11)
NAMES = ("RoiINFO.mat", "StabilityData.mat", "RespData.mat")


def describe(x, depth=0, max_depth=4):
    """Structure only: type, shape, dtype, field names; object arrays describe their FIRST element only."""
    pad = "  " * depth
    if hasattr(x, "_fieldnames"):
        lines = [f"{pad}struct fields={list(x._fieldnames)}"]
        if depth < max_depth:
            for f in x._fieldnames:
                lines.append(f"{pad}  .{f}:")
                lines += describe(getattr(x, f), depth + 2, max_depth)
        return lines
    if isinstance(x, np.ndarray):
        line = f"{pad}ndarray shape={x.shape} dtype={x.dtype}"
        if x.dtype == object and x.size and depth < max_depth:
            return [line + " (first element:)"] + describe(x.flat[0], depth + 1, max_depth)
        return [line]
    return [f"{pad}{type(x).__name__}"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    a = ap.parse_args(argv)
    if a.mouse not in DEV:
        raise SystemExit(f"mouse {a.mouse} is not DEV {DEV}")
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    prefix = f"Main experiments/Mouse {a.mouse}/"
    for entry in pin["assets"]:
        if entry["path"].startswith(prefix) and entry["path"].rsplit("/", 1)[-1] in NAMES:
            mat = read_response(a.data_dir / entry["path"], pin, "dev", data_dir=a.data_dir)
            print(f"== {entry['path']}")
            for k, v in mat.items():
                if not k.startswith("__"):
                    print(f"  {k}:")
                    print("\n".join(describe(v, 2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
