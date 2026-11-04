r"""D-R0 layout dump: the deposit's variable NAMES, shapes and dtypes for one DEV mouse, no values (pre-reg 2.3-2.4).

MOVED 2026-11-04 from `tools/d_r0_layout.py` (owner "Move D-R0 tools"; P4c ratchet: a tools/ script is never
governed); evidence file section 11 records the move.

`read_field` in `instruments/d_r0_part_a.py` needs the RespData / RoiINFO / StabilityData layout, which round 1
could not read (no mouse data). This prints structure only, through `ophys_ingest.read_response(..., split="dev")`,
so every file it opens is logged in tools/d_r0_dev_log.jsonl (this file hashed as `__main__`) and a held-out file is
refused. A structure read computes no RDI, agreement or control value (evidence section 10: it counts zero
revisions, and it may never print a response value -- the self-check holds that line).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sensorimotor.ophys_layout --mouse 10 [--data-dir DIR]
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sensorimotor.ophys_layout     # no argument: the self-check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from snn_data_lake.sensorimotor.ophys_ingest import DATA_DIR, read_response

ROOT = Path(__file__).resolve().parents[3]
PIN_PATH = ROOT / "tools" / "marks_goard_pin.json"
DEV = (2, 3, 10, 11)
NAMES = ("RoiINFO.mat", "StabilityData.mat", "RespData.mat")


def describe(x, depth=0, max_depth=8):
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
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return _selfcheck()
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


# Sentinel VALUES planted in the synthetic deposit: none may appear in a structure dump.
_LEAKS = ("987654.25", "987654", "LEAKSTRING", "4242", "31337")


def _synthetic():
    """A RespData-shaped .mat through the same loadmat call read_response makes (squeeze_me, struct_as_record)."""
    from scipy.io import loadmat, savemat

    cells = np.empty(2, dtype=object)  # a ragged cell array: loadmat keeps it as dtype=object
    cells[0], cells[1] = np.full((2, 2), 31337.0), np.zeros(3)
    resp = {
        "RespMat_onTime": np.full((3, 4, 5), 987654.25),
        "label": "LEAKSTRING",
        "trials": np.int32(4242),
        "cells": cells,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "RespData.mat"
        savemat(path, {"RespData": {"NatMov": resp}})
        return loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _selfcheck():
    failures = []
    mat = _synthetic()
    text = "\n".join(describe(mat["RespData"]))
    for want in (
        "struct fields=['NatMov']",
        ".RespMat_onTime:",
        "ndarray shape=(3, 4, 5) dtype=float64",
        ".label:",
        ".trials:",
        "(first element:)",
        "ndarray shape=(2, 2) dtype=float64",
    ):
        if want not in text:
            failures.append(f"structure dump lacks {want!r}")
    leaked = [s for s in _LEAKS if s in text]
    if leaked:
        failures.append(f"structure dump LEAKS values {leaked}")
    if sorted(random.Random(20261103).sample(range(1, 13), 4)) != list(DEV):
        failures.append("DEV is not the sealed split")
    with (
        tempfile.TemporaryDirectory() as empty
    ):  # never the deposit, even when the DEV guard is broken
        try:
            main(["--mouse", "1", "--data-dir", empty])
            failures.append("a held-out mouse (1) was not refused")
        except SystemExit as exc:
            if "is not DEV" not in str(exc):
                failures.append(f"held-out mouse refused for another reason: {exc}")
        except Exception as exc:  # a held-out mouse that reaches the reader is the red, named
            failures.append(
                f"a held-out mouse (1) passed the DEV guard and reached the reader: {exc}"
            )
    for f in failures:
        print("FAIL", f)
    if failures:
        return 1
    print(
        "ophys_layout self-check OK: a synthetic RespData dump names every field, shape and dtype (object arrays "
        f"by their first element), none of {len(_LEAKS)} planted values leaks, a held-out mouse is refused"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
