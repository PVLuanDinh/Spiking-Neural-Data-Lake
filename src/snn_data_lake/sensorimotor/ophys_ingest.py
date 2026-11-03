r"""D-R0's reader for the Marks & Goard (2021) two-photon deposit (Dryad doi:10.25349/D9M606, CC0-1.0):
ONLY the `.mat` reading is new. Licence, pin and byte refusal are `organoid_ingest`'s (`gate_license`,
`gate_bytes`, `load_pin`), refusals are `h01_ingest.IngestRefused`, the lake guard is `substrate.home.
assert_not_live_home` (pre-registration specs/022-unified-brain-arc/evidence/d-r0-representational-drift-
instrument.md 2.1, 2.4).

  read_response(path, entry, pin) -- licence, then streamed bytes == pin, then a clean src/ and tools/ tree,
                                     then ONE line appended to `tools/d_r0_dev_log.jsonl` (commit sha,
                                     timestamp, files, sha256 of every snn_data_lake module loaded and of
                                     the calling script), then `scipy.io.loadmat(squeeze_me=True,
                                     struct_as_record=False)`. Every `.mat` read is logged.
  dirty_paths(root)               -- `git status --porcelain` under src/ and tools/, untracked included,
                                     EXCEPT the dev log itself (else its first line would dirty the tree)

Nothing here writes the lake (P4b): the log append first passes `assert_not_live_home`. The tool's pin file is
`tools/marks_goard_pin.json` (the launcher's step; nothing here computes it). Offline, never on a tick.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sensorimotor.ophys_ingest
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from scipy.io import loadmat

from snn_data_lake.growth.h01_ingest import IngestRefused
from snn_data_lake.sensorimotor.organoid_ingest import gate_bytes, gate_license
from snn_data_lake.substrate.home import assert_not_live_home

# distinct-from: instruments/replay_hydration.py, gates/move_gate.py, gates/p5b_gate.py, instruments/g5_part_d_adaptors.py, instruments/rpe_charge.py -- the graph-exists hits for rev-parse, porcelain status and module hashing, each read: `_head_of` takes a path and returns "" on failure (this must refuse), `code_hash` and `native_identity` hash ONE module, move_gate/p5b_gate census a tree; none refuses a dirty src/tools tree or logs a read's loaded-module hashes

ROOT = Path(__file__).resolve().parents[3]
PIN_PATH = ROOT / "tools" / "marks_goard_pin.json"
LOG_PATH = ROOT / "tools" / "d_r0_dev_log.jsonl"


def _git(root, *args):
    proc = subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        raise IngestRefused(f"git {' '.join(args)} failed in {root}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def dirty_paths(root=ROOT, log_path=LOG_PATH):
    rel = Path(log_path).resolve().relative_to(Path(root).resolve()).as_posix()
    out = _git(root, "status", "--porcelain", "--untracked-files=all", "--", "src", "tools")
    paths = [line[3:].split(" -> ")[-1].strip('"') for line in out.splitlines()]
    return [p for p in paths if p != rel]


def source_hashes():
    """sha256 of every snn_data_lake module loaded now, plus the running script (the calling tool)."""
    mods = {
        n: getattr(m, "__file__", None)
        for n, m in sys.modules.items()
        if n.startswith("snn_data_lake")
    }
    mods["__main__"] = getattr(sys.modules.get("__main__"), "__file__", None)
    return {
        n: hashlib.sha256(Path(f).read_bytes()).hexdigest() for n, f in sorted(mods.items()) if f
    }


def _append_log(entry, log_path, live_home=None):
    assert_not_live_home(Path(log_path).parent, live_home)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def read_response(path, entry, pin, root=ROOT, log_path=LOG_PATH, live_home=None):
    gate_license(pin)
    gate_bytes(path, entry)
    dirty = dirty_paths(root, log_path)
    if dirty:
        raise IngestRefused(
            f"dirty tree under src/ or tools/ ({len(dirty)}): {dirty[:3]}; commit first"
        )
    _append_log(
        {
            "commit": _git(root, "rev-parse", "HEAD").strip(),
            "time": datetime.now(UTC).isoformat(),
            "files": [{"path": entry["path"], "sha256": entry["sha256"]}],
            "sources": source_hashes(),
        },
        log_path,
        live_home,
    )
    return loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _selfcheck():
    """Every refusal's constructed red, then a green control, on a synthetic .mat in a temp git repo."""
    import numpy as np
    from scipy.io import savemat

    from snn_data_lake.sensorimotor.organoid_ingest import stream_sha256
    from snn_data_lake.substrate.home import LiveHomeRefused

    failures = []

    def must_refuse(label, call, exc=IngestRefused, needle=""):
        try:
            call()
        except exc as e:
            if needle not in str(e):
                failures.append(f"{label}: refused for the wrong reason: {e}")
            return
        failures.append(f"{label}: accepted the constructed red")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "src").mkdir()
        (td / "tools").mkdir()
        (td / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
        log = td / "tools" / "d_r0_dev_log.jsonl"
        for cmd in (
            ["init", "-q"],
            ["add", "."],
            ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"],
        ):
            _git(td, *cmd)
        mat = td / "resp.mat"
        savemat(mat, {"RespMat": np.arange(6.0).reshape(2, 3), "Info": {"week": 3.0, "name": "m"}})
        sha, size = stream_sha256(mat)
        entry = {"path": "resp.mat", "size": size, "sha256": sha}
        pin = {"license": "spdx:CC0-1.0", "assets": [entry]}
        rd = lambda p=mat, e=entry, pn=pin: read_response(p, e, pn, td, log)  # noqa: E731

        must_refuse(
            "licence empty",
            lambda: read_response(mat, entry, {**pin, "license": ""}, td, log),
            needle="(a)",
        )
        flipped = bytearray(mat.read_bytes())
        flipped[len(flipped) // 2] ^= 1
        (td / "flip.mat").write_bytes(bytes(flipped))
        must_refuse("one byte flipped", lambda: rd(td / "flip.mat"), needle="(b)")
        (td / "tools" / "stray.py").write_text("y = 2\n", encoding="utf-8")
        must_refuse("dirty tree, untracked", rd, needle="dirty tree")
        (td / "tools" / "stray.py").unlink()
        (td / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
        must_refuse("dirty tree, modified", rd, needle="dirty tree")
        (td / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
        if log.exists():
            failures.append("a refused read appended to the log")

        data = rd()
        if not (data["Info"].week == 3.0 and data["RespMat"].shape == (2, 3)):
            failures.append("green control: loadmat did not round-trip a struct and a matrix")
        rd()
        lines = [json.loads(s) for s in log.read_text(encoding="utf-8").splitlines()]
        if (
            len(lines) != 2
            or len(lines[0]["commit"]) != 40
            or lines[0]["files"][0]["sha256"] != sha
        ):
            failures.append(
                f"dev log: want 2 entries with a commit sha and the file sha, got {lines}"
            )
        names = set(lines[0]["sources"])
        if (
            not {
                "__main__",
                "snn_data_lake.growth.h01_ingest",
                "snn_data_lake.sensorimotor.organoid_ingest",
            }
            <= names
        ):
            failures.append(f"dev log: loaded-module hashes missing: {sorted(names)[:4]}")
        if dirty_paths(td, log):
            failures.append("the dev log dirtied the tree")

        # P4b: a log path under the (stand-in) lake home must raise, and write nothing
        lake = td / "lake"
        lake.mkdir()
        must_refuse(
            "write under the lake home",
            lambda: _append_log({}, lake / "x.jsonl", live_home=lake),
            LiveHomeRefused,
        )
        if (lake / "x.jsonl").exists():
            failures.append("a refused lake write left a file")
    for f in failures:
        print("FAIL", f)
    if failures:
        return 1
    print(
        "ophys_ingest self-check OK: reds (licence, bytes, dirty untracked, dirty modified, lake write) refuse, "
        "green control loads a struct and a matrix, two reads log two lines, the log never dirties the tree"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
