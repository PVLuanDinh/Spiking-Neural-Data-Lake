"""The safety envelope's line endings (policy audit pa-p5env-1; owner, verbatim, in chat
2026-10-24: "Yes, approve the envelope-hash line-ending fix").

envelope_hash_gate hashes the five files' WORKING bytes. The baseline once mixed CRLF-checkout and
LF-blob hashes, so the gate was red on any fresh checkout. Three checks keep the fix true: every
one of the five carries `eol=lf` (so every checkout holds the stored blob), the baseline holds each
file's LF git-blob hash, and -- the approval's boundary -- the five blobs are byte-identical to
c9d0c097, the tree the owner approved this on. An approved envelope change later moves
`_APPROVED_AT` in the same commit as its baseline."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIVE = ("reflex", "serial_arm", "so101_arm", "arm_config", "interpreter")
_APPROVED_AT = "c9d0c097"


def _git(*args):
    return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, check=True).stdout


@pytest.mark.parametrize("name", _FIVE)
def test_envelope_file_is_checked_out_lf(name):
    out = _git("check-attr", "eol", "--", f"src/snn_data_lake/brainstem/{name}.py").decode()
    assert out.strip().endswith("eol: lf"), out


@pytest.mark.parametrize("name", _FIVE)
def test_baseline_holds_the_lf_blob_hash(name):
    baseline = json.loads((_ROOT / "tools" / "envelope_hashes.json").read_text(encoding="utf-8"))
    blob = _git("cat-file", "blob", f"HEAD:src/snn_data_lake/brainstem/{name}.py")
    assert baseline["files"][f"brainstem/{name}.py"]["sha256"] == hashlib.sha256(blob).hexdigest()


@pytest.mark.parametrize("name", _FIVE)
def test_envelope_blob_unchanged_since_the_approval(name):
    path = f"src/snn_data_lake/brainstem/{name}.py"
    assert _git("rev-parse", f"HEAD:{path}") == _git("rev-parse", f"{_APPROVED_AT}:{path}")
