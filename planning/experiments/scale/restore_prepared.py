"""Rebuild a lost frame cache without changing the original cohort's dataset identity."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MASTER = Path("/Users/arthurkhvan/.codex/visualizations/2026/11/03/01a19f80-851f-7960-9c7a-18ef1711b019/shd-recovery")
original = (MASTER / "original-data-manifest.json").read_bytes()
expected = json.loads(original)
location = json.loads((MASTER / "original-cache-location.json").read_text())
data = Path(location["prepared"])
if not (data / "manifest.json").exists():
    isolated = MASTER / "preparation-rebuild"
    for name in ("data", "reports"):
        (isolated / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "prepare_shd.py", isolated / "prepare_shd.py")
    env = dict(os.environ, PYTHONPATH="/private/tmp/sndl-shd-prep-site")
    subprocess.run([sys.executable, str(isolated / "prepare_shd.py"), "--data-dir", str(data.parent)], env=env, check=True)
    rebuilt = json.loads((data / "manifest.json").read_text())
    assert rebuilt["files"] == expected["files"]
    assert rebuilt["archive_sha256"] == expected["archive_sha256"]
    assert rebuilt["raw_sha256"] == expected["raw_sha256"]
    shutil.copyfile(data / "manifest.json", isolated / "rebuilt-manifest.json")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

for name, receipt in expected["files"].items():
    assert sha(data / name) == receipt["sha256"], name
assert hashlib.sha256(original).hexdigest() == location["manifest_sha256"]
(data / "manifest.json").write_bytes(original)
print("Every cached array matches the original; original manifest identity restored.")
