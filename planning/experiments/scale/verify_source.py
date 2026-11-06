"""Verify the local export against the saved, pinned GitHub tree response."""
import hashlib
import json
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/private/tmp/sndl-class-source-tlefts_z")
expected = json.loads((ROOT / "reports/source-expected-blobs.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--imported-only", action="store_true")
args = parser.parse_args()
selected = None
if args.imported_only:
    start = json.loads((ROOT / "runs/shd-joint-s8000/events.jsonl").open().readline())
    selected = {"src/" + p for p in start["source_files"]}
assert expected["commit"] == "c4c10b285f1741aa798d5ee9c5296ae371fabecf"
assert expected["truncated"] is False
files = {}
for entry in expected["blobs"]:
    if selected is not None and entry["path"] not in selected:
        continue
    path = SOURCE / entry["path"]
    body = os.readlink(path).encode() if entry["mode"] == "120000" else path.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    assert actual == entry["sha"], entry["path"]
    files[entry["path"]] = {"git_blob": actual, "sha256": hashlib.sha256(body).hexdigest()}
result = {"repository": expected["repository"], "commit": expected["commit"], "remote_tree_url": expected["url"],
          "verified_blobs": len(files), "all_match": True, "scope": "imported-only" if args.imported_only else "full-original-export", "source_root": str(SOURCE), "files": files}
destination = "source-recovery-audit.json" if args.imported_only else "source-audit.json"
(ROOT / "reports" / destination).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
