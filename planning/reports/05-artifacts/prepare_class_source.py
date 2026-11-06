"""Recreate/verify the exact source export used by the class-order pilot."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEAD = "c4c10b285f1741aa798d5ee9c5296ae371fabecf"
receipt = HERE / "class-source-export.json"
info = json.loads(receipt.read_text()) if receipt.exists() else {}
directory = Path(info["source_root"]) if info.get("source_root") else None
if directory is None or not directory.exists():
    directory = Path(tempfile.mkdtemp(prefix="sndl-class-source-", dir="/private/tmp"))
    raw = subprocess.check_output(["git", "-C", str(ROOT / "repo"), "archive", "--format=tar", HEAD, "src", "tools"], timeout=30)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(directory, filter="data")
    info = {"source_root": str(directory), "head": HEAD, "export_bytes": len(raw)}
    receipt.write_text(json.dumps(info, indent=2) + "\n")
listing = subprocess.check_output(["git", "-C", str(ROOT / "repo"), "ls-tree", "-r", HEAD, "--", "src", "tools"], text=True)
count = 0
for line in listing.splitlines():
    meta, name = line.split("\t", 1)
    mode, kind, expected = meta.split()
    path = directory / name
    data = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if kind != "blob" or expected != actual:
        raise RuntimeError(f"Source export mismatch: {name}")
    count += 1
verified = {**info, "verified_git_blobs": count, "all_match": True}
(HERE / "class-source-verification.json").write_text(json.dumps(verified, indent=2) + "\n")
print(json.dumps(verified))
