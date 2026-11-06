"""Recover exact existing cp314t wheels to /private/tmp, without changing project venv."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
runtime = Path(tempfile.mkdtemp(prefix="sndl-class-packages-", dir="/private/tmp"))
site = runtime / "site"
site.mkdir()
receipts = []
for line in (HERE.parent / "04-artifacts/runtime-requirements.txt").read_text().splitlines():
    package, version = line.split("==")
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30) as response:
        metadata = json.load(response)
    candidates = []
    for item in metadata["urls"]:
        name = item["filename"]
        pure = name.endswith("-none-any.whl")
        native = "macosx_" in name and ("arm64" in name or "universal2" in name) and "-cp314-cp314t-" in name
        if item["packagetype"] == "bdist_wheel" and (pure or native):
            candidates.append(item)
    if not candidates:
        raise RuntimeError(f"No exact cp314t/arm64 wheel for {package}=={version}")
    item = sorted(candidates, key=lambda row: ("universal2" in row["filename"], row["filename"]))[0]
    print("recover", item["filename"], flush=True)
    with urllib.request.urlopen(item["url"], timeout=60) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != item["digests"]["sha256"]:
        raise RuntimeError("Wheel digest mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if not parts or ".." in parts or member.filename.startswith("/"):
                raise RuntimeError("Unsafe wheel path")
            if parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in ("purelib", "platlib"):
                    continue
                dest = site.joinpath(*parts[2:])
            else:
                dest = site.joinpath(*parts)
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(member))
    receipts.append({"package": package, "version": version, "wheel": item["filename"],
                     "url": item["url"], "sha256": digest, "bytes": len(raw)})
    (runtime / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
result = {"runtime_root": str(runtime), "site": str(site), "receipts": receipts}
(HERE / "class-runtime-recovery.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"site": str(site), "packages": len(receipts)}), flush=True)
