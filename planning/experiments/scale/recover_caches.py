"""Recover exact pinned caches after the observed temporary-directory loss."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MASTER = Path("/Users/arthurkhvan/.codex/visualizations/2026/11/03/01a19f80-851f-7960-9c7a-18ef1711b019/shd-recovery")
MASTER.mkdir(parents=True, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download(url, name, expected=None):
    path = MASTER / name
    if not path.exists() or (expected and sha(path) != expected):
        part = path.with_suffix(path.suffix + ".part")
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=30) as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                if expected:
                    assert sha(part) == expected, name
                part.replace(path)
                break
            except Exception:
                if attempt == 1:
                    raise
        print(json.dumps({"download": name, "bytes": path.stat().st_size}), flush=True)
    if expected:
        assert sha(path) == expected
    return path


def install(receipts, site):
    site = Path(site)
    site.mkdir(parents=True, exist_ok=True)
    for item in receipts:
        wheel = download(item["url"], item["wheel"], item["sha256"])
        with zipfile.ZipFile(wheel) as z:
            z.extractall(site)


def pip_receipts(path):
    report = json.loads(path.read_text())
    return [{"url": e["download_info"]["url"], "wheel": e["download_info"]["url"].rsplit("/", 1)[1],
             "sha256": e["download_info"]["archive_info"]["hashes"]["sha256"]} for e in report["install"]]


def main():
    # Preserve exact pre-interruption manifests before the deterministic preprocessor regenerates.
    for name, src in {"original-data-manifest.json": ROOT / "reports/data-manifest.json",
                      "original-cache-location.json": ROOT / "data/cache-location.json"}.items():
        if not (MASTER / name).exists():
            shutil.copyfile(src, MASTER / name)
    runtime = json.loads((ROOT.parents[1] / "reports/05-artifacts/class-runtime-recovery.json").read_text())
    install(runtime["receipts"], runtime["site"])
    install(pip_receipts(ROOT / "diagnostics/prep-install.json"), "/private/tmp/sndl-shd-prep-site")
    install(pip_receipts(ROOT / "diagnostics/plot-install.json"), "/private/tmp/sndl-shd-plot-site")
    source = Path("/private/tmp/sndl-class-source-tlefts_z")
    source.mkdir(exist_ok=True)
    if (MASTER / "source-files.json").exists():
        # Authenticated GitHub blob responses, saved after the private archive URL refused access.
        for entry in json.loads((MASTER / "source-files.json").read_text()):
            body = entry["content"].encode()
            assert hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == entry["sha"]
            path = source / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
    else:
        raise RuntimeError("Recover imported source blobs with the authenticated GitHub connector; public archive access is unavailable")
    shd = download("https://zenkelab.org/datasets/shd/shd_train.h5.gz", "shd_train.h5.gz", "e95bdc00c03c36537ff587d85d14efeee14a01b3846e14a70defc7e8a14e387c")
    data = Path("/private/tmp/sndl-scale-data-20261103")
    data.mkdir(exist_ok=True)
    shutil.copyfile(shd, data / "shd_train.h5.gz")
    print(json.dumps({"cache_recovery": "complete", "persistent_download_receipts": str(MASTER)}), flush=True)


if __name__ == "__main__":
    main()
