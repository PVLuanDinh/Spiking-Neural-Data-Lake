"""Train-only SHD adapter. Run under the isolated h5py preparation interpreter."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_MD5 = "d47c9825dee33347913e8ce0f2be08b0"
URL = "https://zenkelab.org/datasets/shd/shd_train.h5.gz"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def frame(times, units, bins):
    times = np.asarray(times, dtype=np.float64)
    units = np.asarray(units, dtype=np.int64)
    assert len(times) == len(units) and len(times) > 0
    assert np.isfinite(times).all() and times.min() >= 0
    assert units.min() >= 0 and units.max() < 700
    lo, hi = float(times.min()), float(times.max())
    indices = np.zeros(len(times), dtype=np.int64)
    if hi > lo:
        indices = np.minimum(((times - lo) / (hi - lo) * bins).astype(np.int64), bins - 1)
    counts = np.bincount(indices * 700 + units, minlength=bins * 700).reshape(bins, 700)
    assert int(counts.sum()) == len(times)
    assert counts.max() <= np.iinfo(np.uint16).max
    return counts.astype(np.uint16), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--data-dir", type=Path, default=Path("/private/tmp/sndl-scale-data-20261103"))
    args = ap.parse_args()
    started = time.monotonic()
    data = args.data_dir.resolve()
    gz = data / "shd_train.h5.gz"
    if not gz.exists():
        part = gz.with_suffix(".gz.part")
        assert digest(part, "md5") == EXPECTED_MD5, "archive checksum mismatch"
        part.rename(gz)
    assert digest(gz, "md5") == EXPECTED_MD5
    raw = data / "shd_train.h5"
    if not raw.exists():
        part = raw.with_suffix(".h5.part")
        with gzip.open(gz, "rb") as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
        part.rename(raw)
    output = data / f"prepared-t{args.bins}"
    output.mkdir(exist_ok=True)
    assert not (output / "manifest.json").exists(), "immutable prepared dataset already exists"
    # Fixture includes duplicate spikes, endpoints, unequal duration and an all-simultaneous case.
    test, _, _ = frame([0.0, 0.5, 0.5, 1.0], [0, 1, 1, 699], 2)
    assert test[0, 0] == 1 and test[1, 1] == 2 and test[1, 699] == 1
    equal, _, _ = frame([0.5, 0.5], [4, 4], 2)
    assert equal[0, 4] == 2
    assert np.array_equal(test, frame([0.0, 0.5, 0.5, 1.0], [0, 1, 1, 699], 2)[0])
    with h5py.File(raw, "r") as f:
        schema = {}
        f.visititems(lambda name, obj: schema.update({name: {"shape": list(obj.shape), "dtype": str(obj.dtype)}}) if isinstance(obj, h5py.Dataset) else None)
        labels = np.asarray(f["labels"], dtype=np.int64)
        speakers = np.asarray(f["extra/speaker"], dtype=np.int64)
        n = len(labels)
        assert n == 8156 and set(labels.tolist()) == set(range(20))
        held_speakers = sorted(set(speakers.tolist()))[-2:]
        dev = np.flatnonzero(np.isin(speakers, held_speakers))
        train = np.flatnonzero(~np.isin(speakers, held_speakers))
        assert set(train).isdisjoint(dev) and len(train) + len(dev) == n
        assert set(speakers[train]).isdisjoint(speakers[dev])
        assert set(labels[train]) == set(labels[dev]) == set(range(20))
        frames = np.lib.format.open_memmap(output / "frames.npy.part", mode="w+", dtype=np.uint16, shape=(n, args.bins, 700))
        event_counts = np.zeros(n, dtype=np.int64)
        limits = np.zeros((n, 2), dtype=np.float64)
        peak_count = 0
        for i in range(n):
            times, units = f["spikes/times"][i], f["spikes/units"][i]
            counts, lo, hi = frame(times, units, args.bins)
            frames[i] = counts
            event_counts[i] = len(times)
            limits[i] = lo, hi
            peak_count = max(peak_count, int(counts.max()))
            if i % 1000 == 0:
                print(json.dumps({"prepared": i, "total": n, "seconds": time.monotonic() - started}), flush=True)
        frames.flush()
        del frames
    (output / "frames.npy.part").rename(output / "frames.npy")
    for name, arr in {"labels": labels, "speakers": speakers, "train_ids": train, "dev_ids": dev, "event_counts": event_counts, "time_limits": limits}.items():
        np.save(output / f"{name}.npy", arr, allow_pickle=False)
    file_receipts = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(output.glob("*.npy"))}
    manifest = {
        "dataset": "SHD official training set only", "url": URL,
        "license": "CC BY 4.0", "publisher_md5": EXPECTED_MD5,
        "archive_sha256": digest(gz), "raw_sha256": digest(raw),
        "schema": schema, "bins": args.bins, "channels": 700, "classes": 20,
        "framing": "per-recording observed min/max equal-width bins, last endpoint clamped; counts preserved",
        "recordings": n, "train_recordings": len(train), "dev_recordings": len(dev),
        "held_speakers": held_speakers, "train_speakers": sorted(set(speakers[train].tolist())),
        "train_class_counts": np.bincount(labels[train], minlength=20).tolist(),
        "dev_class_counts": np.bincount(labels[dev], minlength=20).tolist(),
        "total_events": int(event_counts.sum()), "events_min_median_max": np.quantile(event_counts, [0, 0.5, 1]).tolist(),
        "duration_min_median_max_seconds": np.quantile(limits[:, 1] - limits[:, 0], [0, 0.5, 1]).tolist(),
        "largest_bin_count": peak_count, "all_events_conserved": True,
        "split_disjoint": True, "all_classes_each_split": True, "official_test_read": False,
        "files": file_receipts, "prepare_script_sha256": digest(__file__),
        "numpy": np.__version__, "h5py": h5py.__version__, "wall_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "data" / "cache-location.json").write_text(json.dumps({"prepared": str(output), "manifest_sha256": digest(output / "manifest.json")}, indent=2) + "\n")
    (ROOT / "reports" / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
