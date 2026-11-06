"""Independent audit of the immutable prepared cache against raw SHD."""
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    location = json.loads((ROOT / "data/cache-location.json").read_text())
    data = Path(location["prepared"])
    manifest = json.loads((data / "manifest.json").read_text())
    assert digest(data / "manifest.json") == location["manifest_sha256"]
    for name, receipt in manifest["files"].items():
        assert digest(data / name) == receipt["sha256"], name
    frames = np.load(data / "frames.npy", mmap_mode="r", allow_pickle=False)
    event_counts = np.load(data / "event_counts.npy", allow_pickle=False)
    labels = np.load(data / "labels.npy", allow_pickle=False)
    speakers = np.load(data / "speakers.npy", allow_pickle=False)
    total = 0
    for i in range(0, len(frames), 128):
        counts = np.asarray(frames[i:i+128]).sum((1, 2), dtype=np.int64)
        assert np.array_equal(counts, event_counts[i:i+128])
        total += int(counts.sum())
    raw = data.parent / "shd_train.h5"
    assert digest(raw) == manifest["raw_sha256"]
    examples = [0, len(frames)//2, len(frames)-1]
    with h5py.File(raw, "r") as f:
        assert np.array_equal(np.asarray(f["labels"]), labels)
        assert np.array_equal(np.asarray(f["extra/speaker"]), speakers)
        for i in examples:
            times = f["spikes/times"][i]
            units = f["spikes/units"][i]
            lo, hi = float(min(times)), float(max(times))
            independent = np.zeros((100, 700), dtype=np.int64)
            for t, u in zip(times, units):
                b = min(99, math.floor((float(t) - lo) / (hi - lo) * 100)) if hi > lo else 0
                independent[b, int(u)] += 1
            assert np.array_equal(frames[i], independent)
        keys = [x.decode("utf-8") for x in np.asarray(f["extra/keys"])]
    report = {"all_array_hashes_match": True, "labels_match_raw": True, "speakers_match_raw": True,
              "every_recording_event_count_matches": True, "total_events": total,
              "independent_scalar_binning_checked_ids": examples,
              "class_keys": keys, "official_test_read": False}
    (ROOT / "reports/data-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
