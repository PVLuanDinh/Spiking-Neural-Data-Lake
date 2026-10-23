"""organoid_ingest -- OG1 Stage 0: public organoid spike-sorted NWB files -> one npz per recording.

Frozen pre-registration: specs/022-unified-brain-arc/evidence/og1-organoid-stream.md section 2.
Data: DANDI 001603 version 0.260923.0623 (CC-BY-4.0), subjects HO1-HO8, the spike-sorted file of every
session (never `_ecephys`). The pin `tools/og1_dandi_pin.json` lists every file with its size and
`dandi:sha2-256`. This module reads a pinned file through five refusal gates and writes the artifact
the census reads. It never touches a Brain or the lake (P4b): "the lake is what the brain LEARNED;
this is what an electrode saw".

OFFLINE DOWNLOAD (this module does NO networking). One `curl -L --fail -o <data>/nwb/<name>
https://api.dandiarchive.org/api/assets/<asset_id>/download/` per pin entry, where <name> is the pin
path with "/" replaced by "__" (`sub-HO1__sub-HO1_ses-20250924T002125.nwb`); `pin_assets()` lists the
asset ids. About 0.64 GB for the 24 files.

GATES. Each is a refusal (`IngestRefused`, reused from `growth/h01_ingest`: it already carries
`.readings`, so a refusal arrives with the evidence of every gate) and a refusal yields no artifact.
  (a) LICENSE   the pin's DANDI license passes `ingest_guard.license_ok` (deny-by-default).
  (b) BYTES     streamed sha256 and size equal the pin, BEFORE h5py opens the file.
  (c) INDEX     `spike_times_index` non-decreasing, last entry == len(spike_times), >= 1 unit; every
                spike time finite and >= 0 (Amendment A4.8); the units table carries one finite
                `x_pos` and `y_pos` per unit (a file without them is refused).
  (d) TIME UNITS  of {seconds, milliseconds, samples at 20 kHz} the declared-seconds reading puts the
                last spike closest to the recording span the file records, and no spike lies at or
                past that span (Amendment A4.8).
  (e') REPRODUCE THE DEPOSITORS' BINNING (Amendment A1.1 replaced the frozen (e): the papers publish no
                per-recording unit count). The 1 ms raster `t_spk_mat` is rebuilt from the units table,
                row = floor((t - starting_time) x rate - 0.5) (Amendment A2.1: the depositors' MATLAB 1-based
                round; A1.1's plain floor was the instrument's defect), columns in units-table order, and
                must equal the stored one cell for cell. Extremum rule: plain floor, round-half-even,
                floor(x + 0.5) and a one-column unit-order shift must each agree strictly less. A mismatch is refused with its fraction and cells; nothing
                is tuned.
DUPLICATES (A1.2). Within a subject, files whose decoded `spike_times` and `spike_times_index` are exactly
equal are duplicates; the lexicographically first session id is kept. `session_manifest` writes
`og1_sessions.json` (npz, subject, session, split, source_sha256, units, duration_s, duplicate_of,
dup_timestamps, out_of_order); the last two count, per file, the within-unit equal-timestamp pairs and
the within-unit decreasing steps (disclosed, never a refusal).

THE RECORDED SPAN. These files carry no `obs_intervals` and no devices. What they record is a binned
raster, `processing/curated_binning/t_spk_mat` (time_bins x units, `starting_time.rate` = 1000 Hz);
gate (d)'s duration is `(time_bins + 0.5) / rate` (Amendment A5.1): the upper edge of the raster under
the A2.1 row rule, row = floor(t x rate - 0.5), so a spike is out of span iff its row is >= time_bins. It is
the depositors' own span, stored beside the spikes.

ARTIFACT (interface contract of the OG1 brief): `og1_<subject>_<session-id>.npz`, `np.savez`, loadable
with `allow_pickle=False`: spike_times f8 (seconds, all units in NWB order), spike_times_index i8 (NWB
cumulative END index), x_pos / y_pos f8 (one per unit, NWB order; Amendment A4.8), duration_s f8 0-d,
subject / session / source_sha256 0-d unicode.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sensorimotor.organoid_ingest            # self-check
  ... organoid_ingest --convert <data dir>   # gates + npz for the 24 pinned files (data dir/nwb -> data
                                           # dir) + og1_sessions.json (duplicate grouping)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from snn_data_lake.growth.h01_ingest import IngestRefused
from snn_data_lake.sensorimotor.ingest_guard import license_ok

PIN_PATH = Path(__file__).resolve().parents[3] / "tools" / "og1_dandi_pin.json"

#: CC-BY-4.0 attribution for the deposited data and the three papers its description names.
ATTRIBUTION = (
    "DANDI 001603 v0.260923.0623, van der Molen et al., CC-BY-4.0; recordings of Sharf et al. 2022 "
    "Nat Commun doi:10.1038/s41467-022-32115-4 (HO1-HO4), Alam El Din et al. 2025 Commun Biol "
    "doi:10.1038/s42003-025-08632-5 (HO5-HO8); analysis paper van der Molen et al. 2025 Nat "
    "Neurosci doi:10.1038/s41593-025-02111-0."
)

#: The three readings of a spike-time unit gate (d) tells apart: multiplier to seconds.
TIME_READINGS = {"seconds": 1.0, "milliseconds": 1e-3, "samples_20khz": 1.0 / 20000.0}

_CHUNK = 1 << 20
_SPAN_ROOT = "processing"


def _open_h5(path):
    """The one place h5py opens a file (a test patches it to prove gate (b) runs first)."""
    import h5py

    return h5py.File(str(path), "r")


# =============================================================================================
# The pin
# =============================================================================================
def load_pin(path=None):
    return json.loads(Path(path or PIN_PATH).read_text(encoding="utf-8"))


def pin_assets(pin=None):
    return list((pin or load_pin())["assets"])


def check_pin_against_freeze(pin, freeze):
    """The tracked pin must equal the freeze-time census on every path and sha256 (and on size, to
    the census's 0.1 MB precision: it stored sizes only as rounded MB). A difference is a STOP and
    never an edit to make them agree. `freeze` is the census list ({path, MB, sha})."""
    fz = {f["path"]: f for f in freeze}
    got = {a["path"]: a for a in pin["assets"]}
    diffs = []
    for p in sorted(set(fz) | set(got)):
        if p not in fz or p not in got:
            diffs.append(f"{p}: in {'pin' if p in got else 'freeze'} only")
            continue
        if fz[p]["sha"] != got[p]["sha256"]:
            diffs.append(f"{p}: sha256 {got[p]['sha256'][:12]} != freeze {fz[p]['sha'][:12]}")
        if abs(got[p]["size"] / 1e6 - fz[p]["MB"]) > 0.051:
            diffs.append(f"{p}: size {got[p]['size']} B vs freeze {fz[p]['MB']} MB")
    if diffs:
        raise IngestRefused(f"pin != freeze census on {len(diffs)} item(s): {diffs[:3]}")
    return len(got)


def session_id(path):
    """`sub-HO2/sub-HO2_ses-20250916T190927_obj-1s1jcwl.nwb` -> `20250916T190927_obj-1s1jcwl`."""
    m = re.search(r"_ses-(.+?)\.nwb$", str(path))
    if not m:
        raise IngestRefused(f"cannot read a session id from {path!r}")
    return m.group(1)


def local_name(entry):
    return entry["path"].replace("/", "__")


def artifact_name(entry):
    return f"og1_{entry['subject']}_{session_id(entry['path'])}.npz"


# =============================================================================================
# Gates
# =============================================================================================
def license_id(pin):
    """DANDI's SPDX string (`spdx:CC-BY-4.0`) -> the allowlist's form (`cc-by-4.0`); '' if absent."""
    raw = pin.get("license") or ""
    return str(raw).split(":")[-1].strip().lower()


def gate_license(pin):
    """(a) The pin's license passes the tree's deny-by-default allowlist."""
    lid = license_id(pin)
    if not license_ok(lid):
        raise IngestRefused(
            f"(a) license {pin.get('license')!r} refused by ingest_guard.license_ok"
        )
    return lid


def stream_sha256(path):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(_CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def gate_bytes(path, entry):
    """(b) Streamed sha256 and size equal the pin. Runs BEFORE h5py opens the file."""
    sha, n = stream_sha256(path)
    if n != entry["size"] or sha != entry["sha256"]:
        raise IngestRefused(
            f"(b) {Path(path).name}: {n} B sha256 {sha[:16]} != pinned {entry['size']} B "
            f"{entry['sha256'][:16]}"
        )
    return sha


def read_units(h5):
    """The NWB units table's two datasets, as float64 / int64 (the file may store the index as
    uint16, which silently wraps past 65535 spikes -- gate (c) turns a wrap into a refusal)."""
    import numpy as np

    if (
        "units" not in h5
        or "spike_times" not in h5["units"]
        or "spike_times_index" not in h5["units"]
    ):
        raise IngestRefused("(c) no units/spike_times + units/spike_times_index in the file")
    return (
        np.asarray(h5["units/spike_times"][()], dtype=np.float64),
        np.asarray(h5["units/spike_times_index"][()], dtype=np.int64),
    )


def read_positions(h5, n_units):
    """(c) The units table's `x_pos` / `y_pos` as float64, one finite value per unit in table order.
    A file whose units table lacks either column is refused (Amendment A4.8)."""
    import numpy as np

    out = []
    for key in ("x_pos", "y_pos"):
        if "units" not in h5 or key not in h5["units"]:
            raise IngestRefused(f"(c) the units table has no {key}: the census needs positions")
        v = np.asarray(h5["units"][key][()], dtype=np.float64)
        if v.shape != (n_units,):
            raise IngestRefused(f"(c) {key} has shape {v.shape}, expected ({n_units},)")
        if not bool(np.all(np.isfinite(v))):
            raise IngestRefused(f"(c) {key} holds a non-finite value")
        out.append(v)
    return out[0], out[1]


def order_counts(spike_times, index):
    """`(dup_timestamps, out_of_order)`: within-unit consecutive equal steps and decreasing steps. The
    step across a unit boundary is not a within-unit step."""
    import numpy as np

    if len(spike_times) < 2:
        return 0, 0
    d = np.diff(spike_times)
    keep = np.ones(len(d), dtype=bool)
    edges = np.asarray(index[:-1], dtype=np.int64)
    keep[edges[(edges > 0) & (edges < len(spike_times))] - 1] = False
    return int(np.count_nonzero((d == 0) & keep)), int(np.count_nonzero((d < 0) & keep))


def gate_index(spike_times, index):
    """(c) At least one unit; index non-decreasing; last entry == len(spike_times); every spike time
    finite and >= 0."""
    import numpy as np

    n = len(index)
    if n < 1:
        raise IngestRefused("(c) zero units (the silently-empty trap)")
    if n > 1 and bool(np.any(np.diff(index) < 0)):
        raise IngestRefused("(c) spike_times_index decreases")
    if int(index[0]) < 0:
        raise IngestRefused("(c) spike_times_index starts negative")
    if int(index[-1]) != len(spike_times):
        raise IngestRefused(
            f"(c) last index {int(index[-1])} != len(spike_times) {len(spike_times)} "
            "(truncated or wrapped index)"
        )
    if not bool(np.all(np.isfinite(spike_times))):
        raise IngestRefused("(c) a spike time is not finite")
    if len(spike_times) and float(spike_times.min()) < 0.0:
        raise IngestRefused(f"(c) a spike time is negative ({float(spike_times.min()):.6g} s)")
    return n


def recorded_span(h5):
    """`(duration_s, where)` from the file's own time series: (bins + 0.5) / rate of the first
    `processing/*/*` series that has a `data` and a `starting_time.rate`. Refuses when none."""
    root = h5.get(_SPAN_ROOT)
    if root is not None:
        for mod in root:
            for name in root[mod]:
                grp = root[mod][name]
                if "data" in grp and "starting_time" in grp:
                    rate = grp["starting_time"].attrs.get("rate")
                    if rate:
                        n = int(grp["data"].shape[0])
                        t0 = float(grp["starting_time"][()])
                        return (
                            (n + 0.5) / float(rate),
                            f"{_SPAN_ROOT}/{mod}/{name}",
                            t0,
                            float(rate),
                        )
    raise IngestRefused("(d) the file records no recording span to compare against")


def gate_time_units(spike_times, duration_s):
    """(d) Of {seconds, milliseconds, samples at 20 kHz}, the declared seconds reading puts the last
    spike STRICTLY closest to the recorded span, and no spike lies at or past the span. Returns every
    reading's distance."""
    if len(spike_times) == 0:
        raise IngestRefused("(d) no spikes: nothing to compare to the recorded span")
    last = float(spike_times.max())
    dist = {k: abs(last * m - duration_s) for k, m in TIME_READINGS.items()}
    best = min(dist, key=dist.get)
    if best != "seconds" or sum(v == dist["seconds"] for v in dist.values()) > 1:
        raise IngestRefused(
            f"(d) last spike {last:.6g} fits {best!r}, not seconds, against span {duration_s:.6g} s "
            f"(distances {dist})"
        )
    if last >= duration_s:
        raise IngestRefused(
            f"(d) last spike {last:.6g} s is at or past the recorded span {duration_s:.6g} s"
        )
    return {"last_spike": last, "distance_s": dist}


#: The row a spike lands in, from x = (t - starting_time) x rate. "declared" is Amendment A2.1: the
#: depositors' MATLAB `raster(round(t x rate), u) = 1` with 1-based rows, i.e. floor(x - 0.5); the
#: other three are the alternatives the extremum rule must find strictly worse.
ROW_RULES = ("declared", "floor", "round", "floor_plus_half")


def _rows(x, mode):
    import numpy as np

    if mode not in ROW_RULES:
        raise ValueError(f"unknown row rule {mode!r}")
    return {
        "declared": lambda: np.floor(x - 0.5),
        "floor": lambda: np.floor(x),
        "round": lambda: np.rint(x),
        "floor_plus_half": lambda: np.floor(x + 0.5),
    }[mode]()


def rebuild_raster(spikes, index, n_bins, rate, t0=0.0, mode="declared"):
    """The depositors' binary raster rebuilt from the units table: bool (n_bins, n_units), cell 1 iff
    the unit has >= 1 spike in the row. `mode` picks the row rule from `ROW_RULES`; the declared one is
    floor((t - t0) x rate - 0.5). Spikes whose row falls outside [0, n_bins) (a row below 0 is a spike
    before 0.5 ms) are not drawn, as the stored raster does not draw them."""
    import numpy as np

    n_units = len(index)
    unit = np.repeat(np.arange(n_units), np.diff(np.concatenate([[0], index])))
    x = (spikes - t0) * rate
    b = _rows(x, mode).astype(np.int64)
    keep = (b >= 0) & (b < n_bins)
    out = np.zeros((n_bins, n_units), dtype=bool)
    out[b[keep], unit[keep]] = True
    return out


def gate_reproduce(spikes, index, stored, t0, rate, max_cells=10):
    """(e') The stored raster equals the declared rebuild EXACTLY, and plain floor, round-half-even,
    floor(x + 0.5) and a one-column unit-order shift (either way) each agree strictly less. Returns
    every agreement; refuses otherwise."""
    import numpy as np

    n_bins, n_cols = stored.shape
    if n_cols != len(index):
        raise IngestRefused(
            f"(e') raster has {n_cols} columns but the units table has {len(index)}"
        )
    stored = stored != 0
    declared = rebuild_raster(spikes, index, n_bins, rate, t0, "declared")

    def frac(a):
        return 1.0 - float(np.count_nonzero(a != stored)) / stored.size

    bad = np.argwhere(declared != stored)
    agree = {
        "declared": frac(declared),
        "floor": frac(rebuild_raster(spikes, index, n_bins, rate, t0, "floor")),
        "round": frac(rebuild_raster(spikes, index, n_bins, rate, t0, "round")),
        "floor_plus_half": frac(rebuild_raster(spikes, index, n_bins, rate, t0, "floor_plus_half")),
        "shift_plus1": frac(np.roll(declared, 1, axis=1)),
        "shift_minus1": frac(np.roll(declared, -1, axis=1)),
    }
    readings = {
        "agreement": agree,
        "mismatch_cells": int(len(bad)),
        "mismatch_fraction": 1.0 - agree["declared"],
    }
    if len(bad):
        raise IngestRefused(
            f"(e') rebuilt raster != stored in {len(bad)} of {stored.size} cells "
            f"(fraction {readings['mismatch_fraction']:.3g}); first (bin, col): "
            f"{[tuple(int(v) for v in c) for c in bad[:max_cells]]}",
            readings,
        )
    rivals = ("floor", "round", "floor_plus_half", "shift_plus1", "shift_minus1")
    worse = [k for k in rivals if not agree[k] < agree["declared"]]
    if worse:
        raise IngestRefused(
            f"(e') the convention is not decided by the data: {worse} agree as well as the declared rule ({agree})",
            readings,
        )
    return readings


def device_strings(h5):
    """Every device-like string the file carries (`general/devices`, `general/extracellular_ephys`).
    The freeze records the HO5-HO8 device as 'MaxTwo, secondhand'; a mismatch is disclosed here,
    never a refusal."""
    out = []
    gen = h5.get("general")
    if gen is not None:
        for k in ("devices", "extracellular_ephys"):
            if k in gen:
                out += [f"{k}/{n}" for n in gen[k]]
    return out


# =============================================================================================
# Ingest one file / all files
# =============================================================================================
def ingest_file(path, entry, pin, out_dir=None):
    """Gates (a)-(d) on one pinned file, then (and only then) the npz. Returns the readings dict.
    A refusal raises `IngestRefused` carrying the readings gathered so far; nothing is written."""
    import numpy as np

    readings = {"file": entry["path"], "subject": entry["subject"]}
    try:
        readings["a_license"] = gate_license(pin)
        readings["source_sha256"] = gate_bytes(path, entry)
        h5 = _open_h5(path)
        try:
            spikes, index = read_units(h5)
            readings["n_units"] = gate_index(spikes, index)
            x_pos, y_pos = read_positions(h5, readings["n_units"])
            dur, where, t0, rate = recorded_span(h5)
            readings.update(duration_s=dur, span_from=where, span_start_s=t0, span_rate_hz=rate)
            readings["d_time_units"] = gate_time_units(spikes, dur)
            readings["e_reproduce"] = gate_reproduce(
                spikes, index, np.asarray(h5[where + "/data"][()]), t0, rate
            )
            readings["devices"] = device_strings(h5) or "none recorded in the file"
            readings["institution"] = _text(h5, "general/institution")
        finally:
            h5.close()
    except IngestRefused as e:
        raise IngestRefused(str(e), readings) from None
    out = Path(out_dir or ".") / artifact_name(entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        spike_times=spikes,
        spike_times_index=index,
        x_pos=x_pos,
        y_pos=y_pos,
        duration_s=np.float64(readings["duration_s"]),
        subject=np.array(entry["subject"]),
        session=np.array(session_id(entry["path"])),
        source_sha256=np.array(readings["source_sha256"]),
    )
    readings["artifact"] = out.name
    return readings


def _text(h5, key):
    try:
        v = h5[key][()]
        return v.decode("utf-8") if isinstance(v, bytes) else str(v)
    except KeyError:
        return None


def convert_all(data_dir, pin=None, only=None):
    """Run the gates on every pinned file under `<data_dir>/nwb`, write the npz beside it. A refused
    file is recorded with its reason and no artifact; the run continues so every file is reported."""
    pin = pin or load_pin()
    data_dir = Path(data_dir)
    rows = []
    for entry in pin["assets"]:
        if only and entry["subject"] not in only:
            continue
        try:
            rows.append(
                {
                    "outcome": "ok",
                    **ingest_file(data_dir / "nwb" / local_name(entry), entry, pin, data_dir),
                }
            )
        except IngestRefused as e:
            rows.append({"outcome": "REFUSED", "reason": str(e), **e.readings})
    return rows


def session_manifest(out_dir, rows, pin=None):
    """A1.2: `og1_sessions.json` beside the npz. Per subject, sessions sorted by id; a file whose
    decoded arrays equal a kept file's is a duplicate of it (`duplicate_of` = that kept session id).
    Counts that agree with different arrays stay DISTINCT. Only files that converted are listed."""
    import numpy as np

    pin = pin or load_pin()
    split = {a["subject"]: a["split"] for a in pin["assets"]}
    out_dir = Path(out_dir)
    ok = sorted(
        (r for r in rows if r["outcome"] == "ok"), key=lambda r: (r["subject"], r["artifact"])
    )
    kept = {}
    manifest = []
    for r in ok:
        with np.load(out_dir / r["artifact"], allow_pickle=False) as z:
            st, ix = z["spike_times"], z["spike_times_index"]
            session = str(z["session"])
        dups, ooo = order_counts(st, ix)
        dup = None
        for k_session, k_st, k_ix in kept.setdefault(r["subject"], []):
            if np.array_equal(st, k_st) and np.array_equal(ix, k_ix):
                dup = k_session
                break
        if dup is None:
            kept[r["subject"]].append((session, st, ix))
        manifest.append(
            {
                "npz": r["artifact"],
                "subject": r["subject"],
                "session": session,
                "split": split[r["subject"]],
                "source_sha256": r["source_sha256"],
                "units": r["n_units"],
                "duration_s": r["duration_s"],
                "duplicate_of": dup,
                "dup_timestamps": dups,
                "out_of_order": ooo,
            }
        )
    (out_dir / "og1_sessions.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    return manifest


# =============================================================================================
# Synthetic fixtures (shared with tests/test_organoid_ingest.py) and the self-check
# =============================================================================================
#: The default fixture units: sub-ms fractions so floor and round bins differ.
_FIXTURE_UNITS = [[0.1006, 0.5004, 1.2007], [0.3009, 1.9003], [0.7002]]


def write_fixture_nwb(
    path, units=None, bins=2000, rate=1000.0, index=None, span=True, raster=None, positions=True
):
    """A small NWB-shaped HDF5 file. `units` = list of per-unit spike-time lists (seconds). The stored
    raster is the declared-convention rebuild of the TRUE units unless `raster` is given. `positions`
    writes the units table's x_pos / y_pos (unit i at (100 i, 50 i)); False leaves them out."""
    import h5py
    import numpy as np

    units = units if units is not None else _FIXTURE_UNITS
    flat = np.array([t for u in units for t in u], dtype=np.float64)
    idx = (
        np.cumsum([len(u) for u in units]).astype(np.int64) if index is None else np.asarray(index)
    )
    with h5py.File(str(path), "w") as f:
        f.create_dataset("units/spike_times", data=flat)
        f.create_dataset("units/spike_times_index", data=idx)
        f.create_dataset("general/institution", data=np.bytes_("Fixture University"))
        if positions:
            f.create_dataset("units/x_pos", data=100.0 * np.arange(len(units)))
            f.create_dataset("units/y_pos", data=50.0 * np.arange(len(units)))
        if span:
            if raster is None:
                true_idx = np.cumsum([len(u) for u in units]).astype(np.int64)
                with np.errstate(invalid="ignore"):  # a NaN fixture time draws no cell
                    raster = rebuild_raster(flat, true_idx, bins, rate).astype("u1")
            f.create_dataset("processing/curated_binning/t_spk_mat/data", data=raster)
            st = f.create_dataset("processing/curated_binning/t_spk_mat/starting_time", data=0.0)
            st.attrs["rate"] = rate
    return Path(path)


def fixture_entry(path, subject="HO1", name="sub-HO1/sub-HO1_ses-20250101T000000.nwb"):
    sha, n = stream_sha256(path)
    return {"path": name, "subject": subject, "size": n, "sha256": sha}


def fixture_pin(entry, license_="spdx:CC-BY-4.0"):
    return {"dandiset": "000000", "version": "fixture", "license": license_, "assets": [entry]}


def _selfcheck():
    """Every gate's constructed red, then a green control, on synthetic fixtures. No download."""
    import tempfile

    import numpy as np

    failures = []

    def must_refuse(label, gate, needle):
        try:
            gate()
        except IngestRefused as e:
            if needle not in str(e):
                failures.append(f"{label}: refused for the wrong reason: {e}")
            return
        failures.append(f"{label}: accepted the constructed red")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = write_fixture_nwb(td / "good.nwb")
        entry = fixture_entry(good)
        pin = fixture_pin(entry)
        rows = ingest_file(good, entry, pin, td / "out")
        with np.load(td / "out" / artifact_name(entry), allow_pickle=False) as z:
            if not (
                z["spike_times"].size == 6
                and int(z["spike_times_index"][-1]) == 6
                and float(z["duration_s"]) == 2.0005
            ):
                failures.append("green control: artifact does not round-trip")
            if not (
                list(z["x_pos"]) == [0.0, 100.0, 200.0]
                and list(z["y_pos"]) == [0.0, 50.0, 100.0]
                and z["x_pos"].dtype == np.float64
            ):
                failures.append("green control: positions do not round-trip")
        if (
            rows["d_time_units"]["distance_s"]["seconds"]
            >= rows["d_time_units"]["distance_s"]["milliseconds"]
        ):
            failures.append("green control: seconds not closest")

        # (a) license empty
        must_refuse("(a)", lambda: gate_license(fixture_pin(entry, "")), "(a)")
        # (b) one byte flipped
        raw = bytearray(good.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        flipped = td / "flipped.nwb"
        flipped.write_bytes(bytes(raw))
        must_refuse("(b)", lambda: gate_bytes(flipped, entry), "(b)")
        # (c) truncated index
        bad_c = write_fixture_nwb(td / "c.nwb", index=[3, 5, 5])
        must_refuse(
            "(c)",
            lambda: ingest_file(bad_c, fixture_entry(bad_c), fixture_pin(fixture_entry(bad_c)), td),
            "(c)",
        )
        # (c) a NaN time, a negative time
        for label, units in (
            ("(c nan)", [[0.1006, float("nan")], [0.3009]]),
            ("(c neg)", [[-0.4, 0.5004], [0.3009]]),
        ):
            bad = write_fixture_nwb(td / f"{label[3:6]}.nwb", units=units)
            e_bad = fixture_entry(bad)
            must_refuse(
                label,
                lambda b=bad, e=e_bad: ingest_file(b, e, fixture_pin(e), td),
                "(c)",
            )
        # (c) a units table without positions
        nopos = write_fixture_nwb(td / "nopos.nwb", positions=False)
        must_refuse(
            "(c pos)",
            lambda: ingest_file(nopos, fixture_entry(nopos), fixture_pin(fixture_entry(nopos)), td),
            "no x_pos",
        )
        # (d) spike times x 1000
        bad_d = write_fixture_nwb(
            td / "d.nwb", units=[[100.0, 500.0, 1200.0], [300.0, 1900.0], [700.0]]
        )
        must_refuse(
            "(d)",
            lambda: ingest_file(bad_d, fixture_entry(bad_d), fixture_pin(fixture_entry(bad_d)), td),
            "(d)",
        )
        # (d) spikes up to 1.9 s in a 1.0 s span
        late = write_fixture_nwb(td / "late.nwb", units=[[0.1006, 1.9003], [0.3009]], bins=1000)
        must_refuse(
            "(d span end)",
            lambda: ingest_file(late, fixture_entry(late), fixture_pin(fixture_entry(late)), td),
            "at or past",
        )
        # (d) no recorded span
        nospan = write_fixture_nwb(td / "n.nwb", span=False)
        must_refuse(
            "(d span)",
            lambda: ingest_file(
                nospan, fixture_entry(nospan), fixture_pin(fixture_entry(nospan)), td
            ),
            "(d)",
        )
        # (e') times x 1000 against the true raster; unit-index off-by-one; one planted bin flip
        fl = np.array([t for u in _FIXTURE_UNITS for t in u])
        ix = np.cumsum([len(u) for u in _FIXTURE_UNITS])
        true_r = rebuild_raster(fl, ix, 2000, 1000.0).astype("u1")
        must_refuse(
            "(e' x1000)", lambda: gate_reproduce(fl * 1000, ix, true_r, 0.0, 1000.0), "(e')"
        )
        must_refuse(
            "(e' col)",
            lambda: gate_reproduce(fl, ix, np.roll(true_r, 1, axis=1), 0.0, 1000.0),
            "(e')",
        )
        flip = true_r.copy()
        flip[777, 1] ^= 1
        must_refuse("(e' flip)", lambda: gate_reproduce(fl, ix, flip, 0.0, 1000.0), "(777, 1)")
        # duplicates: identical arrays -> duplicate; one spike moved by 1e-9 -> distinct
        dup_rows = []
        for sess, units in (
            ("20250101T000001", _FIXTURE_UNITS),
            ("20250101T000002", _FIXTURE_UNITS),
            (
                "20250101T000003",
                [
                    [t + 1e-9 if (i, j) == (0, 0) else t for j, t in enumerate(u)]
                    for i, u in enumerate(_FIXTURE_UNITS)
                ],
            ),
        ):
            q = write_fixture_nwb(td / f"{sess}.nwb", units=units)
            e2 = fixture_entry(q, name=f"sub-HO1/sub-HO1_ses-{sess}.nwb")
            dup_rows.append({"outcome": "ok", **ingest_file(q, e2, fixture_pin(e2), td / "dup")})
        man = session_manifest(
            td / "dup", dup_rows, {"assets": [{"subject": "HO1", "split": "dev"}]}
        )
        if [m["duplicate_of"] for m in man] != [None, "20250101T000001", None]:
            failures.append(f"duplicates: wrong grouping {[m['duplicate_of'] for m in man]}")
        if order_counts(np.array([1.0, 1.0, 0.5, 2.0, 0.1]), np.array([4, 5])) != (1, 1):
            failures.append("order_counts: wrong count on a planted unit")
        # pin != freeze
        freeze = [{"path": entry["path"], "MB": entry["size"] / 1e6, "sha": "0" * 64}]
        must_refuse("pin", lambda: check_pin_against_freeze(pin, freeze), "pin != freeze")
        if list((td / "out").glob("*.npz")) != [td / "out" / artifact_name(entry)]:
            failures.append("a refusal wrote an artifact")
    if failures:
        for f in failures:
            print(f"organoid_ingest self-check FAILED: {f}")
        return 1
    print(
        "organoid_ingest self-check OK: reds (a)(b)(c)(c nan, negative, no positions)(d)(d-span)(d past span)(e' x1000, column shift, flip)(pin) "
        "refuse, green control converts, duplicate rule groups"
    )
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--convert"] and len(argv) == 2:
        rows = convert_all(argv[1])
        session_manifest(argv[1], rows)
        print(json.dumps(rows, indent=1, default=str))
        return 0 if all(r["outcome"] == "ok" for r in rows) else 1
    return _selfcheck()


if __name__ == "__main__":
    sys.exit(main())
