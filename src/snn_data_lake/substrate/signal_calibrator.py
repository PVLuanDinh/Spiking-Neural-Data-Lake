"""
SIGNAL CALIBRATOR -- distil a live scalar signal's operating point from the brain's OWN operation.

Record a named live neural signal (e.g. the salience drive fed to LocusCoeruleus each tick) as a running
P2 QUANTILE (substrate/streaming_quantile.P2Quantile -- 5 markers, O(1) memory, NEVER stores the raw
observations), persist the full estimator state so it RESUMES across restarts (the always-on daemon
restarts on every new build), and expose the distilled {p, value, n}.

Primary consumer (adopt-better): brain.py _w8_tick reads the distilled quantile and CONSUMES IT LIVE into
the LC salience gate every tick (self._lc.salience_th = result value, when the quantile is valid), so the
threshold is baseline-relative at runtime rather than a constant kept fixed. The older path --
reasoning/brain_native_selfbuild.ThresholdRecalibrationSource freezing the value into the source as a
literal -- is the inferior, superseded mechanism (it remains only as a target-less extension point).

HONEST: this is plumbing, not learning. It records the brain's own signal distribution and reports a
percentile of it -- a data operating point. Whether recalibrating a threshold TO that point is an
improvement is decided elsewhere (the gate certifies compatibility, not correctness). Deterministic
(P2 has no RNG); ASCII-only prints.

OPEN P4b VIOLATION -- STATED HERE BECAUSE IT LIVES HERE (found 2026-08-04, STILL OPEN 2026-08-05,
and it is now the LAST JSON member of a family that has otherwise been closed).
`save_all` writes `json.dump` to a path the brain gives it as `<home>/signal_calib.json`: a SIDECAR
FILE beside `brain_lake.bin`, holding LEARNED state. P4b names that in its own words -- "what the
brain learns goes IN the lake... not a sidecar file, not a JSON blob" -- and the state is not inert:
`cortex/brain.py` restores it at construction and CONSUMES it on the live wake tick (`self._lc.
salience_th = float(r["value"])`), so a running-baseline value that ARBITRATES the LC salience gate
is kept outside the one lake. It is the same defect `data/selfbuild_state.json` was, with the same
two costs: a second file that does not follow NORD_HOME, and a store `consolidate()` cannot see.

WHY IT IS STILL NOT FIXED, restated after a second attempt on 2026-08-05, because the reason has
CHANGED and the new one is sharper. The lake-side channel is no longer a bespoke `_T_CALIB` pair --
that was retired unwritten -- it is `home.save_named_state("signal_calib.json", blob)` /
`load_named_state(...)`, the same generic record the six other sidecars moved onto, and the second
of those already imports this exact 950-byte file on first read. So the destination is built,
proven cross-process, and free.

WHAT BLOCKS IT IS THE SIGNATURE, and specifically that EVERY method here takes an explicit `path`:
  * `cortex/brain.py` passes `state_path("signal_calib.json")` at both call sites, and BOTH sit
    inside bare `except Exception: pass`. A signature change made from this file alone would not
    fail loudly -- it would SILENTLY stop persisting and cold-start the LC gate every boot.
  * `basal_ganglia/brain_native_selfbuild.py`'s `ThresholdRecalibrationSource` self-check passes a
    TEMP `calib_path` and asserts an honest-null "with NO recorded data". Ignoring that path and
    reading the one lake instead would (a) make that assert contaminable by a `probe_th` record left
    by a previous run and (b) write probe records INTO the operator's real 516 MB brain_lake.bin
    during a gate run. Dropping the argument instead makes `save_all(calib_path, [sc])` pass a
    string where the calibrator list is expected -- a loud break in another module's self-check.
Both files are outside this one's area. The honest disposition is therefore: destination built and
proven, call sites owed, and this paragraph is the label until they land. The edit, for whoever owns
them, is `save_all(calibrators)` / `load_into(calibrators)` / `load_result(name)` here, the two
`brain.py` sites dropping their `state_path(...)` argument, and `brain_native_selfbuild`'s
self-check setting an isolated NORD_HOME instead of a temp `calib_path` -- that last one is not
optional, it is where this change goes wrong if it is applied in pieces.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.signal_calibrator
"""

import json
import os

from snn_data_lake.substrate.streaming_quantile import P2Quantile

# the P2Quantile estimator-state attributes that fully define its running estimate (all JSON-serializable)
_STATE_ATTRS = ("p", "_buf", "_q", "_np", "_npd", "_dnp", "_n_seen")


def _snapshot_p2(q):
    return {a: getattr(q, a) for a in _STATE_ATTRS}


def _restore_p2(d):
    q = P2Quantile(float(d["p"]))
    for a in _STATE_ATTRS:
        if a in d:
            setattr(q, a, d[a])
    return q


class SignalCalibrator:
    """One named live signal -> a persisted running P2 quantile of its recorded distribution.

    observe(x) folds one live sample; result() -> {p, value, n} (value None until >=1 sample); snapshot()/
    load() serialize the FULL P2 state so accumulation survives a restart. O(1) memory -- raw observations
    are never stored."""

    def __init__(self, name, p=0.75):
        self.name = str(name)
        self.q = P2Quantile(p)

    def observe(self, x):
        self.q.add(float(x))

    def result(self):
        n = self.q.n_seen
        return {"p": self.q.p, "value": (self.q.quantile() if n >= 1 else None), "n": n}

    def snapshot(self):
        return _snapshot_p2(self.q)

    def load(self, d):
        if d:
            self.q = _restore_p2(d)
        return self

    # ---- durable multi-signal persistence (a plain JSON keyed by signal name) ----
    @staticmethod
    def save_all(path, calibrators):
        """Atomically persist a list of SignalCalibrators to `path` (JSON: name -> P2 state).

        THIS IS THE P4b VIOLATION the module docstring names: a sidecar JSON holding learned state.
        The replacement is `home.save_named_state("signal_calib.json", blob)` (tag `_T_STATE`, the
        one record type every other former sidecar now uses, and it migrates this file on first
        read). Switching to it is a coordinated edit with `cortex/brain.py` and
        `basal_ganglia/brain_native_selfbuild.py` -- see the module docstring for why the `path`
        argument, not the destination, is what is blocking."""
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({c.name: c.snapshot() for c in calibrators}, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @staticmethod
    def _load_raw(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    @staticmethod
    def load_into(path, calibrators):
        """Restore each calibrator's state from `path` (missing names are left fresh)."""
        raw = SignalCalibrator._load_raw(path)
        for c in calibrators:
            if c.name in raw:
                c.load(raw[c.name])
        return calibrators

    @staticmethod
    def load_result(path, name):
        """Read the distilled {p, value, n} for `name` from `path` WITHOUT a live object -- the read path
        the DataIngestSource uses. Returns None if the file/name is absent or has no samples yet."""
        d = SignalCalibrator._load_raw(path).get(name)
        if not d:
            return None
        q = _restore_p2(d)
        n = q.n_seen
        if n < 1:
            return None
        try:
            return {"p": q.p, "value": q.quantile(), "n": n}
        except ValueError:
            return None


# ------------------------------------------------------------------- self-check -------------------------
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def main():
    import tempfile

    width = 92
    _emit("=" * width)
    _emit("SIGNAL CALIBRATOR -- durable running-quantile of a live signal (P2, O(1), resumable)")
    _emit("=" * width)

    # (1) a running quantile approximates the true quantile of a known distribution
    c = SignalCalibrator("salience_th", p=0.75)
    data = [i / 1000.0 for i in range(1001)]  # uniform [0,1], true P75 = 0.75
    for x in data:
        c.observe(x)
    r = c.result()
    assert r["n"] == 1001, r
    assert abs(r["value"] - 0.75) < 0.03, f"P75 estimate off: {r['value']}"
    _emit(
        f"  [OK] P75 of uniform[0,1] over {r['n']} samples -> {r['value']:.4f} (true 0.75); O(1) memory"
    )

    # (2) snapshot -> save_all -> load_result reproduces the estimate durably (resume across restart)
    work = tempfile.mkdtemp(prefix="signal_calibrator_selfcheck_")
    try:
        path = os.path.join(work, "signal_calib.json")
        SignalCalibrator.save_all(path, [c])
        got = SignalCalibrator.load_result(path, "salience_th")
        assert got is not None and abs(got["value"] - r["value"]) < 1e-9 and got["n"] == 1001, got
        _emit(
            f"  [OK] persisted + read back durably: value {got['value']:.4f} n={got['n']} (byte-stable)"
        )

        # (3) load_into RESUMES accumulation: a fresh calibrator restored from disk + more data keeps going
        c2 = SignalCalibrator("salience_th", p=0.75)
        SignalCalibrator.load_into(path, [c2])
        assert c2.result()["n"] == 1001, "restore must recover the sample count"
        for x in data:
            c2.observe(x)  # fold the same distribution again
        assert c2.result()["n"] == 2002 and abs(c2.result()["value"] - 0.75) < 0.03
        _emit(
            f"  [OK] resume: restored n=1001 then folded 1001 more -> n={c2.result()['n']}, P75 stable"
        )

        # (4) honest empties: no file / unknown signal / zero samples -> None (never a fabricated value)
        assert SignalCalibrator.load_result(os.path.join(work, "nope.json"), "x") is None
        assert SignalCalibrator.load_result(path, "unknown_signal") is None
        SignalCalibrator.save_all(path, [SignalCalibrator("fresh", p=0.5)])
        assert SignalCalibrator.load_result(path, "fresh") is None, (
            "0 samples -> None, no fabricated value"
        )
        _emit(
            "  [OK] honest empties: absent file / unknown signal / 0 samples -> None (no fabrication)"
        )
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)

    _emit("-" * width)
    _emit(
        "self-check OK: records a live signal's distribution as a resumable P2 quantile (O(1), no raw"
    )
    _emit(
        "obs stored), persists + reads back durably; honest None when there is no data. Plumbing only --"
    )
    _emit(
        "distils a data operating point; whether to recalibrate a threshold to it is judged by the gate."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
