"""
SIGNAL CALIBRATOR -- distil a live scalar signal's operating point from the brain's OWN operation.

Step-0 of honest self-recalibration (the first real DataIngestSource): record a named live neural signal
(e.g. the salience drive fed to LocusCoeruleus each tick) as a running P2 QUANTILE
(substrate/streaming_quantile.P2Quantile -- 5 markers, O(1) memory, NEVER stores the raw observations),
persist the full estimator state so it RESUMES across restarts (the always-on daemon restarts on every new
build), and expose the distilled {p, value, n}. reasoning/brain_native_selfbuild.ThresholdRecalibrationSource
reads that distilled operating point and proposes replacing a hand-guessed threshold with the data value,
gate-verified.

HONEST: this is plumbing, not learning. It records the brain's own signal distribution and reports a
percentile of it -- a data operating point. Whether recalibrating a threshold TO that point is an
improvement is decided elsewhere (the gate certifies compatibility, not correctness). Deterministic
(P2 has no RNG); persistence is a plain JSON of the estimator markers; ASCII-only prints.

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
        """Atomically persist a list of SignalCalibrators to `path` (JSON: name -> P2 state)."""
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
