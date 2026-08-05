"""
SIGNAL CALIBRATOR -- distil a live scalar signal's operating point from the brain's OWN operation.

Record a named live neural signal (e.g. the salience drive fed to LocusCoeruleus each tick) as a running
P2 QUANTILE (substrate/streaming_quantile.P2Quantile -- 5 markers, O(1) memory, NEVER stores the raw
observations), persist the full estimator state IN THE ONE LAKE so it RESUMES across restarts (the
always-on daemon restarts on every new build), and expose the distilled {p, value, n}.

Primary consumer (adopt-better): `cortex/brain.py::Brain._tick_ensemble` reads the distilled quantile and
CONSUMES IT LIVE into the LC salience gate every tick (`self._lc.salience_th = float(r["value"])`, when
the quantile is valid), so the threshold is baseline-relative at runtime rather than a constant kept
fixed. *(Named `_w8_tick` here until 2026-08-05. `_w8_tick` is the CALLER; the assignment lives one frame
down in `_tick_ensemble`, which `_w8_tick` invokes. Cited by SYMBOL for the reason POLICIES.md gives.)*
The older path -- `basal_ganglia/brain_native_selfbuild.ThresholdRecalibrationSource` freezing the value
into the source as a literal -- is the inferior, superseded mechanism (a target-less extension point).

HONEST: this is plumbing, not learning. It records the brain's own signal distribution and reports a
percentile of it -- a data operating point. Whether recalibrating a threshold TO that point is an
improvement is decided elsewhere (the gate certifies compatibility, not correctness). Deterministic
(P2 has no RNG); ASCII-only prints.

P4b CLOSED 2026-08-05 -- this was the LAST JSON member of the sidecar family, and the one the other
SEVEN could not be fixed alongside. (Seven, re-counted from `home.py`'s `_T_STATE` tag comment rather
than inherited: recall_trace, time_cells, valence, gf_library, active_library, reasoning_curriculum,
nightly_skills. Those seven rode `save_state`, so fixing that ONE function moved all of them and no
caller changed; this module wrote its own `json.dump`, which is exactly why it did not come along.
Measured after this change: the only remaining `state_path(...)` WRITER under `src/` is
`cortex/brain.py`'s `language_acq.npz`, a numpy binary the JSON facade cannot carry and which
`home.save_state` already names as open debt -- plus `brainstem/signal_loop.py`'s `unknowns.jsonl`,
which is not JSON-facade state and is NOT dispositioned here. Every other `state_path` hit in `src/`
is a self-check asserting the sidecar is absent.) What it was: `save_all(path, calibrators)` wrote
`json.dump` to
`<home>/signal_calib.json`, a SIDECAR beside `brain_lake.bin`, holding LEARNED state that
`cortex/brain.py` restored at construction and CONSUMED on the live wake tick. P4b names that in its own
words -- "what the brain learns goes IN the lake under a record tag -- not a sidecar file, not a JSON
blob" -- and it carried the same two costs `data/selfbuild_state.json` did: a second file, and a store
`consolidate()` could not see.

WHAT IT IS NOW: one `_T_STATE` record under the name `signal_calib.json`, via
`home.save_named_state` / `home.load_named_state` -- the SAME generic record those seven former
sidecars ride, not a bespoke tag and pair (`_T_CALIB` was retired unwritten under P2 for exactly that
reason). `load_named_state` imports a pre-existing `<home>/signal_calib.json` into the lake on first
read and LEAVES THE FILE ON DISK, so a brain that had already learned an operating point keeps it.

WHY THE FIRST TWO ATTEMPTS STOPPED SHORT -- taken from what this docstring itself recorded on
2026-08-04 and 2026-08-05, and kept because the hazard is the interesting part and it is still there
for the next person: every one of the three consumer sites sits inside a bare `except Exception`, and
`brain.py`'s load handler additionally NULLS both calibrators. A half-applied signature change DOES NOT
ERROR -- it silently stops persisting, `n` resets to 0 every boot, the live `r["n"] >= 50` guard then
holds `self._lc.salience_th` at the frozen module default, and every dashboard reads healthy. The five
files move together or not at all: this one, `cortex/brain.py` (load at construction, consume on the
wake tick, save in `close()`), `basal_ganglia/brain_native_selfbuild.py`
(`ThresholdRecalibrationSource`), `cortex/metacognitive_monitor.py` (constructor only, no persistence)
and `tests/test_signal_calibrator_lake.py`, which did not exist before this change and is what makes the
silence audible.

`lake_path=None` means THE ONE LAKE. A path is an ISOLATED LAKE -- the same store elsewhere, which is
what a self-check needs; it is never a path to a sidecar and never a second store. (Wording and shape
copied from `BrainNativeSelfBuildProposer.__init__`'s `lake_path`.)

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.signal_calibrator
"""

import json
import os

from snn_data_lake.substrate.home import load_named_state, save_named_state
from snn_data_lake.substrate.streaming_quantile import P2Quantile

# The _T_STATE record name. It is spelled with the `.json` suffix ON PURPOSE: it is the name of the
# pre-P4b sidecar, and `home._migrate_sidecar` keys on exactly that to import `<home>/signal_calib.json`
# once, on first read. Renaming it would orphan every operator's recorded operating point.
_STATE_NAME = "signal_calib.json"

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

    # ---- durable multi-signal persistence (ONE _T_STATE record, keyed by signal name) ----
    @staticmethod
    def save_all(calibrators, lake_path=None):
        """Persist a list of SignalCalibrators INTO THE ONE LAKE as one `_T_STATE` record.

        The record is the WHOLE map (name -> P2 state), exactly as the retired sidecar file was, so
        callers must pass every calibrator they own -- a partial save supersedes the names it omits.
        `lake_path=None` is the one lake; a path is an isolated lake (a self-check's own store).

        Returns whatever `save_named_state` returns: True if a record was written, False if the blob
        was unchanged (the lake is append-only and `consolidate()` keeps every non-FACT record, so an
        identical state must not be re-appended). A False here is not a failure."""
        blob = json.dumps(
            {c.name: c.snapshot() for c in calibrators}, ensure_ascii=True, sort_keys=True
        ).encode("ascii")
        return save_named_state(_STATE_NAME, blob, lake_path)

    @staticmethod
    def _load_raw(lake_path=None):
        """The stored {name: P2 state} map, or {} when the brain never stored one.

        Every failure is an empty map rather than a raise, for the reason
        `BrainNativeSelfBuildProposer._load` states: a broken or absent record must never block
        construction. The callers here sit inside bare `except Exception` handlers that would swallow
        a raise anyway -- and one of them nulls the calibrators when it does, which is the silent
        cold-start this module's history is made of."""
        try:
            blob = load_named_state(_STATE_NAME, lake_path)
        except Exception:  # noqa: BLE001 - a broken record must never block construction
            return {}
        if not blob:
            return {}
        try:
            raw = json.loads(bytes(blob).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def load_into(calibrators, lake_path=None):
        """Restore each calibrator's state from the lake (missing names are left fresh)."""
        raw = SignalCalibrator._load_raw(lake_path)
        for c in calibrators:
            if c.name in raw:
                c.load(raw[c.name])
        return calibrators

    @staticmethod
    def load_result(name, lake_path=None):
        """Read the distilled {p, value, n} for `name` WITHOUT a live object -- the read path
        `basal_ganglia/brain_native_selfbuild.ThresholdRecalibrationSource.candidates` uses, and its
        ONLY caller in the tree. *(Said "the read path the DataIngestSource uses" until 2026-08-05;
        `DataIngestSource` is a NotImplemented extension point and has never called this.)*
        Returns None if the name is absent or has no samples yet."""
        d = SignalCalibrator._load_raw(lake_path).get(name)
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
    import shutil
    import subprocess
    import sys
    import tempfile

    width = 92
    _emit("=" * width)
    _emit("SIGNAL CALIBRATOR -- durable running-quantile of a live signal (P2, O(1), resumable)")
    _emit("=" * width)

    # (1) a running quantile approximates the true quantile of a known distribution (no persistence)
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

    # ISOLATION: every persistence property below writes to a lake, and the DEFAULT lake is the
    # operator's real 500+ MB brain_lake.bin. RESTORE-NOT-POP: capture NORD_HOME BEFORE the try (so the
    # finally can never NameError) and restore exactly it. A bare `os.environ.pop` here would delete the
    # CALLER's value -- tests/test_selfchecks.py runs every module's __main__ in ONE interpreter, and
    # that exact bug once sent ~17 self-checks at the operator's live brain (see home.py::main).
    _prior_home = os.environ.get("NORD_HOME")
    work = tempfile.mkdtemp(prefix="signal_calibrator_selfcheck_")
    try:
        os.environ["NORD_HOME"] = work
        src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        iso = os.path.join(work, "isolated_lake.bin")  # an isolated LAKE, not a sidecar

        # (2) save_all -> load_result reproduces the estimate durably (resume across restart)
        SignalCalibrator.save_all([c], iso)
        got = SignalCalibrator.load_result("salience_th", iso)
        assert got is not None and abs(got["value"] - r["value"]) < 1e-9 and got["n"] == 1001, got
        _emit(
            f"  [OK] persisted + read back durably: value {got['value']:.4f} n={got['n']} (byte-stable)"
        )

        # (3) load_into RESUMES accumulation: a fresh calibrator restored from the lake + more data
        c2 = SignalCalibrator("salience_th", p=0.75)
        SignalCalibrator.load_into([c2], iso)
        assert c2.result()["n"] == 1001, "restore must recover the sample count"
        for x in data:
            c2.observe(x)  # fold the same distribution again
        assert c2.result()["n"] == 2002 and abs(c2.result()["value"] - 0.75) < 0.03
        _emit(
            f"  [OK] resume: restored n=1001 then folded 1001 more -> n={c2.result()['n']}, P75 stable"
        )

        # (4) honest empties: empty lake / unknown signal / zero samples -> None (never fabricated)
        empty_lake = os.path.join(work, "empty_lake.bin")
        assert SignalCalibrator.load_result("salience_th", empty_lake) is None
        assert SignalCalibrator.load_result("unknown_signal", iso) is None
        SignalCalibrator.save_all([SignalCalibrator("fresh", p=0.5)], iso)
        assert SignalCalibrator.load_result("fresh", iso) is None, (
            "0 samples -> None, no fabricated value"
        )
        # ...and that write superseded the whole map, which is the sidecar's semantics kept exactly:
        assert SignalCalibrator.load_result("salience_th", iso) is None, (
            "save_all stores the WHOLE map; a partial save must supersede the names it omits"
        )
        _emit(
            "  [OK] honest empties: empty lake / unknown signal / 0 samples -> None (no fabrication)"
        )

        # (5) MIGRATION of the pre-P4b sidecar, on the DEFAULT lake, and the file is LEFT ON DISK.
        # This must run BEFORE (6): `home._migrate_sidecar` fires only while the lake holds no record
        # under this name, so a lake write first would make the migration structurally unreachable.
        legacy = os.path.join(work, _STATE_NAME)
        leg = SignalCalibrator("legacy_sig", p=0.75)
        for x in data:
            leg.observe(x)
        with open(legacy, "w", encoding="utf-8") as f:
            json.dump({leg.name: leg.snapshot()}, f, indent=2, sort_keys=True)  # the OLD format
        mig = SignalCalibrator.load_result("legacy_sig")  # default lake -> migration path
        assert mig is not None and mig["n"] == 1001, (
            f"an existing sidecar must be imported, not reported as a cold start: {mig}"
        )
        assert load_named_state(_STATE_NAME) is not None, (
            "the migration must leave the blob IN the lake, not merely return it"
        )
        assert os.path.exists(legacy), (
            "the legacy sidecar must be left on disk, inert -- never deleted"
        )
        _emit(
            f"  [OK] pre-P4b sidecar imported into the lake (n={mig['n']}) and LEFT on disk at "
            f"{_STATE_NAME}"
        )

        # (6) CROSS-PROCESS, the only proof P4b accepts: a SEPARATE python writes, EXITS, and this
        # process reads the state back out of the lake. An in-process round-trip would pass against a
        # pure in-memory store and prove nothing about durability.
        child = (
            "import sys, os\n"
            f"sys.path.insert(0, {src_root!r})\n"
            f"os.environ['NORD_HOME'] = {work!r}\n"
            "from snn_data_lake.substrate.signal_calibrator import SignalCalibrator\n"
            "c = SignalCalibrator('cross_proc', p=0.75)\n"
            "for i in range(1001):\n"
            "    c.observe(i / 1000.0)\n"
            "assert SignalCalibrator.save_all([c]) is True\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode
        assert rc == 0, f"child process failed to persist a calibrator (rc={rc})"
        xp = SignalCalibrator.load_result("cross_proc")
        assert xp is not None and xp["n"] == 1001 and abs(xp["value"] - 0.75) < 0.03, (
            f"state written by a SEPARATE process was not read back from the lake: {xp}"
        )
        _emit(
            f"  [OK] CROSS-PROCESS: a separate python recorded n={xp['n']} and exited; this process "
            f"read P75 {xp['value']:.4f} back from the one lake"
        )
    finally:
        # RESTORE, never pop -- see the comment above the try.
        if _prior_home is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = _prior_home
        shutil.rmtree(work, ignore_errors=True)

    _emit("-" * width)
    _emit(
        "self-check OK: records a live signal's distribution as a resumable P2 quantile (O(1), no raw"
    )
    _emit(
        "obs stored), persists IN THE ONE LAKE (_T_STATE, cross-process proven) and imports a pre-P4b"
    )
    _emit(
        "sidecar once; honest None when there is no data. Plumbing only -- distils a data operating"
    )
    _emit("point; whether to recalibrate a threshold to it is judged by the gate.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
