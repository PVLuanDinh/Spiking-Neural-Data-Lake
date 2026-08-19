# T005b evidence — format-bridge round-trip at live scale

**Method**: `python -m snn_data_lake.substrate.lake_bridge --live` with `NORD_HOME` pointed at
the barrier backup `D:\backups\snn_lake_pre022_20260819` (byte-faithful copy of the live lake,
taken 2026-08-19 with no writer running). Deliberately NOT the running daemon's home:
constructing a second BinLog runs tail-repair, and a frame the daemon is mid-appending would
read as torn and be truncated under it — the writer-vs-probe hazard.

**Result** (pinned host, 2026-08-19): round-trip OK over **156,327 records**, bridge digest ==
direct BinLog digest (`074f544d4de55fb9…`), and the single-byte mutation control MOVED the
digest in the same command — the equality is evidence, not vacuous.

Constructed-lake arms (7 records + torn-tail + boundary-sensitivity + premature-segment
refusal): `tests/test_lake_bridge.py`, 6/6 green; the bare `run_selfchecks` entry re-runs the
constructed round-trip on every gate run.
