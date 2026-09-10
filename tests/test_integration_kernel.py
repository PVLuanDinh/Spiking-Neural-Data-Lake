"""T070's reds: the integration instrument ported onto the kernel's port (specs/022 Phase 4 / C4,
tasks.md T070; pre-registration `specs/022-unified-brain-arc/evidence/t070-integration-kernel.md`,
sections 1-2 frozen at f09adc4).

Committed RED, before `cortex/integration_measure.py` carries a kernel port.

R3 -- THE G5 CHAIN (2.4). The port replays from `substrate/replay_gate.capture` / `restore` (T017),
the ONE snapshot mechanism in the tree, and three things must hold on the harness fixture
(`prior_drift_harness.synthetic_artifact(64, out_degree=8)` through `h01_ingest.population`):
  (i)   the restored BASELINE continuation is byte-identical to the uninterrupted original run over
        the same ticks -- replay_gate's own criterion, read through the port;
  (ii)  a state-frame-only "restore" (`KernelState.from_blob` into a fresh `KernelTick`, the tick
        runtime -- rings, accumulators, tick_index -- dropped) DIVERGES from the original on that
        fixture. The fixture must carry in-flight spikes at the capture tick for this control to
        mean anything, so the count is asserted beside the divergence;
  (iii) the port REFUSES a snapshot with its RNG positions stripped (`SnapshotRefused`) -- a
        positionless restore would re-seed, and a re-seeded twin walks a different trajectory.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_integration_kernel.py -q --no-cov
"""

import io
import json

import pytest

np = pytest.importorskip("numpy")

from snn_data_lake.cortex import integration_kernel as M  # noqa: E402
from snn_data_lake.cortex import prior_drift_harness as H  # noqa: E402
from snn_data_lake.substrate import replay_gate as G  # noqa: E402


@pytest.fixture
def scratch_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    return tmp_path


def _fixture():
    return H.synthetic_artifact(64, out_degree=8)


def _raster_bytes(kt, stream, n_units, amp, steps):
    """The uninterrupted ORIGINAL run's continuation, drawn by the port's own drive function."""
    amp_q = amp * float(kt.state.arrays["thr"].max()) * (1 << kt.syn.frac_bits)
    rows = np.zeros((steps, n_units), dtype=np.int8)
    for t in range(steps):
        rows[t, kt.tick(M.background_drive(stream, n_units, amp_q))] = 1
    return rows.tobytes()


# =============================================================================================
# R3 (i): the port's BASELINE restore continues the original run byte for byte
# =============================================================================================
def test_r3_i_the_restored_baseline_continuation_is_byte_identical_to_the_original(scratch_home):
    syn, params = M.kernel_port(_fixture())
    cap = M.warm_capture(syn, params, seed=0, amp=M.FIXTURE_AMP)
    original = _raster_bytes(cap.kt, cap.stream, syn.n_units, M.FIXTURE_AMP, M.STEPS)
    pulse = M.Pulse(units=(0, 1, 2), at=0, length=M.PULSE_LEN, current=M.PULSE_CURRENT)
    resp = M.kernel_causal_response(cap.blob, syn, params, M.FIXTURE_AMP, pulse)
    assert resp.base.tobytes() == original, (
        "the restored BASELINE must be the original run's own continuation (G5's criterion)"
    )
    assert resp.footprint.shape == (M.STEPS - M.PULSE_LEN, syn.n_units)


# =============================================================================================
# R3 (ii): a frame-only restore (runtime dropped) DIVERGES -- the chain matters, shown
# =============================================================================================
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_r3_ii_a_frame_only_restore_diverges_from_the_original(scratch_home, seed):
    syn, params = M.kernel_port(_fixture())
    cap = M.warm_capture(syn, params, seed=seed, amp=M.FIXTURE_AMP)
    assert cap.inflight > 0, (
        f"seed {seed}: the fixture carries NO in-flight spikes at the capture tick -- the fixture "
        "is wrong, not the chain (2.4 R3 (ii))"
    )
    original = _raster_bytes(cap.kt, cap.stream, syn.n_units, M.FIXTURE_AMP, M.STEPS)
    frame_only = M.frame_only_continuation(cap.blob, syn, params, M.FIXTURE_AMP, M.STEPS)
    assert frame_only.tobytes() != original, (
        f"seed {seed}: a restore that drops the tick runtime reproduced the original run with "
        f"{cap.inflight} spikes in flight -- the G5 chain would be a RESET here, not a restore"
    )


# =============================================================================================
# R3 (iii): a positions-stripped snapshot is REFUSED by the port
# =============================================================================================
def test_r3_iii_the_port_refuses_a_positions_stripped_blob(scratch_home):
    syn, params = M.kernel_port(_fixture())
    cap = M.warm_capture(syn, params, seed=0, amp=M.FIXTURE_AMP)
    with io.BytesIO(cap.blob) as f:
        z = np.load(f, allow_pickle=False)
        kept = {k: z[k] for k in z.files if k != "rng_positions"}
    buf = io.BytesIO()
    np.savez(buf, **kept)
    stripped = buf.getvalue()
    assert "rng_positions" not in np.load(io.BytesIO(stripped)).files  # the control mutated
    pulse = M.Pulse(units=(0, 1, 2), at=0, length=M.PULSE_LEN, current=M.PULSE_CURRENT)
    with pytest.raises(G.SnapshotRefused):
        M.kernel_causal_response(stripped, syn, params, M.FIXTURE_AMP, pulse)


def test_the_snapshot_the_port_captures_names_the_drive_stream(scratch_home):
    """The stream is captured by POSITION under the name the port restores it by."""
    syn, params = M.kernel_port(_fixture())
    cap = M.warm_capture(syn, params, seed=0, amp=M.FIXTURE_AMP)
    with np.load(io.BytesIO(cap.blob), allow_pickle=False) as z:
        positions = json.loads(str(z["rng_positions"]))
    assert list(positions) == [M.DRIVE_STREAM]
