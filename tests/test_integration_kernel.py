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


# =============================================================================================
# R1 -- the row's red, "four regime arms must separate" (2.4): the instrument with the pulse
# REMOVED reads every arm's PCI exactly 0, so (b1)(b2)(b3)(b6) all read NOT MET
# =============================================================================================
SEEDS = (0, 1, 2, 3)


def _conjunct(v, name):
    return v["conjuncts"][name]


def test_r1_a_pulse_less_run_reads_every_arm_zero_and_the_separation_conjuncts_NOT_MET(
    scratch_home,
):
    art = _fixture()
    readings = {s: M.kernel_regimes(art, s, amp=M.FIXTURE_AMP, pulse_current=0.0) for s in SEEDS}
    for s, arms in readings.items():
        for name, r in arms.items():
            assert r.pci == 0.0, f"seed {s} arm {name}: a pulse-less run must read PCI exactly 0"
            assert r.ones == 0, f"seed {s} arm {name}: the footprint must be EMPTY without a pulse"
    v = M.verdict_t070(readings)
    for name in ("b1", "b2", "b3", "b6"):
        assert not _conjunct(v, name)["met"], f"{name} must read NOT MET on a pulse-less run: {v}"
    assert v["pass"] is False


# =============================================================================================
# R2 -- the registry's dissociation (C-R5, Codex (b)): a "partition" that did NOT cut -- ONE edge
# class left uncut -- propagates, its PCI exceeds the local-echo floor, and the instrument must
# report (b6) NOT MET for it. The un-cut arm is the red's INPUT; the cut arm is the measurement.
# =============================================================================================
def test_the_uncut_partition_keeps_exactly_one_sign_class_of_edges():
    art = _fixture()
    uncut = M.arm_uncut(art)
    cut = M.arm_partitioned(art)
    assert cut["pre_idx"].size == 0 and cut["post_idx"].size == 0 and cut["n_syn"].size == 0
    assert 0 < uncut["pre_idx"].size < art["pre_idx"].size
    assert set(art["sign"][uncut["pre_idx"]].tolist()) == {1}, "the uncut class is the E edges"
    for key in ("agglo_id", "sign", "layer"):
        assert np.array_equal(uncut[key], art[key]), "the same units, only the edges differ"


def test_r2_an_uncut_partition_reads_ABOVE_the_null_and_b6_reads_NOT_MET(scratch_home):
    art = _fixture()
    readings = {s: M.kernel_regimes(art, s, amp=M.FIXTURE_AMP) for s in SEEDS}
    red = {
        s: {**arms, "partitioned": M.run_regime(M.arm_uncut(art), "partitioned", s, M.FIXTURE_AMP)}
        for s, arms in readings.items()
    }
    above = [s for s in SEEDS if red[s]["partitioned"].pci > red[s]["integrated"].null]
    assert above, "the uncut 'partition' must read above the local-echo null on some seed"
    v = M.verdict_t070(red)
    assert not _conjunct(v, "b6")["met"], (
        f"(b6) must read NOT MET for a partition that did not cut: {v}"
    )
    assert "partitioned <= null" in _conjunct(v, "b6")["note"]


# =============================================================================================
# Amendment 2.11 (iii): the kernel's partition loop runs with threads = synapses.p_parts, and the
# INTEGRATED rasters are byte-identical at threads = 1 and threads = p_parts (the determinism
# gate's invariant, read THROUGH the port's two restores). The lineage's pinned P is 1, so the
# invariance is asserted on a P = 4 port built through the seam that exists for this test.
# =============================================================================================
def test_the_port_defaults_threads_to_the_tables_partition_count(scratch_home):
    syn, params = M.kernel_port(_fixture())
    assert params["threads"] == syn.p_parts
    syn4, params4 = M.kernel_port(_fixture(), p_parts=4)
    assert syn4.p_parts == 4 and params4["threads"] == 4
    _syn1, params1 = M.kernel_port(_fixture(), threads=1, p_parts=4)
    assert params1["threads"] == 1


def test_seed_0_integrated_rasters_are_byte_identical_at_threads_1_and_threads_p_parts(
    scratch_home,
):
    art = _fixture()
    pulse = M.Pulse(units=M.perturbation_sets(64, 0)[0], at=0, length=M.PULSE_LEN, current=1.2)
    rasters = {}
    for threads in (1, None):
        syn, params = M.kernel_port(art, threads=threads, p_parts=4)
        assert params["threads"] == (1 if threads == 1 else 4)
        cap = M.warm_capture(syn, params, seed=0, amp=M.FIXTURE_AMP)
        resp = M.kernel_causal_response(cap.blob, syn, params, M.FIXTURE_AMP, pulse)
        rasters[params["threads"]] = (resp.base.tobytes(), resp.pert.tobytes(), cap.inflight)
    assert rasters[1] == rasters[4], (
        "the INTEGRATED rasters diverged between threads=1 and threads=p_parts -- the "
        "determinism gate's thread-count invariance is broken through the port"
    )
    assert rasters[1][2] > 0, "the fixture carried no in-flight spike at the capture tick"
    assert rasters[1][0] != rasters[1][1], "the pulse left no footprint -- the run is vacuous"


# =============================================================================================
# Amendment 2.11: the campaign's SCHEDULE -- every arm reported as it completes (a stalled run is
# distinguishable from a running one within minutes), the seeds in a thread pool assembled in the
# order given (the pin byte-identical whatever the scheduling), and the cost rule: ONE measured
# seed projects the campaign, and a projection past the ceiling REFUSES the run before it starts.
# =============================================================================================
def test_kernel_regimes_reports_each_arm_as_it_completes_in_arm_order(scratch_home):
    art = _fixture()
    seen = []
    readings = M.kernel_regimes(
        art, 0, amp=M.FIXTURE_AMP, report=lambda r, secs: seen.append((r.arm, secs))
    )
    assert [arm for arm, _ in seen] == list(M.ARMS), "one report per arm, in arm order"
    assert all(secs > 0.0 for _, secs in seen)
    assert readings == M.kernel_regimes(art, 0, amp=M.FIXTURE_AMP), "reporting changes no reading"


def test_pooled_seeds_read_the_same_arms_as_the_serial_run_in_the_order_given(scratch_home):
    art = _fixture()
    seeds = (2, 0, 1)
    pooled, walls = M.pooled_regimes(art, seeds, M.FIXTURE_AMP, workers=3)
    assert list(pooled) == [2, 0, 1] and list(walls) == [2, 0, 1], "the order given, not completion"
    for s in seeds:
        assert pooled[s] == M.kernel_regimes(art, s, amp=M.FIXTURE_AMP), f"seed {s} differs pooled"
        assert walls[s] > 0.0


def test_a_seed_that_projects_the_campaign_past_the_ceiling_is_refused():
    assert M.CAMPAIGN_CEILING_S == 3 * 3600
    slow = {arm: 1_200.0 for arm in M.ARMS}  # a fake 6,000 s seed
    p = M.campaign_projection(slow, n_seeds=8, workers=8)
    assert (p.seed_s, p.serial_s, p.scheduled_s) == (6_000.0, 48_000.0, 12_000.0)
    assert p.refused, "6,000 s x (1 + ceil(7 / 8)) = 12,000 s > 10,800 s must REFUSE"
    quick = {arm: 60.0 for arm in M.ARMS}  # a fake 300 s seed
    q = M.campaign_projection(quick, n_seeds=8, workers=8)
    assert (q.serial_s, q.scheduled_s, q.refused) == (2_400.0, 600.0, False)
    one = M.campaign_projection(quick, n_seeds=8, workers=1)
    assert (one.scheduled_s, one.refused) == (2_400.0, False), "one worker: scheduled == serial"
    assert M.campaign_projection({arm: 500.0 for arm in M.ARMS}, 8, workers=1).refused
