"""Tests for the connectome router -- signed, delayed spike propagation over the connectome."""

from snn_data_lake import connectome
from snn_data_lake.thalamus.connectome_router import ConnectomeRouter


def _run(router, drive, ticks):
    out = {}
    for _ in range(ticks):
        out = router.step(drive)
    return out


def test_regions_are_the_connectome_nodes():
    r = ConnectomeRouter(seed=0)
    assert set(r.regions) == set(connectome.NODES)
    assert r.n == len(connectome.NODES)


def test_step_returns_a_rate_per_region_in_unit_range():
    r = ConnectomeRouter(seed=0)
    out = r.step({"ec": 1.0})
    assert set(out) == set(r.regions)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_downstream_targets_fire_when_a_region_is_driven():
    # ec -> dg (perforant path) -> ca3 (mossy fibres): connectome-downstream targets light up.
    r = ConnectomeRouter(seed=0)
    out = _run(r, {"ec": 1.0}, 6)
    assert out["dg"] > 0.2
    assert out["ca3"] > 0.2


def test_isolated_regions_stay_quiet():
    # The olfactory pair {ob, pir} has no in-edge from the rest of the graph; driving ec elsewhere
    # must never make them fire.
    r = ConnectomeRouter(seed=0)
    out = _run(r, {"ec": 1.0}, 8)
    assert out["ob"] == 0.0
    assert out["pir"] == 0.0


def test_undriven_network_is_silent():
    r = ConnectomeRouter(seed=0)
    out = _run(r, None, 10)
    assert all(v == 0.0 for v in out.values())


def test_inhibitory_edge_reduces_target_versus_excitatory():
    def mean_target(drive, target, ticks=14, tail=6, seed=1):
        r = ConnectomeRouter(seed=seed)
        rates = [r.step(drive)[target] for _ in range(ticks)]
        return sum(rates[-tail:]) / tail

    exc_only = mean_target({"cb_deep": 1.0}, "thal_relay")  # dentatothalamic (exc)
    # gpi_snr's sole out-edge is the inhibitory pallidothalamic gate onto thal_relay
    exc_plus_inh = mean_target({"cb_deep": 1.0, "gpi_snr": 1.0}, "thal_relay")
    assert exc_plus_inh < exc_only


def test_positive_loop_reverberates_more_than_no_loop():
    # hippocampal trisynaptic loop ec->dg->ca3->ca1->sub->ec is all-excitatory and closes.
    rev_loop = ConnectomeRouter(seed=2).reverberation("ec", 40)
    # actuator is a pure sink with no out-edge -> activity cannot persist.
    rev_none = ConnectomeRouter(seed=2).reverberation("actuator", 40)
    assert rev_loop > 0.01  # persists above baseline
    assert rev_loop > 3.0 * rev_none  # loop clearly out-reverberates the no-loop sink


def test_reverberation_is_side_effect_free():
    # measuring reverberation must not disturb the live stepping state
    r = ConnectomeRouter(seed=5)
    before = r.step({"ctx_sensory": 0.7})
    r.reverberation("ec", 20)
    after = r.step({"ctx_sensory": 0.7})
    # a follow-up identical driven step on an untouched router matches
    ref = ConnectomeRouter(seed=5)
    ref.step({"ctx_sensory": 0.7})
    assert after == ref.step({"ctx_sensory": 0.7})
    assert isinstance(before, dict)


def test_determinism_same_seed_identical_steps():
    d1 = ConnectomeRouter(seed=7)
    d2 = ConnectomeRouter(seed=7)
    seq = [{"ctx_sensory": 0.8}, {"ctx_motor": 0.5}, None, {"vta_snc": 1.0}, None]
    assert [d1.step(s) for s in seq] == [d2.step(s) for s in seq]


def test_determinism_same_seed_identical_reverberation():
    a = ConnectomeRouter(seed=3).reverberation("ec", 25)
    b = ConnectomeRouter(seed=3).reverberation("ec", 25)
    assert a == b


def test_different_seed_changes_thresholds_but_keeps_topology():
    r0 = ConnectomeRouter(seed=0)
    r1 = ConnectomeRouter(seed=1)
    assert r0.regions == r1.regions  # topology is seed-independent
    assert not (r0.thr == r1.thr).all()  # heterogeneity is seeded
