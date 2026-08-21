"""TypedPort + WiringManifest + manifest_gate (specs/022 T018/T019) -- the contract's checks
and every red, through the public surface (the module self-checks prove the same properties
through __main__; these survive a refactor that moves them).
"""

import pytest

from snn_data_lake.substrate.manifest_gate import run as gate_run
from snn_data_lake.substrate.wiring_manifest import (
    ARTIFACT,
    Port,
    WiringManifest,
)


def _out(name="s_out", region="neuromodulation", module="raphe_serotonin", **kw):
    args = dict(
        name=name,
        region=region,
        shape=(4,),
        dtype="f4",
        unit="rate_hz",
        encoding="VEC_DENSE",
        kind="OUT",
        declaring_module=module,
        delivery_policy="all-or-none",
    )
    args.update(kw)
    return Port(**args)


def _in(name="d_in", region="thalamus", module="global_workspace", **kw):
    args = dict(
        name=name,
        region=region,
        shape=(4,),
        dtype="f4",
        unit="rate_hz",
        encoding="VEC_DENSE",
        kind="IN",
        declaring_module=module,
        join_policy="sum",
    )
    args.update(kw)
    return Port(**args)


def _pair() -> WiringManifest:
    m = WiringManifest()
    m.declare_port("salience", _out())
    m.declare_port("ws", _in())
    m.declare_edge("salience", "s_out", "ws", "d_in")
    return m


def test_construction_refuses_underspecified_ports():
    with pytest.raises(ValueError):
        _in(join_policy=None)  # IN without a join: the silent-sum door stays shut
    with pytest.raises(ValueError):
        _out(delivery_policy=None)  # OUT without delivery
    with pytest.raises(ValueError):
        _out(delivery_policy=("graded", ""))  # graded needs a real association source
    with pytest.raises(ValueError):
        _in(delivery_policy="all-or-none")  # policy on the wrong direction
    with pytest.raises(ValueError):
        _out(encoding="DENSE")  # unknown encoding
    with pytest.raises(ValueError):
        _out(shape=(0,))  # degenerate shape


def test_five_checks_green_on_a_clean_manifest():
    m = _pair()
    assert m.all_findings() == []
    assert m.warnings() == []


def test_structure_and_type_reds():
    m = _pair()
    m.declare_edge("ghost", "nope", "ws", "d_in")
    m.declare_edge("salience", "s_out", "ws", "d_in")  # duplicate of the original
    findings = m.validate_structure()
    assert any("not a declared port" in f for f in findings)
    assert any("duplicate edge" in f for f in findings)
    m2 = _pair()
    m2.declare_port("ws2", _in(name="d2_in", unit="volts", dtype="i8"))
    m2.declare_edge("salience", "s_out", "ws2", "d2_in")
    reds = m2.check_types()
    assert any("dtype" in f for f in reds) and any("unit" in f for f in reds)


def test_concat_join_infers_summed_axis():
    m = WiringManifest()
    m.declare_port("a", _out(name="a_out", shape=(3, 2)))
    m.declare_port("b", _out(name="b_out", shape=(5, 2), module="acetylcholine"))
    m.declare_port("ws", _in(shape=(8, 2), join_policy="concat"))
    m.declare_edge("a", "a_out", "ws", "d_in")
    m.declare_edge("b", "b_out", "ws", "d_in")
    assert m.infer_types()[("ws", "d_in")] == (8, 2)
    assert m.check_types() == []


def test_deserialization_reds_join_and_region():
    m = _pair()
    m.declare_edge("salience", "s_out", "ws", "d_in")  # 2 producers into one IN
    d = m.to_dict()
    d["ports"][1][1]["join_policy"] = None
    d["ports"][0][1]["region"] = "cerebellum"
    bad = WiringManifest.from_dict(d, strict=False)
    assert bad.join_check(), "policy-stripped fan-in must red (the undeclared-join red)"
    assert bad.region_check(), "a region the module does not hold must red"
    with pytest.raises(ValueError):
        WiringManifest.from_dict(d, strict=True)  # strict reload refuses the same artifact


def test_hdf5_round_trip_full_fidelity():
    pytest.importorskip("nir")
    m = _pair()
    assert WiringManifest.from_hdf5_bytes(m.to_hdf5_bytes()).to_dict() == m.to_dict()


def test_gate_green_on_committed_artifact_and_divergence_red(tmp_path):
    ok, report = gate_run()
    assert ok, f"the committed artifact must satisfy its own gate:\n{report}"
    assert "EMPTY" in report or "ports" in report, "the census line went missing"
    pytest.importorskip("nir")
    diverged = _pair()
    p = tmp_path / "diverged.nir"
    diverged.commit_artifact(p)
    ok2, report2 = gate_run(p)
    assert not ok2 and "DIVERGENCE" in report2, (
        "a committed artifact the registrants would not rebuild must red"
    )


def test_artifact_is_tracked_and_loadable():
    # The premise updates WITH each registrant commit, deliberately (its failure message
    # demands exactly that): T025 landed the serotonin patience wire (2 ports, 1 edge);
    # T026 added the interoceptive channel (arousal + drives producers, the insula
    # body_in concat terminus and its edge-less interoception_out); T027 added the glia
    # diffuse channel (glial_gain_out -> synaptic_gain_in, a deliberate self-edge); T028
    # added THE predictor port (percept_surprise.surprise_out -> the brain's
    # undirected-salience competition); T029 added the RPE third-factor port
    # (dopamine_nuclei.da_out -> r_stdp's eligibility gate). A count above the pin means
    # a registrant landed without updating this premise; below means a declaration was
    # lost.
    pytest.importorskip("nir")
    m = WiringManifest.load_artifact(ARTIFACT)
    assert (len(m.ports), len(m.edges)) == (20, 10), (
        "the T034 artifact carries the patience wire + the interoceptive channel + the "
        "glia self-edge + the predictor port + the third-factor port + the workspace "
        "composite + the report channel AND its brain consumer (20 ports, 10 edges); a "
        "different census means a registrant landed without updating this test's premise"
    )
    assert set(m.edges) == {
        ("raphe_serotonin", "five_ht_out", "evidence_accumulation", "patience_in"),
        ("homeostatic_drives", "drive_deficit_out", "interoceptive_integration", "body_in"),
        ("interoceptive_arousal", "arousal_out", "interoceptive_integration", "body_in"),
        ("astrocyte_modulation", "glial_gain_out", "astrocyte_modulation", "synaptic_gain_in"),
        ("percept_surprise", "surprise_out", "brain", "salience_surprise_in"),
        ("dopamine_nuclei", "da_out", "r_stdp", "dopamine_in"),
        ("autonomous_loop", "coalition_out", "global_workspace", "coalition_in"),
        ("reentrant_workspace", "reroute_out", "brain", "reroute_in"),
        ("global_workspace", "broadcast_out", "report_channel", "report_in"),
        ("report_channel", "report_out", "brain", "report_in"),
    }, "the committed edges must be exactly the declared subscriptions"
    assert m.all_findings() == [], "the committed artifact must satisfy all five checks"
