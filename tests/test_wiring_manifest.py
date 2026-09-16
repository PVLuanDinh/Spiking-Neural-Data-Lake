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


def test_an_edge_less_OUT_port_warns():
    """THE REACHABLE RED for the OUT arm (specs/022 T039l).

    Before it, `warnings()` filtered `p.kind == 'IN'` and nothing in the tree fired on a
    producer wired to nothing -- which is how `interoception_out` sat edge-less from T026 to
    T039j, the only such port in a 24-port manifest, unremarked by any gate, warning or
    test."""
    m = WiringManifest()
    m.declare_port("salience", _out())
    w = m.warnings()
    assert any("OUT port" in x for x in w), f"a dangling OUT port produced no warning: {w}"
    assert "s_out" in " ".join(w)


def test_the_OUT_arm_does_not_warn_about_an_edged_producer():
    """THE CONTROL. Without it the arm above could be satisfied by warning about
    EVERY OUT port -- which would fire on the whole manifest and mean nothing."""
    m = _pair()  # one OUT, one IN, one edge between them
    assert m.warnings() == []


def test_the_two_arms_stay_distinguishable_in_the_text():
    """A reader triaging a warning list needs to know whether a CONSUMER is unfed or
    a PRODUCER unheard. They are different defects and must not read alike."""
    m = WiringManifest()
    m.declare_port("salience", _out())
    m.declare_port("ws", _in())
    w = m.warnings()
    assert len(w) == 2, w
    assert sum("OUT port" in x for x in w) == 1
    assert sum(x.startswith("IN port") for x in w) == 1


def test_a_REF_port_is_covered_by_refs_not_by_warnings():
    """REF is the audited escape hatch with its own census; warnings() deliberately
    covers neither direction for it, so a REF port appears in neither arm."""
    m = WiringManifest()
    m.declare_port(
        "hatch",
        Port(
            name="ref_port",
            region="substrate",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="REF",
            declaring_module="wiring_manifest",
        ),
    )
    assert m.warnings() == []
    assert m.refs() == [("hatch", "ref_port")]


def test_the_committed_artifact_has_no_dangling_port_in_either_direction():
    """The live pin. Measured at T039l: zero danglers either way. T046a declared ONE
    edge-less OUT port (lp_out, its consumer dated to T047 -- the T030 broadcast_out
    precedent) and this pin carried it EXACTLY; T047 landed the consumer's edge and the
    pin, red on that landing as designed, returns to zero -- an UNDATED dangler reds."""
    pytest.importorskip("nir")
    m = WiringManifest.load_artifact(ARTIFACT)
    assert m.warnings() == [], f"the committed artifact has a dangling port: {m.warnings()}"


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
    # (dopamine_nuclei.da_out -> r_stdp's eligibility gate). T039j completed the declared
    # interoceptive TRACT: the medulla->insula shortcut is RETIRED and the ascending half
    # now relays through `thal_relay` (arousal_out -> visceral_in -> body_in, +1 net edge,
    # +2 ports), and `interoception_out` -- which this premise itself called EDGE-LESS
    # above, and which was the only OUT port in the manifest sourcing no edge -- gained its
    # declared consumer (region_coherence.interoception_in, +1 port +1 edge). T046a added
    # the |LP| pair (surprise_in typed-merge + the edge-less lp_out, dated to T047) and
    # the DISCHARGED T028 disposition's edge (surprise_out -> surprise_in, declared by the
    # consumer): +2 ports +1 edge. T047 landed lp_out's dated consumer (growth_stop.lp_in
    # + the edge, declared by the consumer): +1 port +1 edge, danglers back to zero. A
    # count above the pin means a registrant landed without updating this premise; below
    # means a declaration was lost.
    pytest.importorskip("nir")
    m = WiringManifest.load_artifact(ARTIFACT)
    # T052-iii added the vision/audition afferent lanes (2 OUT ports at the boundary,
    # 2 IN ports on brain, 2 edges -- landing WITH their consumers, the T028 precedent:
    # _perceive_real/_perceive_audio now acquire every frame through the contracts).
    # T060 added the executive task-set FiLM wire (film_out + brain.task_film_in + the
    # edge, declared by the consumer): +2 ports +1 edge.
    # T082 added the belief wire (recursive_tom.belief_out + brain.social_risk_in + the edge,
    # declared by the PRODUCER -- the T082 record's 2.10 (d)(1)): +2 ports +1 edge. This premise
    # was NOT updated in the registrant commit `af07f81`; the stacked battery on `9d6a33c` read
    # (38, 20) against (36, 19) here and went RED on this one test (2026-09-16) -- the premise's
    # own rule ("a count above the pin means a registrant landed without updating this premise")
    # fired exactly as written, and this is the stacked fix.
    assert (len(m.ports), len(m.edges)) == (38, 20), (
        "the artifact carries the patience wire + the interoceptive TRACT (relayed "
        "through thal_relay, with interoception_out's consumer) + the glia self-edge + the "
        "predictor port + the third-factor port + the workspace composite + the report "
        "channel and its brain consumer + the proprioceptive afferent lane + the efferent "
        "lane (action_out -> efferent_in) + the |LP| pair with its discharged surprise "
        "edge + the growth gate's lp_in consumer + the vision/audition event-frame lanes "
        "+ the T060 executive FiLM wire + the T082 belief wire (38 ports, 20 edges); a "
        "different census means a registrant landed without updating this test's premise"
    )
    assert set(m.edges) == {
        ("raphe_serotonin", "five_ht_out", "evidence_accumulation", "patience_in"),
        ("homeostatic_drives", "drive_deficit_out", "interoceptive_integration", "body_in"),
        # T039j: the ascending half RELAYS. The retired shortcut was
        # ("interoceptive_arousal", "arousal_out", "interoceptive_integration", "body_in") --
        # medulla straight to cortex, a route the connectome declares nowhere.
        ("interoceptive_arousal", "arousal_out", "interoceptive_relay", "visceral_in"),
        ("interoceptive_relay", "relay_out", "interoceptive_integration", "body_in"),
        ("interoceptive_integration", "interoception_out", "region_coherence", "interoception_in"),
        ("astrocyte_modulation", "glial_gain_out", "astrocyte_modulation", "synaptic_gain_in"),
        ("percept_surprise", "surprise_out", "brain", "salience_surprise_in"),
        ("dopamine_nuclei", "da_out", "r_stdp", "dopamine_in"),
        ("autonomous_loop", "coalition_out", "global_workspace", "coalition_in"),
        ("reentrant_workspace", "reroute_out", "brain", "reroute_in"),
        ("global_workspace", "broadcast_out", "report_channel", "report_in"),
        ("report_channel", "report_out", "brain", "report_in"),
        ("afferent_boundary", "proprio_out", "brain", "proprio_in"),
        ("autonomous_loop", "action_out", "efferent_boundary", "efferent_in"),
        # T060: the executive task-set FiLM wire (the consumer declares the edge)
        ("task_set", "film_out", "brain", "task_film_in"),
        # T082: the belief wire, declared by the PRODUCER (recursive_tom.declare_ports; the T082
        # record's 2.10 (d)(1)) -- the brain's social_risk_in join is `sum`.
        ("recursive_tom", "belief_out", "brain", "social_risk_in"),
        # T046a: T028's |LP| disposition discharged -- the consumer declares the edge.
        ("percept_surprise", "surprise_out", "learning_progress", "surprise_in"),
        # T047: T046a's lp_out disposition discharged -- the growth gate consumes |LP|.
        ("learning_progress", "lp_out", "growth_stop", "lp_in"),
        # T052-iii: T036's S1 disposition discharged -- the perception tiers consume the
        # packs through the boundary contracts (identity-on-valid, fail-closed-to-no-percept).
        ("afferent_boundary", "vision_out", "brain", "vision_in"),
        ("afferent_boundary", "audition_out", "brain", "audition_in"),
    }, "the committed edges must be exactly the declared subscriptions"
    assert m.all_findings() == [], "the committed artifact must satisfy all five checks"
