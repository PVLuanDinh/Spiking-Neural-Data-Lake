"""specs/022 T039j: the declared interoceptive tract is complete at the typed layer.

`connectome.py` declares `viscera -> nts_rvlm -> thal_relay -> insula -> {gwt, ctx_pfc}`. Until
this task the MIDDLE hop existed only as a node id, so the wiring manifest declared a SHORTCUT the
anatomy does not: `interoceptive_arousal.arousal_out -> interoceptive_integration.body_in`, medulla
straight to cortex. `interoception_out` was meanwhile the one OUT port in the whole manifest with no
outgoing edge.

WHAT THIS FILE DOES NOT CLAIM, because the repo already adjudicated it and being wrong here would
be worse than the gap. `specs/BACKLOG_RANKED.md` row 59 says in as many words: *do not "fix"
FINDING 1 by adding region-to-region imports* -- `p6_gate`'s `unimplemented_declared_pairs`
measures IMPORT TOPOLOGY, not traffic, and these routes were ALREADY load-bearing through
`thalamus/connectome_router.py`, which turns every declared edge into a delayed LIF projection. So
no test here asserts that a dead route was brought to life. What is asserted is the typed layer:
the manifest now matches the anatomy, the relay transmits rather than forwards, and every edge on
the tract carries a measured charge.

Run: pytest tests/test_interoceptive_tract.py
"""

import pytest

from snn_data_lake.insula.interoceptive_integration import (
    BODY_CHANNELS,
    RELAY_EDGE,
    InteroceptiveIntegration,
)
from snn_data_lake.substrate.region_coherence import INTEROCEPTION_EDGE
from snn_data_lake.thalamus.interoceptive_relay import (
    VISCERAL_EDGE,
    InteroceptiveRelay,
)

QUIET = [0.05, 0.05]
STEP = [0.40, 0.40]


def _settled(n=300, v=QUIET):
    r = InteroceptiveRelay()
    for _ in range(n):
        r.relay(v)
    return r


# ======================================================================================
# The manifest: the tract as DECLARED
# ======================================================================================
@pytest.fixture(scope="module")
def manifest():
    from snn_data_lake.substrate.manifest_gate import rebuild_from_registrants

    return rebuild_from_registrants()


def test_the_manifest_no_longer_shortcuts_the_thalamic_hop(manifest):
    """The anatomy relays through thalamus. A manifest edge from medulla straight to insula
    described a path the connectome declares nowhere."""
    edges = set(manifest.edges)
    assert VISCERAL_EDGE in edges, "the ascending edge does not land on the relay"
    assert RELAY_EDGE in edges, "the relayed edge does not reach the terminus"
    shortcut = (
        "interoceptive_arousal",
        "arousal_out",
        "interoceptive_integration",
        "body_in",
    )
    assert shortcut not in edges, (
        "the medulla->insula shortcut is still declared; the tract now has two descriptions"
    )


def test_interoception_out_is_no_longer_edge_less(manifest):
    """It was the ONLY OUT port in the manifest sourcing no edge -- a declared-ahead disposition
    carried since T026."""
    assert INTEROCEPTION_EDGE in set(manifest.edges)
    out_ports = {k for k, p in manifest.ports.items() if p.kind == "OUT"}
    sourced = {(e[0], e[1]) for e in manifest.edges}
    assert not (out_ports - sourced), f"OUT ports sourcing no edge: {sorted(out_ports - sourced)}"


def test_every_edge_on_the_tract_carries_a_measured_charge():
    """A declared edge that carries nothing is the defect the edge-less disposition was honest
    about. Each of the three is asserted from the COMMITTED attribution, so a route that stopped
    carrying would fail here rather than pass on its declaration."""
    from snn_data_lake.substrate.port_charge_probe import read_attribution

    for edge in (VISCERAL_EDGE, RELAY_EDGE, INTEROCEPTION_EDGE):
        rec = read_attribution(edge)
        assert rec is not None, f"no committed attribution for {edge}"
        assert rec.get("diverged") is True, f"{edge} recorded as non-diverged"


def test_no_attribution_cites_an_edge_the_manifest_does_not_declare(manifest):
    """A record for a retired edge is a claim about a route that does not exist. The medulla's
    charge MOVED when the ascending edge moved; the old record was retired, not left beside it."""
    import json
    import pathlib

    ev = json.loads(
        (
            pathlib.Path(__file__).resolve().parents[1] / "tools" / "region_charge_evidence.json"
        ).read_text(encoding="utf-8")
    )
    declared = set(manifest.edges)
    for a in ev["attributions"]:
        edge = tuple(a["edge"])
        if edge[0] == edge[2]:
            continue  # a self-edge (the astrocyte diffuse channel) is not a manifest route
        assert edge in declared, f"attribution cites an undeclared edge: {edge}"


# ======================================================================================
# The relay: it TRANSMITS, it does not forward
# ======================================================================================
def test_a_change_after_quiet_is_amplified_and_a_persistent_one_is_not():
    """The tonic/burst distinction, which is the whole reason this is a hop and not a wire."""
    r = _settled()
    burst = r.relay(STEP)[0]
    assert burst > STEP[0], f"a step after quiet relayed faithfully ({burst:.4f})"
    assert r.mode("stress") == "burst"
    for _ in range(300):
        held = r.relay(STEP)[0]
    assert held < burst and abs(held - STEP[0]) < abs(burst - STEP[0]), (
        f"sustained input never relaxed to tonic (burst {burst:.4f}, held {held:.4f})"
    )


def test_a_channel_that_was_never_quiet_cannot_burst():
    """THE GUARD that stops this being a gain applied to everything."""
    r = _settled(v=STEP)
    assert r.relay(STEP)[0] <= STEP[0] + 1e-12


def test_a_pass_through_relay_cannot_reproduce_the_amplification():
    """THE REACHABLE RED. Pin de-inactivation at zero and the module's whole claim must fail --
    otherwise it is a rename of the identity function."""
    r = _settled()
    r._deinact = dict.fromkeys(r.channels, 0.0)
    assert r.relay(STEP)[0] <= STEP[0] + 1e-12


def test_the_relayed_pair_never_leaves_the_range_its_port_declares():
    r = _settled(v=[0.0, 0.0])
    for v in ([1.0, 1.0], [0.0, 0.0], [0.5, 1.0], [0.9, 0.1]):
        assert all(0.0 <= g <= 1.0 for g in r.relay(v))


def test_the_relay_is_deterministic():
    """No RNG at all, so it adds nothing to the G5 snapshot and replay stays byte-identical."""
    frames = [QUIET] * 40 + [STEP] * 10 + [QUIET] * 10
    a, b = InteroceptiveRelay(), InteroceptiveRelay()
    assert [a.relay(f) for f in frames] == [b.relay(f) for f in frames]


def test_routing_the_real_medullary_signal_through_it_moves_the_real_terminus():
    """THE COMPOSITION, end to end and with the real modules. A hop that changed nothing
    downstream would be presence without participation, which P3 refuses."""
    from snn_data_lake.medulla.interoceptive_arousal import InteroceptiveArousal

    organ = InteroceptiveArousal(seed=0)
    q = organ.sense(0.0)
    hit = organ.sense(3.0)
    direct = InteroceptiveIntegration(channels=BODY_CHANNELS)
    relayed = InteroceptiveIntegration(channels=BODY_CHANNELS)
    hop = InteroceptiveRelay()

    def frame(vec):
        return dict(
            energy_deficit=0.0,
            curiosity_deficit=0.0,
            social_deficit=0.0,
            stress=vec[0],
            caution=vec[1],
        )

    qv = [q["stress"], q["caution"]]
    for _ in range(300):
        direct.observe(**frame(qv))
        relayed.observe(**frame(hop.relay(qv)))
    hv = [hit["stress"], hit["caution"]]
    assert relayed.observe(**frame(hop.relay(hv))) > direct.observe(**frame(hv))


# ======================================================================================
# P2: this is a different LAYER from the incumbent, not a second mechanism for one job
# ======================================================================================
def test_the_router_propagates_topology_and_carries_no_typed_payload():
    """THE P2 CHECK, and it is asserted rather than argued because the incumbent is real:
    `thalamus/connectome_router` already turns every declared connectome edge into a delayed LIF
    projection, and BACKLOG row 59 records the tract as load-bearing through it. That module
    propagates an ABSTRACT graded drive over the anatomy graph; it never touches the typed
    {stress, caution} payload or the terminus. If it ever does, these two ARE one job and this
    module must merge into it rather than sit beside it."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src/snn_data_lake/thalamus/connectome_router.py"
    ).read_text(encoding="utf-8")
    for token in ("stress", "caution", "interoceptive_integration", "InteroceptiveIntegration"):
        assert token not in src, (
            f"connectome_router now references {token!r} -- it has grown into the typed "
            "interoceptive path, and P2 says one job gets one mechanism"
        )


def test_the_brain_routes_the_live_body_frame_through_the_hop():
    """Without this the relay is a module with a self-check and no tick."""
    from snn_data_lake.cortex.brain import make_brain

    b = make_brain(seed=0, persist=False)
    try:
        assert b._thal_relay is not None, "the brain built no thalamic relay"
        for _ in range(12):
            b.tick()
        assert b._thal_relay._seen > 0, (
            "the relay exists on the brain but no tick sent the body frame through it"
        )
    finally:
        b.close()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
