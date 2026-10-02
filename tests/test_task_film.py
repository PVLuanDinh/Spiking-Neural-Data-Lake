"""specs/022 T060 -- the executive task-set FiLM surface (FR-034), pytest arms.

The heavy arms (charge A/B with specificity control, committed-evidence reproduction,
cost measurement) run in task_set's registered self-check; these pin the decomposition
identity, the declared anatomy, and the refusal shapes at unit grain.
"""

from snn_data_lake.faculties.task_set import FILM_EDGE, TaskSetLayer, declare_ports


def test_film_decomposes_bias_exactly():
    from snn_data_lake.engine.autonomous_loop import ACTIONS

    n = len(ACTIONS)
    layer = TaskSetLayer(n_sets=3, n_in=4, n_actions=n, seed=0)
    feat = [0.8, 0.2, 0.5, 0.1]
    for settle in (0, 3, 9):
        for _ in range(settle):
            layer.step()
        b = layer.bias(feat)
        f = layer.film(feat)
        assert len(f) == n and all(len(pair) == 2 for pair in f)
        assert all(abs(g * (-dt) - bi) < 1e-12 for (g, dt), bi in zip(f, b, strict=True))
    # cold shift: the gain half collapses with the settle confidence, the identity holds
    layer.shift(1)
    b = layer.bias(feat)
    f = layer.film(feat)
    assert all(g == 0.0 for g, _ in f) or all(
        abs(g * (-dt) - bi) < 1e-12 for (g, dt), bi in zip(f, b, strict=True)
    )
    assert b == [0.0] * n  # just-shifted: confidence 0 -> the preference is silent


def test_declared_anatomy_matches_the_live_shape():
    # the declared width must be DERIVED from the live action vocabulary -- the panel
    # measured the literal form staying green while a grown ACTIONS left the tracked
    # anatomy stating a wire the brain no longer carried
    from snn_data_lake.engine.autonomous_loop import ACTIONS
    from snn_data_lake.engine.wiring_manifest import WiringManifest

    m = WiringManifest()
    declare_ports(m)
    port = m.ports[("task_set", "film_out")]
    assert port.shape == (len(ACTIONS), 2) and port.kind == "OUT"
    assert port.delivery_policy == ("graded", "settle_confidence")
    assert FILM_EDGE == ("task_set", "film_out", "brain", "task_film_in")
    # ...and the live faculty emits exactly that width
    layer = TaskSetLayer(n_sets=2, n_in=3, n_actions=len(ACTIONS), seed=0)
    assert len(layer.film([0.1, 0.2, 0.3])) == len(ACTIONS)


def test_committed_manifest_carries_the_film_wire():
    # the tracked artifact must hold the declared edge (regenerated in the same commit)
    from snn_data_lake.gates.manifest_gate import rebuild_from_registrants

    m = rebuild_from_registrants()
    assert ("task_set", "film_out") in m.ports and ("brain", "task_film_in") in m.ports
    assert ["task_set", "film_out", "brain", "task_film_in"] in [list(e) for e in m.edges]


def test_charge_evidence_is_committed_and_reproduces():
    from snn_data_lake.faculties.task_set import _measure_film_charge
    from snn_data_lake.instruments.port_charge_probe import read_attribution

    rec = read_attribution(FILM_EDGE)
    assert rec is not None and rec.get("diverged") is True
    live, pinned_out, reads, spec = _measure_film_charge()
    assert live != pinned_out and spec
    assert rec.get("measured") == {"reads": reads, "first": list(live[0]), "last": list(live[-1])}
