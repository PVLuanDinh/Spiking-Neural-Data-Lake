"""specs/022 T111 (G3) P2: ONE grower -- NGSNN.fit's candidate lifecycle factored for a second
consumer.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.5 and 2.9 C6. The percept
consumer must be BYTE-IDENTICAL across the refactor: `growth_fingerprint` and the trajectory are
pinned below on `SyntheticPerceptStream` seeds 9000-9003 (200 and 600 ticks) and on
`online_readout`'s self-check engines, each value MEASURED on the unrefactored tree (base a363ad7a /
e1b3ae3c's ng_snn.py) on 2026-10-04, cpu and cuda equal. The float32 1e12-bonus tie is preserved
byte-for-byte (its fix is its own later commit, owner question O5). C6's red: reversing the
admission tie-break must change the fingerprint. Every later stacked commit on
`neurogenesis_engine.py` (R2's hydration, R3's replay_rows, G7's persistence arrays) re-runs this
file. Each run builds a bare engine (no Brain, no lake writer).
"""

import json
import zlib

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.growth import candidate_lifecycle  # noqa: E402
from snn_data_lake.growth.neurogenesis_engine import (  # noqa: E402
    NeurogenesisSelfImprove,
    SyntheticPerceptStream,
    growth_fingerprint,
)

# (seed, ticks) -> (growth_fingerprint, trajectory crc, grown units); measured before the refactor
STREAM = {
    (9000, 200): (452866023, 11),
    (9001, 200): (3078134973, 7),
    (9002, 200): (97899718, 10),
    (9003, 200): (3362449400, 7),
    (9000, 600): (4153597545, 12),
    (9001, 600): (3409593730, 15),
    (9002, 600): (2073996705, 14),
    (9003, 600): (1976927636, 21),
}
TRAJ = {
    (9000, 200): 869405703,
    (9001, 200): 1746441905,
    (9002, 200): 3364797868,
    (9003, 200): 1141811203,
    (9000, 600): 1455820575,
    (9001, 600): 1991505230,
    (9002, 600): 2287058612,
    (9003, 600): 1128170985,
}
ONLINE = {
    (0, "batch"): (3259111491, 3695338077, 5),
    (0, "delta"): (3452081889, 3695338077, 5),
    (0, "delta_open"): (3249242654, 320847952, 13),
    (1, "batch"): (1808905792, 4245214917, 2),
    (1, "delta"): (459976505, 4245214917, 2),
    (1, "delta_open"): (2441742469, 934839829, 7),
}


def _traj(engine):
    return zlib.crc32(json.dumps(engine.trajectory, default=str).encode())


def _stream(seed, ticks):
    torch.set_num_threads(1)
    eng = NeurogenesisSelfImprove(seed=seed, device="cpu")
    SyntheticPerceptStream(seed=seed).drive(eng, ticks)
    return eng


@pytest.mark.parametrize("seed,ticks", sorted(STREAM))
def test_percept_consumer_is_byte_identical(seed, ticks):
    eng = _stream(seed, ticks)
    assert (growth_fingerprint(eng), int(eng.grown_units)) == STREAM[(seed, ticks)]
    assert _traj(eng) == TRAJ[(seed, ticks)]


@pytest.mark.parametrize("seed,name", sorted(ONLINE))
def test_online_readout_engines_are_byte_identical(seed, name):
    from snn_data_lake.readout.online_readout import ClosingGate, OnlineReadout, _run

    torch.set_num_threads(1)
    readout = None if name == "batch" else OnlineReadout("delta")
    gate = None if name == "delta_open" else ClosingGate(3)
    eng = _run(seed, readout, gate)
    assert (growth_fingerprint(eng), _traj(eng), int(eng.grown_units)) == ONLINE[(seed, name)]


def test_reversed_admission_tie_break_changes_the_fingerprint(monkeypatch):
    # C6's red: the equivalence check must be able to see a changed admission order
    original = candidate_lifecycle.select_block

    def reversed_order(*args, **kwargs):
        kwargs["reverse"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_lifecycle, "select_block", reversed_order)
    changed = [
        s
        for s in (9000, 9001, 9002, 9003)
        if growth_fingerprint(_stream(s, 200)) != STREAM[(s, 200)][0]
    ]
    assert changed, "reversing the tie-break changed no fingerprint: the equivalence check is blind"


def test_heldout_prune_is_consumer_agnostic():
    # the one-pass held-out discipline over any rate matrix: a duplicated column earns nothing
    g = torch.Generator().manual_seed(9900)
    S = torch.rand(40, 3, generator=g)
    S = torch.cat([S, S[:, :1]], 1)  # column 3 duplicates column 0
    y = torch.tensor([i % 2 for i in range(40)])
    Y = torch.zeros(40, 2)
    Y[torch.arange(40), y] = 1.0
    from snn_data_lake.growth.ng_snn import NGSNN

    solve = NGSNN(device="cpu")._solve_beta
    res = candidate_lifecycle.heldout_prune(S, Y, y.tolist(), solve)
    assert res.n_drop >= 1 and len(res.keep) >= 1
    assert set(res.keep) <= set(range(4)) and len(res.keep) + res.n_drop == 4


# ---- the trace-trained candidate source (2.5; 2.9 C7) -------------------------------------------


def _xor_items():
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features

    labels = ["GREET", "GOAL", "ARM"]
    learner = SpikingInteractionLearner(labels, n_in=52, seed=9900, t_post=6)
    texts = ["cat dog sun sky", "cat dog sky sun", "dog cat sun sky", "dog cat sky sun"]
    ys = [0, 1, 1, 0]
    items = []
    for t, y in zip(texts, ys):
        ranks = tuple(learner.encode(text_features(t, 52)))
        drives = [learner._drive(n, list(ranks)) for n in learner.neurons]
        onehot = [1.0 if c == y else 0.0 for c in range(len(labels))]
        items.append((ranks, onehot, candidate_lifecycle.label_probabilities(drives)))
    return items


def test_trace_training_is_deterministic_per_seed():
    items = _xor_items()
    a = candidate_lifecycle.trace_train([0.1] * 52, 1.0, 12, 8.0, items, 0.05, 20, seed=9901)
    b = candidate_lifecycle.trace_train([0.1] * 52, 1.0, 12, 8.0, items, 0.05, 20, seed=9901)
    c = candidate_lifecycle.trace_train([0.1] * 52, 1.0, 12, 8.0, items, 0.05, 20, seed=9902)
    assert a == b and a != c


def test_trace_training_moves_the_row():
    items = _xor_items()
    w0 = (0.1,) * 52
    assert candidate_lifecycle.trace_train(w0, 1.0, 12, 8.0, items, 0.05, 20, seed=9901) != w0
    assert candidate_lifecycle.trace_train(w0, 1.0, 12, 8.0, items, 0.0, 20, seed=9901) == w0


def test_contribution_scores_dead_and_constant_features_zero():
    resid = [[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]]
    assert candidate_lifecycle.contribution([0.0, 0.0, 0.0], resid) == 0.0
    assert candidate_lifecycle.contribution([0.3, 0.3, 0.3], resid) == 0.0
    assert candidate_lifecycle.contribution([1.0, 0.0, 0.2], resid) > 0.0


def _gate_findings(source):
    import ast

    from snn_data_lake.gates.eprop_transport_gate import _findings_in

    tree = ast.parse(source)
    return [f for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) for f in _findings_in(fn)]


def test_the_trace_rule_carries_no_weight_transport():
    import inspect

    assert _gate_findings(inspect.getsource(candidate_lifecycle)) == []


def test_a_transposed_readout_in_the_trace_rule_is_flagged():
    # C7's red: L_j read through the transposed readout weights must be caught by the gate
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(candidate_lifecycle.trace_train))
    variant = src.replace(
        "L = sum(b * (yc - pc) for b, yc, pc in zip(feedback, y, p))",
        "L = (W_out.T @ err)[0]",
    )
    assert variant != src
    assert _gate_findings(variant), "the transport gate did not flag a transposed readout"
