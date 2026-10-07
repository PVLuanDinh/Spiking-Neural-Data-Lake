"""G4 Part C fixtures (g4-wiring-prior.md 3.2, 2.14 A13): the rule on SYNTHETIC label records.

No H01 file, no Brain, no lake, no deciding number. The record the rule reads is a Part B record
(tangential frame); T3's 3-D soma-distance record is refused by construction. Records are the label
dicts of `placement.place_labeled` (unit_id, position = (ap, dv, x_um, y_um, depth), layer, sign).

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_wiring_prior.py -q --no-cov
"""

import math

import numpy as np
import pytest

from snn_data_lake.growth import wiring_prior as wp
from snn_data_lake.reservoir.cortical_microcircuit import CONN_PROB

REC = {
    "source": "part_b",
    "frame": "tangential",
    "a": 1.0,
    "sigma_um": 50.0,
    "c": 0.0,
    "lambda_um": 100.0,
}


def _u(uid, x, y, layer="Layer 4", sign=1, cell=(0, 0)):
    return {"unit_id": uid, "position": (cell[0], cell[1], x, y, 0.5), "layer": layer, "sign": sign}


def _p(src, dst, rec=REC, partners=None):
    return wp.attach_probabilities([src], [dst], rec, partners=partners)[0, 0]


def test_refuses_the_3d_t3_record():
    t3 = {
        "artifact_sha256": "a1d0491a",
        "model": "mix",
        "width_um": {"sigma": 42.757, "lambda": 104.266},
    }
    with pytest.raises(wp.RecordRefused):
        _p(_u("a", 0, 0), _u("b", 10, 0), t3)


def test_refuses_a_non_tangential_part_b_record():
    with pytest.raises(wp.RecordRefused):
        _p(_u("a", 0, 0), _u("b", 10, 0), dict(REC, frame="3d"))


@pytest.mark.parametrize("layer", ["Layer 1", "White matter"])
def test_unmodelled_layer_is_refused_explicitly(layer):
    with pytest.raises(wp.LayerNotModelled):
        _p(_u("a", 0, 0, layer=layer), _u("b", 10, 0))


def test_layer_target_reads_conn_prob_at_zero_distance():
    # source L4e (pop 2) -> target L5e (pop 4): CONN_PROB[target][source]
    got = _p(_u("a", 0, 0, "Layer 4", 1), _u("b", 0, 0, "Layer 5", 1))
    assert math.isclose(got, CONN_PROB[4][2])


def test_l2_and_l3_aggregate_to_one_population():
    l2 = _p(_u("a", 0, 0, "Layer 2", -1), _u("b", 0, 0, "Layer 4", 1))
    l3 = _p(_u("a", 0, 0, "Layer 3", -1), _u("b", 0, 0, "Layer 4", 1))
    assert l2 == l3 == CONN_PROB[2][1]


def test_kernel_falls_with_tangential_distance_and_ignores_depth():
    near, far = _u("n", 10, 0), _u("f", 200, 0)
    src = _u("s", 0, 0)
    assert _p(src, near) > _p(src, far) > 0
    deep = dict(near, position=(0, 0, 10, 0, 0.9))
    assert _p(src, deep) == _p(src, near)


def test_gaussian_core_value():
    d = 60.0
    want = CONN_PROB[2][2] * math.exp(-(d**2) / (2 * 50.0**2))
    assert math.isclose(_p(_u("s", 0, 0), _u("t", d, 0)), want)


def test_no_kernel_across_map_cells():
    assert _p(_u("s", 0, 0, cell=(0, 0)), _u("t", 1, 0, cell=(1, 0))) == 0.0


def test_shared_partner_bonus_and_beta_zero_default():
    s, t = _u("s", 0, 0), _u("t", 30, 0)
    base = _p(s, t)
    partners = {"s": {"x", "y"}, "t": {"x", "y", "z"}}
    assert _p(s, t, dict(REC, beta=0.0), partners) == base
    assert math.isclose(_p(s, t, dict(REC, beta=0.5), partners), min(1.0, base * 2.0))


def test_probabilities_are_in_unit_interval_and_shape_is_new_by_existing():
    new = [_u(f"n{i}", i, 0) for i in range(3)]
    old = [_u(f"o{i}", 0, i, "Layer 5", -1) for i in range(5)]
    p = wp.attach_probabilities(new, old, dict(REC, beta=10.0), partners={"n0": {"q"}, "o0": {"q"}})
    assert p.shape == (3, 5) and np.all((p >= 0) & (p <= 1))


def test_deterministic_and_inputs_not_mutated():
    new, old = [_u("n", 0, 0)], [_u("o", 5, 5)]
    snap = repr((new, old))
    a = wp.attach_probabilities(new, old, REC)
    b = wp.attach_probabilities(new, old, REC)
    assert np.array_equal(a, b) and repr((new, old)) == snap
