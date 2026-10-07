"""G4 Part B instrument (amendment 2026-10-07): the tangential frame, the F / G / P arms, the
planted generator, the reading and the exposure gate, on SYNTHETIC populations (a few hundred
units near the white-matter circle); no H01 file, no network, no Brain, no lake.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_h01_partb_tangential.py -q --no-cov
"""

import math

import numpy as np
import pytest

from snn_data_lake.instruments import h01_distance_kernel as t3
from snn_data_lake.instruments import h01_partb_tangential as pb

N = 240


def _pop(seed=0, n=N, monotone=True):
    rng = np.random.default_rng(seed)
    cx, cy = t3.region_centre()
    r = rng.uniform(1500.0, 1800.0, n)
    th = rng.uniform(-1.0, -0.85, n)
    z = rng.uniform(0.0, 60.0, n)
    band = np.minimum(((1800.0 - r) / 300.0 * 7).astype(int), 6)  # layer 0 outermost
    if not monotone:
        band = rng.permutation(band)
    xyz = np.c_[cx + r * np.cos(th), cy + r * np.sin(th), z]
    return pb.make_pop(xyz, band, rng.choice([-1, 1], n))


@pytest.fixture(scope="module")
def pop():
    return _pop()


@pytest.fixture(scope="module")
def n_all(pop):
    return pb._all_pairs_n(pop)


def test_frame_violation_monotone_vs_shuffled():
    p, q = _pop(), _pop(monotone=False)
    assert pb.frame_violation(p.layer, p.fr[0]) == 0.0
    assert pb.frame_violation(q.layer, q.fr[0]) > 0.25


def test_dtan_wraps_theta_and_ignores_nothing_in_z():
    fr = (
        np.array([1000.0, 1000.0]),
        np.array([math.pi - 0.005, -math.pi + 0.005]),
        np.array([0.0, 30.0]),
    )
    d = pb.d_tan(fr, np.array([0]), np.array([1]))[0]
    assert math.isclose(
        d, math.hypot(1000.0 * 0.01, 30.0), rel_tol=1e-9
    )  # unwrapped it would be ~6283


def test_all_pairs_counts_every_ordered_pair_once(pop, n_all):
    assert int(n_all.sum()) == N * (N - 1)


def test_hists_partition_the_same_pairs_and_are_deterministic(pop, n_all):
    truth = {"form": "gauss", "width": 60.0}
    h, _ = pb.build_hists(pop, 10350, n_all=n_all, truth=truth, n_edges=1200)
    assert int(h.n_fold.sum()) == int(h.n_block.sum()) == N * (N - 1)
    assert int(h.k_fold.sum()) == int(h.k_block.sum()) > 0
    assert 0.85 * 1200 < int(h.k_fold.sum()) < 1.15 * 1200
    h2, _ = pb.build_hists(pop, 10350, n_all=n_all, truth=truth, n_edges=1200)
    assert np.array_equal(h.k_fold, h2.k_fold) and np.array_equal(h.n_fold, h2.n_fold)
    h3, _ = pb.build_hists(pop, 10351, n_all=n_all, truth=truth, n_edges=1200)
    assert not np.array_equal(h.n_fold, h3.n_fold)


def test_planted_scale_refuses_a_clipped_world(pop, n_all):
    with pytest.raises(ValueError):
        pb.planted_scale({"form": "gauss", "width": 60.0}, pop, n_all, 10**8, np.ones(pb.N_CELL))


def test_kernel_world_beats_its_flat_twin_and_flat_world_does_not(pop, n_all):
    g = pb._unit(pop, 10316, n_all, {"form": "gauss", "width": 60.0}, 1200)
    f = pb._unit(pop, 10308, n_all, {"form": "flat"}, 1200)
    assert g["neuron"]["G-F"] > 100.0 and g["interior"]["G-F"] != 0.0
    assert all(r["G-F"] > 0 for r in g["region"])
    assert abs(f["neuron"]["G-F"]) < 0.05 * g["neuron"]["G-F"]


def test_gate_refuses_before_a_planted_record_and_on_a_perturbed_one(pop):
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(None, pop)
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted({"population_sha256": "x", "n_edges_target": 1}, pop)


def _st(g_f=1, p_g=1, z_gf=1, z_pg=1, r_gf=1, r_pg=1):
    return {
        "neuron": {"G-F": g_f, "P-G": p_g},
        "interior": {"G-F": z_gf, "P-G": z_pg},
        "region_aggregate": {"G-F": r_gf, "P-G": r_pg},
    }


@pytest.mark.parametrize(
    "kw, want",
    [
        ({}, ("PASS", "pooled")),
        ({"p_g": 0}, ("RED_POOL", "global")),
        ({"g_f": 0}, ("RED_PATHWAY_FLAT", None)),
        ({"g_f": -1}, ("RED_PATHWAY_FLAT", None)),
        ({"z_pg": -1}, ("NOT_READ_BOUNDARY", None)),
        ({"r_gf": -1}, ("NOT_READ_DISAGREE", None)),
        ({"r_pg": 0, "z_pg": 0}, ("PASS", "pooled")),  # an unresolved state never vetoes
    ],
)
def test_reading_rows(kw, want):
    assert pb.read_partb(_st(**kw)) == want


def test_reading_refusals_come_first():
    assert pb.read_partb(_st(g_f=0), frame_ok=False) == ("REFUSED_FRAME", None)
    assert pb.read_partb(_st(), planted_ok=False) == ("REFUSED", None)


def test_band_boundary_is_strict_and_degenerate_is_not_met():
    vals = [5.0] * 8
    assert pb._state(vals, 1.0) == 0  # zero variance: NOT MET whatever its value (ruling 9)
    vals = [5.0 + 0.1 * i for i in range(8)]
    assert pb._state(vals, 1.0) == 1
    assert pb._state(vals, 6.0) == 0  # mean 5.35 does not exceed the band


def test_dbar_on_a_known_graph():
    layer = np.array([1, 1, 2, 0, 6])
    pre = np.array([0, 1, 0, 2, 3, 4])
    post = np.array([1, 0, 2, 3, 0, 1])
    got = pb.dbar(layer, pre, post)
    # in-layer nodes {0, 1, 2}; edges 0-1 (both directions), 0-2; 2->3 and 3->0 leave the layers
    assert got["N"] == 3 and got["E"] == 2 and got["dbar"] == 4 / 3 and got["reciprocal_pairs"] == 1


def test_tangential_density_is_positive_per_band(pop):
    d = pb.tangential_density(pop)
    assert d["pooled"] > 0 and all(v > 0 for v in d.values())


def _mk(name, truth, first, expect):
    return pb.PlanRow(name, truth, tuple(range(first, first + 8)), expect)


def _mini_plan():
    return (
        _mk("flat_calibration", {"form": "flat"}, 10300, None),
        _mk("flat_validation", {"form": "flat"}, 10308, "red_flat"),
        _mk("gauss", {"form": "gauss", "width": 60.0}, 10316, "red_pool"),
    )


def test_planted_stage_holds_and_the_gate_recomputes_it(pop):
    plan = _mini_plan()
    rec = pb.run_planted(pop, plan=plan, n_edges=1200)
    assert rec["all_rows_hold"], {k: v.get("reading") for k, v in rec["rows"].items()}
    assert pb.validate_planted(rec, pop, plan, 1200) == rec["bands"]
    bad = dict(rec, bands={**rec["bands"], "neuron": {c: 0.0 for c in pb.CONTRASTS}})
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(bad, pop, plan, 1200)
    other = dict(rec, population_sha256="0" * 64)
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(other, pop, plan, 1200)


def test_pooling_positive_control_and_its_null():
    """Pathway widths planted -> P beats G by a wide margin; the same world without them -> P does
    not beat G (RED_POOL is reachable). Two classes, so every cell is populous."""
    rng = np.random.default_rng(1)
    p0 = _pop()
    p2 = pb.make_pop(p0.xyz, np.where(rng.random(N) < 0.5, 1, 2), np.ones(N, dtype=int))
    n_all = pb._all_pairs_n(p2)
    wide = {"form": "gauss", "width": 60.0, "pathway": (20.0, 250.0)}
    flat = {"form": "gauss", "width": 60.0}
    pw = [pb._unit(p2, s, n_all, wide, 1500)["neuron"]["P-G"] for s in (10332, 10333, 10334)]
    nl = [pb._unit(p2, s, n_all, flat, 1500)["neuron"]["P-G"] for s in (10340, 10341, 10342)]
    assert min(pw) > 50.0 and max(nl) < 25.0
