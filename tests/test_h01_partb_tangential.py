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
    assert pb.frame_violation(p.layer, p.fr[0]) == (0.0, 0)
    assert pb.frame_violation(q.layer, q.fr[0])[0] > 0.25


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
    h, _ = pb.build_hists(pop, 10020, n_all=n_all, truth=truth, n_edges=1200)
    assert int(h.n_fold.sum()) == int(h.n_block.sum()) == N * (N - 1)
    assert int(h.k_fold.sum()) == int(h.k_block.sum()) > 0
    assert 0.85 * 1200 < int(h.k_fold.sum()) < 1.15 * 1200
    h2, _ = pb.build_hists(pop, 10020, n_all=n_all, truth=truth, n_edges=1200)
    assert np.array_equal(h.k_fold, h2.k_fold) and np.array_equal(h.n_fold, h2.n_fold)
    h3, _ = pb.build_hists(pop, 10021, n_all=n_all, truth=truth, n_edges=1200)
    assert not np.array_equal(h.n_fold, h3.n_fold)


def test_planted_scale_refuses_a_clipped_world(pop, n_all):
    with pytest.raises(ValueError):
        pb.planted_scale({"form": "gauss", "width": 60.0}, pop, n_all, 10**8, np.ones(pb.N_CELL))


def test_kernel_world_beats_its_flat_twin_and_flat_world_does_not(pop, n_all):
    g = pb._unit(pop, 10030, n_all, {"form": "gauss", "width": 60.0}, 1200)
    f = pb._unit(pop, 10031, n_all, {"form": "flat"}, 1200)
    assert g["neuron"]["G-F"] > 100.0 and g["interior"]["G-F"] != 0.0
    assert all(r["G-F"] > 0 for r in g["region"])
    assert abs(f["neuron"]["G-F"]) < 0.05 * g["neuron"]["G-F"]


def test_gate_refuses_before_a_planted_record_and_on_a_perturbed_one(pop):
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(None, pop)
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted({"population_sha256": "x", "n_edges_target": 1}, pop)


def _st(g_f=1, p_g=1, z_gf=1, z_pg=1, r_gf=1, r_pg=1, blocks=(1, 1, 1, 1)):
    return {
        "region_blocks": [{"G-F": b, "P-G": 1} for b in blocks],
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
        ({"blocks": (1, 1, -1, 1)}, ("RED_PATHWAY_FLAT_REGION", None)),  # one region is enough
        ({"blocks": (1, 0, 1, 1)}, ("RED_PATHWAY_FLAT_REGION", None)),
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
        _mk("flat_calibration", {"form": "flat"}, 10040, None),
        _mk("flat_validation", {"form": "flat"}, 10048, "red_flat"),
        _mk("gauss", {"form": "gauss", "width": 60.0}, 10030, "red_pool"),
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
    pw = [pb._unit(p2, s, n_all, wide, 1500)["neuron"]["P-G"] for s in (10050, 10051, 10052)]
    nl = [pb._unit(p2, s, n_all, flat, 1500)["neuron"]["P-G"] for s in (10060, 10061, 10062)]
    assert min(pw) > 50.0 and max(nl) < 25.0


# ---- constructed reds, the pooling gate and the record (fast: no planted stage) ----
def test_unit_trap_nm_fed_as_um_is_refused(pop):
    with pytest.raises(t3.ArtifactRefused):
        pb.make_pop(pop.xyz * 1000.0, pop.layer, pop.sign)


def test_inner_split_never_trains_on_a_held_out_fold():
    n = np.ones((pb.K, pb.K, 1, 1), dtype=np.int64)
    k = np.zeros_like(n)
    f, g = 1, 3
    k[g, :, :, :] = 1  # a pair touching the inner-held-out fold g, any partner
    k[:, g, :, :] = 1
    k[f, :, :, :] = 1  # a pair touching the outer-held-out fold f
    k[:, f, :, :] = 1
    ntr, ktr, nte, kte = pb.inner_split(n, k, f, g)
    assert int(ktr.sum()) == 0 and int(kte.sum()) > 0  # nothing touching f or g is in train
    assert (
        int(ntr.sum()) == 3 * 3 and int(nte.sum()) == 4 * 4 - 3 * 3
    )  # test = pairs touching g, not f


def test_leak_check_catches_a_leaky_split():
    assert pb.leak_free([0, 2, 4], [0, 2, 3, 4], f=1, g=3)
    assert not pb.leak_free([0, 2, 3, 4], [0, 2, 3, 4], f=1, g=3)  # g left in train
    assert not pb.leak_free([0, 1, 2], [0, 1, 2, 3], f=1, g=3)  # f left in train


def test_unsighted_pooling_turns_pg_into_not_read_and_keeps_earlier_rows():
    assert pb.read_partb(_st(), pooling_sighted=False) == ("NOT_READ_POOLING_UNSIGHTED", None)
    assert pb.read_partb(_st(g_f=0), pooling_sighted=False) == ("RED_PATHWAY_FLAT", None)
    assert pb.read_partb(_st(), frame_ok=False, pooling_sighted=False) == ("REFUSED_FRAME", None)


def _fake_unit(seed, gf=0.0, pg=0.0, form="gauss", w=60.0, block_gf=None):
    z = {"G-F": gf, "P-G": pg}
    region = [dict(z) for _ in range(pb.NB_REG)]
    if block_gf is not None:
        region[0] = {"G-F": block_gf, "P-G": pg}
    return {
        "seed": seed,
        "neuron": dict(z),
        "interior": dict(z),
        "region": region,
        "forms": {
            "gauss": 10.0 if form == "gauss" else 0.0,
            "exp": 0.0 if form == "gauss" else 10.0,
            "mix": 11.0,
        },
        "th_forms": {f: [[math.log(w), math.log(w), 0.0]] * pb.K for f in pb.KINDS},
        "pen": [4.0] * pb.K,
        "edge_instances": 100,
    }


def _fake_rows(plan, pg_gate):
    rows = {}
    for r in plan:
        ramp = [100.0 + i for i in range(8)]
        if r.expect is None or r.expect == "red_flat":
            units = [_fake_unit(s) for s in r.seeds]
        elif r.expect == "pooling_gate":
            units = [
                _fake_unit(s, gf=ramp[i], pg=ramp[i] if pg_gate else 0.0)
                for i, s in enumerate(r.seeds)
            ]
        else:
            units = [
                _fake_unit(s, gf=ramp[i], form=r.truth["form"], w=r.truth.get("width", 60.0))
                for i, s in enumerate(r.seeds)
            ]
        rows[r.name] = {"units": units}
    return rows


def test_pooling_gate_is_a_gate_not_a_caveat(pop):
    plan = pb.PLAN
    for sighted in (True, False):
        rows = _fake_rows(plan, sighted)
        bands = pb.calibrate_bands(rows[plan[0].name]["units"])
        assert pb.pooling_sighted(plan, rows, bands) is sighted
    assert pb.pooling_sighted(plan[:2], rows, bands) is False  # no gate row at all: not sighted
    # the strict gate does not refuse an unsighted record; the reading does
    rec = {
        "population_sha256": pb.fingerprint(pop),
        "n_edges_target": 1200,
        "plan": [r._asdict() for r in plan],
        "rows": _fake_rows(plan, False),
        "bands": pb.calibrate_bands(_fake_rows(plan, False)[plan[0].name]["units"]),
    }
    assert pb.validate_planted(rec, pop, plan, 1200) == rec["bands"]
    rec["rows"]["gauss100"]["units"] = [
        _fake_unit(s, w=75.0) for s in plan[2].seeds
    ]  # width off by 25%
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(rec, pop, plan, 1200)


def test_record_is_readable_by_wiring_prior(pop, n_all):
    from snn_data_lake.growth import wiring_prior as wp

    truth = {"form": "gauss", "width": 60.0}
    h, _ = pb.build_hists(pop, 10020, n_all=n_all, truth=truth, n_edges=1200)
    rng = np.random.default_rng(0)
    edges = (rng.integers(0, N, 400), rng.integers(0, N, 400))
    units = [_fake_unit(10020 + i) for i in range(8)]
    rec = pb.build_record(
        pop, h, units, "PASS", "pooled", edges, "a" * 64, {"blob_sha256": "b" * 64}
    )
    for key in (
        "source",
        "frame",
        "a",
        "c",
        "sigma_um",
        "lambda_um",
        "beta",
        "beta_fallback",
        "dbar",
    ):
        assert key in rec
    assert rec["beta"] == 0.0 and rec["beta_fallback"] is True and rec["pathway_amplitudes"]
    assert (
        20.0 < rec["sigma_um"] < 200.0
    )  # the planted 60 um, recovered to within a factor of a few
    cells = {int(c): v for c, v in rec["pathway_amplitudes"].items()}
    assert all(0.0 < v for v in cells.values())
    a = {"unit_id": "a", "position": (0, 0, 0.0, 0.0, 0.5), "layer": "Layer 4", "sign": 1}
    b = dict(a, unit_id="b", position=(0, 0, 25.0, 0.0, 0.5))
    p = wp.attach_probabilities([a], [b], rec)[0, 0]
    assert (
        0.0 < p < 1.0
        and p < wp.attach_probabilities([a], [dict(b, position=(0, 0, 1.0, 0.0, 0.5))], rec)[0, 0]
    )


def test_region_blocks_use_the_nested_training_only_split():
    n = np.ones((pb.NB_REG, pb.NB_REG, 1, 1), dtype=np.int64)
    k = np.zeros_like(n)
    k[2, :], k[:, 2], k[0, :], k[:, 0] = (
        1,
        1,
        1,
        1,
    )  # touching the inner-held-out block 2 or outer block 0
    ntr, ktr, nte, kte = pb.inner_split(n, k, 0, 2)
    assert int(ktr.sum()) == 0 and int(ntr.sum()) == 2 * 2 and int(nte.sum()) == 3 * 3 - 2 * 2


def test_development_seeds_only_in_the_synthetic_tests():
    import pathlib
    import re

    lines = pathlib.Path(__file__).read_text(encoding="utf-8").splitlines()
    text = " ".join(ln for ln in lines if "campaign" not in ln and "findall" not in ln)
    campaign = set(range(10100, 10300)) | set(range(10300, 10358)) | {10007}
    drawn = {int(m) for m in re.findall(r"\b(1\d{4})\b", text)}
    assert not drawn & campaign, sorted(drawn & campaign)


def test_frame_statistic_is_soma_level_and_a_small_disorder_is_a_small_fraction():
    r = np.linspace(2000.0, 1000.0, 1000)  # pia outward first
    layer = np.repeat(np.arange(7), [143, 143, 143, 143, 143, 143, 142])
    assert pb.frame_violation(layer, r) == (0.0, 0)  # a clean boundary is not a violation
    mixed = layer.copy()
    mixed[500], mixed[501] = mixed[501] + 3, mixed[500]  # one deep soma stranded among shallow ones
    frac, count = pb.frame_violation(mixed, r)
    assert (
        0 < count <= 2 * pb.FRAME_WINDOW + 1 and frac == count / 1000
    )  # per soma, not per cross-band pair


def test_flat_validation_bar_is_no_positive_under_any_scoring():
    row = pb.PLAN[1]
    ramp = [50.0 + i for i in range(8)]
    calib = [_fake_unit(s) for s in pb.PLAN[0].seeds]
    bands = pb.calibrate_bands(calib)
    quiet = [_fake_unit(s) for s in row.seeds]
    assert pb._holds(row, quiet, bands)
    # a positive in ONE region block alone (neuron folds quiet): RED verdict, but the bar fails
    leak = [_fake_unit(s, block_gf=ramp[i]) for i, s in enumerate(row.seeds)]
    assert not pb._holds(row, leak, bands)
    assert not pb.no_positive_kernel(pb.states(leak, bands))


T3_PATH = t3.REPO / "tools" / "g4_t3_campaign.json"


def test_t3_gate_accepts_the_real_record_and_refuses_each_defect(tmp_path):
    import json

    ref = pb.check_t3_record(T3_PATH)
    assert (
        ref["verdict"] == "RED_EXP" and ref["K"] == "mix" and ref["git_head"] and ref["blob_sha256"]
    )
    good = json.loads(T3_PATH.read_text(encoding="utf-8"))

    def defect(name, edit):
        rec = json.loads(json.dumps(good))
        edit(rec)
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(pb.ExposureRefused):
            pb.check_t3_record(f)

    defect("sha", lambda r: r.__setitem__("artifact_sha256", "0" * 64))
    defect("verdict", lambda r: r["h01"]["reading"].__setitem__("verdict", "RED_FLAT"))
    defect("refused", lambda r: r["h01"]["reading"].__setitem__("verdict", "REFUSED"))
    defect("nokernel", lambda r: r["h01"]["reading"].__setitem__("K", None))
    defect("head", lambda r: r.pop("git_head"))
    defect("harness", lambda r: r.pop("harness_sha256"))


def test_score_h01_needs_a_t3_ref_and_provenance_names_the_harness():
    with pytest.raises(pb.ExposureRefused):
        pb.score_h01("data/h01/h01_c3_graph.npz", {"rows": {}}, t3_ref=None) if (
            t3.REPO / "data/h01/h01_c3_graph.npz"
        ).exists() else (_ for _ in ()).throw(pb.ExposureRefused("fixture"))
    prov = pb.provenance()
    assert len(prov["git_head"]) == 40 and set(prov["harness_sha256"]) == set(pb.HARNESS)


def test_planted_plan_is_compared_in_full_not_by_names(pop):
    plan = pb.PLAN
    rows = _fake_rows(plan, False)
    rec = {
        "population_sha256": pb.fingerprint(pop),
        "n_edges_target": 1200,
        "plan": pb.plan_dicts(plan),
        "rows": rows,
        "bands": pb.calibrate_bands(rows[plan[0].name]["units"]),
    }
    assert pb.validate_planted(rec, pop, plan, 1200) == rec["bands"]
    for edit in (
        lambda p: p[2]["truth"].__setitem__(
            "width", 150.0
        ),  # a different truth under the same name
        lambda p: p[3].__setitem__("expect", "red_flat"),  # a weakened expectation
        lambda p: p[2].__setitem__("seeds", p[2]["seeds"][::-1]),  # reordered seeds
        lambda p: p.reverse(),  # reordered rows
    ):
        bad = dict(rec, plan=pb._plan_json(pb.plan_dicts(plan)))
        edit(bad["plan"])
        with pytest.raises(pb.ExposureRefused):
            pb.validate_planted(bad, pop, plan, 1200)
