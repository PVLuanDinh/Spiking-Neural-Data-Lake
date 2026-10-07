"""G4 Part B instrument (amendment 2026-10-07): the tangential frame, the F / G / P arms, the
planted generator, the reading and the exposure gate, on SYNTHETIC populations (a few hundred
units near the white-matter circle); no H01 file, no network, no Brain, no lake.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_h01_partb_tangential.py -q --no-cov
"""

import json
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


_POP = _pop()
_PATH_CELLS = pb.pathway_expected(next(r.truth for r in pb.PLAN if r.truth.get("pathway")), _POP)[0]


@pytest.fixture(scope="module")
def pop():
    return _POP


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


def _fake_unit(seed, gf=0.0, pg=0.0, form="gauss", w=60.0, block_gf=None, pathway=None):
    z = {"G-F": gf, "P-G": pg}
    region = [dict(z) for _ in range(pb.NB_REG)]
    if block_gf is not None:
        region[0] = {"G-F": block_gf, "P-G": pg}
    chk = {
        "cells": list(_PATH_CELLS),
        "planted": [60.0, 160.0],
        "fitted": list(pathway or (60.0, 160.0)),
    }
    return {
        "seed": seed,
        "pathway_check": chk,
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


def _fb(x):
    """A form-band pair, the same width for both comparisons."""
    return {"form_gap_gauss": x, "form_gap_exp": x}


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
        assert pb.pooling_sighted(plan, rows, bands, _POP) is sighted
    assert (
        pb.pooling_sighted(plan[:2], rows, bands, _POP) is False
    )  # no gate row at all: not sighted
    # the strict gate does not refuse an unsighted record; the reading does
    rec = {
        "population_sha256": pb.fingerprint(pop),
        "n_edges_target": 1200,
        "identity": pb.identity(),
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
    bands = _fb(0.5)  # constant gaps: the mixture is not a MET upgrade, the Gaussian is selected
    t3_ref = {"blob_sha256": "b" * 64, "K": "mix"}
    rec = pb.build_record(pop, h, units, "PASS", "pooled", edges, "a" * 64, t3_ref, bands)
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
    assert pb._holds(row, quiet, bands, _POP)
    # a positive in ONE region block alone (neuron folds quiet): RED verdict, but the bar fails
    leak = [_fake_unit(s, block_gf=ramp[i]) for i, s in enumerate(row.seeds)]
    assert not pb._holds(row, leak, bands, _POP)
    assert not pb.no_positive_kernel(pb.states(leak, bands))


T3_PATH = t3.REPO / "tools" / "g4_t3_campaign.json"


def test_t3_gate_accepts_the_real_record_and_refuses_each_defect(tmp_path):
    import json

    ref = pb.check_t3_record(T3_PATH)
    assert (
        ref["verdict"] == "RED_EXP" and ref["K"] == "mix" and ref["git_head"] and ref["blob_sha256"]
    )
    good = json.loads(T3_PATH.read_text(encoding="utf-8"))

    def defect(name, edit, why):
        rec = json.loads(json.dumps(good))
        edit(rec)
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(pb.ExposureRefused, match=why):
            pb.check_t3_record(f)

    defect("sha", lambda r: r.__setitem__("artifact_sha256", "0" * 64), "different artifact")
    defect(
        "verdict", lambda r: r["h01"]["reading"].__setitem__("verdict", "RED_FLAT"), "not a READ"
    )
    defect("refused", lambda r: r["h01"]["reading"].__setitem__("verdict", "REFUSED"), "not a READ")
    defect("nokernel", lambda r: r["h01"]["reading"].__setitem__("K", None), "not a READ")
    defect("head", lambda r: r.pop("git_head"), "no git head")
    defect("harness", lambda r: r.pop("harness_sha256"), "no git head")
    # F7 (fold-lens round 2): fields that all look right on a file that is NOT the committed blob
    fake = json.loads(json.dumps(good))
    fake["git_head"], fake["harness_sha256"] = "fake", {"x": "y"}
    f = tmp_path / "fake.json"
    f.write_text(json.dumps(fake), encoding="utf-8")
    with pytest.raises(pb.ExposureRefused, match="committed"):
        pb.check_t3_record(f)
    copy = tmp_path / "copy.json"
    copy.write_bytes(T3_PATH.read_bytes())  # the same bytes anywhere are the same record
    assert pb.check_t3_record(copy)["blob_sha256"] == ref["blob_sha256"]


def test_score_h01_demands_the_verified_t3_reference_not_any_truthy_value():
    ref = pb.check_t3_record(T3_PATH)
    pb.verify_t3_ref(ref)
    for bad in (
        {"K": "mix"},
        {**ref, "blob_sha256": "0" * 64},
        {**ref, "verdict": "REFUSED"},
        {**ref, "K": None},
        "yes",
    ):
        with pytest.raises(pb.ExposureRefused):
            pb.verify_t3_ref(bad)
    with pytest.raises(pb.ExposureRefused, match="T3"):
        pb.score_h01("no-such-artifact.npz", {"rows": {}}, t3_ref={"K": "mix"})


def test_score_h01_needs_a_t3_ref_and_provenance_names_the_harness():
    with pytest.raises(pb.ExposureRefused):
        pb.score_h01(
            "no-such-artifact.npz", {"rows": {}}, t3_ref=None
        )  # refused before any file is read
    prov = pb.provenance()
    assert len(prov["git_head"]) == 40 and set(prov["harness_sha256"]) == set(pb.HARNESS)


def test_planted_plan_is_compared_in_full_not_by_names(pop):
    plan = pb.PLAN
    rows = _fake_rows(plan, False)
    rec = {
        "population_sha256": pb.fingerprint(pop),
        "n_edges_target": 1200,
        "identity": pb.identity(),
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


def test_form_bands_are_calibrated_for_the_contrast_selection_uses():
    """F4 (fold-lens round 2): the band for 'mixture over the chosen single form' is calibrated on
    that contrast, one band per single form, not on the mixture over the better of the two."""

    def units(triples):
        return [
            dict(_fake_unit(10020 + i), forms={"gauss": g, "exp": e, "mix": m})
            for i, (g, e, m) in enumerate(triples * 4)
        ]

    cal = units(
        [(100.0, 0.0, 101.0), (0.0, 100.0, 101.0)]
    )  # the mixture beats the better form by 1
    bands = pb.calibrate_bands(cal)
    assert bands["form_gap_gauss"] == 101.0 and bands["form_gap_exp"] == 101.0
    # its edge over the CHOSEN single form swings 1 / 101 (mean 51): inside its own band, no upgrade
    assert pb.select_form(cal, bands) == "gauss"
    clear = units(
        [(0.0, -500.0, 300.0), (0.0, -500.0, 310.0)]
    )  # a large, varying advantage over Gaussian
    assert pb.select_form(clear, _fb(1.0)) == "mix"


def _banded_frame(order):
    """Somata whose r falls from 1800 to 1500 um; `order` is each one's band, outermost first."""
    n = len(order)
    r = np.linspace(1800.0, 1500.0, n)
    th = np.linspace(-1.0, -0.85, n)
    cx, cy = t3.region_centre()
    xyz = np.c_[cx + r * np.cos(th), cy + r * np.sin(th), np.linspace(0.0, 60.0, n)]
    return xyz, np.asarray(order)


def test_a_wholly_inverted_band_order_is_refused_though_few_somata_straddle_a_boundary(monkeypatch):
    """M1 / F3 (fold-lens round 2): bands [2]*333 + [1]*333 + [0]*334 from the outside in is a
    reversed frame that only 20 / 1000 somata can see at W = 5; the whole-band order catches it."""
    inverted = [2] * 333 + [1] * 333 + [0] * 334
    xyz, layer = _banded_frame(inverted)
    pop = pb.make_pop(xyz, layer, np.ones(len(layer), dtype=int))
    assert (
        pb.frame_violation(pop.layer, pop.fr[0])[0] <= pb.FRAME_VIOLATION_MAX
    )  # the soma statistic is blind
    assert pb.frame_order_refused(pop.layer, pop.fr[0])
    xyz, layer = _banded_frame(sorted(inverted))  # the right way round: deeper bands further in
    ok = pb.make_pop(xyz, layer, np.ones(len(layer), dtype=int))
    assert not pb.frame_order_refused(ok.layer, ok.fr[0])
    assert not pb.frame_order_refused(
        np.array([0, 0, 2, 2]), np.array([9.0, 8.0, 7.0, 6.0])
    )  # a gap band
    z = {
        "xyz_nm": pop.xyz * 1000.0,
        "layer": pop.layer,
        "sign": pop.sign,
        "pre_idx": [0],
        "post_idx": [1],
    }
    monkeypatch.setattr(t3, "_verified", lambda path, pin=None: (z, "s" * 64))
    monkeypatch.setattr(pb, "validate_planted", lambda *a, **k: {})
    out = pb.score_h01("x.npz", {}, t3_ref=pb.check_t3_record(T3_PATH))
    assert out["reading"] == "REFUSED_FRAME"


def test_a_source_change_during_the_campaign_cannot_be_stamped_with_the_new_identity(
    pop, monkeypatch
):
    """S3 / F8 (fold-lens round 2): results drawn under source A must not leave stamped with B's
    hashes, in the planted stage or in the H01 read."""
    state = {"src": "A"}
    monkeypatch.setattr(pb, "_file_sha", lambda rel: state["src"])
    real_unit = pb._unit

    def unit(*a, **k):
        state["src"] = "B"  # the file changes on disk while the campaign draws
        return real_unit(*a, **k)

    monkeypatch.setattr(pb, "_unit", unit)
    plan = (pb.PlanRow("flat_calibration", {"form": "flat"}, (10020, 10021), None),)
    with pytest.raises(pb.ExposureRefused, match="changed"):
        pb.run_planted(pop, plan=plan, n_edges=600)
    state["src"] = "A"
    monkeypatch.setattr(pb, "_unit", real_unit)
    assert pb.run_planted(pop, plan=plan, n_edges=600)["identity"]["impl_sha256"]  # no drift: fine
    # the H01 read: same guard around its draws
    z = {
        "xyz_nm": pop.xyz * 1000.0,
        "layer": pop.layer,
        "sign": pop.sign,
        "pre_idx": [0, 1],
        "post_idx": [1, 2],
    }
    monkeypatch.setattr(t3, "_verified", lambda path, pin=None: (z, "s" * 64))
    monkeypatch.setattr(pb, "validate_planted", lambda *a, **k: {})

    def flip_unit(*a, **k):
        state["src"] = "B"
        return {}

    monkeypatch.setattr(pb, "score_unit", flip_unit)
    state["src"] = "A"
    with pytest.raises(pb.ExposureRefused, match="changed"):
        pb.score_h01("x.npz", {}, t3_ref=pb.check_t3_record(T3_PATH))


def test_h01_read_holds_one_seeds_histograms_at_a_time(monkeypatch, pop):
    """s2 / F10 (fold-lens round 2): the previous seed's histogram set is freed before the next
    one is allocated (about 1.7 GB each at the archived 4,000-bin size)."""
    import weakref

    refs, alive_at_build = [], []

    def build(pop_, seed, edges=None, **kw):
        alive_at_build.append(sum(r() is not None for r in refs))
        z = np.zeros((1, 1, 1, 1), dtype=np.int64)
        h = pb.Hists(*([z] * 6))
        refs.append(weakref.ref(h))
        return h, None

    monkeypatch.setattr(pb, "build_hists", build)
    monkeypatch.setattr(pb, "score_unit", lambda pop_, h: {})
    units, first = pb._score_seeds(pop, (np.zeros(1), np.ones(1)), (1, 2, 3))
    assert alive_at_build == [0, 0, 0] and [u["seed"] for u in units] == [1, 2, 3]
    assert first[0].shape == (1, 1) and len(refs) == 3


def test_pathway_recovery_is_derived_from_the_authorised_truth_not_the_record():
    """F6 (fold-lens round 2): the record's own `planted` and `cells` are never trusted."""
    truth = next(r.truth for r in pb.PLAN if r.truth.get("pathway"))
    cells, planted = pb.pathway_expected(truth, _POP)
    assert len(cells) == 2 and planted == [60.0, 160.0]

    def rec(**edit):
        chk = {"cells": list(cells), "planted": list(planted), "fitted": [60.0, 160.0]}
        return [{"pathway_check": {**chk, **edit}}]

    assert pb.pathway_recovered(truth, rec(), _POP)
    assert not pb.pathway_recovered(truth, rec(planted=[], fitted=[75.0, 190.0]), _POP)
    assert not pb.pathway_recovered(truth, rec(fitted=[75.0, 190.0], planted=[75.0, 190.0]), _POP)
    assert not pb.pathway_recovered(
        truth, rec(cells=[1, 2]), _POP
    )  # wrong pathways, self-consistent
    assert not pb.pathway_recovered(truth, rec(fitted=[60.0]), _POP)
    assert not pb.pathway_recovered(truth, rec(fitted=[60.0, float("nan")]), _POP)
    assert not pb.pathway_recovered(truth, [{}], _POP)
    assert not pb.pathway_recovered(
        {"form": "gauss", "width": 100.0}, rec(), _POP
    )  # no pathway truth


def test_pathway_recovery_is_enforced_by_the_gate_on_both_widths():
    plan = pb.PLAN
    base = _fake_rows(plan, True)
    bands = pb.calibrate_bands(base[plan[0].name]["units"])
    assert pb.pooling_sighted(plan, base, bands, _POP) is True
    gate = next(r for r in plan if r.expect == "pooling_gate")
    cases = (
        ((60.0, 160.0), True),
        ((68.0, 150.0), True),
        ((75.0, 160.0), False),
        ((60.0, 190.0), False),
        ((60.0, 160.0, 1.0), False),
    )
    for fitted, ok in cases:
        units = [
            dict(u, pathway_check={**u["pathway_check"], "fitted": list(fitted)})
            for u in base[gate.name]["units"]
        ]
        rows = {**base, gate.name: {"units": units}}
        assert pb.pooling_sighted(plan, rows, bands, _POP) is ok, fitted  # P-G fires in every case


def test_exported_form_is_the_scored_form_and_the_selection_is_disclosure_only(pop, n_all):
    """F1 (fold-lens round 2): the verdict was earned by the mixture's G / P scores, so the mixture
    is what the record exports whatever form the tangential selection prefers; the selection and
    its disagreement with T3's K are recorded."""
    truth = {"form": "gauss", "width": 60.0}
    h, _ = pb.build_hists(pop, 10020, n_all=n_all, truth=truth, n_edges=1200)
    edges = (np.zeros(2, dtype=int), np.ones(2, dtype=int))

    def rec(units, t3_k, gap):
        return pb.build_record(
            pop, h, units, "PASS", "global", edges, "a" * 64, {"K": t3_k}, _fb(gap)
        )

    gauss_units = [_fake_unit(10020 + i) for i in range(8)]
    r = rec(gauss_units, "mix", 0.5)
    assert pb.select_form(gauss_units, _fb(0.5)) == "gauss"
    assert (
        r["form"] == "mix" and r["a"] == 1.0 and r["c"] > 0.0
    )  # the scored kernel, not the selected one
    assert r["form_selected"] == "gauss" and r["form_disagrees_t3"] is True
    exp_units = [_fake_unit(10020 + i, form="exp") for i in range(8)]
    r = rec(exp_units, "exp", 0.5)
    assert r["form"] == "mix" and r["form_selected"] == "exp" and r["form_disagrees_t3"] is False
    mix_units = [_fake_unit(10020 + i) for i in range(8)]
    for i, u in enumerate(mix_units):
        u["forms"] = {
            "gauss": 10.0,
            "exp": 0.0,
            "mix": 20.0 + i,
        }  # a varying, large mixture advantage
    r = rec(mix_units, "mix", 5.0)
    assert r["form"] == "mix" and r["form_selected"] == "mix" and r["form_disagrees_t3"] is False
    assert pb.select_form(mix_units, _fb(50.0)) == "gauss"  # inside the band: no upgrade


def test_pooled_record_amplitudes_belong_to_the_delivered_widths_and_the_consumer_uses_them(
    pop, n_all
):
    from snn_data_lake.growth import wiring_prior as wp

    p2 = pb.make_pop(
        pop.xyz, np.where(np.random.default_rng(1).random(N) < 0.5, 1, 2), np.ones(N, dtype=int)
    )
    h, _ = pb.build_hists(
        p2,
        10020,
        n_all=pb._all_pairs_n(p2),
        truth={"form": "gauss", "width": 60.0, "pathway": (20.0, 250.0)},
        n_edges=1500,
    )
    units = [_fake_unit(10020 + i, form="gauss") for i in range(8)]
    edges = (np.zeros(2, dtype=int), np.ones(2, dtype=int))
    rec = pb.build_record(p2, h, units, "PASS", "pooled", edges, "a" * 64, {"K": "gauss"}, _fb(9))
    d = (np.arange(p2.nb) + 0.5) * pb.BIN
    n, k = h.n_fold.sum((0, 1)), h.k_fold.sum((0, 1))
    th = pb.fit_global(d, n, k, "mix")
    delta = pb.fit_delta(d, n, k, th, "mix", rec["penalty"])
    amp, _ = pb._profile(n, k, pb.shape(d[None, :] * np.exp(-delta)[:, None], th, "mix"))
    for c, v in rec["pathway_amplitudes"].items():
        assert math.isclose(
            v, float(amp[int(c)]), rel_tol=1e-9
        )  # N1: profiled AFTER the final offsets
    cells = [int(c) for c in rec["pathway_widths"]]
    assert cells, "no pooled pathway widths: the consumer half would be vacuous"
    # B2: the consumer's kernel for that pathway is the record's own width, not the global one
    c = cells[0]
    f = wp.kernel_shape(rec, c)
    sg, lam = rec["pathway_widths"][c]
    want = (rec["a"] * math.exp(-(40.0**2) / (2 * sg**2)) + rec["c"] * math.exp(-40.0 / lam)) / (
        rec["a"] + rec["c"]
    )
    assert math.isclose(float(f(np.array([40.0]))[0]), want)


def test_pathway_check_measures_the_scored_estimator_the_record_exports(pop):
    """F2 (fold-lens round 2): B1's recovery check fits the SAME mixture P that decides and exports,
    not a separate Gaussian fit: its fitted widths are the record's own pathway widths."""
    rng = np.random.default_rng(1)
    p2 = pb.make_pop(pop.xyz, np.where(rng.random(N) < 0.5, 1, 2), np.ones(N, dtype=int))
    truth = {"form": "gauss", "width": 100.0, "pathway": (60.0, 160.0)}
    h, _ = pb.build_hists(p2, 10052, n_all=pb._all_pairs_n(p2), truth=truth, n_edges=1500)
    unit = {"pen": [4.0] * pb.K}
    chk = pb.pathway_check(p2, h, truth, unit)
    edges = (np.zeros(2, dtype=int), np.ones(2, dtype=int))
    units = [_fake_unit(10020 + i) for i in range(8)]
    rec = pb.build_record(p2, h, units, "PASS", "pooled", edges, "a" * 64, {"K": "mix"}, _fb(9))
    assert rec["penalty"] == 4.0 and chk["cells"]
    for c, w in zip(chk["cells"], chk["fitted"], strict=True):
        assert math.isclose(w, rec["pathway_widths"][c][0], rel_tol=1e-9), (c, w)


def test_zero_exposure_cell_takes_the_pooled_amplitude_in_both_arms():
    nb = 50
    d = (np.arange(nb) + 0.5) * pb.BIN
    ntr = np.zeros((pb.N_CELL, nb))
    ktr = np.zeros_like(ntr)
    ntr[1], ktr[1, :5] = 1000.0, 20.0  # one trained cell; cell 0 has no training exposure
    nte = np.zeros_like(ntr)
    kte = np.zeros_like(ntr)
    nte[0], kte[0, :3] = 200.0, 6.0
    th = np.log([20.0, 30.0, 0.1])
    p = pb.shaped_p(d, ntr, ktr, th, "gauss")
    g = pb.shape(d[None, :], th, "gauss")
    pooled = ktr.sum() / float((ntr * g).sum())
    assert np.allclose(p[0], pooled * g[0])  # not 0.5 / 1e-300
    flat_rate = ktr.sum() / ntr.sum()
    ll_f = pb.flat_ll(ntr, ktr, nte, kte)
    want = float((kte[0] * math.log(flat_rate) + (nte[0] - kte[0]) * math.log1p(-flat_rate)).sum())
    assert math.isclose(ll_f, want)
    ll_g = pb._ll(p, nte, kte)
    assert ll_g > -2000.0 and ll_f > -2000.0  # neither arm is flung to the probability clip


def test_inner_penalty_shape_never_sees_the_inner_held_out_edges(monkeypatch):
    n = np.full((pb.K, pb.K, pb.N_CELL, 8), 50, dtype=np.int64)
    k = np.zeros_like(n)
    for g in range(
        pb.K
    ):  # a distinct, huge edge count touching each fold, so any leak shows in k.sum()
        k[g, :, 1, 0] += 3**g
        k[:, g, 1, 0] += 3**g
    seen = []
    real = pb.fit_global

    def spy(
        d, nn, kk, kind="mix"
    ):  # no `init`: a warm start carried in from the outer fit would TypeError
        seen.append(float(kk.sum()))
        return real(d, nn, kk, kind)

    monkeypatch.setattr(pb, "fit_global", spy)
    f = 1
    pb.choose_penalty(np.arange(8) + 0.5, n, k, f, "gauss")
    expect = []
    for g in (h for h in range(pb.K) if h != f):
        _, ktr, _, _ = pb.inner_split(n, k, f, g)
        expect.append(float(ktr.sum()))
    assert seen == expect  # fitted once per inner TRAINING split, on that split's counts only


def test_output_guard_refuses_protected_inputs_aliases_and_existing_files(tmp_path):
    art = tmp_path / "a.npz"
    art.write_bytes(b"x")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(art)
    except OSError:  # symlinks need privileges on Windows: a hard link is the same alias
        import os

        os.link(art, alias)
    for bad in (
        art,  # an input
        alias,  # an alias of an input (resolved path / samefile)
        t3.LIVE_HOME / "out.json",  # inside the live lake
        tmp_path / "brain_lake.bin",
        tmp_path / "no_such_dir" / "o.json",
    ):
        with pytest.raises(t3.OutputRefused):
            t3.safe_output(bad, [art])
    fresh = t3.safe_output(tmp_path / "ok.json", [art])
    t3.write_exclusive(fresh, "{}")
    with pytest.raises(t3.OutputRefused):  # now it exists: never truncated
        t3.safe_output(fresh, [art])
    with pytest.raises(FileExistsError):  # a path that appeared after the check: O_EXCL
        t3.write_exclusive(fresh, "{}")


def test_cli_refuses_before_computing_when_out_is_an_input(tmp_path):
    art = tmp_path / "a.npz"
    art.write_bytes(b"not an artifact")  # never read: the refusal comes first
    with pytest.raises(t3.OutputRefused):
        pb.main(["planted", "--artifact", str(art), "--out", str(art)])
    assert art.read_bytes() == b"not an artifact"


def test_a_stale_planted_record_cannot_authorise_a_changed_instrument(pop, monkeypatch):
    plan = pb.PLAN
    rows = _fake_rows(plan, False)
    rec = {
        "population_sha256": pb.fingerprint(pop),
        "n_edges_target": 1200,
        "identity": pb.identity(),
        "plan": pb.plan_dicts(plan),
        "rows": rows,
        "bands": pb.calibrate_bands(rows[plan[0].name]["units"]),
    }
    assert pb.validate_planted(rec, pop, plan, 1200) == rec["bands"]
    monkeypatch.setattr(pb, "PENALTIES", (2.0, 8.0))  # a changed frozen setting
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(rec, pop, plan, 1200)
    monkeypatch.undo()
    cx, cy = t3.region_centre()
    monkeypatch.setattr(t3, "region_centre", lambda: (cx + 5.0, cy))  # a moved frame centre
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(rec, pop, plan, 1200)
    monkeypatch.undo()
    monkeypatch.setattr(pb, "_file_sha", lambda rel: "0" * 64)  # changed implementation
    with pytest.raises(pb.ExposureRefused):
        pb.validate_planted(rec, pop, plan, 1200)
    monkeypatch.undo()
    for edit in (lambda r: r.pop("identity"), lambda r: r["identity"]["impl_sha256"].popitem()):
        bad = json.loads(json.dumps(rec))
        edit(bad)
        with pytest.raises(pb.ExposureRefused):
            pb.validate_planted(bad, pop, plan, 1200)


def test_penalty_grid_is_computed_once_per_inner_split_and_picks_match_fit_delta(monkeypatch):
    rng = np.random.default_rng(3)
    n = np.full((pb.K, pb.K, pb.N_CELL, 12), 40, dtype=np.int64)
    k = rng.binomial(n, 0.05)
    calls = []
    real = pb.delta_table

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(pb, "delta_table", spy)
    pb.choose_penalty(np.arange(12) + 0.5, n, k, 0, "gauss")
    assert len(calls) == pb.K - 1  # one table per inner split, not len(PENALTIES) of them
    monkeypatch.undo()
    d, th = np.arange(12) + 0.5, np.log([4.0, 4.0, 1.0])
    nn, kk = n.sum((0, 1)), k.sum((0, 1))
    table = pb.delta_table(d, nn, kk, th, "gauss")
    for pen in pb.PENALTIES:
        assert np.array_equal(pb.pick_delta(table, pen), pb.fit_delta(d, nn, kk, th, "gauss", pen))


def test_build_record_takes_aggregates_only_and_matches_the_full_histogram(pop, n_all):
    h, _ = pb.build_hists(
        pop, 10020, n_all=n_all, truth={"form": "gauss", "width": 60.0}, n_edges=1200
    )
    units = [_fake_unit(10020 + i) for i in range(8)]
    edges = (np.zeros(2, dtype=int), np.ones(2, dtype=int))
    args = (units, "PASS", "global", edges, "a" * 64, {"K": "gauss"}, _fb(0.5))
    full = pb.build_record(pop, h, *args)
    agg = pb.build_record(pop, (h.n_fold.sum((0, 1)), h.k_fold.sum((0, 1))), *args)
    assert full == agg


def test_no_argument_self_check_runs_and_passes(capsys):
    assert pb.main([]) == 0
    assert "self-check OK" in capsys.readouterr().out
