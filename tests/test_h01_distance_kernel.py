"""T3 (G4 / T112): the H01 distance-kernel instrument -- every check of 2.11 with its constructed
red.

The pre-registration is `specs/022-unified-brain-arc/evidence/g4-wiring-prior.md` section 2 with its
2.14 amendments (A1-A14), frozen at 857163d4 before this file or the instrument existed. Every test
here runs on SYNTHETIC populations (a few hundred units in a small slab): no H01 file, no network,
no
Brain, no lake. The checks are 2.11 (a)-(k):

  (a) artifact pin: a one-byte change and a wrong n_edges are refused
  (b) units: positions left in nm make the planted width recovery fail -> the row fails (REFUSED)
  (c) layout: bins reshaped bin-major from group-major counts make the recovery fail (the pilot's
  incident)
  (d) split: a train set that touches the held-out fold is refused
  (e) determinism + stream independence (A9)
  (f) the width record reproduces on refit; a perturbed record is refused
  (g) the reading function: one constructed input per A3 row, plus the A2 selection cases
  (h) a planted-flat Gaussian fit is named DEGENERATE
  (i) the band boundary is strict; a degenerate set reads NOT MET (ruling 9)
  (j) the region-block x band table refuses a block with an empty band
  (k) the H01 entry point refuses before the planted record holds (A8)

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_h01_distance_kernel.py -q --no-cov
"""

import hashlib
import math

import pytest

np = pytest.importorskip("numpy")

from snn_data_lake.instruments import h01_distance_kernel as KD  # noqa: E402

N_SYN = 400
BOX_UM = (400.0, 300.0, 40.0)
EDGES_SYN = 2500
WIDTH = 30.0


def _population(seed=20261003, n=N_SYN, box=BOX_UM):
    rng = np.random.default_rng(seed)
    xyz = rng.random((n, 3)) * np.array(box)
    layer = rng.integers(0, 7, size=n)
    deg = rng.integers(0, 12, size=n)
    return KD.Population(xyz_um=xyz, layer=layer, out_deg=deg.copy(), in_deg=deg[::-1].copy())


def _plan():
    return (
        KD.PlanRow("flat_calibration", ("flat",), tuple(range(10008, 10016)), None, False),
        KD.PlanRow("flat_validation", ("flat",), tuple(range(10016, 10024)), "red_flat", False),
        KD.PlanRow("gauss", ("gauss", WIDTH), tuple(range(10024, 10032)), "gauss", True),
        KD.PlanRow("exp", ("exp", WIDTH), tuple(range(10032, 10040)), "exp", False),
    )


def _write_artifact(path, base, seed=20261004):
    """A synthetic artifact with the real file's keys: positions in nm, Gaussian-kernel edges."""
    rng = np.random.default_rng(seed)
    xyz = base.xyz_um
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    p = 0.3 * np.exp(-(d**2) / (2 * WIDTH**2))
    np.fill_diagonal(p, 0.0)
    pre, post = np.nonzero(rng.random(p.shape) < p)
    np.savez(
        path,
        xyz_nm=np.round(xyz * 1000.0).astype(np.int64),
        layer=base.layer.astype(np.int8),
        pre_idx=pre.astype(np.int32),
        post_idx=post.astype(np.int32),
    )
    return path, (hashlib.sha256(path.read_bytes()).hexdigest(), len(xyz), len(pre))


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return _write_artifact(tmp_path_factory.mktemp("h01syn") / "syn_c3_graph.npz", _population())


@pytest.fixture(scope="module")
def pop(artifact):
    path, pin = artifact
    return KD.load_population(path, pin=pin)


@pytest.fixture(scope="module")
def stage(pop):
    return KD.run_planted_stage(pop, plan=_plan(), n_edges=EDGES_SYN)


# ---- (a) the artifact pin ------------------------------------------------------------------------
def test_a_artifact_pin_refuses_a_changed_byte_and_a_wrong_edge_count(tmp_path, artifact):
    path, pin = artifact
    good = KD.load_population(path, pin=pin)
    assert len(good.xyz_um) == N_SYN and good.out_deg.sum() == pin[2] == good.in_deg.sum()
    with pytest.raises(KD.ArtifactRefused):
        KD.load_population(path, pin=(pin[0], pin[1], pin[2] + 1))
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    bad = tmp_path / "flipped.npz"
    bad.write_bytes(bytes(raw))
    with pytest.raises(KD.ArtifactRefused):
        KD.load_population(bad, pin=pin)


def test_a_the_real_pin_is_read_from_the_pin_file():
    sha, n_nodes, n_edges = KD.pinned_artifact()
    assert sha.startswith("a1d0491a") and sha.endswith("0549")
    assert (n_nodes, n_edges) == (KD.N_NODES, KD.N_EDGES) == (14215, 70001)


# ---- (b) the unit trap ---------------------------------------------------------------------------
def test_b_positions_are_converted_from_nm_and_nm_positions_are_refused(artifact, pop):
    path, _ = artifact
    with np.load(path) as z:
        assert np.array_equal(pop.xyz_um, z["xyz_nm"].astype(np.float64) / 1000.0)
    nm = KD.Population(
        xyz_um=pop.xyz_um * 1000.0, layer=pop.layer, out_deg=pop.out_deg, in_deg=pop.in_deg
    )
    with pytest.raises(KD.ArtifactRefused, match="units"):
        KD.Geometry.of(nm)
    good = KD.PlanRow("gauss", ("gauss", WIDTH), (10040,), None, True)
    assert KD.width_recovered(KD.planted_units(pop, good, EDGES_SYN)[0], good.truth)


# ---- (c) the pilot's bin-layout incident ---------------------------------------------------------
def test_c_bins_reshaped_bin_major_fail_the_planted_recovery(pop):
    row = KD.PlanRow("gauss", ("gauss", WIDTH), (10041,), None, True)
    geo = KD.Geometry.of(pop)
    rng_fold, rng_draw, _ = KD.seed_streams(10041)
    fold = KD.unit_folds(rng_fold, len(pop.xyz_um))
    hist = KD.pair_histograms(pop, geo, fold, truth=row.truth, n_edges=EDGES_SYN, rng=rng_draw)
    assert KD.width_recovered(KD.score_unit(hist, geo), row.truth)
    g = KD.N_GROUPS
    scrambled = hist._replace(
        n=hist.n.reshape(-1).reshape(geo.nb, g, g).transpose(1, 2, 0).copy(),
        k=hist.k.reshape(-1).reshape(geo.nb, g, g).transpose(1, 2, 0).copy(),
    )
    assert not KD.width_recovered(KD.score_unit(scrambled, geo), row.truth)


# ---- (d) the split -------------------------------------------------------------------------------
def test_d_a_train_set_touching_the_held_out_fold_is_refused():
    labels = np.arange(KD.N_GROUPS) // KD.N_BLOCKS
    KD.check_split(labels != 2, labels, 2)
    leaky = labels != 2
    leaky[np.flatnonzero(labels == 2)[0]] = True
    with pytest.raises(AssertionError):
        KD.check_split(leaky, labels, 2)


def test_d_train_plus_test_is_every_pair(pop):
    geo = KD.Geometry.of(pop)
    rng_fold, _, _ = KD.seed_streams(10042)
    hist = KD.pair_histograms(pop, geo, KD.unit_folds(rng_fold, N_SYN))
    for f in range(KD.K_FOLDS):
        ntr, _, nte, _ = KD.split(hist.n, hist.k, KD.FOLD_OF, f)
        assert ntr.sum() + nte.sum() == N_SYN * (N_SYN - 1)


# ---- (e) determinism and the RNG streams (A9) ----------------------------------------------------
def test_e_same_seed_same_histograms_and_fits_different_seed_different_folds(pop):
    row = KD.PlanRow("gauss", ("gauss", WIDTH), (10043,), None, True)
    a = KD.planted_units(pop, row, EDGES_SYN)[0]
    b = KD.planted_units(pop, row, EDGES_SYN)[0]
    assert a["hist_sha256"] == b["hist_sha256"] and a["params"] == b["params"]
    f1 = KD.unit_folds(KD.seed_streams(10043)[0], N_SYN)
    f2 = KD.unit_folds(KD.seed_streams(10044)[0], N_SYN)
    assert not np.array_equal(f1, f2)


def test_e_the_edge_draw_does_not_depend_on_the_fold_partition(pop):
    geo = KD.Geometry.of(pop)
    folds_a = KD.unit_folds(KD.seed_streams(10045)[0], N_SYN)
    folds_b = KD.unit_folds(KD.seed_streams(10046)[0], N_SYN)
    truth = ("gauss", WIDTH)
    ha = KD.pair_histograms(
        pop, geo, folds_a, truth=truth, n_edges=EDGES_SYN, rng=KD.seed_streams(10047)[1]
    )
    hb = KD.pair_histograms(
        pop, geo, folds_b, truth=truth, n_edges=EDGES_SYN, rng=KD.seed_streams(10047)[1]
    )
    assert np.array_equal(ha.k.sum((0, 1)), hb.k.sum((0, 1)))
    assert not np.array_equal(ha.k, hb.k)
    s = KD.seed_streams(10048)
    assert np.array_equal(
        KD.unit_folds(s[0], N_SYN), KD.unit_folds(KD.seed_streams(10048)[0], N_SYN)
    )


# ---- (f) the width record ------------------------------------------------------------------------
def test_f_width_record_reproduces_and_a_perturbed_record_is_refused(artifact, stage):
    path, pin = artifact
    fx = dict(pin=pin, plan=_plan(), n_edges=EDGES_SYN)
    rec = KD.final_width_record(path, "gauss", stage, **fx)
    KD.verify_width_record(rec, path, stage, **fx)
    assert abs(rec["width_um"]["sigma"] - WIDTH) / WIDTH < 0.15
    bad = dict(rec, params=[rec["params"][0], rec["params"][1] * (1 + 1e-4)])
    with pytest.raises(KD.RecordRefused):
        KD.verify_width_record(bad, path, stage, **fx)
    with pytest.raises(
        KD.RecordRefused
    ):  # cp2 #4: the physical width is verified, not only the params
        KD.verify_width_record(
            dict(rec, width_um={"sigma": rec["width_um"]["sigma"] * 1.01}), path, stage, **fx
        )
    with pytest.raises(KD.RecordRefused):
        KD.verify_width_record(dict(rec, width_um={}), path, stage, **fx)
    with pytest.raises(KD.ExposureRefused):
        KD.final_width_record(path, "gauss", None, **fx)
    with pytest.raises(
        KD.ExposureRefused
    ):  # cp2 #2: the verifier is an edge read, gated like the rest
        KD.verify_width_record(rec, path, None, **fx)


# ---- (g) the reading function (A2-A3) ------------------------------------------------------------
def _inputs(neuron=None, interior=None, region=None):
    """Constructed contrast inputs: 8 seed values per neuron/interior contrast, one value per
    region block.
    Unset contrasts sit at 0 (unresolved); bands are 1.0 everywhere."""
    jitter = np.array([0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 0.98, 1.02])
    n = {c: list(jitter * 0.0) for c in KD.CONTRAST_NAMES}
    z = {c: list(jitter * 0.0) for c in KD.CONTRAST_NAMES}
    r = [{c: 0.0 for c in KD.CONTRAST_NAMES} for _ in range(KD.N_BLOCKS)]
    for c, v in (neuron or {}).items():
        n[c] = list(jitter * v)
    for c, v in (interior or {}).items():
        z[c] = list(jitter * v)
    for c, v in (region or {}).items():
        for blk in r:
            blk[c] = v
    bands = {
        "neuron": {c: 1.0 for c in KD.CONTRAST_NAMES},
        "interior": {c: 1.0 for c in KD.CONTRAST_NAMES},
        "region": [{c: 1.0 for c in KD.CONTRAST_NAMES} for _ in range(KD.N_BLOCKS)],
    }
    return n, z, r, bands


GAUSS_WORLD = {"gauss-flat": 50.0, "exp-flat": 40.0, "exp-gauss": -10.0}
GAUSS_REGION = {"gauss-flat": 20.0, "exp-flat": 15.0, "exp-gauss": -5.0}


@pytest.mark.parametrize(
    "neuron,interior,region,kw,verdict,kernel",
    [
        (GAUSS_WORLD, GAUSS_WORLD, GAUSS_REGION, {"planted_ok": False}, "REFUSED", "gauss"),
        (
            GAUSS_WORLD,
            GAUSS_WORLD,
            GAUSS_REGION,
            {"region_valid": False},
            "NOT_READ_REGION_INVALID",
            "gauss",
        ),
        ({}, {}, {}, {}, "RED_FLAT", None),
        (
            GAUSS_WORLD,
            dict(GAUSS_WORLD, **{"exp-gauss": 10.0}),
            GAUSS_REGION,
            {},
            "NOT_READ_BOUNDARY",
            "gauss",
        ),
        (
            GAUSS_WORLD,
            GAUSS_WORLD,
            dict(GAUSS_REGION, **{"exp-gauss": 5.0}),
            {},
            "NOT_READ_DISAGREE",
            "gauss",
        ),
        (
            GAUSS_WORLD,
            GAUSS_WORLD,
            dict(GAUSS_REGION, **{"gauss-flat": 0.5}),
            {},
            "RED_FLAT_REGION",
            "gauss",
        ),
        (
            {"gauss-flat": 40.0, "exp-flat": 50.0, "exp-gauss": 10.0},
            {"gauss-flat": 40.0, "exp-flat": 50.0, "exp-gauss": 10.0},
            {"gauss-flat": 15.0, "exp-flat": 20.0, "exp-gauss": 5.0},
            {},
            "RED_EXP",
            "exp",
        ),
        (GAUSS_WORLD, GAUSS_WORLD, GAUSS_REGION, {}, "PASS_FAVOURS_GAUSSIAN", "gauss"),
    ],
)
def test_g_each_reading_row_is_reachable(neuron, interior, region, kw, verdict, kernel):
    n, z, r, bands = _inputs(neuron, interior, region)
    out = KD.read_t3(n, z, r, bands, **kw)
    assert out["verdict"] == verdict, out
    assert out["K"] == kernel


def test_g_exponential_only_over_flat_reads_red_exp():
    world = {"exp-flat": 30.0}
    n, z, r, bands = _inputs(world, world, {"exp-flat": 10.0})
    out = KD.read_t3(n, z, r, bands)
    assert out["C"] == ["exp"] and out["K"] == "exp" and out["verdict"] == "RED_EXP", out


def test_g_mixture_only_over_flat_reads_pass_mixture():
    world = {"mix-flat": 30.0}
    n, z, r, bands = _inputs(world, world, {"mix-flat": 10.0})
    out = KD.read_t3(n, z, r, bands)
    assert out["C"] == ["mix"] and out["K"] == "mix" and out["verdict"] == "PASS_MIXTURE", out


def test_g_tie_order_picks_the_gaussian_when_e_minus_g_is_unresolved():
    world = {"gauss-flat": 50.0, "exp-flat": 50.0}
    n, z, r, bands = _inputs(world, world, {"gauss-flat": 10.0, "exp-flat": 10.0})
    out = KD.read_t3(n, z, r, bands)
    assert (
        out["K"] == "gauss" and out["a"] == 0 and out["verdict"] == "PASS_GAUSSIAN_NOT_REFUSED"
    ), out


def test_g_a_raw_sign_disagreement_inside_the_bands_does_not_veto():
    n, z, r, bands = _inputs(GAUSS_WORLD, dict(GAUSS_WORLD, **{"exp-gauss": 0.5}), GAUSS_REGION)
    out = KD.read_t3(n, z, r, bands)
    assert out["states"]["interior"]["exp-gauss"] == 0
    assert out["verdict"] == "PASS_FAVOURS_GAUSSIAN", out


def test_g_mixture_upgrade_needs_both_instruments():
    world = dict(GAUSS_WORLD, **{"mix-flat": 60.0, "mix-gauss": 10.0})
    n, z, r, bands = _inputs(
        world, world, dict(GAUSS_REGION, **{"mix-flat": 25.0, "mix-gauss": 5.0})
    )
    assert KD.read_t3(n, z, r, bands)["K"] == "mix"
    n, z, r, bands = _inputs(
        world, world, dict(GAUSS_REGION, **{"mix-flat": 25.0, "mix-gauss": 0.5})
    )
    assert KD.read_t3(n, z, r, bands)["K"] == "gauss"


# ---- (h) a degenerate width is named -------------------------------------------------------------
def test_h_a_planted_flat_gaussian_fit_is_named_degenerate(stage):
    units = stage["rows"]["flat_validation"]["units"]
    assert any(u["degenerate"]["gauss"] for u in units), [u["width_all"] for u in units]


# ---- (i) the band boundary and ruling 9 ----------------------------------------------------------
def test_i_band_boundary_is_strict_and_a_degenerate_set_is_not_met():
    vals = [v * 5.0 for v in (0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 0.98, 1.02)]
    mean = math.fsum(vals) / len(vals)
    assert KD.seed_state(vals, 1.0) == 1
    assert KD.seed_state(vals, mean) == 0  # mean exactly at the band: NOT MET (strict)
    assert KD.seed_state([-v for v in vals], 1.0) == -1
    assert KD.seed_state([5.0] * 8, 1.0) == 0  # degenerate above the band: NOT MET (ruling 9)
    assert KD.value_state(1.0, 1.0) == 0 and KD.value_state(1.0001, 1.0) == 1


# ---- (j) the region instrument covers every band -------------------------------------------------
def test_j_a_block_with_an_empty_band_refuses_the_region_instrument(pop):
    geo = KD.Geometry.of(pop)
    assert geo.table.shape == (KD.N_BLOCKS, KD.N_BANDS) and geo.region_valid
    layer = pop.layer.copy()
    layer[(geo.blocks == 0) & (layer == 6)] = 5
    holed = KD.Population(xyz_um=pop.xyz_um, layer=layer, out_deg=pop.out_deg, in_deg=pop.in_deg)
    assert not KD.Geometry.of(holed).region_valid


# ---- (k) exposure order (A8) ---------------------------------------------------------------------
def test_k_h01_scoring_refuses_before_the_planted_record_holds(artifact, stage):
    path, pin = artifact
    fx = dict(pin=pin, plan=_plan(), n_edges=EDGES_SYN)
    with pytest.raises(KD.ExposureRefused):
        KD.score_h01(path, None, **fx)
    tampered = dict(
        stage,
        bands=dict(stage["bands"], neuron=dict(stage["bands"]["neuron"], **{"gauss-flat": 0.0})),
    )
    with pytest.raises(KD.ExposureRefused):
        KD.score_h01(path, tampered, **fx)
    with pytest.raises(KD.ExposureRefused):  # a fixture must name its plan and density
        KD.score_h01(path, stage, pin=pin)
    with pytest.raises(KD.ExposureRefused):  # the record's density is not the authorized one
        KD.score_h01(path, stage, pin=pin, plan=_plan(), n_edges=EDGES_SYN + 1)


def test_k_the_validator_recomputes_and_never_trusts_stored_verdicts(artifact, stage):
    """cp2 #2: a stored all_rows_hold / holds is never read; rows are re-read from their units."""
    path, pin = artifact
    fx = dict(pin=pin, plan=_plan(), n_edges=EDGES_SYN)
    rows = dict(stage["rows"])
    rows["gauss"] = dict(
        rows["gauss"],
        units=[
            dict(u, seed=s)
            for u, s in zip(rows["flat_validation"]["units"], _plan()[2].seeds, strict=True)
        ],
        holds=True,
    )
    lying = dict(stage, rows=rows, all_rows_hold=True)
    with pytest.raises(KD.ExposureRefused, match="does not hold"):
        KD.score_h01(path, lying, **fx)
    missing = dict(stage, rows={k: v for k, v in stage["rows"].items() if k != "exp"})
    with pytest.raises(KD.ExposureRefused):
        KD.score_h01(path, missing, **fx)
    reseeded = dict(stage["rows"])
    reseeded["exp"] = dict(
        reseeded["exp"],
        units=reseeded["exp"]["units"][:7] + [dict(reseeded["exp"]["units"][7], seed=99999)],
    )
    with pytest.raises(KD.ExposureRefused, match="seeds"):
        KD.score_h01(path, dict(stage, rows=reseeded), **fx)


def test_k_a_record_for_another_population_or_with_self_consistent_forged_bands_is_refused(
    artifact, stage
):
    """cp2 #2: the fingerprint binds the record to its population; bands + their hash are
    recomputed from the
    calibration units, so a forger who also rewrites bands_sha256 still fails."""
    path, pin = artifact
    fx = dict(pin=pin, plan=_plan(), n_edges=EDGES_SYN)
    with pytest.raises(KD.ExposureRefused, match="different population"):
        KD.score_h01(path, dict(stage, population_sha256="0" * 64), **fx)
    forged = dict(stage["bands"], neuron=dict(stage["bands"]["neuron"], **{"gauss-flat": 0.0}))
    with pytest.raises(KD.ExposureRefused, match="recompute"):
        KD.score_h01(path, dict(stage, bands=forged, bands_sha256=KD.bands_sha256(forged)), **fx)


def test_k_a_calibration_only_plan_authorizes_nothing(pop, artifact):
    """cp2 #2: all([]) over no checkable rows read True; a one-row plan must refuse."""
    path, pin = artifact
    cal_only = KD.run_planted_stage(pop, plan=_plan()[:1], n_edges=EDGES_SYN)
    assert cal_only["all_rows_hold"] is True  # the stored flag would have opened the gate
    with pytest.raises(KD.ExposureRefused):
        KD.score_h01(path, cal_only, pin=pin, plan=_plan()[:1], n_edges=EDGES_SYN)
    with pytest.raises(KD.ExposureRefused):
        KD.score_h01(path, cal_only, pin=pin, plan=_plan(), n_edges=EDGES_SYN)


def test_k_a_reduced_plan_never_authorizes_the_production_artifact(artifact, stage, monkeypatch):
    """cp2 #2: on the pin file's artifact only A6's registered plan (or its reserve) authorizes,
    whatever
    plan the caller names -- here the synthetic artifact is made 'production' by pinning its sha."""
    path, pin = artifact
    monkeypatch.setattr(KD, "pinned_artifact", lambda: pin)
    for call in (
        lambda: KD.score_h01(path, stage, pin=pin, plan=_plan(), n_edges=EDGES_SYN),
        lambda: KD.final_width_record(
            path, "gauss", stage, pin=pin, plan=_plan(), n_edges=EDGES_SYN
        ),
        lambda: KD.verify_width_record(
            {"model": "gauss"}, path, stage, pin=pin, plan=_plan(), n_edges=EDGES_SYN
        ),
    ):
        with pytest.raises(KD.ExposureRefused, match="registered plan"):
            call()


def test_k_a_region_invalid_population_reads_not_read_and_touches_no_edge_distance(
    tmp_path, monkeypatch
):
    base = _population()
    layer = base.layer.copy()
    layer[(KD.region_blocks(base.xyz_um) == 0) & (layer == 6)] = 5
    holed = KD.Population(xyz_um=base.xyz_um, layer=layer, out_deg=base.out_deg, in_deg=base.in_deg)
    path, pin = _write_artifact(tmp_path / "holed.npz", holed)
    pop = KD.load_population(path, pin=pin)
    rec = KD.run_planted_stage(pop, plan=_plan(), n_edges=EDGES_SYN)
    assert (
        rec["stopped"].startswith("NOT_READ_REGION_INVALID")
        and not rec["rows"]
        and not rec["all_rows_hold"]
    )

    def no_edges(*a, **kw):
        assert kw.get("edges") is None, "an edge distance was read"
        raise AssertionError("no pair pass may run on a stopped record")

    monkeypatch.setattr(KD, "pair_histograms", no_edges)
    out = KD.score_h01(path, rec, pin=pin, plan=_plan(), n_edges=EDGES_SYN)
    assert out["reading"]["verdict"] == "NOT_READ_REGION_INVALID" and out["reading"]["K"] is None


# ---- the planted stage end to end on the synthetic population ------------------------------------
def test_planted_stage_holds_and_bands_are_locked(stage):
    assert stage["region_valid"]
    assert stage["bands_sha256"] == KD.bands_sha256(stage["bands"])
    rows = stage["rows"]
    assert rows["flat_validation"]["reading"]["verdict"] == "RED_FLAT", rows["flat_validation"][
        "reading"
    ]
    assert rows["gauss"]["reading"]["verdict"] == "PASS_FAVOURS_GAUSSIAN", rows["gauss"]["reading"]
    assert rows["exp"]["reading"]["verdict"] == "RED_EXP", rows["exp"]["reading"]
    assert all(r["holds"] for name, r in rows.items() if name != "flat_calibration"), {
        k: (v["holds"], v["reading"]["verdict"]) for k, v in rows.items()
    }
    assert stage["all_rows_hold"]


def test_a_failed_planted_row_makes_the_stage_not_hold(pop):
    plan = _plan()[:2] + (
        KD.PlanRow(
            "gauss_expected_exp", ("gauss", WIDTH), tuple(range(10024, 10032)), "exp", False
        ),
    )
    out = KD.run_planted_stage(pop, plan=plan, n_edges=EDGES_SYN)
    assert not out["rows"]["gauss_expected_exp"]["holds"] and not out["all_rows_hold"]


def test_h01_scoring_on_a_synthetic_gaussian_artifact_reads_the_gaussian(artifact, stage):
    path, pin = artifact
    out = KD.score_h01(
        path, stage, seeds=tuple(range(10050, 10058)), pin=pin, plan=_plan(), n_edges=EDGES_SYN
    )
    assert out["reading"]["verdict"] == "PASS_FAVOURS_GAUSSIAN", out["reading"]
    assert out["region_identical_across_seeds"]


# ---- the optimizer (no scipy dependency; scipy only cross-checks when installed) -----------------
def test_nelder_mead_finds_a_quadratic_minimum():
    x, fx, _ = KD.nelder_mead(
        lambda v: float(((v - np.array([1.0, -2.0])) ** 2).sum()), np.array([3.0, 3.0])
    )
    assert np.allclose(x, [1.0, -2.0], atol=1e-5) and fx < 1e-9


def test_nelder_mead_matches_scipy_when_scipy_is_installed():
    opt = pytest.importorskip("scipy.optimize")

    def rosen(v):
        return float(100.0 * (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2)

    ours = KD.nelder_mead(rosen, np.array([-1.2, 1.0]))[0]
    ref = opt.minimize(
        rosen,
        [-1.2, 1.0],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000},
    ).x
    assert np.allclose(ours, ref, atol=1e-6), (ours, ref)


def test_frozen_plan_matches_the_pre_registration():
    names = [r.name for r in KD.PLANTED]
    assert names[:2] == ["flat_calibration", "flat_validation"] and len(KD.PLANTED) == 13
    seeds = [s for r in KD.PLANTED for s in r.seeds]
    assert len(seeds) == len(set(seeds)) == 104 and 10007 not in seeds
    assert KD.PLANTED[0].seeds == tuple(range(10140, 10148)) and KD.PLANTED[1].seeds == tuple(
        range(10270, 10278)
    )
    reserve = [s for r in KD.reserve_plan() for s in r.seeds]
    assert reserve == list(range(10600, 10704))
    assert KD.CONFIRMATORY_SEEDS == tuple(range(10100, 10108))


def test_cp2_15_degenerate_is_bounded_by_the_largest_pair_distance_not_the_box_diagonal():
    """Codex cp2 #15 / 2.3: extrema on different somata make the box diagonal exceed every pair
    distance."""
    xyz = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
    geo = KD.Geometry.of(
        KD.Population(
            xyz_um=xyz, layer=np.zeros(4, int), out_deg=np.ones(4, int), in_deg=np.ones(4, int)
        )
    )
    assert geo.dpair == pytest.approx(10 * math.sqrt(2)) and geo.dmax == pytest.approx(
        10 * math.sqrt(3)
    )


def test_cp2_14_the_mixture_starts_from_the_fitted_gaussian_and_exponential_unchanged(monkeypatch):
    """Codex cp2 #14 / 2.3: no unregistered amplitude adjustment of the start."""
    d = np.arange(1, 60) + 0.5
    n = np.full(d.shape, 1000.0)
    k = np.round(n * 0.2 * np.exp(-d / 20)).astype(float)
    real, seen = KD.fit_model, {}

    def spy(m, d, n, k, init=None):
        seen[m] = init if init is not None else seen.get(m)
        return real(m, d, n, k, init)

    monkeypatch.setattr(KD, "fit_model", spy)
    th = KD.fit_all(d, n, k)
    assert np.allclose(seen["mix"], np.r_[th["gauss"], th["exp"]])


@pytest.mark.parametrize("truth", [("gauss", 30.0), ("dc_exp", 20.0), ("mix", 30.0, 60.0, 0.5)])
def test_cp2_8_the_planted_scale_makes_the_exact_pairwise_clipped_expectation_hit_the_target(
    pop, truth
):
    """Codex cp2 #8 / A10: E = sum over every ordered pair of min(1, c w) = n_edges, checked
    against a
    brute-force full-matrix sum, in a regime where clipping bites (40,000 of 159,600 pairs'
    worth)."""
    geo = KD.Geometry.of(pop)
    target = 40_000
    c, expected, share = KD.solve_scale(pop, geo, truth, target)
    d = np.sqrt(((pop.xyz_um[:, None] - pop.xyz_um[None]) ** 2).sum(-1))
    w = KD.kernel_shape(truth, d)
    if truth[0].startswith("dc_"):
        w = w * (pop.out_deg[:, None] * pop.in_deg[None, :])
    np.fill_diagonal(w, 0.0)
    off = ~np.eye(len(d), dtype=bool)
    brute = float(np.minimum(c * w[off], 1.0).sum())
    assert abs(brute - target) <= 1e-6 * target and abs(expected - brute) <= 1e-6 * target
    assert share == pytest.approx(float((c * w[off] >= 1.0).sum()) / brute) and share > 0
    hist = KD.pair_histograms(
        pop,
        geo,
        KD.unit_folds(KD.seed_streams(10050)[0], N_SYN),
        truth=truth,
        n_edges=target,
        rng=KD.seed_streams(10050)[1],
    )
    assert hist.planted["expected_edges_drawn"] == pytest.approx(brute, rel=1e-9)


def test_cp2_6_the_statistic_refuses_an_unregistered_seed_allocation(artifact, stage, monkeypatch):
    """Codex cp2 #6 / 2.12: sixteen partitions, repeated seeds, or exposed seeds never reach the t
    statistic."""
    path, pin = artifact
    fx = dict(pin=pin, plan=_plan(), n_edges=EDGES_SYN)
    for bad in (
        tuple(range(10050, 10066)),
        (10050,) * 8,
        tuple(range(10050, 10057)),
        (10050.0,) + tuple(range(10051, 10058)),
    ):
        with pytest.raises(KD.ExposureRefused, match="seeds"):
            KD.score_h01(path, stage, seeds=bad, **fx)
    with pytest.raises(ValueError):
        KD.seed_state([1.0] * 16, 0.0)
    monkeypatch.setattr(
        KD, "pinned_artifact", lambda: pin
    )  # the synthetic file stands for production
    KD._check_seeds(True, KD.CONFIRMATORY_SEEDS)
    KD._check_seeds(True, KD.CONFIRMATORY_RESERVE)
    with pytest.raises(KD.ExposureRefused, match="registered"):
        KD._check_seeds(True, tuple(range(10050, 10058)))
    with pytest.raises(KD.ExposureRefused, match="seeds"):  # a bool is not a seed (G7 D2)
        KD._check_seeds(False, (True,) + tuple(range(10051, 10058)))


def test_cp2_9_split_membership_is_asserted_on_the_actual_train_and_test_indexing(pop):
    """Codex cp2 #9 / 2.11 (d): ntr/nte are compared with an independent per-group loop, so a
    mutant that
    lets a held-out group into the training index fails even though train + test still sums to
    every pair."""
    geo = KD.Geometry.of(pop)
    hist = KD.pair_histograms(pop, geo, KD.unit_folds(KD.seed_streams(10043)[0], N_SYN))
    n = hist.n
    for f in range(KD.K_FOLDS):
        ntr, ktr, nte, kte = KD.split(n, hist.k, KD.FOLD_OF, f)
        ref_tr = np.zeros_like(ntr)
        ref_te = np.zeros_like(nte)
        for gi in range(KD.N_GROUPS):
            for gj in range(KD.N_GROUPS):
                touches = KD.FOLD_OF[gi] == f or KD.FOLD_OF[gj] == f
                if touches:
                    ref_te += n[gi, gj]
                else:
                    ref_tr += n[gi, gj]
        assert (ntr == ref_tr).all() and (nte == ref_te).all()


def test_cp2_9_every_registered_planted_truth_runs_through_the_synthetic_pipeline(pop):
    """Codex cp2 #9 / 2.11 (g): all 13 A6 truths (mixture, degree-corrected, the scale ladder) are
    drawn at the
    fixture density, the exact expectation holds, and the degree-corrected rows record their
    clipped share."""
    assert len(KD.PLANTED) == 13
    geo = KD.Geometry.of(pop)
    for row in KD.PLANTED:
        row = row._replace(seeds=row.seeds[:1])
        (u,) = KD.planted_units(pop, row, EDGES_SYN, geo=geo)
        p = u["planted"]
        assert p["expected_edges_drawn"] == pytest.approx(EDGES_SYN, rel=1e-6), row.name
        assert abs(p["edges"] - EDGES_SYN) < 6 * math.sqrt(EDGES_SYN), row.name
        assert "clipped_share_of_edges" in p and np.isfinite(u["neuron"]["gauss-flat"]), row.name


def test_cp2_r4_a_stopped_record_with_a_foreign_plan_authorizes_nothing_on_production(
    tmp_path, monkeypatch
):
    """G7 R4: the NOT_READ branch reads no edge, but it must still describe the authorized plan."""
    base = _population()
    layer = base.layer.copy()
    layer[(KD.region_blocks(base.xyz_um) == 0) & (layer == 6)] = 5
    path, pin = _write_artifact(
        tmp_path / "holed.npz",
        KD.Population(xyz_um=base.xyz_um, layer=layer, out_deg=base.out_deg, in_deg=base.in_deg),
    )
    rec = KD.run_planted_stage(KD.load_population(path, pin=pin), plan=_plan(), n_edges=EDGES_SYN)
    monkeypatch.setattr(KD, "pinned_artifact", lambda: pin)
    with pytest.raises(KD.ExposureRefused, match="registered plan"):
        KD.score_h01(path, rec, pin=pin)


def test_cp2_h1_a_resaved_copy_under_a_caller_pin_is_still_production(
    artifact, stage, tmp_path, monkeypatch
):
    """G7 H1: production is recognised by content, so new bytes + a caller-supplied pin + a reduced
    plan or
    unregistered seeds authorize nothing."""
    path, pin = artifact
    z = np.load(path)
    copy = tmp_path / "resaved.npz"
    np.savez_compressed(copy, **{k: z[k] for k in z.files})
    cpin = (hashlib.sha256(copy.read_bytes()).hexdigest(), pin[1], pin[2])
    assert cpin[0] != pin[0]
    monkeypatch.setattr(
        KD, "pinned_artifact", lambda: pin
    )  # the 'real' H01: same counts, different sha
    for call in (
        lambda: KD.score_h01(copy, stage, pin=cpin, plan=_plan(), n_edges=EDGES_SYN),
        lambda: KD.final_width_record(
            copy, "gauss", stage, pin=cpin, plan=_plan(), n_edges=EDGES_SYN
        ),
    ):
        with pytest.raises(KD.ExposureRefused, match="registered plan"):
            call()
