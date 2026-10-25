"""E1b harness (instruments/rpe_charge): the reds the pre-registration owes, all brain-free.

One real-brain smoke is marked slow; everything else runs on planted series or constructed runs.
The E1 numbers in `E1_*` are E1's recorded per-seed values (REPORT-TABLES, N=600, H = ticks >= 300).
"""

import copy
import json
import math
import os
import random
import statistics
import types

import pytest

from snn_data_lake.instruments import rpe_charge as R

E1_SEEDS = range(8)
E1_R = {
    "real": [0.2265, -0.9935, 0.0634, 0.4067, 0.3041, 0.4737, 0.4013, 0.1665],
    "pfrozen": [0.2196, -0.9935, 0.2198, 0.4805, 0.4843, 0.5127, 0.4882, 0.1909],
    "pshuf": [0.2068, -0.9923, 0.2198, 0.4805, 0.3128, 0.6029, 0.3449, 0.2703],
}
E1_MAE = {
    "real": [0.4307, 0.5284, 0.6753, 0.5057, 0.4157, 0.4595, 0.4878, 0.3093],
    "vfrozen": [0.4515, 0.7914, 0.6843, 0.4294, 0.2222, 0.3435, 0.3009, 0.1496],
    "vshuf": [0.392, 1.0304, 0.8656, 0.5143, 0.6302, 0.4329, 0.5125, 0.4463],
}
E1_REPLAY = {  # E1 replay rows (e2v2.json "real", seeds 0-7); planted rows are recomputed
    "0": {"learn": 0.430711, "frozen": 0.390189, "shuf": 0.434431},
    "1": {"learn": 0.528399, "frozen": 1.063897, "shuf": 0.567258},
    "2": {"learn": 0.675305, "frozen": 0.637511, "shuf": 0.645877},
    "3": {"learn": 0.505729, "frozen": 0.497381, "shuf": 0.528494},
    "4": {"learn": 0.415703, "frozen": 0.356554, "shuf": 0.480637},
    "5": {"learn": 0.459531, "frozen": 0.503659, "shuf": 0.46967},
    "6": {"learn": 0.487768, "frozen": 0.416854, "shuf": 0.475799},
    "7": {"learn": 0.309349, "frozen": 0.255679, "shuf": 0.416695},
}


def _e1_runs(tier="no-encoder"):
    runs = {}
    for arm in ("real", "pfrozen", "pshuf", "vfrozen", "vshuf"):
        for s in E1_SEEDS:
            r = E1_R.get(arm, E1_R["real"])[s]
            m = E1_MAE.get(arm, E1_MAE["real"])[s]
            runs[(arm, s)] = R.synth_run(arm, s, tier, r=r, mae_value=m)
    return runs


def _pins(tmp_path, names, *, exist):
    env = {}
    d = tmp_path / ("present" if exist else "absent")
    d.mkdir(exist_ok=True)
    for k in names:
        p = d / k
        if exist:
            p.write_bytes(b"x")
        env[k] = str(p)
    return env


def _env(tmp_path, *, production, pc_exists=False):
    names = (*R.ENC, *R.PACKS)
    env = _pins(tmp_path, names, exist=production)
    pc = tmp_path / "pc_cortex.json"
    if pc_exists:
        pc.write_bytes(b"x")
    env[R.PCPIN] = str(pc)
    return env


# ---------------------------------------------------------------------------------------- tier pin
def test_tier_refusal_fires_on_a_mismatched_request(tmp_path, monkeypatch):
    # R1: production requested, no-encoder paths pinned -> refused before any brain exists
    assert "missing" in R.preflight("production", _env(tmp_path, production=False))
    # R2: no-encoder requested, production-style paths that exist -> refused
    assert "reachable" in R.preflight("no-encoder", _env(tmp_path, production=True))
    # the pc cortex pin must be missing in BOTH tiers
    assert "must be missing" in R.preflight(
        "no-encoder", _env(tmp_path, production=False, pc_exists=True)
    )
    (tmp_path / "pc_cortex.json").unlink()
    # an unpinned (or relative) variable is a refusal, not a cwd-relative default
    env = _env(tmp_path, production=False)
    env["SNN_SHD_PACK"] = "models/shd_pack.npz"
    assert "ABSOLUTE" in R.preflight("no-encoder", env)
    del env["SNN_SHD_PACK"]
    assert "ABSOLUTE" in R.preflight("no-encoder", env)
    # production with files that exist but whose sha256 prefix is not the pinned one
    prod = _env(tmp_path, production=True)
    assert "sha256" in R.preflight("production", prod)
    pref = {k: R._sha_prefix(prod[k]) for k in R.ENC}
    monkeypatch.setattr(R, "ENC", pref)
    assert R.preflight("production", prod) is None
    assert R.preflight("no-encoder", _env(tmp_path, production=False)) is None


def test_tier_verdict_is_not_satisfied_by_the_wrong_tier():
    stdlib = {"perception_tier": "stdlib", "perceiving_real": False, "perceiving_audio": False}
    prod = {"perception_tier": "eprop-snn", "perceiving_real": True, "perceiving_audio": True}
    assert not R.tier_verdict("production", stdlib)[0]
    assert not R.tier_verdict("no-encoder", prod)[0]
    assert R.tier_verdict("production", prod)[0] and R.tier_verdict("no-encoder", stdlib)[0]
    half = dict(prod, perceiving_audio=False)  # a tier label alone is not the tier
    assert not R.tier_verdict("production", half)[0]


def test_refusals_are_explicit_exits_not_asserts(tmp_path):
    with pytest.raises(R.Refused) as e:
        R.check_optimize(types.SimpleNamespace(optimize=1))
    assert e.value.code == 3
    out = tmp_path / "r.json"
    env = dict(_env(tmp_path, production=False), E1B_TREE_SHA="a" * 40, E1B_TREE_CLEAN="1")
    with pytest.raises(
        R.Refused
    ):  # production requested on no-encoder pins: no brain, no result file
        R.run_arm("real", 8, 1, "production", str(tmp_path / "home"), str(out), env=env)
    assert not out.exists()
    for bad in ({}, {"E1B_TREE_SHA": "abc", "E1B_TREE_CLEAN": "1"}, {"E1B_TREE_SHA": "a" * 40}):
        with pytest.raises(R.Refused):
            R.tree_pin(bad)
    assert R.tree_pin({"E1B_TREE_SHA": "a" * 40, "E1B_TREE_CLEAN": "1"}) == {
        "sha": "a" * 40,
        "clean": True,
    }


# ------------------------------------------------------------------------------- wrappers / patches
def _broken_dopamine(window, reward):
    raise RuntimeError("planted patch bug")


def test_a_broken_patch_is_not_a_perfect_neuter():
    from snn_data_lake.plasticity.valence_stdp import ValenceLearner

    vl = ValenceLearner()
    vl.dopamine = _broken_dopamine  # the neuter body raises on every call
    arm = types.SimpleNamespace(_valence=vl)
    with R.ArmPatch("vfrozen") as ap:
        ap.bind(arm)
        n_err = 0
        for t in range(5):
            ap.tick = t
            try:
                vl.learn([0.0] * 4, 1.0)
            except RuntimeError:
                n_err += 1  # the tick loop swallows exactly this and carries on
    assert n_err == 5 and len(ap.exc) == 5  # recorded, then re-raised
    rec = R.synth_run("vfrozen", 8, "production", r=0.1, mae_value=0.1)
    rec.update(swallowed_exceptions=len(ap.exc), errs=n_err, learn_calls=0)
    bad = R.run_problems("vfrozen", 8, rec, "production")
    assert any("swallowed" in b for b in bad) and any("learn_calls==0" in b for b in bad)


def test_patches_are_restored_and_scoped_to_the_arm_instance():
    from snn_data_lake.plasticity.valence_stdp import ValenceLearner, _pattern

    orig, win = ValenceLearner.learn, _pattern(1)
    mine, foreign = ValenceLearner(), ValenceLearner()
    with R.ArmPatch("vshuf", perm=[1.0, -1.0, 1.0]) as ap:
        ap.bind(types.SimpleNamespace(_valence=mine))
        mine.learn(win, 0.5)
        foreign.learn(win, 0.5)
        assert ap.v["calls"] == 1 and ap.v["foreign_learn"] == 1
        assert ap.v["r"] == [0.5] and ap.v["r_sent"] == [1.0]  # true reward kept, shuffled one sent
        assert any(foreign.w) and any(
            mine.w
        )  # vshuf still learns, from the shuffled reward; foreign is unpatched
    assert ValenceLearner.learn is orig


# --------------------------------------------------------------------------------------- controls
def test_planted_control_replay_passes_and_planted_null_does_not():
    rows = R.planted_rows("learn")
    ta, tb, ok = R.control_pass(rows)
    assert ok and ta > R.BAR and tb > R.BAR
    _, _, ok_null = R.control_pass(R.planted_rows("null"))
    assert not ok_null  # a stream where learning cannot help is not a control that passes


def test_planted_harm_goes_green_in_the_harm_direction():
    ph = R.planted_rows("harm")
    assert R.ttest([r["learn"] - r["frozen"] for r in ph.values()]) > R.BAR
    assert R.harm_control_pass(ph)[1]
    assert not R.harm_control_pass(R.planted_rows("null"))[1]


# ------------------------------------------------------------------------------- statistics / VOID
def test_status_is_void_not_not_met_on_a_failed_control():
    h = R.harm([0.1 + 0.01 * i for i in range(8)])
    assert R.status(h, R.DH) == "MET"
    assert R.status(h, R.DH, ok=False) == "VOID (control failed)"
    assert R.status(R.harm([0.1] * 5), R.DH).startswith("NO VERDICT")
    flat = R.harm([0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.3, -0.3])
    assert "INCONCLUSIVE" in R.status(flat, R.DH)  # mde above the claimed effect
    tight = R.harm([-0.001, 0.001] * 4)
    s_tight = R.status(tight, R.DH)
    assert "powered" in s_tight and "INCONCLUSIVE" not in s_tight


def test_mde_factor_matches_the_hand_computation():
    f = R.mde_exact_factor(8, R.BAR)
    assert f == pytest.approx(0.979, abs=0.01)  # frozen section 5 (A&S 26.7.10)
    assert R.MDE_F == 1.00 and R.MDE_F >= f


def test_compare_reproduces_e1_report_numbers():
    runs = _e1_runs()
    e2 = {
        "real": E1_REPLAY,
        "planted": R.planted_rows("learn"),
        "planted_harm": R.planted_rows("harm"),
    }
    a = R.analyse("no-encoder", runs, e2, seeds=E1_SEEDS, ep1_min=6, logs_pass=True)
    assert a["ev1"]["t"] == pytest.approx([-1.020, 1.982], abs=0.002)
    assert a["ev2"]["t"] == pytest.approx([0.536, 1.657], abs=0.002)
    assert a["ep2"]["t"] == pytest.approx([-2.818, -1.826], abs=0.002)
    assert a["hs"]["H-P1"]["t"] == pytest.approx(2.818, abs=0.002)  # the E1 "mirrored bar" signal
    assert a["hs"]["H-P1"]["k"] == 6 and a["hs"]["H-V3"]["k"] == 5
    assert a["pc_ok"] is True


@pytest.mark.skipif(not os.path.isdir(R.E1_RESULTS), reason="E1 lane results not on this machine")
def test_compare_reproduces_e1_numbers_from_the_recorded_json():
    runs = {}
    for arm in ("real", "pfrozen", "pshuf", "vfrozen", "vshuf"):
        for s in E1_SEEDS:
            with open(os.path.join(R.E1_RESULTS, f"{arm}_{s}.json"), encoding="utf-8") as f:
                runs[(arm, s)] = json.load(f)
    h0 = 300
    d = [R.mean_r(runs[("real", s)], h0) - R.mean_r(runs[("pfrozen", s)], h0) for s in E1_SEEDS]
    assert R.ttest(d) == pytest.approx(-2.818, abs=0.002)
    d1 = [R.mae(runs[("vfrozen", s)], h0) - R.mae(runs[("real", s)], h0) for s in E1_SEEDS]
    assert R.ttest(d1) == pytest.approx(-1.020, abs=0.002)


# --------------------------------------------------------------------------- fixtures and validity
def test_compare_fixtures_print_met_continue_and_void():
    lines = R.fixture_lines()
    assert [ln.split(" ")[1] for ln in lines] == ["harm", "noharm", "void"]
    assert (
        "H-P1 MET" in lines[0]
        and "H-P1 CONTINUE at look 1" in lines[1]
        and "NOT MET" not in lines[1]
    )
    assert "H-V1 VOID (control failed)" in lines[2]
    void = R.analyse("production", R.fixture_runs("void"), R.fixture_e2("void"), logs_pass=True)
    other = R.analyse(
        "no-encoder", R.fixture_runs("void", "no-encoder"), R.fixture_e2("void"), logs_pass=True
    )
    assert not R.null_precondition("ValenceLearner", void, other)[0]  # VOID feeds no NULL-RECORD


def test_validity_voids_a_tier_on_encoder_use_sha_and_missing_logs():
    runs = R.fixture_runs("harm", "production")
    ok = R.analyse("production", runs, R.fixture_e2(), logs_pass=True)
    assert ok["valid"], ok["bad"]
    zero = {
        k: dict(v, tier=dict(v["tier"], encoder_calls={"vision": 0, "audio": 0, "pc": 0}))
        for k, v in runs.items()
    }
    assert any(
        "zero encoder use" in b
        for b in R.analyse("production", zero, R.fixture_e2(), logs_pass=True)["bad"]
    )
    other = {
        k: dict(v, tier=dict(v["tier"], tree_sha="b" * 40)) if k[1] == 9 else v
        for k, v in runs.items()
    }
    assert any(
        "tree sha" in b
        for b in R.analyse("production", other, R.fixture_e2(), logs_pass=True)["bad"]
    )
    nolog = R.analyse("production", runs, R.fixture_e2(), logs_pass=False)
    assert not nolog["valid"] and nolog["hs"]["H-P1"]["status"] == "VOID (harness validity)"
    assert not nolog["hs"]["H-P1"]["met"]  # a VOID tier never reports a measured harm


def test_load_refuses_a_result_of_the_other_tier(tmp_path):
    run = R.fixture_runs("harm", "no-encoder")[("real", 8)]
    (tmp_path / "real_8.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(SystemExit, match="TIER MISMATCH"):
        R.load_run(str(tmp_path), "production", "real_8")
    assert R.load_run(str(tmp_path), "no-encoder", "real_8")["seed"] == 8
    assert R.load_run(str(tmp_path), "no-encoder", "real_9") is None


def test_logs_gate_wants_both_red_lines_and_the_three_fixture_lines(tmp_path):
    assert not R.logs_gate(str(tmp_path))
    red = "".join(
        f"RED tier={t} rc=3 result_file_exists=False\n" for t in ("production", "no-encoder")
    )
    (tmp_path / "tier_red.log").write_text(red, encoding="utf-8")
    (tmp_path / "fixture.log").write_text("\n".join(R.fixture_lines()) + "\n", encoding="utf-8")
    assert R.logs_gate(str(tmp_path))
    (tmp_path / "tier_red.log").write_text(red.splitlines()[0] + "\n", encoding="utf-8")
    assert not R.logs_gate(str(tmp_path))  # one tier's red only
    (tmp_path / "tier_red.log").write_text(red.splitlines()[0] * 2 + "\n", encoding="utf-8")
    assert not R.logs_gate(str(tmp_path))  # the same tier twice is not both tiers
    (tmp_path / "tier_red.log").write_text(red, encoding="utf-8")
    (tmp_path / "fixture.log").write_text("\n".join(R.fixture_lines()[:2]) + "\n", encoding="utf-8")
    assert not R.logs_gate(str(tmp_path))  # a short fixture log


def test_fixture_gate_checks_the_verdicts_not_just_the_log_against_itself(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(R, "BAR", 99.0)  # a broken compare: nothing can be MET
    lines = R.fixture_lines()
    assert not R.fixture_ok(lines)
    assert R.main(["fixture", str(tmp_path)]) == 1 and not (tmp_path / "fixture.log").exists()
    red = "".join(f"RED tier={t} rc=3 result_file_exists=False\n" for t in R.WANT)
    (tmp_path / "tier_red.log").write_text(red, encoding="utf-8")
    (tmp_path / "fixture.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not R.logs_gate(str(tmp_path))  # the log agrees with the broken code, the gate must not
    capsys.readouterr()


# ---------------------------------------------------------------------------------- cross-tier
@pytest.fixture
def both_tiers(monkeypatch):
    """Revision 2's both-tiers rule (SINGLE_TIER_HARM_DELETES = False; addendum 3 sets True)."""
    monkeypatch.setattr(R, "SINGLE_TIER_HARM_DELETES", False)


def _tier(kind, tier):
    runs = R.fixture_runs(kind, tier)
    return R.analyse(tier, runs, R.fixture_e2(tier=tier), logs_pass=True), runs


def _stopped(kind, tier):
    """`_tier` with a still-running primary replaced by a powered stop at its own cap (SD .005).

    Addendum 2 writes no null while the design continues; addendum 3 none when underpowered."""
    a, runs = _tier(kind, tier)
    for k, design in R.SEQ.items():
        if "seq" in a["hs"][k] and a["hs"][k]["seq"]["decision"] == "continue":
            q = R.seq_decision(_series(0.0, 0.005, design["looks"][-1]), **design)
            a["hs"][k]["seq"], a["hs"][k]["status"] = q, q["status"]
    return a, runs


def _same_real(pr, nr):
    """No-encoder runs with the production `real` series: tiers indistinguishable here."""
    return {k: dict(v, v=pr[k]["v"], series=pr[k]["series"]) for k, v in nr.items()}


def test_cross_tier_requires_agreement_and_a_tier_effect(both_tiers):
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    both = R.cross(pa, pr, na, nr)
    assert both["agree"] >= R.AGREE_MIN and both["tier_effect"]
    assert both["h_p1"].startswith("harm on both tiers") and ".0025" in both["h_p1"]
    # tiers whose same-seed series are identical are replicates: no cross-tier wording
    same = _same_real(pr, nr)
    flat = R.cross(pa, pr, na, same)
    assert not flat["tier_effect"] and "indistinguishable" in flat["h_p1"]


def test_both_tiers_met_but_disagreeing_in_sign_is_not_harm_on_both_tiers(both_tiers):
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    na = copy.deepcopy(na)
    d = na["hs"]["H-P1"]["d"]
    for i in range(3):  # three seeds flip sign; the tier's own bar is untouched (still MET)
        d[i] = -abs(d[i])
    out = R.cross(pa, pr, na, nr)
    assert out["agree"] == len(d) - 3 < R.AGREE_MIN
    assert "agrees on only" in out["h_p1"] and "tier finding" in out["h_p1"]


def test_one_tier_harm_is_a_tier_finding_and_a_missing_verdict_is_pending(both_tiers):
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("noharm", "no-encoder")
    one = R.cross(pa, pr, na, nr)
    assert "tier finding" in one["h_p1"]
    assert "candidate" not in one["h_p1"].replace("no deletion candidate", "")
    void = R.analyse("no-encoder", nr, R.fixture_e2(), logs_pass=False)
    assert R.cross(pa, pr, void, nr)["h_p1"].startswith("PENDING")


def test_null_record_precondition_excludes_a_measured_harm_and_a_void_tier():
    (pa, _), (na, _) = _stopped("noharm", "production"), _stopped("noharm", "no-encoder")
    ok, why = R.null_precondition("_plan_w", pa, na)
    assert ok, why  # both tiers null at adequate power, no harm, controls PASS
    (ha, _) = _tier("harm", "production")
    assert not R.null_precondition("_plan_w", ha, na)[
        0
    ]  # a measured harm gets the harm route, never a null record
    void = R.analyse(
        "no-encoder", R.fixture_runs("noharm", "no-encoder"), R.fixture_e2("void"), logs_pass=True
    )
    assert not R.null_precondition("ValenceLearner", pa, void)[
        0
    ]  # a failed control feeds no NULL-RECORD


# ----------------------------------------------------------------------------------- self-check
def test_selfcheck_main_runs_brain_free(capsys):
    assert R.main() == 0
    assert "rpe_charge self-check OK" in capsys.readouterr().out


@pytest.mark.slow
def test_real_brain_smoke_no_encoder(tmp_path, monkeypatch):
    names = (*R.ENC, *R.PACKS)
    for k in (*names, R.PCPIN, "NORD_HOME", "NORD_GPU", "SNDL_LIVE_HOME_GUARD"):
        monkeypatch.setenv(k, "")  # registered for restoration; run_arm sets the real values
    env = {k: str(tmp_path / "NO-ENCODER" / k) for k in (*names, R.PCPIN)}
    env.update(E1B_TREE_SHA="c" * 40, E1B_TREE_CLEAN="1")
    out = tmp_path / "real_8.json"
    rec = R.run_arm("real", 8, 3, "no-encoder", str(tmp_path / "home"), str(out), env=env)
    assert out.exists() and rec["tier"]["reported"] == "stdlib" and rec["errs"] == 0
    assert rec["tier"]["state_api"] == "stdlib" and rec["tier"]["encoder_calls"] == {
        "vision": 0,
        "audio": 0,
        "pc": 0,
    }
    assert math.isfinite(rec["run_s"])


# ------------------------------------------------------------- run_arm refusals (stub brain only)
def _prod_env(tmp_path, monkeypatch):
    env = _env(tmp_path, production=True)
    pref = {k: R._sha_prefix(env[k]) for k in R.ENC}
    monkeypatch.setattr(R, "ENC", pref)
    for k in ("NORD_HOME", "SNDL_LIVE_HOME_GUARD", "NORD_GPU", *env):
        monkeypatch.setenv(k, "")
    return dict(env, E1B_TREE_SHA="a" * 40, E1B_TREE_CLEAN="1")


def _stub_brain(monkeypatch, tier="stdlib", real=False, audio=False):
    brain = types.SimpleNamespace(
        _perception_tier=tier, _perceiving_real=real, _perceiving_audio=audio
    )
    monkeypatch.setattr("snn_data_lake.engine.brain.make_brain", lambda **kw: brain)


def test_run_arm_refuses_a_brain_that_reports_the_wrong_tier(tmp_path, monkeypatch):
    env = _prod_env(tmp_path, monkeypatch)
    _stub_brain(monkeypatch)  # pins pass, the brain fell back to stdlib
    out = tmp_path / "nested" / "r.json"
    with pytest.raises(R.Refused):
        R.run_arm("real", 8, 1, "production", str(tmp_path / "home"), str(out), env=env)
    assert not out.exists()
    assert out.parent.is_dir()  # the output directory exists before any tick runs


def test_run_arm_refuses_the_live_home_and_persistence_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a regression must not leave directories in the repo root
    env = _prod_env(tmp_path, monkeypatch)
    _stub_brain(monkeypatch, "eprop-snn", True, True)
    live = os.path.expanduser("~/.snn_data_lake")
    out = str(tmp_path / "r.json")
    for home in (live, os.path.join(live, "sub"), "\\\\?\\" + live):
        with pytest.raises(R.Refused):
            R.run_arm("real", 8, 1, "production", home, out, env=env)
    monkeypatch.setenv("NORD_PERSIST", "1")
    with pytest.raises(R.Refused):
        R.run_arm("real", 8, 1, "production", str(tmp_path / "home"), out, env=env)
    assert not os.path.exists(out)


def test_shuffle_arms_refuse_a_missing_or_foreign_series(tmp_path, monkeypatch):
    env = _prod_env(tmp_path, monkeypatch)
    _stub_brain(monkeypatch, "eprop-snn", True, True)
    home, out = str(tmp_path / "home"), str(tmp_path / "r.json")
    with pytest.raises(R.Refused):
        R.run_arm("vshuf", 8, 1, "production", home, out, env=env)  # no --series
    good = R.synth_run("real", 8, "production", sha="a" * 40)
    cases = {
        "wrong arm": dict(good, arm="vfrozen"),
        "wrong seed": dict(good, seed=9),
        "other tier": R.synth_run("real", 8, "no-encoder", sha="a" * 40),
        "fell back to stdlib": dict(good, tier=dict(good["tier"], reported="stdlib")),
        "other tree": R.synth_run("real", 8, "production", sha="b" * 40),
    }
    for name, series in cases.items():
        path = tmp_path / f"{name.replace(' ', '_')}.json"
        path.write_text(json.dumps(series), encoding="utf-8")
        with pytest.raises(R.Refused):
            R.run_arm("pshuf", 8, 1, "production", home, out, series=str(path), env=env)
    assert not os.path.exists(out)


# ------------------------------------------------------------------- patches on stand-ins
def test_pshuf_swaps_the_rpe_and_restores_it_even_when_the_call_raises(monkeypatch):
    from snn_data_lake.engine.autonomous_loop import AutonomousArm

    seen = []

    def fake(self, unit, pattern=None, qa=None):
        seen.append(self._rpe)
        if unit == "boom":
            raise RuntimeError("planted")
        return "ok"

    monkeypatch.setattr(AutonomousArm, "_pfc_bias", fake)
    arm = types.SimpleNamespace(_rpe=0.25, _plan_w=0.1, _plan_lr=0.02)
    with R.ArmPatch("pshuf", perm=[0.9, -0.9]) as ap:
        ap.bind(arm)
        assert AutonomousArm._pfc_bias(arm, "u") == "ok"
        with pytest.raises(RuntimeError):
            AutonomousArm._pfc_bias(arm, "boom")
        assert arm._rpe == 0.25 and seen == [0.9, -0.9]
        assert ap.p["rpe_entry"] == [0.25] and ap.p["rpe_used"] == [0.9] and ap.p["calls"] == 1
        assert len(ap.exc) == 1 and ap.exc[0]["where"] == "_pfc_bias"
        other = types.SimpleNamespace(_rpe=0.5)
        AutonomousArm._pfc_bias(other, "u")  # not the bound arm: passed through untouched
        assert seen[-1] == 0.5 and ap.p["calls"] == 1
    assert AutonomousArm._pfc_bias is fake


def test_pfrozen_zeroes_the_plan_rate_and_the_pc_stub_is_counted_and_restored():
    from snn_data_lake.faculties.prefrontal_executive import PrefrontalExecutive

    arm = types.SimpleNamespace(_plan_lr=0.02)
    with R.ArmPatch("pfrozen") as ap:
        ap.bind(arm)
    assert arm._plan_lr == 0.0
    orig = PrefrontalExecutive._decide_spiking
    with R.ArmPatch("pc") as ap:
        assert PrefrontalExecutive._decide_spiking(None, None) == (0, 1) and ap.pc_calls == 1
    assert PrefrontalExecutive._decide_spiking is orig


def test_real_arm_records_the_prediction_before_learning():
    from snn_data_lake.plasticity.valence_stdp import ValenceLearner, _pattern

    vl, win = ValenceLearner(), _pattern(1)
    with R.ArmPatch("real") as ap:
        ap.bind(types.SimpleNamespace(_valence=vl))
        for r in (1.0, 1.0, -1.0):
            vl.valence(win)
            vl.learn(win, r)
    v = ap.v
    assert v["calls"] == 3 and None not in v["pred"]
    assert all(
        abs(p - (r - rp)) < 1e-9 for p, r, rp in zip(v["pred"], v["r"], v["rpe"], strict=True)
    )
    assert (
        v["pred"][0] == 0.0 and v["pred"][1] > 0.0
    )  # the second prediction already saw the first learn


def test_valence_exceptions_are_recorded_then_reraised():
    from snn_data_lake.plasticity.valence_stdp import ValenceLearner

    vl = ValenceLearner()
    vl.n = None  # makes _active() fail inside valence()
    with R.ArmPatch("real") as ap:
        ap.bind(types.SimpleNamespace(_valence=vl))
        with pytest.raises(TypeError):
            vl.valence([0.5])
    assert [e["where"] for e in ap.exc] == ["valence"]


def test_install_counters_count_and_reraise():
    class Enc:
        def encode(self, x):
            return x

        def classify(self, x):
            raise ValueError("planted")

    exc = []
    brain = types.SimpleNamespace(_perception_enc=Enc(), _audio_enc=None, perceive_pc=lambda x: x)
    ec = R.install_counters(brain, exc)
    brain._perception_enc.encode(1)
    brain._perception_enc.encode(2)
    with pytest.raises(ValueError):
        brain._perception_enc.classify(3)
    brain.perceive_pc(4)
    assert ec == {"vision": 3, "audio": 0, "pc": 1} and exc[0]["where"] == "vision.classify"


# -------------------------------------------------------------------------- validity gates
def _mut(f):
    def go(run):
        f(run)
        return run

    return go


RUN_PROBLEMS = [
    ("swallowed", _mut(lambda r: r.update(swallowed_exceptions=1))),
    ("pattern series length", _mut(lambda r: r.update(pattern_len_ok=False))),
    ("tick errs", _mut(lambda r: r.update(errs=2))),
    ("learn_calls==0", _mut(lambda r: r.update(learn_calls=0))),
    ("pred is None", _mut(lambda r: r["v"]["pred"].__setitem__(3, None))),
    ("pred != r - rpe", _mut(lambda r: r["v"]["rpe"].__setitem__(3, 9.0))),
    ("persist on", _mut(lambda r: r.update(persist=True))),
    ("is not production", _mut(lambda r: r["tier"].update(requested="no-encoder"))),
    ("state() tier", _mut(lambda r: r["tier"].update(state_api="stdlib"))),
    ("perceiving flags", _mut(lambda r: r["tier"].update(perceiving_audio=False))),
    ("encoder sha256", _mut(lambda r: r["tier"].update(sha={}))),
    ("optimize=1", _mut(lambda r: r["tier"].update(optimize=1))),
    ("pc cortex active", _mut(lambda r: r["tier"].update(pc_cortex_tier="pc-cortex"))),
    ("zero encoder use", _mut(lambda r: r["tier"]["percept_ticks"].update(vision=0))),
    ("zero encoder use", _mut(lambda r: r["tier"]["percept_ticks"].update(audio=0))),
    ("ticks=300", _mut(lambda r: r.update(ticks=300))),
    ("pc cortex active", _mut(lambda r: r["tier"]["encoder_calls"].update(pc=1))),
]


@pytest.mark.parametrize(("needle", "mutate"), RUN_PROBLEMS, ids=[n for n, _ in RUN_PROBLEMS])
def test_run_problems_names_each_broken_invariant(needle, mutate):
    assert R.run_problems("real", 8, R.synth_run("real", 8, "production"), "production") == []
    bad = R.run_problems("real", 8, mutate(R.synth_run("real", 8, "production")), "production")
    assert any(needle in b for b in bad), bad


def test_run_problems_arm_specific_gates():
    bad = R.run_problems(
        "pfrozen",
        8,
        dict(R.synth_run("pfrozen", 8, "production"), plan_lr_final=0.02, pfc_bias_calls=0),
        "production",
    )
    assert any("plan_lr_final" in b for b in bad) and any("pfc_bias_calls==0" in b for b in bad)
    bad = R.run_problems(
        "pc",
        8,
        R.synth_run(
            "pc",
            8,
            "production",
        )
        | {"pc_stub_calls": 0},
        "production",
    )
    assert any("stub calls==0" in b for b in bad)
    ne = R.synth_run("real", 8, "no-encoder")
    ne["tier"]["encoder_calls"]["vision"] = 4
    assert any("encoder use" in b for b in R.run_problems("real", 8, ne, "no-encoder"))


def _analysis(kind="harm", tier="production", e2="ok", **kw):
    return R.analyse(tier, R.fixture_runs(kind, tier), R.fixture_e2(e2), logs_pass=True, **kw)


def test_analyse_voids_a_tier_on_each_section_4_check():
    runs = R.fixture_runs("harm")
    dirty = {k: dict(v, tier=dict(v["tier"], tree_clean=False)) for k, v in runs.items()}
    assert any(
        "tree_clean" in b
        for b in R.analyse("production", dirty, R.fixture_e2(), logs_pass=True)["bad"]
    )
    differ = dict(runs)
    differ[("real_r2", 8)] = R.synth_run("real", 8, "production", shift=1)
    assert any(
        "repeat arm differs" in b
        for b in R.analyse("production", differ, R.fixture_e2(), logs_pass=True)["bad"]
    )
    flat = dict(runs)
    flat[("pc", 8)] = R.synth_run("pc", 8, "production", shift=0)
    flat[("pc", 8)]["series"] = runs[("real", 8)]["series"]
    assert any(
        "pc seed 8 flips == 0" in b
        for b in R.analyse("production", flat, R.fixture_e2(), logs_pass=True)["bad"]
    )
    no_harm_ctl = dict(R.fixture_e2(), planted_harm=None)
    a = R.analyse("production", runs, no_harm_ctl, logs_pass=True)
    assert not a["valid"] and a["hs"]["H-V1"]["status"] == "VOID (control failed)"


def test_a_partial_campaign_evaluates_the_complete_endpoints_and_marks_the_rest_pending():
    runs = {k: v for k, v in R.fixture_runs("harm").items() if k[0] in ("real", "pfrozen")}
    a = R.analyse("production", runs, None, logs_pass=True)
    assert a["valid"] and a["pending"]
    # stage A alone is pending: no sequential decision before every validity check has run (re-panel
    # A-M2)
    assert a["hs"]["H-P1"]["status"].startswith("PENDING") and "seq" not in a["hs"]["H-P1"]
    assert a["hs"]["H-V1"]["status"].startswith("NO VERDICT") and a["hs"]["H-V3"][
        "status"
    ].startswith("NO VERDICT")
    assert "PENDING (partial campaign" in R.render(a)
    assert not R.null_precondition("_plan_w", a, a)[0]


def test_sensitivity_flags_a_seed_sensitive_primary():
    runs = R.fixture_runs("harm")
    for arm in R.ARMS:  # seed 9 is degenerate in every arm and carries the whole signal
        runs[(arm, 9)] = R.synth_run(
            arm, 9, "production", r=-0.99 + (0.0 if arm == "real" else 0.002), wiggle=0.01
        )
    a = R.analyse("production", runs, R.fixture_e2(), logs_pass=True)
    assert a["degenerate"] == [9] and set(a["sens"]) == {"H-P1", "H-V1"}
    assert a["sens"]["H-P1"]["n"] == 7


def test_render_prints_the_harm_block_and_the_tier():
    txt = R.render(_analysis())
    assert (
        "tier production" in txt
        and "H-P1" in txt
        and "ALL HOLD" in txt
        and "VOID" not in txt.split("## Dispositions")[0]
    )


# ------------------------------------------------------------------------------ cross / null
def test_cross_voids_the_pair_when_the_two_tiers_ran_on_different_trees(both_tiers):
    # Revision 2's both-tiers rule; under addendum 3 the other tree is reported beside
    # (addendum 6, C-M3)
    pa, pr = _tier("harm", "production")
    other = {
        k: dict(v, tier=dict(v["tier"], tree_sha="b" * 40))
        for k, v in R.fixture_runs("harm", "no-encoder").items()
    }
    na = R.analyse(
        "no-encoder", other, _bind(R.fixture_e2(tier="no-encoder"), other), logs_pass=True
    )
    assert na["valid"] and "PENDING" in R.cross(pa, pr, na, other)["h_p1"]


def test_h_v1_sign_agreement_is_its_own_not_h_p1s(both_tiers):
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    for a in (pa, na):
        a["hs"]["H-V1"]["met"] = True  # force H-V1 MET on both tiers
    na = copy.deepcopy(na)
    d = na["hs"]["H-V1"]["d"]
    na["hs"]["H-V1"]["d"] = [-abs(x) if i < 4 else abs(x) for i, x in enumerate(d)]
    pa["hs"]["H-V1"]["d"] = [abs(x) for x in pa["hs"]["H-V1"]["d"]]
    out = R.cross(pa, pr, na, nr)
    assert out["agree"] == 8 and out["agree_v1"] == 4  # H-P1 agrees on all, H-V1 on half
    assert "agrees on only" in out["h_v1"] and out["h_p1"].startswith("harm on both tiers")


def test_h_v1_both_tiers_names_the_in_brain_check_and_the_single_tier_flag(both_tiers, monkeypatch):
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    for a in (pa, na):
        a["hs"]["H-V1"]["met"] = True
        a["hs"]["H-V1"]["d"] = [abs(x) + 0.01 for x in a["hs"]["H-V1"]["d"]]
        a["in_brain"]["t"] = 0.0
    assert "harm-deletion candidate for the learning update" in R.cross(pa, pr, na, nr)["h_v1"]
    pa["in_brain"]["t"] = 9.0
    assert "objective finding" in R.cross(pa, pr, na, nr)["h_v1"]
    na["hs"]["H-V1"]["met"] = False
    monkeypatch.setattr(R, "SINGLE_TIER_HARM_DELETES", True)
    out = R.cross(pa, pr, na, nr)
    assert "harm on the production tier" in out["h_v1"] and "objective finding" in out["h_v1"]


def test_null_precondition_branches():
    pa, _ = _stopped("noharm", "production")
    na, _ = _stopped("noharm", "no-encoder")
    ok, why = R.null_precondition("ValenceLearner", pa, na)  # the positive path
    assert ok and why.startswith("production stopped powered"), why
    hn = copy.deepcopy(na)
    hn["hs"]["H-P1"]["met"] = True
    assert "tier finding" in R.null_precondition("_plan_w", pa, hn)[1]

    def with_(a, **ch):
        a = copy.deepcopy(a)
        for k, v in ch.items():
            a[k] = v
        return a

    assert "control failed" in R.null_precondition("_plan_w", with_(pa, ctl_ok=False), na)[1]
    harmed = copy.deepcopy(pa)
    harmed["hs"]["H-P1"]["met"] = True
    assert "MET" in R.null_precondition("_plan_w", harmed, na)[1]
    obj = copy.deepcopy(pa)
    obj["ep2"]["met"] = True
    assert "objective endpoint was met" in R.null_precondition("_plan_w", obj, na)[1]
    wide = copy.deepcopy(pa)
    wide["ep2"]["mde"] = 1.0
    assert "INCONCLUSIVE" in R.null_precondition("_plan_w", wide, na)[1]
    wide_v = copy.deepcopy(pa)
    wide_v["ev2"]["mde"] = 1.0
    assert "INCONCLUSIVE" in R.null_precondition("ValenceLearner", wide_v, na)[1]


# ------------------------------------------------------------------------ bridge / CLI / logs
def _write_runs(d, runs):
    d.mkdir(parents=True, exist_ok=True)
    for (arm, s), run in runs.items():
        name = f"real_{s}_r2" if arm == "real_r2" else f"{arm}_{s}"
        (d / f"{name}.json").write_text(json.dumps(run), encoding="utf-8")


def test_bridge_readout_tree_equivalence_and_tree_sha(tmp_path):
    new = {
        (a, s): R.synth_run(
            a,
            s,
            "no-encoder",
            r=0.1 * s + (0.05 + 0.001 * s if a == "pfrozen" else 0.0),
            sha="c" * 40,
        )
        for a in ("real", "pfrozen")
        for s in R.PSEEDS
    }
    old = {k: {"v": v["v"], "series": v["series"]} for k, v in new.items()}
    _write_runs(tmp_path / "bridge", new)
    _write_runs(tmp_path / "e1", old)
    br = R.bridge_readout(str(tmp_path / "bridge"), str(tmp_path / "e1"), tree_shas=["c" * 40])
    assert br["tree_equivalent"] and br["tree_ok"] and br["bridge_h_p1"]["n"] == 8
    assert (
        br["bridge_h_p1"]["t"] > R.BAR and br["bridge_h_p1"]["k"] == 8
    )  # pfrozen earns more: positive d
    assert "tree-equivalent" in R.render_cross(
        R.cross(*_tier("harm", "production"), *_tier("harm", "no-encoder")), br
    )
    moved = dict(old)
    moved[("pfrozen", 3)] = {
        "v": dict(old[("pfrozen", 3)]["v"], r=[9.0] * 600),
        "series": old[("pfrozen", 3)]["series"],
    }
    _write_runs(tmp_path / "e1", moved)
    br2 = R.bridge_readout(str(tmp_path / "bridge"), str(tmp_path / "e1"), tree_shas=["d" * 40])
    assert not br2["tree_equivalent"] and not br2["tree_ok"]
    assert "BRIDGE VOID" in R.render_cross(
        R.cross(*_tier("harm", "production"), *_tier("harm", "no-encoder")), br2
    )
    assert "BRIDGE not run" in R.render_cross(
        R.cross(*_tier("harm", "production"), *_tier("harm", "no-encoder"))
    )


def test_cli_replay_compare_cross_fixture_roundtrip(tmp_path, capsys):
    work = tmp_path / "work"
    for tier in ("production", "no-encoder"):
        _write_runs(work / "results" / tier, R.fixture_runs("harm", tier))
    logs = work / "logs"
    assert R.main(["fixture", str(logs)]) == 0
    (logs / "tier_red.log").write_text(
        "".join(
            f"RED tier={t} rc=3 result_file_exists=False\n" for t in ("production", "no-encoder")
        ),
        encoding="utf-8",
    )
    assert R.logs_gate(str(logs))
    for tier in ("production", "no-encoder"):
        d = str(work / "results" / tier)
        assert R.main(["replay", d, tier, "600"]) == 0
        e2 = json.load(open(os.path.join(d, "e2v2.json"), encoding="utf-8"))
        assert (
            set(e2) >= {"real", "planted", "planted_harm", "tier"}
            and e2["tier"]["requested"] == tier
        )
        assert R.main(["compare", d, tier, str(work / f"{tier}.md")]) == 0
        md = (work / f"{tier}.md").read_text(encoding="utf-8")
        assert (
            "ALL HOLD" in md and "PENDING" not in md
        )  # the loader found real_8_r2 and both pc runs
    assert (
        R.main(
            [
                "cross",
                str(work / "results" / "production"),
                str(work / "results" / "no-encoder"),
                str(work / "cross.md"),
            ]
        )
        == 0
    )
    assert "Cross-tier" in (work / "cross.md").read_text(encoding="utf-8")
    assert R.main(["nope"]) == 2
    capsys.readouterr()


def test_cli_compare_refuses_a_replay_record_of_the_other_tier(tmp_path):
    d = tmp_path / "work" / "results" / "production"
    _write_runs(d, R.fixture_runs("harm", "production"))
    (d / "e2v2.json").write_text(
        json.dumps(dict(R.fixture_e2(), tier={"requested": "no-encoder"})), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="TIER MISMATCH"):
        R.main(["compare", str(d), "production"])


# ------------------------------------------------------- out path, PENDING, bridge validity
def test_run_arm_refuses_an_out_path_that_is_a_link(tmp_path, monkeypatch):
    env = _prod_env(tmp_path, monkeypatch)
    _stub_brain(monkeypatch, "eprop-snn", True, True)
    target = tmp_path / "somewhere_else.bin"
    target.write_bytes(b"precious")
    out = tmp_path / "res" / "r.json"
    out.parent.mkdir()
    os.link(target, out)  # a hardlink: writing `out` would truncate `target`
    with pytest.raises(R.Refused):
        R.run_arm("real", 8, 1, "production", str(tmp_path / "home"), str(out), env=env)
    assert target.read_bytes() == b"precious"


def _drop(runs, pred):
    return {k: v for k, v in runs.items() if not pred(k)}


def test_cross_is_pending_when_a_tiers_endpoint_is_incomplete(both_tiers):
    # Revision 2's both-tiers rule; under addendum 3 a short no-encoder tier never holds
    # (addendum 6, C-M3)
    (pa, pr), _ = _tier("harm", "production"), _tier("harm", "no-encoder")
    runs = _drop(R.fixture_runs("harm", "no-encoder"), lambda k: k[0] == "pfrozen" and k[1] >= 11)
    na = R.analyse("no-encoder", runs, R.fixture_e2(tier="no-encoder"), logs_pass=True)
    assert na["valid"] and na["hs"]["H-P1"]["status"].startswith("NO VERDICT")
    out = R.cross(pa, pr, na, runs)
    assert out["h_p1"].startswith("PENDING") and "tier finding" not in out["h_p1"]
    short = _drop(R.fixture_runs("harm", "no-encoder"), lambda k: k[0] == "real" and k[1] == 15)
    nb = R.analyse("no-encoder", short, R.fixture_e2(tier="no-encoder"), logs_pass=True)
    assert R.cross(pa, pr, nb, short)["h_p1"].startswith(
        "PENDING"
    )  # never "indistinguishable" on missing data


def test_h_v1_candidate_needs_complete_in_brain_pairs():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    for a in (pa, na):
        a["hs"]["H-V1"]["met"] = True
        a["hs"]["H-V1"]["d"] = [abs(x) + 0.01 for x in a["hs"]["H-V1"]["d"]]
        a["in_brain"] = R.harm([])  # stage B (vfrozen) has not run
    out = R.cross(pa, pr, na, nr)
    assert (
        "in-brain check incomplete" in out["h_v1"] and "harm-deletion candidate" not in out["h_v1"]
    )


def test_e_endpoints_print_no_verdict_on_incomplete_seeds():
    runs = _drop(R.fixture_runs("harm"), lambda k: k[0] == "pshuf" and k[1] >= 14)
    a = R.analyse("production", runs, R.fixture_e2(), logs_pass=True)
    assert not a["ep2"]["full"] and not a["ep2"]["met"]
    assert a["disp"]["_plan_w"].startswith("NO VERDICT")
    txt = R.render(a)
    assert "NO VERDICT (incomplete seeds)" in txt and "5/8 -> p 0.363" in txt


def test_bridge_incomplete_and_invalid_runs_are_not_a_tree_statement(tmp_path):
    new = {
        (a, s): R.synth_run(
            a,
            s,
            "no-encoder",
            r=0.1 * s + (0.05 + 0.001 * s if a == "pfrozen" else 0.0),
            sha="c" * 40,
        )
        for a in ("real", "pfrozen")
        for s in R.PSEEDS
    }
    old = {k: {"v": v["v"], "series": v["series"]} for k, v in new.items()}
    _write_runs(tmp_path / "e1", old)
    cross_txt = R.cross(*_tier("harm", "production"), *_tier("harm", "no-encoder"))
    part = _drop(new, lambda k: k == ("pfrozen", 7))
    _write_runs(tmp_path / "partial", part)
    br = R.bridge_readout(str(tmp_path / "partial"), str(tmp_path / "e1"), tree_shas=["c" * 40])
    assert not br["complete"] and "BRIDGE incomplete" in R.render_cross(cross_txt, br)
    bad = copy.deepcopy(new)
    bad[("pfrozen", 3)]["tier"]["tree_sha"] = "d" * 40
    bad[("pfrozen", 2)]["errs"] = 5
    _write_runs(tmp_path / "bad", bad)
    br2 = R.bridge_readout(str(tmp_path / "bad"), str(tmp_path / "e1"), tree_shas=["c" * 40])
    assert br2["complete"] and not br2["tree_ok"]
    assert any("tree sha" in p for p in br2["problems"]) and any(
        "errs" in p or "tick" in p for p in br2["problems"]
    )
    assert "BRIDGE VOID" in R.render_cross(cross_txt, br2)


# ---------------------------------- replay window, E-V2 control, seed sensitivity, in-learn guard
def test_replay_scores_exactly_the_held_out_window():
    run = R.synth_run("real", 8, "production", r=1.0)
    run["v"]["pattern"] = [[0.0]] * 600
    row = R.replay({8: run}, 600)["real"]["8"]
    assert (
        row["n_scored"] == 300 and row["frozen"] == 1.0
    )  # ticks 300..599, the frozen predictor is 0 vs r = 1


@pytest.mark.skipif(not os.path.isdir(R.E1_RESULTS), reason="E1 lane results not on this machine")
def test_replay_reproduces_e1s_recorded_rows():
    runs = {}
    for s in E1_SEEDS:
        with open(os.path.join(R.E1_RESULTS, f"real_{s}.json"), encoding="utf-8") as f:
            runs[s] = json.load(f)
    rows = R.replay(runs, 600)["real"]
    for s in E1_SEEDS:
        for k in ("learn", "frozen", "shuf"):
            assert rows[str(s)][k] == pytest.approx(E1_REPLAY[str(s)][k], abs=1e-5)


def _e2_with(frozen_gain, planted):
    rows = {
        str(s): {
            "learn": 0.5,
            "frozen": 0.5 + frozen_gain + 0.01 * (s % 3),
            "shuf": 0.5 + frozen_gain + 0.005 * (s % 4),
        }
        for s in R.SEEDS
    }
    return {
        "real": rows,
        "planted": R.planted_rows(planted),
        "planted_harm": R.planted_rows("harm"),
    }


def test_e_v2_needs_its_planted_control_not_only_the_t_bars():
    runs = R.fixture_runs("harm")
    good = R.analyse("production", runs, _e2_with(0.1, "learn"), logs_pass=True)
    assert good["pc_ok"] and good["ev2"]["met"]
    bad = R.analyse("production", runs, _e2_with(0.1, "null"), logs_pass=True)
    assert bad["pc_ok"] is False and bad["ev2"]["met"] is False and bad["ctl_ok"] is False


def test_seed_sensitive_is_flagged_when_dropping_the_degenerate_seed_flips_the_primary():
    # A-M3: full t (~4.4) clears the fixed BAR but not look 1's boundary; without seed 9 t clears
    # both, so a
    # fixed-BAR check reads both sides as crossed and hides the flip
    runs = {}
    for s in R.SEEDS:
        d = -0.05 if s == 9 else 0.1 + 0.0002 * (s % 3)
        base = -0.99 if s == 9 else 0.2
        for arm in R.ARMS:
            r = base + (d if arm in ("pfrozen", "pshuf") else 0.0)
            runs[(arm, s)] = R.synth_run(
                arm, s, "production", r=r, mae_value=0.4, shift=1 if arm != "real" else 0
            )
    s0 = R.SEEDS[0]
    runs[("real_r2", s0)] = copy.deepcopy(runs[("real", s0)])
    for s in (s0, s0 + 1):
        runs[("pc", s)] = R.synth_run("pc", s, "production", r=0.2, shift=2)
    # the replay rows bound to these runs: a VOID tier takes no sequential decision to guard
    a = R.analyse("production", runs, _bind(R.fixture_e2(), runs), logs_pass=True)
    assert a["valid"], a["bad"]
    assert a["degenerate"] == [9]
    assert not a["hs"]["H-P1"]["met"]  # the degenerate seed hides the harm the other seven show
    q = a["hs"]["H-P1"]["seq"]
    assert R.BAR < a["hs"]["H-P1"]["t"] < q["bound_t"] < a["sens"]["H-P1"]["t"], (q, a["sens"])
    assert a["sens"]["H-P1"]["seed_sensitive"] is True
    assert "seed-sensitive" in R.render(a)
    assert (
        _analysis()["sens"]["H-P1"]["seed_sensitive"] is False
    )  # no degenerate seed: nothing to flag


def test_prediction_is_none_when_learn_runs_without_a_preceding_valence_read():
    from snn_data_lake.plasticity.valence_stdp import ValenceLearner, _pattern

    vl = ValenceLearner()
    with R.ArmPatch("real") as ap:
        ap.bind(types.SimpleNamespace(_valence=vl))
        vl.learn(_pattern(1), 1.0)  # learn()'s own internal valence read is not the prediction
    assert ap.v["pred"] == [None]
    bad = R.run_problems(
        "real",
        8,
        dict(
            R.synth_run("real", 8, "no-encoder"),
            v=dict(R.synth_run("real", 8, "no-encoder")["v"], pred=[None]),
        ),
        "no-encoder",
    )
    assert any("pred is None" in b for b in bad)


def test_e_v2_is_not_met_on_a_partial_replay_record():
    e2 = _e2_with(0.1, "learn")
    e2["real"] = {k: v for k, v in e2["real"].items() if int(k) < 13}  # 5 of 8 seeds replayed
    a = R.analyse("production", R.fixture_runs("harm"), e2, logs_pass=True)
    assert a["ev2"]["t"][0] > R.BAR and min(a["ev2"]["k"]) >= R.K_MIN  # the t bars alone would pass
    assert not a["ev2"]["full"] and not a["ev2"]["met"]
    assert a["disp"]["ValenceLearner"].startswith("NO VERDICT")


def test_cross_withholds_candidate_wording_while_validity_checks_are_pending():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    pa["pending"] = ["repeat arm missing", "pc seed 8 missing"]
    out = R.cross(pa, pr, na, nr)
    assert (
        "PENDING (validity checks not yet run" in out["h_p1"]
        and "candidate" not in out["h_p1"].split("PENDING")[0]
    )
    assert "harm on the production tier" not in out["h_p1"]


def test_null_precondition_needs_one_tree_across_both_tiers():
    pa, _ = _tier("noharm", "production")
    other = {
        k: dict(v, tier=dict(v["tier"], tree_sha="b" * 40))
        for k, v in R.fixture_runs("noharm", "no-encoder").items()
    }
    na = R.analyse(
        "no-encoder", other, _bind(R.fixture_e2(tier="no-encoder"), other), logs_pass=True
    )
    assert na["valid"] and "tree sha" in R.null_precondition("_plan_w", pa, na)[1]


def test_replay_rows_must_match_the_current_real_run():
    runs = R.fixture_runs("harm")
    stale = R.fixture_e2()
    stale["real"]["8"]["N"] = 400  # replayed with --ticks 400
    assert any(
        "replay row seed 8" in b
        for b in R.analyse("production", runs, stale, logs_pass=True)["bad"]
    )
    resized = R.fixture_e2()
    resized["real"]["9"]["n_calls"] = 123  # replay of an earlier, different real run
    assert any(
        "replay row seed 9" in b
        for b in R.analyse("production", runs, resized, logs_pass=True)["bad"]
    )


def test_run_arm_refuses_a_directory_as_the_out_path(tmp_path, monkeypatch):
    env = _prod_env(tmp_path, monkeypatch)
    _stub_brain(monkeypatch, "eprop-snn", True, True)
    out = tmp_path / "res"
    out.mkdir()
    with pytest.raises(R.Refused):
        R.run_arm("real", 8, 1, "production", str(tmp_path / "home"), str(out), env=env)


def test_render_reports_the_per_seed_values_and_the_exact_mde_factor():
    txt = R.render(_analysis())
    for needle in (
        "exact noncentral-t cross-check 0.978",
        "E-V1 per-seed",
        "E-V2 per-seed",
        "E-P2 per-seed",
        "r_vfrozen - r_real per seed",
    ):
        assert needle in txt


def test_cross_thresholds_are_pinned_at_their_boundaries(both_tiers):
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")

    def flipped(k):
        a = copy.deepcopy(na)
        d = a["hs"]["H-P1"]["d"]
        a["hs"]["H-P1"]["d"] = [-x if i < k else x for i, x in enumerate(d)]
        return a

    assert R.cross(pa, pr, flipped(2), nr)["h_p1"].startswith(
        "harm on both tiers"
    )  # 6/8 agree: passes
    assert "agrees on only 5/8" in R.cross(pa, pr, flipped(3), nr)["h_p1"]  # 5/8: fails

    def with_differing(k):
        same = {key: dict(v, v=pr[key]["v"], series=pr[key]["series"]) for key, v in nr.items()}
        for s in R.SEEDS[:k]:
            same[("real", s)] = nr[("real", s)]  # only k seeds keep their own (different) series
        return R.cross(pa, pr, na, same)

    assert not with_differing(5)["tier_effect"] and "indistinguishable" in with_differing(5)["h_p1"]
    assert with_differing(6)["tier_effect"]


def test_null_precondition_refuses_a_tier_with_pending_checks():
    (pa, _), (na, _) = _stopped("noharm", "production"), _stopped("noharm", "no-encoder")
    assert R.null_precondition("_plan_w", pa, na)[0]
    pending = copy.deepcopy(pa)
    pending["pending"] = ["repeat arm missing"]
    assert "no verdict" in R.null_precondition("_plan_w", pending, na)[1]


def test_run_arm_refuses_a_symlinked_out_path(tmp_path, monkeypatch):
    env = _prod_env(tmp_path, monkeypatch)
    _stub_brain(monkeypatch, "eprop-snn", True, True)
    target = tmp_path / "elsewhere.bin"
    target.write_bytes(b"precious")
    out = tmp_path / "res" / "r.json"
    out.parent.mkdir()
    try:
        os.symlink(target, out)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need extra privilege here")
    with pytest.raises(R.Refused):
        R.run_arm("real", 8, 1, "production", str(tmp_path / "home"), str(out), env=env)
    assert target.read_bytes() == b"precious"


def test_load_runs_maps_every_tag_to_its_own_seed(tmp_path):
    _write_runs(tmp_path, R.fixture_runs("harm", "production"))
    runs = R.load_runs(str(tmp_path), "production")
    assert runs[("pc", 8)]["seed"] == 8 and runs[("pc", 9)]["seed"] == 9
    assert ("real_r2", 8) in runs and len(runs) == 43


# ------------------------------------ addendum 2: the sequential design (P1), R-S1..R-S5
# Planted per-seed differences d_s ~ N(mu, SD) with E1's SDs of the paired differences
# (no-encoder tier, the degenerate seed kept, as Revision 2 rules): H-P1 0.0695, H-V1 0.2040.
# Each simulated campaign draws the full cap of seeds once and walks the looks on its
# prefixes, as the campaign would.
SD_P1, SD_V1 = 0.0695, 0.2040
N_SIM = 4000  # fast twin; the slow twin runs 40000
# The addendum's stated rates (addendum 4: e1b-fix-row/derive_v1.py `verify`, 40000 campaigns, RNG
# seeds 7070 / 8080,
# not the tests' 303 / 404). "normal": d_s ~ N(mu, SD); "shape": E1's measured d_s, centred,
# resampled, plus mu (R-S1b); "smooth*": the kernel bootstrap (addenda 7-9).
STATED = {
    ("H-V1", "shape", "harm_by_look2"): 0.0004,
    ("H-V1", "shape", "power"): 0.8164,
    ("H-V1", "shape", "futility_null"): 0.0,
    ("H-V1", "shape", "inconclusive_null"): 0.3649,
    # addendum 8 (H-V1 cap 152): e1b-a8-row/derive_a8.py `harness H-V1 S` (seeds 9494 / 9595)
    ("H-V1", "smooth", "harm_by_look2"): 0.0,
    ("H-V1", "smooth", "power"): 0.8145,
    ("H-V1", "smooth", "futility_null"): 0.0,
    ("H-V1", "smooth", "inconclusive_null"): 0.3715,
    # addendum 9 (H-P1 cap 176): e1b-a9-row/derive_a9.py `harness <shape> 176` (seeds 9696 / 9797);
    # "shape" = raw, "smooth" = S, "smooth_u" = U, "smooth_h05" = H05
    ("H-P1", "normal", "harm_by_look2"): 0.0,
    ("H-P1", "normal", "power"): 0.9926,
    ("H-P1", "normal", "futility_null"): 0.9018,
    ("H-P1", "normal", "inconclusive_null"): 0.0,
    ("H-P1", "shape", "harm_by_look2"): 0.0,
    ("H-P1", "shape", "power"): 0.7995,
    ("H-P1", "shape", "futility_null"): 0.9036,
    ("H-P1", "shape", "inconclusive_null"): 0.0,
    ("H-P1", "smooth", "harm_by_look2"): 0.0,
    ("H-P1", "smooth", "power"): 0.9518,
    ("H-P1", "smooth", "futility_null"): 0.9014,
    ("H-P1", "smooth", "inconclusive_null"): 0.0,
    ("H-P1", "smooth_u", "harm_by_look2"): 0.0,
    ("H-P1", "smooth_u", "power"): 0.8163,
    ("H-P1", "smooth_u", "futility_null"): 0.9100,
    ("H-P1", "smooth_u", "inconclusive_null"): 0.0,
    ("H-P1", "smooth_h05", "harm_by_look2"): 0.0,
    ("H-P1", "smooth_h05", "power"): 0.8028,
    ("H-P1", "smooth_h05", "futility_null"): 0.9030,
    ("H-P1", "smooth_h05", "inconclusive_null"): 0.0,
}
# the planted shapes each primary's cap is powered on (P1 item 2): H-P1 the smoothed S, U and H05
# (addendum 9), H-V1 the raw resample (addendum 4) and S (addendum 8); H-P1's raw shape is a
# disclosed reference
CAP_SHAPE = {"H-P1": ("smooth", "smooth_u", "smooth_h05"), "H-V1": ("shape", "smooth")}
# addendum 8's variants S, U, H05 (addendum 9's cap set): (factor on h, variance-corrected)
SMOOTH = {"smooth": (1.0, True), "smooth_u": (1.0, False), "smooth_h05": (0.5, True)}
# E1's measured per-seed d_s (addendum 2's table, no-encoder tier, seed 1 kept): the planted SHAPE
# of R-S1b
E1_D = {
    "H-P1": [-0.0069, 0.0000, 0.1564, 0.0738, 0.1802, 0.0390, 0.0869, 0.0244],
    "H-V1": [0.0405, -0.5355, 0.0378, 0.0083, 0.0591, -0.0441, 0.0709, 0.0537],
}
CLAIMED = {"H-P1": R.DH, "H-V1": R.DM}
SD_OF = {"H-P1": SD_P1, "H-V1": SD_V1}


def _se(p, n=N_SIM):
    return math.sqrt(p * (1 - p) / n)


def _campaigns(mu, sd, n_sim, seed, key="H-P1"):
    rng = random.Random(seed)
    for _ in range(n_sim):
        yield [rng.gauss(mu, sd) for _ in range(R.SEQ[key]["looks"][-1])]


def _centred(key):
    """E1's measured d_s, centred: no harm in the mean (R-S1b's null)."""
    m = sum(E1_D[key]) / len(E1_D[key])
    return [x - m for x in E1_D[key]]


def _shape_campaigns(key, mu, n_sim, seed):
    """R-S1b: resample E1's centred d_s, shifted by mu."""
    pool = [x + mu for x in _centred(key)]
    rng = random.Random(seed)
    for _ in range(n_sim):
        yield [rng.choice(pool) + rng.gauss(0, 1e-9) for _ in range(R.SEQ[key]["looks"][-1])]


def _bandwidth(key):
    """Silverman's rule of thumb 0.9 min(SD, IQR / 1.34) n^(-1/5) on E1's d_s (addendum 7)."""
    q1, _, q3 = statistics.quantiles(E1_D[key], n=4, method="inclusive")
    return 0.9 * min(statistics.stdev(E1_D[key]), (q3 - q1) / 1.34) * len(E1_D[key]) ** -0.2


def _smooth_campaigns(key, mu, n_sim, seed, dist):
    """R-S1b smoothed (addendum 7): a Gaussian kernel bootstrap of E1's centred d_s, shifted by mu;
    `dist` keys `SMOOTH`."""
    pool = _centred(key)
    f, shrunk = SMOOTH[dist]
    h = f * _bandwidth(key)
    c = math.sqrt(1 + h * h / statistics.pvariance(pool)) if shrunk else 1.0
    rng = random.Random(seed)
    for _ in range(n_sim):
        yield [
            (rng.choice(pool) + rng.gauss(0, h)) / c + mu for _ in range(R.SEQ[key]["looks"][-1])
        ]


def _planted(key, dist, mu, n_sim, seed):
    if dist == "normal":
        return _campaigns(mu, SD_OF[key], n_sim, seed, key)
    if dist in SMOOTH:
        return _smooth_campaigns(key, mu, n_sim, seed, dist)
    return _shape_campaigns(key, mu, n_sim, seed)


def _walk(ds, key="H-P1", *, futility=True):
    """The pre-registered stop rule of `key` on one campaign: (decision, look, powered) at the
    first stop."""
    looks = R.SEQ[key]["looks"]
    for j, n in enumerate(looks, 1):
        dec = R.seq_decision(ds[:n], **R.SEQ[key])
        if dec["decision"] == "harm" or (futility and dec["decision"] in ("futility", "cap")):
            return dec["decision"], j, dec["powered"]
    return "cap", len(looks), dec["powered"]


def _naive(ds, key="H-P1"):
    """Optional stopping the instrument must see: a fixed one-sided .05 t test at every look."""
    from snn_data_lake.instruments.pool_selection import t_quantile

    for n in R.SEQ[key]["looks"]:
        h = R.harm(ds[:n])
        if h["t"] > t_quantile(0.95, n - 1) and h["k"] >= math.ceil(R.K_MIN * n / R.N_SEEDS):
            return True
    return False


def _type_one(n_sim, seed, key="H-P1"):
    camps = list(_campaigns(0.0, SD_OF[key], n_sim, seed, key))
    seq = sum(_walk(ds, key, futility=False)[0] == "harm" for ds in camps) / n_sim
    naive = sum(_naive(ds, key) for ds in camps) / n_sim
    return seq, naive


def _type_one_shape(n_sim, seed, key, dist):
    camps = _planted(key, dist, 0.0, n_sim, seed)
    return sum(_walk(ds, key, futility=False)[0] == "harm" for ds in camps) / n_sim


def test_rs1_rs2_sequential_type_one_holds_and_naive_optional_stopping_is_seen():
    seq, naive = _type_one(N_SIM, 101)
    # R-S1: futility is non-binding, so the type-I rate is taken with it ignored (conservative)
    assert seq <= 0.05 + 3 * _se(0.05), seq
    # R-S2: the same null series tested naively at every look exceed .05 by > 3 SE (R-S1 sees)
    assert naive > 0.05 + 3 * _se(0.05), naive


def test_rs1_h_v1_type_one_holds_under_the_normal_null_too():
    assert _type_one(N_SIM, 111, "H-V1")[0] <= 0.05 + 3 * _se(0.05)


# R-S1b planted shapes (legend above STATED)
SHAPES = [
    ("H-P1", "shape"),
    ("H-P1", "smooth"),
    ("H-P1", "smooth_u"),
    ("H-P1", "smooth_h05"),
    ("H-V1", "shape"),
    ("H-V1", "smooth"),
]


@pytest.mark.parametrize("key,dist", SHAPES)
def test_rs1b_type_one_holds_under_e1s_measured_per_seed_shape(key, dist):
    # R-S1b (re-panel A-B2): 4000 campaigns, each drawing E1's centred d_s to the endpoint's cap,
    # raw ("shape") or kernel-smoothed ("smooth*", addenda 7-9). Under the raw shape the plain t
    # boundaries of addendum 2 gave H-V1 a type I of 0.2185 (rv-arc1-a r2/typeI_shape.out).
    p = _type_one_shape(N_SIM, 505, key, dist)
    assert p <= 0.05 + 3 * _se(0.05), (key, dist, p)


def test_the_smoothed_shapes_bandwidth_is_the_preregistered_one():
    # addendum 7: Silverman's h on the 8 values, as derive_smooth.py `bw` (numpy) printed it
    assert round(_bandwidth("H-P1"), 4) == 0.0381 and round(_bandwidth("H-V1"), 4) == 0.0265


@pytest.mark.slow
def test_rs1_rs2_slow_twin_40000_campaigns():
    seq, naive = _type_one(40000, 202)
    assert seq <= 0.05 + 3 * _se(0.05, 40000) and naive > 0.05 + 3 * _se(0.05, 40000), (seq, naive)
    for key, dist in SHAPES:
        p = _type_one_shape(40000, 606, key, dist)
        assert p <= 0.05 + 3 * _se(0.05, 40000), (key, dist, p)


PLANTED = [("H-P1", "normal"), *SHAPES]


@pytest.mark.parametrize("key,dist", PLANTED)
def test_rs3_a_planted_harm_of_the_claimed_size_crosses_at_the_stated_rate(key, dist):
    camps = _planted(key, dist, CLAIMED[key], N_SIM, 303)
    walks = [_walk(ds, key) for ds in camps]
    by2 = sum(d == "harm" and j <= 2 for d, j, _ in walks) / N_SIM
    power = sum(d == "harm" for d, _, _ in walks) / N_SIM
    for got, what in ((by2, "harm_by_look2"), (power, "power")):
        want = STATED[(key, dist, what)]
        tol = 3 * (_se(want) if want else _se(0.01))  # _se(0) is 0: a zero pin borrows R-S4's floor
        assert abs(got - want) <= tol, (key, what, got, want)
    if dist == "normal" or dist in CAP_SHAPE[key]:  # P1 item 2: power at the cap >= .8
        assert power >= 0.8 - 3 * _se(0.8), (key, power)


@pytest.mark.parametrize("key,dist", PLANTED)
def test_rs4_a_planted_zero_effect_stops_and_reads_inconclusive_at_the_stated_rates(key, dist):
    camps = _planted(key, dist, 0.0, N_SIM, 404)
    walks = [_walk(ds, key) for ds in camps]
    fut = sum(d == "futility" for d, _, _ in walks) / N_SIM
    inc = sum(d != "harm" and not pw for d, _, pw in walks) / N_SIM  # a non-harm stop, underpowered
    for got, what in ((fut, "futility_null"), (inc, "inconclusive_null")):
        want = STATED[(key, dist, what)]
        assert abs(got - want) <= 3 * max(_se(want), _se(0.01)), (key, what, got, want)


def test_rs5_a_decision_at_a_non_look_seed_count_is_refused():
    for n in (7, 12, 20, 33):
        with pytest.raises(R.Refused):
            R.seq_decision([0.01] * n, R.DH)
    assert R.seq_decision([0.01] * 16, R.DH)["look"] == 2


def test_boundaries_spend_alpha_by_obrien_fleming_and_tighten_toward_the_cap():
    b = R.seq_boundaries()
    assert [x["n"] for x in b] == list(R.LOOKS)
    z = [x["z"] for x in b]
    assert (
        z == sorted(z, reverse=True) and z[-1] > 1.6449
    )  # each look stricter than a fixed .05 test
    assert all(x["t"] > x["z"] for x in b)  # t boundary at df n-1 sits above its z twin


def _extend(runs, e2, seeds, kind="noharm", tier="production"):
    """A later look's seeds: `real` + `pfrozen` (up to H-P1's cap) and a replay row each."""
    p1_end = R.SEEDS[0] + R.P1_CAP
    wig = 0.01 if tier == "production" else 0.0  # as fixture_runs: the tiers' `real` series differ
    for s in seeds:
        base, dh = (
            0.2 + 0.01 * (s % 5),
            (0.1 + 0.001 * ((s * 7) % 5 - 2)) if kind == "harm" else 0.0,  # as fixture_runs
        )
        arms = [("real", base), ("pfrozen", base + dh)] if s < p1_end else [("real", base)]
        for arm, r in arms:
            runs[(arm, s)] = R.synth_run(
                arm, s, tier, r=r, mae_value=0.4, wiggle=wig, shift=arm != "real"
            )
        e2["real"][str(s)] = dict(e2["real"]["8"])
    return runs, e2


def _look2_runs(kind, tier="production"):
    """Look 2: block 1 is the full campaign; seeds 16-23 add `real` + `pfrozen` + replay rows."""
    runs, e2 = _extend(
        R.fixture_runs(kind, tier), R.fixture_e2(tier=tier), range(16, 24), kind, tier
    )
    return runs, _bind(e2, runs)


def _stop_h_v1_at_look_1(e2):
    """H-V1 crosses at look 1: a planted learning harm on the look-1 block's replay rows."""
    for s in R.SEEDS:
        row = e2["real"][str(s)]
        row["frozen"] = row["learn"] - 0.1 - 0.0001 * (s % 3)


def test_compare_at_look_2_decides_the_primaries_sequentially_and_pends_nothing_else():
    runs, e2 = _look2_runs("harm")
    a = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    assert a["valid"] and a["pending"] == [], a["pending"]
    sq = a["hs"]["H-P1"]["seq"]
    # the look-1 prefix already crossed: that first stop is the decision of record (addendum 2)
    assert sq["look"] == 1 and sq["decision"] == "harm" and a["hs"]["H-P1"]["met"]
    assert f"look 1 of {len(R.P1_LOOKS)}" in R.render(a)
    assert a["hs"]["H-V2"]["status"].startswith(
        "NO VERDICT"
    )  # secondaries are decided at look 1 only
    with pytest.raises(R.Refused):
        R.analyse("production", runs, e2, seeds=range(8, 20), logs_pass=True)


def test_a_continuing_design_writes_no_null_record_and_says_continue():
    (pa, _), (na, _) = _tier("noharm", "production"), _tier("noharm", "no-encoder")
    assert pa["hs"]["H-P1"]["seq"]["decision"] == "continue"
    assert "CONTINUE" in pa["hs"]["H-P1"]["status"] and not pa["hs"]["H-P1"]["met"]
    ok, why = R.null_precondition("_plan_w", pa, na)
    assert not ok and "sequential" in why


def test_cross_tier_agreement_scales_with_the_look(both_tiers):
    # H-P1 continues at look 1 (no harm), so both tiers read it over 16 seeds at look 2 (A-M4)
    runs, e2 = _look2_runs("noharm")
    pa = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    assert pa["n_owed"]["H-P1"] == 16, pa["n_owed"]
    pa["hs"]["H-P1"]["met"], pa["hs"]["H-P1"]["d"] = True, [0.1] * 16
    na = copy.deepcopy(pa)  # the other tier: MET too, but d_s flips sign on seeds 16-23
    na["hs"]["H-P1"]["d"] = [x if i < 8 else -x for i, x in enumerate(pa["hs"]["H-P1"]["d"])]
    nruns = copy.deepcopy(runs)
    for k in [k for k in nruns if k[0] == "real"]:
        nruns[k]["series"]["action"] = [x + 1 for x in nruns[k]["series"]["action"]]
    out = R.cross(pa, runs, na, nruns)
    assert out["tier_effect"] and out["agree"] == 8
    # 8 of 16 would pass the look-1 count (6); at look 2 the rule is 12 of 16
    assert "both tiers MET but d_s sign agrees on only 8/16" in out["h_p1"], out["h_p1"]


# ------------- addendum 3: owner ruling #4616 "production tier decides" + futility
def test_ruling_production_met_alone_is_the_deletion_candidate():
    assert R.SINGLE_TIER_HARM_DELETES is True
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("noharm", "no-encoder")
    out = R.cross(pa, pr, na, nr)["h_p1"]
    assert "harm-deletion candidate" in out and "tier finding" not in out, out


def test_ruling_a_no_encoder_only_crossing_is_a_tier_finding():
    (pa, pr), (na, nr) = _tier("noharm", "production"), _tier("harm", "no-encoder")
    out = R.cross(pa, pr, na, nr)["h_p1"]
    assert "tier finding" in out and "harm on the production" not in out, out
    assert "candidate" not in out.replace("no deletion candidate", ""), out


def test_ruling_a_void_production_tier_is_refused_never_a_candidate():
    (_, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    void = R.analyse("production", pr, R.fixture_e2(), logs_pass=False)
    out = R.cross(void, pr, na, nr)
    for key in ("h_p1", "h_v1"):
        assert out[key].startswith("REFUSED") and "candidate" not in out[key], out[key]


def test_ruling_sign_disagreement_is_reported_beside_not_a_gate():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    na = copy.deepcopy(na)
    d = na["hs"]["H-P1"]["d"]
    na["hs"]["H-P1"]["d"] = [-abs(x) if i < 3 else x for i, x in enumerate(d)]
    out = R.cross(pa, pr, na, nr)["h_p1"]
    assert "harm-deletion candidate" in out and "agrees on 5/8" in out, out


def _series(mean, sd_, n=8):
    """n per-seed differences (n a multiple of 8) with exactly this mean and sample SD."""
    z = [-1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5] * (n // 8)
    return [mean + sd_ * x / R.sd(z) for x in z]


def _futility_at_look_1(sd_, m):
    """Eight per-seed differences with mean -m * sd_ and sample SD sd_: a look-1 futility stop."""
    return _series(-m * sd_, sd_)


def test_futility_at_look_1_with_sd_above_the_bound_is_inconclusive_not_null():
    from snn_data_lake.instruments.pool_selection import t_quantile

    ds = _futility_at_look_1(0.0695, 2)  # E1's H-P1 SD; conditional power ~.003
    q = R.seq_decision(
        ds, R.DH, sd_pre=0.01
    )  # a pre-registration whose SD opened look 1 to futility
    assert q["decision"] == "futility" and q["n"] == 8
    want = (R.seq_boundaries()[0]["t"] + t_quantile(0.8, 7)) * R.sd(ds) / math.sqrt(8)
    assert abs(q["mde"] - want) < 1e-9 and not q["powered"]  # the mde at the STOPPED seed count
    assert "INCONCLUSIVE when underpowered" in q["status"] and "NULL-RECORD path" not in q["status"]
    pa, _ = _stopped("noharm", "production")
    na, _ = _stopped("noharm", "no-encoder")
    pa["hs"]["H-P1"]["seq"] = q
    ok, why = R.null_precondition("_plan_w", pa, na)
    assert not ok and "INCONCLUSIVE" in why, why


def test_a_powered_futility_stop_takes_the_null_record_path_and_its_10x_retest():
    # a small SD turns the claimed effect into a large drift: only a far negative mean stops early
    q = R.seq_decision(_futility_at_look_1(0.005, 25), R.DH, sd_pre=0.005)
    assert q["decision"] == "futility" and q["powered"], q
    assert "NULL-RECORD" in q["status"] and "10x" in q["status"]


# ------------- decision of record and the NULL-RECORD gate
def test_seq_record_walks_the_looks_to_the_first_stop():
    look1 = _series(0.03, 0.07)  # t 1.21 < 7.20, conditional power ~.9: CONTINUE
    assert R.seq_decision(look1, R.DH)["decision"] == "continue"
    q = R.seq_record(look1 + [0.1] * 8, R.DH)
    assert q["look"] == 2 and q["decision"] == "harm", q
    # a look-1 futility stop stays the decision of record even when the next block would cross
    q = R.seq_record(_futility_at_look_1(0.0695, 2) + [0.5] * 8, R.DH, sd_pre=0.01)
    assert q["look"] == 1 and q["decision"] == "futility", q
    assert R.seq_record(look1, R.DH)["decision"] == "continue"


def test_the_decision_of_record_names_its_look_and_seeds():
    runs, e2 = _look2_runs("harm")
    a = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    assert a["hs"]["H-P1"]["status"] == "MET at look 1 (8 seeds)"
    assert "decision of record" in R.render(a)
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("noharm", "no-encoder")
    assert "MET at look 1 (8 seeds)" in R.cross(pa, pr, na, nr)["h_p1"]


def test_the_no_encoder_report_says_the_production_tier_decides():
    na, _ = _tier("noharm", "no-encoder")
    assert "the production tier decides" in R.render(na)
    pa, _ = _tier("noharm", "production")
    assert "the production tier decides" not in R.render(pa)


def test_null_precondition_needs_a_stopped_powered_production_primary():
    pa, _ = _stopped("noharm", "production")
    na, _ = _stopped("noharm", "no-encoder")
    no_seq = copy.deepcopy(pa)
    del no_seq["hs"]["H-V1"]["seq"]
    ok, why = R.null_precondition("ValenceLearner", no_seq, na)
    assert not ok and "no sequential stop" in why, why
    harmed = copy.deepcopy(pa)
    harmed["hs"]["H-P1"]["met"] = True
    harmed["hs"]["H-P1"]["seq"] = dict(harmed["hs"]["H-P1"]["seq"], decision="harm", powered=False)
    assert "harm route" in R.null_precondition("_plan_w", harmed, na)[1]


def test_a_no_encoder_continue_does_not_block_the_production_null():
    pa, _ = _stopped("noharm", "production")
    na, _ = _tier("noharm", "no-encoder")
    assert na["hs"]["H-P1"]["seq"]["decision"] == "continue"
    ok, why = R.null_precondition("_plan_w", pa, na)
    assert ok, why


def test_a_no_encoder_only_crossing_says_production_is_still_running():
    (pa, pr), (na, nr) = _tier("noharm", "production"), _tier("harm", "no-encoder")
    assert "CONTINUE" in R.cross(pa, pr, na, nr)["h_p1"]


def test_render_cross_reports_agreement_and_tier_effect_as_no_gate():
    txt = R.render_cross(R.cross(*_tier("harm", "production"), *_tier("harm", "no-encoder")))
    assert "required >=" not in txt and "no gate" in txt, txt


def test_ruling_a_void_no_encoder_tier_is_reported_beside_never_a_veto():
    # re-panel B-N4: under #4616 the production tier decides; a VOID no-encoder tier has no verdict
    # to give
    (pa, pr), (_, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    void = R.analyse("no-encoder", nr, R.fixture_e2(tier="no-encoder"), logs_pass=False)
    out = R.cross(pa, pr, void, nr)["h_p1"]
    assert not out.startswith("PENDING") and "harm-deletion candidate" in out, out
    assert "no-encoder VOID: reported beside, no verdict" in out, out


def test_ruling_indistinguishable_tiers_do_not_gate_the_production_candidate():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    same = _same_real(pr, nr)
    out = R.cross(pa, pr, na, same)
    assert not out["tier_effect"], out
    assert "harm-deletion candidate" in out["h_p1"] and "indistinguishable" in out["h_p1"]


def test_ruling_neither_tier_met_is_neither_candidate_nor_tier_finding():
    out = R.cross(*_stopped("noharm", "production"), *_stopped("noharm", "no-encoder"))["h_p1"]
    assert out.startswith("not met on the production tier") and "tier finding" not in out, out
    assert "candidate" not in out, out


def test_ruling_pending_checks_hold_the_candidate_even_without_a_tier_effect():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    same = _same_real(pr, nr)
    pa["pending"] = ["repeat arm missing"]
    out = R.cross(pa, pr, na, same)["h_p1"]
    assert (
        "PENDING (validity checks not yet run" in out and "candidate" not in out.split("PENDING")[0]
    )


def test_render_cross_scales_the_both_tiers_thresholds_with_the_look(both_tiers):
    runs, e2 = _look2_runs("harm")
    pa = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    txt = R.render_cross(R.cross(pa, runs, copy.deepcopy(pa), runs))
    assert "sign agreement 16/16 (required >= 12)" in txt, txt
    assert "seeds; required >= 12)" in txt, txt


def test_the_mde_is_taken_at_the_stopped_look_and_at_the_cap():
    from snn_data_lake.instruments.pool_selection import t_quantile

    bs = R.seq_boundaries()
    q = R.seq_decision(_series(-0.2, 0.0695, 16), R.DH, sd_pre=0.01)  # look 2, far below zero
    assert q["decision"] == "futility" and q["look"] == 2, q
    assert abs(q["mde"] - (bs[1]["t"] + t_quantile(0.8, 15)) * 0.0695 / 4) < 1e-9
    q = R.seq_decision(_series(0.0, 0.0695, 32), R.DH)  # E1's SD reaches the cap without crossing
    assert q["decision"] == "cap" and q["powered"], q
    assert abs(q["mde"] - (bs[3]["t"] + t_quantile(0.8, 31)) * 0.0695 / math.sqrt(32)) < 1e-9
    assert "NOT MET at the cap" in q["status"] and "NULL-RECORD path" in q["status"]


def test_seq_record_refuses_a_non_look_seed_count():
    for n in (7, 12, 20, 33):
        with pytest.raises(R.Refused):
            R.seq_record([0.01] * n, R.DH)


def test_the_in_brain_check_reads_the_look_1_block_at_later_looks():
    runs, e2 = _look2_runs("harm")
    pa = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    pa["hs"]["H-V1"]["met"], pa["in_brain"]["t"] = (
        True,
        0.0,
    )  # vfrozen pairs exist for seeds 8-15 only
    assert pa["in_brain"]["n"] == R.N_SEEDS
    na = copy.deepcopy(pa)
    na["hs"]["H-V1"]["met"] = False
    nruns = copy.deepcopy(runs)
    for k in [k for k in nruns if k[0] == "real"]:
        nruns[k]["series"]["action"] = [x + 1 for x in nruns[k]["series"]["action"]]
    out = R.cross(pa, runs, na, nruns)["h_v1"]
    assert "in-brain check holds" in out, out


def test_the_candidate_line_reprints_the_no_encoder_decision_of_record():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("noharm", "no-encoder")
    out = R.cross(pa, pr, na, nr)["h_p1"]
    assert f"no-encoder CONTINUE at look 1 of {len(R.P1_LOOKS)}, n=8 (no verdict)" in out, out


# ------------- e1b-fix (arc 2): re-panel round 2 A-B1, A-M2, C R2-M1, B N2 / N4, addendum 4 (owner
# ruling)
def _bind(e2, runs):
    """Stamp each replay row with the binding of the `real` run it stands for (what `replay`
    writes)."""
    for k, row in e2["real"].items():
        if ("real", int(k)) in runs:
            row.update(R.replay_binding(runs[("real", int(k))]))
    return e2


def _probe_campaign(tier, n_seeds):
    """rv-arc1-a r2/look2_null_probe.py: block 1 real beats pfrozen and pshuf by 0.1 (E-P2 MET);
    later blocks add real + pfrozen only, pfrozen below real (no harm)."""
    wig, runs = (0.01 if tier == "production" else 0.0), {}
    for s in range(8, 8 + n_seeds):
        base, dh = 0.3 + 0.01 * (s % 5), -0.1 - 0.004 * ((s * 7) % 5 - 2)
        for arm in R.ARMS if s < 16 else ("real", "pfrozen"):
            r = base + (dh if arm in ("pfrozen", "pshuf") else 0.0)
            runs[(arm, s)] = R.synth_run(
                arm, s, tier, r=r, mae_value=0.4 + 0.02 * (s % 3), wiggle=wig, shift=arm != "real"
            )
    runs[("real_r2", 8)] = copy.deepcopy(runs[("real", 8)])
    for s in (8, 9):
        runs[("pc", s)] = R.synth_run("pc", s, tier, r=0.2, mae_value=0.4, wiggle=wig, shift=2)
    e2 = R.fixture_e2(tier=tier)
    for s in range(16, 8 + n_seeds):
        e2["real"][str(s)] = dict(e2["real"]["8"])
    return R.analyse(tier, runs, _bind(e2, runs), seeds=range(8, 8 + n_seeds), logs_pass=True)


def test_a_look_1_met_objective_still_blocks_the_null_record_at_look_2():
    # re-panel A-B1: at look 2 E-P2 was recomputed over 16 seeds with no pshuf past look 1 (met ->
    # False), and the
    # _plan_w NULL-RECORD opened on data whose objective was met at t = 47
    for n in (8, 16):
        pa, na = _probe_campaign("production", n), _probe_campaign("no-encoder", n)
        assert pa["valid"] and pa["pending"] == [], (n, pa["bad"], pa["pending"])
        assert pa["ep2"]["full"] and pa["ep2"]["met"] and len(pa["ep2"]["d"][0]) == R.N_SEEDS, n
        ok, why = R.null_precondition("_plan_w", pa, na)
        assert not ok and "objective endpoint was met" in why, (n, why)


def test_e_v2_and_the_dispositions_are_read_on_the_look_1_block_at_later_looks():
    runs, e2 = _look2_runs("harm")
    for s in range(16, 24):  # later replay rows that would move E-V2 if they were read
        e2["real"][str(s)].update(learn=0.9, frozen=0.1, shuf=0.1)
    a1 = R.analyse("production", R.fixture_runs("harm"), R.fixture_e2(), logs_pass=True)
    a2 = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    assert len(a2["ev2"]["d"][0]) == R.N_SEEDS and a2["ev2"] == a1["ev2"]
    assert (a2["ev1"], a2["ep1"], a2["ep2"], a2["disp"]) == (
        a1["ev1"],
        a1["ep1"],
        a1["ep2"],
        a1["disp"],
    )


def test_a_replay_row_bound_to_another_series_or_tree_voids_the_tier():
    # re-panel A-M2: N and n_calls alone tie nothing (a re-run on a new tree keeps both)
    runs = R.fixture_runs("noharm")
    assert R.analyse("production", runs, R.fixture_e2(), logs_pass=True)["valid"]
    for field, value in (("digest", "0" * 64), ("tree_sha", "b" * 40)):
        e2 = R.fixture_e2()
        e2["real"]["9"][field] = value
        a = R.analyse("production", runs, e2, logs_pass=True)
        assert not a["valid"] and any("replay row seed 9" in b for b in a["bad"]), (field, a["bad"])
    e2 = R.fixture_e2()
    del e2["real"]["9"]["digest"]  # a replay record written before the binding is stale
    assert not R.analyse("production", runs, e2, logs_pass=True)["valid"]


def test_a_replay_row_past_h_v1s_stop_is_still_binding_checked():
    # addendum 9 (C-m12): a row past H-V1's stop is not read, yet a stale one still voids the tier
    runs, e2 = _extend(R.fixture_runs("noharm"), R.fixture_e2(), range(16, 168))
    e2 = _bind(e2, runs)
    e2["real"]["165"]["digest"] = "0" * 64
    a = R.analyse("production", runs, e2, seeds=range(8, 168), logs_pass=True)
    assert a["n_owed"]["H-V1"] < 165 - R.SEEDS[0], a["n_owed"]
    assert not a["valid"] and any("replay row seed 165" in b for b in a["bad"]), a["bad"]


def test_replay_writes_the_binding_of_the_run_it_replayed():
    run = R.fixture_runs("noharm")[("real", 8)]
    row = R.replay({8: run})["real"]["8"]
    assert row["tree_sha"] == run["tier"]["tree_sha"] == R.SYNTH_SHA
    assert row["digest"] == R.replay_binding(run)["digest"] and len(row["digest"]) == 64
    other = R.fixture_runs("noharm", "no-encoder")[("real", 8)]  # same seed, another series
    assert R.replay_binding(other)["digest"] != row["digest"]


def test_a_pending_check_withholds_the_sequential_decision():
    runs = R.fixture_runs("harm")
    del runs[("real_r2", 8)]  # the repeat arm has not run yet
    a = R.analyse("production", runs, R.fixture_e2(), logs_pass=True)
    assert a["valid"] and a["pending"], a["pending"]
    for k in ("H-P1", "H-V1"):
        h = a["hs"][k]
        assert "seq" not in h and not h["met"] and h["status"].startswith("PENDING"), (
            k,
            h["status"],
        )


def test_a_continue_is_never_printed_as_not_met():
    # re-panel C R2-M1: addendum 2 "CONTINUE is no verdict: it is never 'not met'"
    q = R.seq_decision(_series(0.03, 0.07), R.DH)
    assert q["decision"] == "continue", q
    assert "NOT MET" not in q["status"] and q["status"].startswith("CONTINUE at look 1"), q[
        "status"
    ]
    assert "(no verdict)" in q["status"]
    out = R.cross(*_tier("noharm", "production"), *_tier("noharm", "no-encoder"))["h_p1"]
    assert "NOT MET" not in out and "CONTINUE on the production tier (no verdict)" in out, out


def test_futility_is_taken_only_at_a_look_powered_by_the_preregistered_sd():
    # re-panel B-N2 + P1 amendment item 2 as folded: a futility stop only where the mde from the
    # pre-registered SD
    # is at or below the claimed effect; before that the campaign continues (non-binding: no type-I
    # cost)
    for n in (8, 16, 24):  # addendum 2's 4-look design at E1's SD: mde 0.199 / 0.065 / 0.042 > DH
        q = R.seq_decision(_series(-0.2, SD_P1, n), R.DH, sd_pre=SD_P1)
        assert q["decision"] == "continue" and q["cp"] < R.FUTILITY_CP, q
        assert not q["futility_open"], q
    # addendum 9, H-P1 cap 176: closed before n=72 (mde_pre at 64: 0.0354), open from 72
    for n in range(8, 72, 8):
        q = R.seq_decision(_series(-0.2, SD_P1, n), **R.SEQ["H-P1"])
        assert q["decision"] == "continue" and not q["futility_open"], q
    q = R.seq_decision(_futility_at_look_1(0.005, 25), R.DH, sd_pre=0.005)
    assert q["decision"] == "futility" and q["futility_open"], q
    # no pre-registered SD: no futility stop at all
    assert R.seq_decision(_futility_at_look_1(SD_P1, 2), R.DH)["decision"] == "continue"
    open_v1 = [n for n in R.SEQ["H-V1"]["looks"][:-1] if R.futility_open(n, **R.SEQ["H-V1"])]
    assert open_v1 == list(R.V1_FUTILITY_LOOKS), open_v1
    open_p1 = [n for n in R.SEQ["H-P1"]["looks"][:-1] if R.futility_open(n, **R.SEQ["H-P1"])]
    assert open_p1 == list(R.P1_FUTILITY_LOOKS), open_p1


def test_addendum_4_fixes_the_h_v1_cap_and_its_looks_in_one_place():
    # owner 2026-10-24 "Raise its cap to its power target": cap = target x inflation, in 8-seed
    # looks
    # the addendum's number: 134 x 1.0624 -> 144 (power .7935), one block more
    assert R.V1_CAP == 152
    assert R.SEQ["H-V1"]["looks"] == tuple(range(8, R.V1_CAP + 1, 8))
    assert R.LOOKS == (
        8,
        16,
        24,
        32,
    )  # addendum 2's default design: the seq_* helpers' default looks
    assert len(R.SEQ["H-V1"]["bounds"]) == len(R.SEQ["H-V1"]["looks"])
    normal = R.seq_boundaries(R.SEQ["H-V1"]["looks"])
    assert all(b >= x["t"] for b, x in zip(R.SEQ["H-V1"]["bounds"], normal, strict=True))


def test_boundaries_past_17_looks_spend_a_positive_alpha_at_look_1():
    # 2 - 2 Phi(z / sqrt(1/19)) is 0.0 in floating point: look 1 of the H-V1 design must not crash
    # or spend nothing
    bs = R.seq_boundaries(tuple(range(8, 153, 8)))
    assert 0 < bs[0]["spent"] < 1e-12 and bs[0]["z"] > 8
    assert abs(bs[-1]["spent"] - R.SEQ_ALPHA) < 1e-12


def test_a_decision_past_the_h_v1_cap_is_refused_and_a_look_past_32_is_accepted():
    cap = R.V1_CAP
    with pytest.raises(R.Refused):
        R.seq_record([0.01] * (cap + 8), **R.SEQ["H-V1"])
    assert R.seq_record([0.01] * 40, **R.SEQ["H-V1"])["n"] in R.SEQ["H-V1"]["looks"]


def test_addendum_9_fixes_the_h_p1_cap_and_its_looks_in_one_place():
    # addendum 9: first 8-seed look with power >= .8 under S, U and H05 in numpy and harness
    # (numbers at rpe_charge.P1_CAP)
    assert R.P1_CAP == 176
    assert R.SEQ["H-P1"]["looks"] == R.P1_LOOKS == tuple(range(8, R.P1_CAP + 1, 8))
    # type I holds under every variant and the raw shape (max .0078): the normal-model boundaries
    assert R.SEQ["H-P1"]["bounds"] is None
    assert R.P1_FUTILITY_LOOKS == tuple(range(72, 169, 8))  # mde_pre at 72 is 0.0316 <= DH
    # one campaign: the longer primary's looks (H-P1's)
    assert R.CAMPAIGN_LOOKS == R.P1_LOOKS and set(R.V1_LOOKS) < set(R.CAMPAIGN_LOOKS)


def test_a_decision_past_the_h_p1_cap_is_refused_and_a_look_at_it_is_accepted():
    with pytest.raises(R.Refused):
        R.seq_record([0.01] * (R.P1_CAP + 8), **R.SEQ["H-P1"])
    for n in (16, 104, R.P1_CAP):
        q = R.seq_decision([0.01 * (i % 3) for i in range(n)], **R.SEQ["H-P1"])
        assert q["n"] == n and q["n_looks"] == len(R.P1_LOOKS), q
    with pytest.raises(R.Refused):  # past H-V1's own cap
        R.seq_decision([0.01] * 160, **R.SEQ["H-V1"])


def test_compare_at_a_look_past_32_decides_both_primaries():
    runs, e2 = _extend(*_look2_runs("noharm"), range(24, 48))  # looks 3-5 (addendum 5)
    a = R.analyse("production", runs, _bind(e2, runs), seeds=range(8, 48), logs_pass=True)
    assert a["valid"] and a["pending"] == [], (a["bad"], a["pending"])
    for k in ("H-P1", "H-V1"):  # each read at its own stop of record, else at the look (A-M4)
        assert a["hs"][k]["n"] == a["n_owed"][k] == a["hs"][k]["seq"]["n"] <= 40, (k, a["n_owed"])
    with pytest.raises(R.Refused):  # 44 seeds is no look
        R.analyse("production", runs, e2, seeds=range(8, 52), logs_pass=True)


def test_cli_look_reaches_the_campaign_cap_and_no_further(monkeypatch):
    seen = []
    monkeypatch.setattr(R, "_analyse_dir", lambda d, tier, seeds: (seen.append(list(seeds)), None))
    monkeypatch.setattr(R, "render", lambda a: "")
    assert R.main(["compare", "x", "production", "--look", "5"]) == 0
    assert seen == [list(range(8, 48))]
    with pytest.raises(R.Refused):
        R.main(["compare", "x", "production", "--look", str(len(R.CAMPAIGN_LOOKS) + 1)])
    assert R.main(["compare", "x", "production", "--look", str(len(R.CAMPAIGN_LOOKS))]) == 0
    assert seen[-1] == list(range(8, 8 + R.P1_CAP))  # look 22: H-P1's cap, past H-V1's


def _past_v1_cap(tier, null_v1=False):
    """Look 20 (160 seeds) on `tier`, past H-V1's cap 152 (addendum 9, as addendum 5): real +
    pfrozen on every seed, replay rows on every seed too, so the rows past H-V1's stop (its cap at
    most: seeds 160-167 at least) must be ignored, not read. `null_v1`: the replay d_s alternate
    +-0.02 from seed 16, so H-V1 never crosses and runs to its cap."""
    runs, e2 = _extend(
        R.fixture_runs("noharm", tier), R.fixture_e2(tier=tier), range(16, 168), tier=tier
    )
    if null_v1:
        for s in range(16, 168):
            row = e2["real"][str(s)]
            row["frozen"] = row["learn"] - (0.02 if s % 2 else -0.02)
    return R.analyse(tier, runs, _bind(e2, runs), seeds=range(8, 168), logs_pass=True), runs


def test_compare_past_the_h_v1_cap_decides_h_p1_and_carries_h_v1_from_its_stop():
    a, _ = _past_v1_cap("production")
    assert a["valid"] and a["pending"] == [], (a["bad"], a["pending"])
    assert a["hs"]["H-P1"]["n"] == 160 and "seq" in a["hs"]["H-P1"]
    n = a["n_owed"]["H-V1"]  # A-M4: its stop of record, never past its cap
    assert a["hs"]["H-V1"]["n"] == n == a["hs"]["H-V1"]["seq"]["n"] <= R.V1_CAP


def test_cross_past_the_h_v1_cap_counts_h_v1_over_its_own_cap():
    (pa, pr), (na, nr) = _past_v1_cap("production"), _past_v1_cap("no-encoder")
    assert pa["hs"]["H-V1"]["met"], pa["hs"]["H-V1"]["status"]  # the "agrees on" branch runs
    out = R.cross(pa, pr, na, nr)
    n = min(pa["n_owed"]["H-V1"], na["n_owed"]["H-V1"])  # A-M4: over the stops, never past the cap
    assert n <= R.V1_CAP and len(out["table_v1"]) == n and len(out["table"]) == 160
    assert f"agrees on {out['agree_v1']}/{n}" in out["h_v1"], out["h_v1"]


def test_both_tiers_past_the_h_v1_cap_count_h_v1_agreement_over_its_own_cap(both_tiers):
    (pa, pr), (na, nr) = _past_v1_cap("production"), _past_v1_cap("no-encoder")
    na["hs"]["H-V1"]["d"] = [-x if i % 2 else x for i, x in enumerate(na["hs"]["H-V1"]["d"])]
    out = R.cross(pa, pr, na, nr)
    assert pa["hs"]["H-V1"]["met"] and na["hs"]["H-V1"]["met"]
    n = min(pa["n_owed"]["H-V1"], na["n_owed"]["H-V1"])
    assert f"agrees on only {out['agree_v1']}/{n}" in out["h_v1"], out["h_v1"]


@pytest.mark.parametrize("extra", [0, 1])
def test_both_tiers_agreement_bar_past_the_h_v1_cap_is_six_eighths_of_its_own_stop(
    both_tiers, extra
):
    # ceil(6 * n / 8) agreeing seeds over H-V1's own stop n (A-M4), never scaled by the 160-seed
    # look
    (pa, pr), (na, nr) = _past_v1_cap("production"), _past_v1_cap("no-encoder")
    n = min(pa["n_owed"]["H-V1"], na["n_owed"]["H-V1"])
    need = math.ceil(R.AGREE_MIN * n / R.N_SEEDS)
    flipped = n - need + extra
    d = na["hs"]["H-V1"]["d"]  # flip from the end of the seeds both tiers read
    na["hs"]["H-V1"]["d"] = [-abs(x) if n - flipped <= i < n else abs(x) for i, x in enumerate(d)]
    pa["hs"]["H-V1"]["d"] = [abs(x) for x in pa["hs"]["H-V1"]["d"]]
    out = R.cross(pa, pr, na, nr)
    assert out["need_agree_v1"] == need != math.ceil(R.AGREE_MIN * 160 / R.N_SEEDS)
    assert out["agree_v1"] == n - flipped
    word = "harm on both tiers" if not extra else f"agrees on only {need - 1}/{n}"
    assert word in out["h_v1"], out["h_v1"]


def test_h_v1_run_to_its_cap_is_carried_from_it_with_a_bar_of_six_eighths_of_the_cap(both_tiers):
    # addendum 9: H-P1's cap 176 lies past H-V1's 152, so H-V1 is the primary carried past its cap
    # (A-M4): its stop of record is the cap stop, and the agreement bar is ceil(6 * 152 / 8) = 114
    (pa, pr), (na, nr) = _past_v1_cap("production", True), _past_v1_cap("no-encoder", True)
    h = pa["hs"]["H-V1"]
    assert pa["valid"] and pa["pending"] == [], (pa["bad"], pa["pending"])
    assert pa["n_owed"]["H-V1"] == h["n"] == h["seq"]["n"] == R.V1_CAP, (pa["n_owed"], h["n"])
    assert h["seq"]["decision"] == "cap" and not h["met"], h["seq"]
    out = R.cross(pa, pr, na, nr)
    assert out["agree_v1_of"] == len(out["table_v1"]) == R.V1_CAP, out["agree_v1_of"]
    assert out["need_agree_v1"] == math.ceil(R.AGREE_MIN * R.V1_CAP / R.N_SEEDS) == 114


def test_h_v1_stopped_at_look_1_ignores_later_rows_and_counts_agreement_over_its_stop():
    # A-M4 (addendum 7): H-V1 crosses at look 1 while the campaign runs on; look 2's replay
    # rows (seeds 16-23) are written, but its verdict and its agreement bar read only its 8 seeds
    def stopped(tier):
        runs, e2 = _look2_runs("noharm", tier)
        _stop_h_v1_at_look_1(e2)
        return R.analyse(tier, runs, e2, seeds=range(8, 24), logs_pass=True), runs

    (pa, pr), (na, nr) = stopped("production"), stopped("no-encoder")
    h = pa["hs"]["H-V1"]
    assert pa["valid"] and h["met"] and h["status"].startswith("MET at look 1"), h["status"]
    assert pa["n_owed"]["H-V1"] == h["n"] == len(h["d"]) == 8, (pa["n_owed"], len(h["d"]))
    out = R.cross(pa, pr, na, nr)
    assert out["agree_v1_of"] == 8 and len(out["table_v1"]) == 8
    assert out["need_agree_v1"] == 6 != math.ceil(R.AGREE_MIN * 16 / R.N_SEEDS)


@pytest.mark.slow
@pytest.mark.parametrize("key,dist", [(k, d) for k, ds in CAP_SHAPE.items() for d in ds])
def test_rs3_slow_twin_power_at_the_cap_is_at_least_point_8(key, dist):
    # P1 item 2 inside the suite: power at each cap >= .8 under every shape it is powered on
    n = 40000
    walks = [_walk(ds, key) for ds in _planted(key, dist, CLAIMED[key], n, 707)]
    power = sum(d == "harm" for d, _, _ in walks) / n
    assert power >= 0.8 - 3 * _se(0.8, n), (key, dist, power)


# ------------- e1b-fold (arc 2 panel): A-M3, A-M4, C-M3 + A-m6, A-m4, A-m3 -- addendum 6, before
# any brain run
def test_degeneracy_is_screened_on_every_seed_of_a_later_look():
    # A-M3: seed 20 runs only real + pfrozen (look 2); the look-1 five-arm screen never saw it
    runs, e2 = _look2_runs("harm")
    for arm, r in (("real", -0.99), ("pfrozen", -0.98)):
        runs[(arm, 20)] = R.synth_run(
            arm, 20, "production", r=r, mae_value=0.4, wiggle=0.01, shift=arm != "real"
        )
    a = R.analyse("production", runs, _bind(e2, runs), seeds=range(8, 24), logs_pass=True)
    assert a["valid"] and 20 in a["degenerate"], (a["bad"], a["degenerate"])


def _degenerate(runs, s):
    """Seed `s` degenerate on `real` + `pfrozen` (a seed up to H-P1's stop runs both)."""
    for arm in ("real", "pfrozen"):
        runs[(arm, s)] = R.synth_run(
            arm, s, "production", r=-0.95, mae_value=0.4, wiggle=0.01, shift=arm != "real"
        )


def test_degeneracy_is_screened_on_seeds_past_the_h_v1_cap():
    # R3-M1 (addendum 9: H-P1's cap 176 lies past H-V1's 152): past H-V1's stop a seed's replay row
    # is written but not read, and H-P1 still reads its `real` + `pfrozen` pair. Seed 20 is read by
    # both; every other seed, 160-167 included, is healthy and not flagged
    runs, e2 = _extend(R.fixture_runs("noharm"), R.fixture_e2(), range(16, 168))
    for s in (20, 165):
        _degenerate(runs, s)
    a = R.analyse("production", runs, _bind(e2, runs), seeds=range(8, 168), logs_pass=True)
    assert a["valid"] and a["degenerate"] == [20, 165], (a["bad"], a["degenerate"])
    # each endpoint drops only the degenerate seeds it reads: H-V1 stops by 159, H-P1 reads both
    seed_165_index = 165 - R.SEEDS[0]
    assert a["n_owed"]["H-V1"] <= R.V1_CAP < seed_165_index < a["n_owed"]["H-P1"], a["n_owed"]
    assert a["sens"]["H-V1"]["dropped"] == [20] and a["sens"]["H-P1"]["dropped"] == [20, 165]
    assert a["sens"]["H-V1"]["n"] == a["hs"]["H-V1"]["n"] - 1, (a["sens"], a["hs"]["H-V1"]["n"])
    assert a["sens"]["H-P1"]["n"] == a["hs"]["H-P1"]["n"] - 2, (a["sens"], a["hs"]["H-P1"]["n"])
    txt = R.render(a)
    assert "every arm it ran): [20, 165]" in txt
    assert "H-V1 without degenerate seeds [20]:" in txt
    assert "H-P1 without degenerate seeds [20, 165]:" in txt


def test_a_degenerate_seed_past_the_h_v1_cap_leaves_h_v1_unchanged():
    # R3-M1: H-V1 stops by seed 159, so seed 165 (degenerate on every arm it ran) never enters it
    runs, e2 = _extend(R.fixture_runs("noharm"), R.fixture_e2(), range(16, 168))
    _degenerate(runs, 165)
    a = R.analyse("production", runs, _bind(e2, runs), seeds=range(8, 168), logs_pass=True)
    sv, h = a["sens"]["H-V1"], a["hs"]["H-V1"]
    assert a["valid"] and a["degenerate"] == [165], (a["bad"], a["degenerate"])
    assert sv["dropped"] == [] and not sv["seed_sensitive"]
    assert (sv["t"], sv["k"], sv["n"]) == (h["t"], h["k"], h["n"])
    assert "H-V1 without degenerate seeds []:" in R.render(a)


@pytest.mark.parametrize("look_end,missing", [(24, 20), (168, 167)])
def test_a_missing_pfrozen_run_up_to_the_h_p1_cap_is_pending(look_end, missing):
    # addenda 7, 9: while H-P1 continues, every `pfrozen` run up to its cap (seed 183) is owed; a
    # lost one is pending, never a silently shorter H-P1 look
    runs, e2 = _extend(R.fixture_runs("noharm"), R.fixture_e2(), range(16, look_end))
    del runs[("pfrozen", missing)]
    a = R.analyse("production", runs, _bind(e2, runs), seeds=range(8, look_end), logs_pass=True)
    assert f"pfrozen_{missing} missing" in a["pending"], a["pending"]


def test_a_seed_past_h_p1s_stop_runs_real_only_and_its_degeneracy_is_still_screened():
    # R3-M1: past H-P1's stop of record (here look 1, harm) a seed owes no `pfrozen` run while
    # H-V1 continues; a degenerate `real` on such a seed is flagged and dropped from H-V1 only
    runs, e2 = _extend(R.fixture_runs("harm"), R.fixture_e2(), range(16, 24), "harm")
    runs = _drop(runs, lambda k: k[0] == "pfrozen" and k[1] >= 16)
    runs[("real", 20)] = R.synth_run("real", 20, "production", r=-0.95, mae_value=0.4, wiggle=0.01)
    a = R.analyse("production", runs, _bind(e2, runs), seeds=range(8, 24), logs_pass=True)
    assert a["valid"] and a["pending"] == [] and a["degenerate"] == [20], (a["bad"], a["pending"])
    assert a["n_owed"]["H-P1"] == 8 and a["n_owed"]["H-V1"] == 16, a["n_owed"]
    assert a["sens"]["H-P1"]["dropped"] == [] and a["sens"]["H-V1"]["dropped"] == [20], a["sens"]


def test_a_primary_stopped_at_look_1_owes_no_later_runs_and_keeps_its_stop_verdict():
    # A-M4 (a): H-P1 crosses at look 1; at look 2 its pfrozen runs for seeds 16-23 are not owed
    runs, e2 = _look2_runs("harm")
    runs = _drop(runs, lambda k: k[0] == "pfrozen" and k[1] >= 16)
    a = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    assert a["valid"] and a["pending"] == [], a["pending"]
    h = a["hs"]["H-P1"]
    assert h["met"] and h["status"].startswith("MET at look 1"), h["status"]
    assert a["n_owed"]["H-P1"] == h["n"] == 8 and h["seq"]["look"] == 1, a["n_owed"]
    assert not a["hs"]["H-V1"]["status"].startswith("PENDING"), a["hs"]["H-V1"]["status"]


def test_h_v1_stopped_at_look_1_keeps_its_verdict_when_its_replay_rows_stop():
    # A-M4 (b): H-V1 crosses at look 1, so look 2's replay rows (seeds 16-23) are never written
    runs, e2 = _look2_runs("noharm")
    _stop_h_v1_at_look_1(e2)
    for s in range(16, 24):
        del e2["real"][str(s)]
    a = R.analyse("production", runs, e2, seeds=range(8, 24), logs_pass=True)
    h = a["hs"]["H-V1"]
    assert a["valid"] and h["met"] and h["status"].startswith("MET at look 1"), h["status"]
    assert a["n_owed"]["H-V1"] == h["n"] == 8, a["n_owed"]


def test_ruling_production_met_with_a_short_no_encoder_tier_is_the_candidate():
    # C-M3 + A-m6: #4616 "production tier decides" -- a short no-encoder tier is reported beside,
    # never a hold
    (pa, pr) = _tier("harm", "production")
    runs = _drop(R.fixture_runs("harm", "no-encoder"), lambda k: k[0] == "pfrozen" and k[1] >= 11)
    na = R.analyse("no-encoder", runs, R.fixture_e2(tier="no-encoder"), logs_pass=True)
    assert na["valid"] and na["hs"]["H-P1"]["status"].startswith("NO VERDICT")
    out = R.cross(pa, pr, na, runs)["h_p1"]
    assert not out.startswith("PENDING") and "harm-deletion candidate" in out, out
    assert "no-encoder short (3/8 seeds): reported beside" in out, out


def test_ruling_a_no_encoder_tier_on_another_tree_or_with_pending_checks_never_holds():
    pa, pr = _tier("harm", "production")
    other = {
        k: dict(v, tier=dict(v["tier"], tree_sha="b" * 40))
        for k, v in R.fixture_runs("harm", "no-encoder").items()
    }
    na = R.analyse(
        "no-encoder", other, _bind(R.fixture_e2(tier="no-encoder"), other), logs_pass=True
    )
    out = R.cross(pa, pr, na, other)["h_p1"]
    assert na["valid"] and not out.startswith("PENDING") and "harm-deletion candidate" in out, out
    assert "no-encoder on another tree" in out, out
    nb, nr = _tier("harm", "no-encoder")
    nb["pending"] = ["repeat arm missing"]
    out = R.cross(pa, pr, nb, nr)["h_p1"]
    assert "PENDING" not in out.split("; no-encoder")[0] and "harm-deletion candidate" in out, out
    assert "no-encoder checks pending (repeat arm missing): reported beside" in out, out


def test_a_zero_observed_sd_is_never_powered():
    # A-m4: every d_s equal gives mde 0; that is no power, and a constant negative mean is t = -inf
    assert R.ttest([-0.01] * 8) == float("-inf") and R.ttest([0.0] * 8) == 0.0
    q = R.seq_decision([-0.01] * R.P1_CAP, **R.SEQ["H-P1"])
    assert q["decision"] == "cap" and not q["powered"], q
    assert R.ZERO_SD in q["status"] and "NULL-RECORD path" not in q["status"], q["status"]
    assert R.ZERO_SD in R.status(R.harm([-0.01] * 8), R.DH) and "powered" not in R.status(
        R.harm([-0.01] * 8), R.DH
    ).replace("not powered", "")
    a = _analysis("noharm")  # pfrozen == real on every seed: H-P1's d_s are all 0
    assert f"{R.ZERO_SD}: H-P1, arm pfrozen" in a["hs"]["H-P1"]["status"], a["hs"]["H-P1"]["status"]


def test_rs1b_out_of_sample_shape_leave_one_out_keeps_type_one():
    # A-m3: R-S1b was calibrated on the 8 E1 values it resamples; each leave-one-out 7-value shape
    # is a held-out shape the calibration never saw. Pooled type I (8 shapes x 500 campaigns) stays
    # at .05 + 3 SE, and no single held-out shape exceeds .05 + 3 SE of its own 500
    n_sim = 500
    for key in ("H-P1", "H-V1"):
        rates = []
        for drop in range(len(E1_D[key])):
            pool = [x for i, x in enumerate(E1_D[key]) if i != drop]
            m = sum(pool) / len(pool)
            pool = [x - m for x in pool]
            rng, cap = random.Random(9000 + drop), R.SEQ[key]["looks"][-1]
            camps = (
                [rng.choice(pool) + rng.gauss(0, 1e-9) for _ in range(cap)] for _ in range(n_sim)
            )
            rates.append(sum(_walk(ds, key, futility=False)[0] == "harm" for ds in camps) / n_sim)
        pooled = sum(rates) / len(rates)
        assert pooled <= 0.05 + 3 * _se(0.05, n_sim * len(rates)), (key, rates)
        assert max(rates) <= 0.05 + 3 * _se(0.05, n_sim), (key, rates)
