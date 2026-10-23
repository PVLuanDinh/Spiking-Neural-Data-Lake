"""E1b harness (instruments/rpe_charge): the reds the pre-registration owes, all brain-free.

One real-brain smoke is marked slow; everything else runs on planted series or constructed runs.
The E1 numbers in `E1_*` are E1's recorded per-seed values (REPORT-TABLES, N=600, H = ticks >= 300).
"""

import copy
import json
import math
import os
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
def test_compare_fixtures_print_met_not_met_and_void():
    lines = R.fixture_lines()
    assert [ln.split(" ")[1] for ln in lines] == ["harm", "noharm", "void"]
    assert "H-P1 MET" in lines[0] and "H-P1 NOT MET" in lines[1]
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
def _tier(kind, tier):
    runs = R.fixture_runs(kind, tier)
    return R.analyse(tier, runs, R.fixture_e2(), logs_pass=True), runs


def test_cross_tier_requires_agreement_and_a_tier_effect():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    both = R.cross(pa, pr, na, nr)
    assert both["agree"] >= R.AGREE_MIN and both["tier_effect"]
    assert both["h_p1"].startswith("harm on both tiers") and ".0025" in both["h_p1"]
    # tiers whose same-seed series are identical are replicates: no cross-tier wording
    same = {k: dict(v, v=pr[k]["v"], series=pr[k]["series"]) for k, v in nr.items()}
    flat = R.cross(pa, pr, na, same)
    assert not flat["tier_effect"] and "indistinguishable" in flat["h_p1"]


def test_both_tiers_met_but_disagreeing_in_sign_is_not_harm_on_both_tiers():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("harm", "no-encoder")
    na = copy.deepcopy(na)
    d = na["hs"]["H-P1"]["d"]
    for i in range(3):  # three seeds flip sign; the tier's own bar is untouched (still MET)
        d[i] = -abs(d[i])
    out = R.cross(pa, pr, na, nr)
    assert out["agree"] == len(d) - 3 < R.AGREE_MIN
    assert "agrees on only" in out["h_p1"] and "tier finding" in out["h_p1"]


def test_one_tier_harm_is_a_tier_finding_and_a_missing_verdict_is_pending():
    (pa, pr), (na, nr) = _tier("harm", "production"), _tier("noharm", "no-encoder")
    one = R.cross(pa, pr, na, nr)
    assert "tier finding" in one["h_p1"] and R.SINGLE_TIER_HARM_DELETES is False
    assert "candidate" not in one["h_p1"].replace("no deletion candidate", "")
    void = R.analyse("no-encoder", nr, R.fixture_e2(), logs_pass=False)
    assert R.cross(pa, pr, void, nr)["h_p1"].startswith("PENDING")


def test_null_record_precondition_excludes_a_measured_harm_and_a_void_tier():
    (pa, _), (na, _) = _tier("noharm", "production"), _tier("noharm", "no-encoder")
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
    assert (
        a["hs"]["H-P1"]["status"] == "MET"
    )  # stage A alone still yields the primary harm endpoint
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
def test_cross_voids_the_pair_when_the_two_tiers_ran_on_different_trees():
    pa, pr = _tier("harm", "production")
    other = {
        k: dict(v, tier=dict(v["tier"], tree_sha="b" * 40))
        for k, v in R.fixture_runs("harm", "no-encoder").items()
    }
    na = R.analyse("no-encoder", other, R.fixture_e2(), logs_pass=True)
    assert na["valid"] and "PENDING" in R.cross(pa, pr, na, other)["h_p1"]


def test_h_v1_sign_agreement_is_its_own_not_h_p1s():
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


def test_h_v1_both_tiers_names_the_in_brain_check_and_the_single_tier_flag(monkeypatch):
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
    assert "SINGLE_TIER_HARM_DELETES = True" in out["h_v1"] and "objective finding" in out["h_v1"]


def test_null_precondition_branches():
    pa, _ = _tier("noharm", "production")
    na, _ = _tier("noharm", "no-encoder")
    assert R.null_precondition("ValenceLearner", pa, na)[0]  # the positive path

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


def test_cross_is_pending_when_a_tiers_endpoint_is_incomplete():
    (pa, pr), _ = _tier("harm", "production"), _tier("harm", "no-encoder")
    runs = _drop(R.fixture_runs("harm", "no-encoder"), lambda k: k[0] == "pfrozen" and k[1] >= 11)
    na = R.analyse("no-encoder", runs, R.fixture_e2(), logs_pass=True)
    assert na["valid"] and na["hs"]["H-P1"]["status"].startswith("NO VERDICT")
    out = R.cross(pa, pr, na, runs)
    assert out["h_p1"].startswith("PENDING") and "tier finding" not in out["h_p1"]
    short = _drop(R.fixture_runs("harm", "no-encoder"), lambda k: k[0] == "real" and k[1] == 15)
    nb = R.analyse("no-encoder", short, R.fixture_e2(), logs_pass=True)
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
    runs = {}
    for s in R.SEEDS:
        d = -0.05 if s == 9 else 0.02 + 0.002 * (s % 3)
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
    a = R.analyse("production", runs, R.fixture_e2(), logs_pass=True)
    assert a["degenerate"] == [9]
    assert not a["hs"]["H-P1"]["met"]  # the degenerate seed hides the harm the other seven show
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
    assert "harm on both tiers" not in out["h_p1"]


def test_null_precondition_needs_one_tree_across_both_tiers():
    pa, _ = _tier("noharm", "production")
    other = {
        k: dict(v, tier=dict(v["tier"], tree_sha="b" * 40))
        for k, v in R.fixture_runs("noharm", "no-encoder").items()
    }
    na = R.analyse("no-encoder", other, R.fixture_e2(), logs_pass=True)
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


def test_cross_thresholds_are_pinned_at_their_boundaries():
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


def test_the_single_tier_flag_only_ever_names_production(monkeypatch):
    monkeypatch.setattr(R, "SINGLE_TIER_HARM_DELETES", True)
    (pa, pr), (na, nr) = _tier("noharm", "production"), _tier("harm", "no-encoder")
    out = R.cross(pa, pr, na, nr)
    assert (
        "no deletion candidate" in out["h_p1"]
        and "SINGLE_TIER_HARM_DELETES = True" not in out["h_p1"]
    )


def test_null_precondition_refuses_a_tier_with_pending_checks():
    (pa, _), (na, _) = _tier("noharm", "production"), _tier("noharm", "no-encoder")
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
