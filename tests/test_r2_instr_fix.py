"""The R2 instrument review's fixes (#6735, lanes/engine-1024/r2-instr-review-row/REPORT.md), each
red first: the licence on every entry point (MAJOR-1), one part per block and per-part verdicts
(MAJOR-2), the reader-kind off arms (MAJOR-4), the bout restarts' chronology (MAJOR-6) and the code
MINORs. Round 2 (integrator rulings): R2-c7's per-seed K1g arms (BLOCK-1 route A). Synthetic
records, synthetic percept events and patched children only; no brain is built and no confirmatory
or reserve seed is ever run."""

import copy
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_r2_p3_percept_transitions import _chain, _hydrate, _pairs
from test_replay_hydration_instrument import REDS, _arms, _differ, _fw_rec, _per_seed, _record

from snn_data_lake.instruments import replay_hydration as rh

CONFIRMATORY = (15010, 15020, 15030, 15040, 15050, 15060, 15067)


def _never(*_args, **_kw):
    raise AssertionError("the entry point went past the licence")


@pytest.fixture
def no_go(monkeypatch):
    monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    monkeypatch.setattr(rh.os, "_exit", _never)  # an unguarded child must not end the test run
    monkeypatch.setattr(rh, "_owned_private_home", _never)


# ---- MAJOR-1: every entry point refuses a confirmatory or reserve seed without the licence ------


@pytest.mark.parametrize("seed", CONFIRMATORY)
def test_run_process_refuses_a_confirmatory_seed_without_the_licence(no_go, tmp_path, seed):
    with pytest.raises(ValueError, match="licence"):
        rh.run_process("twin_K1", seed, str(tmp_path), str(tmp_path / "o.json"), nonce="n")


@pytest.mark.parametrize("seed", CONFIRMATORY)
def test_proc_command_line_refuses_a_confirmatory_seed_without_the_licence(no_go, tmp_path, seed):
    argv = ["--proc", "twin_K1", str(seed), str(tmp_path), str(tmp_path / "o.json"), "r2", "n"]
    with pytest.raises(ValueError, match="licence"):
        rh.main(argv)


@pytest.mark.parametrize("seed", CONFIRMATORY)
def test_run_failwrite_refuses_a_confirmatory_seed_without_the_licence(
    no_go, tmp_path, monkeypatch, seed
):
    monkeypatch.setattr(rh, "_fw_reference", _never)
    with pytest.raises(ValueError, match="licence"):
        rh.run_failwrite("fw_ref_A", seed, str(tmp_path / "o.json"))
    assert not (tmp_path / "o.json").exists(), "a refused arm writes no record"


@pytest.mark.parametrize("seed", CONFIRMATORY)
def test_verdict_refuses_to_score_a_confirmatory_seed_without_the_licence(no_go, seed):
    with pytest.raises(ValueError, match="licence"):
        rh.verdict({seed: _per_seed()}, [seed])


def test_verdict_and_run_process_still_admit_the_dev_block(no_go):
    v = rh.verdict({s: _per_seed() for s in rh.DEV_SEEDS}, list(rh.DEV_SEEDS))
    assert v["seeds"] == list(rh.DEV_SEEDS)
    with pytest.raises(AssertionError, match="past the licence"):
        rh.run_process("twin_K1", 15000, "h", "o.json", nonce="n")


# ---- MAJOR-2: the licence names ONE part's block; each part's verdict is its own (2.3, 2.6, D4)

P1_BLOCK, P1_RESERVE = tuple(range(15010, 15018)), tuple(range(15020, 15028))
P2_BLOCK, P2_RESERVE = tuple(range(15030, 15038)), tuple(range(15040, 15048))


@pytest.mark.parametrize(
    "go, seeds",
    [
        ("1", [15010]),  # a bare go names no block: it licenses nothing
        ("P1", [15030]),  # P2's block under P1's go
        ("P1", [15040]),  # P2's reserve under P1's go
        ("P2", [15017]),  # P1's block under P2's go
        ("P1", [15000, 15010]),  # one block per run: dev seeds never mix into a confirmatory read
        ("P1", [15010, 15030]),
        ("P3", [15010]),  # P1's block under P3's go
        ("P2", [15050]),  # P3's block under P2's go
        ("P1", [15060]),  # P3's reserve under P1's go
        ("P1", [15068]),  # unassigned
        ("P1", [15999]),
    ],
)
def test_the_licence_admits_only_the_named_parts_block_and_reserve(monkeypatch, go, seeds):
    monkeypatch.setenv("R2_CONFIRMATORY_GO", go)
    with pytest.raises(ValueError, match="licence"):
        rh.check_seeds(seeds)


def test_the_licence_names_the_part_it_admits(monkeypatch):
    monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    assert rh.check_seeds(rh.DEV_SEEDS) == "dev"
    monkeypatch.setenv("R2_CONFIRMATORY_GO", "P1")
    assert rh.check_seeds(P1_BLOCK) == "P1" and rh.check_seeds(P1_RESERVE[:1]) == "P1"
    monkeypatch.setenv("R2_CONFIRMATORY_GO", "P2")
    assert rh.check_seeds(P2_BLOCK) == "P2" and rh.check_seeds(P2_RESERVE[-1:]) == "P2"


def _part_verdict(monkeypatch, part, seeds, **over):
    monkeypatch.setenv("R2_CONFIRMATORY_GO", part)
    return rh.verdict({s: _per_seed(**over) for s in seeds}, list(seeds))


def test_p2_verdict_is_c6_only_a_p1_red_never_reverts_p2(monkeypatch):
    v = _part_verdict(monkeypatch, "P2", P2_BLOCK, c1={"K1": False}, failwrite={"ok": False})
    assert v["part"] == "P2" and v["verdict"] == "PASS", v
    v = _part_verdict(monkeypatch, "P2", P2_BLOCK, c6={"K3": False})
    assert v["verdict"] == "RED"


def test_p1_verdict_never_joins_c6_a_p2_red_never_reverts_p1(monkeypatch):
    v = _part_verdict(monkeypatch, "P1", P1_BLOCK, c6={"K3": False})
    assert v["part"] == "P1" and v["verdict"] == "PASS", v
    assert v["R2-c6"] == "not P1's verdict"
    v = _part_verdict(monkeypatch, "P1", P1_BLOCK, c1={"K1": False})
    assert v["verdict"] == "RED"


def test_the_dev_block_keeps_the_joined_reading(monkeypatch):
    monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    v = rh.verdict({s: _per_seed(c6={"K3": False}) for s in rh.DEV_SEEDS}, list(rh.DEV_SEEDS))
    assert v["part"] == "dev" and v["verdict"] == "RED"


# ---- MAJOR-4: a reader-kind guard-off arm fires only on its own effect (2.4, A11.2) -------------


def _off_arm(arms, kind):
    off = arms[f"{kind}_off"]
    if kind == "HELDOUT":  # the explicit 2.4 check: the planted id sits in the neutered arm's ring
        off["planted"] = off["before"]["ring_ids"][0]
    return off


@pytest.mark.parametrize("kind", rh._READER_KINDS)
def test_a_latched_reader_off_arm_is_invalid_never_fired(kind):
    arms = _arms()
    off = _off_arm(arms, kind)
    off["before"]["ring"][0] = ["MEMORY", "admitted"]
    off["fault"] = {"phase": "hydrate", "cause": "ValueError: another guard latched"}
    c = rh.seed_comparisons(arms)
    assert not c["planted"][kind]["guard_off_flagged"], "a latched off arm read FIRED"
    assert f"{kind}_off" in c["incomplete"] and not c["arms_complete"]


@pytest.mark.parametrize("kind", rh._READER_KINDS)
def test_a_reader_off_arm_fires_on_ring_or_draws_not_on_weights_alone(kind):
    arms = _arms()
    _differ(_off_arm(arms, kind))  # weights only: not where 2.4 says this red must fire
    assert not rh.seed_comparisons(arms)["planted"][kind]["guard_off_flagged"]
    for where in ("ring", "ring_ids", "drawn", "drawn_ids"):
        arms = _arms()
        off = _off_arm(arms, kind)
        if where.startswith("ring"):
            off["before"][where][-1] = ["MEMORY", "admitted"] if where == "ring" else "9-9"
        else:
            off["rows"][0][where] = ["9-9"]
        c = rh.seed_comparisons(arms)
        assert c["planted"][kind]["guard_off_flagged"], (kind, where)
        assert f"{kind}_off" not in c["incomplete"]


# ---- MAJOR-6: the chronology/motor check covers the bout restarts too (2.2, R2-c5's half) --------


@pytest.mark.parametrize("role", ["res_Bpre", "res_Bpost"])
@pytest.mark.parametrize("leak", ["tick", "motor"])
def test_chrono_covers_the_bout_crash_point_restarts(role, leak):
    arms = _arms()
    assert rh.seed_comparisons(arms)["chrono"][role] is True
    if leak == "tick":  # the redo of the bout effect ticked the brain
        arms[role]["construct_chrono"]["step"] = 1
    else:  # or reached So101Arm.apply()
        arms[role]["construct_motor"] = 1
    assert rh.seed_comparisons(arms)["chrono"][role] is False


# ---- the code MINORs of #6735 --------------------------------------------------------------------


def test_minor2_each_red_reports_which_records_it_moves():
    arms = _arms()
    _differ(arms["O_w"])
    attribution = rh.seed_comparisons(arms)["attribution"]
    assert attribution["O_w"] == ["rows.weights"] and attribution["O_wm"] == []


def test_minor3_c1_needs_an_unlatched_restore_even_when_the_faults_match():
    arms = _arms()
    for role in ("res_K1", "twin_K1"):
        arms[role]["fault"] = {"phase": "reconcile", "cause": "same"}
    assert rh.seed_comparisons(arms)["c1"]["K1"] is False


def test_minor7_the_scope_discloses_the_ctxread_off_arm_construction():
    v = rh.verdict({15000: _per_seed()}, [15000])
    assert "CTXREAD" in v["scope"]


_SURVIVORS = [f"fw_{k}_{a}" for k in ("t", "b") for a in ("absent", "landed", "torn")]


def test_minor8_posed_is_verified_only_when_every_survivor_vii_call_refused():
    arms = {r: _fw_rec(role=r) for r in _SURVIVORS}
    arms["fw_red_posed"] = _fw_rec(role="fw_red_posed")
    arms["fw_red_posed"]["calls"][-1]["exc"] = None  # a red's reopened latch is not a survivor
    assert rh.fw_seed(arms)["posed"] == "VERIFIED"
    broken = json.loads(json.dumps(arms))
    broken["fw_t_landed"]["calls"][-1]["exc"] = None  # pass 2, C(vii), the second survivor
    assert rh.fw_seed(broken)["posed"] == "UNVERIFIED"
    del arms["fw_b_torn"]
    assert rh.fw_seed(arms)["posed"] == "UNVERIFIED"


def test_minor9_a_hung_arm_is_recorded_invalid_not_a_campaign_abort(tmp_path, monkeypatch):
    def hang(args, **kwargs):
        raise rh.subprocess.TimeoutExpired(args, kwargs.get("timeout"))

    monkeypatch.setattr(rh.subprocess, "run", hang)
    arms = rh.run_seed(15000, tmp_path, roles=("twin_K1",))
    assert arms["twin_K1"] == {"role": "twin_K1", "error": "timeout"}


def test_minor10_c4_and_c6_hold_on_every_seed_an_invalid_one_included():
    s1, s2 = rh.DEV_SEEDS[:2]
    for over, line in ((dict(c6={"K3": False}), "R2-c6"), (dict(c4_comparator=False), "R2-c4")):
        per = {s1: _per_seed(), s2: _per_seed(**over)}
        per[s2]["c2"]["I_K2_differs"] = False
        v = rh.verdict(per, [s1, s2])
        assert v["invalid_seeds"] == [s2] and v[line] == "RED", (line, v[line])


def test_minor11_a_red_missing_from_the_first_seed_is_still_counted():
    s1, s2 = rh.DEV_SEEDS[:2]
    per = {s1: _per_seed(), s2: _per_seed(reds={**dict.fromkeys(REDS, True), "FW:RED-X": False})}
    v = rh.verdict(per, [s1, s2])
    assert "FW:RED-X" in v["red_fires_on_no_seed"] and v["reds"]["FW:RED-X"] is False


def _quiet_child(launched):
    def child(args, **kwargs):
        launched.append(args[3])
        return rh.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    return child


def test_minor12_roles_bring_their_parents(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(rh.subprocess, "run", _quiet_child(launched))
    arms = rh.run_seed(15000, tmp_path, roles=("res_K1", "fw_rs2_k_post"))
    assert launched == ["pre_K1", "res_K1", "fw_k_post", "fw_rs_k_post", "fw_rs2_k_post"]
    assert set(arms) == set(launched)


def test_minor12_the_base_head_needs_its_own_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(rh.subprocess, "run", _never)
    with pytest.raises(ValueError, match="base-src"):
        rh.run_seed(15000, tmp_path, head="base")


def test_minor12_m_post_sets_the_post_boundary_turns(monkeypatch):
    turns = []

    class Fake:
        def respond(self, u, human=False):
            turns.append(("respond", u))

        def correct(self, u, label):
            turns.append(("correct", u))

    monkeypatch.setattr(rh, "M_POST", 8)
    rh._post_boundary(Fake(), [f"u{i}" for i in range(20)])
    assert len(turns) == 8 and [t for t, _ in turns].count("correct") == 2


def test_minor5_a_base_tree_not_at_e1b3ae3c_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "_head_of", lambda path: "deadbeef" * 5)
    monkeypatch.setattr(rh.subprocess, "run", _never)
    monkeypatch.setattr(rh, "run_seed", _never)
    with pytest.raises(ValueError, match="e1b3ae3c"):
        rh.run_campaign(tmp_path, rh.DEV_SEEDS[:1], base_src=str(tmp_path))


def test_minor5_the_base_seed_run_is_refused_off_the_registered_head(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "_head_of", lambda path: "deadbeef" * 5)
    monkeypatch.setattr(rh.subprocess, "run", _never)
    with pytest.raises(ValueError, match="e1b3ae3c"):
        rh.run_seed(15000, tmp_path, head="base", src=str(tmp_path))


def test_minor5_the_verdict_records_which_heads_ran(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "_head_of", lambda path: "e1b3ae3c" + "0" * 32)
    monkeypatch.setattr(rh, "run_seed", lambda *a, **kw: {})
    monkeypatch.setattr(rh, "seed_comparisons", lambda arms, base=None: _per_seed())
    rh.run_campaign(tmp_path, rh.DEV_SEEDS[:1], base_src=str(tmp_path))
    prov = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))["provenance"]
    assert prov["base_src"].startswith("e1b3ae3c") and prov["src"]
    assert len(prov["instrument_sha256"]) == 64


# ---- ROUND 2, BLOCK-1 route A: R2-c7's per-seed K1g arms (2.10; integrator ruling) ---------------

P3_BLOCK, P3_RESERVE = tuple(range(15050, 15058)), tuple(range(15060, 15068))
PERCEPT_REASONS = {
    "DUP": "duplicate_id",
    "REORDER": "seq_regression",
    "HELDOUT": "heldout",
    "MISSING": "missing_slot",
    "WIDTH": "width",
    "VOCAB": "vocab",
    "INSTRUMENT": "instrument",
}
#: (role, parent): session 1 closed cleanly (so the lake holds a session boundary for CROSS), the
#: killed session 2 and its uninterrupted twin, then the restoring arms on the killed home's copy
P3_TABLE = (
    ("pre_K1g0", None),
    ("pre_K1g", "pre_K1g0"),
    ("twin_K1g", "pre_K1g0"),
    ("K1g_POS", "pre_K1g0"),
    ("res_K1g", "pre_K1g"),
    ("K1g_CROSS", "pre_K1g"),
    ("O_buf", "pre_K1g"),
) + tuple((f"K1g_{k}_{side}", "pre_K1g") for k in PERCEPT_REASONS for side in ("on", "off"))


def test_p3_licence_admits_its_block_and_reserve_and_names_the_part(monkeypatch):
    monkeypatch.setenv("R2_CONFIRMATORY_GO", "P3")
    assert rh.check_seeds(P3_BLOCK) == "P3" and rh.check_seeds(P3_RESERVE) == "P3"


@pytest.mark.parametrize("arm", P3_TABLE, ids=[r for r, _ in P3_TABLE])
def test_every_registered_k1g_arm_is_in_the_role_table_with_its_parent(arm):
    assert arm in rh.P3_ROLES
    assert rh.PERCEPT_PLANTED == PERCEPT_REASONS


@pytest.mark.parametrize("role", [r for r, _ in P3_TABLE])
def test_run_process_hands_every_k1g_arm_to_the_percept_process(tmp_path, monkeypatch, role):
    calls = []
    monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    monkeypatch.setattr(rh, "_owned_private_home", lambda path, token: Path(path))
    monkeypatch.setattr(rh, "run_percept", lambda r, seed, out: calls.append((r, seed)))
    monkeypatch.setattr(rh, "_patch_restore", _never)  # never the interaction learner's arms
    rh.run_process(role, 15000, str(tmp_path), str(tmp_path / "o.json"), nonce="n")
    assert calls == [(role, 15000)]


def test_run_seed_runs_each_k1g_arm_on_a_copy_of_its_parents_home(tmp_path, monkeypatch):
    launched = []

    def child(args, **kwargs):
        launched.append(args[3])
        (Path(args[5]) / f"{args[3]}.mark").write_text("x", encoding="ascii")
        return rh.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(rh.subprocess, "run", child)
    homes = {}
    real_copy = rh.shutil.copytree

    def copied(src, dst, **kw):
        out = real_copy(src, dst, **kw)
        homes[Path(dst).name.split("-")[0]] = sorted(p.name for p in Path(dst).glob("*.mark"))
        return out

    monkeypatch.setattr(rh.shutil, "copytree", copied)
    rh.run_seed(15000, tmp_path, roles=[r for r, _ in P3_TABLE])
    assert launched == [r for r, _ in P3_TABLE]
    assert homes["res_K1g"] == ["pre_K1g.mark", "pre_K1g0.mark"]
    assert homes["twin_K1g"] == homes["K1g_POS"] == ["pre_K1g0.mark"]


def _lake():
    """Two sessions of synthetic percept chains: the twin's hydration makes 2 + 3 pairs."""
    return _chain(1, 1, [0, 1, 2]) + _chain(2, 4, [3, 0, 1, 2])


def _keep(monkeypatch):
    """Every guard an arm patches is restored after the test."""
    import snn_data_lake.growth.neurogenesis_engine as ne
    import snn_data_lake.substrate.learner_events as le

    for mod, name in (
        (ne, "_percept_refusal"),
        (ne, "hydrate_transitions"),
        (le, "_refusal"),
        (le, "_identity_refusal"),
    ):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    cls = ne.NeurogenesisSelfImprove
    monkeypatch.setattr(cls, "restore_transitions", cls.restore_transitions)


def test_k1g_cross_arm_pairs_across_the_session_boundary(monkeypatch):
    _keep(monkeypatch)
    twin = _pairs(_hydrate(_lake()))
    assert len(twin) == 5
    rh._patch_restore("K1g_CROSS")
    assert len(_pairs(_hydrate(_lake()))) == 6, "CROSS: one pair more than the twin's"


def test_o_buf_arm_restores_the_grower_with_no_transition_hydration(monkeypatch):
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    _keep(monkeypatch)
    rh._patch_restore("O_buf")
    eng = NeurogenesisSelfImprove(refit_every=4, device="cpu", seed=15000, online_readout=None)
    assert eng.restore_transitions(_lake(), n_units=9, current_session=3) == 0
    assert eng._buf_x == [] and eng._buf_y == [] and eng._transitions_hydrated


@pytest.mark.parametrize("kind", sorted(PERCEPT_REASONS))
def test_k1g_planted_percept_is_refused_on_and_forms_one_pair_off(monkeypatch, kind):
    from snn_data_lake.substrate import learner_events as le

    _keep(monkeypatch)
    events, reason = _lake(), PERCEPT_REASONS[kind]
    planted = rh._percept_red(kind, events)
    twin = _pairs(_hydrate(events))
    on = _hydrate(events + [copy.deepcopy(planted)])
    assert on["refusals"].get(reason, 0) >= 1 and _pairs(on) == twin, on["refusals"]
    reader = le._refusal(copy.deepcopy(planted), le.KIND_SPECS["percept"], False)
    if kind in ("HELDOUT", "MISSING", "INSTRUMENT"):  # the shared reader refuses these first
        assert reader == reason
    rh._neuter(kind, percept=True)
    off = _hydrate(events + [copy.deepcopy(planted)])
    assert not off["refusals"] and len(_pairs(off)) == len(twin) + 1, off["refusals"]
    assert le._refusal(copy.deepcopy(planted), le.KIND_SPECS["percept"], False) is None
    if kind in ("DUP", "REORDER"):
        assert le._identity_refusal(planted, {planted["event_id"]}, (9, 9)) is None


class _FakeNeuro:
    refit_every, ticks, _model, _n_units = 4, 5, None, 2

    def __init__(self):
        self.trajectory = [(4, 2, 0.5)]

    def tick(self):
        self.ticks += 1
        if self.ticks % self.refit_every == 0:
            self.trajectory.append((self.ticks, 2, 0.75))


def test_k1g_runs_only_the_grower_sleep_step_to_the_next_refit():
    neuro = _FakeNeuro()
    model = rh._next_refit(neuro)
    assert neuro.ticks == 8 and model["ticks"] == 8 and model["refit_row"] is True
    assert model["traj"] == [[4, 2, 0.5], [8, 2, 0.75]] and model["W"] is None


def _k1g_rec(role):
    return {
        "role": role,
        "seed": 15000,
        "fault": None,
        "reader": {"refusals": {}},
        "before": {
            "buf_x": [[0.5, 0.25]],
            "buf_y": [1],
            "cats": [0, 1],
            "refusals": {},
            "prev_none": True,
            "observed": 0,
            "n_units": 2,
        },
        "model": {
            "W": [[0.1]],
            "b": [0.0],
            "beta": [[0.2]],
            "units": 2,
            "traj": [[4, 2, 0.5]],
            "ticks": 4,
            "refit_row": True,
        },
    }


def _k1g_arms():
    arms = {r: _k1g_rec(r) for r, _ in P3_TABLE if not r.startswith("pre_")}
    arms.update({r: {"role": r} for r, _ in P3_TABLE if r.startswith("pre_")})
    for role in ("K1g_CROSS", "O_buf"):  # each red moves the hydrated buffer
        arms[role]["before"]["buf_y"] = [0]
    arms["K1g_POS"]["model"]["W"] = [[0.3]]  # the shuffled buffer's refit differs
    for kind, reason in PERCEPT_REASONS.items():
        arms[f"K1g_{kind}_on"]["before"]["refusals"] = {reason: 1}
        arms[f"K1g_{kind}_off"]["before"]["buf_y"] = [1, 0]
    return arms


def test_c7_is_met_on_the_registered_arms():
    c = rh.c7_seed(_k1g_arms())
    assert c["ok"] and c["K1g"] and c["positive_control"] and c["arms_complete"], c
    assert set(c["reds"]) == {"CROSS", "O_buf", *PERCEPT_REASONS} and all(c["reds"].values())
    assert rh.seed_comparisons({**_arms(), **_k1g_arms()})["c7"]["ok"]


@pytest.mark.parametrize(
    "where, key, value",
    [("before", k, v) for k, v in (("buf_x", [[0.5, 0.5]]), ("buf_y", [0]), ("cats", [1]))]
    + [("model", k, v) for k, v in (("W", [[0.2]]), ("b", [1.0]), ("beta", [[0.0]]))]
    + [("model", k, v) for k, v in (("units", 3), ("traj", []), ("ticks", 8))]
    + [("before", "prev_none", False), ("before", "observed", 1), ("before", "refusals", {"x": 1})],
)
def test_k1g_restart_bar_is_the_buffer_the_next_refit_and_an_untouched_chronology(
    where, key, value
):
    arms = _k1g_arms()
    arms["res_K1g"][where][key] = value
    c = rh.c7_seed(arms)
    assert c["K1g"] is False and c["ok"] is False


def test_k1g_a_latched_restore_never_meets_the_bar():
    arms = _k1g_arms()
    for role in ("res_K1g", "twin_K1g"):
        arms[role]["fault"] = {"phase": "reconcile", "cause": "same"}
    assert rh.c7_seed(arms)["K1g"] is False


@pytest.mark.parametrize("role, red", [("K1g_CROSS", "CROSS"), ("O_buf", "O_buf")])
def test_each_k1g_red_fires_only_when_it_moves_the_record(role, red):
    arms = _k1g_arms()
    arms[role] = _k1g_rec(role)  # equal to the twin
    assert rh.c7_seed(arms)["reds"][red] is False
    arms = _k1g_arms()
    arms[role]["fault"] = {"phase": "hydrate", "cause": "ValueError: x"}
    assert rh.c7_seed(arms)["reds"][red] is False
    arms[role] = {"role": role, "error": "exit 1"}
    c = rh.c7_seed(arms)
    assert c["reds"][red] is False and role in c["incomplete"] and not c["ok"]


@pytest.mark.parametrize("kind", sorted(PERCEPT_REASONS))
def test_each_k1g_planted_kind_reads_its_own_guard(kind):
    reason, on, off = PERCEPT_REASONS[kind], f"K1g_{kind}_on", f"K1g_{kind}_off"
    arms = _k1g_arms()
    arms[on]["before"]["refusals"], arms[on]["reader"]["refusals"] = {}, {reason: 1}
    assert all(rh.c7_seed(arms)["planted"][kind].values()), "the reader's count is a refusal too"
    arms[on]["reader"]["refusals"] = {}
    assert rh.c7_seed(arms)["planted"][kind]["guard_on_refused"] is False
    arms = _k1g_arms()
    arms[on]["before"]["buf_y"] = [0]  # refused, but the restore moved: not ok unless latched
    assert rh.c7_seed(arms)["planted"][kind]["guard_on_ok"] is False
    arms[on]["fault"] = {"phase": "reconcile", "cause": "R2 closed: incomplete reconciliation"}
    assert rh.c7_seed(arms)["planted"][kind]["guard_on_ok"] is True
    arms = _k1g_arms()
    arms[off]["before"]["refusals"] = {reason: 1}  # the neutered guard still refused it
    assert rh.c7_seed(arms)["planted"][kind]["guard_off_flagged"] is False
    arms = _k1g_arms()
    arms[off]["before"]["buf_y"] = [1]  # admitted, but nothing moved
    assert rh.c7_seed(arms)["planted"][kind]["guard_off_flagged"] is False
    arms = _k1g_arms()
    arms[off]["fault"] = {"phase": "hydrate", "cause": "another guard latched"}
    c = rh.c7_seed(arms)
    assert c["planted"][kind]["guard_off_flagged"] is False and off in c["incomplete"]


def test_k1g_positive_control_the_shuffled_buffer_must_reach_the_refit():
    arms = _k1g_arms()
    arms["K1g_POS"] = _k1g_rec("K1g_POS")
    c = rh.c7_seed(arms)
    assert c["positive_control"] is False and c["ok"] is False


def test_k1g_a_twin_with_no_refit_after_the_boundary_is_incomplete():
    arms = _k1g_arms()
    arms["twin_K1g"]["model"]["refit_row"] = False
    c = rh.c7_seed(arms)
    assert "twin_K1g" in c["incomplete"] and not c["ok"]


def _p3_verdict(monkeypatch, seeds=P3_BLOCK, **over):
    monkeypatch.setenv("R2_CONFIRMATORY_GO", "P3")
    return rh.verdict(
        {s: _per_seed(c7=rh.c7_seed(_k1g_arms()), **over) for s in seeds}, list(seeds)
    )


def test_p3_verdict_is_r2_c7_only_a_p1_red_never_reverts_p3(monkeypatch):
    v = _p3_verdict(monkeypatch, c1={"K1": False}, c2={"I_K1_equal": False, "I_K2_differs": False})
    assert v["part"] == "P3" and v["verdict"] == "PASS" and v["R2-c7"] == "PASS", v
    assert v["R2-c1"] == "not P3's verdict" and not v["invalid_seeds"]


def test_p3_verdict_reads_red_when_c7_fails_on_one_seed(monkeypatch):
    monkeypatch.setenv("R2_CONFIRMATORY_GO", "P3")
    bad = _k1g_arms()
    bad["res_K1g"]["model"]["W"] = [[9.0]]
    per = {s: _per_seed(c7=rh.c7_seed(_k1g_arms())) for s in P3_BLOCK}
    per[P3_BLOCK[3]]["c7"] = rh.c7_seed(bad)
    v = rh.verdict(per, list(P3_BLOCK))
    assert v["R2-c7"] == "RED" and v["verdict"] == "RED"


def test_p3_an_incomplete_k1g_arm_makes_the_seed_invalid_never_a_pass(monkeypatch):
    monkeypatch.setenv("R2_CONFIRMATORY_GO", "P3")
    crashed = _k1g_arms()
    crashed["K1g_VOCAB_off"] = {"role": "K1g_VOCAB_off", "error": "exit 1"}
    per = {s: _per_seed(c7=rh.c7_seed(_k1g_arms())) for s in P3_BLOCK}
    per[P3_BLOCK[0]]["c7"] = rh.c7_seed(crashed)
    v = rh.verdict(per, list(P3_BLOCK))
    assert v["invalid_seeds"] == [P3_BLOCK[0]] and v["R2-c7"] == "NOT MET"
    assert v["verdict"] == "NOT MET"


def test_p3_a_k1g_red_that_fires_on_no_seed_halts_p3(monkeypatch):
    monkeypatch.setenv("R2_CONFIRMATORY_GO", "P3")
    quiet = _k1g_arms()
    quiet["K1g_CROSS"] = _k1g_rec("K1g_CROSS")
    v = rh.verdict({s: _per_seed(c7=rh.c7_seed(quiet)) for s in P3_BLOCK}, list(P3_BLOCK))
    assert v["halted"] and "K1g:CROSS" in v["red_fires_on_no_seed"] and v["R2-c7"] == "RED"


def test_p1_and_p2_verdicts_never_join_r2_c7(monkeypatch):
    for part, block in (("P1", P1_BLOCK), ("P2", P2_BLOCK)):
        v = _part_verdict(monkeypatch, part, block, c7={"ok": False})
        assert v["verdict"] == "PASS" and v["R2-c7"] == f"not {part}'s verdict", (part, v)
    monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    v = rh.verdict({s: _per_seed(c7={"ok": False}) for s in rh.DEV_SEEDS}, list(rh.DEV_SEEDS))
    assert v["R2-c7"] == "RED" and "not_built" not in v


# ---- ROUND 2, MAJOR-2 (a): R2-c3's reds by part, from the registration's own lines ---------------
#: 2.4's table and FAILWRITE's reds (A3: "then the P1 seeds of 2.6") are P1's; the 2026-10-06 P2
#: amendment's "Reds" names O_sleepskip and O_cursor (K4). Every other red joins both parts.
P1_ONLY_REDS = ("O_wm", "O_hyd", "O_rng", "O_w", "O_stack", "O_rw", "CHRONO", "MOTOR", "FW:KIND")
P2_ONLY_REDS = ("O_sleepskip", "O_cursor")
P1_ONLY_PLANTED = ("DUP", "REORDER", "HELDOUT", "MISSING")


def _quiet_red(name):
    reds = {**dict.fromkeys(REDS, True), "FW:KIND": True}
    reds[name] = False
    return reds


@pytest.mark.parametrize("red", P1_ONLY_REDS)
def test_a_p1_red_never_reverts_or_halts_p2(monkeypatch, red):
    v = _part_verdict(monkeypatch, "P2", P2_BLOCK, reds=_quiet_red(red))
    assert v["R2-c3"] == "PASS" and not v["halted"] and v["verdict"] == "PASS", v
    v = _part_verdict(monkeypatch, "P1", P1_BLOCK, reds=_quiet_red(red))
    assert v["R2-c3"] == "RED"


@pytest.mark.parametrize("red", P2_ONLY_REDS)
def test_a_p2_red_never_reverts_or_halts_p1(monkeypatch, red):
    v = _part_verdict(monkeypatch, "P1", P1_BLOCK, reds=_quiet_red(red))
    assert v["R2-c3"] == "PASS" and not v["halted"] and v["verdict"] == "PASS", v
    v = _part_verdict(monkeypatch, "P2", P2_BLOCK, reds=_quiet_red(red))
    assert v["R2-c3"] == "RED"


@pytest.mark.parametrize("kind", P1_ONLY_PLANTED)
def test_a_2_4_planted_red_is_p1s_only(monkeypatch, kind):
    entry = {"guard_on_ok": True, "guard_on_refused": False, "guard_off_flagged": True}
    planted = {
        k: {"guard_on_ok": True, "guard_on_refused": True, "guard_off_flagged": True}
        for k in rh.PLANTED
    }
    planted[kind] = entry
    assert _part_verdict(monkeypatch, "P2", P2_BLOCK, planted=planted)["R2-c3"] == "PASS"
    assert _part_verdict(monkeypatch, "P1", P1_BLOCK, planted=planted)["R2-c3"] == "RED"


def test_the_o_rng_lone_replay_control_is_p1s_only(monkeypatch):
    controls = {"O_rng_lone_replay_equal": False}
    assert _part_verdict(monkeypatch, "P2", P2_BLOCK, c3_controls=controls)["R2-c3"] == "PASS"
    assert _part_verdict(monkeypatch, "P1", P1_BLOCK, c3_controls=controls)["R2-c3"] == "RED"


@pytest.mark.parametrize("red", ["O_bout", "SLEEPDIGEST", "CTXREAD", "BOUTDIGEST"])
def test_a_red_no_line_assigns_still_joins_both_parts(monkeypatch, red):
    for part, block in (("P1", P1_BLOCK), ("P2", P2_BLOCK)):
        if red in rh.PLANTED:
            planted = {
                k: {"guard_on_ok": True, "guard_on_refused": True, "guard_off_flagged": k != red}
                for k in rh.PLANTED
            }
            v = _part_verdict(monkeypatch, part, block, planted=planted)
        else:
            v = _part_verdict(monkeypatch, part, block, reds=_quiet_red(red))
        assert v["R2-c3"] == "RED", (part, red)


# ---- ROUND 2, MAJOR-3 option (i): K3's "rest of the night, then 6" (2.2; R2-c6) ------------------
# The harness carries circadian phase across the restart; the brain does not (finding iii): a
# restoring process starts a fresh clock, so twin_K3 measures how many sleep ticks its cut night had
# left and every K3 restoring arm records exactly that many plus S=6.

K3_RESTORES = ("res_K3", "O_sleepskip", "SLEEPDIGEST_on", "SLEEPDIGEST_off")


class _Learner:
    def __init__(self):
        import random

        self.rng = random.Random(0)

    def weights(self):
        return [[0.0]]


class _NightBrain:
    """Mid-night: two sleep ticks left, a day of three wake ticks, then the next night."""

    day_period = 10

    def __init__(self):
        from types import SimpleNamespace

        self.script = ["asleep"] * 2 + ["awake"] * 3 + ["asleep"] * 9
        self._nights, self._was = 1, "asleep"
        self.brain = SimpleNamespace(_ixn=_Learner())

    def tick(self):
        state = self.script.pop(0)
        if state == "awake" and self._was == "asleep":
            self._nights += 1  # `_enter_wake` runs in the tick that wakes
        self._was = state
        return state


def test_twin_k3_records_the_rest_of_its_night_then_s_sleep_ticks():
    brain = _NightBrain()
    rows = rh._sleep_rows(brain, [], until_wake=True)
    assert len(rows) == 2 + rh.S_SLEEP and brain._nights == 2


def test_a_k3_restore_records_the_twins_measured_count_plus_s():
    brain = _NightBrain()
    brain._was = "awake"  # a fresh clock: the restore does not know it was cut mid-night
    assert len(rh._sleep_rows(brain, [], rh.S_SLEEP + 2)) == 2 + rh.S_SLEEP


@pytest.mark.parametrize("role", K3_RESTORES)
def test_a_k3_restoring_arm_refuses_to_run_without_the_twins_count(no_go, tmp_path, role):
    with pytest.raises(ValueError, match="rest of the night"):
        rh.run_process(role, 15000, str(tmp_path), str(tmp_path / "o.json"), nonce="n")


def test_the_proc_command_line_carries_the_k3_count(monkeypatch):
    seen = {}
    monkeypatch.setattr(rh, "run_process", lambda *a, **kw: seen.update(kw))
    rh.main(["--proc", "res_K3", "15000", "h", "o.json", "r2", "n", "4"])
    assert seen == {"nonce": "n", "k3_rest": 4}


@pytest.mark.parametrize("measured", [4, None])
def test_run_seed_hands_twin_k3s_measured_count_to_every_k3_restore(
    tmp_path, monkeypatch, measured
):
    argv = {}

    def child(args, **kwargs):
        argv[args[3]] = args
        if args[3] == "twin_K3" and measured is not None:
            rec = {"role": "twin_K3", "before": {}, "rows": [], "k3_rest": measured}
            Path(args[6]).write_text(json.dumps(rec), encoding="utf-8")
        return rh.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(rh.subprocess, "run", child)
    rh.run_seed(15000, tmp_path, roles=K3_RESTORES)
    assert "twin_K3" in argv, "a K3 restore's selection brings the twin it reads"
    for role in K3_RESTORES:
        want = [] if measured is None else [str(measured)]
        assert argv[role][9:] == want, (role, argv[role])
    assert argv["pre_K3"][9:] == [] and argv["twin_K3"][9:] == []


# ---- ROUND 3 (#6898): MINOR-2 K3 attribution, MINOR-3 roles by part ----------------------------


def _k3_rec(rows):
    return {"before": {}, "fault": None, "rows": [{"weights": [[w]]} for w in rows]}


def test_minor2_a_k3_difference_in_the_rest_of_the_night_is_attributed_to_the_rest():
    # twin_K3 records R=2 rest-of-night rows, then 6; the restore differs only in row 1 (the
    # rest-of-night half, where the restore has no wake/sleep transition the twin had)
    twin = _k3_rec([0, 0, 1, 1, 1, 1, 1, 1])
    res = _k3_rec([5, 0, 1, 1, 1, 1, 1, 1])
    assert rh._k3_split(res, twin, 2) == {"rest": False, "night": True}


def test_minor2_a_k3_difference_after_the_rest_is_attributed_to_the_six():
    twin = _k3_rec([0, 0, 1, 1, 1, 1, 1, 1])
    res = _k3_rec([0, 0, 1, 1, 1, 1, 1, 7])
    assert rh._k3_split(res, twin, 2) == {"rest": True, "night": False}


def test_minor2_a_k3_pair_that_agrees_everywhere_reads_true_on_both_halves():
    twin = _k3_rec([0, 0, 1, 1, 1, 1, 1, 1])
    assert rh._k3_split(copy.deepcopy(twin), twin, 2) == {"rest": True, "night": True}


def test_minor2_a_missing_k3_record_reads_false_on_both_halves():
    twin = _k3_rec([0, 0, 1, 1, 1, 1, 1, 1])
    assert rh._k3_split(None, twin, 2) == {"rest": False, "night": False}


def test_minor2_seed_comparisons_reports_the_k3_split_beside_c6():
    arms = _arms()
    arms["twin_K3"]["k3_rest"] = 1
    c = rh.seed_comparisons(arms)
    assert set(c["c6_k3_split"]) == {"rest", "night"}


def test_minor2_the_disclosure_names_the_transition_asymmetry():
    assert "no wake/sleep transition" in rh.__doc__


def _launched_roles(tmp_path, monkeypatch, seed, go):
    if go is None:
        monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    else:
        monkeypatch.setenv("R2_CONFIRMATORY_GO", go)
    launched = []

    def child(args, **kwargs):
        launched.append(args[3])
        return rh.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(rh.subprocess, "run", child)
    rh.run_seed(seed, tmp_path)
    return launched


@pytest.mark.parametrize(("seed", "go"), [(15010, "P1"), (15030, "P2")])
def test_minor3_a_p1_or_p2_read_runs_no_k1g_arm(tmp_path, monkeypatch, seed, go):
    launched = _launched_roles(tmp_path, monkeypatch, seed, go)
    assert launched == [r for r, _ in rh.R2_ROLES]


def test_minor3_a_p3_read_runs_only_the_k1g_arms(tmp_path, monkeypatch):
    launched = _launched_roles(tmp_path, monkeypatch, 15050, "P3")
    assert launched == [r for r, _ in rh.P3_ROLES]


def test_minor3_the_dev_block_still_runs_both_tables(tmp_path, monkeypatch):
    launched = _launched_roles(tmp_path, monkeypatch, 15000, None)
    assert launched == [r for r, _ in rh.R2_ROLES + rh.P3_ROLES]


# ---- ROUND 3 (#6898) MINOR-1: the instrument names R2-c7's binding reading -----------------------


def test_minor1_the_instrument_names_itself_the_binding_r2_c7_reading_with_its_definitions():
    doc = rh.__doc__
    assert "binding reading of R2-c7" in doc
    assert "ticks % refit_every" in doc  # its next-refit definition
    assert "kill_growth" in doc  # its kill point; the pytest's kill_prototypes is not in it


def test_minor1_the_k1g_kill_point_is_inside_the_flush_growth_call_not_the_tick_after():
    doc = rh.run_percept.__doc__
    assert "tick after" not in doc and "tick after" not in rh.__doc__
    assert "_flush_growth" in doc and "before `_flush_prototypes`" in doc


# ---- ROUND 4 (A, F): chrono_clean against the restored record, never against 0 -----------------
#: circadian-p4b round 2 restores `_step` across a restart: state()["circadian_restored"] is the
#: adopted record ({"step": ...}) or None, and a K3/K1g restore adopts the last sleep onset's step.


def _constructed(step, restored, **over):
    rec = _record(**over)
    rec["construct_chrono"]["step"] = step
    rec["construct_circadian_restored"] = restored
    return rec


def _parent(step):
    """What the parent pre_ process wrote: the step its last `_flush_circadian` call flushed."""
    return {"role": "pre_K0", "flushed_step": step}


def test_round4_a_clock_constructed_at_the_restored_step_is_clean():
    assert rh.chrono_clean(_constructed(6, {"step": 6}), _parent(6))


@pytest.mark.parametrize("step", [0, 5, 7])
def test_round4_a_clock_constructed_off_the_restored_step_is_unclean(step):
    assert not rh.chrono_clean(_constructed(step, {"step": 6}), _parent(6))


@pytest.mark.parametrize(("step", "clean"), [(0, True), (6, False)])
def test_round4_with_no_restored_record_the_clock_must_still_start_at_zero(step, clean):
    assert rh.chrono_clean(_constructed(step, None)) is clean


def test_round4_nights_must_still_be_zero_at_construction():
    rec = _constructed(6, {"step": 6})
    rec["construct_chrono"]["nights"] = 1
    assert not rh.chrono_clean(rec, _parent(6))


def test_round4_a_motor_call_or_a_replay_tick_is_unclean_on_a_restored_clock():
    assert not rh.chrono_clean(_constructed(6, {"step": 6}, construct_motor=1), _parent(6))
    ticked = _constructed(6, {"step": 6})
    ticked["replay_windows"][0]["after"]["step"] = 7
    assert not rh.chrono_clean(ticked, _parent(6))


# ---- ROUND 5 (#7103 MINOR-2): the step term meets the harness-side flush, not only the brain -----
#: construct_chrono.step and construct_circadian_restored.step are both read from the brain under
#: test, and `_load_circadian` sets them together, so a restore that adopted the WRONG record (the
#: stale phase of #6952 A-1) agrees with itself. The third operand is the step the parent pre_
#: process last flushed, recorded by the harness in the parent's own result.


def test_round5_a_restore_that_adopted_the_wrong_record_is_unclean_though_it_agrees_with_itself():
    assert rh.chrono_clean(_constructed(6, {"step": 6}), _parent(6))
    assert not rh.chrono_clean(_constructed(6, {"step": 6}), _parent(4)), "stale phase adopted"


@pytest.mark.parametrize(
    "parent", [None, {"role": "pre_K0"}, {"role": "pre_K0", "flushed_step": None}]
)
def test_round5_a_restored_record_with_no_parent_flush_is_unclean_not_waved_through(parent):
    assert not rh.chrono_clean(_constructed(6, {"step": 6}), parent)


def test_round5_with_no_restored_record_and_no_parent_flush_the_clock_starts_at_zero():
    assert rh.chrono_clean(_constructed(0, None), None)
    assert rh.chrono_clean(_constructed(0, None), {"role": "pre_K0"})


def test_round5_a_restore_that_dropped_the_record_its_parent_flushed_is_unclean():
    assert not rh.chrono_clean(_constructed(0, None), _parent(4)), "dropped, not 'no circadian'"


def _night_crossed(arms):
    """Every arm that records a construction clock restored a night-crossed record (step 6), and
    every parent pre_ process last flushed step 6."""
    for role, rec in arms.items():
        if "construct_chrono" in rec:
            rec["construct_chrono"]["step"] = 6
            rec["construct_circadian_restored"] = {"step": 6}
        elif role.startswith("pre_"):
            rec["flushed_step"] = 6
    return arms


def test_round4_clean_night_crossed_restores_read_clean_and_the_chrono_reds_stay_quiet():
    c = rh.seed_comparisons(_night_crossed(_arms()))
    assert all(c["chrono"].values()), c["chrono"]
    assert c["reds"]["CHRONO"] is False and c["reds"]["MOTOR"] is False


@pytest.mark.parametrize(
    "role",
    ["res_K0", "res_K1", "res_K2", "res_K3", "res_K4", "res_B", "res_Bpre", "res_Bpost", "res_rw"],
)
def test_round5_each_restoring_arm_is_unclean_when_its_own_parent_flushed_another_step(role):
    arms = _night_crossed(_arms())
    arms[dict(rh.R2_ROLES)[role]]["flushed_step"] = 4
    c = rh.seed_comparisons(arms)["chrono"]
    assert c[role] is False and sum(not v for v in c.values()) == 1, c


@pytest.mark.parametrize("red", ["CHRONO", "MOTOR"])
def test_round4_each_chrono_red_fires_only_when_its_defect_is_planted(red):
    arms = _night_crossed(_arms())
    assert rh.seed_comparisons(arms)["reds"][red] is False
    if red == "CHRONO":
        arms["CHRONO"]["construct_chrono"]["step"] = 7  # the hydration tick the red plants
    else:
        arms["MOTOR"]["construct_motor"] = 1
    c = rh.seed_comparisons(arms)
    assert c["reds"][red] is True
    assert all(v for r, v in c["chrono"].items()), "a planted red never dirties a clean arm"


def test_round4_the_restored_record_is_read_at_construction_before_the_chrono_red_ticks():
    src = inspect.getsource(rh.run_process)
    assert "construct_circadian_restored" in src
    assert src.index("construct_circadian_restored") < src.index('if role == "CHRONO"')


# ---- ROUND 5 (#7103 MAJOR-1, integrator #7111 (2) as amended by #7142): the K1g chronology -------
# ---- branches on the CODE UNDER TEST: whether the pre_ arm flushed a circadian record ------------


def _k1g_restored(arms, *, step=6, restored=6, flushed=6, nights=0):
    """A K1g restore on a head with the circadian fix: the parent pre_K1g last flushed `flushed`
    (harness-side), the restore constructed its clock at `step` from a record at `restored`."""
    arms["res_K1g"]["construct_chrono"] = {"step": step, "nights": nights}
    arms["res_K1g"]["construct_circadian_restored"] = {"step": restored}
    arms["pre_K1g"]["flushed_step"] = flushed
    return arms


def test_round5_a_head_without_circadian_is_refused_by_name_not_scored():
    arms = _k1g_arms()  # pre_K1g flushed no circadian record: the code under test has none
    arms["pre_K1g"]["flushed_step"] = None
    c = rh.c7_seed(arms)
    assert c["chrono"] == {"scored": False, "refused": rh.K1G_CHRONO_REFUSED}
    assert "flushed no circadian record" in rh.K1G_CHRONO_REFUSED
    assert "claude/circadian-p4b-1029" in rh.K1G_CHRONO_REFUSED  # NIT-1: the commit that lifts it
    assert "flush-first pending" not in rh.K1G_CHRONO_REFUSED
    assert rh.c7_seed(_k1g_arms())["chrono"] == c["chrono"], (
        "a parent that recorded no flush at all"
    )


def test_round5_a_refused_k1g_chronology_neither_passes_nor_fails_c7():
    arms = _k1g_arms()
    arms["res_K1g"]["construct_chrono"] = {"step": 6, "nights": 0}
    arms["twin_K1g"]["construct_chrono"] = {"step": 0, "nights": 0}
    c = rh.c7_seed(arms)
    assert c["ok"] and c["K1g"], c


def test_round5_the_refusal_is_not_keyed_on_a_restored_record_that_is_absent_or_null():
    for restored in (None, "absent"):  # the parent flushed none and the restore carries none
        arms = _k1g_arms()
        if restored is None:
            arms["res_K1g"]["construct_circadian_restored"] = None
        assert rh.c7_seed(arms)["chrono"] == {"scored": False, "refused": rh.K1G_CHRONO_REFUSED}


def test_round5_a_k1g_restore_that_carries_the_flushed_record_is_scored_and_clean():
    c = rh.c7_seed(_k1g_restored(_k1g_arms()))
    assert c["chrono"] == {"scored": True, "clean": True}
    assert c["ok"] and c["K1g"], c


def test_round5_a_k1g_restore_that_dropped_the_flushed_record_is_a_chrono_fail_not_refused():
    arms = _k1g_restored(_k1g_arms())
    arms["res_K1g"]["construct_circadian_restored"] = (
        None  # pre_ flushed step 6; the restore lost it
    )
    arms["res_K1g"]["construct_chrono"]["step"] = 0
    c = rh.c7_seed(arms)
    assert c["chrono"] == {"scored": True, "clean": False}
    assert c["K1g"] is True and c["ok"] is False
    del arms["res_K1g"]["construct_circadian_restored"]  # or never reported one
    assert rh.c7_seed(arms)["chrono"] == {"scored": True, "clean": False}


@pytest.mark.parametrize(
    "over",
    [
        {"step": 7},  # the construction ticked, or the clock was not the record's
        {"step": 0, "restored": 0},  # a stale phase: it agrees with itself, not with the parent
        {"flushed": 4},  # the parent flushed another step than the record adopted
        {"nights": 1},
    ],
)
def test_round5_a_k1g_restore_whose_clock_is_off_the_flushed_record_is_scored_unclean(over):
    c = rh.c7_seed(_k1g_restored(_k1g_arms(), **over))
    assert c["chrono"] == {"scored": True, "clean": False}
    assert c["K1g"] is True and c["ok"] is False, "the bar's other half still reads; ok needs both"


def test_round5_the_percept_process_records_the_clock_the_chronology_read_needs():
    src = inspect.getsource(rh.run_percept)
    assert "construct_chrono" in src and "construct_circadian_restored" in src
    assert "_track_clock" in src, "pre_K1g0 and pre_K1g record the step they last flushed"
    assert src.index("_track_clock") < src.index("make_brain(")


# ---- ROUND 5 (#7111 (1), #7103 MINOR-1): K3_phase is EQUAL to twin_K3's own onset step -----------
#: tick order at circadian@1d87efd4: `_step += 1` (:3934), then `_enter_sleep` (:3939), whose
#: persist block flushes the clock first (:3822-3824). twin_K3 records the step of the onset of the
#: night it is cut in; the restore adopts exactly that record. The fixture's onset is step 8 and
#: its cut step 10: J_NIGHT - 1 = 2 ticks later.


def test_round5_the_k3_restore_resumes_the_phase_of_the_cut_night():
    assert rh.seed_comparisons(_arms())["c6"]["K3_phase"] is True


@pytest.mark.parametrize("restored", [None, {"step": 0}, {"step": 7}, {"step": 9}, {"step": 10}])
def test_round5_a_k3_restore_off_the_onset_step_by_any_gap_is_a_red(restored):
    arms = _arms()  # onset 8; 7 and 9 are the off-by-one gaps, 10 is the cut (gap J_NIGHT)
    arms["res_K3"]["construct_circadian_restored"] = restored
    assert rh.seed_comparisons(arms)["c6"]["K3_phase"] is False


@pytest.mark.parametrize("onset", [None, "8"])
def test_round5_a_twin_k3_with_no_integer_onset_step_reads_false(onset):
    arms = _arms()
    arms["twin_K3"]["onset_step"] = onset
    assert rh.seed_comparisons(arms)["c6"]["K3_phase"] is False


def test_round5_twin_k3_records_the_onset_step_of_the_night_it_is_cut_in():
    src = inspect.getsource(rh.run_process)
    assert "onset_step" in src and src.index("onset_step") < src.index("_sleep_rows(")
    assert "cut_step" not in src, "the onset replaces the cut step: the cut is not the record"
    assert "_track_clock" in src and src.index("_track_clock") < src.index("make_brain(")
    assert "cut_step" not in inspect.getsource(rh._k3_phase)


def test_round5_the_k3_phase_docstring_cites_the_tick_order_not_a_window():
    doc = " ".join(rh._k3_phase.__doc__.split())
    assert "_enter_sleep" in doc and "0..J_NIGHT" not in doc


# ---- ROUND 5 (#7103 NIT-2): c6 reads NOT MET on every seed until circadian stacks ---------------


def test_round5_the_c6_docstring_says_a_red_k3_phase_is_not_an_instrument_fault():
    doc = " ".join(rh.seed_comparisons.__doc__.split())
    assert "K3_phase" in doc and "D-79" in doc and "circadian" in doc
    assert "NOT MET" in doc


# ---- ROUND 6 (#7183 (C), integrator second amendment to #7111 (2)): the FOUR-cell K1g table ----
#: (pre_ flushed a circadian record?, restore carries one?): (yes, yes) scored; (yes, no) FAIL;
#: (no, yes) FAIL by name 'a restored record with no flushed lineage' (an isolation defect: a stale
#: or shared home, or the live lake); (no, no) refused by name. One red per FAIL cell.


def test_round6_a_restored_record_with_no_flushed_lineage_is_a_named_chrono_fail():
    arms = _k1g_arms()  # pre_K1g flushed none, yet the restore carries a record (the new cell)
    arms["pre_K1g"]["flushed_step"] = None
    arms["res_K1g"]["construct_circadian_restored"] = {"step": 6}
    c = rh.c7_seed(arms)
    assert c["chrono"] == {"scored": True, "clean": False, "fail": rh.K1G_CHRONO_NO_LINEAGE}
    assert "restored record with no flushed lineage" in rh.K1G_CHRONO_NO_LINEAGE
    assert c["K1g"] is True and c["ok"] is False, "the bar's other half reads; ok needs the chrono"


def test_round6_the_no_lineage_fail_is_not_a_refusal_and_needs_no_clock_match():
    arms = _k1g_arms()
    arms["pre_K1g"]["flushed_step"] = None
    arms["res_K1g"]["construct_circadian_restored"] = {"step": 0}
    arms["res_K1g"]["construct_chrono"] = {"step": 0, "nights": 0}  # a self-consistent stale record
    c = rh.c7_seed(arms)["chrono"]
    assert c["scored"] is True and c["clean"] is False and "refused" not in c


def test_round6_the_four_cells_are_four_distinct_readings():
    reads = {}
    for flushed in (None, 6):
        for restored in (None, {"step": 6}):
            arms = _k1g_restored(_k1g_arms()) if flushed else _k1g_arms()
            arms["pre_K1g"]["flushed_step"] = flushed
            arms["res_K1g"]["construct_circadian_restored"] = restored
            if flushed is None and restored is None:
                arms["res_K1g"]["construct_chrono"] = {"step": 0, "nights": 0}
            reads[(flushed, restored is not None)] = rh.c7_seed(arms)["chrono"]
    assert reads[(6, True)] == {"scored": True, "clean": True}
    assert reads[(6, False)] == {"scored": True, "clean": False}
    assert reads[(None, True)]["fail"] == rh.K1G_CHRONO_NO_LINEAGE
    assert reads[(None, False)] == {"scored": False, "refused": rh.K1G_CHRONO_REFUSED}


# ---- ROUND 6 (#7183 (C) NIT, closes #7103 NIT-2): K3_phase keyed on the same bit ---------------


def test_round6_k3_phase_on_a_head_without_circadian_is_refused_by_name_never_not_met():
    arms = _arms()
    arms["pre_K3"]["flushed_step"] = None  # the code under test has no clock to resume
    arms["res_K3"]["construct_circadian_restored"] = None
    c = rh.seed_comparisons(arms)
    assert "K3_phase" not in c["c6"], "not-applicable is not a False conjunct of c6's all()"
    assert c["k3_phase_refused"] == rh.K3_PHASE_REFUSED
    assert "pre_K3 flushed no circadian record" in rh.K3_PHASE_REFUSED
    assert all(c["c6"].values()), "a refused K3_phase leaves c6 to its own comparisons"


def test_round6_a_refused_k3_phase_does_not_hide_a_real_c6_red():
    arms = _arms()
    arms["pre_K3"]["flushed_step"] = None
    arms["res_K3"]["rows"][0]["weights"][0][0] += 1.0
    c = rh.seed_comparisons(arms)["c6"]
    assert c["K3"] is False and not all(c.values())


def test_round6_k3_phase_is_scored_and_not_refused_when_pre_k3_flushed():
    c = rh.seed_comparisons(_arms())  # pre_K3 flushed step 8
    assert c["c6"]["K3_phase"] is True and "k3_phase_refused" not in c
    arms = _arms()
    arms["res_K3"]["construct_circadian_restored"] = None  # pre_ flushed, the restore dropped it
    c = rh.seed_comparisons(arms)
    assert c["c6"]["K3_phase"] is False and "k3_phase_refused" not in c


def test_round6_the_c6_docstring_names_the_refusal_and_not_applicable():
    doc = " ".join(rh.seed_comparisons.__doc__.split())
    assert "not-applicable" in doc and "K3_PHASE_REFUSED" in doc
    assert "reads NOT MET on every seed" not in doc, "retired: only a head WITH circadian is scored"


# ---- ROUND 4 (B, #6987 MINOR-3 residual): a P3 campaign launches no base role -------------------


def _campaign_calls(tmp_path, monkeypatch, seeds, go):
    if go is None:
        monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    else:
        monkeypatch.setenv("R2_CONFIRMATORY_GO", go)
    heads = []
    monkeypatch.setattr(rh, "_head_of", lambda path: "head")
    monkeypatch.setattr(rh, "_check_base", lambda src: "base-head")
    monkeypatch.setattr(rh, "run_seed", lambda seed, out, head="r2", **kw: heads.append(head) or {})
    monkeypatch.setattr(rh, "seed_comparisons", lambda arms, base=None: {})
    monkeypatch.setattr(rh, "verdict", lambda per_seed, seeds: {"verdict": "x"})
    rh.run_campaign(tmp_path, seeds, base_src="base")
    return heads, json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))


def test_round4_a_p3_campaign_launches_no_base_role(tmp_path, monkeypatch):
    heads, v = _campaign_calls(tmp_path, monkeypatch, (15050, 15051), "P3")
    assert heads == ["r2", "r2"]
    assert v["provenance"]["base_src"] is None, "no base head ran, and the provenance says so"


@pytest.mark.parametrize(("seeds", "go"), [((15000, 15001), None), ((15010,), "P1")])
def test_round4_other_campaigns_still_run_the_base_head_per_seed(tmp_path, monkeypatch, seeds, go):
    heads, v = _campaign_calls(tmp_path, monkeypatch, seeds, go)
    assert heads == ["r2", "base"] * len(seeds)
    assert v["provenance"]["base_src"] == "base-head"


# ---- ROUND 4 (C, #6987 NIT): the K3 halves are not independent -------------------------------


def test_round4_the_k3_split_says_the_halves_are_not_independent():
    doc = " ".join(rh._k3_split.__doc__.split())
    assert "not independent" in doc
    assert "'night' False attributes to the six rows only when 'rest' is True" in doc


# ---- ROUND 4 (E, #6991): one kill point, inside the _flush_growth call, at brain.py:3798-3800 ---


def test_round4_no_text_in_the_instrument_still_names_the_tick_after_as_the_kill_point():
    text = Path(rh.__file__).read_text(encoding="utf-8")
    assert "tick after" not in text
    assert "3798-3800" in text and "3797-3799" not in text


# ---- ROUND 4 (D, #6926/#6987 MINOR-1): one K1g procedure, the instrument's ---------------------


def test_round4_the_pytest_holds_no_second_k1g_procedure():
    import test_r2_p3_percept_transitions as p3t

    assert not hasattr(p3t, "_restart_bar")
    for retired in ("kill_prototypes", "kill_growth", "to_next_refit", "TWIN", "BOUNDARY"):
        assert retired not in p3t._CHILD, retired


def test_round4_the_pytest_restart_bar_runs_the_instruments_roles():
    import test_r2_p3_percept_transitions as p3t

    src = inspect.getsource(
        p3t.test_restart_hydrated_buffer_and_next_refit_equal_the_uninterrupted_twin
    )
    assert "run_seed" in src and "percept_equal" in src
    assert (
        "mode"
        not in inspect.signature(
            p3t.test_restart_hydrated_buffer_and_next_refit_equal_the_uninterrupted_twin
        ).parameters
    )


def test_round4_the_instrument_says_it_is_the_only_r2_c7_reading():
    doc = " ".join(rh.__doc__.split())
    assert "by a second procedure" not in doc
    assert "holds no second copy" in doc


# ---- ROUND 7 (#7204 MINOR-1, #7210 MINOR-1): 'pre_ flushed' means WROTE, not CALLED --------------
#: At circadian@1abbbcff `_flush_circadian` returns False and writes nothing when it raises or the
#: keep fails (engine/brain.py:5394-5403). A failed call must not move `flushed_step`; it goes to
#: `flush_failed_steps` and reads as its own named cause, never as 'the restore dropped the record'.


def _fake_brain(monkeypatch):
    """A stand-in Brain with the two methods `_track_clock` wraps; no brain, no torch. Its flush
    returns the instance's `ok`."""

    class Brain:
        ok = True
        _step = 0

        def _flush_circadian(self, *a, **kw):
            return self.ok

        def _enter_sleep(self, *a, **kw):
            return "slept"

    monkeypatch.setitem(sys.modules, "snn_data_lake.engine.brain", SimpleNamespace(Brain=Brain))
    return Brain()


def test_round7_a_flush_that_returns_false_leaves_flushed_step_at_the_last_truthy_step(
    monkeypatch,
):
    brain, result = _fake_brain(monkeypatch), {}
    rh._track_clock("pre_K0", result)
    brain._step = 4
    assert brain._flush_circadian() is True and result["flushed_step"] == 4
    brain.ok, brain._step = False, 6
    assert brain._flush_circadian() is False
    assert result["flushed_step"] == 4, "CALLED at step 6 is not WROTE"
    assert result["flush_failed_steps"] == [6]


def test_round7_a_pre_arm_whose_every_flush_failed_wrote_none(monkeypatch):
    brain, result = _fake_brain(monkeypatch), {}
    brain.ok = False
    rh._track_clock("pre_K0", result)
    for step in (3, 5):
        brain._step = step
        assert brain._flush_circadian() is False
    assert result["flushed_step"] is None and result["flush_failed_steps"] == [3, 5]


def test_round7_the_wrapper_passes_the_flush_result_through_and_notes_the_step(monkeypatch):
    brain, result = _fake_brain(monkeypatch), {}
    rh._track_clock("pre_K1g", result)
    brain._step = 8
    assert brain._flush_circadian() is True
    assert result["flushed_step"] == 8 and not result.get("flush_failed_steps")


def test_round7_a_role_that_is_not_pre_notes_no_flush_and_twin_k3_notes_its_onset(monkeypatch):
    brain, result = _fake_brain(monkeypatch), {}
    seen = rh._track_clock("twin_K3", result)
    brain._step = 10
    assert brain._flush_circadian() is True and "flushed_step" not in result
    assert brain._enter_sleep() == "slept" and seen["onset"] == 10


@pytest.mark.parametrize("restored", [None, {"step": 6}])
def test_round7_a_failed_flush_on_pre_k1g_is_its_own_named_chrono_fail(restored):
    arms = _k1g_arms()
    arms["pre_K1g"]["flushed_step"] = None
    arms["pre_K1g"]["flush_failed_steps"] = [6]
    arms["res_K1g"]["construct_circadian_restored"] = restored
    c = rh.c7_seed(arms)
    assert c["chrono"] == {"scored": True, "clean": False, "fail": rh.K1G_CHRONO_FLUSH_FAILED}
    assert c["chrono"]["fail"] != rh.K1G_CHRONO_NO_LINEAGE
    assert c["K1g"] is True and c["ok"] is False
    assert "returned False" in rh.K1G_CHRONO_FLUSH_FAILED
    assert "dropped" not in rh.K1G_CHRONO_FLUSH_FAILED


def test_round7_a_failed_flush_after_a_written_one_still_scores_against_the_written_step():
    arms = _k1g_restored(_k1g_arms())  # pre_K1g WROTE step 6; the restore carries 6
    arms["pre_K1g"]["flush_failed_steps"] = [9]
    assert rh.c7_seed(arms)["chrono"] == {"scored": True, "clean": True}


def test_round7_a_failed_flush_on_pre_k3_is_a_false_k3_phase_with_its_own_name():
    arms = _arms()
    arms["pre_K3"]["flushed_step"] = None
    arms["pre_K3"]["flush_failed_steps"] = [8]
    c = rh.seed_comparisons(arms)
    assert c["c6"]["K3_phase"] is False and "k3_phase_refused" not in c
    assert c["k3_phase_flush_failed"] == rh.K3_PHASE_FLUSH_FAILED
    assert "returned False" in rh.K3_PHASE_FLUSH_FAILED


def test_round7_the_docstrings_say_flushed_means_wrote():
    for fn in (rh._track_clock, rh.c7_seed, rh.seed_comparisons):
        assert "WROTE" in " ".join(fn.__doc__.split()), fn.__name__


# ---- ROUND 7 (#7210 item 4): the round-5 red, FLIPPED not deleted --------------------------------


def test_round5_the_branch_is_not_keyed_on_the_restored_record():
    """Kept under its round-5 name (a deleted red is a P8 defect). Despite 'not keyed' it now
    asserts the keyed FAIL: the parent flushed none and the restore reports a record, so the
    chronology reads FAIL (NO_LINEAGE), the branch keyed on the code under test (#7255 NIT-2)."""
    arms = _k1g_arms()  # the parent flushed none, yet the restore reports a record
    arms["res_K1g"]["construct_circadian_restored"] = {"step": 6}
    c = rh.c7_seed(arms)["chrono"]
    assert c == {"scored": True, "clean": False, "fail": rh.K1G_CHRONO_NO_LINEAGE}


# ---- ROUND 7 (#7210 NIT): the K3_phase refusal does not hide the isolation defect --------------


def test_round7_a_restored_record_with_a_non_int_parent_flush_is_still_unclean_for_res_k3():
    assert not rh.chrono_clean(_constructed(6, {"step": 6}), _parent(None))


def test_round7_the_k3_phase_docstring_says_chrono_clean_still_catches_the_isolation_defect():
    doc = " ".join(rh._k3_phase.__doc__.split())
    assert "chrono_clean" in doc and "res_K3" in doc and "isolation" in doc


# ---- ROUND 8 (#7255 MINOR-1, NIT-1): the failed flush for EVERY restoring arm; 'unwritten' ------
#: MINOR-1: a parent whose every `_flush_circadian` call returned False after a failed write wrote
#: no record, so a restoring arm that constructs a fresh clock must not read clean as 'a head
#: without circadian'. NIT-1: the same False return also means persist off or an unchanged record
#: (circadian brain.py:5419-5420, :5436), told apart by `_circadian_flush_failed`, which only a
#: failure bumps.
_RESTORING_ROLES = (
    "res_K0",
    "res_K1",
    "res_K2",
    "res_K3",
    "res_K4",
    "res_B",
    "res_Bpre",
    "res_Bpost",
    "res_rw",
)


def _fresh_restore(arms, role, *, parent_failed):
    """`role` restored no record and constructed a fresh clock; its parent either wrote nothing
    after a failed flush (`parent_failed`) or never flushed (a head without circadian)."""
    parent = arms[dict(rh.R2_ROLES)[role]]
    parent["flushed_step"] = None
    parent["flush_failed_steps"] = [3] if parent_failed else []
    arms[role]["construct_chrono"] = {"step": 0, "nights": 0}
    arms[role]["construct_circadian_restored"] = None
    return arms


@pytest.mark.parametrize("role", _RESTORING_ROLES)
def test_round8_a_parent_whose_every_flush_failed_is_its_own_named_chrono_fail(role):
    arms = _fresh_restore(_arms(), role, parent_failed=True)
    c = rh.seed_comparisons(arms)
    assert c["chrono"][role] is False, "a failed parent flush passed as 'a head without circadian'"
    assert c["chrono_flush_failed"] == {role: rh.CHRONO_FLUSH_FAILED}


@pytest.mark.parametrize("role", _RESTORING_ROLES)
def test_round8_a_parent_that_never_flushed_still_reads_clean_and_is_not_named(role):
    arms = _fresh_restore(_arms(), role, parent_failed=False)
    c = rh.seed_comparisons(arms)
    assert c["chrono"][role] is True and "chrono_flush_failed" not in c


def test_round8_a_failed_flush_after_a_written_one_does_not_name_the_arm():
    arms = _fresh_restore(_arms(), "res_K0", parent_failed=True)
    arms["pre_K0"]["flushed_step"] = 4  # it also WROTE step 4: the restore must carry it
    c = rh.seed_comparisons(arms)
    assert c["chrono"]["res_K0"] is False and "chrono_flush_failed" not in c


def test_round8_chrono_flush_failed_names_only_a_raised_write_or_a_failed_keep():
    text = " ".join(rh.CHRONO_FLUSH_FAILED.split())
    assert "raised" in text and "keep" in text and "persist" not in text
    assert "dropped" not in text and "returned False" in text
    assert "none wrote" in text and "every time" not in text
    assert "none wrote" in " ".join(rh._parent_flush_failed.__doc__.split())


def test_round8_the_clock_clean_docstring_names_the_failed_flush():
    assert "flush_failed_steps" in " ".join(rh._clock_clean.__doc__.split())


class _CountingBrain:
    """Fake Brain whose flush mimics circadian brain.py:5412-5439: False with the failure counter
    bumped (a raised write or a failed keep), False without it (persist off, unchanged record)."""

    _step = 0
    _circadian_flush_failed = 0
    mode = "wrote"

    def _flush_circadian(self, *a, **kw):
        if self.mode == "failed":
            self._circadian_flush_failed += 1
        return self.mode == "wrote"

    def _enter_sleep(self, *a, **kw):
        return "slept"


def _counting_brain(monkeypatch, mode):
    monkeypatch.setitem(
        sys.modules, "snn_data_lake.engine.brain", SimpleNamespace(Brain=_CountingBrain)
    )
    brain = _CountingBrain()
    brain.mode = mode
    return brain


@pytest.mark.parametrize("mode", ["unchanged", "persist_off"])
def test_round8_a_false_return_without_the_failure_counter_is_unwritten_not_failed(
    monkeypatch, mode
):
    brain, result = _counting_brain(monkeypatch, mode), {}
    rh._track_clock("pre_K0", result)
    brain._step = 5
    assert brain._flush_circadian() is False
    assert result["flushed_step"] is None
    assert result["flush_unwritten_steps"] == [5] and not result["flush_failed_steps"]


def test_round8_a_false_return_that_bumped_the_failure_counter_is_failed(monkeypatch):
    brain, result = _counting_brain(monkeypatch, "failed"), {}
    rh._track_clock("pre_K0", result)
    brain._step = 5
    assert brain._flush_circadian() is False
    assert result["flush_failed_steps"] == [5] and not result["flush_unwritten_steps"]


def test_round8_a_head_without_the_failure_counter_still_reads_a_false_return_as_failed(
    monkeypatch,
):
    brain, result = _fake_brain(monkeypatch), {}  # round 7's stand-in: no counter at all
    brain.ok = False
    rh._track_clock("pre_K0", result)
    brain._step = 5
    assert brain._flush_circadian() is False
    assert result["flush_failed_steps"] == [5] and not result["flush_unwritten_steps"]


@pytest.mark.parametrize("restored", [None, {"step": 6}])
def test_round8_unwritten_pre_k1g_is_not_the_failed_write_cell(restored):
    arms = _k1g_arms()
    arms["pre_K1g"]["flushed_step"] = None
    arms["pre_K1g"]["flush_failed_steps"] = []
    arms["pre_K1g"]["flush_unwritten_steps"] = [6]
    arms["res_K1g"]["construct_circadian_restored"] = restored
    chrono = rh.c7_seed(arms)["chrono"]
    assert chrono.get("fail") != rh.K1G_CHRONO_FLUSH_FAILED
    if restored is None:
        assert chrono == {"scored": False, "refused": rh.K1G_CHRONO_UNWRITTEN}
    else:
        assert chrono == {"scored": True, "clean": False, "fail": rh.K1G_CHRONO_NO_LINEAGE}


def test_round8_the_texts_say_which_cause_they_name():
    failed = " ".join(rh.K1G_CHRONO_FLUSH_FAILED.split())
    assert "raised" in failed and "keep" in failed and "persist" not in failed
    assert "persist" in rh.K1G_CHRONO_UNWRITTEN and "unchanged" in rh.K1G_CHRONO_UNWRITTEN
    assert "raised" not in rh.K1G_CHRONO_UNWRITTEN
    assert "persist" in rh.K3_PHASE_UNWRITTEN and "unchanged" in rh.K3_PHASE_UNWRITTEN
    assert "persist" not in rh.K3_PHASE_FLUSH_FAILED


def test_round8_unwritten_pre_k3_is_refused_by_name_not_the_failed_write_fail():
    arms = _arms()
    arms["pre_K3"]["flushed_step"] = None
    arms["pre_K3"]["flush_failed_steps"] = []
    arms["pre_K3"]["flush_unwritten_steps"] = [8]
    arms["res_K3"]["construct_circadian_restored"] = None
    c = rh.seed_comparisons(arms)
    assert "K3_phase" not in c["c6"] and "k3_phase_flush_failed" not in c
    assert c["k3_phase_refused"] == rh.K3_PHASE_UNWRITTEN


# ---- ROUND 8 (#7255 NIT-2): the flipped round-5 red says what it now asserts ---------------------


def test_round8_the_flipped_round5_red_documents_the_keyed_fail():
    fn = test_round5_the_branch_is_not_keyed_on_the_restored_record
    doc = " ".join((fn.__doc__ or "").split())
    assert "keyed FAIL" in doc and "not keyed" in doc
