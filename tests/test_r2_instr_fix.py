"""The R2 instrument review's fixes (#6735, lanes/engine-1024/r2-instr-review-row/REPORT.md), each
red first: the licence on every entry point (MAJOR-1), one part per block and per-part verdicts
(MAJOR-2), the reader-kind off arms (MAJOR-4), the bout restarts' chronology (MAJOR-6) and the code
MINORs. Synthetic records and patched children only; no brain is built and no confirmatory or
reserve seed is ever run."""

import json

import pytest
from test_replay_hydration_instrument import REDS, _arms, _differ, _fw_rec, _per_seed

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
        ("P3", [15050]),  # P3 has no campaign instrument (BLOCK-1): its seeds stay unlicensed
        ("P3", [15060]),
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
