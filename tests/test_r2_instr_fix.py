"""The R2 instrument review's fixes (#6735, lanes/engine-1024/r2-instr-review-row/REPORT.md), each
red first: the confirmatory-seed licence on every entry point (MAJOR-1). Synthetic records and
patched children only; no brain is built and no confirmatory or reserve seed is ever run."""

import pytest
from test_replay_hydration_instrument import _per_seed

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
