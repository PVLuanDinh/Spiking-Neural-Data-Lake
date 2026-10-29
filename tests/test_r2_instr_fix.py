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
