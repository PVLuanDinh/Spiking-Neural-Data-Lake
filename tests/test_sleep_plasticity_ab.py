"""R3 / T104 P1 -- the sleep-plasticity arms' instrument (instruments/sleep_plasticity_ab.py).

Fast tests are pure: the statistic wrapper (rulings 9 / 9a, the enumerated sign-flip rate), K11's
table reachability, K12's seed rule, K14's decoy red, the census, and one constructed red per
record-level check on a synthetic record that is green before the plant. One integration test
drives two real arm-seeds (ON, OFF; 2 sleep ticks; a red-block seed) through the two-process path
in a scratch home.
"""

import copy
import os
from fractions import Fraction

import pytest

from snn_data_lake.instruments import sleep_plasticity_ab as m

DEV = m.SEED_SETS["dev"]


def test_offline_selfcheck():
    m.main()


def test_verdict_superiority_degenerate_is_not_met_and_fpr_is_enumerated():
    v = m.verdict("c", [Fraction(1, 20)] * 8, DEV)
    assert v.degenerate and not v.met
    assert v.fpr == 8 / 256  # the eight 7+/1- flips give t = 3.0 under the actual rule (Codex 21)
    assert v.ref_fpr == 0.0  # the incumbent's shortcut, labelled reference


def test_verdict_ni_degenerate_reads_by_common_value():
    assert m.verdict("ni", [m.EPS_A] * 8, DEV, kind="ni").met  # a lossless arm is WITHIN BOUND (9a)
    assert not m.verdict("ni", [0] * 8, DEV, kind="ni").met  # one A item lost on every seed -> RED
    assert not m.verdict("ni", [-m.EPS_A] * 8, DEV, kind="ni").met


def test_verdict_nondegenerate_matches_paired_contrast():
    d = [0.2, 0.3, 0.25, 0.4, 0.15, 0.35, 0.3, 0.2]
    v = m.verdict("c", d, DEV)
    ref = m.contrast("c", d)
    assert (v.met, v.t, v.fpr) == (ref.met, ref.t, ref.fpr)
    assert not m.verdict("noise", [0.1, -0.1, 0.12, -0.11, 0.09, -0.1, 0.1, -0.12], DEV).met


@pytest.mark.parametrize("diffs,seeds", [([0.1] * 7, DEV[:7]), ([0.1] * 8, (300,) + DEV[1:])])
def test_k12_refuses_partial_lists_and_foreign_seeds(diffs, seeds):
    with pytest.raises(ValueError):
        m.verdict("red", diffs, seeds)


def test_k11_every_row_reached_and_the_inventory_is_complete():
    ok, misses = m.table_reach_ok()
    assert ok, misses


def test_k11_red_a_row_without_input(monkeypatch):
    short = dict(m.TABLE_REACH)
    short["F3"] = short["F3"][:-1]
    monkeypatch.setattr(m, "TABLE_REACH", short)
    assert not m.table_reach_ok()[0]


def test_k14_decoy_red_fires(tmp_path):
    f = tmp_path / "brain_lake.bin"
    f.write_bytes(b"x")
    before = m.lake_snapshot(str(tmp_path))
    assert m.live_lake_untouched(str(tmp_path), before)[0]
    f.write_bytes(b"xy")
    assert not m.live_lake_untouched(str(tmp_path), before)[0]


def test_scratch_home_refusals(monkeypatch):
    monkeypatch.setenv("NORD_HOME", m.LIVE_LAKE)
    monkeypatch.setenv("NORD_PERSIST", "1")
    with pytest.raises(RuntimeError):
        m.require_scratch_home()
    monkeypatch.delenv("NORD_HOME")
    with pytest.raises(RuntimeError):
        m.require_scratch_home()


def test_seed_sets_inside_the_block_and_disjoint():
    seen = set()
    for name, seeds in m.SEED_SETS.items():
        assert len(seeds) == 8 and all(m.SEED_BLOCK[0] <= s <= m.SEED_BLOCK[1] for s in seeds), name
        assert not seen & set(seeds), name
        seen |= set(seeds)
    assert not seen & set(m.RED_SEEDS)


def test_census_counts_the_tests_read():
    read = [
        m.verdict("C1", [0.1, 0.2] * 4, DEV),
        m.verdict("C1-NI", [m.EPS_A] * 8, DEV, kind="ni"),
        m.verdict("C-SENS", [0.1, 0.2] * 4, DEV, kind="ctl"),
    ]
    lines = m.census(read)
    assert lines[-1].startswith("tests read in all: 3")
    assert "C1-NI=MET (degenerate)" in lines[2]


def _record(arm="ON", seed=16000, family="F1"):
    """A synthetic record that every record-level check reads green."""
    return {
        "family": family, "arm": arm, "seed": seed, "budget": 2, "d0": "d0", "d1": "d1",
        "weights_d1": "w", "weights_post": "w", "taught": ["t1"], "ring_keys": ["t1"],
        "heldout_keys": ["h1"], "drawn": ["t1"],
        "tick": {"presentations": [m.K, m.K], "turnover": [None, None], "cacc": [None, None]},
        "att_calls": [[1, 3], [2, 3]], "k8a": [], "stored_any": False,
        "chrono_before": {"nights": 0, "wake_ticks": 0, "arm_ticks": 0},
        "chrono_after": {"nights": 1, "wake_ticks": 0, "arm_ticks": 0},
        "inproc": [1, 0], "restored": [1, 0], "n_a": 1, "eval_before": "e", "eval_after": "e",
        "live_lake_untouched": True,
    }  # fmt: skip


PLANTS = {
    "K1": lambda r: r.update(d0="other"),
    "K2": lambda r: r["tick"].update(presentations=[m.K - 1, m.K]),
    "K3": lambda r: r.update(drawn=["h1"]),
    "K4": lambda r: r.update(eval_after="moved"),
    "K5": lambda r: r.update(restored=[0, 0]),
    "K6": lambda r: r.update(weights_post="moved"),
    "K8a": lambda r: r.update(k8a=[{"before": {}, "after": {}}]),
    "K8b": lambda r: r["chrono_after"].update(nights=2),
    "K14": lambda r: r.update(live_lake_untouched=False),
}


@pytest.mark.parametrize("check", sorted(PLANTS))
def test_record_check_reds(check):
    arm = "OFF" if check == "K6" else "ON"
    base = [_record(arm), _record("ON-TWIN") if check != "K6" else _record("ON")]
    assert not any(m.check_records("F1", base).values()), m.check_records("F1", base)
    planted = copy.deepcopy(base)
    PLANTS[check](planted[0])
    assert m.check_records("F1", planted)[check], check


def test_k13_red_unequal_attractor_draws():
    a, b = _record("AT-ON", family="F3"), _record("AT-OFF", family="F3")
    assert not m.check_records("F3", [a, b])["K13"]
    b["att_calls"] = [[1, 2], [2, 2]]
    assert m.check_records("F3", [a, b])["K13"]


def test_f3_ladder_rule_and_the_unfrozen_refusal():
    below, at = [34] * 40, [35] * 40  # per-tick correct counts of 36
    assert m.ladder_qualifies([below] * 8)
    assert not m.ladder_qualifies([below] * 7 + [at])  # one seed at 35/36 fails the rung
    assert not m.ladder_qualifies([below] * 7)  # a partial seed list never qualifies
    assert m.F3_W_LADDER == (1, 2, 3, 6, 12)


def test_two_real_arm_seeds_restore_exactly(tmp_path):
    """ON and OFF on a red-block seed, 2 sleep ticks: every record check green, restore exact, the
    ledger k per tick, OFF's weights untouched by its bout, and the live lake untouched."""
    out = str(tmp_path)
    seed = m.RED_SEEDS[5]
    recs = [m.run_arm(out, "F1", arm, seed, budget=2) for arm in ("ON", "OFF")]
    fails = m.check_records("F1", recs)
    assert not any(fails.values()), fails
    on, off = recs
    assert on["inproc"] == on["restored"] and off["inproc"] == off["restored"]
    assert on["tick"]["presentations"] == off["tick"]["presentations"] == [m.K, m.K]
    assert sum(on["tick"]["deposits"]) > 0 and sum(off["tick"]["deposits"]) == 0
    assert on["d0"] == off["d0"] and on["d1"] == off["d1"]
    assert off["weights_d1"] == off["weights_post"] and on["weights_d1"] != on["weights_post"]
    assert all(r["live_lake_untouched"] for r in recs)
    assert not os.path.exists(
        os.path.join(out, "homes", f"F1_ON_{seed}_n2")
    )  # scratch homes removed
