"""G5 Part D D1's reader (evidence/g5-readout-variants.md D.4, notes (c), (e), (f)): constructed
per-seed accuracy records through every read, each red able to fire. No Brain, no pool, no seed
block: the seeds in the records are fixture integers."""

import pytest

from snn_data_lake.instruments import g5_part_d_reading as r

SEEDS = tuple(range(20261003, 20261011))
UP = (0.30, 0.25, 0.35, 0.28, 0.32, 0.27, 0.31, 0.26)  # eight clearly positive paired gaps


def _rec(gaps, base=0.4):
    """records[seed][cell][arm]: every arm sits at `base` unless `gaps` raises it, per (cell, arm)
    -> UP scale. RSTDP_D is the incumbent on CARRY features; other feature sources are suffixed."""
    out = {}
    for k, s in enumerate(SEEDS):
        out[s] = {}
        for cell in r.CELLS:
            out[s][cell] = {arm: base + 0.01 * (k % 3) for arm in r.ALL_ARMS}
        for (cell, arm), scale in gaps.items():
            out[s][cell][arm] += UP[k] * scale
    return out


def _good():
    g = {("S0C0", "RSTDP_D"): 1.0}
    g.update({("S0C1", a): 1.0 for a in ("RSTDP_BRAIN", "RSTDP_D")})
    g.update({("S0C1", "DELTA"): 2.0, ("S0C1", "RSTDP_BRAIN_SHUF"): 0.0})
    g.update({("S1C0", "RSTDP_D"): 1.0, ("S1C1", "RSTDP_D"): 1.0})
    return g


CERT = {"expansion_solves_s0c1_at_64": True, "stateless_solved_s1": False}


def test_a_clean_fixture_reads_every_bar_met():
    v = r.read_d1(_rec(_good()), CERT, log=lambda *_: None)
    assert not v["halted"]
    assert v["PD-R0"] and v["PD-R1"] and v["PD-R1b"]
    assert v["PD-R2"]["S0C1"] is True and v["PD-R2"]["S1C0"] is True and v["PD-R2"]["S1C1"] is True
    assert v["PD-R3"]["DELTA"] == "REPLACES"


def test_pd_r0_not_met_halts_the_read_and_prints_no_other_verdict():
    gaps = _good()
    del gaps[("S0C0", "RSTDP_D")]
    v = r.read_d1(_rec(gaps), CERT, log=lambda *_: None)
    assert v["halted"] and v["PD-R0"] is False and "PD-R1" not in v


def test_fewer_than_eight_seeds_or_a_missing_arm_halts_the_instrument():
    rec = _rec(_good())
    rec.pop(SEEDS[0])
    with pytest.raises(r.ReadHalt):
        r.read_d1(rec, CERT, log=lambda *_: None)
    rec = _rec(_good())
    del rec[SEEDS[3]]["S0C1"]["RSTDP_BRAIN"]
    with pytest.raises(r.ReadHalt):
        r.read_d1(rec, CERT, log=lambda *_: None)


def test_pd_r1_needs_the_shuffle_contrast_and_pd_r1b_needs_the_bypass_contrast():
    gaps = _good()
    gaps[("S0C1", "RSTDP_BRAIN_SHUF")] = 1.0  # the shuffled producer does as well: PD-R1 must fail
    v = r.read_d1(_rec(gaps), CERT, log=lambda *_: None)
    assert v["PD-R1"] is False
    gaps = _good()
    gaps[("S0C1", "RSTDP_BYPASS")] = (
        1.0  # the bare outcome does as well as the nucleus: PD-R1b must fail
    )
    v = r.read_d1(_rec(gaps), CERT, log=lambda *_: None)
    assert v["PD-R1"] is True and v["PD-R1b"] is False


def test_prodneuter_is_counted_once_with_notune_and_its_inequality_is_an_instrument_halt():
    rec = _rec(_good())
    for s in SEEDS:
        rec[s]["S0C1"]["RSTDP_BRAIN_PRODNEUTER"] = rec[s]["S0C1"]["RSTDP_NOTUNE"]
    v = r.read_d1(rec, CERT, log=lambda *_: None)
    assert v["prodneuter_equals_notune"] is True
    rec[SEEDS[0]]["S0C1"]["RSTDP_BRAIN_PRODNEUTER"] += 0.1  # a deposit with d = 0 moved a weight
    with pytest.raises(r.ReadHalt):
        r.read_d1(rec, CERT, log=lambda *_: None)


def test_pd_r2_is_not_read_without_the_certified_expansion_and_halts_on_a_stateless_s1_solve():
    v = r.read_d1(
        _rec(_good()), {**CERT, "expansion_solves_s0c1_at_64": False}, log=lambda *_: None
    )
    assert v["PD-R2"]["S0C1"] is None  # NOT READ, not red
    with pytest.raises(r.ReadHalt):  # HALT (hidden state; G2 row 15)
        r.read_d1(_rec(_good()), {**CERT, "stateless_solved_s1": True}, log=lambda *_: None)


def test_pd_r2_s0c1_needs_carry_to_beat_both_the_expansion_and_neuter_carry():
    gaps = _good()
    gaps[("S0C1", "EXPANSION_H")] = 1.0  # the expansion does as well as carry
    v = r.read_d1(_rec(gaps), CERT, log=lambda *_: None)
    assert v["PD-R2"]["S0C1"] is False


def test_pd_r3_reverts_a_rule_whose_own_control_also_beats_the_incumbent():
    gaps = _good()
    gaps[("S0C1", "DELTA_CTRL")] = 2.0
    v = r.read_d1(_rec(gaps), CERT, log=lambda *_: None)
    assert v["PD-R3"]["DELTA"] == "REVERT"
    gaps = _good()
    del gaps[("S0C1", "DELTA")]  # the rule does not beat the incumbent: ties keep the incumbent
    assert r.read_d1(_rec(gaps), CERT, log=lambda *_: None)["PD-R3"]["DELTA"] == "REVERT"
