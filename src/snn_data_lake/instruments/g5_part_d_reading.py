"""G5 / T113 Part D D1's reader (specs/022 evidence/g5-readout-variants.md D.4 and the dated notes (c), (e), (f)).

Pure functions over per-seed held-out accuracies: records[seed][cell][arm] -> float. No Brain, no pool, no seed
block runs here; the cell runner that fills the records is a later slice. The reads are the registered reds:

  PD-R0   S0C0: RSTDP_D - RSTDP_NOTUNE MET (opens the read; NOT MET halts it, nothing else is printed)
  PD-R1   S0C1: RSTDP_BRAIN - RSTDP_NOTUNE MET and RSTDP_BRAIN - RSTDP_BRAIN_SHUF MET ("outcome-driven scalar
          tuning"); PRODNEUTER is the same learner as NOTUNE (note (f)), counted once, equality asserted
  PD-R1b  S0C1: RSTDP_BRAIN - RSTDP_BYPASS MET ("the nucleus adds")
  PD-R2   S0C1: RSTDP_D (CARRY) beats EXPANSION_H and RSTDP_D@neuter_carry, read only with the certified expansion
          (else None = NOT READ); S1C0/S1C1: RSTDP_D - RSTDP_D@reset MET, and a stateless arm solving an S1 cell
          HALTS (hidden state; G2 row 15)
  PD-R3   per rule R in DELTA / PC / SOM: R - RSTDP_D MET and its own control (R_CTRL - RSTDP_D) NOT MET -> REPLACES,
          else REVERT (ties keep the incumbent)

The statistic is g5_readout_cells.reading (record 2.4): paired_contrast at T_BAR, >= 5 non-tied, degenerate NOT MET.
Every contrast is built from two cells of the same seed, over exactly N_SEEDS seeds; anything else HALTS the instrument
(the archived Part B reader accepted fewer seeds as "all hold": Codex g5_b M2).
"""

from snn_data_lake.instruments.g5_readout_cells import _line, reading

N_SEEDS = 8
CELLS = ("S0C0", "S0C1", "S1C0", "S1C1", "AII")
RULES = ("DELTA", "PC", "SOM")
ALL_ARMS = (
    "RSTDP_D",
    "RSTDP_NOTUNE",
    "RSTDP_BRAIN",
    "RSTDP_BRAIN_PRODNEUTER",
    "RSTDP_BRAIN_SHUF",
    "RSTDP_BYPASS",
    "RSTDP_D@reset",
    "RSTDP_D@neuter_carry",
    "RSTDP_D@neuter_reset",
    "EXPANSION_H",
    *RULES,
    *(f"{r}_CTRL" for r in RULES),
)
EQ_TOL = 1e-12


class ReadHalt(RuntimeError):
    """The instrument, not the hypothesis, is broken: the read prints no verdict."""


def _diffs(records, cell, a, b):
    seeds = sorted(records)
    if len(seeds) != N_SEEDS:
        raise ReadHalt(f"{len(seeds)} seeds, the registered cohort is {N_SEEDS}")
    out = []
    for s in seeds:
        try:
            out.append(records[s][cell][a] - records[s][cell][b])
        except KeyError as e:
            raise ReadHalt(f"seed {s}: cell {cell} is missing arm {e}") from e
    return out


def _met(records, cell, a, b, log):
    ok, c, x = reading(f"{cell}: {a} - {b}", _diffs(records, cell, a, b))
    log(_line(f"{cell}: {a} - {b}", ok, c, x))
    return ok


def read_d1(records, cert, log=print):
    """The D1 verdict dict. `cert`: expansion_solves_s0c1_at_64 (G2's solved at the certified H, read beside the
    expansion arm) and stateless_solved_s1 (a neuter_reset / expansion arm solved a PLAIN S1 cell: HALT)."""
    if cert.get("stateless_solved_s1"):
        raise ReadHalt(
            "a stateless arm solved an exact-bound S1 cell: hidden state or a broken instrument (G2 row 15)"
        )
    v = {"halted": False}
    v["PD-R0"] = _met(records, "S0C0", "RSTDP_D", "RSTDP_NOTUNE", log)
    if not v["PD-R0"]:
        log(
            "PD-R0 NOT MET: the D1 read HALTS (the incumbent's teaching does not beat the untuned neurons)"
        )
        return {"halted": True, "PD-R0": False}
    gap = max(abs(d) for d in _diffs(records, "S0C1", "RSTDP_BRAIN_PRODNEUTER", "RSTDP_NOTUNE"))
    if gap > EQ_TOL:
        raise ReadHalt(
            f"PRODNEUTER != NOTUNE (max gap {gap:.3g}): a deposit with d = 0 moved a weight"
        )
    v["prodneuter_equals_notune"] = True
    v["PD-R1"] = _met(records, "S0C1", "RSTDP_BRAIN", "RSTDP_NOTUNE", log) & _met(
        records, "S0C1", "RSTDP_BRAIN", "RSTDP_BRAIN_SHUF", log
    )
    v["PD-R1b"] = _met(records, "S0C1", "RSTDP_BRAIN", "RSTDP_BYPASS", log)
    r2 = {}
    if cert.get("expansion_solves_s0c1_at_64"):
        r2["S0C1"] = _met(records, "S0C1", "RSTDP_D", "EXPANSION_H", log) & _met(
            records, "S0C1", "RSTDP_D", "RSTDP_D@neuter_carry", log
        )
    else:
        r2["S0C1"] = None
        log("PD-R2 S0C1 NOT READ: the expansion did not solve S0C1 at the certified H = 64")
    for cell in ("S1C0", "S1C1"):
        r2[cell] = _met(records, cell, "RSTDP_D", "RSTDP_D@reset", log)
    v["PD-R2"] = r2
    v["PD-R3"] = {}
    for rule in RULES:
        beats = _met(records, "S0C1", rule, "RSTDP_D", log)
        ctrl = _met(records, "S0C1", f"{rule}_CTRL", "RSTDP_D", log)
        v["PD-R3"][rule] = "REPLACES" if beats and not ctrl else "REVERT"
    return v


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the reader's constructed reds (pure records; no Brain, no pool, no seed block)
# --------------------------------------------------------------------------------------------------
_FIXTURE_SEEDS = range(20261003, 20261011)  # SEEDS.md's RNG fixture row; the pinned test's cohort
_UP = (0.30, 0.25, 0.35, 0.28, 0.32, 0.27, 0.31, 0.26)  # eight clearly positive paired gaps


def _records(gaps, base=0.4):
    """records[seed][cell][arm] over the eight fixture seeds: every arm at `base` unless `gaps` raises
    it by the seed's _UP scale -- the pinned test's fixture, so the self-check and the test agree."""
    out = {}
    for k, s in enumerate(_FIXTURE_SEEDS):
        out[s] = {cell: {arm: base + 0.01 * (k % 3) for arm in ALL_ARMS} for cell in CELLS}
        for (cell, arm), scale in gaps.items():
            out[s][cell][arm] += _UP[k] * scale
    return out


def _clean_gaps():
    """A clean fixture: RSTDP_D opens S0C0 and the S1 cells, the Brain producer beats NOTUNE / SHUF /
    BYPASS on S0C1, and DELTA beats the incumbent while its own control does not."""
    g = {("S0C0", "RSTDP_D"): 1.0, ("S1C0", "RSTDP_D"): 1.0, ("S1C1", "RSTDP_D"): 1.0}
    g.update({("S0C1", a): 1.0 for a in ("RSTDP_BRAIN", "RSTDP_D")})
    g.update({("S0C1", "DELTA"): 2.0, ("S0C1", "RSTDP_BRAIN_SHUF"): 0.0})
    return g


def self_check():
    """The reader's constructed reds at gate cost: a clean fixture reads PD-R0/R1/R1b/R2/R3 MET; a
    NOT-MET PD-R0 HALTS the read and prints no other verdict; a cohort that is not N_SEEDS, a missing
    arm, PRODNEUTER != NOTUNE and a stateless S1 solve each HALT the instrument; PD-R2 S0C1 is NOT
    READ without the certified expansion; a rule its own control also matches REVERTS; one records
    object reads the same twice. The fuller pin of every red is tests/test_g5_part_d_reading.py.
    Returns True; raises on the first failure."""
    cert = {"expansion_solves_s0c1_at_64": True, "stateless_solved_s1": False}
    quiet = []

    def _halt(records, header):
        try:
            read_d1(records, cert, log=quiet.append)
        except ReadHalt:
            return
        raise AssertionError(f"{header} did not halt the instrument")

    clean = _records(_clean_gaps())
    verdict = read_d1(_records(_clean_gaps()), cert, log=quiet.append)
    assert verdict["PD-R0"] and verdict["PD-R1"] and verdict["PD-R1b"], verdict
    assert verdict["PD-R2"] == {"S0C1": True, "S1C0": True, "S1C1": True}, verdict
    assert verdict["PD-R3"]["DELTA"] == "REPLACES", verdict
    assert verdict["prodneuter_equals_notune"] is True, verdict
    assert read_d1(_records({}), cert, log=quiet.append) == {"halted": True, "PD-R0": False}
    short = _records(_clean_gaps())
    short.pop(_FIXTURE_SEEDS[0])
    _halt(short, "a cohort of seven seeds")
    no_arm = _records(_clean_gaps())
    del no_arm[_FIXTURE_SEEDS[3]]["S0C1"]["RSTDP_BRAIN"]
    _halt(no_arm, "a missing S0C1 arm")
    no_rule = _records(_clean_gaps())
    for seed in no_rule:
        del no_rule[seed]["S0C1"][RULES[0]]
    _halt(no_rule, "a missing entire PD-R3 rule")
    prod = _records(_clean_gaps())
    for seed in prod:
        prod[seed]["S0C1"]["RSTDP_BRAIN_PRODNEUTER"] = prod[seed]["S0C1"]["RSTDP_NOTUNE"]
    prod[_FIXTURE_SEEDS[0]]["S0C1"]["RSTDP_BRAIN_PRODNEUTER"] += 0.1
    _halt(prod, "PRODNEUTER != NOTUNE")
    try:
        read_d1(_records(_clean_gaps()), {**cert, "stateless_solved_s1": True}, log=quiet.append)
    except ReadHalt:
        pass
    else:
        raise AssertionError("a stateless S1 solve did not halt the instrument")
    blind = read_d1(
        _records(_clean_gaps()), {**cert, "expansion_solves_s0c1_at_64": False}, log=quiet.append
    )
    assert blind["PD-R2"]["S0C1"] is None, blind
    gaps = _clean_gaps()
    gaps[("S0C1", "DELTA_CTRL")] = 2.0
    assert read_d1(_records(gaps), cert, log=quiet.append)["PD-R3"]["DELTA"] == "REVERT"
    once = read_d1(clean, dict(cert), log=quiet.append)
    twice = read_d1(clean, dict(cert), log=quiet.append)
    assert once == twice, once
    print(
        "g5_part_d_reading self-check OK: a clean fixture reads PD-R0/R1/R1b/R2/R3 MET; PD-R0 NOT MET "
        "halts and prints no other verdict; seven seeds, a missing arm, PRODNEUTER != NOTUNE and a "
        "stateless S1 solve each HALT the instrument; PD-R2 S0C1 is NOT READ without the certified "
        "expansion; a rule its own control also matches REVERTS; one records object reads the same twice"
    )
    return True


def main(argv=None):
    # carried_state_driver's pattern: no arguments runs the self-check (the reader has no fixture
    # smoke CLI -- it is pure functions over records, so `argv` is accepted and ignored)
    self_check()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
