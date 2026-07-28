"""
charge_probe -- Policy 3's SECOND TIER, the one the module gate cannot reach.

`substrate/dormancy_gate.py` proves a module is IMPORTED. It does not prove the module's output changes
behaviour, and docs/POLICIES.md is explicit that those are different claims: "reachability proves a
module is imported, NOT that its output changes behaviour". The difference is settled by neutering the
signal on a live brain and watching what changes. That is what this does.

WHY IT LIVES HERE. POLICIES.md cited `scratchpad/smoke_drive_charge.py` as the instrument keeping P3
honest -- and `scratchpad/` is gitignored, so a fresh clone had NO charge probe at all. The entire
second tier of the project's headline policy was untracked local files. Three defects are fixed by
moving it into the tree:

  1. it is now version-controlled, so the evidence standard travels with the repo;
  2. the repo root is derived from __file__ instead of a hardcoded "D:/AI-project/..." absolute path;
  3. it constructs through `cortex.brain.make_brain()`, the Policy-4 sanctioned construction path. The
     original called `Brain(...)` directly -- the instrument that enforced P3 was quietly violating P4.

METHOD. Two runs at the same seed. A = wired. B = the named drive source neutered every tick, so any
behavioural divergence is attributable to that signal alone. The observables are BEHAVIOURAL, not
telemetry: `arm_stopped` (the act/hold decision) and `map_position` (the resulting trajectory).

VERDICT. "BEHAVIOURAL CHARGE OK" (exit 0) if neutering changed a decision or a trajectory on any tick;
"CONSUMED-BUT-NOT-DECISIVE" (exit 1) otherwise. Exit-code contract preserved from the original so
existing callers keep working.

HONESTY. A source whose precondition never fires reports NOT-ELIGIBLE rather than a charge verdict --
"the wire is dead" and "the wire never got a chance to fire" are different findings, and reporting the
second as the first is how a wiring wave talks itself into a number.

  PYTHONPATH=src python tools/charge_probe.py [source] [ticks]
  PYTHONPATH=src python tools/charge_probe.py drive 500
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cortex.brain import make_brain  # noqa: E402

# drive source -> the learned per-(key, action) value table to clear alongside its weight. The
# undirected sources (novelty, ignition, timing, reroute, surprise) have no table; zeroing the weight
# is the whole neuter for them.
_VALUE_TABLE = {
    "drive": "_drive_value",
    "place": "_place_value",
    "percept": "_percept_value",
    "recall": "_recall_value",
}


def run(source, n, neuter):
    b = make_brain(seed=0, persist=False)  # Policy 4: the ONE sanctioned construction path
    stopped, pos = [], []
    for _ in range(n):
        b.tick()
        st = b.state()
        stopped.append(bool(st.get("arm_stopped")))
        pos.append(tuple(st.get("map_position") or ()))
        if neuter:
            tbl = _VALUE_TABLE.get(source)
            if tbl is not None:
                getattr(b, tbl).clear()
            b._drive_w[source] = 0.0
    return b, stopped, pos


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "drive"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    probe = make_brain(seed=0, persist=False)
    known = sorted(probe._drive_w)
    probe.close()
    if source not in known:
        print(f"unknown drive source {source!r}; known sources: {', '.join(known)}")
        return 2

    bA, sA, pA = run(source, n, False)
    bB, sB, pB = run(source, n, True)

    w = round(bA._drive_w[source], 4)
    tbl = _VALUE_TABLE.get(source)
    pairs = len(getattr(bA, tbl)) if tbl else None
    act_flips = sum(1 for x, y in zip(sA, sB) if x != y)
    pos_flips = sum(1 for x, y in zip(pA, pB) if x != y)
    bA.close()
    bB.close()

    print(f"source                                        : {source}")
    if pairs is not None:
        print(f"({source}, action) values learned from dopamine : {pairs} pairs")
    print(f"source weight (0.1 uniform init -> ...)       : {w}  (moved={w != 0.1})")
    print(f"arm HOLD rate  A(wired) vs B(neutered)        : {sum(sA) / n:.3f} vs {sum(sB) / n:.3f}")
    print(f"act/hold decision changed by the signal       : {act_flips}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {pos_flips}/{n} ticks")

    # ELIGIBILITY, before any verdict, and the first version of this check was WRONG in a way worth
    # recording. It treated a moved weight as evidence the source had activated. It is not: the
    # three-factor update in brain.py:1523-1528 is COMPETITIVE, `dw = lr * da * (e - mean_e)`, so it
    # moves EVERY key's weight on EVERY tick -- a source whose eligibility is permanently 0 still gets
    # pushed steadily DOWN by the -mean_e term. `place` demonstrated it: weight 0.1 -> 0.0373 over 500
    # ticks while its value table stayed EMPTY and it changed nothing. A varying number that proves
    # nothing is the mirror image of the constant-wearing-a-signal's-clothes defect, and it is easier
    # to miss because movement reads as life.
    #
    # So eligibility keys on the LEARNED VALUE TABLE, which only fills when the source actually fired.
    # Undirected sources have no table, so their eligibility is genuinely UNKNOWN from outside and is
    # reported as such rather than assumed either way.
    if tbl is not None and not pairs:
        print(
            "NOT-ELIGIBLE: the source never activated in this run (its learned value table is empty, "
            "and the weight moving only reflects the competitive -mean_e term) -- no charge verdict"
        )
        return 2
    if tbl is None:
        print("NOTE: undirected source, no value table -- eligibility cannot be established from here")

    if act_flips > 0 or pos_flips > 0:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
