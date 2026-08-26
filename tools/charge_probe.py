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


class _PinnedWeights(dict):
    """A `_drive_w` whose neutered key CANNOT rise. This is the airtight neuter, and the naive
    version it replaces was measurably leaky.

    Zeroing the weight after each `b.tick()` is not enough. `brain.py:1523-1528` applies the
    competitive update `dw = lr * da * (e - mean_e)` to every key INSIDE the tick, and `:1539-1550`
    consumes the weight afterwards -- so a key zeroed at the end of tick N is re-incremented partway
    through tick N+1 and is non-zero by the time the drive is computed. Measured on `recall`: over 60
    ticks the post-tick weight was non-zero on **12 of them**, up to 0.0033 against a 0.1 nominal.

    That leak biases arm B toward arm A, which pushes verdicts toward CONSUMED-BUT-NOT-DECISIVE --
    the direction that makes a live faculty look dead and invites retiring it. A partial neuter is
    worse than no neuter, because it still returns a confident answer.
    """

    def __init__(self, src, pinned):
        super().__init__(src)
        self._pinned = pinned

    def __setitem__(self, k, v):
        super().__setitem__(k, 0.0 if k == self._pinned else v)


def _numeric_reads(st, prefix="", out=None, depth=0):
    """Every numeric leaf of a snapshot, flattened to `section.key`.

    These are the values the brain itself later reads. They exist here because the two flip counts
    below are ARGMAX tests, and an argmax over ELEVEN competing drive sources almost never moves when
    one is removed -- so a source that genuinely shifts the decision variable scored 0/60 and read as
    inert. SpatialTemporal.md measured that directly: neutering all eleven sources gives 47/60
    trajectory flips, any single one gives 0/60. The pathway is decisive; no single voice is, because
    ten others carry the sum. A graded observable separates "changes nothing" from "changes something
    the brain reads, just not enough to flip the winner" -- which is exactly the rubric's 41-60 band
    ("changes an internal read") sitting underneath its 61-80 band ("changes a decision")."""
    out = {} if out is None else out
    if depth > 2 or not isinstance(st, dict):
        return out
    for k, v in st.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            _numeric_reads(v, key + ".", out, depth + 1)
    return out


# The rule-grain sibling of this probe lives in plasticity/rule_composer.rule_charge
# (T051b-iii): the same neuter-A/B pattern applied to composed learning rules on a
# caller-built trajectory, consuming the composer's per-rule attribution. One pattern,
# two grains; neither re-implements the other.
def run(source, n, neuter):
    b = make_brain(seed=0, persist=False)  # Policy 4: the ONE sanctioned construction path
    tbl = _VALUE_TABLE.get(source)
    if neuter:
        b._drive_w = _PinnedWeights(b._drive_w, source)
        b._drive_w[source] = 0.0
    stopped, pos, reads = [], [], []
    for _ in range(n):
        b.tick()
        st = b.state()
        stopped.append(bool(st.get("arm_stopped")))
        pos.append(tuple(st.get("map_position") or ()))
        reads.append(_numeric_reads(st))
        if neuter and tbl is not None:
            getattr(b, tbl).clear()
    return b, stopped, pos, reads


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "drive"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    probe = make_brain(seed=0, persist=False)
    known = sorted(probe._drive_w)
    probe.close()
    if source not in known:
        print(f"unknown drive source {source!r}; known sources: {', '.join(known)}")
        return 2

    bA, sA, pA, rA = run(source, n, False)
    bB, sB, pB, rB = run(source, n, True)

    w = round(bA._drive_w[source], 4)
    tbl = _VALUE_TABLE.get(source)
    pairs = len(getattr(bA, tbl)) if tbl else None
    act_flips = sum(1 for x, y in zip(sA, sB, strict=True) if x != y)
    pos_flips = sum(1 for x, y in zip(pA, pB, strict=True) if x != y)
    read_ticks, totals = 0, {}
    for a, c in zip(rA, rB, strict=True):
        moved = {k: abs(a[k] - c[k]) for k in set(a) & set(c) if a[k] != c[k]}
        if moved:
            read_ticks += 1
        for k, v in moved.items():
            totals[k] = totals.get(k, 0.0) + v
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:5]
    bA.close()
    bB.close()

    print(f"source                                        : {source}")
    if pairs is not None:
        print(f"({source}, action) values learned from dopamine : {pairs} pairs")
    print(f"source weight (0.1 uniform init -> ...)       : {w}  (moved={w != 0.1})")
    print(f"arm HOLD rate  A(wired) vs B(neutered)        : {sum(sA) / n:.3f} vs {sum(sB) / n:.3f}")
    print(f"act/hold decision changed by the signal       : {act_flips}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {pos_flips}/{n} ticks")
    # Deliberately "N of M", NOT "N/M". coverage_score's `_fingerprint` harvests every bare `N/M` in a
    # probe's stdout, so printing this as a ratio appended a THIRD observable to every artifact's
    # fingerprint and made all of them fail `--rerun` -- an instrument change invalidating the evidence
    # it was built to check. The measurement itself did not move: the two flip counts reproduce
    # byte-identically either way, which is what proves this line is additive and not behavioural.
    print(f"internal reads changed by the signal          : {read_ticks} of {n} ticks")
    for k, v in top:
        print(f"    {k:<40} total |delta| {v:.6g}")

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
    if act_flips > 0 or pos_flips > 0:
        print("BEHAVIOURAL CHARGE OK")
        return 0

    # THE MASKED NEGATIVE, and it is the reason this tier exists. Both flip counts above are ARGMAX
    # tests over eleven competing drive sources, so removing one voice almost never moves the winner:
    # SpatialTemporal.md measured all-eleven neuter at 47/60 trajectory flips and every single-source
    # neuter at 0/60. Reporting that as CONSUMED-BUT-NOT-DECISIVE and nothing else invited reading it
    # as inertness, which is the verdict that gets a live faculty retired.
    #
    # A non-zero read delta is a POSITIVE result, and it is sound for undirected sources too for the
    # same reason the flip verdict is: divergence proves the signal mattered, whatever its eligibility.
    # It is deliberately NOT exit 0. Exit 0 is the >40 gate in coverage_score (F8), and this evidence
    # supports the rubric's 41-60 band ("changes an internal read") while the 61-80 band ("changes a
    # decision, action or learning update") still requires a flip. Returning 0 here would let weaker
    # evidence satisfy a stronger claim, which is the inflation this whole instrument exists to stop.
    # ORDERING IS LOAD-BEARING, and the first version got it wrong. This branch sat ABOVE the
    # undirected-source UNKNOWN branch below, so a source whose eligibility this probe CANNOT
    # determine stopped returning 2 (UNKNOWN) and started returning 1 -- converting "I cannot ask
    # this question" into a confident verdict. That is the exact inversion the constitution's third
    # standing invariant forbids, and it silently changed the recorded exit code of every UNKNOWN
    # artifact in the ledger: st_timing_60 went `recorded exit 2 / now 1` and failed --rerun while its
    # observables were byte-identical. Caught by an artifact refusing to reproduce, not by review.
    # A graded read is only a verdict for a source whose eligibility is KNOWN.
    if read_ticks > 0 and tbl is not None:
        print(
            f"INTERNAL-READ CHANGED on {read_ticks} of {n} ticks: the signal is CONSUMED and moves values "
            f"the brain later reads, but never flips the act/hold winner or the trajectory. Rubric "
            f"41-60 evidence ('changes an internal read'), NOT 61-80 ('changes a decision')."
        )
        return 1

    # An undirected source (novelty / ignition / timing / reroute / surprise) has no value table, so
    # there is no way from out here to tell "fired and did not matter" from "never fired". A NEGATIVE
    # verdict on those would be a confident answer to a question this probe cannot ask -- and the
    # negative verdict is the dangerous one, because it is the one that gets a faculty retired. So
    # they return UNKNOWN (exit 2), not CONSUMED-BUT-NOT-DECISIVE. A POSITIVE verdict above is still
    # sound for them: divergence proves the signal mattered regardless of eligibility.
    if tbl is None:
        print(
            "UNKNOWN: undirected source with no value table -- no divergence observed, but eligibility "
            "cannot be established from outside, so this is NOT evidence the source is inert"
        )
        return 2
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
