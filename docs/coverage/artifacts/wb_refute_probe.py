"""wb_refute_probe -- adversarial re-measurement of WholeBrain's one band-41+ claim.

`docs/coverage/WholeBrain.md` scores `homeostatic_drive_to_action` at band 65 on
`wb_drive_60.txt`, which is `tools/charge_probe.py drive 60`: exit 0, "act/hold decision changed
0/60 ticks", "arm trajectory changed 22/60 ticks". That artifact reproduces. This probe asks the
three questions it does not answer, using the SAME neuter -- `charge_probe._PinnedWeights` is
imported rather than copied, so the experiment is identical by construction and any difference in
the numbers is mine, not the harness's.

  1. SELF-CONTROL. `charge_probe` runs A and B and no second A. Nothing in the recorded artifact
     rules out 22/60 being run-to-run nondeterminism rather than the neuter. A vs A is the control
     that costs one extra run and settles it. (`docs/coverage/artifacts/affect_ab_probe.py` has
     one; the tracked probe does not.)

  2. DIVERGENCE STRUCTURE. "22/60 ticks" counts ticks on which the observable DIFFERS, not
     independent divergences. A trajectory that splits once at tick 38 and never re-converges
     yields 60-38 = 22 differing ticks from ONE event. `docs/coverage/Executive.md:113-116`
     measured exactly that for `taskset` and asserts `drive` behaves the same way; this re-derives
     it for `drive` directly. Reported as first-divergence tick, equal->differ ONSETS, and
     differ->equal re-convergences.

  3. THE SELECTED ACTION. `charge_probe`'s observables are `arm_stopped` and `map_position`
     (`brain.py:2785` -- the allocentric cognitive map, dead-reckoned from the real end-effector
     stream at `autonomous_loop.py:1595`). The three-way ENGAGE / RELEASE / HOLD choice is
     `_last_res["action"]` and the probe never looks at it. Rubric band 61-80 is "the action /
     value / weight measurably differs", so whether the choice moved is the load-bearing question.

VERDICT / exit code
  0  MULTI-EVENT: more than one divergence onset, or the selected action changed -- a repeated
     behavioural divergence, which is what an effect size looks like.
  1  SINGLE-SPLIT: one onset, no re-convergence, no action change -- the 22/60 is one event and
     cannot carry a mid-range band claim.
  2  HARNESS NONDETERMINISTIC: A vs A diverges, so no A-vs-B count from this harness means
     anything.

  PYTHONPATH=src python docs/coverage/artifacts/wb_refute_probe.py [source] [ticks]
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import charge_probe  # noqa: E402  -- the SAME neuter and the SAME construction path, imported


def _run(source, n, neuter):
    b = charge_probe.make_brain(seed=0, persist=False)  # Policy 4 path, via charge_probe
    tbl = charge_probe._VALUE_TABLE.get(source)
    if neuter:
        b._drive_w = charge_probe._PinnedWeights(b._drive_w, source)
        b._drive_w[source] = 0.0
    stopped, pos, act = [], [], []
    for _ in range(n):
        b.tick()
        st = b.state()
        stopped.append(bool(st.get("arm_stopped")))
        pos.append(tuple(st.get("map_position") or ()))
        act.append((getattr(b, "_last_res", None) or {}).get("action"))
        if neuter and tbl is not None:
            getattr(b, tbl).clear()
    b.close()
    return stopped, pos, act


def _structure(xa, xb):
    """(differing ticks, first differing tick 1-based, equal->differ onsets, returns to equal)."""
    d = [a != b for a, b in zip(xa, xb, strict=True)]
    first = d.index(True) + 1 if any(d) else None
    onsets = sum(1 for i, x in enumerate(d) if x and not (i and d[i - 1]))
    back = sum(1 for i, x in enumerate(d) if i and d[i - 1] and not x)
    return sum(d), first, onsets, back


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "drive"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    sA, pA, aA = _run(source, n, False)
    sA2, pA2, aA2 = _run(source, n, False)  # the control charge_probe does not run
    sB, pB, aB = _run(source, n, True)

    ctl = (
        sum(1 for x, y in zip(sA, sA2, strict=True) if x != y),
        sum(1 for x, y in zip(pA, pA2, strict=True) if x != y),
        sum(1 for x, y in zip(aA, aA2, strict=True) if x != y),
    )
    s_flips = sum(1 for x, y in zip(sA, sB, strict=True) if x != y)
    p_flips, p_first, p_ons, p_back = _structure(pA, pB)
    a_flips, a_first, a_ons, _ = _structure(aA, aB)

    print(f"source                                        : {source}")
    print(f"ticks per arm                                 : {n}")
    print(
        f"SELF-CONTROL A vs A: stop / traj / action     : "
        f"{ctl[0]}/{n} ticks, {ctl[1]}/{n} ticks, {ctl[2]}/{n} ticks"
    )
    print(f"act/hold decision changed by the signal       : {s_flips}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {p_flips}/{n} ticks")
    print(f"  trajectory: first differing tick            : {p_first}")
    print(f"  trajectory: equal->differ ONSETS            : {p_ons}")
    print(f"  trajectory: differ->equal re-convergences   : {p_back}")
    print(f"selected action changed by the signal         : {a_flips}/{n} ticks")
    print(f"  action: first differing tick / onsets       : {a_first} / {a_ons}")
    print(f"distinct actions A(wired) vs B(neutered)      : {len(set(aA))} vs {len(set(aB))}")
    print(f"  actions seen in A                           : {sorted(set(map(str, aA)))}")

    if any(ctl):
        print(
            "HARNESS NONDETERMINISTIC: two identical wired runs diverge, so no A-vs-B count from "
            "this harness is attributable to the neuter"
        )
        return 2
    if p_ons <= 1 and p_back == 0 and a_flips == 0:
        print(
            f"SINGLE-SPLIT: the trajectory divergence is ONE event at tick {p_first} that never "
            f"re-converges, and the selected action never changed -- '{p_flips}/{n} ticks' is a "
            f"divergence-onset count, not an effect size, and no decision moved"
        )
        return 1
    print("MULTI-EVENT: repeated divergence and/or a changed action")
    return 0


if __name__ == "__main__":
    sys.exit(main())
