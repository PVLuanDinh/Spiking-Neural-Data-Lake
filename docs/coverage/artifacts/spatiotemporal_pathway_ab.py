"""HARNESS CONTROL, not a faculty charge: is the charge observable sensitive at all on the daemon path?

WHY. `tools/charge_probe.py drive 60` is the coverage ledger's known-good control -- 22/60 trajectory
flips, exit 0, cited at COVERAGE_BASELINE.md:90. Re-run through `spatiotemporal_probe.py`, i.e. with the
repo root importable so the e-prop perception tier loads exactly as it does under `learn_daemon.py`, the
SAME control returns 0/60 and exit 1 (`docs/coverage/artifacts/st_drive_control_60.txt`). Two readings
were possible and they lead to opposite conclusions:

  (a) the corrected harness is degenerate -- nothing varies, so every verdict on it is worthless; or
  (b) the drive vector is REDUNDANT -- with real perception live, ten other sources carry the sum, so
      removing any ONE of them no longer moves the argmax.

This script decides it. Three arms at one seed, 11 drive sources:

  A  all wired                      (the live brain)
  B  ALL eleven sources pinned to 0  (maximal intervention)
  C  every source pinned to 0 EXCEPT the named one (leave-one-in)

A-vs-B is the sensitivity control: if the whole top-down drive pathway is removed and the trajectory
still does not move, reading (a) holds and no single-source verdict from this harness means anything.
MEASURED: 47/60 trajectory flips. So the observable is sensitive, the pathway is decisive, and reading
(b) is the correct one -- `charge_probe`'s leave-one-out neuter systematically UNDER-detects charge in
the regime the daemon actually runs in.

C-vs-B isolates one source's own contribution against the same all-quiet background. Read it with care
and do NOT promote a band on it: C runs the brain in a counterfactual regime it never occupies (a
mostly-zero eligibility vector also changes the competitive weight update), so a flip there shows the
source is SUFFICIENT to move the arm when it is the only voice, not that it is NECESSARY in situ.
Sufficiency in an artificial regime is not the live claim the 61-80 band asks for.

EXIT CODE is deliberately about the CONTROL only: 0 when A-vs-B diverges (the harness is sensitive),
1 when it does not (the harness is degenerate and every verdict recorded on it must be withdrawn).
It is NOT a charge verdict for the named source, and it must never be cited as one.

  python docs/coverage/artifacts/spatiotemporal_pathway_ab.py <source> <ticks>
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

from snn_data_lake.cortex.brain import make_brain  # noqa: E402

# the learned per-(key, action) value tables, cleared alongside a pinned weight -- same list and same
# reason as tools/charge_probe.py:46-51 (a pinned weight alone leaks through the value table).
_TABLES = ("_drive_value", "_place_value", "_percept_value", "_recall_value")


class _Pinned(dict):
    """A `_drive_w` whose pinned keys CANNOT rise -- charge_probe.py:54's idiom, generalised from one
    key to a set, because the competitive update at brain.py:1538-1541 re-increments every key inside
    the tick and a post-tick zero is measurably leaky."""

    def __init__(self, src, pinned):
        super().__init__(src)
        self._pinned = set(pinned)

    def __setitem__(self, k, v):
        super().__setitem__(k, 0.0 if k in self._pinned else v)


def run(n, pinned):
    b = make_brain(seed=0, persist=False)  # Policy 4: the ONE sanctioned construction path
    if pinned:
        b._drive_w = _Pinned(b._drive_w, pinned)
        for k in pinned:
            b._drive_w[k] = 0.0
    stopped, pos = [], []
    for _ in range(n):
        b.tick()
        st = b.state()
        stopped.append(bool(st.get("arm_stopped")))
        pos.append(tuple(st.get("map_position") or ()))
        if pinned:
            for t in _TABLES:
                getattr(b, t).clear()
    tier, real = b._perception_tier, bool(b._perceiving_real)
    b.close()
    return stopped, pos, tier, real


def _flips(x, y):
    return sum(1 for a, b in zip(x, y, strict=True) if a != b)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "place"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    probe = make_brain(seed=0, persist=False)
    keys = sorted(probe._drive_w)
    probe.close()
    if source not in keys:
        print(f"unknown drive source {source!r}; known: {', '.join(keys)}")
        return 2

    sA, pA, tier, real = run(n, ())
    print(f"harness perception tier                       : {tier} (perceiving_real={real})")
    if not real:
        print("HARNESS NOT READY: real perception off -- this is not the daemon regime")
        return 2
    sB, pB, _, _ = run(n, tuple(keys))
    sC, pC, _, _ = run(n, tuple(k for k in keys if k != source))

    print(f"source under test                             : {source}")
    print(f"drive sources pinned in arm B / arm C          : {len(keys)} / {len(keys) - 1}")
    print(f"CONTROL  A(all wired) vs B(all neutered)      : {_flips(pA, pB)}/{n} ticks")
    print(f"CONTROL  act/hold A vs B                      : {_flips(sA, sB)}/{n} ticks")
    print(f"leave-one-in  C({source} only) vs B(none)     : {_flips(pC, pB)}/{n} ticks")
    print(f"leave-one-in  act/hold C vs B                 : {_flips(sC, sB)}/{n} ticks")
    print(
        f"distinct trajectories A / B / C               : {len(set(pA))} / {len(set(pB))} / {len(set(pC))}"
    )
    if _flips(pA, pB) or _flips(sA, sB):
        print(
            "HARNESS SENSITIVE: removing the whole top-down drive pathway moves the arm, so a "
            "single-source 0/60 on this harness is a MASKED negative, not proof of inertness"
        )
        return 0
    print(
        "HARNESS DEGENERATE: even neutering all eleven sources changed nothing -- withdraw every "
        "single-source verdict recorded on this harness"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
