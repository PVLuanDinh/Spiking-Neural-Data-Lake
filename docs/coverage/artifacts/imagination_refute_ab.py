"""imagination_refute_ab -- the adversarial re-measure of Imagination's one band-41+ claim.

WHY IT EXISTS. `docs/coverage/Imagination.md` scores `latent_world_model` at 65 on
`imag_jepa_60.txt`: pin `JepaWorldModel.prediction_error` to 0.0 and the arm's trajectory diverges
on 47/60 ticks and its action-value table on 56/60. That artifact is real and it reproduces
(`coverage_score.py --rerun imag_jepa_60`, exit 0, observables identical at HEAD). So the forgery
check is not where this claim can be attacked. What CAN be attacked is what the two large numbers
mean, and the auditor's own probe cannot answer it, because `imagination_ab.py` records three
observables of which exactly one is per-tick -- `arm_stopped` -- and that one took ONE distinct
value on the wired run, i.e. it is degenerate. The other two, `map_position` and the value-table
checksum, are CUMULATIVE STATE: once two runs differ they keep differing, so `47/60` is
arithmetically `N - onset` and not an effect size. `docs/coverage/Executive.md:110-116` records
exactly this trap for `22/60`.

WHAT THIS ADDS, and it is the whole point:

  1. THE SELECTED ACTION (`b._last_res["action"]`), per tick, non-cumulative -- the observable that
     says whether a DECISION changed, which is the band 61-80 question. `executive_ab_probe.py`
     already reports it; the Imagination probe does not, and band 65 was awarded without it.
  2. THE GREEDY POLICY over the action-value table (argmax per grown unit). Band 61-80 counts a
     learning update, but "the stored numbers differ" is arithmetically guaranteed the moment a
     nonzero summand is removed from a reward. Whether the LEARNED PREFERENCE differs is the
     question that separates a learning update from a bookkeeping difference.
  3. FIRST-DIVERGENCE TICKS for every observable, so a cumulative count can be read as the onset
     number it is.
  4. ARM C, THE SCRAMBLE CONTROL -- the auditor's own stated caveat, unmeasured in their file: "a
     stricter version would inject a SHUFFLED error of the same magnitude". Arm C replays arm A's
     own prediction errors in a deterministically shuffled order, so the multiset of magnitudes is
     IDENTICAL to A and only the pairing between error and transition is destroyed. If C diverges
     from A about as much as B does, the A/B is measuring sensitivity to a varying scalar in the
     reward channel and cannot attribute the divergence to the world model's PREDICTIVE CONTENT.

METHOD is `imagination_ab.py`'s, so the numbers are directly comparable: `make_brain(seed=0,
persist=False)` (Policy 4), the neuter installed on the CLASS before construction because `arm.jepa`
is sized lazily on the first tick, restored before return. Arm A is wrapped by a RECORDER that
returns the true value unchanged -- behaviour-preserving, and the check that it is behaviour-
preserving is that this file's A-vs-B counts must equal `imag_jepa_60.txt`'s (47/60 and 56/60).

EXIT CODE, contract-compatible with `charge_probe.py` and `imagination_ab.py` so no reader has to
learn a third contract: 0 = something behavioural or a learning update diverged, 1 = nothing did,
2 = the signal never fired so there is no verdict.

  PYTHONPATH=src python docs/coverage/artifacts/imagination_refute_ab.py [ticks]
"""

import os
import random
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel  # noqa: E402
from snn_data_lake.cortex.brain import make_brain  # noqa: E402

_TRUE_PERR = JepaWorldModel.prediction_error


def _qsum(arm):
    """Auditor's observable, verbatim: checksum of the grown-unit action-value table."""
    rows = sorted((u, sorted(row.items())) for u, row in arm.q.items())
    return zlib.crc32(repr(rows).encode())


def _policy(arm):
    """The LEARNED PREFERENCE the table encodes: argmax action per unit, ties broken by name."""
    pol = []
    for unit, row in sorted(arm.q.items()):
        if row:
            pol.append((unit, max(sorted(row.items()), key=lambda kv: kv[1])[0]))
    return tuple(pol)


def run(mode, n, errs):
    """One arm. mode: 'wired' records the true errors, 'zero' pins 0.0, 'scramble' replays `errs`
    shuffled. Returns (act, action, pos, qtab, policy, fired)."""
    seen = []
    if mode == "wired":

        def _perr(self, *a, **k):
            v = float(_TRUE_PERR(self, *a, **k))
            seen.append(v)
            return v

    elif mode == "zero":

        def _perr(self, *a, **k):
            seen.append(0.0)
            return 0.0

    else:
        bag = list(errs)
        random.Random(1234).shuffle(bag)

        def _perr(self, *a, **k):
            v = bag[len(seen) % len(bag)] if bag else 0.0
            seen.append(v)
            return v

    JepaWorldModel.prediction_error = _perr
    try:
        b = make_brain(seed=0, persist=False)
        arm = b.brain.arm
        act, action, pos, qtab, policy = [], [], [], [], []
        for _ in range(n):
            b.tick()
            st = b.state()
            act.append(bool(st.get("arm_stopped")))
            action.append((b._last_res or {}).get("action"))
            pos.append(tuple(st.get("map_position") or ()))
            qtab.append(_qsum(arm))
            policy.append(_policy(arm))
        b.close()
    finally:
        JepaWorldModel.prediction_error = _TRUE_PERR
    return act, action, pos, qtab, policy, seen


def _flips(xs, ys):
    return sum(1 for x, y in zip(xs, ys, strict=True) if x != y)


def _first(xs, ys):
    return next((i + 1 for i in range(len(xs)) if xs[i] != ys[i]), None)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    aA, cA, pA, qA, yA, errs = run("wired", n, None)
    aB, cB, pB, qB, yB, _ = run("zero", n, None)
    aC, cC, pC, qC, yC, _ = run("scramble", n, errs)

    nz = sum(1 for e in errs if e != 0.0)
    print("signal neutered                               : jepa_prediction (re-measure)")
    print(f"prediction_error calls on the wired run (A)   : {len(errs)} ({nz} nonzero)")
    print(
        "distinct A values act/action/traj/qtab/policy : "
        f"{len(set(aA))} / {len(set(cA))} / {len({repr(p) for p in pA})} / "
        f"{len(set(qA))} / {len(set(yA))}"
    )
    print("--- arm B: prediction_error pinned to 0.0 (the auditor's neuter) ---")
    print(f"act/hold decision changed by the signal       : {_flips(aA, aB)}/{n} ticks")
    print(f"selected action changed by the signal         : {_flips(cA, cB)}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {_flips(pA, pB)}/{n} ticks")
    print(f"action-value table changed by the signal      : {_flips(qA, qB)}/{n} ticks")
    print(f"greedy policy changed by the signal           : {_flips(yA, yB)}/{n} ticks")
    print(
        "first divergence tick  action/traj/qtab/policy: "
        f"{_first(cA, cB)} / {_first(pA, pB)} / {_first(qA, qB)} / {_first(yA, yB)}"
    )
    print("--- arm C: A's own errors, shuffled (magnitude-matched, content destroyed) ---")
    print(f"selected action changed by the scramble       : {_flips(cA, cC)}/{n} ticks")
    print(f"arm trajectory changed by the scramble        : {_flips(pA, pC)}/{n} ticks")
    print(f"action-value table changed by the scramble    : {_flips(qA, qC)}/{n} ticks")
    print(f"greedy policy changed by the scramble         : {_flips(yA, yC)}/{n} ticks")
    print(
        "first divergence tick  action/traj/qtab/policy: "
        f"{_first(cA, cC)} / {_first(pA, pC)} / {_first(qA, qC)} / {_first(yA, yC)}"
    )

    if not nz:
        print("NOT-ELIGIBLE: the prediction error was zero on every call -- no charge verdict")
        return 2
    if _flips(aA, aB) or _flips(cA, cB) or _flips(pA, pB) or _flips(qA, qB):
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
