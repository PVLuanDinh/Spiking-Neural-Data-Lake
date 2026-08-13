"""crosscat_jepa_vs_conflict_ab -- is the cross-category fingerprint collision real duplication?

WHY IT EXISTS. `docs/coverage/Imagination.md:310-313` records that its `latent_world_model`
(`imag_jepa_action_60.txt`) and `docs/coverage/Executive.md`'s `acc_conflict_monitoring_evc`
(`exec_conflict_60.txt`) report the SAME three behavioural numbers -- act/hold `0/60`, trajectory
`47/60`, selected action `23/60` -- with the same first-divergence tick, and that `coverage_score.py`'s
W3 check is structurally incapable of seeing it because W3 iterates `prints` inside the per-category
loop. Both artifacts exit 0 and both rows are banded 61, so unlike the four within-category W3
warnings already adjudicated as false positives, these two bands genuinely rest on the colliding
numbers. If the two probes are two probe points on one serial chain, two above-40 bands are inflated
-- the defect that cost Executive 9 points in the withdrawn baseline, where four "faculties" had
byte-identical action traces on 122/122 wake ticks.

THE TEST THE TWO EXISTING PROBES CANNOT RUN. Each is a two-arm A/B against its own wired arm, so
neither can compare the two NEUTERED arms to each other. A shared flip COUNT is weak evidence; the
strong question is whether the two neuters produce the same TRACE. This probe runs four arms at one
seed through the Policy-4 path and answers, in order of decisiveness:

  1. B1 vs B2, per tick. If the two neutered arms are identical on every tick, the two rows are ONE
     measurement whatever category they were filed under -- the withdrawn baseline's exact shape.
     If they differ, the equal counts are a coincidence of arithmetic, not a shared cause.
  2. The DIVERGENCE TICK SETS, not just their sizes. Two signals can each flip 23 ticks on disjoint
     ticks; `23 == 23` cannot tell that from `the same 23 ticks`.
  3. THE MARGINAL, both ways (arm C = both neutered). If C vs B1 is 0/60 the jepa error is nested
     inside the conflict channel and adds nothing; likewise C vs B2 for conflict. Two nonzero
     marginals is the mechanical statement that neither row is a downstream consequence of the other.
  4. WHAT HAPPENS AT THE SHARED ONSET TICK -- circadian phase, whether `prediction_error` fired, and
     the wired `_conflict` value -- because two independent signals gated by one upstream event
     (a first sleep transition, a first real percept) can first diverge on the same tick honestly.
  5. ARM D, THE UNRELATED-NEUTER CONTROL. `_drive_w['novelty']` is a drive source belonging to
     neither category's roster, pinned with charge_probe's own `_PinnedWeights`. If a third,
     unrelated perturbation ALSO first diverges at the same tick, then the onset tick and the flip
     tick set are properties of where THIS arm's gate arbitration is knife-edge at seed 0 -- a
     harness constant -- and not a shared cause between the two faculties under test. Without this
     arm, "the shared tick set is an artefact" is an inference; with it, it is a measurement.

METHOD is `imagination_refute_ab.py`'s and `executive_ab_probe.py`'s, so the numbers are directly
comparable to both recorded artifacts: `make_brain(seed=0, persist=False)`, the neuter installed on
the CLASS before construction (`arm.jepa` is sized lazily on the first tick), restored in `finally`.
Arm A wraps `prediction_error` in a per-tick RECORDER that returns the true value unchanged, so arm
A's own counts must reproduce `imag_jepa_action_60.txt`'s 23/60 and 47/60 and `exec_conflict_60.txt`'s
23/60 and 47/60; that equality is the check that this harness is behaviour-preserving.

EXIT CODE, contract-compatible with charge_probe.py so no reader learns a third contract:
  0 = the two neuters are DISTINCT measurements (the collision is a false positive)
  1 = the two neutered arms are identical on every observable (REAL duplication)
  2 = a precondition failed, so there is no verdict

  PYTHONPATH=src python docs/coverage/artifacts/crosscat_jepa_vs_conflict_ab.py [ticks]
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel  # noqa: E402
from snn_data_lake.cortex.brain import make_brain  # noqa: E402
from snn_data_lake.cortex.executive_control import ExecutiveControl  # noqa: E402

_TRUE_PERR = JepaWorldModel.prediction_error
_TRUE_CONF = ExecutiveControl.conflict


class _PinnedWeights(dict):
    """charge_probe.py:54's pinned `_drive_w`, reused in effect: the neutered key cannot be raised
    again by the competitive three-factor update inside the next tick."""

    def __init__(self, src, pinned):
        super().__init__(src)
        self._pinned = pinned

    def __setitem__(self, k, v):
        super().__setitem__(k, 0.0 if k == self._pinned else v)


def run(kill_jepa, kill_conflict, n, kill_drive=None):
    """One arm. Returns (act, action, pos, phase, perr_calls, conflict) per tick."""
    calls = [0]

    def _perr(self, *a, **k):
        calls[0] += 1
        if kill_jepa:
            return 0.0
        return float(_TRUE_PERR(self, *a, **k))

    JepaWorldModel.prediction_error = _perr
    if kill_conflict:
        ExecutiveControl.conflict = lambda self, drives: 0.0
    try:
        b = make_brain(seed=0, persist=False)
        arm = b.brain.arm
        if kill_drive:
            b._drive_w = _PinnedWeights(b._drive_w, kill_drive)
            b._drive_w[kill_drive] = 0.0
        act, action, pos, phase, fired, conf = [], [], [], [], [], []
        for _ in range(n):
            before = calls[0]
            phase.append(b.tick())
            st = b.state()
            act.append(bool(st.get("arm_stopped")))
            action.append((b._last_res or {}).get("action"))
            pos.append(tuple(st.get("map_position") or ()))
            fired.append(calls[0] - before)
            conf.append(getattr(arm, "_conflict", None))
        b.close()
    finally:
        JepaWorldModel.prediction_error = _TRUE_PERR
        ExecutiveControl.conflict = _TRUE_CONF
    return act, action, pos, phase, fired, conf


def _ticks(xs, ys):
    """The 1-indexed ticks on which two traces differ. A SET, not a count."""
    return tuple(i + 1 for i, (x, y) in enumerate(zip(xs, ys, strict=True)) if x != y)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    aA, cA, pA, phA, fA, kA = run(False, False, n)
    aB1, cB1, pB1, _, _, _ = run(True, False, n)
    aB2, cB2, pB2, _, _, _ = run(False, True, n)
    aC, cC, pC, _, _, _ = run(True, True, n)
    aD, cD, pD, _, _, _ = run(False, False, n, kill_drive="novelty")

    d = {
        "A-B1 (jepa)": (_ticks(aA, aB1), _ticks(cA, cB1), _ticks(pA, pB1)),
        "A-B2 (conflict)": (_ticks(aA, aB2), _ticks(cA, cB2), _ticks(pA, pB2)),
        "A-D (novelty ctl)": (_ticks(aA, aD), _ticks(cA, cD), _ticks(pA, pD)),
        "B1-B2 (DECISIVE)": (_ticks(aB1, aB2), _ticks(cB1, cB2), _ticks(pB1, pB2)),
        "B1-C (marg conf)": (_ticks(aB1, aC), _ticks(cB1, cC), _ticks(pB1, pC)),
        "B2-C (marg jepa)": (_ticks(aB2, aC), _ticks(cB2, cC), _ticks(pB2, pC)),
    }
    print(f"arms: A wired / B1 jepa_prediction=0 / B2 conflict=0 / C both,  seed 0, n={n}")
    print(f"prediction_error calls on the wired run (A)   : {sum(fA)}")
    print(f"wake ticks on the wired run (A)               : {phA.count('awake')}/{n}")
    print(
        f"distinct A values act/action/traj             : {len(set(aA))} / {len(set(cA))} / "
        f"{len({repr(x) for x in pA})}"
    )
    # Printed one observable per line in charge_probe's canonical `n/N ticks` form, NOT as a compact
    # triple, and the reason is the hole documented above `coverage_score._fingerprint`: that regex
    # matches only `(\d+)/(\d+) ticks`, so a probe reporting `0/60 23/60 47/60` fingerprints as the
    # EMPTY tuple and `--rerun` degrades to an exit-code check -- vacuous as a forgery check, which is
    # exactly what it must not be for the artifact adjudicating two band-61 rows. This file is not
    # named in any evidence table's artifact column, so a rich fingerprint here cannot create a
    # spurious W3/W5 collision: only artifacts a ROW names are fingerprinted by the scorer.
    print("--- reproduction check against the three recorded artifacts ---")
    print("--- (A-B1 must equal imag_jepa_action_60, A-B2 exec_conflict_60, A-D novelty_60) ---")
    for label in ("A-B1 (jepa)", "A-B2 (conflict)", "A-D (novelty ctl)"):
        act, cmd, pos = d[label]
        print(f"{label:<17} act/hold changed by the neuter : {len(act)}/{n} ticks")
        print(f"{label:<17} selected action changed        : {len(cmd)}/{n} ticks")
        print(f"{label:<17} arm trajectory changed        : {len(pos)}/{n} ticks")
    print("--- the comparison neither existing probe can make ---")
    for label in ("B1-B2 (DECISIVE)", "B1-C (marg conf)", "B2-C (marg jepa)"):
        act, cmd, pos = d[label]
        print(f"{label:<17} act/hold differs              : {len(act)}/{n} ticks")
        print(f"{label:<17} selected action differs       : {len(cmd)}/{n} ticks")
        print(f"{label:<17} arm trajectory differs        : {len(pos)}/{n} ticks")
    same_cmd = d["A-B1 (jepa)"][1] == d["A-B2 (conflict)"][1]
    print("--- divergence TICK SETS, not counts ---")
    print(f"action-flip ticks, jepa neuter                : {d['A-B1 (jepa)'][1]}")
    print(f"action-flip ticks, conflict neuter            : {d['A-B2 (conflict)'][1]}")
    print(f"action-flip ticks, novelty CONTROL neuter     : {d['A-D (novelty ctl)'][1]}")
    print(f"the two action-flip tick sets are identical   : {same_cmd}")
    ctl = set(d["A-D (novelty ctl)"][1])
    tgt = set(d["A-B1 (jepa)"][1])
    print(f"control tick set == the two under test        : {ctl == tgt}")
    print(
        f"control shares {len(ctl & tgt)} of the {len(tgt)} flip ticks     "
        f"      : jaccard {len(ctl & tgt) / max(1, len(ctl | tgt)):.3f}"
    )
    print(
        f"onset tick action/traj, jepa   (1-indexed)    : "
        f"{(d['A-B1 (jepa)'][1] or [None])[0]} / {(d['A-B1 (jepa)'][2] or [None])[0]}"
    )
    print(
        f"onset tick action/traj, conflict (1-indexed)  : "
        f"{(d['A-B2 (conflict)'][1] or [None])[0]} / {(d['A-B2 (conflict)'][2] or [None])[0]}"
    )
    onset = (d["A-B1 (jepa)"][1] or [None])[0]
    if onset:
        pre = sum(fA[: onset - 1])
        print(f"--- both signals were LIVE before the shared onset tick {onset} ---")
        print(f"prediction_error calls on ticks 1..{onset - 1:<24}: {pre}")
        print(
            f"wired _conflict was nonzero on ticks 1..{onset - 1:<19}: "
            f"{sum(1 for x in kA[: onset - 1] if x)}"
        )
        print(
            "so neither neuter's onset is that signal's FIRST FIRING; it is the first tick on "
            "which a perturbation propagated to the gate's winner."
        )
    print("--- what the arms actually chose at the first shared divergent ticks ---")
    shown = sorted(set(d["A-B1 (jepa)"][1][:6]) | set(d["A-B2 (conflict)"][1][:6]))
    for t in shown:
        i = t - 1
        print(
            f"  tick {t:>3} {phA[i]:<6} perr_calls={fA[i]} conflict={kA[i]}"
            f"  A={cA[i]} B1={cB1[i]} B2={cB2[i]} C={cC[i]}"
        )

    if sum(fA) == 0:
        print("NOT-ELIGIBLE: prediction_error never fired -- no verdict")
        return 2
    identical = all(not any(x) for x in d["B1-B2 (DECISIVE)"])
    if identical:
        print("REAL DUPLICATION: the two neutered arms are identical on every observable")
        return 1
    print("DISTINCT MEASUREMENTS: the two neuters produce different arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
