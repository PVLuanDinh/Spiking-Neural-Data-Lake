"""executive_ab_probe -- a neuter-A/B for the Executive signals tools/charge_probe.py cannot reach.

WHY IT EXISTS. charge_probe neuters exactly one class of signal: a key of `Brain._drive_w`
(charge_probe.py:46-51, :81). Of the eleven drive sources only `taskset` is an Executive faculty.
Conflict monitoring, EVC control allocation, the STN hyperdirect brake, the prefrontal goal vote and
the multi-step sequence planner are NOT drive sources, so under charge_probe alone every one of them
was capped at band 40 by an artifact format rather than by a measurement -- and
docs/COVERAGE_RUBRIC.md:65-68 is explicit that only neutering proves charge. This probe supplies the
neuter for those five, so the cap becomes evidence instead of a gap.

METHOD, deliberately identical to charge_probe's so the two are comparable. Two runs at the same
seed through the Policy-4 construction path `cortex.brain.make_brain(seed=0, persist=False)`. A =
wired. B = the named signal neutered AT ITS SOURCE FUNCTION for the whole run. Patching the source
rather than the consequence is what makes the neuter airtight: a value zeroed after `b.tick()` is
recomputed inside the next tick before anything reads it, which is the measured leak charge_probe's
_PinnedWeights exists to prevent (charge_probe.py:54-67). A source-function patch cannot leak,
because the value is never produced in the first place.

OBSERVABLES are behavioural, not telemetry, and are the same two charge_probe reports --
`arm_stopped` (the act/hold decision) and `map_position` (the resulting trajectory) -- plus the
SELECTED ACTION, which is strictly more informative here: the arm loop chooses among seven commands,
so a trajectory can diverge while act/hold never does, and the action is the decision that actually
changed. All three are printed in charge_probe's `n/N ticks` form so coverage_score.py's fingerprint
check can compare two faculties and catch a serial chain scored twice.

VERDICT and exit codes, preserved from charge_probe so a reader does not have to learn a second
contract:
  exit 0  BEHAVIOURAL CHARGE OK          -- neutering changed an action, a trajectory or act/hold
  exit 1  CONSUMED-BUT-NOT-DECISIVE      -- the signal is live and consumed, nothing behavioural moved
  exit 2  NOT-ELIGIBLE / UNKNOWN         -- the signal never activated, so there is no charge verdict

HONESTY. Eligibility is checked BEFORE any verdict and from the wired run's own trace, not assumed:
a signal that was zero on every tick of arm A cannot be neutered in any meaningful sense and reports
NOT-ELIGIBLE rather than a negative, because "the wire is dead" and "the wire never fired" are
different findings and reporting the second as the first is how a faculty gets wrongly retired. A
positive verdict needs no eligibility argument -- divergence proves the signal mattered.

  PYTHONPATH=src python docs/coverage/artifacts/executive_ab_probe.py [signal] [ticks]
  signals: conflict  evc  stn  plan  pfc_bias  pfc_gain
"""

import os
import sys

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.basal_ganglia.action_gating import BasalGangliaGate  # noqa: E402
from snn_data_lake.cortex.brain import make_brain  # noqa: E402
from snn_data_lake.cortex.executive_control import ExecutiveControl  # noqa: E402
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive  # noqa: E402

SIGNALS = ("conflict", "evc", "stn", "plan", "pfc_bias", "pfc_gain", "taskset")

_ORIG = {
    "conflict": ExecutiveControl.conflict,
    "evc": ExecutiveControl.evc,
    "plan_sequence": ExecutiveControl.plan_sequence,
    "select": BasalGangliaGate.select,
    "top_down_bias": PrefrontalExecutive.top_down_bias,
}

# signal -> (what is neutered, the wired-run attribute whose activity proves eligibility)
_ELIG = {
    "conflict": ("ExecutiveControl.conflict -> 0.0", "_conflict"),
    "evc": ("ExecutiveControl.evc -> 0.0", "_control_gain"),
    "stn": ("the stn argument of BasalGangliaGate.select -> 0.0", "_stn_elig"),
    "plan": ("ExecutiveControl.plan_sequence -> []", "_planned_seq"),
    "pfc_bias": ("PrefrontalExecutive.top_down_bias -> zeros", "_td_flipped"),
    "pfc_gain": ("AutonomousArm.pfc_gain pinned at its birth value", "pfc_gain"),
    # `taskset` IS a charge_probe-reachable drive source and charge_probe's verdict for it
    # (docs/coverage/artifacts/taskset_60.txt, exit 0) stands. This mode is not a replacement: it
    # re-runs the same neuter to obtain the SELECTED-ACTION flip count, which charge_probe does not
    # print, and which is the only one of the three observables that is not arithmetically implied by
    # the divergence-onset tick (trajectory always equals N - onset; see the header of the evidence
    # file). Same pin as charge_probe.py:54-76, for the same anti-leak reason.
    "taskset": ("Brain._drive_w['taskset'] pinned at 0.0", "_ext_drive"),
}


class _PinnedWeights(dict):
    """charge_probe.py:54's pinned `_drive_w`, reused verbatim in effect: the neutered key cannot be
    raised again by the competitive three-factor update inside the next tick."""

    def __init__(self, src, pinned):
        super().__init__(src)
        self._pinned = pinned

    def __setitem__(self, k, v):
        super().__setitem__(k, 0.0 if k == self._pinned else v)


def _pin_pfc_gain(arm):
    """Pin `pfc_gain` at its birth value so it cannot be re-learned. A plain attribute write would be
    undone by autonomous_loop.py:1419 on the next rewarded tick, so the pin is a read-only property
    on a per-instance subclass of the arm's type -- the same reasoning as charge_probe's pinned dict
    subclass rather than zeroing after the tick. The setter SWALLOWS writes instead of raising: the
    learning rule at :1419 is not inside a try, so an AttributeError there would crash the tick and
    the probe would be measuring an exception rather than a neuter."""
    born = float(arm.pfc_gain)
    cls = type(arm)
    pinned = type(
        cls.__name__ + "_PinnedGain",
        (cls,),
        {"pfc_gain": property(lambda s: born, lambda s, v: None)},
    )
    arm.__class__ = pinned
    return born


def _restore():
    ExecutiveControl.conflict = _ORIG["conflict"]
    ExecutiveControl.evc = _ORIG["evc"]
    ExecutiveControl.plan_sequence = _ORIG["plan_sequence"]
    BasalGangliaGate.select = _ORIG["select"]
    PrefrontalExecutive.top_down_bias = _ORIG["top_down_bias"]


def _patch(signal):
    if signal == "conflict":
        ExecutiveControl.conflict = lambda self, drives: 0.0
    elif signal == "evc":
        ExecutiveControl.evc = lambda self, *a, **k: 0.0
    elif signal == "plan":
        ExecutiveControl.plan_sequence = lambda self, *a, **k: []
    elif signal == "stn":
        _sel = _ORIG["select"]
        BasalGangliaGate.select = lambda self, drive, da, stn=0.0: _sel(self, drive, da, stn=0.0)
    elif signal == "pfc_bias":
        PrefrontalExecutive.top_down_bias = lambda self, strength=1.0: [0.0] * self.n_options


def run(signal, n, neuter):
    _restore()
    if neuter:
        _patch(signal)
    b = make_brain(seed=0, persist=False)
    arm = b.brain.arm
    if neuter and signal == "pfc_gain":
        _pin_pfc_gain(arm)
    if neuter and signal == "taskset":
        b._drive_w = _PinnedWeights(b._drive_w, "taskset")
        b._drive_w["taskset"] = 0.0
    stopped, pos, act, sig = [], [], [], []
    key = _ELIG[signal][1]
    for _ in range(n):
        b.tick()
        st = b.state()
        stopped.append(bool(st.get("arm_stopped")))
        pos.append(tuple(st.get("map_position") or ()))
        act.append((b._last_res or {}).get("action"))
        v = getattr(arm, key, None)
        if v is None:
            v = getattr(b, key, None)
        sig.append(tuple(v) if isinstance(v, list) else v)
    b.close()
    _restore()
    return stopped, pos, act, sig


def main():
    signal = sys.argv[1] if len(sys.argv) > 1 else "conflict"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if signal not in SIGNALS:
        print(f"unknown signal {signal!r}; known: {', '.join(SIGNALS)}")
        return 2

    sA, pA, aA, eA = run(signal, n, False)
    sB, pB, aB, _ = run(signal, n, True)

    act_flips = sum(1 for x, y in zip(sA, sB, strict=True) if x != y)
    pos_flips = sum(1 for x, y in zip(pA, pB, strict=True) if x != y)
    cmd_flips = sum(1 for x, y in zip(aA, aB, strict=True) if x != y)
    # ACTIVITY, uniformly defined: ticks on which the tracked signal differed from its value on tick
    # 1. Not "nonzero" -- pfc_gain's resting value is 0.5, so a nonzero count would report 60/60 for
    # a weight that never moved, which is the movement-reads-as-life error charge_probe.py:124-135
    # records for the drive weights.
    active = sum(1 for x in eA if x != eA[0])
    first = next((i for i in range(n) if pA[i] != pB[i]), None)

    what, key = _ELIG[signal]
    print(f"signal                                        : {signal}")
    print(f"neuter                                        : {what}")
    print(f"wired activity ({key} != tick-1 value)".ljust(46) + f": {active}/{n} ticks")
    print(f"distinct actions A(wired) vs B(neutered)      : {len(set(aA))} vs {len(set(aB))}")
    print(f"act/hold decision changed by the signal       : {act_flips}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {pos_flips}/{n} ticks")
    print(f"selected action changed by the signal         : {cmd_flips}/{n} ticks")
    print(f"first trajectory divergence at tick           : {first}")

    # ELIGIBILITY BEFORE VERDICT. A signal that never activated in arm A was not really neutered in
    # arm B, so a "nothing changed" result there says nothing about the wire. Only the NEGATIVE
    # verdict needs this guard; divergence is sound regardless.
    if active == 0 and not (act_flips or pos_flips or cmd_flips):
        print(
            "NOT-ELIGIBLE: the signal was inactive on every tick of the wired run, so arm B removed "
            "nothing -- no charge verdict"
        )
        return 2
    if act_flips or pos_flips or cmd_flips:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
