"""affect_ab_probe -- neuter-A/B for the Affect signals `tools/charge_probe.py` has no source for.

WHY THIS EXISTS, AND WHY IT IS NOT tools/charge_probe.py. charge_probe neuters one key of
`Brain._drive_w`, so the only Affect faculty it can reach is the empathy `help` drive -- and on a
bare wake tick that source's precondition never fires (`_help_drive` is set in `respond()`, measured
0.0 on 60/60 ticks), so it can only ever return NOT-ELIGIBLE/UNKNOWN. Every other Affect signal (the
tonic stress state, the categorical emotion, the learned appraisal, the neuromodulator-fused core
affect) is consumed somewhere OTHER than `_drive_w` and is therefore outside that probe's reach.
Refusing to measure them would have left the whole category scored from static reading, which
`docs/COVERAGE_RUBRIC.md:65-68` says has been wrong in BOTH directions in this repo.

METHOD -- deliberately the same as charge_probe's, so the two are comparable:
  * two runs at the same seed through `cortex.brain.make_brain()` (the Policy-4 construction path);
  * A = wired. B = ONE affect signal pinned at its CONSUMER BOUNDARY for the whole run, by replacing
    the bound method that returns it. Pinning the RETURN (not zeroing state after the tick) is the
    airtight form -- the same reason charge_probe uses a `_PinnedWeights` dict subclass rather than
    a post-tick zero: a signal whose state is zeroed after the tick is re-derived in the next one;
  * observables are BEHAVIOURAL, never telemetry -- the action `AutonomousArm._select` returned, the
    `arm_stopped` decision, and the resulting `map_position` trajectory.

TWO HONESTY GUARDS, both of which can make this probe refuse to return a verdict:
  1. SELF-CONTROL. Every run first measures A against a second UN-neutered A. This brain is
     deterministic at a fixed seed, so the self-control must diverge on 0 ticks. If it does not, the
     harness is noisy and no divergence it reports would mean anything -> exit 2, no verdict. This
     is the built-in version of the `drive 60` control COVERAGE_BASELINE.md:106 cites.
  2. ELIGIBILITY. The neutered signal is sampled every tick in run A. If it took only ONE distinct
     value, it is a constant, and neutering a constant proves nothing about the faculty ->
     exit 2, NOT-ELIGIBLE. "The signal is inert" and "the signal never varied" are different
     findings, and reporting the second as the first is how a coverage number gets invented.

VERDICT / EXIT CODES, matching charge_probe so the ledger reads one way:
  0  BEHAVIOURAL CHARGE OK        -- neutering changed an action, a stop decision, or a trajectory
  1  CONSUMED-BUT-NOT-DECISIVE    -- the signal fired and varied, and nothing behavioural moved
  2  NOT-ELIGIBLE / harness degenerate -- no verdict is claimed

  PYTHONPATH=src python docs/coverage/artifacts/affect_ab_probe.py [signal] [ticks]
  signals: cortisol_caution cortisol_learnrate categorical_emotion appraisal affect_integrator
           empathy_help
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.amygdala.affective_appraisal import AffectiveAppraisal  # noqa: E402
from snn_data_lake.brainstem.autonomous_loop import AutonomousArm  # noqa: E402
from snn_data_lake.cortex.affect_core import AffectCore  # noqa: E402
from snn_data_lake.cortex.brain import make_brain  # noqa: E402
from snn_data_lake.neuromodulation.affect_integrator import AffectIntegrator  # noqa: E402
from snn_data_lake.neuromodulation.cortisol import Cortisol  # noqa: E402

# The distress turn that makes the empathy precondition fire. `empathy_help` is the ONLY signal here
# whose live path is the chat turn rather than the wake tick, so it is the only one that needs it.
DISTRESS = "i'm scared and i feel awful"

SIGNALS = (
    "cortisol_caution",  # Cortisol.caution_bias -> BG dopamine damping (autonomous_loop.py:724)
    "cortisol_learnrate",  # Cortisol.learn_rate_scale -> value update (autonomous_loop.py:1434)
    "categorical_emotion",  # AffectCore.recognize -> episodic recall cue (brain.py:1447)
    "appraisal",  # AffectiveAppraisal.appraise -> coherence raster (brain.py:2041)
    "affect_integrator",  # AffectIntegrator.integrate -> emergent emotion (brain.py:1941)
    "empathy_help",  # AffectCore.empathize -> _drive_w["help"] (brain.py:3078)
)

_ORIG = {
    "cb": Cortisol.caution_bias,
    "lrs": Cortisol.learn_rate_scale,
    "rec": AffectCore.recognize,
    "app": AffectiveAppraisal.appraise,
    "integ": AffectIntegrator.integrate,
}


class _PinnedWeights(dict):
    """A `_drive_w` whose neutered key cannot rise -- copied in intent from charge_probe's guard:
    the three-factor update in brain.py touches every key INSIDE the tick, so a post-tick zero
    leaks back before the drive is consumed."""

    def __init__(self, src, pinned):
        super().__init__(src)
        self._pinned = pinned

    def __setitem__(self, k, v):
        super().__setitem__(k, 0.0 if k == self._pinned else v)


def _restore():
    Cortisol.caution_bias = _ORIG["cb"]
    Cortisol.learn_rate_scale = _ORIG["lrs"]
    AffectCore.recognize = _ORIG["rec"]
    AffectiveAppraisal.appraise = _ORIG["app"]
    AffectIntegrator.integrate = _ORIG["integ"]


def _neuter(signal):
    """Pin ONE affect signal at its consumer boundary. Nothing else in the brain is touched."""
    if signal == "cortisol_caution":
        Cortisol.caution_bias = lambda self: 0.0
    elif signal == "cortisol_learnrate":
        Cortisol.learn_rate_scale = lambda self, signed: 1.0
    elif signal == "categorical_emotion":
        AffectCore.recognize = lambda self, v, a, jitter=0.0: "calm"
    elif signal == "appraisal":
        AffectiveAppraisal.appraise = lambda self, features: (0.0, 0.0)
    elif signal == "affect_integrator":
        AffectIntegrator.integrate = lambda self, threat, reward, mood, cortisol: (0.0, 0.0)


def _sample(signal, brain):
    """The value of the neutered signal on this tick -- the eligibility evidence."""
    arm = getattr(brain.brain, "arm", None)
    cort = getattr(arm, "cort", None)
    if signal == "cortisol_caution":
        return round(float(_ORIG["cb"](cort)), 6) if cort is not None else None
    if signal == "cortisol_learnrate":
        return round(float(cort.level), 6) if cort is not None else None
    if signal == "categorical_emotion":
        return brain.state().get("emotion")
    if signal == "appraisal":
        return brain.state().get("affective_appraisal")
    if signal == "affect_integrator":
        return brain.state().get("emergent_emotion")
    if signal == "empathy_help":
        return round(float(getattr(brain, "_help_drive", 0.0)), 6)
    return None


def run(signal, n, neuter):
    """One run. Returns (actions, stopped, positions, sampled signal values)."""
    _restore()
    if neuter and signal != "empathy_help":
        _neuter(signal)
    chosen = []
    _sel = AutonomousArm._select

    def _spy(self, *a, **k):
        r = _sel(self, *a, **k)
        chosen.append(r)
        return r

    AutonomousArm._select = _spy
    try:
        b = make_brain(seed=0, persist=False)  # Policy 4: the ONE sanctioned construction path
        if signal == "empathy_help":
            for _ in range(5):  # settle, then make the empathy precondition fire
                b.tick()
                b.state()
            b.respond(DISTRESS)
            if neuter:  # pin the drive the empathic resonance feeds
                b._drive_w = _PinnedWeights(b._drive_w, "help")
                b._drive_w["help"] = 0.0
        stopped, pos, seen = [], [], []
        for _ in range(n):
            b.tick()
            st = b.state()
            stopped.append(bool(st.get("arm_stopped")))
            pos.append(tuple(st.get("map_position") or ()))
            seen.append(_sample(signal, b))
        b.close()
    finally:
        AutonomousArm._select = _sel
        _restore()
    return chosen, stopped, pos, seen


def _diff(a, b):
    m = min(len(a), len(b))
    return sum(1 for x, y in zip(a[:m], b[:m], strict=True) if x != y), m


def main():
    signal = sys.argv[1] if len(sys.argv) > 1 else "cortisol_caution"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if signal not in SIGNALS:
        print(f"unknown signal {signal!r}; known: {', '.join(SIGNALS)}")
        return 2

    aA, sA, pA, seen = run(signal, n, False)
    aC, sC, pC, _ = run(signal, n, False)  # self-control: A vs a second un-neutered A
    aB, sB, pB, _ = run(signal, n, True)

    ctl_a, _ = _diff(aA, aC)
    ctl_s, _ = _diff(sA, sC)
    ctl_p, _ = _diff(pA, pC)
    act, m = _diff(aA, aB)
    stop, _ = _diff(sA, sB)
    traj, _ = _diff(pA, pB)
    distinct = len({repr(x) for x in seen})

    print(f"signal                                        : {signal}")
    print("consumer-boundary neuter                      : pinned for the whole run")
    print(f"distinct values the signal took in A          : {distinct}  (sample {seen[:4]})")
    print(
        f"SELF-CONTROL A vs A: action / stop / traj     : {ctl_a}/{m} ticks, "
        f"{ctl_s}/{n} ticks, {ctl_p}/{n} ticks"
    )
    print(f"selected action changed by the signal         : {act}/{m} ticks")
    print(f"arm stop decision changed by the signal       : {stop}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {traj}/{n} ticks")

    if ctl_a or ctl_s or ctl_p:
        print(
            "NOT-ELIGIBLE: the SELF-CONTROL diverged, so this harness is not deterministic at a "
            "fixed seed and any divergence it reports above would be noise -- no charge verdict"
        )
        return 2
    if distinct <= 1:
        print(
            "NOT-ELIGIBLE: the signal took ONE value for the whole run (a constant, not a signal), "
            "so neutering it cannot say whether the faculty charges behaviour -- no charge verdict"
        )
        return 2
    if act or stop or traj:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
