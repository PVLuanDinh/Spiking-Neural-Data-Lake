"""imagination_ab -- the neuter-A/B for Imagination, which `tools/charge_probe.py` cannot reach.

WHY THIS EXISTS. `charge_probe.py` neuters a key of `Brain._drive_w`. Every Imagination signal is
outside that dict: the eleven drive sources are novelty / ignition / place / timing / reroute / help
/ percept / recall / surprise / drive / taskset, and `surprise` there is the Oja-PCA percept-latent
reconstruction residual set in `_perceive_real` (cortex/brain.py:1196-1206), NOT a forward-model
rollout error. So `charge_probe imagination` is not a typo away from working -- the category has no
probe at all, and every band in `docs/coverage/Imagination.md` would have been capped at 40 by the
scorer's F6 for want of an artifact rather than for want of charge. This is the missing probe.

WHY IT LIVES HERE. It was written by the Imagination auditor, whose write scope was this directory,
and it belongs in `tools/` beside `charge_probe.py`. Moving it costs one edit to the `command:` line
of the artifacts citing it; it is here so the evidence is tracked and re-runnable today rather than
blocked on a file another agent was editing in the same pass.

METHOD, copied from charge_probe so the two are comparable. Two runs of the ONE sanctioned
construction path `make_brain(seed=0, persist=False)` (Policy 4). A = wired. B = one named signal
pinned dead for every tick. The pin is a PROPERTY, never a post-tick reset: a value cleared after
`tick()` is recomputed before its consumer reads it next tick -- the measured leak that
`charge_probe._PinnedWeights` documents -- and a partial neuter biases the verdict toward "inert",
the direction that gets a live faculty retired.

OBSERVABLES, in the order they raise a band. Two are charge_probe's own (`arm_stopped` = the
act/hold decision, `map_position` = the resulting trajectory). The third is the arm's action-value
table, because rubric band 61-80 is "changes a decision, action, OR LEARNING UPDATE" and a world
model can be decisive through what the brain LEARNS without moving it on the same tick. `secondary`
is a read that is neither -- a narrator reply or a telemetry scalar -- and it is what separates band
41-60 from band 21-40.

VERDICT / EXIT CODE, contract-compatible with charge_probe:
  0  BEHAVIOURAL CHARGE OK       act/hold, trajectory or value-table diverged
  1  CONSUMED-BUT-NOT-DECISIVE   none of those diverged (a secondary read may still have)
  2  NOT-ELIGIBLE                the signal never fired, so there is no verdict to give

  PYTHONPATH=src python docs/coverage/artifacts/imagination_ab.py <signal> [ticks]
  signals: jepa_prediction | model_based_plan | counterfactual | dream_replay
"""

import os
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel  # noqa: E402
from snn_data_lake.cortex.brain import make_brain  # noqa: E402

SIGNALS = ("jepa_prediction", "model_based_plan", "counterfactual", "dream_replay")
PROMPT = "what if you engaged the arm right now?"


def _pin_attr(obj, name, value):
    """Give `obj` a private subclass on which `name` is a property returning `value`.

    Assignments inside the tick are swallowed, so the neutered signal CANNOT come back. Mutating
    `obj.__class__` is confined to this probe's own throwaway Brain instances.
    """
    base = type(obj)
    pinned = type(
        f"_Pinned{base.__name__}",
        (base,),
        {name: property(lambda _s: value, lambda _s, _v: None)},
    )
    obj.__class__ = pinned


def _qsum(arm):
    """Deterministic checksum of the grown-unit action-value table (the learning update)."""
    rows = sorted((u, sorted(row.items())) for u, row in arm.q.items())
    return zlib.crc32(repr(rows).encode())


def _fired(signal, brain, arm, mode, neuter):
    """Did the signal fire this tick? Only meaningful on arm A (B has it pinned dead)."""
    if neuter:
        return 1
    if signal == "dream_replay":
        return int(mode == "asleep")
    if signal == "jepa_prediction":
        return int(arm.jepa is not None)
    if signal == "model_based_plan":
        return int(bool(list(arm._planned_seq)))
    return int(brain._imagined_best is not None)


def run(signal, n, neuter):
    """One arm of the A/B. Returns (act, pos, qtab, secondary, fired)."""
    restore = None
    if neuter and signal == "jepa_prediction":
        # Pinned on the CLASS, not the instance: `arm.jepa` is sized lazily on the first tick, so an
        # instance patch would leave tick 1 un-neutered. Restored before this returns.
        restore = JepaWorldModel.prediction_error
        JepaWorldModel.prediction_error = lambda *_a, **_k: 0.0
    b = make_brain(seed=0, persist=False)
    arm = b.brain.arm
    if neuter and signal == "model_based_plan":
        _pin_attr(arm, "_planned_seq", [])
    elif neuter and signal == "counterfactual":
        _pin_attr(b, "_cfsim", None)
    elif neuter and signal == "dream_replay":
        b.attractor.sleep = lambda *_a, **_k: None
    act, pos, qtab, fired = [], [], [], 0
    for _ in range(n):
        mode = b.tick()
        st = b.state()
        act.append(bool(st.get("arm_stopped")))
        pos.append(tuple(st.get("map_position") or ()))
        qtab.append(_qsum(arm))
        fired += _fired(signal, b, arm, mode, neuter)
    if signal == "counterfactual":
        reply = b.respond(PROMPT)
        secondary = str(reply.get("reply") if isinstance(reply, dict) else reply)
    else:
        secondary = repr(round(float(b._consolidation_acc), 6))
    b.close()
    if restore is not None:
        JepaWorldModel.prediction_error = restore
    return act, pos, qtab, secondary, fired


def main():
    signal = sys.argv[1] if len(sys.argv) > 1 else "jepa_prediction"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if signal not in SIGNALS:
        print(f"unknown signal {signal!r}; known: {', '.join(SIGNALS)}")
        return 2

    a_act, a_pos, a_q, a_sec, a_fired = run(signal, n, False)
    b_act, b_pos, b_q, b_sec, _ = run(signal, n, True)
    act_f = sum(1 for x, y in zip(a_act, b_act, strict=True) if x != y)
    pos_f = sum(1 for x, y in zip(a_pos, b_pos, strict=True) if x != y)
    q_f = sum(1 for x, y in zip(a_q, b_q, strict=True) if x != y)
    distinct = f"{len(set(a_act))} / {len({repr(p) for p in a_pos})} / {len(set(a_q))}"

    print(f"signal neutered                               : {signal}")
    print(f"ticks the signal was live (A)                 : {a_fired}/{n}")
    print(f"distinct A values  act / traj / value-table   : {distinct}")
    print(f"act/hold decision changed by the signal       : {act_f}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {pos_f}/{n} ticks")
    print(f"action-value table changed by the signal      : {q_f}/{n} ticks")
    print(f"secondary read changed by the signal          : {a_sec != b_sec}")

    if not a_fired:
        print("NOT-ELIGIBLE: the signal never fired in this run -- no charge verdict")
        return 2
    if act_f or pos_f or q_f:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
