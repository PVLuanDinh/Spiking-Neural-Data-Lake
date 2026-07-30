"""Social neuter-A/B: does theory-of-mind belief divergence change the arm's behaviour?

WHY THIS EXISTS ALONGSIDE tools/charge_probe.py. That probe neuters one key of `Brain._drive_w`
and never chats, so it can address exactly one Social faculty (`help`, recorded at exit 2). The
wire under test here is neither: `cortex/recursive_tom` divergence -> `_false_beliefs` ->
`_tom_protect` -> `arm._social_risk` (brain.py:1627), read as caution by `autonomous_loop.py:728`
in the basal-ganglia gate and `:659` in the non-BG fallback. It is an ATTRIBUTE ON THE ARM, and it
is only ever raised by a dialogue turn. Same observables as charge_probe (`arm_stopped`,
`map_position`) and the same exit-code contract, so an artifact recorded here is comparable to one
recorded from that file.

THE NEUTER IS A DATA DESCRIPTOR, and the naive version was measurably leaky. Assigning a dict
subclass to `arm.__dict__` neuters nothing: CPython stores instance attributes with PyDict_SetItem
on the internal dict pointer, bypassing a subclass __setitem__. Measured: the "neutered" arm still
read 0.34 and the run returned NOT-DECISIVE. `property(getter -> 0.0, setter -> no-op)` is a data
descriptor, so it wins over the instance dict in both directions. Same defect class
tools/charge_probe.py:54-67 documents for `_drive_w`, one layer down, and it fails in the same
direction -- a partial neuter returns a confident negative, and the negative is the verdict that
gets a live faculty retired.

BOTH ARMS RUN THE IDENTICAL DIALOGUE, so everything else a chat turn does is held constant and any
divergence is attributable to `_social_risk` alone.

THREE CONTROLS RUN IN THE SAME PROCESS AND THE PROBE FAILS IF ANY OF THEM FIRES. A bespoke probe
that only ever reports the result it was written to find is not evidence, so the falsifiers are not
optional here:

  C1 sham    -- patch the same kind of descriptor onto an attribute NOTHING reads. Non-zero means
                the class patch itself perturbs the run, not the signal.
  C2 no-chat -- neuter `_social_risk` with no dialogue, so the signal is 0.0 in BOTH arms.
                Non-zero means the harness reports divergence without a signal.
  C3 determ. -- two identical unpatched runs. Non-zero means the comparison is noise.

VERDICT. "BEHAVIOURAL CHARGE OK" (exit 0) if a control-clean run diverges;
"CONSUMED-BUT-NOT-DECISIVE" (exit 1) if it does not; "NOT-ELIGIBLE" / "CONTROL FAILED" (exit 2) if
the signal never rose or a falsifier fired -- eligibility is reported separately from charge,
because "the wire is dead" and "the wire never got a chance to fire" are different findings.

  PYTHONPATH=src python docs/coverage/artifacts/social_tom_ab_probe.py [ticks]
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cortex.brain import make_brain  # noqa: E402

# A Sally-Anne turn pair that satisfies brain.py:83 `_TOM_PLACE` and then one of the 13 absence
# idioms in brain.py:89 `_TOM_ABSENT`, the ONLY route by which `_false_beliefs` is ever written.
DIALOGUE = (
    "i put the marble in the basket.",
    "someone moved the marble to the table while i was away.",
)
SIGNAL = "_social_risk"


def run(n, attr=None, chat=True):
    """One arm. `attr` is neutered on the arm's CLASS by a data descriptor for the whole run; the
    class attribute is restored in a finally block so arms cannot contaminate each other."""
    b = make_brain(seed=0, persist=False)
    arm = b.brain.arm
    cls = type(arm)
    had = attr in cls.__dict__ if attr else False
    saved = cls.__dict__.get(attr) if attr else None
    if attr:
        setattr(cls, attr, property(lambda self: 0.0, lambda self, v: None))
    stopped, pos, seen = [], [], []
    try:
        for i in range(n):
            if chat and i == n // 2:
                for line in DIALOGUE:
                    b.respond(line)
            b.tick()
            st = b.state()
            stopped.append(bool(st.get("arm_stopped")))
            pos.append(tuple(st.get("map_position") or ()))
            seen.append(float(getattr(arm, SIGNAL, 0.0) or 0.0))
    finally:
        if attr:
            if had:
                setattr(cls, attr, saved)
            else:
                delattr(cls, attr)
        b.close()
    return stopped, pos, seen


def diff(a, b):
    return sum(1 for p, q in zip(a, b, strict=True) if p != q)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    wired = run(n)
    neutered = run(n, attr=SIGNAL)
    sham = run(n, attr=SIGNAL + "_sham")
    quiet_a = run(n, chat=False)
    quiet_b = run(n, attr=SIGNAL, chat=False)
    twin = run(n)

    w_max = round(max(wired[2]), 4)
    n_max = round(max(neutered[2]), 4)
    w_live = sum(1 for v in wired[2] if v)
    act = diff(wired[0], neutered[0])
    traj = diff(wired[1], neutered[1])
    c1 = (diff(wired[0], sham[0]), diff(wired[1], sham[1]))
    c2 = (diff(quiet_a[0], quiet_b[0]), diff(quiet_a[1], quiet_b[1]))
    c3 = (diff(wired[0], twin[0]), diff(wired[1], twin[1]))

    print(f"signal                                    : arm.{SIGNAL} (recursive_tom)")
    print(f"dialogue turns injected at tick {n // 2:<9} : {len(DIALOGUE)}")
    print(f"signal A(wired) max / live ticks           : {w_max} / {w_live} of {n}")
    print(f"signal B(neutered) max                     : {n_max}  (held={n_max == 0.0})")
    print(
        f"arm HOLD rate A(wired) vs B(neutered)      : "
        f"{sum(wired[0]) / n:.3f} vs {sum(neutered[0]) / n:.3f}"
    )
    print(f"act/hold decision changed by the signal    : {act}/{n} ticks")
    print(f"arm trajectory changed by the signal       : {traj}/{n} ticks")
    print(f"C1 sham descriptor, nothing reads it       : {c1[0]}/{n} ticks act, {c1[1]}/{n} traj")
    print(f"C2 neutered with NO dialogue               : {c2[0]}/{n} ticks act, {c2[1]}/{n} traj")
    print(f"C3 determinism, two identical wired runs   : {c3[0]}/{n} ticks act, {c3[1]}/{n} traj")

    if n_max != 0.0:
        print("CONTROL FAILED: the neuter leaked -- B still read the signal, no verdict is sound")
        return 2
    for name, (a, t) in (("C1 sham", c1), ("C2 no-dialogue", c2), ("C3 determinism", c3)):
        if a or t:
            print(f"CONTROL FAILED: {name} diverged, so A vs B divergence is not attributable")
            return 2
    if not w_live:
        print(
            "NOT-ELIGIBLE: the signal never rose, so the dialogue did not establish a false "
            "belief -- no charge verdict"
        )
        return 2
    if act or traj:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
