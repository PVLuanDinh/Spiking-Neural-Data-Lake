"""social_tom_refute_probe -- adversarial re-measurement of Social's one band-41+ claim.

WHAT IS UNDER ATTACK. `docs/coverage/Social.md` scores `tom_belief_guided_caution` at band 61 on
`social_tom_caution_60.txt`: an exit-0 neuter-A/B of `arm._social_risk` reporting act/hold 0/60 and
"arm trajectory changed by the signal 24/60 ticks". That artifact is genuine -- it reproduces under
`coverage_score.py --rerun` -- so the attack is not forgery. It is whether 24/60 trajectory ticks,
with the act/hold decision unmoved, is a band-61 "changes a decision, action, or learning update".

`docs/coverage/Executive.md:23-52` established in the same sweep that charge_probe's trajectory
count is ARITHMETICALLY DETERMINED by the divergence-onset tick (`n == N - onset`, verified on seven
A/Bs) because the arm's position integrates its own motion. The Social probe prints only that
column, so this probe adds the observables that decide the band:

  * the SELECTED ACTION flip count -- not onset-implied, because the arm re-chooses every tick and
    actions can re-converge while position cannot;
  * the onset tick and whether the trajectory divergence is contiguous from it, which settles
    whether 24/60 is 24 effects or one effect restated 24 times;
  * the same run at a second and third SEED, which is the caveat Executive.md:240-243 names as the
    biggest one on every 61 in this sweep and which nobody had spent.

MODES.
  arm    [n] [seed]  the auditor's own neuter (data descriptor on `arm._social_risk`), plus the
                     observables above. Same dialogue, same construction path, same exit contract.
  source [n] [seed]  neuter `Brain._false_beliefs` -- the ToM READOUT one layer upstream -- instead
                     of the arm attribute. If the two produce the same behavioural fingerprint they
                     are one experiment, and the readout rows and the wire row are one cause scored
                     more than once (the defect that cost Executive 9 points).
  trigger            rubric rule 4: feed near-miss paraphrases of the SAME false-belief content
                     through a live brain's `respond()` and count which ones raise the signal at
                     all, and to what value. Separates the belief from the hand-written surface
                     regex at brain.py:83-102.

NEUTER. Data descriptor on the owning CLASS, restored in a finally block, exactly as the audited
probe does it -- a dict-subclass or a post-tick zero leaks (charge_probe.py:54-67,
social_tom_ab_probe.py:12-19). Both arms run the IDENTICAL dialogue at the same tick, so everything
else a chat turn does is held constant.

EXIT CONTRACT, preserved from charge_probe/social_tom_ab_probe so artifacts stay comparable:
  0 BEHAVIOURAL CHARGE OK   1 CONSUMED-BUT-NOT-DECISIVE   2 NOT-ELIGIBLE / CONTROL FAILED

  PYTHONPATH=src python docs/coverage/artifacts/social_tom_refute_probe.py arm 60 0
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cortex.brain import Brain, make_brain  # noqa: E402

# The audited probe's dialogue, byte-identical, so this measures the same event it did.
DIALOGUE = (
    "i put the marble in the basket.",
    "someone moved the marble to the table while i was away.",
)

# Near-miss paraphrases of the SECOND turn. Same belief content in every one: an object moved to a
# new place while the speaker was absent, so the speaker holds a false belief about where it is.
PARAPHRASE = (
    ("someone moved the marble to the table while i was away.", "the audited wording"),
    ("while i was away, someone moved the marble to the table.", "idiom first"),
    ("someone put the marble on the shelf when i wasn't looking.", "different verb+idiom, in list"),
    ("the marble was moved to the table while i was away.", "passive voice"),
    ("someone took the marble out of the basket while i was away.", "verb not in the list"),
    ("my sister hid the keys in the drawer behind my back.", "third-party subject"),
    ("someone moved the marble to the table while you were away.", "absent party is the brain"),
    ("someone relocated the marble to the table while i was away.", "synonym verb"),
    ("the marble is on the table now, i never saw it move.", "no put/move clause"),
    ("someone moved my marble to the kitchen table while i was away.", "two-word location"),
)


def _neuter(mode):
    """(owner class, attribute) the mode pins to a constant via a data descriptor."""
    return {"arm": (None, "_social_risk"), "source": (Brain, "_false_beliefs")}[mode]


def run(mode, n, seed, neuter, chat=True, sham=False):
    cls, attr = _neuter(mode)
    if sham:
        attr = attr + "_sham"
    zero = [] if attr.startswith("_false_beliefs") else 0.0
    patched, saved, had = None, None, False
    try:
        if neuter and cls is not None:
            patched, had = cls, attr in cls.__dict__
            saved = cls.__dict__.get(attr)
            setattr(cls, attr, property(lambda self: zero, lambda self, v: None))
        b = make_brain(seed=seed, persist=False)
        arm = b.brain.arm
        if neuter and cls is None:  # the arm's class is only knowable after construction
            patched, had = type(arm), attr in type(arm).__dict__
            saved = type(arm).__dict__.get(attr)
            setattr(patched, attr, property(lambda self: zero, lambda self, v: None))
        stopped, pos, act, sig = [], [], [], []
        try:
            for i in range(n):
                if chat and i == n // 2:
                    for line in DIALOGUE:
                        b.respond(line)
                b.tick()
                st = b.state()
                stopped.append(bool(st.get("arm_stopped")))
                pos.append(tuple(st.get("map_position") or ()))
                act.append((b._last_res or {}).get("action"))
                sig.append(float(getattr(arm, "_social_risk", 0.0) or 0.0))
        finally:
            b.close()
    finally:
        if patched is not None:
            if had:
                setattr(patched, attr, saved)
            else:
                delattr(patched, attr)
    return stopped, pos, act, sig


def diff(a, b):
    return sum(1 for p, q in zip(a, b, strict=True) if p != q)


def trigger(n_ticks=6):
    """Rule-4 reachability: which surface forms of one false-belief event raise the signal."""
    print("rubric rule 4 -- which SURFACE FORMS of the same false-belief content fire the wire")
    print(f"{'fires':>6}  {'fb':>3}  {'risk':>6}  form")
    fired = 0
    for text, why in PARAPHRASE:
        b = make_brain(seed=0, persist=False)
        try:
            b.respond("i put the marble in the basket.")
            b.respond(text)
            for _ in range(n_ticks):
                b.tick()
            fb = len(getattr(b, "_false_beliefs", None) or [])
            risk = float(getattr(b.brain.arm, "_social_risk", 0.0) or 0.0)
        finally:
            b.close()
        ok = bool(fb) and risk > 0.0
        fired += ok
        print(f"{'YES' if ok else 'no':>6}  {fb:>3}  {risk:>6.4f}  {text}   [{why}]")
    print(f"surface forms of one belief event that fire  : {fired}/{len(PARAPHRASE)}")
    return 0 if fired else 2


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "arm"
    if mode == "trigger":
        return trigger()
    if mode not in ("arm", "source"):
        print(f"unknown mode {mode!r}; known: arm, source, trigger")
        return 2
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    sA, pA, aA, gA = run(mode, n, seed, False)
    sB, pB, aB, gB = run(mode, n, seed, True)
    shamA = run(mode, n, seed, True, sham=True)

    act, traj, cmd = diff(sA, sB), diff(pA, pB), diff(aA, aB)
    onset = next((i for i in range(n) if pA[i] != pB[i]), None)
    cmd_at = [i for i in range(n) if aA[i] != aB[i]]
    contig = onset is not None and all(pA[i] != pB[i] for i in range(onset, n))
    c1 = (diff(sA, shamA[0]), diff(pA, shamA[1]), diff(aA, shamA[2]))

    print(f"mode / neutered                            : {mode} / {_neuter(mode)[1]}")
    print(f"seed / ticks / dialogue at tick            : {seed} / {n} / {n // 2}")
    print(
        f"signal A(wired) max / live ticks            : {round(max(gA), 4)} / "
        f"{sum(1 for v in gA if v)} of {n}"
    )
    print(
        f"signal B(neutered) max                     : {round(max(gB), 4)}  (held={max(gB) == 0.0})"
    )
    print(f"arm HOLD rate A(wired) vs B(neutered)      : {sum(sA) / n:.3f} vs {sum(sB) / n:.3f}")
    print(f"act/hold decision changed by the signal    : {act}/{n} ticks")
    print(f"arm trajectory changed by the signal       : {traj}/{n} ticks")
    print(f"SELECTED ACTION changed by the signal      : {cmd}/{n} ticks   at {cmd_at}")
    print(f"first trajectory divergence at tick        : {onset}")
    print(
        f"trajectory diff contiguous from onset      : {contig}   (N - onset = "
        f"{None if onset is None else n - onset})"
    )
    print(f"distinct actions A vs B                    : {len(set(aA))} vs {len(set(aB))}")
    print(
        f"C1 sham descriptor, nothing reads it       : {c1[0]}/{n} ticks act, {c1[1]} traj, "
        f"{c1[2]} action"
    )

    if max(gB) != 0.0 and mode == "arm":
        print("CONTROL FAILED: the neuter leaked -- B still read the signal")
        return 2
    if any(c1):
        print("CONTROL FAILED: the sham descriptor perturbed the run")
        return 2
    if not any(gA):
        print("NOT-ELIGIBLE: the signal never rose in the wired arm -- no charge verdict")
        return 2
    if act or traj or cmd:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
