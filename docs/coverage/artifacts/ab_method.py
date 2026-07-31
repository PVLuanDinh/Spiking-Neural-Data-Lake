"""Neuter-A/B for a named method: patch it to a no-op, re-run at the same seed, diff what moved.

`tools/charge_probe.py`'s neuter pins a DRIVE-SOURCE WEIGHT, which only applies to the eleven
undirected drive sources. Most learning rules are not drive sources, so they need this instead: run
the brain twice at the same seed, once with the rule intact and once with it replaced by a no-op,
and report which observables differ.

Two classes of observable, kept apart on purpose:

  BEHAVIOURAL  arm_stopped, map_position  -- what the brain DOES
  READS        every other numeric leaf of state() -- what the brain later READS

A rule can move the second without moving the first. That is the rubric's 41-60 band ("changes an
internal read") sitting underneath 61-80 ("changes a decision"), and collapsing them is how a live
faculty gets reported as inert. `AffectiveAppraisal.learn` is the worked case: it charges `w_aro`
while the only non-telemetry consumer reads `valence`, so the pools are disjoint and every
behavioural hash stays byte-identical.

Counts print as `N of M`, never `N/M`: coverage_score's `_fingerprint` harvests every bare ratio in
a probe's stdout, so a ratio here would append an observable to every recorded artifact's
fingerprint and fail them all under --rerun.

Usage:
  python scratchpad/ab_method.py <module> <Class> <method> [ticks]
  e.g.  python scratchpad/ab_method.py amygdala.affective_appraisal AffectiveAppraisal learn 400
"""

import hashlib
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from snn_data_lake.cortex.brain import make_brain  # noqa: E402

_BEHAVIOURAL = ("arm_stopped", "map_position")


def _leaves(o, prefix="", out=None, depth=0):
    """Numeric leaves of a snapshot, INCLUDING numbers inside tuples and lists.

    The first version walked only int/float/bool/dict and reported `0 of 120` for a rule whose one
    measured effect is `appraised`, a (valence, arousal) TUPLE. The null was an artifact of the
    harness's own blindness, not a property of the brain -- the same shape of defect this whole
    exercise exists to catch, reproduced inside the measuring device on its first run.
    """
    out = {} if out is None else out
    if depth > 3 or not isinstance(o, dict):
        return out
    for k, v in o.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            _leaves(v, key + ".", out, depth + 1)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, bool):
                    out[f"{key}[{i}]"] = float(x)
                elif isinstance(x, (int, float)):
                    out[f"{key}[{i}]"] = float(x)
    return out


def run(mod, cls, meth, n, neuter):
    m = importlib.import_module(f"snn_data_lake.{mod}")
    target = getattr(m, cls)
    original = getattr(target, meth)
    calls = [0]

    def counted(self, *a, **kw):
        calls[0] += 1
        return original(self, *a, **kw)

    def noop(self, *a, **kw):
        calls[0] += 1
        return None

    setattr(target, meth, noop if neuter else counted)
    try:
        b = make_brain(seed=0, persist=False)
        beh, reads = hashlib.blake2b(digest_size=8), []
        for _ in range(n):
            b.tick()
            st = b.state()
            beh.update(repr([st.get(k) for k in _BEHAVIOURAL]).encode())
            reads.append(_leaves(st))
        b.close()
        return beh.hexdigest(), reads, calls[0]
    finally:
        setattr(target, meth, original)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    mod, cls, meth = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 400

    hA, rA, nA = run(mod, cls, meth, n, False)
    hB, rB, nB = run(mod, cls, meth, n, True)

    moved_ticks, totals = 0, {}
    for a, c in zip(rA, rB, strict=True):
        moved = {k: abs(a[k] - c[k]) for k in set(a) & set(c) if a[k] != c[k]}
        if moved:
            moved_ticks += 1
        for k, v in moved.items():
            totals[k] = totals.get(k, 0.0) + v

    print(f"target                    : {mod}.{cls}.{meth}")
    print(f"ticks                     : {n}")
    print(f"invocations A(wired)      : {nA}")
    print(f"invocations B(neutered)   : {nB}")
    print(f"behavioural digest A      : {hA}")
    print(f"behavioural digest B      : {hB}")
    print(f"behaviour changed         : {'YES' if hA != hB else 'NO'}")
    print(f"internal reads changed    : {moved_ticks} of {n} ticks")
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {k:<44} total |delta| {v:.6g}")

    if nA == 0:
        print("NEVER-CALLED: the rule did not run on this path -- a neuter proves nothing here")
        return 2
    if hA != hB:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    if moved_ticks:
        print(
            f"STATE CHANGED on {moved_ticks} of {n} ticks, behaviour did not. The fields above moved."
        )
        print(
            "NOT a band claim on its own. This harness reads state(), which mixes values the brain "
            "later CONSUMES with values only telemetry shows. Rubric 41-60 requires the former: "
            "before citing this as 41-60 evidence, grep each field above for a NON-TELEMETRY reader. "
            "If every reader is state()/a log/the cockpit, the honest verdict is CALLED-BUT-DISCARDED."
        )
        return 1
    print("CHANGED NOTHING: called, and no observable differs with it neutered")
    return 1


if __name__ == "__main__":
    sys.exit(main())
