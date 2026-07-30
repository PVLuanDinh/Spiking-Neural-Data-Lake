"""exec_refute_probe -- the adversarial pass over docs/coverage/Executive.md's band-41+ rows.

WHAT IT ADDS THAT THE AUDITOR'S PROBE CANNOT. `executive_ab_probe.py` neuters ONE signal against the
wired brain, so every one of its verdicts answers "does this signal matter at all". It cannot answer
the two questions the rubric's adversarial pass exists to ask:

  1. MARGINAL CONTRIBUTION. Given that signal Y is already scored, does neutering X ON TOP OF Y
     change anything Y did not already change? If not, X and Y are two probe points on ONE serial
     chain and scoring both is counting one cause twice -- the defect COVERAGE_BASELINE.md:12
     records as costing this category 9 points. Three arms: A(wired), B(Y neutered), C(Y+X
     neutered); the reported number is C vs B, not C vs A.
  2. SEED DEPENDENCE. Every figure in the audited file is seed 0, which the auditor names as its own
     biggest caveat. `seeds` re-runs a single-signal A/B at other seeds. A charge that exists only
     at seed 0 is a property of one trajectory, not of the wire.

It also tracks the INTERNAL reads a band-41-60 claim rests on (`_decision_margin`, `_control_gain`,
`_metacog_rel`) so "changes an internal read" is measured rather than argued from the call graph.

NEUTERS ARE THE AUDITOR'S OWN, IMPORTED, NOT REIMPLEMENTED. `executive_ab_probe._patch`,
`._pin_pfc_gain` and `._PinnedWeights` are used verbatim, so a difference between my numbers and
theirs cannot be a difference of neuter. Construction is the Policy-4 make_brain() path.

EXIT CODES -- note that `marginal` does NOT mean the same thing as a charge probe's exit code, and
the difference is deliberate:
  marginal  exit 0  X ADDS behaviour beyond Y      -> the two rows are separable, both stand
            exit 1  X adds NOTHING beyond Y        -> nested; one chain; merge the rows
            exit 2  the run was not eligible
  seeds     exit 0  the charge reproduces at every seed tried
            exit 1  at least one seed shows no behavioural divergence
  internal  exit 0  at least one tracked internal read differs
            exit 1  no tracked internal read differs

  PYTHONPATH=src python docs/coverage/artifacts/exec_refute_probe.py marginal <base> <extra> <ticks>
  PYTHONPATH=src python docs/coverage/artifacts/exec_refute_probe.py seeds <signal> <ticks> <s,s,s>
  PYTHONPATH=src python docs/coverage/artifacts/exec_refute_probe.py internal <signal> <ticks>
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

import executive_ab_probe as eap  # noqa: E402  the audited probe, reused verbatim

from snn_data_lake.cortex.brain import make_brain  # noqa: E402


def run(signals, n, seed=0):
    """One arm. `signals` is the set of signals neutered for the whole run ([] = the wired arm)."""
    eap._restore()
    for s in signals:
        eap._patch(s)  # no-op for pfc_gain / taskset, which are instance-level
    b = make_brain(seed=seed, persist=False)
    arm = b.brain.arm
    if "pfc_gain" in signals:
        eap._pin_pfc_gain(arm)
    if "taskset" in signals:
        b._drive_w = eap._PinnedWeights(b._drive_w, "taskset")
        b._drive_w["taskset"] = 0.0
    tr = {k: [] for k in ("stop", "pos", "act", "margin", "ctl", "rel", "elig")}
    for _ in range(n):
        b.tick()
        st = b.state()
        tr["stop"].append(bool(st.get("arm_stopped")))
        tr["pos"].append(tuple(st.get("map_position") or ()))
        tr["act"].append((b._last_res or {}).get("action"))
        tr["margin"].append(round(float(getattr(arm, "_decision_margin", 0.0) or 0.0), 9))
        tr["ctl"].append(round(float(getattr(arm, "_control_gain", 0.0) or 0.0), 9))
        tr["rel"].append(getattr(b, "_metacog_rel", None))
        # eligibility: the tracked activity attribute of the FIRST named signal (wired arm only)
        key = eap._ELIG[signals[0]][1] if signals else "_conflict"
        v = getattr(arm, key, None)
        if v is None:
            v = getattr(b, key, None)
        tr["elig"].append(tuple(v) if isinstance(v, list) else v)
    b.close()
    eap._restore()
    return tr


def _flips(a, b_, k):
    return sum(1 for x, y in zip(a[k], b_[k], strict=True) if x != y)


def _line(label, v, n):
    print(f"{label}".ljust(46) + f": {v}/{n} ticks")


def marginal(base, extra, n):
    A = run([], n)
    B = run([base], n)
    C = run([base, extra], n)
    print(f"mode                                          : marginal ({extra} given {base})")
    print("arm A                                         : wired")
    print(f"arm B                                         : {base} neutered")
    print(f"arm C                                         : {base} + {extra} neutered")
    print(f"neuter B                                      : {eap._ELIG[base][0]}")
    print(f"neuter C adds                                 : {eap._ELIG[extra][0]}")
    for lbl, x, y in (("B vs A", A, B), ("C vs A", A, C), ("C vs B (MARGINAL)", B, C)):
        _line(f"{lbl} selected action changed", _flips(x, y, "act"), n)
    for lbl, x, y in (("B vs A", A, B), ("C vs A", A, C), ("C vs B (MARGINAL)", B, C)):
        _line(f"{lbl} arm trajectory changed", _flips(x, y, "pos"), n)
    _line("C vs B (MARGINAL) act/hold changed", _flips(B, C, "stop"), n)
    _line("C vs B (MARGINAL) decision_margin changed", _flips(B, C, "margin"), n)
    _line("C vs B (MARGINAL) control_gain changed", _flips(B, C, "ctl"), n)
    _line("C vs B (MARGINAL) metacog_rel changed", _flips(B, C, "rel"), n)
    m = _flips(B, C, "act") + _flips(B, C, "pos") + _flips(B, C, "stop")
    if _flips(A, B, "act") == 0 and _flips(A, B, "pos") == 0:
        print("NOT-ELIGIBLE: arm B is identical to arm A, so this comparison tests nothing")
        return 2
    if m:
        print(f"SEPARABLE: {extra} changes behaviour that {base} did not already change")
        return 0
    print(
        f"NESTED: neutering {extra} on top of {base} changes NO action, NO trajectory and NO "
        f"act/hold -- one serial chain, not two faculties"
    )
    return 1


def seeds(signal, n, seed_list):
    print(f"mode                                          : seeds ({signal})")
    print(f"neuter                                        : {eap._ELIG[signal][0]}")
    worst = 0
    for s in seed_list:
        A = run([], n, seed=s)
        B = run([signal], n, seed=s)
        a, p, h = _flips(A, B, "act"), _flips(A, B, "pos"), _flips(A, B, "stop")
        act = sum(1 for x in A["elig"] if x != A["elig"][0])
        _line(f"seed {s} wired activity", act, n)
        _line(f"seed {s} selected action changed", a, n)
        _line(f"seed {s} arm trajectory changed", p, n)
        _line(f"seed {s} act/hold changed", h, n)
        if not (a or p or h):
            worst = 1
    if worst:
        print(f"SEED-DEPENDENT: {signal} showed no behavioural divergence at some seed")
        return 1
    print(f"SEED-ROBUST: {signal} diverged behaviourally at every seed tried")
    return 0


def internal(signal, n):
    A = run([], n)
    B = run([signal], n)
    print(f"mode                                          : internal ({signal})")
    print(f"neuter                                        : {eap._ELIG[signal][0]}")
    _line("selected action changed", _flips(A, B, "act"), n)
    _line("decision_margin changed", _flips(A, B, "margin"), n)
    _line("control_gain changed", _flips(A, B, "ctl"), n)
    _line("metacog_rel changed", _flips(A, B, "rel"), n)
    print(
        f"decision_margin distinct values A / B         : {len(set(A['margin']))} / "
        f"{len(set(B['margin']))}"
    )
    tot = _flips(A, B, "margin") + _flips(A, B, "ctl") + _flips(A, B, "rel")
    if tot:
        print("INTERNAL READ CHANGED")
        return 0
    print("NO TRACKED INTERNAL READ CHANGED")
    return 1


def tsdrive(n, seed_list):
    """Per-seed ELIGIBILITY for the taskset S-R readout, because `seeds taskset` returning 0/60 has
    two incompatible readings and the difference decides a band. If the readout's drive contribution
    is nonzero and the arm still does not diverge, the wire is CONSUMED-BUT-NOT-DECISIVE at that
    seed. If the contribution is zero, the neuter removed nothing and the seed says nothing at
    all -- reporting the second as the first is how a live faculty gets wrongly called dead
    (charge_probe.py:124-135)."""
    print("mode                                          : tsdrive (taskset eligibility per seed)")
    inert = 0
    for s in seed_list:
        eap._restore()
        b = make_brain(seed=s, persist=False)
        live, mx, pref = 0, 0.0, 0
        for _ in range(n):
            b.tick()
            tp = getattr(b, "_task_pref", None)
            w = float(b._drive_w.get("taskset", 0.0))
            if tp:
                pref += 1
                c = max(w * max(0.0, min(1.0, float(x))) for x in tp)
                mx = max(mx, c)
                if c > 1e-12:
                    live += 1
        b.close()
        _line(f"seed {s} _task_pref present", pref, n)
        _line(f"seed {s} taskset drive term > 0", live, n)
        print(f"seed {s} max taskset drive contribution".ljust(46) + f": {mx:.6f}")
        if live == 0:
            inert = 1
    if inert:
        print(
            "NOT-ELIGIBLE at some seed: the taskset drive term was zero, the neuter removed nothing"
        )
        return 2
    print("ELIGIBLE at every seed: the taskset drive term was nonzero on live ticks")
    return 0


def main():
    if len(sys.argv) < 2:
        print(
            "modes: marginal <base> <extra> <ticks> | seeds <signal> <ticks> <s,s> | "
            "internal <signal> <ticks> | tsdrive <ticks> <s,s>"
        )
        return 2
    mode = sys.argv[1]
    if mode == "marginal":
        return marginal(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    if mode == "seeds":
        return seeds(sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")])
    if mode == "internal":
        return internal(sys.argv[2], int(sys.argv[3]))
    if mode == "tsdrive":
        return tsdrive(int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")])
    print(f"unknown mode {mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
