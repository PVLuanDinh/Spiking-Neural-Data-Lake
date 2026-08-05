#!/usr/bin/env python3
"""red_reachability -- a criterion that cannot fail is not a measurement (P8).

Built for specs/010-policy-compliance-audit. This is the executable half of P8's honesty guard and
of falsifier F11 (docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:1064):

    "Any success criterion ... turns out to have no reachable red -> That criterion is DELETED,
     not weakened."

WHAT IT CHECKS, cheapest and most decisive first:

  1. reads_system -- does the criterion take the system under test as a parameter? A zero-argument
     criterion is invariant to EVERY change to the system at every level of the computational
     hierarchy, which is unfalsifiable in the strict sense of Kleiner & Hoel 2020
     (arXiv:2006.07390). This alone is sufficient for NO_RED, and it is free, so it runs first.
  2. tautological self-assert -- does the module assert the very verdict it just computed
     (`present, _ = probe_x(); assert present`)? That is a check that cannot go red by construction.

WHAT IT DELIBERATELY DOES NOT DO: judge whether a criterion measures the RIGHT thing. A criterion
can be red-reachable and still be measuring something irrelevant. No gate can tell you a criterion
is meaningful, which is why P8's guard is review-enforced and this instrument is only its cheapest
half.

UNKNOWN IS A FIRST-CLASS OUTCOME, never folded into pass (Standing Invariant 3). Reporting UNKNOWN
as NO_RED would make this gate commit the exact sin it exists to catch.

Reproduce:
    python tools/red_reachability.py            # report, exit 0
    python tools/red_reachability.py --assert   # exit 1 on any unexempted NO_RED
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

#: A function is a CRITERION if its name matches one of these shapes. Kept NARROW on purpose: the
#: method-dormancy-gate's recorded lesson is that widening the scan re-introduces a
#: rubber-stamp-sized list, and a narrow gate that holds beats a broad one exempted into
#: meaninglessness.
CRITERION_PREFIXES = ("probe_",)
CRITERION_SUFFIXES = ("_present", "_verdict", "_is_present")

#: Verbs that indicate the function returns a pass/fail judgement rather than a reading.
VERDICT_NAMES = {"verdict", "present", "passes", "is_present"}

#: EXEMPT follows dormancy_gate.EXEMPT: a disposition AND a reason, and every entry must still match
#: a real violation or the gate fails loudly -- an exemption that blesses nothing reads as coverage.
EXEMPT = {}


def _is_criterion(name):
    return (
        name.startswith(CRITERION_PREFIXES)
        or name.endswith(CRITERION_SUFFIXES)
        or name in VERDICT_NAMES
    )


def _real_params(fn):
    """Parameters excluding `self`/`cls`. A criterion with none reads no system."""
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.posonlyargs]
    args += [a.arg for a in fn.args.kwonlyargs]
    return [a for a in args if a not in ("self", "cls")]


def _tautological_asserts(tree, criterion_names):
    """Find `x, _ = probe_y(); assert x` -- an assert on a value the same scope just computed.

    This is the shape that kept `consciousness_probe` green: `main()` asserted every verdict it had
    itself produced, so the check could not go red no matter what the probes reported, and the one
    marker with a real red had to have its seed pinned to a passing value to keep the gate quiet.
    """
    produced = {}
    out = {}
    # ONE pass over the whole tree, keyed by (line, var), because `ast.walk` yields Module AND the
    # FunctionDef inside it -- walking both double-counted every hit on the first run (8 reported,
    # 4 real).
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            fname = _called_name(stmt.value)
            if fname and fname in criterion_names:
                for tgt in stmt.targets:
                    for nm in _bound_names(tgt):
                        produced[nm] = fname
        elif isinstance(stmt, ast.Assert) and _is_bare_truthiness(stmt.test):
            for nm in _referenced_names(stmt.test):
                if nm in produced:
                    out[(getattr(stmt, "lineno", 0), nm)] = produced[nm]
    return [(line, nm, fn) for (line, nm), fn in sorted(out.items())]


def _is_bare_truthiness(test):
    """Is this `assert verdict` / `assert all(verdicts)` rather than `assert verdict == expected`?

    THE DISTINCTION IS THE WHOLE POINT, and the first run got it wrong by flagging this tool's own
    fixed code. `assert present` says "the criterion passed" and can never go red -- that is the
    shape that kept `consciousness_probe` green for its whole life. But `assert acc is None` and
    `assert 0.0 < frac < 1.0` assert a FALSIFIABLE PROPERTY of the result (it reports UNKNOWN
    without a brain; the marker reaches both verdicts across seeds) and are exactly what a repaired
    self-check should look like.

    Flagging both would punish the fix as hard as the defect, and would teach the next reader to
    stop asserting anything about a criterion's output. Only bare truthiness counts.
    """
    if isinstance(test, ast.Name):
        return True
    if isinstance(test, ast.Call) and _called_name(test) in ("all", "any"):
        return True
    return False


def _called_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _bound_names(tgt):
    if isinstance(tgt, ast.Name):
        return [tgt.id]
    if isinstance(tgt, (ast.Tuple, ast.List)):
        out = []
        for e in tgt.elts:
            out.extend(_bound_names(e))
        return out
    return []


def _referenced_names(node):
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


def scan():
    """Return (criteria, tautologies). criteria: list of dicts, one per discovered criterion."""
    criteria = []
    tautologies = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_criterion(
                node.name
            ):
                names.add(node.name)
                params = _real_params(node)
                criteria.append(
                    {
                        "name": f"{path.stem}.{node.name}",
                        "file": rel,
                        "line": node.lineno,
                        "params": params,
                        "reads_system": bool(params),
                    }
                )
        if names:
            for line, var, fn in _tautological_asserts(tree, names):
                tautologies.append({"file": rel, "line": line, "var": var, "criterion": fn})
    return criteria, tautologies


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_assert = "--assert" in argv

    criteria, tautologies = scan()

    no_red = [c for c in criteria if not c["reads_system"] and c["name"] not in EXEMPT]
    ok = [c for c in criteria if c["reads_system"]]

    print("=" * 78)
    print("red_reachability -- a criterion that cannot fail is not a measurement (P8)")
    print("=" * 78)
    n_files = sum(1 for _ in SRC.rglob("*.py"))
    print(f"scanned {n_files} files under src/, found {len(criteria)} criteria")
    print()

    print("--- NO_RED: zero-argument criteria (invariant to the system under test) ---")
    if not no_red:
        print("  (none)")
    for c in no_red:
        print(f"  {c['name']:44s} {c['file']}:{c['line']}")
    print()

    print("--- TAUTOLOGICAL ASSERTS: assert on a verdict the same scope just computed ---")
    if not tautologies:
        print("  (none)")
    for t in tautologies:
        print(f"  {t['file']}:{t['line']}  assert {t['var']}  <- from {t['criterion']}()")
    print()

    print("--- READS THE SYSTEM (has a reachable red at this gate's resolution) ---")
    for c in ok:
        print(f"  {c['name']:44s} ({', '.join(c['params'])})")
    print()

    stale = sorted(set(EXEMPT) - {c["name"] for c in criteria})
    print(
        f"NO_RED: {len(no_red)}   TAUTOLOGICAL: {len(tautologies)}   "
        f"READS_SYSTEM: {len(ok)}   EXEMPT: {len(EXEMPT) - len(stale)}"
    )
    if stale:
        print("  STALE EXEMPTIONS (match no real criterion -- delete them):")
        for name in stale:
            print(f"    {name}")
    print()
    print("  NOTE: this gate proves a criterion CAN be falsified, never that it measures the right")
    print("  thing. That half is review-enforced (P8) and no instrument can replace it.")
    print()

    failed = bool(no_red or tautologies or stale)
    print("gate FAIL: at least one criterion cannot report a failure" if failed else "gate OK")
    return 1 if (do_assert and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
