#!/usr/bin/env python3
"""instrument_guard -- the instruments are part of the brain too (P4c), so they get a gate.

WHY THIS EXISTS. P4c is explicit that there is no "outside the brain" in this project -- *"not
`research/`, not `training/`, not `integrations/`, not `gcp/` or `infra/`, not `tools/`, not a
script at the repo root"* -- and it names the shape of the failure: `learn_daemon.py` sat at the
repo root, *"in no REGION, in neither `run_selfchecks.py` list, not a `dormancy_gate` entrypoint,
with no self-check and not one assert"*, and promoting it *"immediately exposed three latent bugs
that being ungoverned had hidden -- `os.chdir()`, `sys.path.insert()` and
`os.environ.setdefault("NORD_GPU", "1")` all ran at MODULE SCOPE"*.

`tools/` is the remainder of that debt, and it is the tier where being wrong is worst: an
instrument does not compute the brain, it computes the NUMBER the brain is judged by. A wrong
instrument is worse than no instrument, because it returns a confident answer.

BUILT AND COUNTED FIRST, per the standing invariant. The counted first run on this tree, 2026-08-04:

    P4-CONSTRUCTION      1  ->  tools/arc_baseline.py called `Brain()` directly
    CWD-DEPENDENT PATH   5  ->  3 in tools/, 2 in tests/, all `sys.path.insert(0, "src"|"tools")`
    MODULE-SCOPE CHDIR   0
    MODULE-SCOPE SEED    0

Six findings across SIX files -- `arc_baseline` (the P4 one) plus `ab_connectome_delay`,
`fanout_bisect`, `deep_fingerprint`, `tests/test_dependency_ledger` and `tests/test_red_reachability`
(one relative `sys.path` entry each). This block read "across five files" until it was recounted
2026-08-05: no file carried two findings, so the file count is the finding count. A gate whose own
headline census is off by one is the thing it exists to catch, one level up.

Six findings is a work list, not an allowlist, so the rule is adoptable in this scope. All six were
cleared in the same pass; the gate asserts clean today and its own self-check constructs the RED for
every rule, so "0 findings" is a measurement rather than a mirror.

The CENSUS half of the same run: 29 tools/ modules, 15 with a self-check, an assert, or a pytest
guard, and 14 with none of the three. That number is REPORTED, never asserted -- see below.

It is registered in `tests/test_instrument_guard.py` rather than in `run_selfchecks.py` or
`.github/workflows/ci.yml`, so it asserts on every `pytest` run without editing a shared file. That
test file also fires each rule against a fixture, and it is what caught this gate's own worst bug:
`pathlib.PurePath("/opt/x").is_absolute()` is FALSE on Windows, so rule 2 reported a POSIX absolute
path as cwd-dependent and the gate's verdict depended on which CI runner executed it.

THE FOUR RULES, and each one is a defect this tree actually produced:

  1. P4 CONSTRUCTION. A file under tools/ or tests/ may not call `Brain(...)`; the one sanctioned
     path is `cortex.brain.make_brain()`. `tools/charge_probe.py` records fixing exactly this in
     itself -- *"the instrument that enforced P3 was quietly violating P4"* -- and
     `tools/arc_baseline.py` was still doing it on 2026-08-04, in the tool that produces the
     published ARC "before" number. `substrate/replication_guard.py` cannot see this: it asserts
     the CLASS is defined once, never that a consumer stopped calling it.
     **BOTH SPELLINGS, since 2026-08-05.** The first version matched only a bare `ast.Name`, so
     `from snn_data_lake.cortex import brain` + `brain.Brain()` -- the most natural way to write the
     same violation -- walked straight past it. MEASURED with a neuter-A/B before and after: the
     attribute form returned `[]` from a gate that fired on the bare form in the same run. Nothing
     in the tier used that spelling, so this closed a latent hole rather than a live finding; a rule
     that can be evaded by an import style is not a rule.

  2. CWD-DEPENDENT sys.path. A RELATIVE literal on `sys.path` is resolved against whatever the cwd
     is at import time. Three tools carried `sys.path.insert(0, "src")`, so they imported the brain
     only when run from the repo root and silently imported something else (or nothing) otherwise.
     Two test modules carried `sys.path.insert(0, "tools")` -- worse, because `tests/test_selfchecks.py`
     runs every module's `__main__` under `monkeypatch.chdir(tmp_path)` IN THE SAME PROCESS, so the
     entry survives into a directory where an unrelated `tools/` would shadow the real one.

  3. MODULE-SCOPE `os.chdir`. P4c's own worked example. `run_selfchecks.py` runs part of the suite
     in ONE process, so a module-scope chdir moves the working directory for every module after it.

  4. MODULE-SCOPE GLOBAL SEEDING. `random.seed(...)` / `np.random.seed(...)` at import time is the
     determinism invariant's named prohibition -- *"use `random.Random(seed)` (per-instance). Never
     module-level import-time seeding"* -- and in a shared pytest process it silently re-seeds
     whatever ran before it.

`os.environ.setdefault(...)` at module scope is deliberately NOT a rule. Six tools use it for
`NORD_PERSIST`, and for a leaf ENTRYPOINT that nothing imports it is the correct way to state a
precondition before the first `snn_data_lake` import -- `tools/lang_coverage.py` even has a test
asserting the ORDERING. Flagging it would have produced six findings that are all correct, which is
the allowlist shape that kills a gate here. The learn_daemon incident was about a module the
IN-PROCESS GATE imports; these are not, and the distinction is the reason this rule is absent
rather than an oversight.

WHAT IT DOES NOT DO. It does not run the instruments, and it does not check that a tool's numbers
are right -- no static pass can. It also does not FAIL a tool for having no self-check: that half is
printed as a CENSUS, because turning it into an assert would pressure someone to add a self-check
that asserts nothing, which is the same theatre P3's honesty guard forbids one tier down.

Reproduce:
    python tools/instrument_guard.py            # report + census, exit 0
    python tools/instrument_guard.py --assert   # exit 1 on any unexempted violation
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The instrument tiers. Both are ungoverned by the three P3 gates and both are part of the brain.
TIERS = ("tools", "tests")

#: Stack classes that must be obtained through their sanctioned constructor rather than called.
#: `AutonomousArm` is deliberately ABSENT: it is the arm, not the brain, and several arm tests
#: legitimately construct one in isolation to exercise the E-STOP without standing up a cortex.
#: P4's sanctioned-path clause is about `cortex.brain.make_brain()`, so this rule is scoped to what
#: that clause actually says.
CONSTRUCTED_VIA_FACTORY = {"Brain": "cortex.brain.make_brain()"}

#: Module-scope calls that mutate state shared by every module imported after them.
BANNED_MODULE_SCOPE = {
    "os.chdir": "moves the working directory for every module after it in a shared process",
    "random.seed": "module-level import-time seeding; use random.Random(seed) per instance",
    "np.random.seed": "module-level import-time seeding; use np.random.default_rng(seed)",
    "numpy.random.seed": "module-level import-time seeding; use np.random.default_rng(seed)",
}

#: Every tag this gate can report. Named once so the self-check can state what fraction of the tags
#: it actually fired a red for, rather than counting its own fixture table against itself.
RULE_NAMES = ("P4-CONSTRUCTION", "CWD-DEPENDENT-PATH", "MODULE-SCOPE-MUTATION", "SYNTAX")

#: key: "relpath -> rule", value: why this site legitimately stays. Follows dormancy_gate.EXEMPT's
#: pattern -- a disposition AND a reason -- and every entry must still match a real violation, so an
#: exemption that blesses nothing FAILS instead of reading as coverage.
EXEMPT: dict[str, str] = {}


def _dotted(node):
    """`os.chdir` from an ast.Attribute/Name call target, else None."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _module_scope_nodes(tree):
    """Every node that executes at IMPORT time: module body, minus function and class bodies.

    A call inside a `def` runs when the function is called, which is the caller's problem, not the
    importer's. A call inside `if __name__ == "__main__":` is deliberately INCLUDED as module scope
    -- it does not run on import, but keeping it out would need a second traversal for no benefit,
    and none of the four banned calls is acceptable there either.
    """
    out = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        out.append(node)
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return out


def _relative_syspath_literal(node):
    """`sys.path.insert(0, "src")` / `sys.path.append("tools")` with a RELATIVE literal -> the arg.

    A joined or `__file__`-derived expression is not a Constant, so it never matches -- which is the
    whole point: the fix for this rule is to derive the path from `__file__`.
    """
    if not isinstance(node, ast.Call):
        return None
    name = _dotted(node.func)
    if name not in ("sys.path.insert", "sys.path.append"):
        return None
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            value = arg.value
            if value and not _is_absolute(value):
                return value
    return None


def _is_absolute(value):
    """Absolute under EITHER convention, because the gate must not depend on the host it runs on.

    The first version used `pathlib.PurePath(value).is_absolute()`, which resolves to
    `PureWindowsPath` on Windows -- where `"/opt/whatever"` is ROOTED BUT NOT ABSOLUTE (it has no
    drive), so `is_absolute()` returns False and the gate reported a POSIX absolute path as a
    cwd-dependent relative one. CI runs {ubuntu, windows}, so that is a gate whose verdict depends
    on the runner: green on Linux, red on Windows, for byte-identical source. Caught by this file's
    own `test_an_absolute_syspath_entry_is_not_flagged` -- the fixture asserting the FIX is not
    punished, which is why that half of a self-check is not optional.
    """
    return (
        pathlib.PurePosixPath(value).is_absolute() or pathlib.PureWindowsPath(value).is_absolute()
    )


def scan():
    """(violations, census). A violation is (relpath, lineno, rule, detail)."""
    violations, census = [], []
    for tier in TIERS:
        base = ROOT / tier
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError as exc:
                violations.append((rel, exc.lineno or 0, "SYNTAX", str(exc)))
                continue

            names = {
                n.name
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            census.append(
                {
                    "file": rel,
                    "tier": tier,
                    "has_main": "main" in names,
                    "has_selfcheck": bool(names & {"_selfcheck", "selfcheck"}),
                    "asserts": sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert)),
                }
            )

            # rule 1 -- P4 construction, anywhere in the file, in EITHER spelling: a bare
            # `Brain()` and a qualified `brain.Brain()` are the same violation, and keying on
            # `ast.Name` alone let the second one through (see rule 1 in the docstring). The
            # dotted form is matched on its LAST component, so `cortex.brain.Brain()` is caught
            # too. A call on an expression (`factory().Brain()`) still escapes -- `_dotted`
            # needs a Name at the base -- which is a known and deliberate floor, not a claim of
            # completeness.
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = _dotted(node.func)
                if called is None:
                    continue
                factory = CONSTRUCTED_VIA_FACTORY.get(called.rsplit(".", 1)[-1])
                if factory:
                    violations.append(
                        (
                            rel,
                            node.lineno,
                            "P4-CONSTRUCTION",
                            f"calls {called}(...) directly; the sanctioned path is {factory}",
                        )
                    )

            # rules 2-4 -- import-time process state
            for node in _module_scope_nodes(tree):
                literal = _relative_syspath_literal(node)
                if literal is not None:
                    violations.append(
                        (
                            rel,
                            node.lineno,
                            "CWD-DEPENDENT-PATH",
                            f"sys.path entry {literal!r} is RELATIVE -- resolved against the cwd at "
                            "import time; derive it from __file__",
                        )
                    )
                if isinstance(node, ast.Call):
                    called = _dotted(node.func)
                    why = BANNED_MODULE_SCOPE.get(called)
                    if why:
                        violations.append(
                            (rel, node.lineno, "MODULE-SCOPE-MUTATION", f"{called}(): {why}")
                        )
    return violations, census


def guarded(entry, test_stems):
    """Can this instrument tell you it is broken? self-check, assert, or a pytest file that drives it.

    A TOOL GUARDED BY A PYTEST FILE COUNTS AS CHECKED. The first run of this census called
    `red_reachability.py` and `dependency_ledger.py` SILENT while `tests/test_red_reachability.py`
    and `tests/test_dependency_ledger.py` fire every one of their buckets against a fixture built to
    trip them. Counting those as unguarded would have been a false finding in the direction that
    matters -- it would have invited someone to bolt a self-check onto a tool that is already the
    best-verified in the tier. Caught by disbelieving this gate's own first output.

    PREFIX, not exact: `coverage_score.py` is guarded by three files (`test_coverage_score_citation`
    / `_cli` / `_fingerprint`) and an exact `test_<stem>` match called the most-tested instrument in
    the tier SILENT. The prefix must still cover the WHOLE stem plus a `_`, so `test_arc_fix.py` does
    not launder `arc_fixdepth_control.py`.

    At module scope on purpose (2026-08-05): it was a closure inside `main()`, so the one judgement
    this gate makes -- as opposed to parses -- had no test that could reach it, and the test that
    claimed to cover it asserted only its own precondition.
    """
    stem = pathlib.Path(entry["file"]).stem
    by_test = any(t == f"test_{stem}" or t.startswith(f"test_{stem}_") for t in test_stems)
    return bool(entry["has_selfcheck"] or entry["asserts"] or by_test)


def census_split(census):
    """(tools, checked, silent) over a census -- the CENSUS half, drivable without running main()."""
    tools = [c for c in census if c["tier"] == "tools"]
    test_stems = {pathlib.Path(c["file"]).stem for c in census if c["tier"] == "tests"}
    checked = [c for c in tools if guarded(c, test_stems)]
    silent = [c for c in tools if not guarded(c, test_stems)]
    return tools, checked, silent


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_assert = "--assert" in argv
    violations, census = scan()

    print("=" * 78)
    print("instrument_guard -- the instruments are part of the brain too (P4c)")
    print("=" * 78)
    print(f"scanned {len(census)} modules across {', '.join(t + '/' for t in TIERS)}")
    print()

    live_exempt = set()
    unexempt = []
    for rel, lineno, rule, detail in violations:
        key = f"{rel} -> {rule}"
        if key in EXEMPT:
            live_exempt.add(key)
            continue
        unexempt.append((rel, lineno, rule, detail))

    print("--- VIOLATIONS ---")
    if not unexempt:
        print("  (none)")
    for rel, lineno, rule, detail in unexempt:
        print(f"  {rule:<22} {rel}:{lineno}")
        print(f"      {detail}")
    print()

    # The CENSUS half. Printed, never asserted: an instrument that has no self-check is a fact worth
    # knowing, and turning it into a failure would buy a self-check that asserts nothing. The
    # judgement itself lives in `guarded()` / `census_split()` at module scope, so a test can drive
    # it directly.
    tools, checked, silent = census_split(census)
    print("--- CENSUS: which instruments can tell you they are broken? ---")
    print(f"  tools/ modules                         : {len(tools)}")
    print(f"  self-check, assert, or a pytest guard  : {len(checked)} of {len(tools)}")
    print(f"  with none of the three (SILENT)        : {len(tools) - len(checked)} of {len(tools)}")
    print("  A silent instrument is not a defect by itself -- most of these are one-shot")
    print("  measurement scripts whose output IS the check. It is open P4c work, and the")
    print("  count is here so it is countable rather than assumed.")
    for c in silent:
        print(f"      SILENT  {c['file']}")
    print()

    stale = sorted(set(EXEMPT) - live_exempt)
    # An exemption's REASON is half of it. `dormancy_gate.EXEMPT`'s pattern is a disposition AND a
    # reason, so "fine" is not an entry -- and this is checked in the gate rather than only in
    # tests/test_instrument_guard.py, because a rule that lives only in a test file is not enforced
    # by the tool that ships.
    thin = sorted(k for k, reason in EXEMPT.items() if not reason or len(reason.strip()) <= 20)
    # Deliberately "N of M", never "N/M": coverage_score's `_fingerprint` harvests every bare ratio
    # in a probe's stdout positionally into the artifact fingerprint.
    print(
        f"VIOLATIONS: {len(unexempt)}   EXEMPT: {len(live_exempt)}   "
        f"SILENT INSTRUMENTS: {len(tools) - len(checked)} of {len(tools)}"
    )
    for key in sorted(live_exempt):
        print(f"  EXEMPT {key}: {EXEMPT[key]}")
    if stale:
        print("  STALE EXEMPTIONS (they match no real violation -- delete them):")
        for key in stale:
            print(f"    {key}")
    if thin:
        print("  NO DISPOSITION (an exemption owes a reason, not an assurance):")
        for key in thin:
            print(f"    {key}: {EXEMPT[key]!r}")
    print()
    print("  NOTE: this gate never RUNS an instrument, so it proves an instrument is well-formed,")
    print("  never that its number is right. No static pass can do the second.")
    print()

    failed = bool(unexempt or stale or thin)
    print("gate FAIL: an instrument is not governed" if failed else "gate OK")
    return 1 if (do_assert and failed) else 0


# --------------------------------------------------------------------------------------------
# Self-check. Every rule gets its RED CONSTRUCTED AND RUN (P8): a gate that has only ever been seen
# green is indistinguishable from a gate that cannot see anything, and this repo has shipped three
# of those. The fixtures below are the exact shapes the four rules exist to catch.
# --------------------------------------------------------------------------------------------
#: label -> (source that must trip it, rule it must be reported under). Keyed by LABEL rather than
#: by rule since 2026-08-05, because two of the four rules share the MODULE-SCOPE-MUTATION tag and a
#: rule-keyed table could hold only one red per tag: the seeding rule and the qualified-`Brain()`
#: spelling had no red here at all, and the self-check still printed "3 of 3 rules". A red per RULE
#: is the claim; a red per tag is not the same thing.
_RED = {
    "P4-CONSTRUCTION (bare name)": (
        "from snn_data_lake.cortex.brain import Brain\n_B = Brain()\n",
        "P4-CONSTRUCTION",
    ),
    "P4-CONSTRUCTION (qualified)": (
        "from snn_data_lake.cortex import brain\n_B = brain.Brain()\n",
        "P4-CONSTRUCTION",
    ),
    "CWD-DEPENDENT-PATH": ('import sys\nsys.path.insert(0, "src")\n', "CWD-DEPENDENT-PATH"),
    "MODULE-SCOPE-MUTATION (os.chdir)": ('import os\nos.chdir("/tmp")\n', "MODULE-SCOPE-MUTATION"),
    "MODULE-SCOPE-MUTATION (random.seed)": (
        "import random\nrandom.seed(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    "MODULE-SCOPE-MUTATION (np.random.seed)": (
        "import numpy as np\nnp.random.seed(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    # An unparseable instrument is reported, never skipped: `ast.parse` failing is the one way a
    # file can carry every defect above and be counted clean, which is the silent-pass shape.
    "SYNTAX": ("def broken(\n", "SYNTAX"),
}
# The REPAIRED shape of all three reds, plus the two things that must NOT be flagged: a module-scope
# `os.environ.setdefault` (deliberate, see the docstring) and an `os.chdir` inside a function body
# (runs when called, not on import).
#
# NOTE what is deliberately absent: a `Brain()` call inside a `def`. The first version of this
# fixture had one, on the theory that "inside a function" excused it -- and the gate correctly
# fired. It should: `tools/arc_baseline.py`'s violation was inside `live_brain_solver()`, and a
# lazily-constructed second brain is still a second brain. Rule 1 walks the whole tree ON PURPOSE,
# and this comment records the fixture being wrong rather than the rule.
_GREEN = (
    "import os\n"
    "import sys\n"
    "_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')\n"
    "sys.path.insert(0, _SRC)\n"
    "os.environ.setdefault('NORD_PERSIST', '0')\n"
    "def build():\n"
    "    import os\n"
    "    os.chdir('/tmp')\n"
    "    from snn_data_lake.cortex.brain import make_brain\n"
    "    return make_brain(seed=0, persist=False)\n"
)


def _selfcheck():
    import tempfile

    global ROOT
    print("INSTRUMENT GUARD self-check")
    real_root = ROOT
    try:
        for label, (source, rule) in sorted(_RED.items()):
            with tempfile.TemporaryDirectory() as td:
                fake = pathlib.Path(td)
                (fake / "tools").mkdir()
                (fake / "tools" / "red.py").write_text(source, encoding="utf-8")
                ROOT = fake
                hits = {v[2] for v in scan()[0]}
            assert rule in hits, (
                f"{label} did NOT fire on an input built to trip it -- the {rule} rule cannot "
                f"detect the defect it exists to detect. saw {sorted(hits)}"
            )

        # The FIX must not be punished as hard as the defect, and neither must the deliberate
        # os.environ.setdefault. A gate that reddens on its own remedy teaches the next reader to
        # stop applying it -- the failure red_reachability records catching in itself.
        with tempfile.TemporaryDirectory() as td:
            fake = pathlib.Path(td)
            (fake / "tools").mkdir()
            (fake / "tools" / "green.py").write_text(_GREEN, encoding="utf-8")
            ROOT = fake
            green = scan()[0]
        assert not green, f"the repaired shape was flagged: {green}"
    finally:
        ROOT = real_root

    print("  reds constructed and fired: " + ", ".join(sorted(_RED)))
    print("  repaired shape and module-scope os.environ.setdefault: not flagged")
    rules = {rule for _src, rule in _RED.values()}
    print(
        f"self-check OK: {len(_RED)} of {len(_RED)} reds fired, covering "
        f"{len(rules)} of {len(RULE_NAMES)} rule tags"
    )


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    rc = main()
    _selfcheck()
    raise SystemExit(rc)
