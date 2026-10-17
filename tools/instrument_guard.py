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

EXTENDED TO `src/` 2026-08-14, AND COUNTED FIRST AGAIN. The gate shipped with
`TIERS = ("tools", "tests")`, so the 374 files of `src/` -- the governed tier -- were never scanned,
and torch seeding was uncovered in every tier. Both halves matter: `src/` is governed for DORMANCY
by the three P3 gates and by nothing at all for DETERMINISM, which is the axis these rules measure.
The counted run over `src/`, before any of it landed:

    P4-CONSTRUCTION        6  -> ALL SIX are `engine/brain.py` constructing its own class
    CWD-DEPENDENT PATH     0
    MODULE-SCOPE CHDIR     0
    MODULE-SCOPE SEED      1  -> reasoning/arc_agi/play.py, `random.seed(0)` under `__main__`
    CONSTRUCTOR SEED       3  -> sensorimotor/recognizer_bank.py, instruments/developmental_loop_brain.py,
                                 reasoning/_loop_common.py

The 6 are why `src/` could not simply be appended to `TIERS`: `make_brain()` lives in the module that
DEFINES `Brain` and must call it, or there is no sanctioned path to be the sanctioned path. Rule 1
now skips the defining module, which is what P4's clause actually says rather than an allowlist.

THE NARROWING IS THE MEASUREMENT, and it is recorded because the naive rule was unadoptable.
A seeding rule that fires ANYWHERE in a function body scores **79** across the tree (73 `src/`,
6 `tests/`) -- the allowlist shape this repo has killed two proposals at. Split by enclosing scope:

    ENTRYPOINT-BODY (main/_selfcheck)   40   a leaf entrypoint seeding its own run
    OTHER-FUNCTION                      35   set_seed() helpers, train_model(), bench runners
    CONSTRUCTOR (__init__)               3   <- the incident
    MODULE-SCOPE                         1
    CLASS-BODY                           0

So the rule is scoped to import-time scope plus CONSTRUCTORS: 4 findings, a work list. The three
constructor hits are the SAME DEFECT `readout/growable_readout.py`'s Determinism note records
being removed by hand on 2026-08-04 -- it recurred three times over and nothing in the tree could
see it, which is exactly what regression protection is for.

The CENSUS half of the same run: 29 tools/ modules, 15 with a self-check, an assert, or a pytest
guard, and 14 with none of the three. That number is REPORTED, never asserted -- see below.
**RE-DERIVED 2026-08-14: 30 / 17 / 13.** The 2026-08-04 figures above are kept as what that run
measured, not as a description of this tree -- a census sentence that outlives its run is the defect
this file is one level up from. The census is `tools/`-only and did NOT widen with `TIERS`: it asks
whether an INSTRUMENT can tell you it is broken, and `src/` has three P3 gates asking its version of
that question already.

It is registered in `tests/test_instrument_guard.py` rather than in `run_selfchecks.py` or
`.github/workflows/ci.yml`, so it asserts on every `pytest` run without editing a shared file. That
test file also fires each rule against a fixture, and it is what caught this gate's own worst bug:
`pathlib.PurePath("/opt/x").is_absolute()` is FALSE on Windows, so rule 2 reported a POSIX absolute
path as cwd-dependent and the gate's verdict depended on which CI runner executed it.

THE FIVE RULES, and each one is a defect this tree actually produced:

  1. P4 CONSTRUCTION. A file in any scanned tier may not call `Brain(...)` unless it DEFINES the
     class; the one sanctioned path is `engine.brain.make_brain()`. `tools/charge_probe.py` records fixing exactly this in
     itself -- *"the instrument that enforced P3 was quietly violating P4"* -- and
     `tools/arc_baseline.py` was still doing it on 2026-08-04, in the tool that produces the
     published ARC "before" number. `gates/replication_guard.py` cannot see this: it asserts
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

  4. IMPORT-TIME GLOBAL SEEDING. `random.seed(...)` / `np.random.seed(...)` / `torch.manual_seed(...)`
     at import time is the determinism invariant's named prohibition -- *"use `random.Random(seed)`
     (per-instance). Never module-level import-time seeding"* -- and in a shared pytest process it
     silently re-seeds whatever ran before it.
     **CLASS BODIES ARE IMPORT TIME, since 2026-08-14.** `_module_scope_nodes` skipped `ClassDef`
     outright, so `class C:` + `random.seed(0)` in its body walked past -- the one construct that
     looks like a container and is really a running statement list. A `def` nested in a class is
     still a function body and is still skipped.
     **ALL SPELLINGS, since 2026-08-14.** The banned names were matched as WRITTEN, so
     `import numpy.random as npr` + `npr.seed(0)` and `from torch import manual_seed` +
     `manual_seed(0)` were the same defect spelled two ways the gate could not read. Call targets
     are now canonicalized through the module's own import table (`_aliases` / `_canon`) before the
     lookup, which is rule 1's 2026-08-05 lesson applied to rules 2-4: *a rule that can be evaded by
     an import style is not a rule*. Nothing in the tree used those spellings, so this closed a
     latent hole rather than a live finding -- measured, not assumed.

  5. GLOBAL SEEDING INSIDE A CONSTRUCTOR. `readout/growable_readout.py` ran
     `torch.manual_seed(seed)` inside `__init__`, so CONSTRUCTING one re-seeded the process-global
     torch RNG for every module built after it. Both harnesses that run this tree do so in ONE
     process -- `run_selfchecks.py` for part of the suite, and `tests/test_selfchecks.py` via
     `runpy.run_module(..., run_name="__main__")` for every CORE module. It was found and removed by
     hand on 2026-08-04, its Determinism note is the record, and NO GATE COULD SEE IT: rule 4 walks
     import-time scope only, and a constructor body is not import time. Three more live instances of
     the same shape were sitting in `src/` when this rule was counted.
     **A restored seed is not a violation.** A seeder inside `with torch.random.fork_rng(...)`, or
     one bracketed by an explicit save/restore in the same function (`random.getstate` /
     `random.setstate`), is NOT flagged -- `sensorimotor/recognizer_bank.py` does exactly that because the
     MoE trainer it calls draws from the module-global `random`, and a gate that reddens on its own
     remedy teaches the next reader to stop applying the remedy.

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

from snn_data_lake.gates import rust_facts

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The instrument tiers. `tools`/`tests` are ungoverned by the three P3 gates; `src` is governed by
#: them for DORMANCY and by nothing at all for DETERMINISM, which is the axis these rules measure.
#: All three are part of the brain (P4c), so all three are scanned.
TIERS = ("tools", "tests", "src")

#: Stack classes that must be obtained through their sanctioned constructor rather than called.
#: `AutonomousArm` is deliberately ABSENT: it is the arm, not the brain, and several arm tests
#: legitimately construct one in isolation to exercise the E-STOP without standing up a cortex.
#: P4's sanctioned-path clause is about `engine.brain.make_brain()`, so this rule is scoped to what
#: that clause actually says.
CONSTRUCTED_VIA_FACTORY = {"Brain": "engine.brain.make_brain()"}

#: Module-scope calls that mutate state shared by every module imported after them. Keys are
#: CANONICAL dotted paths -- call targets are resolved through the module's own import table first,
#: so `np.random.seed` needs no separate entry and `npr.seed` cannot slip past for lack of one.
BANNED_MODULE_SCOPE = {
    "os.chdir": "moves the working directory for every module after it in a shared process",
    "random.seed": "process-global seeding; use random.Random(seed) per instance",
    "numpy.random.seed": "process-global seeding; use np.random.default_rng(seed)",
    "torch.manual_seed": "process-global seeding; use torch.Generator().manual_seed(seed)",
    "torch.random.manual_seed": "process-global seeding; use torch.Generator().manual_seed(seed)",
    "torch.cuda.manual_seed": "process-global seeding; use torch.Generator().manual_seed(seed)",
    "torch.cuda.manual_seed_all": "process-global seeding; use torch.Generator().manual_seed(seed)",
    "snn_data_lake._native.set_seed": "seeds the extension's process-global RNG at import time; call it from the run",
}

#: seeder -> the call that puts back what it took. A seeder bracketed by its own restorer in the
#: same function is not a violation; see rule 5. `torch.random.fork_rng` restores ALL of them and is
#: matched by name rather than through this table.
RESTORERS = {
    "random.seed": "random.setstate",
    "numpy.random.seed": "numpy.random.set_state",
    "torch.manual_seed": "torch.set_rng_state",
    "torch.random.manual_seed": "torch.set_rng_state",
    "torch.cuda.manual_seed": "torch.cuda.set_rng_state",
    "torch.cuda.manual_seed_all": "torch.cuda.set_rng_state_all",
}

#: The subset of BANNED_MODULE_SCOPE that is process-global RNG state. Derived, never re-typed: a
#: seeder added to one list and forgotten in the other is the drift this file exists to catch.
GLOBAL_SEEDERS = frozenset(RESTORERS)

#: Rust (`native/`, R0.2 row 16): a call whose LAST path segment is one of these mutates process state
#: shared by every module after it, the `os.chdir` / `os.environ` shape one tier down.
NATIVE_PROCESS_MUTATORS = frozenset({"set_current_dir", "set_var"})

#: Rust process-global RNG seeders, matched on the call path's last one or two segments: `fastrand::seed`
#: (the crate-global generator) and any free `set_seed` (the extension's own global, `_native.set_seed`).
NATIVE_SEEDERS = frozenset({"fastrand::seed", "set_seed"})

#: What every `#[pymodule]` must DECLARE (Phase 0 item 5: pyo3 0.30.1's plain form no longer re-enables
#: the GIL, so the declaration is the rule, not the effect). Spaces are ignored when matching.
GIL_DECLARATION = "gil_used=false"

RUST_BLIND_SPOTS = (
    "calls inside macro bodies (println!/vec!/..) are not seen as calls",
    "a seeder reached through a helper fn called from an init/ctor body is one hop away and unseen",
    "init/ctor = #[pymodule] fn, #[new] fn, or an impl fn named new; other constructors are not scanned",
)

#: Every tag this gate can report. Named once so the self-check can state what fraction of the tags
#: it actually fired a red for, rather than counting its own fixture table against itself.
RULE_NAMES = (
    "P4-CONSTRUCTION",
    "CWD-DEPENDENT-PATH",
    "MODULE-SCOPE-MUTATION",
    "CONSTRUCTOR-SEED",
    "SYNTAX",
    "NATIVE-PROCESS-MUTATION",
    "PYMODULE-GIL",
    "NATIVE-INIT-SEED",
    "RUST-FACTS-UNAVAILABLE",
)

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


def _aliases(tree):
    """Name bound in this module -> the dotted path it actually refers to.

    Closes the alias and from-import spellings. `import numpy.random as npr` + `npr.seed(0)`, and
    `from torch import manual_seed` + `manual_seed(0)`, are the same defect as the canonical form,
    and matching the name AS WRITTEN let both walk straight past rules 2-4. This is rule 1's
    2026-08-05 lesson generalized: a rule that an import style can evade is not a rule.

    Relative `from . import x` is recorded with its leading dots (`._native`), since 2026-10-17: the
    extension's `_native.set_seed` IS reachable that way from inside the package, and `_native_seed`
    matches it by its last two segments. A rebinding that is not an import (`random = something_else`)
    is a known floor, stated rather than claimed away.
    """
    tbl = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                tbl[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            mod = "." * node.level + (node.module or "")
            for a in node.names:
                if a.name != "*":
                    tbl[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
    return tbl


def _canon(dotted, tbl):
    """`npr.seed` -> `numpy.random.seed` under one module's import table."""
    if dotted is None:
        return None
    head, _, rest = dotted.partition(".")
    base = tbl.get(head)
    if base is None:
        return dotted
    return _native_seed(f"{base}.{rest}" if rest else base)


def _native_seed(dotted):
    """`._native.set_seed` / `snn_data_lake._native.set_seed` -> the one BANNED_MODULE_SCOPE key."""
    if dotted.split(".")[-2:] == ["_native", "set_seed"]:
        return "snn_data_lake._native.set_seed"
    return dotted


def _calls_in(node, tbl):
    """Every canonical call target anywhere under `node`. Used to spot a seeder's own restorer."""
    return {_canon(_dotted(n.func), tbl) for n in ast.walk(node) if isinstance(n, ast.Call)}


def _module_scope_nodes(tree):
    """Every node that executes at IMPORT time: module body AND class bodies, minus function bodies.

    A call inside a `def` runs when the function is called, which is the caller's problem, not the
    importer's. A call inside `if __name__ == "__main__":` is deliberately INCLUDED as module scope
    -- it does not run on plain import, but `tests/test_selfchecks.py` executes every CORE module's
    `__main__` with `runpy.run_module(..., run_name="__main__")` IN THE SAME PROCESS, so a banned
    call there reaches every module the suite touches afterwards. That stance shipped untested (both
    tiers were clean); scanning `src/` produced its first finding and its first mechanism.

    CLASS BODIES were skipped outright until 2026-08-14, which was a real hole rather than a
    simplification: a class body is a statement list that RUNS on import, so `class C:` +
    `random.seed(0)` in its body is a module-scope seed wearing a container's clothes. A `def`
    nested inside a class is still a function body and is still skipped.
    """
    out = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        out.append(node)
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return out


def _constructor_seeds(tree, tbl):
    """(lineno, canonical) for every process-global seeder called inside an `__init__`. Rule 5.

    A constructor body is not import time, so rule 4 structurally cannot reach it -- which is how
    `readout/growable_readout.py` re-seeded the process-global torch RNG from `__init__` for the
    life of that feature, and how three more instances of the same shape were live in `src/` when
    this rule was counted.

    NOT flagged when the scope puts back what it took: a `with torch.random.fork_rng(...)` block, or
    the seeder's own restorer called in the same function. The exclusion is scoped to the function,
    which is a floor -- it does not prove the restore is on every path -- and it is deliberate:
    flagging `sensorimotor/recognizer_bank.py`'s getstate/setstate bracket would have been a false finding
    against correct code, and a gate that reddens on its own remedy teaches the reader to drop it.
    """
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name != "__init__":
            continue
        present = _calls_in(fn, tbl)
        if any(c and c.rsplit(".", 1)[-1] == "fork_rng" for c in present):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            called = _canon(_dotted(node.func), tbl)
            if called in GLOBAL_SEEDERS and RESTORERS[called] not in present:
                out.append((node.lineno, called))
    return out


def _relative_syspath_literal(node, tbl):
    """`sys.path.insert(0, "src")` / `sys.path.append("tools")` with a RELATIVE literal -> the arg.

    A joined or `__file__`-derived expression is not a Constant, so it never matches -- which is the
    whole point: the fix for this rule is to derive the path from `__file__`.

    Canonicalized through `tbl` since 2026-08-14, so `import sys as s` + `s.path.insert(0, "src")`
    is the same finding as the plain spelling.
    """
    if not isinstance(node, ast.Call):
        return None
    name = _canon(_dotted(node.func), tbl)
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


def scan(native=None):
    """(violations, census). A violation is (relpath, lineno, rule, detail).

    `native` injects a `(status, facts)` fixture for the Rust half (self-check and tests); None reads
    the real `native/` through `rust_facts`. Both run through `rust_scan`, the ONE Rust entry point.
    """
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

            tbl = _aliases(tree)
            names = {
                n.name
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            defined_here = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
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
            # dotted form is matched on its LAST component, so `engine.brain.Brain()` is caught
            # too. A call on an expression (`factory().Brain()`) still escapes -- `_dotted`
            # needs a Name at the base -- which is a known and deliberate floor, not a claim of
            # completeness.
            # THE DEFINING MODULE IS EXCLUDED (2026-08-14, with `src/` in scope): `make_brain()`
            # lives in the module that defines `Brain` and must call it, or there is no sanctioned
            # path to be the sanctioned path. All SIX `src/` hits on the first counted run were
            # `engine/brain.py` constructing its own class. P4's clause is about CONSUMERS, so this
            # is the rule reading what the policy says -- not an allowlist for one file.
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = _dotted(node.func)
                if called is None:
                    continue
                stem = called.rsplit(".", 1)[-1]
                factory = CONSTRUCTED_VIA_FACTORY.get(stem)
                if factory and stem not in defined_here:
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
                literal = _relative_syspath_literal(node, tbl)
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
                    called = _canon(_dotted(node.func), tbl)
                    why = BANNED_MODULE_SCOPE.get(called)
                    if why:
                        violations.append(
                            (rel, node.lineno, "MODULE-SCOPE-MUTATION", f"{called}(): {why}")
                        )

            # rule 5 -- a constructor may not re-seed the process
            for lineno, called in _constructor_seeds(tree, tbl):
                violations.append(
                    (
                        rel,
                        lineno,
                        "CONSTRUCTOR-SEED",
                        f"{called}() runs in __init__ -- CONSTRUCTING this re-seeds the "
                        "process-global RNG for every module built after it, and both harnesses "
                        "run modules in ONE process; draw from a per-instance generator, or "
                        "bracket it with torch.random.fork_rng / getstate+setstate",
                    )
                )
    violations.extend(rust_scan(native)[1])
    return violations, census


def _init_or_ctor(fn):
    """A `#[pymodule]` fn, a `#[new]` fn, or an impl fn named `new`: the bodies rule 5 means in Rust."""
    return bool(
        rust_facts._attr(fn["attrs"], "pymodule")
        or rust_facts._attr(fn["attrs"], "new")
        or (fn["impl"] and fn["name"] == "new")
    )


def _declares_gil(attr):
    return GIL_DECLARATION in {a.replace(" ", "") for a in rust_facts._top_args(attr)}


def _native():
    """`(status, facts)` for the `.rs` under this ROOT (the reader's binary stays at its own root);
    an unreadable tree is carried as the exception so the scan fails CLOSED, never skips."""
    try:
        files = rust_facts.rust_files(ROOT)
        if not files:
            return "native/ ABSENT (0 .rs files)", []
        return f"native/ PRESENT ({len(files)} .rs files)", rust_facts.facts(files)
    except rust_facts.RustFactsUnavailable as exc:
        return "native/ UNREADABLE", exc


def rust_scan(native=None):
    """(status, violations) over every `.rs` under native/ -- the ONE Rust entry point (R0.2 row 16).

    Rules 3-5 in Rust: `set_current_dir`/`set_var` anywhere (process state shared by every module after
    it), every `#[pymodule]` declaring `gil_used = false`, and no process-global seeder in an init/ctor
    body. Reads `rust_facts` only; `RustFactsUnavailable` with `.rs` present is a finding, never a skip.
    """
    status, file_facts = _native() if native is None else native
    if isinstance(file_facts, Exception):
        return status, [
            ("native/", 0, "RUST-FACTS-UNAVAILABLE", f"RustFactsUnavailable: {file_facts}")
        ]
    out = []
    for f in file_facts:
        p = pathlib.Path(f["file"])
        rel = (p.relative_to(ROOT) if p.is_absolute() and ROOT in p.parents else p).as_posix()
        by_id = {fn["id"]: fn for fn in f["fns"]}
        for c in f["calls"]:
            tail = c["path"].split("::")
            fn = by_id.get(c["fn_id"])
            qual = f"{fn['impl']}::{fn['name']}" if fn and fn["impl"] else (fn or {}).get("name")
            if tail[-1] in NATIVE_PROCESS_MUTATORS:
                out.append(
                    (
                        rel,
                        c["line"],
                        "NATIVE-PROCESS-MUTATION",
                        f"{c['path']}() in {qual}: mutates process state shared by every module after it",
                    )
                )
            if (
                fn
                and _init_or_ctor(fn)
                and ("::".join(tail[-2:]) in NATIVE_SEEDERS or tail[-1] in NATIVE_SEEDERS)
            ):
                out.append(
                    (
                        rel,
                        c["line"],
                        "NATIVE-INIT-SEED",
                        f"{c['path']}() in {qual}: re-seeds a process-global RNG from an init/ctor body; "
                        "seed a per-instance generator",
                    )
                )
        for item in (*f["fns"], *f["mods"]):
            attr = rust_facts._attr(item["attrs"], "pymodule")
            if attr and not _declares_gil(attr):
                how = (
                    "declares gil_used = true"
                    if "true" in attr
                    else "does not declare gil_used = false"
                )
                out.append(
                    (
                        rel,
                        item["line"],
                        "PYMODULE-GIL",
                        f"#[pymodule] {item['name']} {how}; pyo3 0.30.1 makes the declaration the rule",
                    )
                )
    return status, out


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
    native = _native()
    violations, census = scan(native)

    print("=" * 78)
    print("instrument_guard -- the instruments are part of the brain too (P4c)")
    print("=" * 78)
    print(f"scanned {len(census)} modules across {', '.join(t + '/' for t in TIERS)}")
    print(f"native/: {native[0]}")
    print("  Rust BLIND SPOTS: " + "; ".join((*rust_facts.BLIND_SPOTS, *RUST_BLIND_SPOTS)))
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
        "from snn_data_lake.engine.brain import Brain\n_B = Brain()\n",
        "P4-CONSTRUCTION",
    ),
    "P4-CONSTRUCTION (qualified)": (
        "from snn_data_lake.engine import brain\n_B = brain.Brain()\n",
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
    "MODULE-SCOPE-MUTATION (torch.manual_seed)": (
        "import torch\ntorch.manual_seed(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    "MODULE-SCOPE-MUTATION (torch.cuda.manual_seed_all)": (
        "import torch\ntorch.cuda.manual_seed_all(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    # The two spellings that used to walk past because the name was matched as WRITTEN.
    "MODULE-SCOPE-MUTATION (alias spelling)": (
        "import numpy.random as npr\nnpr.seed(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    "MODULE-SCOPE-MUTATION (from-import spelling)": (
        "from torch import manual_seed as ms\nms(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    # A class body is a statement list that RUNS on import.
    "MODULE-SCOPE-MUTATION (class body)": (
        "import random\n\n\nclass C:\n    random.seed(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    # Rule 5's incident, reconstructed: the shape growable_readout carried and no gate could see.
    "CONSTRUCTOR-SEED": (
        "import torch\n\n\nclass R:\n    def __init__(self, seed=0):\n"
        "        torch.manual_seed(seed)\n",
        "CONSTRUCTOR-SEED",
    ),
    # An unparseable instrument is reported, never skipped: `ast.parse` failing is the one way a
    # file can carry every defect above and be counted clean, which is the silent-pass shape.
    "SYNTAX": ("def broken(\n", "SYNTAX"),
    # R0.2 row 16, Python edge: the extension's global seeder at import time, in each import spelling.
    "MODULE-SCOPE-MUTATION (_native.set_seed)": (
        "from snn_data_lake import _native\n_native.set_seed(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
    "MODULE-SCOPE-MUTATION (_native.set_seed, relative import)": (
        "from . import _native\n_native.set_seed(0)\n",
        "MODULE-SCOPE-MUTATION",
    ),
}

#: R0.2 row 16: the Rust reds, one fixture `native/sndl-python/src/lib.rs` carrying all three defects
#: (pre-registered: EXACTLY 6 findings, two per tag), and its repaired twin (EXACTLY 0).
_RUST_RED = """
use pyo3::prelude::*;
use std::env::set_var;

#[pyclass]
pub struct Loop { seed: u64 }

#[pymethods]
impl Loop {
    #[new]
    fn new(seed: u64) -> Self { fastrand::seed(seed); Loop { seed } }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    std::env::set_current_dir("..").ok();
    set_var("NORD_GPU", "1");
    set_seed(0);
    m.add_class::<Loop>()?;
    Ok(())
}

#[pymodule(gil_used = true)]
mod _legacy { #[pymodule_export] use super::Loop; }

fn set_seed(_s: u64) {}
"""
_RUST_RED_TAGS = {
    "NATIVE-PROCESS-MUTATION": 2,
    "PYMODULE-GIL": 2,
    "NATIVE-INIT-SEED": 2,
}
_RUST_GREEN = """
use pyo3::prelude::*;

#[pyclass]
pub struct Loop { rng: fastrand::Rng }

#[pymethods]
impl Loop {
    #[new]
    fn new(seed: u64) -> Self { Loop { rng: fastrand::Rng::with_seed(seed) } }
}

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Loop>()?;
    Ok(())
}

#[pyo3::pymodule(gil_used = false)]
mod _legacy { #[pymodule_export] use super::Loop; }

fn set_seed(_s: u64) {}
fn reseed(s: u64) { set_seed(s) }
"""


def fixture_native(text):
    """`(status, facts)` for a constructed one-file tree `native/sndl-python/src/lib.rs` carrying `text`."""
    f = rust_facts.facts_for_source(text, name="lib.rs")
    f["file"] = "native/sndl-python/src/lib.rs"
    return "native/ PRESENT (1 .rs files, fixture)", [f]


# The REPAIRED shape of all three reds, plus the two things that must NOT be flagged: a module-scope
# `os.environ.setdefault` (deliberate, see the docstring) and an `os.chdir` inside a function body
# (runs when called, not on import).
#
# NOTE what is deliberately absent: a `Brain()` call inside a `def`. The first version of this
# fixture had one, on the theory that "inside a function" excused it -- and the gate correctly
# fired. It should: `tools/arc_baseline.py`'s violation was inside `live_brain_solver()`, and a
# lazily-constructed second brain is still a second brain. Rule 1 walks the whole tree ON PURPOSE,
# and this comment records the fixture being wrong rather than the rule.
#
# THREE MORE REMEDIES were added 2026-08-14 with rules 4-5: the two BRACKETED seeds (fork_rng and
# getstate/setstate) and a DEFINING module constructing its own class. Each is the shape the fix
# for a real finding takes, so each must stay green or the fix reads as the defect.
_GREEN = (
    "import os\n"
    "import random\n"
    "import sys\n"
    "import torch\n"
    "_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')\n"
    "sys.path.insert(0, _SRC)\n"
    "os.environ.setdefault('NORD_PERSIST', '0')\n"
    "def build():\n"
    "    import os\n"
    "    os.chdir('/tmp')\n"
    "    from snn_data_lake.engine.brain import make_brain\n"
    "    return make_brain(seed=0, persist=False)\n"
    "class Forked:\n"
    "    def __init__(self, seed=0):\n"
    "        with torch.random.fork_rng(devices=[]):\n"
    "            torch.manual_seed(seed)\n"
    "            self.w = torch.rand(3)\n"
    "class Bracketed:\n"
    "    def __init__(self, seed=0):\n"
    "        state = random.getstate()\n"
    "        random.seed(seed)\n"
    "        try:\n"
    "            self.v = random.random()\n"
    "        finally:\n"
    "            random.setstate(state)\n"
    "class PerInstance:\n"
    "    def __init__(self, seed=0):\n"
    "        self.rng = random.Random(seed)\n"
    "        self.gen = torch.Generator().manual_seed(seed)\n"
    "class Brain:\n"
    "    pass\n"
    "def make_brain(**kw):\n"
    "    return Brain(**kw)\n"
)


def _selfcheck():
    import tempfile

    global ROOT
    print("INSTRUMENT GUARD self-check")

    # The two seeder tables are written separately and must agree, or a seeder banned at module
    # scope is silently unbannable in a constructor (or the reverse). Checked here rather than
    # trusted, because "derived, never re-typed" is a comment and this is the measurement.
    missing = sorted(GLOBAL_SEEDERS - set(BANNED_MODULE_SCOPE))
    assert not missing, f"RESTORERS names a seeder BANNED_MODULE_SCOPE does not: {missing}"

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

    # R0.2 row 16: the Rust reds through the SAME entry point, injected; then the repaired twin.
    rust = rust_scan(fixture_native(_RUST_RED))[1]
    tags = {t: sum(1 for v in rust if v[2] == t) for t in _RUST_RED_TAGS}
    assert tags == _RUST_RED_TAGS and len(rust) == 6, (
        f"Rust RED arm: expected {_RUST_RED_TAGS}, saw {rust}"
    )
    rust_green = rust_scan(fixture_native(_RUST_GREEN))[1]
    assert not rust_green, f"the repaired Rust shape was flagged: {rust_green}"
    unreadable = rust_scan(("native/ UNREADABLE", rust_facts.RustFactsUnavailable("constructed")))[
        1
    ]
    assert [v[2] for v in unreadable] == ["RUST-FACTS-UNAVAILABLE"], unreadable

    print("  reds constructed and fired: " + ", ".join(sorted(_RED)))
    print("  repaired shape and module-scope os.environ.setdefault: not flagged")
    print(
        "  Rust reds fired: "
        + ", ".join(f"{t} x{n}" for t, n in sorted(_RUST_RED_TAGS.items()))
        + "; RUST-FACTS-UNAVAILABLE x1; repaired Rust twin: not flagged"
    )
    rules = (
        {rule for _src, rule in _RED.values()} | set(_RUST_RED_TAGS) | {"RUST-FACTS-UNAVAILABLE"}
    )
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
