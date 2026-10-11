#!/usr/bin/env python3
"""Find bounded static P8 defects; this is not a proof of reachable RED.

Narrow criterion names are checked for literal results with no formal-input loads, and for
invariant literal success on supported normal-return paths (including simple scalar aliases and
tuple/dict verdict fields). Logging or copying an input does not make a fixed PASS depend on it.
The existing computed-verdict bare-assert check is also retained.

``reads_system`` records syntactic, unshadowed formal-name loads in the function body, excluding
annotations/defaults and nested local names. It does NOT establish verdict dependence. Captured
closures, parameter rebinding, unsupported control and dynamic/indirect access remain UNPROVEN
unless a supported
fixed-success return is independently evident. Global/call-backed returns remain UNPROVEN even
without formal inputs; absence of formal inputs is reported separately. This deliberately
replaces the old, unsound claim that every zero-argument function is invariant to the system.

This is not a Python symbolic interpreter: exceptions, arbitrary calls, external state and
general control/data flow require review and constructed failing controls. All non-defective
criteria remain reachability-UNPROVEN; neither parameter use nor gate exit 0 proves P8 compliance.

Reproduce:
    python tools/red_reachability.py            # report, exit 0
    python tools/red_reachability.py --assert   # exit 1 on unexempted supported defects
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
    """Formal inputs excluding conventional receiver names; include variadic inputs."""
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.posonlyargs]
    args += [a.arg for a in fn.args.kwonlyargs]
    args += [a.arg for a in (fn.args.vararg, fn.args.kwarg) if a is not None]
    return [a for a in args if a not in ("self", "cls")]


class _Bindings(ast.NodeVisitor):
    """Local bindings in one scope, without descending into child scopes."""

    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef

    def visit_Lambda(self, node):
        pass

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

    def visit_Import(self, node):
        self.names.update(a.asname or a.name.split(".")[0] for a in node.names)

    def visit_ImportFrom(self, node):
        self.names.update(a.asname or a.name for a in node.names)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)


class _BodyFacts(ast.NodeVisitor):
    """Inspect formal-name loads, with explicit boundaries rather than inferred execution."""

    def __init__(self, params):
        self.params = set(params)
        self.shadowed = set()
        self.loads = set()
        self.bindings = {}
        self.returns = []
        self.uncertain = set()
        self.refusal = False
        self.controls_unproven = False

    def _bind(self, name):
        self.bindings[name] = self.bindings.get(name, 0) + 1
        if name in self.params - self.shadowed:
            self.uncertain.add("formal input rebound")

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.params - self.shadowed:
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id)

    def visit_AnnAssign(self, node):
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def _definition_access(self, expressions):
        child = _BodyFacts(self.params - self.shadowed)
        for expression in expressions:
            if expression is not None:
                child.visit(expression)
        if child.loads or child.uncertain:
            self.uncertain.add("nested definition expression may access input")

    def _nested(self, body, args=()):
        bound = _Bindings()
        for stmt in body:
            bound.visit(stmt)
        candidates = self.params - self.shadowed - set(args) - bound.names
        child = _BodyFacts(candidates)
        for stmt in body:
            child.visit(stmt)
        if child.loads:
            self.uncertain.add("nested scope may access formal input")
        if child.uncertain:
            self.uncertain.add("nested scope has unresolved access/bindings")

    def visit_FunctionDef(self, node):
        self._bind(node.name)
        self._definition_access(
            node.decorator_list
            + node.args.defaults
            + node.args.kw_defaults
            + [node.returns]
            + [
                arg.annotation
                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            ]
            + [arg.annotation for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
        )
        args = _real_params(node) + ["self", "cls"]
        self._nested(node.body, args)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._bind(node.name)
        self._definition_access(
            node.decorator_list + node.bases + [kw.value for kw in node.keywords]
        )
        self._nested(node.body)

    def visit_Lambda(self, node):
        self._definition_access(node.args.defaults + node.args.kw_defaults)
        self._nested([node.body], _real_params(node) + ["self", "cls"])

    def _comprehension(self, node, values):
        outer = self.shadowed.copy()
        # Only the first iterator runs in the enclosing scope. Every target is local throughout
        # the comprehension's inner scope, including references before its binding executes.
        self.visit(node.generators[0].iter)
        for gen in node.generators:
            self.shadowed.update(_bound_names(gen.target))
        for index, gen in enumerate(node.generators):
            if index:
                self.visit(gen.iter)
            for condition in gen.ifs:
                self.visit(condition)
        for value in values:
            self.visit(value)
        self.shadowed = outer

    def visit_ListComp(self, node):
        self._comprehension(node, [node.elt])

    visit_SetComp = visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._comprehension(node, [node.key, node.value])

    def visit_Call(self, node):
        if _called_name(node) in {"locals", "globals", "vars", "eval", "exec"}:
            self.uncertain.add("dynamic or indirect input access")
        self.generic_visit(node)

    def visit_Global(self, node):
        self.uncertain.add("global/nonlocal binding")

    visit_Nonlocal = visit_Global

    def visit_Import(self, node):
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self._bind(alias.asname or alias.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self._bind(node.name)
        self.generic_visit(node)

    def visit_Return(self, node):
        self.returns.append(node.value)
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.refusal = True
        self.generic_visit(node)

    visit_Assert = visit_Yield = visit_YieldFrom = visit_Raise

    def visit_NamedExpr(self, node):
        self.uncertain.add("assignment expression")
        self.generic_visit(node)

    def visit_If(self, node):
        try:
            ast.literal_eval(node.test)
        except (ValueError, TypeError):
            self.controls_unproven = True
        self.generic_visit(node)

    visit_While = visit_If

    def visit_For(self, node):
        self.controls_unproven = True
        self.generic_visit(node)

    visit_AsyncFor = visit_With = visit_AsyncWith = visit_Try = visit_TryStar = visit_For

    def visit_Match(self, node):
        # Pattern captures are bindings in string AST fields. Do not infer aliases through this
        # unsupported control form, rather than trying to evaluate or enumerate patterns.
        self.uncertain.add("unsupported match/pattern bindings")
        self.generic_visit(node)


def _returns_on_all_paths(body):
    """Only straight-line/if termination; an implicit None refusal is not fixed success."""
    for stmt in body:
        if isinstance(stmt, ast.Return):
            return True
        if (
            isinstance(stmt, ast.If)
            and _returns_on_all_paths(stmt.body)
            and _returns_on_all_paths(stmt.orelse)
        ):
            return True
    return False


def _fixed_success(node, aliases):
    if isinstance(node, ast.Name):
        node = aliases.get(node.id)
    if isinstance(node, ast.Constant):
        return node.value is True or node.value == "PASS"
    if isinstance(node, ast.Tuple) and node.elts:
        return _fixed_success(node.elts[0], aliases)
    if isinstance(node, ast.Dict):
        fields = {"verdict", "present", "passes", "is_present", "success", "passed", "status"}
        if any(not isinstance(key, ast.Constant) for key in node.keys):
            return False
        verdicts = [
            value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and key.value in fields
        ]
        return bool(verdicts) and all(_fixed_success(value, aliases) for value in verdicts)
    return False


def _input_analysis(fn, params):
    facts = _BodyFacts(params)
    for stmt in fn.body:
        facts.visit(stmt)
    # A single, top-level scalar binding is the entire supported alias language. Dynamic access
    # or other unresolved bindings disable alias inference, but cannot alter a literal return.
    aliases = {}
    if not facts.uncertain:
        for stmt in fn.body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Pass)):
                break
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and facts.bindings.get(target.id) == 1:
                    value = (
                        aliases.get(stmt.value.id)
                        if isinstance(stmt.value, ast.Name)
                        else stmt.value
                    )
                    if isinstance(value, ast.Constant):
                        aliases[target.id] = value
    fixed = (
        facts.returns
        and not facts.refusal
        and _returns_on_all_paths(fn.body)
        and all(_fixed_success(value, aliases) for value in facts.returns)
    )
    literal_results = True
    for value in facts.returns:
        try:
            ast.literal_eval(aliases.get(value.id, value) if isinstance(value, ast.Name) else value)
        except (ValueError, TypeError):
            literal_results = False
    if fixed:
        reason = "invariant success on supported normal-return paths"
        no_red = True
    elif facts.uncertain or facts.refusal or facts.controls_unproven:
        reason = "; ".join(sorted(facts.uncertain)) or (
            "explicit refusal/yield path" if facts.refusal else "external/unsupported control"
        )
        no_red = False
    elif not facts.loads and literal_results:
        reason = "formal inputs ignored" if params else "no formal inputs; literal result"
        no_red = True
    else:
        reason = (
            "input use observed; reachability unproven"
            if facts.loads
            else "external/unsupported result"
        )
        no_red = False
    return {
        "reads_system": bool(facts.loads),
        "input_loads": sorted(facts.loads),
        "has_formal_inputs": bool(params),
        "no_red": no_red,
        "analysis": "NO_RED" if no_red else "UNPROVEN",
        "reason": reason,
    }


def _tautological_asserts(tree, criterion_names):
    """Find `x, _ = probe_y(); assert x` -- an assert on a value the same scope just computed.

    This is the shape that kept `consciousness_probe` (RETIRED at T081, 2026-09-16) green: `main()` asserted every verdict it had
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
    shape that kept `consciousness_probe` (RETIRED at T081, 2026-09-16) green for its whole life. But `assert acc is None` and
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
    if isinstance(tgt, ast.Starred):
        return _bound_names(tgt.value)
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
                        **_input_analysis(node, params),
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

    no_red = [c for c in criteria if c["no_red"] and c["name"] not in EXEMPT]
    ok = [c for c in criteria if c["reads_system"]]
    unproven = [c for c in criteria if c["analysis"] == "UNPROVEN"]
    no_formal = [c for c in criteria if not c["has_formal_inputs"]]

    print("=" * 78)
    print("red_reachability -- a criterion that cannot fail is not a measurement (P8)")
    print("=" * 78)
    n_files = sum(1 for _ in SRC.rglob("*.py"))
    print(f"scanned {n_files} files under src/, found {len(criteria)} criteria")
    print()

    print("--- NO_RED: supported missing-input or invariant-success defects ---")
    if not no_red:
        print("  (none)")
    for c in no_red:
        print(f"  {c['name']:44s} {c['file']}:{c['line']}  {c['reason']}")
    print()

    print("--- TAUTOLOGICAL ASSERTS: assert on a verdict the same scope just computed ---")
    if not tautologies:
        print("  (none)")
    for t in tautologies:
        print(f"  {t['file']}:{t['line']}  assert {t['var']}  <- from {t['criterion']}()")
    print()

    print("--- FORMAL INPUT LOADS (syntactic use; not proof of reachable RED) ---")
    for c in ok:
        print(f"  {c['name']:44s} ({', '.join(c['input_loads'])})")
    print()

    print(
        "--- UNPROVEN: no supported defect established; reachable RED still requires evidence ---"
    )
    for c in unproven:
        print(f"  {c['name']:44s} {c['reason']}")
    print()

    print("--- NO FORMAL INPUTS (global/call-backed cases may remain UNPROVEN) ---")
    for c in no_formal:
        print(f"  {c['name']:44s} {c['analysis']}")
    print()

    stale = sorted(set(EXEMPT) - {c["name"] for c in criteria})
    print(
        f"NO_RED: {len(no_red)}   TAUTOLOGICAL: {len(tautologies)}   "
        f"READS_SYSTEM: {len(ok)}   EXEMPT: {len(EXEMPT) - len(stale)}   "
        f"UNPROVEN: {len(unproven)}   NO_FORMAL: {len(no_formal)}"
    )
    if stale:
        print("  STALE EXEMPTIONS (match no real criterion -- delete them):")
        for name in stale:
            print(f"    {name}")
    print()
    print("  NOTE: this gate detects bounded static defects. Input loads and exit 0 do not prove")
    print(
        "  reachable RED or meaningful measurement. UNPROVEN needs constructed controls and review."
    )
    print()

    failed = bool(no_red or tautologies or stale)
    print(
        "gate FAIL: supported P8 defect found"
        if failed
        else "gate OK: no supported defect found; reachability UNPROVEN"
    )
    return 1 if (do_assert and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
