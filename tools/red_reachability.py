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

The Rust half (R0.2, DISPOSITION row 20) reads `native/**/*.rs` through `snn_data_lake.gates.rust_facts`
only: a criterion-shaped Rust fn with no formal inputs that owns no fact is NO_RED, and a Python
criterion whose every return takes such a fn's verdict through `_native` inherits it. Everything the
facts cannot see stays UNPROVEN (`RUST_BLIND_SPOTS`). `native/` absent prints as ABSENT; `.rs`
present with cargo or a parse failure is a RED, never a skip.

Reproduce:
    python tools/red_reachability.py              # report, exit 0
    python tools/red_reachability.py --assert     # exit 1 on unexempted supported defects
    python tools/red_reachability.py --self-check # the constructed Rust RED arm and GREEN twin
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _rust_facts():
    """Lazy: this tool must still run with no PYTHONPATH (dependency_ledger's bootstrap)."""
    here = str(SRC)
    if here not in sys.path:
        sys.path.insert(0, here)
    from snn_data_lake.gates import rust_facts

    return rust_facts


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

#: Printed beside every Rust verdict, after `rust_facts.BLIND_SPOTS`.
RUST_BLIND_SPOTS = (
    "rsfacts records no return/tail expression: a Rust criterion is NO_RED only when it has no formal "
    "inputs AND owns no fact (call, method call, index, assign, let, for, macro), so a literal FAILURE "
    "and a single-segment const/static read also count as a literal result",
    "a Rust criterion with formal inputs (a receiver included: a field read has no fact) is UNPROVEN "
    "whether or not its body reads them",
    "a Python wrapper takes the Rust verdict only through a call whose callee NAME is a `_native` name "
    "of that module (python_native_refs); an alias or a helper in between leaves it UNPROVEN",
    "tools/rsfacts is gate tooling, outside the brain rules",
)
_FACT_KINDS = ("calls", "mcalls", "index", "assigns", "lets", "fors", "macros")

#: The constructed Rust RED arm and GREEN twin (pre-registered; also tests/test_r02b_verdict_rust.py).
RUST_RED_SRC = """use pyo3::prelude::*;
#[pyfunction]
pub fn probe_binding() -> (bool, &'static str) { (true, "spiking 100% vs rate control 0%") }
#[pyfunction]
pub fn reentry_present() -> bool { true }
#[pyfunction]
pub fn probe_measured(brain: &Brain) -> (bool, &'static str) { (brain.bound(), "measured") }
pub fn probe_env() -> (bool, &'static str) { (std::env::var("X").is_ok(), "env") }
#[cfg(test)]
fn probe_test() -> bool { true }
#[pymodule(gil_used = false)]
mod _native {
    #[pymodule_export]
    use super::{probe_binding, reentry_present, probe_measured};
}
"""
RUST_GREEN_SRC = """use pyo3::prelude::*;
#[pyfunction]
pub fn probe_binding(brain: &Brain) -> (bool, &'static str) { (brain.bound(), "measured") }
#[pyfunction]
pub fn reentry_present(trace: &[f64]) -> bool { trace.iter().any(|x| *x > 0.5) }
#[pymodule(gil_used = false)]
mod _native {
    #[pymodule_export]
    use super::{probe_binding, reentry_present};
}
"""
PY_RED_SRC = """from snn_data_lake import _native
import snn_data_lake._native as n


def probe_binding():
    return _native.probe_binding()


def reentry_present():
    return n.reentry_present()


def probe_measured(brain):
    return _native.probe_measured(brain)
"""
PY_GREEN_SRC = """from snn_data_lake import _native


def probe_binding(brain):
    return _native.probe_binding(brain)
"""
RED_NO_RED = {
    "native/sndl-core/src/plasticity.rs::probe_binding",
    "native/sndl-core/src/plasticity.rs::reentry_present",
    "wrap.probe_binding",
    "wrap.reentry_present",
}
#: Review-fold cases (PREREG addenda): (label, Rust source, expected analyses through `rust_criteria`).
RUST_FOLD_CASES = [
    (
        "receiver-field",
        "pub struct B { ok: bool }\nimpl B { pub fn passes(&self) -> bool { self.ok } }\n",
        ["UNPROVEN"],
    ),
    (
        "mut-receiver-literal",
        "pub struct B;\nimpl B { pub fn probe_x(&mut self) -> bool { true } }\n",
        ["UNPROVEN"],
    ),
    ("bodiless-trait", "pub trait T { fn probe_t(&self) -> bool; }\n", []),
    ("unit-return", "pub fn probe_u() { }\n", []),
]
#: Python wrappers over RUST_RED_SRC that must each stay UNPROVEN (one guard apiece).
PY_WRAPPER_CONTROLS = """from snn_data_lake import _native
import snn_data_lake._native as n


def probe_mixed(x):
    if x:
        return _native.probe_binding()
    return True, "literal"


def probe_two_twins(x):
    if x:
        return _native.probe_binding()
    return n.reentry_present()


def probe_refusal():
    assert _native.probe_binding()
    return _native.probe_binding()


def probe_fallthrough(x):
    if x:
        return _native.probe_binding()
"""
#: ...and the positive control: two returns, ONE Rust twin -> NO_RED.
PY_WRAPPER_SAME_TWIN = """from snn_data_lake import _native


def probe_same_twin(x):
    if x:
        return _native.probe_binding()
    return _native.probe_binding()
"""


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


def _input_analysis(fn, params, twin=None):
    """`twin(return_value)` -> the NO_RED Rust criterion a `_native` call returns, else None."""
    facts = _BodyFacts(params)
    for stmt in fn.body:
        facts.visit(stmt)
    twins = [twin(value) for value in facts.returns] if twin else []
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
    elif (
        twins
        and all(twins)
        and len({t["name"] for t in twins}) == 1
        and not facts.refusal
        and _returns_on_all_paths(fn.body)
    ):
        reason = "takes the Rust verdict of " + twins[0]["name"]
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


def native_status(root=None):
    """(status line, file facts) for `native/`; `RustFactsUnavailable` -> a RED status and None."""
    rf = _rust_facts()
    try:
        return rf.native_status(ROOT if root is None else root)
    except rf.RustFactsUnavailable as exc:
        return f"native/ UNAVAILABLE (RED: RustFactsUnavailable: {exc})", None


def _criterion(name, file, line, params, no_red, reason):
    return {
        "name": name,
        "file": file,
        "line": line,
        "params": params,
        "reads_system": False,
        "input_loads": [],
        "has_formal_inputs": bool(params),
        "no_red": no_red,
        "analysis": "NO_RED" if no_red else "UNPROVEN",
        "reason": reason,
    }


def rust_criteria(native):
    """One criterion dict per criterion-shaped Rust fn -- the ONE Rust entry point `scan()` calls.

    `native` is `(status, file_facts)` from `native_status()` or a fixture. Facts owned by a fn are
    attributed by `fn_id` (rust_facts schema 2); a fn that owns none and takes no input returns a
    literal or a const, which no input can change. An unavailable reader is one NO_RED entry.
    """
    status, file_facts = native
    if file_facts is None:
        return [_criterion("native/", "native/", 0, [], True, status)]
    out = []
    for f in file_facts:
        p = pathlib.Path(f["file"])
        rel = (p.relative_to(ROOT) if p.is_absolute() and ROOT in p.parents else p).as_posix()
        for fn in f["fns"]:
            if (
                fn["cfg_test"]
                or f.get("test_file")
                or not fn["has_body"]
                or not fn["ret"]
                or not _is_criterion(fn["name"])
            ):
                continue
            params = [a.split(":")[0].strip() for a in fn["params"]]
            owned = sum(1 for k in _FACT_KINDS for x in f.get(k, []) if x["fn_id"] == fn["id"])
            if params:
                no_red, reason = (
                    False,
                    "formal inputs: use unobservable in facts; reachability unproven",
                )
            elif owned:
                no_red, reason = False, "computation observed; reachability unproven"
            else:
                no_red, reason = True, "no formal inputs; no computation (literal or const result)"
            out.append(_criterion(f"{rel}::{fn['name']}", rel, fn["line"], params, no_red, reason))
    return out


def _twins(native, rust, src):
    """{python module: {`_native` name: NO_RED Rust criterion}} -- the verdicts a wrapper may take."""
    rf = _rust_facts()
    by_key = {(c["file"], c["name"].rsplit("::", 1)[-1]): c for c in rust if c["no_red"]}
    no_red = {}
    for py, e in rf.pyo3_exports(native[1] or []).items():
        p = pathlib.Path(e["file"])
        rel = (p.relative_to(ROOT) if p.is_absolute() and ROOT in p.parents else p).as_posix()
        if e["kind"] == "function" and (rel, e["rust"]) in by_key:
            no_red[py] = by_key[(rel, e["rust"])]
    refs = rf.python_native_refs(src) if no_red else {}
    return {mod: {n: no_red[n] for n in names if n in no_red} for mod, names in refs.items()}


def scan(native=None, src=None):
    """Return (criteria, tautologies). criteria: list of dicts, one per discovered criterion."""
    src = SRC if src is None else src
    native = native_status() if native is None else native
    criteria = rust_criteria(native)
    twins = _twins(native, criteria, src)
    tautologies = []
    for path in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(ROOT if ROOT in path.parents else src.parent).as_posix()
        mine = twins.get(_rust_facts()._module_of(path, src), {})

        def twin(value, mine=mine):
            return mine.get(_called_name(value)) if isinstance(value, ast.Call) else None

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
                        **_input_analysis(node, params, twin if mine else None),
                    }
                )
        if names:
            for line, var, fn in _tautological_asserts(tree, names):
                tautologies.append({"file": rel, "line": line, "var": var, "criterion": fn})
    return criteria, tautologies


# distinct-from: gates/gradient_claim_gate.py::fixture_native -- that one writes a two-crate tree and loads the
#   whole gate module (and reasoning.eprop) into a tools script; this is the reader's own facts_for_source plus the
#   fixture file name. Both belong in rust_facts (integrator row).
def fixture_facts(text):
    """(status, facts) for one constructed sndl-core file, through `rust_facts.facts_for_source`."""
    f = _rust_facts().facts_for_source(text, name="plasticity.rs")
    f["file"] = "native/sndl-core/src/plasticity.rs"
    return "native/ PRESENT (1 .rs files, fixture)", [f]


def self_check():
    """The constructed Rust RED arm and GREEN twin, through `scan()` with injected facts."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "src"
        (src / "pkg").mkdir(parents=True)
        (src / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        wrap = src / "pkg" / "wrap.py"
        wrap.write_text(PY_RED_SRC, encoding="utf-8")
        red = {c["name"] for c in scan(fixture_facts(RUST_RED_SRC), src)[0] if c["no_red"]}
        assert red == RED_NO_RED, red
        wrap.write_text(PY_GREEN_SRC, encoding="utf-8")
        green = [c["name"] for c in scan(fixture_facts(RUST_GREEN_SRC), src)[0] if c["no_red"]]
        assert green == [], green
        # review-fold cases: receivers and filters, the wrapper guards, the one-twin positive
        for label, rs, expected in RUST_FOLD_CASES:
            got = [c["analysis"] for c in rust_criteria(fixture_facts(rs))]
            assert got == expected, (label, got)
        status, facts = fixture_facts(RUST_RED_SRC)
        facts[0]["test_file"] = True
        assert rust_criteria((status, facts)) == []
        wrap.write_text(PY_WRAPPER_CONTROLS, encoding="utf-8")
        py = {c["name"]: c["analysis"] for c in scan(fixture_facts(RUST_RED_SRC), src)[0]}
        assert all(v == "UNPROVEN" for k, v in py.items() if k.startswith("wrap.")), py
        wrap.write_text(PY_WRAPPER_SAME_TWIN, encoding="utf-8")
        py = {c["name"]: c["analysis"] for c in scan(fixture_facts(RUST_RED_SRC), src)[0]}
        assert py["wrap.probe_same_twin"] == "NO_RED", py
    print(
        f"red_reachability self-check OK: Rust RED arm {len(red)} NO_RED, GREEN twin 0, "
        f"{len(RUST_FOLD_CASES)} fold cases + wrapper guards exact"
    )
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-check" in argv:
        return self_check()
    do_assert = "--assert" in argv

    native = native_status()
    criteria, tautologies = scan(native)

    no_red = [c for c in criteria if c["no_red"] and c["name"] not in EXEMPT]
    ok = [c for c in criteria if c["reads_system"]]
    unproven = [c for c in criteria if c["analysis"] == "UNPROVEN"]
    no_formal = [c for c in criteria if not c["has_formal_inputs"]]

    print("=" * 78)
    print("red_reachability -- a criterion that cannot fail is not a measurement (P8)")
    print("=" * 78)
    n_files = sum(1 for _ in SRC.rglob("*.py"))
    n_rust = sum(1 for c in criteria if c["file"].startswith("native/"))
    print(f"scanned {n_files} files under src/, found {len(criteria) - n_rust} criteria")
    print(f"{native[0]}; Rust criteria {n_rust}")
    print("  Rust BLIND SPOTS: " + "; ".join((*_rust_facts().BLIND_SPOTS, *RUST_BLIND_SPOTS)))
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
