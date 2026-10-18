"""Policy-4 replication guard -- there is only ONE brain; no build may re-implement the brain stack.

Policy 4 (user, 2026-07-24): "there is only 1 brain; a build must never replicate brain functionality
-- utilize the original, not re-implement it, because it causes bloat." The canonical brain is a clean
3-layer COMPOSITION (verified): engine.brain.Brain (cognition + circadian clock) composes
engine.live_brain.LiveBrain (perception + workspace) composes engine.autonomous_loop.AutonomousArm
(action + reward + arm + SAFETY). Composition is fine; RE-DEFINING one of those stack classes a second
time (a parallel "brain") is the bloat.

This gate MECHANIZES "one brain" the same way dormancy_gate mechanizes "no dormancy": it AST-scans
snn_data_lake for definitions of the canonical stack classes and ASSERTS each is defined exactly once,
in its canonical module -- unless the duplicate is on the PENDING_RETIREMENT allowlist (a known replica
scheduled for a Policy-4 consolidation wave, with a reason). A NEW second `class Brain` / `class
LiveBrain` / `class AutonomousArm` that is not allowlisted FAILS run_selfchecks -- so a replica cannot
be introduced silently. The allowlist shrinks to empty as the consolidation waves retire the replicas.

The companion to this guard is engine.brain.make_brain(): the single sanctioned CONSTRUCTION path.
Consumers obtain the brain through make_brain() rather than calling Brain(...) directly.

RUST (R0.2 row 4, 2026-10-17): the port moves code into crates under `native/`, so a Rust `struct`/`enum`/
`trait`/`type` named a CANON key, or a PyO3 export whose Python-visible name is one (`#[pyclass(name =
"LiveBrain")]`, `#[pyo3(name = "Brain")]`), is the same replica spelled in Rust. Read through
`rust_facts` only; keyed `(name, "<rel .rs>::<rust item>")` in PENDING_RETIREMENT; cfg(test) items and
test files are not scanned (the Python half never scanned tests/ either). `native/` absent prints as
ABSENT; `.rs` present and unreadable FAILS (closed). PyO3 exports resolve across files (a
`#[pymodule_export]` or a registering helper may sit in another file) and a finding carries the export's
own file. The bare run also executes `_selfcheck()`: the Python replica red, the Rust RED arm (exactly 3)
/ GREEN twin (0), the cross-file export red (1) / twin (0), the two-file name collision (2) and the
helper-registered export (1) / twin (0), so the gate is seen red.

  python -m snn_data_lake.gates.replication_guard   # (also run by run_selfchecks.py)
"""

import ast
import os
import tempfile
from pathlib import Path

from snn_data_lake.gates import rust_facts

_PKG = "snn_data_lake"
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../snn_data_lake
ROOT = Path(_PKG_DIR).parents[1]  # the repo root `native/` sits under (tests inject a fixture root)
RUST_KINDS = frozenset({"struct", "enum", "trait", "type", "union"})
RUST_BLIND_SPOTS = (
    '`m.add("Brain", ..)` string aliases are not in the rust_facts export map',
    "cfg(test) items and test files are not scanned (the Python half never scanned tests/ either)",
    "exports resolve in views (each #[pymodule] file paired with every other file, each with all the"
    " non-#[pymodule] files, and the whole tree); inside one view rust_facts keeps ONE entry per Python"
    " name and one #[pyclass] per Rust name, so an export that collides in every view that sees it keeps"
    " the later file (two exports of one Python name in ONE file; a registration chain through a"
    " #[pymodule]-carrying helper that collides with another export of that name)",
)

# the canonical single-definition stack classes -> the ONE module each MUST be defined in
CANON = {
    "Brain": f"{_PKG}.engine.brain",
    "LiveBrain": f"{_PKG}.engine.live_brain",
    "AutonomousArm": f"{_PKG}.engine.autonomous_loop",
}

# known DUPLICATE definitions still on disk, pending retirement in a Policy-4 consolidation wave.
# key = (class_name, module_path); value = reason. Empty this out as the replicas are retired.
#
# EMPTY as of v4.55 -- there is now literally ONE Brain class. The last replica
# (reasoning/brain_loop.Brain) was retired once every capability was verified canonical: deduction ->
# reasoning/derivation.py (a verbatim port of its _relation_graph/_derive), abduction ->
# pc_abduction.AbductionExpert via LakeChat._abduce_reply, induction -> schema_induction (same
# refuse-to-generalise-on-thin-evidence guard), teach/think/boot/repl -> LakeChat, develop ->
# AutonomousArm.develop + developmental_core, start_learning -> the one learn_daemon, and
# self_rewrite -> Brain.self_improve, which is STRICTLY better gated (crc32 byte-identity firewall
# over the 9 safety/verifier files, NORD_SELFMOD=1 off-switch, and it applies NOTHING -- it returns a
# proposal, where self_rewrite committed on utility gain). A new entry here means a replica shipped.
PENDING_RETIREMENT = {}


def _mpath(pyfile, pkg_dir):
    rel = os.path.relpath(pyfile, os.path.dirname(pkg_dir)).replace("\\", "/")[:-3]
    return rel.replace("/", ".")


def _class_sites(pkg_dir):
    """All definition sites of the canonical stack class NAMES: {name: [module_path, ...]}."""
    sites = {name: [] for name in CANON}
    for root, _d, files in os.walk(pkg_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            mod = _mpath(p, pkg_dir)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in CANON:
                    sites[node.name].append(mod)
    return sites


def _rel(file):
    # distinct-from: tools/instrument_guard.py::rust_scan -- the same inline expression; the shared home
    #   is rust_facts (frozen for this lane; integrator row) and gates cannot import tools/
    # distinct-from: tools/red_reachability.py::rust_criteria -- same
    # distinct-from: tools/red_reachability.py::_twins -- same
    p = Path(file)
    return (p.relative_to(ROOT) if p.is_absolute() and ROOT in p.parents else p).as_posix()


def _native():
    """`(status, facts)` for the `.rs` under ROOT; an unreadable tree is carried as the exception."""
    try:
        files = rust_facts.rust_files(ROOT)
        if not files:
            return "native/ ABSENT (0 .rs files)", []
        return f"native/ PRESENT ({len(files)} .rs files)", rust_facts.facts(files)
    except rust_facts.RustFactsUnavailable as exc:
        return "native/ UNREADABLE", exc


def _has_pymodule(f):
    # distinct-from: tools/instrument_guard.py::rust_scan -- the same #[pymodule] test over fns + mods;
    #   shared home is rust_facts (frozen for this lane; integrator row), gates cannot import tools/
    return any(
        rust_facts._attr(x["attrs"], "pymodule") for x in (*f.get("fns", []), *f.get("mods", []))
    )


def rust_replicas(native=None):
    """(status, replicas, failures): Rust definitions and PyO3 exports named a CANON key -- the ONE Rust
    entry point; `native=(status, facts)` injects a fixture, None reads the real tree."""
    status, file_facts = _native() if native is None else native
    if isinstance(file_facts, Exception):
        return status, [], [f"RustFactsUnavailable: {file_facts}"]
    found = []
    production = [f for f in file_facts if not f["test_file"]]
    for f in production:
        rel = _rel(f["file"])
        for t in f["types"]:
            if t["kind"] in RUST_KINDS and t["name"] in CANON and not t["cfg_test"]:
                found.append((t["name"], f"{rel}::{t['name']}"))
    # rust_facts keeps ONE entry per name per view, so each export needs a view where nothing else wears
    # its name: root + each other file, root + all helpers (helper chains), the whole tree (chains through
    # a #[pymodule]-carrying helper). The root goes last so its own #[pyclass] wins a Rust-name tie.
    roots = [f for f in production if _has_pymodule(f)]
    helpers = [f for f in production if not _has_pymodule(f)]
    pairs = [[d, f] for f in roots for d in production if d is not f]
    for group in [*pairs, *([*helpers, f] for f in roots), production]:
        for py, ex in rust_facts.pyo3_exports(group).items():
            key = (py, f"{_rel(ex['file'])}::{ex['rust']}")
            if py in CANON and key not in found:
                found.append(key)
    return status, [k for k in found if k not in PENDING_RETIREMENT], []


def audit(pkg_dir=None, native=None):
    sites = _class_sites(_PKG_DIR if pkg_dir is None else pkg_dir)
    replicas, missing = [], []
    for name, canon_mod in CANON.items():
        found = sites[name]
        if canon_mod not in found:
            missing.append((name, canon_mod))
        for mod in found:
            if mod == canon_mod:
                continue
            if (name, mod) not in PENDING_RETIREMENT:
                replicas.append((name, mod))  # an UNSANCTIONED second definition
    status, rust, failures = rust_replicas(native)
    return {
        "sites": sites,
        "replicas": replicas + rust,
        "missing": missing,
        "native": status,
        "failures": failures,
    }


def main():
    a = audit()
    print("=" * 66)
    print("REPLICATION GUARD -- Policy 4: there is only ONE brain (no re-implementation)")
    print("=" * 66)
    for name, canon in CANON.items():
        dups = [m for m in a["sites"][name] if m != canon]
        tag = f"  DUPLICATES: {dups}" if dups else ""
        print(f"  {name:<14} defined in {len(a['sites'][name])} site(s){tag}")
    if PENDING_RETIREMENT:
        print("--- allowlisted replicas pending retirement ---")
        for (name, mod), reason in PENDING_RETIREMENT.items():
            print(f"  {name} @ {mod.replace(_PKG + '.', '')}: {reason[:60]}...")
    print(f"  native/: {a['native']}")
    print("  Rust BLIND SPOTS: " + "; ".join((*rust_facts.BLIND_SPOTS, *RUST_BLIND_SPOTS)))
    print("=" * 66)
    assert not a["failures"], f"the Rust half could not run (fail closed): {a['failures']}"
    assert not a["missing"], f"canonical stack class not found where expected: {a['missing']}"
    assert not a["replicas"], (
        f"Policy 4: unsanctioned brain-stack replica(s) -- a second definition of a canonical class "
        f"that is not on PENDING_RETIREMENT: {a['replicas']}. Utilize the one brain (engine.brain."
        f"make_brain / the canonical nesting), do not re-implement it."
    )
    _pend = f" (+ {len(PENDING_RETIREMENT)} allowlisted replica(s))" if PENDING_RETIREMENT else ""
    print(f"replication guard OK: one canonical Brain/LiveBrain/AutonomousArm{_pend}")


# R0.2 row 4: the gate's own reds. Python: a second `class Brain` beside the three canonical modules.
# Rust: a renamed pyclass, a bare type and a renamed pyfunction, each wearing a CANON name (3), with a
# cfg(test) `Brain` as the control that test scope is not counted; the twin keeps every item unrenamed.
_PY_CANON = {
    "engine/brain.py": "class Brain:\n    pass\n",
    "engine/live_brain.py": "class LiveBrain:\n    pass\n",
    "engine/autonomous_loop.py": "class AutonomousArm:\n    pass\n",
}
_RUST_RED = """
use pyo3::prelude::*;

#[pyclass(name = "LiveBrain")]
pub struct NativeLoop;

pub struct Brain;

#[pyfunction]
#[pyo3(name = "AutonomousArm")]
fn arm() {}

#[cfg(test)]
mod tests { pub struct Brain; }

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeLoop>()?;
    m.add_function(wrap_pyfunction!(arm, m)?)?;
    Ok(())
}
"""
_RUST_GREEN = (
    _RUST_RED.replace('name = "LiveBrain"', 'name = "NativeLoop"')
    .replace("pub struct Brain;", "pub struct Core;")
    .replace('#[pyo3(name = "AutonomousArm")]\n', "")
)
_RUST_RED_REPLICAS = [
    ("AutonomousArm", "native/sndl-python/src/lib.rs::arm"),
    ("Brain", "native/sndl-python/src/lib.rs::Brain"),
    ("LiveBrain", "native/sndl-python/src/lib.rs::NativeLoop"),
]
# Codex B1 r02b-instr 1: the pyclass and its #[pymodule_export] in DIFFERENT files; the twin only unrenames.
_RUST_XFILE_LIB = """
use pyo3::prelude::*;
mod replica;

#[pymodule(gil_used = false)]
mod _native {
    #[pymodule_export]
    use super::replica::NativeLoop;
}
"""
_REPLICA_RS = "native/sndl-python/src/replica.rs"
_RUST_XFILE_RED = {_REPLICA_RS: '#[pyclass(name = "LiveBrain")]\npub struct NativeLoop;\n'}
_RUST_XFILE_GREEN = {_REPLICA_RS: '#[pyclass(name = "NativeLoop")]\npub struct NativeLoop;\n'}
_RUST_XFILE_REPLICAS = [("LiveBrain", f"{_REPLICA_RS}::NativeLoop")]
# a SECOND file exporting its own `LiveBrain` (rust_facts keeps one entry per name) -- both sites must
# survive, so an allowlisted one can never hide the other.
_RUST_COLLIDE = {
    **_RUST_XFILE_RED,
    "native/sndl-core/src/b.rs": 'use pyo3::prelude::*;\n#[pyclass(name = "LiveBrain")]\npub struct Loop2;\n'
    "#[pymodule(gil_used = false)]\nmod _b { #[pymodule_export] use super::Loop2; }\n",
}
_RUST_COLLIDE_REPLICAS = [*_RUST_XFILE_REPLICAS, ("LiveBrain", "native/sndl-core/src/b.rs::Loop2")]
# the #[pymodule] only delegates: `replica::register(m)` adds the renamed class from replica.rs.
_RUST_HELPER_LIB = """
use pyo3::prelude::*;
mod replica;

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    replica::register(m)
}
"""
_RUST_HELPER_RED = {
    _REPLICA_RS: 'use pyo3::prelude::*;\n#[pyclass(name = "LiveBrain")]\npub struct NativeLoop;\n'
    "pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<NativeLoop>() }\n"
}
_RUST_HELPER_GREEN = {
    k: v.replace('name = "LiveBrain"', 'name = "NativeLoop"') for k, v in _RUST_HELPER_RED.items()
}


def fixture_native(text, extra=None):
    """`(status, facts)` for a constructed tree: `native/sndl-python/src/lib.rs` carrying `text`, plus the
    `extra` {repo-relative path: text} files."""
    facts = []
    for rel, src in {"native/sndl-python/src/lib.rs": text, **(extra or {})}.items():
        f = rust_facts.facts_for_source(src, name=rel.rsplit("/", 1)[-1])
        f["file"] = rel
        facts.append(f)
    return f"native/ PRESENT ({len(facts)} .rs files, fixture)", facts


def _selfcheck():
    print("REPLICATION GUARD self-check")
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "snn_data_lake"
        for rel, text in {**_PY_CANON, "replica.py": "class Brain:\n    pass\n"}.items():
            (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            (pkg / rel).write_text(text, encoding="utf-8")
        red = audit(pkg_dir=str(pkg), native=("native/ ABSENT (0 .rs files)", []))
        assert red["replicas"] == [("Brain", "snn_data_lake.replica")] and not red["missing"], red
        (pkg / "replica.py").unlink()
        green = audit(pkg_dir=str(pkg), native=("native/ ABSENT (0 .rs files)", []))
        assert not green["replicas"] and not green["missing"], green
    _, rust, failures = rust_replicas(fixture_native(_RUST_RED))
    assert sorted(rust) == _RUST_RED_REPLICAS and not failures, (rust, failures)
    _, twin, _ = rust_replicas(fixture_native(_RUST_GREEN))
    assert not twin, f"the repaired Rust shape was flagged: {twin}"
    _, xfile, _ = rust_replicas(fixture_native(_RUST_XFILE_LIB, _RUST_XFILE_RED))
    assert xfile == _RUST_XFILE_REPLICAS, f"cross-file export not attributed to replica.rs: {xfile}"
    _, xtwin, _ = rust_replicas(fixture_native(_RUST_XFILE_LIB, _RUST_XFILE_GREEN))
    assert not xtwin, f"the unrenamed cross-file pyclass was flagged: {xtwin}"
    _, both, _ = rust_replicas(fixture_native(_RUST_XFILE_LIB, _RUST_COLLIDE))
    assert sorted(both) == sorted(_RUST_COLLIDE_REPLICAS), f"a colliding export was masked: {both}"
    _, helper, _ = rust_replicas(fixture_native(_RUST_HELPER_LIB, _RUST_HELPER_RED))
    assert helper == _RUST_XFILE_REPLICAS, f"helper-registered export not resolved: {helper}"
    _, htwin, _ = rust_replicas(fixture_native(_RUST_HELPER_LIB, _RUST_HELPER_GREEN))
    assert not htwin, f"the unrenamed helper-registered class was flagged: {htwin}"
    _, _, failures = rust_replicas(
        ("native/ UNREADABLE", rust_facts.RustFactsUnavailable("constructed"))
    )
    assert failures, "RustFactsUnavailable with .rs present did not fail closed"
    print(
        "  reds fired: Python replica x1; Rust RED arm x3 (cfg(test) Brain not counted); repaired twin 0;"
        " cross-file export x1, twin 0; two files exporting one name x2; helper-registered x1, twin 0;"
    )
    print("  RustFactsUnavailable -> failure. self-check OK")


if __name__ == "__main__":
    main()
    _selfcheck()
