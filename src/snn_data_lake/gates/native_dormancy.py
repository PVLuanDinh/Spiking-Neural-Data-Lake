"""
native_dormancy -- Policy 3's Rust half: ONE function (`audit`) behind dormancy_gate and
method_dormancy_gate once code moves into `native/` (plan section 10, stage R0.2).

THE DEFECT. Both gates read Python only. A `#[pyfunction]` nothing imports, a `pub mod` nothing live
uses, or a learning-verb fn only a `#[cfg(test)]` calls would sit in the brain's governed tier (every
build is part of the brain, P4c) with no dormancy check at all.

THREE ASSERTING RULES (Rust half active only when `native/` is PRESENT):
  D1 export liveness: every PyO3 class/function export (`rust_facts.pyo3_exports`) is referenced
     (`rust_facts.python_native_refs`) by a Python module `dormancy_gate.reachable()` proves live, else
     a `NATIVE_EXEMPT {py_name: reason}` row. Members are live by use, not by import: a constructor
     (`Cls.__new__`) iff the class name is loaded in a live `_native`-referencing module, a getter iff
     its property is attribute-loaded there, a setter iff it is attribute-assigned or loaded.
  D2 module liveness: (a) every `.rs` under `native/**/<crate>/src` (a crate = any directory holding a
     `Cargo.toml` or a `src/lib.rs|main.rs`, at any depth) is in its crate's `mod` tree from
     `lib.rs`/`main.rs` (`mod x;` -> `x.rs` | `x/mod.rs`, `#[path]` wins); (b) every module of a
     library crate other than the binding crate (the one defining a `#[pymodule]`) is reached from
     the file of a LIVE export through `use`/path references (`crate::`, `super::`, `self::`,
     `<crate>::`, `use` aliases), transitively. Exempt key: the file path under `native/`.
  M1 learning-verb fns (`method_dormancy_gate.is_learn_verb`, same matcher as the Python half): an
     EXPORTED method is live iff its Python-visible name is a name/attribute in a live module that
     references `_native`; an exported FUNCTION is live iff its name is LOADED (not merely imported)
     in a live module that imports it (one imported nowhere is D1's, not counted twice); a
     non-exported fn needs a
     Rust caller (call path or method call, last segment) outside `#[cfg(test)]`, `#[test]`,
     `tests/`, `benches/`, `examples/`. Exempt key: `Type::fn`, bare `fn`, or the Python name.
REPORT only: `pub fn` items of library crates referenced by name nowhere in `native/`.

Stale and phantom exempt rows are red, like `dormancy_gate.EXEMPT`. FAIL CLOSED: `.rs` present and the
reader unavailable -> an UNAVAILABLE finding in both groups, never a skip. A `native/` that is absent
prints `native/ ABSENT`, distinct from "0 findings".

OWN BLIND SPOTS (printed beside the verdict with `rust_facts.BLIND_SPOTS`): liveness is per FILE, not
per item (a live export's file reaches a module even when the item it names is unused); a `use`, call
or path inside an inline `mod` resolves from the file's module; trait dispatch and generics match by last
segment name; a Python facade re-export of an exported learning fn (called only through the
re-exporting module) gives a false M1.
Test code is the reader's `test_file` (one classifier); only `build.rs` is excluded here by name.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.gates.native_dormancy
"""

from __future__ import annotations

import ast
import posixpath
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from snn_data_lake.gates import rust_facts as rf
from snn_data_lake.gates.method_dormancy_gate import is_learn_verb

NATIVE_EXEMPT: dict[str, str] = {}  # py export name | native-relative .rs path -> reason
NATIVE_EXEMPT_METHOD: dict[str, str] = {}  # "Type::fn" | "fn" | python name -> reason

BLIND_SPOTS = (
    "liveness is per file, not per item (a live export's file reaches its modules)",
    "a use, call or path inside an inline mod resolves from the file's module (the file's own children "
    "first, then any inline module's)",
    "trait dispatch and generics match a caller by last segment name",
    "a Python facade re-export (`from ._native import f` in a live module that never calls it, "
    "used elsewhere through that module) gives a false M1: loads are read only from _native-referencing modules",
)


def _rel(f, base):
    return Path(f["file"]).relative_to(base).as_posix()


def _is_test(rel, f):
    """Test code per the reader's one classifier (`test_file`); a build script is its own rule."""
    return f["test_file"] or posixpath.basename(rel) == "build.rs"


def _is_root(rel):
    return rel.endswith(("/src/lib.rs", "/src/main.rs")) or rel in ("src/lib.rs", "src/main.rs")


def _crate_of_root(rel):
    return posixpath.dirname(posixpath.dirname(rel))


# distinct-from: rust_facts.mod_files import_manifest._targets atlas_node_gate._member_file -- those name files a Rust/Python module declaration loads; this reads the FILE back out of a dormancy tree node (`<file>` or `<file>::<inline path>`)
def _file(node):
    return node.partition("::")[0]


def _aliases(f):
    return {
        u["alias"] or u["path"].split("::")[-1]: u["path"]
        for u in f.get("uses", [])
        if not u["cfg_test"]
    }


class _Tree:
    """Crates, their mod trees and the `use`-alias map of every file under native/."""

    def __init__(self, by_rel, cargo_dirs=()):
        self.facts = by_rel
        self.crate_of, self.root, self.children, self.parent, self.alias = {}, {}, {}, {}, {}
        self.flat = {}  # file -> children of its INLINE modules (a path written inside one resolves from the file)
        self.dirs = {*cargo_dirs, *(_crate_of_root(r) for r in by_rel if _is_root(r))}
        for rel in by_rel:
            d = self.crate_dir(rel)
            if d is not None and rel.startswith(posixpath.join(d, "src") + "/"):
                self.crate_of[rel] = d
        for c in sorted(self.dirs):
            r = next(
                (
                    posixpath.join(c, "src", n)
                    for n in ("lib.rs", "main.rs")
                    if posixpath.join(c, "src", n) in by_rel
                ),
                None,
            )
            if r:
                self.root[c] = r
        self.names = {posixpath.basename(c).replace("-", "_"): c for c in self.root if c}
        self.in_tree = set()
        for r in self.root.values():
            self._walk_decl(r)
        for rel, f in by_rel.items():
            self.alias[rel] = _aliases(f)

    def crate_dir(self, rel):
        """Deepest crate directory holding `rel` ("" is native/ itself), None if none does."""
        return max(
            (d for d in self.dirs if not d or rel.startswith(d + "/")), key=len, default=None
        )

    def _walk_decl(self, rel, via_path=False):
        """Mod tree below `rel`, files resolved by the reader's ONE resolver (`rf.mod_files`, cfg(test)
        declarations kept: a test-only module is still in the tree). A file loaded through `#[path]`
        is a mod-rs file, so its children sit beside it. Tree nodes are module PATHS: a file, or
        `<file>::<inline path>` for an inline module, so `lib::used` and `lib::alternate::used` never
        share a children slot (Codex r6 reader 3); `flat` keeps the inline modules' children by file
        for paths written inside them (blind spot: those resolve from the file)."""
        if rel in self.in_tree:
            return
        self.in_tree.add(rel)
        mod_rs = via_path or posixpath.basename(rel) in rf.MOD_RS
        for m in self.facts[rel].get("mods", []):
            node = f"{rel}::{m['mod']}" if m["mod"] else rel
            is_path, cands = rf.mod_files(PurePosixPath(rel), m, mod_rs, cfg_test=True)
            hit = next(
                (c for c in (posixpath.normpath(str(c)) for c in cands) if c in self.facts),
                f"{node}::{m['name']}" if m["inline"] else None,
            )
            if not hit:
                continue
            self.children.setdefault(node, {})[m["name"]] = hit
            self.parent[hit] = node
            if m["mod"]:
                self.flat.setdefault(rel, {}).setdefault(m["name"], hit)
            if not m["inline"]:
                self._walk_decl(hit, is_path)

    def reach(self, rel, segs, depth=0):
        """Files touched by resolving the path `segs` written in file `rel` (an inline-module node
        counts as its file)."""
        out, cur, i = set(), rel, 0
        if segs[0] == "crate":
            cur, i = self.root.get(self.crate_of.get(rel), rel), 1
        elif segs[0] in self.names:
            cur, i = self.root[self.names[segs[0]]], 1
        elif segs[0] == "self":
            i = 1
        for k in range(i, len(segs)):
            s = segs[k]
            if s == "super":
                cur = self.parent.get(cur, cur)
            elif s in self.children.get(cur, {}):
                cur = self.children[cur][s]
            elif s in self.flat.get(cur, {}):
                cur = self.flat[cur][s]
            elif depth < 6 and s in self.alias.get(f := _file(cur), {}):
                out |= self.reach(f, self.alias[f][s].split("::") + segs[k + 1 :], depth + 1)
                break
            else:
                break
            out.add(_file(cur))
        return out | {_file(cur)}

    def refs(self, rel):
        f = self.facts[rel]
        paths = [u["path"] for u in f.get("uses", []) if not u["cfg_test"]]
        paths += [c["path"] for c in f.get("calls", []) if not c["cfg_test"]]
        paths += list(f.get("path_roots", []))
        paths += list(f.get("paths", []))
        return {x for p in paths for x in self.reach(rel, p.split("::"))}


def _py_idents(src, dotted):
    """(names loaded, attributes loaded, attributes stored) of one live module; an `import .. as x`
    alias is not a use, but loading `x` loads the name it stands for."""
    base = Path(src, *dotted.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.exists():
            nodes = list(ast.walk(ast.parse(p.read_text(encoding="utf-8"))))
            loads = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for n in nodes:
                if isinstance(n, ast.ImportFrom):
                    loads |= {a.name for a in n.names if a.asname in loads}
            attrs = [n for n in nodes if isinstance(n, ast.Attribute)]
            return (
                loads,
                {n.attr for n in attrs if isinstance(n.ctx, ast.Load)},
                {n.attr for n in attrs if isinstance(n.ctx, ast.Store)},
            )
    return set(), set(), set()


def _gate(viol, universe, exempt):
    """viol/universe: {id: (message, exempt keys)}. Findings no row exempts, plus red STALE/PHANTOM rows."""
    out = [viol[v][0] for v in viol if not any(k in exempt for k in viol[v][1])]
    for k in sorted(exempt):
        hits = [v for v in universe if k in universe[v][1]]
        if not hits:
            out.append(f"PHANTOM {k!r} names no export, module file or function ({exempt[k]})")
        elif not any(v in viol for v in hits):
            out.append(f"STALE {k!r} exempts something now live ({exempt[k]})")
    return out


def audit(root=rf.ROOT, live=None, src=None, exempt=None, exempt_method=None):
    """-> {"status", "d": D1/D2 findings, "m": M1 findings, "report": [...]}; fails closed."""
    exempt = NATIVE_EXEMPT if exempt is None else exempt
    exempt_method = NATIVE_EXEMPT_METHOD if exempt_method is None else exempt_method
    src = Path(src) if src is not None else Path(root) / "src"
    try:
        files = rf.rust_files(
            root
        )  # the reader binary is always the repo's, the .rs may be a fixture
        status = (
            f"native/ PRESENT ({len(files)} .rs files)" if files else "native/ ABSENT (0 .rs files)"
        )
        fl = rf.facts(files, rf.ROOT)
    except rf.RustFactsUnavailable as exc:
        msg = f"UNAVAILABLE Rust reader failed with .rs present ({exc}); fail closed"
        return {
            "status": "native/ PRESENT (reader unavailable)",
            "d": [msg],
            "m": [msg],
            "report": [],
        }
    base = Path(root) / rf.NATIVE_DIR
    by_rel = {_rel(f, base): f for f in fl}
    if live is None:
        from snn_data_lake.gates import dormancy_gate

        live = dormancy_gate.reachable()
    cargo = {
        "" if p.parent == base else p.parent.relative_to(base).as_posix()
        for p in base.rglob("Cargo.toml")
        if "target" not in p.relative_to(base).parts
    }
    tree = _Tree(by_rel, cargo)
    exports = rf.pyo3_exports(fl)
    refs = rf.python_native_refs(src)
    live_py = {m for m in refs if m in live}
    used = set().union(*(refs[m] for m in live_py)) if live_py else set()
    star = "*" in used
    ids = {m: _py_idents(src, m) for m in live_py}
    loads = set().union(*(i[0] for i in ids.values())) if ids else set()
    attr_load = set().union(*(i[1] for i in ids.values())) if ids else set()
    attr_store = set().union(*(i[2] for i in ids.values())) if ids else set()
    # D1
    d_viol, d_univ, live_files = {}, {}, set()
    for name, e in exports.items():
        if e["kind"] == "method":
            continue
        cls, _, member = name.partition(".")
        by_use = {  # a member is live by use of its class / property, not by importing the class
            "constructor": cls in loads | attr_load,
            "getter": member in attr_load,
            "setter": member in attr_load | attr_store,
        }
        msg = f"D1 {name}: PyO3 export {e['rust']} ({_rel(e, base)}:{e['line']}) has no live Python reference"
        d_univ[f"D1 {name}"] = (msg, {name})
        if e["kind"] in by_use:
            if not (star or by_use[e["kind"]]):
                d_viol[f"D1 {name}"] = (msg, {name})
        elif star or name in used:
            live_files.add(_rel(e, base))
        else:
            d_viol[f"D1 {name}"] = (msg, {name})
    # D2: (a) tree membership, (b) reachability from live-export files in non-binding library crates
    binding = {
        tree.crate_of[r]
        for r, f in by_rel.items()
        if r in tree.crate_of
        and any(
            rf._attr(x["attrs"], "pymodule") and not x["cfg_test"]
            for x in (*f.get("mods", []), *f.get("fns", []))
        )
    }
    reached, stack = set(live_files), list(live_files)
    while stack:
        for r in tree.refs(stack.pop()):
            if r not in reached:
                reached.add(r)
                stack.append(r)
    for rel in sorted(tree.crate_of):
        c = tree.crate_of[rel]
        d_univ[f"D2 {rel}"] = ("", {rel})
        if rel not in tree.in_tree:
            d_viol[f"D2 {rel}"] = (
                f"D2a {rel}: not in the mod tree of crate {c} (orphan file)",
                {rel},
            )
        elif (
            c not in binding
            and rel != tree.root.get(c)
            and posixpath.basename(tree.root.get(c, "")) == "lib.rs"
            and rel not in reached
        ):
            d_viol[f"D2 {rel}"] = (
                f"D2b {rel}: in the mod tree but no live export reaches it through use/paths",
                {rel},
            )
    d = _gate(d_viol, d_univ, exempt)
    # M1
    callers = set()
    for rel, f in by_rel.items():
        if _is_test(rel, f):
            continue
        callers |= {c["path"].split("::")[-1] for c in f.get("calls", []) if not c["cfg_test"]}
        callers |= {c["name"] for c in f.get("mcalls", []) if not c["cfg_test"]}
    pos = {(e["file"], e["line"]): (n, e["kind"]) for n, e in exports.items()}
    idents = loads | attr_load | attr_store
    m_viol, m_univ = {}, {}
    for rel, f in by_rel.items():
        if _is_test(rel, f) or rel not in tree.crate_of:
            continue
        for fn in f.get("fns", []):
            if fn["cfg_test"] or not fn["has_body"] or not is_learn_verb(fn["name"]):
                continue
            key = f"{fn['impl'].split('::')[-1]}::{fn['name']}" if fn["impl"] else fn["name"]
            ex = pos.get((f["file"], fn["line"]))
            keys = {key, fn["name"], *([ex[0]] if ex else [])}
            if ex and ex[1] == "function" and ex[0] not in used and not star:
                continue  # imported nowhere: D1's finding
            why = None
            if ex and ex[1] == "function":
                if ex[0] not in loads | attr_load and not star:
                    why = f"exported as {ex[0]} but no live Python module loads that name"
            elif ex and ex[0].split(".")[-1] not in idents and not star:
                why = f"exported as {ex[0]} but no live Python module uses that name"
            elif not ex and fn["name"] not in callers:
                why = f"({rel}:{fn['line']}) no Rust caller outside #[cfg(test)]/#[test]/tests/"
            m_univ[f"M1 {key}"] = ("", keys)
            if why:
                m_viol[f"M1 {key}"] = (f"M1 {key} {why}", keys)
    m = _gate(m_viol, m_univ, exempt_method)
    # REPORT: pub fns of library crates with no reference by name anywhere in native/
    named = set()
    for f in fl:
        named |= {c["path"].split("::")[-1] for c in f.get("calls", [])} | {
            c["name"] for c in f.get("mcalls", [])
        }
        named |= {s for u in f.get("uses", []) for s in u["path"].split("::")} | set(
            f.get("path_roots", [])
        )
    report = [
        f"unreferenced pub fn {fn['name']} ({rel}:{fn['line']})"
        for rel, f in sorted(by_rel.items())
        if rel in tree.crate_of and tree.crate_of[rel] not in binding and not _is_test(rel, f)
        for fn in f.get("fns", [])
        if fn["vis"].startswith("pub")
        and not fn["cfg_test"]
        and fn["has_body"]
        and fn["name"] not in named
    ]
    return {"status": status, "d": d, "m": m, "report": report}


# ---- the shared fixture: the constructed RED arm and its GREEN twin ---------------------------

FIXTURE_LIVE = {"snn_data_lake.user"}

_CORE_LIB = "pub mod reservoir;\npub mod plasticity;\npub mod dendrite;\n"
_NEURON = "pub fn leak(v: f64) -> f64 { v * 0.9 }\n"
_DENDRITE = "pub fn integrate(x: f64) -> f64 { x }\n"
_STDP_RED = """#[derive(Default)]
pub struct Stdp { w: f64 }
impl Stdp { pub fn potentiate(&mut self, dw: f64) { self.w += dw; } }
#[cfg(test)]
mod tests { use super::*; #[test] fn pot() { let mut s = Stdp::default(); s.potentiate(1.0); } }
"""
_STDP_GREEN = _STDP_RED + "pub fn nudge(s: &mut Stdp) { s.potentiate(0.1); }\n"
_PY_RED = """use pyo3::prelude::*;
use sndl_core::reservoir::leak;
use sndl_core::plasticity::Stdp;

#[pyfunction]
fn lif_step(v: f64) -> f64 { leak(v) }

#[pyfunction]
fn decay(v: f64) -> f64 { let _s = Stdp::default(); leak(v) }

#[pymodule]
mod _native {
    #[pymodule_export]
    use super::lif_step;
    #[pymodule_export]
    use super::decay;
}
"""
_PY_GREEN = _PY_RED.replace(
    "use sndl_core::plasticity::Stdp;",
    "use sndl_core::plasticity::Stdp;\nuse sndl_core::dendrite::integrate;",
).replace("leak(v) }\n\n#[pymodule]", "integrate(leak(v)) }\n\n#[pymodule]")
_USER_RED = "from snn_data_lake._native import decay\n\ndecay(1.0)\n"
_USER_GREEN = "from snn_data_lake._native import decay, lif_step\n\ndecay(1.0)\nlif_step(1.0)\n"


def _write(root, files):
    root = Path(root)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def write_fixture(root, red, prefix=""):
    """Build the fixture native/ tree plus a Python package under `root`. RED arm plants exactly
    three defects (exported lif_step unreferenced, `pub mod dendrite` unused, Stdp::potentiate only
    called from #[cfg(test)]); the GREEN twin fixes each. `prefix` nests the crates under
    `native/<prefix>` (e.g. `crates/`)."""
    _write(
        root,
        {
            f"native/{prefix}sndl-core/src/lib.rs": _CORE_LIB,
            f"native/{prefix}sndl-core/src/reservoir.rs": _NEURON,
            f"native/{prefix}sndl-core/src/dendrite.rs": _DENDRITE,
            f"native/{prefix}sndl-core/src/plasticity.rs": _STDP_RED if red else _STDP_GREEN,
            f"native/{prefix}sndl-python/src/lib.rs": _PY_RED if red else _PY_GREEN,
            "src/snn_data_lake/__init__.py": "",
            "src/snn_data_lake/user.py": _USER_RED if red else _USER_GREEN,
        },
    )


# ---- fold round 1: one (RED, GREEN) fixture pair per reproduced defect ------------------------

_PYO3 = "use pyo3::prelude::*;\n"
_CORE_RES = {
    "native/sndl-core/src/lib.rs": "pub mod reservoir;\n",
    "native/sndl-core/src/reservoir.rs": "pub struct Cell;\nimpl Cell { pub fn tick(&mut self) {} }\n",
}
_CORE_DEN = {
    "native/sndl-core/src/lib.rs": "pub mod dendrite;\n",
    "native/sndl-core/src/dendrite.rs": _DENDRITE,
}
_LEARN_LIB = (
    _PYO3
    + "#[pyfunction]\nfn learn() {}\n#[pymodule]\nmod _native { #[pymodule_export] use super::learn; }\n"
)
_NET_LIB = (
    _PYO3
    + "#[pyclass]\nstruct Net;\n#[pymethods]\nimpl Net {\n #[new]\n fn new() -> Self { Net }\n"
    " #[getter(weight)]\n fn read_weight(&self) -> usize { 1 }\n"
    " #[setter(bias)]\n fn write_bias(&mut self, v: usize) {}\n}\n"
    "#[pymodule]\nmod _native { #[pymodule_export] use super::Net; }\n"
)


def _cell_lib(field):
    return (
        _PYO3
        + f"#[pyclass]\nstruct Net {{ {field} }}\n#[pymethods]\nimpl Net {{ fn tick(&mut self) {{}} }}\n"
        "#[pymodule]\nmod _native { #[pymodule_export] use super::Net; }\n"
    )


def _test_lib(use):
    return (
        _PYO3 + f"{use}\n#[pyfunction]\nfn decay(v: f64) -> f64 {{ v }}\n"
        "#[pymodule]\nmod _native { #[pymodule_export] use super::decay; }\n"
        "#[cfg(test)]\nmod tests { #[test] fn t() { let _ = sndl_core::dendrite::integrate(1.0); } }\n"
    )


def _qualified_binding(py):
    return {
        "native/sndl-python/src/lib.rs": _PYO3
        + "mod helpers;\n#[pyfunction]\nfn lif_step(v: f64) -> f64 { v }\n"
        "#[pyo3::pymodule]\nmod _native { #[pymodule_export] use super::lif_step; }\n",
        "native/sndl-python/src/helpers.rs": "pub fn aid() {}\n",
        "src/snn_data_lake/user.py": py,
    }


_STDP_BIND = (
    _PYO3
    + "pub struct Stdp;\nimpl Stdp { pub fn potentiate(&mut self) {} }\n#[pyfunction]\nfn run() {}\n"
    "#[pymodule]\nmod _native { #[pymodule_export] use super::run; }\n"
)
_DRIVER = "fn drive() { let mut s = super::Stdp; s.potentiate(); }\n"


# distinct-from: mlp_q.py __init__.py immortal_retention.py sleep_pruner_census.py so101_arm.py -- a Q row, a module alias loader, a call walker, a class resolver, an arm disturbance; this WRITES a fixture tree (sibling of _qualified_binding / _cell_lib / _test_lib)
def _ping_fixture(core, body):
    """`core` files under native/sndl-core/ plus a live, registered `ping` whose body is `body`."""
    return {
        **{f"native/sndl-core/{k}": v for k, v in core.items()},
        "native/sndl-python/src/lib.rs": _PYO3 + f"#[pyfunction]\nfn ping() {{ {body} }}\n"
        "#[pymodule]\nmod _native { #[pymodule_export] use super::ping; }\n",
        "src/snn_data_lake/user.py": "from snn_data_lake._native import ping\n\nping()\n",
    }


_USED_TWICE = {
    "src/lib.rs": 'pub mod used;\nmod alternate {\n #[path = "../../tests/used.rs"]\n pub mod used;\n}\n',
    "src/used.rs": "pub struct Used;\n",
    "tests/used.rs": "pub struct Other;\n",
}
_INLINE_LOCAL = {
    "src/lib.rs": "pub mod a {\n pub mod b;\n pub fn f() {}\n}\n",
    "src/a/b.rs": "pub fn g() {}\n",
}
_NESTED_SIBLINGS = {
    "src/lib.rs": (
        'mod a {\n mod b {\n  #[path = "../../x1.rs"]\n  pub mod used;\n }\n}\n'
        'mod c {\n #[path = "../x2.rs"]\n pub mod used;\n}\n'
    ),
    "src/x1.rs": "pub struct X1;\n",
    "src/x2.rs": "pub struct X2;\n",
}


# name -> (RED files, GREEN files, RED (D, M) counts); every GREEN twin is 0/0
FOLD_CASES = {
    "pyo3-qualified-pymodule": (
        _qualified_binding(""),
        _qualified_binding("from snn_data_lake._native import lif_step\n\nlif_step(1.0)\n"),
        (1, 0),
    ),
    "d1-import-only-learn-fn": (
        {
            "native/sndl-python/src/lib.rs": _LEARN_LIB,
            "src/snn_data_lake/user.py": "from snn_data_lake._native import learn\n",
        },
        {
            "native/sndl-python/src/lib.rs": _LEARN_LIB,
            "src/snn_data_lake/user.py": "from snn_data_lake._native import learn\n\nlearn()\n",
        },
        (0, 1),
    ),
    "d2-qualified-type-path": (
        {
            **_CORE_RES,
            "native/sndl-python/src/lib.rs": _cell_lib("x: usize"),
            "src/snn_data_lake/user.py": "from snn_data_lake._native import Net\n\nNet()\n",
        },
        {
            **_CORE_RES,
            "native/sndl-python/src/lib.rs": _cell_lib("inner: sndl_core::reservoir::Cell"),
            "src/snn_data_lake/user.py": "from snn_data_lake._native import Net\n\nNet()\n",
        },
        (1, 0),
    ),
    "c1-constructor-getter-setter": (
        {
            "native/sndl-python/src/lib.rs": _NET_LIB,
            "src/snn_data_lake/user.py": "from snn_data_lake._native import Net\n",
        },
        {
            "native/sndl-python/src/lib.rs": _NET_LIB,
            "src/snn_data_lake/user.py": (
                "from snn_data_lake._native import Net\n\nn = Net()\nn.weight\nn.bias = 2\n"
            ),
        },
        (3, 0),
    ),
    "cfg-test-use-and-path": (
        {
            **_CORE_DEN,
            "native/sndl-python/src/lib.rs": _test_lib(
                "#[cfg(test)]\nuse sndl_core::dendrite::integrate;"
            ),
            "src/snn_data_lake/user.py": "from snn_data_lake._native import decay\n\ndecay(1.0)\n",
        },
        {
            **_CORE_DEN,
            "native/sndl-python/src/lib.rs": _test_lib("use sndl_core::dendrite::integrate;"),
            "src/snn_data_lake/user.py": "from snn_data_lake._native import decay\n\ndecay(1.0)\n",
        },
        (1, 0),
    ),
    # fold round 3
    "cfg-test-pymodule-is-not-a-binding-crate": (
        {
            "native/sndl-core/src/lib.rs": "pub mod dendrite;\n#[cfg(test)]\n#[pyo3::pymodule]\nmod _t {}\n",
            "native/sndl-core/src/dendrite.rs": _DENDRITE,
        },
        {
            "native/sndl-core/src/lib.rs": "pub mod dendrite;\n#[cfg(test)]\n#[pyo3::pymodule]\nmod _t {}\n",
            "native/sndl-core/src/dendrite.rs": _DENDRITE,
            "native/sndl-python/src/lib.rs": _test_lib("use sndl_core::dendrite::integrate;"),
            "src/snn_data_lake/user.py": "from snn_data_lake._native import decay\n\ndecay(1.0)\n",
        },
        (1, 0),
    ),
    "production-path-spelled-by-a-test-call": (
        {
            **_CORE_DEN,
            "native/sndl-python/src/lib.rs": _test_lib(""),
            "src/snn_data_lake/user.py": "from snn_data_lake._native import decay\n\ndecay(1.0)\n",
        },
        {
            **_CORE_DEN,
            "native/sndl-python/src/lib.rs": _test_lib("").replace(
                "{ v }", "{ let _f = sndl_core::dendrite::integrate; v }", 1
            ),
            "src/snn_data_lake/user.py": "from snn_data_lake._native import decay\n\ndecay(1.0)\n",
        },
        (1, 0),
    ),
    "path-included-test-dir-file-is-production": (
        {
            "native/sndl-python/src/lib.rs": _STDP_BIND,
            "native/sndl-python/tests/driver.rs": _DRIVER,
            "src/snn_data_lake/user.py": "from snn_data_lake._native import run\n\nrun()\n",
        },
        {
            "native/sndl-python/src/lib.rs": _STDP_BIND
            + '#[path = "../tests/driver.rs"]\nmod driver;\n',
            "native/sndl-python/tests/driver.rs": _DRIVER,
            "src/snn_data_lake/user.py": "from snn_data_lake._native import run\n\nrun()\n",
        },
        (0, 1),
    ),
    # fold round 5 (Codex r6 reader 3): `lib::used` (src/used.rs) and `lib::alternate::used` (a #[path]
    # include of tests/used.rs) are distinct tree nodes; a live path through the second leaves the
    # first unreached (RED), one through the first reaches it (GREEN).
    "same-named-modules-at-different-paths": (
        _ping_fixture(_USED_TWICE, "let _ = sndl_core::alternate::used::Other;"),
        _ping_fixture(_USED_TWICE, "let _ = sndl_core::used::Used;"),
        (1, 0),
    ),
    # review panel on 8f2aee72: `super` climbs out of an inline node to the file (GREEN reaches src/used.rs
    # through alternate -> lib -> used; RED's single super stays inside alternate)
    "super-out-of-an-inline-node": (
        _ping_fixture(
            {
                **_USED_TWICE,
                "tests/used.rs": "pub struct Other;\nfn g() { let _ = super::used::Other; }\n",
            },
            "let _ = sndl_core::alternate::used::Other;",
        ),
        _ping_fixture(
            {
                **_USED_TWICE,
                "tests/used.rs": "pub struct Other;\nfn g() { let _ = super::super::used::Used; }\n",
            },
            "let _ = sndl_core::alternate::used::Other;",
        ),
        (1, 0),
    ),
    # a path written INSIDE an inline module (`b::g()` in `mod a { pub mod b; .. }`) resolves from the file
    # (blind spot) and must still reach the inline module's own child
    "path-written-inside-an-inline-module": (
        _ping_fixture(_INLINE_LOCAL, "sndl_core::a::f();"),
        _ping_fixture(
            {
                **_INLINE_LOCAL,
                "src/lib.rs": "pub mod a {\n pub mod b;\n pub fn f() { b::g(); }\n}\n",
            },
            "sndl_core::a::f();",
        ),
        (1, 0),
    ),
    # nested (`a::b::used`) and sibling (`c::used`) inline modules declaring one child name stay distinct
    "nested-and-sibling-inline-modules": (
        _ping_fixture(_NESTED_SIBLINGS, "let _ = sndl_core::a::b::used::X1;"),
        _ping_fixture(
            _NESTED_SIBLINGS, "let _ = sndl_core::a::b::used::X1; let _ = sndl_core::c::used::X2;"
        ),
        (1, 0),
    ),
}


# ---- self-check and gate entry ----------------------------------------------------------------


def selfcheck():
    """RED arm = exactly 3 findings, GREEN twin = 0, unparseable Rust = red on both gates, then the
    nested-crate arm and every FOLD_CASES pair at their exact counts."""
    with tempfile.TemporaryDirectory() as t:
        write_fixture(t, red=True)
        res = audit(t, live=FIXTURE_LIVE, src=Path(t) / "src")
        assert len(res["d"]) == 2 and len(res["m"]) == 1, f"RED arm must fire exactly 3: {res}"
    with tempfile.TemporaryDirectory() as t:
        write_fixture(t, red=False)
        res = audit(t, live=FIXTURE_LIVE, src=Path(t) / "src")
        assert not res["d"] and not res["m"], f"GREEN twin must be clean: {res}"
        (Path(t) / "native/sndl-core/src/reservoir.rs").write_text("fn broken( {", encoding="utf-8")
        res = audit(t, live=FIXTURE_LIVE, src=Path(t) / "src")
        assert res["d"] and res["m"], "unparseable Rust did not fail closed"
    for prefix, red in (("crates/", True), ("crates/", False)):  # nested crates stay governed
        with tempfile.TemporaryDirectory() as t:
            write_fixture(t, red=red, prefix=prefix)
            res = audit(t, live=FIXTURE_LIVE, src=Path(t) / "src")
            want = (2, 1) if red else (0, 0)
            assert (len(res["d"]), len(res["m"])) == want, f"nested {red=} arm: {res}"
    for name, (red_files, green_files, want) in FOLD_CASES.items():
        for files, want_arm in ((red_files, want), (green_files, (0, 0))):
            with tempfile.TemporaryDirectory() as t:
                _write(t, files)
                res = audit(t, live=FIXTURE_LIVE, src=Path(t) / "src")
                got = (len(res["d"]), len(res["m"]))
                assert got == want_arm, f"fold case {name}: want {want_arm}, got {res}"


def check(group):
    """The gates' hook: print the Rust verdict and assert group `d` (dormancy) or `m` (method)."""
    selfcheck()
    res = audit()
    print(f"native_dormancy[{group}]: {res['status']}; {len(res[group])} finding(s)")
    print("  BLIND SPOTS: " + "; ".join((*rf.BLIND_SPOTS, *BLIND_SPOTS)))
    for r in res["report"]:
        print(f"  REPORT {r}")
    assert not res[group], "native dormancy: " + "; ".join(res[group])


def main():
    selfcheck()
    res = audit()
    print(
        f"native_dormancy: self-check OK (RED 3 findings, GREEN 0, fail-closed, nested crates, fold-1, fold-3 and fold-5 reds) | {res['status']} | "
        f"D {len(res['d'])} M {len(res['m'])}"
    )
    print("native_dormancy BLIND SPOTS: " + "; ".join((*rf.BLIND_SPOTS, *BLIND_SPOTS)))
    assert not res["d"] and not res["m"], res
    return 0


if __name__ == "__main__":
    sys.exit(main())
