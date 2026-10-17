"""
native_dormancy -- Policy 3's Rust half: ONE function (`audit`) behind dormancy_gate and
method_dormancy_gate once code moves into `native/` (plan section 10, stage R0.2).

THE DEFECT. Both gates read Python only. A `#[pyfunction]` nothing imports, a `pub mod` nothing live
uses, or a learning-verb fn only a `#[cfg(test)]` calls would sit in the brain's governed tier (every
build is part of the brain, P4c) with no dormancy check at all.

THREE ASSERTING RULES (Rust half active only when `native/` is PRESENT):
  D1 export liveness: every PyO3 class/function export (`rust_facts.pyo3_exports`) is referenced
     (`rust_facts.python_native_refs`) by a Python module `dormancy_gate.reachable()` proves live, else
     a `NATIVE_EXEMPT {py_name: reason}` row.
  D2 module liveness: (a) every `.rs` under `native/<crate>/src` is in its crate's `mod` tree from
     `lib.rs`/`main.rs` (`mod x;` -> `x.rs` | `x/mod.rs`, `#[path]` wins); (b) every module of a
     library crate other than the binding crate (the one defining a `#[pymodule]`) is reached from
     the file of a LIVE export through `use`/path references (`crate::`, `super::`, `self::`,
     `<crate>::`, `use` aliases), transitively. Exempt key: the file path under `native/`.
  M1 learning-verb fns (`method_dormancy_gate.is_learn_verb`, same matcher as the Python half): an
     EXPORTED method is live iff its Python-visible name is a name/attribute in a live module that
     references `_native` (an exported function is D1's, not counted twice); a non-exported fn needs a
     Rust caller (call path or method call, last segment) outside `#[cfg(test)]`, `#[test]`,
     `tests/`, `benches/`, `examples/`. Exempt key: `Type::fn`, bare `fn`, or the Python name.
REPORT only: `pub fn` items of library crates referenced by name nowhere in `native/`.

Stale and phantom exempt rows are red, like `dormancy_gate.EXEMPT`. FAIL CLOSED: `.rs` present and the
reader unavailable -> an UNAVAILABLE finding in both groups, never a skip. A `native/` that is absent
prints `native/ ABSENT`, distinct from "0 findings".

OWN BLIND SPOTS (printed beside the verdict with `rust_facts.BLIND_SPOTS`): liveness is per FILE, not
per item (a live export's file reaches a module even when the item it names is unused); a `use`
inside an inline `mod` resolves from the file's module; trait dispatch and generics match by last
segment name; a `#[path]` inside an inline `mod` is not followed.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.gates.native_dormancy
"""

from __future__ import annotations

import ast
import posixpath
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from snn_data_lake.gates import rust_facts as rf
from snn_data_lake.gates.method_dormancy_gate import is_learn_verb

NATIVE_EXEMPT: dict[str, str] = {}  # py export name | native-relative .rs path -> reason
NATIVE_EXEMPT_METHOD: dict[str, str] = {}  # "Type::fn" | "fn" | python name -> reason

BLIND_SPOTS = (
    "liveness is per file, not per item (a live export's file reaches its modules)",
    "use inside an inline mod resolves from the file's module; #[path] in an inline mod not followed",
    "trait dispatch and generics match a caller by last segment name",
)

_NOT_SRC = ("tests", "benches", "examples")
_PATH_ATTR = re.compile(r'path\s*=\s*"([^"]+)"')


def _rel(f, base):
    return Path(f["file"]).relative_to(base).as_posix()


def _is_src(rel):
    p = rel.split("/")
    return len(p) > 2 and p[1] == "src"


def _in_test_dir(rel):
    p = rel.split("/")
    return len(p) > 1 and (p[1] in _NOT_SRC or p[-1] == "build.rs")


def _aliases(f):
    return {
        u["alias"] or u["path"].split("::")[-1]: u["path"]
        for u in f.get("uses", [])
        if not u["cfg_test"]
    }


class _Tree:
    """Crates, their mod trees and the `use`-alias map of every file under native/."""

    def __init__(self, by_rel):
        self.facts = by_rel
        self.crate_of, self.root, self.children, self.parent, self.alias = {}, {}, {}, {}, {}
        for rel in by_rel:
            if _is_src(rel):
                self.crate_of[rel] = rel.split("/")[0]
        for c in sorted(set(self.crate_of.values())):
            r = next(
                (f"{c}/src/{n}" for n in ("lib.rs", "main.rs") if f"{c}/src/{n}" in by_rel), None
            )
            if r:
                self.root[c] = r
        self.names = {c.replace("-", "_"): c for c in self.root}
        self.in_tree = set()
        for r in self.root.values():
            self._walk_decl(r)
        for rel, f in by_rel.items():
            self.alias[rel] = _aliases(f)

    def _walk_decl(self, rel):
        if rel in self.in_tree:
            return
        self.in_tree.add(rel)
        here = posixpath.dirname(rel)
        base = (
            here
            if posixpath.basename(rel) in ("lib.rs", "main.rs", "mod.rs")
            else posixpath.splitext(rel)[0]
        )
        kids = self.children.setdefault(rel, {})
        for m in self.facts[rel].get("mods", []):
            if m["inline"]:
                kids[m["name"]] = rel
                continue
            attr = next((a for a in m["attrs"] if _PATH_ATTR.fullmatch(a)), None)
            d = posixpath.join(base, *m["mod"].split("::")) if m["mod"] else base
            if attr:
                cands = [
                    posixpath.normpath(posixpath.join(here, _PATH_ATTR.fullmatch(attr).group(1)))
                ]
            else:
                cands = [
                    posixpath.join(d, m["name"] + ".rs"),
                    posixpath.join(d, m["name"], "mod.rs"),
                ]
            hit = next((c for c in cands if c in self.facts), None)
            if hit:
                kids[m["name"]] = hit
                self.parent[hit] = rel
                self._walk_decl(hit)

    def reach(self, rel, segs, depth=0):
        """Files touched by resolving the path `segs` written in file `rel`."""
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
            elif depth < 6 and s in self.alias.get(cur, {}):
                out |= self.reach(cur, self.alias[cur][s].split("::") + segs[k + 1 :], depth + 1)
                break
            else:
                break
            out.add(cur)
        return out | {cur}

    def refs(self, rel):
        f = self.facts[rel]
        paths = [u["path"] for u in f.get("uses", []) if not u["cfg_test"]]
        paths += [c["path"] for c in f.get("calls", []) if not c["cfg_test"]]
        paths += list(f.get("path_roots", []))
        return {x for p in paths for x in self.reach(rel, p.split("::"))}


def _py_idents(src, dotted):
    base = Path(src, *dotted.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.exists():
            tree = ast.parse(p.read_text(encoding="utf-8"))
            return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
                n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
            }
    return set()


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
    tree = _Tree(by_rel)
    exports = rf.pyo3_exports(fl)
    refs = rf.python_native_refs(src)
    live_py = {m for m in refs if m in live}
    used = set().union(*(refs[m] for m in live_py)) if live_py else set()
    star = "*" in used
    # D1
    d_viol, d_univ, live_files = {}, {}, set()
    for name, e in exports.items():
        if e["kind"] == "method":
            continue
        msg = f"D1 {name}: PyO3 export {e['rust']} ({_rel(e, base)}:{e['line']}) has no live Python reference"
        d_univ[f"D1 {name}"] = (msg, {name})
        if star or name in used:
            live_files.add(_rel(e, base))
        else:
            d_viol[f"D1 {name}"] = (msg, {name})
    # D2: (a) tree membership, (b) reachability from live-export files in non-binding library crates
    binding = {
        tree.crate_of[r]
        for r, f in by_rel.items()
        if r in tree.crate_of
        and any(
            a.startswith("pymodule")
            for x in (*f.get("mods", []), *f.get("fns", []))
            for a in x["attrs"]
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
        if _in_test_dir(rel):
            continue
        callers |= {c["path"].split("::")[-1] for c in f.get("calls", []) if not c["cfg_test"]}
        callers |= {c["name"] for c in f.get("mcalls", []) if not c["cfg_test"]}
    pos = {(e["file"], e["line"]): (n, e["kind"]) for n, e in exports.items()}
    idents = set().union(*(_py_idents(src, m) for m in live_py)) if live_py else set()
    m_viol, m_univ = {}, {}
    for rel, f in by_rel.items():
        if _in_test_dir(rel) or not _is_src(rel):
            continue
        for fn in f.get("fns", []):
            if fn["cfg_test"] or not fn["has_body"] or not is_learn_verb(fn["name"]):
                continue
            key = f"{fn['impl'].split('::')[-1]}::{fn['name']}" if fn["impl"] else fn["name"]
            ex = pos.get((f["file"], fn["line"]))
            keys = {key, fn["name"], *([ex[0]] if ex else [])}
            if ex and ex[1] == "function":
                continue
            why = None
            if ex and ex[0].split(".")[-1] not in idents and not star:
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
        if _is_src(rel) and tree.crate_of[rel] not in binding and not _in_test_dir(rel)
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


def write_fixture(root, red):
    """Build the fixture native/ tree plus a Python package under `root`. RED arm plants exactly
    three defects (exported lif_step unreferenced, `pub mod dendrite` unused, Stdp::potentiate only
    called from #[cfg(test)]); the GREEN twin fixes each."""
    root = Path(root)
    files = {
        "native/sndl-core/src/lib.rs": _CORE_LIB,
        "native/sndl-core/src/reservoir.rs": _NEURON,
        "native/sndl-core/src/dendrite.rs": _DENDRITE,
        "native/sndl-core/src/plasticity.rs": _STDP_RED if red else _STDP_GREEN,
        "native/sndl-python/src/lib.rs": _PY_RED if red else _PY_GREEN,
        "src/snn_data_lake/__init__.py": "",
        "src/snn_data_lake/user.py": _USER_RED if red else _USER_GREEN,
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)


# ---- self-check and gate entry ----------------------------------------------------------------


def selfcheck():
    """RED arm = exactly 3 findings, GREEN twin = 0, unparseable Rust = red on both gates."""
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
        f"native_dormancy: self-check OK (RED 3 findings, GREEN 0, fail-closed) | {res['status']} | "
        f"D {len(res['d'])} M {len(res['m'])}"
    )
    print("native_dormancy BLIND SPOTS: " + "; ".join((*rf.BLIND_SPOTS, *BLIND_SPOTS)))
    assert not res["d"] and not res["m"], res
    return 0


if __name__ == "__main__":
    sys.exit(main())
