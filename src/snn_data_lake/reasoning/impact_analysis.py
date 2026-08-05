"""
IMPACT ANALYSIS -- pre-mutation blast-radius for the self-build (the "LSP" harness principle, native).

Anthropic's Claude Code harness exposes an LSP tool family whose distinctive contribution is
PRE-MUTATION BLAST-RADIUS: incomingCalls / findReferences answer "what depends on this symbol before I
change it", file-grouped, so a change is scoped + surgical. A SELF-MODIFYING system should run exactly
that safety gate before editing its own source. SNDL computes it NATIVELY with zero deps -- stdlib `ast`
(precise symbol references, not a text grep) + the EXISTING graphify graph (graphify-out/graph.json:
nodes carry source_file; links are source->target with a relation) for cross-file import/call edges. No
language-server subprocess, no JSON-RPC, no multi-language routing (all agent/editor-specific, honestly
NOT transferred to a single-language Python sim).

blast_radius(repo, target_rel, symbol) returns the set of OTHER files that reference the target module
(graphify import/call edges) or the target symbol (ast references) -- the modules whose self-checks a
mutation there could break. The self-build consumes this to (a) prefer SURGICAL mutations (lower radius,
relative to the candidate set -- no fixed threshold) and (b) surface the radius in the audit/rationale.
It NEVER weakens the gate: the full run_selfchecks + byte-exact rollback remain the safety backstop;
blast radius is an ADDITIONAL, advisory, transparency signal (the "consume computed-but-not-consumed
graphify edges" master lever).

Determinism (per CLAUDE.md): pure functions of the tree + graph; sorted outputs; ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.impact_analysis
"""

import ast
import json
import os

from snn_data_lake.substrate.self_mod_harness import _repo_root

# graphify link relations that mean "the source depends on the target" (an incoming edge = a referrer).
_DEP_RELATIONS = frozenset(
    {"calls", "imports", "uses", "references", "reads", "instantiates", "inherits", "depends_on"}
)


def _rel(path):
    return str(path).replace("\\", "/")


def _graph_referrers(graph, target_rel):
    """Files that reference target_rel via a graphify dependency edge (incoming link into any node whose
    source_file is target_rel). Returns a set of referrer source_files (excluding target_rel itself)."""
    target_rel = _rel(target_rel)
    node_src = {}
    target_ids = set()
    for n in graph.get("nodes", ()):
        nid = n.get("id")
        sf = _rel(n.get("source_file") or "")
        node_src[nid] = sf
        if sf == target_rel:
            target_ids.add(nid)
    if not target_ids:
        return set(), False
    referrers = set()
    for link in graph.get("links", ()):
        if link.get("target") in target_ids and link.get("relation") in _DEP_RELATIONS:
            src_file = node_src.get(link.get("source")) or _rel(link.get("source_file") or "")
            if src_file and src_file != target_rel:
                referrers.add(src_file)
    return referrers, True


def _symbol_referrers(repo, symbol, exclude_rel):
    """Files under src/snn_data_lake that reference `symbol` as a Name or an attribute (ast-precise -- a
    match inside a string/comment does NOT count). Excludes the definition file exclude_rel."""
    if not symbol:
        return set()
    exclude_rel = _rel(exclude_rel)
    root = os.path.join(repo, "src", "snn_data_lake")
    out = set()
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = _rel(os.path.relpath(full, repo))
            if rel == exclude_rel:
                continue
            try:
                with open(full, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=full)
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Name) and node.id == symbol) or (
                    isinstance(node, ast.Attribute) and node.attr == symbol
                ):
                    out.add(rel)
                    break
    return out


def blast_radius(repo_root=None, target_rel=None, symbol=None, graph_path=None):
    """Pre-mutation blast radius of editing `symbol` (or the whole module) in `target_rel`.

    Returns {radius, files, modules, via_graph, via_ast}: `files` = the OTHER source files that reference
    the target module (graphify dep edges) or the symbol (ast); `radius` = len(files); `modules` = their
    dotted module names; the flags say which sources contributed. A mutation there could break exactly
    these files' self-checks -- so a smaller radius is more surgical/safe."""
    repo = os.path.abspath(repo_root) if repo_root else _repo_root()
    target_rel = _rel(target_rel or "")
    files = set()
    via_graph = False
    gp = graph_path or os.path.join(repo, "graphify-out", "graph.json")
    try:
        with open(gp, encoding="utf-8") as f:
            graph = json.load(f)
        g_ref, via_graph = _graph_referrers(graph, target_rel)
        files |= g_ref
    except (OSError, ValueError):
        via_graph = False
    ast_ref = _symbol_referrers(repo, symbol, target_rel)
    files |= ast_ref
    modules = sorted(
        {
            f[len("src/") : -len(".py")].replace("/", ".")
            for f in files
            if f.startswith("src/") and f.endswith(".py")
        }
    )
    return {
        "radius": len(files),
        "files": sorted(files),
        "modules": modules,
        "via_graph": via_graph,
        "via_ast": bool(ast_ref),
    }


# ------------------------------------------------------------------- self-check -------------------------
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def main():
    import tempfile

    width = 96
    _emit("=" * width)
    _emit(
        "IMPACT ANALYSIS -- pre-mutation blast radius for the self-build (stdlib ast + graphify graph)"
    )
    _emit("=" * width)

    work = tempfile.mkdtemp(prefix="impact_analysis_selfcheck_")
    try:
        # a toy repo: a target module defines CONST; two other modules reference it, one does not.
        def w(rel, body):
            p = os.path.join(work, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)

        w(
            "src/snn_data_lake/neuromodulation/target.py",
            "CONST = 0.5\n\n\ndef f():\n    return CONST\n",
        )
        w(
            "src/snn_data_lake/cortex/user_a.py",
            "from snn_data_lake.neuromodulation.target import CONST\n\nx = CONST + 1\n",
        )
        w(
            "src/snn_data_lake/cortex/user_b.py",
            "import snn_data_lake.neuromodulation.target as t\n\ny = t.CONST\n",
        )
        w("src/snn_data_lake/cortex/unrelated.py", "Z = 3  # references nothing of the target\n")

        target = "src/snn_data_lake/neuromodulation/target.py"

        # (1) AST-only (no graph file present): CONST references found in user_a + user_b, NOT unrelated
        r = blast_radius(work, target, symbol="CONST")
        assert not r["via_graph"], "no graph file -> graph path contributes nothing"
        assert r["radius"] == 2, r
        assert set(r["modules"]) == {
            "snn_data_lake.cortex.user_a",
            "snn_data_lake.cortex.user_b",
        }, r
        assert "snn_data_lake.cortex.unrelated" not in r["modules"], (
            "an unrelated file must NOT be in the radius"
        )
        _emit(
            f"  [OK] ast blast radius of CONST = {r['radius']} ({', '.join(m.split('.')[-1] for m in r['modules'])}); unrelated excluded"
        )

        # (2) a match in a STRING/COMMENT does not count (ast-precise, not a text grep)
        w(
            "src/snn_data_lake/cortex/user_c.py",
            "S = 'CONST appears here in a string only'  # and CONST in a comment\n",
        )
        r2 = blast_radius(work, target, symbol="CONST")
        assert "snn_data_lake.cortex.user_c" not in r2["modules"], (
            "a string/comment mention must NOT count"
        )
        _emit(
            "  [OK] ast-precise: a CONST inside a string/comment is NOT a reference (grep would false-positive)"
        )

        # (3) graphify graph edges also contribute (incoming dep link into a target node)
        graph = {
            "nodes": [
                {"id": "tgt_f", "source_file": target},
                {"id": "caller", "source_file": "src/snn_data_lake/cortex/graph_only.py"},
            ],
            "links": [{"source": "caller", "target": "tgt_f", "relation": "calls"}],
        }
        gp = os.path.join(work, "graph.json")
        with open(gp, "w", encoding="utf-8") as f:
            json.dump(graph, f)
        r3 = blast_radius(work, target, symbol="CONST", graph_path=gp)
        assert r3["via_graph"] and "snn_data_lake.cortex.graph_only" in r3["modules"], r3
        assert r3["radius"] >= 3, r3  # user_a + user_b (ast) + graph_only (graph)
        _emit(
            f"  [OK] graphify dep edge adds a referrer: radius {r3['radius']} incl. graph-only caller"
        )

        # (4) no symbol + no graph -> radius 0 (a module with no known referrers is fully surgical)
        r4 = blast_radius(work, "src/snn_data_lake/cortex/unrelated.py")
        assert r4["radius"] == 0, r4
        _emit(
            "  [OK] a module with no referrers -> radius 0 (a fully surgical, low-risk mutation site)"
        )
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)

    _emit("-" * width)
    _emit(
        "self-check OK: native pre-mutation blast radius (stdlib ast symbol refs + graphify dep edges);"
    )
    _emit(
        "string/comment mentions excluded; advisory only -- the full gate + byte-exact rollback stay the"
    )
    _emit(
        "safety backstop. The self-build consumes it to prefer SURGICAL changes + scope re-verification."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
