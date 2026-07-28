"""
import_manifest -- the CROSS-TIER canary. Policy 3, the half no other gate covers.

THE DEFECT THIS EXISTS FOR. The governed tiers (cortex/, thalamus/, substrate/, brainstem/, ...) reach
into the UNGOVERNED tiers (reasoning/, training/) through LAZY imports that sit inside bare
`except Exception:` swallows. `cortex/brain.py` alone holds 67 such swallows. Delete or break one of
those targets and:

  run_selfchecks    green -- the module is simply no longer in the list
  pytest            green -- pyproject omits */reasoning/* and */training/* from the coverage denominator
  ruff              green -- pyproject extend-excludes both directories
  dormancy_gate     green -- BRAIN_MODULES is built from REGIONS + TORCH_TIER, and neither contains them
  replication_guard green -- it only matches three exact class names

...while the brain silently loses the faculty. `Brain.__init__` sets the attribute to None, every tick
after that runs without it, and a static reader still credits the faculty as wired. Two of the targets
(`self_build_loop`, `self_mod_harness`) ARE Policy 0's self-building-code avenue, so the whole avenue
can be deleted with every gate green. One (`rose_language`) is the language faculty.

WHAT THIS ASSERTS. Every ungoverned module that a governed module imports must still be IMPORTABLE.
That is deliberately the weaker of the two claims available -- it proves the target exists and loads,
NOT that its output changes behaviour. Charge is Policy 3's second tier and is proven per faculty by
the runtime neuter-A/B, not here. Import-ability is exactly the property the nine swallows destroy
silently, and it is the property a deletion breaks, so it is the one worth a gate.

The manifest is DERIVED by AST walk, never hand-maintained -- the lesson from
dormancy_gate._external_roots(): a hand-written list cannot see a wire nobody wrote down.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.import_manifest
"""

import ast
import importlib
import os

# The tiers Policy 3 mechanizes (REGIONS keys + the top-level modules). A module here is covered by
# brain_atlas parity, the dormancy gate, ruff and the coverage denominator.
GOVERNED = (
    "cortex",
    "thalamus",
    "substrate",
    "brainstem",
    "hippocampus",
    "plasticity",
    "neuromodulation",
    "cerebellum",
    "basal_ganglia",
    "amygdala",
    "midbrain",
    "hypothalamus",
    "medulla",
    "epithalamus",
    "claustrum",
    "lakehouse",
)

# The tiers no gate covers. A governed -> ungoverned edge is what this module protects.
UNGOVERNED = ("reasoning", "training")

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _targets(tree):
    """Every `snn_data_lake.<ungoverned>...` module named by an import in one parsed file.
    Covers `import a.b`, `from a.b import c`, and the LAZY/GUARDED forms (an import nested inside a
    function or a try block is an ordinary ast.Import node -- ast.walk sees it either way, which is
    the entire point)."""
    for n in ast.walk(tree):
        names = []
        if isinstance(n, ast.ImportFrom) and n.module:
            names = [n.module]
        elif isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        for m in names:
            if any(m.startswith(f"snn_data_lake.{u}") for u in UNGOVERNED):
                yield m, n.lineno


def manifest():
    """{ungoverned module -> sorted ['region/file.py:lineno', ...]} for every governed importer."""
    out = {}
    for region in GOVERNED:
        rdir = os.path.join(_PKG_DIR, region)
        if not os.path.isdir(rdir):
            continue  # a REGIONS key with no directory on disk (pons / olfactory) -- nothing to scan
        for dp, _dn, fn in os.walk(rdir):
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    tree = ast.parse(open(p, encoding="utf-8").read())
                except (OSError, SyntaxError):
                    continue
                rel = os.path.relpath(p, _PKG_DIR).replace(os.sep, "/")
                for m, ln in _targets(tree):
                    out.setdefault(m, set()).add(f"{rel}:{ln}")
    return {m: sorted(s) for m, s in sorted(out.items())}


def broken(man=None):
    """The manifest entries that do NOT import. Each is a silent faculty loss on the live brain.
    A real import, not importlib.util.find_spec: find_spec would catch a DELETED target but not a
    BROKEN one, and the swallows hide both identically."""
    man = manifest() if man is None else man
    bad = {}
    for m in man:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001 -- the point is to REPORT it, which the callers don't
            bad[m] = f"{type(e).__name__}: {e}"
    return bad


def main():
    man = manifest()
    # A manifest that has gone empty means the scan broke, not that the coupling vanished -- a
    # constant where a computed value belongs is this project's recurring silent-degradation tell.
    assert man, "manifest is EMPTY -- the AST scan is broken, not the coupling"
    assert len(man) >= 10, f"manifest collapsed to {len(man)} targets -- scan regression?"

    bad = broken(man)

    print("=" * 78)
    print("CROSS-TIER IMPORT MANIFEST (governed -> ungoverned, Policy 3)")
    print("=" * 78)
    for m, sites in man.items():
        short = m.replace("snn_data_lake.", "")
        mark = "BROKEN" if m in bad else "ok"
        print(f"  [{mark:>6}] {short}")
        for s in sites:
            print(f"             <- {s}")
    n_sites = sum(len(v) for v in man.values())
    print(
        f"\n{len(man)} ungoverned modules are load-bearing for the governed tiers, {n_sites} call sites."
    )
    print(
        "None of them is covered by ruff, brain_atlas parity, the dormancy gate, or the coverage gate."
    )

    assert not bad, (
        "a module the live brain imports does not import: "
        + "; ".join(f"{m} ({e})" for m, e in bad.items())
        + " -- the caller swallows this and sets the faculty to None, so no other gate would fail"
    )
    print(f"self-check OK: all {len(man)} cross-tier targets import")


if __name__ == "__main__":
    main()
