"""
import_manifest -- the CROSS-TIER canary. Policy 3, the half no other gate covers.

THE DEFECT THIS EXISTS FOR. The governed tiers (cortex/, thalamus/, substrate/, brainstem/, ...) reach
into the UNGOVERNED tiers (reasoning/, training/) through LAZY imports that sit inside bare
`except Exception:` swallows.

Those imports come in TWO SHAPES and until 2026-08-05 this gate could only see one. The in-package
shape is prefixed (`snn_data_lake.reasoning.schema_induction`; the example here was `rose_language`
until its 2026-08-12 promotion took it out of the manifest). The ROOT shape is a bare top-level name,
because the root tier is not inside the package at all. `UNGOVERNED_ROOT` below is that second shape.

**HISTORICAL, and the reason the ROOT shape exists at all.** It was added 2026-08-05 because
`cortex/brain.py::Brain.__init__` built its perception tier from THREE bare-root imports --
`training.eprop_nmnist`, `training.surrogate_snn`, `training.eprop_shd` -- every one invisible to
the one gate built for exactly that failure. All three sat in the body of ONE `try` under a single
`except Exception` with no `as name`, and one shared handler is the worse arrangement, not the
milder one: breaking the vision import aborted the whole block and took the AUDIO tier with it, in
silence.

**NONE OF THAT IS STILL TRUE, and the tuple below is now empty.** `eprop_nmnist` went in-package at
the 2026-08-12 PROMOTE lane; `surrogate_snn` was RETIRED and `eprop_shd` PROMOTED at specs/013
(2026-08-13), which also SPLIT that one `try` so vision and audio now carry their own handlers and
fail independently. Read the paragraph above as the record of a defect that has been fixed, not as a
description of the tree. Deliberately no interior line cites: this very block once quoted a `try`
span (`:557-643`) that insertions above it had rotted, which is the failure the paragraph below
names -- and it rotted a SECOND time across specs/013, caught only by that lane's pre-push review.
Name the enclosing SYMBOL and let `ast` find it.

RE-COUNTED 2026-08-04, WITH THE DEFINITION STATED, because the figure here was unreproducible. This
said "`cortex/brain.py` alone holds 67 such swallows" and a paragraph below said "nine swallows" --
two numbers for one quantity, neither saying what it counted. POLICIES.md records the same figure
being wrong and re-derived once already. Counted by `ast`: a handler is a SWALLOW when it is
`except Exception:` with no `as name`; it is IMPORT-WRAPPING when its `try` BODY contains an
`Import`/`ImportFrom` node. On this tree: `cortex/brain.py` holds **116** swallows, **44** of them
import-wrapping; across the governed tier, **166** swallows and **58** import-wrapping.

The DENOMINATOR needs the same treatment the numerator just got, and did not get it: this said
"across the 223 governed-tier files" and 223 is not reproducible by any definition tried. Re-derived
2026-08-05, both readings stated because they differ: `dormancy_gate.BRAIN_MODULES` (REGIONS +
TORCH_TIER) is **205** modules, all 205 resolving to a file on disk; every `.py` in the directories
those modules live in, `__init__.py` included, is **220**. Both give the SAME 166 / 58, because the
15 `__init__` files hold no swallows -- which is why an unreproducible denominator survived beside
two exact numerators. Re-derive rather than inherit; the definition above is the whole instrument.

Delete or break one of those targets and:

  run_selfchecks    green -- the module is simply no longer in the list
  pytest            green -- pyproject omits */reasoning/* and */training/* from the coverage denominator
  ruff              green -- pyproject extend-excludes both directories
  dormancy_gate     green -- BRAIN_MODULES is built from REGIONS + TORCH_TIER, and neither contains them
  replication_guard green -- it only matches three exact class names

...while the brain silently loses the faculty. `Brain.__init__` sets the attribute to None, every tick
after that runs without it, and a static reader still credits the faculty as wired. Three of the
targets (`reasoning.godel_machine` and `reasoning.self_improve`, both from `infra/self_mod_harness`,
plus `reasoning.impact_analysis` from `infra/brain_native_selfbuild`) ARE Policy 0's
self-building-code avenue, so the whole avenue can be deleted with every gate green. (THREE were the
perception tier via `UNGOVERNED_ROOT` until specs/013 on 2026-08-13 promoted or retired all of them;
that tuple is empty now and the perception tier no longer appears in this manifest at all.)
(A fourth named here until 2026-08-12, `rose_language` the
language faculty, left the manifest when its promotion made the brain.py reach governed->governed.)
(This said "Two of the targets (`self_build_loop`, `self_mod_harness`)" until 2026-08-05 and named the
wrong side of the edge: both are governed IMPORTERS -- `self_build_loop` was promoted into
`basal_ganglia/` and is not a target at all, and `self_mod_harness` is the substrate module doing the
importing. Re-derived by reading this gate's own printed output rather than inherited.)

WHAT THIS ASSERTS. Every ungoverned module that a governed module imports must still be IMPORTABLE.
That is deliberately the weaker of the two claims available -- it proves the target exists and loads,
NOT that its output changes behaviour. Charge is Policy 3's second tier and is proven per faculty by
the runtime neuter-A/B (`tools/charge_probe.py`), not here. Import-ability is exactly the property
those swallows destroy silently, and it is the property a deletion breaks, so it is the one worth a
gate. ("the nine swallows" stood here against "67" four paragraphs up -- the second of the two
irreconcilable counts the re-derivation above replaces.)

The manifest is DERIVED by AST walk, never hand-maintained -- the lesson from
dormancy_gate._external_roots(): a hand-written list cannot see a wire nobody wrote down.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.gates.import_manifest
"""

import ast
import importlib
import os
import sys

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
    "gates",  # G8 S1a (evidence g8-reorganization.md 2.5 item 6, F15): the package it creates
    "instruments",  # G8 M2: the package created by the instrument move
    "sensorimotor",  # G8 M3: the package created by the sensorimotor move
    "infra",  # G8 M4: the package created by the infrastructure move
    "growth",  # G8 M5: the package created by the growth move
    "reservoir",  # G8 M6: the package created by the reservoir move
    "readout",  # G8 M7: the package created by the readout move
    "sleep",  # G8 M8: the package created by the sleep move
)

# The tiers no gate covers. A governed -> ungoverned edge is what this module protects.
# IN-PACKAGE shape: matched as the prefix `snn_data_lake.<u>`.
UNGOVERNED = ("reasoning", "training")

# ROOT shape: tiers that live OUTSIDE the package, so a governed module names them by their BARE
# top-level name and the prefixed test above can never match. ADDED 2026-08-05 because
# `cortex/brain.py` built its perception tier from three such imports -- `training.eprop_nmnist`,
# `training.surrogate_snn`, `training.eprop_shd` -- all invisible to the one gate built for exactly
# that failure. Widening the scan is what made them visible; it did NOT make brain.py's own import
# CWD-proof (see broken()).
#
# ALL THREE ARE NOW GONE, and the tuple is empty as of specs/013 C3 (2026-08-13): eprop_nmnist went
# in-package at the 2026-08-12 PROMOTE lane, surrogate_snn was RETIRED at C1, eprop_shd was promoted
# at C3. The shape is kept, with its zero case pre-authorised at the `unmatched` assert, because the
# NEXT bare-root import is the one it exists to catch -- add a name here the day one is measured.
#
# Line cites in this block are deliberately coarse: the 2026-08-12 review pass caught this very
# comment quoting a `try` span (:557-643) that insertions above it had rotted -- exactly the failure
# :18 names -- so it no longer pins interior line numbers it cannot keep current.
UNGOVERNED_ROOT = ()
# EMPTIED 2026-08-13 (specs/013 C3): `training.eprop_shd` was the LAST bare-root import from the
# governed tier, and it became in-package (`snn_data_lake.sensorimotor.eprop_shd`) at that commit. The
# empty tuple is the pre-authorised terminal state -- see the note at the `unmatched` assert below,
# which says in so many words that removing the name is the correct edit once `training/` is
# promoted INTO the package, and must not leave an assert that can never again be satisfied.
#
# HONEST LIMIT, because the name invites over-reading: this records "no AST-VISIBLE root-shape
# imports", NOT "no governed -> training/ edges". A surviving invisible one, measured at specs/013:
# `reasoning/eprop_lm_serve.py` does `sys.path.insert` on <root>/training and then
# `import eprop_lm as trainer` -- a bare-NAME import that `_targets()` cannot see by construction.

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# <repo>, the parent of `src/`. Pure path arithmetic evaluated at import -- no side effect (nothing is
# inserted on sys.path here; see broken(), which scopes and undoes that itself).
_ROOT_DIR = os.path.dirname(os.path.dirname(_PKG_DIR))


def _targets(tree):
    """Every ungoverned module named by an import in one parsed file -- BOTH shapes: the in-package
    `snn_data_lake.<UNGOVERNED>...` prefix and a bare `<UNGOVERNED_ROOT>[.sub]` root name.
    Covers `import a.b`, `from a.b import c`, and the LAZY/GUARDED forms (an import nested inside a
    function or a try block is an ordinary ast.Import node -- ast.walk sees it either way, which is
    the entire point).

    The root test is `m == r or m.startswith(f"{r}.")`, deliberately NOT `m.startswith(r)`: the loose
    form would also claim a third-party `trainingkit`, which is a gate reporting on a module nobody
    in this tree imports."""
    for n in ast.walk(tree):
        names = []
        if isinstance(n, ast.ImportFrom) and n.module:
            names = [n.module]
        elif isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        for m in names:
            if any(m.startswith(f"snn_data_lake.{u}") for u in UNGOVERNED) or any(
                m == r or m.startswith(f"{r}.") for r in UNGOVERNED_ROOT
            ):
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
    BROKEN one, and the swallows hide both identically.

    COST OF THE ROOT TIER, MEASURED so it is not discovered as a mystery slowdown. All three
    `training.*` modules `import torch` at module scope, which the 11 `reasoning.*` targets did not
    pull in: measured by calling this function with the manifest filtered both ways in one process,
    `'torch' in sys.modules` is False after the 11 `reasoning.*` targets and True after all 14.
    RE-MEASURED 2026-08-05 and the previous figure here (`1.83s -> 7.27s`) did NOT reproduce -- it
    also never said WHICH span it timed, which is why it could not be checked. Both spans, stated so
    the next reader re-derives rather than inherits; fair alternating A/B, 4 runs each, same
    interpreter and invocation, median of the middle two, WARM filesystem cache:
      this function alone, 11 targets -> 14:  0.17s -> 1.40s
      whole-process `python -m ...`, 11 -> 14: 1.03s -> 2.60s
    Cache state moves these (a cold first torch import is several times either figure), so the
    durable reading is the RATIO and the sign, not the seconds. That is torch's import, paid once
    per process. It is NOT worth avoiding with
    find_spec -- proving these modules LOAD is the entire assertion, and find_spec would green a
    module whose body raises, which is a constructed and RUN red for this gate (a `training/` whose
    `eprop_nmnist.py` raises at import is caught here as `RuntimeError`, and find_spec would miss
    it). Green under NORD_TORCH=0
    as well, because that is a runtime off-switch, not an uninstall: torch is an unconditional core
    dependency, so if it is ever absent these three targets correctly report BROKEN.

    The ROOT tier is resolved against `_ROOT_DIR`, never against the CWD. MEASURED 2026-08-05 with
    `PYTHONPATH=src` (on `training.eprop_nmnist`, then still root-tier; `training.eprop_shd` carries
    the same shape today): the root module imports when the process starts in the repo root
    (`python -m` puts the CWD at sys.path[0] -- printed 'D:\\AI-project\\spiking-neural-data-lake')
    and raises `ModuleNotFoundError: No module named 'training'` from any other directory. A
    CWD-resolved gate would therefore report the root-tier perception modules BROKEN whenever it ran from
    elsewhere: a red about the caller, not about the tree, and a gate whose colour depends on the CWD
    is not an instrument. The insert is function-scoped and undone in `finally` -- there is
    deliberately no module-scope sys.path mutation, because run_selfchecks runs part of the suite in
    one process.

    NOT COVERED -- and this was a real defect in a file this gate does not own, CLOSED 2026-08-13 by
    specs/013 and kept here because the shape recurs. A bare `from training....` import in
    `cortex/brain.py::Brain.__init__` resolves only if the REPO ROOT is on
    sys.path, which `PYTHONPATH=src` alone does not put there -- it arrives as sys.path[0], set from
    the start directory at interpreter start. A brain whose process did not start in the repo root
    therefore dropped its whole perception tier into a shared `except Exception` and ticked on at the
    fallback tier. BOTH halves are gone now: specs/013 promoted the last such import in-package and
    SPLIT that handler, so vision and audio fail independently. What survives is the LESSON -- the
    next bare-root import re-creates it. MEASURED then, and the invocation form decides it, so do not
    generalise either way: under `-m` or `script.py` sys.path[0] is an ABSOLUTE path fixed at
    interpreter start, and a later `os.chdir` to the repo root does NOT repair the import (printed
    `sys.path unchanged by chdir: True`, then `ModuleNotFoundError: No module named 'training'`);
    under `python -c` sys.path[0] is `''`, which re-resolves against the CWD, so there the same chdir
    DOES repair it. This gate proves those modules EXIST AND LOAD; it does not prove brain.py can
    reach them from an arbitrary CWD, and it must not be read as if it did."""
    man = manifest() if man is None else man
    bad = {}
    added = _ROOT_DIR not in sys.path
    if added:
        sys.path.insert(0, _ROOT_DIR)
    try:
        for m in man:
            try:
                importlib.import_module(m)
            except Exception as e:  # noqa: BLE001 -- the point is to REPORT it, callers don't
                bad[m] = f"{type(e).__name__}: {e}"
    finally:
        if added:
            sys.path.remove(_ROOT_DIR)
    return bad


def main():
    man = manifest()
    # A manifest that has gone empty means the scan broke, not that the coupling vanished -- a
    # constant where a computed value belongs is this project's recurring silent-degradation tell.
    assert man, "manifest is EMPTY -- the AST scan is broken, not the coupling"
    # COUNTED BEFORE IT WAS MADE TO ASSERT (the project rule: a widened gate is counted first, and a
    # rubber-stamp-sized list of failures would condemn the gate, not the tree). MEASURED 2026-08-05
    # on this tree: 14 targets / 15 call sites, of which 11/12 are `reasoning.*` (unchanged) and 3/3
    # are the newly visible root tier -- and 0 BROKEN. Three real modules, not an allowlist.
    # The floor is 13, not 14, and not the old 10:
    #   - 13 keeps exactly the one-target slack the previous 10-against-11 pair had, because
    #     promotion out of `reasoning/` into a governed region legitimately SHRINKS this manifest
    #     (P4c's stated target is zero `reasoning/`), and a floor with no slack turns every correct
    #     promotion into a red.
    #   - dropping the `UNGOVERNED_ROOT` tuple outright also trips this line today, because the
    #     count falls 14 -> 11 (constructed and RUN). But that is a COINCIDENCE of the current
    #     `reasoning.*` count, not a property: this floor is a collapse detector, and the assert
    #     below is what actually watches the root tier. Leaving the floor at 10 would have let a
    #     collapse recur silently.
    # Floor lowered 13 -> 12 on 2026-08-12: laminar_local_readout promoted to cortex/, so its
    # single manifest entry (the brain.py:1382 lazy reach) legitimately left the manifest --
    # exactly the shrink-by-promotion case the comment above licenses. The count floor and the
    # root-tier blindness detector are two separate lines (see below) precisely so this edit
    # cannot silently delete the detector.
    # Floor lowered 12 -> 11 on 2026-08-12 (#10 lane 4f): brain_adapters/executive_control
    # promoted+renamed to cortex/competence_executive, so its one target (the dlb lazy reach,
    # two call sites) legitimately left the manifest -- same shrink-by-promotion case, and the
    # floor keeps its one-target slack rather than parking on the new count of 12.
    # Floor lowered 11 -> 10 on 2026-08-12 (PROMOTE lane, nef_compute -> cortex/): its one
    # target (the nengo_bridge:471 function-level reach) legitimately left the manifest -- same
    # shrink-by-promotion case, one-target slack kept. Three more READY promotions in this lane
    # (vsa_structure_mapping, rose_language, eprop_nmnist) each remove one target; each lowers
    # this floor by one IN ITS OWN COMMIT, per the batch discipline the regenerated
    # promotion-plans.json records (a second removal without its lowering reds this line).
    # Floor lowered 10 -> 9 on 2026-08-12 (PROMOTE lane 2/4, vsa_structure_mapping -> cortex/):
    # its one target (the brain.py:924 lazy reach) left the manifest; same convention.
    # Floor lowered 9 -> 8 on 2026-08-12 (PROMOTE lane 3/4, rose_language -> cortex/): its one
    # target (the brain.py _rose_serve lazy reach; tools/ is not scanned) left the manifest.
    # Floor lowered 8 -> 7 on 2026-08-12 (PROMOTE lane 4/4, eprop_nmnist -> cortex/ TORCH_TIER):
    # its root-shape target (brain.py's perception-tier construction reach) became in-package.
    # The lane is complete; the next lowering belongs to whichever future promotion removes a
    # target, same convention.
    # Floor lowered 7 -> 6 on 2026-08-13 (specs/013 C1, surrogate_snn RETIRED): its root-shape
    # target (brain.py's surrogate-BPTT fallback branch) left the manifest because the branch and
    # the module were DELETED, not promoted -- the first lowering in this file licensed by a
    # retirement rather than a move. The removal lands in THIS commit; a floor lowered without its
    # removal is silencing a check, which is the one thing this line must never become.
    # Floor lowered 6 -> 5 on 2026-08-13 (specs/013 C3, eprop_shd -> cortex/): its root-shape
    # target became in-package, the same shrink-by-promotion case, one-target slack kept. This
    # removal empties UNGOVERNED_ROOT, so BOTH edits land in this commit -- lowering the floor
    # without the removal, or emptying the tuple without the floor, reds this file either way.
    assert len(man) >= 5, f"manifest collapsed to {len(man)} targets -- scan regression?"

    # ROOT-TIER COVERAGE, asserted STRUCTURALLY so that no COUNT arbitrates it. The comment above
    # used to claim the floor of 13 was "also the value that makes root-tier blindness itself red".
    # Measured true today and fragile by construction: it holds only while `reasoning.*` happens to
    # number 11, and P4c's stated target is to drive that count to ZERO. Whoever lowers the floor
    # during the next correct promotion would silently delete the blindness detector along with it
    # -- this tree's recurring silent-weakening shape, and the reason the two jobs are now two lines.
    #
    # This also MECHANIZES the rule `UNGOVERNED_ROOT`'s own comment states and could until now only
    # ask for ("Add a name here only when a bare-name import of it is measured"): a declared root
    # tier that no governed module actually imports by bare name is a gate claiming coverage it does
    # not have. Adding "research" or "integrations" -- which two audits proposed -- now fails here
    # instead of passing as an inert allowlist entry.
    #
    # Empty `UNGOVERNED_ROOT` is vacuously clean ON PURPOSE, checked because a new criterion owes its
    # zero/empty case: if `training/` is ever promoted INTO the package, removing the name is the
    # correct edit and must not leave an assert that can never again be satisfied.
    unmatched = sorted(set(UNGOVERNED_ROOT) - {m.split(".")[0] for m in man})
    assert not unmatched, (
        f"declared root tier(s) {unmatched} appear in UNGOVERNED_ROOT but are imported by NO "
        "governed module -- either the bare-name branch of _targets() has regressed (the perception "
        "tier goes invisible again) or the name was added on hope rather than on a measurement"
    )

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
