#!/usr/bin/env python3
"""dependency_ledger -- every import declared, every declaration used (P7).

Built for specs/010-policy-compliance-audit. P7 (docs/POLICIES.md, owner policy 2026-08-04) makes
the dependency LEDGER the thing that matters, not the dependency COUNT:

    "It would be another dependency" is NOT a valid reason to reject anything. ... A
    declared-but-never-imported dependency is not a capability.

This gate checks the ledger in THREE directions and nothing else. It has no opinion about whether a
dependency should exist -- that is P7's judgement half, which is enforced by review because no gate
can tell you a framework was refused for a bad reason.

    UNDECLARED : imported under src/ and absent from pyproject.toml ENTIRELY -- core, any extra,
                 or build-system requires.

                 WHAT GREEN DOES AND DOES NOT PROVE, corrected 2026-08-05 by measurement rather
                 than left to be over-read. This line used to end "A clean `pip install .` produces
                 a tree that raises ModuleNotFoundError", which describes the RED but was quietly
                 read as the meaning of the GREEN. It is not. Green means every shipped import is
                 NAMED somewhere in pyproject; it does not mean `pip install .` imports cleanly.
                 A distribution declared only in an EXTRA is green here while a plain
                 `pip install .` still raises ModuleNotFoundError -- and with a marker
                 (`arcengine; python_version >= "3.12"`) it is green while even
                 `pip install ".[arc]"` installs NOTHING on this project's own `>=3.11` floor
                 (marker.evaluate() measured False on 3.9/3.10/3.11, True on 3.12+).

                 THE STRICTER CHECK IS NOT BUILT, and the reason is its COUNTED output, per this
                 project's own bar of counting a proposed gate before adopting it. "Unguarded under
                 src/ and declared only in an extra" emits 9 rows on this tree -- arcengine,
                 bindsnet, deltalake, ml-genn, nengo, polars, scikit-learn, tonic,
                 torchvision -- and 8 of them are the intended optional-tier shape, i.e. an
                 allowlist-shaped gate that trains a reader to ignore it (the same failure
                 `_is_sibling` was added to avoid). The ONE row that is not that shape is
                 arcengine, whose marker is False on the declared floor; it is recorded in
                 ALL_EXCLUDES["arc"] and in pyproject's own `arc` comment instead of being gated,
                 because the fix it would demand -- guarding the two imports, or raising the
                 project floor -- is a decision outside this gate.
    UNUSED     : declared in pyproject.toml and imported nowhere, with no stated reason.
    MISDESCRIBED : pyproject's own PROSE about its dependencies contradicts the machine-readable
                 lists beside it -- an install instruction that does not install what it claims.

THE THIRD DIRECTION WAS ADDED 2026-08-04, and it is the third of the three properties P7's gate
specification names, not an invention of this file. POLICIES.md's P7 asks for exactly:

    "(3) `pyproject.toml`'s own self-description matches its dependency list -- `:1` and `:27`
     both say "torch + numpy" while `mujoco==3.11.0` is a third unconditional core dependency."

That named defect is now FIXED in the file (both the header comment and `project.description` say
torch + numpy + mujoco), so the check lands green on the sentence that motivated it -- which is why
it ships with a constructed RED rather than resting on the incident (P8 burden 1). It is a section
of THIS tool rather than a second one, because a second pyproject-reading gate beside this one is
the additive shape P2 forbids: one mechanism reads the ledger, and self-description is a third
direction of the same ledger, not a different job.

WHAT MISDESCRIBED CHECKS, and each is a claim pyproject makes about itself in prose:

    CORE-PROSE  : every distribution in `[project.dependencies]` is NAMED in `project.description`
                  and in the leading header comment. A hard dependency the package's own one-line
                  description omits is the literal defect P7 cited.
    CORE-ONLY   : no distribution declared ONLY in an extra is named in `project.description`.
                  Scoped to `description` alone and never to the header comment, which legitimately
                  discusses extras, retired tiers and tooling by name -- widening it there scored 6
                  false positives on the first run, which is the dangerous direction.
    ALL-EXTRA   : every extra is reachable from `[all]` or is listed in ALL_EXCLUDES with a reason.
                  `all`'s own comment says "every addon"; an extra outside it that nobody decided to
                  exclude makes that sentence false, and `pip install ".[all]"` then silently does
                  not install a tier some other comment promises it does.

WHY THE IMPORT-NAME TABLE IS EXPLICIT AND TESTED, never inferred: the import name and the
distribution name differ often enough that guessing produces FALSE UNDECLARED reports, which is the
dangerous direction (Standing Invariant 3: the negative verdict is the one that retires a working
thing). `sklearn` is `scikit-learn`; `PIL` is `pillow`; `cv2` is `opencv-python`.

Reproduce:
    python tools/dependency_ledger.py             # report, exit 0
    python tools/dependency_ledger.py --assert    # exit 1 on any unexempted violation
    python tools/dependency_ledger.py --self-check  # CONSTRUCT AND RUN the red for each criterion
"""

import ast
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"

#: import name -> distribution name, for the cases where they differ. Explicit and tested; a wrong
#: entry here manufactures a false UNDECLARED, so this table is the instrument's main correctness
#: risk and is asserted in tests/test_dependency_ledger.py.
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "serial": "pyserial",
    "dateutil": "python-dateutil",
    "OpenGL": "pyopengl",
    "webview": "pywebview",
    "huggingface_hub": "huggingface-hub",
    "ml_genn": "ml-genn",
}

#: First-party roots -- this repo's own top-level packages/dirs. Not third-party, never reported.
#: `training`, `research`, `integrations`, `tools` are repo directories imported flat by scripts
#: that run from the repo root; the naive first run reported `training` as an undeclared PyPI
#: package because `cortex/brain.py` lazily imports it.
FIRST_PARTY = {"snn_data_lake", "training", "research", "integrations", "tools", "gcp", "infra"}

#: THE TWO DIRECTIONS ARE SCOPED ASYMMETRICALLY, ON PURPOSE.
#:
#: UNDECLARED is scoped to `src/` because `pyproject.toml:packages = ["src/snn_data_lake"]` -- only
#: `src/` ships in the wheel, so only an undeclared import THERE makes `pip install .` produce a
#: broken tree. `research/`, `integrations/` and `gcp/` are optional, ruff-excluded, not-shipped
#: tiers; an undeclared import in one of those breaks a research script, which is a real but
#: different and much lower-severity problem, and folding it in here produced 14 entries where 1 is
#: actionable.
#:
#: USED is scanned BROADLY, because a dependency imported anywhere in the repo is being used --
#: reporting `airllm` or `pywebview` as UNUSED because they live in `integrations/` would pressure a
#: real dependency out of the tree, the exact inverse of what P7 wants.
DECLARE_ROOTS = ("src",)
USE_ROOTS = ("src", "integrations", "tools", "tests", "training", "research", "gcp", "infra")

#: Distributions that are TOOLS, not imports: they are invoked as executables or plugins and are
#: correctly never imported by this project's code. Declared-and-unimported is their normal state,
#: so reporting them would be noise that trains a reader to ignore this gate.
TOOL_DISTS = {
    "pytest",
    "pytest-cov",
    "pytest-xdist",  # the battery-pilot parallel runner (owner-approved 2026-08-20): a pytest
    # plugin loaded by -n, correctly never imported by this project's code
    "ruff",
    "pre-commit",
    "pyinstaller",
    "hatchling",
}

#: EXEMPT entries follow dormancy_gate.EXEMPT's pattern: each carries a disposition and a REASON,
#: and the gate asserts every entry still matches a real violation, so an exemption that blesses
#: nothing fails loudly instead of reading as coverage.
EXEMPT = {
    # pyarrow's exemption DELETED 2026-08-19 (specs/022 T005): the staleness assert below caught
    # it blessing nothing the moment pyarrow was promoted to core with a real importer
    # (substrate/lake_segments.py, the sealed-segment container). An exemption that matches no
    # violation is deleted, not kept as decoration -- that is this gate's own rule working.
    "pillow": (
        "TRANSITIVE, declared explicitly on purpose. `[hf] = datasets, huggingface-hub, pillow` -- "
        "`datasets` needs pillow to decode Image features, which substrate/hf_stream.py streams. "
        "SEPARATELY TRACKED (specs/010 T021): substrate/arm_view.py::encode_png() hand-writes a PNG "
        "encoder while pillow sits declared here, which docs/POLICIES.md names as P7 incident #4 "
        "(the 'hand-written PNG encoder' row of P7's incident table). Adopting pillow there and "
        "DELETING the hand-rolled encoder is the P7+P2 move; it needs a measurement of what it "
        "buys, so it is not done here. Both citations were LINE numbers until 2026-08-04 and both "
        "had rotted -- `arm_view.py:46-74` landed on the frame-size constants and "
        "`POLICIES.md:819` on a P6 paragraph. Named by SYMBOL and by SECTION only: the 2026-08-04 "
        "correction still carried 'the def is at :55', and by the next morning encode_png had "
        "moved to :59 under an unrelated edit -- a replacement line cite rots on the same "
        "timescale as the one it replaced, which is the argument for not writing one at all."
    ),
}

#: The project's own distribution name -- `[all]` is a self-referential extra
#: (`spiking-neural-data-lake[arm,lake,gpu,...]`) and is not a dependency to account for.
SELF_DIST = "spiking-neural-data-lake"

#: The extra that claims to aggregate the others. `[all]`'s own comment reads "every addon".
AGGREGATE_EXTRA = "all"

#: Extras DELIBERATELY not reachable from `[all]`, each with the reason -- the ALL_EXCLUDES table
#: follows dormancy_gate.EXEMPT's pattern, and like EXEMPT it is asserted STALE-FREE below: an entry
#: naming an extra that IS in `[all]` (or that no longer exists) fails loudly rather than reading as
#: coverage. The point of the table is that every exclusion is a decision somebody made and can be
#: read, instead of `[all]` quietly meaning "the five extras someone remembered".
ALL_EXCLUDES = {
    "dev": "test/lint tooling, never a runtime dependency of the brain. `pip install .[dev]` is the contributor path and `[all]` is the capability path; folding dev in would put pytest and ruff into every capability install.",
    "exe": "BUILD-TIME ONLY (pyinstaller). It produces dist/snn-ui.exe from an already-installed tree; it is not a capability the installed brain gains.",
    "genn": "needs a C++ compiler and a CUDA toolchain on the host, so `pip install .[all]` would fail on any machine without them. This is the one exclusion `[all]`'s own comment already named.",
    "llm": "airllm -- the on-device LLM backend, deliberately quarantined (CLAUDE.md layout: `local_llm/ (optional LLM backend, quarantined)`). It is a multi-GB model-serving stack, and the brain's language path is brain-native by design.",
    "desktop": "pywebview -- a native GUI toolkit for the cockpit window. The cockpit is already reachable in any browser at :8137, so this buys a window, not a capability, and it pulls a platform GUI stack.",
    "hf": "datasets + huggingface-hub -- a network-backed dataset tier. substrate/hf_stream.py's self-check is deliberately OFFLINE and needs none of it (run_selfchecks CORE: 'offline self-check (no network, no datasets dep)'), so `[all]` stays installable with no network stack.",
    "nengo": "installing nengo CHANGES THE GATE: run_selfchecks grows NENGO_GATED from skipped to run, and nengo ships a pytest plugin whose hook signature is invalid under pytest 9.1.1 -- survivable only because `-p no:nengo` is in addopts. An aggregate extra must not silently move the gate count. RE-MEASURED 2026-08-05: `run_selfchecks.py --list` prints 280 modules in .venv-lake, where `importlib.util.find_spec('nengo')` is None and no NENGO_GATED entry is listed at all; installing nengo adds it. This entry quoted POLICIES.md P5's `277 -> 278` until today -- stale by the +3 that specs/010's CORE promotions produced, and re-running it is what showed that. The structural claim this exclusion actually rests on is that the count MOVES, which is unchanged; the absolute was decoration, so the new absolute is given only for the half that was measured here.",
    "arc": (
        "ADVERTISES A CAPABILITY THAT DOES NOT RUN. `[all]` is the capability path, and folding "
        "`arc` into it would promise an ARC-AGI faculty that raises on its first call: "
        "reasoning/arc_agi/play.py binds `arc_agi` to the FIRST-PARTY package (which is a single "
        "docstring line) and then calls `arc_agi.Arcade()`, which does not exist there -- an "
        "AttributeError on every interpreter, with arcengine installed or not. play_brain.py is "
        "worse: bare `from play import ...` with its sys.path bootstrap gutted to `pass`, so it "
        "cannot resolve from an installed wheel at all. Declaring the dependency makes both "
        "modules IMPORTABLE, never RUNNABLE. "
        "THIS REASON WAS PROMOTED FROM SECONDARY TO PRIMARY ON 2026-08-05, and the reason it "
        "replaced is recorded because the replacement is the finding. The original argument was "
        "INTERPRETER-CONDITIONAL: arcengine declares `Requires-Python: >=3.12` while pyproject "
        'declared `requires-python = ">=3.11"`, so the spec carried the marker '
        '`python_version >= "3.12"` and `pip install ".[all]"` RESOLVED on 3.11 by installing '
        "nothing -- an aggregate that meant two different things on two supported interpreters. "
        "Raising the floor to >=3.12 the same day made that argument describe nothing, so the "
        "exclusion stood for several hours on a reason that had become void. "
        "NO GATE CAUGHT IT, AND NONE CAN AS BUILT: STALE-EXCLUDE fires when an excluded extra IS "
        "in [all] or does not exist -- it checks membership STRUCTURALLY and never whether the "
        "written justification is still true. A prose reason is exactly the thing this file "
        "verifies for pyproject (MISDESCRIBED) and does not verify for itself. Recorded rather "
        "than silently rewritten, because 'the justification rotted while the gate stayed green' "
        "is the defect this project keeps finding, and here it landed inside the ledger. "
        "ALSO RECORDED, offered as a reason and rejected because it did "
        "not survive re-running: play.py's docstring says the dependency 'requires ... "
        "numpy>=2.2 (conflicts with tonic in .venv-cu128)', and that does NOT reproduce. "
        "Measured: arcengine 0.9.3 declares `numpy>=1.24.0` (not >=2.2), arc-agi 0.9.9 declares "
        "no numpy at all, and tonic 1.6.0 declares `numpy<2.0.0` -- 1.24 <= n < 2.0 satisfies "
        "both, so there is no measured numpy conflict to exclude on."
    ),
}


def _word_names(text, dists):
    """Which of `dists` are named in `text` as whole words. Substring matching would say `torch`
    is present in `snntorch` and `torchvision`, so the boundary is explicit."""
    found = set()
    for d in dists:
        if re.search(rf"(?<![\w.-]){re.escape(d)}(?![\w.-])", text, re.IGNORECASE):
            found.add(d)
    return found


def _header_comment(text):
    """The leading `#` comment block of pyproject.toml -- the file's prose about itself, which
    tomllib cannot see and which is where P7's cited `:1` defect lived."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            out.append(s.lstrip("#").strip())
        elif s:
            break
    return "\n".join(out)


def _extras_in_all(specs):
    """The extras `[all]` pulls in, from its self-referential `pkg[a,b,c]` specs."""
    got = set()
    for spec in specs:
        m = re.search(r"\[(.*?)\]", spec)
        if m:
            got |= {x.strip() for x in m.group(1).split(",") if x.strip()}
    return got


def misdescribed(text):
    """pyproject's PROSE about its dependencies vs the machine-readable lists beside it.

    Takes the file TEXT rather than reading the module-level path, so `--self-check` can run each
    criterion against a MUTATED copy and prove the red is reachable (P8 burden 1).

    Returns a list of (tag, message).
    """
    data = tomllib.loads(text)
    proj = data.get("project", {})
    core = {_dist_of(s) for s in proj.get("dependencies", [])} - {""}
    extras = proj.get("optional-dependencies") or {}
    extra_only = {_dist_of(s) for specs in extras.values() for s in specs} - core - {SELF_DIST, ""}
    desc = proj.get("description", "")
    header = _header_comment(text)
    out = []

    # CORE-PROSE: a hard dependency the file's own descriptions omit. P7's literal cited defect.
    for dist in sorted(core - _word_names(desc, core)):
        out.append(("CORE-PROSE", f"core dependency {dist!r} is not named in project.description"))
    for dist in sorted(core - _word_names(header, core)):
        out.append(
            ("CORE-PROSE", f"core dependency {dist!r} is not named in the header comment block")
        )

    # CORE-ONLY: description promoting an extra's package to core by prose.
    for dist in sorted(_word_names(desc, extra_only)):
        out.append(
            ("CORE-ONLY", f"project.description names {dist!r}, which is declared only in an extra")
        )

    # ALL-EXTRA: `[all]` says "every addon"; hold it to that or to a stated exclusion.
    if AGGREGATE_EXTRA in extras:
        reachable = _extras_in_all(extras[AGGREGATE_EXTRA])
        for name in sorted(set(extras) - reachable - {AGGREGATE_EXTRA} - set(ALL_EXCLUDES)):
            out.append(
                (
                    "ALL-EXTRA",
                    f"extra [{name}] is neither in [{AGGREGATE_EXTRA}] nor in ALL_EXCLUDES "
                    f'-- `pip install ".[{AGGREGATE_EXTRA}]"` does not install it',
                )
            )
        for name in sorted(set(ALL_EXCLUDES) & reachable):
            out.append(
                (
                    "STALE-EXCLUDE",
                    f"ALL_EXCLUDES[{name!r}] blesses nothing: it IS in [all] -- delete it",
                )
            )
        for name in sorted(set(ALL_EXCLUDES) - set(extras)):
            out.append(
                ("STALE-EXCLUDE", f"ALL_EXCLUDES[{name!r}] names an extra that does not exist")
            )
    return out


def _norm(name):
    """Normalize a distribution name for comparison (PEP 503-ish)."""
    return name.lower().replace("_", "-")


def _declared():
    """Every distribution named in pyproject, mapped to where it was declared."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    proj = data.get("project", {})
    out = {}
    for spec in proj.get("dependencies", []):
        out.setdefault(_dist_of(spec), set()).add("core")
    for extra, specs in (proj.get("optional-dependencies") or {}).items():
        for spec in specs:
            out.setdefault(_dist_of(spec), set()).add(extra)
    for spec in data.get("build-system", {}).get("requires", []):
        out.setdefault(_dist_of(spec), set()).add("build-system")
    out.pop("", None)
    return out


def _dist_of(spec):
    """'brian2>=2.10' -> 'brian2'; 'pkg[extra]==1.0' -> 'pkg'. Self-referential extras dropped."""
    s = spec.strip()
    for sep in ("[", ">", "<", "=", "!", "~", ";", " "):
        s = s.split(sep)[0]
    return _norm(s.strip())


def _imported(roots=USE_ROOTS):
    """Every third-party top-level import under src/, with its sites and whether it is guarded.

    An import is GUARDED when it sits inside a try/except or a function body -- it can fail without
    breaking module import. Unguarded module-top imports are the ones that break a clean install,
    so the distinction drives severity. It never excuses an absent declaration.
    """
    hits = {}
    files = []
    for rootname in roots:
        d = ROOT / rootname
        if d.is_dir():
            files.extend(sorted(d.rglob("*.py")))
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        guarded_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Try, ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    guarded_nodes.add(id(child))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import -> first-party
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in names:
                if not root or root in FIRST_PARTY or root in sys.stdlib_module_names:
                    continue
                if _is_sibling(path, root):
                    continue
                dist = _norm(IMPORT_TO_DIST.get(root, root))
                rel = path.relative_to(ROOT).as_posix()
                hits.setdefault(dist, []).append((rel, node.lineno, id(node) in guarded_nodes))
    return hits


def _is_sibling(path, root):
    """Is `root` a module living beside `path`, imported flat rather than as a package?

    ADDED after the first run, which is the whole point of counting a gate's output before adopting
    it. The naive version reported `arcengine`, `model`, `env`, `play`, `data`, `rule-inducer`,
    `softhebb-mnist` and more as UNDECLARED third-party packages. Every one is a LOCAL module in the
    same directory, imported flat because those scripts run with their own directory on sys.path.

    That first run was allowlist-shaped -- 13 entries, mostly false -- which under this project's
    adoption rule kills the gate or fixes the instrument. Fixing the instrument is correct here
    because the defect is a resolution bug, not a judgement call: a name that resolves to a FILE
    inside src/ is first-party however it was imported. Narrowing by name instead would have been
    the forbidden move (Standing Invariant 2), and would have hidden real findings too.
    """
    return root in _repo_module_names()


_REPO_MODULES = None


def _repo_module_names():
    """Every module name that exists as a FILE in this repo's own directories.

    Resolution by existence, not by location. The narrower sibling-directory check missed
    `eprop_lm`, `eprop_lm_alif` and `ar_spiking_lm`, which live in `training/` and are imported
    flat by serve scripts that put `training/` on `sys.path` -- the importer and the module are in
    different top-level dirs, so no ancestor walk from the importer finds them.

    What survives this check is the honest residue. `arcengine` does NOT exist anywhere in the
    repo, so it stays reported: a real, unguarded, undeclared third-party import.
    """
    global _REPO_MODULES
    if _REPO_MODULES is None:
        names = set()
        for rootname in USE_ROOTS:
            d = ROOT / rootname
            if not d.is_dir():
                continue
            for p in d.rglob("*.py"):
                names.add(p.stem)
            for p in d.rglob("__init__.py"):
                names.add(p.parent.name)
        _REPO_MODULES = names
    return _REPO_MODULES


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_assert = "--assert" in argv

    declared = _declared()
    imported = _imported(USE_ROOTS)  # broad: decides USED
    shipped = _imported(DECLARE_ROOTS)  # src/ only: decides UNDECLARED (see the scoping note)

    undeclared = {d: sites for d, sites in shipped.items() if d not in declared}
    unused = {
        d: loc
        for d, loc in declared.items()
        if d not in imported and d not in TOOL_DISTS and d != SELF_DIST
    }

    print("=" * 78)
    print("dependency_ledger -- every import declared, every declaration used (P7)")
    print("=" * 78)
    n_files = sum(1 for _ in SRC.rglob("*.py"))
    print(f"scanned {n_files} files under src/, {len(declared)} distributions in pyproject")
    print()

    live_exempt = set()

    print("--- UNDECLARED: imported under src/ but absent from pyproject ---")
    if not undeclared:
        print("  (none)")
    for dist, sites in sorted(undeclared.items()):
        if dist in EXEMPT:
            live_exempt.add(dist)
            continue
        unguarded = [s for s in sites if not s[2]]
        kind = f"{len(unguarded)} UNGUARDED of {len(sites)}" if unguarded else "all guarded"
        print(f"  {dist:24s} ({kind})")
        for rel, line, guarded in sites[:6]:
            print(f"      {'  ' if guarded else '!!'} {rel}:{line}")
    print()

    print("--- UNUSED: declared in pyproject, imported nowhere under src/ ---")
    shown_unused = []
    for dist, where in sorted(unused.items()):
        if dist in EXEMPT:
            live_exempt.add(dist)
            continue
        shown_unused.append(dist)
        print(f"  {dist:24s} [{', '.join(sorted(where))}]")
    if not shown_unused:
        print("  (none)")
    print()

    print("--- MISDESCRIBED: pyproject's prose about its dependencies vs its own lists ---")
    mis = misdescribed(PYPROJECT.read_text(encoding="utf-8"))
    if not mis:
        print("  (none)")
    for tag, msg in mis:
        print(f"  {tag:14s} {msg}")
    print()

    # An exemption that blesses nothing is worse than no exemption: it reads as coverage. Assert
    # every EXEMPT entry still matches a real violation (the citation_guard.EXEMPT pattern).
    stale = sorted(set(EXEMPT) - live_exempt)

    n_undeclared = len([d for d in undeclared if d not in EXEMPT])
    print(
        f"UNDECLARED: {n_undeclared}   UNUSED: {len(shown_unused)}   "
        f"MISDESCRIBED: {len(mis)}   EXEMPT: {len(live_exempt)}   "
        f"ALL_EXCLUDES: {len(ALL_EXCLUDES)}"
    )
    for dist in sorted(live_exempt):
        print(f"  EXEMPT {dist}: {EXEMPT[dist]}")
    if stale:
        print()
        print("  STALE EXEMPTIONS (they match no real violation -- delete them):")
        for dist in stale:
            print(f"    {dist}")
    print()

    failed = bool(n_undeclared or shown_unused or stale or mis)
    if failed:
        print("gate FAIL: the dependency ledger does not balance")
    else:
        print("gate OK: every import is declared, every declaration is used, prose matches lists")

    if do_assert and failed:
        return 1
    return 0


#: Each MISDESCRIBED criterion paired with a MUTATION of pyproject that must make it fire. This is
#: P8 burden 1 made executable: "construct the input that makes it fail, and record it. If no such
#: input exists, the criterion is DELETED, not weakened." A criterion whose red is only asserted in
#: prose is the shape this project has been burned by; `--self-check` runs them.
#:
#: The mutations are applied to a STRING COPY. Nothing on disk is touched.
_REDS = [
    (
        "CORE-PROSE",
        "drop mujoco from project.description (P7's literally cited defect)",
        lambda t: t.replace(
            "torch, numpy and mujoco are core dependencies.",
            "torch and numpy are core dependencies.",
        ),
    ),
    (
        "CORE-PROSE",
        "drop mujoco from the header comment block",
        # ANCHOR REWRITTEN 2026-08-19, by the same incident class the STALE-EXCLUDE comment
        # below records: promoting pyarrow appended it to the header's member list, the old
        # full-list anchor ("torch + numpy + mujoco are UNCONDITIONAL...") stopped matching,
        # and --self-check reported this criterion red-less -- the vacuous-red guard catching
        # its editor again. Anchor on the STABLE PREFIX ("torch + numpy + mujoco"), which
        # survives any future promotion appended after mujoco.
        lambda t: t.replace("torch + numpy + mujoco", "torch + numpy"),
    ),
    (
        "CORE-ONLY",
        "description promotes an extra-only package (pyserial) to core prose",
        lambda t: t.replace(
            "torch, numpy and mujoco are core dependencies.",
            "torch, numpy, mujoco and pyserial are core dependencies.",
        ),
    ),
    (
        "ALL-EXTRA",
        "a new extra nobody decided about",
        lambda t: t.replace('arm = ["pyserial"]', 'zzznew = ["requests"]\narm = ["pyserial"]'),
    ),
    (
        "STALE-EXCLUDE",
        "an ALL_EXCLUDES entry that is actually in [all]",
        # ANCHOR REWRITTEN 2026-08-05, and the reason is that this harness caught its own author.
        # It read `t.replace("[arm,lake,gpu,nmnist,train]", ...)` -- the literal MEMBER LIST of
        # [all]. Adding `nir` to [all] in the same pass changed that list, the mutation then
        # changed nothing, and `--self-check` printed `BROKEN STALE-EXCLUDE mutation anchor no
        # longer matches` and exited 1. That is the vacuous-red guard doing exactly its job, so
        # the fix is not to re-pin the new member list -- which would rot on the next edit to
        # [all], i.e. on precisely the edit this criterion exists to police. Anchor on the
        # SELF-REFERENTIAL PREFIX instead: `spiking-neural-data-lake[` is the syntax that makes
        # [all] an aggregate at all, it is what `_extras_in_all` parses, and it survives any
        # change to which extras are listed after it. Injecting an ALL_EXCLUDES member as the
        # first element makes `set(ALL_EXCLUDES) & reachable` non-empty, which is the red.
        lambda t: t.replace(f"{SELF_DIST}[", f"{SELF_DIST}[genn,"),
    ),
]


def self_check():
    """CONSTRUCT AND RUN the red for every MISDESCRIBED criterion. Returns 0 iff all reachable."""
    text = PYPROJECT.read_text(encoding="utf-8")
    print("=" * 78)
    print("dependency_ledger --self-check: is every MISDESCRIBED criterion falsifiable?")
    print("=" * 78)
    bad = []
    for tag, what, mutate in _REDS:
        mutated = mutate(text)
        if mutated == text:
            # A mutation that changed nothing proves nothing -- the anchor string has drifted and
            # the "red" below would be vacuously green. This is the failure mode a red-reachability
            # harness is most likely to hide from itself.
            print(f"  BROKEN  {tag:14s} mutation anchor no longer matches: {what}")
            bad.append((tag, what))
            continue
        tags = {t for t, _ in misdescribed(mutated)}
        ok = tag in tags
        print(f"  {'RED OK' if ok else 'NO RED'}  {tag:14s} {what}")
        if not ok:
            bad.append((tag, what))
    print()
    baseline = misdescribed(text)
    print(f"unmutated tree reports {len(baseline)} MISDESCRIBED finding(s)")
    for t, m in baseline:
        print(f"  {t:14s} {m}")
    print()
    if bad:
        print(f"self-check FAIL: {len(bad)} criterion/criteria with no reachable red")
        return 1
    print(f"self-check OK: {len(_REDS)}/{len(_REDS)} criteria proven falsifiable")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        sys.exit(self_check())
    sys.exit(main())
