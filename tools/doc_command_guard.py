"""doc_command_guard -- a documented `python -m snn_data_lake.<mod>` must be RUNNABLE (P4c).

WHY THIS EXISTS, and why `citation_guard.py` does not already cover it.

`citation_guard.py` asserts that a tracked file never presents an UNTRACKED `.py` as a
runnable reproduction path. It resolves the FILE. It cannot resolve a `python -m` command,
because a dotted module path is not a filename -- and the failure this gate catches is a
command whose file is present, tracked, and still exits 1.

The measured incident, 2026-08-04. `docs/RUNNING.md` is the project's how-to-run page. Six
of its commands used the FLAT back-compat alias (`python -m snn_data_lake.interpreter`).
The alias layer in `snn_data_lake/__init__.py` resolves imports but has no `get_code`, so
every one of them died with:

    AttributeError: '_AliasLoader' object has no attribute 'get_code'

`SECURITY.md` documents that exact failure and gives the FULL dotted path as the fix, so the
project already knew -- the how-to page just did not follow it. A seventh command pointed at
`snn_data_lake.reasoning.spiking_storage_prototype`, which had been PROMOTED to
`hippocampus/` by the reasoning-promotion program (P4c debt), leaving the doc pointing at a
module that no longer exists there.

That is the shape: citation_guard went green on this page the whole time, because every
`.py` it could see was tracked. The command, which is what a reader actually types, was
never checked by anything.

WHAT IT CHECKS (static, ~1 s, no subprocess, no import of the target):

  1. the dotted path resolves to a real file under `src/snn_data_lake/`;
  2. the path is NOT a flat back-compat alias -- i.e. the module is addressed by its FULL
     `snn_data_lake.<region>.<module>` path, which is the only form `-m` can execute;
  3. the target module has an `if __name__ == "__main__":` block, so `-m` does something.

WHAT IT DOES NOT CHECK, stated so the green is not over-read: it never runs the command, so
it proves ADDRESSABILITY, not that the self-check passes. `run_selfchecks.py` is what proves
the second, and it is the right place for it -- this gate exists for the commands
`run_selfchecks.py` does not know about, which is every command printed in a document.

Usage:
    python tools/doc_command_guard.py            # report, exit 0
    python tools/doc_command_guard.py --assert   # exit 1 on any unexempted violation
    python tools/doc_command_guard.py --selfcheck
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "snn_data_lake"

# Documents whose commands are addressed to a reader. CHANGELOG.md is deliberately absent:
# a changelog entry describes a tree that existed, so a command that no longer runs is
# CORRECT there -- the same reasoning that killed the broad citation rule at 475 findings.
# CLAUDE.md was REMOVED from this tuple 2026-08-05, and the reason is about the GATE, not the file.
# This scope is a FILESYSTEM GLOB (see _iter_docs below), so it is git-independent -- unlike
# tools/citation_guard.py, which walks `git ls-files` and therefore sees only tracked files. CLAUDE.md
# is deliberately gitignored (`.gitignore`, "local-only -- kept on disk, not published to the public
# repo"), so naming it here made this gate scan a document that EXISTS IN NO CLONE: 55 documents on
# the owner's machine, 54 in CI and for every other reader. A gate whose corpus depends on who runs it
# cannot be used to compare two runs, which is most of what a gate is for.
# Dropping it costs no coverage, measured: CLAUDE.md contains 0 matches for the CMD regex below
# (it has no `python -m snn_data_lake.<...>` invocations), so the delta is 1 document and 0 commands.
# What IS lost is real and is recorded rather than hidden: CLAUDE.md carries 18 `file.py:NNN`
# citations that no gate checks, because citation_guard skips untracked files by construction. That
# is an accepted, documented exception under P4c -- see docs/POLICIES.md -- not an oversight.
SCOPE = (
    "README.md",
    "SECURITY.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "docs/**/*.md",
)

# `python -m snn_data_lake.a.b`, tolerating `.venv/bin/python`, env prefixes and flags.
#
# The negative lookahead is load-bearing and was added after the first counted run. Prose
# writes the TEMPLATE `python -m snn_data_lake.reasoning.<mod>` to tell a reader how to
# address any module in a tier; the regex stops at `.reasoning` (because `<mod>` is not an
# identifier) and would then report a real instruction as a broken command. Four of the
# first run's 24 findings were exactly that. Skipping a trailing `<` or `{` keeps the
# finding count a work list instead of padding it with correct lines.
CMD = re.compile(r"-m\s+(?P<mod>snn_data_lake(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![\w.]*[<{])")

# key: "document -> dotted module", value: why it is allowed to be unrunnable.
EXEMPT: dict[str, str] = {
    "SECURITY.md -> snn_data_lake.reflex": (
        "NEGATIVE EXAMPLE, quoted on purpose. SECURITY.md documents that the flat alias "
        "raises AttributeError under -m and prints this command to show the reader what "
        "NOT to type. Exempting it is correct; fixing it would delete the warning."
    ),
    "docs/RUNNING.md -> snn_data_lake.interpreter": (
        "NEGATIVE EXAMPLE, quoted on purpose. Same shape as the SECURITY.md entry: the "
        "note added 2026-08-04 quotes the flat form to show a reader the command that "
        "fails, immediately above the full dotted paths that work."
    ),
}


def _iter_docs():
    seen = set()
    for pat in SCOPE:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def _module_file(dotted: str) -> Path | None:
    """Resolve `snn_data_lake.x.y` to src/snn_data_lake/x/y.py, else None."""
    parts = dotted.split(".")
    if parts[0] != "snn_data_lake":
        return None
    cand = PKG.joinpath(*parts[1:]).with_suffix(".py")
    return cand if cand.is_file() else None


def _has_main(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
        ):
            return True
    return False


def scan():
    """Return (violations, ok_count, exempt_used). A violation is (doc, line, mod, why)."""
    violations = []
    ok = 0
    exempt_used = set()
    for doc in _iter_docs():
        rel = doc.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for m in CMD.finditer(line):
                dotted = m.group("mod")
                key = f"{rel} -> {dotted}"
                if key in EXEMPT:
                    exempt_used.add(key)
                    continue
                target = _module_file(dotted)
                if target is None:
                    # Distinguish the two failures: a flat alias that WOULD resolve as a
                    # module elsewhere in the package is a different (and more common)
                    # defect than a path to nothing, and the fix differs.
                    tail = dotted.split(".")[-1]
                    elsewhere = [
                        p.relative_to(PKG).as_posix()
                        for p in PKG.rglob(f"{tail}.py")
                        if "__pycache__" not in p.parts
                    ]
                    if len(dotted.split(".")) == 2 and elsewhere:
                        why = (
                            "FLAT ALIAS -- resolves as an import, CANNOT execute under -m "
                            "('_AliasLoader' has no get_code). Use the full dotted path: "
                            + ", ".join(
                                "snn_data_lake." + e[:-3].replace("/", ".") for e in elsewhere
                            )
                        )
                    elif elsewhere:
                        why = "MOVED -- module now at: " + ", ".join(
                            "snn_data_lake." + e[:-3].replace("/", ".") for e in elsewhere
                        )
                    else:
                        why = "NO SUCH MODULE under src/snn_data_lake/"
                    violations.append((rel, lineno, dotted, why))
                elif not _has_main(target):
                    violations.append(
                        (
                            rel,
                            lineno,
                            dotted,
                            "NO __main__ BLOCK -- `-m` on it is a no-op",
                        )
                    )
                else:
                    ok += 1
    return violations, ok, exempt_used


def selfcheck() -> None:
    """Prove the walk is still alive AND that the gate can go red (P8)."""
    # 1. the resolver really resolves, and really rejects.
    assert _module_file("snn_data_lake.brainstem.interpreter") is not None, (
        "resolver cannot find a module that exists -- the walk is dead"
    )
    assert _module_file("snn_data_lake.interpreter") is None, (
        "resolver accepted a FLAT alias path; that is the exact form this gate exists to reject"
    )
    assert _module_file("snn_data_lake.nope.nope") is None
    assert _module_file("os.path") is None

    # 2. __main__ detection distinguishes a runnable module from a plain one.
    runnable = PKG / "brainstem" / "interpreter.py"
    assert _has_main(runnable), f"{runnable} lost its __main__ block, or the AST walk broke"

    # 3. REACHABLE RED (P8/F11): construct the input that makes the gate fire. Without
    #    this the gate could be structurally incapable of failing and look identical.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "red.md"
        bad.write_text("run `python -m snn_data_lake.interpreter`\n", encoding="utf-8")
        hits = list(CMD.finditer(bad.read_text(encoding="utf-8")))
        assert len(hits) == 1, "the command regex stopped matching a documented -m command"
        assert _module_file(hits[0].group("mod")) is None, (
            "the red input no longer produces a violation -- gate is unfalsifiable"
        )

    # 4. every EXEMPT entry still blesses a REAL violation. A line-keyed exemption rots;
    #    one that blesses nothing reads as coverage while covering nothing.
    _, _, used = scan()
    unused = set(EXEMPT) - used
    assert not unused, f"EXEMPT entries that no longer match any command: {sorted(unused)}"
    print(f"  selfcheck OK: resolver alive, red reachable, {len(EXEMPT)} exemption(s) all live")


def main() -> int:
    print("=" * 78)
    print("doc_command_guard -- a documented `python -m` command must be runnable (P4c)")
    print("=" * 78)
    violations, ok, used = scan()
    ndocs = len(list(_iter_docs()))
    print(f"scanned {ndocs} documents; {ok + len(violations)} `python -m` command(s) found")
    print()
    if violations:
        print("--- UNRUNNABLE COMMANDS ---")
        for rel, lineno, dotted, why in violations:
            print(f"  {rel}:{lineno}")
            print(f"      python -m {dotted}")
            print(f"      {why}")
        print()
    print(f"RUNNABLE: {ok}   UNRUNNABLE: {len(violations)}   EXEMPT: {len(used)}")
    for k in sorted(used):
        print(f"  EXEMPT {k}: {EXEMPT[k]}")
    print()
    print("  NOTE: this gate proves a command is ADDRESSABLE, never that it passes.")
    print("  run_selfchecks.py proves the second, for the modules it knows about.")
    print()
    selfcheck()
    if violations and "--assert" in sys.argv:
        print("\ngate FAIL: a document prints a command a reader cannot run")
        return 1
    print("\ngate OK" if not violations else "\nreport only (pass --assert to enforce)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
