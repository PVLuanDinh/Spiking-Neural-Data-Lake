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

  4. (R0.2 row 15, 2026-10-17) `cargo test|run|bench [-p <pkg>] [--test X] [--bin X]
     [--example X] [--features F]` resolves against the tree's Cargo manifests (repo root,
     `native/**`, `tools/**`; `target/` skipped; read with `tomllib`): `<pkg>` is a
     `[package].name`; `--test X` is `tests/X.rs` | `tests/X/main.rs` | a `[[test]] name`;
     `--bin X` is `src/bin/X.rs` | `src/bin/X/main.rs` | a `[[bin]] name` | the package's own
     `src/main.rs`; `--example X` is `examples/X.rs` | `examples/X/main.rs` | a `[[example]]
     name`; every feature is a `[features]` key or an optional dependency. No `-p` resolves the
     target against every crate (a workspace run);
  5. `maturin develop|build -m <path>` needs `<path>` to be an existing Cargo.toml under the
     root (no `-m`: a Cargo.toml at the repo root);
  6. `python -m snn_data_lake._native` is the compiled extension, reported as NATIVE EXTENSION
     (built or not), never as NO SUCH MODULE. It is still unrunnable: `-m` on an extension
     module has no code object to execute (runpy), whichever way it was built.

WHAT IT DOES NOT CHECK, stated so the green is not over-read: it never runs the command, so
it proves ADDRESSABILITY, not that the self-check passes. `run_selfchecks.py` is what proves
the second, and it is the right place for it -- this gate exists for the commands
`run_selfchecks.py` does not know about, which is every command printed in a document.
Rust blind spots: Cargo manifests are found on DISK (the same git-independent view as the
documents above; a gitignored crate resolves here and not in CI); `--manifest-path`, `--workspace`, `--all-targets`, cargo aliases and
`.cargo/config.toml` are not read; a `[[test]]`-less test file whose name differs from its
target (a `harness = false` layout) resolves only by the table. `native/` ABSENT is printed as
such, never as 0 cargo findings.

Usage:
    python tools/doc_command_guard.py            # report, exit 0
    python tools/doc_command_guard.py --assert   # exit 1 on any unexempted violation
    python tools/doc_command_guard.py --selfcheck
"""

from __future__ import annotations

import ast
import re
import shlex
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "snn_data_lake"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# the package is imported LAZILY (inside scan / selfcheck / _native_line), as p4c_ratchet does:
# loading this tool runs no package __init__ (its interpreter guard, its alias finder).

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
# `cargo test -p sndl-core --test x`, `maturin develop -m native/x/Cargo.toml`: the rest of the
# command up to a shell separator or the closing backtick.
CARGO = re.compile(r"\bcargo\s+(?P<verb>test|run|bench)\b(?P<rest>[^`\n|;&]*)")
MATURIN = re.compile(r"\bmaturin\s+(?P<verb>develop|build)\b(?P<rest>[^`\n|;&]*)")
_TARGET_DIRS = {"test": "tests", "bin": "src/bin", "example": "examples"}

# key: "document -> dotted module", value: why it is allowed to be unrunnable.
EXEMPT: dict[str, str] = {
    # The "SECURITY.md -> snn_data_lake.reflex" entry lived here until 2026-08-29 and is
    # DELETED, not re-pointed. It exempted a quoted NEGATIVE EXAMPLE -- the flat
    # `python -m snn_data_lake.reflex` that SECURITY.md printed to show a reader the form
    # that raises AttributeError. `518b94c0` removed that quoted command (the prose warning
    # "Use the FULL dotted path -- the flat back-compat alias cannot execute a module as
    # `-m`" survives above the working dotted block) and did not remove the exemption in the
    # same change, so the entry blessed nothing for as long as it outlived its command.
    # Found by this gate's own both-ways staleness assert, which is the thing that assert is
    # for. The sibling RUNNING.md entry below is NOT stale: it still quotes its flat form
    # inline at docs/RUNNING.md:49.
    "docs/RUNNING.md -> snn_data_lake.interpreter": (
        "NEGATIVE EXAMPLE, quoted on purpose. The note added 2026-08-04 quotes the flat "
        "form to show a reader the command that fails, immediately above the full dotted "
        "paths that work. (This read 'Same shape as the SECURITY.md entry' until "
        "2026-08-29; that entry is gone, and a reason citing a deleted sibling is the same "
        "rot the staleness assert above exists to catch.)"
    ),
}


def _iter_docs(root=ROOT):
    seen = set()
    for pat in SCOPE:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def _manifests(root=ROOT):
    """{package name: (crate dir, parsed Cargo.toml)} for the manifests a documented command can name."""
    out = {}
    pats = ("Cargo.toml", "native/**/Cargo.toml", "tools/**/Cargo.toml")
    for p in sorted(q for pat in pats for q in root.glob(pat)):
        if "target" in p.relative_to(root).parts:
            continue
        try:
            table = tomllib.loads(p.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            continue  # a manifest cargo cannot read declares no package: its commands red below
        name = table.get("package", {}).get("name")
        if name:
            out[name] = (p.parent, table)
    return out


def _tokens(rest):
    try:
        return shlex.split(rest)
    except ValueError:
        return rest.split()


def _opts(tokens, *flags):
    """Every value given to any of `flags` (`-p x`, `--package=x`): cargo repeats selectors."""
    out = []
    for i, t in enumerate(tokens):
        if t in flags and i + 1 < len(tokens):
            out.append(tokens[i + 1])
        for f in flags:
            if t.startswith(f + "="):
                out.append(t[len(f) + 1 :])
            elif len(f) == 2 and len(t) > 2 and t.startswith(f):  # attached: `-pnope`
                out.append(t[2:])
    return out


def _has_target(crate, table, kind, name):
    d = crate / _TARGET_DIRS[kind]
    in_table = any(t.get("name") == name for t in table.get(kind, []))
    own_bin = (
        kind == "bin" and name == table["package"]["name"] and (crate / "src/main.rs").is_file()
    )
    return (d / f"{name}.rs").is_file() or (d / name / "main.rs").is_file() or in_table or own_bin


def _has_feature(table, feature):
    """A `[features]` key, or an optional dependency's implicit feature -- which cargo removes
    once any feature spells the dependency as `dep:<name>`."""
    features = table.get("features", {})
    deps = table.get("dependencies", {})
    optional = {k for k, v in deps.items() if isinstance(v, dict) and v.get("optional")}
    suppressed = {
        f.removeprefix("dep:") for fs in features.values() for f in fs if f.startswith("dep:")
    }
    return feature in features or feature in optional - suppressed


def _feature_why(feature, crates, manifests):
    """`feat` | `pkg/feat` (a tree crate's or a declared dependency's feature); `dep:` is manifest-only."""
    if feature.startswith("dep:"):
        return f"NO SUCH FEATURE {feature!r} -- `dep:` is manifest-only syntax, cargo rejects it on the command line"
    owner, _, feat = feature.rpartition("/")
    if owner in manifests:
        tables = [manifests[owner][1]]
    elif owner:
        if any(owner in t.get("dependencies", {}) for _, t in crates):
            return None  # a dependency's feature list is not in this tree
        return f"NO SUCH FEATURE OWNER {owner!r} -- neither a tree crate nor a declared dependency"
    else:
        tables = [t for _, t in crates]
    return None if any(_has_feature(t, feat) for t in tables) else f"NO SUCH FEATURE {feature!r}"


def _cargo_why(rest, manifests):
    """Why a `cargo test|run|bench <rest>` cannot run against `manifests`, else None."""
    tokens = _tokens(rest)
    tokens = (
        tokens[: tokens.index("--")] if "--" in tokens else tokens
    )  # the rest goes to the program
    if not manifests:
        return "NO CARGO MANIFEST in the tree (native/ ABSENT)"
    pkgs = _opts(tokens, "-p", "--package")
    for pkg in pkgs:
        if pkg not in manifests:
            return f"NO SUCH CRATE {pkg!r} -- the manifests declare: {', '.join(sorted(manifests))}"
    crates = [manifests[p] for p in pkgs] or list(manifests.values())
    for kind in _TARGET_DIRS:
        for name in _opts(tokens, "--" + kind):
            if not any(_has_target(c, t, kind, name) for c, t in crates):
                names = ", ".join(t["package"]["name"] for _, t in crates)
                return f"NO SUCH {kind.upper()} TARGET {name!r} in {names}"
    for feature in re.split(r"[,\s]+", " ".join(_opts(tokens, "--features", "-F"))):
        if feature and (why := _feature_why(feature, crates, manifests)):
            return why
    return None


def _maturin_why(rest, root):
    """Why a `maturin develop|build <rest>` cannot run from `root`, else None."""
    manifests = _opts(_tokens(rest), "-m", "--manifest-path")
    if not manifests:
        return (
            None
            if (root / "Cargo.toml").is_file()
            else "NO Cargo.toml AT THE REPO ROOT -- pass -m <crate>/Cargo.toml"
        )
    for manifest in manifests:
        p = (root / manifest).resolve()
        if not (p.is_file() and p.name == "Cargo.toml" and root.resolve() in p.parents):
            return f"NO SUCH MANIFEST {manifest!r} under the repo root"
    return None


def _native_line(root=ROOT):
    """The reader's own `native/` status line (git view; a hidden .rs is REPORTED here, the Rust
    gates are the ones that red on it), never 0 findings."""
    from snn_data_lake.gates.rust_facts import RustFactsUnavailable, rust_files

    try:
        n = len(rust_files(root))
    except RustFactsUnavailable as exc:
        return f"native/ UNREADABLE -- {exc}"
    return f"native/ PRESENT ({n} .rs files)" if n else "native/ ABSENT (0 .rs files)"


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


def scan(root=ROOT):
    """Return (violations, ok_count, exempt_used). A violation is (doc, line, command, why).
    `root` is the injectable tree for both the documents and the Cargo manifests; fixture
    trees never fork the logic. `python -m` paths always resolve against the real package."""
    violations = []
    ok = 0
    exempt_used = set()
    from snn_data_lake.gates.rust_facts import NATIVE_MODULE

    manifests = _manifests(root)
    for doc in _iter_docs(root):
        rel = doc.relative_to(root).as_posix()
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for pattern, why_of in (
                (CARGO, lambda rest: _cargo_why(rest, manifests)),
                (MATURIN, lambda rest: _maturin_why(rest, root)),
            ):
                for m in pattern.finditer(line):
                    why = why_of(m.group("rest"))
                    if why:
                        violations.append((rel, lineno, m.group(0).strip(), why))
                    else:
                        ok += 1
            for m in CMD.finditer(line):
                dotted = m.group("mod")
                key = f"{rel} -> {dotted}"
                if key in EXEMPT:
                    exempt_used.add(key)
                    continue
                target = _module_file(dotted)
                if dotted == NATIVE_MODULE:
                    violations.append(
                        (
                            rel,
                            lineno,
                            f"python -m {dotted}",
                            "NATIVE EXTENSION -- a compiled module; `-m` has no code object "
                            "to run (runpy), built or not",
                        )
                    )
                elif target is None:
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
                    violations.append((rel, lineno, f"python -m {dotted}", why))
                elif not _has_main(target):
                    violations.append(
                        (
                            rel,
                            lineno,
                            f"python -m {dotted}",
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

    from snn_data_lake.gates.rust_facts import NATIVE_MODULE

    # 5. REACHABLE RUST REDS (R0.2 row 15), through scan(root=fixture): a documented cargo
    #    target that no crate has, a crate no manifest declares, a maturin manifest that is
    #    not there, and `python -m` on the extension -- each exactly one violation -- beside the
    #    GREEN twin (the target exists -> 0 violations, 1 runnable).
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        crate = root / "native" / "sndl-core"
        (crate / "src").mkdir(parents=True)
        (crate / "Cargo.toml").write_text(
            '[package]\nname = "sndl-core"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        (crate / "src" / "lib.rs").write_text("pub fn f() {}\n", encoding="utf-8")
        (root / "docs").mkdir()
        doc = root / "docs" / "red.md"
        for line, prefix in (
            ("cargo test -p sndl-core --test lake_roundtrip", "NO SUCH TEST TARGET"),
            ("cargo test -p nope", "NO SUCH CRATE"),
            ("maturin develop -m native/sndl-python/Cargo.toml", "NO SUCH MANIFEST"),
            (f"python -m {NATIVE_MODULE}", "NATIVE EXTENSION"),
        ):
            doc.write_text(f"run `{line}`\n", encoding="utf-8")
            found, ok, _ = scan(root=root)
            assert len(found) == 1 and ok == 0 and found[0][3].startswith(prefix), (line, found)
        (crate / "tests").mkdir()
        (crate / "tests" / "lake_roundtrip.rs").write_text("#[test]\nfn t() {}\n", encoding="utf-8")
        doc.write_text("run `cargo test -p sndl-core --test lake_roundtrip`\n", encoding="utf-8")
        found, ok, _ = scan(root=root)
        assert found == [] and ok == 1, "the GREEN twin (target present) is not runnable"
    print(
        f"  selfcheck OK: resolver alive, red reachable, cargo/maturin reds reachable (4) with "
        f"their green twin, {len(EXEMPT)} exemption(s) all live"
    )


def main() -> int:
    print("=" * 78)
    print("doc_command_guard -- a documented `python -m` command must be runnable (P4c)")
    print("=" * 78)
    violations, ok, used = scan()
    ndocs = len(list(_iter_docs()))
    manifests = _manifests()
    print(
        f"scanned {ndocs} documents; {ok + len(violations)} `python -m` / cargo / maturin command(s) found"
    )
    print(f"Cargo manifests: {', '.join(sorted(manifests)) or 'none'}; {_native_line()}")
    print()
    if violations:
        print("--- UNRUNNABLE COMMANDS ---")
        for rel, lineno, cmd, why in violations:
            print(f"  {rel}:{lineno}")
            print(f"      {cmd}")
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
