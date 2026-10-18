"""r02b-cache (Rust port arc 2, DISPOSITION rows 13 / 15 / 17): the three CARGO-class gates must SEE
Rust bytes and files. Each test here is a pre-registered red (lanes/r0-arc2-1017/PREREG.md, section
"r02b-cache") with its clean control, run through the gate's real entry point with injected inputs.

A gate that has no injection point for an input mis-reports that input as "not an input" -- the
exact stale-green shape row 13 names -- so `_seam` turns a missing keyword into an assertion
failure that says so, never into a bare TypeError (red-first rule: a red fails on the mis-report,
not on an error).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from snn_data_lake.gates import selfcheck_cache as sc
from snn_data_lake.gates.rust_facts import NATIVE_MODULE, RustFactsUnavailable, rust_files

ROOT = Path(__file__).resolve().parents[1]
ENV = {"NORD_GPU": "1"}
EXT_SUFFIX = importlib.machinery.EXTENSION_SUFFIXES[0]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dcg = _tool("doc_command_guard")
p4c = _tool("p4c_ratchet")


def _seam(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except TypeError as exc:
        pytest.fail(
            f"{fn.__name__} has no injection point for this input ({exc}): the gate cannot see it"
        )


def _crate(root, name="sndl-core", lib="mod a;\n"):
    crate = root / "native" / name
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2024"\n', encoding="utf-8"
    )
    (crate / "src" / "lib.rs").write_text(lib, encoding="utf-8")
    (crate / "src" / "a.rs").write_text("pub fn a() {}\n", encoding="utf-8")
    return crate


# ---- 1. selfcheck_cache: the Rust surface joins the carry key ------------------------------------


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """A PLAIN module (no scan / roster / data / env) with a multi-file closure, picked as the
    self-check picks its probe."""
    store = str(tmp_path_factory.mktemp("store") / "s.json")
    base = sc.SelfcheckCache(store_path=store, env=ENV)
    mods = sorted(
        d[len(sc._PKG_PREFIX) :]
        for d in base._import_graph()
        if d.startswith(sc._PKG_PREFIX) and Path(sc._module_file(d)).is_file()
    )
    for m in ["engine.signal_ring"] + mods:
        if m not in mods:
            continue
        if (
            not any(base.describe(m)[k] for k in ("scans", "roster", "data", "envs", "native"))
            and len(base._closure(m)) > 1
        ):
            return store, m, base.closure_key(m)
    pytest.fail("no plain multi-file module to probe")


def _native_fixture(tmp_path):
    crate = _crate(tmp_path)
    return tmp_path / "native", crate / "src" / "lib.rs", crate / "Cargo.toml"


def _cache(store, native, **kw):
    return _seam(sc.SelfcheckCache, store_path=store, env=ENV, native_root=str(native), **kw)


def _inj(m):
    """The brief's injection: the probe's closure gains the extension as a graph node."""
    return {sc._PKG_PREFIX + m: {NATIVE_MODULE}}


def test_cache_rust_edit_flips_native_reader_key(tmp_path, probe):
    store, m, k_plain = probe
    native, lib, _ = _native_fixture(tmp_path)
    inj = _inj(m)
    base = _cache(store, native, graph_overrides=inj)
    k0 = base.closure_key(m)
    d = base.describe(m)
    assert d.get("native") and d["native"] != "ABSENT", (
        f"a module whose closure reads {NATIVE_MODULE} is keyed as if native/ were absent: {d}"
    )
    flipped = _cache(store, native, graph_overrides=inj, file_hash_overrides={str(lib): "d" * 64})
    assert flipped.closure_key(m) != k0, (
        "a native/**/*.rs byte change did not flip the reader's key (STALE GREEN)"
    )
    assert k0 != k_plain, "injecting the _native dependency did not change the key at all"
    assert base.why(m) == "native"


def test_cache_derived_native_import_is_keyed_without_a_graph_node(tmp_path, probe):
    """The PRODUCTION path: the import graph keys .py files only, so a reader is found by
    deriving its own source (absolute spelling here; the relative pins are below)."""
    store, m, k_plain = probe
    native, lib, _ = _native_fixture(tmp_path)
    probe_file = sc._module_file(sc._PKG_PREFIX + m)
    src = Path(probe_file).read_text(encoding="utf-8") + f"\nfrom {NATIVE_MODULE} import zzz\n"
    text = {probe_file: src}
    reader = _cache(store, native, file_text_overrides=text)
    k0 = reader.closure_key(m)
    assert reader.describe(m)["native"] == "2 .rs files", reader.describe(m)  # lib.rs, a.rs
    assert k0 != k_plain
    flipped = _cache(
        store, native, file_text_overrides=text, file_hash_overrides={str(lib): "d" * 64}
    )
    assert flipped.closure_key(m) != k0, (
        "a derived _native reader carried a stale .rs (STALE GREEN)"
    )


@pytest.mark.parametrize(
    "name",
    [
        "sndl-core/Cargo.toml",
        "sndl-core/Cargo.lock",
        "sndl-core/build.rs",
        "rust-toolchain.toml",
        "sndl-core/params.json",  # Codex B1 r02b-cache 1
    ],
)
def test_cache_every_rust_input_under_native_flips_the_key(tmp_path, probe, name):
    store, m, _ = probe
    native, _, _ = _native_fixture(tmp_path)
    path = native / name
    path.write_text("x\n", encoding="utf-8")
    inj = _inj(m)
    k0 = _cache(store, native, graph_overrides=inj).closure_key(m)
    fork = _cache(store, native, graph_overrides=inj, file_hash_overrides={str(path): "e" * 64})
    assert fork.closure_key(m) != k0, f"{name} change did not flip the key"


def test_cache_root_manifest_and_extension_flip_and_target_does_not(tmp_path, probe):
    store, m, _ = probe
    native, _, _ = _native_fixture(tmp_path)
    toolchain = tmp_path / "rust-toolchain.toml"  # beside native/: the repo root's pin
    toolchain.write_text('[toolchain]\nchannel = "1.96.0"\n', encoding="utf-8")
    built = native / "sndl-core" / "target" / "debug" / "build.rs"
    built.parent.mkdir(parents=True)
    built.write_text("// cargo output\n", encoding="utf-8")
    inj = _inj(m)
    base = _cache(store, native, graph_overrides=inj)
    k0 = base.closure_key(m)
    ext = Path(base.describe(m)["ext"])
    assert ext.suffix in (".pyd", ".so"), base.describe(m)
    for path in (toolchain, ext):
        fork = _cache(store, native, graph_overrides=inj, file_hash_overrides={str(path): "e" * 64})
        assert fork.closure_key(m) != k0, f"{path.name} change did not flip the key"
    same = _cache(store, native, graph_overrides=inj, file_hash_overrides={str(built): "e" * 64})
    assert same.closure_key(m) == k0, "a file under target/ (cargo output) flipped the key"


def test_cache_native_edit_leaves_a_plain_module_alone(tmp_path, probe):
    store, m, k_plain = probe
    native, lib, manifest = _native_fixture(tmp_path)
    inj = _inj(m)
    same = _cache(store, native, graph_overrides=inj)
    again = _cache(store, native, graph_overrides=inj)
    assert same.closure_key(m) == again.closure_key(m), "the native row is not deterministic"
    flips = {str(lib): "d" * 64, str(manifest): "e" * 64, same.describe(m)["ext"]: "f" * 64}
    untouched = _cache(store, native, file_hash_overrides=flips)
    assert untouched.closure_key(m) == k_plain, (
        "a Rust edit flipped a module that never reads _native"
    )
    assert not untouched.describe(m)["native"] and not untouched.describe(m)["ext"]


def test_cache_record_rehashes_the_rust_surface(tmp_path, probe):
    store, m, _ = probe
    native, lib, _ = _native_fixture(tmp_path)
    inj = _inj(m)
    reader = _cache(store, native, graph_overrides=inj)
    reader.closure_key(m)
    lib.write_text("pub fn f() {}\n", encoding="utf-8")  # a Rust edit between key and record
    reader.record(m, "now")
    fresh = _cache(store, native, graph_overrides=inj)
    assert fresh.consult(m) is not None, "record() stored a key built from the stale Rust surface"
    lib.write_text("pub fn g() {}\n", encoding="utf-8")
    assert _cache(store, native, graph_overrides=inj).consult(m) is None


@pytest.mark.parametrize(
    "src,want",
    [
        ("from snn_data_lake._native import f\n", True),
        ("import snn_data_lake._native\n", True),
        ("from snn_data_lake import _native\n", True),
        ("from ._native import f\n", True),
        ("from . import _native\n", True),
        ("from .other._native import f\n", False),
        ("import os\n", False),
        ("import snn_data_lake\nsnn_data_lake._native.f()\n", True),  # attribute spelling
        ('__all__ = ("_native",)\n', True),
        ('__all__: list[str] = ["_native"]\n', True),
        ("def broken(:\n", True),  # unparseable: fail closed, a reader until proven otherwise
    ],
)
def test_cache_facts_derive_a_native_reader(src, want):
    facts = sc._facts(src, 1, re.compile(r"(?!x)x"))
    assert facts.get("native", False) is want, facts


def test_cache_facts_resolve_a_deeper_relative_spelling():
    null = re.compile(r"(?!x)x")
    assert sc._facts("from .._native import f\n", 2, null)["native"] is True
    assert sc._facts("from ._native import f\n", 2, null)["native"] is False


@pytest.mark.parametrize(
    "init,line",
    [
        (sc._PKG + ".__init__", "from . import _native\n"),
        (sc._PKG + ".__init__", '__all__ = ["_native"]\n'),  # `from snn_data_lake import *`
        (sc._PKG + ".engine.__init__", "from .. import _native\n"),  # a region __init__
    ],
)
def test_cache_init_reexport_lifts_the_rust_surface_into_the_envelope(tmp_path, probe, init, line):
    store, m, _ = probe
    native, lib, _ = _native_fixture(tmp_path)
    init = sc._module_file(init)
    src = Path(init).read_text(encoding="utf-8") + "\n" + line
    base = _cache(store, native, file_text_overrides={init: src})
    fork = _cache(
        store, native, file_text_overrides={init: src}, file_hash_overrides={str(lib): "d" * 64}
    )
    assert fork.envelope != base.envelope, (
        f"{init} + {line!r} did not key the .rs into the envelope"
    )


def test_cache_ext_row_hashes_the_extension_find_spec_loads(tmp_path, probe, monkeypatch):
    store, m, _ = probe
    native, _, _ = _native_fixture(tmp_path)
    loaded = tmp_path / (
        sc._NATIVE_LEAF + EXT_SUFFIX
    )  # a built extension found elsewhere on sys.path
    loaded.write_bytes(b"MZ")
    spec = importlib.machinery.ModuleSpec(NATIVE_MODULE, None, origin=str(loaded))
    monkeypatch.setattr(sc.importlib.util, "find_spec", lambda name: spec)
    inj = _inj(m)
    base = _cache(store, native, graph_overrides=inj)
    assert Path(base.describe(m)["ext"]) == loaded
    k0 = base.closure_key(m)
    canonical = str(Path(sc._repo_root()) / "src" / sc._PKG / (sc._NATIVE_LEAF + EXT_SUFFIX))
    flip_loaded = _cache(
        store, native, graph_overrides=inj, file_hash_overrides={str(loaded): "f" * 64}
    )
    flip_unused = _cache(
        store, native, graph_overrides=inj, file_hash_overrides={canonical: "f" * 64}
    )
    assert flip_loaded.closure_key(m) != k0, "the loaded extension's bytes are not in the key"
    assert flip_unused.closure_key(m) == k0, "an unloaded canonical file moved the key"

    def broken(name):
        raise ImportError(name)

    def odd(name):
        raise ValueError(name)

    stale = importlib.machinery.ModuleSpec(NATIVE_MODULE, None, origin=str(tmp_path / "gone.pyd"))
    for find_spec in (broken, odd, lambda name: stale):
        monkeypatch.setattr(sc.importlib.util, "find_spec", find_spec)
        got = _cache(store, native, graph_overrides=inj).describe(m)["ext"]
        assert Path(got) == Path(canonical), (find_spec, got)


def test_cache_manifest_only_native_is_absent_in_describe_but_keyed(tmp_path, probe):
    store, m, _ = probe
    manifest = tmp_path / "native" / "Cargo.toml"
    manifest.parent.mkdir()
    manifest.write_text("[workspace]\n", encoding="utf-8")
    base = _cache(store, tmp_path / "native", graph_overrides=_inj(m))
    assert base.describe(m)["native"] == "ABSENT"  # no .rs: the sibling gates' ABSENT
    assert base.why(m) == "native"
    fork = _cache(
        store,
        tmp_path / "native",
        graph_overrides=_inj(m),
        file_hash_overrides={str(manifest): "e" * 64},
    )
    assert fork.closure_key(m) != base.closure_key(m), "a manifest-only native/ is not keyed"


def test_cache_real_tree_prints_native_absent_distinctly(probe):
    store, m, _ = probe
    inj = _inj(m)
    d = _seam(sc.SelfcheckCache, store_path=store, env=ENV, graph_overrides=inj).describe(m)
    if rust_files(ROOT):
        assert d["native"] and d["native"] != "ABSENT"
    else:
        assert d["native"] == "ABSENT", d
    assert d["ext"].endswith((".pyd", ".so"))


def test_cache_git_view_of_native_data_files(tmp_path, probe):
    """Codex B1 r02b-cache 1 controls, in a fixture git repo: a tracked and an
    untracked-unignored data file flip a `_native` reader's key, a gitignored one and target/
    do not (the git branch, not the hash-everything fallback) while a gitignored .rs and
    Cargo.lock still do (disk view); with `no_git=True` the fallback keys the ignored file
    too (fail closed)."""
    store, m, _ = probe
    native, _, _ = _native_fixture(tmp_path)
    crate = native / "sndl-core"
    tracked, untracked, ignored = (crate / n for n in ("params.json", "new.json", "scratch.json"))
    for p in (tracked, untracked, ignored):
        p.write_text("{}\n", encoding="utf-8")
    hidden_rs, hidden_lock = crate / "src" / "gen.rs", crate / "Cargo.lock"
    hidden_rs.write_text("pub fn g() {}\n", encoding="utf-8")
    hidden_lock.write_text("# lock\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("scratch.json\ngen.rs\nCargo.lock\n", encoding="utf-8")
    built = crate / "target" / "debug" / "out.json"
    built.parent.mkdir(parents=True)
    built.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "native/sndl-core/params.json"], check=True
    )
    inj = _inj(m)

    def key(path, **kw):
        fake = {str(path): "e" * 64}
        return _cache(
            store, native, graph_overrides=inj, file_hash_overrides=fake, **kw
        ).closure_key(m)

    k0 = _cache(store, native, graph_overrides=inj).closure_key(m)
    assert key(tracked) != k0, "a tracked data file change did not flip the key"
    assert key(untracked) != k0, "an untracked-unignored data file change did not flip the key"
    assert key(ignored) == k0, "a gitignored data file flipped the key under git's view"
    assert key(hidden_rs) != k0, "a gitignored .rs change did not flip the key"
    assert key(hidden_lock) != k0, "a gitignored Cargo.lock change did not flip the key"
    assert key(built) == k0, "a file under target/ flipped the key"
    kn = _cache(store, native, graph_overrides=inj, no_git=True).closure_key(m)
    assert key(ignored, no_git=True) != kn, "no_git fallback did not key the ignored file"


def test_cache_tracked_native_data_file_flips_the_key_on_the_real_tree(probe):
    """Codex B1 r02b-cache 1: every file git tracks under native/ (here the RNG parity
    fixture the extension's tests read) keys a `_native` reader, not only .rs and manifests."""
    store, m, _ = probe
    data = ROOT / "native" / "sndl-core" / "tests" / "fixtures_rng.json"
    if not data.is_file():
        pytest.skip("no tracked non-Rust file under native/ on this tree")
    inj = _inj(m)
    k0 = _seam(sc.SelfcheckCache, store_path=store, env=ENV, graph_overrides=inj).closure_key(m)
    fork = _seam(
        sc.SelfcheckCache,
        store_path=store,
        env=ENV,
        graph_overrides=inj,
        file_hash_overrides={str(data): "e" * 64},
    )
    assert fork.closure_key(m) != k0, "a tracked native/ data file change did not flip the key"


# ---- 2. doc_command_guard: cargo / maturin commands resolve against the Cargo manifests ----------


def _doc(root, line):
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "cmd.md").write_text(f"run `{line}`\n", encoding="utf-8")


def _scan(root):
    violations, ok, _ = _seam(dcg.scan, root=root)
    return violations, ok


def test_doc_cargo_test_target_missing_is_a_violation(tmp_path):
    _crate(tmp_path)
    _doc(tmp_path, "cargo test -p sndl-core --test lake_roundtrip")
    violations, ok = _scan(tmp_path)
    assert len(violations) == 1 and ok == 0, violations
    assert violations[0][3].startswith("NO SUCH TEST TARGET"), violations[0]


def test_doc_cargo_test_target_present_is_runnable(tmp_path):
    crate = _crate(tmp_path)
    (crate / "tests").mkdir()
    (crate / "tests" / "lake_roundtrip.rs").write_text("#[test]\nfn t() {}\n", encoding="utf-8")
    _doc(tmp_path, "cargo test -p sndl-core --test lake_roundtrip")
    violations, ok = _scan(tmp_path)
    assert violations == [] and ok == 1


@pytest.mark.parametrize(
    "kind,name,decl,file,prefix",
    [
        ("test", "gone", 'name = "gone"\npath = "tests/deleted.rs"', None, "NO SUCH TEST TARGET"),
        ("test", "gone", 'name = "gone"\npath = "t/x.rs"', "t/x.rs", None),
        (
            "test",
            "gone",
            'name = "gone"',
            "tests/gone.rs",
            None,
        ),  # no `path`: tests/<name>.rs implied
        (
            "test",
            "gone",
            'name = "gone"',
            None,
            "NO SUCH TEST TARGET",
        ),  # the name alone never counts
        (
            "test",
            "gone",
            'name = "gone"\npath = "tests/deleted.rs"',
            "tests/gone.rs",
            "NO SUCH TEST TARGET",
        ),
        (
            "test",
            "gone",
            'name = "gone"\npath = 1',
            "tests/gone.rs",
            None,
        ),  # non-str path: inferred
        (
            "test",
            "gone",
            'name = "other"\npath = "t/x.rs"',
            "t/x.rs",
            "NO SUCH TEST TARGET",
        ),  # unrelated
        ("test", "gone", 'name = "other"\npath = "tests/deleted.rs"', "tests/gone.rs", None),
        ("bin", "gone", 'name = "gone"\npath = "src/bin/deleted.rs"', None, "NO SUCH BIN TARGET"),
        ("bin", "gone", 'name = "gone"\npath = "src/cli.rs"', "src/cli.rs", None),
        (
            "bin",
            "sndl-core",
            'name = "sndl-core"\npath = "src/bin/deleted.rs"',
            "src/main.rs",
            "NO SUCH BIN TARGET",
        ),
        (
            "example",
            "gone",
            'name = "gone"\npath = "examples/deleted.rs"',
            None,
            "NO SUCH EXAMPLE TARGET",
        ),
        ("example", "gone", 'name = "gone"\npath = "ex/demo.rs"', "ex/demo.rs", None),
    ],
)
def test_doc_cargo_table_entry_needs_its_declared_source_on_disk(
    tmp_path, kind, name, decl, file, prefix
):
    """Codex B1 r02b-cache 2: a same-name `[[kind]]` entry's declared `path` alone decides; an
    unrelated entry neither satisfies a missing target nor overrides a valid inferred one; a
    non-string `path` (cargo rejects the manifest) falls to the inferred layout here."""
    crate = _crate(tmp_path)
    (crate / "Cargo.toml").write_text(
        f'[package]\nname = "sndl-core"\nversion = "0.1.0"\n[[{kind}]]\n{decl}\n',
        encoding="utf-8",
    )
    if file:
        (crate / file).parent.mkdir(parents=True, exist_ok=True)
        (crate / file).write_text("fn main() {}\n", encoding="utf-8")
    _doc(tmp_path, f"cargo {'test' if kind == 'test' else 'run'} -p sndl-core --{kind} {name}")
    violations, ok = _scan(tmp_path)
    if prefix is None:
        assert violations == [] and ok == 1, violations
    else:
        assert ok == 0 and len(violations) == 1 and violations[0][3].startswith(prefix), violations


def test_doc_cargo_unknown_crate_is_a_violation(tmp_path):
    _crate(tmp_path)
    _doc(tmp_path, "cargo test -p nope")
    violations, ok = _scan(tmp_path)
    assert len(violations) == 1 and violations[0][3].startswith("NO SUCH CRATE"), violations


def _full_crate(root):
    """A crate with every target kind, a feature and an optional dependency."""
    crate = _crate(root, lib="pub fn f() {}\n")
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "sndl-core"\nversion = "0.1.0"\n'
        '[features]\nsimd = []\nwire = ["dep:serde"]\n'
        '[dependencies]\nserde = { version = "1", optional = true }\nlog = "0.4"\n'
        '[[test]]\nname = "tabled"\npath = "t/x.rs"\n',
        encoding="utf-8",
    )
    (crate / "t").mkdir()
    (crate / "t" / "x.rs").write_text("#[test]\nfn t() {}\n", encoding="utf-8")
    (crate / "src" / "bin").mkdir()
    (crate / "src" / "bin" / "tool.rs").write_text("fn main() {}\n", encoding="utf-8")
    (crate / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (crate / "examples" / "demo").mkdir(parents=True)
    (crate / "examples" / "demo" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    return crate


@pytest.mark.parametrize(
    "line,prefix",
    [
        ("cargo run -p sndl-core --bin tool", None),
        ("cargo run -p sndl-core --bin sndl-core", None),  # the package's own src/main.rs
        ("cargo run -p sndl-core --bin nope", "NO SUCH BIN TARGET"),
        ("cargo run --example demo", None),  # no -p: every crate is searched
        ("cargo run -p sndl-core --example=nope", "NO SUCH EXAMPLE TARGET"),
        ("cargo test -p sndl-core --test tabled", None),  # a [[test]] table entry
        ("cargo bench -p sndl-core --features simd,wire", None),
        (
            "cargo bench -p sndl-core -F serde",
            "NO SUCH FEATURE",
        ),  # implicit feature suppressed by dep:
        ("cargo bench -p sndl-core -F dep:serde", "NO SUCH FEATURE"),  # manifest-only syntax
        ("cargo bench -p sndl-core --features sndl-core/simd", None),
        ("cargo bench -p sndl-core --features log/std", None),  # a declared dependency's feature
        ("cargo bench -p sndl-core --features nothere/simd", "NO SUCH FEATURE OWNER"),
        ("cargo bench -p sndl-core --features nope", "NO SUCH FEATURE"),
        ("cargo bench -p sndl-core --features simd --features nope", "NO SUCH FEATURE"),
        ("cargo test -p sndl-core -- --features nope --test nope", None),  # program arguments
        ("cargo test -p sndl-core -p nope", "NO SUCH CRATE"),
        ("cargo test", None),
    ],
)
def test_doc_cargo_targets_and_features_resolve(tmp_path, line, prefix):
    _full_crate(tmp_path)
    _doc(tmp_path, line)
    violations, ok = _scan(tmp_path)
    if prefix is None:
        assert violations == [] and ok == 1, (line, violations)
    else:
        assert ok == 0 and len(violations) == 1 and violations[0][3].startswith(prefix), (
            line,
            violations,
        )


def test_doc_manifests_skip_target_and_unreadable_toml(tmp_path):
    crate = _crate(tmp_path)
    built = crate / "target" / "package" / "ghost-0.1.0" / "Cargo.toml"
    built.parent.mkdir(parents=True)
    built.write_text('[package]\nname = "ghost"\n', encoding="utf-8")
    broken = _crate(tmp_path, "broken")
    (broken / "Cargo.toml").write_text("[package\nname = oops\n", encoding="utf-8")
    assert set(dcg._manifests(tmp_path)) == {"sndl-core"}
    for line in ("cargo test -p ghost", "cargo test -p broken"):
        _doc(tmp_path, line)
        violations, _ = _scan(tmp_path)
        assert len(violations) == 1 and violations[0][3].startswith("NO SUCH CRATE"), (
            line,
            violations,
        )
    _doc(tmp_path, "maturin develop -m native/sndl-core/src/lib.rs")  # exists, is not a manifest
    violations, _ = _scan(tmp_path)
    assert len(violations) == 1 and violations[0][3].startswith("NO SUCH MANIFEST"), violations


def test_doc_two_commands_on_one_line_and_an_unbalanced_quote(tmp_path):
    _crate(tmp_path)
    _doc(tmp_path, "cargo test -p sndl-core && cargo test -p nope")
    violations, ok = _scan(tmp_path)
    assert ok == 1 and len(violations) == 1 and "nope" in violations[0][3], violations
    _doc(tmp_path, 'cargo test -p sndl-core --features "simd')  # shlex fails: plain split
    violations, ok = _scan(tmp_path)
    assert ok == 0 and len(violations) == 1 and violations[0][3].startswith("NO SUCH FEATURE")


def test_doc_native_line_reports_hidden_rust(tmp_path):
    _repo(tmp_path)
    (tmp_path / ".gitignore").write_text("*.rs\n", encoding="utf-8")
    assert dcg._native_line(tmp_path).startswith("native/ UNREADABLE"), dcg._native_line(tmp_path)


def test_doc_cargo_without_any_manifest_is_a_violation(tmp_path):
    _doc(tmp_path, "cargo test")
    violations, _ = _scan(tmp_path)
    assert len(violations) == 1 and violations[0][3].startswith("NO CARGO MANIFEST"), violations


def test_doc_maturin_manifest_resolves(tmp_path):
    _doc(tmp_path, "maturin develop -m native/sndl-python/Cargo.toml")
    violations, _ = _scan(tmp_path)
    assert len(violations) == 1 and violations[0][3].startswith("NO SUCH MANIFEST"), violations
    _crate(tmp_path, "sndl-python")
    violations, ok = _scan(tmp_path)
    assert violations == [] and ok == 1


def test_doc_maturin_without_m_needs_a_root_manifest_and_m_stays_inside_the_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "outside"\n', encoding="utf-8")
    _doc(root, "maturin build")
    violations, _ = _scan(root)
    assert len(violations) == 1 and violations[0][3].startswith("NO Cargo.toml AT THE REPO ROOT")
    _doc(root, "maturin build -m ../Cargo.toml")
    violations, _ = _scan(root)
    assert len(violations) == 1 and violations[0][3].startswith("NO SUCH MANIFEST"), violations
    (root / "Cargo.toml").write_text('[package]\nname = "inside"\n', encoding="utf-8")
    _doc(root, "maturin build")
    assert _scan(root) == ([], 1)


def test_doc_python_m_native_is_the_extension_not_a_missing_module(tmp_path):
    _doc(tmp_path, f"python -m {NATIVE_MODULE}")
    violations, _ = _scan(tmp_path)
    assert len(violations) == 1, violations
    assert violations[0][2] == f"python -m {NATIVE_MODULE}"
    why = violations[0][3]
    assert "NATIVE EXTENSION" in why and "NO SUCH MODULE" not in why, why


def test_doc_real_tree_is_green_and_names_the_native_status():
    violations, ok, used = _seam(dcg.scan)
    assert violations == [], violations
    assert ok > 0 and used == set(dcg.EXEMPT)
    line = dcg._native_line()
    assert line.startswith("native/ PRESENT" if rust_files(ROOT) else "native/ ABSENT"), line


# ---- 3. p4c_ratchet: .rs and Cargo manifests join the corpus with a membership rule --------------


def _repo(tmp_path, crate_name="sndl-core"):
    """The gate's own fixture builder, one construction per red."""
    crate = p4c._fixture_repo(tmp_path, crate_name)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "p4c_ungoverned_baseline.json").write_text(
        '{"total": 0, "zones": {}, "paths": {}}\n', encoding="utf-8"
    )
    return crate


def _census(root):
    out = _seam(p4c.census, root=root)
    return out[0], out[1], (out[3] if len(out) > 3 else "(census reports no native/ status)")


def test_p4c_untracked_rs_outside_the_mod_tree_reds_its_crate_zone(tmp_path):
    _repo(tmp_path)
    zones, excl, native = _census(tmp_path)
    assert zones.get("native/sndl-core", []) == ["native/sndl-core/src/scratch.rs"], zones
    assert p4c.compare({z: len(v) for z, v in zones.items()}, {}) == [("native/sndl-core", 1, 0)]
    assert excl["native/** test code"] == 1 and native.startswith("native/ PRESENT"), (excl, native)


def test_p4c_mod_tree_members_and_test_code_are_not_findings(tmp_path):
    crate = _repo(tmp_path)
    (crate / "src" / "scratch.rs").unlink()
    zones, _, _ = _census(tmp_path)
    # lib + bin roots, #[path], cfg(test) child, build.rs, manifest all reached; the workspace
    # manifest is governed while a tier crate exists
    assert "native/sndl-core" not in zones, zones
    assert "native" not in zones, zones
    assert all((crate / f).is_file() for f in p4c.FIXTURE_GOVERNED)


def test_p4c_native_with_only_a_manifest_is_absent_and_governed(tmp_path):
    crate = _repo(tmp_path)
    for p in list(crate.rglob("*.rs")):
        p.unlink()
    zones, _, native = _census(tmp_path)
    assert native.startswith("native/ ABSENT (0 .rs files)"), native
    assert "native" not in zones and "native/sndl-core" not in zones, zones


def test_p4c_update_baseline_uses_the_given_root(tmp_path):
    _repo(tmp_path)
    real = ROOT / "tools" / "p4c_ungoverned_baseline.json"
    before = real.stat().st_mtime_ns
    assert _seam(p4c.main, ["--update-baseline"], root=tmp_path) == 1  # a rise is refused
    (tmp_path / "tools" / "p4c_ungoverned_baseline.json").unlink()
    assert p4c.main(["--update-baseline"], root=tmp_path) == 0
    assert (tmp_path / "tools" / "p4c_ungoverned_baseline.json").is_file()
    assert real.stat().st_mtime_ns == before, "the real baseline was touched by a fixture run"


def test_p4c_crate_outside_native_tier_reds_on_every_file(tmp_path):
    crate = _repo(tmp_path, "other")
    (crate / "src" / "scratch.rs").unlink()
    zones, _, _ = _census(tmp_path)
    assert zones.get("native/other") == sorted(f"native/other/{f}" for f in p4c.FIXTURE_GOVERNED)
    # no tier crate: the workspace manifest counts too
    assert zones.get("native") == ["native/Cargo.toml"], zones


def test_p4c_membership_follows_each_crates_own_roots(tmp_path):
    """A `#[path]` edge between crates governs nothing: a non-tier root reaching a tier orphan
    leaves it counted, and a tier root reaching into a non-tier crate governs nothing there."""
    crate = _repo(tmp_path)
    other = _crate(tmp_path, "other", lib='#[path = "../../sndl-core/src/scratch.rs"]\nmod s;\n')
    (other / "src" / "orphan.rs").write_text("pub fn o() {}\n", encoding="utf-8")
    lib = crate / "src" / "lib.rs"
    lib.write_text(
        lib.read_text(encoding="utf-8") + '#[path = "../../other/src/orphan.rs"]\nmod foreign;\n',
        encoding="utf-8",
    )
    zones, _, _ = _census(tmp_path)
    assert zones.get("native/sndl-core") == ["native/sndl-core/src/scratch.rs"], zones
    assert zones.get("native/other") == [
        "native/other/Cargo.toml",
        "native/other/src/a.rs",
        "native/other/src/lib.rs",
        "native/other/src/orphan.rs",
    ], zones


def test_p4c_unreadable_rust_fails_closed(tmp_path):
    crate = _repo(tmp_path)
    (crate / "src" / "a.rs").write_text("pub fn a( {\n", encoding="utf-8")
    with pytest.raises(RustFactsUnavailable):
        _census(tmp_path)
    assert _seam(p4c.main, ["--assert"], root=tmp_path) == 1
    assert _seam(p4c.main, [], root=tmp_path) == 1


def test_p4c_rust_hidden_from_git_fails_closed_never_absent(tmp_path):
    _repo(tmp_path)
    (tmp_path / ".gitignore").write_text("*.rs\n", encoding="utf-8")
    with pytest.raises(RustFactsUnavailable):
        _census(tmp_path)
    assert _seam(p4c.main, ["--assert"], root=tmp_path) == 1


def test_p4c_assert_reds_through_main(tmp_path):
    _repo(tmp_path)
    assert _seam(p4c.main, ["--assert"], root=tmp_path) == 1
    (tmp_path / "native" / "sndl-core" / "src" / "scratch.rs").unlink()
    assert p4c.main(["--assert"], root=tmp_path) == 0


def test_p4c_real_tree_counts_unchanged_and_native_status_printed(capsys):
    zones, excl, native = _census(None)
    listed = {p for v in zones.values() for p in v}
    assert "tools/rsfacts/src/main.rs" not in listed and "tools/rsfacts/Cargo.toml" not in listed
    assert excl["tools/rsfacts/**"] == 2, excl
    if rust_files(ROOT):
        assert native.startswith("native/ PRESENT"), native
    else:
        assert native.startswith("native/ ABSENT"), native
    baseline = p4c._load_baseline()
    assert p4c.compare({z: len(v) for z, v in zones.items()}, baseline["zones"]) == []
    assert p4c.main([]) == 0
    assert "native/" in capsys.readouterr().out
