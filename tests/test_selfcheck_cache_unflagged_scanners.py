"""selfcheck_cache v3: a tree-walker whose listing root enters as a parameter (or through
`git ls-files`) must be keyed to the tree surface (lane carryfix, Opus B3 review M1).

Pre-registered reds (rows/carryfix-REPORT.md): R1-R3 FAIL on 682c54fe's selfcheck_cache.py --
the impact_analysis override left replication_guard / native_ratchet unchanged, the lib.rs
override left rust_facts unchanged -- and pass after the fail-closed derivation. C1 / C2 are the
key-invariance controls that must hold on both sides. The subjects are the module's own red (n)
constants; like red (i) / (n), the real-tree reds need git (without it every scanner key is a
per-instance nonce and the flip would be vacuous). Every cache here uses a tmp store.

B4 (Opus closing review of the carryfix): R4 / R5 FAIL on the root resolver (675fd358) -- the
review's constructed shapes S2-S6, S8 are no scan there and a tree change leaves the key -- and
pass once every listing scans."""

from __future__ import annotations

import os

import pytest

from snn_data_lake.gates import selfcheck_cache as sc

ROOT = sc._repo_root()
IMPACT = os.path.join(ROOT, *sc._IMPACT_FILE)
LIB_RS = os.path.join(ROOT, *sc._LIB_RS_FILE)
ENV = {"NORD_GPU": "1"}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return str(tmp_path_factory.mktemp("carryfix") / "store.json")


@pytest.fixture(scope="module")
def base(store):
    cache = sc.SelfcheckCache(store_path=store, env=ENV)
    # list the tree surface DIRECTLY (B4 C1): reaching it through closure_key(_WALKERS[0]) lists
    # it only if that walker is ALREADY a scanner, so on the defective logic every red here
    # skipped with this message's false reason instead of failing
    cache._tree_hash()
    if cache.tree_surface == 0:
        pytest.skip("no git: the tree surface is a nonce, the real-tree reds are vacuous")
    return cache


@pytest.fixture(scope="module")
def probe(base):
    _desc, _plain, probe = sc.plain_probe(base)
    assert probe is not None, "no plain module with a multi-file closure"
    return probe


def _fork(store, overrides):
    return sc.SelfcheckCache(store_path=store, env=ENV, file_hash_overrides=overrides)


@pytest.mark.parametrize("mod", sc._WALKERS[:2])
def test_r1_r2_parameter_rooted_walkers_are_keyed_to_the_tree(base, store, mod):
    assert os.path.isfile(IMPACT)
    assert base.describe(mod)["scans"], f"{mod} walks the package yet is not a scanner"
    fork = _fork(store, {IMPACT: "0" * 64})
    assert fork.closure_key(mod) != base.closure_key(mod), (
        f"a change to impact_analysis.py did not flip {mod} (stale PASS would carry)"
    )


@pytest.mark.parametrize("mod", [sc._WALKERS[2], sc._RUST_FACTS_READER])
def test_r3_rust_surface_flips_rust_facts_readers(base, store, mod):
    assert os.path.isfile(LIB_RS)
    fork = _fork(store, {LIB_RS: "0" * 64})
    assert fork.closure_key(mod) != base.closure_key(mod), f"lib.rs did not flip {mod}"


def test_rust_facts_readers_carry_the_native_and_rsfacts_rows(base, store, probe, tmp_path):
    assert base.describe("gates.native_ratchet")["native"], "a rust_facts reader keys native:"
    rsfacts = tmp_path / "rsfacts"
    (rsfacts / "target").mkdir(parents=True)
    fixture = rsfacts / "fixture.rs"
    fixture.write_text("pub fn f() {}\n", encoding="utf-8")
    built = rsfacts / "target" / "built.bin"
    built.write_bytes(b"\0")
    kw = dict(store_path=store, env=ENV, rsfacts_root=str(rsfacts))
    plain = sc.SelfcheckCache(**kw)
    k = plain.closure_key("gates.native_ratchet")
    flipped = sc.SelfcheckCache(**kw, file_hash_overrides={str(fixture): "0" * 64})
    assert flipped.closure_key("gates.native_ratchet") != k, "tools/rsfacts/** is not keyed"
    target = sc.SelfcheckCache(**kw, file_hash_overrides={str(built): "0" * 64})
    assert target.closure_key("gates.native_ratchet") == k, "rsfacts target/ must not key"
    assert flipped.closure_key(probe) == plain.closure_key(probe)


def test_c1_plain_module_unchanged_by_the_overrides(base, store, probe):
    assert _fork(store, {IMPACT: "0" * 64}).closure_key(probe) == base.closure_key(probe)
    assert _fork(store, {LIB_RS: "0" * 64}).closure_key(probe) == base.closure_key(probe)


# B4 C2: the Opus closing review's constructed stale shapes (MAJOR 2) that its root resolver
# passed. Inline, NOT read from sc._SYNTHETIC: against the pre-fix module that table holds none
# of them, so a table-driven red could never fire there.
BYPASS = {
    "S2": "import os\nos.environ.setdefault('ZZZ_S2', os.path.dirname(__file__))\n"
    "os.listdir(os.environ['ZZZ_S2'])\n",
    "S3": "import os\nfor r, d, f in os.walk(os.environ['PYTHONPATH']):\n    pass\n",
    "S4": "import os\nfrom snn_data_lake.substrate import home\n"
    "home.home = lambda: os.path.dirname(__file__)\nos.listdir(home.home())\n",
    "S5": "import os\nfrom snn_data_lake.substrate.home import home\n"
    "globals()['home'] = os.getcwd\nos.listdir(home())\n",
    "S6": "import os, shutil\nfrom snn_data_lake.substrate.home import home\n"
    "snap = os.path.join(home(), 'snap')\nshutil.copytree(os.path.dirname(__file__), snap)\n"
    "os.walk(snap)\n",
    "S8": "import os\nfrom snn_data_lake.substrate.home import home\n"
    "os.listdir(home() + '/.' + './x')\n",
}


@pytest.mark.parametrize("shape", sorted(BYPASS))
def test_r4_bypass_shape_is_a_scan(shape):
    assert sc._facts(BYPASS[shape], 1, sc._NULL_RE)["scans"] is True, f"{shape} is not a scan"


@pytest.mark.parametrize("shape", sorted(BYPASS))
def test_r5_bypass_shape_on_the_probe_flips_on_a_tree_change(base, store, probe, shape):
    """The review's harness: the shape appended to the plain probe's source, then a file no
    closure names (impact_analysis.py) overridden -- the probe's key must flip. Every fork
    carries the shape; a second fork without the override is the quiet-instrument control."""
    path = os.path.normpath(sc._module_file(sc._PKG_PREFIX + probe))
    text = base._read_text(path) + "\n" + BYPASS[shape]

    def fork(over=None):
        return sc.SelfcheckCache(
            store_path=store, env=ENV, file_text_overrides={path: text}, file_hash_overrides=over
        )

    quiet = fork().closure_key(probe)
    assert fork().closure_key(probe) == quiet, "the instrument moved with no input change"
    moved = fork({IMPACT: "0" * 64}).closure_key(probe)
    assert moved != quiet, f"{shape} on {probe}: a tree change left the key (stale PASS carries)"


# B4 C3: any git argv is a scan (review S7). Inline for the same reason as BYPASS.
GIT_ARGV = {
    "S7-diff": "import subprocess\nout = subprocess.run(['git', 'diff', '--name-only'], "
    "capture_output=True).stdout\n",
    "S7-status": "import subprocess\n"
    "out = subprocess.check_output('git status --short', shell=True)\n",
}


@pytest.mark.parametrize("shape", sorted(GIT_ARGV))
def test_r6_git_argv_is_a_scan(shape):
    assert sc._facts(GIT_ARGV[shape], 1, sc._NULL_RE)["scans"] is True, f"{shape} is not a scan"


@pytest.mark.parametrize("shape", sorted(GIT_ARGV))
def test_r7_git_argv_on_the_probe_flips_on_a_tree_change(base, store, probe, shape):
    path = os.path.normpath(sc._module_file(sc._PKG_PREFIX + probe))
    text = base._read_text(path) + "\n" + GIT_ARGV[shape]

    def fork(over=None):
        return sc.SelfcheckCache(
            store_path=store, env=ENV, file_text_overrides={path: text}, file_hash_overrides=over
        )

    quiet = fork().closure_key(probe)
    assert fork().closure_key(probe) == quiet, "the instrument moved with no input change"
    moved = fork({IMPACT: "0" * 64}).closure_key(probe)
    assert moved != quiet, f"{shape} on {probe}: a tree change left the key (stale PASS carries)"


# B4 C4: instruments.eligible_circuit_census reads src files by LITERAL relpath (its DRIVERS and
# every RECORDS citation) under a 1-file import closure. Built from parts, never one literal.
# capability_bar's surface check reads infra/cockpit.py the same way (TRACKED_SURFACES): the
# second live stale carry the rule closes.
READS = {
    "instruments.eligible_circuit_census": ("engine", "brain.py"),
    "gates.capability_bar": ("infra", "cockpit.py"),
}


@pytest.mark.parametrize("reader", sorted(READS))
def test_r8_literal_relpath_read_flips_its_reader(base, store, probe, reader):
    path = os.path.join(ROOT, "src", "snn_data_lake", *READS[reader])
    dotted = sc._PKG_PREFIX + ".".join(READS[reader])[: -len(".py")]
    assert os.path.isfile(path)
    assert dotted not in base._closure(reader), "the red needs a NON-import read"
    assert not base.describe(reader)["scans"], f"{reader} is a scanner: the red would be vacuous"
    fork = _fork(store, {path: "0" * 64})
    assert fork.closure_key(reader) != base.closure_key(reader), (
        f"a {READS[reader][1]} change did not flip {reader} (stale PASS carries)"
    )
    if dotted not in base._closure(probe):
        assert fork.closure_key(probe) == base.closure_key(probe), "the read moved the plain probe"


def test_r9_literal_relpath_derivation():
    src = (
        "A = 'src/snn_data_lake/engine/brain.py'\nB = 'docs\\\\x.md'\n"
        "C = ('see src/x.py', '../escape/x.py', 'no_slash.py', 'a/../b.py')\n"
    )
    paths = sc._facts(src, 1, sc._NULL_RE)["paths"]
    assert paths == frozenset({"src/snn_data_lake/engine/brain.py", "docs/x.md"}), paths


@pytest.mark.xfail(
    strict=True,
    reason="DISCLOSED blind spot (selfcheck_cache docstring 6, review S1): string dispatch hides "
    "the lister. If this passes, the derivation learned it: update the disclosure, drop the xfail",
)
def test_s1_string_dispatch_stays_a_disclosed_blind_spot():
    src = "import os\nfor r, d, f in getattr(os, 'walk')(os.path.dirname(__file__)):\n    pass\n"
    assert sc._facts(src, 1, sc._NULL_RE)["scans"] is True


@pytest.mark.parametrize("name", sorted(sc._SYNTHETIC))
def test_c2_derivation_pins(name):
    """Every synthetic pin the self-check holds, from the one source table: every scan shape,
    including the review's fail-open ones (C2 deleted the home() / env-dir exclusions)."""
    src, field, want = sc._SYNTHETIC[name]
    assert sc._facts(src, 1, sc._NULL_RE)[field] == want
