"""The Rust halves of red_reachability (P8, row 20) and conclusion_protocol (C5, row 12).

Every test drives the gate's ONE Rust entry point through its real caller (`scan`/`main` and
`asserted_findings`/`_assert`) on a constructed `native/` tree under a temp git repo, with the
rsfacts build pinned to the real checkout; the injection tests call the same function with
`native=` fixture facts, the way the self-checks do. Logic is never forked.
"""

import pathlib
import sys

import pytest

from snn_data_lake.gates import conclusion_protocol as cp
from snn_data_lake.gates import native_dormancy, rust_facts

_TOOLS = str(pathlib.Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

red_reachability = pytest.importorskip("red_reachability")

CORE = "native/sndl-core/src/plasticity.rs"
BIND = "native/sndl-python/src/lib.rs"
PY_BIND = "#[pyfunction]\nfn add(a: f64, b: f64) -> f64 { a + b }\n"
RR, CP = red_reachability, cp  # the fixtures live beside the rules they fire (RUST_RED_SRC & co.)


@pytest.fixture
def real_rsfacts(monkeypatch):
    """Pin the rsfacts build to this checkout while the gate's root points at a fixture repo."""
    build = rust_facts.binary
    monkeypatch.setattr(rust_facts, "binary", lambda root=None: build(rust_facts.ROOT))


def _rr_tree(tmp_path, monkeypatch, rs, py):
    native_dormancy._write(
        tmp_path, {CORE: rs, BIND: PY_BIND, "src/pkg/__init__.py": "", "src/pkg/wrap.py": py}
    )
    monkeypatch.setattr(red_reachability, "ROOT", tmp_path)
    monkeypatch.setattr(red_reachability, "SRC", tmp_path / "src")


def _cp_tree(tmp_path, monkeypatch, rs):
    native_dormancy._write(tmp_path, {CORE: rs, BIND: PY_BIND, "src/snn_data_lake/__init__.py": ""})
    monkeypatch.setattr(cp, "REPO", tmp_path)


def _no_red(criteria):
    return {c["name"] for c in criteria if c["no_red"]}


def test_rr_red_arm_has_exactly_the_preregistered_no_reds(
    tmp_path, monkeypatch, capsys, real_rsfacts
):
    _rr_tree(tmp_path, monkeypatch, RR.RUST_RED_SRC, RR.PY_RED_SRC)
    criteria, _ = red_reachability.scan()
    assert _no_red(criteria) == RR.RED_NO_RED, sorted(_no_red(criteria))
    rust = [c for c in criteria if c["file"].startswith("native/")]
    assert len(rust) == 4, [c["name"] for c in rust]
    assert sorted(c["name"] for c in criteria if c["analysis"] == "UNPROVEN") == [
        f"{CORE}::probe_env",
        f"{CORE}::probe_measured",
        "wrap.probe_measured",
    ]
    assert red_reachability.main(["--assert"]) == 1
    out = capsys.readouterr().out
    assert "native/ PRESENT" in out and "takes the Rust verdict" in out


def test_rr_green_twin_has_zero_no_reds(tmp_path, monkeypatch, capsys, real_rsfacts):
    _rr_tree(tmp_path, monkeypatch, RR.RUST_GREEN_SRC, RR.PY_GREEN_SRC)
    criteria, _ = red_reachability.scan()
    assert _no_red(criteria) == set()
    assert len([c for c in criteria if c["file"].startswith("native/")]) == 2
    assert len([c for c in criteria if c["analysis"] == "UNPROVEN"]) == 3
    assert red_reachability.main(["--assert"]) == 0
    assert "native/ PRESENT" in capsys.readouterr().out


def test_rr_rust_facts_unavailable_with_rs_present_is_red(tmp_path, monkeypatch, capsys):
    _rr_tree(tmp_path, monkeypatch, RR.RUST_GREEN_SRC, RR.PY_GREEN_SRC)
    monkeypatch.setattr(rust_facts.shutil, "which", lambda _name: None)
    assert red_reachability.main(["--assert"]) == 1
    assert "RustFactsUnavailable" in capsys.readouterr().out


def test_rr_real_tree_prints_absent_and_python_counts_hold(capsys):
    assert red_reachability.main(["--assert"]) == 0
    out = capsys.readouterr().out
    assert "native/ ABSENT (0 .rs files)" in out
    assert "NO_RED: 0 " in out
    assert rust_facts.BLIND_SPOTS[0] in out


def test_rr_injection_reaches_the_same_entry_point(tmp_path, monkeypatch):
    _rr_tree(tmp_path, monkeypatch, "", RR.PY_RED_SRC)
    criteria, _ = red_reachability.scan(native=RR.fixture_facts(RR.RUST_RED_SRC))
    assert _no_red(criteria) == RR.RED_NO_RED


# ---- review fold 1 (PREREG addendum): receivers, filters, wrapper guards, schema rejection ----

RR_FOLD1 = [
    (
        "receiver-field",
        "pub struct B { ok: bool }\nimpl B { pub fn passes(&self) -> bool { self.ok } }\n",
        ["UNPROVEN"],
    ),
    (
        "mut-receiver-literal",
        "pub struct B;\nimpl B { pub fn probe_x(&mut self) -> bool { true } }\n",
        ["UNPROVEN"],
    ),
    ("bodiless-trait", "pub trait T { fn probe_t(&self) -> bool; }\n", []),
    ("unit-return", "pub fn probe_u() { }\n", []),
]


@pytest.mark.parametrize("case", RR_FOLD1, ids=lambda c: c[0])
def test_rr_fold1_receiver_and_filter_cases(case):
    _, src, expected = case
    criteria = red_reachability.rust_criteria(RR.fixture_facts(src))
    assert [c["analysis"] for c in criteria] == expected, criteria


def test_rr_fold1_test_file_facts_are_not_criteria():
    status, facts = RR.fixture_facts(RR.RUST_RED_SRC)
    facts[0]["test_file"] = True
    assert red_reachability.rust_criteria((status, facts)) == []


RR_WRAPPER_CONTROLS = """from snn_data_lake import _native
import snn_data_lake._native as n


def probe_mixed(x):
    if x:
        return _native.probe_binding()
    return True, "literal"


def probe_two_twins(x):
    if x:
        return _native.probe_binding()
    return n.reentry_present()


def probe_refusal():
    assert _native.probe_binding()
    return _native.probe_binding()


def probe_fallthrough(x):
    if x:
        return _native.probe_binding()
"""


def test_rr_fold1_wrapper_guards_each_keep_unproven(tmp_path, monkeypatch):
    _rr_tree(tmp_path, monkeypatch, "", RR_WRAPPER_CONTROLS)
    criteria, _ = red_reachability.scan(native=RR.fixture_facts(RR.RUST_RED_SRC))
    py = {c["name"]: c["analysis"] for c in criteria if c["name"].startswith("wrap.")}
    assert py == {
        "wrap.probe_mixed": "UNPROVEN",
        "wrap.probe_two_twins": "UNPROVEN",
        "wrap.probe_refusal": "UNPROVEN",
        "wrap.probe_fallthrough": "UNPROVEN",
    }, py
    assert len([c for c in criteria if c["no_red"]]) == 2  # the two Rust NO_REDs only


def test_rr_fold1_schema_rejection_is_red(tmp_path, monkeypatch, capsys):
    _rr_tree(tmp_path, monkeypatch, RR.RUST_GREEN_SRC, RR.PY_GREEN_SRC)

    def reject(*_a, **_k):
        raise rust_facts.RustFactsUnavailable("schema 1 != 2")

    monkeypatch.setattr(rust_facts, "facts", reject)
    assert red_reachability.main(["--assert"]) == 1
    assert "schema 1 != 2" in capsys.readouterr().out


def test_rr_self_check_runs_both_arms(capsys):
    assert red_reachability.main(["--self-check"]) == 0
    assert "self-check OK: Rust RED arm 4 NO_RED, GREEN twin 0" in capsys.readouterr().out


def test_cp_red_arm_has_exactly_the_three_preregistered_findings(
    tmp_path, monkeypatch, real_rsfacts
):
    _cp_tree(tmp_path, monkeypatch, CP.RUST_RED_SRC)
    found = cp.asserted_findings()
    assert len(found) == 3, found
    assert [f.split(":")[1] for f in found] == ["1", "2", "3"], found
    assert any("COMBINE-FN" in f and "score" in f for f in found), found
    assert any("COMBINE-CALL" in f and "sum" in f for f in found), found
    assert any("COMBINE-EXPR" in f for f in found), found
    assert cp._assert() == 1


def test_cp_green_twin_is_zero(tmp_path, monkeypatch, real_rsfacts):
    _cp_tree(tmp_path, monkeypatch, CP.RUST_GREEN_SRC)
    assert cp.asserted_findings() == []
    assert cp._assert() == 0


def test_cp_native_call_naming_two_instruments_is_a_finding(tmp_path):
    (tmp_path / "bridge.py").write_text(CP.PY_BRIDGE_SRC, encoding="utf-8")
    found = cp.aggregate_findings(tmp_path)
    assert len(found) == 1 and found[0].startswith("bridge.py:5:"), found


def test_cp_rust_facts_unavailable_with_rs_present_is_red(tmp_path, monkeypatch):
    _cp_tree(tmp_path, monkeypatch, CP.RUST_GREEN_SRC)
    monkeypatch.setattr(rust_facts.shutil, "which", lambda _name: None)
    found = cp.asserted_findings()
    assert len(found) == 1 and "RustFactsUnavailable" in found[0], found


def test_cp_real_tree_prints_absent_and_asserts_nothing():
    assert cp.asserted_findings() == []
    text = cp.report()
    assert "native/ ABSENT (0 .rs files)" in text
    assert rust_facts.BLIND_SPOTS[0] in text


def test_cp_injection_reaches_the_same_entry_point():
    assert len(cp.rust_aggregate_findings(native=CP.fixture_facts(CP.RUST_RED_SRC))) == 3
    assert cp.rust_aggregate_findings(native=CP.fixture_facts(CP.RUST_GREEN_SRC)) == []


CP_FOLD1 = [
    (
        "compound-assign",
        "pub fn score(mut integration: f64, agency: f64) -> f64 "
        "{ integration += agency; integration }\n",
        ["COMBINE-EXPR"],
    ),
    (
        "snake-case",
        "pub fn score(integration_reading: f64, agency_reading: f64) -> f64 "
        "{ (integration_reading + agency_reading) / 2.0 }\n",
        ["COMBINE-FN"],
    ),
    (
        "free-mean-call",
        "pub fn m(integration: f64, agency: f64) -> f64 { stats::mean(&[integration, agency]) }\n",
        ["COMBINE-CALL"],
    ),
    (
        "fold",
        "pub fn f(integration: f64, agency: f64) -> f64 "
        "{ [integration, agency].iter().fold(0.0, |a, b| a + b) }\n",
        ["COMBINE-CALL"],
    ),
    (
        "index",
        "pub fn g(r: &[f64], integration: usize, agency: usize) -> f64 "
        "{ r[integration + agency] }\n",
        ["COMBINE-EXPR"],
    ),
    (
        "macro-tokens",
        "pub fn h(integration: f64, agency: f64) -> f64 { dbg!(integration + agency) }\n",
        ["COMBINE-EXPR"],
    ),
    (
        "mcall-receiver",
        "pub fn k(integration: f64, agency: f64) -> f64 { (integration + agency).sqrt() }\n",
        ["COMBINE-EXPR"],
    ),
    (
        "control-deref",
        "pub fn c(integration: &f64, agency: &f64) -> Vec<f64> { vec![*integration, *agency] }\n",
        [],
    ),
    (
        "control-plain-call",
        "pub fn c2(integration: f64, agency: f64) -> f64 { pick(integration, agency) }\n",
        [],
    ),
]


@pytest.mark.parametrize("case", CP_FOLD1, ids=lambda c: c[0])
def test_cp_fold1_each_detection_path_fires_or_stays_quiet(case):
    _, src, expected = case
    found = cp.rust_aggregate_findings(native=CP.fixture_facts(src))
    assert [f.split(" ")[1] for f in found] == expected, found


def test_cp_fold1_native_edge_on_the_default_root(tmp_path, monkeypatch, real_rsfacts):
    _cp_tree(tmp_path, monkeypatch, CP.RUST_GREEN_SRC)
    (tmp_path / "src" / "snn_data_lake" / "bridge.py").write_text(
        CP.PY_BRIDGE_SRC, encoding="utf-8"
    )
    found = cp.asserted_findings()
    assert len(found) == 1 and found[0].startswith("bridge.py:5:"), found


def test_cp_fold1_schema_rejection_is_an_asserted_finding(tmp_path, monkeypatch):
    _cp_tree(tmp_path, monkeypatch, CP.RUST_GREEN_SRC)

    def reject(*_a, **_k):
        raise rust_facts.RustFactsUnavailable("schema 1 != 2")

    monkeypatch.setattr(rust_facts, "facts", reject)
    found = cp.asserted_findings()
    assert len(found) == 1 and "schema 1 != 2" in found[0], found
    assert cp._assert() == 1
