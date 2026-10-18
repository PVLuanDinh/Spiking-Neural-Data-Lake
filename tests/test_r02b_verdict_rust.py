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


def _reject_schema(monkeypatch):
    """The reader refusing its own output (schema mismatch), as rust_facts.facts does."""

    def reject(*_a, **_k):
        raise rust_facts.RustFactsUnavailable("schema 1 != 2")

    monkeypatch.setattr(rust_facts, "facts", reject)


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


# ---- PREREG addenda: receivers, filters, wrapper guards, detection paths, schema rejection ----


@pytest.mark.parametrize("case", RR.RUST_FOLD_CASES, ids=lambda c: c[0])
def test_rr_receiver_and_filter_cases(case):
    _, src, expected = case
    criteria = red_reachability.rust_criteria(RR.fixture_facts(src))
    assert [c["analysis"] for c in criteria] == expected, criteria


def test_rr_test_file_facts_are_not_criteria():
    status, facts = RR.fixture_facts(RR.RUST_RED_SRC)
    facts[0]["test_file"] = True
    assert red_reachability.rust_criteria((status, facts)) == []


def test_rr_wrapper_guards_each_keep_unproven(tmp_path, monkeypatch):
    _rr_tree(tmp_path, monkeypatch, "", RR.PY_WRAPPER_CONTROLS)
    criteria, _ = red_reachability.scan(native=RR.fixture_facts(RR.RUST_RED_SRC))
    py = {c["name"]: c["analysis"] for c in criteria if c["name"].startswith("wrap.")}
    assert py == RR.PY_WRAPPER_EXPECTED, py
    assert len([c for c in criteria if c["no_red"]]) == 2  # the two Rust NO_REDs only


def test_rr_wrapper_with_two_returns_to_one_twin_takes_its_verdict(tmp_path, monkeypatch):
    _rr_tree(tmp_path, monkeypatch, "", RR.PY_WRAPPER_SAME_TWIN)
    criteria, _ = red_reachability.scan(native=RR.fixture_facts(RR.RUST_RED_SRC))
    same = next(c for c in criteria if c["name"] == "wrap.probe_same_twin")
    assert same["analysis"] == "NO_RED" and "plasticity.rs::probe_binding" in same["reason"], same
    assert len([c for c in criteria if c["no_red"]]) == 3


def test_rr_schema_rejection_is_red(tmp_path, monkeypatch, capsys):
    _rr_tree(tmp_path, monkeypatch, RR.RUST_GREEN_SRC, RR.PY_GREEN_SRC)
    _reject_schema(monkeypatch)
    assert red_reachability.main(["--assert"]) == 1
    assert "schema 1 != 2" in capsys.readouterr().out


def test_rr_self_check_runs_both_arms(capsys):
    assert red_reachability.main(["--self-check"]) == 0
    out = capsys.readouterr().out
    assert (
        "self-check OK: Rust RED arm 4 NO_RED, GREEN twin 0, 4 fold cases + wrapper guards exact"
        in out
    )


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


@pytest.mark.parametrize("case", CP.RUST_FOLD_CASES, ids=lambda c: c[0])
def test_cp_each_detection_path_fires_or_stays_quiet(case):
    _, src, expected = case
    found = cp.rust_aggregate_findings(native=CP.fixture_facts(src))
    assert [f.split(" ")[1] for f in found] == expected, found


def test_cp_native_edge_on_the_default_root(tmp_path, monkeypatch, real_rsfacts):
    _cp_tree(tmp_path, monkeypatch, CP.RUST_GREEN_SRC)
    (tmp_path / "src" / "snn_data_lake" / "bridge.py").write_text(
        CP.PY_BRIDGE_SRC, encoding="utf-8"
    )
    found = cp.asserted_findings()
    assert len(found) == 1 and found[0].startswith("bridge.py:5:"), found


def test_cp_schema_rejection_is_an_asserted_finding(tmp_path, monkeypatch):
    _cp_tree(tmp_path, monkeypatch, CP.RUST_GREEN_SRC)
    _reject_schema(monkeypatch)
    found = cp.asserted_findings()
    assert len(found) == 1 and "schema 1 != 2" in found[0], found
    assert cp._assert() == 1


def test_cp_assert_and_report_share_one_native_snapshot(monkeypatch):
    calls = []
    real = cp.native_status

    def counted(root=None):
        calls.append(root)
        return real(root)

    monkeypatch.setattr(cp, "native_status", counted)
    assert cp._assert() == 0
    assert len(calls) == 1, calls
    cp.report(real())
    assert len(calls) == 1, calls
