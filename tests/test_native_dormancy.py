"""R0.2 r02-dorm: Rust half of the P3 dormancy gates (D1 export, D2 module, M1 learning fn).

Every case drives `native_dormancy.audit`, the one function `method_dormancy_gate.main` and the
dormancy gate hook call; the fixture `native/` tree is built by the module's own `write_fixture`, so
the test and the gate self-check share one tree (pre-registered: RED arm = exactly 3 findings,
GREEN twin = 0, unparseable Rust = red).
"""

import pytest

from snn_data_lake.gates import method_dormancy_gate as mdg
from snn_data_lake.gates import native_dormancy as nd


def _run(tmp_path, red, live=None, **kw):
    live = nd.FIXTURE_LIVE if live is None else live
    nd.write_fixture(tmp_path, red=red)
    return nd.audit(root=tmp_path, live=live, src=tmp_path / "src", **kw)


def test_red_arm_fires_exactly_the_three_preregistered_findings(tmp_path):
    res = _run(tmp_path, red=True)
    assert res["status"].startswith("native/ PRESENT")
    assert len(res["d"]) == 2 and len(res["m"]) == 1, res
    assert any(f.startswith("D1") and "lif_step" in f for f in res["d"])
    assert any(f.startswith("D2") and "dendrite.rs" in f for f in res["d"])
    assert res["m"][0].startswith("M1") and "Stdp::potentiate" in res["m"][0]


@pytest.mark.parametrize("name", sorted(nd.FOLD_CASES))
def test_fold_case_red_fires_exactly_and_green_twin_is_clean(tmp_path, name):
    red, green, (nd_d, nd_m) = nd.FOLD_CASES[name]
    nd._write(tmp_path / "red", red)
    res = nd.audit(tmp_path / "red", live=nd.FIXTURE_LIVE, src=tmp_path / "red" / "src")
    assert (len(res["d"]), len(res["m"])) == (nd_d, nd_m), res
    nd._write(tmp_path / "green", green)
    res = nd.audit(tmp_path / "green", live=nd.FIXTURE_LIVE, src=tmp_path / "green" / "src")
    assert res["d"] == [] and res["m"] == [], res


def test_crates_nested_below_native_are_still_governed(tmp_path):
    nd.write_fixture(tmp_path, red=True, prefix="crates/")
    res = nd.audit(root=tmp_path, live=nd.FIXTURE_LIVE, src=tmp_path / "src")
    assert len(res["d"]) == 2 and len(res["m"]) == 1, res
    green = tmp_path / "green"
    nd.write_fixture(green, red=False, prefix="crates/")
    res = nd.audit(root=green, live=nd.FIXTURE_LIVE, src=green / "src")
    assert res["d"] == [] and res["m"] == [], res


def test_green_twin_is_clean(tmp_path):
    res = _run(tmp_path, red=False)
    assert res["d"] == [] and res["m"] == [], res
    assert any("nudge" in r for r in res["report"]), "pub-item report line missing"


def test_an_empty_live_set_reds_the_export_python_does_reference(tmp_path):
    res = _run(tmp_path, red=False, live=set())
    assert any(f.startswith("D1") and "decay" in f for f in res["d"]), res


def test_unparseable_rust_fails_closed_on_both_gates(tmp_path):
    nd.write_fixture(tmp_path, red=False)
    (tmp_path / "native/sndl-core/src/reservoir.rs").write_text("fn broken( {", encoding="utf-8")
    res = nd.audit(root=tmp_path, live=nd.FIXTURE_LIVE, src=tmp_path / "src")
    assert res["d"] and res["m"] and "UNAVAILABLE" in res["d"][0] + res["m"][0], res


def test_an_orphan_rs_file_is_not_in_any_mod_tree(tmp_path):
    nd.write_fixture(tmp_path, red=False)
    (tmp_path / "native/sndl-core/src/orphan.rs").write_text("pub fn o() {}\n", encoding="utf-8")
    res = nd.audit(root=tmp_path, live=nd.FIXTURE_LIVE, src=tmp_path / "src")
    assert [f for f in res["d"] if f.startswith("D2a") and "orphan.rs" in f], res


def test_stale_and_phantom_native_exempt_rows_red(tmp_path):
    res = _run(
        tmp_path,
        red=False,
        exempt={"lif_step": "live now", "no_such_export": "typo"},
        exempt_method={"Stdp::potentiate": "has a caller now"},
    )
    assert any("STALE" in f and "lif_step" in f for f in res["d"]), res
    assert any("PHANTOM" in f and "no_such_export" in f for f in res["d"]), res
    assert any("STALE" in f and "Stdp::potentiate" in f for f in res["m"]), res


def test_an_exempt_row_silences_exactly_its_finding(tmp_path):
    res = _run(
        tmp_path,
        red=True,
        exempt={"lif_step": "r", "sndl-core/src/dendrite.rs": "r"},
        exempt_method={"Stdp::potentiate": "r"},
    )
    assert res["d"] == [] and res["m"] == [], res


def test_real_tree_is_clean():
    """native/ exists since R0.4 (sndl-core); the real tree is clean via its NATIVE_EXEMPT rows."""
    res = nd.audit()
    assert res["status"].startswith(("native/ ABSENT", "native/ PRESENT")), res["status"]
    assert res["d"] == [] and res["m"] == [], res


def test_the_method_gate_reds_on_an_m1_finding(monkeypatch):
    monkeypatch.setattr(
        nd, "audit", lambda *a, **k: {"status": "x", "d": [], "m": ["M1 planted"], "report": []}
    )
    with pytest.raises(AssertionError, match="M1 planted"):
        nd.check("m")


def test_the_module_self_check_passes():
    nd.selfcheck()


def test_learn_verb_matcher_is_the_method_gates():
    assert mdg.is_learn_verb("potentiate") and not mdg.is_learn_verb("lif_step")
    assert not mdg.is_learn_verb("__fit__")


def test_an_exemption_never_hides_an_orphan_file(tmp_path):
    """Codex B1 integrator 1: an exempt row justifies D2b (no binding yet), never a D2a orphan."""
    import subprocess

    files = {
        "native/c/Cargo.toml": '[package]\nname = "c"\n',
        "native/c/src/lib.rs": "pub mod kept;\n",
        "native/c/src/kept.rs": "pub fn k() {}\n",
        "native/c/src/orphan.rs": "pub fn o() {}\n",
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    res = nd.audit(
        tmp_path,
        live=set(),
        src=tmp_path / "src",
        exempt={"c/src/orphan.rs": "x", "c/src/kept.rs": "x"},
    )
    assert [d for d in res["d"] if d.startswith("D2a")] == [
        "D2a c/src/orphan.rs: not in the mod tree of crate c (orphan file)"
    ], res
    assert not [d for d in res["d"] if d.startswith("D2b")], res


def test_a_facade_reexport_of_an_unused_learn_fn_is_not_a_use(tmp_path):
    """#3595: `from ._native import learn` plus `__all__` in a live facade nothing calls through is
    an import, not a use -- M1 must fire, and D1 (import counts there) stays quiet."""
    facade = 'from snn_data_lake._native import learn\n\n__all__ = ["learn"]\n'
    nd._write(
        tmp_path,
        {"native/sndl-python/src/lib.rs": nd._LEARN_LIB, "src/snn_data_lake/facade.py": facade},
    )
    res = nd.audit(tmp_path, live={"snn_data_lake.facade"}, src=tmp_path / "src")
    assert res["d"] == [], res
    assert len(res["m"]) == 1 and "learn" in res["m"][0], res
