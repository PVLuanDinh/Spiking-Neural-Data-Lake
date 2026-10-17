"""R0.2: the shared Rust reader's positive control and its fail-closed red, run under pytest."""

from snn_data_lake.gates import rust_facts


def test_rust_facts_self_check_fires_its_red_and_maps_both_pyo3_styles():
    rust_facts.self_check()


def test_native_status_is_reported_absent_not_as_zero_findings(tmp_path):
    status, facts = rust_facts.native_status(tmp_path)
    assert status.startswith("native/ ABSENT") and facts == []


def _git_tree(root, files):
    import subprocess

    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)


def test_test_only_item_type_paths_are_not_production_paths():
    """Codex r3 F-1: a #[cfg(test)] const / struct must not leak its type paths into `paths`."""
    f = rust_facts.facts_for_source(
        "#[cfg(test)]\nconst UNUSED: Option<sndl_core::reservoir::Cell> = None;\n"
        "#[cfg(test)]\nstruct Probe { c: sndl_core::sleep::Pass }\n"
    )
    assert f["paths"] == [], f["paths"]
    assert {"sndl_core::reservoir::Cell", "sndl_core::sleep::Pass"} <= set(f["test_paths"])


def test_a_rebound_wrapper_does_not_retarget_an_earlier_registration():
    """Codex r3 F-2: `let f = wrap(learn); add(f); let f = wrap(decay);` registers learn only."""
    src = (
        "#[pyfunction]\nfn learn() {}\n#[pyfunction]\nfn decay() {}\n#[pymodule]\n"
        "fn m(m: &Bound<'_, PyModule>) -> PyResult<()> {\n let f = wrap_pyfunction!(learn, m)?;\n"
        " m.add_function(f)?;\n let f = wrap_pyfunction!(decay, m)?;\n Ok(())\n}\n"
    )
    assert sorted(rust_facts.pyo3_exports([rust_facts.facts_for_source(src)])) == ["learn"]


def test_a_crate_tests_dir_file_is_test_code_without_cfg(tmp_path):
    """Codex r3 F-5: native/<crate>/tests/*.rs is test code even with no #[cfg(test)]."""
    _git_tree(
        tmp_path,
        {
            "native/sndl-core/src/lib.rs": "pub fn f() {}\n",
            "native/sndl-core/tests/reference.rs": (
                "fn reference() { sndl_core::f(); let d = 1; }\n"
            ),
        },
    )
    files = rust_facts.rust_files(tmp_path)
    by = {p.name: f for p, f in zip(files, rust_facts.facts(files), strict=True)}
    ref = by["reference.rs"]
    assert ref["test_file"] is True and ref["paths"] == [] and "sndl_core::f" in ref["test_paths"]
    assert all(r["cfg_test"] for k in ("fns", "calls", "lets") for r in ref[k]), ref
    assert by["lib.rs"]["test_file"] is False


def test_a_let_is_not_visible_inside_its_own_initializer():
    """Codex r4 N-1: `let f = wrap(learn); let f = m.add_function(f)?;` registers learn."""
    src = (
        "#[pyfunction]\nfn learn() {}\n#[pymodule]\n"
        "fn m(m: &Bound<'_, PyModule>) -> PyResult<()> {\n"
        " let f = wrap_pyfunction!(learn, m)?;\n let f = m.add_function(f)?;\n Ok(())\n}\n"
    )
    assert sorted(rust_facts.pyo3_exports([rust_facts.facts_for_source(src)])) == ["learn"]


def test_a_tests_dir_file_included_by_a_production_path_mod_is_production(tmp_path):
    """Codex r4 N-2: `#[path = "../tests/reference.rs"] mod reference;` makes it production code."""
    _git_tree(
        tmp_path,
        {
            "native/sndl-core/src/lib.rs": '#[path = "../tests/reference.rs"]\nmod reference;\n',
            "native/sndl-core/tests/reference.rs": "pub fn reference() {}\n",
            "native/sndl-core/tests/t.rs": "fn t() {}\n",
        },
    )
    files = rust_facts.rust_files(tmp_path)
    flags = {p.name: f["test_file"] for p, f in zip(files, rust_facts.facts(files), strict=True)}
    assert flags == {"lib.rs": False, "reference.rs": False, "t.rs": True}, flags


def test_a_crate_nested_under_tests_is_classified_by_its_own_root(tmp_path):
    """Codex r4 N-3: the DEEPEST crate root decides; a crate under tests/ stays production."""
    _git_tree(
        tmp_path,
        {
            "native/outer/Cargo.toml": '[package]\nname = "outer"\n',
            "native/outer/src/lib.rs": "pub fn o() {}\n",
            "native/outer/tests/t.rs": "fn t() {}\n",
            "native/outer/tests/inner/Cargo.toml": '[package]\nname = "inner"\n',
            "native/outer/tests/inner/src/lib.rs": "pub fn i() {}\n",
        },
    )
    files = rust_facts.rust_files(tmp_path)
    flags = {p.relative_to(tmp_path).as_posix(): f["test_file"]
             for p, f in zip(files, rust_facts.facts(files), strict=True)}  # fmt: skip
    assert flags == {
        "native/outer/src/lib.rs": False,
        "native/outer/tests/t.rs": True,
        "native/outer/tests/inner/src/lib.rs": False,
    }, flags
