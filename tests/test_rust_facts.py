"""R0.2: the shared Rust reader's positive control and its fail-closed red, run under pytest."""

from snn_data_lake.gates import rust_facts


def test_rust_facts_self_check_fires_its_red_and_maps_both_pyo3_styles():
    rust_facts.self_check()


def test_native_status_is_reported_absent_not_as_zero_findings(tmp_path):
    status, facts = rust_facts.native_status(tmp_path)
    assert status.startswith("native/ ABSENT") and facts == []
