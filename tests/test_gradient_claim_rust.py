"""The Rust half of the gradient-claim census (R0.2): burn `train_step` must owe a declaration.

Every test drives `gradient_claim_gate.rust_census`, the ONE function `census()` (so `main()`,
`--assert` and `--self-check`) calls for Rust, with fixture facts injected; logic is never forked.
"""

import importlib

import pytest

from snn_data_lake.gates import rust_facts

gate = importlib.import_module("snn_data_lake.gates.gradient_claim_gate")

BASELINE = {
    "counts": {gate.MISSING_DECLARATION: 0, gate.UNDETERMINABLE: 0},
    "rules_seen_floor": 0,
}


def _findings(rules, failures):
    return gate.assert_findings(
        gate.counts(rules), gate.counts(rules, False), 1, BASELINE, failures
    )


def test_red_arm_has_exactly_the_two_preregistered_findings():
    rules, failures, n_rs = gate.rust_census(gate.fixture_native(gate.RUST_RED_SRC))
    assert n_rs == 2
    by_state = {}
    for r in rules:
        by_state.setdefault(r["state"], []).append(r["qualname"])
    assert by_state[gate.MISSING_DECLARATION] == [
        "native/sndl-core/src/plasticity.rs::Trainer::train_step"
    ]
    assert by_state[gate.PHANTOM_DECLARATION] == [
        "native/sndl-core/src/plasticity.rs::Ghost::learn"
    ]
    assert not failures
    assert all(r["governed"] for r in rules)
    found = _findings(rules, failures)
    assert len(found) == 2, found
    assert any(gate.MISSING_DECLARATION in f for f in found)
    assert any(gate.PHANTOM_DECLARATION in f for f in found)


def test_green_twin_has_zero_findings_and_classifies_the_rules():
    rules, failures, _ = gate.rust_census(gate.fixture_native(gate.RUST_GREEN_SRC))
    states = {r["qualname"].split("::", 1)[1]: r["state"] for r in rules}
    assert states == {
        "Trainer::train_step": gate.CLASSIFIED_CLAIMANT,
        "Trainer::local_update": gate.CLASSIFIED_LOCAL,
    }
    assert _findings(rules, failures) == []


@pytest.mark.parametrize("case", gate.RUST_FOLD1_REDS, ids=lambda c: c[0].split()[0])
def test_fold1_red_has_exact_counts_through_rust_census(case):
    _, src, states, n_fail, n_find = case
    rules, failures, _ = gate.rust_census(gate.fixture_native(src))
    assert [r["state"] for r in rules] == states
    assert len(failures) == n_fail
    assert len(_findings(rules, failures)) == n_find


@pytest.mark.parametrize("case", gate.RUST_FOLD3_REDS, ids=lambda c: c[0].split()[0])
def test_fold3_red_has_exact_counts_through_rust_census(case):
    _, src, states, n_fail, n_find = case
    rules, failures, _ = gate.rust_census(gate.fixture_native(src))
    assert [r["state"] for r in rules] == states
    assert len(failures) == n_fail
    assert len(_findings(rules, failures)) == n_find


def test_fold3_literals_use_the_shared_lexer_and_decode_escapes():
    got = gate._rust_triples(r'[("a", "LOCAL", "caf\u{e9}\x41\n\\u"), ("b", "CONTROL", r"\u{e9}")]')
    assert got == [("a", "LOCAL", "caf\u00e9A\n\\u"), ("b", "CONTROL", "\\u{e9}")]
    for bad in (
        r'[("a", "LOCAL", "\q")]',
        r'[("a", "LOCAL", "\u{110000}")]',
        '[("a", "LOCAL", \'c\')]',
    ):
        with pytest.raises(ValueError):
            gate._rust_triples(bad)


def test_fold1_index_chain_and_raw_string_forms():
    assert gate._ident("self . layers [0] . w") == "w"
    assert gate._ident("self . w [[i , j]]") == "w"
    ok = gate._rust_triples('[("a::b", "LOCAL", r##"x "# y"##), ("c", "CONTROL", "q\\n"),]')
    assert ok == [("a::b", "LOCAL", 'x "# y'), ("c", "CONTROL", "q\n")]
    for bad in ('[("a", "LOCAL", concat!("x"))]', '[("a", "LOCAL")]', '[(1, "LOCAL", "r")]'):
        with pytest.raises(ValueError):
            gate._rust_triples(bad)


def test_enumeration_is_by_verb_or_weight_mutation_and_skips_tests():
    rules, _, _ = gate.rust_census(gate.fixture_native(gate.RUST_GREEN_SRC))
    names = {r["qualname"].rsplit("::", 2)[-1] for r in rules}
    assert "local_update" in names  # `self.w1 += ..` -- no verb in the name beyond `update`
    assert "forward" not in names  # reader: no verb, no weight write
    assert "train_smoke" not in names  # #[cfg(test)]


def test_weight_write_without_a_verb_name_is_enumerated():
    src = "pub struct S { w: f64 }\nimpl S { pub fn nudge(&mut self, d: f64) { self.w += d; } }\n"
    rules, _, _ = gate.rust_census(gate.fixture_native(src))
    assert [r["qualname"].split("::", 1)[1] for r in rules] == ["S::nudge"]
    assert rules[0]["state"] == gate.UNDETERMINABLE


def test_unparseable_declaration_is_a_failure_not_a_skip():
    src = (
        "pub const GRADIENT_CLAIM: &[(&str, &str, &str)] = "
        '&[("S::nudge", concat!("LO", "CAL"), "r")];\n'
        "impl S { pub fn nudge(&mut self) { self.w += 1.0; } }\n"
    )
    rules, failures, _ = gate.rust_census(gate.fixture_native(src))
    assert failures and "GRADIENT_CLAIM" in failures[0][1]
    assert gate.assert_findings(
        gate.counts(rules), gate.counts(rules, False), 1, BASELINE, failures
    )


def test_unknown_disposition_is_a_failure():
    src = 'pub const GRADIENT_CLAIM: &[(&str, &str, &str)] = &[("S::nudge", "LOCAl", "typo")];\n'
    _, failures, _ = gate.rust_census(gate.fixture_native(src))
    assert failures


def test_rust_facts_unavailable_is_red_never_a_skip(monkeypatch):
    def boom(root=rust_facts.ROOT):
        raise rust_facts.RustFactsUnavailable("cargo not on PATH")

    monkeypatch.setattr(rust_facts, "native_status", boom)
    rules, failures, _ = gate.rust_census()
    assert rules == []
    assert failures and "RustFactsUnavailable" in failures[0][1]
    assert gate.assert_findings(
        gate.counts(rules), gate.counts(rules, False), 1, BASELINE, failures
    )


def test_absent_native_adds_nothing_and_census_python_counts_are_untouched():
    rules, failures, n_rs = gate.rust_census(("native/ ABSENT (0 .rs files)", []))
    assert (rules, failures, n_rs) == ([], [], 0)
    py_rules, py_fail, py_n = gate.census(native=("native/ ABSENT (0 .rs files)", []))
    both, b_fail, b_n = gate.census(native=gate.fixture_native(gate.RUST_RED_SRC))
    assert not py_fail and not b_fail
    assert [r for r in both if r["qualname"].startswith("native/")] and b_n == py_n + 2
    assert [r for r in both if not r["qualname"].startswith("native/")] == py_rules


@pytest.mark.parametrize("flag", ["--assert", "--self-check"])
def test_main_reaches_the_rust_half(flag):
    assert gate.main([flag]) == 0


def test_an_optimizer_step_spelled_inside_a_string_is_not_a_step(tmp_path):
    """Codex r3 F-4: `self.model = "opt.step(".len();` is an innocent length assignment."""
    from snn_data_lake.gates import gradient_claim_gate as gcg

    d = tmp_path / "native" / "sndl-core" / "src"
    d.mkdir(parents=True)
    p = d / "lib.rs"
    p.write_text(
        "pub struct T { model: usize }\nimpl T {\n    pub fn tick(&mut self) {\n"
        '        self.model = "opt.step(".len();\n    }\n}\n',
        encoding="utf-8",
    )
    rules, failures, _ = gcg.rust_census(native=("PRESENT", rust_facts.facts([p])), root=tmp_path)
    assert not [r for r in rules if r["qualname"].endswith("tick")], rules
    assert failures == []
