"""The Rust half of eprop_transport_gate: fixture RED arm, GREEN twin, fail-closed, entry point."""

import subprocess

import pytest

from snn_data_lake.gates import eprop_transport_gate as gate
from snn_data_lake.gates import rust_facts


def _tree(root, sources):
    """A constructed `native/` tree under `root` (a git repo: rust_facts reads git's view)."""
    for rel, text in sources.items():
        p = root / "native" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _scan(tmp_path, body):
    _tree(tmp_path, {"sndl-core/src/lib.rs": "impl N {\n" + body + "\n}\n"})
    return gate.scan_rust(tmp_path)[1]


def _rules(found):
    return sorted(f[3] for f in found)


def test_red_arm_is_exactly_the_two_preregistered_findings(tmp_path):
    _tree(tmp_path, gate._rust_fixture(red=True))
    status, found = gate.scan_rust(tmp_path)
    assert status.startswith("native/ PRESENT")
    assert _rules(found) == ["T1", "T2"], found


def test_green_twin_is_zero(tmp_path):
    _tree(tmp_path, gate._rust_fixture(red=False))
    assert gate.scan_rust(tmp_path)[1] == []


def test_real_tree_prints_absent_and_passes(capsys):
    assert gate.main(["--assert"]) == 0
    assert "native/ ABSENT" in capsys.readouterr().out


def test_main_reds_on_a_red_tree_and_greens_on_the_twin(tmp_path, monkeypatch, capsys):
    red, green = tmp_path / "red", tmp_path / "green"
    red.mkdir(), green.mkdir()
    _tree(red, gate._rust_fixture(red=True))
    _tree(green, gate._rust_fixture(red=False))
    monkeypatch.setattr(gate, "_REPO_ROOT", red)
    assert gate.main(["--assert"]) == 1
    assert "gate FAIL: Rust weight transport" in capsys.readouterr().out
    monkeypatch.setattr(gate, "_REPO_ROOT", green)
    assert gate.main(["--assert"]) == 0


def test_rust_facts_unavailable_with_rs_present_is_red(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, gate._rust_fixture(red=False))
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(rust_facts.shutil, "which", lambda _name: None)
    assert gate.main(["--assert"]) == 1
    assert "RustFactsUnavailable" in capsys.readouterr().out


T1_SPELLINGS = [
    "let d = self.w2.t().dot(e); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.reversed_axes(); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.view().permuted_axes([1, 0]); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.view().permuted_axes((1, 0)); self.w1.scaled_add(-lr, &d);",
    "self.w2.swap_axes(0, 1); self.w1.scaled_add(-lr, &self.w2);",
    "let d = self.w2.transpose(); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.swap_dims(0, 1); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.permute([1, 0]); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.movedim(0, 1); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.transpose_mut(); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.adjoint(); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.tr_mul(&e); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.gemv_tr(1.0, &e); self.w1.scaled_add(-lr, &d);",
    "self.w1 -= &self.w2.t().dot(e);",
    "let d = self.w2.t().dot(e); let d2 = d * 2.0; self.w1.scaled_add(-lr, &d2);",
    "let d = self.w2.t().dot(e); for w in self.w1.iter_mut() { *w -= d; }",
]


@pytest.mark.parametrize("body", T1_SPELLINGS)
def test_t1_spellings_fire(tmp_path, body):
    found = _scan(tmp_path, f"fn f(&mut self, e: &A, lr: f64) {{ {body} }}")
    assert _rules(found) == ["T1"], found


T1_CLEAN = [
    "let h = self.w.t().dot(x); h",  # a transpose feeding an ACTIVATION
    "let d = x.t().dot(e); self.w1.scaled_add(-lr, &d);",  # transpose of a non-weight
    "let d = self.w2.t().dot(e); let d = e.clone(); self.w1.scaled_add(-lr, &d);",  # re-let clears
    "self.w2.permuted_axes([0, 1]);",  # identity permutation
    "let d = self.fb.t().dot(e); self.w1.scaled_add(-lr, &d);",  # fixed feedback, not forward
    "for w in self.w1.iter_mut() { *w -= 1.0; }",
]


@pytest.mark.parametrize("body", T1_CLEAN)
def test_t1_controls_stay_clean(tmp_path, body):
    assert _scan(tmp_path, f"fn f(&mut self, e: &A, x: &A, lr: f64) {{ {body} }}") == []


@pytest.mark.parametrize(
    "write, read",
    [
        ("self.w[[i, j]] += d;", "let g = self.w[[j, i]];"),
        ("self.w[(i, j)] += d;", "let g = self.w[(j, i)];"),
        ("self.w[i][j] += d;", "let g = self.w[j][i];"),
        ("self.w[[i, j]] += d;", "let g = self.w[j][i];"),
        ("self.w[i][j] += d;", "let g = self.w[(j, i)];"),
    ],
)
def test_t2_index_forms_are_normalised(tmp_path, write, read):
    found = _scan(
        tmp_path, f"fn a(&mut self, d: f64) {{ {write} }}\nfn b(&self) -> f64 {{ {read} g }}"
    )
    assert _rules(found) == ["T2"], found


def test_t2_symmetry_check_and_same_order_read_are_clean(tmp_path):
    same_order = "fn a(&mut self, d: f64) { self.w[[i, j]] += d; let g = self.w[[i, j]]; }"
    validator = "fn check(&self) -> bool { self.w[[i, j]] == self.w[[j, i]] }"
    assert _scan(tmp_path, same_order) == []
    assert _scan(tmp_path / "v", validator) == []


FOLD1_RED = [
    "self.w2.transpose_mut(); self.w1 += &self.w2;",  # T-1 in place, returns unit
    "self.w2.transpose_mut(); self.w1.scaled_add(-lr, &self.w2);",
    "y.gemv_tr(1.0, &self.w2, e, 0.0); self.w1.scaled_add(-lr, &y);",  # weight as the ARGUMENT
    "self.w1.gemv_tr(1.0, &self.w2, e, 0.0);",
    "let d = e.tr_mul(&self.w2); self.w1.scaled_add(-lr, &d);",
    "let d = e.ad_mul(&self.w2); self.w1.scaled_add(-lr, &d);",
    "let d = self.w2.t().to_owned(); self.w1 += &d; let d = e.clone();",  # T-2 same line
    'let s = "x"; let d = self.w2.t().dot(e); self.w1.scaled_add(-lr, &d);',  # strip keeps code
    'let s = r#"quote " inside"#; let d = self.w2.t().dot(e); self.w1.scaled_add(-lr, &d);',
]


@pytest.mark.parametrize("body", FOLD1_RED)
def test_fold1_reds_fire_exactly_once(tmp_path, body):
    found = _scan(tmp_path, f"fn f(&mut self, e: &A, lr: f64) {{ {body} }}")
    assert _rules(found) == ["T1"], found


FOLD1_CLEAN = [
    'let s = "w2.t()"; *w += s.len() as f64;',  # T-3 transpose spelling inside a string
    'let s = r#"w2.t()"#; *w += s.len() as f64;',
    "self.fb.transpose_mut(); self.w1 += &self.fb;",  # fixed feedback, not forward
    "self.w2.transpose_mut(); let h = self.w2.dot(e); h",  # activation only
    "y.gemv_tr(1.0, &self.fb, e, 0.0); self.w1.scaled_add(-lr, &y);",
    "y.gemv_tr(1.0, &self.w2, e, 0.0); y",  # activation only
    "let d = self.w2.t().to_owned(); let d = e.clone(); self.w1 += &d;",  # shadow BEFORE update
    "let d = e.tr_dot(&self.w2); self.w1 += d;",  # tr_dot is a SCALAR (Oja/Hebbian y = w . x)
    "let y = x.tr_dot(&self.w); self.w += y * lr;",
]

FOLD3_VERBS = {
    "gemm_tr": "{r}.gemm_tr(1.0, &{w}, &e, 0.0);",
    "gemm_ad": "{r}.gemm_ad(1.0, &{w}, &e, 0.0);",
    "gemv_ad": "{r}.gemv_ad(1.0, &{w}, e, 0.0);",
    "tr_copy_from": "{r}.tr_copy_from(&{w});",
    "ad_copy_from": "{r}.ad_copy_from(&{w});",
}
FOLD3_RED = [
    t.format(r=r, w="self.w2") + tail
    for t in FOLD3_VERBS.values()
    for r, tail in (("y", " self.w1.scaled_add(-lr, &y);"), ("self.w1", ""))
]
FOLD3_CLEAN = [
    t.format(r=r, w=w) + tail
    for t in FOLD3_VERBS.values()
    for r, w, tail in (
        ("y", "self.fb", " self.w1.scaled_add(-lr, &y);"),  # fixed feedback
        ("y", "self.w2", " y"),  # activation only
    )
]


@pytest.mark.parametrize("body", FOLD3_RED)
def test_fold3_implicit_transpose_writers_fire_exactly_once(tmp_path, body):
    found = _scan(tmp_path, f"fn f(&mut self, e: &A, lr: f64) {{ {body} }}")
    assert _rules(found) == ["T1"], found


@pytest.mark.parametrize("body", FOLD3_CLEAN)
def test_fold3_feedback_and_activation_controls_stay_clean(tmp_path, body):
    assert _scan(tmp_path, f"fn f(&mut self, e: &A, lr: f64) {{ {body} }}") == []


FOLD4_RED = [
    # N5-3: the writer inside the assignment's block RHS taints y BEFORE the assignment is evaluated
    "self.w1 = {\n    let mut y = self.fb.clone();\n    y.tr_copy_from(&self.w2);\n    y\n};",
]
FOLD4_CLEAN = [
    # N5-3: the inner let shadows the tainted name before the assignment ends
    "let d = self.w2.t().to_owned(); self.w1 = {\n    let d = e.clone();\n    d\n};",
]


@pytest.mark.parametrize("body", FOLD4_RED)
def test_fold4_reds_fire_exactly_once(tmp_path, body):
    found = _scan(tmp_path, f"fn f(&mut self, e: &A, x: &A, lr: f64) {{ {body} }}")
    assert _rules(found) == ["T1"], found


@pytest.mark.parametrize("body", FOLD4_CLEAN)
def test_fold4_controls_stay_clean(tmp_path, body):
    assert _scan(tmp_path, f"fn f(&mut self, e: &A, x: &A, lr: f64) {{ {body} }}") == []


@pytest.mark.parametrize("body", FOLD1_CLEAN)
def test_fold1_controls_stay_clean(tmp_path, body):
    assert _scan(tmp_path, f"fn f(&mut self, e: &A, w: &mut f64, lr: f64) {{ {body} }}") == []


def test_fold1_red_arm_is_exactly_three_t1_and_the_string_fn_is_silent(tmp_path):
    _tree(tmp_path, gate._rust_fixture(red=True, fold=1))
    found = gate.scan_rust(tmp_path)[1]
    assert _rules(found) == ["T1", "T1", "T1"], found
    assert sorted(f[2] for f in found) == ["as_arg", "in_place", "same_line"], found


def test_fold1_green_twin_is_zero(tmp_path):
    _tree(tmp_path, gate._rust_fixture(red=False, fold=1))
    assert gate.scan_rust(tmp_path)[1] == []


TEST_ONLY_REFERENCE = """\
pub struct N { w1: A, w2: A }
#[cfg(test)]
mod tests {
    use super::*;
    fn bptt_reference(n: &mut N, e: &A) { let d = n.w2.t().dot(e); n.w1 += &d; }
    #[test]
    fn swapped_read() { let g = w[[j, i]]; w[[i, j]] += g; }
}
"""


def test_test_only_rust_is_out_of_scope_like_python_tests(tmp_path):
    """Panel lens C (MINOR): a #[cfg(test)] BPTT reference / test body is test code -- the Rust
    analogue of tests/, which the Python half never scans -- so it must not red T1 or T2."""
    _tree(tmp_path, {"sndl-core/src/lib.rs": TEST_ONLY_REFERENCE})
    assert gate.scan_rust(tmp_path)[1] == []


def test_a_crate_tests_dir_reference_is_out_of_scope(tmp_path):
    """Codex r3 F-5: native/<crate>/tests/reference.rs (no #[cfg(test)]) is test code."""
    _tree(
        tmp_path,
        {
            "sndl-core/src/lib.rs": "pub struct N { w1: A, w2: A }\n",
            "sndl-core/tests/reference.rs": (
                "fn reference(n: &mut N, e: &A) { let d = n.w2.t().dot(e); n.w1 += &d; }\n"
            ),
        },
    )
    assert gate.scan_rust(tmp_path)[1] == []


def test_fold3_red_arm_is_exactly_five_t1_and_the_tr_dot_fn_is_silent(tmp_path):
    _tree(tmp_path, gate._rust_fixture(red=True, fold=3))
    found = gate.scan_rust(tmp_path)[1]
    assert _rules(found) == ["T1"] * 5, found
    assert sorted(f[2] for f in found) == ["ad_copy", "gemm_ad", "gemm_tr", "gemv_ad", "tr_copy"], (
        found
    )


def test_fold3_green_twin_is_zero(tmp_path):
    _tree(tmp_path, gate._rust_fixture(red=False, fold=3))
    assert gate.scan_rust(tmp_path)[1] == []
