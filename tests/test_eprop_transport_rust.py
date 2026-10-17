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
