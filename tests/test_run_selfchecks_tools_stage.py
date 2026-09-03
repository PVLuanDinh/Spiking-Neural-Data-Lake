"""H1 (2026-09-02): the tools-gate stage of run_selfchecks has a reachable red.

`run_selfchecks.run_tools_gates` runs each tools/ gate as a subprocess and returns the failed
names. A gate that exits non-zero must come back failed (the red); one that exits zero must not
(the positive control), so a stage that swallowed exit codes could not pass this file.
"""

import pathlib

import run_selfchecks as rs

ROOT = pathlib.Path(rs.__file__).resolve().parent


def test_tools_stage_red_and_control(tmp_path):
    red = tmp_path / "red_gate.py"
    red.write_text("raise SystemExit(1)\n", encoding="utf-8")
    green = tmp_path / "green_gate.py"
    green.write_text("print('ok')\n", encoding="utf-8")
    lines = []
    failed = rs.run_tools_gates([[str(red)], [str(green), "--assert"]], emit=lines.append)
    assert failed == ["tools/red_gate.py"], (failed, lines)
    assert any(line.startswith("FAIL  tools/red_gate.py") for line in lines), lines
    assert any(line.startswith("PASS  tools/green_gate.py") for line in lines), lines
    assert "1/2 tools gates passed" in lines[-1], lines


def test_default_gate_list_names_the_six_tools_gates():
    names = [argv[0] for argv in rs.TOOLS_GATES]
    assert names == [
        "tools/p4c_ratchet.py",
        "tools/p6_gate.py",
        "tools/dependency_ledger.py",
        "tools/red_reachability.py",
        "tools/citation_guard.py",
        "tools/doc_command_guard.py",
    ]
    for name in names:
        assert (ROOT / name).exists(), name
