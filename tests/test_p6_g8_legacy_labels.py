"""P6's anatomy census keeps frozen labels when G8 moves code into instruments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import p6_gate  # noqa: E402

from snn_data_lake.gates.move_gate import _gate_lines  # noqa: E402


def _moved_developmental_loop():
    old = "/".join(("hippocampus", "developmental_loop_brain.py"))
    new = "/".join(("instruments", "developmental_loop_brain.py"))
    source = ROOT / "src/snn_data_lake" / new
    if source.is_file():
        return {}
    source = ROOT / "src/snn_data_lake" / old
    return {old: None, new: source.read_text(encoding="utf-8")}


def test_moved_source_keeps_hippocampus_connections():
    pairs = p6_gate.region_imports(str(ROOT), _moved_developmental_loop())
    assert ("hippocampus", "basal_ganglia") in pairs
    assert ("hippocampus", "thalamus") in pairs


def test_moved_target_uses_frozen_label():
    probe = "from snn_data_lake.instruments import developmental_loop_brain\n"
    overlay = _moved_developmental_loop()
    overlay["/".join(("amygdala", "_p6_g8_probe.py"))] = probe
    pairs = p6_gate.region_imports(str(ROOT), overlay)
    assert ("amygdala", "hippocampus") in pairs


def test_p6_census_ignores_report_path_width_and_order():
    old = "/".join(("cortex", "report_order_probe.py"))
    new = "/".join(("instruments", "report_order_probe.py"))
    peer = "/".join(("reasoning", "spike_chat_lake.py"))
    before = [f"  {old}    [FACULTY] 1 write", f"about the tree: {old}, {peer}"]
    after = [f"  {new} [FACULTY] 1 write", f"about the tree: {peer}, {new}"]
    assert _gate_lines("p6_gate", before, {}) == _gate_lines(
        "p6_gate", after, {("instruments", "report_order_probe"): "cortex"}
    )
