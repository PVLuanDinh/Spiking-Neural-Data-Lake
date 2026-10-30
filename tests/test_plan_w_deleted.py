"""T125 (E1d route (iii), evidence/e1c-plan-w-powered.md section 8): the learned `_plan_w` update
and the plan nudge `bias[...] += _plan_w * gain` are OFF the live autonomous-loop path.

Brain-free: `AutonomousArm._pfc_bias` is called unbound on a namespace carrying only what its
reactive branch reads, with a stub PFC whose top-down bias is all zeros. A planned sequence is held
and a positive dopamine step is applied on every call, with the old `_plan_*` state planted at the
values that made the old rule grow `_plan_w` and nudge. The break this catches: any `_plan_w` term
(an update or a nudge) back on the live path."""

import ast
import types
from pathlib import Path

import snn_data_lake
from snn_data_lake.engine.autonomous_loop import ACTIONS, AutonomousArm
from snn_data_lake.gates.native_dormancy import _py_idents

T126 = (
    "tasks.md row T126 (replacement for the deleted `_plan_w` nudge) holds `_planned_seq` unread "
    "until its pre-registration opens; a reader here must land through T126, not around it"
)


class _ZeroPFC:
    def update(self, evidence):
        pass

    def top_down_bias(self, strength=1.0):
        return [0.0] * len(ACTIONS)


def _arm():
    return types.SimpleNamespace(
        pfc=_ZeroPFC(),
        pfc_gain=1.0,
        _exec=None,
        hjepa_serve=None,
        jepa=None,
        _z_goal=None,
        q={0: {a: 0.0 for a in ACTIONS}},
        _rpe=0.5,
        _planned_seq=[ACTIONS[0]],
        # the deleted rule's state, planted so the OLD rule would grow `_plan_w` and nudge
        _plan_w=0.1,
        _plan_lr=0.02,
        _plan_elig=1.0,
        _plan_base=0.0,
        _plan_tau=0.02,
    )


def test_planned_first_action_bias_carries_no_plan_w_term_after_a_dopamine_step():
    arm = _arm()
    for _ in range(3):
        bias = AutonomousArm._pfc_bias(arm, 0)
        assert list(bias) == [0.0] * len(ACTIONS)


def test_live_path_does_not_update_plan_w():
    arm = _arm()
    for _ in range(3):
        AutonomousArm._pfc_bias(arm, 0)
    assert arm._plan_w == 0.1
    assert arm._plan_elig == 1.0 and arm._plan_base == 0.0


def _planned_seq_readers(pkg):
    """Modules of package dir `pkg` that LOAD the attribute `_planned_seq` (stores are the beam
    writers and the H-JEPA reset, not readers; `+=` stores but reads the old value)."""
    out = []
    for p in sorted(Path(pkg).rglob("*.py")):
        parts = p.relative_to(pkg.parent).with_suffix("").parts
        dotted = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        augs = (
            n.target
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
            if isinstance(n, ast.AugAssign)
        )
        if "_planned_seq" in _py_idents(pkg.parent, dotted)[1] or any(
            isinstance(t, ast.Attribute) and t.attr == "_planned_seq" for t in augs
        ):
            out.append(dotted)
    return out


def test_planned_seq_has_no_reader_in_src():
    readers = _planned_seq_readers(Path(snn_data_lake.__file__).parent)
    assert readers == [], f"`_planned_seq` gained a reader in {readers}: {T126}"


def test_planted_planned_seq_load_reads_as_a_reader(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "writer.py").write_text("class A:\n    def f(self):\n        self._planned_seq = []\n")
    (pkg / "reader.py").write_text("class A:\n    def f(self):\n        x = self._planned_seq\n")
    (pkg / "extender.py").write_text(
        "class A:\n    def f(self):\n        self._planned_seq += [0]\n"
    )
    assert _planned_seq_readers(pkg) == ["pkg.extender", "pkg.reader"]
