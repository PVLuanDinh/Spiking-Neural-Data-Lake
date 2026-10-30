"""T125 (E1d route (iii), evidence/e1c-plan-w-powered.md section 8): the learned `_plan_w` update
and the plan nudge `bias[...] += _plan_w * gain` are OFF the live autonomous-loop path.

Brain-free: `AutonomousArm._pfc_bias` is called unbound on a namespace carrying only what its
reactive branch reads, with a stub PFC whose top-down bias is all zeros. A planned sequence is held
and a positive dopamine step is applied on every call, with the old `_plan_*` state planted at the
values that made the old rule grow `_plan_w` and nudge. The break this catches: any `_plan_w` term
(an update or a nudge) back on the live path."""

import types

from snn_data_lake.engine.autonomous_loop import ACTIONS, AutonomousArm


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
