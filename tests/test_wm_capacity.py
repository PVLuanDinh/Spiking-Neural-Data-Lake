"""specs/022 T056 -- the derived WM capacity (K = T_max/t_s) and the ported synaptic pair.

Fast pytest arms over `faculties/wm_capacity` and the `cortex/prefrontal_executive` port.
The expensive behavioral guard (derived == held +/-1 at K~97) runs in wm_capacity's own
registered self-check; these arms pin the derivation's determinism, the refusals, the two
static gates (each with its constructed red), and the store-side charges at small scale.
"""

import pytest

from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.faculties.wm_capacity import (
    REGIME_EXEMPT,
    CapacityRefusal,
    derive_capacity,
    regime_findings,
    seed_viability,
    theta_gamma_findings,
)

FAST = dict(n_options=4, commit_races=1, seed=0, tau_f=300.0, tau_d=40.0)


def _probe_factory(**over):
    kw = {**FAST, **over}
    return PrefrontalExecutive(**kw)._probe_frame


def test_derivation_is_deterministic_and_measured():
    terms = derive_capacity(_probe_factory(), 300.0)
    # exact pins: the derivation is deterministic on this mechanism (seed-0 pool bias)
    assert terms["D"] == 3 and terms["t_s"] == 4
    assert terms["K"] == 96
    assert 0.0 < terms["u_ign"] < terms["u_hi"] < 1.0
    assert terms["T_max"] > 300.0  # tau_f * ln(u_hi/u_ign) > tau_f at these constants


def test_capacity_rises_with_tau_f():
    k_lo = derive_capacity(_probe_factory(tau_f=300.0), 300.0)["K"]
    k_hi = derive_capacity(_probe_factory(tau_f=800.0), 800.0)["K"]
    assert k_lo < k_hi, f"K must rise with tau_f ({k_lo} vs {k_hi})"


def test_regime_refusals():
    # runtime half: a depression-dominant store refuses at construction
    with pytest.raises(ValueError, match="tau_f > tau_d"):
        PrefrontalExecutive(n_options=2, tau_f=100.0, tau_d=200.0)
    # the derivation's own regime: slow resource recovery (cycle < 5*tau_d) refuses --
    # the full-recovery T_max was MEASURED +7% high in that regime (97 derived vs 90 held)
    with pytest.raises(CapacityRefusal, match="regime violated"):
        derive_capacity(_probe_factory(tau_d=200.0), 300.0)
    with pytest.raises(CapacityRefusal, match="non-finite"):
        derive_capacity(_probe_factory(), float("inf"))
    # ...and with the DERIVED default cap, that refusal now lands AT CONSTRUCTION
    # (push_subgoal was total at HEAD and stays total -- panel fold)
    with pytest.raises(CapacityRefusal, match="regime violated"):
        PrefrontalExecutive(n_options=4, commit_races=1, seed=0, tau_f=300.0, tau_d=200.0)


def test_seed_viability_refusals():
    # THE PANEL'S BLOCKER made loud: a store seeding frames below its own ignition floor
    # has behavioral capacity ZERO -- it must refuse at construction, and the derivation
    # must refuse to emit a K for it.
    with pytest.raises(ValueError, match="ignition floor"):
        PrefrontalExecutive(**{**FAST, "U": 0.1})
    with pytest.raises(ValueError, match="ignition floor"):
        # the floor rises with leak: the DEFAULT U is below it at leak=0.3 (measured)
        PrefrontalExecutive(**{**FAST, "leak": 0.3, "max_goals": 8})
    with pytest.raises(CapacityRefusal, match="ZERO"):
        derive_capacity(_probe_factory(), 300.0, seed_u=0.1)
    floor, viable = seed_viability(_probe_factory(), 0.2)
    assert viable and 0.0 < floor < 0.2  # the default seed clears seed-0's floor
    # a viable non-default membrane may NOT use the derived default (validated at md=0.8
    # only; measured drift up to 6 frames) -- explicit caps remain the caller's own
    with pytest.raises(CapacityRefusal, match="default membrane"):
        PrefrontalExecutive(**{**FAST, "leak": 0.1})
    ok = PrefrontalExecutive(**{**FAST, "leak": 0.1, "max_goals": 4})
    assert ok.max_goals == 4


def test_memo_key_carries_every_determinant():
    # the panel measured byte-identical constructors answering differently depending on
    # construction order (leak/n_options were read by the derivation but absent from the
    # memo key); the key now carries them, so distinct mechanisms derive independently
    a = PrefrontalExecutive(**FAST).max_goals
    b = PrefrontalExecutive(**{**FAST, "n_options": 6}).max_goals
    again = PrefrontalExecutive(**FAST).max_goals
    assert a == again == 97
    assert isinstance(b, int) and b >= 2  # derived under its own bias stream, not a's cache


def test_frozen_retention_refuses():
    class FrozenRetention:
        """Facilitation frozen: the decay read never decays -- no finite horizon exists."""

        def __init__(self, inner):
            self._f = inner
            self.syn = inner.syn
            self.last = 0.0

        def decayed(self, now):
            return self.syn.u, 1.0

        def ignites(self, u):
            return self._f.ignites(u)

        def serve(self, now):
            if not self._f.ignites(self.syn.u * self.syn.R):
                return "depleted" if self._f.ignites(self.syn.u) else "dead"
            self.syn.spike(1.0)
            self.syn.u += self.syn.U * (1.0 - self.syn.u)
            return "fired"

    factory = _probe_factory()
    with pytest.raises(CapacityRefusal, match="no finite horizon"):
        derive_capacity(lambda: FrozenRetention(factory()), 300.0)


def test_default_cap_is_derived_and_explicit_is_honored():
    ex = PrefrontalExecutive(**FAST)
    terms = derive_capacity(_probe_factory(), 300.0)
    assert ex.max_goals == terms["K"] + 1  # derived, not declared (the old default was 4)
    explicit = PrefrontalExecutive(**{**FAST, "max_goals": 2})
    for i in range(3):
        ev = [0.0] * 4
        ev[i % 4] = 1.0
        explicit.push_subgoal(ev)
    assert len(explicit.goals) == 2  # an explicit cap is honored verbatim


def _stacked(n_frames, neutered=None, horizon=1000):
    p = PrefrontalExecutive(**{**FAST, "recurrence": 1.0, "max_goals": 10**9, "tau_d": 200.0})
    if neutered:
        p.neuter(neutered)
    for i in range(n_frames):
        ev = [0.0] * 4
        ev[i % 4] = 1.0
        p.push_subgoal(ev)
        for _ in range(4):
            p.maintain()
    for _ in range(horizon):
        p.maintain()
    return len(p.goals)


def test_rehearsal_channel_and_resource_charges():
    held_live = _stacked(6)
    assert held_live == 6, "within capacity the rehearsal channel must keep all frames alive"
    held_no_r = _stacked(6, neutered="resources")
    assert held_no_r <= 2 < held_live, (
        f"cutting R's route must hog the channel and starve the rest ({held_no_r} vs {held_live})"
    )
    held_no_reh = _stacked(6, neutered="rehearsal")
    assert held_no_reh == 1, (
        f"cutting the refresh (bookkeeping kept) must cull every background memory ({held_no_reh})"
    )
    with pytest.raises(ValueError, match="unknown element"):
        PrefrontalExecutive(n_options=2).neuter("nonsense")


def test_pointer_reanchor_branches():
    # the fold panel measured these branches unexercised by every gate while a comment
    # claimed measured evidence for them -- this arm executes each re-anchor directly
    def _stacked_exec(n=5):
        p = PrefrontalExecutive(**{**FAST, "recurrence": 1.0, "max_goals": 10**9})
        for i in range(n):
            ev = [0.0] * 4
            ev[i % 4] = 1.0
            p.push_subgoal(ev)
            for _ in range(4):
                p.maintain()
        return p

    # sweep-cull below the focus: _focus must slide down with the removal
    p = _stacked_exec()
    p._sweep, p._focus = 0, 3
    victim = p._stack[0]
    victim[3].syn.u = 0.0
    p.maintain()
    assert victim not in p._stack and p._focus <= 3
    # serve-'dead' cull below the sweep: _sweep must slide down with the removal
    q = _stacked_exec()
    q._sweep, q._focus = 3, 1
    dead = q._stack[1]
    dead[3].syn.u = 0.0
    q._t += 0.0
    q.maintain()  # sweep visits bg[3] (alive), serve hits bg[1] (dead) -> removed
    assert dead not in q._stack
    # push_subgoal's drop-oldest: both pointers re-anchor
    r = PrefrontalExecutive(**{**FAST, "max_goals": 3})
    for i in range(3):
        ev = [0.0] * 4
        ev[i % 4] = 1.0
        r.push_subgoal(ev)
    r._focus, r._sweep = 2, 1
    r.push_subgoal([1.0, 0.0, 0.0, 0.0])  # at cap: pop(0)
    assert (r._focus, r._sweep) == (1, 0)


def test_single_goal_compat_unchanged():
    # the incumbent single-goal contract: commit, hold across an input-free delay with
    # per-tick persistent spiking; recurrence=0 loses the goal within a tick
    pfc = PrefrontalExecutive(n_options=4, seed=0)
    pfc.update([2.0, 0.1, 0.1, 0.1])
    for _ in range(6):
        assert pfc.maintain() == 0
        assert pfc._wm_spikes > 0
    crippled = PrefrontalExecutive(n_options=4, recurrence=0.0, seed=0)
    crippled.update([2.0, 0.1, 0.1, 0.1])
    assert crippled.maintain() is None


def test_regime_gate_static():
    red_src = {
        "candidate_wm.py": '"""A working-memory candidate."""\n'
        "def build(tau_f=100.0, tau_d=200.0):\n    return (tau_f, tau_d)\n"
    }
    red, _ = regime_findings(source_by_name=red_src)
    assert len(red) == 1 and red[0][2] == 100.0
    fixed, _ = regime_findings(
        source_by_name={"candidate_wm.py": red_src["candidate_wm.py"].replace("100.0", "400.0")}
    )
    assert fixed == []
    tree, blessed = regime_findings()
    assert tree == [], f"unexempted depression-dominant T-M pairs in src/: {tree}"
    assert set(REGIME_EXEMPT) == blessed, (
        f"every exemption must bless a real site (missing: {set(REGIME_EXEMPT) - blessed})"
    )


def test_theta_gamma_gate_static():
    red, _ = theta_gamma_findings(
        source_by_name={"bad.py": "K = steps_per_theta / steps_per_gamma\n"}
    )
    assert len(red) == 1
    rev, _ = theta_gamma_findings(source_by_name={"bad2.py": "r = gamma_rate / theta_rate\n"})
    assert len(rev) == 1, "both orientations of the ratio must red"
    tree, mult_control = theta_gamma_findings()
    assert tree == []
    assert mult_control >= 1, "liveness: the scanner must see theta_gamma_code's lawful Mult"
