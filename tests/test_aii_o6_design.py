"""Reds for `tools/aii_o6_design.py` (O-6 of the A_ii stack-repair pre-registration DRAFT, section 4 + Fold 2).

The script is the committed sample-size arithmetic of `specs/022-unified-brain-arc/evidence/aii-stack-repair-prereg-DRAFT.md`:
an exact stdlib binomial lattice for a sequential design (looks, Lan-DeMets O'Brien-Fleming spending, non-binding futility,
cap), a joint lattice over the candidate / original / bare arms (RD-4, RD-5, the paired guard, P(VOID)) and a Monte Carlo
check (RNG seed 20261029). VERIFY THE VERIFIER: each test below fires a constructed case the design must catch. Four reds
are the brief's; the rest pin the DRAFT's printed numbers (a mismatch is OWED to the integrator, never edited to fit).

stdlib only. Loaded by ABSOLUTE path, as tests/test_red_reachability.py:18-22 does for its tool.
Run: pytest tests/test_aii_o6_design.py   (campaign rule 1: NOT run while the E1d campaign holds the host)
"""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "aii_o6_design.py"
_spec = importlib.util.spec_from_file_location("aii_o6_design", _PATH)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

D = M.derive()  # alpha 0.05, looks 8/15/23: the DRAFT's registered design


def _close(got, want, tol):
    assert abs(got - want) <= tol, f"{got} != {want} (tol {tol})"


# ---- the four reds named by the brief -------------------------------------------------------------------------------


def test_red_planted_null():
    """A candidate that is a planted null (rate 0) never reaches MET; at the null bound the realised alpha is <= 0.05,
    and a planted LAX boundary (x >= 1 at every look) is caught by the same alpha check (so the check can fire)."""
    assert M.joint(D, 0.0, 0.0, 1.0)["met"] == 0.0
    assert M.cum_alpha(D["looks"], D["eff"], D["p0"])[-1] <= 0.05 + 1e-12
    lax = [1, 1, 1]
    assert M.cum_alpha(D["looks"], lax, D["p0"])[-1] > 0.05
    assert M.alpha_ok(D, D["eff"]) and not M.alpha_ok(D, lax)


def test_red_planted_effect():
    """A planted effect (rate 0.9) is detected with power >= 0.97 at the measured control points; a planted never-crossing
    boundary (x >= cap + 1) has power 0, so the power check can fire too."""
    _close(M.joint(D, 0.9, 0.0, 1.0)["met"], 1.0, 1e-3)
    assert M.joint(D, 0.9, 0.0, 1.0)["met"] >= 0.97
    never = [n + 1 for n in D["looks"]]
    assert M.cum_alpha(D["looks"], never, 0.9)[-1] == 0.0
    assert M.power_ok(D, D["eff"]) and not M.power_ok(D, never)


def test_red_original_always_solves_voids_at_l1():
    """An original stack that always solves (RD-4) VOIDs at L1 with certainty, whatever the candidate does."""
    for pc in (0.0, D["p0"], D["p1"], 0.9):
        j = M.joint(D, pc, 1.0, 1.0)
        assert j["void"] == 1.0 and j["void_l1"] == 1.0 and j["met"] == 0.0, (pc, j)


def test_red_measured_point_rates():
    """At the measured point rates (original 0/12 -> rate 0, bare 4/4 -> rate 1) no validity red can fire (P(VOID) = 0)
    and P(MET) at p1 is the DRAFT's 0.8199."""
    j = M.joint(D, D["p1"], 0.0, 1.0)
    assert j["void"] == 0.0
    _close(j["met"], 0.8199, 5e-5)


# ---- the DRAFT's printed numbers (section 4, section 3.3) ------------------------------------------------------------


def test_inputs_and_fixed_sample_target():
    _close(D["p0"], 0.2209, 5e-5)
    _close(D["p1"], 0.4729, 5e-5)
    assert (D["n_fixed"], D["c_fixed"]) == (23, 9)
    _close(D["alpha_fixed"], 0.0491, 5e-5)
    _close(D["power_fixed"], 0.8394, 5e-5)


def test_looks_spending_and_efficacy_boundaries():
    assert D["looks"] == [8, 15, 23]
    for got, want in zip(D["spend"], (0.00089, 0.01523, 0.05), strict=True):
        _close(got, want, 5e-6)
    assert D["eff"] == [7, 8, 9]
    for got, want in zip(D["alpha_real"], (0.00017, 0.00806, 0.04997), strict=True):
        _close(got, want, 5e-6)


def test_futility_bounds():
    assert D["fut"] == [1, 3]
    _close(M.cp_upper(1, 8, 0.95), 0.471, 5e-4)
    _close(M.cp_upper(3, 15, 0.95), 0.440, 5e-4)


def test_cap_power_and_inflation():
    assert D["cap"] == 23 and D["inflation"] == 1.00
    _close(D["power_cap_ignored"], 0.8405, 5e-5)
    _close(D["power_cap_obeyed"], 0.8199, 5e-5)
    d24 = M.derive(looks=[8, 16, 24])
    assert d24["eff"][-1] == 10 or d24["eff"] == [7, 8, 10]
    _close(d24["power_cap_ignored"], 0.785, 5e-4)


def test_validity_tolerances():
    assert D["rd4"] == [6, 9, 11]
    assert D["rd5"] == [1, 2, 5]
    _close(D["void4_err"], 0.0078, 5e-5)
    _close(D["void5_err"], 0.0086, 5e-5)


def test_operating_characteristics_guard_idle():
    want = {0.0: 0.000, 0.2209: 0.047, 0.35: 0.398, 0.4729: 0.820, 0.6: 0.980, 0.8: 0.9999, 0.9: 1.000, 1.0: 1.000}
    for p, w in want.items():
        pc = D["p0"] if p == 0.2209 else D["p1"] if p == 0.4729 else p
        _close(M.joint(D, pc, 0.0, 1.0)["met"], w, 5e-4)
    _close(M.joint(D, D["p0"], 0.0, 1.0)["en"], 14.8, 0.05)
    _close(M.joint(D, D["p1"], 0.0, 1.0)["en"], 18.7, 0.05)
    _close(M.joint(D, 0.9, 0.0, 1.0)["en"], 9.3, 0.05)
    _close(M.joint(D, 1.0, 0.0, 1.0)["en"], 8.0, 0.05)


def test_joint_worst_case_met_and_void():
    """Worst case the measured counts allow: original at p0 AND bare at p1 (guard included)."""
    want = {"p0": (0.0450, 0.0114), 0.35: (0.3853, 0.0138), "p1": (0.8032, 0.0133), 0.6: (0.9683, 0.0112), 0.9: (0.9916, 0.0084)}
    for key, (wm, wv) in want.items():
        pc = D[key] if isinstance(key, str) else key
        j = M.joint(D, pc, D["p0"], D["p1"])
        _close(j["met"], wm, 5e-5)
        _close(j["void"], wv, 5e-5)


def test_alpha_and_looks_are_parameters():
    """O-13 is re-opened: under a different family alpha the design re-derives, with realised alpha under it."""
    d = M.derive(alpha=0.025)
    assert d["alpha"] == 0.025
    assert d["n_fixed"] > D["n_fixed"] and d["looks"][-1] == d["cap"]
    assert d["alpha_real"][-1] <= 0.025 + 1e-12
    assert d["eff"][-1] > D["eff"][-1] or d["cap"] > D["cap"]
    e = M.derive(looks=[10, 20, 23])
    assert e["looks"] == [10, 20, 23] and e["alpha_real"][-1] <= 0.05 + 1e-12


def test_monte_carlo_seeded_and_matches_exact():
    r = M.mc(D, [D["p0"], D["p1"]], reps=20000, seed=20261029)
    _close(r[D["p0"]], M.joint(D, D["p0"], 0.0, 1.0)["met"], 0.008)
    _close(r[D["p1"]], M.joint(D, D["p1"], 0.0, 1.0)["met"], 0.012)
    assert r == M.mc(D, [D["p0"], D["p1"]], reps=20000, seed=20261029)
