# ruff: noqa: E501
"""Reds for `src/snn_data_lake/instruments/aii_o6_design.py` (O-6 of the A_ii stack-repair pre-registration DRAFT, section 4 + Fold 2).

The script is the committed sample-size arithmetic of `specs/022-unified-brain-arc/evidence/aii-stack-repair-prereg-DRAFT.md`:
an exact stdlib binomial lattice for a sequential design (looks, Lan-DeMets O'Brien-Fleming spending, non-binding futility,
cap), a joint lattice over the candidate / original / bare arms (RD-4, RD-5, the paired guard, P(VOID)) and a Monte Carlo
check (RNG seed 20261029). VERIFY THE VERIFIER: each test below fires a constructed case the design must catch. Four reds
are the brief's; the rest pin the DRAFT's printed numbers (a mismatch is OWED to the integrator, never edited to fit).

stdlib only (the module under test is registered in the governed tree, so it is imported by name).
Run: pytest tests/test_aii_o6_design.py   (campaign rule 1: NOT run while the E1d campaign holds the host)
"""

import contextlib
import io
import math

from snn_data_lake.instruments import aii_o6_design as M

D = M.derive(
    alpha=0.05
)  # Fold 2 reproduction (alpha 0.05, looks 8/15/23): the SUPERSEDED design, a regression test of Fold 2
A1 = (
    0.05 * 6 / math.pi**2
)  # D-82 series, candidate 1: 0.0303964, computed here independently of the script
D1 = M.derive(alpha=A1)  # the REGISTERED design (Fold 3): n 27, cap 30, looks 10/20/30


def _close(got, want, tol):
    assert abs(got - want) <= tol, f"{got} != {want} (tol {tol})"


def _raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc:
        return True
    return False


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
    d24 = M.derive(alpha=0.05, looks=[8, 16, 24])
    assert d24["eff"][-1] == 10 or d24["eff"] == [7, 8, 10]
    _close(d24["power_cap_ignored"], 0.785, 5e-4)


def test_validity_tolerances():
    assert D["rd4"] == [6, 9, 11]
    assert D["rd5"] == [1, 2, 5]
    _close(D["void4_err"], 0.0078, 5e-5)
    _close(D["void5_err"], 0.0086, 5e-5)


def test_operating_characteristics_guard_idle():
    want = {
        0.0: 0.000,
        0.2209: 0.047,
        0.35: 0.398,
        0.4729: 0.820,
        0.6: 0.980,
        0.8: 0.9999,
        0.9: 1.000,
        1.0: 1.000,
    }
    for p, w in want.items():
        pc = D["p0"] if p == 0.2209 else D["p1"] if p == 0.4729 else p
        _close(M.joint(D, pc, 0.0, 1.0)["met"], w, 5e-4)
    _close(M.joint(D, D["p0"], 0.0, 1.0)["en"], 14.8, 0.05)
    _close(M.joint(D, D["p1"], 0.0, 1.0)["en"], 18.7, 0.05)
    _close(M.joint(D, 0.9, 0.0, 1.0)["en"], 9.3, 0.05)
    _close(M.joint(D, 1.0, 0.0, 1.0)["en"], 8.0, 0.05)


def test_joint_worst_case_met_and_void():
    """Worst case the measured counts allow: original at p0 AND bare at p1 (guard included)."""
    want = {
        "p0": (0.0450, 0.0114),
        0.35: (0.3853, 0.0138),
        "p1": (0.8032, 0.0133),
        0.6: (0.9683, 0.0112),
        0.9: (0.9916, 0.0084),
    }
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


# ---- Round 5 (enforcer 7019): one red per REGISTERED number at alpha_1 = 0.0303964 ------------------------------------
# Sections above pin the SUPERSEDED alpha-0.05 design (Fold 2 regression). The tests below pin what the record registers.


def test_alpha1_is_the_default_design():
    _close(M.ALPHA_1, A1, 1e-12)
    assert M.derive()["alpha"] == M.ALPHA_1 and M.derive()["looks"] == [10, 20, 30]


def test_registered_alpha1_design():
    assert (D1["n_fixed"], D1["c_fixed"]) == (27, 11)
    _close(D1["alpha_fixed"], 0.0227, 5e-5)
    _close(D1["power_fixed"], 0.8085, 5e-5)
    assert D1["looks"] == [10, 20, 30] and D1["cap"] == 30 and D1["inflation"] == 1.11
    assert D1["eff"] == [8, 10, 12]
    for got, want in zip(D1["spend"], (0.00018, 0.00802, 0.03040), strict=True):
        _close(got, want, 5e-6)
    for got, want in zip(D1["alpha_real"], (0.00017, 0.00563, 0.02214), strict=True):
        _close(got, want, 5e-6)
    assert D1["fut"] == [1, 5] and D1["fut_cap"] == 9
    assert D1["rd4"] == [7, 10, 14] and D1["rd5"] == [1, 4, 8]
    _close(D1["void4_err"], 0.0077, 5e-5)
    _close(D1["void5_err"], 0.0088, 5e-5)
    _close(D1["power_cap_ignored"], 0.8400, 5e-5)
    _close(D1["power_cap_obeyed"], 0.8320, 5e-5)
    _close(D1["power_joint_worst"], 0.8198, 5e-5)


def test_registered_alpha1_joint_rows():
    want = {
        "p0": (0.0210, 0.0097),
        0.35: (0.3377, 0.0135),
        "p1": (0.8198, 0.0128),
        0.6: (0.9801, 0.0100),
        0.9: (0.9962, 0.0038),
    }
    for key, (wm, wv) in want.items():
        pc = D1[key] if isinstance(key, str) else key
        j = M.joint(D1, pc, D1["p0"], D1["p1"])
        _close(j["met"], wm, 5e-5)
        _close(j["void"], wv, 5e-5)
    _close(M.joint(D1, D1["p1"], 0.0, 1.0)["met"], 0.8320, 5e-5)
    assert M.joint(D1, D1["p1"], 0.0, 1.0)["void"] == 0.0


def test_registered_alpha1_drift_row():
    want = {
        "p0": (0.0212, 0.0053),
        0.25: (0.0500, 0.0139),
        0.30: (0.1431, 0.0542),
        0.35: (0.2668, 0.1483),
    }
    for key, (wm, wv) in want.items():
        q = D1["p0"] if key == "p0" else key
        j = M.joint(D1, q, q, 1.0)
        _close(j["met"], wm, 5e-5)
        _close(j["void"], wv, 5e-5)


def test_registered_alpha1_planted_mutations_are_caught():
    """The pins above must fire on a planted cap 31 / efficacy [8,10,13] / lax boundary (the reviewer's plants)."""
    assert M.derive(alpha=A1, looks=[11, 21, 31])["cap"] != D1["cap"]
    off = dict(D1, eff=[8, 10, 13])
    assert abs(M.joint(off, D1["p1"], D1["p0"], D1["p1"])["met"] - 0.8198) > 5e-5
    lax = dict(D1, eff=[1, 1, 1])
    assert (
        M.cum_alpha(lax["looks"], lax["eff"], D1["p0"])[-1] > A1
    )  # a lax boundary spends more than alpha_1
    assert (
        M.joint(lax, D1["p0"], D1["p0"], D1["p1"])["met"] > A1
    )  # and the joint lattice sees it (joint(0)==0 would not)


# ---- MAJOR-2: the cap clause (cap = target x inflation, power at the cap >= .8, smallest such cap) --------------------


def _joint_power_at_cap(cap):
    lk = [math.ceil(cap / 3), round(2 * cap / 3), cap]
    d = M._design(lk, A1, D1["p0"], D1["p1"], D1["conf"], D1["void_err"])
    return M.joint(d, D1["p1"], D1["p0"], D1["p1"])["met"]


def test_cap_is_the_smallest_reaching_joint_power():
    assert _joint_power_at_cap(D1["cap"]) >= 0.80
    assert _joint_power_at_cap(D1["cap"] - 1) < 0.80
    assert (
        D1["cap"] > D1["n_fixed"]
    )  # deleting the power condition would stop at the fixed-sample target 27 (.7869)
    for cap, want in ((27, 0.7869), (28, 0.7365), (29, 0.7822), (31, 0.8384), (32, 0.7980)):
        _close(_joint_power_at_cap(cap), want, 5e-5)


def test_no_cap_reaching_power_raises():
    assert _raises(ValueError, M.derive, alpha=A1, cap_max=28)


# ---- NIT: failing bare arm, VOID-first ordering, MC over the joint lattice, input validation ----------------------------


def test_red_failing_bare_arm_voids_at_l1():
    j = M.joint(D1, D1["p1"], D1["p0"], 0.0)  # RD-5: a bare learner that never solves
    assert j["void"] == 1.0 and j["void_l1"] == 1.0 and j["met"] == 0.0


def test_void_is_checked_before_met():
    """Candidate crosses and the original is invalid in the same cell: VOID wins even with the guard off."""
    j = M.joint(D1, 1.0, 1.0, 1.0, guard=False)
    assert j["void"] == 1.0 and j["met"] == 0.0


def test_monte_carlo_covers_the_joint_lattice():
    r = M.mc_joint(D1, (D1["p1"], D1["p0"], D1["p1"]), reps=20000, seed=20261029)
    ex = M.joint(D1, D1["p1"], D1["p0"], D1["p1"])
    _close(r["met"], ex["met"], 0.012)
    _close(r["void"], ex["void"], 0.006)
    assert r == M.mc_joint(D1, (D1["p1"], D1["p0"], D1["p1"]), reps=20000, seed=20261029)


def test_inputs_are_validated():
    assert _raises(
        ValueError, M.derive, conf=0.99
    )  # no n < 400 meets the target: raise, not a TypeError on None
    for bad in ([], [20, 10, 30], [0, 5, 9], [10, 10, 30]):
        assert _raises(ValueError, M.derive, alpha=A1, looks=bad), bad
    assert _raises(SystemExit, M.main, ["--looks", "20,10"])


# ---- MINOR-1 / MINOR-4: report() derives its bound lines from the design; MET bounds are at 1 - alpha_1 ----------------


def _report(d):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M.report(d, reps=500, seed=1)
    return buf.getvalue()


def test_report_bound_lines_follow_the_design():
    out = _report(D1)
    for frag in (
        "x=8/10",
        "x=10/20",
        "x=12/30",
        "0.457",
        "0.280",
        "0.233",
        "x=10/30",
        "x=11/30",
        "0.499",
        "0.533",
        "1-alpha_1",
    ):
        assert frag in out, frag
    assert "9/23" not in out
    old = _report(D)
    assert (
        "x=7/8" in old and "x=9/23" in old
    )  # and the alpha-0.05 run prints ITS looks, not alpha_1's
