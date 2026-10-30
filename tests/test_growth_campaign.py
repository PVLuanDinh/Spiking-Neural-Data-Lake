"""specs/022 T111 (G3) P4b: the growth campaign's score, verdict and isolation (2.8, 2.12a).

No learner runs here: the arm runners wait on the grown-segment port (owner question #2). These
tests charge what the instrument itself decides -- the failure penalty grid, the restricted score,
ruling 9 vs 9a on a degenerate sample, the 'fewer neurons' licence, preflight, and campaign
isolation (C13's fixture reds).
"""

import sys

import pytest

from snn_data_lake.instruments import growth_campaign as gc

OK = {"solved": True, "future_solved": True, "size": 18}
FAIL = {"solved": False, "future_solved": False, "size": None}


@pytest.mark.parametrize("batch,penalty", [(9, 54), (1, 53), (4, 56), (13, 65)])
def test_failure_penalty_is_the_next_grid_size_above_the_budget(batch, penalty):
    assert gc.failure_penalty(52, batch) == penalty and penalty > 52


def test_failure_penalty_refuses_nonpositive():
    with pytest.raises(ValueError):
        gc.failure_penalty(0, 9)


def test_restricted_score_penalises_every_kind_of_failure():
    assert gc.restricted_score(OK, 52, 9) == 18
    assert gc.restricted_score(FAIL, 52, 9) == 54
    assert (
        gc.restricted_score({**OK, "future_solved": False}, 52, 9) == 54
    )  # validation solve, future fails
    with pytest.raises(ValueError):
        gc.restricted_score({**OK, "size": None}, 52, 9)


def test_ruling_9_and_9a_read_a_degenerate_sample_differently():
    assert not gc.superiority("d", [9.0] * 8).met  # superiority: NOT MET
    assert gc.non_inferiority("e", [9.0] * 8).met  # NI: common value > 0 MET
    assert not gc.non_inferiority("e", [0.0] * 8).met


def _arm(extra, sizes=(18, 27, 18, 27, 18, 27, 18, 27)):
    return [{**OK, "size": s + extra * (i % 3)} for i, s in enumerate(sizes)]


def test_primary_passes_on_a_planted_win_and_licenses_fewer_neurons_only_when_both_solve():
    both = gc.primary(
        {"TARGETED": _arm(0), "PRESIZED_SELECTED": _arm(18), "RANDOM": _arm(27)}, 52, 9
    )
    assert both["pass"] and both["fewer_neurons"]
    # RANDOM never solves: the contrast still passes on the penalty, but it is a solve-rate result
    only_rate = gc.primary(
        {"TARGETED": _arm(0), "PRESIZED_SELECTED": _arm(18), "RANDOM": [FAIL] * 8}, 52, 9
    )
    assert only_rate["pass"] and not only_rate["fewer_neurons"]
    assert only_rate["solve_rates"] == {"TARGETED": 8, "PRESIZED_SELECTED": 8, "RANDOM": 0}


def test_primary_is_not_met_when_either_contrast_fails_or_is_degenerate():
    tied = gc.primary(
        {"TARGETED": _arm(0), "PRESIZED_SELECTED": _arm(0), "RANDOM": _arm(18)}, 52, 9
    )
    assert not tied["pass"] and not tied["D_sel"]["met"]
    uniform = (
        gc.primary(  # every seed wins by exactly one batch: ruling 9, degenerate reads NOT MET
            {
                "TARGETED": _arm(0),
                "PRESIZED_SELECTED": [{**OK, "size": o["size"] + 9} for o in _arm(0)],
                "RANDOM": [{**OK, "size": o["size"] + 9} for o in _arm(0)],
            },
            52,
            9,
        )
    )
    assert uniform["D_sel"]["contrast"].degenerate and not uniform["pass"]


def test_primary_below_the_floor_is_not_met():
    small = gc.primary(
        {"TARGETED": _arm(0), "PRESIZED_SELECTED": _arm(4), "RANDOM": _arm(4)}, 52, 9
    )
    assert not small["pass"]  # mean difference under one batch of 9


def test_primary_refuses_unpaired_arms():
    with pytest.raises(ValueError):
        gc.primary(
            {"TARGETED": _arm(0), "PRESIZED_SELECTED": _arm(0)[:7], "RANDOM": _arm(0)}, 52, 9
        )


@pytest.mark.parametrize("n", [0, 7])
def test_primary_refuses_any_seed_count_but_the_registered_eight_by_name(n):
    # the read uses the 7-d.f. bar (paired_contrast.T_BAR), so any other n reads at the wrong d.f.
    with pytest.raises(ValueError, match="exactly 8 paired seeds"):
        gc.primary(
            {"TARGETED": _arm(0)[:n], "PRESIZED_SELECTED": _arm(9)[:n], "RANDOM": _arm(9)[:n]},
            52,
            9,
        )


# ---- preflight
def test_preflight_refuses_foreign_partial_and_ungated_confirmatory_seeds():
    for bad in ([], [9000, 9000], [123], [8000], [9100], [9100, 9101], [9300, 9301]):
        with pytest.raises(PermissionError):
            gc.preflight(bad)
    with pytest.raises(PermissionError):  # the exact block still needs the written go
        gc.preflight(list(gc.CONFIRMATORY_SEEDS))
    gc.preflight(list(gc.CONFIRMATORY_SEEDS), go="lead, written")
    gc.preflight(list(gc.DEV_SEEDS[:3]))
    gc.preflight(list(gc.RED_SEEDS[:3]))


@pytest.mark.parametrize("go", [False, "", 0, 1, ["written"]])
def test_preflight_requires_the_go_to_be_a_non_empty_string(go):
    with pytest.raises(PermissionError):
        gc.preflight(list(gc.CONFIRMATORY_SEEDS), go=go)


@pytest.mark.parametrize("gap", [9116, 9199, 9216, 9299, 9316, 9500, 9899])
def test_preflight_refuses_seeds_in_the_unregistered_gaps_of_g3s_block(gap):
    with pytest.raises(PermissionError):
        gc.preflight([gap])


def test_protected_seed_blocks_sit_outside_g3s_block():
    assert not set(gc.FAMILIES[0]) & {
        *range(300, 362),
        6000,
        6001,
        *range(6100, 6108),
        *range(7000, 7008),
    }
    assert all(gc.BLOCK[0] <= s < gc.BLOCK[1] for fam in gc.FAMILIES for s in fam)


# ---- campaign isolation (2.12a, C13's fixture reds)
def test_each_arm_gets_its_own_fresh_home(tmp_path):
    a = gc.arm_home(tmp_path, "primary", 9000, "TARGETED")
    b = gc.arm_home(tmp_path, "primary", 9000, "RANDOM")
    assert a != b and a.is_dir() and b.is_dir()
    with pytest.raises(FileExistsError):
        gc.arm_home(tmp_path, "primary", 9000, "TARGETED")
    with pytest.raises(ValueError):
        gc.arm_home(tmp_path, "primary", 9000, "NOPE")
    with pytest.raises(PermissionError):
        gc.arm_home(tmp_path, "primary", 123, "TARGETED")


def test_a_home_under_the_live_lake_is_refused(tmp_path, monkeypatch):
    from snn_data_lake.substrate import home

    monkeypatch.setattr(home, "_default_live_home", lambda: str(tmp_path / "live"))
    (tmp_path / "live").mkdir()
    with pytest.raises(home.LiveHomeRefused):
        gc.arm_home(tmp_path / "live", "primary", 9000, "TARGETED")


def test_arms_with_different_ledger_snapshots_refuse_to_pair(tmp_path):
    f1, f2 = tmp_path / "a.bin", tmp_path / "b.bin"
    f1.write_bytes(b"ledger")
    f2.write_bytes(b"ledger")
    same = gc.pair_snapshots([gc.snapshot_digest(f1), gc.snapshot_digest(f2)])
    assert same == gc.snapshot_digest(f1)
    f2.write_bytes(b"ledger+1")  # the red: one arm inherited another's spend
    with pytest.raises(PermissionError):
        gc.pair_snapshots([gc.snapshot_digest(f1), gc.snapshot_digest(f2)])
    with pytest.raises(PermissionError):
        gc.pair_snapshots([])


def test_child_without_its_home_refuses_and_a_matching_home_starts(tmp_path):
    with pytest.raises(PermissionError):
        gc.child_home(["x"], {"NORD_HOME": str(tmp_path)})  # no --home
    with pytest.raises(PermissionError):
        gc.child_home(["x", "--home", str(tmp_path)], {})  # no NORD_HOME
    with pytest.raises(PermissionError):
        gc.child_home(["x", "--home", str(tmp_path)], {"NORD_HOME": str(tmp_path / "other")})
    assert gc.child_home(["x", "--home", str(tmp_path)], {"NORD_HOME": str(tmp_path)}) == tmp_path


def test_child_home_refuses_a_home_holding_an_alias_to_the_live_lake(tmp_path, monkeypatch):
    # home.require_isolated_home scans every entry; assert_not_live_home alone does not
    import os

    from snn_data_lake.substrate import home

    live = tmp_path / "live"
    live.mkdir()
    (live / "brain_lake.bin").write_bytes(b"lake")
    monkeypatch.setattr(home, "_default_live_home", lambda: str(live))
    iso = tmp_path / "arm"
    iso.mkdir()
    try:
        os.symlink(live / "brain_lake.bin", iso / "brain_lake.bin")
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")
    with pytest.raises(home.LiveHomeRefused):
        gc.child_home(["x", "--home", str(iso)], {"NORD_HOME": str(iso)})


def test_launch_hands_the_home_as_argument_and_environment(tmp_path):
    code = (
        "import sys;from snn_data_lake.instruments import growth_campaign as gc;"
        "print(gc.child_home(sys.argv[1:]))"
    )
    done = gc.launch([sys.executable, "-c", code], tmp_path)
    assert done.returncode == 0 and str(tmp_path) in done.stdout, done.stderr
    bare = gc.subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert (
        bare.returncode != 0 and "PermissionError" in bare.stderr
    )  # a child launched bare refuses


def _listing(path):
    return sorted(str(p) for p in path.rglob("*"))


@pytest.mark.parametrize("sub", [(), ("newsub", "deeper")])
def test_a_refused_arm_home_creates_nothing_under_the_live_lake(tmp_path, monkeypatch, sub):
    # G7's probe: the guard ran AFTER mkdir(parents=True), so a refusal left directories behind
    from snn_data_lake.substrate import home

    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(home, "_default_live_home", lambda: str(live))
    before = _listing(live)
    with pytest.raises(home.LiveHomeRefused):
        gc.arm_home(live.joinpath(*sub), "primary", 9000, "TARGETED")
    assert _listing(live) == before


def test_arm_home_refuses_a_name_that_could_leave_the_root(tmp_path):
    before = _listing(tmp_path)  # the conftest may already hold a home here
    for bad in ("../x", "a/b", "a\\b", "C:x", ""):
        with pytest.raises(ValueError):
            gc.arm_home(tmp_path, bad, 9000, "TARGETED")
    assert _listing(tmp_path) == before
