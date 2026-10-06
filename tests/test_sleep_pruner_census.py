"""G6 / T114 part (a): the sleep-pruner census and the per-pruner A/B (checks C1-C6, C25).

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md (2.3, 2.11, 2.16).
Brain-backed cases build ONE short scratch brain per arm under throwaway_home (never the live lake).
"""

import os

import pytest

from snn_data_lake.instruments import sleep_pruner_census as spc

LIVE_HOME = os.path.join(os.path.expanduser("~"), ".snn_data_lake")


def test_self_check_runs_brain_free():
    spc.main()


def test_static_census_regenerates_section_one_table():
    cen = spc.census_static()
    assert cen.found() == sorted(spc.EXPECTED) and cen.missing() == []
    assert cen.unclassified() == []
    by = {(s.cls, s.method): s for s in cen.sites}
    assert by[("AttractorNet", "downscale_prune")].host == "_sleep_tick"
    assert by[("NGSNN", "prune_unearned")].module == "growth.neurogenesis_engine"
    assert by[("LifelongConsolidator", "_prune_to_budget")].host == "_enter_sleep"


def test_c1_a_planted_prune_call_appears_and_a_deleted_site_reds():
    add = spc.census_static(sources=spc.planted_sources("add"))
    assert [s for s in add.unclassified() if s.method == "evict_weakest"], (
        "C1: the planted call must appear"
    )
    gone = spc.census_static(sources=spc.planted_sources("delete"))
    assert gone.missing() == ["P-DP"], "C1: removing downscale_prune from the walk must red"


def test_c2_each_removal_extractor_reads_a_planted_count():
    assert spc._removed_dict({"pruned": 3}, None, None) == 3
    assert spc._removed_int(2, None, None) == 2
    assert spc._removed_units(None, 12, 6) == 6
    assert spc._removed_lake((40, 7), None, None) == 7
    assert spc._removed_lake(["a"], 4, None) == 3


def test_c3_dp_sabotage_removes_the_same_count_strongest_first():
    from snn_data_lake.sleep.sleep_replay import AttractorNet

    net = AttractorNet(5)
    strongest = sorted(
        ((net.W[r][c], r, c) for r in range(len(net.W)) for c in range(len(net.W)) if r != c),
        reverse=True,
    )[:4]
    run = spc.ArmRun(seed=0, arm="dp_sabotage", ticks=0)
    out = spc._dp_sabotage(run, [4]).__get__(net)()
    assert out["pruned"] == 4 and run.sabotage_shortfall == 0
    assert all(net.W[r][c] == 0.0 for _, r, c in strongest)


def test_c3_ey_sabotage_matches_the_count_and_never_empties():
    import torch

    from snn_data_lake.growth.ng_snn import NGSNN, _make_task

    xtr, ytr, _, _ = _make_task("cpu", seed=0)
    m = NGSNN(seed=0, device="cpu")
    m.fit(xtr, ytr)
    L = int(m.n_units)
    assert L >= 2
    run = spc.ArmRun(seed=0, arm="ey_sabotage", ticks=0)
    sab = spc._ey_sabotage(run, [1, L + 3])
    assert sab(m, xtr, ytr) == 1 and int(m.n_units) == L - 1 and len(m.last_keep) == L - 1
    assert sab(m, xtr, ytr) == L - 2 and int(m.n_units) == 1  # capped: one unit always survives
    assert run.sabotage_shortfall == (L + 3) - (L - 2)
    assert isinstance(m.W, torch.Tensor)


def test_c3_ll_sabotage_drops_the_highest_utility_units():
    from snn_data_lake.sleep.lifelong_consolidation import D, LifelongConsolidator, _make_tasks

    tasks = _make_tasks(seed=0)
    ll = LifelongConsolidator(dim=D, budget=3, seed=0)
    ll.learn_task(0, tasks[0]["train"])
    n = ll.n_units
    assert n >= 2
    ll.budget = n - 1
    top = sorted(range(n), key=lambda i: (-ll.utility[i], i))[0]
    expect = [p for i, p in enumerate(ll.learner.prototypes) if i != top]
    expect_u = [u for i, u in enumerate(ll.utility) if i != top]
    spc._ll_top_utility(ll)
    assert ll.n_units == n - 1 and ll.learner.prototypes == expect and ll.utility == expect_u


def test_scratch_home_guard_refuses_the_live_home(monkeypatch, tmp_path):
    """The shared guard via a PLANTED decoy live dir: unset, equal, contained, containing and a
    junction alias are each refused; a fresh sibling dir is accepted (the red is sighted)."""
    import subprocess

    from snn_data_lake.substrate.home import LiveHomeRefused

    decoy = tmp_path / "decoy_live"
    (decoy / "sub").mkdir(parents=True)
    alias = tmp_path / "selfcheck_home_alias"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(decoy)], capture_output=True)
    if r.returncode != 0:
        os.symlink(decoy, alias, target_is_directory=True)
    ok = tmp_path / "scratch"
    ok.mkdir()
    monkeypatch.delenv("NORD_HOME", raising=False)
    with pytest.raises(LiveHomeRefused):
        spc._assert_scratch_home(live_home=decoy)
    for bad in (decoy, decoy / "sub", alias, tmp_path):
        monkeypatch.setenv("NORD_HOME", str(bad))
        with pytest.raises(LiveHomeRefused):
            spc._assert_scratch_home(live_home=decoy)
    monkeypatch.setenv("NORD_HOME", str(ok))
    assert spc._assert_scratch_home(live_home=decoy) == os.path.realpath(ok)
    monkeypatch.setenv("NORD_HOME", LIVE_HOME)
    with pytest.raises(LiveHomeRefused):
        spc._assert_scratch_home()


def test_outcome_windows_refuse_short_or_late_runs():
    """A2 + the amendment 2026-10-04: night k = the k-th `_enter_wake` reading, from 1;
    window 2..36; REFUSED with fewer than 36 readings or no stored pattern at k = 2."""
    r = spc.ArmRun(seed=0, arm="live", ticks=900)
    r.nights = [[k, k >= 2, 12] for k in range(1, 37)]  # 12001's shape: unstored at k = 1 is fine
    assert spc.outcome(r, "P-DP") == (12 * 35, 1260, None)
    r.nights = [[k, k >= 3, 12] for k in range(1, 37)]
    assert "first window night" in spc.outcome(r, "P-DP")[2]
    r.nights = [[k, True, 12] for k in range(1, 36)]
    assert "ends before" in spc.outcome(r, "P-DP")[2]
    r.refits, r.expected_refits = [[20, 8, 4], [40, 12, 6]], 3
    assert "scheduled" in spc.outcome(r, "P-EY")[2]
    r.expected_refits = 2
    assert spc.outcome(r, "P-EY") == (10, 20, None)
    r.ll_learned, r.ll = 2, [[0, 40, 40], [1, 30, 40]]
    assert "of 3 tasks" in spc.outcome(r, "P-LL")[2]
    assert spc.DELTA["P-EY"] == 1.0 / 1111.0 and spc.DELTA["P-DP"] == 1.0 / 1260.0


def test_the_live_arm_records_one_based_readings(short_runs):
    ks = [n[0] for n in short_runs["live"].nights]
    assert ks == list(range(1, len(ks) + 1))


# ---- brain-backed: ONE short scratch brain per arm (throwaway home; never the live lake) ----
SHORT = 60


@pytest.fixture(scope="module")
def short_runs():
    from snn_data_lake.substrate.home import throwaway_home

    before = _live_mtime()
    runs = {}
    with throwaway_home():
        runs["live"] = spc.run_arm(12001, "live", ticks=SHORT)
    with throwaway_home():
        runs["live2"] = spc.run_arm(12001, "live", ticks=SHORT)
    with throwaway_home():
        runs["dp_neuter"] = spc.run_arm(12001, "dp_neuter", ticks=SHORT)
    assert _live_mtime() == before, "the live lake must not change"
    return runs


def _live_mtime():
    try:
        return max(os.path.getmtime(os.path.join(LIVE_HOME, f)) for f in os.listdir(LIVE_HOME))
    except (FileNotFoundError, ValueError):
        return None


def test_c6_the_live_arm_is_byte_repeatable(short_runs):
    assert short_runs["live"].digest == short_runs["live2"].digest


def test_c5_the_dp_neuter_touches_no_other_pruner(short_runs):
    ok, other = spc.surgical_ok(short_runs["live"], short_runs["dp_neuter"], "P-DP")
    assert ok, f"C5: neutering P-DP changed {other}'s activity series"
    assert "P-DP" not in short_runs["dp_neuter"].removals  # the neutered site never ran


def test_runtime_census_counts_and_discovers_no_miss():
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        spc._assert_scratch_home()
        b = make_brain(seed=12001)
        cen = spc.census_static(runtime_types=spc.runtime_types_of(b))
        seen = set()
        with spc.discovering(seen):
            rec = spc.census_runtime(b, SHORT)
    assert spc.census_misses(cen, seen) == [], (
        "sleep called a prune-vocabulary method the census missed"
    )
    assert rec.calls("P-DP") > 0 and rec.calls("P-SR") > 0, "the short run must reach sleep ticks"
    assert ("AttractorNet", "downscale_prune") in seen and ("StructuralRewiring", "rewire") in seen


def test_the_discovery_net_reds_on_a_planted_unclassified_pruner():
    """The runtime net's own red: a prune-vocabulary package function called inside a sleep host
    that the static census never classified must surface as a miss."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    ns = {"__name__": "snn_data_lake.planted_g6"}
    exec("def evict_planted():\n    return 1\n", ns)  # noqa: S102 -- a constructed package-scoped callee
    with throwaway_home():
        spc._assert_scratch_home()
        b = make_brain(seed=12001)
        cen = spc.census_static()
        real = b.attractor.downscale_prune

        def planted(*a, **k):
            ns["evict_planted"]()
            return real(*a, **k)

        b.attractor.downscale_prune = planted
        seen = set()
        with spc.discovering(seen):
            spc.census_runtime(b, 40)
    assert ("?", "evict_planted") in spc.census_misses(cen, seen)


def test_c2_a_planted_pruner_removing_k_items_reads_k(monkeypatch):
    """C2 through counting_patches, not only the extractors on planted return values."""
    import types

    class Planted:
        def prune(self, k):
            return k

    monkeypatch.setattr(spc, "_import", lambda mod: types.SimpleNamespace(Planted=Planted))
    rec = spc.Recorder()
    sites = {("x", "Planted", "prune"): ("P-DP", spc._removed_int, None)}
    with spc._patched(spc.counting_patches(rec, sites=sites)):
        Planted().prune(3)
        Planted().prune(4)
    assert rec.removals("P-DP") == 7 and rec.calls("P-DP") == 2


def test_c5_a_planted_cross_pruner_change_is_caught():
    """C5 red: a neuter changing ANOTHER pruner's series before its first removal."""
    live = spc.ArmRun(seed=0, arm="live", ticks=9)
    live.removals = {"P-DP": [[5, 2]], "P-EY": [[1, 1]]}
    twin = spc.ArmRun(seed=0, arm="dp_neuter", ticks=9)
    twin.removals = {"P-EY": [[1, 1]]}
    assert spc.surgical_ok(live, twin, "P-DP") == (True, None)
    twin.removals = {"P-EY": [[1, 9]]}
    assert spc.surgical_ok(live, twin, "P-DP") == (False, "P-EY")


def test_c6_the_digest_moves_when_the_run_moves():
    a = spc.ArmRun(seed=0, arm="live", ticks=9, nights=[[2, True, 30]]).finish()
    b = spc.ArmRun(seed=0, arm="live", ticks=9, nights=[[2, True, 31]]).finish()
    assert (
        a.digest != b.digest
        and a.digest
        == spc.ArmRun(seed=0, arm="live", ticks=9, nights=[[2, True, 30]]).finish().digest
    )


def test_out_target_inside_or_aliasing_the_live_home_is_refused(tmp_path):
    """BLOCKER (Codex panel): `--out` was opened for write unguarded, after throwaway_home."""
    import subprocess

    from snn_data_lake.substrate.home import LiveHomeRefused

    decoy = tmp_path / "a" / "decoy_live"
    decoy.mkdir(parents=True)
    (decoy / "brain_lake.bin").write_bytes(b"x")
    link = tmp_path / "a" / "selfcheck_home_link"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(decoy)], capture_output=True)
    if r.returncode != 0:
        os.symlink(decoy, link, target_is_directory=True)
    for bad in (
        decoy / "brain_lake.bin",
        decoy / "new.json",
        link / "brain_lake.bin",
        link / "new.json",
    ):
        with pytest.raises(LiveHomeRefused):
            spc._guard_out(str(bad), live_home=decoy)
    (tmp_path / "b").mkdir()
    ok = tmp_path / "b" / "out.json"
    assert spc._guard_out(str(ok), live_home=decoy) == os.path.realpath(ok)


def test_cli_refuses_an_out_in_the_live_home_before_any_brain_runs():
    """The guard fires in `_cli` itself (no brain is built, nothing is written)."""
    from snn_data_lake.substrate.home import LiveHomeRefused

    target = os.path.join(LIVE_HOME, "g6_out_guard_probe.json")
    with pytest.raises(LiveHomeRefused):
        spc._cli(["--arm", "12001", "live", "--out", target])
    assert not os.path.exists(target)
