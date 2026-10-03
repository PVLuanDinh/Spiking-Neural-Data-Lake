"""G1(b)'s campaign (instruments/pool_selection.py): section 2.5's ordering and eligibility on
hand-built tables, section 2.6's power rule, R-b7's refusals, the step-order refusal and section
2.8's taxonomy and exit codes (g1b-pool-selection.md, frozen at 23c7e97).

Nothing here runs a campaign: steps run on a scratch OUT at a stand-in revision, and their jobs go
through stand-in runners (or one in-process replay of the 48-event fixture)."""

import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.instruments.paired_contrast import T_BAR_S31
from snn_data_lake.reservoir import recurrent_pool as rp


def entry(order, d, d_task, cost):
    return {"order": order, "D": d, "D_task": d_task, "cost": cost}


def clean():
    return "h1", False


def write_summary(out, step, status="COMPLETED", head="h1", dirty=False):
    out.mkdir(parents=True, exist_ok=True)
    md._save(
        ps._summary_path(out, step), {"step": step, "status": status, "head": head, "dirty": dirty}
    )


def summary(out, step):
    return md._load(ps._summary_path(out, step))


# ---- section 2.5: the ordering -------------------------------------------------------------------


def test_band_d_is_the_larger_of_the_floor_and_sigma_dev():
    assert ps.band_d(0.001) == 0.005 and ps.band_d(0.0123) == 0.0123
    best = entry(0, 0.030, 0.0, 1.0)
    near, far = entry(1, 0.026, 0.9, 1.0), entry(2, 0.020, 0.9, 1.0)
    assert ps.ordering([best, near], ps.band_d(0.001))[0] is near  # 0.004 below: near-tied
    assert ps.ordering([best, far], ps.band_d(0.001))[0] is best  # 0.010 below: outside 0.005
    assert ps.ordering([best, far], ps.band_d(0.012))[0] is far  # inside a sigma_dev of 0.012


def test_d_task_within_0_005_is_still_tied_and_the_cost_decides():
    dear = entry(0, 0.03, 0.200, 9.0)
    assert ps.ordering([dear, entry(1, 0.03, 0.196, 3.0)], 0.005)[0]["order"] == 1  # tied: cheaper
    assert ps.ordering([dear, entry(1, 0.03, 0.194, 3.0)], 0.005)[0] is dear  # 0.006: D_task


def test_costs_within_ten_percent_are_tied_and_order_decides():
    cheap = entry(5, 0.03, 0.2, 10.0)
    assert ps.ordering([entry(0, 0.03, 0.2, 10.9), cheap], 0.005)[0]["order"] == 0  # 9% dearer
    assert ps.ordering([entry(0, 0.03, 0.2, 11.2), cheap], 0.005)[0] is cheap  # 12% dearer


def test_exact_ties_go_to_manifest_order_or_the_lower_index():
    tied = [entry(i, 0.01, 0.1, 2.0) for i in (7, 2, 4)]
    assert [e["order"] for e in ps.ordering(tied, 0.005)] == [2, 4, 7]
    # selection's order is section 2.3's candidate index (hand-built here: g4, g2, g3 were 4, 2, 3)
    selection = [entry(i, 0.01, 0.1, 2.0) for i in (4, 2, 3)]
    assert ps.ordering(selection, 0.005)[0]["order"] == 2
    assert ps.ordering([], 0.005) == []


def test_the_ordering_places_every_entry_once():
    rows = [(0.03, 0.1, 5.0), (0.01, 0.3, 1.0), (0.029, 0.2, 2.0), (0.02, 0.0, 1.0)]
    ranked = ps.ordering([entry(i, *row) for i, row in enumerate(rows)], 0.005)
    # 0.029 is near-tied with 0.03 and wins on D_task; then 0.03 alone; then 0.02; then 0.01
    assert [e["order"] for e in ranked] == [2, 0, 3, 1]


# ---- section 2.5: eligibility, degeneracy, the statistic ----------------------------------------


def test_eligibility_non_inferiority_degeneracy_and_completion():
    row = {"ok": True, "carried_live": [0.70, 0.80, 0.90, 0.80], "degenerate": [False] * 4}
    assert ps.eligibility(row, 0.84) == (True, [])  # 0.80 >= 0.79
    assert ps.eligibility(row, 0.86) == (False, ["inferior to (e1-0) by more than 0.05"])
    assert ps.eligibility(dict(row, degenerate=[False, False, True, False]), 0.5) == (
        False,
        ["Carried phi degenerate on a live recording"],
    )
    assert ps.eligibility(dict(row, ok=False), 0.5) == (
        False,
        ["a job it needs did not complete OK"],
    )
    assert not ps.eligibility(row, None)[0]  # no (e1-0) reference: nothing is non-inferior


def test_degeneracy_rule():
    assert ps.degenerate({"rows": 944, "constant_fraction": 0.90, "distinct_rows": 50})
    assert not ps.degenerate({"rows": 944, "constant_fraction": 0.89, "distinct_rows": 2})
    assert ps.degenerate({"rows": 944, "constant_fraction": 0.1, "distinct_rows": 1})
    assert ps.degenerate({"rows": 0, "constant_fraction": None, "distinct_rows": 0})
    assert ps.degenerate(None)


def test_recording_accuracy_is_the_mean_of_available_segments_and_halts_on_none():
    scores = {
        "warm-up": {"n": 5, "available": False},
        "repeated-canonical": {"available": True, "accuracy": 0.5},
        "unseen-order": {"available": True, "accuracy": 1.0},
    }
    assert ps.recording_accuracy(scores) == 0.75
    with pytest.raises(ps.Halt, match="scored segments"):
        ps.recording_accuracy({"warm-up": {"n": 5, "available": False}})


# ---- section 2.6: the power rule ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("sigma", "delta", "expected"),
    [
        (0.02, 0.01, 27),
        (0.02, 0.02, 8),
        (0.02, 0.03, 8),
        (0.0436, 0.01, None),
        (0.0436, 0.02, 31),
        (0.0436, 0.03, 15),
        (0.08, 0.01, None),
        (0.08, 0.02, None),
        (0.08, 0.03, 46),
    ],
)
def test_power_rule_reproduces_section_2_6(sigma, delta, expected):
    result = ps.power_s(sigma, delta)
    assert result == (
        {"S": 64, "capped": True} if expected is None else {"S": expected, "capped": False}
    )


def test_t_quantiles_match_the_printed_tables():
    # Student's t quantiles to 7 places (scipy.stats.t.ppf agreed to 2e-14 on d.f. 1-69 at build)
    assert abs(ps.t_quantile(0.95, 7) - 1.8945786) < 1e-7  # the plan's 1.895 bar at 7 d.f.
    assert abs(ps.t_quantile(0.95, 30) - 1.6972609) < 1e-7
    assert abs(ps.t_quantile(0.80, 7) - 0.8960296) < 1e-7
    assert abs(ps.t_quantile(0.80, 30) - 0.8537673) < 1e-7
    assert abs(ps.t_quantile(0.975, 10) - 2.2281389) < 1e-7
    assert ps._t_cdf(0.0, 5) == 0.5 and abs(ps._t_cdf(1.0, 1) - 0.75) < 1e-12
    assert abs(ps._t_cdf(-1.5, 6) + ps._t_cdf(1.5, 6) - 1.0) < 1e-15


# ---- G1(c) P1-B: the S = 31 bar tied to the quantile (g1c-carried-state.md 2.3, 2.10) -----------


def bar_ties_to_quantile(bar, df):
    """The bar is the fourth-place round-UP of t_quantile(0.95, df), at most 1e-4 above it."""
    lo = ps.t_quantile(0.95, df)
    return lo <= bar < lo + 1e-4 and math.ceil(lo * 1e4) / 1e4 == bar


def test_the_s31_bar_is_the_fourth_place_round_up_of_the_30_df_quantile():
    assert bar_ties_to_quantile(T_BAR_S31, 30)
    assert ps.power_s(ps.SIGMA_PROXY)["S"] - 1 == 30  # S = 31 seeds is 30 d.f.


@pytest.mark.parametrize(
    ("bar", "df"),
    [(1.895, 30), (T_BAR_S31, 7), (T_BAR_S31, 29), (T_BAR_S31, 31)],
    ids=["the 7 d.f. bar at 30", "S31 at 7", "S31 at 29", "S31 at 31"],
)
def test_the_bar_tie_reds_fire_on_a_wrong_bar_or_a_wrong_df(bar, df):
    assert not bar_ties_to_quantile(bar, df)


# ---- R-b7: the binding and the table -------------------------------------------------------------


def pool_cell(seed, *, bound=None, status="OK"):
    config = rp.grid("lsm")[0]
    filed = ps._bound(seed, ps._spec("lsm", config, "carried"))
    binding = {
        "candidate": "lsm",
        "form": "carried",
        "config_crc32": rp.crc_json(bound or config),
        "realization_seed": rp.realization_seed("lsm", seed),
        "window": rp.WINDOW,
        "noise": False,
        "w_in": ps._w_in_crc(seed),
    }
    digest = {"arm": "pool", "seed": seed, "pool": binding} if status == "OK" else None
    return dict(filed, index=0, status=status, digest=digest)


def test_r_b7_refuses_its_reds_and_passes_the_well_formed_tables():
    reds = ps.r_b7_reds()
    assert reds["verdict"] and reds["control"]
    assert reds["red"] == {
        "wrong configuration": True,
        "duplicate seed": True,
        "tuning seed in selection": True,
        "window-0 replay filed as window 256": True,
        "hosted replay filed under a port cell": True,
        "port replay filed under a hosted cell": True,
        "hosted binding without its policy": True,
    }


def test_build_table_refusals():
    assert len(ps.build_table([pool_cell(seed) for seed in ps.TUNE_SEEDS], ps.TUNE_SEEDS)) == 4
    with pytest.raises(ps.Halt, match="R-b7") as wrong:
        ps.build_table([pool_cell(200, bound=rp.grid("lsm")[1])], ps.TUNE_SEEDS)
    assert "config_crc32" in wrong.value.detail
    with pytest.raises(ps.Halt, match="R-b7") as duplicate:
        ps.build_table([pool_cell(200), pool_cell(200)], ps.TUNE_SEEDS)
    assert "duplicate seed 200" in duplicate.value.detail
    with pytest.raises(ps.Halt, match="R-b7") as tuning:
        ps.build_table([pool_cell(200)], ps.SELECT_SEEDS + ps.SELECT_TASK_SEEDS)
    assert "not a seed of this phase" in tuning.value.detail
    # a TIMEOUT cell is filed without a binding (its configuration is only made ineligible)
    assert len(ps.build_table([pool_cell(200, status="TIMEOUT")], ps.TUNE_SEEDS)) == 1


def test_binding_problems_name_each_disagreement():
    cell = pool_cell(201)
    assert ps.binding_problems(cell) == []
    assert ps.binding_problems(dict(cell, digest=dict(cell["digest"], seed=202))) == ["seed"]
    short = dict(cell["digest"]["pool"], window=16)  # the self-checks' shortened window
    assert ps.binding_problems(dict(cell, digest=dict(cell["digest"], pool=short))) == ["window"]
    noisy = dict(cell["digest"]["pool"], noise=True)  # R-b4's red filed as a clean replay
    assert ps.binding_problems(dict(cell, digest=dict(cell["digest"], pool=noisy))) == ["noise"]
    reference = {
        "arm": "e1-0",
        "seed": 201,
        "candidate": None,
        "config": None,
        "form": None,
        "noise": False,
        "digest": {"arm": "e1-0", "seed": 201, "pool": None},
    }
    assert ps.binding_problems(reference) == []
    assert ps.binding_problems(dict(reference, digest=dict(reference["digest"], arm="e1"))) == [
        "arm"
    ]


# ---- G1(c) P3a: the window in the filing, the key and the spec -----------------------------------


def window_cell(seed, ran_at, filed_as=None):
    """A Settled cell whose replay ran at window `ran_at`, filed under window `filed_as` (None: the
    default, which files no window at all)."""
    config = rp.grid("lsm")[0]
    spec = ps._spec("lsm", config, "settled", window=rp.WINDOW if filed_as is None else filed_as)
    binding = {
        "candidate": "lsm",
        "form": "settled",
        "config_crc32": rp.crc_json(config),
        "realization_seed": rp.realization_seed("lsm", seed),
        "window": ran_at,
        "noise": False,
        "w_in": ps._w_in_crc(seed),
    }
    digest = {"arm": "pool", "seed": seed, "pool": binding}
    return dict(ps._bound(seed, spec), index=0, status="OK", digest=digest)


def test_a_window_zero_replay_filed_as_window_256_is_refused_and_filed_as_zero_passes():
    zero = window_cell(204, ran_at=0)  # run at window 0, filed under the default window
    assert ps.binding_problems(zero) == ["window"]
    with pytest.raises(ps.Halt, match="R-b7") as refused:
        ps.build_table([zero], ps.SELECT_SEEDS)
    assert "['window']" in refused.value.detail
    filed = window_cell(204, ran_at=0, filed_as=0)
    assert ps.binding_problems(filed) == [] and len(ps.build_table([filed], ps.SELECT_SEEDS)) == 1


@pytest.mark.parametrize(
    ("ran_at", "filed_as", "problems"),
    [
        (16, None, ["window"]),  # the short-window red: the self-checks' 16, filed as 256
        (rp.WINDOW, 0, ["window"]),  # a default-window replay filed as window 0
        (0, 16, ["window"]),  # window 0 filed as some other window
        (16, 16, []),
        (rp.WINDOW, rp.WINDOW, []),
        (rp.WINDOW, None, []),
    ],
)
def test_the_binding_holds_a_replay_to_the_window_it_is_filed_under(ran_at, filed_as, problems):
    assert ps.binding_problems(window_cell(204, ran_at, filed_as)) == problems


def test_the_window_is_in_the_cell_key_only_off_the_default():
    config = rp.grid("lsm")[0]
    default = window_cell(204, rp.WINDOW)
    key = ps.cell_key(default)
    assert key == ("pool", "lsm", rp.crc_json(config), "settled", 204, False) and len(key) == 6
    # naming the default window files nothing
    assert ps.cell_key(window_cell(204, rp.WINDOW, rp.WINDOW)) == key
    zero = window_cell(204, 0, 0)
    assert ps.cell_key(zero) == (*key, 0)
    # Settled@256 and Settled@0 of one seed are two cells; the same window twice is a duplicate
    assert len(ps.build_table([default, zero], ps.SELECT_SEEDS)) == 2
    with pytest.raises(ps.Halt, match="R-b7") as duplicate:
        ps.build_table([zero, window_cell(204, 0, 0)], ps.SELECT_SEEDS)
    assert "duplicate seed 204" in duplicate.value.detail


def test_spec_and_bound_carry_a_window_only_off_the_default():
    config = rp.grid("lsm")[0]
    plain = {"candidate": "lsm", "config": config, "form": "carried"}
    assert ps._spec("lsm", config, "carried") == plain
    assert list(ps._spec("lsm", config, "carried")) == ["candidate", "config", "form"]
    assert ps._spec("lsm", config, "carried", window=rp.WINDOW) == plain
    assert ps._spec("lsm", config, "carried", noise=True) == dict(plain, noise=True)
    zero = ps._spec("lsm", config, "settled", window=0)
    assert zero == {"candidate": "lsm", "config": config, "form": "settled", "window": 0}
    assert ps._spec("lsm", config, "carried", True, 0) == dict(plain, noise=True, window=0)
    bound = ps._bound(204, ps._spec("lsm", config, "carried"))
    assert bound == {"arm": "pool", "seed": 204, **plain, "noise": False}
    assert list(bound) == ["arm", "seed", "candidate", "config", "form", "noise"]
    settled = ps._bound(204, ps._spec("lsm", config, "settled"))
    assert ps._bound(204, zero) == dict(settled, window=0)
    assert ps._bound(204, dict(plain, window=rp.WINDOW)) == bound  # a named default files nothing


# the sha256 of g1_surface() computed at the G1(b) parent (7cd4c604), before the window existed
G1B_SURFACE_SHA256 = "ea54ab059b817620aad56b8f33b6d502344920c604f8086cc133366242edc35b"


def g1_surface(seed=204):
    """Every G1(b) spec, filing, cell key and binding verdict over the whole grid, every form and
    both noise settings, as one text in dict order (so key order is part of it)."""
    rows = []
    for config in rp.grid("lsm"):
        for form in rp.FORMS:
            for noise in (False, True):
                spec = ps._spec("lsm", config, form, noise=noise)
                bound = ps._bound(seed, spec)
                binding = {
                    "candidate": "lsm",
                    "form": form,
                    "config_crc32": rp.crc_json(config),
                    "realization_seed": rp.realization_seed("lsm", seed),
                    "window": rp.WINDOW,
                    "noise": noise,
                    "w_in": ps._w_in_crc(seed),
                }
                cell = dict(bound, digest={"arm": "pool", "seed": seed, "pool": binding})
                rows.append([spec, bound, list(ps.cell_key(cell)), ps.binding_problems(cell)])
    return json.dumps(rows, separators=(",", ":"))


def test_g1b_keys_specs_and_bindings_are_byte_identical_at_the_default():
    text = g1_surface()
    assert '"window"' not in text  # no G1(b) spec or filing names the default window
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == G1B_SURFACE_SHA256


# ---- G1(c) P4b: the host in the filing, the key, the spec and the binding ------------------------


def host_cell(seed, ran_on="port", filed_as="port", policy=True):
    """A Settled cell whose replay ran on host `ran_on`, filed under host `filed_as` (the port files
    no host at all); a hosted binding carries "host" and, if `policy`, "host_policy"."""
    config = rp.grid("lsm")[0]
    spec = ps._spec("lsm", config, "settled", host=filed_as)
    binding = {
        "candidate": "lsm",
        "form": "settled",
        "config_crc32": rp.crc_json(config),
        "realization_seed": rp.realization_seed("lsm", seed),
        "window": rp.WINDOW,
        "noise": False,
        "w_in": ps._w_in_crc(seed),
    }
    if ran_on != "port":
        binding = dict(binding, host=ran_on)
        if policy:
            binding["host_policy"] = {"host": ran_on, "frac_bits": 12}
    digest = {"arm": "pool", "seed": seed, "pool": binding}
    return dict(ps._bound(seed, spec), index=0, status="OK", digest=digest)


def test_the_host_is_in_the_spec_the_filing_and_the_key_only_off_the_port():
    config = rp.grid("lsm")[0]
    plain = {"candidate": "lsm", "config": config, "form": "carried"}
    assert ps._spec("lsm", config, "carried", host="port") == plain
    assert ps._spec("lsm", config, "carried", host="kernel") == dict(plain, host="kernel")
    assert list(ps._spec("lsm", config, "carried", host="kernel")) == [*plain, "host"]
    both = ps._spec("lsm", config, "carried", True, 0, "kernel")
    assert both == dict(plain, noise=True, window=0, host="kernel")
    assert list(both) == ["candidate", "config", "form", "noise", "window", "host"]
    bound = ps._bound(204, ps._spec("lsm", config, "carried"))
    assert list(bound) == ["arm", "seed", "candidate", "config", "form", "noise"]
    assert ps._bound(204, dict(plain, host="port")) == bound  # a named port files nothing
    assert ps._bound(204, ps._spec("lsm", config, "carried", host="kernel")) == dict(
        bound, host="kernel"
    )
    key = ps.cell_key(host_cell(204))
    assert len(key) == 6 and ps.cell_key(host_cell(204, filed_as="port")) == key
    hosted = ps.cell_key(host_cell(204, "kernel", "kernel"))
    assert hosted == (*key, "host", "kernel")
    assert ps.cell_key(dict(host_cell(204), host="port")) == key  # naming the port adds nothing


@pytest.mark.parametrize(
    ("ran_on", "filed_as", "policy", "problems"),
    [
        ("port", "port", True, []),
        ("kernel", "kernel", True, []),
        ("kernel", "port", True, ["host"]),  # a hosted replay filed under a port cell
        ("port", "kernel", True, ["host"]),  # a port replay filed under a hosted cell
        ("kernel", "kernel", False, ["host_policy"]),  # a hosted binding with no policy
    ],
)
def test_the_binding_holds_a_replay_to_the_host_it_is_filed_under(
    ran_on, filed_as, policy, problems
):
    assert ps.binding_problems(host_cell(204, ran_on, filed_as, policy)) == problems
    # a policy that names another host is no policy of this one
    wrong = host_cell(204, "kernel", "kernel")
    wrong["digest"]["pool"]["host_policy"] = {"host": "port"}
    assert ps.binding_problems(wrong) == ["host_policy"]


def test_a_port_and_a_hosted_replay_of_one_seed_are_two_cells_and_a_mismatch_halts():
    table = ps.build_table([host_cell(204), host_cell(204, "kernel", "kernel")], ps.SELECT_SEEDS)
    assert len(table) == 2
    with pytest.raises(ps.Halt, match="R-b7") as twice:
        ps.build_table([host_cell(204, "kernel", "kernel")] * 2, ps.SELECT_SEEDS)
    assert "duplicate seed 204" in twice.value.detail
    for cell in (host_cell(204, "kernel", "port"), host_cell(204, "port", "kernel")):
        with pytest.raises(ps.Halt, match="R-b7") as refused:
            ps.build_table([cell], ps.SELECT_SEEDS)
        assert "['host']" in refused.value.detail


def named_port_surface(seed=204):
    """g1_surface() with the host NAMED as the port wherever a host can be named (the spec's
    argument and the cell's key): the rows must be the ones the default makes."""
    rows = []
    for config in rp.grid("lsm"):
        for form in rp.FORMS:
            for noise in (False, True):
                spec = ps._spec("lsm", config, form, noise=noise, host="port")
                bound = ps._bound(seed, dict(spec, host="port"))
                binding = {
                    "candidate": "lsm",
                    "form": form,
                    "config_crc32": rp.crc_json(config),
                    "realization_seed": rp.realization_seed("lsm", seed),
                    "window": rp.WINDOW,
                    "noise": noise,
                    "w_in": ps._w_in_crc(seed),
                }
                cell = dict(
                    bound, host="port", digest={"arm": "pool", "seed": seed, "pool": binding}
                )
                rows.append([spec, bound, list(ps.cell_key(cell)), ps.binding_problems(cell)])
    return json.dumps(rows, separators=(",", ":"))


def test_naming_the_port_leaves_g1bs_keys_specs_and_bindings_byte_identical():
    text = named_port_surface()
    assert '"host"' not in text
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == G1B_SURFACE_SHA256
    assert text == g1_surface()


def test_tabulate_binds_a_window_zero_replay_under_its_own_cell(tmp_path):
    stream = md._synthetic_stream()
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    path = tmp_path / "17-stream.json.gz"
    md._save(path, stream)
    config = rp.R_B3_WITNESS["lsm"]
    cells = [
        ps._cell(path, tmp_path / f"w{window}.json.gz", 17, index=0, spec=spec)
        for window, spec in (
            (rp.WINDOW, ps._spec("lsm", config, "settled")),
            (0, ps._spec("lsm", config, "settled", window=0)),
        )
    ]

    def runner(spec):
        ps.execute(spec)
        return {"status": "OK", "path": spec["out"]}

    table = ps._tabulate(cells, (17,), {17: md._load(path)}, runner)
    crc = rp.crc_json(rp.canonical("lsm", config))
    default = table[("pool", "lsm", crc, "settled", 17, False)]
    zero = table[("pool", "lsm", crc, "settled", 17, False, 0)]
    assert (default["digest"]["pool"]["window"], zero["digest"]["pool"]["window"]) == (rp.WINDOW, 0)
    assert zero["digest"]["pool"]["movement"] == {} and zero["digest"]["pool"]["boundary"]
    assert default["digest"]["pool"]["boundary"] is None  # 48 percepts: no window boundary


def test_tabulate_binds_real_replays_run_in_process(tmp_path):
    stream = md._synthetic_stream()
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    path = tmp_path / "17-stream.json.gz"
    md._save(path, stream)
    config = rp.R_B3_WITNESS["lsm"]
    cells = [
        ps._cell(
            path, tmp_path / f"{form}.json.gz", 17, index=0, spec=ps._spec("lsm", config, form)
        )
        for form in ps.FIT_FORMS
    ]
    cells.append(ps._cell(path, tmp_path / "reference.json.gz", 17, arm="e1-0"))

    def runner(spec):
        ps.execute(spec)
        return {"status": "OK", "path": spec["out"]}

    table = ps._tabulate(cells, (17,), {17: md._load(path)}, runner)
    carried = table[("pool", "lsm", rp.crc_json(rp.canonical("lsm", config)), "carried", 17, False)]
    assert carried["digest"]["pool"]["window"] == rp.WINDOW
    assert carried["digest"]["readout"]["unseen-order"]["n"] == 24
    report = ps._report(carried)
    assert report["status"] == "OK" and report["frozen"] is None  # 48 percepts: no window boundary
    assert table[("e1-0", None, None, None, 17, False)]["digest"]["pool"] is None


def test_input_response_mask_is_every_unit_for_g1():
    stream = md._synthetic_stream()
    lsm = ps.input_response("lsm", rp.R_B5_PERSISTENCE_WITNESS, stream)
    assert lsm["driven"] == 256 and lsm["mask"] == list(range(256)) and lsm["sd"] > 0.0


def test_twins_differ_separates_the_pool_from_the_learners():
    base = {
        "phi": [1, 2],
        "pool": {"state": 5},
        "predictions": [0],
        "fallbacks": [False],
        "fingerprint": {"W": 1},
        "refits": [],
        "readout": {"predictions": [0], "fingerprint": {"W": 2}},
    }
    assert ps._twins_differ(base, dict(base)) == (False, False)
    assert ps._twins_differ(base, dict(base, phi=[1, 3])) == (True, False)
    assert ps._twins_differ(base, dict(base, pool={"state": 6})) == (True, False)
    guessed = dict(base, readout={"predictions": [1], "fingerprint": {"W": 2}})
    assert ps._twins_differ(base, guessed) == (False, True)
    assert ps._twins_differ(base, dict(base, fingerprint={"W": 9})) == (False, True)


# ---- the step order and section 2.8's taxonomy -------------------------------------------------


def test_a_step_refuses_to_start_out_of_order(tmp_path):
    with pytest.raises(ps.Refused, match="no summary"):
        ps.require(tmp_path, "mechanism", clean)
    write_summary(tmp_path, "record", status="HALTED")
    with pytest.raises(ps.Refused, match="ended HALTED"):
        ps.require(tmp_path, "mechanism", clean)
    write_summary(tmp_path, "record", head="h0")
    with pytest.raises(ps.Refused, match="not at the clean head h1"):
        ps.require(tmp_path, "mechanism", clean)
    write_summary(tmp_path, "record", dirty=True)
    with pytest.raises(ps.Refused, match="dirty=True"):
        ps.require(tmp_path, "mechanism", clean)
    write_summary(tmp_path, "record")
    with pytest.raises(ps.Refused, match="the tree is dirty"):
        ps.require(tmp_path, "mechanism", lambda: ("h1", True))
    assert ps.require(tmp_path, "mechanism", clean) == "h1"
    with pytest.raises(ps.Refused, match="no retries"):
        ps.require(tmp_path, "record", clean)


def test_run_step_writes_each_verdict_with_its_exit_code(tmp_path):
    assert (
        ps.run_step("record", tmp_path, revision=clean, function=lambda out: {"n": 1}) == ps.EXIT_OK
    )
    assert (summary(tmp_path, "record")["status"], summary(tmp_path, "record")["n"]) == (
        "COMPLETED",
        1,
    )

    def halts(out):
        raise ps.Halt("R-b6", {"difference": 0.01})

    assert ps.run_step("mechanism", tmp_path, revision=clean, function=halts) == ps.EXIT_HALT
    halted = summary(tmp_path, "mechanism")
    assert (halted["status"], halted["check"], halted["detail"]) == (
        "HALTED",
        "R-b6",
        {"difference": 0.01},
    )
    assert (halted["head"], halted["dirty"]) == ("h1", False)


def test_a_failed_job_aborts_and_so_does_a_head_that_moved(tmp_path):
    def fails(out):
        raise ps.Abort([{"status": "FAIL", "path": "x"}])

    assert ps.run_step("record", tmp_path / "a", revision=clean, function=fails) == ps.EXIT_ABORT
    assert summary(tmp_path / "a", "record")["status"] == "ABORTED"
    heads = iter([("h1", False), ("h2", False)])
    moved = tmp_path / "b"
    assert ps.run_step("record", moved, revision=lambda: next(heads), function=lambda out: {}) == 1
    assert summary(moved, "record")["moved"] == {"start": "h1", "end": "h2", "dirty": False}


def test_run_jobs_aborts_on_a_fail_and_records_a_timeout():
    jobs = [{"out": f"job{i}"} for i in range(6)]

    def timing(job):
        return {"status": "TIMEOUT" if job["out"] == "job2" else "OK", "path": job["out"]}

    assert [s["status"] for s in ps.run_jobs(jobs, timing)] == [
        "OK",
        "OK",
        "TIMEOUT",
        "OK",
        "OK",
        "OK",
    ]

    def failing(job):
        return {"status": "FAIL" if job["out"] == "job0" else "OK", "path": job["out"]}

    with pytest.raises(ps.Abort) as abort:
        ps.run_jobs(jobs, failing)
    assert {"status": "FAIL", "path": "job0"} in abort.value.jobs


def test_main_refuses_without_writing_a_summary(tmp_path):
    assert ps.main(["mechanism", str(tmp_path)]) == ps.EXIT_REFUSED
    pilot = tmp_path / "pilot"
    assert ps.main(["instrument", str(tmp_path), "--pilot", str(pilot)]) == ps.EXIT_REFUSED
    assert not list(tmp_path.glob("*.summary.json"))


# ---- G1(c) P1-B: R-b6's seed is a step parameter; 4242 stays G1(b)'s default ---------------------


class Cut(Exception):
    """Stand-in for run_jobs where a test stops step_instrument once R-b6's streams are built."""


def stub_instrument(monkeypatch, tmp_path, *, cut):
    """Stub every call step_instrument makes: R-b2, R-b7's reds, the recording, rb6_stream, _save
    and its one run_jobs batch (cut=True raises Cut there), then what it reads back after the
    batch. Returns what the step asked for: rb6_stream's (seed, history_free) calls, the files
    saved and the batches handed to run_jobs."""
    asked = {"streams": [], "saved": [], "batches": []}
    saved = {}

    def rb6_stream(seed, *, history_free=False):
        asked["streams"].append((seed, history_free))
        return {"seed": seed, "history_free": history_free}

    def save(path, value):
        asked["saved"].append(Path(path))
        saved[str(path)] = value

    def run_jobs(jobs):
        asked["batches"].append(list(jobs))
        if cut:
            raise Cut
        return [{"status": "OK"} for _ in jobs]

    def checked(job, stream):
        return {"k": 2 if job["fault"] else 1, "readout": 1 if job["readout"] else None}

    def contrast(stream, first, second):
        return {"readout": {"difference": 0.0 if stream["history_free"] else 0.1}}

    monkeypatch.setattr(md, "check_r_b2", lambda: {"verdict": True})
    monkeypatch.setattr(ps, "r_b7_reds", lambda: {"verdict": True})
    monkeypatch.setattr(ps, "_recorded", lambda out: {ps.PILOT_SEED: "OK"})
    monkeypatch.setattr(ps, "_recording", lambda out, seed: tmp_path / "recorded.json.gz")
    monkeypatch.setattr(md, "rb6_stream", rb6_stream)
    monkeypatch.setattr(md, "_save", save)
    monkeypatch.setattr(ps, "run_jobs", run_jobs)
    monkeypatch.setattr(md, "_load", lambda path: saved.get(str(path), {}))
    monkeypatch.setattr(md, "_source_digest", lambda stream: "digest")
    monkeypatch.setattr(ps, "_checked", checked)
    monkeypatch.setattr(ps, "_contrast", contrast)
    monkeypatch.setattr(ps, "_pilot_run", lambda pilot, arm: {"k": 1, "readout": 1})
    monkeypatch.setattr(ps, "_same", lambda first, second: first["k"] == second["k"])
    return asked


def test_rb6_families_name_the_seed_and_nothing_else():
    folder = Path("x")
    assert ps._rb6_families(folder, 4242) == {
        "rb6": folder / "rb6-4242-stream.json.gz",
        "rb6red": folder / "rb6red-4242-stream.json.gz",
    }
    names = [path.name for path in ps._rb6_families(folder, 6000).values()]
    assert names == ["rb6-6000-stream.json.gz", "rb6red-6000-stream.json.gz"]
    assert not any("4242" in name for name in names)


def test_rb6_seed_defaults_to_g1bs_4242():
    parameter = inspect.signature(ps.step_instrument).parameters["rb6_seed"]
    assert parameter.default == ps.RB6_SEED == 4242


@pytest.mark.parametrize(("kwargs", "seed"), [({}, 4242), ({"rb6_seed": 6000}, 6000)])
def test_step_instrument_builds_and_names_r_b6_on_its_seed(tmp_path, monkeypatch, kwargs, seed):
    asked = stub_instrument(monkeypatch, tmp_path, cut=True)
    out = tmp_path / "out"
    with pytest.raises(Cut):
        ps.step_instrument(out, tmp_path / "pilot", **kwargs)
    folder = out / "instrument"
    assert asked["streams"] == [(seed, False), (seed, True)]
    assert asked["saved"] == [
        folder / f"rb6-{seed}-stream.json.gz",
        folder / f"rb6red-{seed}-stream.json.gz",
    ]
    (batch,) = asked["batches"]
    rb6_jobs = [job for job in batch if Path(job["stream"]).name.startswith("rb6")]
    assert len(rb6_jobs) == 4  # {rb6, rb6red} x {e1, e1-0}, each replaying its own seed's stream
    assert {Path(job["stream"]).name for job in rb6_jobs} == {
        f"rb6-{seed}-stream.json.gz",
        f"rb6red-{seed}-stream.json.gz",
    }
    other = {4242: 6000, 6000: 4242}[seed]  # the seed it must not carry anywhere
    assert not any(str(other) in Path(job["stream"]).name for job in batch)


def test_the_summary_names_the_seed_only_when_it_is_not_g1bs(tmp_path, monkeypatch):
    asked = stub_instrument(monkeypatch, tmp_path, cut=False)
    default = ps.step_instrument(tmp_path / "default", tmp_path / "pilot")
    assert list(default) == ["R-b2", "R-b7 reds", "R-b6", "R-b8"]  # G1(b)'s keys, no rb6_seed
    assert ps.step_instrument(tmp_path / "explicit", tmp_path / "pilot", rb6_seed=4242) == default
    other = ps.step_instrument(tmp_path / "other", tmp_path / "pilot", rb6_seed=6000)
    assert other == {**default, "rb6_seed": 6000}
    # the step stays ONE run_jobs batch: 4 R-b6 replays, an on and an off per scored arm, 2 mutants
    assert [len(batch) for batch in asked["batches"]] == [4 + 2 * len(md.SCORED_ARMS) + 2] * 3


def test_main_passes_the_rb6_seed_to_the_instrument_step(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    names = [f"{ps.PILOT_SEED}-stream.json.gz"]
    names += [f"{ps.PILOT_SEED}-{arm}-causal.json" for arm in md.SCORED_ARMS]
    for name in names:
        (pilot / name).touch()
    given = []

    def run_step(step, out, **kwargs):
        given.append((step, kwargs))
        return ps.EXIT_OK

    monkeypatch.setattr(ps, "run_step", run_step)
    base = ["instrument", str(tmp_path / "out"), "--pilot", str(pilot)]
    assert ps.main(base) == ps.EXIT_OK
    assert ps.main([*base, "--rb6-seed", "6000"]) == ps.EXIT_OK
    assert given == [
        ("instrument", {"pilot": str(pilot), "rb6_seed": 4242}),
        ("instrument", {"pilot": str(pilot), "rb6_seed": 6000}),
    ]


def mechanism_runner(*, red=True, green=True, refuse="lsm", timeout=None):
    """Stand-in children for the mechanism step: one candidate's red or green made to fail."""

    def runner(spec):
        candidate = spec["candidate"]
        if candidate == timeout:
            return {"status": "TIMEOUT", "path": spec["out"]}
        result = {
            "candidate": candidate,
            "greens": {"R-b0": green or candidate != refuse},
            "reds": {"R-b0": red or candidate != refuse},
            "numbers": {},
        }
        md._save(spec["out"], result)
        return {"status": "OK", "path": spec["out"]}

    return runner


def test_mechanism_refuses_a_failed_green_and_halts_on_a_silent_red(tmp_path, monkeypatch):
    first, second, third = tmp_path / "first", tmp_path / "second", tmp_path / "third"
    write_summary(first, "record")
    monkeypatch.setattr(ps, "_child", mechanism_runner(green=False))
    assert ps.run_step("mechanism", first, revision=clean) == ps.EXIT_OK
    completed = summary(first, "mechanism")
    assert completed["refused"] == {"lsm": ["R-b0"]}
    assert completed["readout_r_b0x"]["verdict"]
    write_summary(second, "record")
    monkeypatch.setattr(ps, "_child", mechanism_runner(timeout="lsm"))
    assert ps.run_step("mechanism", second, revision=clean) == ps.EXIT_OK
    assert summary(second, "mechanism")["refused"] == {
        "lsm": ["TIMEOUT: its mechanism checks did not finish"]
    }
    write_summary(third, "record")
    monkeypatch.setattr(ps, "_child", mechanism_runner(red=False))
    assert ps.run_step("mechanism", third, revision=clean) == ps.EXIT_HALT
    halted = summary(third, "mechanism")
    assert (halted["check"], halted["detail"]) == ("a red that does not fire", ["lsm R-b0"])


def test_record_halts_on_a_recording_without_scored_segments(tmp_path, monkeypatch):
    def runner(job):
        if job["seed"] == 101:
            return {"status": "TIMEOUT", "path": job["out"]}
        stream = md._synthetic_stream()
        stream["final"] = {"head": "h1", "dirty": False}
        if job["seed"] == 203 and job["out"].startswith(str(tmp_path / "bad")):
            for event in stream["events"]:
                event["segment"] = "warm-up"
        md._save(job["out"], stream)
        return {"status": "OK", "path": job["out"]}

    monkeypatch.setattr(ps, "_record_child", runner)
    assert ps.run_step("record", tmp_path / "good", revision=clean) == ps.EXIT_OK
    recordings = summary(tmp_path / "good", "record")["recordings"]
    assert recordings["101"] == {"status": "TIMEOUT"}
    assert (
        recordings["100"]["segments"] == {"unseen-order": 24} and recordings["100"]["events"] == 48
    )
    assert ps.run_step("record", tmp_path / "bad", revision=clean) == ps.EXIT_HALT
    assert summary(tmp_path / "bad", "record")["check"] == "scored segments"


def test_the_selectable_set_is_g1_after_section_2_9():
    # section 2.1's dated amendment (2026-09-25, pre-data) set ("lsm", "esn"); its campaign is the
    # record at f27ae70 / 0bd1de7, and section 2.9 then deleted the losing ports
    assert ps.SELECTABLE == ("lsm",) and set(ps.SELECTABLE) <= set(rp.CANDIDATES)


def test_a_reference_that_would_lead_is_ranked_but_never_selected():
    def cell(candidate, order, d, eligible=True):
        return {**entry(order, d, 0.1, 1.0), "candidate": candidate, "eligible": eligible}

    # hand-built entries: sorn and esn stand in for references (neither is selectable now)
    sorn, esn, lsm = cell("sorn", 3, 0.050), cell("esn", 2, 0.020), cell("lsm", 1, 0.010)
    ranked, reference = ps.rank_selection([lsm, sorn, esn], 0.005)
    assert [e["candidate"] for e in reference] == ["sorn", "esn", "lsm"]
    assert [e["candidate"] for e in ranked] == ["lsm"]  # the winner is lsm, not sorn or esn
    # g1 not eligible: nothing is selected even though the references are
    ranked, reference = ps.rank_selection([sorn, esn, cell("lsm", 1, 0.01, eligible=False)], 0.005)
    assert ranked == [] and [e["candidate"] for e in reference] == ["sorn", "esn"]


def test_the_noise_red_reads_the_state_chain_not_one_final_sample():
    # campaign 1's g3 shape on seed 203: the final state converged, the chain and phi did not
    def run(state, chain, phi):
        return {"pool": {"state": state, "state_chain": chain}, "phi": phi}

    plain, kicked = run(7, 100, 1), run(7, 200, 2)
    assert plain["pool"]["state"] == kicked["pool"]["state"]  # the pre-amendment red was blind here
    assert ps.noise_red_fires(plain, kicked)
    assert not ps.noise_red_fires(plain, run(9, 100, 2))  # chain unchanged: the red does not fire
    assert not ps.noise_red_fires(plain, run(7, 200, 1))  # phi unchanged: the red does not fire
    # the twins' comparison sees a chain-only difference as a pool difference
    twin = {**run(7, 101, 1), "predictions": [], "fallbacks": [], "fingerprint": 0, "refits": []}
    base = {**plain, "predictions": [], "fallbacks": [], "fingerprint": 0, "refits": []}
    assert ps._twins_differ({**base, "readout": None}, {**twin, "readout": None})[0]


def test_self_check_runs():
    assert ps.main([]) == ps.EXIT_OK


# ---- G1(c) P3c: the transport arms in the filing, the key, the binding and the jobs --------------


def transport_cell(arm, seed=204, *, encoder=None, transport=None, ran=None):
    """A transport arm's cell: filed under `arm` (and the encoder / transport it names, if any),
    its replay bound to what the filing names unless `ran` overrides a field of the binding."""
    config = rp.grid("lsm")[0]
    filed = ps._bound(
        seed, ps._spec("lsm", config, "carried"), arm=arm, encoder=encoder, transport=transport
    )
    binding = {
        "candidate": "lsm",
        "form": "carried",
        "config_crc32": rp.crc_json(config),
        "realization_seed": rp.realization_seed("lsm", seed),
        "window": rp.WINDOW,
        "noise": False,
        "w_in": ps._w_in_crc(seed),
        "encoder": 123456,
        "transport": {"form": "copy", "copied_at_event": 256, "L": 3},
    }
    binding.update(ran or {})
    return dict(filed, index=0, status="OK", digest={"arm": arm, "seed": seed, "pool": binding})


def test_a_transport_filing_names_its_arm_and_carries_encoder_and_transport_only_when_given():
    config = rp.grid("lsm")[0]
    spec = ps._spec("lsm", config, "carried")
    assert ps._bound(204, spec, arm="pool") == ps._bound(204, spec)  # the default arm is pool
    copy_bound = ps._bound(204, spec, arm="pool-copy")
    assert copy_bound == dict(ps._bound(204, spec), arm="pool-copy")
    assert list(copy_bound) == ["arm", "seed", "candidate", "config", "form", "noise"]
    named = ps._bound(204, spec, arm="pool-copy", encoder=7, transport={"form": "copy"})
    assert list(named) == [*copy_bound, "encoder", "transport"]
    assert (named["encoder"], named["transport"]) == (7, {"form": "copy"})
    assert "encoder" not in ps._bound(204, spec, arm="pool-swap", transport={"form": "swap"})
    assert ps._spec("lsm", config, "carried") == {
        "candidate": "lsm",
        "config": config,
        "form": "carried",
    }


def test_the_cell_key_gains_the_encoder_and_the_transport_only_when_the_cell_names_them():
    config = rp.grid("lsm")[0]
    base = ("pool-copy", "lsm", rp.crc_json(config), "carried", 204, False)
    assert ps.cell_key(transport_cell("pool-copy")) == base
    named = transport_cell("pool-copy", encoder=7, transport={"form": "copy"})
    assert ps.cell_key(named) == (*base, "transport", rp.crc_json({"form": "copy"}), "encoder", 7)
    only = transport_cell("pool-copy", transport={"form": "copy"})
    assert ps.cell_key(only) == (*base, "transport", rp.crc_json({"form": "copy"}))
    # the same arm filed under two transports is two cells; the arm already separates the rest
    other = transport_cell("pool-copy", transport={"form": "copy", "copied_at_event": 300})
    assert ps.cell_key(only) != ps.cell_key(other)
    assert ps.cell_key(transport_cell("pool-act")) != ps.cell_key(transport_cell("pool-copy"))
    # a window and a transport together: the window first, then the transport
    config_cell = dict(transport_cell("pool-copy", transport={"form": "copy"}), window=0)
    assert ps.cell_key(config_cell)[6:] == (0, "transport", rp.crc_json({"form": "copy"}))


@pytest.mark.parametrize("arm", ["pool-copy", "pool-act", "pool-swap"])
def test_a_transport_replay_filed_under_its_own_arm_binds_and_under_another_does_not(arm):
    assert ps.binding_problems(transport_cell(arm)) == []
    cell = transport_cell(arm)
    wrong = dict(cell, digest=dict(cell["digest"], arm="pool"))
    assert ps.binding_problems(wrong) == ["arm"]
    plain = dict(transport_cell("pool"), digest=dict(cell["digest"], arm="pool"))
    assert ps.binding_problems(plain) == []  # a plain pool cell never asks for the transport keys
    bare = dict(cell, digest=dict(cell["digest"], pool=None))
    assert ps.binding_problems(bare) == ["pool"]
    other = rp.grid("lsm")[1]
    refiled = transport_cell(arm, ran={"config_crc32": rp.crc_json(other)})
    assert ps.binding_problems(refiled) == ["config_crc32"]


def test_the_encoder_and_transport_a_cell_names_must_be_the_replays():
    ok = transport_cell("pool-copy", encoder=123456, transport={"form": "copy", "L": 3})
    assert ps.binding_problems(ok) == []
    wrong_encoder = transport_cell("pool-copy", encoder=1, ran={})
    assert ps.binding_problems(wrong_encoder) == ["encoder"]
    wrong_event = transport_cell("pool-copy", transport={"copied_at_event": 300})
    assert ps.binding_problems(wrong_event) == ["transport"]
    wrong_form = transport_cell("pool-copy", transport={"form": "swap"})
    assert ps.binding_problems(wrong_form) == ["transport"]
    # a replay whose binding lacks what the cell names disagrees on it
    lacking = transport_cell("pool-copy", encoder=123456)
    del lacking["digest"]["pool"]["encoder"]
    assert ps.binding_problems(lacking) == ["encoder"]
    unrecorded = transport_cell("pool-act", transport={"form": "activity"})
    del unrecorded["digest"]["pool"]["transport"]
    assert ps.binding_problems(unrecorded) == ["transport"]


def test_a_replay_job_names_a_tap_or_a_side_only_when_given():
    plain = ps._replay_job("s", "a", "o")
    assert list(plain) == ["kind", "stream", "arm", "pool", "readout", "fault", "out"]
    assert ps._replay_job("s", "a", "o", pool=None, readout=True, fault=None) == plain
    tapped = ps._replay_job("s", "a", "o", tap=256, fault="leaky_tap")
    assert tapped == dict(plain, fault="leaky_tap", tap=256) and "side" not in tapped
    sided = ps._replay_job("s", "pool-copy", "o", pool={"x": 1}, side="tap.json.gz")
    assert sided["side"] == "tap.json.gz" and "tap" not in sided


def test_the_tap_faults_are_the_harness_context_managers():
    assert ps.FAULTS["late_tap"] is md._late_tap
    assert ps.FAULTS["leaky_tap"] is md._leaky_tap
    assert ps.FAULTS["writing_tap"] is md._writing_tap
    assert list(ps.FAULTS)[:3] == [None, "e1_one_element", "writing_readout"]  # G1(b)'s, in order


def test_execute_forwards_a_tap_and_a_side_only_when_the_job_names_them(tmp_path, monkeypatch):
    path = tmp_path / "17-stream.json.gz"
    md._save(path, md._synthetic_stream())
    side = tmp_path / "side.json.gz"
    md._save(side, {"arm": "a", "tap": {"b": 16, "marker": 1}})
    bare = tmp_path / "bare.json.gz"
    md._save(bare, {"b": 16, "marker": 2})
    calls = []

    def spy(stream, arm, **kwargs):
        calls.append((arm, kwargs))
        return {"arm": arm}

    monkeypatch.setattr(md, "replay", spy)
    ps.execute(ps._replay_job(path, "a", tmp_path / "o1.json"))
    ps.execute(ps._replay_job(path, "a", tmp_path / "o2.json", tap=16))
    ps.execute(ps._replay_job(path, "pool-copy", tmp_path / "o3.json", pool={"p": 1}, side=side))
    ps.execute(ps._replay_job(path, "pool-swap", tmp_path / "o4.json", pool={"p": 1}, side=bare))
    # a default job is the call G1(b) made, keyword for keyword
    assert calls[0] == ("a", {"readout": True, "pool": None})
    assert calls[1] == ("a", {"readout": True, "pool": None, "tap": 16})
    assert calls[2] == (
        "pool-copy",
        {"readout": True, "pool": {"p": 1}, "side": {"b": 16, "marker": 1}},
    )
    assert calls[3][1]["side"] == {"b": 16, "marker": 2}  # a bare tap file is as good as a result


def test_a_tap_job_feeds_a_transport_job_through_files_in_process(tmp_path):
    stream = md._synthetic_stream()
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    path = tmp_path / "17-stream.json.gz"
    md._save(path, stream)
    tap_out, copy_out, leaky_out = (
        tmp_path / f"{name}.json.gz" for name in ("tap", "copy", "leaky")
    )
    ps.execute(ps._replay_job(path, "a", tap_out, tap=16))
    ps.execute(ps._replay_job(path, "a", leaky_out, tap=16, fault="leaky_tap"))
    spec = ps._spec("lsm", rp.R_B5_PERSISTENCE_WITNESS, "carried", window=16)
    ps.execute(ps._replay_job(path, "pool-copy", copy_out, pool=spec, side=tap_out))
    tap, copied, leaky = (md._load(p) for p in (tap_out, copy_out, leaky_out))
    assert (
        tap["tap"]["w_at_b"]["event"] == 16 and copied["pool"]["transport"]["copied_at_event"] == 16
    )
    assert copied["pool"]["transport"]["L"] == len(tap["tap"]["w_at_b"]["W"])
    assert leaky["tap"]["w_at_b"]["crc_W"] != tap["tap"]["w_at_b"]["crc_W"]  # the final model's
    # the copy's own digest binds under the filing that names its arm, window, encoder and transport
    record = copied["pool"]["transport"]
    filed = ps._bound(
        17, spec, arm="pool-copy", encoder=copied["pool"]["encoder"], transport={"form": "copy"}
    )
    cell = dict(filed, digest=ps.digest(copied, stream))
    assert record["form"] == "copy" and ps.binding_problems(cell) == []
    assert ps.binding_problems(dict(cell, arm="pool-act", digest=dict(cell["digest"]))) == ["arm"]
