"""G1(b)'s campaign (instruments/pool_selection.py): section 2.5's ordering and eligibility on
hand-built tables, section 2.6's power rule, R-b7's refusals, the step-order refusal and section
2.8's taxonomy and exit codes (g1b-pool-selection.md, frozen at 23c7e97).

Nothing here runs a campaign: steps run on a scratch OUT at a stand-in revision, and their jobs go
through stand-in runners (or one in-process replay of the 48-event fixture)."""

import functools
import hashlib
import inspect
import json
import math
import os
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
        "hosted binding with a contradictory policy": True,
        "hosted binding with one policy field changed": True,
        "unsupported host": True,
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


def hosted_policy(config):
    """The policy a real hosted pool of `config` records: what a hosted replay's binding carries."""
    return rp.make_pool("lsm", config, 204, host="kernel").host_policy()


def host_cell(seed, ran_on="port", filed_as="port", policy=True):
    """A Settled cell whose replay ran on host `ran_on`, filed under host `filed_as` (the port files
    no host at all); a hosted binding carries "host" and, if `policy`, its whole "host_policy"."""
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
            binding["host_policy"] = hosted_policy(config)
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


# the G1(c) P4 panel: R-b7 held only policy["host"], so a replay claiming frac_bits 4, delay 2,
# thr 99 and one refractory tick entered the table. The WHOLE policy is held now, at the lineage's
# options
POLICY_FIELDS = (
    "host",
    "frac_bits",
    "p_parts",
    "decay_f32",
    "v_reset",
    "refrac_ticks",
    "delay",
    "thr",
)


def policy_cell(edit=None, drop=(), seed=204):
    """A hosted cell filed as hosted whose binding's policy is the real one with `edit` applied and
    the `drop` fields removed."""
    cell = host_cell(seed, "kernel", "kernel")
    real = cell["digest"]["pool"]["host_policy"]
    cell["digest"]["pool"]["host_policy"] = {
        key: value for key, value in {**real, **(edit or {})}.items() if key not in drop
    }
    return cell


def test_the_lineage_policy_is_the_policy_of_a_real_hosted_pool_of_every_leak():
    grid = rp.grid("lsm")
    assert tuple(hosted_policy(grid[0])) == POLICY_FIELDS
    for leak in (0.5, 0.7, 0.9):
        config = next(config for config in grid if config["lambda"] == leak)
        assert rp.KernelLsmPool.lineage_policy(config) == hosted_policy(config)
    assert hosted_policy(next(c for c in grid if c["lambda"] == 0.9))["decay_f32"] != 0.9


@pytest.mark.parametrize("field", POLICY_FIELDS)
def test_a_hosted_binding_whose_policy_changes_or_loses_one_field_is_refused(field):
    value = hosted_policy(rp.grid("lsm")[0])[field]
    other = "port" if isinstance(value, str) else value + 1
    assert ps.binding_problems(policy_cell({field: other})) == ["host_policy"]
    assert ps.binding_problems(policy_cell(drop=(field,))) == ["host_policy"]
    with pytest.raises(ps.Halt, match="R-b7") as refused:
        ps.build_table([policy_cell({field: other})], ps.SELECT_SEEDS)
    assert "host_policy" in refused.value.detail
    assert (
        ps.binding_problems(policy_cell({field: value})) == []
    )  # the unchanged field is the control


def test_a_missing_short_extended_or_contradictory_hosted_policy_is_refused():
    base = hosted_policy(rp.grid("lsm")[0])
    assert ps.binding_problems(policy_cell()) == []
    contradictory = dict(base, frac_bits=4, delay=2, thr=99.0, refrac_ticks=1)  # the panel's mutant
    for bad in ({"host": "kernel"}, contradictory, {}, dict(base, extra=1), None, "kernel", [1]):
        cell = host_cell(204, "kernel", "kernel")
        cell["digest"]["pool"]["host_policy"] = bad
        assert ps.binding_problems(cell) == ["host_policy"], bad
        with pytest.raises(ps.Halt, match="R-b7"):
            ps.build_table([cell], ps.SELECT_SEEDS)
    cell = host_cell(204, "kernel", "kernel")
    del cell["digest"]["pool"]["host_policy"]  # no policy at all
    assert ps.binding_problems(cell) == ["host_policy"]
    # the policy of another configuration's leak is no policy of this one
    other = next(c for c in rp.grid("lsm") if c["lambda"] != rp.grid("lsm")[0]["lambda"])
    cell = host_cell(204, "kernel", "kernel")
    cell["digest"]["pool"]["host_policy"] = hosted_policy(other)
    assert ps.binding_problems(cell) == ["host_policy"]


def test_an_unsupported_host_is_refused_though_its_cell_and_binding_agree():
    cell = host_cell(204, "kernel", "kernel")
    binding = cell["digest"]["pool"]
    cell["host"], binding["host"], binding["host_policy"]["host"] = "gpu", "gpu", "gpu"
    assert ps.binding_problems(cell) == ["host"]
    with pytest.raises(ps.Halt, match="R-b7"):
        ps.build_table([cell], ps.SELECT_SEEDS)


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


# ---- G1(c) P5 panel F1: a step that started cannot start again, crash included -------------------


def test_a_step_interrupted_before_its_summary_is_written_cannot_be_started_again(tmp_path):
    def interrupted(out):
        assert summary(out, "record")["status"] == "RUNNING"  # filed BEFORE the body ran
        raise KeyboardInterrupt  # not an Exception: run_step's own handlers never see it

    with pytest.raises(KeyboardInterrupt):
        ps.run_step("record", tmp_path, revision=clean, function=interrupted)
    crashed = summary(tmp_path, "record")
    assert (crashed["status"], crashed["head"], crashed["dirty"]) == ("RUNNING", "h1", False)
    ran = []
    with pytest.raises(ps.Refused, match="never finished.*RUNNING.*no retries"):
        ps.run_step("record", tmp_path, revision=clean, function=lambda out: ran.append(1) or {})
    assert ran == [] and summary(tmp_path, "record") == crashed  # refused, nothing run or rewritten
    with pytest.raises(ps.Refused, match="no summary|ended RUNNING"):  # and no later step either
        ps.require(tmp_path, "mechanism", clean)


def test_a_normal_step_replaces_its_running_summary_with_the_one_it_always_wrote(tmp_path):
    seen = []

    def body(out):
        seen.append(summary(out, "record")["status"])
        return {"n": 1}

    assert ps.run_step("record", tmp_path, revision=clean, function=body) == ps.EXIT_OK
    assert seen == ["RUNNING"]
    assert summary(tmp_path, "record") == {
        "step": "record",
        "status": "COMPLETED",
        "head": "h1",
        "dirty": False,
        "n": 1,
    }

    def halts(out):
        raise ps.Halt("R-b6", {"difference": 0.01})

    assert ps.run_step("mechanism", tmp_path, revision=clean, function=halts) == ps.EXIT_HALT
    assert summary(tmp_path, "mechanism")["status"] == "HALTED"
    with pytest.raises(ps.Refused, match="ended HALTED"):  # a halted predecessor still blocks
        ps.require(tmp_path, "instrument", clean)


def test_a_second_start_that_passed_require_is_refused_by_the_exclusive_create(
    tmp_path, monkeypatch
):
    ran = []
    monkeypatch.setattr(
        ps, "require", lambda out, step, revision, steps=ps.STEPS: "h1"
    )  # both saw "no summary"
    assert (
        ps.run_step("record", tmp_path, revision=clean, function=lambda out: ran.append(1) or {})
        == 0
    )
    with pytest.raises(ps.Refused, match="concurrent start filed its summary first"):
        ps.run_step("record", tmp_path, revision=clean, function=lambda out: ran.append(2) or {})
    assert ran == [1]
    with pytest.raises(ps.Refused, match="concurrent start"):  # the mechanism itself: mode "x"
        ps._file_running(tmp_path, "record", "h1")
    assert (
        summary(tmp_path, "record")["status"] == "COMPLETED"
    )  # the finished summary was never rewritten


# ---- G1(c) re-panel R2: a caller's provenance is filed before the body, kept in every summary ----


def test_without_provenance_every_summary_is_byte_for_byte_what_run_step_always_wrote(tmp_path):
    """G1(b)'s steps pass no provenance: the RUNNING marker and the final summaries keep their exact
    keys and order (compared as text, not as dicts)."""
    seen = {}

    def body(out):
        seen["marker"] = ps._summary_path(out, "record").read_text(encoding="utf-8")
        return {"n": 1}

    def halts(out):
        raise ps.Halt("R-b6", {"difference": 0.01})

    def aborts(out):
        raise ps.Abort([{"status": "FAIL", "path": "x"}])

    base = {"step": "record", "status": "RUNNING", "head": "h1", "dirty": False}
    assert ps.run_step("record", tmp_path / "ok", revision=clean, function=body) == ps.EXIT_OK
    assert seen["marker"] == json.dumps(base)
    done = ps._summary_path(tmp_path / "ok", "record").read_text(encoding="utf-8")
    assert done == json.dumps({**base, "status": "COMPLETED", "n": 1})
    assert ps.run_step("record", tmp_path / "halt", revision=clean, function=halts) == ps.EXIT_HALT
    stopped = ps._summary_path(tmp_path / "halt", "record").read_text(encoding="utf-8")
    assert stopped == json.dumps(
        {**base, "status": "HALTED", "check": "R-b6", "detail": {"difference": 0.01}}
    )
    assert ps.run_step("record", tmp_path / "abort", revision=clean, function=aborts) == 1
    failed = ps._summary_path(tmp_path / "abort", "record").read_text(encoding="utf-8")
    assert failed == json.dumps(
        {**base, "status": "ABORTED", "failed": [{"status": "FAIL", "path": "x"}]}
    )


def test_run_step_files_a_callers_provenance_before_the_body_and_keeps_it_in_every_summary(
    tmp_path,
):
    """Re-panel R2: the provenance is in the exclusive RUNNING marker before the body runs (so a
    crash leaves it), in COMPLETED, HALTED and ABORTED summaries (a moved head and a defect
    included), is never passed to the step's function, and is never displaced by the body's own
    keys."""
    provenance = {"seed_set": "live"}
    base = {"step": "record", "status": "RUNNING", "head": "h1", "dirty": False}
    seen = {}

    def body(out, **kwargs):
        seen["marker"], seen["kwargs"] = summary(out, "record"), kwargs
        return {"n": 1, "seed_set": "claimed by the body"}

    def halts(out):
        raise ps.Halt("R-b6", {"difference": 0.01})

    def aborts(out):
        raise ps.Abort([{"status": "FAIL", "path": "x"}])

    def defect(out):
        raise ValueError("a defect in the parent")

    def interrupted(out):
        raise KeyboardInterrupt

    run = functools.partial(ps.run_step, "record", revision=clean, provenance=provenance)
    assert run(tmp_path / "ok", function=body, extra=2) == ps.EXIT_OK
    assert seen["marker"] == {**base, "seed_set": "live"} and seen["kwargs"] == {"extra": 2}
    assert summary(tmp_path / "ok", "record") == {
        **base,
        "status": "COMPLETED",
        "n": 1,
        "seed_set": "live",
    }
    assert run(tmp_path / "halt", function=halts) == ps.EXIT_HALT
    assert run(tmp_path / "abort", function=aborts) == ps.EXIT_ABORT
    assert run(tmp_path / "defect", function=defect) == ps.EXIT_ABORT
    heads = iter([("h1", False), ("h2", False)])
    assert (
        ps.run_step(
            "record",
            tmp_path / "moved",
            revision=lambda: next(heads),
            function=lambda o: {},
            provenance=provenance,
        )
        == ps.EXIT_ABORT
    )
    for name, status in (
        ("halt", "HALTED"),
        ("abort", "ABORTED"),
        ("defect", "ABORTED"),
        ("moved", "ABORTED"),
    ):
        found = summary(tmp_path / name, "record")
        assert (found["status"], found["seed_set"]) == (status, "live"), name
    with pytest.raises(KeyboardInterrupt):  # the marker a crash leaves carries it too
        run(tmp_path / "crash", function=interrupted)
    assert summary(tmp_path / "crash", "record") == {**base, "seed_set": "live"}
    with pytest.raises(ps.Refused, match="never finished"):  # and the crash still allows no retry
        run(tmp_path / "crash", function=lambda out: {})


def test_provenance_that_names_a_key_every_summary_carries_is_refused_before_anything_is_filed(
    tmp_path,
):
    for clash in ("step", "status", "head", "dirty"):
        with pytest.raises(ValueError, match=clash):
            ps.run_step(
                "record",
                tmp_path / clash,
                revision=clean,
                function=lambda out: {},
                provenance={clash: "x"},
            )
        assert not list((tmp_path / clash).glob("*.summary.json"))


def test_prefix_held_reads_both_learners_up_to_the_cut_and_nothing_after():
    base = {"predictions": [1, 2, 3, 4], "readout": {"predictions": [4, 3, 2, 1]}}
    same = {"predictions": [1, 2, 3, 9], "readout": {"predictions": [4, 3, 2, 9]}}
    assert ps.prefix_held(base, same, 3) == (True, True)  # the suffix after the cut moved
    assert ps.prefix_held(base, same, 4) == (False, False)
    assert ps.prefix_held(base, {**same, "predictions": [1, 9, 3, 9]}, 3) == (False, True)
    assert ps.prefix_held(base, {**same, "readout": {"predictions": [4, 9, 2, 9]}}, 3) == (
        True,
        False,
    )


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


# ---- G1(c) P5a: the step, job and table machinery takes G1(c)'s parameters, G1(b)'s defaults -----


CUSTOM_STEPS = ("alpha", "beta", "gamma")
POWER_S = "snn_data_lake.instruments.pool_selection:power_s"  # a pure function of the instruments


def test_the_new_parameters_default_to_g1bs_values():
    assert inspect.signature(ps.require).parameters["steps"].default is ps.STEPS
    assert inspect.signature(ps.run_step).parameters["steps"].default is ps.STEPS
    phase = inspect.signature(ps._phase_table).parameters
    assert (phase["paths"].default, phase["window"].default, phase["host"].default) == (
        None,
        rp.WINDOW,
        "port",
    )


def test_require_walks_a_custom_step_list_with_g1bs_refusals(tmp_path):
    steps = CUSTOM_STEPS
    # the first step of the list needs no predecessor
    assert ps.require(tmp_path, "alpha", clean, steps) == "h1"
    with pytest.raises(ps.Refused, match="alpha has no summary"):
        ps.require(tmp_path, "beta", clean, steps)
    write_summary(tmp_path, "alpha", status="HALTED")
    with pytest.raises(ps.Refused, match="ended HALTED"):
        ps.require(tmp_path, "beta", clean, steps)
    write_summary(tmp_path, "alpha", head="h0")
    with pytest.raises(ps.Refused, match="not at the clean head h1"):
        ps.require(tmp_path, "beta", clean, steps)
    write_summary(tmp_path, "alpha", dirty=True)
    with pytest.raises(ps.Refused, match="dirty=True"):
        ps.require(tmp_path, "beta", clean, steps)
    write_summary(tmp_path, "alpha")
    with pytest.raises(ps.Refused, match="the tree is dirty"):
        ps.require(tmp_path, "beta", lambda: ("h1", True), steps)
    assert ps.require(tmp_path, "beta", clean, steps) == "h1"
    with pytest.raises(ps.Refused, match="no retries"):
        ps.require(tmp_path, "alpha", clean, steps)
    # a step's predecessor is the one before it in THIS list: gamma wants beta, not alpha
    with pytest.raises(ps.Refused, match="beta has no summary"):
        ps.require(tmp_path, "gamma", clean, steps)
    # a G1(b) step is no step of this list (the index lookup fails as it always did)
    with pytest.raises(ValueError):
        ps.require(tmp_path, "mechanism", clean, steps)


def test_require_at_the_default_list_is_g1bs_step_order(tmp_path):
    with pytest.raises(ps.Refused, match="record has no summary"):
        ps.require(tmp_path, "mechanism", clean)
    with pytest.raises(ps.Refused, match="record has no summary"):
        ps.require(tmp_path, "mechanism", clean, ps.STEPS)
    assert ps.require(tmp_path, "record", clean) == ps.require(tmp_path, "record", clean, ps.STEPS)
    write_summary(tmp_path, "alpha")  # a custom step's summary is no predecessor of G1(b)'s
    with pytest.raises(ps.Refused, match="record has no summary"):
        ps.require(tmp_path, "mechanism", clean)


def test_run_step_passes_its_step_list_to_require_and_not_to_the_function(tmp_path):
    def first(out):
        return {"n": 1}

    def second(out, extra=0):
        return {"extra": extra}

    steps = CUSTOM_STEPS
    assert ps.run_step("alpha", tmp_path, revision=clean, function=first, steps=steps) == ps.EXIT_OK
    assert summary(tmp_path, "alpha")["n"] == 1
    assert (
        ps.run_step("beta", tmp_path, revision=clean, function=second, steps=steps, extra=3)
        == ps.EXIT_OK
    )
    assert summary(tmp_path, "beta")["extra"] == 3  # the other keywords still reach the function
    write_summary(tmp_path / "halted", "beta", status="HALTED")
    with pytest.raises(ps.Refused, match="gamma: beta ended HALTED"):
        ps.run_step("gamma", tmp_path / "halted", revision=clean, function=first, steps=steps)
    # the same call without `steps` is G1(b)'s list: alpha is not a step there, and record is
    with pytest.raises(ValueError):
        ps.run_step("alpha", tmp_path / "default", revision=clean, function=first)
    assert ps.run_step("record", tmp_path / "default", revision=clean, function=first) == ps.EXIT_OK


def fake_record_child(unscored=()):
    """A stand-in for memory_diagnostic's --record child: the 48-event fixture under the seed's name
    (`unscored` seeds get one with no scored segment). The fixtures are built HERE, in the calling
    thread: md._synthetic_stream asks threadpoolctl, which can fail when its first call comes from
    several worker threads at once."""
    stream = md._synthetic_stream()
    stream["final"] = {"head": "h1", "dirty": False}
    blind = dict(stream, events=[dict(event, segment="warm-up") for event in stream["events"]])

    def runner(job):
        md._save(job["out"], blind if job["seed"] in unscored else stream)
        return {"status": "OK", "path": job["out"]}

    return runner


def test_record_seeds_takes_any_seed_list_and_step_record_is_g1bs(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_record_child", fake_record_child())
    custom = ps._record_seeds(tmp_path / "custom", (300, 301))
    assert list(custom["recordings"]) == ["300", "301"]
    assert {row["status"] for row in custom["recordings"].values()} == {"OK"}
    assert sorted(path.name for path in (tmp_path / "custom" / "record").iterdir()) == [
        "300-stream.json.gz",
        "301-stream.json.gz",
    ]
    default = ps.step_record(tmp_path / "default")
    assert list(default["recordings"]) == [str(seed) for seed in ps.RECORD_SEEDS]
    assert default == ps._record_seeds(tmp_path / "again", ps.RECORD_SEEDS)
    # the CLI's help line reads this docstring
    assert ps.step_record.__doc__.startswith("Seeds 100-103 and 200-207")


def test_record_seeds_halts_on_a_recording_without_scored_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_record_child", fake_record_child(unscored=(301,)))
    with pytest.raises(ps.Halt, match="scored segments") as halted:
        ps._record_seeds(tmp_path, (300, 301))
    assert "recording 301" in halted.value.detail


def fixture_stream(tmp_path, name, *, shuffled=False):
    """The 48-event fixture bound to this head, saved under `name`; `shuffled` swaps the percept
    fields (x, c) of its last two events, a stand-in shuffle that keeps every binding key."""
    stream = md._synthetic_stream()
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    if shuffled:
        last, before = stream["events"][-1], stream["events"][-2]
        for key in ("x", "c"):
            last[key], before[key] = before[key], last[key]
    md._save(tmp_path / name, stream)
    return tmp_path / name, stream


def in_process(spec):
    ps.execute(spec)
    return {"status": "OK", "path": spec["out"]}


def test_a_cell_validates_against_its_own_stream_only_when_it_names_it(tmp_path):
    path, plain = fixture_stream(tmp_path, "17-stream.json.gz")
    shuffled_path, shuffled = fixture_stream(tmp_path, "17-shuffled.json.gz", shuffled=True)
    assert md._source_digest(plain) != md._source_digest(shuffled)
    streams = {17: plain, "shuffled": shuffled}

    def cells(**keys):
        """One reference replay of the SHUFFLED file, filed under seed 17, naming `keys`."""
        cell = ps._cell(shuffled_path, tmp_path / "shuffled-e1-0.json.gz", 17, arm="e1-0")
        return [dict(cell, **keys)]

    # without the keys the cell is held to seed 17's recording: another stream's replay is refused
    with pytest.raises(ps.Halt, match="R-b7") as refused:
        ps._tabulate(cells(), (17,), streams, in_process)
    assert "replay identity e1-0 causal" in refused.value.detail
    # a stream_key makes the cell's stream, source and scoring the shuffled one
    table = ps._tabulate(cells(stream_key="shuffled"), (17,), streams, in_process)
    (cell,) = table.values()
    assert cell["status"] == "OK"
    assert cell["digest"] == ps.digest(md._load(cell["job"]["out"]), shuffled)
    # a source_key alone makes the source the shuffled one (the provenance stays seed 17's)
    assert len(ps._tabulate(cells(source_key="shuffled"), (17,), streams, in_process)) == 1
    # a source that is not the stream the replay ran on is refused, with or without a stream_key
    for keys in ({"stream_key": "shuffled", "source_key": 17}, {"source_key": 17}):
        with pytest.raises(ps.Halt, match="R-b7") as wrong:
            ps._tabulate(cells(**keys), (17,), streams, in_process)
        assert "replay identity" in wrong.value.detail


def test_a_cell_that_names_no_keys_is_tabulated_as_it_always_was(tmp_path):
    path, plain = fixture_stream(tmp_path, "17-stream.json.gz")
    cell = ps._cell(path, tmp_path / "e1-0.json.gz", 17, arm="e1-0")
    assert "stream_key" not in cell and "source_key" not in cell
    table = ps._tabulate([cell], (17,), {17: plain}, in_process)
    assert list(table) == [("e1-0", None, None, None, 17, False)]
    assert table[("e1-0", None, None, None, 17, False)]["digest"] == ps.digest(
        md._load(cell["job"]["out"]), plain
    )


def capture_tabulate(monkeypatch):
    """Stand in for _tabulate where a test reads what _phase_table hands it."""
    taken = {}

    def tabulate(cells, phase_seeds, streams, runner=None):
        taken.update(cells=cells, seeds=phase_seeds, streams=streams)
        return "table"

    monkeypatch.setattr(ps, "_tabulate", tabulate)
    return taken


def phase_table(tmp_path, monkeypatch, recorded=None, **kwargs):
    """_phase_table over seed 17 (live) and 5001 (synthetic) with the record summary and the
    synthetic writer stood in for; returns (cells, the _synthetic calls). `recorded` is what the
    record step says of seed 17 (OK by default)."""
    taken, synthetic = capture_tabulate(monkeypatch), []

    def writer(folder, seeds):
        synthetic.append((folder, tuple(seeds)))
        return {seed: folder / f"task-{seed}-stream.json.gz" for seed in seeds}

    monkeypatch.setattr(ps, "_recorded", lambda out: {17: "OK"} if recorded is None else recorded)
    monkeypatch.setattr(ps, "_synthetic", writer)
    standing = {"lsm": [(0, rp.R_B3_WITNESS["lsm"])]}
    out, folder = tmp_path / "out", tmp_path / "folder"
    table = ps._phase_table(out, folder, standing, (17,), (5001,), ("carried",), ("a",), **kwargs)
    assert table == "table"
    return taken["cells"], synthetic


def test_phase_table_builds_its_stream_paths_unless_it_is_given_them(tmp_path, monkeypatch):
    cells, synthetic = phase_table(tmp_path, monkeypatch)
    assert synthetic == [(tmp_path / "folder", (5001,))]
    assert [Path(cell["job"]["stream"]) for cell in cells] == [
        tmp_path / "out" / "record" / "17-stream.json.gz",  # the pool cell of the live seed
        tmp_path / "folder" / "task-5001-stream.json.gz",  # the pool cell of the synthetic seed
        tmp_path / "out" / "record" / "17-stream.json.gz",  # the reference arm on the live seed
    ]


def test_phase_table_files_every_cell_under_the_paths_it_is_given(tmp_path, monkeypatch):
    paths = {17: tmp_path / "xor-17.json.gz", 5001: tmp_path / "twin-5001.json.gz"}
    cells, synthetic = phase_table(tmp_path, monkeypatch, paths=paths)
    assert synthetic == []  # the given paths are not overwritten by the default synthetic writer
    assert [Path(cell["job"]["stream"]) for cell in cells] == [
        paths[17],
        paths[5001],
        paths[17],
    ]
    # the record summary is still what says a live seed is missing (inherited as a TIMEOUT)
    assert [cell["status"] for cell in cells] == [None, None, None]
    cells, _ = phase_table(tmp_path, monkeypatch, recorded={}, paths=paths)
    assert [cell["status"] for cell in cells] == ["TIMEOUT", None, "TIMEOUT"]


def test_phase_table_names_a_window_or_a_host_in_its_specs_only_when_given(tmp_path, monkeypatch):
    config = rp.R_B3_WITNESS["lsm"]
    cells, _ = phase_table(tmp_path, monkeypatch)
    pool = [cell["job"]["pool"] for cell in cells if cell["arm"] == "pool"]
    assert pool == [ps._spec("lsm", config, "carried")] * 2
    assert all(list(spec) == ["candidate", "config", "form"] for spec in pool)
    named = phase_table(tmp_path, monkeypatch, window=rp.WINDOW, host="port")[0]
    assert [cell["job"]["pool"] for cell in named if cell["arm"] == "pool"] == pool
    cells, _ = phase_table(tmp_path, monkeypatch, window=0, host="kernel")
    pool_cells = [cell for cell in cells if cell["arm"] == "pool"]
    assert [cell["job"]["pool"] for cell in pool_cells] == [
        ps._spec("lsm", config, "carried", window=0, host="kernel")
    ] * 2
    assert all((cell["window"], cell["host"]) == (0, "kernel") for cell in pool_cells)
    assert [cell for cell in cells if cell["arm"] == "a"][0]["job"]["pool"] is None


def call_spec(tmp_path, target, args=None, name="call.json"):
    return {"kind": "call", "target": target, "args": args or {}, "out": str(tmp_path / name)}


def test_the_call_kind_runs_its_target_and_stamps_the_revision(tmp_path):
    spec = call_spec(tmp_path, POWER_S, {"sigma": 0.0436})
    ps.execute(spec)
    revision = dict(zip(("head", "dirty"), md._revision(), strict=True))
    assert md._load(spec["out"]) == {"S": 31, "capped": False, "revision": revision}
    # a spec without args calls its target with none (crc_json, in the reservoir package, needs one)
    bare = call_spec(tmp_path, "snn_data_lake.reservoir.recurrent_pool:crc_json", name="bare.json")
    bare.pop("args")
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        ps.execute(bare)
    assert not Path(bare["out"]).exists()


@pytest.mark.parametrize(
    "target",
    [
        "os:getcwd",  # outside both packages
        "snn_data_lake.faculties.time_cells:tick",  # inside the project, outside the two prefixes
        "snn_data_lake.instruments_other.module:power_s",  # a lookalike of the instruments prefix
        "snn_data_lake.instruments:power_s",  # the package itself carries no trailing dot
        "snn_data_lake.instruments.pool_selection",  # no function named
        "snn_data_lake.instruments.pool_selection:",  # an empty function name
        "snn_data_lake.instruments.no_such_module:power_s",  # an unknown module
        "snn_data_lake.instruments.pool_selection:no_such_function",  # an unknown function
        "snn_data_lake.instruments.pool_selection:Path",  # a callable the module only imports
        "snn_data_lake.instruments.pool_selection:EXIT_OK",  # not a callable
        "snn_data_lake.instruments.pool_selection:band_d",  # a callable that returns no dict
    ],
)
def test_the_call_kind_refuses_a_target_it_may_not_run(tmp_path, target):
    spec = call_spec(tmp_path, target, {"sigma_dev": 0.01})
    with pytest.raises(ValueError, match="REFUSED: call target"):
        ps.execute(spec)
    assert not Path(spec["out"]).exists()  # nothing is saved for a refused call


def test_the_call_kind_runs_in_a_child_and_a_refused_target_fails_there(tmp_path, monkeypatch):
    # the child imports the code under test whatever the interpreter's installed copy is
    src = Path(ps.__file__).resolve().parents[2]
    inherited = [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(src), *inherited]))
    spec = call_spec(tmp_path, POWER_S, {"sigma": 0.0436})
    assert ps._child(spec) == {"status": "OK", "path": spec["out"]}
    result = md._load(spec["out"])
    assert (result["S"], result["capped"]) == (31, False) and set(result["revision"]) == {
        "head",
        "dirty",
    }
    refused = call_spec(tmp_path, "os:getcwd", name="refused.json")
    status = ps._child(refused)
    assert status["status"] == "FAIL" and "REFUSED: call target 'os:getcwd'" in status["stderr"]
    assert not Path(refused["out"]).exists()
