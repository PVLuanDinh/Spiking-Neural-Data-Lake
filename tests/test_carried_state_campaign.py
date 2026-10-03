"""G1(c)'s campaign module (instruments/carried_state_campaign.py): the frozen constants of
g1c-carried-state.md sections 2.2, 2.3 and 2.12, the seed-set refusals, section 2.3's contrast
row (a degenerate row is NOT MET; a short list halts; skew-flagging) and section 2.4 (ii)'s oracle
check (only the decoder row decides; an incomplete oracle is a blind harness). Pure functions on
hand fixtures: no recording, no pool, no campaign."""

import ast
import inspect
import math
import statistics

import pytest

from snn_data_lake.instruments import carried_state_campaign as ccc
from snn_data_lake.instruments import paired_contrast as pc
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.reservoir import recurrent_pool as rp

BLIND = "blind harness: oracle incomplete"


def green():
    return ccc._jittered(0.05, 0.01, 1)


# ---- the frozen constants (section 2.2, 2.3, 2.12) ----------------------------------------------


def test_the_seed_sets_are_the_texts():
    assert ccc.LIVE_SEEDS == tuple(range(300, 331)) and len(ccc.LIVE_SEEDS) == 31
    assert ccc.RESERVE_SEEDS == tuple(range(331, 362))
    assert ccc.IDENTITY_SEED == 100 and ccc.CAUSALITY_SEED == 203
    assert ccc.GATE_SEEDS == (204, 205, 206, 207)
    assert ccc.GATE_SYNTHETIC_SEEDS == (5005, 5006, 5007, 5008)
    assert (ccc.RB6_SEED, ccc.RB6_RESERVE_SEED) == (6000, 6001)
    assert ccc.SHUFFLE_RED_SEEDS == tuple(range(6100, 6108))
    assert ccc.G2II_CHECK_SEEDS == tuple(range(7000, 7008))
    assert ccc.G2II_DEV_SEEDS == tuple(range(7100, 7104))
    exposed = {
        *range(8),
        *range(100, 104),
        *range(200, 212),
        *range(4242, 4248),
        *range(5001, 5013),
    }
    assert ccc.EXPOSED_SEEDS == frozenset(exposed) and isinstance(ccc.EXPOSED_SEEDS, frozenset)


def test_the_live_seed_count_is_the_power_rule():
    assert len(ccc.LIVE_SEEDS) == ps.power_s(ps.SIGMA_PROXY)["S"] == ps.power_s(0.0436)["S"]
    assert len(ccc.RESERVE_SEEDS) == len(ccc.LIVE_SEEDS)


def test_live_reserve_and_exposed_are_pairwise_disjoint():
    live, reserve = set(ccc.LIVE_SEEDS), set(ccc.RESERVE_SEEDS)
    assert not live & reserve and not live & ccc.EXPOSED_SEEDS and not reserve & ccc.EXPOSED_SEEDS
    fresh = {ccc.RB6_SEED, ccc.RB6_RESERVE_SEED, *ccc.SHUFFLE_RED_SEEDS, *ccc.G2II_CHECK_SEEDS}
    fresh |= set(ccc.G2II_DEV_SEEDS)
    assert not fresh & (live | reserve | ccc.EXPOSED_SEEDS)
    # the seeds G1(b) already used are exposed, read here by their named jobs only
    assert {ccc.IDENTITY_SEED, ccc.CAUSALITY_SEED, *ccc.GATE_SEEDS} <= ccc.EXPOSED_SEEDS
    assert set(ccc.GATE_SYNTHETIC_SEEDS) <= ccc.EXPOSED_SEEDS


def test_the_tuned_configuration_is_the_text_and_lies_in_the_grid():
    assert ccc.TUNED == {"k": 8, "w": 0.25, "lambda": 0.9, "g": 4, "T_in": 1}
    assert rp.canonical("lsm", ccc.TUNED) == ccc.TUNED
    with pytest.raises(ValueError):  # the red: a configuration off the grid is refused
        rp.canonical("lsm", {**ccc.TUNED, "k": 9})


def test_the_bar_the_bound_and_the_design_effect():
    assert ccc.T_CRIT == pc.T_BAR_S31 == 1.6973
    assert ps.t_quantile(0.95, 30) <= ccc.T_CRIT < ps.t_quantile(0.95, 30) + 1e-4
    assert ccc.SKEW_BOUND == -1.0 and ccc.DELTA == ps.DELTA == 0.02


# ---- validate_seed_set --------------------------------------------------------------------------


def test_the_live_seed_set_validates_and_comes_back_a_tuple():
    assert ccc.validate_seed_set(list(range(300, 331))) == ccc.LIVE_SEEDS
    assert ccc.validate_seed_set(ccc.LIVE_SEEDS) == ccc.LIVE_SEEDS


def test_an_exposed_seed_is_refused_wherever_it_sits():
    with pytest.raises(ValueError, match="exposed"):
        ccc.validate_seed_set([*range(300, 330), 203])
    with pytest.raises(ValueError, match="exposed"):
        ccc.validate_seed_set([300, 5005])
    for seed in (0, 7, 100, 211, 4242, 4247, 5001, 5012):
        with pytest.raises(ValueError, match="exposed"):
            ccc.validate_seed_set([300, seed], allow_reserve=True)  # not rescued by allow_reserve
    assert ccc.validate_seed_set([8, 99, 104, 212, 4241, 4248, 5000, 5013]) == (
        8, 99, 104, 212, 4241, 4248, 5000, 5013,
    )  # fmt: skip


def test_a_reserve_seed_is_refused_unless_allowed_and_a_duplicate_never_passes():
    with pytest.raises(ValueError, match="reserve"):
        ccc.validate_seed_set([300, 331])
    assert ccc.validate_seed_set(ccc.RESERVE_SEEDS, allow_reserve=True) == ccc.RESERVE_SEEDS
    for allowed in (False, True):
        with pytest.raises(ValueError, match="duplicate"):
            ccc.validate_seed_set([300, 301, 300], allow_reserve=allowed)


# ---- skewness and sign_counts -------------------------------------------------------------------


def test_skewness_is_the_biased_third_over_second_moment_to_the_three_halves():
    assert math.isclose(ccc.skewness([0.0, 0.0, 3.0]), 2.0 / 2.0**1.5, rel_tol=1e-12)
    assert ccc.skewness([1.0, 2.0, 3.0]) == 0.0
    assert ccc.skewness([-v for v in (0.0, 0.0, 3.0)]) == -ccc.skewness([0.0, 0.0, 3.0])
    xs = [0.5, 0.1, 0.4, 0.4, 0.9, 0.0, 0.3]
    mean = sum(xs) / len(xs)
    m2 = sum((x - mean) ** 2 for x in xs) / len(xs)
    m3 = sum((x - mean) ** 3 for x in xs) / len(xs)
    assert math.isclose(ccc.skewness(xs), m3 / m2**1.5, rel_tol=1e-12)


def test_skewness_of_a_constant_or_empty_sample_is_zero_not_noise():
    assert ccc.skewness([0.03] * 31) == 0.0 and ccc.skewness([0.1] * 7) == 0.0
    assert ccc.skewness([]) == 0.0 and ccc.skewness([4.0]) == 0.0


def test_sign_counts_split_positive_negative_and_zero():
    assert ccc.sign_counts([1.0, -1.0, 0.0, 0.0, 2.0]) == {"positive": 2, "negative": 1, "zero": 2}
    assert ccc.sign_counts([]) == {"positive": 0, "negative": 0, "zero": 0}
    assert ccc.sign_counts(ccc.SYMMETRIC_ZERO_MEAN) == {"positive": 15, "negative": 15, "zero": 1}


# ---- contrast_row (section 2.3) -----------------------------------------------------------------


def test_a_row_is_a_plain_dict_with_every_field_and_its_numbers_match_the_statistic():
    diffs = green()
    row = ccc.contrast_row("green", diffs)
    assert list(row) == [
        "name", "n", "mean", "sd", "t", "bar", "lower_bound", "fpr", "band_population",
        "skewness", "signs", "degenerate", "met", "skew_flagged", "delta",
    ]  # fmt: skip
    assert row["name"] == "green" and row["n"] == 31 and row["bar"] == ccc.T_CRIT
    assert row["delta"] == ccc.DELTA and row["degenerate"] is False
    assert row["met"] is True and row["skew_flagged"] is False
    assert math.isclose(row["mean"], statistics.fmean(diffs))
    assert math.isclose(row["sd"], statistics.stdev(diffs))  # the n - 1 sample sd
    assert row["t"] == pc.paired_t(diffs) > ccc.T_CRIT
    assert math.isclose(row["lower_bound"], row["mean"] - ccc.T_CRIT * row["sd"] / math.sqrt(31))
    assert (row["fpr"], row["band_population"]) == pc.permutation_fpr_detail(diffs, ccc.T_CRIT)
    assert 0.0 < row["fpr"] < 0.10 and row["skewness"] == ccc.skewness(diffs)
    assert row["signs"] == {"positive": 31, "negative": 0, "zero": 0}


@pytest.mark.parametrize("count", [0, 1, 29, 30, 32])
def test_a_list_that_is_not_31_values_halts(count):
    values = (green() * 2)[:count]
    with pytest.raises(ValueError, match=f"{count} differences"):
        ccc.contrast_row("short", values)


def test_n_required_and_the_bar_are_parameters_with_the_campaign_defaults():
    five = [0.1, 0.2, 0.15, 0.12, 0.18]
    row = ccc.contrast_row("five", five, bar=2.132, n_required=5)
    assert row["n"] == 5 and row["bar"] == 2.132 and row["met"]
    with pytest.raises(ValueError):
        ccc.contrast_row("five", five)  # the default demands 31


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, "x"])
def test_a_non_finite_or_non_numeric_difference_halts(bad):
    with pytest.raises(ValueError):
        ccc.contrast_row("bad", [*green()[:30], bad])


def test_a_constant_positive_row_is_degenerate_and_not_met_where_contrast_says_met():
    constant = [0.03] * 31
    assert pc.contrast("constant", constant, ccc.T_CRIT).met is True  # contrast's own rule: the red
    row = ccc.contrast_row("constant +0.03", constant)
    assert row["degenerate"] is True and row["met"] is False and row["skew_flagged"] is False
    assert row["t"] == 0.0 and row["sd"] == 0.0 and row["skewness"] == 0.0 and row["mean"] == 0.03
    assert row["lower_bound"] == 0.03 and row["signs"]["positive"] == 31
    # the exact rate is still printed for a degenerate row (2.3), by the same count the others use
    assert (row["fpr"], row["band_population"]) == pc.permutation_fpr_detail(constant, ccc.T_CRIT)
    assert 0.0 < row["fpr"] < 0.5


def test_an_all_zero_row_is_degenerate_and_not_met_with_nothing_to_flip():
    row = ccc.contrast_row("all zero", [0.0] * 31)
    assert row["degenerate"] and not row["met"] and row["t"] == 0.0
    assert (row["fpr"], row["band_population"]) == (0.0, 0)
    assert row["signs"] == {"positive": 0, "negative": 0, "zero": 31}


def test_the_named_symmetric_zero_mean_fixture_is_not_met():
    fixture = ccc.SYMMETRIC_ZERO_MEAN
    assert len(fixture) == 31 and sum(fixture) == 0.0 and sorted(set(fixture)) == [-0.02, 0.0, 0.02]
    row = ccc.contrast_row("15 pairs of +-0.02 and one 0.0", fixture)
    assert not row["met"] and not row["degenerate"] and abs(row["t"]) < 1e-12
    assert row["signs"] == {"positive": 15, "negative": 15, "zero": 1}
    assert 0.0 < row["fpr"] < 0.10  # sign balance is never a decision rule (Part F)


def test_a_negative_row_and_a_zero_mean_row_are_not_met():
    negative = ccc.contrast_row("negative", [-d for d in green()])
    assert not negative["met"] and negative["t"] < 0.0
    zero_mean = [0.0] * 31
    zero_mean[0], zero_mean[1] = 0.01, -0.01
    assert not ccc.contrast_row("two nonzero", zero_mean)["met"]


def test_a_long_left_tail_on_a_met_row_is_skew_flagged_and_a_symmetric_one_is_not():
    tail = ccc.contrast_row("left tail", ccc._left_tail())
    assert tail["met"] and tail["skewness"] < ccc.SKEW_BOUND and tail["skew_flagged"]
    assert tail["t"] > ccc.T_CRIT
    symmetric = ccc.contrast_row("symmetric met", green())
    assert symmetric["met"] and abs(symmetric["skewness"]) < 1.0 and not symmetric["skew_flagged"]


def test_skew_flagging_needs_a_met_row():
    shifted = ccc.contrast_row("tail below zero", [d - 0.1 for d in ccc._left_tail()])
    assert shifted["skewness"] < ccc.SKEW_BOUND and not shifted["met"]
    assert shifted["skew_flagged"] is False


# ---- oracle_check (section 2.4 (ii)) ------------------------------------------------------------


def test_the_decoder_not_grower_red_the_grower_reads_f64_above_a_and_the_decoder_does_not():
    decoder = ccc._oracle_table(ccc._jittered(-0.005, 0.005, 2))  # -0.01 .. 0.0
    grower = ccc._oracle_table(ccc._jittered(0.05, 0.01, 4))
    result = ccc.oracle_check(decoder, grower)
    assert result["grower"]["met"] is True and result["decoder"]["met"] is False
    assert result["met"] is False  # only the decoder row decides
    assert result["decoder"]["t"] < 0.0 and result["decoder"]["name"].endswith("reference readout")
    assert result["grower"]["name"].endswith("the grower")
    assert result["decoder"]["n"] == result["grower"]["n"] == 31


def test_a_decoder_that_reads_f64_above_a_is_met_with_its_exact_rate_and_band():
    result = ccc.oracle_check(
        ccc._oracle_table(ccc._jittered(0.05, 0.01, 5)),
        ccc._oracle_table(ccc._jittered(0.05, 0.01, 6)),
    )
    decoder = result["decoder"]
    assert result["met"] is True and decoder["met"] and result["grower"]["met"]
    assert decoder["t"] > ccc.T_CRIT and 0.0 < decoder["fpr"] < 0.10
    assert decoder["band_population"] >= 0 and decoder["lower_bound"] > ccc.DELTA


def test_the_oracle_reads_the_difference_per_seed_not_the_means():
    readout = {s: {"f64": 0.60, "a": 0.50} for s in ccc.LIVE_SEEDS}
    readout[300] = {"f64": 0.90, "a": 0.50}  # one big seed
    result = ccc.oracle_check(readout, readout)
    assert result["decoder"]["signs"]["positive"] == 31
    assert math.isclose(result["decoder"]["mean"], (30 * 0.10 + 0.40) / 31, rel_tol=1e-9)


def test_the_symmetric_zero_mean_and_all_zero_and_constant_decoders_are_not_met():
    grower = ccc._oracle_table(green())
    for diffs, degenerate in (
        (ccc.SYMMETRIC_ZERO_MEAN, False),
        ([0.0] * 31, True),
        ([0.03] * 31, True),
    ):
        result = ccc.oracle_check(ccc._oracle_table(diffs), grower)
        assert result["met"] is False and result["decoder"]["degenerate"] is degenerate
        assert result["grower"]["met"] is True


def test_a_missing_seed_halts_the_oracle_as_a_blind_harness():
    complete = ccc._oracle_table(green())
    missing = {s: v for s, v in complete.items() if s != 315}
    for readout, grower in ((missing, complete), (complete, missing)):
        with pytest.raises(ValueError, match=BLIND):
            ccc.oracle_check(readout, grower)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, "x"])
@pytest.mark.parametrize("key", ["f64", "a"])
def test_a_none_nan_or_non_numeric_cell_halts_the_oracle(bad, key):
    complete = ccc._oracle_table(green())
    broken = {**complete, 330: {**complete[330], key: bad}}
    with pytest.raises(ValueError, match=BLIND):
        ccc.oracle_check(broken, complete)
    with pytest.raises(ValueError, match=BLIND):
        ccc.oracle_check(complete, broken)


def test_a_cell_without_its_key_halts_the_oracle():
    complete = ccc._oracle_table(green())
    with pytest.raises(ValueError, match=BLIND):
        ccc.oracle_check({**complete, 301: {"a": 0.5}}, complete)
    with pytest.raises(ValueError, match=BLIND):
        ccc.oracle_check(complete, {**complete, 301: None})


def test_the_default_seeds_are_the_live_seeds_and_another_count_halts():
    complete = ccc._oracle_table(green())
    assert inspect.signature(ccc.oracle_check).parameters["seeds"].default == ccc.LIVE_SEEDS
    with pytest.raises(ValueError, match="30 differences"):
        ccc.oracle_check(complete, complete, seeds=ccc.LIVE_SEEDS[:30])


# ---- the module as a whole ----------------------------------------------------------------------

LEXICON = ("learn", "update", "train", "adapt", "consolidate", "reinforce", "fit", "plast")
LEXICON += ("potentiate", "depress", "calibrate")


def test_no_definition_is_named_into_the_gradient_claim_census():
    tree = ast.parse(inspect.getsource(ccc))
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.ClassDef)]
    assert "contrast_row" in names and "oracle_check" in names  # the walk sees the module's defs
    assert [(n, w) for n in names for w in LEXICON if w in n.lower()] == []


def test_the_self_check_passes_from_any_directory(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.iterdir())  # an autouse fixture may have made its own scratch here
    ccc.main()
    printed = capsys.readouterr().out
    assert printed.startswith("self-check OK") and printed.isascii()
    assert "exact FPR" in printed and "band" in printed and "skew-flagged" in printed
    assert sorted(tmp_path.iterdir()) == before  # it wrote nothing
