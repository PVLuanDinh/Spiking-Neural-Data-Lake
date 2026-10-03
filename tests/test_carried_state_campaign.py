"""G1(c)'s campaign module (instruments/carried_state_campaign.py): the frozen constants of
g1c-carried-state.md sections 2.2, 2.3 and 2.12, the seed-set refusals, section 2.3's contrast
row (a degenerate row is NOT MET; a short list halts; skew-flagging) and section 2.4 (ii)'s oracle
check (only the decoder row decides; an incomplete oracle is a blind harness). Pure functions on
hand fixtures: no recording, no pool, no campaign."""

import ast
import contextlib
import functools
import inspect
import itertools
import json
import math
import statistics

import numpy as np
import pytest

from snn_data_lake.instruments import carried_state_campaign as ccc
from snn_data_lake.instruments import memory_diagnostic as md
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
    assert "shuffle (blocks of 8)" in printed and "Settled invariance holds" in printed
    assert "G-R1: 8 table fixtures exact, 32 input combinations" in printed
    assert sorted(tmp_path.iterdir()) == before  # it wrote nothing


# ---- fold of the P1 panel (Codex gpt-6.1-sol): F1 the oracle validates its seed set
def test_fold_f1_the_oracle_refuses_a_duplicate_seed_set():
    table = {300: {"f64": 0.55, "a": 0.5}, 301: {"f64": 0.56, "a": 0.5}}
    with pytest.raises(ValueError, match="duplicate"):
        ccc.oracle_check(table, table, seeds=(300,) * 30 + (301,))


def test_fold_f1_the_oracle_refuses_an_exposed_seed_and_an_unauthorized_reserve_seed():
    diffs = green()
    exposed = (*ccc.LIVE_SEEDS[:-1], 203)
    table = {s: {"f64": 0.5 + d, "a": 0.5} for s, d in zip(exposed, diffs, strict=True)}
    with pytest.raises(ValueError):
        ccc.oracle_check(table, table, seeds=exposed)
    reserve = {s: {"f64": 0.5 + d, "a": 0.5} for s, d in zip(ccc.RESERVE_SEEDS, diffs, strict=True)}
    with pytest.raises(ValueError):
        ccc.oracle_check(reserve, reserve, seeds=ccc.RESERVE_SEEDS)
    authorized = ccc.oracle_check(reserve, reserve, seeds=ccc.RESERVE_SEEDS, allow_reserve=True)
    assert authorized["met"] and authorized["decoder"]["n"] == 31


# ---- fold of the P2 panel (Codex gpt-6.1-sol): F2 one structural check of a seed list
@pytest.mark.parametrize("seeds", [(), (300, 300), (300.0,), (True,)])
def test_fold_p2_distinct_seeds_refuses_empty_duplicate_and_non_int(seeds):
    with pytest.raises(ValueError):
        ccc.distinct_seeds(seeds)
    with pytest.raises(ValueError):
        ccc.validate_seed_set(seeds)


def test_fold_p2_distinct_seeds_returns_the_tuple():
    assert ccc.distinct_seeds([300, 301]) == (300, 301)


# ---- P3b: the within-block time shuffle, its checks and G-R1's reading (2.5, 2.6, 2.10) -------

BLOCK = (
    8  # the 48-event fixture is shorter than one 256-event block (the module's docstring says so)
)


def canon(value):
    return json.dumps(value, sort_keys=True)


def test_the_shuffle_constants_and_the_field_sets_are_the_text():
    assert (ccc.SHUFFLE_BLOCK, ccc.SHUFFLE_K) == (256, 4)
    assert isinstance(ccc.SHUFFLE_TAG, int) and not isinstance(ccc.SHUFFLE_TAG, bool)
    assert ccc.PERCEPT_FIELDS == ("x", "c", "source", "attended", "prediction", "fallback")
    assert ccc.SLOT_FIELDS == ("t", "tick", "segment", "causal_lp")
    assert not set(ccc.PERCEPT_FIELDS) & set(ccc.SLOT_FIELDS)


def test_the_two_field_sets_are_exactly_the_keys_the_recorder_writes_to_an_event():
    # the row record_stream's observe() appends (the Dict that carries "causal_lp"), plus the
    # "segment" key _segments() adds: the sets restate that code, so a new recorded key fails here
    tree = ast.parse(inspect.getsource(md.record_stream))
    rows = [
        {k.value for k in node.keys if isinstance(k, ast.Constant)}
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and "causal_lp" in {k.value for k in node.keys if isinstance(k, ast.Constant)}
    ]
    assert len(rows) == 1, rows
    assert "segment" in inspect.getsource(md._segments)
    assert {*rows[0], "segment"} == {*ccc.PERCEPT_FIELDS, *ccc.SLOT_FIELDS}


def test_shuffle_seed_is_the_seed_sequence_of_recording_tag_and_k():
    expected = np.random.default_rng(np.random.SeedSequence([17, ccc.SHUFFLE_TAG, 2]))
    assert ccc.shuffle_seed(17, 2).permutation(50).tolist() == expected.permutation(50).tolist()
    draw = [ccc.shuffle_seed(17, k).permutation(50).tolist() for k in (0, 1)]
    assert draw[0] != draw[1]  # k drives the draw
    assert ccc.shuffle_seed(18, 0).permutation(50).tolist() != draw[0]  # so does the seed


def test_blocks_are_consecutive_groups_by_index_within_a_segment():
    stream = ccc._stream_fixture()
    assert ccc.blocks(stream, BLOCK) == [list(range(s, s + 8)) for s in range(0, 48, 8)]
    # a final partial block is its own block (24 warm-up events in groups of 10)
    assert [len(g) for g in ccc.blocks(stream, 10)] == [10, 10, 4, 10, 10, 4]
    # x_1 (segment None) is in no block, and the groups count by index within the segment
    stream["events"][0]["segment"] = None
    first = ccc.blocks(stream, BLOCK)[0]
    assert first == list(range(1, 9)) and 0 not in {i for g in ccc.blocks(stream, BLOCK) for i in g}
    with pytest.raises(ValueError):
        ccc.blocks(stream, 0)


def test_the_half_moved_assertion_fires_on_a_drawn_identity_and_never_re_draws():
    drawn = []
    for k in range(40):
        try:
            drawn.append(ccc.block_permutations(17, k, [2]))
        except AssertionError as error:
            assert "under half" in str(error)
            drawn.append(None)
    assert None in drawn and any(
        d is not None for d in drawn
    )  # a 2-block is the identity half the time
    for permutation in ccc.block_permutations(17, 0, [8, 8, 5, 1]):
        assert sorted(permutation) == list(range(len(permutation)))
    assert ccc.block_permutations(17, 3, [8, 8]) == ccc.block_permutations(17, 3, [8, 8])
    assert ccc.block_permutations(17, 0, [1]) == [
        [0]
    ]  # a block of one cannot move and is not asserted


def test_shuffle_blocks_moves_percepts_whole_and_leaves_every_slot_field_and_sleep_row():
    stream = ccc._sourced_fixture()
    before = canon(stream)
    shuffled = ccc.shuffle_blocks(stream, 0, block=BLOCK)
    assert canon(stream) == before  # the input is untouched
    assert shuffled["shuffle"] == {"seed": 17, "block": BLOCK, "k": 0}
    assert shuffled["seed"] == stream["seed"] and shuffled["config"] == stream["config"]
    assert shuffled["policy"] == stream["policy"] and shuffled["final"] == stream["final"]
    assert canon(shuffled["sleep"]) == canon(stream["sleep"])
    for old, new in zip(stream["events"], shuffled["events"], strict=True):
        assert {k: old[k] for k in ccc.SLOT_FIELDS} == {k: new[k] for k in ccc.SLOT_FIELDS}
    for group in ccc.blocks(stream, BLOCK):
        held = sorted(canon({k: stream["events"][i][k] for k in ccc.PERCEPT_FIELDS}) for i in group)
        now = sorted(
            canon({k: shuffled["events"][i][k] for k in ccc.PERCEPT_FIELDS}) for i in group
        )
        assert held == now
        assert [stream["events"][i]["x"] for i in group] != [
            shuffled["events"][i]["x"] for i in group
        ]
    shuffled["events"][3]["x"][0] = 99.0  # deep copy: no aliasing into the input
    assert canon(stream) == before
    result = ccc.check_shuffle_transform(stream, ccc.shuffle_blocks(stream, 0, block=BLOCK), BLOCK)
    assert result == {"ok": True, "reasons": []}


def test_the_same_seed_and_k_give_the_same_stream_and_another_k_another_one():
    stream = ccc._stream_fixture()
    one = [canon(ccc.shuffle_blocks(stream, 0, block=BLOCK)) for _ in range(2)]
    assert one[0] == one[1]
    digests = {md._source_digest(ccc.shuffle_blocks(stream, k, block=BLOCK)) for k in range(4)}
    assert len(digests) == 4 and md._source_digest(stream) not in digests


def test_x_1_and_every_event_in_no_block_never_move():
    stream = ccc._sourced_fixture()
    stream["events"][0]["segment"] = None
    shuffled = ccc.shuffle_blocks(stream, 1, block=BLOCK)
    assert canon(shuffled["events"][0]) == canon(stream["events"][0])
    assert ccc.check_shuffle_transform(stream, shuffled, BLOCK)["ok"]


def test_the_default_block_is_256_and_a_stream_longer_than_a_block_is_shuffled_in_256s():
    stream = ccc._stream_fixture()
    stream["events"] = [dict(stream["events"][i % 48], t=i + 1, tick=2 * i + 1) for i in range(600)]
    stream["sleep"] = stream["sleep"] * 13
    for event in stream["events"]:
        event["segment"] = "unseen-order"
    assert [len(g) for g in ccc.blocks(stream)] == [256, 256, 88]
    shuffled = ccc.shuffle_blocks(stream, 0)
    assert (
        shuffled["shuffle"]["block"] == 256 and ccc.check_shuffle_transform(stream, shuffled)["ok"]
    )
    # the default boundary is delivery count 256: events 0-255 are deliveries 1-256 (the window)
    runs = {"phi": ccc._percept_phi(stream)}, {"phi": ccc._percept_phi(shuffled)}
    result = ccc.settled_invariance(*runs, stream)
    assert [r["start"] for r in result["per_block"]] == [256, 512] and result["excluded"] == 1
    assert result["ok"] is True


TRANSFORM_REDS = {
    # name -> (the mutant shuffle, the reason check_shuffle_transform must give for it)
    "cross-block": (lambda s: ccc._mutant(s, BLOCK, whole_segment=True), "(x, c) pairs changed"),
    "c-only": (lambda s: ccc._mutant(s, BLOCK, fields=("c",)), "(x, c) pairs changed"),
    "identity": (lambda s: ccc._mutant(s, BLOCK, identity=True), "under half"),
    "slot field tick": (
        lambda s: ccc._mutant(s, BLOCK, fields=(*ccc.PERCEPT_FIELDS, "tick")),
        "slot field",
    ),
    "source stays": (
        lambda s: ccc._mutant(s, BLOCK, fields=("x", "c", "attended", "prediction", "fallback")),
        "did not travel together",
    ),
    "sleep row moved": (lambda s: ccc._sleep_moved(ccc.shuffle_blocks(s, 0, block=BLOCK)), "sleep"),
    "binding dropped": (
        lambda s: ccc._binding_dropped(ccc.shuffle_blocks(s, 0, block=BLOCK)),
        "binding",
    ),
}


@pytest.mark.parametrize("name", sorted(TRANSFORM_REDS))
def test_each_transform_red_fails_the_check_for_its_own_reason(name):
    build, reason = TRANSFORM_REDS[name]
    stream = ccc._sourced_fixture()
    result = ccc.check_shuffle_transform(stream, build(stream), BLOCK)
    assert result["ok"] is False
    assert any(reason in r for r in result["reasons"]), result["reasons"]


def test_the_identity_red_also_leaves_the_digest_unchanged():
    stream = ccc._sourced_fixture()
    result = ccc.check_shuffle_transform(stream, ccc._mutant(stream, BLOCK, identity=True), BLOCK)
    assert any("digest" in r for r in result["reasons"]), result["reasons"]


def test_a_stream_that_is_not_the_transforms_output_for_its_declared_k_fails():
    stream = ccc._sourced_fixture()
    wrong_k = ccc.shuffle_blocks(stream, 1, block=BLOCK)
    wrong_k["shuffle"]["k"] = 0
    result = ccc.check_shuffle_transform(stream, wrong_k, BLOCK)
    assert not result["ok"] and any("not the transform's output" in r for r in result["reasons"])


def test_every_registered_k_passes_on_a_stream_whose_last_block_is_two_events():
    # The P3 panel's F3: the check drew k + 1, a shuffle nobody registered, and failed a valid k = 3
    # because k = 4 moves under half of a 2-event block. Only the declared k is drawn now.
    stream = md.rb6_stream(81111)
    for i in (1198, 1199):
        stream["events"][i]["segment"] = "last-two"
    sizes = [len(group) for group in ccc.blocks(stream)]
    assert sizes == [256, 256, 256, 256, 174, 2]
    with pytest.raises(AssertionError, match="under half"):
        ccc.block_permutations(81111, ccc.SHUFFLE_K, sizes)  # the draw the old check made for k = 3
    for k in range(ccc.SHUFFLE_K):
        shuffled = ccc.shuffle_blocks(stream, k)
        assert ccc.check_shuffle_transform(stream, shuffled) == {"ok": True, "reasons": []}


def test_k_drives_the_permutation_on_the_fixture_without_the_check_drawing_k_plus_one():
    stream = ccc._sourced_fixture()
    drawn = [ccc.shuffle_blocks(stream, k, block=BLOCK) for k in range(ccc.SHUFFLE_K)]
    assert len({md._source_digest(s) for s in drawn}) == ccc.SHUFFLE_K  # k changes the stream
    shuffled = drawn[1]
    assert ccc.check_shuffle_transform(stream, shuffled, BLOCK)["ok"]
    # a stream declaring another k than its own draw is refused for that, never for what a
    # neighbouring k would do
    shuffled["shuffle"]["k"] = 2
    result = ccc.check_shuffle_transform(stream, shuffled, BLOCK)
    assert not result["ok"] and any("not the transform's output" in r for r in result["reasons"])
    assert not any("k + 1" in r for r in result["reasons"])


def test_an_event_key_in_neither_set_refuses_the_transform_and_the_check():
    stream = ccc._sourced_fixture()
    stream["events"][5]["extra"] = 1
    with pytest.raises(ValueError, match="'extra'"):
        ccc.shuffle_blocks(stream, 0, block=BLOCK)
    clean = ccc._sourced_fixture()
    shuffled = ccc.shuffle_blocks(clean, 0, block=BLOCK)
    shuffled["events"][5]["extra"] = 1
    result = ccc.check_shuffle_transform(clean, shuffled, BLOCK)
    assert not result["ok"] and any("unknown key" in r for r in result["reasons"])


def test_a_block_whose_source_pointers_are_not_one_contiguous_pass_is_refused():
    jumped = ccc._sourced_fixture()
    jumped["events"][9]["source"]["vision"]["pointer"] = 100  # a gap inside block [8..15]
    with pytest.raises(ValueError, match="contiguous pass"):
        ccc.shuffle_blocks(jumped, 0, block=BLOCK)
    backward = ccc._sourced_fixture()
    backward["events"][10]["source"]["audio"]["pointer"] = 3  # a block that goes back
    with pytest.raises(ValueError, match="audio pointers"):
        ccc.shuffle_blocks(backward, 0, block=BLOCK)
    interleaved = ccc._sourced_fixture()
    interleaved["events"][10]["segment"] = "unseen-order"  # a segment that is not one run
    with pytest.raises(ValueError, match="not consecutive"):
        ccc.shuffle_blocks(interleaved, 0, block=BLOCK)
    assert ccc.shuffle_blocks(
        ccc._sourced_fixture(), 0, block=BLOCK
    )  # the unbroken fixture is fine


def test_a_shuffled_stream_is_never_shuffled_again_and_a_stream_without_blocks_is_refused():
    stream = ccc._stream_fixture()
    with pytest.raises(ValueError, match="already shuffled"):
        ccc.shuffle_blocks(ccc.shuffle_blocks(stream, 0, block=BLOCK), 1, block=BLOCK)
    for event in stream["events"]:
        event["segment"] = None
    with pytest.raises(ValueError, match="no segment"):
        ccc.shuffle_blocks(stream, 0, block=BLOCK)


# ---- settled_invariance -------------------------------------------------------------------------


def test_settled_invariance_holds_on_post_boundary_blocks_and_never_counts_a_pre_boundary_one():
    stream = ccc._stream_fixture()
    shuffled = ccc.shuffle_blocks(stream, 0, block=BLOCK)
    plain_run, shuffled_run = {"phi": ccc._percept_phi(stream)}, {"phi": ccc._percept_phi(shuffled)}
    result = ccc.settled_invariance(plain_run, shuffled_run, stream, boundary=16, block=BLOCK)
    assert result["ok"] is True and result["excluded"] == 2  # events 0-15 are deliveries 1-16
    assert [row["start"] for row in result["per_block"]] == [16, 24, 32, 40]
    assert all(r["multiset_equal"] and r["sequence_differs"] for r in result["per_block"])
    # a block that STRADDLES the boundary is not "wholly after" it: delivery 17 > 16, 20 is not
    straddle = ccc.settled_invariance(plain_run, shuffled_run, stream, boundary=20, block=BLOCK)
    assert [row["start"] for row in straddle["per_block"]] == [24, 32, 40] and straddle[
        "excluded"
    ] == 3
    # a boundary past every delivery leaves nothing to count, and an empty proof is no proof
    assert ccc.settled_invariance(plain_run, shuffled_run, stream, boundary=48, block=BLOCK) == {
        "ok": False, "per_block": [], "excluded": 6,
    }  # fmt: skip


def test_a_pre_boundary_block_that_differs_is_never_counted():
    stream = ccc._stream_fixture()
    shuffled = ccc.shuffle_blocks(stream, 0, block=BLOCK)
    plain_run, shuffled_run = {"phi": ccc._percept_phi(stream)}, {"phi": ccc._percept_phi(shuffled)}
    shuffled_run["phi"][3] = 12345  # inside block [0..7], before delivery 16
    assert ccc.settled_invariance(plain_run, shuffled_run, stream, boundary=16, block=BLOCK)["ok"]
    shuffled_run["phi"][27] = 12345  # inside block [24..31], after it: counted, and it breaks
    broken = ccc.settled_invariance(plain_run, shuffled_run, stream, boundary=16, block=BLOCK)
    assert not broken["ok"] and not broken["per_block"][1]["multiset_equal"]


def test_a_carried_like_pair_whose_multisets_differ_is_not_ok():
    stream = ccc._stream_fixture()
    shuffled = ccc.shuffle_blocks(stream, 0, block=BLOCK)
    result = ccc.settled_invariance(
        {"phi": ccc._history_phi(stream)},
        {"phi": ccc._history_phi(shuffled)},
        stream,
        boundary=16,
        block=BLOCK,
    )
    assert result["ok"] is False and not all(r["multiset_equal"] for r in result["per_block"])


def test_settled_invariance_needs_one_phi_per_event_and_a_sequence_that_moved():
    stream = ccc._stream_fixture()
    run = {"phi": ccc._percept_phi(stream)}
    with pytest.raises(ValueError, match="phi"):
        ccc.settled_invariance(run, {"phi": run["phi"][:-1]}, stream, boundary=16, block=BLOCK)
    same = ccc.settled_invariance(run, run, stream, boundary=16, block=BLOCK)  # nothing moved
    assert same["ok"] is False and not any(r["sequence_differs"] for r in same["per_block"])


# ---- survives / incremental ---------------------------------------------------------------------


def test_survives_is_true_for_a_planted_order_independent_gain_and_false_for_a_destroyed_one():
    assert ccc.survives(ccc._jittered(0.05, 0.01, 7)) is True
    assert ccc.survives(ccc._jittered(-0.002, 0.01, 8)) is False
    assert ccc.survives([0.03] * 31) is False  # degenerate is never met (2.3)
    with pytest.raises(ValueError, match="30 differences"):
        ccc.survives(ccc._jittered(0.05, 0.01, 7)[:30])


def test_survives_takes_the_bar_and_the_count_of_its_caller_and_keeps_the_campaigns_defaults():
    gain = [0.05 + 0.002 * i for i in range(8)]
    assert ccc.survives(gain, bar=pc.T_BAR, n_required=8) is True  # 8 seeds at Part F's bar
    assert ccc.survives([-g for g in gain], bar=pc.T_BAR, n_required=8) is False
    with pytest.raises(ValueError, match="8 differences, 31 required"):
        ccc.survives(gain)  # the campaign's own call still takes 31 at T_CRIT
    with pytest.raises(ValueError, match="31 differences, 8 required"):
        ccc.survives(ccc._jittered(0.05, 0.01, 7), bar=pc.T_BAR, n_required=8)


def test_incremental_is_the_contrast_over_d_minus_d_shuf():
    d = [0.08 + 0.001 * i for i in range(31)]
    d_shuf = [0.03] * 31
    row = ccc.incremental(d, d_shuf)
    assert row["n"] == 31 and row["met"] and math.isclose(row["mean"], statistics.fmean(d) - 0.03)
    assert row["name"].startswith("D - D_shuf")
    assert (
        not ccc.incremental(d, d)["met"] and ccc.incremental(d, d)["degenerate"]
    )  # the shuffle ate all of D
    with pytest.raises(ValueError):
        ccc.incremental(d, d_shuf[:30])


# ---- g_r1_reading (2.5) -------------------------------------------------------------------------

MET = ccc.contrast_row("met", ccc._jittered(0.05, 0.01, 1))
NOT_MET = ccc.contrast_row("not met", ccc._jittered(-0.005, 0.005, 2))
FLAGGED = ccc.contrast_row("met, left tail", ccc._left_tail())
REFUSED = "REFUSED (blind harness)"


def reading(verdict, wording=None, skew=False):
    return {"verdict": verdict, "wording": wording, "skew_flagged": skew}


def inputs(*, ok=True, primary=MET, history=NOT_MET, shuf=NOT_MET, sighted=True):
    return {
        "blind_harness_ok": ok,
        "primary": primary,
        "stream_history": history,
        "d_shuf": shuf,
        "shuffle_sighted": sighted,
    }


G_R1_ROWS = {
    # one fixture per row of 2.5's table, built so a reading that skips its row reads otherwise
    "1 REFUSED beats a met primary, a met D_shuf and a sighted control": (
        inputs(ok=False, shuf=MET),
        reading(REFUSED),
    ),
    "2 pool-null": (inputs(primary=NOT_MET, history=MET, shuf=MET), reading("RED", "pool-null")),
    "2 stream-null": (
        inputs(primary=NOT_MET, history=NOT_MET, shuf=MET),
        reading("RED", "stream-null"),
    ),
    "3 survives, control sighted": (inputs(shuf=MET), reading("RED", "survives the shuffle")),
    "3 survives, control blind": (
        inputs(shuf=MET, sighted=False),
        reading("RED", "survives the shuffle"),
    ),
    "4 not read": (inputs(sighted=False), reading("NOT READ")),
    "5 pass": (inputs(), reading("PASS")),
    "5 pass, skew-flagged": (inputs(primary=FLAGGED), reading("PASS", None, True)),
}


@pytest.mark.parametrize("name", sorted(G_R1_ROWS))
def test_g_r1_reads_each_row_of_the_table_to_its_exact_output(name):
    given, expected = G_R1_ROWS[name]
    assert ccc.g_r1_reading(**given) == expected


def test_a_met_primary_with_skew_below_minus_one_carries_the_flag_in_every_row_it_reaches():
    assert FLAGGED["skew_flagged"] and not NOT_MET["skew_flagged"]
    flag = {"skew": True}
    assert ccc.g_r1_reading(**inputs(primary=FLAGGED, ok=False)) == reading(REFUSED, **flag)
    assert ccc.g_r1_reading(**inputs(primary=FLAGGED, shuf=MET)) == reading(
        "RED", "survives the shuffle", **flag
    )
    assert ccc.g_r1_reading(**inputs(primary=FLAGGED, sighted=False)) == reading("NOT READ", **flag)
    assert ccc.g_r1_reading(**inputs(primary=FLAGGED)) == reading("PASS", **flag)
    # a primary that is not met cannot be flagged
    assert ccc.g_r1_reading(**inputs(primary=NOT_MET))["skew_flagged"] is False


def test_pass_needs_a_sighted_harness_a_met_primary_a_sighted_shuffle_and_an_unmet_d_shuf():
    # every combination of the five inputs against an independent statement of the table
    for ok, primary, history, shuf, sighted in itertools.product(
        (True, False), (MET, NOT_MET), (MET, NOT_MET), (MET, NOT_MET), (True, False)
    ):
        got = ccc.g_r1_reading(
            **inputs(ok=ok, primary=primary, history=history, shuf=shuf, sighted=sighted)
        )
        if not ok:
            want = reading(REFUSED)
        elif primary is NOT_MET:
            want = reading("RED", "pool-null" if history is MET else "stream-null")
        elif shuf is MET:
            want = reading("RED", "survives the shuffle")
        else:
            want = reading("PASS" if sighted else "NOT READ")
        assert got == want
        passes = ok and primary is MET and shuf is NOT_MET and sighted
        assert (got["verdict"] == "PASS") is bool(passes)  # no blind-control combination reads PASS


def test_a_reading_that_ignores_any_one_input_is_caught_by_a_row_fixture():
    ignoring = (
        (
            "blind_harness_ok",
            True,
            "1 REFUSED beats a met primary, a met D_shuf and a sighted control",
        ),
        ("primary", MET, "2 stream-null"),
        ("primary", NOT_MET, "5 pass"),
        ("stream_history", NOT_MET, "2 pool-null"),
        ("stream_history", MET, "2 stream-null"),
        ("d_shuf", NOT_MET, "3 survives, control blind"),
        ("shuffle_sighted", True, "4 not read"),
        ("shuffle_sighted", False, "5 pass"),
    )
    for name, constant, row in ignoring:
        given = inputs()
        given.update(G_R1_ROWS[row][0])
        assert ccc.g_r1_reading(**{**given, name: constant}) != ccc.g_r1_reading(**given), (
            name,
            row,
        )


def test_g_r1_reading_refuses_inputs_that_are_not_a_row_or_a_bool():
    with pytest.raises(ValueError, match="blind_harness_ok"):
        ccc.g_r1_reading(**inputs(ok=1))
    with pytest.raises(ValueError, match="primary"):
        ccc.g_r1_reading(**inputs(primary=True))
    with pytest.raises(ValueError, match="stream_history"):
        ccc.g_r1_reading(**inputs(primary=NOT_MET, history=None))
    with pytest.raises(ValueError, match="d_shuf"):
        ccc.g_r1_reading(**inputs(shuf={"t": 3.0}))
    with pytest.raises(ValueError, match="shuffle_sighted"):
        ccc.g_r1_reading(**inputs(sighted=None))
    # an input the deciding row never reads may be absent: the REFUSED row needs only the harness
    assert ccc.g_r1_reading(**inputs(ok=False, shuf=None, history=None, sighted=None)) == reading(
        REFUSED
    )


# ---- one real replay on the 48-event fixture ----------------------------------------------------


def test_a_shuffled_stream_replays_as_an_ordinary_stream_under_the_revision_binding():
    stream = ccc._stream_fixture()
    shuffled = ccc.shuffle_blocks(stream, 0, block=BLOCK)
    assert md._source_digest(shuffled) != md._source_digest(stream)
    plain_run = md.replay(stream, "a")
    shuffled_run = md.replay(
        shuffled, "a"
    )  # arm "a" feeds x alone: a Settled-like, history-free phi
    md._validate_run(plain_run, stream, stream, "a", "causal")
    md._validate_run(shuffled_run, shuffled, shuffled, "a", "causal")  # the revision binding holds
    assert (
        shuffled_run["source_digest"] == md._source_digest(shuffled) != plain_run["source_digest"]
    )
    assert shuffled_run["revision"] == plain_run["revision"] == {
        "head": stream["final"]["head"], "dirty": stream["final"]["dirty"],
    }  # fmt: skip
    # the shuffled replay feeds the same phi multiset per block in another order
    invariance = ccc.settled_invariance(plain_run, shuffled_run, stream, boundary=16, block=BLOCK)
    assert invariance["ok"] is True, invariance


# ---- 2.10's shuffle controls on REAL replays (the P3 panel's F2) ---------------------------------


@pytest.fixture(scope="module")
def pooled():
    stream = ccc._sourced_fixture()
    return stream, ccc.shuffle_blocks(stream, 0, block=BLOCK)


def test_settled_holds_its_invariance_on_real_pool_replays_of_the_fixture(pooled):
    stream, shuffled = pooled
    result = ccc._pool_invariance(stream, shuffled, "settled", block=BLOCK)
    assert result["ok"] is True and result["excluded"] == 2
    assert [row["start"] for row in result["per_block"]] == [16, 24, 32, 40]
    assert all(row["multiset_equal"] and row["sequence_differs"] for row in result["per_block"])


@pytest.mark.parametrize(
    ("form", "carrier"),
    [("carried", None), ("settled", "v"), ("settled", "s")],
    ids=["the Carried red", "Settled skipping v", "Settled skipping s"],
)
def test_carried_and_a_settled_skipping_a_carrier_fail_the_invariance_on_real_pool_replays(
    pooled, form, carrier
):
    stream, shuffled = pooled
    result = ccc._pool_invariance(stream, shuffled, form, block=BLOCK, carrier=carrier)
    assert result["ok"] is False and len(result["per_block"]) == 4
    assert ccc._equal_blocks(result) == 0  # the multiset moved in every post-boundary block


def test_the_pool_controls_report_their_counts(pooled):
    stream, shuffled = pooled
    assert ccc._check_pool_controls(stream, shuffled, BLOCK) == {
        "blocks": 4, "settled": 4, "carried": 0, "skipped": {"v": 0, "s": 0},
    }  # fmt: skip


def test_a_pool_control_whose_red_cannot_fire_is_refused(pooled, monkeypatch):
    stream, shuffled = pooled
    # a skip-a-carrier fault that skips nothing leaves an honest Settled, which holds the invariance
    monkeypatch.setattr(md, "_skipped_carrier", lambda carrier: contextlib.nullcontext())
    with pytest.raises(AssertionError, match="skipping"):
        ccc._check_pool_controls(stream, shuffled, BLOCK)
    monkeypatch.undo()
    # a Carried that resets (the form's reset standing in for it) holds the invariance too
    original = ccc._pool_invariance

    def resetting(stream, shuffled, form, **kwargs):
        return original(stream, shuffled, "settled" if form == "carried" else form, **kwargs)

    monkeypatch.setattr(ccc, "_pool_invariance", resetting)
    with pytest.raises(AssertionError, match="Carried held"):
        ccc._check_pool_controls(stream, shuffled, BLOCK)


@pytest.fixture(scope="module")
def recordings():
    return [md.rb6_stream(seed) for seed in ccc.SHUFFLE_RED_SEEDS]


def _arms():
    return tuple(functools.partial(ccc._arm_rows, arm=arm) for arm in ("a", "e1", "e1-0"))


@pytest.mark.parametrize("arm", ["a", "e1", "e1-0"])
def test_the_decoder_alone_reads_what_replays_reference_readout_reads(recordings, arm):
    shuffled = ccc.shuffle_blocks(recordings[0], 0)
    run = md.replay(shuffled, arm)
    rows = ccc._arm_rows(shuffled, arm)
    pairs = list(zip(rows, (e["c"] for e in shuffled["events"]), strict=True))
    with rp._blas_threads(1):
        guesses = md._drive_readout(md.ReferenceReadout(), pairs)
    assert guesses == run["readout"]["predictions"]
    assert ccc._decoder_accuracy(shuffled, rows) == ps.recording_accuracy(
        ps.digest(run, shuffled)["readout"]
    )


def test_the_lookahead_red_instrument_feeds_the_decoder_the_current_category():
    stream = ccc._stream_fixture()
    rows = ccc._lookahead_rows(stream)
    cats = [event["c"] for event in stream["events"]]
    assert len(rows) == 48 and {len(row) for row in rows} == {64}
    assert all(rows[i] == md.onehot_32(cats[i + 1]) + [0.0] * 32 for i in range(47))
    assert rows[47] == [0.0] * 64  # nothing follows the last event
    # the reference readout predicts c_i from row i - 1: after its first solve it reads the category
    # of the event it is asked about, in the stream as recorded and in a shuffled one alike
    shuffled = ccc.shuffle_blocks(stream, 0, block=BLOCK)
    for s in (stream, shuffled):
        assert ccc._decoder_accuracy(s, ccc._lookahead_rows(s)) == 1.0
    assert ccc._decoder_accuracy(shuffled, ccc._arm_rows(shuffled, "a")) < 1.0


def test_a_planted_current_category_gain_survives_the_shuffle_and_the_history_gain_does_not(
    recordings,
):
    arm_a, e1, e1_0 = _arms()
    bar, n = pc.T_BAR, len(recordings)
    assert n == 8 and ccc.SHUFFLE_RED_SEEDS == tuple(range(6100, 6108)) and bar == 1.895
    current = ccc._decoder_gaps(recordings, ccc._lookahead_rows, arm_a, shuffles=ccc.SHUFFLE_K)
    assert ccc.survives(current, bar=bar, n_required=n) is True
    assert ccc.contrast_row("current", current, bar=bar, n_required=n)["t"] > 100.0
    plain = ccc._decoder_gaps(recordings, e1, e1_0, shuffles=0)  # R-b6 unshuffled: the sighted half
    assert ccc.survives(plain, bar=bar, n_required=n) is True
    assert statistics.fmean(plain) >= md.RB6_BAR
    history = ccc._decoder_gaps(recordings, e1, e1_0, shuffles=ccc.SHUFFLE_K)
    row = ccc.contrast_row("history", history, bar=bar, n_required=n)
    assert ccc.survives(history, bar=bar, n_required=n) is False
    assert row["t"] < 0.0 and statistics.fmean(history) < md.RB6_BAR


def test_the_decoder_decisions_report_their_numbers():
    numbers = ccc._check_decoder_decisions()
    assert numbers["history_shuffles"] == 1
    assert numbers["current_t"] > numbers["unshuffled_t"] > pc.T_BAR > numbers["history_t"]


def test_the_self_check_goes_red_when_the_decoder_always_predicts_zero(monkeypatch):
    monkeypatch.setattr(md.ReferenceReadout, "predict", lambda self: 0)
    with pytest.raises(AssertionError, match="planted current-category gain"):
        ccc._check_shuffle()  # every gain is 0: degenerate, NOT MET


def test_the_self_check_goes_red_when_the_pool_is_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("the pool is unavailable")

    monkeypatch.setattr(rp, "make_pool", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        ccc._check_shuffle()


def test_the_orchestrators_one_liner_no_longer_leaves_the_self_check_green(monkeypatch):
    monkeypatch.setattr(md.ReferenceReadout, "predict", lambda self: 0)

    def unavailable(*args, **kwargs):
        raise RuntimeError("the pool is unavailable")

    monkeypatch.setattr(rp, "make_pool", unavailable)
    with pytest.raises((AssertionError, RuntimeError)):
        ccc._check_shuffle()


# ---- G1(c) P4b: section 2.9's migration gate -----------------------------------------------------


def test_the_migration_tolerance_is_the_frozen_number_and_the_default_of_within():
    assert ccc.MIGRATION_TOL == 0.005
    assert inspect.signature(ccc.within).parameters["tol"].default == 0.005
    assert list(inspect.signature(ccc.migration_reading).parameters) == [
        "parity_ok",
        "d_live_host",
        "d_live_port",
        "d_task_host",
        "d_task_port",
        "r_b3_ok",
        "r_b4_ok",
    ]


@pytest.mark.parametrize(
    ("d_host", "d_port", "expected"),
    [
        (0.005, 0.0, True),  # inclusive at the tolerance, either sign, either side
        (0.0, 0.005, True),
        (-0.005, 0.0, True),
        (0.0049, 0.0, True),  # the boundary reds
        (0.1049, 0.1, True),
        (0.0051, 0.0, False),
        (0.1051, 0.1, False),
        (0.0, 0.0051, False),
        (np.float64(0.0049), np.float32(0.0), True),
        (math.nan, 0.0, False),  # never a pass
        (0.0, math.nan, False),
        (math.nan, math.nan, False),
        (None, 0.0, False),
        (0.0, None, False),
        (None, None, False),
        (math.inf, 0.0, False),
        (math.inf, math.inf, False),
        (-math.inf, math.inf, False),
        ("0.0", 0.0, False),
        (True, 0.0, False),
        ([0.0], 0.0, False),
    ],
)
def test_within_is_inclusive_at_the_tolerance_and_never_passes_what_it_cannot_read(
    d_host, d_port, expected
):
    assert ccc.within(d_host, d_port) is expected


def test_within_takes_its_tolerance_as_a_finite_non_negative_number():
    assert ccc.within(0.02, 0.0, 0.02) and not ccc.within(0.0201, 0.0, 0.02)
    for tol in (-0.001, math.nan, None, math.inf, "0.005", True):
        with pytest.raises(ValueError, match="tol"):
            ccc.within(0.0, 0.0, tol)


GOOD = {
    "parity_ok": True,
    "d_live_host": 0.012,
    "d_live_port": 0.010,
    "d_task_host": 0.090,
    "d_task_port": 0.0892,
    "r_b3_ok": True,
    "r_b4_ok": True,
}


def test_migration_reading_is_hosted_only_when_every_part_of_the_gate_holds():
    assert ccc.migration_reading(**GOOD) == "hosted"
    edge = dict(GOOD, d_live_host=0.0049, d_live_port=0.0, d_task_host=0.0049, d_task_port=0.0)
    assert ccc.migration_reading(**edge) == "hosted"
    assert ccc.migration_reading(**dict(GOOD, parity_ok=np.True_, r_b3_ok=np.True_)) == "hosted"
    # a conjunct needs BOTH D legs: 0.0051 on either leg (or both) is the port
    for live, task in ((0.0049, 0.0051), (0.0051, 0.0049), (0.0051, 0.0051)):
        row = dict(GOOD, d_live_host=live, d_live_port=0.0, d_task_host=task, d_task_port=0.0)
        assert ccc.migration_reading(**row) == "port", (live, task)


@pytest.mark.parametrize(
    "change",
    [
        {
            "d_task_host": 0.0051,
            "d_task_port": 0.0,
        },  # the hosted-Reset red: live passes, D_task fails
        {"d_live_host": 0.0051, "d_live_port": 0.0},
        {"d_live_host": math.nan},
        {"d_live_port": None},
        {"d_task_host": math.nan},
        {"d_task_port": None},
        {"d_task_host": math.inf},
        {"parity_ok": False},
        {"r_b3_ok": False},
        {"r_b4_ok": False},
        {"parity_ok": None},  # a missing result is no pass
        {"r_b3_ok": 1},  # a truthy non-bool is no pass
        {"r_b4_ok": "True"},
        {"parity_ok": np.False_},
    ],
)
def test_migration_reading_is_the_port_for_every_failing_missing_or_unreadable_input(change):
    assert ccc.migration_reading(**dict(GOOD, **change)) == "port"


def test_a_reading_that_ignores_one_input_is_caught_by_a_row_that_input_alone_fails():
    # each input held at a passing constant flips its own failing row to "hosted"
    rows = (
        ("parity_ok", True, {"parity_ok": False}),
        ("r_b3_ok", True, {"r_b3_ok": False}),
        ("r_b4_ok", True, {"r_b4_ok": False}),
        ("d_live_host", 0.0, {"d_live_host": 0.0051, "d_live_port": 0.0}),
        ("d_live_port", 0.0051, {"d_live_host": 0.0051, "d_live_port": 0.0}),
        ("d_task_host", 0.0, {"d_task_host": 0.0051, "d_task_port": 0.0}),
        ("d_task_port", 0.0051, {"d_task_host": 0.0051, "d_task_port": 0.0}),
    )
    for key, constant, change in rows:
        failing = dict(GOOD, **change)
        assert ccc.migration_reading(**failing) == "port", key
        assert ccc.migration_reading(**{**failing, key: constant}) == "hosted", key


def test_the_migration_check_runs_its_reds_and_the_hosted_equality_on_the_tuned_configuration():
    result = ccc._check_migration()
    assert result["equal"] == {"carried": True, "settled": True} and result["rows"] == 10
