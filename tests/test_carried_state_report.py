"""G1(c)'s report stage and golden check (instruments/carried_state_report.py), g1c-carried-state.md
2.3, 2.5 and 2.11: the pure assembly over a hand-built summary set (one clean head, COMPLETED
stages, every contrast row, the G-R1 reading of every row of 2.5's table, the recomputed rows held
equal to the stored ones), the markdown, the digests, the golden check (deterministic, and moved by
a one-ulp change to the pool's leak on the surviving host) and its CLI.

Nothing here records or builds a Brain; no live seed (300-361), synthetic seed 6000 or learner seed
7000-7007 is run. The summaries are hand numbers over the live seeds' names (300-330 are only keys
here: no recording exists), the golden's replays are the 48-event fixture."""

import copy
import json
import statistics

import pytest

from snn_data_lake.instruments import carried_state_campaign as cs
from snn_data_lake.instruments import carried_state_driver as drv
from snn_data_lake.instruments import carried_state_report as rep
from snn_data_lake.instruments import carried_state_stages as st
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.reservoir import recurrent_pool as rp

SEEDS = [str(seed) for seed in cs.LIVE_SEEDS]
_RAW = [((i * 7) % 11 - 5) / 500 for i in range(31)]
JITTER = [x - statistics.fmean(_RAW) for x in _RAW]  # zero-mean, non-degenerate (a symmetric null)
OTHER = [((i * 5) % 13 - 6) / 400 for i in range(31)]  # a second spread, so two rows never coincide
OTHER = [x - statistics.fmean(OTHER) for x in OTHER]


def per_seed(center, spread=JITTER):
    return {seed: center + spread[i] for i, seed in enumerate(SEEDS)}


def stamped(body=None, **fields):
    return {"status": "COMPLETED", "head": "h1", "dirty": False, **fields, **(body or {})}


def segment(accuracy):
    return {
        "n": 944,
        "available": True,
        "accuracy": accuracy,
        "coverage": 1.0,
        "prediction_entropy": 1.2,
        "prior_fallback": 0.0,
    }


def cells():
    return {
        form: {
            seed: {
                "status": "OK",
                "readout": {"unseen-order": segment(0.5 + JITTER[i])},
                "grower": {"unseen-order": segment(0.4 + OTHER[i])},
                "degeneracy": {"rows": 944, "constant_fraction": 0.1, "distinct_rows": 900},
                "startup": {"transient": {"0": 0.1}, "flagged": i % 2 == 0},
            }
            for i, seed in enumerate(SEEDS)
        }
        for form in rp.FORMS
    }


def fit_cells():
    """The cells of Carried and Settled alone, the variants a control or a transport arm replays."""
    return {form: seeds for form, seeds in cells().items() if form in ps.FIT_FORMS}


BUNDLE = {
    "which": "live",
    "live": list(cs.LIVE_SEEDS),
    "gate": list(cs.GATE_SEEDS),
    "gate_synthetic": list(cs.GATE_SYNTHETIC_SEEDS),
    "rb6": cs.RB6_SEED,
}


def references(oracle_gain, history_gain):
    table = {
        learner: {
            seed: {
                "a": 0.5,
                "f64": 0.5 + oracle_gain + (JITTER if learner == "readout" else OTHER)[i],
                "e1": 0.5 + history_gain + (OTHER if learner == "readout" else JITTER)[i],
                "e1-0": 0.5,
            }
            for i, seed in enumerate(SEEDS)
        }
        for learner in st.LEARNERS
    }
    ints = {
        learner: {int(seed): row for seed, row in rows.items()} for learner, rows in table.items()
    }
    return stamped(
        {
            "accuracy": table,
            "references": {  # each arm's per-seed diagnostic table, as ps._reference files it
                arm: {
                    "per_seed": {
                        seed: {
                            "status": "OK",
                            "readout": {"unseen-order": segment(0.5)},
                            "grower": {},
                        }
                        for seed in SEEDS
                    }
                }
                for arm in ("a", "f64", "e1", "e1-0")
            },
            "oracle": ps._plain(cs.oracle_check(ints["readout"], ints["grower"])),
            "stream_history": {
                learner: ps._plain(
                    cs.contrast_row(
                        f"stream history (e1) - (e1-0) through the {learner}",
                        [
                            table[learner][seed]["e1"] - table[learner][seed]["e1-0"]
                            for seed in SEEDS
                        ],
                    )
                )
                for learner in st.LEARNERS
            },
        }
    )


def response(center):
    """A hand-built per-seed input response (driven units, drive mean and sd) for one variant."""
    return {
        seed: {"driven": 256, "mean": center + i / 1000, "sd": 1.0 + i / 10000}
        for i, seed in enumerate(SEEDS)
    }


def summaries(
    *,
    oracle_gain=0.1,
    history_gain=0.0,
    d=0.05,
    d_shuf=0.0,
    rb6_ok=True,
    sighted=True,
    host="kernel",
    controls_host=None,
    d_values=None,
):
    """A hand-built, internally consistent summary set of the eleven stages."""
    primary_d = d_values or per_seed(d)
    arms = {
        arm: {
            "d": {"readout": per_seed(0.02 * k, OTHER)},
            "records": {seed: {"transport": {"form": arm}, "encoder": 7} for seed in SEEDS},
            "cells": fit_cells(),
            "input_response": response(0.1 * (k + 1)),
        }
        for k, arm in enumerate(md.TRANSPORT_ARMS)
    }
    body = {
        "record": stamped(live_seeds=list(cs.LIVE_SEEDS), seed_set="live", bundle=BUNDLE),
        "mechanism": stamped(refused={}, configuration={"verdicts": {"R-b3": True}}),
        "instrument": stamped(
            rb6_ok=rb6_ok, rb6_clauses={"positive": rb6_ok, "history_free": True}
        ),
        "references": references(oracle_gain, history_gain),
        "construct": stamped(run=True, verdict=True, item_3="RUN", reason=None),
        "migrate": stamped(
            reading="hosted" if host == "kernel" else "port",
            use=host,
            conjuncts={"parity": host == "kernel", "r_b3": True, "r_b4": True},
            mean={"port": {"live": 0.0, "task": 0.1}, "kernel": {"live": 0.0, "task": 0.1}},
            machinery_red={"fired": True},
            port_vs_g1b={"port": [0.0] * 4},
            quantization={"rows": {"12": {"D_live": 0.0, "D_task": 0.1}}},
            parity_horizon={
                "report_only": True,
                "rows": {
                    "12": {"first_diverging_tick": 45, "equal_fraction": 0.95},
                    "16": {"first_diverging_tick": None, "equal_fraction": 1.0},
                    "20": None,
                },
            },
        ),
        "causality": stamped(cuts=[12, 24, 36], **{"R-b1": {}, "R-b1 red": {"12": True}}),
        "primary": stamped(
            host=host,
            d={"readout": primary_d, "grower": per_seed(0.01, OTHER)},
            cells=cells(),
            input_response={seed: {"driven": 256, "mean": 0.0, "sd": 1.0} for seed in SEEDS},
        ),
        "controls": stamped(
            host=controls_host or host,
            sighted=sighted,
            halves={"pool_free": sighted, "pool_positive": True},
            d_shuf={"readout": per_seed(d_shuf, OTHER), "grower": per_seed(0.0, JITTER)},
            window0={
                "d0": {"readout": per_seed(0.03), "grower": per_seed(0.0, OTHER)},
                "w": {"readout": per_seed(-0.02, OTHER), "grower": per_seed(0.0)},
                "scored_rows_through_boundary": {seed: 0 for seed in SEEDS},
                "cells": fit_cells(),
            },
            cells={
                "shuffle": {str(k): fit_cells() for k in range(cs.SHUFFLE_K)},
                "window0": fit_cells(),
            },
        ),
        "transport": stamped(
            b=256,
            taps={seed: {"L": 3} for seed in SEEDS},
            arms=arms,
            forbidden_request={"raised": True},
        ),
        "separation": stamped(
            run=True,
            item_3="RUN",
            reason=None,
            reading={"stream_ok": True},
            host=host,
            window=256,
            stream_checks={seed: True for seed in SEEDS},
            accuracy={
                name: {arm: per_seed(0.8) for arm in ("a", *ps.FIT_FORMS)}
                for name in ("xor", "twin")
            },
            mean={name: {arm: 0.8 for arm in ("a", *ps.FIT_FORMS)} for name in ("xor", "twin")},
            per_text={
                name: {
                    arm: {seed: {"00": 0.5, "01": None} for seed in SEEDS}
                    for arm in ("a", *ps.FIT_FORMS)
                }
                for name in ("xor", "twin")
            },
            distinct_settled_rows={name: {seed: 4 for seed in SEEDS} for name in ("xor", "twin")},
            cells={
                name: {arm: cells()["carried"] for arm in ("a", *ps.FIT_FORMS)}
                for name in ("xor", "twin")
            },
            input_response={"xor": response(0.5), "twin": response(0.6)},
        ),
    }
    return body


def reading(**kwargs):
    return rep.assemble(summaries(**kwargs))["g_r1"]


# ---- provenance: one clean head ------------------------------------------------------------------


def test_the_summaries_of_one_clean_campaign_assemble_and_every_other_set_is_refused():
    report = rep.assemble(summaries())
    assert report["head"] == "h1" and report["dirty"] is False and report["host"] == "kernel"
    assert rep.provenance_problems(summaries()) == []
    base = summaries()
    missing = {k: v for k, v in base.items() if k != "transport"}
    assert rep.provenance_problems(missing) == ["transport: no summary"]
    halted = {**base, "controls": {**base["controls"], "status": "HALTED"}}
    dirty = {**base, "primary": {**base["primary"], "dirty": True}}
    moved = {**base, "migrate": {**base["migrate"], "head": "h2"}}
    for bad, why in (
        (missing, "no summary"),
        (halted, "not COMPLETED"),
        (dirty, "dirty"),
        (moved, "more than one head"),
    ):
        with pytest.raises(ValueError, match=why):
            rep.assemble(bad)
    short = copy.deepcopy(base)
    del short["primary"]["d"]["readout"][
        "315"
    ]  # a seed missing from a per-seed list: never averaged over
    with pytest.raises(ValueError, match="lacks what the report reads"):
        rep.assemble(short)
    thin = copy.deepcopy(base)
    del thin["controls"]["window0"]
    with pytest.raises(ValueError, match="lacks what the report reads"):
        rep.assemble(thin)


def test_a_stored_row_that_disagrees_with_its_recomputation_is_refused():
    base = summaries()
    base["references"]["oracle"]["decoder"]["mean"] += 1e-9  # the stage's stored oracle row, moved
    with pytest.raises(ValueError, match="oracle: the stored row differs"):
        rep.assemble(base)
    base = summaries()
    base["references"]["stream_history"]["readout"]["t"] += 1.0
    with pytest.raises(ValueError, match="stream history: the stored row differs"):
        rep.assemble(base)
    reserve = summaries()
    reserve["record"]["seed_set"] = (
        "reserve"  # live seeds filed as reserve: validate_seed_set refuses
    )
    reserve["record"]["live_seeds"] = list(cs.LIVE_SEEDS)
    assert (
        rep.assemble(reserve)["seed_set"] == "reserve"
    )  # a live seed is not a reserve seed: allowed
    exposed = summaries()
    exposed["record"]["live_seeds"] = [100, *cs.LIVE_SEEDS[1:]]  # an exposed seed
    with pytest.raises(ValueError):
        rep.assemble(exposed)


def test_a_later_report_needs_no_migration_but_the_archived_reading_still_must_agree():
    later = summaries()
    del later["migrate"]
    assert rep.provenance_problems(later) == []
    report = rep.assemble(later)
    assert report["host"] == "kernel" and report["migration"] is None
    assert "## Migration gate: not run" in rep.render(report)
    archived = summaries()["migrate"]
    assert rep._migration(archived)["reading"] == "hosted"
    archived["conjuncts"]["r_b3"] = False
    with pytest.raises(ValueError, match="recorded reading.*not 2.9"):
        rep._migration(archived)


# ---- every contrast row, and 2.5's table ---------------------------------------------------------


def test_every_contrast_row_of_the_campaign_is_built_over_31_seeds_at_the_bar():
    report = rep.assemble(summaries())
    names = list(report["rows"])
    assert names[:4] == [
        "oracle (f64) - (a), readout",
        "oracle (f64) - (a), grower",
        "stream history, readout",
        "stream history, grower",
    ]
    for expected in (
        "primary D, readout",
        "primary D, grower",
        "D_shuf, readout",
        "D - D_shuf, readout",
        "D0, readout",
        "W = D0 - D, readout",
        "D_t pool-copy, readout",
        "D_t - D pool-act, readout",
        "D_t - D_swap pool-copy, readout",
        "D_t - D_swap pool-act, readout",
    ):
        assert expected in names, expected
    assert (
        "D_t - D_swap pool-swap, readout" not in names
    )  # the swap control is not contrasted with itself
    for row in report["rows"].values():
        assert row["n"] == 31 and row["bar"] == cs.T_CRIT and 0.0 <= row["fpr"] <= 1.0
        assert set(row["signs"]) == {"positive", "negative", "zero"} and "skewness" in row
    assert (
        report["rows"]["primary D, readout"]["met"] and not report["rows"]["D_shuf, readout"]["met"]
    )
    assert report["rows"]["oracle (f64) - (a), readout"]["met"]


def test_the_g_r1_reading_of_every_row_of_2_5_through_the_whole_report():
    assert reading() == {"verdict": "PASS", "wording": None, "skew_flagged": False}
    # row 1: either half of the blind harness failing refuses, whatever else reads
    assert reading(oracle_gain=0.0)["verdict"] == "REFUSED (blind harness)"
    assert reading(rb6_ok=False)["verdict"] == "REFUSED (blind harness)"
    assert reading(oracle_gain=0.0, d=0.0)["verdict"] == "REFUSED (blind harness)"  # even a null
    # row 2: a primary that is not met is a null, worded by the stream-history reading
    assert reading(d=0.0) == {"verdict": "RED", "wording": "stream-null", "skew_flagged": False}
    assert reading(d=0.0, history_gain=0.1)["wording"] == "pool-null"
    # row 3: a met D_shuf is RED, whether or not the control is sighted
    assert reading(d_shuf=0.05)["wording"] == "survives the shuffle"
    assert reading(d_shuf=0.05, sighted=False)["wording"] == "survives the shuffle"
    # rows 4 and 5: a blind control is NOT READ, never PASS; and a PASS needs the control on the
    # primary's host
    assert reading(sighted=False)["verdict"] == "NOT READ"
    assert reading(controls_host="port")["verdict"] == "NOT READ"
    assert rep.assemble(summaries(controls_host="port"))["inputs"]["shuffle_sighted"] is False


def test_a_met_primary_with_a_long_left_tail_carries_the_skew_flag_in_the_reading():
    tail = {seed: 0.05 + JITTER[i] / 10 for i, seed in enumerate(SEEDS)}
    tail[SEEDS[-1]] = -0.30
    row = rep.assemble(summaries(d_values=tail))["rows"]["primary D, readout"]
    assert row["met"] and row["skew_flagged"] and row["skewness"] < cs.SKEW_BOUND
    assert reading(d_values=tail) == {"verdict": "PASS", "wording": None, "skew_flagged": True}
    assert reading(d_values=tail, d_shuf=0.05)["skew_flagged"]  # in every row


def test_the_report_inputs_name_what_each_reading_read():
    inputs = rep.assemble(summaries())["inputs"]
    assert inputs["blind_harness_ok"] and inputs["oracle_met"] and inputs["shuffle_sighted"]
    assert inputs["primary_host"] == inputs["controls_host"] == "kernel"
    assert inputs["rb6_clauses"] == {"positive": True, "history_free": True}
    assert inputs["d_shuf_survives"] is False
    report = rep.assemble(summaries(host="port"))
    assert report["host"] == "port" and report["migration"]["use"] == "port"
    assert report["shortened"] is False
    short = summaries()
    short["primary"]["shortened"] = True
    assert rep.assemble(short)["shortened"] is True


# ---- the markdown, the tables and the files ------------------------------------------------------


def test_the_markdown_carries_the_reading_every_row_and_the_per_segment_tables():
    report = rep.assemble(summaries())
    text = rep.render(report)
    assert text.isascii() and "## G-R1 reads: PASS" in text
    assert text == rep.render(report)  # pure: the same report, the same text
    for needle in (
        "primary D, readout",
        "| exact FPR |",
        "## Migration gate: hosted (use kernel)",
        "## Carried: accuracy, entropy, coverage, prior-fallback per seed and segment",
        "## Settled:",
        "## Front-end:",
        "## Degeneracy, startup and input response (Carried)",
        "quantization sensitivity (REPORT ONLY",
        "## Per seed",
        "## Transport (b = 256)",
    ):
        assert needle in text, needle
    carried = text.split("## Carried:")[1].split("## Reset:")[0]
    assert carried.count("unseen-order") == 31  # one segment row per seed
    flagged = rep.assemble(summaries(d_values={**per_seed(0.05), SEEDS[-1]: -0.30}))
    assert "skew-flagged" in rep.render(flagged)
    shortened = summaries()
    shortened["primary"]["shortened"] = True
    assert "SHORTENED" in rep.render(rep.assemble(shortened))


def test_the_report_keeps_every_contrasted_variants_tables_and_the_separation_measurements():
    """P5 panel F9: tables for the references, the shuffle, window 0 and every transport arm beside
    the
    primary's; the separation's accuracy, per-text accuracy and distinct Settled rows pass
    through."""
    report = rep.assemble(summaries())
    tables = report["tables"]
    assert set(tables) == {
        "cells",
        "input_response",
        "variant_response",
        "references",
        "shuffle",
        "window0",
        "transport",
        "separation",
    }
    assert set(tables["references"]) == {"a", "f64", "e1", "e1-0"}
    assert set(tables["shuffle"]) == {str(k) for k in range(cs.SHUFFLE_K)}
    assert set(tables["window0"]) == set(ps.FIT_FORMS)
    assert set(tables["transport"]) == set(md.TRANSPORT_ARMS)
    assert set(tables["separation"]) == {"xor", "twin"}
    sep, saved = report["separation"], summaries()["separation"]
    assert set(rep.SEPARATION_KEYS) == set(sep)
    for key in ("accuracy", "mean", "per_text", "distinct_settled_rows", "stream_checks"):
        assert sep[key] == saved[key], key
    assert report["seed_bundle"] == BUNDLE
    text = rep.render(report)
    for needle in (
        "seed bundle: ",
        "## Reference arm (a): accuracy",
        "## Reference arm (e1-0): accuracy",
        "## carried, within-block shuffle 3: accuracy",
        "## settled at window 0: accuracy",
        "## carried through pool-swap: accuracy",
        "## Separation xor, arm settled: accuracy",
        "## Separation measurements (decoder accuracy)",
        "00:0.50 01:-",  # a per-text accuracy
    ):
        assert needle in text, needle
    # a summary missing one of the new tables is refused, never filled in
    for stage, drop in (("controls", "cells"), ("references", "references"), ("record", "bundle")):
        thin = summaries()
        del thin[stage][drop]
        with pytest.raises(ValueError, match="lacks what the report reads"):
            rep.assemble(thin)
    thin = summaries()
    del thin["transport"]["arms"]["pool-act"]["cells"]
    with pytest.raises(ValueError, match="lacks what the report reads"):
        rep.assemble(thin)
    # a separation that was NOT RUN leaves the report with no separation table, and says so
    unrun = summaries()
    unrun["separation"] = stamped(run=False, item_3="NOT RUN", reason="a clause failed")
    report = rep.assemble(unrun)
    assert report["tables"]["separation"] is None and report["separation"]["run"] is False
    assert "## Separation measurements" not in rep.render(report)
    assert not [name for name in report["tables"]["variant_response"] if "Separation" in name]


def test_the_report_keeps_and_renders_the_input_response_of_every_variant_whose_input_differs():
    """Re-panel R3 (2.3: the driven-unit mask, the mean and sd of per-unit drive beside every
    contrasted variant): the transport arms' and the separation streams' statistics are retained and
    rendered, one table a variant; the primary's stay; window 0 and the shuffles are the primary's
    (carried_state_controls says why); and nothing else of the report moves."""
    base = summaries()
    report = rep.assemble(base)
    kept = report["tables"]["variant_response"]
    arms = {f"through {arm}": base["transport"]["arms"][arm] for arm in md.TRANSPORT_ARMS}
    assert set(kept) == {*arms, "Separation xor", "Separation twin"}
    for name, arm in arms.items():
        assert kept[name] == arm["input_response"]
    assert kept["Separation xor"] == base["separation"]["input_response"]["xor"]
    assert report["tables"]["input_response"] == base["primary"]["input_response"]
    text = rep.render(report)
    assert "window 0 and the within-block shuffles drive the pool with the primary's" in text
    for name, per_seed in kept.items():
        table = text.split(f"### {name}\n\n")[1].split("\n\n")[0]  # that variant's table
        assert table.count("\n") == len(SEEDS) + 1  # a header, its rule and one row a seed
        first = per_seed[SEEDS[0]]
        assert (
            f"| {SEEDS[0]} | {first['driven']} | {first['mean']:.4f} | {first['sd']:.4f} |" in table
        )
    # reporting only: other numbers in those tables move nothing else (rows, G-R1, per seed)
    moved = summaries()
    for arm in moved["transport"]["arms"].values():
        arm["input_response"] = response(7.0)
    moved["separation"]["input_response"] = {"xor": response(8.0), "twin": response(9.0)}
    other = rep.assemble(moved)
    assert other["tables"]["variant_response"] != kept
    for part in ("rows", "g_r1", "per_seed", "migration", "transport", "separation"):
        assert other[part] == report[part], part
    assert {k: v for k, v in other["tables"].items() if k != "variant_response"} == {
        k: v for k, v in report["tables"].items() if k != "variant_response"
    }
    # a summary short of them is refused, never filled in: a transport arm, or a separation that ran
    for stage, path in (("transport", ("arms", "pool-swap")), ("separation", ())):
        thin = summaries()
        holder = thin[stage]
        for step in path:
            holder = holder[step]
        del holder["input_response"]
        with pytest.raises(ValueError, match="lacks what the report reads"):
            rep.assemble(thin)


def test_the_report_stage_writes_both_files_and_files_the_digest(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for stage, body in summaries().items():
        md._save(ps._summary_path(out, stage), {"step": stage, **body})
    assert rep.load_summaries(out).keys() == set(rep.STAGES_READ)
    assert drv.run_stage("report", out, revision=lambda: ("h1", False)) == ps.EXIT_OK
    filed = md._load(ps._summary_path(out, "report"))
    written = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert (
        (out / "report.md").read_text(encoding="utf-8")
        == rep.render(written)
        == rep.render(rep.assemble(summaries()))
    )
    assert filed["digest"] == {
        "report": rep.digest_of(written),
        "markdown": rep.digest_of((out / "report.md").read_text(encoding="utf-8")),
    }
    assert filed["g_r1"] == written["g_r1"] and set(filed["files"]) == {"report.json", "report.md"}
    # a summary set that is not one clean campaign halts the stage and writes neither file
    bad = tmp_path / "bad"
    bad.mkdir()
    for stage, body in summaries().items():
        md._save(ps._summary_path(bad, stage), {"step": stage, **body})
    md._save(ps._summary_path(bad, "controls"), {"step": "controls", **stamped(), "head": "h2"})
    assert drv.run_stage("report", bad, revision=lambda: ("h1", False)) == ps.EXIT_HALT
    assert md._load(ps._summary_path(bad, "report"))["check"] == "report"
    assert not (bad / "report.json").exists() and not (bad / "report.md").exists()


# ---- digests and the golden check ----------------------------------------------------------------


def test_the_digest_is_canonical_and_the_differences_name_the_paths():
    assert rep.digest_of({"a": 1, "b": [1, 2]}) == rep.digest_of({"b": [1, 2], "a": 1})
    assert rep.digest_of({"a": 1}) != rep.digest_of({"a": 2})
    assert len(rep.digest_of({})) == 64
    assert rep.differences({"a": {"b": "1", "c": "2"}}, {"a": {"b": "1", "c": "3"}}) == ["/a/c"]
    assert rep.differences({"a": "1"}, {"a": "1"}) == []
    assert rep.differences({"a": "1"}, {}) == ["/a"] and rep.differences({}, {"b": 1}) == ["/b"]
    assert rep.differences("x", "y") == ["<root>"]


@pytest.fixture(scope="module")
def golden_pair():
    base = rep.golden()
    return base, rep.golden()


def test_the_golden_is_deterministic_and_covers_every_arm_form_on_the_surviving_host(golden_pair):
    base, again = golden_pair
    assert base["scope"] == {
        "hosts": ["kernel"],
        "window": rep.GOLDEN_WINDOW,
        "b": rep.GOLDEN_B,
        "percepts": rep.GOLDEN_PERCEPTS,
        "arms": list(rep.GOLDEN_ARMS),
    }
    assert set(base["arms"]) == set(rep.GOLDEN_ARMS) and len(base["arms"]) == 9
    for host in (st.HOST,):
        assert set(base["replays"][host]) == {
            *(f"pool {form}" for form in rp.FORMS),
            *(f"pool {form}@0" for form in ps.FIT_FORMS),  # window 0 (P5 panel F4)
            *(f"{arm} {form}" for arm in md.TRANSPORT_ARMS for form in ("carried", "settled")),
        }
        assert len(base["hand"][host]) == len(base["r_b3"][host]) == 64
    assert base["report"] is None and len(base["digest"]) == 64
    assert again == base


def test_a_one_ulp_change_to_the_pools_leak_moves_the_golden(golden_pair):
    base, _ = golden_pair
    host = st.HOST
    with rep.leak_nudged():
        nudged = rep.golden()
    assert nudged["digest"] != base["digest"]
    assert nudged["arms"] == base["arms"]  # an arm with no pool is not moved: the red is specific
    assert nudged["hand"][host] != base["hand"][host]
    assert nudged["r_b3"][host] != base["r_b3"][host]
    moved = [n for n, found in nudged["replays"][host].items() if found != base["replays"][host][n]]
    # every pool replay: pool x four forms, Carried and Settled at window 0, each arm x two
    assert len(moved) == 12, moved
    assert rp.LsmPool.__init__.__name__ == "__init__"  # restored on exit
    assert rep.golden()["hand"][host] == base["hand"][host]


def test_a_fault_in_the_window_zero_intervention_alone_moves_the_golden(golden_pair):
    """P5 panel F4: the golden used to replay the pool only at window 16, so a pool whose features
    were
    zeroed at window 0 -- the registered Settled@0 intervention -- left every golden digest
    equal."""
    import numpy as np

    base, _ = golden_pair
    assert {"pool carried@0", "pool settled@0"} <= set(base["replays"][st.HOST])
    original = rp._Pool.percept_drive

    def broken(self, u):
        if self.window == 0:  # zero features only for the window-0 intervention
            self.count += 1
            return np.zeros(self.n)
        return original(self, u)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rp._Pool, "percept_drive", broken)
        faulty = rep.golden()
    moved = sorted(
        name
        for name, found in faulty["replays"][st.HOST].items()
        if found != base["replays"][st.HOST][name]
    )
    assert moved == ["pool carried@0", "pool settled@0"]  # exactly the window-0 records, no other
    assert faulty["digest"] != base["digest"]
    for part in ("hand", "r_b3"):  # the other golden parts stay as they were: the red is specific
        assert faulty[part][st.HOST] == base[part][st.HOST]
    assert faulty["arms"] == base["arms"]


def test_the_leak_red_moves_the_leak_by_exactly_one_ulp_of_the_hosts_arithmetic():
    import numpy as np

    with rep.leak_nudged():
        pool = rp.make_pool("lsm", cs.TUNED, 3, "carried")
    assert np.float32(pool.lam) == np.nextafter(np.float32(cs.TUNED["lambda"]), np.float32(2.0))
    assert np.float32(pool.lam) != np.float32(cs.TUNED["lambda"])
    assert rp.make_pool("lsm", cs.TUNED, 3, "carried").lam == cs.TUNED["lambda"]  # restored


def test_the_golden_recomputes_the_report_over_an_outs_saved_summaries(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    for stage, body in summaries().items():
        md._save(ps._summary_path(out, stage), {"step": stage, **body})
    # the cheap parts stood in for: this test is about the report entry, which the real parts feed
    monkeypatch.setattr(rep, "_hand_record", lambda: {"host": st.HOST})
    monkeypatch.setattr(rep, "_chain_record", lambda: {"host": st.HOST})
    monkeypatch.setattr(rep, "_replay_records", lambda fixture, tap: {"x": {"host": st.HOST}})
    monkeypatch.setattr(rep, "_core", lambda run: {"arm": run["arm"]})
    found = rep.golden(out)
    assert found["scope"]["hosts"] == ["kernel"]  # the surviving host the migrate summary names
    report = rep.assemble(summaries())
    assert found["report"] == {
        "report": rep.digest_of(report),
        "markdown": rep.digest_of(rep.render(report)),
    }
    # moving one saved number moves the report entry (and nothing else)
    moved = summaries()
    moved["primary"]["d"]["readout"]["300"] += 0.001
    md._save(ps._summary_path(out, "primary"), {"step": "primary", **moved["primary"]})
    other = rep.golden(out)
    assert rep.differences(found, other) == ["/digest", "/report/markdown", "/report/report"]
    migration = md._load(ps._summary_path(out, "migrate"))
    md._save(ps._summary_path(out, "migrate"), {**migration, "use": "port"})
    with pytest.raises(ValueError, match="cannot run OUT's host"):
        rep.golden(out)
    md._save(ps._summary_path(out, "migrate"), migration)
    # an OUT that cannot be assembled is refused, never a silent None
    ps._summary_path(out, "transport").unlink()
    with pytest.raises(ValueError, match="transport: no summary"):
        rep.golden(out)
    assert st.use_of(out) == st.HOST


# ---- the golden command --------------------------------------------------------------------------


def test_the_golden_command_prints_a_digest_compares_it_and_names_what_moved(
    tmp_path, monkeypatch, capsys
):
    canned = {"scope": {"hosts": ["kernel"]}, "hand": {"kernel": "a"}, "digest": "d1"}
    monkeypatch.setattr(rep, "golden", lambda out: canned)
    monkeypatch.setattr(drv, "preflight", lambda out: [])
    assert drv.main(["golden"]) == ps.EXIT_OK
    assert json.loads(capsys.readouterr().out) == canned
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(canned), encoding="utf-8")
    assert drv.main(["golden", "--against", str(saved)]) == ps.EXIT_OK
    assert "golden: equal" in capsys.readouterr().out
    saved.write_text(json.dumps({**canned, "hand": {"kernel": "b"}}), encoding="utf-8")
    assert drv.main(["golden", "--against", str(saved)]) == ps.EXIT_ABORT
    assert "MOVED at /hand/kernel" in capsys.readouterr().out
    monkeypatch.setattr(
        rep, "golden", lambda out: (_ for _ in ()).throw(ValueError("transport: no summary"))
    )
    assert drv.main(["golden", str(tmp_path / "out")]) == ps.EXIT_REFUSED
    assert "REFUSED: transport: no summary" in capsys.readouterr().err
    monkeypatch.setattr(drv, "preflight", lambda out: ["the working directory is wrong"])
    assert (
        drv.main(["golden"]) == ps.EXIT_REFUSED
    )  # the campaign's own preflight guards the command too
    assert "working directory" in capsys.readouterr().err


def test_the_module_self_check_runs(capsys):
    rep.self_check()
    out = capsys.readouterr().out
    assert "carried_state_report self-check OK" in out and "a one-ulp leak moved 14 records" in out
