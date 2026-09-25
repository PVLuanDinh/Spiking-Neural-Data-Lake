"""G1(b)'s campaign (cortex/pool_selection.py): section 2.5's ordering and eligibility on
hand-built tables, section 2.6's power rule, R-b7's refusals, the step-order refusal and section
2.8's taxonomy and exit codes (g1b-pool-selection.md, frozen at 23c7e97).

Nothing here runs a campaign: steps run on a scratch OUT at a stand-in revision, and their jobs go
through stand-in runners (or one in-process replay of the 48-event fixture)."""

import pytest

from snn_data_lake.cortex import memory_diagnostic as md
from snn_data_lake.cortex import pool_selection as ps
from snn_data_lake.cortex import recurrent_pool as rp


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


def test_r_b7_refuses_its_three_reds_and_passes_the_well_formed_table():
    reds = ps.r_b7_reds()
    assert reds["verdict"] and reds["control"]
    assert reds["red"] == {
        "wrong configuration": True,
        "duplicate seed": True,
        "tuning seed in selection": True,
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
