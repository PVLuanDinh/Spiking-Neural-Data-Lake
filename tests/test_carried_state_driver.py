"""G1(c)'s campaign driver (instruments/carried_state_driver.py): the preflight, the CLI, the stage
chain, the first five stages (record, mechanism, instrument, references, construct) of
g1c-carried-state.md section 2.13 each with its constructed red, and -- at the end -- the
MINI-CAMPAIGN: every one of the eleven stages run end to end through the report on 31 stand-in live
seeds.

Nothing here records or builds a Brain, and no live seed (300-361), synthetic seed 6000 or learner
seed 7000-7007 is run: stages run on a scratch OUT at a stand-in revision, their jobs go through
stand-in runners (monkeypatched `ps._child` / `ps._record_child`) or an in-process replay of the
48-event fixture, and the streams are that fixture (or R-b6's family on the exposed seed 4243,
shortened where a test says so). The mini-campaign shortens the window and b to 16 and R-b6's family
to 300 events, and says so in every summary it writes (`scale`, `shortened`)."""

import copy
import functools
import inspect
import itertools
import json
import threading
from pathlib import Path

import pytest

from snn_data_lake.instruments import carried_state_campaign as cs
from snn_data_lake.instruments import carried_state_driver as drv
from snn_data_lake.instruments import carried_state_report as rep
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps

# under xdist one worker owns the family, so its module fixtures are built once
pytestmark = pytest.mark.xdist_group("carried_state")


REST = (203, 204, 205, 206, 207)  # recorded with the live seeds after seed 100 (2.13)


def clean():
    return "h1", False


def write_summary(out, step, status="COMPLETED", head="h1", dirty=False, **body):
    out.mkdir(parents=True, exist_ok=True)
    summary = {"step": step, "status": status, "head": head, "dirty": dirty, **body}
    md._save(ps._summary_path(out, step), summary)


def summary(out, step):
    return md._load(ps._summary_path(out, step))


def fixture(seed=17, *, nudge=False):
    """The 48-event fixture bound to this head under `seed`; `nudge` changes one x (perturbed)."""
    stream = md._synthetic_stream()
    stream["seed"] = seed
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    if nudge:
        stream["events"][5]["x"][0] += 0.25
    return stream


def pilot_dir(tmp_path, *, nudge=False, causal=False):
    """A pilot directory with seed 100's stream (`causal`: and empty files for its eight causal
    replays, enough to START the instrument stage; the stage's own tests build real ones)."""
    pilot = tmp_path / "pilot"
    pilot.mkdir(parents=True, exist_ok=True)
    md._save(pilot / f"{ps.PILOT_SEED}-stream.json.gz", fixture(ps.PILOT_SEED, nudge=nudge))
    for arm in md.SCORED_ARMS if causal else ():
        (pilot / f"{ps.PILOT_SEED}-{arm}-causal.json").touch()
    return pilot


def in_process(spec):
    ps.execute(spec)
    return {"status": "OK", "path": spec["out"]}


# ---- the stage list, the registrations' shape and the module itself ------------------------------


def test_the_stages_are_section_2_13s_and_every_one_is_built():
    assert drv.STAGES == (
        "record",
        "mechanism",
        "instrument",
        "references",
        "construct",
        "causality",
        "primary",
        "controls",
        "transport",
        "separation",
        "report",
    )
    assert tuple(drv.STAGE_FUNCTIONS) == drv.STAGES and not hasattr(drv, "OWED")
    assert drv.REFERENCE_ARMS == ("a", "f64", "e1", "e1-0") and drv.POOL == ps.SELECTABLE[0]
    # the six stages of part 2 are the three modules' own: the driver imports them, they never
    # import it
    assert [
        drv.STAGE_FUNCTIONS[stage].__module__.rsplit(".", 1)[1] for stage in drv.STAGES[5:]
    ] == [
        "carried_state_stages",
        "carried_state_stages",
        "carried_state_controls",
        "carried_state_controls",
        "carried_state_controls",
        "carried_state_report",
    ]


def test_every_module_of_the_driver_stays_under_the_soft_ceiling():
    for module in (drv, drv.st, drv.ctl, drv.rep):
        lines = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8").count("\n")
        assert lines < 800, (module.__name__, lines)


def test_the_self_check_runs(capsys):
    assert drv.main([]) == ps.EXIT_OK
    assert "carried_state_driver self-check OK" in capsys.readouterr().out


# ---- the preflight (CLI only) --------------------------------------------------------------------


def clear(tmp_path):
    root, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    return {"cwd": root, "root": root, "env": {}, "assets": lambda: {}}, elsewhere / "out"


def test_the_preflight_passes_a_clean_launch_and_names_each_refusal_alone(tmp_path):
    good, out = clear(tmp_path)
    assert drv.preflight(out, **good) == []  # the control
    (wrong,) = drv.preflight(out, **{**good, "cwd": out.parent})
    assert "working directory" in wrong
    (inside,) = drv.preflight(good["root"] / "out", **good)
    assert "inside the repository" in inside
    assert len(drv.preflight(good["root"], **good)) == 1  # OUT is the repository itself
    for name in ("SNN_EPROP_CKPT", "SNN_PERCEPTION_PACK", "SNN_EPROP_AUDIO_CKPT", "SNN_SHD_PACK"):
        (problem,) = drv.preflight(out, **{**good, "env": {name: "x"}})
        assert name in problem
    for name in ("NORD_DEVICE", "NORD_GPU"):
        (problem,) = drv.preflight(out, **{**good, "env": {name: ""}})  # set, even empty, is set
        assert name in problem
    assert set(drv.ENV_REFUSED) == {*md.ASSET_ENV, "NORD_DEVICE", "NORD_GPU"}


def test_the_preflight_refuses_a_missing_or_failing_asset_and_reports_every_reason(tmp_path):
    good, out = clear(tmp_path)

    def failing():
        raise ValueError("REFUSED: asset hash eprop_nmnist.pt: 00")

    (problem,) = drv.preflight(out, **{**good, "assets": failing})
    assert "asset hash eprop_nmnist.pt" in problem
    everything = drv.preflight(
        good["root"] / "out",
        **{**good, "cwd": out.parent, "env": {"NORD_GPU": "1"}, "assets": failing},
    )
    assert len(everything) == 4


def test_the_default_assets_are_the_recorders_own_check(monkeypatch, tmp_path):
    good, out = clear(tmp_path)
    asked = []
    monkeypatch.setattr(md, "_assets", lambda: asked.append("asked") or {})
    assert drv.preflight(out, **{k: v for k, v in good.items() if k != "assets"}) == []
    assert asked == ["asked"]


def test_the_repo_root_is_where_the_recorder_reads_models_and_git():
    root = drv.repo_root()
    assert (root / "models").is_dir() and (root / "pyproject.toml").is_file()
    assert (root / "src" / "snn_data_lake" / "instruments").is_dir()


def test_the_cli_refuses_without_writing_anything(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    monkeypatch.chdir(tmp_path)  # not the repository root
    assert drv.main(["mechanism", str(out)]) == ps.EXIT_REFUSED
    assert "working directory" in capsys.readouterr().err and not out.exists()
    monkeypatch.chdir(drv.repo_root())
    monkeypatch.setenv("NORD_GPU", "1")
    assert drv.main(["mechanism", str(out)]) == ps.EXIT_REFUSED
    assert "NORD_GPU" in capsys.readouterr().err and not out.exists()
    monkeypatch.delenv("NORD_GPU")
    inside = drv.repo_root() / "never-written-out"
    assert drv.main(["mechanism", str(inside)]) == ps.EXIT_REFUSED
    assert "inside the repository" in capsys.readouterr().err and not inside.exists()


def test_the_cli_passes_each_stage_only_what_it_takes(tmp_path, monkeypatch):
    given = []
    monkeypatch.setattr(drv, "preflight", lambda out: [])
    monkeypatch.setattr(drv, "run_stage", lambda stage, out, **kw: given.append((stage, kw)) or 0)
    out = str(tmp_path / "out")
    for argv in (
        ["record", out, "--pilot", "p"],
        ["record", out, "--pilot", "p", "--seeds", "reserve"],
        ["mechanism", out],
        ["instrument", out, "--pilot", "p"],
        ["references", out],
        ["construct", out],
        *([stage, out] for stage in drv.STAGES[5:]),
    ):
        assert drv.main(argv) == 0
    assert given == [
        ("record", {"pilot": "p", "seeds": "live"}),
        ("record", {"pilot": "p", "seeds": "reserve"}),
        ("mechanism", {}),
        ("instrument", {"pilot": "p"}),
        ("references", {}),
        ("construct", {}),
        *((stage, {}) for stage in drv.STAGES[5:]),
    ]


def test_the_cli_refuses_a_missing_pilot_a_reserve_without_repair_and_an_early_stage(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(drv, "preflight", lambda out: [])
    out = tmp_path / "out"
    assert drv.main(["record", str(out)]) == ps.EXIT_REFUSED
    assert "--pilot DIR is required" in capsys.readouterr().err
    pilot = pilot_dir(tmp_path)
    assert drv.main(["record", str(out), "--pilot", str(pilot), "--seeds", "reserve"]) == 3
    assert "REPAIR.txt" in capsys.readouterr().err
    assert drv.main(["instrument", str(out), "--pilot", str(pilot)]) == ps.EXIT_REFUSED
    assert "causal.json" in capsys.readouterr().err  # the eight causal replays are missing
    monkeypatch.setattr(md, "_revision", clean)  # the CLI reads the real tree: stand in a clean one
    assert drv.main(["causality", str(out)]) == ps.EXIT_REFUSED
    assert "causality: construct has no summary" in capsys.readouterr().err
    assert not list(out.glob("*.summary.json"))  # nothing was written by any refusal


def test_the_cli_refuses_the_retired_migration_stage(tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        drv.main(["migrate", str(tmp_path)])
    assert refused.value.code == 2
    assert "invalid choice: 'migrate'" in capsys.readouterr().err


# ---- the stage chain (pool_selection's own refusals, at stand-in revisions) ----------------------


def test_a_stage_refuses_to_start_before_its_predecessor_and_a_control_runs(tmp_path):
    out = tmp_path / "out"
    pilot = str(pilot_dir(tmp_path, causal=True))  # the instrument stage needs one to start at all
    for stage in drv.STAGES[1:5]:
        extra = {"pilot": pilot} if stage == "instrument" else {}
        with pytest.raises(ps.Refused, match="no summary"):
            drv.run_stage(stage, out, revision=clean, function=lambda _o, **_k: {}, **extra)
    assert not list(out.glob("*.summary.json"))
    write_summary(out, "record")
    code = drv.run_stage("mechanism", out, revision=clean, function=lambda _o: {"n": 0})
    assert code == ps.EXIT_OK and summary(out, "mechanism")["step"] == "mechanism"


def test_a_predecessor_at_another_head_or_that_did_not_complete_refuses(tmp_path):
    write_summary(tmp_path / "head", "record", head="h2")
    with pytest.raises(ps.Refused, match="not at the clean head"):
        drv.run_stage("mechanism", tmp_path / "head", revision=clean, function=lambda _o: {})
    write_summary(tmp_path / "halted", "record", status="HALTED")
    with pytest.raises(ps.Refused, match="HALTED"):
        drv.run_stage("mechanism", tmp_path / "halted", revision=clean, function=lambda _o: {})
    write_summary(tmp_path / "dirty", "record", dirty=True)
    with pytest.raises(ps.Refused, match="not at the clean head"):
        drv.run_stage("mechanism", tmp_path / "dirty", revision=clean, function=lambda _o: {})


def test_a_dirty_tree_refuses_and_a_moved_head_aborts(tmp_path):
    write_summary(tmp_path / "out", "record")
    with pytest.raises(ps.Refused, match="dirty"):
        drv.run_stage(
            "mechanism", tmp_path / "out", revision=lambda: ("h1", True), function=lambda _o: {}
        )
    moves = iter([("h1", False), ("h2", False)])
    code = drv.run_stage(
        "mechanism", tmp_path / "out", revision=lambda: next(moves), function=lambda _o: {}
    )
    assert code == ps.EXIT_ABORT
    moved = summary(tmp_path / "out", "mechanism")
    assert moved["status"] == "ABORTED" and moved["moved"] == {
        "start": "h1",
        "end": "h2",
        "dirty": False,
    }


def test_a_stage_never_runs_twice_in_one_out(tmp_path):
    write_summary(tmp_path / "out", "record")
    assert drv.run_stage("mechanism", tmp_path / "out", revision=clean, function=lambda _o: {}) == 0
    with pytest.raises(ps.Refused, match="already ran"):
        drv.run_stage("mechanism", tmp_path / "out", revision=clean, function=lambda _o: {})


@pytest.mark.parametrize("stage", drv.STAGES[5:])
def test_a_later_stage_refuses_before_its_predecessor_and_writes_no_summary(tmp_path, stage):
    out = tmp_path / "out"
    previous = drv.STAGES[drv.STAGES.index(stage) - 1]
    with pytest.raises(ps.Refused, match=f"{previous} has no summary"):
        drv.run_stage(stage, out, revision=clean)
    assert not list(out.glob("*.summary.json"))
    with pytest.raises(ps.Refused, match="unknown stage"):
        drv.run_stage("nonsense", out, revision=clean)


# ---- the seeds and 2.6's half-moved rule ---------------------------------------------------------


def test_the_live_seeds_are_validated_and_the_reserve_is_the_repair_set():
    assert drv.live_seeds() == cs.LIVE_SEEDS == tuple(range(300, 331))
    assert drv.live_seeds("reserve") == cs.RESERVE_SEEDS == tuple(range(331, 362))
    assert len(drv.live_seeds()) == len(drv.live_seeds("reserve")) == 31


def test_the_half_moved_rule_holds_on_every_preregistered_size_of_every_live_and_reserve_seed():
    sizes = drv.preregistered_sizes()
    assert len(sizes) == 8 and {s[:4] for s in sizes} == {(256, 256, 207, 256)}
    assert [s[4] for s in sizes] == list(range(207, 215))
    for seed in (*cs.LIVE_SEEDS, *cs.RESERVE_SEEDS):
        drv.check_half_moved(seed, sizes)


def test_the_half_moved_rule_fires_on_a_draw_that_moves_under_half_and_on_an_empty_recording():
    with pytest.raises(ps.Halt, match="half-moved") as halted:
        drv.check_half_moved(
            cs.LIVE_SEEDS[0], [[2] * 12]
        )  # 12 two-blocks: an identity draw is certain
    assert "under half" in halted.value.detail
    with pytest.raises(ps.Halt, match="half-moved"):
        drv.check_half_moved(cs.LIVE_SEEDS[0], [[]])
    drv.check_half_moved(cs.LIVE_SEEDS[0], [[256, 207]])  # the control passes


# ---- the record stage ----------------------------------------------------------------------------


def recorder(*, timeout=(), nudge=(), fail=()):
    """A stand-in for memory_diagnostic's --record child: the 48-event fixture under the seed's
    name (a seed in `timeout` times out, one in `fail` FAILs, one in `nudge` records a perturbed
    stream). The fixtures are built here, in the calling thread (md._synthetic_stream asks
    threadpoolctl)."""
    plain, nudged = fixture(), fixture(nudge=True)

    def child(job):
        if job["seed"] in timeout:
            return {"status": "TIMEOUT", "path": job["out"]}
        if job["seed"] in fail:
            return {"status": "FAIL", "path": job["out"]}
        md._save(job["out"], dict(nudged if job["seed"] in nudge else plain, seed=job["seed"]))
        return {"status": "OK", "path": job["out"]}

    return child


def spy_batches(monkeypatch):
    batches = []
    real = ps._record_seeds

    def spy(out, seeds):
        batches.append(tuple(seeds))
        return real(out, seeds)

    monkeypatch.setattr(ps, "_record_seeds", spy)
    return batches


def record(tmp_path, monkeypatch, *, seeds="live", nudge_pilot=False, **kwargs):
    monkeypatch.setattr(ps, "_record_child", recorder(**kwargs))
    batches = spy_batches(monkeypatch)
    out = tmp_path / "out"
    pilot = str(pilot_dir(tmp_path, nudge=nudge_pilot))
    code = drv.run_stage("record", out, revision=clean, pilot=pilot, seeds=seeds)
    return code, out, batches


def test_record_runs_seed_100_alone_then_one_batch_and_files_pool_selections_summary(
    tmp_path, monkeypatch, capsys
):
    code, out, batches = record(tmp_path, monkeypatch)
    assert code == ps.EXIT_OK
    assert batches == [(100,), (*REST, *cs.LIVE_SEEDS)]  # alone first, then ONE batch
    body = summary(out, "record")
    assert len(body["recordings"]) == 37 and {r["status"] for r in body["recordings"].values()} == {
        "OK"
    }
    assert ps._recorded(out) == {seed: "OK" for seed in (100, *REST, *cs.LIVE_SEEDS)}
    assert body["live_seeds"] == list(cs.LIVE_SEEDS) and body["seed_set"] == "live"
    assert (
        body["bundle"]
        == {  # ONE complete bundle, read by migrate, instrument and controls
            "which": "live",
            "live": list(range(300, 331)),
            "gate": [204, 205, 206, 207],
            "gate_synthetic": [5005, 5006, 5007, 5008],
            "rb6": 6000,
        }
        and body["reserve_claim"] is None
    )
    assert body["identity"]["equal"] and body["identity"]["new"] == body["identity"]["pilot"]
    assert body["projection"]["rounds"] == 9 and body["projection"]["workers"] == ps.WORKERS
    assert body["repair"] is None and body["shuffle"]["k"] == cs.SHUFFLE_K
    assert body["shuffle"]["recorded_blocks"]["300"] == [24, 24]  # the fixture: one block a segment
    assert len(body["shuffle"]["preregistered_blocks"]) == 8
    printed = capsys.readouterr().out
    assert "record: seed 100 alone took" in printed and "36 recordings project to" in printed


def test_a_recording_that_times_out_halts_naming_the_seed(tmp_path, monkeypatch):
    code, out, batches = record(tmp_path, monkeypatch, timeout=(315,))
    assert code == ps.EXIT_HALT and len(batches) == 2
    halted = summary(out, "record")
    assert halted["check"] == "record" and halted["detail"] == {"not OK": {"315": "TIMEOUT"}}
    code, out, batches = record(tmp_path / "again", monkeypatch, timeout=(ps.PILOT_SEED,))
    assert code == ps.EXIT_HALT and batches == [(100,)]  # no batch after a missing seed 100
    assert summary(out, "record")["detail"] == {"seed": 100, "status": "TIMEOUT"}


def test_a_perturbed_seed_100_stream_halts_on_the_digest_before_the_batch(tmp_path, monkeypatch):
    code, out, batches = record(tmp_path, monkeypatch, nudge=(ps.PILOT_SEED,))
    assert code == ps.EXIT_HALT and batches == [(100,)]
    halted = summary(out, "record")
    assert halted["check"] == "R-b8 stream identity"
    assert halted["detail"]["new"] != halted["detail"]["pilot"]
    # the same stream against a tampered pilot halts the same way (the pilot is the other side)
    code, out, batches = record(tmp_path / "pilot-side", monkeypatch, nudge_pilot=True)
    assert code == ps.EXIT_HALT and summary(out, "record")["check"] == "R-b8 stream identity"


def test_the_half_moved_assertion_runs_before_the_first_recording(tmp_path, monkeypatch):
    def refuses(seed, k, sizes):
        raise AssertionError(f"shuffle (seed {seed}, k {k}): a block moves under half")

    monkeypatch.setattr(cs, "block_permutations", refuses)
    code, out, batches = record(tmp_path, monkeypatch)
    assert code == ps.EXIT_HALT and batches == []  # nothing was recorded
    halted = summary(out, "record")
    assert halted["check"] == "shuffle half-moved" and "under half" in halted["detail"]


def test_the_half_moved_assertion_runs_on_each_recordings_own_block_sizes_before_any_replay(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cs, "blocks", lambda stream: [[0, 1]] * 12)  # twelve 2-blocks per recording
    code, out, batches = record(tmp_path, monkeypatch)
    assert code == ps.EXIT_HALT and len(batches) == 2  # it ran after the recordings
    halted = summary(out, "record")
    assert halted["check"] == "shuffle half-moved" and "under half" in halted["detail"]


REST_RESERVE = (203, *cs.GATE_RESERVE_SEEDS)  # the reserve bundle's: gate 208-211, not 204-207


def test_a_reserve_record_needs_a_repair_file_naming_the_first_out(tmp_path, monkeypatch):
    first = tmp_path / "first"
    write_summary(first, "record", seed_set="live")  # an ORIGINAL live campaign
    out = tmp_path / "out"
    with pytest.raises(ps.Refused, match="REPAIR.txt"):
        record(tmp_path, monkeypatch, seeds="reserve")
    assert not out.exists()  # no summary, no directory: it never started
    pilot = str(pilot_dir(tmp_path))
    out.mkdir()
    for text, why in (
        (f"{first}\n", "first line"),  # names the first OUT and no repair
        ("somewhere else\nthe repair\n", "first OUT"),  # names no OUT that ran
        (f"{out}\nthe repair\n", "very OUT"),  # names itself (once it holds a record summary)
    ):
        (out / drv.REPAIR_FILE).write_text(text, encoding="utf-8")
        write_summary(out, "record")
        (problem,) = drv.start_problems("record", out, pilot, "reserve")
        assert why in problem
        ps._summary_path(out, "record").unlink()
    (out / drv.REPAIR_FILE).write_text(f"{first}\na fold of finding 7\n", encoding="utf-8")
    code, _, batches = record(tmp_path, monkeypatch, seeds="reserve")
    # P5 panel F2: the WHOLE bundle moves -- gate 208-211, not 204-207 -- and the record names it
    assert code == ps.EXIT_OK and batches == [(100,), (*REST_RESERVE, *cs.RESERVE_SEEDS)]
    body = summary(out, "record")
    assert body["seed_set"] == "reserve" and body["live_seeds"] == list(cs.RESERVE_SEEDS)
    assert body["bundle"] == {
        "which": "reserve",
        "live": list(range(331, 362)),
        "gate": [208, 209, 210, 211],
        "gate_synthetic": [5009, 5010, 5011, 5012],
        "rb6": 6001,
    }
    assert "a fold of finding 7" in body["repair"]
    assert not set(cs.LIVE_SEEDS) & {int(seed) for seed in body["recordings"]}
    assert not set(cs.GATE_SEEDS) & {int(seed) for seed in body["recordings"]}
    assert (first / drv.RESERVE_CLAIM_FILE).read_text(encoding="utf-8").strip() == str(out)
    assert body["reserve_claim"] == str(first / drv.RESERVE_CLAIM_FILE)


def test_a_repair_is_made_once_of_an_original_live_campaign_and_never_chained(
    tmp_path, monkeypatch
):
    """P5 panel F5: REPAIR validation reads the original's record summary (seed_set live) and
    CLAIMS its
    single reserve attempt exclusively before anything is recorded; a reserve-to-reserve chain and a
    second attempt naming the same original OUT are both refused."""
    original, earlier = tmp_path / "original", tmp_path / "earlier-reserve"
    write_summary(original, "record", seed_set="live")
    write_summary(earlier, "record", seed_set="reserve")
    pilot = str(pilot_dir(tmp_path))

    def attempt(name, first):
        out = tmp_path / name
        out.mkdir()
        (out / drv.REPAIR_FILE).write_text(f"{first}\na stand-in repair\n", encoding="utf-8")
        return out

    chain = attempt("chain", earlier)  # a repair of a repair
    (problem,) = drv.start_problems("record", chain, pilot, "reserve")
    assert "never repaired again" in problem and "'reserve'" in problem
    with pytest.raises(ps.Refused, match="never repaired again"):
        drv.run_stage("record", chain, revision=clean, pilot=pilot, seeds="reserve")
    assert (
        not list(chain.glob("*.summary.json")) and not (earlier / drv.RESERVE_CLAIM_FILE).exists()
    )
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    (unreadable / "record.summary.json").write_text("{not json", encoding="utf-8")
    (problem,) = drv.start_problems("record", attempt("bad", unreadable), pilot, "reserve")
    assert "cannot be read" in problem
    first = attempt("repair1", original)  # repair 1: accepted, and it claims the one attempt
    assert drv.start_problems("record", first, pilot, "reserve") == []
    monkeypatch.setattr(ps, "_record_child", recorder())
    assert (
        drv.run_stage("record", first, revision=clean, pilot=pilot, seeds="reserve") == ps.EXIT_OK
    )
    assert (original / drv.RESERVE_CLAIM_FILE).read_text(encoding="utf-8").strip() == str(first)
    second = attempt("repair2", original)  # a second attempt on the same original OUT
    (problem,) = drv.start_problems("record", second, pilot, "reserve")
    assert "already gave its one reserve attempt" in problem and "halts G1(c)" in problem
    with pytest.raises(ps.Refused, match="already gave its one reserve attempt"):
        drv.run_stage("record", second, revision=clean, pilot=pilot, seeds="reserve")
    assert not list(second.glob("*.summary.json"))


def test_a_race_for_the_one_reserve_attempt_halts_the_loser_before_it_records(
    tmp_path, monkeypatch
):
    original = tmp_path / "original"
    write_summary(original, "record", seed_set="live")
    pilot = str(pilot_dir(tmp_path))
    first, second = (tmp_path / name for name in ("a", "b"))
    for out in (first, second):
        out.mkdir()
        (out / drv.REPAIR_FILE).write_text(f"{original}\na repair\n", encoding="utf-8")
    assert (
        drv.start_problems("record", second, pilot, "reserve") == []
    )  # both passed the start check
    batches = spy_batches(monkeypatch)
    monkeypatch.setattr(ps, "_record_child", recorder())
    assert drv.run_stage("record", first, revision=clean, pilot=pilot, seeds="reserve") == 0
    monkeypatch.setattr(drv, "start_problems", lambda *a, **k: [])  # the check it passed earlier
    before = list(batches)
    code = drv.run_stage("record", second, revision=clean, pilot=pilot, seeds="reserve")
    assert code == ps.EXIT_HALT and batches == before  # claimed by the other: nothing recorded
    halted = summary(second, "record")
    assert halted["check"] == "record: the reserve attempt is already claimed"
    assert halted["detail"] == {"original": str(original), "by": str(first)}


# ---- re-panel R2: the record stage's seed set survives a crash, a halt and an abort --------------

ENDINGS = {"interrupted": "RUNNING", "halted": "HALTED", "aborted": "ABORTED"}


def reserve_start(out, original):
    """OUT's REPAIR.txt for a reserve record: line 1 names `original`, the rest is the repair."""
    out.mkdir(parents=True, exist_ok=True)
    (out / drv.REPAIR_FILE).write_text(f"{original}\na repair of executed code\n", encoding="utf-8")


def ended_record(tmp_path, monkeypatch, how, seeds):
    """The record stage on the `seeds` bundle, ended `how`: "interrupted" (a KeyboardInterrupt after
    seed 100's recording, so only the exclusive RUNNING marker stands), "halted" (a TIMEOUT) or
    "aborted" (a FAILed job). Returns its OUT. A reserve record is itself a repair of an original
    live one."""
    work = tmp_path / f"{how}-{seeds}"
    out = work / "out"
    if seeds == "reserve":
        write_summary(work / "original", "record", seed_set="live")
        reserve_start(out, work / "original")
    monkeypatch.setattr(
        ps,
        "_record_child",
        recorder(
            timeout=(cs.CAUSALITY_SEED,) if how == "halted" else (),
            fail=(cs.CAUSALITY_SEED,) if how == "aborted" else (),
        ),
    )
    if how == "interrupted":
        real, batches = ps._record_seeds, []

        def second_batch_dies(folder, which):
            batches.append(which)
            if len(batches) == 2:
                raise KeyboardInterrupt
            return real(folder, which)

        monkeypatch.setattr(ps, "_record_seeds", second_batch_dies)
    pilot = str(pilot_dir(work))
    try:
        drv.run_stage("record", out, revision=clean, pilot=pilot, seeds=seeds)
    except KeyboardInterrupt:
        assert how == "interrupted"
    return out


@pytest.mark.parametrize("how", ENDINGS)
def test_an_original_live_record_that_did_not_complete_files_its_seed_set_and_is_repairable(
    tmp_path, monkeypatch, how
):
    """Re-panel R2: 2.11's golden rule can require a reserve restart after a crashed, halted or
    aborted ORIGINAL live record; its seed set was filed before the body ran, so a repair can name
    it."""
    original = ended_record(tmp_path, monkeypatch, how, "live")
    found = summary(original, "record")
    assert (found["status"], found["seed_set"]) == (ENDINGS[how], "live")
    assert (found["head"], found["dirty"]) == ("h1", False)
    repair = tmp_path / "repair" / "out"
    reserve_start(repair, original)
    pilot = str(pilot_dir(tmp_path / "repair"))
    assert drv.start_problems("record", repair, pilot, "reserve") == []
    monkeypatch.undo()  # the repair records (stand-in children) and claims the one attempt
    monkeypatch.setattr(ps, "_record_child", recorder())
    assert drv.run_stage("record", repair, revision=clean, pilot=pilot, seeds="reserve") == 0
    assert summary(repair, "record")["seed_set"] == "reserve"
    assert (original / drv.RESERVE_CLAIM_FILE).read_text(encoding="utf-8").strip() == str(repair)


@pytest.mark.parametrize("how", ENDINGS)
def test_a_reserve_record_that_did_not_complete_is_never_repaired_in_its_turn(
    tmp_path, monkeypatch, how
):
    """The rule stays whole: an original whose record summary reads seed_set reserve, in ANY status,
    is refused (a repair is never repaired again, 2.11); the refused start writes nothing."""
    attempted = ended_record(tmp_path, monkeypatch, how, "reserve")
    assert summary(attempted, "record")["seed_set"] == "reserve"
    chain = tmp_path / "chain" / "out"
    reserve_start(chain, attempted)
    pilot = str(pilot_dir(tmp_path / "chain"))
    (problem,) = drv.start_problems("record", chain, pilot, "reserve")
    assert "never repaired again" in problem and "'reserve'" in problem
    with pytest.raises(ps.Refused, match="never repaired again"):
        drv.run_stage("record", chain, revision=clean, pilot=pilot, seeds="reserve")
    assert not list(chain.glob("*.summary.json"))
    assert not (attempted / drv.RESERVE_CLAIM_FILE).exists()


@pytest.mark.parametrize("status", ["RUNNING", "HALTED", "ABORTED", "COMPLETED"])
def test_a_record_summary_that_names_no_seed_set_is_still_refused_as_an_original(tmp_path, status):
    """A summary with no seed set (None) is never accepted as an original, whatever its status: the
    repair rule is not weakened to admit what the stage did not file."""
    original = tmp_path / "original"
    write_summary(original, "record", status=status)
    repair = tmp_path / "repair"
    reserve_start(repair, original)
    (problem,) = drv.start_problems("record", repair, str(pilot_dir(tmp_path)), "reserve")
    assert "seed_set None" in problem and "only the ORIGINAL live campaign is repaired" in problem


@pytest.mark.parametrize("seeds", ["live", "reserve"])
def test_the_record_stages_running_marker_carries_its_seed_set_before_the_body_runs(
    tmp_path, seeds
):
    out = tmp_path / "out"
    if seeds == "reserve":
        write_summary(tmp_path / "original", "record", seed_set="live")
        reserve_start(out, tmp_path / "original")
    seen = []

    def body(folder, **kwargs):
        seen.append(summary(folder, "record"))
        return {}

    code = drv.run_stage(
        "record", out, revision=clean, function=body, pilot=str(pilot_dir(tmp_path)), seeds=seeds
    )
    assert code == ps.EXIT_OK
    assert seen == [
        {"step": "record", "status": "RUNNING", "head": "h1", "dirty": False, "seed_set": seeds}
    ]
    assert summary(out, "record")["seed_set"] == seeds  # kept in the summary the body did not fill
    # the provenance is the record stage's alone: a later stage's marker names no seed set
    later = []
    drv.run_stage(
        "mechanism",
        out,
        revision=clean,
        function=lambda folder: later.append(summary(folder, "mechanism")) or {},
    )
    assert later == [{"step": "mechanism", "status": "RUNNING", "head": "h1", "dirty": False}]


# ---- the mechanism stage -------------------------------------------------------------------------


def mechanism_children(
    specs, *, refuse=False, red=True, config=True, timeout=False, persistence=True
):
    def child(spec):
        specs.append(spec)
        if spec["kind"] == "mechanism":
            body = {"candidate": "lsm", "greens": {"R-b0": not refuse}, "reds": {"R-b0": red}}
            md._save(spec["out"], {**body, "numbers": {}})
        elif spec["kind"] == "configuration":
            if timeout:
                return {"status": "TIMEOUT", "path": spec["out"]}
            result = {name: {"verdict": True, "numbers": {}} for name in ps.CONFIGURATION_CHECKS}
            result["R-b3"] = {"verdict": True, "numbers": {"settings": [[1, 1], [24, 4]]}}
            result["R-b5 freeze"]["verdict"] = config
            result["R-b5 persistence"]["verdict"] = persistence
            md._save(spec["out"], result)
        else:
            raise AssertionError(f"unexpected job kind {spec['kind']}")
        return {"status": "OK", "path": spec["out"]}

    return child


def run_mechanism(tmp_path, monkeypatch, **kwargs):
    specs = []
    monkeypatch.setattr(ps, "_child", mechanism_children(specs, **kwargs))
    out = tmp_path / "out"
    write_summary(out, "record")
    return drv.run_stage("mechanism", out, revision=clean), out, specs


def test_mechanism_runs_g1bs_checks_and_one_configuration_job_for_tuned(tmp_path, monkeypatch):
    code, out, specs = run_mechanism(tmp_path, monkeypatch, persistence=False)
    assert code == ps.EXIT_OK
    (configuration,) = [s for s in specs if s["kind"] == "configuration"]
    assert (configuration["candidate"], configuration["config"]) == ("lsm", cs.TUNED)
    body = summary(out, "mechanism")
    assert body["refused"] == {} and body["configuration"]["config"] == cs.TUNED
    assert body["configuration"]["verdicts"]["R-b5 persistence"] is False  # reported, never a halt
    assert body["configuration"]["R-b3 settings"] == [[1, 1], [24, 4]]


@pytest.mark.parametrize(
    ("kwargs", "check", "detail"),
    [
        ({"refuse": True}, "mechanism", {"lsm": ["R-b0"]}),  # a failed green: no runner-up
        ({"config": False}, "mechanism: TUNED configuration", {"failed": ["R-b5 freeze"]}),
        ({"timeout": True}, "mechanism: TUNED configuration", None),
        ({"red": False}, "a red that does not fire", ["lsm R-b0"]),
    ],
)
def test_mechanism_halts_on_any_failure_of_the_pool_under_test(
    tmp_path, monkeypatch, kwargs, check, detail
):
    code, out, _ = run_mechanism(tmp_path, monkeypatch, **kwargs)
    halted = summary(out, "mechanism")
    assert code == ps.EXIT_HALT and halted["check"] == check
    if detail is not None:
        assert halted["detail"] == detail
    else:
        assert halted["detail"]["status"] == "TIMEOUT"


# ---- the instrument stage (real replays, in process, on the 48-event fixture) --------------------


def instrument_out(out):
    """An OUT the instrument stage may start in: predecessor summaries, seed 100's recording."""
    write_summary(out, "record", recordings={"100": {"status": "OK"}}, seed_set="live")
    write_summary(out, "mechanism")
    ps._folder(out, "record")
    md._save(ps._recording(out, ps.PILOT_SEED), fixture(ps.PILOT_SEED))


@pytest.fixture
def instrument_world(tmp_path, monkeypatch):
    """OUT with seed 100's recording (the fixture), a pilot of its own eight causal replays, and
    R-b6's family shortened to 300 events on the exposed seed 4243 (never 6000); jobs run here."""
    monkeypatch.setattr(ps, "_child", in_process)
    monkeypatch.setattr(md, "RB6_EVENTS", 300)
    monkeypatch.setattr(md, "RB6_WARM_UP", 64)
    stream = fixture(ps.PILOT_SEED)
    out, pilot = tmp_path / "out", tmp_path / "pilot"
    instrument_out(out)
    pilot.mkdir()
    md._save(pilot / f"{ps.PILOT_SEED}-stream.json.gz", stream)
    for arm in md.SCORED_ARMS:
        md._save(pilot / f"{ps.PILOT_SEED}-{arm}-causal.json", md.replay(stream, arm))
    return out, pilot


def run_instrument(out, pilot):
    stage = functools.partial(drv.stage_instrument, rb6_seed=4243)
    return drv.run_stage("instrument", out, revision=clean, function=stage, pilot=str(pilot))


def test_instrument_records_r_b6_a_sighted_shuffle_control_and_r_b8_when_the_harness_sees(
    instrument_world,
):
    out, pilot = instrument_world
    assert run_instrument(out, pilot) == ps.EXIT_OK
    body = summary(out, "instrument")
    assert body["rb6_clauses"] == {"positive": True, "history_free": True} and body["rb6_ok"]
    assert body["shuffle_control"]["sighted"] and body["shuffle_aa"]["held"]
    assert body["shuffle_control"]["unshuffled"] >= md.RB6_BAR > body["shuffle_control"]["shuffled"]
    assert body["R-b8"]["source_digest"]["new"] == body["R-b8"]["source_digest"]["pilot"]
    assert body["rb6_seed"] == 4243 and "R-b2" in body and "R-b7 reds" in body


def test_a_blind_r_b6_is_recorded_not_halted_and_r_b8_is_still_read(instrument_world, monkeypatch):
    out, pilot = instrument_world
    monkeypatch.setattr(md, "RB6_BAR", 10.0)  # the positive clause cannot reach it: a blind harness
    assert run_instrument(out, pilot) == ps.EXIT_OK  # COMPLETED: the later stages still run
    body = summary(out, "instrument")
    assert body["rb6_clauses"] == {"positive": False, "history_free": True} and not body["rb6_ok"]
    assert not body["shuffle_control"]["sighted"]  # a blind result is a reading too
    assert all(body["R-b8"]["pre_build"].values())  # R-b8 ran after the blind R-b6
    # and so does the history-free red failing (the instrument reads history where none exists)
    monkeypatch.setattr(md, "RB6_BAR", -10.0)
    again = pilot.parent / "again"
    instrument_out(again)
    assert run_instrument(again, pilot) == ps.EXIT_OK
    assert summary(again, "instrument")["rb6_clauses"] == {"positive": True, "history_free": False}
    assert not summary(again, "instrument")["rb6_ok"]


def test_a_tampered_pilot_halts_r_b8_even_under_a_blind_r_b6(instrument_world, monkeypatch):
    out, pilot = instrument_world
    md._save(pilot / f"{ps.PILOT_SEED}-stream.json.gz", fixture(ps.PILOT_SEED, nudge=True))
    monkeypatch.setattr(md, "RB6_BAR", 10.0)
    assert run_instrument(out, pilot) == ps.EXIT_HALT
    halted = summary(out, "instrument")
    assert halted["check"] == "R-b8 pre-build"
    assert halted["detail"]["source_digest"]["new"] != halted["detail"]["source_digest"]["pilot"]


def test_pool_selections_default_still_halts_on_r_b6_and_keeps_its_summary_shape(
    instrument_world, monkeypatch
):
    out, pilot = instrument_world
    default = ps.step_instrument(out, pilot, rb6_seed=4243)
    assert list(default) == ["R-b2", "R-b7 reds", "R-b6", "R-b8", "rb6_seed"]  # no rb6_clauses
    monkeypatch.setattr(md, "RB6_BAR", 10.0)
    with pytest.raises(ps.Halt) as blind:
        ps.step_instrument(out, pilot, rb6_seed=4243)
    assert blind.value.check == "R-b6"
    monkeypatch.setattr(md, "RB6_BAR", -10.0)
    with pytest.raises(ps.Halt) as silent:
        ps.step_instrument(out, pilot, rb6_seed=4243, rb6_halts=True)
    assert silent.value.check == "R-b6 red"
    assert inspect.signature(ps.step_instrument).parameters["rb6_halts"].default is True


@pytest.mark.parametrize(("seed_set", "rb6"), [("live", 6000), ("reserve", 6001)])
def test_the_stage_runs_the_bundles_r_b6_seed_as_a_reading_and_never_builds_that_stream(
    tmp_path, monkeypatch, seed_set, rb6
):
    """P5 panel F2: the default is the record's bundle (6000 live, 6001 reserve), not a constant."""
    asked = {}

    def stand_in(out, pilot, *, rb6_seed, rb6_halts):
        asked.update(rb6_seed=rb6_seed, rb6_halts=rb6_halts)
        for name, path in ps._rb6_families(ps._folder(out, "instrument"), rb6_seed).items():
            history_free = name == "rb6red"
            md._save(path, md.rb6_stream(4243, history_free=history_free))  # never seed 6000/6001
        return {"rb6_clauses": {"positive": True, "history_free": True}}

    monkeypatch.setattr(ps, "step_instrument", stand_in)
    out = tmp_path / "out"
    write_summary(out, "record", seed_set=seed_set)
    body = drv.stage_instrument(out, tmp_path / "pilot")
    assert asked == {"rb6_seed": rb6, "rb6_halts": False}
    assert inspect.signature(drv.stage_instrument).parameters["rb6_seed"].default is None
    assert body["rb6_ok"] and body["shuffle_control"]["sighted"] and body["shuffle_aa"]["held"]
    assert (out / "instrument" / f"rb6-{rb6}-stream.json.gz").is_file()
    # a record that names no seed set cannot choose a bundle: a halt, never a silent live default
    write_summary(tmp_path / "bare", "record")
    with pytest.raises(ps.Halt, match="seed bundle"):
        drv.stage_instrument(tmp_path / "bare", tmp_path / "pilot")


def test_the_pool_free_shuffle_control_sees_history_and_is_blind_without_it():
    # on R-b6's family (exposed seed 4243) the decoder's (e1) - (e1-0) gap clears 0.05 unshuffled
    # and falls under it shuffled; the history-free twin has no gap to see (the control's red)
    sighted = drv.shuffle_control(md.rb6_stream(4243))
    assert sighted["sighted"] and sighted["unshuffled"] >= md.RB6_BAR > sighted["shuffled"]
    assert sighted["shuffles"] == cs.SHUFFLE_K and sighted["bar"] == md.RB6_BAR
    blind = drv.shuffle_control(md.rb6_stream(4243, history_free=True))
    assert not blind["sighted"] and blind["unshuffled"] < md.RB6_BAR


# ---- the references stage ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_runs():
    """One real in-process replay of the fixture per reference arm; the stand-in relabels it."""
    stream = fixture()
    return {arm: md.replay(stream, arm) for arm in drv.REFERENCE_ARMS}, [
        e["c"] for e in stream["events"]
    ]


def predictions(truth, hits):
    """48 predictions that get exactly `hits` of the 24 scored (unseen-order) events right."""
    return [
        None if i < 24 else (truth[i] if i < 24 + hits else (truth[i] + 1) % 3) for i in range(48)
    ]


def replays(base_runs, hits, *, timeout=None):
    """A stand-in replay child: the arm's real run, relabelled to the seed and rescored to
    `hits(arm, seed, learner)` scored hits for each learner."""
    runs, truth = base_runs

    def child(spec):
        seed = int(Path(spec["stream"]).name.split("-")[0])
        if (spec["arm"], seed) == timeout:
            return {"status": "TIMEOUT", "path": spec["out"]}
        run = copy.deepcopy(runs[spec["arm"]])
        run["seed"] = seed
        run["predictions"] = predictions(truth, hits(spec["arm"], seed, "grower"))
        run["readout"]["predictions"] = predictions(truth, hits(spec["arm"], seed, "readout"))
        md._save(spec["out"], run)
        return {"status": "OK", "path": spec["out"]}

    return child


def reference_world(tmp_path, seeds=cs.LIVE_SEEDS, seed_set="live"):
    out = tmp_path / "out"
    write_summary(
        out,
        "record",
        recordings={str(s): {"status": "OK"} for s in seeds},
        live_seeds=list(seeds),
        seed_set=seed_set,
    )
    for step in ("mechanism", "instrument"):
        write_summary(out, step)
    ps._folder(out, "record")
    for seed in seeds:
        md._save(ps._recording(out, seed), fixture(seed))
    return out


def sees_the_oracle(arm, seed, learner):
    """f64 reads 4-6 events above (a) on every seed, through both learners; (e1) 0-2 above (e1-0)"""
    base = {"a": 6 + seed % 5, "f64": 10 + seed % 5 + seed % 3, "e1-0": 8 + seed % 4}
    return base.get(arm, 8 + seed % 4 + seed % 3)


def test_references_reads_the_oracle_and_the_stream_history_and_writes_inspected(
    tmp_path, monkeypatch, base_runs
):
    out = reference_world(tmp_path)
    monkeypatch.setattr(ps, "_child", replays(base_runs, sees_the_oracle))
    assert drv.run_stage("references", out, revision=clean) == ps.EXIT_OK
    body = summary(out, "references")
    assert body["inspected"] is True and (out / drv.INSPECTED_FILE).is_file()
    assert body["live_seeds"] == list(cs.LIVE_SEEDS)
    assert body["oracle"]["met"] and body["oracle_verdict"] == "MET"
    assert body["oracle"]["decoder"]["n"] == 31 and body["oracle"]["decoder"]["bar"] == cs.T_CRIT
    assert body["accuracy"]["readout"]["300"]["a"] == 6 / 24  # 300 % 5 = 0: 6 of the 24 scored
    assert body["accuracy"]["readout"]["300"]["f64"] == 10 / 24
    assert (
        body["stream_history"]["readout"]["n"] == 31 and body["stream_history"]["grower"]["n"] == 31
    )
    assert body["stream_history"]["readout"]["met"]  # (e1) is 0-2 events above (e1-0) on every seed
    assert set(body["references"]) == set(drv.REFERENCE_ARMS)
    assert body["references"]["a"]["per_seed"]["300"]["readout"]["unseen-order"]["n"] == 24
    assert sorted(path.name for path in (out / "references").glob("ref-f64-*.json.gz"))[:2] == [
        "ref-f64-300.json.gz",
        "ref-f64-301.json.gz",
    ]


def test_an_oracle_the_decoder_does_not_see_is_refused_and_recorded_not_halted(
    tmp_path, monkeypatch, base_runs
):
    def grower_only(arm, seed, learner):
        # the grower reads f64 above (a) on every seed; the decoder reads them equal
        if learner == "grower":
            return sees_the_oracle(arm, seed, learner)
        return 6 + seed % 5 if arm in ("a", "f64") else sees_the_oracle(arm, seed, learner)

    out = reference_world(tmp_path)
    monkeypatch.setattr(ps, "_child", replays(base_runs, grower_only))
    assert drv.run_stage("references", out, revision=clean) == ps.EXIT_OK  # COMPLETED: no halt
    body = summary(out, "references")
    assert body["oracle_verdict"] == "REFUSED (blind harness)" and not body["oracle"]["met"]
    assert body["oracle"]["decoder"]["degenerate"] and body["oracle"]["grower"]["met"]
    assert body["inspected"] is True and body["stream_history"]["readout"]["met"]


def test_a_reference_replay_that_times_out_halts_and_inspection_is_already_marked(
    tmp_path, monkeypatch, base_runs
):
    out = reference_world(tmp_path)
    monkeypatch.setattr(ps, "_child", replays(base_runs, sees_the_oracle, timeout=("f64", 310)))
    assert drv.run_stage("references", out, revision=clean) == ps.EXIT_HALT
    halted = summary(out, "references")
    assert halted["check"] == "references" and halted["detail"] == {"not OK": [["f64", 310]]}
    assert (out / drv.INSPECTED_FILE).is_file()  # replays of live seeds had been written (2.11)


def test_the_reserve_seeds_are_read_when_the_record_says_reserve_and_refused_when_it_does_not(
    tmp_path, monkeypatch, base_runs
):
    monkeypatch.setattr(ps, "_child", replays(base_runs, sees_the_oracle))
    reserve = reference_world(tmp_path / "a", cs.RESERVE_SEEDS, "reserve")
    assert drv.run_stage("references", reserve, revision=clean) == ps.EXIT_OK
    assert summary(reserve, "references")["live_seeds"] == list(cs.RESERVE_SEEDS)
    mislabelled = reference_world(
        tmp_path / "b", cs.RESERVE_SEEDS, "live"
    )  # reserve seeds, live set
    assert drv.run_stage("references", mislabelled, revision=clean) == ps.EXIT_HALT
    halted = summary(mislabelled, "references")
    assert halted["check"] == "references: oracle" and "reserve seeds" in halted["detail"]


# ---- the construct stage -------------------------------------------------------------------------


def construction_child(specs, *, verdict=True, status="OK"):
    def child(spec):
        specs.append(spec)
        if status != "OK":
            return {"status": status, "path": spec["out"]}
        md._save(
            spec["out"], {"verdict": verdict, "a1": {"ok": True}, "seeds": [7000], "epochs": 400}
        )
        return {"status": "OK", "path": spec["out"]}

    return child


def run_construct(tmp_path, monkeypatch, **kwargs):
    specs = []
    monkeypatch.setattr(ps, "_child", construction_child(specs, **kwargs))
    out = tmp_path / "out"
    write_summary(out, "references")
    return drv.run_stage("construct", out, revision=clean), out, specs


def test_construct_is_one_call_job_to_the_construction_check_with_no_arguments(
    tmp_path, monkeypatch
):
    code, out, specs = run_construct(tmp_path, monkeypatch)
    assert code == ps.EXIT_OK
    (job,) = specs
    assert job["kind"] == "call" and job["args"] == {}
    assert job["target"] == "snn_data_lake.instruments.inseparable_mapping:construction_check"
    from snn_data_lake.instruments import inseparable_mapping as im

    assert ps._call_target(job["target"]) is im.construction_check  # P5a's resolver accepts it
    body = summary(out, "construct")
    assert body["run"] and body["verdict"] is True and body["item_3"] == "RUN"


def test_a_false_verdict_completes_and_makes_item_three_not_run(tmp_path, monkeypatch):
    code, out, _ = run_construct(tmp_path, monkeypatch, verdict=False)
    assert code == ps.EXIT_OK  # COMPLETED: G1(c) continues (2.8)
    body = summary(out, "construct")
    assert body["verdict"] is False and body["item_3"] == "NOT RUN" and "g1c 2.8" in body["reason"]
    assert body["run"] and body["result"]["a1"] == {"ok": True}


def test_a_construction_check_that_times_out_is_not_run_and_one_that_fails_aborts(
    tmp_path, monkeypatch
):
    code, out, _ = run_construct(tmp_path / "t", monkeypatch, status="TIMEOUT")
    assert code == ps.EXIT_OK
    body = summary(out, "construct")
    assert body["run"] is False and body["verdict"] is None and body["item_3"] == "NOT RUN"
    code, out, _ = run_construct(
        tmp_path / "f", monkeypatch, status="FAIL"
    )  # a defect in the check
    assert code == ps.EXIT_ABORT and summary(out, "construct")["status"] == "ABORTED"


# ---- the mini-campaign: every stage, end to end, to the report -----------------------------------

MINI_SCALE = {
    "window": 16,
    "b": 16,
    "percepts": 64,
    "kicks": 48,
}  # disclosed: the 48-event fixture's
MINI_RB6 = 4243  # exposed; the stage is told this seed, never 6000
# the G2(ii) stream, shortened and told so: the separation stage ENFORCES its structural checks and
# arm (a)'s floor, which 60 events do not reach (the stand-in world measured 120 does); 120 fails
# seed 328's lag-1 repeat-rate check by chance, 140 passes the checks for all 31 stand-in seed names
MINI_SEPARATION_EVENTS = 140


def mini_children():
    """The mini-campaign's `ps._child`: the mechanism, configuration and construction jobs are
    stand-ins (the construction check's learner seeds 7000-7007 are never run), every other job runs
    in-process; a replay's execution id gets a process of its own, as R-b4's twins have in a
    campaign."""
    counter, lock = itertools.count(1000), threading.Lock()

    def child(spec):
        kind, ok = spec["kind"], {"status": "OK", "path": spec["out"]}
        if kind == "mechanism":
            body = {
                "candidate": "lsm",
                "greens": {"R-b0": True},
                "reds": {"R-b0": True},
                "numbers": {},
            }
            md._save(spec["out"], body)
            return ok
        if kind == "configuration":
            result = {name: {"verdict": True, "numbers": {}} for name in ps.CONFIGURATION_CHECKS}
            result["R-b3"] = {"verdict": True, "numbers": {"settings": [[1, 1]]}}
            md._save(spec["out"], result)
            return ok
        if kind == "call" and spec["target"].endswith(":construction_check"):
            md._save(
                spec["out"], {"verdict": True, "a1": {"ok": True}, "seeds": [7000], "epochs": 400}
            )
            return ok
        ps.execute(spec)
        if kind == "replay":
            run = md._load(spec["out"])
            with lock:
                run["execution_id"] = f"{next(counter)}-" + run["execution_id"].split("-", 1)[1]
            md._save(spec["out"], run)
        return ok

    return child


def mini_functions(build):
    stages = drv.STAGE_FUNCTIONS
    return {
        "instrument": functools.partial(drv.stage_instrument, rb6_seed=MINI_RB6),
        "causality": functools.partial(stages["causality"], scale=MINI_SCALE),
        "primary": functools.partial(stages["primary"], scale=MINI_SCALE),
        "controls": functools.partial(stages["controls"], scale=MINI_SCALE, rb6_seed=MINI_RB6),
        "transport": functools.partial(stages["transport"], scale=MINI_SCALE),
        "separation": functools.partial(stages["separation"], scale=MINI_SCALE, build=build),
    }


def stand_in(seed):
    """A tiny synthetic recording under `seed`: the 48-event fixture, seed 100's exactly (it is the
    pilot's, and R-b8 compares them), every other seed's with its category held for 2 + seed % 3
    events, so that the stand-ins differ and the contrasts have a spread over their seeds."""
    stream = fixture(seed)
    if seed != ps.PILOT_SEED:
        for index, event in enumerate(stream["events"]):
            event["c"] = (index // (2 + seed % 3)) % 3
    return stream


def mini_recorder(streams):
    def child(job):
        md._save(job["out"], streams[job["seed"]])
        return {"status": "OK", "path": job["out"]}

    return child


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """OUT after all eleven stages. Tiny synthetic streams (`stand_in`, built from the 48-event
    fixture, which is seed 100's) stand in for the 37 recordings; the pilot is seed 100's honest
    stream and eight causal replays."""
    root = tmp_path_factory.mktemp("mini")
    out, pilot = root / "out", root / "pilot"
    with pytest.MonkeyPatch.context() as patch:
        streams = {seed: stand_in(seed) for seed in (ps.PILOT_SEED, *REST, *cs.LIVE_SEEDS)}
        patch.setattr(md, "RB6_EVENTS", 300)
        patch.setattr(md, "RB6_WARM_UP", 64)
        patch.setattr(im, "STREAM_WARM_UP", 24)
        patch.setattr(
            im, "STREAM_EVENTS", MINI_SEPARATION_EVENTS
        )  # the structural checks are enforced
        patch.setattr(ps, "_record_child", mini_recorder(streams))
        patch.setattr(ps, "_child", mini_children())
        stream = fixture(ps.PILOT_SEED)
        pilot.mkdir()
        md._save(pilot / f"{ps.PILOT_SEED}-stream.json.gz", stream)
        for arm in md.SCORED_ARMS:
            md._save(pilot / f"{ps.PILOT_SEED}-{arm}-causal.json", md.replay(stream, arm))
        functions = mini_functions(
            lambda seed, lab: im._build_stream(seed, lab, n_events=MINI_SEPARATION_EVENTS)
        )
        codes = {}
        for stage in drv.STAGES:
            extra = {"pilot": str(pilot)} if stage in ("record", "instrument") else {}
            codes[stage] = drv.run_stage(
                stage, out, revision=clean, function=functions.get(stage), **extra
            )
    return out, pilot, codes


def test_the_mini_campaign_runs_every_stage_end_to_end_at_one_clean_head(mini):
    out, _, codes = mini
    assert codes == {stage: ps.EXIT_OK for stage in drv.STAGES}
    bodies = {stage: summary(out, stage) for stage in drv.STAGES}
    assert {body["status"] for body in bodies.values()} == {"COMPLETED"}
    assert {(body["head"], body["dirty"]) for body in bodies.values()} == {("h1", False)}
    assert [body["step"] for body in bodies.values()] == list(drv.STAGES)
    assert bodies["record"]["live_seeds"] == list(cs.LIVE_SEEDS)
    assert len(bodies["record"]["recordings"]) == 37
    assert bodies["references"]["inspected"] is True and (out / drv.INSPECTED_FILE).is_file()
    assert "migrate" not in bodies
    for stage in ("primary", "controls", "transport", "separation"):  # each binds to that host
        assert bodies[stage]["host"] == "kernel" and bodies[stage]["shortened"] is True
        assert bodies[stage]["scale"] == MINI_SCALE
    assert bodies["primary"]["live_seeds"] == list(cs.LIVE_SEEDS)
    assert bodies["separation"]["run"] is True and bodies["construct"]["item_3"] == "RUN"
    assert bodies["controls"]["sighted"] is True and bodies["instrument"]["rb6_ok"] is True


def test_the_mini_campaigns_report_reads_31_seeds_and_marks_itself_shortened(mini):
    out, _, _ = mini
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    text = (out / "report.md").read_text(encoding="utf-8")
    assert report["shortened"] is True and "SHORTENED" in text and text.isascii()
    assert report["head"] == "h1" and report["dirty"] is False and report["host"] == "kernel"
    assert report["live_seeds"] == list(cs.LIVE_SEEDS) and len(report["per_seed"]) == 31
    assert all(row["n"] == 31 and row["bar"] == cs.T_CRIT for row in report["rows"].values())
    wanted = {"primary D, readout", "D_shuf, readout", "D0, readout", "W = D0 - D, readout"}
    assert wanted <= set(report["rows"])
    assert len(report["tables"]["cells"]) == 4 and len(report["tables"]["cells"]["carried"]) == 31
    # P5 panel F9: the diagnostic tables of EVERY contrasted variant, and the separation
    # measurements
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
    # re-panel R3: the input response of the transport arms and of both separation streams, per seed
    assert set(tables["variant_response"]) == {
        *(f"through {arm}" for arm in md.TRANSPORT_ARMS),
        "Separation xor",
        "Separation twin",
    }
    assert all(len(per_seed) == 31 for per_seed in tables["variant_response"].values())
    assert "### through pool-swap" in text and "### Separation twin" in text
    assert set(tables["references"]) == set(drv.REFERENCE_ARMS)
    assert set(tables["shuffle"]) == {str(k) for k in range(cs.SHUFFLE_K)}
    assert set(tables["window0"]) == set(ps.FIT_FORMS) and set(tables["transport"]) == set(
        md.TRANSPORT_ARMS
    )
    assert set(tables["separation"]) == {"xor", "twin"}
    for variant in (
        tables["references"]["a"],
        tables["window0"]["settled"],
        tables["shuffle"]["0"]["carried"],
    ):
        assert len(variant) == 31 and variant["300"]["readout"]
    separation = report["separation"]
    assert {"accuracy", "per_text", "distinct_settled_rows", "mean", "stream_checks"} <= set(
        separation
    ) and all(held is True for held in separation["stream_checks"].values())
    assert report["seed_bundle"]["which"] == "live" and report["seed_bundle"]["rb6"] == 6000
    assert "## Reference arm (a): accuracy" in text and "## Separation measurements" in text
    assert "## carried at window 0: accuracy" in text and "## settled through pool-copy" in text
    # the stand-ins differ, so no contrast is degenerate: every row has a spread over its 31 seeds
    for name in ("primary D, readout", "D_shuf, readout", "oracle (f64) - (a), readout"):
        row = report["rows"][name]
        assert not row["degenerate"] and row["sd"] > 0 and row["fpr"] > 0, name
    inputs, rows = report["inputs"], report["rows"]
    # the reading is 2.5's table over the report's own rows and inputs, whichever row it lands on
    assert report["g_r1"] == cs.g_r1_reading(
        blind_harness_ok=inputs["blind_harness_ok"],
        primary=rows["primary D, readout"],
        stream_history=rows["stream history, readout"],
        d_shuf=rows["D_shuf, readout"],
        shuffle_sighted=inputs["shuffle_sighted"],
    )
    assert report["g_r1"]["verdict"] in {"PASS", "RED", "NOT READ", "REFUSED (blind harness)"}
    assert inputs["blind_harness_ok"] == (
        all(inputs["rb6_clauses"].values()) and inputs["oracle_met"]
    )
    assert inputs["shuffle_sighted"] is True and report["separation"]["run"] is True
    assert report["migration"] is None
    assert "## Migration gate: not run" in text
    assert set(report["transport"]["records"]) == set(md.TRANSPORT_ARMS)
    assert f"## G-R1 reads: {report['g_r1']['verdict']}" in text
    filed = summary(out, "report")
    assert filed["g_r1"] == report["g_r1"] and filed["digest"]["report"] == rep.digest_of(report)


def test_the_report_is_a_pure_function_of_the_saved_summaries(mini):
    out, _, _ = mini
    again = rep.assemble(rep.load_summaries(out))
    saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(ps._plain(again)))
    assert (out / "report.md").read_text(encoding="utf-8") == rep.render(again)
    assert rep.assemble(rep.load_summaries(out)) == again  # and deterministic


def test_the_golden_over_the_mini_campaigns_summaries_agrees_with_the_report_stage(mini):
    out, _, _ = mini
    found = rep.golden(out)
    assert found["scope"]["hosts"] == ["kernel"]  # the surviving host the migrate summary names
    assert found["report"] == summary(out, "report")["digest"]
    assert rep.golden(out) == found  # in-process at one head: byte-identical


def test_no_seed_the_campaign_may_not_run_was_run_by_the_mini_campaign(mini):
    out, _, _ = mini
    names = [path.name for path in out.rglob("*") if path.is_file()]
    banned = (6000, *range(7000, 7008), *cs.RESERVE_SEEDS)
    for seed in banned:
        assert not [name for name in names if f"-{seed}-" in name or f"-{seed}." in name], seed
    assert (out / "instrument" / f"rb6-{MINI_RB6}-stream.json.gz").is_file()


@pytest.mark.parametrize("stage", drv.STAGES)
def test_a_stage_of_the_mini_campaign_never_runs_twice_and_a_dirty_tree_refuses_it(
    mini, tmp_path, stage
):
    out, pilot, _ = mini
    extra = {"pilot": str(pilot)} if stage in ("record", "instrument") else {}
    with pytest.raises(ps.Refused, match="already ran"):
        drv.run_stage(stage, out, revision=clean, function=lambda _o, **_k: {}, **extra)
    dirty = tmp_path / "dirty"  # a dirty-tree stand-in refuses before anything is written
    with pytest.raises(ps.Refused, match="dirty"):
        drv.run_stage(
            stage, dirty, revision=lambda: ("h1", True), function=lambda _o, **_k: {}, **extra
        )
    assert not list(dirty.glob("*.summary.json"))


def test_a_tampered_pilot_halts_r_b8_in_the_chain_and_no_later_stage_starts(tmp_path, monkeypatch):
    code, out, batches = record(tmp_path, monkeypatch, nudge_pilot=True)
    assert code == ps.EXIT_HALT and batches == [(100,)]  # halted on seed 100 before the batch
    halted = summary(out, "record")
    assert halted["check"] == "R-b8 stream identity"
    assert halted["detail"]["new"] != halted["detail"]["pilot"]
    pilot = str(pilot_dir(tmp_path, causal=True))  # enough to START the instrument stage
    for stage in drv.STAGES[1:]:  # the chain stops: the predecessor HALTED, or never ran
        extra = {"pilot": pilot} if stage == "instrument" else {}
        with pytest.raises(ps.Refused, match="HALTED|no summary"):
            drv.run_stage(stage, out, revision=clean, function=lambda _o, **_k: {}, **extra)
    assert [p.name for p in out.glob("*.summary.json")] == ["record.summary.json"]
