"""G1(c)'s campaign driver, part 2 (instruments/carried_state_stages.py and
carried_state_controls.py): the migrate, causality, primary, controls, transport and separation
stages of g1c-carried-state.md 2.13, each with its constructed red.

Nothing here records or builds a Brain, and no live seed (300-361), synthetic seed 6000 or learner
seed 7000-7007 is run. The world is a scratch OUT whose recordings are the 48-event fixture under
stand-in seeds (three standing in for the live seeds: the stages take no contrast, the report does),
R-b6's family is the exposed seed 4243 shortened to 300 events, and the pool's window and the tap's
b are 16 -- all disclosed: every summary names its `scale` and `shortened`. Jobs run in-process
through a stand-in `ps._child`."""

import contextlib
import functools
import hashlib
import itertools
import json
import math
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from snn_data_lake.instruments import carried_state_campaign as cs
from snn_data_lake.instruments import carried_state_controls as ctl
from snn_data_lake.instruments import carried_state_driver as drv
from snn_data_lake.instruments import carried_state_stages as st
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.reservoir import recurrent_pool as rp

SCALE = {"window": 16, "b": 16, "percepts": 64, "kicks": 48}
LIVE = (300, 301, 302)
RB6 = 4243  # exposed; never 6000
SEEDS = (100, 203, *cs.GATE_SEEDS, *LIVE)
HOSTED_SEEDS = (*cs.GATE_SEEDS, *cs.GATE_SYNTHETIC_SEEDS)  # every hosted cell, live bundle (2.9)


def clean():
    return "h1", False


def fixture(seed):
    stream = md._synthetic_stream()
    stream["seed"] = seed
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    return stream


class Children:
    """A stand-in for `ps._child`: jobs run in-process. A replay's execution id gets a process of
    its own (so R-b4's twins are two processes, as they are in a campaign) unless `patch_pid` is
    False; `timeouts` names jobs that time out, `overrides` replaces a "call" job's result by
    function name, and `tamper` edits a finished replay (spec, run) before it is saved."""

    def __init__(self, *, patch_pid=True, timeouts=None, overrides=None, tamper=None):
        self.patch_pid, self.timeouts = patch_pid, timeouts
        self.overrides, self.tamper = overrides or {}, tamper
        self.counter, self.lock, self.specs = itertools.count(1000), threading.Lock(), []

    def __call__(self, spec):
        self.specs.append(spec)
        if self.timeouts and self.timeouts(spec):
            return {"status": "TIMEOUT", "path": spec["out"]}
        if spec["kind"] == "call":
            name = spec["target"].split(":")[1]
            if name in self.overrides:
                override = self.overrides[name]
                md._save(spec["out"], override(spec) if callable(override) else override)
                return {"status": "OK", "path": spec["out"]}
        ps.execute(spec)
        if spec["kind"] == "replay":
            run = md._load(spec["out"])
            if self.patch_pid:
                with self.lock:
                    run["execution_id"] = (
                        f"{next(self.counter)}-" + run["execution_id"].split("-", 1)[1]
                    )
            if self.tamper:
                self.tamper(spec, run)
            md._save(spec["out"], run)
        return {"status": "OK", "path": spec["out"]}


def write_summary(out, step, status="COMPLETED", **body):
    out.mkdir(parents=True, exist_ok=True)
    md._save(
        ps._summary_path(out, step),
        {"step": step, "status": status, "head": "h1", "dirty": False, **body},
    )


def summary(out, step):
    return md._load(ps._summary_path(out, step))


def base_out(out, *, construct=True):
    """OUT after the record-to-construct stages, by hand: the recordings, R-b6's family on 4243
    and the summaries the pool stages read."""
    ps._folder(out, "record")
    for seed in SEEDS:
        md._save(ps._recording(out, seed), fixture(seed))
    families = ps._rb6_families(ps._folder(out, "instrument"), RB6)
    md._save(families["rb6"], md.rb6_stream(RB6))
    md._save(families["rb6red"], md.rb6_stream(RB6, history_free=True))
    write_summary(
        out,
        "record",
        recordings={str(seed): {"status": "OK"} for seed in SEEDS},
        live_seeds=list(LIVE),
        seed_set="live",
    )
    write_summary(out, "mechanism")
    write_summary(
        out,
        "instrument",
        rb6_seed=RB6,
        rb6_ok=True,
        shuffle_control={"sighted": True},
        shuffle_aa={"held": True},
    )
    write_summary(out, "references")
    write_summary(
        out,
        "construct",
        run=True,
        verdict=construct,
        item_3="RUN" if construct else "NOT RUN",
        reason=None if construct else "a clause failed",
    )


SEPARATION_EVENTS = (
    120  # the G2(ii) stream, shortened: 60 events leave arm (a) under the twin floor
)


def short_stream(seed, labelling):
    return im._build_stream(seed, labelling, n_events=SEPARATION_EVENTS)


def shorten_separation(patch):
    """The disclosed shortening of the separation world: a 24-event warm-up, and the structural
    stream
    checks (which the stage now ENFORCES) told the shortened length, so a clean stream still passes
    them."""
    patch.setattr(im, "STREAM_WARM_UP", 24)
    patch.setattr(im, "STREAM_EVENTS", SEPARATION_EVENTS)


@contextlib.contextmanager
def shortened(children=None):
    """The disclosed shortening of the world: R-b6's family and the synthetic gate streams at 300
    events."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(md, "RB6_EVENTS", 300)
        patch.setattr(md, "RB6_WARM_UP", 64)
        patch.setattr(ps, "_child", children or Children())
        yield patch


def stage(name, **kwargs):
    return functools.partial(drv.STAGE_FUNCTIONS[name], scale=SCALE, **kwargs)


def run(out, name, **kwargs):
    code = drv.run_stage(name, out, revision=clean, function=stage(name, **kwargs))
    return code, summary(out, name)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The six pool stages run through the step protocol on three stand-in live seeds."""
    out = tmp_path_factory.mktemp("world") / "out"
    with shortened():
        base_out(out)
        codes = {}
        for name in ("migrate", "causality", "primary"):
            codes[name], _ = run(out, name)
        codes["controls"], _ = run(out, "controls", rb6_seed=RB6)
        codes["transport"], _ = run(out, "transport")
        with pytest.MonkeyPatch.context() as patch:
            shorten_separation(patch)
            codes["separation"], _ = run(out, "separation", build=short_stream)
    return out, codes


def copy_of(world, tmp_path, *drop):
    """A private copy of the world's OUT with the named stages' summaries removed (to re-run)."""
    out = tmp_path / "out"
    shutil.copytree(world[0], out)
    for name in drop:
        ps._summary_path(out, name).unlink(missing_ok=True)
    return out


def call(function, out, **kwargs):
    """Run a stage function directly (no protocol), returning its body or the Halt it raised."""
    try:
        return function(out, **kwargs)
    except ps.Halt as halt:
        return halt


# ---- the shared helpers --------------------------------------------------------------------------


def test_the_stage_functions_are_the_drivers_and_every_stage_completed(world):
    _, codes = world
    assert codes == {name: ps.EXIT_OK for name in codes} and len(codes) == 6
    assert tuple(st.STAGE_FUNCTIONS) == ("migrate", "causality", "primary")
    assert tuple(ctl.STAGE_FUNCTIONS) == ("controls", "transport", "separation")
    assert all(
        drv.STAGE_FUNCTIONS[name] is f
        for name, f in {**st.STAGE_FUNCTIONS, **ctl.STAGE_FUNCTIONS}.items()
    )


def test_settings_default_to_the_real_scale_and_refuse_an_unknown_key():
    assert st.settings() == st.REAL_SCALE == {"window": 256, "b": 256, "percepts": 512, "kicks": 16}
    assert st.settings({"window": 16})["window"] == 16 and st.settings({"window": 16})["b"] == 256
    with pytest.raises(ValueError, match="unknown scale keys"):
        st.settings({"nope": 1})


def test_the_cell_key_is_pool_selections_own_for_every_variant():
    for host, window in (("port", rp.WINDOW), ("kernel", rp.WINDOW), ("port", 0), ("kernel", 16)):
        cell = ps._cell(
            Path("s"),
            Path("o"),
            7,
            spec=ps._spec("lsm", cs.TUNED, "settled", window=window, host=host),
        )
        assert st.key("settled", 7, host=host, window=window) == ps.cell_key(cell)
    assert (
        len({st.key("carried", 7, host=h, window=w) for h in st.HOSTS for w in (0, 16, 256)}) == 6
    )
    noise = st.key("carried", 7, host="port", window=256, noise=True)
    assert noise != st.key("carried", 7, host="port", window=256) and noise[5] is True
    copy = st.key(
        "carried",
        7,
        host="port",
        window=256,
        arm="pool-copy",
        transport={"form": "copy"},
        encoder=5,
    )
    assert copy[0] == "pool-copy" and copy[-4:] == (
        "transport",
        rp.crc_json({"form": "copy"}),
        "encoder",
        5,
    )


def test_a_table_that_is_not_ok_halts_and_two_groups_may_hold_one_key(tmp_path):
    with shortened():
        path = tmp_path / "17-stream.json.gz"
        md._save(path, fixture(17))
        streams = {17: md._load(path)}

        def cells(tag):
            return [
                st.pool_cell(
                    path, tmp_path / f"{tag}-{form}.json.gz", 17, form, host="port", window=16
                )
                for form in ps.FIT_FORMS
            ]

        first, second = st.run_cells([(cells("a"), (17,), streams), (cells("b"), (17,), streams)])
        assert first.keys() == second.keys() and len(first) == 2  # one key in two tables: fine
        st.need_ok(first, "x")
        gap = st.accuracy(first, st.key("carried", 17, host="port", window=16)) - st.accuracy(
            first, st.key("settled", 17, host="port", window=16)
        )
        assert (
            st.gaps(st.accuracies(first, (17,), ps.FIT_FORMS, host="port", window=16))["readout"][
                "17"
            ]
            == gap
        )
        with pytest.raises(ps.Halt, match="R-b7"):  # one table never holds a key twice
            st.run_cells([(cells("c") + cells("d"), (17,), streams)])
        broken = {k: dict(cell, status="TIMEOUT") for k, cell in first.items()}
        halt = call(lambda _o: st.need_ok(broken, "somewhere"), None)
        assert halt.check == "somewhere" and len(halt.detail["not OK"]) == 2
        with pytest.raises(ps.Halt):
            st.accuracy(broken, st.key("carried", 17, host="port", window=16))


# ---- migrate -------------------------------------------------------------------------------------


def test_migrate_reads_the_gate_on_both_hosts_and_files_the_quantization_beside_it(world):
    out, _ = world
    body = summary(out, "migrate")
    assert body["reading"] == "hosted" and body["use"] == "kernel" and body["shortened"] is True
    assert body["scale"] == SCALE and body["window"] == 16 and all(body["conjuncts"].values())
    assert set(body["mean"]) == {"port", "kernel"} and set(body["mean"]["port"]) == {"live", "task"}
    assert cs.within(body["mean"]["kernel"]["live"], body["mean"]["port"]["live"])
    assert cs.within(body["mean"]["kernel"]["task"], body["mean"]["port"]["task"])
    assert body["machinery_red"]["fired"] and body["machinery_red"]["isolated"] == "port"
    assert body["machinery_red"]["control"] == "hosted" and not cs.within(
        body["machinery_red"]["D_task_reset"], body["machinery_red"]["D_task_port"]
    )
    # the gate's four checks ran as "call" jobs and every one holds
    names = ("parity", "r_b3", "r_b3_red", "r_b4_red")
    assert all(body["checks"][name]["verdict"] for name in names)
    assert body["checks"]["r_b3"]["numbers"]["host"] == "kernel"
    # R-b4 on the hosted form is a fresh-process TWIN of every hosted cell (P5 panel F7), beside
    # the red
    twins = body["r_b4_twins"]
    # "every hosted cell" (2.9 as amended): the live gate seeds AND the synthetic gate cells (R1)
    assert set(twins) == {f"{form} {seed}" for form in ps.FIT_FORMS for seed in HOSTED_SEEDS}
    assert len(twins) == 16
    assert all(
        row == {"pool_differs": False, "learner_differs": False, "executions": True}
        for row in twins.values()
    )
    assert (
        body["seed_set"] == "live"
        and (out / "migrate" / "twin" / "twin-carried-204.json.gz").is_file()
        and (out / "migrate" / "twin" / "twin-settled-5005.json.gz").is_file()
    )
    # the replays: TUNED x {carried, settled} x {port, hosted} on 204-207 and 5005-5008
    for host in st.HOSTS:
        for where, seeds in (("live", cs.GATE_SEEDS), ("task", cs.GATE_SYNTHETIC_SEEDS)):
            assert set(body["accuracy"][host][where]["readout"]["carried"]) == {
                str(s) for s in seeds
            }
            assert set(body["d"][host][where]) == {"readout", "grower"}
    assert (
        out / "migrate" / "synthetic" / "task-5005-stream.json.gz"
    ).is_file()  # regenerated at this head
    assert (
        len(md._load(out / "migrate" / "synthetic" / "task-5005-stream.json.gz")["events"]) == 300
    )
    hosted = md._load(out / "migrate" / "kernel" / "lsm-00-carried-204.json.gz")["pool"]
    assert hosted["host"] == "kernel" and hosted["host_policy"] == rp.KernelLsmPool.lineage_policy(
        cs.TUNED
    )
    assert "host" not in md._load(out / "migrate" / "port" / "lsm-00-carried-204.json.gz")["pool"]
    # the port's per-seed D beside G1(b)'s four numbers
    versus = body["port_vs_g1b"]
    assert versus["paired"] is True and versus["g1b_seeds"] == [204, 205, 206, 207]
    assert versus["seeds"] == [204, 205, 206, 207] and versus["g1b"] == [
        -0.0063,
        0.0082,
        0.0007,
        -0.0276,
    ]
    assert [round(a - b, 12) for a, b in zip(versus["port"], versus["g1b"], strict=True)] == [
        round(x, 12) for x in versus["difference"]
    ]


def test_the_quantization_sensitivity_is_report_only_and_its_instrument_agrees_at_the_binding(
    world,
):
    out, _ = world
    quant = summary(out, "migrate")["quantization"]
    binding = rp.KernelLsmPool.lineage_policy(cs.TUNED)["frac_bits"]
    assert quant["report_only"] and quant["binding_frac_bits"] == binding == 12
    # the six widths of 2.9's "frac_bits 4-32" (P5 panel F8), the same tuple the parity horizon
    # reads
    assert st.FRAC_BITS_SWEEP == (4, 8, 12, 16, 24, 32)
    assert set(quant["rows"]) == {str(b) for b in st.FRAC_BITS_SWEEP} and all(
        row["complete"] for row in quant["rows"].values()
    )
    assert quant["table_agreement"] == {"live": True, "task": True}
    table = summary(out, "migrate")["accuracy"]["kernel"]
    for where, seeds in (("live", cs.GATE_SEEDS), ("task", cs.GATE_SYNTHETIC_SEEDS)):
        for (
            form
        ) in ps.FIT_FORMS:  # the call job at the binding equals the hosted table's own replay
            assert {s: quant["rows"]["12"]["accuracy"][form][str(s)] for s in seeds} == {
                s: table[where]["readout"][form][str(s)] for s in seeds
            }
    # never filed in R-b7's table: the hosted table holds the binding's cells alone
    assert sorted(p.name for p in (out / "migrate" / "kernel").glob("*.json.gz")) == sorted(
        f"lsm-00-{form}-{seed}.json.gz"
        for form in ps.FIT_FORMS
        for seed in (*cs.GATE_SEEDS, *cs.GATE_SYNTHETIC_SEEDS)
    )
    assert (
        rp.HOSTS["kernel"].lineage_policy(cs.TUNED)["frac_bits"] == 12
    )  # a binding R-b7 still holds


def test_the_parity_horizon_is_reported_at_the_one_sweeps_widths_and_never_decides(world):
    out, _ = world
    body = summary(out, "migrate")
    horizon = body["parity_horizon"]
    assert horizon["report_only"] and set(horizon["rows"]) == {str(b) for b in st.FRAC_BITS_SWEEP}
    assert set(horizon["rows"]) == set(body["quantization"]["rows"])  # one tuple, two readers
    rows = {int(bits): row for bits, row in horizon["rows"].items()}
    assert all(row["ticks"] == 48 and row["frac_bits"] == bits for bits, row in rows.items())
    assert (
        rows[4]["equal_fraction"] < rows[32]["equal_fraction"] == 1.0
    )  # a coarse lattice parts early
    assert rows[32]["first_diverging_tick"] is None
    assert "parity_horizon" not in body["conjuncts"]  # reported beside the gate, never a conjunct


def test_the_parity_horizon_of_one_recording_reads_where_the_hosted_pool_parts_from_the_port(
    tmp_path,
):
    path = tmp_path / "rb6.json.gz"
    md._save(path, md.rb6_stream(RB6))
    narrow, binding, wide = (st.parity_horizon(str(path), b) for b in (4, 12, 32))
    assert narrow["first_diverging_tick"] == 0 and narrow["equal_fraction"] < 0.1
    assert binding["first_diverging_tick"] is None or binding["first_diverging_tick"] > 0
    assert wide["first_diverging_tick"] is None and wide["equal_fraction"] == 1.0
    assert wide["ticks"] == 1200 and wide["report_only"] and wide["frac_bits"] == 32
    # the same recording at the binding gives the same answer twice (deterministic)
    assert st.parity_horizon(str(path), 12) == binding


def migrate_of(world, tmp_path, children, *, patch=None):
    out = copy_of(
        world, tmp_path, "migrate", "causality", "primary", "controls", "transport", "separation"
    )
    with shortened(children) as monkeypatch:
        # the sweep is report-only, and the one tuple both its readers use: the reds skip it, the
        # binding width alone stays (the quantization instrument's agreement)
        monkeypatch.setattr(st, "FRAC_BITS_SWEEP", ())
        if patch:
            patch(monkeypatch)
        return call(functools.partial(st.stage_migrate, scale=SCALE), out), out


@pytest.mark.parametrize("name", ["check_host_parity", "check_r_b3"])
def test_a_failed_gate_conjunct_reads_port_and_the_stage_still_completes(world, tmp_path, name):
    held = {"verdict": False, "numbers": {}}
    body, _ = migrate_of(world, tmp_path, Children(overrides={name: held}))
    assert isinstance(body, dict) and body["reading"] == "port" and body["use"] == "port"
    key = {"check_host_parity": "parity", "check_r_b3": "r_b3"}[name]
    assert body["conjuncts"][key] is False and sum(not v for v in body["conjuncts"].values()) == 1


def test_a_hosted_twin_that_differs_in_the_pool_fails_the_gate_and_in_a_learner_halts(
    world, tmp_path
):
    """P5 panel F7: R-b4 on the hosted form is the fresh-process TWIN comparison, as the primary's
    is."""
    pool_only = lambda m: m.setattr(ps, "_twins_differ", lambda a, b: (True, False))  # noqa: E731
    body, _ = migrate_of(world, tmp_path, Children(), patch=pool_only)
    assert isinstance(body, dict) and body["reading"] == "port" and body["use"] == "port"
    assert (
        body["conjuncts"]["r_b4"] is False and sum(not v for v in body["conjuncts"].values()) == 1
    )
    assert all(row["pool_differs"] for row in body["r_b4_twins"].values())
    # a learner-state difference is a shared-instrument failure (2.9, 2.11): it halts, never reads
    # "port"
    learner = lambda m: m.setattr(ps, "_twins_differ", lambda a, b: (False, True))  # noqa: E731
    halt, _ = migrate_of(world, tmp_path / "learner", Children(), patch=learner)
    assert isinstance(halt, ps.Halt) and halt.check == "R-b4"
    assert all(row["learner_differs"] for row in halt.detail.values())
    # twins in one process, and a timed-out twin, halt as the primary's do
    halt, _ = migrate_of(world, tmp_path / "one", Children(patch_pid=False))
    assert isinstance(halt, ps.Halt) and halt.check == "R-b4"
    twin = lambda spec: "twin-" in spec["out"]  # noqa: E731
    halt, _ = migrate_of(world, tmp_path / "late", Children(timeouts=twin))
    assert isinstance(halt, ps.Halt) and halt.check == "R-b4 TIMEOUT" and len(halt.detail) == 16


def test_a_hosted_twin_difference_confined_to_a_synthetic_gate_cell_is_read_like_a_live_one(
    world, tmp_path
):
    """Re-panel R1 (2.9 as amended: the twins cover "every hosted cell"): the hosted table also
    holds the synthetic gate cells, so a difference on one of them alone halts R-b4 when it is a
    learner's and reads "port" when it is the pool's; a clean run stays "hosted" with sixteen
    twins."""
    synthetic = set(cs.GATE_SYNTHETIC_SEEDS)
    cells = {f"{form} {seed}" for form in ps.FIT_FORMS for seed in synthetic}

    def only(pool, learner):
        def found(a, b):
            mine = a["seed"] in synthetic
            return pool and mine, learner and mine

        return lambda m: m.setattr(ps, "_twins_differ", found)

    halt, _ = migrate_of(world, tmp_path / "learner", Children(), patch=only(False, True))
    assert isinstance(halt, ps.Halt) and halt.check == "R-b4"
    assert {name for name, row in halt.detail.items() if row["learner_differs"]} == cells
    assert len(halt.detail) == 16 and not any(row["pool_differs"] for row in halt.detail.values())
    body, _ = migrate_of(world, tmp_path / "pool", Children(), patch=only(True, False))
    assert isinstance(body, dict) and body["reading"] == "port" and body["use"] == "port"
    failed = [name for name, held in body["conjuncts"].items() if not held]
    assert failed == ["r_b4"]
    assert {name for name, row in body["r_b4_twins"].items() if row["pool_differs"]} == cells
    body, _ = migrate_of(world, tmp_path / "clean", Children(), patch=only(False, False))
    assert body["reading"] == "hosted" and len(body["r_b4_twins"]) == 16


def test_a_pool_noise_red_that_does_not_fire_halts_the_migration_it_is_no_conjunct(world, tmp_path):
    """The red stays a separate requirement (P5 panel F7): it halts when it fires on nothing, and
    the
    verdict of a hosted pool whose ordinary twins agree does not depend on it."""
    blind = {"verdict": False, "numbers": {}}
    halt, _ = migrate_of(world, tmp_path, Children(overrides={"pool_noise_red": blind}))
    assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire"
    assert "hosted R-b4 pool-noise red" in halt.detail
    gone = lambda spec: spec["kind"] == "call" and spec["target"].endswith("pool_noise_red")  # noqa: E731
    halt, _ = migrate_of(world, tmp_path / "gone", Children(timeouts=gone))
    assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire"


def test_a_gate_check_that_times_out_reads_port_and_a_blind_red_halts(world, tmp_path):
    def slow(spec):
        return spec["kind"] == "call" and spec["target"].endswith("check_r_b3")

    body, _ = migrate_of(world, tmp_path, Children(timeouts=slow))
    assert (
        body["reading"] == "port"
        and body["conjuncts"]["r_b3"] is False
        and body["checks"]["r_b3"] is None
    )
    blind = {"verdict": False, "numbers": {}}
    halt, _ = migrate_of(
        world, tmp_path / "blind", Children(overrides={"check_r_b3_hosted_red": blind})
    )
    assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire"
    assert "hosted R-b3 float-accumulation red" in halt.detail

    def gone(spec):
        return spec["kind"] == "call" and spec["target"].endswith("check_r_b3_hosted_red")

    halt, _ = migrate_of(world, tmp_path / "gone", Children(timeouts=gone))
    assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire"


def test_a_machinery_red_that_does_not_fire_halts(world, tmp_path):
    halt, _ = migrate_of(
        world, tmp_path, Children(), patch=lambda m: m.setattr(cs, "within", lambda *a, **k: True)
    )
    assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire"
    assert halt.detail["migration gate machinery red"]["fired"] is False


def test_a_hosted_replay_that_times_out_halts_and_so_does_a_missing_gate_recording(world, tmp_path):
    def hosted(spec):
        return (
            spec["kind"] == "replay"
            and spec.get("arm") == "pool"
            and (spec["pool"] or {}).get("host") == "kernel"
        )

    halt, _ = migrate_of(world, tmp_path, Children(timeouts=hosted))
    assert (
        isinstance(halt, ps.Halt)
        and halt.check == "migrate: kernel replays"
        and halt.detail["not OK"]
    )
    out = copy_of(world, tmp_path / "gone", "migrate")
    record = summary(out, "record")
    record["recordings"]["205"]["status"] = "TIMEOUT"
    md._save(ps._summary_path(out, "record"), record)
    with shortened():
        halt = call(functools.partial(st.stage_migrate, scale=SCALE), out)
    assert halt.check == "migrate" and halt.detail == {"gate recordings not OK": [205]}


def test_the_reserve_campaign_migrates_on_the_reserve_gate_seeds_and_pairs_nothing_with_g1b(
    world, tmp_path
):
    """P5 panel F2: the record's seed set picks ONE complete bundle; reserve = gate 208-211 and
    synthetic 5009-5012, and G1(b)'s 204-207 numbers stay labelled, never paired with them."""
    out = copy_of(
        world, tmp_path, "migrate", "causality", "primary", "controls", "transport", "separation"
    )
    record = summary(out, "record")
    record["seed_set"] = "reserve"
    record["recordings"].update({str(s): {"status": "OK"} for s in cs.GATE_RESERVE_SEEDS})
    md._save(ps._summary_path(out, "record"), record)
    for seed in cs.GATE_RESERVE_SEEDS:
        md._save(ps._recording(out, seed), fixture(seed))
    with shortened() as monkeypatch:
        monkeypatch.setattr(st, "FRAC_BITS_SWEEP", ())
        body = call(functools.partial(st.stage_migrate, scale=SCALE), out)
    assert isinstance(body, dict) and body["seed_set"] == "reserve"
    for where, seeds in (
        ("live", cs.GATE_RESERVE_SEEDS),
        ("task", cs.GATE_SYNTHETIC_RESERVE_SEEDS),
    ):
        assert set(body["accuracy"]["kernel"][where]["readout"]["carried"]) == {
            str(s) for s in seeds
        }
    assert set(body["r_b4_twins"]) == {
        f"{form} {seed}"
        for form in ps.FIT_FORMS
        for seed in (*cs.GATE_RESERVE_SEEDS, *cs.GATE_SYNTHETIC_RESERVE_SEEDS)
    }
    versus = body["port_vs_g1b"]
    assert versus["seeds"] == [208, 209, 210, 211] and versus["g1b_seeds"] == [204, 205, 206, 207]
    assert versus["paired"] is False and versus["difference"] is None
    assert versus["g1b"] == list(st.G1B_PER_SEED_D) and len(versus["port"]) == 4
    assert (out / "migrate" / "synthetic" / "task-5009-stream.json.gz").is_file()
    # a record that names no seed set cannot choose a bundle: a halt, never a silent live default
    del record["seed_set"]
    md._save(ps._summary_path(out, "record"), record)
    with shortened():
        halt = call(functools.partial(st.stage_migrate, scale=SCALE), out)
    assert isinstance(halt, ps.Halt) and halt.check == "seed bundle"


def test_a_quantization_instrument_that_disagrees_with_the_hosted_table_halts(world, tmp_path):
    def off(spec):
        real = st.quantized_decoder(**spec["args"])
        return {**real, "accuracy": real["accuracy"] + 0.25}

    halt, _ = migrate_of(world, tmp_path, Children(overrides={"quantized_decoder": off}))
    assert (
        isinstance(halt, ps.Halt)
        and halt.check == "the quantization instrument disagrees with the hosted table"
    )
    assert halt.detail == {"live": False, "task": False}


# ---- causality -----------------------------------------------------------------------------------


def test_causality_holds_on_the_surviving_host_at_three_cuts_for_both_learners(world):
    out, _ = world
    body = summary(out, "causality")
    assert body["host"] == "kernel" and body["seed"] == 203 and body["cuts"] == [12, 24, 36]
    assert set(body["R-b1"]) == {f"{form} {cut}" for form in ps.FIT_FORMS for cut in (12, 24, 36)}
    assert all(row == {"grower": True, "readout": True} for row in body["R-b1"].values())
    assert body["R-b1 red"] == {
        "12": True,
        "24": True,
        "36": True,
    }  # the (leak) arm changes predictions at every cut
    hosted = md._load(out / "causality" / "carried.json.gz")["pool"]
    assert hosted["host"] == "kernel"


def test_a_leak_that_does_not_change_predictions_halts_causality_as_a_red_that_does_not_fire(
    world, tmp_path
):
    out = copy_of(world, tmp_path, "causality")
    with shortened() as monkeypatch:
        monkeypatch.setattr(
            md, "mutate_suffix", lambda stream, t_star, seed: stream
        )  # no cut is visible
        halt = call(functools.partial(st.stage_causality, scale=SCALE), out)
    assert halt.check == "a red that does not fire" and halt.detail["R-b1 leak"] == {
        "12": False,
        "24": False,
        "36": False,
    }


def test_a_pool_whose_predictions_read_the_suffix_halts_r_b1(world, tmp_path):
    out = copy_of(world, tmp_path, "causality")
    original = md._source_digest(md._load(ps._recording(out, 203)))
    real = md.replay

    def leaky(stream, arm, *args, **kwargs):
        run = real(stream, arm, *args, **kwargs)
        if (
            arm == "pool" and md._source_digest(stream) != original
        ):  # a mutated suffix moves an early prediction
            run["predictions"][0] = 99
        return run

    with shortened() as monkeypatch:
        monkeypatch.setattr(md, "replay", leaky)
        halt = call(functools.partial(st.stage_causality, scale=SCALE), out)
    assert halt.check == "R-b1" and not all(all(row.values()) for row in halt.detail.values())
    out = copy_of(world, tmp_path / "gone", "causality")
    record = summary(out, "record")
    record["recordings"]["203"]["status"] = "TIMEOUT"
    md._save(ps._summary_path(out, "record"), record)
    with shortened():
        assert call(functools.partial(st.stage_causality, scale=SCALE), out).check == "R-b1"
    out = copy_of(world, tmp_path / "late", "causality")
    with shortened(
        Children(timeouts=lambda spec: spec["kind"] == "replay" and spec.get("arm") == "leak")
    ):
        assert call(functools.partial(st.stage_causality, scale=SCALE), out).check == "R-b1 TIMEOUT"


# ---- one R-b1 construction for both bodies (G1(c) P5 panel F10) ----------------------------------

# sha256[:16] of every job spec (paths made OUT-relative) that each body built on the 48-event
# fixture, at 96d3d999 BEFORE the shared helper replaced the two hand-written bodies: the helper
# must reproduce them.
PINNED_R_B1_JOBS = {"check203": ("8ea8c9bf3d80ee82", 15), "causality": ("4c3e0fc99d859dee", 12)}
PINNED_R_B1_BODIES = {
    "check203": {
        "R-b1": {
            "lsm carried": {"12": True, "24": True, "36": True},
            "lsm settled": {"12": True, "24": True, "36": True},
        },
        "R-b1 red": {"12": True, "24": True, "36": True},
        "R-b4": {"lsm": {"executions": True, "learner_differs": False, "pool_differs": False}},
        "R-b4 red": {"lsm": True},
        "cuts": [12, 24, 36],
        "refused": {},
    },
    "causality": {
        "R-b1": {
            f"{form} {cut}": {"grower": True, "readout": True}
            for form in ("carried", "settled")
            for cut in (12, 24, 36)
        },
        "R-b1 red": {"12": True, "24": True, "36": True},
        "cuts": [12, 24, 36],
        "host": "port",
        "scale": SCALE,
        "seed": 203,
        "window": 16,
    },
}


def scrub(value, root):
    """The value with every path OUT-relative and forward-slashed (a temp name never pins)."""
    if isinstance(value, dict):
        return {k: scrub(v, root) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(v, root) for v in value]
    if isinstance(value, str):
        return value.replace(str(root), "<OUT>").replace("\\", "/")
    return value


def digest_of_jobs(specs, root):
    jobs = {scrub(spec, root)["out"]: scrub(spec, root) for spec in specs}
    return hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()[:16], len(jobs)


def test_both_r_b1_bodies_build_one_helper_and_it_reproduces_what_each_built_before(tmp_path):
    seed = ps.CHECK_SEED
    g1b, driver = tmp_path / "g1b", tmp_path / "driver"
    for out in (g1b, driver):
        ps._folder(out, "record")
        md._save(ps._recording(out, seed), fixture(seed))
        write_summary(
            out, "record", recordings={str(seed): {"status": "OK"}}, live_seeds=[], seed_set="live"
        )
    write_summary(g1b, "tune", candidates={"lsm": {"tuned": {"config": cs.TUNED}}}, refused={})
    write_summary(driver, "migrate", use="port")
    built, real_spec, bodies, specs = [], ps._spec, {}, {}

    class Spy(ps.RB1Check):
        def __init__(self, *args, **kwargs):
            built.append(kwargs.get("leak", "leak"))
            super().__init__(*args, **kwargs)

    def small(candidate, config, form, noise=False, window=rp.WINDOW, host="port"):
        window = (
            SCALE["window"] if window == rp.WINDOW else window
        )  # the 48-event fixture: window 16
        return real_spec(candidate, config, form, noise=noise, window=window, host=host)

    for name, body in (
        ("check203", ps.step_check203),
        ("causality", functools.partial(st.stage_causality, scale=SCALE)),
    ):
        children = Children()
        with shortened(children) as patch:
            patch.setattr(ps, "RB1Check", Spy)
            patch.setattr(ps, "_spec", small)
            bodies[name] = scrub(body(g1b if name == "check203" else driver), tmp_path)
        specs[name] = children.specs
    assert built == ["R-b1 leak", "leak"]  # both bodies called the one helper, once each
    for name, specs_run in specs.items():
        assert digest_of_jobs(specs_run, tmp_path) == PINNED_R_B1_JOBS[name], name
        assert bodies[name] == PINNED_R_B1_BODIES[name], name
    for name, folder in (("check203", g1b / "check203"), ("causality", driver / "causality")):
        original = md._load(ps._recording(g1b if name == "check203" else driver, seed))
        for cut in (
            12,
            24,
            36,
        ):  # the cut streams are the recording's suffix mutation, seeded as before
            moved = md._load(folder / f"{seed}-cut-{cut}.json.gz")
            assert moved == md.mutate_suffix(original, cut, original["seed"] + cut), (name, cut)
    assert hasattr(st, "_held") is False  # the stage's own prefix reader is retired


def test_the_r_b1_helper_refuses_to_read_a_replay_bound_to_another_filing(tmp_path):
    seed, folder = ps.CHECK_SEED, tmp_path / "rb1"
    folder.mkdir()
    ps._folder(tmp_path, "record")
    stream, path = fixture(seed), ps._recording(tmp_path, seed)
    md._save(path, stream)
    spec = ps._spec("lsm", cs.TUNED, "carried", window=16)
    check = ps.RB1Check(folder, seed, path, stream, {"pool x": ("x", spec)})
    assert check.cuts == (12, 24, 36) and set(check.bound) == {"pool x"}
    assert sorted(check.jobs) == sorted(
        [f"pool x {tag}" for tag in ("base", 12, 24, 36)]
        + [f"leak {tag}" for tag in ("base", 12, 24, 36)]
    )
    with shortened(Children()):
        statuses = ps.run_jobs(list(check.jobs.values()))
        assert {s["status"] for s in statuses} == {"OK"}
        assert check.red() == {"12": True, "24": True, "36": True}
        assert all(all(equal) for equal in check.held("pool x").values())
        elsewhere = ps._spec(
            "lsm", cs.TUNED, "settled", window=16
        )  # filed under a spec it did not run
        check.bound["pool x"] = ps._bound(seed, elsewhere)
        with pytest.raises(ps.Halt, match="R-b7"):
            check.held("pool x")


# ---- primary -------------------------------------------------------------------------------------


def test_primary_runs_four_forms_on_every_live_seed_with_its_tables(world):
    out, _ = world
    body = summary(out, "primary")
    assert (
        body["host"] == "kernel"
        and body["window"] == 16
        and body["live_seeds"] == list(LIVE)
        and body["shortened"]
    )
    assert set(body["accuracy"]["readout"]) == set(rp.FORMS)
    for learner in st.LEARNERS:
        for seed in map(str, LIVE):
            a = body["accuracy"][learner]
            assert body["d"][learner][seed] == a["carried"][seed] - a["settled"][seed]
    for form in rp.FORMS:  # the per-seed, per-form, per-segment tables 2.3 requires
        for seed in map(str, LIVE):
            cell = body["cells"][form][seed]
            assert (
                cell["status"] == "OK"
                and cell["readout"]
                and cell["grower"]
                and "degeneracy" in cell
            )
            for learner in st.LEARNERS:
                row = cell[learner]["unseen-order"]
                assert {"accuracy", "coverage", "prediction_entropy", "prior_fallback", "n"} <= set(
                    row
                )
    assert all(
        body["cells"]["carried"][str(s)]["startup"]["flagged"] in (True, False) for s in LIVE
    )
    assert body["cells"]["settled"]["300"]["startup"] is None
    assert set(body["input_response"]["300"]) == {"driven", "mask", "mean", "sd"}
    assert body["input_response"]["300"]["driven"] == 256
    # R-b4: a second fresh-process replay of Carried and Settled on every seed; the pool-noise red
    # fired
    assert set(body["R-b4"]) == {f"{form} {seed}" for form in ps.FIT_FORMS for seed in LIVE}
    assert all(
        row == {"pool_differs": False, "learner_differs": False, "executions": True}
        for row in body["R-b4"].values()
    )
    assert body["R-b4 red"] == {"seed": 100, "fired": True}
    assert (out / "primary" / "twin-carried-300.json.gz").is_file() and (
        out / "primary" / "noise-red-noise.json.gz"
    ).is_file()
    host = md._load(out / "primary" / "lsm-00-front_end-301.json.gz")["pool"]
    assert host["host"] == "kernel" and host["window"] == 16


def primary_of(world, tmp_path, children=None, patch=None):
    out = copy_of(world, tmp_path, "primary", "controls", "transport", "separation")
    with shortened(children) as monkeypatch:
        if patch:
            patch(monkeypatch)
        return call(functools.partial(st.stage_primary, scale=SCALE), out)


def test_an_r_b4_difference_halts_whether_the_pool_or_a_learner_moved(world, tmp_path):
    for found, name in (
        ((True, False), "pool"),
        ((False, True), "learner"),
        ((True, True), "both"),
    ):
        halt = primary_of(
            world,
            tmp_path / name,
            patch=lambda m, f=found: m.setattr(ps, "_twins_differ", lambda a, b: f),
        )
        assert isinstance(halt, ps.Halt) and halt.check == "R-b4", name
        assert any(row["pool_differs"] or row["learner_differs"] for row in halt.detail.values())


def test_twins_that_ran_in_one_process_halt_and_a_timed_out_twin_halts(world, tmp_path):
    halt = primary_of(world, tmp_path, Children(patch_pid=False))
    assert (
        isinstance(halt, ps.Halt)
        and halt.check == "R-b4"
        and halt.detail == "the twins ran in one process"
    )
    twin = lambda spec: "twin-" in spec["out"]  # noqa: E731
    halt = primary_of(world, tmp_path / "late", Children(timeouts=twin))
    assert halt.check == "R-b4 TIMEOUT" and len(halt.detail) == 6
    cell = lambda spec: spec["out"].endswith("lsm-00-reset-301.json.gz")  # noqa: E731
    halt = primary_of(world, tmp_path / "cell", Children(timeouts=cell))
    assert halt.check == "primary" and len(halt.detail["not OK"]) == 1


def test_a_pool_noise_red_that_does_not_fire_halts_the_primary(world, tmp_path):
    halt = primary_of(
        world, tmp_path, patch=lambda m: m.setattr(ps, "noise_red_fires", lambda a, b: False)
    )
    assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire"
    assert halt.detail == {"R-b4 pool noise (seed 100)": False}
    noise = lambda spec: "noise-red-" in spec["out"]  # noqa: E731
    halt = primary_of(world, tmp_path / "late", Children(timeouts=noise))
    assert halt.check == "R-b4 TIMEOUT"


def test_a_replay_bound_to_another_host_is_refused_by_r_b7(world, tmp_path):
    def tamper(spec, run):
        if (
            spec["arm"] == "pool"
            and run["pool"].get("host") == "kernel"
            and spec["out"].endswith("settled-300.json.gz")
        ):
            run["pool"]["host_policy"] = {**run["pool"]["host_policy"], "frac_bits": 4}

    halt = primary_of(world, tmp_path, Children(tamper=tamper))
    assert isinstance(halt, ps.Halt) and halt.check == "R-b7" and "host_policy" in halt.detail


# ---- controls ------------------------------------------------------------------------------------


def test_controls_run_the_pool_control_the_shuffle_and_window_zero_on_the_surviving_host(world):
    out, _ = world
    body = summary(out, "controls")
    assert body["host"] == "kernel" and body["window"] == 16 and body["shortened"]
    control = body["control"]
    assert control["seed"] == RB6 and control["bar"] == md.RB6_BAR
    assert (
        control["positive"]["unshuffled"] >= md.RB6_BAR > control["positive"]["shuffled"]
        and control["positive"]["sighted"]
    )
    assert (
        control["aa"]["held"]
        and max(control["aa"]["unshuffled"], control["aa"]["shuffled"]) < md.RB6_BAR
    )
    invariance = control["invariance"]
    assert invariance["settled_ok"] and invariance["red_fired"]
    assert len(invariance["settled"]) == len(invariance["carried_red"]) == cs.SHUFFLE_K
    for held, red in zip(invariance["settled"], invariance["carried_red"], strict=True):
        assert (
            held["ok"]
            and held["multiset_equal"] == held["blocks"] >= 1
            and held["sequence_differs"] == held["blocks"]
        )
        assert (
            not red["ok"] and red["multiset_equal"] < red["blocks"]
        )  # Carried's history reaches phi: the red
    assert body["halves"] == {
        "pool_free": True,
        "pool_positive": True,
        "aa": True,
        "settled_invariance": True,
        "carried_red": True,
    }
    assert body["sighted"] is True


def test_the_shuffle_writes_k_streams_a_seed_and_d_shuf_is_the_mean_over_them(world):
    out, _ = world
    body = summary(out, "controls")
    shuffle = out / "controls" / "shuffle"
    for seed in LIVE:
        files = sorted(p.name for p in shuffle.glob(f"{seed}-shuf*-stream.json.gz"))
        assert files == [f"{seed}-shuf{k}-stream.json.gz" for k in range(cs.SHUFFLE_K)]
        plain, moved = md._load(ps._recording(out, seed)), md._load(shuffle / files[1])
        assert cs.check_shuffle_transform(plain, moved) == {"ok": True, "reasons": []}
        assert moved["shuffle"] == {"seed": seed, "block": 256, "k": 1}
        # each shuffled stream's replays are bound to the shuffled stream, not the recording
        run = md._load(shuffle / f"{seed}-shuf1-carried.json.gz")
        assert run["source_digest"] == md._source_digest(moved) != md._source_digest(plain)
        assert body["d_shuf"]["readout"][str(seed)] == pytest.approx(
            sum(body["d_shuf_k"][str(seed)]) / cs.SHUFFLE_K, abs=1e-12
        )
        assert len(body["d_shuf_k"][str(seed)]) == cs.SHUFFLE_K


def test_window_zero_reports_d0_and_w_and_holds_carried_identical_to_the_primarys(world):
    out, _ = world
    controls, primary = summary(out, "controls"), summary(out, "primary")
    zero = controls["window0"]
    assert zero["carried_identical"] is True
    for learner in st.LEARNERS:
        for seed in map(str, LIVE):
            d0 = (
                zero["accuracy"][learner]["carried"][seed]
                - zero["accuracy"][learner]["settled"][seed]
            )
            assert zero["d0"][learner][seed] == d0
            assert zero["w"][learner][seed] == d0 - primary["d"][learner][seed]
            # W is exactly accuracy(Settled@window) - accuracy(Settled@0), as the pre-registration
            # defines it
            assert zero["w"][learner][seed] == pytest.approx(
                primary["accuracy"][learner]["settled"][seed]
                - zero["accuracy"][learner]["settled"][seed],
                abs=1e-12,
            )
    assert zero["scored_rows_through_boundary"] == {str(seed): 0 for seed in LIVE}
    settled = md._load(out / "controls" / "window0" / "lsm-00-settled-300.json.gz")["pool"]
    assert (
        settled["window"] == 0 and settled["boundary"] is not None
    )  # the boundary fingerprint at construction


def test_scored_rows_through_a_boundary_counts_events_in_scored_segments_up_to_it():
    stream = md._synthetic_stream()  # 24 warm-up then 24 unseen-order events
    assert [ctl.scored_rows_through(stream, b) for b in (0, 24, 25, 30, 48, 99)] == [
        0,
        0,
        1,
        6,
        24,
        24,
    ]
    live = {
        "events": [
            {"segment": s} for s in ("warm-up", "repeated-canonical", None, "repeated-permuted")
        ]
    }
    assert ctl.scored_rows_through(live, 4) == 2 and ctl.scored_rows_through(live, 1) == 0


def controls_of(world, tmp_path, children=None, patch=None, **kwargs):
    out = copy_of(world, tmp_path, "controls", "transport", "separation")
    with shortened(children) as monkeypatch:
        if patch:
            patch(monkeypatch, out)
        return call(functools.partial(ctl.stage_controls, scale=SCALE, rb6_seed=RB6, **kwargs), out)


def test_a_blind_control_reads_as_not_sighted_and_the_stage_still_completes(world, tmp_path):
    blind = controls_of(world, tmp_path, patch=lambda m, out: m.setattr(md, "RB6_BAR", 10.0))
    assert (
        isinstance(blind, dict)
        and blind["sighted"] is False
        and blind["halves"]["pool_positive"] is False
    )
    assert (
        blind["halves"]["pool_free"] and blind["halves"]["settled_invariance"]
    )  # only the one half fails
    pool_free = controls_of(
        world,
        tmp_path / "pf",
        patch=lambda m, out: write_summary(
            out,
            "instrument",
            rb6_seed=RB6,
            rb6_ok=True,
            shuffle_control={"sighted": False},
            shuffle_aa={"held": True},
        ),
    )
    assert (
        pool_free["sighted"] is False
        and pool_free["halves"]["pool_free"] is False
        and pool_free["halves"]["pool_positive"]
    )
    aa = controls_of(
        world,
        tmp_path / "aa",
        patch=lambda m, out: write_summary(
            out,
            "instrument",
            rb6_seed=RB6,
            rb6_ok=True,
            shuffle_control={"sighted": True},
            shuffle_aa={"held": False},
        ),
    )
    assert aa["sighted"] is False and aa["halves"]["pool_free"] is False


def carried_check_returns(found):
    """A patch for `controls_of`: Settled's K invariance checks run for real (they come first), and
    every
    CARRIED one (the red) returns `found`."""
    real, calls = cs.settled_invariance, []

    def patch(monkeypatch, out):
        def invariance(plain_run, shuffled_run, stream, **kwargs):
            calls.append(1)
            result = real(plain_run, shuffled_run, stream, **kwargs)
            return result if len(calls) <= cs.SHUFFLE_K else found

        monkeypatch.setattr(cs, "settled_invariance", invariance)

    return patch


def test_a_carried_red_that_does_not_fire_halts_the_controls_and_a_vacuous_one_is_not_a_fire(
    world, tmp_path
):
    """P5 panel F3: the red fires only on a nonempty tested block set with an actual multiset
    difference; equal multisets with an identical sequence, an empty set and a held invariance do
    not, and a red that does not fire HALTS (2.11), it no longer reads `sighted: False`."""
    vacuous = {  # the panel's reproduction: nothing moved, so `ok` is False and the old red "fired"
        "ok": False,
        "per_block": [{"multiset_equal": True, "sequence_differs": False, "n": 8}],
        "excluded": 1,
    }
    empty = {"ok": False, "per_block": [], "excluded": 1}  # no block tested at all
    held = {  # Carried's invariance HOLDS: history did not reach phi, the red is blind
        "ok": True,
        "per_block": [{"multiset_equal": True, "sequence_differs": True}],
        "excluded": 0,
    }
    for name, found in (("vacuous", vacuous), ("empty", empty), ("held", held)):
        halt = controls_of(world, tmp_path / name, patch=carried_check_returns(found))
        assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire", name
        assert "Settled invariance, Carried red" in halt.detail
    assert not ctl.history_reaches_phi(vacuous) and not ctl.history_reaches_phi(empty)
    assert not ctl.history_reaches_phi(held)
    moved = {"per_block": [{"multiset_equal": True}, {"multiset_equal": False}]}
    assert ctl.history_reaches_phi(moved)  # one block whose multiset differs is the red firing
    # the control: the real Carried red fires on the stand-in world and the stage completes
    assert controls_of(world, tmp_path / "real")["halves"]["carried_red"] is True


def test_settled_invariance_failing_is_a_reading_and_the_blind_controls_stay_readings(
    world, tmp_path
):
    broken = {
        "ok": False,
        "per_block": [{"multiset_equal": False, "sequence_differs": True}],
        "excluded": 0,
    }
    always = controls_of(
        world,
        tmp_path / "b",
        patch=lambda m, out: m.setattr(cs, "settled_invariance", lambda *a, **k: broken),
    )
    assert (
        not always["halves"]["settled_invariance"]
        and always["halves"]["carried_red"]
        and always["sighted"] is False
    )


def test_a_shuffle_that_is_refused_or_fails_its_own_check_halts(world, tmp_path):
    def refuse(stream, k, **kwargs):
        raise ValueError("REFUSED: block at event 5: vision pointers are not one contiguous pass")

    halt = controls_of(
        world, tmp_path, patch=lambda m, out: m.setattr(cs, "shuffle_blocks", refuse)
    )
    assert (
        isinstance(halt, ps.Halt)
        and halt.check == "shuffle refused"
        and "contiguous pass" in halt.detail["why"]
    )

    def failed(stream, shuffled, **kwargs):
        return {"ok": False, "reasons": ["a slot field changed"]}

    halt = controls_of(
        world, tmp_path / "f", patch=lambda m, out: m.setattr(cs, "check_shuffle_transform", failed)
    )
    assert halt.check == "shuffle transform" and halt.detail["reasons"] == ["a slot field changed"]


def test_a_carried_at_zero_that_differs_from_the_primarys_carried_halts(world, tmp_path):
    def corrupt(monkeypatch, out):  # the primary's Carried replay of seed 301, one phi row changed
        path = out / "primary" / "lsm-00-carried-301.json.gz"
        run = md._load(path)
        run["phi"][3] += 1
        md._save(path, run)

    halt = controls_of(world, tmp_path, patch=corrupt)
    assert (
        isinstance(halt, ps.Halt)
        and halt.check == "window 0: Carried@0 is not the primary's Carried"
    )
    assert halt.detail == {"301": ["phi"]}
    differs = ctl._carried_differences(
        {
            "phi": [1],
            "predictions": [1],
            "fallbacks": [1],
            "fingerprint": 1,
            "refits": [],
            "readout": {"predictions": [1], "fingerprint": 1},
            "pool": {"state_chain": 1, "state": 1},
        },
        {
            "phi": [1],
            "predictions": [2],
            "fallbacks": [1],
            "fingerprint": 1,
            "refits": [],
            "readout": {"predictions": [1], "fingerprint": 2},
            "pool": {"state_chain": 3, "state": 1},
        },
    )
    assert differs == ["predictions", "readout fingerprint", "pool state_chain"]


def test_a_control_on_another_rb6_seed_than_the_instruments_halts(world, tmp_path):
    out = copy_of(world, tmp_path, "controls", "transport", "separation")
    with shortened():
        halt = call(functools.partial(ctl.stage_controls, scale=SCALE, rb6_seed=4244), out)
    assert halt.check == "controls" and "4243" in halt.detail


@pytest.mark.parametrize(("seed_set", "rb6"), [("live", 6000), ("reserve", 6001)])
def test_the_controls_default_to_the_seed_bundles_rb6_seed_and_never_run_it_here(
    world, tmp_path, seed_set, rb6
):
    """P5 panel F2: `rb6_seed` defaults to the record's bundle (6000 live, 6001 reserve). Stand-ins
    only:
    the pool control is replaced by a spy that stops the stage, so neither seed is ever run."""
    out = copy_of(world, tmp_path, "controls", "transport", "separation")
    record = summary(out, "record")
    md._save(ps._summary_path(out, "record"), {**record, "seed_set": seed_set})
    instrument = summary(out, "instrument")
    md._save(ps._summary_path(out, "instrument"), {**instrument, "rb6_seed": rb6})
    asked = []

    class Stop(Exception):
        pass

    def spy(out_, folder, host, window, seed):
        asked.append(seed)
        raise Stop

    with shortened() as patch:
        patch.setattr(ctl, "_pool_control", spy)
        with pytest.raises(Stop):
            ctl.stage_controls(out, scale=SCALE)
    assert asked == [rb6]
    # the instrument stage that ran the OTHER bundle's R-b6 seed is refused, never mixed
    other = 6001 if rb6 == 6000 else 6000
    md._save(ps._summary_path(out, "instrument"), {**instrument, "rb6_seed": other})
    halt = call(functools.partial(ctl.stage_controls, scale=SCALE), out)
    assert halt.check == "controls" and str(other) in halt.detail


# ---- transport -----------------------------------------------------------------------------------


def test_transport_runs_the_three_arms_on_the_taps_of_every_seed(world):
    out, _ = world
    body = summary(out, "transport")
    assert body["host"] == "kernel" and body["b"] == 16 and body["live_seeds"] == list(LIVE)
    assert (
        body["forbidden_request"]["raised"]
        and "forbidden feature channel" in body["forbidden_request"]["message"]
    )
    assert set(body["arms"]) == set(md.TRANSPORT_ARMS)
    for seed in map(str, LIVE):
        tap = body["taps"][seed]
        assert tap["event"] == 16 and tap["L"] >= 1
        copy = body["arms"]["pool-copy"]["records"][seed]
        record = copy["transport"]
        assert (
            record["form"] == "copy" and record["L"] == tap["L"] and record["copied_at_event"] == 16
        )
        assert record["padding_fraction"] == (rp.N - tap["L"]) / rp.N  # from the binding
        assert len(record["raw_row_norms"]) == tap["L"] and record["mean_row_norm"] > 0
        random = rp.w_in(int(seed))
        block = np.asarray(
            md._load(out / "transport" / f"tap-{seed}.json.gz")["tap"]["w_at_b"]["W"],
            dtype=np.float64,
        )
        assert copy["encoder"] == md._crc(rp.transported_encoder(random, block))
        swap = body["arms"]["pool-swap"]["records"][seed]
        assert swap["transport"]["form"] == "swap" and swap["transport"]["swapped_at_event"] == 16
        assert swap["encoder"] == md._crc(rp.fresh_encoder(int(seed), random))
        act = body["arms"]["pool-act"]["records"][seed]
        assert (
            act["encoder"] is None
            and act["transport"]["events"] == 48
            and "silent_events" in act["transport"]
        )
    for arm in md.TRANSPORT_ARMS:
        arm_d = body["arms"][arm]["d"]
        for learner in st.LEARNERS:
            acc = body["arms"][arm]["accuracy"][learner]
            assert arm_d[learner]["300"] == acc["carried"]["300"] - acc["settled"]["300"]


def received(stream, arm, host, window, tap=None):
    """The drive g u a pool of `arm` ACTUALLY takes in a fresh replay of `stream`, read where it
    takes it (`_Pool.percept_drive`, spied): an instrument independent of ps.input_response, which
    the saved statistics must agree with (two instruments that must agree)."""
    seen = []
    real = rp._Pool.percept_drive

    def spy(self, u):
        seen.append(cs.TUNED["g"] * np.asarray(u, dtype=np.float64))
        return real(self, u)

    spec = ps._spec(st.POOL, cs.TUNED, "carried", window=window, host=host)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rp._Pool, "percept_drive", spy)
        md.replay(stream, arm, pool=spec, side=tap)
    assert len(seen) == len(stream["events"])  # one drive a delivery
    return {"mean": float(np.mean(seen)), "sd": float(np.std(seen))}


def agree(saved, fresh):
    assert saved["driven"] == rp.N and saved["mask"] == list(range(rp.N))
    assert saved["mean"] == pytest.approx(fresh["mean"], rel=1e-9, abs=1e-12)
    assert saved["sd"] == pytest.approx(fresh["sd"], rel=1e-9, abs=1e-12)


def test_transport_files_each_arms_input_response_as_its_pool_actually_received_it(world):
    """Re-panel R3 (2.3: the input-response statistics beside every contrasted variant): pool-copy
    and pool-swap change the encoder at the event the replay recorded, pool-act drives it with the
    tap's grower activity; each seed's saved statistics equal what a fresh replay's pool took, and
    differ from the primary's (a helper that ignored the variant would file the primary's)."""
    out, _ = world
    body, primary = summary(out, "transport"), summary(out, "primary")["input_response"]
    for arm in md.TRANSPORT_ARMS:
        assert set(body["arms"][arm]["input_response"]) == {str(seed) for seed in LIVE}
        for seed in LIVE:
            stream = md._load(ps._recording(out, seed))
            tap = ps._side(str(out / "transport" / f"tap-{seed}.json.gz"))
            saved = body["arms"][arm]["input_response"][str(seed)]
            agree(saved, received(stream, arm, body["host"], body["window"], tap))
            assert (saved["mean"], saved["sd"]) != (
                primary[str(seed)]["mean"],
                primary[str(seed)]["sd"],
            )
    # the control for the instrument itself: the primary's own statistics agree with a fresh replay
    for seed in LIVE:
        stream = md._load(ps._recording(out, seed))
        agree(primary[str(seed)], received(stream, "pool", body["host"], body["window"]))


def test_separation_files_the_input_response_of_its_own_streams(world):
    """Re-panel R3: the XOR and twin streams are other streams, so other inputs through the seed's
    encoder: filed per stream and seed, equal to a fresh replay's received drive, and not the
    primary's."""
    out, _ = world
    body, primary = summary(out, "separation"), summary(out, "primary")
    assert set(body["input_response"]) == {"xor", "twin"}
    for name, per_seed in body["input_response"].items():
        assert set(per_seed) == {str(seed) for seed in LIVE}
        for seed in LIVE:
            stream = md._load(out / "separation" / f"{name}-{seed}-stream.json.gz")
            agree(per_seed[str(seed)], received(stream, "pool", body["host"], body["window"]))
            assert per_seed[str(seed)]["mean"] != primary["input_response"][str(seed)]["mean"]


def test_the_shuffles_and_window_zero_drive_the_pool_with_the_primarys_input_response(world):
    """Re-panel R3: why carried_state_controls files no statistics of its own for them. The per-unit
    drive statistics are those of the same numbers: a shuffle moves whole percepts inside blocks
    (the multiset of x, and the recording's seed, so W_in, are kept), and the window is read by the
    pool's adaptation only, never by the drive. Shown through a fresh replay's received drive."""
    out, _ = world
    host = summary(out, "primary")["host"]
    stream = md._load(ps._recording(out, LIVE[0]))
    plain = ps.input_response(st.POOL, cs.TUNED, stream)
    base = received(stream, "pool", host, 16)
    assert received(stream, "pool", host, 0) == base  # window 0: bit for bit the same drive
    for k in range(cs.SHUFFLE_K):
        shuffled = cs.shuffle_blocks(stream, k)
        assert cs.check_shuffle_transform(stream, shuffled)["ok"]
        assert shuffled["seed"] == stream["seed"]
        again = ps.input_response(st.POOL, cs.TUNED, shuffled)
        assert again["mask"] == plain["mask"] and again["driven"] == plain["driven"]
        for key in ("mean", "sd"):  # the sum is reordered: equal as numbers, to float rounding
            assert again[key] == pytest.approx(plain[key], rel=1e-12, abs=1e-14)
        agree(plain, received(shuffled, "pool", host, 16))


def transport_of(world, tmp_path, children=None, patch=None):
    out = copy_of(world, tmp_path, "transport", "separation")
    with shortened(children) as monkeypatch:
        if patch:
            patch(monkeypatch)
        return call(functools.partial(ctl.stage_transport, scale=SCALE), out)


def test_a_forbidden_map_request_that_does_not_raise_halts_the_transport(world, tmp_path):
    halt = transport_of(
        world, tmp_path, patch=lambda m: m.setattr(md, "_w_live_mutant", contextlib.nullcontext)
    )
    assert isinstance(halt, ps.Halt) and halt.check == "a red that does not fire"
    assert halt.detail["transport forbidden-map request"]["raised"] is False
    late = lambda spec: spec["out"].endswith("forbidden-request.json")  # noqa: E731
    assert (
        transport_of(world, tmp_path / "late", Children(timeouts=late)).check == "transport TIMEOUT"
    )


def test_a_tap_or_an_arm_that_times_out_halts_and_a_copy_bound_to_the_wrong_encoder_is_refused(
    world, tmp_path
):
    def tap(spec):
        return "tap-301" in spec["out"]

    halt = transport_of(world, tmp_path, Children(timeouts=tap))
    assert halt.check == "transport TIMEOUT" and halt.detail == {"tap": [301]}

    def arm(spec):
        return spec.get("arm") == "pool-act" and spec["out"].endswith("settled-302.json.gz")

    halt = transport_of(world, tmp_path / "arm", Children(timeouts=arm))
    assert halt.check == "transport" and len(halt.detail["not OK"]) == 1

    def tamper(spec, run):  # the copy ends with another encoder than the tap says it must
        if spec["arm"] == "pool-copy":
            run["pool"]["encoder"] += 1

    halt = transport_of(world, tmp_path / "enc", Children(tamper=tamper))
    assert halt.check == "R-b7" and "encoder" in halt.detail

    def swapped(spec, run):
        if spec["arm"] == "pool-act":
            run["pool"]["encoder"] = 5

    halt = transport_of(world, tmp_path / "act", Children(tamper=swapped))
    assert (
        halt.check == "transport" and "pool-act swapped an encoder" in halt.detail
    )  # its cell names no encoder


# ---- separation ----------------------------------------------------------------------------------


def test_separation_reads_the_xor_and_the_twin_through_arm_a_carried_and_settled(world):
    out, _ = world
    body = summary(out, "separation")
    assert (
        body["run"] is True
        and body["item_3"] == "RUN"
        and body["host"] == "kernel"
        and body["live_seeds"] == list(LIVE)
    )
    assert set(body["accuracy"]) == {"xor", "twin"} and set(body["accuracy"]["xor"]) == {
        "a",
        "carried",
        "settled",
    }
    # every structural check held on the shortened stream (told its length), or the stage had halted
    assert all(body["stream_checks"][str(seed)] is True for seed in LIVE)
    assert body["reading"]["stream_ok"] is True
    for name in ("xor", "twin"):
        for arm in ("a", "carried", "settled"):
            assert body["mean"][name][arm] == pytest.approx(
                sum(body["accuracy"][name][arm].values()) / 3, abs=1e-12
            )
            assert set(body["per_text"][name][arm]["300"]) == {"00", "01", "10", "11"}
        assert all(1 <= n <= 4 for n in body["distinct_settled_rows"][name].values())
    reading = body["reading"]
    scored = SEPARATION_EVENTS - 24  # the 24-event warm-up is not scored
    assert reading["scored"] == scored and reading["floor"] == 0.75
    assert reading["ceiling"] == pytest.approx(
        0.75 + 4 * math.sqrt(0.75 * 0.25 / scored), abs=1e-12
    )
    assert set(reading["forms"]) == {"carried", "settled"}
    assert (out / "separation" / "xor-300-stream.json.gz").is_file() and (
        out / "separation" / "twin-300-stream.json.gz"
    ).is_file()
    xor, twin = (
        md._load(out / "separation" / "xor-300-stream.json.gz"),
        md._load(out / "separation" / "twin-300-stream.json.gz"),
    )
    assert [e["x"] for e in xor["events"]] == [
        e["x"] for e in twin["events"]
    ]  # one set of x, two labellings
    assert xor["synthetic"]["labelling"] == "xor" and twin["synthetic"]["labelling"] == "twin"


def separation_of(world, tmp_path, build, children=None):
    out = copy_of(world, tmp_path, "separation")
    with shortened(children) as patch:
        shorten_separation(patch)
        return call(functools.partial(ctl.stage_separation, scale=SCALE, build=build), out), out


def test_a_failed_structural_stream_check_halts_before_any_replay(world, tmp_path):
    """P5 panel F6: one corrupted event label used to certify as valid (the checks were computed and
    never enforced). The control (clean streams) completes; the corrupted one halts naming the check
    and the seed, and no replay of any stream ran."""
    body, _ = separation_of(world, tmp_path / "clean", short_stream)
    assert isinstance(body, dict) and body["stream_checks"] == {str(seed): True for seed in LIVE}

    def corrupted(seed, labelling):
        stream = short_stream(seed, labelling)
        if seed == LIVE[0] and labelling is im.label_xor:
            event = stream["events"][40]
            event["c"] = 1 - event["c"]  # one corrupted event label
        return stream

    children = Children()
    halt, out = separation_of(world, tmp_path / "bad", corrupted, children)
    assert isinstance(halt, ps.Halt) and halt.check == "separation: stream checks"
    assert halt.detail == {str(LIVE[0]): ["c_is_the_previous_texts_xor_label"]}
    assert children.specs == []  # halted BEFORE any replay: no job ran
    assert (out / "separation" / f"xor-{LIVE[0]}-stream.json.gz").is_file()  # the streams are kept

    def wrong_length(seed, labelling):  # 60 events where the checks are told 120
        return im._build_stream(seed, labelling, n_events=60)

    halt, _ = separation_of(world, tmp_path / "length", wrong_length, Children())
    assert isinstance(halt, ps.Halt) and halt.check == "separation: stream checks"
    assert all("exactly_1200_events" in names for names in halt.detail.values())


def test_a_stream_that_reads_wrong_halts_and_a_pool_blind_to_the_code_stays_a_reading(
    world, tmp_path
):
    """The accuracy reading's `stream_ok` False (arm (a) above the affine ceiling on XOR, or under
    the
    floor on the twin) is the harness defect 2.11 names: it halts. 'pool blind to the code' is a
    reading, whatever a form reads."""
    real_read = ctl.read_separation

    def wrong(means, scored):
        return {**real_read(means, scored), "stream_ok": False, "stream_wrong": {"a_twin": True}}

    def blind(means, scored):
        reading = real_read(means, scored)
        for row in reading["forms"].values():
            row.update(readable=False, reading="pool blind to the code")
        return reading

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ctl, "read_separation", wrong)
        halt, _ = separation_of(world, tmp_path / "wrong", short_stream)
    assert isinstance(halt, ps.Halt) and halt.check == "separation: the stream is wrong"
    assert halt.detail == {"a_twin": True}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ctl, "read_separation", blind)
        body, _ = separation_of(world, tmp_path / "blind", short_stream)
    assert isinstance(body, dict) and body["reading"]["stream_ok"] is True
    assert {row["reading"] for row in body["reading"]["forms"].values()} == {
        "pool blind to the code"
    }


def test_every_contrasted_variant_keeps_its_per_segment_diagnostic_tables(world):
    """P5 panel F9: the controls, transport and separation summaries carry ps._report's table of
    every
    cell (both learners, per segment, with degeneracy and startup) for the report to keep."""
    out, _ = world
    controls, transport = summary(out, "controls"), summary(out, "transport")
    separation = summary(out, "separation")
    seeds = {str(seed) for seed in LIVE}

    def check(cells):
        assert set(cells) == set(ps.FIT_FORMS) and all(
            set(per_seed) == seeds for per_seed in cells.values()
        )
        for per_seed in cells.values():
            for cell in per_seed.values():
                assert cell["status"] == "OK" and {"readout", "grower", "degeneracy"} <= set(cell)
                row = cell["readout"]["unseen-order"]
                assert {"accuracy", "prediction_entropy", "coverage", "prior_fallback"} <= set(row)

    assert set(controls["cells"]["shuffle"]) == {str(k) for k in range(cs.SHUFFLE_K)}
    for cells in controls["cells"]["shuffle"].values():
        check(cells)
    check(controls["cells"]["window0"])
    assert controls["window0"]["cells"] == controls["cells"]["window0"]
    assert set(transport["arms"]) == set(md.TRANSPORT_ARMS)
    for arm in md.TRANSPORT_ARMS:
        check(transport["arms"][arm]["cells"])
    assert set(separation["cells"]) == {"xor", "twin"}
    for arms in separation["cells"].values():
        assert set(arms) == {"a", "carried", "settled"}
        assert all(set(per_seed) == seeds for per_seed in arms.values())
        assert all(
            cell["status"] == "OK" for per_seed in arms.values() for cell in per_seed.values()
        )


def test_separation_is_not_run_when_the_construction_check_did_not_read_true(world, tmp_path):
    out = copy_of(world, tmp_path, "separation")
    write_summary(
        out,
        "construct",
        run=True,
        verdict=False,
        item_3="NOT RUN",
        reason="a clause of the construction check failed (g1c 2.8)",
    )
    assert call(functools.partial(ctl.stage_separation, scale=SCALE), out) == {
        "run": False,
        "reason": "a clause of the construction check failed (g1c 2.8)",
        "item_3": "NOT RUN",
    }
    write_summary(out, "construct", run=False, verdict=None, item_3="NOT RUN", reason="TIMEOUT")
    assert call(functools.partial(ctl.stage_separation, scale=SCALE), out)["reason"] == "TIMEOUT"
    write_summary(out, "construct", run=False, verdict=None, item_3="NOT RUN", reason=None)
    assert call(functools.partial(ctl.stage_separation, scale=SCALE), out)["run"] is False


def test_the_reading_rules_of_2_8_on_hand_numbers():
    hand = {
        "xor": {"a": 0.50, "carried": 0.65, "settled": 0.99},
        "twin": {"a": 1.0, "carried": 0.85, "settled": 0.60},
    }
    reading = ctl.read_separation(hand, 944)
    assert reading["ceiling"] == pytest.approx(0.8064, abs=1e-4) and reading["stream_ok"]
    assert (
        reading["forms"]["carried"]["readable"]
        and reading["forms"]["carried"]["reading"] == "readable"
    )
    assert (
        not reading["forms"]["settled"]["readable"]
        and reading["forms"]["settled"]["reading"] == "pool blind to the code"
    )
    for patch, clause in (
        ({"xor": {**hand["xor"], "a": 0.81}}, "a_xor_at_or_above_ceiling"),
        ({"twin": {**hand["twin"], "a": 0.7499}}, "a_twin_below_floor"),
    ):
        wrong = ctl.read_separation({**hand, **patch}, 944)
        assert wrong["stream_wrong"][clause] and not wrong["stream_ok"]
    edge = ctl.read_separation({**hand, "twin": {**hand["twin"], "settled": 0.75}}, 944)
    assert edge["forms"]["settled"]["readable"]  # the floor is inclusive
    texts = ctl._per_text(im._build_stream(300, im.label_xor, n_events=40), [0] * 40)
    assert set(texts) == {"00", "01", "10", "11"}


def test_the_per_text_accuracy_groups_events_by_the_text_whose_phi_the_decoder_reads():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(im, "STREAM_WARM_UP", 8)
        stream = im._build_stream(301, im.label_xor, n_events=40)
        truth = [e["c"] for e in stream["events"]]
        perfect = ctl._per_text(stream, truth)
        assert {v for v in perfect.values() if v is not None} == {1.0}
        wrong = ctl._per_text(stream, [1 - c for c in truth])
        assert {v for v in wrong.values() if v is not None} == {0.0}
