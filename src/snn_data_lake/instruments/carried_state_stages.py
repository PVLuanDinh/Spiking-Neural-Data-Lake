r"""G1(c) / T109 part (c): the campaign driver's pool stages, part 2a -- causality, primary.

The frozen text is specs/022-unified-brain-arc/evidence/g1c-carried-state.md (2.10 R-b1, 2.3, 2.5, 2.9,
2.11, 2.13). `carried_state_driver` owns the stage list, the CLI and the first five stages; the stages here
run through the same `pool_selection.run_step` protocol (`function=` of `carried_state_driver.run_stage`),
so the single-head, dirty-tree, order and no-retry refusals and the exit codes are pool_selection's own.
This module also holds the helpers every pool stage shares (`carried_state_controls` and
`carried_state_report` import them): the cell key and accuracy readers of a pool table, `run_cells` (one
worker batch for several R-b7 tables), `use_of` (the host every stage's summary names) and `settings`
(the real window, b, R-b3 size and kick count, and the one place a test may shorten them).

The migrate stage (2.9's migration gate: the port against the KernelTick host) ran once, at 5761f3a7, and
chose the hosted form (g1c-carried-state.md 3.3); 2.9 then deleted the loser (the port) and the host
switch, and with them this stage, whose every reading compared the two. Its summary stays in that
campaign's OUT, and the report stage still reads it there (carried_state_report).

  causality   (2.10 R-b1) G1(a)'s R-a2 on the pool's Carried and Settled, both learners, three cuts, on
              seed 203; the (leak) red must change predictions at every cut; a failure halts.
  primary     (2.1, 2.3, 2.5) four forms x the live seeds; R-b4 twins (a second fresh-process replay of
              Carried and Settled on every seed, compared with pool_selection._twins_differ: a difference
              halts); the pool-noise red on seed 100.

A blind-harness reading is a READING, never a halt (the stage COMPLETES); a red that does not fire, a
missing or TIMEOUT cell, an R-b4 difference and R-b1 halt (2.11).

Names avoid the verbs the gradient-claim census reads. Deterministic except R-b4's unseeded kicks (a red).
"""

import tempfile
from pathlib import Path

from snn_data_lake.instruments import carried_state_campaign as cs
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.reservoir import recurrent_pool as rp

# Sites the duplicate-capability hook flagged on this module's docstring (read; none runs this campaign's stages):
# distinct-from: growth/h01_ingest.py::_selfcheck -- one ingest module's self-check of its own gates on synthetic fixtures
# distinct-from: gates/amdahl_gate.py::paired_ceiling -- a paired bootstrap band on a timing ceiling; no replay, no stage
# distinct-from: gates/doc_budget_gate.py::_selfcheck -- constructs the reds of one doc-budget gate's statuses
# distinct-from: gates/perf_baseline.py::_score -- scores a perf baseline's p50 and p95; no pool, no table
# distinct-from: instruments/pool_selection.py -- G1(b)'s ten steps; the stages here CALL its step protocol, jobs and R-b7 tables (run_step, run_jobs, _phase_table, _tabulate) and add G1(c)'s own stages, with no second job runner or table
POOL = ps.SELECTABLE[0]  # the one candidate: the pool G1(b) selected
LEARNERS = (
    "readout",
    "grower",
)  # D is read through the reference readout; the grower's is beside it
# The host every pool stage's summary names: the one pool implementation's (KernelLsmPool's policy
# "host"), the migration gate's recorded `use` (g1c 3.3)
HOST = "kernel"
REAL_SCALE = {  # what every stage runs at; only a test names another (disclosed in each summary)
    "window": rp.WINDOW,
    "b": md.TRANSPORT_B,
    "percepts": rp.R_B3_PERCEPTS,
    "kicks": 16,
}


# Sites the duplicate-capability hook flagged on the helpers below (read; none keys, reads or tabulates a pool replay):
# distinct-from: gates/selfcheck_cache.py::SelfcheckCache.why -- names the class of input that keys one module's selfcheck carry
# distinct-from: instruments/everyday_battery.py::Item -- one battery question with its frozen key and two constructed replies
# distinct-from: instruments/pool_selection.py::digest -- reduces ONE replay to its binding and scored segments; the readers below call it through the tables, never reimplement it
# distinct-from: instruments/report_order_probe.py::t081_constructed_read -- reads one constructed recording of another probe
def settings(scale=None):
    """The window, b, R-b3 percept count and kick count a stage runs at: REAL_SCALE, or REAL_SCALE with
    the keys of `scale` replaced (a test shortens them on a 48-event fixture and says so in the summary)."""
    unknown = sorted(set(scale or {}) - set(REAL_SCALE))
    if unknown:
        raise ValueError(f"unknown scale keys {unknown}")
    return {**REAL_SCALE, **(scale or {})}


def use_of(out):
    """The host a stage's summary names: HOST. An OUT that holds a migrate summary (the campaign at
    5761f3a7) must name the same host there; another `use` halts (no stage can replay the port)."""
    path = ps._summary_path(out, "migrate")
    if path.exists():
        use = md._load(path)["use"]
        if use != HOST:
            raise ps.Halt("migrate summary", f"use {use!r} is not the pool's host {HOST!r}")
    return HOST


# The seed bundle of the campaign in `out`: the record summary's `seed_set` names it once, and migrate,
# instrument and controls read their gate and R-b6 seeds from it (2.11)
def bundle_of(out):
    try:
        return cs.seed_bundle(ps._summary(out, "record")["seed_set"])
    except (KeyError, ValueError, OSError) as error:
        raise ps.Halt("seed bundle", f"the record summary names no seed set: {error!r}") from error


# ---- cells, tables and accuracies (one reader for every pool stage) --------------------------------


def key(form, seed, *, window, arm="pool", transport=None, encoder=None, noise=False):
    """The R-b7 table key of one pool cell of the TUNED configuration (pool_selection.cell_key)."""
    cell = {
        "arm": arm,
        "candidate": POOL,
        "config": cs.TUNED,
        "form": form,
        "seed": seed,
        "noise": noise,
        "window": window,
    }
    if transport is not None:
        cell["transport"] = transport
    if encoder is not None:
        cell["encoder"] = encoder
    return ps.cell_key(cell)


def need_ok(table, check):
    """2.10: a missing or TIMEOUT cell halts (never read over what completed)."""
    late = sorted(str(k) for k, cell in table.items() if cell["status"] != "OK")
    if late:
        raise ps.Halt(check, {"not OK": late})


def accuracy(table, cell_key, learner="readout"):
    """One learner's accuracy on a cell of an OK table (a halt on any other status)."""
    cell = table[cell_key]
    if cell["status"] != "OK":
        raise ps.Halt("a cell is not OK", str(cell_key))
    return ps.recording_accuracy(cell["digest"][learner])


def pool_table(out, folder, live, forms, paths, *, window, task=()):
    """pool_selection's phase table for the TUNED configuration over `forms` x `live` (+ `task` for
    Carried and Settled) at `window`: run, validated, bound into R-b7's table."""
    standing = {POOL: [(0, cs.TUNED)]}
    return ps._phase_table(
        out,
        folder,
        standing,
        tuple(live),
        tuple(task),
        tuple(forms),
        (),
        paths,
        window=window,
    )


def pool_cell(path, out_path, seed, form, *, window, stream_key=None):
    """One pool replay as a table cell; `stream_key` names a stream other than the seed's own recording
    (a shuffled one): the cell is then scored on it and its replay's source is held to it."""
    spec = ps._spec(POOL, cs.TUNED, form, window=window)
    cell = ps._cell(path, out_path, seed, spec=spec)
    if stream_key is not None:
        cell["stream_key"] = cell["source_key"] = stream_key
    return cell


def run_cells(groups):
    """ONE worker batch for every group's cells, then each group's R-b7 table (`groups` are
    (cells, phase_seeds, streams) triples; two groups may hold the same key, a table never does)."""
    todo = [cell for cells, _, _ in groups for cell in cells if cell["status"] is None]
    for cell, status in zip(todo, ps.run_jobs([cell["job"] for cell in todo]), strict=True):
        cell["status"] = status["status"]
    return [ps._tabulate(cells, seeds, streams) for cells, seeds, streams in groups]


def accuracies(table, seeds, forms, *, window, **named):
    """{learner: {form: {str(seed): accuracy}}} over an OK table of the TUNED configuration."""
    return {
        learner: {
            form: {
                str(seed): accuracy(table, key(form, seed, window=window, **named), learner)
                for seed in seeds
            }
            for form in forms
        }
        for learner in LEARNERS
    }


def gaps(acc, first="carried", second="settled"):
    """{learner: {str(seed): acc[first] - acc[second]}}: D (Carried - Settled) per seed."""
    return {
        learner: {
            seed: acc[learner][first][seed] - acc[learner][second][seed]
            for seed in acc[learner][first]
        }
        for learner in acc
    }


# The per-segment diagnostic tables 2.3 requires of EVERY contrasted variant: pool_selection's own `_report`
# of each cell (both learners' accuracy, entropy, coverage, prior-fallback rate, degeneracy, startup), as
# {form: {str(seed): report}} over an OK table whose cell keys differ only by form and seed
def variant_tables(table, seeds, forms, *, window, **named):
    return {
        form: {
            str(seed): ps._report(table[key(form, seed, window=window, **named)]) for seed in seeds
        }
        for form in forms
    }


# R-b4's fresh-process TWINS: a second replay of Carried and Settled on every seed of `seeds`, held
# against the table's own first one (pool_selection._twins_differ). A TIMEOUT, twins that ran in one
# process and a LEARNER difference halt here (a shared-instrument failure, 2.11); what a POOL difference
# means is the caller's (the primary halts on it).
# Sites the duplicate-capability hook flagged on twin_rows (read; neither compares two replays):
# distinct-from: sleep/structural_rewiring.py::StructuralRewiring -- the slow-clock structural pass over a substrate's rows, mutated in place; no replay
# distinct-from: substrate/episodic_recall.py::_RowStore -- an append-only float32 row buffer with capacity doubling; no replay
def twin_rows(folder, seeds, paths, table, window):
    jobs, bound = {}, {}
    for form in ps.FIT_FORMS:
        spec = ps._spec(POOL, cs.TUNED, form, window=window)
        for seed in seeds:
            bound[(form, seed)] = ps._bound(seed, spec)
            name = folder / f"twin-{form}-{seed}.json.gz"
            jobs[(form, seed)] = ps._replay_job(paths[seed], "pool", name, pool=spec)
    statuses = dict(zip(jobs, ps.run_jobs(list(jobs.values())), strict=True))
    late = sorted(f"{form} {seed}" for (form, seed), s in statuses.items() if s["status"] != "OK")
    if late:
        raise ps.Halt("R-b4 TIMEOUT", late)
    rows = {}
    for (form, seed), job in jobs.items():
        stream = md._load(paths[seed])
        first = md._load(table[key(form, seed, window=window)]["job"]["out"])
        second = ps._checked(job, stream, cell=bound[(form, seed)])
        pool, learner = ps._twins_differ(first, second)
        apart = first["execution_id"].split("-")[0] != second["execution_id"].split("-")[0]
        rows[f"{form} {seed}"] = {
            "pool_differs": pool,
            "learner_differs": learner,
            "executions": apart,
        }
    if not all(row["executions"] for row in rows.values()):
        raise ps.Halt("R-b4", "the twins ran in one process")
    if any(row["learner_differs"] for row in rows.values()):
        raise ps.Halt("R-b4", rows)
    return rows


# ---- causality (2.10 R-b1) -----------------------------------------------------------------------


# Sites the duplicate-capability hook flagged on stage_causality (read; neither builds R-b1's replay jobs):
# distinct-from: instruments/pool_selection.py::step_check203 -- G1(b)'s sibling wrapper (the tuning summary's candidates on the port, with R-b4's twins); both CALL pool_selection.RB1Check, the one construction of R-b1's jobs, cuts and prefix reads, so this is no second mechanism
# distinct-from: faculties/theory_of_mind.py::_run_sally_anne -- stages the Sally-Anne false-belief task for a TheoryOfMind; no replay, no cut
def stage_causality(out, *, scale=None):
    """R-b1 on seed 203: G1(a)'s R-a2 on the pool's Carried and Settled at three cuts, both
    learners; the (leak) arm must change predictions at every cut. A failure halts (2.10, 2.11). The
    jobs, the cuts and the prediction-prefix reads are `pool_selection.RB1Check`'s."""
    sc = settings(scale)
    window, seed, host = sc["window"], cs.CAUSALITY_SEED, use_of(out)
    if ps._recorded(out).get(seed) != "OK":
        raise ps.Halt("R-b1", f"seed {seed}'s recording is missing")
    folder = ps._folder(out, "causality")
    path = ps._recording(out, seed)
    stream = md._load(path)
    pools = {  # explicit specs: the pool's Carried and Settled
        form: (form, ps._spec(POOL, cs.TUNED, form, window=window)) for form in ps.FIT_FORMS
    }
    check = ps.RB1Check(folder, seed, path, stream, pools)
    cuts = check.cuts
    statuses = dict(zip(check.jobs, ps.run_jobs(list(check.jobs.values())), strict=True))
    late = sorted(name for name, status in statuses.items() if status["status"] != "OK")
    if late:
        raise ps.Halt("R-b1 TIMEOUT", late)
    red = check.red()
    held = {}
    for form in ps.FIT_FORMS:
        for cut, (grower, readout) in check.held(form).items():
            held[f"{form} {cut}"] = {"grower": grower, "readout": readout}
    if not all(red.values()):
        raise ps.Halt("a red that does not fire", {"R-b1 leak": red})
    if not all(all(row.values()) for row in held.values()):
        raise ps.Halt("R-b1", held)
    return {
        "host": host,
        "window": window,
        "scale": sc,
        "seed": seed,
        "cuts": list(cuts),
        "R-b1": held,
        "R-b1 red": red,
    }


# ---- primary (2.1, 2.3, 2.5) ---------------------------------------------------------------------


def _noise_red(out, folder, window):
    """R-b4's pool-noise red on seed 100: a replay with unseeded kicks must move the state chain and phi
    of an otherwise identical plain Carried replay (pool_selection.noise_red_fires)."""
    seed = cs.IDENTITY_SEED
    path = ps._recording(out, seed)
    stream = md._load(path)
    jobs, bound = {}, {}
    for name, noise in (("plain", False), ("noise", True)):
        spec = ps._spec(POOL, cs.TUNED, "carried", noise=noise, window=window)
        bound[name] = ps._bound(seed, spec)
        jobs[name] = ps._replay_job(path, "pool", folder / f"noise-red-{name}.json.gz", pool=spec)
    statuses = dict(zip(jobs, ps.run_jobs(list(jobs.values())), strict=True))
    if any(status["status"] != "OK" for status in statuses.values()):
        raise ps.Halt("R-b4 TIMEOUT", {name: s["status"] for name, s in statuses.items()})
    plain, noisy = (ps._checked(jobs[name], stream, cell=bound[name]) for name in jobs)
    fired = bool(ps.noise_red_fires(plain, noisy))
    if not fired:
        raise ps.Halt("a red that does not fire", {"R-b4 pool noise (seed 100)": False})
    return {"seed": seed, "fired": fired}


def stage_primary(out, *, scale=None):
    """Four forms x the live seeds, R-b4 twins on every seed, the pool-noise red
    on seed 100; per seed D through both learners, and the per-seed, per-form, per-segment tables 2.3
    requires (`ps._report` of every cell: both learners' accuracy, entropy, coverage, fallback rate,
    degeneracy, startup), with the input-response statistics of each recording."""
    sc = settings(scale)
    window, host = sc["window"], use_of(out)
    live = tuple(ps._summary(out, "record")["live_seeds"])
    folder = ps._folder(out, "primary")
    paths = {seed: ps._recording(out, seed) for seed in live}
    table = pool_table(out, folder, live, rp.FORMS, paths, window=window)
    need_ok(table, "primary")
    twins = twin_rows(folder, live, paths, table, window)
    if any(row["pool_differs"] for row in twins.values()):  # the primary halts on a pool difference
        raise ps.Halt("R-b4", twins)
    noise = _noise_red(out, folder, window)
    streams = ps._streams(paths)
    acc = accuracies(table, live, rp.FORMS, window=window)
    return {
        "host": host,
        "window": window,
        "scale": sc,
        "shortened": sc != REAL_SCALE,
        "live_seeds": list(live),
        "accuracy": acc,
        "d": gaps(acc),
        "cells": {
            form: {str(seed): ps._report(table[key(form, seed, window=window)]) for seed in live}
            for form in rp.FORMS
        },
        "input_response": {
            str(seed): ps.input_response(POOL, cs.TUNED, streams[seed]) for seed in live
        },
        "R-b4": twins,
        "R-b4 red": noise,
    }


STAGE_FUNCTIONS = {
    "causality": stage_causality,
    "primary": stage_primary,
}


# Sites the duplicate-capability hook flagged on the self-check (read; none checks a pool stage):
# distinct-from: gates/selfcheck_cache.py::SelfcheckCache.describe -- the composition of one module's selfcheck key
# distinct-from: instruments/memory_diagnostic.py::_synthetic_stream -- the 48-event hand stream this self-check CALLS
# distinct-from: instruments/report_order_probe.py::t081_constructed_read -- reads one constructed recording of another probe
# distinct-from: brainstem/serial_arm.py -- the real-arm actuator sink's module docstring (read only; no brainstem file is touched)
# distinct-from: engine/brain.py::Brain.propose_improvements -- the brain's proposer of candidate self-improvements
def self_check():
    """cwd-free, no recording, no Brain, no live seed: the scale, the cell key, the readers and the
    gap arithmetic on hand numbers, and `use_of` over an OUT with no migrate summary, with the one the
    campaign at 5761f3a7 wrote (`use` "kernel") and with one naming the deleted port (a halt)."""
    assert tuple(STAGE_FUNCTIONS) == ("causality", "primary")
    assert settings() == REAL_SCALE and settings({"window": 16})["window"] == 16
    try:
        settings({"nope": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown scale key was accepted")
    # the key is pool_selection's: a window other than the default is part of it
    assert key("carried", 7, window=0) != key("carried", 7, window=rp.WINDOW)
    hand = {"readout": {"carried": {"1": 0.7}, "settled": {"1": 0.5}}}
    assert gaps(hand) == {"readout": {"1": 0.7 - 0.5}}
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        assert use_of(out) == HOST
        md._save(ps._summary_path(out, "migrate"), {"use": "kernel"})
        assert use_of(out) == HOST == rp.KernelLsmPool.lineage_policy(cs.TUNED)["host"]
        md._save(ps._summary_path(out, "migrate"), {"use": "port"})
        try:
            use_of(out)
        except ps.Halt:
            pass
        else:
            raise AssertionError("a migrate summary naming the deleted port was read")
    print(
        "carried_state_stages self-check OK: scale, cell keys, gaps, use_of (no migrate summary, the "
        "recorded hosted use, and a port use refused)"
    )


if __name__ == "__main__":
    self_check()
