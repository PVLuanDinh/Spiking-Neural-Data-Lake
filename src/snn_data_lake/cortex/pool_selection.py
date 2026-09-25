"""G1(b) / T109 part (b): the pool-selection campaign.

The frozen text is specs/022-unified-brain-arc/evidence/g1b-pool-selection.md section 2 (commit
23c7e97). Ten steps over one OUT directory, each a CLI subcommand. A step refuses to start unless the
previous step's summary exists, COMPLETED, and was written at the same clean head (memory_diagnostic.
_revision), and it writes OUT/<step>.summary.json with `head` and `dirty`:

  record      seeds 100-103 and 200-207 through memory_diagnostic.record_stream (the pilot's workers)
  mechanism   R-b0x's readout fixtures; per candidate R-b0, R-b3's red, R-b5 (i)-(iii) with the
              reds of (i) and (ii), R-b4's pool-noise red (the pool module's checks, full window)
  instrument  R-b2, R-b7's reds, R-b6 on 4242 with its history-free red, R-b8's two halves with reds
  screen      recurrent_pool.cost on seed 200 through recurrent_pool.screen, then the manifest
  threads     R-b3's equality, with R-b5 (i)-(iii), on every manifest configuration
  sigma       sigma_dev on 100-103 (section 2.6), sigma_plan and S
  tune        candidates x configurations x {carried, settled} on 200-203 and 5001-5004, (e1-0)
  check203    R-b1 at the three cuts and the per-candidate R-b4 twins on seed 203
  select      the tuned candidates x the four forms on 204-207, 5005-5008, (e1-0) and (e1)
  winner      the winner's Carried and Settled R-b4 twins on 204-207

Section 2.5 (the statistic, eligibility, the ordering), section 2.6's power rule and section 2.8's
taxonomy are the pure functions below. Every job runs in its own child process under section 2.10's
limits: 4 workers, a 900 s timeout, torch and BLAS at one thread (R-b3 sets its own). Exit codes:
0 COMPLETED (candidate refusals and TIMEOUTs recorded in the summary), 1 ABORTED (a job FAILed: a
defect), 2 HALTED (the shared instrument failed; the summary names the check), 3 refused to start
(no summary written). OUT must lie outside the repository: a file written inside it dirties the
head, and the step then ends ABORTED.

One module on purpose, past the 800-line soft ceiling: the build brief lands the campaign as this
one new file beside the harness (memory_diagnostic.py) and the pools (recurrent_pool.py), and its
size is the size of section 2's ten steps and their checks.
"""

import argparse
import contextlib
import functools
import io
import math
import statistics
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from snn_data_lake.cortex import memory_diagnostic as md
from snn_data_lake.cortex import recurrent_pool as rp

STEPS = (
    "record",
    "mechanism",
    "instrument",
    "screen",
    "threads",
    "sigma",
    "tune",
    "check203",
    "select",
    "winner",
)
# Section 2.10's seeds.
DEV_SEEDS = (100, 101, 102, 103)
TUNE_SEEDS = (200, 201, 202, 203)
SELECT_SEEDS = (204, 205, 206, 207)
TUNE_TASK_SEEDS = (5001, 5002, 5003, 5004)
SELECT_TASK_SEEDS = (5005, 5006, 5007, 5008)
RECORD_SEEDS = DEV_SEEDS + TUNE_SEEDS + SELECT_SEEDS
RB6_SEED = 4242
SCREEN_SEED = 200
CHECK_SEED = 203
PILOT_SEED = 100  # R-b8's pre-build half: G1(a)'s pilot3 recording of seed 100 at 829fb40
WORKERS = 4
JOB_TIMEOUT_S = 900
# Section 2.5.
BAND_D_FLOOR = 0.005
TASK_BAND = 0.005
COST_TIE = 0.10
NON_INFERIORITY = 0.05
DEGENERATE_FRACTION = 0.9
# Section 2.6.
SIGMA_PROXY = 0.0436
DELTA = 0.02
POWER_FLOOR = 8
POWER_CAP = 64
EXIT_OK, EXIT_ABORT, EXIT_HALT, EXIT_REFUSED = 0, 1, 2, 3
FIT_FORMS = ("carried", "settled")  # the forms D, D_task and eligibility read
# Section 2.1: the candidates the selection may name. ONE constant, so that a dated pre-data
# amendment to the eligible winners edits one line. It must stay a subset of
# recurrent_pool.CANDIDATES. The dated amendment of 2026-09-25 (pre-data, evidence section 2.1: the
# owner's engine names its reservoir a liquid state machine, and the owner chose "Spiking only can
# win") set it to ("lsm", "esn"), with g3 (SORN) and g4 (RBN) run as reference points. That campaign
# is held by the record: campaign 2 ran at f27ae70 and 0bd1de7 records its selection (winner g1
# LSM). Section 2.9 then deleted the losing ports, so g1 is the one candidate left to name.
SELECTABLE = ("lsm",)


class Halt(Exception):
    """Section 2.8: the shared instrument failed (a readout fixture, a red that does not fire, R-b1,
    R-b2, R-b6, R-b7, R-b8, an R-b4 difference in a learner's state). Nothing is selected."""

    def __init__(self, check, detail=None):
        super().__init__(check)
        self.check = check
        self.detail = detail


class Abort(Exception):
    """Section 2.8: a job FAILed. A defect: the campaign aborts and re-runs at the fixed head."""

    def __init__(self, jobs):
        super().__init__(f"{len(jobs)} job(s) failed")
        self.jobs = jobs


class Refused(Exception):
    """The step-order refusal: the step never starts and writes no summary."""


def _plain(value):
    """JSON-safe: numpy scalars and arrays, tuples, paths and non-string keys made plain."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


# ---- section 2.5: the statistic, eligibility and the ordering (pure) -----------------------------


def recording_accuracy(scores):
    """One learner's accuracy on one stream: the mean over its available scored segments (a None
    prediction is a miss). A live recording with none is a harness defect and halts (section 2.5);
    on a synthetic stream this is the accuracy over its 944 unseen-order events."""
    values = [row["accuracy"] for row in scores.values() if row.get("available")]
    if not values:
        raise Halt("scored segments", "a stream with no available scored segment (section 2.5)")
    return statistics.fmean(values)


def band_d(sigma_dev):
    """band_D = max(0.005, sigma_dev): 2 sigma_dev / sqrt(4) of a four-recording mean."""
    return max(BAND_D_FLOOR, float(sigma_dev))


def degenerate(stats):
    """Section 2.5: at least 90% of units constant over the scored rows, or fewer than two distinct
    phi rows (no scored rows at all counts as degenerate)."""
    return bool(
        stats is None
        or not stats["rows"]
        or stats["constant_fraction"] >= DEGENERATE_FRACTION
        or stats["distinct_rows"] < 2
    )


def eligibility(entry, reference_mean):
    """Section 2.5's eligibility of a configuration (tuning) or a tuned candidate (selection): every
    job it needs completed OK; its mean Carried accuracy through the readout on the phase's LIVE
    recordings at least (e1-0)'s minus 0.05; its Carried phi degenerate on none of them. Front-end
    never enters. Returns (eligible, reasons)."""
    if not entry["ok"]:
        return False, ["a job it needs did not complete OK"]
    reasons = []
    if reference_mean is None:
        reasons.append("no (e1-0) reference on every live recording")
    elif statistics.fmean(entry["carried_live"]) < reference_mean - NON_INFERIORITY:
        reasons.append("inferior to (e1-0) by more than 0.05")
    if any(entry["degenerate"]):
        reasons.append("Carried phi degenerate on a live recording")
    return not reasons, reasons


def _pick(entries, band):
    best = max(entry["D"] for entry in entries)
    tier = [entry for entry in entries if best - entry["D"] <= band]
    best = max(entry["D_task"] for entry in tier)
    tier = [entry for entry in tier if best - entry["D_task"] <= TASK_BAND]
    low = min(entry["cost"] for entry in tier)
    tier = [entry for entry in tier if entry["cost"] - low <= COST_TIE * low]
    return min(tier, key=lambda entry: entry["order"])


def ordering(entries, band):
    """Section 2.5's ordering, applied until every entry is placed; its head is the winner. (1) D
    within `band` of the best D is near-tied; (2) among those, D_task within 0.005 of the best is
    still tied; (3) among those, the lower cost wins, costs within 10% of the lowest tied; (4) the
    lowest `order` (manifest position in tuning, candidate index in selection). Descriptive only."""
    rest, ranked = list(entries), []
    while rest:
        head = _pick(rest, band)
        ranked.append(head)
        rest = [entry for entry in rest if entry is not head]
    return ranked


# distinct-from: cortex/access_consciousness.py -- `focused()` reads whether a workspace content is the
#   arbiter's head; this filters benchmark entries to the selectable set before section 2.5's ordering.
def rank_selection(entries, band):
    """(ranked, reference): the ordering over the eligible SELECTABLE entries, whose head is the
    winner, and the same ordering over every eligible entry, reported so a reference candidate that
    would have led is visible (dated amendment 2026-09-25). A reference never becomes the winner."""
    eligible = [e for e in entries if e["eligible"]]
    ranked = ordering([e for e in eligible if e["candidate"] in SELECTABLE], band)
    return ranked, ordering(eligible, band)


# ---- section 2.6: G1(c)'s power rule -------------------------------------------------------------


# distinct-from: substrate/paired_contrast.py -- `paired_t` and its fixed T_BAR (7 d.f.) READ a paired
#   t; section 2.6's rule needs t quantiles at 0.95 and 0.80 for every d.f. from 7 to 63. scipy is
#   installed but undeclared, and pyproject is outside this build's files: the quantiles below are
#   exact for an integer d.f. (Abramowitz and Stegun 26.7.3-26.7.4) and pinned by 2.6's own table.
def _t_cdf(t, df):
    """Student's t CDF at an integer df >= 1."""
    if t < 0:
        return 1.0 - _t_cdf(-t, df)
    theta = math.atan(t / math.sqrt(df))
    sin, cos = math.sin(theta), math.cos(theta)
    if df % 2:
        if df == 1:
            inside = 2.0 * theta / math.pi
        else:
            term = total = cos
            for k in range(3, df - 1, 2):
                term *= (k - 1) / k * cos * cos
                total += term
            inside = 2.0 / math.pi * (theta + sin * total)
    else:
        term = total = 1.0
        for k in range(2, df - 1, 2):
            term *= (k - 1) / k * cos * cos
            total += term
        inside = sin * total
    return 0.5 + 0.5 * inside


@functools.cache
def t_quantile(p, df):
    """The p-quantile (0.5 < p < 1) of Student's t at an integer df, by bisection on _t_cdf."""
    low, high = 0.0, 64.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if _t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def power_s(sigma, delta=DELTA):
    """Section 2.6: the smallest S >= 8 with (t_{0.95,S-1} + t_{0.80,S-1}) sigma / delta <= sqrt(S),
    capped at 64; `capped` discloses a binding cap (reduced power)."""
    for s in range(POWER_FLOOR, POWER_CAP + 1):
        if (t_quantile(0.95, s - 1) + t_quantile(0.80, s - 1)) * sigma / delta <= math.sqrt(s):
            return {"S": s, "capped": False}
    return {"S": POWER_CAP, "capped": True}


# ---- R-b7: the binding and the table (section 2.7) -----------------------------------------------


@functools.cache
def _w_in_crc(seed):
    return md._crc(rp.w_in(seed))


def cell_key(cell):
    config = (
        None
        if cell["config"] is None
        else rp.crc_json(rp.canonical(cell["candidate"], cell["config"]))
    )
    return (
        cell["arm"],
        cell["candidate"],
        config,
        cell["form"],
        cell["seed"],
        bool(cell.get("noise")),
    )


def binding_problems(cell):
    """R-b7: the fields in which a replay's digest disagrees with the cell it is filed under."""
    digest = cell["digest"]
    problems = [key for key in ("arm", "seed") if digest.get(key) != cell[key]]
    binding = digest.get("pool")
    if cell["arm"] != "pool":
        return problems + (["pool"] if binding is not None else [])
    if binding is None:
        return [*problems, "pool"]
    config = rp.canonical(cell["candidate"], cell["config"])
    expected = {
        "candidate": cell["candidate"],
        "form": cell["form"],
        "config_crc32": rp.crc_json(config),
        "realization_seed": rp.realization_seed(cell["candidate"], cell["seed"]),
        "window": rp.WINDOW,
        "noise": bool(cell.get("noise")),
        "w_in": _w_in_crc(cell["seed"]),
    }
    return problems + [key for key, value in expected.items() if binding.get(key) != value]


def build_table(cells, phase_seeds):
    """R-b7: coverage unique and complete. Each OK replay is bound to the cell it is filed under;
    one result per (arm, candidate, configuration, form, seed); only the phase's own seeds (so a
    tuning-seed replay is refused in the selection table). Any refusal HALTS (section 2.8)."""
    allowed = frozenset(phase_seeds)
    table = {}
    for cell in cells:
        key = cell_key(cell)
        if cell["seed"] not in allowed:
            raise Halt("R-b7", f"REFUSED: seed {cell['seed']} is not a seed of this phase")
        if key in table:
            raise Halt("R-b7", f"REFUSED: duplicate seed {cell['seed']} for {key[:4]}")
        if cell["status"] == "OK":
            problems = binding_problems(cell)
            if problems:
                raise Halt(
                    "R-b7", f"REFUSED: a replay filed under {key} is bound elsewhere: {problems}"
                )
        table[key] = cell
    return table


def _refused_by_r_b7(build):
    try:
        build()
    except Halt as halt:
        return halt.check == "R-b7"
    return False


def r_b7_reds():
    """R-b7's three constructed refusals (a wrong configuration, a duplicate seed, a tuning seed in
    the selection table), beside the well-formed table they are cut from, which must pass."""
    config, other = rp.grid("lsm")[0], rp.grid("lsm")[1]

    def cell(seed, bound=config):
        binding = {
            "candidate": "lsm",
            "form": "carried",
            "config_crc32": rp.crc_json(bound),
            "realization_seed": rp.realization_seed("lsm", seed),
            "window": rp.WINDOW,
            "noise": False,
            "w_in": _w_in_crc(seed),
        }
        return {
            "arm": "pool",
            "candidate": "lsm",
            "index": 0,
            "config": config,
            "form": "carried",
            "seed": seed,
            "noise": False,
            "status": "OK",
            "digest": {"arm": "pool", "seed": seed, "pool": binding},
        }

    control = len(build_table([cell(seed) for seed in SELECT_SEEDS], SELECT_SEEDS)) == 4
    red = {
        "wrong configuration": _refused_by_r_b7(
            lambda: build_table([cell(204, bound=other)], SELECT_SEEDS)
        ),
        "duplicate seed": _refused_by_r_b7(
            lambda: build_table([cell(204), cell(204)], SELECT_SEEDS)
        ),
        "tuning seed in selection": _refused_by_r_b7(
            lambda: build_table([cell(200)], SELECT_SEEDS)
        ),
    }
    return {"verdict": bool(control and all(red.values())), "control": control, "red": red}


# ---- replays as table cells ----------------------------------------------------------------------


def _scores(stream, predictions, fallbacks):
    scored = md.score(stream, predictions, fallbacks, floors=False)
    return {segment: row for segment, row in scored.items() if row.get("available")}


def digest(run, stream):
    """A replay reduced to what the tables read: its binding and both learners' scored segments."""
    readout = run.get("readout")
    count = len(stream["events"])
    return {
        "arm": run["arm"],
        "seed": run.get("seed"),
        "pool": run.get("pool"),
        "grower": _scores(stream, run["predictions"], run["fallbacks"]),
        "readout": None
        if readout is None
        else _scores(stream, readout["predictions"], [False] * count),
        "seconds": run.get("seconds"),
    }


def _replay_job(stream, arm, out, *, pool=None, readout=True, fault=None):
    return {
        "kind": "replay",
        "stream": str(stream),
        "arm": arm,
        "pool": pool,
        "readout": readout,
        "fault": fault,
        "out": str(out),
    }


# The cell helpers below file replays for R-b7 (sites read, none files a replay under a binding):
# distinct-from: cortex/memory_diagnostic.py -- replay() RUNS the pool arm from a spec; these name one
# distinct-from: cortex/time_cells.py -- tick advances a ramp and latches crossed cells
# distinct-from: neuromodulation/drive_reduction.py -- NeedPool is a drive's LIF population
# distinct-from: reasoning/rpm_solver.py -- _induce_attr predicts a missing matrix cell
# distinct-from: thalamus/binding_instrument.py -- _t080_sparse_latent draws a sparse latent
def _spec(candidate, config, form, noise=False):
    """A pool arm's spec for memory_diagnostic.replay (R-b4's red when `noise`)."""
    spec = {"candidate": candidate, "config": config, "form": form}
    return dict(spec, noise=True) if noise else spec


def _bound(seed, spec):
    """What one pool replay must be bound to (R-b7), for binding_problems."""
    return {
        "arm": "pool",
        "seed": seed,
        "candidate": spec["candidate"],
        "config": spec["config"],
        "form": spec["form"],
        "noise": bool(spec.get("noise")),
    }


def _cell(stream, out, seed, *, arm="pool", index=None, spec=None, missing=False):
    """One table cell: a replay job and what it is filed under. A missing recording makes its cells
    ineligible without running them (section 2.8: a TIMEOUT, inherited from the recording's)."""
    filed = (
        _bound(seed, spec)
        if arm == "pool"
        else {
            "arm": arm,
            "seed": seed,
            "candidate": None,
            "config": None,
            "form": None,
            "noise": False,
        }
    )
    return {
        **filed,
        "index": index,
        "status": "TIMEOUT" if missing else None,
        "job": _replay_job(stream, arm, out, pool=spec),
        "digest": None,
    }


def _checked(job, stream, source=None, *, cell=None):
    """Load a replay, holding it to G1(a)'s provenance and, for a cell, to R-b7's binding."""
    run = md._load(job["out"])
    try:
        md._validate_run(
            run, stream, source if source is not None else stream, job["arm"], "causal"
        )
    except ValueError as error:
        raise Halt("R-b7", str(error)) from error
    if cell is not None:
        problems = binding_problems(
            dict(cell, digest={"arm": run["arm"], "seed": run.get("seed"), "pool": run.get("pool")})
        )
        if problems:
            raise Halt("R-b7", f"REFUSED: {job['out']} is bound elsewhere: {problems}")
    return run


def _tabulate(cells, phase_seeds, streams, runner=None):
    """Run every cell's replay, then load, validate and digest each OK result into R-b7's table."""
    todo = [cell for cell in cells if cell["status"] is None]
    for cell, status in zip(todo, run_jobs([cell["job"] for cell in todo], runner), strict=True):
        cell["status"] = status["status"]
    for cell in cells:
        if cell["status"] == "OK":
            stream = streams[cell["seed"]]
            cell["digest"] = digest(_checked(cell["job"], stream), stream)
    return build_table(cells, phase_seeds)


def _report(cell):
    if cell["status"] != "OK":
        return {"status": cell["status"]}
    found = cell["digest"]
    binding = found["pool"] or {}
    return {
        "status": "OK",
        "readout": found["readout"],
        "grower": found["grower"],
        "seconds": found["seconds"],
        **{
            key: binding.get(key)
            for key in ("degeneracy", "startup", "movement", "pool_ms", "frozen")
            if binding
        },
    }


def _entry(
    table, candidate, index, config, cost, live, task, *, order, forms=FIT_FORMS, blocked=False
):
    """One configuration's (tuning) or tuned candidate's (selection) row: D and D_task through the
    readout, the grower's beside them, and every reported quantity of its cells."""
    crc = rp.crc_json(rp.canonical(candidate, config))

    def cell(form, seed):
        return table[("pool", candidate, crc, form, seed, False)]

    needed = [cell(form, seed) for form in FIT_FORMS for seed in live + task]
    entry = {
        "candidate": candidate,
        "index": index,
        "config": config,
        "cost": cost,
        "order": order,
        "ok": not blocked and all(c["status"] == "OK" for c in needed),
        "cells": {
            form: {
                str(seed): _report(cell(form, seed))
                for seed in (live + task if form in FIT_FORMS else live)
            }
            for form in forms
        },
    }
    if not entry["ok"]:
        return entry
    for learner in ("readout", "grower"):
        live_acc = {
            form: [recording_accuracy(cell(form, seed)["digest"][learner]) for seed in live]
            for form in FIT_FORMS
        }
        task_acc = {
            form: [recording_accuracy(cell(form, seed)["digest"][learner]) for seed in task]
            for form in FIT_FORMS
        }
        d = [c - s for c, s in zip(live_acc["carried"], live_acc["settled"], strict=True)]
        d_task = [c - s for c, s in zip(task_acc["carried"], task_acc["settled"], strict=True)]
        row = {
            "D": statistics.fmean(d),
            "D_task": statistics.fmean(d_task),
            "per_seed_D": d,
            "per_seed_D_task": d_task,
        }
        if learner == "readout":
            entry.update(row, carried_live=live_acc["carried"])
        else:
            entry["grower"] = row
    entry["degenerate"] = [
        degenerate(cell("carried", seed)["digest"]["pool"]["degeneracy"]) for seed in live
    ]
    return entry


def _reference(table, arm, seeds):
    """A reference arm's accuracies through both learners per live recording, and the readout's mean
    (None unless every replay completed OK)."""
    cells = [table[(arm, None, None, None, seed, False)] for seed in seeds]
    per_seed = {str(c["seed"]): _report(c) for c in cells}
    if any(c["status"] != "OK" for c in cells):
        return {"per_seed": per_seed, "mean": None}
    return {
        "per_seed": per_seed,
        "mean": statistics.fmean(recording_accuracy(c["digest"]["readout"]) for c in cells),
        "grower_mean": statistics.fmean(recording_accuracy(c["digest"]["grower"]) for c in cells),
    }


# distinct-from: cortex/recurrent_pool.py -- `drive` is the encoder itself and `_percept` builds fixture
#   percepts; this READS the drive each candidate's input rule applies, over a recording (reported)
def input_response(candidate, config, stream):
    """Section 2.5's reported input-response statistics on one recording: the driven-unit mask, and
    the mean and sd of the per-unit drive over the driven units and the recording's percepts (g1:
    the signed drive g u, on every unit)."""
    config = rp.canonical(candidate, config)
    encoder = rp.w_in(stream["seed"])
    mask = np.ones(rp.N, dtype=bool)
    rows = []
    for event in stream["events"]:
        u = rp.drive(encoder, event["x"])
        rows.append(config["g"] * u)
    driven = np.asarray(rows)[:, mask]
    return {
        "driven": int(np.count_nonzero(mask)),
        "mask": np.flatnonzero(mask).tolist(),
        "mean": float(driven.mean()) if driven.size else None,
        "sd": float(driven.std()) if driven.size else None,
    }


# ---- jobs: one child process each (section 2.10) -------------------------------------------------


def mechanism_checks(candidate):
    """Section 2.7's mechanism checks for one candidate at the full window on the development
    fixtures, each green beside its red: R-b0; R-b3's order red on the declared witness; R-b5 (i)
    after Reset and Settled with the skip-a-carrier red; R-b5 (iii) (green only: its one red was
    SORN's plasticity left on, deleted with g3 in section 2.9); R-b5 (ii) and its reset-every-percept
    red on g1's fixture; R-b4's pool-noise red."""
    config, _ = rp.R_B5_ERASURE_WITNESS[candidate]
    source = {"lsm": rp.check_r_b0_lsm}
    numbers, greens, reds = {}, {}, {}
    numbers["R-b0"] = source[candidate]()
    greens["R-b0"], reds["R-b0"] = numbers["R-b0"]["verdict"], numbers["R-b0"]["red"]
    numbers["R-b3 red"] = rp.check_r_b3_red(candidate)
    reds["R-b3"] = numbers["R-b3 red"]["verdict"]
    for form in ("reset", "settled"):
        numbers[f"R-b5 erasure {form}"] = rp.check_r_b5_erasure(candidate, config, form)
        numbers[f"R-b5 erasure {form} red"] = rp.check_r_b5_erasure_red(candidate, form)
        greens[f"R-b5 erasure {form}"] = numbers[f"R-b5 erasure {form}"]["verdict"]
        reds[f"R-b5 erasure {form}"] = numbers[f"R-b5 erasure {form} red"]["verdict"]
    numbers["R-b5 freeze"] = rp.check_r_b5_freeze(candidate, config)
    greens["R-b5 freeze"] = numbers["R-b5 freeze"]["verdict"]
    numbers["R-b4 pool noise"] = rp.pool_noise_red(candidate, config)
    reds["R-b4 pool noise"] = numbers["R-b4 pool noise"]["verdict"]
    if candidate == "lsm":
        numbers["R-b5 persistence"] = rp.check_r_b5_persistence()
        numbers["R-b5 persistence red"] = rp.check_r_b5_persistence(carried_form="reset")
        greens["R-b5 persistence"] = numbers["R-b5 persistence"]["verdict"]
        reds["R-b5 persistence"] = not numbers["R-b5 persistence red"]["verdict"]
    return {"candidate": candidate, "greens": greens, "reds": reds, "numbers": numbers}


def configuration_checks(candidate, config):
    """One manifest configuration before any tuning (section 2.7): R-b3's equality at BLAS 1 and 24
    and torch 1 and 4, R-b5 (i) after Reset and Settled, (iii), and its own persistence ((ii),
    reported only)."""
    return {
        "R-b3": rp.check_r_b3(candidate, config),
        "R-b5 erasure reset": rp.check_r_b5_erasure(candidate, config, "reset"),
        "R-b5 erasure settled": rp.check_r_b5_erasure(candidate, config, "settled"),
        "R-b5 freeze": rp.check_r_b5_freeze(candidate, config),
        "R-b5 persistence": rp.check_r_b5_persistence(candidate, config),
    }


CONFIGURATION_GREENS = ("R-b3", "R-b5 erasure reset", "R-b5 erasure settled", "R-b5 freeze")
CONFIGURATION_CHECKS = (*CONFIGURATION_GREENS, "R-b5 persistence")  # (ii) is reported only
FAULTS = {
    None: contextlib.nullcontext,
    "e1_one_element": md._e1_mutant,  # R-b8's pre-build red
    "writing_readout": md._writing_readout,  # R-b8's with/without red
}


def execute(spec):
    """One job, inside its child process, at torch and BLAS one thread (R-b3 sets its own)."""
    torch.set_num_threads(1)
    kind = spec["kind"]
    with rp._blas_threads(1):
        if kind == "replay":
            stream = md._load(spec["stream"])
            with FAULTS[spec.get("fault")]():
                result = md.replay(
                    stream, spec["arm"], readout=spec.get("readout", True), pool=spec.get("pool")
                )
        elif kind == "cost":
            stream = md._load(spec["stream"])
            percepts = [event["x"] for event in stream["events"]]
            result = rp.cost(spec["candidate"], spec["config"], stream["seed"], percepts)
        elif kind == "mechanism":
            result = mechanism_checks(spec["candidate"])
        elif kind == "configuration":
            result = configuration_checks(spec["candidate"], spec["config"])
        else:
            raise ValueError(f"REFUSED: unknown job kind {kind!r}")
    if kind != "replay":  # a replay carries its own revision; section 2.10: dirty beside head
        result = dict(result, revision=dict(zip(("head", "dirty"), md._revision(), strict=True)))
    md._save(spec["out"], _plain(result))


def _child(spec):
    """memory_diagnostic._child's pattern for this module's jobs: the spec beside its output, a
    900 s timeout, OK / FAIL / TIMEOUT written to <out>.status.json."""
    out = spec["out"]
    job = f"{out}.job.json"
    md._save(job, _plain(spec))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "snn_data_lake.cortex.pool_selection", "job", job],
            capture_output=True,
            text=True,
            timeout=JOB_TIMEOUT_S,
        )
        if result.returncode:
            status = {"status": "FAIL", "path": out, "stderr": result.stderr[-1000:]}
        else:
            status = {"status": "OK", "path": out}
    except subprocess.TimeoutExpired:
        status = {"status": "TIMEOUT", "path": out}
    md._save(f"{out}.status.json", status)
    return status


def _record_child(job):
    return md._child(["--record", str(job["seed"]), job["out"]], job["out"])


def run_jobs(jobs, runner=None):
    """Section 2.10's workers: 4 child processes at a time. A FAIL cancels the jobs not yet started
    and ABORTS (section 2.8); a TIMEOUT is recorded and makes what needed it ineligible."""
    runner = runner or _child
    statuses = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(runner, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            status = future.result()
            statuses[futures[future]] = status
            if status["status"] == "FAIL":
                for other in futures:
                    other.cancel()
    if any(status is not None and status["status"] == "FAIL" for status in statuses):
        raise Abort(
            [
                status or {"status": "CANCELLED", "path": jobs[i].get("out")}
                for i, status in enumerate(statuses)
                if status is None or status["status"] != "OK"
            ]
        )
    return statuses


# ---- the step protocol (section 2.8's taxonomy) --------------------------------------------------


def _summary_path(out, step):
    return Path(out) / f"{step}.summary.json"


def require(out, step, revision):
    """Refuse to start unless the previous step's summary exists, COMPLETED, and was written at this
    same clean head; refuse a dirty tree, and a step that already ran here (section 2.8: no retries)."""
    head, dirty = revision()
    if dirty:
        raise Refused(
            f"{step}: the tree is dirty; every step runs at one clean head (section 2.10)"
        )
    if _summary_path(out, step).exists():
        raise Refused(f"{step}: already ran in {out}; section 2.8 allows no retries")
    index = STEPS.index(step)
    if index:
        previous = STEPS[index - 1]
        path = _summary_path(out, previous)
        if not path.exists():
            raise Refused(f"{step}: {previous} has no summary in {out}")
        summary = md._load(path)
        if summary.get("status") != "COMPLETED":
            raise Refused(f"{step}: {previous} ended {summary.get('status')}")
        if summary.get("head") != head or summary.get("dirty") is not False:
            raise Refused(
                f"{step}: {previous} was written at {summary.get('head')} "
                f"(dirty={summary.get('dirty')}), not at the clean head {head}"
            )
    return head


def run_step(step, out, *, revision=None, function=None, **kwargs):
    """Run one step under section 2.8's taxonomy and write its summary: COMPLETED (exit 0), HALTED
    naming the shared check (exit 2), ABORTED on a FAILed job or a head that moved (exit 1)."""
    revision = revision or md._revision
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    head = require(out, step, revision)
    function = function or STEP_FUNCTIONS[step]
    try:
        body, status, code = function(out, **kwargs), "COMPLETED", EXIT_OK
    except Halt as halt:
        body, status, code = {"check": halt.check, "detail": halt.detail}, "HALTED", EXIT_HALT
    except Abort as abort:
        body, status, code = {"failed": abort.jobs}, "ABORTED", EXIT_ABORT
    except Exception as error:  # a defect in the parent is a FAIL too (section 2.8), recorded
        body = {"error": repr(error), "traceback": traceback.format_exc()[-4000:]}
        status, code = "ABORTED", EXIT_ABORT
    end, dirty = revision()
    if status == "COMPLETED" and (end != head or dirty):
        body = {"moved": {"start": head, "end": end, "dirty": dirty}}
        status, code = "ABORTED", EXIT_ABORT
    summary = {"step": step, "status": status, "head": head, "dirty": dirty, **body}
    md._save(_summary_path(out, step), _plain(summary))
    print(f"{step}: {status}" + (f" ({body['check']})" if status == "HALTED" else ""))
    return code


def _summary(out, step):
    return md._load(_summary_path(out, step))


def _folder(out, step):
    folder = Path(out) / step
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _recording(out, seed):
    return Path(out) / "record" / f"{seed}-stream.json.gz"


def _recorded(out):
    return {int(seed): row["status"] for seed, row in _summary(out, "record")["recordings"].items()}


def _synthetic(folder, seeds):
    paths = {}
    for seed in seeds:
        paths[seed] = folder / f"task-{seed}-stream.json.gz"
        md._save(paths[seed], md.rb6_stream(seed))
    return paths


def _streams(paths):
    return {seed: md._load(path) for seed, path in paths.items() if Path(path).exists()}


# ---- the steps -----------------------------------------------------------------------------------


def step_record(out):
    """Seeds 100-103 and 200-207 through memory_diagnostic's own --record (the pilot's workers)."""
    folder = _folder(out, "record")
    jobs = [{"seed": seed, "out": str(folder / f"{seed}-stream.json.gz")} for seed in RECORD_SEEDS]
    recordings = {}
    for job, status in zip(jobs, run_jobs(jobs, _record_child), strict=True):
        row = {"status": status["status"]}
        if status["status"] == "OK":
            stream = md._load(job["out"])
            count = len(stream["events"])
            available = _scores(stream, [None] * count, [False] * count)
            if not available:
                raise Halt("scored segments", f"recording {job['seed']} has none (section 2.5)")
            row.update(
                events=count,
                segments={segment: available[segment]["n"] for segment in available},
                source_digest=md._source_digest(stream),
                recorded_at=stream["final"]["head"],
                recorded_dirty=stream["final"]["dirty"],
            )
        recordings[str(job["seed"])] = row
    return {"recordings": recordings}


def step_mechanism(out):
    """R-b0x's readout fixtures here; each candidate's pool checks in its own child process. A red
    that does not fire halts; a failed green refuses its candidate (section 2.8)."""
    folder = _folder(out, "mechanism")
    readout = md.check_r_b0x_readout()
    if not all(readout["red"].values()):
        raise Halt("R-b0x readout: a red that does not fire", readout)
    if not readout["verdict"]:
        raise Halt("R-b0x readout", readout)
    jobs = [
        {"kind": "mechanism", "candidate": c, "out": str(folder / f"{c}.json")}
        for c in rp.CANDIDATES
    ]
    results, refused = {}, {}
    for job, status in zip(jobs, run_jobs(jobs), strict=True):
        if status["status"] == "OK":
            results[job["candidate"]] = md._load(job["out"])
        else:
            refused[job["candidate"]] = ["TIMEOUT: its mechanism checks did not finish"]
    silent = [
        f"{c} {name}" for c, r in results.items() for name, fired in r["reds"].items() if not fired
    ]
    if silent:
        raise Halt("a red that does not fire", silent)
    for candidate, result in results.items():
        failed = [name for name, ok in result["greens"].items() if not ok]
        if failed:
            refused[candidate] = failed
    verdicts = {c: {"greens": r["greens"], "reds": r["reds"]} for c, r in results.items()}
    return {"readout_r_b0x": readout, "verdicts": verdicts, "refused": refused}


def _pilot_run(pilot, arm):
    for suffix in (".json", ".json.gz"):
        path = Path(pilot) / f"{PILOT_SEED}-{arm}-causal{suffix}"
        if path.exists():
            return md._load(path)
    raise Halt("R-b8 pre-build", f"the pilot has no {PILOT_SEED}-{arm}-causal replay")


def _contrast(stream, first, second):
    """(first) - (second) on a synthetic stream's scored events, through both learners."""
    row = {}
    for learner in ("readout", "grower"):
        a, b = (recording_accuracy(digest(run, stream)[learner]) for run in (first, second))
        row[learner] = {"first": a, "second": b, "difference": a - b}
    return row


def _same(first, second, keys=md.GROWER_KEYS):
    return all(first[key] == second[key] for key in keys)


def step_instrument(out, pilot):
    """R-b2 and R-b7's reds here; R-b6 on 4242 and its history-free red, and R-b8's two halves and
    their reds, each replay in its own child. Every failure here is the shared instrument's."""
    folder = _folder(out, "instrument")
    r_b2 = md.check_r_b2()
    if not r_b2["verdict"]:
        raise Halt("R-b2", r_b2)
    r_b7 = r_b7_reds()
    if not r_b7["verdict"]:
        raise Halt("R-b7 reds", r_b7)
    if _recorded(out).get(PILOT_SEED) != "OK":
        raise Halt("R-b8", f"seed {PILOT_SEED}'s recording at this head is missing")
    families = {
        "rb6": folder / f"rb6-{RB6_SEED}-stream.json.gz",
        "rb6red": folder / f"rb6red-{RB6_SEED}-stream.json.gz",
    }
    md._save(families["rb6"], md.rb6_stream(RB6_SEED))
    md._save(families["rb6red"], md.rb6_stream(RB6_SEED, history_free=True))
    new = _recording(out, PILOT_SEED)
    jobs = {}
    for name, path in families.items():
        for arm in ("e1", "e1-0"):
            jobs[f"{name} {arm}"] = _replay_job(path, arm, folder / f"{name}-{arm}.json.gz")
    for arm in md.SCORED_ARMS:
        jobs[f"on {arm}"] = _replay_job(new, arm, folder / f"{PILOT_SEED}-{arm}-on.json.gz")
        jobs[f"off {arm}"] = _replay_job(
            new, arm, folder / f"{PILOT_SEED}-{arm}-off.json.gz", readout=False
        )
    jobs["e1 mutant"] = _replay_job(
        new, "e1", folder / f"{PILOT_SEED}-e1-mutant.json.gz", fault="e1_one_element"
    )
    jobs["writing readout"] = _replay_job(
        new, "a", folder / f"{PILOT_SEED}-a-writing.json.gz", fault="writing_readout"
    )
    statuses = dict(zip(jobs, run_jobs(list(jobs.values())), strict=True))
    late = sorted(name for name, status in statuses.items() if status["status"] != "OK")
    if late:
        raise Halt("instrument TIMEOUT", late)
    rb6 = {}
    for name, path in families.items():
        stream = md._load(path)
        runs = [_checked(jobs[f"{name} {arm}"], stream) for arm in ("e1", "e1-0")]
        rb6[name] = _contrast(stream, *runs)
    if not rb6["rb6red"]["readout"]["difference"] < md.RB6_BAR:
        raise Halt("R-b6 red", rb6)
    if not rb6["rb6"]["readout"]["difference"] >= md.RB6_BAR:
        raise Halt("R-b6", rb6)
    stream = md._load(new)
    pinned = md._load(Path(pilot) / f"{PILOT_SEED}-stream.json.gz")
    on = {arm: _checked(jobs[f"on {arm}"], stream) for arm in md.SCORED_ARMS}
    off = {arm: _checked(jobs[f"off {arm}"], stream) for arm in md.SCORED_ARMS}
    mutant = _checked(jobs["e1 mutant"], stream)
    written = _checked(jobs["writing readout"], stream)
    pilot_runs = {arm: _pilot_run(pilot, arm) for arm in md.SCORED_ARMS}
    r_b8 = {
        "source_digest": {"new": md._source_digest(stream), "pilot": md._source_digest(pinned)},
        "pre_build": {arm: _same(on[arm], pilot_runs[arm]) for arm in md.SCORED_ARMS},
        "pre_build_red": not _same(mutant, pilot_runs["e1"]),
        "with_without": {
            arm: _same(on[arm], off[arm]) and off[arm]["readout"] is None for arm in md.SCORED_ARMS
        },
        "with_without_red": not _same(written, on["a"]),
    }
    if not r_b8["pre_build_red"]:
        raise Halt("R-b8 pre-build red", r_b8)
    if not r_b8["with_without_red"]:
        raise Halt("R-b8 with/without red", r_b8)
    digests = r_b8["source_digest"]
    if digests["new"] != digests["pilot"] or not all(r_b8["pre_build"].values()):
        raise Halt("R-b8 pre-build", r_b8)
    if not all(r_b8["with_without"].values()):
        raise Halt("R-b8 with/without", r_b8)
    return {"R-b2": r_b2, "R-b7 reds": r_b7, "R-b6": rb6, "R-b8": r_b8}


def step_screen(out):
    """Section 2.4: each candidate's draw order costed on seed 200 (batches of the draws still
    needed, one child each), walked by recurrent_pool.screen, then recurrent_pool.manifest."""
    folder = _folder(out, "screen")
    if _recorded(out).get(SCREEN_SEED) != "OK":
        raise Halt("screen", f"seed {SCREEN_SEED}'s recording is missing")
    stream = _recording(out, SCREEN_SEED)
    screens = {}
    for candidate in rp.CANDIDATES:
        order, measured, position, kept = rp.draw_order(candidate), {}, 0, 0
        while kept < rp.MANIFEST_SIZE and position < len(order):
            batch = order[position : position + rp.MANIFEST_SIZE - kept]
            jobs = [
                {
                    "kind": "cost",
                    "candidate": candidate,
                    "config": config,
                    "stream": str(stream),
                    "out": str(folder / f"{candidate}-{position + i:03d}.json"),
                }
                for i, config in enumerate(batch)
            ]
            for config, job, status in zip(batch, jobs, run_jobs(jobs), strict=True):
                ok = status["status"] == "OK"
                measured[rp.crc_json(config)] = md._load(job["out"]) if ok else None
            kept += sum(
                measured[rp.crc_json(c)] is not None
                and measured[rp.crc_json(c)]["cost_ms"] <= rp.COST_BOUND_MS
                for c in batch
            )
            position += len(batch)

        def cost_ms(config, measured=measured):
            timing = measured[rp.crc_json(config)]
            return math.inf if timing is None else timing["cost_ms"]  # a TIMEOUT is over the bound

        screen = rp.screen(candidate, cost_ms)
        for row in screen["dropped"]:
            if math.isinf(row["cost_ms"]):
                row.update(cost_ms=None, status="TIMEOUT")
        screen["timings"] = [measured[rp.crc_json(c)] for c in screen["configs"]]
        screens[candidate] = screen
    manifest = rp.manifest(screens)
    return {
        "manifest": manifest,
        "crc32": manifest["crc32"],
        "draw_order_crc32": rp.draw_order_crc32(),
        "screens": screens,
    }


def step_threads(out):
    """R-b3's equality, with R-b5 (i) and (iii) and (ii)'s report, on every manifest configuration
    of every candidate still standing. A failed green refuses the candidate (no single-thread
    waiver); a TIMEOUT makes its configuration ineligible."""
    folder = _folder(out, "threads")
    refused = dict(_summary(out, "mechanism")["refused"])
    manifest = _summary(out, "screen")["manifest"]["configs"]
    jobs = [
        {
            "kind": "configuration",
            "candidate": c,
            "index": i,
            "config": config,
            "out": str(folder / f"{c}-{i:02d}.json"),
        }
        for c in rp.CANDIDATES
        if c not in refused
        for i, config in enumerate(manifest[c])
    ]
    results, ineligible = {}, {}
    for job, status in zip(jobs, run_jobs(jobs), strict=True):
        candidate, index = job["candidate"], job["index"]
        if status["status"] != "OK":
            ineligible.setdefault(candidate, []).append(index)
            continue
        result = md._load(job["out"])
        results.setdefault(candidate, {})[str(index)] = {
            "verdicts": {name: result[name]["verdict"] for name in CONFIGURATION_CHECKS},
            "R-b3 settings": result["R-b3"]["numbers"]["settings"],
        }
        failed = [name for name in CONFIGURATION_GREENS if not result[name]["verdict"]]
        if failed:
            refused.setdefault(candidate, []).append(f"configuration {index}: {failed}")
    return {"results": results, "refused": refused, "ineligible": ineligible}


def step_sigma(out):
    """Section 2.6: each standing candidate's first surviving manifest configuration, Carried and
    Settled on 100-103; sigma_dev is the largest per-seed SD of Carried minus Settled through the
    readout; sigma_plan = max(0.0436, sigma_dev); S by the power rule."""
    folder = _folder(out, "sigma")
    refused = _summary(out, "threads")["refused"]
    manifest = _summary(out, "screen")["manifest"]["configs"]
    standing = {c: [(0, manifest[c][0])] for c in rp.CANDIDATES if c not in refused}
    table = _phase_table(out, folder, standing, DEV_SEEDS, (), FIT_FORMS, ())
    per_candidate = {}
    for c, [(_, config)] in standing.items():
        crc = rp.crc_json(rp.canonical(c, config))
        pairs = [
            tuple(table[("pool", c, crc, form, seed, False)] for form in FIT_FORMS)
            for seed in DEV_SEEDS
        ]
        if any(cell["status"] != "OK" for pair in pairs for cell in pair):
            per_candidate[c] = {"complete": False}
            continue
        per_candidate[c] = {"complete": True, "config": config}
        for learner in ("readout", "grower"):
            d = [
                recording_accuracy(a["digest"][learner]) - recording_accuracy(b["digest"][learner])
                for a, b in pairs
            ]
            per_candidate[c][learner] = {"per_seed": d, "sd": statistics.stdev(d)}
    measured = [row["readout"]["sd"] for row in per_candidate.values() if row["complete"]]
    if not measured:
        raise Halt("sigma", "no candidate's first configuration completed on 100-103")
    sigma_dev = max(measured)
    sigma_plan = max(SIGMA_PROXY, sigma_dev)
    return {
        "per_candidate": per_candidate,
        "sigma_dev": sigma_dev,
        "sigma_plan": sigma_plan,
        "band_D": band_d(sigma_dev),
        **power_s(sigma_plan),
    }


def _phase_table(out, folder, standing, live, task, forms, references):
    """Cells for standing {candidate: [(index, config)]} x forms x (live + task), and the reference
    arms on the live recordings; run, validated and bound into R-b7's table."""
    recorded = _recorded(out)
    tasks = _synthetic(folder, task)
    paths = {seed: _recording(out, seed) for seed in live} | tasks
    cells = []
    for candidate, configs in standing.items():
        for index, config in configs:
            for form in forms:
                for seed in live + (task if form in FIT_FORMS else ()):
                    name = f"{candidate}-{index:02d}-{form}-{seed}.json.gz"
                    cells.append(
                        _cell(
                            paths[seed],
                            folder / name,
                            seed,
                            index=index,
                            spec=_spec(candidate, config, form),
                            missing=seed in live and recorded.get(seed) != "OK",
                        )
                    )
    for arm in references:
        for seed in live:
            missing = recorded.get(seed) != "OK"
            name = f"ref-{arm}-{seed}.json.gz"
            cells.append(_cell(paths[seed], folder / name, seed, arm=arm, missing=missing))
    return _tabulate(cells, live + task, _streams(paths))


def step_tune(out):
    """Section 2.5's tuning: per standing candidate, D and D_task through the readout on 200-203 and
    5001-5004, eligibility against (e1-0), the ordering; its head is the tuned configuration."""
    folder = _folder(out, "tune")
    threads = _summary(out, "threads")
    refused = dict(threads["refused"])
    blocked = threads["ineligible"]
    band = _summary(out, "sigma")["band_D"]
    screen = _summary(out, "screen")
    manifest = screen["manifest"]["configs"]
    standing = {c: list(enumerate(manifest[c])) for c in rp.CANDIDATES if c not in refused}
    table = _phase_table(out, folder, standing, TUNE_SEEDS, TUNE_TASK_SEEDS, FIT_FORMS, ("e1-0",))
    reference = _reference(table, "e1-0", TUNE_SEEDS)
    candidates = {}
    for candidate, configs in standing.items():
        costs = screen["screens"][candidate]["costs_ms"]
        entries = []
        for index, config in configs:
            entry = _entry(
                table,
                candidate,
                index,
                config,
                costs[index],
                TUNE_SEEDS,
                TUNE_TASK_SEEDS,
                order=index,
                blocked=index in blocked.get(candidate, ()),
            )
            entry["eligible"], entry["reasons"] = eligibility(entry, reference["mean"])
            entries.append(entry)
        ranked = ordering([e for e in entries if e["eligible"]], band)
        tuned = None
        if ranked:
            head = ranked[0]
            tuned = {"index": head["index"], "config": head["config"], "cost": head["cost"]}
        else:
            refused[candidate] = ["no eligible configuration in tuning"]
        candidates[candidate] = {
            "entries": entries,
            "ranking": [e["index"] for e in ranked],
            "tuned": tuned,
        }
    return {"band_D": band, "reference": reference, "candidates": candidates, "refused": refused}


# distinct-from: cortex/recurrent_pool.py -- check_r_b3_red is R-b3's thread-order witness red, not
#   R-b4's replay noise red; prior_drift_harness.red1 and growing_learner.load_state_dict share only words.
def noise_red_fires(plain, noisy):
    """R-b4's (pool-noise) red, section 2.7 as amended 2026-09-25: the kicks must change the
    continuation-state CHAIN (the pool's fingerprint at every percept boundary) AND the phi
    fingerprints. The final state alone is one sample: in campaign 1, g3 forgot every kick by the
    last percept of seed 203 while its phi changed, so a final-state red was blind."""
    chains = plain["pool"]["state_chain"] != noisy["pool"]["state_chain"]
    return chains and plain["phi"] != noisy["phi"]


def _twins_differ(first, second):
    """R-b4's comparison of two fresh-process replays: (phi or pool state differ, a learner differs).
    Pool state is the final fingerprint and, amended 2026-09-25, the per-percept state chain."""
    a, b = first["pool"] or {}, second["pool"] or {}
    pool = (
        first["phi"] != second["phi"]
        or a.get("state") != b.get("state")
        or a.get("state_chain") != b.get("state_chain")
    )
    learner = not _same(first, second, ("predictions", "fallbacks", "fingerprint", "refits")) or (
        (first["readout"] or {}).get("predictions") != (second["readout"] or {}).get("predictions")
        or (first["readout"] or {}).get("fingerprint")
        != (second["readout"] or {}).get("fingerprint")
    )
    return pool, learner


def step_check203(out):
    """R-b1: G1(a)'s R-a2 on every tuned candidate's Carried and Settled forms on seed 203 at the
    three cuts, for both learners, with (leak) as the red. The per-candidate R-b4 twins (Carried,
    fresh processes), with a (pool-noise) replay as the harness-side red."""
    folder = _folder(out, "check203")
    tune = _summary(out, "tune")
    refused = dict(tune["refused"])
    tuned = {c: row["tuned"] for c, row in tune["candidates"].items() if row["tuned"]}
    if _recorded(out).get(CHECK_SEED) != "OK":
        raise Halt("R-b1", f"seed {CHECK_SEED}'s recording is missing")
    path = _recording(out, CHECK_SEED)
    stream = md._load(path)
    count = len(stream["events"])
    cuts = (count // 4, count // 2, 3 * count // 4)
    changed = {}
    for cut in cuts:
        changed[cut] = folder / f"{CHECK_SEED}-cut-{cut}.json.gz"
        md._save(changed[cut], md.mutate_suffix(stream, max(1, cut), stream["seed"] + cut))
    jobs, bound = {}, {}
    for c, t in tuned.items():
        for form in FIT_FORMS:
            spec = _spec(c, t["config"], form)
            bound[f"R-b1 {c} {form}"] = _bound(CHECK_SEED, spec)
            name = f"R-b1 {c} {form}"
            jobs[f"{name} base"] = _replay_job(
                path, "pool", folder / f"{c}-{form}.json.gz", pool=spec
            )
            for cut in cuts:
                out_path = folder / f"{c}-{form}-cut-{cut}.json.gz"
                jobs[f"{name} {cut}"] = _replay_job(changed[cut], "pool", out_path, pool=spec)
        for twin in ("1", "2", "noise"):
            spec = _spec(c, t["config"], "carried", noise=twin == "noise")
            bound[f"R-b4 {c} {twin}"] = _bound(CHECK_SEED, spec)
            out_path = folder / f"{c}-twin-{twin}.json.gz"
            jobs[f"R-b4 {c} {twin}"] = _replay_job(path, "pool", out_path, pool=spec)
    jobs["R-b1 leak base"] = _replay_job(path, "leak", folder / "leak.json.gz")
    for cut in cuts:
        out_path = folder / f"leak-cut-{cut}.json.gz"
        jobs[f"R-b1 leak {cut}"] = _replay_job(changed[cut], "leak", out_path)
    statuses = dict(zip(jobs, run_jobs(list(jobs.values())), strict=True))
    late = sorted(name for name, status in statuses.items() if status["status"] != "OK")
    if any(name.startswith("R-b1") for name in late):
        raise Halt("R-b1 TIMEOUT", late)
    sources = {cut: md._load(changed[cut]) for cut in cuts}

    def load(name, cut=None, filed=None):
        return _checked(jobs[name], stream, sources.get(cut), cell=bound.get(filed))

    def before(first, second, cut):
        """Both learners' predictions up to the cut: (grower equal, readout equal)."""
        return (
            first["predictions"][:cut] == second["predictions"][:cut],
            first["readout"]["predictions"][:cut] == second["readout"]["predictions"][:cut],
        )

    leak = load("R-b1 leak base")
    red = {str(cut): not any(before(leak, load(f"R-b1 leak {cut}", cut), cut)) for cut in cuts}
    r_b4, noise_red = {}, {}
    for c in tuned:
        if any(name.startswith(f"R-b4 {c} ") for name in late):
            refused[c] = ["R-b4 TIMEOUT on 203: its tuned configuration is ineligible"]
            continue
        first, second, noisy = (
            load(f"R-b4 {c} {t}", filed=f"R-b4 {c} {t}") for t in ("1", "2", "noise")
        )
        noise_red[c] = noise_red_fires(first, noisy)  # section 2.7, amended 2026-09-25
        pool_differs, learner_differs = _twins_differ(first, second)
        r_b4[c] = {
            "pool_differs": pool_differs,
            "learner_differs": learner_differs,
            "executions": first["execution_id"] != second["execution_id"],
        }
    if not all(red.values()) or not all(noise_red.values()):
        raise Halt("a red that does not fire", {"R-b1 leak": red, "R-b4 pool noise": noise_red})
    r_b1 = {}
    for c in tuned:
        for form in FIT_FORMS:
            name = f"R-b1 {c} {form}"
            base = load(f"{name} base", filed=name)
            r_b1[f"{c} {form}"] = {
                str(cut): all(before(base, load(f"{name} {cut}", cut, filed=name), cut))
                for cut in cuts
            }
    if not all(all(held.values()) for held in r_b1.values()):
        raise Halt("R-b1", r_b1)
    for c, row in r_b4.items():
        if not row["executions"]:
            raise Halt("R-b4", f"{c}: the twins are one execution")
        if row["pool_differs"]:
            refused[c] = ["R-b4: phi or pool state differ between fresh-process twins on 203"]
        elif row["learner_differs"]:
            raise Halt("R-b4 learner state", {c: row})
    return {
        "cuts": list(cuts),
        "R-b1": r_b1,
        "R-b1 red": red,
        "R-b4": r_b4,
        "R-b4 red": noise_red,
        "refused": refused,
    }


def step_select(out):
    """Section 2.5's selection: the tuned candidates x the four forms on 204-207 and x {carried,
    settled} on 5005-5008, with (e1-0) and (e1) on 204-207 through both learners; eligibility, the
    ordering by candidate index, and the winner or nothing."""
    folder = _folder(out, "select")
    refused = dict(_summary(out, "check203")["refused"])
    tune = _summary(out, "tune")
    band = _summary(out, "sigma")["band_D"]
    tuned = {
        c: row["tuned"]
        for c, row in tune["candidates"].items()
        if row["tuned"] and c not in refused
    }
    standing = {c: [(t["index"], t["config"])] for c, t in tuned.items()}
    table = _phase_table(
        out, folder, standing, SELECT_SEEDS, SELECT_TASK_SEEDS, rp.FORMS, ("e1-0", "e1")
    )
    references = {arm: _reference(table, arm, SELECT_SEEDS) for arm in ("e1-0", "e1")}
    streams = _streams({seed: _recording(out, seed) for seed in SELECT_SEEDS})
    entries = []
    for c, t in tuned.items():
        entry = _entry(
            table,
            c,
            t["index"],
            t["config"],
            t["cost"],
            SELECT_SEEDS,
            SELECT_TASK_SEEDS,
            order=rp.INDEX[c],
            forms=rp.FORMS,
        )
        entry["eligible"], entry["reasons"] = eligibility(entry, references["e1-0"]["mean"])
        entry["selectable"] = c in SELECTABLE
        entry["input_response"] = {
            str(seed): input_response(c, t["config"], streams[seed])
            for seed in SELECT_SEEDS
            if seed in streams
        }
        entries.append(entry)
    ranked, reference = rank_selection(entries, band)
    winner = None
    if ranked:
        head = ranked[0]
        winner = {"candidate": head["candidate"], "index": head["index"], "config": head["config"]}
    return {
        "selectable": list(SELECTABLE),
        "band_D": band,
        "references": references,
        "entries": entries,
        "ranking": [e["candidate"] for e in ranked],
        "reference_ranking": [e["candidate"] for e in reference],
        "winner": winner,
        # section 2.8: selection outcomes are inspected once any selection replay is written
        "inspected": True,
        "refused": refused,
    }


def step_winner(out):
    """R-b4 after selection: the winner's Carried and Settled forms, two fresh-process twins on each
    of 204-207. A phi or pool-state difference refuses the winner (a repair after inspection:
    section 2.8 re-runs selection on 208-211 and 5009-5012); a learner difference halts."""
    folder = _folder(out, "winner")
    winner = _summary(out, "select")["winner"]
    if winner is None:
        return {"winner": None, "note": "nothing was selected; no twins to run"}
    jobs = {}
    for form in FIT_FORMS:
        spec = _spec(winner["candidate"], winner["config"], form)
        for seed in SELECT_SEEDS:
            for twin in ("1", "2"):
                out_path = folder / f"{form}-{seed}-twin-{twin}.json.gz"
                jobs[f"{form} {seed} {twin}"] = _replay_job(
                    _recording(out, seed), "pool", out_path, pool=spec
                )
    statuses = dict(zip(jobs, run_jobs(list(jobs.values())), strict=True))
    late = sorted(name for name, status in statuses.items() if status["status"] != "OK")
    if late:
        return {"winner": winner, "verdict": "UNSHOWN", "timeouts": late, "refused": True}
    rows = {}
    for form in FIT_FORMS:
        spec = _spec(winner["candidate"], winner["config"], form)
        for seed in SELECT_SEEDS:
            stream = md._load(_recording(out, seed))
            first, second = (
                _checked(jobs[f"{form} {seed} {twin}"], stream, cell=_bound(seed, spec))
                for twin in ("1", "2")
            )
            pool_diff, learner_diff = _twins_differ(first, second)
            rows[f"{form} {seed}"] = {"pool_differs": pool_diff, "learner_differs": learner_diff}
            if not pool_diff and learner_diff:
                raise Halt("R-b4 learner state", rows)
    refused = any(row["pool_differs"] for row in rows.values())
    return {
        "winner": winner,
        "verdict": "REFUSED" if refused else "HELD",
        "R-b4": rows,
        "refused": refused,
    }


STEP_FUNCTIONS = {
    "record": step_record,
    "mechanism": step_mechanism,
    "instrument": step_instrument,
    "screen": step_screen,
    "threads": step_threads,
    "sigma": step_sigma,
    "tune": step_tune,
    "check203": step_check203,
    "select": step_select,
    "winner": step_winner,
}


# ---- the self-check (under 20 s; never a campaign) -----------------------------------------------


def _hand_entry(order, d, d_task, cost):
    return {"order": order, "D": d, "D_task": d_task, "cost": cost}


def _refuses(out, step, revision):
    try:
        require(out, step, revision)
    except Refused:
        return True
    return False


def _clean():
    return "h1", False


def self_check():
    # section 2.5's ordering on hand-built tables: band_D, the D_task tie, the cost tie, exact ties
    a, b = _hand_entry(0, 0.030, 0.00, 10.0), _hand_entry(1, 0.020, 0.20, 10.0)
    assert ordering([a, b], band_d(0.001))[0] is a  # 0.010 apart: outside the 0.005 floor
    assert ordering([a, b], band_d(0.012))[0] is b  # inside sigma_dev's band: D_task decides
    c = _hand_entry(2, 0.030, 0.197, 5.0)
    assert ordering([a, b, c], band_d(0.012))[0] is c  # D_task within 0.005: the cheaper wins
    assert ordering([_hand_entry(3, 0.030, 0.197, 5.4), c], 0.005)[0] is c  # 8% dearer: tied
    assert ordering([_hand_entry(0, 0.030, 0.197, 5.4), c], 0.005)[0]["order"] == 0
    assert ordering([_hand_entry(0, 0.030, 0.197, 5.6), c], 0.005)[0] is c  # 12% dearer
    ranked = ordering([_hand_entry(i, 0.01, 0.1, 1.0) for i in (3, 1, 2)], 0.005)
    assert [e["order"] for e in ranked] == [1, 2, 3]  # exact ties: manifest order or index
    good = {"ok": True, "carried_live": [0.80] * 4, "degenerate": [False] * 4}
    assert eligibility(good, 0.85)[0] and not eligibility(good, 0.86)[0]
    assert not eligibility(dict(good, degenerate=[False, True, False, False]), 0.5)[0]
    assert not eligibility(dict(good, ok=False), 0.5)[0]
    assert degenerate({"rows": 5, "constant_fraction": 0.9, "distinct_rows": 5})
    assert not degenerate({"rows": 5, "constant_fraction": 0.89, "distinct_rows": 2})
    # section 2.6's power rule at its own published values
    assert power_s(0.0436) == {"S": 31, "capped": False}
    assert power_s(0.0436, 0.03)["S"] == 15 and power_s(0.08, 0.03)["S"] == 46
    assert power_s(0.0436, 0.01)["capped"]
    # R-b7's three refusals, beside the well-formed table
    reds = r_b7_reds()
    assert reds["verdict"], reds
    # the step order, the taxonomy and the exit codes, on a scratch OUT at a stand-in revision (the
    # stand-in steps' own status lines are swallowed so a log never shows a HALTED campaign step)
    with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
        out = Path(scratch)

        def halts(_):
            raise Halt("R-b6", "stand-in")

        assert _refuses(out, "mechanism", _clean)  # no previous summary
        assert _refuses(out, "record", lambda: ("h1", True))  # a dirty tree
        assert run_step("record", out, revision=_clean, function=lambda _: {"n": 0}) == EXIT_OK
        assert _refuses(out, "record", _clean)  # no retries
        assert _refuses(out, "mechanism", lambda: ("h2", False))  # another head
        assert run_step("mechanism", out, revision=_clean, function=halts) == EXIT_HALT
        assert _summary(out, "mechanism")["check"] == "R-b6"
        assert _refuses(out, "instrument", _clean)  # the previous step HALTED
    # one pool-arm replay on a short synthetic stream; its window is shortened to 16 for this
    # 48-event fixture ONLY, and R-b7 refuses that binding in any campaign table
    fixture = md._synthetic_stream()
    fixture["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    spec = {"candidate": "lsm", "config": rp.R_B3_WITNESS["lsm"], "form": "settled", "window": 16}
    run = md.replay(fixture, "pool", pool=spec)
    md._validate_run(run, fixture, fixture, "pool", "causal")
    assert run["pool"]["frozen"] and run["pool"]["boundary"] is not None
    filed = _bound(fixture["seed"], _spec("lsm", spec["config"], "settled"))
    assert binding_problems(dict(filed, digest=digest(run, fixture))) == ["window"]
    print(
        "pool_selection self-check OK: ordering, eligibility, power, R-b7 reds, step order, exit codes, pool arm"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pool_selection",
        description="G1(b)'s pool-selection campaign (g1b-pool-selection.md section 2); no "
        "arguments runs the self-check",
    )
    commands = parser.add_subparsers(dest="command")
    for step in STEPS:
        sub = commands.add_parser(step, help=STEP_FUNCTIONS[step].__doc__.split("\n")[0])
        sub.add_argument("out", help="the campaign directory (outside the repository)")
        if step == "instrument":
            sub.add_argument(
                "--pilot",
                required=True,
                help="G1(a)'s pilot3 directory at 829fb40 (100-stream.json.gz, 100-<arm>-causal.json)",
            )
    commands.add_parser(
        "job", help="run one job spec (the campaign's child processes)"
    ).add_argument("spec")
    args = parser.parse_args(argv)
    if args.command is None:
        self_check()
        return EXIT_OK
    if args.command == "job":
        execute(md._load(args.spec))
        return EXIT_OK
    extra = {}
    if args.command == "instrument":
        missing = [
            name
            for name in [f"{PILOT_SEED}-stream.json.gz"]
            + [f"{PILOT_SEED}-{arm}-causal.json" for arm in md.SCORED_ARMS]
            if not (Path(args.pilot) / name).exists()
            and not (Path(args.pilot) / f"{name}.gz").exists()
        ]
        if missing:
            print(f"REFUSED: the pilot directory lacks {missing}", file=sys.stderr)
            return EXIT_REFUSED
        extra["pilot"] = args.pilot
    try:
        return run_step(args.command, args.out, **extra)
    except Refused as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
