r"""G1(c) / T109 part (c): the campaign driver's pool stages, part 2a -- migrate, causality, primary.

The frozen text is specs/022-unified-brain-arc/evidence/g1c-carried-state.md (2.9, 2.10 R-b1, 2.3, 2.5,
2.11, 2.13). `carried_state_driver` owns the stage list, the CLI and the first five stages; the stages here
run through the same `pool_selection.run_step` protocol (`function=` of `carried_state_driver.run_stage`),
so the single-head, dirty-tree, order and no-retry refusals and the exit codes are pool_selection's own.
This module also holds the helpers every pool stage shares (`carried_state_controls` and
`carried_state_report` import them): the cell key and accuracy readers of a pool table, `run_cells` (one
worker batch for several R-b7 tables), `use_of` (the host the migration gate chose) and `settings` (the
real window, b, R-b3 size and kick count, and the one place a test may shorten them).

  migrate     (2.9) "call" jobs for the host parity fixtures and the hosted R-b3 (with its red) and R-b4;
              the TUNED configuration x {carried, settled} x {port, hosted} on 204-207 and on the
              synthetic 5005-5008 regenerated at this head; D_live and D_task per host; the machinery red
              (a hosted Reset standing in for Carried must fail the D_task conjunct);
              `migration_reading` -> "hosted" or "port"; the quantization sensitivity (the hosted pool
              alone at frac_bits 12, 16, 24 through the same decoder, REPORT-ONLY, never filed in R-b7's
              table) and the parity horizon (the hosted pool against the port over one gate recording at
              frac_bits 4-32, REPORT-ONLY). Every later stage reads `use` from this summary.
  causality   (2.10 R-b1) G1(a)'s R-a2 on the surviving host's Carried and Settled, both learners, three
              cuts, on seed 203; the (leak) red must change predictions at every cut; a failure halts.
  primary     (2.1, 2.3, 2.5) four forms x the live seeds on the surviving host; R-b4 twins (a second
              fresh-process replay of Carried and Settled on every seed, compared with
              pool_selection._twins_differ: a difference halts); the pool-noise red on seed 100.

A blind-harness reading or a failed migration conjunct is a READING, never a halt (the stage COMPLETES); a
red that does not fire, a missing or TIMEOUT cell, an R-b4 difference and R-b1 halt (2.11).

Names avoid the verbs the gradient-claim census reads. Deterministic except R-b4's unseeded kicks (a red).
"""

import statistics
import tempfile
from pathlib import Path

import numpy as np

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
HOSTS = ("port", "kernel")  # the migration gate's two hosts (recurrent_pool.HOSTS)
REAL_SCALE = {  # what every stage runs at; only a test names another (disclosed in each summary)
    "window": rp.WINDOW,
    "b": md.TRANSPORT_B,
    "percepts": rp.R_B3_PERCEPTS,
    "kicks": 16,
}
QUANT_BITS = (12, 16, 24)  # 2.9: the quantization-sensitivity report's frac_bits (report only)
PARITY_BITS = tuple(range(4, 33))  # 2.9: the parity horizon is reported for frac_bits 4-32
# G1B 3.4: g1's per-seed D on 204-207, which the port's D here is reported against (2.9)
G1B_PER_SEED_D = (-0.0063, 0.0082, 0.0007, -0.0276)
RP = "snn_data_lake.reservoir.recurrent_pool:"
HERE = "snn_data_lake.instruments.carried_state_stages:"


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
    """The host the migration gate chose ("port" or "kernel"): every later stage binds its replays to it."""
    use = ps._summary(out, "migrate")["use"]
    if use not in HOSTS:
        raise ps.Halt("migrate summary", f"use {use!r} is not a host")
    return use


# ---- cells, tables and accuracies (one reader for every pool stage) --------------------------------


def key(form, seed, *, host, window, arm="pool", transport=None, encoder=None, noise=False):
    """The R-b7 table key of one pool cell of the TUNED configuration (pool_selection.cell_key)."""
    cell = {
        "arm": arm,
        "candidate": POOL,
        "config": cs.TUNED,
        "form": form,
        "seed": seed,
        "noise": noise,
        "window": window,
        "host": host,
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


def pool_table(out, folder, live, forms, paths, *, host, window, task=()):
    """pool_selection's phase table for the TUNED configuration over `forms` x `live` (+ `task` for
    Carried and Settled), on `host` at `window`: run, validated, bound into R-b7's table."""
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
        host=host,
    )


def pool_cell(path, out_path, seed, form, *, host, window, stream_key=None):
    """One pool replay as a table cell; `stream_key` names a stream other than the seed's own recording
    (a shuffled one): the cell is then scored on it and its replay's source is held to it."""
    spec = ps._spec(POOL, cs.TUNED, form, window=window, host=host)
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


def accuracies(table, seeds, forms, *, host, window, **named):
    """{learner: {form: {str(seed): accuracy}}} over an OK table of the TUNED configuration."""
    return {
        learner: {
            form: {
                str(seed): accuracy(
                    table, key(form, seed, host=host, window=window, **named), learner
                )
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


def mean_of(values):
    return statistics.fmean(values.values())


# ---- migrate (2.9) -------------------------------------------------------------------------------


def _call(target, out, **args):
    return {"kind": "call", "target": target, "args": args, "out": str(out)}


# Sites the duplicate-capability hook flagged on the next functions (read; none runs a hosted pool's gate):
# distinct-from: gates/__init__.py -- the gates package's docstring
# distinct-from: growth/h01_ingest.py::_selfcheck -- one ingest module's self-check on synthetic fixtures
def quantized_decoder(stream, form, frac_bits, window=rp.WINDOW):
    """REPORT-ONLY (2.9's quantization sensitivity; a "call" target): the hosted pool alone at
    `frac_bits` over one stream's percepts, its phi through the reference readout scored as the campaign's
    tables score it. md.replay cannot run a non-binding frac_bits (a pool spec has no such key, and R-b7
    holds a hosted binding's whole policy to the lineage's), so this is a direct run, never filed in an
    R-b7 table; at the binding frac_bits it must equal the table's own replay (the migrate stage asserts
    it: two instruments that must agree)."""
    data = md._load(stream)
    pool = rp.make_pool(
        POOL, cs.TUNED, data["seed"], form, window=window, host="kernel", frac_bits=int(frac_bits)
    )
    rows = [pool.percept(event["x"]).tolist() for event in data["events"]]
    return {
        "frac_bits": int(frac_bits),
        "form": form,
        "window": window,
        "report_only": True,
        "accuracy": cs._decoder_accuracy(data, rows),
    }


# Sites the duplicate-capability hook flagged on the parity horizon (read; none compares two hosts' activity):
# distinct-from: gates/gradient_claim_gate.py::counts -- per-state counts of governed learning rules
# distinct-from: gates/determinism_gate.py::binding_frac_bits -- reads the MEASURED binding frac_bits from tools/determinism_baseline.json; the horizon below runs the hosted pool at EVERY width of 2.9's range against the port and reports where they part
def parity_horizon(stream, frac_bits, window=rp.WINDOW):
    """REPORT-ONLY (2.9's parity horizon; a "call" target): the hosted pool at `frac_bits` against the
    float64 port over one recording's percepts in Carried form: the first tick at which their activity
    differs (None: never) and the fraction of ticks with equal activity. Whole-trajectory parity is not
    expected at the binding frac_bits (2.9's prediction); this reports how far it holds, at each width."""
    data = md._load(stream)
    sequence = [event["x"] for event in data["events"]]
    trains = {}
    for host, options in (("port", {}), ("kernel", {"frac_bits": int(frac_bits)})):
        pool = rp.make_pool(
            POOL, cs.TUNED, data["seed"], "carried", window=window, host=host, **options
        )
        rows = trains[host] = []
        pool.observer = lambda _pool, activity, rows=rows: rows.append(np.array(activity))
        for x in sequence:
            pool.percept(x)
    same = [
        bool(np.array_equal(a, b)) for a, b in zip(trains["port"], trains["kernel"], strict=True)
    ]
    return {
        "frac_bits": int(frac_bits),
        "ticks": len(same),
        "first_diverging_tick": next((i for i, equal in enumerate(same) if not equal), None),
        "equal_fraction": sum(same) / len(same),
        "report_only": True,
    }


def _gate_jobs(folder, sc):
    """The four pool-only checks of 2.9 (i) and (iii), each a "call" job."""
    return {
        "parity": _call(RP + "check_host_parity", folder / "parity.json"),
        "r_b3": _call(
            RP + "check_r_b3",
            folder / "r_b3.json",
            candidate=POOL,
            config=cs.TUNED,
            host="kernel",
            window=sc["window"],
            percepts=sc["percepts"],
        ),
        "r_b3_red": _call(
            RP + "check_r_b3_hosted_red",
            folder / "r_b3-red.json",
            candidate=POOL,
            window=sc["window"],
            percepts=sc["percepts"],
        ),
        "r_b4": _call(
            RP + "pool_noise_red",
            folder / "r_b4.json",
            candidate=POOL,
            config=cs.TUNED,
            window=sc["window"],
            after=sc["kicks"],
            host="kernel",
        ),
    }


def _quantization_jobs(folder, paths, seeds, bits, sc):
    """The hosted pool at each of `bits` over every gate stream and form, as "call" jobs; keyed
    {bits: {form: {str(seed): job}}}."""
    return {
        b: {
            form: {
                str(seed): _call(
                    HERE + "quantized_decoder",
                    folder / f"quant-{b}-{form}-{seed}.json",
                    stream=str(paths[seed]),
                    form=form,
                    frac_bits=b,
                    window=sc["window"],
                )
                for seed in seeds
            }
            for form in ps.FIT_FORMS
        }
        for b in bits
    }


def _quantization(quant, statuses, acc, gate, task, binding):
    """The report-only readings per frac_bits, and the agreement of the call-job instrument with the
    hosted table at the binding frac_bits (two instruments that must agree, or the stage halts)."""
    seeds = (*gate, *task)
    rows = {}
    for b, forms in quant.items():
        found = {
            form: {
                seed: md._load(job["out"])["accuracy"]
                for seed, job in jobs.items()
                if statuses[job["out"]]["status"] == "OK"
            }
            for form, jobs in forms.items()
        }
        complete = all(len(found[form]) == len(seeds) for form in found)

        def mean_gap(where, found=found):
            return statistics.fmean(
                found["carried"][str(seed)] - found["settled"][str(seed)] for seed in where
            )

        rows[str(b)] = {
            "complete": complete,
            "D_live": mean_gap(gate) if complete else None,
            "D_task": mean_gap(task) if complete else None,
            "accuracy": found,
        }
    held = rows[str(binding)]
    agree = {
        where: held["complete"]
        and all(
            held["accuracy"][form][str(seed)] == acc[where]["readout"][form][str(seed)]
            for form in ps.FIT_FORMS
            for seed in group
        )
        for where, group in (("live", gate), ("task", task))
    }
    if not all(agree.values()):
        raise ps.Halt("the quantization instrument disagrees with the hosted table", agree)
    return {
        "report_only": True,
        "binding_frac_bits": binding,
        "rows": rows,
        "table_agreement": agree,
    }


def stage_migrate(out, *, scale=None):
    """The migration gate (2.9): hosted or port for every later stage; the quantization sensitivity beside it."""
    sc = settings(scale)
    window, gate, task = sc["window"], cs.GATE_SEEDS, cs.GATE_SYNTHETIC_SEEDS
    recorded = ps._recorded(out)
    absent = [seed for seed in gate if recorded.get(seed) != "OK"]
    if absent:
        raise ps.Halt("migrate", {"gate recordings not OK": absent})
    folder = ps._folder(out, "migrate")
    paths = {
        **{seed: ps._recording(out, seed) for seed in gate},
        **ps._synthetic(ps._folder(out, "migrate/synthetic"), task),  # regenerated at this head
    }
    binding = rp.KernelLsmPool.lineage_policy(cs.TUNED)["frac_bits"]
    bits = sorted({binding, *QUANT_BITS})
    checks = _gate_jobs(folder, sc)
    quant = _quantization_jobs(folder, paths, (*gate, *task), bits, sc)
    horizon = {
        b: _call(
            HERE + "parity_horizon",
            folder / f"horizon-{b}.json",
            stream=str(paths[gate[0]]),
            frac_bits=b,
            window=window,
        )
        for b in PARITY_BITS
    }
    batch = [
        *checks.values(),
        *horizon.values(),
        *(job for forms in quant.values() for jobs in forms.values() for job in jobs.values()),
    ]
    statuses = {job["out"]: status for job, status in zip(batch, ps.run_jobs(batch), strict=True)}
    found = {
        name: md._load(job["out"]) if statuses[job["out"]]["status"] == "OK" else None
        for name, job in checks.items()
    }
    if found["r_b3_red"] is None or not found["r_b3_red"]["verdict"]:  # a red that does not fire
        raise ps.Halt(
            "a red that does not fire", {"hosted R-b3 float-accumulation red": found["r_b3_red"]}
        )
    tables = {}
    for host in HOSTS:
        tables[host] = pool_table(
            out,
            ps._folder(out, f"migrate/{host}"),
            gate,
            ps.FIT_FORMS,
            paths,
            host=host,
            window=window,
            task=task,
        )
        need_ok(tables[host], f"migrate: {host} replays")
    reset = [
        pool_cell(
            paths[seed],
            ps._folder(out, "migrate/reset") / f"reset-{seed}.json.gz",
            seed,
            "reset",
            host="kernel",
            window=window,
        )
        for seed in task
    ]
    (reset_table,) = run_cells([(reset, task, ps._streams(paths))])
    need_ok(reset_table, "migrate: the machinery red's replays")
    acc = {
        host: {
            where: accuracies(tables[host], seeds, ps.FIT_FORMS, host=host, window=window)
            for where, seeds in (("live", gate), ("task", task))
        }
        for host in HOSTS
    }
    d = {host: {where: gaps(acc[host][where]) for where in acc[host]} for host in HOSTS}
    mean = {
        host: {where: mean_of(d[host][where]["readout"]) for where in d[host]} for host in HOSTS
    }
    standin = statistics.fmean(
        accuracy(reset_table, key("reset", seed, host="kernel", window=window))
        - acc["kernel"]["task"]["readout"]["settled"][str(seed)]
        for seed in task
    )
    parity_ok = bool(found["parity"] and found["parity"]["verdict"])
    r_b3_ok = bool(found["r_b3"] and found["r_b3"]["verdict"])
    r_b4_ok = bool(found["r_b4"] and found["r_b4"]["verdict"])
    reading = cs.migration_reading(
        parity_ok,
        mean["kernel"]["live"],
        mean["port"]["live"],
        mean["kernel"]["task"],
        mean["port"]["task"],
        r_b3_ok,
        r_b4_ok,
    )
    # the machinery red: a hosted Reset standing in for Carried must fail the D_task conjunct alone
    red_fires = not cs.within(standin, mean["port"]["task"])
    isolated = cs.migration_reading(True, 0.0, 0.0, standin, mean["port"]["task"], True, True)
    control = cs.migration_reading(
        True, 0.0, 0.0, mean["port"]["task"], mean["port"]["task"], True, True
    )
    red = {
        "D_task_reset": standin,
        "D_task_port": mean["port"]["task"],
        "fired": red_fires,
        "isolated": isolated,
        "control": control,
    }
    if not (red_fires and isolated == "port" and control == "hosted"):
        raise ps.Halt("a red that does not fire", {"migration gate machinery red": red})
    port = [d["port"]["live"]["readout"][str(seed)] for seed in gate]
    return {
        "window": window,
        "scale": sc,
        "shortened": sc != REAL_SCALE,
        "reading": reading,
        "use": "kernel" if reading == "hosted" else "port",
        "conjuncts": {
            "parity": parity_ok,
            "d_live_within": cs.within(mean["kernel"]["live"], mean["port"]["live"]),
            "d_task_within": cs.within(mean["kernel"]["task"], mean["port"]["task"]),
            "r_b3": r_b3_ok,
            "r_b4": r_b4_ok,
        },
        "mean": mean,
        "d": d,
        "accuracy": acc,
        "checks": found,
        "machinery_red": red,
        "quantization": _quantization(quant, statuses, acc["kernel"], gate, task, binding),
        "parity_horizon": {
            "report_only": True,
            "rows": {
                str(b): md._load(job["out"]) if statuses[job["out"]]["status"] == "OK" else None
                for b, job in horizon.items()
            },
        },
        "port_vs_g1b": {
            "seeds": list(gate),
            "port": port,
            "g1b": list(G1B_PER_SEED_D),
            "difference": [a - b for a, b in zip(port, G1B_PER_SEED_D, strict=True)],
        },
    }


# ---- causality (2.10 R-b1) -----------------------------------------------------------------------


# Sites the duplicate-capability hook flagged on stage_causality (read; neither builds R-b1's replay jobs):
# distinct-from: instruments/pool_selection.py::step_check203 -- G1(b)'s sibling wrapper (the tuning summary's candidates on the port, with R-b4's twins); both CALL pool_selection.RB1Check, the one construction of R-b1's jobs, cuts and prefix reads, so this is no second mechanism
# distinct-from: faculties/theory_of_mind.py::_run_sally_anne -- stages the Sally-Anne false-belief task for a TheoryOfMind; no replay, no cut
def stage_causality(out, *, scale=None):
    """R-b1 on seed 203: G1(a)'s R-a2 on the surviving host's Carried and Settled at three cuts, both
    learners; the (leak) arm must change predictions at every cut. A failure halts (2.10, 2.11). The
    jobs, the cuts and the prediction-prefix reads are `pool_selection.RB1Check`'s."""
    sc = settings(scale)
    window, seed, host = sc["window"], cs.CAUSALITY_SEED, use_of(out)
    if ps._recorded(out).get(seed) != "OK":
        raise ps.Halt("R-b1", f"seed {seed}'s recording is missing")
    folder = ps._folder(out, "causality")
    path = ps._recording(out, seed)
    stream = md._load(path)
    pools = {  # explicit specs: the surviving host's Carried and Settled
        form: (form, ps._spec(POOL, cs.TUNED, form, window=window, host=host))
        for form in ps.FIT_FORMS
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


def _twin_rows(folder, live, paths, table, host, window):
    """R-b4: a second fresh-process replay of Carried and Settled on every seed, held against the
    primary's own; a difference in the pool or a learner, or twins in one process, halts."""
    jobs, bound = {}, {}
    for form in ps.FIT_FORMS:
        spec = ps._spec(POOL, cs.TUNED, form, window=window, host=host)
        for seed in live:
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
        first = md._load(table[key(form, seed, host=host, window=window)]["job"]["out"])
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
    if any(row["pool_differs"] or row["learner_differs"] for row in rows.values()):
        raise ps.Halt("R-b4", rows)
    return rows


def _noise_red(out, folder, host, window):
    """R-b4's pool-noise red on seed 100: a replay with unseeded kicks must move the state chain and phi
    of an otherwise identical plain Carried replay (pool_selection.noise_red_fires)."""
    seed = cs.IDENTITY_SEED
    path = ps._recording(out, seed)
    stream = md._load(path)
    jobs, bound = {}, {}
    for name, noise in (("plain", False), ("noise", True)):
        spec = ps._spec(POOL, cs.TUNED, "carried", noise=noise, window=window, host=host)
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
    """Four forms x the live seeds on the surviving host, R-b4 twins on every seed, the pool-noise red
    on seed 100; per seed D through both learners, and the per-seed, per-form, per-segment tables 2.3
    requires (`ps._report` of every cell: both learners' accuracy, entropy, coverage, fallback rate,
    degeneracy, startup), with the input-response statistics of each recording."""
    sc = settings(scale)
    window, host = sc["window"], use_of(out)
    live = tuple(ps._summary(out, "record")["live_seeds"])
    folder = ps._folder(out, "primary")
    paths = {seed: ps._recording(out, seed) for seed in live}
    table = pool_table(out, folder, live, rp.FORMS, paths, host=host, window=window)
    need_ok(table, "primary")
    twins = _twin_rows(folder, live, paths, table, host, window)
    noise = _noise_red(out, folder, host, window)
    streams = ps._streams(paths)
    acc = accuracies(table, live, rp.FORMS, host=host, window=window)
    return {
        "host": host,
        "window": window,
        "scale": sc,
        "shortened": sc != REAL_SCALE,
        "live_seeds": list(live),
        "accuracy": acc,
        "d": gaps(acc),
        "cells": {
            form: {
                str(seed): ps._report(table[key(form, seed, host=host, window=window)])
                for seed in live
            }
            for form in rp.FORMS
        },
        "input_response": {
            str(seed): ps.input_response(POOL, cs.TUNED, streams[seed]) for seed in live
        },
        "R-b4": twins,
        "R-b4 red": noise,
    }


STAGE_FUNCTIONS = {
    "migrate": stage_migrate,
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
    gap arithmetic on hand numbers, and the hosted pool's quantized decoder on the 48-event fixture at the
    binding frac_bits agreeing with a replay of the hosted pool arm (two instruments that must agree)."""
    assert tuple(STAGE_FUNCTIONS) == ("migrate", "causality", "primary")
    assert settings() == REAL_SCALE and settings({"window": 16})["window"] == 16
    try:
        settings({"nope": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown scale key was accepted")
    assert len(G1B_PER_SEED_D) == len(cs.GATE_SEEDS) and 12 in QUANT_BITS
    # the key is pool_selection's: a hosted cell differs from the port's, a window from the default
    port = key("carried", 7, host="port", window=rp.WINDOW)
    kernel = key("carried", 7, host="kernel", window=rp.WINDOW)
    assert port[:6] == kernel[:6] and port != kernel and "kernel" in kernel
    assert key("carried", 7, host="port", window=0) != port
    hand = {"readout": {"carried": {"1": 0.7}, "settled": {"1": 0.5}}}
    assert gaps(hand) == {"readout": {"1": 0.7 - 0.5}}
    # the quantized decoder at the lineage's frac_bits equals the hosted replay's own readout accuracy
    stream = md._synthetic_stream()
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    bits = rp.KernelLsmPool.lineage_policy(cs.TUNED)["frac_bits"]
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "fixture.json"
        md._save(path, stream)
        for form in ps.FIT_FORMS:
            spec = ps._spec(POOL, cs.TUNED, form, window=16, host="kernel")
            run = md.replay(stream, "pool", pool=spec)
            replayed = ps.recording_accuracy(ps.digest(run, stream)["readout"])
            direct = quantized_decoder(str(path), form, bits, window=16)
            assert direct["accuracy"] == replayed and direct["report_only"], (
                form,
                direct,
                replayed,
            )
    print(
        "carried_state_stages self-check OK: scale, cell keys, gaps, the quantized decoder at "
        f"frac_bits {bits} equal to the hosted replay on the fixture"
    )


if __name__ == "__main__":
    self_check()
