r"""G1(c) / T109 part (c): the campaign's report stage and its golden check (g1c-carried-state.md 2.3, 2.5, 2.11).

  assemble / render   PURE: the saved summaries of the eleven stages in, one machine-readable report and
                      its markdown out. Every contrast row of the campaign is built here by
                      `carried_state_campaign.contrast_row` from the per-seed numbers the stages saved
                      (the oracle and the stream-history rows are recomputed from the references stage's
                      accuracies and held equal to the rows that stage stored: two instruments that must
                      agree), and 2.5's G-R1 reading is `g_r1_reading` over them. One head, a clean tree
                      and COMPLETED on every stage are asserted first.
  stage_report        loads the summaries, writes OUT/report.json and OUT/report.md, files the report's
                      digest.
  golden              the golden check (2.11), in-process at the current head: the pool's hand fixtures and
                      R-b3's fixture chain on the pool (the one implementation G1(c) 2.9 kept; an OUT's
                      migrate summary must name its host), a full replay of the 48-event synthetic fixture through
                      every arm and form (predictions, phi, both learners' fingerprints, pool state
                      chains), and `assemble` / `render` recomputed over OUT's saved summaries. It returns a
                      digest dict; `golden OUT --against FILE` compares two. Its RED is a one-ulp change to
                      the pool's leak (`leak_nudged`), which must move the digest.

The fixture is shorter than one window, so the golden replays it at GOLDEN_WINDOW and GOLDEN_B (16, as
the self-checks do; disclosed here, and the digest names them). A change to code the campaign executed
after its first recording is tested by comparing two golden digests (2.11): equal, or the campaign
restarts on the reserve seeds.

Deterministic; ASCII-only output. Names avoid the verbs the gradient-claim census reads.
"""

import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from snn_data_lake.instruments import carried_state_campaign as cs
from snn_data_lake.instruments import carried_state_stages as st
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.reservoir import recurrent_pool as rp

# Sites the duplicate-capability hook flagged on this module's docstring (read; none assembles a campaign's report):
# distinct-from: engine/learn_daemon.py::_selfcheck -- a daemon's self-check that must never start its drive loop
# distinct-from: gates/manifest_gate.py::run -- the wiring manifest's CI body: five checks over a committed artifact
# distinct-from: gates/policy_manifest.py::_selfcheck_tools -- the tools/ gate names run_selfchecks lists
# distinct-from: gates/selfcheck_cache.py::SelfcheckCache._refresh_derived -- recomputes the cache's rosters and envelope from the tree
# distinct-from: infra/self_mod_harness.py::SelfModHarness._run_ruff -- runs ruff inside a sandbox

STAGES_READ = tuple(
    "record mechanism instrument references construct migrate causality primary controls transport separation".split()
)
# The migration gate's stage ran in the campaign at 5761f3a7 (g1c-carried-state.md 3.3) and was deleted with
# the port and the host switch (2.9): that campaign's OUT holds its summary, which is read and held to the
# same provenance as the rest; a later campaign has none
STAGES_OPTIONAL = ("migrate",)
# what the report keeps of the separation stage: its reading AND the measurements it was read from
SEPARATION_KEYS = tuple(
    "run item_3 reason reading host window stream_checks accuracy mean per_text distinct_settled_rows".split()
)


# Sites the duplicate-capability hook flagged on the report's helpers (read; none digests a campaign's summaries):
# distinct-from: instruments/pool_selection.py::_plain -- makes a value JSON-safe; digest_of CALLS it and hashes the result
# distinct-from: faculties/recursive_tom.py::run_t082_dry_run -- one seed's apparatus dry run of another task
# distinct-from: growth/cortical_prior.py::validate -- lists the reference data missing a mandatory attribute
# distinct-from: instruments/binding_instrument.py::run_t080_dry_run -- one seed's apparatus dry run of another task
def digest_of(value):
    """sha256 of the canonical JSON of `value` (numpy scalars and paths made plain first)."""
    text = json.dumps(ps._plain(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def load_summaries(out):
    """The stage summaries OUT holds, by stage (a missing one is simply absent)."""
    return {
        stage: md._load(path)
        for stage in STAGES_READ
        if (path := ps._summary_path(out, stage)).exists()
    }


def provenance_problems(summaries):
    """Why these summaries are not one campaign (empty: clear): a stage missing, not COMPLETED, written on
    a dirty tree, or at another head than the rest (2.13: one clean head)."""
    problems = [
        f"{stage}: no summary"
        for stage in STAGES_READ
        if stage not in summaries and stage not in STAGES_OPTIONAL
    ]
    for stage, body in summaries.items():
        if body.get("status") != "COMPLETED":
            problems.append(f"{stage}: {body.get('status')}, not COMPLETED")
        if body.get("dirty") is not False:
            problems.append(f"{stage}: written on a dirty tree")
    heads = sorted({str(body.get("head")) for body in summaries.values()})
    if len(heads) > 1:
        problems.append(f"more than one head: {heads}")
    return problems


def _row(name, values, seeds):
    return cs.contrast_row(name, [values[seed] for seed in seeds])


def _agree(stored, fresh, what):
    """Two instruments that must agree: a row a stage stored equals its recomputation here."""
    if digest_of(stored) != digest_of(fresh):
        raise ValueError(f"{what}: the stored row differs from its recomputation")


def _references(summaries, live):
    """The oracle and stream-history rows, recomputed from the references stage's accuracies and held
    equal to the rows that stage stored."""
    record, refs = summaries["record"], summaries["references"]
    table = {
        learner: {int(seed): row for seed, row in refs["accuracy"][learner].items()}
        for learner in st.LEARNERS
    }
    oracle = cs.oracle_check(
        table["readout"],
        table["grower"],
        seeds=tuple(record["live_seeds"]),
        allow_reserve=record["seed_set"] == "reserve",
    )
    history = {
        learner: cs.contrast_row(
            f"stream history (e1) - (e1-0) through the {learner}",
            [
                refs["accuracy"][learner][seed]["e1"] - refs["accuracy"][learner][seed]["e1-0"]
                for seed in live
            ],
        )
        for learner in st.LEARNERS
    }
    _agree(refs["oracle"], oracle, "oracle")
    _agree(refs["stream_history"], history, "stream history")
    return oracle, history


def _rows(summaries, live, oracle, history):
    prim, ctl, trn = summaries["primary"], summaries["controls"], summaries["transport"]
    d, grower, shuf = prim["d"]["readout"], prim["d"]["grower"], ctl["d_shuf"]
    zero = ctl["window0"]
    rows = {
        "oracle (f64) - (a), readout": oracle["decoder"],
        "oracle (f64) - (a), grower": oracle["grower"],
        "stream history, readout": history["readout"],
        "stream history, grower": history["grower"],
        "primary D, readout": _row(
            "D = accuracy(Carried) - accuracy(Settled), reference readout", d, live
        ),
        "primary D, grower": _row("D, the grower's (beside, never decisive)", grower, live),
        "D_shuf, readout": _row(
            "D_shuf = mean over K shuffles of Carried - Settled, reference readout",
            shuf["readout"],
            live,
        ),
        "D_shuf, grower": _row("D_shuf, the grower's (beside)", shuf["grower"], live),
        "D - D_shuf, readout": cs.incremental(
            [d[seed] for seed in live], [shuf["readout"][seed] for seed in live]
        ),
        "D0, readout": _row(
            "D0 = Carried - Settled@0, reference readout", zero["d0"]["readout"], live
        ),
        "W = D0 - D, readout": _row("W = D0 - D, reference readout", zero["w"]["readout"], live),
    }
    swap = trn["arms"]["pool-swap"]["d"]["readout"]
    for arm in md.TRANSPORT_ARMS:
        moved = trn["arms"][arm]["d"]["readout"]
        rows[f"D_t {arm}, readout"] = _row(f"D_t = Carried - Settled through {arm}", moved, live)
        rows[f"D_t - D {arm}, readout"] = _row(
            f"D_t - D ({arm} minus the random encoder)", {s: moved[s] - d[s] for s in live}, live
        )
        if arm != "pool-swap":
            rows[f"D_t - D_swap {arm}, readout"] = _row(
                f"D_t - D_swap ({arm} minus the swap control)",
                {s: moved[s] - swap[s] for s in live},
                live,
            )
    return rows


def _migration(mig):
    """The migration gate's record (2.9), as the campaign at 5761f3a7 wrote it (None for a campaign after
    its deletion). Its reading is recomputed from the recorded conjuncts and means by 2.9's pure
    function and must equal the recorded one: two instruments that must agree."""
    if mig is None:
        return None
    conj, mean = mig["conjuncts"], mig["mean"]
    again = cs.migration_reading(
        conj["parity"],
        mean["kernel"]["live"],
        mean["port"]["live"],
        mean["kernel"]["task"],
        mean["port"]["task"],
        conj["r_b3"],
        conj["r_b4"],
    )
    if again != mig["reading"]:
        raise ValueError(f"migrate: the recorded reading {mig['reading']!r} is not 2.9's {again!r}")
    kept = (
        "reading use conjuncts mean machinery_red port_vs_g1b quantization parity_horizon".split()
    )
    return {key: mig[key] for key in kept}


def _assemble(summaries):
    record, prim, ctl = summaries["record"], summaries["primary"], summaries["controls"]
    inst, sep = summaries["instrument"], summaries["separation"]
    arms = summaries["transport"]["arms"]
    shown = sep["input_response"] if sep["run"] else {}  # a separation that ran files its own (R3)
    live = [str(seed) for seed in record["live_seeds"]]
    oracle, history = _references(summaries, live)
    rows = _rows(summaries, live, oracle, history)
    blind_ok = bool(inst["rb6_ok"]) and bool(oracle["met"])  # 2.4 (i) and (ii)
    sighted = (
        bool(ctl["sighted"]) and ctl["host"] == prim["host"]
    )  # the binding names the primary's host
    g_r1 = cs.g_r1_reading(
        blind_harness_ok=blind_ok,
        primary=rows["primary D, readout"],
        stream_history=rows["stream history, readout"],
        d_shuf=rows["D_shuf, readout"],
        shuffle_sighted=sighted,
    )
    cells = prim["cells"]
    zero = ctl["window0"]
    return {
        "head": record["head"],
        "dirty": False,
        "seed_set": record["seed_set"],
        "seed_bundle": record["bundle"],
        "live_seeds": record["live_seeds"],
        "host": prim["host"],
        "shortened": any(bool(body.get("shortened")) for body in summaries.values()),
        "g_r1": g_r1,
        "inputs": {
            "blind_harness_ok": blind_ok,
            "rb6_clauses": inst.get("rb6_clauses"),
            "oracle_met": bool(oracle["met"]),
            "shuffle_sighted": sighted,
            "shuffle_halves": ctl["halves"],
            "d_shuf_survives": rows["D_shuf, readout"]["met"],
            "controls_host": ctl["host"],
            "primary_host": prim["host"],
        },
        "rows": rows,
        "migration": _migration(summaries.get("migrate")),
        "per_seed": {
            seed: {
                "D": prim["d"]["readout"][seed],
                "D_grower": prim["d"]["grower"][seed],
                "D_shuf": ctl["d_shuf"]["readout"][seed],
                "D0": zero["d0"]["readout"][seed],
                "W": zero["w"]["readout"][seed],
                "startup_flagged": (cells["carried"][seed]["startup"] or {}).get("flagged"),
                "scored_rows_through_boundary": zero["scored_rows_through_boundary"][seed],
            }
            for seed in live
        },
        "tables": {  # 2.3: the per-segment diagnostics of EVERY contrasted variant, not the primary's alone
            "cells": cells,
            "input_response": prim["input_response"],
            "variant_response": {  # 2.3: every pool variant whose input is not the primary's (R3)
                **{f"through {arm}": body["input_response"] for arm, body in arms.items()},
                **{f"Separation {name}": rows for name, rows in shown.items()},
            },
            "references": {
                arm: row["per_seed"] for arm, row in summaries["references"]["references"].items()
            },
            "shuffle": ctl["cells"]["shuffle"],
            "window0": ctl["cells"]["window0"],
            "transport": {arm: body["cells"] for arm, body in arms.items()},
            "separation": sep.get("cells"),
        },
        "transport": {
            "b": summaries["transport"]["b"],
            "taps": summaries["transport"]["taps"],
            "records": {arm: body["records"] for arm, body in arms.items()},
            "forbidden_request": summaries["transport"]["forbidden_request"],
        },
        "construct": {
            key: summaries["construct"].get(key) for key in ("run", "verdict", "item_3", "reason")
        },
        "separation": {key: sep.get(key) for key in SEPARATION_KEYS},
        "causality": {
            "cuts": summaries["causality"]["cuts"],
            "R-b1": summaries["causality"]["R-b1"],
            "R-b1 red": summaries["causality"]["R-b1 red"],
        },
        "mechanism": {
            "refused": summaries["mechanism"]["refused"],
            "configuration": summaries["mechanism"]["configuration"]["verdicts"],
        },
    }


def assemble(summaries):
    """The campaign's report from its stage summaries (PURE); ValueError names what is wrong: a stage
    missing, not COMPLETED, dirty or at another head, a summary short of what the report reads, a contrast
    over other than the live seeds, or a stored row that disagrees with its recomputation."""
    problems = provenance_problems(summaries)
    if problems:
        raise ValueError("; ".join(problems))
    try:
        return _assemble(summaries)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"a summary lacks what the report reads: {error!r}") from error


# ---- the markdown ----------------------------------------------------------------------------------


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _contrast_lines(rows):
    names = "contrast|n|mean|sd|t|bar|lower bound|exact FPR|band|skew|+ / - / 0|met|flags"
    keys = ("n", "mean", "sd", "t", "bar", "lower_bound", "fpr", "band_population", "skewness")
    body = []
    for name, row in rows.items():
        s = row["signs"]
        flags = ", ".join(flag for flag in ("degenerate", "skew_flagged") if row[flag])
        signs = f"{s['positive']} / {s['negative']} / {s['zero']}"
        body.append((name, *(row[key] for key in keys), signs, row["met"], flags))
    return _table(names.split("|"), body)


def _segment_lines(cells):
    names = "seed|segment|n|readout acc|entropy|coverage|fallback|grower acc|entropy|coverage|"
    names += "fallback|distinct rows|startup flagged"
    keys, body = ("accuracy", "prediction_entropy", "coverage", "prior_fallback"), []
    for seed, cell in cells.items():
        pool = (
            (cell.get("degeneracy") or {}).get("distinct_rows"),
            (cell.get("startup") or {}).get("flagged"),
        )
        for segment, row in (cell.get("readout") or {}).items():
            theirs = (cell.get("grower") or {}).get(segment, {})
            body.append(
                (
                    seed,
                    segment,
                    row["n"],
                    *(row[k] for k in keys),
                    *(theirs.get(k) for k in keys),
                    *pool,
                )
            )
    return _table(names.split("|"), body)


# Site the duplicate-capability hook flagged on the next two helpers (read; they gather and format tables of a report):
# distinct-from: reasoning/rpm_solver.py::_parse_raven -- parses a 'raven:' RPM query string into context cells and candidates
# {variant name: {seed: cell}} of every contrasted variant but the primary's four forms (2.3)
def _variant_cells(tables):
    found = {f"Reference arm ({arm})": cells for arm, cells in tables["references"].items()}
    for k, forms in tables["shuffle"].items():
        found |= {f"{form}, within-block shuffle {k}": cells for form, cells in forms.items()}
    found |= {f"{form} at window 0": cells for form, cells in tables["window0"].items()}
    for arm, forms in tables["transport"].items():
        found |= {f"{form} through {arm}": cells for form, cells in forms.items()}
    for stream, arms in (tables["separation"] or {}).items():
        found |= {f"Separation {stream}, arm {arm}": cells for arm, cells in arms.items()}
    return found


def _text_accuracies(per_text):
    return " ".join(f"{t}:{'-' if hit is None else f'{hit:.2f}'}" for t, hit in per_text.items())


# Sites the duplicate-capability hook flagged on the render and the report stage (read; none writes this campaign's report):
# distinct-from: gates/growth_coverage_gate.py::self_check -- constructs every criterion's red of one coverage gate
# distinct-from: growth/h01_ingest.py::IngestRefused -- the exception an ingest gate raises with its readings
# distinct-from: gates/doc_budget_gate.py::_selfcheck -- constructs the reds of one doc-budget gate's statuses
# distinct-from: gates/docs_only_battery.py::code_hash -- hashes the non-Markdown lines of git ls-files
# distinct-from: instruments/pool_selection.py::_tabulate -- runs and validates cells into R-b7's table; the stage here only reads the summaries those tables produced
def _migration_lines(mig):
    """The report's migration-gate section: the record's, line for line, or one line saying no gate ran."""
    if mig is None:
        return [
            "",
            "## Migration gate: not run (2.9 deleted the port and the gate after the campaign at 5761f3a7)",
        ]
    quantization = {
        bits: {"D_live": row["D_live"], "D_task": row["D_task"]}
        for bits, row in mig["quantization"]["rows"].items()
    }
    horizon = {
        bits: None
        if row is None
        else {
            "first_diverging_tick": row["first_diverging_tick"],
            "equal_fraction": row["equal_fraction"],
        }
        for bits, row in mig["parity_horizon"]["rows"].items()
    }
    return [
        "",
        f"## Migration gate: {mig['reading']} (use {mig['use']})",
        "",
        "conjuncts: " + json.dumps(mig["conjuncts"], sort_keys=True),
        "mean D (live, task) per host: " + json.dumps(mig["mean"], sort_keys=True),
        "machinery red: " + json.dumps(mig["machinery_red"], sort_keys=True),
        "port D vs G1(b): " + json.dumps(mig["port_vs_g1b"], sort_keys=True),
        "quantization sensitivity (REPORT ONLY, never used to explain a failure away): "
        + json.dumps(quantization, sort_keys=True),
        "parity horizon, hosted vs port, per frac_bits (REPORT ONLY): "
        + json.dumps(horizon, sort_keys=True),
    ]


def render(report):
    """The report as markdown (PURE): the G-R1 reading, every contrast row, the migration gate, the
    per-seed values, the per-seed, per-form, per-segment tables (both learners' accuracy, entropy,
    coverage, prior-fallback rate), degeneracy, startup and input response, the transport records and the
    separation reading."""
    g, sep = report["g_r1"], report["separation"]
    lines = [
        "# G1(c) campaign report",
        "",
        f"head {report['head']} (clean), seeds {report['seed_set']}, surviving host {report['host']}",
        "seed bundle: " + json.dumps(report["seed_bundle"], sort_keys=True),
    ]
    if report["shortened"]:
        lines.append("SHORTENED: a test's window and sizes, never a campaign")
    reading = g["verdict"] + (f" ({g['wording']})" if g["wording"] else "")
    reading += " -- skew-flagged" if g["skew_flagged"] else ""
    inputs = json.dumps(report["inputs"], sort_keys=True)
    lines += ["", f"## G-R1 reads: {reading}", "", "inputs: " + inputs]
    lines += ["", "## Contrast rows (bar T_BAR_S31, exact sign-flip rate beside each)", ""]
    lines += [_contrast_lines(report["rows"])]
    lines += _migration_lines(report["migration"])
    lines += ["", "## Per seed (the reference readout; the grower's D beside it)", ""]
    names = "seed|D|D grower|D_shuf|D0|W|startup flagged|scored rows <= boundary"
    keys = ("D", "D_grower", "D_shuf", "D0", "W", "startup_flagged", "scored_rows_through_boundary")
    lines += [
        _table(
            names.split("|"), [(s, *(r[k] for k in keys)) for s, r in report["per_seed"].items()]
        )
    ]
    forms = dict(zip(rp.FORMS, ("Carried", "Reset", "Settled", "Front-end"), strict=True))
    variants = {name: report["tables"]["cells"][form] for form, name in forms.items()}
    variants.update(
        _variant_cells(report["tables"])
    )  # 2.3: every contrasted variant, not the primary's
    for name, cells in variants.items():
        heading = f"## {name}: accuracy, entropy, coverage, prior-fallback per seed and segment"
        lines += ["", heading, "", _segment_lines(cells)]
    carried, response = report["tables"]["cells"]["carried"], report["tables"]["input_response"]
    lines += ["", "## Degeneracy, startup and input response (Carried)", ""]
    names = "seed|distinct rows|constant fraction|startup flagged|driven|drive mean|drive sd"
    rows = [
        (
            seed,
            (cell.get("degeneracy") or {}).get("distinct_rows"),
            (cell.get("degeneracy") or {}).get("constant_fraction"),
            (cell.get("startup") or {}).get("flagged"),
            *(response[seed][key] for key in ("driven", "mean", "sd")),
        )
        for seed, cell in carried.items()
    ]
    lines += [_table(names.split("|"), rows)]
    lines += ["", "## Input response of the variants whose input is not the primary's", ""]
    lines += [
        "window 0 and the within-block shuffles drive the pool with the primary's inputs (above)"
    ]
    for name, per_seed in report["tables"]["variant_response"].items():
        rows = [(s, *(r[k] for k in ("driven", "mean", "sd"))) for s, r in per_seed.items()]
        lines += ["", f"### {name}", "", _table("seed|driven|drive mean|drive sd".split("|"), rows)]
    forbidden = json.dumps(report["transport"]["forbidden_request"], sort_keys=True)
    lines += [
        "",
        f"## Transport (b = {report['transport']['b']})",
        "",
        "forbidden-map request: " + forbidden,
        "records per arm and seed (L, padding fraction, raw row norms, encoder crc, silent events): "
        "see report.json",
        "",
        "## Construction check and separation",
        "",
        "construct: " + json.dumps(report["construct"], sort_keys=True),
        "separation: " + json.dumps({k: sep[k] for k in SEPARATION_KEYS[:4]}, sort_keys=True),
        "",
    ]
    if sep["run"]:  # the measurements the reading was read from, per stream, arm and seed
        distinct = sep["distinct_settled_rows"]
        measured = [
            (
                name,
                arm,
                seed,
                accuracy,
                _text_accuracies(sep["per_text"][name][arm][seed]),
                distinct[name][seed] if arm == "settled" else None,
            )
            for name, arms in sep["accuracy"].items()
            for arm, per_seed in arms.items()
            for seed, accuracy in per_seed.items()
        ]
        headers = "stream|arm|seed|accuracy|per-text accuracy|distinct Settled rows".split("|")
        lines += [
            "## Separation measurements (decoder accuracy)",
            "",
            _table(headers, measured),
            "",
        ]
    return "\n".join(lines)


# ---- the report stage ------------------------------------------------------------------------------


def stage_report(out):
    """Load every summary, assemble the report, write OUT/report.json and OUT/report.md, file the digest.
    A summary set that is not one clean campaign, or a short summary, halts."""
    summaries = load_summaries(out)
    try:
        report = assemble(summaries)
    except ValueError as error:
        raise ps.Halt("report", str(error)) from error
    markdown = render(report)
    with open(Path(out) / "report.json", "w", encoding="utf-8") as file:
        json.dump(ps._plain(report), file, indent=1, allow_nan=False)
    with open(Path(out) / "report.md", "w", encoding="utf-8") as file:
        file.write(markdown)
    return {
        "g_r1": report["g_r1"],
        "files": ["report.json", "report.md"],
        "shortened": report["shortened"],
        "digest": {"report": digest_of(report), "markdown": digest_of(markdown)},
    }


# ---- the golden check (2.11) -----------------------------------------------------------------------

GOLDEN_WINDOW = (
    16  # the 48-event fixture is shorter than one window: its replays run at 16 (disclosed)
)
GOLDEN_B = 16  # and the tap boundary of its transport arms
GOLDEN_PERCEPTS = 64  # R-b3's fixture chain at the self-check's small size (window 32 below)
GOLDEN_ARMS = (
    "a",
    "f64",
    "d",
    "d0",
    "e1",
    "e1-0",
    "e2",
    "e2-0",
    "leak",
)  # the deterministic non-pool arms


def _fixture():
    stream = md._synthetic_stream()
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    return stream


# Sites the duplicate-capability hook flagged on the golden's helpers (read; none records a replay's golden):
# distinct-from: reservoir/recurrent_pool.py::check_r_b5_freeze -- R-b5 (iii) over a pool's four forms; the hand record below runs the same four forms and hashes them
# distinct-from: instruments/pool_selection.py::digest -- reduces a replay to what the tables read; _core keeps what the GOLDEN compares (state chains, fingerprints, phi)
# distinct-from: infra/cockpit.py::_http_get -- drives one GET through the cockpit's real handler
# distinct-from: instruments/binding_instrument.py::t080_recognitions -- another campaign's decomposition of a binder's handle
# distinct-from: instruments/prior_drift_harness.py::ArmResult -- the scored record of one wiring of another harness
def _core(run):
    """What a replay's golden record keeps: the grower's predictions, phi, fingerprint and refits, the
    readout's predictions and fingerprint, and the pool's state chain (and its transport record)."""
    pool = run.get("pool")
    return {
        "predictions": run["predictions"],
        "fallbacks": run["fallbacks"],
        "phi": run["phi"],
        "fingerprint": run["fingerprint"],
        "refits": run["refits"],
        "readout": {
            "predictions": run["readout"]["predictions"],
            "fingerprint": run["readout"]["fingerprint"],
        },
        "pool": None
        if pool is None
        else {
            key: pool.get(key)
            for key in ("state_chain", "state", "boundary", "frozen", "encoder", "transport")
        },
    }


def _hand_record():
    """The pool's hand fixture (recurrent_pool's HAND_*) through all four forms at a one- and a
    four-percept window: every step's activity, membrane and spikes as hex, and the pool's
    fingerprint."""
    bits = rp.KernelLsmPool.lineage_policy(cs.TUNED)["frac_bits"]
    record = {}
    for form in rp.FORMS:
        for window in (1, 4):
            pool = rp._hosted(
                rp.KernelLsmPool,
                bits,
                rp.HAND_PRE,
                rp.HAND_W,
                rp.HAND_CFG,
                form,
                window=window,
                s_settle=1,
            )
            rows = rp._percepted(pool, rp.HAND_U)
            record[f"{form}/{window}"] = [
                [row[0].hex(), row[1].hex(), row[2].hex(), row[3]] for row in rows
            ] + [pool.fingerprint()]
    return record


def _chain_record():
    """R-b3's fixture chain: the cycled fixture percepts through the TUNED configuration's Carried and
    Settled pools, every step's activity, phi, final state and the chain of the state at every percept
    boundary (recurrent_pool._record)."""
    seq = rp.cycled_fixture(GOLDEN_PERCEPTS)
    return {
        form: rp._record(
            rp.make_pool(st.POOL, cs.TUNED, rp.DEV_SEED, form, window=32), seq, chain=True
        )
        for form in ("carried", "settled")
    }


def _replay_records(fixture, tap):
    """The fixture through the pool arm in every form and through every transport arm (Carried and
    Settled), at GOLDEN_WINDOW; one golden record per replay."""
    record = {}
    for form in rp.FORMS:
        spec = ps._spec(st.POOL, cs.TUNED, form, window=GOLDEN_WINDOW)
        record[f"pool {form}"] = _core(md.replay(fixture, "pool", pool=spec))
    for (
        form
    ) in ps.FIT_FORMS:  # window 0, the registered Settled-from-first-percept intervention (2.6)
        spec = ps._spec(st.POOL, cs.TUNED, form, window=0)
        record[f"pool {form}@0"] = _core(md.replay(fixture, "pool", pool=spec))
    for arm in md.TRANSPORT_ARMS:
        for form in ps.FIT_FORMS:
            spec = ps._spec(st.POOL, cs.TUNED, form, window=GOLDEN_WINDOW)
            record[f"{arm} {form}"] = _core(md.replay(fixture, arm, pool=spec, side=tap))
    return record


def golden(out=None):
    """The golden digest dict (2.11): see the module docstring. With `out` the report stage's pure
    functions are recomputed over OUT's saved summaries (a set that cannot be assembled is a
    ValueError, never a silent None); without it the "report" entry is None. The digest names the pool's
    host (st.HOST), as the golden check at 5761f3a7 did for the surviving host; an OUT whose migrate
    summary names another host is a ValueError."""
    if out is not None:
        try:
            st.use_of(out)
        except ps.Halt as halt:
            raise ValueError(f"the golden check cannot run OUT's host: {halt}") from halt
    host = st.HOST
    fixture = _fixture()
    tap = md.replay(fixture, "a", tap=GOLDEN_B)["tap"]
    body = {
        "scope": {
            "hosts": [host],
            "window": GOLDEN_WINDOW,
            "b": GOLDEN_B,
            "percepts": GOLDEN_PERCEPTS,
            "arms": list(GOLDEN_ARMS),
        },
        "hand": {host: digest_of(_hand_record())},
        "r_b3": {host: digest_of(_chain_record())},
        "arms": {arm: digest_of(_core(md.replay(fixture, arm))) for arm in GOLDEN_ARMS},
        "replays": {
            host: {
                name: digest_of(record) for name, record in _replay_records(fixture, tap).items()
            }
        },
        "report": None,
    }
    if out is not None:
        report = assemble(load_summaries(out))
        body["report"] = {"report": digest_of(report), "markdown": digest_of(render(report))}
    return {**body, "digest": digest_of(body)}


def differences(first, second, prefix=""):
    """The paths at which two golden digests (nested dicts of strings) differ, sorted."""
    if not (isinstance(first, dict) and isinstance(second, dict)):
        return [] if first == second else [prefix or "<root>"]
    found = []
    for key in sorted(set(first) | set(second), key=str):
        found += differences(first.get(key), second.get(key), f"{prefix}/{key}")
    return found


@contextlib.contextmanager
def leak_nudged():
    """The golden check's RED: while it holds every pool is built with its leak moved ONE ULP of the
    pool's own arithmetic (float32: the membrane decays in float32, where a float64 ulp is invisible).
    The pool constructor is wrapped, never the configuration: `canonical` would refuse an off-grid
    configuration and the mutant would never run."""
    original = rp.LsmPool.__init__

    def nudged(self, pre, w, config, *args, **kwargs):
        moved = float(np.nextafter(np.float32(config["lambda"]), np.float32(2.0)))
        original(self, pre, w, dict(config, **{"lambda": moved}), *args, **kwargs)

    rp.LsmPool.__init__ = nudged
    try:
        yield
    finally:
        rp.LsmPool.__init__ = original


# ---- the self-check (no recording, no Brain, no live seed) -----------------------------------------


# Sites the duplicate-capability hook flagged on the self-check helpers (read; none checks a campaign report or golden):
# distinct-from: engine/learn_daemon.py::_selfcheck -- a daemon's self-check that must never start its drive loop
# distinct-from: faculties/access_consciousness.py::_post -- the one submit site a faculty's self-check uses on a shared bus
# distinct-from: gates/selfcheck_cache.py::SelfcheckCache.describe -- the composition of one module's selfcheck key
# distinct-from: infra/self_mod_harness.py::SelfModHarness._run_selfchecks -- runs run_selfchecks.py inside a sandbox
# distinct-from: faculties/self_model.py::_band -- the one-hot bucket present in a reserved band of channels
def _check_provenance():
    ok = {stage: {"status": "COMPLETED", "head": "h1", "dirty": False} for stage in STAGES_READ}
    assert provenance_problems(ok) == []
    assert provenance_problems({k: v for k, v in ok.items() if k != "primary"}) == [
        "primary: no summary"
    ]
    halted = {**ok, "controls": {**ok["controls"], "status": "HALTED"}}
    assert provenance_problems(halted) == ["controls: HALTED, not COMPLETED"]
    dirty = {**ok, "primary": {**ok["primary"], "dirty": True}}
    assert provenance_problems(dirty) == ["primary: written on a dirty tree"]
    moved = {**ok, "migrate": {**ok["migrate"], "head": "h2"}}
    assert provenance_problems(moved) == ["more than one head: ['h1', 'h2']"]
    for bad in (halted, dirty, moved, {k: v for k, v in ok.items() if k != "transport"}):
        try:
            assemble(bad)
        except ValueError:
            continue
        raise AssertionError("assemble accepted a summary set that is not one clean campaign")
    return ok


def _check_digests():
    assert digest_of({"a": 1, "b": [1, 2]}) == digest_of({"b": [1, 2], "a": 1})
    assert digest_of({"a": 1}) != digest_of({"a": 2})
    assert differences({"a": {"b": "1", "c": "2"}}, {"a": {"b": "1", "c": "3"}}) == ["/a/c"]
    assert differences({"a": "1"}, {"a": "1"}) == [] and differences({"a": "1"}, {}) == ["/a"]


def _check_golden():
    """The golden digest, twice at one head (equal: it is deterministic), and under a one-ulp change to
    the pool's leak (it must move, and only where a pool is read: the arms with no pool do not move)."""
    started = time.perf_counter()
    base = golden()
    again = golden()
    assert again == base, "the golden is not deterministic"
    host = st.HOST
    with leak_nudged():
        nudged = golden()
    assert nudged["arms"] == base["arms"], "the leak red moved an arm with no pool"
    assert nudged["digest"] != base["digest"]
    moved = [part for part in ("hand", "r_b3") if nudged[part][host] != base[part][host]] + [
        f"replays/{name}"
        for name, found in nudged["replays"][host].items()
        if found != base["replays"][host][name]
    ]
    assert "r_b3" in moved and len(moved) > 2, moved
    return base, moved, time.perf_counter() - started


def self_check():
    _check_provenance()
    _check_digests()
    base, moved, seconds = _check_golden()
    print(
        "carried_state_report self-check OK: provenance (a missing, halted, dirty or second-head summary "
        "refused), digests, the golden deterministic over "
        f"{len(base['replays'][st.HOST])} pool replays and {len(base['arms'])} arms; a one-ulp "
        f"leak moved {len(moved)} records ({seconds:.1f} s)"
    )


if __name__ == "__main__":
    self_check()
