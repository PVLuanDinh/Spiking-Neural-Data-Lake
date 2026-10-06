r"""G1(c) / T109 part (c): the campaign driver's pool stages, part 2b -- controls, transport, separation.

The frozen text is specs/022-unified-brain-arc/evidence/g1c-carried-state.md (2.6, 2.7, 2.8, 2.11, 2.13).
These stages run through `pool_selection.run_step` like every other (`carried_state_driver.run_stage`);
the helpers they share with migrate, causality and primary are `carried_state_stages`'s.

  controls    (2.6) (a) the shuffle's POOL positive control on the rb6 family (6000 and its history-free
              twin) on the surviving host: D_task (Carried - Settled) >= 0.05 unshuffled and < 0.05
              shuffled, the A/A twin under 0.05 both ways, Settled invariance with its Carried red,
              joined with the instrument stage's pool-free control into `sighted`; (b) the shuffle on the
              live seeds: K = 4 shuffled streams per seed written to OUT/controls, Carried and Settled
              replays on each (stream_key / source_key cells, one R-b7 table per k), D_shuf per seed the
              mean over K; (c) window 0: Settled@0 per seed (D0 = D + W), the scored rows at or before the
              boundary, and Carried@0 asserted identical to the primary's Carried (a difference halts).
  transport   (2.7) a tap job (arm (a), tap = b) per seed; pool-copy, pool-act and pool-swap x {carried,
              settled} on the surviving host with the tap as `side`; per form D_t; the forbidden-map
              request red as one replay that must RAISE; padding fractions and raw norms from the bindings;
              each arm's input-response statistics (2.3) from its own encoder and input.
  separation  (2.8) when construct's verdict is True: G2(ii)'s XOR and twin streams per live seed through
              arm (a), Carried and Settled; decoder accuracy, per-text accuracy, distinct Settled phi
              rows and the reading rules. A failed structural stream check (before any replay) or an arm
              (a) outside its ceiling or floor is a wrong STREAM, a harness defect: it halts (2.11). Each
              stream's input-response statistics (2.3) are filed per stream and seed.

A red that does not fire (the Carried red of Settled invariance included), a missing or TIMEOUT cell, a
wrong separation stream and a Carried@0 that differs from Carried halt (2.11); a blind control reads as
`sighted: False` and a pool that cannot read the code ("pool blind to the code") as a reading.
Deterministic. Names avoid the verbs the gradient-claim census reads.
"""

import math
import statistics

import numpy as np

from snn_data_lake.instruments import carried_state_campaign as cs
from snn_data_lake.instruments import carried_state_stages as st
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.reservoir import recurrent_pool as rp

# Sites the duplicate-capability hook flagged on this module's docstring (read; none runs this campaign's stages):
# distinct-from: instruments/pool_selection.py -- G1(b)'s ten steps; the stages here CALL its step protocol, jobs and R-b7 tables (run_step, run_jobs, _tabulate) through carried_state_stages' helpers and add G1(c)'s own stages, with no second job runner or table

# ---- controls (2.6) ------------------------------------------------------------------------------


# Sites the duplicate-capability hook flagged on the shuffle's helpers (read; none shuffles a recording or reads a control):
# distinct-from: instruments/carried_state_campaign.py::shuffle_blocks -- the transform itself, CALLED here and held to check_shuffle_transform; this wraps a halt around it
# distinct-from: instruments/everyday_battery.py::Control -- one positive-control question of a battery over the brain's answers
# distinct-from: faculties/recursive_tom.py::_t082_s2 -- one per-seed check of another task's world predictor
# distinct-from: gates/amdahl_gate.py::_padded_step -- a stand-in for LIFNeuron.step with a known extra cost; no shuffle, no table
# distinct-from: gates/selfcheck_cache.py::self_check -- the P8 reds of the selfcheck cache's keys
# distinct-from: instruments/memory_diagnostic.py::_PoolArm -- the pool arm a replay runs; these stages file its replays, they never build a pool
def _shuffled(stream, k):
    """Shuffle k of `stream`, held to the transform's own check; a refusal or a failed check halts (a
    harness defect: a recording that cannot be shuffled is never silently left out)."""
    try:
        shuffled = cs.shuffle_blocks(stream, k)
    except ValueError as error:
        raise ps.Halt(
            "shuffle refused", {"seed": stream.get("seed"), "k": k, "why": str(error)}
        ) from error
    checked = cs.check_shuffle_transform(stream, shuffled)
    if not checked["ok"]:
        raise ps.Halt(
            "shuffle transform", {"seed": stream.get("seed"), "k": k, "reasons": checked["reasons"]}
        )
    return shuffled


def _gap(table, seed, window, learner="readout"):
    """accuracy(Carried) - accuracy(Settled) of one seed's cells in an OK table."""
    return st.accuracy(table, st.key("carried", seed, window=window), learner) - st.accuracy(
        table, st.key("settled", seed, window=window), learner
    )


def _invariance_row(result):
    return {
        "ok": result["ok"],
        "blocks": len(result["per_block"]),
        "excluded": result["excluded"],
        "multiset_equal": sum(row["multiset_equal"] for row in result["per_block"]),
        "sequence_differs": sum(row["sequence_differs"] for row in result["per_block"]),
    }


# 2.6's RED of Settled invariance: the same check on Carried, whose history reaches phi, must find the
# multisets DIFFERENT in a nonempty set of tested blocks. A result with no tested block, or whose
# multisets are all equal (the shuffle moved nothing), proves nothing: that red did NOT fire (2.11).
def history_reaches_phi(result):
    blocks = result["per_block"]
    return bool(blocks) and any(not row["multiset_equal"] for row in blocks)


def _pool_control(out, folder, window, seed):
    """2.6's pool positive control, A/A and Settled invariance on the rb6 family (the streams the
    instrument stage wrote) on the surviving host: Carried and Settled on each stream plain and under
    the K shuffles."""
    families = ps._rb6_families(ps._folder(out, "instrument"), seed)
    plain = {name: md._load(path) for name, path in families.items()}
    groups, labels = [], []
    for name, path in families.items():
        for label, k in (("plain", None), *((f"shuf{i}", i) for i in range(cs.SHUFFLE_K))):
            stream = plain[name] if k is None else _shuffled(plain[name], k)
            where = path if k is None else folder / f"{name}-{label}-stream.json.gz"
            if k is not None:
                md._save(where, stream)
            ident = f"{name}/{label}"
            cells = [
                st.pool_cell(
                    where,
                    folder / f"{name}-{label}-{form}.json.gz",
                    seed,
                    form,
                    window=window,
                    stream_key=ident,
                )
                for form in ps.FIT_FORMS
            ]
            groups.append((cells, (seed,), {ident: stream}))
            labels.append((name, label))
    tables = dict(zip(labels, st.run_cells(groups), strict=True))
    for label, table in tables.items():
        st.need_ok(table, f"controls: rb6 {label}")
    gap = {label: _gap(table, seed, window) for label, table in tables.items()}

    def read(name):
        shuffled = [gap[(name, f"shuf{k}")] for k in range(cs.SHUFFLE_K)]
        return {"unshuffled": gap[(name, "plain")], "shuffled": statistics.fmean(shuffled)}

    def run(label, form):
        cell = tables[label][st.key(form, seed, window=window)]
        return md._load(cell["job"]["out"])

    def invariance(form):
        return [
            cs.settled_invariance(
                run(("rb6", "plain"), form),
                run(("rb6", f"shuf{k}"), form),
                plain["rb6"],
                boundary=window,
            )
            for k in range(cs.SHUFFLE_K)
        ]

    bar = md.RB6_BAR
    positive, twin = read("rb6"), read("rb6red")
    settled, carried = invariance("settled"), invariance("carried")
    return {
        "seed": seed,
        "bar": bar,
        "positive": {
            **positive,
            "sighted": bool(positive["unshuffled"] >= bar > positive["shuffled"]),
        },
        "aa": {**twin, "held": bool(max(twin.values()) < bar)},
        "invariance": {
            "settled": [_invariance_row(r) for r in settled],
            "carried_red": [_invariance_row(r) for r in carried],
            "settled_ok": all(r["ok"] for r in settled),
            "red_fired": all(history_reaches_phi(r) for r in carried),
        },
    }


def _live_shuffles(out, folder, live, window):
    """K tables (one R-b7 table per shuffle k: a seed's cell key does not name its stream): Carried and
    Settled replays of every live seed's k-th shuffled stream; returns ([accuracies] over k, {k: the
    cells' per-segment diagnostic tables} for the report, 2.3)."""
    paths = {seed: ps._recording(out, seed) for seed in live}
    per_k, tables = [], {}
    for k in range(cs.SHUFFLE_K):
        streams, cells = {}, []
        for seed in live:
            ident, where = f"{seed}/{k}", folder / f"{seed}-shuf{k}-stream.json.gz"
            streams[ident] = _shuffled(md._load(paths[seed]), k)
            md._save(where, streams[ident])
            cells += [
                st.pool_cell(
                    where,
                    folder / f"{seed}-shuf{k}-{form}.json.gz",
                    seed,
                    form,
                    window=window,
                    stream_key=ident,
                )
                for form in ps.FIT_FORMS
            ]
        (table,) = st.run_cells([(cells, live, streams)])
        st.need_ok(table, f"controls: shuffle {k}")
        per_k.append(st.accuracies(table, live, ps.FIT_FORMS, window=window))
        tables[str(k)] = st.variant_tables(table, live, ps.FIT_FORMS, window=window)
    return per_k, tables


IDENTITY_KEYS = ("phi", "predictions", "fallbacks", "fingerprint", "refits")


def _carried_differences(first, second):
    """What differs between two Carried replays: the grower's keys, the readout's and the pool's chain."""
    found = [name for name in IDENTITY_KEYS if first[name] != second[name]]
    found += [
        f"readout {name}"
        for name in ("predictions", "fingerprint")
        if first["readout"][name] != second["readout"][name]
    ]
    found += [
        f"pool {name}"
        for name in ("state_chain", "state")
        if first["pool"][name] != second["pool"][name]
    ]
    return found


def scored_rows_through(stream, boundary):
    """The scored rows (events in a scored segment) at or before delivery `boundary`: the rows window 0
    changes beyond the readout's training rows (2.6; the window follows the pool's delivery count)."""
    return sum(
        1
        for index, event in enumerate(stream["events"])
        if index + 1 <= boundary and event.get("segment") in md.SCORED_SEGMENTS
    )


def _window_zero(out, folder, live, window, primary):
    """Settled@0 and Carried@0 on every live seed: D0 = accuracy(Carried) - accuracy(Settled@0), W = D0 - D,
    the scored rows at or before the boundary, and Carried@0 held identical to the primary's Carried."""
    paths = {seed: ps._recording(out, seed) for seed in live}
    table = st.pool_table(out, folder, live, ps.FIT_FORMS, paths, window=0)
    st.need_ok(table, "controls: window 0")
    acc = st.accuracies(table, live, ps.FIT_FORMS, window=0)
    differs = {}
    for seed in live:
        first = md._load(ps._folder(out, "primary") / f"{st.POOL}-00-carried-{seed}.json.gz")
        second = md._load(table[st.key("carried", seed, window=0)]["job"]["out"])
        found = _carried_differences(first, second)
        if found:
            differs[str(seed)] = found
    if differs:
        raise ps.Halt("window 0: Carried@0 is not the primary's Carried", differs)
    d0 = st.gaps(acc)
    return {
        "accuracy": acc,
        "cells": st.variant_tables(table, live, ps.FIT_FORMS, window=0),
        "d0": d0,
        "w": {
            learner: {seed: d0[learner][seed] - primary["d"][learner][seed] for seed in d0[learner]}
            for learner in d0
        },
        "scored_rows_through_boundary": {
            str(seed): scored_rows_through(md._load(paths[seed]), window) for seed in live
        },
        "carried_identical": True,
    }


def stage_controls(out, *, scale=None, rb6_seed=None):
    """The shuffle's pool controls, the shuffle on the live seeds and window 0 (2.6), all on the
    surviving host. `rb6_seed` is the seed bundle's (6000 live, 6001 reserve; only a test names
    another). A Carried red that does not fire (no tested block, or no multiset moved) halts (2.11)."""
    sc = st.settings(scale)
    window, host = sc["window"], st.use_of(out)
    record, primary = ps._summary(out, "record"), ps._summary(out, "primary")
    instrument = ps._summary(out, "instrument")
    rb6_seed = st.bundle_of(out).rb6 if rb6_seed is None else rb6_seed
    if instrument.get("rb6_seed", ps.RB6_SEED) != rb6_seed:
        raise ps.Halt("controls", f"the instrument stage ran rb6 on {instrument.get('rb6_seed')}")
    live = tuple(record["live_seeds"])
    folder = ps._folder(out, "controls")
    control = _pool_control(out, ps._folder(out, "controls/rb6"), window, rb6_seed)
    if not control["invariance"]["red_fired"]:  # a red that does not fire halts (2.11)
        detail = {"Settled invariance, Carried red": control["invariance"]["carried_red"]}
        raise ps.Halt("a red that does not fire", detail)
    per_k, shuffle_cells = _live_shuffles(out, ps._folder(out, "controls/shuffle"), live, window)
    d_k = [st.gaps(acc) for acc in per_k]
    d_shuf = {
        learner: {str(seed): statistics.fmean(d[learner][str(seed)] for d in d_k) for seed in live}
        for learner in st.LEARNERS
    }
    zero = _window_zero(out, ps._folder(out, "controls/window0"), live, window, primary)
    pool_free = bool(instrument["shuffle_control"]["sighted"] and instrument["shuffle_aa"]["held"])
    halves = {
        "pool_free": pool_free,
        "pool_positive": control["positive"]["sighted"],
        "aa": control["aa"]["held"],
        "settled_invariance": control["invariance"]["settled_ok"],
        "carried_red": control["invariance"]["red_fired"],
    }
    return {
        "host": host,
        "window": window,
        "scale": sc,
        "shortened": sc != st.REAL_SCALE,
        "folder": str(folder),
        "sighted": all(halves.values()),
        "halves": halves,
        "control": control,
        "d_shuf": d_shuf,
        "d_shuf_k": {str(seed): [d["readout"][str(seed)] for d in d_k] for seed in live},
        # 2.3's input-response statistics beside these two variants are the PRIMARY's (re-panel R3), so none
        # is filed here: both replay the recording's own seed (the same W_in); a shuffle only permutes whole
        # percepts inside blocks (the multiset of x is kept: check_shuffle_transform, held per stream by
        # `_shuffled`); and the window is read by the pool's adaptation, never by the drive u = W x. The
        # per-unit drive statistics are those of the same numbers -- window 0's bit for bit, a shuffle's
        # up to the last bits of a reordered float sum (tests/test_carried_state_stages.py shows both
        # through a fresh replay's received drive).
        "cells": {"shuffle": shuffle_cells, "window0": zero["cells"]},
        "window0": zero,
    }


# ---- transport (2.7) -----------------------------------------------------------------------------

HERE = "snn_data_lake.instruments.carried_state_controls:"


# Sites the duplicate-capability hook flagged on the transport stage (read; none files a transport arm's replays):
# distinct-from: instruments/memory_diagnostic.py::check_transport_channels -- the channel check on ONE fixture stream and its reds; the call job below is the campaign's one forbidden-request replay on a live seed's own tap
def forbidden_request(stream, side, form, window=rp.WINDOW):
    """RED (2.7, 2.10; a "call" target): pool-copy keeps its permitted declaration and REQUESTS "w_live"
    while that value is present in the replay's values; the replay must RAISE "forbidden feature
    channel". The permitted replay runs first (the control)."""
    data = md._load(stream)
    spec = ps._spec(st.POOL, cs.TUNED, form, window=window)
    tap = ps._side(side)
    md.replay(data, "pool-copy", pool=spec, side=tap)
    message = ""
    with md._w_live_mutant():
        try:
            md.replay(data, "pool-copy", pool=spec, side=tap)
        except ValueError as error:
            message = str(error)
    return {
        "control_ran": True,
        "raised": "forbidden feature channel" in message,
        "message": message,
    }


def _transport_cell(path, out_path, seed, arm, form, side, encoder, *, window):
    """One transport arm's replay as a table cell, filed under its arm, the form of its transport and
    the crc of the encoder it must end with (None for the activity form, which swaps nothing)."""
    spec = ps._spec(st.POOL, cs.TUNED, form, window=window)
    filed = ps._bound(
        seed, spec, arm=arm, encoder=encoder, transport={"form": md.TRANSPORT_FORM[arm]}
    )
    job = ps._replay_job(path, arm, out_path, pool=spec, side=side)
    return {**filed, "index": None, "status": None, "job": job, "digest": None}


def _arm_key(arm, form, seed, encoders, *, window):
    return st.key(
        form,
        seed,
        window=window,
        arm=arm,
        transport={"form": md.TRANSPORT_FORM[arm]},
        encoder=encoders.get((seed, arm)),
    )


# Sites the duplicate-capability hook flagged on `_transport_response` (read; none computes a drive statistic, and the one that does, pool_selection.input_response, is CALLED below, never re-implemented):
# distinct-from: faculties/forward_model.py::ForwardModel.predict -- predicts the next sensed state from an efference copy; no pool, no encoder drive
# distinct-from: faculties/agency_monitor.py::_t079_score -- reads one joint's controllability from one arm of one seed's record
# distinct-from: faculties/ring_attractor.py::ProprioceptiveEstimator -- a ring-attractor dead-reckoner of a joint angle from commanded deltas
# distinct-from: faculties/task_set.py::TaskSetLayer -- a layer of rule mappings with a working-memory pool and a conflict monitor
# distinct-from: instruments/attention_schema.py::g4_rule -- another campaign's G4 branches over per-seed decodes of a vector; no pool, no encoder
# distinct-from: instruments/everyday_battery.py::run_seed -- one seed's reading of a scratch brain's replies to questions
# distinct-from: readout/snn_moe_classifier.py::_new_rng -- seeds a numpy Generator from the global random stream; this draws nothing
def _transport_response(arm, stream, tap, matrix, record):
    """2.3's input-response statistics of one transport arm on one seed, from the arm's ACTUAL input and
    encoder through pool_selection's one helper. pool-act drives the seed's random encoder with the tap's
    grower activity, zero-padded as the arm pads it; pool-copy and pool-swap drive it with x, through the
    random encoder up to the event the replay recorded (`copied_at_event`, `swapped_at_event`) and
    through the transported or the fresh `matrix` from it. A replay that never installed one (a None
    event) was refused by R-b7's encoder binding before this runs; were it to arrive, it fails loud."""
    if arm == "pool-act":
        padded = [md._padded_activity(activity) for activity in tap["activity"]]
        return ps.input_response(st.POOL, cs.TUNED, stream, inputs=padded)
    first = record["transport"]["copied_at_event" if arm == "pool-copy" else "swapped_at_event"]
    steps = [(0, matrix)] if first == 0 else [(0, rp.w_in(stream["seed"])), (first, matrix)]
    return ps.input_response(st.POOL, cs.TUNED, stream, encoders=steps)


def stage_transport(out, *, scale=None):
    """The transport family (2.7, secondary): tap, pool-copy, pool-act and pool-swap on the surviving host."""
    sc = st.settings(scale)
    window, b, host = sc["window"], sc["b"], st.use_of(out)
    live = tuple(ps._summary(out, "record")["live_seeds"])
    folder = ps._folder(out, "transport")
    paths = {seed: ps._recording(out, seed) for seed in live}
    streams = ps._streams(paths)
    taps = {
        seed: ps._replay_job(paths[seed], "a", folder / f"tap-{seed}.json.gz", tap=b)
        for seed in live
    }
    statuses = dict(zip(taps, ps.run_jobs(list(taps.values())), strict=True))
    late = [seed for seed, status in statuses.items() if status["status"] != "OK"]
    if late:
        raise ps.Halt("transport TIMEOUT", {"tap": late})
    side = {seed: ps._checked(job, streams[seed])["tap"] for seed, job in taps.items()}
    first = live[0]
    request = {
        "kind": "call",
        "target": HERE + "forbidden_request",
        "args": {
            "stream": str(paths[first]),
            "side": taps[first]["out"],
            "form": "carried",
            "window": window,
        },
        "out": str(folder / "forbidden-request.json"),
    }
    (status,) = ps.run_jobs([request])
    if status["status"] != "OK":
        raise ps.Halt("transport TIMEOUT", {"forbidden request": status["status"]})
    red = md._load(request["out"])
    if not red["raised"]:
        raise ps.Halt("a red that does not fire", {"transport forbidden-map request": red})
    matrices, encoders, cells = {}, {}, []
    for seed in live:
        random = rp.w_in(seed)
        block = np.asarray(side[seed]["w_at_b"]["W"], dtype=np.float64)
        matrices[(seed, "pool-copy")] = rp.transported_encoder(random, block)
        matrices[(seed, "pool-swap")] = rp.fresh_encoder(seed, random)
        for arm in ("pool-copy", "pool-swap"):
            encoders[(seed, arm)] = md._crc(matrices[(seed, arm)])
        for arm in md.TRANSPORT_ARMS:
            for form in ps.FIT_FORMS:
                name = folder / f"{arm}-{form}-{seed}.json.gz"
                cells.append(
                    _transport_cell(
                        paths[seed],
                        name,
                        seed,
                        arm,
                        form,
                        taps[seed]["out"],
                        encoders.get((seed, arm)),
                        window=window,
                    )
                )
    (table,) = st.run_cells([(cells, live, streams)])
    st.need_ok(table, "transport")
    arms = {}
    for arm in md.TRANSPORT_ARMS:
        acc = {
            learner: {
                form: {
                    str(seed): st.accuracy(
                        table,
                        _arm_key(arm, form, seed, encoders, window=window),
                        learner,
                    )
                    for seed in live
                }
                for form in ps.FIT_FORMS
            }
            for learner in st.LEARNERS
        }
        records = {}
        for seed in live:
            binding = table[_arm_key(arm, "carried", seed, encoders, window=window)]["digest"][
                "pool"
            ]
            if arm == "pool-act" and binding["encoder"] is not None:
                raise ps.Halt("transport", f"pool-act swapped an encoder on seed {seed}")
            records[str(seed)] = {"transport": binding["transport"], "encoder": binding["encoder"]}
        cells = {  # each seed's own encoder crc is in its cell key, so the tables are read per seed
            form: {
                str(seed): ps._report(table[_arm_key(arm, form, seed, encoders, window=window)])
                for seed in live
            }
            for form in ps.FIT_FORMS
        }
        # 2.3 (re-panel R3): the input response of THIS arm's own encoder and input, per seed
        response = {
            str(seed): _transport_response(
                arm, streams[seed], side[seed], matrices.get((seed, arm)), records[str(seed)]
            )
            for seed in live
        }
        arms[arm] = {
            "accuracy": acc,
            "d": st.gaps(acc),
            "records": records,
            "cells": cells,
            "input_response": response,
        }
    return {
        "host": host,
        "window": window,
        "b": b,
        "scale": sc,
        "shortened": sc != st.REAL_SCALE,
        "live_seeds": list(live),
        "taps": {
            str(seed): {key: side[seed]["w_at_b"][key] for key in ("event", "observed", "crc_W")}
            | {"L": len(side[seed]["w_at_b"]["W"])}
            for seed in live
        },
        "forbidden_request": red,
        "arms": arms,
    }


# ---- separation (2.8) ----------------------------------------------------------------------------

TWIN_FLOOR = 0.75  # the affine ceiling on the XOR set, and the floor a form must clear on the twin (2.8's amendment)
CEILING_SD = (
    4  # arm (a)'s XOR reading must stay under 0.75 + this many binomial sd at the scored n (2.8)
)


def read_separation(means, scored):
    """2.8's reading rules over mean decoder accuracies `means` = {"xor" | "twin": {"a" | "carried" |
    "settled": mean}} at `scored` scored events per stream: the affine ceiling 0.75 + 4 binomial sd
    (arm (a) on XOR at or above it, or on the twin below 0.75, means the STREAM is wrong), and a pool
    form's XOR reading is "readable" only if its twin mean reaches 0.75, else "pool blind to the code"."""
    ceiling = TWIN_FLOOR + CEILING_SD * math.sqrt(TWIN_FLOOR * (1 - TWIN_FLOOR) / scored)
    wrong = {
        "a_xor_at_or_above_ceiling": bool(means["xor"]["a"] >= ceiling),
        "a_twin_below_floor": bool(means["twin"]["a"] < TWIN_FLOOR),
    }
    forms = {}
    for form in ps.FIT_FORMS:
        readable = bool(means["twin"][form] >= TWIN_FLOOR)
        forms[form] = {
            "xor": means["xor"][form],
            "twin": means["twin"][form],
            "readable": readable,
            "reading": "readable" if readable else "pool blind to the code",
        }
    return {
        "ceiling": ceiling,
        "floor": TWIN_FLOOR,
        "scored": scored,
        "stream_wrong": wrong,
        "stream_ok": not any(wrong.values()),
        "arm_a": {"xor": means["xor"]["a"], "twin": means["twin"]["a"]},
        "forms": forms,
    }


def _per_text(stream, predictions):
    """Accuracy per previous text over the scored events (the decoder predicts c_t from phi_{t-1}, so each
    label answers the PREVIOUS presentation): {"uv": [hits, n]} with u = dog precedes cat, v = sky precedes sun."""
    references = {uv: im._scaled(im.TEXTS[uv]) for uv in im.UV}
    texts = [im._decode(event["x"], references) for event in stream["events"]]
    tally = {f"{u}{v}": [0, 0] for u, v in im.UV}
    for index, event in enumerate(stream["events"]):
        previous = texts[index - 1] if index else None
        if event.get("segment") in md.SCORED_SEGMENTS and previous is not None:
            row = tally[f"{previous[0]}{previous[1]}"]
            row[0] += int(predictions[index] == event["c"])
            row[1] += 1
    return {text: (hits / n if n else None) for text, (hits, n) in tally.items()}


def stage_separation(out, *, scale=None, build=im.stream):
    """The separation reading (2.8): G2(ii)'s XOR and twin streams per live seed through arm (a), Carried
    and Settled on the surviving host, when construct's verdict is True; otherwise {"run": False, ...}.
    `build` is inseparable_mapping.stream (only a test names a shorter one)."""
    construct = ps._summary(out, "construct")
    if not construct.get("verdict"):
        reason = construct.get("reason") or "the construction check did not read True"
        return {"run": False, "reason": reason, "item_3": "NOT RUN"}
    sc = st.settings(scale)
    window, host = sc["window"], st.use_of(out)
    live = tuple(ps._summary(out, "record")["live_seeds"])
    folder = ps._folder(out, "separation")
    labellings = {"xor": im.label_xor, "twin": im.label_twin}
    streams = {name: {} for name in labellings}
    for name, labelling in labellings.items():
        for seed in live:
            streams[name][seed] = build(seed, labelling)
            md._save(folder / f"{name}-{seed}-stream.json.gz", streams[name][seed])
    failed = {}  # the structural stream checks hold BEFORE any replay: a wrong stream is a harness defect
    for seed in live:
        found = im.stream_checks(seed, xor=streams["xor"][seed], twin=streams["twin"][seed])
        if not found["ok"]:
            failed[str(seed)] = sorted(name for name, held in found["checks"].items() if not held)
    if failed:
        raise ps.Halt("separation: stream checks", failed)
    groups = []
    for name in labellings:
        cells = []
        for seed in live:
            where = folder / f"{name}-{seed}-stream.json.gz"
            cells.append(ps._cell(where, folder / f"{name}-a-{seed}.json.gz", seed, arm="a"))
            cells += [
                st.pool_cell(
                    where,
                    folder / f"{name}-{form}-{seed}.json.gz",
                    seed,
                    form,
                    window=window,
                )
                for form in ps.FIT_FORMS
            ]
        groups.append((cells, live, streams[name]))
    tables = dict(zip(labellings, st.run_cells(groups), strict=True))
    for name, table in tables.items():
        st.need_ok(table, f"separation: {name}")
    counts = {
        sum(e.get("segment") in md.SCORED_SEGMENTS for e in streams[name][seed]["events"])
        for name in labellings
        for seed in live
    }
    if len(counts) != 1:
        raise ps.Halt("separation", f"the streams hold different scored counts {sorted(counts)}")
    arms = ("a", *ps.FIT_FORMS)

    def cell(name, arm, seed):
        if arm == "a":
            return tables[name][("a", None, None, None, seed, False)]
        return tables[name][st.key(arm, seed, window=window)]

    acc = {
        name: {
            arm: {
                str(seed): ps.recording_accuracy(cell(name, arm, seed)["digest"]["readout"])
                for seed in live
            }
            for arm in arms
        }
        for name in labellings
    }
    per_text = {
        name: {
            arm: {
                str(seed): _per_text(
                    streams[name][seed],
                    md._load(cell(name, arm, seed)["job"]["out"])["readout"]["predictions"],
                )
                for seed in live
            }
            for arm in arms
        }
        for name in labellings
    }
    means = {
        name: {arm: statistics.fmean(acc[name][arm].values()) for arm in arms}
        for name in labellings
    }
    reading = read_separation(means, counts.pop())
    if not reading[
        "stream_ok"
    ]:  # arm (a) outside its ceiling or floor: the stream is wrong (2.8, 2.11)
        raise ps.Halt("separation: the stream is wrong", reading["stream_wrong"])
    return {
        "run": True,
        "item_3": "RUN",
        "host": host,
        "window": window,
        "scale": sc,
        "shortened": sc != st.REAL_SCALE,
        "live_seeds": list(live),
        "stream_checks": {str(seed): True for seed in live},  # every check held, or it halted
        "accuracy": acc,
        "mean": means,
        "per_text": per_text,
        "cells": {
            name: {
                arm: {str(seed): ps._report(cell(name, arm, seed)) for seed in live} for arm in arms
            }
            for name in labellings
        },
        # 2.3 (re-panel R3): the input response of each OTHER stream through the seed's own encoder
        # (arm (a) has no pool encoder; Carried and Settled see the same drive)
        "input_response": {
            name: {
                str(seed): ps.input_response(st.POOL, cs.TUNED, streams[name][seed])
                for seed in live
            }
            for name in labellings
        },
        "distinct_settled_rows": {
            name: {
                str(seed): cell(name, "settled", seed)["digest"]["pool"]["degeneracy"][
                    "distinct_rows"
                ]
                for seed in live
            }
            for name in labellings
        },
        "reading": reading,
    }


STAGE_FUNCTIONS = {
    "controls": stage_controls,
    "transport": stage_transport,
    "separation": stage_separation,
}


def self_check():
    """cwd-free, no recording, no Brain, no live seed: the separation reading rules on hand numbers and
    the scored-row count on the 48-event fixture."""
    assert tuple(STAGE_FUNCTIONS) == ("controls", "transport", "separation")
    hand = {
        "xor": {"a": 0.50, "carried": 0.65, "settled": 0.99},
        "twin": {"a": 1.0, "carried": 0.85, "settled": 0.60},
    }
    reading = read_separation(hand, 944)
    assert reading["stream_ok"] and abs(reading["ceiling"] - 0.8064) < 1e-4, reading
    assert reading["forms"]["carried"]["reading"] == "readable"
    assert reading["forms"]["settled"]["reading"] == "pool blind to the code"
    high = read_separation({**hand, "xor": {**hand["xor"], "a": 0.82}}, 944)
    assert high["stream_wrong"]["a_xor_at_or_above_ceiling"] and not high["stream_ok"]
    low = read_separation({**hand, "twin": {**hand["twin"], "a": 0.70}}, 944)
    assert low["stream_wrong"]["a_twin_below_floor"] and not low["stream_ok"]
    stream = md._synthetic_stream()
    assert scored_rows_through(stream, 24) == 0 and scored_rows_through(stream, 30) == 6
    assert scored_rows_through(stream, 0) == 0 and scored_rows_through(stream, 48) == 24
    print(
        "carried_state_controls self-check OK: separation reading rules (ceiling "
        f"{reading['ceiling']:.4f}, both stream-wrong reds, readable vs blind), scored rows through "
        "a boundary"
    )


if __name__ == "__main__":
    self_check()
