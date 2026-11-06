"""Exposed-fixture actual Brain integration; raw events remain in the isolated lake.

No future is read until the actual winner is frozen. A failed acquisition is
reported OPEN and never converted to a successful presentation result.
"""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

from snn_data_lake.engine.brain import make_brain
from snn_data_lake.growth.growth_loop import (
    BudgetValues, CanonicalGrowthAdapter, GrowthLoop, Ledger, make_admission,
)
from snn_data_lake.instruments import taught_problems as tp
from snn_data_lake.readout.posed_problem import engine_model_digest
from snn_data_lake.substrate import home


RECIPE = {
    "schema": "g9.engine.cycle.v1", "batch_size": 18, "candidate_pool": 1024,
    "input_sigma": 1.0, "threshold_min": 0.5, "threshold_max": 2.0,
    "window": 12, "t_post_g": 15.0, "trace_epochs": 20, "trace_eta": 0.05,
    "feed_norm": 1.0, "return_weight": 0.2, "return_delay": 0,
    "max_epochs": 400, "stable_epochs": 10, "selection": "all-active/no-prune",
    "feedback_gate": "corrective",
    "core_plasticity": "frozen",
}


def emit(kind, **fields):
    print("ENGINE " + json.dumps({"kind": kind, **fields}, sort_keys=True), flush=True)


def channel_probe(brain, adapter, *, feed_off=False, return_off=False):
    """Observe the real OFF read; ablate conductances, never plasticity masks."""
    from snn_data_lake.instruments import g5_readout_cells as cells
    from snn_data_lake.growth.grown_code import latencies, segment_psp
    from snn_data_lake.readout import posed_problem as posed

    snap = brain._g9_snapshot()
    digest = engine_model_digest(brain.brain._ixn, brain._g9_actors)
    actor = brain._g9_actors[adapter.problem.id]
    real_features, real_drives = cells.pool_features, posed.chat_grown_drives
    pool_trace, grown_trace = [], []

    def features(pool, text, *, feed=None):
        result = real_features(pool, text, feed=feed)
        pool_trace.append({"text": text, "feed_l2": math.hypot(*feed), "features": result.tolist()})
        return result

    def drives(learner, code):
        result = real_drives(learner, code)
        if learner is brain.brain._ixn:
            context, units, afferents = posed._chat_context(learner)
            grown_trace.append({
                "latencies": latencies(units, code, learner.tau, afferents=afferents),
                "psp": segment_psp(units, code, learner.tau, context["t_post_g"], afferents=afferents),
                "grown_drives": result,
                "afferent_l1": sum(abs(w) for rows in afferents.values() for _, w in rows),
            })
        return result

    try:
        if feed_off:
            actor.feed = [[0.0] * len(actor.units) for _ in actor.feed]
        if return_off:
            actor.port.weights[:] = [0.0] * len(actor.port.weights)
        cells.pool_features, posed.chat_grown_drives = features, drives
        reads = adapter.read_retention(adapter.problem.id)
        result = {"reads": reads, "correct": sum(a == b for ep in reads for a, b in ep),
                  "pool_trace": pool_trace, "grown_trace": grown_trace,
                  "return_weights_l2": math.hypot(*actor.port.weights)}
        assert len(pool_trace) == len(grown_trace) == adapter.problem.presentations(adapter.seed, "validation")
    finally:
        cells.pool_features, posed.chat_grown_drives = real_features, real_drives
        brain._g9_restore(snap)
    assert engine_model_digest(brain.brain._ixn, brain._g9_actors) == digest
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("old", "grow", "restore"))
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    home.require_isolated_home()
    t0 = time.monotonic()
    journal_only = os.environ.get("ENGINE_JOURNAL_ONLY") == "1"
    if args.phase == "restore":
        from snn_data_lake.instruments.joint_mask_search import RetentionSets

        original_read = RetentionSets._read
        original_load = home.load_state

        def forbidden(*a, **k):
            raise AssertionError("boot attempted to re-read heldout data")

        def load(name, *a, **k):
            return None if journal_only and name == "interaction_learner.json" else original_load(name, *a, **k)

        RetentionSets._read = staticmethod(forbidden)
        home.load_state = load
        try:
            brain = make_brain(seed=9900, persist=True)
        finally:
            RetentionSets._read = staticmethod(original_read)
            home.load_state = original_load
    else:
        brain = make_brain(seed=9900, persist=True)
    assert brain._r2_fault is None, brain._r2_fault
    ledger = Ledger(home.knowledge_path(), BudgetValues(tp.FIRST_BUDGET, None))
    append = ledger.record_event

    def observed(row):
        append(row)
        if row["kind"] != "engine_epoch" or row["epoch"] % 25 == 0 or row["stable_run"]:
            emit("progress", event=row)

    ledger.record_event = observed
    recipe = dict(RECIPE)
    if args.phase == "old":
        recipe.pop("core_plasticity")
    receipt = {"scope": "single exposed fixture engineering, task-conditioned, no-prune",
               "seed": 9900, "phase": args.phase, "recipe": recipe,
               "journal_only": journal_only}

    def core_state():
        attrs = brain.brain._ixn._snapshot()["state"]["tree"]["attrs"]
        return {key: attrs[key] for key in ("neurons", "stack")}
    try:
        pid = "aii" if args.phase == "old" else "ii.xor"
        core_before = core_state()
        adapter = CanonicalGrowthAdapter.prepare_cycle(brain, tp.PROBLEMS[pid], ledger=ledger, recipe=recipe)
        if args.phase != "restore":
            diagnoses = []
            diagnose = adapter.diagnose

            def observed_diagnosis():
                kind = diagnose()
                row = adapter.evidence()
                diagnoses.append(row)
                emit("diagnosis", problem_id=pid, units=len(adapter.segment), evidence=row)
                return kind

            adapter.diagnose = observed_diagnosis
            before = adapter.read_retention(pid)
            emit("before", problem_id=pid, reads=before)
            admission = make_admission(site=(0.5 / 14, 1.5 / 4), centre_um=(0.0, 0.0),
                                       density_per_mm2=2.7e4, seed=9900)
            loop = GrowthLoop(adapter, pid, ledger, adapter.source, RECIPE["batch_size"],
                              admission=admission, resolve_problem=lambda p: p in tp.PROBLEMS)
            rec = loop.run()
            receipt["diagnoses"] = diagnoses
            receipt["problem"] = {k: v for k, v in vars(rec).items() if k != "history"}
            receipt["validation"] = {pid: adapter.read_retention(pid)}
            for old in adapter._engine_binding["retention"]:
                receipt["validation"][old] = adapter.read_retention(old)
            emit("loop_end", **receipt["problem"])
            if args.phase == "grow" and rec.state == "SOLVED":
                assert core_state() == core_before, "protected core/stack changed during B"
                receipt["protected_core_unchanged"] = True
                # A planted old-feature lesion must fail feasibility while B
                # sees exactly the same decisions. Nothing is journaled/learned.
                from snn_data_lake.instruments import delayed_cue as dc
                snap = brain._g9_snapshot()
                digest = engine_model_digest(brain.brain._ixn, brain._g9_actors)
                try:
                    for neuron in brain.brain._ixn.neurons:
                        for bucket in dc.BUCKETS_AII.values():
                            neuron.w[bucket] = 0.0
                    current = adapter.read_retention(pid)
                    old = adapter.read_retention("aii")
                    feasible, _ = adapter.evaluate((True,) * adapter.n)
                    assert current == receipt["validation"][pid]
                    assert not adapter._correct(old) and not feasible
                    receipt["old_skill_lesion_refused"] = True
                    receipt["lesioned_old_correct"] = sum(a == b for ep in old for a, b in ep)
                finally:
                    brain._g9_restore(snap)
                assert engine_model_digest(brain.brain._ixn, brain._g9_actors) == digest
                # Existing lifecycle silencing, applied to the actual grown
                # segment; the old task remains readable via its own context.
                from snn_data_lake.growth.candidate_lifecycle import silence_units
                snap = brain._g9_snapshot()
                try:
                    actor = brain._g9_actors[pid]
                    actor.units = silence_units(actor.units, [u.unit_id for u in actor.units])
                    receipt["grown_off"] = {p: adapter.read_retention(p) for p in (pid, "aii")}
                finally:
                    brain._g9_restore(snap)
                assert engine_model_digest(brain.brain._ixn, brain._g9_actors) == digest
                receipt["channel_probes"] = {
                    name: channel_probe(brain, adapter, feed_off=f, return_off=r)
                    for name, f, r in (("baseline", False, False), ("feed_off", True, False),
                                       ("return_off", False, True), ("both_off", True, True))
                }
                baseline = receipt["channel_probes"]["baseline"]
                assert baseline["reads"] == receipt["validation"][pid]
                for probe in receipt["channel_probes"].values():
                    probe["pool_feature_l2_delta"] = math.sqrt(sum(
                        (x - y) ** 2 for a, b in zip(probe["pool_trace"], baseline["pool_trace"], strict=True)
                        for x, y in zip(a["features"], b["features"], strict=True)
                    ))
                meter, selection = adapter.select_cycle()
                receipt["selection"] = selection
                receipt["final_judgment"] = meter.judge(meter.winner)
                receipt["final_reads"] = adapter.final_reads
        else:
            previous = json.loads(Path(args.receipt).read_text())
            receipt["validation"] = {p: adapter.read_retention(p) for p in previous["validation"]}
            # RetentionSets returns tuples; JSON receipts carry lists. Compare
            # the exact serialized values across that representation boundary.
            assert json.loads(json.dumps(receipt["validation"])) == previous["validation"]
            # Learning-OFF reads above restore the complete original state.
            assert engine_model_digest(brain.brain._ixn, brain._g9_actors) == previous["model_digest"]
            assert json.loads(json.dumps({p: a.state_dict() for p, a in brain._g9_actors.items()})) == previous["actors"]
            if previous.get("selection"):
                from snn_data_lake.instruments.joint_mask_search import SelectionLeak
                from snn_data_lake.readout.posed_problem import PosedRefused
                meter, selection = adapter.select_cycle()
                assert selection == previous["selection"]
                try:
                    meter.judge(meter.winner)
                except (SelectionLeak, PosedRefused):
                    receipt["final_repeat_refused"] = True
                else:
                    raise AssertionError("durable final consumption was re-armed")
            receipt["fresh_process_equal"] = True
        receipt["model_digest"] = engine_model_digest(brain.brain._ixn, brain._g9_actors)
        receipt["core_state"] = core_state()
        receipt["actors"] = {p: a.state_dict() for p, a in brain._g9_actors.items()}
        receipt["source_events"] = [r for r in ledger.events() if r.get("kind") == "engine_source"]
        receipt["seconds"] = time.monotonic() - t0
        path = Path(args.receipt)
        if args.phase == "restore":
            tag = "" if os.environ.get("ENGINE_EXECUTION_ID", "1") == "1" else "-" + os.environ["ENGINE_EXECUTION_ID"]
            path = path.with_name(path.stem + ("-restore-journal" if journal_only else "-restore") + tag + ".json")
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        emit("receipt", path=str(path), seconds=receipt["seconds"])
    except BaseException as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        receipt["seconds"] = time.monotonic() - t0
        failure = "-" + args.phase + "-" + os.environ.get("ENGINE_EXECUTION_ID", "1") + "-failed.json"
        Path(args.receipt).with_name(Path(args.receipt).stem + failure).write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        raise
    finally:
        brain.close()
        del brain
        gc.collect()
    emit("closed")


if __name__ == "__main__":
    main()
