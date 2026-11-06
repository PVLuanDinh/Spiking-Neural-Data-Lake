"""Exposed-fixture actual Brain integration; raw events remain in the isolated lake.

No future is read until the actual winner is frozen. A failed acquisition is
reported OPEN and never converted to a successful presentation result.
"""
import argparse
import gc
import json
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
}


def emit(kind, **fields):
    print("ENGINE " + json.dumps({"kind": kind, **fields}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("old", "grow", "restore"))
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    home.require_isolated_home()
    t0 = time.monotonic()
    brain = make_brain(seed=9900, persist=True)
    assert brain._r2_fault is None, brain._r2_fault
    ledger = Ledger(home.knowledge_path(), BudgetValues(tp.FIRST_BUDGET, None))
    append = ledger.record_event

    def observed(row):
        append(row)
        if row["kind"] != "engine_epoch" or row["epoch"] % 25 == 0 or row["stable_run"]:
            emit("progress", event=row)

    ledger.record_event = observed
    receipt = {"scope": "single exposed fixture engineering, task-conditioned, no-prune",
               "seed": 9900, "phase": args.phase, "recipe": RECIPE}
    try:
        pid = "aii" if args.phase == "old" else "ii.xor"
        adapter = CanonicalGrowthAdapter.prepare_cycle(brain, tp.PROBLEMS[pid], ledger=ledger, recipe=RECIPE)
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
                meter, selection = adapter.select_cycle()
                receipt["selection"] = selection
                receipt["final_judgment"] = meter.judge(meter.winner)
                receipt["final_reads"] = adapter.final_reads
        else:
            previous = json.loads(Path(args.receipt).read_text())
            receipt["validation"] = {p: adapter.read_retention(p) for p in previous["validation"]}
            assert receipt["validation"] == previous["validation"]
            # Learning-OFF reads above restore the complete original state.
            assert engine_model_digest(brain.brain._ixn, brain._g9_actors) == previous["model_digest"]
            assert {p: a.state_dict() for p, a in brain._g9_actors.items()} == previous["actors"]
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
        receipt["actors"] = {p: a.state_dict() for p, a in brain._g9_actors.items()}
        receipt["source_events"] = [r for r in ledger.events() if r.get("kind") == "engine_source"]
        receipt["seconds"] = time.monotonic() - t0
        path = Path(args.receipt)
        if args.phase == "restore":
            path = path.with_name(path.stem + "-restore.json")
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        emit("receipt", path=str(path), seconds=receipt["seconds"])
    finally:
        brain.close()
        del brain
        gc.collect()
    emit("closed")


if __name__ == "__main__":
    main()
