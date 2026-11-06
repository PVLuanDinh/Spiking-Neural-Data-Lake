"""Serial, capped pilot launcher; gated on the first real-data positive control."""
from pathlib import Path
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]


def main():
    gate = ROOT / "runs/shd-joint-s8000/summary.json"
    waiting = time.monotonic()
    while not gate.exists():
        assert time.monotonic() - waiting < 900, "initial positive control not complete after 15 minutes"
        assert not (gate.parent / "failure.json").exists(), "initial positive control failed"
        time.sleep(5)
    first = json.loads(gate.read_text())
    assert first["positive_control_gate_pass"], "real-data learnability gate failed: stop expansion"
    budget = 4 * 3600 - first["wall_seconds"]
    start = time.monotonic()
    records = [{"run_id": "shd-joint-s8000", "status": "complete", "wall_seconds": first["wall_seconds"]}]
    schedule = [(seed, arm) for seed in range(8000, 8004) for arm in ("joint", "sequential", "repeat", "replay") if (seed, arm) != (8000, "joint")]
    for seed, arm in schedule:
        remaining = budget - (time.monotonic() - start)
        if remaining < 600:
            records.append({"status": "budget_stop", "next_seed": seed, "next_arm": arm})
            break
        run_id = f"shd-{arm}-s{seed}"
        print(json.dumps({"event": "launch", "run_id": run_id, "remaining_seconds": remaining}), flush=True)
        path = ROOT / "diagnostics" / f"{arm}-s{seed}.log"
        command = ["zsh", str(ROOT / "runtime.sh"), str(ROOT / "pilot.py"), "--run-id", run_id,
                   "--seed", str(seed), "--arm", arm, "--max-seconds", str(int(min(3600, remaining)))]
        stamp = time.monotonic()
        with open(path, "w") as output:
            try:
                result = subprocess.run(command, cwd=WORKSPACE, stdout=output, stderr=subprocess.STDOUT,
                                        timeout=min(3700, remaining))
                rc = result.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        record = {"run_id": run_id, "exit_code": rc, "wall_seconds": time.monotonic() - stamp}
        summary = ROOT / "runs" / run_id / "summary.json"
        if rc == 0 and summary.exists():
            values = json.loads(summary.read_text())
            record.update(status="complete", accuracy=values["final_accuracy"], macro_accuracy=values["final_macro_accuracy"])
        else:
            record["status"] = "failed"
        records.append(record)
        (ROOT / "reports/campaign-progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if rc != 0:
            break
    (ROOT / "reports/campaign-progress.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"event": "campaign_complete", "records": records}), flush=True)


if __name__ == "__main__":
    main()
