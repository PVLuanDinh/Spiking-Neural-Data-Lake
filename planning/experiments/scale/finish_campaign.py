"""Finish the two unstarted arms after the documented infrastructure retry."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
gate = ROOT / "runs/shd-sequential-s8003/summary.json"
start = time.monotonic()
while not gate.exists():
    assert not (gate.parent / "failure.json").exists()
    assert time.monotonic() - start < 1200
    time.sleep(5)
records = []
for arm in ("repeat", "replay"):
    run_id = f"shd-{arm}-s8003"
    assert not (ROOT / "runs" / run_id).exists()
    print(json.dumps({"event": "launch", "run_id": run_id}), flush=True)
    with open(ROOT / "diagnostics" / f"{arm}-s8003.log", "w") as output:
        result = subprocess.run(["zsh", str(ROOT / "runtime.sh"), str(ROOT / "pilot.py"),
                                 "--run-id", run_id, "--seed", "8003", "--arm", arm,
                                 "--max-seconds", "3600"], cwd=WORKSPACE, stdout=output,
                                stderr=subprocess.STDOUT, timeout=3700)
    record = {"run_id": run_id, "exit_code": result.returncode}
    records.append(record)
    (ROOT / "reports/campaign-resume.json").write_text(json.dumps(records, indent=2) + "\n")
    assert result.returncode == 0
    print(json.dumps(record), flush=True)
print("Remaining registered arms complete", flush=True)
