"""Bounded controller for one sequential, isolated actual-Brain acceptance phase."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("old", "grow", "restore", "restore_journal"))
parser.add_argument("--timeout", type=int, default=300)
parser.add_argument("--run-id", default="02")
parser.add_argument("--execution", default="1")
args = parser.parse_args()
if not all(x.isascii() and x.isdecimal() for x in (args.run_id, args.execution)):
    raise ValueError("explicit numeric engineering attempt identity required")
prefix = "cycle-" + args.run_id
phase_stamp = args.phase if args.execution == "1" else args.phase + "-" + args.execution
root = Path(__file__).resolve().parent
out = root / (prefix + "-" + phase_stamp + ".log")
archive_dir = root / (prefix + "-" + phase_stamp + "-source")
if out.exists() or archive_dir.exists():
    raise RuntimeError("an attempt's raw output/source cannot be overwritten")
archive_dir.mkdir()
workspace = root.parents[2]
runtime = Path("/Users/arthurkhvan/.codex/visualizations/2026/11/03/01a19f80-851f-7960-9c7a-18ef1711b019/engine-runtime")
source = runtime / "working-source"
pins = {}
for rel in (
    "src/snn_data_lake/growth/growth_loop.py",
    "src/snn_data_lake/growth/capacity_certificate.py",
    "src/snn_data_lake/engine/brain.py",
    "src/snn_data_lake/readout/posed_problem.py",
    "src/snn_data_lake/instruments/joint_mask_search.py",
    "src/snn_data_lake/readout/interaction_learning.py",
    "src/snn_data_lake/plasticity/plasticity_rule_stack.py",
    "src/snn_data_lake/plasticity/homeostatic_plasticity_stack.py",
):
    shutil.copyfile(workspace / "repo" / rel, source / rel)
    pins[rel] = hashlib.sha256((source / rel).read_bytes()).hexdigest()
    archived = archive_dir / rel
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / rel, archived)
script = root / "cycle_acceptance.py"
shutil.copyfile(script, root / (prefix + "-" + phase_stamp + "-script.py"))
pointer = root / (prefix + "-home.json")
if args.phase == "old":
    if pointer.exists():
        raise RuntimeError("reuse is forbidden: keep the prior attempt and choose a new controller pointer")
    private = tempfile.mkdtemp(prefix="acceptance-home-", dir=runtime)
    pointer.write_text(json.dumps({"home": private}, indent=2) + "\n")
else:
    private = json.loads(pointer.read_text())["home"]
receipt = root / (prefix + ("-old.json" if args.phase == "old" else "-grow.json"))
env = dict(os.environ, ENGINE_SOURCE_ROOT=str(source), NORD_HOME=private, NORD_PERSIST="1",
           SNDL_R2_CHILD_DEADLINE_NS=str(time.time_ns() + (args.timeout - 10) * 1_000_000_000),
           ENGINE_JOURNAL_ONLY="1" if args.phase == "restore_journal" else "0",
           ENGINE_EXECUTION_ID=args.execution)
phase = "restore" if args.phase == "restore_journal" else args.phase
cmd = ["zsh", str(root / "run-engine.sh"), str(root / "cycle_acceptance.py"), phase,
       "--receipt", str(receipt)]
started = time.monotonic()
info = {"phase": args.phase, "home": private, "command": cmd, "source_sha256": pins,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
(root / (prefix + "-" + phase_stamp + "-controller.json")).write_text(json.dumps(info, indent=2) + "\n")
with out.open("wb") as stream:
    try:
        result = subprocess.run(cmd, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=args.timeout)
        info["exit_code"] = result.returncode
    except subprocess.TimeoutExpired:
        info["timeout"] = True
info["seconds"] = time.monotonic() - started
(root / (prefix + "-" + phase_stamp + "-controller.json")).write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, sort_keys=True))
print(out.read_text()[-5000:])
