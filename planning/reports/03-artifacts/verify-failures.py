"""Exercise saved-run error handling on throwaway copies; never edit real records."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    audit = load_module("recompute")
    # Positive first, through precisely the same reader as the mutated cases.
    manifest, rows = audit.load_run(HERE / "pilot-1")
    audit.compare(rows)
    outcomes = {"complete_run_positive": {"detected": True}}
    mutations = {
        "worker_failed": lambda m: m["runs"][1].update(exit_code=17, exclusion_reason="fixture worker failure"),
        "worker_timeout": lambda m: m["runs"][1].update(exit_code=None, exclusion_reason="fixture worker timeout"),
        "hash_corruption": lambda m: m["runs"][1].update(result_sha256="0" * 64),
        "incomplete_run": lambda m: m.update(status="HALT"),
        "missing_manifest_arm": lambda m: m["runs"].pop(1),
        "declared_exclusion": lambda m: m["exclusions"].append({"reason": "fixture unavailable arm"}),
    }
    with tempfile.TemporaryDirectory(prefix="sndl-stage03-errors-") as temporary:
        folder = Path(temporary) / "copy"
        shutil.copytree(HERE / "pilot-1", folder)
        for name, mutate in mutations.items():
            changed = copy.deepcopy(manifest)
            mutate(changed)
            (folder / "manifest.json").write_text(json.dumps(changed))
            try:
                audit.load_run(folder)
            except audit.Invalid as error:
                outcomes[name] = {"detected": True, "reason": str(error), "synthetic": True}
            else:
                raise RuntimeError("undetected saved-run defect: " + name)
    # Verify the one-command entry point's stop-on-failure without running learners.
    runner = load_module("run-stage")
    from unittest.mock import patch
    import subprocess
    for fail_at in (0, 1, 2):
        seen = []
        def fake_run(command, **kwargs):
            seen.append(command)
            if len(seen) - 1 == fail_at:
                raise subprocess.CalledProcessError(9, command)
        try:
            with patch.object(runner.sys, "argv", ["run-stage.py", "--run-id", "fixture-only"]), patch.object(runner.subprocess, "run", fake_run):
                runner.main()
        except subprocess.CalledProcessError:
            if len(seen) != fail_at + 1:
                raise RuntimeError("pipeline continued after a failed stage")
            outcomes[f"pipeline_halts_at_stage_{fail_at + 1}"] = {"detected": True, "synthetic": True}
        else:
            raise RuntimeError("pipeline swallowed an error")
    output = HERE / "failure-checks.json"
    if output.exists():
        raise FileExistsError("Failure records are append-only; use a new output name")
    output.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print("Saved-run positive and", len(outcomes) - 1, "intentional failures detected; no training.")


if __name__ == "__main__":
    main()
