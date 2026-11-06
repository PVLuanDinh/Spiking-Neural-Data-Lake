"""Stage-02 diagnostic adapter over existing SNDL instruments; no source edits."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "repo"
HERE = Path(__file__).resolve().parent
HEAD = "c4c10b285f1741aa798d5ee9c5296ae371fabecf"
SEED = 8000
# (problem, existing reader, lr, shuffle, stop at stable-10, history window)
CASES = {
    "s0c0": ("i.S0C0", "incumbent", None, False, True, 3),
    "s0c0-lr0": ("i.S0C0", "incumbent", 0.0, False, False, 3),
    "s0c0-shuffle": ("i.S0C0", "incumbent", None, True, False, 3),
    "s1c0": ("i.S1C0", "incumbent", None, False, False, 3),
    "s0c1": ("i.S0C1", "incumbent", None, False, False, 3),
    "s1c1": ("i.S1C1", "incumbent", None, False, False, 3),
    "history-s0c0": ("i.S0C0", "history", None, False, True, 3),
    "history-s1c0": ("i.S1C0", "history", None, False, True, 3),
    "history-s1c1": ("i.S1C1", "history", None, False, False, 3),
    "presized-s0c1": ("i.S0C1", "presized", None, False, True, 3),
    "presized-history-s1c1": ("i.S1C1", "presized_history", None, False, True, 3),
    "presized-s1c0": ("i.S1C0", "presized", None, False, False, 3),
    "history-window1": ("i.S1C0", "history", None, False, False, 1),
    "history-constant-cue": ("i.S1C0", "constant_cue", None, False, False, 3),
    "presized-s0c1-shuffle": ("i.S0C1", "presized", None, True, False, 3),
    "ac1d": ("ac1d", "incumbent", None, False, True, 3),
    "ac1d-lr0": ("ac1d", "incumbent", 0.0, False, False, 3),
    "c1d": ("c1d", "incumbent", None, False, False, 3),
    "presized-c1d": ("c1d", "presized", None, False, False, 3),
}
PHASES = {
    "checks": ["checks"],
    "s0c0": ["s0c0", "s0c0-lr0", "s0c0-shuffle"],
    "repeat": ["s0c0"],
    "cells": list(CASES)[3:15],
    "combos": list(CASES)[15:],
}


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def runtime_check():
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("Use .venv-01/bin/python (CPython 3.14t)")
    if sysconfig.get_config_var("Py_GIL_DISABLED") != 1 or sys._is_gil_enabled():
        raise RuntimeError("The GIL must be off")


def worker(case):
    runtime_check()
    from snn_data_lake.substrate.home import require_isolated_home

    home = Path(require_isolated_home())
    if any(home.iterdir()):
        raise RuntimeError("The diagnostic home must start empty")
    from snn_data_lake.instruments import delayed_cue as dc
    from snn_data_lake.instruments import taught_campaign as tc
    from snn_data_lake.instruments import taught_problems as tp

    tc.preflight((SEED,), HERE, arms=((case,),), require_clean=True)
    if case == "checks":
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            tp.main()
            tc.main()
        construction = {
            "b2": dc.check_b2(), "b2_c1d": dc.check_b2_combo("c1d"),
            "b2a": dc.check_b2a(), "witnesses": dc.check_witnesses(),
            "b10": dc.check_b10(SEED), "b10s": dc.check_b10s(SEED),
        }
        for family in dc.COMBO_FAMILIES:
            parts, _ = dc.combo_partition(family, SEED)
            construction["b13_" + family] = dc.check_b13(parts)
            construction["b14_" + family] = dc.check_b14(family, parts)
        out = {
            "case": case,
            "selfchecks_stdout": stdout.getvalue(),
            "construction": construction,
            "b1": "NOT RUN: firewall imports the unavailable mujoco package",
            "generator_reds": "NOT RUN: full P1 reds require the unavailable firewall import",
            "contract_reds": tp.construction_reds(tp.canonical_incumbent),
            "b9": tp.check_b9(),
            "b11": tp.check_b11(),
            "reading_reds": tc.reading_reds(),
            "float_lp": dc.lp_measurements(),
            "b8": "NOT RUN: full Brain dependencies unavailable; no identity claim",
        }
        assert all(check["ok"] for check in construction.values())
        assert out["contract_reds"]["all_fired"]
    else:
        pid, arm, lr, shuffled, early, window = CASES[case]
        problem = tp.PROBLEMS[pid]
        reader = (
            tc.constant_cue_reader(problem, SEED)
            if arm == "constant_cue"
            else tc.make_reader(arm, problem, SEED, h=tc.H, lr=lr, window=window)
        )
        unit = reader.unit.unit if isinstance(reader.unit, tc.Frozen) else reader.unit
        epochs = tc.EPOCHS
        initial_state = tp.learner_state(unit)
        initial_weights = dump(unit.weights()).encode("ascii")
        b12 = tp.check_b12(unit, problem, SEED)
        assert b12["ok"]

        def read(split):
            before = tp.learner_state(unit)
            records = []
            reads = tp.read_split(problem, SEED, split, reader.route, records=records)
            assert before == tp.learner_state(unit), "evaluation changed full learner state"
            return {
                "reads": reads,
                "records": records,
                "solved": tp.solved(problem, SEED, split, reads),
                "accuracy": sum(a == b for a, b in reads) / len(reads),
                "state_unchanged": True,
            }

        before_read = read("validation")
        epoch_reads, b3 = [], []

        def on_epoch_end(k, _reader, records):
            labels = [ep[-1].label for ep in problem.episodes(SEED, "validation", read_index=k)]
            epoch_reads.append([
                {"context": context, "probe": probe, "answer": answer, "label": label}
                for (context, probe, answer), label in zip(records, labels, strict=True)
            ])
            if arm == "incumbent" and pid in ("i.S1C0", "i.S1C1"):
                b3.append(tp.check_b3_stream(records)["ok"])

        attempt = tc.attempt(
            problem, reader, SEED, epochs, judge_future=False,
            stop_when_stable=early, shuffled=shuffled, on_epoch_end=on_epoch_end,
        )
        validation, future = read("validation"), read("future")
        sequence = tp.teaching_sequence(problem, SEED, attempt["epochs_run"])
        # For shuffled arms this list is the true target sequence, not the labels actually taught.
        taught_labels = None
        if shuffled:
            shuffler = tc._shuffled_labels(SEED)
            taught_labels = [
                shuffler([ep[-1].label for ep in epoch])
                for epoch in problem.teach_epochs(SEED, attempt["epochs_run"])
            ]
        out = {
            "case": case, "seed": SEED, "problem": pid, "arm": arm,
            "config": {"lr": unit.lr, "shuffled": shuffled, "stop_when_stable": early,
                       "window": window, "max_epochs": epochs, "stable": tp.STABLE,
                       "n_in": unit.n_in, "responses": unit.responses, "t_post": unit.t_post},
            "contract": {
                "first_teach_epoch": list(problem.teach_epochs(SEED, 1))[0],
                "validation_episodes": problem.episodes(SEED, "validation"),
                "future_episodes": problem.episodes(SEED, "future"),
                "delay_turns": problem.delay_turns,
                "first_growth_budget": tp.open_budget(problem, [], seed=SEED),
                "n_teach_codes": problem.n_teach_codes(SEED),
                "n_union_codes": problem.n_union_codes(),
            },
            "before": before_read, "attempt": attempt,
            "epoch_reads": epoch_reads, "validation": validation, "future": future,
            "teaching_sequence": sequence, "shuffled_labels_actually_taught": taught_labels,
            "b12": b12, "b3_every_read": all(b3) if b3 else None,
            "initial_state_sha256": sha(initial_state),
            "final_state_sha256": sha(tp.learner_state(unit)),
            "initial_weights_sha256": sha(initial_weights),
            "final_weights_sha256": sha(dump(unit.weights()).encode("ascii")),
            "replay_entries_after_training": len(unit._replay),
        }
    runtime_check()
    if any(home.iterdir()):
        raise RuntimeError("Unexpected writes in the isolated diagnostic home")
    out.update(gil_enabled_after=False, isolated_home_empty=True)
    print(dump(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["checks", *CASES])
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    runtime_check()
    if args.worker:
        worker(args.worker)
        return
    if not args.phase or not args.run_id or not args.run_id.replace("-", "").isalnum():
        parser.error("Use --phase PHASE --run-id NEW-ID")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if head != HEAD or dirty:
        raise RuntimeError("The pinned clean checkout is required")
    out_dir = HERE / args.run_id
    out_dir.mkdir(exist_ok=False)
    source_paths = [
        REPO / "src/snn_data_lake/instruments" / (name + ".py")
        for name in ("taught_campaign", "taught_problems", "delayed_cue")
    ]
    manifest = {
        "phase": args.phase, "head": head, "python": sys.version,
        "machine": platform.machine(), "platform": platform.platform(),
        "protocol_sha256": sha((HERE / "protocol.md").read_bytes()),
        "adapter_sha256": sha(Path(__file__).read_bytes()),
        "source_sha256": {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in source_paths},
        "runs": [],
    }
    start = time.monotonic()
    env = os.environ.copy()
    for key in ("NORD_ALLOW_GIL", "PYTHON_GIL"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for case in PHASES[args.phase]:
        if time.monotonic() - start > 1200:
            raise TimeoutError("20 minute diagnostic budget exceeded")
        case_start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="sndl-02-") as task_home:
            env["NORD_HOME"] = task_home
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", case],
                cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
            )
            (out_dir / (case + ".stdout.json")).write_text(result.stdout, encoding="ascii")
            (out_dir / (case + ".stderr.txt")).write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"{case} failed ({result.returncode}); see {out_dir}")
            row = json.loads(result.stdout)
            if any(Path(task_home).iterdir()) or not row["isolated_home_empty"]:
                raise RuntimeError("Diagnostic home is not empty")
        elapsed = time.monotonic() - case_start
        manifest["runs"].append({"case": case, "seconds": elapsed,
                                 "result_sha256": sha(result.stdout.encode("ascii")),
                                 "stderr_empty": not result.stderr, "exit_code": result.returncode})
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="ascii")
        a = row.get("attempt")
        print(case, f"{elapsed:.2f}s", "checks OK" if a is None else
              f"before={row['before']['accuracy']:.3f} final={row['validation']['accuracy']:.3f} "
              f"max={a['max_accuracy']:.3f} stable={a['stable']} epochs={a['epochs_run']} "
              f"future={row['future']['accuracy']:.3f}", flush=True)
        if args.phase == "s0c0" and a["stable"] != (case == "s0c0"):
            raise RuntimeError("S0C0 control prerequisite failed; stop diagnostics")


if __name__ == "__main__":
    main()
