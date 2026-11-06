"""Bounded stage-04 orchestration; all learning and pool dynamics are adopted."""

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT / "repo"
HEAD = "c4c10b285f1741aa798d5ee9c5296ae371fabecf"
SEEDS = (8000, 8001, 8002, 8003)
TASKS = ("i.S0C0", "i.S1C0", "ac1d")
MODES = ("carry", "reset")
ARMS = ("rst", "rst_frozen", "rst_shuffle", "delta", "delta_frozen", "delta_shuffle")
EPOCHS = 30

spec = importlib.util.spec_from_file_location("stage03", HERE.parent / "03-artifacts/evaluate.py")
stage03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage03)
dumps, digest, require = stage03.dumps, stage03.digest, stage03.require


def hashed(value):
    return digest(dumps(value).encode("ascii"))


def save(path, value):
    path.write_text(dumps(value) + "\n", encoding="ascii")


def state(unit):
    if hasattr(unit, "_snapshot"):
        return stage03.snapshot(unit)
    return {"beta": unit.beta.tolist(), "dtype": str(unit.beta.dtype),
            "labels": unit.labels, "n_in": unit.n_in, "ready": unit._ready,
            "buffer": unit._buf, "rule": dict(vars(unit.rule))}


def answer(unit, x):
    # DeltaAdaptor.respond has an offline-adaptor first-read hook. Delta is online;
    # its existing pure _answer also handles the zero-step/frozen beta correctly.
    return unit.respond(x) if hasattr(unit, "_snapshot") else unit._answer(x)


def guarded_read(unit, rows):
    before = hashed(state(unit))

    def forbidden(*args, **kwargs):
        raise stage03.EvaluationError("update during read")

    with contextlib.ExitStack() as stack:
        names = ("learn_turn", "interact", "_deposit") if hasattr(unit, "_snapshot") else ("learn",)
        for name in names:
            stack.enter_context(patch.object(type(unit), name, forbidden))
        answers = [answer(unit, row["x"]) for row in rows]
    after = hashed(state(unit))
    if before != after:
        raise stage03.EvaluationError("state/replay changed during read")
    return {"answers": answers, "before": before, "after": after}


def param_state(pool):
    import numpy as np

    result = {"fingerprint": pool.params_fingerprint(), "host": [pool.frac_bits, pool.p_parts, pool.threads],
              "config": pool.config, "form": pool.form}
    for name, arr in {**pool.params(), **pool._live_edges(), "w_in": pool.w_in}.items():
        a = np.asarray(arr)
        result[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                        "sha256": digest(a.tobytes())}
    return result


def upper_bound(rows, key=lambda row: row["x"]):
    groups = defaultdict(Counter)
    for row in rows:
        groups[dumps(key(row))][row["target"]] += 1
    return {"correct_max": sum(max(c.values()) for c in groups.values()), "n": len(rows),
            "unique": len(groups), "conflicting_groups": sum(len(c) > 1 for c in groups.values())}


def features(seed):
    from snn_data_lake.instruments import taught_problems as tp, taught_campaign as tc
    from snn_data_lake.instruments.g5_part_d_cells import _Drive

    out = {"seed": seed, "tasks": {}}
    for task in TASKS:
        problem = tp.PROBLEMS[task]
        teach = list(problem.teach_epochs(seed, EPOCHS))
        true = [[ep[-1].label for ep in epoch] for epoch in teach]
        shuffle = tc._shuffled_labels(seed)
        shifted = [shuffle(labels) for labels in true]
        raw = {"train": [ep for epoch in teach for ep in epoch],
               "dev": list(problem.episodes(seed, "validation", read_index=0)),
               "final": list(problem.episodes(seed, "future", read_index=0))}
        task_out = {"epoch_size": len(teach[0]), "true": true, "shuffle": shifted, "modes": {}}
        for mode in MODES:
            drive = _Drive(seed, mode, problem)
            before = param_state(drive._pool)
            paths = {}
            for split, episodes in raw.items():
                rows = []
                for episode in episodes:
                    turns = drive.walk(episode)
                    rows.append({"texts": [t.text for t in episode], "target": episode[-1].label,
                                 "x": turns[-1].x})
                paths[split] = rows
            after = param_state(drive._pool)
            require(before == after, "pool parameters changed")
            repeat = _Drive(seed, mode, problem)
            probe = raw["dev"][0]
            require(repeat.walk(probe)[-1].x == paths["dev"][0]["x"], "feature fresh repeat differs")
            # Process an unrelated prefix, then the same episode: boundary reset must erase it.
            repeat.walk(raw["dev"][-1])
            require(repeat.walk(probe)[-1].x == paths["dev"][0]["x"], "episode prefix leaked")
            bare = tp.PROBLEMS["i.S0C0"].intents
            unit = __import__("snn_data_lake.instruments.g5_part_d_cells", fromlist=["_learner"])._learner(sorted(bare), 256, seed)
            task_out["modes"][mode] = {"paths": paths, "parameters_before": before, "parameters_after": after,
                "path_sha256": hashed(paths), "dev_feature_bound": upper_bound(paths["dev"]),
                "final_feature_bound": upper_bound(paths["final"]),
                "dev_rank_bound": upper_bound(paths["dev"], key=lambda r: unit.encode(r["x"])),
                "checks": {"fresh_repeat": True, "boundary_reset": True}}
        if task == "ac1d":
            require(task_out["modes"]["carry"]["paths"] == task_out["modes"]["reset"]["paths"],
                    "one-turn carry/reset mismatch")
        out["tasks"][task] = task_out
    return out


def checks():
    from snn_data_lake.instruments import taught_problems as tp, taught_campaign as tc
    from snn_data_lake.instruments.g5_part_d_adaptors import delta
    from snn_data_lake.instruments.g5_part_d_cells import _learner

    results = {}
    for task, kind in (("i.S0C0", "incumbent"), ("i.S1C0", "history")):
        p = tp.PROBLEMS[task]
        reader = tc.make_reader(kind, p, 8000)
        run = tc.attempt(p, reader, 8000, tc.EPOCHS, stop_when_stable=True, judge_future=True)
        require(run["stable"] and run["future_solved"], "upstream positive failed")
        results[task] = run
    reds = {}
    x = [1.0, 0.0, 0.0, 0.0]
    for family, bad in (("rst", "update"), ("rst", "replay"), ("delta", "update"), ("delta", "beta")):
        u = _learner(["GOAL", "GREET"], 4, 8900) if family == "rst" else delta(["GOAL", "GREET"], 4)
        original = u.respond if family == "rst" else u._answer

        def broken(v):
            if bad == "replay":
                u._replay.append((v, "GOAL", None))
            elif bad == "beta":
                u.beta[0, 0] += 1
            elif family == "rst":
                u.interact(v, "GOAL")
            else:
                u.learn(v, "GOAL")
            return original(v)

        method = "respond" if family == "rst" else "_answer"
        # The checkpoint excludes this transient instance callable only on the delta
        # explicit state reader. For RSTDP patch the class for checkpoint cleanliness.
        def substitute(self, v):
            return broken(v)
        with patch.object(type(u), method, substitute):
            try:
                guarded_read(u, [{"x": x}])
            except stage03.EvaluationError:
                reds[f"{family}_{bad}"] = True
            else:
                reds[f"{family}_{bad}"] = False
    require(all(reds.values()), "guard failed to catch corruption")
    return {"positive": results, "guard_reds": reds}


def arm(seed, name, inputs):
    from snn_data_lake.instruments.g5_part_d_cells import _learner
    from snn_data_lake.instruments.g5_part_d_adaptors import delta

    require(inputs["seed"] == seed, "wrong feature seed")
    out = {"seed": seed, "arm": name, "cases": {}}
    for task, data in inputs["tasks"].items():
        for mode in MODES:
            case = data["modes"][mode]
            paths = case["paths"]
            labels = sorted(set(v for epoch in data["true"] for v in epoch))
            u = _learner(labels, 256, seed) if name.startswith("rst") else delta(labels, 256)
            initial = state(u)
            reads = [guarded_read(u, paths["dev"])]
            updates = 0
            for epoch in range(EPOCHS):
                start, size = epoch * data["epoch_size"], data["epoch_size"]
                taught = data["shuffle"] if name.endswith("shuffle") else data["true"]
                for row, label in zip(paths["train"][start:start + size], taught[epoch], strict=True):
                    if not name.endswith("frozen"):
                        if name.startswith("rst"):
                            u.interact(row["x"], label)
                        else:
                            u.learn(row["x"], label)
                        updates += 1
                reads.append(guarded_read(u, paths["dev"]))
            frozen = state(u)
            final = guarded_read(u, paths["final"])
            require(state(u) == frozen, "final changed frozen state")
            if name.endswith("frozen"):
                require(initial == frozen, "no-tune state changed")
            out["cases"][f"{task}/{mode}"] = {"path_sha256": case["path_sha256"],
                "initial": initial, "final_state": frozen, "updates": updates,
                "epochs": EPOCHS, "dev": reads, "final": final}
    return out


def worker(args):
    stage03.runtime()
    from snn_data_lake.substrate.home import require_isolated_home
    require_isolated_home()
    import torch
    import numpy as np
    torch.set_num_threads(1)
    start, cpu = time.monotonic(), time.process_time()
    if args.worker == "checks":
        result = checks()
    elif args.worker == "features":
        result = features(args.seed)
    else:
        result = arm(args.seed, args.worker, json.loads(Path(args.inputs).read_text()))
    stage03.runtime()
    require(not list(Path(os.environ["NORD_HOME"]).iterdir()), "isolated home not empty")
    print(dumps({"schema": "sndl.stage04.worker.v1", "kind": args.worker, "seed": args.seed,
        "result": result, "runtime": {"python": sys.version, "numpy": np.__version__,
            "torch": torch.__version__, "gil": sys._is_gil_enabled()},
        "cost": {"wall_seconds": time.monotonic()-start, "cpu_seconds": time.process_time()-cpu,
                 "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}))


def checkout():
    require(subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == HEAD, "source HEAD changed")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(), "source dirty")


def run(args):
    checkout()
    stage03.runtime()
    require(args.run_id and Path(args.run_id).name == args.run_id, "simple new run-id required")
    out = HERE / args.run_id
    out.mkdir()
    manifest = {"schema": "sndl.stage04.run.v1", "head": HEAD, "status": "RUNNING", "runs": [],
                "seeds": list(SEEDS), "tasks": list(TASKS), "arms": list(ARMS), "modes": list(MODES), "epochs": EPOCHS}
    manifest["provenance"] = {}
    sources = [HERE / "evaluate.py", HERE / "protocol.md", HERE.parent / "03-artifacts/evaluate.py"]
    sources += sorted((REPO / "src/snn_data_lake").rglob("*.py"))
    sources += [REPO / "tools/kernel_partition_baseline.json", REPO / "tools/determinism_baseline.json"]
    for source in sources:
        key = str(source.relative_to(ROOT))
        manifest["provenance"][key] = digest(source.read_bytes())
        if source in sources[:3] or any(v in key for v in ("g5_part_d", "g5_readout", "recurrent_pool", "kernel_", "interaction_learning", "rank_order", "r_stdp.py", "online_readout", "taught_", "delayed_cue")):
            target = out / "provenance" / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    started = time.monotonic()

    def job(kind, seed=8000, suffix=""):
        require(time.monotonic()-started < 1200, "whole block time limit")
        key = f"{kind}-{seed}{suffix}"
        cmd = [sys.executable, "-B", str(HERE / "evaluate.py"), "--worker", kind, "--seed", str(seed)]
        if kind in ARMS:
            cmd += ["--inputs", str(out / f"features-{seed}.json") + ".inputs"]
        env = os.environ.copy()
        for k in ("NORD_ALLOW_GIL", "NORD_GIL_FALLBACK", "PYTHON_GIL"):
            env.pop(k, None)
        env.update(PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="sndl04-") as home:
            env["NORD_HOME"] = home
            before = time.monotonic()
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, timeout=120)
                raw, err, code = proc.stdout, proc.stderr, proc.returncode
                failure = None if code == 0 else f"worker exit {code}"
            except subprocess.TimeoutExpired as exc:
                raw, err, code = exc.stdout or b"", exc.stderr or b"", None
                failure = "timeout 120s"
            empty = not list(Path(home).iterdir())
        (out / f"{key}.json").write_bytes(raw)
        (out / f"{key}.stderr.txt").write_bytes(err)
        entry = {"key": key, "command": cmd, "sha256": digest(raw), "bytes": len(raw), "returncode": code,
                 "seconds": time.monotonic()-before, "home_empty": empty, "failure": failure}
        manifest["runs"].append(entry)
        save(out / "manifest.json", manifest)
        require(failure is None and empty, f"HALT {key}: {failure or 'unexpected home write'}")
        data = json.loads(raw)
        require(data["kind"] == kind and data["seed"] == seed, "worker identity mismatch")
        if kind == "features" and not suffix:
            save(out / f"features-{seed}.json.inputs", data["result"])
        print(f"{key}: OK ({entry['seconds']:.2f}s)", flush=True)
        return data["result"]

    try:
        job("checks", 8900)
        for seed in SEEDS:
            job("features", seed)
            for name in ARMS:
                job(name, seed)
        # Full 8000 rerun, including features, each in a fresh process.
        for name in ("features", *ARMS):
            repeated = job(name, 8000, "-repeat")
            first = json.loads((out / f"{name}-8000.json").read_text())["result"]
            require(first == repeated, f"new-process repeat differs: {name}")
        checkout()
        manifest["status"] = "COMPLETE"
    except BaseException as exc:
        manifest["status"] = "HALT"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["total_seconds"] = time.monotonic()-started
        save(out / "manifest.json", manifest)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--worker", choices=("checks", "features", *ARMS))
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--inputs")
    args = parser.parse_args()
    worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
