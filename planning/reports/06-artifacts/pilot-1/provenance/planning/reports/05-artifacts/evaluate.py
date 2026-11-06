"""Stage 05 orchestration over existing fixed-pool, readout and lake components."""

import argparse
import contextlib
import copy
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
TASKS = ("i.S0C0", "ac1d")
SEEDS = (8000, 8001, 8002, 8003)
ARMS = ("none", "replay", "batch", "batch_shuffle")
EPOCHS = 30
SLEEP_PASSES = 10
STATE_NAME = "stage05-readout-v1"
spec = importlib.util.spec_from_file_location("stage04", HERE.parent / "04-artifacts/evaluate.py")
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)
dumps, digest, require, hashed, save = s4.dumps, s4.digest, s4.require, s4.hashed, s4.save


def lake_file():
    return Path(os.environ["NORD_HOME"]) / "brain_lake.bin"


def lake_hash():
    path = lake_file()
    return digest(path.read_bytes()) if path.exists() else None


def snapshot(unit, replay):
    return {"readout": s4.state(unit), "replay": copy.deepcopy(replay)}


def read(unit, replay, rows):
    from snn_data_lake.growth.ng_snn import NGSNN

    before, disk = hashed(snapshot(unit, replay)), lake_hash()

    def forbidden(*args, **kwargs):
        raise RuntimeError("update during evaluation")

    with patch.object(type(unit), "learn", forbidden), patch.object(NGSNN, "_solve_beta", forbidden):
        result = s4.guarded_read(unit, rows)
    require(before == hashed(snapshot(unit, replay)), "evaluation changed state/replay")
    require(disk == lake_hash(), "evaluation wrote lake")
    return {"answers": result["answers"], "state_sha256": before, "lake_sha256": disk}


def score(answers, rows):
    require(len(answers) == len(rows) and bool(rows), "invalid response count")
    correct = sum(a == r["target"] for a, r in zip(answers, rows, strict=True))
    return {"correct": correct, "n": len(rows), "accuracy": correct / len(rows)}


def read_both(unit, replay, data, split):
    return {task: read(unit, replay, data["tasks"][task][split]) for task in TASKS}


def batch_update(unit, rows, labels=None):
    import torch
    from snn_data_lake.growth.ng_snn import NGSNN

    solver = NGSNN(device="cpu", seed=8900)
    xs = torch.tensor([row["x"] for row in rows], dtype=torch.float32)
    ys = torch.zeros(len(rows), len(unit.labels), dtype=torch.float32)
    target = labels if labels is not None else [row["target"] for row in rows]
    for i, label in enumerate(target):
        ys[i, unit.labels.index(label)] = 1.0
    unit.beta = solver._solve_beta(xs, ys)
    unit._ready = True
    require(solver.W is None and solver.beta is None, "solver grew a model")
    return {"presentations": len(rows), "solves": 1, "c1": solver.c1, "k": solver.k, "c0": solver.c0}


def encode_checkpoint(unit, replay, data, phase):
    body = {"schema": "sndl.stage05.checkpoint.v1", "head": HEAD, "seed": data["seed"],
            "phase": phase, "pool": data["pool"], "state": snapshot(unit, replay)}
    return dumps({"body": body, "sha256": hashed(body)}).encode("ascii")


def save_checkpoint(unit, replay, data, phase):
    from snn_data_lake.substrate.home import save_named_state

    blob = encode_checkpoint(unit, replay, data, phase)
    require(save_named_state(STATE_NAME, blob, path=str(lake_file())), "checkpoint not written")
    return digest(blob)


def decode_checkpoint(blob, expected):
    require(blob is not None, "missing checkpoint")
    require(digest(blob) == expected, "checkpoint receipt mismatch (missing/stale/corrupt write)")
    envelope = json.loads(blob)
    require(set(envelope) == {"body", "sha256"}, "checkpoint envelope schema")
    body = envelope["body"]
    require(hashed(body) == envelope["sha256"], "checkpoint checksum")
    require(body["schema"] == "sndl.stage05.checkpoint.v1" and body["head"] == HEAD, "checkpoint version")
    return body


def restore_unit(body):
    import torch
    from snn_data_lake.instruments.g5_part_d_adaptors import delta

    state = body["state"]["readout"]
    unit = delta(state["labels"], state["n_in"], mu=state["rule"]["mu"])
    require(state["dtype"] == "torch.float32", "unsupported dtype")
    beta = torch.tensor(state["beta"], dtype=torch.float32)
    require(list(beta.shape) == [state["n_in"], len(state["labels"])], "bad beta shape")
    require(bool(torch.isfinite(beta).all()), "nonfinite beta")
    unit.beta, unit._ready, unit._buf = beta, state["ready"], state["buffer"]
    unit.rule.__dict__.update(state["rule"])
    require(s4.state(unit) == state, "restore lost readout fields")
    return unit, body["state"]["replay"]


def rebuild_rows(body, stimuli):
    from snn_data_lake.instruments import taught_problems as tp
    from snn_data_lake.instruments.g5_part_d_cells import _Drive

    result = {}
    for task in TASKS:
        drive = _Drive(body["seed"], "carry", tp.PROBLEMS[task])
        require(s4.param_state(drive._pool) == body["pool"][task], "pool recipe/topology mismatch")
        result[task] = {}
        for split, rows in stimuli[task].items():
            rebuilt = []
            for row in rows:
                turns = tuple(tp.Turn(t, role, None) for t, role in zip(row["texts"], row["roles"], strict=True))
                x = drive.walk(turns)[-1].x
                require(hashed(x) == row["x_sha256"], "rebuilt feature mismatch")
                rebuilt.append({"x": x})
            result[task][split] = rebuilt
        require(s4.param_state(drive._pool) == body["pool"][task], "restore changed pool parameters")
    return result


def prepare(seed):
    from snn_data_lake.instruments import taught_problems as tp
    from snn_data_lake.instruments.g5_part_d_cells import _Drive

    data = {"seed": seed, "tasks": {}, "pool": {}}
    for task in TASKS:
        problem = tp.PROBLEMS[task]
        drive = _Drive(seed, "carry", problem)
        before = s4.param_state(drive._pool)

        def row(episode):
            return {"texts": [t.text for t in episode], "roles": [t.role for t in episode],
                    "target": episode[-1].label, "x": drive.walk(episode)[-1].x, "task": task}

        epochs = [[row(ep) for ep in epoch] for epoch in problem.teach_epochs(seed, EPOCHS)]
        block = {"epochs": epochs, "dev": [row(ep) for ep in problem.episodes(seed, "validation")],
                 "final": [row(ep) for ep in problem.episodes(seed, "future")]}
        require(s4.param_state(drive._pool) == before, "pool changed")
        data["tasks"][task], data["pool"][task] = block, before
        if task == "ac1d":
            sets = [set(dumps(r["texts"]) for r in rows) for rows in (epochs[0], block["dev"], block["final"])]
            require(all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "ac1d split leakage")
        old = json.loads((HERE.parent / f"04-artifacts/pilot-2/features-{seed}.json").read_text())["result"]
        expected = old["tasks"][task]["modes"]["carry"]
        require(expected["parameters_before"] == before, "stage04 pool mismatch")
        for split, rows in (("train", [r for e in epochs for r in e]), ("dev", block["dev"]), ("final", block["final"])):
            comparable = [{k: r[k] for k in ("texts", "target", "x")} for r in rows]
            require(comparable == expected["paths"][split], "stage04 feature mismatch")
    train = [r for task in TASKS for r in data["tasks"][task]["epochs"][0]]
    by_x = {}
    for row in train:
        key = hashed(row["x"])
        require(key not in by_x or by_x[key] == row["target"], "conflicting training labels")
        by_x[key] = row["target"]
    require(len({dumps(r["texts"]) for r in train}) == len(train), "A/B episodes collide")
    return data


def joint(data):
    from snn_data_lake.instruments.g5_part_d_adaptors import delta

    unit = delta(["GOAL", "GREET"], 256)
    for e in range(EPOCHS):
        for task in TASKS:
            for row in data["tasks"][task]["epochs"][e]:
                unit.learn(row["x"], row["target"])
    reads = read_both(unit, [], data, "dev")
    scores = {t: score(reads[t]["answers"], data["tasks"][t]["dev"]) for t in TASKS}
    require(all(v["accuracy"] == 1.0 for v in scores.values()), "joint positive failed; do not run sleep arms")
    return {"seed": data["seed"], "reads": reads, "scores": scores, "state": s4.state(unit)}


def train(data, arm):
    from snn_data_lake.instruments.g5_part_d_adaptors import delta
    from snn_data_lake.instruments.taught_campaign import _shuffled_labels
    from snn_data_lake.substrate.home import save_named_state, load_named_state

    unit, replay = delta(["GOAL", "GREET"], 256), []
    save_named_state("stage05-foreign", b"preserve-this-tenant", path=str(lake_file()))
    result = {"seed": data["seed"], "arm": arm, "reads": {}, "states": {}, "checkpoints": {}}
    result["reads"]["before"] = read_both(unit, replay, data, "dev")
    for task, phase in zip(TASKS, ("after_a", "after_b"), strict=True):
        for epoch in data["tasks"][task]["epochs"]:
            for row in epoch:
                unit.learn(row["x"], row["target"])
        replay.extend(copy.deepcopy(data["tasks"][task]["epochs"][0]))
        result["checkpoints"][phase] = save_checkpoint(unit, replay, data, phase)
        result["reads"][phase] = read_both(unit, replay, data, "dev")
        result["states"][phase] = snapshot(unit, replay)
    prior = snapshot(unit, replay)
    start, cpu = time.monotonic(), time.process_time()
    rows = replay * SLEEP_PASSES
    labels = [r["target"] for r in rows]
    work = {"presentations": 0, "solves": 0}
    if arm == "replay":
        for row in rows:
            unit.learn(row["x"], row["target"])
        work["presentations"] = len(rows)
    elif arm in ("batch", "batch_shuffle"):
        if arm == "batch_shuffle":
            labels = _shuffled_labels(data["seed"])(labels)
        work = batch_update(unit, rows, labels)
    else:
        require(arm == "none", "unknown arm")
        require(prior == snapshot(unit, replay), "no-update changed state")
    result["offline"] = {**work, "wall_s": time.monotonic() - start,
                         "cpu_s": time.process_time() - cpu, "labels": labels,
                         "rows_sha256": hashed(rows)}
    result["checkpoints"]["after_sleep"] = save_checkpoint(unit, replay, data, "after_sleep")
    result["reads"]["after_sleep"] = read_both(unit, replay, data, "dev")
    result["reads"]["final"] = read_both(unit, replay, data, "final")
    result["states"]["after_sleep"] = snapshot(unit, replay)
    require(replay == prior["replay"], "offline changed replay membership")
    require(load_named_state("stage05-foreign", path=str(lake_file())) == b"preserve-this-tenant", "foreign record lost")
    result["lake_sha256"] = lake_hash()
    return result


def restored(stimuli, expected):
    from snn_data_lake.substrate.home import load_named_state

    disk = lake_hash()
    blob = load_named_state(STATE_NAME, path=str(lake_file()))
    body = decode_checkpoint(blob, expected)
    unit, replay = restore_unit(body)
    rows = rebuild_rows(body, stimuli)
    answers = {split: {t: read(unit, replay, rows[t][split]) for t in TASKS} for split in ("dev", "final")}
    require(disk == lake_hash(), "restore read changed lake bytes")
    require(load_named_state("stage05-foreign", path=str(lake_file())) == b"preserve-this-tenant", "foreign tenant lost")
    return {"state": snapshot(unit, replay), "reads": answers, "lake_sha256": disk, "seed": body["seed"]}


def controls():
    from snn_data_lake.instruments.g5_part_d_adaptors import delta
    from snn_data_lake.substrate.home import save_named_state, load_named_state

    reds = {}

    def caught(name, fn):
        try:
            fn()
        except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            reds[name] = True
        else:
            reds[name] = False

    rows = [{"x": [1., 0., 0., 0.], "target": "GOAL"}, {"x": [0., 1., 0., 0.], "target": "GREET"}]
    unit, replay = delta(["GOAL", "GREET"], 4), []
    batch_update(unit, rows * 10)
    require(score(read(unit, replay, rows)["answers"], rows)["accuracy"] == 1., "batch positive failed")
    unit.beta *= -1
    require(score(read(unit, replay, rows)["answers"], rows)["accuracy"] == 0., "batch sign control failed")
    original = type(unit)._answer
    for mode in ("learn", "beta", "replay"):
        def broken(self, x):
            if mode == "learn":
                self.learn(x, "GOAL")
            elif mode == "beta":
                self.beta[0, 0] += 1
            else:
                replay.append({"x": x})
            return original(self, x)
        with patch.object(type(unit), "_answer", broken):
            caught("evaluation_" + mode, lambda: read(unit, replay, rows))
    caught("wrong_response_count", lambda: score([], rows))
    fake = {"seed": 8900, "pool": {}}
    a = encode_checkpoint(unit, replay, fake, "after_a")
    b = encode_checkpoint(unit, replay, fake, "after_sleep")
    require(decode_checkpoint(b, digest(b))["phase"] == "after_sleep", "checkpoint positive failed")
    caught("missing_checkpoint", lambda: decode_checkpoint(None, digest(b)))
    caught("missing_latest_write", lambda: decode_checkpoint(a, digest(b)))
    corrupt = bytearray(b)
    corrupt[-3] ^= 1
    caught("corrupt_checkpoint", lambda: decode_checkpoint(bytes(corrupt), digest(b)))
    env = json.loads(b)
    env["body"]["phase"] = "tampered"
    altered = dumps(env).encode("ascii")
    caught("internal_checksum", lambda: decode_checkpoint(altered, digest(altered)))
    save_named_state(STATE_NAME, b, path=str(lake_file()))
    require(load_named_state(STATE_NAME, path=str(lake_file())) == b, "lake positive failed")
    require(all(reds.values()), "constructed guard did not fire")
    return {"reds": reds, "batch_positive": 1., "batch_sign_control": 0.}


def worker(args):
    s4.stage03.runtime()
    from snn_data_lake.substrate.home import require_isolated_home
    require_isolated_home()
    import torch
    import numpy as np
    torch.set_num_threads(1)
    start, cpu = time.monotonic(), time.process_time()
    data = json.loads(Path(args.inputs).read_text()) if args.inputs else None
    if args.worker == "prepare":
        result = prepare(args.seed)
    elif args.worker == "joint":
        result = joint(data)
    elif args.worker == "train":
        result = train(data, args.arm)
    elif args.worker == "restore":
        result = restored(data, args.expected)
    else:
        result = controls()
    require(not sys._is_gil_enabled(), "GIL enabled after worker")
    print(dumps({"result": result, "cost": {"wall_s": time.monotonic() - start,
          "cpu_s": time.process_time() - cpu, "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
          "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "gil": False}}))


def run(args):
    s4.checkout()
    out = HERE / args.run_id
    require(Path(args.run_id).name == args.run_id, "simple new run-id required")
    out.mkdir()
    manifest = {"schema": "sndl.stage05.run.v1", "status": "RUNNING", "head": HEAD, "runs": [],
                "seeds": list(SEEDS), "arms": list(ARMS), "epochs": EPOCHS, "sleep_passes": SLEEP_PASSES}
    sources = [HERE / "evaluate.py", HERE / "protocol.md", HERE / "recompute.py",
               HERE.parent / "04-artifacts/evaluate.py", HERE.parent / "03-artifacts/evaluate.py"]
    sources += sorted((REPO / "src/snn_data_lake").rglob("*.py"))
    manifest["sources"] = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sources}
    for p in sources[:5]:
        dest = out / "provenance" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    started = time.monotonic()

    def job(key, kind, home, inputs=None, seed=8000, arm=None, expected=None, fail=False):
        require(time.monotonic() - started < 1200, "whole block time limit")
        command = [sys.executable, "-B", str(HERE / "evaluate.py"), "--worker", kind, "--seed", str(seed)]
        for flag, value in (("--inputs", inputs), ("--arm", arm), ("--expected", expected)):
            if value is not None:
                command += [flag, str(value)]
        env = os.environ.copy()
        for name in ("PYTHON_GIL", "SNN_ALLOW_GIL"):
            env.pop(name, None)
        env.update(PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", NORD_HOME=str(home),
                   SNDL_LIVE_HOME_GUARD="1", PYTHONUTF8="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        t = time.monotonic()
        process = subprocess.run(command, cwd=REPO, env=env, capture_output=True, timeout=120)
        (out / f"{key}.json").write_bytes(process.stdout)
        (out / f"{key}.stderr.txt").write_bytes(process.stderr)
        record = {"key": key, "command": command, "returncode": process.returncode,
                  "expected_failure": fail, "home": str(home), "wall_s": time.monotonic() - t,
                  "stdout_sha256": digest(process.stdout), "stderr_sha256": digest(process.stderr)}
        manifest["runs"].append(record)
        save(out / "manifest.json", manifest)
        require((process.returncode != 0) if fail else (process.returncode == 0), f"worker {key} unexpected exit")
        if fail:
            return None
        require(not process.stderr, f"worker {key} stderr")
        return json.loads(process.stdout)["result"]

    try:
        with tempfile.TemporaryDirectory(prefix="sndl-stage05-checks-") as h:
            job("controls", "controls", Path(h))
        for seed in SEEDS:
            for repeat in (False, True) if seed == 8000 else (False,):
                suffix = "-repeat" if repeat else ""
                with tempfile.TemporaryDirectory(prefix="sndl-stage05-features-") as h:
                    data = job(f"features-{seed}{suffix}", "prepare", Path(h), seed=seed)
                inputs = out / f"inputs-{seed}{suffix}.json"
                save(inputs, data)
                stimuli = {t: {s: [{"texts": r["texts"], "roles": r["roles"], "x_sha256": hashed(r["x"])}
                            for r in data["tasks"][t][s]] for s in ("dev", "final")} for t in TASKS}
                eval_inputs = out / f"stimuli-{seed}{suffix}.json"
                save(eval_inputs, stimuli)
                with tempfile.TemporaryDirectory(prefix="sndl-stage05-joint-") as h:
                    job(f"joint-{seed}{suffix}", "joint", Path(h), inputs, seed)
                for arm in ARMS:
                    key = f"{arm}-{seed}{suffix}"
                    home = out / "homes" / key
                    home.mkdir(parents=True)
                    trained = job(key, "train", home, inputs, seed, arm)
                    expected = trained["checkpoints"]["after_sleep"]
                    loaded = job(f"restore-{key}", "restore", home, eval_inputs, seed, expected=expected)
                    require(loaded["state"] == trained["states"]["after_sleep"], "new process complete state mismatch")
                    for split, old_split in (("dev", "after_sleep"), ("final", "final")):
                        for task in TASKS:
                            require(loaded["reads"][split][task]["answers"] == trained["reads"][old_split][task]["answers"], "new process answers mismatch")
                    if seed == 8000 and arm == "batch" and not repeat:
                        integrity(out, home, expected, eval_inputs, job)
        for arm in ARMS:
            a = json.loads((out / f"{arm}-8000.json").read_text())["result"]
            b = json.loads((out / f"{arm}-8000-repeat.json").read_text())["result"]
            # Disk digests include the same deterministic envelope; timing is the only removed field.
            for row in (a, b):
                row["offline"].pop("wall_s")
                row["offline"].pop("cpu_s")
            require(a == b, f"repeat differs: {arm}")
        require(json.loads((out / "inputs-8000.json").read_text()) == json.loads((out / "inputs-8000-repeat.json").read_text()), "feature repeat differs")
        manifest["status"] = "COMPLETE"
        manifest["wall_s"] = time.monotonic() - started
        s4.checkout()
        save(out / "manifest.json", manifest)
        print(dumps({"status": "COMPLETE", "run": str(out), "wall_s": manifest["wall_s"]}))
    except BaseException as exc:
        manifest.update(status="HALT", error=f"{type(exc).__name__}: {exc}", wall_s=time.monotonic() - started)
        save(out / "manifest.json", manifest)
        raise


def integrity(out, home, expected, inputs, job):
    # Each damaged copy has its own NORD_HOME. The valid original is never edited.
    raw = (home / "brain_lake.bin").read_bytes()
    import struct
    offsets, pos = [], 8
    while pos < len(raw):
        offsets.append(pos)
        size = struct.unpack_from("<I", raw, pos)[0]
        pos += 4 + size
    require(pos == len(raw) and len(offsets) == 4, "unexpected lake framing")
    for mode in ("missing", "stale", "corrupt", "recipe"):
        damaged = out / "homes" / f"damaged-{mode}"
        damaged.mkdir()
        target = damaged / "brain_lake.bin"
        if mode == "missing":
            target.write_bytes(raw[:8])
        elif mode == "stale":
            target.write_bytes(raw[:offsets[-1]])
        elif mode == "corrupt":
            bad = bytearray(raw)
            bad[-12] ^= 1
            target.write_bytes(bad)
        else:
            # Change a valid envelope, including checksum/receipt, so reconstruction must catch it.
            size = struct.unpack_from("<I", raw, offsets[-1])[0]
            payload = raw[offsets[-1] + 4:offsets[-1] + 4 + size]
            n = payload[1]
            blob = json.loads(payload[2 + n + 4:])
            blob["body"]["pool"][TASKS[0]]["fingerprint"] = "bad-recipe"
            blob["sha256"] = hashed(blob["body"])
            changed = dumps(blob).encode("ascii")
            new_payload = payload[:2 + n + 4] + changed
            target.write_bytes(raw[:offsets[-1]] + struct.pack("<I", len(new_payload)) + new_payload)
            expected_recipe = digest(changed)
        job(f"integrity-{mode}", "restore", damaged, inputs,
            expected=expected_recipe if mode == "recipe" else expected, fail=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--worker", choices=("controls", "prepare", "joint", "train", "restore"))
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--inputs")
    parser.add_argument("--expected")
    args = parser.parse_args()
    worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
