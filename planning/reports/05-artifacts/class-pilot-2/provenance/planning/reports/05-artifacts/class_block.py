"""Bounded class-order continuation; all pool/learning/storage mechanics are adopted."""

import argparse
import copy
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT / "repo"
spec = importlib.util.spec_from_file_location("stage05", HERE / "evaluate.py")
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)
save, dumps, digest, hashed, require = s5.save, s5.dumps, s5.digest, s5.hashed, s5.require
SEEDS, ARMS = s5.SEEDS, s5.ARMS
BUDGETS = (30, 100, 300)
PROFILE = "sndl.stage05.class-block.v1"
PREFIX = "stage05-class-block-"
GROUPS = ("A", "B")


def group(label):
    require(label in ("GREET", "GOAL"), "unknown label")
    return "A" if label == "GREET" else "B"


def scores(answers, rows):
    require(len(answers) == len(rows), "answer count")
    result = {}
    for g in GROUPS:
        indices = [i for i, row in enumerate(rows) if group(row["target"]) == g]
        require(bool(indices), "empty group")
        correct = sum(answers[i] == rows[i]["target"] for i in indices)
        result[g] = {"correct": correct, "n": len(indices), "accuracy": correct / len(indices)}
    result["macro"] = sum(result[g]["accuracy"] for g in GROUPS) / 2
    return result


def envelope(body):
    return dumps({"body": body, "sha256": hashed(body)}).encode("ascii")


def checkpoint(unit, replay, data, phase, **extra):
    from snn_data_lake.substrate.home import save_named_state

    body = {"schema": PROFILE, "head": s5.HEAD, "seed": data["seed"], "phase": phase,
            "pool": data["pool"], "state": s5.snapshot(unit, replay), **extra}
    blob = envelope(body)
    require(save_named_state(PREFIX + phase, blob, path=str(s5.lake_file())), "checkpoint not written")
    return digest(blob)


def decode(blob, expected):
    require(blob is not None, "missing checkpoint")
    require(digest(blob) == expected, "checkpoint receipt mismatch")
    env = json.loads(blob)
    body = env["body"]
    require(hashed(body) == env["sha256"], "checkpoint checksum")
    require(body["schema"] == PROFILE and body["head"] == s5.HEAD, "checkpoint profile/version")
    return body


def load_checkpoint(phase, expected):
    from snn_data_lake.substrate.home import load_named_state

    body = decode(load_named_state(PREFIX + phase, path=str(s5.lake_file())), expected)
    require(body["phase"] == phase, "checkpoint phase mismatch")
    return body


def rebuild(body, stimuli):
    from snn_data_lake.instruments import taught_problems as tp
    from snn_data_lake.instruments.g5_part_d_cells import _Drive

    drive = _Drive(body["seed"], "carry", tp.PROBLEMS["ac1d"])
    require(s5.s4.param_state(drive._pool) == body["pool"], "pool recipe/topology mismatch")
    rows = {}
    for split, items in stimuli.items():
        rows[split] = []
        for item in items:
            turns = tuple(tp.Turn(text, role, None) for text, role in zip(item["texts"], item["roles"], strict=True))
            x = drive.walk(turns)[-1].x
            require(hashed(x) == item["x_sha256"], "rebuilt feature mismatch")
            rows[split].append({"x": x})
    require(s5.s4.param_state(drive._pool) == body["pool"], "pool parameters changed")
    return rows


def prepare(seed):
    from snn_data_lake.instruments import taught_problems as tp
    from snn_data_lake.instruments.g5_part_d_cells import _Drive

    problem = tp.PROBLEMS["ac1d"]
    drive = _Drive(seed, "carry", problem)
    pool = s5.s4.param_state(drive._pool)
    data = {"seed": seed, "pool": pool, "train": [], "orders": []}
    index = {}

    def row(ep):
        return {"texts": [t.text for t in ep], "roles": [t.role for t in ep],
                "target": ep[-1].label, "x": drive.walk(ep)[-1].x}

    for epoch in problem.teach_epochs(seed, max(BUDGETS)):
        order = []
        for ep in epoch:
            key = dumps([t.text for t in ep])
            if key not in index:
                index[key] = len(data["train"])
                data["train"].append(row(ep))
            order.append(index[key])
        data["orders"].append(order)
    for split, name in (("validation", "dev"), ("future", "final")):
        data[name] = [row(ep) for ep in problem.episodes(seed, split)]
    require(s5.s4.param_state(drive._pool) == pool, "pool changed")
    require(len(data["train"]) == 32 and len(index) == 32, "train count")
    sets = [{dumps(r["texts"]) for r in data[split]} for split in ("train", "dev", "final")]
    require(all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "split leakage")
    old = json.loads((HERE / f"pilot-1/inputs-{seed}.json").read_text())
    require(pool == old["pool"]["ac1d"], "old pool mismatch")
    for split, expected in (("train", old["tasks"]["ac1d"]["epochs"][0]),
                            ("dev", old["tasks"]["ac1d"]["dev"]), ("final", old["tasks"]["ac1d"]["final"])):
        require(data[split] == [{k: r[k] for k in ("texts", "roles", "target", "x")} for r in expected], "old feature mismatch")
    for split, count in (("train", 16), ("dev", 16), ("final", 24)):
        require(all(sum(group(r["target"]) == g for r in data[split]) == count for g in GROUPS), "group count")
    return data


def teach(unit, data, epoch, g=None):
    for i in data["orders"][epoch]:
        row = data["train"][i]
        if g is None or group(row["target"]) == g:
            unit.learn(row["x"], row["target"])


def discover(data):
    from snn_data_lake.instruments.g5_part_d_adaptors import delta
    from snn_data_lake.substrate.home import save_named_state

    unit = delta(["GOAL", "GREET"], 256)
    joint = []
    for epoch in range(30):
        teach(unit, data, epoch)
        if epoch >= 20:
            reading = s5.read(unit, [], data["dev"])
            result = scores(reading["answers"], data["dev"])
            require(all(result[g]["accuracy"] == 1. for g in GROUPS), "joint positive failed")
            joint.append({"epoch": epoch + 1, "read": reading, "scores": result,
                          "state": s5.snapshot(unit, [])})
    save_named_state("class-block-foreign", b"preserve-this-tenant", path=str(s5.lake_file()))
    unit, replay = delta(["GOAL", "GREET"], 256), []
    initial = s5.snapshot(unit, replay)
    for epoch in range(30):
        teach(unit, data, epoch, "A")
    replay.extend(copy.deepcopy([r for r in data["train"] if group(r["target"]) == "A"]))
    receipt = checkpoint(unit, replay, data, "after_a")
    after_a = {"receipt": receipt, "state": s5.snapshot(unit, replay), "read": s5.read(unit, replay, data["dev"])}
    after_a["scores"] = scores(after_a["read"]["answers"], data["dev"])
    replay.extend(copy.deepcopy([r for r in data["train"] if group(r["target"]) == "B"]))
    looks = {}
    for epoch in range(max(BUDGETS)):
        teach(unit, data, epoch, "B")
        if epoch + 1 in BUDGETS:
            n = epoch + 1
            receipt = checkpoint(unit, replay, data, f"b{n}", b_epochs=n)
            reading = s5.read(unit, replay, data["dev"])
            looks[str(n)] = {"receipt": receipt, "state": s5.snapshot(unit, replay), "read": reading,
                             "scores": scores(reading["answers"], data["dev"])}
    return {"seed": data["seed"], "joint": joint, "initial": initial, "after_a": after_a, "looks": looks}


def offline(data, arm, budget, expected):
    from snn_data_lake.instruments.taught_campaign import _shuffled_labels

    body = load_checkpoint(f"b{budget}", expected)
    require(body["seed"] == data["seed"] and body["pool"] == data["pool"], "wrong input checkpoint")
    unit, replay = s5.restore_unit(body)
    require(s5.hashed(replay) == s5.hashed([r for g in GROUPS for r in data["train"] if group(r["target"]) == g]), "wrong memory")
    before = s5.snapshot(unit, replay)
    rows = [data["train"][i] for order in data["orders"][:10] for i in order]
    labels = [r["target"] for r in rows]
    start, cpu = time.monotonic(), time.process_time()
    work = {"presentations": 0, "solves": 0}
    if arm == "replay":
        for row in rows:
            unit.learn(row["x"], row["target"])
        work["presentations"] = len(rows)
    elif arm in ("batch", "batch_shuffle"):
        if arm == "batch_shuffle":
            labels = _shuffled_labels(data["seed"])(labels)
        work = s5.batch_update(unit, rows, labels)
    else:
        require(arm == "none" and before == s5.snapshot(unit, replay), "no-update mutation")
    cost = {"wall_s": time.monotonic() - start, "cpu_s": time.process_time() - cpu}
    require(replay == before["replay"], "offline memory changed")
    receipt = checkpoint(unit, replay, data, "result", b_epochs=budget, arm=arm)
    reads = {split: s5.read(unit, replay, data[split]) for split in ("dev", "final")}
    return {"seed": data["seed"], "arm": arm, "budget": budget, "before": before,
            "after": s5.snapshot(unit, replay), "input_receipt": expected, "receipt": receipt,
            "reads": reads, "scores": {split: scores(reads[split]["answers"], data[split]) for split in reads},
            "work": {**work, **cost, "rows_sha256": hashed(rows), "labels": labels}, "lake_sha256": s5.lake_hash()}


def restore(stimuli, expected):
    from snn_data_lake.substrate.home import load_named_state

    disk = s5.lake_hash()
    body = load_checkpoint("result", expected)
    unit, replay = s5.restore_unit(body)
    rows = rebuild(body, stimuli)
    reads = {split: s5.read(unit, replay, values) for split, values in rows.items()}
    require(disk == s5.lake_hash(), "restore wrote lake")
    require(load_named_state("class-block-foreign", path=str(s5.lake_file())) == b"preserve-this-tenant", "foreign tenant lost")
    return {"seed": body["seed"], "arm": body["arm"], "budget": body["b_epochs"],
            "state": s5.snapshot(unit, replay), "reads": reads, "lake_sha256": disk}


def controls():
    from snn_data_lake.instruments.g5_part_d_adaptors import delta
    from snn_data_lake.instruments.g5_part_d_cells import _Drive
    from snn_data_lake.instruments.taught_problems import PROBLEMS

    old = s5.controls()
    pool = s5.s4.param_state(_Drive(8900, "carry", PROBLEMS["ac1d"])._pool)
    data = {"seed": 8900, "pool": pool}
    unit = delta(["GOAL", "GREET"], 256)
    receipt = checkpoint(unit, [], data, "result", b_epochs=30, arm="none")
    body = load_checkpoint("result", receipt)
    require(s5.snapshot(*s5.restore_unit(body)) == body["state"], "new checkpoint positive")
    require(rebuild(body, {"dev": [], "final": []}) == {"dev": [], "final": []}, "pool positive")
    reds = {}

    def caught(name, fn, reason):
        try:
            fn()
        except (RuntimeError, ValueError) as exc:
            require(reason in str(exc), "wrong constructed failure")
            reds[name] = True
        else:
            reds[name] = False

    blob = envelope(body)
    caught("missing", lambda: decode(None, receipt), "missing checkpoint")
    caught("stale_receipt", lambda: decode(blob, "wrong"), "checkpoint receipt")
    changed = copy.deepcopy(body)
    changed["phase"] = "bad-phase"
    bad = dumps({"body": changed, "sha256": hashed(body)}).encode("ascii")
    caught("checksum", lambda: decode(bad, digest(bad)), "checksum")
    changed = copy.deepcopy(body)
    changed["schema"] = "wrong-profile"
    bad = envelope(changed)
    caught("profile", lambda: decode(bad, digest(bad)), "profile/version")
    changed = copy.deepcopy(body)
    changed["pool"]["fingerprint"] = "wrong-pool"
    caught("real_pool_recipe", lambda: rebuild(changed, {"dev": [], "final": []}), "pool recipe/topology")
    require(all(reds.values()), "new control failed")
    return {"previous_guards": old, "new_reds": reds}


def worker(args):
    s5.s4.stage03.runtime()
    from snn_data_lake.substrate.home import require_isolated_home
    require_isolated_home()
    import torch
    import numpy as np
    torch.set_num_threads(1)
    data = json.loads(Path(args.inputs).read_text()) if args.inputs else None
    start, cpu = time.monotonic(), time.process_time()
    if args.worker == "prepare":
        result = prepare(args.seed)
    elif args.worker == "discover":
        result = discover(data)
    elif args.worker == "offline":
        result = offline(data, args.arm, args.budget, args.expected)
    elif args.worker == "restore":
        result = restore(data, args.expected)
    else:
        result = controls()
    require(not sys._is_gil_enabled(), "GIL enabled after worker")
    print(dumps({"result": result, "cost": {"wall_s": time.monotonic() - start,
           "cpu_s": time.process_time() - cpu, "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
           "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "gil": False}}))


def run(args):
    s5.s4.checkout()
    require(Path(args.run_id).name == args.run_id, "simple new run-id")
    out = HERE / args.run_id
    out.mkdir()
    manifest = {"schema": PROFILE, "head": s5.HEAD, "status": "RUNNING", "seeds": list(SEEDS),
                "budgets": list(BUDGETS), "arms": list(ARMS), "runs": []}
    sources = [HERE / name for name in ("class_block.py", "class_block_analysis.py", "class-block-protocol.md", "evaluate.py")]
    sources += [HERE.parent / "04-artifacts/evaluate.py", HERE.parent / "03-artifacts/evaluate.py", HERE / "recompute.py"]
    manifest["source_tree"] = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", s5.HEAD + ":src/snn_data_lake"], text=True
    ).strip()
    manifest["source_receipt_scope"] = "clean pinned source tree; SHA256 receipts for seven external harness/protocol files"
    manifest["sources"] = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sources}
    for p in sources[:7]:
        dest = out / "provenance" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    start = time.monotonic()

    def job(key, kind, home, inputs=None, seed=8000, arm=None, budget=None, expected=None):
        require(time.monotonic() - start < 1200, "block timeout")
        command = [sys.executable, "-B", str(HERE / "class_block.py"), "--worker", kind, "--seed", str(seed)]
        for flag, value in (("--inputs", inputs), ("--arm", arm), ("--budget", budget), ("--expected", expected)):
            if value is not None:
                command += [flag, str(value)]
        env = os.environ.copy()
        env.update(NORD_HOME=str(home), SNDL_LIVE_HOME_GUARD="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(REPO / "src"), PYTHONUTF8="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        t = time.monotonic()
        p = subprocess.run(command, cwd=REPO, env=env, capture_output=True, timeout=120)
        (out / f"{key}.json").write_bytes(p.stdout)
        (out / f"{key}.stderr.txt").write_bytes(p.stderr)
        manifest["runs"].append({"key": key, "kind": kind, "command": command, "returncode": p.returncode,
                                 "wall_s": time.monotonic() - t, "stdout_sha256": digest(p.stdout), "stderr_sha256": digest(p.stderr)})
        save(out / "manifest.json", manifest)
        require(p.returncode == 0 and not p.stderr, f"worker {key} failed")
        return json.loads(p.stdout)["result"]

    try:
        with tempfile.TemporaryDirectory(prefix="sndl-class-controls-") as h:
            job("controls", "controls", Path(h))
        discoveries, inputs = {}, {}
        for seed in SEEDS:
            for suffix in ("", "-repeat") if seed == 8000 else ("",):
                with tempfile.TemporaryDirectory(prefix="sndl-class-features-") as h:
                    data = job(f"features-{seed}{suffix}", "prepare", Path(h), seed=seed)
                key = f"{seed}{suffix}"
                inputs[key] = out / f"inputs-{key}.json"
                save(inputs[key], data)
                stimuli = {split: [{"texts": row["texts"], "roles": row["roles"], "x_sha256": hashed(row["x"])}
                                   for row in data[split]] for split in ("dev", "final")}
                save(out / f"stimuli-{key}.json", stimuli)
                home = out / "homes" / f"discovery-{key}"
                home.mkdir(parents=True)
                discoveries[key] = job(f"discovery-{key}", "discover", home, inputs[key], seed)
        require(discoveries["8000"] == discoveries["8000-repeat"], "discovery repeat mismatch")
        require(json.loads(inputs["8000"].read_text()) == json.loads(inputs["8000-repeat"].read_text()), "feature repeat mismatch")
        qualifying = []
        details = {}
        for budget in BUDGETS:
            details[str(budget)] = {}
            for seed in SEEDS:
                discovery = discoveries[str(seed)]
                before = discovery["after_a"]["scores"]["A"]["accuracy"]
                after = discovery["looks"][str(budget)]["scores"]
                details[str(budget)][str(seed)] = {"a_before": before, "a_after": after["A"]["accuracy"],
                    "b_after": after["B"]["accuracy"], "forget_a": before - after["A"]["accuracy"]}
            if all(v["a_before"] >= .9 and v["b_after"] >= .9 and v["forget_a"] >= .1 for v in details[str(budget)].values()):
                qualifying.append(budget)
        selected = qualifying[0] if qualifying else None
        selection = {"selected_b_epochs": selected, "all_looks": details,
                     "rule": "first declared budget with A>=.90 before B, B>=.90 and A loss>=.10 on every exposed seed"}
        save(out / "selection.json", selection)
        manifest["selection_sha256"] = digest((out / "selection.json").read_bytes())
        if selected is None:
            manifest["status"] = "NO_SENSITIVE_REGIME"
        else:
            for seed in SEEDS:
                for suffix in ("", "-repeat") if seed == 8000 else ("",):
                    base = f"{seed}{suffix}"
                    for arm in ARMS:
                        key = f"{arm}-{base}"
                        home = out / "homes" / key
                        shutil.copytree(out / "homes" / f"discovery-{base}", home)
                        expected = discoveries[base]["looks"][str(selected)]["receipt"]
                        raw = job(key, "offline", home, inputs[base], seed, arm, selected, expected)
                        require(raw["before"] == discoveries[base]["looks"][str(selected)]["state"], "arm initial state differs")
                        loaded = job(f"restore-{key}", "restore", home, out / f"stimuli-{base}.json", seed, expected=raw["receipt"])
                        require(loaded["state"] == raw["after"] and loaded["reads"] == raw["reads"] and loaded["lake_sha256"] == raw["lake_sha256"], "restart mismatch")
            for arm in ARMS:
                a = json.loads((out / f"{arm}-8000.json").read_text())["result"]
                b = json.loads((out / f"{arm}-8000-repeat.json").read_text())["result"]
                for v in (a, b):
                    v["work"].pop("wall_s")
                    v["work"].pop("cpu_s")
                require(a == b, f"offline repeat differs: {arm}")
            manifest["status"] = "COMPLETE"
        manifest["wall_s"] = time.monotonic() - start
        s5.s4.checkout()
        save(out / "manifest.json", manifest)
        print(dumps({"status": manifest["status"], "selection": selection, "wall_s": manifest["wall_s"]}))
    except BaseException as exc:
        manifest.update(status="HALT", error=f"{type(exc).__name__}: {exc}", wall_s=time.monotonic() - start)
        save(out / "manifest.json", manifest)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id")
    p.add_argument("--worker", choices=("controls", "prepare", "discover", "offline", "restore"))
    p.add_argument("--seed", type=int, default=8000)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--budget", type=int)
    p.add_argument("--inputs")
    p.add_argument("--expected")
    args = p.parse_args()
    worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
