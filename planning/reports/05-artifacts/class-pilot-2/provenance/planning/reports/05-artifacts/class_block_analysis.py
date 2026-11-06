"""Stdlib-only independent scoring of the class-order pilot; no SNDL imports."""

import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import struct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage05_reader", HERE / "recompute.py")
r5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r5)
require, hashed, sha, dumps = r5.require, r5.hashed, r5.sha, r5.dumps
SEEDS = (8000, 8001, 8002, 8003)
BUDGETS = (30, 100, 300)
ARMS = ("none", "replay", "batch", "batch_shuffle")
PROFILE = "sndl.stage05.class-block.v1"


def scores(answers, rows):
    require(len(answers) == len(rows), "answer count")
    result = {}
    for g, label in (("A", "GREET"), ("B", "GOAL")):
        indices = [i for i, row in enumerate(rows) if row["target"] == label]
        require(bool(indices), "empty group")
        n = sum(answers[i] == label for i in indices)
        result[g] = {"correct": n, "n": len(indices), "accuracy": n / len(indices)}
    result["macro"] = sum(result[g]["accuracy"] for g in ("A", "B")) / 2
    return result


def reading(read, state, rows):
    require(read["state_sha256"] == hashed(state), "read state mismatch")
    answers, _ = r5.predicted(state, rows)
    require(read["answers"] == answers, "answers inconsistent with beta")
    return scores(answers, rows)


def choose(details):
    require(set(details) == {str(n) for n in BUDGETS}, "missing budget")
    for n in BUDGETS:
        cases = details[str(n)]
        require(set(cases) == {str(s) for s in SEEDS}, "missing seed")
    for n in BUDGETS:
        if all(v["a_before"] >= .9 and v["b_after"] >= .9 and v["forget_a"] >= .1 for v in details[str(n)].values()):
            return n
    return None


def validate_discovery(data, raw):
    require(raw["seed"] == data["seed"], "discovery identity")
    a_rows = [r for r in data["train"] if r["target"] == "GREET"]
    b_rows = [r for r in data["train"] if r["target"] == "GOAL"]
    require(len(a_rows) == len(b_rows) == 16 and len(data["orders"]) == 300, "train budget")
    require(all(sorted(order) == list(range(32)) for order in data["orders"]), "epoch membership")
    sets = [{dumps(r["texts"]) for r in data[split]} for split in ("train", "dev", "final")]
    require([len(v) for v in sets] == [32, 8, 12] and all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "split overlap/count")
    require([row["epoch"] for row in raw["joint"]] == list(range(21, 31)), "joint read windows")
    for row in raw["joint"]:
        require(reading(row["read"], row["state"], data["dev"]) == row["scores"], "joint score mismatch")
        require(row["scores"]["A"]["accuracy"] == row["scores"]["B"]["accuracy"] == 1., "joint positive failed")
        require(row["state"]["readout"]["rule"]["calls"] == row["epoch"] * 32, "joint update count")
    after_a = raw["after_a"]
    require(after_a["state"]["replay"] == a_rows, "after-A memory")
    require(after_a["state"]["readout"]["rule"]["calls"] == 480, "A update count")
    require(reading(after_a["read"], after_a["state"], data["dev"]) == after_a["scores"], "after-A score")
    require(set(raw["looks"]) == {str(v) for v in BUDGETS}, "discovery look count")
    details = {}
    for n in BUDGETS:
        row = raw["looks"][str(n)]
        require(row["state"]["replay"] == a_rows + b_rows, "look memory")
        rule = row["state"]["readout"]["rule"]
        require(rule["calls"] == 480 + n * 16 and rule["widened"] == 0, "B update count")
        require(reading(row["read"], row["state"], data["dev"]) == row["scores"], "look score mismatch")
        before = after_a["scores"]["A"]["accuracy"]
        details[str(n)] = {"a_before": before, "a_after": row["scores"]["A"]["accuracy"],
                           "b_after": row["scores"]["B"]["accuracy"], "forget_a": before - row["scores"]["A"]["accuracy"]}
    return details


def lake(raw):
    require(raw[:8] == b"SNDLBIN1", "lake magic")
    pos, states, foreign = 8, {}, False
    while pos < len(raw):
        require(pos + 4 <= len(raw), "torn frame")
        n = struct.unpack_from("<I", raw, pos)[0]
        payload = raw[pos + 4:pos + 4 + n]
        require(len(payload) == n and payload[:1] == b"\x07", "lake frame/tag")
        pos += 4 + n
        k = payload[1]
        name, blob = payload[2:2+k].decode("ascii"), payload[2+k+4:]
        require(name not in states, "duplicate tenant")
        if name == "class-block-foreign":
            require(blob == b"preserve-this-tenant", "foreign tenant mutation")
            foreign = True
        else:
            env = json.loads(blob)
            body = env["body"]
            require(hashed(body) == env["sha256"] and body["schema"] == PROFILE, "checkpoint checksum/profile")
            require(name == "stage05-class-block-" + body["phase"], "tenant phase mismatch")
            states[name] = {"blob_sha256": sha(blob), "body": body}
    require(foreign, "foreign tenant missing")
    return states


def validate_arm(data, discovery, budget, arm, raw, restored):
    require(raw["seed"] == data["seed"] == restored["seed"] and raw["arm"] == arm == restored["arm"], "arm identity")
    require(raw["budget"] == restored["budget"] == budget, "selected budget")
    before = discovery["looks"][str(budget)]
    require(raw["before"] == before["state"] and raw["input_receipt"] == before["receipt"], "arm initial state")
    require(raw["after"]["replay"] == raw["before"]["replay"], "memory changed/leaked")
    require(raw["after"] == restored["state"] and raw["reads"] == restored["reads"], "restart state/answers differ")
    require(raw["lake_sha256"] == restored["lake_sha256"], "restart wrote lake")
    rows = [data["train"][i] for order in data["orders"][:10] for i in order]
    require(raw["work"]["rows_sha256"] == hashed(rows), "offline row mismatch")
    true = [row["target"] for row in rows]
    if arm == "batch_shuffle":
        require(Counter(raw["work"]["labels"]) == Counter(true) and raw["work"]["labels"] != true, "shuffle labels")
    else:
        require(raw["work"]["labels"] == true, "teaching labels changed")
    require(raw["work"]["presentations"] == (0 if arm == "none" else 320), "offline presentation budget")
    require(raw["work"]["solves"] == int(arm.startswith("batch")), "offline solve budget")
    require(raw["after"]["readout"]["rule"]["calls"] == 480 + 16 * budget + (320 if arm == "replay" else 0), "offline update count")
    if arm == "none":
        require(raw["after"] == raw["before"], "no-update mutation")
    else:
        require(raw["after"]["readout"]["beta"] != raw["before"]["readout"]["beta"], "offline beta inert")
    require(set(raw["reads"]) == {"dev", "final"}, "missing split")
    for split in ("dev", "final"):
        require(reading(raw["reads"][split], raw["after"], data[split]) == raw["scores"][split], "arm score mismatch")
    return {"scores": raw["scores"], "work": {k: v for k, v in raw["work"].items() if k != "labels"}}


def analyze(records):
    load = lambda name: json.loads((records / f"{name}.json").read_text())
    result = lambda name: load(name)["result"]
    manifest = load("manifest")
    require(manifest["status"] in ("COMPLETE", "NO_SENSITIVE_REGIME"), "incomplete run")
    require(manifest["schema"] == PROFILE and manifest["seeds"] == list(SEEDS) and manifest["budgets"] == list(BUDGETS) and manifest["arms"] == list(ARMS), "manifest design")
    wanted = {"controls"}
    for seed in SEEDS:
        for suffix in ("", "-repeat") if seed == 8000 else ("",):
            wanted |= {f"features-{seed}{suffix}", f"discovery-{seed}{suffix}"}
            if manifest["status"] == "COMPLETE":
                wanted |= {f"{prefix}{arm}-{seed}{suffix}" for prefix in ("", "restore-") for arm in ARMS}
    require(len(manifest["runs"]) == len(wanted) and {v["key"] for v in manifest["runs"]} == wanted, "missing/duplicate worker")
    for row in manifest["runs"]:
        stdout, stderr = (records / f"{row['key']}.json").read_bytes(), (records / f"{row['key']}.stderr.txt").read_bytes()
        require(row["returncode"] == 0 and not stderr and sha(stdout) == row["stdout_sha256"] and sha(stderr) == row["stderr_sha256"], "worker transport")
    controls = result("controls")
    require(all(controls["previous_guards"]["reds"].values()) and all(controls["new_reds"].values()), "constructed controls")
    details, discovery_rows = {str(n): {} for n in BUDGETS}, {}
    for seed in SEEDS:
        data, raw = load(f"inputs-{seed}"), result(f"discovery-{seed}")
        require(data == result(f"features-{seed}"), "input copy differs")
        own = validate_discovery(data, raw)
        for n, values in own.items():
            details[n][str(seed)] = values
        states = lake((records / "homes" / f"discovery-{seed}/brain_lake.bin").read_bytes())
        require(len(states) == 4, "discovery checkpoint count")
        for phase, reading_row in [("after_a", raw["after_a"])] + [(f"b{n}", raw["looks"][str(n)]) for n in BUDGETS]:
            stored = states["stage05-class-block-" + phase]
            require(stored["blob_sha256"] == reading_row["receipt"] and stored["body"]["state"] == reading_row["state"] and stored["body"]["pool"] == data["pool"], "discovery lake state")
        discovery_rows[str(seed)] = {"after_a": raw["after_a"]["scores"], "looks": {n: r["scores"] for n, r in raw["looks"].items()}}
    selection = load("selection")
    require(sha((records / "selection.json").read_bytes()) == manifest["selection_sha256"], "selection receipt")
    require(selection["all_looks"] == details and selection["selected_b_epochs"] == choose(details), "selection rule")
    selected = selection["selected_b_epochs"]
    require((selected is not None) == (manifest["status"] == "COMPLETE"), "selection/status mismatch")
    require(result("discovery-8000") == result("discovery-8000-repeat") and load("inputs-8000") == load("inputs-8000-repeat"), "discovery repeat")
    summary = {"schema": PROFILE, "status": manifest["status"], "head": manifest["head"], "selected_b_epochs": selected,
               "selection": selection, "discovery": discovery_rows, "cases": {}, "contrasts": {}, "controls": controls,
               "scope": "finite exposed class-order diagnostic; no population claim"}
    if selected is not None:
        for seed in SEEDS:
            data, discovery = load(f"inputs-{seed}"), result(f"discovery-{seed}")
            for arm in ARMS:
                raw, restored = result(f"{arm}-{seed}"), result(f"restore-{arm}-{seed}")
                summary["cases"][f"{arm}/{seed}"] = validate_arm(data, discovery, selected, arm, raw, restored)
                blob = (records / "homes" / f"{arm}-{seed}/brain_lake.bin").read_bytes()
                require(sha(blob) == raw["lake_sha256"], "lake bytes changed")
                states = lake(blob)
                require(len(states) == 5, "offline lake record count")
                body = states["stage05-class-block-result"]
                require(body["blob_sha256"] == raw["receipt"] and body["body"]["state"] == raw["after"] and body["body"]["pool"] == data["pool"], "offline lake state")
        for arm, base in (("replay", "none"), ("batch", "none"), ("batch", "replay"), ("batch", "batch_shuffle")):
            contrasts = {}
            for g in ("A", "B", "macro"):
                vals = []
                for seed in SEEDS:
                    a, b = (summary["cases"][f"{v}/{seed}"]["scores"]["final"] for v in (arm, base))
                    vals.append(a[g] - b[g] if g == "macro" else a[g]["accuracy"] - b[g]["accuracy"])
                contrasts[g] = {"paired": vals, "mean": statistics.mean(vals), "sample_sd": statistics.stdev(vals)}
            summary["contrasts"][f"{arm}-{base}"] = contrasts
        for arm in ARMS:
            a, b = result(f"{arm}-8000"), result(f"{arm}-8000-repeat")
            for v in (a, b):
                v["work"].pop("wall_s")
                v["work"].pop("cpu_s")
            require(a == b, "offline repeat")
    summary["cost"] = {"block_wall_s": manifest["wall_s"], "subprocess_wall_s": sum(r["wall_s"] for r in manifest["runs"]),
                       "workers": len(manifest["runs"]), "raw_stdout_bytes": sum((records / f"{r['key']}.json").stat().st_size for r in manifest["runs"])}
    return summary


def sensitivity(records, summary):
    synthetic = {str(n): {str(s): {"a_before": 1., "a_after": .75, "b_after": 1., "forget_a": .25} for s in SEEDS} for n in BUDGETS}
    require(choose(synthetic) == 30, "selector positive")
    null = copy.deepcopy(synthetic)
    for block in null.values():
        for row in block.values():
            row.update(a_after=1., forget_a=0.)
    require(choose(null) is None, "selector zero-effect control")
    reds = {"selector_positive": True, "selector_zero_effect": True}
    load = lambda name: json.loads((records / f"{name}.json").read_text())
    data, raw = load("inputs-8000"), load("discovery-8000")["result"]
    for kind in ("joint", "answer", "memory", "budget", "missing_look", "split"):
        d, r = copy.deepcopy((data, raw))
        if kind == "joint":
            r["joint"][0]["scores"]["A"]["accuracy"] = 0.
        elif kind == "answer":
            r["looks"]["30"]["read"]["answers"][0] = "bad"
        elif kind == "memory":
            r["looks"]["30"]["state"]["replay"].append(d["dev"][0])
        elif kind == "budget":
            r["looks"]["100"]["state"]["readout"]["rule"]["calls"] -= 1
        elif kind == "missing_look":
            del r["looks"]["300"]
        else:
            d["final"][0] = d["train"][0]
        try:
            validate_discovery(d, r)
        except (ValueError, KeyError, TypeError):
            reds[kind] = True
        else:
            reds[kind] = False
    if summary["selected_b_epochs"] is not None:
        arm, restored = load("batch-8000")["result"], load("restore-batch-8000")["result"]
        for kind in ("restore", "offline_answer", "offline_budget", "offline_memory"):
            a, r = copy.deepcopy((arm, restored))
            if kind == "restore":
                r["state"]["readout"]["beta"][0][0] += 1.
            elif kind == "offline_answer":
                a["reads"]["final"]["answers"][0] = "bad"
            elif kind == "offline_budget":
                a["work"]["presentations"] -= 1
            else:
                a["after"]["replay"].append(data["dev"][0])
            try:
                validate_arm(data, raw, summary["selected_b_epochs"], "batch", a, r)
            except (ValueError, KeyError, TypeError):
                reds[kind] = True
            else:
                reds[kind] = False
    require(all(reds.values()), "offline sensitivity failed")
    return reds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = analyze(args.records)
    reds = sensitivity(args.records, summary)
    args.output.mkdir()
    for name, value in (("summary", summary), ("sensitivity", reds)):
        (args.output / f"{name}.json").write_text(dumps(value) + "\n", encoding="ascii")
    lines = ["| Seed | After A: A | B30: A/B | B100: A/B | B300: A/B |", "|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        row = summary["discovery"][str(seed)]
        cell = lambda s: "/".join(f"{s[g]['accuracy']:.0%}" for g in ("A", "B"))
        lines.append(f"| {seed} | {row['after_a']['A']['accuracy']:.0%} | " + " | ".join(cell(row["looks"][str(n)]) for n in BUDGETS) + " |")
    if summary["cases"]:
        lines += ["", "| Seed | Arm | A final | B final | Macro |", "|---:|---|---:|---:|---:|"]
        for seed in SEEDS:
            for arm in ARMS:
                row = summary["cases"][f"{arm}/{seed}"]["scores"]["final"]
                lines.append(f"| {seed} | {arm} | {row['A']['correct']}/{row['A']['n']} | {row['B']['correct']}/{row['B']['n']} | {row['macro']:.1%} |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    print(dumps({"status": "VALID", "selected_b_epochs": summary["selected_b_epochs"],
                 "summary_sha256": sha((args.output / "summary.json").read_bytes()), "sensitivity": reds}))


if __name__ == "__main__":
    main()
