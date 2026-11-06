"""Strict stdlib-only stage-05 offline reader. No SNDL imports or learning."""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

TASKS = ("i.S0C0", "ac1d")
SEEDS = (8000, 8001, 8002, 8003)
ARMS = ("none", "replay", "batch", "batch_shuffle")


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def hashed(value):
    return sha(dumps(value).encode("ascii"))


def require(ok, why):
    if not ok:
        raise ValueError(why)


def score(answers, rows):
    require(len(answers) == len(rows) and len(rows) > 0, "response count")
    correct = sum(a == r["target"] for a, r in zip(answers, rows, strict=True))
    return {"correct": correct, "n": len(rows), "accuracy": correct / len(rows)}


def predicted(state, rows):
    beta, labels = state["readout"]["beta"], state["readout"]["labels"]
    out, losses = [], []
    for row in rows:
        scores = [sum(x * w[c] for x, w in zip(row["x"], beta, strict=True)) for c in range(len(labels))]
        out.append(labels[max(range(len(labels)), key=scores.__getitem__)])
        high = max(scores)
        losses.append(high + math.log(sum(math.exp(v - high) for v in scores)) - scores[labels.index(row["target"])] )
    return out, statistics.mean(losses)


def lake_checkpoint(raw, expected):
    require(raw[:8] == b"SNDLBIN1", "lake magic")
    pos, states, foreign = 8, [], False
    while pos < len(raw):
        require(pos + 4 <= len(raw), "torn frame")
        n = struct.unpack_from("<I", raw, pos)[0]
        payload = raw[pos + 4:pos + 4 + n]
        require(len(payload) == n, "torn payload")
        pos += 4 + n
        require(payload[:1] == b"\x07", "unexpected lake tenant tag")
        k = payload[1]
        name = payload[2:2+k].decode("ascii")
        blob = payload[2+k+4:]
        if name == "stage05-readout-v1":
            states.append(blob)
        elif name == "stage05-foreign":
            require(blob == b"preserve-this-tenant", "foreign tenant mutation")
            foreign = True
        else:
            raise ValueError("unexpected named tenant")
    require(len(states) == 3 and foreign, "checkpoint/foreign record count")
    require(sha(states[-1]) == expected, "latest checkpoint receipt")
    env = json.loads(states[-1])
    require(hashed(env["body"]) == env["sha256"], "checkpoint checksum")
    return env["body"]


def validate_case(data, arm, raw, restored, joint):
    require(raw["arm"] == arm and raw["seed"] == data["seed"] == restored["seed"] == joint["seed"], "identity")
    require(set(raw["reads"]) == {"before", "after_a", "after_b", "after_sleep", "final"}, "missing phases")
    replay = [r for t in TASKS for r in data["tasks"][t]["epochs"][0]]
    require(raw["states"]["after_a"]["replay"] == data["tasks"][TASKS[0]]["epochs"][0], "A replay membership")
    require(raw["states"]["after_b"]["replay"] == replay == raw["states"]["after_sleep"]["replay"], "train-only replay membership")
    require(raw["offline"]["rows_sha256"] == hashed(replay * 10), "offline rows differ")
    labels = [r["target"] for r in replay * 10]
    if arm == "batch_shuffle":
        require(Counter(raw["offline"]["labels"]) == Counter(labels), "shuffle marginals")
        require(raw["offline"]["labels"] != labels, "shuffle vacuous")
    else:
        require(raw["offline"]["labels"] == labels, "teaching labels changed")
    require(raw["offline"]["presentations"] == (0 if arm == "none" else 400), "offline budget")
    require(raw["offline"]["solves"] == int(arm.startswith("batch")), "solve budget")
    for phase, calls in (("after_a", 240), ("after_b", 1200), ("after_sleep", 1600 if arm == "replay" else 1200)):
        st = raw["states"][phase]["readout"]
        require(st["rule"]["calls"] == calls and st["rule"]["widened"] == 0, "update budget")
        require(len(st["beta"]) == 256 and all(len(w) == 2 for w in st["beta"]), "beta shape")
        require(all(math.isfinite(v) for w in st["beta"] for v in w), "nonfinite beta")
    if arm == "none":
        require(raw["states"]["after_b"] == raw["states"]["after_sleep"], "no-update state changed")
    else:
        require(raw["states"]["after_b"]["readout"]["beta"] != raw["states"]["after_sleep"]["readout"]["beta"], "offline beta inert")
    require(restored["state"] == raw["states"]["after_sleep"], "restore state differs")
    require(restored["lake_sha256"] == raw["lake_sha256"], "restore wrote lake")
    scores = {}
    for phase, reads in raw["reads"].items():
        require(set(reads) == set(TASKS), "missing task")
        scores[phase] = {}
        for t in TASKS:
            split = "final" if phase == "final" else "dev"
            rows = data["tasks"][t][split]
            scores[phase][t] = score(reads[t]["answers"], rows)
            if phase != "before":
                state = raw["states"]["after_sleep" if phase == "final" else phase]
                require(reads[t]["state_sha256"] == hashed(state), "read state mismatch")
                want, _ = predicted(state, rows)
                require(reads[t]["answers"] == want, "answers inconsistent with beta")
            if phase in ("after_sleep", "final"):
                rsplit = "dev" if phase == "after_sleep" else "final"
                require(reads[t] == restored["reads"][rsplit][t], "new-process read differs")
    for t in TASKS:
        require(score(joint["reads"][t]["answers"], data["tasks"][t]["dev"])["accuracy"] == 1., "joint positive failed")
    a, b = TASKS
    endpoint = {"scores": scores,
                "forget_a": scores["after_a"][a]["accuracy"] - scores["after_b"][a]["accuracy"],
                "headroom_a": 1. - scores["after_b"][a]["accuracy"],
                "headroom_b": 1. - scores["after_b"][b]["accuracy"],
                "cost": {k: raw["offline"][k] for k in ("wall_s", "cpu_s", "presentations", "solves")}}
    endpoint["train_diagnostic"] = {phase: {t: {
        "accuracy": score(predicted(raw["states"][phase], data["tasks"][t]["epochs"][0])[0], data["tasks"][t]["epochs"][0])["accuracy"],
        "cross_entropy": predicted(raw["states"][phase], data["tasks"][t]["epochs"][0])[1]}
        for t in TASKS} for phase in ("after_b", "after_sleep")}
    return endpoint


def analyze(records):
    manifest = json.loads((records / "manifest.json").read_text())
    require(manifest["status"] == "COMPLETE", "incomplete run")
    require(manifest["seeds"] == list(SEEDS) and manifest["arms"] == list(ARMS), "cohort/arm mismatch")
    successes = {r["key"] for r in manifest["runs"] if not r["expected_failure"]}
    wanted = {"controls"}
    for seed in SEEDS:
        for suffix in ("", "-repeat") if seed == 8000 else ("",):
            wanted |= {f"features-{seed}{suffix}", f"joint-{seed}{suffix}"}
            wanted |= {f"{prefix}{arm}-{seed}{suffix}" for arm in ARMS for prefix in ("", "restore-")}
    require(successes == wanted and len(manifest["runs"]) == len(wanted) + 4, "missing/extra worker")
    errors = {"missing": "missing checkpoint", "stale": "checkpoint receipt mismatch",
              "corrupt": "checkpoint receipt mismatch", "recipe": "pool recipe/topology mismatch"}
    for row in manifest["runs"]:
        output = (records / f"{row['key']}.json").read_bytes()
        stderr = (records / f"{row['key']}.stderr.txt").read_bytes()
        require(sha(output) == row["stdout_sha256"] and sha(stderr) == row["stderr_sha256"], "transport hash")
        if row["expected_failure"]:
            mode = row["key"].removeprefix("integrity-")
            require(row["returncode"] != 0 and errors[mode] in stderr.decode(), "wrong integrity failure")
        else:
            require(row["returncode"] == 0 and not stderr, "worker transport failed")
    load = lambda key: json.loads((records / f"{key}.json").read_text())["result"]
    controls = load("controls")
    require(all(controls["reds"].values()) and controls["batch_positive"] == 1. and controls["batch_sign_control"] == 0., "controls failed")
    summary = {"schema": "sndl.stage05.summary.v1", "head": manifest["head"], "cases": {}, "contrasts": {},
               "scope": "exposed diagnostic seeds; descriptive only; no scientific NULL or population claim"}
    for seed in SEEDS:
        data = json.loads((records / f"inputs-{seed}.json").read_text())
        require(load(f"features-{seed}") == data, "feature data differs")
        a_text = {dumps(r["texts"]) for r in data["tasks"][TASKS[0]]["epochs"][0]}
        b_text = {dumps(r["texts"]) for r in data["tasks"][TASKS[1]]["epochs"][0]}
        require(not a_text & b_text, "A/B episode overlap")
        sets = [{dumps(r["texts"]) for r in rows} for rows in (data["tasks"]["ac1d"]["epochs"][0], data["tasks"]["ac1d"]["dev"], data["tasks"]["ac1d"]["final"])]
        require(all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "ac1d split overlap")
        baseline = load(f"none-{seed}")
        for arm in ARMS:
            raw, restored = load(f"{arm}-{seed}"), load(f"restore-{arm}-{seed}")
            for phase in ("after_a", "after_b"):
                require(raw["states"][phase] == baseline["states"][phase], "unpaired wake state")
            require(raw["offline"]["rows_sha256"] == baseline["offline"]["rows_sha256"], "unpaired replay rows")
            lake = (records / "homes" / f"{arm}-{seed}" / "brain_lake.bin").read_bytes()
            require(sha(lake) == raw["lake_sha256"], "lake bytes changed")
            checkpoint = lake_checkpoint(lake, raw["checkpoints"]["after_sleep"])
            require(checkpoint["state"] == raw["states"]["after_sleep"] and checkpoint["pool"] == data["pool"], "lake decoded state differs")
            summary["cases"][f"{arm}/{seed}"] = validate_case(data, arm, raw, restored, load(f"joint-{seed}"))
    for arm, base in (("replay", "none"), ("batch", "none"), ("batch", "replay"), ("batch", "batch_shuffle")):
        differences = {t: [summary["cases"][f"{arm}/{s}"]["scores"]["final"][t]["accuracy"] - summary["cases"][f"{base}/{s}"]["scores"]["final"][t]["accuracy"] for s in SEEDS] for t in TASKS}
        differences["macro"] = [sum(differences[t][i] for t in TASKS) / 2 for i in range(len(SEEDS))]
        summary["contrasts"][f"{arm}-{base}"] = {t: {"paired": vals, "mean": statistics.mean(vals), "sample_sd": statistics.stdev(vals)} for t, vals in differences.items()}
    summary["controls"] = controls
    summary["integrity_failures_detected"] = list(errors)
    summary["cost"] = {"block_wall_s": manifest["wall_s"], "subprocess_wall_s": sum(r["wall_s"] for r in manifest["runs"]),
                       "successful_workers": len(wanted), "expected_failed_workers": 4,
                       "raw_stdout_bytes": sum((records / f"{r['key']}.json").stat().st_size for r in manifest["runs"])}
    for arm in ARMS:
        a, b = load(f"{arm}-8000"), load(f"{arm}-8000-repeat")
        for row in (a, b):
            row["offline"].pop("wall_s")
            row["offline"].pop("cpu_s")
        require(a == b, "repeat differs")
    return summary


def sensitivity(records):
    data = json.loads((records / "inputs-8000.json").read_text())
    get = lambda key: json.loads((records / f"{key}.json").read_text())["result"]
    raw, restored, joint = get("batch-8000"), get("restore-batch-8000"), get("joint-8000")
    require(validate_case(data, "batch", raw, restored, joint), "positive reader failed")
    cases = {}
    for kind in ("answer", "missing_task", "replay", "restore", "budget", "joint", "state", "labels"):
        d, r, q, j = copy.deepcopy((data, raw, restored, joint))
        if kind == "answer":
            r["reads"]["final"][TASKS[0]]["answers"][0] = "bad-answer"
        elif kind == "missing_task":
            del r["reads"]["after_b"][TASKS[0]]
        elif kind == "replay":
            r["states"]["after_sleep"]["replay"].append(d["tasks"]["ac1d"]["final"][0])
        elif kind == "restore":
            q["state"]["readout"]["beta"][0][0] += 1
        elif kind == "budget":
            r["offline"]["presentations"] -= 1
        elif kind == "joint":
            j["reads"][TASKS[0]]["answers"] = ["bad"] * 32
        elif kind == "state":
            r["reads"]["after_b"][TASKS[0]]["state_sha256"] = "bad-state"
        else:
            r["offline"]["labels"][0] = "bad-label"
        try:
            validate_case(d, "batch", r, q, j)
        except (ValueError, KeyError, TypeError):
            cases[kind] = True
        else:
            cases[kind] = False
    require(all(cases.values()), "offline sensitivity failed")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, reds = analyze(args.records), sensitivity(args.records)
    args.output.mkdir()
    (args.output / "summary.json").write_text(dumps(summary) + "\n", encoding="ascii")
    (args.output / "sensitivity.json").write_text(dumps(reds) + "\n", encoding="ascii")
    lines = ["| Seed | Arm | A after A | A after B | A after offline | B after offline | A final | B final |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for arm in ARMS:
            scores = summary["cases"][f"{arm}/{seed}"]["scores"]
            cells = [scores[p][t] for p, t in (("after_a", TASKS[0]), ("after_b", TASKS[0]), ("after_sleep", TASKS[0]), ("after_sleep", TASKS[1]), ("final", TASKS[0]), ("final", TASKS[1]))]
            lines.append(f"| {seed} | {arm} | " + " | ".join(f"{v['correct']}/{v['n']}" for v in cells) + " |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    print(dumps({"status": "VALID", "summary_sha256": sha((args.output / "summary.json").read_bytes()), "sensitivity": reds}))


if __name__ == "__main__":
    main()
