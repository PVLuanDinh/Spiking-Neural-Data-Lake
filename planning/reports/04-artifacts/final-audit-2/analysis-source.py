"""Independent stdlib audit and summaries of stage-04 records; no SNDL imports."""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile

SEEDS = (8000, 8001, 8002, 8003)
TASKS = ("i.S0C0", "i.S1C0", "ac1d")
MODES = ("carry", "reset")
ARMS = ("rst", "rst_frozen", "rst_shuffle", "delta", "delta_frozen", "delta_shuffle")


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hashed(value):
    return sha(dumps(value).encode("ascii"))


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def load(path):
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {"checks-8900"} | {f"{a}-{s}" for s in SEEDS for a in ("features", *ARMS)} | {f"{a}-8000-repeat" for a in ("features", *ARMS)}
    need(manifest["status"] == "COMPLETE", "incomplete/failed manifest")
    need(manifest["head"] == "c4c10b285f1741aa798d5ee9c5296ae371fabecf", "wrong source HEAD")
    for key, want in (("seeds", SEEDS), ("tasks", TASKS), ("modes", MODES), ("arms", ARMS)):
        need(manifest[key] == list(want), f"wrong {key}")
    need(manifest["epochs"] == 30, "wrong epochs")
    need(len(manifest["runs"]) == len(expected) and {r["key"] for r in manifest["runs"]} == expected, "missing/duplicate worker")
    workers = {}
    for entry in manifest["runs"]:
        need(entry["returncode"] == 0 and entry["failure"] is None and entry["home_empty"], "worker error/home write")
        raw = (path / (entry["key"] + ".json")).read_bytes()
        need(sha(raw) == entry["sha256"] and len(raw) == entry["bytes"], "raw hash/size mismatch")
        record = json.loads(raw)
        need(record["schema"] == "sndl.stage04.worker.v1" and record["runtime"]["gil"] is False, "runtime/schema mismatch")
        workers[entry["key"]] = record
    for file in (path / "provenance").rglob("*"):
        if file.is_file():
            key = str(file.relative_to(path / "provenance"))
            need(sha(file.read_bytes()) == manifest["provenance"][key], "provenance copy changed")
    for seed in SEEDS:
        inputs = json.loads((path / f"features-{seed}.json.inputs").read_text())
        need(inputs == workers[f"features-{seed}"]["result"], "saved worker input differs from recorded features")
    return manifest, workers


def bound(rows):
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[dumps(row["x"])][row["target"]] += 1
    return {"correct_max": sum(max(g.values()) for g in grouped.values()), "n": len(rows),
            "unique": len(grouped), "conflicting_groups": sum(len(g) > 1 for g in grouped.values())}


def score(read, rows):
    need(read["before"] == read["after"], "evaluation changed state")
    need(len(read["answers"]) == len(rows), "missing/extra answer")
    need(set(read["answers"]) <= {"GREET", "GOAL"}, "invalid answer")
    by_x = defaultdict(set)
    for a, row in zip(read["answers"], rows, strict=True):
        by_x[dumps(row["x"])].add(a)
    need(all(len(v) == 1 for v in by_x.values()), "same features produced different answers")
    return sum(a == r["target"] for a, r in zip(read["answers"], rows, strict=True))


def stable(flags):
    return any(all(flags[i:i+10]) for i in range(len(flags)-9))


def weights(snapshot):
    if "beta" in snapshot:
        return snapshot["beta"]
    return [n["attrs"]["w"] for n in snapshot["whole"]["state"]["tree"]["attrs"]["neurons"]]


def target(task, texts):
    if task == "i.S0C0":
        return {"oak": "GREET", "elm": "GOAL"}[texts[-1].split()[0]]
    if task == "i.S1C0":
        return {"jar": "GREET", "pen": "GOAL"}[texts[0]]
    return {"fog": "GREET", "gel": "GOAL"}[texts[-1].split()[0]]


def audit(workers):
    checks = workers["checks-8900"]["result"]
    need(all(checks["guard_reds"].values()) and len(checks["guard_reds"]) == 4, "guard control failed")
    need(all(p["stable"] and p["future_solved"] for p in checks["positive"].values()), "upstream positive failed")
    rows, bounds = [], []
    for name in ("features", *ARMS):
        need(workers[f"{name}-8000"]["result"] == workers[f"{name}-8000-repeat"]["result"], "new-process repeat differs")
    for seed in SEEDS:
        features = workers[f"features-{seed}"]["result"]
        need(features["seed"] == seed and set(features["tasks"]) == set(TASKS), "feature identity/tasks mismatch")
        for task in TASKS:
            data = features["tasks"][task]
            n = 32 if task == "ac1d" else 8
            need(data["epoch_size"] == n and len(data["true"]) == len(data["shuffle"]) == 30, "epoch budget mismatch")
            need(all(Counter(a) == Counter(b) and a != b and len(a) == n for a,b in zip(data["true"], data["shuffle"], strict=True)), "shuffle/marginals wrong")
            for mode in MODES:
                case = data["modes"][mode]
                paths = case["paths"]
                need(case["parameters_before"] == case["parameters_after"], "network parameters moved")
                need(case["path_sha256"] == hashed(paths), "feature hash mismatch")
                need(all(case["checks"].values()), "pool path guard failed")
                need([r["target"] for r in paths["train"]] == sum(data["true"], []), "training targets/order mismatch")
                need([len(paths[k]) for k in ("train", "dev", "final")] == [30*n, 32, 48 if task == "ac1d" else 32], "split size wrong")
                for split, records in paths.items():
                    for row in records:
                        need(row["target"] == target(task, row["texts"]), "wrong independently reconstructed target")
                        need(len(row["x"]) == 256 and all(v in (0.0,1.0) for v in row["x"]), "noncanonical T_in=1 feature")
                if task == "ac1d":
                    sets = [set(tuple(r["texts"]) for r in paths[k]) for k in ("train", "dev", "final")]
                    need([len(s) for s in sets] == [32,8,12] and all(not sets[i] & sets[j] for i,j in ((0,1),(0,2),(1,2))), "split overlap")
                    need(paths == data["modes"]["reset"]["paths"], "one-turn carry/reset mismatch")
                for split in ("dev", "final"):
                    b = bound(paths[split])
                    need(case[f"{split}_feature_bound"] == b, "feature information bound wrong")
                    bounds.append({"seed": seed, "task": task, "mode": mode, "split": split, **b})
                for name in ARMS:
                    worker = workers[f"{name}-{seed}"]
                    need(worker["kind"] == name and worker["seed"] == seed, "arm identity mismatch")
                    result = worker["result"]
                    need(result["seed"] == seed and result["arm"] == name and set(result["cases"]) == {f"{t}/{m}" for t in TASKS for m in MODES}, "arm cases missing")
                    r = result["cases"][f"{task}/{mode}"]
                    need(r["epochs"] == 30 and len(r["dev"]) == 31 and r["path_sha256"] == case["path_sha256"], "arm budget/features wrong")
                    need(r["updates"] == (0 if name.endswith("frozen") else 30*n), "update count wrong")
                    family = name.split("_")[0]
                    twin = workers[f"{family}-{seed}"]["result"]["cases"][f"{task}/{mode}"]
                    need(r["initial"] == twin["initial"], "initial family states differ")
                    need(r["dev"][0]["before"] == hashed(r["initial"]), "initial read not bound to snapshot")
                    need(r["final"]["before"] == hashed(r["final_state"]) == r["dev"][-1]["after"], "checkpoint/final not frozen")
                    if name.endswith("frozen"):
                        need(r["initial"] == r["final_state"], "frozen state changed")
                        need(all(v["before"] == hashed(r["initial"]) for v in r["dev"]), "frozen intermediate changed")
                    else:
                        need(weights(r["initial"]) != weights(r["final_state"]), "trained synapses never moved")
                    counts = [score(read, paths["dev"]) for read in r["dev"]]
                    final = score(r["final"], paths["final"])
                    flags = [c == len(paths["dev"]) for c in counts[1:]]
                    rows.append({"seed": seed, "task": task, "mode": mode, "arm": name,
                                 "before": counts[0], "dev_n": len(paths["dev"]), "dev": counts[1:],
                                 "final_correct": final, "final_n": len(paths["final"]),
                                 "final_accuracy": final/len(paths["final"]), "stable10": all(flags[-10:]),
                                 "ever_stable10": stable(flags),
                                 "stable_at_epoch": next((i+10 for i in range(21) if all(flags[i:i+10])), None),
                                 "state_changed": r["initial"] != r["final_state"], "updates": r["updates"]})
    return {"rows": rows, "feature_bounds": bounds}


def sensitivity(workers):
    faults = {}

    def test(name, change):
        edited = copy.deepcopy(workers)
        change(edited)
        try:
            audit(edited)
        except (ValueError, KeyError, IndexError):
            faults[name] = True
        else:
            faults[name] = False

    case = "i.S0C0/carry"
    test("missing_arm", lambda w: w.pop("rst-8001"))
    test("missing_answer", lambda w: w["rst-8001"]["result"]["cases"][case]["final"]["answers"].pop())
    test("wrong_feature_binding", lambda w: w["rst-8001"]["result"]["cases"][case].update(path_sha256="bad"))
    test("wrong_budget", lambda w: w["delta-8001"]["result"]["cases"][case].update(updates=0))
    test("learning_on_read", lambda w: w["rst-8001"]["result"]["cases"][case]["dev"][2].update(after="changed"))
    test("frozen_moved", lambda w: w["rst_frozen-8001"]["result"]["cases"][case]["final_state"].update(bad=True))
    test("broken_positive", lambda w: w["checks-8900"]["result"]["positive"]["i.S0C0"].update(stable=False))
    test("broken_guard", lambda w: w["checks-8900"]["result"]["guard_reds"].update(delta_beta=False))
    test("wrong_target", lambda w: w["features-8001"]["result"]["tasks"]["ac1d"]["modes"]["carry"]["paths"]["final"][0].update(target="bad"))
    test("pool_parameters_moved", lambda w: w["features-8001"]["result"]["tasks"]["ac1d"]["modes"]["carry"]["parameters_after"].update(bad=True))
    need(all(faults.values()), "an intentional fault was accepted")
    return faults


def transport_checks(path):
    """Real loader, actual records, altered local manifests; source files read-only."""
    results = {}
    for name in ("incomplete", "worker_error", "home_write", "missing_worker", "raw_hash", "wrong_header", "corrupt_bytes"):
        with tempfile.TemporaryDirectory(prefix="sndl04-audit-") as fresh:
            root = Path(fresh)
            for source in path.iterdir():
                if source.name != "manifest.json":
                    (root/source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
            manifest = json.loads((path/"manifest.json").read_text())
            if name == "incomplete":
                manifest["status"] = "RUNNING"
            elif name == "worker_error":
                manifest["runs"][0]["failure"] = "synthetic timeout"
            elif name == "home_write":
                manifest["runs"][0]["home_empty"] = False
            elif name == "missing_worker":
                manifest["runs"].pop()
            elif name == "raw_hash":
                manifest["runs"][0]["sha256"] = "bad"
            elif name == "wrong_header":
                manifest["seeds"] = [8000]
            else:
                key = manifest["runs"][0]["key"]
                file = root/(key+".json")
                file.unlink()
                file.write_bytes(b"broken")
            (root/"manifest.json").write_text(dumps(manifest))
            try:
                load(root)
            except (ValueError, KeyError, FileNotFoundError):
                results[name] = True
            else:
                results[name] = False
    need(all(results.values()), "corrupt transport accepted")
    return results


def describe(values):
    return {"values": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values), "inference": "DESCRIPTIVE ONLY; N=4 exposed diagnostic seeds"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, workers = load(args.records)
    summary = audit(workers)
    faults = sensitivity(workers)
    transport = transport_checks(args.records)
    by = {(r["seed"],r["task"],r["mode"],r["arm"]):r for r in summary["rows"]}
    contrasts = []
    for task in TASKS:
        for mode in MODES:
            for a,b in (("rst","rst_shuffle"),("rst","rst_frozen"),("delta","delta_shuffle"),("delta","delta_frozen"),("delta","rst")):
                contrasts.append({"task":task,"mode":mode,"a":a,"b":b,**describe([by[s,task,mode,a]["final_accuracy"]-by[s,task,mode,b]["final_accuracy"] for s in SEEDS])})
        for name in ("rst","delta"):
            contrasts.append({"task":task,"arm":name,"a":"carry","b":"reset",**describe([by[s,task,"carry",name]["final_accuracy"]-by[s,task,"reset",name]["final_accuracy"] for s in SEEDS])})
    summary.update(status="VALID_DIAGNOSTIC_RECORDS", head=manifest["head"], contrasts=contrasts,
                   analysis_sha256=sha(Path(__file__).read_bytes()),
                   cost={"total_seconds":manifest["total_seconds"],"subprocess_seconds":sum(r["seconds"] for r in manifest["runs"]),
                         "raw_bytes":sum(r["bytes"] for r in manifest["runs"]),"workers":len(manifest["runs"])})
    summary["baseline_eligibility"] = [{"task":t,"mode":m,"family":f,
        "stable_final_with_controls":all(by[s,t,m,f]["stable10"] and by[s,t,m,f]["final_correct"] == by[s,t,m,f]["final_n"]
            and not by[s,t,m,f+"_frozen"]["ever_stable10"] and not by[s,t,m,f+"_shuffle"]["ever_stable10"] for s in SEEDS)}
        for t in TASKS for m in MODES for f in ("rst","delta")]
    args.output.mkdir()
    for name, obj in (("summary.json",summary),("sensitivity.json",faults),("transport.json",transport)):
        (args.output/name).write_text(dumps(obj)+"\n",encoding="ascii")
    lines = ["| Task | Mode | Seed | Arm | Before | Last dev | Final | Stable-10 | First stable epoch |", "|---|---|---:|---|---:|---:|---:|---|---:|"]
    for row in summary["rows"]:
        lines.append(f"| {row['task']} | {row['mode']} | {row['seed']} | {row['arm']} | {row['before']}/{row['dev_n']} | {row['dev'][-1]}/{row['dev_n']} | {row['final_correct']}/{row['final_n']} | {row['stable10']} | {row['stable_at_epoch']} |")
    (args.output/"table.md").write_text("\n".join(lines)+"\n",encoding="ascii")
    print(f"Audit OK: {len(summary['rows'])} cases, {len(faults)} analysis faults and {len(transport)} transport faults caught; fresh-process repeat equal.")
    for task in TASKS:
        for mode in MODES:
            print(task,mode,{a:[by[s,task,mode,a]["final_correct"] for s in SEEDS] for a in ARMS})


if __name__ == "__main__":
    main()
