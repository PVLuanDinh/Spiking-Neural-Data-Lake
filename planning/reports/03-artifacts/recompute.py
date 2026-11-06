"""Offline stage-03 audit: stdlib and saved records only; no SNDL imports/training."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SEEDS = (8000, 8001, 8002, 8003)
ARMS = ("trained", "frozen", "shuffled")


class Invalid(ValueError):
    pass


def need(ok, reason):
    if not ok:
        raise Invalid(reason)


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def accuracy(answers):
    need(bool(answers), "empty answers")
    return sum(r["answer"] == r["target"] for r in answers) / len(answers)


def any_stable(flags):
    return any(all(flags[i:i + 10]) for i in range(len(flags) - 9))


def validate_answers(answers, episodes):
    need(len(answers) == len(episodes), "missing or extra presentation")
    for r, ep in zip(answers, episodes, strict=True):
        need(r["context"] == [t[0] for t in ep[:-1]] and r["probe"] == ep[-1][0], "wrong input/order")
        need(r["target"] == ep[-1][2], "target/order mismatch")
        # Independent task-label rule, rather than trusting a generator's saved target.
        words = r["probe"].split()
        need(len(words) == 4 and words[0] in ("fog", "gel"), "not an A_c1d probe")
        need(r["target"] == ("GREET" if words[0] == "fog" else "GOAL"), "wrong A_c1d truth")


def validate_row(row):
    need(row["schema"] == "sndl.stage03.arm.v1", "unknown row schema")
    seed, arm = row["seed"], row["arm"]
    need(seed in SEEDS and arm in ARMS and row["problem"] == "ac1d", "unexpected seed/arm/problem")
    need(row["isolated_home_empty"] and not row["gil_enabled_after"], "runtime/home failed")
    c, a = row["config"], row["attempt"]
    n = a["epochs_run"]
    need(c["stable"] == 10 and c["max_epochs"] == 400 and c["matched_epochs"] == n, "budget/config mismatch")
    need(10 <= n <= 400, "invalid epoch count")
    contract = row["contract"]
    need(len(contract["train_first_epoch"]) == 32 and len(contract["dev"]) == 32 and len(contract["final"]) == 48, "split count")
    sets = [{ep[-1][0] for ep in contract[k]} for k in ("train_first_epoch", "dev", "final")]
    need([len(s) for s in sets] == [32, 8, 12], "split unique count")
    need(all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "train/dev/final overlap")
    evaluations = row["evaluations"]
    need(len(evaluations) == n + 2, "missing evaluation/epoch")
    need([r["phase"] for r in evaluations] == ["before_dev"] + ["epoch_dev"] * n + ["final"], "final before freeze or repeated")
    flags, accuracies = [], []
    for i, r in enumerate(evaluations):
        final = i == n + 1
        need(r["split"] == ("future" if final else "validation"), "wrong split role")
        need(r["read_index"] == (i - 1 if 1 <= i <= n else 0), "wrong read index")
        need(r["state_before"] == r["state_after"], "evaluation changed state/cache")
        need(r["learning_calls"] == 0 and not r["reader_saw_targets"], "learning or target leak during evaluation")
        validate_answers(r["answers"], contract["final" if final else "dev"])
        if 1 <= i <= n:
            accuracies.append(accuracy(r["answers"]))
            flags.append(accuracies[-1] == 1.0)
    need(accuracies == a["accuracy_ends"] and flags == a["solved_ends"], "epoch metrics mismatch")
    need(a["validation_read_indices"] == list(range(n)), "missing epoch index")
    stable = len(flags) >= 10 and all(flags[-10:])
    need(a["stable"] == stable and a["never"] == (not any(flags)), "stability/never mismatch")
    need(a["max_accuracy"] == max(accuracies), "max metric mismatch")
    need(a["first_solved_end"] == next((i for i, f in enumerate(flags) if f), None), "first solved mismatch")
    if arm == "trained" and stable:
        need(not any_stable(flags[:-1]), "trained did not stop at first stable-10")
    need(len(row["teaching"]) == n * 32, "missing teaching presentations")
    for epoch in range(n):
        tape = row["teaching"][epoch * 32:(epoch + 1) * 32]
        need({r["probe"] for r in tape} == sets[0], "epoch did not use exactly the train split")
        need([r["index"] for r in tape] == list(range(epoch * 32, (epoch + 1) * 32)), "training order index")
        for r in tape:
            need(r["target"] == ("GREET" if r["probe"].split()[0] == "fog" else "GOAL"), "wrong train truth")
            need(r["update"] == (arm != "frozen"), "frozen teaching policy mismatch")
        need(Counter(r["target"] for r in tape) == Counter(r["taught"] for r in tape), "shuffle changed label marginals")
        need(arm == "shuffled" or all(r["target"] == r["taught"] for r in tape), "wrong teaching labels")
        if arm == "shuffled":
            need(any(r["target"] != r["taught"] for r in tape), "shuffle was identity")
    for key, snapshot in (("initial_sha256", "initial_snapshot"), ("final_sha256", "final_snapshot")):
        need(sha(dumps(row[snapshot]).encode("ascii")) == row[key], "snapshot hash mismatch")
    need(evaluations[0]["state_before"] == row["initial_sha256"], "initial evaluation state mismatch")
    need(evaluations[-2]["state_after"] == row["checkpoint_before_final"] == row["final_sha256"] == evaluations[-1]["state_before"], "checkpoint was not frozen before final")
    if arm == "frozen":
        need(row["initial_sha256"] == row["final_sha256"], "untrained twin changed state/cache")
        need(all(e["state_before"] == row["initial_sha256"] for e in evaluations), "untrained twin changed transiently between reads")
    for split, k in (("validation", "dev"), ("future", "final")):
        validate_answers(row["oracle"][split], contract[k])
        need(accuracy(row["oracle"][split]) == 1.0, "oracle control failed: HALT")
    return {"seed": seed, "arm": arm, "before_dev_accuracy": accuracy(evaluations[0]["answers"]),
            "last_dev_accuracy": accuracies[-1], "max_dev_accuracy": max(accuracies),
            "final_accuracy": accuracy(evaluations[-1]["answers"]), "stable10": stable,
            "ever_stable10": any_stable(flags), "epochs": n, "train_presentations": len(row["teaching"]),
            "updates": sum(t["update"] for t in row["teaching"]),
            "final_correct": sum(r["answer"] == r["target"] for r in evaluations[-1]["answers"]),
            "final_presentations": 48}


def compare(rows):
    need(sorted((r["seed"], r["arm"]) for r in rows) == sorted((s, a) for s in SEEDS for a in ARMS), "missing/duplicate seed or arm: NOT RUN")
    metrics = [validate_row(r) for r in rows]
    pairs = []
    for seed in SEEDS:
        rs = {r["arm"]: r for r in rows if r["seed"] == seed}
        ms = {r["arm"]: r for r in metrics if r["seed"] == seed}
        positive = rs["trained"]
        for arm in ARMS:
            need(rs[arm]["contract"] == positive["contract"], "unmatched data/splits")
            need(rs[arm]["initial_snapshot"] == positive["initial_snapshot"], "unmatched initial conditions")
            need(rs[arm]["config"] == positive["config"], "unmatched config/budget")
            need([(t["index"], t["probe"], t["target"]) for t in rs[arm]["teaching"]] ==
                 [(t["index"], t["probe"], t["target"]) for t in positive["teaching"]], "unmatched train sequence")
        need(ms["trained"]["stable10"], "learned dev positive failed: HALT")
        need(ms["trained"]["final_accuracy"] == 1.0, "learned final positive failed: HALT")
        need(not ms["frozen"]["ever_stable10"] and not ms["shuffled"]["ever_stable10"], "negative control stable: HALT")
        pairs.append({"seed": seed, "trained_minus_shuffled": ms["trained"]["final_accuracy"] - ms["shuffled"]["final_accuracy"],
                      "trained_minus_frozen": ms["trained"]["final_accuracy"] - ms["frozen"]["final_accuracy"]})
    diffs = [r["trained_minus_shuffled"] for r in pairs]
    # This descriptive distinction is intentionally not a statistical verdict.
    observed = "NO_OBSERVED_FINAL_DIFFERENCE" if all(d == 0 for d in diffs) else "OBSERVED_FINAL_DIFFERENCE"
    return {"scope": "exposed-dev exploratory instrument pilot; no population/null claim",
            "observed": observed, "rows": metrics, "paired": pairs,
            "primary_mean": statistics.mean(diffs), "primary_sd": statistics.stdev(diffs),
            "n_independent_seeds": len(SEEDS), "population_verdict": "NOT ATTEMPTED"}


def load_run(folder):
    m = json.loads((folder / "manifest.json").read_text())
    need(m["schema"] == "sndl.stage03.run.v1" and m["status"] == "COMPLETE", "incomplete/error/timeout run: NOT RUN")
    need(not m["exclusions"] and not m["source_dirty"], "exclusions/dirty tree: INVALID")
    need(m["seeds"] == list(SEEDS) and m["arms"] == list(ARMS), "changed registered set")
    need(len(m["runs"]) == 14, "missing/extra subprocess")
    for relative, expected in m["files_sha256"].items():
        need(sha((folder / "provenance" / relative).read_bytes()) == expected, "provenance hash mismatch")
    rows, repeated, checks = [], None, None
    keys = []
    for entry in m["runs"]:
        key = entry["key"]
        keys.append(key)
        raw = (folder / (key + ".json")).read_bytes()
        need(entry["result_sha256"] == sha(raw) and entry["stdout_bytes"] == len(raw), "result hash mismatch")
        need(entry["exit_code"] == 0 and not entry["exclusion_reason"] and entry["home_empty_parent"], "worker error/timeout")
        need(entry["stderr_bytes"] == 0 and not (folder / (key + ".stderr.txt")).read_bytes(), "worker stderr")
        r = json.loads(raw)
        if key == "checks-8900":
            checks = r
        elif key == "trained-8000-repeat":
            repeated = raw
        else:
            need(r["arm"] == entry["arm"] and r["seed"] == entry["seed"] and key == f"{r['arm']}-{r['seed']}", "manifest identity mismatch")
            need(entry["epochs_requested"] == (400 if r["arm"] == "trained" else r["attempt"]["epochs_run"]), "manifest budget mismatch")
            rows.append(r)
    need(len(set(keys)) == 14 and checks is not None and repeated is not None, "missing/duplicate special worker")
    need(repeated == (folder / "trained-8000.json").read_bytes(), "independent repeat differs")
    need(checks["isolated_home_empty"] and not checks["gil_enabled_after"], "checks runtime failed")
    need(all(checks["guard_reds"].values()) and checks["contract_reds"]["all_fired"], "planted red did not fire")
    need(all(c["ok"] for c in checks["checks"].values()) and checks["s0c0"]["stable"], "positive check failed")
    return m, rows


def sensitivity(rows):
    tests = {}

    def fixture(change):
        changed = copy.deepcopy(rows)
        change(changed)
        return changed

    def rejects(name, change):
        try:
            compare(fixture(change))
        except Invalid as error:
            tests[name] = {"detected": True, "reason": str(error)}
        else:
            raise Invalid("sensitivity failed: " + name)

    def final_answers(rs, mode):
        for row in rs:
            for i, r in enumerate(row["evaluations"][-1]["answers"]):
                if mode == "success":
                    r["answer"] = r["target"] if row["arm"] == "trained" else ("GOAL" if r["target"] == "GREET" else "GREET")
                elif mode == "none":
                    r["answer"] = r["target"]

    success = compare(fixture(lambda r: final_answers(r, "success")))
    no_effect = compare(fixture(lambda r: final_answers(r, "none")))
    need(success["primary_mean"] == 1 and success["observed"] == "OBSERVED_FINAL_DIFFERENCE", "success fixture missed")
    need(no_effect["primary_mean"] == 0 and no_effect["observed"] == "NO_OBSERVED_FINAL_DIFFERENCE", "no-effect fixture missed")
    tests["constructed_success"] = {"detected": True, "delta": 1.0}
    tests["constructed_no_final_effect"] = {"detected": True, "delta": 0.0, "not_a_scientific_null": True}
    rejects("missing_arm", lambda rs: rs.pop())
    rejects("missing_seed", lambda rs: rs.__setitem__(slice(None), [r for r in rs if r["seed"] != 8003]))
    rejects("duplicate_arm_seed", lambda rs: rs.append(copy.deepcopy(rs[0])))
    rejects("missing_answer", lambda rs: rs[0]["evaluations"][-1]["answers"].pop())
    rejects("wrong_target", lambda rs: rs[0]["evaluations"][-1]["answers"][0].__setitem__("target", "wrong"))
    rejects("changed_state_or_replay", lambda rs: rs[0]["evaluations"][-1].__setitem__("state_after", "changed"))
    rejects("target_leak", lambda rs: rs[0]["evaluations"][-1].__setitem__("reader_saw_targets", True))
    rejects("training_during_read", lambda rs: rs[0]["evaluations"][-1].__setitem__("learning_calls", 1))
    rejects("broken_oracle", lambda rs: rs[0]["oracle"]["future"][0].__setitem__("answer", "wrong"))
    rejects("overlapping_splits", lambda rs: rs[0]["contract"]["final"].__setitem__(0, rs[0]["contract"]["dev"][0]))
    rejects("wrong_budget", lambda rs: rs[1]["config"].__setitem__("matched_epochs", 11))
    rejects("wrong_teaching_order", lambda rs: rs[1]["teaching"].reverse())
    rejects("broken_learned_positive", lambda rs: rs[0]["attempt"].__setitem__("stable", False))
    def failed_positive(rs):
        row = rs[0]
        for evaluation in row["evaluations"][1:-1]:
            for answer in evaluation["answers"]:
                answer["answer"] = "wrong"
        a = row["attempt"]
        a.update(accuracy_ends=[0.0] * a["epochs_run"], solved_ends=[False] * a["epochs_run"],
                 max_accuracy=0.0, first_solved_end=None, stable=False, never=True)
    rejects("failed_positive_consistent_metrics", failed_positive)
    rejects("failed_final_positive", lambda rs: rs[0]["evaluations"][-1]["answers"][0].__setitem__("answer", "wrong"))
    def false_negative_control(rs):
        row = rs[2]  # shuffled at seed 8000
        for evaluation in row["evaluations"][1:-1]:
            for answer in evaluation["answers"]:
                answer["answer"] = answer["target"]
        a = row["attempt"]
        a.update(accuracy_ends=[1.0] * a["epochs_run"], solved_ends=[True] * a["epochs_run"],
                 max_accuracy=1.0, first_solved_end=0, stable=True, never=False)
    rejects("negative_control_solves_stably", false_negative_control)
    return {"kind": "synthetic apparatus fixtures derived from copies; not experiment observations", "tests": tests}


def summary(folder):
    manifest, rows = load_run(folder)
    result = compare(rows)
    result["head"] = manifest["head"]
    result["analysis_sha256"] = sha(Path(__file__).read_bytes())
    result["raw_sha256"] = {e["key"]: e["result_sha256"] for e in manifest["runs"]}
    result["successful_subprocesses"] = len(manifest["runs"])
    result["wall_seconds_sum"] = sum(e["seconds"] for e in manifest["runs"])
    result["cpu_seconds_sum"] = sum(e["cpu_seconds"] for e in manifest["runs"])
    result["raw_output_bytes"] = sum(e["stdout_bytes"] for e in manifest["runs"])
    result["independent_repeat_byte_equal"] = True
    return result, sensitivity(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        result, tests = summary(args.run.resolve())
    except (Invalid, KeyError, ValueError, OSError, TypeError) as e:
        (args.output / "invalid.json").write_text(dumps({"status": "INVALID / NO SCIENTIFIC READING", "reason": str(e)}))
        raise
    for name, value in (("summary", result), ("sensitivity", tests)):
        (args.output / (name + ".json")).write_text(dumps(value) + "\n", encoding="ascii")
    lines = ["| Seed | Arm | Before dev | Last dev | Final | Stable-10 | Epochs | Updates |",
             "|---|---|---:|---:|---:|---|---:|---:|"]
    for r in result["rows"]:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['before_dev_accuracy']:.3f} | {r['last_dev_accuracy']:.3f} | {r['final_correct']}/48 | {r['stable10']} | {r['epochs']} | {r['updates']} |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("Audit OK; replay-safe evaluation; complete matched seed/arm rows; independent repeat equal.")
    print("Primary paired differences:", result["paired"])
    print("Pilot SD:", result["primary_sd"], "; synthetic cases:", len(tests["tests"]))


if __name__ == "__main__":
    main()
