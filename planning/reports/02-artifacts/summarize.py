"""Recompute the stage-02 diagnostic tables from saved answers, without SNDL imports or training."""

import hashlib
import itertools
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = ("checks-3", "s0c0-1", "repeat-1", "cells-1", "combos-1")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    rows, hashes, elapsed = {}, {}, 0.0
    manifests = []
    for run in RUNS:
        manifest = json.loads((HERE / run / "manifest.json").read_text())
        manifests.append(manifest)
        for recorded in manifest["runs"]:
            case = recorded["case"]
            path = HERE / run / (case + ".stdout.json")
            raw = path.read_bytes()
            assert sha(raw) == recorded["result_sha256"], path
            assert recorded["exit_code"] == 0 and recorded["stderr_empty"], path
            assert not (HERE / run / (case + ".stderr.txt")).read_bytes(), path
            row = json.loads(raw)
            assert row["isolated_home_empty"] and not row["gil_enabled_after"], path
            hashes[str(path.relative_to(HERE))] = sha(raw)
            elapsed += recorded["seconds"]
            if case == "checks":
                continue
            if run == "repeat-1":
                assert raw == (HERE / "s0c0-1/s0c0.stdout.json").read_bytes()
                continue
            a = row["attempt"]
            assert len(row["epoch_reads"]) == a["epochs_run"]
            assert len(row["teaching_sequence"]) == a["epochs_run"] * row["contract"]["n_teach_codes"]
            reference = row["contract"]["validation_episodes"]
            accuracy, flags = [], []
            for records in row["epoch_reads"]:
                assert len(records) == len(reference)
                for rec, episode in zip(records, reference, strict=True):
                    assert rec["context"] == [t[0] for t in episode[:-1]]
                    assert rec["probe"] == episode[-1][0] and rec["label"] == episode[-1][2]
                correct = [rec["answer"] == rec["label"] for rec in records]
                accuracy.append(sum(correct) / len(correct))
                flags.append(all(correct))
            assert accuracy == a["accuracy_ends"] and flags == a["solved_ends"]
            for name in ("before", "validation", "future"):
                read = row[name]
                split = "future" if name == "future" else "validation"
                episodes = row["contract"][split + "_episodes"]
                assert len(read["reads"]) == len(episodes)
                assert [label for _, label in read["reads"]] == [e[-1][2] for e in episodes]
                correct = [answer == label for answer, label in read["reads"]]
                assert read["accuracy"] == sum(correct) / len(correct)
                assert read["solved"] == all(correct) and read["state_unchanged"]
            stable = row["config"]["stable"]
            longest = max((len(list(g)) for v, g in itertools.groupby(flags) if v), default=0)
            first_stable = next((i + 1 for i in range(stable - 1, len(flags))
                                 if all(flags[i + 1 - stable:i + 1])), None)
            rows[case] = {
                "problem": row["problem"], "reader": row["arm"], "seed": row["seed"],
                "before_accuracy": row["before"]["accuracy"],
                "final_validation_accuracy": row["validation"]["accuracy"],
                "max_validation_accuracy": max(accuracy),
                "future_accuracy": row["future"]["accuracy"],
                "future_correct": sum(a == b for a, b in row["future"]["reads"]),
                "future_n": len(row["future"]["reads"]),
                "epochs": a["epochs_run"], "teach_presentations": len(row["teaching_sequence"]),
                "first_solved_epoch_1based": next((i + 1 for i, f in enumerate(flags) if f), None),
                "first_stable_epoch_1based": first_stable, "longest_solved_streak": longest,
                "ever_stable": longest >= stable, "last10_solved": a["stable"],
                "solved_read_count": sum(flags), "b3_every_read": row["b3_every_read"],
                "weights_changed": row["initial_weights_sha256"] != row["final_weights_sha256"],
                "source": str(path.relative_to(HERE)),
            }
    assert len(rows) == 19
    for field in ("head", "protocol_sha256", "adapter_sha256", "source_sha256"):
        assert all(m[field] == manifests[0][field] for m in manifests), field
    assert manifests[0]["protocol_sha256"] == sha((HERE / "protocol.md").read_bytes())
    assert manifests[0]["adapter_sha256"] == sha((HERE / "run-baseline.py").read_bytes())
    positives = ("s0c0", "history-s0c0", "history-s1c0", "presized-s0c1",
                 "presized-history-s1c1", "ac1d")
    for case in positives:
        assert rows[case]["ever_stable"] and rows[case]["future_accuracy"] == 1.0, case
    for case in set(rows) - set(positives):
        assert not rows[case]["ever_stable"], case
    summary = {
        "scope": "one exposed dev seed; diagnostic, no population inference or full G2 verdict",
        "head": manifests[0]["head"], "seed": 8000, "rows": rows,
        "independent_s0c0_repeat_byte_equal": True,
        "s0c0_result_sha256": hashes["s0c0-1/s0c0.stdout.json"],
        "successful_subprocesses": sum(len(m["runs"]) for m in manifests),
        "sum_subprocess_wall_seconds": elapsed,
        "result_sha256": hashes,
        "audit": "saved raw answers match saved targets and all epoch/final metrics; hashes verified",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    lines = ["| Case | Before | Final validation | Max validation | Stable-10 | Epochs | Future |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for case, r in rows.items():
        lines.append(
            f"| {case} | {r['before_accuracy']:.3f} | {r['final_validation_accuracy']:.3f} | "
            f"{r['max_validation_accuracy']:.3f} | {'yes' if r['ever_stable'] else 'no'} | "
            f"{r['epochs']} | {r['future_correct']}/{r['future_n']} |"
        )
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    print("Audit OK; independent S0C0 repeat byte-identical; subprocesses:", summary["successful_subprocesses"])
    print("Sum of measured subprocess wall seconds:", round(elapsed, 2))


if __name__ == "__main__":
    main()
