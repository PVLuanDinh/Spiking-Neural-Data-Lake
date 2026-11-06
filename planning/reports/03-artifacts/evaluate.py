"""Bounded stage-03 adapter over existing SNDL readers. Outputs stay outside repo/."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import sysconfig
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT / "repo"
HEAD = "c4c10b285f1741aa798d5ee9c5296ae371fabecf"
SEEDS = (8000, 8001, 8002, 8003)
ARMS = ("trained", "frozen", "shuffled")


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.write_text(dumps(value) + "\n", encoding="ascii")


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def runtime():
    require(sys.version_info[:2] == (3, 14), "CPython 3.14t required")
    require(sysconfig.get_config_var("Py_GIL_DISABLED") == 1 and not sys._is_gil_enabled(), "GIL must be off")


def snapshot(unit):
    # r2.v1 state_dict intentionally excludes lake-derived replay/authority caches.
    return {
        "whole": unit._snapshot(),
        "hydration": {k: getattr(unit, k) for k in ("_hydrated", "_hydration_count", "_hydration_refusals")},
    }


def state_hash(unit):
    return digest(dumps(snapshot(unit)).encode("ascii"))


class EvaluationError(RuntimeError):
    pass


def guarded_read(raw_read, unit, problem, seed, split, route, *, read_index=0, records=None):
    before = state_hash(unit)

    def forbidden(*args, **kwargs):
        raise EvaluationError("learning method called during evaluation")

    def label_free(turns):
        if any(t.label is not None for t in turns):
            raise EvaluationError("target supplied to reader")
        return route(turns)

    # Patch class attributes: no callable is inserted in the checkpoint's instance tree.
    with contextlib.ExitStack() as stack:
        for name in ("learn_turn", "interact", "_deposit"):
            stack.enter_context(patch.object(type(unit), name, forbidden))
        reads = raw_read(problem, seed, split, label_free, read_index=read_index, records=records)
    after = state_hash(unit)
    if before != after:
        raise EvaluationError("evaluation changed learned state or replay cache")
    return reads, before, after


def instrument_checks(tp, tc, dc):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        tp.main()
        tc.main()
    p = tp.PROBLEMS["ac1d"]
    u = tp.canonical_incumbent(8900)
    original_read = tp.read_split
    records = []
    reads, _, _ = guarded_read(original_read, u, p, 8900, "validation",
                               lambda t: u.route_text(t[-1].text), records=records)
    require(len(reads) == p.presentations(8900, "validation"), "guard positive did not run")
    reds = {}
    for name, route in (
        ("learning_during_read", lambda t: u.learn_turn(t[-1].text, None, feedback=dc.GREET)),
        ("replay_append_during_read", lambda t: (u._replay.append(([0.0] * u.n_in, dc.GREET, 0.0)), dc.GREET)[1]),
    ):
        fresh = tp.canonical_incumbent(8900)
        u = fresh
        try:
            guarded_read(original_read, u, p, 8900, "validation", route)
        except EvaluationError:
            reds[name] = True
        else:
            reds[name] = False
    u = tp.canonical_incumbent(8900)
    old, whole = tp.learner_state(u), state_hash(u)
    u._replay.append(([0.0] * u.n_in, dc.GREET, 0.0))
    reds["r2_checkpoint_omits_replay_but_snapshot_detects_it"] = (
        old == tp.learner_state(u) and whole != state_hash(u)
    )
    try:
        guarded_read(lambda p, s, sp, route, **kw: route((tp.Turn("x", "probe", "GOAL"),)),
                     u, p, 8900, "validation", lambda t: dc.GREET)
    except EvaluationError:
        reds["leaked_target"] = True
    else:
        reds["leaked_target"] = False
    contract_reds = tp.construction_reds(tp.canonical_incumbent)
    parts, _ = dc.combo_partition("ac1d", 8000)
    checks = {"b9": tp.check_b9(), "b11": tp.check_b11(), "b13": dc.check_b13(parts),
              "b14": dc.check_b14("ac1d", parts)}
    require(all(v["ok"] for v in checks.values()), "construction/control failed")
    require(contract_reds["all_fired"] and all(reds.values()), "a planted red did not fire")
    # A separate, actual S0C0 positive at the same existing channel.
    s0 = tp.PROBLEMS["i.S0C0"]
    r = tc.make_reader("incumbent", s0, 8000)
    a = tc.attempt(s0, r, 8000, tc.EPOCHS, stop_when_stable=True)
    require(a["stable"], "S0C0 channel positive failed")
    return {"checks": checks, "contract_reds": contract_reds, "guard_reds": reds,
            "s0c0": a, "selfchecks_stdout": output.getvalue(),
            "b1_b8_full_battery": "NOT RUN: minimal source-level runtime; no Brain identity claim"}


def arm_worker(tp, tc, dc, arm, seed, epochs):
    p = tp.PROBLEMS["ac1d"]
    r = tc.make_reader("incumbent", p, seed)
    u = r.unit
    raw_read = tp.read_split
    initial = snapshot(u)
    evaluations, teaching = [], []
    phase = "before_dev"

    def read(problem, read_seed, split, route, *, read_index=0, records=None):
        require(split in ("validation", "future"), "unregistered evaluation split")
        require(split != "future" or phase == "final", "final read before checkpoint freeze")
        recs = records if records is not None else []
        result, before, after = guarded_read(raw_read, u, problem, read_seed, split, route,
                                             read_index=read_index, records=recs)
        targets = problem.episodes(seed, split, read_index=read_index)
        evaluations.append({
            "phase": phase, "split": split, "read_index": read_index,
            "state_before": before, "state_after": after,
            "learning_calls": 0, "reader_saw_targets": False,
            "answers": [{"context": list(ctx), "probe": probe, "answer": answer,
                         "target": ep[-1].label}
                        for (ctx, probe, answer), ep in zip(recs, targets, strict=True)],
        })
        # Keep the authoritative existing solved predicate in the live path.
        tp.solved(problem, seed, split, result)
        return result

    teach = r.teach
    true_sequence = tp.teaching_sequence(p, seed, tc.EPOCHS if arm == "trained" else epochs)

    def record_teach(turns, label):
        idx = len(teaching)
        entry = true_sequence[idx]
        require(turns[-1].text == entry[1], "training order mismatch")
        teaching.append({"index": idx, "probe": turns[-1].text, "target": entry[2],
                         "taught": label, "update": arm != "frozen"})
        if arm != "frozen":
            return teach(turns, label)
        return None

    r.teach = record_teach
    with patch.object(tp, "read_split", read):
        read(p, seed, "validation", r.route)
        phase = "epoch_dev"
        attempt = tc.attempt(p, r, seed, tc.EPOCHS if arm == "trained" else epochs,
                             judge_future=False, stop_when_stable=arm == "trained",
                             shuffled=arm == "shuffled")
        frozen_hash = state_hash(u)
        phase = "final"
        read(p, seed, "future", r.route)
    final = snapshot(u)
    oracle = dc.combo_witness_answer("ac1d")
    oracle_reads = {}
    for split in ("validation", "future"):
        recs = []
        values = raw_read(p, seed, split, lambda t: oracle(t[-1].text), records=recs)
        oracle_reads[split] = [{"context": list(c), "probe": t, "answer": a, "target": label}
                               for (c, t, a), (_, label) in zip(recs, values, strict=True)]
        require(tp.solved(p, seed, split, values), "independent oracle failed")
    contract = {
        "train_first_epoch": list(p.teach_epochs(seed, 1))[0],
        "dev": p.episodes(seed, "validation"), "final": p.episodes(seed, "future"),
    }
    return {
        "schema": "sndl.stage03.arm.v1", "problem": "ac1d", "arm": arm, "seed": seed,
        "config": {"max_epochs": 400, "matched_epochs": attempt["epochs_run"], "stable": tp.STABLE,
                   "lr": u.lr, "n_in": u.n_in, "responses": u.responses,
                   "checkpoint_rule": "first trained dev stable-10; controls matched to trained budget"},
        "contract": contract, "teaching": teaching, "evaluations": evaluations, "attempt": attempt,
        "initial_snapshot": initial, "final_snapshot": final,
        "initial_sha256": digest(dumps(initial).encode("ascii")), "final_sha256": state_hash(u),
        "checkpoint_before_final": frozen_hash, "oracle": oracle_reads,
        "replay_entries": len(u._replay), "weights": u.weights(),
    }


def worker(args):
    runtime()
    require(args.seed in SEEDS or (args.worker == "checks" and args.seed == 8900), "seed outside exposed dev/scratch")
    require(1 <= args.epochs <= 400, "invalid epoch budget")
    from snn_data_lake.substrate.home import require_isolated_home
    from snn_data_lake.instruments import delayed_cue as dc
    from snn_data_lake.instruments import taught_campaign as tc
    from snn_data_lake.instruments import taught_problems as tp

    home = Path(require_isolated_home())
    require(not any(home.iterdir()), "home must start empty")
    tc.preflight((args.seed,), HERE, arms=((args.worker,),), require_clean=True)
    out = instrument_checks(tp, tc, dc) if args.worker == "checks" else arm_worker(tp, tc, dc, args.worker, args.seed, args.epochs)
    runtime()
    require(not any(home.iterdir()), "unexpected home/lake write")
    out.update(isolated_home_empty=True, gil_enabled_after=False)
    print(dumps(out))


def checkout():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    require(head == HEAD and not dirty, "pinned clean source checkout required")
    return head


def run(args):
    runtime()
    head = checkout()
    require(args.run_id and args.run_id.replace("-", "").isalnum(), "provide a new alphanumeric --run-id")
    out = HERE / args.run_id
    out.mkdir(exist_ok=False)
    provenance = out / "provenance"
    provenance.mkdir()
    files = [HERE / f for f in ("protocol.md", "post-data-amendments.md", "evaluate.py", "recompute.py", "power.py", "run-stage.py")]
    files += [REPO / "src/snn_data_lake" / p for p in (
        "instruments/taught_problems.py", "instruments/taught_campaign.py", "instruments/delayed_cue.py",
        "readout/interaction_learning.py", "plasticity/plasticity_rule_stack.py")]
    hashes = {}
    for p in files:
        relative = str(p.relative_to(ROOT))
        hashes[relative] = digest(p.read_bytes())
        target = provenance / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    manifest = {"schema": "sndl.stage03.run.v1", "head": head, "source_dirty": False,
                "seed_status": "exposed development; exploratory instrument pilot only",
                "seeds": list(SEEDS), "arms": list(ARMS), "runs": [], "exclusions": [],
                "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                "files_sha256": hashes, "status": "RUNNING", "worker_timeout_seconds": 120,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    start = time.monotonic()
    env = os.environ.copy()
    for key in ("NORD_ALLOW_GIL", "PYTHON_GIL"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")

    def job(arm, seed, epochs=400, suffix=""):
        require(time.monotonic() - start < 1200, "20-minute block budget reached")
        key = f"{arm}-{seed}{suffix}"
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", arm,
               "--seed", str(seed), "--epochs", str(epochs)]
        t0, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_CHILDREN)
        with tempfile.TemporaryDirectory(prefix="sndl-stage03-") as temporary:
            env["NORD_HOME"] = temporary
            try:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=120)
                raw, error, code = completed.stdout, completed.stderr, completed.returncode
                why = None if code == 0 and not error else "worker error or nonempty stderr"
            except subprocess.TimeoutExpired as e:
                raw, error, code, why = e.stdout or b"", e.stderr or b"", None, "worker timeout at 120 seconds"
            empty_home = not any(Path(temporary).iterdir())
        cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        (out / (key + ".json")).write_bytes(raw)
        (out / (key + ".stderr.txt")).write_bytes(error)
        entry = {"key": key, "arm": arm, "seed": seed, "epochs_requested": epochs,
                 "command": cmd, "exit_code": code, "exclusion_reason": why,
                 "seconds": time.monotonic() - t0,
                 "cpu_seconds": cpu.ru_utime + cpu.ru_stime - usage.ru_utime - usage.ru_stime,
                 "stdout_bytes": len(raw), "stderr_bytes": len(error), "home_empty_parent": empty_home,
                 "result_sha256": digest(raw)}
        if not empty_home:
            entry["exclusion_reason"] = "unexpected isolated home write"
        manifest["runs"].append(entry)
        save(out / "manifest.json", manifest)
        require(entry["exclusion_reason"] is None, f"HALT {key}: {entry['exclusion_reason']}")
        return json.loads(raw)

    try:
        job("checks", 8900)
        for seed in SEEDS:
            positive = job("trained", seed)
            require(positive["attempt"]["stable"], f"HALT learned positive failed at {seed}")
            require(all(r["answer"] == r["target"] for r in positive["evaluations"][-1]["answers"]), "HALT learned final positive failed")
            for arm in ("frozen", "shuffled"):
                row = job(arm, seed, positive["attempt"]["epochs_run"])
                flags = row["attempt"]["solved_ends"]
                stable = row["config"]["stable"]
                require(not any(all(flags[i:i + stable]) for i in range(len(flags) - stable + 1)), f"HALT negative control stable: {arm}/{seed}")
        repeated = job("trained", 8000, suffix="-repeat")
        require((out / "trained-8000.json").read_bytes() == (out / "trained-8000-repeat.json").read_bytes(), "new-process repeat differs")
        manifest["status"] = "COMPLETE"
    except BaseException as error:
        manifest["status"] = "HALT"
        manifest["exclusions"].append({"reason": str(error), "scope": "no scientific reading; preserve all raw files"})
        raise
    finally:
        manifest["total_seconds"] = time.monotonic() - start
        save(out / "manifest.json", manifest)
    checkout()
    print(f"Saved complete diagnostic run: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--worker", choices=("checks", *ARMS))
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
