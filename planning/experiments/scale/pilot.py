"""Bounded SHD experiment wrapper around the pinned upstream e-prop implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from snn_data_lake import require_free_threaded
from snn_data_lake.sensorimotor.eprop_shd import EpropAudioNet, eprop_grads
from snn_data_lake.plasticity.eprop_kernel import _apply_grads

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SOURCE_SHA = "c4c10b285f1741aa798d5ee9c5296ae371fabecf"
SOURCE_ROOT = Path("/private/tmp/sndl-class-source-tlefts_z/src")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def model_hash(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def score(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    assert target.shape == prediction.shape and target.ndim == 1 and len(target)
    assert np.all((0 <= target) & (target < 20))
    assert np.all((0 <= prediction) & (prediction < 20))
    confusion = np.bincount(target * 20 + prediction, minlength=400).reshape(20, 20)
    denom = confusion.sum(1)
    per_class = np.divide(confusion.diagonal(), denom, out=np.full(20, np.nan), where=denom > 0)
    return {"accuracy": float((target == prediction).mean()), "macro_accuracy": float(np.nanmean(per_class)),
            "correct": int((target == prediction).sum()), "n": len(target),
            "per_class_accuracy": per_class.tolist(), "confusion": confusion.tolist()}


class Experiment:
    def __init__(self, args):
        require_free_threaded("SHD pilot", strict=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        # Reuse the source export whose 483 Git blobs were verified in the prior session.
        # Current Git pack reads stall on dataless files; do not silently claim fresh verification.
        provenance = json.loads((WORKSPACE / "planning/reports/05-artifacts/class-source-verification.json").read_text())
        assert provenance["head"] == SOURCE_SHA and provenance["all_match"]
        assert Path(provenance["source_root"]) / "src" == SOURCE_ROOT
        imported = {}
        for name, module in list(sys.modules.items()):
            file = getattr(module, "__file__", None)
            if name.startswith("snn_data_lake") and file:
                path = Path(file).resolve()
                assert path.is_relative_to(SOURCE_ROOT), (name, str(path))
                imported[str(path.relative_to(SOURCE_ROOT))] = sha(path)
        actual = SOURCE_SHA
        self.args = args
        cache = json.loads((ROOT / "data" / "cache-location.json").read_text())
        self.data = Path(cache["prepared"])
        assert sha(self.data / "manifest.json") == cache["manifest_sha256"]
        self.manifest = json.loads((self.data / "manifest.json").read_text())
        for name in ("labels", "speakers", "train_ids", "dev_ids"):
            setattr(self, name, np.load(self.data / f"{name}.npy", allow_pickle=False))
        self.frames = np.load(self.data / "frames.npy", mmap_mode="r", allow_pickle=False)
        assert self.frames.shape == (8156, 100, 700)
        self.train_set = set(self.train_ids.tolist())
        assert self.train_set.isdisjoint(self.dev_ids.tolist())
        assert set(self.speakers[self.train_ids]).isdisjoint(self.speakers[self.dev_ids])
        assert set(self.labels[self.train_ids]) == set(self.labels[self.dev_ids]) == set(range(20))
        self.started = time.monotonic()
        self.out = ROOT / "runs" / args.run_id
        self.out.mkdir(exist_ok=False)
        self.history = []
        self.updates = 0
        self.exposures = 0
        self.current_exposures = 0
        self.replay_exposures = 0
        self.repeat_exposures = 0
        self.seen = set()
        self.net = EpropAudioNet(hidden=64, seed=args.seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.002, betas=(0.9, 0.999))
        self.class_order = np.random.default_rng(args.seed).permutation(20)
        self.task_classes = self.class_order.reshape(10, 2)
        self.emit({"event": "start", "source_sha": actual, "seed": args.seed, "arm": args.arm,
                   "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "gil": sys._is_gil_enabled(), "device": "cpu", "threads": torch.get_num_threads(),
                   "platform": platform.platform(), "config": self.net.config(), "batch": 16, "lr": 0.002,
                   "dataset_manifest_sha256": sha(self.data / "manifest.json"),
                   "runner_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
                   "source_files": imported, "source_verification": provenance, "fresh_git_verification": False,
                   "class_order": self.class_order.tolist(), "initial_model_hash": model_hash(self.net),
                   "official_test_read": False})

    def emit(self, row):
        row = dict(row, elapsed_s=time.monotonic() - self.started, peak_rss_bytes=rss_bytes(),
                   updates=self.updates, exposures=self.exposures)
        with open(self.out / "events.jsonl", "a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def guard(self):
        assert rss_bytes() <= 3 * 1024 ** 3, "3 GiB RSS cap"
        assert time.monotonic() - self.started <= self.args.max_seconds, "run wall-time cap"
        require_free_threaded("SHD pilot runtime", strict=True)

    def batch(self, ids):
        return torch.from_numpy(np.asarray(self.frames[ids], dtype=np.float32)), torch.from_numpy(self.labels[ids].copy())

    def update(self, ids, kind):
        self.guard()
        assert set(ids.tolist()) <= self.train_set
        if kind == "replay":
            assert set(ids.tolist()) <= self.seen
        x, y = self.batch(ids)
        grads, loss, spike_rate = eprop_grads(self.net, x, y)
        assert np.isfinite(loss) and np.isfinite(spike_rate)
        assert all(torch.isfinite(g).all().item() for g in grads.values()), "nonfinite gradient"
        _apply_grads(self.net, grads, self.opt)
        assert all(torch.isfinite(p).all().item() for p in self.net.parameters()), "nonfinite parameter"
        self.updates += 1
        self.exposures += len(ids)
        if kind == "current":
            self.current_exposures += len(ids)
        elif kind == "replay":
            self.replay_exposures += len(ids)
        else:
            self.repeat_exposures += len(ids)
        return loss, spike_rate

    @torch.no_grad()
    def evaluate(self, checkpoint, learned_tasks=0):
        before = model_hash(self.net)
        logits, latent = [], []
        for i in range(0, len(self.dev_ids), 32):
            self.guard()
            x, _ = self.batch(self.dev_ids[i:i + 32])
            a, b = self.net(x)
            assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
            logits.append(a.numpy())
            latent.append(b.numpy())
        logits, latent = np.concatenate(logits), np.concatenate(latent)
        pred = logits.argmax(1)
        targets = self.labels[self.dev_ids]
        metrics = score(targets, pred)
        task_scores = []
        for classes in self.task_classes:
            mask = np.isin(targets, classes)
            task_scores.append(float((pred[mask] == targets[mask]).mean()))
        after = model_hash(self.net)
        assert before == after, "evaluation modified model"
        path = self.out / f"predictions-{checkpoint}.npz"
        np.savez_compressed(path, ids=self.dev_ids, speakers=self.speakers[self.dev_ids], targets=targets,
                            predictions=pred, logits=logits, hidden_spike_counts=latent)
        row = {"event": "evaluation", "checkpoint": checkpoint, **metrics,
               "task_accuracy": task_scores, "learned_tasks": learned_tasks,
               "seen_task_mean_accuracy": float(np.mean(task_scores[:learned_tasks])) if learned_tasks else None,
               "model_hash": after, "evaluation_weights_unchanged": True,
               "predictions": path.name, "predictions_sha256": sha(path)}
        self.history.append(row)
        self.emit(row)
        return row

    def checks(self):
        perfect = score(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
        fault = score(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]))
        assert perfect["accuracy"] == perfect["macro_accuracy"] == 1
        assert fault["accuracy"] == fault["macro_accuracy"] == 0
        assert np.asarray(perfect["confusion"]).sum() == 4
        x, y = self.batch(self.train_ids[:16])
        twins = [EpropAudioNet(hidden=64, seed=self.args.seed) for _ in range(2)]
        for twin in twins:
            opt = torch.optim.Adam(twin.parameters(), lr=0.002)
            for _ in range(2):
                grads, loss, rate = eprop_grads(twin, x, y)
                _apply_grads(twin, grads, opt)
        assert model_hash(twins[0]) == model_hash(twins[1]), "CPU update nondeterminism"
        assert model_hash(twins[0]) != model_hash(self.net), "no learning update"
        self.emit({"event": "apparatus_checks", "scoring_fault_detected": True,
                   "deterministic_two_updates": True, "training_ids_only": True, "speaker_disjoint": True,
                   "frame_count_conservation": self.manifest["all_events_conserved"]})

    def joint(self):
        rng = np.random.default_rng(self.args.seed)
        for epoch in range(1, 3):
            ids = rng.permutation(self.train_ids)
            loss_total, spike_total, count = 0.0, 0.0, 0
            for start in range(0, len(ids), 16):
                batch_ids = ids[start:start + 16]
                loss, rate = self.update(batch_ids, "current")
                self.seen.update(batch_ids.tolist())
                loss_total += loss
                spike_total += rate * len(batch_ids)
                count += len(batch_ids)
                if count in (256, 1024, 4096) or self.updates % 32 == 0:
                    self.emit({"event": "training", "epoch": epoch, "epoch_current_exposures": count,
                               "mean_loss_per_example_timestep": loss_total / count,
                               "mean_hidden_spike_fraction": spike_total / count})
            self.evaluate(f"epoch-{epoch}", 10)

    def sequential(self):
        order_rng = np.random.default_rng(self.args.seed)
        buffer_rng = np.random.default_rng(self.args.seed + 0x100000)
        replay_rng = np.random.default_rng(self.args.seed + 0x200000)
        buffer = []
        processed = 0
        stream = []
        for block, classes in enumerate(self.task_classes, 1):
            ids = order_rng.permutation(self.train_ids[np.isin(self.labels[self.train_ids], classes)])
            block_start = time.monotonic()
            loss_total, spike_total, n_updates = 0.0, 0.0, 0
            for start in range(0, len(ids), 16):
                current = ids[start:start + 16]
                stream.extend(current.tolist())
                loss, rate = self.update(current, "current")
                loss_total += loss / len(current)
                spike_total += rate
                n_updates += 1
                if self.args.arm in ("repeat", "replay"):
                    if self.args.arm == "replay" and buffer:
                        historical = replay_rng.choice(buffer, size=len(current), replace=len(buffer) < len(current))
                        loss, rate = self.update(historical, "replay")
                    else:
                        loss, rate = self.update(current, "repeat")
                    loss_total += loss / len(current)
                    spike_total += rate
                    n_updates += 1
                # Algorithm R uniform reservoir, never development or future examples.
                for sample in current.tolist():
                    assert sample not in self.seen
                    self.seen.add(sample)
                    processed += 1
                    if len(buffer) < 400:
                        buffer.append(sample)
                    else:
                        j = int(buffer_rng.integers(processed))
                        if j < 400:
                            buffer[j] = sample
                assert set(buffer) <= self.seen <= self.train_set
                if self.updates % 32 == 0:
                    self.emit({"event": "training", "block": block, "classes": classes.tolist(),
                               "block_current_exposures": min(start + 16, len(ids)),
                               "mean_loss_per_example_timestep": loss_total / n_updates,
                               "mean_hidden_spike_fraction": spike_total / n_updates,
                               "buffer_size": len(buffer) if self.args.arm == "replay" else 0})
            self.emit({"event": "block_complete", "block": block, "classes": classes.tolist(),
                       "current_examples": len(ids), "training_seconds": time.monotonic() - block_start,
                       "buffer_size": len(buffer) if self.args.arm == "replay" else 0})
            self.evaluate(f"block-{block}", block)
        assert len(stream) == len(set(stream)) == len(self.train_ids)
        np.save(self.out / "stream_ids.npy", np.asarray(stream), allow_pickle=False)
        if self.args.arm == "replay":
            np.save(self.out / "final_replay_ids.npy", np.asarray(buffer), allow_pickle=False)

    def run(self):
        self.checks()
        self.evaluate("initial")
        if self.args.arm == "joint":
            self.joint()
        else:
            self.sequential()
        final = self.history[-1]
        summary = {"status": "complete", "seed": self.args.seed, "arm": self.args.arm,
                   "final_accuracy": final["accuracy"], "final_macro_accuracy": final["macro_accuracy"],
                   "positive_control_gate_pass": final["macro_accuracy"] > 0.10 if self.args.arm == "joint" else None,
                   "wall_seconds": time.monotonic() - self.started, "peak_rss_bytes": rss_bytes(),
                   "updates": self.updates, "exposures": self.exposures, "unique_train_examples": len(self.seen),
                   "current_exposures": self.current_exposures, "replay_exposures": self.replay_exposures,
                   "repeat_exposures": self.repeat_exposures,
                   "official_test_read": False, "scope": "fixed-width audio component, not full engine",
                   "history": self.history}
        if self.args.arm != "joint":
            matrix = np.asarray([r["task_accuracy"] for r in self.history[1:]])
            forgetting = [float(matrix[t:, t].max() - matrix[-1, t]) for t in range(9)]
            summary["forgetting_by_task"] = forgetting
            summary["mean_forgetting"] = float(np.mean(forgetting))
            summary["task_accuracy_matrix"] = matrix.tolist()
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        self.emit({"event": "complete", **{k: v for k, v in summary.items() if k not in ("history", "task_accuracy_matrix")}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, choices=range(8000, 8004), default=8000)
    ap.add_argument("--arm", choices=("joint", "sequential", "repeat", "replay"), required=True)
    ap.add_argument("--max-seconds", type=int, default=3600)
    args = ap.parse_args()
    experiment = Experiment(args)
    try:
        experiment.run()
    except BaseException as exc:
        failure = {"status": "failed", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                   "updates": experiment.updates, "exposures": experiment.exposures, "wall_seconds": time.monotonic() - experiment.started}
        (experiment.out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
