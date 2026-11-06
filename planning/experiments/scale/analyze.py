"""Independently rescore saved predictions and produce scientific figures + HTML report."""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
COLORS = {"joint": "#596776", "sequential": "#d16b47", "repeat": "#9875be", "replay": "#188e88"}
NAMES = {"joint": "Joint IID, two passes", "sequential": "Sequential, one update", "repeat": "Sequential, repeat current", "replay": "Sequential, replay history"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pct(values):
    return np.asarray(values) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", default="shd-")
    parser.add_argument("--metadata-only", action="store_true", help="Rescore using durable evaluation metadata; no temporary dataset cache needed")
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True)
    cache = json.loads((ROOT / "data/cache-location.json").read_text())
    data = Path(cache["prepared"])
    if args.metadata_only or not all((data / name).exists() for name in ("manifest.json", "labels.npy", "dev_ids.npy")):
        data = REPORT / "evaluation-metadata"
    manifest = json.loads((data / "manifest.json").read_text())
    assert sha(data / "manifest.json") == cache["manifest_sha256"]
    for name in ("labels.npy", "dev_ids.npy"):
        assert sha(data / name) == manifest["files"][name]["sha256"]
    labels = np.load(data / "labels.npy", allow_pickle=False)
    dev = np.load(data / "dev_ids.npy", allow_pickle=False)
    summaries, failures, verified, activity = [], [], 0, []
    speaker_metrics = []
    prediction_diagnostics = []
    interruptions = [json.loads(p.read_text()) for p in (ROOT / "runs").glob("interrupted-*/interruption.json")]
    source_hashes = None
    frozen_fingerprint = None
    source_audit = json.loads((REPORT / "source-audit.json").read_text())
    assert source_audit["all_match"] and source_audit["verified_blobs"] == 483
    for directory in sorted((ROOT / "runs").iterdir()):
        if not directory.name.startswith(args.run_prefix):
            continue
        if (directory / "failure.json").exists():
            failures.append({"run_id": directory.name, **json.loads((directory / "failure.json").read_text())})
        if not (directory / "summary.json").exists():
            continue
        s = json.loads((directory / "summary.json").read_text())
        s["run_id"] = directory.name
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        start = events[0]
        assert start["event"] == "start" and start["official_test_read"] is False and start["gil"] is False
        fingerprint = {k: start[k] for k in ("runner_sha256", "protocol_sha256", "dataset_manifest_sha256", "config", "lr", "batch")}
        if frozen_fingerprint is None:
            frozen_fingerprint = fingerprint
        assert fingerprint == frozen_fingerprint, "training configuration changed across cohort"
        if source_hashes is None:
            source_hashes = start["source_files"]
        assert start["source_files"] == source_hashes
        for path, value in start["source_files"].items():
            assert source_audit["files"]["src/" + path]["sha256"] == value
        for row in s["history"]:
            path = directory / row["predictions"]
            assert sha(path) == row["predictions_sha256"]
            with np.load(path, allow_pickle=False) as p:
                assert np.array_equal(p["ids"], dev)
                assert np.array_equal(p["targets"], labels[dev])
                assert p["logits"].shape == (len(dev), 20)
                assert p["hidden_spike_counts"].shape == (len(dev), 64)
                assert np.array_equal(p["predictions"], p["logits"].argmax(1))
                assert np.isfinite(p["logits"]).all() and np.isfinite(p["hidden_spike_counts"]).all()
                assert np.all((p["hidden_spike_counts"] >= 0) & (p["hidden_spike_counts"] <= 100))
                confusion = np.zeros((20, 20), dtype=np.int64)
                for target, prediction in zip(p["targets"], p["predictions"]):
                    confusion[int(target), int(prediction)] += 1
                accuracy = np.trace(confusion) / confusion.sum()
                macro = np.mean(np.diag(confusion) / confusion.sum(1))
                assert abs(accuracy - row["accuracy"]) < 1e-12
                assert abs(macro - row["macro_accuracy"]) < 1e-12
                assert np.array_equal(confusion, np.asarray(row["confusion"]))
                order = np.asarray(start["class_order"])
                assert np.array_equal(np.sort(order), np.arange(20))
                task_scores = [float(np.mean(p["predictions"][np.isin(p["targets"], classes)] == p["targets"][np.isin(p["targets"], classes)])) for classes in order.reshape(10, 2)]
                assert np.allclose(task_scores, row["task_accuracy"], rtol=0, atol=1e-12)
                activity.append({"run_id": directory.name, "arm": s["arm"], "seed": s["seed"],
                                 "checkpoint": row["checkpoint"], "learned_tasks": row["learned_tasks"],
                                 "mean_heldout_hidden_spike_fraction": float(p["hidden_spike_counts"].mean() / 100)})
                if row is s["history"][-1]:
                    prediction_counts = np.bincount(p["predictions"], minlength=20)
                    prediction_diagnostics.append({"run_id": directory.name, "arm": s["arm"],
                                                   "predicted_class_counts": prediction_counts.tolist(),
                                                   "number_of_predicted_classes": int(np.count_nonzero(prediction_counts)),
                                                   "dominant_prediction_fraction": float(prediction_counts.max() / prediction_counts.sum())})
                    for speaker in np.unique(p["speakers"]):
                        mask = p["speakers"] == speaker
                        truth, pred = p["targets"][mask], p["predictions"][mask]
                        per_class = [np.mean(pred[truth == c] == c) for c in range(20) if np.any(truth == c)]
                        speaker_metrics.append({"run_id": directory.name, "speaker": int(speaker), "n": int(mask.sum()),
                                                "accuracy": float(np.mean(pred == truth)), "macro_accuracy": float(np.mean(per_class))})
            assert row["evaluation_weights_unchanged"]
            verified += 1
        if s["arm"] != "joint":
            stream = np.load(directory / "stream_ids.npy", allow_pickle=False)
            assert len(stream) == len(set(stream.tolist())) == manifest["train_recordings"]
            assert set(stream.tolist()).isdisjoint(dev.tolist())
            matrix = np.asarray(s["task_accuracy_matrix"])
            strict_forgetting = [float(matrix[t:-1, t].max() - matrix[-1, t]) for t in range(9)]
            # Check the registered 'best earlier minus final' definition, without clipping gains.
            s["forgetting_summary_clipping_difference"] = float(np.max(np.abs(np.asarray(strict_forgetting) - s["forgetting_by_task"])))
            s["forgetting_by_task"] = strict_forgetting
            s["mean_forgetting"] = float(np.mean(strict_forgetting))
            last_classes = set(start["class_order"][-2:])
            per_class = s["history"][-1]["per_class_accuracy"]
            s["final_old_class_macro_accuracy"] = float(np.mean([v for c, v in enumerate(per_class) if c not in last_classes]))
            s["final_latest_class_macro_accuracy"] = float(np.mean([v for c, v in enumerate(per_class) if c in last_classes]))
        summaries.append(s)
    assert summaries, "no complete runs"
    grouped = {arm: [s for s in summaries if s["arm"] == arm] for arm in NAMES}
    for seed in sorted({s["seed"] for s in summaries}):
        runs = [s for s in summaries if s["seed"] == seed]
        assert len({s["history"][0]["model_hash"] for s in runs}) == 1
        sequential = [s for s in runs if s["arm"] != "joint"]
        if len(sequential) > 1:
            streams = [np.load(ROOT / "runs" / s["run_id"] / "stream_ids.npy", allow_pickle=False) for s in sequential]
            assert all(np.array_equal(streams[0], other) for other in streams[1:])
        matched = {s["arm"]: s for s in sequential}
        if "repeat" in matched and "replay" in matched:
            assert matched["repeat"]["exposures"] == matched["replay"]["exposures"]
            assert matched["repeat"]["updates"] == matched["replay"]["updates"]
    endpoints = []
    for arm, runs in grouped.items():
        if not runs:
            continue
        values = pct([s["final_macro_accuracy"] for s in runs])
        endpoints.append({"arm": arm, "n": len(runs), "macro_accuracy_mean_pct": float(values.mean()),
                          "overall_accuracy_mean_pct": float(np.mean([100*s["final_accuracy"] for s in runs])),
                          "macro_accuracy_sd_pp": float(values.std(ddof=1)) if len(values) > 1 else None,
                          "min_pct": float(values.min()), "max_pct": float(values.max()),
                          "mean_wall_seconds": float(np.mean([s["wall_seconds"] for s in runs])),
                          "max_peak_rss_bytes": max(s["peak_rss_bytes"] for s in runs)})
        if arm != "joint":
            endpoints[-1].update(mean_forgetting_pp=float(np.mean([100*s["mean_forgetting"] for s in runs])),
                                 final_old_class_macro_mean_pct=float(np.mean([100*s["final_old_class_macro_accuracy"] for s in runs])),
                                 final_latest_class_macro_mean_pct=float(np.mean([100*s["final_latest_class_macro_accuracy"] for s in runs])))
    repeats = {s["seed"]: s for s in grouped["repeat"]}
    replays = {s["seed"]: s for s in grouped["replay"]}
    pairs = [{"seed": seed, "replay_minus_repeat_macro_pp": 100 * (replays[seed]["final_macro_accuracy"] - repeats[seed]["final_macro_accuracy"]),
              "repeat_forgetting_pp": 100 * repeats[seed]["mean_forgetting"],
              "replay_forgetting_pp": 100 * replays[seed]["mean_forgetting"]}
             for seed in sorted(set(repeats) & set(replays))]
    report = {"scope": "exploratory fixed-width SHD audio component pilot; no full-engine inference",
              "completed_runs": len(summaries), "failures": failures, "independently_verified_prediction_checkpoints": verified,
              "infrastructure_interruptions": interruptions, "initialized_training_attempts": len(summaries) + len(interruptions),
              "dataset_manifest_sha256": cache["manifest_sha256"], "official_test_read": False,
              "scoring_metadata_source": str(data),
              "endpoints": endpoints, "paired_results": pairs, "source_files": source_hashes,
              "frozen_training_fingerprint": frozen_fingerprint,
              "heldout_hidden_activity": activity, "final_speaker_metrics": speaker_metrics,
              "final_prediction_diagnostics": prediction_diagnostics,
              "total_training_run_wall_seconds": sum(s["wall_seconds"] for s in summaries),
              "total_exposures": sum(s["exposures"] for s in summaries),
              "logical_replay_frame_bytes": 400 * 100 * 700 * 2,
              "source_verification_note": "All 483 original export blobs matched the pinned GitHub tree before cache loss; all five imported files were reverified after recovery. Every run's imported hashes match this audit. Immutable start logs precede the fresh audit.",
              "runs": [{k: v for k, v in s.items() if k not in ("history", "task_accuracy_matrix")} for s in summaries]}
    baseline = pct([s["history"][0]["macro_accuracy"] for s in grouped["joint"]])
    if len(baseline):
        report["frozen_initialization_macro_mean_pct"] = float(baseline.mean())
        report["frozen_initialization_macro_sd_pp"] = float(baseline.std(ddof=1)) if len(baseline) > 1 else None
    if len(pairs) > 1:
        deltas = np.array([p["replay_minus_repeat_macro_pp"] for p in pairs])
        report["paired_mean_pp"] = float(deltas.mean())
        report["paired_sd_pp"] = float(deltas.std(ddof=1))
    (REPORT / "pilot-results.json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for s in grouped["joint"]:
        axes[0].plot([0, 1, 2], pct([r["macro_accuracy"] for r in s["history"]]), marker="o", alpha=0.7, label=str(s["seed"]))
    axes[0].axhline(5, ls=":", color="gray", label="Uniform chance")
    axes[0].set(xlabel="Passes through training pool", ylabel="Held-out macro accuracy (%)", title="Real-data learnability reference", xticks=[0, 1, 2], ylim=(0, 100))
    axes[0].legend(fontsize=8, ncols=2, title="Exposed development seed")
    for arm in ("sequential", "repeat", "replay"):
        runs = grouped[arm]
        if not runs:
            continue
        curves = np.array([pct([r["macro_accuracy"] for r in s["history"]]) for s in runs])
        for curve in curves:
            axes[1].plot(range(11), curve, color=COLORS[arm], alpha=0.2, lw=1)
        axes[1].plot(range(11), curves.mean(0), color=COLORS[arm], marker="o", lw=2, label=NAMES[arm])
    axes[1].axhline(5, ls=":", color="gray")
    axes[1].set(xlabel="Completed two-class blocks", ylabel="Held-out macro accuracy, all 20 classes (%)", title="Learning and retention under a class stream", xticks=range(0, 11, 2), ylim=(0, 100))
    axes[1].legend(fontsize=8)
    fig.savefig(REPORT / "learning-curves.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
    for i, arm in enumerate(NAMES):
        runs = grouped[arm]
        for j, s in enumerate(runs):
            ax.scatter(i + (j - 1.5) * 0.045, 100 * s["final_macro_accuracy"], color=COLORS[arm], s=35)
        if runs:
            values = pct([s["final_macro_accuracy"] for s in runs])
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0, fmt="_", color="black", markersize=24, capsize=5)
    ax.axhline(5, ls=":", color="gray", label="Uniform chance")
    ax.set(xticks=range(4), xticklabels=["Joint IID\n2 passes", "Sequential\n1 update", "Repeat current\n2 updates", "Replay history\n2 updates"], ylabel="Final held-out macro accuracy (%)", ylim=(0, 100), title="Every completed seed; black bars are mean ± sample SD")
    fig.savefig(REPORT / "final-accuracy.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    for arm in ("sequential", "repeat", "replay"):
        runs = grouped[arm]
        if not runs:
            continue
        values = np.array([[a["mean_heldout_hidden_spike_fraction"] for a in activity if a["run_id"] == s["run_id"]] for s in runs])
        for curve in values:
            ax.plot(range(11), pct(curve), color=COLORS[arm], alpha=0.2)
        ax.plot(range(11), pct(values.mean(0)), marker="o", color=COLORS[arm], label=NAMES[arm])
    ax.set(xlabel="Completed two-class blocks", ylabel="Mean hidden spikes per bin (%)", title="Measured held-out activity; no learning during evaluation", ylim=(0, None))
    ax.legend(fontsize=8)
    fig.savefig(REPORT / "hidden-activity.png")
    plt.close(fig)
    if grouped["repeat"] and grouped["replay"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        im = None
        for ax, arm in zip(axes, ("repeat", "replay")):
            matrix = np.mean([s["task_accuracy_matrix"] for s in grouped[arm]], axis=0)
            matrix[np.triu_indices(10, 1)] = np.nan
            im = ax.imshow(pct(matrix), vmin=0, vmax=100, cmap="viridis", origin="upper", aspect="equal")
            ax.set(xticks=range(10), xticklabels=range(1, 11), yticks=range(10), yticklabels=range(1, 11), xlabel="Task position in each seed's stream", ylabel="Checkpoint after block", title=NAMES[arm])
        fig.colorbar(im, ax=axes, label="Held-out task accuracy (%)", shrink=0.85)
        fig.suptitle("Retention matrices averaged by stream position; future tasks masked")
        fig.savefig(REPORT / "retention-matrices.png")
        plt.close(fig)
    if grouped["joint"] and grouped["replay"]:
        keys = json.loads((REPORT / "data-audit.json").read_text())["class_keys"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), constrained_layout=True)
        for ax, arm in zip(axes, ("joint", "replay")):
            confusion = np.sum([s["history"][-1]["confusion"] for s in grouped[arm]], axis=0)
            normalized = confusion / confusion.sum(1, keepdims=True)
            im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            ax.set(xticks=range(20), xticklabels=keys, yticks=range(20), yticklabels=keys,
                   xlabel="Predicted word", ylabel="True word", title=NAMES[arm])
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(im, ax=axes, label="Fraction of each true class", shrink=0.8)
        fig.suptitle("Final held-out confusion; English and German digits")
        fig.savefig(REPORT / "confusion-matrices.png")
        plt.close(fig)
    rows = "".join(f"<tr><td>{NAMES[e['arm']]}</td><td>{e['n']}</td><td>{e['macro_accuracy_mean_pct']:.2f}</td><td>{e['macro_accuracy_sd_pp']:.2f}</td><td>{e['overall_accuracy_mean_pct']:.2f}</td><td>{e['mean_wall_seconds']/60:.1f}</td><td>{e['max_peak_rss_bytes']/2**20:.0f}</td></tr>" if e['macro_accuracy_sd_pp'] is not None else f"<tr><td>{NAMES[e['arm']]}</td><td>1</td><td>{e['macro_accuracy_mean_pct']:.2f}</td><td>—</td><td>{e['overall_accuracy_mean_pct']:.2f}</td><td>{e['mean_wall_seconds']/60:.1f}</td><td>{e['max_peak_rss_bytes']/2**20:.0f}</td></tr>" for e in endpoints)
    single_class = sum(d["number_of_predicted_classes"] == 1 for d in prediction_diagnostics if d["arm"] == "sequential")
    retention_rows = "".join(f"<tr><td>{NAMES[e['arm']]}</td><td>{e['final_old_class_macro_mean_pct']:.2f}</td><td>{e['final_latest_class_macro_mean_pct']:.2f}</td><td>{e['mean_forgetting_pp']:.2f}</td></tr>" for e in endpoints if e["arm"] != "joint")
    paired = f"Replay minus repeated-current updates: mean {report['paired_mean_pp']:.2f} percentage points, sample SD {report['paired_sd_pp']:.2f}, over {len(pairs)} paired exposed seeds." if len(pairs) > 1 else "Paired variability is not yet estimable from the available complete runs."
    figures = "".join(f'<figure><img src="{name}"><figcaption>{caption}</figcaption></figure>' for name, caption in [
        ("learning-curves.png", "Thin lines show individual seeds; thick lines show their mean. All scores use the complete 20-class vocabulary without a task mask."),
        ("final-accuracy.png", "Descriptive sample variation across initializations and streams, not confidence intervals or a powered confirmation."),
        ("retention-matrices.png", "Rows show evaluation after each block; columns show task position. Different seeds use different class orders."),
        ("confusion-matrices.png", "Each row is normalized by its true class. Exact confusion counts are available in the raw run records."),
        ("hidden-activity.png", "Secondary exploratory diagnostic from the saved hidden spike counts. Association with accuracy does not identify a cause.")]
        if (REPORT / name).exists())
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>SHD real-data pilot</title><style>
body{{font:17px/1.6 system-ui,sans-serif;max-width:1080px;margin:48px auto;padding:0 24px;color:#20303c}}h1{{font-size:34px;line-height:1.15}}h2{{font-size:23px;margin-top:36px}}.scope{{background:#edf4f4;border-left:5px solid #188e88;padding:16px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{border-bottom:1px solid #ccd8df;text-align:left;padding:10px}}img{{width:100%}}figure{{margin:30px 0}}figcaption{{font-size:14px;color:#52626f}}code{{font-size:14px}}a{{color:#087e79}}</style>
<h1>Real SHD recordings: learning and forgetting in the audio component</h1>
<p>Exploratory local pilot · registered November 3, completed November 4, 2026 · source c4c10b285f · exposed development seeds 8000–8003</p>
<p class="scope">These are measurements of the existing, fixed-width e-prop audio learner. The complete SNDL Engine remains blocked by G9_A_UNREADY. This study does not demonstrate growing state, recursive self-improvement, or superiority over transformers or other SNNs.</p>
<p>{len(summaries)} completed runs; {len(failures)} recorded algorithm/run failures; {len(interruptions)} interrupted infrastructure attempt retained; {verified} saved prediction checkpoints independently rescored. The interrupted attempt is not an additional independent seed. Official SHD test data was never downloaded or used.</p>
<h2>Data and method</h2><p><a href="https://zenkelab.org/datasets/shd/">Official SHD training archive</a> (CC BY 4.0), publisher MD5 verified. {manifest['train_recordings']:,} training recordings from eight speakers; {manifest['dev_recordings']:,} development recordings from speakers 10 and 11. All 20 classes occur in both sets. Inputs are 100 temporal bins × 700 channels of spike counts. All {manifest['total_events']:,} events are preserved; per-recording duration normalization removes absolute duration. Model: 64 recurrent LIF units, source-default dynamics and gains, Adam 0.002, batch 16, CPU, one thread, GIL off.</p>
<p>Sequential arms see ten two-class blocks. The replay arm uses at most 400 earlier training recordings in a label-blind reservoir; it and the repeated-current control receive exactly matched updates and batch sizes. A 400-frame logical replay payload is 56 MB (53.4 MiB), in addition to the shared dataset cache. Evaluation covers all held-out recordings with learning disabled and unchanged model hashes.</p>
<h2>Final held-out results</h2><table><thead><tr><th>Arm</th><th>Seeds</th><th>Macro accuracy (%)</th><th>Sample SD (pp)</th><th>Overall accuracy (%)</th><th>Mean run (min)</th><th>Max worker RSS (MiB)</th></tr></thead><tbody>{rows}</tbody></table><p>{paired}</p><p>{single_class} of {len(grouped['sequential'])} one-update sequential runs ended predicting a single class. A constant one-class prediction scores exactly 5% macro accuracy across 20 classes; this is a measured collapse, not a rounded or fabricated progress indicator. Exact prediction counts and overall accuracy are retained.</p>
<h2>Retention and recent learning</h2><table><thead><tr><th>Arm</th><th>Earlier 18 classes, macro (%)</th><th>Latest 2 classes, macro (%)</th><th>Peak-relative forgetting (pp)</th></tr></thead><tbody>{retention_rows}</tbody></table><p>These class subsets are scored after global 20-class prediction, without an output mask. Peak-relative forgetting measures the drop from the best earlier checkpoint on each of the first nine tasks. Higher learned peaks can produce larger subsequent drops even alongside higher final accuracy; the endpoints answer different questions. The independent analysis corrected the collector's secondary summary formula, which clipped improvement on earlier tasks to zero. Raw collector summaries are retained; this report uses the registered best-earlier-minus-final definition without clipping. Primary accuracy scores are unaffected.</p>
{figures}
<h2>Interpretation and limits</h2><p>The IID reference tests whether this representation and update rule can learn the real recordings. The sequential contrasts test forgetting and access to past examples in this component. Replay has historical-data access; its result is not an architecture comparison. Four previously exposed seeds and two held-out speakers cannot establish population-level generalization or a powered effect. The same development split informed feasibility; there is no final-test benchmark score.</p>
<p>Infrastructure incidents are retained in the deviation ledger: dataless Documents archive/source stalls, storage pressure, and loss of worker processes and temporary caches during the final seed. The partial attempt survives; its replacement uses the same seed, code, exact wheel hashes and byte-identical regenerated inputs. The completed-cohort exposure count excludes that interrupted attempt's additional recorded work. A fresh audit compared all 483 original exported files with the pinned GitHub tree; the five imported files were recovered and reverified after cache loss. Every run's imported hashes match the audit. Resource figures measure worker peak RSS and run wall time, not total system memory, energy or calendar duration including setup/interruption.</p>
<h2>Next research steps</h2><p>Use the paired variance and measured throughput to design a separately registered component confirmation cohort; keep the official test sealed until the frozen analysis. This replay-versus-repeat variance cannot size a growth-versus-fixed-capacity engine study. Complete and verify the engine's retention selection, source/admission, G6 progress carrier and durable final-retention bindings, then introduce an event-data Problem adapter. Only after that readiness trajectory should growth, reservoir/feed/return, saved state and recursive improvement receive a full-engine comparison. Extend to SSC after that route and its controls work. Its 100-bin UInt16 cache alone would need about 14.8 GB for all official splits, before raw archives and working space; given observed storage pressure, use chunked preparation or streaming for that extension.</p>
<p><a href="pilot-results.json">Machine-readable results</a> · <a href="data-manifest.json">Dataset provenance and checks</a> · <a href="source-audit.json">Fresh source audit</a> · <a href="../PROTOCOL.md">Pre-outcome protocol</a> · <a href="../DEVIATIONS.md">Deviations</a></p></html>'''
    (REPORT / "pilot-report.html").write_text(document)
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "source_files", "failures", "heldout_hidden_activity", "final_speaker_metrics")}, indent=2))


if __name__ == "__main__":
    main()
