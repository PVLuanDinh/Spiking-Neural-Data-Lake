"""Build a local, transferable research handoff without training or source edits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
REPO = WORKSPACE / "repo"
BASE = "c4c10b285f1741aa798d5ee9c5296ae371fabecf"
BRANCH = "engine-cycle-integration"
NAME = "SNDL_CONTEXT_PACKET_2026-11-04"
REPORT = "SNDL_RESEARCH_HANDOFF_2026-11-04.md"


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True
    ).stdout


def load(rel: str):
    return json.loads((WORKSPACE / rel).read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    revision = git("rev-parse", "HEAD").decode().strip()
    if not revision.startswith("26501ed3"):
        raise RuntimeError("Source revision changed; review and update the handoff first")
    if git("status", "--porcelain"):
        raise RuntimeError("Source is no longer clean; do not package an ambiguous snapshot")

    scale = load("planning/experiments/scale/reports/pilot-results.json")
    data = load("planning/experiments/scale/reports/data-manifest.json")
    audit = load("planning/experiments/engine-integration/cycle-04-independent-audit.json")
    completion = load("planning/experiments/engine-integration/completion.json")
    context = {
        "schema": "sndl.research.handoff.v1",
        "as_of_date": "2026-11-04",
        "timezone": "America/Chicago",
        "purpose": "Evidence-grounded teammate and Claude review before further research decisions",
        "source": {
            "origin": git("remote", "get-url", "origin").decode().strip(),
            "base": BASE,
            "branch": BRANCH,
            "commit": revision,
            "source_working_tree_clean": True,
            "implementation_pushed": False,
            "main_merged": False,
            "full_operator_battery_performed": False,
        },
        "showcase": {
            "timing": "in a few days; exact date not supplied",
            "duration_minutes_approx": 10,
            "audience": "technical, potentially unfamiliar with this topic or AI",
            "beta_page_id": "page_25fc2658473c819194876bbbe931bd18",
            "beta_url": "https://chatgpt.com/space/page_25fc2658473c819194876bbbe931bd18",
            "beta_status": "outdated; predates the SHD pilot and complete Engine cycle",
            "presentation_edited_in_this_handoff": False,
        },
        "assessment": {
            "bounded_full_engine_integration_demonstrated": True,
            "effective_real_data_continual_learning_demonstrated": False,
            "growth_superiority_demonstrated": False,
            "recursive_self_improvement_demonstrated": False,
            "recommended_next_step": "Diagnose basic SHD learnability before a larger integrated efficacy campaign",
        },
        "shd": {
            "scope": scale["scope"],
            "metric_units": "accuracy in percent; variability and differences in percentage points",
            "variability": "sample standard deviation across four exposed development seeds, not confidence intervals",
            "seeds": [8000, 8001, 8002, 8003],
            "data": {k: data[k] for k in [
                "dataset", "url", "publisher_md5", "archive_sha256", "bins", "channels",
                "classes", "framing", "recordings", "train_recordings", "dev_recordings",
                "held_speakers", "train_speakers", "total_events", "largest_bin_count",
                "all_events_conserved", "split_disjoint", "all_classes_each_split", "official_test_read"
            ]},
            "completed_runs": scale["completed_runs"],
            "initialized_training_attempts": scale["initialized_training_attempts"],
            "prediction_checkpoints_independently_rescored": scale["independently_verified_prediction_checkpoints"],
            "infrastructure_interruptions": scale["infrastructure_interruptions"],
            "frozen_training_fingerprint": scale["frozen_training_fingerprint"],
            "endpoints": scale["endpoints"],
            "paired_results": scale["paired_results"],
            "paired_mean_pp": scale["paired_mean_pp"],
            "paired_sd_pp": scale["paired_sd_pp"],
            "total_completed_cohort_training_exposures": scale["total_exposures"],
            "summed_training_run_wall_seconds_excludes_setup_and_recovery": scale["total_training_run_wall_seconds"],
            "cautions": [
                "Fixed width 64; no full Brain, structural growth, or persistence experiment",
                "Joint reference was only two passes, not a converged upper bound",
                "Replay and repeat match updates and batch sizes; sequential has half the exposures",
                "Replay improves final accuracy but not the registered mean forgetting endpoint",
                "Development-only split and exposed seeds; no confirmatory significance claim",
                "Per-recording time normalization removes absolute duration",
                "Secondary forgetting summary corrected from raw matrices; training and accuracy unchanged",
            ],
        },
        "engine": {
            "scope": "single exposed fixture, actual persisted full Brain, task-conditioned, no-prune",
            "seed": 9900,
            "curriculum": ["aii", "ii.xor"],
            "unique_inputs_per_task": 4,
            "presentations_per_validation_or_final_task": 16,
            "final_set_uses_unseen_semantic_inputs": False,
            "feedback_gate": "corrective",
            "old_core_during_B": "frozen; grown outputs remain plastic",
            "candidate_pool": 1024,
            "admitted_batch_size": 18,
            "batch_size_is_optimal_or_autonomously_discovered": False,
            "audit": audit,
            "completion": completion,
            "cautions": [
                "All-plastic full-Engine control: B16/16, A0/16; final selection refused",
                "Grown/return ablations show dependence, not superiority over equal-capacity controls",
                "Feed changes internal features but no answer-accuracy effect established",
                "No full-Engine SHD, independent efficacy cohort, uncued composition, pruning benefit, or RSI",
                "Source-owned BETA for a later positive-spend task remains unset",
                "Original persistent lake and host runtime are not included in this packet",
            ],
        },
        "next_experiments_proposed_not_executed": [
            "Small fixed balanced subset fitting and learner diagnostics",
            "Longer joint training with training/development convergence curves",
            "Simple reference learner on identical split and representation",
            "Analysis of incremental collapse, confusion, reservoir coverage, and per-speaker errors",
            "Later growth-on/off and equal-capacity full-Engine efficacy comparison",
        ],
        "operational_context": {
            "host_used": "Apple M2 Mac, 8 GB RAM, CPU",
            "python_used": "CPython 3.14.6 free-threaded with GIL off",
            "torch_used": "2.15.0 CPU",
            "numpy_used": "2.5.4",
            "native_build": "external Apple Silicon cp314t extension; not included",
            "branch_preference": "Never use a codex prefix",
            "wrapper_caveat": "run-engine.sh defaults to baseline; set ENGINE_SOURCE_ROOT for repaired source",
            "do_not_use_audio_wrapper_for_full_Brain": True,
        },
    }
    write_json(HERE / "CONTEXT.json", context)

    with (HERE / "SHD_PER_SEED_METRICS.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "macro_accuracy_pct", "overall_accuracy_pct",
                         "exposures", "updates", "old_class_macro_pct", "latest_class_macro_pct",
                         "mean_forgetting_pp"])
        for run in sorted(scale["runs"], key=lambda r: (r["seed"], r["arm"])):
            writer.writerow([
                run["arm"], run["seed"], 100 * run["final_macro_accuracy"],
                100 * run["final_accuracy"], run["exposures"], run["updates"],
                *[100 * run[k] if k in run else "" for k in [
                    "final_old_class_macro_accuracy", "final_latest_class_macro_accuracy", "mean_forgetting"]],
            ])

    patch = HERE / "engine-cycle-integration.patch"
    patch.write_bytes(git("diff", "--binary", BASE, revision))
    bundle = HERE / "engine-cycle-integration.bundle"
    if bundle.exists():
        raise RuntimeError("Refusing to overwrite an existing Git bundle")
    git("bundle", "create", str(bundle), BRANCH, "^" + BASE)
    verification = subprocess.run(
        ["git", "bundle", "verify", str(bundle)], cwd=REPO,
        check=True, capture_output=True, text=True
    )
    (HERE / "GIT_BUNDLE_VERIFICATION.txt").write_text(verification.stdout + verification.stderr)

    readme = """# SNDL Research Context Packet

This packet is a November 4, 2026 snapshot for a teammate and Claude-assisted review. It does not run an experiment or grant access to the original computer, private presentation, or persistent lake.

## Read first

1. SNDL_RESEARCH_HANDOFF_2026-11-04.md is the full English decision report.
2. CLAUDE_START_HERE.md is a ready-to-paste review request.
3. CONTEXT.json contains compact machine-readable scope, metrics, caveats, and source identity.
4. SHD_PER_SEED_METRICS.csv preserves the individual seed results.
5. Consult the report's evidence index before inspecting larger files.

For Claude on the web, extract the ZIP yourself and upload those first three files. Add the relevant protocols and result JSONs next. Do not assume that a subscription grants local/GitHub/Page access or that every binary prediction record can be read as text. For a coding agent, point it at the extracted directory. Ask it to report which files it actually inspected.

## Evidence and source

`evidence/scale` preserves the pilot scripts, protocol, deviations, reports, scoring metadata, and completed/interrupted prediction files. Raw/prepared SHD data and temporary caches are excluded. Model training requires verified data regeneration and a compatible environment.

`evidence/engine-integration` preserves scripts, phase source overlays, receipts, failures, diagnostic probes, and test records. Installed packages, compiled extensions, and the actual persisted lake are excluded. A restore receipt is evidence of a restore, not a restorable lake image.

`source/repo` is the exact tracked source snapshot at COMMIT_PLACEHOLDER, excluding the `research` directory of literature PDFs. No untracked files, credentials, .git object database, environment, or live home are copied. It is a source export, not a fully installed runnable application.

`history` contains the original proposal and earlier presentation planning documents. They describe historical intent and earlier evidence; current claims must follow the new report and actual records.

## Recover the actual local commits

The implementation has not been pushed. `engine-cycle-integration.bundle` contains the local branch commits since base c4c10b285f1741aa798d5ee9c5296ae371fabecf. An existing clone must contain that base revision. From such a clone, replace the bundle path below with its absolute extracted location:

```sh
git bundle verify /absolute/path/engine-cycle-integration.bundle
git fetch /absolute/path/engine-cycle-integration.bundle engine-cycle-integration:engine-cycle-integration
git switch engine-cycle-integration
git rev-parse HEAD
```

The last output must be COMMIT_PLACEHOLDER. If the local branch already exists, inspect it first and choose another meaningful branch name rather than overwriting it. Do not use a codex prefix.

The patch is an alternative for a clean checkout of the exact base, not something to apply on top of the implementation bundle. `git apply --check` can validate it before application. The patch does not preserve commit history; the bundle does.

## Before running code

Read source/repo/AGENTS.md, docs/AGENT_COLD_START.md, docs/POLICIES.md, SECURITY.md, and evidence/engine-integration/RUNTIME.md. Rebuild a compatible isolated runtime on the receiving host. The copied shell wrappers contain original host paths. Use new experiment IDs and isolated homes, do not overwrite evidence, and distinguish exploratory development from confirmatory experiments.

## Verify transfer integrity

MANIFEST.json records every payload member's size and SHA256. The manifest excludes itself. PACKET_BUILD_RECEIPT.json, delivered alongside the ZIP, records the ZIP hash and verification result. The packet retains original raw records; compact summaries do not replace them.
""".replace("COMMIT_PLACEHOLDER", revision)
    (HERE / "PACKET_README.md").write_text(readme)

    entries = []
    excluded = []
    output = HERE / (NAME + ".zip")
    if output.exists():
        raise RuntimeError("Refusing to overwrite an existing context packet")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        def add_bytes(member: str, payload: bytes, provenance: str) -> None:
            zf.writestr(NAME + "/" + member, payload)
            entries.append({"member": member, "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(), "provenance": provenance})

        for filename in [REPORT, "CLAUDE_START_HERE.md", "CONTEXT.json", "SHD_PER_SEED_METRICS.csv",
                         "PACKET_README.md", "engine-cycle-integration.patch", "engine-cycle-integration.bundle",
                         "GIT_BUNDLE_VERIFICATION.txt", "build_packet.py"]:
            add_bytes(filename, (HERE / filename).read_bytes(), "handoff")

        for family in ["scale", "engine-integration"]:
            parent = WORKSPACE / "planning" / "experiments" / family
            for path in sorted(parent.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(parent)
                if "__pycache__" in relative.parts or (family == "scale" and relative.parts[0] == "data"):
                    excluded.append({"workspace_member": str(path.relative_to(WORKSPACE)),
                                     "reason": "regenerable/raw data cache or Python bytecode"})
                    continue
                add_bytes("evidence/" + family + "/" + relative.as_posix(), path.read_bytes(),
                          str(path.relative_to(WORKSPACE)))

        history = [
            WORKSPACE / "planning/SHOWCASE_PLAN.md",
            WORKSPACE / "planning/SHOWCASE_DATA_AND_COMPARISON.md",
            WORKSPACE / "planning/SHOWCASE_DATA_EXAMPLE.json",
            Path("/Users/arthurkhvan/Downloads/SNDL-Engine-Project-Proposal-APA.pdf"),
        ]
        for path in history:
            if path.is_file():
                add_bytes("history/" + path.name, path.read_bytes(), "historical proposal/planning context")
            else:
                excluded.append({"member": path.name, "reason": "not accessible at packet build"})

        tracked = git("ls-tree", "-r", "--name-only", revision).decode().splitlines()
        top_levels = sorted({p.split("/")[0] for p in tracked if not p.startswith("research/")})
        proc = subprocess.Popen(["git", "archive", "--format=tar", revision, "--", *top_levels],
                                cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                if member.isfile():
                    stream = tf.extractfile(member)
                    assert stream is not None
                    add_bytes("source/repo/" + member.name, stream.read(), "Git " + revision)
                elif member.issym():
                    # Preserve the tracked link target as link content; no filesystem traversal.
                    add_bytes("source/repo/" + member.name, member.linkname.encode(), "Git symlink target")
        if proc.wait() != 0:
            raise RuntimeError(proc.stderr.read().decode() if proc.stderr else "git archive failed")

        manifest = {
            "schema": "sndl.handoff.manifest.v1", "source_commit": revision,
            "member_count_excluding_manifest": len(entries), "files": entries,
            "excluded": excluded,
            "excluded_source_directory": "research (third-party literature)",
            "excluded_external_assets": ["raw/prepared SHD data", "persistent brain_lake.bin",
                                         "installed runtime", "native wheel", "private presentation CRDT"],
        }
        manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode()
        zf.writestr(NAME + "/MANIFEST.json", manifest_payload)
        (HERE / "MANIFEST.json").write_bytes(manifest_payload)

    with zipfile.ZipFile(output) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for row in entries:
            payload = zf.read(NAME + "/" + row["member"])
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError("SHA256 mismatch: " + row["member"])

    receipt = {
        "source_commit": revision,
        "archive": str(output), "archive_bytes": output.stat().st_size,
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "manifest_members_verified": len(entries), "zip_crc_verified": True,
        "all_manifest_sha256_verified": True, "git_bundle_verified": True,
        "source_working_tree_clean_after": not bool(git("status", "--porcelain")),
        "experiments_run_in_this_handoff": False,
        "presentation_edited_in_this_handoff": False,
    }
    write_json(HERE / "PACKET_BUILD_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
