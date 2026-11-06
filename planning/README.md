# Research showcase planning and evidence

This directory imports the existing research workspace materials on November 5,
2026. It accompanies the bounded Engine implementation at source commit
`26501ed3d0ac118c227b0f1919f520844e564948`, based on
`c4c10b285f1741aa798d5ee9c5296ae371fabecf`. The imported files retain their original
bytes, dates stated in their text, failed attempts, and limitations. Publication
of the archive does not rerun or certify an experiment.

## Start here

- `handoff/SNDL_RESEARCH_HANDOFF_2026-11-04.md`: English research status and
  decision report, with an evidence index.
- `handoff/CONTEXT.json` and `handoff/SHD_PER_SEED_METRICS.csv`: compact scope,
  provenance, and individual recorded SHD results.
- `experiments/scale/PROTOCOL.md`, `RUNBOOK.md`, `DEVIATIONS.md`, and `LEDGER.md`:
  the exploratory SHD pilot design and execution record.
- `experiments/scale/reports/pilot-results.json` and `pilot-report.html`: pilot
  analysis. `runs/` retains prediction files and the interrupted attempt.
- `experiments/engine-integration/LEDGER.md`, `completion.json`, and
  `cycle-04-independent-audit.json`: bounded full-Engine integration evidence.
  The matching implementation record is
  `../specs/022-unified-brain-arc/evidence/g9-engine-cycle-integration.md`.
- `experiments/engine-integration/RUNTIME.md`: requirements and original runtime
  provenance. Source overlays preserve the actual phase code; they do not
  replace the canonical implementation under `src/`.
- `reports/`, the top-level planning documents, and `experiments/engine/` retain
  earlier diagnostic work and its artifacts. They are historical records,
  not interchangeable with the later SHD or full-Engine experiments.

## Scope and corrections when reading the archive

The SHD pilot measures a fixed-width audio learning component, not the complete
Engine. Its results do not establish growth, effective continual learning, or
recursive self-improvement. Joint training is a short reference run, not a
converged benchmark. Replay improves final accuracy over the matched repeat
control but does not improve the registered mean-forgetting endpoint.

The Engine demonstration uses explicit task context, corrective feedback, a
protected old core, and repeated constructed inputs. The grown-unit and return
ablations establish dependence within that fixture. They do not establish an
advantage over an equal-capacity system or generalization to new semantic inputs.
The earlier all-plastic cycle and protected-core cycle use different source
versions; their difference cannot be attributed solely to freezing the core.

The November 4 handoff files are preserved snapshots. Their statements that the
implementation was unpushed and that the presentation was outdated describe the
handoff date. The presentation has since been revised in its native Space:
https://chatgpt.com/space/page_25fc2658473c819194876bbbe931bd18
This repository does not contain an exported copy of that live deck.

## Integrity and reproduction

`SNAPSHOT_MANIFEST.json` lists the imported payloads with byte sizes and SHA256
hashes, plus every excluded file and its reason. `IMPORT_VERIFICATION.json`
records copy comparisons, Python syntax parsing, JSON/JSONL parsing, prediction
archive CRC checks, and the publication-time credential-pattern scan. These are
archive checks, not fresh behavioral tests or a full operator battery.

The receipt also lists historical stdout captures with a `.json` suffix that
are not valid JSON. They are retained unchanged as diagnostic records and must
not be treated as successfully parsed structured results. Imported payloads
were read directly or recovered from the existing transfer packet, checking
its per-file SHA256 and the original source size/mtime; the manifest records
the copy source for each file.

Original log whitespace and the metrics CSV's CRLF line endings are also kept.
The receipt identifies the resulting archival-only `git diff --check` warnings;
the import does not normalize or rewrite measured records to suppress them.

The original host shell wrappers and packet builder are archived unchanged.
They contain original workspace paths and layout assumptions; they are not
portable launchers for this imported location. In particular, `run-engine.sh`
defaults to the baseline source snapshot. A future rerun must configure a
compatible isolated environment and explicitly select the intended Engine
source with `ENGINE_SOURCE_ROOT`. Follow the relevant protocol and runtime
record, use a new output directory and isolated home, and preserve these records.

Raw/prepared SHD data, incomplete downloads, Python environments and bytecode,
isolated runtime homes and lake images, and duplicate ZIP/Git transfer exports
are excluded. Small scoring/split metadata arrays and raw prediction archives
are included so that analysis can be inspected independently of training.
The older packet manifests describe the separately delivered ZIP, not the
contents of this directory; use `SNAPSHOT_MANIFEST.json` for this Git snapshot.

This branch has not been rebased or integration-tested against the newer remote
`main`. The operator's full verification and review remain necessary before
landing the proposed changes there. No safety-envelope file or execution
certification record is changed by this archive import.
