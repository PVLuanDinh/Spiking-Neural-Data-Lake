# SNDL Research Context Packet

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

`source/repo` is the exact tracked source snapshot at 26501ed3d0ac118c227b0f1919f520844e564948, excluding the `research` directory of literature PDFs. No untracked files, credentials, .git object database, environment, or live home are copied. It is a source export, not a fully installed runnable application.

`history` contains the original proposal and earlier presentation planning documents. They describe historical intent and earlier evidence; current claims must follow the new report and actual records.

## Recover the actual local commits

The implementation has not been pushed. `engine-cycle-integration.bundle` contains the local branch commits since base c4c10b285f1741aa798d5ee9c5296ae371fabecf. An existing clone must contain that base revision. From such a clone, replace the bundle path below with its absolute extracted location:

```sh
git bundle verify /absolute/path/engine-cycle-integration.bundle
git fetch /absolute/path/engine-cycle-integration.bundle engine-cycle-integration:engine-cycle-integration
git switch engine-cycle-integration
git rev-parse HEAD
```

The last output must be 26501ed3d0ac118c227b0f1919f520844e564948. If the local branch already exists, inspect it first and choose another meaningful branch name rather than overwriting it. Do not use a codex prefix.

The patch is an alternative for a clean checkout of the exact base, not something to apply on top of the implementation bundle. `git apply --check` can validate it before application. The patch does not preserve commit history; the bundle does.

## Before running code

Read source/repo/AGENTS.md, docs/AGENT_COLD_START.md, docs/POLICIES.md, SECURITY.md, and evidence/engine-integration/RUNTIME.md. Rebuild a compatible isolated runtime on the receiving host. The copied shell wrappers contain original host paths. Use new experiment IDs and isolated homes, do not overwrite evidence, and distinguish exploratory development from confirmatory experiments.

## Verify transfer integrity

MANIFEST.json records every payload member's size and SHA256. The manifest excludes itself. PACKET_BUILD_RECEIPT.json, delivered alongside the ZIP, records the ZIP hash and verification result. The packet retains original raw records; compact summaries do not replace them.
