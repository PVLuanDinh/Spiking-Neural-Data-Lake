# Reproducing the component pilot

The upstream source is read-only. No Git branch was created. Scripts and results live in `planning/experiments/scale`; the regenerable dataset and dependency caches live under `/private/tmp` to avoid observed macOS dataless-file stalls in Documents.

## Environment

- Training: workspace `.venv-04/bin/python`, CPython 3.14.6 free-threaded, Torch 2.15.0 CPU, NumPy 2.5.4, GIL off. `runtime.sh` sets one numerical thread and a local Python bytecode cache.
- Runtime packages: `/private/tmp/sndl-class-packages-749rpbye/site`.
- Source: `/private/tmp/sndl-class-source-tlefts_z/src`; original export verified against 483 Git blobs at `c4c10b285f1741aa798d5ee9c5296ae371fabecf`. Current imported-file hashes are recorded in each run. Fresh Git reads remain affected by dataless pack files; no current verification is falsely claimed.
- Preparation: bundled CPython 3.12 plus `/private/tmp/sndl-shd-prep-site` (h5py 3.16.0, NumPy 2.5.4).
- Plotting: same CPython 3.12 plus `/private/tmp/sndl-shd-plot-site`; exact installation receipt in `diagnostics/plot-install.json`.
- Cached dataset location and manifest hash: `data/cache-location.json`. Cache is temporary and may need regeneration after OS cleanup. Do not treat it as an archival location.

## Preparation and verification

Download `https://zenkelab.org/datasets/shd/shd_train.h5.gz` to `/private/tmp/sndl-scale-data-20261103/shd_train.h5.gz.part`. The preprocessor verifies publisher MD5 before accepting it. Do not download the official test for exploratory tuning.

From the workspace root, with `prep_python` set to the bundled Python executable:

```zsh
PYTHONPATH=/private/tmp/sndl-shd-prep-site "$prep_python" planning/experiments/scale/prepare_shd.py
PYTHONPATH=/private/tmp/sndl-shd-prep-site "$prep_python" planning/experiments/scale/verify_data.py
```

Prepared manifests are immutable: an existing complete cache is intentionally refused by the preparation script. Reuse its verified manifest rather than overwriting it. If regenerating after cache loss, retain the prior workspace receipts and compare new array hashes with `reports/data-manifest.json`.

## Training and analysis

`pilot.py` refuses to overwrite an existing run directory. To reproduce a single arm, use a fresh run ID:

```zsh
zsh planning/experiments/scale/runtime.sh planning/experiments/scale/pilot.py \
  --run-id replica-joint-s8000 --seed 8000 --arm joint --max-seconds 3600
```

The four allowed arms are `joint`, `sequential`, `repeat`, and `replay`; registered exposed seeds are 8000–8003. `campaign.py` serially runs the original `shd-*` cohort after the first positive-control gate and enforces the aggregate four-hour cap. It is a fresh-cohort launcher, not a resume/overwrite command. Keep failed and completed original directories.

To rescore the original cohort and regenerate figures/report:

```zsh
PYTHONPATH=/private/tmp/sndl-shd-plot-site \
MPLCONFIGDIR=/private/tmp/sndl-scale-mpl XDG_CACHE_HOME=/private/tmp/sndl-scale-cache \
"$prep_python" planning/experiments/scale/analyze.py --run-prefix shd-
```

Use `--run-prefix replica-` for independently named reproduction outputs so duplicate seeds are not pooled into the original cohort. The analyzer verifies saved-file hashes, IDs, raw-logit argmaxes, scoring, matching initial weights, identical paired streams, and matched replay/repeat updates/exposures. It never trains a model or reads official test data.

## Scope

This is an event-audio component study, not a full-engine study, a persistence test, or a transformer comparison. No learned lake state is claimed or written. The G9 readiness refusal is preserved. See `PROTOCOL.md`, `DEVIATIONS.md`, raw `runs/`, and the generated HTML/JSON reports. Keep all negative outcomes and interpret speaker holdout and seed variation within their actual limits.

## Fresh source audit

The local Git pack stall was subsequently bypassed with a read-only GitHub tree query at the pinned SHA. All 483 exported blobs freshly matched. The saved expected tree is `reports/source-expected-blobs.json`; verify it locally with:

```zsh
"$prep_python" planning/experiments/scale/verify_source.py
```

`reports/source-audit.json` records every file's Git blob ID and SHA256. The analyzer checks every run's imported source hashes against this fresh audit. Earlier immutable start logs correctly record that the fresh verification had not happened at launch.

## Recovery after the observed local cache loss

The final run continued on November 4. The infrastructure interruption and exact retry are retained separately; no completed seed was replaced. The retried run reproduced all nine shared checkpoints bit for bit. See `reports/data-recovery-audit.json`, `reports/source-recovery-audit.json`, and `reports/retry-reproduction-audit.json`.

Exact wheel archives, authenticated source-blob responses, the SHD archive and original manifests are now retained under the second writable workspace root, in `shd-recovery`. `runtime.sh` uses its persistent bytecode cache. Current training sources are the five imported, reverified files; the historical full 483-file audit is retained. Use `verify_source.py --imported-only` for the recovered source subset.

To recreate lost temporary packages/source/data while preserving the original cohort's identity:

```zsh
"$prep_python" planning/experiments/scale/recover_caches.py
"$prep_python" planning/experiments/scale/restore_prepared.py
"$prep_python" planning/experiments/scale/verify_source.py --imported-only
```

The recovery helper verifies every wheel/source/archive hash and the preparation helper checks every array against the original receipts. It restores the exact original manifest after a byte-identical rebuild. Do not overwrite the original receipts with new preparation timing metadata.

The final report can be regenerated without the temporary dataset or any training: `reports/evaluation-metadata` retains all small scoring arrays with their original hashes. Add `--metadata-only` to the analysis command. Matplotlib now defaults to its documented `MPL_IGNORE_SYSTEM_FONTS=1` option, avoiding the stalled macOS system-font scan. The `analyze.py` outputs are canonical for the secondary forgetting metric: the frozen collector's original summary had clipped later improvements; the final analysis recomputes best-earlier-minus-final from raw matrices. Classification scores are unaffected.
