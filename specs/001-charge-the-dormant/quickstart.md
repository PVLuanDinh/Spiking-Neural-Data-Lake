# Quickstart: reproducing every verdict in this feature

Every command below is runnable from the repo root. None of them touches the operator's live brain.

## Prerequisites — and why each one matters

```powershell
$env:PYTHONPATH = "D:\AI-project\spiking-neural-data-lake\src;D:\AI-project\spiking-neural-data-lake"
$env:NORD_HOME  = "$env:TEMP\charge_ab"
```

- **Repo root on `PYTHONPATH`, not just `src`.** With `src` alone the top-level `training/` package
  is unimportable, `cortex/brain.py`'s eprop encoder import raises, and the handler silently sets
  `_perception_tier="stdlib"` / `_perceiving_real=False`. The brain then perceives nothing and a
  probe reports NOT-ELIGIBLE — a property of the harness, not the brain. This has already produced
  one false verdict in this ledger.
- **`NORD_HOME` on a temp dir.** The daemon appends to the real lake continuously, and brain
  construction reads it. Without isolation, two seed-0 brains built seconds apart differ ~42% of
  the time.

## 1. The current dormancy picture

```powershell
.\.venv-lake\Scripts\python.exe -m snn_data_lake.substrate.method_dormancy_gate
```

Expect: `0 dormant, 16 UNCERTAIN reported not asserted`, 1 open exemption. **`uncertain` is a list
of questions, not failures** — a red gate here would create pressure to silence them.

## 2. A neuter-A/B for a rule that IS called

The pattern the 2026-07-31 adjudication used: patch the method to a no-op, re-run at the same seed,
diff behavioural hashes (actions, coherence, elapsed, bound) against a baseline run.

A verdict is one of: a behavioural hash moved (charge), only telemetry moved (inert — the
`AffectiveAppraisal` shape), or nothing moved.

Record it so a later reader can re-execute:

```powershell
.\.venv-lake\Scripts\python.exe tools\coverage_score.py record --name <name> --kind ab `
  --by "<who> / <TIER>" -- <the probe command>
```

**The `--by` field must name the perception tier.** `AffectiveAppraisal.learn` charges in the
stdlib tier and is inert in eprop-snn; a tier-free verdict is not a verdict.

## 3. A drive-source A/B (the built-in neuter)

Only for the eleven undirected drive sources:

```powershell
.\.venv-lake\Scripts\python.exe tools\charge_probe.py drive 60
```

Reads: `act/hold decision changed`, `arm trajectory changed`, and the graded `internal reads
changed: N of M ticks` with top state fields by |delta|. The known-good control gives 0/60 flips
but 55 of 60 ticks of changed internal reads — decisive-looking nulls on the flip counts alone are
masked negatives, because ten other voices carry the sum.

## 4. Verify an artifact still reproduces

```powershell
.\.venv-lake\Scripts\python.exe tools\coverage_score.py --rerun <name>
```

Note the flag spelling: `--rerun NAME`. A positional `rerun NAME` is refused with exit 2 — it used
to fall through to a full report and exit 0, which once made a sweep report artifacts as
reproducing while rerunning none of them.

## 5. The three gates

```powershell
.\.venv-lake\Scripts\python.exe run_selfchecks.py      # expect 277/277
.\.venv-lake\Scripts\python.exe -m pytest -q           # expect exit 0
.\.venv-lake\Scripts\python.exe -m ruff check .        # expect All checks passed
.\.venv-lake\Scripts\python.exe tools\coverage_score.py score --check   # expect exit 0, 26.4%
```

## Acceptance walkthrough

The feature is done when, for each of the nine rules, this sequence answers in one line:

1. `method_dormancy_gate` shows its disposition, or it no longer appears as a question.
2. Its artifact re-runs to the same verdict under `--rerun`.
3. Its ledger row either cites the artifact, or states why the band did not move.

**What "done" is not**: the gate reporting zero. A gate reports zero when everything is resolved,
and also when everything has been silenced. The artifacts are what tell the two apart.
