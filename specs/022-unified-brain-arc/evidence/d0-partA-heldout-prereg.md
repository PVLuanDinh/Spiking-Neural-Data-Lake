# D0 Part A: the held-out read of the sealed Allen V1 mice (pre-registration)

**Status: PRE-REGISTRATION, written before any sealed trace was read.** Owner instruction, 2026-11-03, verbatim: *"Write
the pre-registration for the sealed Allen mice"*. Author: cloud session `session_01TeA2yYimjAiHiGFzw6hqHX`, on branch
`cloud/drift-d0-instrument-1102`.

**It freezes when** the integrator commits it (with any panel or policy-enforcer folds) and the owner answers section 9.
Until then nothing in the sealed set is read.

**What exists to build on:**
- the instrument: `instruments/representational_drift.py`, with its six constructed reds;
- the development reading: `research/representational-drift/PARTA-DEV-2026-11-02.md` (4 mice, descriptive);
- the split: `tools/d0_partA_split_pin.json`;
- the units: `tools/d0_partA_heldout_units_pin.json`.

**What it is for.** MAP D0's rule: the instrument must reproduce the published directions on held-out mice before it may
read the engine (Part B). This read decides that.

## 1. Units and seal

**The data:** Allen Brain Observatory Visual Coding 2-photon, under this filter:
- VISp, excitatory Cre lines;
- container not failed;
- sessions A, B and C/C2;
- eye tracking passing in all three sessions.

That gives 53 eligible containers from 48 mice. 4 mice are DEV, and **49 containers from 44 mice are sealed**. No sealed
mouse is a DEV mouse.

**The unit is the mouse,** n = **44**.
- Each mouse contributes exactly one container: its lowest container id.
- The other 5 containers of the multi-container mice are excluded and never read.
- The 44 units, their sessions and dates, and the 5 exclusions are pinned in `tools/d0_partA_heldout_units_pin.json`,
  together with the sha256 of the pin they were derived from.

**Licence:** CC-BY-4.0 (DANDI 000728), which is on `ingest_guard.LICENSE_ALLOW`.

**Storage:** the read writes derived arrays to scratch or `data/drift/` only (gitignored), never the repo or the lake.

## 2. The pipeline, frozen as the DEV reading ran it

**Extraction:** exactly the DEV extraction, per session.
- **Source:** the Allen processed NWB, range-read: `DfOverF`, `cell_specimen_ids`, running speed, and the Natural Movie
  One frame table.
- **Pupil:** `new_pupil_areas` on `synced_frame_timestamps` from the public eye-gaze file.
- **Grid:** 10 repeats × 30 one-second segments.
- **Matching:** only cells present in all three sessions, by `cell_specimen_id`.

**The frozen code** is the three inline blocks of `PARTA-DEV-2026-11-02.md` section 7. Saved verbatim, each with one
trailing newline, they hash to the sha256 values in the units pin under `frozen_pipeline_sha256`:
- the range reader, `cb00d4cb…`;
- the extraction, `d7829a63…`;
- the DEV analysis, `14493239…`.

**The held-out analysis** reuses the DEV analysis's functions unchanged (`residualize`, `rsa`, the instrument calls). It
adds only the per-mouse statistics below. Before the read it is committed as its own inline block, with its own sha256.

## 3. Per-mouse statistics (each computed once per unit)

**Notation:** P = the 3 session pairs (AB, AC, BC). `within` = the mean over the 3 sessions of the odd- vs even-repeat
`pv_correlation`. `between(p)` = the mean of the two opposite-half `pv_correlation`s for pair p.

| id | statistic | definition |
|---|---|---|
| d1 | drift | `within` − mean over p of `between(p)` (raw) |
| d2 | geometry beats single cells | mean over p of [RSA correlation(p) − median `tuning_correlation`(p)] (raw) |
| d3 | elapsed time | the OLS slope of `between(p)` on \|days between the two sessions of p\|, over the 3 pairs (raw). All 44 units have unequal gaps (checked from dates only) |
| d4 | behaviour share | mean over p of [`between(p)` after running + pupil removal − `between(p)` raw] |

**The behavioural model,** frozen as in the DEV reading:
- **Fit:** for each cell, its 300 samples are fitted as a segment effect plus z-scored running speed plus z-scored pupil
  area, by least squares on the samples where both behaviours exist. Only the segment effect is kept.
- **Run-only fallback:** a session with fewer than 150 complete (running + pupil) samples uses running alone. The number
  of such sessions is reported. (150 is half the samples, the minimum for which the segment effect still has 5 samples
  per segment on average: a structural constant, not one tuned on data.)
- **Secondary readout:** d4 with running alone, on all samples.

## 4. Hypotheses, test and decision

**The test.** Each hypothesis is an exact one-sided sign test over the 44 units: the count of units with the predicted
sign, against a binomial(44, 0.5) null. Units with a statistic of exactly 0 are dropped from that test, and the count
dropped is reported.
- Four hypotheses, Bonferroni-corrected: α = 0.05 / 4 = **0.0125** each.
- **That means MET at ≥ 30 of 44 units.** The critical count is recomputed exactly if units are dropped.

| H | prediction | source | DEV (4 mice, descriptive) |
|---|---|---|---|
| H1 | d1 > 0: drift exists | Deitch et al. 2021; Marks & Goard 2021 | 4/4, mean 0.234, SD 0.178 |
| H2 | d2 > 0: geometry is preserved better than single cells | Deitch et al. 2021 | 4/4, mean 0.183, SD 0.133 |
| H3 | d3 < 0: similarity falls with days apart | Deitch et al. 2021; Marks & Goard 2021 | one mouse spans enough days: 0.447 → 0.296 → 0.185 |
| H4 | d4 > 0: removing behaviour reduces apparent drift | Sadeh & Clopath 2022 | 2/4, mean ≈ 0.000, SD 0.004: **DEV predicts NOT MET** |

**Reported beside each test, whatever the outcome:**
- the count;
- the exact one-sided p;
- the median and mean of the statistic with a bootstrap 95% interval (10,000 resamples of units, seed 0);
- for H4, the share mean(d4) / mean(d1).

## 5. Reds and controls, all run BEFORE the sealed read, on DEV or synthetic data only

| red | what it shows | MET when |
|---|---|---|
| R-A | the instrument's six constructed reds | `python -m snn_data_lake.instruments.representational_drift` exits 0 |
| R-B, A/A through the whole pipeline | a session paired with itself reads no drift | each DEV session, used as all three sessions of a pseudo-mouse, reads d1 = 0 exactly |
| R-C, planted behavioural drift | the behaviour control can remove drift that IS behaviour | synthetic cells with a fixed movie tuning plus a gain on a running trace whose distribution differs across the three synthetic sessions: raw d1 > 0, and after removal `between` matches `within` to ≥ 0.999 (noise-free, so this is a numerical tolerance) |
| R-D, shuffled geometry | the RSA reading is not structural | permuting segment order in one session of each DEV mouse drops that mouse's RSA correlation below its unshuffled value (4 of 4) |
| R-E, the seal | nothing sealed was read before the freeze | a scan of scratch, `data/drift/` and the repo finds no derived file for any sealed container id |

**If any red does not reach MET, the read does not happen.** The red is repaired under a dated amendment first.

## 6. Sample size (P1)

**Design:** a one-shot fixed count (P1 item 3). The sealed set is finite, and it is read once and in full. There is no
sequential look: stopping early would gain nothing, and adding mice later is impossible by construction.

**The claimed effect:** at least **75%** of units show the predicted sign (p+ ≥ 0.75). DEV implies more for H1 and H2:
Φ(mean/SD) = 0.906 and 0.915, and 4 of 4 mice. 0.75 is the conservative claim.

**Power,** exact binomial at α 0.0125:
- n = 44 rejects at ≥ 30 of 44;
- power is **0.886** at p+ = 0.75, 0.672 at 0.70, and 0.394 at 0.65;
- the fixed-sample target for power .8 at 0.75 is 37, so 44 meets it.

**What a miss means:**
- A NOT MET reading is a null only for effects of p+ ≥ 0.75.
- If the true p+ lies between 0.5 and 0.75, the reading is **INCONCLUSIVE**, with the mde stated (P8 item 4).
- H4 has no positive DEV estimate. It is powered for the same p+ ≥ 0.75 claim, and any weaker behavioural share reads
  INCONCLUSIVE, not "behaviour plays no role".

**Statistic route** (MAP owner decision 4): `paired_contrast` refuses n > 40 (`MITM_MAX_N`, never approximated). The
recommended route is the exact sign test above, as one small function in the instrument with its own constructed red:
binomial(n, 0.5) tail p matches a brute-force enumeration for n ≤ 20.
- **The alternative** is the first 40 units by container id through `paired_contrast`. That is a paired t with an exact
  sign-flip FPR, not a sign test, so its power comes from the DEV SDs and is computed at the freeze. For reference, a sign
  test on those 40 rejects at ≥ 28 of 40 with power 0.821 at p+ 0.75. The owner chooses (section 9).

## 7. Dispositions (P8, including the null-scale amendment)

| outcome | disposition |
|---|---|
| H1 and H2 MET | the instrument reproduces the published directions on held-out mice. Part B (engine readings, MAP D2/D3 and G5/G9) may cite it. |
| H1 or H2 NOT MET (powered) | **INSTRUMENT BLIND for Part A** (MAP D0: repair or delete the instrument), never "mice do not drift". Part B stays blocked. |
| H1 or H2 INCONCLUSIVE | Part B stays blocked; the mde is reported. |
| H3 NOT MET or INCONCLUSIVE | reported with its mde. It does not block Part B, which needs H1 and H2. |
| H4 NOT MET | "Sadeh & Clopath's behavioural share is not reproduced by a linear 1-second running + pupil model on Allen V1 Natural Movie One". It is about the method and these data, not a refutation of the paper. |
| H4 MET | the behavioural share is reported as measured, and every Part B reading must carry the engine analogue of the behaviour control. |

- **Nothing in the engine is touched by any outcome.** The instrument is pure measurement, and no mechanism is on trial.
  So the null-scale gate's re-test axes (`units` or `ticks`) do not apply to a Part A outcome (MAP owner decision 2: this
  file proposes ruling Part A outcomes as instrument validation, not mechanism nulls).

## 8. Order and what this does not claim

**Order:**
1. Freeze.
2. Commit the held-out analysis block with its sha256.
3. Run R-A to R-E on DEV or synthetic data and record them.
4. Read the 44 sealed units once.
5. Record every statistic and every red in a results file.
6. Panel.

There is no re-run, no change of rule after the read, and no exclusion beyond those pre-registered (section 3's
fallback, and a session whose NWB or eye file fails to load, which is reported by id and counted against the test's n).

**What it does not claim:**
- nothing about the engine;
- nothing about gratings versus movies (that needs the Marks & Goard deposit, routed to a local session);
- nothing about drift beyond 22 days, the longest span in the sealed set.

## 9. Owner decisions before the freeze

1. **The statistic route:** the exact sign test (recommended), or the first 40 units through `paired_contrast`.
2. **The claimed effect** p+ ≥ 0.75 (the powered claim), or a different one.
3. **Rule Part A outcomes as instrument validation** (section 7), so that no null-scale axis is needed: MAP decision 2.
4. **Where the read runs:** this cloud session (now allowed to reach `api.brain-map.org` and the Allen bucket), or the
   owner host.

## 10. Owner decisions, 2026-11-15 (final project; owner in chat: "Yes, all four")

1. **Statistic route:** the exact one-sided sign test over all 44 units (section 4). MET at >= 30 of 44.
2. **Claimed effect:** p+ >= 0.75 (power 0.886 at alpha 0.0125).
3. **Part A outcomes are instrument validation** (section 7): no null-scale axis applies.
4. **Where the read runs:** the owner host (this PC), through `tools/d0_allen/fetch.py`, which runs the frozen
   extraction unchanged and refuses the sealed split until `tools/d0_allen/FROZEN` exists.

The three frozen blocks are saved as `tools/d0_allen/{rangefile,extract,analyze_dev}.py`; their sha256 equal
`frozen_pipeline_sha256` in the units pin (checked 2026-11-15, and re-checked by `fetch.py` on every run).

## 11. Freeze, 2026-11-15

- Held-out analysis: `tools/d0_allen/analyze_heldout.py`, sha256 `5a7a81f7949ee363b1f4ee4b156293a5f017293f3e7594aad5d9173a44d9a2b1`, at commit `f0802d0a27e4376088b92b4b216df22041aeb09a`.
- Pre-read checks (section 5), run at that commit on DEV/synthetic data only, all MET: R-A (instrument self-check,
  exit 0), R-SIGN (exact tail == enumeration, n 1..20), R-B (12 self-paired pseudo-units, max |d1| = 0), R-C
  (planted behavioural drift: raw d1 0.626, residualized between/within >= 0.999), R-D (RSA drops 4/4:
  0.328->0.130, 0.645->-0.016, 0.700->0.094, 0.696->0.034), R-E (0 sealed-derived files).
- R-C as written in section 5 (a running trace whose distribution differs across sessions) cannot be removed by the
  frozen model when the difference is segment-locked (it is collinear with the segment design) or a per-session
  mean (absorbed as a per-cell offset). The constructed case therefore uses segment x repeat running, centred per
  session. Disclosed here, before the read.
- DEV note: the DEV reading predates the section 3 run-only fallback; with it, DEV d4 is 3/4 positive (mean 0.0013)
  instead of 2/4. Descriptive only.
- The sealed split is read ONCE from here (`tools/d0_allen/FROZEN` now exists).

## 12. Results, 2026-11-15 (sealed read, once, at the frozen analysis c8b203cf; 44/44 units scored, 0 missing)

| H | k / n (crit 30) | exact p | mean [bootstrap 95%] | median | verdict |
|---|---|---|---|---|---|
| H1 d1 > 0 | 44/44 | 5.7e-14 | +0.217 [+0.192, +0.242] | +0.217 | MET |
| H2 d2 > 0 | 17/44 | 0.95 | -0.028 [-0.073, +0.017] | -0.044 | NOT MET (observed share 0.39) |
| H3 d3 < 0 | 31/44 | 0.0048 | -0.0069 [-0.0190, +0.0040] | -0.0080 | MET (sign test); mean CI includes 0 |
| H4 d4 > 0 | 13/44 | 0.998 | -0.0046 [-0.0073, -0.0019] | -0.0045 | NOT MET; share d4/d1 = -0.021 |

- Matched cells per unit: median 59.5, range 6-224. Run-only fallback sessions: 13 of 132. Secondary d4 (running only):
  mean -0.0024, 16/44 positive.
- EXPLORATORY (not pre-registered, decides nothing): mean d2 is -0.053 in the 30 units with < 100 matched cells and
  +0.025 in the 14 with >= 100, consistent with segment-level RSA being noisy at low cell counts.
- Per-unit statistics: research/representational-drift/allen_heldout_result.json.
