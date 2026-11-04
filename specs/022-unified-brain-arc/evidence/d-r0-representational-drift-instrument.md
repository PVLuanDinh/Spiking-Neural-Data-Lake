# D-R0 — the representational-drift instrument, validated on held-out Marks & Goard mice

**PRE-REGISTRATION, registered 2026-11-03 (owner: "Yes, register it"), session db0ca840. Frozen text = review round 6.**
Round 1: lenses A AGREE-WITH-FIXES, B AGREE-WITH-FIXES, C CHALLENGE (2 BLOCK). Rounds 2-3: all AGREE-WITH-FIXES, no
BLOCK. Round 4: A AGREE; B AGREE-WITH-FIXES (1 MAJOR, settled by simulation, §2.2); C AGREE-WITH-FIXES (2 MINOR).
Round 5: B AGREE, C AGREE (NITs folded: SE wording from the bar, MOV path not simulated, A/A bar 0.01, dirty-tree
definition, simulation sha256). Panel CLOSED. Enforcer #7830 on round 5: all #7824 items CLOSED; 2 new MINOR (dev-log
dirty-tree deadlock, per-bar SE trigger) folded in round 6 (this text).
Findings and dispositions: §5 (round 1), §6 (round 2, plus enforcer #7824), §7 (rounds 3-4).
Round 6 (enforcer #7830 MINORs) was reviewed: enforcer AGREE #7832 on board plan #7823. Panel lenses A, B, C AGREE (round 5).
Committed ALONE, before any red, instrument code or response-data byte. Later changes are dated append-only amendments.
Target path when registered: `specs/022-unified-brain-arc/evidence/d-r0-representational-drift-instrument.md`,
committed ALONE before any red, any instrument code, and any response-data byte (MAP.md §5.2).
Source design: `research/representational-drift/MAP.md` §4 D0 (branch `research/representational-drift-map` @81c1c331).
Owner rulings applied: `lanes/drift-map-s6-rulings-1103.md` plus, this round, owner 2026-11-03 "All eligible" (§2.3).

## 0. Records
- **No NULL-RECORD is written by this file's outcomes.** P8 amendment item 4: an underpowered result is INCONCLUSIVE,
  not a null. Item 2's 10x re-test is infeasible on this deposit (§2.6), and the owner ruled that case INCONCLUSIVE
  (6.2a). INSTRUMENT BLIND reaches only "repair or delete" through `author_agreement_verdict` or an opposite-direction
  crossing (§2.4), neither of which is a null; if a BLIND disposition ever deletes code, its deletion argument is
  registered at that time in a form the owner approves (a failed positive control is not a STRUCTURAL null under P8
  item 4). A repair after BLIND is a NEW pre-registration, never a further revision under this one. (Panels A4, A-N4,
  C2, C-N4.)
- Verdict words per conjunct: MET / NOT MET / NOT READ. Overall: REPRODUCED / INSTRUMENT BLIND / INCONCLUSIVE.

## 0a. Ambition pass (raising-ambition, self-apply; owner 2026-10-25)
- Axis **Depth**. Rung picked: **2 — reproduce a published drift finding from raw response matrices on a sealed held-out
  split**, the authors' derived stability values used only as a cross-check of the computation.
- Rung above (3: predict unit-level drift on unseen mice) ruled out by MAP §5.2 (Part B frozen only after Part A) and P8
  (a new claim owes its own pre-registration).
- Outside judge: the published direction (Marks & Goard 2021) and the authors' `StabilityData`, scored by the frozen
  pipeline on held-out mice. **Disclosed input:** population selection uses the authors' derived responsiveness flags
  and quality ratings (§1.5); these were computed by the authors from all mice, held-out included. (Panel B5.)

## 1. Territory (measured 2026-11-03)
1.1 **No drift instrument exists.** Grep of `src/` for `loadmat|orthogonal_procrustes|pv_corr|population_vector`: zero
    definitions (known positive `def license_ok` found by the same tool); `graph-exists.js` UNKNOWN on three phrasings.
    `population_vector` was retired 2026-07-24 (b09605f2) as a Georgopoulos direction decoder — a different job.
    Distinct-from: `prior_drift_harness.DriftRecord`, `infra/drift_monitor.psi`.
1.2 **Data:** `data/drift/marks-goard-2021/` (gitignored), 155 `.mat` + README; MATLAB 5.0 -> `scipy.io.loadmat`
    (owner 6.6). Dryad doi:10.25349/D9M606, licence CC0-1.0 (Dryad API, checked 2026-11-03). Paper: Marks & Goard, Nat
    Commun 12 (2021), doi:10.1038/s41467-021-25436-3; one Author Correction (figure number only). Authors' code:
    github.com/ucsb-goard-lab/Representational-Drift-Project (read by panel lens B, §1.4).
1.3 **Main experiments:** 12 mice, 13 fields (Mouse 2 = Windows 1, 2); 5-7 weekly sessions; Mouse 7 = weeks 1,2,3,5,6,7,
    Mouse 8 = weeks 1,2,4,5,6 (README).
1.4 **The authors' computation (from their code, panel lens B):**
    - Population: `dualresp = PDG_Responsive_thresh & NatMov_Responsive_thresh & quality >= 3`
      (`Main figures/nestedstats_dataprep.m`). Presence is per session (`presence(dualresp, 2:end)`); each session's
      values average the neurons present that session (`MixedEffectsModels_RDI.m`, `StabilityAnalyzer.extractRDI`).
    - CC method 'Random' (`StabilityProcessor.m`): per iteration, trials split into random halves with the same indices in
      both sessions; ws = corr(half1, half2) within a session; bs = corr(s1 half1, s2 half2) and corr(s1 half2, s2 half1);
      each pair averaged, rectified with max(., 0) per iteration; 50 iterations for 8 trials (100 if more). After
      averaging, CC_ws <= 0 is set to 0.001. `RDI = (CC_ws - CC_bs) / (CC_ws + CC_bs)`, session 1 the reference.
    - PDG input `RespMat_Full`, no subsampling (one trial = the 12-orientation sweep with gray, 720 frames at 10 Hz).
      MOV input `RespMat_onTime` (300 frames), subsampled to 8 of 30 trials, 10 times, averaged
      (`Stability_masterscript.m`, `NatMov_Processor`).
    - Control RDI: mean of 10 random half-splits of session 1 (`RDI_control` in the deposit); for MOV, the mean over the
      10 trial subsamples of that mean.
    - Responsiveness: 1000x circular-shift null of session-1 reliability plus a t-test over sessions
      (`ChronicImaging_Mouse.evaluateROIs`); code PERCENTILE = 99.9 vs "99th" in the paper (disclosed mismatch, not ours
      to resolve; the deposit flags are used as they are).
    - Statistics: per session, `RDI ~ 1 + StimID + (StimID|MouseID)`, REML.
1.5 **Paper's result (HTML article):** MOV RDI greater than PDG on all sessions (p < 0.001 sessions 2-6, p < 0.01 session
    7), n = 824...698 neurons from 13...9 fields; the difference was small or not significant in "mice 5, 11, 12"
    (Supp. Fig. 5a). Whether the paper's mouse numbers equal the Dryad folder numbers is ASSUMED, not shown. Effect
    magnitude is not stated in the text. The paper: dual-responsive = "20% of total".
1.6 **Eligible counts (RoiINFO only: flags, quality, presence; no response data; measured 2026-11-03):**
    dual & quality >= 3 & present in session 1, per field:
    DEV: M2 W1 83, M2 W2 81, M3 123, M10 84, M11 65. HELD-OUT: M1 37, M4 20, M5 116, M6 49, M7 39, M8 17, M9 45, M12 98.
    The session-1 requirement removes 0 neurons in every field (the same counts result from the authors' rule, present in
    any later session), so the population equals the authors' exactly (panel B-N1).
    Present per session, held-out (dual & quality >= 3), session rows in order:
    M1 [37,32,36,37,33]; M4 [20,20,18,19,18]; M5 [116,115,115,114,114,112,113]; M6 [49,49,47,48,45,48,48];
    M7 [39,39,39,39,38,39] (weeks 1,2,3,5,6,7); M8 [17,17,17,16,15] (weeks 1,2,4,5,6); M9 [45,45,43,43,43,42];
    M12 [98,92,96,92,94,85,85]. Every held-out mouse has neurons present in each of its weeks 2-5, so all 8 are scorable
    (panel A-A3).
1.7 **Duplicates, hash-confirmed:** Prism M1 W1/W2, M2, M3, M4 = Main M2 W1/W2, M1, M12, M11. Scrambled M2 vs Main M10:
    same counts, different bytes, unresolved. Part A uses Main only.
1.8 **Disclosure — read before this draft:** every RoiINFO.mat (fields above), three file headers, the paper HTML, and
    (lens B) the authors' code. No RespData, StabilityData or OrientationData byte has been read. The author knows the
    paper's headline; the split was drawn before §1.5 was read and is not redrawn.
1.9 **Licence excludes Deitch et al.** (Allen ToU not allowlisted; owner 6.3); geometry is NOT READ in D-R0.

## 2. Pre-registration

### 2.1 Deliverables (built only after this file is committed)
Only what Part A itself consumes is built (P3 per-method, honesty guard: a synthetic self-check is a caller, not a
charge; enforcer #7824 MAJOR-1, panel C8):
- `src/snn_data_lake/instruments/representational_drift.py`:
  - `stability_index` — the authors' computation exactly as §1.4 (consumer: Part A E1). Correlations use
    `numpy.corrcoef`; distinct-from `faculties/grid_place_map._pearson` (a faculty's private helper for place-field
    population vectors, not importable as an instrument without making a live faculty a dependency) (enforcer MINOR-4).
  - `DriftReading` result type.
  - NOT built here: `readout_null_split` and `decoder_across_days` move to D1's own pre-registration with their reds
    (null_split, capacity_guard) and their P2 disposition against `instruments/nef_compute.solve_decoders`,
    `growth/ng_snn` ridge solve and the hjepa ridge readouts; `pv_correlation`, `tuning_correlation`,
    `procrustes_realign` wait for D2/D5.
- `src/snn_data_lake/sensorimotor/ophys_ingest.py` — ONLY the `loadmat` reading is new. Pin, licence and byte refusal
  reuse the existing machinery (`load_pin`, `check_pin_against_freeze`, `gate_license`, `stream_sha256`, `gate_bytes`,
  `IngestRefused` in `organoid_ingest` / `h01_ingest`); if importing them from a live module trips `dormancy_gate`
  staleness, they are LIFTED to one shared home in their own commit first (P2; panel C7). Pin: `tools/marks_goard_pin.json`
  (sha256 of all 155 files + README). Never writes the lake (P4b); its self-check asserts no path under the lake home is
  opened for write, with a constructed red (a write attempt must raise).
- `tools/d_r0_partA.py` -> `tools/d_r0_partA.json`.
- Registration in one commit: `REGIONS`, `run_selfchecks.CORE` (synthetic self-checks < 60 s, no data dependency),
  `dormancy_gate.EXEMPT` rows for BOTH modules, each naming its consumers (run_selfchecks, `tools/d_r0_partA.py`) and
  a retire condition: INSTRUMENT BLIND -> repair (new pre-registration) or delete; no D-R* study consuming
  `stability_index` registered within 3 pushed arcs after the verdict -> owner asked (panel C15). Tests, ASCII,
  `random.Random(seed)` per instance, < 800 lines per file. No `GRADIENT_CLAIM` entry: nothing here learns or solves.
- NOT here: the sequential-design P2 lift out of `rpe_charge` (owner 6.5) lands with its first sequential consumer (D1).

### 2.2 Reds — constructed, run RED and recorded before any mouse reading (`*_verdict`, red_reachability scan)
**Synthetic generator, fixed now** (panels A-N2, C-C3): 200 neurons, 8 trials per session, 300 frames, seed 20261103.
Each neuron's mean response = a smooth random time course (Gaussian process, length scale 10 frames, unit variance);
each trial = mean + i.i.d. Gaussian noise of SD sigma = 1.0 (per-frame SNR 1). Planted drift of dose d: session-2 mean
= sqrt(1-d^2) * pattern1 + d * pattern2 (pattern2 independent, same law). **Reference expectation** for any dose: the
same estimator on the same construction with 2,000 neurons and seed 20261104, computed once and pinned in
`tools/d_r0_synthetic_ref.json` before any red runs.
| Criterion | Passes when | Constructed RED (must fail the pass condition) |
|---|---|---|
| `aa_stability_verdict` | two sessions drawn independently from ONE generative model (same mean response, independent noise, equal trial counts) read mean RDI within +-0.01 of 0 (panel B-N2: compared with 0, not with the 4-vs-4 control; tightened from 0.05 in round 5 because the simulated SE is 0.0005, so a +0.04 bias no longer passes — lens C round 5) | planted drift d = 0.55 (30% of variance moved) must read mean RDI > 0.01 (simulated: about 0.10) |
| `planted_drift_dose_verdict` | doses d in {0.4, 0.55, 0.7, 0.85, 1.0} read RDI monotone increasing, each within max(10% of the pinned reference, 3 x the reference's 200-neuron SE) (round-3 A: at d = 0.2 the 10% band was below sampling noise) | an estimator that skips the between-session pairing (uses ws for bs) must read a flat dose curve and fail |
| `mov_vs_mov_control_verdict` (P8 item 2 control: a known answer through the WHOLE pipeline; replaces the round-2 stimulus swap, which was symmetric by construction — panels A-N1, C-N2) | on each DEV mouse, MOV trials split INTERLEAVED (odd vs even, so within-session adaptation is not in the null; round-3 B/C). Arm A = session-1 odd trials vs week-w odd trials; arm B = session-1 even vs week-w even; each arm through the identical MOV path (8 of 15 x 10 subsamples, the same 8 indices in both sessions of an arm as in `StabilityProcessor.m`, 50 splits, rectify, floor). D_null = arm A minus arm B, per mouse, weeks 2-5. Pass: abs(D_null) <= max(0.05, 3 x neuron-bootstrap SE of D_null), bootstrap 1,000 resamples, seed 20261105, fixed now. The contiguous split (1-15 vs 16-30) is reported, not decided on | on synthetic data, a planted drift in arm A only (d = 0.6, arm B d = 0) must give per-mouse D_null above the same bar (same per-mouse scale as the test) |
| `author_agreement_verdict` (positive control) | see §2.4: DEV and HELD-OUT conjuncts, relative and absolute bars | session labels shuffled within neuron must drop agreement below the bar (ceil computed on UNSHUFFLED data) |
The held-out exact sign-flip false-positive rate (`permutation_fpr`) is printed beside E1 as the test's own null.
Any red that cannot be made to fire deletes its criterion (P8 item 1, F11); if `author_agreement_verdict` or
`mov_vs_mov_control_verdict` is deleted, REPRODUCED becomes unreachable and the outcome is INCONCLUSIVE.
**Every red is re-run and MET at the frozen sha (§2.3) before any HELD-OUT byte** (panel C-N3). A synthetic criterion
that is NOT MET (its red fires but the build fails the pass condition) stops the run: repair, dated amendment, no mouse
reading until it is MET.
**`mov_vs_mov_control_verdict` NOT MET on DEV data** (round-3 A, C): the pipeline may be revised within the §2.4 cap
(each revision a logged run at a new code sha); still NOT MET when the cap is reached -> INSTRUMENT BLIND, HELD-OUT never
read.
**What the controls do not test (disclosed, round-3 B, C):** the reproduction inherits the authors' method, including its
PDG/MOV asymmetry (PDG 720 frames with gray periods and all 8 trials; MOV 300 frames, 8 of 30 subsampled) and RDI's
dependence on reliability (lower CC_ws, higher RDI). The MOV-vs-MOV control cannot see either. A reliability-matched
sensitivity (D within bins of session-1 CC_ws, PDG and MOV) is reported, not decided on. The dose reference is made by the
same estimator, so the dose criterion tests precision and monotonicity; correctness rests on `author_agreement_verdict`.
If any DEV revision changes the estimator, the synthetic reference is re-pinned at the frozen sha.

**Why these bars (P1).** Pre-registered acceptance bars on synthetic, noise-controlled inputs:
- **Design simulation (synthetic only, 2026-11-03):** a throwaway implementation of §1.4's computation on this
  generator (200 neurons, seed 20261103; `lanes/d-r0-design-sim.py`, not the deliverable; re-run it with the .venv-lake python) gave mean RDI (SE):
  d = 0: +0.0002 (0.0005); 0.40: 0.049 (0.0023); 0.55: 0.102 (0.0047); 0.60: 0.127 (0.0057); 0.70: 0.191 (0.0085);
  0.85: 0.356 (0.0144); 1.0: 0.845 (0.0146). So the 200-neuron SE is far below 0.017; the aa_stability red (d = 0.55, about 0.10)
  sits about 19 SE above its 0.01 bar; the mov_vs_mov synthetic red (d = 0.6, about 0.127) sits about 13 SE above the
  0.05 floor (about 7.6 SE at 65 neurons); adjacent doses are
  separated by about 10 difference-SEs (round-4 B-N8 and A-2 rested on an SE the
  round-4 text mis-implied; corrected here). The formal reference is still pinned from the BUILT estimator. The script (sha256
  ec4e086eeb32aeaf2d29ecab067a643b43e5d04d91ea913a2a4f85a47be6c72c) simulates the 8-trial PDG path only; the MOV path
  (8 of 15 x 10 subsamples) was not simulated, and averaging over subsamples can only lower its SE (lens B round 5).
- +-0.01 RDI (aa_stability, about 20 SE at the simulated A/A SE of 0.0005) and +-0.05 (the mov_vs_mov floor on real
  data): small fractions of RDI's [-1, 1] range, well below the reds. The trigger is PER BAR: if the pinned reference
  shows a 200-neuron SE above (bar / 3) for any bar (aa_stability: SE > 0.0033; mov_vs_mov floor: SE > 0.017), that
  bar becomes 3 x the SE and every red must clear it (enforcer #7830 MINOR-2); written in a dated amendment BEFORE any red runs and before
  any mouse byte; never after.
- On real DEV data the mov_vs_mov bar is max(0.05, 3 x neuron-bootstrap SE), CAPPED: a bar above 0.10 is NOT MET (a field
  too noisy to test is not a pass; round-4 C-1). The realised DEV bar is recorded. The synthetic mov_vs_mov red is run
  at 200 neurons AND at 65 neurons (the smallest DEV field), each against its own bootstrap bar (round-4 A-1, C-1).
- 5 doses / tolerance max(10%, 3 SE): the fewest points that make monotonicity a test; the 3-SE term keeps a correct
  build from failing on sampling noise.
- At most 5 DEV pipeline revisions (§2.4): §1.4 has five separable steps (population, trial split, rectify/floor, MOV
  subsample, session reference); one correction per step.

### 2.3 Part A — the reproduction
- **Claim tested:** MOV responses drift more than PDG responses across weekly sessions (paper's direction, §1.5).
- **Split, sealed (literal):** `random.Random(20261103).sample(list(range(1, 13)), 4)` -> **DEV = Mouse 2, 3, 10, 11;
  HELD-OUT = Mouse 1, 4, 5, 6, 7, 8, 9, 12.** HELD-OUT response bytes are read ONCE, by the frozen pipeline.
- **Population (the paper's, §1.4):** per field, every neuron with `PDG_Responsive_thresh & NatMov_Responsive_thresh &
  quality >= 3` and present in session 1; each later session uses the neurons present in it. **All eligible neurons,
  no subsample — owner 2026-11-03 "All eligible"** (replaces the 40-per-field draw of ruling 6.2/6.4: at 40, only 4
  held-out mice qualified, §1.6). Held-out neurons per mouse: 17-116 (§1.6). Units are recorded as min / median / max.
- **Endpoint E1 (only endpoint):** per mouse, `D = mean over weeks w in {2, 3, 4, 5} present for that mouse of
  [mean over neurons present in week w of (RDI_MOV - RDI_PDG)]`, RDI by §1.4 against session 1, weeks by real index
  (Mouse 7: weeks 2, 3, 5; Mouse 8: weeks 2, 4, 5). Weeks 2-5 because every field has weeks 2 and 5 and RDI grows
  with time (panels A7, B7). Residual, disclosed: Mouse 7 lacks week 4 and Mouse 8 lacks week 3, so their D leans later;
  a sensitivity on the common weeks {2, 5} is reported (panel A-A7). Mouse 2 (DEV only) averages its two windows
  (disclosed deviation: the paper treats each window as a field).
- Mice are weighted equally (standard for a per-mouse test), although their neuron counts run 17-116; each mouse's
  neuron count is reported beside its D (panels A-N6, C-N5).
- **Test:** `paired_contrast.contrast("E1", D_heldout, bar)` — paired one-sided t, ties counted, MET iff t > bar, with
  the exact sign-flip false-positive rate printed beside it. **n = 8 is fixed now** (every held-out mouse has >= 17
  eligible neurons in session 1); bar `T_BAR = 1.895` (7 d.f.). If a held-out mouse cannot be scored (no eligible
  neuron present in any of weeks 2-5), n drops and **n < 8 is INCONCLUSIVE before any reading** (power below .8 at
  dz = 1.0, §2.5; panels A3, C6). Rounding follows the `T_BAR` convention (third decimal place),
  not `T_BAR_S31`'s fourth (enforcer NIT-2).
- **Sensitivities (reported, never decided on):** E1 over every available week; and the paper's single-stimulus sets as
  SEPARATE sets (`movresp` neurons for RDI_MOV, `pdgresp` for RDI_PDG), not a paired difference (panel B9). Control
  RDI (`RDI_control`) is reported beside each, never subtracted.
- **Frozen before HELD-OUT is read** (panel C6): the code commit of `representational_drift.py`, `ophys_ingest.py` and
  the Part A driver (`src/snn_data_lake/instruments/d_r0_part_a.py` since section 11) is named by sha in a dated amendment;
  HELD-OUT is run only from that sha.

### 2.4 Verdict rule (written before data)
- **`author_agreement_verdict`**, per stimulus, unit = one neuron-session (eligible population, weeks 2-5):
  - Ceiling: our estimator run twice with seeds 1 and 2 on DEV; `ceil = Spearman(run1, run2)` and
    `self_mad = median |run1 - run2|` (the authors' RDI is stochastic; panel B4).
  - Our estimator's iterations and subsampling are FIXED at the authors' (50 per pair at 8 trials, MOV 8 of 30 x 10;
    §1.4), so a noisier estimator cannot loosen its own bars (panel C-N1). `ceil` is computed on unshuffled DEV data.
  - DEV pass, ALL of: `ceil >= 0.9`; Spearman(ours, `StabilityData.RDI`) >= max(0.9 x ceil, 0.8); median
    |ours - theirs| <= min(2 x self_mad, 0.05). CC_ws and CC_bs reported against `StabilityData.CC_ws` / `CC_bs` the
    same way. At least 200 neuron-sessions per stimulus. (Absolute floors: panels A-N3, C-N1.)
  - Attempts: `ophys_ingest` itself appends a log entry (code sha, timestamp, files) to append-only
    `tools/d_r0_dev_log.jsonl` whenever it reads any RespData/StabilityData bytes, so no script or REPL bypasses it
    (round-3 C). Each entry carries the commit sha AND a sha256 of the source files actually loaded
    (`representational_drift.py`, `ophys_ingest.py`, the calling tool); `ophys_ingest` refuses to read response bytes
    from a dirty working tree, where dirty = non-empty `git status --porcelain` under `src/` or `tools/`, untracked files
    included, EXCEPT the single path `tools/d_r0_dev_log.jsonl` itself (otherwise the first logged read would dirty the
    tree and refuse every later read; enforcer #7830 MINOR-1). The log is committed in the SAME dated amendment that names the frozen sha, before any
    HELD-OUT byte, so the DEV attempt count is fixed in git before the held-out read (enforcer #7832 NIT-1), and again with
    the closing record; the logged hash covers every `snn_data_lake` module actually loaded (round-4 C-2, round-5 C NIT). A revision = a logged run with a new source hash. At most 5 (reason in §2.2); a 6th is not allowed: NOT MET.
    No revision after the DEV SD pilot has computed D, so DEV's direction cannot steer the pipeline (panel C-N4).
  - HELD-OUT pass (run once, by the frozen sha): the same floors and bars, with `ceil` and `self_mad` RECOMPUTED on
    unshuffled held-out data (round-3 A). (Panels A6, B4, C5.)
- **REPRODUCED:** every surviving red MET; `author_agreement_verdict` MET on DEV and HELD-OUT; `contrast("E1", ...).met` with
  t > bar in the published direction.
- **INSTRUMENT BLIND:** `author_agreement_verdict` NOT MET on DEV (HELD-OUT then never read) or on HELD-OUT; or
  `mov_vs_mov_control_verdict` NOT MET on DEV at the revision cap (HELD-OUT never read); or t < -bar (opposite
  direction). Disposition: repair (new pre-registration) or delete (T063 2.9). Never "mice do not drift".
- **INCONCLUSIVE:** E1 not met with agreement and control MET (a non-crossing; the 10x re-test is infeasible, owner
  6.2a); or n < 8 (power below .8 for the claimed effect, §2.5); or a control criterion deleted (§2.2). Licenses nothing;
  no NULL-RECORD (§0).

### 2.5 Sample size (P1)
- **Claimed effect, fixed now (not from DEV):** a standardized per-mouse effect **dz = 1.0** — the smallest effect this
  fixed deposit can test: at n = 8, one-sided .05, power is .815 at dz = 1.0 (dz for .8 = 0.978; scipy noncentral t,
  computed 2026-11-03). The claim this design tests is therefore "a large, consistent per-mouse direction"; a smaller
  true effect reads INCONCLUSIVE, by design and stated in advance. (Panels A1, C6.)
- **Fixed-sample target (P1 item 1):** n = 8 held-out mice at power .8 for dz = 1.0, one-sided .05, single endpoint
  (power .815; the whole held-out set). Added at registration after `sample_size_gate` R-FIELD found the word missing
  (2026-11-03); the number was already fixed in this section.
- **Measured SD:** DEV pilot, after DEV agreement is MET, using one window per mouse (Mouse 2 Window 1 only, so the
  window average does not shrink the SD; panel A2): the paired SD of D over the 4 DEV mice, recorded in a dated amendment
  before any HELD-OUT byte, with its 80% interval (x0.69 to x2.27 for 3 d.f.) and the raw-unit mde (1.0 x SD) at both
  ends.
- **One-shot (P1 item 3):** a fixed retrospective deposit; no mice can be added; looks and a cap do not apply. Single
  endpoint, no multiplicity correction. Item 3's "a result at an underpowered count is INCONCLUSIVE" is met, not
  argued around: power for the CLAIMED effect (dz = 1.0) is .815 >= .8 at n = 8, so the count is not underpowered for the
  claim; if n falls below 8, power is recomputed for the new n and, if it is below .8, REPRODUCED is unreachable and
  the outcome is INCONCLUSIVE (enforcer MINOR-1). At dz = 1.0 power is below .8 for any n < 8 (about .76 at n = 7), so
  in practice the rule is n < 8 -> INCONCLUSIVE; all 8 held-out mice are scorable (§1.6).
- **Disclosed:** the effect is fixed by design (dz = 1.0); only the SD comes from the 4 DEV mice. No Source Data
  magnitude is used (enforcer MINOR-3).
- **P8 item 3 (attribution) is N/A:** D-R0 invents no method with a gain to attribute; it reimplements the authors'
  published computation, and per-step correctness is tested by the agreement conjunct and the 5-step revision log
  (enforcer NIT-4).

### 2.6 Null scale (owner 6.2 and 6.2a)
The owner's path was 40 units -> 400-unit re-test -> INSTRUMENT BLIND, and INCONCLUSIVE when the re-test is infeasible.
Measured (§1.6): the largest held-out field has 116 eligible neurons; no field approaches 400, and under "All eligible"
(§2.3) there is no larger scale left to re-test at. **The re-test is infeasible with certainty**, so a non-crossing E1
reads INCONCLUSIVE, and the null route to INSTRUMENT BLIND is structurally unreachable for Part A (panels A5, C2).
Units also scale only the within-mouse noise, not the between-mouse variance the test runs on (panel A5).

### 2.7 Licensed and refused sentences
- Licensed (only on REPRODUCED): "On 8 held-out Marks & Goard mice, D-R0's `stability_index` reproduces the published
  direction — natural-movie responses drift more than grating responses — and agrees with the authors' own stability
  values." It validates `stability_index` only (panel C9), and it does not exclude a reliability or stimulus-structure
  account of the difference (§2.2, round-3 B).
- Refused always: "the engine drifts like mouse cortex"; "mice do not drift"; "geometry is preserved" (Deitch not read);
  any sentence about the SNDL engine.

### 2.8 What D-R0 unblocks
D1 (L1b's owed control) builds `readout_null_split` and `decoder_across_days` under its own pre-registration, with
their reds, consumers and P2 dispositions (enforcer #7824 MAJOR-1). D-R0 contributes the validated `stability_index`
and the ingest path only.

## 3. Results
(empty until the reds run)

## 4. Open items before freeze
- Panel re-review of this fold (lenses A, B, C) and the `policy-enforcer` AGREE on board plan #7823.
- Confirm `organoid_ingest` / `h01_ingest` symbol names at the registration commit (§2.1) (verified by the enforcer,
  #7830).
- The integrator runs `gates/sample_size_gate.py --assert` on this file at the registration commit (enforcer #7830
  NIT-1: not run in review), and `gates/null_scale_gate.py --assert` (enforcer #7832).
- Enforcer AGREE on round 6: board #7832 (ledger line in lanes/policy-enforcer-row/LEDGER.md).

## 7. Round-3 panel findings and their disposition
A: dose tolerance below noise -> doses 0.4-1.0, tolerance max(10%, 3 SE) (§2.2); MOV-vs-MOV NOT MET on DEV -> revise
within cap, else BLIND (§2.2); real-data bar -> max(0.05, 3 x bootstrap SE) (§2.2); held-out ceil/self_mad recomputed
(§2.4); "several times larger" corrected; n < 8 note (§2.5). B: half pairing stated, odd/even interleaved, contiguous
reported (§2.2); reliability and PDG/MOV asymmetry disclosed, reliability-matched sensitivity, §2.7 narrowed; 8-of-15
noise covered by the bootstrap bar. C: re-derive rule = 3 x reference SE (§2.2); attempt log written by `ophys_ingest`
on every response read (§2.4); MOV-vs-MOV NOT MET routed (§2.2); odd/even split; asymmetry disclosed; dose reference is
precision only, re-pinned at frozen sha (§2.2); "every surviving red" (§2.4).
**Round 4:** A AGREE (minors: DEV bar sensitivity -> synthetic red also at 65 neurons, realised bar recorded; adjacent
dose reversal -> simulation shows ~10 difference-SEs; wording -> 10% line replaced, mov_vs_mov route in §2.4, n < 8
aligned). B N8 MAJOR (reds under own bar) -> settled by the design simulation: SE200 at d = 0.55 is 0.0047, reds 20+ SE
above the bar (§2.2); NIT same 8 indices -> stated. C-1 self-loosening DEV bar -> capped at 0.10 (§2.2); C-2 dirty tree
-> source hash logged, dirty tree refused (§2.4).

## 6. Round-2 panel findings and their disposition
A-A3 residual -> per-week present counts stated, all 8 scorable (§1.6). A-A7 residual -> disclosed, sensitivity {2, 5}
(§2.3). A-N1 / C-N2 swap control symmetric by construction -> replaced by MOV-vs-MOV negative control with a same-scale
red (§2.2). A-N2 / C-C3 bars unverifiable -> generator, sigma, reference pinned (§2.2). A-N3 / C-N1 self-loosening
agreement bars -> fixed iterations + absolute floors (§2.4). A-N4 / C-N4 BLIND deletion form, repair = new pre-reg
(§0). A-N5 cap reason -> §2.2 bars. A-N6 / C-N5 equal weighting -> disclosed, counts reported (§2.3). B-N1 session-1 rule
-> removes 0 neurons, population equals the authors' (§1.6). B-N2 A/A vs 4-vs-4 control -> compared with 0 (§2.2).
B-N3 "20% of total" quoted (§1.5). B-N4 MOV control averaging (§1.4). C-N3 reds re-run at frozen sha; synthetic NOT
MET stops the run (§2.2). C-N4 attempts auto-logged; no revision after the SD pilot (§2.4).
**Enforcer #7824 (on round 1):** MAJOR-1 P3 -> D1 functions and their reds moved to D1's pre-registration (§2.1, §2.8);
MAJOR-2 PENDING record -> §0 (no NULL-RECORD); MINOR-1 P1 item 3 -> power .815 for the claimed effect, recomputed if n
drops (§2.5); MINOR-2 40-draw seed -> moot ("All eligible"); MINOR-3 effect source -> fixed dz, SD from DEV, disclosed
(§2.5); MINOR-4 P2 hits -> `_pearson` distinct-from; ridge hits move with the decoder to D1 (§2.1); NIT-1, NIT-3 moot
(no records); NIT-2 rounding named (§2.3); NIT-4 attribution N/A argued (§2.5).

## 5. Round-1 panel findings and their disposition
A1 effect from DEV circular -> fixed dz = 1.0 (§2.5). A2 window average shrinks SD -> M2 W1 only (§2.5). A3 n knowable
now -> counted, n = 8 fixed, n < 5 rule (§1.6, §2.3). A4 PENDING record trap -> no NULL-RECORD (§0). A5 infeasibility
certain -> stated (§2.6). A6 agreement unit -> defined (§2.4). A7 session-count scale -> weeks 2-5 (§2.3). A8 RDI blow-up
-> authors' floor and rectify make RDI bounded (§1.4). A9 verdict terms -> `.met`, t < -bar (§2.4). A10 shuffle
under-specified -> replaced (§2.2). A11 seeds=1 -> moot (no NULL-RECORD). A12 "stricter" -> removed.
B1 quality >= 3 -> added (§2.3). B2 per-session presence -> adopted (§2.3). B3 exact computation -> pinned (§1.4).
B4 stochastic ceiling -> relative bars (§2.4). B5 flags are author-derived -> disclosed (§0a). B6 attainability ->
counted; owner "All eligible" (§2.3). B7 week scale -> weeks 2-5. B8 control RDI -> deposit value, reported. B9
sensitivity sets separate -> adopted. B10 mouse numbering -> stated as assumed (§1.5). B11 Mouse 2 -> disclosed. B12
real-week indexing -> used for weeks only. B13 model -> transcribed exactly (§1.4).
C1 null_split red unreachable -> N = 40, r = 20 (§2.2). C2 PENDING trap -> §0. C3 A/A copy reads negative -> independent
draws vs control (§2.2). C4 shuffle near-automatic pass -> stimulus-swap control with fixed planted dz (§2.2). C5
circular agreement -> held-out conjunct, absolute bar, attempt cap (§2.4). C6 post-data decisions -> n, effect, freeze
sha fixed (§2.3, §2.5). C7 P2 ingest machinery -> reuse or lift (§2.1). C8 consumer-less functions -> not built (§2.1).
C9 overclaim -> licensed sentence narrowed (§2.7). C10 rotation red -> dose red uses a broken estimator (§2.2). C11
procrustes bar, C12 orientation period -> moot (functions not built). C13 training accuracy -> stated (§2.2). C14
opposite direction -> t < -bar (§2.4). C15 EXEMPT rows -> both, consumers, retire conditions, P4b red (§2.1). C16
presence deviation -> adopted the paper's rule instead. C17 backticks -> moot.

## 8. Amendment 2026-11-03 (launcher-d0; enforcer #7886 NIT-2): a crashed HELD-OUT run
The build (claude/int3@b3e9816d) makes "HELD-OUT response bytes are read ONCE" (§2.3) literal: the driver refuses a
HELD-OUT run once the dev log records any HELD-OUT mouse file. A HELD-OUT run that stops after its first logged read
(crash, kill, power loss) therefore cannot be re-run. Such a run is reported AS IT STANDS: the endpoint D is computed
only if every one of the 8 HELD-OUT mice was scored; otherwise the verdict is INCONCLUSIVE (§2.3, n < 8), with the
scored mice and the point of failure disclosed. It is NOT re-run, and no log line is removed to re-enable it, without
an owner ruling in chat recorded here as a dated amendment.

## 9. Amendment 2026-11-03 (launcher-d0; owner 2026-11-03 "Yes, amend"): the MOV trial count is T, read per field
Written before any DEV pipeline run (no RDI, no agreement value, no control value computed on mouse data). The layout dump
(`tools/d_r0_layout.py` at claude/int3@5ca170ce/597b58c7; dev log at 1cdd1619: names, shapes and dtypes only) shows
`RespData.NatMov.RespMat_onTime` with T = 30 trials for DEV Mice 2, 3, 11 and T = 20 for DEV Mouse 10. §1.4, §2.2 and
§2.4 assumed T = 30. The HELD-OUT mice's T is unknown and is not read before their single run. Therefore, everywhere:
- MOV path (§1.4, §2.4 agreement): 8 of T trials, 10 subsamples, the same 8 indices in both sessions -- the authors'
  pick of 8 is unchanged; "8 of 30" reads "8 of T". The PDG path (8 trials, no subsampling) is unchanged.
- `mov_vs_mov_control_verdict` (§2.2): each interleaved arm holds floor(T/2) trials (odd vs even indices) and runs
  8 of floor(T/2) x 10; "8 of 15" reads "8 of floor(T/2)" (Mouse 10: 8 of 10). The reported contiguous split is the
  first floor(T/2) vs the rest. Bar, seed and bootstrap are unchanged.
- Fail closed: a field with T < 8 (MOV path) or floor(T/2) < 8 (control) is refused by the code, never run on fewer
  trials; a HELD-OUT field refused this way counts as unscored (§2.3: n < 8 is INCONCLUSIVE, §8 applies).
- The synthetic halves of §2.2 (generator §2.2, 15 trials per arm) are unchanged: they test the estimator, not the deposit.
- Disclosed: Mouse 10's MOV RDI averages over C(20, 8) picks against C(30, 8) for the others, so its MOV estimate is
  noisier at equal reps; it is a DEV mouse and decides nothing alone.

## 10. Amendment 2026-11-03 (launcher-d0; owner 2026-11-03 "Yes, don't count"; enforcer #7901 MINOR-1): what the 5-revision cap counts
§2.4 says "A revision = a logged run with a new source hash. At most 5." The dev log at claude/int3@1cdd1619 already holds
two calling-tool hashes, both from `tools/d_r0_layout.py` (`__main__` 12e5dca7... at 5ca170ce and 662fe632... at
597b58c7): structure-only reads (names, shapes, dtypes) that computed no RDI, agreement or control value. Ruling:
- Those two runs do NOT count toward the cap. The cap bounds revisions that could be steered by a DEV outcome
  (§2.4, panel C-N4); a run that reads no value cannot be. This is disclosed here, not hidden: 18 DEV structure reads.
- From now on the counter is mechanical: a revision is a distinct `sources` hash set among dev-log entries whose
  `__main__` hash equals the sha256 of `tools/d_r0_partA.py` at that entry's `commit`. Every other calling tool
  counts zero revisions, and it may never compute or print a response value (a tool that does is a pipeline run and
  goes through `tools/d_r0_partA.py`, or the cap is breached).
- Cap unchanged: at most 5 such revisions; the rest of §2.4 (no revision after the SD pilot, NOT MET at the cap) stands.

## 11. Amendment 2026-11-04 (owner 2026-11-04 "Move D-R0 tools"; P4c ratchet): the two D-R0 tools are package modules
`tools/p4c_ratchet.py --assert` read tools 32 > baseline 30: a `tools/` script is never governed (governed = a
REGIONS-registered `src/snn_data_lake/<region>/<module>.py`). Both scripts moved, logic unchanged, each with a
self-check that reds on a constructed input (REGIONS + `run_selfchecks` + `dormancy_gate.EXEMPT`, one commit):
- `tools/d_r0_partA.py` -> `src/snn_data_lake/instruments/d_r0_part_a.py`, run as
  `python -m snn_data_lake.instruments.d_r0_part_a --split dev|heldout`. Its result file stays `tools/d_r0_partA.json`.
- `tools/d_r0_layout.py` -> `src/snn_data_lake/sensorimotor/ophys_layout.py`, run as
  `python -m snn_data_lake.sensorimotor.ophys_layout --mouse N`.
- Every rule above that names `tools/d_r0_partA.py` (2.1 registration, 2.3 freeze, section 10 counter) names the new
  module for every dev-log entry whose `commit` is at or after the move commit; the section 10 counter reads the
  sha256 of the file the path held at that entry's `commit` (old path before the move, new path after). Under
  `python -m` the calling tool is still `__main__` and `ophys_ingest.source_hashes` still hashes that file; nothing in
  the read path or the log format changed. Historical text naming the old paths (sections 9, 10) is unchanged.
  §2.3's freeze bullet (line 192) was re-worded in place, path only; the original text is at 928f117f. Reason: citation_guard STRICT scope covers §2.3 (its heading matches `reproduc`), so the old path would red the gate.
