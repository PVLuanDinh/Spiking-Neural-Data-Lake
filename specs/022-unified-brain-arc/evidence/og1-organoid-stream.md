# OG1 -- organoid succession stream: public organoid recordings as SNDL's first stream with measured succession

**Status: FROZEN 2026-10-23 on the owner's approval in chat, verbatim: "Approve OG1 as written, including the KCL
exclusion. Register it and freeze it on int3, then start Arc 1."** Registered as T118 (tasks.md) and as a dated row in
PLAN-FINAL. **This commit precedes every data read.** Before the freeze, only DANDI's public METADATA was read (asset
paths, sizes, sha256, `variableMeasured`, related DOIs); no recording byte has been read.

Changes from the reviewed draft are confined to the "SETTLED at freeze" items in section 1, the download size, and the
owner line in section 8. Origin: owner, 2026-10-23, "plan a row that ingests the public organoid data (HNOCA, KCL
figshare CC0, DANDI) under the H01 pattern, with its pre-registered test". Section 6 excludes KCL; the owner
approved that exclusion.

## 0. What orientation found (static reading at main 06744945 / int3 2e40dcd0, plus public metadata read 2026-10-23)

1. **The gap is the stream, measured twice.** W2 section 3.1c: the live packs' ordered next-label MI 0.2498 vs
   shuffle mean 0.2579 (122 of 200 shuffles at or above; audio 1.0851 vs 1.1015, 119 of 200): the ordered pack is the
   median shuffle. Phase 2 (`l-parallel-arms.md`): handed exactly what persistence uses, the learner matches
   persistence and no more; "the limit is now the stream's transitions".
2. **E4 is blocked on this exact gap.** `lanes/research-1019/learning-rules/LEARNING-RULES-BRIEF.md` E4 (reservoir
   plasticity vs the FIXED `KernelLsmPool`) is blocked on BOTH a stream with measured succession AND an LIF
   implementation; the candidate stream was the event camera, to be tested first "with W2's ordered next-label MI vs
   shuffle". OG1 is a second candidate, recorded neural tissue, and supplies the stream half only.
3. **A published claim exists for these bytes.** van der Molen et al. 2025, Nat. Neurosci., "Preconfigured neuronal
   firing sequences in human brain organoids", doi:10.1038/s41593-025-02111-0 (DANDI `relatedResource`
   IsDescribedBy). DANDI 001603 version 0.260923.0623, CC-BY-4.0, with spike-sorted data for every subject. The paper already reports sequences in these recordings. **Stage 1 therefore asks whether SNDL's
   encoding can read the published succession, beyond persistence, and at what parameters. It does not ask whether
   succession exists.** Its null is first a claim about the instrument (section 3, the arbiter).
4. **Template = H01.** `growth/h01_ingest.py` + `growth/cortical_prior.py`: outside data is a pinned artifact under
   gitignored `data/`, read by refusal gates that each carry a constructed red, and never written to the lake ("The
   lake is what the brain LEARNED. This is what a microscope saw." Here, an electrode).
5. **Reuse map (P2; each checked 2026-10-23).**
   - `sensorimotor/ingest_guard.license_ok`: deny-by-default allowlist, contains `cc-by-4.0` and `cc0-1.0`.
   - `instruments/attention_schema.mi_bits`: the tree's one MI estimator (plug-in, no smoothing).
   - `infra/_mathutils.q95`.
   - `faculties/open_world_loop._dpmeans`: state discretization. The module is live via `sleep/nightly_skills`'
     lazy import and is not in `dormancy_gate.EXEMPT`, so an instrument importing it is not a landing-day red.
   - `h5py` 3.16.0 is installed in `.venv-lake` (3.14.6 free-threading, GIL disabled) and declared in the `nir`
     extra. `engine/wiring_manifest` imports it directly. No `pynwb`: NWB units are plain HDF5 datasets
     (`units/spike_times`, `units/spike_times_index`).
   - Stage 2: `sensorimotor/afferent_boundary` (the ONE afferent lane) and `instruments/taught_problems` (the ONE
     problem contract).
   - **New code is exactly two pieces:** the NWB units reader (about 30 lines) and the run-null surrogate (about
     20 lines).
6. **Not re-proposed:**
   - The 2026-09-30 reservoir relational probe (re-proposal guard).
   - "A reservoir buys Gf" (refuted, arXiv 1910.03354 note).
   - OG1 claims a stream, never fluid intelligence.

## 1. Data (pinned at freeze)

| Split | Subjects | Source (per DANDI's description) | File used | Verified 2026-10-23 |
|---|---|---|---|---|
| DEV | HO1, HO2, HO3, HO4 | Sharf et al. 2022 Nat. Commun., doi:10.1038/s41467-022-32115-4 (UCSC), MaxWell HD-MEA, sectioned organoids | every spike-sorted NWB of the subject (not `_ecephys`) | Units on all: HO1 1 file 24 MB, HO2 9 files 187.7 MB, HO3 9 files 147.5 MB, HO4 1 file 22.5 MB |
| HELD-OUT | HO5, HO6, HO7, HO8 | Alam El Din et al. 2025 Commun. Biol., doi:10.1038/s42003-025-08632-5 (JHU) | the same | Units on all: HO5 185.7 MB, HO6 30.4 MB, HO7 18.3 MB, HO8 21.3 MB (1 file each) |

- **SETTLED at freeze (2026-10-23, metadata only):**
  - Units presence: all 8 subjects, 24 files, from DANDI `variableMeasured`.
  - The three DOIs, from DANDI `relatedResource`.
  - The HO5-HO8 device is MaxWell **MaxTwo** HD-MEA. This is SECONDHAND: MaxWell's own announcement of Alam El Din et
    al. 2025, whose methods also include theta-burst stimulation and pharmacology. Stage 0 records each file's own
    NWB device entry, and a mismatch is disclosed, never a refusal.
  - Per-file sha256 for all 24 files, cached at freeze for the pin in
    `D:\AI-project\lanes\organoid-og1-1023\og1_dandi_assets_freeze.json` (sha256 prefix `0DA8C4C17133F414`). The
    tracked pin `tools/og1_dandi_pin.json` is rebuilt from the DANDI API at Stage 0 and must equal this cache.
- **STILL OPEN:** the paper's per-recording unit counts (gate (e); the amendment rule in section 2 applies).
- **Several sessions per subject.** HO2 and HO3 each carry 9 spike-sorted sessions, beside 9 raw `_ecephys` files.
  Rule: E is computed per session, a subject's E is the median over its sessions, and a session that fails
  readability is dropped and counted. The pin lists every session file used.
- **Not downloaded:** every raw `*_ecephys.nwb` file (HO1's and HO4's are about 2.15 GB each; HO2's and HO3's 9 each
  total about 20 GB per subject).
- **Excluded from the verdict:** primary tissue (Pr1-Pr8) and murine organoids and slices (MO, MS). They may be read
  later as disclosure only; that is not pre-registered here.
- **Pin `tools/og1_dandi_pin.json`:**
  - dandiset 001603 and version 0.260923.0623.
  - Per asset: path, size, and DANDI's `dandi:sha2-256` (read 2026-10-23: HO5 `994089b04fee86bd...`, HO1
    `fcf777c3f2b37cf5...`).
  - The DANDI citation string, verbatim.
  - CC-BY-4.0 attribution rides in the module as a constant, as H01's does.
- **Storage:** `data/organoid/og1_<subject>.npz` (unit spike times in seconds), gitignored through `data/`. Never
  written to `brain_lake.bin` (P4b).

## 2. Stage 0 -- ingest (no Brain, no lake)

**Module:** `sensorimotor/organoid_ingest.py`, in the afferent package.
- graph-exists for "organoid MEA electrophysiology recording ingest" returned NO LEXICAL MATCH, which is UNKNOWN, not
  absence. Re-query in the tree's vocabulary at build.
- Registered in `__init__` REGIONS and `run_selfchecks`.
- Its only caller until Stage 2 is the Stage 1 instrument. Its dormancy disposition (EXEMPT-with-reason until Stage 2
  wires it, or reached through the instrument) is decided at build against the gate's own output.

**Gates.** Each is a refusal, each carries a constructed red recorded in the self-check, and a refusal yields no
artifact.

| Gate | Passes when | Constructed red |
|---|---|---|
| (a) LICENSE | `license_ok("cc-by-4.0")` | a pin whose license is empty: refused |
| (b) BYTES | streamed sha256 and size equal the pin, checked BEFORE `h5py` opens the file | a one-byte-flipped copy: refused |
| (c) INDEX | `spike_times_index` non-decreasing, last entry == `len(spike_times)`, at least one unit (the silently-empty trap made loud) | a truncated index: refused |
| (d) TIME UNITS | of {seconds, milliseconds, samples at 20 kHz}, the declared seconds reading puts the last spike closest to the session duration the file records | spike times x 1000: refused |
| (e) REPRODUCE | each subject's unit count equals the count the paper reports | a file with one unit dropped: refused |

On (e): if the paper reports no per-recording unit count, a dated pre-data amendment replaces (e) before any read.
It is never silently weakened.

## 3. Stage 1 -- succession census (no Brain). The cheap decider

**Question.** Can SNDL's state encoding read succession in these recordings that persistence does not explain?

**Encoding.**
1. Bin width W. For each bin, the population vector of per-unit spike counts, L2-normalized; all-zero bins form
   their own "silent" state.
2. Fit `_dpmeans` with threshold lam on the bins in a SEEDED RANDOM ORDER (`random.Random(f"og1.fit:{subject}")`), so
   cluster ids cannot encode recording time.
3. Label the bins in time order. The result is the state sequence s.
4. Units cannot be matched across organoids, so the clusters are fit per subject. Only (W, lam) transfers.

**Statistic.** E = `mi_bits(s[:-1], s[1:])` minus the `q95` of the same quantity over 200 RUN-NULL surrogates.

**The run null (the one new instrument piece).**
1. Collapse s to runs of (state, length).
2. Keep the run-length sequence exactly.
3. Redraw each run's state from the empirical distribution of run states, excluding the previous run's state
   (renormalized), and re-expand.

This keeps persistence (the run lengths) and the no-self-merge structure, and destroys which-state-follows-which.
- Each surrogate's realized marginal deviation is reported; marginals are preserved only approximately (disclosed).
- Seeds: `random.Random(f"og1.null:{subject}:{k}")`.
- W2's whole-sequence shuffle is reported as a disclosure line only. Organoid bins are autocorrelated, so persistence
  alone beats that shuffle, and it never decides.

**Readability is counted on TRANSITIONS, not bins.**
- A subject is readable when its run count R >= 10 x the number of distinct observed (state, next-state) pairs among
  run transitions. That is `mi_read`'s `N_MIN_PER_CELL = 10`, counted where the information is.
- A subject below that bar is REFUSED.
- Long runs inflate bin counts on the diagonal while the transition sample stays small, which is why bins are the
  wrong denominator.

**Parameter selection (DEV only: HO1-HO4).**
- Grid: W in {5, 10, 20, 50} ms, x lam at the {0.25, 0.5, 0.75} quantiles of each subject's pairwise bin distances
  (scale-free across unit counts).
- Pick the cell with the highest dev-median E among cells readable on all four dev subjects. Ties go to the larger W.
- The chosen cell is frozen in the record and applied unchanged to the held-out set.

**Primary verdict (HELD-OUT only: HO5-HO8).**
- PASS iff E > 0 on at least 3 of 4 held-out subjects. Under the null each subject passes with probability about
  0.05, so P(>= 3 of 4) is about 0.0005.
- **Transfer guard:** if the frozen (W, lam) is unreadable on 2 or more of 4 held-out subjects, the outcome is an
  INSTRUMENT REFUSAL. There is no scientific verdict, a dated amendment follows, and the disposition below does not
  fire.

**Positive controls.** These run first; any failure means the harness is BLIND and no verdict is read.
- PC1: W2's deterministic 3-bit successor stream, re-expanded with the real dev run-length distribution: must PASS.
- PC2: one dev subject's sequence with its run-state order replaced by a fixed cycle at the real run lengths: must
  PASS.

**Reachable reds.** These must FAIL; a pass on any one is an instrument defect, and no verdict is read.
- R1: SNDL's own live packs (the W2 3.1c vision and audio label streams): must FAIL.
- R2: one extra run-null draw per held-out subject, set aside before the 200: must FAIL on at least 3 of 4.
- R3: a rate-matched Poisson surrogate per held-out subject (each unit's own rate, the same bin grid): must FAIL on at
  least 3 of 4.

**Disposition.** The arbiter rule applies because the paper already claims succession in these bytes.

| Outcome | Then |
|---|---|
| PASS | Stage 2 opens, under its own dated pre-data amendment |
| FAIL (readable, controls green) | Before any deletion, run the ARBITER: the paper's own statistic (within-burst rank-order consistency against its shuffle, as the paper defines it, pinned at freeze), on the same bytes |
| FAIL, and the arbiter PASSES | SNDL's encoding is blind to the published succession. This is a measured negative about the ENCODING, recorded in refuted-and-negatives. The ingest stays; a successor encoding owes a new pre-registration |
| FAIL, and the arbiter FAILS too | Nothing in the tree reads what the paper read. The ingest module and its artifacts are DELETED (P8 item 5) and the negative is recorded. E4 stays blocked and the event camera stays the candidate |

The verdict's scope is spontaneous activity of human organoids at the frozen encoding (the HO5-HO8 device is
unverified). It never reads as "organoids have no succession".

## 4. Stage 2 -- the stream as an engine problem (only on a Stage 1 PASS; owes a dated pre-data amendment)

These points are fixed now; the amendment fills in the rest.
- **Input.** Spike trains enter through `sensorimotor/afferent_boundary` as a new declared modality row (P2: the ONE
  afferent lane), feeding the reservoir input through a typed `engine/wiring_manifest` port.
- **Problem.** Next-state prediction over Stage 1's frozen states, on the last 30% of each recording
  (chronologically held out) and on the held-out subjects.
  - **Open design question:** `taught_problems.Problem` is text-only (`Turn(text, role, label)` routed by
    `route_text`). The amendment decides how a spike-stream kind extends the ONE contract. A second contract is never
    an option.
- **Score.** Normalized held-out accuracy relative to the persistence predictor (P4d's solved predicate, P1 case (a)).
- **Arms:**
  - A0: persistence, the brain-free floor.
  - A1: the incumbent engine (fixed `KernelLsmPool` plus the online delta readout).
  - A2: the reservoir removed, readout on the afferent input. This is P4d's reachable red: A1 must beat it.
  - A3: A1 run on the run-null stream. This is the control that is not the method: it must NOT beat A0, and if it
    does, the verdict is void.
- **Seeds.** Dev 17000-17007, confirmatory 17010-17017. No use of either block was found in `evidence/` or `tools/` on
  2026-10-23; recheck at freeze. Each run uses a throwaway NORD_HOME with `persist=False`, never the live lake.
- **What it unblocks.** E4's stream half only. E4's LIF half stays open, and OG1 never claims a benefit from
  reservoir plasticity.

## 5. Stage 3 -- HNOCA composition prior for the grower (GATED)

- **Gate.** The ONE grower merge (owner #3403, #3423) has landed on main. On 2026-10-23 `growth/growth_loop.py` still
  exists on main, so the gate is CLOSED.
- **Data.**
  - The HNOCA core: CELLxGENE collection `de379e5f-52d0-498c-9801-0f850823c847`, "The Human Neural Organoid Atlas",
    1,767,346 cells, h5ad 18.8 GB (He et al. 2024 Nature, DOI 10.1038/s41586-024-08172-8).
  - Its license is to be VERIFIED when the gate opens (Stage 3 reads nothing before then).
  - Read with `h5py`, `obs` only: cell type, age in days, protocol, region. The output is a counts table,
    `data/organoid/hnoca_composition.npz`.
  - Pin: the sha256 we compute ourselves (CELLxGENE publishes none), the size, and the dataset version id.
- **Entry.** A second keyed dataset in `growth/cortical_prior.py`, the ONE place external constants live, with units,
  provenance and accuracy_bound mandatory. Developing tissue and H01's adult cortex become two datasets in one
  mechanism.
- **Test.** P4d: "anatomical data is a fitted prior that must beat a shuffled copy". Three arms of the merged grower,
  each admitting types and Dale signs by a different table:
  - HNOCA's age-indexed fractions.
  - H01's static fractions.
  - HNOCA with its age labels permuted (same marginals).

  All three are scored on the held-out readout of the problem the grower grows on. The test is pre-registered in its
  own amendment when the gate opens.

## 6. Excluded: KCL figshare (10.18742/30394168, CC0)

- **Why excluded:**
  - 001603 already supplies a second independent lab for held-out replication.
  - KCL's raw data is Axion `.raw` continuous voltage, about 2.9 GB per file, 28 files per part over three parts.
    Reading it needs a new decoder AND new spike detection: two new mechanisms with no job.
  - Its well-average CSVs (development time course, chemical LTP) feed no SNDL mechanism.
- **Re-admission condition:** Stage 1 FAILS on 001603, and a second preparation (dissociated 2D, Axion) is wanted to
  separate the encoding from the preparation. KCL then enters under its own pre-registration.

## 7. Build, review, battery (the per-arc rhythm)

**Arc 1 = Stage 0 + Stage 1:** this file frozen, the ingest, and the census instrument.
- fleet.py lanes in parallel on disjoint files:
  - L-ingest: `sensorimotor/organoid_ingest.py`, the pin, gates (a)-(e) with their reds.
  - L-census: the run null, the census, PC1-PC2, R1-R3.
- Reviewer lanes take each result as it lands.
- Then: the integrator promotes onto int3, a panel of three lenses runs, the fold, the Opus closing review, ONE
  battery, the closing record commit, the push.
- The census campaign runs ONCE on the pushed head.

**Estimates, not commitments:**
- Download: about 0.64 GB (24 units files), minutes.
- Building Stage 0: about 3 h. Building Stage 1: about 3 h.
- Census compute: under 1 h on CPU.
- Panel, fold, review and battery: about 3 h (the battery alone is about 70 min).
- Stage 2: days, and only after Stage 1.

## 8. Not claimed, disclosed, owed

- **Not claimed:**
  - fluid intelligence
  - consciousness
  - any reservoir-plasticity benefit
  - "organoids learn"
  - that matching organoid statistics improves the engine (owner rule 2026-09-30: judge against engine function).
    That judgement is Stage 2's, not Stage 1's.
- **Disclosed:**
  - The dev/held-out split is by lab, and lab is confounded with protocol and preparation.
  - Spike-sorting quality is the depositors'; the sorter is to be read from the NWB metadata at freeze.
  - (W, lam) transfers across organoids, the clusters do not.
  - The run null preserves marginals only approximately.
- **Owner decisions, 2026-10-23 (verbatim):** "Approve OG1 as written, including the KCL exclusion. Register it and
  freeze it on int3, then start Arc 1." That settles registration, the freeze and KCL.
- **Still a build-time note:** `h5py` is imported from a sensorimotor module. It is already declared (`nir` extra), and
  the dependency ledger decides the tier at build.

## Amendment A1, 2026-10-23 (PRE-DATA for Stage 1; append-only; the frozen text above is unchanged)

**Trigger.** Stage 0's STOP rule fired (board #4421, lane og1-ingest). No per-recording unit count is published:
- Sharf 2022 reports "224 units ... from four different organoids", an aggregate over a filtered subset.
- Alam El Din 2025 reports one unlabelled recording.
- van der Molen 2025 reports none.
- The bioRxiv preprint was unreachable.

**What had been read when A1 was written, disclosed.** Section 2 says (e)'s replacement lands "before any read". It did
not. My brief let the lane build (a)-(d) meanwhile, and the lane ran (a)-(d) on all 24 real files and wrote the npz
artifacts. The lane read, and the integrator saw in its REPORT:
- file structure
- unit counts per file
- the recorded span (`processing/curated_binning/t_spk_mat` rows / rate)
- the last spike time
- total spike counts per file
- the fact that some files share identical counts

No activity statistic of any kind was computed: no rate, burst, state, transition or MI. Stage 1 has read nothing.
Every rule below is fixed without reference to any Stage 1 quantity.

**A1.1 Gate (e) is REPLACED by (e') REPRODUCE-THE-DEPOSITORS'-BINNING.**
- Each file stores the depositors' own 1 ms binary raster `t_spk_mat` (rows = bins at `rate` = 1000 Hz, columns =
  units, 1 = at least one spike in the bin), with its `starting_time`.
- (e') rebuilds that raster from `units/spike_times` + `spike_times_index` and requires EXACT equality, cell for cell.
- Declared convention: bin = floor((t - starting_time) x rate), and columns in units-table order.
- Exactness is the point, as with H01's pinned 45,404 / 49,377: it proves the decoded index, the unit order and the
  time unit are the depositors' own, on the same bytes.
- **Extremum rule:** under the alternative convention (round instead of floor), and under a one-column unit-order
  shift, the agreement must be strictly lower than under the declared convention. So the bar decides between readings
  rather than being a threshold.
- Constructed reds: spike times x 1000; a unit-index off-by-one; a planted single-bin flip.
- **On a mismatch:** REFUSED, the mismatch fraction and the cells are reported, and nothing is tuned. A file that
  stores no `t_spk_mat` is REFUSED under (e'); none of the 24 lacks it, per the structure probe.

**A1.2 The DUPLICATE rule (it amends section 1's "several sessions per subject" and section 3's multi-session rule).**
- Within one subject, two files are DUPLICATES iff their decoded `spike_times` (float64) and `spike_times_index`
  (int64) arrays are exactly equal (`numpy.array_equal`).
- One per duplicate group is kept, the lexicographically first session id; the dropped ones are counted and listed.
- Files whose counts agree but whose arrays differ are DISTINCT recordings and are kept.
- Stage 0 writes the grouping into its session manifest, and Stage 1's per-subject median runs over distinct recordings
  only.
- Every pinned file still passes gates (a)-(d) and (e'); the pin keeps all 24.
- Why: the probe found HO2 and HO3 files in groups with identical unit counts, spike counts, byte sizes and last-spike
  times under different sha256. Counting one recording several times would weight it several times in the median
  without adding a measurement.

**A1.3 Gate (d), clarified, not changed.** The span (d) compares against is the depositors' `t_spk_mat` rows / rate:
180 s for HO1-HO4 (HO3 has two files at 180.149 s) and about 600 s for HO5-HO8. It is the depositors' record stored beside
the spikes, not an independent clock. Disclosed as such.

**A1.4 Disclosures.**
- The files carry no device group, no electrodes table and no `obs_intervals`, so the HO5-HO8 device stays the
  secondhand "MaxTwo".
- The files name UCSB, not UCSC, for HO1-HO4.
- The sorter is Kilosort2 with ISI / rate / SNR filtering, from `experiment_description`.
- `spike_times_index` is stored as uint16 in 7 of 8 subjects; it is cast to int64, and (c) refuses a wrapped index.
- The pin matches the freeze census on all 24 paths and sha256, and on sizes at the census's 0.1 MB precision only.
- Load during the insula battery: the lane's real-file conversion (about 01:35-01:38) ran before it read the
  integrator's HOLD (board #4416), inside the insula docs-only battery's window. That is disclosed to the insula lead.

## Amendment A2, 2026-10-23 (PRE-DATA for Stage 1; append-only; the text above is unchanged)

**Trigger 1: (e') refused all 24 real files under A1.1's declared convention** (board #4458, lane og1-ingest, commit
2533719f).
- Mismatch fraction: 0.0004-0.0014 of cells.
- The ones counts were equal.
- Every mismatch was an adjacent-bin pair.

The integrator then probed the depositors' representation itself (`og1_raster_probe.py` and `og1_raster_exact.py`,
kept in the OG1 lane folder). It read Stage 0 representation only: spike-time grid, stored-bin offsets, per-file
mismatch counts. It computed no activity statistic. Findings:
- Every spike time sits on the 20 kHz sample grid.
- A spike whose sub-millisecond remainder is >= 0.5 lands where floor puts it, every time; a remainder < 0.5 misses
  floor every time.
- Stored bins start 10-29 samples before the spike.

That is MATLAB writing `raster(round(t x rate), u) = 1` with 1-BASED indexing (each file's `identifier` is
`HO*_sorted.mat`), i.e. 0-based row = floor(t x rate - 0.5). **A1.1's floor declaration was the INSTRUMENT'S DEFECT,
not the data's**: the H01 gate (e) precedent, whose replacement was likewise written after its own measurement.

**A2.1 (e')'s declared convention is REPLACED:** row = floor((t - starting_time) x rate - 0.5). Spikes whose row
falls below 0 (t < 0.5 ms; 28 spikes across 4 files: 8 + 8 in two HO2 files, 6 + 6 in two HO3 files) are dropped, as the stored raster does.
- Measured on all 24 files: this convention rebuilds every stored raster with **0 mismatched cells (24 of 24, about
  3.6e8 cells)**.
- The alternatives tried, with total mismatched cells: floor 365,476; round-half-even 704,145; floor(x + 0.5) 728,849.
  The lane's own runs also showed ceil and floor +/- 1 to be inexact.
- Seven conventions were tried in all; exactly one is exact. That count is disclosed, so the exactness is not read as
  a first guess.
- The extremum rule stands: the declared convention must be exact, and floor, round-half-even, floor(x + 0.5) and a
  one-column unit shift must each be strictly worse.
- The reds stay: times x 1000, a unit-index off-by-one, a planted single-bin flip.
- Exact equality is still required; nothing is toleranced.

**Trigger 2: the census lane's findings F1-F4** (board #4490; `rows/og1-census-REPORT.md` on
claude/og1-census-1023), measured on SYNTHETIC streams only.

**A2.2 What "a red must FAIL" means.**
- A red FAILS iff it does not PASS the verdict's own test (readable and E > 0). This adopts the lane's F3 reading.
  Each red's readability is recorded beside its outcome.
- An UNREADABLE red is VACUOUS. It is run and reported, and carries no evidential weight; P8 item 4 requires the
  disclosure.
- **R1** (SNDL's live packs, 256 frames, both packs unreadable) is therefore VACUOUS by construction (F4). It stays
  in the record and is never counted as a demonstrated red.
- **Per-subject rule:** each held-out subject needs at least one NON-vacuous red, readable and failing, among R2 and
  R3. A subject without one is VOID: neither pass nor fail.
- **Refusal rule:** if 2 or more of the 4 held-out subjects are VOID, the outcome is an INSTRUMENT REFUSAL, mirroring
  the transfer guard. The verdict reads only non-VOID subjects and still needs E > 0 on at least 3 of the 4 subjects,
  so a VOID subject counts against PASS.

**A2.3 Disclosed, bar unchanged.**
- **F1:** on streams generated as the run null, the census passes 6-9% of the time, not 5%, because the run-state
  distribution is estimated from the data under test. The family-wise P(>= 3 of 4) is therefore about 0.001-0.003,
  not the frozen 0.0005. The bar is kept, because 0.003 is still a strong bar. The t3 configuration was chosen after
  seeing pooled rates; disclosed.
- **F2:** the "R2/R3 must fail on >= 3 of 4" bar falsely voids a sound harness about 3% of the time. That error runs
  in the conservative direction; disclosed.

## Amendment A3, 2026-10-23 (PRE-DATA for Stage 1; append-only; the text above is unchanged)

**Trigger.** Reviewer og1-rv-a's challenge C-1 (board #4529) on the census lane's Round 3 code:
- A VOID subject was dropped before the transfer guard counted unreadable subjects.
- So {one VOID, one unreadable, two E > 0} returned FAIL.
- FAIL opens the arbiter and deletion path, yet only two subjects were informative.

The integrator ruled on the board at 2026-10-23 03:20 (#4534). The ruling is that board record, and this text records
it. It was committed here at promotion, because int3 was frozen for another arc's battery. Stage 1 had read nothing.

**A3.1 One guard, on the UNION.**
- A held-out subject is NON-INFORMATIVE if it is VOID (A2.2) OR unreadable at the frozen (W, lam) (section 3's
  transfer guard).
- If 2 or more of the 4 held-out subjects are non-informative, the outcome is INSTRUMENT REFUSAL.
- Otherwise PASS iff E > 0 on at least 3 of 4, with a non-informative subject counting as not passing; else FAIL.
- VOID subjects are still read for readability; no E of theirs enters the verdict.

This replaces the two separate counts (VOID >= 2 in A2.2; unreadable >= 2 in section 3) with their union. Each
separate count still implies a refusal, so the rule is stricter and never looser: FAIL, and with it the arbiter and
any deletion, now needs at least 3 informative subjects.

**A3.2 Built and reviewed before any data read.**
- Lane og1-census Round 4 (36898fc9, result #4537) built it. Its tests show rv-a's case red on the Round 3 code
  (FAIL) and green as INSTRUMENT_REFUSAL.
- Both reviewers agreed (#4540, #4541), and rv-a resolved C-1 (#4539).
- The same round verified the R1 pack digests before any compute and recorded the campaign's wall time.

## Amendment A4, 2026-10-23 (PRE-DATA for Stage 1; append-only; the text above is unchanged)

**Trigger.** The three-lens panel on the Arc 1 stack (`claude/og1-arc1-1023` 0b382dd3; lens files
`D:\AI-project\lanes\og1-panel-{a,b,c}-1023-row\LENS-{A,B,C}.md`, with their probes):
- Lens A: BLOCK, A-1 to A-12.
- Lens B: APPROVE-WITH-FOLDS, B1 to B16.
- Lens C: APPROVE-WITH-FOLDS, C1 to C10.

The integrator then measured the frozen encoding on synthetic, real-shaped sessions. The files are in
`D:\AI-project\lanes\organoid-og1-1023\`:
- `og1_timing.py` and `og1_timing.out.txt`
- `og1_pos_probe.py` and `og1_pos_probe.out.txt`

**What had been read when A4 was written, disclosed:**
- Stage 0 metadata of the 24 files: unit counts, and per-unit `x_pos` / `y_pos` (all 24 carry them, no NaN).
- State counts and run counts of the frozen encoding on SYNTHETIC streams.
- No Stage 1 statistic, state sequence, transition count or MI on organoid data. Stage 1 has still read nothing.

**Scope, stated first.** After A4 the census reads SPATIALLY ORGANISED succession at the grain of electrode-position
groups. A sequence whose unit order is spatially interleaved within a group is invisible to it. That is a disclosed
limit of the encoding, and it narrows what a PASS or a FAIL means: neither verdict is about "the published
preconfigured sequences" in general.

**A4.1 The encoding is REPLACED (A-3, and the integrator's measurement).**

*Why the frozen encoding goes.* Population vectors are L2-normalised and non-negative, so every pairwise distance is at
most sqrt(2), and two bins that each hold a single spike from different units sit at exactly sqrt(2). On sparse bins,
DP-means therefore has no middle ground:
- with lam below sqrt(2), every single-unit bin opens its own state;
- with lam at or above sqrt(2), everything merges into one state.

Measured on synthetic sessions shaped like HO1 (131 units, 180 s, 31,756 spikes):
- bursty firing gives 107 / 129 / 130 / 131 states at W = 50 / 20 / 10 / 5 ms, i.e. about one state per unit;
- uniform firing gives 17 / 90 / 20 states at W = 50 / 20 / 10 ms, varying erratically with the float tie lens A
  found.

With K about 130 states, the distinct pairs dwarf the 3,000-11,000 runs, so every dev session would be unreadable:
an INSTRUMENT REFUSAL by construction. `_dpmeans` is retired from this instrument together with this measurement.
That is P2 applied as written: a mechanism measured unfit for the job leaves it; it is not duplicated. The median
split below adds no clustering mechanism.

*The replacement.* For each SESSION:
- its units are split into G spatial groups from the Units table's positions (metadata, blind to activity):
  - G = 2: x below or at the session's median x -> group 0, else group 1;
  - G = 4: the same split on x, crossed with the same split on y;
  - a unit exactly at the median goes to the lower group;
- a bin's state is the bit code of the groups holding at least 1 spike in that bin; 0 is silent.

So G = 2 gives at most 4 states and G = 4 at most 16, and readability (runs >= 10 x distinct observed pairs) needs
at most 120 or 2,400 runs. The lam quantile grid, `LAM_SAMPLE` and the float32/float64 question all go with DP-means.

**A4.2 The grid is W x G.** W in {5, 10, 20, 50} ms x G in {2, 4}: 8 cells.
- The selection rule is section 3's: the best dev-median E among cells readable on all 4 dev subjects; ties go to the
  larger W, then the smaller G.
- An EMPTY readable set is an INSTRUMENT REFUSAL: a dated amendment follows, never a silent fallback.

**A4.3 The statistic is read on RUN transitions (A-1).**
- r is a session's run-state sequence: consecutive equal bin states are collapsed, so r has no self-transitions.
- E = `mi_bits(r[:-1], r[1:])` minus the q95 of the same quantity over 200 null surrogates.
- The bin-level statistic is retired. Lens A measured why: a deterministic burst cycle separated by long silences gave
  bin-level E of about -2 bits (pass 0 of 10). The same null on run transitions separated a true chain (1.620 against
  null q95 0.284) from no succession (0.261 against 0.287).
- Persistence is excluded by construction, because a run sequence has no self-transitions.
- `_marginal_dev` is retired: at run level there is no bin marginal for the null to distort. That also closes B11's
  duplicate of `reservoir/neural_sampling._tv`.

**A4.4 The null is BLOCK-LOCAL (A-2).**
- A session's R runs are split into B contiguous blocks of near-equal run count, B = max(1, min(10, floor(R / 50))).
- Each surrogate run's state is drawn from the run-state distribution of its OWN block, excluding the previous run's
  state and renormalised.
- B is recorded per session. At B = 1, drift is uncontrolled; that is recorded as such and disclosed in the verdict.
- Seeds stay `og1.null:{subject}:{k}`.

**A4.5 Controls and reds.**
- PC1: the 3-bit successor map at run level.
- PC2 now keeps the real (state, length) coupling. It is the dev session (with the most runs) at the chosen cell, its
  silent runs kept in place, and its NON-silent run codes replaced in order by a deterministic cycle over the
  non-silent codes it actually shows. It must PASS.
- R1 stays vacuous (A2.2).
- R2 is one set-aside draw of the block-local null.
- R3 is per-unit Poisson at each unit's mean rate on the real bin grid, with the units' real positions.
- At least ONE held-out subject must carry a non-vacuous R3 (readable and failing), else the harness is BLIND (B9).
- Synthetic, SPIKE-level tests, each with positions and with its red shown against the Round 4 code where it
  differs. The shapes are those of lens A's probes, used as a specification rather than as scripts:
  - planted spatial propagation (units ordered by x fire in sequence after a shared onset, long silences between)
    -> PASS;
  - the same units and onsets with no propagation order -> FAIL;
  - drift halves and slow drift with no succession -> FAIL;
  - semi-Markov dwell with no succession -> FAIL.

**A4.6 Declared choices, now in the tracked text (B5).**
1. The fit unit is the SESSION, not the subject. This is a dated correction of section 3's "per subject": the
   sessions of HO2 and HO3 differ in units (4 distinct recordings each), so a subject-level fit is undefined.
2. PC1 uses 400 runs (`PC1_RUNS`).
3. The R2/R3 ">= 3 of 4 fail" bar stands, with vacuous fails counted as failing (A2.2), plus A4.5's non-vacuous R3.
4. The grid tie rule is as in A4.2.
5. PC2's session is chosen as in A4.5.
6. R3's seed string is `og1.poisson:{subject}:{session}`.
7. R2 and R3 use the subject's first kept session; each held-out subject has 1.
8. Campaign order: dev grid -> chosen cell -> controls -> held-out census. The held-out census is not run if any
   control fails.
9. The W2 whole-sequence shuffle is a disclosure on the final readings only.
10. VOID subjects are read for readability only; no E is computed for them (A-8, B10.1).
11. Record JSON uses `default=float`.

**A4.7 Campaign entry and provenance (A-5, A-6, A-10, B4, B7, B14, C2-C6, C8).**
- The entry is `python -m snn_data_lake.instruments.og1_succession --campaign <data_dir> <record_path>`. Any other
  argv is refused, and an existing record path is refused (run once).
- **Before any compute:**
  - the session manifest must cover every pinned asset (24) and every DEV and HELD_OUT subject;
  - its splits must equal the instrument's DEV and HELD_OUT;
  - the pack digests must verify, and `PACK_SHA256` must equal `memory_diagnostic.ASSETS`, checked by a test that
    reads it via `ast`.
  Otherwise the campaign refuses, and the refusal itself is written as the record.
- **The record carries:**
  - git HEAD;
  - the freeze, cited as `a3a15806 (= ccf6a45b by patch-id 968e796794d13546f2a57cb123d64ea0a2c900c3)`;
  - the sha256 of `og1_succession.py`, `attention_schema.py`, `infra/_mathutils.py`, `organoid_ingest.py`, the
    manifest and every npz;
  - B and the run count per session;
  - the wall time.
- **Data:** the campaign reads the main checkout's gitignored `data/organoid/`, filled with the 24 npz files and
  `og1_sessions.json`. The record names the directory and the exact invocation. The files are
  `og1_<subject>_<session>.npz`; that supersedes section 1's single-session pattern. Pass Windows-form paths: the
  venv is a Windows interpreter, and a `/mnt/...` path silently resolves to `D:\mnt\...` (C5).

**A4.8 Stage 0 additions (A-4, A-11, A-12; contract change for A4.1).**
- Gate (c) refuses non-finite and negative spike times.
- Gate (d) refuses any spike at or past `duration_s`.
- The npz gains `x_pos` and `y_pos` (float64, one per unit, NWB order).
- The session manifest records, per file, the counts of within-unit duplicate timestamps and of out-of-order spikes.
  Lens A measured more duplicates in held-out than in dev, which is confounded with lab and disclosed.
- Lens A measured all 24 real files clean on the new refusals: 0 NaN, 0 negative, last spike <= `duration_s`.

**A4.9 What a FAIL means now (B1, B2, C1; it supersedes section 3's FAIL rows).** The P8 arc's amendment to
`docs/POLICIES.md` is to land on int3 before this arc. Its items 1-3 bind OG1. Outcomes:
- **PASS:** Stage 2, under its own dated amendment.
- **INSTRUMENT REFUSAL or BLIND:** no scientific reading; a dated amendment; nothing else changes.
- **FAIL (readable, controls green):** the outcome HALTS. No negative is recorded and the tree does not change until
  two things exist:
  1. a dated amendment that pins the arbiter in full (the burst definition, the rank-order statistic, its shuffle,
     its bar, the subjects it reads) BEFORE the arbiter reads any byte (B2);
  2. the NULL-RECORD, written into this file.
  Then P8 amendment items 2 and 3 apply:
  - both modules stay off every live path (they already are);
  - their dormancy rows take the `scale-retest` disposition, naming the NULL-RECORD, `since=<the arc's base on
    origin/main>`;
  - ONE re-test is owed, at least 10x larger on the DATA axis: at least 40 distinct held-out organoid recordings, or 10x
    the held-out recording-seconds, with the same endpoint, bar and control.
  - Section 3's "DELETED (P8 item 5)" branch is superseded: item 3 allows no deletion before that re-test.
- **The record format** (an example, not a record):
```
NULL-RECORD: og1_succession | units=<total units over the 4 held-out recordings> | ticks=<total bins at the chosen W> | seeds=200 | mde=<(z.95 + z.80) x sd of the 200 null values, median over subjects, bits> | retest=PENDING since <40-hex base sha>
```
- **Re-test state today:** no spike-sorted human organoid source at 10x exists on DANDI.
  - 001872: human, 48 wells, raw ecephys only, so it would need spike sorting, a new Stage 0 mechanism.
  - 001336: human, 3 subjects, raw only.
  - 001374: mouse.
  - The organoid lab's data (owner's channel) is the other candidate.
  - Under P8 amendment item 3, an OG1 null would keep every later battery red until a RETEST-RECORD exists. Whether to
    launch the campaign before a 10x source is pinned is the owner's call, asked on 2026-10-23.

**A4.10 Disclosures owed (B4, B10, B12, B13, C10).**
- **The freeze re-landing:**
  - ccf6a45b was authored 2026-10-23 01:21:21. The first Stage 0 real-file read was about 01:35.
  - The re-landed freeze is a3a15806, authored 02:32:15, after that read.
  - Their patch-ids are equal (968e7967...).
  - ccf6a45b is kept on the local branch `claude/og1-freeze-1023` and is not pushed.
- **The census lane's HOLD-window load:** pytest, a 500-seed calibration and benchmarks, synthetic only, about
  01:31-01:45, inside the insula docs-battery window. Disclosed to its lead (#4435). A1.4 named only the ingest lane's
  load.
- **"Pre-data" means pre-Stage-1-data.** A1 and A2 were written after Stage 0 reads; A2.1 was written after (e')
  refused all 24 files under A1.1.
- **A3's order:** its code (af49969d) preceded its text (8b306943); ruling #4534 preceded both.
- **P2 debt:** `organoid_ingest.stream_sha256` sits beside `gates/selfcheck_cache._sha256_file`. Consolidation is
  deferred; only the byte count differs.
- **h5py:** it is now also imported by `organoid_ingest`, recorded in the `pyproject.toml` `[nir]` comment.
- **The CHANGELOG's "persistence alone cannot pass"** was untrue (A2.3 F1 measured 6-9%) and is corrected in the same
  fold.

## Amendment A5, 2026-10-23 (PRE-DATA for Stage 1; append-only; the text above is unchanged)

**Trigger.** Lane og1-ingest4's STOP (board #4570, commits 86eb2337..a2cd0fc3). A4.8's new refusal, "any spike at or
past `duration_s`", with `duration_s` = `t_spk_mat` rows / rate, refused 5 of 24 real files:

| file | last spike | span | spikes past |
|---|---|---|---|
| HO3, 2 files | 180.14945 s | 180.149 s | 2 |
| HO5 | 600.0184 s | 600.018 s | 1 |
| HO7 | 599.9404 s | 599.94 s | 1 |
| HO8 | 599.9701 s | 599.97 s | 1 |

The overshoot is 0.1-0.45 ms. The lane did not weaken the gate. A4.8's sentence "lens A measured ... last spike <=
`duration_s`" did not hold at that rule.

**Cause: the rule's span was inconsistent with A2.1, not a data defect.**
- Under A2.1's declared convention, row = floor((t - starting_time) x rate - 0.5).
- A spike lands in the last row, n_rows - 1, for t up to (n_rows + 0.5) / rate.
- So the depositors' raster itself covers [0.5 / rate, (n_rows + 0.5) / rate).
- Every one of the 5 overshooting spikes falls inside that upper edge, because each overshoot is below 0.5 ms. Gate
  (e') had already reproduced these rasters exactly with those spikes in them.

**A5.1 The span is the A2.1 raster's own upper edge.**
- `duration_s` = (n_rows + 0.5) / rate.
- A spike is OUT OF SPAN iff its A2.1 row is >= n_rows, equivalently t >= (n_rows + 0.5) / rate.
- Gate (d)'s new refusal uses that rule.
- No tolerance is introduced: the edge follows from A2.1, which was fixed before this rule was written.
- Constructed reds stay: a 1.0 s raster with spikes to 1.9 s is still refused, and a spike at exactly
  (n_rows + 0.5) / rate is refused.
- The census's bin grid covers [0, `duration_s`) under the same definition.

**Disclosed:** A4.8's "last spike <= `duration_s`" statement is withdrawn by this amendment. The rule A5 replaces was
written by the integrator without checking it against A2.1. What A5 read: the 5 refusals' last-spike times and
spans, from Stage 0 only. Stage 1 has read nothing.

## Amendment A6, 2026-10-23 (PROCESS ONLY: no statistic; append-only; the text above is unchanged)

**Owner decision, verbatim: "1".** It answered the integrator's question:
> "Which path for OG1? 1. Stage 0 plus a Stage 1 battery: push Stage 0 now, fund the independent acceptance battery and
> one candidate (stop at Stage 0 if it fails), and rule on the battery-red cost before any census code. 2. Stage 0,
> then redirect: ..."

The owner had asked: "Compact and use agent-fleet to debate on the next step".

**The debate.** Thread OG1-DEBATE: positions #4719 (continue) and #4720 (against), exchanges #4721-#4728, and the
judge's verdict **#4729**. The lanes' files are in `D:\AI-project\lanes\og1-deb-{a,b,j}-1023-row\`.

**A6.1 Stage 0 is pushed without Stage 1.**
- This arc carries the ingest (`sensorimotor/organoid_ingest`, the pin, gates (a)-(e') and the A4.8/A5 refusals), its
  registration, and Amendments A3-A6.
- The census module is NOT on main:
  - the A3-design census lives on branch `claude/og1-arc1-1023`, superseded and not pushed;
  - the A4-design census lives on `claude/og1-census5-1023` (task OG1-CENSUS-R5), not pushed, and blind on a
    real-shape no-sequence manifest (#4682).

**A6.2 Stage 1 status: NOT BUILT.** Four null designs are retired, all on synthetic data, before any organoid Stage 1
read:

| null | retired by | evidence |
|---|---|---|
| bin-level, length-blind | panel lens A, A-1 | LENS-A.md |
| run-level frequency redraw | the real-shape BLIND finding | #4682 |
| quasi-independence (pooled Luce chain) | reads burst composition; its bin-permuted twins pass 10/10 | #4713, D2 |
| the gate "E_full > E_popcount" as spatial attribution | envelope-only, no-order bursts gated "spatial" 10/10 at W = 5 ms, 3/10 at 10-20 ms | #4729 |

The within-burst permutation null (#4713 D3) is a CANDIDATE only. A4's statistic sections stand frozen but are not
implemented on main.

**A6.3 The Stage 1 plan (the judge's conditions, #4729, adopted).**
1. **An ACCEPTANCE BATTERY,** written and frozen by a lane that authored none of the nulls, BEFORE any candidate is
   measured on it.
   - Its must-FAIL "no spatial order" reds include x6 and a2 (at most 1/10) and the judge's envelope-only rows
     (asymmetric, 600 s, sparse, at W = 5, 10 and 20 ms).
   - The red class is "no spatial order", not "exchangeable".
2. **Its must-PASS rows:** i, k, b4, and b's f = 0.25 dose (at least 8/10). The f = 0.1 dose is recorded as the
   floor. x3 is dropped (one burst).
3. **Attribution is a CONDITIONAL surrogate that keeps the envelope** (the population-count sequence and the run
   lengths), never an E comparison across alphabets. This is unsolved: the judge's stratified first attempt leaks
   (6/10).
4. **ONE candidate gets ONE battery run.** If it fails, Stage 1 is recorded NOT BUILT, with the diagnosis (#4682,
   #4692, #4713, #4729). No organoid byte has been read by Stage 1, so a failure carries no NULL-RECORD.
5. **An envelope read is never cited as succession** for E4 or Stage 2, under any typing.
6. **No census code enters `src/` before the owner rules on the battery-red cost of a readable FAIL** (A4.9; P8
   amendment item 3; no 10x source exists).

**A6.4 Owner decisions still open:**
- the battery-red cost of a readable FAIL with no 10x source;
- whether an ENVELOPE outcome may be typed as a non-null result, and only behind a validated attribution, of which
  none exists today.

**What A6 read:** nothing from organoid data. Its facts are the board items cited above.

## Amendment A7, 2026-10-23 (PROCESS ONLY: the owner's ruling on A6.4's first decision; append-only)

**Owner decision, verbatim: "2".** It answered the integrator's question about a readable FAIL with no 10x source
(A4.9; A6.3 item 6):
> "1. **Hold (recommended):** no census code goes into `src/` and Stage 1 reads no organoid data until a 10x source
> exists. 2. **Hold and build the source:** a Stage 0b that spike-sorts DANDI 001872, or uses your lab's recordings.
> 3. **Run now:** a FAIL turns every later SNDL battery red. Which one: 1, 2 or 3?"

**A7.1 The real-data Stage 1 run HOLDS.** No census code enters `src/`, and no Stage 1 function reads an organoid
byte, until a re-test source that meets A4.9 is ingested by a pushed Stage 0 mechanism. A4.9's bar: at least 40
distinct held-out organoid recordings, or 10x the held-out recording-seconds, with the same endpoint, bar and
control. Pinning the source first means a readable FAIL can be re-tested at once, so it never leaves a PENDING record
that reds later batteries.

**A7.2 The synthetic work continues unchanged.** A6.3 items 1-5 stand: one independently frozen acceptance battery,
one candidate, one run, synthetic only (lanes og1-acc and og1-census5, thread OG1-STAGE1).

**A7.3 Stage 0b is PLANNED, not built.** It is a mechanism that produces the 10x source. Two candidate routes:
- **(a) Spike-sorting DANDI 001872's raw ecephys.** This is a new Stage 0 mechanism. It owes its own pre-registration,
  its own refusal gates, and a sorter positive control (ground-truth recordings with a pinned accuracy bar), frozen
  before any 001872 byte beyond the DANDI metadata API is read.
- **(b) The lab's recordings,** under the owner-interprets rule (Claude never opens raw lab files). How a census could
  read them under that rule is an owner question, asked with the Stage 0b plan.
The Stage 0b pre-registration is a separate dated amendment. Research for it (sources, licences, sizes, sorter,
validation) reads public metadata and documentation only.

**A7.4 Still open:** whether an ENVELOPE outcome may be typed as a non-null result (A6.4, second item).

**What A7 read:** nothing from organoid data.

## Amendment A8, 2026-10-23 (Stage 0b pre-registration: the sorted 10x re-test source; FROZEN)

**Status: FROZEN 2026-10-23 on the owner's approval in chat, verbatim: "I approve Stage 0b v6 D1-D6 as recommended".**
D7 was decided earlier the same day, verbatim: "a" (board #4745).
- **Text:** the integrator's draft v6 (`lanes\og1-stage0b-PREREG-DRAFT.md`), unchanged except this header and the
  heading levels. Section numbers below are A8's own ("section 2.4" means A8 section 2.4). "This draft" means A8.
- **Review:** five rounds by og1-0b-rv-a (instrument lens) and og1-0b-rv-b (policy and integration lens):
  - round 1: #4735, #4736;
  - round 2: #4737, #4738;
  - round 3: #4739, #4740;
  - round 4: #4741, #4742;
  - round 5: #4743, #4744.
  - Round 5 found 0 BLOCKs, and both lanes reproduced the k table cell for cell.
  - The round-5 folds (V5-1, V5-2, N5-1..5, the census row, the D6 citation) are integrator-only, not re-reviewed.
    This is disclosed, and they go to the Stage 0b arc's panel.
- **Nothing is built, and no data byte has been read.**
- **Process:** A8 is process and design only. Its first build step is the C0 synthetic calibration, whose addendum
  needs the panel and the owner before any raw byte (section 5).

**Origin:** the owner's ruling "2" (A7, commit 86e25877). A7.1 holds the real-data Stage 1 run until a re-test source
that meets A4.9 is ingested by a pushed Stage 0 mechanism.

**Inputs:**
- `lanes\og1-0bsrc-1023-row\SOURCES.md` and `lanes\og1-0bsort-1023-row\SORTER.md`, with the raw JSON in `api/`;
- the reviews `lanes\og1-0b-rv-{a,b}-1023-row\REVIEW{,2,3,4,5}.md`.
- Fold tags: [a-V3-n] / [b-R3-xx] (round 3), [a-V4-n] / [b-R4-xx] (round 4), [a-V5-n] / [b-R5-xx] (round 5). Earlier
  tags are kept where a text still carries them [a N5-5].

**What Stage 0b is for, and only for:** the ONE re-test source a readable Stage 1 FAIL would owe under A4.9 and P8
amendment item 3. It stays sealed until a NULL-RECORD exists (section 8).

---

### A8.0 Owner decisions (approved as recommended; D7 = "a")

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | When to build | Build now on lane worktrees, where the code, C0, the control and the bridge all run. **Push only after the A6.3 candidate PASSES.** 001872 is sorted after the push. | A control or bridge FAIL then truly means NOT BUILT. A failed candidate leaves nothing dormant on main. |
| D2 | A separate CPython 3.12 venv for the sorter, located by env var `OG1_SORT_VENV` | Yes (an exception to the 3.14t rule, for one tool) | Kilosort4 needs `faiss-cpu`, which has no cp314t wheel. The brain, `src/` and the tests stay on 3.14t. |
| D3 | Kilosort4 (GPL-3.0 per GitHub) as a subprocess tool | Yes | It is never imported under `src/`. P7's ledger scans `src/` only, so no `sort` extra is declared. The tool env is held to P7 by gate (f): the hashed lock, a normalised `pip freeze --all` that must equal the lock, and the script's sha. Under `tools/instrument_guard.py`: no module-scope seeding (the GT seeds live inside functions), no relative `sys.path` and no module-scope `chdir`, and the file must parse under 3.14t `ast`. Tests and (f) HASH `og1_sort.py`; they never import it [b-R3-N3]. |
| D4 | The lab route (section 10) | (ii) if the lab agrees, otherwise (i) | Option (iii) cannot be the re-test. |
| D5 | Selection | The seeded prefix (about 44 files, ~121 GB, all 24 wells). The alternative is all 96 (249 GB, about twice the GPU). | **Disclosure:** the seed was fixed in v1, at the 47th size percentile. Reviewer A's well-first fix changed the RULE, which moved the same seed to the 97th percentile of 2,000 seeds (121.0 GB; median 114). Only seed shopping was avoided. File size tracks channels and activity. |
| D6 | **P4c `tools` baseline +1** for the ONE new script `tools/og1_sort.py` | Yes. The ratchet's own hatch allows "a hand-edit of the JSON inside a reviewed commit" (`p4c_ratchet.py:124-125`, re-read at round 5). | **Why the governed route is unavailable:** a `src/` home would put kilosort and spikeinterface under P7's `DECLARE_ROOTS` ledger, and it needs Python 3.12. The same edit raises `zones.tools` and `total` by 1 and inserts the path into `paths.tools` in sorted order. The removal arc lowers the baseline with `p4c_ratchet --update-baseline` (a down-ratchet), never to a hard-coded number [a-V4-5, b-R4-N2]. |
| D7 | **What a RETEST BLIND writes. DECIDED by the owner, 2026-10-23, verbatim: "a".** P8 has no honest exit for "the re-test ran and could not read". Item 4's "structural" means a null with no path at all, and such a null is deleted without a re-test [a-V4-1, b-R4-B2]. | **(a), as an owner-approved exception to P8 item 3,** stated in A8 and inside the record itself: `retest=STRUCTURAL: NOT an item-4 structural null; the 001872 re-test read BLIND or REFUSED (A8 s11); no deletion licensed; lab-route re-test owed` [a N5-3]. A lab-route RETEST-RECORD replaces it with `REF:` in the same commit. Both `scale-retest` rows move to `permanent` (awaiting-prerequisite: the lab-route re-test), re-dated, in the same commit, which avoids R-EXEMPT and R-EXPIRED (reviewer A verified this against `exempt_problems` and `_assert_scale_retest`). | The options offered and not chosen were (b) a P8 token amendment and (c) keeping PENDING with red batteries. **Disclosed: one clause of (a) is enforced by text only.** "While STRUCTURAL stands, no delete-on-null statement may name `og1_succession` or `organoid_ingest`" has no mechanical check, because R-UNRECORDED is satisfied by any NULL-RECORD in the section. Review enforces it, and the gate prints the STRUCTURAL reason on every run [a V5 Part 3]. |

---

### A8.1 Sources and pins (`tools/og1_dandi_pin_0b.json`; every entry has a `role`)

- **`role: retest`: DANDI 001872, version `0.260630.1038`.**
  - CC-BY-4.0, human brain organoid, MaxTwo 24-well.
  - 120 NWB files, 261,232,362,266 B; `ElectricalSeries` only.
  - Each entry: `dandi:sha2-256`, size, path, age.
  - Eligible: the 96 `sub-sample-well*` files.
- **`role: bridge`:** 001603 HO1 raw `ca0fba55-...` (`sub-HO1_ses-20250924T011900`, 2,149,923,599 B), paired to
  Stage 0 DEV sorted asset `27acb682`; and HO4 raw `95c70f1e-...` (`sub-HO4_ses-20250924T011900`, 2,178,130,991 B),
  paired to `0ca461d3`.
  - **The pairing is by SUBJECT,** because the raw and sorted session stamps differ. It is confirmed from NWB
    identifiers only after the freeze; a mismatch is BRIDGE FAIL [b-R3-B1].
- **Disclosures:**
  - **Same lab as the held-out set:** a scale re-test within one lab, never a cross-lab replication.
  - **Ages are inconsistent:** one well index carries 4 ages within 6 h, and a well's files mix P52-P75 labels.
  - **The 96 files come from 24 subjects.**
  - **Spontaneity is Unknown** (gate (l) decides it).
  - **No paper is linked.**
  - **The seed and the rule were fixed after the metadata was seen.**

### A8.2 Stage 1 clause (P8 item 2; the integrator authored A4.9 and the Stage 1 pre-registration; this panel reviews it)

1. **The OG1 NULL-RECORD is written, superseding A4.9's format, as**
   ```
   NULL-RECORD: og1_succession_w<W>ms | units=n/a | ticks=<int> | seeds=200 | mde=<number> | retest=<R>
   ```
   - `units=n/a` makes a units-axis re-test red mechanically.
   - The 423 held-out units are recorded in prose.
   - W in the method name makes a re-test at another W red R-RETEST-REF.
   - Every example record in A8 sits inside a code fence [b-R3-N1].
2. **One bin count, for the NULL-RECORD, the RETEST-RECORD and the census's own grid** (which drops the partial last
   bin): `bins(d, W_ms) = round(d * 1_000_000) // (W_ms * 1000)`, per file.
3. **Null ticks**, from the Stage 0 manifest's HO5-HO8 `duration_s` (2,399.942 s in total):

| W | null ticks | 10x (PASS bar) | 11x (stop target) |
|---|---|---|---|
| 5 ms | 479,987 | 4,799,870 | 5,279,857 |
| 10 ms | 239,993 | 2,399,930 | 2,639,923 |
| 20 ms | 119,995 | 1,199,950 | 1,319,945 |
| 50 ms | 47,997 | 479,970 | 527,967 |

4. **The re-test verdict rule** follows Stage 1's A3.1/A2.2 rule with the well as the subject. It keeps Stage 1's
   per-subject aggregation, the median, and it differs only in the binomial-matched k, which is disclosed. It is
   pinned now, so the ONE final re-test's bar is never chosen after a held-out FAIL is seen [b-R3-B2, a-V3-2, b-R4-F5]:
   - **p0:** nominally 0.05, since a subject passes the null "with probability about 0.05" (evidence L145-146) and
     E = MI - q95 of 200 nulls (L469). A2.3 F1 (L379-382) measured 6-9%, but on the run null that A6.2 RETIRED
     [a-V5-2]. The census that reads 001872 uses the A6.3 candidate's null, whose rate is not yet measured. So:
     - **p0 range = [0.05, max(0.09, p0_hi)] in steps of 0.01.**
     - p0_hi is the upper 95% Clopper-Pearson limit of the candidate's `E>0` rate over every seed of the frozen A6.3
       battery's must-FAIL (no spatial order) rows.
     - That battery is synthetic and run once, before any held-out read, so k is still fixed before any FAIL is
       seen.
     - The table below is the floor, at p0_hi <= 0.09.
   - **Per file:** the frozen census reads each accepted file with the same endpoint, statistic, bar and control as
     the held-out read. It yields `E` and one of `informative`, `VOID` or `unreadable`.
   - **Per well (24 wells; the independent unit):**
     - NON-INFORMATIVE iff none of its accepted files is informative. A well with no accepted file is also
       non-informative, but it is counted separately in the disclosure [a N-5].
     - Otherwise its class is `E>0` iff the **median** E over its informative files is > 0 (Stage 1's own aggregation,
       evidence L66-68).
     - A median > 0 needs at least one file with E > 0. So the per-well null rate is at most
       `p_well = 1 - (1 - p0)^m_max`, where `m_max` is the largest number of informative files in any one well. The
       bound is exact for ANY-file under independence, so it is conservative for the median.
   - **BLIND** iff 12 or more of the 24 wells are non-informative (half, which Stage 1 refuses at), OR no well has a
     non-vacuous R3 (A4.5).
   - **Otherwise the denominator is fixed at n = 24,** with non-informative wells counted as not passing.
     - **Charges** iff `#wells(E>0) >= k`. k is the LARGEST, over the p0 range, of the smallest integer with
       `P(Binomial(24, p_well(p0)) >= k) <= P(Binomial(4, p0) >= 3)`.
     - So at every p0 in the range, the re-test's false-positive tail is at or below Stage 1's own bar.
     - Else `null`.
     - Pinned k:

| m_max | matched k at p0 = .05/.06/.07/.08/.09 | **pinned k** | worst tail / Stage 1 tail |
|---|---|---|---|
| 1 | 7/7/7/8/8 | **8** | 0.31 |
| 2 | 9/10/10/10/11 | **11** | 0.39 |
| 3 | 11/12/12/13/13 | **13** | 0.67 |
| 4 | 12/13/14/15/15 | **15** | 0.61 |

     - The 44-file prefix gives m_max = 2, so k = 11 at the floor. A p0_hi above 0.09 raises k by the same formula.
     - **Disclosed:**
       - m_max is fixed by post-read informativeness (a readability count, never an E sign), so k depends on the data
         but never on E;
       - the median rule has lower power than an ANY-file rule at the same k. The faithful rule is kept anyway
         [a N5-1].
   - The RETEST-RECORD's `ticks` = the sum of `bins()` over every accepted file at W.

### A8.3 Selection, acceptance and the bar

- **Re-test duration:** `d_i = n_samples / rate` of the file's single ElectricalSeries. The og1b manifest records it
  as `duration_s`.
  - This gives the same bins as `(n + 0.5)/rate`, because rate x W is an integer number of samples at every grid W.
- **PASS bar:** `sum_i bins(d_i, W) >= 10 x null ticks` at EVERY W in {5, 10, 20, 50} ms.
- **Stop target:** at least 11x at every W.
- **The order is content-blind; acceptance is gated, and the gates read content.**
- **The order:**
```python
rng = random.Random(20261023)
by_well = {well: {age: asset}}        # the 96 role:retest `sub-sample-well*` assets; age = wasAttributedTo[0].age.value
wells = sorted(by_well); rng.shuffle(wells)
ages = ["P52D", "P57D", "P71D", "P75D"]; rng.shuffle(ages)     # identity order at this seed (disclosed)
order = [by_well[w][ages[(i + r) % 4]] for r in range(4) for i, w in enumerate(wells)]
```
  - First 8:
    1. well014 P52D `ses-20260622T185903`
    2. well009 P57D `T202514`
    3. well012 P71D `T225444`
    4. well020 P75D `20260623T005311`
    5. well016 P52D `T185903`
    6. well002 P57D `T195034`
    7. well018 P71D `T231116`
    8. well022 P75D `20260623T005311`
  - Planning estimate: 11x at file 44 (26,401.8 s, 121.0 GB, 24 wells).
- **Each file's first completed verdict binds, per pin.** Only a run with no verdict is re-run. Every run is logged,
  and gate (m) reds on two completed runs of one file under one pin.
- **A pin change re-runs the WHOLE processed prefix (accepted AND refused files) under the new pin,** after C0, the
  control and the bridge re-run under it [a-V3-4].
- **Disclosed with the result:**
  - accepted and refused counts per age block and per gate, against bytes per second;
  - the G3 refusal rate against channel count [a N-4].

### A8.4 Pipeline and code placement

```
main venv, organoid_ingest (orchestrator; ordered driver for (m)):
  (f) tool pin  ->  (n) overlap rule by role  ->  download  ->  (b)=(g) stream_sha256 + size  ->  (a) licence
  ->  (l) endpoint and format: no stimulation; exactly ONE ElectricalSeries with a rate; record its HDF5 filters; refuse an
      unsupported one before the sorter  ->  read rate, n_samples, map (attributes + electrodes table only)
  ->  subprocess: $OG1_SORT_VENV/python tools/og1_sort.py sort <raw> <out>        [COMPUTE]
       G3 control on this map  ->  Kilosort4 (pinned params, drift correction OFF)  ->  motion estimate
       ->  quality metrics  ->  unit locations  ->  hand-off artifact + peak VRAM
  ->  VERIFY (h, i, j, k) on the recorded numbers, plus an EXACT RECOMPUTE from spike_samples (below)
  ->  (c) index  ->  (d) span, seconds passed to gate_time_units  ->  (e'') round-trip
  ->  write og1b_* to data/organoid_0b/ (retest) or og1bb_* to data/organoid_bridge/ (bridge)
```

- **Pinned definitions.** The tool computes these in numpy, and VERIFY recomputes them. They are compared with
  exact equality, where two NaNs count as equal (`numpy.array_equal(a, b, equal_nan=True)`). A disagreement refuses
  the file; a NaN alone never does [a-V5-1]:
  - `spike_samples` is sorted ascending per unit. An unsorted unit is refused by (c) [a-V4-3].
  - `isi_violation_fraction = count(diff(spike_samples) < round(0.0015 * rate)) / (n_spikes - 1)`, with
    **NaN when n_spikes < 2**. A NaN fails that unit only under (i), and is never a crash [a-V4-3, b-R4-F6]. This is
    JHU's criterion, the fraction of ISIs under 1.5 ms. It is NOT SpikeInterface's `isi_violations_ratio`.
  - `firing_rate_hz = n_spikes / (n_samples / rate)`, evaluated in exactly that expression order. The floats in
    `sort.json` round-trip through Python `json` repr.
- **Hand-off artifact** (`sort.npz` + `sort.json`):
  - per unit: `spike_samples` int64, `unit_id`, `x_um`, `y_um`, `isi_violation_fraction`, `firing_rate_hz`, `snr`,
    `amplitude_cutoff`, `presence_ratio`;
  - per file: `rate_hz`, `n_samples`, `motion_peak_um`, the G3 metrics, the tool pin hash and peak VRAM.
- **The `organoid_ingest` sorted path** shares licence, `stream_sha256`, the index checks and the manifest writer. It
  adds: a pin path, a role and an attribution parameter; 10 kHz `TIME_READINGS`; the ordered driver.
  - (e') and `gate_reproduce` are unchanged for 001603.
  - hdf5plugin goes into the sort lock.
- **`tools/og1_sort.py`** is the ONE new script (D6), with subcommands `sort`, `c0` and `control`.
- **Tridesclous2** (MIT, CPU, in the lock) is a disclosure cross-check on the control and the bridge only.

### A8.5 Calibration C0 and the sorter control

**C0 (synthetic only) comes first.** Its outputs go into ONE dated addendum to A8, reviewed by the three-lens panel
and **approved by the owner in chat** before any raw byte (bridge included). **The rules are fixed here, so C0
measures values but chooses none.**
- **`batch_size`:** the largest of {4 s, 2 s, 1 s} of samples whose peak VRAM is at most (card total - 1 GB), with
  the card otherwise idle under the `resource:gpu` claim. If none fits, CONTROL FAIL [a N-2].
- **Drift:**
  - doses {0, 2.5, 5, 10, 20} um, plus 40 um only if all up to 20 pass;
  - **bound = the largest d such that EVERY dose <= d passes bars 1-5**;
  - (j) accepts iff the measured peak is at or below bound - 2.5 um;
  - if bound - 2.5 <= 0, CONTROL FAIL;
  - the estimator (SpikeInterface `estimate_motion`, method and params pinned, run in x and in y) must recover every
    planted dose within 2.5 um, or CONTROL FAIL.
  - R4, "the dose above the bound fails", is true by definition, so it is **reported, never counted as evidence**.
  - The independent drift red is (j)'s planted 2x-bound file [a N-3].
- **Bar 3b, overmerge:**
  - **The 20 close pairs are planted rate-matched,** with a firing-rate ratio of at most 2 within each pair, and
    **17.5-35 um apart on G1** (one to two pitches), drawn in the GT design. SpikeInterface then flags every merged
    pair, because each member's agreement is at least 1/3 > 0.2 [a-V3-1, b-R3-B3, b-R4-N5].
  - N = 5 (25% of the planted close pairs).
  - **3b counts ALL sorter units,** not only quality-passing ones.
  - **3b binds on G1 only.** On G2 (electrodes at least 35 um apart), pairs closer than 35 um can share an electrode,
    so a correct sorter may over-merge them. G2's 3b count is reported.
  - **R2's margin: R2 >= N + 1 = 6 overmerged.**
  - R5's margin: the shuffled-position median error is at least 3x bar 5.
  - A missed margin is CONTROL FAIL.
- **Presence-ratio trigger:** if more than 10% of GT units with SNR >= 5 fail presence ratio > 0.8, that threshold
  is dropped. It is the only threshold C0 may drop.
- **Rejected addendum:** if the panel or the owner rejects the C0 addendum, the result is CONTROL FAIL, and a
  redesign needs a new amendment [b-R3-N4].

**Ground truth:** SpikeInterface `generate_ground_truth_recording`.
- Geometries:
  - G1 dense: 32 x 32 at 17.5 um;
  - G2 sparse: 1,020 electrodes, at least 35 um apart, seeded;
  - G3: each file's own map.
- GT units: 100 on G1; `max(100, round(300 x n_channels / 1,020))` on G2/G3.
- Rates: 10 and 20 kHz. Noise: 2.2 uV rms, and 2x. SNR 3-20.
- Organoid-shaped bursting trains: shared-envelope bursts (20-50x for 100-500 ms, every 2-20 s), background
  0.05-5 Hz, 1.5 ms refractory. 600 s each.
- G3 runs one arm per file: 1x noise, no drift.

**Metrics:** `compare_sorter_to_ground_truth` at its 0.105.3 defaults, **with `exhaustive_gt=True`** (the GT is
synthetic, so it is exhaustive). This is the one non-default, and is disclosed. Without it, SpikeInterface asserts
before counting false, redundant or overmerged units (`paircomparisons.py:562, 608, 649`) [b-R4-B1].

**Definitions:**
- **SNR** = `compute_snrs` at the arm's noise.
- **"Median accuracy"** is over ALL GT units with SNR >= 5, with 0 for an unmatched unit.

**PASS bars, on GT units with SNR >= 5, no drift:**
1. At least 80% well-detected on G1 and G2 at 1x noise; at least 70% at 2x.
2. Median accuracy >= 0.85, and median precision >= 0.9.
3. Of the quality-passing units, at most 10% false or redundant. Zero quality-passing units = FAIL.
   - 3b: at most 5 overmerged (G1).
4. Burst-window recall no more than 0.1 below whole-recording recall.
5. Median localization error at most 1.0 x the map's median nearest-electrode spacing.
6. Each file's G3 passes bars 1-5 (3b excluded) at 1x noise. A failure refuses that file under (h).

**Reds (each must FAIL its named bar):**
- R1: +1.0 ms shift reds bar 1.
- R2: rate-matched pairs merged red bar 3b, with at least 6.
- R3: one unit per channel reds bars 1 and 3.
- R5: shuffled positions red bar 5, by at least 3x.
- R6: burst-window spikes deleted red bar 4.
- (j)'s red: a planted 2x-bound file is refused.

**Anti-vacuity:** the GT compared with itself PASSES every bar.

**Where it runs:**
- The COMPUTE halves and the reds run in the sort venv. The results go to the tracked artifact
  `tools/og1_sort_control.json`.
- The battery does not re-run them (GPU-hours); this is disclosed.
- run_selfchecks VERIFY checks the artifact against every bar, checks each red's recorded FAIL, and checks the pin
  hash against (f). Its red is a planted below-bar artifact.

### A8.6 The real-data bridge (HO1/HO4, `role: bridge`, DEV), run before the push

- **Run:** the pinned pipeline on each raw file, then `compare_two_sorters(depositor, ours, delta_time=0.4)`.
- **Floor:** at least 50% of depositor units with SNR >= 5 matched at agreement >= 0.5 (chosen, not derived).
- **Red:** an ISI shuffle of our units must fall under the floor.
- **Outputs:** `data/organoid_bridge/og1bb_*`, with their own manifest. They are UNSEALED (DEV) and read by
  `organoid_ingest.load_bridge()` [a-V3-5, b-R3-F6].
- **Sorter-dependence typing (binding):**
  - **When:** after SOURCE PASS, and before the held-out campaign.
  - **How:** the frozen candidate reads HO1/HO4 on both unit sets, and each read gets a per-subject class (`E>0` /
    `E<=0` / non-informative).
  - **Mismatch:** any class difference types the source **SORTER-DEPENDENT**. With 2 subjects, a mismatch is
    reachable by chance; that errs conservative, and is disclosed.
  - **Owner rules before the held-out campaign:**
    - **"yes":** the campaign proceeds, and every RETEST-RECORD's prose carries "sorter-dependent".
    - **"no":** the lab route, then NO SOURCE. This strands the 48-97 h campaign; A7.1 forces that order, and this
      is disclosed.
  - The MaxOne-to-MaxTwo transfer is assumed, and disclosed.

### A8.7 Gates

| Gate | Passes when | Red runs in |
|---|---|---|
| (a) licence | `license_ok` | run_selfchecks (existing) |
| (b)=(g) bytes | `stream_sha256` + size equal the pin, before any reader | run_selfchecks (existing hasher) |
| (c) index | non-finite or negative sample indices refused | run_selfchecks |
| (d) span (sorted kind) | every spike is in [0, n_samples); seconds passed to `gate_time_units`; 10/20 kHz | run_selfchecks |
| (e'') round-trip (sorted kind) | the npz rebuilds the tool's spikes exactly; red: one spike moved by one sample | run_selfchecks |
| (f) tool pin | the `og1_sort.py` sha, the lock sha, normalised `pip freeze --all` (`name==ver`) == the lock, the params sha, and the control artifact's pin hash all agree | run_selfchecks (planted mismatches) |
| (h) control bound | the control artifact passes, and this file's G3 passed | VERIFY on the tracked artifact |
| (i) unit quality | per unit: `isi_violation_fraction` <= 0.01, FR >= 0.05 Hz, SNR >= 5, amp cutoff < 0.1, presence ratio > 0.8 (unless C0 drops it). A per-unit non-finite metric fails that unit only. At least 1 unit kept; drop counts recorded | VERIFY + exact recompute (a planted disagreement must be refused; a planted 1-spike unit must give "unit dropped, file kept") |
| (j) drift | the recorded peak is at or below bound - 2.5 um | VERIFY (2x-bound red) |
| (k) positions | finite (x, y) inside the routed hull, from the pinned method | VERIFY |
| (l) endpoint and format | no stimulation; one ElectricalSeries with a rate; supported filters | run_selfchecks (HDF5 fixtures) |
| (m) selection (retest) | exactly the section 3 order prefix; no two completed runs per file per pin; refusals logged | run_selfchecks |
| (n) overlap | retest: no shared sha, session or subject with the 24 Stage 0 assets (pre-download). bridge: its subject MUST equal its paired Stage 0 DEV sorted asset's subject | run_selfchecks |
| seal | no file in `src/`, `tools/`, `tests/` or `run_selfchecks.py`, other than `organoid_ingest` itself, contains `organoid_0b/` or `og1b_`. The check lives in organoid_ingest's self-check and excludes its own module | run_selfchecks (static) |

### A8.8 HELD-OUT seal

- **Separation:** the retest outputs go only to `data/organoid_0b/` (prefix `og1b_`, their own
  `og1b_sessions.json`). The bridge goes to `data/organoid_bridge/` (`og1bb_`).
- **`load_0b()` refuses unless** a new public `null_scale_gate.nulls_for(path, method_re)`, restricted to
  `og1-organoid-stream.md`, returns an `og1_succession_w*ms` NULL-RECORD whose `retest=` is EITHER:
  - `PENDING since <sha>`, with the sha on or after A8's landing commit. This uses git ancestry, which fails closed
    without git history (disclosed) [b-R3-N5]; OR
  - `REF:` to an existing OG1 RETEST-RECORD, so the data can be re-read for audit, reproduction or Stage 2 [b-R3-F2,
    a-V3-5].
  - Red: no such record, refused.
- **Disclosed holes:** a string built at runtime, and a person running `np.load` by hand, are not caught by any
  gate. This text forbids both.
- **Only counts leave the seal:** the npz, the manifest, and the per-gate drop and acceptance counts.

### A8.9 Resources and claims

- **Disk:** 746 GB free. Raw kept is 121-175 GB (up to 249 GB if refusals push toward all 96). Scratch is about 50 GB
  per file, deleted after each sort.
- **GPU:** 12,227 MiB. `batch_size` is set by C0's rule. OOM is an INSTRUMENT REFUSAL.
- **Claims:**
  - Each file claims `resource:gpu` and `resource:timing` with `--ttl 8h`, renewed, and releases both between files.
  - It never starts under another session's battery or CUDA claim.
- **Time (T4-class anchors):**
  - C0 + control: 13-28 h;
  - with the bridge: 17-36 h;
  - the campaign: 48-97 h;
  - **each pin change: up to another 17-36 h plus 48-97 h** [a-V3-4].

### A8.10 The lab route (D4)

**The ASK:**
- at least 40 distinct organoids, at least 600 s each, spontaneous only;
- NWB 2.x;
- sorted units with positions, plus sorter, params and curation; or raw with geometry;
- HD-MEA, preferably MaxWell;
- the lab named;
- CC0, CC-BY-4.0 or written permission.

**How a census could read it:**
- **(i) Owner-run:** the frozen pushed pipeline runs on the owner's machine outside any Claude session; only the
  RETEST-RECORD fields and the verdict come back.
- **(ii) Licensed:** an ordinary pinned ingest from an independent lab.
- **(iii) Interpreted only:** not usable as the re-test.

### A8.11 Outcomes (exhaustive; each writes a dated evidence line)

| State | When | Consequence |
|---|---|---|
| NOT BUILT (candidate) | the A6.3 candidate fails its one run | 0b is not pushed; the branches are kept |
| CONTROL FAIL | C0 or the control misses a bar or a margin; (j) cannot be built; bound - 2.5 <= 0; no batch_size fits; the C0 addendum is rejected (all pre-push). **The expected main risk is bar 3b:** 20 rate-matched pairs 17.5-35 um apart that share the burst envelope are hard for any sorter, so "at most 5 overmerged" may fail a sound sorter. That is a fair, pre-registered failure [a N5-2] | not pushed; lab route only |
| BRIDGE FAIL | under the floor, non-finite, or the post-freeze subject pairing fails (pre-push) | not pushed |
| PUSHED | reviewed, ONE battery green, pushed | the campaign may start |
| INSTRUMENT REFUSAL (campaign) | OOM; pin drift; a claim that never frees; a non-finite PER-FILE value; a tool crash | HALT; the prefix is not consumed. A fix that changes ANY pin re-runs C0, the control and the bridge, then the WHOLE processed prefix, under the new pin |
| CONTROL FAIL (G3) | 10 consecutive G3 refusals (chosen: at a true per-file rate <= 20%, a given run of 10 has probability 1.0e-7, about 1e-5 over the campaign). G3 refusals are logged under (m) and consume the prefix | halt; recorded apart from SOURCE FAIL; lab route |
| SOURCE FAIL | all 96 processed and some W under 10x | per-gate counts; lab route |
| SOURCE PASS | 10x at every W; all gates green | A7.1's hold lifts for the census code; then the bridge typing |
| SORTER-DEPENDENT | the bridge typing finds a mismatch | the owner rules: "yes" proceeds with the typing in the prose; "no" goes to the lab route, then NO SOURCE |
| SOURCE UNUSED | Stage 1 PASSES on held-out (only then) [b-R3-F5] | 001872 is never read by Stage 1; it retires with organoid_ingest at Stage 2 unless Stage 2 consumes it |
| SOURCE HELD | Stage 1 is INSTRUMENT REFUSAL or BLIND on held-out | sealed until the dated amendment's re-run resolves (A4.9); then this table again |
| RETEST REFUSAL | the census crashes, or yields a non-finite E, on the 001872 read | halt. **The fix is bounded, pinned now** [a-V4-2, b-R4-F3]: it may touch only the crashing or non-finite code path, never the endpoint, statistic, bar, controls or aggregation. The fixed census must reproduce the held-out per-subject E and verdict byte-exactly before the re-read. The fix diff is reviewed by the panel and by a lane that has not seen the crash's data. **ONE re-read only;** a second REFUSAL, or one unresolved at the arc's closing record, is RETEST BLIND |
| RETEST BLIND | 12 or more wells non-informative, or no non-vacuous R3 (census controls such as PC1/PC2 failing count here) | the 001872 read happens while the record is `PENDING` (so `load_0b` opens). The record is then REWRITTEN per **D7** in the same arc, and the rows take D7's disposition. `load_0b` refuses after STRUCTURAL until a lab `REF:` re-admits; any audit re-read of the BLIND data happens before the rewrite [b-R4-N4]. Lab route |
| RETEST null | the rule in section 2.4 reads `null` | RETEST-RECORD `result=null`; then the A4.9 arbiter branch, read from the arbiter's RUN and dated RESULT on the subjects its amendment pins [a-V4-4, b-R4-F2]. **Arbiter PASS:** `og1_succession` is deleted (P8 deletion via amendment item 2) and an ENCODING negative is recorded. `organoid_ingest` (both kinds) and `tools/og1_sort.py` stay, and its row moves to `permanent` (awaiting-prerequisite: a successor encoding's pre-registration), re-dated [b-R4-F1]. **Arbiter REFUSAL or BLIND:** `og1_succession` is deleted (item 2 still applies), and the ingest stays as in the PASS branch (section 3 deletes the ingest only on an arbiter FAIL). **Arbiter FAIL:** `og1_succession`, `organoid_ingest` (both kinds) and `tools/og1_sort.py` are removed IN THE SAME ARC, together with every registration [b-R4-N1]: REGIONS (`__init__.py`), the g8 `_MOVED` pin, `run_selfchecks.py`, both dormancy rows, `tests/test_organoid_ingest.py`, the `pyproject.toml` h5py comment, the pins, the lock, the control JSON and `data/organoid_*`. The P4c baseline is down-ratcheted |
| RETEST charges | the rule reads `charges` | RETEST-RECORD `result=charges`. The census stands; both rows move to `wire` in the same commit [b-R4-F1]; the sorted kind stands until Stage 2 |
| NO SOURCE | 001872 fails and the lab route fails | the A7.1 hold stands |

**The lab route keeps the sorter alive** [b-R4-F4]. On SOURCE FAIL, CONTROL FAIL (G3) or a SORTER-DEPENDENT "no", the
sorted kind and `tools/og1_sort.py` STAY while a raw lab route (section 10) is open. They are removed only once the
lab declines, or delivers sorted units. While they stay, the row is re-dated in every arc, and the owner is asked
again after 3 arcs with no lab delivery [a N5-4].

**The arc that writes `PENDING` also lands** the arbiter amendment, the arbiter's RUN and its dated RESULT, the
NULL-RECORD, the 001872 census read, and the RETEST-RECORD (or D7's rewrite). So no PENDING record crosses a closing
record. **Rebase hazard (disclosed)** [b-R4-N3]: if another arc's closing record reaches main first, rebasing onto it
fires R-EXPIRED. The cure is to re-set `since` to the new base, which is legal because that tree lacks the record.
Until then, a RETEST REFUSAL fix holds the arc open.

### A8.12 Build rhythm and the P3 row

**Lanes, in parallel on disjoint files:**
- L-sortdrv: `tools/og1_sort.py` (claimed) and the lock;
- L-sortctl: delivers the `c0` and `control` subcommands as a patch, plus `tools/og1_sort_control.json`;
- L-ingest0b: `organoid_ingest`'s sorted path, the VERIFY halves and recompute, `load_0b`/`load_bridge`, the gates and
  reds, and `null_scale_gate.nulls_for`.

**Order:** reviewers as results land; C0 and its addendum (panel plus owner); then the control and the bridge on the
lane heads; then panel, fold, Opus closing review, ONE battery (including D6's baseline edit), the closing record.
The push is gated on D1.

**P3, the two rows.** The `organoid_ingest` EXEMPT row (`dormancy_gate.py:761-775`) exists today. The census's own
row for `og1_succession` does not exist until the census lands in `src/`; it is created then, and takes
`scale-retest` on a readable FAIL (A4.9) [b-R5]. Each row holds one disposition at a time: `organoid_ingest` is
`wire` today, and `scale-retest` after a readable census FAIL. Both rows then move, in the same commit as the record
change, to:
- `wire` on RETEST charges;
- `permanent` (awaiting-prerequisite) on RETEST null with an arbiter PASS, REFUSAL or BLIND, or on D7(a);
- deletion with the module on an arbiter FAIL.

The 0b text appended to its reason:
- "Stage 0b's sorted kind (A8) is 001872's re-test reader.
- SOURCE UNUSED: it retires with organoid_ingest at Stage 2 unless Stage 2 consumes it.
- SOURCE FAIL, CONTROL FAIL (G3), or a 'no' on SORTER-DEPENDENT: it stays while a raw lab route is open, and is then
  removed in the next arc, re-dated, with the P4c baseline down-ratcheted.
- RETEST BLIND: D7.
- RETEST null: the A8 section 11 arbiter branch, in the same arc.
- RETEST charges: it stands until Stage 2."

**Estimates, not commitments:** build about a day of lane time; GPU 17-36 h pre-push, plus 48-97 h for the campaign.

**What A8 read:** registry, package and paper metadata (via the lanes); the 001872 asset listing's paths,
ages, sizes and start/end times; our Stage 0 manifest's held-out durations. No data file, no NWB header, no lab file.

## Erratum E1, 2026-10-23 (PROCESS ONLY: wording; append-only; from the Stage 0 closing review, board #4822 F3)

A6.4's second item and A7.4 ask "whether an ENVELOPE outcome may be typed as a non-null result". Read alone, that
can sound like a positive. The judge's term (#4729) is meant: an ENVELOPE outcome would be recorded as a
**no-NULL-RECORD outcome**, never as succession and never as a positive. A6.3 item 5 stands: an envelope read is never
cited as succession, for E4 or Stage 2, under any typing. The open owner question is only whether such an outcome
may close Stage 1 without a NULL-RECORD.

**What E1 read:** nothing from organoid data.

## Amendment A9, 2026-10-23 (PROCESS ONLY: the Stage 1 outcome; append-only)

**Result.** The frozen acceptance battery ran ONCE on the ONE candidate. The overall result is **FAIL** (board #4854).
- **The battery:** lane og1-acc, which authored no null; `ACCEPTANCE.md` cabb7d9b..., `battery.py` d9a8a38b...,
  frozen at #4761 before any candidate was measured.
- **The candidate:** og1-census5 round 8, `rows/candidate.py` 51f622ae..., frozen at #4830 and copied by the integrator
  at #4843.
- **The run:** `RESULT.json` 897c3b93...; seeds 1000-1009; 0 errors; wall time 620 s. Scored task OG1-STAGE1 total=1.
- **must-PASS** (spatial and readable, at least 8/10): P-i 10, P-k 10, P-b4 10, **P-b25 7 (MISS)**.
- **must-FAIL** (at most 1/10): all 21 met. These include:
  - the envelope rows at W = 5, 10 and 20 ms;
  - the composition row (envgrad);
  - drift d and e;
  - semi-Markov, i.i.d., Poisson and random bursts;
  - every twin.
- **FLOOR** R-b10: 0/10. **INFO** S-b4 4, S-k 0, S-b25 1.

**Consequences, as pre-registered:**
1. **Stage 1 is NOT BUILT** (A6.3 item 4).
   - Stage 1 read no organoid byte, so there is no NULL-RECORD.
   - There is no re-run, and no second candidate under A6.
2. **Stage 0b is "NOT BUILT (candidate)"** (A8 section 11).
   - It is not pushed.
   - The lane branches `claude/og1-0b-{drv,ctl,ing}-1023` are kept at their final heads, recorded on the board.
   - A8 stays frozen as the plan for any future Stage 1 pre-registration that needs a 10x source.
3. **`organoid_ingest`'s dormancy row takes its pre-registered branch.** It stands as the ingest of reference data with
   no live consumer, re-dated, and the owner decides its future.

**Diagnosis** (A6.3 item 4; with #4682, #4692, #4713 and #4729):
- **The earlier designs:** the four retired designs read envelope or composition as order.
- **This candidate's specificity:** it is an envelope-preserving conditional surrogate, and it is the first design
  to reject every red row, those included.
- **Where it misses:** sensitivity, at the weakest planted dose. P-b25 (f = 0.25) is og1-acc's disclosed
  re-implementation of #4722 from its text, and the candidate scored 7/10 there. The reference within-burst
  permutation null also scored 7/10 there (REFERENCE-RUNS.md).
- This is a description, never a licence to re-run: A6.3 allows ONE run.

**What A9 read:** no organoid data. It read the battery and its result, on synthetic data only.

## Amendment A10, 2026-10-23 (PROCESS ONLY: the owner's ruling on organoid_ingest's future; append-only)

**Owner ruling, verbatim:** "Keep organoid_ingest as a reference-data ingest for a possible new Stage 1
pre-registration; its dormancy row stands, re-dated."

- `sensorimotor/organoid_ingest` stays in the tree, off every live path, with its pin, tests and self-check.
- Its EXEMPT `wire` row in `gates/dormancy_gate.py` is re-dated 2026-10-23 and quotes this ruling.
- **A new Stage 1 needs a new dated pre-registration.** Nothing in A1-A9 authorizes another candidate or another
  battery run. That pre-registration needs a "Sample size (P1)" section (the owner's rule, board #4876). If it needs
  a 10x re-test source, it starts from A8 (frozen; Stage 0b NOT BUILT; branches kept).
- **Integrator addition, disclosed:** the row carries a review trigger. If no new Stage 1 pre-registration is frozen
  within 3 pushed arcs, the owner is asked again whether to keep the ingest or retire it by deletion. This is a P3
  safeguard against a kept row standing unexamined; it does not change the ruling.

**What A10 read:** nothing from organoid data.
