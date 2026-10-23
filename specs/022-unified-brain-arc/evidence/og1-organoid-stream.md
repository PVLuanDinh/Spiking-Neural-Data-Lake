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
