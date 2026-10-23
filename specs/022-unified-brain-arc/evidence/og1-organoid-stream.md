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
