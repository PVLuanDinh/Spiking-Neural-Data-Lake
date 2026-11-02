# D0: the representational-drift instrument and the owner's HW5 plan, applied to the SNDL engine (DRAFT pre-registration)

**Status: DRAFT.**
- **Built:** the instrument and its constructed reds (`instruments/representational_drift.py`, branch
  `cloud/drift-d0-instrument-1102`).
- **Not built:** any data reader, any engine reading, any claim.
- **Author:** cloud session `session_01TeA2yYimjAiHiGFzw6hqHX`, at `origin/main` `c4c10b28`.
- **It decides nothing.** The integrator owns the row (if the owner makes one), the final pre-registration, every run,
  the panel and the battery. The policy-enforcer reviews it.

**The instruction it answers** (owner, 2026-11-02, verbatim): *"Review github and proceed with hw5 plan to apply for
SNDL"*.

**What "the HW5 plan" is.** The owner's course proposal, "HW5: Research idea refinement" (Luan Dinh). Its three
aims:
1. drift rate and stimulus dependence, with and without behavioural state;
2. whether drift avoids the directions a linear decoder uses;
3. how little re-alignment rescues a frozen decoder.

They are, nearly one for one, `research/representational-drift/MAP.md` section 4's **D0** ("the instrument, validated
on synthetic and mouse data") and the engine studies D2-D5 that read through it. This file binds the two.

## 1. What exists now

**`instruments/representational_drift.py`.** Pure measurement on arrays its caller supplies. It opens no lake, builds
no brain and is never constructed by a brain.

| function | what it reads | HW5 aim |
|---|---|---|
| `pv_correlation`, `drift_index` | population-vector correlation per stimulus, averaged | 1 |
| `tuning_correlation` | each neuron's response profile across stimuli, between sessions | 1 |
| `preferred_shift` | circular change of preferred orientation (the 0/180 wrap) | 1 (gratings) |
| `readout_null_split`, `random_readout_null_fractions` | the share of a population change invisible to a readout W, against random readouts of matched rank | 2 |
| `decoder_accuracy` | the closed-form readout (REUSED from `prior_drift_harness.fit_readout`, `DIRECT_SOLVE`), with a capacity guard that refuses an interpolating fit | 3 |
| `procrustes_realign`, `recovered` | the best orthogonal re-alignment of one session onto another | 3 |
| `read_drift` -> `DriftReading` | all of the above on one labelled pair of sessions (parity split; the re-alignment fitted only on calibration class means, never on judged trials) | 1-3 |

**The constructed reds, MET at `ddaaf233`.** These are MAP D0's list, each run before any claim. They are in `main()` and
in `tests/test_representational_drift.py`.
- **R1.** A/A reads exactly zero drift.
- **R2.** A planted orthogonal rotation is recovered at four doses (with more samples than neurons, so the rotation is
  identifiable), and drift grows with the dose.
- **R3.**
  - Drift confined to null(W) reads a null fraction of 1, and row-space drift reads 0 (both within 1e-12).
  - A matched-rank random readout reads (n - rank) / n within 0.02 on isotropic change.
  - A zero change is refused.
- **R4.** 170 -> 10 degrees reads +20 on the orientation circle.
- **R5.** A decoder with features + bias > training samples is refused (`CapacityRefusal`).
- **R6.** A non-orthogonal drift (half the neurons gain 50%) is NOT recovered.
- **End to end:** A/A reads null; a planted rotation degrades the frozen decoder, and the re-alignment raises it
  again.

## 2. Part A: validate on mouse data (the HW5 proposal, unchanged in substance)

**The rule MAP D0 sets, kept.** The instrument must reproduce two published directions on DEVELOPMENT mice before it
reads a sealed held-out set, and before it reads the engine at all:
- Marks & Goard (2021): natural-movie drift exceeds grating drift.
- Deitch, Rubin & Ziv (2021): population geometry is preserved while single cells drift.

**If it fails to reproduce them, the INSTRUMENT IS BLIND** (repair or delete). It is never "mice do not drift".

**Fixed scope** (the owner's Lab 6 reviewers: "make the scope smaller", "fix factors at a smaller scale"):
- V1 only;
- excitatory Cre lines;
- cells matched across all three Allen two-photon sessions;
- Natural Movie One (shown in every session);
- one confound: running speed and pupil size.

The Marks & Goard Dryad deposit supplies the gratings-versus-movie comparison.

**Behavioural control** (Sadeh & Clopath, 2022, eLife 11:e77907). Each cell's response is fitted as movie frame plus
running speed plus pupil size. Drift is read on the stimulus part alone, beside the raw reading. A reproduction that
holds only on the raw reading is reported as such.

**The null-space reading,** with Rule et al. (2020), eLife 9:e51121, as the prior. Drift is expected to be BIASED toward
null(W) but not CONFINED to it. So the claim is "fraction above the matched-rank random control", never "fraction near
1".

**Data route** (MAP D0, unchanged):
- 2-photon ΔF/F and the Dryad deposit go through a new `sensorimotor/ophys_ingest.py` (h5py, no pynwb, as OG1 did).
- Data lives in gitignored `data/drift/`, never the lake.
- Pins go in `tools/*_pin.json`.
- **Not built here:** the reader needs the data, and the data needs the licence decision below.

**Blocked in this session, MEASURED 2026-11-02.** The cloud session's egress proxy refused `api.dandiarchive.org`,
`datadryad.org` and `api.brain-map.org` (`connect_rejected`). Part A runs on the owner host, or here once those hosts are
allowed.

## 3. Part B: the same instrument on the engine (each attaches to an existing plan row; none is built here)

| HW5 aim | engine reading | populations (MAP section 1) | attaches to |
|---|---|---|---|
| 1, drift rate | `drift_index` and `tuning_correlation` on lake-stored probes re-presented across ticks and sleeps | prototypes (S1), turnover (S2), predictive coding (S4) | MAP D2 / D3 |
| 1, the behaviour control | the engine analogue of running and pupil: the brain's own state variables at each probe, regressed out the same way | the same | MAP D2 / D3, and a pre-registered list of state variables |
| 2, null space | `readout_null_split` of the population change against the incumbent readout's weights (`OnlineReadout` / the intent learner's rows), beside the random control | the reservoir and readout | G5 (T113) |
| 3, rescue | `procrustes_realign` across a growth step, a sleep bout and a fresh-process restart | the G9 reservoir | G5 Part E (the relational readout) and G9's criterion 1 (T117), both already folded from the owner's drift report |

**Engine controls** (none is the method):
- **The FROZEN twin's noise floor:** no growth, no sleep, same seed.
- **A planted drift that preserves geometry:** reservoir unit identities permuted. This is G5 Part E's PE-R1: the fixed
  readout must fail on it, and the relational one must pass.
- **The A/A pair.**

**Order.**
1. Part A first: Part B reads only through an instrument that is not BLIND.
2. Engine runs that change brain code wait for E1d. E1d may cite its calibration only while brain code outside the pc
   harness equals `1b6e53ab`.
3. MAP D5 and G9's criterion 1 wait for the G9 harness, which does not exist yet (`research/g9-critical-path-1029.md`,
   on `cloud/g9-critical-path-1029`).

**This instrument is not L1b-STALE.** T121 (MAP D1) is parked and is not built here. If that row is unparked, its
`readout_staleness` and this module's `decoder_accuracy` / `read_drift` must be reconciled under P2 before either lands.

## 4. Sample size (P1)

**Part A.**
- The paired unit is the mouse (or the field of view), for `instruments/paired_contrast.py`. That tool refuses
  n > 40 (`MITM_MAX_N`): MAP owner decision 4.
- There is no measured SD yet. The DEV pilot measures the per-mouse paired SD of the gratings-versus-movie drift
  difference, and the target n at power .8 follows from it, before any held-out mouse is read.
- The sequential machinery exists only in `instruments/rpe_charge.py`. Lifting it into a shared module is MAP owner
  decision 5 (a P2 merge).

**Part B.** Each engine study pre-registers its own seeds and SD in its own row.

## 5. Null disposition (P8, including the 2026-10-23 null-scale amendment)

- A Part A failure is ruled **before data** as one of two dispositions (MAP owner decision 2):
  - INSTRUMENT BLIND (repair or delete);
  - a null on an axis the null-scale gate admits (`units` or `ticks`).

  "More mice", "more sessions" and "more days" are not legal 10x axes.
- An engine null names its scale and waits for one 10x re-test before anything is removed. Owner-drawn P4d parts are
  never deleted on a null.

## 6. What this does not claim

- Nothing about the engine. No engine population has been read.
- Nothing about mice. No data has been read.
- That drift helps or harms anything. The instrument measures change; benefit claims belong to their rows.

## 7. Owner decisions this needs (MAP section 6, still open, plus one new)

1. Make this a tasks row, and give its red family a letter (MAP proposes `D-R*`).
2. The Part A null axis (section 5).
3. **Licences.** `ingest_guard.license_ok` is deny-by-default. The Allen Terms of Use are not on the allowlist (unless
   the DANDI copy carries an SPDX id), and the Dryad deposit is probably CC0. Both are unconfirmed.
4. Allow more than 40 paired units, or block the data.
5. Lift the sequential-design code out of `rpe_charge`.
6. The dependency route: extend h5py (recommended) or an AllenSDK subprocess venv.
7. **New:** allow `api.dandiarchive.org`, `datadryad.org` and `api.brain-map.org` in this cloud environment's network
   settings, or run Part A on the owner host.
