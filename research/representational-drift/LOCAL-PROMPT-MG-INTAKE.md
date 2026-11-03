# Prompt for a LOCAL SNDL session: Marks & Goard (2021) intake for D0 Part A

Paste everything below the line into a Claude Code session opened on the owner's Windows host, in
`D:\AI-project\spiking-neural-data-lake`. It is written for that session; it decides nothing beyond the intake below.

---

You are a local SNDL session on the owner's Windows host. The owner (Luan Dinh) asks you to process the Marks & Goard
(2021) Dryad deposit, which the cloud session could not download (Dryad's AWS WAF CAPTCHA blocks cloud clients). The
owner downloaded it in a browser to:

`D:\AI-project\spiking-neural-data-lake\research\Main_experiments.zip`

## 0. Before anything: the SNDL protocol

1. Read `docs/AGENT_COLD_START.md` first, then `AGENTS.md`. `docs/POLICIES.md` governs every step.
2. **Rules, in force for this task:**
   - never push `main` or `claude/*`;
   - no force-push, no branch deletion;
   - one change per commit, each ending with your own session's `Co-Authored-By` and `Claude-Session` trailer lines;
   - never touch the five `brainstem/` envelope files;
   - no D1/L1b-STALE build, no OrganoidSciences data, no Codex in any form;
   - **never write to the live lake** (`brain_lake.bin`): this is measurement data from a microscope, never brain
     memory;
   - never run `policy_manifest --certify`;
   - this host is memory-starved: one heavy run at a time, and respect `HEAVY-RUN.lock` if it exists.
3. **Python:** use `.venv-lake/Scripts/python.exe` with `PYTHONPATH` set to the repo's `src` (as in `docs/AGENT_COLD_START.md`).
4. **Gate commits with `&&`, never `;`,** so a red gate stops the commit. The doc gates are:
   - `python -m snn_data_lake.gates.doc_budget_gate --assert`
   - `python tools/doc_command_guard.py --assert`
   - `python tools/citation_guard.py`
   - `python -m snn_data_lake.gates.null_scale_gate --assert`
   - `python tools/red_reachability.py --assert`
   - `python tools/p4c_ratchet.py --assert`

   Run `ruff` too if you touch Python.

## 1. Context to read (from git)

`git fetch origin cloud/drift-d0-instrument-1102`, then work on a NEW branch off it: `local/d0-mg-intake-1103`. Read
these on that branch:
- `research/representational-drift/MAP.md`, section 4 D0 (on main): the instrument's purpose and the rule that it is
  validated on mouse data before it reads the engine;
- `src/snn_data_lake/instruments/representational_drift.py`: the instrument you must use, with no second copy of its
  functions (P2);
- `research/representational-drift/PARTA-DEV-2026-11-02.md`: how the Allen DEV reading was done (range reads, a
  DEV/sealed split fixed before any trace, derived arrays never in the repo);
- `specs/022-unified-brain-arc/evidence/d0-partA-heldout-prereg.md`: the Allen pre-registration. Marks & Goard is the
  missing gratings-versus-movie comparison (HW5 RQ1a; MAP D0's "reproduce Marks & Goard's gratings-vs-movies
  direction").

## 2. Verify the file and keep it out of git

1. **Expected size:** 14,538,685,882 bytes. **Expected sha256** (from the Dryad API):
   `a561b7a4bead9afa5e7712dbcc50fa8be3f4065fd54596ef6eb37635401cd994`. Check it with
   `Get-FileHash -Algorithm SHA256`. If it differs, stop and report: do not process a different file.
2. `git check-ignore -v research/Main_experiments.zip` must print `.gitignore:...:research/*.zip`. Never `git add` it,
   and never copy it into the lake.
3. **Licence:** the Dryad record says CC0-1.0, and `sensorimotor/ingest_guard.license_ok("cc0-1.0")` must return True.
4. **Do not unzip the whole archive** (14.5 GB). Read members with Python's `zipfile`, one member at a time.

## 3. Stage 0: the manifest (structure only, no traces read)

1. List every member: name, compressed and uncompressed size, CRC. Read any README or documentation member in full.
2. **Work out from the files and the README:**
   - the formats (likely MATLAB `.mat`; v7.3 files are HDF5, so open them with `h5py`, and older ones with
     `scipy.io.loadmat`);
   - which mouse and field of view each file is;
   - the session dates or intervals;
   - how cells are matched across sessions;
   - where the drifting-grating and natural-movie responses are;
   - their trial, repeat and frame structure.
3. **Write `tools/d0_mg_manifest_pin.json`:**
   - the zip's sha256 and size;
   - the licence;
   - a member summary (names, sizes and roles, not data);
   - the per-mouse / per-field-of-view list with session count and span in days.

   Commit it alone.

## 4. Stage 1: the DEV / sealed split, before any trace is read

Fix a deterministic rule from the manifest alone: sort fields of view by mouse id, then field-of-view id, and take the
first 3 distinct mice as DEV; seal the rest. Write `tools/d0_mg_split_pin.json`: the rule, the DEV ids, the sealed ids,
and the sha256 of the manifest pin. Commit it alone. **No sealed trace is ever read in this task.**

## 5. Stage 2: the DEV reading only (descriptive)

1. **Extract only the DEV mice** into gitignored `data\drift\marks_goard\dev\` as small npz files:
   - natural movie: cells × repeats × time-bins;
   - gratings: cells × direction (or orientation) × repeats;
   - in both, only cells matched across sessions.

   Record the binning you chose and why (match the movie's frame structure; 1 s bins if the movie runs at 30 Hz, like the
   Allen reading).
2. **With the instrument (`representational_drift`), per DEV mouse and per stimulus type:**
   - the within-session split-half `pv_correlation` (the ceiling);
   - the between-session opposite-half `pv_correlation` for each session pair, with its days apart;
   - the median `tuning_correlation`;
   - for gratings, the per-cell `preferred_shift` (orientation, period 180) between sessions.
3. **The question,** descriptive only: is the movie's between-session correlation, relative to its ceiling, lower than
   the gratings', as Marks & Goard report? Report the numbers. Make no claim, because DEV is not a test.
4. **Run the instrument's reds first:** `python -m snn_data_lake.instruments.representational_drift` must exit 0.

## 6. Report

Write `research/representational-drift/MG-INTAKE-2026-11-03.md` (use the actual date). It must contain:
- what you verified (hash, ignore rule, licence);
- the manifest summary;
- the split rule;
- the DEV numbers in tables;
- anything that differs from the paper's description;
- **the exact code that ran, inline.** Do not add new `.py` files to the repo: `tools/p4c_ratchet.py` counts them.

Commit it, run the gates (with `&&`), and push `local/d0-mg-intake-1103`. Then give the owner a short summary to relay
to the cloud session: branch, heads, the three pins, and the DEV numbers.

**Not in scope:** reading any sealed Marks & Goard mouse, writing a pre-registration for them (the cloud session drafts
that from your manifest and DEV numbers), any change to `src/`, and anything in the lake.
