# Docs-only battery: run the readers of the changed Markdown, not the whole suite

**Owner, 2026-10-02:** "add the docs-only battery rule"; design chosen in chat the same day: "Runtime reader map
(Recommended)", over a static grep version and dropping the idea. This file is the arc's pre-registration and record.

## 1. Protocol (written before the first run)

### 1.1 What is built

`src/snn_data_lake/gates/docs_only_battery.py`, one module:

- **The instrument.** It is an audit hook (`sys.addaudithook`). A `.pth` line installed in the gate venv activates it
  only when `SNDL_DOC_AUDIT` is set; without that variable it does nothing. While it is on, it records, per Python
  process, every `.md` file opened inside the repo. It also records every in-repo directory listing (`os.listdir`,
  `os.scandir`, `glob.glob`) and every `subprocess.Popen`. Each record is attributed to the pytest test file that
  was running: `PYTEST_CURRENT_TEST`, or `SNDL_DOC_AUDIT_TEST`, which the hook injects into a child's `env` dict at
  spawn time because a dict copied at import time carries a stale test id.
- **The map.** `tools/doc_reader_map.json` is built from one full pytest run under the instrument, during the
  arc's full battery. For each test file it lists the Markdown files read, the directories listed, and whether
  the file spawned `git ls-files`/`git grep`. Reads made at collection time, outside any test, go to `pytest_all`.
  The map also stores `code_hash`: a sha256 over `git ls-files -s`, excluding `*.md`, the record and the map.
- **The battery** (`--run OUTDIR`), for a diff `origin/main..HEAD`. The run refuses before any stage if:
  - any changed path is not `*.md` (refusal **R1**);
  - the map's `code_hash` differs from the tree's (refusal **R2**: the map describes other code);
  - the map is missing or has the wrong version.

  Otherwise it runs these stages, each with its own `rc` line:
  - `policy_manifest --assert` (halts on red);
  - `policy_manifest --certify`;
  - `run_selfchecks.py`, in full: it includes the 6 tools gates, and `--carry` stays barred from certification
    batteries;
  - the 16 battery gates with `--assert`;
  - `doc_budget_gate --assert`;
  - pytest on the SELECTED test files only, with `--no-cov`.

  A test file is selected when:
  - it read a changed file;
  - or the change ADDS or DELETES a `.md` and the test file listed that file's directory (or an ancestor, for a
    recursive glob), or enumerated tracked files with git.

  A collection-time read of a changed file runs the FULL pytest, with coverage. Last, the run writes a `battery`
  block into `tools/policy_certification.json`: kind, the changed files, the selected files and the map's hash.
- **Why the self-checks stay in full.** P2: `gates/selfcheck_cache.py` is the incumbent selector for self-checks,
  and its `--carry` is barred from certification batteries. A second self-check selector would be a second
  mechanism for one job.

### 1.2 Reachable reds and controls (pre-registered)

| id | red / control | expected | how |
|---|---|---|---|
| R1 | a diff touching any non-`.md` path | REFUSED, rc != 0, no stage runs | self-check on a constructed change list; live: a `.py` touch on a scratch branch |
| R2 | map `code_hash` != tree | REFUSED | self-check on a constructed map; live: a map from a different head |
| R3 | a planted docs-only break that a reader test catches | docs-only battery RED, and the red test file is in the selection | live, scratch commit, never pushed |
| R4 | the same tree, no planted break | GREEN | live |
| H+ | instrument positive control: a child process with the hook opens a temp `.md` | one `read` record | self-check |
| H- | the same child without `SNDL_DOC_AUDIT` | no log file | self-check |
| HS | spawn injection: a child given an env dict WITHOUT the audit variables spawns a grandchild | the grandchild still logs, with the parent's test id | self-check |
| C1 | **control that is not the method:** the FULL pytest on the R3 planted break | its red test files are a SUBSET of the docs-only selection | live, one full pytest run (about 70 min) |
| CV | instrument coverage: python `Popen` events seen by parents vs `start` records from children, in the map run | counts reported side by side; a child with no `start` record is a blind spot, named | map build |

**Disposition on a null.**
- C1 finds a full-pytest red outside the selection: the rule does NOT land. That selection mechanism is deleted,
  and the docs-only battery falls back to the full pytest.
- R3 does not go red: the instrument is blind. Same disposition.
- R1 or R2 does not refuse: the build is wrong. Fix it before anything else.

**Named blind spots (disclosed now, not after a run).**
1. An in-process cache filled by test A and consumed by test B in another file shows the read only under A. C1
   samples this; it cannot prove its absence.
2. A Python child started without the gate venv's `site` (for example `-S` or `-I`) carries no hook. CV counts this.
3. A non-Python child that reads Markdown is invisible, except `git ls-files`/`git grep`, which are recorded as
   scans.
4. Only pytest is mapped. Self-checks and gates always run in full.

**Cost, measured on one unit before the full run:** the hook's overhead on one docs-reading test file, with and
without `SNDL_DOC_AUDIT` (section 2).

## 2. Results

*(Filled after the runs; not before.)*
