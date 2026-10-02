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

## 2. Results (2026-10-02; worktree branch `claude/docs-only-battery`)

### 2.1 Pre-registered rows

| id | verdict | what ran | reading |
|---|---|---|---|
| cost | MET | `tests/test_policy_manifest.py` alone, hook off vs on | 57 s vs 61 s (+7 %); the whole instrumented pytest took 76 min, against 71 min for the last plain battery |
| H+ / H- / HS | MET | module self-check, in real child processes | a read is recorded under its test file; nothing is logged without `SNDL_DOC_AUDIT`; a stripped env dict is re-armed, and the grandchild logs under the parent's test file |
| R1 | MET, live | `--run` on this branch | REFUSED in seconds, naming the 6 non-`.md` paths |
| R2 | MET, live, but not through `run()` | `check_map(M0, code_hash(tree))` | REFUSED: M0 was measured on `1428628a`. `--run` cannot reach R2 on this branch because R1 fires first; disclosed |
| R3 | MET | planted break on scratch branch `scratch/r3-planted` (never pushed): `SECURITY.md`'s `reflex.py` row renamed | the selection was `test_self_mod_safety.py` and `test_selfchecks.py`, and it went RED: 3 failures, 303 passed, 17 min 08 s |
| C1 | MET | the FULL pytest on the same planted break | the same 3 failures in the same 2 files; nothing outside the selection; 1,704 passed, 77 min 23 s |
| CV | MET by token pairing; the full-suite reading is in the arc battery's map | see 2.3 | 0 unpaired python spawns on a 2-file run; the constructed hookless child (`-S -I`) is named |
| R4 | NOT MET | needs a committed map on `origin` | the first docs-only push after this arc is its run, recorded in that push's record |

**Disposition: the rule lands.** C1 held: the full pytest found no red outside the selection.

**What C1 covers.** One planted break, read DIRECTLY in-process by its two reader files, under the map M0 that the
`1428628a` hook built. It does NOT test blind spot 1, an in-process cache filled by one test file and read by another.

### 2.2 Map M0 (instrumented full pytest on `1428628a`, 76 min)

- **Test files that READ Markdown: 5.**
  - `test_selfchecks.py` reads every tracked `.md` (335 of them), because it runs the tools gates in-process. Every
    docs-only diff therefore selects it, and it sets the floor: about 17 min.
  - `test_ratified_row_gates.py` reads 5, `test_conclusion_plate.py` 3, `test_policy_manifest.py` 1 and
    `test_self_mod_safety.py` 1.
- **`docs/POLICIES.md` is read at COLLECTION.** Any POLICIES edit therefore runs the full pytest, with coverage.
- **Collection-time directory listings were all infrastructure:** the import system's finder (`.`, `tools`,
  `training`, every `src/` package) and pytest's literal `testpaths` glob. Left in, they would have forced the full
  pytest on any added `.md`. Fixed in `9835b684` (2.4).

### 2.3 Coverage (CV)

The pre-registered count compared totals: 323 child starts against 302 python spawns seen by their parents. Pairing
each child to its parent by `ppid` is impossible on this host, because the venv's `python.exe` is a launcher, so a
child's `ppid` names the launcher: 277 of M0's child starts pointed at processes that never logged.

`896396aa` therefore stamps a per-spawn token into the child's environment, and the map lists every python spawn
whose token never started as a named blind spot. On a 2-file real run, 0 were unpaired. The constructed red, a
child started with `-S -I`, is named. The full-suite count is printed by the arc battery's `--build-map` and is
recorded with its record.

### 2.4 Defects the runs found in this arc's own instrument (all folded, each with a red)

1. **The `.pth` ran twice per process** (site-packages is reached twice), doubling every record. Fix: an install guard.
2. **Paths were recorded `normcase`d** (lowercase on Windows), so `docs/POLICIES.md` would not match git's spelling.
   Fix: real case recorded, and selection compares case-insensitively.
3. **The `.pth` embeds the hook at install time,** so a later probe ran the OLD hook. Fix: `--build-map` refuses a
   stale installed hook.
4. **The first instrumented pytest redded 3 tests, all from this arc:**
   - the hook OVERWROTE a nested self-check's own audit variables;
   - `test_g8_move_gate` pins `len(_MOVED)`, and the new REGIONS row derives one more alias: 286 to 287;
   - plus the infrastructure listings in 2.2.
5. **The overwrite fix (`setdefault`) would have pinned the FIRST test's id** into a shared module-level env dict such
   as `run_selfchecks._ENV`: `Popen` audits the caller's own dict, uncopied. Flagged by the advisor review. Fix: a
   dict with no audit dir, or this hook's own, is re-stamped on every spawn; a dict naming a different audit dir is
   left alone. New red: two spawns through one dict under two test ids.
6. **Proving that red exposed one more defect:** with the `.pth` installed, every self-check child ran the INSTALLED
   hook, not the code under test, and a `setdefault` mutant passed. Fix: self-check children run with `-S`. Both
   mutants (a `setdefault` test id, and no injection at all) are now killed.

### 2.5 What the rule buys, counted

- **Past arcs:** 3 of the last 15 record-to-record arcs were docs-only (about 20 %). The 2026-10-01 doc-sync arc was
  not, because its `doc_budget_gate` ceiling raise is a `.py` change. A docs change that needs a ceiling raise still
  takes the full battery.
- **Estimated docs-only battery:**
  - certify, about 1 min;
  - `run_selfchecks` in full, about 26 min;
  - gates, about 5 min;
  - the selected pytest, at least the 17 min that `test_selfchecks.py` needs;
  - total about 50 min, against about 2 h 15 min. A POLICIES edit or a collection-time reader runs the full pytest.
