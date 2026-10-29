# P4b: persist the four live learners a restart forgets -- DRAFT pre-registration (2026-10-29)

**Status: DRAFT, design only. No `src/` code.**
- Written by the cloud session (`session_01TeA2yYimjAiHiGFzw6hqHX`) at the launcher's instruction (2026-10-29,
  item 4).
- The integrator owns the final pre-registration, its row number, the build, landing and the battery.
- The policy-enforcer reviews.
- This file freezes nothing. The integrator's own commit of the final text is the freeze, and it re-derives every
  territory number below at its head.

**Base:** `ee224e50`. Every file:line below is read at that commit.

**Upstream evidence:** `research/restart-probe-1028/FINDINGS.md` on branch `cloud/restart-probe-1028`, head
`a8893811`.

## 1. Territory (read at `ee224e50`, 2026-10-29)

### 1.1 The policy

P4b, `docs/POLICIES.md` section "P4b -- THE DURABLE LAKE IS PART OF THE BRAIN":
- "What the brain learns goes IN the lake ... each under its own record tag in `brain_lake.bin`. Not a sidecar file,
  not a JSON blob."
- "A restart is not amnesia. A brain constructed over an existing lake starts with what that lake holds. Restore is
  gated on `persist`..."
- Honesty guard: "'it persists' is a claim about a NEW process ... Prove it by reading the lake back from disk after
  the writer is gone."

The 2026-10-24 policy audit BLOCKed P4b sidecars (`pa-p4bd-1`, fixed in arc 2). So the remedy here must be a lake
record, never a file beside the lake.

### 1.2 The measured gap

The cloud restart probe ran at `c1df3887`, stdlib tier. `make_brain(seed, persist=True)` on a scratch home; parent 300
ticks then `close()`; then a child on the same home and a fresh brain on an empty home, each its own interpreter.

| learner | where it lives | how it learns | result |
|---|---|---|---|
| JEPA world model | `AutonomousArm.jepa` (`engine/autonomous_loop.py:2067`, built lazily on the first tick) | `train_step` every tick (`:2509`) | **NOT RESTORED**: child bit-identical to fresh after the building tick, seeds 7 and 11 |
| `ValenceLearner` | `AutonomousArm._valence` (`:2337`, built lazily on the first rewarded tick) | `learn` (`:2341`) | **NOT RESTORED**: same; planted `w` not read back |
| `_plan_w` | `AutonomousArm._plan_w` (`:614`, a float, initial 0.1) | `:1538` | **NOT RESTORED**: child = fresh = 0.1; planted 0.4321 reads back 0.1 |
| `LearnedAttention` | `Brain._learned_attn` (`engine/brain.py:1214`) | three-factor update of `w` (`faculties/learned_attention.py:173`) | **NOT RESTORED** (planted run). It did not learn in 300 stdlib ticks |
| *positive control:* arm `GrowingLearner` prototypes | `growing_learners.json` | | **RESTORED** |

The home after `close()` held `brain_lake.bin` and `skills.spkl` only. These four have no record of any kind; they
are not sidecar violations, they are absent.

### 1.3 The incumbent persistence path (P2: adopt it, do not add a second)

`Brain._flush_prototypes` (`engine/brain.py:5153`) writes ONE named-state record, `growing_learners.json`
(`PROTOTYPES_STATE`, `:89`, `_T_STATE`).
- Its sub-records are `arm`, `vis`, `aud`, `q`, `affect` and `lifelong`.
- Its two callers are the night's persist block (`:3799`) and `close()` (`:6156`).
- The restore runs at construction (around `:1150-1190`). It applies a per-key `_accept` with refusal. A refused
  sub-record is kept under `REFUSED_PREFIX` (`:91`; `_keep_refused`), and `_prototypes_carried` carries it until the
  live store supersedes it.
- `_record_id` (`:94`) is a sha256 prefix of the canonical JSON.

Arc 2's `01802dd4` (pa-p4bd-3) added `q`, `affect` and `lifelong` this way, with `tests/test_p4b_restart_restore.py`
as its cross-process red. **This row extends the same record and the same test pattern. It adds no new store and no
new tag.**

### 1.4 What each learner's state is

- **JEPA.** It already has a validated serializer, `save_weights` / `load_weights`, formats `snn_jepa/1` (linear)
  and `snn_jepa/2` (deep encoder) (`faculties/jepa_world_model.py:613`, `:669`).
  - The format carries actions, `in_dim`, `latent_dim`, `tau`, `lr`, the lambdas, the predictor (`Wz`, `Wa`, `b`) and
    the online plus EMA-target encoders.
  - `load_weights` validates every shape and resets the VICReg ring buffer `_buf` (`:758`), so the buffer is treated
    as derived.
  - The method's own docstring records P4b as open: "Closing it means a `_T_*` record tag ... recorded here so the
    next reader finds it".
- **ValenceLearner** (`plasticity/valence_stdp.py:47`): `w` (list, length `n`), `n`, `lr`. It has no state methods.
- **`_plan_w`**: one float on the arm.
- **LearnedAttention** (`faculties/learned_attention.py`):
  - learned: `w` (`n_bands`, clipped to `[w_min, w_max]`) and the running salience expectation `_exp`;
  - transient, recomputed every step: `_gate`, `_elig`, `_focus_*`;
  - the RNG `_rng` is used only at construction.

## 2. Pre-registration (DRAFT)

### 2.1 The mechanism

One change to `Brain._flush_prototypes` and to the construction-time restore. It adds four sub-records to
`growing_learners.json`: `jepa`, `valence`, `plan_w` and `attention`.

**Serializers, one per learner** (P2: no second format):

| learner | serializer |
|---|---|
| JEPA | `save_weights` and `load_weights` are split into `to_record() -> dict` and `from_record(dict)`. The two file methods become thin wrappers, so the `snn_jepa/1` and `/2` format and its validation are the one mechanism. At `ee224e50` their only callers are the module's own self-check (`faculties/jepa_world_model.py:1005-1036`); the docstring's train-tier caller, `reasoning/jepa_scaffold_torch.py`, was retired in `48e9912d` (2026-10-18). `faculties/spiking_jepa.py` has its own `save_weights` and `load_weights`; it is a different model and out of scope. |
| ValenceLearner | gains `state_dict()` / `load_state_dict()`: `{"n", "lr", "w"}`, refused unless `len(w) == n == the live N` |
| `_plan_w` | `{"plan_w": float}`, refused unless finite |
| LearnedAttention | gains `state_dict()` / `load_state_dict()`: `{"dim", "n_bands", "w", "exp"}`, refused unless `dim` and `n_bands` match and every `w` lies in `[w_min, w_max]` |

**Lazy adoption.** JEPA and Valence do not exist at construction. A restored `jepa` or `valence` sub-record is held
in `_prototypes_carried` exactly as a refused record is today. It is **adopted at the lazy build site**
(`autonomous_loop.py:2067` and `:2337`): built, then `from_record` / `load_state_dict`, then accept or refuse. An
accepted record leaves the carried set. A refused one is kept under `REFUSED_PREFIX` by the existing keep path.

The flush writes a live learner's state only once that learner exists. Until then it carries the restored record
forward unchanged, the same rule `_flush_prototypes` already applies to a refused sub-record.

**Identity.** Each sub-record carries its `_record_id`. In addition:
- `jepa` records `actions` and `in_dim`, which must equal the live arm's `ACTIONS` and percept width, or the record is
  refused whole;
- `plan_w`, `valence` and `jepa` belong to the arm. As the code already does for `q`, a record written for a
  different arm record id (`arm=_record_id(arm)`) is refused once the arm is superseded.

**Not persisted, and why.** These are derived state, rebuilt or recomputed at restore, and disclosed:
- JEPA's `_buf`: VICReg statistics over recent latents; `load_weights` already resets it;
- LearnedAttention's `_gate`, `_elig`, `_focus_*`: recomputed every step.

**Gating.** `persist=False` reads and writes nothing, as today.

### 2.2 Conjuncts (each MET / NOT MET, none secondary)

Each conjunct is a cross-process test in the `tests/test_p4b_restart_restore.py` pattern, every home a fresh temp
dir passing `assert_not_live_home`:

- **C1, restored (the probe's red, inverted).** *(Folded 2026-10-29, launcher N1: the read point.)* For each of the
  four learners, a child process over the parent's home reads back exactly the parent's flushed state. Equality is
  exact (`==` on the JSON-canonical record, compared through `_record_id`), never a tolerance.
  - **The read point for JEPA and Valence.** They are built lazily, and the build tick also learns: the JEPA build at
    `autonomous_loop.py:2067` runs at the top of the step and its `train_step` at `:2509` runs later in the SAME step;
    Valence is built and then `learn`s in one block (`:2337-2341`). So the child's state after that tick is already
    one step past the restored record, and reading it there would compare the wrong thing.
  - **The rule.** The comparison is made AT THE ADOPTION POINT: immediately after `from_record` /
    `load_state_dict` at the build site, before that tick's first `train_step` / `learn`.
  - **How, with no test-only seam (P3).** The adoption step already decides accept or refuse. Accepting computes
    `_record_id(learner.to_record())` (Valence and attention: `state_dict()`) of the learner as built, and reports
    it through the existing production restore channel, `state()["prototypes_restore"]` (`engine/brain.py:4238`),
    for example `adopted: {"jepa": <id>}`. That is the same channel that already reports refusals and adoptions.
  - C1 reads that id in the child after the build tick, and the parent's flushed sub-record id from the lake
    (`home.load_state("growing_learners.json")` in the child process). It is MET iff the two ids are equal.
  - **Equal ids mean equal state.** A brain-free round trip pins that: `to_record(from_record(r)) == r` exactly, per
    learner (C1a, a unit test).
  - `_plan_w` and LearnedAttention exist at construction, so they are read at 0 ticks, before any step.
- **C2, planted.** The parent writes known values before `close()`:
  - attention `w = [0.25, 5.5, 0.5, 4.0]`, `_plan_w = 0.4321`, Valence `w[i] = 0.123`;
  - a JEPA predictor entry set to a sentinel.

  The child reads exactly those values, at C1's read points: the adoption-point id for JEPA and Valence, 0 ticks for
  `_plan_w` and attention. This covers learners that do not move on a tier (LearnedAttention on stdlib).
- **C3, killed, not closed.** A parent killed with `os._exit` after the night's flush, and no `close()`, still comes
  back with what that flush wrote, as `tests/test_growing_learner_restart.py`'s killed red does for the prototypes.
- **C4, refusal whole.** For each learner, a record of the wrong width or shape is REFUSED:
  - JEPA with `in_dim + 1`;
  - Valence with `n + 1`;
  - attention with `n_bands + 1` or `w` out of bounds;
  - `_plan_w` NaN.

  The live learner then starts fresh (bit-identical to a fresh brain's), the restore note names the refusal, and the
  refused bytes are kept under `REFUSED_PREFIX`. Never half-restored.
- **C5, foreign records preserved.** Every lake record of another tag or name is byte-identical across a flush.
- **C6, no sidecar.** After `close()` the home holds no file that was not there before this change: the directory
  listing, against the base `brain_lake.bin` + `skills.spkl`. `save_weights` has no caller on any live path (a
  `git grep` red in the test).
- **C7, hermetic gates unchanged.** `persist=False` brains are byte-identical before and after the change, so the
  determinism, replay and envelope gates and every pinned digest are untouched. This is the T120 hermeticity census's
  concern.

### 2.3 Reds and controls

- **The red that must flip is the probe itself.** At the base (`ee224e50`), C1 and C2 read NOT MET for all four
  learners: child bit-identical to fresh, as `FINDINGS.md` records. The build commit carries this red. It must be run
  RED on the base before the build and GREEN after.
- **Positive control, in the same command.** The arm prototypes (`arm`) are restored on the base and after the change.
  If they are not, the harness is broken and no reading is taken.
- **Negative control.** A child over an EMPTY home equals a fresh brain for all four. A restore that invents state
  fails this.
- **Refusal reds:** C4's four constructed bad records, each of which must be refused.
- **Sidecar red:** C6's planted extra file must be detected by the listing check.

### 2.4 Sample size (P1)

**No SD and no power.**
- Every conjunct is an exact, deterministic identity: equality of canonical records, a refusal, a byte comparison.
  It is not a stochastic effect, so there is no SD to measure and no power to state.
- A fixed count is used, as a one-shot, which P1 item 3 allows for a non-stochastic claim. The reason is stated here.

**Seeds.**
- The cross-process tests use seeds 7 and 11, the probe's seeds, both of which reach the lazy JEPA and Valence builds
  within 300 ticks.
- The integrator may add one seed that reaches neither lazy build. That would exercise "carried until built" across
  a restart in which the learner was never built.

### 2.5 Production-tier confirmation (owner host)

The cloud can run only the stdlib tier. After the build lands on a branch, the integrator runs the probe in
`FINDINGS.md` (its code is included verbatim) on the owner's host:
- `perception_tier` must read `eprop-snn`;
- seeds 7 and 11, 300 ticks.

It reports, per learner, `child1_eq_parent` (expected True after the change) and `child1_eq_fresh1` (expected
False).

On that tier LearnedAttention may learn naturally. If it does not, C2's planted reading stands alone and the record
says so.

Run this one heavy run at a time: the owner host is memory-constrained.

### 2.6 Disposition

- **All of C1-C7 MET** on stdlib and the production confirmation consistent: the four learners persist. P4b's
  "restart is not amnesia" holds for them. The JEPA docstring's P4b note is closed in the same commit.
- **Any conjunct NOT MET:** the build does not land. A partial (for example, three of four) lands only by the owner's
  decision, with the missing learner's gap recorded.
- **The red does not fire on the base** (C1 already MET before the change): the harness is BLIND. No change lands;
  repair the harness.

### 2.7 What this row does NOT claim or do

- It does not claim any behavioural or capability benefit from persistence. That would be its own pre-registered
  measurement.
- It changes no learning rule and no hyperparameter.
- No sidecar, no new record tag, no new store.
- It does not touch the five `brainstem/` envelope files.
- It has no effect on E1c, whose runs use fresh per-seed brains.
- It does not persist derived state (JEPA's `_buf`, attention's transient gate and eligibility).

### 2.8 Order and commit order (the integrator's)

**Landing order (launcher ruling, 2026-10-29).** This row lands AFTER the E1d campaign. E1d may cite its calibration
only while the brain code outside the pc harness equals `1b6e53ab`. This row changes brain code
(`engine/brain.py`, `engine/autonomous_loop.py`, `faculties/jepa_world_model.py`, `faculties/learned_attention.py`,
`plasticity/valence_stdp.py`), so landing it earlier would void that citation.

Commit order:
1. The final pre-registration, alone.
2. The red: the cross-process test, committed and shown RED on the base.
3. The build: serializers, record keys, lazy adoption, and the JEPA wrapper retirement (section 2.9, item 3).
4. The test GREEN, stdlib.
5. The production-tier confirmation record.
6. Panel, fold and battery.

### 2.9 Questions, two of them resolved by measurement (fold of 2026-10-29)

1. **Same record or a sibling: RESOLVED, same record** (launcher N2: measured, not guessed).

   **Probe.** Stdlib tier, `ee224e50`, `make_brain(persist=True)` on scratch homes, seeds 7 and 11, 300 ticks each,
   instance-level wraps on that probe brain only. At every night flush it read:
   - the JEPA record's canonical-JSON bytes (the `save_weights` payload, `sort_keys`, ASCII);
   - `growing_learners.json`'s bytes, and whether it changed;
   - tag 0x07's (`_T_STATE`) bytes that tick and its `AppendBudget.bar(7)`.

   | quantity | seed 7 (12 flushes) | seed 11 (13 flushes) |
   |---|---|---|
   | JEPA record, canonical bytes | 56,587-56,648 | 55,702-56,540 |
   | `growing_learners.json` bytes today | 9,820-19,680 | 5,694-13,814 |
   | `growing_learners.json` changed at the flush | 12 of 12 | 13 of 13 |
   | tag 0x07 bytes on the flush tick | 3.29-16.11 MB | 3.28-16.10 MB |
   | tag 0x07 bar (`FLOOD_FACTOR` 8) | 99,048 B for the first 2 flushes (8 x the recorded 12,381 B seed), 0.42 MB at the 3rd, then 26.2-48.1 MB | the same shape: 99,048 B, then 0.43 MB, then 26.2-48.1 MB |
   | FLOOD verdicts | flushes 1-3 FLOOD, then none | flushes 1-3 FLOOD, then none |
   | headroom after flush 3 | at least 21.78 MB | at least 21.77 MB |

   **Reading.**
   - JEPA adds about 56.6 KB per flush: 0.35-1.7% of the tag's bytes on a flush tick.
   - It flips no flood verdict. The first three floods happen with or without it, and every later flush keeps at
     least 21.7 MB of headroom with it added.
   - A sibling record saves nothing measurable. `growing_learners.json` changed at every flush in both seeds, so it is
     rewritten nightly anyway. A sibling under the same `_T_STATE` and the same flush writes the same bytes in the same
     tick.
   - **Recommendation: the same record** (P2, one persistence path; the pa-p4bd-3 precedent).
   - **Re-measure at freeze on the production tier.** There, the prototypes may stabilise so that
     `growing_learners.json` is sometimes unchanged. A sibling would then avoid rewriting at most the record's own
     ~20 KB on such nights, still under 1% of the tag's flush bytes.
   - **Side finding, not this row's.** Every fresh brain FLOODs tag 0x07 on its first three night flushes, because
     the window is still seeded from `tools/lake_append_baseline.json`'s August value (12,381 B, bar 99,048 B)
     against 3-16 MB written. This is the stuck-alarm shape `AppendBudget`'s docstring describes, still present for
     a new brain's first nights. Reported to the integrator.
2. **Write frequency: RESOLVED by item 1.** JEPA is written at every night flush, about 56.6 KB, inside the measured
   headroom after the seeded window.
3. **The JEPA file wrappers: RESOLVED, retire them** (launcher N3; P3: a self-check caller is reachability, not
   charge).
   - At `ee224e50`, `JepaWorldModel.save_weights` / `load_weights` have no caller outside the module's own self-check
     (`faculties/jepa_world_model.py:1005-1036`).
   - Their one train-tier caller, `reasoning/jepa_scaffold_torch.py::verify_transfer`, was retired in `48e9912d`.
   - `faculties/spiking_jepa.py`'s same-named methods belong to a different class and are out of scope.
   - So the build keeps `to_record` / `from_record` (both `snn_jepa/1` and `/2`, with the same validation). It
     retires the two file methods, moves the self-check to a record round trip, and corrects the docstring's P4b note
     in the same commit.
   - A future deep-encoder import re-adds a reader of the record, never a file path.

## 3. Fold log

- **2026-10-29, launcher review of `b2d0b800`** (verdict AGREE as a draft; N1-N3 plus the order ruling).
  - C1 gains its read point: the adoption-point id through `state()["prototypes_restore"]`, no test-only seam, plus
    the C1a round trip. C2 follows it.
  - Section 2.8 records that this row lands after E1d (the `1b6e53ab` ruling).
  - Section 2.9 item 1 is resolved by the measurement above (same record), item 2 by item 1, and item 3 by retiring
    the wrappers.
  - The N2 probe's code is the session's scratch, not tracked. Its numbers are reproducible with instance wraps of
    `Brain._flush_prototypes` and `AppendBudget.end_tick` on a scratch brain.
