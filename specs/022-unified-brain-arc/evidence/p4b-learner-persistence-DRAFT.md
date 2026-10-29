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

## 4. Addendum 1 (2026-10-29, lane persist-prep): freeze-readiness against `c4c10b28`

Appended only; nothing above is edited. Re-derivation table: `lanes/engine-1024/persist-prep-row/CLAIMS.md`. Static
reads only (no run, no import). Base for this addendum: `c4c10b28`. No `src/` file the draft cites changed between
`ee224e50` and `c4c10b28`, and brain code equals `1b6e53ab` (the section 2.8 premise holds).

### 4.1 Claims: 28 HOLD, 1 MOVED, 0 WRONG, 2 not re-derivable without a run
- **MOVED:** `ValenceLearner` is `plasticity/valence_stdp.py:46`; the draft's `:47` is its `__init__`. (Off by one at
  the draft's own base too.)
- **Not re-derivable (need a brain run, so left for the integrator's re-run at freeze):** the probe table in 1.2 and the
  section 2.9 item 1 flood measurement. The latter's probe code is untracked scratch, so the freeze must commit it or
  re-measure.

### 4.2 Three corrections the freeze must make (found while re-deriving)
1. **C1's read channel is not free.** `state()["prototypes_restore"]` (`brain.py:4238`) is `_prototypes_restore_note`,
   which is `None` when every sub-record was accepted (`brain.py` ~1180). C1 reads an `adopted` id on SUCCESS. Decide
   and write down: a new `adopted` field (a second channel) or a change to the note's None-on-success contract; and
   update the readers (`infra/brain_telemetry.py:388`, `instruments/eligible_circuit_census.py:311,962`, the P4b
   tests that read the note).
2. **Lazy-build gates.** JEPA exists only under `_jepa_on` (`autonomous_loop.py:471`, from `jepa`/`mlp_q`/
   `reward_shaped_encoder`); Valence only under `_dopa_on` (`:409`, `:2328`), and the same block builds
   `DopamineNuclei` and `LateralHabenula` (`:2335-2336`). The freeze states the default flags of `make_brain`, the
   flags of the test brain, and whether `_dopa`/`_lhb` state is derived (and why) or owed a sub-record.
3. **`_learned_attn` can be None** (`brain.py:1219`, import failure). The attention legs of C2/C4 need the rule
   "absent faculty: nothing written, nothing restored".

### 4.3 Policy fit
- **P4b.** Sub-records of `growing_learners.json` follow `01802dd4` (q, affect, lifelong), but P4b's text says "each
  under its own record tag" and the JEPA docstring (`jepa_world_model.py:634`) says a new `_T_*` tag. The freeze
  must cite the policy-enforcer's acceptance of the sub-record precedent, or the precedent's own reasoning.
  The honesty guard (a new process reads the lake back) is met by C1/C2/C3 as designed; the killed-parent leg (C3) is
  present.
- **P1 (sample size).** Section 2.4 says "no SD, no power; fixed count, one-shot" (P1 item 3 allows a one-shot with
  power stated before data). It states no power. The freeze needs one line: the claims are exact identities, so
  power is 1 by construction and a planted red (C4 refusals, the empty-home control) is the sensitivity proof.
  The sequential design (P1 item 2) does not apply: there is no stochastic effect and no seed block.
- **P2.** Met on the mechanism (one record, one format, retire the two JEPA file wrappers). Owed: section 2.9 item 3
  retires `save_weights`/`load_weights`, and the test's `git grep` red (C6) must show zero callers after, plus a
  statement that `to_record` has a live caller (the flush) so it is not the P3 self-check-only shape.
- **P3.** A restored record is "wired" only if the restored learner changes an outcome; the draft deliberately
  claims no behavioural benefit (2.7), and the incumbent consumers (`train_step`, `learn`, `_pfc_bias`, `attend`)
  read the restored state, so the disposition is "wired + charged by existing consumers". The freeze says that
  explicitly and names the dormancy-gate/census rows the new methods add.
- **P4b + C7.** The `persist=False` byte-identity claim (C7) needs a named check (digest pins or the T120 census) and
  an instrument-sensitivity red; today it only names a concern.
- **P8.** Reachable reds: C1/C2 red on the base (section 2.3) is reachable. Control that is not the method: the
  empty-home child and the arm-prototype positive control are. Owed: the C4 refusals must be shown to red against a
  restore that accepts blindly (the planted-bad-record test is the red, but name it).

### 4.4 What a freeze still owes (checklist)
1. Resolve 4.2 items 1-3 in the text.
2. Re-run the probe table (1.2) and the 2.9 measurement at the freeze head; commit the probe code.
3. State the P1 one-shot power line (4.3) and the C7 check name.
4. Cite the sub-record-vs-own-tag acceptance (4.3 P4b).
5. Production-tier (`eprop-snn`) confirmation, as 2.5, run by the integrator after the campaign.
6. Fix the `valence_stdp.py` line to `:46`.
7. Say which learners the 300-tick seeds actually build (needs 4.2 item 2).

### 4.5 Raising-ambition record (as E1d's section 0)
- **Axis:** Scope of what "a restart is not amnesia" covers: how many of the brain's live learners a new process
  recovers.
- **Ladder.**
  - **R1:** persist the four probed learners (JEPA, Valence, `_plan_w`, LearnedAttention) in the existing record, tested
    cross-process. This is the draft's design.
  - **R2:** R1 plus a census that proves NO other live learner is left behind: enumerate every learned-state
    attribute reachable from `Brain` and the arm (anything a `train_step`/`learn`/three-factor update writes) and show
    each is either in the record or on a disclosed derived-state list; `_dopa`/`_lhb` and any other lazy faculty are
    decided there.
  - **R3:** a persisted-brain continuation test: a restarted brain's next N ticks equal the uninterrupted brain's
    (bit-exact trajectory), so persistence is shown to be full continuation and not just state equality.
- **Pick: R1 for the build, with R2's census as a freeze precondition.** The draft is R1. The census is cheap and static
  (a stdlib scan of assigned learner attributes), and it is what stops "the four" from being a cherry-picked set.
- **The rung above, and the rule that rules it out.** R3 is ruled out by the arc's own constraints, not by cost:
  full continuation needs the RNG streams, the VICReg buffer and the transient gate/eligibility state, which the draft
  discloses as derived (2.1), and P2 forbids a second state format to carry them. The existing eligible-circuit census
  (`instruments/eligible_circuit_census.py:421`) already records "save_weights/load_weights ... NOT demonstrated full
  continuation", so R3 is a different, separately pre-registered claim.
- **Range.** Clinical-grade backup verification (restore into a clean system and diff); database recovery drills
  (a kill-9 restart, not only a clean shutdown: C3).
- **Outside judge** (none is the method): the child-over-empty-home negative control; the arm-prototype positive control
  in the same command; the integrator's re-run of the probe on the production tier; the policy-enforcer's review.

## 5. Panel fold 1, 2026-10-29, pre-data (lane fold-6231, board #6231)

Appended only; nothing above is edited. Base `c4c10b28`. Static reads only (no run, no import, no pytest). Sources:
lens A #6288 (A1-A5), lens B #6283 (B1 = the MAJOR, B2-B7 = its minors in report order: serializer idiom, retirement
refs, C1 channel, `_dopa`/`_lhb`, P4b wording, P2 misuse), lens C #6293 (C-1a..c), policy-enforcer #6248 (MINOR-1) and
its #6301 concurrence. Each finding is FIXED (the design now says X), DISPUTED (evidence) or OPEN (needs a run or an
owner). Where this section and an earlier one disagree, THIS section binds at the freeze, and 5.9 lists the earlier
text it supersedes.

### 5.1 A1 (MAJOR, the instrument is the method): FIXED
C1/C2/C1a graded the serializer with itself. Added to C1 and C2, for JEPA, Valence and attention:
- **Live-array leg.** In the child, at the adoption point, read the arrays the learner computes with, not its record:
  `jepa.Wz[i][j]` and `jepa.Wa[i][j]`, `valence.w[i]`, attention `w[b]`, each `==` the parent's flushed value (read from
  the parent's own live learner before `close()`, written to a side file under the test's temp dir by the parent).
  A `from_record` that caches the dict and leaves the live arrays fresh now fails this leg.
- **Functional leg.** The parent, before `close()`, evaluates one fixed prediction (`jepa` on a fixed (pattern, action)),
  one `valence(pattern)` and one `attend` on a fixed input, and records the outputs. The child evaluates the same at the
  adoption point (before that tick's `train_step`/`learn`). MET iff equal.
- C1a (`to_record(from_record(r)) == r`) stays as a unit test but is no longer cited as evidence that ids imply state.

### 5.2 A2 (MAJOR, undisclosed learned state `_plan_base`): FIXED (census run is OPEN, see 5.8)
c4c10b28 `engine/autonomous_loop.py:1538-1545`: `_plan_w` updates from `_plan_lr * da * (_plan_elig - _plan_base)`, then
`_plan_base` is an EMA of `_plan_elig` (`:1541-1543`) and `_plan_elig` is reset to 0.0 (`:1545`). The `plan_w` sub-record is
now `{"plan_w": float, "plan_base": float, "plan_elig": float}`, each refused unless finite (`plan_w` also within
[0.0, 1.0] as clamped at `:1539`; `plan_base`, `plan_elig` within [0.0, 1.0] as an EMA/indicator of a 0/1 trace). Persisting
`plan_elig` costs one float and removes the question whether a flush can land between its set and its reset. C1/C2 read
`plan_base` and `plan_elig` at 0 ticks, and C2 plants `plan_base = 0.3` (not the init 0.0). A **next-update leg** is
added to C1: after restore, the first `_plan_w` update from an identical `_rpe` and `_plan_elig` equals the parent's would-be
update (computed in the parent from its live state before `close()`), so a missing running baseline fails.
`_dopa`/`_lhb`: see 5.5 (B4). The R2 census (4.5) is now a **freeze precondition with its output committed**, with a
planted red: a learner attribute added in a scratch subclass must be reported missing. The census is a stdlib static scan
and is queued for the integrator (not run here).

### 5.3 A3 (MINOR, base red is a crash): FIXED
The base red for C1/C2 is the live-array leg of 5.1, which EXISTS on the base (`jepa.Wz` etc. are plain attributes) and
reads child == fresh != parent, an assertion failure on state. The `adopted` channel is read only AFTER the build commit
(it does not exist on the base), and the test is written so the base run fails at the live-array assertion, not at a
KeyError: the adopted-id read is guarded by `if "adopted" in note_channel` for the base red only, and the build commit
removes the guard (a test commit that guards then an unguard commit would be two tests; instead the red commit carries
the live-array leg alone and the green commit adds the id leg). Red/green queued for the integrator (no run here).

### 5.4 A4 + C-1b (MINOR, C6 git-grep red): FIXED
Dropped the "`save_weights` has zero callers" grep (A4: vacuously green once the def is gone; C-1b: as written it also
hits `spiking_jepa.py:868/:1315`, `readout/snn_classifier.py:170` and the census string at
`instruments/eligible_circuit_census.py:421`, which are out of scope). C6 now has two legs: (i) the directory listing
against the base `brain_lake.bin` + `skills.spkl` with a planted extra file (already in 2.3); (ii) no `open(` or
`Path.write_*` is reachable from `JepaWorldModel.to_record` / `from_record` (an `ast` walk of those two function bodies
that returns no such call), shown to red with a planted `open(` in a scratch copy. The retirement of
`JepaWorldModel.save_weights`/`load_weights` is checked by an `ast` test that the two defs are absent from
`faculties/jepa_world_model.py`, with a planted re-added def as its red. The build commit lists every remaining
reference to the retired names: `instruments/eligible_circuit_census.py:421` (a boundary string, rewritten),
`sensorimotor/rsnn_encoder.py:54` and `instruments/carried_state_campaign.py:162` (comments, rewritten), and the module's
own self-check at `faculties/jepa_world_model.py:1005-1036` (rewritten to the record methods) (B3).

### 5.5 B4 (MINOR, `_dopa`/`_lhb`): FIXED statically, census still runs
Lens B's anchors hold at c4c10b28: the arm calls only `DopamineNuclei.da_level` (`autonomous_loop.py:2344`, a pure function
of its argument, `faculties/dopamine_nuclei.py:126-129`) and `LateralHabenula.dopamine` (`:2358`); the critic cache `v` is
written only by `present(learn=True)` (`faculties/habenula.py` `:152-165`), which has no live call. Disposition, as
4.2 item 2 asked: `_dopa` and `_lhb` are **derived (seeded thresholds), nothing to persist**, with that citation. This is a
static claim, not a run; R2's census (5.2) re-checks it and reds if either object gains a learned attribute.

### 5.6 C7 (B1 MAJOR, no named instrument, no reachable red): FIXED by naming; one sub-claim OPEN
The draft's C7 claimed byte-identity of `persist=False` brains with no instrument. Replaced by two named checks:
- **Instrument.** `tests/test_brain_hermeticity.py` (T120 rung 3, the existing hermeticity census) is extended to the four
  new sub-records: a `persist=False` brain must neither read nor write them, and a brain over a temp home must keep them in
  that home only.
- **Reachable red.** A planted monkeypatch that forces the restore ON for a `persist=False` brain, over a home that holds
  a planted `jepa`/`valence`/`plan_w`/`attention` record, must flip the test (the four learners would read the planted values
  instead of the fresh ones). Without that flip the check is vacuous.
- **OPEN:** "every pinned digest of the determinism/replay/envelope gates is unchanged" needs a gate run to read
  and cannot be shown statically. It is kept as an integrator post-campaign read, not as a conjunct of this draft; if the
  freeze cannot name the gate whose pinned digest a `persist=False` brain feeds, that sub-claim is deleted (not weakened).

### 5.7 B2, B5, B6, B7 and the enforcer's MINOR-1
- **B2 (JEPA idiom): FIXED.** The tree's idiom is `state_dict`/`load_state_dict` (`growth/growing_learner.py:117,127`,
  the same record's other sub-records). JEPA gets `state_dict()` / `load_state_dict()`, not `to_record`/`from_record`.
  Everywhere above, read `to_record` as `state_dict` and `from_record` as `load_state_dict`; the freeze rewrites the
  names in sections 2.1-2.3.
- **B5 (C1 channel): FIXED.** Not a second field. The one note `_prototypes_restore_note` is extended: it carries an
  `adopted` map `{learner: id}` on success too, its None-on-success contract is changed to "None only if nothing was
  adopted or refused", and its readers are updated in the build commit (`brain.py:1180-1184` writer; readers
  `infra/brain_telemetry.py:388`, `instruments/eligible_circuit_census.py:311,962`, and the P4b restart tests). This
  resolves 4.2 item 1 by the first of its two options' cousin: ONE channel, changed.
- **B6 + enforcer MINOR-1 (P4b "own record tag"): PARTLY FIXED, the ruling is OPEN and the owner's.** The sub-records live
  in `growing_learners.json`, a named record under `_T_STATE` inside `brain_lake.bin`, i.e. in the lake, not a sidecar, per
  precedent `01802dd4` (q, affect, lifelong). P4b (`POLICIES.md:380-381`) reads "each under its own record tag ... not a JSON
  blob". The policy-enforcer's #6248 states that #5191 agreed on P2 grounds and did NOT rule on that clause, so 4.3's
  "cite the policy-enforcer's acceptance" is **withdrawn** (#5191 does not settle it). The freeze carries EITHER an owner
  ruling that the sub-record form meets P4b, OR a one-line P4b amendment through the adding-a-policy bar; until then the
  freeze states the tension and the chosen form. `jepa_world_model.py:634` ("a `_T_*` record tag") is corrected in the
  build commit to name the sub-record.
- **B7 (R3 ruled out on P2): FIXED.** 4.5 said P2 forbids a second state format to carry RNG streams. Carrying them in the
  same record is not a second format. R3 stays ruled out on its stated scope alone: a separately pre-registered
  continuation claim that needs the RNG streams, the VICReg buffer and the transient gate state, which this draft
  discloses as derived.

### 5.8 A5 (MINOR, numbers from untracked scratch): OPEN, carried
2.9's table and the FLOOD figures come from untracked scratch. Nothing in this fold re-derives them (no run). The same-record
decision is NOT frozen on those numbers until the probe code is committed or the figures re-measured at the freeze head
(4.4 item 2, unchanged). Also OPEN (run needed): the R2 census output (5.2), the base red and green runs (5.3), and
the production-tier confirmation (4.4 item 5).

### 5.9 C-1a and C-1c (lens C minors): FIXED
- **C-1a.** 2.1's serializer row says the JEPA file methods "become thin wrappers". That is superseded: 2.9 item 3 and
  4.3 retire them (5.4). 4.4 gains **item 8: rewrite 2.1's JEPA row to retire, not wrap, `save_weights`/`load_weights`,
  and rename `to_record`/`from_record` to `state_dict`/`load_state_dict` (5.7 B2)**. 4.4 also gains **item 9: commit the
  R2 census output (5.2); item 10: the owner ruling or P4b amendment (5.7 B6)**.
- **C-1c.** On stdlib `LearnedAttention` does not learn (1.2), so its child's fresh `w` equals the parent's unlearned `w`
  and C1's id and live-array legs hold even with a broken restore. 2.3 called C1 "the red that must flip" for all four.
  Corrected: **C2 (planted values) is the only red for any learner that does not move on the tier.** 4.4 item 7 lists
  them: on the stdlib tier, `LearnedAttention` (confirmed in 1.2) and possibly `ValenceLearner` and `_plan_w`, which
  depends on 4.2 item 2's flags and on whether the 300-tick seeds reach a rewarded tick; the freeze states, per
  learner, whether C1 or only C2 can fire.

### 5.10 Not disputed, not silently dropped
No finding in this round is DISPUTED. Lens A's HOLD list, lens C's verification list (lines 1-293 byte-identical to
`3ced10df`; 4.1 MOVED correct) and lens B's P1 note (one-shot fine; add the power line, already owed in 4.4 item 3)
require no change. The P1 power line stays as 4.3 wrote it.
