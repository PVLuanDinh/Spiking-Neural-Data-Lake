# E1d -- E1c's powered `_plan_w` harm test (H-P1), re-run with a positive control that cannot collide with the held goal

> **Status: FROZEN, pre-data.** Frozen at the commit that dropped "(DRAFT pre-registration)" from the title, after all
> four reviewers AGREEd on `3e11db18` (fold 4): panel A #5890, B #5895, C #5893 and policy-enforcer #5894. Every
> challenge on #5855 is resolved.
> - Nothing below has been run in a brain on seeds 23000 and up.
> - Later changes are dated pre-data amendments, appended below section 12's heading. Any other change is a new
>   pre-registration on a fresh block.
> - The panel on the built stack before its battery, and the step-6b bind review (section 9), are separate, later
>   reviews.
> - **Task:** T124 [E1], registered in tasks.md by the freezing commit. Board claims: #5853 (this file), #5854 (seeds
>   23000-23999).
> - **Why it exists:** E1c (T122) is VOID at look 1 (#5840). Its positive control `pc` read 0 flips on seed 21001.
>   - Diagnosis #5845, accepted by delegate D-37: a brain property, not a harness defect.
>   - Delegate D-36: E1c's frozen text routes a VOID to "repair, or a new pre-registration on 21704 and up".
>   - D-37 rules on this successor: seeds from 23000, a held-goal pc, a measured base rate and the order of work.
> - **Inherits:** owner #4616 ("the production tier alone decides a `_plan_w` deletion"). The owner's go for this
>   question is "_plan_w first" (#5609).
> - **Read against:** the integration head `1b6e53ab` (origin/main `ee224e50` plus E1c addendum 5). The brain code
>   equals `c1df3887`'s, as E1c recorded.
> - **Review fold 1:** panel A #5858, B #5859, C #5860 and policy-enforcer #5857 (all CHALLENGE). Folded below; each
>   item is traced in section 11.

## 0. Raising-ambition record (self-apply, owner 2026-10-25)

- **Axis:** Depth. The question is E1c's; what failed was a control, so this row makes the controls excellent and does
  not widen the question.
- **Ladder.**
  - **R1:** replace the pc with a held-goal perturbation and red it on 21001 (D-37's floor).
  - **R2:** also check the new pc's REACH across a seed population before the GO, against fresh same-build baselines,
    with a comparator that is itself shown to fire. Record the held goal in every run, so the goal-frozen stratum is
    OBSERVED in the campaign, not estimated.
  - **R3:** a dose-graded control (rotations k = 1..6, flips monotone in k).
- **Pick: R2.**
  - R3 is ruled out: the validity check needs a control that cannot be blind. It does not need a dose-response, and a
    dose series adds runs that no decision reads (P1: no run without a reader).
- **Range.**
  - Clinical assay validation: a positive control's reach is shown on the population it guards, not on a single
    sample.
  - Metrology: calibrate across the operating range, not at a point, and show that the comparator itself can read a
    difference.
  - Adversarial: the known blind seed 21001 is the red the new control must survive.
- **Outside judge** (none of these is the method itself):
  - the `pfrozen` twin;
  - the A/A repeat;
  - the new pc's red on 21001;
  - its 64-seed reach read (R2c) against fresh baselines;
  - R1's brain-free compare fixture.

## 1. What E1d is

E1d is **E1c's frozen design**: `e1c-plan-w-powered.md` at `082b66f8`, plus its addenda 1-5, as they stand at
`1b6e53ab`. Only sections 2-10 below change it. Everything else holds verbatim, by reference to that file:
- E1c section 1 (question and licence);
- sections 4 and 5 (endpoints, and the sample-size design 5.3), restated per file in section 4b as P1 requires;
- section 8 except its VOID route (section 8 below);
- addenda 2-4's R5 rule (D-31, mechanical), with "this file" meaning THIS file (section 7);
- addendum 5 (D-34: serial, the pilot's T/P/GPU, the 48 h line).

**E1c's look-1 ENDPOINT data decides nothing here** (D-36). t = 1.531, k 34/64, SD 0.3542, the d_s values and the tail
census were seen. They appear in no design number, seed choice, stratum or threshold of this file.

E1c's look-1 CONTROL outcome is different: `pc_21001` read 0 flips. That outcome is the declared cause of sections 2-3,
which is the legitimate kind of input.

## 2. Why E1c is VOID, and the measured facts this design rests on

- **The old pc.** It stubbed `PrefrontalExecutive._decide_spiking` to return winner 0.
  - On production, the race reaches behaviour only through `_gate_primary`. That gate rewrites the held goal only when
    the margin clears `gate_threshold`, and `top_down_bias` reads the held goal.
  - On seed 21001 no commitment ever clears the gate: 293 of 293 gate calls hold goal 0.
  - So "force 0" changed nothing (#5845).
- **Base rate** (D-37 item 3, #5850):
  - A seed is **goal-frozen** when its held goal never changes after its first write.
  - 9/64 = 14.1% (Wilson 95% [7.6%, 24.6%]) of calibration seeds 28000-28063 are goal-frozen, all at goal 0. The first
    held goal is 0 on 64/64 seeds.
  - Moves per seed: median 2, max 6.
  - Source: tree `1b6e53ab`, pyd `5c5b39c1`, instrument `diag_pc.py` `d0205e91` (record mode). Its records of 21000
    and 21001 equal E1c's `real_21000`/`real_21001` in `series`, `v` and `p`. Summary:
    `lanes\engine-1024\e1c-void-diag\CALIB-SUMMARY.txt`.
- **The brain-code boundary for citing the base rate** (D-38 item 2).
  - **Brain code** is every path under `src/` and `native/` EXCEPT the three allow-listed files below.
  - `git diff --name-only 1b6e53ab <GO tree> -- src native` must list only these, each behaviour-neutral for the reason
    given:
    - `src/snn_data_lake/instruments/rpe_charge.py`: the harness (B1-B3). Brain code reaches it only through the
      `__init__.py` canon list (no live call). Its effect on behaviour is what R2c's 64-seed identity cross-check reads
      (section 6).
    - `src/snn_data_lake/gates/dormancy_gate.py`: B1 changes only the string text of the `rpe_charge` EXEMPT entry.
      `region_wiring.py` imports the module, but no value in that entry is read as behaviour.
    - `src/snn_data_lake/gates/docs_only_battery.py`: winpaths-r1 (B4). It has zero importers in `src`.
  - Any other path voids the citation, and the base rate is then re-measured on the GO tree before the GO.
  - Brain modules DO import `instruments/` and `gates/` elsewhere (e.g. `brain.py:3946`, `:5368`;
    `agency_monitor.py:119`; `region_wiring.py:559`). That is why the boundary is this allow-list and not a pair of
    folders.
- **Inputs outside the diff.** The perception packs under `spiking-neural-data-lake\models` are pinned by path in
  `run_arm.ps1`, but only the encoder checkpoints are hashed (`rpe_charge.py:1017`, `:1251`). R2c's 64-seed identity
  cross-check is the guard that covers them: same behaviour on 64 seeds, or a halt.
- **Build identity** (D-38 item 1). Two different `_native` builds of one commit (`5c5b39c1`, `a6176948`) give
  byte-identical `series` and `v` on 2 seeds (21000, 21001). The claim is "identical behaviour on 2 seeds", not
  "metadata only".
  - R2c's identity cross-check extends it to 64 seeds: the calibration runs are on `5c5b39c1`, the fresh ones on the
    pinned GO build.
  - The pin stays either way.

## 3. Arms and runs (changes to E1c section 3)

- **Arms, tier, harness settings:** as E1c (`real`, `pfrozen`; production; 600 ticks; held-out from tick 300).
- **The new positive control `pc` REPLACES the constant-winner pc** (P2: one control mechanism, never two side by side).
  - **The stub.** The `pc` arm is `real` plus one class-wide stub of `PrefrontalExecutive.top_down_bias`.
    - For calls whose instance is the arm's own PFC (`slf is <the AutonomousArm>.pfc`, scoped as the existing
      `_pfc_bias` wrapper already scopes), it calls the real method and returns the vector rotated by one, as a list:
      `out[(i + 1) % n] = bias[i]`, with n = `n_options` = len(ACTIONS) = 13.
    - Calls on any other instance pass through unrotated and uncounted.
  - **What behaviour sees.** On every `_pfc_bias` call (one per awake tick: 291-303 per 600-tick run) the bias points at
    (held + 1) mod 13, not at the held goal.
    - This rests on two facts: the live stack depth is 1 (`maintain()` and `push_subgoal()` have no live caller,
      `autonomous_loop.py:1408-1415`), and the gain is never 0 (`pfc_gain` clipped to [0.1, 1], `:2295-2296`).
  - **Disclosed (P8 item 4).** D-37 rule 2 says "perturb the HELD goal itself". This stub perturbs the goal's READ: the
    held goal, and so the `pc` arm's recorded `held_goal`, stay the brain's own.
    - On the live path the two are equivalent, because the held goal reaches behaviour only through `top_down_bias`
      (#5845; one frame).
    - The read-side form lands AFTER the gate on every call, which a write-side stub of `_gate_primary` would also
      do. It keeps the `held_goal` record meaningful in the `pc` arm.
  - **The planned-sequence nudge is added after the stub** (`:1545-1551`) and is not rotated. So the pc moves only the
    held-goal channel, and `_plan_w`'s channel stays intact.
  - **Delegate D-42 (launcher-1028)** rules that this read-side form satisfies D-37 rule 2. On the ACTION path it is
    behaviourally the write-side form, and it keeps the `pc` arm's `held_goal` record true. Two conditions:
    - R2a and R2c stay binding; failing either is VOID before the GO.
    - Every reader of the held goal or the plan channel that the stub does NOT rotate is disclosed here by name. Read
      at `1b6e53ab`, none of them reaches action selection by goal index:
      - `_lake_write(pattern, self.pfc.goal, action)` (`autonomous_loop.py:2163-2168`) gets the UNROTATED held goal.
        It is active on production, because `LiveBrain` passes `lake_wm=True` (`live_brain.py:87`). It is storage
        only: it writes a goal one-hot into the lake working memory after the action is chosen. Its read-back,
        `LiveBrain.recall`, feeds only the chat and intent replies (`live_brain.py:473`, `:850`) and a self-check
        (`:999`). The `wm_events` count it adds feeds `wm_load` (`live_brain.py:261`) at one write per tick in every
        arm, whatever the goal index.
      - The tick record's `"goal"` field (`autonomous_loop.py:2549`): telemetry.
      - `LiveBrain.goal` (`live_brain.py:315-317`). It feeds `goal_on` (`:267-268`), which reads whether a goal is held
        (the same in every arm, since the stub never empties the stack), and the chat reply text (`:440`).
      - The plan nudge reads `_planned_seq[0]`, the PLANNER's first action, not the held goal (`:1545-1551`).
      - `Brain._observe_self_wake` (`brain.py:3004-3005`) reads the UNROTATED held goal on every wake tick.
        - It passes it to the self-model (`self_model.observe(goal=...)`, `:3018`) and into the self-surprise vector
          (`goal / 10`, `:3068`), which the metacognition checks read.
        - Panel C found no route from there to action selection. So in the `pc` arm the self-model sees the brain's own
          goal while behaviour sees the rotated one. That is a disclosed asymmetry of the read-side form, off the
          action path.
      - `brain.py:255` checks only whether the tick record has a `"goal"` key (region tagging), which is the same in
        every arm.
  - **Validity for every `pc` run:** `pc_stub_calls == pfc_bias_calls > 0`, with 600 ticks. Anything else is a harness
    red. The equality catches a partly bypassed stub; foreign-instance calls are already excluded by the scope.
- **Controls at look 1** (E1c section 3, with new seeds):
  - `repeat` on 23000 (0 action flips, recorded series identical);
  - `pc` on 23000 and 23001 (flips > 0).
  - A failed one VOIDs the campaign.
  - **The controls run FIRST in look 1's order:** `real_23000`, `real_23000` repeat, `pc_23000`, `real_23001`,
    `pc_23001`. The driver then runs the existing compare on that partial look, into its OWN report
    (`reports\controls-look1.md`), so the full look-1 report cannot overwrite it.
    - **The driver continues ONLY when the partial look's disposition is PENDING** (`e1c_disposition`,
      `rpe_charge.py:1750-1751`) and its pending items are exactly the expected set: every look-1 run except the five
      controls. VOID, a missing line, other pending items, a crash or unexpected text all halt it. A VOID costs 5 runs,
      not about 133.
    - The partial read is not a look: it has no boundary, spends no alpha and makes no decision.
  - **The pilot under controls-first.** The pilot T is the first campaign run's (`real_23000`), as E1c section 10 keeps
    the first run's T. P per arm comes from `pilot_watch -Seed 23000` on 23000's `real` and `pfrozen` runs, whenever
    each runs.
  - **Residual risk, stated.** R2c's 64/64 bounds the per-seed blind rate at about 4.6% (one-sided 95%). So at that
    bound the two-seed look-1 pc VOIDs about 9% of the time, down from about 26% under the old pc.
- **A new per-call record, `p["held_goal"]`** (every arm; harness only, no brain change).
  - After each `_pfc_bias` call the wrapper appends `pfc.goal` next to `p["tick"]` (`rpe_charge.py:802-806`), or None
    while nothing is held. It is guarded: `pfc = getattr(slf, "pfc", None)`.
  - It is a pure property read (`prefrontal_executive.py:415-418`), so it uses no RNG and writes no state.
  - It sits in `p`, never in the per-tick `series` (600 entries), and it is never zipped with per-tick keys (the #5689
    m4 lesson).
- **Each result records its compiled extension:** the loaded `_native` module's file path and sha256 go in the tier
  block. The analysis requires ONE value across a tier's runs, as it already requires one tree sha
  (`rpe_charge.py:1351-1354`).
- **Each result must match its file name.** `load_run` checks that the record's `seed` and `arm` match the file's tag
  (with `pc_<s>` and `real_<s>_r2` mapped). A mismatch is a harness red. Today `load_run` (`rpe_charge.py:1147-1158`)
  checks only the tier, so a relabelled or stray file would enter a look silently.

## 4. The goal-frozen stratum (beside, decides nothing)

- **Definition.** A campaign seed is **goal-frozen** when its `real` run's `p["held_goal"]` takes one value after its
  first non-None entry.
  - Live, a goal never returns to None after its first write, because `maintain()` has no live caller. So this
    record-level definition and #5850's gate-level "no moves" read the same thing. Off the live path they could differ.
- **At each look**, the verdict line adds, beside the decision and with no bar or disposition:
  - the count of frozen seeds;
  - H-P1's t and k/n within the frozen stratum and within the moving stratum, labelled DESCRIPTIVE ONLY. The record
    makes no stratum-conditioned claim, so a subgroup read can never become a finding later (P8);
  - the count of seeds whose `real` and `pfrozen` frozen status disagrees.
- **Why descriptive only.** Frozen status is read after treatment: the goal path depends on actions, and `_plan_w`'s
  nudge moves actions.
- **Firewall, made mechanical.**
  - The look's disposition output must be byte-identical with `held_goal` present, stripped and shuffled. That is a
    red test (B2).
  - A missing or all-None `held_goal` prints "stratum n/a (held_goal missing on N seeds)". It never VOIDs and never
    blocks the verdict. A missing record is then a disclosed harness defect for the record, not a gate.
  - The stratum lines are fail-soft. B2's tests cover 0 frozen, 1 frozen, all-equal d_s and a missing `held_goal`.
- **Why the stratum is only beside.**
  - `_plan_w`'s nudge goes to the first action of `plan_sequence`'s beam search, which does not read the held goal.
    So H-P1 is defined on frozen seeds too. Excluding them would change the population the frozen design was sized on.
  - The stratum answers D-37 item 3's question: what H-P1 measures on goal-frozen seeds.
- **Expected:** about 14% of seeds, from section 2. That number is design input only. The observed count is reported
  as seen.

## 4b. Sample size (P1)

This section is restated here because the P1 amendment's gate reads each file at HEAD. It is unchanged from E1c 5.1,
5.1b, 5.3 and A3-1.

| Field | Value |
|---|---|
| Measured SD | 0.3097 (E1b production H-P1, n = 80; bootstrap 95% CI 0.206-0.400) |
| Claimed effect | DH = 0.035 |
| Power and test | 0.8, one-sided alpha .05 |
| Fixed-sample target | 485 (601 at the 80% upper SD 0.345) |
| Design | 11 looks of 64; Lan-DeMets O'Brien-Fleming spending |
| Harm rule | t > bound AND k >= ceil(n/2) |
| Futility | non-binding; it opens only at n = 640, and only where it would read powered (A3-1, D-25) |
| Hard cap | 704 |
| Simulated before data | type I and power on the zero-preserving shapes (E1c 5.1b/5.4; R5 re-checks on this tree) |

A futility stop or an uncrossed cap is NOT a null; P8 null-scale item 4 decides it.

## 5. Seeds (replaces E1c section 6)

- **Confirmatory block:** 23000-23703, in order. 23704-23999 stay claimed and unexposed. They are NOT a repair block
  (section 8).
- **Grep check, 2026-10-29, at `1b6e53ab`.**
  - 0 hits for 23000-23999 in specs, src and tests (892 `.md`/`.py` files).
  - The same scan finds 6 files for 21000-21999, so the detector fires.
  - The grep sees literal seeds only, never computed sub-seed streams. Those are listed next.
- **Not reused:**
  - E1c's 21000-21999 (VOID; its look-1 endpoint data decides nothing here);
  - L1b-STALE / D1's 22000-22999 (#5629, `d1-readout-staleness.md:536-537`, `:581-582`);
  - the calibration block 28000-28063;
  - E1b's 8-183 and E1's 0-7.
- **Internal sub-seeds** (E1c section 6's note, re-derived for this block):
  - `GridPlaceMap` builds six modules, mi = 0..5 (default `spacings` has six entries, `grid_place_map.py:75`). Module
    mi seeds its cells with run seed + 1000 x mi (`:116`).
  - **Outbound.** E1d's mi >= 1 streams are 24000-24703, and so on up to 28000-28703. mi = 5 covers the calibration
    run seeds 28000-28063.
  - **Inbound.**
    - D1's 22000-22999 at mi = 1 land on 23000-23999 (the whole block).
    - E1c's exposed 21000-21063 at mi = 2 seeded 23000-23063.
  - Each campaign's comparisons are within one seed, so none of these overlaps affects a result. Disclosed; no block
    moves.
- **Repeat control:** 23000.

## 6. Reds (changes to E1c section 7; each run before look 1 unless it says "at look 1"; a red that does not fire halts the campaign)

- **R1, R4, R4b, R5:** as E1c, on the GO tree.
  - R1 runs inside the GO commit's full `tests/test_rpe_charge.py` gate log.
  - R5 of record runs on the fresh stream `r5go-<tree>` under D-31.
- **The pre-GO pin file.** Step 5 (section 9) writes `lanes\e1d-plan-w-work\NATIVE.pin`, holding `TREE=<40 hex>`,
  `NATIVE_SHA256=<64 hex>` and the toolchain line. It also writes `TREE_SHA`, which `run_arm.ps1` reads.
  - `r2.ps1` checks every launch against `NATIVE.pin`.
  - The GO's TREE and NATIVE_SHA256 must equal `NATIVE.pin`'s.
- **The R2 tool.** `lanes\e1d-plan-w-work\r2.ps1` is in `DRIVER.sha256`.
  - It runs every R2 brain run through the bound `run_arm.ps1`, on the run tree at `NATIVE.pin`'s TREE.
  - Before every launch it checks the native pin.
  - It writes results ONLY to `lanes\e1d-plan-w-work\r2\`, and it refuses any target under `results\` or the run tree.
  - It writes `logs\r2.log`. Its header names TREE, DRIVER_SHA256 and NATIVE_SHA256, and it carries one line per
    step: `R2-0 PASS|FAIL`, `R2A PASS|FAIL`, `IDENTITY <k>/65 PASS|FAIL` (64 seeds plus 21001),
    `CLASSIFIER PASS|FAIL` (28001 frozen, 28000 mover) and `R2C <k>/64 PASS|FAIL`.
- **The comparator** for every R2 read is look 1's `flips()` on `series.action` over all 600 ticks
  (`rpe_charge.py:327-330`).
  - "Identical" means the parsed `series`, `v` and `p` are equal, minus `p["held_goal"]` where only one side has it.
  - It never compares raw bytes.
- **The tag check does not apply to R2's own reads.** The calibration and comparator files (`cal_<seed>`, `c0_`,
  `c1_`) carry no campaign arm in their names. `r2.ps1` loads them, and its own `r2\` outputs, with the record loader
  WITHOUT the campaign's tag check. That check guards only `results\production`.
- **R2-0, the comparator reds** (brain-free, run by `r2.ps1` before any brain run). Inputs: `c0_21001.json` and
  `c1_21001.json`, whose sha256 `r2.ps1` pins, refusing on a mismatch, plus `cal_21001.json` under `CAL-BASELINE.sha256`.
  - c0 against cal: 0 flips, "identical".
  - c1 against cal: 378 flips, "not identical".
  - c0 with `p["held_goal"]` planted on ONE side only: "identical", which exercises the drop branch.
  - c0 with one value perturbed in `v`, then in `p`, then in one non-action `series` key: each 0 flips and "not
    identical". This separates the full comparator from an actions-only one.
  - Any reading any other way halts R2.
- **R2a, the blind seed** (red): `real` and the new `pc` on seed 21001. The `pc` must flip (> 0 action flips) and pass
  the pc validity items (section 3). 21001 is where the old pc read 0.
  - The `real_21001` run also joins the identity set below.
- **R2b, in-campaign** (at look 1): the new `pc` on 23000 and 23001 flips > 0, run first (section 3).
- **R2c, reach across a population** (a GO gate; it reads REACH or liveness, not a graded sensitivity).
  - **Fresh baselines.** Run `real` on seeds 28000-28063 through `r2.ps1` on the GO tree and the pinned build.
  - **The 64-seed identity cross-check.**
    - Each fresh `real` must equal its `cal_<seed>.json` under the comparator. Those files are bound by
      `CAL-BASELINE.sha256` (66 lines, own sha256 `568d2fea8da4adfbaef27b4f99911b68b248e6b9a12a25a5cd5c5dd038695504`).
      A file that differs from the manifest refuses R2c.
    - Fresh `real_21001` (R2a) must equal `cal_21001.json` too.
    - ANY difference halts R2c, and the integrator diagnoses: a harness or build effect on behaviour. Before the GO it
      also voids the base-rate citation (section 2).
    - **The classifier's positive control** is free here: fresh 28001 must classify frozen, and fresh 28000 must
      classify mover. A misclassification is `CLASSIFIER FAIL`, which halts R2c like an identity difference: the
      stratum's instrument is then wrong, and the integrator diagnoses.
  - **The read.** Run the new `pc` on each of the 64 seeds through `r2.ps1`, and compare each with its FRESH `real`.
    - Each run must pass the pc validity items.
    - Each seed's flip count and first-flip tick are recorded beside, with the 9 known frozen seeds called out, so a
      near-blind seed (1-2 flips) shows.
  - **The GO requires 64/64 seeds with flips > 0.** One blind seed falsifies "this control cannot be blind". A 64/64
    pass claims a blind rate below about 4.6% (one-sided 95%, 3/64) on the calibration population, not a rate of zero.
    - The integrator then diagnoses before any GO.
    - A replacement pc needs a dated addendum AND a fresh calibration block with its own manifest-bound baseline: a pc
      re-read on these 64 seeds would be in-sample.
- **R2-dry, the driver's look step end to end** (brain-free, run BEFORE any R2 brain run, so a driver fix it forces
  costs no R2 re-run).
  - **Where.** A scratch work folder `lanes\e1d-plan-w-work\dry\`, with its own `results\`, `reports\` and `logs\`. It
    refuses to run if its results path resolves under the real `results\` or the run tree.
  - **Input.** A complete 64-seed look-1 folder (`real`, `pfrozen`, `pc` on s0 and s0+1, the `real` repeat on s0) on
    seeds 23000-23063.
    - It is written by `rpe_charge`'s own `fixture_runs("noharm", ...)` (`synth_run` per arm, `rpe_charge.py:2247`) in
      E1d format. B2 gives it `p["held_goal"]` with a planted frozen count of 9.
    - The "noharm" fixture at n = 64 has no harm crossing, and futility does not open until 640. So its look-1
      disposition is CONTINUE, and plan decide prints `NEXT continue`.
    - The exact rendered disposition line is pinned as a string by B2's test, before the bind and its step-6b review.
      That string is R2-dry's bar.
    - `dry\logs` also holds a PLANTED `fixture.log` and `tier_red.log` in the pass form. They are disclosed as planted,
      because a real tier red builds a brain.
  - **The sequence.** The driver's own functions run in order: the controls-first partial compare and its
    PENDING-only continue, then plan owed, `compare_look 1`, plan decide, and the status quote. Brain launches are
    stubbed, as `driver_check` stubs them.
  - **Expected, line by line:**
    - every step rc=0;
    - the partial look reads PENDING, with exactly the owed items;
    - the full look's VALIDITY is valid;
    - the stratum line prints 9 frozen;
    - the disposition line equals B2's pinned CONTINUE string;
    - plan decide prints NEXT continue.
  - **Its red.** Overwrite ONLY the s0 `pc` record's `series.action` with the s0 `real`'s. The record keeps `arm` "pc"
    and its stub counts, so only "pc seed s0 flips == 0" fires. The partial look must read VOID on exactly that item,
    and the driver must halt before run 6.
  - **Reds for the other halt branches** of the PENDING-only continue. Each must halt before run 6:
    - the partial report with no disposition line;
    - one extra pending item;
    - one missing expected pending item;
    - a compare that exits non-zero.
  - **No live board.** R2-dry posts only to a scratch board folder (`dry\board`), never to the live board or thread E1D.
  - A seed/arm-against-tag mismatch (section 3) is a harness red. It joins the VALIDITY bad list, so the look is VOID.
  - R2's own reads parse every fresh E1d-format JSON with the same record loader (without the campaign's tag check, as
    stated above), so real-brain JSON parsing is exercised there.
    `synth_run`'s planted `held_goal` also puts the stratum path in the GO-commit gate log.
- **R3, repeat:** as E1c, on 23000 (at look 1, run first).
- **R6, the stub's vector** (unit test in `tests/test_rpe_charge.py`, brain-free). It REPLACES the old pc test at
  `:913-923`.
  - Pin the goal with `_gate_primary(0, float("inf"))` (no race, no `_native`), and assert goal == 0.
  - The arm is bound to that PFC (`ap.bind(SimpleNamespace(pfc=p))`), because the stub rotates only for the bound arm's
    PFC.
  - Under the stub, with strength 0.37: the stub returns a list, index 1 reads 0.37 and index 0 reads 0.
  - The wrap case at n = len(ACTIONS): goal 12 maps to index 0.
  - A goal-None instance stays all zeros.
  - A second instance on the same class passes through unrotated and uncounted.
  - After the arm's `__exit__`, the original method is back.
  - R6 tests the VECTOR. R2a and R2c test behaviour.
- **GO binding** (`campaign.sh`).
  - The GO names `R2_LOG=<path>`, checked like `check_gate_log`.
    - Its header must carry the GO's TREE, DRIVER_SHA256 and NATIVE_SHA256, which must also equal `NATIVE.pin`'s.
    - It must hold all five pass lines: `R2-0 PASS`, `R2A PASS`, `IDENTITY 65/65 PASS`, `CLASSIFIER PASS`,
      `R2C 64/64 PASS`.
  - **The results guard, at EVERY start and every look.** `campaign.sh` refuses to launch unless every JSON in
    `results\production` has a DONE row in `logs\runs.log`.
    - `run_one`'s resume rule, which already moves an undone run's log and home aside as `*.attempt-<stamp>`, also
      moves that run's JSON aside, so a run that died before DONE never blocks a resume.
    - It gets a `driver_check` case, plus a mutant that checks only once.
  - The R2 runs and the GO are on ONE tree, so `git diff --name-only <R2 tree> <GO tree>` is empty by construction.
  - A rebuilt `_native` re-runs R2a and R2c.
  - Each check gets a `driver_check` case and a mutant.

## 7. Driver and binding (changes to E1c addenda 2-5; delegates D-35, D-37, D-38)

- **Folders and names.**
  - The work folder is `lanes\e1d-plan-w-work`; the run tree is `lanes\e1d-plan-w-run`, detached at the GO's TREE.
  - The board names are `e1d-b3` and thread `E1D`, and the look posts say "E1d".
  - D-31's no-GO addendum check reads THIS file (`campaign.sh`'s spec path). Its mutant red: an addendum to E1c's file
    must not unlock an E1d no-GO.
  - `render_e1c` prints the pre-registration id ("E1d (T124)") in its header.
- **The bound set** (`DRIVER.sha256`): E1c's set, re-pathed, plus `r2.ps1`, `pilot_watch.ps1` and `tree_peak.ps1`.
  - It is bound BEFORE the R2 runs, because R2 runs through it.
  - This changes E1c A5-2's "out of band" wording. The watcher still decides no verdict.
  - Staged with reds at #5848.
- **The compiled `_native`.**
  - Its build is not byte-reproducible (`native-repro/repro.log`), so it is built in the run tree at TREE BEFORE R2,
    and its sha is pinned. R2 runs on it, and the GO's `NATIVE_SHA256` must equal R2's.
  - `campaign.sh` refuses at the start, at every look and before EVERY launch, on: no pin, no file, or other bytes.
    `r2.ps1` does the same.
  - The out-of-band `native_guard.ps1` is retired (D-38, P2). B's look-boundary hash may run as a monitor.
- **The look report:** `compare_look()` creates `reports\` before the compare tool writes into it. That is E1c look
  1's FileNotFoundError, fixed with its red.
- **Controls first** (section 3): `plan.py` orders look 1's controls first, and `campaign.sh` runs the partial-look
  compare after them into `reports\controls-look1.md`. It continues only on the expected PENDING set. Both get a
  `driver_check` case.
- **The perception packs are pinned at TREE:** `run_arm.ps1` points the two pack pins (`SNN_PERCEPTION_PACK`,
  `SNN_SHD_PACK`) at the run tree's git-tracked `models\`, not at the main checkout's.
  - That closes the campaign-window gap: the main checkout can change during about 10 h; TREE cannot.
  - The encoder checkpoints stay gitignored and hashed (ENC).
  - R2c's identity cross-check, fresh runs against calibration runs that used the main checkout's packs, shows the
    two give the same behaviour.
- **Pilot observer.**
  - The integrator starts `pilot_watch.ps1 -Seed 23000` before the campaign.
  - It selects each pilot run by its `-Out` result path, `results\production\<arm>_<seed>.json`; the tier-red probes
    write `logs\red\`.
  - A result from fewer than 5 samples is a MISS, never cited as P.
  - The decoy control is `PILOT-DECOY.log`. A5-2's planted-child allocation control is re-run on the new watcher
    before P is cited.
- **Stale text fixed in the bind:** `campaign.sh:137`'s "real_21000", the `plan.py` docstring, and
  `tests/test_rpe_charge.py:418` and `:761`'s E1c seeds.
- **Carried unchanged:**
  - D-31's R5 rule;
  - D-33's battery memory rule (no battery during the campaign);
  - D-34's serial regime, pilot measurements and 48 h line;
  - the single-instance lock;
  - the GO-commit gate log;
  - the stall watch and hysteresis.

## 8. Disposition (E1c section 8, with one change)

- HARM MET, NOT MET powered, and NOT MET underpowered: as E1c section 8, verbatim.
- **VOID:** no verdict.
  - **Repair** means a defect found while ZERO brain runs on 23000-23703 are exposed, that is before the first campaign
    run. It is fixed by a dated addendum on the same seeds.
  - After any exposed campaign run, a VOID routes ONLY to a new pre-registration on a fresh block the integrator
    claims on the board. Never 21000-21999, 22000-22999, 23000-23999 or 28000-28063.

## 9. Build rows and order (one change per commit; graph-exists before each; enforcer review of each build)

- **B0, E1c's closing record** (E1c section 12, below its freeze). VOID at look 1, with the results as seen and
  deciding nothing, and:
  - smoke 1 as a FAILED LAUNCH (environment cause: the compiled `_native` was missing), not a run;
  - the `NATIVE_SHA256` and toolchain line;
  - the `real_21000` observer miss and its 21003 supplement;
  - the R5 of record PASS, and the pilot's T = 26 s;
  - E1c's REPRODUCTION TREE is `1b6e53ab`. B3 moves `E1C_SEEDS`, so E1c is not reproducible from HEAD.
  - The D-37 item 4 P3 opening, as a record line: the planner gate never clears on 9/64 calibration seeds (14.1%), so
    the race is "live but inert" there. It is carried in the integrator's openings ledger (2026-10-28), and this row
    fixes nothing in it.
- **B1, the pc replacement** in `instruments/rpe_charge.py`:
  - the scoped `top_down_bias` rotation stub REPLACES the `_decide_spiking` stub, with the same arm name and counter;
  - the validity check becomes `pc_stub_calls == pfc_bias_calls > 0`;
  - R6 replaces the old pc test (red first: R6 fails against the old stub);
  - the module docstring and the dormancy_gate EXEMPT text that describe the pc are updated to match.
- **B2, `p["held_goal"]`, the native identity in the tier block, and the stratum lines** (fail-soft), with:
  - the guarded read;
  - the planted `held_goal` in `synth_run`;
  - the firewall red (present / stripped / shuffled give a byte-identical disposition);
  - tests for 0 frozen, 1 frozen, all-equal d_s and a missing `held_goal`;
  - the one-native-value analysis check, with its red.
- **B3, the campaign constants.**
  - `E1C_SEEDS` becomes 23000-23703: a single constant, not a second design copy (P2).
  - R4's test and `tests/test_rpe_charge.py:418` and `:761` are re-pointed at it.
  - `E1C`, `E1C_SEEDS`, `E1C_ARMS` and the `--e1c` flag keep their names. They name the design, which E1d shares
    unchanged, so there is no rename churn (P2).
  - `render_e1c`'s header names E1d.
- **B4, the cloud intake** (`lanes\REPORT-CLOUD-INTAKE-2026-10-28.md`):
  - the pinned heads in the stack: docs-sweep-r2 `d982f619` (on the enforcer's AGREE), winpaths-r1 `73aaeb56`,
    plan-amend `fcda0984` + `a85e0566` with the ceiling hunk dropped, restart-probe `a8893811`, and session-record
    `8dad59f4`.
  - Only `gates/docs_only_battery.py` is under `src/`, and it is on the section 2 allow-list.
- **B2 also adds** the `load_run` seed/arm-against-tag check, with its red.
- **The driver bind** (out of git, `lanes\e1d-plan-w-work`):
  - `r2.ps1` (with `NATIVE.pin`, the pinned c0/c1 hashes and its five pass lines);
  - R2-dry;
  - controls first, with the PENDING-only partial-look check;
  - the first-launch guard;
  - the run-tree pack pins;
  - names, the spec path and the stale text;
  - each with `driver_check` cases and mutants, including one per `R2_LOG` pass line.
- **Order.**
  1. The draft review (panel and enforcer), then the freeze commit.
  2. B0-B4.
  3. The panel (3 lenses) on the stack, then the fold.
  4. ONE battery, the closing record, push.
  5. Create the run tree at TREE, build `_native` there, and write `NATIVE.pin` and `TREE_SHA`.
  6. Bind `DRIVER.sha256` (with `r2.ps1`).
  6b. **The bind's review.** The three panel lenses and the policy-enforcer read the bound driver set and AGREE on its
      exact hashes (the `DRIVER.sha256` manifest sha), as every E1c bind was read (E1c addendum 2; A4-3's five AGREEs).
      - Nothing in step 7 or later runs before those AGREEs.
      - Any fold re-binds and is read again.
  7. R2-dry (brain-free).
  8. `r2.ps1`: R2-0, R2a, the identity cross-check and classifier, then R2c's read.
  9. The GO-commit gate log.
  10. The GO (TREE, DRIVER_SHA256, NATIVE_SHA256, GATE_LOG, R2_LOG).
  11. R5 of record, the pilot (with `pilot_watch -Seed 23000` running), and the looks.

## 10. Cost and resources (changes to E1c section 10)

- **Campaign:** E1c's pilot measured T = 26 s per run, serial.
  - The expected stop is about 9.5 h and the cap 10.2 h (E1c A5-3 arithmetic, at T = 26 s).
  - Under 48 h, so serial (D-34).
- **Before the GO (R2):** 130 runs, about 57 min.
  - R2a: 2 runs.
  - R2c: 64 fresh `real` and 64 `pc`.
  - One native build, about 10 s.
- **Resources:** `resource:timing` is held for all brain runs. No battery and no other brain campaign runs beside them.

## 11. Fold record and what stays open

- **Fold 1** (A #5858, B #5859, C #5860, enforcer #5857). Each finding, and where it went:
  - **Binding and order (A M1, B M1, C M2, enforcer M1):** sections 6 (GO binding, R2 tool), 7 (native) and 9
    (order).
  - **Enforcer m1** (D1's 22000-22999, the inbound mi = 1 overlap): sections 5 and 8.
  - **Enforcer m2** (where the P3 opening lives): section 9 B0.
  - **Enforcer m3** (the 4.6% bound): sections 3 and 6.
  - **Enforcer n1** (descriptive only): section 4.
  - **Enforcer n2** (E1c's reproduction tree): section 9 B0.
  - **R2c vacuity and comparator (C M1, A m5, B m7, C m8):** section 6 (fresh baselines, the 64-seed identity
    cross-check, R2-0, per-seed flip counts, the residual stated, in-sample replacement).
  - **The look step end to end and `held_goal` placement (C M3, B m1, B m2):** sections 3, 4 and 6 (R2-dry, `p`, the
    guarded read, `synth_run`).
  - **The stratum firewall and its post-treatment nature (A m3, A m4):** section 4.
  - **P1 section (A m2):** section 4b.
  - **D-31 spec path, names and stale text (A m1, C m2, B n2, C n3, B m8):** sections 7 and 9.
  - **One native per tier (C m3):** sections 3 and 9 (B2).
  - **Seeds and VOID routing (A m7, C m4, A n3):** sections 5 and 8.
  - **The brain-code boundary and the packs (B m5, C m5):** section 2.
  - **D-37 wording (A m6):** section 3, disclosed.
  - **The stub's reach and scoping (B m4, C m1):** section 3.
  - **Controls first (C m6):** sections 3 and 7.
  - **Build identity wording (C m7):** section 2.
  - **R6 (A n4, B m6, B m3, C n1):** section 6.
  - **A n1, A n2, B n1, C n2:** sections 1, 3 and 4.
  - **B n3** (the stale `autonomous_loop.py:1427-1429` comment, which says the margin always clears): carried as an
    opening, not edited here. Any `src` change outside the allow-list would void the D-38 citation.
- **Fold 2** (fold-1 delta reads: enforcer AGREE #5865, A AGREE #5868 with nits, B CHALLENGE #5870, C CHALLENGE
  #5871):
  - **R2-dry buildability, pass bar and order (C D1, B N1, C d4):** section 6, R2-dry (synth input, named scratch
    folder, expected lines, red); section 9 (it runs before R2).
  - **Stray or relabelled results (C D2):** section 3 (`load_run` tag check); section 6 (R2 output folder, first-launch
    guard).
  - **The pre-GO pin (C d1, B nit):** section 6 (`NATIVE.pin`).
  - **R2_LOG lines (C d2, B nit, A nit):** section 6 (five pass lines; CLASSIFIER FAIL halts).
  - **Comparator reds and binding (C d3, B N2, A nit):** section 6 (R2-0).
  - **The partial-look stop fails open (C d5):** section 3 and section 7 (PENDING-only, its own report, not a look).
  - **B nits:** the foreign-call wording and the 9% bound wording (section 3), R6's bind (section 6), the pack pins
    (section 7).
  - **A nits:** the pilot under controls-first (section 3), 130 runs (section 10).
  - The D-37 acknowledgement of the read-side pc is requested from the delegate. Its ruling is cited at the freeze.
- **Fold 3** (fold-2 delta reads: A AGREE #5876, B AGREE #5879 with n4 and n5, C CHALLENGE #5881, and the
  delegate's ruling D-42):
  - **D-42** (the read-side pc satisfies D-37 rule 2, with its conditions): section 3, cited, with every un-rotated
    reader of the held goal disclosed by name.
  - **B n4** (planted fixture and tier logs in `dry\`) and **B n5** (overwrite only `series.action`; a tag mismatch is
    a VOID item): section 6, R2-dry.
  - **C M** (the new driver bytes run unreviewed): section 9, step 6b.
  - **C e1** (R2-dry's red keeps `arm` "pc"; reds for the other halt branches): section 6.
  - **C e2** (the results guard at every start and look; a dead run's JSON moved aside): section 6.
  - **C nits** (R2's reads outside the tag check; a scratch board for R2-dry): section 6.
- **Fold 4** (fold-3 delta reads: A AGREE #5884, B AGREE #5886 with one wording nit, C CHALLENGE #5888):
  - **C MINOR** (a missed un-rotated reader): `Brain._observe_self_wake` and `brain.py:255` added to section 3's D-42
    list; "on the ACTION path" added.
  - **C NIT** (R2-dry's fixture kind and exact disposition): section 6 names `fixture_runs("noharm")`, CONTINUE, and
    B2's pinned string.
  - **B nit** (R2's reads and the tag check): section 6 wording.
- **Freeze.** Fold 4 (`3e11db18`) was AGREEd by A #5890, B #5895, C #5893 and the enforcer #5894. The freeze commit
  adds only:
  - the title and status change;
  - panel B #5895's wording nit on `brain.py:255`;
  - this line;
  - the tasks.md T124 row.
- **Carried to the step-6b bind review:** panel C #5893's note on R2-dry's all-zero fixture data.

## 12. Record

(empty until the campaign's results; appended below the freeze, never edited above it)

## Addendum 1 (2026-10-29, pre-data: no brain run on 23000 and up; before the bind and the GO): record identity, R3's nonce, a second R2-dry look

Appended below the freeze. Nothing above is edited. This addendum comes from the build reviews of B1-B3:
- build-ra #5911 and build-rb #5912, which AGREE;
- the policy-enforcer #5909, which AGREEs;
- the delegate's ruling that build-rb MINOR-1 must be fixed before any GO.
It changes no endpoint, threshold, seed, look or disposition.

**A1-1. Every result record carries its own identity, and R3 requires a fresh run.**
- At build head `38469120`, a byte-copy of `real_23000.json` saved as `real_23000_r2.json` passes `load_run`'s tag
  check AND R3 (0 flips, identical series): the record held no tag of its own. A control that a file copy satisfies
  measures nothing (P8: a control that is not the method itself).
- **The record.** Each record written by `rpe_charge run` now carries `arm`, `seed`, its out-file tag (the `--out`
  file's stem) and a per-run nonce (uuid4, pid, start time).
- **The loader.** `load_run` refuses a record whose embedded tag differs from its file name. For this design, a record
  with no embedded identity is refused too.
- **R3.** The repeat's nonce must DIFFER from `real_23000`'s, and its recorded series must stay identical.
- **Its red:** a byte-copy repeat FAILS R3.

**A1-2. The native check's scope (section 3).** "ONE value across a tier's runs" binds THIS design.
- Here, a record without `tier.native`, or two values in a tier, is VOID, as built.
- Legacy designs (E1, E1b) whose records carry no native value print "native n/a" and are not VOIDed. A mix of present
  and absent, or two values, stays VOID.
- This keeps E1b's compare reproducible at HEAD. E1c reproduces from `1b6e53ab` (section 9, B0).

**A1-3. R2-dry runs TWO looks (section 6).**
- The first is `fixture_runs("noharm")`. Its d_s are all zero, so its statistics are degenerate (t 0, SD 0), with
  CONTINUE and the pinned line.
- The second is a non-degenerate kind (named in the fold report), whose d_s are non-zero and not all equal, on
  23000-23063 with the planted frozen count 9. Its rendered look-1 disposition line is pinned as a string by the
  fold's test, and that string is the second bar.
- The second look exercises the stratum's t and k/n on real numbers, so a seed-pairing bug in the stratum code cannot
  hide behind zeros (build-rb MINOR-4; panel C's note for the step-6b bind review).
