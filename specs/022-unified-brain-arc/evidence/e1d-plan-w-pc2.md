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

## Addendum 2 (2026-10-29, pre-data: no brain run on 23000 and up; before the battery, the push and the bind): the stack panel's in-tree items, and what the bind owes

Appended below the freeze and addendum 1. Nothing above is edited.
- The stack panel on `ba872834`: stack-a #5934 AGREE, stack-b #5956 AGREE, stack-c #5933 CHALLENGE (a test-only MAJOR).
- TREE is the pushed head, so in-tree items cannot wait for the bind: they land here and in the stack fold E1D-SFOLD.
- No endpoint, threshold, seed, look or disposition changes.

**A2-1. Erratum to A1-1.** Where A1-1 says `load_run` "refuses" a record whose embedded tag differs from its file
name, or (for this design) a record with no embedded identity, the build marks that record a harness red that joins
VALIDITY's bad list. The LOOK is then VOID. `load_run` itself never raises on it.

**A2-2. R2-dry's second look reads HARM MET (A1-3).** The non-degenerate fixture (kind "harm") crosses the harm boundary
at look 1, as planted.
- Its pinned line is HARM MET, and `plan decide` prints `NEXT stop`.
- That stop IS that look's pass. The driver's dry sequence expects `NEXT stop` on the second look and `NEXT continue`
  on the first.
- A pairing mutant in its test proves the pin can see a seed-to-record mix-up (E1D-SFOLD S4).

**A2-3. The classifier R2c's CLASSIFIER check calls** is the module-level `goal_frozen(run)` in `rpe_charge.py` (lifted
from `goal_stratum` by E1D-SFOLD S2). That is the stratum's own instrument, not a copy.

**A2-4. Proof before R2a, inside the GO-commit gate** (E1D-SFOLD).
- S1: a slow real-brain test runs the `pc` arm on the no-encoder brain and asserts:
  - the stub's scope matches the arm's own PFC (`pc_stub_calls == pfc_bias_calls > 0`);
  - the record's identity tag;
  - the loaded native file and sha256;
  - `held_goal`'s length.
  Its red forces the scope check false.
- S3: a test pins the five-control partial look as PENDING, with exactly the expected pending set and no bad item.

**A2-5. Owed by the driver bind** (out of tree; read at step 6b):
- `r2.ps1` checks, before the GO, that every fresh R2 record carries `identity.tag` equal to its file name and that
  the nonces are pairwise distinct (build-ra #5924);
- the bind asserts that the analysis's one native sha256 equals `NATIVE.pin`'s (stack-c);
- `run_one`'s resume rule moves an undone run's JSON aside as `<tag>.json.attempt-<stamp>` (not `*.json`), so the
  results guard never refuses a resume (stack-c);
- `reds.sh` uses seed 23000 (stack-b);
- no manifest hashes fixture JSON bytes, which now carry a random nonce (build-ra);
- disclosed: no dry look exercises the tail census (|d| > 0.4) (build-rb).

## Addendum 3 (2026-10-29, pre-data): the pc validity equality binds this design only (E1D-SFOLD S5), and r2.ps1 passes it

Appended below addendum 2. Nothing above is edited.

**A3-1. Scope.**
- B1's validity rule `pc_stub_calls == pfc_bias_calls > 0` (section 3) binds THIS design.
- Legacy designs (E1, E1b) keep their original rule (`stub calls==0` is the harness red), because their pc records come
  from the old `_decide_spiking` stub, which counted about 25 race calls per `_pfc_bias` call.
- Found by re-running E1b's last production compare (look 19) at the arc head on E1b's own result files:
  - before S5 it read VOID ("pc_8: stub calls=7475 != pfc_bias_calls=299");
  - after S5 it reads ALL HOLD, with decision lines identical to E1b's recorded report.

**A3-2. Owed by the bind** (stack-c, on #5962): `run_problems`' default design does not apply the equality, so
`r2.ps1`'s R2a and R2c reads must pass this design (`design=E1C`) explicitly. Otherwise a stub that fires but misses
some calls would pass. It gets a `driver_check` mutant: the default design must fail to catch a partial bypass.

**A3-3. Erratum to A2-2** (stack-a #5969 m1).
- A2-2 says the pairing mutant proves the second dry look's pin "can see a seed-to-record mix-up". It proves less: the
  pin detects a ONE-SEED shift of the d values against their seeds (E1D-SFOLD S4).
- It does NOT detect these, because the harm fixture's d values repeat with period 5 and the 9 frozen seeds are
  planted first (`S4-red.log`: a reversal reproduces the pinned line exactly):
  - a reversal;
  - a shift by a multiple of 5;
  - a positional mix-up confined to the first 9 seeds.
- These are the dry pin's disclosed limits. The live campaign pairs by seed key, not by position.

## Results (campaign of record; DRAFT while the campaign runs)

Append-only record of the E1d campaign (T124). Written 2026-10-29 by lane e1d-record while the campaign runs. Every
number below is quoted from the file named beside it, never recomputed. Nothing here is a verdict: the decision of
record at each look is the line the campaign's compare tool printed, and a section marked OWED is empty on purpose.

### R0. Campaign identity

The GO line, board #6181 (2026-10-29 11:07:59), written verbatim to `lanes/e1d-plan-w-work/GO`:

```
GO TREE=c4c10b285f1741aa798d5ee9c5296ae371fabecf DRIVER_SHA256=6696ddbcb54e404d571ec7767908ba6bc21548ccaecf8b4ef6564e6e8c2d2f7e GATE_LOG=D:\AI-project\lanes\engine-1024\e1d-go-row\VERIFY-c4c10b285f1741aa798d5ee9c5296ae371fabecf.log NATIVE_SHA256=fb25d9a9bbc7e9e889721e46d952054b59150ab9d2a5644757caf7e97368110b R2_LOG=D:\AI-project\lanes\e1d-plan-w-work\logs\r2.log BATTERY_TREE=ec4df693684983b7c6dd4f8e030fe3947ce9183d NATIVE_RUSTC=1.96.0+ac68faa20 NATIVE_CARGO=1.96.0+30a34c682 NATIVE_MATURIN=1.15.0
```

| Field | Value (from #6181) |
|---|---|
| TREE | c4c10b285f1741aa798d5ee9c5296ae371fabecf |
| DRIVER_SHA256 | 6696ddbcb54e404d571ec7767908ba6bc21548ccaecf8b4ef6564e6e8c2d2f7e |
| NATIVE_SHA256 | fb25d9a9bbc7e9e889721e46d952054b59150ab9d2a5644757caf7e97368110b |
| GATE_LOG | `D:\AI-project\lanes\engine-1024\e1d-go-row\VERIFY-c4c10b285f1741aa798d5ee9c5296ae371fabecf.log` |
| R2_LOG | `D:\AI-project\lanes\e1d-plan-w-work\logs\r2.log` (sha256 c4a331a0, per #6175) |
| BATTERY_TREE | ec4df693684983b7c6dd4f8e030fe3947ce9183d |

From #6181's RECORD: the GO-commit gate was "209 passed in 292 s, ruff + format clean". Disclosed there: gate run 1 was
RED (18 failed / 191 passed, `ImportError: cannot import name '_native'`, no compiled `_native` in the fresh gate
worktree); it was fixed by copying the pinned pyd (fb25d9a9), so the gate tested the campaign's exact binary. D-52: the
run tree c4c10b28 and the battery-verified tree ec4df693 differ only in `tools/doc_reader_map.json` and
`tools/policy_certification.json`. The campaign started 2026-10-29T11:08:12-05:00 (`logs/campaign.console` line 1).

### R1. Pre-steps of record

- **Tier reds** (`logs/tier_red.log`, run 11:08:19-05:00, `logs/campaign.console` line 3):
  - TR1 "tier=production pins=no-encoder ... REFUSED tier=production (preflight): SNN_EPROP_CKPT=D:\AI-project\lanes\e1d-plan-w-work\NO-ENCODER\eprop_nmnist.pt missing rc=3 result_file_exists=False"
  - TR2 "tier=no-encoder pins=production ... REFUSED tier=no-encoder (preflight): SNN_EPROP_CKPT=D:\AI-project\spiking-neural-data-lake\models\eprop_nmnist.pt is reachable rc=3 result_file_exists=False"
- **Fixture** (`logs/fixture.log`, three lines, as quoted):
  - "fixture harm H-P1 MET at look 1 (8 seeds) | H-V1 CONTINUE at look 1 of 19, n=8 (no verdict): next look n=16"
  - "fixture noharm H-P1 CONTINUE at look 1 of 22, n=8 (no verdict): next look n=16 [degenerate: zero variance: H-P1, arm pfrozen] | H-V1 CONTINUE at look 1 of 19, n=8 (no verdict): next look n=16"
  - "fixture void H-P1 VOID (control failed) | H-V1 VOID (control failed)"
  - Note: the fixture lines name an "H-V1" endpoint and "look 1 of 22" or "of 19", which are not E1d section 4b's 11-look
    H-P1 design. They are quoted as logged; the integrator reads the fixture's own design.
- **R5 of record** (`logs/r5.log`, start 2026-10-29T11:08:19-05:00, tree c4c10b28, fresh stream `r5go-c4c10b285f17`):
  - "R5 instrument check: n=80 mean=-0.00347 SD=0.3097 (want n=80, mean -0.00347, SD 0.3097)"
  - "R5 OK  shift harm 0.8865 vs 0.8855 +- 0.0259"; "R5 OK  up harm 0.8090 vs 0.8145 +- 0.0317"
  - "R5 OK  null futility-powered 0.7375 vs 0.7450 +- 0.0355"
  - "R5 OK  null harm (type I) 0.0029 + 2 SE = 0.0037 <= 0.005"; "R5 OK  null INCONCLUSIVE 0.0050 <= 0.0131"
  - "R5 OK  up harm power 0.8196 - 2 SE = 0.8142 >= 0.8 (P1 floor)"
  - "R5 OK  code identity: the pinned-stream replay equals EXACT, bit for bit"
  - "R5 PASS", "rc=0"
- **Pilot T** (`logs/campaign.console` line 6, 11:10:01-05:00): "PILOT: T=27s -> T=27 TIMEOUT_S=600 FAULT_S=108 FLAT_MIN=3"
  (`logs/thresholds.env` carries the same four values). The pilot run is production/real_23000, rc=0, 27 s.
- **R2** (#6175, 2026-10-29T09:53:41-0500, claim=pass, `logs/r2.log` sha256 c4a331a0468c063afbc82335e6d51593b48c79d61ee48717d92ae5e062549f3c):
  "R2 DONE: ALL PASS (r2.ps1 pid 37584, 08:57:00 -> 09:53:10, about 56 min, 130 serial brain runs under HEAVY-RUN.lock
  + resource:timing". Its items: R2-0 PASS (7 comparator reds), R2A PASS (pc_21001 flips=378 first=4), IDENTITY 65/65
  PASS, CLASSIFIER PASS, RECORDS PASS, R2C 64/64 PASS (min 229 on 28032; max first-flip tick 24).

### R2. The reading rules (quoted, not paraphrased)

These are the frozen rules a look is read under. Nothing below is new text.

- **Section 8, dispositions** (E1d section 8: "HARM MET, NOT MET powered, and NOT MET underpowered: as E1c section 8,
  verbatim"). E1c section 8, verbatim:
  - "**HARM MET** at a look: The learned `_plan_w` trajectory is a measured harm on production at 600 ticks. P8
    null-scale item 4: 'a measured harm ... is deleted without a re-test'."
  - "**NOT MET at a futility or cap stop, POWERED** (mde <= 0.035): 'Powered: no harm of 0.035 or more on production at
    600 ticks.' `_plan_w` stays as it is."
  - "**NOT MET, UNDERPOWERED** (mde > 0.035; the design makes this rare): INCONCLUSIVE, with its mde recorded. Nothing
    changes."
  - E1d section 8's own **VOID**: "no verdict. **Repair** means a defect found while ZERO brain runs on 23000-23703 are
    exposed, that is before the first campaign run. It is fixed by a dated addendum on the same seeds. After any exposed
    campaign run, a VOID routes ONLY to a new pre-registration on a fresh block the integrator claims on the board."
- **The sequential design's per-look decision line** (E1c section 5, "Harm at a look", carried by E1d section 4b):
  "**Harm at a look** = t above the look's boundary AND k >= ceil(4n/8), meaning at least half of the seeds are
  positive." The boundary is "`rpe_charge.seq_boundaries` (Lan-DeMets O'Brien-Fleming, one-sided alpha .05)", per-look t
  "7.627, 4.639, 3.644, 3.094, 2.736, 2.480, 2.285, 2.130, 2.002, 1.896 and 1.804". Futility is "non-binding"; it "OPENS
  at n = 640" (E1d section 4b: "it opens only at n = 640, and only where it would read powered"). Hard cap 704.
  "A futility stop or an uncrossed cap is NOT a null; P8 null-scale item 4 decides it" (E1d section 4b).
- **The stratum** (E1d section 4, "beside, decides nothing"): "H-P1's t and k/n within the frozen stratum and within the
  moving stratum, labelled DESCRIPTIVE ONLY. The record makes no stratum-conditioned claim, so a subgroup read can never
  become a finding later (P8)". And: "The look's disposition output must be byte-identical with `held_goal` present,
  stripped and shuffled."
- **The tail census** (E1c section 4, "Beside, no bar"): "the descriptive tail census (|d| > 0.4 and the first
  divergence tick of those seeds)". The look report heads it "Tail census (|d| > 0.4; beside, no bar)".

### R3. Look 1 of 11 (n = 64)

Source: `lanes/e1d-plan-w-work/reports/compare-production-look1.md` (report line numbers below). Board record: #6200
(2026-10-29 12:12:11, "E1d look 1 done"; the report is quoted in it). The brief names #6196 for look 1; #6196 is a
`resource:timing` renew by e1d-b3, so the look-1 record is #6200.

- VALIDITY (line 141): "**VALIDITY: ALL HOLD**". Repeat arm (line 138): "action flips 0 first=None; recorded series identical:
  True". pc seed 23000: "action flips 383 (first 8)"; pc seed 23001: "action flips 430 (first 4)" (lines 139-140). Tree sha
  c4c10b285f1741aa798d5ee9c5296ae371fabecf, native sha256 fb25d9a9...68110b (lines 136-137).
- H-P1 (line 144): "t=1.042 (31/64 seeds > 0), SD=0.2315 -> **CONTINUE: no verdict (CONTINUE at look 1 of 11, n=64 (no
  verdict): next look n=128 | beside (no bar): k 31/64, trim10 +0.0266, HL +0.0231 [-0.0256, +0.0784], zeros 0.016,
  |d|>0.4 7)**"
- Decision of record (line 146): "look 1 of 11, n=64: t=1.042 vs boundary 7.627, 31/64 > 0 (need 32), conditional power
  0.988 (futility < 0.1, closed at this look) -> continue"
- Stratum, DESCRIPTIVE ONLY (line 145): "8/64 seeds goal-frozen; H-P1 within frozen t=-0.248 (3/8), within moving t=1.211
  (28/56); real/pfrozen frozen status disagrees on 7 seeds (pfrozen held_goal missing on 0)"
- Sensitivity (line 147): "H-P1 without degenerate seeds [23008]: t=1.042 (31/63)". Exact-zero (line 154): "n=64:
  exact-zero 0.016". E-P2 (line 150): "no stop of record yet".
- Tail census (lines 156-164): 7 seeds with |d| > 0.4: 23019 (+0.4390, tick 8), 23022 (+0.4384, 211), 23025 (+0.4629,
  150), 23041 (+0.7285, 38), 23049 (+0.5888, 49), 23058 (-0.6588, 46), 23061 (-0.4162, 197).
- Perception tier (line 166): "production (reported eprop-snn)". Wall (line 169): "total run_s over runs = 3450s (131
  runs)".

### R4. Look 2 of 11 (n = 128)

Source: `lanes/e1d-plan-w-work/reports/compare-production-look2.md`. Board record: #6263 (2026-10-29 13:21:23).

- VALIDITY (line 269): "**VALIDITY: ALL HOLD**". Repeat arm (line 266): "action flips 0 first=None; recorded series identical:
  True". pc seed 23000: "action flips 383 (first 8)"; pc seed 23001: "action flips 430 (first 4)" (lines 267-268).
- H-P1 (line 272): "t=0.542 (70/128 seeds > 0), SD=0.3379 -> **CONTINUE: no verdict (CONTINUE at look 2 of 11, n=128 (no
  verdict): next look n=192 | beside (no bar): k 70/128, trim10 +0.0198, HL +0.0229 [-0.0150, +0.0637], zeros 0.008,
  |d|>0.4 22)**"
- Decision of record (line 274): "look 2 of 11, n=128: t=0.542 vs boundary 4.639, 70/128 > 0 (need 64), conditional power
  0.773 (futility < 0.1, closed at this look) -> continue"
- Stratum, DESCRIPTIVE ONLY (line 273): "16/128 seeds goal-frozen; H-P1 within frozen t=0.442 (9/16), within moving t=0.453
  (61/112); real/pfrozen frozen status disagrees on 7 seeds (pfrozen held_goal missing on 0)"
- Sensitivity (line 275): "H-P1 without degenerate seeds [23008]: t=0.541 (70/127)". Exact-zero (lines 282-283): "n=64:
  exact-zero 0.016", "n=128: exact-zero 0.008". E-P2 (line 278): "no stop of record yet".
- Tail census (lines 287-308): 22 seeds with |d| > 0.4 (the 7 of look 1, plus 23065 +0.5326, 23068 +1.4218, 23069
  -0.5162, 23072 -0.4300, 23082 -0.5967, 23086 +0.6256, 23098 -1.2385, 23100 -0.7583, 23101 -0.5626, 23105 -0.4406,
  23109 +0.7346, 23118 +0.9959, 23120 -0.4359, 23121 +0.4562, 23126 -1.3311).
- Perception tier (line 310): "production (reported eprop-snn)". Wall (line 313): "total run_s over runs = 7232s (259
  runs)". Board #6263 states: "Look 3 started (n=192)."

### R5. Look 3 of 11 (n = 192)

Source: `lanes/e1d-plan-w-work/reports/compare-production-look3.md` (450 lines), sha256 `53d6b324cd5edcb3fdd4c0cc79a2077f14b9a3d155e26b612f38ad34adcd6a48`. Board record: #6394 (2026-10-29 14:24:40, e1d-b3, status): "E1d look 3 done: 387 runs done in total; per-run median 28s (tier production = eprop-snn); pauses so far 0". Report lines copied verbatim below, each prefixed by its line number in the report; no interpretation.

```text
L392: - tree sha(s) in this tier: ['c4c10b285f1741aa798d5ee9c5296ae371fabecf']
L393: - native sha256(s) in this tier: ['fb25d9a9bbc7e9e889721e46d952054b59150ab9d2a5644757caf7e97368110b']
L394: - repeat arm (real seed 23000 again): action flips 0 first=None; recorded series identical: True
L395: - pc seed 23000: action flips 383 (first 8)
L396: - pc seed 23001: action flips 430 (first 4)
L397: - **VALIDITY: ALL HOLD**
L400: - t=1.219 (101/192 seeds > 0), SD=0.3115 -> **CONTINUE: no verdict (CONTINUE at look 3 of 11, n=192 (no verdict): next look n=256 | beside (no bar): k 101/192, trim10 +0.0303, HL +0.0279 [-0.0023, +0.0638], zeros 0.016, |d|>0.4 30)**
L401: - stratum (DESCRIPTIVE ONLY, decides nothing): 26/192 seeds goal-frozen; H-P1 within frozen t=1.176 (14/26), within moving t=0.906 (87/166); real/pfrozen frozen status disagrees on 15 seeds (pfrozen held_goal missing on 0)
L402: - sequential decision of record: look 3 of 11, n=192: t=1.219 vs boundary 3.644, 101/192 > 0 (need 96), conditional power 0.880 (futility < 0.1, closed at this look) -> continue
L403: - sensitivity H-P1 without degenerate seeds [23008]: t=1.219 (101/191)
L406: - no stop of record yet
L412: - n=192: exact-zero 0.016
L414: ## Tail census (|d| > 0.4; beside, no bar)
L416: - seed 23019: d +0.4390, first divergence tick 8
L417: - seed 23022: d +0.4384, first divergence tick 211
L418: - seed 23025: d +0.4629, first divergence tick 150
L419: - seed 23041: d +0.7285, first divergence tick 38
L420: - seed 23049: d +0.5888, first divergence tick 49
L421: - seed 23058: d -0.6588, first divergence tick 46
L422: - seed 23061: d -0.4162, first divergence tick 197
L423: - seed 23065: d +0.5326, first divergence tick 44
L424: - seed 23068: d +1.4218, first divergence tick 32
L425: - seed 23069: d -0.5162, first divergence tick 52
L426: - seed 23072: d -0.4300, first divergence tick 71
L427: - seed 23082: d -0.5967, first divergence tick 60
L428: - seed 23086: d +0.6256, first divergence tick 149
L429: - seed 23098: d -1.2385, first divergence tick 87
L430: - seed 23100: d -0.7583, first divergence tick 322
L431: - seed 23101: d -0.5626, first divergence tick 49
L432: - seed 23105: d -0.4406, first divergence tick 76
L433: - seed 23109: d +0.7346, first divergence tick 163
L434: - seed 23118: d +0.9959, first divergence tick 103
L435: - seed 23120: d -0.4359, first divergence tick 154
L436: - seed 23121: d +0.4562, first divergence tick 28
L437: - seed 23126: d -1.3311, first divergence tick 12
L438: - seed 23130: d +0.5564, first divergence tick 120
L439: - seed 23141: d -0.6272, first divergence tick 9
L440: - seed 23142: d +0.7788, first divergence tick 44
L441: - seed 23144: d +0.4982, first divergence tick 56
L442: - seed 23151: d +0.5854, first divergence tick 75
L443: - seed 23153: d -0.5578, first divergence tick 20
L444: - seed 23157: d -0.4943, first divergence tick 125
L445: - seed 23177: d +0.4998, first divergence tick 20
L447: - perception tier of every number above: production (reported eprop-snn)
L450: wall: total run_s over runs = 10747s (387 runs)
```

### R6. Look 4 of 11 (n = 256)

Source: `lanes/e1d-plan-w-work/reports/compare-production-look4.md` (587 lines), sha256 `acf3eacc9fb2696f12d82c0fde768636846c5183a4d4f9579449d00782310e7c`. Board record: #6543 (2026-10-29 15:31:22, e1d-b3, status): "E1d look 4 done: 515 runs done in total; per-run median 29s (tier production = eprop-snn); pauses so far 0". Report lines copied verbatim below, each prefixed by its line number in the report; no interpretation.

```text
L520: - tree sha(s) in this tier: ['c4c10b285f1741aa798d5ee9c5296ae371fabecf']
L521: - native sha256(s) in this tier: ['fb25d9a9bbc7e9e889721e46d952054b59150ab9d2a5644757caf7e97368110b']
L522: - repeat arm (real seed 23000 again): action flips 0 first=None; recorded series identical: True
L523: - pc seed 23000: action flips 383 (first 8)
L524: - pc seed 23001: action flips 430 (first 4)
L525: - **VALIDITY: ALL HOLD**
L528: - t=1.283 (136/256 seeds > 0), SD=0.2986 -> **CONTINUE: no verdict (CONTINUE at look 4 of 11, n=256 (no verdict): next look n=320 | beside (no bar): k 136/256, trim10 +0.0292, HL +0.0270 [+0.0012, +0.0561], zeros 0.016, |d|>0.4 38)**
L529: - stratum (DESCRIPTIVE ONLY, decides nothing): 33/256 seeds goal-frozen; H-P1 within frozen t=0.273 (16/33), within moving t=1.260 (120/223); real/pfrozen frozen status disagrees on 20 seeds (pfrozen held_goal missing on 0)
L530: - sequential decision of record: look 4 of 11, n=256: t=1.283 vs boundary 3.094, 136/256 > 0 (need 128), conditional power 0.883 (futility < 0.1, closed at this look) -> continue
L531: - sensitivity H-P1 without degenerate seeds [23008, 23208]: t=1.283 (136/254)
L534: - no stop of record yet
L541: - n=256: exact-zero 0.016
L543: ## Tail census (|d| > 0.4; beside, no bar)
L545: - seed 23019: d +0.4390, first divergence tick 8
L546: - seed 23022: d +0.4384, first divergence tick 211
L547: - seed 23025: d +0.4629, first divergence tick 150
L548: - seed 23041: d +0.7285, first divergence tick 38
L549: - seed 23049: d +0.5888, first divergence tick 49
L550: - seed 23058: d -0.6588, first divergence tick 46
L551: - seed 23061: d -0.4162, first divergence tick 197
L552: - seed 23065: d +0.5326, first divergence tick 44
L553: - seed 23068: d +1.4218, first divergence tick 32
L554: - seed 23069: d -0.5162, first divergence tick 52
L555: - seed 23072: d -0.4300, first divergence tick 71
L556: - seed 23082: d -0.5967, first divergence tick 60
L557: - seed 23086: d +0.6256, first divergence tick 149
L558: - seed 23098: d -1.2385, first divergence tick 87
L559: - seed 23100: d -0.7583, first divergence tick 322
L560: - seed 23101: d -0.5626, first divergence tick 49
L561: - seed 23105: d -0.4406, first divergence tick 76
L562: - seed 23109: d +0.7346, first divergence tick 163
L563: - seed 23118: d +0.9959, first divergence tick 103
L564: - seed 23120: d -0.4359, first divergence tick 154
L565: - seed 23121: d +0.4562, first divergence tick 28
L566: - seed 23126: d -1.3311, first divergence tick 12
L567: - seed 23130: d +0.5564, first divergence tick 120
L568: - seed 23141: d -0.6272, first divergence tick 9
L569: - seed 23142: d +0.7788, first divergence tick 44
L570: - seed 23144: d +0.4982, first divergence tick 56
L571: - seed 23151: d +0.5854, first divergence tick 75
L572: - seed 23153: d -0.5578, first divergence tick 20
L573: - seed 23157: d -0.4943, first divergence tick 125
L574: - seed 23177: d +0.4998, first divergence tick 20
L575: - seed 23200: d -0.4239, first divergence tick 69
L576: - seed 23204: d +0.6167, first divergence tick 104
L577: - seed 23207: d +0.6531, first divergence tick 157
L578: - seed 23210: d +0.4490, first divergence tick 125
L579: - seed 23217: d -0.6207, first divergence tick 32
L580: - seed 23235: d -0.8695, first divergence tick 87
L581: - seed 23238: d -0.5417, first divergence tick 22
L582: - seed 23239: d +0.4039, first divergence tick 68
L584: - perception tier of every number above: production (reported eprop-snn)
L587: wall: total run_s over runs = 14417s (515 runs)
```

### R7. Look 5 of 11 (n = 320)

Source: `lanes/e1d-plan-w-work/reports/compare-production-look5.md` (724 lines), sha256 `26a33f5d6c52c1c9dec98df211bd6f5c4ed05177717cdbd0ac03ba0b77acc35a`. Board record: #6662 (2026-10-29 16:34:47, e1d-b3, status): "E1d look 5 done: 643 runs done in total; per-run median 29s (tier production = eprop-snn); pauses so far 0". Report lines copied verbatim below, each prefixed by its line number in the report; no interpretation.

```text
L648: - tree sha(s) in this tier: ['c4c10b285f1741aa798d5ee9c5296ae371fabecf']
L649: - native sha256(s) in this tier: ['fb25d9a9bbc7e9e889721e46d952054b59150ab9d2a5644757caf7e97368110b']
L650: - repeat arm (real seed 23000 again): action flips 0 first=None; recorded series identical: True
L651: - pc seed 23000: action flips 383 (first 8)
L652: - pc seed 23001: action flips 430 (first 4)
L653: - **VALIDITY: ALL HOLD**
L656: - t=0.676 (164/320 seeds > 0), SD=0.3038 -> **CONTINUE: no verdict (CONTINUE at look 5 of 11, n=320 (no verdict): next look n=384 | beside (no bar): k 164/320, trim10 +0.0219, HL +0.0198 [-0.0038, +0.0449], zeros 0.019, |d|>0.4 46)**
L657: - stratum (DESCRIPTIVE ONLY, decides nothing): 38/320 seeds goal-frozen; H-P1 within frozen t=0.433 (19/38), within moving t=0.571 (145/282); real/pfrozen frozen status disagrees on 27 seeds (pfrozen held_goal missing on 0)
L658: - sequential decision of record: look 5 of 11, n=320: t=0.676 vs boundary 2.736, 164/320 > 0 (need 160), conditional power 0.668 (futility < 0.1, closed at this look) -> continue
L659: - sensitivity H-P1 without degenerate seeds [23008, 23208, 23292, 23308, 23312]: t=0.676 (163/315)
L662: - no stop of record yet
L670: - n=320: exact-zero 0.019
L672: ## Tail census (|d| > 0.4; beside, no bar)
L674: - seed 23019: d +0.4390, first divergence tick 8
L675: - seed 23022: d +0.4384, first divergence tick 211
L676: - seed 23025: d +0.4629, first divergence tick 150
L677: - seed 23041: d +0.7285, first divergence tick 38
L678: - seed 23049: d +0.5888, first divergence tick 49
L679: - seed 23058: d -0.6588, first divergence tick 46
L680: - seed 23061: d -0.4162, first divergence tick 197
L681: - seed 23065: d +0.5326, first divergence tick 44
L682: - seed 23068: d +1.4218, first divergence tick 32
L683: - seed 23069: d -0.5162, first divergence tick 52
L684: - seed 23072: d -0.4300, first divergence tick 71
L685: - seed 23082: d -0.5967, first divergence tick 60
L686: - seed 23086: d +0.6256, first divergence tick 149
L687: - seed 23098: d -1.2385, first divergence tick 87
L688: - seed 23100: d -0.7583, first divergence tick 322
L689: - seed 23101: d -0.5626, first divergence tick 49
L690: - seed 23105: d -0.4406, first divergence tick 76
L691: - seed 23109: d +0.7346, first divergence tick 163
L692: - seed 23118: d +0.9959, first divergence tick 103
L693: - seed 23120: d -0.4359, first divergence tick 154
L694: - seed 23121: d +0.4562, first divergence tick 28
L695: - seed 23126: d -1.3311, first divergence tick 12
L696: - seed 23130: d +0.5564, first divergence tick 120
L697: - seed 23141: d -0.6272, first divergence tick 9
L698: - seed 23142: d +0.7788, first divergence tick 44
L699: - seed 23144: d +0.4982, first divergence tick 56
L700: - seed 23151: d +0.5854, first divergence tick 75
L701: - seed 23153: d -0.5578, first divergence tick 20
L702: - seed 23157: d -0.4943, first divergence tick 125
L703: - seed 23177: d +0.4998, first divergence tick 20
L704: - seed 23200: d -0.4239, first divergence tick 69
L705: - seed 23204: d +0.6167, first divergence tick 104
L706: - seed 23207: d +0.6531, first divergence tick 157
L707: - seed 23210: d +0.4490, first divergence tick 125
L708: - seed 23217: d -0.6207, first divergence tick 32
L709: - seed 23235: d -0.8695, first divergence tick 87
L710: - seed 23238: d -0.5417, first divergence tick 22
L711: - seed 23239: d +0.4039, first divergence tick 68
L712: - seed 23262: d -0.5629, first divergence tick 108
L713: - seed 23271: d -1.3950, first divergence tick 46
L714: - seed 23283: d +0.5252, first divergence tick 35
L715: - seed 23291: d +0.4246, first divergence tick 139
L716: - seed 23295: d +0.5248, first divergence tick 54
L717: - seed 23302: d -1.1643, first divergence tick 56
L718: - seed 23309: d -0.6623, first divergence tick 28
L719: - seed 23319: d +0.4112, first divergence tick 59
L721: - perception tier of every number above: production (reported eprop-snn)
L724: wall: total run_s over runs = 17903s (643 runs)
```

### R8. Looks 6 to 11: OWED

Each is filled from its report with the same lines as R3 and R4 (the full set: VALIDITY, H-P1, sequential decision of record, stratum, sensitivity, exact-zero, E-P2 "no stop of record yet", the tail-census list, the perception-tier line and the wall line), verbatim with report line numbers. R5 to R7 carry that full set since round 3. Nothing is entered until the report exists.

| Look | n | Boundary t (E1c section 5) | Source report | State |
|---|---|---|---|---|
| 6 | 384 | 2.480 | `reports/compare-production-look6.md` | OWED |
| 7 | 448 | 2.285 | `reports/compare-production-look7.md` | OWED |
| 8 | 512 | 2.130 | `reports/compare-production-look8.md` | OWED |
| 9 | 576 | 2.002 | `reports/compare-production-look9.md` | OWED |
| 10 | 640 | 1.896 | `reports/compare-production-look10.md` | OWED (futility opens here, section 4b) |
| 11 | 704 | 1.804 | `reports/compare-production-look11.md` | OWED (the hard cap) |

Look 1 and 2 boundaries (7.627, 4.639) are in R3 and R4; looks 3 to 5 are quoted from their reports in R5 to R7. The per-look t list is E1c section 5's, quoted above; the mapping of the eleven values to looks 1-11 is by order.

### R9. Disposition: OWED

OWED. It is the integrator's, under section 8, from the stop of record (harm boundary crossed with k >= ceil(n/2),
futility at n = 640 where powered, or the uncrossed cap at n = 704), and it names its mde where P8 null-scale item 4
needs it. The campaign's own final report (`reports/` and `logs/campaign.console`) is the source. A futility stop or an
uncrossed cap is NOT a null.
