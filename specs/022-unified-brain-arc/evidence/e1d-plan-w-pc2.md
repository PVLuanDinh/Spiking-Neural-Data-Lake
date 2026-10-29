# E1d -- E1c's powered `_plan_w` harm test (H-P1), re-run with a positive control that cannot collide with the held goal (DRAFT pre-registration)

> **Status: DRAFT, pre-data.** Nothing below has been run in a brain on seeds 23000 and up. It freezes at the commit
> that drops "(DRAFT pre-registration)" from the title, after this DRAFT's review: three panel lenses and the
> policy-enforcer AGREE (as E1c froze on #5641). The panel on the built stack before its battery (section 9) is a
> separate, later review.
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

## 0. Raising-ambition record (self-apply, owner 2026-10-25)

- **Axis:** Depth. The question is E1c's; what failed was a control, so this row makes the controls excellent and does
  not widen the question.
- **Ladder.**
  - **R1:** replace the pc with a held-goal perturbation and red it on 21001 (D-37's floor).
  - **R2:** also MEASURE the new pc's sensitivity across a seed population before the GO, not on one seed. Record the
    held goal in every `real` run, so the goal-frozen stratum is OBSERVED in the campaign, not estimated.
  - **R3:** a dose-graded control (rotations k = 1..6, flips monotone in k).
- **Pick: R2.**
  - R3 is ruled out: the validity check needs a control that cannot be blind. It does not need a dose-response, and a
    dose series adds runs that no decision reads (P1: no run without a reader).
- **Range.**
  - Clinical assay validation: a positive control's sensitivity is shown on the population it guards, not on a single
    sample.
  - Metrology: calibrate across the operating range, not at a point.
  - Adversarial: the known blind seed 21001 is the red the new control must survive.
- **Outside judge** (none of these is the method itself): the `pfrozen` twin; the A/A repeat; the new pc's red on
  21001; its 64-seed sensitivity read (R2c); R1's brain-free compare fixture.

## 1. What E1d is

E1d is **E1c's frozen design**: `e1c-plan-w-powered.md` at `082b66f8`, plus its addenda 1-5, as they stand at
`1b6e53ab`. Only sections 2-9 below change it. Everything else holds verbatim, by reference to that file:
- E1c section 1 (question and licence);
- sections 4 and 5 (endpoints, and the sample-size design 5.3: cap 704, 11 looks of 64, LD-OBF, harm = t > bound AND
  k >= ceil(n/2), futility only where powered from n = 640);
- section 8 (disposition: route (iii) on harm; powered no-harm; INCONCLUSIVE);
- addenda 2-4's R5 rule (D-31, mechanical);
- addendum 5 (D-34: serial, the pilot's T/P/GPU, the 48 h line).

**E1c's look-1 data decides nothing here** (D-36). t = 1.531, 34/64 and SD 0.3542 were seen, and they appear in:
- no design number;
- no seed choice;
- no stratum;
- no threshold of this file.

## 2. Why E1c is VOID, and the measured fact this design rests on

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
  - Source: tree `1b6e53ab`, pyd `5c5b39c1`, instrument `diag_pc.py` `d0205e91` (record mode, transparent on 21000 and
    21001), summary `lanes\engine-1024\e1c-void-diag\CALIB-SUMMARY.txt`.
  - This file may cite the base rate only while its BRAIN CODE equals `1b6e53ab`'s (D-38 item 2). Brain code means
    everything under `src/snn_data_lake/` outside `instruments/` and `gates/`, plus `native/`.
  - **The GO proves it.** `git diff --name-only 1b6e53ab <GO tree> -- src native` must list ONLY these three paths:
    - `src/snn_data_lake/instruments/rpe_charge.py` (B1-B3);
    - `src/snn_data_lake/gates/dormancy_gate.py` (B1's EXEMPT text);
    - `src/snn_data_lake/gates/docs_only_battery.py` (winpaths-r1, B4).
  - Any other path there voids the citation, and the base rate is then re-measured on the GO tree before the GO.
- **Build identity** (D-38 item 1). Two different `_native` builds of one commit (`5c5b39c1`, `a6176948`) give
  byte-identical `series` and `v` on seeds 21000 and 21001. The bytes differ in metadata only. The pin stays anyway.

## 3. Arms and runs (changes to E1c section 3)

- **Arms, tier, harness settings:** as E1c (`real`, `pfrozen`; production; 600 ticks; held-out from tick 300).
- **The new positive control `pc` REPLACES the constant-winner pc** (P2: one control mechanism, never two side by side).
  - The `pc` arm is `real` plus one class-wide stub of `PrefrontalExecutive.top_down_bias`. The stub calls the real
    method, then rotates the vector by one place: `out[(i + 1) % n] = bias[i]`.
  - So on every tick where a goal is held, behaviour is biased toward (held + 1) mod 13 instead of the held goal. That
    differs from the brain's own held goal on every such tick, on every seed (D-37 rule 2). It lands AFTER the gate,
    which the old pc's race stub did not.
  - The stub counts its calls (`pc_stub_calls`). Zero calls is a harness red, as before.
  - The only live caller is `autonomous_loop.py:1530`. The other callers are self-check bodies.
- **Controls at look 1** (E1c section 3, with new seeds):
  - `repeat` on 23000 (0 action flips, recorded series identical);
  - `pc` on 23000 and 23001 (flips > 0).
  - A failed one VOIDs the campaign.
- **A new recorded series, `held_goal`** (every arm; harness only, no brain change).
  - After each `_pfc_bias` call the harness records `pfc.goal`, or None while nothing is held. That is about 300
    entries per 600-tick run: the probes counted 293-299 calls.
  - The harness already wraps `_pfc_bias` to record `plan_w`. This adds one read, and it writes nothing.

## 4. The goal-frozen stratum (beside, decides nothing)

- A campaign seed is **goal-frozen** when its `real` run's `held_goal` series (one entry per `_pfc_bias` call) takes
  one value after its first non-None entry.
- At each look, the verdict line adds, beside the decision and with no bar or disposition:
  - the count of frozen seeds;
  - H-P1's t and k/n within the frozen stratum and within the moving stratum.
- **Why it is only beside.**
  - `_plan_w`'s nudge goes to the first action of `plan_sequence`'s beam search, which does not read the held goal.
    So H-P1 is defined on frozen seeds too. Excluding them would change the population the frozen design was sized on.
  - The stratum answers D-37 item 3's question: what H-P1 measures on goal-frozen seeds.
- **Expected:** about 14% of seeds, from section 2. That number is design input only. The observed count is reported
  as seen.

## 5. Seeds (replaces E1c section 6)

- **Confirmatory block:** 23000-23703, in order. 23704-23999 stay unexposed for a repair.
- **Grep check, 2026-10-29, at `1b6e53ab`.** 0 hits for 23000-23999 in specs, src and tests (892 `.md`/`.py` files).
  The same scan finds 6 files for 21000-21999, so the detector fires.
- **Not reused:**
  - E1c's 21000-21999 (VOID; its look-1 data decides nothing here);
  - L1b-STALE's 22000-22039;
  - the calibration block 28000-28063;
  - E1b's 8-183 and E1's 0-7.
- **Internal sub-seeds** (E1c section 6's note, re-derived for this block):
  - `GridPlaceMap` builds six modules, mi = 0..5 (default `spacings` has six entries, `grid_place_map.py:75`). Module
    mi seeds its cells with run seed + 1000 x mi (`:116`).
  - So E1d's mi >= 1 streams are 24000-24703, 25000-25703, and so on up to 28000-28703.
  - mi = 5 covers the calibration run seeds 28000-28063, which are a different campaign. Each campaign's comparisons
    are within one seed, so nothing is affected. Disclosed; no block moves.
- **Repeat control:** 23000.

## 6. Reds (changes to E1c section 7; each RUN before look 1, and a red that does not fire halts the campaign)

- **R1, R4, R4b, R5:** as E1c, re-run on the E1d tree.
  - R1 runs inside the GO commit's full `tests/test_rpe_charge.py` gate log.
  - R5 of record runs on the fresh stream `r5go-<tree>` under D-31.
- **R2 is replaced by three items:**
  - **R2a, the blind seed** (red, before the GO): on the E1d tree, run `real` and the new `pc` on seed 21001. The
    `pc` must flip (> 0 action flips). 21001 is where the old pc read 0.
  - **R2b, in-campaign** (look 1): the new `pc` on 23000 and 23001 flips > 0, as section 3 states.
  - **R2c, sensitivity across a population** (before the GO, design gate).
    - **Baseline.** The 64 recorded `real` runs of calibration seeds 28000-28063, plus controls 21000 and 21001.
      - They are `lanes\engine-1024\e1c-void-diag\cal_<seed>.json`, bound by `CAL-BASELINE.sha256` (66 lines), whose
        own sha256 is `568d2fea8da4adfbaef27b4f99911b68b248e6b9a12a25a5cd5c5dd038695504`.
      - A baseline file that differs from the manifest refuses R2c.
    - **Identity check first** (2 runs).
      - On the E1d tree, through the bound `run_arm.ps1 -Arm real`, run 28001 (goal-frozen) and 28000 (a mover).
      - Each run's recorded `series`, minus the new `held_goal` key, and its `v` must be byte-identical to its
        `cal_<seed>.json`.
      - This shows that the E1d harness, with its added `held_goal` read, is behaviourally the same instrument as the
        one that recorded the baseline (the diag wrapper's transparency was shown on 21000 and 21001, #5845). A
        difference stops R2c, and the integrator diagnoses.
    - **The read.** Through the bound `run_arm.ps1 -Arm pc`, run the new `pc` on each of the 64 seeds. Compare each
      run's action series with its baseline's.
    - **The GO requires 64/64 seeds with flips > 0.** One blind seed falsifies the claim that this control cannot be
      blind. The integrator then diagnoses before any GO, and a new pc needs a dated addendum.
- **R3, repeat:** as E1c, on 23000.
- **New: R6, the stub reaches behaviour on a pinned goal** (unit test, brain-free). A `PrefrontalExecutive` holding
  goal 0 reads bias 0 at index 0 and `strength` at index 1 under the stub, and the original method again after the
  arm's `__exit__`.

## 7. Driver and binding (changes to E1c addenda 2-5; delegates D-35, D-37, D-38)

- **The work folder is `lanes\e1d-plan-w-work`; the run tree is `lanes\e1d-plan-w-run`.** The board names are
  `e1d-b3` and thread `E1D`.
- **The bound set** (`DRIVER.sha256`): E1c's set, re-pathed, plus `pilot_watch.ps1` and `tree_peak.ps1`, which move
  INTO the bound set.
  - This changes E1c A5-2's "out of band" wording. The watcher still decides no verdict.
  - Staged with reds at #5848.
- **The compiled `_native`.**
  - Its build is not byte-reproducible (`native-repro/repro.log`), so it is built in the run tree BEFORE the GO, and
    the GO pins `NATIVE_SHA256=<64 hex>`.
  - `campaign.sh` refuses at the start, at every look and before EVERY launch, on: no pin, no file, or other bytes.
  - The out-of-band `native_guard.ps1` is retired (D-38, P2). B's look-boundary hash may run as a monitor.
- **The look report:** `compare_look()` creates `reports\` before the compare tool writes into it. That is E1c look
  1's FileNotFoundError, fixed with its red.
- **Pilot observer:**
  - `pilot_watch.ps1` selects each pilot run by its `-Out` result path, `results\production\<arm>_<seed>.json`. The
    tier-red probes write `logs\red\`.
  - A result from fewer than 5 samples is a MISS, never cited as P.
  - Decoy control: `PILOT-DECOY.log`.
- **Carried unchanged:**
  - D-31's R5 rule;
  - D-33's battery memory rule (no battery during the campaign);
  - D-34's serial regime, pilot measurements and 48 h line;
  - the single-instance lock;
  - the GO-commit gate log;
  - the stall watch and hysteresis.

## 8. Disposition (E1c section 8, with one change)

- HARM MET, NOT MET powered, and NOT MET underpowered: as E1c section 8, verbatim.
- **VOID:** no verdict. Repair, or a new pre-registration on seeds the integrator claims fresh on the board. Never
  21000-21999, 22000-22039, 23000-23999 or 28000-28063.

## 9. Build rows (one change per commit; graph-exists before each; enforcer review of each build)

- **B0, E1c's closing record** (E1c section 12, below its freeze). VOID at look 1, with the results as seen and
  deciding nothing, and:
  - smoke 1 as a FAILED LAUNCH (environment cause: the compiled `_native` was missing), not a run;
  - the `NATIVE_SHA256` and toolchain line;
  - the `real_21000` observer miss and its 21003 supplement;
  - the R5 of record PASS, and the pilot's T = 26 s.
- **B1, the pc replacement** in `instruments/rpe_charge.py`:
  - the `top_down_bias` rotation stub REPLACES the `_decide_spiking` stub, with the same arm name and the same counter;
  - the module docstring and the dormancy_gate EXEMPT text that describe the pc are updated to match;
  - R6 as a test, red first: it fails against the old stub.
- **B2, the `held_goal` series and the frozen-stratum lines** in the compare report (beside, no disposition). A test
  covers each.
- **B3, the campaign constants:** `E1C_SEEDS` becomes 23000-23703, a single constant and not a second design copy (P2),
  with R4's test re-pointed at it.
  - `E1C`, `E1C_SEEDS`, `E1C_ARMS` and the `--e1c` flag keep their names. They name the design, which E1d shares
    unchanged, so there is no rename churn (P2). The driver files are re-pathed (#5848) and gain seed 23000 in `plan.py` and the
  controls.
- **B4, the cloud intake** (`lanes\REPORT-CLOUD-INTAKE-2026-10-28.md`):
  - the pinned heads cherry-picked in the stack: winpaths-r1, plan-amend with the ceiling hunk dropped, restart-probe,
    session-record @ `1dcf2639`;
  - docs-sweep-r2 once the enforcer reads its re-cut.
  - They are docs, tests and ONE gate file (`gates/docs_only_battery.py`, winpaths-r1). None touches brain code as
    section 2 defines it, so D-38 item 2 still holds.
- **Then:**
  1. the panel (3 lenses) on the stack;
  2. the fold;
  3. ONE battery;
  4. the closing record;
  5. push;
  6. then the runs before the GO: R2a (2 runs), R2c's identity check (2 runs) and read (64 runs), the GO-commit gate
     log, the native build and pin, the GO, R5 of record, the pilot and the looks.

## 10. Cost and resources (changes to E1c section 10)

- **Campaign:** E1c's pilot measured T = 26 s per run, serial.
  - The expected stop is about 9.5 h and the cap 10.2 h (E1c A5-3 arithmetic, at T = 26 s).
  - Under 48 h, so serial (D-34).
- **Before the GO:**
  - R2a: 2 runs, about 1 min;
  - R2c: 2 identity runs and 64 pc runs, about 29 min;
  - one native build, about 10 s.
- **Resources:** `resource:timing` is held for all brain runs. No battery and no other brain campaign runs beside them.

## 11. Open before the freeze

- The panel's three lenses and the policy-enforcer, on this draft.
- docs-sweep-r2: the cloud's re-cut of the POLICIES.md cell and README.md:235 (enforcer #5852 M1).

## 12. Record

(empty until the campaign's results; appended below the freeze, never edited above it)
