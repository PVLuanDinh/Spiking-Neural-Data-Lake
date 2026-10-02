# Codebase Digest -- every region read IN FULL

**Generated 2026-08-17 from the `/learn-codebase` pass of session `f20e7c6f`.** Fourteen agents each read one region of the tree end to end -- 481 files, 149,799
lines, ~1.9M tokens, which is 8-10 context windows and cannot be held inline. This file is the
result, extracted from the session transcript rather than regenerated: the pass has already been
paid for once and re-running it would spend the same budget to produce the same text.

**How to use it.** Read the region you are touching before editing it. `graphify-out/SRC_INVENTORY.md`
is the per-file docstring index; this is the analysis layer on top -- wiring, invariants, and the
defects each region carries.

**Status: point-in-time.** These are observations at the commit noted below, not live state.
Line numbers in this tree rot within days; cite by SYMBOL and re-verify before relying on any
claim here. A finding recorded as OPEN may have been fixed since.

**G8 note (2026-10-02).** The package names below (`cortex/`, `hippocampus/`, `basal_ganglia/`, and the rest) are the
pre-G8 anatomy packages this pass read; G8 moved their modules into functional packages. File-path citations of moved
modules were re-pointed by the G8 stages; section headers, counts and package-level prose stay as read. The frozen
`snn_data_lake.LEGACY_REGIONS` table and plan A.5a map each old path to its current one.

---



## Region digest 1 of 25

<task-notification>
<task-id>ae12db95495fb1722</task-id>
<tool-use-id>toolu_01Uyt4eq8XC56cRpzPB33cRt</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\ae12db95495fb1722.output</output-file>
<status>completed</status>
<summary>Agent "Read the test suite in full" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 73 files read in full. Digest below.

---

## 1. COUNT — 73 files

**tests/ (63):** conftest.py, test_access_strength_graded.py, test_affect_integrator.py, test_affective_appraisal.py, test_analogy_bench.py, test_ar_lm_provenance.py, test_arc_diagnostic.py, test_arc_fix.py, test_arc_residual.py, test_arm_benchmark.py, test_arm_config.py, test_arm_view.py, test_arm_vocabulary.py, test_blimp_bench_serve.py, test_brain_growth_restart.py, test_cockpit_actuation.py, test_connectome_router.py, test_counterfactual_sim.py, test_coverage_score_citation.py, test_coverage_score_cli.py, test_coverage_score_fingerprint.py, test_data_quality.py, test_dependency_ledger.py, test_distance_units_contract.py, test_drift_monitor.py, test_episode_consolidation.py, test_episode_durability.py, test_episodic_recall.py, test_eprop_train_tier_unreachable.py, test_estop_self_clear.py, test_generative_self_model.py, test_grid_place_map.py, test_growth_rng_is_hardware_independent.py, test_idle_sentinel_not_contact.py, test_instrument_guard.py, test_interpreter.py, test_lang_coverage.py, test_learned_attention.py, test_learned_evc.py, test_make_packs_download.py, test_method_dormancy_keys.py, test_real_rpm_bench.py, test_recursive_tom.py, test_red_reachability.py, test_reflex.py, test_reflex_channel_layout.py, test_region_coherence.py, test_second_order_access.py, test_self_mod_safety.py, test_selfbuild_route.py, test_selfchecks.py, test_serial_arm.py, test_signal_calibrator_lake.py, test_signal_loop.py, test_so101_arm.py, test_sparse_connectivity.py, test_spiking_grid_can.py, test_task_set.py, test_time_cells.py, test_timing_heldout.py, test_vsa_structure_mapping.py, test_wavefront_planner.py, test_workspace_occupancy.py
**instruments/ (6):** ensemble_rbf_ab.py, genn_equivalence.py, genn_smoke.py, gwt_deep_split_ab.py, jepa_highdim_probe.py, rankingdomain_nightkb_ab.py
**pin/ (2):** capture.py, compare.py — **probe_015/ (2):** criterion.py, sweep.py

## 2. PER FILE

- **conftest.py** — autouse per-test `NORD_HOME` isolation. Red: any test writing the operator's real lake, or a function-scope→session-scope regression.
- **test_access_strength_graded** — `access_strength()` is graded/monotone/bounded, not pinned at 1.0 over drive 0.871–1.899. Red: readout collapsing to a step function.
- **test_affect_integrator** — valence sign structure, arousal ranking, emergent emotion prototypes per region, slow mood accumulation, same-seed determinism.
- **test_affective_appraisal** — training moves a context toward target vs naive baseline; a disjoint context stays exactly (0,0). Red: leakage across non-overlapping feature bands.
- **test_analogy_bench** — bench honesty (pass rule needs systematicity&gt;0, gold ≥ recovered), three negative controls separated from real analogies, solver routes without hijacking 9 other experts.
- **test_ar_lm_provenance** — `provenance_status()` separates "recorded-but-empty-field" from "pre-schema". Red: an empty provenance block reading as recorded.
- **test_arc_diagnostic** — synthetic controls classify budget vs coverage; `node_bound` finite/positive; determinism.
- **test_arc_fix** — typed mint→searcher bottleneck: records reconstruct from declarative fields, memo refuses unseen, LOO rejects a train-only affine, direct-apply path retired, two runs identical.
- **test_arc_residual** — descriptor alphabet is dense/versioned/int-only; induced selector actually selects (degenerate always-true selector must NOT reproduce the record); neither recolor induction subsumes the other.
- **test_arm_benchmark** — retired arm no longer probed (AST); torque is physical (gravity on pitch, zero on yaw); payload analytic, model unmutated.
- **test_arm_config** — dbin/present boundaries, IR gate polarity, SHA-pinned MJCF, joint limits read from model not transcribed, ctrl≠stop ranges, arm_hardware derived.
- **test_arm_view** — PNG encoder decoded by hand against the format; unavailable degrades with a reason; no render without demand; TTL; foreign-thread refusal; arm owns one view.
- **test_arm_vocabulary** — every vocabulary site names the same joints; produced tokens accepted by parser AND wire; exactly one body + one sink (AST); retired arm gone from disk/import/atlas.
- **test_blimp_bench_serve** — the serve is *charged*: swapping the model's opinion swaps the reported number under both scorings and changes the printed table; unavailable prints n/a and scores nothing.
- **test_brain_growth_restart** — `slow`-marked. Parent grows → child in a NEW interpreter restores units/placement/census; negative control (empty lake) + positive control (probe sees a written record).
- **test_cockpit_actuation** — allow-list refusal at the front door, submit never actuates, latched arm records "swallowed", client E-STOP latches, `serve()` really forwards the queue, pump bounded.
- **test_connectome_router** — topology = connectome nodes, downstream fires, isolated regions silent, undriven silent, inhibitory edge reduces, loop reverberates, side-effect-free, deterministic.
- **test_counterfactual_sim** — learning cuts imagination error 10×, do()-ranking, determinism, no input mutation, single-transition learning converges.
- **test_coverage_score_citation** — `#anchor` pins a citation to its enclosing def; drift refused with the symbol it landed in; `_cite_kind` census; past-EOF hard fail.
- **test_coverage_score_cli** — unknown/positional/typo'd argv named and refused with exit 2 without reaching `report()`/`rerun()`; valid invocations not refused.
- **test_coverage_score_fingerprint** — real `N/M` counters captured; dates/paths/decimals rejected; turns-probes no longer fingerprint empty.
- **test_data_quality** — schema/gold/duplicate/immutability gates reject at exact boundaries; valid pipeline passes.
- **test_dependency_ledger** — import→dist table correctness, guarded vs unguarded distinction, UNDECLARED bucket fires on a fixture, assert-mode verdict constructed both ways over a fixture repo.
- **test_distance_units_contract** — safety path reads raw metres, spike path reads normalized; substituting one for the other moves the trip point exactly 3× and is strictly MORE conservative (swept).
- **test_drift_monitor** — stable stream quiet, drift warns in both directions, novelty boundary, buffer capped, PSI≈0 on identical distribution.
- **test_episode_consolidation** — sparse binary assembly, every slot matters, completion at capacity, non-catastrophic forgetting, replay non-destructive, ring eviction, episode read-out with honest misses, unbalanced-replay control.
- **test_episode_durability** — P4b: episodes re-read from the lake by a path the writer doesn't own; restart not amnesic; recall past working memory; foreign records preserved.
- **test_episodic_recall** — complete/gist/narrate plus `apply_schema` transfer to unstored situations and five honest-refusal paths; consensus gate proven load-bearing by relaxing it.
- **test_eprop_train_tier_unreachable** — derives the backprop-trainer hazard set (backward+step in one function), asserts it is non-empty, excludes the reference gradient, asserts unreachable, and constructs the RED with one monkeypatched import edge.
- **test_estop_self_clear** — self-clear only when the breach reads clear; sensor fault fails closed; every clear counted and exported.
- **test_generative_self_model** — surprise falls on a repeating trajectory, rises on a novel transition, `reset()` clears trace not weights.
- **test_grid_place_map** — place binding/silence, derived firing bar, one-label-many-places regression, int-percept key-typing regression, determinism.
- **test_growth_rng_is_hardware_independent** — NGSNN's generator is CPU-bound whatever `device=`; same seed → same stream; different seed moves it.
- **test_idle_sentinel_not_contact** — dissociation on `w_exc`: real contact binds, −1 sentinel binds nothing but still stops, clear reading neither.
- **test_instrument_guard** — every rule fired against a reconstructed defect; import-style evasion closed; remedies not flagged; live tier clean; stale/undispositioned exemptions constructed and asserted.
- **test_interpreter** — reflex &gt; instinct &gt; match priority ladder and the reward each returns; joint token derived from arm_config.
- **test_lang_coverage** — restriction rule is score-independent (signature + perturbation), local loop agrees with `blimp_bench.score`, floors exit 2, floors-pass emits a set, absent data SKIPs.
- **test_learned_attention** — gate shape/bounds, uniform probe reveals gate, reward raises / punishment lowers a band, goal biases, focus winner, determinism.
- **test_learned_evc** — allocation monotone in conflict×stakes, paying-off vs wasteful regimes move value/cost opposite ways, withheld trials freeze weights (three-factor eligibility), RPE shrinks.
- **test_make_packs_download** — the never-executed download branch: `_open_dataset` fetches, indices pinned, empty pack is a schema violation, fingerprint drift refuses to write, output is DEFLATED, dry-run writes nothing.
- **test_method_dormancy_keys** — same-named methods in one module stay distinct; exemption keys suppress exactly one rule via the shipped `audit()` path; exemption branch reachability control.
- **test_real_rpm_bench** — above-chance accuracy on held-out matrices, strict ≤ accuracy, failures kept and never claim `known`, two broken controls near chance, perception re-derived from raster.
- **test_recursive_tom** — false-belief transfer, second-order divergence, non-location attitudes, depth-3 chains, unheld paths return None.
- **test_red_reachability** — NO_RED and TAUTOLOGICAL buckets fired on the reconstructed pre-fix `consciousness_probe`; repaired shape not flagged; live probe re-asserted. *RETIRED at T081 (2026-09-16): `reasoning/consciousness_probe.py` is deleted; its 20-seed PCI sweep lives in `instruments/integration_measure.main()`; "are you conscious" answers from the report channel (`evidence/t081-report-order.md` 2.6).*
- **test_reflex** — threshold-exact firing, severe breach in one step, reverse-sign danger, worst-channel wins, out-of-range channels ignored, guard side-channel.
- **test_reflex_channel_layout** — channel *meanings* pinned behaviourally with two asymmetric readings, so a 6↔7 renumber inverts all four; narrow frame surfaces a disabled safety channel.
- **test_region_coherence** — sync/desync phase order, ignition bias monotone and threshold-lowering, scalar rolling path, empty/single-region safety.
- **test_second_order_access** — high on clean ignition, exactly 0.0 on failed ignition under a loud first-order report, monotone in access strength, spike-count-exact, calibration honest at zero.
- **test_self_mod_safety** — both self-modification protection lists name real files; arm-safety basenames present in both.
- **test_selfbuild_route** — the night-8 cadence reaches the recorder; `proposed == 0` is a safety tripwire against a unit test writing the live repo; opt-in gate control.
- **test_selfchecks** — runs every CORE + TORCH_GATED module's `__main__` in-process for coverage; only assertion is `SystemExit.code in (0, None)`.
- **test_serial_arm** — calibration gate has no bypass (port, injected writer, second escape-hatch parameter), E-STOP freezes the wire, unknown commands never transmitted, write error fail-safe.
- **test_signal_calibrator_lake** — P4b for calibration state: lands in the lake with no sidecar, unchanged state not re-appended, written in one process and read in a *separate* one, legacy sidecar migrates, restarted brain's LC gate takes the restored quantile (with a no-state control).
- **test_signal_loop** — normalize/resize edge cases, parse_line contract, Lowe ratio rejects an ambiguous window.
- **test_so101_arm** — E-STOP latch + idempotence, HOME bitwise reproducible and velocity-clearing, clamping reported, probe reports without preventing, `blocked` true only on real refusal, gripper/trailing-junk gate regressions, brain↔arm charge A/B with an airtight neuter.
- **test_sparse_connectivity** — function preserved within a pre-registered tolerance FIRST, savings second; deterministic symmetric mask; weights never escape it; fixed fanout collapses as the store grows.
- **test_spiking_grid_can** — path integration tracks the velocity integral, landmark reset bounds drift, graded correction monotone, first sighting imprints only, kept-failing raw-mixture control.
- **test_task_set** — single-set persistence, shift under sustained conflict, switch-cost drop-then-recover.
- **test_time_cells** — decode monotone/Weber/RMSE-falls-with-width, ascending-order invariant with a retired-rule control arm, degenerate-ladder disqualification control, growth non-destructive/refusable/RNG-neutral, persist-restore with four lineage refusals, live default-on/off-switch.
- **test_timing_heldout** — split disjoint and covers all 63 gaps, blind readout refused, live readout admitted, null-control gap holds a test interval, CLI refuses an uncomputable dispersion.
- **test_vsa_structure_mapping** — solar→atom and 3-object chain recovered at systematicity 1.0, junk target low, wrong mapping penalised, one-to-one.
- **test_wavefront_planner** — a *measured* blocked query forces the RRT, shortcut preserves clearance, colliding start/goal refused, E-STOP latch respected, MOVETO-only vocabulary.
- **test_workspace_occupancy** — live occupancy is not pinned at capacity (deletion-on-null trigger), co-occupancy floor, both producers hold slots and contest the head, access-without-focus occurs.
- **instruments/ensemble_rbf_ab** — MNIST ensemble A/B; asserts pipeline liveness + decorrelation ordering, beats-best-member behind `ENSEMBLE_RUN=1`.
- **instruments/genn_equivalence** — GPU port vs numpy replica: exact counts, dW≤1e-4, theta≤1e-5, 2× bitwise.
- **instruments/genn_smoke** — GeNN builds, loads, steps, records ≥1 spike.
- **instruments/gwt_deep_split_ab** — reproduces the credited binding Jaccard vs all-fired-pairs strawman and rate control.
- **instruments/jepa_highdim_probe** — invariance loss drops, no collapse vs ablation, learned latent beats random encoder.
- **instruments/rankingdomain_nightkb_ab** — surviving `_NightKB` head not measurably worse than the reconstructed retired one at equal exposure.
- **pin/capture.py** — bit-exact tensor capture of six carriers, with `--perturb` and `--reshape-defect` sensitivity modes.
- **pin/compare.py** — `torch.equal` (never `allclose`); exits 1 on any difference **or on zero tensors compared**.
- **probe_015/criterion.py** — alignment criterion qualified by identity=1.0, permutation-null median, constructed reachable red, and a measured noise floor.
- **probe_015/sweep.py** — four feedback pathways through the shipped `dh`; transport-reads-1.0 control gates the verdict.

## 3. COVERAGE MAP

**Well covered.** `brainstem/` (safety envelope) is by far the strongest: 11 files across reflex, serial_arm, so101_arm, arm_config, interpreter, autonomous_loop, vocabulary, view. `hippocampus/` memory (episode_consolidation, episodic_recall, sparse_connectivity, grid_place_map, spiking_grid_can). `substrate/` persistence + P4b (episode_durability, signal_calibrator_lake, data_quality, drift_monitor, region_coherence, home). `tools/` gates (coverage_score ×3, dependency_ledger, red_reachability, instrument_guard, lang_coverage, timing_heldout). `faculties/time_cells` is the single most thoroughly tested module in the tree.

**Thin.** `amygdala/` — one 3-test file plus one dissociation. `neuromodulation/` — `affect_integrator` only; cortisol, LC (only indirectly), dopamine/serotonin/glia untested. `cerebellum/` — `wavefront_planner` only; jepa is a manual instrument. `faculties/gwt_deep` — covered only by an ungated instrument.

**Covered by nothing.** `basal_ganglia/` has **no dedicated test file at all** — `latent_q`/`mlp_q` appear only as fixtures in `test_method_dormancy_keys` and in the manual `probe_015/sweep.py`; `valence_stdp`, `self_build_loop`'s payload, and Go/NoGo gating have no test. `plasticity/` has **no test_*.py** — STDP/homeostatic/developmental rules are exercised only inside `instruments/`, which no gate runs. Also untested: `tools/p4c_ratchet.py` and `tools/p6_gate.py` (both assert in CI, neither has a guard file, unlike their three siblings); `engine/learn_daemon`; the cockpit beyond `/api/act`; `reservoir/cortical_microcircuit.thalamic_drive`; `godel_machine.attempt_on_repo`; `plasticity/eprop_kernel` directly; `consciousness_probe` other than as a scan target.

**Structurally unrunnable by any gate.** All 10 files under `instruments/`, `pin/`, `probe_015/` — `ensemble_rbf_ab.py` states `--collect-only` exits 5 for that directory, and pin/probe files are deliberately not named `test_*`. That is 14% of the files here executing only by hand.

## 4. VACUITY AUDIT

### A. Assertion is constant regardless of the code under test

1. **`test_task_set.py::test_single_task_set_is_held_despite_relentless_bad_signals`** — constructed with `n_sets=1`, then asserts `layer.active_set == 0`. With one set there is no other value the field can hold. Delete the entire shifting mechanism and this still passes. No reachable red.
2. **`test_arc_diagnostic.py::test_controls_cover_both_sides_of_the_verdict`** — iterates `synthetic_tasks()` and asserts the `want` labels include both `budget_proven` and `coverage_proven`. `want` is a literal in the fixture tuple; `diagnose_task` is never called. This asserts a property of a hardcoded list, not of the classifier. (Its sibling `test_synthetic_controls_classify_correctly` additionally has to pass `baseline_nodes=20` for `syn_budget`, so the budget side of the classifier is only exercised at a configuration the product does not ship.)
3. **`test_arc_residual.py::test_authored_object_attributes_survive_inside_the_alphabet`** — asserts six literal strings are members of `DESCRIPTOR_NAMES`. Two literals compared to each other; no behaviour involved.
4. **`test_second_order_access.py::test_state_is_functional_only_with_no_phenomenal_key`** — line 95 asserts `set(st)` equals four literal key names; lines 101-102 then loop over those same keys asserting no `_BANNED` token appears in them. Given the preceding assert passed, the loop is true by construction and cannot fire. It reads as a phenomenal-language guard and is a no-op.
5. **`test_analogy_bench.py::test_pass_rule_requires_structural_evidence`** — line 49 asserts `r["passed"] == (r["exact"] and r["sys"] &gt; 0.0)`. Lines 50-51 then re-derive the same implication for the `sys == 0.0` case. The second assertion cannot fail if the first passed.
6. **`test_learned_attention.py::test_different_seed_signature_reproducible`** and **`test_learned_evc.py::test_different_seed_signature_reproducible`** — both construct **two instances at the same seed (1)** and assert equal signatures. The name promises the different-seed property; the body tests same-seed determinism, already covered by `test_determinism_same_seed` in each file. Nothing in either file proves a different seed produces a different signature — so a `signature()` that returned a constant passes both files. (Contrast `test_second_order_access.py`, which does assert `build(0) != build(3)`.)
7. **`test_affective_appraisal.py::test_same_seed_and_protocol_is_deterministic`** — the local `train(seed)` hardcodes `AffectiveAppraisal(dim=dim, seed=0)` and uses its `seed` argument only for the training RNG, then asserts `train(7) == train(7)`. The parameter it claims to vary is discarded, and no different-seed control exists.

### B. Only assertion is "the call did not raise"

8. **`test_learned_attention.py::test_selfcheck_main_runs`**, **`test_learned_evc.py::test_selfcheck_main_runs`**, **`test_region_coherence.py::test_selfcheck_main_runs`** — each body is `mod.main()` with the comment "asserts internally; must not raise". The test asserts nothing itself and is fully duplicated by `test_selfchecks.py`, which runs every module's `__main__`. If `main()`'s own asserts were stripped, all four sites stay green.
9. **`test_selfchecks.py::test_core_selfcheck_in_process`** — the only assertion is inside `except SystemExit`. A module whose `main()` returns normally is asserted about **nothing at all**; a module whose `main()` contains zero asserts contributes a green. Its real value is coverage instrumentation, not verification, and the file's docstring frames it that way — but it is the sole gate on several modules' behaviour.
10. **`test_dependency_ledger.py::test_gate_runs_on_the_real_tree_and_reports_a_work_list`** and **`test_red_reachability.py::test_gate_runs_on_the_real_tree`** — both assert `main([]) == 0`. Both docstrings state report mode *always* exits 0. The only reachable red is an uncaught exception. `test_dependency_ledger.py` explicitly deleted a sibling for exactly this shape (the note at line 167), but left this one.
11. **`test_episodic_recall.py::test_selfcheck_main_passes`** and **`test_analogy_bench.py::test_bench_main_self_check_passes`** — `assert main() == 0`. Same shape one level removed: the test cannot distinguish a self-check that measured something from one whose asserts were removed.

### C. Count/shape asserted, content never

12. **`test_method_dormancy_keys.py::test_uncertain_is_reported_not_asserted`** — after the 2026-08-05 repair the whole body is `isinstance(r["uncertain"], list)` and `isinstance(r["dormant"], list)`. `audit()` returning two empty lists forever passes. The file's own docstring concedes the behaviour is asserted elsewhere; as written this test measures Python's return type.
13. **`test_arc_fix.py::test_records_are_deterministic_and_mdl_then_name_ordered`** — the name claims **MDL-then-name ordering**; the body asserts only that two calls agree and that names are ASCII. The ordering property the title advertises is never checked. (It is partially checked in `test_mdl_order_prefers_functional_over_memo_on_the_surviving_path`, but not the ordering rule itself.)
14. **`test_time_cells.py::test_growth_preserves_learned_thresholds`** — `survivors = [t for t in clock.thresholds if t in snapshot]; assert len(survivors) &gt;= len(snapshot)` is satisfiable by duplicate values rather than by preservation. The following per-element loop is the real assertion; the count line adds a way to pass without it.

### D. try/except that swallows the failure

15. **`test_eprop_train_tier_unreachable.py::hazard_modules`** — `except SyntaxError: continue`. A governed module that will not parse is silently dropped from the hazard set, so a `backward()+step()` trainer inside an unparseable file becomes invisible to the file's central constraint. `test_the_hazard_set_is_not_empty` mitigates total blindness but not a single dropped module.
16. **`test_arm_vocabulary.py::test_exactly_one_arm`** — same `except SyntaxError: continue` inside the tree walk. A second arm implementation in a file that fails to parse would not appear in `found`, and the P4 "exactly one body" assertion passes.

### E. Would still pass if the feature were deleted or neutered

17. **`test_growth_rng_is_hardware_independent.py::test_same_seed_same_stream_across_requested_devices`** — the whole file's RED. On any machine without a CUDA card `NGSNN(device="cuda")` degrades to CPU, so the test compares CPU against CPU and passes with the fix reverted. CI has no card (POLICIES P5), so **in CI this test cannot fail**. Its own premise-check sibling below is worse:
18. **`test_growth_rng_is_hardware_independent.py::test_cpu_and_cuda_torch_generators_really_do_diverge`** — guards with `if not torch.cuda.is_available(): return`, a bare `return`, not `pytest.skip`. On CI it **reports PASSED while asserting nothing**. A skip would at least be visible in the report; a green here actively misrepresents coverage of the premise the file rests on.
19. **`test_red_reachability.py::test_stale_exemption_is_reported`** and **`test_dependency_ledger.py::test_stale_exemption_fails_loudly`** — both are `for name, reason in EXEMPT.items(): assert len(reason) &gt; 20`. If `EXEMPT` is empty the loop body never runs and the test passes having checked nothing. This is precisely the shape `test_instrument_guard.py::test_stale_exemptions_fail_loudly` documents fixing (it constructs a bogus entry because "with `EXEMPT == {}` both of its assertions were true by construction"). The repair was applied to one of the three gate-guard files and not the other two.
20. **`test_selfbuild_route.py`** — self-disclosed, and the disclosure is accurate: a proposer that raises leaves `_selfbuild_last` written normally (`Brain.propose_improvements` has its own `except Exception: return []`), and a proposer that returns nothing is indistinguishable from a healthy one. The route is pinned; the payload is not.
21. **`test_episode_consolidation.py::test_recall_empty_memory_returns_zeros`** — ends `assert ec.retained() == 1.0  # vacuously retained when nothing is stored`. This pins as *correct behaviour* the exact reading (`retained() == 1.000` over an empty store) that produced the P4b incident. It is deliberate, but it means the tree's own test suite blesses a 1.0 that carries no information.
22. **`test_region_coherence.py::test_silent_bank_not_spuriously_coherent`** — asserts only `out["phase_order"] &lt; sync["phase_order"]`. A silent bank reading 0.94 against sync's 0.95 passes while the docstring's claim ("must not read as synchronized") is false. Every other test in the file uses absolute bounds (`&gt; 0.95`, `&lt; 0.20`); this one does not.
23. **`test_learned_evc.py::test_control_withheld_when_not_worth_it`** — `allocate(0.02, 0.02) &lt; 0.05` at a product of 0.0004, on a function already asserted bounded in [0,1] and monotone in conflict×stakes. Almost any monotone formula satisfies it; the cost-versus-benefit decision it claims to test is not isolated.
24. **`test_lang_coverage.py::test_lexicon_is_from_the_exposure_plan_not_an_organ`** and **`::test_nord_home_is_isolated`** — both assert over the tool's raw source text (`"language_organ" not in src`; `src.index(...) &lt; src.index(...)`). An aliased import, a re-export, or a reordered import block defeats them without changing behaviour.
25. **`test_counterfactual_sim.py::test_persistence_prior_at_init`** — tolerance 0.5 against inputs whose magnitudes are ≤0.5. Any function returning roughly half the input passes; so does a near-zero one for three of the four coordinates. The "persistence prior" claim is not discriminated from "outputs something small".

### F. Missing negative controls

26. **`test_generative_self_model.py`** — three tests, no control that an *untrained* model does not also show falling surprise. Nothing separates learning from eligibility-trace decay or from the trajectory simply becoming more predictable to any accumulator. No determinism test either, unlike every sibling module file.
27. **`test_task_set.py`** — proves a shift happens under sustained conflict + low reward. Nothing proves a shift does **not** happen under low conflict + high reward. A layer that shifts every N steps regardless passes both remaining tests (`test_switch_cost_drops_then_recovers` also only requires that a shift occurred).
28. **`test_learned_attention.py::test_focus_reports_winner`** — asserts band 1 wins on a loud band-1 input. Nothing asserts focus abstains, or reports low strength, on a flat/silent input. A `focus` that always names argmax of noise passes.
29. **`test_vsa_structure_mapping.py`** — has a junk-target case but no permutation null and no untrained/random-codebook control. Nothing rules out the mapping being recoverable from surface token identity rather than from the spike codes.
30. **`test_affect_integrator.py`** — asserts `build(0) == build(0)` but never that a different seed produces a different emotion map, so a `signature()` insensitive to seed passes.
31. **`test_self_mod_safety.py::test_arm_safety_files_are_named_by_both_lists`** — compares the two lists to *each other*; if neither was updated after a rename, the comparison still holds. `test_so101_arm.py::test_the_live_gate_is_in_both_safety_kernels` closes this for one file only (`so101_arm`); the other four safety-envelope files have no such anchor.

### G. Structural notes that bear on all of the above

- **`test_brain_growth_restart.py`** carries `pytestmark = pytest.mark.slow` **and** a `pytest.skip` when the parent grows nothing in budget. The tree's only brain-level P4b restart proof can therefore be absent from a default run and, when it does run, can skip itself.
- **`test_ar_lm_provenance.py::test_shipped_artifact_is_reported_as_unrecorded_not_silently_passed`** and **`test_blimp_bench_serve.py::test_real_spiking_lm_serve_charges_the_score_when_present`** both skip when `models/` is absent. `models/` is gitignored, so on any fresh clone and in CI these two never execute — the only tests that touch the real LM artifact.
- The best-constructed files in the tree are **`test_instrument_guard.py`**, **`test_red_reachability.py`**, **`test_eprop_train_tier_unreachable.py`**, **`test_signal_calibrator_lake.py`**, **`test_idle_sentinel_not_contact.py`**, **`test_reflex_channel_layout.py`** and **`test_time_cells.py`** — each fires its criterion against a constructed defect and asserts the *repaired* shape is not flagged. The vacuity findings above are concentrated in files that lack that pattern, and in three gate-guard files where the pattern was applied inconsistently (items 19 and 10).</result>
<usage><subagent_tokens>331573</subagent_tokens><tool_uses>76</tool_uses><duration_ms>462820</duration_ms></usage>
</task-notification>

---


## Region digest 2 of 25

<task-notification>
<task-id>a12d0c3457821b943</task-id>
<tool-use-id>toolu_01S2nzBQjgfCtrspvd74k1aw</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a12d0c3457821b943.output</output-file>
<status>completed</status>
<summary>Agent "Read plasticity+BG+neuromod" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 42 files read in full. Digest follows.

## 1. COUNT

**plasticity/ — 18 files**: `__init__`, `bcm_plasticity`, `benna_fusi`, `competitive_learning`, `criticality_regulator`, `deep_stdp`, `developmental_pruning`, `homeostatic_plasticity_stack`, `homeostatic_scaling`, `inhibitory_stdp`, `intrinsic_plasticity`, `lifelong_consolidation`, `meta_plasticity`, `oja_pca`, `plasticity_rule_stack`, `short_term_plasticity`, `snn_mnist_stdp`, `snn_mnist_stdp_genn`

**basal_ganglia/ — 10 files**: `__init__`, `action_gating`, `brain_decision`, `brain_native_selfbuild`, `evidence_accumulation`, `latent_q`, `mlp_q`, `r_stdp`, `self_build_loop`, `valence_stdp`

**neuromodulation/ — 14 files**: `__init__`, `acetylcholine`, `affect_integrator`, `astrocyte_modulation`, `cortisol`, `developmental_envelope`, `drive_reduction`, `empowerment`, `generalizing_novelty`, `intrinsic_motivation`, `locus_coeruleus`, `naca`, `raphe_serotonin`, `spiking_curiosity`

## 2. PER MODULE

### plasticity/
- **bcm_plasticity** — `BCMNeuron` (dw = lr·x·y·(y−θ), θ tracks ⟨y²⟩), `develop_selectivity`, `_shape_patterns`. θ is the metaplastic variable read by `plasticity_rule_stack._BCMMetaGate`; that is its only live consumer. Gotcha: BCM is only stable for inputs in ~[0, 0.5) — `_BCMMetaGate._squash` exists solely to keep drive in that band.
- **benna_fusi** — `BennaFusiSynapse` (N coupled beakers, geometric capacities/conductances, `deposit`/`diffuse`/`weight`/`deep_mean`), `BennaFusiLearner`. Consumed by `plasticity_rule_stack._BennaFusiConsolidation` and `faculties/self_improvement_suite`. Invariant: `weight` is beaker u₁ only; `n_levels=1` and `g0=0.0` are the two shipped controls and both must stay reachable or the retention claim goes vacuous.
- **competitive_learning** — `CompetitiveLearner` (FEELING, Demin &amp; Nekhaev 2018: eq.6 feed-forward activity-max + eq.8 learnable lateral inhibition over `LIFNeuron`). **No importer outside itself.** Gotcha: three declared discrete-time deviations (L1 renorm, winner-gated plasticity, baseline lateral init) are load-bearing — removing any one stops convergence.
- **criticality_regulator** — `CriticalityRegulator`: recurrent LIF pool, σ measured via `neuronal_avalanches.BranchingNetwork.measured_branching_ratio`, one knob (`inhib_gain`) driven by a slew-limited integrator. Constructed by `engine/brain.py` (n=64) and `wholistic_loop`. **Critical gotcha, stated in-file: the loop closes on this object's OWN pool, never on LiveBrain neurons** — brain.py reports its σ as `criticality`, which describes the side-pool only.
- **deep_stdp** — `FcStdp`/`ConvStdp`/`OneLayerNet`/`TwoLayerNet`, conv-STDP→pool→proto-STDP on MNIST. Imports constants + `load_mnist` from `snn_mnist_stdp`. Records an honest NEGATIVE (2-layer 8.8% vs 1-layer 63.6%, map collapse cos≈0.96). Gate runs `_mechanics_check` only; `DEEPSTDP_RUN=1` opts into the 8-min benchmark.
- **developmental_pruning** — module-level `_develop`/`_prune_magnitude`/`_prune_random`/`run`. Magnitude beats random at matched sparsity. **Zero call sites tree-wide** (only docstring mentions + gate registry); it is in `dormancy_gate.EXEMPT`.
- **homeostatic_plasticity_stack** — `HomeostaticPlasticityStack` with `.pre_gain(id)` / `.stabilize(id,w,act)` / `.learning_step(...)`; three thin adapters wrapping `DynamicSynapse`, `HomeostaticNeuron`, `IntrinsicPlasticity` in a fixed PRE→[R-STDP]→POST sandwich. Attached to the canonical brain's interaction learner. Invariant: `stack=None` must stay byte-identical; `_HomeostaticScaling` is a soft CEILING (trims only, never boosts) or routing discrimination flattens.
- **homeostatic_scaling** — `HomeostaticNeuron.hebb/stabilize/step` (Turrigiano multiplicative). Live as the stack's POST phase. Its wiring note carries the ONE measured charge figure for the whole stack family (max|dw| 15.30 in weights; routing accuracy 1.000 in every arm, i.e. **charge in state, not in outcome**).
- **inhibitory_stdp** — `InhibitorySTDP` (Vogels 2011; α = 2·ρ₀·τ fixes the rate fixed point). Reused by `plasticity_rule_stack._EIBalance`. Gotcha: pre-traces bump AFTER the pre-update, deliberately — reordering changes the fixed point.
- **intrinsic_plasticity** — `IntrinsicPlasticity.adapt` (Triesch; tunes gain a / bias b to an exponential output target, input-scale invariant). Live as the stack's cellular-homeostasis POST phase.
- **lifelong_consolidation** — `LifelongConsolidator` (grow via `GrowingLearner` ↔ prune by utility ↔ consolidate via `ReplayBuffer` interleaved replay). Constructed by `engine/brain.py` + `wholistic_loop`. Gotcha: `task_decay` is applied once per task boundary — within-task credit just accumulates, and that contrast is the entire non-regression story.
- **meta_plasticity** — `_Unit`, `MetaPlasticLearner` (mechanism (a) LP-gated global lr via `SpikingCuriosityDrive`; mechanism (b) per-unit BCM sliding θ). Used by `faculties/self_improvement_suite`. **Its own self-check proves (a) is a pure tick-keyed schedule with no local information**; only (b) survives a per-unit history shuffle. Do not re-describe (a) as metaplasticity.
- **oja_pca** — `OjaNeuron` (numpy-backed, w += lr·y·(x − y·w)), `sanger` (Sanger/GHA deflation, stdlib lists). `OjaNeuron` is live in `faculties/percept_surprise`; **`sanger` has no importer.** Gotcha: `unit()` is deliberately kept for the seeded init so start vectors stay byte-identical to the pre-numpy path.
- **plasticity_rule_stack** — `PlasticityRuleStack(HomeostaticPlasticityStack)` + `_BCMMetaGate` / `_EIBalance` / `_BennaFusiConsolidation` + `relax()`. The stack actually attached to the brain. Invariant: with `ei=metaplastic=consolidation=False` it must be byte-identical to the base. Flagged in-file as ARM-ADJACENT — the deep anchor shifts the effective-weight argmax and can bias intent routing.
- **short_term_plasticity** — `DynamicSynapse` (Tsodyks-Markram u·R) is live as the stack PRE phase; **`novelty_filter` (the streaming_hub-facing helper) has no consumer.** The docstring says so explicitly.
- **snn_mnist_stdp** — the rule-family HUB: `M/T/BETA/LR/THRESH/W_NORM/…`, `load_mnist`, `StdpNetwork`, `assign_labels`, `evaluate`. Unguarded module-top `torchvision` import is a declared disposition (guarding it would make 17 dependents import-but-fail-at-call). `torch.manual_seed` correctly moved out of module scope.
- **snn_mnist_stdp_genn** — GeNN 5 codegen of the burst+x_tar rule (`GENN_*` C++ strings, `burst_xtar_rule`, `_wta_neuron`, `build_model`, `pack_schedule`, `_run_pass`, `run_mnist`, `oracle`). Gate-safe: never compiles unless `GENN_RUN=1`. Gotchas: every custom model object must be pinned on `model._snn_keepalive` (GeNN holds raw pointers → 0xC0000005); `steps = T+3` because the last winner's LTP lands a kernel late.

### basal_ganglia/
- **action_gating** — `kwta(values,k)` (THE single definition of that primitive; `superior_colliculus` imports it) and `BasalGangliaGate.select(drive, da, stn)` — D1/D2 → GPi → LIF thalamic relay disinhibition. Live in `autonomous_loop` + `rpm_solver`. Invariant: `stn ≥ 0` can only make release HARDER; `stn=0.0` must stay byte-identical.
- **brain_decision** — `BrainDecision.choose/observe/grow_trigger/reliability/explore`, fusing `LatentQ` value + `MetacognitiveMonitor`. Reached only by `instruments/developmental_loop_brain`. **Records a measured NULL: at the shipped α=0.8 the value channel contributes exactly zero to the argmax (real == value-DEAD bit-identical).** The self-check deliberately asserts neither direction so a later fix does not read as a regression.
- **brain_native_selfbuild** — `RunningBaseline`, `SelfBuildSource`/`DehardcodeSource`/`DataIngestSource`/`ThresholdRecalibrationSource`, `BrainNativeSelfBuildProposer.propose/observe/should_skip/note_cycle/ledger`, 4-layer firewall (`_forbidden`, basename-matched, `_SELF_REL` derived from `__file__`). **See §5 — the proposer cannot propose anything.**
- **evidence_accumulation** — `DriftDiffusion`, `RaceAccumulator` over a shared `_simulate` (LIF pools + one shared inhibitory pool; bound = sustained population rate). Live in `reasoning_curriculum` and three `arc_agi` modules. Nothing is learned here — pure decision dynamics.
- **latent_q** — `LatentQ` (shared linear Q(z,a) = W[a]·z + b[a], semi-gradient/Widrow-Hoff). Used by `brain_decision` and `autonomous_loop` (default OFF). Gotcha: transfer and interference are the same shared-W property; the `_TabularQ` control must stay unable to transfer.
- **mlp_q** — `MlpQ` (one ReLU hidden layer, same API as LatentQ). **Weight transport RESOLVED 2026-08-14 via Kolen-Pollack**: a separate feedback matrix `B` takes the same increment as `W2` and never reads it; check (5) re-derives the cosine live (shipped ≈0.90 vs frozen ≈0.2, transport identity 1.0). Gotcha: default OFF in `AutonomousArm`; MlpQ is NOT a strict upgrade (it loses ~0.01 on a linear rule).
- **r_stdp** — `RSTDPNeuron` (`pre_spike`/`post_spike`/`step`/`reward`) — THE three-factor rule, eight importers, the project's reuse anchor. Plus a measured population-broadcast surface: `calibrate`, `parallel_plan`, `partition`, `reward_population`, `_FanOut`. Gotchas: the crossover S* is DERIVED per host, never a constant; every worker must reach the join barrier on the raising path (a fixed hang); byte-identity is a claim about SUCCESSFUL updates only.
- **self_build_loop** — `SelfBuildLoop.promote/run`, `_FIREWALLED` (harness `_PROTECTED` + `SELF_BUILD_LOOP_REL` derived from `__file__`). Apply → verify envelope → gate → keep or byte-exact rollback; the envelope is re-verified at COMMIT, not only at apply, so a green gate that touched a protected file is a red gate. Off unless `enable=True` AND `NORD_SELFMOD=1` (the daemon sets it).
- **valence_stdp** — `ValenceLearner.valence/act/dopamine/learn` — RPE (reward − predicted) three-factor over feature presence. Consumed by `drive_reduction`. Distinct from r_stdp on both axes: no spike timing, no delay bridge.

### neuromodulation/
- **acetylcholine** — `Acetylcholine` (leaky EMA level; `feedforward_gain`/`recurrent_gain`/`plasticity_scale`/`mode`) + `RecurrentStore` demo. Constructed in `engine/brain.py`. `MODE_TH=0.5` is a fixed absolute cut.
- **affect_integrator** — `AffectIntegrator.integrate/observe/label/describe/signature`: LC-NE LIF arousal pool + opponent LIF valence pools + emergent emotion prototypes via `GrowingLearner` over a `PopulationEncoder` code. Live in `engine/brain.py`. Gotcha: `_mood_slow` is the only state carried across `integrate()` calls — a fresh instance is a different measurement.
- **astrocyte_modulation** — `Astrocyte` (per-domain slow Ca + threshold gliotransmission + slow gain relaxation), `modulated_transmission`. Live in `engine/brain.py`. Distinctness rests on LOCALITY (per-domain) and the long τ_ca=50 vs τ_m=10.
- **cortisol** — `Cortisol.step/reflex_threshold_scale/learn_rate_scale/caution_bias`. Slow tonic HPA abstraction; `demo_episode` couples it to `brainstem/reflex`. `TAU/GAIN/MAXMOD` are fixed module constants.
- **developmental_envelope** — `DevelopmentalEnvelope.update/explore_noise/forward_model_gain/mix/explore_sample`. Live in `autonomous_loop`. Invariant: `mix` must stay a CONVEX combination (an extrapolating variant escapes to 0.4885 and the assert fires).
- **drive_reduction** — `NeedPool` (LIF need pool; `rate`/`spikes`/`membrane` control) + `DriveReductionLearner` (reward = D_before − D_after, eligibility = spiking rate DROP, gated into `ValenceLearner.learn(lr_scale=elig)`). Used by `self_improvement_suite`. **Two honest negatives kept: the learned policy TIES (never beats) the `set_points` controller, and the non-spiking membrane eligibility fails only at seed 0 — it ties on seeds 1-6.**
- **empowerment** — `Empowerment.empowerment/reward/step_reward`. `step_reward` is live (`autonomous_loop`, enabled at `brain.py`); **`reward` has zero call sites and the docstring says so** — kept as unconsumed capability (a planner would use it), not merged, because merging would double the live per-tick cost.
- **generalizing_novelty** — `GeneralizingNovelty.bucket/pseudo_count/bonus` (frozen random-hyperplane SimHash → crc32 bucket → 1/√N pseudo-count). Live in `brain.py`, `self_improvement_suite`, `arc_agi/play_brain`. **Refuses the overclaim: SimHash INHERITS an encoder collapse; the collapse demo is asserted to FAIL, not fixed.** Thresholds were re-derived from a 21-seed sweep after the pinned seed proved unrepresentative.
- **intrinsic_motivation** — `CuriosityDrive` (two-EMA learning progress = max(0, slow − fast), habituating novelty, `intrinsic_reward`, `curiosity_value`, `most_curious`). The rate original. Gotcha: `intrinsic_reward` calls `observe` internally — never also call `observe` for the same error.
- **locus_coeruleus** — `LocusCoeruleus.step/gain/mode` — 64-LIF pool, NA = windowed spike count / window. Live in `brain.py`. **Two measured findings kept in the docstring: (1) the LC's live input is effectively a literal — `min(1.0, _pc_surprise_ema)` pinned it at 1.0 on 119 of 120 ticks; (2) the phasic trace winds up to 2.71 against a 1.0 input clamp, and an anti-windup fix measured zero live effect, so it was deliberately not applied.**
- **naca** — `NACABroadcast.lam/f_local/modulate` — a MODULATOR that returns an additive correction on an existing R-STDP delta, never a plasticity rule. Live in `autonomous_loop` and wired opt-in into `interaction_learning`. Invariant: λ=0 ⇒ correction is exactly zero ⇒ byte-identical to plain R-STDP.
- **raphe_serotonin** — `DorsalRaphe.step/discount_rate/value/choose/patience/tolerated_wait` — hyperbolic discount rate log-spaced between `K_HI` and `K_LO`. Live in `brain.py`. Monotonicity is structural, not seed luck.
- **spiking_curiosity** — `SpikingCuriosityDrive` (LP from EMITTED integer spike counts of a fast/slow LIF pool pair, each calibrated to `_E_REF`) + `SpikingCurriculum` (advance gate is a LIF accumulator; injection solved so it spikes at exactly `patience`). Selected by `brain.py` via a flag. **Records open P2 debt: both this and the rate `CuriosityDrive` are live behind that flag, and the measurement supports retiring neither** (spiking is not better on behaviour — 24.1/6.8/17.1 vs the rate drive's 41.8/1.5/4.7 visit split).

## 3. REGION NOTES

**plasticity/** is a two-layer architecture: a flat family of standalone rule primitives (BCM, Oja, Tsodyks-Markram, Turrigiano, Triesch, Vogels iSTDP, Benna-Fusi) plus two COMPOSERS that are the only route any of them takes to the live brain — `HomeostaticPlasticityStack` (PRE short-term gain → [R-STDP deposit] → POST scaling + intrinsic) and its superset `PlasticityRuleStack` (+ BCM metaplastic LR gate PRE, + iSTDP E/I trim and Benna-Fusi consolidation POST). Data flow: `engine/brain.py` constructs the stack and assigns it to `brain._ixn.stack`; per learning step the learner calls `pre_gain(id)` → runs its own `RSTDPNeuron.reward()` → calls `stabilize(id, w, activity)`. Everything else in the directory (MNIST STDP, GeNN port, developmental pruning, competitive learning, lifelong consolidation, criticality) is a standalone experiment or a side-pool with its own entry point.

**basal_ganglia/** holds four separable stages plus the self-build loop. (1) LEARNING: `r_stdp` (three-factor, distal reward) and `valence_stdp` (RPE over feature presence) — deliberately non-overlapping. (2) VALUE: `latent_q` (linear) and `mlp_q` (one hidden layer), both over the JEPA latent, both default OFF on the arm. (3) SELECTION: `action_gating` Go/NoGo disinhibition — the only stage that releases an action. (4) DECISION-BY-INTEGRATION: `evidence_accumulation` spiking attractors. Layered on top, `brain_decision` fuses value + metacognition for study-objective selection. Separately, `brain_native_selfbuild` (propose) + `self_build_loop` (gated promote) form the P0 self-improvement route, reached from `Brain._enter_wake` → `_maybe_autonomous_self_build` → `SelfBuildLoop.run`.

**neuromodulation/** is a set of diffuse scalar/population modulators on distinct axes — ACh (encode vs retrieve), cortisol (tonic stress), 5-HT (patience/discount), LC-NE (adaptive gain), astrocyte (slow LOCAL glial gain) — plus five intrinsic drives on orthogonal axes: learning-progress curiosity (rate `intrinsic_motivation` and spiking `spiking_curiosity`), visitation pseudo-count (`generalizing_novelty`), homeostatic drive reduction (`drive_reduction`), and controllability (`empowerment`). `affect_integrator` is the fusion point — it is the only module that consumes four modulator readouts at once and returns a (valence, arousal) core with emergent emotion prototypes. `naca` is architecturally odd for the directory: it modulates plasticity rather than state. Almost every module here is constructed directly by `engine/brain.py` or `engine/autonomous_loop.py`.

## 4. LEARNING RULES

| # | Rule | Where | Notes |
|---|---|---|---|
| 1 | **R-STDP / three-factor** (Izhikevich 2007) | `plasticity/r_stdp.RSTDPNeuron` | The anchor; 8 importers. Local: eligibility × broadcast scalar. |
| 2 | **RPE valence** (Schultz dopamine) | `plasticity/valence_stdp.ValenceLearner.learn` | δ = reward − predicted; not spike-timing. |
| 3 | **Semi-gradient / Widrow-Hoff (linear)** | `faculties/latent_q.LatentQ.update` | One-layer; no backward pass at all. |
| 4 | **Hand-written backprop through 1 hidden layer** | `faculties/mlp_q.MlpQ.update` | **Was weight transport; RESOLVED via Kolen-Pollack.** Owes and PAYS the cosine: check (5) measures shipped ≈0.90 &gt; 0.5 vs frozen ≈0.2, positive control = 1.0. |
| 5 | **Three-factor bandit over self-build candidates** | `brain_native_selfbuild.BrainNativeSelfBuildProposer.observe` | Reuses RSTDPNeuron; reward = gate outcome ±1. |
| 6 | **STDP (unsupervised, WTA)** | `instruments/snn_mnist_stdp`, `deep_stdp`, `snn_mnist_stdp_genn` | Pure spike-timing; no labels, no backprop. |
| 7 | **BCM** (sliding θ ~ ⟨y²⟩) | `plasticity/bcm_plasticity`; per-unit spiking variant in `meta_plasticity` | |
| 8 | **Oja / Sanger (GHA)** | `plasticity/oja_pca` | Local Hebbian + self-normalizing decay. |
| 9 | **Tsodyks-Markram short-term** | `plasticity/short_term_plasticity` | Not a weight rule — a dynamic gain. |
| 10 | **Homeostatic synaptic scaling** (Turrigiano) | `plasticity/homeostatic_scaling` | Multiplicative, ratio-preserving. |
| 11 | **Intrinsic plasticity** (Triesch) | `plasticity/intrinsic_plasticity` | Adapts the cell, not the synapse. |
| 12 | **Inhibitory STDP** (Vogels 2011) | `plasticity/inhibitory_stdp` | α = 2ρ₀τ ⇒ negative-feedback rate controller. |
| 13 | **Benna-Fusi cascade metaplasticity** | `plasticity/benna_fusi` | Linear diffusion, honestly labelled spiking-ADJACENT. |
| 14 | **Competitive activity-maximization (FEELING)** | `plasticity/competitive_learning` | Unsupervised, learnable lateral inhibition. |
| 15 | **Developmental magnitude pruning** | `sleep/developmental_pruning` (synapses), `lifelong_consolidation` (units) | Delta-rule training + magnitude cut. |
| 16 | **Softmax delta rule** | `sleep/developmental_pruning._train_epoch` | Rate readout, single layer. |
| 17 | **Bounded metaplasticity** (LP-gated lr + BCM θ) | `plasticity/meta_plasticity` | (a) is proven a global schedule, not local. |
| 18 | **Criticality homeostat** | `reservoir/criticality_regulator` | σ error → one inhibitory-gain knob. |
| 19 | **Growing/Hebbian competitive (SOM-like)** | `faculties/affect_integrator` (via `GrowingLearner`) | Recruit-or-refine. |
| 20 | **NACA neuromodulatory credit assignment** | `plasticity/naca` | Explicitly gradient-free, no weight transport; a modulator on rule #1. |
| 21 | **Drive-reduction three-factor** | `faculties/drive_reduction` | Feeds rule #2 with a spiking eligibility as `lr_scale`. |
| 22 | **Hebbian associative (Hopfield-ish)** | `faculties/acetylcholine.RecurrentStore` | Demo store only. |

**BPTT / weight transport audit:** no BPTT anywhere in these three directories. The single weight-transport site was `mlp_q.MlpQ.update`'s `dh` line and it is **closed** — a separate feedback matrix `B` receives the same local increment as `W2` and never reads it, and the module re-derives the cosine on every gate run rather than quoting it. `snn_mnist_stdp` / `deep_stdp` use the Brian2-equation bridge only for membrane integration with an unreachable threshold — no `.backward()`, no autograd, weights written by STDP alone. Nothing else that learns owes an unpaid cosine.

## 5. DEFECTS

**P0 — the self-build route can propose NOTHING, and the emptiness is structural, not a config gap.**
`brain_native_selfbuild.py:425` — `DEHARDCODE_TRANSFORMS = ()`, and `:337` — `RECALIB_TARGETS = ()`. Therefore `:431` `_ALL_IDS = ()`, the RSTDPNeuron is built with `n_syn = max(1, 0) = 1` over an empty id map, and `DehardcodeSource.candidates()` iterates an empty tuple. `propose()` reaches `if not cands` on every call and logs `honest_null`. `DataIngestSource` returns `[]` unless `enabled=True`, and enabling it raises `NotImplementedError`. `ThresholdRecalibrationSource` is target-less by design (its one real target moved to live consumption in `brain.py::_w8_tick`). **Every mechanism around the proposer is real and tested — firewall, learner, ledger, circuit breaker, byte-exact rollback — and the menu it re-orders is empty.** This exactly matches the measured `{'night': 8, 'proposed': 0, 'kept': 0}` in CLAUDE.md. The gap is authoring gate-safe transforms, not wiring; and the module's own docstring records WHY that is hard (an absolute threshold encodes global calibration a short per-instance running baseline cannot reconstruct, so the naive swap gets rejected by the gate).

**Broken import (silently swallowed).** `instruments/snn_mnist_stdp_genn.py:113` — `from snn_data_lake.reasoning import snn_mnist_stdp as base`. **That module does not exist**; `reasoning/` holds only `snn_mnist_stdp_fast.py`. `snn_mnist_stdp` was promoted to `plasticity/` (the genn file's own `# distinct-from` header at `:6-8` documents the move) and the oracle import was not repointed. `main():606` catches it with `except Exception: have_oracle = False`, so on a GeNN box the API-drift canary is silently skipped and `run_mnist()` would `ImportError` at line 113. This is the silent-degradation shape — a bare except converting a stale path into a green gate.

**Stale docstrings (claim outliving what it described).**
- `instruments/deep_stdp.py:62-63` — "this module lives in `src/snn_data_lake/reasoning/`". It lives in `plasticity/`. Its self-check comment at `:288` also refers to "the genn precedent" by behaviour rather than path.
- `instruments/deep_stdp.py:96` — "mirrors `snn_mnist_stdp.StdpNetwork`", correct, but `:16` cites the readout "used by snn_mnist_stdp" without the region prefix used everywhere else.
- `plasticity/r_stdp.py:196-198` — names `readout/interaction_learning`, `faculties/task_set`, `plasticity/benna_fusi`, `plasticity/naca` as candidate adopters of `reward_population`; still accurate, but the surrounding "ZERO live callers" claim is load-bearing and re-verified below.
- `faculties/locus_coeruleus.py:54` — `SALIENCE_TH = 0.50` is described in `brain_native_selfbuild.py:329-336` as having been retired in favour of live consumption, yet the constant is still the module's default and `brain.py` overrides it at runtime. Two statements of one operating point.

**Dormant (defined, never called outside their own module — verified by full-tree grep including the gates and registries):**
- `plasticity/short_term_plasticity.py::novelty_filter` — the module docstring already discloses this ("has no live consumer; the CLASS does").
- `plasticity/oja_pca.py::sanger` — no importer. `OjaNeuron` is live; `sanger` is not, and nothing declares it exempt.
- `plasticity/bcm_plasticity.py::develop_selectivity` (and `_selectivity`, `_shape_patterns`) — `main()` only. `BCMNeuron` itself is live via `_BCMMetaGate`.
- `plasticity/competitive_learning.py::CompetitiveLearner` — the whole class has no importer; the FEELING implementation is reachable only through its own `__main__`.
- `plasticity/r_stdp.py::reward_population` (+ `calibrate`, `parallel_plan`, `partition`, `_FanOut`) — zero live callers, disclosed in-file at `:180-199`, and explicitly noted as **invisible to both P3 gates** (module-level function, so `method_dormancy_gate` has no opinion; `dormancy_gate` passes on the module's account).
- `faculties/empowerment.py::Empowerment.reward` — zero call sites, disclosed at `:90-109` with a reasoned refusal to merge into `step_reward`.
- `sleep/developmental_pruning.py` — the entire module has no functional caller (only docstring references from `lifelong_consolidation` and `reasoning/developmental_loop`); it is covered by `dormancy_gate.EXEMPT:245`.

**Duplicated mechanism (P2 status):**
- **`kwta` is clean** — exactly one definition (`action_gating.py:46`); `superior_colliculus` imports it rather than keeping its byte-identical copy. No violation.
- **Open, declared P2 debt:** `intrinsic_motivation.CuriosityDrive` and `spiking_curiosity.SpikingCuriosityDrive` are BOTH live, selected by a `spiking_curiosity` flag in `autonomous_loop` that `brain.py` sets True. `spiking_curiosity.py:45-54` states this plainly and explains why the measurement supports retiring neither. A flag keeping both paths is the additive shape P2 names.
- **Near-duplicate, adjudicated:** `latent_q.LatentQ` vs `mlp_q.MlpQ` — kept as a measured capacity tradeoff (MlpQ wins XOR 0.993 vs 0.467, loses the linear rule 0.983 vs 0.993), so neither dominates.
- **Magnitude-pruning is implemented twice** — over synapses in `developmental_pruning._prune_magnitude` and over units in `lifelong_consolidation._prune_to_budget`. The docstring calls this "reimplemented faithfully… one level up". Same rule, two levels of granularity; defensible, but it is a second site for one rule and the synapse-level original has no caller.

**Hardcoded constants where a running-baseline value belongs (P1):**
- `faculties/acetylcholine.py:38` `MODE_TH = 0.5` — an absolute cut on a leaky-EMA level with no running baseline behind it.
- `faculties/locus_coeruleus.py:54-55` `SALIENCE_TH = 0.50`, `PHASIC_TH = 0.45` — the exact shape `DehardcodeSource` was built to remove, and the module's own docstring records that a running-baseline swap here was rejected by the gate.
- `faculties/cortisol.py:22-24` `TAU/GAIN/MAXMOD` — three fixed behaviour-arbitrating scalars, no calibration path.
- `faculties/raphe_serotonin.py:38-39` `K_HI=2.0 / K_LO=0.02` — fixed discount endpoints.
- `plasticity/naca.py:45-47` `ETA/LAM_INV/THETA_MAX` — the gate's shape constants; `_consolidation_reversal_check` overrides all three (`eta=0.82, lam_inv=1.0`), so the shipped module defaults are not the values the self-check validates.
- `plasticity/meta_plasticity.py:62-84` — a fifteen-constant block; `_TUNED_LR = 0.12` is explicitly a value chosen so mechanism (b) still shows a gain, which is tuning against the module's own benchmark.
- `faculties/brain_decision.py:136` `self.alpha = 0.8` — measured NOT load-bearing at its shipped value; declared, not silently kept.
- `infra/brain_native_selfbuild.py:118` `_FUTILE_LIMIT = 3` and `:367` `MIN_SAMPLES = 50` — both declared in-file as P1 debt with reasons, matching none of P1's four exceptions.

**Criterion-quality note (positive):** across all three directories the self-checks consistently pair each claim with a control that must fail, delete rather than weaken criteria with no reachable red (`criticality_regulator:354`, `self_build_loop:349`, `developmental_envelope:179`, `brain_native_selfbuild:1028`, `snn_mnist_stdp_genn:566`, `spiking_curiosity:474`), and keep honest negatives asserted as failures rather than fixed (`generalizing_novelty` collapse inheritance, `deep_stdp` deep-hurts, `drive_reduction` ties-the-baseline, `meta_plasticity` stationary null, `brain_decision` fusion null). Two instrument defects were found and recorded rather than smoothed: `mlp_q`'s cosine criterion could not discriminate any alignment rule, and `drive_reduction`'s membrane control was testing the seed rather than the mechanism.</result>
<usage><subagent_tokens>434525</subagent_tokens><tool_uses>50</tool_uses><duration_ms>446832</duration_ms></usage>
</task-notification>

---


## Region digest 3 of 25

<task-notification>
<task-id>a0def43bd1abfa33d</task-id>
<tool-use-id>toolu_01Mm86koWgVX53GeBdzpahCj</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a0def43bd1abfa33d.output</output-file>
<status>completed</status>
<summary>Agent "Read cortex C in full" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 34 files read in full and claims verified. Digest:

## 1. COUNT

**34 .py files** — `paradigm_b_engine, paradigm_b_matcher, pc_cortex_serve, percept_surprise, placement, population_encoding, predictive_coding, prefrontal_executive, quality_diversity, rank_order_coding, reasoning_curriculum, recognizer_bank, recursive_tom, resonate_and_fire, rose_language, rsnn_encoder, second_order_access, self_improvement_suite, self_model, snn_classifier, snn_moe_classifier, spike_preprocessing, spiking_lm_serve, srm_neuron, task_set, tempotron, theory_of_mind, theta_gamma_code, time_cells, trained_brain, vsa_reasoning, vsa_structure_mapping, wholistic_loop, world_kb_stream` (13,882 lines; no `o*` file exists in this dir)

## 2. PER MODULE

**paradigm_b_engine** — In-storage spike-query engine over `.spk` telemetry: `match_sequence(events,ordered,W)` (ordered polychronous-group detector, candidate state machine) + `SpikeQueryEngine.{coincidence,sequence}`. Only `match_sequence` is consumed (by `sensorimotor/gesture_recognition.py:31`); `SpikeQueryEngine` has no external constructor. Reads `substrate/spike_telemetry_hub.{disk_query,synth}`; partial-seek is the point (queries touch only named channels).

**paradigm_b_matcher** — CPU reference for the LIF coincidence detector. `compile_query`, `match_stream` (raw distinct-channel window), `subdetector_match` (per-channel one-shot refractory; the form the GeNN port mirrors), `total_count_match` (deliberately WRONG detector kept as the spam control). Consumed by `paradigm_b_engine` and `recognizer_bank.ParadigmBRecognizer`. Gotcha: `match_stream` and `subdetector_match` do *not* return equal match lists — the docstring says "must equal" and the self-check text then explains the delta.

**pc_cortex_serve** — numpy, torch-free forward server for a torch-trained deep spiking predictive-coding hierarchy (`snn_pcnet/1`). `PcCortexServe.load` is the trust boundary (validates format tag, dims chain, gen_weights count/shape, prec length, readout shape); `settle/infer/classify/reconstruct/generate/encode/predict/make_goal_value/surprise/make_efe_value`. Duck-types into `PrefrontalExecutive.plan()`. **Invariant:** `predict()` deliberately ignores `command` (PC has no action channel) so `plan()` ties across options — that is disclosed, not a bug. Its real planning contribution is `surprise()` as the EFE epistemic term.

**percept_surprise** — Online unsupervised surprise = normalized reconstruction residual off a top-k Oja/Sanger subspace (`plasticity/oja_pca.OjaNeuron`, stacked + Gram-Schmidt deflated), standardized against a running EMA mean+variance. `PerceptSurprise.{observe,surprise_only}`. Consumed by `engine/brain.py`. **Gotcha:** `main()` raises `AssertionError` (not SKIP) when `models/nmnist_pack.npz` is absent — deliberate, because SKIP exits 0 and the gate scores exit status.

**placement** — Every grown neuron gets a fractional cortical depth, a layer, and an immutable Dale sign from the H01 prior. `layer_of`, `assign_sign`, `place`, `layer_histogram`, `DEPTH_BANDS`, `LAYER_RULE_AGREEMENT=0.9195`. Reads `growth/cortical_prior.{LAYER_CIRCLES,LAYER_E_FRACTION,LAYER_OCCUPANCY}`. **Two traps pinned by asserts:** `LAYER_CIRCLES[0]` is WHITE MATTER (not L1) and circles 1..5 descend 6→2, with L1 the outer remainder; and Layer-1 excitatory is refused *structurally*, not just because the measured fraction is 0.0.

**population_encoding** — RBF population code for a continuous scalar. `PopulationEncoder.{activations,encode,decode}`; sigma tied to center spacing. Consumed by `theory_of_mind`, `recursive_tom`, `quality_diversity`, `self_improvement_suite`, `faculties/spiking_pid`. Gotcha: `decode` returns `None` on an all-silent population — callers must handle it.

**predictive_coding** — 2-level spiking Rao-Ballard coder; signed ON/OFF LIF error populations, numpy-vectorized. `PredictiveCodingNetwork.{reset,set_state,infer_step,infer,infer_batch,learn_step,learn,errors,energy,representable_input}`. `learn_step` is Hebbian-on-spikes (`w += lr_w * outer(cum_e/n_steps, r)`) and is the per-tick hook `LiveBrain.tick` consumes. **Invariant:** `s_e0`/`s_e1` stay Python *lists* because callers do `net.s_e0 + net.s_e1` (concatenation, not array add).

**prefrontal_executive** — Spiking working memory + Wang-2002 WTA decision attractor + gated updating + top-down bias + model-based `plan()`. `PrefrontalExecutive.{deliberate,update,plan,push_subgoal,maintain,top_down_bias,goal,goals}`. Widely consumed (brain, pc_cortex_serve, others). **Gotcha:** goals/bias/winner are *identical on every seed* (measured seeds 0..199); only reaction time is stochastic — the self-check asserts that inverted invariant deliberately.

**quality_diversity** — Bounded MAP-Elites / Go-Explore over spiking policies. `SpikingPolicy` (12 LIF neurons, closed-loop drift in a 2-D latent box), `behavior_descriptor`, `param_descriptor` (control), `QualityDiversityArchive`, `random_search`, `greedy_hillclimb`. Consumed by `self_improvement_suite._rung_qd`. **Honest negative kept in-file:** a scale-matched random search matches/beats QD on raw coverage; the archive is load-bearing only on the deceptive task.

**rank_order_coding** — Thorpe rank-order (gain-invariant) matcher. `rank_code`, `RankOrderMatcher.label`. Enrolled on `sensorimotor/signal_loop.build_default_library`; beats the Van Rossum template under a monotonic rig-gain change (asserted `rank_gain == rank_clean` exactly).

**reasoning_curriculum** — Transitive-inference env + neurogenesis-grown spiking learner the sleep loop trains. `ReasoningEnv`, `InconsistentEnv` (cyclic control), `ReasoningLearner` (VSA hypervector identity → coincidence cleanup → recruit; Bradley-Terry rank nudge; `decide()` routes through `faculties/evidence_accumulation.DriftDiffusion`), `develop`, `MemorizationLearner`. `state_dict/load_state_dict` persist the skill. **Honest negative asserted:** the small→large curriculum does NOT beat jump-to-largest.

**recognizer_bank** — The P2 MERGE: six thin adapters over existing spiking recognizers behind one `recognize(features)-&gt;(label,confidence,known)`. `RecognizerBank`, `build_default_bank`, `RecognizerBankExpert`. Gates on margin + agreement + ≥1 *gatekeeper* member. **Gotcha:** `MoeRecognizer` seeds *and restores* the module-global RNG so its decision stays a pure function of input; the OOD floor is a rate (0.70 over 200 draws), not all-or-nothing.

**recursive_tom** — N-agent depth-k nested beliefs over VSA role-filler propositions. `RecursiveToM.{observe_event,predict_first_order,predict_nested,predict_chain,divergence,world_content}`; `_spiking_readout` does PopulationEncoder→LIF-WTA. Consumed by `brain.py` and `engine/autonomous_loop.py`. **Invariant:** `observe_event` is the ONLY mutator; the world always moves but a belief path updates only if every agent on it witnessed — that asymmetry *is* the false-belief substrate.

**resonate_and_fire** — Izhikevich resonator (damped 2-D oscillator) = spiking band-pass filter. `ResonateFire.{step,run}`. A bank decomposes a two-tone signal without an FFT. Detection threshold is `&gt;= 2` spikes (a lone spike is the onset transient). Referenced only from sibling neuron-model docstrings + `neuron_zoo`.

**rose_language** — Murphy R.O.S.E assembly (not a new model): grounded lexis (predictive-coding latent + VSA hv bound to a `world_kb_stream` referent), VSA constituency, theta-gamma/polychronization timescale structure, SLDA-RFF competence head with a Forward-Forward local refit. `RoseLanguage.{ground,parse,compose,unbind,constituency,order_by_phase,dependency_order_ok,competence,nightly_refit,comprehend,generate}`. Never raises out of a public method. **Known degeneracy documented:** `_learnable_relations()` uses `&gt; MEAN`, so a single-relation KB returns `()` and competence reports unavailable.

**rsnn_encoder** — Pure-stdlib serving half of a DCLS-delays + adaptive-LIF encoder (`snn_rsnn/1`). `RsnnEncoder.{features,forward_spikes,load_weights,save_weights,_validate_geometry}`; integer-tap ring-buffer delay lines. Its only importer is `reasoning/rsnn_delays_torch.py` (ungoverned tier).

**second_order_access** — Spiking metacognition *of* workspace access: a LIF pool excited by availability/sustain/first-order confidence and hard-inhibited by ignition failure. `SecondOrderAccess.{observe,second_order_confidence,hold,calibration,state}`. Consumed by `brain.py`. **Invariant:** confidence is read from EMITTED SPIKES, never the subthreshold membrane; `state()` must contain no phenomenal-vocabulary key (asserted against `_BANNED`). Calibration is deliberately kept &lt; 1.0.

**self_improvement_suite** — Composition layer: `census(seed)` runs 8 self-improvement rungs at reduced scale under one seed, each guarded to `SENTINEL=-999.0`. `brain.py:461` imports `census`. **Exact invariant:** `growth_exact_diff` must be exactly 0.0 even at reduced scale.

**self_model** — The brain tracks its own signals and narrates freshly. `SelfModel` (per-signal `(last, ema, trend)` with fixed-decimal EMA, `confidence()` = PRODUCT of three uncertainty proxies, bounded ring + durable autobiographical trace via `substrate/lake_working_memory`), `SelfNarrator.{introspect,benevolence}`. **Gotcha:** the durable goal band is mod-8, so recall is faithful only for goals 0–7; `close(remove=True)` is required to avoid litter.

**snn_classifier** — The original trainable spiking classifier; supervised local delta rule on spike counts. `templates,noisy,dataset,forward,train,evaluate,save_weights,load_weights,extra_templates,difficulty_sweep`. `N=64`/`STEPS=20` are the width contract for `spike_preprocessing`, `recognizer_bank`, `snn_moe_classifier`, `trained_brain`. **Invariant:** `forward()` seeds a per-sample RNG from the sample bits so recognition is a pure function of `(W, sample)`.

**snn_moe_classifier** — Ported Nord SpikeDrivenMoE: firing-rate routing (no learned router), top-k sparse experts, homeostatic load balance. `sample_spikes,expert_scores,route,expert_forward,forward,train,evaluate,usage_entropy`. Learning is an e-prop-shaped three-factor delta (no `.backward()`), and it IS on a live path via `recognizer_bank.MoeRecognizer`. **Invariant:** `_new_rng()` seeds a numpy Generator *from* the module-global `random`, which is what preserves the consumer's determinism contract; `main()` snapshots/restores the global RNG in a `try/finally`.

**spike_preprocessing** — The encoding/distance substrate. `encode_latency` (deterministic, ≤1 spike/channel), `encode_poisson`, `precompute/save_cache/load_cache`, `van_rossum_filter/distance` (closed-form geometric fast path, guarded per call, value- and decision-identical to the general loop), `victor_purpura_distance`, `denoise`. Widest consumer set in the slice. **Gotcha:** the fast path is only taken when *every* channel has ≤1 in-range spike.

**spiking_lm_serve** — Zero-torch forward for a trained spiking LM (`snn_lm/1`), RWKV-style linear-attention state + Brian2-compatible subtract-reset LIF. `generate,save_lm,load_lm,_forward_logits`.

**srm_neuron** — Spike Response Model: membrane as a sum of `eps` (PSP) and `eta` (soft-reset/refractory) kernels. `SRMNeuron`, `IFNeuron` (hard-reset control), `run`. **Gotcha:** `potential()` sums over *all* stored `inputs`/`fires` with no pruning — O(t) per step, unbounded growth on a long run.

**task_set** — Executive attentional SET + set-shifting (Wisconsin-style). `TaskSetLayer.{bias,step,observe,shift,active_set,last_switch_cost}` + `_RunningMean`. Persistent-activity LIF pool holds the active set; ACC conflict/reward monitor gates a three-factor R-STDP vote over candidate sets (reuses `plasticity/r_stdp.RSTDPNeuron`). Consumed by `brain.py`. **The good pattern:** "conflict is high" is judged against a *learned* running mean of good-tick conflict, which freezes during a bad patch so the trigger does not habituate.

**tempotron** — Gütig-Sompolinsky spike-TIMING neuron, one per command (one-vs-rest). `Tempotron.{vmax,train}`, `TempotronMatcher.label`, `_pattern`. Enrolled on the same `build_default_library` + `encode_latency` codes; used as a gatekeeper in `recognizer_bank` (trained to stay SILENT for non-members = its intrinsic OOD signal).

**theory_of_mind** — Sally-Anne false-belief tracker. `BeliefPopulation` (PopulationEncoder code + LIF WTA readout), `TheoryOfMind.{place,move,predict_from_belief,predict_from_world}`. `predict_from_world` is the control that must FAIL (asserted `world_acc == 0.0`).

**theta_gamma_code** — Lisman-Idiart phase-multiplexed working memory: K gamma slots inside one theta cycle, one LIF population per slot. `ThetaGammaCode.{encode,encode_sequence,readout,slot_of,phase_of,spike_events,spike_raster,raster_with_ensembles}`. Consumed by `rose_language`, `gwt_deep`, `grid_cells`, `synfire_chain`. **Causal arrow (corrected in-file):** capacity K is a kwarg and `steps_per_theta = K * steps_per_gamma` is derived FROM it — not the reverse.

**time_cells** — Ramping-to-threshold population clock, geometrically tiled. `TimeCellClock.{tick,decode,learn_interval,grow,grow_if_earned,would_earn,heldout_rmse,resolvable,first_passage_sd,save,restored}` + `interval_grid`. Persists learned thresholds INTO `brain_lake.bin` under `_T_STATE` (P4b), proven across a real subprocess boundary. **Invariants:** growth only at a reset boundary; `learn_interval` REFUSES an order-inverting update rather than clamping; `_restore_refusal` refuses on any seed/drive/noise_sd lineage mismatch.

**trained_brain** — Zero-torch inference over GPU-trained multi-layer spiking weights (`snn_brain/1`). `predict_scores,predict,accuracy,make_brain,save_brain,load_brain`. **Invariant asserted:** a single-layer brain must equal `snn_classifier.forward` argmax exactly (`faithful == 1.0`).

**vsa_reasoning** — The spike-polarity VSA substrate for the whole slice. `hv,bind,bundle,permute,coincidence,coincidence_many,sim,cleanup,is_spike_code,record_recovery`, `D=512`. `coincidence_many` is the numpy-batched form, proven bit-exact against the scalar counter (it exists because `coincidence` was 74.5% of the live wake tick via `episodic_recall`). **Invariant:** `coincidence_many` rows must stay in INSERTION order so `argmax`/stable-argsort tie-break matches `max()`.

**vsa_structure_mapping** — Gentner structure-mapping analogy over VSA codes. `StructureMapper.{map,score,encode,_resonator_mapping,_hillclimb,_signature,_prop_hv}`. Consumed by `reasoning_cortex`, `analogy_bench`, `brain.py`. **Gotchas:** `_prop_hv` memoization narrowed the accepted input — arguments must now be HASHABLE (a list arg raises TypeError); `_prop_cache`/`_target_cache` are unbounded and safe only because every caller builds a fresh mapper.

**wholistic_loop** — L13 capstone: one arbitrated non-regressing tick composing CuriosityDrive + MetacognitiveMonitor (the spine) + LifelongConsolidator (grow/prune/replay) + CriticalityRegulator. `WholisticLoop.{run,_tick,_perceive,_heldout_acc}`. Each spine part is individually ablatable so the self-check has a failing control for each. **Honest disclosure in-file:** criticality→competence is a hand-built perception-fidelity bridge, not emergent, so the `iso` arm exists to remove that confound.

**world_kb_stream** — The fact store + masked-slot (KB-completion) adapter. `load_kb,_bundled,_extra_facts,masked_tail_stream,tail_vocab,_most_frequent_tail_baseline`. `load_kb`/`_bundled` deliberately avoid numpy at module scope so the reasoning bridge can import them numpy-free. Grounds `rose_language`. Honest bar: the readout must only MATCH-OR-BEAT the per-relation most-frequent baseline.

## 3. REGION NOTES

**Four layers, bottom-up.** (a) *Neuron models* — `srm_neuron`, `resonate_and_fire` are scalar single-cell primitives kept in stdlib on measured merit (numpy pays on populations, not one neuron). (b) *Codes/substrates* — `spike_preprocessing` (latency + Van Rossum/Victor-Purpura), `population_encoding` (RBF), `vsa_reasoning` (two-line spike-polarity HDC), `theta_gamma_code` (phase). These are the widest-fan-out nodes; `spike_preprocessing.{N,T}` and `snn_classifier.N` are the same 64-wide contract that `recognizer_bank` depends on. (c) *Faculties* — recognition (`snn_classifier`→`tempotron`/`rank_order_coding`/`snn_moe_classifier`, merged by `recognizer_bank`), perception (`predictive_coding`→`pc_cortex_serve`, `percept_surprise`), executive (`prefrontal_executive`→`task_set`), social (`theory_of_mind`→`recursive_tom`), timing (`time_cells`), self (`self_model`, `second_order_access`), language (`world_kb_stream`→`rose_language`), analogy (`vsa_reasoning`→`vsa_structure_mapping`). (d) *Composition* — `self_improvement_suite` (8-rung census), `wholistic_loop` (arbitrated tick), `quality_diversity` (outer QD loop).

**Main data structures.** Spike-event lists `[(t, channel)]` (preprocessing/paradigm-B); ±1 hypervectors of length D=512 (VSA family); RBF spike-count vectors (population family); float64 weight/cause arrays (predictive coding, MoE); JSON weight artifacts with a format tag + validating loader (`snn_pcnet/1`, `snn_rsnn/1`, `snn_lm/1`, `snn_brain/1`, `snn_classifier.W/1`) — five separate serve-frozen formats, all with the same trust-boundary shape.

**Connection to the rest of the brain.** Upward into `engine/brain.py` and `live_brain.py`: `percept_surprise`, `recursive_tom`, `task_set`, `second_order_access`, `wholistic_loop`, `self_improvement_suite.census`, `reasoning_curriculum`, `vsa_structure_mapping`, `time_cells`. Sideways: `basal_ganglia` (`r_stdp`, `evidence_accumulation`), `hippocampus` (`experience_replay`, `growing_learner`, `spiking_storage_prototype`, `function_preserving_growth`, `sleep_consolidation`), `plasticity` (`oja_pca`, `benna_fusi`, `meta_plasticity`, `criticality_regulator`, `lifelong_consolidation`), `neuromodulation` (`intrinsic_motivation`, `spiking_curiosity`, `drive_reduction`, `generalizing_novelty`), `cerebellum` (`forward_model`, `polychronization`), `brainstem` (`signal_loop` command library), `substrate` (`home`/`_T_STATE` lake, `lake_working_memory`, `spike_telemetry_hub`, `_mathutils`).

**Learning rules present, and none is backprop.** Delta rule (`snn_classifier`), three-factor e-prop-shaped delta (`snn_moe_classifier`), Hebbian-on-spikes (`predictive_coding.learn_step`), Oja/Sanger (`percept_surprise`), Bradley-Terry/Elo (`reasoning_curriculum`), R-STDP (`task_set`, `time_cells.learn_interval`), tempotron peak-voltage gradient, Forward-Forward/CSDP (`rose_language.nightly_refit`), streaming SLDA sufficient statistics. Every torch-trained artifact enters through a serve-only validating loader — the training rule never crosses into this slice.

## 4. DEFECTS

**Dormant (defined, never called outside its own `__main__`)**
- `readout/predictive_coding.py::PredictiveCodingNetwork.infer_batch` — **zero callers tree-wide** (verified). Its docstring claims "This is where the numpy form actually pays" — a performance justification for a path nothing takes. P3 shape 1.
- `faculties/self_improvement_suite.py::SelfImprovementSuite` — the class has no importer; `brain.py:461` imports the bare `census` function. Its docstring ("so the unified Brain can hold and run the whole self-improvement ladder as one object") describes a wiring that does not exist.
- `substrate/paradigm_b_engine.py::SpikeQueryEngine` — never constructed outside its own `main()`; only the module-level `match_sequence` is consumed (`gesture_recognition.py:31`). The class *is* the module's headline claim.
- `sensorimotor/spike_preprocessing.py::{encode_poisson, denoise}` — zero external importers. `denoise` is self-described as "opt-in (callers pass already-encoded events)" — there are no callers.
- `readout/snn_classifier.py::{extra_templates, difficulty_sweep}` — reachable only via `python -m ... snn_classifier sweep`, which no gate or entrypoint invokes.
- `sensorimotor/rsnn_encoder.py::RsnnEncoder` — its only importer is `reasoning/rsnn_delays_torch.py:598`, i.e. the ungoverned P4c tier. No governed consumer.
- `sensorimotor/recognizer_bank.py::RecognizerBankExpert` — self-described "NOT registered by default"; its only importer is `reasoning/reasoning_cortex.py:1031`, also ungoverned.
- `faculties/percept_surprise.py::_NoveltyBaseline` — self-check-only class (documented as such, but it is live code in the module body).

**Duplicated across modules (P2)**
- `_cosine` is defined **7 times** in `src/`: `plasticity/eprop_kernel.py:177`, `faculties/metacognitive_monitor.py:397`, `faculties/wholistic_loop.py:146`, `faculties/affective_memory.py:68`, `growth/growing_learner.py:40`, `reasoning/eprop.py:282`, `reasoning/weight_perturbation.py:164`. The 2026-08-15 pass that merged 8 copies of `clamp01` into `infra/_mathutils.clamp01` did not do the same for cosine — and `wholistic_loop` already imports `clamp01`/`mean` from `_mathutils`, so the seam is open and unused.
- **The spiking WTA belief readout is written twice.** `faculties/theory_of_mind.py::BeliefPopulation.believed_location` and `faculties/recursive_tom.py::_spiking_readout` are the same mechanism (PopulationEncoder → one LIFNeuron per bin → most-firing wins, ties to lowest index) with the same four constants **re-declared verbatim** in both files: `_WINDOW=20`, `_TAU_M=10.0`, `_DRIVE=0.3`, `_MAX_SPIKES=12`. `recursive_tom`'s docstring even says it uses "exactly the readout theory_of_mind.BeliefPopulation uses" — it reimplements it instead of importing it.
- `_seed_of` exists in both `readout/snn_classifier.py:86` (bit-packing) and `sensorimotor/trained_brain.py:35` (crc32 of quantized bytes) — same name, same role, different arithmetic; a reader will assume they agree.

**Stale (comment/docstring contradicting the code or the tree)**
- `sensorimotor/pc_cortex_serve.py` — line 7 cites the trainer as `research/pc_cortex_torch.py`; **that path does not exist** (verified). Line 272 of the *same docstring* cites it correctly as `reasoning/pc_cortex_torch.py`. One file, two cites, one wrong.
- `sensorimotor/rsnn_encoder.py` — cites `research/rsnn_delays_torch.py` (module docstring, twice) and `research/developmental_loop_slda.py`. The actual file is `src/snn_data_lake/reasoning/rsnn_delays_torch.py` (verified).
- `sensorimotor/world_kb_stream.py` (docstring tail) — "the `language_readout_slda` reach remains governed-&gt;reasoning until C5 lands". `main()` imports `from snn_data_lake.readout.language_readout_slda import SLDA_RFF_np` — the reach is governed→governed today; C5 landed and the note did not.
- **Retired-rule residue advertised in four module titles.** `trained_brain.py` ("zero-dep"), `spiking_lm_serve.py` ("Zero-dep inference"), `rsnn_encoder.py` ("Pure stdlib ... zero-dep forward"), `snn_classifier.py` ("zero-dep JSON", "clone-and-run zero-dep install"). The zero-dependency rule was abolished 2026-07-20 and superseded by P7. Two siblings in this same slice (`resonate_and_fire.py`, `srm_neuron.py`) carry explicit dated P7 correction notes for exactly this wording; these four do not.
- `substrate/paradigm_b_matcher.py::subdetector_match` docstring — "it must equal match_stream". It does not, by design (different refractory), and the module's own print then explains the delta. The docstring states an invariant the code deliberately violates.
- `sensorimotor/recognizer_bank.py` module docstring calls `spiking_storage_prototype` "FACTORED k-WTA associative-memory recall"; `StorageRecognizer`'s own docstring calls the same thing "Hopfield ASSOCIATIVE memory".

**Global-RNG leak — same defect, fixed in one module and not its neighbours (P2/determinism)**
- `readout/snn_classifier.py::main` line 249 calls `random.seed(0)` with **no save/restore**. `sensorimotor/spike_preprocessing.py::main` line 252 does the same. `readout/snn_moe_classifier.py::main` documents this exact leak with a measurement ("after `random.seed(4242)` + two draws ... LEAKED") and fixes it with a `getstate/seed/setstate` in a `try/finally`. `run_selfchecks.py` runs every module in ONE process, so these two still clobber the RNG stream of every module that runs after them. The fix pattern is already in the same directory.

**Hardcoded where a learned/adaptive value belongs (P1)**
- `faculties/second_order_access.py::SecondOrderAccess.__init__` — `w_avail=0.35, w_strength=0.45, w_conf=0.35, w_fail=0.90, c_hold=0.30, avail_majority=0.5`. These fuse the four first-order signals and `c_hold` directly arbitrates `hold()`, i.e. whether the brain acts on content. None is baseline-relative, and none is fitted from the module's own running statistics — contrast `task_set._RunningMean`, which does exactly the P1 form for the same kind of judgement in the same slice.
- `faculties/task_set.py::TaskSetLayer.observe` — `self._shifter.reward(dopamine if dopamine &gt; 0.0 else 0.05)`. A bare `0.05` floor injected into the R-STDP gate whenever the RPE is non-positive; every other quantity in that method is derived from the learned baseline or the reward sign.
- `faculties/wholistic_loop.py` — `FIDELITY_FLOOR=0.45`, `FIDELITY_SLOPE=0.55`, `SUBSTRATE_NOISE=0.40`. The module itself names these "a BUILT bridge ... hand-authored", and the `abl_crit["aggregate"] &lt; full["aggregate"] - 0.05` assert rides on them. Honest, but still three tuned constants deciding a self-check verdict.
- `faculties/prefrontal_executive.py::__init__` — `_theta = 0.26 * bound` (a magic multiplier converting a decision bound into a sustained-firing-rate threshold), plus `_g_ext=0.55, _b0=0.02, _w_plus=0.50, _w_cross=0.50, _w_ie=1.2, _init_v=0.10, _trace_decay=0.90, _wm_gain=0.55`. The WTA winner and the chronometric RT both depend on them.
- `faculties/reasoning_curriculum.py` — `_VIGILANCE=0.30` (the neurogenesis recruit threshold), `_DRIFT_SCALE=2.2`, `_DECAY=0.03`. `_VIGILANCE` decides whether an item is novel enough to grow a unit — an arbitration, not a shape parameter.
- `faculties/time_cells.py` — `GRID_PER_GAP=10`, `GRID_TEST_STRIDE=5` at module scope, governing the held-out split that `grow_if_earned` gates growth on. The file argues the *discipline* was ported from `ng_snn.prune_unearned`; the two numbers are still literals.
- `faculties/theta_gamma_code.py` — `steps_per_gamma=6`, `tau_m=8.0`, `i_drive=0.8`, `vth ∈ [0.9, 1.1)`, all hand-set so each population "still fires by step 2". The capacity K is properly a kwarg; the oscillation's own timing is not.
- `sensorimotor/recognizer_bank.py` — per-member abstention cutoffs written inline and independently: `0.5*thr` (tempotron), `0.05` (LCA), `0.05` margin (MoE), `0.5` overlap (storage), `0.34` (paradigm-B), plus `min_agree=0.5, min_margin=0.15`. Seven magic thresholds deciding out-of-distribution rejection, none derived from a running baseline.

**Other**
- `growth/srm_neuron.py::SRMNeuron.potential` sums over the full `self.inputs` and `self.fires` history with no window or pruning — O(t) per step and unbounded memory. Fine for the 40-step self-check; a live consumer stepping it for thousands of ticks would degrade quadratically. The `eta` kernel decays to ~0 after a few `tau_r`, so a bounded window is safe.
- `sensorimotor/trained_brain.py::main` imports `from snn_data_lake import snn_classifier as scl` — the pre-3.0 flat alias, while the module's real home is `snn_data_lake.readout.snn_classifier`. It resolves via the back-compat layer in `__init__.py`, but it is the only such import in this slice and will break silently when that layer is retired.</result>
<usage><subagent_tokens>413144</subagent_tokens><tool_uses>42</tool_uses><duration_ms>464610</duration_ms></usage>
</task-notification>

---


## Region digest 4 of 25

<task-notification>
<task-id>a805283625c36c880</task-id>
<tool-use-id>toolu_01NCDyERJ71hUQtjZLWgTrLb</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a805283625c36c880.output</output-file>
<status>completed</status>
<summary>Agent "Read cortex B in full" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>Both parallel readers returned. Full digest:

## 1. COUNT

**28 .py files read in full** (bare filename starting g–n, non-recursive):
`generative_self_model, gesture_recognition, heterogeneous_microcircuit, hodgkin_huxley, integration_measure, interaction_learning, is_snn_forward, izhikevich_neuron, kb_ranking, laminar_local_readout, language_acquisition, language_readout_slda, language_self_build, lca_sparse_coding, learned_attention, learned_evc, learned_matcher, lif_neuron, live_brain, metacognitive_monitor, multimodal_encoder, nef_compute, neural_sampling, neurogenesis_engine, neuron_zoo, neuronal_avalanches, nightly_skills, nl_interface`

22 read directly; the 6 largest (`nightly_skills`, `live_brain`, `metacognitive_monitor`, `neurogenesis_engine`, `neuron_zoo`, `interaction_learning`) read end-to-end by two parallel full-readers, whose wiring claims I spot-verified.

## 2. PER MODULE

**lif_neuron** — Canonical LIF primitive; `LIFNeuron{step,run,reset_state}`. The most-reused symbol in the slice (15 cortex modules import it). Gotcha: refractory clamps `v = v_reset` and integrates *nothing*, so a refractory neuron discards current rather than accumulating it.

**izhikevich_neuron** — 2-var (v,u) phenomenological cell; `simulate()`, `REGIMES` dict. No class — only a function + params, so `heterogeneous_microcircuit._IzhCell` and `neuron_zoo` each wrap it with their own stepping shim. Gotcha: `simulate` records the spike *before* reset, so `spikes[0]` can precede any real upstroke.

**hodgkin_huxley** — The one genuinely biophysical cell (Na m³h / K n⁴ conductances, forward Euler dt=0.01 ms). `HodgkinHuxley{step,run,run_const,spike_count}`. Only consumer is `neuron_zoo._HodgkinHuxleyCell`. Gotcha: `alpha_n`/`alpha_m` have guarded removable singularities at V=−55/−40 mV; `seed` is accepted and unused.

**is_snn_forward** — Stacked LIF with intrinsically-stable sparse init (depth-invariant rate, no BatchNorm) + geometric per-layer tau hierarchy. `ISSNNForward{forward,layer_embeddings}`, `_tau_schedule`. Consumed by `spiking_jepa`, `cortical_prior`, `placement`. Gotcha: scalar `tau_m` must stay byte-identical to `[tau]*n`.

**neuronal_avalanches** — Galton-Watson branching cascades with σ as explicit control parameter; `BranchingNetwork{avalanche,run,measured_branching_ratio,powerlaw_slope}`. Consumed by `brain.py` and `reservoir/criticality_regulator`. Gotcha: `measured_branching_ratio` is only unbiased on a **fully observed** trace — subsampling drives a critical system to read subcritical (documented, deliberately not asserted).

**neural_sampling** — Boltzmann MCMC via single-site Gibbs on stochastic sigmoid neurons; `BoltzmannSampler{step,run,sample,clamp,analytic_distribution}`. **Zero real importers tree-wide** — `dormancy_gate:255` EXEMPT-permanent ("LIBRARY: no live query asks for samples from an energy model"). Gotcha: constructor hard-rejects asymmetric W and nonzero diagonal.

**nef_compute** — NEF represent + transform: LIF tuning curves + ridge least-squares decoders (numpy/LAPACK). `NEFEnsemble{rates,activity,solve_decoders,decode}`. Consumed by `nengo_bridge` (cross-engine ground truth) and `reasoning/nef_integrator`. Gotcha: `X_INT_RANGE`/`R_MAX_RANGE`/`T_REF`/`TAU_RC` are a **contract** with `nengo_bridge` — changing them deletes the comparison, not just the numbers.

**lca_sparse_coding** — Rozell LCA (leaky integration + lateral inhibition) + `LCAMatcher` recognizer over the enrolled command library. Consumed by `recognizer_bank`. Gotcha: `encode` default steps 300→80 for perf; accuracy measured flat, but the Gram matrix is O(M²) per step.

**integration_measure** — PCI (perturb → causal footprint → Lempel-Ziv). `PerturbationalComplexity{pci,gzip_pci,causal_response,mean_firing_rate}` + 4 regime factories. Consumed by `brain.py`, `region_coherence`, `consciousness_probe`. Gotcha: **one PCI number is not reportable** — only contrasts are; the class has no port for an external system's state, and the brain's old `pci` telemetry was deleted 2026-08-05 because it read one constant value across 900 ticks.

**neuron_zoo** — ONE heterogeneous population behind one contract; 9 cell adapters + `Population`, which *races* serial vs each thread count on real steps. `NeuronZoo`, `make_neuron`, `sample_population`, `SAFE_KINDS`/`OPTIN_KINDS`. Gotcha: the committed plan holds index **ranges**, never cell objects — caching slices breaks serial/parallel byte-identity.

**heterogeneous_microcircuit** — Worked demo composing LIF+Izhikevich+AdEx through `basic_components` with local STDP. `HeterogeneousLayer`, `run_circuit`. **Zero real importers** (both `basic_components` mentions are comments); `dormancy_gate:217` EXEMPT-permanent "PALETTE EXAMPLE". Gotcha: `self.cells` must stay bound to `Population`'s copied list or the layer reports one population and steps another.

**multimodal_encoder** — Fuses 4 modalities into banded [0,1]^64 feeding `encode_latency`. `encode_multimodal()` (fixed) + `MultiModalEncoder` (growable width). Six live consumers (`autonomous_loop`, `neurogenesis_engine`, `growing_learner`, `multisensory_integration`, `esp32_encoder_path`, `hjepa_arm_torch`) — **all call the fixed function**. Gotcha: band averaging keeps [0,1] regardless of element count; `None` → zeros, never an exception.

**generative_self_model** — Learned e-prop/predictive-coding forward model of the brain's own self-signal; `GenerativeSelfModel{predict,observe,reset}`, surprise = settled LIF spike rate. **Live and charged**: `brain.py:842` builds it (`dim=11`), `brain.py:2428 _self_surprise` returns its output as the brain's surprise. Gotcha: `reset()` must clear the trace but never the learned `w`.

**learned_attention** — Learned reticular gate (salience × surprise × goal × plastic synapse → divisive normalization → spiking WTA), three-factor `learn(reward)`. **Live and charged**: `brain.py:919` builds it, `brain.py:1302 _attend_real_stream` calls `attend()` and its gate *is* the arm's sensory channel. Gotcha: `learn()` reads eligibility from the *last* `attend()` — reordering silently decouples reward from the gate that earned it.

**learned_evc** — PFC/ACC control allocation with dopamine-learned cost/value scalars replacing `executive_control.evc`'s fixed `cost=0.1`. `LearnedEVC{allocate,learn,state,signature}`. Consumed by `brain.py`, `autonomous_loop`, `executive_control`, `metacognitive_monitor`. Gotcha: only `engaged=True` trials carry eligibility — the weights encode engage-vs-withhold *contrast*, so feeding only engaged trials learns nothing.

**learned_matcher** — Benchmarks template vs learned spiking classifier vs hybrid (learned + Van Rossum novelty gate). `train_learned`, `hybrid_match`, `calibrate_gate`. Consumed by `recognizer_bank`. Gotcha: `hybrid_match` returns `(idx, dist, encoded)` so `signal_loop` reuses all three — do not shrink the tuple.

**gesture_recognition** — Order-aware IDLE/APPROACH/RETREAT via `paradigm_b_engine.match_sequence` on NEAR/MID/FAR bins. Consumed by `live_arm`, `demo`. Gotcha: falls back to synthetic data when `robot arm/*.csv` is absent — a green run may not have touched real captures.

**nl_interface** — Safety firewall: an LLM only ever emits a *label*; the command string is this module's own constant, re-checked against `serial_arm.is_valid_command`. `NLInterface.interpret`, `StubBackend`, `ARM_LABELS`. Consumed by `live_brain.py:114`. Gotcha: cue matching is **word-boundary** anchored and questions/negations are rejected — except `stop`, deliberately exempt. Bare-substring matching previously issued arm commands on 15 of 16 ordinary sentences.

**laminar_local_readout** — No-backprop drop-in for LaminarReadout: L4 fixed tanh expansion → L2/3 feedback alignment through fixed random B → L5 local delta rule → L6 fixed random gain, plus neurogenesis-freeze. `LaminarLocalReadout{train_step,predict,grow,observe,infer}`. Live: `brain.py` builds it per real percept (in_dim=64) and its `infer()` confidence gates the percept drive. Gotcha: `_observe_lr` is a running-baseline step (μ/EMA of ‖[a4,1]‖²) — the `+1` is *derived* (the bias unit's presynaptic activation), not an epsilon; the old `1e-12` form drove `max|b1|` to 2.6e12 and pinned confidence at 1.0 on a silent percept.

**kb_ranking** — Open-vocab KB completion; from-scratch numpy DistMult with negative sampling + filtered Hits@k/MRR. `DistMult`, `load_dataset`. Consumed by `nightly_skills:117`, `reasoning_cortex`, `schema_induction`. Gotcha: falls back to bundled facts offline — the headline Hits@10 means different things on the two paths.

**language_self_build** — Autonomous self-supervised spiking assemblies (phonetics/semantics/syntax) via R-STDP, curiosity + curriculum, consolidated to the lake. `_positional_code`, `_train_eval`, `_bigram_baseline`, `_unigram_baseline`, `_frequent`. Live on the sleep tick via `nightly_skills`. Gotcha: the SYNTAX order-sensitivity result is a **measured NULL** (sign is seed-chosen, 2/6 positive); its assert was deleted, not weakened.

**language_readout_slda** — SLDA-on-Random-Fourier-Features decoder over the *same* spiking context code — the honest "spiking representation + strong decoder" hybrid. `SLDA_RFF_np{observe,scores,predict}`, `_split`, `_ngram_acc`. Gotcha: the equal-budget comparison is the real one — at equal data SLDA-RFF roughly **ties** the best n-gram; the "decisive win" was a 15× data confound.

**language_acquisition** — Online brain-native char-level acquisition: predictive-coding latent + local error-modulated Hebbian readout, no backprop. `LanguageAcquisition{expose,generate,developmental_curve,save,load}`, `load_lm`. Live: `brain.py:4064` builds it, `:3124`/`:4209` save, `:4071` load. Gotcha: proof of learning is the **developmental curve**, not fluency.

**live_brain** — Composition layer wrapping `AutonomousArm` with every region on; adds opt-in predictive coding, the grounded `converse()` firewall, and a 17-region telemetry snapshot. `LiveBrain{tick,converse,chat,alive,spike_snapshot,recall}`. Only constructor is `brain.py:229`. Gotcha: `workspace` must stay object-**identical** to `arm.workspace` (a copy is the P6 defect); `tick()` swallows every predictive-coding exception, so a broken PC path degrades silently.

**metacognitive_monitor** — Lesionable spiking reliability monitor: LIF pool excited by confidence+LP, inhibited by surprise-without-progress and above-baseline conflict; gates `learn_gate/explore/grow_trigger/hold`. Consumed by `brain.py:293`, `wholistic_loop`, `brain_decision`, several reasoning modules. Gotcha: the HOLD bar is a running quantile of the pool's own rate — never re-introduce an absolute floor; `lesion=True` must stay byte-identical to no-monitor.

**interaction_learning** — R-STDP intent router: rank-order latency code → one `RSTDPNeuron` per label → tempotron peak-membrane readout → `w[i] += lr*d*e[i]`. `SpikingInteractionLearner`, `text_features`. Live via `live_brain.py:125`. Gotcha: `t_post &lt; n_in` is the generalisation knob — `t_post=n_in` memorises.

**nightly_skills** — One bounded night of autonomous cross-domain skill formation on the sleep loop: pick a frontier domain, study a budget slice, register in the proximity router, plus open-world discovery and autotelic goal-babbling. `Domain`, `_NightKB`, `NightlySkills{night,save,load,route,state,close}`. Sole live caller `brain.py:3076-3090`. Gotcha: `_ensure_lake()` **empties** `&lt;home&gt;/skills.spkl` on every construction — that file is scratch; durable memory is `&lt;home&gt;/skills/*.npz`. `_EXEMPLAR_CAP=150` is a persisted-format contract.

**neurogenesis_engine** — Grows NG-SNN units predicting the brain's own next perception; refits seeded from the prior model (grow-on-top), prunes earn-your-place. `NeurogenesisSelfImprove{tick,growth_state_bytes,restore_growth_state,substrate_is_retained}`. Live at `brain.py:1151` (restore) and `:3098` (sleep-tick persist into the ONE lake under `_T_GROWTH`). Gotcha: `_placement` must stay index-aligned with `model.W` across both growth and prune; never bump `GROWTH_FORMAT` — restore refuses on tag mismatch and would discard every existing record.

## 3. REGION NOTES

The slice is four strata stacked on one primitive.

**Cell layer** (`lif_neuron`, `izhikevich_neuron`, `hodgkin_huxley`) — a fidelity ladder from imposed-spike LIF through reduced phenomenological Izhikevich to conductance-based HH. `neuron_zoo` is the P2 unifier: nine cells behind one `step(current)-&gt;0/1` contract, plus `Population`, which decides serial-vs-threaded by *racing on real steps* rather than a width threshold. `heterogeneous_microcircuit` is the worked composition demo above it.

**Circuit/measurement layer** (`is_snn_forward`, `nef_compute`, `neural_sampling`, `neuronal_avalanches`, `integration_measure`, `lca_sparse_coding`) — each answers a distinct question about spiking dynamics: depth-invariant propagation, computing f(x) by decoder solve, posterior sampling, criticality statistics, perturbational complexity, sparse coding by lateral inhibition. Most are libraries; only `neuronal_avalanches` and `integration_measure` reach the live tick.

**Perception/control layer** (`multimodal_encoder`, `learned_attention`, `laminar_local_readout`, `learned_evc`, `metacognitive_monitor`, `generative_self_model`) — the live wake path. Flow: sensors → `encode_multimodal` → `LearnedAttention.attend` (gate *is* the arm's sensory channel) → `LaminarLocalReadout.observe/infer` (confidence gates percept drive) → `LearnedEVC.allocate` / `MetacognitiveMonitor` gates → `GenerativeSelfModel` self-surprise. Every learning rule here is three-factor local or feedback alignment; no backprop anywhere in the slice.

**Language/skill layer** (`corpus`-fed: `language_self_build`, `language_readout_slda`, `language_acquisition`, `kb_ranking`, `nightly_skills`, `interaction_learning`, `nl_interface`) — the sleep path. `brain._enter_sleep` → `nightly_skills.night()` → `Domain.study` → `language_self_build._positional_code`. The recurring honest finding across this layer: **spiking representation + strong decoder**; the pure spike-timing readout underperforms n-grams, and `language_readout_slda` closes the gap with SLDA-RFF while keeping the representation spiking.

`live_brain` is the composition seam binding all of it to `AutonomousArm`; `nl_interface` is the safety firewall on the chat→arm route. Persistence flows two ways: growth → `brain_lake.bin` under `_T_GROWTH` (P4b-correct), language → a `.npz` sidecar (P4b-violating, below).

## 4. DEFECTS

### Dormant (grep-verified, zero call sites outside the defining module's own `main()`)

- **`multimodal_encoder.py:MultiModalEncoder`** — the growable-width encoder has **zero** references in `src/` or `tests/`, while the fixed-width `encode_multimodal()` it was built to supersede has ~6 live consumers. The better mechanism is the dead one (P2 inverted + P3).
- `multimodal_encoder.py:MultiModalEncoder.add_modality/band/modalities` — dead with their class.
- `metacognitive_monitor.py:abstain` — zero call sites tree-wide, **and** it is a pure alias of `hold()` (P2 duplicate + P3 dormant in one symbol).
- `metacognitive_monitor.py:conflict_excess` — self-check only, undocumented.
- `live_brain.py:LiveBrain.run` (`:191`), `jepa_on` (`:320`), `plan_latent` (`:324`), `deliberate_ahead` (`:347`), `plan_goals` (`:376`) — no callers.
- `live_brain.py:teach` (`:531`) / `route` (`:544`) — **learning-verb methods with no live-path call site and no `method_dormancy_gate.EXEMPT_METHOD` entry**. This is the shape that gate exists to catch.
- `live_brain.py:425-431` — the `MemoryLake` branch of `converse()` is unreachable; nothing in `src/` ever assigns a `MemoryLake` to `arm.lake_wm`.
- `interaction_learning.py:route_text` (`:265`) — self-check only.
- `nightly_skills.py:_ID_NAMES` (`:138`) — module-level map with zero reads; `route()` rebuilds it inline.
- `nightly_skills.py:NightlySkills.route` (`:1036`), `.state` (`:1049`), `.close` (`:1069`) — no live-path caller. **The brain forms and persists a skill every night and never routes to one.** The uncalled `close()` is exactly why `_ensure_lake`'s Windows-OSError fallback exists — defect and workaround both shipped.
- `neuron_zoo.py:ALL_KINDS` (`:277`) — zero references, not even in its own `main()`.
- `neuron_zoo.py:make_neuron` (`:333`), `NeuronZoo.kinds` (`:298`) — own `main()` only.
- **`neuron_zoo.py:Population` / `sample_population`** — no live-tick consumer: one consumer (`heterogeneous_microcircuit`) is EXEMPT-permanent, the other (`interaction_learning._drive`) is behind a `readout_zoo` flag **no caller in the tree ever sets**. ~40% of the file (the measured population-parallelism mechanism) is charged only by self-checks.
- `neurogenesis_engine.py:self._pruned` (`:239`) — written every refit, read nowhere.
- `neurogenesis_engine.py:layer_census` (`:290`), `placement` (`:280`), `growth_fingerprint` (`:487`) — zero `src/` references.
- `integration_measure.py:gzip_pci` — self-check only. `is_snn_forward.py:layer_embeddings` and `language_acquisition.py:developmental_curve` — likewise.
- `laminar_local_readout.py:grow` — self-documented at `:209-215`: no live call site, escapes `method_dormancy_gate` because "grow" is not in `LEARN_VERBS`. Honestly recorded rather than fake-wired; correct handling.
- `neural_sampling`, `heterogeneous_microcircuit` — whole modules with zero real importers, but both are properly `dormancy_gate.EXEMPT`-with-reason. **Not violations**, listed for completeness.

### Duplicated across modules (P2)

- **`generative_self_model.py`** — the docstring headline says it "reuses primitives from `predictive_coding.py` and `eprop.py`, **does not reimplement them**". The module imports **only `random`**. `_lif_step` is a hand-copy of `predictive_coding._lif` (signed ON/OFF pair), and the eligibility trace, leaky readout and e-prop update are all re-written locally. Stale claim *and* a duplicated live mechanism.
- **`generative_self_model` vs `faculties/self_model`** — the former says it "replaces the hand-coded proxy aggregate in `faculties/self_model.py`", but `brain.py:52` still imports `SelfModel, SelfNarrator` alongside it. Adopt-without-retire: both survive.
- **Hand-rolled LIF where `lif_neuron.LIFNeuron` exists** — `learned_attention.py:_lif_rate` (`:104-114`) and `learned_evc.py:_population_gain` (`:127-143`) each write their own threshold+reset loop, while 15 other cortex modules import the canonical class.
- **Device selection twice inside one module** — `neurogenesis_engine.py:203` routes the refit through `infra.device.select` on a measured element count, but `:124` still does `device or ("cuda" if torch.cuda.is_available() else "cpu")`, and **that** availability-decided device is what `restore_growth_state:411` builds every restored model on. `main():629` repeats it a third time.
- `live_brain.py:_reply_to_intent` (`:30`) vs `interaction_learning.py:_brain_intent` (`:440`) — two divergent reply→intent mappers (7 vs 5 labels, different branch order); one builds the self-check's ground truth, the other teaches the live router.
- `live_brain.py:route`/`teach` reimplement `interaction_learning.py:route_text`/`learn_turn` (same body).
- `metacognitive_monitor.py:_World` (`:437`) duplicates `wholistic_loop.py:_World` (`:268`), and `:504-506` duplicates the engage gate at `wholistic_loop.py:418-420`.
- `nightly_skills.py:Domain` vs `_NightKB` — duck-typed twins with two separate `study()`/`signature_exemplars()` bodies.
- `nightly_skills.py:282-291` MT19937 npz (de)serialization vs `neurogenesis_engine.py:333/377` vs `substrate/home.py` — three independent "learned arrays → opaque blob" serializers.
- `live_brain.py:255` (`hippocampus`) and `:276` (`substrate`) are the **same expression** — two telemetry "regions" that can never differ.

### Stale docstrings/comments contradicting the code

- `neurogenesis_engine.py:35-39` — "DEVICE IS MEASURED, NOT ASSUMED (P5a)… this line read `cuda if available else cpu`". Line `:124` still reads exactly that.
- `live_brain.py:497-498` — `_reasoning_reply` says "Off by default (reasoning=False)" while the signature is `reasoning=True` (`:81`) and `:84` says "DEFAULT". Self-contradiction inside one file.
- `live_brain.py:385,473` — routes commands through "gated `ArmSim.apply`"; **`ArmSim` does not exist in `src/`** (the live gate is `So101Arm.apply`). This is the same dead name `nl_interface.py:20-23` and `CLAUDE.md` both already corrected — it survived here.
- `live_brain.py:196-211` — `spike_snapshot` claims real per-tick activity for "EACH of the 17 regions"; `olfactory` is a literal `0.0`, `substrate` is a copy of `hippocampus`, `midbrain` is constant on every live path.
- `metacognitive_monitor.py:12-13,41-42` — advertises GROW as a live gate (`grow_trigger` has no `brain.py` consumer); `:43-46` documents `hold()/abstain()` as a pair when `abstain` is dead.
- `neuron_zoo.py:321` — `rng = random.Random(seed)  # …determinism anchor`, consumed at `:325` as `rng.randint(0, 0)`, identically 0. A vacuous draw described as an anchor.
- `neuron_zoo.py:316` — `sample`'s "each neuron gets a distinct sub-seed" is true for `autapse`/`hh` only; four other cells accept `seed` and ignore it.
- `nightly_skills.py:1043-1045` documents `_ID_NAMES` as superseded, in a comment, without deleting it.
- `nightly_skills.py:_sweep_orphan_scratch` (`:536`) — docstring promises "Returns the number removed"; the only caller (`:520`) discards it.
- `interaction_learning.py:451` — tests `"working-memory goal" in r`, a phrasing `converse()` no longer emits; the branch survives only via a looser fallback.
- `brain.py:258-260` cites `interaction_learning.py:93/:206/:229`; actual lines are `:92/:200/:224` — line cites re-drifted after the last correction.
- `neurogenesis_engine.py:727` prints `(&lt;{cfg['l_max']})` → renders `(&lt;None)` while the assert beside it bounds `buffer_cap`=1000.

### Hardcoded where a learned/adaptive value belongs (P1)

- `neurogenesis_engine.py:178` — `holdout = min(60, n // 3)`; the denominator of **every accuracy claim in the module**, and the one fixed value here with no P1 disposition note (contrast `l_max`/`lr`/`batch`, which all have one).
- `neurogenesis_engine.py:99-115` — `peak=0.75`, `jitter=0.12`, `eps_tol=0.28`; `peak`/`jitter` set how predictable the world is and `eps_tol` decides when a grown unit **earns its place**. No dispositions.
- `metacognitive_monitor.py:152-170` — `hold_p` was correctly converted to a running quantile (`:220`), which makes `r_explore=0.30` and `nov_hi=0.4` — absolute cutoffs on the *same* reliability scale — an inconsistency the module's own reasoning refutes. `base=0.4` (passed at `brain.py:293`) is a raw tonic LIF current.
- `neuron_zoo.py:68-71` — `_ADEX_SCALE=1500 / _RES_SCALE=130 / _HH_SCALE=45 / _IZHI_SCALE=55`: fixed per-kind current gains that arbitrate which kind dominates a mixed pool's argmax, set by comment rather than derived from each model's measured rheobase.
- `live_brain.py:239` — `wm_load = n(wm / 200.0)`, and the whole 17-row map `:252-277` is hand-picked coefficients **presented as measured spike rates** (this is what makes the `spike_snapshot` docstring above a reporting defect, not a cosmetic one).
- `interaction_learning.py:435` — `1.0 + 0.1*(pos+1) + 0.01*len(word)`: magic tie-breakers that silently make word position and word **length** part of the spike code.
- `learned_attention.py:62-68` — `lr=0.35`, `sigma=0.5`, `alpha_goal=1.5`, `alpha_surprise=0.6`, `exp_decay=0.2`, `lif_gain=5.0` on the **live** gate that feeds the arm's sensory channel; no P1 disposition. (Contrast `laminar_local_readout._observe_lr`, which did the conversion properly — that is the in-slice model to copy.)
- `is_snn_forward.py:23-29` — `STABLE_GAIN=2.5`, `WEAK_GAIN=1.0`, `SPARSITY=0.4`, module-level and named "hard-coded" by the docstring itself.

### P4b violation (live)

**`language_acquisition.py:232 save` / `:236 load`** write the learned language weights `W` to a `np.savez` **sidecar** (`language_acq.npz`) beside `brain_lake.bin` rather than into it — the arrangement P4b forbids in those exact words. Confirmed live: `brain.py:3124` (sleep consolidation), `:4209` (shutdown "never-forget"), `:4071` (load on the language-organ build path). The module documents this against itself at `:218-231`; the fix is a lake record tag plus three `brain.py` call sites.

### E-prop invariant

**No violation found in any of the 28 files.** Zero `.backward(` / `autograd` / `requires_grad` / `torch.optim` across the slice. Learning rules are three-factor R-STDP (`interaction_learning`, `learned_attention`, `learned_evc`), feedback alignment (`laminar_local_readout`), local delta/Rescorla-Wagner (`language_acquisition`), e-prop readout gradient (`generative_self_model`), closed-form ridge (`nef_compute`, `language_readout_slda`, `neurogenesis_engine`'s NGSNN), or analytic non-temporal gradients on a model with no time axis (`kb_ranking.DistMult`, where BPTT is undefined).</result>
<usage><subagent_tokens>263560</subagent_tokens><tool_uses>40</tool_uses><duration_ms>674953</duration_ms></usage>
</task-notification>

---


## Region digest 5 of 25

<task-notification>
<task-id>aa790dcd993f71c91</task-id>
<tool-use-id>toolu_01CK1SsGZNXt5xhsnSQFyFie</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\aa790dcd993f71c91.output</output-file>
<status>completed</status>
<summary>Agent "Read hippocampus+cerebellum" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 35 files read in full. Digest:

## 1. COUNT

**hippocampus/ — 25 .py files read**
`__init__.py, affective_memory.py, allocentric_map.py, content_signature.py, contrastive_episode_analysis.py, developmental_loop_brain.py, episode_consolidation.py, episodic_recall.py, experience_replay.py, function_preserving_growth.py, grid_cells.py, grid_place_map.py, growable_readout.py, growing_learner.py, hippocampal_memory.py, lake_consolidation.py, ng_snn.py, open_world_loop.py, proximity_replay.py, ring_attractor.py, skill_builder.py, sleep_consolidation.py, sleep_replay.py, spiking_grid_can.py, spiking_storage_prototype.py`

**cerebellum/ — 10 .py files read**
`__init__.py, forward_model.py, gain_adaptation.py, hjepa_serve.py, jepa_world_model.py, polychronization.py, spiking_jepa.py, spiking_pid.py, synfire_chain.py, wavefront_planner.py`

## 2. PER MODULE

### hippocampus/

- **`__init__.py`** — Docstring-only; records that `hippocampus.hippocampus` was RETIRED (Policy-3) with its 3 roles redistributed. No re-exports: consumers must import submodules directly. Consistent with `dormancy_gate.py:252`.
- **`affective_memory.py`** — Capacity-bounded affect-tagged episodic store; evicts by `_importance = |valence|+arousal+0.5|reward|` (not recency), retrieves by content (`recall_similar`) OR affect (`recall_by_emotion`), and `consolidate()` returns an importance-weighted sample to feed `ReplayBuffer`. **Live**: constructed `autonomous_loop.py:1920`; `recall_by_emotion` called `brain.py:3851`. Gotcha: keys are `zlib.crc32` of a 6-dp ASCII image of the fragment — near-identical fragments collide by design.
- **`allocentric_map.py`** — Landmark→(x,y) map with dead-reckoning drift and landmark-resight drift reset (`correct_gain` fraction of the discrepancy); `nearest(valence_pref)` recalls by confidence±valence. **Live** (`autonomous_loop.py:1965,1974`). Zero-coupling by design: keeps its own `est` rather than driving `grid_cells`.
- **`content_signature.py`** — 64-dim crc32 bag-of-tokens key (`signature`, `SIG=64`). Extracted from `skill_builder` so `spike_chat_lake` can import it without numpy. **Invariant: `SIG` is a persisted-format constant** — changing it orphans every stored key; the self-check pins `crc32(b"dog")%64==61`, `crc32(b"cat")%64==40`.
- **`contrastive_episode_analysis.py`** — Pools success/failure context dicts, ranks features by Cohen's-d (`analyze`), predicts success for a candidate context (`advice`). Constructed at `autonomous_loop.py:1938`. Gotcha: seeded jitter breaks exact |d| ties only.
- **`developmental_loop_brain.py`** — TRAIN-TIER MNIST continual loop wiring real brain subsystems (ConvLatencyStdp → readout, `HippocampalMemory`, `BrainDecision`, `PrefrontalExecutive.plan`, `LocusCoeruleus`×`DopamineModulator` lr fold, `GlobalWorkspace`). Bare `__main__` = bounded mechanics check; `DLB_RUN=1` = full 5.8-min run. **Invariant: must never be wired to a live learning consumer** — it trains by Adam+`.backward()` under an explicit e-prop exemption that expires the moment `dormancy_gate.reachable()` reaches it.
- **`episode_consolidation.py`** — Real episode dicts → K-of-N sparse assemblies → covariance-Hebbian attractor with ring eviction; `replay()` is balanced noise-driven reactivation + STDP; `recall_episode()` returns the readable dict behind a completed cue, or an honest miss. Called from `brain.py:935-962` (sized by `capacity_for`). Gotcha: `dim` is DERIVED (`required_dim = 4×capacity`); a narrower dim is refused, not degraded. `fanout=None` deliberately has no default (measured per deployment by `tools/fanout_bisect.py`).
- **`episodic_recall.py`** — VSA/HDC episodic memory over `faculties.vsa_reasoning`: bind role⊗filler, bundle into one trace; `complete()` pattern-completes held-out slots, `gist()` extracts family invariants, `apply_schema()` predicts slots of a never-stored situation. Live via `brain.py:786`. **Key invariant: both decision thresholds are DERIVED per call** from `_superposition_margin(S)`, `_prototype_margin(m,S)` and `sigma=1/sqrt(D)` — nothing frozen. `_RowStore` mirrors are append-only and must stay in insertion order or the fused argmax tie-break changes.
- **`experience_replay.py`** — Fixed-capacity ring of `(unit, action, reward)` + forward/reverse SWR sweep applying the live RPE rule to `q`. Widely consumed (`brain.py:53`, `sleep_consolidation`, `hippocampal_memory`, `quality_diversity`, `reasoning_curriculum`, `lifelong_consolidation`, `substrate/lake_memory`). **Safety invariant: `replay()` mutates only the value table** — never selects or emits an action.
- **`function_preserving_growth.py`** — Net2Net split on a LIF layer: copy incoming weights + membrane state exactly, halve both fan-outs → bit-identical output (`max abs diff == 0.0`, not a tolerance). `add_unit_gradmax` inserts a zero-fanout unit. Consumed by `faculties/self_improvement_suite.py:72`. Gotcha: exactness depends on `math.fsum` for the downstream current — a plain `+` fold breaks the last bit.
- **`grid_cells.py`** — Analytic oscillatory-interference grid cell (3 plane waves 60° apart), `rate()`/`step()` path integration, `rate_map()`. Reused (not reimplemented) by `grid_place_map`. Gotcha: integration is EXACT — no drift, no reset; that gap is `allocentric_map`/`spiking_grid_can`'s job.
- **`grid_place_map.py`** — Multi-scale grid population → one-shot Hebbian place cells binding percept↔location; `place_at()` content-queries, `recall_place()` decodes location from the template alone. Live at `brain.py:758`. **The firing bar is DERIVED per query** (`_match_bar` = midpoint between 1.0 and the population's off-field mean) plus a `1/sqrt(n_grid)` noise floor; the old fixed `place_threshold=0.5` was removed. `_key()` normalises every percept to `str` — a non-str percept previously raised on 28 of 31 live `observe` calls.
- **`growable_readout.py`** — Two continual-learning readouts. `GrowableReadout`: MLP whose hidden width grows by a **(1,0) split** (parent keeps fan-out, duplicate starts at zero) — the equal (0.5,0.5) halve is a phantom under Adam. `LaminarColumns`: per-task isolated L4+L2/3 columns + shared L5, whole-column freeze. Both freeze by masking gradients **and zeroing Adam moments** — without the second, momentum drifts a "frozen" row. Consumed by `developmental_loop_brain` and `reasoning/autonomous_brain`.
- **`growing_learner.py`** — ART-style online prototype recruiter over the 64-d fused multimodal vector; recruit when best cosine &lt; `vigilance`, else running-average refine. Heavily consumed (`brain.py:54`, `autonomous_loop`, `metacognitive_monitor`, `affect_integrator`, `lifelong_consolidation`, `similarity_sized_growth`). Gotcha: `_best` raises on a width mismatch rather than letting `zip` truncate.
- **`hippocampal_memory.py`** — Adapter mapping loop exemplars onto real hippocampus: one `ReplayBuffer` per digit (balance for free), `SleepConsolidator` for the sleep pass. Holds only row INDICES + labels, never features. Only consumer: `developmental_loop_brain`. Gotcha: `reward=1.0` is a pinned marker — `ReplayBuffer.replay` is never called here.
- **`lake_consolidation.py`** — Sleep-time prune of the SYMBOLIC lake: caps re-derivable ingest (wordnet/conceptnet) to a budget, ranks the learned cap by `RecallTrace` (use-it-or-lose-it), preserves append order. `RecallTrace` persists **into `brain_lake.bin` under `_T_STATE`** (P4b), proven by a subprocess readback; a legacy JSON sidecar is imported once and left inert. Live at `brain.py:3059`, bumped from `spike_chat_lake:315`.
- **`ng_snn.py`** — Contribution-grown LIF classifier + one-shot analytical ridge (no backprop). `l_max=None` DERIVES capacity from the training rows; an earn-your-place stop scores on rows the growth solve never fit. Consumed by `growth/neurogenesis_engine.py:50` and `tools/capacity_curve.py`. **Invariant: the RNG is CPU-bound whatever `device` is** — CPU/CUDA streams differ at the same seed, so binding it to the device made "same seed → same brain" false across hardware.
- **`open_world_loop.py`** — Discovers domains from a mixed unlabeled stream by DP-means/vigilance over frozen `signature()` space; bounded novelty buffer harvested and CLEARED (O(n) not O(n²)); merges centroids within λ. `OpenWorldController` + `dna_stream` consumed by `sleep/nightly_skills.py:561,645`.
- **`proximity_replay.py`** — `LakeMemory`: latency-encode exemplars, append to the durable `.spkl` on per-exemplar time bands, retrieve by Victor-Purpura proximity. `cross_process_durability_check()` compares the **writer's RAM digest** against a child process's disk digest — the parent side deliberately is NOT a second disk read (that form passed with 60/120 exemplars missing). Widely consumed (`skill_builder`, `nightly_skills`, `language_self_build`, `spike_chat_lake`, several reasoning loops).
- **`ring_attractor.py`** — 1-D CAN ring: persistent bump = angle memory, velocity input integrates to absolute angle. `ProprioceptiveEstimator` self-calibrates its shift gain and dead-reckons COMMANDED joint deltas, so an uncommanded disturbance diverges → slip signal. Live at `live_arm.py:67,379`. **Read-only observer — never commands the arm.**
- **`skill_builder.py`** — One domain-agnostic loop: stream → curiosity picks frontier → SLDA-RFF readout → consolidate signature exemplars to the lake → `SkillRegistry.route()` by proximity, no domain id. `SkillRegistry`/`math_stream` consumed by `nightly_skills` and `open_world_loop`.
- **`sleep_consolidation.py`** — Wake/quiescent lifecycle wrapping `sleep_replay.AttractorNet` (the spiking driver that earns the retention gap) with `ReplayBuffer` rate replay alongside. Asserts only the sleep-vs-no-sleep GAP; absolute retention is printed "fragile". **Honest negative baked in: LP-priority provably adds nothing** (asserted equal). Consumed by `hippocampal_memory` and `self_improvement_suite:76`.
- **`sleep_replay.py`** — Golden-2022 attractor net: homeostatic budget normalisation is the forgetting substrate; noise-driven replay + covariance STDP restores task A. `accuracy()` is a fused numpy batch path; `main()` compares it CELL-BY-CELL against scalar `recall()` with a reachable red. `downscale_prune` (SHY) called at `brain.py:2840`. **Invariant: the batch accumulation must stay k-OUTER**, not `Z @ W.T`.
- **`spiking_grid_can.py`** — 2-D CAN spiking grid sheet (numpy, difference-of-von-Mises, separable → O(n³)); moving LIF bump path-integrates velocity; `observe_landmark` imprints Hebbian once and CORRECTS on re-sighting. Live at `brain.py:837,1541`. **Kept failure: a graded mixture at anchor/bump gap &gt; ~0.21 period extinguishes the bump (amp→0.000)**; `capture_threshold=0.2` makes large conflict re-clamp instead. Numpy port is 1-ulp, not bit-identical.
- **`spiking_storage_prototype.py`** — Factored (never materialised) Hopfield k-WTA associative memory; criteria judged against a do-nothing cue control AND a permutation null, each with an executed RED (`RED_CONFIGS`/`check_reds`). Live consumer verified: `sensorimotor/recognizer_bank.py:232 StorageRecognizer`. Gotcha: numpy was measured SLOWER at every live width (0.64–0.67×) — a rejection with a measurement, per P7.

### cerebellum/

- **`__init__.py`** — One-line docstring only.
- **`forward_model.py`** — Efference-copy forward model + comparator over an abstract 1-D plant: `predict`/`residual`/`detect`/`update` (LMS). Cancels reafference, flags exafference. Consumed by `live_brain.py:363`, `prefrontal_executive.py:372`, and `jepa_world_model`'s own baseline. **Read-only — emits no command.** `ForwardModel.update` has a `method_dormancy_gate` EXEMPT entry (`:95`).
- **`gain_adaptation.py`** — Scalar flocculus LMS gain calibrator whose defining property is SIGN FLIP: `dg = lr*(W-g)*u²` drives g through zero to −|W| when the world reverses. Dead-zone rejects sub-threshold noise. Live at `autonomous_loop.py:38,1979`.
- **`hjepa_serve.py`** — Forward-only server for a torch-trained 2-level H-JEPA (`snn_hjepa/1` JSON): `encode0/encode1/predict0/predict1`, `predict` aliases `predict1` so it duck-types into `PrefrontalExecutive.plan`. **Trust boundary: `load()` validates format tag, geometry, and every layer's shape AND chaining**; no `save()` by design. Live at `autonomous_loop.py:2677`.
- **`jepa_world_model.py`** — Minimal linear JEPA: encoder `unit(tanh(We x))`, EMA/stop-grad target, residual action-conditioned predictor, hand-rolled VICReg. Wz is decayed + row-scaled contractive and predictions clamped → 40-step rollout bounded. **`_feedback_gz` returns `list(gp)` — the `Wz^T gp` transport term is removed** (measured 0.627× held-out MSE, better on 5/5 seeds); the e-prop assert perturbs Wz and requires the backward signal unchanged. Live via `autonomous_loop.py:40` (`jepa=True` on the wake tick).
- **`polychronization.py`** — Delay network where LIF coincidence detectors ignite only when delayed spikes converge; two PNGs over one shared anchor pool, selected by TIMING alone (rate pattern is silent). Numpy inbox is bit-exact with the retired dict-of-dicts because arrival order is unchanged. Consumed by `faculties/rose_language.py:377`.
- **`spiking_jepa.py`** — Genuinely spiking + genuinely local JEPA: event retina → 2 recurrent LIF layers → readout; three-factor local rule `lr·m_j·sg_j·e_ij`. **Two transport sites were fixed 2026-08-04**: `_feedback_gz` (residual skip only) and `_readout_grad`'s top-down modulator (now feedback alignment through a fixed random `_Br` drawn from a crc32 stream, so existing weights stay byte-identical). Claims 3 and 5 are explicitly DROPPED to telemetry — the ablation does not collapse and learning does not beat a random encoder. `_Br` is deliberately NOT serialized but IS re-shaped on load.
- **`spiking_pid.py`** — LIF-population PID: P = matched ON/OFF firing rate, I = cumulative net integrator spikes (leak 0.9995 ≈ perfect integrator), D = fast/slow opponent rate difference. `_calibrate()` derives the NEF decode weights per instance. `drive_to` beats fair P-only/no-control baselines under load. Live at `live_arm.py:179,407`; drives `_PID_JOINT="shoulder_lift"` **by name, not index** (index 0 is shoulder_pan).
- **`synfire_chain.py`** — Feedforward pools of leaky coincidence neurons; a synchronous packet propagates to an attractor with SHARPENED jitter, a weak or dispersed one extinguishes. Live at `brain.py:914`. Gotcha: `_inject` uses a fixed derived seed so `propagate()` is idempotent.
- **`wavefront_planner.py`** — **No longer a wavefront**: bidirectional RRT over 5 joints (36⁶ ≈ 2.2B cells retired the grid), `shortcut()`, damped-least-squares Jacobian `ik()` on its OWN MjData, `execute()` via MOVETO. Live at `live_arm.py:208`. **Safety invariants: every motion leaves through `arm.apply()`; collision checks use `probe_contact` on a body that is never stepped; a partial path returns `None`, never a truncated plan; a latched (E-STOP) arm is provably not moved.** Limits come from `ctrl_lo/hi`, not `jnt_range`.

## 3. REGION NOTES

**hippocampus** splits into four sub-systems that meet only at the durable lake:

1. *Spatial* — `grid_cells` (analytic hex rate field, exact PI) → `grid_place_map` (Hebbian place cells binding percept↔location) → `spiking_grid_can` (real recurrent LIF sheet, drifts + landmark reset); `ring_attractor` is the 1-D heading twin; `allocentric_map` is the name↔(x,y) layer with its own drift model. Flow into `brain.py`: `_place_here` ← `place_at()` → `SpikingGridCAN.observe_landmark` (`brain.py:1541`) → R-STDP place-value table.
2. *Episodic* — `episodic_recall` (VSA bind/bundle traces, reconstruction + gist + schema transfer) and `episode_consolidation` (sparse attractor at capacity, readable read-out) are complementary; `affective_memory` supplies the somatic-marker/importance layer over `experience_replay`.
3. *Replay/consolidation* — `experience_replay` (rate SWR) + `sleep_replay` (spiking attractor + covariance STDP) fused by `sleep_consolidation`, adapted for the MNIST loop by `hippocampal_memory`, and applied to the SYMBOLIC lake by `lake_consolidation` (the only module that prunes `brain_lake.bin`).
4. *Growth/skills* — `growing_learner` (ART recruit) and `ng_snn` (contribution-grown spiking classifier) are the two neurogenesis mechanisms; `function_preserving_growth` and `growable_readout` are the exact-growth primitives; `content_signature` → `skill_builder` → `open_world_loop` is the skill-index/router chain over `proximity_replay.LakeMemory`.

**cerebellum** is two disjoint halves. *Motor*: `spiking_pid` (feedback), `gain_adaptation` (feedforward gain+sign), `forward_model` (efference copy comparator), `wavefront_planner` (RRT+IK) — all three motor modules reach `brainstem` and every command passes `So101Arm.apply()`. *Timing*: `polychronization` (delay-matched PNGs) and `synfire_chain` (zero-lag volleys) — deliberately distinct codes, cross-referenced in each other's docstrings. *World models*: `jepa_world_model` (live on the wake tick via `autonomous_loop`), `spiking_jepa` (spiking/local twin), `hjepa_serve` (frozen 2-level server). All three expose `predict(state, command)` so they duck-type into `PrefrontalExecutive.plan()`.

## 4. DEFECTS

**Dormant — defined, zero call sites outside their own `__main__`/definition (verified by tree-wide grep):**
- `contrastive_episode_analysis.py::ContrastiveEpisodeAnalyzer.top_features` — no callers anywhere, not even its own self-check.
- `contrastive_episode_analysis.py::ContrastiveEpisodeAnalyzer.advice` — called only inside its own `main()`; the class IS constructed at `autonomous_loop.py:1938`, so the *analyzer* is live but its READOUT half is not.
- `episode_consolidation.py::EpisodeConsolidator.episode_rate_observed` — `brain.py:957` names it in a COMMENT only; no call site.
- `episode_consolidation.py::EpisodeConsolidator.degree` — zero references tree-wide.
- `ng_snn.py::NGSNN.prune_unearned` — zero call sites (only its own docstring at `:75`); the counterpart `grow-on-top` seeding it exists to balance IS consumed.
- `proximity_replay.py::LakeMemory.retrieve_balanced` — own `main()` only.
- `growing_learner.py::GrowingLearner.grow_dim` — own `main()` only.
- `grid_cells.py::GridModulePopulation` — own `main()` only (the multi-scale population the docstring calls "the real entorhinal organization" drives nothing).
- `spiking_grid_can.py::SpikingGridCAN.known_landmarks` — zero references.
- `faculties/spiking_jepa.py::SpikingJEPA` — the whole module has zero consumers; it is EXEMPT in `dormancy_gate.py:294` as "LIBRARY/BENCHMARK", so this is declared, not hidden.

**Stale — docstring contradicts the code:**
- `affective_memory.py` header: *"WIRING (documented only -- this module edits nothing else)"* — FALSE since `autonomous_loop.py:1920` constructs the store and `brain.py:3851` calls `recall_by_emotion`.
- `contrastive_episode_analysis.py` header: *"Integration (documentation only -- not wired here)... would CONSUME... and FEED faculties/prefrontal_executive"* — FALSE for the class (`autonomous_loop.py:1938`); true only for the `advice`→executive half.
- `ng_snn.py:44` and `:47` both name `.venv-cu128` as the torch tier / run command. **That env was RETIRED 2026-08-05** (CLAUDE.md P5: `.venv-lake` now carries CUDA). A documented command that cannot run.
- `growable_readout.py:829` prints `"-- research"` in its self-check banner; the file was promoted to `hippocampus/` on 2026-08-11 and its own header says so.
- `spiking_grid_can.py` docstring pins measured drift figures (`0.1392 → 0.2955 → 0.4424 → 0.5950`, `66.9x`, `1.05 ms`) as prose. No assert reads them, so nothing reddens when they rot.

**Duplicated across modules:**
- `_clamp`, `_predict_pre`, `predict`, `make_goal_value`, `latent_std`, `_spectral_bound`, `_feedback_gz` — `faculties/spiking_jepa.py` copies all seven verbatim from `faculties/jepa_world_model.py` (self-labelled "copied from jepa_world_model"). Both `_feedback_gz` bodies are literally `return list(gp)`. This is two predictors for one job (P2).
- `_cosine` — `faculties/affective_memory.py` re-implements the function `growth/growing_learner.py::_cosine` already exports (and which `growth/similarity_sized_growth.py:61` imports by name).
- `_mean`/`_std` — `faculties/synfire_chain.py:55,59` re-implement `infra/_mathutils.mean`, which four sibling modules in these two directories import.
- `developmental_loop_brain.py::_LinearReadout` — self-declared verbatim copy of `reasoning/_loop_common.LinearReadout`, recorded as debt **M21** and knowingly not adopted (adopting would create governed→reasoning module-top imports).
- `growable_readout.py` — `_restore_adam` and `_zero_frozen_moments` exist twice (`GrowableReadout` and `LaminarColumns`). Documented as a deliberate non-merge (per-tensor vs name-keyed state), so this is disclosed duplication.
- `open_world_loop.py:51` imports `signature` from `skill_builder` (a back-compat re-export) rather than from `content_signature`, which it already imports at `:49` for `SIG`.

**Hardcoded where an adaptive/learned value belongs (P1):**
- `sleep_replay.py` module scope — `N=24, K=6, G_INH=0.10, BUDGET=2.5, A_STORE=0.30, STORE_REPS_A/B=5/4, CUE_FRAC=0.34, SLEEP_EPISODES=500, NOISE_AMP=0.9, NOISE_P=0.20, LR_SLEEP=0.10`. An entire hand-tuned block on a module the sleep tick depends on. Self-disclosed as "small, hand-tuned"; none is derived from a running baseline.
- `growing_learner.py::GrowingLearner.__init__` — `vigilance=0.85` is the neurogenesis threshold, frozen. This is the outlier in its own region: `grid_place_map.place_at` deleted its equivalent constant in favour of a derived bar, and `episodic_recall` derives BOTH its thresholds from store geometry.
- `spiking_jepa.py:94-95` — `_FF_GAIN = STABLE_GAIN * 3.0`, `_REC_GAIN = STABLE_GAIN * 1.4`: bare multipliers on an imported fixed point, with no measurement attached. Plus ~14 fixed `lambda_*`/`tau_*`/`espp_alpha` constants set in `__init__`.
- `jepa_world_model.py:118-124` — `gamma=1.0, eps=1e-4, z_clip=3.0, rho_max=0.9, wd=1e-3, lambda_roll=0.3` as "fixed internal constants (not exposed on the contract)". `lambda_roll` weights a term the module measured as load-bearing, so it is a tuned value with no exposure and no adaptation.
- `sleep_consolidation.py:78-81` — `_LR_VALUE=0.3`, `_REPLAY_BUDGET=16`, plus the a-priori `FORGET_MARGIN=0.30`/`RETAIN_MARGIN=0.20` gate thresholds.
- `episode_consolidation.py:83-100` — `ACTIVE_FRAC=0.08, LR_REPLAY=0.02, REPLAY_DROP=0.4, REPLAY_FLIP=0.15, REACTIVATE_MIN=0.7, RETAIN_THRESH=0.9, MIN_EPISODE_CONF=0.5`. (Contrast: `DIM_PER_EPISODE=4` carries its measurement, and `fanout` deliberately refuses a default — both good P1 practice in the same file.)
- `contrastive_episode_analysis.py:65-71` — labelled "tunables (hand-set)": `ADVICE_GAIN=0.5, ADVICE_MIN_EFFECT=0.5, ADVICE_TOP_K=8`.
- `hippocampal_memory.py:76-78` — `_DEFAULT_SLEEP_EPISODES=60`, `_DEFAULT_CAPACITY=8192`.
- `lake_consolidation.py:120-124` — `rederivable_budget=3000`, `learned_cap=250000`, both governing what survives the night.
- `wavefront_planner.py` — `max_step_deg=12.0, iters=600, goal_bias=0.2, step_deg=4.0, tol_deg=3.0, damping=0.05`. `step_deg` is the swept-collision resolution, i.e. a safety-relevant constant chosen rather than derived.
- `spiking_pid.py:100,104` — `kp=0.6, ki=0.05, kd=0.15, window=40, err_max=90.0`; `gain_adaptation.py:66` — `lr=0.02, dead=0.05, clip=3.0`.
- *(Legitimate exception, noted so it is not re-flagged:* `content_signature.SIG=64` *is a declared persisted-format constant with a pinned-bin assert.)*

**e-prop / BPTT (per the standing invariant):**
- `growable_readout.py::GrowableReadout.train_step` and `LaminarColumns.train_step` — `loss.backward()` + Adam. Self-declared "INHERITED E-PROP DEBT, restated not laundered"; permitted only because the module is TORCH_GATED and off the live tick.
- `developmental_loop_brain.py::_LinearReadout.train_step` — `.backward()` + Adam, with the dopamine multiplier scaling that path. Adjudicated permitted as train-tier offline, with an explicit binding constraint that making the module reachable converts it into a violation "that nothing would announce".
- No BPTT found in either region. `jepa_world_model._feedback_gz` and `spiking_jepa._feedback_gz`/`_readout_grad` were the two transport sites and both are closed (residual-skip broadcast, and feedback alignment via fixed random `_Br`), each with a constructed-and-run RED in its self-check. `ng_snn` is gradient-free (analytical ridge); `episode_consolidation`, `episodic_recall`, `sleep_replay`, `grid_place_map`, `spiking_storage_prototype` have no gradients at all.</result>
<usage><subagent_tokens>488504</subagent_tokens><tool_uses>49</tool_uses><duration_ms>501625</duration_ms></usage>
</task-notification>

---


## Region digest 6 of 25

<task-notification>
<task-id>a9ae68c381d5ed671</task-id>
<tool-use-id>toolu_01Lpc2Dc5awXARHweFBnFUic</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a9ae68c381d5ed671.output</output-file>
<status>completed</status>
<summary>Agent "Read cortex A in full" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 21 files read in full (the three largest via dedicated full-read agents; I read brain.py's tail, head and key symbols first-hand and independently verified every load-bearing claim below).

---

## 1. COUNT

**21 files, 14,601 lines** (no `f*.py` exists in this directory; `__init__.py` is outside the a–f slice).

`access_consciousness.py, adex_neuron.py, affect_core.py, agency_monitor.py, alif_neuron.py, autapse_neuron.py, autotelic_goals.py, brain.py, competence_executive.py, corpus_stream.py, cortical_microcircuit.py, cortical_prior.py, counterfactual_sim.py, crossmodal_binder.py, deep_stdp_latency.py, dendritic_neuron.py, divisive_normalization.py, eprop_kernel.py, eprop_nmnist.py, eprop_shd.py, executive_control.py`

&gt; Note: `Get-Content | Measure-Object -Line` silently drops blank lines and under-counts every file (brain.py reads 4347 that way, 4595 truly). Use `[System.IO.File]::ReadLines`.

## 2. PER MODULE

**access_consciousness.py** (1,907) — GNW access index: grow neurons → VSA bundle → LIF rheobase ignition in `GlobalWorkspace` → broadcast to N taps → `access_index = ignition × availability × integration`. One class `AccessConsciousness`; `bind_assembly`/`step`/`ignited`/`access_index`/`state`. Built once at `brain.py:2626` **with `ws=self.workspace`** (shared arm bus) and driven each wake tick; `_access_ema` reaches cockpit, telemetry, self_model, and the "are you conscious" reply. Gotcha: a guest must never call `_ws.step()` (only the owner clocks the arbiter), and the bus must arrive as a **constructor** param — taps register in `__init__`, so a later rebind is silently dead.

**adex_neuron.py** (163) — Brette–Gerstner two-variable AdEx (`class AdEx`), scalar stdlib, forward Euler, `_MAX_EXP_ARG=30` caps the spike upswing. Consumed by `growth/neuron_zoo.py:54` and `growth/heterogeneous_microcircuit.py:32`. Deliberately not vectorized (P2: one definition of the dynamics).

**affect_core.py** (574) — Russell-circumplex categorical emotion via spiking WTA over 8 LIF prototype units, affective contagion (`empathize`), and `SecondOrderToM` (nested belief populations). `Brain` uses `recognize()` per tick (`brain.py:3246`), `empathize()` only inside `respond()` (`:3739`), `_tom2.observe/predict_second_order` in the ToM dialogue path. The module's own P6 block records the honest null: own affect is a live constant.

*DATED NOTE 2026-09-16 (specs/022 T082): RETIRED at T082 (2026-09-16) -- the `SecondOrderToM` class and the `_tom2.observe` / `predict_second_order` dialogue path this entry describes were deleted under P2 after R5 measured `faculties/recursive_tom.py`'s `predict_nested` reproducing them on 1,200 of 1,200 ordered pairs over 200 constructed witness scripts (the comparison run against the live class before its deletion). The entry is left standing rather than rewritten. What remains live in this module is unchanged: `recognize()` per tick and `empathize()` inside the reply path (`evidence/t082-social-cognition.md` 2.1).*

**agency_monitor.py** (512) — per-channel controllability = streaming |Pearson(cmd, feedback)| over five EMAs, plus a two-counter `ok`/`lost` state machine with hysteresis. `AgencyMonitor.observe()` is fed every tick from `AutonomousArm.step`; `agency()`/`controllability()`/`status()` have zero readers. `warmup` is derived from `alpha` (`2.4/alpha`) — a genuine P1 conversion.

**alif_neuron.py** (176) — Bellec LSNN adaptive-threshold LIF (`V_th_eff = v_threshold + beta·a`), reduces exactly to `LIFNeuron` at `beta=0` (asserted). The neuron model the repo's e-prop learns on. Only consumer: `neuron_zoo.py:55`.

**autapse_neuron.py** (246) — TDA-LIF: one LIF with a bank of time-delayed self-connections; delayed-match self-check proves the autapses are load-bearing (ablate → chance). Consumer: `neuron_zoo.py:56`. `_LinearReadout.train` is plain logistic SGD — a readout head, not a claimed brain rule.

**autotelic_goals.py** (570) — SI-5 rung: `AchievedOutcomeBuffer` (self-set goals from own outcomes), `GoalConditionedReadout` (two `Tempotron`s over a state⊕goal bump code), HER relabeling in `collect_labels`, SAGG-RIAC region choice by re-pointing `SpikingCuriosityDrive`. Consumed by `sleep/nightly_skills.py:106` and `faculties/self_improvement_suite.py:54`. Headline is measured on frozen held-out goals — the one metric hindsight cannot inflate.

**brain.py** (4,595) — the composition root. One class `Brain` + `make_brain(**kwargs)` (the P4-sanctioned constructor) + a 9-claim `__main__` self-check. `__init__` (~920 lines) builds `LiveBrain`→`AutonomousArm`, then ~45 optional faculties each in its own `try`. `tick()` → `clock.step` → edge-only `_enter_wake`/`_enter_sleep` → `_wake_tick` or `_sleep_tick`. Public surface is exactly twelve methods: `perceive_event, propose_improvements, perceive_pc, recall_episode_durable, tick, state, respond, reason, imagine, self_improve, close` + the `workspace` property. Wake-phase ORDER is load-bearing (`_route_percept_to_action` and `_c1_tick` write signals consumed on the *next* tick). Determinism is asserted (two `seed=0` brains byte-identical incl. `self_state`), so nothing may read disk at construction outside a `self._persist` gate.

**competence_executive.py** (541) — `DopamineModulator` (RPE over competence gain → bounded lr multiplier via `DopamineNuclei`), plus the public `CompetenceDynamics` + `goal_value` injected into `PrefrontalExecutive.plan`. Consumer: `instruments/developmental_loop_brain.py:135,251,391`. Renamed at promotion because `executive_control` basename was taken (`brain_atlas.py:81` asserts basename uniqueness). `_self_dotted()` derives its own import path rather than hardcoding it.

**corpus_stream.py** (324) — real CMUdict + Gutenberg ingestion via `sensorimotor.web_ingest.fetch_corpus`, streaming `(context, next-token)` pairs at phoneme and word granularity, with a bundled offline sample. Consumers: `nightly_skills`, `language_self_build`, `language_readout_slda`, `skill_builder`, `open_world_loop`. Caches to `tempfile.gettempdir()` — deliberately outside the lake (P4b: caches are not learned state).

**cortical_microcircuit.py** (603) — Potjans–Diesmann 8-population LIF column, downscaled with the variance-preserving `w/√scale` correction. `thalamic_drive(thalamic_hz)` is a real, anatomically-targeted input port (nonzero `CONN_PROB_TH` for L4e/L4i/L6e/L6i only), exactly 0.0 at 0 Hz, refuses negative rates. Its own comments record two hard-won lessons: the parallel form loses at every scale (Amdahl 1.34× ceiling, RNG stream must stay serial), and `PORT_SEEDS` was raised 3→6 after a measured 8.6% false-red rate on the layer-specificity assert.

**cortical_prior.py** (341) — the one place an externally-sourced constant lives: H01 human-cortex reference data with commit hash + per-file SHA-256, every entry carrying `value/units/provenance/accuracy_bound` (`validate()` fails an entry missing any). Consumed via `growth/placement.py:52`; its dormancy exemption was **retired 2026-08-16** once `neurogenesis_engine → placement → cortical_prior` reached a live root. `e_fraction()` returns `None`, never a default. Must never enter the lake.

**counterfactual_sim.py** (293) — learned latent forward model `z_next = z + W[a]·z + b[a]` (persistence prior) trained by e-prop-style eligibility × broadcast error, plus a `do()`-operator ranking candidate actions by imagined rollout. Live at `brain.py:2234/2250`. The docstring's derivation of why the one-step trace collapses to the raw pre-synaptic vector is correct and worth preserving.

**crossmodal_binder.py** (456) — binds a vision and an audio latent into one recoverable object by composing three existing primitives: `MultisensoryCell` (super-additivity, congruence mapped onto the collicular spatial rule), `vsa_reasoning` spike-XOR bind (self-inverse → either modality unbinds the other), and `Claustrum` (coincidence-gated assembly). Live at `brain.py:1508/1510`.

**deep_stdp_latency.py** (424) — Kheradpisheh-2018 anti-collapse machinery: DoG ON/OFF → latency rank-order → conv-IF first-spike two-axis WTA → multiplicative soft-bound STDP. Fully unsupervised; labels touch only the final neuron-label vote. `ConvLatencyStdp` is the perception tier for six reasoning-tier loops. `TwoLayerNet` is now an **alias** to `instruments/deep_stdp.TwoLayerNet` (merged 2026-08-16, 2 copies → 1). Acceptance criterion is kernel diversity &lt; 0.6, not accuracy.

**dendritic_neuron.py** (195) — two-compartment pyramidal cell with BAC firing: apical+basal coincidence ignites a Ca²⁺ plateau whose burst rate exceeds the basal-only maximum (supralinear). Consumer: `neuron_zoo.py:57`.

**divisive_normalization.py** (161) — Carandini–Heeger `r_i = d_i^n/(σ^n + Σd_j^n)`; stateless, instantaneous pool gain control, distinct from homeostatic scaling (slow, set-point) and LCA lateral inhibition (subtractive). Live: lazily imported at `autonomous_loop.py:950` and applied at `:963`.

**eprop_kernel.py** (435) — the ONE torch e-prop engine both perception encoders import: `_SpikeFn` surrogate, `EpropLIFBase` (`psi`/`flatten_frames`/`offdiag_mask`/`forward`), `bptt_grads` (the independent autograd reference), `_cosine`, `_apply_grads`, `evaluate`, `EPROP_BPTT_COSINE_FLOOR=0.5`, and a surrogate-OFF negative control. Documents its own scope limit honestly (the 4-arg backward returns no `v_th` gradient, so it must not be adopted by ALIF trainers) and what the cosine cannot see (anything upstream of the forward/gradient fork).

**eprop_nmnist.py** (797) / **eprop_shd.py** (830) — vision (N-MNIST → 10) and audio (SHD 700ch → 20) twins: recurrent-LIF + leaky readout, hand-built `eprop_grads` under `@torch.no_grad()`, `EpropEncoder`/`EpropAudioEncoder` are the brain's transducers (loaded at `brain.py:647/675` from `models/eprop_*.pt`). Bare argv runs the smoke, never a dataset train. Neither calls `.backward()`.

**executive_control.py** (458) — three fixed-dynamics faculties: `conflict(drives)` (Botvinick Hopfield co-activation energy, normalized, permutation-invariant, exactly 0 on one-hot), `evc(conflict, value_gain, cost)` (withhold when net ≤ 0, saturating otherwise), `plan_sequence` (beam/exhaustive multi-step search that beats a 1-step greedy on a delayed-reward valley). Live at `autonomous_loop.py:1102/1108/1157`; consumed as decision uncertainty by `metacognitive_monitor.py:133` and by `learned_evc`.

## 3. REGION NOTES

**Four strata in one directory.** (i) *Single-neuron models* — adex/alif/autapse/dendritic/divisive_normalization: pure-stdlib scalar dynamics, each with a `beta=0`/`w=0`/ablation control that collapses it onto a simpler sibling. They reach a live path only through `growth/neuron_zoo.Population` and `heterogeneous_microcircuit`. (ii) *Cognitive faculties* — affect_core, crossmodal_binder, counterfactual_sim, executive_control, agency_monitor, autotelic_goals, access_consciousness: composed lazily by `Brain.__init__` in "batch" try-blocks (W6/W8/C1) and driven from `_wake_tick`. (iii) *Training tier* — eprop_kernel + its two consumers, deep_stdp_latency: torch, checkpoint-loaded into the brain as frozen encoders, never trained on the live tick. (iv) *Reference/ingest* — cortical_prior (H01 measurement), corpus_stream (real text), cortical_microcircuit (Potjans-Diesmann validation dynamics).

**The spine is `brain.py`.** Everything else in the slice is either something it composes or something that composes it. The single live driver is `engine/learn_daemon.py:249` → `make_brain()`; `cockpit.py`, `brain_telemetry.py`, `nengo_bridge.py` and the gate tools are the other construction sites. Data flows: percept → encoder latent → `_consume_perception` → `_update_access` (GNW ignition) → `_route_percept_to_action` (writes `arm._ext_drive` for the next tick) → basal-ganglia gate → `So101Arm.apply()`. Learned state goes to the one lake via `substrate/home.py` (`_T_GROWTH`, `_T_STATE`, episodes); `access_index`, `conflict`, `agency_signal`, `metacog reliability` go out as telemetry to the cockpit.

**The dominant style is a defensive one.** `brain.py` contains **122** `except Exception` handlers. That is what makes the composition survive a missing optional faculty, and it is also the single largest silent-degradation surface in the tree: `brain.py:2724` wraps the whole access drive so any bug there zeroes `_access_state` without a word.

## 4. DEFECTS

**Corrections to claims currently in the tree** (I verified these directly, they contradict what is written):

- `engine/brain.py::Brain.close()` **does** flush growth — `self._flush_growth()` is the first statement after the idempotency guard, and `_flush_growth` documents itself as "ONE mechanism, TWO callers -- the sleep tick and `close()`". Both `CLAUDE.md` and my brain.py reader assert the opposite ("close() has no growth flush", specs/019 `[B2.09]`). **The defect is the claim, not the code** — `[B2.09]` should be re-scoped or closed.
- `faculties/access_consciousness.py` docstring states three times that the live path builds this faculty **with no `ws=`**, on a private single-submitter bus. `brain.py:2626` is `AccessConsciousness(seed=self._seed, ws=self.workspace)`. Every figure gated on "un-repointed" (the 56/56 byte-identical A/B, the measured null) describes a configuration that no longer ships. `brain.py:2668` repeats the stale claim.
- `CLAUDE.md` cites `Brain.self_improve` at `brain.py:3767`; it is at **`:4162`**. Zero call sites tree-wide (confirmed by exhaustive grep) — real P0/P3 dormancy, and `gates/method_dormancy_gate.LEARN_VERBS` omits "improve", so no gate can see it.

**Dormant (P3), grep-confirmed zero live callers:**

- `faculties/access_consciousness.py::AccessConsciousness.content_vector` — zero call sites anywhere, including its own self-check.
- `faculties/access_consciousness.py::focused` / `state()["focused"]` and `::n_bound` — no consumer outside this module, while `:1886` justifies the key by "the eight live consumers of this dict" and `brain.py:2526` says ten.
- `faculties/affect_core.py::AffectCore.affect` and `::own_emotion` — zero callers outside `main()`; `own_v`/`own_a` are written only by `empathize()`, whose one live site is inside `respond()`. P3 shape 5 (a constant wearing a signal's clothes). The module says so itself and is correct.
- `faculties/agency_monitor.py::agency` / `::controllability` / `::status` / `::snapshot` — write-only faculty; `observe()` runs every tick, nothing reads out.
- `reservoir/cortical_microcircuit.py::thalamic_drive` and `::CorticalMicrocircuit` — zero references outside the module. The port is real and measurable; nothing drives it (specs/019 `[B2.08]`).
- `engine/brain.py::Brain.perceive_pc` — called only inside brain.py's own `main()`. `::recall_episode_durable` — only caller is a test, so the P4b lake-read path never fires in production.
- `faculties/executive_control.py::ExecutiveControl.drive_entropy` and `faculties/crossmodal_binder.py::recover_complement` / `::superadditivity_index` — self-check only.
- `engine/brain.py` write-only state: `_plastic_relaxed`, `_lake_kept`, `_lake_pruned` (set, never read, absent from `state()`).
- `sensorimotor/eprop_shd.py::build_parser` defines `--train`; `::main` **never reads `args.train`**. Its help text claims it is "Required for a train run … so that bare argv can never start one", but `main` branches on `args.smoke or len(sys.argv)==1`, so `python -m …eprop_shd --hidden 64` starts a real dataset train. `eprop_nmnist.main` does it correctly.

**Duplicated mechanisms (P2):**

- `plasticity/eprop_kernel.py::_TinyNet.eprop_grads` implements a **different** factorization from both consumers: no kappa-filtered `ebar_in`/`ebar_rec`, and `g_out += err.t() @ z` on raw `z`. Both `eprop_shd.eprop_grads:295-303` and `eprop_nmnist.eprop_grads:306-314` use `ebar_*` and `zbar`. So the kernel's gold-standard cosine assert validates a rule **neither shipped encoder runs** — the one criterion the e-prop invariant leans on is measuring the wrong object.
- `sensorimotor/eprop_shd.py::eprop_grads` vs `sensorimotor/eprop_nmnist.py::eprop_grads` — still line-for-line identical; the specs/013 C4 merge lifted nine shared definitions but left the learning rule itself duplicated, along with `train_and_eval`, `_make_synthetic`, `run_smoke`, `build_parser`, `main`, and the encoder `load/_prep/encode/classify`. The duplication has already leaked twice: the on-device accumulation fix (shd only) and the global-seeding removal (shd only) each landed in one twin.
- `engine/brain.py` constructs `ReentrantWorkspace` at `:790` **beside** the `GlobalWorkspace`/`AccessConsciousness` pair at `:2626` — two live workspaces reporting ignition and availability in one brain; `faculties/reentrant_workspace.py:10` names access_consciousness as the incumbent it duplicates, and `faculties/second_order_access.py` is a third access surface.
- `engine/brain.py` holds **four** theory-of-mind instances (`_tom`, a throwaway in the braincog block, `_tom2`, `_recursive_tom`) with `_tom_update` fanning one event into three; and three surprise/novelty estimators feeding one drive vector.
- `engine/brain.make_brain` vs `sensorimotor/trained_brain.make_brain` — same name, different signature, both live.

**Stale docstrings/comments contradicting the code:**

- `engine/brain.py` module docstring: "the gated **ArmSim** is the only path to motion". **No `class ArmSim` exists in `src/`** (verified) — the gate is `brainstem/so101_arm.So101Arm.apply()`. This is the same defect `CLAUDE.md` already records for the safety-envelope file list, still live in the module that names the safety contract. Same docstring's run line invokes the retired `.venv-cu128`, and its "torch is OPTIONAL" paragraph contradicts P7's unconditional-core-dependency status.
- `engine/brain.py::perceive_event` / `_perceive_real` still say "surrogate-gradient deep SNN (training tier)"; `__init__` states that encoder was retired 2026-08-13 and only `EpropEncoder` loads.
- `sensorimotor/eprop_shd.py` cites `training/eprop_nmnist.py` four times (promoted to `cortex/` 2026-08-12) — in a file whose docstring opens with a nine-line argument against stale paths. Both eprop consumers carry post-merge section headers announcing code that is no longer under them, and neither module docstring mentions `eprop_kernel` at all.
- `faculties/access_consciousness.py:625-627` cites `:507`/`:508` for lines that are 614/615 and still reasons about `_LABEL` after `content_key` replaced it with a rotating tuple.
- `engine/brain.py::state()` ACh/cortisol comments cite `:996`/`:1767`/`:1754`; all rotted.

**Hardcodes where a learned/adaptive value belongs (P1):**

- `faculties/executive_control.py::evc` — `cost` and `value_gain` are developer constants, and `faculties/learned_evc.py` exists precisely to learn them via dopamine-gated three-factor plasticity. `autonomous_loop.py:1108` still falls back to the literal `cost=0.1, value_gain=stakes+0.3`. The superseding mechanism is built; P2 says finish the adoption and delete the fallback.
- `faculties/competence_executive.py::DopamineModulator.LR_MIN/LR_MAX` (0.5/2.0) — the module's own P1 adjudication is honest and thorough (three of four live calls land exactly on a bound), and correctly declines conversion pending the open three-RPE-mechanism P2 finding. Track it, don't re-litigate it.
- `faculties/access_consciousness.py` — `ignition_threshold=0.55` repeated as a literal in three constructors; `_salience`'s `0.75 + 0.25*completeness` blend (two chosen weights); harness `_own_stale_head(lo=0.55…)` pins its sweep floor to the default rheobase while the docstring warns the red must be re-swept if the floor moves, and nothing reads the floor off the bus.
- `engine/brain.py` — `_PLASTIC_REST_STEPS=40`, `STORE_EVERY=3`, `SELFBUILD_EVERY=8`, `EPISODES_PER_DAY_AT_BOOT=1.2` (docstring concedes it does not track production live), learning rates `0.25`/`0.02`, EMA pairs 0.98/0.02, 0.95/0.05, 0.9/0.1, `MetacognitiveMonitor(base=0.4)`, `GrowingLearner(vigilance=0.86)`, and the whole `_language` decode block.
- `sensorimotor/eprop_shd.py` / `eprop_nmnist.py` — smoke `lr=5e-3` + bare `range(80)`, CLI `--lr 2e-3`, `--batch-size 64`, `--eval-limit 2000`; none covered by either file's P1 block. `eprop_nmnist::_chance_bound` derives its accuracy bar from `n_classes × n_test`; `eprop_shd` deleted its bar and never built the derived one.

**Vacuous / weakened criteria (P8):**

- `engine/brain.py::main` claim A computes `cA_pass = (sum_birth != sum_grown) and (ov &lt; 0.6) and (FIXED == FIXED)`. **`FIXED == FIXED` is a constant `True`** — a conjunct that cannot fail, sitting in a criterion whose whole point is that the old fixed template is dead. Delete it or make it compare a generated summary against `FIXED`.
- `sensorimotor/eprop_shd.py::run_smoke` asserts only shapes + `isinstance` for the checkpoint round-trip — exactly the criterion `eprop_nmnist::run_smoke` records as measured-unfalsifiable and replaced with `cfg_exact`/`latent_exact`/`logit_exact`. shd compounds it by building its smoke net with *default* dynamics where nmnist deliberately uses non-default to keep the probe non-vacuous. shd effectively has no round-trip criterion.

**Other:**

- `faculties/access_consciousness.py:295-298` subscribes four taps with `inbox.append` and never trims them; `step()` reads only `tap[-1]`. On the always-on daemon each tap grows one tuple per broadcast for the life of the process — the same unbounded-growth class the `_members` `deque(maxlen=…)` was added to fix.
- `faculties/access_consciousness.py::n_bound` reports `_n_assemblies`, a monotonic count of every `bind_assembly` call ever, while its docstring says "assemblies fused into the currently ignited content". On the live brain it equals total wake ticks, and it is surfaced as telemetry at `brain.py:3422`.
- `engine/brain.py::_attend_real_stream(latent, gain)` never uses `gain`; the caller `_safe_ctx` computes `g = 0.5 * self._real_novelty` and discards it — a magic constant feeding a dead parameter.
- `faculties/access_consciousness.py::_fused_calls` is a process-wide class counter incremented from a `@staticmethod` — a non-atomic `+=` under the free-threaded interpreter this repo targets.
- **e-prop invariant is clean across the slice.** No `.backward()` on any hidden-weight path; the only autograd is `eprop_kernel.bptt_grads` (the independent reference) and `eprop_kernel._fwd_bwd` (the surrogate-off negative control), both measurement-only and pinned by `tests/test_eprop_train_tier_unreachable.py`.</result>
<usage><subagent_tokens>323993</subagent_tokens><tool_uses>56</tool_uses><duration_ms>663590</duration_ms></usage>
</task-notification>

---


## Region digest 7 of 25

<task-notification>
<task-id>a52bfa6d8e4e41c4d</task-id>
<tool-use-id>toolu_01Ae1wfL2amwA97ZtvJqSZkP</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a52bfa6d8e4e41c4d.output</output-file>
<status>completed</status>
<summary>Agent "Read reasoning A in full" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 38 files read in full. Digest:

## 1. COUNT — 38 .py files

`src/snn_data_lake/reasoning/`: `analogy_bench.py`, `ar_spiking_lm_serve.py`, `arc_agi/brain_player.py`, `arc_agi/env.py`, `arc_bench.py`, `arc_diagnostic.py`, `arc_mint.py`, `arc_residual.py`, `arc_scene.py`, `arc_solver.py`, `autonomous_brain.py`, `autonomous_brain_recovered.py`, `autonomous_enrich.py`, `bayesian_snn.py`, `blimp_bench.py`, `burst_multiplexing.py`, `cellular_neural_network.py`, `concept_reflection.py`, `consciousness_probe.py`, `deep_stdp_2stage.py`, `delay_learning.py`, `dense_associative_memory.py`, `derivation.py`, `developmental_core.py`, `developmental_loop.py`, `developmental_loop_lake99.py`, `developmental_loop_slda.py`, `developmental_loop_wm.py`, `dishbrain_learning.py`, `eprop.py`, `eprop_lm_alif_serve.py`, `eprop_lm_serve.py`, `event_quadtree_compression.py`, `fast_features.py`, `fly_hash.py`, `free_energy_control.py`, `ng_snn/data.py`, `ng_snn/extend_3conv.py`

## 2. PER MODULE

**analogy_bench.py** — 12 hand-written Gentner structure-mapping items + 3 negative controls scoring `faculties.vsa_structure_mapping.StructureMapper`. PASS requires gold map AND systematicity&gt;0. Gotcha: a **fresh** `StructureMapper` per item is mandatory (its rng advances inside `map()`); 3 kept failures are deliberate, don't "fix" them. Only slice module importing `cortex/` for scoring; not in `run_selfchecks`.

**ar_spiking_lm_serve.py** — `SpikingLMServe`: forward-only numpy replay of a frozen Spiking-RWKV char LM (`models/ar_spiking_lm.npz`). `available()/logprob()/generate()/provenance()/provenance_status()`. Consumed by `blimp_bench.SERVES`. Gotcha: honest-unavailable contract (`logprob → None`, never a number); `_try_train_inline` restores `sys.path` in `finally` — do not revert.

**arc_agi/env.py** — 4 hand-authored ARC-AGI-3-style interactive grid games (`KeyDoor`/`PushBox`/`ColorLock`/`ToggleToGoal`), uniform `reset/step/snapshot/restore`, exported as `GAMES`. Invariant: `step()` never terminates on timeout — the step cap lives in the runner so Go-Explore restore is unconstrained.

**arc_agi/brain_player.py** — `BrainPlayer`: curiosity Go-Explore over `env.GAMES` driving real `faculties.evidence_accumulation.RaceAccumulator` + `faculties.intrinsic_motivation.CuriosityDrive`, plus tabular value iteration. Gotcha: `import env as arc_env` is a **bare top-level import** — only resolves if `arc_agi/` is cwd; unimportable as a package module.

**arc_bench.py** — ARC-AGI harness: `load_tasks/score/score_task/_grids_equal` + `identity_solver`/`constant_solver` controls and an in-`main` oracle. Called by `arc_solver.main`, `arc_diagnostic`, `arc_mint`. Invariant: exact match, pass@2, controls must score ~0 or the harness is declared broken.

**arc_diagnostic.py** — Classifies each `arc_solver` failure as budget- vs coverage-limited via a **provable node bound** `P+P²+…+P^D` derived from `arc_solver._pool` (read-only reflection). `probe_pair/diagnose_task/diagnose/node_bound`. Gotcha: reflects into `arc_solver._BASE`/`_pool`, so reordering `_BASE` breaks its bound; deliberately excluded from `run_selfchecks` (~27s), covered by `tests/test_arc_diagnostic.py`.

**arc_mint.py** — The Gf Phase-3 "mint": induces out-of-DSL primitives from `arc_scene` measurements. `mint_records(train) -&gt; TypedPrimitiveRecord`, `compile_record`, `induce_rule`, `_build_selected_recolor` (Design B). Only consumer surface is `arc_solver._pool` as `mint:*`. Invariant: declarative fields alone must recompile `fn` (auditable bottleneck); memo-class rules (`rank`/`lookup`) return `None` on unseen attrs, never a guess.

**arc_residual.py** — Versioned 32-predicate descriptor alphabet (`DESCRIPTORS`, `DESCRIPTOR_VERSION=1`) + greedy `induce_selector` conjunction learner + per-grid `_DESC_CACHE`. Consumed by `arc_mint`. Invariant: bumping `DESCRIPTOR_VERSION` invalidates every stored record; `clear_cache()` must be called per task by `arc_solver.synthesize` (it is, via `_pool`).

**arc_scene.py** — 13 frozen measurement generators over `arc_solver.components()`: `palette/morph/object_holes/shape_class/relations/scene/symmetry_axes/shape_match/segment/ray/perimeter/voronoi/period/separators/progression/color_pairing/border_pairs/crossings`. Measurement-only, never mutates `components()` output (copies the dict). Feeds `arc_mint` + `arc_residual`.

**arc_solver.py** — The ONE end-to-end ARC path: `components()` perception, ~62-entry `_BASE` DSL, `_fitted_prims`, `_pool`, `synthesize`, `make_solver`. Also imported by `real_rpm_bench` and lazily by `cortex/reasoning_cortex.py:338` — the single governed-tier consumer of this slice. Invariant: `_BASE` is never mutated or reordered (determinism + `arc_diagnostic` depend on it); `ArcLibrary` and the fix-point combinator were measured null and **deleted**, and `main` asserts `ArcLibrary` did not come back.

**autonomous_brain.py** — Standalone torch/GPU enclosed loop (conv1 latency-STDP → prefrontal WM → `GrowableReadout` grow-and-freeze → `LakeMemory`). Lands ~47% by design; the docstring names the readout as the cause. No importers.

**autonomous_brain_recovered.py** — Thin wrapper over `developmental_core.develop_cycles` (~95% online / ~98% sleep-refit) + regression asserts. Its own docstring records that `develop_cycles`' `lake=`/`emit=`/`reflector=` injection points have **zero in-tree callers** since `brain_loop` retired.

**autonomous_enrich.py** — `AutonomousEnricher`: curiosity-picks a frontier topic, fetches via `web_ingest_gate.WebAccessGate`, writes one dedup'd durable fact to `substrate.home.KnowledgeLog`. `--once/--rounds/--forever`; offline `_BundledGate` self-check. Registered in `run_selfchecks`. Gotcha: `NORD_HOME` is save-and-**restore**, not set-and-pop — reverting breaks in-process gate runs.

**bayesian_snn.py** — Bernoulli weight-mask ensemble over a prototype readout; `bayesian_predict` returns (modal, disagreement) as calibrated uncertainty. Registered in `run_selfchecks`. Fully self-contained; nothing consumes its uncertainty.

**blimp_bench.py** — BLiMP minimal-pair harness: `paradigms/load_pairs/score` + `random/constant/length` controls, `per_char` length-normalized rescoring, `cached_logprob`, `UNSCORABLE` sentinel, `SERVES`/`frozen_serves()` wiring all three LM serves. Invariant: `UNSCORABLE` is deliberately NOT a parameter — a caller-supplied sentinel resurrects the length bias inside `per_char`.

**burst_multiplexing.py** — Naud-Sprekeler two-channel spike code: `encode/decode/simulate` with event-rate and burst-fraction channels; the defining assert is low crosstalk, not just recovery. Decoder rescales only by the code's own known range (`E_LO..B_HI`), never a fit to the sources.

**cellular_neural_network.py** — Chua-Yang CNN with feedback restricted to center-only so the settled output is the closed form `sign((B*u)+z)`. Edge-detect + inversion templates with **hand-derived** expected bitmaps. Zero imports, zero RNG.

**concept_reflection.py** — `ConceptReflector.reflect()` runs a `train_of_thought.TrainOfThought` chain over the brain's own concept prototypes and Hebbian-wires each consecutive pair (`association_graph`, `strongest_link`). Called by `developmental_core`. Invariant: the ignition threshold is read **live** off `tot.ws.ignition_threshold` — no literal fallback (`getattr(..., 0.55)` was deleted as a second copy of the value).

**consciousness_probe.py** — P8's worked example. `probe_integration` (PCI, 20-seed distribution, the one surviving marker) and `probe_access` (verdict permanently `None` — a *reading*, imported by name from `spike_chat_lake`). `deleted_access_marker_arithmetic` re-derives the deletion evidence live every run. Invariant: the asserts test the *instrument* (`0&lt;frac&lt;1`; `lo != hi`), never the answer; do not assert `verdict is None` — that would lock out the successor.

**deep_stdp_2stage.py** — `ConvLatencyStdp2` + `ThreeLayerNet`: stacks a second latency conv-STDP stage on the frozen stage-1 first-spike volley (MIN-pool between stages because earlier = stronger). Asserts baseline &gt; chance and both stages' kernel cosine &lt; 0.6; deliberately asserts **no** depth delta. Not in `run_selfchecks`.

**delay_learning.py** — Three-factor learning of a synaptic DELAY (not weight) for temporal-order classification; `train_delay/coincidence/accuracy`. Registered in `run_selfchecks`.

**dense_associative_memory.py** — Degree-n Hopfield sweep (n=2 classical vs n=4 dense) plus modern softmax vs log-sum-ReLU energies at P=2N; uses `stats_bounds` for ±3σ band separation. Gotcha: the docstring still contains `—` and `×`, which the file's own comment measures as cp437-fatal.

**derivation.py** — Stateless multi-hop transitive deduction over a `records` list: `relation_graph/derive/reason_question`, BFS ≤6 hops, `_TRANSITIVE` whitelist. Ported out of the retired `brain_loop`; consumed by `spike_chat_lake` and referenced by `schema_induction.py:565`. Invariant: absence of a proof returns `None`, never "no"; bare `is` is excluded (predication ≠ class membership).

**developmental_core.py** — `develop_cycles(cfg, lake=, reflector=, emit=)` — the ONE extracted loop body (perceive → WM → SLDA-RFF → grow RFF + sleep-replay → consolidate → reflect → RBF refit). `cfg_from_env`. Sole caller: `autonomous_brain_recovered`. Gotcha: `_fresh_slda` rebuilds from all accumulated batches — order-invariance is what makes growth non-forgetting.

**developmental_loop.py** — `GrowableSTDP` (ART-vigilance neurogenesis + optional criticality-ranked pruning) + `assign_and_eval` (disjoint assign/score halves — a real bug fixed in v3.72). Records soft-kWTA and pruning as measured **negatives**, kept off by default. Not in `run_selfchecks`.

**developmental_loop_lake99.py** — Richer JL-projected proximity key + `_rbf_refit_eval` (the sleep-refit RBF-SVM, imported by `developmental_core`) + `LINEAR_LAKE_REF = 0.902`, explicitly the last surviving citation of a retired module's number.

**developmental_loop_slda.py** — `SLDA_RFF` (order-invariant streaming LDA on Random Fourier Features) + `_acc`; both imported by `developmental_core`. Gotcha: RFF weights are generated on CPU with a CPU generator then moved — `torch.randn(generator=cpu)` under `set_default_device(cuda)` errors.

**developmental_loop_wm.py** — Prefrontal persistent-activity WM in the control path; `_wm_persistence_probe` measures hold ticks + distractor robustness. Asserts accuracy holds (faculty, not accuracy lever). No importers.

**dishbrain_learning.py** — Kagan et al. embodied Pong learned purely from feedback *predictability*; `DishBrainAgent.feedback` + the contingent-vs-random-feedback control that must stay flat. Registered in `run_selfchecks`. Docstring explicitly distinguishes it from `free_energy_control`, `closed_loop`, `r_stdp`.

**eprop.py** — The reference e-prop implementation: `Net.grad_eprop` vs an **independent** `Net.grad_bptt` sharing no code, cosine asserted &gt; 0.5 over hidden weights only (`_flat_hidden` excludes `Wout` because it is exact in both). This is the file the cross-cutting e-prop invariant is anchored on; registered in `run_selfchecks`.

**eprop_lm_alif_serve.py** — `EpropAlifLMServe`: numpy forward for a frozen deep-ALIF (two-timescale, `RHO`/`BETA` adaptation) e-prop char LM. Enforces zero `W_rec` diagonal on load. In `SERVES`, registered in `run_selfchecks`.

**eprop_lm_serve.py** — `EpropLMServe`: same contract, flat single-layer recurrent LIF. Registered. Gotcha: `_try_train_inline` inserts `&lt;repo&gt;/training` into `sys.path` and **never removes it** (unlike `ar_spiking_lm_serve`, which was fixed).

**event_quadtree_compression.py** — 2-D lossless quadtree codec over the channel×time event grid; benchmarked against gzip and `spike_compression`'s 1-D delta+varint. Registered in `run_selfchecks`.

**fast_features.py** — `batched_features` and `batched_features_and_keys`: batched-conv2d replacements for the per-image `conv.features` loop, gated on &lt;1e-4 equivalence to the per-image reference AND on being faster. Consumed by `autonomous_brain`, `developmental_core`, `developmental_loop_slda`. Holds the surviving definition of `_global_key` after `developmental_loop_lake` retired.

**fly_hash.py** — `FlyHash`/`FlyHashIndex`: Dasgupta random-expansion + WTA sparse tag LSH; recall@1 and near-vs-far tag separation asserted. Registered in `run_selfchecks`. Gotcha: imports the **legacy flat** paths `snn_data_lake.signal_loop` and `snn_data_lake.spike_preprocessing`, resolving only through the `__init__.py` back-compat alias layer — the only slice module doing so.

**free_energy_control.py** — Predictive-coding gate: a control neuron spikes only if the action lowers `(x-target)²` beyond a dead-band; compared against always-act bang-bang under noise. Registered in `run_selfchecks`.

**ng_snn/data.py** — Tonic/`DiskCachedDataset` N-MNIST streaming loader (`make_nmnist_loaders`, picklable `ToFloatTensor` for Windows workers). `DEFAULT_T = 15` is a self-labelled **PLACEHOLDER** ("SET FROM THE PAPER"). No in-tree callers.

**ng_snn/extend_3conv.py** — One-shot script: loads a 2-block SoftHebb checkpoint, greedily trains block 2, dumps 13824-d features + ridge probe. Gotcha: bare `from softhebb_mnist import SoftHebbCNN`, relative `DATA="../../data"`, and a hardcoded `softhebb_mnist.pt` cwd path — runnable only from inside `ng_snn/`.

## 3. REGION NOTES

Four disjoint clusters, only one of which touches the governed tree.

**(a) The ARC/Gf stack** is the most mature and the only one with a real internal dependency graph: `arc_solver` (perception `components()` + DSL + iterative-deepening search) is the hub; `arc_scene` measures on top of it; `arc_residual` turns those measurements into a versioned descriptor alphabet + selector; `arc_mint` induces typed primitives from both; and `arc_mint.mint_records → arc_solver._pool` is the single typed bottleneck back in. `arc_bench` scores, `arc_diagnostic` reflects into `_pool`/`_BASE` to classify failures, `analogy_bench` scores a `cortex/` module. Data flow is entirely tuple-of-tuples grids + `TypedPrimitiveRecord`. Its one governed consumer is a lazy `from ...arc_solver import synthesize` at `cortex/reasoning_cortex.py:338`.

**(b) The developmental/MNIST loop family** — `developmental_core.develop_cycles` is the extracted single source of truth, wrapped by `autonomous_brain_recovered`; `developmental_loop_slda` (SLDA_RFF), `developmental_loop_lake99` (`_rbf_refit_eval`), `concept_reflection`, `fast_features` are its parts. `autonomous_brain`, `developmental_loop`, `developmental_loop_wm`, `deep_stdp_2stage` are earlier, superseded, still-standing benchmarks with no importers. Everything here is torch-tier and touches `instruments.deep_stdp_latency`, `instruments.snn_mnist_stdp`, `substrate.proximity_replay`, `faculties.prefrontal_executive`, `neuromodulation.*`.

**(c) The language/e-prop serve tier** — `eprop.py` is the invariant's reference implementation; the three `*_serve.py` modules are numpy forward-only replays of frozen artifacts, all wired into `blimp_bench.SERVES`, all honest-unavailable when `models/*.npz` is absent.

**(d) Standalone one-file mechanism demos** — `bayesian_snn`, `burst_multiplexing`, `cellular_neural_network`, `delay_learning`, `dense_associative_memory`, `dishbrain_learning`, `event_quadtree_compression`, `fly_hash`, `free_energy_control`. Each is self-contained, registered in `run_selfchecks`, and produces a number nothing in the brain consumes.

**Import health:** all 38 import cleanly under the package layout except `arc_agi/brain_player.py` and `ng_snn/extend_3conv.py` (bare sibling imports) and `ng_snn/data.py` (requires `tonic`) — consistent with the ~30% no-import figure.

## 4. DEFECTS

**Dormant / unreachable**
- `arc_agi/brain_player.py:54` — `import env as arc_env`: a bare top-level import of a sibling. Combined with `:47-49` (`for _p in (...): if _p not in sys.path: pass` — a loop whose body is `pass`, left behind by the migration), this module cannot be imported as `snn_data_lake.reasoning.arc_agi.brain_player` at all. Same shape at `ng_snn/extend_3conv.py:17` (`from softhebb_mnist import SoftHebbCNN`) plus `DATA="../../data"` and a cwd-relative `softhebb_mnist.pt`.
- `developmental_core.py:89` — `develop_cycles(..., lake=, reflector=, emit=)`: three injection parameters with zero in-tree callers since `brain_loop` retired. The module's own docstring says so; only the `lake=None` standalone branch is ever exercised. Honest, but still P3 debt.
- `autonomous_brain.py`, `developmental_loop.py`, `developmental_loop_wm.py`, `deep_stdp_2stage.py`, `ng_snn/data.py`, `ng_snn/extend_3conv.py`, `arc_agi/brain_player.py` — no importers, no `run_selfchecks` registration, no test. These read as **dead/superseded**, not promotable: `autonomous_brain` is explicitly superseded by `autonomous_brain_recovered`, and `developmental_loop_wm`'s WM faculty is absorbed by `developmental_core`.
- `analogy_bench.py`, `consciousness_probe.py` — absent from `run_selfchecks.py` (verified in the same command that found 22 other slice entries, so this is a controlled negative). `consciousness_probe.probe_access` is live via `spike_chat_lake`, but the module's own gate never runs in the smoke suite.
- `arc_scene.py` — of its ~20 public generators, `arc_mint` consumes only 3 (`object_holes`, `period`, `shape_class`) and `arc_residual` most of the rest; `voronoi`, `progression`, `color_pairing`, `locus_meet`, `ray` have no consumer outside the module's own self-check. The docstring's "the Phase-3 mint will CONSUME these" is a promise, not a state.

**Duplication (P2)**
- Grid validation: `arc_mint._ok` vs `arc_solver._ok` — two near-identical "legal ARC grid" predicates.
- Grid equality: `arc_bench._grids_equal` vs `arc_diagnostic._eq` — byte-for-byte the same function under two names, in files that already import each other.
- Env readers: `envi`/`envf` live in `_loop_common` and are used by `developmental_core`, `developmental_loop_wm`, `developmental_loop_lake99`, `autonomous_brain` — but `developmental_loop._envi`, `developmental_loop_slda._envi/_envf`, `deep_stdp_2stage._env/_envf` and `fast_features._env` are four private re-implementations of the same two functions.
- `_evidence(option, n=10, strong=1.0, weak=0.05)` is defined identically three times: `autonomous_brain.py:58`, `developmental_core.py:72`, `developmental_loop_wm.py:42`.
- `_filter_objs` and `_tile` exist in both `arc_mint` and `arc_solver` with the same names and different semantics — a name collision waiting to mislead an editor.
- Enclosed-region flood fill is written three times: `arc_mint.enclosed_regions`, `arc_scene.object_holes`, `arc_solver._fill_enclosed`.

**Stale prose contradicting code**
- `developmental_core.py:23` — "imported LAZILY by brain_loop.py (never at its top level)". `brain_loop.py` does not exist; the file's own lines 9-16 say so. Two adjacent statements, one of them false.
- `ng_snn/data.py:2` and `:18`, `ng_snn/extend_3conv.py:6` — docstrings give `research/ng_snn/...` paths and `research/ng_snn/data.py` run commands. The files live in `src/snn_data_lake/reasoning/ng_snn/`; the quoted commands do not run.
- `fly_hash.py:18-19` — imports `snn_data_lake.signal_loop` / `snn_data_lake.spike_preprocessing`, the pre-3.0 flat paths, surviving only on the back-compat alias layer.
- `consciousness_probe.py:121-136` — the ESCALATED block declares a `spike_chat_lake` assert "RED between the two halves of this change". If that half landed, this is now a stale description of a resolved state; if it did not, the gate is red. Either way the paragraph cannot be both current and correct.
- `dense_associative_memory.py` docstring still contains `—` and `×`, which the file's own measured comment (`:164-171`) identifies as cp437-fatal for direct invocation.

**Hardcoded where adaptive belongs (P1)**
- `arc_diagnostic.py:79` — `WORTH_FRACTION = 0.15` decides a printed "WORTH BUILDING / NOT worth building" verdict from an unjustified literal; `_P_FALLBACK = 64` (`:75`) is a magic pool-size overestimate.
- `arc_mint.py:678` — the `3 +` in the Design-B MDL cost is a hand-assigned skeleton constant. Self-declared as a surviving P1 offender in-file, which is honest but still open.
- `arc_residual.py:108` — `SELECTOR_LITERAL_CAP = 4`, carefully framed as a resource guard rather than an arbiter. The framing is sound; the number is still a literal.
- `concept_reflection.py:49` — `threshold=0.55` default. The module went to real trouble to stop the *rheobase* being a literal (reads it live off the workspace) but left the similarity gate — the value that actually decides which concepts ignite — as the same hardcoded 0.55.
- `ng_snn/data.py:30` — `DEFAULT_T = 15`, labelled `PLACEHOLDER … SET FROM THE PAPER` and never set.
- `deep_stdp_2stage.py:409-417` — collapse bar `div &lt; 0.6` on both stages; `developmental_loop.py:297` — `NORD_VIG` default `0.85` gates all neurogenesis; `burst_multiplexing.py:34-39` — `E_LO/E_HI/B_LO/B_HI/GROUP_GAP` are the encoder's declared code range (defensible) but the assert thresholds `0.7`/`0.3`/`+0.4` are not.

**Promotability read**
Promotable now: `derivation`, `eprop`, the three `*_serve` modules, `arc_bench`/`blimp_bench`, `fast_features`, `arc_solver`+`arc_scene`+`arc_residual`+`arc_mint` as one unit (they only make sense together). Merge-then-retire: the developmental-loop family into `developmental_core`. Retire: `autonomous_brain`, `developmental_loop_wm`, `arc_agi/brain_player`, `ng_snn/extend_3conv`. Duplicate-of-governed: none outright, but `fly_hash`'s top-k overlaps `faculties/action_gating` (which cites it by name) and `sensorimotor/lca_sparse_coding` lists it as on-thesis kin.</result>
<usage><subagent_tokens>377676</subagent_tokens><tool_uses>45</tool_uses><duration_ms>391890</duration_ms></usage>
</task-notification>

---


## Region digest 8 of 25

<task-notification>
<task-id>a011d64b73bb624fd</task-id>
<tool-use-id>toolu_01PLnUjb9FoPvSb98d1tHcZR</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a011d64b73bb624fd.output</output-file>
<status>completed</status>
<summary>Agent "Read reasoning B in full" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 20 files read completely (8,486 lines). Digest below.

---

## 1. COUNT — 20 files

`src/snn_data_lake/reasoning/gf.py`, `godel_machine.py`, `hjepa_arm_torch.py`, `hjepa_torch.py`, `hopfield_baseline.py`, `impact_analysis.py`, `index_compression.py`, `jepa_scaffold_torch.py`, `lake_reason.py`, `loop_latency.py`, `make_results_plot.py`, `mcr_hdc.py`, `model_builder.py`, `nef_integrator.py`, `nengo_gui_brain.py`, `neural_heterogeneity.py`, `nmnist_ingest.py`, `ng_snn/nmnist_pipeline.py`, `ng_snn/nmnist_spatiotemporal.py`, `ng_snn/nmnist_spatiotemporal_3conv.py`

## 2. PER MODULE

**gf.py** (1,909 L) — Five *combined-not-merged* Gf rule inducers in one file, private helpers tagged `_pi/_ogf/_gfi/_al/_sl`. Public: `induce_program`/`induce_program_full`/`consistent_programs`; `OpenGf` (iterative deepening + DreamCoder library learning + `solve_unbounded`); `synthesize_sequence`/`solve_series`/`number_series`/`letter_series`/`letter_analogy`/`analogy_over`/`raven_dist3`/`raven_arith_grid`/`raven_cell_attrs`/`word_number_relation`; `ActiveLearner`/`World`/`Library`/`consistent_hypotheses`/`function_from_pairs`/`string_from_pairs`; `MLP` (numpy Gc baseline). Wired: `reasoning_cortex`, `rpm_solver`, `spike_chat_lake` (live chat Gf branch), `vsa_synthesize`, `gates/eprop_transport_gate` (EXEMPT entry), `tools/arc_baseline`, run_selfchecks. **Gotchas**: `letter_series` must return `None` (not raise) on off-alphabet chars — `spike_chat_lake._gf_request` selects tokens by `str.isalpha()` and calls it *unguarded*, so removing that guard is an uncaught KeyError on the live reply path. `MLP.train` performs weight transport (`g @ self.w2.T`) — adjudicated a CONTROL, exempted in `eprop_transport_gate.EXEMPT`; the docstring explicitly forbids "fixing" it to e-prop because `main_sl` asserts `mdl_ok &lt; ind_ok`. Both `OpenGf._STATE` and `Library._STATE` are `.json` *record names inside* `brain_lake.bin`, not files (P4b closure); both self-checks spawn a real subprocess to prove it.

**godel_machine.py** (359 L) — Schmidhuber Gödel machine over real git worktrees. `GodelMachine(repo, verify, utility)` + `.attempt(name, rewrite)`; `attempt_on_repo(...)` is the real-repo adapter with `_SAFETY_KERNEL` (7 immutable files). Imported live by `infra/self_mod_harness`, `gates/import_manifest`, 2 tests, run_selfchecks. **Gotcha**: `attempt_on_repo` has **zero call sites tree-wide** — the module's own docstring says so (P3 shape 1: imported, never invoked). The `verify()` protected-path check parses `git status --porcelain -z --no-renames` *before* trusting the worktree's gate; the `-z`/`--no-renames` flags are load-bearing (a `git mv` of a protected file would otherwise parse as `R "old" -&gt; new` and be missed).

**hjepa_torch.py** (733 L) — 2-level nonlinear H-JEPA trainer on a synthetic XOR two-timescale world. `build_hjepa`, `vicreg_terms`, `TwoTimescaleWorld`, `train_step`, `evaluate_hierarchy_vs_flat`, `export_hjepa` (`snn_hjepa/1`). torch imported lazily; `nn.Module` defined inside `build_hjepa`. **Zero referrers, not in run_selfchecks** — nothing gates its asserts. **Gotcha**: the docstring discloses that claim (a) is satisfied by a *random* 1-step-trained encoder, so (a) measures only that the readout path is nonlinear, not what SSL learned; claim (b) is the discriminating one. Read (a) as the weaker.

**hjepa_arm_torch.py** (772 L) — Arm-native twin of the above (`ArmTwoTimescaleWorld`, obs `[0,1]^64` mirroring `encode_multimodal` bands, actions ENGAGE/RELEASE/HOLD, `build_hjepa_arm`, `export_hjepa_arm`, `verify_transfer_hjepa` bridging into `faculties/hjepa_serve.HJepaServe`). Zero referrers, not in run_selfchecks. **Gotcha**: **its PRIMARY claim is a measured NULL** — `mse_hier 0.0985` vs flat `0.0966`, and un-learned raw window averaging decodes perfectly (MSE 0.0000, acc 1.000). There is **no assert on the primary claim**; it prints a `*** NULL` banner and returns 0. The bridge check is the only thing asserted. Also: MSE and accuracy disagree in direction, and the file picks MSE — disclosed in-docstring.

**jepa_scaffold_torch.py** (524 L) — Deeper action-conditioned latent JEPA trainer exporting `snn_jepa/2` for `faculties/jepa_world_model`. `build_model`, `vicreg_terms`, `SyntheticStream`, `train_step`, `save/load_checkpoint`, `export_jepa` (asserts every shape), `verify_transfer` (float64 CPU clone vs core, tol 1e-4, measured 2.22e-16). Zero referrers, not in run_selfchecks. **Gotcha**: under `NORD_GPU` the default device is cuda, so every bridge probe tensor must be created with explicit `device="cpu"` — the comment records this as a bug that already bit.

**hopfield_baseline.py** (165 L) — Textbook dense bipolar Hopfield capacity yardstick (AGS 0.138·N bound) that `spiking_storage_prototype`'s claims should be read against. `make_patterns`/`train`/`recall`/`corrupt`/`overlap`/`mean_recall`. Only referrer: run_selfchecks. **Gotcha**: `random.seed(0)` was moved *into* `main()` (this file genuinely uses the global RNG, so the seed could not be deleted like the vestigial cases elsewhere).

**impact_analysis.py** (246 L) — Native pre-mutation blast radius: stdlib `ast` symbol references ∪ graphify `graph.json` dependency edges. `blast_radius(repo_root, target_rel, symbol, graph_path)` → `{radius, files, modules, via_graph, via_ast}`. Consumed by `infra/brain_native_selfbuild` (prefers surgical mutations), `gates/import_manifest`, run_selfchecks. Advisory only — never weakens the gate. Clean, well-scoped; the best promotion candidate in the slice.

**index_compression.py** (105 L) — Order-invariant d-gap + LEB128 varint codec for lookup metadata (id/edge/adjacency lists), reusing `spike_compression._leb128`. `adjacency`/`encode`/`decode`/`raw_bytes`. Only referrer: run_selfchecks. Lossless round-trip asserted; beats gzip.

**lake_reason.py** (333 L) — Contradiction-correction over stored `(subject, relation, object)` chat-lake triples. `Reasoner(facts).respond(text)` returns a correction or `None`. Consumed by `spike_chat_lake`. **Gotchas**: abstains on any negation (`not|never|n't`) because the shallow parser would "correct" a true negated claim; a "because" reason is attached only when the linking triple's relation is in `_CAUSAL_RELS` (co-mention alone is insufficient — that guard exists to stop fabricated causality); `_by_subj_rel` stores a **list** per key, fixing a last-wins bug where a second same-root triple hid the first.

**loop_latency.py** (100 L) — Sensor→command latency instrumentation over `signal_loop` + `spike_preprocessing` (encode → van Rossum match), p50/p95/p99. Only referrer: run_selfchecks. **Gotcha**: three asserts were *deleted* here for being unfallible (percentile monotonicity by construction; `enc+match == total` reassociated arithmetic; `enc_mean &gt; 0`, measured green under a neutered encoder). The surviving discriminating criterion is `len(set(best)) &gt; 1`, which reads the matcher's own output — previously thrown into `_`.

**make_results_plot.py** (308 L) — Hand-laid SVG of recorded per-version results. `bars`/`panel`/`build_svg`/`render`/`main`. Only referrer: run_selfchecks. **Gotchas**: it previously ran everything at module scope (writing `assets/results.svg` CWD-relative as an import side effect) and had *zero* asserts while registered as a self-check. Output path is now derived from `__file__`. The module documents that `assets/` is gitignored and **nothing reads the SVG** — a self-check that renders an artifact with zero consumers. Assert (2) was rewritten after its stated red was run and *passed* (it recomputed the formula instead of parsing `bars()` output).

**mcr_hdc.py** (119 L) — Modular Composite Representation: phase-mod-q HDC (`mcr_hv`/`mcr_bind`/`mcr_unbind`/`mcr_bundle`/`mcr_sim`) vs bipolar baseline, `record_recovery`. Consumed by `reasoning_cortex` + run_selfchecks. Claims ~4.6× fewer bits at comparable recovery.

**model_builder.py** (250 L) — Procedural "build me a 3D model of X": `box`/`cylinder`/`cone`/`translate`/`merge`, `PartCatalog`, `RECIPES`, `ModelBuilder.decompose|build|save_obj|knowledge`. Decomposition walks the assembly plan through `train_of_thought.TrainOfThought` with real ignition gating. Consumed by `spike_chat_lake` + run_selfchecks. **Gotcha**: the mesh is deliberately *not* stored in the lake — only the process facts (`knowledge()` asserts no geometry leaks).

**nef_integrator.py** (117 L) — Recurrent NEF line attractor over `instruments.nef_compute.NEFEnsemble`: `NEFIntegrator.step/run/reset`. Only referrer: run_selfchecks. **Gotcha**: `_say()` asserts ASCII at emission — this module shipped a U+2014 em dash for the life of the feature because cp1252 tolerated it; the invariant is ASCII, not "whatever the console accepts".

**nengo_gui_brain.py** (248 L) — Whole brain as a live spiking Nengo network (observation-only). `build_model(seed, day_period)`, PEP-562 `__getattr__` deferring `model`, `_module_scope_is_inert()` (AST). **Zero referrers anywhere** (loaded by the `nengo` CLI by path). **Gotchas**: previously constructed `Brain(...)` directly at module scope — a P4 violation *and* an import-time lake open. `_module_scope_is_inert`'s watch list was blind to its own stated red until `build_model` was added to it. The live check is **skipped in `.venv-lake`** (no nengo), so criterion (3)'s red is unreachable in the gate env; the `except` is narrowed to `ModuleNotFoundError` with `exc.name != "nengo"` re-raised.

**neural_heterogeneity.py** (236 L) — Perez-Nieves 2021 replication: heterogeneous-τ LIF population vs a homogeneous control at the *same N and same mean τ*, on a concentration×level task. `make_taus`/`prototype`/`dataset`/`count`/`features`/`standardize`/`train_readout`/`accuracy`/`selectivity_spread`/`run_experiment`. Only referrer: run_selfchecks. Imports `snn_data_lake.lif_neuron` and `snn_data_lake._mathutils` via the legacy flat alias path. Measured 100.0% vs 70.2%.

**nmnist_ingest.py** (246 L) — N-MNIST event ingestion medallion (BRONZE gzip event store → SILVER T-binned raster + ICR → GOLD rate vector → nearest-prototype). Falls back to deterministic synthetic events without `tonic`. Only referrer: run_selfchecks. **Gotcha**: the old Bronze assert (`len(raw)//calcsize == n_ev`) was true by construction while the success line claimed "roundtrip intact" with nothing ever decoding a byte; it is now a real decode-and-compare, and the comment records that the discriminating red is a *same-width field transposition*, not a width change.

**ng_snn/nmnist_pipeline.py** (485 L) — NG-SNN Milestone 2: sum T=15 event frames → log1p → per-sample max → 2-conv SoftHebb (2→96→384, greedy, no backprop) → ridge linear probe (front-end ceiling) + NG-SNN 20k random-LIF pool with analytical activity-dependent ridge (`NGSNN._rates`, Eq. 9–12). `to_rate_image`, `SoftHebbNMNIST`, `train_block`, `extract_and_save`, `train_minmax`, `ridge_linear_probe`, `ngsnn_head`, `_retry_oom`. **Gotcha**: `train_block` asserts exactly one block's conv weights changed ("GREEDY VIOLATION") — the real correctness guard.

**ng_snn/nmnist_spatiotemporal.py** (598 L) — Same skeleton, but trains on **per-timestep frames** ((B,T)→(B·T)) and temporally pools features (`mean` | `meanmax`). Adds `frames_to_batch`, `st_features`, `_ridge_chunk`, `_cg_solve` (CG on the resident GPU Gram when a d×d LU factorization copy would OOM at d_FE=49152). Reaches ~99.19% vs the 97.25% static baseline.

**ng_snn/nmnist_spatiotemporal_3conv.py** (633 L) — The 2-conv file plus a third SoftHebb block (384→1536, AvgPool(2,2,0)). Everything else is a byte-level copy. `d_conv` is measured live from a forward pass, not hardcoded.

## 3. REGION NOTES

Three disjoint clusters share this letter range:

- **Fluid-intelligence / symbolic induction** (`gf.py`, plus `model_builder`, `lake_reason` as chat-path consumers). This is the only genuinely *wired* cluster: `spike_chat_lake` reaches `gf.letter_series`, `lake_reason.Reasoner`, and `model_builder.ModelBuilder` on the live reply path; `reasoning_cortex` and `rpm_solver` pull `gf` + `mcr_hdc`. Core data structures are nested-tuple expression trees (`('+', ('n',), ('c',2))`) evaluated by per-cluster `_evaluate_*`, with layer caches keyed on `(max_size, library-names)` and durable learned-primitive libraries persisted as `_T_STATE` records inside `brain_lake.bin`.
- **JEPA/H-JEPA torch trainers** (`jepa_scaffold_torch`, `hjepa_torch`, `hjepa_arm_torch`). Offline train-wide/serve-narrow: torch trains, exports a JSON artifact (`snn_jepa/2`, `snn_hjepa/1`), and a pure-Python core server (`faculties/jepa_world_model`, `faculties/hjepa_serve`) replays it on the live tick. The connection to the brain is *through the artifact format*, not through an import — which is exactly why all three have zero referrers.
- **Benchmarks / instruments / baselines** (`hopfield_baseline`, `index_compression`, `mcr_hdc`, `neural_heterogeneity`, `nmnist_ingest`, `loop_latency`, `make_results_plot`, `nef_integrator`) plus the **NG-SNN GPU reproduction** (`ng_snn/nmnist_*`). Fourteen of the twenty are registered in `run_selfchecks.py`, so most of this slice *is* already gated despite being ruff-excluded.

Two modules are genuinely infrastructural to P0: `impact_analysis` (consumed by `brain_native_selfbuild` to prefer surgical mutations) and `godel_machine` (the self-rewrite protocol). Both are the highest-value promotion targets.

A striking property of this slice: nearly every module carries an in-file forensic record of a defect that its own gate could not see — vacuous asserts, module-scope side effects, stale run commands, false stated reds. Several document reds that were *constructed and run and did not fire*, then corrected. That auditing convention is the most valuable thing here and should survive promotion verbatim.

## 4. DEFECTS

**Dormant (defined, never invoked)**
- `godel_machine.py::attempt_on_repo` — zero call sites tree-wide (module imported by `self_mod_harness`, function never called). P3 shape 1. Self-documented; wiring belongs in the caller.
- `hjepa_torch.py`, `hjepa_arm_torch.py`, `jepa_scaffold_torch.py` — zero referrers, absent from `run_selfchecks`. Their asserts never execute in any gate.
- `nengo_gui_brain.py` — zero referrers; entered only via the `nengo` CLI by file path. Its one live criterion is unreachable in `.venv-lake`.
- `make_results_plot.py::render` — writes `assets/results.svg`, which is gitignored and read by nothing (module documents this as a P3 shape-2 consumer gap).
- `ng_snn/nmnist_pipeline.py`, `nmnist_spatiotemporal.py`, `nmnist_spatiotemporal_3conv.py` — no referrers, not in any gate, and **not importable as package modules**: all three do `from softhebb_mnist import ...` / `from data import ...` (bare top-level names) beside a `pass  # migrated: sys.path bootstrap removed` marker, while `ng_snn/__init__.py` is a docstring only. The bootstrap was deleted without converting the imports to relative form. This is almost certainly part of the ~30% of `reasoning/` that does not import in the gate env.

**Duplicated across modules (P2)**
- `vicreg_terms` — three byte-identical copies: `hjepa_torch.py`, `hjepa_arm_torch.py`, `jepa_scaffold_torch.py`.
- `_ridge_fit` / `_ridge_pred` / `_mse` — identical in `hjepa_torch.py` and `hjepa_arm_torch.py`.
- `train_step`, `build_hjepa`/`build_hjepa_arm`, `export_hjepa`/`export_hjepa_arm` — `hjepa_arm_torch` reuses `hjepa_torch`'s `_HJepa` "wholesale" *by copy-paste*, its own docstring says so.
- `ng_snn/nmnist_spatiotemporal_3conv.py` is ~95% a copy of `nmnist_spatiotemporal.py` (only the third block differs). `train_block`, `extract_and_save`, `train_minmax`, `_norm_chunk`, `_ridge_chunk`, `_cg_solve`, `ridge_linear_probe`, `ngsnn_head`, `_retry_oom`, `frames_to_batch`, `st_features` are all duplicated verbatim; `ngsnn_head`/`train_minmax`/`_norm_chunk`/`_retry_oom` are duplicated a *third* time in `nmnist_pipeline.py`.
- `gf.py` — five near-identical `_render_*`/`_evaluate_*`/`_layers`/`_enumerate` families and three separate layer caches (`_LAYER_CACHE`, `OpenGf._layer_cache`, `_ENUM_CACHE`). This is deliberate and documented ("combined, NOT merged"), but it is exactly the P2 shape and should be adjudicated rather than inherited silently on promotion.

**Stale (comment/docstring contradicting the code or the tree)**
- `ng_snn/nmnist_pipeline.py:2`, `nmnist_spatiotemporal.py:2`, `nmnist_spatiotemporal_3conv.py:2` — every docstring header still says `research/ng_snn/...` and every run command is `.venv-cu128/Scripts/python research/ng_snn/&lt;file&gt;.py`. The files are at `src/snn_data_lake/reasoning/ng_snn/`, and `.venv-cu128` was **retired 2026-08-05** per CLAUDE.md. Three unrunnable commands.
- `hjepa_torch.py:87`, `hjepa_arm_torch.py:95`, `jepa_scaffold_torch.py:75` — `Run:` lines still name the retired `.venv-cu128`.
- `nengo_gui_brain.py:5` — same retired `.venv-cu128\Scripts\nengo.exe`.
- `hjepa_torch.py:5` and `jepa_scaffold_torch.py:6` claim "torch 2.13.0+cpu"; CLAUDE.md records the gate env at torch 2.11.0+cu128. Recorded results, so not necessarily wrong — but unverifiable against the current env.
- `hjepa_arm_torch.py::export_hjepa_arm` and `hjepa_torch.py::export_hjepa` write `"unverified_on_hardware": True` **unconditionally into the exported artifact**, including on runs whose docstrings state the run was executed. A hardcoded provenance flag that no run can clear.
- `mcr_hdc.py:10` — `Run: python mcr_hdc.py`, a path that no longer exists (module is `snn_data_lake.reasoning.mcr_hdc`). Same shape in `neural_heterogeneity.py:35` (`python neural_heterogeneity.py`) and `nmnist_ingest.py:23` (`python nmnist_ingest.py`).
- `hopfield_baseline.py:4` refers to `spiking_storage_prototype.py` — verify that module still exists under that name before promoting.

**Hardcoded where an adaptive/measured value belongs (P1)**
- `gf.py::_MAX_SIZE = 7`, `_CONSTS_pi = (1,2,3)`, `_CONSTS_ogf`, `_CONSTS_al`, `_SHIFTS = range(-15,16)`, `OpenGf.__init__(budget=12, cap=60000)`, `_enumerate(cap=20000)`, `ActiveLearner(max_size=7, span=range(1,13), top_n=12, max_queries=10)`. The file argues these are *resource guards*, and `solve_unbounded`/`grow_budget` genuinely make the ceiling raisable — but none is derived from a running baseline. The `_CONSTS` sets in particular are a fixed grammar prior, not a budget.
- `ng_snn/nmnist_pipeline.py:449`, `nmnist_spatiotemporal.py:556`, `nmnist_spatiotemporal_3conv.py:588` — `FIELD = 99.5` ("field SOTA") hardcoded in three places and printed as the gap denominator. A literature number frozen in code, exactly the class `make_results_plot` was corrected for.
- `ng_snn/nmnist_spatiotemporal_3conv.py:623` and `nmnist_spatiotemporal.py:590` — prior-run results (`99.19%`, `97.25%`) hardcoded into report strings as the baseline being beaten. Hand-maintained, will rot.
- `hjepa_torch.py:119` `XOR_LINEAR_FLOOR = 0.25` — this one is legitimately a *provable* floor, not a tuned constant.
- `hjepa_arm_torch.py:120` `REGIME_TO_ACTION = {"0": "RELEASE", "1": "ENGAGE"}` — a fixed policy map exported into the served artifact, with `HOLD` declared "never optimal" by fiat rather than by measurement.
- `neural_heterogeneity.py:46-61` — 16 fixed experiment constants (`N`, `T`, `PER`, `A`, `CONC_STEPS`, `DIFF_STEPS`, `O_HI/O_LO`, `TAU_MIN/MAX`, `V_TH`, `REFRACTORY`, `NOISE`, `EPOCHS`, `LR`) plus asserts tuned to the measured margin (`het &gt; homo + 0.12`, `het &gt; long + 0.1`). Defensible as a pinned replication; the assert thresholds are the fragile part.
- `loop_latency.py:20` `BUDGET_MS = 50.0` — explicitly labelled generous and machine-dependent; the file says the check is structural, so this is honest.
- `nengo_gui_brain.py:92-94` — `_COUNT_RADIUS`/`_RADIUS_HEADROOM`/`_N_NEURONS`, with an explicit in-file P1 disposition arguing they are display scale and arbitrate no brain decision. Correct handling; cite as the model for the others.

**Other**
- `nmnist_ingest.py::load_events` catches bare `except Exception` around the whole tonic path and silently degrades to synthetic data — the "silent degradation" shape the tree elsewhere records. It does tag the source string honestly, which mitigates it, but a download failure and an ImportError are not the same event.
- `ng_snn/*` all set `torch.backends.cudnn.benchmark = True` and call `torch.cuda.empty_cache()` unguarded in several paths that also claim to run on CPU (`DEVICE = "cuda" if available else "cpu"`); `torch.cuda.empty_cache()` on a CPU-only box is a no-op, so this is benign, but `_retry_oom` catching `torch.cuda.OutOfMemoryError` is dead on CPU.

**Promotability summary** — Promote: `impact_analysis`, `lake_reason`, `index_compression`, `mcr_hdc`, `nef_integrator`, `hopfield_baseline`, `neural_heterogeneity`, `model_builder`, `godel_machine` (with the `attempt_on_repo` wire as separate work), `gf` (largest, already load-bearing on the chat path). Merge-then-retire: the three JEPA trainers (one shared `vicreg_terms`/ridge/export module), and the three `ng_snn/nmnist_*` into one parameterized pipeline. Candidates for retirement: `make_results_plot` (zero consumers, gitignored output) and `nengo_gui_brain` (zero referrers, unverifiable in the gate env) — both need an owner decision, not a code change.</result>
<usage><subagent_tokens>291081</subagent_tokens><tool_uses>31</tool_uses><duration_ms>385406</duration_ms></usage>
</task-notification>

---


## Region digest 9 of 25

<task-notification>
<task-id>a7c1d4130bb0a9726</task-id>
<tool-use-id>toolu_013vLztB3fQjzLneTbKpQd8C</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a7c1d4130bb0a9726.output</output-file>
<status>completed</status>
<summary>Agent "Read brainstem+thalamus" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>All 19 files read in full (6,918 + 5,675 lines confirmed against the brief).

## 1. COUNT

**brainstem — 13 .py, 6,918 lines**: `__init__.py`(1) `arm_bridge.py`(149) `arm_config.py`(339) `arm_hardware.py`(580) `autonomous_loop.py`(3074) `closed_loop.py`(256) `cpg_oscillator.py`(167) `interpreter.py`(162) `live_arm.py`(464) `reflex.py`(169) `serial_arm.py`(230) `signal_loop.py`(822) `so101_arm.py`(505)

**thalamus — 6 .py, 5,675 lines**: `__init__.py`(5) `connectome_router.py`(450) `global_workspace.py`(3547) `gwt_deep.py`(388) `reentrant_workspace.py`(955) `reticular_gate.py`(330)

## 2. PER MODULE

**brainstem/arm_config.py** — The single source of calibration + the SO-101 body vocabulary. `JOINTS` (6 `JointSpec` rows in qpos order, MJCF name canonical), `ROTATABLE`/`TOKENS`/`MOVETO_TOKENS`, `token_for`/`command_for`/`moveto_for`, `HOME_QPOS`, `dbin`/`present`, `CONTACT`, `SAFETY_CALIBRATED`. `JOINT_LIMITS` and `MAX_REACH_MM` are PEP-562 lazy attrs read from the vendored MuJoCo model, never transcribed. Gotcha: `ctrl_lo/hi` (actuator, what binds a command) and `stop_lo/hi` (jnt_range, the physical stop) are measured **unequal** and must not be conflated — the gate clamps to ctrl, `probe_contact` gates on stop. Model load is lazy on purpose (~134 ms, wide fan-in).

**brainstem/so101_arm.py** — THE gate that owns the body. `So101Arm.apply()` is the only mutation path (latch→vocabulary→clamp→actuate→report); `probe_contact`/`_verdict`/`OUT_OF_LIMIT`/`GROUND`/`SELF`; `state()`, `collision()`, `trajectory_breach()`, `set_substeps()`, `inject_disturbance()`. Gotchas: `fullmatch` not `match` (a `.match` once executed a rotate and silently dropped a trailing `EMERGENCY_STOP`); `_ROTATABLE` is derived from `arm_config.TOKENS` so `JOINT_GRIPPER_ROTATE` is rejected; ROTATE accumulates on `_commanded`, never on achieved qpos; a latched arm returns **before** `mj_step`, so it is bitwise frozen.

**brainstem/serial_arm.py** — Drop-in `So101Arm` surface that also writes each command to the ESP32. Three wire guards: calibration (`assert_calibrated`), E-STOP freeze (`_ALLOWED_WHEN_STOPPED`), and the allow-list (`is_valid_command`, regex alternated over `arm_config.ROTATABLE` — built independently of the mirror, which is what caught a malformed token the mirror accepted). `allow_uncalibrated=True` **requires** an injected `writer=`, so the escape hatch cannot open a real port. Serial write errors raise (fail-safe).

**brainstem/reflex.py** — Unlearned nociceptor fast-path over RAW signed samples. `Reflex.step(raw)` (one leaky LIF, severity = fraction past a channel's range), `DEFAULT_RULES` {ch6→STOP, ch7→WITHDRAW}, `reflex_guard()` side-channel generator. Gotchas: non-finite reading ⇒ `sev=inf` (fail-closed — NaN made every comparison False and failed OPEN); a `persist=10` latch catches sustained marginal breaches the leak would miss; an absent rule channel is reported to stderr once and recorded in `.missing`, never silently skipped. Consumed by `signal_loop`/`closed_loop`, **not** by `live_arm`/`autonomous_loop` (those run their own distance-based contact reflex inline).

**brainstem/interpreter.py** — Matched label → concrete command, with the priority `reflex &gt; AVOID veto &gt; match`. `Interpreter.interpret(ev)-&gt;(cmd, reward)`, `COMMANDS` (derived from `arm_config.ROTATABLE`, not hand-listed), `REFLEX_CMD`, `run_pipe`. Rewards: reflex −1.0, AVOID −0.3, executed 0.0 (+0.5 only under `--assume-success`) — self-declared honest placeholder pending real task sensing.

**sensorimotor/live_arm.py** — Sensor→recognize→Interpreter→arm loop. `run(samples, arm, interp, emit, stride, observer)`, `serial_samples`, `_serial_samples_from` (one-UART duplex), `SENSOR_COMMANDS`. Contact reflex is `not isfinite(d) or d &lt; CONTACT` → `EMERGENCY_STOP`. CLI opt-ins (`--reach`, `--sweep`, `--plan-xyz`, `--proprioception`) all route through gated `apply()`. `_OBS_JOINT` bound by NAME because qpos index 0 became shoulder_pan (yaw), not the old pitch.

**faculties/cpg_oscillator.py** — `MatsuokaCPG` half-center oscillator + `cpg_sweep(arm,...)`, which drives `arm.apply(moveto_for(...))` rate-limited inside `arm_config.ground_free_band(_CPG_JOINT)`, breaking on collision. Gotcha: uses the ABSOLUTE MOVETO verb — differencing into a relative delta made the setpoint run away from the lagging servo and drove the arm into the floor after 85/1200 steps.

**sensorimotor/closed_loop.py** — In-process wiring of reflex + match + valence(dopamine) + cortisol + Interpreter. `run_episode(signals, ..., bridge, valence_init)`. Holds **no arm**; it computes commands and rewards only. P4b: the learned valence policy is a `_T_STATE` record in `brain_lake.bin`, proven by a child interpreter, not a sidecar.

**sensorimotor/arm_bridge.py** — Authenticated UDP `OUTCOME` back-channel for the split-process deployment. `OutcomeBridge`, `send_outcome`, `parse_outcome` (clamped to [-1,1], constant-time token compare). Per-instance random token by default; `token=""` opts out.

**sensorimotor/signal_loop.py** — Encode→match→emit pipeline. `parse_line` (rejects non-finite whole lines), `normalize`, `build_matcher` (Van Rossum + Lowe ratio) / `build_learned_matcher` (hybrid, default), `run`, `run_live` (closed loop w/ cortisol-sharpened reflex), `cluster_unknowns`/`learn_unknowns` (vocabulary growth), `load_library`/`save_library` into the one lake under `SIG_STATE`.

**engine/autonomous_loop.py** — The big learned sensorimotor loop. `AutonomousArm.step()` is the tick: reflex/E-STOP → perceive → grow a unit → build value row (tabular / `latent_q` / `mlp_q`) → `_select`/`_bg_select` → allow-list re-check → `arm.apply()` → postural recovery → reward/RPE → ~20 opt-in representation-only faculties. `ACTIONS` = ENGAGE/RELEASE/HOLD + 10 derived JOINT tokens, asserted against the wire allow-list at import. `coverage_reward` is the physical objective. Gotchas: every faculty flag defaults False (a default arm is byte-identical); `_estop_self_clear` is opt-in via `NORD_ESTOP_SELF_CLEAR=1` precisely because turning it on reddens ten "the E-STOP freeze must hold" assertions.

**sensorimotor/arm_hardware.py** — Build TARGET only; `built` is always False and nothing reads a sensor. `budget()` (hand formula), `model_budget()` (MuJoCo `qfrc_bias` + payload Jacobian), `buildable_configurations()`, `state()` (memoised — the sweep was ~22% of the live tick). Headline asserted so it cannot be erased: shoulder needs 15.0 kg·cm against 7.5 rated, and the model is harsher than the formula (24.41), making wrist_pitch a third underpowered joint. 500 g at 300 mm has **zero** feasible configurations.

**faculties/global_workspace.py** — The production multi-slot GWT bus. `GlobalWorkspace.submit/step/broadcast/broadcast_all/broadcast_hold/subscribe`, `band_claim`/`band_bid`/`next_baseline` (the ONE shared derivation of a compliant claim, so two producers compete in one coordinate system), `capacity` = `floor(1/(self_boost+hysteresis))` = 4, `grip` = `hold/capacity`, `_defended` (tenancy scoped to the SOURCE), `_rank_key` (admission, carries the hold + final membrane) vs `_focus_key` (head, carries only the grip). Controls kept live: `OneSubsidyArbiter`, `NoAllowanceArbiter`, `_SingleSlotControl`. Gotchas: `submit()` is single-threaded only (buffer order seeds the jitter) — concurrent specialists use `open_slots`/`submit_from`; `_fold_parts` must stay `sum()` over concatenated chunks (Neumaier compensation; a naive `acc +=` fold silently changes membranes and therefore admission).

**faculties/reentrant_workspace.py** — The non-saturated GNW: ignition can FAIL, re-entry is a closed loop, subscribers are re-routed, and `expect()` gives predictive completion with a STRUCTURAL no-hallucination guard (a prior is added only where bottom-up support exists, so magnitude 10.0 conjures nothing). `submit/expect/subscribe/broadcast_to/step/access_strength/association/inbox`; `reentry=False` is the fair control. `access_strength()` is a 32-unit heterogeneous readout — it replaced a single-pool rate that was pinned at 1.000. Consumed by `engine/brain.py::_c1_tick`.

**faculties/gwt_deep.py** — Phase-binding demonstration + the spiking-vs-rate control. `spiking_broadcast`/`spiking_decode` (theta-gamma slots, binding by synchrony) vs `rate_broadcast`/`rate_decode` (superposition), scored by F1 in `_pair_accuracy`, with `_reds()` reconstructing the failing input for every criterion. Honest: K=1 TIES. Dormancy-EXEMPT "permanent" — nothing on the live tick binds K objects.

**faculties/reticular_gate.py** — Genuinely spiking TRN attentional gate: LIF relay per channel + inhibitory pool with spread thresholds, feedback giving emergent divisive normalization. `attention/gate/mask/channels/sdr/random_sdr`. `mask()` only REDISTRIBUTES a fixed gain budget. The random-permutation control (same gains, wrong channels) is what proves selectivity rather than energy. Called from `autonomous_loop._attend`.

**faculties/connectome_router.py** — Makes the static `connectome.py` graph propagate spikes: one LIF pool per node, edges become signed/delayed projections (exc adds, inh subtracts, neuromod scales gain). `step(region_drive)`, `reverberation(region, ticks)`, `reset_state`. Representation-only — imports no arm module; `actuator`/`viscera` are inert sinks. The threaded `_lif_update` is row-partitioned and byte-identical to the whole-array form.

## 3. REGION NOTES

**brainstem** holds two distinct stacks that share only `Reflex` and `Interpreter`. The *signal* stack (`signal_loop` → `reflex` → `interpreter` → `closed_loop`, with `arm_bridge` as the cross-process reward transport) works on raw multi-channel windows, encodes to spikes, matches against a learned library, and closes a dopamine/cortisol loop. The *arm* stack (`arm_config` → `so101_arm` ← `serial_arm`, driven by `live_arm`, `cpg_oscillator`, `autonomous_loop`) works on a proximity scalar and a 6-DOF MuJoCo body. `arm_hardware` is orthogonal: a build spec with no runtime input. The invariant tying it all together is that `arm_config` owns the vocabulary and the limits, and every producer *derives* its tokens from it rather than typing them — that is what makes a single allow-list sufficient. Upward, `autonomous_loop.AutonomousArm` is the composition `engine/brain.py` and `engine/live_brain.py` drive; downward, `So101Arm.apply()` is the only thing that writes qpos.

**thalamus** holds three workspace implementations answering different questions, plus a gate and a router. `global_workspace` is the live multi-slot broadcast bus (the one `autonomous_loop._workspace_broadcast` and `faculties/access_consciousness` post to, and `live_brain.thought` reads); `reentrant_workspace` is the anti-saturation GNW with a genuinely closed re-entry loop that `engine/brain.py` subscribes to; `gwt_deep` is a retained benchmark. Data flow is uniformly submit→settle(LIF competition under shared inhibition)→ignite→admit up to capacity→broadcast→decay. `reticular_gate` is a *pre*-workspace attentional stage feeding `autonomous_loop`'s percept; `connectome_router` is a parallel representation layer over the whole-brain wiring graph that touches no actuator.

## 4. SAFETY ENVELOPE

**What each file enforces**

- **`arm_config.py`** — declares the physical facts nothing else may restate: `CONTACT = 0.04` m (contact-reflex trip), `TRAJ_DEV_MAX = 0.15` m, `IN_RANGE_MAX`, the 6-row `JOINTS` table, `HOME_QPOS`, and `SAFETY_CALIBRATED = False`. `JOINT_LIMITS` is **read from the MuJoCo model**, keeping `ctrl_*` (binds a command) separate from `stop_*` (solver-enforced physical stop). `ROTATABLE`/`TOKENS` decide *which joints are addressable at all* — the gripper is excluded by construction, so `JOINT_GRIPPER_ROTATE` cannot exist.
- **`reflex.py`** — the unlearned nociceptor arc on raw signed channels. Fires on a single severe breach, on a 10-sample sustained marginal breach (no dead band), and **fail-closed on any non-finite reading**. An absent safety channel is announced, never silently disabled.
- **`interpreter.py`** — command validation and priority. A `reflex` key preempts everything (`STOP`→`EMERGENCY_STOP`, `WITHDRAW`→`RETRACT_ALL`, unknown→`EMERGENCY_STOP`); an `AVOID` instinct vetoes to `HOLD`; only then does a confident match map through `COMMANDS`. An unmapped or novel label yields `None` — no command.
- **`so101_arm.py`** — the gate. `apply()` runs, in order: **(0) TYPE** — non-string is a no-op; **(1) LATCH** — if `stopped` and the command is not `HOME`/`RETRACT_ALL`, return *before* `mj_step`, so a frozen arm is bitwise frozen and repeats are inert; `EMERGENCY_STOP` is edge-guarded (sets the latch, pins `ctrl := qpos`, steps once); `HOME`/`RETRACT_ALL` clear the latch via a hard reset; **(2) VOCABULARY** — `_JOINT_TOKEN.fullmatch` plus membership in `_ROTATABLE`; **(3) CLAMP** to `ctrl_lo/ctrl_hi` with `clamped` reported true exactly when the command was altered; **(4) ACTUATE** `substeps × mj_step`; **(5) REPORT**. `inject_disturbance` also honours the latch unless the keyword-only `allow_latched=True` is passed (self-checks only).
- **`serial_arm.py`** — the wire. Refuses construction against real servos while `SAFETY_CALIBRATED` is False (checked for **every** writer, including injected ones); `allow_uncalibrated=True` requires an injected writer so it can never open a real port; once the mirror latches, only `EMERGENCY_STOP`/`HOME`/`RETRACT_ALL` reach the wire; anything failing `is_valid_command` touches the mirror only; a write failure raises rather than driving blind.

**Every path to `So101Arm.apply()`**

1. **Autonomous (primary).** `AutonomousArm.step()` → if `distance` is non-finite or `&lt; CONTACT`: `arm.apply("EMERGENCY_STOP")` and return, before any learned action. Otherwise: optional `NORD_ESTOP_SELF_CLEAR` → `arm.apply("HOME")` (gated on the breach predicate having just read clear, and counted in `_estop_clears`) → `_select`/`_bg_select` picks from `ACTIONS` → **defence in depth**: `if action not in _UNCHECKED and not is_valid_command(action): raise ValueError` → `arm.apply(action)` → postural recovery may `arm.apply(_A_POS)`, gated on `collision == "GROUND" and not stopped`.
2. **Live sensor.** `live_arm.run()` → `not isfinite(d) or d &lt; CONTACT` → `Interpreter.interpret({"reflex":"STOP"})` → `"EMERGENCY_STOP"` → `arm.apply(cmd)`. Otherwise W samples → `classify` → `interpret({"match": gesture})` → `SENSOR_COMMANDS` (ENGAGE/RELEASE/HOLD) → `arm.apply(cmd)`. `arm` is `So101Arm` or `SerialArm` (which mirrors into a private `So101Arm` **and** gates the wire).
3. **Rhythm.** `cpg_sweep` → rate-limited `arm.apply(moveto_for(...))`, breaking on `arm.collision()`.
4. **CLI opt-ins** (`--reach` spiking PID, `--plan-xyz`/`--plan` wavefront planner) — cerebellum modules that emit MOVETO tokens already on the allow-list; no new token, no new route.
5. **Split-process.** `signal_loop --feedback | interpreter --pipe` emits command text; the actuator half is `SerialArm`, so the wire allow-list + freeze apply identically. `arm_bridge` carries only rewards, clamped to [-1,1] and token-authenticated.

Every one of these terminates at `So101Arm.apply()`. There is no second writer of `qpos` except `inject_disturbance`, which is latch-honouring and has no live-tick caller.

## 5. DEFECTS

**Likely real bug (static, not run — I am read-only):**
- `sensorimotor/live_arm.py::main` `_MOTION_CMD = "JOINT_0_ROTATE(+5deg)"`. `so101_arm._JOINT_TOKEN` captures `([A-Z_]+)`, so a **digit cannot match** — this token takes the "unrecognised → no command change, but time still passes" branch and then runs `substeps × mj_step`. The control immediately above it asserts the token moves an *unlatched* arm; after 120 CPG steps that assertion is satisfiable by residual settling under gravity rather than by commanded motion. The comment block explicitly says this control exists because "a frozen arm and an inert command are indistinguishable" — the replacement literal appears to have the same property as the `JOINT_A_ROTATE` it replaced. Worth running: pick a token from `arm_config.command_for(...)` instead.

**Dormant / write-only (all self-declared, none force-wired):**
- `autonomous_loop.py::AutonomousArm._attend` — `_bound_modalities` is structurally 0 forever (the `norm &gt; 0.15` cut is unreachable against divisive normalization's 0.25 equal-share ceiling); three replacements built and reverted. Belongs in `dormancy_gate.EXEMPT`, still not filed.
- `autonomous_loop.py::AutonomousArm.step` (agency block) — `AgencyMonitor.observe()` fed every tick; `agency()`/`controllability()`/`status()` have zero readers. P3 shape 4.
- `autonomous_loop.py::AutonomousArm._pfc_bias` — PFC working memory (`maintain`/`push_subgoal`) unreachable from this loop by construction (no input-free interval); wiring measured and reverted.
- `autonomous_loop.py::AutonomousArm.__init__` — `_dev_explore`, `_dev_fm_gain`, `_motor_gain`, `_naca_credit` are computed-and-stored readouts documented as "never routed".
- `reentrant_workspace.py::ReentrantWorkspace.association` / `.inbox` — zero external readers per the module's own AST census; accessor-tier dormancy.
- `connectome_router.py::_lif_update` — the threaded branch is unreachable in production (live width 960 is ~500× below the crossover and nothing raises `pool_size`); the module calls it "the weakest-standing of the three parallel paths".
- `arm_config.py::IR_PRESENT = None` — the IR-confirmation branch in `present()` cannot fire until someone calibrates polarity.

**Stale text contradicting the code:**
- `live_arm.py::main` — `arm = So101Arm()  # stdlib backend (pybullet if installed)`. `So101Arm` is MuJoCo-only; there is no stdlib or pybullet backend. The `--reach`/`--sweep` print strings also still say "joint A", a joint that no longer exists (the code correctly binds `shoulder_lift` by name).
- `so101_arm.py::So101Arm.apply` — inside the LATCH branch, `if command == "EMERGENCY_STOP": return self.state()` is followed by an identical `return self.state()`. The branch is dead; the comment above it carries the real content.
- `closed_loop.py` / `signal_loop.py` — import via the legacy flat aliases (`from snn_data_lake import arm_bridge as AB`, `signal_loop as S`, `learned_matcher as L`) while the modules live under `brainstem/`. Works through the back-compat layer, but it is the pre-3.0 path.

**Duplicated constant:**
- `arm_hardware.py` declares `GRAVITY = 9.81` at module scope, commented "only used for documentation", and separately `G_MS2 = 9.81` which is the one the payload math actually uses. Two spellings of one constant, one of them dead — delete `GRAVITY`.

**Hardcoded where a learned/derived value belongs (excluding the sanctioned physical-limit exception):**
- `live_arm.py::SLIP_DEG = 15.0` — arbitrates the slip watchdog on an *estimator mismatch*, not a physical limit, and is not baseline-relative. This is the clearest P1 candidate in the two directories.
- `reentrant_workspace.py::ReentrantWorkspace._ACC_LO/_ACC_HI` — the module itself records this as OPEN P1 (fixed constants arbitrating `access_strength`), with a stated reason for deferring (a running band breaks deterministic gate reproduction, and a rewrite of this exact shape already shipped a division singularity elsewhere).
- `interpreter.py::Interpreter.interpret` — the `+0.5` under `--assume-success`; self-labelled an honest placeholder pending real task-success sensing.

**Known unbounded growth (documented, unfixed):**
- `signal_loop.py::run` — the `encoded` return value grows for the life of the process on the never-returning `--serial` path. The fix is a two-file edit with `sensorimotor/sensor_csv_ingest.py` (switch it to `stats["events"]`, then drop `encoded`); the docstring warns explicitly against doing half of it.

**Not defects — sanctioned fixed values inside the envelope:** `arm_config.CONTACT`, `TRAJ_DEV_MAX`, `NEAR_MAX`/`MID_MAX`/`IN_RANGE_MAX`, `reflex.DEFAULT_RULES` channel ranges, and every `JOINT_LIMITS` value (the last are read from the model, not typed). These stay fixed *because* they are safety constants.</result>
<usage><subagent_tokens>412703</subagent_tokens><tool_uses>29</tool_uses><duration_ms>379023</duration_ms></usage>
</task-notification>

---


## Region digest 10 of 25

<task-notification>
<task-id>ad9450294a6e67d73</task-id>
<tool-use-id>toolu_013KWDmFkTfJdCX3AJUo4Xd9</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\ad9450294a6e67d73.output</output-file>
<status>completed</status>
<summary>Agent "Read small regions + src root" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>34 files read in full (5,758 lines). Digest below.

## 1. COUNT — 34 .py files

| location | n | files |
|---|---|---|
| package root | 3 | `__init__.py`, `brain_atlas.py`, `connectome.py` |
| training | 7 | `__init__.py`, `cli.py`, `mnist.py`, `mnist_genn.py`, `offload` (retired 2026-10-02), `spiking_lm.py`, `trainer.py` |
| midbrain | 5 | `__init__.py`, `dopamine_nuclei.py`, `multisensory_integration.py`, `sc_deep_layers.py`, `superior_colliculus.py` |
| hypothalamus | 4 | `__init__.py`, `circadian_clock.py`, `homeostatic_drives.py`, `set_points.py` |
| amygdala | 3 | `__init__.py`, `affective_appraisal.py`, `fear_conditioning.py` |
| medulla | 3 | `__init__.py`, `interoceptive_arousal.py`, `vital_nuclei.py` |
| lakehouse | 3 | `__init__.py` (0 bytes), `delta_demo.py`, `medallion.py` |
| claustrum | 2 | `__init__.py`, `claustrum.py` |
| epithalamus | 2 | `__init__.py`, `habenula.py` |
| olfactory | 1 | `__init__.py` |
| pons | 1 | `__init__.py` |

## 2. PER MODULE

**training/`__init__.py`** — docstring only; describes the tier as core Torch+Brian2 → JSON weight export. Stale: documents a `training/device.py` that does not exist and "mirrors research/torch_device.py" (also gone); omits `mnist.py`/`mnist_genn.py`/`spiking_lm.py` (= AUDIT_BACKLOG M55).
**training/cli.py** — `snn-train` console script; argparse over four paths (default readout / `--mnist` / `--genn` / `--lm`), lazily importing `trainer`, `offload`, `mnist`, `mnist_genn`, `spiking_lm`. Gotcha: **both device imports target a retired module** (see DEFECTS).
**training/trainer.py** — `SpikingReadout` (`nn.Linear(N,C,bias=False)` + `BrianLIF`), surrogate-grad BPTT + Adam over `readout.snn_classifier`'s 8x8/4-class task; `train_model`/`evaluate`/`save_checkpoint`/`load_checkpoint`/`export_to_core`. Invariant: `fc.weight` is `[C][N]` == the core's `W[c][i]`, so export is a `.tolist()` with no transpose; the docstring's honest caveat is that the trainer optimises LIF spike dynamics while the core serves a *linear* spike-count readout, so accuracy does not carry over.
**training/mnist.py** — 784→H→10 MnistBrain whose forward is byte-for-byte `sensorimotor.trained_brain.predict` (beta=0.9, thr=1.0, reset-to-zero, output *accumulates*); AMP + DataLoader workers on CUDA only; `export_brain` + `verify_transfer` re-score through the zero-dep core. Invariant: shapes `[H][784]`/`[10][H]` are asserted before `make_brain`/`save_brain`.
**training/mnist_genn.py** — GeNN/ml_genn 2.x sibling of `mnist.py` (SpikeInput→LIF→LeakyIntegrate, EventProp+Adam). **Self-declared UNVERIFIED against any live ml_genn install**; every API touchpoint carries a `# VERIFY(ml_genn):`. `_beta_to_tau` maps `tau_mem = -dt/ln(beta)` — an approximation, so transfer to `trained_brain` is explicitly lossy.
**training/offload** (retired 2026-10-02) — ZeRO-Offload demonstration: master weight + Adam state on CPU, `torch.utils.checkpoint` over the STEPS unroll. Imports `SpikingReadout`, `rate_encode`, `_to_tensors`, `_generator`, `evaluate` from `trainer` (private-name coupling). Reached only via `cli --offload`.
**training/spiking_lm.py** — char-level SpikeGPT-lineage LM: `SpikingBlock` (LIF-in → causal exp-decay token mixer → LIF-out → FFN + graded residual), `SpikingLM`, `train`/`generate`/`perplexity`, checkpoint/resume, `export_lm` → `snn_lm/1` JSON served by `sensorimotor.spiking_lm_serve`. Torch import is guarded so the module imports torch-free (`_Module = object` fallback); `main()` prints shape and exits 0. `main()` uses `infra.device.resolve_device` — the correct selector — while `cli.py` uses the retired one.

**faculties/dopamine_nuclei.py** — `DopamineNuclei`: TD(0) critic over a trial chain emitting RPE as a LIF population rate (`da_level`, `phasic`, `present`). Source of dopamine only — no synapses, no action. Gotcha: a deleted vacuous assert is documented in-file (value cache is seed-independent because `present()` consumes no randomness); the two surviving determinism asserts have a constructed reachable red.
**faculties/superior_colliculus.py** — `SuperiorColliculus.saccade()`: surround-contrast rectification → LIF race → first spike wins, lateral IPSPs suppress losers. Reuses `faculties.action_gating.kwta` for the argmax cross-check so the two competition circuits tie-break identically (`(-v, i)`). Invariant: a flat/high-energy map must produce **zero** spikes — contrast, not energy, drives it.
**faculties/sc_deep_layers.py** — `SCDeepLayers`: OPN gate → charge-conserving (subtractive-reset) LIF burst pool → gain-1 zero-leak line-attractor integrator; displacement is `gain * accumulated_spike_count`, asserted to within 1e-12 so it can never be scalar arithmetic. Composes `SuperiorColliculus` via `gaze_from_map`. Invariant: **returns DATA, imports nothing from brainstem/, actuates nothing.** `self.gain` is calibrated at construction from a unit-error burst.
**faculties/multisensory_integration.py** — `MultisensoryCell`: one `growth.lif_neuron.LIFNeuron` fed two summed currents; reproduces Stein–Meredith super-additivity, inverse effectiveness and the temporal coincidence rule from threshold + F-I saturation + leak alone, untrained. Docstring carries an explicit self-correction (super-additivity is threshold rectification, *not* F-I convexity).

**sleep/circadian_clock.py** — `CircadianClock` phase oscillator, tau=24.2, sinusoidal PRC toward `DAY_TARGET`; `step` returns a dawn tick on wrap; `is_rest`/`consolidation_gate`/`baseline` are pure functions of phase. Arnold-tongue behaviour asserted (locks at Tz=24, ignores Tz=18) plus jet-lag re-entrainment.
**faculties/set_points.py** — `Homeostat` (comparator → antagonistic proportional effector → plant with `leak` toward `ambient`); `temp_setpoint(phase)` is PACED by a real `CircadianClock`, not a stand-in. Fair control: gain=0 open loop relaxes to ambient. Same class regulates osmolarity.
**faculties/homeostatic_drives.py** — `Drive` + `HomeostaticDrives`: energy/curiosity/social deficits with drift+replenish; `deficit()` percept, `dominant_drive()`, `motivational_bias()` (softmax over deficit×priority), and a **local two-factor reward-modulated delta rule** whose reward is the drive reduction the performed activity caused. Control: `shuffle=True` credits a random activity and collapses the value gap. Invariant: value-side only — selects no action.

**faculties/fear_conditioning.py** — `Amygdala`: per-`us_id` `w_exc` (three-factor CS-US potentiation via `pair()`) + `w_inh` (extinction), CeA readout `threat()` = max over categories, `rest()` drives time-labile inhibitory recovery → spontaneous recovery. **Load-bearing status note in the docstring: on the live brain `threat()` returns exactly 0.0 on all 38 calls per 80 wake ticks** because `pair()`'s only call site is the contact/E-STOP branch and no aversive events occur; disposition is P3 EXEMPT-awaiting-prerequisite, explicitly NOT "wire it".
**faculties/affective_appraisal.py** — `AffectiveAppraisal`: three LIF pools (`w_pos`/`w_neg`/`w_aro`) over the same latency code; `learn()` is an RPE-gated three-factor Hebbian/anti-Hebbian update scaled by a `neuromod` scalar; `step(decay)` is slow forgetting. Gotcha: `self.seed` is stored and never used (documented in-file).

**faculties/vital_nuclei.py** — `MedullaVitalNuclei`: pre-Bötzinger relaxation oscillator whose rate is set by a CO2 negative-feedback loop, plus an NTS/RVLM baroreflex. Fair controls: `feedback=False` freezes drive at `drive0` (CO2 drifts) / zeroes baro control (pressure stranded).
**faculties/interoceptive_arousal.py** — `InteroceptiveArousal.sense(load)` **composes** the real `MedullaVitalNuclei` chemoreflex and a real `faculties.set_points.Homeostat`, reads their residual set-point error, subtracts a resting baseline computed at construction, and maps to `stress`/`caution`/`gain`∈[1,cap]. Invariant: caution-only, no actuator, imports nothing from brainstem/.

**faculties/claustrum.py** — `Claustrum`: ≥k distinct areas active in a trailing window → fire + bind assembly → `broadcast` gain applied to bound areas only. `scatter_async` is the un-riggable control (identical per-area spike counts, spacing `window+1`); `single_area_stream` is louder than the whole sync budget and must still bind nothing. Docstring explicitly disclaims the claustrum-as-consciousness hypothesis.

**faculties/habenula.py** — `LateralHabenula`: rectified negative RPE → LIF burst; a self-contained downstream DA scalar with `da = max(0, da_base + gain*pos - w_rmtg*lhb_rate)`. Imports no other region. Causal control: `w_rmtg=0` abolishes the omission dip. Per-call reseed makes the spike count a pure function of the outcome.

**lakehouse/medallion.py** — Bronze→Silver→Gold Parquet pipeline over `substrate.spike_telemetry_hub.synth`, gated by `substrate.data_quality.gate` before Gold; produces ICR, synchrony CV, a co-firing materialized view, and a latency handoff. Needs `polars`.
**lakehouse/delta_demo.py** — delta-rs small-file/OPTIMIZE/time-travel PoC. Needs `polars` + `deltalake`. Writes/deletes `lakehouse/data/delta_bronze` via `shutil.rmtree`.
**lakehouse/`__init__.py`** — **0 bytes.**

## 3. THE MAP

**`__init__.py`** — Runs `_require_free_threaded()` at import: refuses to import unless `sysconfig.get_config_var("Py_GIL_DISABLED")`, with a narrow `NORD_ALLOW_GIL=1` warn-only escape. Declares `REGIONS`: **17 keys** — cortex, hippocampus, basal_ganglia, amygdala, cerebellum, brainstem, thalamus, hypothalamus, midbrain, **pons: []**, medulla, epithalamus, **olfactory: []**, claustrum, neuromodulation, plasticity, substrate. Plus two annexes with the *same* anatomical status but excluded from the alias layer: `TORCH_TIER` (cortex: neurogenesis_engine, language_acquisition, eprop_nmnist, eprop_kernel, eprop_shd; hippocampus: ng_snn) and `TRAIN_TIER` (plasticity: snn_mnist_stdp, deep_stdp; cortex: deep_stdp_latency; hippocampus: developmental_loop_brain) — gated on `torch` and `torchvision` respectively. **Alias layer:** `_MOVED = {m: f"{r}.{m}"}` built from REGIONS only (with an `m != r` guard kept for a future region==module collision); `_AliasFinder` (a `MetaPathFinder` appended to `sys.meta_path`) resolves single-segment legacy `snn_data_lake.&lt;flat&gt;` names, and `_AliasLoader.create_module` returns the *already-imported regional module object* — a true alias, not a copy. `exec_module` then restores the real module's `__spec__`/`__loader__`, because the import machinery stamps the alias spec onto the shared object and would otherwise break `runpy`/`get_code` on the real module. A module-level `__getattr__` covers `from snn_data_lake import lif_neuron`.

**`brain_atlas.py`** — Renders the map and self-checks it. `region_of(module)` scans REGIONS then both annexes; `atlas()` returns REGIONS. `main()` prints per-region counts and a three-way total, then asserts: (a) **filesystem parity per region** — `_region_files(region)` vs `REGIONS ∪ TORCH_TIER ∪ TRAIN_TIER`, no unmapped files and no phantom entries (this replaced a hardcoded count assert); (b) regions are **disjoint** across all three tiers; (c) **every** `_MOVED` legacy path resolves to the *identical object* (`old is new`); (d) every region subpackage imports; (e) annex modules region-resolve and import when torch/torchvision are present, SKIP-with-notice otherwise.

**`connectome.py`** — The EDGES layer. **40 NODES** (`node_id -&gt; (region|None, human name)`; `actuator` and `viscera` have region `None`), **83 edges**, **14 canonical loops**. Each edge is built by `_e(source, target, tract, sign, cls, loop=None, weight=1.0, delay=1)`: `sign ∈ {exc, inh, da, ne, 5ht, ach}`, `cls ∈ {relay, feedback, broadcast, descending}`. Loops: thalamocortical, cbgtc, hippocampal, cortico_cerebellar, papez, orienting, reward_opponent, olfactory, claustro_cortical, respiratory, baroreflex, mesolimbic, olivocerebellar, global_workspace. Graph helpers: `has_edge`, `reachable` (BFS with `block_node`/`block_edge`), `route`, `is_closed_loop`, `cortical_nodes`. `main()` asserts every loop is a closed directed cycle; the **arm-safety invariant** — the actuator's only in-edge is `(bs_motor, actuator)`, blocking `bs_motor` disconnects *every* cortical node plus `sc_deep`/`pbc`/`nts_rvlm`/`viscera` from the actuator; and a **fair control** that the graphify EXTRACTED import graph carries no sign/loop/tract keys (skipped, not failed, if `graphify-out/graph.json` is unreadable).

**CONDUCTION DELAY — the field exists and is uniformly dead.** `_e` carries a `delay` parameter (ticks, added 2026-08-04) and every edge dict has a `"delay"` key, so this is more than tract names. But **not one of the 83 `_e(...)` calls passes a `delay=` argument** — all 83 take the default `1`. The consumer is real and complete: `faculties/connectome_router.py:85` reads `int(e.get("delay", default_delay))`, `:90-95` computes `_maxdelay`, `:107` sizes the ring buffer, `:119` indexes `(self._t - self._delay) % L`. With every delay at 1, `_maxdelay == 1` brain-wide and the ring buffer is 2 rows — i.e. every tract (thalamocortical radiation, corticostriatal, pallidothalamic) arrives in exactly one tick. `_e`'s own docstring states the default is deliberate and that real per-tract values must come from anatomy (DTI-derived), pending the `tools/ab_connectome_delay.py` A/B.

## 4. DEFECTS

- **`training/cli.py:86` and `:141` import `snn_data_lake.reasoning.torch_device` — that module does not exist** (retired 2026-08-04 in the P2 merge into `infra/device.py`; `infra/device.py:270` records the retirement). Consequences: (a) line 141 sits inside `except ImportError:` so the *entire* default/`--mnist`/`--genn` path exits 1 printing `snn-train needs the training extra: pip install ".[train]"` **even with torch installed** — a swallowed failure reported as the wrong cause; (b) line 86 is **unguarded**, so `snn-train --lm` raises a bare `ModuleNotFoundError` traceback, which is exactly what the module docstring promises never happens. `gates/dormancy_gate.py:66` registers `snn_data_lake.training.cli` as a live ENTRYPOINT, so the gate counts a CLI that cannot execute past its own imports. Fix is one symbol: `from snn_data_lake.infra.device import resolve_device`. Note `training/spiking_lm.py::main` already uses the correct selector — the same job resolved two ways in one subpackage (P2).
- **`training/__init__.py:24-25`** documents `device.py` in this package (absent) and says it "Mirrors research/torch_device.py" (also absent), and omits three of the six real modules. Stale doc = AUDIT_BACKLOG M55.
- **`lakehouse/` and `training/` are not keys in `REGIONS`**, so `brain_atlas.main()`'s filesystem-parity assert never iterates them: 9 of the files in this read (7 training + 2 lakehouse) are invisible to the atlas map. Under P4c ("no outside the brain") these are undispositioned. `lakehouse/__init__.py` is **0 bytes** — no docstring, no disposition.
- **`connectome.py::_e` — `delay=1` on all 83 edges.** A structural hardcode where an anatomical value belongs (P1): the producer supplies a constant to a consumer built to vary. Uniform 1-tick conduction is the one thing STDP and coincidence detection are for.
- **`training/mnist_genn.py` — entire module is an unexecuted scaffold.** `_extract_core_weights` guesses that `conn.connectivity.weight` holds the trained array after `network.load(...)`; ~10 `# VERIFY(ml_genn):` sites; `metrics["unverified_against_live_genn"] = True` is carried into the exported brain's meta. Reachable only via `snn-train --genn`, which is currently unreachable anyway (defect 1).
- **Duplicated `rate_encode` / `_generator` / `_datasets`** across `training/trainer.py`, `training/mnist.py` and `training/mnist_genn.py` (three encodings of the same Bernoulli rate coder, two `_datasets` bodies identical). `offload` (retired 2026-10-02) imported trainer's *private* `_to_tensors`/`_generator` — a cross-module private coupling.
- **`faculties/fear_conditioning.py::Amygdala.pair`** takes no US magnitude: the third factor is binary (which method you call), so a stronger shock cannot condition faster. Flagged in-file as open work; the `us_id` channel is where it would live. Also `LR_EXC`, `LR_INH`, `US_CURRENT`, `SAT_EXC`, `THETA_CEA` are hand-tuned module constants where learned/adaptive values belong (P1), as are `TH_BLA`/`LEAK`.
- **`faculties/affective_appraisal.py::AffectiveAppraisal.__init__`** stores `self.seed` and never reads it (self-documented dead attribute).
- **`faculties/sc_deep_layers.py::SCDeepLayers.__init__`** hardcodes `v_threshold=1.0`, `cur=1.2`, `drain=30`, `hold_steps=20`, `n_hold=8` as instance literals outside the signature — not tunable by a caller, not learned.
- **`faculties/homeostatic_drives.py::DEFAULT_DRIVES`** — drift/replenish/priority per drive are designer-chosen literals (disclosed honestly in the docstring, but they are the numbers that decide `dominant_drive()`); `BETA=6.0` and `LR=0.05` likewise.
- **`lakehouse/medallion.py:33` and `delta_demo.py:30`** hardcode `LAKE = "lakehouse/data"` **relative to CWD**, and both docstrings give run commands (`lakehouse/medallion.py`, `PYTHONPATH=.`) from a repo layout that predates the move into `src/snn_data_lake/`. `delta_demo.py:45` does an unconditional `shutil.rmtree` on that path. Neither is registered in the atlas; both are also a potential P4b tension (a second durable store on disk beside `brain_lake.bin`).

**olfactory/ and pons/ are NOT real modules — they are docstring-only placeholders.** Each is a lone `__init__.py` of 5 lines containing only prose, and `REGIONS["olfactory"]` / `REGIONS["pons"]` are both **`[]`**. But the connectome declares four nodes in them — `ob`, `pir` (olfactory) and `pn`, `prf` (pons) — carrying seven edges and the closed `olfactory` loop plus the pontine limb of `cortico_cerebellar` and the ARAS broadcast. So the wiring layer asserts traffic through two regions that contain no code at all: `brain_atlas`'s parity assert passes vacuously (empty set == empty set), and nothing can be dormant because nothing exists. That is a declared-but-unbuilt region pair, not an empty directory.</result>
<usage><subagent_tokens>271211</subagent_tokens><tool_uses>42</tool_uses><duration_ms>344894</duration_ms></usage>
</task-notification>

---


## Region digest 11 of 25

<task-notification>
<task-id>a6132118105ec8c3f</task-id>
<tool-use-id>toolu_01QAXJUcG64zKpYUWViPog8V</tool-use-id>
<output-file>C:\Users\vinhl\AppData\Local\Temp\claude\D--AI-project\f20e7c6f-afba-48b3-a647-63fd44b47c4c\tasks\a6132118105ec8c3f.output</output-file>
<status>completed</status>
<summary>Agent "Read tools/ gates in full" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>Read all 30 `.py` files in `tools/` in full, plus `run_selfchecks.py` and `learn_daemon.py`. CI wiring verified against `.github/workflows/ci.yml`; test coverage verified against `tests/`.

## 1. COUNT — 32 files

`tools/` (30): ab_connectome_delay, ar_lm_disjoint, ar_lm_ngram_control, arc_baseline, arc_census, arc_designb, arc_fixdepth_control, arc_probe_ids, arc_score, arm_benchmark, blimp_baseline, capacity_curve, charge_probe, citation_guard, coverage_score, deep_fingerprint, dependency_ledger, doc_command_guard, fanout_bisect, instrument_guard, lang_coverage, lang_diag, lang_probe, lang_scale, p4c_ratchet, p6_gate, promotion_sweep, red_reachability, regime_split, timing_heldout. Repo root (2): run_selfchecks.py, learn_daemon.py.

## 2. PER TOOL

**Gates / guards**
- `p4c_ratchet.py` — per-zone ceiling on ungoverned `.py` files (git corpus minus AST-derived REGIONS/TORCH_TIER/TRAIN_TIER). `--assert --self-check --update-baseline`. **CI ci.yml:193, :209.**
- `p6_gate.py` — workspace write:read topology + connectome-vs-import diff; 5 ratchets. `--assert --self-check --root PATH`. **CI :301, :314.**
- `dependency_ledger.py` — UNDECLARED / UNUSED / MISDESCRIBED over pyproject. `--assert --self-check`. **CI :253, :264.**
- `red_reachability.py` — flags zero-arg criteria and tautological `assert verdict`. `--assert` only. **CI :231** (no self-check exists).
- `citation_guard.py` — tracked file may not cite an untracked `.py` as a runnable path. No flags; asserts unconditionally in `main()`. **CI :223.**
- `instrument_guard.py` — 5 determinism/P4 rules over `tools`+`tests`+`src`, plus a "can this instrument tell you it's broken" census. `--assert --selfcheck`. **Not in ci.yml; asserts via `tests/test_instrument_guard.py` under pytest (:329).**
- `doc_command_guard.py` — documented `python -m snn_data_lake.…` must resolve, be non-alias, have `__main__`. `--assert --selfcheck`. **Wired nowhere.**
- `coverage_score.py` — refuses band claims unsupported by recorded artifacts; `record`/`--rerun`/`--repin`/`--check`. **Deliberately not in CI** (documented). Three pytest files drive it.
- `charge_probe.py` — P3 tier-2 neuter-A/B on a live brain; exit 0 charge / 1 not-decisive / 2 NOT-ELIGIBLE-or-UNKNOWN. No flags. **Not wired, no test.**
- `promotion_sweep.py` — classifies out-of-gate modules across P0–P8. `--summary`. Report-only, unwired.
- `run_selfchecks.py` — the canonical module list (CORE/RESEARCH/TORCH_GATED/NENGO_GATED/TRAIN_GATED); `--list --in-process`; import-time `_assert_tiers_agree_with_regions()`. **CI :111.**
- `learn_daemon.py` — 33-line shim to `engine/learn_daemon.py`; no logic by design.

**Measurement instruments** (all report-only, none in CI)
- `ab_connectome_delay.py` — two-verdict A/B on per-tract delay (field vs. spread) with mean-matched, bracket, and magnitude-recruitment controls; 2 constructed reds in `_selfcheck`.
- `timing_heldout.py` — held-out A/B for one grown time cell; `--seeds --regimes --bounding`; eligibility gate exits 2.
- `fanout_bisect.py` — F* bisect with pre-registered tolerances; `--brain` else `_selfcheck` (4 constructed properties).
- `deep_fingerprint.py` — 127k-leaf object-graph diff of a Brain; `--demo` A/A control else `_selfcheck`.
- `arm_benchmark.py` — MuJoCo SO-101 torque/reach; `--payload-kg`.
- `capacity_curve.py` — NG-SNN capacity; modes `control|grid|saturation`; `control` carries two real asserts.
- `regime_split.py` — time-clock saturation regime; `--seeds --ticks`.
- `lang_scale.py` / `lang_diag.py` / `lang_probe.py` / `lang_coverage.py` — corpus-scale BLiMP table (`--verify` equivalence check), 7 ablation probes, 88-dim decodability, frozen restricted pair set (exit 2 = NOT-P2-ELIGIBLE).
- `blimp_baseline.py`, `ar_lm_ngram_control.py`, `ar_lm_disjoint.py` — language "before" numbers, n-gram control, contamination proof (exit 1 on contamination).
- `arc_score.py`, `arc_baseline.py`, `arc_designb.py`, `arc_probe_ids.py` — ARC rates, three-solver baseline, Design-B telemetry + label-shuffle control, per-ID accounting.
- `arc_census.py`, `arc_fixdepth_control.py` — **permanent SKIP records**; print a retired verdict and exit 0. They measure nothing at HEAD.

## 3. THE GATE INVENTORY

| gate | goes RED when | red reachable? | ratchet or absolute | self-check constructs a failure? |
|---|---|---|---|---|
| `p4c_ratchet` | any zone's live ungoverned count &gt; its baseline (absent zone ⇒ baseline 0) | YES — add any `.py` git sees outside REGIONS | **RATCHET** (baseline total 202, per-zone) | YES but **only over `compare()` on synthetic dicts**; 4 reds + 2 controls incl. cross-zone laundering. Never exercises `census()`/`_corpus()`/`_governed()`. |
| `p6_gate` (5 criteria) | `private_instance_modules&gt;6`, `unread_writer_modules&gt;6`, `cross_module_reads&lt;1`, `undeclared_import_pairs&gt;11`, `unimplemented_declared_pairs&gt;21`; **or** any liveness problem | YES — 6 constructed overlay mutations, one per criterion (incl. relative-import spelling) | **RATCHET**, zero slack. Explicitly *not* "P6 satisfied": `fac_cross` moved **0 → 1 at T039h** (this cell said "= 0 and deliberately unasserted" for three days after — US1 audit B5); the floor + identity live in `tests/test_interoceptive_specialist.py` | YES, strongest in the tree: in-memory overlay mutations + `_mutate_text` anchor-rot guard + **3 green controls** (attribute hoist, the P6 fix itself, an extra private site) + determinism re-run + `liveness()` (parse failures, KNOWN_WRITERS/READERS floors, role re-derivation, connectome non-empty) |
| `dependency_ledger` | any unexempted UNDECLARED, any UNUSED, any MISDESCRIBED (CORE-PROSE/CORE-ONLY/ALL-EXTRA/STALE-EXCLUDE), or a stale EXEMPT | YES — 5 string-copy mutations of pyproject | **ABSOLUTE** ("the ledger balances") | YES, and it guards against vacuous reds: a mutation that changes nothing prints `BROKEN` and exits 1 (this fired on its own author, see `_REDS` STALE-EXCLUDE comment) |
| `red_reachability` | any zero-arg criterion, any tautological `assert x`, or a stale EXEMPT | Partly. NO_RED and TAUTOLOGICAL are reachable in principle. **STALE-EXEMPT is structurally unfireable: `EXEMPT = {}`** | **ABSOLUTE** | **NO — there is no `--self-check` flag at all.** Covered externally by `tests/test_red_reachability.py` |
| `citation_guard` | a STRICT (shell-fence / Reproduce-heading) citation of an untracked `.py`, or a stale EXEMPT key | YES; BROAD count is printed and never asserted | **ABSOLUTE** (0-unexempted) | No dedicated flag, but carries two **liveness floors** (`n_files &gt; 100`, `n_cites &gt; 50`) — the only gate here that guards the "walk died ⇒ zero findings ⇒ looks clean" shape by assert |
| `doc_command_guard` | a doc's `-m` command names a missing module, a flat alias, or a module with no `__main__` | YES — `selfcheck()` writes a temp `.md` with `snn_data_lake.interpreter` and asserts the regex matches and the resolver rejects | **ABSOLUTE** | YES, and it also asserts EXEMPT entries still bless a real violation. **But `--assert` is invoked by nothing** |
| `instrument_guard` | P4-CONSTRUCTION, CWD-DEPENDENT-PATH, MODULE-SCOPE-MUTATION, CONSTRUCTOR-SEED, SYNTAX; or stale/thin EXEMPT | YES for all 5 tags — 13 labelled fixtures written to a temp tree | **ABSOLUTE** | YES, best-in-class: 13 reds + a `_GREEN` fixture asserting the *repaired* shapes are not flagged + a cross-table consistency assert (`GLOBAL_SEEDERS ⊆ BANNED_MODULE_SCOPE`) |
| `coverage_score` | F1–F14 refusals (missing category, unresolved citation, band&gt;40 without an exit-0 artifact, band&gt;80 without held-out, `--rerun` mismatch) | YES — `--rerun` re-executes and compares exit code + fingerprint | **ABSOLUTE**, but not enforced anywhere | No self-check flag; three pytest files pin `_fingerprint`, citations, CLI |
| `charge_probe` | exit 1 = consumed-but-not-decisive; exit 2 = NOT-ELIGIBLE/UNKNOWN | YES behaviourally | Neither — a **verdict producer**, not a gate | **NO self-check, no asserts, no test file** |
| `ab_connectome_delay::_selfcheck` | criterion 1/2 fail their green-or-red pair | YES — each criterion evaluated on a green input *and* a constructed red | verdict tool | YES (and it records deleting a 3rd "criterion" that was `assert _selfcheck(...)` against a function returning literal `True`) |
| `fanout_bisect::_selfcheck`, `deep_fingerprint::_selfcheck` | 4 properties each | YES (incl. "a failing dense baseline must refuse the sweep") | instrument | YES |
| `run_selfchecks::_assert_tiers_agree_with_regions` | a REGIONS/TORCH_TIER module listed in RESEARCH, or in no tier | YES | **ABSOLUTE** | No — and see defect 6 |

## 4. DEFECTS

1. **`p4c_ratchet.py::census` has no liveness floor — the flagship P4c gate can pass by seeing nothing.** If `_corpus()` returns an empty list (git present but returning no paths), every zone count is 0, every zone is ≤ its baseline, and `main()` prints `gate OK: 0 ungoverned builds`. `p4c_ratchet.py::_selfcheck` cannot catch this: it runs `compare()` against a hand-written `base = {"src/snn_data_lake/reasoning": 161, "tools": 30, …}` and never touches the walk. `p6_gate.py::liveness` and `citation_guard.py::main`'s `assert n_files &gt; 100` both guard exactly this shape; the ratchet does not. A broken walk here reads as a fully promoted tree.
2. **`red_reachability.py` has no `--self-check`.** CLAUDE.md and `docs/POLICIES.md` P7/P8 both state the four policy gates ship "each with `--assert` AND its own `--self-check`"; ci.yml:231 runs only `--assert`, and no such flag is parsed in the file. Additionally `red_reachability.py::EXEMPT = {}` makes `stale` permanently empty, so one of the three terms in `failed = bool(no_red or tautologies or stale)` cannot fire. Same file: a zero-criterion `scan()` prints `gate OK` and exits 0.
3. **`doc_command_guard.py` is wired nowhere.** It has a working `--assert` and a genuinely constructed red, but it appears in no ci.yml step and has no `tests/test_doc_command_guard.py`. Its documented `--selfcheck` flag is also never parsed — `main()` calls `selfcheck()` unconditionally at line 254 and the flag is inert.
4. **`instrument_guard.py::EXEMPT = {}`** makes two of three failure terms unreachable: `failed = bool(unexempt or stale or thin)` where `stale = set(EXEMPT) - live_exempt` and `thin` both derive from an empty dict.
5. **`charge_probe.py` is the least-verified instrument in the tier and the most consequential.** It is P3's second tier — its exit code is what retires a faculty — and it has no `--self-check`, no `assert`, and no test file, so it is one of `instrument_guard`'s own SILENT count. Worse: CLAUDE.md P8 states "`tools/charge_probe.py` carries an external positive control, 47/60 flips, proving the harness is sensitive at all." **It does not.** The 47/60 figure appears only in prose comments (`charge_probe.py:83-87`, `:189-191`) citing `SpatialTemporal.md`; no control is implemented or run. A harness whose sensitivity claim lives in a doc is the "harness false negative" shape.
6. **`run_selfchecks.py::_assert_tiers_agree_with_regions` swallows its own precondition.** `except Exception: return` on the `from snn_data_lake import REGIONS, TORCH_TIER` import means any package import failure silently deletes the entire governed-in-RESEARCH check — a check whose recorded output was "1, now 0" and which caught two real promotions.
7. **Duplicated governed-set derivation, and the two copies disagree.** `p4c_ratchet.py::_governed` reads `REGIONS`, `TORCH_TIER` **and `TRAIN_TIER`**; `promotion_sweep.py::_governed` reads only `REGIONS` and `TORCH_TIER`. So every `TRAIN_TIER` module (`instruments.snn_mnist_stdp`, `instruments.deep_stdp`, `instruments.deep_stdp_latency`, `instruments.developmental_loop_brain`) is governed to the ratchet and ungoverned to the sweep — `promotion_sweep`'s P2-DUP column undercounts and its target list is wrong by that set.
8. **`p6_gate.py::BASELINE` is a hardcoded dict in source**, unlike `p4c_ungoverned_baseline.json`, which is machine-generated and whose writer *refuses a rise* (`p4c_ratchet.py::main`, `--update-baseline`). Raising a P6 ceiling is a one-line source edit with no mechanical refusal — only the docstring's "raising a ceiling is an admission" convention. Also `cross_module_reads` is a floor of **1**, resting on a single call site in `engine/live_brain.py`.
9. **The `N of M` / `N/M` print convention is a real cross-tool coupling with no gate.** `coverage_score.py::FLIPS` harvests every bare ratio in a probe's stdout positionally into the artifact fingerprint, so any tool printing `N/M` silently invalidates `--rerun` for every artifact it produced. This is enforced only by hand-written comments in four files (`charge_probe.py:157`, `instrument_guard.py:548`, `regime_split.py:86`, `timing_heldout.py:37`). Nothing checks it.
10. **`pop_opt` is duplicated verbatim in nine tools** (arc_baseline, arc_score, arc_designb, arc_probe_ids, ar_lm_disjoint, ar_lm_ngram_control, blimp_baseline, lang_scale, lang_coverage) — P2's additive shape at instrument scale.
11. **`capacity_curve.py` is the odd tool out on two axes.** It has module-top `import torch` + `from snn_data_lake.growth.ng_snn import NGSNN` with **no `sys.path` bootstrap** (every other tool derives `src` from `__file__`), so it only runs under an external `PYTHONPATH`; and its three reproduction commands name `.venv-cu128/Scripts/python`, which `docs/POLICIES.md` P5 records as **RETIRED 2026-08-05**. A tracked reproduction path naming a deleted interpreter is exactly what `citation_guard` exists to catch, and it cannot — the guard resolves `.py` citations, not venv paths.
12. **`lang_coverage.py:50-51` assigns `os.environ["NORD_HOME"]` and calls `tempfile.mkdtemp()` at module scope.** `instrument_guard.BANNED_MODULE_SCOPE` covers `os.chdir` and the seeders but has no rule for direct `os.environ[...] =` (only `setdefault` is discussed, and deliberately unruled). Importing this module mutates process env and leaks a temp dir; `lang_diag`/`lang_probe` import from `lang_scale`, not this file, so the blast radius is currently one file — but the rule gap is general.
13. **`coverage_score.py::report` still orders `--check` behind the refusals** (`if fail: … return 1` precedes the `--check` block), so on any refused ledger the baseline-agreement check is unreachable. The file documents this having already caused a real outage (the `README.md` F2 case); the ordering is unchanged.
14. **`citation_guard.py::_EXEMPT_FILES` self-exempts `tools/citation_guard.py`.** Justified (its docstring quotes the offending paths), but it means the one file whose docstring is densest in `.py` paths is outside its own gate.
15. **Two tools are permanent no-ops.** `arc_census.py::main` and `arc_fixdepth_control.py::main` print a SKIP and return 0 unconditionally at HEAD. They are honest instruments-of-record, but they inflate the `tools/` denominator in `instrument_guard`'s census and can never redden anything.</result>
<usage><subagent_tokens>333148</subagent_tokens><tool_uses>39</tool_uses><duration_ms>416574</duration_ms></usage>
</task-notification>

---


## Region digest 12 of 25

Read IN FULL every Python file in these locations of D:\AI-project\spiking-neural-data-lake:
  src\snn_data_lake\*.py            (package root: __init__.py, brain_atlas.py, connectome.py -- 3 files, 1,275 lines)
  src\snn_data_lake\training        (7 files, 1,776 lines)
  src\snn_data_lake\midbrain        (5 files, 1,205 lines)
  src\snn_data_lake\hypothalamus    (4 files, 1,015 lines)
  src\snn_data_lake\amygdala        (3 files, 674 lines)
  src\snn_data_lake\medulla         (3 files, 542 lines)
  src\snn_data_lake\claustrum       (2 files, 275 lines)
  src\snn_data_lake\lakehouse       (3 files, 266 lines)
  src\snn_data_lake\epithalamus     (2 files, 215 lines)
  src\snn_data_lake\olfactory       (1 file)
  src\snn_data_lake\pons            (1 file)

RULES:
- List each location, then read EVERY .py file COMPLETELY. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read per location + their bare filenames.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. THE MAP: __init__.py declares REGIONS and a back-compat alias layer; brain_atlas.py renders the region->module map; connectome.py declares region->region wiring (typed/directed/signed loops). Describe all three precisely -- what the region list is, how the alias layer works, and what the connectome actually declares. Note specifically whether the connectome carries any conduction-DELAY field or only tract names.
4. DEFECTS: anything dormant, duplicated, stale, or a hardcoded constant where a learned value belongs. Cite file:symbol. Note that olfactory/ and pons/ are 6-line files -- say plainly whether they are real modules or empty placeholders.

CONTEXT: governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop, no hardcodes where adaptive values belong.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 13 of 25

Read IN FULL every Python file in D:\AI-project\spiking-neural-data-lake\tools (30 files, 9,898 lines), plus these repo-root files if they exist: run_selfchecks.py, learn_daemon.py.

RULES:
- List the directory, then read EVERY .py file COMPLETELY. Use Read with offset/limit to page through large files (p6_gate.py is 1,648 lines) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their bare filenames comma-separated.
2. PER TOOL: 1-3 dense lines -- what it does, its CLI surface (--assert / --self-check / other flags), and whether it is wired into CI.
3. THE GATE INVENTORY -- the most important section. For each gate/ratchet (p4c_ratchet, p6_gate, dependency_ledger, red_reachability, citation_guard, doc_command_guard, charge_probe and any others), state precisely:
   - what condition makes it go RED
   - whether that red is REACHABLE (can you construct an input that fails it?) or whether it is structurally impossible to fail
   - whether it is a RATCHET (asserts "no worse") or an ABSOLUTE assert (asserts "policy satisfied") -- these are very different claims
   - whether it has its own --self-check, and whether that self-check actually constructs a failing case rather than just running the happy path
4. DEFECTS: gates that cannot fail, gates wired in bare (imported but never asserted), duplicated logic across gates, stale baselines, or thresholds hardcoded where they should be derived. Cite file:symbol.

CONTEXT: governed by docs/POLICIES.md (P0-P8). The project's own rule is "adding a policy means building its gate and COUNTING its output FIRST", and it has been burned before by gates that were green because they executed nothing. Be adversarial.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 14 of 25

Read IN FULL every Python file in these three directories of D:\AI-project\spiking-neural-data-lake:
  src\snn_data_lake\plasticity      (18 files, 5,311 lines)
  src\snn_data_lake\basal_ganglia   (10 files, 4,729 lines)
  src\snn_data_lake\neuromodulation (14 files, 4,104 lines)

RULES:
- List all three directories, then read EVERY .py file COMPLETELY. Use Read with offset/limit to page through large files (brain_native_selfbuild.py 1,248; r_stdp.py 877) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read per directory + their bare filenames.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of each region, main data structures and flows, how they connect to the rest of the brain.
4. LEARNING RULES: enumerate every distinct learning rule implemented across these three directories (STDP, R-STDP, e-prop, homeostatic, developmental, ...), say where each lives, and flag any that uses BPTT or weight transport -- the project's invariant is e-prop, NOT backprop, and anything that learns owes cosine > 0.5 against an independent gradient.
5. DEFECTS: anything dormant (defined but never called), duplicated across modules (two implementations of one rule is a P2 violation), stale docstrings, or hardcoded constants where a learned/running-baseline value belongs. Cite file:symbol.

CONTEXT: governed by docs/POLICIES.md (P0-P8). basal_ganglia/ also holds the self-build loop, which is the P0 self-improvement route -- report honestly whether it can actually propose anything.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 15 of 25

Read IN FULL every Python file in these two directories of D:\AI-project\spiking-neural-data-lake:
  src\snn_data_lake\brainstem  (13 files, 6,918 lines)
  src\snn_data_lake\thalamus   (6 files, 5,675 lines)

RULES:
- List both directories, then read EVERY .py file COMPLETELY. Use Read with offset/limit to page through large files (autonomous_loop.py 3,074; global_workspace.py 3,547; reentrant_workspace.py 955) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state. These directories contain the SAFETY ENVELOPE and must not be modified.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read per directory + their bare filenames.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of each region, main data structures and flows, how they connect to the rest of the brain.
4. SAFETY ENVELOPE: describe precisely what reflex.py, serial_arm.py, so101_arm.py, arm_config.py and interpreter.py each enforce, and trace the path every motor command takes to the So101Arm.apply() gate. This section is the most important part of your digest.
5. DEFECTS: anything dormant, duplicated, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned value belongs -- but note that raw physical limits inside the safety envelope are a SANCTIONED exception and stay fixed. Cite file:symbol.

CONTEXT: brainstem/ holds reflexes, actuation and the sensorimotor loop; thalamus/ holds the global workspace (GWT broadcast hub) and the reticular attentional gate. Governed by docs/POLICIES.md (P0-P8).
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 16 of 25

Read IN FULL every Python file whose bare filename starts with a letter in p-z, anywhere under D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\reasoning (RECURSIVE -- it has subpackages such as arc_agi/).

RULES:
- Enumerate the tree first (Glob **/*.py), filter to your slice by bare filename, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files (spike_chat_lake.py is 2,319 lines) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their repo-relative paths comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

CONTEXT: reasoning/ is a 111-module subtree that is ruff-excluded and outside the three dormancy gates; ~30% of it does not even import in the gate env. Per policy P4c it is still part of the brain and is being promoted into the governed tree file by file. Note which modules look promotable, which look like duplicates of governed modules, and which look dead. This repo is governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop, no hardcoded constants where adaptive values belong.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 17 of 25

Read IN FULL every Python file whose bare filename starts with a letter in g-o, anywhere under D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\reasoning (RECURSIVE -- it has subpackages such as arc_agi/).

RULES:
- Enumerate the tree first (Glob **/*.py), filter to your slice by bare filename, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files (gf.py is 1,909 lines) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their repo-relative paths comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

CONTEXT: reasoning/ is a 111-module subtree that is ruff-excluded and outside the three dormancy gates; ~30% of it does not even import in the gate env. Per policy P4c it is still part of the brain and is being promoted into the governed tree file by file. Note which modules look promotable, which look like duplicates of governed modules, and which look dead. This repo is governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop, no hardcoded constants where adaptive values belong.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 18 of 25

Read IN FULL every Python file whose bare filename starts with a letter in a-l, in D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\substrate (non-recursive).

RULES:
- List the directory first, filter to your slice, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files (cockpit.py 1,334 lines; home.py 1,285 lines) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their bare filenames comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

CONTEXT: substrate/ holds the durable lake (home.py -> brain_lake.bin, the ONE store; record tags like _T_GROWTH), the cockpit (the single frontend + the brain's sensorimotor surface), telemetry, and the policy GATES (dormancy_gate, method_dormancy_gate, eprop_transport_gate, doc_budget_gate, replication_guard, import_manifest). Pay special attention to how the lake reads/writes records and to what each gate actually asserts. Governed by docs/POLICIES.md (P0-P8), especially P4b: the durable lake IS part of the brain, and "it persists" is a claim about a NEW process.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 19 of 25

Read IN FULL every Python file whose bare filename starts with a letter in a-f, anywhere under D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\reasoning (RECURSIVE -- it has subpackages such as arc_agi/).

RULES:
- Enumerate the tree first (Glob **/*.py), filter to your slice by bare filename, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their repo-relative paths comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

CONTEXT: reasoning/ is a 111-module subtree that is ruff-excluded and outside the three dormancy gates; ~30% of it does not even import in the gate env. Per policy P4c it is still part of the brain and is being promoted into the governed tree file by file. Note which modules look promotable, which look like duplicates of governed modules, and which look dead. This repo is governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop, no hardcoded constants where adaptive values belong.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 20 of 25

Read IN FULL every Python file in these two directories of D:\AI-project\spiking-neural-data-lake:
  src\snn_data_lake\hippocampus  (25 files, 10,630 lines)
  src\snn_data_lake\cerebellum   (10 files, 4,677 lines)

RULES:
- List both directories, then read EVERY .py file COMPLETELY. Use Read with offset/limit to page through large files (growable_readout.py 1,078; episodic_recall.py 1,009; spiking_jepa.py 1,368; jepa_world_model.py 1,175) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read per directory + their bare filenames.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of each region, main data structures and flows, how they connect to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

CONTEXT: hippocampus/ holds memory formation, episodic recall, replay, spatial coding, and ng_snn.py (a contribution-grown spiking classifier used by neurogenesis). cerebellum/ holds motor coordination, timing, and forward/inverse + JEPA world models. Governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop (flag any BPTT), no hardcoded constants where adaptive values belong.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 21 of 25

Read IN FULL every Python file in D:\AI-project\spiking-neural-data-lake\tests (73 files, 10,659 lines).

RULES:
- List the directory, then read EVERY .py file COMPLETELY. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state. Do NOT run pytest.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their bare filenames comma-separated.
2. PER TEST FILE: 1-2 dense lines -- what behaviour it pins, and what would have to break for it to go red.
3. COVERAGE MAP: which brain regions/modules are well covered, and which are covered by nothing.
4. VACUITY AUDIT -- the most important section. Find tests that CANNOT FAIL or that prove less than they appear to:
   - assertions on a value that is constant regardless of the code under test
   - tests whose only assertion is that a call did not raise
   - tests that assert a count/shape but never the content
   - tests with a try/except that swallows the failure
   - tests that would still pass if the feature under test were deleted or neutered
   - negative controls that are missing entirely (a test proving X happens, with nothing proving X does not happen when it should not)
   Cite file:test_function and say exactly why the test is weak.

CONTEXT: this repo is a spiking neural network brain governed by docs/POLICIES.md, whose central honesty rule is that a criterion must have a REACHABLE RED -- you must be able to construct the input that makes it fail. Tests here have historically passed while the feature they pinned was dead. Be skeptical and specific.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 22 of 25

Read IN FULL every Python file whose bare filename starts with a letter in m-z, in D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\substrate (non-recursive).

RULES:
- List the directory first, filter to your slice, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their bare filenames comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

CONTEXT: substrate/ holds the durable lake, telemetry, the learn_daemon (the brain's DRIVE -- wake/sleep cycle and tick loop), and the policy GATES (method_dormancy_gate, replication_guard, import_manifest and others). Pay special attention to what each gate actually asserts and whether it can go red. Governed by docs/POLICIES.md (P0-P8), especially P3 (no dormancy; reachability is NOT charge) and P4b (the durable lake IS part of the brain).
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 23 of 25

Read IN FULL every Python file whose bare filename starts with a letter in a-f, in D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\cortex (non-recursive).

RULES:
- List the directory first, filter to your slice, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files (e.g. brain.py is 4,567 lines) until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their bare filenames comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

This repo is a spiking neural network brain governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop, brain-native constants not hardcodes. Flag violations you notice.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 24 of 25

Read IN FULL every Python file whose bare filename starts with a letter in g-n, in D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\cortex (non-recursive).

RULES:
- List the directory first, filter to your slice, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their bare filenames comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

This repo is a spiking neural network brain governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop, brain-native constants not hardcodes. Flag violations you notice.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---


## Region digest 25 of 25

Read IN FULL every Python file whose bare filename starts with a letter in o-z, in D:\AI-project\spiking-neural-data-lake\src\snn_data_lake\cortex (non-recursive).

RULES:
- List the directory first, filter to your slice, then read EVERY file in that slice COMPLETELY. Use Read with offset/limit to page through large files until you have seen every line. Do not skim, do not sample, do not stop early.
- You are READ-ONLY. Do not Write, Edit, or run any command that mutates state.

Return a digest, under ~250 lines, no preamble:
1. COUNT: number of .py files you read + their bare filenames comma-separated.
2. PER MODULE: 1-3 dense lines each -- what it does, key public symbols, how it is wired (who calls it, what it calls), and any invariant/gotcha a future editor must know.
3. REGION NOTES: architecture of this slice, main data structures and flows, how it connects to the rest of the brain.
4. DEFECTS: anything dormant (defined but never called), duplicated across modules, stale (docstring/comment contradicting the code), or a hardcoded constant where a learned/adaptive value belongs. Cite file:symbol.

This repo is a spiking neural network brain governed by docs/POLICIES.md (P0-P8): no dormancy, one mechanism per job, e-prop not backprop, brain-native constants not hardcodes. Flag violations you notice.
<context_window_protection>
  <priority_instructions>
    Every byte a tool returns enters your conversation memory and costs reasoning capacity for the rest of the session. The context-mode tools let you do the work in a sandbox and surface only the derived answer — the raw bytes stay out. Think-in-Code: program the analysis, do not compute it by reading raw data into your conversation.
  </priority_instructions>

  <deferred_tool_bootstrap>
    The context-mode tools below may be DEFERRED in your harness — their schemas are not loaded yet, so calling them directly fails (e.g. "tool not found" / InputValidationError). Load them ONCE before your first ctx_* call:
    ToolSearch(query: "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,mcp__plugin_context-mode_context-mode__ctx_search,mcp__plugin_context-mode_context-mode__ctx_execute,mcp__plugin_context-mode_context-mode__ctx_execute_file,mcp__plugin_context-mode_context-mode__ctx_fetch_and_index")
    After that they are callable. If any ctx_* call fails as not-found, ToolSearch it and retry — do NOT fall back to Bash/Read just because the schema was not loaded yet.
  </deferred_tool_bootstrap>

  <tool_selection_hierarchy>
    0. MEMORY: mcp__plugin_context-mode_context-mode__ctx_search(sort: "timeline")
       - On resume or compaction, query prior decisions, errors, plans, user prompts before asking the user — auto-captured session memory is searchable.
    1. GATHER: mcp__plugin_context-mode_context-mode__ctx_batch_execute(commands, queries)
       - Primary research tool. Runs commands in parallel, auto-indexes each output, and (when queries are passed) returns matching sections in the same round trip — no follow-up search call.
       - Each command: {label: "section header", command: "shell command"}; the label becomes the FTS5 chunk title — descriptive labels improve search.
    2. FOLLOW-UP: mcp__plugin_context-mode_context-mode__ctx_search(queries: ["q1", "q2", ...])
       - Multiple related questions about anything already indexed (your captures + session memory). Batch every question in one array; the ranking pipeline runs per-query and the round-trip cost is paid once.
    3. PROCESSING: mcp__plugin_context-mode_context-mode__ctx_execute(language, code) | mcp__plugin_context-mode_context-mode__ctx_execute_file(path, language, code)
       - Derive answers FROM data: filter, count, aggregate, parse, transform. Only what you console.log() enters your conversation; the raw bytes stay in the sandbox.
  </tool_selection_hierarchy>

  <when_not_to_use>
    - You intend to PROCESS the output (filter, count, parse, aggregate) → use mcp__plugin_context-mode_context-mode__ctx_batch_execute or mcp__plugin_context-mode_context-mode__ctx_execute. Bash stays correct when you intend to OBSERVE a short fixed output (git status on a clean tree, whoami, pwd) or when you are mutating state (git, mkdir, rm, mv, navigation).
    - You want to analyze, summarize, or extract from a file → use mcp__plugin_context-mode_context-mode__ctx_execute_file. Read stays correct when you intend to Edit the file (Edit needs the exact bytes in your conversation to match against).
    - WebFetch → use mcp__plugin_context-mode_context-mode__ctx_fetch_and_index; full network access, results indexed for mcp__plugin_context-mode_context-mode__ctx_search, raw page bytes never enter your conversation.
    - mcp__plugin_context-mode_context-mode__ctx_execute and mcp__plugin_context-mode_context-mode__ctx_execute_file for file writes → these run code in a subprocess and discard the sandbox FS; they are for analysis, processing, and computation only.
  </when_not_to_use>

  <file_writing_policy>
    File writes use the native Write or Edit tool — mcp__plugin_context-mode_context-mode__ctx_execute, mcp__plugin_context-mode_context-mode__ctx_execute_file, and Bash subprocesses do not persist edits to the host filesystem.
    Applies to all file types: code, configs, plans, specs, YAML, JSON, markdown.
  </file_writing_policy>

  <output_constraints>
    <artifact_policy>
      Write artifacts (code, configs, PRDs) to files. Return only: file path + 1-line description.
    </artifact_policy>
  </output_constraints>
  <session_continuity>
    Skills, roles, and decisions set during this session remain active until the user revokes them.
    Do not drop behavioral directives as context grows.
  </session_continuity>

</context_window_protection>

---
