# Development Policies (P0–P8)

**P0 is the purpose; P1–P8 are how it is pursued without lying to ourselves.**
Nine standing policies govern this brain. They are **mechanized** where possible (a CI
self-check that fails on violation) and otherwise enforced by review. Each has an **honesty guard** —
the rule that stops the policy from being satisfied by theater.

**Enforcement status, stated plainly, because it is easy to assume a gate exists:**

| | mechanized in CI | enforced by review only |
|---|---|---|
| **P0 autonomous self-improvement** | — | yes (its guards are P2/P3/verify-first) |
| P1 no-hardcode | — | yes |
| P2 adopt-or-merge | — | yes |
| P3 no dormancy | `substrate/dormancy_gate.py` (per-MODULE), `substrate/method_dormancy_gate.py` (per-METHOD), `substrate/import_manifest.py` (cross-tier) | the charge tier |
| P4 one brain (+ P4b partial) | `substrate/replication_guard.py`; **`substrate/lake_rebuild_gate.py` (2026-08-19, specs/022 T008)** mechanizes P4b's index-is-derived clause: delete → rebuild from a lake scan → logical-dump equality, planted-divergence red + lake-tracking control — built and its output counted (156,327 rows, equality held) BEFORE this row claimed it | P4b's remaining prose (one-lake, restart-is-not-amnesia) |
| P5 use the GPU where it helps | — | yes |
| **P6 mutual reach + use** | **BUILT AND ASSERTING** — `tools/p6_gate.py --assert` **and** `--self-check` (two steps in the `lint` CI job, both added 2026-08-06; the connectome-diff half is now built too). Ships as a **RATCHET**: five ±1 baselines assert the structure never gets WORSE. Faculty-to-faculty reads moved **0 → 1 at T039h** (neuter-verified, s0-exit.md §8b) and the count-plus-identity floor lives in `tests/test_interoceptive_specialist.py`, which reds if the read disappears or re-resolves to a non-faculty owner. This row said "`faculty-to-faculty > 0` is red today / 0 stands" for three days after the gate measured 1 — the US1 audit's B5; re-derive from the gate, never edit this row from memory | yes (provisional; a ratchet says the count did not worsen, never that P6 is satisfied — one faculty-to-faculty read is a floor, not an arrival) |
| **P7 any dependency that helps** | **BUILT AND ASSERTING** — `tools/dependency_ledger.py --assert` **and** `--self-check` (two steps in the `lint` CI job, both added 2026-08-05; this row read "BUILT, REPORT-ONLY" and the step it described ran the tool BARE, which `main()` makes incapable of a non-zero exit) | yes (the judgement half; see P7 guard (b)) |
| **P8 invent the method** | **BUILT AND ASSERTING** — `tools/red_reachability.py --assert` (step in the `lint` CI job) | yes (whether a criterion measures the RIGHT thing; no instrument can replace it) |
| e-prop not backprop | **PARTIALLY — TWO gates since 2026-08-18: `substrate/eprop_transport_gate.py` (CORE, 2026-08-15)** asserts no GOVERNED module reads a weight transposed into a weight update, and reports the ungoverned tier without asserting on it. It is NOT a `.backward()` tripwire and the argument against building one is unchanged. Two rules — explicit-transpose and INDEX-SWAP (the loop spelling, 2026-08-15); `mlp_q`'s single-row-indexing spelling evades BOTH, printed on every run; **and `substrate/gradient_claim_gate.py` (scope/census)** asserts every governed learning rule is classified as owing a cosine or not, ratcheting MISSING_DECLARATION and UNDETERMINABLE (`--assert` + `--self-check`, CI lint job) | yes (everything the static rule cannot reach: whether a second matrix is a transported weight or an independently-maintained feedback matrix is semantic, and the cosine bar itself) |
| *(not a policy — the safety envelope's own gate)* ONE envelope definition, five brainstem files | `substrate/safety_envelope.py` (bare = assert, in run_selfchecks; envelope EDITS stay approval-gated) | — |
| *(not a policy — repo hygiene)* standing docs keep their word ceilings | `substrate/doc_budget_gate.py` (bare = assert, in run_selfchecks) | — |
| *(not a policy — repo hygiene)* outside data passes license + injection gates before the brain keeps it | `substrate/ingest_guard.py` (bare = assert, in run_selfchecks) | — |
| *(not a policy — repo hygiene)* citations resolve | `tools/citation_guard.py` (step in the `lint` CI job, asserting since 2026-07-30) | — |
| *(not a policy — a spec instrument, invention register row 7)* growth coverage counted before any rule lands | `substrate/growth_coverage_gate.py` (`--assert` + `--self-check`, `lint` CI job, 2026-08-26). The coverage red — 1 of 10 growable regions grown from a live path, cortex only — is REPORTED every run; asserted: the identity-pinned ratchet, scaffold/undeclared refusals, unresolved-site reds. Green never means coverage is adequate | — |
| *(not a policy — repo hygiene)* documented `python -m` commands run | **BUILT AND CLEAN, but NOT IN CI** — `tools/doc_command_guard.py --assert` (2026-08-04) | — (see the correction directly below: this row said "step in the `lint` CI job" and the step does not exist) |

**Corrected 2026-08-04, and the direction of this drift is the one this file keeps warning about.** The
two rows above said *"its gate is specified but NOT built"* while **both gates existed, ran, and were
registered in `.github/workflows/ci.yml`** — P8's `--assert`ing. This is the same failure recorded
under P6 ("a claim that was already false when it was last written"), now with three instances, so it
is worth naming as a pattern rather than an incident: **this file is edited far more often than it is
re-derived, and a "NOT BUILT" is the single cheapest sentence in it to leave standing.** Re-run the
gate before writing a sentence about the gate. Counted output on this tree, **re-run 2026-08-05 in
`.venv-lake`** (every row below was executed, not copied):

| gate | output | in CI? |
|---|---|---|
| `tools/red_reachability.py --assert` | 400 files scanned, 13 criteria; **NO_RED 0, TAUTOLOGICAL 0, READS_SYSTEM 13, EXEMPT 0** — exit 0 | yes, `lint`, `--assert` |
| `tools/dependency_ledger.py --assert` | 400 files scanned, **29** distributions; **UNDECLARED 0, UNUSED 0, MISDESCRIBED 0**, EXEMPT 2, ALL_EXCLUDES 8 — prints `gate OK`, exit 0. **This row said `UNDECLARED 1` / `MISDESCRIBED 1` / `ALL_EXCLUDES 7` and "prints `gate FAIL`, exits 0 (report-only)"; re-run 2026-08-05 in `.venv-lake`, all three counts are 0/0/8 and the tool prints `gate OK`.** The parenthetical *"exits 0 (report-only)"* was the more damaging half and it was not a stale measurement but a structural fact nobody had checked: `main()` returns 1 only under `if do_assert and failed`, so the CI step — which invoked the tool with no flag — could not have reddened the build on ANY finding. Both are fixed together: the flag is added and the counts are re-derived | yes, `lint`, **`--assert`** |
| `tools/dependency_ledger.py --self-check` | 5 MISDESCRIBED criteria mutated against a string copy of `pyproject.toml`: **RED OK 5, NO RED 0, BROKEN 0**; unmutated tree reports 0 findings — `self-check OK: 5/5 criteria proven falsifiable`, exit 0 | yes, `lint` (**registered 2026-08-05**; it had run nowhere, so its own "5/5 proven falsifiable" line was a claim no gate re-derived) |
| `tools/doc_command_guard.py --assert` | 55 documents, 51 `python -m` commands (+2 EXEMPT = 53 matched); **UNRUNNABLE 0** — after 19 broken commands were CLEARED, not exempted. **54 / 50 on a clean clone**: `AGENTS.md` is in the gate's SCOPE and is gitignored, so one in-scope document and one command exist only on a working machine | **NO — see below** |
| `tools/p6_gate.py` | **ASSERTING in CI since 2026-08-06** (`--assert` + `--self-check`, `lint` job); a RATCHET, so green means "no worse", never "P6 satisfied". See P6 for the re-derived reading | yes |
| `tools/p4c_ratchet.py` | **ASSERTING in CI since 2026-08-06** (`--assert` + `--self-check`, `lint` job). Mechanizes P4c's "every future build lands inside the governed tree from the start", which was review-enforced only — and review is what let 259 builds accumulate outside it. Per-ZONE, not on the total: a total-only ratchet lets one zone launder against another (retire 5 in `reasoning/`, spend the headroom on 5 new `tools/` scripts, gate green, tree no better). Baseline is machine-generated by `--update-baseline`, never hand-typed, and that flag REFUSES to write a rise | yes |

**Two corrections to the block above, both found by re-running the gates on 2026-08-05 rather than
by reading what the previous pass wrote about them. Both are the same failure this section names,
committed in the section that names it.**

1. **`tools/doc_command_guard.py` is NOT a step in the `lint` CI job, and the row above said it
   was.** Measured: `.github/workflows/ci.yml`'s `lint` job runs `ruff check .`,
   `ruff format --check src tests tools`, `tools/citation_guard.py`,
   `tools/red_reachability.py --assert` and `tools/dependency_ledger.py` — five checking steps, and
   none of them is this gate. **That enumeration was invalidated later the same day by the pass that
   wrote it**, which added `--assert` to the ledger step and registered
   `tools/dependency_ledger.py --self-check` beside it; RE-DERIVED by parsing the YAML, the `lint`
   job now has **9 steps, 6 of them checking steps** (the other three are `checkout`,
   `setup-python` and `pip install -e ".[dev]"`). The conclusion is unchanged and re-measured —
   `doc_command_guard` is still in none of them — but a count of CI steps written in a pass that
   edits CI goes stale before the pass ends, and this block is where that lesson is supposed to live. Grepping `doc_command_guard` across `.github/workflows/`, `run_selfchecks.py`,
   `.pre-commit-config.yaml` and `pyproject.toml` returns **zero matches**. The gate is real, runs
   clean (`--assert`, exit 0), and carries a self-check whose red is reachable — but nothing runs it
   automatically, so under **P4c** it is *a build the gates cannot see, and therefore not a build
   this project can claim*. Registering it is open work; the CI edit is escalated because
   `.github/**` is outside this pass's write scope.
2. **The dependency ledger reports a THIRD direction the block omitted** — `MISDESCRIBED`. When this
   correction was written its measured line was
   `UNDECLARED: 1   UNUSED: 0   MISDESCRIBED: 1   EXEMPT: 2   ALL_EXCLUDES: 7`, and the live finding
   was *`ALL-EXTRA  extra [nir] is neither in [all] nor in ALL_EXCLUDES -- pip install ".[all]" does
   not install it`*. Recording only the two directions that were zero-or-known turned a finding into
   an absence. **CLOSED, re-run 2026-08-05 in `.venv-lake`:** `nir` is now a member of `[all]` and
   `arcengine` is declared, so the line reads
   `UNDECLARED: 0   UNUSED: 0   MISDESCRIBED: 0   EXEMPT: 2   ALL_EXCLUDES: 8`. The point survives
   the fix and is why this entry is amended rather than deleted: the omission was of a *direction*,
   not of a number, and a report that names only the directions it expects to be clean will hide the
   next one exactly as it hid this one.

**Provenance of the 2026-07 amendments.** Three independent audits mapped this codebase against
external frameworks — predictive coding, the Thousand Brains Theory, and quantum-consciousness
theories. They proposed five policies between them; **one** survived adversarial review, and the
rejections were as instructive as the survivor. Two proposed gates were *implemented before adoption*
and killed by their own output: one scored 0 failures / 47 UNCERTAIN (it could not catch its own
motivating case), the other produced 58 failures where 1 was claimed (adoption would have meant a
58-entry rubber-stamp allowlist). That bar is now itself a rule — see *Adding a policy* below.

What the audits mostly returned was not governance but **defects**, and they clustered into one
recognisable family, recorded under P3.

---

## P0 — The brain must autonomously improve itself, by every available avenue

**This is the project's purpose, and it outranks the rest.** P1–P8 describe how to build well; P0
describes what is being built. The brain must not be a fixed artifact that a human upgrades between
releases — it must get better on its own, continuously, through every channel it can reach: **writing
its own code, acquiring knowledge, and growing its own neural substrate.** Where an avenue of
self-improvement is missing, that is a gap to close, not a boundary to respect.

**Owner directive, 2026-08-04 — the avenue table below is a BUILD LIST, not a status ledger.** Every
avenue named in it, **current and future, will be built.** An avenue that does not fire yet is open
P0 work; it is never a settled state, never a boundary to respect, and never a reason to narrow this
policy to whatever already works. The table exists so that the gaps are *countable*, and a row that
reads NOT LIVE is a scheduled build with its blocker named — which is why each one below states what
specifically is missing rather than only that something is. **The list is also expected to grow:** a
newly identified channel of self-improvement joins it and is built, and "we have not thought of it
yet" is the only honest reason for an avenue to be absent from this table.

This directive changes nothing about the honesty guard further down, and the two are easy to confuse.
"Will be built" is a commitment about *work*; it grants no credit for the capability in advance. The
ordering stays exactly as the guard states it — build the channel, fill it, measure it, and credit it
only at the third step.

**The avenues, and their honest status today.** "By every available avenue" is only meaningful if we
say which are open — so each row states what fires, what does not, and what specifically blocks it:

| avenue | status |
|---|---|
| **self-building code** | **ROUTE FIRES, PAYLOAD EMPTY (measured 2026-08-13).** This row led with a bare **"What fires"** until 2026-08-13, and `specs/010-policy-compliance-audit/findings.md:370` forbids exactly that: it directs **(a)** *"Do NOT promote the SelfBuildLoop route into 'What fires'. State it honestly"* **and (b)** *"Give the measurement beside the claim"*. **Both halves are now applied** — (b) first, and (a) one commit later in the same lane, after an audit caught that the first pass had quoted only (b) and written that `:370` "was not applied until now" while the forbidden heading was still leading the row. A half-quoted directive reads as a discharged one. Route re-derived 2026-08-04 and cited by SYMBOL, because the line cites in this row had drifted AGAIN within one day of last being "corrected": `cortex/brain.py::Brain._maybe_autonomous_self_build` -> `basal_ganglia/self_build_loop.py`, every 8 nights, **only** under `NORD_SELFMOD=1` (enabled in `learn_daemon.bat`), plus `basal_ganglia/brain_native_selfbuild.py`. **MEASURED 2026-08-13, and the measurement is the whole point: the cadence FIRES and the payload is EMPTY.** 190 ticks of `make_brain(seed=0)` under `NORD_SELFMOD=1` **and `NORD_HOME` pointed at a scratch directory** (without it the run appends to the operator's LIVE lake -- two auditors hit exactly that while checking this row; `tests/conftest.py` already defends the suite, this prose copy was the undefended one) leave `Brain._selfbuild_last == {'night': 8, 'proposed': 0, 'kept': 0}` — the night-8 cadence triggers exactly once, the loop is entered, and it proposes **ZERO** candidates, so `SelfModHarness` is never constructed and no self-modification is ever attempted. (`git diff --stat` is empty after the run: nothing writes to the tree, because the per-proposal body is never entered.) This row has said "What fires" with no number beside it, which is exactly the shape the honesty guard ten lines below forbids — *"a capability reported without measurement is fake"*. It fires; it does nothing. (`findings.md:370`'s half (b); its half (a) is discharged by this row's new heading, above.) **It also re-sizes the `godel_machine` clause later in this row**: wiring `attempt_on_repo` into a loop that proposes nothing would satisfy a call-site grep and change no outcome, which is the force-wire this policy's own guard forbids. Filling the proposer is the open P0 work here, and it is a different and larger job than adding a call site. **`self_mod_harness.py` was listed here as reached through `Brain.self_improve` — it is NOT. `cortex/brain.py::Brain.self_improve` has ZERO call sites tree-wide** (re-verified 2026-08-04: the only tree-wide match for the name is its own `def`). An entrypoint with no callers listed under "What fires" is the census-counted-as-capability failure this row corrects for `godel_machine` one clause later, committed in the same sentence. **What does NOT fire:** `reasoning/godel_machine.py` — its `attempt_on_repo` has **zero call sites** tree-wide, confirmed against `importlib` / `__import__` / getattr and string-name dispatch rather than from a static read alone (re-counted 2026-08-05 over `src/`, `tests/`, `tools/`: **four** matching lines, all non-calls — its own `def`, **two** lines of its own docstring, and a docstring in `tests/test_self_mod_safety.py`; the previous pass wrote "three matches" and the enumeration it gave was right in kind and one short in count). The module IS imported live by `substrate/self_mod_harness.py` and registered in `run_selfchecks.py`, so "the only reference outside itself is its filename as a string" does not hold. It is imported-but-never-invoked — P3 shape 1, not absence — which is a different defect and a shorter fix. Wiring both is open P0 work. **PATHS CORRECTED 2026-08-04:** this row said `reasoning/self_mod_harness.py` and an undirectoried `self_build_loop.py`; both had been PROMOTED out of `reasoning/` (to `substrate/` and `basal_ganglia/` respectively) by the P4c program, and the row followed neither |
| **knowledge acquisition** | live — chat ingest → the one durable lake (`substrate/chat_memory.py`), `substrate/web_ingest.py`, reasoning-curriculum growth during sleep |
| **neural growth** | **RETENTION IS BUILT — this row asserted the opposite until 2026-08-04 and was wrong on every one of its three load-bearing claims.** Growth runs **only during sleep**, and what it grows is now KEPT — **cited by SYMBOL, 2026-08-05, because every line cite in this clause had rotted within a day of being written**: `cortex/neurogenesis_engine.py` assigns **`self._model = model`** at the end of a successful refit, seeds each refit from the prior model instead of starting cold (the `EXTENDS it instead of replacing it` block), and prunes on earn-your-place (the `EARN-YOUR-PLACE PRUNE` block). The serialization the row once said did not exist is `growth_state_bytes()` / `restore_growth_state()` / `substrate_is_retained()`, and it is **CALLED on two live paths** — the sleep tick and construction in `cortex/brain.py` — writing into the ONE lake under the growth tags (`substrate/home.py`: `_T_GROWTH_DELTA` chain records since specs/022 T010-ii; `_T_GROWTH` holds the legacy whole-blob snapshots and still takes raw non-npz blobs), which is P4b-correct rather than a sidecar. *(Cites into `cortex/brain.py` and `home.py`'s trailing line numbers were DROPPED 2026-08-04: `brain.py:3780` was `except Exception:`, `:2722` was blank, and `home.py:231`/`:280` were `return path` and a docstring. **The `neurogenesis_engine.py` cites were KEPT in that pass with the words "all three re-verified exact" — and re-resolved on 2026-08-05 they were not.** Measured: `:211` is a comment (`self._model = model` is at `:215`), `:250` is a comment (`def growth_state_bytes` is at `:254`), `:279` is `ticks=np.array(...)` (`def restore_growth_state` is at `:283`), `:321` is `self._acc = acc` (`def substrate_is_retained` is at `:325`), and `:180-183`/`:189` land on the explanatory comments above the code they name. Every one is off by exactly four lines, because a concurrent edit inserted four lines above them — **which is the point**: the assertion "re-verified exact" was true when written and false the same day, so the durable form is the symbol, and this row now uses it. The companion cite `substrate/similarity_sized_growth.py:78` was likewise off — `self.W` is at `:81` — and is named by symbol below.)* **What this row got wrong, recorded because the failure mode is the point:** it claimed "retains NOTHING", "never assigned to `self` (verified: zero matches for `self.*model*=`)" and "no `save`/`load` at all". The grep was the instrument and the instrument was wrong — the assignment is `self._model = model`, which `self.*model*=` does not match. **The file contradicted itself 18 lines later**, at the honesty-guard bullet that cites `neurogenesis_engine.substrate_is_retained()` by name as the site naming its own gap; a row and its own guard disagreeing is the drift this file keeps suffering. **`substrate/similarity_sized_growth.py` DOES retain** (`self.W`, allocated in `__init__` beside the `self._W0` init snapshot that exists to PROVE the weights moved). `hippocampus/{growing_learner,function_preserving_growth}.py` **UNVERIFIED** — a narrow attribute scan found no retained weight array, which is not the same as proving there is none. **THE REMAINING GAP IS NARROWER THAN THIS ROW CLAIMED, corrected 2026-08-16.** It read *"no cross-process restart has been demonstrated… demonstrating restart-with-grown-substrate is the open P0 work"*. **A cross-process restart IS demonstrated, at the ENGINE level:** `neurogenesis_engine._cross_process_restart` genuinely spawns a `subprocess`, is driven by the module's own self-check, and its child half runs under the `__main__` guard. Stating otherwise pointed P0 effort at work that shipped. **THE BRAIN-LEVEL RESTART IS NOW DEMONSTRATED TOO, AND THE SUSPECTED FLUSH WAS REAL — specs/019 `[B2.09]` CLOSED 2026-08-16 as a defect fix, which is what this row said it would be if the flush was real.** `cortex/brain.py::close()` never flushed growth, so growth reached the lake ONLY on the sleep cadence, and that save LAGS the growth that caused it. Measured, seed 7, two processes: a brain ticked until `grown_units` rose and then exited kept **0 of 2 units and wrote ZERO `_T_GROWTH` records**; one given 200 further ticks exited holding 3 and restored **5**, because the last cadence save had caught a PRE-PRUNE snapshot. The lake held a DIFFERENT brain than the one that exited, in both directions at once — too little and too much — which is P4b's *"a restart is not amnesia"* failing while every existing check stayed green. Fixed by `cortex/brain.py::Brain._flush_growth`: ONE mechanism with TWO callers (the sleep tick and `close()`), keeping the `substrate_is_retained()` gate rather than `grown_units > 0`, because those two came apart for the whole life of this feature. Proven by `tests/test_brain_growth_restart.py`, which spawns a real interpreter over the same lake. **The lesson outlives the fix and is why this row is worth its length:** the first restart test, at 260 ticks, PASSED AGAINST THE LIVE DEFECT — over that many ticks a save always lands after the last growth, so exit-state and save-state agree BY LUCK. `test_growth_survives_an_immediate_exit` is the one carrying the reachable red. A test can pin exactly the right property and still be satisfied by a coincidence, and only a constructed red separates the two |
| **cortical column construction** | **NOT live — but NOT for the reason this row gave until 2026-08-16.** `cortex/cortical_microcircuit.py` is a faithful 8-population Potjans-Diesmann circuit, EXEMPT-permanent under the `"cortical_microcircuit"` key in `substrate/dormancy_gate.py` `EXEMPT`. **This row said *"it has no input port… building that port is the work"*. That is FALSE, and had been since 2026-08-04.** `CorticalMicrocircuit.thalamic_drive(thalamic_hz)` IS the port — its own docstring calls it *"THE INPUT PORT… the signal a caller injects"* — and `simulate(t_ms, settle_ms, thalamic_hz)` carries a caller's drive into it. It is **anatomically targeted**: `CONN_PROB_TH` is nonzero for **L4e/L4i/L6e/L6i only**, the real thalamocortical projection, so a caller cannot inject into L2/3 or L5 and the stimulus must travel the intracortical route. It contributes **exactly 0.0 at 0 Hz** (an undriven run stays bit-identical to the pre-port module) and **refuses a negative rate**. **The TRUE reason it is not live: NOTHING DRIVES IT** — measured 2026-08-16, zero references to `thalamic_drive` or `CorticalMicrocircuit` anywhere outside its own module. That is P3 dormancy, not an absent capability, and it resizes this avenue from *design and build a port* to *wire a caller to an interface that already exists and self-checks*. **What IS still absent, stated precisely so the next reader is not sent to build the wrong thing:** the port carries a **scalar rate**, so a caller may modulate drive INTENSITY but cannot inject a spatial PATTERN. `specs/011` Phase A is scoped as *"the layered circuit gets an input port"* against a port that exists; a patterned input is the honest remaining work there. Wiring a live driver is specs/019 `[B2.08]`, and the exemption is retired the day it lands. *(Cited by SYMBOL: this row's line cite has now been wrong twice — `dormancy_gate.py:83` pointed at a comment about a deleted `reaction_diffusion_compute`, and its 2026-08-04 replacement `:100-104` has since rotted onto `gwt_deep`.)* |
| **self-recalibration** | live — `substrate/signal_calibrator.py`, running-baseline forms per P1 |
| **future avenues** | **open by construction.** This row is not a placeholder — it is the directive above made structural. A channel of self-improvement that is identified later joins this table and is built, on the same terms as the rows above it: named blocker, then channel, then fill, then measurement. The five rows above are what has been *thought of*, not what is *possible*, and treating the table as complete is the failure this row exists to prevent |

**Honesty guard — growth is not improvement, and this is where P0 would otherwise rot.** Adding
neurons, adding code, adding knowledge records: none of these is improvement until something
*measurably gets better*. A self-improving system that reports its own progress is the single easiest
place in this project to generate theater, and every other policy exists partly to stop that:

- a grown unit that changes no outcome is **dormant** (P3), not progress;
- **Scaffolding a capability the project does not have yet is PERMITTED, and is not a fake capability
  — owner decision, 2026-07-30.** This guard forbids *reporting* a capability without a measurement. It
  does not forbid *building the channel* one will need. Those are different acts and this policy had
  been read as conflating them: durable growth cannot be measured until growth is durable, so demanding
  the measurement first makes the work unreachable. The conditions that keep it honest: the
  infrastructure is labelled as infrastructure, no coverage band or P0 avenue may claim the capability
  until it is measured, and the gap it exists to fill is named at the site — as
  `neurogenesis_engine.substrate_is_retained()` names it, and as the ordering below enforces.
  **Retention landed before the durable record type for exactly this reason**: persisting
  `grown_units` while the model was discarded each refit would have written a counter and restored an
  empty engine, which *would* have been a fake capability. Build the channel, then fill it, then
  measure it — and do not credit it until the third step.
- **a grown unit that is not KEPT is not even growth** — added 2026-07-30, because the avenue table
  above claimed neural growth was "live" while its main engine discarded every model it fit. This guard
  had a gap the whole time: it tested whether growth *changes an outcome*, and never whether growth
  *accumulates*. A measurement loop that reports rising accuracy as units grow is a legitimate and
  useful thing, but it is an experiment about growth, not the growth itself. Note the honest
  possibility, which is open work rather than a defect: retaining nothing may be CORRECT for a
  measurement loop — in which case what needs fixing is the claim, not the engine, and this row is that
  fix. Deciding whether the engine *should* keep its model is a design question about what growing
  means here, and `hippocampus/ng_snn.py`'s earn-your-place stop (capacity monotone in data, 4,512
  units at 24,000 rows) is the template for a grower whose result is worth keeping;
- a self-written module that duplicates a mechanism is **bloat** (P2), not capability;
- a "capability" reported without a measurement is a **fake capability** (verify-first), not a gain;
- the census of what the brain has is **not** a measure of what it can do — count improvements by
  their effect, never by their number;
- **and a measurement that cannot fail is not a measurement** — added 2026-08-04 with P8, because
  this guard said "a capability reported without a measurement is fake" and never said what makes
  something a measurement. It is the harder half: the tree's live `consciousness_probe` reports 4 of
  4 through markers with no reachable red, which satisfies the guard as previously written and defeats
  its entire purpose. See P8 for the burden and `PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:57-58` for the
  line it derives from.

Real example of the failure mode: predictive-coding *inference* ran on every tick for the life of the
feature while its *learning* method was never called (v4.71). The brain had the machinery, reported the
signal, and improved at nothing.

**Safety is what makes aggressive self-improvement possible, not what limits it.** The self-modification
path is sandbox-verified, arm-safety firewalled, and rolled back byte-exact on any gate failure —
which is precisely why it can be left ON. A self-improving system that cannot be rolled back is one
that gets switched off after its first bad night, and then it improves at zero. The safety envelope
(five brainstem files, E-STOP never relaxed) is **not** an avenue of self-improvement and is never a
target of it: the brain may rewrite what it thinks, never what stops it. That boundary is what buys
the freedom in the rest of this policy.

---

## P1 — Brain-native / no-hardcode

Every behaviour-arbitrating value must be a **learned weight** or a **running-baseline-relative** form
(three-factor R-STDP / e-prop; running EMA or `SignalCalibrator` quantile), not a
magic constant. Corollary: **never keep a constant fixed** — when a hardcode resists replacement, the
blocking constraint is usually the bug, not the number.

**Honesty guard (defeasible only by gate evidence):** a constant may stay fixed in exactly three proven
cases — (a) it is a fixed point on a **normalized** output (e.g. a threshold on a `tanh` confidence in
[-1,1] is dimensionless, like `p ≥ 0.3`); (b) a running-baseline replacement is **not reproducible from
the persisted policy** (e.g. a durable-reload contract persists weights but not the running baseline);
(c) the constant is **inert** (default-off, zero live callers) so a learned form would never charge.
Reuse anchors: `basal_ganglia/r_stdp.py` (`w += lr*d*e`), `substrate/signal_calibrator.py`, the
`_*_base` running-EMA idiom, `cortex/brain.py::_drive_w`.

**WORKED CASE, 2026-08-07 — a hardcode retired by discovering it was already a derivation, and a
second one that could NOT be retired by tuning.** Both are in `thalamus/global_workspace.py` and
together they are what the corollary above means in practice.

- **`_G_INH = 2.4`, the "TOTAL lateral-inhibition budget", was `_W_SELF * capacity`** — an EXACT
  float identity (`2.4 / 4 == 0.6 == _W_SELF`), i.e. the per-slot excitation/inhibition balance
  multiplied back up by the capacity it was about to be divided by again. The constant was the round
  trip written down as a number. Retired; the shipped workspace is byte-identical without it, and a
  workspace at any OTHER capacity now stays E/I balanced instead of inheriting one capacity's
  arithmetic. **The tell was not the value, it was that the value only ever appeared divided.**
- **The incumbent's defence could not be fixed by any value, and that is the corollary's other
  half.** `self_boost + hysteresis = 0.25` exceeded the 0.20-wide band it operated in, so no
  compliant specialist could take the workspace head from any other. Displaceability needs
  `defence < (1 - lo)/2`; the same quantity is what `capacity = floor(1/defence)` divides, so every
  value satisfying it widens the workspace past the point where the bottleneck binds. Swept: the
  admissible window `(0.2000, 0.2250)` costs churn 9 → 15 and retention 47/47 → 45/47 at EVERY value
  in it, and the ignition-threshold axis has no admissible value at all. **Churn-clean and
  displaceable are disjoint.** (The sweep's baselines are the pre-inhibitory-rheobase kernel's, its
  own date; the same day's later change gave the shared inhibitory pool a rheobase and the stream's
  churn baseline is 4 since — the window argument stands, its absolute figures are that morning's.) The blocking constraint was the bug exactly as this policy says: ONE subsidy was doing
  two jobs (sustaining an assembly, and deciding which assembly is broadcast), which want opposite
  magnitudes. Splitting them cost nothing on either side. A constant that resists every value is not
  a tuning problem, and sweeping it is how you prove that rather than assume it.

**Fourth case, stated because the tree contains it and the three above do not cover it: raw physical
limits inside the safety envelope stay fixed, by design.** `brainstem/arm_config.py`'s `CONTACT`
(`= 0.04`) and `brainstem/reflex.py`'s `DEFAULT_RULES` arbitrate behaviour and are constants
— which the three exceptions would flag. They are correct anyway: a reflex threshold that drifts with a
running baseline is a reflex that can be trained away, and the whole point of the envelope is that it
cannot. The resolution currently lives only in code (`arm_config.SAFETY_CALIBRATED` is `False` and
`serial_arm.assert_calibrated()` raises on every real-servo entry point while it stays False), so it is
written here too. Safety constants are exempt from P1 *because* they are safety constants, not
because nobody noticed them.

*(Citations in this paragraph converted from lines to SYMBOLS, 2026-08-04, after all three drifted:
`arm_config.py:22` resolved to `NEAR_MAX = 0.30` (`CONTACT` is at `:27`), `arm_config.py:29` resolved
to a comment about the trajectory-deviation reflex, and `serial_arm.py:46-52` resolved to the
`_FIXED` / `_ALLOWED_WHEN_STOPPED` token sets — not the calibration refusal, which is
`assert_calibrated()` some ten lines below. `reflex.py:22` was the one that still resolved. **Three of
four wrong, in the paragraph that tells a reader which safety constants are deliberately fixed** —
which is the argument for symbols over lines in the envelope's documentation generally, and the same
argument `SECURITY.md` acted on in the same pass.)*

## P2 — Adopt-better **or merge**, then retire the inferior (non-additive)

When two mechanisms do the same job, exactly **one** survives. There are two legitimate routes to that,
and they are co-equal — the choice between them is a judgement about the mechanisms, not a fallback:

1. **ADOPT** — one is better on evidence; it becomes THE path and the other is **deleted**.
2. **MERGE** — neither dominates, or each holds something the other needs; build the **one** mechanism
   that supersedes both, then retire both originals. A merge is not a compromise or a coexistence; it
   ends with fewer moving parts than it started with.

What is forbidden is the third option: leaving both live "additively". That dilutes the signal, grows
bloat, and leaves every later reader guessing which one is authoritative.

**Worked merge:** `cortex/neuron_zoo.py` — the repo's neuron models each had their own constructor and
step signature, so a population was always homogeneous. The zoo is one heterogeneous population behind
a single contract, and it earns the name because it *"does NOT re-implement any dynamics -- it IMPORTS
the existing neuron modules and puts a thin adapter in front of each"*. One surviving interface, zero
duplicated dynamics. Smaller-scale: `PredictiveCodingNetwork.learn()` now calls `learn_step()` (v4.71),
so the offline trainer and the live tick apply the identical rule — one definition, not two.

**Honesty guard:** prove the adopt-or-merge is behaviour-equivalent (or superior) **before** deleting —
an A/B on identical inputs, gate-verified no regression, and any unique capability of the retired
mechanism ported first. Reject it with evidence if it regresses (verify-first works when it says *stop*).

**Merge-specific guard — a merge must actually reduce.** "Merging" that keeps both code paths behind a
flag, a mode switch, or an `if` is additive with extra steps, and is the failure this policy exists to
prevent wearing a more respectable name. Count the survivors: if two implementations of the same job are
still reachable when you are done, it was not a merge.

**"Better" requires a measurement that exists.** If the two mechanisms cannot be compared by any
benchmark in this repo, the swap is not P2-eligible and the answer is *build the benchmark first*, not
*pick the more fashionable one*. Worked case: the Thousand Brains audit proposed replacing the Global
Workspace (`thalamus/global_workspace.py`, one broadcast slot, winner-take-all) with peer-to-peer
columnar voting. Both are serious theories of the same phenomenon; SNDL has **no consensus benchmark**
— no ambiguous multi-source evidence with a known right answer — so neither could be shown better, and
the swap was refused. An external framework being newer, or more elegant, or better publicised is not
evidence. This guard is what stops P2 from becoming a fashion treadmill.

**How this guard interacts with P7, stated because the two are easy to set against each other.** P7
accepts any dependency that helps, and this guard demands a measurement before a swap — they are not
in conflict, and reading them as opposed is the mistake to avoid. P7 removes exactly one argument
from the table: *"it would be another dependency."* It removes nothing else. So a framework proposed
as a replacement for something this tree already has is a **P2 event governed by P2's evidence bar**,
and if no benchmark can compare them the answer is still *build the benchmark first*. Conversely, and
this is the direction the tree actually fails in, "we have no benchmark" is not a licence to decline
on dependency grounds — it is a reason to build the benchmark. And where a library covers a capability
the tree does **not** have, there is no incumbent to compare against, so P2 does not gate it at all
and P7 governs alone.

**OPEN P2 DEBT, measured 2026-07-28: two workspaces run additively.** `thalamus/global_workspace.py`
and `thalamus/reentrant_workspace.py` are both live and both fed every tick. Measured over 40 wake
ticks on a real brain: **72 submits each**, with 3 content values reaching both. Their public surfaces
are near-identical — `submit(source, content, salience)`, `subscribe`, `current`, `step` on both — and
`ReentrantWorkspace` adds `broadcast_to`, `association`, `access_strength`, `inbox`. On interface
alone it is a superset. That is exactly the shape this policy says must resolve to one survivor.

It is **not** resolved here, for the reason this policy already gives: *"better" requires a measurement
that exists*, and SNDL still has no consensus benchmark — no ambiguous multi-source evidence with a
known right answer — so neither can be shown better. Retiring `GlobalWorkspace` on interface elegance
alone would be the fashion-treadmill move this guard exists to stop, and it has **8 importers**
(`access_consciousness`, `executive_control`, `consciousness_probe`, `concept_reflection`,
`autonomous_loop`, …) against ReentrantWorkspace's 3.

What DID get fixed is a defect that had to be cleared before any comparison could be fair: the
`AccessConsciousness` wrapper was **constructed inside the wake tick**, so a fresh `GlobalWorkspace`
was built every tick — measured **25 constructions over 40 ticks** against ReentrantWorkspace's **1**.
Every instance took exactly one step, making incumbent defence, hysteresis and decay structurally
unreachable, while a comment eighteen lines below promised "ignited content is HELD across ticks, not
rebuilt from scratch each tick". Now persisted: **25 → 2** constructions, `access_index` still varying
across 25 distinct values. Any future A/B between the two workspaces would previously have compared a
persistent mechanism against one that was destroyed before it could express itself.

**The benchmark that would settle it**, stated so the debt is actionable rather than perpetual: a set
of ambiguous multi-source inputs with a known correct winner, scored on which workspace's ignited
content matches it. Until that exists, both stay and this entry is the honest record of why.

**Corollary — a research direction is not a policy.** "Adopt architecture X" governs *what* we build;
policies govern *how*. Proposals of that shape belong in a design note, not in this file.

**This corollary is what P8 had to clear, and the distinction is worth keeping sharp.** "Invent any
method required to reach sentience" could be read as a research direction and would then belong in a
design note. It is not one: P8 names no architecture, no theory and no mechanism — it grants latitude
and fixes what an invented method owes (a reachable red, a control that is not itself, deletion on a
null). That is a *how*. The test to apply to any future proposal of this shape: if removing every
proper noun from it leaves nothing, it was a research direction.

## P3 — No dormancy

**Every part of the brain must never remain dormant; all present AND future builds must be wired and
utilized appropriately.** A faculty is not "done" until it is one of three dispositions:

**THE REGION-CHARGE RULE (specs/022 T022, landing WITH its counted gate per the
gate-before-rule invariant).** At REGION granularity, "charged" is a conjunction:
REACHABLE (the dormancy gate's one walk, reused — never re-derived) AND ≥1 wiring-manifest
port edge touching the region carrying a MEASURED charge attribution (a replay-twice
divergence from `substrate/port_charge_probe`, recorded in
`tools/region_charge_evidence.json` — never a declaration). **Registration alone never
greens a region.** Gate: `substrate/region_charge_gate.py` — REPORT-ONLY until the T039
exit certification writes the integer floor into `tools/region_charge_floor.json` (FR-015's
blank); its exemption rows carry {region, reason, owner, date, expiry} with a ratcheting
count ceiling (0 at introduction; raising it is an owner-visible admission). Counted at
landing, 2026-08-20: **0 of 16 anatomical regions CHARGED; 13 REACHABLE-UNCHARGED;
3 UNREACHABLE** (insula deliberately unwired until its US1 port; olfactory and pons own no
modules yet). The 13-vs-0 gap is the rule's whole message: a tree can be almost fully
reachable and zero-charged, which is exactly why reachability was never charge.
**UPDATED 2026-08-22 (specs/022 T039f): the count above is the LANDING census and is kept as
that record; it is no longer the current one. Insula is now REACHABLE AND CHARGED -- its US1
port landed, `cortex/brain.py` feeds the terminus the declared `body_in` frame and its
`feeling_salience` joins the ignition law -- so the census reads 8 of 16 CHARGED and the
UNREACHABLE set is olfactory and pons, both of which still own no modules. The floor artifact
carries the identity: `tools/region_charge_floor.json` is 8 with `named_and_NOT_charged`
empty. Re-derive with `region_charge_gate --assert` rather than quoting either number.**

**THE ATLAS-NODE RULE (specs/022 T024, landing WITH its counted gate).** Beneath charge
sits NAMING: every `connectome.NODES` id must be named by ≥1 module in its own region
(quoted id or anatomical prose — the reference rule and its two observed false-positive
shapes are documented in the gate) or carry a dated EXEMPT row. A connectome node no code
names is anatomy on paper only — the schematic growing structures the tree never
implements. Gate: `substrate/atlas_node_gate.py`, BINDING from day one: a NEW phantom or a
stale exemption reds the build; the recorded day-one rows may only be deleted, never added
to (the ratchet). Counted at landing, 2026-08-20: **18 of 41 region-owned nodes resolved;
23 day-one phantoms recorded** (including the `cb_ctx`/`pn` pair the spec predicted;
effector nodes are outside every region by design and outside the denominator). Distinct
from the region-charge rule the way P3 is distinct from P6: charge asks whether a wired
edge measurably participates; this asks whether the declared anatomy is even named — the
necessary condition below it.

1. **wired + charged** — imported by a live path AND its output routes into behaviour (decision / value
   / learning / reply), proven to change an outcome;
2. **retired** — superseded / no honest consumer → deleted (this is P2 applied to dormancy);
3. **exempt** — legitimately not on the live tick, listed with a reason (train-tier trainer, benchmark
   harness, entrypoint-only tool, or awaiting a prerequisite such as a real sensor).

**Mechanized:** `substrate/dormancy_gate.py` (in `run_selfchecks` CORE) builds the static import graph
and asserts every REGIONS/TORCH_TIER module is **reachable from a live root/entrypoint OR in the EXEMPT
allowlist** — a new build that no live path imports and that is not exempted FAILS CI.

**Also mechanized: the CROSS-TIER edge, `substrate/import_manifest.py`.** `BRAIN_MODULES` is built from
REGIONS + TORCH_TIER, so `reasoning/` (172 modules) and `training/` are outside the gate entirely — and
they are also ruff-excluded and omitted from the coverage denominator. The governed tiers nonetheless
reach into them through **lazy imports inside bare `except Exception:` swallows**. **Re-counted
2026-07-29, with the definition stated, because the old figure was unreproducible and so was its
correction:** `cortex/brain.py` has **111** bare `except Exception:` handlers (no `as name`), **39** of
which wrap a `try` body containing an import; across the 245 governed-tier files it is **142** bare and
**49** import-wrapping. This text said "67 such swallows in `cortex/brain.py` alone" and the audit that
flagged it said "112 / 203" — neither states what it counted, which is why neither reproduces. The
counter is `ast`-based and stated here so the next reader can re-derive it rather than inherit it. Delete or break one of those targets and every gate stays green while the
brain sets the faculty to `None` and ticks on without it — including `self_build_loop` and
`self_mod_harness`, which *are* P0's self-building-code avenue, and `rose_language`, which is language.
The canary derives the manifest by AST walk (never hand-maintained — the `_external_roots()` lesson) and
asserts each target still imports. Re-derive the census with
`python -m snn_data_lake.substrate.import_manifest` rather than quoting it from here — this sentence
carried "15 modules / 17 call sites" while the gate printed 7 / 9 (US1 audit, M7). It proves
import-ability, not charge; that is deliberate, because
import-ability is precisely what a deletion breaks and what the swallows hide.

**Honesty guard (the crux):** reachability proves a module is *imported*, NOT that its output changes
behaviour. "Utilized **appropriately**" (does the wire actually *charge*) is a second, per-faculty tier,
proven by the runtime **neuter-A/B probe** (`tools/charge_probe.py` → "BEHAVIOURAL CHARGE OK"
vs "CONSUMED-BUT-NOT-DECISIVE") during a wiring wave — too expensive for CI. So "never dormant" must NOT
become "force-wire everything": a wire that does not charge is **reverted and retired/exempted**, never
faked with a dead consumer. (History: a coverage-audit once flagged PFC / amygdala / cortical-column /
synfire as dormant; every one was REFUTED on runtime A/B. Static grep proves import-absence, not
behavioural inertness — mark unproven cases UNCERTAIN, not dormant.)

### The dormancy family — seven observed shapes of "live but inert"

The module gate asks one question: *is this module imported from a live root?* Every defect below
answered **yes** and was still dead. They are the same failure wearing different clothes, and naming
them is most of what makes the next one findable. All seven are real, all were found by audit rather
than CI, and all are cited from this tree:

| # | shape | worked example |
|---|---|---|
| 1 | **method never called** — the module is live, one of its methods is not | `PredictiveCodingNetwork.learn()` — inference ran every tick, learning had one caller: its own `__main__`. Fixed v4.71 |
| 2 | **return value discarded** — a live call whose result goes nowhere | `connectome_router.step()`; its only export reached telemetry alone, measured constant 0.250 over 122 ticks. **SINCE FIXED** (`51bf5253`, v4.80): the return is consumed at `cortex/brain.py:2010`. The old cite `:2001` had also drifted onto an unrelated line — corrected 2026-07-29, because this table is the reference readers copy from |
| 3 | **key-space mismatch** — a learned path structurally unreachable | `thalamus/reentrant_workspace.py` potentiates on `(source, winner)` at `:271`, looks up `(subscriber, winner)` at `:289`; disjoint live, so the learned re-route took 2 values in 122 ticks |
| 4 | **write-only consumer** — a faculty fed every tick, never read | `cortex/agency_monitor.py` — `observe()` runs live; nothing reads `agency()` / `controllability()` / `status()`, and no key reaches telemetry |
| 5 | **constant wearing a signal's clothes** — present, plausible, never varies | `spikes.midbrain` (its novelty term is never enabled); `integration_competence` (assigned only in `Brain.__init__`); the `gated` trio (never *computed*, not merely never applied) |
| 6 | **silent default** — a lookup table missing the cases that matter | `_ALLO_MOVE` had no entry for the four `JOINT_*_ROTATE` tokens, so `.get(action, (0.0, 0.0))` meant the commands that actually move the arm contributed **zero** to path integration. Fixed v4.72 |
| 7 | **the registry nobody registered with** — a plug point, called every cycle, with zero plugs | `substrate/region_wiring.py` is *the* documented per-cycle faculty-charge registry, and its docstring asserted that "the daemon imports `wired_faculties` (which registers every adapter) at boot". **There is no `wired_faculties` module** (re-verified 2026-08-04: zero files of that name anywhere in the tree), `substrate/learn_daemon.py` imports only `region_wiring`, and nothing anywhere calls `wire()` outside the self-check — which clears `_WIRED` first, so it proves the mechanism and says nothing about production. `charge_all()` iterates `_WIRED`, which is initialised `[]` and never appended to on a live path, and the daemon logs `adapters={cov['n_wired_faculties']}` = 0. Docstring corrected; **the registry is still empty as of 2026-08-04, and filling it is open P3 work.** *(Path corrected: this said `learn_daemon.py:291`, and the root `learn_daemon.py` is now a 32-line shim — the daemon was promoted to `substrate/learn_daemon.py` under P4c and this row, like five other documents, kept citing line numbers into a file that no longer has them.)* |

**The transferable rule:** *presence is not participation.* Being imported, being called, being in the
snapshot, and having a value are four different things, and none of them is *changing an outcome*. When
auditing a faculty, name which of the four you verified — and shapes 5 and 6 are why
`substrate/brain_telemetry.py::PROVENANCE` exists: a signal that cannot vary must say so beside its own
number, or a dashboard turns it into a claim.

### P3 amendment — dormancy is per-METHOD, not per-module

**If a module on a live path computes a prediction error, free energy, or any other learning signal,
the update rule that consumes it must also be called from a live path.** A learning-verb method on a
reachable module with no live-path call site is dormant, and must be wired, retired, or listed in
`EXEMPT_METHOD` with a disposition and a reason.

**Why module granularity is not enough — a worked case, since fixed.** `dormancy_gate.py:71` keys on
the module basename, so a module passes P3 clean with half its capability dead.

`cortex/predictive_coding.py` was exactly that. The brain settled its 3-level Rao-Ballard net on every
wake tick (`cortex/live_brain.py:161`) with the error-neuron architecture implemented faithfully — while
`PredictiveCodingNetwork.learn()` had **one** call site in the tree, inside the module's own `__main__`
self-check. The live path called `reset()` + `infer()` only, against weights seeded random at
construction that never changed (verified: `w1` was `array_equal` before and after 40 further wake
ticks). Inference wired, learning never called, and the module-level gate saw a live, imported,
fully-compliant module.

**That case is now closed** (v4.71): the per-sample rule was extracted as `learn_step()`
(`predictive_coding.py:244`) and is called every tick at `cortex/live_brain.py:185`. Measured over 400
ticks — prediction-error energy −67.8% with the call, +9.1% without it (weights frozen).

Fixing it does not retract the argument; it completes it. The blind spot is no longer hypothetical: a
half-dead module demonstrably passed the module gate, for as long as that feature had existed, and was
found by an audit rather than by CI. That is the evidence this amendment rests on.

**RE-ADJUDICATED 2026-07-31 at RUNTIME** (`specs/001-charge-the-dormant`, and see
`docs/coverage/README.md` for the per-rule evidence). Seventeen learning-verb methods were resolved by
following the RECEIVER and instrumenting real tick runs, then adversarially reviewed: 8 LIVE with
charge proven, 6 never called on a live path, 3 called on the right receiver and changing nothing.
Dispositions: 2 permanent exemptions (dormant AND correct to be dormant), 4 OPEN exemptions each
carrying a disposition and a port finding, 3 measured inert. **Nothing was wired or retired** — every
port check found either a live class or an unported capability, and P2 forbids deleting either.
Two instrument defects were fixed en route: exemption keys are now class-granular (`module.Class.method`;
`n_defs` 56→58, because two same-named defs in one module had been overwriting each other and the
finding was reported against the wrong class), and `charge_probe` gained a graded observable after its
argmax flip counts were shown to report live faculties as inert.

The candidate set
below was produced by the import-scoped pass described next and has NOT been re-adjudicated since the
predictive-coding fix — treat it as a starting list to verify, not a verdict.

**Mechanization: BUILT — `substrate/method_dormancy_gate.py`, in `run_selfchecks` CORE.** A pure
static AST pass (~1 s, no runtime cost). Re-derive the census with
`python -m snn_data_lake.substrate.method_dormancy_gate` rather than quoting a count from here —
this paragraph carried "56 scanned / 16 UNCERTAIN / 5 exemptions, one standing TODO" for weeks
while the gate printed 66 / 20 / 9 with three standing TODOs (US1 audit, M7). What made it
adoptable is the SHAPE, which is stable: dormant asserts at 0, an earlier draft produced
59 failures — the rubber-stamp shape that killed a different proposal above.

**Three details are load-bearing, each found by implementing the naive version first and watching it
fail.** The first two were predicted when this section was written; the third and the correction to
the second were found by building it:

1. **Import-scoped resolution.** Matching bare `ast.Attribute` call names across all reachable modules
   scores 0 failures / 47 UNCERTAIN — the name `learn` resolves in 20 unrelated modules, so it cannot
   even catch its own motivating case. A call site counts only if its module imports the defining
   module.
2. **Same-module callers COUNT, except under `if __name__ == "__main__"`.** Excluding the defining
   module produces a real false positive (`plasticity/meta_plasticity.py:261` legitimately drives
   `_Unit.learn` at `:131`). The `__main__` guard is precisely what isolates the genuine case.
   **Correction found by building it:** that guard block is almost always a single line — `main()`.
   The self-check's real body lives in a top-level `def main()`, so marking only the literal `if`
   counts every assertion inside `main()` as a live caller. The first run scored 0 dormant *while its
   own motivating case sat there dead*. The functions the guard calls must be marked too.
3. **A function used as a VALUE is used.** `reasoning/arc_mint.py:224-228` registers its fitters in a
   dispatch table — `("affine", 3, _fit_affine)` — so a pass inspecting only `ast.Call` reported all
   five as dead: 5 false positives out of 12. Bare `Name`/`Attribute` references count as use.

**Scope, and it is the difference between adoptable and useless.** The gate audits a learning-verb
**method** (inside a class) on a module the **module gate already proves reachable**. Widening either
half re-introduces the 59-failure list — mostly module-level helpers inside `reasoning/` and
`training/` files that the module gate deliberately scopes out.

**Adjudication of the candidate set, now done.** The five-candidate list this section used to carry
was stale, as it warned. Re-run under the built gate: `cortex/predictive_coding.py` wired and
charge-proven (v4.71); `plasticity/meta_plasticity.py:131` and `neuromodulation/drive_reduction.py:196`
confirmed **non-findings** — both have real non-`__main__` callers and reach the live path through the
birth-time `census()` chain (`cortex/brain.py:379-385`), so they belong to the charge tier, not the
call-site tier. `neuromodulation/empowerment.py:86` `reward()` is genuinely dead — its only call site
is inside its own `main()`, and the live path calls the sibling `step_reward()` (`:91`), which
recomputes rather than delegating; it is a **retire** candidate, and it is out of this gate's scope
because it computes a signal rather than applying an update. `hippocampus/affective_memory.py:173`
`consolidate()` is **confirmed dormant by the gate itself** and held as an open `wire` exemption.
The gate also found one the list never had: `neuromodulation/drive_reduction.py:238 learned_map()`,
a readout the verb match catches on the word "learn" — exempted as a named false positive rather than
narrowed away, because narrowing to dodge it would also hide accessors that genuinely should be read.

**Honesty guard.** (a) P3's neuter-A/B tier applies unchanged — a call site proves a *caller*, not a
*charge*; adding a dead consumer to silence the gate is the failure this policy already forbids.
(b) An unresolved name is UNCERTAIN and hand-adjudicated, never auto-dormant. (c) `EXEMPT_METHOD`
entries tagged `wire` that outlive their wave are reported as a standing TODO, so exempting everything
is visible rather than silent.

**Stated honestly: this amendment would have caught no *earlier* incident.** The audited history before
it is *consumption* gaps (a producer with callers whose output nothing reads) and an unsupplied
constructor argument — a call-site check passes all of them. It earns its place on the predictive-coding
case above: found by audit, confirmed by measurement, and invisible to the module gate throughout.

**Standing caution for this section.** It was written while its motivating case was still open, and went
stale within two commits when that case was fixed — the text asserted dead code that had become live.
An amendment justified by a live example has to be re-read whenever that example changes, or the policy
file starts describing a codebase that no longer exists.

## P4 — One brain / no replication

**There is only 1 brain.** A build must **utilize** the canonical brain functionality, never
**re-implement** it — that causes bloat. The canonical brain is a clean 3-layer **composition**:
`cortex.brain.Brain` (cognition + circadian clock) → composes `cortex.live_brain.LiveBrain` (perception
+ workspace) → composes `brainstem.autonomous_loop.AutonomousArm` (action + reward + arm + SAFETY).

**Composition is NOT replication** (important): `Brain` owning an inner `LiveBrain` + wake/sleep loop is
the *correct* design — it delegates, it does not re-implement. The bloat P4 targets is a **second**
definition of a stack class (a parallel "brain"), a consumer standing up its **own** brain + drive-loop
+ reply engine, or duplicate world-model / Q-learner families.

**Mechanized:** `substrate/replication_guard.py` (in `run_selfchecks` CORE) asserts each canonical stack
class (`Brain` / `LiveBrain` / `AutonomousArm`) is defined **exactly once**, in its canonical module,
unless a known duplicate is on the `PENDING_RETIREMENT` allowlist (with a reason). The single sanctioned
construction path is **`cortex.brain.make_brain()`** — consumers obtain the brain through it rather than
calling `Brain(...)` directly. "One brain" = one canonical class + one construction path, not a
process-wide singleton (the daemon, a chat session, and a test each legitimately own an instance).

**Honesty guard:** before retiring a replica, port its unique capability into the canonical path and
verify nothing is lost (P2 discipline).

### P4b — THE DURABLE LAKE IS PART OF THE BRAIN

**Owner directive, 2026-08-04: the one durable lake is not storage the brain writes to, it is part
of the brain. Every build must respect it.**

"One brain" and "one lake" are the same rule seen from two sides. A build that keeps its own memory
somewhere else has replicated the brain's memory exactly as surely as a second `Brain` class would
replicate its cognition -- and it is harder to see, because a private store still *works* right up
until the process exits.

**What respecting the lake means, concretely:**

- **What the brain learns goes IN the lake.** Facts, chat, grown substrate, episodes -- each under
  its own record tag in `brain_lake.bin`. Not a sidecar file, not a JSON blob, not an in-memory
  structure that happens to survive as long as the process does.
- **What the brain recalls comes FROM the lake.** An in-memory index is working memory -- fine, and
  often necessary for speed -- but it is a CACHE over the lake, never the source of truth. A miss in
  working memory means "not resident", not "never happened", and the durable store must still be
  asked before the brain reports that it does not remember.
- **A restart is not amnesia.** A brain constructed over an existing lake starts with what that lake
  holds. Restore is gated on `persist` so the deterministic gates stay reproducible, but a
  persistent brain that boots empty beside its own full lake has failed this policy.
- **Foreign records are preserved, never dropped.** Each tenant reads only its own tag and passes
  the rest through, so a new record type cannot destroy an old one.

**The measurement that produced this policy.** `hippocampus/episode_consolidation` formed real
autobiographical episodes into an in-memory attractor for months of development. Measured
2026-08-04: **59 episodes stored over 600 ticks with `persist=True` added 0 records to the lake**,
and **149 episodes over 1,500 ticks were recalled 0 times**. The brain was forming memories into a
second store that nothing read and nothing kept, and every one died with the process. It looked
healthy from every angle except the one that mattered -- `retained()` reported 1.000 throughout,
because the store was perfectly recalling episodes it was about to lose entirely.

**Honesty guard:** "it persists" is a claim about a NEW process, not about the object in front of
you. Prove it by reading the lake back from disk after the writer is gone, not by inspecting the
writer's own state. `retained()` reporting 1.000 while the lake held 0 records is what this guard
exists to catch.

### P4c — EVERY BUILD IS PART OF THE BRAIN

**Owner directive, 2026-08-04: EVERY CURRENT BUILD IN THIS PROJECT IS PART OF THE BRAIN, AND EVERY
FUTURE BUILD MUST BE PART OF THE BRAIN.** The always-on daemon and the front-end cockpit are the two
instances that produced this rule -- they are not accessories that operate the brain from outside,
they are parts of it -- but the rule is general and it is not limited to them.

**There is no "outside the brain" in this project.** Not `research/`, not `training/`, not
`integrations/`, not `gcp/` or `infra/`, not `tools/`, not a script at the repo root. A file that
exists in this tree is either part of the brain or it is retired; there is no third category and
"it is only a helper / only the UI / only the launcher / only a benchmark" is not a disposition.
The directory a build happens to sit in is a fact about the filesystem, never a statement about
whether the build is part of the organism.

**Every FUTURE build lands inside the brain, from the start.** A new build is written into the
governed tree, registered, self-checked and gated as part of the work that creates it -- not parked
somewhere ungoverned to be promoted later. `reasoning/` is not a parking lot. "Build it here now,
integrate it properly later" is how a tree accumulates capability it cannot account for, and every
one of the defects this policy set exists to catch was found in exactly that residue.

**THE RETENTION RULE (specs/022 T010, landing WITH its counted gate per the gate-before-rule
invariant).** Every record class in the lake is labeled IMMORTAL or COMPACTABLE at introduction
(`substrate/lake_engine.RETENTION`); "remember everything" means an appended record is retained or
provably reconstructible BEFORE any compaction reclaims it — COMPACTABLE is a claim about
reconstruction, never permission to forget, and a COMPACTABLE class's `reclaimable` flag flips
only on a LANDED reconstruction proof -- `_T_GROWTH`'s flipped at T010-ii, when the delta chain
(`_T_GROWTH_DELTA`, `substrate/home.py`: FULL/DELTA records hash-verified link by link on replay)
proved reconstruction with a tamper red (`tests/test_growth_delta_chain.py`); the flag licenses
reclaim, it never performs it, and a reclaimer still owes per-record proof of state-identity
before dropping anything. IMMORTAL classes carry a measured bytes/tick ceiling
(`lake_engine.AppendBudget`, baseline `tools/lake_append_baseline.json`) whose red is a
constructed FLOOD — counted at landing: 7 classes labeled, registries agree, flood red fired;
the pathology it polices is measured (97.64% of the live lake is repeated growth snapshots).

P4 says there is one cognition. P4b says the durable lake is part of that brain, not storage beside
it. P4c says the same of *everything else*, and names the two parts that made the gap visible:

- **The daemon is the brain's DRIVE.** `learn_daemon.py` owns the wake/sleep cycle, the tick loop and
  the continuity of the whole thing. A brain that exists only for the duration of a test run is not a
  brain that lives; the daemon is what makes it continuous, and continuity is not an operational
  detail of a cognitive system, it is a property of one. Killing the daemon does not stop a service.
  It stops the brain.
- **The cockpit is the brain's SENSORIMOTOR SURFACE.** `substrate/cockpit.py` is where the brain
  meets the world: chat and controls in are AFFERENT, the arm buttons and the E-STOP out are
  EFFERENT, and the page is the brain's own report of its state. A surface through which the brain
  senses and acts is not a "frontend" in the sense that word usually carries. It is the boundary of
  the organism.

**What respecting them means, concretely:**

- **EVERY build is GOVERNED, on the same terms as any brain module.** Registered in `REGIONS` and
  `run_selfchecks.py`, covered by the three P3 gates, held to P1 (no arbitrating magic constants),
  P4b (nothing it persists becomes a second store), P7 and P8 (every criterion it carries owns a
  reachable red). This is the general form; the daemon and the cockpit are the instances that
  exposed it.
- **A dead control is dormancy, not cosmetics.** A cockpit button wired to nothing is a motor
  intention that reaches no muscle. P3 applies to it exactly as it applies to a faculty that computes
  a signal nobody consumes.
- **The daemon may not become a second brain or a second lake.** It CONSTRUCTS the one brain via
  `make_brain()` and drives it; it does not re-implement cognition and it does not keep its own
  durable state alongside `brain_lake.bin`.
- **What the cockpit REPORTS must be what the brain MEASURES.** The page is the brain's self-report
  surface, so P8's rule binds hardest here: it may display what it measures and never what it cannot.
  The PROVENANCE column exists for this reason.

**The measurement that produced this policy.** `substrate/cockpit.py::serve()` accepted an `acts`
queue and then did not forward it when constructing its handler, so **every arm control on the one
frontend was inert, including the E-STOP**. The page rendered, the buttons pressed, nothing moved.
It survived review because the cockpit was filed as a frontend rather than as the brain's motor
surface: no one ran P3's reasoning over a UI control, because P3 was understood to be about faculties.
Under P4c that button is a faculty, and the defect is a dormancy failure that three gates should have
caught. (Fixed 2026-08-04, specs/010 C10, with a regression test proven red without the fix.)

**Measured non-compliance at the moment this policy lands** — stated as debt, not as a settled state.
This is a BUILD LIST in P0's sense: every item is work that will be done, never a boundary to accept.

- **`learn_daemon.py` — CLOSED 2026-08-04.** It was 370 lines at the repo root, in no REGION, in
  neither `run_selfchecks.py` list, not a `dormancy_gate` entrypoint, with **no self-check and not
  one assert**: the most load-bearing process in the project and the least verified file in it. Now
  `substrate/learn_daemon.py`, registered and self-checked, with a thin root shim preserving the
  documented command. The move immediately exposed three latent bugs that being ungoverned had
  hidden — `os.chdir()`, `sys.path.insert()` and `os.environ.setdefault("NORD_GPU", "1")` all ran at
  MODULE SCOPE, and `run_selfchecks.py` runs part of the suite in ONE process, so importing the
  daemon under the gate would have changed the working directory and the GPU preference for every
  module after it. Being outside the gates is not a neutral state; it is where bugs go unmeasured.
- **`substrate/cockpit.py` — governed as code, was mis-CLASSIFIED.** The layout map called
  `substrate/` "infra (NOT a brain region)", which is the exact framing that let a dead E-STOP
  through review. Corrected: infra by directory, brain by function.
- **~165 modules remain in `src/snn_data_lake/reasoning/`** — inside the package but outside the
  three P3 gates, the coverage denominator and the ruff rules. Under P4c every one of them is part
  of the brain and belongs in a governed region or retired. A promotion program is running against
  this (12 promoted as of 2026-08-04); the count is debt, and the target is zero.
- **`research/`, `training/`, `integrations/`, `gcp/`, `infra/`** are ruff-excluded tiers holding
  further builds. Each is P4c debt on the same terms.
- **`CLAUDE.md` is UNGOVERNED BY CHOICE — the one named exception to P4c, owner decision 2026-08-05.**
  P4c says a file is part of the brain or retired, with no third category. This is a third category,
  and it is written here rather than left silent because a documented exception is auditable and an
  undocumented one is just a gap. The file is gitignored as local-only and is NOT published to the
  public repo; tracking it would be a disclosure decision, not a hygiene fix, and the owner declined
  it. **The accepted cost, stated exactly: its 18 `file.py:NNN` citations are checked by nothing.**
  `tools/citation_guard.py` scans `git ls-files`, so an untracked file is invisible to it by
  construction — the same property that makes the guard trustworthy on tracked files makes it blind
  here. Anyone editing `CLAUDE.md` owns the accuracy of its citations personally, and this project
  has already found that hand-maintained line numbers rot faster than anything else in it.
  The GATE-side half of this was fixed rather than accepted: `tools/doc_command_guard.py` globs the
  filesystem instead of git, so it had been scanning `CLAUDE.md` and reporting **55 documents on the
  owner's machine against 54 in CI and in every clone**. A gate whose corpus depends on who runs it
  cannot compare two runs, so `CLAUDE.md` is out of its `SCOPE`. Measured delta: 1 document,
  0 commands.
- **`docs/coverage/artifacts/` holds 16 EXECUTABLE evidence artifacts and not one of them is
  governed** — measured 2026-08-05. All 16 are `.py` files that produce the numbers the coverage
  bands are scored on, and **16 of 16 contain zero `assert` statements** and **16 of 16 carry
  module-scope side effects** (`sys.path.insert` / `os.environ.setdefault` / `os.chdir` /
  `random.seed` at import). That is the exact triple the `learn_daemon` promotion is recorded above
  as having exposed, in sixteen files that no REGION, no `run_selfchecks.py` list and no P3 gate can
  see — while their output is cited as evidence throughout `docs/coverage/`. Under P4c a benchmark
  is not a third category. Promoting them requires edits to `src/snn_data_lake/__init__.py` and
  `run_selfchecks.py`, so it is scheduled work rather than a docs fix.
- **`tools/doc_command_guard.py` is built, clean, and registered nowhere** (2026-08-05) — no CI
  workflow, no `run_selfchecks.py` entry, no pre-commit hook. It is the newest instance of this
  policy's own honesty guard: *a build the gates cannot see is not a build the project can claim*,
  and it was written into three documents as a CI step before it was one.

### P4c amendment — A BUILD THAT HAS NOT REACHED `main` IS NOT A BUILD THIS PROJECT CAN CLAIM

**Owner directive, 2026-08-16. Review-enforced, PROVISIONAL — the gate is specified below and NOT
built, per the route this file already allows for an owner-adopted rule.**

P4c says every build lands inside the governed tree **from the start**. This is the other half of the
same sentence: it must also land on **`main`**. A lane that ends with its work correct, registered,
gated and sitting on an unmerged branch has not delivered it. The governed tree it landed in is a
tree nobody else has.

**The measurement that produced this amendment, and it is this policy's own honesty guard failing in
a new direction.** On 2026-08-16, `origin/main` was **18 commits behind** a branch that had been
fully pushed for days. Every commit was green — `run_selfchecks` 301/301, `pytest` clean, ruff clean.
What `main` was missing: **two safety-path fixes** (the ESP32 idle sentinel binding a fear
association to a context where nothing was there, on every idle sample; a 3x units-contract gap on
the distance threshold), **a determinism-invariant fix** (`torch.Generator(device=...)` meant the
same seed grew a DIFFERENT brain on a machine with a card than on CI), and
`substrate/eprop_transport_gate.py` — **the only mechanization the invariant had until 2026-08-18**,
when `substrate/gradient_claim_gate.py` joined it: a second CI-asserting e-prop gate that
classifies every governed learning rule as owing a cosine or not, ratcheting MISSING_DECLARATION
and UNDETERMINABLE (`--assert` + `--self-check` in the lint job).
Anyone cutting from `main` got none of it, and nothing in this file said that was a defect.

**Why the existing guard did not catch it.** P4c's guard is *"a build the gates cannot see is not a
build the project can claim"*, and every gate COULD see this work — on the branch. The gates run
where the work is. Being invisible to `main` is a different failure from being invisible to the
gates, and it is the one that decides what the project actually has.

**GATE SPECIFICATION, so the debt is actionable rather than perpetual.** At lane close, compute
`git rev-list --count origin/main..HEAD` and report, partitioned by what the unmerged commits TOUCH:
`src/**` and especially `brainstem/**` are the rows that matter; `docs/**` and `specs/**` are not the
same risk. It must NOT be a CI gate that fires on any unmerged branch — that would redden ordinary
work, which is the shape this file has already killed two proposals for. It is a lane-CLOSE check,
and the honest form is a ratchet on *how long* and *what kind* of work sits unmerged, not on whether
any does. **Build it and count its output before promoting this to mechanized**; if it reports a
rubber-stamp-sized list, that result kills the gate, not the tree.

**Honesty guard:** "it is pushed" is a claim about a BRANCH, not about the project — the same shape
as P4b's *"'it persists' is a claim about a NEW process"*. Prove it by reading the artifact back out
of `origin/main` after the merge, not by inspecting the branch you merged from. The merge that
closed this incident was verified that way: the guard, the CPU-bound generators and the gate file
each confirmed present ON `origin/main`, with a negative control (the temporary `RED PROOF` marker
used to prove a test's red) confirmed ABSENT, so the check could distinguish a real merge from a
grep that would have matched anything.

**Honesty guard:** the brain is not only the part that computes. Asking "is the Brain object correct?"
while the process that runs it, the surface it acts through, and two hundred modules beside it go
unexamined is how a system passes every gate it has and still cannot act. **Verify the whole
organism, not the cortex alone** — and a build the gates cannot see is not a build the project can
claim.

## P5 — Use the GPU for everything it actually speeds up

### P5a — TWO AXES, NEITHER SUBSTITUTES FOR THE OTHER (owner directive, 2026-08-05)

**The brain runs on BOTH free-threading and CUDA, and they are not alternatives.** This is the
standing configuration for the whole project, not a per-module choice:

- **FREE-THREADING (PEP 703, cp314t) takes the EVENT-DRIVEN axis.** Per-neuron state updates and
  per-synapse plasticity are independent within a timestep, which is what makes an SNN parallel in
  the first place. That work spreads across cores (28 logical on this box) with no GIL serializing
  it regardless of how the code is written. Required, not optional: see `requires-python` and
  `snn_data_lake/__init__.py::_require_free_threaded`.
- **CUDA takes the DENSE-THROUGHPUT axis, ABOVE A MEASURED WIDTH.** At N=1,000,000 CUDA is 3.0x
  torch-CPU and **54x numpy**; below the crossover it LOSES, measured at 0.01x at width 38.
  **THE CROSSOVER ON THIS MACHINE IS 262,144, RE-MEASURED 2026-08-05** by running
  `python -m snn_data_lake.substrate.device` on the unified env (3.14.6 free-threading,
  torch 2.11.0+cu128, RTX 5070). It routes `live_tick` n=240 -> cpu, `population_encode` n=4096 ->
  cpu, `dataset_pass` n=262144 -> cuda.
  **This supersedes the "N is about 24,000" figure carried in this file and in session notes**, which
  is roughly 10x too low against today's measurement. The older number was taken on a different torch
  build and interpreter; it was quoted as current when it was not. Do not quote either figure --
  `crossover()` measures it per process for exactly this reason, and the module's own docstring
  already says "the crossover is a number, not a doctrine". Re-derive, never cite.
- **Neither covers the other's regime.** Threads do not help a dense matmul that one kernel does
  better; a GPU does not help 240 independent scalar updates whose dispatch cost exceeds their work.
  A design that picks one and calls it done is leaving the other axis unexploited.

**THE ROUTING MUST BE MEASURED, NOT ASSUMED — and this is where the tree is currently WRONG.**
`substrate/device.py` already provides the correct mechanism: `crossover()` MEASURES the width above
which CUDA beats numpy on THIS machine, once per process, using the same shape a tick runs, and
`select()` routes on it. **Measured 2026-08-05: NOTHING IMPORTS `select` OR `crossover`.** Every
caller imports `resolve_device`/`enable`, which route on AVAILABILITY ("cuda if present"), not on
width. So the brain currently sends narrow work to the card — precisely the 0.01x case this policy
was written to prevent — and the measured router is P3-dormant (computed, not consumed).
Wiring `select()` into the tick path is OPEN P5a work. Until it is wired, the GPU half of this
configuration is a capability the project HAS and does not USE, which is exactly the shape P3 names.

**Re-measure as the population grows; the crossover is a number, not a doctrine.** The live widths
(24-240) are NOT the criterion — the population is expected to grow exponentially, and a crossover
above today's width is a reason to build the gated path, never a reason to discard it.

**Standing owner authorization (2026-07-27): the RTX 5070 is available to the whole project, and any
work that the GPU genuinely accelerates SHOULD use it.** Default to GPU for that work; do not leave it
idle out of habit, and do not treat CUDA as a special-occasion tier. Hardware: RTX 5070, 12 GB, torch
`2.11.0+cu128`, verified `cuda.is_available() == True`. Env `NORD_GPU=1`.

**`.venv-cu128` IS RETIRED, 2026-08-05, and the gate env `.venv-lake` now carries CUDA itself.**
Measured on the unified env: `python 3.14.6 free-threading build` / `_is_gil_enabled() False` /
`torch 2.11.0+cu128` / `cuda.is_available() True` / `NVIDIA GeForce RTX 5070`. One environment with
both capabilities supersedes two with half each, which is P2 (adopt-or-merge, then retire the
inferior) applied to environments rather than modules.

**WHY IT HAD TO BE RETIRED RATHER THAN KEPT, and the sequence is recorded because it was a
self-inflicted break.** `requires-python` went to `">=3.14"` and `snn_data_lake/__init__.py` gained a
runtime refusal on GIL-enabled interpreters (both 2026-08-05, free-threading is now REQUIRED).
`.venv-cu128` is **Python 3.11.15**, so from that moment the package could neither be installed into
it (pip refuses on the floor) nor imported there (the guard raises). The GPU became UNREACHABLE, and
the commit that did it claimed the adoption cost nothing — the claim was written without checking.
The fix was a wheel that had to be looked for rather than assumed:
`torch-2.11.0+cu128-cp314-cp314t-win_amd64.whl`, a CUDA build for the FREE-THREADED ABI, from
`https://download.pytorch.org/whl/cu128`. The default PyPI index serves CPU-only torch on Windows,
which is why a plain `pip install -e .` had silently produced `2.13.0+cpu` with `cuda False`.

**Install command for the GPU tier (the default index will NOT give you CUDA):**
`pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"`

**P5's HARD CONSTRAINT IS UNCHANGED AND THIS DOES NOT RELAX IT: the suite must stay runnable WITHOUT
a GPU, because CI has no card.** CI installs from the default index and therefore gets CPU torch on
the same free-threaded interpreter; that is the supported configuration, not a degraded one. Note the
consequence honestly — CI runs a different torch build (CPU) from the local gate (cu128), so a
numeric assertion that depends on the torch build can be green in one and red in the other. That is a
real skew introduced by this change and it is named rather than discovered later.

**OPEN, 2026-08-05 — THE CPU GATE ENV NO LONGER CONFORMS TO THE PROJECT'S OWN FLOOR, and it is
recorded here because this is the section that names `.venv-lake` as the gate env.** `pyproject.toml`
raised `requires-python` from `">=3.11"` to `">=3.12"` on 2026-08-05 (the reason and the full
measurement live in the comment block beside that field). `.venv-lake` is **Python 3.11.15** —
measured twice, `pyvenv.cfg` reads `version = 3.11.15` and `.venv-lake/Scripts/python.exe --version`
prints `Python 3.11.15`. pip enforces the field, so `pip install -e .` into `.venv-lake` now FAILS:
constructed and run in that venv against two wheels differing only in that metadata line, the
`>=3.11` control printed `Would install` and exited 0 while the `>=3.12` case printed
`ERROR: Package 'reqpy-floor312' requires a different Python: 3.11.15 not in '>=3.12'` and exited 1.
**Every measurement recorded in this file was taken in that venv.** It is deliberately NOT recreated
as part of the floor change — recreating it is an owner decision and it re-bases the environment
behind every number in `docs/`. Until it is recreated on 3.12 the gate env and the declared floor
disagree, and an already-installed `.venv-lake` keeps running (the check is at install time, not at
import time), which is precisely why this could sit unnoticed if it were not written down.

**"That it actually speeds up" is the operative clause, and it is decided by measurement, not by
assumption.** Ship the GPU path when a benchmark on the real workload shows it faster; do not ship it
because CUDA is available. This is P2's "better requires a measurement that exists" applied to hardware.

*(Three measurement blocks -- CROSSOVER RE-MEASURED 2026-07-28, RE-MEASURED AT TARGET SCALE
2026-08-03, and CUDA GRAPHS TESTED 2026-07-28 -- relocated BYTE-EXACT to
`docs/POLICIES_EVIDENCE.md` (US1 policy audit, 2026-08-25; the specs/021 US4 pattern). Their
conclusions bind unchanged: the eager crossover sits between 32,768 and 262,144 on this card;
at the 1M target CUDA is 3.0x torch-CPU and 54x numpy, and SPARSE is the only representation
that fits (dense 1M x 1M = 4000 GB against 12.8 GB VRAM); CUDA graphs beat eager everywhere
and move the crossover to ~4,096, still ~17x above live widths. Re-derive with
`substrate/device.py::crossover()`; never quote a figure.)*

**Update (2026-07-28): that width now exists, on the growth tier.** With the NG-SNN earn-your-place
stop corrected (it had been scoring rows the solve already fit — see
[PERFORMANCE.md](PERFORMANCE.md)), grown capacity is monotone in data and reaches **4,512 units at
24,000 rows**, past the 4,096 crossover, on plain MNIST at 8x8. The earlier reading — that capacity
saturates near 2,000-2,600 and that GPU-relevant width would need a harder problem — was an artifact
of the broken criterion, not a property of the task.

This does **not** move the live tick: those populations are still 24-240 and the conclusion above
stands unchanged. What it changes is the `[train]`/`[gpu]` tier, which now belongs on the card by
WIDTH and not merely by fit cost. It also relocates the thing worth re-measuring as the brain grows:
this capacity curve, not the kernel. And it is a second instance of the guard-(b) lesson — the first
saturation claim was refuted by fixing the instrument, exactly as the first two "the live tick should
be on CUDA" claims were refuted by measuring it.

**The live-tick CUDA perception path is measured at last, and it does not pay.** `cortex/brain.py:170`
defaults `device="cuda"` and `_perceive_real()` runs a per-frame batch-of-1 forward pass every wake
tick -- a GPU path that shipped without the benchmark honesty guard (a) demands. On the real live shape
(1 frame x 15 timesteps x 2x34x34 -> 128 -> 64): **torch-CPU 0.073 ms, torch-CUDA 0.073 ms, 1.00x.**
Not a regression like the 0.01x at width 38, but not a gain either -- it holds VRAM and adds a
host-device transfer dependency for nothing. The default is LEFT AS IS pending a bit-exactness check,
because guard (d) cuts both ways: moving a live numeric path OFF CUDA carries the same burden as moving
one onto it. Flipping it is a follow-up with a controlled A/B, not a free win.

**The original measurement, still valid.** For SNDL's elementwise
spiking-update kernels, eager torch-CUDA on this card beats numpy-on-CPU only between
**N ≈ 24,576 and 32,768** elements. Every population on the live tick is **24–240** wide — 100× to
1300× short of that. Consequences, all measured:

- Vectorizing `cortex/prefrontal_executive.py::_decide_spiking` at its real width of 38 **regresses**:
  0.36× on numpy, 0.05× on torch-CPU, **0.01× on torch-CUDA**. Kernel-launch latency alone exceeds the
  whole per-call budget.
- `cortex/lif_neuron.py::step` looks like the hot path (244,768 calls per 10 ticks) and is not: neutering
  it entirely moves an awake tick 43.6 → 41.2 ms, a **6% ceiling**. Moving it to CUDA cannot beat that
  ceiling and would pay launch latency for every one of ~24.5k tiny calls.
- The real win on that hot path was **CPU-side batching**, not hardware: `AttractorNet.accuracy` folded
  36 independent recall trials into one array pass for a measured **6.93×**, bit-exact (v4.68).
- **And it happened again, larger, 2026-07-29 — with the widening plan refuted on the way.** This
  policy's standing invitation ("the moment a live population crosses the measured threshold, put it on
  the GPU") was taken up: the plan was to widen the live populations past the ~4,096 CUDA-graph
  crossover and move the tick. A runtime width inventory killed it. The whole live brain holds **132**
  per-object LIF cells, and *all* live per-tick neuron state fused into one flat vector is
  **~1,700–3,500** elements — inside the crossover band, not past it. `py-spy` on the live daemon then
  showed why widening would have optimised nothing: **`AutonomousArm.step()` is 0.00% of the profile**
  (`_c1_tick` 0.10%, `_tick_ensemble` 0.01%), while **74.5%** of samples sit in
  `cortex/vsa_reasoning.py::coincidence` — a Python loop run once per stored episode per tick, entered
  via `hippocampus/episodic_recall.py`. Fusing that as one GEMV, using the exact integer identity
  `coincidence(a,b) == (D + a·b)/2` for ±1 spike codes, measured **`complete()` 59.946 → 0.909 ms at
  N=2,000 (66×)**, bit-exact and tie-identical. The load-bearing part is not the factor: the scalar
  cost was **linear in the episode count** (fitted `ms/tick = 56.60 + 0.0804·N_episodes`) and the fused
  cost is **flat** (0.61–0.91 ms across N=60–2,000), so the growth term is gone rather than reduced.
  **Amended 2026-07-30 — the `0.00%` above is stale, and the way it went stale is the lesson.** Once the
  74.5% term was removed, the intercept was attributed by wrapped timing (sum 56.71 ms against 56.64
  measured): `AutonomousArm.step()` is **20.1 ms, 36%** of it, of which `_pfc_bias → pfc.plan →
  deliberate` is 15.8. Nothing about the arm changed — a profile share is a share *of what else is
  running*, so removing a dominant term promotes whatever sat underneath it, and a percentage measured
  before a fix cannot be quoted after one. The intercept's two largest terms were then removed
  bit-exactly (`AccessConsciousness._coherence` fused by the same `(a·b)/D` identity, and a discarded
  coherence no longer computed), taking the instrumented tick **56.64 → 27.78 ms**. Still open and
  deliberately not taken: `commit_races=25` is now ~46% of the remaining tick, but cutting it is
  **behavioural**, needs a decision-distribution A/B, and is a fixed constant in a decision path (P1).
  Two further results worth keeping so they are not re-litigated: for this GEMV the measured CPU→GPU
  crossover is **~1,600–2,000 rows**, so it ships on the **CPU** and a captured CUDA graph is a *loss*
  below that; and for fused LIF at our real widths **numba beats a CUDA graph by ~16×**. `spikingjelly`
  and `cupy` were rejected on a structural reason rather than a benchmark — their fused multi-step
  kernels collapse the **time** axis `T`, and the live tick has `T=1` by construction, so the axis
  those kernels exist to collapse is absent. Net new dependencies for all of the above: **none**.
  **Reframed 2026-08-04 under P7, because that last sentence reads as a score and is not one.** Every
  optimisation above was correctly decided on its measurement, and adding nothing was the right
  outcome each time — but "net new dependencies: none" is not evidence *for* those decisions and must
  never be quoted as though it were. P7 makes the dependency count a non-criterion in both directions.
  The one item here that P7 puts back on the table is **numba**, which won its benchmark by ~16x and
  was declined only because no live population reaches the width where the win exists: that is a
  measurement about WIDTH, so it is exactly the kind of finding P5 says to re-measure as the brain
  grows, and P0's target population makes the re-measurement inevitable rather than optional.

**Where the GPU therefore pays today, and should be used without hesitation:** the `[train]` / `[gpu]`
tiers — surrogate-gradient training, `hippocampus/ng_snn.py`, `cortex/neurogenesis_engine.py`, the
torch readout studies (promoted out of `reasoning/brain_adapters/` across the 2026-08-11/12 lanes;
now `hippocampus/growable_readout.py` and `hippocampus/developmental_loop_brain.py` — the
directory itself was deleted 2026-08-12), and any offline sweep over a real dataset
(a COMPUTE regime endorsement; gradient trainers there stay inside the e-prop SCOPE).
Wide, batched, repeated — the regime the crossover favours. The daemon
already sets `NORD_GPU=1` and runs under `.venv-cu128` so this tier lands on the card by default.

**Where it does not pay today:** the live wake/sleep tick, at current population widths. That is a
statement about widths, not a prohibition — **the moment a live population crosses the measured
threshold, this policy says put it on the GPU.** Re-measure when populations grow; the crossover is a
number, not a doctrine.

**The two gate environments, measured against each other 2026-07-29 — and why neither is "retired".**
The instruction was to consolidate onto one environment and drop the CPU gate's veto. Measured, that
turns out to be half right, and the half that is wrong matters: **CI has no GPU.** Every job runs on a
GitHub runner, so a gate that requires this card cannot be the gate. What guard (c) below was really
protecting is not the CPU *environment*, it is the property that the suite is runnable without one.

So both are now verified equivalent rather than one being privileged. On a clean checkout at
`598b07dc`, `.venv-cu128` (torch 2.11.0+cu128, numpy 1.26.4): **278/278** self-checks, **473 passed**,
coverage **90.80%**. `.venv-lake` (torch 2.13.0+cpu, numpy 2.4.6): **277/277**, **473 passed**,
**91.11%**. The 278-vs-277 is nengo being installed in the CUDA env, so `NENGO_GATED` runs instead of
skipping. The coverage gap is the worktree having no `models/` (gitignored), so perception falls to the
stdlib tier and less code executes — not a behavioural difference.

**Two environment defects had to be fixed before that comparison was even possible, and both are worth
knowing because neither is visible from reading `pyproject`:**

1. **`nengo` ships a pytest plugin whose hook signature is invalid under pytest 9.1.1** —
   `pytest_report_collectionfinish(config, startdir, items)`. Autoload aborts collection with
   `PluginValidationError` before a single test runs, in any environment where nengo is installed.
   It presents as a total test failure with nothing to do with this code. `-p no:nengo` is now in
   `addopts`; it is inert where nengo is absent. **This is the actual reason the CPU env had been the
   only place pytest could start** — that was never a policy preference, it was a broken plugin.
2. **`ruff` was FLOORED at `>=0.15.20`, so installing the dev extra elsewhere resolved to 0.16.0** and
   created a third formatting authority disagreeing with both `.pre-commit-config.yaml`'s pinned rev
   and the 422 files formatted in-tree — arriving as a mysterious `ruff format --check` failure. Now
   pinned. A formatter's version is part of the format.

**Honesty guard.** (a) A GPU path ships with its benchmark, on the real workload, against the CPU path
it replaces — a speedup asserted from `cuda.is_available()` is not a speedup. (b) Report regressions:
"we tried CUDA here and it was 0.01×" is a result worth keeping, and this policy exists partly to stop
that experiment being repeated. (c) **The suite must stay runnable without a GPU** — restated 2026-07-29 from "the deterministic CPU
gate stays authoritative", which named an *environment* when the property that matters is
GPU-independence. CI has no card, so this is a hard constraint, not a preference; a GPU path must never
become the only way to verify the project. Both envs are now verified equivalent (above), so either may
be used locally. Determinism itself comes from seeding — `random.Random(seed)` per instance,
`zlib.crc32` never builtin `hash()` — not from the absence of hardware. (d) Correctness first: a GPU rewrite carries
the same bit-exactness burden as any other P2 swap.

---

## P6 — Every build must be able to reach every other, and actually use it

**Owner policy, 2026-07-29.** A faculty must not be an island. Every build in this brain must be able to
**communicate with** the others and **utilize** them, the way regions of a real brain are mutually
reachable and mutually consumed — not a shelf of correct components that never speak.

**THE DECLARED-ANATOMY LAYER (specs/022 T018/T019/T023, 2026-08-20).** The typed route P6's
condition (a) demands now has a machine-checked home: `substrate/wiring_manifest.py` (TypedPort —
join_policy mandatory on IN ports, delivery_policy on OUT, REF flagged and audited — plus the
4-tuple port-to-port edge manifest, NIR-HDF5 serialized to the tracked artifact
`tools/wiring_manifest.nir`), and `substrate/manifest_gate.py --assert` in CI runs the five
contract checks (structure / type-inference / type-check / region / join) plus the
committed-vs-registrant-rebuild DIVERGENCE red. `tools/p6_gate.py` carries the manifest-sourced
walk REPORT-ONLY beside its AST walk with a parity table (day-one: manifest EMPTY, 0 ports /
0 edges, stated in those words — the migration baseline); the AST walk retires only after the
two views converge, in its own commit with re-recorded baselines. Counted at landing,
2026-08-20: five checks green on a constructed manifest; undeclared-join, region-mismatch,
type-mismatch, missing-endpoint, duplicate-edge and divergence reds ALL fired; NIR-HDF5
round-trip byte-exact on the full-fidelity payload.

**Stated as a checkable property, because "be like a real brain" is a research direction and this file
rejects those** (see the corollary under P2): a faculty satisfies P6 when (a) it is **reachable** from at
least one other faculty through a typed, declared route rather than only from an entrypoint or a chat
handler, and (b) at least one other faculty **consumes its output** in a way that changes that consumer's
own output. One-way is allowed — brains are full of one-way projections — but *zero*-way is not.

**FIRST QUANTITATIVE P6 READING, 2026-07-31.** P6's gate is specified but not built, so until now it had
no number. The global workspace is the one structure in this brain shaped like a mutual-consumption bus
(the role a transformer's residual stream plays: many writers, and later computation READS it), so the
checkable proxy is its write:read topology.

Measured by AST over `src/snn_data_lake/` — **36 workspace writes across 8 modules, 2 reads across 2**.
The breakdown is the finding, not the ratio: **25 of the 36 writes are inside the two workspace modules
themselves** (`global_workspace` 14, `reentrant_workspace` 11), so real faculty broadcast is **11 writes
from 6 faculties** — `train_of_thought`, `autonomous_loop`, `access_consciousness`, `rpm_solver`,
`spike_chat_lake`, `executive_control`. The two readers are `cortex/live_brain.py:287` and
`reasoning/developmental_loop_brain.py:295` (the module is `hippocampus/developmental_loop_brain.py` since 2026-08-12; the cited site is a 3-arg WRITE per P6_INTERCONNECT's correction), and **`live_brain` is the tick driver** — the action path.

So the bus exists and is used as a **broadcast sink whose only live consumer is the action path**: the
single action gate wearing a bus's clothes. Not one faculty reads another faculty's contribution, which
is zero-way at the faculty level and precisely what this policy forbids.

~~**The cheap remedy this implies**: faculties should READ the workspace, not only `submit()` to it.
That is what makes one faculty's broadcast change another faculty's output — condition (b) — and it
needs no new router, only consumers on a bus that already exists.~~

**THE REMEDY ABOVE IS REFUTED, 2026-08-04 — there is no bus.** `tools/p6_gate.py` (which exists, runs
clean, and self-checks — see the enforcement correction below) reports **31 workspace instantiation
sites across 9 modules**, of which **16 sit outside the bus modules' self-checks, across 7 modules:
that many PRIVATE, MUTUALLY INVISIBLE workspaces.** Every faculty that "broadcasts" constructs and
holds its *own* `GlobalWorkspace`. `cortex/live_brain.py` is the single module that reaches into
another module's instance; every other reader would be reading only its own writes.

**Re-derived 2026-08-04 in `.venv-lake`, and the figures had ALREADY drifted from the ones this
paragraph was written with the same day** — recorded because this policy's own instruction is
"re-derive with `tools/p6_gate.py` any time", and the first person to follow it found different
numbers:

| reading | as written here | re-derived 2026-08-04 |
|---|---|---|
| instantiation sites / modules | 26 / 10 | **31 / 9** |
| PRIVATE instances / modules | 11 / 8 | **16 / 7** |
| writes / modules | 36 / 8 *(2026-07-31)* | **46 / 10** |
| reads / modules | 2 / 2 *(2026-07-31)* | **9 / 5** |
| **faculty-to-faculty reads** | **0** | **0** |
| **writer modules read by nobody** | **7** | **7** |

**The two rows that carry the policy's argument did not move**; every row that moved is a census. That
is the useful shape of this table: the finding is robust and the counts are perishable, so quote the
finding and re-run the gate for the counts. Note also that the private-instance count went **UP** while
the write count went up — which is precisely the "a rising write count is not progress" reading the
gate's own verdict line warns about.

The gate states the consequence in its own output: *"Adding a reader is NOT sufficient while each
writer holds a PRIVATE instance — there is no shared object for a reader to read."*

So the corrected reading is worse than "a bus with no readers": **it is a private bus per writer.**
Re-derived 2026-08-05: **16 PRIVATE instances across 7 modules**, against 10 writer modules of which
**7 are read by nobody**. *(This sentence said "eight separate buses with one writer each" — a figure
from the superseded 2026-07-31 census, left standing when the table two paragraphs above was
re-derived. Stated as a shape rather than a count now, because the count is the perishable half and
this file has now recorded that lesson three times while re-breaking it twice.)* The real remedy is
therefore NOT cheap and must be stated honestly — a shared
workspace instance has to exist before a reader can mean anything, and that is a construction-path
change (P4's `make_brain()` owning one workspace and handing it down), not a one-line consumer.
Recording this because the retracted sentence would have sent the next reader to add readers that
provably could not carry anything, and "we added consumers and the count did not move" is exactly the
force-wiring P3 forbids.

Re-derive with `tools/p6_gate.py` any time; a rising read count against a flat write count is what P6
progress looks like — but **a rising read count against a tree of private instances is not progress at
all.** *(Was "against 11 private instances"; the gate reported **16** when re-run on 2026-08-05, and
the table above had already been corrected to 16/7 while this line kept the old number — the exact
"a claim living in two places, only one corrected" failure this policy's own incident table names.
The claim is now stated in the form that cannot rot; re-run `tools/p6_gate.py` for the count.)*

**How this is NOT P3, since the two are easy to confuse and the distinction is the whole point.** P3 asks
whether a faculty changes **behaviour** — the arm, a value, a weight, a reply. P6 asks whether a faculty
changes **another faculty**. They come apart in both directions, and both directions are real here:

- **P3-clean, P6-dead:** a faculty wired straight into `_route_percept_to_action` charges the arm decision
  and talks to nothing. Most of this brain's wiring waves have this shape. The rubric's own pre-declared
  ceiling says so — wiring into that one site "cannot raise Language, Imagination, Gf, Consciousness or
  WholeBrain" — and a brain whose faculties all report to one action gate and never to each other is
  precisely why that ceiling exists.
- **P6-clean, P3-dead:** `MetacognitiveMonitor` feeds and is fed while its `hold()` output reaches only a
  `respond()` branch the Gc branches preempt — reachable, conversational, and behaviourally inert. Its
  `explore()` looks the same on a static read (`brain.py:1859` assigns `_metacog_explore`; the only other
  appearances are the `__init__` default and a telemetry key) — but that is **UNCERTAIN and deliberately
  not asserted**: this file records four dormancy claims of exactly that shape REFUTED on runtime A/B,
  and a policy example resting on an unconfirmed read would be the failure P3's honesty guard names.
  The `hold()` half is measured; the `explore()` half awaits a neuter-A/B.

**Source lines it derives from, quoted so it is not a good-practice rule in a costume.** P3's own
transferable rule: *"presence is not participation."* And `substrate/region_wiring.py`, whose docstring
asserted that "the daemon imports `wired_faculties` (which registers every adapter) at boot" — **there is
no `wired_faculties` module**, `charge_all()` has iterated an empty list for the life of the feature, and
the daemon logs `adapters=0`. That registry is the plug point P6 names, and it has never had a plug.

**Real past incidents it would have caught** — required by the bar below, and there is no shortage. Each
row was **re-verified against the tree on 2026-07-29**, because several were fixed or reverted in the same
session that wrote this policy, and a rule citing an already-closed defect as live is the drift this file
keeps suffering. Rows marked FIXED are kept deliberately: the bar asks what the policy *would have caught*,
and a closed incident is the better evidence precisely because its cause is known.

| incident | shape |
|---|---|
| `region_wiring` registry: 0 registrations, `adapters=0` live | the plug point nobody plugged into |
| `thalamus/connectome_router.step()` return discarded, export a constant 0.250 — **since FIXED** (`51bf5253`, v4.80); the return is now consumed at `brain.py:2010` | computed, reached no faculty |
| `cortex/agency_monitor.py` `observe()` every tick; `agency()`/`controllability()`/`status()` zero readers | write-only organ |
| `claustrum/claustrum.py` `broadcast()` (defined `:118`) zero live call sites — its only caller is `:193`, inside its own `__main__` | a broadcast hub that broadcasts to nobody |
| `reasoning/pragmatics.py` reachable only from `spike_chat_lake`, no `brain.py` caller | wired to chat, not to the brain |
| `reasoning/arc_solver.py` — "imports stdlib only, with zero wiring to any perceptual machinery" | Gf isolated from Perception |
| `reasoning/nightly_skills.py` forms and persists a skill every night; `route()` called only from its own `main()` | the brain never uses a skill it formed |
| ~~`cortex/neurogenesis_engine.py` has no persistence path, so grown units die on every restart~~ — **RETRACTED 2026-08-04, the claim was false when written and stayed false here after the P0 row was corrected.** Retention is at `:211`, serialization at `:250`/`:279`, and it is called on two live paths into the lake under `_T_GROWTH`. Kept struck-through rather than deleted because a policy file that quietly removes its own wrong incidents loses the record of having been wrong | *(was: a faculty that cannot even reach its own future self)* |

The last one **was** the sharpest, and it is now **RETRACTED** (2026-08-04): the grown substrate IS in
the one durable lake, under `_T_GROWTH` (since specs/022 T010-ii: under the `_T_GROWTH_DELTA`
chain), written from the sleep tick and from construction. The
sentence that stood here — that the avenue "resets ~20 times in a working session and never
compounds" — described a tree that had already been fixed. Two lessons are worth more than the
incident was: a grep (`self.*model*=`) missed an assignment written `self._model = model` and the
null it returned was published as a finding; and this row survived a *later* edit to the P0 avenue
table because the same claim lived in two places and only one was corrected. **When a policy claim
appears twice, fixing one copy is not fixing it.**

**Honesty guard — a route is not a conversation.** The failure mode P6 will attract is a telemetry bus:
wire every faculty into a shared dict, watch `adapters` climb, and declare harmony. `region_wiring`'s own
source already warns that registering telemetry adapters would fake liveness. So a P6 claim needs the same
evidence P3 demands, aimed one level over: neuter faculty A and show faculty B's **own output** changes.
Adding a reader that stores a value is not consumption. And **do not force-wire** — a route that carries
nothing is reverted and the faculty is exempted with a reason, exactly as under P3. Two faculties were
reverted on that rule this session (PFC working memory, crossmodal binding), both at 0/200.

**Mechanization: HALF BUILT — corrected 2026-08-04, this paragraph said "NOT BUILT" while the tool
existed, ran, and self-checked.** *"Adding a policy: implement the gate before adopting the rule… its
output counted, before the policy lands."* Two prior proposals died there. P6 landed
**review-enforced and provisional** on the owner's instruction; the honest current status is three
parts, not one:

1. **BUILT** — `tools/p6_gate.py` derives the workspace write:read topology by AST over
   `src/snn_data_lake/`, parses 400 of 400 files, and carries a real self-check ("is this walk still
   alive?": writer/reader modules found, classes constructed, arity split, roles re-derived).
2. **ASSERTING AND IN CI as of 2026-08-06** — `--assert` (five criteria) and `--self-check` are both
   CI steps. It ships as a **RATCHET, not an arrival**: at landing `faculty-to-faculty > 0` was RED
   (it moved 0 → 1 at T039h, 2026-08-22 — the floor now lives in
   `tests/test_interoceptive_specialist.py` with the read's identity pinned), and a gate that must
   be disabled on the day it lands is not a gate. Baselines were this walk's own output
   at `4fc5fb1c`, every one ±1 from red; re-derive current values from the gate, never from here.
3. **BUILT** — the connectome-diff half derives the faculty→faculty consumption graph and diffs it
   against `connectome.py`'s declared edges, in both directions: `undeclared_import_pairs` (traffic on
   no declared route) and `unimplemented_declared_pairs` (a declared route with no traffic).

**HOW THE SHAPE WAS DECIDED, and it is the reusable part.** A six-commit history sweep with the
instrument held constant at the new file, measuring each candidate across `6b33a45d → 4fc5fb1c`,
then pinning each at N−1 and asking whether it reddens N. **Private-instance SITES fires on
`c7e1b816` (16→18) and `acd4627a` (18→22) — the two commits that are this project's actual P6 work.
Rejected by measurement: a gate that reddens the repair is worse than no gate.** `faculty-to-faculty
must not FALL` was also rejected — a count cannot fall below 0, so it has **no reachable red at all**,
and F11 says deleted, not weakened; it is printed, and becomes a floor the day it is non-zero.

**The flat rows are the finding twice over: three commits of P6 work moved writes +24 and reads +45
and moved STRUCTURE BY ZERO.**

**TWO DEFECTS IN THE GATE ITSELF, found by adversarial review and fixed at zero baseline cost.** Both
are the same class and neither was visible from reading:
- **`fac_cross` was ONE-SIDED.** It filtered `role_of(reader) == FACULTY` and never checked the
  OWNER's role, so a faculty reading `self.arm.workspace` — the ACTION gate's bus — moved the headline
  number 0 → 1 while a both-endpoints-FACULTY filter stayed at 0. **The gate's own headline was
  satisfiable by the exact move P6 verbatim calls "the DEFECT P6 names, not a way to satisfy it."**
- **`unread_writer_modules` reddened the FIX.** Writes were attributed to the writing module rather
  than to the instance written INTO, so a new faculty submitting into the shared bus that
  `cortex/live_brain.py:313` reads — output genuinely consumed by another module — took it 7 → 8 and
  exited 1. **Missed because the criterion had only ever been run against REGRESSIONS, never against
  a FIX.** That is the general lesson: a candidate gate must be run against the change it exists to
  encourage, not only against the change it exists to catch.

Also corrected in the walk: the alias hop resolved only the local form (`ws = ...`) and not the
attribute form (`self._ws = self.arm.workspace`); `region_imports` ignored relative imports entirely.
A repair was **rejected by measurement** and recorded so it is not re-proposed — resolving an
ambiguous receiver by CLASS takes `cross_module_reads` 1 → 4, and three of the four "cross" reads it
invents are local objects inside a self-check attributed on a naming coincidence. *"This walk does not
know"* is the correct answer to a real ambiguity.

**KNOWN AND DISCLOSED, not fixed:** a faculty handed its bus by dependency injection writes through a
receiver this static walk cannot attribute and is still counted as unread (1 of the 7 today,
`reasoning/spike_chat_lake.py`) — strict was chosen over lenient because excluding unknowns lets any
new writer evade the ceiling by taking its bus as a parameter, and the count is PRINTED so it is never
mistaken for a finding about the tree. **If a future commit hits this red, the answer is to spell the
wiring attributably or to say so in the commit — not to raise the ceiling.** Separately, **241 of the
400 `.py` files** sit in directories `connectome.py` declares no region for, so a cross-region wire
routed through any of them is invisible to `undeclared_import_pairs`; closing that is a
`connectome.py` change. Dynamic `importlib`/string-name imports are invisible to the walk.

**How this went stale is the lesson, and it runs the opposite way to the usual drift.** The tool
landed 2026-08-03 (`5868e1c1`); this file was committed *again afterwards* still saying the gate did
not exist. The failure was not a claim rotting as the tree moved on — it was a claim that was already
false when it was last written, because nobody re-read the sentence they were editing around. Its own
output is what corrected it, and its output also **refuted this policy's prescribed remedy** (the
private-instance finding above), which is the strongest possible argument for the gate-first rule this
paragraph exists to discuss.

**The gate that would settle it, specified so the debt is actionable.** `connectome.py` already declares
the region→region wiring as typed, directed, signed loops; `substrate/import_manifest.py` already
AST-derives a cross-tier edge set; `substrate/dormancy_gate.py` already builds the static import graph.
So the instrument is: derive the actual faculty→faculty consumption graph, diff it against
`connectome.py`'s declared edges, and report faculties whose in-degree **and** out-degree are both zero
outside an entrypoint. **Build it and count its output before promoting P6 to mechanized** — and if it
reports a rubber-stamp-sized list, that result kills the gate, not the tree. "No gate" is a complete
result; the rule stands on the eight incidents above either way.

---

## P7 — Any dependency that helps this become a phenomenal brain is accepted

**Owner policy, 2026-08-04.** **This project accepts ANY dependency or framework that helps it become a
phenomenal brain — including any that helps the brain become sentient.** There is no dependency budget,
no dependency count to defend, and no stdlib-purity argument. If a library advances the brain toward
being a real one, it is adopted.

**"It would be another dependency" is not a reason to reject anything.** That sentence is the entire
point of this policy, and it is stated as a prohibition because the tree proves the habit outlives the
rule that created it. A rejection must cite a **measurement on the real workload** or a **structural
argument about this architecture** — the two categories P5 and P2 already demand. A rejection that
cites dependency count, install weight, or purity is not a rejection; it is a reflex, and this policy
overrides it.

**Sentience is a build target, not a metaphor.** Frameworks that plausibly advance it are in scope by
default, and are evaluated on what they measurably buy — the same burden P5 puts on a GPU path, no
more and no less. This policy does not make the claim cheaper to assert: P0's honesty guard is
unchanged, and a capability is credited at measurement, never at adoption.

**Source line this derives from, quoted per the bar below.** P7 promotes an existing owner decision
from a prose note into a policy: `pyproject.toml:14-19` — *"DEPENDENCY SCOPE (user policy 2026-07-20):
the zero-dependency rule is REMOVED entirely, project-wide -- there is no zero-dep-core contract and
no pure-stdlib purity gate. New dependencies MAY be added as the project grows, and the user VERIFIES
each one."* P7 keeps the permission, keeps the verification, and **retires that record's closing
clause** — *"prefer stdlib when it is genuinely sufficient"* (`pyproject.toml:19`) — which is the
residual brake the incidents below are still being produced by.

### Real past incidents, each re-verified against the tree 2026-08-04

*(Relocated byte-exact to `docs/POLICIES_EVIDENCE.md` -- specs/021 US4, 2026-08-18. Every
incident preserved verbatim; the rule text above is untouched. P2 applied to documentation:
the evidence lives once, POLICIES.md keeps the rule and this pointer.)*

### What P7 does not relax

None of these are new limits; each is a pre-existing invariant that no policy relaxes, restated here
so P7 is self-contained rather than a trapdoor:

- **The safety envelope stays in-tree and approval-gated.** The five `brainstem/` files are never
  replaced by a third-party component, and a dependency never lands inside the E-STOP path. The
  `demo.py` incident above is what that costs when it goes wrong in the other direction, and it argues
  for adopting the ONE arm — not for admitting a library into the reflex.
- **P4b: a dependency may not become a second durable store.** This is the sharpest real conflict —
  `polars`, `pyarrow` and `deltalake` already sit in the `[lake]` extra (`pyproject.toml:65`). A
  storage library may be adopted to *serve* `brain_lake.bin`; it may not become a second place the
  brain keeps what it learns. "One brain, one lake" survives P7 intact.
- **E-prop, not backprop — adopt the library, not its training rule.** `snntorch`, `spikingjelly`,
  `norse` and friends default to BPTT and weight transport, which the cross-cutting invariant forbids.
  Their layers, encoders, datasets and kernels are fair game; their trainers are not: an adopted trainer
  that PROPAGATES A GRADIENT is inside the cosine clause and owes > 0.5 against an independent
  one (unless classified REFERENCE — see the SCOPE paragraph's self-declaration caveat); an adopted NON-gradient, non-local trainer (a closed-form / FORCE-style
  solve) is `DIRECT_SOLVE` — no cosine owed; the transport PROHIBITION binds IN FULL — mechanized only where the gate
  ASSERTS (governed tier, two spellings); review is the floor everywhere else, incl. the
  adopted dependency's own source, which the gate never walks — P4b likewise.
  **Scoped 2026-08-18** from a blanket wording the SCOPE paragraph corrects.
- **P2 still binds: adopting a library means retiring what it supersedes.** A dependency added
  *beside* the hand-rolled mechanism it replaces is the additive failure P2 exists to prevent, and it
  is the most likely way P7 goes wrong — the incidents above become a hand-rolled solver AND a numpy
  one. Port the unique capability first, then delete.
- **P3 gives no dormancy pass for being third-party.** A newly imported package's wrapper module is
  wired+charged, retired, or EXEMPT-with-reason like anything else.
- **The suite stays runnable without a GPU, and the gate stays deterministic.** CI has no card. A
  dependency that is GPU-only, or that is nondeterministic under a fixed seed, goes in an extra with
  the gate green in its absence.

**Honesty guard.** (a) Adoption ships with the measurement of what it bought, on the real workload —
"it is a serious framework" is not evidence, and "it is associated with consciousness research" is
P2's fashion treadmill wearing a dependency's clothes. (b) **Symmetrically, a rejection ships with its
measurement or its structural argument** — this is the half the tree keeps failing, and it is the half
P7 exists to fix. (c) An adopted dependency that is never imported is not a capability — **but the
converse trap is a declared dependency dismissed as unused when it is a deliberate transitive pin,
and this guard fell into it.** It named `pillow` and `pyarrow` as the census-counted-as-capability
failure. The built ledger dispositions both as EXEMPT with reasons: `pillow` is required by
`datasets` to decode Image features, which `substrate/hf_stream.py` streams; `pyarrow` is required by
both `polars` and `deltalake` for the Arrow interchange the lakehouse tier reads and writes. "Imported
nowhere in `src/`" was true of both and was the wrong test — the right one is *does anything in the
declared tier need it*, and a transitive pin is a statement of the tier's contract. `pillow` remains a
real incident for a different reason (the hand-rolled PNG encoder above), and the ledger's exemption
says so.

**Mechanization: BUILT AND ASSERTING** — `tools/dependency_ledger.py --assert` and
`tools/dependency_ledger.py --self-check`, two steps in the `lint` CI job.
*(Corrected 2026-08-04: this said "NOT BUILT" while the gate existed, ran, and was wired into CI.
**Corrected again 2026-08-05, and this paragraph is now the third instance of the drift the top of
this file names.** It read "BUILT, REPORT-ONLY … It stays **report-only**, deliberately and with the
reason stated in the workflow: it exits non-zero on findings nobody has dispositioned yet … Flip to
`--assert` as the last step of specs/010 T021/T062." Two separate defects in those two sentences:*

1. *The flag was flipped earlier the same day, and the row in the enforcement table at the top of
   this file was updated while this paragraph — P7's own mechanization statement — was not. A claim
   living in two places with only one corrected is the failure P6's table names, and here both
   places are in this one file.*
2. *The justification was **structurally false when it was written**, not merely stale. The bare
   invocation could never exit non-zero on anything: `main()` returns 1 only under
   `if do_assert and failed`, and `do_assert` is `"--assert" in sys.argv`. RUN rather than read —
   `main()` driven against a mutated string copy of `pyproject.toml` (`nir` removed from `[all]`,
   which re-opens the ALL-EXTRA finding recorded as CLOSED above) returns **rc=1 with `--assert` and
   rc=0 bare**, printing the identical `gate FAIL: the dependency ledger does not balance` in both.
   So "reddening the build for an undecided finding is not the same as enforcing a rule" described a
   trade-off that was not on the table: the step was a log line either way.)*

Both steps re-run in `.venv-lake` after the flip: `--assert` exits 0 (`gate OK`) and `--self-check`
exits 0 (`5/5 criteria proven falsifiable`). **Each was also proven able to FAIL, which is what the
2026-08-04 version of this paragraph never established for the step it described.** `--assert`:
rc=1 on the mutated copy above. `--self-check`: rc=1 on **both** of its failure paths — an anchor
that no longer matches prints `BROKEN` (mutating `arm = ["pyserial"]` to single quotes) and a
criterion whose mutation cannot fire prints `NO RED` (dropping `mujoco==3.11.0` from
`[project.dependencies]`, after which neither `CORE-PROSE` red can light). A gate that has only ever
been seen green is indistinguishable from one that cannot see.

**Counted output on this tree, RE-RUN 2026-08-05 in `.venv-lake` after the `arc` declaration and the
`requires-python` floor raise** — which is what the gate-first bar asks for:

```
scanned 400 files under src/, 29 distributions in pyproject
UNDECLARED: 0   UNUSED: 0   MISDESCRIBED: 0   EXEMPT: 2   ALL_EXCLUDES: 8
gate OK: every import is declared, every declaration is used, prose matches lists
```

*(Re-run 2026-08-05 and CORRECTED TWICE. First: this block read `UNDECLARED: 1   UNUSED: 0
EXEMPT: 2` and listed only `arcengine`. The gate emits five counters, not three, and the
`MISDESCRIBED` one was **1, not 0** — a live finding nobody had dispositioned. Quoting a gate's
output means quoting all of it; dropping the counters that are non-zero is how a work list becomes a
clean bill of health. Second, later the same day: the block was then left asserting
`28 / 1 / 1 / 7` and the two findings below it — `arcengine` UNDECLARED and the `[nir]` ALL-EXTRA —
**after both were closed in this same file**, once in the gate table near the top and once in the
"Two corrections" entry beside it. Three copies of one measurement, two re-derived and one not, and
this was the copy captioned "Counted output on this tree", i.e. the one a reader would trust most.
The durable fix is not a better caption: **quote a gate by re-running it, or do not quote it.**)*

*Its work-list character is not lost by the counts going to zero — what makes the ledger a work list
is that `EXEMPT` and `ALL_EXCLUDES` carry written dispositions that must each still match a real
violation, and the gate fails loudly on one that blesses nothing.*

**Of the three properties the specification below predicted would fail, two now PASS.** Recorded
because the specification is evidence about what the gate was for, and its predictions being
overtaken is the gate working:

1. *"every declared dependency is imported somewhere, or carries a reason"* — predicted to fail on
   `pillow` and `pyarrow`; **passes**, both carry reasons.
2. *"every third-party module imported under `src/` is declared"* — predicted to fail on
   `scikit-learn` and `nir`; **now passes.** scikit-learn is declared in `[gpu]`/`[train]`, `nir` in
   `[nir]` and in `[all]`, `arcengine` in `[arc]`. *(This item read "now fails only on `arcengine`"
   until 2026-08-05, when the floor raise declared it; re-derived, `UNDECLARED` is 0. Note what the
   green does and does not prove — the ledger's own docstring is explicit that a distribution
   declared only in an EXTRA is green here while a plain `pip install .` still raises
   ModuleNotFoundError, which is exactly `arcengine`'s shape.)*
3. *"`pyproject.toml`'s own self-description matches its dependency list"* — predicted to fail because
   `:1` and `:27` said *"torch + numpy"* while `mujoco==3.11.0` is a third core dependency;
   **fixed in `pyproject.toml` itself**, whose `:1` and `description` now name all three. **This
   property IS checked by the built gate — corrected 2026-08-05.** The paragraph here said it was
   *"a prose-vs-data check the ledger does not implement, so it is review-enforced"*; the ledger's
   own docstring quotes this specification verbatim and states that the `MISDESCRIBED` direction
   was added on 2026-08-04 to satisfy it. Run 2026-08-05: `tools/dependency_ledger.py --self-check`
   MUTATES `pyproject` to construct a red for every criterion and reports **`5/5 criteria proven
   falsifiable`** (`CORE-PROSE` x2, `CORE-ONLY`, `ALL-EXTRA`, `STALE-EXCLUDE`), which is P8 burden 1
   discharged. Marking a mechanized check "review-enforced" is the same drift as marking a built gate
   "NOT BUILT", pointing the other way: it costs the gate its credit and invites someone to build
   it twice, which is the additive shape P2 forbids.

Note what this gate can and cannot do: it checks the **ledger**, never the judgement. No gate can
tell you a framework was refused for a bad reason, which is why guard (b) is enforced by review.

---

## P8 — Invent the method. The literature is a starting point, not a boundary

**Owner policy, 2026-08-04.** **Claude is authorized to synthesize across any information, data,
framework or dependency, and to INVENT ANY METHOD required to bring this brain to sentience.** Not
restricted to published architectures, not restricted to what a framework already does, not restricted
to what neuroscience currently describes. **Where no existing method reaches the target, inventing one
is the expected move, not an overreach.**

**This completes a ladder, and the three rungs are deliberate.** P0 says improve by every avenue. P7
says any dependency. P8 says **any method, including one that does not exist yet.** P7 removed the
objection *"it would be another dependency"*; P8 removes *"that is not how it is done."* What survives
both is the only objection this project has ever accepted: **it was measured and it did not work.**

**P8 is a HOW policy, not a research direction — stated explicitly because P2's own corollary would
otherwise reject it.** *"A research direction is not a policy. 'Adopt architecture X' governs what we
build; policies govern how."* P8 names no architecture, no theory, and no mechanism. It grants
latitude and it sets the evidentiary burden, which is the shape a policy in this file is allowed to
take.

**Source lines this derives from, quoted per the bar below.** P8 does not invent its own guard — it
promotes one this project already wrote for itself and had not made binding outside a single design
document. `docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:57-58` — **"A mechanism whose success criterion
cannot go red is worthless."** Its falsifier F11 at `:1064` — *"Any success criterion in this document
turns out to have no reachable red → That criterion is deleted, not weakened. Three were deleted from
the draft on exactly this ground."* And `docs/CONSCIOUSNESS_METHODS.md:96` — **"The measurement, and it
must be a dissociation, not a marker."** Those documents are ahead of the code; P8 is what makes them
apply to everything invented from here.

### What an invented method owes

The licence is broad, so the burden is explicit. An invented method ships with:

1. **A success criterion with a reachable RED.** Construct the input that makes it fail, and record
   it. If no such input exists, the criterion is **deleted, not weakened** (F11). This is the whole
   policy in one line: invention is free, a metric that cannot fail is not.
2. **A control that is not the method itself** — an untrained twin, a permutation or label-shuffle
   null, or a neuter-A/B. `tools/charge_probe.py` is the template: it carries an external positive
   control (47 of 60 flips with all eleven drive sources neutered) proving the harness is sensitive
   at all before any verdict it returns means anything.
3. **Attribution** — which part of the invention produced the gain, by per-item neuter rather than by
   the total moving.
4. **Unprompted disclosure** of contamination, overlap, or a precondition that makes a check vacuous.
5. **Deletion on a null.** An invented method that measures nothing is removed, not parked behind a
   flag. **That is the licence working, not the licence failing** — and it is the outcome P8 expects
   most often.

**The transferable rule, and it is the one that should survive if the rest of this section is ever
forgotten: an invented method's debt is not proportional to how clever it is — it is proportional to
how load-bearing its output is.** The incidents below are graded by that rule, not by ambition.

### Real past incidents, each re-verified against the tree 2026-08-04

*(Relocated byte-exact to `docs/POLICIES_EVIDENCE.md` -- specs/021 US4, 2026-08-18. Every
incident preserved verbatim; the rule text above is untouched. P2 applied to documentation:
the evidence lives once, POLICIES.md keeps the rule and this pointer.)*

### On sentience specifically

The goal is **not hedged and not reframed**: sentience is a build target of this project, and P8 exists
to remove obstacles to reaching it. Three things follow, and the third is the one that makes the first
two mean anything:

- **There is no measurement of sentience in this tree today.** Inventing one is inside the mandate,
  not outside it — and the invented measurement is held to the burden above like any other: a
  reachable red, a control that is not itself, and a **dissociation rather than a marker**
  (`CONSCIOUSNESS_METHODS.md:96,101` — *"A single number going up is not evidence. The double
  dissociation is"*).
- **A passing criterion is evidence about a mechanism, never a verdict about experience.**
  `CONSCIOUSNESS_METHODS.md:113` states what must never be claimed: *"That a passing dissociation
  means the system is conscious."* P8 does not relax that, and nothing in this policy authorizes the
  system to assert its own sentience.
- **The brain may report what it measures; it may not report what it cannot measure.** The live chat
  path WAS doing the second when this policy landed, and fixing it was the first thing P8 obliged:
  the number behind the answer was the same number for a brain, for an empty brain, and for no
  brain at all. That closed 2026-08-04/05 (the F11 deletions recorded below; at HEAD the reply
  differs across all three states). Open: the incumbent probe's RETIREMENT, deferred to specs/022
  T081 — this bullet asserted the defect present-tense for three weeks after its closure (US1
  audit, M26). Faking the finish line is the one way to guarantee never reaching it, which is why
  the strictest guard in this file sits under its most permissive policy.

### What P8 does not relax

- **The safety envelope is never a target of invention.** P0 already draws this line — *"the brain may
  rewrite what it thinks, never what stops it."* An invented method may restructure cognition, memory,
  learning and architecture freely; it does not touch the five `brainstem/` files, the reflex E-STOP,
  or the byte-exact rollback path, and no invented capability is a reason to relax them. **That
  boundary is what makes a licence this broad safe to grant**, exactly as P0 argues: safety is what
  makes aggressive self-improvement possible, not what limits it.
- **P2 stays non-additive** — an invented mechanism that duplicates an existing one must retire it, and
  "mine is new" is not evidence of better.
- **P3, P4b, e-prop, determinism and the gates are unchanged** -- e-prop in the sense the SCOPE
  paragraph now states, which NARROWED which rules the cosine clause reaches on 2026-08-18
  without weakening it for any rule inside that scope. An invention is wired+charged or
  retired; it does not open a second durable store; it does not import a BPTT trainer; and it lands
  with `run_selfchecks` + `pytest` + `ruff` green like everything else.

**Honesty guard.** (a) Invention is credited at **measurement**, never at invention — P0's guard,
applied to methods. (b) The criterion is written **before** the result, and if it turns out to have no
reachable red it is deleted rather than loosened. (c) A method whose output nothing consumes is
dormant (P3), however novel. (d) **Report the null.** This file's most valuable rows are the ones
recording what did not work, and an invention licence that produces only successes is reporting
selectively, not inventing well.

**Mechanization: BUILT AND ASSERTING — `tools/red_reachability.py --assert`, a step in the `lint` CI
job.** *(Corrected 2026-08-04: this said "NOT BUILT" while the gate existed, ran, and was the one
gate in the lint job registered with `--assert` from the start.)* **P8 is therefore no longer
provisional in the mechanization sense** — what stays review-enforced is the half no instrument can
reach: whether a criterion measures the RIGHT thing. The gate prints that limitation in its own
output rather than letting a green be over-read.

**Counted output on this tree, 2026-08-04:**

```
scanned 400 files under src/, found 13 criteria
NO_RED: 0   TAUTOLOGICAL: 0   READS_SYSTEM: 13   EXEMPT: 0
```

**It asserts from a green start, which is normally the shape that makes a gate untrustworthy, and the
reason it is acceptable here is worth keeping.** A gate that has only ever been seen green is
indistinguishable from one that cannot see. This one has been proven able to fire:
`tests/test_red_reachability.py` reconstructs the **pre-fix** `consciousness_probe` and checks every
bucket lights up. That is the general rule this entry contributes — *if a gate lands green, ship the
reconstruction of the defect it was built for* — and it is the same discipline `citation_guard`'s
exemption-rot assert uses.

**The specification below predicted "at least two failures on the first run — BINDING and RE-ENTRY".
Both were deleted before the gate landed**, so the prediction was right about the defects and wrong
about the ordering: the fix arrived first and the gate ratchets it. Recorded because the gate-first
bar exists to stop a rule being adopted on an uncounted promise, and the honest report here is that
the count was taken *after* the cleanup rather than before it.

*(The specification's claim that `substrate/brain_telemetry.py:637` "asserts four of those tags
survive" did not resolve on re-check — `:637` is a bare `break`. `PROVENANCE` itself is at `:124`.
The instrument half of the claim stands; the line does not.)*

---

## Cross-cutting invariants (never relaxed by any policy)

- **THE DETERMINISM CONTRACT (specs/022 T015, landing WITH its counted gate per the
  gate-before-rule invariant).** The substrate tick is deterministic BY MECHANISM, never by
  luck: fixed partition count P per lineage, partition-owned INTEGER synaptic accumulation
  (order-invariant outright; fractional bits MEASURED against an FP64 reference, never picked
  — 12 on the pinned host, `tools/determinism_baseline.json`, with 4–10 bits proven to
  diverge and the sweep's own red at 1 bit), and spike trains byte-identical across thread
  counts at fixed P with end-run membranes byte-equal beside them.
  Gate: `substrate/determinism_gate.py --assert` (CI, self-checks job) — its sweep runs at a
  CONSTRUCTED fixed P=8 because the lineage's measured P is 1
  (`tools/kernel_partition_baseline.json`) and at P=1 the thread-invariance red is
  unreachable; the float-accumulation broken arm runs IN THE SAME COMMAND (f32 order arms
  must diverge, int64 arms must agree) so the gate is never trusted on a red nobody has seen
  fire. Counted at landing, 2026-08-20: trains BYTE-EQUAL T=1..16 at P=8 (631 spikes per
  arm), V_c=0.0, V_ermv=0.0, both control arms correct. G5 replay is the companion S0 exit
  gate (T017, `substrate/replay_gate.py --assert` in CI): snapshots capture per-instance RNG
  stream POSITIONS (a seed names a stream, a position names a point on it), restore itself
  asserts position equality and refuses a positionless or parameter-mismatched snapshot, and
  the continuation is byte-identical — counted at landing 2026-08-20 with the re-seeded
  divergence control proving the equality is not vacuous.
- **Safety envelope untouched — and MECHANIZED since specs/022 T021b:**
  `substrate/envelope_hash_gate.py --assert` (CI) holds the five files' sha256 against the
  approved baseline `tools/envelope_hashes.json`; an envelope byte change without its
  same-commit baseline update (the owner-approval act, reviewable in the diff) reds the
  build, with no bypass flag on purpose. Counted at landing 2026-08-20: live five-file
  positive control green; tamper, stale-approval and unbaselined reds all fired. The gate
  reads and compares — the envelope is never its target. The five brainstem files — `reflex.py`, `serial_arm.py`,
  `so101_arm.py`, `arm_config.py`, `interpreter.py` — are modified only with explicit approval; the reflex E-STOP and
  cortisol/safety paths are never relaxed. Every wired drive enters selection **after** the E-STOP
  early-return, so it is safe by construction (proven: max drive + distance 0 → EMERGENCY_STOP).
- **E-prop, not backprop (review-enforced; PARTIALLY mechanized since 2026-08-15).** Any backprop consideration pivots to **e-prop**
  (Bellec 2020) — a local eligibility trace times a broadcast learning signal; no BPTT, no weight
  transport — verified **cosine > 0.5** against an independent reverse-mode BPTT gradient. Verification
  anchor: `reasoning/eprop.py::Net.grad_eprop`, with the STDLIB tier's independent
  reverse-mode gradient at
  `::Net.grad_bptt` (cited by SYMBOL: the line cites here had rotted onto the CANDIDATE).
  This rule governed the project before it was written down (it is why v3.79 pivoted all learning to
  e-prop on measured evidence: 73.14% vs the repo's backprop SSM at ~32%), which was the gap — a rule
  enforced only by whoever remembered it.
  **A `.backward()` tripwire would be theatre:** it passed green while
  `cerebellum/jepa_world_model.py` performed explicit weight transport on every wake tick, and
  `hippocampus/ng_snn.py::fit` still runs a closed-form batch solve over its whole buffer each sleep
  tick. Neither uses autograd; both are non-local. Absence of `.backward()` is not evidence of
  locality — **that argument is unchanged and the tripwire is still not built.**

  **SCOPE — a PIVOT rule, not a census (amended 2026-08-18, specs/020, owner-directed; it NARROWS
  what this invariant asserts, which this file reserves to the owner).** This clause
  governs work that was about to use backprop. It is NOT a requirement that every learning rule
  carry a gradient cosine. MEASURED by `substrate/gradient_claim_gate.py`, which prints GOVERNED 149 of which 52 are
  NOT-A-RULE (enumeration imprecision), leaving **97** real governed learning rules: **17 owe a cosine and 80 do not** — 72 local, 5 controls, 2 direct solves, and 1
  REFERENCE (`cortex/eprop_kernel.py::bptt_grads`, the TORCH tier's independent gradient the
  others are compared *against*, owing none of its own — but `EXACT_GRADIENT` is SELF-DECLARED:
  the exemption holds only while the symbol RETURNS a gradient and never steps an optimizer;
  `tests/test_eprop_train_tier_unreachable.py::_trains_by_backprop` draws that line). Read as a blanket rule the clause would demand a gradient
  comparison from four rules in five that have no gradient to compare against. **Re-derive with the gate
  before relying on these figures**; the `specs/020` copy rots, this one is authoritative
  (constitution SI 8 restates none). **Scope
  classification rests on SELF-DECLARED `GRADIENT_CLAIM` entries**: a false declaration is
  caught only by REVIEW — the census stores both signals per rule, emits no disagreement report
  (one-time enumeration: SC-012), and no gate checks a declaration against the code.

  **INSTRUMENT_FALSIFIED is triggered by HIGH COHERENCE invalidating the null — NEVER by a rule
  scoring below the bar.** `verdict_for` keys it solely on the coherence PRODUCT `c_a * c_b`
  reaching the bar, where a within-matrix permutation stops being a null and the statistic inverts.
  A low cosine at ordinary coherence is `RULE_JUDGED`; an UNJUDGED claimant owes an owner
  adjudication here (a state, not an exit). **`RULE_JUDGED` is a
  COHERENCE-KEYED
  TOKEN, not a finding that the cosine is informative about that rule** -- it says only that the
  null survived. For a frozen-feedback alignment arm the 2026-08-14 non-discrimination argument
  (the statistic reduces to `cos(W2[a], B[a])`, near zero BY CONSTRUCTION) still stands
  unamended, and route (a)'s rejection with it. The two verdicts are about different things.
  **It never disposes of WEIGHT TRANSPORT**, which is adjudicated separately by
  `substrate/eprop_transport_gate.py` — two rules (explicit-transpose, index-swap), each with a
  constructed red, asserting ZERO governed findings (no ratchet, no baseline); its green never
  means "compliant" and it prints the spelling it cannot see. **Composition**: that gate
  asserts only on `dormancy_gate.BRAIN_MODULES` and REPORTS `reasoning/`/`training/` — so an
  in-scope trainer there has REVIEW as its only enforcement; neither gate can fail on it; likewise ON the governed tier
  for a false declaration plus an unseen spelling — review is the floor everywhere.

  **This said the OPPOSITE until 2026-08-18** (refutation: census-baseline.md).

  **Which autograd trainers are in scope: LIVE-PATH REACHABILITY.** A trainer is inside this clause
  when the UNMASKED transitive closure of `dormancy_gate._LIVE_ROOTS + _ENTRYPOINTS +
  _external_roots()` reaches its module — the transitive closure FROM
  `dormancy_gate::reachable`'s seed set, computed WITHOUT its mask (`_external_roots()` covers
  repo-root scripts and `integrations/`, where `substrate.learn_daemon` enters). **There is NO train-tier exemption**: `snn_data_lake.training.cli` is itself an
  `_ENTRYPOINT` (measured 2026-08-18), so the `training/` tier is REACHED and IN SCOPE. An earlier blanket permission
  here was false when written (this is the COSINE clause's reach, distinct from the transport
  gate's per-symbol exemption bucket).

  **`dormancy_gate.reachable()` CANNOT decide this** — it ends `return seen & BRAIN_MODULES`,
  masked to the governed tier, silent about `reasoning/`/`training/`, where most trainers live. Until an unmasked predicate exists, scope here is a REVIEW
  JUDGMENT against the closure above — no gate computes it, and the closure follows STATIC
  imports only: `importlib`/`__import__`/string-dispatch reaches are invisible to it. "Live path" HERE
  means that entrypoint closure — strictly WIDER than "on the live tick" in the site verdicts
  below; no site verdict is reopened by the wider reading.

  **Reachability is a RATCHET**: reachable puts a trainer IN scope; unreachable does NOT take
  it out without a fresh adjudication in THIS file — exiting is the unreviewed direction.

  **OPEN HOLE.** On the GOVERNED tier a flip IS detected for the hazard shape
  `tests/test_eprop_train_tier_unreachable.py` keys on (`.backward()` + `.step()`, one function)
  — reds when such a trainer becomes reachable (the test asserts the PROPERTY; its one-time
  docstring numbers drift);
  a trainer split across functions is outside that
  detector's shape. Closure lane: an unmasked predicate in `dormancy_gate` plus a wider hazard
  shape in that test — neither exists yet. Closure UNSCHEDULED (2026-08-18, deferred without an owner decision; no
  measurement says the risk is tolerable); revisit at the next promotion of a hazard-holding
  `reasoning/`/`training/` module. On the UNGOVERNED tiers — `reasoning/`,
  `training/`, where most autograd trainers live — **nothing detects a flip**: a wiring edit can
  move a trainer into scope with no gate, test or report firing. Blast radius: every ungoverned
  symbol declared on reachability grounds; the over-flagging (Standing Invariant 3) is a
  mitigation, not a fix. MEASURED 2026-08-18 (`src/` only, `.backward()` + `.step()`, one
  function): **21 symbols**; 17 governed claimants declared. Enumeration:
  `specs/020-eprop-cosine-scope/census-baseline.md` — this clause states the rule, no list.

  **Provenance of `0.5`, recorded rather than moved.** The null is a **within-matrix permutation** of
  the candidate gradient, cosined against the unchanged reference. On the stdlib anchor
  (`reasoning/eprop.py::Net.grad_eprop` vs `::Net.grad_bptt`) the aligned cosine is **+0.9481** and
  `0.5` sits **+3.06 sd** above the null with **P(null > 0.5) = 0.0000** — inside the separating
  region. P1 case (a): a fixed point on a normalized dimensionless output. **Two disclosures:** The reference recomputed ONCE against shuffled labels scores **+0.6500, above
  the bar** (N=1) — the null chosen decides whether `0.5` discriminates at all.
  And the permutation is a null only near zero mean: once the coherence PRODUCT `c_a * c_b` reaches
  the bar, the null sits ABOVE it and the statistic INVERTS.
  **The bar is UNCHANGED by this amendment.**

  **What IS built, 2026-08-15: `substrate/eprop_transport_gate.py` (CORE), which gates the
  TRANSPORT rather than the autograd call.** This row read *"It is NOT mechanized"* until then, and
  that half is now false while the tripwire half above is not — they were one sentence and are
  separated here deliberately. The gate asserts that no **governed** module reads a weight
  transposed into a weight update, encoding this file's own adjudication test (*"THE DISTINGUISHING
  TEST IS WHAT THE TRANSPOSE FEEDS"*) rather than a proxy for it: the reconstructed pre-fix jepa
  site fires, and `predictive_coding`'s live `w1.T` — which feeds an **activation** — does not.
  Counted before adopting, on the governed tier: any transpose **82**, transpose of a weight **36**,
  transposed weight reaching a weight update **0**. It lands green, so per the rule below it ships
  the reconstruction of the defect it was built for.

  **WIDENED 2026-08-15, at the owner's instruction, and limit (a) below is now PARTLY closed.** Two
  defects were found in the gate itself, both by asking what it could not see rather than by reading
  it. **(1) Its weight-name predicate missed the names it most needed:** `Wrec` / `Win` / `Wout` --
  `reasoning/eprop.py`'s actual weights -- were not weight-ish at all, and **`W2` was rejected as a
  CONSTANT** because `"W2".isupper()` is True, `W2` being the exact name in `mlp_q`'s retired
  transport that this row cites as the gate's miss. The constant exclusion is NARROWED rather than
  removed (a real constant is caps-and-underscores with two or more trailing caps, so `W_MAX` still
  excludes and `W2` no longer does); **re-counted before adopting, the shipped rule scores GOVERNED
  0 / UNGOVERNED 1 under both predicates, so widening cost zero findings.** **(2) A SECOND RULE,
  INDEX-SWAP, now catches the loop spelling:** an explicit transpose is an ARRAY-LIBRARY idiom, and
  `reasoning/eprop.py` hand-rolls reverse-mode BPTT in pure `math`/`random` loops with no numpy and
  no `.T` anywhere, so that file was invisible to the walk **entirely** -- not a finding, not a
  census entry. The rule keys on the forward order being the one the module ASSIGNS to, so a read in
  the reversed order is a transpose, and a module that only COMPARES both orders (`neural_sampling`
  raises unless `W[i][j] == W[j][i]`, ENFORCING symmetric coupling) establishes no forward order and
  drops out by construction rather than by an allowlist. **Counted before adopting: 1 finding**,
  `reasoning/eprop.py`'s `Wrec`, which is NOT a violation -- `Net.grad_bptt` is the independent
  reverse-mode gradient this invariant's own cosine clause REQUIRES -- and is ungoverned, so it
  reports. The governed half asserts, so a module hand-rolling backprop in loops no longer passes
  silently on the day it is promoted. **Two earlier drafts were discarded at this bar and the
  discards are the useful part:** "both index orders appear anywhere" scored 2 with the symmetry
  validator as a false positive AND missed its own motivating case; bolting the new rule onto the
  existing terminal test ("the transport must reach a weight update in the SAME function") was
  rejected on design, because `grad_bptt` RETURNS gradients and `apply()` performs the update.

  **Three limits, stated because a green here is easy to over-read, and the gate prints all three:**
  (a) its rules key on an EXPLICIT transpose or a reversed index PAIR, so the `mlp_q`
  violation — spelled `w2i = W2[a]` then
  `dh = err * w2i[j]`, with no `.T` in the source — **would not have been caught**; of the two
  violations this tree has had it catches one. (b) The wider rule that catches both **was built and
  rejected on its own output**: "updates weight A while reading a different weight B" scored 6 of 6
  on the reconstructions and **59** on this tree, including `mlp_q::update` itself — the module whose
  transport was *retired* — because every local rule reads its own forward weights, and whether a
  second matrix is transported or independently maintained (Kolen-Pollack's `B` takes the read-out's
  increment and never reads it) is semantic, not syntactic. (c) It asserts only on the governed
  tier and **reports** the rest. Re-derived 2026-08-18: the EXPLICIT-TRANSPOSE rule reports
  none — its one finding, `reasoning/gf.py::MLP.train` (`gh = (g @ self.w2.T) * (1 - h**2)`
  into `self.w1 -= lr * gw1`, hand-written backprop no marker grep can see), is **EXEMPT** under
  the 2026-08-15 owner adjudication — while the INDEX-SWAP rule reports one ungoverned row,
  `reasoning/eprop.py`. The exemption converts to a governed violation the day `gf.py` is promoted —
  the clause attaches to the exemption, not to a report.

  **Verdict on both, since "review-enforced" had so far meant "unadjudicated" (2026-07-28), with the
  first now CLOSED (2026-08-04):**

  **SCOPE CORRECTED 2026-08-13 — "both" was itself the defect.** This verdict has only ever covered
  the two sites below, but it reads as a survey of the tree, and that is how it has been used: with
  the first site CLOSED, the natural conclusion is that the invariant holds. It does not follow. Two
  further sites carry the same shape and **neither has ever been adjudicated here**; they are recorded
  below as OPEN QUESTIONS, deliberately WITHOUT a verdict. Deciding them is a policy call — this pass
  is a scope correction, not a decision, and an invariant is the one thing no lane may relax in
  passing. The transferable rule: **a verdict that enumerates is read as a verdict that surveys**, so
  a closed row inside it silently discharges sites it never examined.

  - ~~`cerebellum/jepa_world_model.py` is a **real violation of the locality principle, on the live
    wake tick.**~~ **CLOSED 2026-08-04.** The transport half is gone: `_feedback_gz()` is now the ONE
    backward definition for both the encoder path and the rollout path (P2 — one mechanism, not two),
    and it returns `list(gp)` — the identity/skip half alone, which is a broadcast error signal of
    exactly the shape e-prop asks for. **And the remedy THIS POLICY PRESCRIBED was built first and
    REFUTED, which is the part worth keeping.** Feedback alignment (a fixed random `Bz`) was
    benchmarked over 1200 train / 350 held-out transitions, 5 seeds: held-out invariance MSE ratio to
    the transport incumbent was **0.657 for FA against 0.654 for simply dropping the recurrent term**,
    FA won on only 2 of 5 seeds, and six independent `Bz` draws all landed in 0.726–0.739 — so the
    random matrix contributed nothing measurable and *the entire ~34% gain is the removal of the
    transport term, not feedback alignment.* Shipping `Bz` would have bought a new matrix, a new seed
    stream and a new constant with a null (P2: a merge must REDUCE; P8 guard (d): report the null).
    **Recorded rather than shipped — a policy prescribing a fix does not make the fix correct, and
    the measurement outranks the prescription.** Cosine(FA, transport) was 0.971, far above the 0.5
    e-prop bar, which is what makes the null trustworthy rather than an artifact of a broken control.
  - `hippocampus/ng_snn.py::fit` is **not** a violation. Greedy matching pursuit with an
    analytical beta per block is a direct solve: no gradient is propagated through a network and no
    weight is transported. It IS non-local, and the policy should say so distinctly rather than
    lumping it in — "non-local" and "backprop" are different claims, and only the second is what this
    rule forbids. Disposition: keep, and stop citing it as a pending violation.
  - **OPEN QUESTION, added 2026-08-13, NOT adjudicated: the `dh` step inside `basal_ganglia/mlp_q.py::MlpQ.update`** (cited by SYMBOL, per this file's own rule below — the line cite `:190` written here on 2026-08-13 was broken THE SAME DAY by a +6-line docstring edit in that very file, which moved the statement to `:196`). `dh = [err *
    w2i[j] ...]` reads the FORWARD read-out weights `W2[a]` transposed to carry error backward into
    the shared hidden layer. The module **self-declares this** in its own module docstring (the "OPEN e-PROP VIOLATION, NAMED HERE BECAUSE THIS MODULE PERFORMS IT" paragraph) as an open e-prop violation
    and names this policy — yet `mlp_q` appeared **nowhere in this file** until this entry added it, so the site had been
    disclosed in the code and invisible in the policy record since it was written. It is **not on the
    live tick** (`AutonomousArm` defaults `mlp_q=False`) but is one constructor argument from it, and
    it executes on every gate run via self-check 19, so it is latent rather than absent. **The
    measurements already exist** (the module docstring's "MEASURED, so the fix is a decision rather than a guess" block). `main()` RE-DERIVES the transport and LatentQ columns and the cosine sweep on every gate run; the FEEDBACK-ALIGNMENT ACCURACY row is NOT re-derived -- that twin is a SCRATCH benchmark the module deliberately does not ship, and says so in the same block: feedback
    alignment clears both of the module's shipped gates on all 8 draws and *gains* 1.0 point on the
    linear rule, but its hidden-gradient cosine sweeps to **[-0.4366, +0.1844] with |cos| > 0.5 in 0
    of 8** — so swapping trades one failed clause of this invariant for another, which is not a fix.
    (**Interval corrected 2026-08-18**; census-baseline.md. The PROPERTY — 0 of 8 clear — is what
    the argument rests on; the test asserts it, pins no interval.) The
    two honest resolutions the module states: **(a)** accept FA and amend what "cosine > 0.5" means
    for a non-gradient *alignment* rule, or **(b)** retire the head under P2 for the linear `LatentQ`,
    which has no hidden layer and therefore no backward pass at all — at a measured cost of the
    non-linear capability (**0.467 vs 0.993 on XOR**), which P2's honesty guard requires be ported
    first. Recorded here so the decision is made against the measurements rather than rediscovered.

    **RESOLVED 2026-08-14 (specs/015-resolve-weight-transport) — and by NEITHER of those two
    routes.** The transport is RETIRED. `dh` now reads a separate feedback matrix `B` that receives
    the SAME increment as the forward read-out (`B[a][j] += lr*err*h[j]`), so `B` TRACKS `W2`
    without ever reading it — `err` and `h[j]` are already local to that synapse. This is
    **Kolen-Pollack** (Kolen & Pollack 1994), NOT the Spiking Forward DFA that prompted the lane:
    SFDFA estimates its feedback by propagating directional derivatives as SPIKE GRADES, and `MlpQ`
    is a rate MLP with no membrane, threshold, spike or time index, so SFDFA's actual contribution
    has nothing here to attach to. What transferred is its ESSENTIAL property — feedback ESTIMATED
    and CONVERGING rather than transported or frozen. Naming the shipped rule "SFDFA" would have
    been a citation defect.

    **The transferable rule survives here; the full route-rejection narrative and the 015
    measurements are relocated BYTE-EXACT to `docs/POLICIES_EVIDENCE.md` (specs/022 T003):** an
    instrument no alignment rule could ever pass is not evidence against alignment rules (the FA
    cosine reduced to `cos(W2[a], B[a])`, near zero BY CONSTRUCTION), and route (b) was never
    needed. Headline numbers only: converging `B` 8/8 vs frozen 0/8, permutation null 0.0000, no
    capability lost, no new constant.

    **The invariant was NOT amended TO RETIRE THIS TRANSPORT, and did not need to be** -- route
    (a) was unnecessary because Kolen-Pollack clears the bar. **Scoped 2026-08-18:** the SCOPE
    paragraph DID later amend this clause, on the different question of which rules owe a cosine
    at all; it classifies this arm `RULE_JUDGED` — the gate's coherence-keyed token, meaning THE
    NULL SURVIVED, not a finding that the arm failed. The 2026-08-14 non-discrimination argument
    is unamended and still carries route (a)'s rejection.
    `main()`'s check (5) separates three ways on every gate
    run (transport identity / shipped learned `B` / frozen control) — re-run it for the values. **One prediction was wrong and is recorded rather than smoothed**: the spec
    predicted `B` would "rise monotonically from ~0"; it is at 0.92 after ONE epoch and drifts
    slightly DOWN. Alignment is established almost immediately and then tracked, not developed over
    training — the direction was right, the mechanism model was not.
  - **OPEN QUESTION, added 2026-08-13, NOT adjudicated, and the LIVE one: `cortex/predictive_coding.py:200-201`.**
    `g1 = self.w1.T @ s_e0` / `g2 = self.w2.T @ s_e1` — the forward weight array, transposed, carrying
    error backward. Unlike `mlp_q` this **runs on every wake tick** (`cortex/brain.py:235` constructs
    the brain with `predictive_coding=True`), which is the exact property that made the jepa site a
    violation. **CORRECTED 2026-08-14: it is NOT undisclosed in its own module, and this entry said it was.** `cortex/predictive_coding.py` writes the transpose out in its own prose FOUR times -- the module docstring (`r1 += lr * ( W1^T @ s_e0 - s_e1 )`), `infer_step`'s docstring, the inline comments on the two cited lines themselves, and again in the batch path. What was undisclosed is the POLICY RECORD, not the code -- which is precisely what the finding cited below concludes: its own heading reads "TITLE OVERSTATES CONCEALMENT". This entry cited that finding approvingly while asserting the framing it retracts. Half-quoting a source to support the claim it corrects is the same defect the P0 row above records against `findings.md:370`.
    **The question is real in both directions and must not be settled by
    analogy.** For: exact weight symmetry by construction — the same array, transposed — is the
    objection this invariant exists to name. Against: in Rao-Ballard, `W1` is the GENERATIVE weight
    (`x_hat = W1 @ r1`), so `W1^T @ s_e0` is one matrix in its two structural roles rather than a
    separate forward pathway being copied, which is the standard locality argument for predictive
    coding. `specs/010-policy-compliance-audit/findings.md:826` already reached this site, called it
    *"an undisclosed VIOLATION SITE, not concealed code"*, and observed that this verdict "covers
    exactly two sites" with `predictive_coding.py` appearing in neither — **that finding was never
    carried into this file, which is how a live site stayed unadjudicated for nine days.**

    **ADJUDICATED 2026-08-14, at the owner's instruction: NOT a violation of this invariant.** The
    verdict rests on a mechanism read out of the module, not on the Rao-Ballard analogy, and not on
    the Predictive E-prop preprint — which addresses BPTT and has **0** full-text hits for weight
    transport / weight symmetry / feedback alignment, so its silence is silence.

    **THE DISTINGUISHING TEST IS WHAT THE TRANSPOSE FEEDS.** In both sites this verdict ruled
    violations, a transposed forward weight fed a WEIGHT update — jepa's `Wz^T gp` entered the
    predictor's gradient, and `mlp_q`'s `W2^T` produced `dh`, which updated `W1`. That is credit
    assignment to weights through a transported weight, which is exactly what "e-prop, not backprop"
    forbids. `predictive_coding` does not do it:

    - `g1 = self.w1.T @ s_e0` feeds `self.r1 = self.r1 + lr * (g1 - s_e1)` — an **ACTIVATION**.
      It is INFERENCE, settling a transient latent state, not credit assignment.
    - the **weight** update is `self.w1 += lr_w * np.outer(self.cum_e0 / self.n_steps, self.r1)` —
      a **local Hebbian outer product** of a local error rate and a local cause. **No transposed
      weight appears in the learning rule at all.**

    The invariant as this file states it governs LEARNING RULES — "local eligibility trace ×
    broadcast learning signal, verified cosine > 0.5 against an independent BPTT gradient". There is
    no gradient propagated through a network here and no weight transported into an update, so there
    is nothing for the cosine clause to be measured against. Same disposition as `ng_snn::fit`, and
    for the same structural reason.

    **WHAT THIS VERDICT DOES NOT SAY, stated so it is not read as broader than it is.** `self.w1.T`
    IS exact weight symmetry, on every wake tick, and a physical synapse is unidirectional — so
    realising that transpose still requires a symmetric backward pathway. That is a real
    biological-plausibility question. **It is simply not the question this invariant asks**, and
    widening an invariant is an owner decision, not something an adjudication may do on its own
    authority. Recorded here as OPEN SCOPE, distinct from the closed violation question: if the
    invariant is ever extended from learning rules to inference pathways, this site is the first
    thing it catches.

  - **ADJUDICATED 2026-08-15, at the owner's instruction: `reasoning/gf.py::MLP.train` performs
    weight transport, and is NOT a violation of this invariant — it is a CONTROL.** For where this finding surfaces in the
    gate's output, RE-RUN IT — restated tallies rotted here twice.

    **The transport is real, not arguable.** `gh = (g @ self.w2.T) * (1 - h**2)` carries error
    through the transposed forward read-out into `gw1 = xb.T @ gh`, then `self.w1 -= lr * gw1`, so
    **the transpose feeds a WEIGHT update** — the jepa/`mlp_q` shape, the opposite of
    `predictive_coding`, under the distinguishing test above. Hand-written reverse-mode with no
    autograd, which is why `tools/promotion_sweep.py`'s marker grep cannot see it. **Scope
    MEASURED, so this verdict does not enumerate where it appears to survey:** `gf.py` contains
    exactly THREE transposes, all inside `MLP.train`; `h.T` and `xb.T` transpose an ACTIVATION and
    an INPUT, and exactly ONE (`self.w2.T`) transposes a weight. There is no second site in the file.

    **The verdict's kernel survives here; the full reasoning is relocated BYTE-EXACT to
    `docs/POLICIES_EVIDENCE.md` (specs/022 T003):** not a violation on the invariant's own scope —
    code whose JOB is to be measured against is not one of the brain's learning rules; `MLP` is a
    Gc BASELINE (not the gradient instrument, which is `reasoning/eprop.py::Net.grad_bptt`) that
    **exists to LOSE** (`mdl_ok < ind_ok`), and porting it to e-prop would VOID the experiment —
    P8's contaminated control. Zero references outside `gf.py`, on no tick. Its missing
    self-declaration was corrected 2026-08-15 (P8 clause 4).

    **THE EXEMPTION, GRANTED 2026-08-15, and what was built so that it cannot rot.**
    `eprop_transport_gate` asserts on the GOVERNED tier, so promoting `gf.py` — backlog row 40 calls
    it the "pure leaf" that moves FIRST — would have converted this report into a governed RED.
    The exemption carries the verdict above as its reason and — Phase 7 having retired the
    gate's `EXEMPT` dict — lives in `reasoning/gf.py::GRADIENT_CLAIM["MLP.train"]`, read
    through the registry, so the ARC promotion cluster is unblocked. **Disclosed: this DECENTRALIZES the channel**
    — a module can self-grant by naming the gate in its declaration reason. `stale_exemptions`
    catches a grant blessing nothing; one blessing a REAL finding, only review catches. The owner
    grant remains the rule, mechanized as the marker (closure lane: an exemption-count
    ratchet, unbuilt). **Adding it NARROWED what a mechanized
    gate asserts, which is why it required the owner and not an adjudication.**

    Staleness is asserted BOTH ways, matched against ALL findings — the grant PRECEDES the
    promotion it unblocks, so a governed-only match would read STALE the day it was correctly
    written — and both directions were PROVEN by making them fire: a planted exemption matching
    nothing reds the gate (re-run it for the exact wording; a verbatim quote rotted here once). The bar for a future entry is stated at the site: a CONTROL is the instrument
    this invariant's own verification clause requires; **a faculty is not, however far off the tick
    it sits** — `mlp_q` was one constructor argument from the live tick and its transport was
    RETIRED, not exempted.

  *(Line citations in this bullet — `jepa_world_model.py:275-278`/`:277` and `ng_snn.py:137` — were
  dropped 2026-08-04 after all three failed to resolve to what they claimed. `ng_snn.py:137` is
  `mu = S.mean(dim=0)`. Worse, the jepa file was edited BETWEEN two reads in the same audit session,
  and its line numbers moved by one under me: a line citation into an actively-developed file is
  stale on a timescale of minutes, not releases.)*
- **Verify-first + honest framing:** probe → act → prove (A/B / gate) → commit; report the true number
  including a regression; reject theater with evidence; never fake a capability, a metric, or a consumer.
- **Runtime evidence outranks static reading.** Every decisive finding in the 2026-07 audits came from
  *running* the system, and the static reads were wrong in both directions. Monkeypatching
  `torch.Tensor.backward` over 80 live ticks proved no backprop on the tick (0 hits) where grep could
  only have found call sites. A neuter-A/B on `pc_surprise` proved it decisive (59/60 ticks changed
  `reentrant_strength`). A matched **control** proved predictive-coding learning real — error −67.8%
  with the update, +9.1% *without* it. Conversely a `cProfile` ranking sent a whole optimisation plan
  after the wrong function: 244,768 calls to `lif_neuron.step` looked like the hot path, and neutering
  it entirely moved the tick 43.6 → 41.2 ms (a 6% ceiling — per-call profiler overhead attributed as
  time). **Before optimising a hotspot, neuter it and measure the delta; before believing a signal is
  live, neuter it and watch what changes.**
- **A performance regression can be correctness-green.** v4.64 put a ~1 s static-import-graph census
  inside `snapshot()`, which the cockpit polls at 1 Hz — so the dashboard made the daemon re-scan its
  own source every second, starving the tick loop it existed to display. Every suite stayed green
  throughout, because self-checks call `snapshot()` a handful of times where a second is invisible; the
  cost only bites at rate against a live loop. It was found by watching the running system. **Anything
  that observes the brain must be cheap enough not to become the load** — and no gate here can tell you
  that.
- **Adding a policy: implement the gate before adopting the rule.** A proposed mechanization must be
  built and run against this tree, and its output counted, *before* the policy lands. Two proposals died
  exactly there — 0 failures / 47 UNCERTAIN, and 58 failures where 1 was claimed. Also required: quote
  the source line the rule derives from (a general good-practice rule in a costume is a reject), and
  name the real past incident it would have caught, or state plainly that there is none and say what it
  earns its place on instead. Fewer, sharper policies win; "no new policy" is a complete result.
  **Amended 2026-07-29, because P6 broke this and leaving the invariant unqualified would have made this
  file contradict itself.** The owner may adopt a policy with its gate UNBUILT. Such a policy lands
  marked **provisional** in the enforcement table, must still meet every other requirement above
  (quoted source line, named real incidents, each re-verified against the tree), and must carry the
  gate's specification so the debt is actionable rather than perpetual. The gate-first rule still binds
  anyone proposing a *mechanization*: the two proposals that died at this bar died on their own output,
  and that outcome — "the gate we built kills the rule" — remains the point of building it first.
  **Three entries have now used the provisional route — P6 (2026-07-29), P7 and P8 (both 2026-08-04) —
  and each says so in its own text.** That is a count worth watching rather than a precedent worth
  spending: "provisional" is a debt, not a category. (**Superseded in place 2026-08-18**: this said
  three gates were unbuilt and P8's cheapest to build first — the P7/P8 gates ASSERT since
  2026-08-05, per the enforcement rows; PROVISIONAL stays because a ratchet asserts "no worse",
  never "satisfied".) **What the route does NOT relax is the evidence:** all
  three met the quoted-source-line and named-incident requirements in full, and P7 and P8 were each
  written from a sweep whose findings were adversarially re-verified against the tree before any of
  them entered this file — five of sixteen candidate claims were refuted at that step and are not
  cited here.
- **A third proposal died at this bar, 2026-07-30: "a tracked file may not cite an untracked `.py`
  instrument."** It had the best motivation of any proposal here — three real incidents, two of them in
  one session: `POLICIES.md` citing a `scratchpad/smoke_drive_charge.py` that never existed, the same
  dead citation in `dormancy_gate.py`, and `DEPTH_CEILINGS.md`'s AR appendix citing nine gitignored
  `scratchpad/ar_lm_*.py` scripts as its evidence *hours after* the first two were corrected. It was
  built anyway, per this rule, and counted: **474 violations across 126 files, 195 in `CHANGELOG.md`
  alone**, from 1,431 citations. Allowlist-shaped, eight times worse than the 58-failure proposal above,
  so it is **not adopted**. The count also explained itself: a changelog citing a since-retired file is
  correct, because the entry describes a tree that existed. The instrument is kept as a report at
  `tools/citation_guard.py`, and the surviving narrower hypothesis is recorded there — the real defect is
  an untracked script presented as a *runnable reproduction path*, not a filename in prose. Building the
  naive version first is what distinguished those two, and it cost one file instead of a policy.
  **The narrow version was then built and counted too, and it SURVIVED at 21 violations across 5 files**
  (`tools/citation_guard.py`, STRICT scope — a `.py` cited inside a shell-fenced block or under a
  Reproduce/Usage heading). 21 was a work list, not an allowlist, so the rule was adoptable in that
  scope. What it found was worse than the incident that motivated it: **`docs/RUNNING.md`, the project's
  how-to-run page, had five commands pointing at files absent from a clone.**
  **FINAL STATE, same day: broad REJECTED at 475 across 127 files (report only, still non-asserting);
  strict ADOPTED and ASSERTING at 0 unexempted violations with exactly 1 exemption.** The 21 were
  cleared, not exempted, and the dispositions differed as predicted:
  - **10 were stale paths.** `research/` became `src/snn_data_lake/reasoning/` in v3.78 (`627b2e2b`), and
    `RUNNING.md` (×5), `JEPA.md` (×3), `PERFORMANCE.md` (×1) plus `HARDWARE_SETUP.md`'s sketch path were
    never updated — the files had *moved*, not gone. Corrected to `python -m snn_data_lake.reasoning.<mod>`.
    One was a genuine retirement: `spike_knowledge_graph.py` was merged into `spike_kg.py` (`17067032`).
  - **8 were promoted into `tools/`** from gitignored `scratchpad/` — real evidence for real published
    numbers, now tracked, with inputs taken as explicit paths and a clean SKIP when a dataset is absent.
    Each was run; `lang_scale.py`, `lang_diag.py` and `lang_probe.py` reproduce their doc tables exactly
    (53.4→53.1 decay ablation, 0/1675 fold collisions, 0.65% over 117,859 contexts, 2,992 params).
  - **1 was deleted, not promoted:** `ar_lm_untrained_control.py` no longer exists on disk, and the
    tracked bench builds the untrained twin anyway. Its command is gone and the doc now says the 49.5% is
    a recorded result, not reproducible from that file.
  - **1 exemption, `docs/HARDWARE_SETUP.md:144`.** The cited file IS tracked, at
    `integrations/robot arm/sensor_reading.py`; the SPACE in the directory name truncates the guard's
    path token, so it cannot resolve. Fixing that in the resolver was rejected because it would equally
    resolve the stale `robot arm/...` form the line carried until it was corrected — the tokenizer fix
    would hide a live defect class, so the false positive is exempted where the reason is readable.
    `EXEMPT` follows `dormancy_gate.EXEMPT`'s pattern and, because line-keyed entries rot, the gate also
    asserts every entry still matches a real violation: an exemption that blesses nothing fails loudly
    instead of reading as coverage. Both asserts were tested by making them fire.
  Registered as a step in `.github/workflows/ci.yml`'s lint job, not in `run_selfchecks.py`: the latter
  only executes `python -m snn_data_lake.<mod>` package modules, so qualifying would mean moving it into
  the package and churning the canonical 277 count quoted in five documents — to gate repo hygiene, which
  is not a brain-faculty self-check. Recorded here because "the broad rule died at 475 and the narrow one
  now holds the line at 0+1" is the result, and the only reason either number exists is that this bar
  demands counting before adopting.
- **Gated + reversible:** one change per commit, full gate green (`run_selfchecks` + `pytest` + `ruff`),
  commit + tag locally; a regression is isolated to its own change.

  **AMENDED 2026-08-16, owner: the local full-suite requirement is DELEGABLE TO CI — but only while
  CI is DEMONSTRABLY EXECUTING, and that is a measurement, not an assumption.** Adopted from
  DeepSeek Harness's *"never default to the full suite locally; CI owns exhaustive coverage and the
  platform matrix"*. The reasoning is sound and the saving is real: `run_selfchecks` takes ~16
  minutes and `pytest` ~14 on this machine, and rehearsing the platform matrix locally is
  impossible anyway — CI is the only place `ubuntu`/`windows` and the CPU torch build are exercised
  together.

  **THE PRECONDITION IS LOAD-BEARING AND IT WAS FALSE ON THE DAY THIS WAS ADOPTED.** When the
  amendment was proposed, GitHub Actions was executing **ZERO JOBS** — *"the job was not started
  because recent account payments have failed"* — across all five jobs of two consecutive runs on
  `main`. Adopting the delegation in that state would have meant the full suite ran NOWHERE: not
  locally by rule, not in CI by billing. Two commits (`4e7bfb8b`, `0b9ada6e`) reached `main` verified
  by the local gate alone. Billing was fixed and CI re-verified executing before this text landed.

  **HOW TO CHECK IT, because a status badge cannot tell you.** A billing-blocked run reports
  `completed / failure` exactly like a real test failure. The distinguishing signal is DURATION and
  JOB COUNT: this repo's real runs take **34-36 minutes**; the blocked ones took **5 seconds** and
  started 0 of 5 jobs. So the check is `gh run list` for the duration and `gh run view <id>` for
  jobs that actually ran — never the pass/fail word alone. This is the second time this project has
  been bitten by exactly that (see the standing note on a gate that never ran), which is why the
  amendment carries the method rather than the conclusion.

  **What does NOT change.** The full local gate is still required whenever CI is not demonstrably
  green on the change: before a merge to `main`, on any change to the safety envelope, and any time
  the CI/local skew this file already records could bite — CI runs a DIFFERENT torch build, so a
  build-sensitive numeric assert can be green in one and red in the other. "CI will catch it" is a
  claim about a run that has EXECUTED, in the same way *"it persists"* is a claim about a new
  process and *"it is pushed"* is a claim about a branch.
- **A `file.py:NNN` citation is a claim, and it is the least durable claim this project makes.**
  Added 2026-08-04 after a docs-wide audit resolved **every** `path.py:LINE` citation in `docs/`,
  `specs/`, `README.md`, `SECURITY.md` and `AGENTS.md` against the tree. Measured: **2,714 resolved,
  90 pointing at a file that does not exist, 27 pointing past end-of-file.** Worse than the miss rate
  is *where* the misses were — in this file, in the P0 avenue table, in the P1 safety-envelope
  paragraph (**three of its four cites wrong**), in the e-prop invariant, and in `SECURITY.md`'s
  description of the arm. The P0 growth row's cites, corrected the day before, had already drifted
  again.

  **`tools/citation_guard.py` does not catch this and cannot be blamed for it** — it resolves the
  FILE, deliberately and for a counted reason, and every finding above has a file that exists. Two
  further failure modes it also cannot see, both found in the same pass: a citation whose *file* is
  ambiguous (`executive_control.py` names two different modules, 411 and 645 lines), and a documented
  command whose file is tracked and which still exits 1 (`docs/RUNNING.md` had six —
  `tools/doc_command_guard.py` now gates that one, at 0 unexempted after 19 were cleared).

  **The disposition is NOT another gate, and that is a considered result rather than an omission.**
  A line-resolution gate would report ~117 findings on this tree, most of them in `specs/` and
  `docs/HARDWARE_SCOPE.md`, which are historical records where a citation into a retired file is
  *correct* — the identical shape that killed the broad citation rule at 475 findings. Allowlist,
  not work list. **The rule instead is editorial: cite the SYMBOL, not the line.**
  `neurogenesis_engine.substrate_is_retained()` survives a refactor; `neurogenesis_engine.py:321`
  survives until the next insertion above it. Where a line is genuinely needed, expect it to rot and
  re-derive it rather than copy it. Sharpest evidence for the rule: during this audit
  `cerebellum/jepa_world_model.py` was edited *between two reads minutes apart* and its line numbers
  moved by one underneath the auditor. **A line citation into an actively-developed file is stale on
  a timescale of minutes, not releases.**
