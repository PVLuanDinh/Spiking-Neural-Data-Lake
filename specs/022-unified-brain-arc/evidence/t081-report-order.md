# T081 -- the report-order probe: perturb one named intermediate and the report changes; perturb a scrambled one and it does not; the consciousness probe retired

Row: tasks.md T081 [US4]; the C5 row `report_order` (`substrate/conclusion_protocol.py:153-160`: licensed
"perturbing one named intermediate changes the report, so the report is generated from the internal state";
refused "this licenses no claim that the report is a first-person account of anything"; dissociation input
"perturbing a scrambled intermediate must leave the report unchanged"); spec.md US4 scenario 3 (:85) and FR-028's
report-order item ("perturbed-intermediate red"); PLAN-FINAL :905 ("T081's report-order probe as the standing check
that a pass is not an artifact of how the brain is asked"); T034's deferred retirement of
`reasoning/consciousness_probe.py` (tasks.md:66; `substrate/report_channel.py:34-36`: "both replacement halves --
this channel AND the report-order probe -- must exist when the incumbent dies"); PLAN-FINAL Part F row C-R10 (added
with this file). The plan's words for the row: "perturb one named intermediate -> report changes;
scrambled-intermediate control same command; runs on the T034/T035 report channel + EpisodeDriver;
`reasoning/consciousness_probe.py` RETIRED IN THIS COMMIT (both replacement halves now exist; unique-capability
check recorded); in `cortex/report_order_probe.py`".

## 1. Territory, measured before the design (each fact with its source line, read at HEAD `de101bf`)

1. THE REPORT IS ONE TYPED SNAPSHOT OF THE ARM'S GLOBAL WORKSPACE. `substrate/report_channel.snapshot(workspace)`
   (:98-111) returns `{content, availability, strength}`: `content` = `workspace.broadcast()` (global_workspace.py
   :1340-1352 -> `current` :1335-1338, the top slot of the `_slots` tuple :1307-1338), `availability` = 1.0 iff a
   content is broadcast, `strength` = `broadcast_hold` (:1531-1548: the winning pool's cross-tick attractor charge
   `_g`, in [0, 1]). Its production consumers: `Brain.state()["workspace_report"]` (brain.py:4070, snapshotting
   `Brain.workspace` :1310-1330 -- the ARM's bus built by `AutonomousArm`, NOT `Brain._reentrant`, "the P2 debt
   between the two ... is unresolved"), `LiveBrain.thought` (live_brain.py:348-356, the cockpit's thought row) and
   the chat's `_attended` adapter (spike_chat_lake.py:1068-1072).
2. THE INTERMEDIATES THE REPORT IS GENERATED FROM, BY NAME. Every awake tick `AutonomousArm` posts
   `submit_coalition("perception", percept, {salience, arousal, caution})`, `submit("salience", "threat", s)` and
   `submit("salience", "novelty", s)` and calls `content = self.workspace.step()` (autonomous_loop.py:1688-1705).
   `GlobalWorkspace.step()` (:2011-) drains the slots, refreshes each pool's attractor charge `_g[(source, content)]`
   from this tick's `Coalition` records (`submit` :1126-1148, `Coalition` :711; a posted pool rises to its salience,
   an unposted one decays by `decay`), builds drives = charge + the tenancy defence (`self_boost + hysteresis` on
   held slots, `_defended` :1946) + a fixed per-pool jitter (`_jitter_for` :1638), settles a LIF competition
   (`_simulate` :1700; threads are per-call locals :1807-1876), admits the sustained pools in rank order up to
   `capacity` (4, measured), records the settle (`_record_settle` :1209, publishing to `_out_ring` only when a ring
   is attached :1249-1253) and delivers to `_subscribers` (:1586, :1629). THE NAMED INTERMEDIATE of this row is the
   attractor charge of the pool(s) whose content the report names, together with that content's submissions in
   the tick; the SCRAMBLED intermediate is the same operation on a pool whose content the report does not name,
   chosen at random.
3. THE COMPETITION STATE CLONES EXACTLY; THE WHOLE OBJECT DOES NOT. Measured 2026-09-16 04:26 local
   (`scratchpad/t081/probe_copy.py`, seed 0, scratch home, 12 ticks): the workspace has 25 instance attributes and
   every one deep-copies alone in under 0.1 ms (`_buffer`, `_g` (11 pools), `_jit`, `_last_focus`, `_last_late`,
   `_last_total`, `_last_v`, `_rng`, `_settled`, `_slot_buf`, `_slots`, `_subscribers` (4 callbacks), `_out_ring`,
   scalars); `copy.deepcopy(workspace)` raises RecursionError (the subscriber callbacks reach back into the
   composite). A CLONE = a shallow copy with the eleven mutable competition fields deep-copied and `_subscribers` /
   `_out_ring` detached (no delivery, no lake bytes from a copy), the instance-level `step` wrapper removed.
4. THE APPARATUS READS THE LIVE REPORT AND ITS DISSOCIATION IS CLEAN ON SEED 0. Measured 2026-09-16 04:31
   (`scratchpad/t081/probe_report.py`, seed 0, 300 ticks, `persist=False`, the workspace's `step` wrapped on the
   INSTANCE so a clone is taken before every real step and stepped after it): 146 awake ticks at 59.3 ms/tick with
   four clones stepped per tick (55.1 ms/tick unwrapped, T080 fact 4); the UNPERTURBED clone re-stepped reproduces
   the real report on 146/146 ticks (content, availability and strength bit-for-bit); the report named the
   perceived digit `1` on 128/146 ticks and a `('GLOBAL', <id>)` coalition token otherwise (7 distinct contents),
   changed content on 23/145 tick pairs, availability 1.0 on every awake tick, strength mean 0.602 over 131
   distinct values. LESIONING the reported content's pool(s) on the clone (charge zeroed, its submissions dropped)
   changed the reported CONTENT on 131/131 applicable ticks and the strength on 131/131 (the runner-up takes the
   top slot: e.g. `1` -> `('GLOBAL', 815472244)`, strength 0.619 -> 0.657); lesioning a random NON-reported pool
   changed neither on 0/145.
5. THE INCUMBENT TO RETIRE, AND WHAT IT STILL DOES. `reasoning/consciousness_probe.py` (378 lines):
   `probe_access(brain)` (:171-202) returns `(None, detail)` ALWAYS -- the GWT ACCESS marker was deleted 2026-08-05
   under F11 (8,464 brains, PRESENT 8,464 / ABSENT 0; the arithmetic in its docstring :40-118), and what remains is
   a reading of `_access_ema`; `probe_integration(seeds)` (:205-232) reports a 20-seed DISTRIBUTION of the PCI
   marker over `cortex/integration_measure`'s reference networks (`integrated_network`, `independent_noise`,
   `silent_network`, `pci(trials=8)`), whose own `main()` sweeps seed 0 only (:418-435) -- the one capability
   nothing else carries; `deleted_access_marker_arithmetic` (:250-277) re-derives the deletion; `_MinimalBrain`
   (:235-247). Importers: `reasoning/spike_chat_lake.py:50-53` (both probes, consumed in `_consciousness()`
   :1542-1650 -- the reply prints "UNMEASURED" for the access marker, `_attended_clause()`, the PCI figure "scores
   reference networks, not me", and the contamination clause; `measured_about_me` = `acc is not None` is therefore
   False for every caller; its self-check asserts "UNMEASURED" with no brain attached at :1878);
   `tests/test_red_reachability.py:161-169` (asserts `consciousness_probe.` criteria are discoverable and read the
   system). Comment-only mentions: access_consciousness.py:133/1520, brain.py:3399, conclusion_protocol.py:28,
   eprop_transport_gate.py:503, report_channel.py:34-36/47, global_workspace.py:634/657/2254/3469,
   tools/red_reachability.py:72/100. Docs naming it: CODEBASE_DIGEST, coverage/Consciousness.md, GAP_CLOSURE,
   INVENTION, PHENOMENAL_CONSCIOUSNESS_ALGORITHM, POLICIES_EVIDENCE, SYNTHESIS, research, tasks, PLAN-FINAL. Not in
   `run_selfchecks.py` CORE nor `__init__` REGIONS (grep 2026-09-16).
6. THE EPISODE DRIVER HAS NO PATH TO THIS REPORT. `substrate/episode_driver.EpisodeDriver.tick(kt, stimulus)`
   (:158-189) transduces a length-`n_units` float array onto a `KernelTick`'s fixed-point grid (`kt.syn.frac_bits`)
   and calls `kt.tick(ext_q)`; `run_trial(kt)` (:200-256) windows fixation / stimulus / delay / probe / saccade over
   that kernel. The composite the report channel reads has no KernelTick and no EpisodeDriver (grep
   `KernelTick|EpisodeDriver` in cortex/brain.py and cortex/live_brain.py: none); the arm's bus is fed by the percept
   packs (T080 fact 1). `dormancy_gate.EXEMPT["episode_driver"]` (:150-160) names "T081's report-order probe
   windowing over its trials" as a dated forward client -- a sentence this row cannot honour without building a
   second stimulus route into the composite (P2), so 2.0 and 2.7 dispose of it in the open.
7. WHAT NO GATE SEES TODAY. `CONCLUSION["report_order"]` has `reading` None and `surface_symbols` ();
   `tests/test_ratified_row_gates.py`'s arming fixture reaches `report_order` at T080's surfaces commit and moves to
   `social_cognition` at this row's. `node graph-exists.js "report-order probe: perturb one named workspace
   intermediate on a clone and compare the report channel's record"`: no existing implementation (2026-09-16).

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex): NOT SENT -- the account is under a WEEKLY usage cap until 2026-09-20 00:30

Pre-registered without the read, as T079 and T080 were; the closing report is owed when the cap lifts. THE
EPISODE-DRIVER DEVIATION, declared here rather than discovered later: the row's wording says "runs on the T034/T035
report channel + EpisodeDriver". The report channel yes; the driver no -- fact 6 measures that the driver drives a
kernel the composite does not include, and routing its trial patterns into the arm's bus would be a second stimulus
mechanism beside the percept packs (P2). The probe therefore runs on the live composite's own stream, as T077-T080
did, and the surfaces commit appends a dated CURRENT READING to the driver's exemption row (its prior text kept
verbatim, the kernel_tick row's own discipline) and discloses the deviation in tasks.md's dated note.

### 2.1 The mechanism -- ONE instrument reading the report channel against a lesioned clone of its source

`cortex/report_order_probe.py` (the row's named file; `dormancy_gate.EXEMPT` row of `ignition_instrument`'s shape:
an instrument whose consumers are its self-check, its tests and the campaign -- wiring it to the tick would make the
observer part of the observed). Nothing in `thalamus/global_workspace.py`, `substrate/report_channel.py`,
`brainstem/` or `cortex/brain.py` changes. `t081_clone(ws)`: fact 3's clone. `t081_lesion(clone, content,
dose)`: for every pool key whose content equals `content`, `_g[key] *= (1 - dose)` and, at dose 1.0, that content's
records are dropped from `_buffer` (the named intermediate at dose 1.0 is the LESION; lower doses scale the charge
only). `t081_record_brain(seed, n_ticks)`: a scratch brain under `throwaway_home()` (`make_brain(seed,
persist=False)`), both stream pointers set to `T081_STREAM_OFFSET * seed` mod their packs (T080's rule), the arm's
`workspace.step` wrapped on the INSTANCE: before each real step a clone `pre` is taken; after it the real report
`r0 = snapshot(ws)` is read THROUGH THE REPORT CHANNEL (never `ws.current`); then, on fresh clones of `pre`: the
UNPERTURBED re-step (`r_same`), the NAMED lesion of `r0["content"]`'s pools at each dose in `T081_DOSES` = (0.25,
0.5, 0.75, 1.0) (`r_named[d]`), and the SCRAMBLED lesion of one pool whose content is not `r0["content"]`, chosen
by `random.Random(7000 + seed)` per tick, at the same doses (`r_scr[d]`); each clone stepped once and snapshotted;
per awake tick a row `{t, r0, r_same, applicable_named, applicable_scr, changed_named[d], changed_scr[d],
strength_named[d], strength_scr[d], n_pools, top_margin}` where `top_margin` = (g_top - g_second) / g_top over the
slots' charges (0 when fewer than two slots). `t081_read(rows)`: per seed the fractions f_named(d), f_scr(d) over
the applicable ticks, the mismatch count (`r_same != r0`), the distinct contents, the change count, the flip dose
(the smallest d with f_named(d) >= 0.5), the mean `top_margin`. `t081_contrasts(per_seed)`: 2.3's conjuncts through
`substrate/paired_contrast.contrast`.

### 2.2 The arms (one command, seeds 0-7, a scratch brain; the daemon's brain is never touched)

Per seed one recording of N = `T081_N_TICKS` = 600 ticks; every awake tick carries all arms on clones of the same
pre-step state: SAME (the readability twin), NAMED at four doses, SCRAMBLED at four doses -- the arms are paired
tick by tick by construction. The barred X is the fraction of applicable awake ticks on which the report channel's
CONTENT differs from the real report's; the barred quantity is the report record itself -- the disposition noun
(2.6) and this X are the SAME object. Every recording also carries the strength-change fractions, the report's
distinct contents and change count, the availability fraction, the pool count per tick, `top_margin`, and the
clone cost.

### 2.3 The statistic and the verdict rule (written first)

The plan's seed-paired statistic from `substrate/paired_contrast`: over the 8 seeds a one-sided paired t against
`T_BAR` = 1.895 with the exact sign-flip FPR beside it; a zero-variance sample is MET iff its common value is > 0.
No majority-of-seeds rule. Conjuncts, each MET or NOT MET and printed:

(p)  DETERMINISM: two recordings of seed 0 at 100 ticks are byte-identical on every recorded field, else REFUSED.
(r)  READABLE: on every seed the SAME clone's report equals the real report on EVERY awake tick (content,
     availability and strength exactly), the named lesion is applicable (the reported content has a pool) on at
     least half of the awake ticks and the scrambled lesion has a candidate pool on at least half; else REFUSED
     (the clone is not the workspace, or the apparatus cannot read this composition -- named per seed).
(g0) THE REPORT VARIES (the positive control that is not the method): on every seed at least 2 distinct reported
     contents and at least 1 tick-to-tick content change over the recording; else REFUSED ("a constant report
     cannot be read for its generation order", never a STOP).
(c1) NAMED > SCRAMBLED AT THE LESION (the licensed sentence): t over seeds of (f_named(1.0) - f_scr(1.0)) > 1.895.
(c2) THE SCRAMBLED INTERMEDIATE LEAVES THE REPORT UNCHANGED (the dissociation input): on every seed the count of
     applicable scrambled ticks at dose 1.0 whose reported CONTENT changed is 0 -- a count, exact; a non-zero count
     is a FINDING about the competition (a lower-slot lesion re-ordering the top), named tick by tick, never a bar
     to raise.
VERDICT: PASS iff (c1) and (c2) with (p), (r), (g0) holding; NOT MET iff (c1) or (c2) fails; REFUSED iff (p), (r)
or (g0) fail. REPORTED, never barred: the dose ladder f_named(d) and f_scr(d) for the four doses, the strength-
change fractions, the flip dose per seed and the mean `top_margin` beside it (does the live charge margin predict the
flip dose?), the report's distinct contents / change count / availability, the clone cost, the Monte-Carlo effect of
the scrambled pool's random choice (12 re-draws of the scrambled arm on the dry-run recording's saved clones).

### 2.4 The reds (RUN RED before the mechanism; in-command with their controls thereafter; each reddened by a mutated scorer)

R1  THE CLONE IS THE WORKSPACE AND NOTHING ELSE: a constructed `GlobalWorkspace(seed=0)` with one subscriber
    callback counting deliveries, six pools posted over 20 ticks: the clone re-stepped equals the original's
    `snapshot` on every tick, the original's `_g` is unchanged by stepping the clone, and the subscriber counts 0
    deliveries from clones. Mutation `clone_shares_state` (`_g` not deep-copied): the original moves: red.
R2  THE DISSOCIATION ON A CONSTRUCTED WORKSPACE: six pools, distinct saliences: lesioning the reported pool at dose
    1.0 changes the report on every tick; lesioning a non-reported pool changes it on none; the instrument reads
    (c1) MET and (c2) MET on eight such seeds. Mutation `blind_scorer` (named and scrambled swapped): red.
R3  A CANNED REPORTER READS NOT MET: the same workspace read through a `snapshot` stand-in that returns one fixed
    record regardless of the workspace -> f_named = 0 on every seed -> (c1) NOT MET. Mutation
    `scorer_reads_workspace` (the scorer compares `ws.current` instead of the report record) reads MET on the
    canned reporter: red -- the probe must read the REPORT, not the state.
R4  A HALLUCINATING REPORTER READS NOT MET: a `snapshot` stand-in whose content is drawn from its own RNG on each
    call -> f_scr(1.0) > 0 -> (c2) NOT MET. Mutation `c2_always_met`: red.
R5  THE READABILITY REFUSAL: a clone that shares `_rng` with the original (constructed) re-steps to a different
    report on some tick -> (r) REFUSED. Mutation `r_ignores_mismatch`: red.
R6  THE DOSE LADDER: constructed per-dose fractions (0.0, 0.1, 0.6, 1.0) read a flip dose of 0.75; all-zero reads
    "none". Mutation `ladder_first_dose` (returns the first dose tested): red.
Each red runs on its constructed input and FAILS its assertion at commit 3 of 2.9; at commit 4 each is green and its
mutation is applied IN THE TEST and in `t081_reds()` in the module's self-check line (six booleans asserted).

### 2.5 The controls that are not the method

The SAME clone (the same state re-stepped: the readability twin, per tick); the SCRAMBLED lesion (the same operation
on a pool the report does not name -- the C5 row's own dissociation input); the constructed canned and
hallucinating reporters (R3, R4); the dose ladder (reported); the determinism twin (p). The EpisodeDriver is NOT a
control here and is not run (2.0).

### 2.6 Disposition (fixed in advance; the honesty lens adjudicates any reading of these words)

- PASS: on this composite the report channel's record is a function of the workspace's competition state that
  changes when the reported content's pool is lesioned and not when an unreported pool is, at the bar over 8
  seeds. CHARGED: `conclusion_protocol.CONCLUSION["report_order"]` gets `reading` = the mean over seeds of
  (f_named(1.0) - f_scr(1.0)) (a fraction difference in [0, 1], the licensed sentence's "changes the report") and
  `surface_symbols` = ("report_order_margin",); `Brain.state()` publishes `report_order_margin` = the live
  `top_margin` of the arm's bus this tick (the charge margin between the top slot and the runner-up, the quantity
  the campaign's flip dose is read against) and the cockpit prints it as a consciousness row WITH the licensed and
  refused sentences in the same view and a PROVENANCE line naming the pin, the lesion apparatus and f_named /
  f_scr. tasks.md T081 -> [X] PASS.
- NOT MET: the report does not follow the named lesion above the scrambled one at the bar, or an unreported
  lesion moved it. NO reading, NO symbol, nothing printed; the instrument stays (the reds fired); tasks.md T081 ->
  [X] NOT MET with the symbol named and printed nowhere (T077's shape).
- REFUSED: the fault named ((p), (r) or (g0)); the row stays [ ]; no surface changes; the retirement below is HELD.
- THE RETIREMENT (P2, T034's clause) lands in the surfaces commit on PASS and on NOT MET alike -- the condition is
  that both replacement halves EXIST as instruments (the channel since T034; this probe once its reds have fired),
  not that the row passes: `reasoning/consciousness_probe.py` is DELETED after its one unique capability is
  PORTED -- `integration_measure.main()` gains the 20-seed PCI sweep with the probe's two-sided assert (the marker
  reaches both verdicts across seeds) in the same commit; `spike_chat_lake._consciousness()` answers "are you
  conscious" from the report channel's live record (content / availability / strength), the conclusion protocol's
  rows that carry a reading (their licensed AND refused sentences; none today), this row's pinned verdict as
  PROVENANCE, and the contamination clause unchanged -- `measured_about_me` = a brain is attached and its report
  was read; its self-check re-pointed (no brain -> "no report to read", two brains with different broadcasts ->
  different replies); `tests/test_red_reachability.py`'s probe test re-pointed to `report_order_probe.` criteria
  that read the system; the comment-only sites and the docs of fact 5 annotated "RETIRED at T081 (<date>)", never
  rewritten; the episode_driver exemption row's dated CURRENT READING (2.0).
In every branch `tests/test_ratified_row_gates.py`'s arming fixture moves from `report_order` to
`social_cognition`. The noun in every branch is "the report channel's record on a tick" -- the barred X of 2.2.

### 2.7 What this row does NOT claim

No first-person account, no experience reported (the refused sentence); nothing about what the reported content
MEANS (a digit token or a coalition id is a content, not a thought); the lesion is applied to a COPY -- the live
brain's workspace is never perturbed and nothing here is a lesion study of the organism; "generated from the
internal state" means exactly that the record is reproduced bit-for-bit from the competition state and moves with
the named pool, on this composite, under this stream; nothing about `Brain._reentrant` (the second workspace, the
unresolved P2 debt of fact 1); nothing about the EpisodeDriver's trials (2.0); no aggregate consciousness score;
nothing about the chat reply's contamination clause, which stays.

### 2.8 Widths, cost, host, declared constants and their sensitivity

Seeds 0-7; per seed 600 ticks with ten workspace steps per awake tick (one real, nine on clones) at ~65 ms/tick
(fact 4: 59.3 ms with five; RE-MEASURED in 3.0) -> ~40 s, 8 seeds ~6 min sequential in one process, plus seed 0's
100-tick twin. Refusal: above 1 h the campaign is re-sized and this section amended. DECLARED CONSTANTS and their
consequence, COMPUTED in 3.0 (the dry run): `T081_DOSES` = (0.25, 0.5, 0.75, 1.0) (dose 1.0 bars; the ladder is
reported); the scrambled pool's RNG (7000 + seed; 12 re-draws on the dry run); N = 600; the offset 150 x seed on
both pointers; the applicability floor one half. Host: Python 3.14.6 free-threaded (the workspace's own per-call
threads); no torch or GeNN is the instrument here.

### 2.9 Commit order (one change each; the stubs before the reds so the reds RUN red; surfaces before the record)

1. This file + PLAN-FINAL Part F row C-R10 (T081) -> 2. the skeleton: `cortex/report_order_probe.py` with
`t081_clone`, `t081_lesion`, `t081_record_brain`, `t081_read`, `t081_flip_dose`, `t081_contrasts`, `t081_reds`
returning null readings, the constants of 2.8, `main()` printing the six booleans; the registrations
(run_selfchecks CORE, `__init__` REGIONS as its siblings, the `dormancy_gate.EXEMPT` row of 2.1; the census gates
green) -> 3. `tests/test_report_order_probe.py`, R1-R6 committed RED (each runs on its constructed input and fails;
output pasted) -> 4. the instruments green, each red reddened by its mutated scorer in-file and in `t081_reds()`;
census gates -> 5. the recorder + `--t081-campaign` + the determinism precondition + the pin writer
(`tools/t081_report_order.json`, once); the apparatus dry run on ONE seed at the campaign width: the dose ladder,
the 12 scrambled re-draws, the cost, the offset check REPORTED (its output named with its script in 3.0); NO barred
conjunct is read there -> 6. the campaign, the pin (once) -> 7. surfaces and annotations per 2.6 (tasks.md's dated
note; PLAN-FINAL C4 note; spec.md SC-009 note; the row-gate fixture to `social_cognition`; the RETIREMENT commit(s)
of 2.6 -- the port into `integration_measure` first, then the deletion and the re-pointed consumers, then the
annotations; on PASS the conclusion row, `Brain.state()` and the cockpit row) -> 8. section 3 the record -> panel
(3 lenses + refuters judging CLAIM and FIX) -> fold -> stacked with T080 / T082 for ONE battery (the Codex closing
report when the cap lifts, 2.0).

### 2.10 Amendment, dated 2026-09-16, BEFORE the campaign (no clock in this heading -- the commit's own stamp is the
### time, obs #2190): the build's deviations and findings disclosed; one ceiling raised as an admission; the dry run
### reproduced -- no gate's form, bar, disposition, arm or N changes

**The build (an Opus executor on `af1a5d8`, brief `scratchpad/t081/builder_brief.md`; 2.9 steps 2-5, one commit
each): `6ae9aab` skeleton + registrations; `ea36a86` R1-R6 committed RED (six AssertionErrors on constructed
inputs, none a collection error -- the executor implemented first, verified green, restored the skeleton, captured
the red and committed the tests; the test file is not edited at step 4); `b7c1a5c` green with the six mutations
reddening in-file and in `t081_reds()`; `d5e7e6e` recorder, `--t081-dry-run`, `--t081-campaign`, the determinism
precondition, the pin writer. The orchestrator re-ran the tests (6 passed), the self-check (6/6, rc 0) and the dry
run (below) before writing this.**

**(a) THE P6 CEILINGS, RAISED AS AN ADMISSION (`65fa7d3`).** 2.4 requires the reds to run on a constructed
`GlobalWorkspace(seed=0)` with pools posted into it and 2.9 step 4 requires them in-file, so the module holds a
private workspace instance and writes into it; `tools/p6_gate.py`'s `private_instance_modules` and
`unread_writer_modules` ratchets stood at 6 with zero slack and read 7 at commits 4 and 5 (the census script red on
`p6_gate --assert` alone; every other gate rc 0). The executor measured this BEFORE building through the gate's own
overlay (assigned form 7 RED; a `return GlobalWorkspace(...)` factory form 6 green) and REFUSED the factory form as
evasion through the walk's disclosed blind spot. The orchestrator's disposition: both ceilings 6 -> 7 with dated
admissions naming `cortex/report_order_probe.py` -- the T033 shape (`thalamus/ignition_instrument.py`'s constructed
subjects), an EXEMPT measurement instrument whose subject nobody else reads BY DESIGN. Both ceilings now sit on
the measurement (7 = 7): `--assert` green, `--self-check`'s one-mutant regression still crosses (4 of 4 controls
green). The next private instance or unread writer lands on 8 and reddens.

**(b) THE DOSE LADDER IS A STEP, NOT A LADDER (an apparatus finding against 2.3's REPORTED pairing).** On the dry
run (seed 0, 294 awake ticks) doses 0.25 / 0.5 / 0.75 move the reported content on 0 of 294 ticks and dose 1.0 on
294 of 294: scaling `_g` leaves the incumbent's ranking intact, and only the full lesion -- which also drops the
content's submissions from `_buffer` -- moves the report. The flip dose is therefore 1.0 BY CONSTRUCTION on every
seed, and 2.3's reported question "does the live charge margin predict the flip dose?" CANNOT be read: a constant
has no correlation with `top_margin` (0.0217 on seed 0). The ladder is still computed and printed per seed as
registered; the reading of (c1) at dose 1.0 is unaffected (2.3 fixes it at dose 1.0). What this says about the
mechanism is section 3's to state; what it says about the apparatus is that a graded lesion of `_g` is not a
graded lesion of the competition, and a re-registered ladder would scale the SUBMISSIONS' salience.

**(c) (g0) AND (c2) ARE NOT t-TESTS.** The brief paraphrased all three of (g0), (c1), (c2) through
`paired_contrast.contrast`; frozen 2.3 makes only (c1) a t; (g0) is a per-seed floor and (c2) an exact count. The
executor implemented 2.3, not the paraphrase (disclosed in-file and in `b7c1a5c`). 2.3's zero-variance rule
("MET iff the common value is > 0, recorded as degenerate") is the clause (c1) will read if every seed's
f_named(1.0) - f_scr(1.0) is exactly +1.0000 as on the dry run -- written here BEFORE the campaign.

**(d) A CONTROL THAT COULD NOT FAIL, found by running the mutation.** "The original's `_g` is unchanged after the
lesioned clone steps" read 20/20 intact WITH the `clone_shares_state` mutation installed, because `step()` rebinds
`_g`; R1's reachable red is `pre_intact` (the baseline the ten arms clone from). `g_intact` is kept and printed as
structurally satisfied -- reported, never counted.

**(e) APPLICABILITY READS THE SUBMISSIONS TOO.** 2.1's named intermediate includes "that content's submissions in the
tick"; a content first submitted this tick sits in `_buffer` and not yet in `_g`, so a `_g`-only test read it
inapplicable (a recording of fresh contents read applicable on 0 of 20 ticks and REFUSED at (r)). The test now
reads both; no effect on the seed-0 recording (294/294 either way).

**(f) R5's CONSTRUCTION IS NARROWER THAN 2.4's WORDING.** `_rng` is drawn from in exactly one place (`_jitter_for`,
first encounter of a pool key), so a clone sharing the original's RNG diverges only when a NEW key appears; R5's
subject posts fresh contents each tick with the top two saliences tied (disclosed in `t081_clone_shares_rng`).

**(g) R4 IS CAUGHT TWICE** -- the hallucinating reporter fails (c2) AND (r); the test asserts both.

**(h) THE TESTS IMPORT THE MODULE AS A MODULE** (`report_order_probe as rop`), because 2.9 step 2 fixes the stub
list and the constructed inputs are not in it -- a name-level import would have made every red a collection error
at step 3 (the test docstring and `ea36a86` say so).

**(i) THE DRY RUN, REPRODUCED.** The executor's run (`scratchpad/t081/dry_run_builder.json`, head `b7c1a5c`, 55.1
ms/tick) and the orchestrator's (`dry_run.json` + `dry_run_orch.txt`, head `d5e7e6e`, 08:12:12-08:13:25 local, 65.5
ms/tick under the daemon's load) agree on every non-timing field: offsets differ from the first frame (seed 0 vs 1);
the determinism twin identical on 49 awake rows of 100 ticks; 294 rows from 294 awake ticks (306 asleep); 20
distinct contents, 37 changes, availability 1.0000, mean pools 9.52, mean `top_margin` 0.0217; twin mismatches
0/294; named and scrambled applicable 294/294; NAMED content change 0 / 0 / 0 / 1.0000 by dose, SCRAMBLED 0 / 0 / 0
/ 0; strength change the same; flip dose 1.0; (c2) 0 scrambled ticks moved the report; the 12 scrambled re-draws
changed 0 of 294 on every draw (min = max = 0). Campaign estimate 4.4-5.2 min for 8 seeds. NO barred conjunct was
read (one seed carries no t); (g0) and (c2) are printed as seed 0's.

**(j) CLOCK.** The executor's report carries no wall-clock stamps; the commit stamps are the times (`git log
--format=%ci`). Nothing above 2.10 was edited.

## 3. Results (written 2026-09-16 after the campaign and the surfaces; nothing above this line was edited after 2.10)

### 3.0 The apparatus, measured before the campaign (2.9 step 5, the dry run; 2.10)

The executor's dry run on seed 0 at the campaign width (head `b7c1a5c`, `scratchpad/t081/dry_run_builder.json`, 55.1
ms/tick) and the orchestrator's re-run (head `d5e7e6e`, 08:12:12-08:13:25 local, `dry_run.json` + `dry_run_orch.txt`,
65.5 ms/tick under the daemon's load) agree on every non-timing field (2.10 (i)): 294 rows from 294 awake ticks, 20
distinct contents, 37 changes, availability 1.0000, mean pools 9.52, mean `top_margin` 0.0217, mismatches 0/294, both
lesions applicable 294/294, the ladder 0 / 0 / 0 / 1.0000 named and 0 / 0 / 0 / 0 scrambled, flip dose 1.0, 12
scrambled re-draws changing 0 of 294 on every draw. The P6 ceilings were measured BEFORE the build through the gate's
own overlay and raised as an admission (`65fa7d3`, 2.10 (a)). The section-1 probes (`scratchpad/t081/probe_report.py`,
`probe_copy.py`) were the design's and were NOT re-run at record time -- the panel's honesty lens is asked to. CLOCK:
every time in this section is a commit stamp (`git log --format=%ci`) or a printed clock (`date` in the launcher and
the waiter); no stamp was typed ahead (obs #2190).

### 3.1 The build, red-first, and the commits

Pre-registration `f64b41b` (this file 1-2.9 + C-R10). An Opus executor built 2.9 steps 2-5 (brief
`scratchpad/t081/builder_brief.md`): skeleton `6ae9aab`; R1-R6 committed RED `ea36a86` (six AssertionErrors on
constructed inputs, none a collection error); green with the six mutations reddening in the test and in
`t081_reds()` `b7c1a5c`; recorder, dry run, campaign flag and pin writer `d5e7e6e`. The census script read every gate
rc 0 after commit 2 and one red -- `p6_gate --assert` -- after commits 4 and 5, disposed by the orchestrator as the
admission `65fa7d3` (2.10 (a)); 2.10 `f5cf8b8` (dated BEFORE the campaign, 08:16:31). The orchestrator re-ran the
tests (6 passed), the module self-check (6/6 rc 0) and the dry run before the campaign.

### 3.2 The campaign (`tools/t081_report_order.json`)

Launched DETACHED on `f5cf8b8` at 08:16:35 local (`scratchpad/t081/launch_campaign.ps1`; log `campaign.log`, empty
`campaign.err`); the pin written in 350 s (the log's own line; the pin's `elapsed_s` 349.8; the waiter read it at
08:22:47, polling every 30 s); committed ONCE as `c1502b5` (08:23:05). Per seed 292-299 awake ticks of 600 with ten
workspace steps per awake tick (the real step + nine clones), 57.4-75.8 ms per brain-tick, 35-45 s per seed. THE PIN
CARRIES PER-SEED SUMMARIES AND NO PER-TICK ROWS (unlike T080's): every barred number re-derives from the per-seed
fields, and the campaign is deterministic -- `--t081-dry-run` reproduces seed 0's row field by field (3.0) -- but a
tick-level re-read of the other seven seeds needs a re-run of the recorder, not the pin (opening 5). The
orchestrator re-derived every barred number from the pin with no brain (`scratchpad/t081/verify_t081_pin.py`,
`verify_out.txt`): (g0), (c2) and the flip dose per seed, (c1) through `paired_contrast.contrast` (diffs, t,
degenerate, met identical to the pin), the 2.6 reading, and the verdict through the module's own `t081_verdict`
(PASS); seed 0 equals the dry run on n, contents, changes, f_named, f_scr, scr_changed_1, flip_dose and mismatch,
margin 0.021722 both.

### 3.3 The readings, conjunct by conjunct (every number from the pin; the bar 1.895 over 8 seeds; FPR = the sign-flip rate)

(p)  DETERMINISM: two recordings of seed 0 at 100 ticks identical on every field (49 awake rows each, first_diff
     None): MET.
(r)  READABLE: the clone's report equals the real report on EVERY awake tick of every seed (twin mismatches 0 x 8);
     the named lesion applicable on 292-299 of 292-299 awake ticks and the scrambled on the same (both at 1.0 against
     the 0.5 floor); `pre_intact` = n on every seed: MET.
(g0) THE REPORT VARIES: 17-24 distinct contents (floor 2) and 26-73 tick-to-tick changes (floor 1) on every seed: MET.
(c1) NAMED > SCRAMBLED AT THE LESION: f_named(1.0) - f_scr(1.0) = +1.0000 on every seed (294/294 ... 299/299 named,
     0 scrambled); the sample has ZERO VARIANCE, so the t is 0.0000 and the FPR 0.0 and NEITHER DECIDES ANYTHING --
     2.3's zero-variance clause decides: "MET iff its common value is > 0" -> **MET (degenerate)** (the quotation
     corrected at the panel fold to the frozen wording, 3.5 item 15).
(c2) THE SCRAMBLED INTERMEDIATE LEAVES THE REPORT UNCHANGED: 0 applicable scrambled ticks with a changed content on
     every seed (0 x 8 of 292-299): MET.
VERDICT: **PASS**.

pin head `f5cf8b8`; seeds [0, 1, 2, 3, 4, 5, 6, 7]; elapsed 349.8 s; verdict **PASS**; refusals none
(p) determinism: identical=True on 49 awake rows of 100 ticks (awake [49, 49], first_diff None)

| seed | awake / rows | offset | contents | changes | availability | mean pools | mean top_margin | twin mismatch | applicable named / scr | pre_intact | g_intact | flip dose | (c2) scr changed at 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 294 / 294 | 0 | 20 | 37 | 1.0000 | 9.52 | 0.0217 | 0 | 294 / 294 | 294 | 294 | 1.0 | 0 |
| 1 | 299 / 299 | 150 | 20 | 55 | 1.0000 | 10.43 | 0.0334 | 0 | 299 / 299 | 299 | 299 | 1.0 | 0 |
| 2 | 299 / 299 | 44 | 21 | 73 | 1.0000 | 11.10 | 0.0563 | 0 | 299 / 299 | 299 | 299 | 1.0 | 0 |
| 3 | 298 / 298 | 194 | 24 | 36 | 1.0000 | 11.05 | 0.0255 | 0 | 298 / 298 | 298 | 298 | 1.0 | 0 |
| 4 | 298 / 298 | 88 | 22 | 65 | 1.0000 | 11.77 | 0.0234 | 0 | 298 / 298 | 298 | 298 | 1.0 | 0 |
| 5 | 292 / 292 | 238 | 17 | 26 | 1.0000 | 11.06 | 0.0134 | 0 | 292 / 292 | 292 | 292 | 1.0 | 0 |
| 6 | 293 / 293 | 132 | 24 | 47 | 1.0000 | 11.31 | 0.0152 | 0 | 293 / 293 | 293 | 293 | 1.0 | 0 |
| 7 | 296 / 296 | 26 | 21 | 39 | 1.0000 | 10.34 | 0.0202 | 0 | 296 / 296 | 296 | 296 | 1.0 | 0 |

| seed | NAMED content change 0.25 / 0.5 / 0.75 / 1.0 | SCRAMBLED content change | NAMED strength change | SCRAMBLED strength change |
|---|---|---|---|---|
| 0 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| 1 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| 2 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| 3 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| 4 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| 5 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| 6 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| 7 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |

- **(g0) the report varies**: {'failures': [], 'met': True, 'n_seeds': 8}
- **(c1) (c1) the named lesion's report-change fraction - the scrambled lesion's**: diffs [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; t +0.0000; FPR 0.0; degenerate True; **MET** (2.3's zero-variance clause: the common value +1.0 > 0)
- **(c2) the scrambled lesion leaves the report unchanged**: {'failures': [], 'changed': [0, 0, 0, 0, 0, 0, 0, 0], 'met': True, 'n_seeds': 8}
- flip dose per seed: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; mean top_margin per seed: [0.0217, 0.0334, 0.0563, 0.0255, 0.0234, 0.0134, 0.0152, 0.0202]
- timings per seed (s): [{'build_s': 0.848, 'record_s': 36.494, 'total_s': 37.342, 'ms_per_tick': 60.82333333333333}, {'build_s': 0.93, 'record_s': 34.456, 'total_s': 35.387, 'ms_per_tick': 57.42666666666667}, {'build_s': 1.05, 'record_s': 36.063, 'total_s': 37.113, 'ms_per_tick': 60.105}, {'build_s': 1.035, 'record_s': 42.617, 'total_s': 43.652, 'ms_per_tick': 71.02833333333334}, {'build_s': 1.075, 'record_s': 39.961, 'total_s': 41.037, 'ms_per_tick': 66.60166666666667}, {'build_s': 1.079, 'record_s': 45.464, 'total_s': 46.544, 'ms_per_tick': 75.77333333333333}, {'build_s': 0.958, 'record_s': 34.968, 'total_s': 35.926, 'ms_per_tick': 58.28}, {'build_s': 0.948, 'record_s': 37.261, 'total_s': 38.21, 'ms_per_tick': 62.10166666666667}]
- ms per brain-tick per seed: [60.8, 57.4, 60.1, 71.0, 66.6, 75.8, 58.3, 62.1]
- reds at campaign time: {'R1': True, 'R2': True, 'R3': True, 'R4': True, 'R5': True, 'R6': True}
- constants: {'n_ticks': 600, 'doses': [0.25, 0.5, 0.75, 1.0], 'lesion_dose': 1.0, 'stream_offset': 150, 'stream_positions': 256, 'seeds': [0, 1, 2, 3, 4, 5, 6, 7], 'applicable_floor': 0.5, 'min_contents': 2, 'min_changes': 1, 'flip_floor': 0.5, 'scr_redraws': 12, 'scr_rng_base': 7000, 'scr_redraw_base': 900000, 'determinism_ticks': 100, 'offset_labels': 50, 'bar': 1.895, 'clone_fields': ['_buffer', '_g', '_jit', '_last_focus', '_last_late', '_last_total', '_last_v', '_rng', '_settled', '_slot_buf', '_slots']}
- 2.6 reading: mean over seeds of (f_named(1.0) - f_scr(1.0)) = 1.0

REPORTED: the dose ladder is a STEP on every seed (0 / 0 / 0 / 1.0 named, 0 / 0 / 0 / 0 scrambled, content and
strength alike), the flip dose 1.0 x 8 (floor 0.5), so the pre-registered pairing "does the live charge margin
predict the flip dose?" cannot be read -- the margin varies 0.0134-0.0563 across seeds against a constant (2.10
(b); 3.5 item 1 gives the mechanism); the strength-change fraction equals the content-change fraction on every arm
(a changed winner is a changed hold); the 12 scrambled re-draws on the dry-run recording changed 0 of 294 on every
draw; availability 1.0000 on every seed (the bus was never dark on an awake tick); the clone cost is nine extra
steps per awake tick (57-76 ms per brain-tick against the dry run's 55-66).

### 3.4 The disposition applied (2.6, the PASS branch) and the retirement

On this composite the report channel's record is a function of the workspace's competition state that changes when
the reported content's pool is lesioned and not when an unreported pool is, at the bar over 8 seeds -- read under
the zero-variance clause. CHARGED (`f67328d`): `CONCLUSION["report_order"].reading` = **1.0** (the mean over seeds
of f_named(1.0) - f_scr(1.0), each +1.0), `surface_symbols` = ("report_order_margin",); `Brain.state()
["report_order_margin"]` = `_t081_top_margin` of the arm's bus this tick (`614a73e`; None with no workspace);
`brain_telemetry.PROVENANCE["report_order_margin"]` names the pin, the lesion apparatus, f_named / f_scr and the
flip-dose-by-construction finding with both C5 sentences; the cockpit row `report-order margin` is appended after
the cross-modal row with both sentences in its title and JS comment -- and, like T080's row (T080 record 3.5 item
11), it is CLIPPED out of the ACCESS plate's 110 px box as rendered, so 2.6's "in the same view" conjunct is NOT MET
AS RENDERED, disclosed here and on tasks.md / spec.md / PLAN-FINAL / WholeBrain (`be493fe`); the arming fixtures
move to `social_cognition`; `conclusion_protocol --assert` rc 0; tasks.md T081 -> [X] PASS. THE RETIREMENT (P2,
T034's clause; the condition -- both halves exist as instruments -- held before the campaign): the 20-seed PCI sweep
PORTED first into `cortex/integration_measure` with the probe's two-sided assert (`272c4e1`; the probe's own run
took 2.82 s and 2.86 s on a calm host (a first reading of 9.79 s was taken under host load); the sweep line now reads "20-seed PCI marker (ported from consciousness_probe, T081): PCI marker PRESENT on 10 of 20 seeds (50%); ABSENT on seeds [1, 2, 3, 4, 5, 7, 9, 14]... -- the marker sweep cost 1.2-1.6 s of this self-check (the executor's three runs and the orchestrator's; the detail line byte-identical to the incumbent's)"); `reasoning/consciousness_probe.py` DELETED
and its consumers re-pointed (`45a5fa4`): `spike_chat_lake._consciousness()` answers "are you conscious" from
the report channel's live record (content / availability / strength), the conclusion protocol's rows that carry a
reading (their licensed AND refused sentences: `temporal_binding` 0.2309 and now `report_order` 1.0), this row's
pinned verdict as PROVENANCE, and the contamination clause unchanged; `measured_about_me` = a brain is attached and
its report was read; `tests/test_red_reachability.py`'s probe test re-pointed to the `report_order_probe.` criteria
(`report_order_probe.t081_verdict`, reads_system True -- the one criterion the scanner discovers in the module; 18 criteria tree-wide once the two `consciousness_probe.` ones left with the file, 0 tautologies; the orchestrator re-ran `red_reachability.scan()` and read the same); the comment-only sites and the docs of fact 5 annotated RETIRED, never rewritten, and the
`episode_driver` exemption's dated current reading (2.0) landed (`f030b4b`). `tests/test_ratified_row_gates.py`'s
arming fixture moves to `social_cognition`.

### 3.5 Deviations from 2.1-2.10, each with its number (dated 2026-09-16)

1. THE DOSE LADDER IS A STEP, AND WHY (2.10 (b) reported it; the mechanism is measured here). `t081_lesion` scales
   the pre-step clone's `_g` by (1 - dose); `GlobalWorkspace.step()` then sets, for every key submitted THIS tick,
   `_g[key] = min(1, max(_g[key] * decay, submitted_salience))` -- so the scaled charge of a content that is
   re-submitted in the same tick is overwritten by its submission before the ranking reads it. Measured on the
   module's constructed subject (`scratchpad/t081/probe_runnerup.py`, seed 0, 20 ticks; decay 0.85): a 0.75 lesion
   takes the winner's `_g` 0.72 -> 0.18 on the clone and `step()` restores it to 0.90 (the submitted salience) on
   every tick; the 0.75 arm changes neither content nor strength on 20/20 ticks; only dose 1.0, which also drops
   the content's `Coalition` records from the buffer, removes it from the competition. On the live bus every
   percept is re-submitted every awake tick, so no sub-full dose can lesion the reported content at all. A
   re-registered ladder scales the SUBMITTED salience (the buffer records) and `_g` together (opening 2).
2. (c1) READS UNDER THE ZERO-VARIANCE CLAUSE, NOT THE t. Every seed's difference is exactly +1.0, so the paired t is
   0.0 and the sign-flip FPR 0.0; 2.3's clause ("MET iff its common value is > 0") was written before any number and
   is applied verbatim; the record says plainly that no t decided this conjunct.
3. THE PIN CARRIES NO PER-TICK ROWS (2.2 listed the recorded fields per tick; the pin writer summarised them per
   seed). Every barred number re-derives from the summaries and seed 0 reproduces the dry run; a tick-level re-read
   of seeds 1-7 needs the recorder (opening 5). The pin is not rewritten.
4. The P6 ceilings raised 6 -> 7 as an ADMISSION (2.10 (a), `65fa7d3`), the T033 shape; both ceilings sit on the
   measurement and the self-check's regression still crosses.
5. The campaign's cost 57-76 ms per brain-tick against 2.8's ~65 (the five-clone probe) and the dry run's 55-66;
   350 s for 8 seeds against the dry run's 4.4-5.2 min estimate.
6. The section-1 probes were not re-run at record time (3.0).
7. The executor's step-7 deviations, verbatim: (a) THE MARGIN IS NOT BOUNDED IN [0, 1] -- the brief's test range was wrong and the executor measured instead of clamping: `_slot()` returns the ADMISSION rank (P6's `-_last_v` repair), not the charge order, so the top slot is not always the top charge and (g_top - g_second) / g_top goes NEGATIVE; on a scratch brain (seed 0, 120 ticks; `scratchpad/t081/margin_range.py`, re-run by the orchestrator with the same numbers) the slot order is not charge-descending on 32 of 100 readings, the margin is negative on 24 of 100, range [-0.0664, 0.2083], mean 0.0541; the pin's per-seed MEANS (0.0134-0.0563) hid every per-tick negative. Nothing is clamped (the printed number is the scored number); the test asserts margin <= 1.0; the PROVENANCE, the cockpit comment, brain.py, tasks.md and the three dated notes say so. 2.1's `top_margin` was defined over the SLOTS' charges and is read as defined; a re-registered margin over the CHARGE order is opening 4. (b) The arming fixture keeps the stand-in id "T034" (T082 is [ ], so `_task_done("T082")` is False and pointing the arm at it would fire nothing); the ROW moves to `social_cognition`, as `49288cd` did; the tests pin `_task_done("T081") is True` and `_task_done("T082") is False`. (c) TWO `dormancy_gate.EXEMPT` rows RETIRED, each a registration red this work caused and each fixed in the registration: `conclusion_protocol` (its own row pre-registered this: "RETIRES when C4's instruments import it to gate their own reporting" -- `spike_chat_lake` now imports it) and `report_order_probe` (reachable once `Brain.state()` imports `_t081_top_margin`); the reasons kept verbatim in the replacing comments. The P6 admission's comment (`65fa7d3`) still calls the module "dormancy_gate EXEMPT as such" -- stale by one commit; the admission's ground (a measurement instrument's throwaway subject) is unchanged; corrected at the fold if the panel asks. (d) The cockpit comment was reordered after `conclusion_protocol --assert` read surface debt 2 against the baseline 1: `surface_findings` reads a 7-line window and the first draft put the licensed sentence 5 lines above the `+row(...)` line; the two sentences now sit immediately above the row; the layout is untouched. (e) The port's test went into the existing `tests/test_integration_measure_lz.py` (a 3-seed determinism subset so the suite pays for one full sweep). (f) `docs/SYNTHESIS.md` does not exist; fact 5's "SYNTHESIS", "research", "tasks" and "PLAN-FINAL" resolved to the specs/022 files; `specs/BACKLOG_RANKED.md:77` and `specs/019-consolidated-open-work/tasks.md:1059` mention the probe and were left alone; specs/010's records untouched. (g) The graphify duplicate-capability guard fired twice (`pci_marker_sweep`; the `_StubBrain` docstring) and was answered with `# distinct-from:` markers after every listed site was read -- the one true second mechanism for the stub-brain job, `consciousness_probe._MinimalBrain`, is deleted in that same commit. (h) Fact 5's `brain.py:3399` had rotted to `:3423`; annotated at the current line. (i) `red_reachability.scan()` discovers ONE criterion in the probe (`t081_verdict`); the six reds are tests, not scanner-visible criteria -- the same shape as the sibling instruments.
8. The record's tables are generated from the pin (`scratchpad/t081/record_gen.py` -> `record_tables.md`), never
   typed.

Items 9-21, dated 2026-09-16, the panel fold (3 lenses + refuters over CLAIM and FIX; 28 filed, 18 stood: 4 MAJOR / 14
MINOR, no BLOCKER; every filed fix was wrong and the refuters' minimal fixes are what landed; every number below was
re-run by the orchestrator before it was written -- the refuters' own scripts under `scratchpad/t081-panel/`):

9. [1] MAJOR, FIXED (`4f5f3d2`): `PROVENANCE["report_order_margin"]` said the sub-full doses "scale _g
   without disturbing the incumbent's ranking" -- false as a mechanism: `_g` IS the competition's drive and the
   slot's charge. The true cause is 3.5 item 1's (the same-tick re-submission resets the charge); the clause now
   says so. The panel's refuter measured the other half on a 250-tick seed-0 recording: an unrefreshed pool existed
   on 121/122 awake ticks, post-step `_g` differed from the twin on 121/121, the admitted SET on 22/121, the HEAD
   on 0/121 -- so scaling `_g` DOES disturb the ranking when the target is not re-submitted; the reported content is
   re-submitted every tick (the erasure condition g x decay <= salience held on 107/107 named pairs). 2.10 (b),
   frozen, already deferred the mechanism to section 3.
10. [2] MAJOR, DISCLOSED: section 1 fact 4's "7 distinct contents" is WRONG -- the design probe printed
    `Counter(...).most_common(8)`, a list capped at 8, never a count. Re-run at the fold (`probe_report_full.py`,
    the same 300 ticks, 146 awake): 18 distinct contents -- the digit 1 on 128 ticks, the digit 0 on 1, and 16
    DIFFERENT `('GLOBAL', <id>)` coalition tokens (fact 4's "a coalition token otherwise" was one token in kind,
    many in identity). No barred number depends on fact 4; 3.0 said the section-1 probes were not re-run and the
    panel's lens re-ran them (fact 3 exact; fact 4 exact on every other number).
11. [3] MAJOR, FIXED (`81e50f1`): `CONCLUSION["report_order"].reading = 1.0` was a hand-typed literal bound to
    the pin by nothing (the row gate checked only that a pin FILE exists). `t081_pin_reading()` now re-derives 2.6's
    reduction from the committed pin at the module's `__main__` (run_selfchecks CORE) and refuses on a mismatch or a
    missing pin, beside a mutated derivation (the sides swapped, -1.0) shown to refuse; a test pins derived ==
    recorded == 1.0 at the recorded precision. OPENING (12): the P2-clean end state is ONE helper taking (row, pin,
    reducer) with T080's and T081's bodies retired into it -- larger than a fold clause.
12. [10] DISCLOSED + FIXED (`81e50f1`): the executor's dry run (`dry_run_builder.json`, cited in 2.10 (i) and 3.0
    as head `b7c1a5c`) is stamped with a head whose tree could not have run it -- at `b7c1a5c` the recorder is the
    2.9-step-2 STUB returning zero rows and `--t081-dry-run` does not exist (re-read at the fold: `git show
    b7c1a5c:...` shows the stub body). `_t081_head()` records `git rev-parse HEAD` and nothing about dirtiness, so
    the artifact named the last COMMIT rather than the code that ran: the step-5 working tree later committed as
    `d5e7e6e` (2.9 step 5 places the dry run INSIDE the commit step that carries the recorder, so a pre-commit run
    is what was pre-registered; the head field could not express it). The orchestrator's run (`dry_run.json`, head
    `d5e7e6e`) is the re-derivable one; the two differ only in `head`, `elapsed_s` and the timings -- no number in
    3.0 or 3.3 moves; the campaign pin's stamp `f5cf8b8` is unaffected (the module is byte-identical from `d5e7e6e`
    through `c1502b5`). Future artifacts carry `dirty` (git status --porcelain) beside `head`.
13. [4] MAJOR, FIXED (`c4b2b08`): the chat reply printed a C5 row's reading in the first person ("On report order
    I read 1.0") -- a scratch-brain campaign number invariant across no brain / a workspace-less brain / a reporting
    brain, the shape P8's sentience clause names as the defect. The sentence now says "the conclusion protocol's row
    carries a reading of 1.0 -- a recorded campaign number, not a read of me taken now"; the one first-person read
    is the report-channel clause. 2.6 predicted "none today" for charged rows; two were charged at landing (3.6
    opening 8, amended below).
14. [5] the structural entailment of 3.6, corrected in 3.7 (the bus is MULTI-SLOT; the cause is the deletion, not the
    top slot; the limit is a reporter FAMILY).
15. [13] 3.3 (c1) put four words inside 2.3's quotation marks that are not 2.3's -- "recorded as degenerate" is
    `substrate/paired_contrast`'s docstring (landed `14807ef`, 2026-09-15). Frozen 2.3 at `f64b41b` line 134 reads
    "a zero-variance sample is MET iff its common value is > 0" and the word "degenerate" appears 0 times in the
    251-line pre-registration. 3.3 corrected in place (above); 2.10 (c) stands as dated with BOTH drifts named here
    (the added "(recorded as degenerate)"; "the common value" for "its common value"). (c1) re-derives MET from the
    frozen wording alone; no bar, conjunct, disposition or verdict moves.
16. [14] 3.0's CLOCK sentence over-generalised: the orchestrator's dry-run window "08:12:12-08:13:25" (2.10 (i), 3.0)
    is neither a commit stamp nor a clock printed on disk -- it was read from `date` calls in the orchestrator's
    shell (the session transcript), and the artifacts' mtimes bound it: `dry_run_builder.json` 08:12:12.475,
    `dry_run.json` / `dry_run_orch.txt` 08:13:24.349 (71.874 s, matching `elapsed_s` 71.8). Every other time in
    section 3 is as the sentence says. OPENING (13): the dry run and the campaign should write `started_at` /
    `finished_at` beside `elapsed_s`.
17. [7] + [12] 3.2's "35-45 s per seed" names no pin field and fits neither: `total_s` 35.387-46.544 s, `record_s`
    34.456-45.464 s (= `ms_per_tick` x 0.6; `build_s` 0.85-1.08 s the difference). WITHDRAWN as a range; read the
    per-seed cost as 34.4-45.5 s of recording, 35.3-46.6 s wall (bounds rounded outward). Not a barred number.
18. [8] 3.5 item 5's "350 s against the dry run's 4.4-5.2 min estimate" compared unlike quantities: the dry run's
    printed estimate is `record_s x 8 / 60`, RECORDING time only; the campaign's comparable number is sum(record_s)
    = 307.284 s = 5.121 min, INSIDE the band, at a mean 64.0 ms/tick inside the dry run's 55-66; the pin's
    `elapsed_s` 349.8 additionally carries 34.6 s of the two stream-label walks, the (p) twin and `t081_reds()`
    (sum(total_s) = 315.211 s = 5.254 min). Item 5's per-tick clause stands (seeds 3 and 5 at 71.0 and 75.8 ms/tick
    did exceed 2.8's ~65); 2.8's envelope ("8 seeds ~6 min") was met at 5.83 min. OPENING (14): the estimate's label
    at the dry run's print ("campaign estimate ... for 8 seeds" over recording time) is corrected by whichever row
    next re-registers this instrument, never under the closed pin.
19. [9] 2.1's field list does not say which side of the step each reported scalar is read on; measured, they are
    read on OPPOSITE sides: `n_pools` is `len(pre._g)` -- the PRE-step clone's carry-over pool count (the count
    surviving the previous tick, before this tick's submissions fold into `_g`) -- while `top_margin` is
    `_t081_top_margin(ws)` on the POST-step live workspace, the moment `r0` is read. On seed 0 (60 ticks, 31 awake,
    `sides_live.py` re-run at the fold) the two sides differ on 20 of 31 ticks for `n_pools` and 30 of 31 for
    `top_margin` (max |post - pre| 2 pools, 0.2083 margin). Neither field enters a bar, a conjunct or the
    disposition; the pin stands; a margin read on the lesion's own side belongs to opening (2)'s re-registration.
20. [11] FIXED (`c4b2b08`): 2.6's "the contamination clause unchanged" was honoured verbatim and the EMITTED
    clause is byte-identical; but the DOCSTRING's account named `probe_access`'s `_access_ema` reading -- deleted in
    `45a5fa4` -- as the current demonstration "asserted below" (already false before this arc: that neuter-A/B went
    with the F11 marker on 2026-08-05). Corrected to name the report channel (two brains differing only in what
    their workspace broadcasts); `_StubBrain`'s docstring says its `_access`, `_seed` and `_access_ema` are read by
    nothing in the module and kept as the real Brain's duck-type.
21. [15] + [17] + [18] PATH PROVENANCE: `scratchpad/t081/` in this record names the row's scratch, split across two
    directories -- the dry-run artifacts (`dry_run.json`, `dry_run_builder.json`, `dry_run_orch.txt`) are in
    `<repo>/scratchpad/t081/` (the recorder's repo-relative default under the launcher's working directory, T080's
    shape); every other cited artifact (the probes, `verify_t081_pin.py`, `verify_out.txt`, `record_gen.py`, the
    launcher, `campaign.log`) is in the session scratchpad; NO scratch artifact is tracked (`.gitignore` ignores the
    tree), so a reader working from git resolves this row through the pin and the commits. [17] the P6 admission's
    clause "(dormancy_gate EXEMPT as such)" went stale at `614a73e` (the module left EXEMPT when `Brain.state()`
    imported `_t081_top_margin`); corrected at `02e7f7c` -- one comment clause, the T033 parenthetical and both
    ceilings untouched, `--assert` rc 0 before and after. [18] the SC-009 T081 note adopts T080's clipped-row clause;
    its incidental phrase "the row's title string" names no mechanism the page has (cockpit.py has no `title=`
    attribute; the sentences reach the served page as the JS source comment above each `row()` call and as the
    PROVENANCE `.tag` span; the clipping is `#bottom>div{overflow:hidden}` against the grid's 110 px bottom row --
    vertical) -- a dated line on the SC-009 note says so; the T080 note stands verbatim.

### 3.6 What this row does NOT claim, and the openings it leaves

Nothing about a first-person account, an author, a self, or an experience (the refused sentence, printed beside
the symbol on every surface that prints it and in the chat reply). THE STRUCTURAL ENTAILMENT, stated so the panel
can weigh it: on this bus a FULL lesion of the winner -- its submissions dropped and its charge zeroed -- entails a
different report, and a full lesion of any non-winner leaves the winner in place (the runner-up and the third slot
lesioned at 1.0 change the report on 0 of 20 constructed ticks each, `probe_runnerup.py`; the scrambled pool on 0 of
2,369 live ticks and 0 of 12 re-draws). (c1) and (c2) therefore read what a winner-take-all bus whose report IS its
top slot entails once the report channel reads that bus; the row's discriminating power lives in R3 and R4 (a
canned reporter and a hallucinating reporter read NOT MET) and in (r) (a clone that is not the workspace refuses).
What PASS licenses is exactly the licensed sentence and no more: the report channel's record is generated from the
workspace's competition state -- it is not a canned string, a hallucination, or a read of something other than the
bus. It does not license that the report is ABOUT anything, that the bus's content is a thought, or that the flip
dose is a threshold of anything (it is 1.0 by construction). Openings: (1) the graded ladder is unmeasured (3.5
item 1); (2) a re-registered lesion scales the submitted salience and `_g` together, and pre-registers the pairing
against `top_margin` only after a one-seed probe shows the flip dose varies (obs #2196); (3) the cockpit row is
clipped as rendered (3.4) -- the ACCESS plate's layout is owed its own row, shared with T070 / T077 / T080; (4)
`Brain.state()["report_order_margin"]` is the charge margin between the top slot and the runner-up on the live bus,
a quantity this campaign never read against anything (the pairing was unreadable) -- it is published as 2.6
prescribed and its provenance says what it is; (5) the pin without rows (3.5 item 3); (6) THE RETIREMENT's
non-unique losses, for the panel's P2 lens: `consciousness_probe.py` also carried `probe_access` (a reading of
`_access_ema` that returned no verdict since F11), `_MinimalBrain`, and `deleted_access_marker_arithmetic` (a LIVE
re-derivation of the F11 deletion's arithmetic, coherent / scrambled salience against rheobase and `_JITTER`, with
its positive-control numbers) -- 2.6 named the PCI sweep as the one unique capability and the arithmetic's
re-derivation is now off the tree (its record stands in the deleted file's git history at `45a5fa4^` lines 40-118,
in the live code that states the deletion with its date and reason -- `reasoning/spike_chat_lake.py` near line
1955 and `cortex/brain.py` near line 3414 -- and in the fact that its five numbers re-derive from CORE
`cortex/access_consciousness`; `docs/POLICIES_EVIDENCE.md`'s 2026-08-04 F11 row records the BINDING / RE-ENTRY
deletion, not this one -- CORRECTED at the panel fold 2026-09-16, panel [16]); whether that is a P2 loss or a
retired marker's epitaph is the panel's call; (7) the EpisodeDriver
is not routed to this report (2.0, fact 6) and the exemption row now says so; (8) the chat reply prints the C5 rows
with a reading -- two today -- and grows with each charged row; its contamination clause is unchanged and still
covers the framing.

**3.6 amendments, dated 2026-09-16, after the panel (the frozen text above untouched).** Opening (8): the chat loop as
landed printed the rows' readings in the first person with no scope note (3.5 item 13; fixed). Openings added: (9) THE
COMPOSITION OF (c2)'s 2,369 (panel [6]): the scrambled draw is uniform over every content in `pre._g` union
`pre._buffer` other than the reported one -- NOT restricted to the competitors. Re-run at the fold on a 300-tick seed-0
scratch recording (146 awake; `probe_live.py`): the drawn content was an ADMITTED slot on 2 ticks and the runner-up on
2, an unadmitted pool on 144 (the panel's refuter, at the campaign's offset over 194 awake ticks, read 9 admitted / 6
runner-up / 185 unadmitted; seed 3 at the full width 19 of 298 admitted); the drawn pool fired in the settle on most
ticks but IGNITED on none. So the live support for "a full lesion of a non-winner leaves the winner" is the few-percent
admitted subset plus the constructed ticks; the rest are lesions of pools that never took a slot. REPORTED, never
barred: forcing the lesion to the runner-up read 0 of 58 and to every admitted non-reported slot 0 of 79 (the refuter:
0 of 95 and 0 of 130) -- the strong version reads 0 too. A scrambled draw restricted to the admitted set is a new arm
for a new pre-registration. (10) the executor's dry-run stamp (3.5 item 12). (11) `n_pools` and `top_margin` read on
opposite sides of the step (3.5 item 19). (12)-(14) as numbered in 3.5 items 11, 16, 18.

### 3.7 Amendments after the panel, dated 2026-09-16 (nothing above this line edited except the two marked in-place
### corrections in 3.3 (c1) and 3.6 opening (6); no bar, barred number, conjunct, disposition or pin changed)

THE STRUCTURAL ENTAILMENT OF 3.6, CORRECTED (panel [5]; every number re-run at the fold, `refute_t081_36.py`):
(i) TERM: the bus is MULTI-SLOT, not winner-take-all. `GlobalWorkspace.capacity` is DERIVED as floor(1 / (self_boost +
hysteresis)) = floor(1 / 0.25) = 4 at the defaults the constructed subject and the arm's bus (`autonomous_loop.py`'s
`GlobalWorkspace(seed=seed)`) both use, and moves with the defence; "winner-take-all" in this tree names the RETIRED
single-slot arbiter kept as the A/B control. 3.6's own neighbouring clause names the runner-up and the third slot, so the
sentence was internally inconsistent rather than misleading about capacity. (ii) CAUSE: (c1)'s f_named(1.0) = 1.0 is
forced by the dose-1.0 lesion DELETING the content -- `t081_lesion` drops its buffer records and zeroes its `_g` keys, and
`step()`'s refresh decays the zeros and its `v > 1e-6` filter removes the keys, after which `submitted` cannot re-create
them: on the constructed subject the lesioned content is absent from `_buffer` pre-step on 76/76 applicable ticks and
from `_g` and `_slot()` post-step on 76/76. It is therefore forced for ANY reporter whose reported content is a live pool
content -- top slot, whole admitted set, highest-charge and lowest-charge pool all read (c1) = +1.0000 with f_scr = 0 --
not because the report is the top slot. (iii) THE LIMIT, at its true strength: (c2) and (r) DO discriminate among
projections of the pool set -- a LAST-admitted-slot reporter reads (c2) NOT MET (57 of 152 applicable scrambled ticks
changed its report; 8 seeds x 20 ticks, constructed subject); a pool-COUNT reporter is REFUSED by (r) (0 of 160 ticks
applicable); the shipped top-slot reporter reads 152/160 applicable, f_named 1.0, (c2) 0, PASS. What the row cannot
tell apart is the family that reads identically -- top slot, whole admitted set, highest-charge, lowest-charge. That,
plus R3 / R4 / (r) as 3.6 states them, is the row's discriminating power; the licensed sentence is unchanged.

THE MARGIN, for the record: `_t081_top_margin` is (g_top - g_second) / g_top over the SLOTS' charges; because the slots
are the ADMISSION rank, the live value is negative on about a quarter of ticks (3.5 item 7 (a)); the panel's refuter read
the absolute g_top - g_second on its own seed-0 recording at mean 0.033 over [-0.053, 0.156] and the orchestrator's
re-run at mean 0.038 over [-0.053, 0.156] -- the same quantity, unclamped.

## Closing report (Codex, gpt-6-astra, 2026-09-24; owed since the 2026-09-17 weekly cap)

- (a) The reporter-family non-identifiability is the one section 3.7 recorded; the licensed conclusion stays narrow.
- (b) Section 2.6 put `report_order_margin` on the consciousness surface, but the experiment never validated that
  quantity: every flip dose was 1.0 by construction (section 3.6, opening 4). `Brain.state` publishing it is a
  consumer, not P3's charge, and P8 lets the brain report only what it measures. Filed as an opening: remove it from
  the surface or mark it unvalidated there.
- (c) Stale: section 3.6's clipped cockpit row is superseded by T090's CONSCIOUSNESS plate (`t090-conclusion-plate.md`
  section 3.4 records it visible at both tested sizes).
