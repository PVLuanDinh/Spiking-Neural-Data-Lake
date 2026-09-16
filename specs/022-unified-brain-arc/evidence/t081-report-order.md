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
