# T080 -- the temporal-binding instrument: features inside one window, bound into one reported object, against a membership-matched null at matched K

Row: tasks.md T080 [US4]; the C5 row `temporal_binding` (`substrate/conclusion_protocol.py:145-152`: licensed
"features inside one window are bound into one reported object more often than features across windows"; refused
"this licenses no claim that there is a felt present, a stream of experience, or a duration lived"; dissociation
input "membership-matched pairing at matched K (the plan's primary null) must not reproduce the binding");
FR-028's binding item ("membership-matched primary null"); register #6's binding line; PLAN-FINAL Part F row C-R9
(added with this file). The plan's words for the row: "membership-matched pairing at matched K as the PRIMARY
preregistered null (scrambled-phase secondary), derived margin; in `thalamus/binding_instrument.py`".

## 1. Territory, measured before the design (each fact with its source line, read at HEAD `f49472c`)

1. THE LIVE BINDER BINDS ONE WINDOW: THE TICK. `cortex/brain.py:1832-1853` (W6): on every awake tick, when both
   `_real_percept` (the vision latent, 64-d, `_perceive_real` :1628-1660, the N-MNIST pack `models/nmnist_pack.npz`
   :696-699, 256 frames (15,2,34,34), 22-36 per digit class) and `_audio_percept` (the audio latent, 128-d,
   `_perceive_audio` :1749-1778, the SHD pack `models/shd_pack.npz` :725-727, 256 frames (100 bins x 700 channels),
   labels 0-19 = English + German spoken digits, 6-19 per label, digit = label mod 10) are present, the brain calls
   `CrossModalBinder.bind(vision, audio)` and publishes `fused["handle"]` as `_xmodal_handle` -> `Brain.state()`
   `crossmodal_handle` (:4127; beside `crossmodal_superadditive` :3983 and `crossmodal_objects` :4128). A congruent
   sighting -- the vision classifier's digit equals the audio classifier's word mod 10 (:1840-1844) -- REGISTERS
   `obj<digit>` once (:1845-1848; K = the taught count, at most 10). The two streams are walked by
   `_real_ptr` / `_audio_ptr`, each advanced on every awake tick, modulo their packs.
2. THE BINDER IS PURE ONCE TAUGHT, AND ITS "BOUND" IS A PER-MODALITY RECOGNITION PAIR. `cortex/crossmodal_binder.py`:
   `register` :120-133 stores each object's sign-random-projection spike codes (`vsa_reasoning.D` = 512) of the
   registering latents; `_recognize` :143-151 snaps a modality code to the nearest registered prototype by spike
   coincidence and refuses below `recog_margin` = 0.15; `bind` :153-204 reports `handle` = the vision object iff
   both modalities are recognised AND name the SAME object (`congruent`), `superadditive` from a fresh
   `MultisensoryCell.enhancement` (:97-118 re-initialises its LIF per trial) and the claustrum assembly from a fresh
   `Claustrum` per call (:206-221). No state crosses calls except the taught set, so the same (vision, audio) pair
   re-bound offline gives the same report: measured 60/60 (fact 5). The module's own docstring: "this binds FIXED
   latent VECTORS ... an object must be REGISTERED before it can be recognised or bound".
3. THE OTHER TWO BINDING INCUMBENTS ARE NOT ON THE TICK. `thalamus/gwt_deep.py` (:1-50): the phase-multiplexed K-object
   binding demonstration with its spiking-vs-rate control, `dormancy_gate.EXEMPT["gwt_deep"]` = "permanent"
   (:172-179, "Nothing on the live tick binds K (color,shape) objects"); its `rate_decode` (:156-168) is a reader
   that KNOWS the K active colours and K active shapes and guesses the pairing -- the membership-matched form at
   matched K this row's null descends from -- and its fixed 0.15 margin at K=2 is "a coin flip" by its own `_reds`
   docstring (12/40 seeds). `cortex/theta_gamma_code.py`: a phase-ordered working-memory buffer, not a binder of
   two features, its own dormancy lane. Neither is this row's mechanism; neither is changed by it.
4. THE DAEMON'S STREAMS ARE INDEPENDENT, SO IN THE WILD THE BINDER'S REPORTS ARE CHANCE BINDINGS. Measured 2026-09-16
   04:12 local (`scratchpad/t080/probe_binder.py`, seed 0, 300 ticks, scratch home, `persist=False`, the packs as
   shipped): the brain builds in 2.0 s and ticks at 55.1 ms; 146 awake ticks, both modalities present on 146/146,
   both pointers advanced on 146/146; the classifiers agree (digit == word mod 10) on 12/146 = 0.082, and at lags
   1 / 2 / 5 / 10 / 30 on 0.076 / 0.090 / 0.092 / 0.081 / 0.095 -- chance at every lag, no within-window structure
   in the shipped walk; digit and word runs last 1.09 and 1.06 ticks. The binder nevertheless reported a handle on
   28/146 ticks (superadditive on the same 28), 8 objects taught by tick 300, and 16 of the 28 CONTRADICT the
   classifiers (ticks 5-7: digit 0 / word 7, digit 4 / word 15, digit 8 / word 13 -> `obj6`, the only object taught
   at that moment): a nearest-prototype recogniser with one prototype names it for every latent above the margin.
5. THE APPARATUS EXISTS AND THE ROW IS READABLE ON IT. Measured 2026-09-16 04:18 (`scratchpad/t080/probe_aligned.py`,
   seed 0): with the audio pack RE-ORDERED at the stimulus so that position i carries a word naming the digit of
   vision position i (per class, cycled; a pure stimulus manipulation on the scratch brain, `_audio_frames` /
   `_audio_labels` replaced, both pointers at 0), the brain teaches all K = 10 objects in 54 ticks (2.9 s; vision
   accuracy 1.00, audio accuracy 0.76). ALIGNED arm, 400 ticks (21.6 s): 197 awake ticks with both modalities,
   handle reported on 60 (0.305), 52 of them naming the perceived digit, the classifiers congruent on 139 (0.71 --
   the audio classifier's accuracy is the ceiling; the binder's yield on congruent ticks 60/139). NULL arm in vivo
   (the SAME re-ordered frames with their positions permuted -- membership matched, K = 10 already taught): 199
   ticks, 16 handles (0.080), classifiers congruent on 22. LAG arms (the aligned audio delayed 1 / 2 / 5 ticks, 200
   ticks each): 13 / 13 / 10 handles of 100 -- the null rate: a word one tick before its digit is never bound. The
   OFFLINE re-bind of the aligned arm's recorded latents through `bind()` reproduces 60/197 exactly; the offline
   membership-matched null (the aligned ticks' audio latents permuted across ticks, 20 draws) reads mean 20.15,
   q95 27, max 27 of 197 -- the in-vivo null's 16 lies inside it. `bind()` costs 4.8 ms per call offline, so a
   2,000-draw null must run on the per-modality recognitions (fact 2), not on `bind()`.
6. WHAT NO GATE SEES TODAY. `crossmodal_handle` and `crossmodal_superadditive` reach `Brain.state()` and the cockpit
   with no sentence beside them; `CONCLUSION["temporal_binding"]` has `reading` None and `surface_symbols` ();
   `tests/test_ratified_row_gates.py:158-162` arms the row-gate fixture on `temporal_binding` as "the next
   symbol-less row" (it moves to `report_order` in this row's surfaces commit, T079's shape). No test names
   `crossmodal_handle` (grep, 2026-09-16). `node graph-exists.js "temporal binding instrument ..."`: NO EXISTING
   IMPLEMENTATION (2026-09-16).

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex): NOT SENT -- the account is under a WEEKLY usage cap until 2026-09-20 00:30

Pre-registered without the read, as T079 was; the closing report is owed when the cap lifts (obs #2183's rule: a
refused read is disclosed, never faked).

### 2.1 The mechanism -- ONE instrument reading the live binder's report against its own re-paired features

`thalamus/binding_instrument.py` (the row's named file; `dormancy_gate.EXEMPT` row of `ignition_instrument`'s
shape: an instrument whose consumers are its self-check, its tests and the campaign -- wiring it to the tick would
make the observer part of the observed). Nothing in `cortex/crossmodal_binder.py` or `brain.py` changes.
`t080_aligned_order(vis_labels, aud_labels, rng)`: per vision position an audio frame index whose label mod 10 equals
the vision label, cycling within each class in a per-seed shuffled order (1.5). `t080_record_brain(seed, arms)`: a
scratch brain under `throwaway_home()` (`make_brain(seed, persist=False)`), the aligned audio pack installed at the
stimulus, BOTH pointers set to `T080_STREAM_OFFSET * seed` mod 256 (T078 opening 9, both streams), the warm-up ticked
until `len(_xmodal_known) == T080_K` (= 10) or `T080_WARMUP_MAX` (= 800) ticks -- the taught set is then FROZEN by the
brain's own one-shot rule, so K is matched across every arm by construction; then the arms of 2.2, each recording
per tick `{t, awake, digit, word, handle, superadditive, vis (64), aud (128), vptr, aptr}`, rows returned, the brain
dropped. `t080_recognitions(rows, binder)`: per scored tick `(obj_v, s_v)` and `(obj_a, s_a)` through the binder's
OWN `_project` and `_recognize`, and `bound = (obj_v is not None and obj_v == obj_a)` -- the decomposition of
`bind()`'s handle (1.2), checked against the recorded handle on every tick (2.3 (r)). `t080_null(recs, seed,
n_draws)`: THE MEMBERSHIP-MATCHED NULL AT MATCHED K -- the audio recognitions of the scored ticks permuted across the
ticks (the same multiset of audio features, the same taught K, the pairing across windows broken), `n_draws` =
2,000 permutations under `numpy.random.default_rng(7000 + seed)`, each draw's bound fraction; `q95` =
`_mathutils.q95` of the draws (THE DERIVED MARGIN), the mean, `q005` and `q995` beside it. `t080_window(lag_rates,
q95)`: the smallest lag whose bound fraction is at or below the margin (the binding window, in ticks; "none" if
every lag arm is above it). `t080_contrasts(per_seed)`: 2.3's conjuncts through `substrate/paired_contrast.contrast`.

### 2.2 The arms (one command, seeds 0-7, a scratch brain; the daemon's brain is never touched)

Per seed, after the warm-up, in an order that ALTERNATES by seed parity (even seeds ALIGNED first, odd seeds NULL
first -- the brain's other state drifts across arms; the binder does not): ALIGNED (N = `T080_N_TICKS` = 600 ticks,
the aligned pack in place); NULL IN VIVO (600 ticks, the aligned pack with its 256 positions permuted under
`random.Random(1000 + seed)` -- the same frames, K taught, the pairing broken at the stimulus); LAG 1, LAG 2, LAG 5
(`T080_LAG_TICKS` = 300 each, the aligned pack with `_audio_ptr` set `d` positions behind `_real_ptr`); SCRAMBLED
PHASE (300 ticks, the aligned pack with every frame's 100 time bins permuted by ONE fixed per-seed permutation --
the secondary null: does the audio feature survive the loss of its spike phase?). The barred X is the fraction of
scored awake ticks (both modalities present) on which the binder reports ONE object (`handle` not None); the
barred quantity is the binder's own report -- the disposition noun (2.6) and this X are the SAME object. Every
recording also carries the taught set, the ticks-to-K, the audio accuracy, the false-binding count (handle !=
`obj<digit>`), the classifier-congruent count, and the yield (bound / congruent).

### 2.3 The statistic and the verdict rule (written first)

The plan's seed-paired statistic from `substrate/paired_contrast`: over the 8 seeds a one-sided paired t against
`T_BAR` = 1.895 with the exact sign-flip FPR beside it; a zero-variance sample is MET iff its common value is > 0.
No majority-of-seeds rule. Conjuncts, each MET or NOT MET and printed:

(p)  DETERMINISM: two ALIGNED recordings of seed 0 at 100 ticks (after identical warm-ups) are byte-identical on
     every recorded field, else REFUSED.
(r)  READABLE: on every seed K = 10 is taught within `T080_WARMUP_MAX`; on every scored awake tick of every arm both
     latents are present; the decomposition's `bound` equals (`handle` is not None) on EVERY scored tick of EVERY arm
     (the two-instrument check of 1.2 / 1.5); else REFUSED (the apparatus's or the instrument's fault, named per seed
     and arm).
(g0) THE STREAM CARRIES WITHIN-WINDOW STRUCTURE (the positive control that is not the method): t over seeds of
     (congruent fraction ALIGNED - congruent fraction NULL) > 1.895, where "congruent" is the CLASSIFIERS' agreement
     (digit == word mod 10), not the binder's; NOT MET -> REFUSED ("the alignment did not reach the brain", never a
     STOP).
(b1) WITHIN > ACROSS AT THE DERIVED MARGIN (the licensed sentence): t over seeds of (R_within - Q95_null) > 1.895,
     R_within = the ALIGNED arm's bound fraction, Q95_null = the q95 of the offline membership-matched null's bound
     fraction on that seed's ALIGNED ticks.
(b2) THE IN-VIVO NULL AGREES WITH THE OFFLINE NULL (two instruments, obs #2106): on every seed the NULL IN VIVO arm's
     bound fraction lies within [q005, q995] of the offline null's draws; else REFUSED (the decomposition is not the
     report, or the stimulus permutation did not take -- named).
VERDICT: PASS iff (b1) with (p), (r), (g0), (b2) holding; NOT MET iff (b1) NOT MET; REFUSED iff (p), (r), (g0) or (b2)
fail. REPORTED, never barred: the binding window per seed from the lag arms (2.1 `t080_window`; on this stateless
binder the expected reading is 1 tick -- a lag arm ABOVE the margin is a finding, printed); the scrambled-phase
arm's bound fraction and audio accuracy beside ALIGNED's; the false-binding fraction per arm; the yield; the
Monte-Carlo sd of Q95_null over 12 independent 2,000-draw nulls on the dry-run recording (2.8); the null's mean and
its product-of-marginals expectation (sum over objects of p_v(o) p_a(o)).

### 2.4 The reds (RUN RED before the mechanism; in-command with their controls thereafter; each reddened by a mutated scorer)

R1  THE NULL IS MEMBERSHIP-MATCHED: on a constructed recognition table (200 ticks, 10 objects, 60 % congruent) every
    draw's multiset of audio objects equals the table's, and the draws' mean bound fraction is within 3 sd of the
    product-of-marginals expectation. Mutation `null_resamples` (audio objects drawn i.i.d. with replacement): the
    multiset differs on a draw: red.
R2  THE DECOMPOSITION IS THE BINDER: a `CrossModalBinder` taught 10 random latent pairs, 200 random (vision, audio)
    latent pairs (a third of them near a prototype): `bind()["handle"] is not None` equals the decomposition on all
    200. Mutation `ignore_margin` (recognition without `recog_margin`): differs on a sub-margin pair: red.
R3  THE WINDOW READER: constructed lag rates (above the margin at lags 1-3, at or below from lag 4) read a window of
    4; all-above reads "none". Mutation `window_first_lag` (returns the first lag tested): red.
R4  THE POSITIVE CONTROL REFUSES AN INDEPENDENT STREAM: constructed classifier tables with independent labels on all
    8 seeds read (g0) NOT MET -> REFUSED; aligned tables read MET. Mutation `g0_always_met`: red.
R5  THE DISSOCIATION ON A CONSTRUCTED REPORT: per-seed tables where the within fraction equals the null's q95 read
    (b1) NOT MET; tables where within exceeds it on every seed read MET. Mutation `blind_scorer` (within and null
    swapped): red.
R6  THE AGREEMENT BAND: a constructed in-vivo null fraction outside [q005, q995] reads (b2) REFUSED; inside reads
    holding. Mutation `band_unbounded` (the band is [0, 1]): red.
Each red runs on its constructed input and FAILS its assertion at commit 3 of 2.9; at commit 4 each is green and its
mutation is applied IN THE TEST and in `t080_reds()` in the module's self-check line (six booleans asserted).

### 2.5 The controls that are not the method

The offline membership-matched null (the same features, the same K, the pairing broken -- the C5 row's own
dissociation input); the NULL IN VIVO arm (the same permutation applied at the stimulus, read by the brain, the
agreement control (b2)); the CLASSIFIERS' congruence (g0), which is not the binder; the lag arms (the across-window
presentation in vivo, reported as the window); the scrambled-phase arm (secondary, reported); the determinism twin
(p). gwt_deep's synthetic K-object task is NOT a control here and is not run.

### 2.6 Disposition (fixed in advance; the honesty lens adjudicates any reading of these words)

- PASS: the live binder reports one object for features presented inside one tick window more often than a null
  that keeps every feature and the taught K and breaks only their pairing across windows, at the bar over 8 seeds.
  CHARGED: `conclusion_protocol.CONCLUSION["temporal_binding"]` gets `reading` = the mean over seeds of
  (R_within - mean_null) (a bound-fraction difference in [-1, 1], the licensed sentence's "more often") and
  `surface_symbols` = ("crossmodal_bound_rate",); `Brain.state()` publishes `crossmodal_bound_rate` = the fraction
  of the last 200 awake ticks on which `crossmodal_handle` was not None (a LIVE quantity -- on the daemon's
  independent streams it reads the null's rate, fact 4, and the row's PROVENANCE line says so) and the cockpit
  prints it as a consciousness row WITH the licensed and refused sentences in the same view and a PROVENANCE line
  naming the aligned apparatus, the pin and the null's q95. tasks.md T080 -> [X] PASS. Nothing is retired: the
  binder is W6's live mechanism, `gwt_deep`'s exemption stands on its own reason (a different task, 1.3).
- NOT MET: the binder does not report within-window features as one object above its membership-matched null at
  the bar. NO reading, NO symbol, nothing printed; the binder stays; the instrument stays (the reds fired); tasks.md
  T080 -> [X] NOT MET with the symbol named and printed nowhere (T077's shape, so the arming arm sees it).
- REFUSED: the fault named ((p), (r), (g0) or (b2)); the row stays [ ]; no surface changes.
In every branch `tests/test_ratified_row_gates.py`'s arming fixture moves from `temporal_binding` to `report_order`
(the next symbol-less row) in the surfaces commit. The noun in every branch is "the binder's report of one object
on a tick" -- the barred X of 2.2.

### 2.7 What this row does NOT claim

No felt present, no stream of experience, no duration lived (the refused sentence); nothing about a temporal
INTEGRATION window -- the window is the tick because the binder is stateless (fact 2), and the lag arms measure the
mechanism's shape, not a tolerance; nothing about the daemon's brain, whose streams are independent (fact 4) so its
reported bindings in the wild are chance congruences and its `crossmodal_superadditive` rides on them -- the row
reports this, the C5 refusals forbid any surface reading it as more; nothing about gwt_deep's K-object task or
theta-gamma phase codes; no aggregate consciousness score; nothing about the classifiers' accuracy beyond its
reported role as the yield's ceiling.

### 2.8 Widths, cost, host, declared constants and their sensitivity

Seeds 0-7; per seed ~54 warm-up ticks (fact 5; capped at 800) + 600 + 600 + 3 x 300 + 300 = ~2,450 brain-ticks at the
measured 55 ms (RE-MEASURED in 3.0 on the built apparatus) -> ~2.3 min, 8 seeds ~19 min sequential in one process
(throwaway_home is process-wide), plus seed 0's 100-tick twin; the offline null 2,000 draws x 600 ticks as a numpy
permutation of the audio-recognition column -- under a second per seed. Refusal: above 2 h the campaign is re-sized
and this section amended. DECLARED CONSTANTS and their consequence, COMPUTED in 3.0 (the dry run): `T080_K` = 10 (the
pack's classes; a seed that teaches fewer refuses at (r)); `recog_margin` 0.15 and `cue_strength` 0.055 (the
binder's own, read not chosen; the yield's sensitivity to the margin is REPORTED on the dry run at 0.10 / 0.15 /
0.20 offline); N = 600 / 300; lags (1, 2, 5); the offset 150 x seed on both pointers (the dry run checks the first 50
vision labels differ between seeds 0 and 1); n_draws 2,000 (the Monte-Carlo sd of Q95 over 12 independent nulls);
`T080_WARMUP_MAX` 800. Host: Python 3.14.6 free-threaded, numpy for the null (a permutation of one integer column;
torch is not the instrument here), the brain's own torch encoders on the RTX 5070 where present.

### 2.9 Commit order (one change each; the stubs before the reds so the reds RUN red; surfaces before the record)

1. This file + PLAN-FINAL Part F row C-R9 (T080) -> 2. the skeleton: `thalamus/binding_instrument.py` with
`t080_aligned_order`, `t080_record_brain`, `t080_recognitions`, `t080_null`, `t080_window`, `t080_contrasts`,
`t080_reds` returning null readings, the constants of 2.8, `main()` printing the six booleans; the registrations
(run_selfchecks CORE, `__init__` REGIONS as its siblings, the `dormancy_gate.EXEMPT` row of 2.1; the census gates
green) -> 3. `tests/test_binding_instrument.py`, R1-R6 committed RED (each runs on its constructed input and fails;
output pasted) -> 4. the instruments green, each red reddened by its mutated scorer in-file and in `t080_reds()`;
census gates -> 5. the recorder + `--t080-campaign` + the determinism precondition + the pin writer
(`tools/t080_temporal_binding.json`, once); the apparatus dry run on ONE seed at the campaign widths: the lag arms,
the scrambled arm, the margin sensitivity, the Monte-Carlo sd, the offset check and the cost REPORTED (its output
named with its script in 3.0); NO barred conjunct is read there (one seed cannot carry a paired t) -> 6. the
campaign, the pin (once) -> 7. surfaces and annotations per 2.6 (tasks.md's dated note; PLAN-FINAL C4 note; spec.md
SC-009 note; the row-gate fixture to `report_order`; on PASS the conclusion row, `Brain.state()` and the cockpit row;
on NOT MET / REFUSED nothing in `conclusion_protocol`, `brain_telemetry` or the cockpit changes) -> 8. section 3 the
record -> panel (3 lenses + refuters judging CLAIM and FIX) -> fold -> stacked with T081 / T082 for ONE battery (the
Codex closing report when the cap lifts, 2.0).

### 2.10 Amendment, dated 2026-09-16 05:20 local, BEFORE the campaign (the build's contradictions and deviations
### disclosed; one readability clause made non-vacuous; the pin gains a companion) -- no gate's form, bar, disposition, arm or N changes

The build (commits `4c5b8c9` skeleton, `4ef9ab3` R1-R6 RED, `8466cf2` green + mutations, `66822fa` recorder + dry run;
an Opus executor, brief `scratchpad/t080/builder_brief.md`) and the dry run on seed 0 (`--t080-dry-run`, 204 s, its
output `scratchpad/t080/dry_run.json`; RE-RUN by the orchestrator at 05:07-05:11 with every count identical --
bound 76 / 24 / 11 / 17 / 17 / 8 over ALIGNED / NULL / LAG 1 / LAG 2 / LAG 5 / SCRAMBLED, the offline null mean 0.0953
q95 0.1224 band [0.0544, 0.1395], window 1, decomposition 1188/1188 -- and only the wall clock moved, 69.8 vs 66.7
ms/tick) contradicted or sharpened the frozen text in these places:

(a) R2 AS WRITTEN WAS UNREACHABLE. 2.4 R2 asks `ignore_margin` to differ "on a sub-margin pair"; with 10 taught pairs
    and 200 probes drawn dense in [0, 1] no probe falls under the 0.15 margin (min coincidence 0.469 / 0.492; the
    sign-random projection of two all-positive vectors is dominated by their common mean). The red's constructed
    input is drawn SPARSE (`T080_R2_DENSITY` = 0.10: 41 / 58 sub-margin probes, 11 changed by the mutation) -- the
    red's input, not a bar; the density sweep is in commit `4ef9ab3`'s message.
(b) THE MARGIN SENSITIVITY OF 2.8 IS FLAT BECAUSE `recog_margin` REFUSES NOTHING IN VIVO: bound 76 at 0.10, 0.15 and
    0.20, 0 sub-margin ticks at any (centred coincidence on the ALIGNED arm: vision min 0.594 / median 0.742, audio
    min 0.504 / median 0.719). The handle is decided entirely by whether the two nearest-prototype winners agree (76
    of 294). This sharpens fact 4: in vivo EVERY latent is above the margin. Reported; no clause moves.
(c) (r)'s SECOND CLAUSE WAS VACUOUS ("on every scored awake tick both latents are present" -- a scored tick is defined
    as such). AMENDED to read "on every AWAKE tick of every arm both latents are present (`awake_missing` = 0), else
    REFUSED (the seed and arm named)". The dry run reads 0 missing on all six arms (294/294, 294/294, 146/146,
    152/152, 150/150, 152/152). A stricter readability clause, adopted before any barred number is read.
(d) NUMBERS THAT MOVED FROM SECTION 1 / 2.8: the warm-up to K = 10 reads 39 ticks (fact 5 measured 54) because the
    built `t080_aligned_order` visits the classes in SORTED order, not the probe's dict-insertion order -- the pairing
    depends on the pack's content, not on numpy's label order; the NULL arm's permutation is drawn from a FRESH
    `random.Random(1000 + seed)` (the probe continued the aligned order's generator); the cost is 66.7-69.8 ms/tick
    over 2,439 ticks (2.8 carried 55 from the unwrapped probe) -> ~22 min for 8 seeds, inside 2.8's 2 h re-size.
(e) THE DERIVED MARGIN'S MONTE-CARLO SD IS EXACTLY 0.00000: 12 independent 2,000-draw nulls all read q95 = 0.12245
    (= 36/294); the statistic is an integer count over 294 ticks, so its sampling noise sits below its own
    discretisation. 2.3 asked for the sd; at this width it is zero.
(f) CLOCK TIMES IN FACTS 4-5 WERE ESTIMATED, NOT READ (obs #2190): the probes were written 04:04:15 and 04:09:02 and
    ran immediately after, not at "04:12" and "04:18". The numbers stand; the stamps are corrected here.
(g) THE PIN GAINS A COMPANION. The builder pinned the decomposition's per-tick columns and omitted the raw latents
    (a 3.7 M-float artifact no conjunct reads -- disclosed in the pin's `rows_omitted`). Without them (r)'s
    "decomposition equals the recorded handle" cannot be re-derived from the artifacts after the fact, so the
    campaign now also writes `tools/t080_latents.npz` (float32, every scored tick of every arm and seed, in each
    arm's `recs` order) and pins its sha256, keys and shapes under `latents_npz`; written once beside the pin (T078's
    npz precedent). Landed as the commit after this amendment.
(h) DEVIATIONS ACCEPTED AS DISCLOSED (the builder's report, §6): the reds' constructed-input builders are real from
    the skeleton commit so step 3's failures are assertions, not TypeErrors; three mutation hooks in the signatures
    (`t080_null(resample=)`, `t080_recognitions(recognize=)`, `t080_contrasts(score=, band=, g0=, bar=)`, T079's
    precedent); `t080_verdict` spells 2.3's rule once; R3's lag tables are module constants; `agree` COUNTS
    DISAGREEMENTS (its docstring says so; (r) requires 0); per-arm classifier accuracy is a difference of the brain's
    cumulative counters (the published `_audio_acc` carries the warm-up); `T080_REDS_N_DRAWS` = 500 for the reds
    (2,000 stays the campaign's); a REFUSED campaign writes the pin and returns 2 (T079's shape) -- a refused pin
    and its companion are deleted before any re-run, and the re-run is disclosed.
(i) TWO DEFECTS IN THE BUILDER'S OWN APPARATUS, found by the dry run and fixed inside `66822fa`: the margin-sensitivity
    reader was handed the decomposition instead of the rows, read 0 scored / 0 bound at every margin, and its
    positive control PASSED on 0 == 0 (now `n > 0` is part of `control_ok`); the per-arm audio accuracy was
    unreadable off the cumulative counter (fixed by differencing) -- which is what exposed (j).
(j) THE SCRAMBLED-PHASE ARM TAKES AT THE CLASSIFIER: audio accuracy 0.7143 (ALIGNED) -> 0.3355 (SCRAMBLED) with
    vision unchanged (0.9286 -> 0.8947), bound 0.2585 -> 0.0526. 2.3 asked for this beside ALIGNED's and predicted
    nothing; reported. The false-binding fraction is non-trivial on every arm (ALIGNED 9/76, NULL 6/24, LAG 1 5/11,
    SCRAMBLED 4/8); reported, never barred.
Nothing else moves: (p), (g0), (b1), (b2), the arms, N, the doses of nothing (there are none here), the derived margin's
form, the disposition, the reds' clauses.
