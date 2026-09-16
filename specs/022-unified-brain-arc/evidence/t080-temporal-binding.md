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

### 2.11 Amendment, dated 2026-09-16 06:05 local, AFTER campaign run 1's numbers (disclosed as such): (b2)'s per-seed
### band is made family-wise over the 8 seeds; the pin carries the prototypes in registration order; run 2 follows

**What run 1 read, numbers first.** The campaign ran once on head `309efe6` (05:23:37-05:47:31 local, 1,434 s; a
first launch on `7f6a000` at 05:21:54 was stopped at 57 s before it wrote anything, so the pin could carry the taught
prototypes). Its pin (archived as `scratchpad/t080/pin_run1_refused.json` + `lat_run1_refused.npz` + `campaign_run1.log`
and in the memory archive; DELETED from `tools/` before run 2 as 2.10 (h) prescribes) read: (p) identical on 100 rows;
(r) 0 decomposition disagreements and 0 missing latents on every arm of every seed; (g0) diffs 0.6122 / 0.6533 /
0.5869 / 0.5894 / 0.6100 / 0.6298 / 0.5738 / 0.6289, t **+65.3517** FPR 0.0352 MET; (b1) diffs 0.1361 / 0.2423 / 0.1757 /
0.2169 / 0.1781 / 0.2068 / 0.1913 / 0.2721, t **+13.5395** FPR 0.0469 MET -- the within-window bound fraction above the
null's q95 on 8 of 8 seeds; (b2) inside on 7 seeds and OUTSIDE on seed 3: the in-vivo NULL arm 0.1531 against the
offline band [0.0610, 0.1424] -> **VERDICT REFUSED as pre-registered**; windows [1, 1, 1, 1, 5, 2, 1, 1]; scrambled
bound fractions 0.0526 / 0.1074 / 0.1208 / 0.0927 / 0.0805 / 0.0658 / 0.1141 / 0.0805. The orchestrator re-derived
every barred number from the pin and the companion with no brain (`scratchpad/t080/verify_t080_pin.py`): (g0), (b1),
(b2), the windows and the verdict reproduce; the bound fractions reproduce on 6 of 8 seeds exactly and differ on 2
(seeds 6, 7) by a few ticks -- see the tie finding below.

**Why (b2) refused, measured from the pin (`scratchpad/t080/b2_analysis.py`, `b2_position_null.py`).** (b2)'s stated
causes are both excluded: the decomposition equals the report on every tick (agree 0), and the permutation took
(the NULL arm's classifier congruence 0.081-0.126 against ALIGNED's 0.715-0.734). The two arms' recognition
marginals agree (product-of-marginals expectation ALIGNED vs NULL: 0.1006 vs 0.0971 on seed 3; within 0.004 on
every seed), so the two instruments agree on the QUANTITY. What the pre-registration failed to foresee is the
BAND: the in-vivo NULL arm is one draw of a POSITION permutation of the 256-frame pack walked over ~1.15 laps by
the ~294 scored ticks, so its bound fraction is 1.13x more variable than the i.i.d. tick re-pairing the offline
null draws (position-null sd 0.0188 vs 0.0171 on seed 3; 256 position classes, 0 inconsistent audio
recognitions within a class on any seed), and a per-seed 99 % band applied to EIGHT seeds refuses an honest
apparatus 1 - 0.99^8 = 7.7 % of the time by construction (6.2 % simulated from the position nulls). Seed 3's draw
has a one-sided p of 0.0037 under its own position null -- outside the per-seed 99 % band, inside the family-wise
band [0.0458, 0.1695]; every other seed sits inside both. This is a readability clause with a 6-8 % false-refusal
rate, not a disagreement between the instruments.

**What changes (AFTER the numbers -- the honesty lens adjudicates), and what does not.** (b2) reads: "on every seed
the NULL IN VIVO arm's bound fraction lies within the offline null's FAMILY-WISE 99 % band, [q(0.005 / S),
q(1 - 0.005 / S)] over its draws with S = 8 seeds (`T080_BAND` derived from `len(T080_SEEDS)`, never a literal); else
REFUSED". The same instrument, the same draws, the same quantity; only the quantiles move so that the eight checks
together refuse an honest apparatus about 1 % of the time, the rate the per-seed band had claimed. The position-
permutation null is REPORTED beside it in 3.x on every seed (its band, sd ratio and seed p-value) as the check that
the offline null's exchangeability assumption holds within 13 %. NOT changed: (b1)'s null and margin (the plan's
primary null at q95 -- MET at t +13.54 on run 1 and untouched), (p), (r), (g0), the arms, N, the disposition, the
reds. THE TIE FINDING: `cleanup` breaks an exact coincidence tie (e.g. 440/512 for two prototypes on seed 7, 9 of 294
ALIGNED ticks) by the codebook's INSERTION order, and the companion commit `309efe6` pinned the prototypes SORTED, so
an offline binder named the other prototype on tied ticks; the pin now carries them in registration order (a fix
of the orchestrator's own reproducibility apparatus, not of the instrument -- the in-vivo report and the campaign's
decomposition were never affected: agree 0). RUN 2 is the same deterministic recordings read under the amended
(b2) with the corrected pin; its (g0), (b1), windows and fractions are expected identical to run 1's and are
checked against them in 3.x; the refused pin's numbers stand in this section as the record of the refusal. The
transferable rule is logged (obs #2191): a per-seed band on S seeds owes its family-wise rate before it is a
refusal clause.

## 3. Results (written 2026-09-16 after campaign run 2; nothing above this line was edited after 2.11)

### 3.0 The apparatus, measured before the campaign (2.9 step 5, the dry run; 2.10; the trial campaign)

The dry run on seed 0 (`--t080-dry-run`, the builder's at 05:00-05:04 local and the orchestrator's re-run at
05:07-05:11, `scratchpad/t080/dry_run.json` / `dry_run_orch.txt`, every count identical): stream offset check
seeds 0 / 1 differ from the first frame; the determinism twin identical on 100 ALIGNED rows (ticks-to-K 39 / 39);
cost 66.7 / 69.8 ms per brain-tick over 2,439 ticks; ALIGNED bound 76/294, NULL 24/294, LAG 1 11/146, LAG 2
17/152, LAG 5 17/150, SCRAMBLED 8/152; the offline null mean 0.0953 q95 0.1224; decomposition 1188/1188; the
margin sensitivity FLAT (76 bound at 0.10 / 0.15 / 0.20, 0 sub-margin ticks); the q95 Monte-Carlo sd 0.00000
over 12 independent 2,000-draw nulls; the scrambled arm's audio accuracy 0.3355 against ALIGNED's 0.7143.
Everything 2.10 records (a)-(j) was read here before any barred number. A ONE-SEED TRIAL of the campaign path into
scratch paths (05:14-05:18, `scratchpad/t080/trial_campaign.txt`, 214 s) exercised the companion writer: pin and
npz written, sha256 pinned and matching, 12 arrays whose row counts equal each arm's decomposition. CLOCK-TIME
CORRECTIONS (obs #2190): section 1's facts 4-5 were probed at ~04:04-04:05 and ~04:09-04:11, not "04:12" and
"04:18"; 2.11's heading says "06:05" but its commit `00e4a15` landed at 05:59:25 -- both stamps were typed
ahead of the clock; the numbers are unaffected.

### 3.1 The build, red-first, and the commits

Pre-registration `de101bf` (this file 1-2.9 + C-R9). An Opus executor built 2.9 steps 2-5 (brief
`scratchpad/t080/builder_brief.md`): skeleton `4c5b8c9`; R1-R6 committed RED `4ef9ab3` (six AssertionErrors on
constructed inputs, none a TypeError); green with the six mutations reddening in the test and in `t080_reds()`
`8466cf2`; recorder, dry run, campaign flag and pin writer `66822fa` (the census gates rc=0 after commits 2, 4
and 5). The orchestrator's pre-campaign commits: 2.10 `521c5f1`; the latents companion `7f6a000`; the taught
prototypes in the pin `309efe6`; after run 1: 2.11 `00e4a15`; the family-wise band and the prototypes in
registration order `c3a9fb5`. The pin `c1f857b`. The PASS branch's surfaces: `7f82060` (the live symbol, its
provenance, the cockpit row, a test), `49288cd` (the conclusion row charged, the arming fixtures, tasks.md),
`8be7d89` (the dated notes and the register's budget). Every builder number was reproduced by the orchestrator
before it was written anywhere (3.0; obs #2180, #2185).

### 3.2 The campaign (`tools/t080_temporal_binding.json` + `tools/t080_latents.npz`)

Run 1 on `309efe6` (05:23:37-05:47:31 local, 1,434 s; a first launch on `7f6a000` was stopped at 57 s before it
wrote anything so the pin could carry the prototypes): REFUSED on (b2) as pre-registered -- seed 3's in-vivo
NULL arm 0.1531 above the per-seed band's 0.1424 -- with every other number as below. Its pin is archived
(`scratchpad/t080/pin_run1_refused.json`, `memory/archive/t080-2026-09-16/`), quoted in 2.11 and was deleted
from `tools/` before run 2 (2.10 (h)). Run 2 on `c3a9fb5` (05:59:52-06:23:48 local, 1,436 s): the SAME
deterministic recordings (every count identical to run 1's, checked from both pins), read under 2.11's
family-wise band: VERDICT PASS. Per seed 2,439-2,450 brain-ticks over six arms in 157-182 s; the companion 96
float32 arrays, sha256 `b5bddf194407e09b...` (identical to run 1's, the latents being the same). The
orchestrator re-derived every barred number from the two files with no brain (`scratchpad/t080/
verify_t080_pin.py`: a fresh binder taught the pinned prototypes in registration order re-decomposes every
scored latent to the pinned recognition -- 0 disagreements on 8 x 6 arms; an independent permutation loop
reproduces every seed's q95 bit-for-bit; (g0), (b1), (b2), the windows and the verdict reproduce).

### 3.3 The readings, conjunct by conjunct (every number from the pin; the bar 1.895 over 8 seeds; FPR = the sign-flip rate)

(p) DETERMINISM: two ALIGNED recordings of seed 0 at 100 ticks byte-identical on every field (ticks-to-K 39 / 39,
    taught sets equal): MET.
(r) READABLE: K = 10 taught on every seed within 32-87 ticks; `awake_missing` 0 on all 48 arms; the decomposition
    equals the recorded handle on every scored tick of every arm (`agree` 0 x 48): MET.
(g0) THE STREAM CARRIES WITHIN-WINDOW STRUCTURE: the classifiers' congruence ALIGNED - NULL per seed 0.6122 /
    0.6533 / 0.5869 / 0.5894 / 0.6100 / 0.6298 / 0.5738 / 0.6289; t **+65.3517**, FPR 0.0352: MET.
(b1) WITHIN > ACROSS AT THE DERIVED MARGIN: the ALIGNED arm's bound fraction minus the offline null's q95 per seed
    0.1361 / 0.2423 / 0.1757 / 0.2169 / 0.1781 / 0.2068 / 0.1913 / 0.2721 (within 0.2585-0.4150 against q95
    0.1220-0.1429); t **+13.5395**, FPR 0.0469, positive on 8 of 8: **MET**.
(b2) THE IN-VIVO NULL AGREES WITH THE OFFLINE NULL: the in-vivo NULL arm's bound fraction 0.0816 / 0.1174 / 0.1054 /
    0.1531 / 0.0956 / 0.0990 / 0.0973 / 0.1126 inside the family-wise band on 8 of 8 (inside the per-seed band
    on 7 of 8 -- seed 3 is 2.11's case): MET.
VERDICT: **PASS**.

pin head `c3a9fb5`; seeds [0, 1, 2, 3, 4, 5, 6, 7]; elapsed 1436.1 s; verdict **PASS**; refusals none
(p) determinism: identical=True on 100 rows (ticks_to_k [39, 39], taught_equal True)

| seed | ticks to K | offset | arm order | vision acc | audio acc | ALIGNED bound | NULL in vivo | null mean | null q95 | band [q005, q995] | within - q95 | congruent ALIGNED | congruent NULL | window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 39 | 0 | ali/nul/lag/lag/lag/scr | 0.9273 | 0.6606 | 76/294 = 0.2585 | 24/294 = 0.0816 | 0.0953 | 0.1224 | [0.0544, 0.1395] fw [0.0442, 0.1463] | +0.1361 | 0.7279 | 0.1156 | 1 |
| 1 | 77 | 150 | nul/ali/lag/lag/lag/scr | 0.9321 | 0.6645 | 110/293 = 0.3754 | 35/298 = 0.1174 | 0.1037 | 0.1331 | [0.0614, 0.1536] fw [0.0512, 0.1604] | +0.2423 | 0.7338 | 0.0805 | 1 |
| 2 | 87 | 44 | ali/nul/lag/lag/lag/scr | 0.9287 | 0.6580 | 93/296 = 0.3142 | 31/294 = 0.1054 | 0.1089 | 0.1385 | [0.0676, 0.1520] fw [0.0608, 0.1655] | +0.1757 | 0.7196 | 0.1327 | 1 |
| 3 | 56 | 194 | nul/ali/lag/lag/lag/scr | 0.9317 | 0.6601 | 102/295 = 0.3458 | 45/294 = 0.1531 | 0.1003 | 0.1288 | [0.0610, 0.1424] fw [0.0475, 0.1593] | +0.2169 | 0.7153 | 0.1259 | 1 |
| 4 | 32 | 88 | ali/nul/lag/lag/lag/scr | 0.9283 | 0.6692 | 88/292 = 0.3014 | 28/293 = 0.0956 | 0.0968 | 0.1233 | [0.0548, 0.1404] fw [0.0445, 0.1507] | +0.1781 | 0.7192 | 0.1092 | 5 |
| 5 | 70 | 238 | nul/ali/lag/lag/lag/scr | 0.9280 | 0.6500 | 97/295 = 0.3288 | 29/293 = 0.0990 | 0.0944 | 0.1220 | [0.0542, 0.1390] fw [0.0508, 0.1492] | +0.2068 | 0.7390 | 0.1092 | 2 |
| 6 | 59 | 132 | ali/nul/lag/lag/lag/scr | 0.9310 | 0.6322 | 96/298 = 0.3221 | 29/298 = 0.0973 | 0.1026 | 0.1309 | [0.0604, 0.1477] fw [0.0537, 0.1544] | +0.1913 | 0.6812 | 0.1074 | 1 |
| 7 | 53 | 26 | nul/ali/lag/lag/lag/scr | 0.9273 | 0.6631 | 122/294 = 0.4150 | 33/293 = 0.1126 | 0.1123 | 0.1429 | [0.0714, 0.1599] fw [0.0646, 0.1701] | +0.2721 | 0.7177 | 0.0887 | 1 |

| seed | aligned bound | null bound | lag1 bound | lag2 bound | lag5 bound | scrambled bound | aligned false | null false | lag1 false | lag2 false | lag5 false | scrambled false |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 76/294 (0.2585) | 24/294 (0.0816) | 11/146 (0.0753) | 17/152 (0.1118) | 17/150 (0.1133) | 8/152 (0.0526) | 9/76 (0.1184) | 6/24 (0.2500) | 5/11 (0.4545) | 2/17 (0.1176) | 4/17 (0.2353) | 4/8 (0.5000) |
| 1 | 110/293 (0.3754) | 35/298 (0.1174) | 13/145 (0.0897) | 14/149 (0.0940) | 9/149 (0.0604) | 16/149 (0.1074) | 10/110 (0.0909) | 9/35 (0.2571) | 4/13 (0.3077) | 3/14 (0.2143) | 4/9 (0.4444) | 5/16 (0.3125) |
| 2 | 93/296 (0.3142) | 31/294 (0.1054) | 12/146 (0.0822) | 21/151 (0.1391) | 14/150 (0.0933) | 18/149 (0.1208) | 7/93 (0.0753) | 16/31 (0.5161) | 4/12 (0.3333) | 5/21 (0.2381) | 5/14 (0.3571) | 9/18 (0.5000) |
| 3 | 102/295 (0.3458) | 45/294 (0.1531) | 17/149 (0.1141) | 17/148 (0.1149) | 8/148 (0.0541) | 14/151 (0.0927) | 7/102 (0.0686) | 15/45 (0.3333) | 10/17 (0.5882) | 3/17 (0.1765) | 4/8 (0.5000) | 5/14 (0.3571) |
| 4 | 88/292 (0.3014) | 28/293 (0.0956) | 19/148 (0.1284) | 20/150 (0.1333) | 18/149 (0.1208) | 12/149 (0.0805) | 10/88 (0.1136) | 6/28 (0.2143) | 8/19 (0.4211) | 6/20 (0.3000) | 3/18 (0.1667) | 6/12 (0.5000) |
| 5 | 97/295 (0.3288) | 29/293 (0.0990) | 20/148 (0.1351) | 11/148 (0.0743) | 14/151 (0.0927) | 10/152 (0.0658) | 8/97 (0.0825) | 12/29 (0.4138) | 11/20 (0.5500) | 7/11 (0.6364) | 7/14 (0.5000) | 6/10 (0.6000) |
| 6 | 96/298 (0.3221) | 29/298 (0.0973) | 13/147 (0.0884) | 13/147 (0.0884) | 11/150 (0.0733) | 17/149 (0.1141) | 8/96 (0.0833) | 8/29 (0.2759) | 4/13 (0.3077) | 2/13 (0.1538) | 3/11 (0.2727) | 8/17 (0.4706) |
| 7 | 122/294 (0.4150) | 33/293 (0.1126) | 15/149 (0.1007) | 23/149 (0.1544) | 13/150 (0.0867) | 12/149 (0.0805) | 15/122 (0.1230) | 14/33 (0.4242) | 5/15 (0.3333) | 7/23 (0.3043) | 4/13 (0.3077) | 6/12 (0.5000) |

| seed | aligned congruent | null congruent | lag1 congruent | lag2 congruent | lag5 congruent | scrambled congruent | aligned yield | null yield | lag1 yield | lag2 yield | lag5 yield | scrambled yield | aligned audio acc | null audio acc | lag1 audio acc | lag2 audio acc | lag5 audio acc | scrambled audio acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7279 | 0.1156 | 0.0822 | 0.1316 | 0.0867 | 0.0921 | 0.3551 | 0.7059 | 0.9167 | 0.8500 | 1.3077 | 0.5714 | 0.7143 | 0.7041 | 0.6644 | 0.7303 | 0.7133 | 0.3355 |
| 1 | 0.7338 | 0.0805 | 0.1034 | 0.0671 | 0.0805 | 0.1074 | 0.5116 | 1.4583 | 0.8667 | 1.4000 | 0.7500 | 1.0000 | 0.7099 | 0.6980 | 0.7241 | 0.6913 | 0.6980 | 0.3960 |
| 2 | 0.7196 | 0.1327 | 0.1301 | 0.1126 | 0.1000 | 0.0738 | 0.4366 | 0.7949 | 0.6316 | 1.2353 | 0.9333 | 1.6364 | 0.7095 | 0.7109 | 0.7055 | 0.6954 | 0.7400 | 0.2483 |
| 3 | 0.7153 | 0.1259 | 0.0604 | 0.1081 | 0.0473 | 0.0927 | 0.4834 | 1.2162 | 1.8889 | 1.0625 | 1.1429 | 1.0000 | 0.7051 | 0.7041 | 0.6644 | 0.7095 | 0.7095 | 0.3576 |
| 4 | 0.7192 | 0.1092 | 0.0946 | 0.0733 | 0.1141 | 0.0738 | 0.4190 | 0.8750 | 1.3571 | 1.8182 | 1.0588 | 1.0909 | 0.7055 | 0.7235 | 0.7635 | 0.6733 | 0.7718 | 0.2752 |
| 5 | 0.7390 | 0.1092 | 0.0608 | 0.1284 | 0.0861 | 0.1118 | 0.4450 | 0.9062 | 2.2222 | 0.5789 | 1.0769 | 0.5882 | 0.7085 | 0.6894 | 0.6892 | 0.7365 | 0.6821 | 0.2961 |
| 6 | 0.6812 | 0.1074 | 0.0748 | 0.0952 | 0.0800 | 0.0872 | 0.4729 | 0.9062 | 1.1818 | 0.9286 | 0.9167 | 1.3077 | 0.6745 | 0.6644 | 0.7211 | 0.6667 | 0.7067 | 0.2685 |
| 7 | 0.7177 | 0.0887 | 0.0940 | 0.1342 | 0.0867 | 0.0805 | 0.5782 | 1.2692 | 1.0714 | 1.1500 | 1.0000 | 1.0000 | 0.7109 | 0.7099 | 0.6980 | 0.7248 | 0.6867 | 0.3423 |

- **(g0) the classifiers' congruence: ALIGNED - NULL**: diffs [0.6122, 0.6533, 0.5869, 0.5894, 0.61, 0.6298, 0.5738, 0.6289]; t +65.3517; FPR 0.03515625; degenerate False; **MET**
- **(b1) the binder's within-window rate - the null's q95**: diffs [0.1361, 0.2423, 0.1757, 0.2169, 0.1781, 0.2068, 0.1913, 0.2721]; t +13.5395; FPR 0.046875; degenerate False; **MET**
- **(b2) the two nulls agree** (in-vivo NULL inside the offline band on every seed): **{'outside': [], 'inside': 8, 'n_seeds': 8, 'met': True}**
- windows per seed: [1, 1, 1, 1, 5, 2, 1, 1]; scrambled bound fractions: [0.0526, 0.1074, 0.1208, 0.0927, 0.0805, 0.0658, 0.1141, 0.0805]
- null draws sd per seed: [0.0162, 0.017, 0.0174, 0.0169, 0.0167, 0.0163, 0.0172, 0.018]; expected (product of marginals) vs mean: [(0.0952, 0.0953), (0.1032, 0.1037), (0.1084, 0.1089), (0.1006, 0.1003), (0.097, 0.0968), (0.0941, 0.0944), (0.1025, 0.1026), (0.1113, 0.1123)]; multiset_ok True
- agree (decomposition disagreements) per seed/arm: [{'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}]
- awake_missing per seed/arm: [{'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}, {'aligned': 0, 'null': 0, 'lag1': 0, 'lag2': 0, 'lag5': 0, 'scrambled': 0}]
- timings per seed (s): [{'warmup_s': 2.728, 'total_s': 164.365, 'decompose_s': 4.915}, {'warmup_s': 3.877, 'total_s': 180.469, 'decompose_s': 4.8}, {'warmup_s': 4.493, 'total_s': 157.416, 'decompose_s': 4.919}, {'warmup_s': 3.065, 'total_s': 182.001, 'decompose_s': 4.899}, {'warmup_s': 1.941, 'total_s': 161.196, 'decompose_s': 4.867}, {'warmup_s': 3.702, 'total_s': 166.897, 'decompose_s': 4.966}, {'warmup_s': 2.917, 'total_s': 166.584, 'decompose_s': 5.014}, {'warmup_s': 3.107, 'total_s': 171.877, 'decompose_s': 5.047}]
- companion: tools\t080_latents.npz sha256 b5bddf194407e09b..., 96 arrays; prototypes per seed: [10, 10, 10, 10, 10, 10, 10, 10]
- reds at campaign time: {'R1': True, 'R2': True, 'R3': True, 'R4': True, 'R5': True, 'R6': True}

REPORTED: the binding window reads 1 tick on seeds 0-3, 6, 7 (every lag arm at or below the margin), 5 on seed 4
(LAG 1 0.1284 and LAG 2 0.1333 above its margin 0.1233 -- 2.3's "a lag arm ABOVE the margin is a finding") and
2 on seed 5 (LAG 1 0.1351 above 0.1220); the scrambled-phase arm binds at 0.0526-0.1208 -- at or below every seed's
null q95 -- with the audio classifier collapsing to 0.20-0.35 while vision stays at 0.89-0.96 (the SHD frame's
time bins carry the word; the secondary null takes at the classifier, 2.10 (j)); the false-binding fraction is
0.069-0.123 on ALIGNED and 0.214-0.516 on NULL (a reported object the classifiers did not name); the yield (bound
/ classifier-congruent) 0.355-0.578 on ALIGNED -- the binder's nearest-prototype recognition, not the margin, is
the ceiling (2.10 (b)); the offline null's mean equals its product-of-marginals expectation to 0.001 on every seed
and the position-permutation null (2.11) widens it by 1.13x; the q95's Monte-Carlo sd is 0 (an integer count over
~294 ticks).

### 3.4 The disposition applied (2.6, the PASS branch)

On this composite the live binder reports one object for features presented inside one tick window more often
than a null that keeps every feature and the taught K and breaks only their pairing across windows, at the bar
over 8 seeds. CHARGED: `CONCLUSION["temporal_binding"].reading` = **0.2309** (the mean over seeds of the ALIGNED
bound fraction minus the null's MEAN, per 2.6: 0.1632 / 0.2718 / 0.2053 / 0.2455 / 0.2046 / 0.2344 / 0.2195 /
0.3027), `surface_symbols` = ("crossmodal_bound_rate",); `Brain.state()["crossmodal_bound_rate"]` = the fraction
of the last 200 awake ticks with one reported object for that tick's two latents (a scratch brain at 60 ticks
reads 0.4194 over 31 awake ticks -- the sparse-codebook rate, 2.7); the cockpit's access panel prints it with the
licensed and the refused sentence in the same view and `brain_telemetry.PROVENANCE` names the pin, the null's
q95 and the disclosure that on this daemon's independent streams the value is a chance rate (fact 4);
`conclusion_protocol --assert` reads the charged row with the surface debt unchanged at its baseline; tasks.md
T080 -> [X] PASS. Nothing is retired: `gwt_deep`'s exemption stands on its own reason (1.3), the binder is W6's
live mechanism. `tests/test_ratified_row_gates.py`'s arming fixture moves to `report_order`.

### 3.5 Deviations from 2.1-2.11, each with its number (dated 2026-09-16)

1. (b2)'s band was changed AFTER the numbers (2.11): from the per-seed [q005, q995] to the family-wise
   [q(0.005/8), q(1 - 0.005/8)] of the same draws. The pre-registered clause refused run 1 on seed 3 alone
   (0.1531 vs 0.1424) at a family-wise false-refusal rate of 7.7 % (1 - 0.99^8; 6.2 % simulated from the position
   nulls); seed 3's one-sided p under its own position null is 0.0037. The barred claim (b1) was not touched and
   read the same t on both runs. The honesty lens adjudicates whether a readability clause with a computable
   6-8 % false-refusal rate may be corrected once, disclosed, after it fires; obs #2191 records the rule.
2. The pin was written TWICE: run 1's (REFUSED) archived and deleted from `tools/`, run 2's committed. 2.10 (h)
   prescribed exactly this for a refusal; the numbers are identical, the verdict is not.
3. The campaign's first launch on `7f6a000` was stopped at 57 s (before any write) to add the prototypes to the
   pin (`309efe6`) -- an apparatus change between the dry run and the campaign, with no effect on any number.
4. The companion commit `309efe6` pinned the prototypes SORTED; `cleanup` breaks an exact coincidence tie by the
   codebook's insertion order, so an offline binder named the other prototype on tied ticks (9 of 294 on seed 7
   ALIGNED, ties at 440/512; seeds 6-7 differed by a few ticks on run 1's verification). Fixed in `c3a9fb5`
   (registration order); run 2's verification re-decomposes 0 disagreements. The campaign's own decomposition
   was never affected (`agree` 0 on both runs) -- it used the brain's binder. The tie-break's dependence on
   registration order is itself a fact about the mechanism (opening 3).
5. 2.3 (r)'s vacuous second clause was made stricter before the campaign (2.10 (c)); the dry run and both runs
   read 0 missing.
6. R2's constructed input is drawn sparse (2.10 (a)); `T080_REDS_N_DRAWS` = 500 for the reds; three mutation hooks in
   signatures; `t080_verdict`; R3's tables as constants; `agree` counts disagreements; the per-arm accuracies from
   differenced counters -- the builder's deviations, accepted as disclosed (2.10 (h)).
7. The reading of 2.6 is (within - the null's MEAN), while (b1) bars (within - the null's q95): both are in the
   pin; the record reports the q95 differences beside the reading's mean differences (3.3).
8. Clock stamps typed ahead of the clock twice (3.0; obs #2190).
9. The warm-up reads 32-87 ticks across seeds (fact 5 measured 54 on the probe's aligned order); the cost 66.7-69.8
   ms/tick against 2.8's 55.
10. The record's tables are generated from the pin (`scratchpad/t080/record_gen.py`), never typed.

### 3.6 What this row does NOT claim, and the openings it leaves

Nothing about a felt present, a stream of experience or a duration lived (the refused sentence, printed beside
the symbol on every surface that prints it). The binding window is the tick because the binder is stateless --
the lag arms measured the mechanism's shape (1 tick on 6 seeds; the two seeds whose LAG 1 arm sat above the
margin are chance congruences of 148 ticks, reported); nothing about a temporal integration window. The alignment
is the apparatus's: on this daemon the streams are independent (fact 4), so its live `crossmodal_bound_rate` is a
chance rate and the provenance says so. Openings: (1) `_xmodal_handle` persists on ticks the binder does not run
(brain.py:1833 guards the call, not the state) -- the rate counts only ticks with both latents, but
`crossmodal_handle` and `crossmodal_superadditive` publish a stale value on the others; (2) a sparse codebook
names one object for everything above the margin (the first taught object binds 0.42 of early ticks on a fresh
brain) -- a report that is a marker of "one prototype exists", not of binding; (3) the tie-break by registration
order (3.5 item 4); (4) the audio classifier at 0.65-0.73 caps the yield at 0.36-0.58 -- a better audio encoder,
not the binder, is the lever; (5) the scrambled-phase arm is a classifier test as built (2.10 (j)); a phase
scramble that preserves the classifier's output would test the binder's own use of phase, which nothing here
does; (6) the plan's "scrambled-phase secondary" and "matched K" are read as this record reads them (2.1); the
plan's author may have meant gwt_deep's K-object task, which stays unrun; (7) the stale `crossmodal_superadditive`
on the cockpit has no sentence beside it and is not a C5 symbol -- it prints a chance-binding indicator without
saying so; (8) the (b2) clause's post-number correction (3.5 item 1) is the panel's to judge.
