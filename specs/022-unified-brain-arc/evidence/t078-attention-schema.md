# T078 -- the attention schema on the live loop: the gates first (specs/022 Phase 4 / C4; plan row :163; invention #8; SC-009's first half)

Sections 1-2 are the PRE-REGISTRATION, written 2026-09-15 (12:05-14:50 local; git's clock is the record of when they froze)
BEFORE a line of this row's code exists in the tree and frozen at the commit that lands them; 2.0 records the Codex design
read that shaped them; amendments are dated appended sections; section 3 is written after the runs. Every number in
sections 1-2 is a source line, a plan constant, or an apparatus probe named with its script and the numbers it printed
(no gate statistic has been read on any brain -- the only brain probe, `scratchpad/t078/probe_streams.py`, printed
marginals and counts, quoted in 1.3, and no MI, no decoder accuracy).

## 1. Territory, measured before the design (each fact with its source line, read at HEAD `da93dd0`)

1. THE SUBJECT MISMATCH is the first fact. The algorithm this row adopts (`docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md`
   5.2) needs a cued multi-step task with a gap ON THE BRAIN ("a `tick(stimulus=None)` port, an EpisodeDriver that owns
   the schedule ... the largest and riskiest single item"). What the tree built for that line (T035,
   `substrate/episode_driver.py`) drives the KERNEL (`EpisodeDriver.tick(kt, stimulus)` on `KernelTick`); `Brain.tick()`
   takes no arguments (brain.py:3808). The attention state the ACS would model lives on the BRAIN loop:
   `cortex/learned_attention.LearnedAttention` (constructed brain.py:1066, `dim=64, n_bands=4`), applied in
   `_attend_real_stream` (:1545-1572): `gated = attend(latent, goal=None)`; `chan` = the four band means of the gated
   features = the arm's sensory channel; `focus` = the winning band and its LIF rate (:1569-1571); `goal` is None
   "documented rather than faked" (:1563-1566). It is TRAINED every awake tick by its local three-factor rule with
   reward = `lp_mean - threat` (:2650-2661), so its weights move -- measured in 1.3, not read. `brainstem/
   autonomous_loop.py:976-995` is the record the algorithm cites: "this loop has no gap". So G3 AS WRITTEN ("the held goal
   at end-of-gap ... the committed first action") has no gap task to bind; this row binds it to the loop that exists
   (2.1) and names the successor that owes the port.
2. What the live loop carries per awake tick, unmodified, and where it is read: the gate `LearnedAttention.gate` (four
   values in [0,1), :181-184) and `focus` (`(band, strength)`, :176-179); the workspace's ignition
   `_access_state["ignited"]` (brain.py:3154-3158) and its sustained hold (`report_channel.snapshot` :98-111:
   availability / strength); the perceptual precision proxy `_pc_surprise` and its EMA (:2406-2408; the metacognitive
   input's `pc_dev` :2482-2485); the BG decision margin `arm._decision_margin` (:2492); the committed action
   `res["action"]` (autonomous_loop.py `step` :1895-1919, the E-STOP path returns EMERGENCY_STOP, the normal path :2474-
   2486 returns the label chosen by the BG gate); the percept's identity per tick: `_perceive_real` (:1629-1660) reads
   frame `i = _real_ptr % len(_real_frames)`, classifies it (`_perceived_digit = pred`) and scores it against
   `label = _real_labels[i]`; the normalized 64-d latent is `res["percept"]`. The CONTENT ground truth for G4 is the
   frame's LABEL, recoverable per tick as `_real_labels[(_real_ptr - 1) % n]` when `_real_ptr` advanced on that tick.
3. APPARATUS FACTS, measured 2026-09-15 10:42 (`scratchpad/t078/probe_streams.py`, seed 0, 300 ticks, scratch home,
   146 awake; marginals and counts only -- NO MI, NO decoder, no gate statistic): the committed action takes ~12
   distinct labels (ENGAGE 48 of 146, six joint rotations 9-14 each); the focus band reads 3 / 2 / 0 / 1 on 82 / 37 / 22 /
   4 awake ticks (concentrated, not constant); the focus STRENGTH (the LIF winner's rate) is near zero -- mean 0.028, max
   0.100 -- so that field of `u_t` is nearly degenerate; the four gate values vary (means 0.12-0.19, sds 0.03-0.06);
   IGNITION fires on 1 of 146 awake ticks (0.7 %), so the `ignited` field is a near-constant on the scratch brain (the
   opposite rail from G1's saturation; disclosed); the vision classifier scores one frame per awake tick over 6+ digit
   classes; (focus_t, action_t+1) fills 39 joint cells at 145 pairs. Consequence fixed here: N = 1,200 ticks per seed
   (about 585 awake, about 580 consecutive awake pairs) and a declared minimum of 10 pairs per observed joint cell (2.3).
4. The C5 row this instrument fills: `substrate/conclusion_protocol.py` :122-133 -- licensed "the system's report of
   what it attends to predicts which input governs its next output"; refused "this licenses no claim that the schema is
   felt, and no claim that the system has a self"; dissociation "a report channel severed from the attention state must
   lose the prediction"; reading None; surfaces `()` ("T078 unbuilt"); `C4_TASK["attention_schema"] = "T078"` (:91).
   The gate rule 1.5 of T077's pre-registration binds here too: once tasks.md marks the row `[X]` the row must declare
   a surface symbol (the arming arm), and `tests/test_ratified_row_gates.py:154` now asserts `_task_done("T078") is
   False` -- the fold of T077 re-pointed that known-negative here (evidence/t077-metacognition.md 3.5 item 9), so the
   commit that marks T078 `[X]`, if any, re-points it again (to T079) in the SAME commit (obs #2179).
5. Governing words, quoted in `scratchpad/t078/design.md` section 0: tasks.md :163 (the row); docs/INVENTION_REGISTER.md
   :111-120 (invention #8's RED-1/2/3, CONTROL, DISCLOSURE, NULL); SYNTHESIS.md :65 ("G3/G4 carried with their stop/void
   wording intact -- no ablation matrix is interpretable below the shuffle floor; a G4 failure VOIDS rather than
   fails"); spec.md FR-028 :145; PLAN-FINAL :914 (C4) and Part F's preamble (S = 8, t > 1.895, FPR printed, no majority-
   of-seeds rule); docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md 3 (G3, G4 with their reds), 4 (`u_t` = [gate(B),
   precision(L), onehot(focus|NONE)(B+1), focus_strength, ignited, drive_margin, sustain] = 2B + L + 5 = 15 at B = 4,
   L = 2), 9.1-9.5 (pin the ACS; two tables; baselines first; the rigging check; a null is a successful outcome), 12
   (build order: G3 before any ACS).
6. The apparatus this row REUSES (P2; no second implementation): the paired one-sided t and its bar from
   `substrate/paired_contrast` (`contrast`, `T_BAR = 1.895`, the exact sign-flip FPR; module-level import -- the harness
   is EXEMPT and a lazy import of it stales five dormancy rows, T077 2.10); the recording pattern of
   `metacognitive_monitor.t077_record_brain` (a scratch brain under `throwaway_home()`, per-tick rows, an instance's
   method wrapped to capture what it is fed and what it returns); the mutated-scorer reachability pattern of
   `metacognitive_monitor._t077_mutated_scorer` / `t077_reds` (every red reddened IN FACT, in the test and in the
   module's self-check line); the binary entropy `integration_measure.source_entropy`. NOT in the tree (graph-exists,
   2026-09-15): a mutual-information estimator with a permutation floor; a held-out decoder with a label-shuffle band.
   Both are this row's new symbols, in `cortex/attention_schema.py` (the row's named file).
7. What T077's fold binds on this design (evidence/t077-metacognition.md 3.5 items 9-27, 3.7, 3.8; `design.md` section
   8): reds reddened in fact by mutated scorers; no positive control that is an identity of the statistic; every per-tick
   input of every barred conjunct in the pin; every declared constant's sensitivity computed here, not later; a neuter
   whose pinned value equals the observed value is entailed, not measured; surfaces and annotations land BEFORE the
   record commit; no scratch citation without its command and numbers; cost re-measured on the built apparatus; the
   disposition names in advance whether a surface symbol is declared; an incumbent status repeated from a module's
   record is re-probed first (item 10 of design.md 8: the gate's weights are measured to MOVE in 3.0's apparatus line).

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex; sent 11:55 on head `ac7b04b` and REFUSED by its usage limit; re-sent after the 14:33 reset)

DISCLOSED AT THE FREEZE: the Codex design read of `scratchpad/t078/design.md` (sections 0-8, its section 7 questions 1-5
and section 8's question 6) was sent at 11:55 and refused by Codex's usage limit ("try again at 2:33 PM"); the re-send is
armed for 14:34 (`scratchpad/t078/codex_design_send.ps1`, queued behind the T077 closing report's re-send). This file is
frozen BEFORE the read, so the read cannot shape 2.1-2.9 in place; its points are folded as a dated appended 2.10 BEFORE
any brain recording exists (2.9 steps 5-6 are held until 2.10 lands or the read is refused a second time, in which case
2.10 says so). What the read WOULD have shaped and what this file fixes without it: (1) G3's scoping to the live loop
(2.1, 2.7: a PASS says only what it says); (2) G4's "indistinguishable" as two positive contrasts with a PASS-WITH-LEAK
branch (2.3); (3) "governs" as the largest gated contribution (2.11.2, not per-tick neuters); (4) the third factor's
outcome as the informative-band accuracy (2.11.3); (5) no control that is the method, no null-as-conjunct, every bar
absolute (2.3, 2.5). If 2.10 changes a gate's FORM, the instrument is re-built before the dry run and the tests' reds are
re-run; if it changes a NUMBER, the old number stays in this file and 2.10 carries the new one with the reason.

### 2.1 The mechanism -- PART I ONLY is built before Part I's record: two gates, their reds, one recording

`cortex/attention_schema.py` (new; registered in `__init__.py` REGIONS, `run_selfchecks.py` CORE, and
`dormancy_gate.EXEMPT` with the reason "T078 Part I instrument on a scratch brain; the ONE live wire (Head A's goal) lands
in Part II or the row STOPS at G3 / VOIDS at G4 and the exemption is retired with the row") carries:

- `mi_bits(x, y)`: the plug-in mutual information in bits of two discrete label sequences of equal length (joint counts
  over the observed cells; no smoothing -- the floor absorbs the plug-in bias).
- `mi_read(x, y, n_perm=2000, seed)`: returns `{n, cells, mi, perm_q95, perm_median, readable, refusal}`; the floor is
  the 95th percentile of `mi_bits(x, permuted y)` over 2,000 permutations of y (a seeded `random.Random`, the same
  convention as `track_error`); REFUSED (`readable=False`, no `mi`) when `n < N_MIN_PER_CELL * cells` with
  `N_MIN_PER_CELL = 10` DECLARED (its consequence computed in 2.8: at the 300-tick probe's 145 pairs / 39 cells the
  reading refuses; at 1,200 ticks it does not) or when x or y has one class.
- `decode_heldout(X, y, seed, n_shuffle=200)`: a multinomial logistic decoder (torch, full-batch gradient descent, 200
  steps, lr 0.1, L2 1e-3, seeded init; CPU or CUDA identically by seed) trained on the FIRST half of the rows in tick
  order and scored on the SECOND half (prequential halves, no lookahead); returns `{n_train, n_test, acc, shuffle_q95,
  shuffle_median, readable}` where the band is 200 decoders trained on label-permuted train halves (batched as one
  (200, d, K) tensor), scored on the true test labels. Capacity is matched by INPUT DIMENSION: every decode in G4 runs
  on d = 15 inputs (the `u_t` vector; the content projected to 15 dimensions by a fixed Gaussian random projection with
  seed `crc32("t078-proj") + seed`); the raw 64-d content decode is REPORTED as the ceiling, never barred.
- `control_state(brain)`: the candidate `u_t` (A4's 15-vector) from the live signals of 1.2 -- gate(4), precision(2:
  `_pc_surprise` and the deviation `pc_dev` of :2482-2485), onehot(focus | NONE)(5), focus_strength, ignited (0/1),
  drive_margin (`arm._decision_margin`), sustain (the workspace snapshot's `strength`) -- content-blind BY TYPE: a pure
  function of those fields, asserted by a test that two brains with equal control fields and different latents give a
  bit-identical `u_t`; IMPOVERISHED by a static assert (no weights, salience, surprise, reward in the vector).
- `t078_record_brain(seed, n_ticks)`: a scratch brain (`make_brain(seed, persist=False)` under `throwaway_home()`),
  `_learned_attn.attend` wrapped to capture its input latent, its gated output, the gate and the focus; per tick: kind,
  awake, `u_t`, gate(4), focus band, focus strength, ignited, sustain, drive_margin, pc_surprise, the committed action
  label, the perceived digit, the frame LABEL (1.2) when a frame was perceived, `chan`(4), the gated latent (64, rounded
  to 4 decimals). The pin (`tools/t078_attention_schema.json`, written ONCE) carries EVERY per-tick field except the
  64-d latents, which go to the companion `tools/t078_latents.npz` (float32, per seed; sha256 in the pin) -- so every
  reading below re-derives offline (T077 3.5 item 23).
- `run_t078_campaign()` (`--t078-campaign`): records seeds 0-7 sequentially in one process (N = 1,200 ticks each), runs
  the determinism precondition (2.2), computes G3 and G4 and every reported reading from the recording, contrasts with
  `paired_contrast.contrast`, prints every conjunct MET / NOT MET / STOP / VOID with t and FPR, pins once.

PART II (the ACS, its report, the third factor, the legibility experiment, the sham -- design.md sections 4-5, invention
#8's RED-1/2/3) is pre-registered in 2.11 in full and is BUILT ONLY IF Part I's record reads G3 PASS and G4 not VOID;
its apparatus is not written before that record. This is the algorithm's own build order (12: G3 before any ACS) and
SYNTHESIS :65's wording.

### 2.2 The arms and the pairing (one command, seeds 0-7, a scratch brain; the daemon's brain is never touched)

RECORDING: seeds 0-7, N = 1,200 ticks each, sequential in one process, under a throwaway NORD_HOME. Per tick the campaign
records the fields of 2.1. DETERMINISM PRECONDITION (G5 on this loop, cheap): before the campaign, two 100-tick
recordings of seed 0 must be byte-identical on (focus band, gate, action, perceived digit, label); a difference REFUSES
the campaign (the instrument's own fault: nothing below is readable on a non-deterministic recording).

G3 PAIRS: consecutive AWAKE ticks (t, t+1) -- X_t = the focus band at t (4 classes), Y_{t+1} = the committed action at
t+1 (the label string; EMERGENCY_STOP never occurs under `_safe_ctx`, and if it does the pair is dropped and counted).
Pairs straddling a sleep bout are dropped and counted (T077 3.2's amendment: no stale pairing). The same pairs serve
every G3 reading.

G3 READINGS (per seed; every one over the same pairs): (primary, BARRED) MI(focus_t ; action_{t+1}) against its floor;
(the estimator's POSITIVE CONTROL, BARRED as a precondition) MI(action_t ; action_{t+1}) against its floor -- actions are
autocorrelated on this loop, so the estimator must read this above its floor or it is blind on this stream and G3 is
REFUSED, not stopped; (reported, never barred) MI(gate16_t ; action_{t+1}) where gate16 is the four gate values each
split at its own seed median (16 cells); (reported) MI(focus_t ; action_t), the same-tick reading, to separate "the
state at t predicts t+1" from co-occurrence; (reported) the joint-cell count and n per seed.

G4 ROWS: awake ticks on which a frame was perceived (the label exists). Target = the frame's LABEL (10 classes; the
perceived digit is a second target, reported). Inputs: `u_t` (15); the content = the gated 64-d latent projected to 15
(matched); the raw gated 64-d latent (the reported ceiling); `chan` (4, reported). Per seed, four decodes, each with its
200-shuffle band.

### 2.3 The statistic and the verdict rule (written first)

Every barred contrast is `paired_contrast.contrast(name, diffs)` over the 8 seeds: a paired one-sided t against the bar
1.895 with the exact sign-flip FPR printed; a zero-variance sample is DEGENERATE and MET iff its common value is > 0
(the lifted rule, verbatim). No majority-of-seeds rule; no statistic normalized by a quantity a control destroys.

PART I, in order:
(p) DETERMINISM: the 100-tick twin recordings byte-identical -> else REFUSED (no reading of any kind).
(r) READABLE: every seed's G3 primary reading and positive-control reading are `readable` (n >= 10 pairs per observed
    cell; two classes on both axes) and every seed's G4 decodes are readable (n_test >= 50 and both halves carry >= 2
    classes) -> else REFUSED with the instrument's fault named (the same REFUSED branch as T077 2.6).
(g3c) THE ESTIMATOR SEES: t(MI(action_t; action_{t+1}) - perm_q95) > 1.895 -> else G3 REFUSED ("the estimator is blind
    on this stream"; not a STOP -- a STOP is a finding about the loop, a REFUSAL is a finding about the instrument).
(g3) TASK BINDING: t(MI(focus_t; action_{t+1}) - perm_q95) > 1.895 -> G3 PASS; else G3 STOP.
(g4a) `u_t` DECODES CONTENT: t(u_acc - u_shuffle_q95) > 1.895 (the 15-d `u_t` decode of the label above its own band).
(g4b) CONTENT DECODES BETTER: t(c15_acc - u_acc) > 1.895 (the matched 15-d content decode above the `u_t` decode).
    G4 VOID iff (g4a) MET and (g4b) NOT MET -- `u_t` carries the content and the content decode cannot be told from it:
    content-blindness is a naming convention here and every downstream schema reading is VOID, not weakened.
    G4 PASS iff (g4a) NOT MET (the schema vector decodes nothing above chance) -- content-blind in information.
    G4 PASS-WITH-LEAK iff (g4a) MET and (g4b) MET -- the vector leaks, but far below the content; the leak's size
    (mean over seeds of u_acc - u_shuffle_q95, in accuracy points) is carried into Part II as a disclosed tolerance and
    into the C5 interpretation: a report that predicts the governing input could be reading the leak.
The 15-d projection is the matched arm; the raw 64-d content decode and the `chan` decode are reported beside (g4b).
PART I VERDICT: G3 {PASS, STOP, REFUSED} x G4 {PASS, PASS-WITH-LEAK, VOID, REFUSED}, every reading with its number.

PART II's conjuncts are in 2.11; they are read only after a Part I record of G3 PASS and G4 PASS or PASS-WITH-LEAK.

### 2.4 The reds (RUN RED before the mechanism; in-command with their controls thereafter; each reddened by a mutated scorer)

R1 THE FLOOR ABSORBS THE PLUG-IN BIAS -- a constructed i.i.d. pair of label streams (|X| = 4, |Y| = 12, n = 600) reads
   `mi` at or below `perm_q95`; the control: y := f(x) (a deterministic map) reads `mi` above `perm_q95` by more than
   1 bit. Mutation that reddens it (run, in-file): a floor computed on UN-permuted y (every draw equals the reading) --
   the i.i.d. stream then reads exactly at its "floor" and the control's margin vanishes.
R2 THE POSITIVE CONTROL IS NOT AN IDENTITY -- MI(a_t; a_{t+1}) on a constructed i.i.d. action stream reads AT its
   floor (below q95), and on a constructed Markov action stream with 0.7 self-transition reads above it: the control can
   fail, so its passing means something. Mutation: an estimator that scores y against itself.
R3 THE REFUSAL -- 145 pairs over 39 cells (the probe's shape) REFUSE (`readable=False`, no `mi`); 600 pairs over 39
   cells are readable; a one-class axis refuses. Mutation: N_MIN_PER_CELL := 0.
R4 THE DECODER'S BAND -- a constructed 15-d input with the label planted in one coordinate decodes above its
   shuffle band; the same input with labels permuted decodes inside it; a 200-shuffle band on n_test = 300 has q95 below
   0.2 (chance 0.1 over 10 classes). Mutation: a decoder scored on its TRAIN half (the band collapses to 1.0).
R5 CONTENT-BLIND BY TYPE -- two `control_state` calls with equal control fields and different latents are bit-identical;
   a vector that carries a latent coordinate is caught by the static field assert. Mutation: append `chan[0]` to `u_t`.
R6 THE VOID RULE CAN FIRE -- on a constructed recording where `u_t` IS a one-hot of the label, (g4a) MET and (g4b) NOT
   MET -> VOID; on a constructed recording where `u_t` is noise, (g4a) NOT MET -> PASS. Mutation: swap the operands of
   (g4b).
Each red is a test in `tests/test_attention_schema.py` AND a line of the module's self-check (`t078_reds()`, six
booleans asserted by main()), committed RED against stubs that return null readings (the tests RUN on their constructed
inputs and FAIL on assertions -- not an ImportError; T077 3.5 item 10).

### 2.5 The controls that are not the method

The permutation floors (every MI), the label-shuffle bands (every decode), the same-tick MI, the action-autocorrelation
control, the matched-dimension content decode, the determinism twin. None reads the schema's output as its own input --
there is no schema in Part I.

### 2.6 Disposition (fixed in advance; the honesty lens adjudicates any reading of these words)

- G3 STOP (the loop's action path does not consume the attention state above the shuffle floor): NO ACS is built, NO
  third factor, NO wire to brain.py; the record says STOP with every number; `cortex/attention_schema.py` stays as the
  Part I instrument (EXEMPT, its reds live in run_selfchecks); tasks.md's row stays `[ ]` with a dated note "STOPPED at
  G3 -- the successor row owes the brain-side stimulus port (5.2) first"; NO surface symbol is declared and none is
  needed (the row is not `[X]`); `CONCLUSION["attention_schema"].reading` stays None; the C5 sentence is NOT written.
  A STOP is the algorithm's own legitimate terminal state (SYNTHESIS :65), not a null of the mechanism.
- G4 VOID: the same as STOP, the note reading "VOID at G4 -- content-blindness is a naming convention on this loop";
  additionally the record names the leak's size and the two decodes.
- G3 PASS and G4 PASS / PASS-WITH-LEAK: Part II is built per 2.11 and 2.9 steps 6-9, its verdict per 2.11.
- REFUSED on (p), (r) or (g3c): no reading; the instrument's own fault named; nothing fires; the row stays open.
- On any Part I outcome the incumbents (`learned_attention`, the loop, the workspace) are untouched: Part I adds no wire.

### 2.7 What this row does NOT claim

No experiential claim (the C5 refused sentence, verbatim on every surface). Part I reads ONE loop: the scratch brain's
own attention control state and committed actions at the deployed width, 1,200 ticks, 8 seeds; nothing about the
daemon's brain, chat, reasoning or the arm's real hardware. G3 on this loop is NOT the algorithm's G3 (no gap, no held
goal): a PASS says the action path consumes the attention state one tick later on this loop; a STOP says it does not.
The perceived digit's LABEL is the content ground truth; the classifier's own accuracy is reported, not adjudicated.
No number in this file is a reading.

### 2.8 Widths, cost, host, declared constants and their sensitivity

Width: the deployed brain (`make_brain`, `LearnedAttention(dim=64, n_bands=4)`), N = 1,200 ticks x 8 seeds, ONE arm
(there is no lesion in Part I) plus the 100-tick determinism twin. Cost: T077's campaign measured 1,139.7 s for 600 x 8
x 2 brains plus arm D; this recording is 9,600 brain-ticks + 200 -> projected ~10-12 min sequential, RE-MEASURED on the
built apparatus in 3.0 before the campaign (T077 3.5 item 20: a projection missed by 1.9x). The floors: 2,000
permutations x 4 MI readings x 8 seeds of an O(n) count -- seconds; the bands: 200 batched decoders x 4 decodes x 8
seeds on torch -- seconds on CPU, CUDA where present. Seeds sequential (`throwaway_home()` sets a process-wide env var;
not thread-safe); the permutation floors run in threads (3.14t, GIL off). DECLARED CONSTANTS and the consequence of
each, computed here: N_MIN_PER_CELL = 10 (at the probe's 145 pairs / 39 cells the primary reading REFUSES -- so any
apparatus dry run below ~400 pairs is refused by construction and cannot preview a G3 number; at 1,200 ticks, ~580 pairs
over <= 48 cells, it reads); n_perm = 2,000 (the q95's Monte-Carlo sd on an MI null of ~0.05 bits is ~0.002 bits at
2,000 draws -- computed on the constructed R1 stream in 3.0); n_shuffle = 200 (q95 of a 200-draw band is the 190th
order statistic; its sd is computed on the constructed R4 stream in 3.0); the projection seed; the decoder's 200 steps /
lr 0.1 / L2 1e-3 (fixed; the raw-64 ceiling decode shows whether capacity, not information, limits the matched decode).
Host: Python 3.14.6 free-threaded, torch 2.11.0+cu128 on the RTX 5070 where the brain already uses it; GeNN not used.

### 2.9 Commit order (one change each; the stubs before the reds so the reds RUN red; surfaces before the record)

1. This file + PLAN-FINAL Part F row C-R7 (T078) -> 2. `cortex/attention_schema.py` skeleton: the six functions of 2.1
as STUBS returning null readings (`mi_read` -> `readable False`, `decode_heldout` -> `acc None`, `control_state` -> the
15 zeros, ...), main() calling `t078_reds()` which returns all False, the three registrations (REGIONS, CORE, EXEMPT) ->
3. `tests/test_attention_schema.py`, R1-R6 committed RED (each runs on its constructed input and fails an assertion; the
output pasted) -> 4. the instruments green, each red reddened by its mutated scorer in-file and in `t078_reds()`; census
gates -> 5. the recorder + `--t078-campaign` (Part I) + the determinism precondition + the pin writer; the apparatus dry
run on ONE seed at 300 ticks (its G3 reading REFUSES by N_MIN; its output named with its script in 3.0) -> 6. the
campaign, the pin (once) -> 7. surfaces and annotations per 2.6 (tasks.md's dated note, PLAN-FINAL C4 note, spec.md
SC-009 note; on a STOP / VOID nothing in `conclusion_protocol` or `brain_telemetry` changes) -> 8. section 3 the record ->
panel (3 lenses + refuters judging CLAIM and FIX) -> fold -> Codex closing report -> the battery on the stacked head.
Part II (if licensed): 2.11's own order, reds first, after step 8.

### 2.10 Amendment, dated 2026-09-15 14:50 local, BEFORE any brain recording exists (the Codex design read refused a
### second time; one readability rule corrected on a constructed fact; two apparatus disclosures) -- no gate's form,
### bar, disposition or N changes

THE CODEX DESIGN READ IS NOT FOLDED. 2.0 named the condition: the read was refused by Codex's usage limit at 11:55 and,
re-sent at 14:41:59 immediately after the T077 closing report's reply (which consumed the reset window's quota), refused
AGAIN at 14:43:10 ("try again at 7:35 PM"; `scratchpad/t078/codex_design_read.md`, 53,577 tokens, no content beyond
"I'll review against the design's stated head"). Per 2.0 the row proceeds: 2.9 steps 5-6 (the dry run, the campaign)
follow this amendment. If a design read lands after this point it is folded as a dated 3.5 item (post-number: it may
name defects, it may not move a bar), never as a pre-registration amendment; the T078 closing report to Codex asks its
five questions and question 6 in the same prompt.

THE POSITIVE CONTROL'S READABILITY RULE, corrected on a CONSTRUCTED fact before any brain run: 2.8 computed
`N_MIN_PER_CELL = 10`'s consequence for the PRIMARY reading only (focus(4) x actions(12), <= 48 observed cells, 480 pairs
at 1,200 ticks). The (g3c) control's axes are actions x actions -- up to 144 observed cells, 1,440 pairs -- and the
builder's committed test (`tests/test_attention_schema.py::test_the_readers_run_on_a_constructed_recording_and_the_
control_axis_can_refuse`, `c955e27`) shows the control REFUSING on a uniform 12-class action stream at every pair count
N = 1,200 can produce. A control that refuses by construction cannot certify the estimator, and a REFUSED G3 would be
the instrument's fault by 2.3's own words. The per-observed-cell minimum protects the SUBTLE primary reading; the
control's job is to detect a STRONG dependence (actions are autocorrelated), and its permutation floor is exact at any
cell count -- sparsity raises the floor, it does not blind it. AMENDED: the control's readability is `n >=
N_MIN_PER_CELL x |Y|` (10 pairs per ACTION CLASS on the next-tick axis; >= 120 pairs at 12 actions), spelled
`mi_read(..., min_rule="per_class")` on the control call ONLY; the primary keeps 2.1's per-cell rule as pre-registered;
R3 gains the twin red (the same 145 pairs REFUSE as the 39-cell primary and READ as a 12-class control). Nothing else
moves: N, n_perm, the bar, (p), (r), (g3c), (g3), (g4a), (g4b), 2.6's branches, 2.9's order. Landed as the commit after
this one, before the dry run.

TWO APPARATUS DISCLOSURES the recorder's build surfaced (code only; no brain was run before this amendment): (i)
`_attend_real_stream` is called up to TWICE per awake tick (vision, then audio -- brain.py:1603-1608), and the brain's
own record then disagrees with itself: `_attn_real` is `va or aa` (the FIRST successful call) while `_attn_focus`,
`LearnedAttention.gate` and `.focus` are last-call state. The recorder takes the LAST call (so the recorded vector
matches what `control_state` reads), records `attend_calls` per tick and `chan` both ways, and the pin COUNTS
`ticks_two_attend_calls` and `ticks_chan_disagrees` per seed -- reported, never adjudicated here. (ii) the `sustain`
field reads `Brain.workspace` (the arm's one workspace, brain.py:1312) through `report_channel.snapshot`; on a scratch
brain it may be a near-constant (1.3's ignition rate), which the decoder's train-half standardization tolerates (a
zero-variance column reads 0) and which G4 reports through `u_t`'s decode rather than hiding.

THE BUILD's own disclosures (the builder's report, `c955e27`; each a 3.5 item at record time, named here so the dry run
is read against them): `mi_read`'s signature is `track_error`'s shape; the decoder's init is an explicit CPU
`torch.Generator` (2.1's "CPU or CUDA identically by seed" is obtained that way, not by a global seed); `mi_read` and
`decode_heldout` return `n_perm` / `k` / `refusal` beside 2.1's keys; R5 carries a vacuity clause 2.4 did not word (a
constant `u_t` would pass content-blindness trivially -- the T077 R2/R5 shape); `REDS_N_PERM = 200` for the reds (the
campaign's floors use 2,000); two private parameters exist only so two mutations can be RUN; `_q95` is a second
spelling of `track_error`'s order statistic (a P2 lift the moment a third caller appears); the `GRADIENT_CLAIM` of the
decoder's autograd fit is declared `NOT_A_RULE` (an offline analysis decoder over a recording; no synapse, never on a
tick) after the gradient-claim gate read the new symbol RED and was answered by the registration, not the gate.


### 2.11 PART II, pre-registered now, built only on G3 PASS and G4 PASS / PASS-WITH-LEAK

2.11.1 THE ACS (adopted; Graziano via the algorithm's A4-A6, scoped to this loop): in `cortex/attention_schema.py`, a
ring of W = 4 frames of `u_t`; ONE reader with its own crc32 child RNG (seed `crc32("t078-acs") + seed`): `W_pred`
predicting the NEXT control state by a three-factor local rule on the prediction error (no backprop; the update is
dw = lr * error * eligibility with a running-mean baseline, the shape `learned_attention.learn` uses); HEAD A: an
endogenous band preference `g_t` (4 values) fed to `LearnedAttention.attend(goal=g_t)` as the top-down goal -- the ONE
wire to brain.py (`_attend_real_stream`'s `goal=None`, guarded: `goal=None` stays the default and the intact loop is
byte-identical with the schema absent). Constraints: content-blind by TYPE (R5); content-blind in INFORMATION (G4 re-run
with the built `u_t`); IMPOVERISHED (the static assert); READ BACK (an input to the reader, never a state() export).
2.11.2 THE REPORT: the reader's predicted next focus band. THE LICENSED CLAIM's measurement: the report at t predicts
which INPUT governs the output at t+1 -- "governs" = the band with the largest gated contribution to `chan` at t+1
(`gate[b] * sal[b]`, the eligibility the gate computes; the per-band neuter is NOT run per tick), scored as held-out
prediction accuracy against the fixed prior (the marginal band frequency) and against the RAW-STATE predictor (the same
reader fed the raw attention state at t with no schema: the train-WITHOUT-it protocol, redundancy-controlled -- same
capacity, same inputs minus the schema's own state). THE DISSOCIATION (C5): the report channel SEVERED from the
attention state -- the reader's input replaced by a cross-tick permutation of `u_t` -- must lose the prediction. The
controls that must reproduce the collapse (9.2): zeros; matched-variance Gaussian; the cross-tick permutation; random
`g_t` from the same distribution. Before any severed contrast is read, the severing must be shown to CHANGE a recorded
downstream quantity (T077 3.5 item 16). The LEGIBILITY experiment (DX2): two readers at identical capacity, one fed the
gated latent projected to 15, one fed `u_t`; both outcomes written now: a TIE means "a model of attention" collapsed
into "an extra informative input" on this loop; a WIN for `u_t` means content-blindness did work; only interpretable
if G4 did not void.
2.11.3 THE THIRD FACTOR (invented, #8): `pi_t` = 1 - min(1, |prediction residual| / its running mean) routed as the
third factor of `LearnedAttention.learn(...)` IN PLACE OF the `lp_mean - threat` scalar (P2: one third factor; the
incumbent scalar becomes the RAW-RESIDUAL arm's factor, not a second live path). RED-1: constant-pin the factor at its
stream mean over the run -> the gate's weight trajectory (`signature()` per tick) and the held-out attention accuracy
below must differ, or the factor is decorative -- run BEFORE any positive claim. RED-2: swap schema-residual for
raw-residual precision (the precision of the first-order prediction error `_pc_surprise`, no schema) -> equal
performance (a TOST with margin = the seed sd of the schema arm's outcome) means adopt plain precision weighting and
DELETE the invention (the register's NULL). RED-3: double ablation under redundancy control -- pin the factor AND sever
the schema; each single ablation's collapse must not be reproduced by the other alone. CONTROL: the capacity-matched
SHAM -- the same reader, the same routing, fed cross-tick-permuted `u_t` -- must NOT reproduce the effect. Three live
arms (schema-precision, raw-residual precision, sham) in ONE command, same seeds. OUTCOME the factor is judged on: the
gate's held-out attention accuracy = the fraction of awake ticks on which the attended band at t+1 is the band whose
salience best predicts the LABEL, that band fixed per seed on the first half of the recording (the informative band),
plus the loop's reward (`lp_mean - threat`) beside it, reported.
2.11.4 SURVIVE (9.1, as a TOST with a margin from the measured seed spread of the intact arm): action agreement with the
intact run and the E-STOP never touched (the reflex path is untouched by construction; asserted).
2.11.5 Statistic: (b1) report above the fixed prior; (b2) above the raw-state predictor; (b3) intact - severed above the
bar with each of the four controls inside the band and t(intact - control) above the bar; (b4) legibility, both outcomes
written; (c1) RED-1 the pinned factor differs (t of the accuracy difference, either sign written); (c2) RED-2 schema vs
raw residual, the contrast in EITHER direction written and the TOST; (c3) RED-3; (c4) the sham does not reproduce; (s)
the TOST on action agreement. Disposition: RED-2 equality -> the third-factor routing DELETED, the ACS (adoption) stays
with its own reading; ACS null ((b1)-(b3) not met) -> the ACS's live wire (Head A) is REMOVED (the loop returns to
`goal=None`, byte-identical), the instrument and the record stay, tasks.md `[X]` NOT MET WITH `surfaces :=
("attention_report",)` DECLARED because 1.5's gate rule requires a symbol for an `[X]` C4 task (the symbol is printed
nowhere; refusal 3 polices it; T077 3.5 item 9 is the precedent this sentence exists to pre-empt); PASS ->
`CONCLUSION["attention_schema"].reading` := the 8-seed mean report accuracy, surfaces `("attention_report",)`, the
cockpit prints the row with the licensed sentence inside its 7-line window, `SURFACE_BASELINE` unchanged or lowered.
Nothing of the incumbents is deleted on any null except the invented third factor on RED-2 (P8 clause 5 targets the
invented method that measures nothing).
