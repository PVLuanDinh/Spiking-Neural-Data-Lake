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

## 3. Results (written 2026-09-15 after the campaign; nothing above this line was edited after 2.10)

### 3.0 The apparatus, measured before the campaign (2.9 step 5; dry run, one seed, 200 ticks, the pin redirected)

`scratchpad/t078/dry_run_module.py` at `677ee16` (the first attempt at `455236d`, 14:48:36, crashed 16 s in on a ragged
latent array -- 3.5 item 1; the re-run 18:29:51): the determinism twin identical over 100 ticks of seed 0; 200 ticks ->
97 awake, 89 consecutive awake pairs (7 straddling a sleep bout dropped, 0 E-STOP), 35 observed (focus, action) cells,
13 distinct actions, 96 G4 rows over 10 classes; EVERY gate reading REFUSED by construction as 2.8 and 2.10 fix it --
the primary "89 pairs over 35 observed joint cells is below 10 per cell (350 required)", the control below 10 per class
(13 classes), G4 "48 held-out rows is below N_TEST_MIN = 50" -- so no gate number was previewed; 42.9 s end to end for
400 brain-ticks plus the reads, i.e. ~0.1 s per brain-tick, which re-projects the campaign (9,600 ticks + the twin +
the reads) at ~17-20 min against 2.8's "~10-12 min" (CORRECTED 2026-09-15 after the panel: the campaign then landed at
644.1 s, INSIDE 2.8's window, so this re-projection was the one that missed, high -- 3.5 item 5; the
refusal rule of 2.8 -- re-size above 3 h -- was not near). Pin shape verified: every per-tick field of 2.1 present,
`gated` absent from the rows and packed to the npz with tick indices (3.5 item 1;
CORRECTED 2026-09-15 after the panel: the npz also packs the STALE latent of every asleep tick, 602-611 per seed --
item 4's appended mechanism). The dry run's npz landed in `tools/`
by the campaign's default path and was deleted before the campaign; its pin stayed in the scratchpad.

### 3.1 The build, red-first, and the commits

`02d677e` the pre-registration (sections 1-2.9, 2.11 + PLAN-FINAL Part F row C-R7; the Codex design read disclosed
absent in 2.0) -> `3485853` `cortex/attention_schema.py` SKELETON with the six functions as stubs returning null
readings, `t078_reds()` all False, the three registrations (REGIONS, run_selfchecks CORE, dormancy EXEMPT) -> `4971a99`
the reds R1-R6 committed RED -- 7 of 7 tests FAILING ON ASSERTIONS against the stubs (R1/R2 `readable is True` ->
`{'readable': False, 'refusal': 'stub'}`; R3's refusal half passed and its readable half failed; R4/R6 `readable is
True`; R5's vacuity clause `control_state(c) != ua` -> two zero vectors equal; the self-check line all six False), not an
ImportError (T077 3.5 item 10's lesson) -> `490a9b8` the instruments GREEN with every red reddened IN FACT by its
mutated scorer (R1 un-permuted floor: the control's margin 1.9787 bits -> 0.0000; R2 self-scored: the i.i.d. action
stream 0.038 <= 0.053 -> 3.5822 above; R3 N_MIN := 0: the 145/39 shape reads 0.1143; R4 train-scored: the band's q95
0.1333 -> 0.3000; R5 chan[0] appended: length 16, the twins differ; R6 operands swapped: VOID -> PASS-WITH-LEAK); the
gradient-claim gate read the decoder's autograd fit RED (MISSING_DECLARATION 1 > baseline 0) and was answered by a
`GRADIENT_CLAIM` declaring it NOT_A_RULE with the gradient stated, not hidden -> `c955e27` the recorder, the two readers,
`t078_determinism`, `t078_projection` and `--t078-campaign`, code only (no brain run) -> `1094234` amendment 2.10 (the
Codex read refused twice; the control's per-class readability; two recorder disclosures) -> `455236d` 2.10's code
(`min_rule="per_class"` on the control call only; R3's twin) -> `677ee16` the latent packer fixed on the dry run's crash
(3.5 item 1) -> the dry run (3.0) -> `9acc730` the pin. Census gates rc=0 after every src commit; the module
self-check green (6/6) from `490a9b8` on; `tests/test_attention_schema.py` 7 RED -> 10 -> 11 green.

### 3.2 The campaign (`tools/t078_attention_schema.json` + `tools/t078_latents.npz`, run at `677ee16`, committed `9acc730`)

Seeds 0-7, N = 1200 ticks, ONE arm, 644.1 s; the determinism twin (p): identical=True over 100 ticks of seed 0. Latents sha256 `f74d85247ccc7d39...`. Reds in-run: 6/6.

| seed | awake | pairs (sleep-dropped / estop) | primary cells | required (10 x cells) / slack | control cells | actions | G4 rows | two-attend ticks | chan disagrees | ignited ticks |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 593 | 544 (48 / 0) | 50 | 500 / +44 (8.8 %) | 130 | 13 | 592 | 592 | 1196 | 1 |
| 1 | 593 | 544 (48 / 0) | 46 | 460 / +84 (18.3 %) | 127 | 13 | 592 | 592 | 1199 | 0 |
| 2 | 598 | 549 (48 / 0) | 47 | 470 / +79 (16.8 %) | 127 | 13 | 597 | 597 | 1199 | 1 |
| 3 | 592 | 543 (48 / 0) | 43 | 430 / +113 (26.3 %) | 143 | 13 | 591 | 591 | 1199 | 1 |
| 4 | 589 | 540 (48 / 0) | 39 | 390 / +150 (38.5 %) | 135 | 13 | 588 | 588 | 1199 | 1 |
| 5 | 590 | 541 (48 / 0) | 35 | 350 / +191 (54.6 %) | 146 | 13 | 589 | 589 | 1191 | 1 |
| 6 | 594 | 545 (48 / 0) | 46 | 460 / +85 (18.5 %) | 144 | 13 | 593 | 593 | 1195 | 1 |
| 7 | 589 | 540 (48 / 0) | 41 | 410 / +130 (31.7 %) | 126 | 13 | 588 | 588 | 1199 | 1 |

The vision classifier's own accuracy, REPORTED and never barred (2.7; appended 2026-09-15 after the panel, 3.5 item
14): 0.9321-0.9331 per seed over the 589-598 perceived frames each (the rows whose `label` is not None),
re-derived offline from the pin's per-tick `digit` and `label`; `run_t078_campaign` does not compute it (`t078_g4`
computes `u_digit`, 2.2's second target, and no classifier accuracy). It adjudicates nothing; its consequence is that
the perceived digit and the frame label are near-interchangeable targets, so 3.3's `u_t -> digit acc` column is close
to a second reading of the (g4a) target. The `primary cells` / `control cells` split and the required / slack column
were added 2026-09-15 after the panel (3.5 items 11 and 14); the other columns are the pin's as first written.

The sleep cadence is clock-driven and seed-independent: 48 bouts in 1,200 ticks on every seed (awake 589-598), so 48
consecutive-awake links straddle a bout and are dropped and counted; no E-STOP pair occurred. `two-attend ticks` equals
the awake ticks that attended on every seed (3.4 i); `chan disagrees` counts ALL 1,200 ticks, awake (the vision-call
channel against the audio-call band means, every awake tick) plus asleep (a stale `_attn_real` against an empty
latent), so it is ~1,195 of 1,200 and is a count, not a reading. APPENDED 2026-09-15 after the panel: it equals the
npz row count on every seed; the agreeing ticks are the pre-first-attend ticks, where the comparison is degenerate
(3.5 item 26). Ignition fired on 1 awake tick per seed (0 on seed 1).

### 3.3 The readings, conjunct by conjunct (every number from the pin; the bar 1.895 over 8 seeds; FPR = the sign-flip rate)

| seed | G3 primary MI | perm median | q95 | control MI | perm median | q95 | gate16 | same-tick MI | perm median | q95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0549 | 0.0577 | 0.0777 | 0.7967 | 0.2120 | 0.2494 | REFUSED (153 cells) | 0.0582 | 0.0570 | 0.0777 |
| 1 | 0.0670 | 0.0556 | 0.0755 | 0.8310 | 0.2113 | 0.2446 | REFUSED (148 cells) | 0.0479 | 0.0556 | 0.0749 |
| 2 | 0.0408 | 0.0558 | 0.0743 | 0.8746 | 0.2093 | 0.2434 | REFUSED (147 cells) | 0.0506 | 0.0554 | 0.0775 |
| 3 | 0.0591 | 0.0553 | 0.0744 | 0.7528 | 0.2168 | 0.2539 | REFUSED (151 cells) | 0.0378 | 0.0558 | 0.0743 |
| 4 | 0.0538 | 0.0529 | 0.0709 | 0.7744 | 0.2115 | 0.2514 | REFUSED (153 cells) | 0.0549 | 0.0535 | 0.0710 |
| 5 | 0.0847 | 0.0511 | 0.0683 | 0.6396 | 0.2166 | 0.2585 | REFUSED (142 cells) | 0.0442 | 0.0512 | 0.0678 |
| 6 | 0.0545 | 0.0573 | 0.0767 | 0.5528 | 0.2135 | 0.2535 | REFUSED (156 cells) | 0.0646 | 0.0575 | 0.0781 |
| 7 | 0.0445 | 0.0515 | 0.0716 | 0.8392 | 0.2067 | 0.2433 | REFUSED (151 cells) | 0.0475 | 0.0524 | 0.0700 |

| seed | u_t acc | band median | band q95 | c15 acc | band median | band q95 | raw64 acc | band q95 | chan acc | band q95 | u_t -> digit acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1858 | 0.1081 | 0.1385 | 0.2601 | 0.1115 | 0.1520 | 0.5236 | 0.1453 | 0.1689 | 0.1453 | 0.1554 |
| 1 | 0.1723 | 0.1149 | 0.1486 | 0.2500 | 0.1149 | 0.1486 | 0.5203 | 0.1453 | 0.1791 | 0.1419 | 0.1588 |
| 2 | 0.1773 | 0.1104 | 0.1405 | 0.2274 | 0.1137 | 0.1472 | 0.5251 | 0.1371 | 0.1773 | 0.1405 | 0.1639 |
| 3 | 0.1385 | 0.1115 | 0.1351 | 0.2669 | 0.1149 | 0.1486 | 0.5372 | 0.1419 | 0.1858 | 0.1453 | 0.1351 |
| 4 | 0.1531 | 0.1190 | 0.1395 | 0.2687 | 0.1156 | 0.1497 | 0.5340 | 0.1463 | 0.1939 | 0.1429 | 0.1361 |
| 5 | 0.1085 | 0.1051 | 0.1424 | 0.1932 | 0.1119 | 0.1458 | 0.4136 | 0.1424 | 0.1458 | 0.1424 | 0.0915 |
| 6 | 0.1818 | 0.1145 | 0.1380 | 0.2593 | 0.1111 | 0.1481 | 0.5152 | 0.1380 | 0.1785 | 0.1414 | 0.1448 |
| 7 | 0.1293 | 0.1122 | 0.1429 | 0.2551 | 0.1122 | 0.1463 | 0.5306 | 0.1395 | 0.1871 | 0.1395 | 0.1327 |

| conjunct | statistic | t | FPR | reads |
|---|---|---|---|---|
| (p) determinism | twin byte-identical | -- | -- | MET |
| (r) readable | G3 primary + control, G4 u + c15 on every seed | -- | -- | MET |
| (g3c) the estimator sees | MI(a_t; a_t+1) - q95 | +12.6296 | 0.0469 | MET |
| (g3) task binding | MI(focus_t; a_t+1) - q95 | -3.0213 | 0.0469 | NOT MET |
| (g4a) u_t decodes content | u_acc - band | +1.4903 | 0.0430 | NOT MET |
| (g4b) content decodes better | c15_acc - u_acc | +9.2116 | 0.0508 | MET |
| reported gate16 | REFUSED on 8/8 seeds: 540-549 pairs over 142-156 observed (gate16, action) cells under the per-cell rule | -- | -- | refused, NOT a number: the pin's `reading.reported.gate16` (t 0.0000, `degenerate`) is eight refusals coerced to 0.0 (3.5 item 16) |
| reported same_tick | | -9.1089 | 0.0508 | not above the bar (never barred) |
| reported raw64_ceiling | | +29.7204 | 0.0352 | above the bar (never barred) |
| reported chan | | +6.5118 | 0.0586 | above the bar (never barred) |

**PART I VERDICT: G3 STOP x G4 PASS** (leak +0.0151 accuracy points, mean over seeds of u_acc - band).

### 3.4 The disposition applied (2.6, the STOP branch at G3; G4 PASS recorded beside it)

(p) MET, (r) MET on every seed, (g3c) MET: the estimator SEES -- the action stream's own one-tick autocorrelation reads
MI 0.5528-0.8746 bits against floors 0.2433-0.2585 on every seed, t +12.6296 (FPR 0.0469); the estimator is not blind on
this loop. (g3) NOT MET: the focus band at t carries MI 0.0408-0.0847 bits about the action committed at t+1 against
floors 0.0683-0.0777 -- BELOW its own floor on 7 of 8 seeds (seed 5 above: 0.0847 against 0.0683), t -3.0213 (FPR
0.0469) -> G3 STOP. By 2.6: NO ACS is built, NO third factor, NO wire to brain.py (`git diff 02d677e..HEAD --
src/snn_data_lake/cortex/brain.py` is empty); the row stays `[ ]` with its dated note; no surface symbol is declared
and none is needed; `CONCLUSION["attention_schema"].reading` stays None; the C5 sentence is not written; the incumbents
(`learned_attention`, the loop, the workspace) are untouched; Part II (2.11) is NOT built. A STOP is the algorithm's own
terminal state (SYNTHESIS :65): on this loop, over 8 seeds, the recorded FOCUS BAND one tick earlier -- identically
argmax(gate), a four-way summary of the four-band gate (3.4 iii) -- does not predict the committed action above chance.
SCOPE, appended 2026-09-15 after the panel (3.5 item 15): the barred X in `t078_g3` is the recorded `focus_band` (the
audio attend call's, 3.4 i; four bands plus the NONE head of the first awake tick) against `y_next`, the committed
action; the only state-wide reading, `gate16`, returned `readable=False` on every seed under the per-cell rule; and the
15-field `u_t` was never read against the action on this row (`t078_g4`'s decodes target the frame label and, for
`u_digit`, the perceived digit) -- so the row says nothing about the rest of `U_FIELDS`, and 2.6's noun "the attention
state" and 2.7's "attention control state" are wider than 2.2's pre-registered X.

G4 PASS, read beside it and at the bar: (g4a) the 15-d `u_t` decodes the perceived frame's LABEL above its own
200-shuffle band on 6 of 8 seeds (0.1858 / 0.1723 / 0.1773 / 0.1385 / 0.1531 / 0.1085 / 0.1818 / 0.1293 against bands
0.1351-0.1486; chance 0.1 over 10 classes; seeds 5 and 7 below; seed 3's margin is one held-out prediction, 41 of 296
against a band of 40 of 296 -- "5 of 8" was a transcription slip, 3.5 item 12), t +1.4903 (FPR 0.0430) -- NOT MET at
1.895, so the vector is content-blind in information AT THE BAR, not comfortably; the leak's mean is +0.0151 accuracy
points (a q95 margin; +0.0439 against the band's centre, 3.5 item 8) and is written as 2.3 requires.
(g4b) the matched 15-d content decode reads 0.1932-0.2687 and beats `u_t` by t +9.2116 (FPR 0.0508) MET; the raw 64-d
ceiling reads 0.4136-0.5372 (t +29.72 over `u_t`) and the 4-d arm channel 0.1458-0.1939 (above its band, t +6.51) --
the content ground truth exists on this loop and the schema vector carries little of it. Had (g4a) crossed the bar
the reading would have been PASS-WITH-LEAK by 2.3's rule; it did not, and the size of the leak is the same number either
way.

Two facts the disposition did not need but the record must carry (both disclosed in 2.10 before the run; their
MAGNITUDE is the campaign's): (i) EVERY awake tick attends TWICE -- `_attend_real_stream` for the vision percept, then
for the audio percept (brain.py:1603-1608) -- on 592 / 592 / 597 / 591 / 588 / 589 / 593 / 588 of the awake ticks; the
recorded gate, focus and `u_t` are the SECOND (audio) call's state, which is also what the brain's own `_attn_focus`,
`LearnedAttention.gate` and `.focus` hold after the tick, while the arm's sensory channel `chan` (`_attn_real = va or
aa`) is the FIRST (vision) call's. The last-call focus band equals the vision channel's argmax band on only 27-153 of
~540 pairs per seed. So (g3) as pre-registered and read is the AUDIO attend's focus band against the next action.
POST-HOC, REPORTED, NEVER BARRED AND NEVER PRE-REGISTERED (`scratchpad/t078/record_gen` companion probe, from the pin):
the vision channel's argmax band as X reads MI 0.0429-0.1252 against floors 0.0637-0.0714, above on seeds 3 and 5 only,
t -0.5262 (FPR 0.0391) -- the STOP does not hinge on which call was recorded; but the vision call's own LIF focus band
was NOT recorded, though it was LIVE on the brain when each row was written (`_attend_real_stream` returns
`{channel, band, focus}`, `_safe_ctx` keeps the vision dict as `_attn_real` -- `va or aa` -- and `Brain.state()`
publishes it as `attention_real`): the recorder omitted an available field, it did not lack the quantity (CORRECTED
2026-09-15 after the panel, 3.5 item 19); a successor records BOTH calls' focus per call and pre-registers which one
the action path consumes (opening 2). (ii) CORRECTED 2026-09-15 after the panel (3.5 item 9; the first wording called
`sustain` near-constant on a misread of the fold's own probe): TWO of `u_t`'s fifteen fields are near-constant on the
scratch brain at this width -- `ignited` is True on exactly 1 awake tick per seed (0 on seed 1; pin key
`per_seed.<s>.ticks_ignited`; 1.3 measured 1 of 146 before the run) and `focus_is_none` (`u[10]`) is 1 on exactly one
awake tick per seed, the first, which carries no latent and so is identically 0 on every G4 decode row (a
zero-variance decode column, the case 2.10 (ii) anticipated for a different field); ONE is low-cardinality -- the
focus STRENGTH takes three levels (0.0 / 0.05 / 0.10) on seven seeds and five on seed 5, is 0.0 on 52-56 % of awake
ticks (25 % on seed 5), mean 0.03-0.06 (1.3 had measured it "nearly degenerate"); and `sustain` is NOT near-constant:
it is the workspace snapshot's `strength` (`report_channel.snapshot`, a graded hold), taking 586-596 distinct values
over the 589-598 awake ticks per seed, spanning 0.5595-0.7500, mean 0.613-0.633, sd 0.030-0.039 (among the three
narrowest sds of the fifteen fields, yet graded), and equal to 0.65 on exactly ONE awake tick per seed -- the first,
the hold's initial value -- so 2.10 (ii)'s hedged "may be a near-constant" is NOT borne out. The widest-sd fields are
`pc_surprise` (567-586 distinct values per seed, sd 0.76-1.08) and `drive_margin` (423-511, sd 0.22-0.46); the four
`gate` fields carry 585-598 distinct values each (`per_seed.<s>.rows[].sustain / .pc_surprise / .drive_margin / .u`).
The reported gate16 reading REFUSED on every seed under
the per-cell rule (16 x 13 = up to 208 cells, 2,080 pairs required) and is recorded as refused, not as a number (the
degenerate t 0.0000 in the pin's `reported.gate16` is eight zero diffs from the refusals); the same-tick MI(focus_t;
action_t) reads BELOW its floor on every seed (t -9.1089) -- the recorded focus does not even co-occur with the action
above chance, which is the reading the STOP is consistent with. (iii) APPENDED 2026-09-15 after the panel: the
recorded focus band is identically argmax(gate) on every attending awake tick of the pin (0 exceptions in 4,730), and
this is structural, not empirical -- `LearnedAttention.attend` takes `np.argmax(rates + 1e-9 * gate)` while
`LearnedAttention._lif_rate(steps=20)` quantizes each rate to a multiple of 0.05 (seven distinct values over the whole
gate range, 0.0 to 0.30), so the 1e-9 tie-break can never cross a quantum and the monotone rate always ranks with the
gate; the LIF race selects no band the gate did not already select, and the rate survives only as `u_t`'s
`focus_strength`. So (g3) reads argmax(gate)_t against action_t+1, and the refused gate16 code is the ONLY reading of
the gate beyond its argmax this record could have carried (opening 6). (iv) APPENDED 2026-09-15 after the panel,
REPORTED and never barred -- the DETECTION FLOOR of the (g3) contrast, measured post-hoc on the pinned pairs
(`scratchpad/t078-panel/fold_detection_floor.py`): a deterministic band-to-action map is planted on a fraction q of
each seed's pairs (the action at t+1 replaced by map(focus_t)) and (g3) is read through the shipped `t078_pairs` /
`mi_read` / `contrast` at N_PERM and the bar; positive control: q = 0 reproduces the pin's t -3.0213 exactly. Over a
grid q = 0.02 / 0.04 / 0.06 / 0.08 / 0.10 / 0.12 / ..., all FOUR maps (four distinct actions; the four most frequent;
two actions by band parity; a permuted own-index map) first fire at q = 0.10 and none fires at 0.08 -- t +3.235 /
+3.080 / +2.335 / +3.663 at the firing point, mean MI margin above the floor +0.0248 / +0.0155 / +0.0148 / +0.0211
bits. So at these pair counts and this bar the pre-registered contrast resolves a planted dependence that rewrites
about one action in ten, an MI margin of roughly 0.015-0.025 bits; the primary's pinned mean margin is -0.0163 bits.
A real dependence weaker than that floor would have read as this STOP (3.5 item 21, opening 8).

### 3.6 What this row does NOT claim, and the openings it leaves

No experiential claim; the C5 refused sentence stands and no licensed sentence exists. The reading is about ONE loop:
the scratch brain's own attention control state (the audio call's, 3.4 i) and its committed actions at the deployed
width, 1,200 ticks, 8 seeds; nothing about the daemon's brain, chat, reasoning or the arm's hardware. A STOP is NOT "the
brain does not attend" and NOT the algorithm's G3 (which binds a held GOAL to a FIRST action across a gap): it says the
recorded FOCUS BAND (identically argmax(gate); 3.4 iii) one tick earlier carries no information about the next
committed action above the shuffle floor on this loop -- nothing is said of the rest of `U_FIELDS` (the state-wide
gate16 reading refused; `u_t` was never read against the action). G4 PASS is NOT "content-blind": it is "not decodable
above chance AT THE BAR over 8 seeds", with a measured leak on 6 of 8 seeds (+0.0439 against the band's centre) -- and
at the pre-registered decoder budget only (3.5 item 13).
OPENINGS, each named for a successor row: (1) the brain-side stimulus port (docs/
PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md 5.2, "the largest and riskiest single item") -- G3 as written needs a held goal
and a gap, and this loop has neither; the successor that builds the port re-runs G3 as the algorithm words it; (2) BOTH
attend calls' focus recorded, with the pre-registered X being the call whose channel the action path consumes (3.4 i);
the direct test of "governs" is the per-band neuter of the vision channel (4 replays per tick), not MI; (3) two
near-constant `u_t` fields at this width (`ignited`, `focus_is_none`) and one low-cardinality field (`focus_strength`)
(3.4 ii, corrected) -- the workspace ignites on ~1 of ~590 awake ticks on a scratch brain, the opposite rail from G1's
saturation; the rest of the control state DOES move (`sustain` and the four gates take a distinct value on nearly every
awake tick), so what a successor owes is the IGNITION rail at a width where the workspace fires; (4)
WHICH field leaks the label on the 6 seeds where `u_t` decodes above its band -- a per-field neuter (drop one of the
fifteen, re-decode) for the successor, not this row's to adjudicate; (5) `_q95` is a second spelling of `track_error`'s
order statistic (P2: lift beside `paired_contrast` when a third caller lands); (6) the gate16 reported reading needs a
readability rule of its own or fewer cells (a per-band median split over 4 bands x 13 actions refuses at N = 1,200) --
and because the focus band is identically argmax(gate) (3.4 iii) it was the ONLY reading of the gate beyond its argmax
this record could have carried, so a successor owes a coarser code or a longer recording before any statement about
"the attention state" as a whole is licensed, and pre-registers 2.6's disposition noun and 2.2's X as the SAME object
(3.5 item 15); (7) pre-register |X| = B + 1 = 5 explicitly (the algorithm's own alphabet: the NONE head of the first
awake tick) rather than deleting the non-attending tick, and buy the primary real headroom by raising N or coarsening Y
-- the cell count is set by the 13 observed actions; dropping the one None pair-head moves seed 0's headroom only from
8.8 % to 10.8 % and leaves (g3) at t -3.1709, still a STOP, so the drop is not the remedy, and a successor that drops
it COUNTS it beside `dropped_sleep` (3.5 item 11); (8) the DETECTION FLOOR of the G3 contrast (3.4 iv): P8 clause 2's
external positive control is MET in the form the policy words it -- `t078_g3` builds (g3c) from `t078_pairs` on the
real pinned rows, the recording-and-pairing path T077's fold found untested there -- but the contrast's minimum
resolvable margin was never pre-registered and (g3c)'s margin sits an order above the primary's whole scale; the
successor behind the stimulus port PRE-REGISTERS a plant map and the margin its contrast must resolve BEFORE the run;
(9) the eight arms share ONE stimulus stream (3.5 item 20): a successor varies the frame ORDER or the pack OFFSET per
seed -- the only change that supplies the missing variance component -- and re-reads (g4a) and (g4b) on the varied
stream; (10) the decoder budget (3.5 item 13): a successor pre-registers DECODER_STEPS from a convergence read on the
constructed stream, not from T077's constant, and reads (g4a) at two budgets so the null half of G4 is budget-robust
or is not claimed; (11) the determinism key (3.5 item 25): a successor pre-registers the twin's key over EVERY
recorded field before its run, so "(p) MET" covers the rows and not five of their eighteen fields; (12) one
strictness for the latent field across the packer and the reader, and a reader that refuses loudly on an empty pack
(3.5 item 26).

### 3.5 Deviations from 2.1-2.11, each with its number (dated 2026-09-15)

1. THE FIRST DRY RUN CRASHED, AND THE WAITER HID IT FOR THREE AND A HALF HOURS (obs #2181). At 14:48:36 the 200-tick dry
   run passed the determinism twin and died 16 s in at the campaign's latent packing: `np.asarray([r["gated"] ...])`
   over ALL rows is ragged because asleep ticks (and awake ticks on which `attend` is not called) record an empty
   `gated`; the builder's constructed-recording test had given every row a latent. The traceback was in the .err file;
   the background waiter grepped only the success line, timed out silently, and the session idled until the owner
   asked at 18:27. Fixed at `677ee16`: `t078_pack_latents` keeps the rows that carry a latent as (m, 64) float32 with
   their tick indices beside (`seed{s}_ticks` in the npz), its red added, the constructed test given the real shape;
   re-run 18:29:51, green (3.0). No brain number was seen by the crashed run beyond the twin's byte-identity.
   CORRECTED 2026-09-15 after the panel (the mechanism): asleep ticks do NOT record an empty `gated` -- the recorder's
   capture dict persists across ticks, so an asleep tick carries the STALE latent of the last attending tick; the empty
   rows are the ticks BEFORE the first `attend` call, and the ragged array came from those. The pinned npz therefore
   packs every tick with a non-empty capture, 602-611 asleep ticks per seed included (`seed<s>_ticks` joined to
   `per_seed.<s>.rows[].awake`), and `t078_g4` selects awake rows with a label (which always attended), so no G4
   number is touched; the packer's docstring and a reader-side filter now carry the correct mechanism (the code half
   of the fold). The crash artifacts `dry_run_module_crash1.log` / `.err` are that run's two output files copied aside
   before the re-run; a scratchpad mtime is the copy's time, not the crash's.
2. THE (g3c) CONTROL's READABILITY RULE was amended BEFORE any brain run (2.10, `1094234`; code `455236d`) on the
   builder's constructed fact: the 12 x 12 action joint refused under the per-cell rule at every pair count N = 1,200
   produces. The primary keeps the per-cell rule. Had the rule not been amended the campaign would have REFUSED at (r)
   on every seed (the pin's per-seed control lines carry 13 distinct actions and 35-50 observed (focus, action) cells).
   CORRECTED 2026-09-15 after the panel (3.5 item 14): that parenthetical cited the PRIMARY's joint, which is READABLE
   under the per-cell rule on every seed (35-50 cells against 540-549 pairs); the counterfactual is true of the
   CONTROL's action x action joint -- `per_seed.<s>.g3.control.cells` 126-146 observed cells against `g3.n_pairs`
   540-549, so the per-cell rule needs 1,260-1,460 pairs and refuses on all 8 seeds, while the per-class rule needs 130
   (13 actions, one above 2.10's constructed 12). Also: 2.10's "Nothing else moves: N, n_perm, the bar, (p), (r),
   ..." and its heading's "no gate's form, bar, disposition or N changes" are FALSE AS WORDED of (r) -- 2.3's (r)
   states the per-observed-cell minimum for the positive-control reading as well as the primary, so the clause 2.10
   amended IS (r)'s; what the amendment gates is whether any verdict is read at all (under the un-amended rule
   `run_t078_campaign`'s (r) block sets both `g3` and `g4` to REFUSED), while (g3)'s own contrast is unchanged by the
   readability rule. The one surface still stating the superseded rule, PLAN-FINAL's C-R7 row, carries a dated
   appended sentence.
3. THE BUILD's seven disclosures (2.10's last paragraph, `c955e27`): `mi_read`'s signature is `track_error`'s shape;
   the decoder's init is an explicit CPU `torch.Generator` (2.1's "identically by seed" is obtained that way); extra
   keys `n_perm` / `k` / `refusal` reported beside 2.1's; R5 carries a vacuity clause 2.4 did not word (a constant
   `u_t` passes content-blindness trivially, the T077 R2/R5 shape); `REDS_N_PERM = 200` for the reds, 2,000 for the
   campaign; two private parameters exist only so two mutations can be RUN; `_q95` is a second spelling of
   `track_error`'s order statistic (a P2 lift when a third caller appears).
4. THE DRY RUN's npz landed in `tools/` by the campaign's default path (only the pin was redirected) and was deleted
   before the campaign; the dry run's own pin stayed in the scratchpad (3.0). The campaign's npz is the one committed.
5. THE COST PROJECTION: 2.8 projected ~10-12 min sequential; the dry run re-projected ~17-20 min at ~0.1 s per
   brain-tick (3.0); the campaign took 644.1 s (3.2). 2.8's refusal rule (re-size above 3 h) was not near. APPENDED
   2026-09-15 after the panel: the campaign's `elapsed_s` (644.1 s = 10.7 min) is INSIDE 2.8's projected window, so
   2.8's projection held and the dry run's per-brain-tick re-projection was the one that missed, high; the 1.9x miss
   2.8 cites from T077 3.5 item 20 did not repeat on this row. No cause is named (unmeasured).
6. THE CODEX DESIGN READ never arrived (2.0, 2.10): refused at 11:55 and at 14:43 by Codex's usage limit; its five
   questions and question 6 go to the T078 closing report. No point of a design read shaped this file after the
   freeze; 2.10 changed one readability rule on a constructed fact and folded two apparatus disclosures. APPENDED
   2026-09-15 after the panel: Codex's full output across both codex blocks of `scratchpad/t078/codex_design_read.md`,
   verbatim: "I’ll apply the graphify and Karpathy review rules, check the design against the permitted sources, and
   keep the refutation within 400 words. I’ll treat your ban on touching `brainstem/` as a read ban and use only the
   excerpt you supplied from that path." and "The current checkout is newer than the design and already contains T078
   implementation entries. I’ll review against the design’s stated head, `ac7b04b`, to avoid judging it using later
   results." -- 2.10 quoted only a fragment of the last sentence, so its "no content beyond" reads "no REVIEW
   content". Two facts, no adjudication: the re-send left at 14:41:59 and returned at 14:43:10
   (`scratchpad/t078/codex_design_send.log`), after `490a9b8` (the instruments green, 12:38:02) and `c955e27` (the
   recorder, 12:47:28); the head Codex named, `ac7b04b`, predates the pre-registration commit `02d677e`.
7. THE 2.2 PAIRING DROPPED exactly 48 pairs on every seed as straddling a sleep bout (about 8 % of consecutive awake
   links, the sleep cadence the T077 lag amendment measured) and 0 E-STOP pairs; both counted in the pin.
8. (appended 2026-09-15, after the panel) THE BARRED (g4a) CONTRAST IS REFERENCED TO THE BAND's 95th PERCENTILE, SO
   ITS NO-SIGNAL EXPECTATION IS BELOW ZERO. On this pin the offset alone, `shuffle_median - shuffle_q95`, reads t
   -14.2145 (mean -0.0287 accuracy points); the same accuracies against the band CENTRE read t +4.5724, positive on 8
   of 8 seeds, and `reading.leak_points` (+0.0151, a q95 margin) is +0.0439 against the centre. The same offset on the
   G3 primary reads t -51.3567 while `(mi - perm_median)` reads t +0.5318, under the bar, so the STOP is unaffected.
   2.3's parenthetical "decodes nothing above chance" and 3.4's "content-blind in information" are SUPERSEDED as
   glosses by this reading -- superseded here, never deleted upstream; the bar, (g4a), (g4b), `g4_rule` and the
   verdict stay as pre-registered; 3.3's tables now carry the pinned `perm_median` / `shuffle_median` beside every q95
   (`scratchpad/t078-panel/fold_verify_pin.py`).
9. (appended 2026-09-15, after the panel) 3.4 (ii)'s `sustain` SENTENCE PROMOTED 2.10 (ii)'s PRE-RUN POSSIBILITY TO A
   MEASURED FACT, AND IT WAS FALSE: the fold's own probe had printed the first four entries of a Counter and the
   writer read "0.65: 12" as the mode; the pin reads 586-596 distinct values per seed (3.4 ii, corrected in place).
   Two adjacent ranges were min/max over two seeds while the text said "per seed" (`pc_surprise` 567-577 -> 567-586;
   `drive_margin` 494-511 -> 423-511). The near-constant fields are `ignited` and `focus_is_none` (1.3 had
   pre-measured the first); `focus_strength` is low-cardinality (1.3: "nearly degenerate"). No gate, bar, conjunct, N
   or disposition moves; 3.3's readings and the verdict are untouched. Two refuters disagreed on the count ("one"
   near-constant vs "three"); the pin decided it (obs #2180).
10. (appended 2026-09-15, after the panel) THE TWO MONTE-CARLO SENSITIVITIES 2.8 PROMISED ("computed on the
   constructed R1 stream in 3.0"; "computed on the constructed R4 stream in 3.0") WERE NOT COMPUTED THERE, and no
   earlier item disclosed it (section 1 item 7's binding, "every declared constant's sensitivity computed here, not
   later"). Computed now, from the pin and 2.8's own streams (`scratchpad/t078-panel/fold_sensitivity.py`; positive
   controls: the pinned floors and bands re-derive bit-for-bit on all 8 seeds): (a) n_perm -- on 2.8's constructed
   stream, 24 independent 2,000-permutation floors give `perm_q95` mean 0.0580 bits, sd 0.00059, so 2.8's declared
   "~0.002 bits" overstated the sd three-fold; on the campaign's own primary, 8 independent floor redraws per seed
   move (g3) t across -3.0396..-2.9005 (sd 0.0425; pinned -3.0213) and (g3c) across +12.6277..+12.7690 (pinned
   +12.6296) -- neither verdict is within a floor redraw of the bar; (b) n_shuffle -- with the decoder's own init HELD
   and only the 200-shuffle RNG varied, 12 band redraws move (g4a) t across +1.4456..+1.6520 (sd 0.0645; pinned
   +1.4903), 0 of 12 above 1.895 -- (g4a) NOT MET is robust to the band's Monte-Carlo noise at the pre-registered
   budget (item 13 records what it is NOT robust to). Both numbers are what 3.0 owed; neither moves a reading.
11. (appended 2026-09-15, after the panel) N_MIN_PER_CELL's REALIZED HEADROOM ON THE PRIMARY, per seed (pairs /
   observed cells / required / slack): 0: 544/50/500/+44 (8.8 %); 1: 544/46/460/+84 (18.3 %); 2: 549/47/470/+79 (16.8
   %); 3: 543/43/430/+113 (26.3 %); 4: 540/39/390/+150 (38.5 %); 5: 541/35/350/+191 (54.6 %); 6: 545/46/460/+85 (18.5
   %); 7: 540/41/410/+130 (31.7 %). The sensitivity band: every seed readable at N_MIN_PER_CELL <= 10, seed 0 refuses
   at 11, no seed readable at 16 or above; `run_t078_campaign` reads readability with `all(...)` over seeds and 2.3
   (r) routes a single unreadable seed to 2.6's REFUSED branch, so the STOP terminal state carried 44 pairs of
   headroom on one seed. 2.8's COMPUTED CEILING WAS MISSED ON BOTH FACTORS: it projected "~580 pairs over <= 48 cells"
   (focus(4) x actions(12)); realized: 13 distinct actions on every seed and FIVE focus classes (a 65-cell ceiling),
   seed 0's 50 observed cells above 48, and 540-549 pairs after the 48 sleep-straddle drops 2.2 itself pre-registers
   (item 7 is the cause). THE G3 X ALPHABET IS 5, NOT 2.2's "4 classes": the pair-head carries `focus_band` None on
   exactly one pair per seed -- the first awake tick, one tick before the first `attend` call -- which the algorithm's
   section 4 already pre-registers as the control-state encoding's own fifth class, `onehot(focus|NONE)(B+1)`; the
   deviation is 2.2's G3-axis wording. At these pair counts a seed with 55 or more observed cells would have REFUSED
   the campaign (2.6's REFUSED branch, not its STOP). A successor sizes N from the observed cell count rather than
   from 2.8's projection (opening 7).
12. (appended 2026-09-15, after the panel) "5 OF 8 SEEDS" WAS 6 OF 8 -- a transcription slip against the pin's
   `reading.g4a.diffs` (six positive entries); five is the count of seeds whose accuracy clears the TOP of the printed
   band range, 0.1486, rather than each seed's own band, which is where the slip came from. Corrected in place in 3.4
   and 3.6 (the (g4a) sentence, the leak sentence, opening 4). It moves no reading: 2.3 decides (g4a) on the paired t
   against the bar and pre-registers "no majority-of-seeds rule", so the verdict and the leak's mean are unchanged.
13. (appended 2026-09-15, after the panel) DECODER_STEPS = 200 IS FAR FROM CONVERGED, and 2.8 listed the decoder's
   three constants under "the consequence of each, computed here" while computing none of them -- it substituted the
   raw-64 ceiling decode, which runs through the same `_fit_and_score` at the same budget as the arms it is meant to
   diagnose (so it is partly the method, P8 clause 2, and cannot separate capacity from budget). Re-derived from the
   pin (`scratchpad/t078-panel/redecode_2000.py`): at the pre-registered 200 steps `decode_heldout` reproduces the pin
   EXACTLY on all 8 seeds (u acc and band, c15 acc) and `contrast` reproduces (g4a) t +1.4903 FPR 0.0430, (g4b) t
   +9.2116, leak +0.0151; at 2,000 steps (ten times the budget; 260 s) the SAME rows read (g4a) t +2.7125 (FPR 0.0586)
   MET, (g4b) t +16.2144 MET, `g4_rule` PASS-WITH-LEAK, leak +0.0189, `u_t` above its band on 7 of 8 seeds (u acc
   0.1622 / 0.1824 / 0.1806 / 0.1486 / 0.1497 / 0.1220 / 0.1616 / 0.1429), and the raw-64 ceiling rises from
   0.4136-0.5372 to 0.7075-0.7407 on seven seeds (0.4542 on seed 5) -- so the ceiling's "capacity" reading at 200
   steps was a budget reading. The pre-registered reading at the pre-registered budget REMAINS the record, and (g4a)
   NOT MET -- the null half of G4 PASS -- is NOT budget-robust: 3.4's "content-blind in information AT THE BAR" is
   conditioned on DECODER_STEPS = 200 and the leak +0.0151 is a 200-step reading (opening 10). Nothing is moved; the
   constants-table comment in `attention_schema.py` no longer asserts the ceiling shows capacity-vs-information (the
   code half).
14. (appended 2026-09-15, after the panel) 3.5 ITEM 2's COUNTERFACTUAL CITED THE PRIMARY's CELLS; the control's own
   cells (126-146) appeared nowhere in section 3 -- corrected by the appended sentence under item 2, and 3.2's table
   now carries `primary cells` and `control cells` so the record has no unqualified "cells"; the vision classifier's
   accuracy line under that table is the same panel's reported addition.
15. (appended 2026-09-15, after the panel) 2.6's DISPOSITION NOUN ("the attention state") AND 2.7's "attention control
   state" ARE WIDER THAN 2.2's PRE-REGISTERED X (the focus band); the two were pre-registered in one document without
   being tied together, and 3.4's and 3.6's verdict sentences repeated the wide noun. Corrected in place (the
   sentences now name the FOCUS BAND, identically argmax(gate)); 2.6 and 2.7 stay as frozen; the successor
   pre-registers the noun and the X as the same object (opening 6). The same scope clause is appended to tasks.md's
   T078 note and PLAN-FINAL's C4 note.
16. (appended 2026-09-15, after the panel) THE PIN's `reading.reported.gate16` IS A CONTRAST FABRICATED FROM EIGHT
   REFUSALS: `run_t078_campaign`'s reported block wrote `(mi or 0.0) - (perm_q95 or 0.0)` and `contrast` read eight
   0.0 diffs as a degenerate t 0.0000 / FPR 0.0 (`degenerate: true` in the pin) -- a number where the reading was
   REFUSED on every seed (142-156 observed (gate16, action) cells under the per-cell rule). 3.3's conjunct table now
   says REFUSED; the code half of the fold makes a refused reported reading carry its refusal and no t; the pin
   predates that fix and still carries the manufactured key, which this item is the reading of. No barred conjunct is
   affected.
17. (appended 2026-09-15, after the panel) THREE STAMPS IN THE FROZEN TEXT ARE FORECASTS, NOT OBSERVATIONS: the
   header's window end ("12:05-14:50") was written before 14:50 existed; the `### 2.0` heading's past-tense "re-sent
   after the 14:33 reset" while its body words the act as armed; the `### 2.10` heading's "14:50" against its commit
   at 14:45:41. Git is the operative record, as the header itself delegates: sections 1-2 froze at `02d677e`
   (12:06:25), 2.10 at `1094234` (14:45:41), its code at `455236d` (14:48:07). No frozen line is edited; the header
   line is byte-identical at `02d677e`.
18. (appended 2026-09-15, after the panel) THE DRY RUN's WIDTH: 2.9 step 5 pre-registers one seed at 300 ticks; it was
   run at 200 (3.0; `scratchpad/t078/dry_run_module.py`, whose docstring carries the reason). 2.9's parenthetical
   justified 300 for G3 only ("its G3 reading REFUSES by N_MIN"); at 200 G4 refuses as well ("48 held-out rows is
   below N_TEST_MIN = 50"), while at 300 the 146 awake ticks 1.3 measured would have cleared `N_TEST_MIN` and a G4
   decode number WOULD have been previewed. The narrowing removed a preview the pre-registered width allowed; at
   neither width was a G3 number previewed. Apparatus only; no gate reading in 3.3 is touched.
19. (appended 2026-09-15, after the panel) THE VISION CALL's FOCUS BAND WAS LIVE ON THE BRAIN AND THE RECORDER OMITTED
   IT (3.4 i, corrected): `_attend_real_stream` returns `{"channel", "band", "focus"}`, `_safe_ctx` keeps the vision
   dict as `_attn_real` (`va or aa`) on every tick that attends twice, and `Brain.state()` publishes it as
   `attention_real`. The hole is the recorder's, not the loop's; the argmax(chan) substitute stood in for something
   the tick held. Opening (2) is narrowed to what a recorder can retire: a successor's `attend` wrapper APPENDS each
   call's `LearnedAttention.focus` to a per-tick list beside `attend_calls` (call 1 vision, call 2 audio; a tick with
   `attend_calls != 2` visibly unresolvable), reading the per-call list rather than `_attn_real["band"]`; the
   pre-registration of WHICH call is X and the per-band neuter of the vision channel are still owed by that successor,
   not supplied by any recording. Not applied to this closed row.
20. (appended 2026-09-15, after the panel) THE EIGHT ARMS SHARE ONE STIMULUS STREAM: the G4 decode target is one
   ordered stream shared by all eight arms -- each seed's label sequence is an exact PREFIX of the longest seed's, the
   shared prefix byte-identical (verified on the first 580 labels of every seed). `_real_frames` / `_real_labels` load
   once from the fixed pack, `Brain.__init__` sets `_real_ptr = 0`, `_perceive_real` walks it by `_real_ptr % len`,
   and `t078_record_brain` seeds only `make_brain(seed=...)`; so 2.2's "seeds 0-7" varies the brain's RNG at the INPUT
   only, and `contrast`'s eight arm-level diffs are not independent in the stimulus factor. Everything downstream of
   the RNG does vary: the awake and G4 row counts (3.2), the gate, the gated latents (the npz's per-seed arrays differ
   on the aligned prefix) and the committed actions. DIRECTION: sharing the stream REMOVES a variance component across
   arms, so a design that also varied it can only ADD variance and lower |t|; the nominal 7 d.f. is on the generous
   side for the same reason. This route cannot carry (g4a) ACROSS the bar and does not reach (g3), (g3c) or (g4b); THE
   VERDICT DOES NOT MOVE -- a scope disclosure (opening 9).
21. (appended 2026-09-15, after the panel) THE DETECTION FLOOR OF THE G3 CONTRAST WAS NEVER MEASURED BEFORE THE RUN:
   the only positive control, (g3c), reads MI 0.55-0.87 bits against floors ~0.25 -- an order of magnitude above the
   primary's whole scale (0.04-0.08) in a sparser joint -- so it certifies that the estimator sees a STRONG dependence
   and says nothing about the margin the contrast resolves at the primary's scale. Measured post-hoc, reported and
   never barred (3.4 iv): on a grid q = 0.02 ... 0.30, all four planted maps first fire at q = 0.10 and none at 0.08
   (t +3.235 / +3.080 / +2.335 / +3.663; mean MI margin +0.0248 / +0.0155 / +0.0148 / +0.0211 bits above the floor);
   the primary's pinned mean margin is -0.0163 bits. The firing FRACTION is the same across the four maps while the MI
   margin resolved at it varies with the map, so the reportable number is the margin in bits the pre-registered
   contrast resolves at 540-549 pairs: roughly 0.015-0.025 (the refuter's own probe reported a firing fraction near
   0.11 that swung two-fold across maps; at this grid, 0.02 steps, the four maps agree at 0.10, so any swing sits
   inside one grid step -- NOT reproduced here, and not the reportable number either way). 2.10 disclosed the regime
   difference before the run ("the per-observed-cell minimum protects the SUBTLE primary reading; the control's job is
   to detect a STRONG dependence ... sparsity raises the floor, it does not blind it"); what it did not do is state
   the floor. The successor pre-registers the plant map and the margin BEFORE its run (opening 8); the STOP stands as
   read.
22. (appended 2026-09-15, after the panel) THREE POST-NUMBER INSTRUMENT CHANGES LANDED IN THE CODE HALF, each touching
   no barred conjunct and each verified by the fold's head check re-deriving every pinned MI, floor, decode, band and
   contrast bit-for-bit (`scratchpad/t078-panel/verify_head.py`, 35 s): (a) `c3ebace` (K5): R3 gained an ADDITIONAL
   conjunct and the scorer catalog a seventh mutation, `blind_min_rule` (the scorer drops `min_rule` and reads
   everything under the per-cell rule), so 2.10's per-class twin is reddened in fact -- the six pre-registered
   mutations leave the twin readable, so the addition is additive and substituting it for one of them would take R3
   False; (b) `5b686ac` (K4/K11): `run_t078_campaign`'s reported block carries REFUSED and no number when any seed's
   reading is unreadable (`_reported_mi`; `test_a_refused_reported_reading_carries_REFUSED_and_never_a_number`) -- the
   pin retains the old eight-zero shape (item 16), and the partial-refusal hazard the refuter named reproduces on
   every seed: coercing ONE seed's same-tick diff to 0.0 moves the reported same-tick t from -9.1089 to -5.2547 (seed
   5) .. -5.8957 (seed 3) with `degenerate` False, a live-looking number where a refusal stood (the refuter's -5.2547
   and the builder's -5.8957 are the same hazard on two seeds); (c) `278e7a5` (K14): `_decode` zeroes a column whose
   train-half sd is raw zero in BOTH standardized halves
   (`test_a_column_constant_in_the_train_half_cannot_amplify_into_the_decode`); on the pin the standardized `u_t` test
   half's |z|max is 4.894 (seed 5) to 11.431 (seed 6) -- the panel's printed range reproduces, the fold-builder's
   alternative range (6.952-17.234) did NOT reproduce with the shipped standardization and is not written -- and every
   seed carries exactly one raw-zero-sd train column (`focus_is_none`; two on seed 1, where `ignited` is also 0),
   which the pre-guard clamp already mapped to 0, so the guard moves no pinned decode. Also `b3886ab` (K10): a
   reachable red for the decoder's no-lookahead and train-half-standardization properties
   (`test_the_decoders_prequential_split_and_train_half_standardization_have_reachable_reds`), a test and not an
   instrument change, landed on a late-signal construction after the refuter's own sign-flip construction failed to
   separate. `tests/test_attention_schema.py` 11 -> 14 passed; the module's self-check line reads 6/6 reds; the
   catalog's printed mutation count, made stale by (a), was corrected by the builder's own `bddc37c`.
23. (appended 2026-09-15, after the panel) `_q95` IS THE THIRD SPELLING OF THE 95th-PERCENTILE ORDER STATISTIC, NOT
   THE SECOND (3.5 item 3 and 3.6 opening (5) say "a second spelling of `track_error`'s"):
   `substrate/_mathutils._p95_of` landed at `37c3203` (2026-09-06), nine days before `02d677e`, and `_mathutils`'s own
   docstring declares it the home for byte-equivalent duplicated helpers (`metacognitive_monitor` already imports from
   it). The panel's filed lift into `substrate/paired_contrast` was judged WRONG by its refuter and is REFUSED:
   `track_error` sorts its `draws` in place and then reads the median from the same list, so a non-mutating shared
   helper would move a pinned T077 number. The code half corrected the deferral's stated reason in the module
   (`8ce41be`, K7); the lift is the successor's, into `_mathutils`, with T077's in-place sort disclosed first. Opening
   (5) is amended by this item, not edited.
24. (appended 2026-09-15, after the panel) SECTION 1 ITEM 6's P2 REUSE OF `integration_measure.source_entropy` NEVER
   HAPPENED: the module imports exactly `paired_contrast` and `report_channel`, and `source_entropy` occurs in prose
   only. Corrected in the module header (`151d4f8`, K8), which also names `thalamus/ignition_instrument.
   _permutation_null` as the nearest incumbent -- a label-permutation band around a refit decoder read IN-SAMPLE -- so
   the header's "ABSENT from the tree" claim for a held-out decoder with a label-shuffle band rests on the word
   held-out alone: thinner than it reads, disclosed. A deviation from 1.6; no number touched.
25. (appended 2026-09-15, after the panel) THE DETERMINISM KEY COMPARES 5 OF THE 18 RECORDED FIELDS, and 3.4 records
   "(p) MET" on that key: the builder verified a mutant recording that passes the key while its full rows differ, and
   that each of the 5 keyed fields fires (p) when mutated. The panel's filed fix (extend the key to every field) was
   judged post-number bar movement by its refuter (2.2 freezes the key; the pin carries `reading.determinism`); the
   code half landed the docstring's scope sentence only (`d9f1fdd`, K2). Opening (11).
26. (appended 2026-09-15, after the panel) THE `chan disagrees` COUNTER EQUALS THE npz ROW COUNT ON EVERY SEED (1196 /
   1199 / 1199 / 1199 / 1199 / 1191 / 1195 / 1199; `per_seed.<s>.ticks_chan_disagrees` against `seed<s>_ticks`): the
   ticks where the two agree are exactly the ticks BEFORE the first `attend` call, where `_attn_real` is absent and
   `chan` falls back to `from_gated`, so the comparison is degenerate there and the counter reads "the two differ on
   every tick where both exist" (3.2's sentence is appended, not moved). Two more apparatus facts from the code half,
   reported: `t078_pack_latents` filters on `r.get("gated")` while `t078_g4` filters on `r["gated"]`, so the pin's own
   rows raise KeyError in the reader (K12; the docstring now names the npz join, no loader added) and return a SILENT
   (0, 64) array from the packer -- the worse failure, and nothing catches it (opening 12); and
   `dormancy_gate.EXEMPT["attention_schema"]` now records which branch fired (`679edc7`, K13) while its "permanent"
   wording and "RETIRED WITH THE ROW" clause stay as frozen, since 2.9 step 7 pre-registers that on a STOP nothing in
   `conclusion_protocol` changes.
27. (appended 2026-09-15 20:55, after the fold) THE CODEX CLOSING REPORT WAS REFUSED, so the fold stands with NO
   external refutation: sent 20:46:58 on head `39e2144` (`scratchpad/t078/codex_closing_prompt.md`, the fold's 3.7
   first, the corrected facts -- 6 of 8, two near-constant fields, the vision band live, the 2,000-step re-read --
   handed to it), it returned at 20:49:26 with "You've hit your usage limit ... try again at Sep 20th, 2026
   12:30 AM" after 71,746 tokens of tool calls (a graph check and six sandbox reads) and zero findings
   (`scratchpad/t078/codex_closing_report.md`). This is a WEEKLY cap, not the five-hour window 2.0 and 2.10 met, so no
   re-send is armed; if a reply lands after this row's push it is folded as a dated item here, never as a
   pre-registration amendment, per 2.10's own rule. Nothing in this record was read by a second model after `02d677e`
   -- the design read (2.0, 2.10) and the closing report both refused -- and the fold is battery-verified only (3.7,
   obs #35). Apparatus note: the first send-waiter fired on its own prompt echo (the prompt contains the words
   "usage limit"); the waiter now reads only Codex's output after the last `codex` marker (obs #2181's shape again).

### 3.7 The panel fold (2026-09-15; three lenses, refuters judging CLAIM and FIX separately)

The panel (instrument / pre-registration-honesty / disposition lenses at head `23eb085`) filed 46 findings; 39 STOOD
after refutation and 7 were REFUTED; 35 of the 39 filed fixes were judged WRONG by their refuters (obs #2146's shape
again). The fold landed the REFUTERS' minimal fixes, never the panel's, in two halves: the CODE half by an Opus
executor (23eb085..bddc37c, 14 commits: (r)'s refusal naming the arm; a refused reported reading carrying REFUSED and
no number; R3's printed description and its per-class twin reddened in fact by a seventh mutation; the standardization
guard with its constructed red; the prequential-halves red; the decoder-budget comment; the determinism docstring's
scope; the packer's and the recorder's real mechanism; `_q95` as the THIRD spelling with the panel's lift REFUSED; the
unperformed `source_entropy` reuse deleted; the pin-and-npz docstring with no loader added; the dormancy exemption
recording its branch; the builder's own stale count corrected -- three brief-versus-refuter conflicts resolved to the
refuter (no `_q95` lift, no pin loader, no exemption rewrite), one between two refuters (K4 versus K11, the more
complete landed), every commit followed by the fifteen census gates rc=0 and the 200-step re-derivation of the pin,
bit-for-bit) and the RECORD half here (3.0 x2, 3.2's table, the classifier line and the counter's meaning, 3.3's
tables with every null's centre, 3.4 x5 with the new (iii) and (iv), 3.6 x4 with openings (7)-(12), 3.5 items 8-26 and
four appended corrections under items 2, 4, 5 and 6). Every number written was REPRODUCED by the fold's own probes
first (`fold_verify_pin.py`, `fold_verify_extra.py`, `fold_verify_builder.py`, `redecode_2000.py`,
`fold_sensitivity.py`, `fold_detection_floor.py`, `verify_head.py`, all under `scratchpad/t078-panel/`); the refuters'
output and the builder's report were data -- one of the builder's five reported numbers did not reproduce and is not
written (item 22).

Where refuters disagreed the pin decided: the count of near-constant `u_t` fields (one vs three; the pin reads two
near-constant, one low-cardinality, item 9). NOT done, and why: no post-hoc sensitivity arm was added to the shipped
CORE self-check (the probes live in the scratchpad and read the pin, as the vision-channel probe does); the pin's
manufactured `reported.gate16` key is not hand-edited (write-once) and is read by item 16; no decoder constant moved
(item 13 records the 2,000-step re-read beside the pre-registered 200); no pin was regenerated; the determinism key
was not extended (item 25), `_q95` was not lifted (item 23), no pin loader was added (item 26) -- each a post-number
change its refuter refused. The 7 refuted findings, by substance: the disposition sentence generalizing a null from
one projection (its scope survives as item 15, the claim as filed did not); `u_t` carrying the action selector's own
margin with no control (the field list is the algorithm's own); R1 carrying an f(x)==f(x) conjunct (its control margin
can fail); the dormancy exemption promising retirement on the STOP branch (the exemption names Part II, not STOP); the
pin not self-attesting its head (the commit does); the invention register's #8 row lacking an outcome line (C-R6 sets
the precedent); the openings owned by no task row (tasks.md's dated note names them).

THE FOLD IS ITSELF UNREVIEWED CODE (obs #35): it was not re-panelled; it is battery-verified only -- the arc's
battery, named in the closing record commit of this push, is its review. The verdict G3 STOP x G4 PASS, the pin, 2.6's
STOP branch and the openings stand as recorded; the fold added openings (7)-(12), moved no bar, and dropped no
conjunct.
