# The four backprop trainers: port the capability, then retire (pre-registration and record)

**The decision.** The owner chose "Port-then-retire" from the session's options on 2026-10-02 (plan G order, dated note
`e46cbcdc`): `training/trainer`, `offload`, `mnist` and `spiking_lm` retire in one arc at the seam after G8-3's push,
outside any campaign. A learning capability is re-implemented e-prop-side in a governed module (a BPTT module is never
MOVED into the governed tree, which would be a promotion); a non-learning capability moves to its natural home first;
otherwise "none". One commit per trainer in dependency order; each carries its e-prop verdict-register entry in
`docs/POLICIES.md` and a re-derived `gradient_claim_gate` census; every head stays importable. `training/mnist_genn.py`
(EventProp/GeNN, imported only by `cli.py`) is not covered; it, `cli.py` and `snn-train` stay, and its disposition goes
to the owner.

## 1. Protocol (written before the first build)

### 1.1 Census at the base `d065a80a` (re-taken by command; the full census: the operator's prep file, out of tree)
- Consumers: `training/cli.py` imports all four lazily; `growth/brian2_backend.py` imports `trainer.SpikingReadout`,
  `_generator` and `rate_encode` in `conjunct_m` (the only forward-parity check of a migrated call site, CI-asserted)
  and lists all four in `MIGRATED_CALL_SITES` (`conjunct_i` asserts the list); `offload` imports `trainer`;
  `tests/test_trainer_checkpoint.py` imports `trainer`. Nothing else.
- `snn-train` is already broken at the base: `cli.py` imports the retired `reasoning.torch_device` at two sites (the
  G8-3 panel's pre-existing finding). Fixed first, by its own commit.
- Retiring `spiking_lm` leaves `sensorimotor/spiking_lm_serve.py` with no importer (dormant); retiring the LAST of
  `trainer`/`spiking_lm` leaves `growth/brian2_backend.py` reachable from no live root.
- An e-prop char-LM exists only outside the governed tree (`training/eprop_lm.py` at the repo root, its model
  `models/eprop_lm.npz`, its server `reasoning/eprop_lm_serve.py`); the only module that reaches that server is the
  benchmark `reasoning/blimp_bench.py` -- no live governed referent.
- `mnist`'s exported brain has a hidden layer (`fc1` [H][784], `fc2` [10][H]; `cli.py` default H 256).
- All four declare no `GRADIENT_CLAIM` (ungoverned `MISSING_DECLARATION`).

### 1.2 Commits, in this order (a session call consistent with the owner's two constraints: offload and mnist before
trainer; the last retirement strands `brian2_backend`, so its EXEMPT row and the `conjunct_m` fixture move land last)
0. `cli.py`: the two `reasoning.torch_device` imports re-pointed to `infra.device`; `snn-train --help` and one CLI path
   run (the G8-3 panel's pre-existing finding).
1. `offload` retires. Learning capability: the same 8x8 `SpikingReadout` training as `trainer` (ported at commit 4); its
   unique part is a BPTT memory technique (gradient checkpointing, CPU-resident Adam) with no e-prop analogue (e-prop
   keeps no sequence-length activation tape; Bellec et al. 2020) -> "none". Non-learning consumed: none.
2. `mnist`: PORT, then retire. An e-prop trainer for `trained_brain`'s 784->H->10 network in `sensorimotor/` (beside
   `trained_brain.py`), reusing `plasticity/eprop_kernel` (P2: no third e-prop implementation): the output layer by its
   local error x eligibility trace; the hidden layer's learning signal through a separate feedback matrix B that receives
   the same increment as `fc2` (Kolen-Pollack, the `mlp_q` precedent in POLICIES' register), never `fc2` transposed. It
   writes the same `save_brain` export. MNIST is read from the local cache (`data/MNIST`); no download.
3. `spiking_lm` and `sensorimotor/spiking_lm_serve.py` retire together. Learning capability: a char-level LM. Its
   e-prop form exists ungoverned (1.1) with no live referent; governing it is a separate P4c arc, filed as a carried
   opening, NOT done here (a promotion inside this arc would be scope creep). With no producer left, the serve core has
   no honest consumer (P3 disposition 2: retired). The lost capability is measured and written: the BPTT model's
   perplexity on its own corpus split, from its own code path, before deletion.
4. `trainer`: the `conjunct_m` fixture (Linear(64,4,bias=False) + `BrianLIF` 25 steps + seeded Bernoulli encoder) moves
   into `growth/brian2_backend.py` as a private fixture; `test_trainer_checkpoint.py` retires; `brian2_backend` gets a
   dormancy EXEMPT row (permanent: "the CI-asserted Brian2 reference simulator, owner decision 2, CPU parity oracle");
   then `trainer` retires.
Every retirement commit: `MIGRATED_CALL_SITES` edited; `cli.py` importable (the retired path refuses with a message);
the p4c `training` zone ratcheted down by its own command; the register entry (shape of the `mlp_q` entry); the
doc-budget ceiling for POLICIES raised by the measured words of that entry ("measured + 0", reason in the commit;
headroom is 10 words); `gradient_claim_gate` census quoted before/after.

### 1.3 Reds and controls
| id | red / control | expected |
|---|---|---|
| T-R1 | `conjunct_i` with a retired module still listed | red, at each retirement commit |
| T-R2 | `conjunct_m` output after the fixture move | byte-identical to the base's; a planted weight change reds |
| T-R3 | after commit 4, the `brian2_backend` EXEMPT row removed in a copy | `dormancy_gate` red |
| T-R4 | the mnist port's hidden-layer update vs an independent BPTT gradient on one batch | cosine > 0.5 (the invariant's bar); a sign-flipped learning signal must fail it |
| T-C1 | P2 A/B: `trainer.train_model` (BPTT) vs `readout/snn_classifier.py::train` (LOCAL), same 8x8/4-class task, S = 8 seeds | Part F's paired one-sided t at 1.895 on test accuracy; "nothing unique to port" ONLY if BPTT does not beat the local rule by the bar; if it does, an e-prop arm on `eprop_kernel` is built and A/B'd before `trainer` retires |
| T-C2 | the mnist port vs `mnist.train_model` on identical seeds and budget | both test accuracies reported (P2's A/B), S = 8 at Part F's bar |

### 1.4 Null disposition (not a stop)
A port that fails T-R4 or loses T-C2 by the bar is NOT landed; the trainer is still retired (the owner's decision; P8:
deletion on a null) and its register entry and the next owner brief record "capability lost: <what>, measured
<numbers>". The arc stops for the owner only if a retirement would change an envelope file or `tools/envelope_hashes.json`.

### 1.5 Not covered, reported
`training/mnist_genn.py` (kept; disposition to the owner). The repo-root `training/ar_spiking_lm.py` (a fifth BPTT
trainer, outside the four) and the root e-prop LM pair: reported only.

## 2. Results (2026-10-02; commits `073b3186`, `5bec2535`, `76c877ba`, `360b957d`)

### 2.1 T-C1: `trainer` (BPTT) vs `snn_classifier` (LOCAL), at `073b3186`
Same data per seed: both call `random.seed(seed)`, `templates()`, `dataset(TRAIN_PER_CLASS)` and `dataset(TEST_PER_CLASS)`
from `readout/snn_classifier` (`trainer` imports them). `trainer.train_model(seed=s, device="cpu")` at its defaults;
`snn_classifier.train` then `evaluate` on the test split. S = 8.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| BPTT | 0.5 | 0.975 | 0.5 | 0.75 | 0.7375 | 0.75 | 0.5 | 1.0 |
| LOCAL | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9875 |

Mean 0.7141 against 0.9984; paired one-sided t (BPTT - LOCAL) = -3.90 against the bar 1.895; exact sign-flip FPR 0.0625.
**BPTT does not beat LOCAL: "nothing unique to port" holds and no e-prop arm is built.** Disclosed: LOCAL is at ceiling,
so BPTT had at most 0.0016 of headroom; three BPTT seeds sat at chance. The verdict covers the pre-registered task only.

### 2.2 The `mnist` port: T-R4 PASS, T-C2 NULL (pre-registered disposition 1.4 applied)
The port (`sensorimotor/eprop_trained_brain`, e-prop with a Kolen-Pollack feedback matrix B, never `fc2` transposed; its
forward equals `trained_brain.predict_scores` exactly) was built by a Sonnet stage builder; every number below was
re-run by the orchestrator and matched to the digit.

T-R4 (seed 0, one epoch; fixed batch = the first 128 test images, spike seed 4242), hidden-layer update vs an
independent BPTT gradient: cosine **0.9933** after training (bar 0.5), 0.0185 at initialisation, **-0.9933** with the
learning signal sign-flipped (the red); cos(B, W2) 0.9991 after training. The readout layer's cosine is 1.0.

T-C2 (one epoch, H 256, T 25, batch 128, lr 1e-3, seeds 0-7, the full 10k test set):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| BPTT (`mnist.train_model`) | .9555 | .9518 | .9588 | .9594 | .9565 | .9605 | .9566 | .9531 |
| port | .9399 | .9337 | .9450 | .9464 | .9411 | .9384 | .9464 | .9442 |

Means 0.9565 against 0.9419; t (BPTT - port) = **9.79** against 1.895. A second instrument agrees: BPTT's trained
weights loaded into the port's network and scored by the PORT's own evaluator read 0.9570 (its own evaluator 0.9565,
largest per-seed gap 0.0027), so the 1.5-point gap is not an evaluator artifact. **Disposition 1.4: the port is NOT
landed** (module, tree diff and every result file archived out of tree); `mnist` still retires.

Disclosed: (a) the one-epoch budget was fixed in the builder's brief before the run (session handoff, checkpoint 18),
not in this file; (b) the port's `KP_DECAY` = 1e-2 was chosen by a sweep on a train-split hold-out (train[:50000] ->
train[50000:]) before T-R4 or T-C2 ran, with the selection rule written after reading the sweep table; (c) the
builder's one-seed split of the gap (about 1.0 point from B != W2, 0.5 from e-prop truncation) was NOT reproduced;
(d) three epochs, or B := W2 (transport, which the e-prop invariant forbids pending the owner's adjudication of the
symmetric-e-prop opening), are untested hypotheses for the owner, not re-runs of a null.

Capability lost: the validated MNIST trainer for `trained_brain` (BPTT, 0.957 at one epoch). `training/mnist_genn`
(kept; disposition to the owner) is not shown to replace it: it imports, trains and exports a loadable brain from a
scratch copy of the local MNIST cache, but scored chance, test 0.0785 and core-served 0.005 (200 images), after one
epoch on the full training set (0.097 / 0.015 at a 10,000-image smoke scale); longer budgets are untested. *Fold
(Codex panel): commit `5bec2535`'s message and the first wording here said no working MNIST trainer remains, which
the one-epoch run does not show.*

Archive (machine-local, outside this repository): the operator's memory archive `trainer-port-null-2026-10-02/`
holds the port module `eprop_trained_brain.py` (sha256 `b0e808e4453e...`), the diff of its four in-tree registrations
(`builder_tree_edits.diff`, `564551da9e44...`), and the drivers with their outputs: `tc2.py` (`4744c5395897...`) and
`tc2_arm.py` (`cb53f72129e1...`) -> `tc2_result.json` (`ca0de99bd729...`); `measure_tr4.py` (`11464d3fa5d9...`) ->
`tr4_result.json` (`56d99df6497c...`); `tc2_one_evaluator.py` (`67fa45015559...`) -> `tc2_one_evaluator.json`
(`af8477f66789...`); `kp_decay_sweep.py` / `.json` and `diag_gap.py` / `.json`. Replay: check out `073b3186`, copy the
module into `src/snn_data_lake/sensorimotor/`, apply the diff, then run `tc2.py <out_dir>` (T-C2) and
`measure_tr4.py <out.json>` (T-R4) with the main checkout's `.venv-lake` and `PYTHONPATH=src`.

### 2.3 `spiking_lm`'s lost capability (commit 3), at `073b3186`
`snn-train --lm --seed s --export-lm <scratch>` from a scratch directory, defaults (25 steps, the built-in 512-character
corpus, vocab 26, d_model 96, 2 blocks, seq_len 64, 60,698 parameters). The code path has no held-out split, so its
perplexity is in-sample: 15.5351, 14.6237 and 14.3979 at seeds 0-2 (final loss 2.7499 / 2.6579 / 2.6459), against
uniform 26, unigram 17.14 and maximum-likelihood bigram 7.20 on the same text; the greedy transfer sample through
`spiking_lm_serve` was degenerate at every seed.

### 2.4 The reds
| id | observed |
|---|---|
| T-R1 | each of the four retired modules planted back into `MIGRATED_CALL_SITES`: `conjunct_i` reds 4 of 4; the committed list passes |
| T-R2 | `conjunct_m`'s seeded (currents, spikes, sums) digest after the fixture move = the trainer path's at the base (sha256 `2b599434...`); a planted 1e-3 weight change gives `67e93ac6...` (spike counts alone, 19 and 19, would not have caught it) |
| T-R3 | at `360b957d`'s tree before its EXEMPT row existed, `dormancy_gate` red on exactly `['growth.brian2_backend']`; with the row: 236 reachable, 62 exempt, 0 dormant |
| T-R4 | 2.2 |

### 2.5 Ratchets per commit
`gradient_claim_gate` MISSING DECLARATION (all scope) 25 -> 24 -> 23 -> 22 -> 21, governed 0 throughout; the p4c
ungoverned total 200 -> 199 -> 198 -> 197 (commit 1 wrote 200); the POLICIES ceiling 14362 -> 14405 -> 14467 -> 14531 ->
14582, each step the measured words of that commit's register entry ("measured + 0", headroom 10 words).

### 2.6 What the build found that the protocol did not say
- `MIGRATED_CALL_SITES` listed eleven modules, not only the four trainers; the other seven are themselves unreachable, so
  1.1's claim that the last retirement strands `brian2_backend` held, and was measured (T-R3), not assumed.
- `sensorimotor/spiking_lm_serve` is the first frozen pre-G8 label with no live module. It left the live `REGIONS` table
  only; `LEGACY_REGIONS` and `brain_geometry`'s `COGNITIVE_CORTEX` are untouched, and `layout()`'s sha256 is identical
  before and after (`8212ce9d...`). The region gates guard `package_of() is None` by design and pass;
  `tests/test_g8_move_gate`'s `_MOVED` pin went 287 -> 286.
- Two dead reproduction paths outside the code (`docs/coverage/Language.md`'s self-check command, `specs/005`'s baseline
  diff) were fixed with dated notes; `amdahl_gate`'s distinct-from marker for the deleted `verify_transfer` was removed.
- Tooling traps, each caught before a commit: a shell heredoc ate a backslash (scripts now go through the file writer),
  the shell's rewritten `grep` returned a false zero on a known positive, and one edit helper misread a file's line
  endings.

### 2.7 For the next owner brief
`mnist_genn`'s disposition (chance after one epoch; longer budgets untested); the symmetric-e-prop adjudication (a B := W2 port would be transport);
governing the root e-prop char-LM; the two untested `mnist` hypotheses in 2.2 (d).
