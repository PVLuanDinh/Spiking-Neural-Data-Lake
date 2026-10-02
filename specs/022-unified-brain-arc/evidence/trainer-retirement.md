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

## 2. Results

*(Filled after the runs; not before.)*
