# Quickstart — Validating the Weight-Transport Resolution

Gate env `.venv-lake`. Run from the repo root. **PowerShell**, not Bash.

**Every step below carries a positive control in the same command** (FR-016). No step may report a
negative — "flat", "none", "unchanged", "clean" — from a command that has not been shown able to
produce a positive. Assert the denominator.

---

## 0. The gate batch — at EVERY commit, including docs-only ones

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py && .venv-lake/Scripts/python.exe -m pytest -q && .venv-lake/Scripts/python.exe -m ruff check . && .venv-lake/Scripts/python.exe -m ruff format --check . && .venv-lake/Scripts/python.exe tools/p4c_ratchet.py --assert && .venv-lake/Scripts/python.exe tools/p6_gate.py --assert && .venv-lake/Scripts/python.exe tools/citation_guard.py --assert && .venv-lake/Scripts/python.exe tools/red_reachability.py --assert && .venv-lake/Scripts/python.exe tools/dependency_ledger.py --assert && .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.import_manifest
```

**There is no docs-only carve-out** anywhere in POLICIES.md. Lane 017 assumed one; it was a finding.
Count the exit codes — a partial log reads as a pass (that happened, in lane 017, to me).

---

## 1. Baseline, before anything changes

Record the module's own numbers: XOR transport 0.993 / FA 0.990 / LatentQ 0.467; linear 0.983 /
0.993 / 0.993. **Control**: assert the self-check produced output at all before reading a verdict
from it.

---

## 2. The criterion's own qualification (C2, before it judges anything)

Per [contracts/criterion.md](contracts/criterion.md), in order:

1. **Identity** — transport against itself scores **1.0**. If not, the harness is wrong and nothing
   below means anything.
2. **Permutation null** — same magnitude, same element multiset, shuffled direction, scores **low**.
   Failing this means the criterion measures magnitude, and it is **deleted, not weakened**.
3. **Reachable red** — construct the input that makes it fail; record the actual output.

**Do not proceed to step 3 until all three are recorded.**

---

## 3. The measurement (C3)

Scratch probe in `tests/probe_015/` — **not** in `mlp_q.main()`, which
re-derives on every gate run (research.md D-4).

Matrix: `{transport, fixed_random, converging, permutation_null}` × `bseed 1..8` × `epoch 1..30`,
recording cosine + held-out accuracy on **both** fixtures.

**Compare against [research.md](research.md) D-3 as written.** The prediction was registered before
the run; the comparison is against that text, not a revised one.

**Controls**: assert the row count is `4 × 8 × 30`; assert the transport variant reads 1.0 at every
epoch (if it does not, the probe is broken); assert the permutation null is present and low.

---

## 4. THE DECISION POINT

| outcome | action |
|---|---|
| converging `B` reaches \|cos\| > 0.5 on **≥ 6 of 8** by pass 30 **and** holds the shipped accuracy gates | proceed to step 5 (adopt) |
| anything else | **FR-007 fires. Stop.** Commit the null with its numbers, restate the site's disposition, update the adjudication to say the resolution was attempted and refuted. The null is a deliverable. |

The bar and the epoch count come from D-3 and **are not renegotiated here.**

---

## 5. Adoption, only on the supporting outcome (C4)

- Replace the feedback vector inside the existing `dh` step. **Delete the transport line.**
- **Count survivors**: exactly ONE credit-assignment path reachable in `mlp_q`. A flag selecting
  between transport and `B` fails FR-008. *Control*: grep the module for the retired form and assert
  **zero** hits — with a positive control proving the grep can match.
- Re-run the module's shipped gates (> 0.85 held-out, > 0.30 over the linear control) on every draw.
- **Name and quantify any capability lost** (FR-009).
- Settle the decay term (P1): derived from `lr`, or absent with a small init — with the measurement
  that decided it.

---

## 6. The `predictive_coding` adjudication (C6)

Independent of step 4's outcome — it ships either way (research.md D-6).

Both directions stated; **not** rested on the Predictive E-prop preprint, which has **0** full-text
hits for transport/symmetry/alignment terms against **30** for BPTT. Silence is not endorsement.

---

## 7. Scope checks, by diff not by promise

```bash
git diff --name-only main..HEAD -- src/snn_data_lake/brainstem/
```

**Must be empty.** This lane carries no safety-envelope approval.

Also: `git diff --name-only main..HEAD -- src/` must list **`mlp_q.py` and nothing else**; the
dependency ledger must show no new entry; and the probe must remain in `tests/probe_015/`, never promoted to
`src/`.

**Re-derive every `file:line` cite after the last edit of the commit that touches it** — and prefer a
symbol or marker. Lane 017 rotted its own cites three times, once in a file it never opened, because
editing a file invalidates every citation *into* it and `citation_guard` structurally cannot see that.
