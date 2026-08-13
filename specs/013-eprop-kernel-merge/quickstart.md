# Quickstart — running and validating the e-prop kernel MERGE

**Gate env**: `D:\AI-project\spiking-neural-data-lake\.venv-lake\Scripts\python.exe`
(python 3.14.6 free-threading, torch 2.11.0+cu128, cuda True, snntorch present, **tonic absent**).
Run everything from the repo root. Use PowerShell, not Bash, for anything touching the network.

---

## 0. The gate batch — run this at EVERY commit, not just at the end

`ruff format --check` is in the batch deliberately: a prior arc shipped formatting drift because
the local batch ran only `ruff check` while CI ran both.

```powershell
.venv-lake\Scripts\python.exe run_selfchecks.py
.venv-lake\Scripts\python.exe -m pytest -q
.venv-lake\Scripts\python.exe -m ruff check .
.venv-lake\Scripts\python.exe -m ruff format --check .
.venv-lake\Scripts\python.exe tools\p4c_ratchet.py --assert
.venv-lake\Scripts\python.exe tools\p6_gate.py --assert
.venv-lake\Scripts\python.exe tools\citation_guard.py --assert
.venv-lake\Scripts\python.exe tools\red_reachability.py --assert
.venv-lake\Scripts\python.exe tools\dependency_ledger.py --assert
.venv-lake\Scripts\python.exe -m snn_data_lake.substrate.import_manifest
```

**Before quoting any local gate number, check for untracked probe residue** — two stray files once
reddened p4c locally and were gone before anyone could reproduce it:

```powershell
git status --porcelain --untracked-files=all
```

Baseline at HEAD `47555a74` (measured): p4c **205** / no zone above baseline; p6 **5/5 ratchets
hold**; import_manifest **8 targets**; citation_guard OK; pytest **91.00%**; self-checks **298**.

---

## Commit 0 — evidence only, no behaviour change

### C0-1. The FR-018 unique-capability audit (gates the deletion)

For each of the **19** `surrogate_snn`-only definitions, write one of three dispositions with its
evidence: **covered** by `eprop_nmnist` / **ported** / **dropped, with reason**.

```powershell
.venv-lake\Scripts\python.exe -c @'
import ast
ns=lambda p:{n.name for n in ast.walk(ast.parse(open(p,encoding=\"utf-8\").read())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
s=ns(\"training/surrogate_snn.py\"); a=ns(\"src/snn_data_lake/cortex/eprop_nmnist.py\")
only=sorted(s-a); print(len(only)); [print(\" \",x) for x in only]
'@
```

Expected: **19** — `SpikingConvNet`, `LeakyLIF`, `SurrogateEncoder`, `_FastSigmoid`,
`_fast_sigmoid`, `_first_layer_grad`, `_flat_dim`, `_fwd_bwd`, `_heaviside_no_surrogate`,
`_prepare`, `_repo_root`, `_self_check`, `_self_check_body`, `_synthetic_frames`,
`assert_supported_reset`, `init_leaky`, `inner`, `lif_layers`, `ng_snn_data_file`.

**Do not wave these two through**: `SpikingConvNet` is a *convolutional* architecture where
`eprop_nmnist` is recurrent fully-connected — a different architecture, not a worse copy;
`_heaviside_no_surrogate` is a surrogate-OFF control. If either is load-bearing and unported, **the
deletion is blocked** (spec §ruling). That is a legitimate outcome, not a plan failure.

### C0-2. The FR-017 bit-exact pin — capture the BEFORE side

**This is the verification for Commit 4. The cosine assert is not.** Serialize, for BOTH consumer
modules, at a fixed seed on synthetic input (no dataset, no `tonic`):

| tensor | source |
|---|---|
| `forward.logits_sum` | `EpropNet.forward` / `EpropAudioNet.forward` |
| `forward.latent_sum` | same |
| `eprop_grads.grads` | the e-prop gradient list |
| `eprop_grads.loss` | scalar |
| `bptt_grads.grads` | the BPTT reference list |

Write to `tests/pin/{nmnist,shd}-before.pt` with `torch.save`, recording the
torch version and device in the same dict. Compare later with **exact** equality
(`torch.equal`), not `allclose` — a tolerance wide enough to survive a build difference is wide
enough to hide the reshape error this exists to catch.

**Runs locally only** (research.md D-3): local is cu128, CI is CPU torch; bit-exact float equality
across two builds is not a contract this project can hold. Its job is to catch a mistake made by
this lane on this machine between two commits minutes apart — same interpreter, same build, same
seeds — which is exactly where bit-exactness is meaningful.

---

## Commit 1 — retire the surrogate tier

### Pre-commit measurements (do NOT skip — these are [F] claims that gate the commit)

```powershell
# 1. Sweep tests/ for the checkpoint PATH STRING (only class-name refs were swept before)
Select-String -Path tests\*.py -Pattern "surrogate_nmnist|surrogate_snn|SurrogateEncoder|torch-snn"

# 2. Re-verify pc_cortex_torch's surro_acc really is always None before deleting that block
.venv-lake\Scripts\python.exe -c "import torch; d=torch.load('models/surrogate_nmnist.pth', map_location='cpu', weights_only=False); print(sorted(d.keys()))"
```

Claim to confirm: the checkpoint has only `['model','config']`, so `surro_acc` is always `None` and
the `:481` branch never fires **[F]** — i.e. the block was never an A/B.

### Acceptance

- Gate batch green.
- `import_manifest` reports **7** targets, floor at `:287` lowered 7 → 6 **in this commit**.
- p4c total 205 → 204, `training` zone 6 → 5 (a fall; the ratchet fires only on a rise).
- Zero live references outside `CHANGELOG.md` (SC-006):

```powershell
Select-String -Path (git ls-files) -Pattern "surrogate_snn|SurrogateEncoder|torch-snn" | Where-Object { $_.Path -notmatch "CHANGELOG" }
```

---

## Commit 2 — split the perception try

### Pre-commit measurements

```powershell
# p6 zero-delta is INFERRED, not measured -- measure it
.venv-lake\Scripts\python.exe tools\p6_gate.py --assert
```

**The recorder observable MUST be compared at >= 12 ticks, never 0.** A 0-tick comparison is
vacuous: at 12 ticks a rollback collapses `/cognition/attention_real` from a 3-key dict to a scalar,
237 → 235 leaf paths **[F]**. Any "recorder shape unchanged" claim measured at 0 ticks is discarded.

### Acceptance

- With the audio encoder blocked at import (`meta_path` blocker), the **vision tier survives** and
  `perception_tier != "stdlib"` (SC-008).
- Leaf-path count unchanged at >= 12 ticks (SC-007).
- Gate batch green.

**Not claimed**: this does not fix the off-root/CWD shape. Off-root, the relative checkpoint paths
at `brain.py:630`/`:664` fail their `os.path.exists` guards and the imports are never attempted
**[F]**. That is NG-004.

---

## Commit 3 — promote `eprop_shd` (highest gate risk: four gates move at once)

### Pre-commit measurements

```powershell
# 1. Does -m actually reach the self-check? It gates on `smoke = args.smoke or len(sys.argv)==1`
.venv-lake\Scripts\python.exe -m snn_data_lake.cortex.eprop_shd    # AFTER the move; expect exit 0

# 2. Depth-counted path logic a move would silently retarget
#    (surrogate_snn.py:337-366 has exactly that shape; eprop_shd was measured by nobody)
Select-String -Path training\eprop_shd.py -Pattern "__file__|dirname|parents\[|os\.pardir|\.\.\\|repo_root"
```

### All seven co-edits land in THIS commit

1. `git mv training/eprop_shd.py src/snn_data_lake/cortex/eprop_shd.py`
2. `"eprop_shd"` → `TORCH_TIER["cortex"]` (else `cortex` = new zone, allowance 0 → RED)
3. `"cortex.eprop_shd"` → `run_selfchecks.py` TORCH_GATED
4. `brain.py:668` → `from snn_data_lake.cortex.eprop_shd import EpropAudioEncoder`
   (the `_PKG` guard at `dormancy_gate.py:353` gives a bare-root import no edge → RED without this)
5. `docs/TRAINING.md:23` → the dotted `-m` form matching `:22` (STRICT citation_guard line → RED)
6. `import_manifest.py:126` → `UNGOVERNED_ROOT = ()`; `:287` floor 6 → 5
7. delete the stale comment at `import_manifest.py:111-113`

### Commit body MUST carry this caveat

`UNGOVERNED_ROOT = ()` records "no **AST-visible** root-shape imports", not "no governed →
`training/` edges". A surviving invisible one: `reasoning/eprop_lm_serve.py:261-264` does
`sys.path.insert` then `import eprop_lm as trainer` — a bare-name import `_targets()` cannot see.

**This commit does NOT discharge the P2 debt.** A move relocates duplication into the governed tree;
it does not retire it. Do not describe it as "the merge".

---

## Commit 4 — extract the kernel (the commit that actually closes the lane)

### C4-1. Re-run the `flatten_frames` subsumption A/B (do not carry the [F])

Prove B's body reproduces A's outputs bit-exactly on A's own input shapes, and that A's body raises
on B's 3-D input. If subsumption does NOT hold on any shape either class receives, **stop** — the
collapse is wrong and the kernel design changes.

### C4-2. The pin comparison — the acceptance step for this commit

```powershell
# capture AFTER, then compare with torch.equal (exact), not allclose
.venv-lake\Scripts\python.exe tests\pin\compare.py
```

Every tensor in the C0-2 table must be **bit-identical**. Any difference blocks the lane (SC-003).

> **Do not substitute the cosine assert for this.** `bptt_grads` computes its reference through the
> same `forward`, therefore the same `flatten_frames`; a defect there moves both sides together and
> the cosine stays high. Standing Invariant 5: a green gate that structurally cannot see the thing
> it checks is worse than no gate.

### C4-3. Demonstrate the kernel self-check's reachable RED (FR-019, SC-010)

Negate the broadcast error term feeding the eligibility product; the e-prop gradient anti-aligns
with the BPTT reference and the cosine falls below the floor. Record the observed value in the
commit body. **If nothing reddens it, remove the assert rather than softening it** — and report
that outcome.

### C4-4. P3 charge evidence

```powershell
.venv-lake\Scripts\python.exe -c "import sys; sys.modules['snn_data_lake.cortex.eprop_kernel']=None; import snn_data_lake.cortex.eprop_nmnist"
.venv-lake\Scripts\python.exe -c "import sys; sys.modules['snn_data_lake.cortex.eprop_kernel']=None; import snn_data_lake.cortex.eprop_shd"
```

Both MUST fail. That is charge demonstrated, not a consumer claimed.

### C4-5. Survivor count (SC-001)

Re-run the AST census from data-model.md §3. Every symbol in §1 must have **exactly one** definition
site tree-wide, and the geometry constants must remain at **two** sites with **distinct** values.

---

## Before every commit message is written (FR-020)

Re-derive every `file:line` cite **after the last edit of that commit**. This lane rewrites both
cited files; POLICIES.md records the project already paying for this — *"a line citation into an
actively-developed file is stale on a timescale of minutes, not releases."* A prior arc in this
repo shipped four commit messages whose self-check counts were written by increment rather than
read from output.

```powershell
.venv-lake\Scripts\python.exe run_selfchecks.py | Select-Object -Last 5   # read the count, don't infer it
```
