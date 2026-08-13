# Phase 1 Data Model — what the kernel holds, what stays behind

All line numbers at HEAD `47555a74` and **all of the below re-derived by the main agent**, not
relayed. Per FR-020 these cites rot the moment Commit 1 lands; re-derive before quoting them in any
commit message.

`A` = `src/snn_data_lake/cortex/eprop_nmnist.py` (vision, N-MNIST, governed today)
`B` = `training/eprop_shd.py` (audio, SHD) → `src/snn_data_lake/cortex/eprop_shd.py` after Commit 3
`K` = `src/snn_data_lake/cortex/eprop_kernel.py` (new, Commit 4)

---

## 1. Definitions that MOVE into the kernel

### 1a. Byte-identical, module scope

| definition | A | B | lines | role |
|---|---|---|---|---|
| `class _SpikeFn` | L352-363 | L344-355 | 12 | triangular surrogate `autograd.Function` |
| `bptt_grads` | L366-387 | L358-379 | 22 | **the independent BPTT reference** |
| `_cosine` | L390-395 | L382-387 | 6 | the e-prop invariant's comparator |
| `_apply_grads` | L526-536 | L515-525 | 11 | Adam step over e-prop grads |
| `evaluate` | L539-553 | L528-542 | 15 | evaluation loop |

`_SpikeFn.forward` (A L353-357 / B L345-349) and `_SpikeFn.backward` (A L359-363 / B L351-355) are
byte-identical too, but are **subsumed by the class** and must not be double-counted.

**Non-overlapping module-scope total: 66 lines per file.**

### 1b. Byte-identical, into a shared LIF/e-prop base class

| definition | A | B | lines |
|---|---|---|---|
| `psi` | L222-224 | L188-190 | 3 |
| `offdiag_mask` | L230-231 | L198-199 | 2 |

### 1c. One-line-different, into the same base class

| definition | A | B | ratio | the single difference |
|---|---|---|---|---|
| `forward` | L233-255 | L201-223 | **0.9939** | docstring only — `A:235` `"""Serve: x [B, T, C, H, W] event frames -> …` vs `B:203` `"""Serve: x [B, T, C] audio frames -> …` |

Per FR-008 the merged docstring describes **both** shapes. Silently adopting one modality's wording
is not acceptable.

### 1d. The one deliberate collapse (D-4, and the site of R-001)

| definition | A | B | disposition |
|---|---|---|---|
| `flatten_frames` | L226-228 (3 lines) | L192-196 (5 lines) | **B's body wins**, one concrete method, not abstract |

`B`: `x.reshape(x.shape[0], x.shape[1], self.n_in)` — subsumes `A`'s
`x.reshape(*x.shape[:-3], self.n_in)` on every shape either class receives; `A`'s body *raises*
`RuntimeError` on `B`'s 3-D input **[F]**. The per-modality seam is `self.n_in`, already a
per-subclass attribute (`A:202`, `B:173`).

**This is the only definition in the table whose merge rests on judgment rather than byte-identity,
and it is the one the cosine assert cannot verify.** See [quickstart.md](quickstart.md) C4-1/C4-2.

### 1e. Shared encoder base

| definition | A | B | lines |
|---|---|---|---|
| encoder `__init__` (`EpropEncoder` / `EpropAudioEncoder`) | L426-428 | L408-410 | 3 |

### 1f. Constants — verified same-valued in both files

| constant | A | B | value |
|---|---|---|---|
| `DEFAULT_ALPHA` | :149 | :133 | 0.9 |
| `DEFAULT_V_TH` | :150 | :134 | 0.6 |
| `DEFAULT_GAMMA` | :151 | :135 | 0.5 |
| `DEFAULT_DAMP` | :152 | :136 | 0.3 |
| `DEFAULT_KAPPA` | :153 | :137 | 0.9 |
| `EPROP_BPTT_COSINE_FLOOR` | **:158 = 0.5** | *absent — bare `0.5` hand-written at `:830`* | 0.5 |

`EPROP_BPTT_COSINE_FLOOR` is one-sided today: named in `A`, hand-written as a magic number in `B`.
Hoisting it into `K` and importing it in both is FR-003 and retires a P1-adjacent hardcode.

**Total leaving each file: ~102 lines, decorator-inclusive.**

---

## 2. Definitions that STAY per-consumer

Measured **not** byte-identical — genuinely different audio-vs-vision data paths:

`eprop_grads`, `encode`, `classify`, `_prep`, `load`, `config`, `main`, `build_parser`, `run_smoke`,
`train_and_eval`, `_make_synthetic`.

`A`-only: `_chance_bound`, `_load_data_module`, `_ng_snn_dir`.
`B`-only: `_make_shd_loaders`, `_self_ref`, `_shd_collate`.

**`encode` (r=0.960) and `classify` (r=0.947) are NOT hoisted** (FR-009). A similarity ratio cannot
separate material from cosmetic difference; hoisting on a ratio is the vacuity a skeptic already
overturned once in this lane's measurement pass.

### Geometry constants — verified to hold DIFFERENT values, so they must not move

| constant | A (vision) | B (audio) |
|---|---|---|
| `NUM_CLASSES` | **10** (:134) | **20** (:106) |
| `DEFAULT_T` | **15** (:135) | **100** (:107) |
| `DEFAULT_HIDDEN` | **64** (:136) | **128** (:108) |

One-sided geometry also stays: `SENSOR_H` 34 (A:131), `SENSOR_W` 34 (A:132), `IN_CH` 2 (A:133),
`CHANCE_SIGMAS` 3.0 (A:164), `CHANNELS` 700 (B:105).

Moving any of these would hardcode one modality's geometry into the other's path — the exact P1
failure the lane is otherwise reducing.

---

## 3. Survivor count (SC-001 — the P2 acceptance table)

P2: *"count the survivors: if two implementations of the same job are still reachable when you are
done, it was not a merge."*

| mechanism | before | after |
|---|---|---|
| triangular surrogate spike fn (`_SpikeFn`) | 2 (A, B) | **1** (K) |
| independent BPTT reference (`bptt_grads`) | 2 | **1** (K) |
| cosine comparator (`_cosine`, torch) | 2 | **1** (K) |
| Adam step (`_apply_grads`) | 2 | **1** (K) |
| evaluation loop (`evaluate`) | 2 | **1** (K) |
| LIF `psi` / `offdiag_mask` / `forward` / `flatten_frames` | 2 each | **1** each (K base class) |
| encoder `__init__` | 2 | **1** (K encoder base) |
| LIF constants (×5) | 2 each | **1** each (K) |
| cosine floor | 1 named + 1 magic literal | **1** named (K) |
| **perception encoder for the brain's vision tier** | **2** (e-prop, surrogate) | **1** (e-prop) |

**File count rises 2 → 3. Mechanism count falls 2 → 1. That is the reduction P2 requires**; a merge
is measured in surviving mechanisms, not in files.

Out-of-scope surrogates that remain (NG-002, honestly counted rather than quietly excluded):
`reasoning/rsnn_delays_torch.py:91`, `training/eprop_lm.py:78`, `training/eprop_lm_alif.py:103`,
`training/ar_spiking_lm.py:74`. These implement a *different* contract (see contracts/, D-7) and are
not "the same job" under P2.

---

## 4. Gate-state transitions

| gate | before | after C1 | after C3 | after C4 |
|---|---|---|---|---|
| `p4c_ratchet` total | 205 | 204 | 203 | 203 |
| `p4c` `training` zone | 6 | 5 | 4 | 4 |
| `import_manifest` targets | 8 | 7 | 6 | 6 |
| `import_manifest` floor (`:287`) | `>= 7` | `>= 6` | `>= 5` | `>= 5` |
| `UNGOVERNED_ROOT` (`:126`) | `("training",)` | `("training",)` | **`()`** | `()` |
| `TORCH_TIER["cortex"]` | *(no eprop_shd)* | — | `+eprop_shd` | `+eprop_kernel` |
| `p6` `undeclared_import_pairs` | 11 | 11 | 11 | **11** (D-1 keeps it flat) |

The p4c ratchet fires only on a **rise** (`tools/p4c_ratchet.py:250`, `live > allowed`), and a
missing baseline zone defaults to 0 (`:249`) — which is why the `TORCH_TIER` registration in C3/C4
is mandatory and not cosmetic: an unregistered `cortex` file becomes a **new zone with allowance 0**
and reddens immediately.

---

## 5. Validation rules (derived from FRs)

1. After C4, each symbol in §1 has **exactly one** definition site tree-wide (SC-001).
2. Geometry constants in §2 remain at **two** distinct sites with **distinct values** — merging them
   is a defect, not an improvement.
3. `K` imports nothing from `A` or `B` (FR-005, cycle).
4. `K`'s self-check has a demonstrated reachable RED (FR-019, SC-010).
5. All 19 surrogate-only defs carry a written disposition before C1 (FR-018, SC-009).
6. Numeric outputs bit-identical across C4 (FR-017, SC-003).
