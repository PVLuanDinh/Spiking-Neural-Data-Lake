# Session record: representational drift, D1, and the pre-build audit (2026-10-27/28)

**Session.** Owner: Luan Dinh. Cloud session "Neuroscience research topics and questions", a Claude Code cloud container
(`https://claude.ai/code/session_01TeA2yYimjAiHiGFzw6hqHX`).

**Scope.** This file records every finding, decision and plan from the session, so that the owner and the operator
sessions work from one account. It is a session record, not a pre-registration and not a governing document. Where it
disagrees with a tracked authority (POLICIES, the plan, an evidence record), the authority wins.

**Base.** `main` = `6612ce72`, the 2026-10-25 closing record (battery GREEN on `c79a77ba`). No commits landed on `main`
during the session.

## Contents

1. [Branches and commits](#1-branches-and-commits)
2. [Research map: representational drift and the engine](#2-research-map-representational-drift-and-the-engine)
3. [D1: L1b's owed co-adaptation control](#3-d1-l1bs-owed-co-adaptation-control)
4. [The cloud environment](#4-the-cloud-environment)
5. [Pre-build audit](#5-pre-build-audit)
6. [Owner decisions 1-5](#6-owner-decisions-1-5)
7. [Consultation with the operator sessions](#7-consultation-with-the-operator-sessions)
8. [Plan, in order](#8-plan-in-order)
9. [Open questions for the owner](#9-open-questions-for-the-owner)
10. [Artifacts](#10-artifacts)

---

## 1. Branches and commits

| branch | commit | what |
|---|---|---|
| `research/representational-drift-map` | `81c1c331` | `research/representational-drift/MAP.md`: the drift research map |
| `d1/readout-staleness-prereg` (contains the map commit) | `430037a6` | D1 pre-registration, committed alone before any build or run |
| | `0ede3cde` | Amendment A0: territory measured before the build; P1 moved to the pre-refit gap |
| | `2e946d8d` | Erratum E1 (the null-scale gate was red on this branch, now fixed) and Amendment A0b (build-path findings) |
| | *(this file)* | the session record |

There is no pull request. Nothing has been pushed to `main`. The operator registers T121 and D-R1 in the plan files.

## 2. Research map: representational drift and the engine

Full map: `research/representational-drift/MAP.md`.

1. **No activity-level drift instrument exists in the tree.** The word "drift" already means three other things there:
   - `instruments/prior_drift_harness.py`: weights against the H01 prior;
   - `infra/drift_monitor.py`: input PSI;
   - `DriftRecord`.
2. **The engine is built to hold codes still.** Grown input rows are frozen after admission, and so are the chat rank
   code, the e-prop encoder and the reservoir after its adaptation window. The real drift sources are:
   - **S1** prototype refinement / relabelling (`growth/growing_learner.py`);
   - **S2** population turnover through admit and prune (`growth/ng_snn.py`);
   - **S4** predictive-coding `r2`, computed every tick and read by nothing but a scalar;
   - **S7** the episode attractor in sleep;
   - **S6** grown-unit latencies under `GrownReturnPort`, G9 scratch homes only.
3. **The engine's one measured drift pathology is the field's central problem.** L1b's Oja rule moved the grown input
   rows under a readout the growth gate held stale, and accuracy collapsed. The record says "the READOUT must
   co-adapt with any moving input" (`evidence/l1b-synaptic-online.md:312`), and its owed control was never run.
4. **Ranked studies.**

   | study | what it asks | attaches to |
   |---|---|---|
   | D0 | the instrument, validated on synthetic data and mouse data (Marks & Goard 2021; Deitch et al. 2021) | prerequisite |
   | **D1** | L1b's owed control | G5/T113, G-R5 |
   | D2 | prototype and population drift on lake-stored probes | G3/T111 |
   | D3 | coordinated drift in predictive coding (Qin et al. 2023) | G5 |
   | D4 | drift across sleep vs wake | G6/R3 |
   | D5 | grown-code drift and restart realignment | G9 |
   | D6 | drift under a Benna-Fusi cascade | R3; no published study measures this |

5. **Deliberately not proposed:** any new plasticity rule, drift-guided pruning, or a sleep-benefit sentence. Each would
   collide with the re-proposal guard (AGENT_COLD_START §10) or P4d's "Not claimed".

## 3. D1: L1b's owed co-adaptation control

Record: `specs/022-unified-brain-arc/evidence/d1-readout-staleness.md`. Proposed registration, for the operator: task
T121, red family D (row D-R1; the letter is free).

### 3.1 Design as it stands (pre-registration plus A0, E1, A0b)

**Question.** Does a stale readout carry L1b's Oja cost to the growth gate's evidence?

**The territory that motivated it.** At HEAD the live prediction co-adapts through L1c's online delta readout, but the
gate's evidence on a closed-gate refit is still scored with `_batch_beta`, the last open-gate solve
(`growth/neurogenesis_engine.py:418-425, 663-675`). So the null-scale triage's claim that the cause "is gone since
L1c" (row 2) holds for the live prediction only. L1b's Oja arm was also never re-read on the scoring path fixed on
2026-09-07.

**Arms.** In-brain, `make_brain` on scratch homes, 800 ticks, HEAD's engine configuration (delta readout, T047 gate):

| arm | input rows | role |
|---|---|---|
| FROZEN | never move | incumbent; data-drift staleness only |
| FROZEN_B | never move | A/A, bit-identical to FROZEN |
| OJA | L1b's rule from `4f26c9c` (mu = 0.05, ungated) | the row L1b owes |
| RAND | same step, random direction, magnitude-matched | separates "any movement" from "Hebbian alignment" |

**Endpoint.** The **pre-refit gap** (A0), read at the start of every refit row, before `_refit` runs: accuracy on the
held-out split under a shadow re-solve on the current `W`, minus accuracy under the gate's stale reader.
- Calibrated per open solve (A0b), because a pruned solve's beta differs from the shadow even when `W` has not moved.
- Read on the rows the stale reader can answer (A0b's vocabulary term).
- Scoring-only: it never feeds back, and R2 asserts identical substrate hashes with the shadow on and off.

**Primaries, fixed sequence.**
- **P1:** `GAP_OJA - GAP_FROZEN > 0` (claimed effect 0.10).
- **P2** (read only if P1 is MET): non-inferiority of OJA against FROZEN under the re-solved reader, margin 0.10.
- **Secondaries:** RAND; the live delta reader; units and the gate-verdict sequence; the post-refit closed-row gap; a
  replication read of L1b's own contrast.

**Statistics.**
- A pilot on seeds 0-7 measures the SD.
- Amendment A1 then fixes the sequential design before any confirmatory seed: Lan-DeMets O'Brien-Fleming boundaries via
  `rpe_charge.seq_boundaries`, non-binding futility, and a cap checked by simulation.
- Confirmatory seeds come from block 17000-17039; the operator confirms the block is free.
- **If the cap exceeds 40**, the exact sign-flip FPR refuses, and the campaign does not run; the owner decides.

**Reds.** R1-R9, plus a positive control of the claimed cause run in-brain at Stage 0. If any fails, the instrument is
BLIND: no verdict.

**Dispositions and licensed sentences** are fixed in advance. Nothing is re-landed. D1 is not the null-scale 10x re-test
(that axis is units, at least 380). D1 is not E3 (`learning-rules-1019.md`).

### 3.2 Feasibility facts measured (A0)

The probe ran on seeds 96-99, outside the pilot and confirmatory seeds: FROZEN only, stdlib tier, no accuracy read.
- Grown units fire on 86-100% of percepts, so the Oja rule will act.
- 2-7 units against 4-10 categories.
- The growth gate closed on 1 of 4 seeds. That is why P1 moved to the pre-refit gap.
- `L - rank(beta)` was 0 on every closed row, so the literature's null-space split will mostly read NOT READ, a
  structural result at this width.

### 3.3 Risks to D1

- **Power.** L1b's paired SD is 0.283, which implies about 60 seeds for a 0.10 effect, past the exact statistic's
  limit of 40. The design bets that the within-substrate gap has a smaller SD; the pilot decides.
- **The engine class will be retired.** T111a retires `NeurogenesisSelfImprove` into `Grower`. The instrument binds
  through `brain.grower.percept.engine`, which forwards dynamically.
- **Tier.** This cloud has only the stdlib tier, because the e-prop encoders are gitignored. The campaign must run on
  the operator's tier, on **one host** (section 5.2).

### 3.4 D1 build: next commit (not started)

**Contents.**
- `instruments/readout_staleness.py`: shadow, calibration, the drift readings, instance wrapper with a fault counter,
  the OJA and RAND operators, the driver;
- brain-free reds R1-R5 and R8;
- `tests/test_readout_staleness.py`.

**Registration.**
- `REGIONS['instruments']` (`src/snn_data_lake/__init__.py:287`);
- `run_selfchecks.CORE` (near `:177`);
- `dormancy_gate.EXEMPT` `permanent`, following the `sleep_plasticity_ab` / `prior_drift_harness` precedent;
- `GRADIENT_CLAIM`: OJA and RAND LOCAL; the shadow NOT_A_RULE; the re-solve DIRECT_SOLVE;
- a `_CENSUS_ADJUDICATED` entry plus `rule_composer --regen-census`;
- **no** `EXEMPT_METHOD` (inert for an exempt module).

**Who.** It can be built and its brain-free reds run in this cloud. Stage 0 onward needs the operator's host.

## 4. The cloud environment

- **Interpreter.** CPython 3.14.6 free-threaded via `uv` (`Py_GIL_DISABLED=1`, `_is_gil_enabled()=False`), venv
  `~/sndl-venv` outside the repo.
- **Torch.** 2.14.1+cu130 from PyPI. `download.pytorch.org` (the CPU index) is refused by the environment's network
  policy. A second venv, `~/sndl-venv-t211`, has torch 2.11.0, the operator's version, and was used only to classify
  failures.
- **Pinned dependencies.** brian2 2.10.1 and mujoco 3.11.0 match the pins. Brian2 enables the GIL in its own process,
  as documented.
- **Hardware.** 4-core Intel Xeon (AVX-512), no GPU, no display.
- **Missing assets.** `models/eprop_*.pt` (gitignored), so perception runs on the stdlib tier.
- **Git.** A full clone, unshallowed (2,626 commits), with all remote branches and tags fetched.

## 5. Pre-build audit

The full report is in the session scratchpad as `SNDL-AUDIT-2026-10-27.md`. Summary:

### 5.1 Gates

- **All 16 battery gates `--assert` green:** dormancy, method_dormancy, region_charge, growth_coverage, manifest,
  determinism, replay, envelope_hash, policy_manifest, gradient_claim, eprop_transport, import_manifest,
  replication_guard, capability_bar, immortal_retention and conclusion_protocol.
- **7/7 tools gates green.**
- The envelope matches its approved hashes as-is.
- `null_scale_gate` was red on the D1 branch only, because of a quoted NULL-RECORD line (D1 erratum E1). Fixed in
  `2e946d8d`.

### 5.2 run_selfchecks and pytest

`run_selfchecks` reported 5 FAILED. pytest ran per file under a bound; 200 of 215 files passed on the first pass.
Every non-green item was classified:

| class | items |
|---|---|
| CPU contention, passes alone | `test_attention_schema`, `test_recurrent_pool`, `test_r02b_p6_rust`, `test_carried_state_driver` |
| Known hang (cold start §4 rule 9) | `test_binding_instrument` |
| Headless (no GL) | `test_arm_view::test_renderer_refuses_a_foreign_thread` aborts instead of skipping |
| Missing e-prop encoders | `test_growing_learner_restart`; `test_grower::test_brain_five_arm_matches_t0` (correct STDLIB refusal) |
| **Windows-only code** | `gates/docs_only_battery._is_python` (POSIX `basename` on a `C:\` path), whose self-check and test fail on Linux; `test_sleep_plasticity_ab::test_guard_out_refuses_a_junction_alias` and `test_sleep_pruner_census::test_out_target_...` shell out to `cmd` |
| **Host-dependent numerics** | `candidate_lifecycle`: self-check plus 14 byte-identity pins; `test_grower`: 8 byte-identity pins; `test_memory_diagnostic` rb6 (0.5498 vs 0.5519); `test_episode_durability`; self-checks `language_self_build`, `language_readout_slda`, `open_world_loop` |
| **Genuine defect** | `test_h01_partb_tangential`: commit `cbd8ce14` is unreachable (decision 1) |

**The host-dependent numerics in detail.** They fail identically under torch 2.14.1 and 2.11.0, and with
`ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2` and `MKL_CBWR=COMPATIBLE`. Unit counts match the pins; only
the float fingerprints differ. The likely cause is OS, compiler or libm (Windows vs Linux). **Unproven which.**

### 5.3 Policy compliance (static, each item verified by command)

**Prior audit (2026-10-24).**
- **Fixed:** both P4b BLOCKs (`language_acq.npz` side file, `nightly_skills`); restart amnesia for the three named
  stores; the envelope line-ending mismatch.
- **Still open:** T120, the second sidecar migration. As a result, test brains are not hermetic.
- **Undetermined:** "the unused consciousness answer", and the fourth prior MAJOR. The report is not in the tree.

**New findings.**

| id | class | finding |
|---|---|---|
| N2 | MAJOR, static only | Four live learners have no lake path: JEPA world model, `ValenceLearner`, `LearnedAttention`, `_plan_w` (decision 5) |
| N3 | MINOR | `substrate/lake_memory`: 0 importers, a false EXEMPT reason, and a potential second durable store; retire candidate |
| N4 | MINOR | 10 EXEMPT `permanent` modules with 0 importers and 0 tests ("library" is not a P3 exemption reason) |
| N5 | MINOR | Duplicate `_AdExCell` (`heterogeneous_microcircuit` vs `neuron_zoo`), P2 |
| N6 | MINOR | 23 non-ASCII print lines in 21 files, 2 of them envelope files (owner approval needed) |
| N7 | MINOR | Stale comments: `home.py` describes `lake.spkl` as durable but nothing writes it; `live_brain.py:872` "OFF by default" while the default is True |
| N8 | NOTE | `lake_index`'s `wire` exemption declares its own deadline (T039) expired |
| N9 | NOTE | 40 `reasoning/` modules outside `REGIONS` (known P4c debt); `reasoning/_loop_common.py` holds an unused `.backward()` + Adam trainer, retire candidate |

**Clean:** the e-prop invariant on live paths, the P7 ledger, determinism, P1 sample.

### 5.4 Documentation against the tree (24 verified)

The 13 claims that would make an agent act wrongly:

| document | stale claim | actual state |
|---|---|---|
| `CONTRIBUTING.md` | Python 3.11-3.13 | cp314t |
| `CONTRIBUTING.md` | "four gates" | many more gates |
| `CONTRIBUTING.md` | no Rust | a Rust toolchain is needed for `_native` |
| `AGENTS.md` | P6 "not built" | asserting in CI |
| `AGENTS.md` | `dependency_ledger` "report-only" | runs with `--assert` |
| `AGENTS.md` | `doc_command_guard` "NOT IN CI" | in `run_selfchecks` |
| `AGENTS.md` | the gate block | misses about 8 gates |
| `AGENTS.md` | "thirteen entries" | misses P5b |
| `AGENTS.md` | (silent) | never states that CI is off |
| `AGENT_COLD_START.md` §3 | envelope defect #1 open | fixed (`293b24f2`) |
| `AGENT_COLD_START.md` §10 | extend `hjepa_torch` | retired (`8538b877`) |
| `AGENT_COLD_START.md` §1 | "four research reservoirs" | three deleted |
| `POLICIES.md` | `doc_command_guard` location (:51 vs :64) | self-contradiction |
| `POLICIES.md` | P5b mechanized (:41 vs :734) | self-contradiction |
| `PLAN-FINAL:1634-1637`, `:1249`, G6/T114 | sleep relax and replay | retired 2026-10-13, "owner1039"; the arms now raise (`brain.py:3630`); no tracked `.md` records it |
| `null-scale-triage.md` row 2 | L1b cause "gone since L1c" | gone only for the live prediction |
| `tasks.md:240-243` | "all G rows OPEN" | five are done |

### 5.5 CI

The GitHub Actions workflow is `disabled_manually`. Its last run was 2026-08-20, and those four runs all failed. The
operator's local battery is the only verification.

## 6. Owner decisions 1-5

Owner instruction (2026-10-27, verbatim): *"1. The unreachable commit cbd8ce14: Ask active sessions on my computer
first and get an answer as well as suggestions from them ... For the rest I approve after you check with the active
sessions."*

| # | decision | recommendation | status |
|---|---|---|---|
| 1 | `cbd8ce14` is unreachable on origin, so `h01_partb_tangential` refuses on every fresh clone | The operator runs `git cat-file -p cbd8ce14:tools/g4_t3_campaign.json \| sha256sum`. If it equals `dc5e341d95a1352f852a1616dd50989c686b1e2ad681317b780607231a566c02` (main's copy), push a tag: `git push origin cbd8ce14:refs/tags/g4-t3-record` (no code or record edit; survives gc). If it differs, that is a provenance finding for G4. | **Awaiting the operator sessions' answer** |
| 2 | Cross-host verification | Fix the three Windows-only code paths now. Later the owner chooses: (a) declare Windows-host pins, skipped elsewhere with a reason; or (b) Linux pins plus CI re-enabled. | Approved pending consultation |
| 3 | The 2026-10-13 sleep retirement is undocumented | A dated plan amendment (sleep tick; G6/T114; R-workstream note) plus an AGENT_COLD_START §10 guard row. Needs the owner's verbatim 2026-10-13 ruling. | Approved pending consultation; ruling text requested |
| 4 | Docs sweep | One docs-only commit fixing the 13 MISLEADS-ACTION items, each with a dated correction note. POLICIES limited to its two factual self-contradictions. Written on a cloud branch for the operator to land with the docs-only battery. | Approved pending consultation |
| 5 | Four learners possibly without persistence | A cross-process restart probe on a scratch home (stdlib tier). Any fix is its own pre-registered arc. | Approved pending consultation |

## 7. Consultation with the operator sessions

| session | id | state when last checked |
|---|---|---|
| SNDL engine build integration | `session_01EYwnz1WFj6qhLzspqxtjXf` | idle; E1b look 3 continued, look 4 owed 32 runs |
| SNDL protocol improvement | `session_01AxHGSJTKf3nG9eJVBdsoCd` | idle; "battery 3 running ... will check green before main push" |

**Messages.**
- The first send (2026-10-27) returned `delivered`, but it does not appear in either transcript, so it was lost.
- The re-send at priority "now" reached both transcripts as new turns at **2026-10-27 23:35:48 UTC**.

**Replies.** None received as of writing. Both sessions were asked to reply to `session_01TeA2yYimjAiHiGFzw6hqHX`.

Each session was also asked to flag conflicts with unpushed work. Both report batteries or campaigns in flight, so
**`main` may move before any of decisions 2-5 is landed**, and the docs sweep in particular must be rebased onto their
push.

Earlier, an FYI about the drift-map commit was delivered to both sessions on 2026-10-27, after the owner approved
cross-session messaging on a trusted device.

## 8. Plan, in order

1. **Wait for both replies.** Relay them to the owner, decision 1 first. Act on nothing until both have answered.
2. **Decision 1.** The operator runs the hash check and pushes the tag or re-pins, per their answer.
3. **Decision 5.** The restart probe (cloud; about an hour). Record the result as a finding, then the owner decides on a
   fix arc.
4. **Decision 4.** The docs-sweep branch (cloud), rebased on the operator's latest push; the operator lands it with the
   docs-only battery.
5. **Decision 3.** The plan amendment and §10 row, once the owner's 2026-10-13 ruling text is in hand.
6. **Decision 2.** Fix the three Windows-only paths (small operator arc). The owner later chooses Windows pins versus
   Linux pins plus CI.
7. **The D1 build commit** (section 3.4). It is independent of 1-6 and can run in parallel once the owner says go.
8. **D1 Stage 0 onward** on the operator's host and tier:
   - Stage 0: positive control, cost, stream agreement;
   - pilot on seeds 0-7;
   - Amendment A1 (the design; feasibility at 40);
   - confirmatory looks;
   - verdict, panel and closing report.
9. **Registration.** The operator adds T121, the dated plan row and D-R1 in Part F.

## 9. Open questions for the owner

1. Your verbatim 2026-10-13 sleep-retirement ruling ("owner1039"), for decision 3.
2. Decision 2's long-term choice: Windows-host pins, or Linux pins plus CI?
3. If D1's pilot puts the cap above 40 seeds: a larger claimed effect, an approved approximate FPR, or stop?
4. Should the D1 build start now, in parallel with decisions 1-6?

## 10. Artifacts

**In the repo (this branch):**
- `research/representational-drift/MAP.md`;
- `specs/022-unified-brain-arc/evidence/d1-readout-staleness.md`;
- this file.

**Session scratchpad** (ephemeral; lost when the container is reclaimed):
- `SNDL-AUDIT-2026-10-27.md` (the full audit report);
- `battery/` (per-file pytest logs, `selfchecks.txt`, gate outputs);
- `rerun_summary.txt` and `t211.txt` (classification runs);
- `d1_feasibility_probe.py` and `probe99.log`, `probe96.log`, `probe97.log`, `probe98.log` (A0's probe);
- the audit agents' probe scripts: `probe_shadow.py`, `probe_r1.py`, `probe_overlap.py`, `probe_brain.py`, `chk.sh`.

**Earlier in the session:**
- the HW4 Word document, filled and then set aside at the owner's request;
- the representational-drift literature survey (summarized in MAP.md §3 and §9).
