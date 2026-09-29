# SNDL: current-state research assessment

Assessment date: 2026-09-20 (America/Chicago). Implementation examined: local `0fd6db5`, package
version `4.80.0`. This report evaluates readiness for an academic study of affordable local
learning. It does not certify a deployment or rerun every historical experiment.

## Verdict

**SNDL is a substantial experimental platform with runnable components and useful evaluation
infrastructure. A publishable end-to-end edge-efficiency result is still to be produced.**

The strongest first-paper direction is a controlled study of adaptation and retention under memory,
latency, and energy budgets. The project already has relevant mechanisms and records informative
failures. Its breadth is useful infrastructure, but the paper needs a narrow task, strong baselines,
causal ablations, target-device measurements, and independent reproduction.

| Dimension | Assessment | Reason |
|---|---|---|
| Implementation | Substantial research prototype | Storage, learning, growth, control, runtime, and test infrastructure exist. |
| Local operation | Demonstrated for selected components | Demo, storage/query examples, scorer checks, and targeted tests ran locally. |
| Robotics | Simulation apparatus available | Gesture-to-command and safety-path demo passes; this is not proof of adaptive physical manipulation. |
| Local growth | Mechanisms and bounded task evidence exist | A corrected width experiment improves next-category accuracy over one unit, with substantial baseline/control limits; affordable adaptation remains the research target. |
| Compute/energy advantage | Unproven for the whole system | The energy example reverses its favorable headline under a fairer baseline. No new hardware joule measurement was made. |
| Affordable edge deployment | Unverified | Current runtime is a desktop-oriented Python stack; target-device compatibility, footprint, quality, and acquisition cost need measurement. |
| General reasoning | Weak on the recorded C0 items | 0/20 correct on each of five seeds; instrument success is distinct from capability. |
| Paper readiness | Ready for a supervised proposal; not an established efficiency paper | Scope, protocol, baselines, statistical campaign, and device replication remain. |

## How this assessment was made

The existing graph and report were consulted first. Their recorded commit `0fd6db58` matches the
examined source. The graph was used for navigation; claims were checked against executable
configuration, implementation, dated experiment records, and selected runtime checks. The graph
CLI launcher failed locally, so graph JSON traversal was used as the fallback.

Claude context was used as requested: the project `CLAUDE.md`, the curated memory index under the
parent `D--AI-project` Claude workspace, the September 17 handoff, and notes on prior negative
results and self-building. The repo-specific Claude workspace had no memory directory; the parent
index explicitly documents that location mismatch. A Claude OS knowledge-base search returned
unrelated material and was not used as project evidence.

Private memory supplied orientation and unpublished-status context, not scientific results. Its
claims were checked against the tree where possible. Personal memory files are not copied into
this repository and are not required to reproduce the public examples. Historical memory commands
were not treated as a request to resume the earlier development campaign.

## Findings that change the presentation

### 1. The old README describes a superseded runtime

[pyproject.toml](../pyproject.toml) declares Python `>=3.14` and five core dependencies: `torch`,
`numpy`, `pyarrow`, `mujoco==3.11.0`, and `brian2==2.10.1`. The
[package guard](../src/snn_data_lake/__init__.py) checks free-threading, and the
[daemon](../src/snn_data_lake/substrate/learn_daemon.py) makes that requirement strict at boot.
[CI](../.github/workflows/ci.yml) uses Python `3.14t`.

The previous README still said Python 3.12+, three core dependencies, and an older CI matrix.
Some supporting documents and local Claude summaries also contain superseded installation or
torch-free statements. The rewrite follows current configuration. A wider historical-document
cleanup remains separate work; those older documents should not be treated as installation
authority without checking them.

### 2. The storage result is real within a defined workload

The [telemetry example](../src/snn_data_lake/substrate/spike_telemetry_hub.py) generated 103,723
events across 256 channels and 100,000 time steps. Its sparse file was 417,976 B. The same raster
dimensions require 3,200,000 B at one bit per cell or 25,600,000 B at one byte per cell: about 7.7x
or 61x larger, respectively. The specified windowed query read 8,492 B, about 2.0% of the sparse file.

Use both dense representations when presenting this result. It demonstrates storage reduction and
partial I/O for this event distribution. It does not compare against every compressed time-series
format or prove that the whole learning system fits in the same number of bytes.

The [temporal query example](../src/snn_data_lake/substrate/paradigm_b_engine.py) also passed: it found
41 forward-order matches versus four reverse-order matches, with 40 versus three in the motif
window, and read 1.3% of the file for the sequence query. This is a constructed order-sensitive
workload; it is not a learned robotics benchmark.

### 3. The existing efficiency headline is baseline-sensitive

The [energy module](../src/snn_data_lake/instruments/energy.py) explicitly reports both comparisons:

| Comparison | Counts or modeled energy | Interpretation |
|---|---|---|
| SNN vs dense computation repeated for all 20 steps | 113,676 SynOps vs 409,600 MACs | The source of the older 3.6x operation reduction. |
| SNN vs one dense pass on the static input | 113,676 SynOps vs 20,480 MACs | About 5.6x more counted operations for the SNN. Operations are not identical in cost. |
| Modeled SNN including memory vs modeled INT8 ANN | 740.4 nJ vs 107.1 nJ | The analytical comparison favors the ANN under these assumptions. |

These figures were rerun and are already disclosed by the implementation. They are not wall-plug
energy measurements. Device costs, sparsity, encoding, memory traffic, and repeated simulation
steps must be measured together before claiming that an inexpensive local machine benefits.

The research opportunity is to locate the crossover, or find a better method, with a fair task
comparison. A favorable theoretical operation count is insufficient.

### 4. Persistent storage and successful neuron paging are different claims

The [S5 record](../specs/022-unified-brain-arc/evidence/s5-paging-null.md) and
[S2 exit assessment](../specs/022-unified-brain-arc/evidence/s2-exit.md) report a memory benefit but
a median raw paged/resident tick ratio of 35.2x (reference-corrected 26.0x) against a 3.0x bound.
The layer was deleted under its predefined failure rule. This was a historical experiment reviewed
here, not rerun in this assessment.

Similarly, the [S6 record](../specs/022-unified-brain-arc/evidence/s6-kernel-wire.md) records declining
the sparse-kernel live-tick conversion at the deployed width after measuring its limited available
speedup. A sparse-kernel implementation or a possible accelerator is not evidence that the deployed
brain receives that acceleration.

Do not present disk-backed experience as evidence that arbitrarily many active neurons can execute
cheaply beyond RAM. Any renewed paging proposal should address the prior cost explicitly.

### 5. Growth exists, but useful affordable growth is still a hypothesis

Relevant implementation includes [neurogenesis](../src/snn_data_lake/growth/neurogenesis_engine.py),
[growth-program locality](../src/snn_data_lake/growth/growth_program.py),
[growth stopping](../src/snn_data_lake/growth/growth_stop.py), and
[function-preserving widening](../src/snn_data_lake/growth/function_preserving_growth.py).
The gate examines learning/accuracy trajectories; function preservation establishes continuity at
widening, not future learning improvement or energy efficiency.

The selected regression tests exercise growth gating, state persistence/restart, and RNG behavior.
A paper still needs to demonstrate the selected controller actually uses the added capacity,
improves on unseen conditions, and retains previous skills after paying its resource costs.

There is also positive task evidence: the corrected
[SC-012a reading and its subsequent review](../specs/022-unified-brain-arc/evidence/sc-012a-width.md)
report mean next-category accuracy of 0.8090 for CAP16 versus 0.6583 for CAP1 across eight seeds
(paired difference +0.1507, reported p=0.0234, six wins and two losses). CAP16 actually realized
9-14 units. This historical result is narrower than a general growth claim: forced width did not
beat the incumbent's own 2-7 units, no width beat the previous-category predictor, and no individual
doubling cleared the criterion. Section 8 further discloses that the shuffled-label veto had no
reachable failure on the corrected instrument, so its passing outcome supplies no discriminating
corroboration. Present the corrected result with that review, not the retracted first reading.
It motivates further controlled experiments without establishing the proposed robotics or
resource-efficiency result.

Source self-modification in `Brain._maybe_autonomous_self_build` is separately opt-in through
`NORD_SELFMOD=1`. The current development task list still contains open proposer, evaluator,
curriculum, and exit-certification work. Do not equate an opt-in mechanism with a completed
autonomous self-improvement system.

The [task list](../specs/022-unified-brain-arc/tasks.md) also explicitly labels the Navier-Stokes
plus liquid-state-machine workstream (T096-T100) as planned and unbuilt. It should be presented
as a possible future investigation, not a current computational-efficiency result or a prerequisite
for the first bounded edge-learning study.

### 6. C0 measures the instrument faithfully, not successful everyday reasoning

[The C0 pin](../tools/c0_everyday_battery.json) names campaign source commit `37135d7`, seeds 0-4,
and its protocol. Each seed has 14 `ABSTAIN`, five `UNRELATED_ROW`, one `SELF_REPORT`, and zero
`CORRECT` among the 20 scored items. Positive controls succeed separately. The
[record](../specs/022-unified-brain-arc/evidence/c0-everyday-battery.md) explicitly scopes `MET` to
the instrument's discriminating key, controls, port fidelity, and concurrency check.

The current checker re-derived 144 stored classifications with no disagreement. This verifies
internal consistency of the recorded data, not a fresh five-seed behavioral campaign. The 20
questions are also not a representative sample permitting a universal reasoning conclusion.

### 7. Release and test status require separate reporting

At assessment start the working tree was clean and local `main` was **26 commits ahead of the
upstream tracking branch**. Claude's September 17 handoff reported that the T090/C0 stack had
not completed its combined release battery and the C0 review was interrupted. Git independently
confirmed the unpublished count. The user subsequently authorized publishing the current project
and the new documentation together; publication validation is recorded below.

The GitHub API returned `state: disabled_manually` for `ci.yml` on 2026-09-20. The most recent
listed run was August 20, not the assessed September source. Its self-check and test jobs actually
executed and failed; it is not a current validation result. This update does not re-enable the
workflow or imply that a new push will execute CI.

The configured CI has substantial gates, but its policy-certification job is advisory and its
performance report does not assert on shared runners. Pytest's 70% coverage floor excludes the
reasoning and training tiers. Neither a green workflow nor a count of modules establishes research
novelty or efficacy.

## Fresh verification in this documentation task

The existing local `.venv-lake` reported CPython 3.14.6 free-threaded, PyTorch 2.11.0+cu128,
NumPy 2.5.1, PyArrow 25.0.1, MuJoCo 3.11.0, and Brian2 2.10.1 after the framework migration;
CUDA was available. This is
environment provenance, not an inexpensive-device result or a fresh clean installation.

| Check | Outcome and scope |
|---|---|
| Simulation demo | Passed: HOLD, ENGAGE, RELEASE, contact-triggered EMERGENCY_STOP. |
| Telemetry storage example | Passed, with the byte and query figures reported above. |
| Temporal query engine | Passed forward/reverse and partial-read checks. |
| Energy example | Passed its analytical checks, including disclosure of the baseline reversal. |
| C0 bare self-check and pin checker | Passed; 144 pinned rows re-derived with zero disagreements. No new campaign run. |
| Targeted pytest | 39 passed in 49.91 s across the everyday-battery, growth-stop wiring, growth restart, and hardware-independent growth RNG test files; coverage disabled for this targeted diagnostic. |
| Full self-check runner | 360/360 module self-checks and 6/6 tools gates passed. The optional Nengo bridge was separately skipped because its extra is not installed. |
| Full pytest/coverage suite | Passed: 1,412 passed, 2 expected failures, 19 warnings in 4,190.14 s (1:09:50). Coverage was 88.52%, above the configured 70% floor and subject to its exclusions. |
| Ruff, citations, dependency ledger, reachable-red check, documentation commands | Passed. Ruff format checked 591 files; citation guard had zero unexempted strict violations; dependency ledger had zero undeclared, unused, or misdescribed dependencies; reachable-red checked 19 criteria; documentation guard found 61 addressable commands and zero unrunnable ones. All 59 local documentation links resolved. |

The full suite's warnings included invalid escape sequences and overflow/invalid arithmetic in
[knowledge-base ranking](../src/snn_data_lake/cortex/kb_ranking.py) during the schema-induction
self-check. These remain numerical-robustness and maintenance follow-ups; passing assertions do
not establish that all intermediate values were finite. They were not suppressed or repaired in
this documentation update.

One non-isolated `perf_baseline --report` run flagged its ambient-normalized wake median as a
regression (213.860 versus historical 183.517) and its tail as high. Other assessment work overlapped
that run. It is a diagnostic observation, not an isolated performance experiment, and supplies no
new speedup claim. A controlled target-device campaign remains necessary.

The first targeted pytest attempt could not create its default temporary/cache directories in the
restricted environment. The successful run used a fresh workspace temporary directory and disabled
the cache provider. The full suite used separate workspace temporary, cache, and coverage paths
with the configured coverage gate enabled. No product code was changed to obtain the pass.
The daemon was not started,
the user's persistent brain was not used as an experiment subject, and no physical arm was actuated.

## Next research decisions

1. Agree with the professor on a bounded continual-control task and practical quality floor.
2. Freeze hardware and resource boundaries, including the runtime's fixed overhead.
3. Build the compact-dense/classical baselines before proposing more architecture.
4. Measure the incremental benefit of growth and persistent experience with controls.
5. Publish the complete measured tradeoff, including negative regimes and uncertainty.

The [academic research plan](ACADEMIC_RESEARCH_PLAN.md) gives the hypotheses, experiment matrix,
measurement protocol, presentation outline, and eight-week deliverables.
