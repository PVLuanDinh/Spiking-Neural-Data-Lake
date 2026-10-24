# P5b RETROACTIVE RECORD: the eleven native modules that landed before P5b's gate (2026-10-24)

**This is a retroactive record, not a pre-registration.** It was written on 2026-10-24 by session B (board
`rust-lead-1024`, thread RUST-PORT), five days after the code it describes landed. Nothing below was
pre-registered under P5b's six criteria; it collects what WAS measured at the time, with its source, and states
plainly where a criterion was never tested. `gates/p5b_gate.py` keys its closed `GRANDFATHERED` table to this
file: one entry per `.rs` file present under `native/` at origin/main `20e1e742f6d1d9deb19dee0b443e6737592a1200`.
A new native module cannot be grandfathered; it needs a real pre-registration cited from its own header
(`//! P5b-PREREG: <repo path>`).

## 1. What landed, and when

| when | commit | what | P5b status at the time |
|---|---|---|---|
| 2026-10-17 | `623b4a14` (R0.4) | `sndl-core/src/rng/*`: the RNG parity kit (MT19937 + CPython `random`, SeedSequence + PCG64, legacy RandomState, `_mathutils.Rng` LCG, fixture-parity harness) | P5b not yet written |
| 2026-10-17 | `c612c244` | `sndl-python/src/lib.rs`: the one PyO3 binding crate (`snn_data_lake._native`) | P5b not yet written |
| 2026-10-17 | `8f37196f` | P5b text (Python by default; six criteria), PROVISIONAL, review-enforced | landed with batch B1 (`d501c7aa`) |
| 2026-10-18 | `eaaa2a3e` and the pfc-p2 stack | `sndl-core/src/prefrontal_executive.rs`: Phase 2, `PrefrontalExecutive._decide_spiking` in Rust, Python body deleted | built while P5b was being written (lane pfc-p2 spawned 2026-10-17 23:45, P5b drafted 2026-10-18 00:05); landed with batch B2 (`5978bbcc`) AFTER P5b was in POLICIES.md, with no six-criterion record |

So the PFC port is the one module that landed under P5b's text without its record. That is a review miss in
batch B2 and is disclosed here, not repaired by back-dating.

## 2. The PFC port against the six criteria (what was measured, and where)

| criterion | verdict | evidence (number, source) |
|---|---|---|
| 1 hot | MET | Phase 0 neuter of `_decide_spiking`: wake mean 113.0 -> 78.6 ms, **-34.5 ms (30.5%)**, n=3, baseline band 112.45-113.38 ms (1%), 25 calls per wake tick (`D:\AI-project\sndl-rust-port-phase0-report-2026-10-17.md` section 1). After the port, input-paired on a fresh build: wake 71.3 ms vs B1 102.5 ms, **-31.2 ms/tick (-30%)** (campaign log, B2 evidence `rows/codex-rv/b2-native-evidence.log`) |
| 2 Python-bound | MET (profiler evidence) | a nested pure-Python loop (t_max up to 200 steps x 13 options x neurons per pool) drawing `rng.random()` per neuron per step, plus `_argmax` at **4.8M lambda calls per 102 ticks** (cProfile 2026-10-17, plan section 3 Phase 2). No numpy/torch inside the loop |
| 3 coarse boundary | MET | ~**1.38 ms per call** in Python (34.5 ms / 25 calls); one `_native.pfc_decide` call runs the whole race (state in, `(winner, rt)` + advanced MT19937 state out) |
| 4 charged | MET | the race kernel itself: lane rust-pfc-charge (#3847, 2026-10-17) neutered `_decide_spiking` (replay-cached and constant stubs): the selected action flips on **130-161 / 220** ticks, lake sha DIFF 3/3 seeds, repeat floor 0/220. (PFC-NEUTER-1023, #4392, neutered the PFC PLANNER parts, not this kernel: top-down bias 106-169, look-ahead 0-234 (7/8 seeds; seed 0 dormant), multi-step search 69-165, conflict control 96-185 wake-tick flips per seed; behaviour charged, own objective not shown, underpowered) |
| 5 Python fix first | **NOT TRIED** | no vectorizing, batching, caching or algorithmic Python change was measured before the port (the plan went from the profile straight to Rust; the cheapest untried candidate is the `_argmax` lambda, 4.8M calls). **0** measurements exist. Open item: a retroactive microbench (the deleted Python body from git, with the cheapest fixes, against the Rust per-call cost on the fixture's 5 full cases) settles whether a Python fix could have met Phase 2's own bar (25 calls <= 3 ms, i.e. <= 0.12 ms per call, ~11x below the Python body) |
| 6 parity oracle | MET | in the repo: `native/sndl-core/tests/fixtures_pfc.json`, recorded from the Python body at `c612c244` (lane pfc-p2 `gen_fixture.py`): **460** calls in 3 chains (300 / 60 / 100; 23 zero-evidence, 206 timeouts) with `(winner, rt)` and the rng state, plus 5 full cases (all on the zero-evidence, no-draw path). Outside the repo: Phase 0's `lanes\p0-measure-1017\fixtures\cnt220.pfc_fixture.json`, all **2,725** calls of a 220-tick run. Constructed red `EXTRA_DRAW` (one extra draw) changed the post-state in **20/20** calls and `(winner, rt)` in 0/20 (Phase 0 section 4), so the post-state comparison is mandatory |

## 3. Per file

| `native/` path | role | record |
|---|---|---|
| `sndl-core/src/prefrontal_executive.rs` | the PFC kernel | section 2 |
| `sndl-core/src/rng/pyrandom.rs`, `sndl-core/src/rng/mt19937.rs` | the CPython MT19937 stream the kernel draws from | section 2 (support of the one port; no criterion of their own) |
| `sndl-core/src/rng/mod.rs`, `sndl-core/src/lib.rs` | module and crate roots (shared helpers, the test-only mutant switch) | section 2 |
| `sndl-python/src/lib.rs` | the binding crate (`pfc_decide`, plus the test-only `panic_probe`) | section 2 |
| `sndl-core/src/rng/lcg.rs`, `legacy.rs`, `pcg64.rs`, `seedseq.rs`, `parity.rs` | R0.4 kit sections no live code uses | **no P5b record possible**: criteria 1-5 do not apply to code nothing runs. `native_dormancy.NATIVE_EXEMPT` keeps them under the reason "retire when that port lands"; the port landed in `5978bbcc` (2026-10-19). Their disposition is P3's (delete, or a new reason), not P5b's, and is put to review separately |
