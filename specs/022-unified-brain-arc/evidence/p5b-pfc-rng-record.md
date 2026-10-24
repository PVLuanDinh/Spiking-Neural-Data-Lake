# P5b record: the PFC port's two RNG support modules after the R0.4 kit disposal (2026-10-24)

Written 2026-10-24 by session B (board `rust-lead-1024`, thread RUST-PORT), when the arc-2 panel's MAJOR-4
(rv-arc2-b, #5196) disposal changed the bytes of two grandfathered modules: `rng/mod.rs` lost the four dead kit
modules' declarations and the three mutant bits only PCG64 read, and `rng/parity.rs` lost the NumPy, legacy and
LCG families. `gates/p5b_gate.py` reds a grandfathered module whose content differs from its cutoff blob
(R-GRANDFATHER-DRIFT), so each now cites this record and leaves the GRANDFATHERED table in the same commit.

**This is a record, not a pre-registration in time.** No new measurement was taken for it. The two modules serve
one port, the PFC decision race (`sndl-core/src/prefrontal_executive.rs`, landed with batch B2 `5978bbcc` on
2026-10-19), and the six criteria below are that port's, copied with their sources from
`p5b-retroactive-record.md` section 2 (this folder). Why these two files carry the port's criteria:

- `rng/mod.rs` holds the shared seed helpers (`words_from_decimal`, `words_from_be_bytes`, `crc32`) and the
  test-only mutant switch that the kernel's PyRandom stream and the kernel's own P8 red (`EXTRA_DRAW`) use.
- `rng/parity.rs` compares that PyRandom stream, cell by cell, with real CPython output (the `stdlib`,
  `stdlib_str_seed` and `cortex_rng` families of `native/sndl-core/tests/fixtures_rng.json`): the RNG half of
  criterion 6. The fixture's other families stay in the file, listed as not compared.

After the disposal neither module serves anything but the PFC port. If the owner's question #5007 (keep, owe or
revert the PFC port) is answered "revert", this record falls with the port.

Criterion 5 is MET only after the fact. No Python fix was tried before the port; one was measured on 2026-10-24
and fell short of the bar (the retroactive record says so in full). The verdict word below is the gate's; the
order of events is as stated here.

P5b-1 hot: MET -- Phase 0 neuter of `_decide_spiking`: wake mean 113.0 -> 78.6 ms, -34.5 ms (30.5%), n=3
P5b-2 python-bound: MET -- a nested pure-Python race loop drawing `rng.random()` per neuron per step, with `_argmax` at 4.8M lambda calls per 102 ticks (cProfile 2026-10-17)
P5b-3 coarse-boundary: MET -- about 1.38 ms per call in Python (34.5 ms over 25 calls per wake tick); one `_native.pfc_decide` call runs the whole race
P5b-4 charged: MET -- neutering `_decide_spiking` flips the selected action on 130-161 of 220 ticks, lake sha DIFF on 3 of 3 seeds (#3847)
P5b-5 python-fix-first: MET -- measured after the port, not before it: the cheapest Python fix runs 703 us per call against the 120 us bar (5.9x short); Rust runs 59 us (`p5b-pfc-retro-bench.md`)
P5b-6 parity-oracle: MET -- `native/sndl-core/tests/fixtures_pfc.json`, 460 calls in 3 chains with the rng state after each, plus the PyRandom families of `fixtures_rng.json` (CPython 3.14.6t) compared by `rng/parity.rs`, each with a planted red (`EXTRA_DRAW`, `SWAP53`, `SINGLE_WORD_SEED`)
P5b-MODULES: sndl-core/src/rng/mod.rs, sndl-core/src/rng/parity.rs
