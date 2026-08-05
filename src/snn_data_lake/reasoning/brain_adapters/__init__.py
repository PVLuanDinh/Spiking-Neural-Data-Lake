"""snn_data_lake.reasoning.brain_adapters -- adapters that let the developmental / NG-SNN
loop use the brain's REAL modules in place of hand-rolled scaffolding.

They compose the shipped `snn_data_lake.*` modules rather than re-implementing them (P4:
utilize, never replicate); `laminar_local_readout.LaminarLocalReadout` additionally runs on the
live wake tick (`cortex/brain.py`).

P4c: these are PART OF THE BRAIN, not "research code beside it". This docstring used to open
"research/brain_adapters" and assert "Zero-dep CORE is never edited from here" -- the path moved
into the package in v3.78 and the zero-dependency rule was abolished 2026-07-20 (P7), so both
clauses were stale. Open debt, stated because P4c makes it countable: only
`laminar_local_readout` has a live consumer; none of the eight modules here is registered in
`REGIONS` or `run_selfchecks.py`, so the three P3 gates cannot see them.
"""
