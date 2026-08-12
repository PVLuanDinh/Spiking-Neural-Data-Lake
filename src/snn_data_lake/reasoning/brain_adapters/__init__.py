"""snn_data_lake.reasoning.brain_adapters -- adapters that let the developmental / NG-SNN
loop use the brain's REAL modules in place of hand-rolled scaffolding.

They compose the shipped `snn_data_lake.*` modules rather than re-implementing them (P4:
utilize, never replicate); `laminar_local_readout.LaminarLocalReadout` additionally runs on the
live wake tick (`cortex/brain.py`).

P4c: these are PART OF THE BRAIN, not "research code beside it". This docstring used to open
"research/brain_adapters" and assert "Zero-dep CORE is never edited from here" -- the path moved
into the package in v3.78 and the zero-dependency rule was abolished 2026-07-20 (P7), so both
clauses were stale. The census here has been DRAINING by promotion/retirement (#10 lanes):
brain_decision -> basal_ganglia/ (2026-08-12), hippocampal_memory + growable_readout ->
hippocampus/ (2026-08-11), gwt_deep -> thalamus/ (2026-08-11), lake_memory -> hippocampus/
proximity_replay (2026-08-11), laminar_readout DELETED (2026-08-12 -- the LaminarReadout-vs-
laminar_local_readout P2 this docstring long owed is RESOLVED: the no-transport local readout
is the one live-tick mechanism, LaminarColumns ported to hippocampus/growable_readout, the
backprop L6 readout retired as the recorded v3.59 failure). laminar_local_readout -> cortex/ (2026-08-12,
runtime-proven live on the wake tick at the move). ONE module remains:
`executive_control` (p6_gate-pinned, its own lane -- carrying the recorded P2 vs
brain_decision.choose(), see that module's docstring, plus its owed rename for the
basename collision with cortex/executive_control).
"""
