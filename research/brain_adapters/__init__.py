"""research/brain_adapters -- thin research-tier wrappers that let the developmental /
NG-SNN training loop use the brain's REAL modules in place of hand-rolled scaffolding.

Ruff-excluded research code. Zero-dep CORE is never edited from here; these adapters only
import and orchestrate the shipped `snn_data_lake.*` modules.
"""
