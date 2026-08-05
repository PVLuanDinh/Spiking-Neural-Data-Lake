"""neuromorphic_torch -- torch/SpikingJelly subproject: SpikeSlicer, IS-SNN, LT-Gate, Max-Former,
and the GPU neuromorphic-chip emulators (neuro_chip_gpu, neuro_chip_gpu_stream).

torch is a core dependency; SpikingJelly is adopted for the four phase modules and installed in this
subproject's own venv so the deterministic gate env stays GPU-free (P5: CI has no card). The header
here read "quarantined ... not core deps, not CI-gated" until 2026-08-04 -- retired, because P7
removed the dependency budget that made "quarantined" mean something and P4c removed the category:
this directory is part of the brain and is NOT gated, which is DEBT, not a disposition.
"""
