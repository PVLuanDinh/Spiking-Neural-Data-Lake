"""gates -- every policy and correctness gate the battery or CI runs with --assert (specs/022 G8).

G8's engine-function package for gates (evidence g8-reorganization.md 2.2). It opens at S1a with
`move_gate`, G8's own G-R8 instrument; the 23 substrate gates join it at M1.
"""

# distinct-from: gates/envelope_hash_gate.py -- a docstring-only package marker, like every
#   other subpackage __init__; the hook matched the word "gate". Also read: gates/policy_manifest.py
#   (certification_fresh), gates/lake_rebuild_gate.py, growth/h01_ingest.py (_selfcheck),
#   instruments/prior_drift_harness.py (arm_no_edges) -- none is a package or a second mechanism.
