"""spatiotemporal_probe -- tools/charge_probe.py's verdict logic, run on the harness the DAEMON uses.

WHY THIS EXISTS, and it is a defect in the harness rather than a second opinion about the brain.

`tools/charge_probe.py:39` puts exactly ONE directory on sys.path: `<root>/src`. The repo ROOT is not
added, and `coverage_score.py record` pins `PYTHONPATH=<root>/src` as well, so a recorded probe run
cannot see the top-level `training/` package either. `cortex/brain.py:562` imports
`training.eprop_nmnist` inside the guarded perception block, so under that harness the import raises,
the handler at `:606-609` sets `_perception_tier="stdlib"` and `_perceiving_real=False`, and
`_perceived_digit` is None on every tick -- even though `models/eprop_nmnist.pt` and
`models/nmnist_pack.npz` are both present on disk.

That single missing sys.path entry silently switches OFF the precondition of the whole W6 spatial
binding (`brain.py:1297` requires `_perceived_digit is not None`). MEASURED, same seed, 60 ticks:

    PYTHONPATH=src            -> perception_tier stdlib,    place_here  0/60, _place_value 0 pairs
    PYTHONPATH=.;src          -> perception_tier eprop-snn, place_here 57/60, _place_value 7 pairs

So `docs/coverage/artifacts/place_60.txt`'s NOT-ELIGIBLE (exit 2, "the source never activated") is a
property of the PROBE, not of the brain. `learn_daemon.py` lives at the repo root, so the always-on
daemon -- the live path the coverage question is actually about -- runs with the root importable and
perceives real N-MNIST frames. Scoring place coding from the src-only harness would have recorded a
rule-4 zero against a faculty that fires 57 ticks out of 60 in production.

WHAT THIS DOES NOT DO. It does not re-implement the neuter or the verdict. It inserts the two paths
and then calls `charge_probe.main()`, so `_PinnedWeights`, the eligibility test, the observables
(`arm_stopped`, `map_position`) and the exit-code contract are all the tracked tool's, unchanged. The
real fix is one line in `tools/charge_probe.py`; that file was being edited by other agents during
this audit, so the correction is quarantined here and flagged in `docs/coverage/SpatialTemporal.md`
instead of applied.

  python docs/coverage/artifacts/spatiotemporal_probe.py <source> <ticks>
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # docs/coverage/artifacts -> root

# the repo ROOT first, so the top-level `training/` package resolves and the e-prop perception tier
# loads exactly as it does under `python learn_daemon.py`; then src; then tools/ for the import below.
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)  # brain.py's model gates are relative paths (os.path.exists("models/..."))

import charge_probe  # noqa: E402  the tracked instrument -- neuter, eligibility and verdict


def main():
    from snn_data_lake.cortex.brain import make_brain

    b = make_brain(seed=0, persist=False)
    tier = getattr(b, "_perception_tier", "?")
    real = bool(getattr(b, "_perceiving_real", False))
    b.close()
    print(f"harness perception tier                       : {tier} (perceiving_real={real})")
    if not real:
        print(
            "HARNESS NOT READY: real perception is off, so this run would reproduce the src-only "
            "result rather than the daemon's -- refusing to report a verdict"
        )
        return 2
    return charge_probe.main()


if __name__ == "__main__":
    sys.exit(main())
