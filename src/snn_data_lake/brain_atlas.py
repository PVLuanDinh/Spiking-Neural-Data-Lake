"""Anatomical atlas of the spiking-neural data lake -- which brain region every module lives in.

As of 3.0 the package is organized by brain anatomy (see snn_data_lake.REGIONS). This module
renders that map and VERIFIES the pre-3.0 back-compat layer: every legacy flat import path
(`snn_data_lake.<module>`) still resolves to the IDENTICAL module object as its new regional
path (`snn_data_lake.<region>.<module>`), so old code keeps working after the move.

  python -m snn_data_lake.brain_atlas        # (also run by run_selfchecks.py)
"""
import importlib

from snn_data_lake import _MOVED, REGIONS


def region_of(module):
    """Return the brain region a module belongs to, or None if unknown."""
    for region, mods in REGIONS.items():
        if module in mods:
            return region
    return None


def atlas():
    """Return the region -> modules map (the authoritative anatomical layout)."""
    return REGIONS


def main():
    total = sum(len(mods) for mods in REGIONS.values())
    print("=" * 66)
    print("BRAIN ATLAS -- snn_data_lake organized by anatomical region")
    print("=" * 66)
    for region in REGIONS:
        mods = REGIONS[region]
        print(f"{region:<16} {len(mods):>2} modules")
    print("-" * 66)
    print(f"{'TOTAL':<16} {total:>2} modules across {len(REGIONS)} regions")
    print("=" * 66)

    # ---- self-check: layout sane + back-compat intact (zero-dep) ----
    assert total == 126, f"expected 126 modules in the atlas, got {total}"
    # regions are disjoint (no module in two regions)
    seen = {}
    for region, mods in REGIONS.items():
        for m in mods:
            assert m not in seen, f"module {m} listed in both {seen[m]} and {region}"
            seen[m] = region

    # every legacy flat path resolves to the SAME object as the new regional path
    for flat, dotted in _MOVED.items():
        new = importlib.import_module(f"snn_data_lake.{dotted}")
        old = importlib.import_module(f"snn_data_lake.{flat}")
        assert old is new, f"back-compat broken for {flat!r}: alias is not the regional module"

    # the hippocampus name collision (module == region): both paths import + the package re-exports
    pkg = importlib.import_module("snn_data_lake.hippocampus")
    mod = importlib.import_module("snn_data_lake.hippocampus.hippocampus")
    assert any(getattr(pkg, n, None) is getattr(mod, n) for n in dir(mod) if not n.startswith("_")), \
        "hippocampus region package must re-export its same-named module"

    # every region subpackage imports cleanly
    for region in REGIONS:
        importlib.import_module(f"snn_data_lake.{region}")

    print(f"self-check OK: {total} modules mapped to {len(REGIONS)} regions; "
          f"all {len(_MOVED)} legacy import paths still resolve to the moved module")


if __name__ == "__main__":
    main()
