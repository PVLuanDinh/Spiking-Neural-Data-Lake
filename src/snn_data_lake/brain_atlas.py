"""Anatomical atlas of the spiking-neural data lake -- which brain region every module lives in.

As of 3.0 the package is organized by brain anatomy (see snn_data_lake.REGIONS). This module
renders that map and VERIFIES the pre-3.0 back-compat layer: every legacy flat import path
(`snn_data_lake.<module>`) still resolves to the IDENTICAL module object as its new regional
path (`snn_data_lake.<region>.<module>`), so old code keeps working after the move.

  python -m snn_data_lake.brain_atlas        # (also run by run_selfchecks.py)
"""

import importlib
import os

from snn_data_lake import _MOVED, REGIONS, TORCH_TIER, TRAIN_TIER


# distinct-from: brain_geometry.py -- not a new region resolver; this module's OWN region_of
# gaining the TRAIN_TIER annex (2026-08-11), same shape as when TORCH_TIER was added.
def region_of(module):
    """Return the brain region a module belongs to, or None if unknown (checks the core REGIONS map
    and the TORCH_TIER / TRAIN_TIER annexes)."""
    for tier in (REGIONS, TORCH_TIER, TRAIN_TIER):
        for region, mods in tier.items():
            if module in mods:
                return region
    return None


def atlas():
    """Return the region -> modules map (the authoritative anatomical layout: REGIONS core tier).
    The torch-tier annex is snn_data_lake.TORCH_TIER."""
    return REGIONS


def _region_files(region):
    """The set of module names (files) actually present in a region subpackage (excludes __init__)."""
    pkg = importlib.import_module(f"snn_data_lake.{region}")
    d = os.path.dirname(pkg.__file__)
    return {f[:-3] for f in os.listdir(d) if f.endswith(".py") and f != "__init__.py"}


def main():
    region_total = sum(len(m) for m in REGIONS.values())
    torch_total = sum(len(m) for m in TORCH_TIER.values())
    train_total = sum(len(m) for m in TRAIN_TIER.values())
    total = region_total + torch_total + train_total
    print("=" * 66)
    print("BRAIN ATLAS -- snn_data_lake organized by anatomical region")
    print("=" * 66)
    for region in REGIONS:
        extra = len(TORCH_TIER.get(region, []))
        tag = f" (+{extra} torch-tier)" if extra else ""
        print(f"{region:<16} {len(REGIONS[region]):>2} modules{tag}")
    print("-" * 66)
    print(
        f"{'TOTAL':<16} {total:>2} modules across {len(REGIONS)} regions "
        f"({region_total} core + {torch_total} torch-tier)"
    )
    print("=" * 66)

    # ---- self-check: the map EXACTLY covers the filesystem, so it cannot silently drift ----
    # every .py module on disk in a region is mapped (REGIONS or TORCH_TIER), and every mapped entry has
    # a file. This replaces the old hard-coded count assert, which masked unmapped modules.
    for region in REGIONS:
        files = _region_files(region)
        mapped = (
            set(REGIONS[region]) | set(TORCH_TIER.get(region, [])) | set(TRAIN_TIER.get(region, []))
        )
        unmapped = files - mapped
        phantom = mapped - files
        assert not unmapped, (
            f"{region}: modules on disk but not in the atlas map: {sorted(unmapped)}"
        )
        assert not phantom, f"{region}: atlas entries with no module file: {sorted(phantom)}"

    # regions are disjoint (no module in two regions, nor in both REGIONS and the torch annex)
    seen = {}
    for tier in (REGIONS, TORCH_TIER, TRAIN_TIER):
        for region, mods in tier.items():
            for m in mods:
                assert m not in seen, f"module {m} listed in both {seen[m]} and {region}"
                seen[m] = region

    # every legacy flat path resolves to the SAME object as the new regional path
    for flat, dotted in _MOVED.items():
        new = importlib.import_module(f"snn_data_lake.{dotted}")
        old = importlib.import_module(f"snn_data_lake.{flat}")
        assert old is new, f"back-compat broken for {flat!r}: alias is not the regional module"

    # (the old hippocampus name collision -- module == region -- is gone: that module was retired in the
    # Policy-3 wave, so no region subpackage re-exports a same-named module any more.)

    # every region subpackage imports cleanly
    for region in REGIONS:
        importlib.import_module(f"snn_data_lake.{region}")

    # torch-tier annex: verify each module region-resolves + imports WHEN torch is present;
    # SKIP-with-notice otherwise. torch is a core dependency so the skip is now a genuine
    # broken-install path rather than a supported configuration -- the NORD_TORCH=0 off-switch that
    # used to force this branch, and the torch-free CI matrix it kept green, were both removed
    # 2026-07-29 with the zero-dependency rule they implemented.
    try:
        import torch  # noqa: F401

        _torch = True
    except Exception:
        _torch = False
    for region, mods in TORCH_TIER.items():
        for m in mods:
            assert region_of(m) == region, f"torch-tier {m} must region-resolve to {region}"
            if _torch:
                importlib.import_module(f"snn_data_lake.{region}.{m}")
            else:
                print(
                    f"SKIP  torch-tier {region}.{m}  (torch absent -- torch-free matrix stays green)"
                )

    # train-tier annex (2026-08-11): same contract, gated on TORCHVISION (the [train] extra --
    # torch alone is not enough; the first post-promotion CI run proved it: torch-tier CI has
    # torch but no torchvision, and the alias round-trip failed on these three).
    try:
        import torchvision  # noqa: F401

        _train = True
    except Exception:
        _train = False
    for region, mods in TRAIN_TIER.items():
        for m in mods:
            assert region_of(m) == region, f"train-tier {m} must region-resolve to {region}"
            if _train:
                importlib.import_module(f"snn_data_lake.{region}.{m}")
            else:
                print(
                    f"SKIP  train-tier {region}.{m}  (torchvision absent -- [train]-free envs stay green)"
                )

    print(
        f"self-check OK: {total} modules mapped to {len(REGIONS)} regions "
        f"({region_total} core + {torch_total} torch-tier); all {len(_MOVED)} legacy import paths "
        f"still resolve to the moved module; map matches the filesystem exactly"
    )


if __name__ == "__main__":
    main()
