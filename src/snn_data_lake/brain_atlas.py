"""Anatomical atlas of the spiking-neural data lake -- which brain region every module lives in.

As of 3.0 the package is organized by brain anatomy (see snn_data_lake.REGIONS). This module
renders that map and VERIFIES the pre-3.0 back-compat layer: every legacy flat import path
(`snn_data_lake.<module>`) still resolves to the IDENTICAL module object as its new regional
path (`snn_data_lake.<region>.<module>`), so old code keeps working after the move.

  python -m snn_data_lake.brain_atlas        # (also run by run_selfchecks.py)

G8 (specs/022 evidence g8-reorganization.md 2.4 S1a, F5): `region_of` answers the FROZEN pre-G8
label (LEGACY_*), which region-keyed consumers such as the wiring manifest's region check keep
reading while G8 moves code into engine-function packages; the map this module renders and checks
against the filesystem is the LIVE code layout (REGIONS), and every file is located through
module_path(), never through a label.
"""

import importlib
import os

from snn_data_lake import (
    _MOVED,
    LEGACY_REGIONS,
    LEGACY_TORCH_TIER,
    LEGACY_TRAIN_TIER,
    REGIONS,
    TORCH_TIER,
    TRAIN_TIER,
    module_path,
    package_of,
)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


# distinct-from: brain_geometry.py -- not a new region resolver; this module's OWN region_of
# gaining the TRAIN_TIER annex (2026-08-11), same shape as when TORCH_TIER was added.
def region_of(module):
    """Return the brain region a module belongs to, or None if unknown: the FROZEN pre-G8 label
    from LEGACY_REGIONS and its LEGACY_TORCH_TIER / LEGACY_TRAIN_TIER annexes (G8 S1a; owner
    decision 4 -- region-keyed rules read the pre-G8 table until G7/G4 re-target them)."""
    for tier in (LEGACY_REGIONS, LEGACY_TORCH_TIER, LEGACY_TRAIN_TIER):
        for region, mods in tier.items():
            if module in mods:
                return region
    return None


def atlas():
    """Return the region -> modules map (the authoritative anatomical layout: REGIONS core tier).
    The torch-tier annex is snn_data_lake.TORCH_TIER."""
    return REGIONS


# distinct-from: gates/atlas_node_gate.py -- node_references reads module TEXT for node ids;
#   this is brain_atlas's own existing file listing, edited in place (F5), not a second one.
#   Also read: substrate/brain_geometry.py (_region_modules, the membership for placement).
def _region_files(region):
    """The module names (files) present on disk beside `region`'s registered modules (excludes
    __init__). Every directory listed is the one module_path() names for a registered module --
    never dirname(pkg.__file__) of the region name (G8 F5: once a module moves, a label is not a
    directory). A region with no registered module names no directory and contributes no file."""
    mods = set(REGIONS.get(region, ())) | set(TORCH_TIER.get(region, ()))
    mods |= set(TRAIN_TIER.get(region, ()))
    dirs = {os.path.join(_PKG_DIR, *module_path(m).split(".")[1:-1]) for m in mods}
    return {
        f[:-3]
        for d in dirs
        if os.path.isdir(d)
        for f in os.listdir(d)
        if f.endswith(".py") and f != "__init__.py"
    }


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
    # All three tiers in the breakdown: until 2026-08-12 this line omitted the train tier, so the
    # printed sum did not close (222 + 4 != 230) -- an arithmetic self-contradiction the parity
    # assert below could never catch because it checks the SET union, not the printed prose.
    print(
        f"{'TOTAL':<16} {total:>2} modules across {len(REGIONS)} regions "
        f"({region_total} core + {torch_total} torch-tier + {train_total} train-tier)"
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
            # the LIVE layout (package_of), not the frozen label region_of answers since G8 S1a
            assert package_of(m) == region, f"torch-tier {m} must package-resolve to {region}"
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
            assert package_of(m) == region, f"train-tier {m} must package-resolve to {region}"
            if _train:
                importlib.import_module(f"snn_data_lake.{region}.{m}")
            else:
                print(
                    f"SKIP  train-tier {region}.{m}  (torchvision absent -- [train]-free envs stay green)"
                )

    print(
        f"self-check OK: {total} modules mapped to {len(REGIONS)} regions "
        f"({region_total} core + {torch_total} torch-tier + {train_total} train-tier); "
        f"all {len(_MOVED)} legacy import paths "
        f"still resolve to the moved module; map matches the filesystem exactly"
    )


if __name__ == "__main__":
    main()
