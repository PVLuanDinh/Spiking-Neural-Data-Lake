"""ENC (specs/022 evidence/encoder-path-root.md): the brain's gitignored model assets resolve
against the REPOSITORY ROOT, not the cwd.

Before the fix a process whose cwd was not the repo root silently built the no-encoder brain
(perception_tier "stdlib", no audio encoder) -- what every in-process self-check measured
(run_selfchecks chdirs each check into a scratch dir), and what the B3 intero bar read as a
"cross-test leak" (28/0/0 against the production tier's 6/9/20).

Skips BY NAME where the encoders are absent from this checkout's models/ (CI, asset-less lanes):
the fix removes the cwd input, not the checkout input.
"""

from pathlib import Path

import pytest

from snn_data_lake.engine.wiring_manifest import REPO

_ENCODERS = (REPO / "models" / "eprop_nmnist.pt", REPO / "models" / "eprop_shd.pt")
_ASSET_VARS = ("SNN_EPROP_CKPT", "SNN_PERCEPTION_PACK", "SNN_EPROP_AUDIO_CKPT", "SNN_SHD_PACK")


@pytest.fixture
def encoders_present():
    missing = [p.name for p in _ENCODERS if not p.exists()]
    if missing:
        pytest.skip(f"production perception tier unavailable in this checkout (absent: {missing})")


def _tier_from(cwd, monkeypatch):
    from snn_data_lake.engine.brain import make_brain

    monkeypatch.chdir(cwd)
    b = make_brain(seed=0, persist=False)
    try:
        # (vision tier, audio encoder, vision PACK rows, audio PACK rows): all four asset sites
        return (
            b._perception_tier,
            b._audio_enc is not None,
            bool(b._perceiving_real),
            bool(b._perceiving_audio),
        )
    finally:
        b.close()


def test_the_root_is_this_checkout():
    """The package under test and the models it resolves come from the SAME tree; a test run from
    a worktree that imported another checkout's package would otherwise pass on its assets."""
    assert REPO == Path(__file__).resolve().parents[1]


def test_a_cwd_without_models_still_builds_the_production_brain(
    tmp_path, monkeypatch, encoders_present
):
    """THE RED: on the parent this tmp cwd has no models/, so the defaults missed and the tier fell
    to stdlib with no audio encoder and no pack rows."""
    for var in _ASSET_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not (tmp_path / "models").exists()
    assert _tier_from(tmp_path, monkeypatch) == ("eprop-snn", True, True, True)


def test_an_env_override_still_governs(tmp_path, monkeypatch, encoders_present):
    """CONTROL, not the method: an override is used exactly as given -- a nonexistent checkpoint
    gives the stdlib vision tier, so the positive assertion above is not vacuous."""
    monkeypatch.setenv("SNN_EPROP_CKPT", str(tmp_path / "absent.pt"))
    tier, *_ = _tier_from(tmp_path, monkeypatch)
    assert tier == "stdlib"


def test_an_empty_override_is_still_an_override(tmp_path, monkeypatch, encoders_present):
    """An empty SNN_EPROP_CKPT means "no checkpoint", as before the fix (memory_diagnostic and
    carried_state_driver read set-but-empty the same way). RED on fd69bcb1's `or`, which fell
    back to the repo-root default and loaded the encoder."""
    monkeypatch.setenv("SNN_EPROP_CKPT", "")
    tier, *_ = _tier_from(tmp_path, monkeypatch)
    assert tier == "stdlib"
