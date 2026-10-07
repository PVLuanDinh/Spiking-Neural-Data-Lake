"""The pytest-level live-home tripwire (owner 2026-10-06, the freeze-lift condition after the
15:42 live-lake write). R2's test called monkeypatch.undo() mid-test; that also reverted
conftest's _hermetic_home NORD_HOME, so home() fell back to the live lake and a persisting
Brain appended to it. Under pytest, home() must REFUSE the live home instead.

Every test points the live home at a DECOY under tmp_path (`_default_live_home` patched), so a red
run against unguarded code writes into the decoy, never into the operator's lake, and the decoy's
listing is an observation with demonstrated sensitivity (test_control_the_decoy_sees_a_write)."""

import os

import pytest

from snn_data_lake.substrate import home as H


@pytest.fixture
def decoy(monkeypatch, tmp_path):
    live = tmp_path / "decoy_live"
    live.mkdir()
    (live / "brain_lake.bin").write_bytes(b"lake")
    monkeypatch.setattr(H, "_default_live_home", lambda: str(live))
    return live


def _listing(d):
    return sorted(
        (os.path.relpath(os.path.join(r, n), d), os.stat(os.path.join(r, n)).st_mtime_ns)
        for r, ds, fs in os.walk(d)
        for n in ds + fs
    )


def test_undo_mid_test_then_home_is_refused(monkeypatch, tmp_path, decoy):
    """R2's exact sequence: a test-local setenv, then undo() -- which also drops the fixture's home.
    undo() restores the LAUNCH environment, so an inherited NORD_HOME is deleted explicitly (the
    incident's state) and the decoy, also reverted by undo(), is re-applied."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path / "local"))
    monkeypatch.undo()
    monkeypatch.delenv("NORD_HOME", raising=False)
    monkeypatch.setattr(H, "_default_live_home", lambda: str(decoy))
    before = _listing(decoy)
    with pytest.raises(H.LiveHomeRefused, match="unset"):
        H.home()
    assert _listing(decoy) == before


def test_home_set_to_the_live_home_is_refused(monkeypatch, decoy):
    monkeypatch.setenv("NORD_HOME", str(decoy))
    with pytest.raises(H.LiveHomeRefused, match="equal"):
        H.home()


def test_control_an_isolated_home_passes(monkeypatch, tmp_path, decoy):
    """The control that is not the method: an isolated home resolves and is returned unchanged."""
    before = _listing(decoy)
    d = tmp_path / "iso"
    monkeypatch.setenv("NORD_HOME", str(d))
    assert os.path.normcase(H.home()) == os.path.normcase(str(d))
    assert d.is_dir()
    assert _listing(decoy) == before


def test_control_the_decoy_sees_a_write(decoy):
    """Sensitivity of the instrument: a write into the decoy changes its listing."""
    before = _listing(decoy)
    (decoy / "probe").write_bytes(b"x")
    assert _listing(decoy) != before
