"""The pytest-level live-home tripwire (owner 2026-10-06, the freeze-lift condition after the
15:42 live-lake write). R2's test called monkeypatch.undo() mid-test; that also reverted
conftest's _hermetic_home NORD_HOME, so home() fell back to the live lake and a persisting
Brain appended to it. Under pytest, home() must REFUSE the live home instead. Brain-free:
these tests call home() only, which on the unguarded code merely re-resolves an existing dir."""

import os

import pytest

from snn_data_lake.substrate import home as H


def _lake_stamp():
    p = os.path.join(H._default_live_home(), "brain_lake.bin")
    if not os.path.exists(p):
        return None
    st = os.stat(p)
    return (st.st_size, st.st_mtime_ns)


def test_undo_mid_test_then_home_is_refused(monkeypatch, tmp_path):
    """R2's exact sequence: a test-local setenv, then undo() -- which also drops the fixture's home."""
    before = _lake_stamp()
    monkeypatch.setenv("NORD_HOME", str(tmp_path / "local"))
    monkeypatch.undo()
    assert "NORD_HOME" not in os.environ  # the fixture's isolation is gone, as in the incident
    with pytest.raises(H.LiveHomeRefused):
        H.home()
    assert _lake_stamp() == before


def test_home_set_to_the_live_home_is_refused(monkeypatch):
    monkeypatch.setenv("NORD_HOME", H._default_live_home())
    with pytest.raises(H.LiveHomeRefused, match="equal"):
        H.home()


def test_control_an_isolated_home_passes(monkeypatch, tmp_path):
    """The control that is not the method: an isolated home resolves and is returned unchanged."""
    before = _lake_stamp()
    d = tmp_path / "iso"
    monkeypatch.setenv("NORD_HOME", str(d))
    assert os.path.normcase(H.home()) == os.path.normcase(str(d))
    assert d.is_dir()
    assert _lake_stamp() == before
