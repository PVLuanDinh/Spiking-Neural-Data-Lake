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


def test_missing_child_of_the_live_home_is_refused_before_creation(monkeypatch, decoy):
    """Panel (p8 + safety lenses, MAJOR): makedirs ran before the guard, so a refused home was
    already created inside the live home."""
    child = decoy / "probe_child"
    monkeypatch.setenv("NORD_HOME", str(child))
    with pytest.raises(H.LiveHomeRefused, match="contained"):
        H.home()
    assert not child.exists()


@pytest.mark.skipif(os.name != "nt", reason="the loopback admin share is a Windows spelling")
def test_share_spelling_of_a_live_child_is_refused_before_creation(monkeypatch, decoy):
    """The same MAJOR through a spelling commonpath cannot compare (a share vs a drive): the
    network refusal must also come before makedirs."""
    drive, rest = os.path.splitdrive(str(decoy))
    share = "\\\\localhost\\" + drive[0] + "$" + rest
    if not os.path.isdir(share):
        pytest.skip("loopback admin share not reachable")
    child = decoy / "share_child"
    monkeypatch.setenv("NORD_HOME", os.path.join(share, "share_child"))
    with pytest.raises(H.LiveHomeRefused, match="network"):
        H.home()
    assert not child.exists()


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_aliased_lake_file_in_an_isolated_home_is_refused(monkeypatch, tmp_path, decoy, alias):
    """Panel (BLOCKER, all three lenses): a directory check misses a lake-file alias; home() must
    run the shared alias scan (require_isolated_home), not only the directory guard."""
    iso = tmp_path / "iso"
    iso.mkdir()
    link = os.link if alias == "hardlink" else os.symlink
    try:
        link(decoy / "brain_lake.bin", iso / "brain_lake.bin")
    except OSError as e:
        pytest.skip(f"{alias} unavailable: {e}")
    monkeypatch.setenv("NORD_HOME", str(iso))
    with pytest.raises(H.LiveHomeRefused):
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


def test_two_spellings_of_one_lake_file_share_one_log(monkeypatch, tmp_path):
    """#1097/#1110: _log keyed on the path string as given, so a detour spelling of one file got its
    own BinLog and its own dedup set (the split-dedup problem the cache exists to prevent)."""
    monkeypatch.setattr(H, "_LOG_CACHE", {})
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = str(tmp_path / "a" / "lake.bin")
    detour = os.path.join(str(tmp_path), "b", "..", "a", "lake.bin")
    assert H._log(plain) is H._log(detour)


_EPISODE = {"who": "arm", "what": "JOINT_SHOULDER_LIFT", "where": "0@2", "when": "290"}


@pytest.mark.parametrize(
    "write",
    [
        lambda p: H.save_episode(_EPISODE, path=p),
        lambda p: H.save_named_state("probe", b"x", path=p),
    ],
    ids=["save_episode", "save_named_state"],
)
def test_an_explicit_path_into_the_live_home_is_refused(monkeypatch, decoy, write):
    """#1097/#1110: an explicit `path=` never reached the live-home guard, so a live
    path wrote the live lake. Armed (pytest) it must raise before anything is created."""
    monkeypatch.setattr(H, "_LOG_CACHE", {})
    before = _listing(decoy)
    with pytest.raises(H.LiveHomeRefused):
        write(str(decoy / "explicit_lake.bin"))
    assert _listing(decoy) == before


def test_control_an_explicit_isolated_path_is_written(monkeypatch, tmp_path, decoy):
    """The control that is not the method: the same write to an isolated path goes through."""
    monkeypatch.setattr(H, "_LOG_CACHE", {})
    target = tmp_path / "iso_lake.bin"
    assert H.save_episode(_EPISODE, path=str(target))
    assert target.is_file()
