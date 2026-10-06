"""specs/022 T102 (R1) / lead ruling 16 -- the ONE shared live-home guard in substrate/home.py.

Every red runs against a DECOY live home passed as `live_home`; no test here touches the real
~/.snn_data_lake. A prefix check on `selfcheck_home_` is not isolation (Codex #14 on G5): a
prefixed junction or symlink can alias the live lake, so the guard resolves before it compares.
"""

import os
import subprocess
import sys

import pytest

from snn_data_lake.substrate import home as H


def _alias(link, target):
    """A directory alias the way an attacker-by-accident would make one: a junction on Windows
    (no admin needed), a symlink elsewhere."""
    if sys.platform == "win32":
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        assert r.returncode == 0, r.stderr
    else:
        os.symlink(target, link, target_is_directory=True)


@pytest.fixture
def decoy(tmp_path):
    live = tmp_path / "decoy" / ".snn_data_lake"
    live.mkdir(parents=True)
    return live


def test_a_prefixed_junction_to_the_live_home_is_refused(tmp_path, decoy):
    link = tmp_path / "selfcheck_home_alias"
    _alias(link, decoy)
    assert link.name.startswith("selfcheck_home_")  # the prefix check alone would accept it
    with pytest.raises(H.LiveHomeRefused, match="equal"):
        H.assert_not_live_home(link, live_home=decoy)


def test_containment_is_refused_in_both_directions(decoy):
    inside = decoy / "sub"
    inside.mkdir()
    with pytest.raises(H.LiveHomeRefused, match="contained"):
        H.assert_not_live_home(inside, live_home=decoy)
    with pytest.raises(H.LiveHomeRefused, match="contains"):
        H.assert_not_live_home(decoy.parent, live_home=decoy)


def test_empty_and_unset_homes_are_refused(monkeypatch, decoy):
    with pytest.raises(H.LiveHomeRefused, match="unset"):
        H.assert_not_live_home("", live_home=decoy)
    monkeypatch.delenv("NORD_HOME", raising=False)
    with pytest.raises(H.LiveHomeRefused, match="unset"):
        H.require_isolated_home(live_home=decoy)  # the default home IS the live lake
    monkeypatch.setenv("NORD_HOME", "")
    with pytest.raises(H.LiveHomeRefused, match="unset"):
        H.require_isolated_home(live_home=decoy)


def test_the_refusal_type_is_catchable_as_value_error(decoy):
    with pytest.raises(ValueError):
        H.assert_not_live_home(decoy, live_home=decoy)


def test_a_fresh_throwaway_home_is_accepted_and_resolved(decoy):
    # the positive: the guard does not refuse a real isolated home, and NORD_HOME holds the
    # RESOLVED path inside the context
    with H.throwaway_home(live_home=decoy) as d:
        assert os.environ["NORD_HOME"] == d
        assert os.path.normcase(d) == os.path.normcase(os.path.realpath(d))
        assert H.require_isolated_home(live_home=decoy) == H.assert_not_live_home(
            d, live_home=decoy
        )


def test_throwaway_home_honours_its_parent(tmp_path, decoy):
    # Codex b3 #11: the multi-GB homes must land under the named parent, not the system TEMP
    parent = tmp_path / "scratch_root"
    parent.mkdir()
    with H.throwaway_home(live_home=decoy, parent=str(parent)) as d:
        assert os.path.normcase(os.path.dirname(d)) == os.path.normcase(os.path.realpath(parent))


def test_a_home_on_another_drive_is_not_contained(tmp_path):
    # commonpath raises on different drives; the guard must read that as "not contained", not crash
    if sys.platform != "win32":
        pytest.skip("drive letters are a Windows concept")
    here = os.path.splitdrive(str(tmp_path))[0].upper()
    other = "D:\\" if here != "D:" else "C:\\"
    if not os.path.isdir(other):
        pytest.skip(f"no {other} on this host")
    assert H.assert_not_live_home(
        tmp_path, live_home=os.path.join(other, "nowhere", ".snn_data_lake")
    )


@pytest.mark.parametrize("prefix", ["\\\\?\\", "//?/", "\\\\.\\"])
def test_extended_windows_spellings_cannot_bypass_the_guard(tmp_path, prefix):
    # Codex int1 BLOCKER (home.py:415): the extended spelling read as a different drive from the
    # ordinary one, so equality and containment were skipped. Normalised on BOTH sides.
    if sys.platform != "win32":
        pytest.skip("DOS device namespace spellings are Windows-only")
    live = tmp_path / "live"
    sub = live / "sub"
    sub.mkdir(parents=True)
    ext = lambda p: prefix + str(p)  # noqa: E731
    with pytest.raises(H.LiveHomeRefused, match="equal"):
        H.assert_not_live_home(ext(live), live_home=str(live))
    with pytest.raises(H.LiveHomeRefused, match="contained"):
        H.assert_not_live_home(ext(sub), live_home=str(live))
    with pytest.raises(H.LiveHomeRefused, match="contained"):
        H.assert_not_live_home(str(sub), live_home=ext(live))  # the LIVE side spelled extended
    with pytest.raises(H.LiveHomeRefused, match="contains"):
        H.assert_not_live_home(ext(tmp_path), live_home=str(live))
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert H.assert_not_live_home(ext(other), live_home=str(live)) == os.path.realpath(other)


def _symlink_or_skip(link, target):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:  # Windows needs developer mode or admin
        pytest.skip(f"cannot create a symlink here: {exc}")


def test_an_isolated_home_whose_lake_file_aliases_the_live_lake_is_refused(tmp_path, monkeypatch):
    # Codex int1 BLOCKER (home.py:433): only the home DIRECTORY was checked, so a dir whose
    # brain_lake.bin (or a sidecar) is a symlink to the live lake passed and _log() bound it
    live = tmp_path / "live"
    live.mkdir()
    (live / "brain_lake.bin").write_bytes(b"live")
    (live / "estop.latch").write_bytes(b"live")
    iso = tmp_path / "iso"
    iso.mkdir()
    monkeypatch.setenv("NORD_HOME", str(iso))
    assert H.require_isolated_home(live_home=str(live))  # the positive: an empty isolated home
    (iso / "notes.json").write_bytes(b"mine")
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes(b"x")
    _symlink_or_skip(iso / "other.bin", elsewhere)  # an alias to a NON-live file is fine
    assert H.require_isolated_home(live_home=str(live))
    _symlink_or_skip(iso / "brain_lake.bin", live / "brain_lake.bin")
    with pytest.raises(H.LiveHomeRefused, match="brain_lake.bin"):
        H.require_isolated_home(live_home=str(live))
    (iso / "brain_lake.bin").unlink()
    _symlink_or_skip(iso / "estop.latch", live / "estop.latch")  # a sidecar alias is refused too
    with pytest.raises(H.LiveHomeRefused, match="estop.latch"):
        H.require_isolated_home(live_home=str(live))


def test_throwaway_home_refuses_a_live_parent_before_creating_anything(tmp_path, monkeypatch):
    # Codex int1 MAJOR (home.py:452): the temp dir was created inside the live home, THEN refused
    live = tmp_path / "live"
    live.mkdir()
    created = []
    real = H.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        H.tempfile, "TemporaryDirectory", lambda *a, **k: created.append(k) or real(*a, **k)
    )
    with pytest.raises(H.LiveHomeRefused):
        with H.throwaway_home(live_home=str(live), parent=str(live)):
            pass
    assert created == [] and list(live.iterdir()) == []
    with H.throwaway_home(live_home=str(live), parent=str(tmp_path)) as d:  # the positive
        assert os.path.isdir(d) and len(created) == 1


def test_a_network_or_device_spelling_is_never_an_isolated_home(tmp_path):
    # R2's refutation by execution: the loopback admin share names the LOCAL decoy live home but
    # realpath never folds it back to the drive, so equality and containment were both skipped
    if sys.platform != "win32":
        pytest.skip("UNC / device namespaces are Windows-only")
    live = tmp_path / "live"
    sub = live / "sub"
    sub.mkdir(parents=True)
    drive, rest = os.path.splitdrive(str(live))
    unc = "\\\\localhost\\" + drive[0] + "$" + rest
    if not os.path.isdir(unc):
        pytest.skip("the loopback administrative share is not reachable on this host")
    for spelled in (unc, unc + "\\sub", "\\\\?\\UNC\\localhost\\" + drive[0] + "$" + rest):
        with pytest.raises(H.LiveHomeRefused, match="network"):
            H.assert_not_live_home(spelled, live_home=str(live))
    with pytest.raises(H.LiveHomeRefused, match="device"):  # not a crash, not an accident
        H.assert_not_live_home("\\\\?\\GLOBALROOT\\Device\\x", live_home=str(live))


def test_namespace_prefix_normalisation_covers_unc():
    # `\\?\UNC\server\share` is the extended spelling of `\\server\share`
    assert H._strip_namespace("\\\\?\\UNC\\srv\\share\\x") == "\\\\srv\\share\\x"
    assert H._strip_namespace("\\\\?\\C:\\x") == "C:\\x"
    assert H._strip_namespace("C:\\x") == "C:\\x"


def test_throwaway_home_accepts_a_parent_that_merely_contains_the_live_home(tmp_path):
    # the over-strict reading (refuse a parent that contains the live home) would break a parent
    # like the user's home directory: the child is a sibling of the live home, not inside it
    live = tmp_path / "up" / ".snn_data_lake"
    live.mkdir(parents=True)
    with H.throwaway_home(live_home=str(live), parent=str(tmp_path / "up")) as d:
        assert os.path.isdir(d) and os.path.dirname(d) == os.path.realpath(tmp_path / "up")
