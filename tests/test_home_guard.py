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


def test_a_hardlink_to_the_live_lake_inside_an_isolated_home_is_refused(tmp_path, monkeypatch):
    # R2's refutation by execution: a hardlink resolves to itself (no alias for realpath) but writes
    # through it hit the live lake's bytes; same file identity (samestat) is the only tell
    live = tmp_path / "live"
    live.mkdir()
    (live / "brain_lake.bin").write_bytes(b"live")
    iso = tmp_path / "iso"
    iso.mkdir()
    monkeypatch.setenv("NORD_HOME", str(iso))
    (iso / "brain_lake.bin").write_bytes(b"mine")  # the positive: same name, different file
    assert H.require_isolated_home(live_home=str(live))
    (iso / "brain_lake.bin").unlink()
    try:
        os.link(live / "brain_lake.bin", iso / "brain_lake.bin")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create a hardlink here: {exc}")
    with pytest.raises(H.LiveHomeRefused, match="hardlink"):
        H.require_isolated_home(live_home=str(live))


def test_the_alias_scan_is_recursive_and_bounded(tmp_path, monkeypatch):
    # lead: a symlink / hardlink inside a SUBDIRECTORY of the isolated home must be caught too, and
    # the walk has a stated bound (a home over _MAX_HOME_ENTRIES entries is refused, not scanned)
    live = tmp_path / "live"
    live.mkdir()
    (live / "brain_lake.bin").write_bytes(b"live")
    iso = tmp_path / "iso"
    (iso / "a" / "b").mkdir(parents=True)
    monkeypatch.setenv("NORD_HOME", str(iso))
    (iso / "a" / "b" / "mine.bin").write_bytes(b"mine")
    assert H.require_isolated_home(live_home=str(live))  # the positive: a nested, ordinary file
    _symlink_or_skip(iso / "a" / "b" / "lake.bin", live / "brain_lake.bin")
    with pytest.raises(H.LiveHomeRefused, match="lake.bin"):
        H.require_isolated_home(live_home=str(live))
    (iso / "a" / "b" / "lake.bin").unlink()
    try:
        os.link(live / "brain_lake.bin", iso / "a" / "hard.bin")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create a hardlink here: {exc}")
    with pytest.raises(H.LiveHomeRefused, match="hardlink"):
        H.require_isolated_home(live_home=str(live))
    (iso / "a" / "hard.bin").unlink()
    monkeypatch.setattr(H, "_MAX_HOME_ENTRIES", 2)  # a, b, mine.bin = 3 entries
    with pytest.raises(H.LiveHomeRefused, match="entries"):
        H.require_isolated_home(live_home=str(live))


def test_a_hardlink_of_a_file_nested_in_the_live_home_is_refused(tmp_path, monkeypatch):
    # R2's re-attack: the live home keeps files under subdirectories too: a hardlink of one
    live = tmp_path / "live"
    (live / "sub2").mkdir(parents=True)
    (live / "sub2" / "nested.bin").write_bytes(b"live")
    iso = tmp_path / "iso"
    iso.mkdir()
    monkeypatch.setenv("NORD_HOME", str(iso))
    (iso / "nested_hl.bin").write_bytes(b"mine")  # the positive: same size class, different file
    assert H.require_isolated_home(live_home=str(live))
    (iso / "nested_hl.bin").unlink()
    try:
        os.link(live / "sub2" / "nested.bin", iso / "nested_hl.bin")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create a hardlink here: {exc}")
    with pytest.raises(H.LiveHomeRefused, match="hardlink"):
        H.require_isolated_home(live_home=str(live))


def test_a_live_home_over_the_entry_bound_is_refused_not_silently_truncated(tmp_path, monkeypatch):
    # the live-side walk used to stop at the bound without saying so, so a later file's hardlink
    # identity was never collected: the guard now fails closed when it cannot see the live home
    live = tmp_path / "live"
    live.mkdir()
    for i in range(4):
        (live / f"f{i}.bin").write_bytes(b"live%d" % i)
    iso = tmp_path / "iso"
    iso.mkdir()
    monkeypatch.setenv("NORD_HOME", str(iso))
    assert H.require_isolated_home(live_home=str(live))  # the positive: under the default bound
    monkeypatch.setattr(H, "_MAX_HOME_ENTRIES", 2)
    with pytest.raises(H.LiveHomeRefused, match="live home"):
        H.require_isolated_home(live_home=str(live))
    monkeypatch.setattr(H, "_MAX_HOME_ENTRIES", 4)  # exactly at the bound: still fully seen
    assert H.require_isolated_home(live_home=str(live))


def test_an_unreadable_live_subdir_is_refused_but_an_absent_live_root_is_not(tmp_path, monkeypatch):
    # Codex fold re-panel: `except OSError: continue` let an unreadable live subdirectory pass, so a
    # hardlink to a file under it was accepted. Any scan error refuses; only an ABSENT root is [].
    live = tmp_path / "live"
    (live / "locked").mkdir(parents=True)
    (live / "locked" / "x.bin").write_bytes(b"live")
    iso = tmp_path / "iso"
    iso.mkdir()
    monkeypatch.setenv("NORD_HOME", str(iso))
    assert H.require_isolated_home(live_home=str(live))  # the positive: everything readable
    real_scandir = os.scandir

    def deny_locked(path="."):
        if os.path.basename(os.fspath(path)) == "locked":
            raise PermissionError(13, "denied", os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", deny_locked)
    with pytest.raises(H.LiveHomeRefused, match="unreadable"):
        H.require_isolated_home(live_home=str(live))
    monkeypatch.undo()
    monkeypatch.setenv("NORD_HOME", str(iso))
    assert H.require_isolated_home(live_home=str(tmp_path / "absent_live"))  # absent root: fine


def test_a_missing_home_is_a_named_refusal_not_a_raw_oserror(tmp_path, monkeypatch, decoy):
    # R2's probe: strict resolution raised FileNotFoundError, which a harness catching only
    # LiveHomeRefused (the class docstring: one type for every refusal) would print as a traceback
    missing = tmp_path / "never_created"
    with pytest.raises(H.LiveHomeRefused, match="missing"):
        H.assert_not_live_home(str(missing), live_home=str(decoy))
    monkeypatch.setenv("NORD_HOME", str(missing))
    with pytest.raises(H.LiveHomeRefused, match="missing"):
        H.require_isolated_home(live_home=str(decoy))


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


# ---- the live-lake stamp (moved here from instruments/sleep_plasticity_ab.py, P2: the ONE copy)
def test_an_unreadable_file_fails_the_live_lake_check_closed(tmp_path, monkeypatch):
    """A file whose stat fails used to be skipped, so a change under it was invisible. It is now
    recorded as None, and the check cannot read untouched while any file is None."""
    decoy = tmp_path / "decoy_live"
    decoy.mkdir()
    (decoy / "brain_lake.bin").write_bytes(b"abc")
    (decoy / "latch.json").write_text("x")
    before = H.lake_snapshot(str(decoy))
    assert H.live_lake_untouched(str(decoy), before) == (True, [])
    real = os.stat

    def flaky(path, *a, **k):
        if str(path).endswith("latch.json"):
            raise PermissionError(path)
        return real(path, *a, **k)

    monkeypatch.setattr(os, "stat", flaky)
    unreadable = H.lake_snapshot(str(decoy))
    assert unreadable["latch.json"] is None  # recorded, never skipped
    ok, changed = H.live_lake_untouched(str(decoy), unreadable)  # None on both reads
    assert not ok and "latch.json" in changed


def test_an_unlistable_directory_fails_the_live_lake_check_closed(tmp_path, monkeypatch):
    """os.walk skips a directory it cannot list unless told otherwise, so a persistently unlistable
    nested directory hid every change under it. It is now recorded as None."""
    decoy = tmp_path / "decoy_live"
    (decoy / "sub").mkdir(parents=True)
    (decoy / "sub" / "a.bin").write_bytes(b"abc")
    before = H.lake_snapshot(str(decoy))
    assert before["sub" + os.sep + "a.bin"] is not None
    real = os.scandir

    def deny(path="."):
        if str(path).endswith("sub"):
            raise PermissionError(13, "denied", str(path))
        return real(path)

    monkeypatch.setattr(os, "scandir", deny)
    blind = H.lake_snapshot(str(decoy))
    assert any(v is None for v in blind.values())  # the unlistable directory is recorded
    ok, changed = H.live_lake_untouched(str(decoy), blind)  # unlistable on both reads
    assert not ok and changed


def test_an_absent_lake_snapshots_empty_and_reads_untouched(tmp_path):
    """An absent path snapshots as empty (documented); an absent ROOT is not an unlistable directory
    (fold lens: it raised K14 RED in run_arm)."""
    gone = str(tmp_path / "never_created")
    before = H.lake_snapshot(gone)
    assert before == {}
    assert H.live_lake_untouched(gone, before) == (True, [])


def test_a_directory_link_under_the_root_fails_the_live_lake_check_closed(tmp_path, monkeypatch):
    """os.walk never descends a directory link: a change behind one was invisible.
    The link itself is recorded as None (never equal), so the check cannot read untouched."""
    decoy = tmp_path / "decoy_live"
    (decoy / "real").mkdir(parents=True)
    (decoy / "real" / "a.bin").write_bytes(b"abc")
    (
        decoy / "link"
    ).mkdir()  # stands in for a link: islink is patched, so no symlink privilege is needed
    real = os.path.islink
    monkeypatch.setattr(os.path, "islink", lambda p: str(p).endswith("link") or real(p))
    snap = H.lake_snapshot(str(decoy))
    assert snap["link"] is None
    ok, changed = H.live_lake_untouched(str(decoy), snap)
    assert not ok and "link" in changed


def test_a_same_size_same_mtime_rewrite_fails_the_live_lake_check(tmp_path):
    """Batch-2 closing report: size and mtime alone miss AAAA -> BBBB with the metadata put back."""
    decoy = tmp_path / "decoy_live"
    decoy.mkdir()
    f = decoy / "brain_lake.bin"
    f.write_bytes(b"AAAA")
    st = os.stat(f)
    before = H.lake_snapshot(str(decoy))
    assert H.live_lake_untouched(str(decoy), before) == (True, [])
    f.write_bytes(b"BBBB")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (os.stat(f).st_size, os.stat(f).st_mtime_ns) == (st.st_size, st.st_mtime_ns)
    ok, changed = H.live_lake_untouched(str(decoy), before)
    assert not ok and changed == ["brain_lake.bin"]
