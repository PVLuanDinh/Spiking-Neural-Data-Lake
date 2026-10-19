"""BinLog.replay() served from the resident copy (Phase 1a, 2026-10-17): the stored objects in
order and no disk pass while stat + tail frame find the file unchanged -- and the disk again the
moment another writer changes it, even when the change keeps the size, the inode or the mtime tick
(Codex B1 p1a MAJOR 1+2 and two review folds, 2026-10-18). Races are injected at their exact window
by monkeypatching the I/O call that opens it (open, fsync, replace, replay, the instance's stat)."""

import builtins
import os
import threading
import time

import pytest

from snn_data_lake.substrate.binlog import _LEN, _MAGIC, BinLog


def _disk(p):
    return list(BinLog(p, dedup=False).replay())  # dedup=False keeps nothing resident: a disk pass


def _served_resident(log):
    """True only if replay() yielded the resident objects (a disk pass yields new bytes objects)."""
    got = list(log.replay())
    return got == log._payloads and all(x is y for x, y in zip(got, log._payloads, strict=True))


def _overwrite_in_place_same_tick(p, offset, data):
    st = os.stat(p)
    with open(p, "r+b") as f:  # an external tool patching bytes: same size, same inode ...
        f.seek(offset)
        f.write(data)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))  # ... and (pinned) the same mtime


def _leftovers(d):
    return [n for n in os.listdir(d) if n.endswith(".compact.tmp")]


def _raw_append(p, payload):
    with open(p, "ab") as g:
        g.write(_LEN.pack(len(payload)) + payload)


def test_unchanged_log_is_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    for i in range(50):
        a.append(b"\x01rec%03d" % i)
    a.append(b"\x01rec000")  # dedup drop: must not land in the resident copy either
    a.extend([b"\x02ext%03d" % i for i in range(5)] + [b"\x02ext000"])
    disk = _disk(p)
    assert disk == [b"\x01rec%03d" % i for i in range(50)] + [b"\x02ext%03d" % i for i in range(5)]
    assert list(a.replay()) == disk and _served_resident(a)
    a.rewrite([b"\x03keep"] + disk[:3])
    assert list(a.replay()) == [b"\x03keep"] + disk[:3] == _disk(p) and _served_resident(a)


def test_empty_log_is_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    assert list(a.replay()) == [] == _disk(p) and not a.stale()
    a.append(b"\x01one")
    assert a.rewrite([]) == 0
    assert list(a.replay()) == [] == _disk(p) and not a.stale()


def test_rewrite_from_a_stale_instance_is_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    BinLog(p).append(b"B")
    assert a.stale()
    a.rewrite([b"X", b"Y"])  # the file IS the survivors now, whatever this instance missed before
    assert not a.stale() and _served_resident(a) and list(a.replay()) == [b"X", b"Y"] == _disk(p)


def test_undedup_instance_is_never_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"), dedup=False)
    a.append(b"A")
    a.append(b"A")
    BinLog(p).rewrite([b"B"])
    a.extend([b"C"])
    assert a._payloads is None and list(a.replay()) == [b"B", b"C"] == _disk(p)
    a.rewrite([b"D"])
    assert a._payloads is None and list(a.replay()) == [b"D"] == _disk(p)


def test_external_append_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"\x01one")
    BinLog(p).append(b"\x01two")  # a second writer over the same file (another process, in effect)
    assert a.stale()
    assert list(a.replay()) == [b"\x01one", b"\x01two"] == _disk(p)
    with open(p, "r+b") as f:  # an external truncation (a torn tail) is seen the same way
        f.truncate(os.path.getsize(p) - 2)
    assert list(a.replay()) == [b"\x01one"] == _disk(p)


def test_equal_size_external_rewrite_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    BinLog(p).rewrite([b"B"])  # same size, new content, another instance
    assert a.stale()
    assert list(a.replay()) == [b"B"] == _disk(p)


def test_equal_size_in_place_overwrite_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"AAAA")
    _overwrite_in_place_same_tick(p, len(_MAGIC) + _LEN.size, b"BBBB")
    assert a._fingerprint() == a._stamp  # the stat cannot see it: only the tail read can
    assert a.stale()
    assert list(a.replay()) == [b"BBBB"] == _disk(p)


def test_local_append_after_an_unseen_in_place_edit_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"AAAA")
    _overwrite_in_place_same_tick(p, len(_MAGIC) + _LEN.size, b"BBBB")
    a.append(b"C")
    assert a.stale()
    assert list(a.replay()) == [b"BBBB", b"C"] == _disk(p)


def test_local_extend_after_an_unseen_in_place_edit_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"AAAA")
    _overwrite_in_place_same_tick(p, len(_MAGIC) + _LEN.size, b"BBBB")
    a.extend([b"C"])
    assert a.stale()
    assert list(a.replay()) == [b"BBBB", b"C"] == _disk(p)


def test_mtime_only_edit_racing_an_empty_extend_is_not_served_resident(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.extend([b"A", b"Z"])
    real_open, raced = builtins.open, []

    def racing_open(path, mode="r", *args, **kw):  # same size, same inode, same tail, newer mtime
        if mode == "ab" and not raced:
            raced.append(1)
            with real_open(p, "r+b") as f:
                f.seek(len(_MAGIC) + _LEN.size)
                f.write(b"B")
            st = os.stat(p)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        return real_open(path, mode, *args, **kw)

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", racing_open)
        a.extend([])  # wrote nothing: must not adopt the foreign mtime
    assert raced and a.stale()
    assert list(a.replay()) == [b"B", b"Z"] == _disk(p)


def test_local_append_after_an_external_append_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    BinLog(p).append(b"B")
    a.append(b"C")
    assert a.stale()
    assert list(a.replay()) == [b"A", b"B", b"C"] == _disk(p)


def test_empty_local_extend_after_an_external_append_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    BinLog(p).append(b"B")
    a.extend([])
    assert a.stale()
    assert list(a.replay()) == [b"A", b"B"] == _disk(p)


def test_local_append_after_an_equal_size_rewrite_is_not_served_resident(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    BinLog(p).rewrite([b"B"])
    a.append(b"C")
    assert a.stale()
    assert list(a.replay()) == [b"B", b"C"] == _disk(p)


def test_equal_size_rewrite_racing_a_local_append_is_not_served_resident(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    real_open, raced = builtins.open, []

    def racing_open(path, mode="r", *args, **kw):  # between append's stale() and its open("ab")
        if mode == "ab" and not raced:
            raced.append(1)
            BinLog(p).rewrite([b"B"])
        return real_open(path, mode, *args, **kw)

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", racing_open)
        a.append(b"C")
    assert raced and a.stale()
    assert list(a.replay()) == [b"B", b"C"] == _disk(p)


def test_external_append_racing_a_local_append_is_not_served_resident(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    real_fsync, raced = os.fsync, []

    def racing_fsync(fd):  # inside append's write window, before its post-write stat
        real_fsync(fd)
        if not raced:
            raced.append(1)
            _raw_append(p, b"B")

    with monkeypatch.context() as m:
        m.setattr(os, "fsync", racing_fsync)
        a.append(b"C")
    assert raced and a.stale()
    assert list(a.replay()) == [b"A", b"C", b"B"] == _disk(p)


def test_identical_tail_append_racing_a_local_append_is_not_served_resident(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    real_fsync, raced = os.fsync, []

    def racing_fsync(fd):  # the foreign frame is byte-identical to ours: only exact growth sees it
        real_fsync(fd)
        if not raced:
            raced.append(1)
            _raw_append(p, b"C")

    with monkeypatch.context() as m:
        m.setattr(os, "fsync", racing_fsync)
        a.append(b"C")
    assert raced and a.stale()
    assert list(a.replay()) == [b"A", b"C", b"C"] == _disk(p)


def test_construction_racing_a_same_tail_rewrite_is_not_served_resident(tmp_path, monkeypatch):
    BinLog(p := str(tmp_path / "lake.bin")).extend([b"A", b"Z"])
    real_replay, raced = BinLog.replay, []

    def racing_replay(self):  # an equal-size, same-tail replacement lands as the boot pass finishes
        yield from real_replay(self)
        if not raced:
            raced.append(1)
            BinLog(p).rewrite([b"B", b"Z"])

    with monkeypatch.context() as m:
        m.setattr(BinLog, "replay", racing_replay)
        a = BinLog(p)
    assert raced and a.stale()  # stamped before the pass: a mid-pass change is never blessed
    assert list(a.replay()) == [b"B", b"Z"] == _disk(p)


def test_rewrite_racing_a_same_tail_rewrite_is_not_served_resident(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    real_replace, raced = os.replace, []

    def racing_replace(src, dst):  # a same-tail rewrite lands between a's os.replace and its stat
        real_replace(src, dst)
        if not raced:
            raced.append(1)
            BinLog(p).rewrite([b"BB", b"Z"])

    with monkeypatch.context() as m:
        m.setattr(os, "replace", racing_replace)
        a.rewrite([b"AA", b"Z"])
    assert raced and a.stale()
    assert list(a.replay()) == [b"BB", b"Z"] == _disk(p)


def test_rewrite_racing_an_external_append_is_not_served_resident(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    real_replace, raced = os.replace, []

    def racing_replace(src, dst):  # an append to the inode just published, before a's stat
        real_replace(src, dst)
        if not raced:
            raced.append(1)
            _raw_append(p, b"B")

    with monkeypatch.context() as m:
        m.setattr(os, "replace", racing_replace)
        a.rewrite([])
    assert raced and a.stale()
    assert list(a.replay()) == [b"B"] == _disk(p)


def test_file_vanishing_after_the_stat_is_not_served_resident(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    monkeypatch.setattr(a, "_fingerprint", lambda: a._stamp)  # the stat still says "unchanged" ...
    os.remove(p)  # ... but the file is gone before the tail read
    assert a.stale()
    assert list(a.replay()) == []


def test_rewrite_refuses_an_empty_payload(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    with pytest.raises(ValueError, match="non-empty"):
        a.rewrite([b"x", b""])  # a 0 length is a torn frame: it would cap the log
    assert not a.stale() and list(a.replay()) == [b"A"] == _disk(p)


def test_rewrite_publishes_from_a_private_temp_file(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    real_replace, seen = os.replace, []

    def spying_replace(src, dst):
        seen.append(src)
        real_replace(src, dst)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", spying_replace)
        a.rewrite([b"B"])
    assert seen and os.path.dirname(seen[0]) == os.path.dirname(p)
    assert seen[0] != p + ".compact.tmp"  # a fixed name lets a second rewrite overwrite the first
    assert not _leftovers(tmp_path)


def test_rewrite_waits_out_a_reader_holding_the_file(tmp_path):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    reader = open(p, "rb")  # on Windows an open handle makes os.replace fail with WinError 5
    threading.Timer(0.05, reader.close).start()
    try:
        assert a.rewrite([b"B"]) == 1
    finally:
        reader.close()
    assert not a.stale() and _served_resident(a) and list(a.replay()) == [b"B"] == _disk(p)


def test_rewrite_retries_a_transient_permission_error(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    real_replace, attempts, naps = os.replace, [], []

    def flaky_replace(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(13, "Access is denied")
        real_replace(src, dst)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", flaky_replace)
        m.setattr(time, "sleep", lambda s: naps.append(s))
        assert a.rewrite([b"B"]) == 1
    assert len(attempts) == 3 and len(naps) == 2
    assert not a.stale() and _served_resident(a) and list(a.replay()) == [b"B"] == _disk(p)


def test_rewrite_gives_up_after_the_retry_budget(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    attempts = []

    def stuck_replace(src, dst):
        attempts.append(1)
        raise PermissionError(13, "Access is denied")

    with monkeypatch.context() as m:
        m.setattr(os, "replace", stuck_replace)
        m.setattr(time, "sleep", lambda s: None)
        with pytest.raises(PermissionError):
            a.rewrite([b"B"])
    assert len(attempts) == 501
    assert not _leftovers(tmp_path)
    assert not a.stale() and list(a.replay()) == [b"A"] == _disk(p)


def test_rewrite_cleans_up_when_the_write_fails_before_publishing(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")

    def full_disk(fd):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m:
        m.setattr(os, "fsync", full_disk)
        with pytest.raises(OSError):
            a.rewrite([b"B"])
    assert not _leftovers(tmp_path)
    assert not a.stale() and _served_resident(a) and list(a.replay()) == [b"A"] == _disk(p)


def test_overlapping_rewrites_publish_from_separate_temp_files(tmp_path, monkeypatch):
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.append(b"A")
    b = BinLog(p)
    real_replace, sources, raced = os.replace, [], []

    def racing_replace(src, dst):  # a's first try is refused; b rewrites during a's retry nap
        sources.append(src)
        if not raced:
            raced.append(1)
            b.rewrite([b"BB"])
            raise PermissionError(13, "Access is denied")
        real_replace(src, dst)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", racing_replace)
        m.setattr(time, "sleep", lambda s: None)
        assert a.rewrite([b"AA"]) == 1
    assert len(set(sources)) == 2 and not _leftovers(tmp_path)
    assert list(a.replay()) == [b"AA"] == _disk(p) and not a.stale() and _served_resident(a)
    assert b.stale() and list(b.replay()) == [b"AA"]


def test_rewrite_racing_a_resident_replay_is_not_served_the_old_list(tmp_path, monkeypatch):
    """Codex B2 p1a-fix MAJOR 1: replay() took the resident list, then stale() judged whatever
    generation was current. A rewrite landing in that window swapped BOTH the list and the stamp, so
    stale() blessed the NEW pair and replay yielded the OLD list -- records already gone before this
    replay began (b's rewrite below), i.e. neither the whole old log nor the whole new one. The race
    is injected at stale()'s first I/O (the instance's stat); the rewrite runs on another thread so
    a validation that held `_lock` would show up as a timeout, not a pass."""
    a = BinLog(p := str(tmp_path / "lake.bin"))
    a.extend([b"AAAA", b"ZZZZ"])
    BinLog(p).rewrite([b"BBBB", b"ZZZZ"])  # AAAA is gone from the log before a's replay starts
    real = a._fingerprint
    fired = []

    def racing_fingerprint():  # immediately before replay's validity check
        if not fired:
            fired.append(True)
            t = threading.Thread(target=a.rewrite, args=([b"CCCC", b"ZZZZ"],))
            t.start()
            t.join(timeout=5)
            assert not t.is_alive(), "rewrite blocked: replay validates while holding the lock"
        return real()

    monkeypatch.setattr(a, "_fingerprint", racing_fingerprint)
    got = list(a.replay())
    assert fired
    assert got == [b"CCCC", b"ZZZZ"] == _disk(p), got  # never [AAAA, ZZZZ]: that log is gone
    assert list(a.replay()) == [b"CCCC", b"ZZZZ"] and _served_resident(a)  # control: current again
