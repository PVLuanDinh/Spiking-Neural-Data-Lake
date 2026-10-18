"""BinLog.replay() served from the resident copy (Phase 1a, 2026-10-17): same bytes, same order, no disk
while the file is unchanged -- and the disk again the moment another writer changes it."""

import builtins
import os

from snn_data_lake.substrate.binlog import BinLog


def _refuse_open(*a, **k):  # the constructed red: a disk read where none is allowed
    raise AssertionError("replay() read the disk on an unchanged file")


def test_resident_replay_matches_disk_and_reads_nothing(tmp_path, monkeypatch):
    p = str(tmp_path / "lake.bin")
    log = BinLog(p)
    for i in range(50):
        log.append(b"\x01rec%03d" % i)
    log.append(b"\x01rec000")  # dedup drop: must not land in the resident copy either
    log.extend([b"\x02ext%03d" % i for i in range(5)] + [b"\x02ext000"])
    disk = list(BinLog(p, dedup=False).replay())  # dedup=False keeps nothing resident: a disk read
    assert disk == [b"\x01rec%03d" % i for i in range(50)] + [b"\x02ext%03d" % i for i in range(5)]
    monkeypatch.setattr(builtins, "open", _refuse_open)
    assert list(log.replay()) == disk  # resident: zero disk reads
    monkeypatch.undo()
    log.rewrite([b"\x03keep"] + disk[:3])
    assert list(log.replay()) == [b"\x03keep"] + disk[:3] == list(BinLog(p, dedup=False).replay())


def test_external_writer_invalidates_resident_copy(tmp_path):
    p = str(tmp_path / "lake.bin")
    log = BinLog(p)
    log.append(b"\x01one")
    other = BinLog(p)  # a second writer over the same file (another process, in effect)
    other.append(b"\x01two")
    assert log.stale()
    assert list(log.replay()) == [b"\x01one", b"\x01two"]  # the disk, not the resident copy
    with open(p, "r+b") as f:  # an external truncation (a torn tail) is seen the same way
        f.truncate(os.path.getsize(p) - 2)
    assert list(log.replay()) == [b"\x01one"]
