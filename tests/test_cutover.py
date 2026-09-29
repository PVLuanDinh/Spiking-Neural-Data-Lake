"""specs/022 T005a: the cutover shutdown barrier over the INCUMBENT BinLog format.

Three properties, each with its red reachable and its positive control in the same test:
  1. A torn trailing frame (the kill-mid-append artifact) is healed EXACTLY -- every complete
     frame survives, only the torn bytes go (the heal is BinLog._repair_torn_tail, which
     predates this spec; the test pins the property the barrier depends on).
  2. The watermark round-trips, and the SHRUNK-LAKE alarm fires: a lake shorter than the last
     clean barrier refuses the boot (P4b -- the one durable lake never loses bytes).
  3. Kill-mid-batch end-to-end: barrier -> extra frames + torn tail (the crash window) ->
     reconstruction heals, validation reads crash-window (ok), never the alarm.
"""

import os
import struct

from snn_data_lake.engine import learn_daemon
from snn_data_lake.substrate.binlog import BinLog


def _mklog(path, n):
    log = BinLog(path, dedup=False)
    for i in range(n):
        log.append(f"record-{i}".encode())
    return log


def test_torn_tail_healed_exactly(tmp_path):
    path = tmp_path / "lake.bin"
    _mklog(str(path), 5)
    clean_len = os.path.getsize(path)

    # POSITIVE CONTROL first, same test: a clean file reconstructs with nothing healed.
    control = BinLog(str(path), dedup=False)
    assert len(list(control.replay())) == 5
    assert os.path.getsize(path) == clean_len, "heal must be a no-op on a clean file"

    # THE RED'S GREEN PATH: a torn frame (u32 length promising 100 bytes, only 10 present --
    # the kill-mid-append artifact) is truncated EXACTLY back to the last complete frame.
    with open(path, "ab") as f:
        f.write(struct.pack("<I", 100) + b"0123456789")
    assert os.path.getsize(path) > clean_len
    healed = BinLog(str(path), dedup=False)
    assert len(list(healed.replay())) == 5, "every complete frame must survive the heal"
    assert os.path.getsize(path) == clean_len, "only the torn bytes may be removed"


def test_watermark_roundtrip_and_shrunk_lake_alarm(tmp_path, monkeypatch):
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    lake = tmp_path / "brain_lake.bin"
    lake.write_bytes(b"x" * 1000)

    mark = learn_daemon._write_watermark()
    assert mark["clean"] is True and mark["lake_bytes"] == 1000
    on_disk = (tmp_path / "daemon.watermark").read_text()
    assert "lake_bytes=1000" in on_disk and "clean=True" in on_disk

    ok, msg = learn_daemon._validate_watermark()
    assert ok and "clean handoff" in msg

    lake.write_bytes(b"x" * 1500)  # crash window: grew past the barrier
    ok, msg = learn_daemon._validate_watermark()
    assert ok and "crash-window" in msg

    lake.write_bytes(b"x" * 200)  # THE RED: the one durable lake SHRANK
    ok, msg = learn_daemon._validate_watermark()
    assert not ok and "ALARM" in msg and "SHRANK" in msg


def test_kill_mid_batch_recovers_to_watermark(tmp_path, monkeypatch):
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    path = tmp_path / "brain_lake.bin"
    _mklog(str(path), 3)
    learn_daemon._write_watermark()
    barrier_len = os.path.getsize(path)

    # The crash window: two more complete frames land, then the process dies mid-append.
    log = BinLog(str(path), dedup=False)
    log.append(b"post-barrier-1")
    log.append(b"post-barrier-2")
    with open(path, "ab") as f:
        f.write(struct.pack("<I", 64) + b"torn")

    healed = BinLog(str(path), dedup=False)
    assert len(list(healed.replay())) == 5, "complete post-barrier frames survive; torn tail goes"
    assert os.path.getsize(path) >= barrier_len, "the heal never cuts below the barrier"
    ok, msg = learn_daemon._validate_watermark()
    assert ok and "crash-window" in msg, "a healed crash window boots; only a SHRUNK lake alarms"
