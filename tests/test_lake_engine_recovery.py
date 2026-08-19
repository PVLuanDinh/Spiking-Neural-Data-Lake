"""specs/022 T006: the consolidated recovery suite -- torn writes at EVERY offset.

test_cutover.py proves single constructed torn frames; this suite sweeps: a lake truncated at
every byte offset inside its final frame must heal to exactly the preceding complete frames,
never fewer, never a parse error. The clean positive control runs in the same test.

HONEST BOUNDARY, stated where the tests are: BinLog v1 frames carry NO per-record CRC, so
mid-file BIT CORRUPTION (not truncation) in the incumbent format is undetectable unless it
lands in a length prefix -- that gap is exactly why every record entering the NEW engine wears
the CRC envelope (lake_segments.wrap), and the envelope/segment reds live in
tests/test_lake_segments.py. This suite proves the TRUNCATION story complete; it does not
claim a corruption story the old format cannot have.
"""

import struct

from snn_data_lake.substrate.binlog import BinLog


def _frames(payloads):
    return b"".join(struct.pack("<I", len(p)) + p for p in payloads)


def test_truncation_sweep_every_offset_in_final_frame(tmp_path):
    payloads = [f"frame-{i}".encode() for i in range(4)]
    path = tmp_path / "lake.bin"
    log = BinLog(str(path), dedup=False)
    for p in payloads:
        log.append(p)
    del log
    full = path.read_bytes()
    magic_len = len(full) - len(_frames(payloads))
    last_frame_start = magic_len + len(_frames(payloads[:-1]))

    # POSITIVE CONTROL: the untruncated file heals nothing and yields all 4.
    assert len(list(BinLog(str(path), dedup=False).replay())) == 4
    assert path.stat().st_size == len(full)

    # THE SWEEP: cut at every offset inside the final frame (from its first byte to one
    # short of complete). Every cut must heal to exactly 3 frames and a stable file size.
    for cut in range(last_frame_start + 1, len(full)):
        path.write_bytes(full[:cut])
        healed = BinLog(str(path), dedup=False)
        got = list(healed.replay())
        assert len(got) == 3, f"cut at {cut}: expected 3 survivors, got {len(got)}"
        assert got == payloads[:-1], f"cut at {cut}: surviving payloads changed"
        assert path.stat().st_size == last_frame_start, f"cut at {cut}: heal not exact"


def test_truncation_at_frame_boundary_is_clean(tmp_path):
    payloads = [b"aa", b"bb", b"cc"]
    path = tmp_path / "lake.bin"
    log = BinLog(str(path), dedup=False)
    for p in payloads:
        log.append(p)
    del log
    full = path.read_bytes()
    boundary = len(full) - (4 + len(payloads[-1]))
    path.write_bytes(full[:boundary])
    got = list(BinLog(str(path), dedup=False).replay())
    assert got == payloads[:-1], "a boundary cut is a complete shorter lake, nothing to heal"
    assert path.stat().st_size == boundary


def test_zero_length_prefix_caps_the_lake(tmp_path):
    # The format's own documented semantic: a u32 length of 0 spells torn-frame; the heal
    # truncates there and everything after it is unreachable BY DESIGN (binlog PROPERTY 9).
    path = tmp_path / "lake.bin"
    log = BinLog(str(path), dedup=False)
    log.append(b"kept")
    del log
    with open(path, "ab") as f:
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 4) + b"lost")
    healed = BinLog(str(path), dedup=False)
    assert list(healed.replay()) == [b"kept"]


def test_envelope_detects_what_the_old_format_cannot(tmp_path):
    # The honest-boundary demonstration, both arms in one test: flip one payload byte in a
    # complete OLD-format frame -> replay returns the corrupted bytes silently (no CRC to
    # object). The same corruption under the NEW engine's envelope -> refused loudly.
    from snn_data_lake.substrate.lake_segments import SegmentCorruption, unwrap, wrap

    path = tmp_path / "lake.bin"
    log = BinLog(str(path), dedup=False)
    log.append(b"raw-old-format")
    del log
    data = bytearray(path.read_bytes())
    idx = data.find(b"raw-old-format")
    data[idx] ^= 0xFF
    path.write_bytes(bytes(data))
    (corrupted,) = list(BinLog(str(path), dedup=False).replay())
    assert corrupted != b"raw-old-format", "old format returns corruption silently (no CRC)"

    enveloped = bytearray(wrap(b"raw-new-format"))
    enveloped[-1] ^= 0xFF
    try:
        unwrap(bytes(enveloped))
        raise AssertionError("envelope failed to refuse a corrupted payload")
    except SegmentCorruption:
        pass
