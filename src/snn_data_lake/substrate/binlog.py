r"""
binlog.py -- a RAW-BINARY, append-only, durable knowledge log on the SSD.

home.KnowledgeLog stores learned knowledge as JSONL (text). This stores it as RAW BINARY: an 8-byte
file magic, then a sequence of length-prefixed records -- [u32 little-endian length][payload bytes].
Every append is flushed + fsync'd, so a learned fact survives a crash / power loss; replay() streams
every record back at the next boot, so a fresh PROCESS reconstructs everything learned in every prior
run (cross-process never-forget). A torn trailing record (crash mid-append) is detected and ignored, so
the log always reloads cleanly.

Payloads are arbitrary bytes -- the caller serializes. For the Spike Chat / developmental-loop fact
schema {cue:[str], answer:str, triple?:[s,r,o], src?:str}, pack_fact / unpack_fact give a compact binary
serialization (a u8 count + length-prefixed UTF-8 fields, struct-packed -- NOT JSON). So all learned
knowledge is written to the SSD as raw binary.

Zero runtime deps (struct + os). dedup=True skips a byte-identical record so the SSD stays lean.

  python -m snn_data_lake.substrate.binlog   # self-check incl. a REAL cross-process binary never-forget proof
"""
import os
import struct

_MAGIC = b"SNDLBIN1"          # 8-byte file magic: a raw-binary knowledge log, format v1
_LEN = struct.Struct("<I")    # u32 little-endian per-record length prefix


class BinLog:
    """An append-only RAW-BINARY durable log: [magic][ [u32 len][payload] ... ]. fsync per append, so a
    record survives a crash; replay() reloads every record in a fresh process (never-forget). Deduped."""

    def __init__(self, path, dedup=True):
        self.path = path
        self.dedup = bool(dedup)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "wb") as f:                       # create: write the magic header once
                f.write(_MAGIC)
                f.flush()
                os.fsync(f.fileno())
        else:
            with open(path, "rb") as f:                       # trust boundary: reject a non-binlog file
                if f.read(len(_MAGIC)) != _MAGIC:
                    raise ValueError(f"{path}: not a {_MAGIC!r} raw-binary log")
            self._repair_torn_tail()                          # a crash mid-append must not orphan future records
        payloads = list(self.replay())                        # the one boot-time full pass
        self._seen = set(payloads) if self.dedup else None
        self._count = len(payloads)                           # maintained -> count() is O(1), not a re-replay

    def _repair_torn_tail(self):
        """A crash mid-append can leave a partial length/payload frame at the end. replay() stops at it
        (short read), but a later append() writes at physical EOF -- BEHIND the torn frame -- so every
        future record would be orphaned on replay. Truncate to the last COMPLETE frame so appends land
        in a readable position. Never-forget must survive a crash, not just a clean exit."""
        valid = len(_MAGIC)
        with open(self.path, "rb") as f:
            f.seek(len(_MAGIC))
            while True:
                head = f.read(_LEN.size)
                if len(head) < _LEN.size:
                    break
                (n,) = _LEN.unpack(head)
                if n == 0 or len(f.read(n)) < n:               # zero-length or torn payload -> stop (both undecodable)
                    break
                valid = f.tell()
        if valid < os.path.getsize(self.path):
            with open(self.path, "r+b") as f:
                f.truncate(valid)

    def append(self, payload):
        """Durably append one raw-binary record (skipped if dedup and byte-identical to a stored one).
        Flush + fsync so it survives a crash. Returns True if written, False if a duplicate."""
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("BinLog.append: payload must be bytes")
        payload = bytes(payload)
        if self.dedup and payload in self._seen:
            return False
        with open(self.path, "ab") as f:
            f.write(_LEN.pack(len(payload)) + payload)         # ONE write: a second process cannot interleave
            f.flush()                                          # between the length and the payload and tear a frame
            os.fsync(f.fileno())                              # durable past process death
        if self.dedup:
            self._seen.add(payload)
        self._count += 1
        return True

    def extend(self, payloads):
        """Durably append MANY records with ONE fsync at the end -- the bulk-ingest path. The fsync, not
        the write, dominates: the self-check below measures this against per-record append() and asserts
        the speedup (~32x for 300 records there; an isolated write-only benchmark showed far more -- the
        ratio depends on the disk and the batch size, so the check asserts a floor, not a headline number).
        Same durability contract for the batch as a whole: once this returns, every record in it survives
        a crash. Skips byte-identical duplicates when dedup. Returns the number written."""
        written = 0
        with open(self.path, "ab") as f:
            for payload in payloads:
                if not isinstance(payload, (bytes, bytearray)):
                    raise TypeError("BinLog.extend: every payload must be bytes")
                payload = bytes(payload)
                if self.dedup and payload in self._seen:
                    continue
                f.write(_LEN.pack(len(payload)) + payload)     # ONE write per record (atomic-append friendly)
                if self.dedup:
                    self._seen.add(payload)
                written += 1
            if written:
                f.flush()
                os.fsync(f.fileno())                          # ONE fsync for the whole batch
        self._count += written
        return written

    def replay(self):
        """Yield every stored record's payload bytes, in append order. A torn trailing record (a crash
        mid-append) is detected by a short read and stops the replay cleanly -- the log never corrupts."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as f:
            if f.read(len(_MAGIC)) != _MAGIC:
                return
            while True:
                head = f.read(_LEN.size)
                if len(head) < _LEN.size:
                    break
                (n,) = _LEN.unpack(head)
                if n == 0:                                    # a zeroed length prefix is a torn frame, not an
                    break                                     # empty record (pack_fact never emits 0 bytes)
                payload = f.read(n)
                if len(payload) < n:                          # torn write -> stop cleanly (never-forget the rest)
                    break
                yield payload

    def count(self):
        """Number of records stored (O(1) maintained counter -- polled per round/request by daemons)."""
        return self._count


# ---- compact RAW-BINARY serialization of a knowledge fact (NOT JSON) --------------------------------
def _s(x):
    b = str(x).encode("utf-8")
    return _LEN.pack(len(b)) + b


def _read_s(buf, i):
    (n,) = _LEN.unpack_from(buf, i)
    i += _LEN.size
    return buf[i:i + n].decode("utf-8"), i + n


def pack_fact(rec):
    """Serialize a fact {cue:[str], answer:str, triple?:[s,r,o], src?:str} to RAW BINARY bytes."""
    cue = [str(c) for c in rec.get("cue", [])]
    trip = [str(t) for t in (rec.get("triple") or [])]
    out = struct.pack("<B", len(cue)) + b"".join(_s(c) for c in cue)
    out += _s(rec.get("answer", ""))
    out += struct.pack("<B", len(trip)) + b"".join(_s(t) for t in trip)
    out += _s(rec.get("src", ""))
    return out


def unpack_fact(payload):
    """Inverse of pack_fact: raw binary bytes -> {cue, answer, triple?, src?}."""
    i = 0
    (ncue,) = struct.unpack_from("<B", payload, i)
    i += 1
    cue = []
    for _ in range(ncue):
        s, i = _read_s(payload, i)
        cue.append(s)
    answer, i = _read_s(payload, i)
    (ntrip,) = struct.unpack_from("<B", payload, i)
    i += 1
    trip = []
    for _ in range(ntrip):
        s, i = _read_s(payload, i)
        trip.append(s)
    src, i = _read_s(payload, i)
    rec = {"cue": cue, "answer": answer}
    if trip:
        rec["triple"] = trip
    if src:
        rec["src"] = src
    return rec


# ---- dense float arrays (SNN weights / thresholds / decay rates / prototypes) as float16 RAW BINARY ----
def pack_array(floats):
    """Serialize a dense float array to RAW BINARY as float16 (IEEE-754 half precision, stdlib struct
    'e') -- HALVES the SSD size vs float32 for maximum sequential read, at a small precision cost. LOSSY:
    fine for STORING learned weights / prototypes / thresholds; keep float32 during active training. Layout:
    [u32 count][count x 2-byte half]. Out-of-range values are CLAMPED to the float16 limits (already a
    lossy store) so one big/inf/nan weight cannot raise OverflowError mid-write and diverge facts from
    prototypes -- storing a saturated prototype beats crashing the develop loop."""
    lim = 65504.0
    a = [0.0 if x != x else lim if x > lim else -lim if x < -lim else float(x) for x in map(float, floats)]
    return _LEN.pack(len(a)) + struct.pack(f"<{len(a)}e", *a)


def unpack_array(payload):
    """Inverse of pack_array: raw float16 binary bytes -> list[float] (widened back to python float)."""
    (n,) = _LEN.unpack_from(payload, 0)
    return list(struct.unpack_from(f"<{n}e", payload, _LEN.size))


def main():
    import shutil
    import subprocess
    import sys
    import tempfile

    work = tempfile.mkdtemp(prefix="snn_binlog_")
    path = os.path.join(work, "knowledge.bin")
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../src
    try:
        print("=" * 74)
        print("BINLOG -- raw-binary, append-only, durable knowledge log on the SSD")
        print("=" * 74)

        # ---- PROPERTY 1: fact packs to RAW BINARY and round-trips byte-exact ----
        fact = {"cue": ["capital", "france"], "answer": "paris.", "triple": ["france", "capital", "paris"],
                "src": "world-kb"}
        blob = pack_fact(fact)
        assert isinstance(blob, bytes) and unpack_fact(blob) == fact, "fact must round-trip through raw binary"

        # ---- PROPERTY 2: durable append + replay (a fresh reader reloads from disk) ----
        log = BinLog(path)
        log.append(pack_fact(fact))
        log.append(pack_fact({"cue": ["capital", "japan"], "answer": "tokyo."}))
        first = [unpack_fact(p)["answer"] for p in BinLog(path).replay()]      # a FRESH reader, no shared RAM

        # ---- PROPERTY 3: REAL cross-process binary never-forget ----
        # A SEPARATE python process appends a packed fact; then THIS process replays it from the binary file.
        child = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            "from snn_data_lake.substrate.binlog import BinLog, pack_fact\n"
            f"BinLog({path!r}).append(pack_fact({{'cue': ['largest','planet'], 'answer': 'jupiter.'}}))\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode
        after = [unpack_fact(p) for p in BinLog(path).replay()]                # reload EVERYTHING from binary

        # ---- PROPERTY 4: raw bytes on disk (magic header; NOT text/JSON) ----
        with open(path, "rb") as f:
            raw = f.read()
        starts_binary = raw[:8] == _MAGIC
        is_not_json = not raw.lstrip().startswith(b"{")

        # ---- PROPERTY 5: dedup keeps it lean ----
        dup_written = BinLog(path).append(pack_fact(fact))

        answers = [r["answer"] for r in after]
        print(f"raw file magic     : {raw[:8]!r}  ({len(raw)} bytes, binary)")
        print(f"in-process append  : {first}")
        print(f"child process      : appended 'jupiter.' in a SEPARATE python (rc={rc})")
        print(f"after restart      : {answers}  <- reloaded purely from the raw-binary file")
        print(f"dedup re-append    : wrote={dup_written} (False = duplicate skipped)")
        print("=" * 74)

        assert first == ["paris.", "tokyo."], "in-process binary append+replay lost a fact"
        assert rc == 0, f"child process failed to append (rc={rc})"
        assert answers == ["paris.", "tokyo.", "jupiter."], (
            f"cross-process BINARY never-forget FAILED: expected 3 facts across 2 processes, got {answers}")
        assert after[0] == fact, "the first fact must reload byte-exact (triple + src preserved)"
        assert starts_binary and is_not_json, "the on-disk knowledge must be RAW BINARY (magic header, not JSON)"
        assert dup_written is False and BinLog(path).count() == 3, "dedup failed: a duplicate grew the lean log"
        # torn-write tolerance: append a truncated record tail, replay must still return the 3 good records.
        with open(path, "ab") as f:
            f.write(_LEN.pack(999))                            # claim 999 bytes then write only a few -> torn
            f.write(b"xy")
        assert BinLog(path).count() == 3, "a torn trailing record must be ignored, not corrupt the replay"
        # CRASH-RECOVERY: after the torn tail, a NEW append (next boot) must survive replay -- the torn
        # frame must be REPAIRED (truncated) at construction, not left to orphan every future record.
        recovered = BinLog(path)                               # __init__ repairs the torn tail
        recovered.append(pack_fact({"cue": ["berlin"], "answer": "berlin."}))
        after = [unpack_fact(p)["answer"] for p in BinLog(path).replay()]
        assert "berlin." in after and len(after) == 4, (
            "a fact appended AFTER a crash-torn tail was orphaned -- never-forget must survive a crash")
        # a ZEROED length prefix (torn write that flushed the header only) is a torn frame, not an empty
        # record -- replay must STOP at it, not yield an undecodable b"" that crashes unpack_fact.
        with open(path, "ab") as f:
            f.write(_LEN.pack(0))
        assert BinLog(path).count() == 4 and all(len(p) > 0 for p in BinLog(path).replay()), (
            "a zero-length frame must be treated as torn (stop), never yielded as an empty record")
        # float16 pack clamps out-of-range instead of raising, so one huge weight can't crash a develop run
        assert unpack_array(pack_array([1e9, -1e9, float("nan")])) == [65504.0, -65504.0, 0.0], (
            "pack_array must clamp out-of-range / non-finite values, not raise OverflowError")

        # ---- PROPERTY 6: extend() bulk-writes with ONE fsync and is MUCH faster than per-record append ----
        import time as _t
        bulk = [pack_fact({"cue": ["bulk", str(i)], "answer": f"bulk fact {i}."}) for i in range(300)]
        p_a = os.path.join(work, "bulk_append.bin")
        p_e = os.path.join(work, "bulk_extend.bin")
        t0 = _t.time()
        la = BinLog(p_a)
        for b in bulk:
            la.append(b)                                       # fsync EVERY record
        t_append = _t.time() - t0
        t0 = _t.time()
        n_ext = BinLog(p_e).extend(bulk)                       # ONE fsync for the batch
        t_extend = _t.time() - t0
        assert n_ext == len(bulk) and BinLog(p_e).count() == len(bulk), "extend must write every record"
        assert [unpack_fact(p) for p in BinLog(p_e).replay()] == [unpack_fact(p) for p in BinLog(p_a).replay()], (
            "extend must produce a byte-identical log to append")
        assert BinLog(p_e).extend(bulk) == 0, "extend must skip duplicates when dedup is on"
        speedup = t_append / max(t_extend, 1e-9)
        print(f"bulk ingest        : append {t_append * 1000:.0f} ms vs extend {t_extend * 1000:.1f} ms "
              f"for {len(bulk)} records ({speedup:.0f}x) -- one fsync, not {len(bulk)}")
        assert speedup > 5, f"extend must be much faster than per-record append (only {speedup:.1f}x)"

        # ---- PROPERTY 7: dense weights/thresholds packed as float16 -> HALF the float32 bytes, round-trips ----
        import array as _arr
        w = [0.1, -0.5, 1.25, 3.14159, -2.71828, 0.0, 100.5, -0.001]
        pk = pack_array(w)
        back = unpack_array(pk)
        f32_bytes = len(_arr.array("f", w).tobytes())          # what float32 would cost
        assert len(back) == len(w) and all(abs(a - b) < 0.06 for a, b in zip(back, w)), (
            "float16 array must round-trip within half-precision tolerance")
        assert len(pk) - _LEN.size == 2 * len(w) and (len(pk) - _LEN.size) * 2 == f32_bytes, (
            f"float16 packing must be HALF of float32 ({len(pk) - _LEN.size} vs {f32_bytes} bytes)")
        print(f"float16 weights    : {len(w)} floats -> {len(pk) - _LEN.size} B (float32 would be {f32_bytes} B; halved)")

        print("self-check OK: knowledge is serialized to RAW BINARY (struct+utf8, not JSON), appended durably "
              "(fsync), and a fact written in one process is REPLAYED in a separate process from the binary file "
              "(never forgotten across a restart); byte-exact round-trip incl. triple/src; dedup keeps it lean; "
              "a torn trailing write is ignored. This is the SSD raw-binary knowledge store.")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    main()
