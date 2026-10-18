r"""
binlog.py -- a RAW-BINARY, append-only, durable knowledge log on the SSD.

home.KnowledgeLog stores learned knowledge as JSONL (text). This stores it as RAW BINARY: an 8-byte
file magic, then a sequence of length-prefixed records -- [u32 little-endian length][payload bytes].
Every append is flushed + fsync'd, so a learned fact survives a crash / power loss; replay() streams
every record back at the next boot, so a fresh PROCESS reconstructs everything learned in every prior
run (cross-process never-forget). A torn trailing record (crash mid-append) is detected and ignored, so
the log always reloads cleanly. A dedup instance keeps its payloads resident and serves replay() from
memory while stale() proves the file unchanged (Phase 1a, 2026-10-17; the proof hardened 2026-10-18).

Payloads are arbitrary bytes -- the caller serializes. For the Spike Chat / developmental-loop fact
schema {cue:[str], answer:str, triple?:[s,r,o], src?:str}, pack_fact / unpack_fact give a compact binary
serialization (a u8 count + length-prefixed UTF-8 fields, struct-packed -- NOT JSON). So all learned
knowledge is written to the SSD as raw binary.

Zero runtime deps (stdlib only). dedup=True skips a byte-identical record so the SSD stays lean.

  python -m snn_data_lake.substrate.binlog   # self-check incl. a REAL cross-process binary never-forget proof
"""

import os
import struct
import tempfile
import threading
import time

_MAGIC = b"SNDLBIN1"  # 8-byte file magic: a raw-binary knowledge log, format v1
_LEN = struct.Struct("<I")  # u32 little-endian per-record length prefix


class BinLog:
    """An append-only RAW-BINARY durable log: [magic][ [u32 len][payload] ... ]. fsync per append, so a
    record survives a crash; replay() reloads every record in a fresh process (never-forget). Deduped."""

    def __init__(self, path, dedup=True):
        self.path = path
        self.dedup = bool(dedup)
        self._lock = threading.Lock()  # see append(): instances are shared across threads
        self._payloads = None  # None until the boot pass below has run: replay() reads the disk
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "wb") as f:  # create: write the magic header once
                f.write(_MAGIC)
                f.flush()
                os.fsync(f.fileno())
        else:
            with open(path, "rb") as f:  # trust boundary: reject a non-binlog file
                if f.read(len(_MAGIC)) != _MAGIC:
                    raise ValueError(f"{path}: not a {_MAGIC!r} raw-binary log")
            self._repair_torn_tail()  # a crash mid-append must not orphan future records
        self._stamp = self._fingerprint()  # see stale(): taken BEFORE the one boot pass, so any
        payloads = list(self.replay())  # write landing during the pass leaves this instance stale
        self._seen = set(payloads) if self.dedup else None
        self._count = len(payloads)  # maintained -> count() is O(1), not a re-replay
        # THE RESIDENT COPY (Phase 1a of the Rust port, 2026-10-17). A dedup instance already holds
        # every payload in `_seen`; keeping the same objects in append ORDER costs one pointer per
        # record (8 bytes x 8.0M = 64 MB on the 3.46 GB operator lake) and lets replay() serve the
        # log from memory while stale() says the file is unchanged. Before this, every consumer
        # paid a full disk pass: MEASURED on the operator lake (7,998,985 records, 3,456,242,033 B,
        # warm page cache) a disk replay() costs 2.9-3.5 s and a pass over this list 0.2-0.6 s, and
        # a boot ran 7 full passes, a consolidation night 9 plus the rewrite. Not kept for
        # dedup=False (nothing is resident there, so keeping it would ADD 3.46 GB, not 64 MB):
        # those instances keep reading the disk, exactly as before.
        self._payloads = payloads if self.dedup else None

    def stale(self):
        """True if the file changed underneath this instance, so its state can no longer be trusted.

        substrate/home.py caches BinLog instances, and a cached instance NEVER re-runs
        `_repair_torn_tail()`. That is a durability hole, not an optimisation detail: the self-check
        truncates the lake to simulate a crash mid-append and then expects the next construction to
        repair it. Served from cache, the repair is skipped, the append lands BEHIND the torn frame,
        and the record is silently lost -- which is precisely what `_repair_torn_tail` exists to
        prevent. Caught by tests/test_selfchecks.py::...[substrate.home].

        Size was the whole check until 2026-10-18 (see below) because every legitimate change goes
        through `append()`, `extend()` or `rewrite()`, each of which updated `_size` under the same
        lock as the write. (This sentence named `append()` ALONE until 2026-08-05, and it was
        accurate when written and wrong by the time the other two took the lock -- extend/rewrite
        maintained no `_size` at all, so every call left this instance permanently stale. A
        docstring that names one of three writers is how the omission stayed invisible.) Any other
        size -- truncation, an external writer, a restored backup -- means this instance's
        `_seen`/`_count` describe a file that no longer exists, and the caller must rebuild rather
        than trust them.

        SIZE ALONE WAS NOT SUFFICIENT once replay() served the resident copy (Codex B1 p1a MAJOR 1+2,
        folded 2026-10-18, then a seven-lens review fold the same day). Reproduced holes: an
        equal-size `rewrite()` by another instance kept the size and so kept this one's resident
        records current-looking; a local `append()` after an external append re-read the size and
        blessed a resident copy that had never seen the external record; an equal-size rewrite
        landing between a writer's pre-write check and its post-write stat was blessed by size
        alone; and a tail check that lived only in replay() let the next local write bless a copy
        the tail check had just refused. The check is now TWO reads, both here so every reader
        (`replay()`, `home._log`, the writers' pre-write `was_current`) shares one rule, kept
        honest by ONE stamping rule:
          1. the file's identity `_fingerprint()` = (size, mtime_ns, inode) equals `_stamp`
             (`os.replace`, i.e. every `rewrite()`, changes the inode; an in-place edit the mtime);
          2. the last frame on disk (4 B + the last payload, read below; never a pass) is the
             resident copy's last payload -- needed because this box updates mtime per timer tick,
             not per write: three writes in one tick measured identical `st_mtime_ns`;
          3. THE STAMPING RULE (`_restamp`, not a read): a writer blesses a new stamp only when it was current
             BEFORE writing and the path afterwards carries the SAME inode grown by EXACTLY its own
             bytes; `rewrite()` only if the path carries the inode AND size of the temp file it
             published; construction stamps BEFORE its one pass. Anything else keeps the old stamp:
             this instance stays stale, replay() reads the disk, `home._log` rebuilds it.
        RESIDUAL, STATED: an equal-size edit of an EARLIER record that keeps the last frame, the
        inode and the mtime tick is invisible to stat + tail, and so is an in-place edit of the LAST
        record that lands inside a writer's own window (between its pre-write check and its stamp)
        -- both need an external in-place writer (reviewed 2026-10-18: no stdlib `os` field carries
        NTFS ChangeTime, `st_ctime_ns` is creation time on Windows 3.14, the lake volume's USN
        journal is inactive). No production writer edits the lake in place; the one raw in-place
        writer in the tree, `instruments/replay_hydration._plant_bout` (r+b truncate on a throwaway
        home), shrinks the file or changes its tail and is caught. Closing the residual for good
        means a writer-side generation counter in the format, not more stat fields.
        """
        try:
            if self._fingerprint() != self._stamp:
                return True
            resident = self._payloads  # None: nothing resident (dedup=False); []: a magic-only file
            if not resident:
                return False
            last = resident[-1]
            with open(self.path, "rb") as f:
                f.seek(self._stamp[0] - _LEN.size - len(last))
                return f.read(_LEN.size) != _LEN.pack(len(last)) or f.read(len(last)) != last
        except OSError:
            return True  # deleted, unreadable or torn under us: certainly not the file we replayed

    def _fingerprint(self):
        """One-stat identity; see stale() for what it cannot see."""
        st = os.stat(self.path)
        return (st.st_size, st.st_mtime_ns, st.st_ino)

    def _restamp(self, ino, size):
        """Adopt the current fingerprint only if the path carries exactly the inode and size this writer
        expects; otherwise keep the old stamp so stale() stays True."""
        try:
            now = self._fingerprint()
        except OSError:
            return
        if now[2] == ino and now[0] == size:
            self._stamp = now

    def _replace(self, tmp):
        """`os.replace(tmp, path)`, retried while a reader holds the log open: on Windows CPython opens
        without FILE_SHARE_DELETE, so a handle open on the destination makes MoveFileEx fail with
        PermissionError ([WinError 5]) -- stale()'s tail read holds one for one frame's read, a disk
        replay for a whole pass. Bounded (5 s); then the error is the caller's."""
        for _ in range(500):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.01)
        os.replace(tmp, self.path)

    def _repair_torn_tail(self):
        """A crash mid-append can leave a partial length/payload frame at the end. replay() stops at it
        (short read), but a later append() writes at physical EOF -- BEHIND the torn frame -- so every
        future record would be orphaned on replay. Truncate to the last COMPLETE frame so appends land
        in a readable position. Never-forget must survive a crash, not just a clean exit.

        `valid` IS AN ARITHMETIC RUNNING OFFSET, NOT `f.tell()`, and that is a measurement rather
        than a style preference. This loop read `valid = f.tell()` per frame until 2026-08-05.
        `BufferedReader.tell()` is `raw.tell() + buffer_offset` and `raw.tell()` is an lseek, so the
        shipped form paid ONE SYSCALL PER RECORD to re-derive a number this loop already knows: both
        `read` calls above are guarded (`len(head) < _LEN.size` and `len(f.read(n)) < n` both break),
        so when control reaches here EXACTLY `_LEN.size + n` bytes were consumed, and the sum from
        `len(_MAGIC)` is the same position `tell()` would return. MEASURED on isolated scratch lakes,
        `.venv-lake` free-threaded (3.14.6, `sys._is_gil_enabled()` False), min of 3 warmed runs:
        this scan costs **76.0 -> 25.9 ms at 100k x 64 B (2.94x), 708.1 -> 240.2 ms at 1M x 64 B
        (2.95x), and 886.4 -> 366.0 ms at 1M x 500 B / 504 MB (2.42x)**, taking a full `BinLog(path)`
        construction from 1562.1 to 1041.7 ms (1.50x) at that scale -- which is 86 percent of the
        live operator lake, 586,919,858 B measured 2026-08-05, so read it as a floor that the lake
        grows past rather than as today's exact figure. RE-VERIFIED INDEPENDENTLY the same day on
        the same env (2.98x / 2.97x / 2.34x on the scan; 2.06x / 2.03x / 1.69x on construction with
        `dedup=False`), and the mechanism was confirmed by isolating what was removed: 1,000,000
        `f.tell()` calls cost 321.8 ms against 16.5 ms for the additions that replaced them, which
        accounts for the whole 356.9 ms saved at 1M records. That per-record 305 ns is a syscall on
        an already-open fd, so it is CACHE-INDEPENDENT: a cold page cache does not shrink the
        absolute saving, it only grows the denominator. The ratios above are therefore the WARM
        (optimistic) end and the milliseconds saved are the honest invariant. The
        returned `valid` was byte-identical at all three scales and at nine constructed edge cases
        (empty file, magic only, one record, clean, torn payload, torn header, zero-length frame,
        zero-length-then-data, wrong magic). Two variants were measured and REJECTED: skipping the
        payload with `f.seek()` instead of reading it is SLOWER (2.13-2.57x, because each seek
        discards the read-ahead buffer), and merging this scan with `replay()` into one pass is
        faster still (2.3x on construction) but restructures the crash-recovery path, which is not
        a change to make in passing -- it is recorded as available, not taken."""
        valid = len(_MAGIC)
        with open(self.path, "rb") as f:
            f.seek(len(_MAGIC))
            while True:
                head = f.read(_LEN.size)
                if len(head) < _LEN.size:
                    break
                (n,) = _LEN.unpack(head)
                if (
                    n == 0 or len(f.read(n)) < n
                ):  # zero-length or torn payload -> stop (both undecodable)
                    break
                valid += _LEN.size + n  # == f.tell(); see the docstring for why it is not f.tell()
        if valid < os.path.getsize(self.path):
            with open(self.path, "r+b") as f:
                f.truncate(valid)

    def append(self, payload):
        """Durably append one raw-binary record (skipped if dedup and byte-identical to a stored one).
        Flush + fsync so it survives a crash. Returns True if written, False if a duplicate."""
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("BinLog.append: payload must be bytes")
        if not payload:
            # A u32 length of 0 is how this FORMAT spells "torn frame", not "empty record":
            # replay() stops at one and _repair_torn_tail() TRUNCATES there. Writing one is
            # therefore not storing nothing, it is capping the lake -- measured 2026-08-05, five
            # records appended after an accepted append(b"") were destroyed by the next
            # construction (362 -> 188 bytes, 0/5 survived) while count() reported all eleven.
            # See PROPERTY 9 in main() for the constructed red. Refused here rather than at the
            # callers because the one durable lake is the thing being protected.
            raise ValueError(
                "BinLog.append: payload must be non-empty (a 0 length is a torn frame)"
            )
        payload = bytes(payload)
        # A BinLog is now SHARED (substrate/home.py caches one per path, because constructing one
        # costs a full tail-repair scan plus a full replay -- 708 ms on a 425 MB lake). Shared means
        # the daemon's tick thread and its request threads can append concurrently, so the
        # check-then-write and the bookkeeping have to be one critical section: without it two
        # threads can both miss the same payload in `_seen` and write it twice, which is exactly
        # the duplicate `dedup` exists to prevent.
        with self._lock:
            if self.dedup and payload in self._seen:
                return False
            was_current = not self.stale()  # see _restamp(): a write never blesses a stale copy
            with open(self.path, "ab") as f:
                f.write(
                    _LEN.pack(len(payload)) + payload
                )  # ONE write: a second process cannot interleave
                f.flush()  # between the length and the payload and tear a frame
                os.fsync(f.fileno())  # durable past process death
            if self.dedup:
                self._seen.add(payload)
                self._payloads.append(payload)  # the resident copy follows the disk, in order
            self._count += 1
            if was_current:  # keep stale() honest about our own writes
                self._restamp(self._stamp[2], self._stamp[0] + _LEN.size + len(payload))
        return True

    def extend(self, payloads):
        """Durably append MANY records with ONE fsync at the end -- the bulk-ingest path. The fsync, not
        the write, dominates: the self-check below measures this against per-record append() and asserts
        the speedup (~32x for 300 records there; an isolated write-only benchmark showed far more -- the
        ratio depends on the disk and the batch size, so the check asserts a floor, not a headline number).
        Same durability contract for the batch as a whole: once this returns, every record in it survives
        a crash. Skips byte-identical duplicates when dedup. Returns the number written.

        THE CROSSOVER IS N=2, MEASURED, AND THERE IS NO WIDTH AT WHICH THIS LOSES. Swept on isolated
        scratch lakes, `.venv-lake` free-threaded (3.14.6), N x `append()` against ONE `extend(N)`,
        every pair verified byte-identical on disk: N=1 **1.0x** (a tie -- one record, one fsync,
        the same call), N=2 1.9x, N=4 3.5x, N=8 7.3x, N=16 9.9x, N=32 21.3x, N=64 34.8x, N=128
        69.7x, N=256 81.6x, N=1024 346.7x, N=4096 1084x, N=10,000 1341x. Per-record cost falls
        738.8 -> 0.334 us and FLATTENS at 0.264 us/rec from N=100,000 (identical at N=1,000,000),
        which is the floor left once the fsync is fully amortised. For contrast a per-record
        `append()` measured 459.0 us/rec with the same payloads, of which **fsync is 77.9 percent**,
        open/close 13.4 percent, `os.path.getsize` 3.2 percent and the lock 0.015 percent (since
        2026-10-18 `append()` stats twice, before and after the write, and reads one tail frame;
        the breakdown was not re-measured). So the gate on batching is NOT a width -- it is the
        durability paragraph below.

        READ THE MULTIPLES AS AN ORDER, NOT AS CONSTANTS. Re-swept independently 2026-08-05 on the
        same machine and env: the SHAPE reproduced exactly -- N=1 a tie (1.00x on min, 0.99x on
        median over 60 alternating trials, so the "never loses" claim survives the single-shot
        noise that first showed 0.8x there), first win at N=2, monotone after, byte-identical on
        disk at every N -- and `append()` re-measured at 452 us/rec against the 459 above. But the
        RATIOS came in roughly half: 1.4x at N=2, 2.1x at N=4, 11.0x at N=32, 196.7x at N=1024.
        The whole difference is on the `extend()` side, which is fsync-bound and therefore a
        property of the disk's state on the day, not of this code. What is stable and load-bearing
        is the crossover (N=2), the monotonicity, and the byte-equality; a specific multiple is
        not, so do not build a threshold on one.

        WHAT BATCHING COSTS, STATED AND BOUNDED, because it is a real durability trade and not a
        free win. `append()` fsyncs per record, so a record is durable the instant its own call
        RETURNS. `extend()` fsyncs once, so NO record in the batch is durable until the whole call
        returns, and a crash before that can lose **any subset of the batch, up to all N of them**
        (the OS may have flushed some pages and not others; nothing here orders them). The bound is
        therefore the BATCH, and for a caller that accumulates records over time the window is its
        buffering interval, not its batch size. Two consequences worth naming:
          * batching is safe where the records already exist before the call -- a bulk import, a
            night's derived schema triples, a replayed corpus -- because the window is microseconds
            and the caller had nothing durable to lose during it;
          * batching is NOT safe by buffering ACROSS ticks anything whose loss breaks P4b's "a
            restart is not amnesia". Autobiographical episodes (`home.save_episode`, one record per
            wake tick) are exactly that: buffering 32 of them to earn the 21.3x above would mean a
            crash costs the brain 32 ticks of memory it believes it formed. Do not.

        LIVE CALLERS TODAY: NONE. Measured 2026-08-05 -- the only `.extend(` on a lake object
        anywhere in `src/`, `tests/` or `tools/` is `home.KnowledgeLog.extend` (which forwards
        here) and this module's own self-check, and `KnowledgeLog.extend` has no caller either. So
        the batched mechanism is built, is up to 1341x faster, and is P3 shape 1 (method never
        called) while the unbatched path is the one in use. The nearest live N-per-tick append loop
        is `engine/brain.py`'s schema induction, which calls `log.append(...)` once per derived
        triple under `budget=32` on every fourth night -- N up to 32, measured 21.3x, and it is the
        record set this paragraph's first bullet describes. Changing it is a one-line edit in a file
        this pass does not own; it is reported, not made."""
        written = grown = 0
        # SAME CRITICAL SECTION AS append(), added 2026-08-04. This method held no lock while
        # append() held one, and the justification append() gives applies here verbatim: the
        # instance is SHARED (substrate/home.py caches one per path), so an unlocked
        # check-then-write on `_seen` lets two threads both miss the same payload and store it
        # twice -- the duplicate `dedup` exists to prevent. It also did not maintain `_size`,
        # so every extend() left the cached instance permanently `stale()` and forced home._log()
        # to rebuild it (a full tail-repair scan plus a full replay) on the very next call,
        # defeating the cache the comment above it exists to justify. RE-MEASURED 2026-08-05 with
        # the conditions stated, because "708 ms on the live lake" was inherited from a comment
        # written against a 425 MB / 80,853-record lake and quoted here as a present-tense fact:
        # on the live lake as it stands TODAY (513.1 MB, 121,177 records, .venv-lake, three
        # consecutive constructions) `BinLog(path)` costs 1316 / 1420 / 1307 ms. The cost is a
        # function of lake SIZE, so quote the derivation, not the number -- it grows as the brain
        # learns, which is the direction that makes losing the cache worse over time, not better.
        with self._lock:
            was_current = not self.stale()  # see _restamp(): a write never blesses a stale copy
            with open(self.path, "ab") as f:
                for payload in payloads:
                    if not isinstance(payload, (bytes, bytearray)):
                        raise TypeError("BinLog.extend: every payload must be bytes")
                    if not payload:  # same torn-frame refusal as append(); see PROPERTY 9
                        raise ValueError(
                            "BinLog.extend: every payload must be non-empty (a 0 length is a torn frame)"
                        )
                    payload = bytes(payload)
                    if self.dedup and payload in self._seen:
                        continue
                    f.write(
                        _LEN.pack(len(payload)) + payload
                    )  # ONE write per record (atomic-append friendly)
                    if self.dedup:
                        self._seen.add(payload)
                        self._payloads.append(payload)  # resident copy follows the disk
                    written += 1
                    grown += _LEN.size + len(payload)
                if written:
                    f.flush()
                    os.fsync(f.fileno())  # ONE fsync for the whole batch
            self._count += written
            if was_current and grown:  # keep stale() honest about our own writes
                self._restamp(self._stamp[2], self._stamp[0] + grown)
        return written

    def replay(self):
        """Yield every stored record's payload bytes, in append order. A torn trailing record (a crash
        mid-append) is detected by a short read and stops the replay cleanly -- the log never corrupts.

        THIS DOES NOT PARALLELISE, AND THAT IS A STRUCTURAL RESULT, NOT A WIDTH TO OUTGROW. Swept
        2026-08-05 on `.venv-lake` free-threaded (3.14.6, `sys._is_gil_enabled()` False, 28 cores)
        and re-run independently the same day: a sequential frame-offset scan followed by DISJOINT
        frame ranges sliced per thread, against the best serial parse, at thread counts
        {2,4,8,14,28}. Best parallel time at each width, as a fraction of serial:
        10k x 64 B **0.54x**, 100k x 64 B 0.57x, 1M x 64 B 0.53x, 100k x 500 B 0.71x,
        1M x 500 B / 504 MB **0.59x**. More threads never helped at any width. Parallel output was
        compared element-by-element against serial at every width and thread count and was
        identical, so this is a speed null and not a correctness one.
        CAUSE: each record's u32 length determines the NEXT offset, so frame parsing is inherently
        sequential; only the slice-and-copy half can be split and that half is memory-bandwidth
        bound. Adding cores does not add bandwidth. Unlike the neuron/synapse updates this build
        parallelises, there is no independence here to exploit -- so do not gate it on a width.
        THE NULL IS NOT AN ARTEFACT OF A DEAD POOL: the same threading pattern running a CPU-bound
        spin scaled 2.00x / 3.80x / 6.49x / 8.23x / 9.68x at 2 / 4 / 8 / 14 / 28 threads in the
        same process, and a per-thread start/stop witness recorded up to 27 genuinely concurrent
        workers during the replay runs. The pool runs; the work does not divide.

        SERVED FROM THE RESIDENT COPY WHILE THE FILE IS UNCHANGED (Phase 1a, 2026-10-17). A
        constructed dedup instance holds every payload in order (see `__init__`); if `stale()` says
        the file on disk is still the one this instance wrote, that list IS the log and is yielded
        as-is -- the same append order, the same objects, no disk pass (since the 2026-10-18 fold
        `stale()` reads one tail frame to say so). Validity is the one rule `substrate/home._log`
        already trusts to hand out a cached instance (`stale()`: fingerprint and tail unchanged by
        any writer but this one); a stale instance, an undedup'd one, or a bare `BinLog.__new__` falls
        through to the disk scan below, which is unchanged. The disk is the source of truth and the
        list is a cache over it (P4b's own wording), so an external truncation, restore or append
        reaches every reader through the same `stale()` (stat + tail frame) that already
        invalidates `_seen` and `_count`."""
        resident = getattr(self, "_payloads", None)  # a bare __new__ instance has no copy
        if resident is not None and not self.stale():
            yield from resident
            return
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
                if n == 0:  # a zeroed length prefix is a torn frame, not an
                    break  # empty record (pack_fact never emits 0 bytes)
                payload = f.read(n)
                if len(payload) < n:  # torn write -> stop cleanly (never-forget the rest)
                    break
                yield payload

    def count(self):
        """Number of records stored (O(1) maintained counter -- polled per round/request by daemons)."""
        return self._count

    def rewrite(self, survivors):
        """LOG COMPACTION -- atomically rewrite the whole log to exactly `survivors` (an iterable of payload
        bytes, in the order given). This is the mechanism the brain's SLEEP consolidation uses to PRUNE +
        CONSOLIDATE the durable lake: replay -> a policy selects/merges survivors -> rewrite. Atomic (temp
        file + fsync + os.replace), so a crash mid-compaction leaves the OLD log intact -- never a half lake.
        Returns the new record count.

        NOTE FOR CALLERS THAT CACHE ANYTHING DERIVED FROM THIS LOG: `_stamp` (`_size` until
        2026-10-18) is re-taken here, so `stale()` deliberately reports False after a rewrite --
        this instance IS current, it is the one that wrote the file (unless another writer touched
        the file between the swap and the stat; then the stamp is kept and `stale()` stays True). A
        caller that hung a derived index off this instance therefore gets no signal that the records
        it was built from are gone, and has to drop it itself; `home.KnowledgeLog.consolidate` does
        exactly that for its `_snn_state_index`. Clearing it here instead would be the wrong side of
        the boundary -- a payload is opaque bytes to BinLog, and it stays that way."""
        survivors = [bytes(p) for p in survivors]
        if not all(survivors):  # same torn-frame refusal as append()/extend(); see PROPERTY 9
            raise ValueError(
                "BinLog.rewrite: every payload must be non-empty (a 0 length is a torn frame)"
            )
        # a private name per call: a fixed name lets a concurrent rewrite publish its bytes under
        # another's inode
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(self.path) or ".",
            prefix=os.path.basename(self.path) + ".",
            suffix=".compact.tmp",
        )
        # UNDER THE SAME LOCK as append()/extend(), added 2026-08-04. This ran unlocked on a SHARED
        # instance while the sleep tick calls it (engine/brain.py, `klog.consolidate(survivors)`).
        # An append that landed between building `tmp` and the os.replace was written to the file
        # this call is about to REPLACE, so the record was durably written and then durably
        # destroyed -- on the one lake, which is the store P4b says is part of the brain. Today
        # every lake writer runs on the daemon's tick thread, so this was latent rather than live;
        # it is the kind of latency that ends the moment a second writer is added, and append()'s
        # own docstring already assumes request threads write here.
        with self._lock:
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(_MAGIC)
                    for p in survivors:
                        f.write(_LEN.pack(len(p)) + p)
                    f.flush()
                    os.fsync(f.fileno())
                    published = os.fstat(f.fileno())  # the identity and extent the path will carry
                self._replace(tmp)
            # BaseException: ENOSPC, a refused swap or an interrupt must not leave a lake-sized temp
            # file behind (unique names pile up)
            except BaseException:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            self._seen = set(survivors) if self.dedup else None
            self._payloads = survivors if self.dedup else None  # the resident copy IS the new log
            self._count = len(survivors)
            self._restamp(published.st_ino, published.st_size)
        return self._count


# ---- compact RAW-BINARY serialization of a knowledge fact (NOT JSON) --------------------------------
def _s(x):
    b = str(x).encode("utf-8")
    return _LEN.pack(len(b)) + b


def _read_s(buf, i):
    (n,) = _LEN.unpack_from(buf, i)
    i += _LEN.size
    return buf[i : i + n].decode("utf-8"), i + n


def pack_fact(rec):
    """Serialize a fact {cue:[str], answer:str, triple?:[s,r,o], src?:str} to RAW BINARY bytes."""
    cue = [str(c) for c in rec.get("cue", [])]
    trip = [str(t) for t in (rec.get("triple") or [])]
    out = struct.pack("<B", len(cue)) + b"".join(_s(c) for c in cue)
    out += _s(rec.get("answer", ""))
    out += struct.pack("<B", len(trip)) + b"".join(_s(t) for t in trip)
    out += _s(rec.get("src", ""))
    return out


def pack_episode(episode):
    """Serialize an episode dict {role: filler, ...} to RAW BINARY bytes.

    Episodes are role=filler slot maps (who/what/where/when/affect), not facts, so they get their
    own encoding rather than being bent into pack_fact's cue/answer/triple shape. Slots are written
    in SORTED order so the same episode always produces the same bytes -- BinLog dedups on the
    payload, and an unstable field order would store the same memory repeatedly.
    """
    slots = sorted((str(k), str(v)) for k, v in dict(episode).items())
    out = struct.pack("<B", min(len(slots), 255))
    for k, v in slots[:255]:
        out += _s(k) + _s(v)
    return out


def unpack_episode(payload):
    """Inverse of pack_episode: raw binary bytes -> {role: filler, ...}."""
    i = 0
    (n,) = struct.unpack_from("<B", payload, i)
    i += 1
    out = {}
    for _ in range(n):
        k, i = _read_s(payload, i)
        v, i = _read_s(payload, i)
        out[k] = v
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
    a = [
        0.0 if x != x else lim if x > lim else -lim if x < -lim else float(x)
        for x in map(float, floats)
    ]
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
    src_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )  # .../src
    try:
        print("=" * 74)
        print("BINLOG -- raw-binary, append-only, durable knowledge log on the SSD")
        print("=" * 74)

        # ---- PROPERTY 1: fact packs to RAW BINARY and round-trips byte-exact ----
        fact = {
            "cue": ["capital", "france"],
            "answer": "paris.",
            "triple": ["france", "capital", "paris"],
            "src": "world-kb",
        }
        blob = pack_fact(fact)
        assert isinstance(blob, bytes) and unpack_fact(blob) == fact, (
            "fact must round-trip through raw binary"
        )

        # ---- PROPERTY 2: durable append + replay (a fresh reader reloads from disk) ----
        log = BinLog(path)
        log.append(pack_fact(fact))
        log.append(pack_fact({"cue": ["capital", "japan"], "answer": "tokyo."}))
        first = [
            unpack_fact(p)["answer"] for p in BinLog(path).replay()
        ]  # a FRESH reader, no shared RAM

        # ---- PROPERTY 3: REAL cross-process binary never-forget ----
        # A SEPARATE python process appends a packed fact; then THIS process replays it from the binary file.
        child = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            "from snn_data_lake.substrate.binlog import BinLog, pack_fact\n"
            f"BinLog({path!r}).append(pack_fact({{'cue': ['largest','planet'], 'answer': 'jupiter.'}}))\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode
        after = [unpack_fact(p) for p in BinLog(path).replay()]  # reload EVERYTHING from binary

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
            f"cross-process BINARY never-forget FAILED: expected 3 facts across 2 processes, got {answers}"
        )
        assert after[0] == fact, "the first fact must reload byte-exact (triple + src preserved)"
        assert starts_binary and is_not_json, (
            "the on-disk knowledge must be RAW BINARY (magic header, not JSON)"
        )
        assert dup_written is False and BinLog(path).count() == 3, (
            "dedup failed: a duplicate grew the lean log"
        )
        # torn-write tolerance: append a truncated record tail, replay must still return the 3 good records.
        with open(path, "ab") as f:
            f.write(_LEN.pack(999))  # claim 999 bytes then write only a few -> torn
            f.write(b"xy")
        assert BinLog(path).count() == 3, (
            "a torn trailing record must be ignored, not corrupt the replay"
        )
        # CRASH-RECOVERY: after the torn tail, a NEW append (next boot) must survive replay -- the torn
        # frame must be REPAIRED (truncated) at construction, not left to orphan every future record.
        recovered = BinLog(path)  # __init__ repairs the torn tail
        recovered.append(pack_fact({"cue": ["berlin"], "answer": "berlin."}))
        after = [unpack_fact(p)["answer"] for p in BinLog(path).replay()]
        assert "berlin." in after and len(after) == 4, (
            "a fact appended AFTER a crash-torn tail was orphaned -- never-forget must survive a crash"
        )
        # a ZEROED length prefix (torn write that flushed the header only) is a torn frame, not an empty
        # record -- replay must STOP at it, not yield an undecodable b"" that crashes unpack_fact.
        with open(path, "ab") as f:
            f.write(_LEN.pack(0))
        assert BinLog(path).count() == 4 and all(len(p) > 0 for p in BinLog(path).replay()), (
            "a zero-length frame must be treated as torn (stop), never yielded as an empty record"
        )
        # float16 pack clamps out-of-range instead of raising, so one huge weight can't crash a develop run
        assert unpack_array(pack_array([1e9, -1e9, float("nan")])) == [65504.0, -65504.0, 0.0], (
            "pack_array must clamp out-of-range / non-finite values, not raise OverflowError"
        )

        # ---- PROPERTY 6: extend() bulk-writes with ONE fsync and is MUCH faster than per-record append ----
        import time as _t

        bulk = [
            pack_fact({"cue": ["bulk", str(i)], "answer": f"bulk fact {i}."}) for i in range(300)
        ]
        p_a = os.path.join(work, "bulk_append.bin")
        p_e = os.path.join(work, "bulk_extend.bin")
        t0 = _t.time()
        la = BinLog(p_a)
        for b in bulk:
            la.append(b)  # fsync EVERY record
        t_append = _t.time() - t0
        t0 = _t.time()
        n_ext = BinLog(p_e).extend(bulk)  # ONE fsync for the batch
        t_extend = _t.time() - t0
        assert n_ext == len(bulk) and BinLog(p_e).count() == len(bulk), (
            "extend must write every record"
        )
        assert [unpack_fact(p) for p in BinLog(p_e).replay()] == [
            unpack_fact(p) for p in BinLog(p_a).replay()
        ], "extend must produce a byte-identical log to append"
        assert BinLog(p_e).extend(bulk) == 0, "extend must skip duplicates when dedup is on"
        speedup = t_append / max(t_extend, 1e-9)
        print(
            f"bulk ingest        : append {t_append * 1000:.0f} ms vs extend {t_extend * 1000:.1f} ms "
            f"for {len(bulk)} records ({speedup:.0f}x) -- one fsync, not {len(bulk)}"
        )
        assert speedup > 5, (
            f"extend must be much faster than per-record append (only {speedup:.1f}x)"
        )

        # ---- PROPERTY 7: dense weights/thresholds packed as float16 -> HALF the float32 bytes, round-trips ----
        import array as _arr

        w = [0.1, -0.5, 1.25, 3.14159, -2.71828, 0.0, 100.5, -0.001]
        pk = pack_array(w)
        back = unpack_array(pk)
        f32_bytes = len(_arr.array("f", w).tobytes())  # what float32 would cost
        assert len(back) == len(w) and all(abs(a - b) < 0.06 for a, b in zip(back, w)), (
            "float16 array must round-trip within half-precision tolerance"
        )
        assert len(pk) - _LEN.size == 2 * len(w) and (len(pk) - _LEN.size) * 2 == f32_bytes, (
            f"float16 packing must be HALF of float32 ({len(pk) - _LEN.size} vs {f32_bytes} bytes)"
        )
        print(
            f"float16 weights    : {len(w)} floats -> {len(pk) - _LEN.size} B (float32 would be {f32_bytes} B; halved)"
        )

        # ---- PROPERTY 8: ONE BinLog, MANY THREADS -- append() is one critical section -----------
        # `self._lock` has guarded append()/extend()/rewrite() since 2026-08-04 and NOTHING in this
        # tree could fail if it were deleted. `substrate/home.py` carries a race probe, but it
        # covers `_LOG_LOCK` -- the CACHE that decides how many BinLog objects exist -- and passes
        # with a single shared instance whose own writer is unguarded. This is the criterion for
        # the WRITER, and P8's rule is why it is here: an unmeasured guard on the one durable store
        # is a claim, not a mechanism.
        #
        # REACHABLE RED, CONSTRUCTED AND RUN 2026-08-05 on the free-threaded `.venv-lake` (3.14.6,
        # `sys._is_gil_enabled()` False, 28 cores), 40 trials x 16 threads on isolated scratch
        # lakes, with `self._lock` swapped for a no-op context manager:
        #     append: 40/40 trials stored the SAME payload more than once (worst 5 copies on disk),
        #             and 40/40 had `_count` disagree with the lake (`+=` is not atomic);
        #     extend: 0/40 duplicated -- ONE record per call leaves almost no window -- but 3/40
        #             still lost a `_count` update, so its lock is load-bearing on a narrower
        #             observable rather than on none.
        # With the lock held: 0/40 and 0/40, both modes. That asymmetry is why the assert below
        # tests append() and checks BOTH the lake and the counter.
        #
        # The flakiness can only produce a FALSE NEGATIVE (miss a broken lock), never a false
        # positive: with the lock held, one record is the invariant and not the likely outcome. A
        # green here is weak evidence; a red is conclusive. The measured per-trial red rate is
        # 40/40, so the trial count is small on purpose -- it buys margin, not sensitivity.
        _LOCK_TRIALS, _LOCK_THREADS = 6, 12
        for _trial in range(_LOCK_TRIALS):
            race_path = os.path.join(work, f"lock_race{_trial}.bin")
            shared = BinLog(race_path, dedup=True)
            victim = pack_fact({"cue": ["race", str(_trial)], "answer": "written exactly once."})
            gate = threading.Barrier(_LOCK_THREADS)

            def _racer(log=shared, v=victim, g=gate):
                g.wait()  # release every thread together, so the window is actually contended
                log.append(v)

            threads = [threading.Thread(target=_racer) for _ in range(_LOCK_THREADS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            on_disk = sum(1 for p in BinLog(race_path).replay() if p == victim)
            assert on_disk == 1, (
                f"trial {_trial}: {_LOCK_THREADS} threads appending one payload to ONE BinLog "
                f"stored it {on_disk} times -- the check-then-write on _seen is not a critical "
                "section, so dedup does not hold and the one durable lake carries duplicates"
            )
            assert shared.count() == 1, (
                f"trial {_trial}: the lake holds 1 record but count() reports {shared.count()} -- "
                "`_count += 1` lost an update, so an O(1) counter the daemon polls per round "
                "disagrees with the store it describes"
            )
        print(
            f"concurrent append  : {_LOCK_TRIALS} trials x {_LOCK_THREADS} threads on ONE BinLog "
            f"-> 1 record, count()==1 (dedup + counter held under free threading)"
        )

        # ---- PROPERTY 9: the WRITER must refuse a zero-length payload ---------------------------
        # A u32 length of 0 is how this format spells "torn frame": replay() stops at one and
        # _repair_torn_tail() TRUNCATES there. The torn-tail checks above test that from the
        # CORRUPTION side -- an externally written zero prefix must stop a replay. Nothing tested
        # the other side, that BinLog's own writer cannot MAKE one, and until 2026-08-05 it could.
        #
        # REACHABLE RED, CONSTRUCTED AND RUN 2026-08-05 with the two guards removed, isolated
        # scratch lake: 5 real records, then append(b"") -> returned True (a durable-write claim),
        # then 5 more real records. count() reported 11; a fresh reader replayed 5; the next
        # construction truncated the file 362 -> 188 bytes and 0/5 of the records written after
        # the empty one survived. extend([...b"" in the middle...]) lost 4 of 7 the same way.
        # With the guards in place both calls raise and nothing reaches the file. That is P4b's
        # "a restart is not amnesia" failing on a write the API ACCEPTED, which is why the refusal
        # belongs at the writer and not in a caller's docstring.
        #
        # LATENT, NOT LIVE, AND SAID PLAINLY: no caller can hit it today -- every home.py record
        # carries a 1-byte type tag and the smallest packer output (pack_episode({})) is 1 byte --
        # so this closes a hole in the store rather than fixing an observed loss.
        p_zero = os.path.join(work, "zero_len.bin")
        zl = BinLog(p_zero)
        for i in range(3):
            zl.append(pack_fact({"cue": ["keep", str(i)], "answer": f"keep {i}."}))
        for label, call in (
            ("append", lambda: zl.append(b"")),
            ("extend", lambda: zl.extend([b""])),
        ):
            try:
                call()
                refused = False
            except ValueError:
                refused = True
            assert refused, (
                f"BinLog.{label} accepted a ZERO-LENGTH payload. That frame is indistinguishable "
                "from a torn write, so replay() stops at it and the next construction TRUNCATES "
                "the lake there -- every record appended afterwards is destroyed while count() "
                "still reports them. The one durable lake must refuse the write, not absorb it"
            )
        zl.append(pack_fact({"cue": ["after"], "answer": "after the refusal."}))
        kept = [unpack_fact(p)["answer"] for p in BinLog(p_zero).replay()]
        assert len(kept) == 4 and kept[-1] == "after the refusal." and zl.count() == 4, (
            f"a refused zero-length payload must leave the lake untouched: expected 4 records with "
            f"the post-refusal one last, got {kept} and count()={zl.count()}"
        )
        print(
            f"zero-length payload: append/extend both refuse it; lake intact at {len(kept)} records "
            f"(a 0 length prefix is a torn frame, so writing one caps the log)"
        )

        print(
            "self-check OK: knowledge is serialized to RAW BINARY (struct+utf8, not JSON), appended durably "
            "(fsync), and a fact written in one process is REPLAYED in a separate process from the binary file "
            "(never forgotten across a restart); byte-exact round-trip incl. triple/src; dedup keeps it lean; "
            "a torn trailing write is ignored. This is the SSD raw-binary knowledge store."
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    main()
