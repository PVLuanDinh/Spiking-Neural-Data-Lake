"""specs/022 T102 (R1) P3 -- the SUCCESSOR deep tier: a derived token -> frame-offset index over the
episodes in the one lake, answering the exact sub-cue the whole-lake scan answers, newest first.

Pre-registration: specs/022-unified-brain-arc/evidence/r1-bounded-recall.md 2.11 P3 and 2.19.
DORMANT BY DESIGN until the deciding run adopts it (P3): nothing in `engine/` calls it; on KEEP the
module is reverted. It is exempt-registered with that reason.

THE INDEX IS DERIVED STATE. Rows live in `LakeIndex` under the reserved tag `TAG` (256, outside the byte
range of any record tag): one pointer row per (slot=value token, frame offset), keyed
`<token hash>:<offset hex>:<frame check>`, so a token's postings sort by offset and "newest first" is a
reverse range read; the frame check (8 hex of the frame's payload) makes a row that points at bytes that
are not the frame it was written for refuse instead of answer. The lake is the only source of truth: every answer is VERIFIED against the record bytes read back
from the lake (`BinLog.read_at` + `lake_recall.is_subcue`) -- an index row is a candidate, never an
answer (constraint C4). The builder reads THROUGH a BinLog INSTANCE it is handed (live: `home._log()`'s
cached one), never constructing a second (C2), and uses `BinLog.replay_located` for framing (C1).
"""

from __future__ import annotations

import contextlib
import hashlib
import os

from snn_data_lake.substrate.binlog import _LEN, unpack_episode
from snn_data_lake.substrate.lake_index import LakeIndex
from snn_data_lake.substrate.lake_recall import is_subcue

TAG = 0x100
_CONTAINER = "lake"
_BATCH = 5000  # records folded per index transaction
_META_KEYS = (
    "ep_watermark",
    "ep_anchor_off",
    "ep_anchor_sha",
    "ep_rows",
    "ep_digest",
    "ep_lake_id",
    "ep_fmt",
)
FMT = "2"  # the posting-row format; an index of another format is rebuilt (derived state, no migration)


def _tok(slot, value):
    return hashlib.blake2b(f"{slot}\x00{value}".encode(), digest_size=8).hexdigest()


def _chk(payload):
    return hashlib.blake2b(payload, digest_size=4).hexdigest()


def _key(tok, offset, chk):
    return f"{tok}:{offset:016x}:{chk}"


def _bounds(tok):
    return f"{tok}:", f"{tok};"  # ';' sorts right after ':' -- every offset of this token


_MOD = 1 << 64


def _row_hash(key, seq):
    return int.from_bytes(hashlib.blake2b(f"{key}|{seq}".encode(), digest_size=8).digest(), "big")


def _anchor(last, end, nrows, digest, lake_id):
    return {
        "ep_watermark": end,
        "ep_anchor_off": last[0],
        "ep_anchor_sha": last[1],
        "ep_rows": nrows,
        "ep_digest": digest,
        "ep_lake_id": lake_id,
        "ep_fmt": FMT,
    }


def decode(payload):
    """The AUTOBIOGRAPHICAL episode a record carries, else None (a learner event, a fact, a foreign
    blob). The kind lives inside the payload, so every record is decoded to be classified."""
    from snn_data_lake.substrate import home as H

    if payload[:1] != H._T_EPISODE:
        return None
    try:
        ep = unpack_episode(payload[1:])
    except Exception:  # noqa: BLE001 - an unreadable record is skipped, as load_episodes skips it
        return None
    return ep if H.is_autobiographical_episode(ep) else None


class IndexCorrupt(RuntimeError):
    """An index row or count disagrees with itself; the caller refuses visibly (S6) or rebuilds."""


# distinct-from: substrate/lake_recall.py::LakeRecall -- LAESA similarity over the WINDOW tier (its
#   `is_subcue` is reused below for verification); substrate/episodic_recall.py -- spike-coincidence ranking;
#   home.load_episodes -- the whole-lake scan this index is the successor candidate for. Sites read.
class EpisodeIndex:
    """`log`: a BinLog INSTANCE (never a path). `index_path`: where the derived rows live."""

    def __init__(self, log, index_path):
        self.log = log
        self._path = index_path
        # opened on first use, inside recall's refusal net (a damaged file refuses)
        self._idx = None
        self._checked = False  # the rows are verified once, then trusted for folds ...
        self._seen_version = None  # ... until another connection changes the index file (M7)

    @property
    def idx(self):
        if self._idx is None:
            self._idx = LakeIndex(self._path)
        return self._idx

    def close(self):
        if self._idx is not None:
            self._idx.close()
            self._idx = None  # the next use reopens instead of refusing on a closed connection

    # -- compaction detection (S5): offsets live in the rows, and a rewrite of the lake moves every one.
    def _lake_id(self):
        """The lake file's identity (device:inode). `BinLog.rewrite` writes a temp file and `os.replace`s
        it, so every compaction gives the lake a new id even when the last frame stays put. An id of
        "0" (a filesystem without stable ids) never compares, and the anchor alone applies."""
        st = os.stat(self.log.path)
        return f"{st.st_dev}:{st.st_ino}"

    def _anchor_ok(self):
        """True when the last folded frame is still where the index says it is, byte for byte. A
        compaction (`BinLog.rewrite`, `KnowledgeLog.consolidate`) or a truncation moves or removes it;
        `BinLog.stale()` cannot see a same-size rewrite. Limit, stated: the lake file's identity
        (below) catches a replacement, the last frame catches an in-place change; an identity reused by
        two rewrites between syncs AND a last frame left at the same offset passes."""
        wm = self._wm()
        if wm == 0:
            return True
        was, now = self.idx.meta("ep_lake_id"), self._lake_id()
        # a malformed marker (no ":") never compares equal to a real id, so it rebuilds, not refuses
        if was is None or (
            was != now and "0" not in (was.partition(":")[2], now.partition(":")[2])
        ):
            return False  # the lake file was replaced (a compaction): every offset may have moved
        if wm > os.path.getsize(
            self.log.path
        ):  # progress ahead of the lake: the lake was cut or swapped
            return False
        off, sha = self.idx.meta("ep_anchor_off"), self.idx.meta("ep_anchor_sha")
        if off is None or sha is None:
            return False
        payload = self.log.read_at(int(off))
        # the watermark IS the end of the anchor frame (offset + length prefix + payload): progress
        # that disagrees with its own anchor would fold from the wrong place and hide appended records
        return (
            payload is not None
            and hashlib.sha256(payload).hexdigest() == sha
            and wm == int(off) + _LEN.size + len(payload)
        )

    def _wm(self):
        """This index's own progress: the byte offset after the last folded frame. NOT
        `applied_watermark` -- the pointer-row builder (gates/lake_rebuild_gate) owns that one."""
        return int(self.idx.meta("ep_watermark") or 0)

    def _reset(self):
        """Drop every posting row and every progress marker in ONE transaction, after voiding the
        cached verdict: a reset that fails halfway must not leave a warm instance trusting an empty
        index under its old watermark (S8)."""
        self._checked = False
        self.idx.clear(TAG, _META_KEYS)

    def rebuild(self):
        """Drop every row and fold the whole lake again (the cold path; also the compaction answer)."""
        self._reset()
        return self.sync()

    def _rows_ok(self):
        """The posting rows are exactly the rows the fold wrote: their count AND their digest (the sum
        mod 2**64 of each row's hash) match the meta the fold kept in the same transaction. A count alone
        is blind to a row-preserving corruption (a key byte flipped, a seq redirected), which would hide
        a record. One streaming pass over the rows, once per process (and again after the index file
        changes under us); a mismatch rebuilds."""
        if self._wm() == 0:
            return True
        n, digest = 0, 0
        for key, seq in self.idx.iter_range(TAG, "", "￿"):
            n += 1
            digest += _row_hash(key, seq)
        return n == int(self.idx.meta("ep_rows") or -1) and digest % _MOD == int(
            self.idx.meta("ep_digest") or -1
        )

    def _guard_cache(self):
        """The row verification is cached per process; another connection committing to the index file
        (a second process, a repair tool, a fault) voids it, so the rows are verified again. Our own
        commits never move `data_version`, so the fold does not re-trigger the pass."""
        version = self.idx.data_version()
        if self._seen_version is not None and version != self._seen_version:
            self._checked = False
        self._seen_version = version

    # -- the tail fold (S3): fold the records after the watermark, advancing it in the same transaction
    def sync(self):
        """Fold every complete frame after the applied watermark; returns the records folded. A
        detected compaction or a damaged row set rebuilds from the start instead. A home with no
        lake yet folds nothing (the same miss every other reader of it answers)."""
        if not os.path.exists(self.log.path):
            return 0
        self._guard_cache()
        if (
            self._wm() > 0
            and self.idx.meta("ep_fmt") != FMT
            or not self._anchor_ok()
            or (not self._checked and not self._rows_ok())
        ):
            self._reset()
        self._checked = True
        wm = self._wm()
        start = wm if wm > 0 else None
        nrows = int(self.idx.meta("ep_rows") or 0)
        digest = int(self.idx.meta("ep_digest") or 0)
        lake_id = self._lake_id()
        rows, folded, end, last = [], 0, wm, None
        for offset, payload in self.log.replay_located(start=start):
            end = offset + _LEN.size + len(payload)
            last = (offset, hashlib.sha256(payload).hexdigest())
            ep = decode(payload)
            if ep is not None:
                chk = _chk(payload)
                rows += [
                    (TAG, _key(_tok(k, v), offset, chk), _CONTAINER, offset) for k, v in ep.items()
                ]
            folded += 1
            if folded % _BATCH == 0:
                nrows += len(rows)
                digest = (digest + sum(_row_hash(r[1], r[3]) for r in rows)) % _MOD
                self.idx.put_batch(rows, meta=_anchor(last, end, nrows, digest, lake_id))
                rows = []
        if last is not None:
            digest = (digest + sum(_row_hash(r[1], r[3]) for r in rows)) % _MOD
            self.idx.put_batch(rows, meta=_anchor(last, end, nrows + len(rows), digest, lake_id))
        return folded

    # -- the query (S4): candidates from postings, every answer verified against the lake's bytes
    def recall(self, cue):
        """The newest recallable episode whose slots contain EVERY cue slot (string-equal), like the
        deep scan: `{"episode", "confidence": 1.0, "source": "lake"}`, or the miss shape. A failure
        of the index or of the lake read REFUSES visibly (S6) -- never a miss, which would hide a record."""
        try:
            try:
                return self._recall(cue)
            except IndexCorrupt:  # the index disagreed with the lake: heal ONCE, from the lake
                self.rebuild()  # touches only the derived index, never the lake
                return self._recall(cue)
        except Exception as exc:  # noqa: BLE001 - every failure is a visible refusal
            return {
                "episode": {},
                "confidence": 0.0,
                "source": "refused",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def _recall(self, cue):
        miss = {"episode": {}, "confidence": 0.0, "source": "miss"}
        if not isinstance(cue, dict) or not cue:
            return miss
        self.sync()
        toks = [_tok(k, v) for k, v in cue.items()]
        counts = [self.idx.count_range(TAG, *_bounds(t)) for t in toks]
        if min(counts) == 0:
            return miss
        rarest = toks[counts.index(min(counts))]
        others = [t for t in toks if t != rarest]
        # lazily, newest first: the first VERIFIED row ends the walk, so a common token's posting list
        # (every episode for who=self) is never loaded whole
        with contextlib.closing(self.idx.iter_range(TAG, *_bounds(rarest), reverse=True)) as rows:
            return self._first_verified(cue, rows, others)

    def _first_verified(self, cue, rows, others):
        miss = {"episode": {}, "confidence": 0.0, "source": "miss"}
        for key, offset in rows:
            parts = key.split(":")
            if len(parts) != 3 or offset != int(parts[1], 16):  # the row disagrees with its own key
                raise IndexCorrupt(f"row {key} points at {offset}")
            chk = parts[2]
            if any(self.idx.get(TAG, _key(t, offset, chk)) is None for t in others):
                continue
            payload = self.log.read_at(offset)
            if payload is not None and _chk(payload) != chk:  # the bytes are not the frame indexed
                raise IndexCorrupt(f"row {key}: the bytes at {offset} are not the frame it indexed")
            ep = None if payload is None else decode(payload)
            if ep is not None and is_subcue(cue, ep):  # VERIFIED against the lake's own bytes
                return {"episode": ep, "confidence": 1.0, "source": "lake"}
        return miss


def main() -> int:
    """Bare invocation: a tiny temp lake -- the index equals the exact scan on constructed cues, never
    answers the None trap, and finds a record appended after the first sync (the tail fold)."""
    import os
    import tempfile

    from snn_data_lake.substrate import home as H

    with tempfile.TemporaryDirectory(prefix="episode_index_") as td:
        path = os.path.join(td, "brain_lake.bin")
        eps = [
            {"who": "self", "what": w, "where": f"p{i % 3}", "when": str(i), "affect": "calm"}
            for i, w in enumerate(["rest", "reach", "grasp", "look"] * 5)
        ]
        for e in eps:
            H.save_episode(e, path=path)
        ix = EpisodeIndex(H._log(path), os.path.join(td, "idx.db"))
        for cue in ({"when": "5"}, {"what": "rest", "where": "p1"}, {"who": "self"}):
            want = next((e for e in reversed(eps) if is_subcue(cue, e)), None)
            if ix.recall(cue)["episode"] != want:
                print(f"episode_index self-check FAILED: {cue}")
                return 1
        if ix.recall({"when": "5", "affect": "None"})["source"] != "miss":
            print("episode_index self-check FAILED: the None trap was answered")
            return 1
        late = {"who": "owner", "what": "wave", "where": "px", "when": "u9", "affect": "joy"}
        H.save_episode(late, path=path)
        if ix.recall({"when": "u9"})["episode"] != late:
            print("episode_index self-check FAILED: the tail fold missed a late record")
            return 1
        ix.close()
    print(
        "episode_index self-check OK: equals the exact scan, refuses the None trap, folds the tail"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
