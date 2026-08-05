r"""
home.py -- the ONE durable persistence home every part of the project boots into, so nothing is
forgotten between runs.

THE PROBLEM this fixes: each entrypoint used to open its OWN ephemeral store -- a temp `.spkl` that
was often DELETED on startup -- so a fact taught to Spike Chat, or a value the closed loop learned,
vanished the moment the process exited. Every boot started blank. This module is the single shared
home on disk that every entrypoint ATTACHES to at boot: one durable spike lake + one append-only
knowledge log, so learned knowledge ACCUMULATES across runs and across entrypoints.

  home()            -> the persistence dir (env NORD_HOME, else ~/.snn_data_lake), created on demand
  lake_path()       -> <home>/lake.spkl        the durable spike lake (StreamingHub / LakeMemory)
  knowledge_path()  -> <home>/brain_lake.bin   the ONE durable brain lake (RAW BINARY, BinLog-backed)
  state_path(name)  -> <home>/<name>           LEGACY sidecar path; see its docstring before using it
  KnowledgeLog      -> the symbolic VIEW of the one lake: .append(rec) durably (binary), .replay() to reload
  save_named_state/load_named_state(name, blob) -> named LEARNED state, IN the one lake (_T_STATE)
  save_state/load_state(name, obj) -> the same thing with a JSON facade over the blob
  banner(part)      -> one-line "<part> booted attached to durable home ..." string

WHAT PERSISTS (and what does not): the durable log holds only genuinely-LEARNED, non-re-derivable
knowledge -- facts a user taught, conversation turns, a learned value function. Knowledge that is
deterministically re-derivable at boot (the world-KB seed, the WordNet dictionary) is REBUILT each
boot instead of stored, so the durable log stays small and never balloons with the same dictionary
every run. "Never forgets" = replay the learned log + re-derive the rest.

SAFETY: this stores knowledge and perceptual/telemetry records only. It NEVER stores or replays motor
commands -- an arm entrypoint attaches for memory, and no remembered command is ever auto-actuated at
boot. Zero runtime deps (stdlib json + os); the data lake never touches torch. ASCII-only on disk
(json ensure_ascii) so Windows cp1252 cannot crash on reload.

  python -m snn_data_lake.substrate.home   # self-check incl. a REAL cross-process never-forget proof
"""

import json
import os
import sys

from snn_data_lake.substrate.binlog import (
    BinLog,
    pack_episode,
    pack_fact,
    unpack_episode,
    unpack_fact,
)

# The ONE brain lake is RAW BINARY (symbolic facts + chat turns ride as compact pack_fact byte-records).
# Each record carries a 1-byte TYPE tag so the symbolic view (KnowledgeLog) reads only its own records
# from the one file. FACT = pack_fact({cue,answer,triple?,src?}).
#
# THIS PARAGRAPH ENDED "Weights/calibration stay separate STATE (save_state), never in the memory
# lake" UNTIL 2026-08-05. It is RETRACTED: it is not a design statement any more, it is the exact
# arrangement P4b forbids ("what the brain learns goes IN the lake under a record tag -- not a
# sidecar file, not a JSON blob"), and the tag list below it had already contradicted it twice --
# _T_GROWTH moved the grown substrate INTO the lake (2026-07-30) and _T_STATE moved the whole
# sidecar FAMILY in (2026-08-05), calibration included. See `save_state()` for the measured census
# and for what is STILL outside; the direction of travel is INTO the lake, and this line was
# pointing the other way.
#
# _T_SPIKE is a RESERVED tag whose writer was RETIRED per P3 (2026-07-29). SpikeLake -- the durable
# AER-exemplar view (record = _T_SPIKE [u32 plen][payload][pack_array(vec)]) -- was measured that day
# at: 19,775 _T_FACT / 0 _T_SPIKE records in the production lake, zero constructors in the tree outside
# this module's own self-check, zero samples in 45 s of live-daemon profiling, and its only designed
# consumer (`reasoning/brain_adapters/lake_memory.py` `durable=`) a reasoning/ orphan off the live path.
# Backlog H4/#10 named the missing prerequisite: a live producer of AER exemplars worth keeping across a
# restart. Until one exists there is nothing to store, so the dead write path is retired instead of kept
# as a format nobody writes. The tag stays reserved, every view SKIPS foreign records and consolidate()
# PRESERVES them, so a historical lake containing spike records still replays fine -- and the class is
# one git revert away if the prerequisite ever materializes.
_T_FACT = b"\x01"
_T_SPIKE = b"\x02"
# _T_GROWTH added 2026-07-30: the grown neural substrate, in THE ONE LAKE rather than a sidecar file
# (owner decision). Safe as an additive tag because the existing readers already ignore what they do
# not recognise -- KnowledgeLog.replay filters on `payload[:1] == _T_FACT`, and consolidate() keeps
# `p for p in ... if p[:1] != _T_FACT`, so a foreign record is skipped by the FACT view and PRESERVED
# through a rewrite. That behaviour is self-checked (commit 9f620246), which is why this needed no
# change to either method.
_T_GROWTH = b"\x03"
# _T_EPISODE added 2026-08-04. Autobiographical episodes were being stored ONLY in
# hippocampus/episode_consolidation's in-memory attractor: measured, 59 episodes stored over 600
# ticks with persist=True added 0 records to the lake, and 149 episodes over 1,500 ticks were
# recalled 0 times. So the brain was forming memories into a second store that nothing read and
# nothing persisted, and every one died with the process. Episodes now land in THE ONE LAKE, where
# they survive a restart and can be recalled from the same store as everything else the brain knows.
_T_EPISODE = b"\x04"
# _T_SELFBUILD added 2026-08-04, promoting reasoning/brain_native_selfbuild.py into the governed
# tree. Its RSTDP three-factor weights -- what the self-build learner knows about which candidate
# sources are worth proposing -- were persisted to `data/selfbuild_state.json`, a repo-relative
# sidecar. That is a SECOND DURABLE STORE holding LEARNED state, which is exactly what P4b forbids:
# "what the brain learns goes IN brain_lake.bin under its own record tag". Two concrete costs of the
# sidecar, neither hypothetical: it sat under the repo rather than under `home()`, so it did not
# follow NORD_HOME and two brains pointed at different lakes still shared one weight file; and
# `consolidate()` could not see it, so sleep-time compaction reasoned about a lake that was missing
# part of what the brain had learned. Same additive-tag safety argument as _T_GROWTH -- the FACT
# view filters on `payload[:1] == _T_FACT` and consolidate() PRESERVES every non-FACT record, so
# existing readers skip this and a rewrite keeps it.
_T_SELFBUILD = b"\x05"
# _T_CALIB was added 2026-08-04 as a per-caller channel for ONE sidecar (`signal_calib.json`) and is
# RETIRED 2026-08-05 WITHOUT EVER BEING WRITTEN, superseded by the generic `_T_STATE` record below.
# P2 (exactly one mechanism survives): a tag + a save_X/load_X pair PER sidecar was on course to make
# seven near-identical channels for one job -- calibration, the recall trace, the time-cell
# thresholds, the valence policy, two Gf libraries, the gesture-signature library -- each with its
# own byte, its own two functions and its own copy of the same six-line replay. `_T_STATE` is that
# one mechanism, keyed by NAME. Retiring this is safe and MEASURED, not assumed: a read-only frame
# scan of the operator's live lake on 2026-08-05 (516,810,138 B, 123,120 records) printed
# `tag 01: 21,033 / tag 03: 73,081 / tag 04: 29,005 / tag 05: 1` and NO 0x06 record at all, which is
# what "the channel was never wired" looks like from the data side. RE-SCANNED INDEPENDENTLY later
# the same day, the daemon having grown the lake to 519,646,815 B / 124,536 records:
# `tag 01: 21,040 / tag 03: 73,490 / tag 04: 30,005 / tag 05: 1`, still no 0x06 and no 0x07. The
# counts move because the lake is live; the ABSENCE is what the retirement rests on and it held.
# (Do the scan read-only, frame by frame -- constructing a BinLog over the live lake would run
# `_repair_torn_tail` and TRUNCATE it under the running daemon.) The byte stays RESERVED (never
# reused) exactly as _T_SPIKE does, so a lake written by an intermediate build still replays: every
# view skips foreign records and consolidate() preserves them. Successor for anyone holding the
# queued consumer edit: `save_named_state("signal_calib.json", blob)` / `load_named_state(...)`.
_T_CALIB = b"\x06"
# _T_STATE added 2026-08-05. THE ONE named-state record, and the close of the P4b sidecar FAMILY --
# not one violation, seven. `save_state()`/`load_state()` used to be `json.dump` to `<home>/<name>`,
# a plain file beside brain_lake.bin, and every one of these rode it: `recall_trace.json` (the
# per-record recall counts that decide which lake records survive the night),
# `time_cells.json` (a learned threshold vector), `valence.json` (the learned value function),
# `gf_library.json` + `active_library.json` (two learned primitive libraries),
# `reasoning_curriculum.json`, `nightly_skills.json`. P4b forbids exactly that, in these words:
# "what the brain learns goes IN the lake under its own record tag -- not a sidecar file, not a JSON
# blob". So `save_state`/`load_state` KEEP THEIR NAMES AND SIGNATURES and now write here; no caller
# changed, which is the point -- a family this wide is closed by fixing the mechanism, not by seven
# consumer edits landing one at a time and leaving the tree half-migrated.
#
# RECORD LAYOUT: _T_STATE + [u8 len(name)] + name (ascii) + [u32 seq, little-endian] + blob.
# `seq` is not decoration. BinLog dedups byte-identical payloads, and latest-wins over a deduped log
# is WRONG for state that can revert to an earlier value: save {} then {"a":1} then {} again and the
# third write is skipped as a duplicate, so the load returns {"a":1} -- a pruned recall trace
# resurrecting itself. `RecallTrace.prune_to()` makes that reachable, and the constructed red is run
# in main() below. `seq` (the number of records already stored under this name) makes every write a
# distinct payload, so dedup can never eat one.
#
# MIGRATION, NOT DELETION: `load_named_state` imports an existing `<home>/<name>` sidecar into the
# lake the first time it is asked for it, and LEAVES THE FILE ON DISK, inert. Those files hold real
# history -- 2,505 B of recall counts and 2,031 B of learned thresholds on the operator's home
# today -- and resetting them would be a policy fix that quietly costs the brain what it learned.
#
# COST, MEASURED rather than waved at: a save/load is a replay of the one lake, and one full pass
# over that 516.8 MB lake costs 349 / 376 / 378 ms (three passes, .venv-lake, 2026-08-05) against
# `load_state`'s former ~0.1 ms file read. RE-MEASURED the same day on a lake the live daemon had
# grown to 519,646,815 B / 124,536 records: 250 / 271 / 279 ms. Both are honest; the figure is
# cache- and load-dependent, so read it as a RANGE (a quarter to four tenths of a second) and never
# as a constant. That is why `_state_index()` exists -- the pass happens once per BinLog instance,
# and `_log()` already replaces the instance whenever the file changed underneath it. Same
# additive-tag safety argument as _T_GROWTH: the FACT view filters on `payload[:1] == _T_FACT` and
# consolidate() PRESERVES every non-FACT record, asserted below.
#
# WHAT THIS TRADED AWAY, and it is not free: a sidecar was OVERWRITTEN in place, so it held exactly
# one copy of the state. A _T_STATE record is APPENDED, and nothing prunes the superseded copies --
# `consolidate()` is the only compaction in the tree and it preserves every non-FACT record by
# design. MEASURED 2026-08-05: 50 saves of a CHANGED ~2 KB blob under one name left 50 records, and
# a full `consolidate()` afterwards pruned 0 of them. The unchanged-blob skip in `save_named_state`
# bounds a state that stops moving; it does nothing for one that keeps moving, and the live callers
# keep moving -- `cortex/brain.py` saves a 76 KB `reasoning_curriculum.json` on the sleep tick and
# `RecallTrace.save()` runs beside it. This is the exact shape that made _T_GROWTH 73,490 records /
# 508,082,042 B, 97.8 percent of the operator's lake. It is recorded here rather than fixed because
# "does the brain keep the HISTORY of its own state" is an owner call, not a refactor: _T_GROWTH
# and _T_SELFBUILD both argue IN WRITING that their history is the more useful artefact, and a
# latest-wins compaction added here would silently delete learned state on a sleep tick for all
# three. Whoever takes it: the compaction is one line in `consolidate()` (keep only the last record
# per name among the _T_STATE foreigners) and it owes a constructed red for "a rewrite kept the
# newest state" before it ships.
_T_STATE = b"\x07"


def home():
    """The durable persistence dir. `NORD_HOME` overrides; default is ~/.snn_data_lake. Created on demand."""
    d = os.environ.get("NORD_HOME") or os.path.join(os.path.expanduser("~"), ".snn_data_lake")
    os.makedirs(d, exist_ok=True)
    return d


def lake_path(name="lake.spkl"):
    """Path to the durable spike lake under home() (the StreamingHub / LakeMemory `.spkl` store)."""
    return os.path.join(home(), name)


def knowledge_path(name="brain_lake.bin"):
    """Path to the ONE durable brain lake under home() -- a raw-binary append-only log (BinLog) holding
    the whole brain's learned MEMORY (symbolic facts + chat turns as pack_fact records; other binary
    record types may fold in under reserved tags). The former knowledge.jsonl is retired into this file."""
    return os.path.join(home(), name)


def state_path(name):
    """Path to `<home>/<name>` -- the LEGACY sidecar location, kept for two honest uses only.

    (1) `_migrate_sidecar` reads it, once, to import a pre-2026-08-05 blob into the one lake.
    (2) Binary blobs the JSON facade cannot carry are still written here by their own writers:
        `cortex/brain.py` -> `language_acq.npz` (numpy) and `reasoning/nightly_skills.py`
        -> `skills*.spkl`. Those are OPEN P4b debt, named in `save_state`'s docstring; the carrier
        they need is `save_named_state`, which takes raw bytes.

    It is NOT the place to put anything the brain LEARNS. That was what this returned a path for
    until 2026-08-05, and the record of what it cost is in the `_T_STATE` tag comment."""
    return os.path.join(home(), name)


#: One BinLog per (path, dedup). Constructing a BinLog is O(file) TWICE -- a full tail-repair scan
#: plus the boot-time replay that builds its dedup set -- measured at 708 ms on the live 425 MB /
#: 80,853-record lake. `save_growth()` used to pay that to append ONE record, and py-spy on the live
#: daemon put the resulting `write` + `_repair_torn_tail` + `rewrite` at about 20 percent of the tick.
#:
#: Sharing is also more correct than not: two instances over one file hold two divergent dedup sets,
#: so a payload could be written twice, each instance believing it was new.
#:
#: ponytail: the cache is keyed on the path string as given. Two spellings of the same file (relative
#: vs absolute, differing case on Windows) would key separately and reintroduce the split-dedup
#: problem. Every caller here goes through knowledge_path(), so today they cannot diverge; normalise
#: with os.path.realpath if a caller ever passes a path from outside.
_LOG_CACHE = {}


def _log(path=None, dedup=True):
    """The shared BinLog for `path`, rebuilt if the file changed underneath it.

    The staleness check is NOT optional. A cached instance never re-runs `_repair_torn_tail()`, so
    without it a crash-torn tail is never repaired and the next append lands behind the torn frame
    and is lost -- the exact failure that repair exists to prevent. `BinLog.stale()` compares the
    on-disk size against what this instance last wrote, which costs a stat (microseconds) against
    the full construction it avoids -- RE-MEASURED 2026-08-05 at 1316 / 1420 / 1307 ms on the live
    lake as it stands (513.1 MB, 121,177 records, `.venv-lake`). The bare "742 ms" that stood here,
    and the "708 ms" two comments up, were both measured on a 425 MB / 80,853-record lake; the cost
    is a function of lake SIZE and the lake grows, so state the conditions or the number rots.
    """
    key = (path or knowledge_path(), bool(dedup))
    log = _LOG_CACHE.get(key)
    if log is None or log.stale():
        log = _LOG_CACHE[key] = BinLog(key[0], dedup=key[1])
    return log


class KnowledgeLog:
    """The symbolic VIEW of the ONE durable brain lake -- a raw-BINARY, append-only, deduped, crash-safe,
    cross-process never-forget store (substrate/binlog.BinLog at <home>/brain_lake.bin). Shared by the
    developmental loop (what it learns autonomously) and Spike Chat (what it converses from).

    Every learned record is written as a 1-byte-TYPE-tagged pack_fact BYTE-record (NOT JSON) and fsync'd,
    so a crash cannot lose it; replay() streams every FACT record back IN ORDER at the next boot (skipping
    any spiking/array records that share the one lake), so a fresh PROCESS reconstructs everything learned.

    LEAN by default: dedup=True (BinLog) skips a byte-identical record, so the loop can re-see the same
    concept forever without bloating the SSD. Append-only otherwise. Same append(dict)/extend/
    replay->dicts/count() API as the retired JSONL store -- every caller is unchanged; only the on-disk
    format is now binary (spiking memory is naturally binary; symbolic text rides along as compact bytes).

    SHARED RECORD SCHEMA (what the loop writes and the chat + reasoner read):
      {"cue": ["content","words"],        # required: retrieval keys (content words)
       "answer": "the answer text.",      # required: the reply/definition text
       "triple": ["subject","rel","obj"], # optional: a structured fact (enables contradiction reasoning)
       "src": "wikipedia:calculus"}       # optional: provenance of an autonomously-learned fact
    """

    def __init__(self, path=None, dedup=True):
        self._bin = _log(path, dedup=dedup)
        # symbolic (FACT) record count -- excludes any spiking records sharing the one lake
        self._count = sum(1 for p in self._bin.replay() if p[:1] == _T_FACT)

    def append(self, record):
        """Durably append one fact/chat record as a TYPE-tagged pack_fact byte-record (skipped if dedup
        and byte-identical). fsync'd by BinLog. Returns True if written, False if a duplicate."""
        if not isinstance(record, dict):
            raise TypeError("KnowledgeLog.append: record must be a dict")
        written = self._bin.append(_T_FACT + pack_fact(record))
        if written:
            self._count += 1
        return written

    def extend(self, records):
        """Durably append many records (one fsync at the end -- for bulk import). Returns #written."""
        payloads = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("KnowledgeLog.extend: every record must be a dict")
            payloads.append(_T_FACT + pack_fact(record))
        written = self._bin.extend(payloads)
        self._count += written
        return written

    def replay(self):
        """Yield every stored FACT record as a dict, in append order. Records of another type (spiking
        exemplars / arrays sharing the one lake) and torn frames are skipped -- the symbolic view never
        crashes on a mixed lake."""
        for payload in self._bin.replay():
            if payload[:1] != _T_FACT:
                continue
            try:
                yield unpack_fact(payload[1:])
            except Exception:  # noqa: BLE001 - a malformed record must never crash the boot replay
                continue

    def count(self):
        """Number of FACT records stored (O(1) maintained counter -- callers poll this per round/request)."""
        return self._count

    def consolidate(self, keep_records):
        """SLEEP CONSOLIDATION: replace the durable FACT records with `keep_records` (the survivors a
        consolidation policy chose -- redundant merged, re-derivable/stale pruned), preserving any non-FACT
        (spiking) records in the one lake untouched. Atomic compaction via BinLog.rewrite. This is the PRUNE
        half of sleep consolidation -- without it the symbolic lake only ever grows. Returns (kept, dropped)."""
        foreign = [
            p for p in self._bin.replay() if p[:1] != _T_FACT
        ]  # spike/array records stay as-is
        before = self._count
        self._bin.rewrite(foreign + [_T_FACT + pack_fact(r) for r in keep_records])
        self._count = self._bin.count() - len(foreign)
        return self._count, before - self._count


def _state_key(name):
    """`name` as the ascii key bytes of a _T_STATE record, or a raised ValueError.

    Both refusals are REACHABLE and both reds are constructed and run in main(): the length prefix
    is one byte, so a 256-character name would wrap and silently address a DIFFERENT state; and a
    non-ascii name cannot be encoded at all. Failing loudly here beats writing a record that only
    the writer can find again."""
    key = str(name)
    try:
        raw = key.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"state name must be ascii: {key!r}") from None
    if not raw or len(raw) > 255:
        raise ValueError(f"state name must be 1..255 ascii bytes, got {len(raw)}: {key!r}")
    return raw


def _state_index(log):
    """{name: (seq, blob)} for `log` -- built once per BinLog instance, maintained on our appends.

    A CACHE, never the source of truth (P4b's own wording). It is attached to the instance `_log()`
    hands out, and `_log()` REPLACES that instance whenever `BinLog.stale()` says the file changed
    underneath it -- so records written by another process are picked up by the rebuild rather than
    missed. Attached to the instance rather than kept in a dict keyed by path precisely so that
    "the log was rebuilt" and "the index is void" are the same event and cannot drift apart.

    Why it exists at all: one full replay of the operator's live lake (516,810,138 B / 123,120
    records) measured 349 / 376 / 378 ms on 2026-08-05, and `cortex/brain.py` calls
    `self._time_clock.save()` on every tick the clock grows and `trace.save()` on every sleep tick.
    Paying that pass per call would be a third of a second on the tick path."""
    idx = getattr(log, "_snn_state_index", None)
    if idx is None:
        idx = {}
        for payload in log.replay():
            if payload[:1] != _T_STATE:
                continue
            n = payload[1]
            idx[payload[2 : 2 + n].decode("ascii", "replace")] = (
                int.from_bytes(payload[2 + n : 6 + n], "little"),
                payload[6 + n :],
            )
        log._snn_state_index = idx
    return idx


def save_named_state(name, blob, path=None):
    """Store `blob` (bytes) under `name` in THE ONE LAKE (_T_STATE). True if written, False if not.

    False means one of two things, both deliberate. A falsy blob is REFUSED, for the reason
    `save_growth` refuses one: persisting "nothing" as though it were a measurement restores an
    empty learner with a confident tally beside it. An UNCHANGED blob is also not written -- the
    lake is append-only and consolidate() preserves every non-FACT record forever, so re-storing a
    byte-identical state on every sleep tick would grow the one store without adding one bit of
    information. A CHANGED state always lands, including a change back to an earlier value (see the
    `seq` field in the _T_STATE tag comment)."""
    key = _state_key(name)
    if not blob:
        return False
    blob = bytes(blob)
    log = _log(path)
    idx = _state_index(log)
    prev = idx.get(str(name))
    if prev is not None and prev[1] == blob:
        return False
    seq = 0 if prev is None else prev[0] + 1
    written = log.append(_T_STATE + bytes([len(key)]) + key + seq.to_bytes(4, "little") + blob)
    if written:
        idx[str(name)] = (seq, blob)
    return written


def load_named_state(name, path=None):
    """The most recent blob stored under `name`, or None if the brain never stored one.

    None is the honest answer for a lake written before this record type existed -- every caller
    treats it as a cold start, which is the pre-2026-08-05 behaviour for a sidecar that was not
    there. On the DEFAULT lake a pre-existing `<home>/<name>` sidecar is imported once instead (see
    `_migrate_sidecar`); an explicit `path` is a caller pointing at a specific lake, so it is not
    silently paired with a sidecar from somewhere else."""
    _state_key(name)
    hit = _state_index(_log(path)).get(str(name))
    if hit is not None:
        return hit[1]
    return _migrate_sidecar(name) if path is None else None


def _migrate_sidecar(name):
    """One-time import of a pre-P4b `<home>/<name>` sidecar INTO the lake. Returns its bytes or None.

    The file is LEFT ON DISK, inert. These sidecars hold real learned history -- 2,505 B of recall
    counts and 2,078 B of learned time-cell thresholds on the operator's home on 2026-08-05 (that
    second figure was recorded here as 2,031 B six hours earlier; the live daemon rewrites the file,
    so a byte count of a LIVE home is a timestamp, not a constant) -- and deleting them to
    "complete" a migration would destroy the very state the migration exists to keep. An unreadable
    sidecar is treated as absent rather than raised: a cold start is recoverable, a boot that dies
    on a stale file is not.

    IT VALIDATES NOTHING, and that is a real edge this carries rather than a claim it makes. It
    imports whatever bytes are at `<home>/<name>`, permanently, and `consolidate()` then preserves
    them forever. MEASURED 2026-08-05: a `<home>/corrupt.json` containing `{not json` is imported as
    a _T_STATE record and `load_state` thereafter returns `default` (a cold start) while the junk
    stays in the lake; a BINARY sidecar imports whole. It cannot validate -- it is the raw-bytes
    carrier, and its JSON-shaped callers come through `load_state`, which is where the parse lives.
    The blast radius is bounded and was checked, not assumed: the only names any live path asks for
    are the seven JSON states, so the binary members of the operator's home (`language_acq.npz`
    24,700 B, `prototypes.bin`, `skills*.spkl`) are never requested by name and never imported. That
    is a fact about today's callers; a future caller that hands a binary name to `load_state` would
    move 24 KB of npz into the one lake and read `default` back out of it.

    IT FIRES ON READ ONLY, and the ordering that makes that sufficient is a fact about the callers
    rather than a guarantee of this function -- stated because it is the way this could quietly lose
    something. A caller that SAVED before it ever LOADED would write seq 0 for that name and the
    sidecar would then never be imported (left on disk unread, not destroyed). Every live path
    loads first, checked by reading them: `Brain.__init__` reloads the reasoning curriculum and
    calls `TimeCellClock.restored()` at construction, `RecallTrace.__init__` loads before any
    `bump`, `OpenGf.__init__` and `Library.load` load in the constructor, `closed_loop --persist`
    loads before `run_episode`, and `signal_loop.load_library` runs at the top of `main()`."""
    legacy = state_path(name)
    try:
        with open(legacy, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if not raw:
        return None
    save_named_state(name, raw)
    return raw


def save_state(name, obj):
    """Durably store a JSON-serialisable state blob under `name` IN THE ONE LAKE (P4b).

    THIS FUNCTION USED TO BE THE SIDECAR MECHANISM P4b FORBIDS -- `json.dump` to `<home>/<name>`,
    a plain file beside brain_lake.bin -- and it was not one violation, it was a family. Measured on
    the LIVE home 2026-08-05 (`C:/Users/vinhl/.snn_data_lake`), beside a 516.8 MB `brain_lake.bin`.
    The MEMBERSHIP of this census is the load-bearing part and it reproduces exactly; the BYTE
    COUNTS are a snapshot of files the running daemon rewrites, and a re-scan hours later returned
    `time_cells.json 2,078` (was 2,031), `nightly_skills.json 8,111` (was 8,133) and
    `reasoning_curriculum.json 76,669` (was 76,670) while the other four were unchanged. Re-scan
    before quoting a digit; do not "correct" one against the other.

        language_acq.npz  24,700 B   the acquired language model (via state_path + np.save)
        reasoning_curriculum.json  76,670 B
        nightly_skills.json  8,133 B    skills.spkl / skills.<id>.spkl  5,748 B each
        recall_trace.json  2,505 B      time_cells.json  2,031 B
        signal_calib.json    950 B      (+ develop_state.json, prototypes.bin, older/stale)

    Every JSON member of that census is closed by this one edit, because the callers name a state
    and this decides where it lives: `cortex/brain.py` (reasoning_curriculum.json),
    `brainstem/closed_loop.py` (valence.json), `hippocampus/lake_consolidation.py`
    (recall_trace.json), `cortex/time_cells.py` (time_cells.json), `reasoning/gf.py`
    (gf_library.json, active_library.json), `reasoning/nightly_skills.py`. Not one of them changed a
    line to move, and none of them can drift back.

    STILL OPEN, and named so it stays countable: `language_acq.npz` and the `skills*.spkl` blobs do
    NOT come through here -- they are numpy/pickle binaries written to `state_path(...)` directly, so
    they need a carrier for raw bytes at their call sites (`save_named_state` already is one) rather
    than a JSON facade. `language_acq.npz` is the LARGEST member of the family at 24,700 B.

    Returns True if a record was written, False if the state was unchanged (see `save_named_state`);
    it returned the sidecar path until 2026-08-05, and no caller in the tree read that value."""
    return save_named_state(
        name, json.dumps(obj, ensure_ascii=True, sort_keys=True).encode("ascii")
    )


def load_state(name, default=None):
    """Load the JSON state stored under `name`; return `default` if it was never stored.

    Reads the ONE lake, falling back ONCE to a pre-P4b `<home>/<name>` sidecar, which it imports and
    leaves in place (`_migrate_sidecar`). A sidecar that is no longer valid JSON is treated as never
    written -- the pre-2026-08-05 form raised on it, and a boot that dies on a stale file it is in
    the middle of retiring is the worse of the two failures."""
    blob = load_named_state(name)
    if blob is None:
        return default
    try:
        return json.loads(blob.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return default


def save_growth(blob, path=None):
    """Append one grown-substrate record to the ONE durable lake. Returns whatever BinLog.append
    returns (a truthy written-flag), or 0 when there was nothing to write.

    (This docstring said "Returns bytes written" for its first ten minutes of life, which the very
    first round-trip run falsified by printing `wrote True bytes`. Recorded rather than quietly
    corrected: a return-value claim nobody checks is the same defect class as a stale citation, and
    this file's own project spent today removing several.)

    Append-only, like every other record here: the newest _T_GROWTH record IS the current substrate,
    and older ones are the growth history rather than garbage. `load_growth` therefore takes the LAST.

    Refuses a falsy blob. `neurogenesis_engine.growth_state_bytes()` returns None (never an empty
    blob) when nothing is grown, and this is the second half of that contract -- together they make it
    impossible to persist "no substrate" as though it were substrate, which is the failure that would
    otherwise restore an empty engine with a confident accuracy beside it."""
    if not blob:
        return 0
    return _log(path).append(_T_GROWTH + bytes(blob))


def save_episode(episode, path=None):
    """Append one autobiographical episode to THE ONE LAKE. True if written, False if a duplicate.

    The lake, not a sidecar and not an in-memory attractor, because an episode that does not
    survive the process is not a memory. Deduping is the BinLog's, on the packed bytes, and
    pack_episode sorts its slots so the same episode is the same bytes.
    """
    if not episode:
        return False
    return _log(path).append(_T_EPISODE + pack_episode(episode))


def load_episodes(path=None, limit=None):
    """Every episode in the lake, oldest first. The durable half of episodic memory.

    Foreign record types are skipped, exactly as KnowledgeLog skips non-FACT records -- one lake,
    several tenants, each reading only its own.
    """
    out = []
    for payload in _log(path).replay():
        if payload[:1] == _T_EPISODE:
            try:
                out.append(unpack_episode(payload[1:]))
            except Exception:
                continue  # a record we cannot read must not stop the ones after it
    return out[-limit:] if limit else out


def save_selfbuild(blob, path=None):
    """Append one self-build learner state blob (JSON bytes) to THE ONE LAKE. 0 if nothing to write.

    Append-only and latest-wins, identical in shape to `save_growth`: the newest _T_SELFBUILD record
    IS the current learner state and the older ones are its learning history, which is the more
    useful artefact anyway -- a weight that moved is the evidence the three-factor rule is doing
    something, and a sidecar that is overwritten in place destroys exactly that evidence.

    Refuses a falsy blob for the same reason save_growth does: persisting "nothing learned" as
    though it were learned state is the failure that restores an empty learner with a confident
    tally beside it."""
    if not blob:
        return 0
    return _log(path).append(_T_SELFBUILD + bytes(blob))


def load_selfbuild(path=None):
    """The most recent self-build learner blob, or None if the lake holds none.

    None is the honest answer for a lake written before this record type existed; the caller starts
    from its uniform weight init, which is exactly the pre-2026-08-04 cold-start behaviour."""
    latest = None
    for payload in _log(path).replay():
        if payload[:1] == _T_SELFBUILD:
            latest = payload[1:]
    return latest


# `save_calibration` / `load_calibration` lived here from 2026-08-04 to 2026-08-05 and are RETIRED,
# not renamed: P2 says exactly one mechanism survives, and they were `save_named_state` with the name
# baked in. They had ZERO consumers in src/, tests/ or tools/ on the day they were retired -- the
# consumer edit they were built for never landed -- and their tag was never written to any lake
# (measured; see the `_T_CALIB` comment). Successor, byte-for-byte the same durability contract:
#     save_named_state("signal_calib.json", blob)   /   load_named_state("signal_calib.json")
# and the second of those also imports the existing 950 B `<home>/signal_calib.json` on first read.


def load_growth(path=None):
    """The most recent grown-substrate blob, or None if the lake holds none.

    None is the honest answer for a lake written before this record type existed, and the caller
    (`Brain._make_neuro`) treats it as "start ungrown" -- so an old lake behaves exactly as it did
    before, which is the backward-compatibility requirement."""
    latest = None
    for payload in _log(path).replay():
        if payload[:1] == _T_GROWTH:
            latest = payload[1:]
    return latest


def banner(part, lake=True, knowledge=True):
    """One-line ASCII boot banner: which durable home this part attached to + how much it remembers."""
    bits = [f"[boot] {part} attached to durable home {home()}"]
    if knowledge:
        bits.append(f"knowledge={KnowledgeLog().count()} learned facts")
    if lake:
        lp = lake_path()
        sz = os.path.getsize(lp) if os.path.exists(lp) else 0
        bits.append(f"lake={sz:,}B")
    return " | ".join(bits)


def main():
    import shutil
    import subprocess
    import tempfile

    work = tempfile.mkdtemp(prefix="snn_home_")
    _prior_home = os.environ.get("NORD_HOME")  # restored in the finally, NOT popped -- see below
    try:
        os.environ["NORD_HOME"] = work  # point home() at an isolated temp dir
        src_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )  # .../src
        kpath = knowledge_path()

        print("=" * 74)
        print("HOME -- the ONE durable persistence home every entrypoint boots into")
        print("=" * 74)
        print(f"home()            : {home()}")
        print(f"lake_path()       : {lake_path()}")
        print(f"knowledge_path()  : {kpath}")

        # ---- PROPERTY 1: durable append + replay within one process ----
        log = KnowledgeLog()
        log.append({"cue": ["capital", "france"], "answer": "paris"})
        log.append({"cue": ["capital", "japan"], "answer": "tokyo"})
        first = list(KnowledgeLog().replay())  # a FRESH object, no shared RAM

        # ---- PROPERTY 2: REAL cross-process never-forget ----
        # A SEPARATE python process appends a fact; then THIS process replays and must see all three.
        # This is the honest proof: a fact taught in one run survives into the next run (new process).
        child = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            "from snn_data_lake.substrate.home import KnowledgeLog\n"
            f"KnowledgeLog({kpath!r}).append({{'cue': ['largest', 'planet'], 'answer': 'jupiter'}})\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode  # a SEPARATE python process
        after = list(KnowledgeLog().replay())  # reload EVERYTHING from disk in this process

        # ---- PROPERTY 3: durable JSON state blob round-trips ----
        save_state("valence.json", {"ENGAGE": 0.83, "RELEASE": -0.11})
        state = load_state("valence.json")
        state_default = load_state("never_written.json", default={"empty": True})

        answers_after = [r["answer"] for r in after]
        print("-" * 74)
        print(f"in-process append : {[r['answer'] for r in first]} (2 facts, fresh reader)")
        print(f"child process     : appended 'jupiter' in a SEPARATE python (rc={rc})")
        print(f"after restart     : {answers_after}  <- reloaded purely from disk")
        print(f"durable state blob: valence.json -> {state}")
        print(f"{banner('home self-check')}")
        print("=" * 74)

        # ---- self-checks (deterministic asserts) ----
        assert [r["answer"] for r in first] == ["paris", "tokyo"], (
            "in-process append+replay lost a fact"
        )
        assert rc == 0, f"child process failed to append (rc={rc})"
        assert answers_after == ["paris", "tokyo", "jupiter"], (
            f"cross-process never-forget FAILED: expected 3 facts across 2 processes, got {answers_after}"
        )
        # append-only: the child's write ADDED to, never replaced, the parent's writes
        assert len(after) == len(first) + 1, (
            "the log must be append-only (child clobbered prior facts)"
        )
        assert state == {"ENGAGE": 0.83, "RELEASE": -0.11}, "durable state blob did not round-trip"
        assert state_default == {"empty": True}, (
            "load_state default not returned for a missing blob"
        )
        assert KnowledgeLog().count() == 3, "count() disagrees with replay()"
        # LEAN: re-appending a KNOWN fact must NOT grow the lake (never store the same fact twice)
        dup_written = KnowledgeLog().append({"cue": ["capital", "france"], "answer": "paris"})
        assert dup_written is False and KnowledgeLog().count() == 3, (
            "dedup failed: a duplicate fact grew the lake"
        )
        # CRASH-RECOVERY (raw-binary): a torn trailing frame (crash mid-append -> a partial length prefix)
        # must NOT crash the next boot; BinLog truncates to the last COMPLETE frame, and a new append lands
        # cleanly after it (never fused onto the garbage) and is replayed.
        with open(kpath, "ab") as f:
            f.write(b"\x02\x00")  # 2 bytes: an incomplete u32 length prefix -> a crashed write
        survived = KnowledgeLog().count()  # construction repairs the torn tail, must NOT raise
        KnowledgeLog().append({"cue": ["moon"], "answer": "luna"})
        recovered = [r["answer"] for r in KnowledgeLog().replay()]
        assert survived == 3, (
            "a torn trailing frame must be skipped, not crash every future KnowledgeLog()"
        )
        assert "luna" in recovered, "an append after a crash-torn binary frame was lost"

        # ---- the ONE lake tolerates FOREIGN record types: skipped by the FACT view, kept by the prune ----
        # _T_SPIKE is a reserved tag (its SpikeLake writer was retired per P3 -- see the tag comment
        # above). What must survive is the CONTAINER contract that let two views share one file: a
        # record of an unknown/foreign type never leaks into the FACT view and is never dropped by the
        # symbolic sleep prune, so any historical lake holding such records still replays fine.
        foreign_rec = _T_SPIKE + b"reserved-tag payload (retired AER exemplar shape)"
        assert BinLog(kpath).append(foreign_rec), "foreign record append failed"
        facts_now = (
            KnowledgeLog().count()
        )  # the FACT view reads ONLY facts (foreign records are skipped)
        assert facts_now == 4, f"foreign records leaked into the FACT count: {facts_now}"
        klog = (
            KnowledgeLog()
        )  # sleep consolidation of the FACT side must PRESERVE the foreign records
        klog.consolidate(list(klog.replay()))
        assert any(p == foreign_rec for p in BinLog(kpath).replay()), (
            "sleep consolidation of facts dropped a foreign (reserved-tag) record"
        )
        assert KnowledgeLog().count() == 4, "sleep consolidation lost facts"

        # PROPERTY 3 again, now that a consolidation has run over the lake: the learned state written
        # through save_state at the top of this check is a FOREIGN record to the FACT view, so the
        # sleep prune must have carried it through untouched. This is the half of P4b the sidecar
        # could never satisfy -- `consolidate()` cannot see a file it does not contain.
        assert load_state("valence.json") == {"ENGAGE": 0.83, "RELEASE": -0.11}, (
            "sleep consolidation of the FACT side destroyed a _T_STATE record"
        )

        # ---- _T_STATE: the ONE named-state record (see the tag comment) --------------------------
        # Proven the way P4b demands -- by a NEW PROCESS reading the lake back after the writer
        # exited, not by inspecting the writer's own object. A same-process round-trip would pass
        # against a pure in-memory store and prove nothing about durability.
        assert load_named_state("signal_calib.json") is None, (
            "a name never stored must read back as None, not as empty state"
        )
        assert save_named_state("signal_calib.json", b"") is False, (
            "an empty blob must be refused, never stored as state"
        )
        assert load_named_state("signal_calib.json") is None, (
            "refusing an empty blob must not create a record"
        )
        child_state = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            "from snn_data_lake.substrate.home import save_named_state\n"
            f"save_named_state('signal_calib.json', b'{{\"salience_th\": 0.61}}', {kpath!r})\n"
        )
        rc_c = subprocess.run([sys.executable, "-c", child_state]).returncode
        assert rc_c == 0, f"child process failed to write a state record (rc={rc_c})"
        assert load_named_state("signal_calib.json") == b'{"salience_th": 0.61}', (
            "state written by a SEPARATE process was not read back from the lake"
        )
        # latest-wins per NAME, and two names do not collide
        save_named_state("signal_calib.json", b'{"salience_th": 0.72}')
        save_named_state("other.json", b"unrelated")
        assert load_named_state("signal_calib.json") == b'{"salience_th": 0.72}', (
            "load_named_state must take the LAST record for its own name"
        )
        assert load_named_state("other.json") == b"unrelated", "state names must not collide"
        # ...and the history is kept rather than overwritten in place
        assert sum(1 for p in BinLog(kpath).replay() if p[:1] == _T_STATE) == 4, (
            "state must be append-only history (valence + 2 calib + other), not an in-place overwrite"
        )
        # UNCHANGED state is not re-stored: the lake is append-only and consolidate() keeps every
        # foreign record forever, so a sleep tick that saves an identical blob must add nothing.
        assert save_named_state("other.json", b"unrelated") is False, (
            "an unchanged state must not be appended again"
        )
        assert sum(1 for p in BinLog(kpath).replay() if p[:1] == _T_STATE) == 4, (
            "an unchanged state grew the one lake"
        )
        # REACHABLE RED, constructed and run: state that REVERTS to an earlier value. BinLog dedups
        # byte-identical payloads, so without the per-name `seq` field this third write is silently
        # skipped and the load returns the MIDDLE value. Measured with `seq` removed from the record:
        # this assert fires with b'B'. `RecallTrace.prune_to()` makes the revert reachable in the
        # live brain -- a pruned trace is a former state of that trace.
        save_named_state("revert.json", b"A")
        save_named_state("revert.json", b"B")
        assert save_named_state("revert.json", b"A") is True, (
            "a revert must be written, not deduped"
        )
        assert load_named_state("revert.json") == b"A", (
            "a state that reverted to an earlier value read back as the intermediate one"
        )
        # a new tenant must not disturb the others (the additive-tag contract, re-asserted)
        assert KnowledgeLog().count() == 4, "_T_STATE records leaked into the FACT view"
        assert any(p == foreign_rec for p in BinLog(kpath).replay()), (
            "_T_STATE writes dropped a foreign record"
        )
        # A name that cannot be encoded in the record is REFUSED BY NAME, not truncated into another
        # name's slot: the length prefix is one byte, so 256 chars would wrap to 0. The message is
        # checked, not merely the exception type -- with the guard deleted, `bytes([256])` raises its
        # own ValueError deeper down, so a bare `except ValueError` passes for the wrong reason and
        # the criterion is vacuous. MEASURED: with `len(raw) > 255` neutered this loop prints "no
        # red" as a bare type check and goes RED once the message is asserted.
        # the non-ascii name is spelled as a \u ESCAPE so this file's own bytes stay pure ASCII
        # (Windows cp1252). What is under test is the VALUE (an i-diaeresis in a state name), not
        # the source encoding, and an escape gives the same value without the mojibake risk this
        # project has already hit once in run_selfchecks.py.
        for bad_name, why in (("x" * 256, "256-byte name"), ("na\u00efve.json", "non-ascii name")):
            try:
                save_named_state(bad_name, b"x")
            except ValueError as exc:
                assert "state name" in str(exc), (
                    f"{why} was rejected by accident ({exc}), not refused by _state_key"
                )
            else:
                raise AssertionError(f"{why} must be refused by _state_key, not stored")

        # ---- one-time MIGRATION of a pre-P4b sidecar, and the file is LEFT ON DISK ----------------
        legacy = state_path("legacy_probe.json")
        with open(legacy, "w", encoding="ascii") as f:
            json.dump({"learned": 7}, f)
        assert load_state("legacy_probe.json") == {"learned": 7}, (
            "an existing sidecar must be imported, not silently reported as a cold start"
        )
        assert load_named_state("legacy_probe.json") == b'{"learned": 7}', (
            "the migration must leave the blob IN the lake, not just return it"
        )
        assert os.path.exists(legacy), (
            "the legacy sidecar must be left on disk, inert -- never deleted"
        )
        # and the imported state is durable in its own right: a SEPARATE process reads it back from
        # the lake with the sidecar renamed out of the way, so nothing can be re-reading the file.
        os.replace(legacy, legacy + ".moved")
        child_read = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            "from snn_data_lake.substrate.home import load_named_state\n"
            f"got = load_named_state('legacy_probe.json', {kpath!r})\n"
            "assert got == b'{\"learned\": 7}', got\n"
        )
        rc_m = subprocess.run([sys.executable, "-c", child_read]).returncode
        assert rc_m == 0, (
            f"a migrated state was not readable from the lake in a NEW process (rc={rc_m})"
        )

        print(
            "self-check OK: knowledge appended in one process is REPLAYED in a separate process (never "
            "forgotten across a restart); append-only; dedup; RAW-BINARY one lake (brain_lake.bin) with "
            "torn-tail recovery; LEARNED STATE lives IN the lake under _T_STATE (cross-process proven, "
            "latest-wins per name, revert-safe, survives sleep consolidation) and a pre-P4b sidecar is "
            "imported once and left on disk. The ONE shared home every entrypoint boots into."
        )
    finally:
        # RESTORE the caller's NORD_HOME, do not just drop it. A bare pop deletes the value the
        # CALLER set, not merely the one this self-check set: run in-process (tests/test_selfchecks.py
        # runs every module's __main__ in one interpreter), it silently un-isolated every module after
        # this one, so ~17 self-checks fell back to `~/.snn_data_lake` and read and wrote the
        # operator's live brain on every pytest run -- while the runner believed it had set an
        # isolated home.
        if _prior_home is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = _prior_home
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
