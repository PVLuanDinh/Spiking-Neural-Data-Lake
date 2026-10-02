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

import contextlib
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import threading

from snn_data_lake.substrate.binlog import (
    BinLog,
    pack_episode,
    pack_fact,
    unpack_episode,
    unpack_fact,
)

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "KnowledgeLog.consolidate": (
        "NOT_A_RULE",
        "replaces durable FACT records with the survivors. Durable-store compaction (P4b), caught on "
        "the verb",
    ),
}

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
# consumer (then reasoning/brain_adapters/lake_memory.py's `durable=`) a reasoning/ orphan off the
# live path. CLOSED 2026-08-11: that orphan parameter and its rebuild_from_durable() were DELETED
# when the module promoted to substrate/proximity_replay.py -- the retirement this comment
# started is complete, no consumer of the retired writer shape remains.
# Backlog H4/#10 named the missing prerequisite: a live producer of AER exemplars worth keeping across a
# restart. Until one exists there is nothing to store, so the dead write path is retired instead of kept
# as a format nobody writes. The tag stays reserved, every view SKIPS foreign records and consolidate()
# PRESERVES an unrecognised tag (its one prune is _T_STATE-only), so a historical lake containing spike
# records still replays fine -- and the class is one git revert away if the prerequisite materializes.
_T_FACT = b"\x01"
_T_SPIKE = b"\x02"
# _T_GROWTH added 2026-07-30: the grown neural substrate, in THE ONE LAKE rather than a sidecar file
# (owner decision). Safe as an additive tag because the existing readers already ignore what they do
# not recognise -- KnowledgeLog.replay filters on `payload[:1] == _T_FACT`, and consolidate() starts
# from `p for p in ... if p[:1] != _T_FACT`, so a foreign record is skipped by the FACT view and
# PRESERVED through a rewrite. That behaviour is self-checked (commit 9f620246), which is why this
# needed no change to either method. (consolidate() gained ONE deletion in 2026-08-05,
# `_prune_state_history`, and it is scoped to _T_STATE precisely so this tag's history stays whole --
# see that function and the _T_STATE comment for why growth history is kept and state history is not.)
_T_GROWTH = b"\x03"
# _T_EPISODE added 2026-08-04. Autobiographical episodes were being stored ONLY in
# sleep/episode_consolidation's in-memory attractor: measured, 59 episodes stored over 600
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
# view filters on `payload[:1] == _T_FACT` and consolidate() PRESERVES this tag (its one prune is
# _T_STATE-only, and this history is deliberately outside it), so existing readers skip this and a
# rewrite keeps it.
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
# view skips foreign records and consolidate() preserves an unrecognised tag. Successor for anyone
# holding the queued consumer edit: `save_named_state("signal_calib.json", blob)` / `load_named_state`.
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
# consolidate() preserves every foreign record EXCEPT superseded _T_STATE history (below), both
# asserted in main().
#
# WHAT THIS TRADED AWAY -- CLOSED 2026-08-05 by `_prune_state_history`, and the trade is recorded
# because the fix is scoped to ONE tag and the reason it is scoped is the load-bearing part. A
# sidecar was OVERWRITTEN in place, so it held exactly one copy of the state. A _T_STATE record is
# APPENDED, and nothing pruned the superseded copies. MEASURED on a scratch lake 2026-08-05, before
# the fix: 50 saves of a CHANGED 2,048 B blob under one name left 50 records (51 _T_STATE records
# with a second name beside them) and a full `consolidate()` afterwards pruned 0 of them; RE-RUN
# after the fix, the same input leaves 2 -- one per name -- and `load_named_state` returns the
# 50th blob, not the 1st. The unchanged-blob skip in `save_named_state` bounds a state that stops
# moving; it did nothing for one that keeps moving, and the live callers keep moving --
# `engine/brain.py` saves a 76 KB `reasoning_curriculum.json` on the sleep tick and
# `RecallTrace.save()` runs beside it. This is the exact shape that made _T_GROWTH 73,490 records /
# 508,082,042 B, 97.8 percent of the operator's lake.
#
# _T_GROWTH AND _T_SELFBUILD ARE DELIBERATELY NOT PRUNED, and that asymmetry is an owner decision
# rather than an oversight: both argue IN WRITING (see `save_growth` / `save_selfbuild`) that their
# older records are the growth/learning HISTORY -- the evidence a weight moved at all -- while a
# named state's older records are bytes no read path can reach again, because `load_named_state`
# and `_state_index` both take the last record under the name. "Latest-wins on read" is what makes
# the prune lossless here and is exactly what does not hold for the other two.
_T_STATE = b"\x07"

# _T_GROWTH_DELTA -- the reconstructible growth CHAIN (specs/022 T010-ii). A NEW record class, not a
# rewrite of _T_GROWTH: the legacy tag is CLOSED to npz writes from this commit (raw non-npz blobs
# still take it -- see save_growth) and every whole-substrate npz save lands here as a FULL record or
# a DELTA against the previous one. The measured pathology this exists to kill: 97.64% of the live
# lake was repeated whole-substrate _T_GROWTH snapshots (83,405 records) whose bytes differed ONLY in
# their `ticks`/`acc` members, so BinLog's byte dedup never fired and nothing could ever reclaim them.
# A DELTA carries the changed .npy members verbatim plus a sha256 manifest of the unchanged ones, so
# the chain is verified link by link on replay -- a tampered or missing link REFUSES (the reader holds
# the last verified state) rather than reconstructing a substrate nobody wrote.
#
# distinct-from: infra/self_build_loop.py -- its _firewalled_bytes snapshots the safety
#   envelope's FILES for byte-exact restore; no record framing, no chain. distinct-from:
#   sensorimotor/arm_bridge.py -- parses UDP reward packets. distinct-from: substrate/lake_engine.py --
#   AppendBudget counts bytes/tick; it labels this class, it does not encode it. All sites read.
_T_GROWTH_DELTA = b"\x08"

# specs/022 T031: COALITION records -- the workspace's settled coalitions (coalition_id,
# tick, proposer_ids, content_ref, base_level, current, typed channels), an IMMORTAL
# class per contracts/lake-records.md (the recall contract operates here; Episode
# consolidation consumes sequences of these). Written by the signal-ring journaller
# draining global_workspace's per-tick publications -- the ONE lake, one more record kind
# in it, never a second store. Payload = the Coalition record's own to_bytes framing
# (global_workspace.Coalition), CRC-covered by the journaller like every record.
# 0x0A, NOT 0x09: 0x09 is _T_SYNDELTA (lake_engine.SynapseDelta.TAG) -- a first draft of
# this constant took 0x09 and would have had replay parse coalition JSON as synapse
# deltas. The retention/disposition registries (lake_engine.RETENTION,
# lake_migrate.TAG_DISPOSITIONS) are the tag census; check BOTH before minting a tag.
_T_COALITION = b"\x0a"

# _T_GOAL added 2026-08-26 (specs/022 T046a): goal event rows -- CREATE (embedding +
# achievement_function_ref) / LP / OUTCOME -- written by growth/learning_progress's
# GoalStore as JSON body rows (the _T_COALITION precedent; pack_fact is the FACT-schema
# codec and silently drops foreign fields -- the store's round-trip red caught that draft),
# folded to the materialized view on replay. SYNTHESIS
# section 3.5 disposition: v1 READ-AS-IS (lake_migrate.TAG_DISPOSITIONS 0x0B; retention
# lake_engine.RETENTION 0x0B -- IMMORTAL: the brain's own intentions are never reclaimed).
# Both registries checked before minting, per the 0x09/0x0A lesson above.
_T_GOAL = b"\x0b"

# DELETED 2026-09-09 (T068 LOSS): `_T_UNITBLOCK` (0x0C) and `_T_BLOCKDELTA` (0x0D), minted
# 2026-09-08 for the block tier of the activation-driven paging layer, went with that layer when
# it measured a 35.2x tick cost against a 3.0x bound fixed before the run
# (specs/022 evidence/s5-paging-null.md 3.4/3.5). NO LAKE EVER HELD A RECORD UNDER EITHER,
# measured before the deletion rather than assumed: the live home holds brain_lake.bin and has no
# `segments/` directory and no `lake_index.db`, so both tags only ever saw scratch segment tiers
# inside self-checks and tests. Their RETENTION rows and TAG_DISPOSITIONS rows went in the same
# commit, so `registry_consistent()` stays true. The tag bytes 0x0C and 0x0D are FREE again --
# the highest tag in use is 0x0B, which is what the next mint must check, per the 0x09/0x0A
# lesson above.

# FULL cadence: a fresh FULL every 64 deltas bounds a cold reconstruction to <= 64 delta applies
# after the newest FULL. A storage-layer engineering constant (same class as lake_migrate's
# seal_every), not brain behaviour -- P1's infrastructure exception.
_GROWTH_CHAIN_EVERY = 64

#: resolved lake path -> {"hashes": {member: sha256}, "since_full": int} -- the writer's view of the
#: chain TAIL (the lake's last verified state, regardless of which object wrote it). Populated by
#: load_growth when the chain is the newest growth record; absent -> the next save writes a FULL,
#: which is the safe cutover (a chain must START with a FULL the reader can verify from).
_GROWTH_CHAIN = {}

_GC_FULL, _GC_DELTA = 0, 1


def _growth_members(blob):
    """The npz member census of `blob` as an ordered [(name, bytes)] list, or None when the blob is
    not an npz archive. np.savez writes a STORED zip of .npy members with epoch timestamps, so the
    chain never needs numpy -- stdlib zipfile decomposes and reassembles it byte-for-byte, keeping
    this module's zero-dep contract. Raw legacy blobs (and the tests' probe records) return None and
    take the whole-blob _T_GROWTH path."""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(bytes(blob))) as z:
            names = z.namelist()
            if not names or any(not n.endswith(".npy") for n in names):
                return None
            return [(n, z.read(n)) for n in names]
    except Exception:
        return None


def _growth_hash(name, data):
    return hashlib.sha256(name.encode("utf-8") + b"\x00" + data).digest()


def _growth_pack(kind, members, changed):
    """One chain record: [ver u8][kind u8][n u16] then per member
    [nlen u8][name][sha256 32B][changed u8][len u32 + bytes, changed only]."""
    out = [struct.pack("<BBH", 1, kind, len(members))]
    for name, data in members:
        raw = name.encode("utf-8")
        out.append(struct.pack("<B", len(raw)) + raw + _growth_hash(name, data))
        if name in changed:
            out.append(struct.pack("<BI", 1, len(data)) + data)
        else:
            out.append(struct.pack("<B", 0))
    return b"".join(out)


def _growth_unpack(payload):
    """-> (kind, [(name, hash, data_or_None)]), or None on ANY malformed byte -- a record this
    parser cannot fully account for (wrong version, short field, trailing bytes) is refused whole,
    never half-read; the _unpack_state incident in main() is why that is a rule here."""
    try:
        ver, kind, n = struct.unpack_from("<BBH", payload, 0)
        if ver != 1 or kind not in (_GC_FULL, _GC_DELTA):
            return None
        off, entries = 4, []
        for _ in range(n):
            nlen = payload[off]
            off += 1
            name = payload[off : off + nlen].decode("utf-8")
            off += nlen
            h = payload[off : off + 32]
            off += 32
            if len(h) != 32:
                return None
            changed = payload[off]
            off += 1
            data = None
            if changed:
                (size,) = struct.unpack_from("<I", payload, off)
                off += 4
                data = payload[off : off + size]
                off += size
                if len(data) != size:
                    return None
            entries.append((name, h, data))
        if off != len(payload):
            return None
        return kind, entries
    except Exception:
        return None


def _growth_apply(state, kind, entries):
    """Apply one chain record to `state` ({name: (hash, bytes)}, insertion-ordered). Returns the new
    state, or None when any hash refuses -- the caller then HOLDS the last verified state (the tamper
    red): a changed member whose bytes do not match its own manifest, a delta whose base is not what
    the chain holds, or a FULL missing a member's bytes all refuse the whole record."""
    new = {}
    for name, h, data in entries:
        if data is not None:
            if _growth_hash(name, data) != h:
                return None
            new[name] = (h, data)
        else:
            if kind == _GC_FULL:
                return None
            prev = None if state is None else state.get(name)
            if prev is None or prev[0] != h:
                return None
            new[name] = prev
    return new


def _growth_zip(state):
    """Reassemble a chain state into npz bytes -- the same container np.savez writes (STORED zip of
    .npy members, epoch timestamps), so `restore_growth_state` reads it unchanged."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, (_h, data) in state.items():
            z.writestr(zipfile.ZipInfo(name), data)
    return buf.getvalue()


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


@contextlib.contextmanager
def throwaway_home():
    """A self-check's home (specs/022 T037a): a fresh temp dir as NORD_HOME for the duration,
    the previous value restored after. Every main() that drives a So101Arm through
    EMERGENCY_STOP or HOME runs under this, because the E-STOP now writes a durable latch
    sidecar into home() -- a self-check run by hand against the real home would otherwise
    leave the live brain's next restart frozen. The runner's isolated NORD_HOME is replaced
    too: a check that exits stopped must not freeze the next check either."""
    prev = os.environ.get("NORD_HOME")
    # best-effort cleanup: a brain can still hold a lake handle in the temp home when the
    # check returns, and on Windows that makes rmdir raise -- which would turn a passed
    # self-check into exit 1 (measured on engine.brain)
    tmp = tempfile.TemporaryDirectory(prefix="selfcheck_home_", ignore_cleanup_errors=True)
    os.environ["NORD_HOME"] = tmp.name
    try:
        yield tmp.name
    finally:
        if prev is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prev
        tmp.cleanup()


def state_path(name):
    """Path to `<home>/<name>` -- the LEGACY sidecar location, kept for two honest uses only.

    (1) `_migrate_sidecar` reads it, once, to import a pre-2026-08-05 blob into the one lake.
    (2) Binary blobs the JSON facade cannot carry are still written here by their own writers:
        `engine/brain.py` -> `language_acq.npz` (numpy) and `sleep/nightly_skills.py (promoted 2026-08-11)`
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

#: Guards the check-then-construct in `_log()`. The cache above is READ and WRITTEN by every lake
#: caller, and under free-threading (3.14 `Py_GIL_DISABLED`) those callers run simultaneously.
#:
#: MEASURED, not reasoned about -- 40 trials x 12 threads on an isolated scratch lake, 2026-08-05,
#: `.venv-lake` free-threaded build: WITHOUT this lock `_log()` handed out up to **12 distinct
#: BinLog instances over the ONE file** (40/40 trials split), and because each instance carries its
#: own `_seen` set, the SAME payload was durably written to the lake up to **5 times with
#: dedup=True** (40/40 trials produced a duplicate). That is the precise failure the comment above
#: predicts in the abstract -- "two instances over one file hold two divergent dedup sets" -- now
#: reproduced. It is worse than a duplicate record, too: `BinLog.__init__` runs `_repair_torn_tail()`,
#: which TRUNCATES, so the unlocked path also had 13 concurrent constructions racing to truncate the
#: one lake while other threads appended to it.
#:
#: COST, measured on the same build over 5 x 200,000 steady-state (cache-hit) calls: `_log()` costs
#: 38.526 us/call with the lock against 36.789 us/call without it, so the lock itself is
#: **1.737 us/call, +4.7 percent of a `_log()`** -- and a `_log()` is dominated by the
#: `os.path.getsize` in `stale()`, not by anything this lock added. Against the operation a `_log()`
#: exists to serve, one durable `append()` at 498 us/record (measured, same build), the lock is
#: **0.35 percent**. Quote the pair, not the percentage alone: the ratio moves with lake size
#: because `stale()` does not, and this file has already had three numbers rot for that reason.
#:
#: Holding it ACROSS the construction is deliberate and is not a pessimisation: the unlocked code
#: did that 388-538 ms construction (124k-record / 522 MB scratch lake) 12 times in parallel and
#: corrupted the lake; this does it once while the others wait and then hands them the same instance.
#:
#: A PLAIN Lock, not an RLock, because no path re-enters: `load_named_state` calls `_log()` and only
#: then calls `_migrate_sidecar` -> `save_named_state` -> `_log()`, which is sequential, not nested,
#: and nothing inside `BinLog.__init__` calls back into this module. A future re-entrant caller must
#: revisit this line; a deadlock here is loud, which is the safer failure for the one durable store.
_LOG_LOCK = threading.Lock()


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

    The check and the construction are ONE critical section under `_LOG_LOCK` -- see that lock for
    the measured duplicate-record red this closes. Splitting them is what made a shared cache hand
    out unshared instances.
    """
    key = (path or knowledge_path(), bool(dedup))
    with _LOG_LOCK:
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
        payload = _T_FACT + pack_fact(record)
        written = self._bin.append(payload)
        if written:
            observe_append(_T_FACT, len(payload))
        if written:
            self._count += 1
        return written

    def extend(self, records):
        """Durably append many records (one fsync at the end -- for bulk import). Returns #written.

        MEASURED 2026-08-05 and stated here because the number is large and the method is unused:
        one `extend(N)` against N x `append()` on isolated scratch lakes is a TIE at N=1, 1.9x at
        N=2, 21.3x at N=32 and 1341x at N=10,000 -- the crossover is N=2 and there is no width at
        which batching loses. The full sweep, the cost breakdown of an `append()`, and the
        DURABILITY WINDOW batching buys (a crash can lose any subset of the batch, up to all of it,
        so an episode stream must NOT be buffered across ticks) live on `BinLog.extend`.

        THIS METHOD HAS NO LIVE CALLER, and neither does the `BinLog.extend` it forwards to --
        measured the same day: the only `.extend(` on a lake object in `src/`, `tests/` or `tools/`
        is this line and binlog's own self-check. So the fast path is P3 shape 1 (method never
        called) while the slow one is in use; `engine/brain.py`'s schema induction is the live loop
        that would fit it (up to `budget=32` `log.append(...)` calls per fourth night, 21.3x at that
        width). Recorded rather than wired: that edit is in a file this pass does not own."""
        payloads = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("KnowledgeLog.extend: every record must be a dict")
            payloads.append(_T_FACT + pack_fact(record))
        written = self._bin.extend(payloads)
        # Only the payloads that actually LANDED. The first form reported every payload it built, so
        # dedup-dropped records were counted as lake traffic that never reached the disk -- an accounting
        # hook must mean bytes appended. extend returns a count, not a mask, so the bytes are attributed
        # to the LARGEST `written` payloads: an over-attribution of at most the difference between the
        # largest and smallest duplicate, and never an invented record. DISCLOSED rather than exact.
        for _p in sorted(payloads, key=len, reverse=True)[: int(written)]:
            observe_append(_T_FACT, len(_p))
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
        consolidation policy chose -- redundant merged, re-derivable/stale pruned), preserving every
        foreign record in the one lake EXCEPT superseded _T_STATE history. Atomic compaction via
        BinLog.rewrite. This is the PRUNE half of sleep consolidation -- without it the symbolic lake
        only ever grows. Returns (kept, dropped), both counting FACT records only.

        `_prune_state_history` is the ONLY thing this drops besides facts, and it is scoped to
        _T_STATE deliberately: a named state has exactly one current value, so every earlier record
        under that name is superseded bytes that `load_named_state` will never return again.
        _T_GROWTH and _T_SELFBUILD are latest-wins in the same way and are NOT pruned, because both
        argue in writing (see `save_growth` / `save_selfbuild`) that their history is the evidence
        their learning moved at all -- that is an owner call about what the brain keeps, not a
        property of the record layout, and it is why this is a per-tag prune and not a general one.
        """
        from snn_data_lake.sleep.lake_consolidation import _src_class, record_key
        from snn_data_lake.substrate.lake_engine import retention_of

        # THE LABEL IS LOAD-BEARING HERE, not merely declared elsewhere. _T_FACT is IMMORTAL with
        # reclaimable=False, so a rewrite that would lose a LEARNED fact is refused outright; only the
        # per-record exemption RETENTION[0x01] declares (bulk re-derivable ingests, which reconstruct at
        # boot) may be dropped. Before specs/022 L6 (2026-09-06) nothing here read the label at all, and the
        # sleep tick's cap could silently discard learned knowledge the registry called never-reclaimed.
        if not retention_of(_T_FACT[0])[1]:
            keep_keys = {record_key(r) for r in keep_records}
            lost = [
                r
                for r in self.replay()
                if _src_class(r) != "re-derivable" and record_key(r) not in keep_keys
            ]
            if lost:
                raise ImmortalRefusal(
                    f"consolidate would drop {len(lost)} LEARNED _T_FACT record(s), but 0x01 is "
                    f"{retention_of(_T_FACT[0])[0]} with reclaimable=False. Only the per-record "
                    "exemption in RETENTION[0x01]'s reason (bulk re-derivable ingests) may be reclaimed."
                )
        foreign = _prune_state_history(
            [p for p in self._bin.replay() if p[:1] != _T_FACT]  # growth/episode/spike stay as-is
        )
        before = self._count
        self._bin.rewrite(foreign + [_T_FACT + pack_fact(r) for r in keep_records])
        self._count = self._bin.count() - len(foreign)
        # The cached {name: (seq, blob)} index is DERIVED from the _T_STATE records this rewrite
        # just replaced, and nothing else drops it: `BinLog.rewrite` maintains `_size`, so
        # `stale()` stays False and `_log()` hands the same instance -- and its index -- straight
        # back. Invalidating belongs here rather than in `rewrite()` because a _T_STATE record is a
        # home.py concept; BinLog carries opaque payloads and must not learn what is inside them.
        #
        # STATED HONESTLY, because the neuter was run rather than reasoned about: with the prune
        # rule as written this line is a NO-OP. `_state_index` points at the last record per name
        # and `_prune_state_history` keeps exactly that record, so the cache cannot disagree with
        # the lake -- measured 2026-08-05 with this line neutered, across save/consolidate/save/
        # revert/consolidate, cached == disk (a separate process) at every step, 0 divergences.
        # It is kept as a COUPLING BREAKER, and the coupling is one token wide: with the same
        # neuter and `last[got[0]] = i` changed to `last.setdefault(got[0], i)`, the same probe
        # prints `cached=b'v3'  disk=b'v1'` -- a live process serving named state that is no longer
        # in the lake. Two functions ninety lines apart currently agree by construction; this line
        # is what stops that agreement from being load-bearing. The cost is one extra replay on the
        # sleep tick that just paid for a full replay and a full rewrite of the same file.
        self._bin.__dict__.pop("_snn_state_index", None)
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


def _unpack_state(payload):
    """`(name, seq, blob)` for ONE _T_STATE record, or None if its layout does not parse.

    THE parser for the layout documented at the `_T_STATE` tag -- tag + [u8 len(name)] + name +
    [u32 seq little-endian] + blob. `_state_index` builds its cache with it and
    `_prune_state_history` decides what to delete with it, so the two cannot disagree about which
    record is the newest under a name by disagreeing about where that name ends.

    `name` comes back as the ascii BYTES `_state_key` writes, never as a `str`, and that is the
    prune's safety rather than a style choice: `decode("ascii", "replace")` maps EVERY non-ascii
    byte to one U+FFFD, so two names that `_state_key` would both refuse would compare EQUAL and
    the prune would delete one as superseded history of the other. Nothing this module writes can
    produce such a name; the prune reads a lake that other builds also write to.

    None means UNPARSEABLE -- a foreign record wearing this tag, a build writing a different
    layout, a restored backup. NOT a torn append: `_repair_torn_tail` truncates at the first short
    FRAME and `replay` stops at one, so the only malformed record that reaches here is a
    well-framed payload this module cannot read, and nothing downstream will remove it for us.
    Both callers treat it as "not one of mine": the index skips it, the prune KEEPS it. Preserving
    a record we cannot read is the recoverable direction; deleting it is not. (`_state_index`
    indexed such a record on garbage offsets until 2026-08-05, and a 1-byte payload raised
    IndexError out of it -- both re-measured 2026-08-05, and the prune inherits the same parser, so
    the pre-2026-08-05 form takes a lake of 6 `moving.json` records + one such payload from 7
    _T_STATE records to 1. Reddened in main().)
    """
    n = payload[1] if len(payload) > 1 else 0
    if n == 0 or len(payload) < 6 + n:
        return None
    return payload[2 : 2 + n], int.from_bytes(payload[2 + n : 6 + n], "little"), payload[6 + n :]


#: The APPEND OBSERVER (specs/022 L4, FR-003). Called `(tag_int, n_bytes)` at every site in THIS MODULE
#: that appends a payload whose tag this module built: 0x01 (facts, singly and in bulk), 0x03 and 0x08
#: (growth, legacy and chain), 0x04 (episodes), 0x05 (self-build) and 0x07 (named state).
#: NOT OBSERVED, and named rather than implied: 0x0A (coalitions) frames onto BinLog through the coalition
#: journaller and never enters this module, and 0x0B (goals) writes through its own store. The first form
#: of this hook sat at three sites covering 0x01 and 0x07 only, while its commit claimed it bounded the
#: five baselined tags -- an adversarial panel measured a live brain writing 0x04 and 0x0A that the budget
#: never saw, and constructed floods on both reading CALM. It is here rather than in BinLog because a tag is a home
#: concept and BinLog carries opaque payloads by design; it is module-global because this module's log cache
#: already is, and threading a budget object through every writer would couple far more than it bounds.
#: DISCLOSED: one observer per process, so two brains in one process would share it -- which P4 forbids for
#: other reasons anyway. A raising observer must never break a write: the call is guarded.
_APPEND_OBSERVER = None


def set_append_observer(fn):
    """Install (or clear, with None) the process-wide append observer. Returns the previous one, so a caller
    can restore what it replaced rather than assuming it was the only one."""
    global _APPEND_OBSERVER
    prev = _APPEND_OBSERVER
    _APPEND_OBSERVER = fn
    return prev


# distinct-from: substrate/lake_engine.py::AppendBudget -- that is the OBSERVER that gets installed
#   here (it accumulates bytes/tick per tag and judges a flood); this is the CALL SITE hook that
#   feeds it. One would be useless without the other and neither re-implements the other.
# distinct-from: substrate/binlog.py::BinLog.extend -- the bulk WRITE path (one fsync for many
#   records). It carries opaque payloads by design and knows no tag, which is why the observation
#   lives here rather than there.
# distinct-from: reservoir/kernel_tick.py::compile_bundles -- "THE ONE" phrasing only; it compiles
#   delivery bundles from edges and touches no lake record.
# distinct-from: sensorimotor/afferent_boundary.py -- "the ONE transduction-in boundary": sensor
#   readings into the stimulus contract, no record, no tag, no bytes/tick.
# distinct-from: faculties/raphe_serotonin.py::declare_ports -- "the FIRST real manifest
#   registrant": a typed wiring-manifest port declaration, not an append observation.
#   All five sites read 2026-09-08.
def observe_append(tag: bytes, n_bytes: int) -> None:
    """THE ONE append-observation path (specs/022 T067). `tag` is a `_T_*` constant, `n_bytes` the
    length of the TAG-PREFIXED payload (never the framed record).

    PROMOTED FROM `_observe_append` with every caller updated, rather than wrapped: a documented
    public function beside a private one would be two names for one observer, and the class this
    row mints writes through `lake_segments`/`UnitBlockStore` rather than through a `save_*` here,
    so it has to call the observer at its own write site. P2 -- one observer, one
    `set_append_observer` hook, one guarded call. The guard is around the OBSERVER only:
    accounting must never break a durable write."""
    fn = _APPEND_OBSERVER
    if fn is None:
        return
    try:
        fn(tag[0], int(n_bytes))
    except Exception:  # noqa: BLE001 - accounting must never break a durable write
        pass


class ImmortalRefusal(ValueError):
    """`consolidate` refused a rewrite that would destroy a record of a non-reclaimable class.

    A ValueError subclass so existing callers that catch ValueError still catch it, and a distinct TYPE so
    the sleep tick can count THIS and nothing else. It shipped as a bare ValueError for one commit, and the
    adversarial panel reproduced two unrelated ValueErrors from the same block being counted as retention
    drift: a lake file that is not a BinLog (binlog.py's magic check, reachable live because the cached log
    is rebuilt whenever it goes stale) and a `recall_trace.json` holding valid JSON that is not an object.
    Neither is a retention defect, and both were being published as one.
    """


def _prune_state_history(payloads):
    """`payloads` with every SUPERSEDED _T_STATE record dropped -- the LAST per name survives.

    The compaction half of `_T_STATE`. A named state has one current value: `load_named_state`
    returns the last record under the name and `_state_index` indexes only that one, so every
    earlier record is bytes no read path can ever reach again. Without this the one lake grows by a
    full copy of the state on every sleep tick that changes it -- MEASURED on a scratch lake
    2026-08-05, 50 saves of a CHANGED 2,048 B blob under one name left 50 records and a full
    `consolidate()` pruned 0 of them. The same shape is what made _T_GROWTH 97.8 percent of the
    operator's lake at the 2026-08-05 scan recorded under `_T_STATE`; do not read that ratio as
    current -- a `stat` of the same file later the same day returned 835,315,078 B against the
    519,646,815 B of that scan, and the tag census behind the ratio was NOT re-derived (a BinLog
    over the live lake would run `_repair_torn_tail` and truncate it under the running daemon).

    Order is preserved and only _T_STATE records are considered, so every other tenant comes back
    in the sequence it went in -- which `load_growth` / `load_selfbuild` depend on, both taking the
    LAST record of their own tag. A record whose layout `_unpack_state` cannot parse is KEPT.
    """
    # THE LABEL IS LOAD-BEARING HERE TOO (specs/022 L6 fold, 2026-09-06). 0x07 is IMMORTAL with
    # reclaimable=False, and this function drops records under it -- which the first L6 landing missed
    # entirely, because its guard was scoped to _T_FACT and the census accepted one retention_of token as
    # covering the whole of `consolidate`. What licenses the drop is the PER-RECORD exemption RETENTION[0x07]
    # now states: a SUPERSEDED frame is unreachable, so losing it loses no state. If that exemption is ever
    # removed from the reason, this returns the payloads untouched rather than dropping a class the registry
    # says is never reclaimed.
    from snn_data_lake.substrate.lake_engine import retention_of

    _label, _reclaimable, _reason = retention_of(_T_STATE[0])
    if not _reclaimable and "SUPERSEDED" not in _reason:
        return list(payloads)  # the label forbids it and states no exemption: drop nothing
    last = {}
    for i, payload in enumerate(payloads):
        if payload[:1] == _T_STATE:
            got = _unpack_state(payload)
            if got is not None:
                last[got[0]] = i  # a later record under the same name supersedes this one
    out = []
    for i, payload in enumerate(payloads):
        if payload[:1] == _T_STATE:
            got = _unpack_state(payload)
            if got is not None and last[got[0]] != i:
                continue
        out.append(payload)
    return out


def _state_index(log):
    """{name: (seq, blob)} for `log` -- built once per BinLog instance, maintained on our appends.

    A CACHE, never the source of truth (P4b's own wording). It is attached to the instance `_log()`
    hands out, and `_log()` REPLACES that instance whenever `BinLog.stale()` says the file changed
    underneath it -- so records written by another process are picked up by the rebuild rather than
    missed. Attached to the instance rather than kept in a dict keyed by path precisely so that
    "the log was rebuilt" and "the index is void" are the same event and cannot drift apart.

    THE ONE PATH THAT BREAKS THAT EQUIVALENCE is a rewrite of the lake by THIS process:
    `BinLog.rewrite` maintains `_size`, so `stale()` stays False and `_log()` hands the same
    instance -- and this index -- back over a file whose _T_STATE records it did not choose. That
    is why `KnowledgeLog.consolidate` drops the attribute itself after compacting; see the comment
    at that call for why the invalidation lives in home.py and not in `rewrite()`.

    Why it exists at all: one full replay of the operator's live lake (516,810,138 B / 123,120
    records) measured 349 / 376 / 378 ms on 2026-08-05, and `engine/brain.py` calls
    `self._time_clock.save()` on every tick the clock grows and `trace.save()` on every sleep tick.
    Paying that pass per call would be a third of a second on the tick path."""
    idx = getattr(log, "_snn_state_index", None)
    if idx is None:
        idx = {}
        for payload in log.replay():
            if payload[:1] != _T_STATE:
                continue
            got = _unpack_state(payload)  # ONE parser, shared with the prune -- see _unpack_state
            if got is not None:
                idx[got[0].decode("ascii", "replace")] = (got[1], got[2])
        log._snn_state_index = idx
    return idx


def save_named_state(name, blob, path=None):
    """Store `blob` (bytes) under `name` in THE ONE LAKE (_T_STATE). True if written, False if not.

    False means one of two things, both deliberate. A falsy blob is REFUSED, for the reason
    `save_growth` refuses one: persisting "nothing" as though it were a measurement restores an
    empty learner with a confident tally beside it. An UNCHANGED blob is also not written -- the
    lake is append-only and the only compaction runs on a sleep tick, so re-storing a byte-identical
    state every time would grow the one store between two sleeps without adding one bit of
    information. A CHANGED state always lands, including a change back to an earlier value (see the
    `seq` field in the _T_STATE tag comment).

    (That second sentence read "consolidate() preserves every non-FACT record forever" until
    2026-08-05, when `_prune_state_history` made it false for exactly this tag. The skip is still
    worth having and its justification is now the weaker one written above: consolidate bounds this
    growth at each sleep, the skip bounds it continuously, and a brain can go a long time between
    sleeps.)"""
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
    _payload = _T_STATE + bytes([len(key)]) + key + seq.to_bytes(4, "little") + blob
    written = log.append(_payload)
    if written:
        observe_append(_T_STATE, len(_payload))
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
    imports whatever bytes are at `<home>/<name>`, permanently: the imported record is the newest
    under its name, so `consolidate()`'s _T_STATE prune keeps it until a real save supersedes it,
    and a name nothing ever saves keeps its junk forever. MEASURED 2026-08-05: a
    `<home>/corrupt.json` containing `{not json` is imported as
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
    and this decides where it lives: `engine/brain.py` (reasoning_curriculum.json),
    `sensorimotor/closed_loop.py` (valence.json), `sleep/lake_consolidation.py`
    (recall_trace.json), `faculties/time_cells.py` (time_cells.json), `reasoning/gf.py`
    (gf_library.json, active_library.json), `sleep/nightly_skills.py (promoted 2026-08-11)`. Not one of them changed a
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
    otherwise restore an empty engine with a confident accuracy beside it.

    THE CHAIN (specs/022 T010-ii). An npz blob no longer lands whole: it is decomposed into its .npy
    members and appended under _T_GROWTH_DELTA as a FULL record (chain start, or every
    _GROWTH_CHAIN_EVERY deltas) or a DELTA carrying only the members whose bytes changed plus a hash
    manifest of the rest. A byte-identical substrate writes NOTHING (returns 0) -- that is the
    97.64%-repeated-snapshot pathology being refused at the door rather than compacted later. The
    surgery lives HERE, in the one persistence mechanism, so both live callers
    (`Brain._flush_growth`, `Brain._make_neuro`) and the engine's real-interpreter restart proof
    inherit the chain without an edit -- P2: one mechanism, not a delta path beside a whole-blob
    path. A blob that is NOT an npz archive (the tests' raw probe records) keeps the legacy
    whole-blob _T_GROWTH contract unchanged."""
    if not blob:
        return 0
    members = _growth_members(blob)
    if members is None:
        _payload = _T_GROWTH + bytes(blob)
        _w = _log(path).append(_payload)
        if _w:
            observe_append(_T_GROWTH, len(_payload))
        return _w
    key = os.path.abspath(path or knowledge_path())
    st = _GROWTH_CHAIN.get(key)
    hashes = {name: _growth_hash(name, data) for name, data in members}
    if st is None or st["since_full"] >= _GROWTH_CHAIN_EVERY:
        kind, changed = _GC_FULL, set(hashes)
    else:
        changed = {name for name in hashes if st["hashes"].get(name) != hashes[name]}
        if not changed and set(st["hashes"]) == set(hashes):
            return 0  # byte-identical substrate: nothing to write, nothing to reclaim later
        kind = _GC_DELTA  # a census change with no changed bytes still writes (all-manifest delta)
    _payload = _T_GROWTH_DELTA + _growth_pack(kind, members, changed)
    wrote = _log(path).append(_payload)
    if wrote:
        observe_append(_T_GROWTH_DELTA, len(_payload))
    if wrote:
        _GROWTH_CHAIN[key] = {
            "hashes": hashes,
            "since_full": 0 if kind == _GC_FULL else st["since_full"] + 1,
        }
    return wrote


def save_episode(episode, path=None):
    """Append one autobiographical episode to THE ONE LAKE. True if written, False if a duplicate.

    The lake, not a sidecar and not an in-memory attractor, because an episode that does not
    survive the process is not a memory. Deduping is the BinLog's, on the packed bytes, and
    pack_episode sorts its slots so the same episode is the same bytes.
    """
    if not episode:
        return False
    _payload = _T_EPISODE + pack_episode(episode)
    _w = _log(path).append(_payload)
    if _w:
        observe_append(_T_EPISODE, len(_payload))
    return _w


def load_episodes(path=None, limit=None, kinds=None):
    """Autobiographical episodes by default, or explicitly selected kinds, oldest first.

    Foreign record types are skipped, exactly as KnowledgeLog skips non-FACT records -- one lake,
    several tenants, each reading only its own.
    """
    out = []
    for payload in _log(path).replay():
        if payload[:1] == _T_EPISODE:
            try:
                episode = unpack_episode(payload[1:])
                kind = episode.get("kind")
                if (kind != "teach") if kinds is None else (kind in kinds):
                    out.append(episode)
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
    _payload = _T_SELFBUILD + bytes(blob)
    _w = _log(path).append(_payload)
    if _w:
        observe_append(_T_SELFBUILD, len(_payload))
    return _w


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


# THE one growth loader, edited in place for the chain -- not a second mechanism.
# distinct-from: growth/neurogenesis_engine.py -- growth_state_bytes PRODUCES the blob this returns.
# distinct-from: substrate/lake_memory.py -- channel-range reads of the .spkl store, a different lake.
# distinct-from: substrate/lake_working_memory.py -- same .spkl channel reads, in-memory half.
# distinct-from: growth/similarity_sized_growth.py -- routes to a frozen expert, no lake records.
# distinct-from: substrate/lake_bridge.py -- iter_records yields raw payloads with no tag semantics.
# All sites read.
def load_growth(path=None):
    """The most recent grown-substrate blob, or None if the lake holds none.

    None is the honest answer for a lake written before this record type existed, and the caller
    (`Brain._make_neuro`) treats it as "start ungrown" -- so an old lake behaves exactly as it did
    before, which is the backward-compatibility requirement.

    LATEST-WINS ACROSS BOTH CLASSES (specs/022 T010-ii): the replay tracks the newest legacy
    _T_GROWTH blob AND the _T_GROWTH_DELTA chain state side by side, and whichever holds the LATER
    lake position is the substrate returned -- so a pre-chain lake restores exactly as before, a
    post-cutover lake restores from the chain, and a raw record written after the chain still
    shadows it (the same latest-wins contract the legacy tag always had). Chain records are
    verified link by link: a record whose hashes refuse is SKIPPED and the state holds at the last
    verified link -- the tamper red -- and a verified FULL always resets the chain, so one bad link
    costs at most the deltas up to the next FULL, never the substrate. A chain win also primes the
    writer's tail cache so the next save is a DELTA rather than a redundant FULL."""
    latest_legacy, legacy_pos = None, -1
    state, chain_pos, since_full = None, -1, 0
    pos = 0
    for payload in _log(path).replay():
        pos += 1
        tag = payload[:1]
        if tag == _T_GROWTH:
            latest_legacy, legacy_pos = payload[1:], pos
        elif tag == _T_GROWTH_DELTA:
            parsed = _growth_unpack(payload[1:])
            if parsed is None:
                continue  # a record we cannot read must not stop the ones after it
            kind, entries = parsed
            new = _growth_apply(state if kind == _GC_DELTA else None, kind, entries)
            if new is None:
                continue  # hash refused: hold the last VERIFIED state, never guess
            state, chain_pos = new, pos
            since_full = 0 if kind == _GC_FULL else since_full + 1
    if chain_pos > legacy_pos and state is not None:
        _GROWTH_CHAIN[os.path.abspath(path or knowledge_path())] = {
            "hashes": {name: h for name, (h, _data) in state.items()},
            "since_full": since_full,
        }
        return _growth_zip(state)
    return latest_legacy


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


def _arm_selfcheck_unguarded(text):
    """T037a census rule for ONE module source: its self-check drives a So101Arm / AutonomousArm
    through EMERGENCY_STOP (so it writes the durable latch sidecar into home()) and is guarded
    neither by throwaway_home() nor by an explicit NORD_HOME set before its first construction.
    Returns True when the module would write a by-hand run's stop into the live home."""
    import ast

    if "EMERGENCY_STOP" not in text or not re.search(r"(So101Arm|AutonomousArm)[(]", text):
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return True  # unparseable: fail closed
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("main", "_main")]
    if not mains:
        return False  # no self-check entry: nothing runs by hand
    if "throwaway_home" in text:
        return False
    for fn in mains:
        seg = ast.get_source_segment(text, fn) or ""
        ctor = re.search(r"(So101Arm|AutonomousArm)[(]", seg)
        guard = seg.find('setdefault("NORD_HOME"')
        if ctor and (guard < 0 or guard > ctor.start()):
            return True
    return False


def _arm_selfcheck_guard_census():
    """Every src module whose self-check can write the T037a sidecar, minus the guarded ones.
    The docstring claim 'every self-check that drives the gate runs under throwaway_home()' is
    policed here rather than trusted -- the third review round found afferent_boundary.py with
    its NORD_HOME guard 56 lines AFTER the E-STOP it trips."""
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for r, _d, files in os.walk(pkg):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(r, f)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    if _arm_selfcheck_unguarded(fh.read()):
                        out.append(os.path.relpath(p, pkg))
    return sorted(out)


def main():
    import shutil
    import subprocess
    import tempfile

    # T037a: the guard census, with its red constructed on synthetic sources
    unguarded = _arm_selfcheck_guard_census()
    assert not unguarded, (
        f"self-checks that can write the durable E-STOP sidecar into the LIVE home: {unguarded} "
        "-- wrap main() in throwaway_home() (or set NORD_HOME before the first arm construction)"
    )
    _bare = (
        "import os\nfrom snn_data_lake.brainstem.so101_arm import So101Arm\n"
        "def main():\n    a = So101Arm()\n    a.apply('EMERGENCY_STOP')\n"
    )
    assert _arm_selfcheck_unguarded(_bare), "census red: a bare arm self-check read as guarded"
    _late = _bare.replace(
        "    a.apply('EMERGENCY_STOP')\n",
        "    a.apply('EMERGENCY_STOP')\n    os.environ.setdefault(\"NORD_HOME\", 'x')\n",
    )
    assert _arm_selfcheck_unguarded(_late), (
        "census red: a guard AFTER the construction read as guarded"
    )
    _early = _bare.replace(
        "def main():\n", "def main():\n    os.environ.setdefault(\"NORD_HOME\", 'x')\n"
    )
    assert not _arm_selfcheck_unguarded(_early), "census: a guard BEFORE the construction must pass"
    _wrapped = _bare.replace(
        "def main():\n", "def main():\n    with throwaway_home():\n        pass\n"
    )
    assert not _arm_selfcheck_unguarded(_wrapped), "census: throwaway_home must pass"
    assert not _arm_selfcheck_unguarded(_bare.replace("EMERGENCY_STOP", "HOLD")), (
        "census: no E-STOP, no sidecar"
    )
    print(
        "arm self-check guard census: 0 unguarded; reds fired (bare, late guard); controls held "
        "(early guard, throwaway_home, no E-STOP)"
    )

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
        #
        # EXTENDED 2026-08-05 from the reserved tag ALONE to EVERY tenant, because `consolidate()`
        # stopped being a pure preserver on that date: it now deletes superseded _T_STATE history
        # (`_prune_state_history`). One reserved-tag record could not tell "the prune is scoped to
        # _T_STATE" apart from "the prune eats whatever it does not recognise" -- both leave a
        # _T_SPIKE record standing, and only the first is correct. Each tenant below is written
        # TWICE, so a prune that generalised to latest-wins across tags would halve them and be
        # caught here rather than by a brain that boots without its grown substrate.
        def _tag_census(p):
            """{tag byte: count} over every record in the lake at `p` -- read-only, frame by frame."""
            out = {}
            for rec in BinLog(p).replay():
                out[rec[:1]] = out.get(rec[:1], 0) + 1
            return out

        foreign_rec = _T_SPIKE + b"reserved-tag payload (retired AER exemplar shape)"
        unknown_rec = b"\x7f" + b"an UNKNOWN tag: no tenant in this module owns 0x7f"
        assert BinLog(kpath).append(foreign_rec), "foreign record append failed"
        assert BinLog(kpath).append(foreign_rec + b" (second)"), "second foreign record failed"
        assert BinLog(kpath).append(unknown_rec), "unknown-tag record append failed"
        assert BinLog(kpath).append(unknown_rec + b" (second)"), "second unknown-tag record failed"
        assert save_growth(b"grown-substrate v1") and save_growth(b"grown-substrate v2"), (
            "growth history must be two DISTINCT records for the prune to be able to eat one"
        )
        assert save_selfbuild(b'{"w": [0.1]}') and save_selfbuild(b'{"w": [0.2]}'), (
            "self-build history must be two DISTINCT records"
        )
        assert save_episode({"who": "spike", "what": "watched a consolidation"}), "episode 1"
        assert save_episode({"who": "spike", "what": "watched another one"}), "episode 2"
        tenants_before = _tag_census(kpath)
        facts_now = (
            KnowledgeLog().count()
        )  # the FACT view reads ONLY facts (foreign records are skipped)
        assert facts_now == 4, f"foreign records leaked into the FACT count: {facts_now}"
        klog = (
            KnowledgeLog()
        )  # sleep consolidation of the FACT side must PRESERVE the foreign records
        klog.consolidate(list(klog.replay()))
        tenants_after = _tag_census(kpath)
        assert any(p == foreign_rec for p in BinLog(kpath).replay()), (
            "sleep consolidation of facts dropped a foreign (reserved-tag) record"
        )
        assert any(p == unknown_rec for p in BinLog(kpath).replay()), (
            "sleep consolidation of facts dropped a record whose tag it does not recognise"
        )
        for tag, who in (
            (_T_GROWTH, "_T_GROWTH (the grown substrate)"),
            (_T_SELFBUILD, "_T_SELFBUILD (the self-build learner)"),
            (_T_EPISODE, "_T_EPISODE (autobiographical episodes)"),
            (_T_SPIKE, "_T_SPIKE (reserved)"),
            (b"\x7f", "an unknown/reserved tag"),
        ):
            assert tenants_after.get(tag, 0) == tenants_before.get(tag, 0) == 2, (
                f"consolidate must leave {who} untouched: "
                f"{tenants_before.get(tag, 0)} -> {tenants_after.get(tag, 0)}, expected 2 -> 2"
            )
        # ORDER, not just count: `load_growth` / `load_selfbuild` take the LAST record of their tag
        # and `load_episodes` returns oldest-first, so a compaction that preserved every record but
        # resequenced them would restore the brain's PREVIOUS substrate while every count above
        # stayed green. RED, run: `return out[::-1]` at the end of `_prune_state_history`.
        assert load_growth() == b"grown-substrate v2", "the newest growth record must still be last"
        assert load_selfbuild() == b'{"w": [0.2]}', (
            "the newest self-build record must still be last"
        )
        assert [e["what"] for e in load_episodes()] == [
            "watched a consolidation",
            "watched another one",
        ], f"consolidate resequenced or dropped an episode: {load_episodes()}"
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
        # UNCHANGED state is not re-stored: the lake is append-only between two sleep ticks, so
        # saving an identical blob in that window must add nothing. (This comment said "consolidate()
        # keeps every foreign record forever" until `_prune_state_history` landed; the skip still
        # earns its place, on the narrower justification written in `save_named_state`.)
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

        # ---- SLEEP CONSOLIDATION PRUNES SUPERSEDED _T_STATE HISTORY (`_prune_state_history`) -----
        # The one deletion `consolidate()` performs outside the FACT side. A named state has ONE
        # current value, so every earlier record under that name is bytes no read path can reach
        # again -- and leaving them there is what made _T_GROWTH 97.8 percent of the operator's lake.
        # MEASURED on a scratch lake before this landed: 50 saves of a CHANGED 2,048 B blob under one
        # name left 50 records and a full consolidate() pruned 0 of them.
        #
        # THE ASSERT THAT CARRIES THIS IS THE VALUE ONE, NOT THE COUNT ONE, and the two are separated
        # deliberately: keeping the FIRST record per name also leaves exactly one record per name, so
        # the count assert passes against a prune that threw the current state away and restored an
        # old one. Both reds were CONSTRUCTED AND RUN 2026-08-05, on this self-check:
        #   * `return list(payloads)` (no prune -- the pre-2026-08-05 behaviour) -> the count assert
        #     fires: "13 records in, 13 out, expected 13 -> 5";
        #   * `last[got[0]] = i` -> `last.setdefault(got[0], i)` (keep the FIRST record per name) ->
        #     the count assert stays GREEN at 13 -> 5 and the VALUE assert below fires. That is the
        #     whole reason the value assert exists as a separate line.
        for i in range(6):
            assert save_named_state("moving.json", bytes([i]) * 64) is True, (
                f"a CHANGED state must always be written (save {i} was skipped)"
            )
        newest_state = bytes([5]) * 64
        census_before = _tag_census(kpath)
        state_before = census_before.get(_T_STATE, 0)
        klog = KnowledgeLog()
        kept, dropped = klog.consolidate(list(klog.replay()))
        census_after = _tag_census(kpath)
        # THE RETURN VALUE, and it is a criterion the prune created rather than inherited. `_count`
        # is recomputed as `self._bin.count() - len(foreign)` where `foreign` is the PRUNED list;
        # against the PRE-prune list it would subtract the deleted _T_STATE records from the FACT
        # tally, and the tuple is what `engine/brain.py` binds to `self._lake_kept` /
        # `self._lake_pruned` on the sleep tick. RED, constructed and run 2026-08-05: with
        # `len(foreign)` replaced by the pre-prune length, this call returns (-4, 8) -- a NEGATIVE
        # count of surviving facts, and 8 phantom facts dropped where none were -- while every
        # other assert in this file stays green.
        assert (kept, dropped) == (4, 0), (
            "consolidate() must report the FACT tally only, unaffected by the _T_STATE records it "
            f"pruned: returned kept={kept} dropped={dropped}, expected (4, 0)"
        )

        # ---- L-R7 (specs/022 L6, 2026-09-06): THE LABEL IS LOAD-BEARING HERE --------------------------
        # Before L6 nothing in this function read the retention label, and the sleep tick's cap could
        # silently discard learned knowledge that RETENTION[0x01] calls never-reclaimed. The red: hand
        # consolidate a survivor set with one learned fact removed; it must REFUSE.
        _facts = [r for r in klog.replay()]
        assert len(_facts) >= 2, _facts
        try:
            klog.consolidate(_facts[1:])
        except ValueError as exc:
            assert "LEARNED" in str(exc), exc
        else:  # pragma: no cover - the red not firing is the failure
            raise AssertionError("consolidate accepted a survivor set that drops a LEARNED fact")
        # CONTROL (not the guard): the per-record exemption still works. A re-derivable fact -- one that
        # reconstructs at boot from its named source -- may be dropped in the same call, so the guard
        # scopes compaction rather than stopping it.
        klog.append({"cue": ["wordnet", "probe"], "answer": "rederivable.", "src": "wordnet:noun"})
        _with_bulk = [r for r in klog.replay()]
        _keep = [r for r in _with_bulk if str(r.get("src", "")).split(":", 1)[0] != "wordnet"]
        _k2, _d2 = klog.consolidate(_keep)
        assert _d2 == 1, (
            f"the re-derivable exemption must still permit a drop: kept={_k2} dropped={_d2}"
        )
        # THE 0x07 HALF, and its red. Run BEFORE the print, so a prune that has quietly stopped pruning
        # cannot pass as a green. The first L6 landing scoped its guard to _T_FACT alone and left this
        # class being reclaimed by the same function -- the adversarial panel of 2026-09-06 measured it.
        from snn_data_lake.substrate import lake_engine as _le

        # CONSTRUCTED input, not the shared lake. The consolidate() above already pruned this lake's
        # state history, so reading it back gave both arms nothing to remove and the red could not fail
        # (measured 19 kept vs 19) -- and writing a fixture INTO the lake to fix that broke a later
        # assertion in this same self-check. _prune_state_history is pure over a payload list, so the red
        # supplies its own: two frames under one name, of which the first is superseded, plus a foreign
        # record that must survive either way.
        def _frame(name, seq, blob):
            return (
                _T_STATE
                + bytes([len(name)])
                + name.encode("ascii")
                + int(seq).to_bytes(4, "little")
                + blob
            )

        _probe = [_frame("l6probe", 1, b"v1"), _frame("l6probe", 2, b"v2"), b"\x04foreign"]
        _was07 = _le.RETENTION[0x07]
        _le.RETENTION[0x07] = (_was07[0], False, "no exemption stated here")
        try:
            _blocked = _prune_state_history(_probe)
        finally:
            _le.RETENTION[0x07] = _was07
        _allowed = _prune_state_history(_probe)
        assert len(_blocked) == 3 and len(_allowed) == 2, (
            "removing 0x07's SUPERSEDED exemption must stop the state prune, not merely be ignored: "
            f"blocked={len(_blocked)} allowed={len(_allowed)}, expected 3 and 2"
        )
        assert _allowed[0] == _probe[1] and _probe[2] in _allowed, (
            f"the prune must keep the NEWEST frame per name and every foreign record: {_allowed}"
        )
        assert _le.RETENTION[0x07] == _was07, "the red must restore the registry it borrowed"
        # THE APPEND OBSERVER (specs/022 L4): it must report the tag and the byte count of what this
        # module writes, and a RAISING observer must never break a durable write -- accounting that can
        # lose a record is worse than no accounting.
        # Its OWN store, never the shared self-check lake: writing a probe into that lake breaks a later
        # assertion twelve lines down, which is a mistake this file has now made twice.
        _obs_store = os.path.join(tempfile.mkdtemp(), "observer_probe.bin")
        _seen = []
        _prev = set_append_observer(lambda tag, n: _seen.append((tag, n)))
        try:
            save_named_state("l4observer", b"o" * 48, _obs_store)
            assert any(t == _T_STATE[0] and n > 48 for t, n in _seen), _seen

            def _boom(tag, n):
                raise RuntimeError("accounting exploded")

            set_append_observer(_boom)
            assert save_named_state("l4observer", b"p" * 48, _obs_store) is True, (
                "a raising append observer broke a durable write"
            )
            assert load_named_state("l4observer", _obs_store) == b"p" * 48
        finally:
            set_append_observer(_prev)
        print(
            f"  append observer   : reports (tag, bytes) at every tag-known site ({len(_seen)} seen); "
            "a raising observer cannot break a write (L4/FR-003)"
        )
        print(
            "  immortal guard    : a survivor set missing a LEARNED fact is REFUSED; a re-derivable "
            "fact is still reclaimed in the same call; removing 0x07's SUPERSEDED exemption stops the "
            f"state prune dead ({len(_blocked)} kept vs {len(_allowed)}) (L-R7)"
        )
        # BOTH numbers in ONE assert on purpose. The 13 is the PRECONDITION -- 5 names carrying 13
        # records -- and on its own it has no red this file can reach that the `is True` loop above
        # does not fire on first, so it is not a separate criterion; folded in here it is what stops
        # `-> 5` from passing vacuously against a lake that never accumulated any history to prune.
        assert (state_before, census_after.get(_T_STATE, 0)) == (13, 5), (
            "consolidate must leave exactly ONE _T_STATE record per name (valence, signal_calib, "
            f"other, revert, moving): {state_before} records in, "
            f"{census_after.get(_T_STATE, 0)} out, expected 13 -> 5"
        )
        assert load_named_state("moving.json") == newest_state, (
            "the prune kept a _T_STATE record for the name, but NOT the newest one -- the brain "
            "would boot on state it had already superseded"
        )
        # the other names are untouched, each still reading back its own newest value
        assert load_named_state("other.json") == b"unrelated", "the prune disturbed a second name"
        assert load_named_state("revert.json") == b"A", "the prune undid a revert"
        assert load_named_state("signal_calib.json") == b'{"salience_th": 0.72}', (
            "the prune took an older calibration record"
        )
        assert load_state("valence.json") == {"ENGAGE": 0.83, "RELEASE": -0.11}, (
            "the prune took an older valence record"
        )
        # ...and NOTHING ELSE moved: the prune is scoped to _T_STATE, not to "latest wins per tag".
        for tag, who in (
            (_T_FACT, "_T_FACT"),
            (_T_GROWTH, "_T_GROWTH"),
            (_T_EPISODE, "_T_EPISODE"),
            (_T_SELFBUILD, "_T_SELFBUILD"),
            (_T_SPIKE, "_T_SPIKE (reserved)"),
            (b"\x7f", "an unknown tag"),
        ):
            assert census_after.get(tag, 0) == census_before.get(tag, 0), (
                f"the _T_STATE prune also ate {who}: "
                f"{census_before.get(tag, 0)} -> {census_after.get(tag, 0)}"
            )
        # P4b's honesty guard: "it persists" is a claim about a NEW PROCESS. The in-process read
        # above goes through the `_snn_state_index` cache; this one cannot.
        child_pruned = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            "from snn_data_lake.substrate.home import load_named_state\n"
            f"got = load_named_state('moving.json', {kpath!r})\n"
            f"assert got == {newest_state!r}, ('a NEW process read the wrong record', got[:1])\n"
        )
        rc_p = subprocess.run([sys.executable, "-c", child_pruned]).returncode
        assert rc_p == 0, (
            f"after the prune a SEPARATE process did not read back the newest state (rc={rc_p})"
        )
        print(
            f"state prune       : {state_before} _T_STATE records -> "
            f"{census_after.get(_T_STATE, 0)} (one per name); newest value intact, "
            f"{sum(census_before.values())} -> {sum(census_after.values())} records in the lake"
        )

        # ---- AN UNPARSEABLE _T_STATE RECORD IS KEPT, AND ITS NAME IS NOT GUESSED ----------------
        # `_unpack_state` returning None is the prune's whole safety margin, and until this block
        # existed NOTHING in the tree could fail if that branch were removed: every _T_STATE record
        # the self-check wrote was well formed, so the pre-2026-08-05 parser passed all of the
        # asserts above. The prune's one act is irreversible, so the parser it decides with owes the
        # unreadable case a criterion of its own.
        #
        # The record below claims an 11-byte name and then stops -- no seq, no blob, 13 bytes where
        # the layout needs 17. Its FRAME is complete, and that is the point: `_repair_torn_tail`
        # truncates at the first short frame and `BinLog.replay` stops at one, so a genuinely torn
        # append can never reach the prune. A well-framed record carrying a payload this module
        # cannot read can -- another build writing tag 0x07, a layout change, a restored backup --
        # and nothing downstream will ever remove it for us.
        #
        # RED, constructed and RUN 2026-08-05: replace `_unpack_state` with the parser this file
        # shipped before that date (`n = payload[1]`, no None branch) and the FIRST assert below is
        # the one that fires -- it is the direct criterion on the parser, so it fires before the
        # damage happens and masks the rest. The damage was therefore measured SEPARATELY rather
        # than asserted from, on an isolated lake carrying 6 `moving.json` records + this one: the
        # old parser reads `(b'moving.json', 0, b'')` off the short buffer -- the name of a LIVE
        # state -- calls this record the newest under that name, and the prune takes _T_STATE from
        # 7 to 1 while `load_named_state('moving.json')` returns b''. The shipped parser takes the
        # same lake 7 -> 2 and reads the newest blob back. A bare 1-byte `b'\x07'` payload raises
        # IndexError straight out of the old parser inside the prune, which is the other half of
        # why the branch is a `return None` and not a guess.
        #
        # The other three asserts are not decoration -- each owns a red the ones above it do not,
        # all three CONSTRUCTED AND RUN 2026-08-05. (a) a prune that DROPS what it cannot parse
        # instead of keeping it -> the `any(...)` assert. (b) `_state_index` alone reverted to its
        # own inline parser, the prune left correct -- the exact drift `_unpack_state` was made THE
        # single parser to prevent -- leaves the parse assert and the census green and fires the
        # `load_named_state` assert, which is the only one that sees it. (c) a prune that keeps
        # this record but loses one good survivor beside it -> `6 -> 5` on the count.
        unreadable_state = _T_STATE + bytes([len(b"moving.json")]) + b"moving.json"
        assert _unpack_state(unreadable_state) is None, (
            "a _T_STATE record too short to hold its own seq field must not parse"
        )
        assert BinLog(kpath).append(unreadable_state), "unreadable-state record append failed"
        unreadable_before = _tag_census(kpath).get(_T_STATE, 0)
        klog = KnowledgeLog()
        klog.consolidate(list(klog.replay()))
        unreadable_after = _tag_census(kpath).get(_T_STATE, 0)
        assert any(p == unreadable_state for p in BinLog(kpath).replay()), (
            "the prune DELETED a _T_STATE record it could not parse -- unreadable is not the same "
            "as superseded, and only one of the two directions is recoverable"
        )
        assert load_named_state("moving.json") == newest_state, (
            "an unreadable _T_STATE record was parsed on garbage offsets and its guessed name "
            "shadowed a live state"
        )
        assert (unreadable_before, unreadable_after) == (6, 6), (
            "the unreadable record must be carried through the prune beside the 5 good ones: "
            f"{unreadable_before} -> {unreadable_after}, expected 6 -> 6"
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

        # ---- THE SHARED LOG CACHE IS ONE CRITICAL SECTION (`_LOG_LOCK`) ---------------------------
        # The lake is reachable from more than one thread (the daemon's tick thread and its request
        # threads), and on a free-threaded build those threads run simultaneously rather than
        # interleaved. `_log()` is a check-then-construct over a shared dict, so without the lock
        # each racing caller builds its OWN BinLog over the ONE file -- and each instance carries
        # its own `_seen` set, so `dedup=True` stops deduping and the same payload is durably
        # written more than once.
        #
        # REACHABLE RED, CONSTRUCTED AND RUN 2026-08-05 on the free-threaded `.venv-lake`
        # (3.14.6, `sys._is_gil_enabled()` False), 40 trials x 12 threads on an isolated scratch
        # lake: with the `with _LOG_LOCK:` removed from `_log()`, `_log()` returned up to 12
        # DISTINCT instances (40/40 trials split) and the victim payload landed in the lake up to
        # 5 times (40/40 trials duplicated). With the lock: 1 instance, 1 record, every trial.
        # The assert below is the same probe, shrunk to run inside the self-check budget.
        #
        # It asserts the RECORD COUNT rather than the instance identity on purpose. Identity is the
        # mechanism and the count is the harm, and a future cache keyed differently could legitimately
        # hand out a fresh instance while still deduping correctly -- that must not fail this check.
        #
        # IT RUNS `_RACE_TRIALS` TRIALS, NOT ONE, AND THAT COUNT IS DERIVED RATHER THAN PICKED. A
        # single trial of this probe was measured firing in only 1 of 3 runs against the unlocked
        # code: the scratch lake is small, so a BinLog construction is quick and the window a racing
        # caller has to slip into is narrow. One trial is therefore a ~1/3-sensitive guard for a
        # defect that corrupts the one durable store, which is not good enough to stand as the only
        # criterion. At the measured per-trial rate, 10 trials miss with probability (2/3)**10, about
        # 2 percent. NOTE THE DIRECTION OF THE FLAKINESS, because it decides whether this is
        # tolerable: it can only ever produce a FALSE NEGATIVE (fail to notice a broken lock), never
        # a false positive -- with the lock held, one instance serves every caller and one record is
        # the invariant, not the likely outcome. A green here is weak evidence; a red is conclusive.
        _RACE_TRIALS = 10
        n_threads = 12
        for trial in range(_RACE_TRIALS):
            race_path = os.path.join(work, f"race_lake{trial}.bin")
            _LOG_CACHE.clear()
            BinLog(race_path).append(pack_fact({"cue": ["seed"], "answer": "seed"}))
            _log(race_path)  # prime the cache...
            BinLog(race_path).append(pack_fact({"cue": ["ext"], "answer": "ext"}))  # ...go stale
            assert _LOG_CACHE[(race_path, True)].stale(), (
                "precondition: the cached instance must be stale, or this probe proves nothing"
            )
            victim = _T_FACT + pack_fact({"cue": ["race"], "answer": "written exactly once"})
            gate = threading.Barrier(n_threads)

            def _racer(p=race_path, v=victim, g=gate):
                g.wait()  # release all threads together, so the window is actually contended
                _log(p).append(v)

            racers = [threading.Thread(target=_racer) for _ in range(n_threads)]
            for t in racers:
                t.start()
            for t in racers:
                t.join()
            n_victim = sum(1 for p in BinLog(race_path).replay() if p == victim)
            assert n_victim == 1, (
                f"trial {trial}: {n_threads} threads sharing the one lake through _log() stored "
                f"the same payload {n_victim} times -- dedup is broken because the cache handed "
                "out more than one BinLog over one file, and each has its own _seen set"
            )
        print(
            f"shared log cache  : {_RACE_TRIALS} trials x {n_threads} concurrent _log() callers "
            f"-> 1 BinLog, 1 record each (dedup held under threads)"
        )

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

        # ---- the growth DELTA CHAIN (_T_GROWTH_DELTA, specs/022 T010-ii) ----
        # A fake-but-valid npz: the chain parses zip MEMBERS, never .npy internals, so plain bytes
        # exercise the whole path without numpy (this module's zero-dep contract). Runs LAST so the
        # legacy latest-wins assert above ("grown-substrate v2") sees the lake it always saw.
        import io as _io
        import zipfile as _zf

        def _fake_npz(**kv):
            b = _io.BytesIO()
            with _zf.ZipFile(b, "w", _zf.ZIP_STORED) as z:
                for k, v in kv.items():
                    z.writestr(_zf.ZipInfo(k + ".npy"), v)
            return b.getvalue()

        first = _fake_npz(W=b"WEIGHTS-v1", ticks=b"t=1")
        assert save_growth(first), "chain cutover: the first npz save must write a FULL record"
        second = _fake_npz(W=b"WEIGHTS-v1", ticks=b"t=2")
        assert save_growth(second), "chain: a changed member must write a DELTA record"
        assert save_growth(second) == 0, (
            "chain: a byte-identical substrate must write NOTHING -- this refusal IS the "
            "97.64%-repeated-snapshot pathology being killed at the door"
        )
        got = load_growth()
        assert _growth_members(got) == _growth_members(second), (
            "chain reconstruction must return the newest substrate's exact members"
        )
        chain_census = _tag_census(kpath)
        assert chain_census.get(_T_GROWTH_DELTA, 0) == 2 and chain_census.get(_T_GROWTH, 0) == 2, (
            "the chain must be a NEW tenant beside the legacy tag (2 chain + 2 legacy), got "
            f"{chain_census.get(_T_GROWTH_DELTA, 0)} chain / {chain_census.get(_T_GROWTH, 0)} legacy"
        )
        chain_recs = [p[1:] for p in BinLog(kpath).replay() if p[:1] == _T_GROWTH_DELTA]
        k0, _e0 = _growth_unpack(chain_recs[0])
        k1, e1 = _growth_unpack(chain_recs[1])
        assert k0 == _GC_FULL and k1 == _GC_DELTA, f"expected FULL then DELTA, got {k0} then {k1}"
        assert [n for n, _h, d in e1 if d is not None] == ["ticks.npy"], (
            "the DELTA must carry ONLY the changed member's bytes (ticks.npy), got "
            f"{[n for n, _h, d in e1 if d is not None]}"
        )

        print(
            "self-check OK: knowledge appended in one process is REPLAYED in a separate process (never "
            "forgotten across a restart); append-only; dedup; RAW-BINARY one lake (brain_lake.bin) with "
            "torn-tail recovery; LEARNED STATE lives IN the lake under _T_STATE (cross-process proven, "
            "latest-wins per name, revert-safe; sleep consolidation KEEPS the newest record per name "
            "and prunes the superseded ones, leaving every other tenant untouched) and a pre-P4b "
            "sidecar is imported once and left on disk. The ONE shared home every entrypoint boots into."
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
