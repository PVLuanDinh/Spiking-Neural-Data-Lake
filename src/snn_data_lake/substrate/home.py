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
  state_path(name)  -> <home>/<name>           a durable JSON state blob (e.g. a learned value fn)
  KnowledgeLog      -> the symbolic VIEW of the one lake: .append(rec) durably (binary), .replay() to reload
  save_state/load_state(name, obj) -> durable JSON state blob round-trip (atomic write)
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

from snn_data_lake.substrate.binlog import BinLog, pack_fact, unpack_fact

# The ONE brain lake is RAW BINARY (symbolic facts + chat turns ride as compact pack_fact byte-records).
# Each record carries a 1-byte TYPE tag so the symbolic view (KnowledgeLog) reads only its own records
# from the one file. FACT = pack_fact({cue,answer,triple?,src?}). Weights/calibration stay separate
# STATE (save_state), never in the memory lake.
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
    """Path to a named durable JSON state blob under home() (e.g. 'valence.json')."""
    return os.path.join(home(), name)


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
        self._bin = BinLog(path or knowledge_path(), dedup=dedup)
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


def save_state(name, obj):
    """Durably write a JSON-serialisable state blob to <home>/<name>, atomically (temp + os.replace)."""
    path = state_path(name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(obj, f, ensure_ascii=True, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic swap: no half-written state
    return path


def load_state(name, default=None):
    """Load a JSON state blob from <home>/<name>; return `default` if it was never written."""
    path = state_path(name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="ascii") as f:
        return json.load(f)


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
    return BinLog(path or knowledge_path()).append(_T_GROWTH + bytes(blob))


def load_growth(path=None):
    """The most recent grown-substrate blob, or None if the lake holds none.

    None is the honest answer for a lake written before this record type existed, and the caller
    (`Brain._make_neuro`) treats it as "start ungrown" -- so an old lake behaves exactly as it did
    before, which is the backward-compatibility requirement."""
    latest = None
    for payload in BinLog(path or knowledge_path()).replay():
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

        print(
            "self-check OK: knowledge appended in one process is REPLAYED in a separate process (never "
            "forgotten across a restart); append-only; dedup; RAW-BINARY one lake (brain_lake.bin) with "
            "torn-tail recovery; durable JSON state round-trips. The ONE shared home every entrypoint boots into."
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
