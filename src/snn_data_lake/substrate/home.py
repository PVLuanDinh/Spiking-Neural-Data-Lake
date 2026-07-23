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
import struct
import sys

from snn_data_lake.substrate.binlog import BinLog, pack_array, pack_fact, unpack_array, unpack_fact

# The ONE brain lake is RAW BINARY (spiking exemplars + skill vectors are native binary; symbolic facts +
# chat turns ride as compact pack_fact byte-records in the SAME file). Each record carries a 1-byte TYPE
# tag so the symbolic view (KnowledgeLog) and the spike consumers (SpikeLake) read only their own records
# from the one file. FACT = pack_fact({cue,answer,triple?,src?}); SPIKE = a feature vector + payload for a
# durable AER exemplar. Weights/calibration stay separate STATE (save_state), never in the memory lake.
_T_FACT = b"\x01"
_T_SPIKE = b"\x02"
_U32 = struct.Struct("<I")


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
    the whole brain's learned MEMORY (symbolic facts + chat turns as pack_fact records; spiking exemplars
    + skill vectors fold in as binary records). The former knowledge.jsonl is retired into this file."""
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


class SpikeLake:
    """The SPIKING view of the ONE durable brain lake -- durable AER exemplars (a feature vector + a small
    payload: replay id / class / task) as _T_SPIKE binary records in the SAME brain_lake.bin, alongside the
    symbolic FACT records. append(vec, payload); replay() -> (vec, payload). The spike PATTERN is a
    deterministic function of the vector (cortex.spike_preprocessing.encode_latency), so only the vector is
    stored (float16 via binlog.pack_array). The proximity retrieval index is REBUILT at boot from these
    records (the LakeChat scratch-index pattern), which makes AER memory genuinely DURABLE + boot-
    reconstructable -- today the payload lives only in RAM and the index resets on boot, so the AER lake is
    durable bytes but not durable memory; this fixes that. Record = _T_SPIKE [u32 plen][payload][pack_array].
    KnowledgeLog.replay skips these (tag != _T_FACT) and KnowledgeLog.consolidate preserves them untouched,
    so the symbolic sleep prune never drops a spike exemplar."""

    def __init__(self, path=None, dedup=True):
        self._bin = BinLog(path or knowledge_path(), dedup=dedup)

    def append(self, vec, payload=""):
        """Durably append one AER exemplar (feature vector + payload). Deduped + fsync'd by BinLog."""
        pb = str(payload).encode("utf-8")
        rec = _T_SPIKE + _U32.pack(len(pb)) + pb + pack_array(list(vec))
        return self._bin.append(rec)

    def replay(self):
        """Yield (feature_vector, payload_str) for every _T_SPIKE record, in append order; FACT/other
        records are skipped, so the spiking view never chokes on the mixed one lake."""
        for p in self._bin.replay():
            if p[:1] != _T_SPIKE:
                continue
            try:
                (plen,) = _U32.unpack_from(p, 1)
                payload = p[5 : 5 + plen].decode("utf-8")
                vec = unpack_array(p[5 + plen :])
                yield vec, payload
            except Exception:  # noqa: BLE001 - a malformed spike record must never crash the boot rebuild
                continue

    def count(self):
        """Number of durable AER exemplars stored (a scan; spike-count is not on a hot polling path)."""
        return sum(1 for p in self._bin.replay() if p[:1] == _T_SPIKE)


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

        # ---- SPIKING view shares the ONE lake: FACT + SPIKE records coexist, isolated, survive the prune ----
        slake = SpikeLake()
        slake.append([0.1, 0.9, 0.5], "exemplar-A")
        slake.append([0.2, 0.8, 0.4], 7)
        spikes = list(
            SpikeLake().replay()
        )  # a FRESH reader off disk -> durable AER memory (vec + payload)
        assert len(spikes) == 2 and spikes[0][1] == "exemplar-A" and spikes[1][1] == "7", spikes
        assert all(abs(a - b) < 1e-2 for a, b in zip(spikes[0][0], [0.1, 0.9, 0.5])), (
            "spike vector lost"
        )
        facts_now = (
            KnowledgeLog().count()
        )  # the FACT view reads ONLY facts (spike records are skipped)
        assert facts_now == 4, f"spike records leaked into the FACT count: {facts_now}"
        klog = (
            KnowledgeLog()
        )  # sleep consolidation of the FACT side must PRESERVE the spike exemplars
        klog.consolidate(list(klog.replay()))
        assert SpikeLake().count() == 2 and KnowledgeLog().count() == 4, (
            "sleep consolidation of facts dropped durable spike exemplars (or lost facts)"
        )

        print(
            "self-check OK: knowledge appended in one process is REPLAYED in a separate process (never "
            "forgotten across a restart); append-only; dedup; RAW-BINARY one lake (brain_lake.bin) with "
            "torn-tail recovery; durable JSON state round-trips. The ONE shared home every entrypoint boots into."
        )
    finally:
        os.environ.pop("NORD_HOME", None)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
