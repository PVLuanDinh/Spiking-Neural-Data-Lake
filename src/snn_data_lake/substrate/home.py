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
  knowledge_path()  -> <home>/knowledge.jsonl  durable append-only symbolic knowledge (cue->answer, ...)
  state_path(name)  -> <home>/<name>           a durable JSON state blob (e.g. a learned value fn)
  KnowledgeLog      -> append-only JSONL knowledge store: .append(rec) durably, .replay() to reload
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


def home():
    """The durable persistence dir. `NORD_HOME` overrides; default is ~/.snn_data_lake. Created on demand."""
    d = os.environ.get("NORD_HOME") or os.path.join(os.path.expanduser("~"), ".snn_data_lake")
    os.makedirs(d, exist_ok=True)
    return d


def lake_path(name="lake.spkl"):
    """Path to the durable spike lake under home() (the StreamingHub / LakeMemory `.spkl` store)."""
    return os.path.join(home(), name)


def knowledge_path(name="knowledge.jsonl"):
    """Path to the durable append-only symbolic-knowledge log under home()."""
    return os.path.join(home(), name)


def state_path(name):
    """Path to a named durable JSON state blob under home() (e.g. 'valence.json')."""
    return os.path.join(home(), name)


class KnowledgeLog:
    """A durable, append-only JSONL knowledge store -- the project's 'never forgets' symbolic memory,
    shared by the developmental loop (what it learns autonomously) and Spike Chat (what it converses from).

    Every learned record (a JSON dict) is appended to disk immediately and fsync-flushed, so a crash
    or an exit cannot lose it. replay() streams every record back IN ORDER at the next boot, so a
    fresh PROCESS reconstructs everything learned in every prior run -- nothing is held only in RAM.

    LEAN by default: dedup=True skips a record byte-identical to one already stored, so the loop can
    re-see the same concept forever without bloating the SSD (the same fact is never written twice).
    Append-only otherwise -- knowledge is added, never silently dropped.

    SHARED RECORD SCHEMA (what the loop writes and the chat + reasoner read):
      {"cue": ["content","words"],        # required: retrieval keys (content words)
       "answer": "the answer text.",      # required: the reply/definition text
       "triple": ["subject","rel","obj"], # optional: a structured fact (enables contradiction reasoning)
       "src": "wikipedia:calculus"}       # optional: provenance of an autonomously-learned fact
    """

    def __init__(self, path=None, dedup=True):
        self.path = path or knowledge_path()
        self.dedup = bool(dedup)
        recs = list(self.replay())  # the one boot-time full pass
        self._seen = {self._key(r) for r in recs} if self.dedup else None
        self._count = len(recs)  # maintained -> count() is O(1), not a re-replay

    @staticmethod
    def _key(record):
        return json.dumps(
            record, ensure_ascii=True, sort_keys=True
        )  # canonical -> dedup key + on-disk line

    def _ends_with_newline(self):
        with open(self.path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b"\n"

    def append(self, record):
        """Durably append one JSON record (skipped if dedup and byte-identical to a stored one).
        Flush + fsync so it survives a hard crash / power loss. Returns True if written, False if a dup."""
        if not isinstance(record, dict):
            raise TypeError("KnowledgeLog.append: record must be a JSON dict")
        line = self._key(record)  # ASCII, canonical
        if self.dedup and line in self._seen:
            return False  # already known -> keep the lake lean
        with open(self.path, "a", encoding="ascii") as f:
            # a prior crash may have left a partial last line with no trailing newline; a plain append
            # would FUSE the new record onto that garbage (losing both). guarantee a clean boundary.
            if os.path.getsize(self.path) and not self._ends_with_newline():
                f.write("\n")
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())  # durable past process death
        if self.dedup:
            self._seen.add(line)
        self._count += 1
        return True

    def extend(self, records):
        """Durably append many records (one fsync at the end -- for bulk import). Returns #written."""
        written = 0
        with open(self.path, "a", encoding="ascii") as f:
            if (
                os.path.getsize(self.path) and not self._ends_with_newline()
            ):  # same torn-tail guard as append()
                f.write("\n")  # -> a crash-torn last line can't fuse the batch
            for record in records:
                if not isinstance(record, dict):
                    raise TypeError("KnowledgeLog.extend: every record must be a JSON dict")
                line = self._key(record)
                if self.dedup and line in self._seen:
                    continue
                f.write(line + "\n")
                if self.dedup:
                    self._seen.add(line)
                written += 1
            if written:
                f.flush()
                os.fsync(f.fileno())
        self._count += written
        return written

    def replay(self):
        """Yield every stored record in append order. Missing file -> nothing learned yet (empty)."""
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn/partial line (crash mid-append) -> skip, never crash the boot

    def count(self):
        """Number of records stored (O(1) maintained counter -- callers poll this per round/request)."""
        return self._count


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
        # CRASH-RECOVERY: a torn last line (crash mid-append, no trailing newline) must NOT crash the
        # next boot, and a new append must land cleanly (not fuse onto the garbage) and be replayed.
        with open(kpath, "a", encoding="ascii") as f:
            f.write('{"cue": ["torn"], "ans')  # partial JSON, no newline -> a crashed write
        survived = KnowledgeLog().count()  # construction must NOT raise on the torn line
        KnowledgeLog().append({"cue": ["moon"], "answer": "luna"})
        recovered = [r["answer"] for r in KnowledgeLog().replay()]
        assert survived == 3, (
            "a torn last line must be skipped, not crash every future KnowledgeLog()"
        )
        assert "luna" in recovered, (
            "an append after a crash-torn line was lost (fused onto the garbage)"
        )
        # ASCII on disk (cp1252-safe): the whole log decodes as ASCII
        with open(kpath, encoding="ascii") as f:
            f.read()

        print(
            "self-check OK: knowledge appended in one process is REPLAYED in a separate process "
            "(never forgotten across a restart); append-only; durable JSON state round-trips; "
            "ASCII-on-disk (cp1252-safe). This is the shared home every entrypoint boots into."
        )
    finally:
        os.environ.pop("NORD_HOME", None)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
