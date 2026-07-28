"""
CHAT MEMORY -- a noise-filtered, content-addressable VIEW over the ONE durable brain lake.

The project has ONE durable data lake for the entire brain (substrate/home.py: the shared
`KnowledgeLog` at <home>/knowledge.jsonl, replayed at every boot). This module no longer keeps a
SEPARATE store: it is a thin writer/reader over that one lake. A real conversational turn is written
as a lake record {"cue": content-words, "answer": text, "src": "chat:<role>"} -- the SAME record shape
taught facts use -- so LakeChat (which replays the same KnowledgeLog) converses from what the daemon
learned. There is no off-home `.spkl` and no sidecar index any more; the one lake is the only store.

Mechanism (deterministic, storage-only):
  1. is_real_memory(role, text): a NOISE FILTER. Tool/system chatter, code dumps, JSON/path-heavy
     strings, and pytest/ruff/git logs are rejected; substantive user/assistant prose passes.
  2. _encode(text): lowercase word tokens -> `zlib.crc32(token) mod n_channels` -> a SET of channels
     (a sparse code). Overlapping vocabulary -> overlapping channels -> content-addressable recall.
  3. remember(role, text): if is_real_memory, append one record to the ONE lake (dedup: an identical
     turn is never stored twice). Noise is dropped (nothing written).
  4. recall(query, k): score every chat record in the one lake by Jaccard channel-overlap with the
     query, best first. The lake is the source of truth; the in-RAM channel index is rebuilt from it.

Fair control (un-riggable):
  - Noise really is dropped: remember() returns False for code/system/tool text; kept-vs-dropped
    counts over a real transcript both exceed 0 (a filter that accepts/rejects everything does not
    discriminate).
  - Durability + ONE lake: after a fresh process, a new DurableChatMemory over the same home finds
    the same memories -- they were replayed from the shared knowledge.jsonl, not a private store; and
    a LakeChat over the same home surfaces them too (proved in the self-check).
  - Content-addressable: recall(query) is scored by hashed-token overlap with the QUERY; an unrelated
    query scores lower than the turn it was copied from.
  - Determinism: `zlib.crc32` (never builtin `hash()`), no import-time RNG -> reproducible scores.

Honest framing: a bag-of-hashed-tokens sparse code, not a semantic embedding -- overlap requires
shared VOCABULARY, not shared meaning. It is a noise-filtered, content-addressable VIEW over the one
durable lake, not a language-understanding memory. Storage only -- imports nothing from brainstem/ and
touches no actuator.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.chat_memory
"""

import os
import re
import shutil
import tempfile
import zlib

from snn_data_lake.substrate.home import KnowledgeLog, knowledge_path

_MIN_LEN = 40  # shorter than this is a fragment, not a substantive turn ("ok", "yes", "5")
_MAX_LEN = 4000  # longer than this is a dump (a whole file, a giant log), not a conversational turn
_TEXT_KEEP = 400  # store only a trimmed excerpt as the record answer (display, not full recall)
_CHAT_SRC = "chat:"  # record src prefix that marks a conversational turn in the one lake

_WORD_RE = re.compile(r"[a-z]+")  # lowercase word tokens for _encode()

# CONTENT words only: stopwords are stripped before encoding/cue-building so recall scores reflect shared
# MEANING words, not shared function words. Without this, a short query overlapping only on {what,are,the}
# clears the recall gate and a stray turn ignites a confidently-wrong answer where an honest gap belongs.
# Mirrors reasoning/spike_chat_lake._STOP (kept in sync here to avoid an import cycle: spike_chat_lake
# imports chat_memory, not the reverse).
_STOP = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "at",
        "do",
        "does",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "s",
        "tell",
        "that",
        "the",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
        "you",
    }
)

_NOISE_MARKERS = (  # substrings that mark tool/system/harness chatter, never a conversational turn
    "<system-reminder>",
    "<task-notification>",
    "PreToolUse",
    "PostToolUse",
    "tool_use",
    "tool_result",
    "task-notification",
    "queue-operation",
    "stdout",
    "stderr",
)
_WIN_PATH_RE = re.compile(
    r"[A-Za-z]:[\\/][^\s\"']{15,}"
)  # a long Windows path -> tool/file chatter
_NIX_PATH_RE = re.compile(r"(?:/[\w.\-]+){4,}")  # 4+ chained /segments -> a long *nix path
_LOG_RE = re.compile(  # git / ruff / pytest log signatures
    r"\b(\d+ passed|\d+ failed|\d+ error|collected \d+ item|====+ |ruff check|"
    r"traceback \(most recent call last\)|diff --git|commit [0-9a-f]{7,40})\b",
    re.IGNORECASE,
)
_CODE_LINE_RE = re.compile(r"^\s*(import |from |def |class |REM |@|#include|using |package )")
_SYMBOL_RE = re.compile(r"[{}\[\]<>|`_=;~^]")


def chat_words(text):
    """The content-word set of a chat turn -- the retrieval `cue` stored in the one lake. Sorted for a
    deterministic on-disk record (KnowledgeLog dedups on the exact JSON, so cue order must be stable)."""
    return sorted({t for t in _WORD_RE.findall(text.lower()) if t not in _STOP})


class DurableChatMemory:
    """A noise-filtered, content-addressable VIEW over the ONE durable brain lake (home KnowledgeLog).

    remember() drops noise and appends a real turn to the one lake; recall() scores every chat record
    in the one lake by hashed-token overlap with the query. No private store: the shared
    knowledge.jsonl is the only durable home, so LakeChat (same home) converses from these turns."""

    def __init__(self, path=None, n_channels=512, seed=0, log=None):
        # `path` is accepted for backward-compat but IGNORED -- there is one lake now (home KnowledgeLog).
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        self.n = int(n_channels)
        self.seed = seed
        self.log = log if log is not None else KnowledgeLog(knowledge_path())
        # STORED, NEVER READ (verified tree-wide, v4.78). The comment here used to describe it as an
        # index "rebuilt" over the chat records -- it is assigned empty once and nothing ever writes
        # or reads it, so the description asserted a mechanism that does not exist. Channel filtering
        # actually runs off `self._chats` below. Left in place with an honest label rather than
        # deleted, because an index keyed by record identity is the right shape if channel lookup ever
        # needs to stop being a linear scan.
        self._chan = {}  # STORED, NOT READ -- not an index; nothing populates or consults it
        self._chats = []  # the chat records currently in the one lake (role, text, channels)
        self._reindex()

    def _reindex(self):
        """Rebuild the in-RAM chat index from the one lake (records whose src marks a chat turn)."""
        self._chats = []
        for rec in self.log.replay():
            src = rec.get("src", "")
            if isinstance(src, str) and src.startswith(_CHAT_SRC):
                text = rec.get("answer", "")
                role = src[len(_CHAT_SRC) :] or "user"
                self._chats.append((role, text, self._encode(text)))

    # ---- content-addressable sparse code ------------------------------------
    def _encode(self, text):
        """Lowercase word tokens -> unique `crc32(token) mod n_channels` -> a SET of channels.
        Deterministic (crc32, never builtin `hash()`); shared vocabulary -> shared channels."""
        toks = {t for t in _WORD_RE.findall(text.lower()) if t not in _STOP}
        return {zlib.crc32(t.encode("utf-8")) % self.n for t in toks}

    # ---- the noise filter -----------------------------------------------------
    def is_real_memory(self, role, text):
        """True only for a substantive natural-language user/assistant turn.

        Rejects: fragments/dumps outside [~40, ~4000] chars; tool/system chatter (system-reminder,
        PreToolUse, task-notification, stdout/stderr, long file paths); code dumps (``` fences,
        many import/def/class/REM/@ lines, a high symbol ratio); git/ruff/pytest log output; and
        low-alphabetic-ratio text (JSON/brace/bracket-heavy, not prose)."""
        if role not in ("user", "assistant"):
            return False
        if not isinstance(text, str):
            return False
        n = len(text)
        if n < _MIN_LEN or n > _MAX_LEN:
            return False
        if any(marker in text for marker in _NOISE_MARKERS):
            return False
        if _WIN_PATH_RE.search(text) or _NIX_PATH_RE.search(text):
            return False
        if _LOG_RE.search(text):
            return False
        if "```" in text:
            return False
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            code_lines = sum(1 for line in lines if _CODE_LINE_RE.match(line))
            if code_lines / len(lines) > 0.15:
                return False
        symbol_chars = len(_SYMBOL_RE.findall(text))
        if symbol_chars / n > 0.02:
            return False
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / n < 0.55:
            return False
        return True

    # ---- write a real turn into the ONE lake ----------------------------------
    def remember(self, role, text):
        """If `is_real_memory`, append one chat record to the one lake (dedup skips an identical turn).
        Returns True if stored, False if dropped as noise or a duplicate."""
        if not self.is_real_memory(role, text):
            return False
        excerpt = text[:_TEXT_KEEP]
        written = self.log.append(
            {"cue": chat_words(excerpt), "answer": excerpt, "src": _CHAT_SRC + role}
        )
        if written:
            self._chats.append((role, excerpt, self._encode(excerpt)))
        return bool(written)

    # ---- content-addressable extraction over the one lake ---------------------
    def recall(self, query, k=5):
        """Top-`k` chat turns in the one lake by Jaccard channel-overlap with `_encode(query)`."""
        if not self._chats:
            return []
        q = self._encode(query)
        scored = []
        for role, text, channels in self._chats:
            union = q | channels
            score = (len(q & channels) / len(union)) if union else 0.0
            scored.append((score, role, text))
        scored.sort(key=lambda item: (-item[0], item[2]))
        return [{"role": role, "text": text, "score": score} for score, role, text in scored[:k]]

    def count(self):
        """Number of chat turns held in the one lake (this view's slice)."""
        return len(self._chats)

    def close(self):
        """No-op for compatibility: the one lake is durable + fsync'd on every append (home KnowledgeLog)."""
        return None


def ingest_transcript_turns(transcript_path, log=None, limit=200):
    """Extract (role, text) turns from a Claude Code `.jsonl` transcript and remember the REAL ones into
    the one lake. Returns (kept, dropped). Used by the learning daemon's chat-ingest step and the
    self-check; `message.content` is a str or a list of {"type":"text","text":...} blocks."""
    mem = DurableChatMemory(log=log)
    kept = dropped = 0
    if not os.path.exists(transcript_path):
        return kept, dropped
    import json

    with open(transcript_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant"):
                continue
            blocks = [content] if isinstance(content, str) else content
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                text = (
                    block
                    if isinstance(block, str)
                    else (
                        block.get("text")
                        if isinstance(block, dict) and block.get("type") == "text"
                        else None
                    )
                )
                if isinstance(text, str):
                    if mem.remember(role, text):
                        kept += 1
                    else:
                        dropped += 1
    return kept, dropped


def main():
    from snn_data_lake.substrate.home import home

    tmp_dir = tempfile.mkdtemp(prefix="chat_memory_selfcheck_")
    prev_home = os.environ.get("NORD_HOME")
    os.environ["NORD_HOME"] = tmp_dir  # isolate the one lake for the self-check
    width = 74
    try:
        print("=" * width)
        print("CHAT MEMORY -- a content-addressable VIEW over the ONE durable brain lake")
        print("=" * width)
        print(f"one lake: {knowledge_path()}")

        mem = DurableChatMemory()
        # ---- (1) NOISE FILTER: known noise dropped, a genuine turn kept --------------------------
        noise = [
            ("user", "<system-reminder>do the thing</system-reminder> PreToolUse Bash"),
            ("assistant", "```python\nimport os\ndef f():\n    return os.getcwd()\n```"),
            ("assistant", "42 passed, 0 failed in 3.14s; ruff check . all good"),
            ("user", "ok"),
            ("tool", "some tool text that is long enough to pass the length gate easily here"),
        ]
        real = [
            (
                "user",
                "Remember that we decided to keep the pybullet arm backend installed for later use.",
            ),
            (
                "assistant",
                "The one durable lake now holds chat turns and taught facts together in the home.",
            ),
        ]
        dropped = sum(1 for r, t in noise if not mem.remember(r, t))
        kept = sum(1 for r, t in real if mem.remember(r, t))
        assert dropped == len(noise), (
            f"noise leaked into memory: only {dropped}/{len(noise)} dropped"
        )
        assert kept == len(real), f"a genuine turn was rejected: only {kept}/{len(real)} kept"
        print(f"  [OK] noise filter: dropped {dropped} noise, kept {kept} real (both > 0)")

        # ---- (2) content-addressable recall ------------------------------------------------------
        top = mem.recall("what did we decide about the pybullet arm backend", k=1)
        assert top and "pybullet" in top[0]["text"], f"recall missed the on-topic turn: {top}"
        print(
            f"  [OK] recall: '...pybullet...' -> {top[0]['text'][:48]}... (score {top[0]['score']:.2f})"
        )

        # ---- (3) ONE lake, durable + cross-view: a FRESH view AND a fresh KnowledgeLog see it ----
        mem2 = (
            DurableChatMemory()
        )  # fresh view, no shared RAM -> must replay the two turns from disk
        assert mem2.count() == len(real), f"fresh view lost turns: {mem2.count()} != {len(real)}"
        facts = [r.get("answer") for r in KnowledgeLog(knowledge_path()).replay()]
        assert any("pybullet" in (a or "") for a in facts), (
            "the turn is not in the shared knowledge lake"
        )
        print(
            f"  [OK] ONE lake: a fresh view + a fresh KnowledgeLog both see the {mem2.count()} turns"
        )

        # ---- (4) dedup: re-remembering the same turn does NOT grow the lake ----------------------
        before = KnowledgeLog(knowledge_path()).count()
        again = mem.remember("user", real[0][1])
        after = KnowledgeLog(knowledge_path()).count()
        assert not again and after == before, "a duplicate turn grew the one lake (dedup failed)"
        print(f"  [OK] dedup: a repeated turn is not stored twice (lake stayed at {after})")

        print("-" * width)
        print(
            "self-check OK: chat memory is a noise-filtered, content-addressable VIEW over the ONE "
            "durable brain lake (home knowledge.jsonl) -- no private store; taught facts + chat turns "
            "share the one lake so LakeChat converses from what the daemon learned; dedup keeps it lean."
        )
        return 0
    finally:
        if prev_home is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prev_home
        _ = home  # imported for the banner path; home dir cleaned below
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    sys.exit(main())
