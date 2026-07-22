"""
Durable, content-addressable CHAT MEMORY on the spike lake.

Today an assistant's conversational memory either lives nowhere (lost at process exit) or in an
opaque vector-DB blob nobody can point at. This module stores REAL conversational turns -- not
tool noise, not code dumps, not system chatter -- as sparse spike patterns in the SAME durable
`.spkl` lake used everywhere else in this project (`lake_working_memory.LakeWorkingMemory`), and
extracts them back by CONTENT (Jaccard overlap of hashed-token channels), not by tick index.

Mechanism (deterministic, storage-only):
  1. is_real_memory(role, text): a NOISE FILTER. Tool/system chatter, code dumps, JSON/path-heavy
     strings, and pytest/ruff/git logs are rejected; substantive user/assistant prose passes.
  2. _encode(text): tokenize to lowercase word tokens, hash each unique token with `zlib.crc32`
     mod `n_channels` -> a SET of channels (a sparse spike pattern). Overlapping vocabulary between
     two texts means overlapping channels -- this is what makes retrieval content-addressable.
  3. remember(role, text): if is_real_memory, `lake.write(tick, _encode(text))` appends the pattern
     to the durable lake; a sidecar JSON index (`<path>.idx.json`) records {role, text (trimmed),
     n_tokens} per tick so retrieval can show what was actually said. Noise is dropped (no tick
     consumed, nothing written).
  4. recall(query, k): `_encode(query)` -> read the channel pattern for EVERY stored tick back out
     of the durable lake (`lake.recall(window=n_ticks)`) -> score each by Jaccard overlap with the
     query's channels -> return the top-k sidecar entries (role, text, score), best first. The
     lake -- not RAM, not the sidecar -- is the source of truth for what was actually written; the
     sidecar only carries the human-readable text/role for display.
  5. On init, if both the `.spkl` and its sidecar index already exist, the lake is `reopen()`-ed
     (replaying the durable log) and the sidecar is loaded, so remembered turns survive a restart.

Fair control (un-riggable):
  - Noise really is dropped: remember() returns False and consumes no tick for code/system/tool
    text; kept-vs-dropped counts over a real transcript show BOTH must be > 0 (a filter that
    accepts or rejects everything is not discriminating).
  - Durability: after closing the process and constructing a fresh `DurableChatMemory` over the
    SAME path, recall() finds the same memories -- reopen() replayed them from the `.spkl`, not
    from any surviving Python object.
  - Content-addressable, not index-addressable: recall(query) is scored by hashed-token overlap
    with the QUERY text, and returns the best-matching stored turn regardless of when it was
    written; an unrelated query scores lower than the turn it was copied from.
  - Determinism: `zlib.crc32` (never builtin `hash()`) and no import-time RNG state -> identical
    text always encodes to the identical channel set, so recall scores are reproducible.

Honest framing: a bag-of-hashed-tokens sparse code, not a semantic embedding -- overlap requires
shared VOCABULARY, not shared meaning ("the car is red" and "the automobile is crimson" would not
match). It is a noise-filtered, durable, content-addressable log, not a language-understanding
memory. Storage only -- imports nothing from brainstem/ and touches no actuator.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.chat_memory
"""

import json
import os
import re
import shutil
import tempfile
import zlib

from snn_data_lake.substrate.lake_working_memory import LakeWorkingMemory

_MIN_LEN = 40  # shorter than this is a fragment, not a substantive turn ("ok", "yes", "5")
_MAX_LEN = 4000  # longer than this is a dump (a whole file, a giant log), not a conversational turn
_TEXT_KEEP = 400  # sidecar stores only a trimmed excerpt (index is for display, not full recall)
_FLUSH_EVERY = 10  # periodic sidecar flush cadence (also always flushed on close())

_WORD_RE = re.compile(r"[a-z]+")  # lowercase word tokens for _encode()

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


class DurableChatMemory:
    """REAL conversational memory, durably held in the spike lake and extracted by content.

    __init__ opens (or resumes) a durable `LakeWorkingMemory` at `path` plus a sidecar JSON index
    at `<path>.idx.json`. remember() drops noise and writes real turns as sparse spike patterns;
    recall() scores stored turns against a query by channel-overlap (Jaccard) read back from the
    durable lake, returning the best content match first."""

    def __init__(self, path="data/chat_memory.spkl", n_channels=512, seed=0):
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        self.path = path
        self.idx_path = path + ".idx.json"
        self.n = int(n_channels)
        self.seed = seed
        self._tick = 0
        self._index = {}  # tick(int) -> {"role": str, "text": str, "n_tokens": int}
        self._dirty = 0  # remembers since the last sidecar flush

        resume = os.path.exists(path) and os.path.exists(self.idx_path)
        if resume:
            self._load_index()
            self._tick = (max(self._index) + 1) if self._index else 0

        self.lake = LakeWorkingMemory(self.n, path=path, seed=seed)
        if resume:
            self.lake.reopen()  # replay the durable log -> memory survives the restart

    # ---- content-addressable sparse code ------------------------------------
    def _encode(self, text):
        """Lowercase word tokens -> unique `crc32(token) mod n_channels` -> a SET of channels.

        Deterministic (crc32, never builtin `hash()`); texts sharing vocabulary share channels,
        which is what makes overlap-scoring in recall() content-addressable."""
        tokens = {t for t in _WORD_RE.findall(text.lower())}
        return {zlib.crc32(t.encode("utf-8")) % self.n for t in tokens}

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

    # ---- write a real turn into the durable lake ------------------------------
    def remember(self, role, text):
        """If `is_real_memory`, encode + write to the durable lake and append the sidecar entry.

        Returns True if stored, False if dropped as noise (no tick consumed, nothing written)."""
        if not self.is_real_memory(role, text):
            return False
        channels = self._encode(text)
        self.lake.write(self._tick, channels)
        self._index[self._tick] = {
            "role": role,
            "text": text[:_TEXT_KEEP],
            "n_tokens": len(_WORD_RE.findall(text.lower())),
        }
        self._tick += 1
        self._dirty += 1
        if self._dirty >= _FLUSH_EVERY:
            self._flush_index()
        return True

    # ---- content-addressable extraction ---------------------------------------
    def recall(self, query, k=5):
        """Top-`k` stored turns by Jaccard channel-overlap with `_encode(query)`, best first.

        Channel sets are read back from the DURABLE lake (not the sidecar, not RAM) so the score
        reflects what was actually written; the sidecar only supplies role/text for display."""
        if not self._index:
            return []
        query_channels = self._encode(query)
        by_tick = dict(self.lake.recall(self._tick))  # {tick: frozenset(channels)}, all ticks
        scored = []
        for tick, entry in self._index.items():
            channels = by_tick.get(tick, frozenset())
            union = query_channels | channels
            score = (len(query_channels & channels) / len(union)) if union else 0.0
            scored.append((score, tick, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"role": entry["role"], "text": entry["text"], "score": score, "tick": tick}
            for score, tick, entry in scored[:k]
        ]

    # ---- sidecar index persistence ---------------------------------------------
    def _load_index(self):
        with open(self.idx_path, encoding="utf-8") as f:
            raw = json.load(f)
        self._index = {int(tick): entry for tick, entry in raw.items()}

    def _flush_index(self):
        raw = {str(tick): entry for tick, entry in self._index.items()}
        tmp_path = self.idx_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(tmp_path, self.idx_path)
        self._dirty = 0

    def close(self):
        """Flush the sidecar index to disk and close the lake handle (durable -- never removed)."""
        self._flush_index()
        self.lake.close(remove=False)


def _extract_turns(transcript_path, limit=200):
    """Pull (role, text) turns out of a Claude Code `.jsonl` transcript for the self-check.

    `message.content` is either a plain string or a list of `{"type": "text", "text": ...}`
    blocks; only text is extracted (tool_use/tool_result blocks are skipped)."""
    turns = []
    if not os.path.exists(transcript_path):
        return (
            turns  # transcript absent (other machine / rotated) -> caller uses a synthetic fallback
        )
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
            if isinstance(content, str):
                turns.append((role, content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str):
                            turns.append((role, text))
    return turns


def main():
    transcript_path = os.environ.get(
        "SNDL_CHAT_TRANSCRIPT_SELFCHECK",
        r"C:\Users\vinhl\.claude\projects\D--AI-project\9e11de63-a8bb-4037-9746-e186c88cce02.jsonl",
    )
    tmp_dir = tempfile.mkdtemp(prefix="chat_memory_selfcheck_")
    path = os.path.join(tmp_dir, "chat_memory.spkl")
    N = 512

    try:
        # ---- (1) NOISE FILTER: known noise must be dropped, a genuine turn must pass -------
        cm = DurableChatMemory(path=path, n_channels=N, seed=0)
        noise_samples = [
            (
                "assistant",
                "```python\ndef foo(x):\n    return x + 1\n\nclass Bar:\n    pass\n```"
                " here is some code for you to review closely please",
            ),
            (
                "user",
                "<system-reminder>\nThis context may or may not be relevant to the current "
                "task at hand and you should not feel obligated to respond to it.\n</system-reminder>",
            ),
            (
                "assistant",
                "PreToolUse:Read hook additional context: MANDATORY graphify-out/graph.json "
                "exists you must run graphify before reading any raw source files in this repo",
            ),
            ("user", "ok"),
        ]
        genuine_sample = (
            "user",
            "Can you explain why the working memory recall window only keeps the last three "
            "ticks instead of the whole history? I want to understand the recency tradeoff.",
        )
        noise_results = [cm.remember(role, text) for role, text in noise_samples]
        genuine_result = cm.remember(*genuine_sample)
        assert all(r is False for r in noise_results), (
            f"noise samples must all be dropped, got {noise_results}"
        )
        assert genuine_result is True, "a genuine conversational sentence must be remembered"

        # ---- pull real turns from an actual transcript and run the filter honestly ---------
        real_turns = _extract_turns(transcript_path, limit=400)
        if (
            len(real_turns) < 4
        ):  # transcript absent (portable / CI / rotated) -> synthetic prose+noise batch
            real_turns = [
                (
                    "user",
                    "I want to understand how the durable spike lake keeps working-memory contents queryable across a reopen instead of losing them when the process exits.",
                ),
                (
                    "assistant",
                    "The lake appends every tick's channels to the .spkl file, and reopen replays them so recall can reconstruct the stored contents after a restart.",
                ),
                ("user", "```python\nimport os\ndef g():\n    return os.getcwd()\n```"),
                (
                    "assistant",
                    "PreToolUse hook additional context: run graphify before grepping raw files in this repo",
                ),
                ("user", "ok"),
            ]
        kept = dropped = 0
        for role, text in real_turns:
            if cm.remember(role, text):
                kept += 1
            else:
                dropped += 1
        assert kept > 0, "the filter dropped every real turn (too strict)"
        assert dropped > 0, "the filter kept every real turn (too permissive / not discriminating)"

        # ---- (3) EXTRACTABLE + REAL not noise: recall a phrase from a remembered real turn --
        # pick a real remembered turn (not the synthetic genuine_sample / noise) to query against
        sample_entry = None
        for entry in cm._index.values():
            if entry["text"] == genuine_sample[1][:_TEXT_KEEP]:
                continue
            if len(entry["text"]) >= 80:
                sample_entry = entry
                break
        assert sample_entry is not None, "need at least one real remembered turn to query"
        query_phrase = sample_entry["text"][:60]
        top = cm.recall(query_phrase, k=5)
        assert top, "recall returned nothing for a phrase copied from a remembered turn"
        top1 = top[0]
        assert top1["text"] == sample_entry["text"], (
            f"top recall hit did not match the source turn: {top1['text'][:60]!r} != "
            f"{sample_entry['text'][:60]!r}"
        )
        unrelated_top = cm.recall("zebra hovercraft plasma quantum origami", k=5)
        unrelated_score = unrelated_top[0]["score"] if unrelated_top else 0.0
        assert top1["score"] > unrelated_score, (
            "a copied-phrase query must score higher than an unrelated query"
        )

        cm.close()

        # ---- (2) DURABLE: a fresh instance over the SAME path sees the same memories --------
        assert os.path.exists(path), ".spkl was not created -- memory is not durable"
        cm2 = DurableChatMemory(path=path, n_channels=N, seed=0)
        top_after_reopen = cm2.recall(query_phrase, k=5)
        durable_ok = bool(top_after_reopen) and top_after_reopen[0]["text"] == sample_entry["text"]

        # ---- (4) Deterministic: same input -> same channels/scores across two runs ----------
        enc_a = cm2._encode(query_phrase)
        enc_b = cm2._encode(query_phrase)
        recall_a = cm2.recall(query_phrase, k=5)
        recall_b = cm2.recall(query_phrase, k=5)
        deterministic = (enc_a == enc_b) and (
            [(r["tick"], r["score"]) for r in recall_a]
            == [(r["tick"], r["score"]) for r in recall_b]
        )
        cm2.close()

        assert durable_ok, "reopened instance did not recall the same memory (not durable)"
        assert deterministic, "encode/recall was not deterministic across repeated calls"

        print(
            f"chat_memory OK: kept={kept} dropped={dropped} durable={durable_ok} "
            f"recall_top1_match={top1['text'] == sample_entry['text']}"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
