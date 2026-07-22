"""
SNDL CHAT-LEARNING DAEMON -- the brain learns from the developer chat, with NO arm sim / no motor loop.

Reads the project's Claude Code chat transcripts (.jsonl) and feeds each user/assistant turn into
Brain.respond(). respond() teaches the spiking INTENT ROUTER, updates THEORY-OF-MIND from the dialogue
(_tom_update), and stores an EPISODIC memory of the exchange. Only a structured arm/lake COMMAND takes
the gated motor path; free conversational text is display-only and actuates NOTHING -- so this loop
never commands the arm. The arm SIMULATOR object still exists inside the brain (it is the reward/value
substrate and cannot be deleted), but here it stays dormant: no _wake_tick, no motor, no _ext_drive.

PERSISTENCE (honest, best-effort): the brain reloads what it saved on a prior run at startup (the
acquired language organ + the nightly-grown reasoning skill), and this daemon calls Brain.close() on exit
-- the brain's own "final durable save" -- so a graceful stop/restart carries learning forward. A hard
crash (the .bat auto-restarts) loses that run's un-saved in-memory deltas; fine-grained intent-router
state persists only to the extent the brain's durable homes capture it. This is the brain's normal
learn-during-run / save-on-close / reload-on-start lifecycle, not a continuous checkpoint.

HONEST on "learning": the brain's language faculty is weak (no grammatical comprehension). Feeding it the
chat ingests the TEXT into its intent-router / ToM / episodic-memory learners and REMEMBERS the exchanges
statistically -- it does NOT understand the conversation, and it learns only from these transcripts, not
a general web/LLM.

  ./.venv-lake/Scripts/python.exe chat_learn_daemon.py
Env: SNDL_CHAT_TRANSCRIPT (glob; default = all project sessions), SNDL_CHAT_CHUNK (lines/file/poll),
SNDL_CHAT_POLL (seconds). ASCII-only output.
"""

# ruff: noqa: E402, E501  -- root-level runner: intentional pre-import path setup + long descriptive lines

import atexit
import glob
import json
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")

from snn_data_lake.cortex.brain import Brain

TRANSCRIPTS = os.environ.get(
    "SNDL_CHAT_TRANSCRIPT", r"C:\Users\vinhl\.claude\projects\D--AI-project\*.jsonl"
)
CHUNK_LINES = int(os.environ.get("SNDL_CHAT_CHUNK", "600"))  # lines per file per poll (bounded backfill)
POLL = float(os.environ.get("SNDL_CHAT_POLL", "15"))  # seconds between polls
MAX_CHARS = int(os.environ.get("SNDL_CHAT_MAXCHARS", "2000"))  # cap per turn fed to respond()


def _turns(path, start):
    """New (role, text) turns from a Claude Code .jsonl beyond line offset `start`, at most CHUNK_LINES
    lines this call. Returns (turns, new_offset). Text is under message.content (a string or a list of
    {type:'text', text:...} blocks); only user/assistant roles with non-empty text are kept."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return [], start
    window = lines[start : start + CHUNK_LINES]
    out = []
    for line in window:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        m = rec.get("message") or {}
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = ""
        text = text.strip()
        if text and role in ("user", "assistant"):
            out.append((role, text))
    return out, min(start + CHUNK_LINES, len(lines))


def main():
    b = Brain(seed=0)  # persist ON; arm sim built but NEVER commanded (free text + no wake tick)
    atexit.register(lambda: _safe_close(b))
    offsets, total = {}, 0
    print(
        f"[chat-learn] started: learning from project chats via respond(); NO arm motor loop; "
        f"save-on-close persistence. glob={TRANSCRIPTS}",
        flush=True,
    )
    while True:
        progressed = False
        for path in sorted(glob.glob(TRANSCRIPTS), key=lambda p: os.path.getmtime(p)):  # oldest-first
            turns, new_off = _turns(path, offsets.get(path, 0))
            if new_off != offsets.get(path, 0):
                progressed = True
            offsets[path] = new_off
            for _role, text in turns:
                try:
                    b.respond(text[:MAX_CHARS])  # teaches intent router + ToM + stores an episode; no arm
                    total += 1
                except Exception as e:  # a bad turn must never kill the daemon
                    print(f"[chat-learn] respond error (continuing): {type(e).__name__}: {e}", flush=True)
            if turns:
                print(
                    f"[chat-learn] +{len(turns)} turns from {os.path.basename(path)} (total {total})",
                    flush=True,
                )
        if not progressed:
            time.sleep(POLL)  # fully caught up -> idle poll; new chat content is picked up next sweep


def _safe_close(b):
    try:
        b.close()  # the brain's final durable save (language organ + autobiographical trace)
        print("[chat-learn] closed: durable state saved.", flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
