"""
SNDL LEARNING DAEMON -- FULL-brain always-on learning + chat ingest to the durable lake + auto-update.

Runs the whole circadian brain forever and learns from everything it can:
  (1) FULL tick(): wake perception/cognition + local plasticity (three-factor / e-prop / feedback
      alignment), sleep NEUROGENESIS (growth happens only in sleep) + replay consolidation + the
      autonomous self-build (every 8 nights, only if NORD_SELFMOD=1). Persistence ON -> learning
      accumulates in data/ and a restart RESUMES (does not start over). The arm is the internal
      SIMULATOR (no real hardware, reflex/E-STOP always in force) -- a dormant reward/action substrate,
      never externally actuated.
  (2) CHAT INGEST: new user/assistant turns from the project chat transcripts are fed to respond()
      (teaches the intent router + theory-of-mind) AND remembered into a DURABLE, content-addressable
      chat memory in the ONE durable brain lake (substrate/chat_memory.DurableChatMemory, a VIEW over the
      shared home knowledge.jsonl) -- but ONLY the REAL turns: the noise filter drops tool output,
      system-reminders, code dumps, and logs, so it saves real memory, not noise. Because it is the SAME
      lake LakeChat replays, LakeChat converses from what the daemon learned. Queryable via .recall(query).
  (3) AUTO-UPDATE: polls `git rev-parse HEAD`; when a NEW COMMIT lands (a new brain build), it persists
      (brain + chat lake) and exits, so the launcher (learn_daemon.bat's restart loop) relaunches with
      the new code. Only COMMITTED builds trigger a reload -- never a mid-edit working tree. Learning
      survives the restart via persist/reload.

HONEST: a simulation you launch, not a live organism. "Learning from the chat" ingests the TEXT into its
intent/ToM/episodic learners + a vocabulary-overlap durable memory -- it remembers, it does not
comprehend. It learns from these transcripts + its own sensory/arm-sim loop, never from a live web/LLM.

  ./.venv-lake/Scripts/python.exe learn_daemon.py            (learn only)
  NORD_SELFMOD=1 ./.venv-lake/Scripts/python.exe learn_daemon.py   (+ autonomous self-build)
Env: SNDL_CHAT_TRANSCRIPT (glob), NORD_HOME (the one-lake home), SNDL_TICKS (ticks/cycle), SNDL_POLL. ASCII only.
"""

# ruff: noqa: E402, E501  -- root-level runner: intentional pre-import path setup + long descriptive lines

import atexit
import glob
import json
import os
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")

from snn_data_lake.cortex.brain import Brain
from snn_data_lake.substrate.chat_memory import DurableChatMemory

TRANSCRIPTS = os.environ.get(
    "SNDL_CHAT_TRANSCRIPT", r"C:\Users\vinhl\.claude\projects\D--AI-project\*.jsonl"
)
TICKS_PER_CYCLE = int(os.environ.get("SNDL_TICKS", "20"))
CHUNK_LINES = int(os.environ.get("SNDL_CHAT_CHUNK", "400"))
POLL = float(os.environ.get("SNDL_POLL", "5"))
LOG_EVERY = int(os.environ.get("SNDL_LOG_EVERY", "50"))

_shutdown_done = False


def _head():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:
        return ""


def _new_turns(path, start):
    """New (role, text) turns from a Claude Code .jsonl beyond line `start`, at most CHUNK_LINES lines."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return [], start
    out = []
    for line in lines[start : start + CHUNK_LINES]:
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
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = ""
        text = text.strip()
        if text and role in ("user", "assistant"):
            out.append((role, text))
    return out, min(start + CHUNK_LINES, len(lines))


def _shutdown(b, mem):
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    try:
        mem.close()
    except Exception:
        pass
    try:
        b.close()
    except Exception:
        pass
    print("[learn-daemon] persisted (brain durable save + chat lake flushed).", flush=True)


def main():
    from snn_data_lake.substrate.home import knowledge_path

    b = Brain(
        seed=0
    )  # FULL brain, persist ON; sim arm = dormant internal substrate (no real hardware)
    mem = (
        DurableChatMemory()
    )  # a VIEW over the ONE durable brain lake (home knowledge.jsonl), noise-filtered
    atexit.register(lambda: _shutdown(b, mem))
    start_head = _head()
    offsets, n, remembered = {}, 0, 0
    sb = "ON" if os.environ.get("NORD_SELFMOD") == "1" else "off"
    print(
        f"[learn-daemon] FULL-brain learning; chat->the ONE lake ({knowledge_path()}); self-build={sb}; "
        f"auto-restart on new commit. head={start_head[:8]}",
        flush=True,
    )
    while True:
        # (1) FULL-brain ticks: wake perception/cognition/learning + sleep neurogenesis/replay/consolidation.
        for _ in range(TICKS_PER_CYCLE):
            try:
                b.tick()
            except Exception as e:  # a bad tick must never kill the daemon
                print(f"[learn-daemon] tick err: {type(e).__name__}: {e}", flush=True)
        # (2) ingest new REAL chat turns -> durable memory + respond() learning (noise filtered out).
        for path in sorted(glob.glob(TRANSCRIPTS), key=lambda p: os.path.getmtime(p)):
            turns, off = _new_turns(path, offsets.get(path, 0))
            offsets[path] = off
            for role, text in turns:
                try:
                    if mem.remember(role, text):  # durable + noise-filtered (real memory only)
                        b.respond(
                            text[:2000]
                        )  # learn from the real turn (free text -> no arm actuation)
                        remembered += 1
                except Exception as e:
                    print(f"[learn-daemon] ingest err: {type(e).__name__}: {e}", flush=True)
        # (3) AUTO-UPDATE: a NEW COMMIT (HEAD changed) -> persist + exit so the launcher loads the new build.
        if _head() != start_head:
            print(
                "[learn-daemon] new commit detected -> persisting + restarting to load the new build.",
                flush=True,
            )
            _shutdown(b, mem)
            os._exit(
                0
            )  # the launcher's restart loop relaunches with the new code; learning reloads from disk
        n += 1
        if n % LOG_EVERY == 0:
            try:
                st = b.state()
                print(
                    f"[learn-daemon] cycle {n}: nights={st.get('nights_elapsed')} "
                    f"grown_units={st.get('grown_units')} reasoning_acc={st.get('reasoning_accuracy')} "
                    f"chat_remembered={remembered}",
                    flush=True,
                )
            except Exception:
                pass
        time.sleep(POLL)


if __name__ == "__main__":
    main()
