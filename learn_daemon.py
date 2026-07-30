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

# GPU-PRIORITIZED, ALWAYS-ON: the live brain prefers CUDA whenever it is available. The [train]/perception
# tier -- neurogenesis growth, the e-prop/surrogate perception encoders, the growable/laminar readouts, the
# develop curriculum -- then runs on the GPU; the light stdlib narrator/bookkeeping stays on CPU (forcing
# scalar ops onto CUDA only adds transfer overhead). Launch under .venv-cu128 (learn_daemon.bat) to get a
# real GPU; in the torch-CPU stdlib venv this resolves to CPU gracefully (byte-identical). Off: NORD_GPU=0.
os.environ.setdefault("NORD_GPU", "1")

from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.substrate.brain_telemetry import Recorder
from snn_data_lake.substrate.chat_memory import DurableChatMemory
from snn_data_lake.substrate.cockpit import Turns, serve

TRANSCRIPTS = os.environ.get(
    "SNDL_CHAT_TRANSCRIPT", r"C:\Users\vinhl\.claude\projects\D--AI-project\*.jsonl"
)
TICKS_PER_CYCLE = int(os.environ.get("SNDL_TICKS", "20"))
CHUNK_LINES = int(os.environ.get("SNDL_CHAT_CHUNK", "400"))
# A tick MEASURED >=240 ms under load (2026-07-30: daemon cycle counter sampled over 7.4 min, one
# 50-cycle boundary crossed; a block is 1000 ticks), so a cycle of 20 is >=5 s. The "~33 ms" this
# comment used to claim was the unloaded nominal figure and is STALE by ~7x -- treat 33 ms as the
# budget, 240 ms as the observed floor. At the old POLL=5 the brain was idle ~88% of the time, which
# is fine for a background learner and useless for a live dashboard: the cockpit polls at 1 Hz and
# had nothing new to show. Raise SNDL_POLL if you want the daemon to stop using a core.
POLL = float(os.environ.get("SNDL_POLL", "0.5"))
# Ingest is BOUNDED per cycle. respond() costs 0.1-6 s per turn, and the ingest loop used to run every
# pending turn before returning to the tick loop -- so a burst of conversation could stall the brain for
# a minute or more. Measured: ~0.25 ticks/sec against a capability of 30. The backlog is not dropped,
# just spread over later cycles; the offsets only advance for turns actually consumed.
MAX_INGEST_PER_CYCLE = int(os.environ.get("SNDL_MAX_INGEST", "4"))
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
    """New (role, text) turns from a Claude Code .jsonl beyond BYTE offset `start`, <= CHUNK_LINES lines.

    `start` is a byte offset, not a line index. The first version called f.readlines() and sliced
    [start:start+CHUNK_LINES], which read the WHOLE file every cycle to look at 400 lines -- across 45
    transcripts totalling 319 MB, that was ~319 MB of I/O per cycle to extract a few kilobytes. It was
    the single largest cost in the daemon loop by two orders of magnitude, and it grew every time the
    user said anything, so the brain got slower the more it was talked to.

    Only complete lines advance the offset: a transcript being appended to while we read it can hand
    back a torn final line, and consuming it would skip that turn forever once the writer completes it.
    """
    out = []
    pos = start
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            f.seek(start)
            for _ in range(CHUNK_LINES):
                line = f.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    break  # torn tail: leave the offset before it, re-read once it is finished
                pos = f.tell()
                _collect_turn(line, out)
    except OSError:
        return [], start
    return out, pos


def _collect_turn(line, out):
    """Parse one transcript line; append (role, text) if it is a real user/assistant turn."""
    try:
        rec = json.loads(line)
    except Exception:
        return
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


_LOCK_FH = None  # keeps the single-instance lock handle alive for the whole process lifetime


def _acquire_singleton():
    """ONE daemon only: hold an OS-level exclusive lock on <home>/learn_daemon.lock. The OS releases it
    automatically on exit OR crash (no stale PID files to reap). Returns True if this process acquired the
    lock; False if another daemon already holds it (a second launch then exits, so exactly one runs)."""
    global _LOCK_FH
    from snn_data_lake.substrate.home import home

    fh = open(os.path.join(home(), "learn_daemon.lock"), "a+")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    _LOCK_FH = fh
    return True


def main():
    from snn_data_lake.substrate.home import knowledge_path

    if (
        not _acquire_singleton()
    ):  # combine into ONE daemon: a second launch detects the lock and exits
        print(
            "[learn-daemon] another instance already holds the single-instance lock -> exiting.",
            flush=True,
        )
        return

    b = make_brain(
        seed=0
    )  # Policy 4: the ONE sanctioned construction path; FULL brain, persist ON; sim arm = dormant substrate
    mem = (
        DurableChatMemory()
    )  # a VIEW over the ONE durable brain lake (home knowledge.jsonl), noise-filtered
    # Telemetry over the ONE brain: a loopback dashboard + an append-only JSONL trace. This does not
    # construct a brain (Policy 4) -- it is a window onto the one above. Loopback-only by design; it is
    # observability for a local process, not a network service.
    tele_port = int(os.environ.get("NORD_TELEMETRY_PORT", "8137"))
    turns = Turns()  # chat typed into the cockpit; drained BELOW, on this thread. See pump() note.
    try:
        serve(b, port=tele_port, turns=turns, block=False)
    except (
        OSError
    ) as e:  # port taken (e.g. a second tool already watching) -- never kill the daemon
        print(f"[learn-daemon] cockpit not started: {e}", flush=True)
    # every=1, NOT TICKS_PER_CYCLE. Recorder.every counts CALLS to write(), and write() is called once
    # per cycle below -- so every=TICKS_PER_CYCLE meant one snapshot per 20 cycles (400 ticks), 20x
    # sparser than the "one snapshot per cycle" the call site claims. The two disagreed silently.
    rec = Recorder(os.path.join(os.path.dirname(__file__), "data", "telemetry.jsonl"), every=1)

    atexit.register(lambda: _shutdown(b, mem))
    atexit.register(rec.close)
    start_head = _head()
    offsets, n, remembered = {}, 0, 0
    backlog = {}  # path -> turns read but not yet ingested (the per-cycle cap defers, never drops)
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
            # (1b) COCKPIT CHAT: drain queued turns HERE, on the thread that owns the tick loop.
            # respond() writes _help_drive into the motor-drive path, exposes the language organ to
            # plasticity the sleep tick durably saves, and can route a command through the gated arm
            # -- and there is no lock anywhere in the brain stack. Answering from the HTTP thread
            # would be a data race on an object that is also writing to disk, so the server only
            # queues and this loop is the only thing that ever touches the brain.
            #
            # Per TICK, not per cycle: a cycle is 20 ticks plus ingest plus POLL, so pumping once per
            # cycle made a typed message wait for the whole cycle before it was even looked at. That
            # is the difference between a chat box and a suggestion box.
            try:
                turns.pump(b, budget=1)
            except Exception as e:  # a bad turn must never stop the loop
                print(f"[learn-daemon] chat err: {type(e).__name__}: {e}", flush=True)
        try:  # record one snapshot per cycle -- the replayable trace a renderer reads later
            rec.write(b)
        except Exception as e:  # telemetry must never take the brain down with it
            print(f"[learn-daemon] telemetry err: {type(e).__name__}: {e}", flush=True)
        # (2) ingest new REAL chat turns -> durable memory + respond() learning (noise filtered out).
        # Budget counts respond() calls, not turns looked at: the noise filter is microseconds, so
        # skipping a thousand tool-output lines is free and must not consume the cap. Turns that do
        # not fit this cycle stay in `backlog` and the byte offset is NOT re-advanced past them, so
        # nothing is dropped -- a burst of conversation is spread over cycles instead of stalling one.
        budget = MAX_INGEST_PER_CYCLE
        for path in sorted(glob.glob(TRANSCRIPTS), key=lambda p: os.path.getmtime(p)):
            if budget <= 0:
                break
            pend = backlog.get(path) or []
            if not pend:
                pend, off = _new_turns(path, offsets.get(path, 0))
                offsets[path] = off
            while pend and budget > 0:
                role, text = pend.pop(0)
                try:
                    if mem.remember(role, text):  # durable + noise-filtered (real memory only)
                        b.respond(
                            text[:2000]
                        )  # learn from the real turn (free text -> no arm actuation)
                        remembered += 1
                        budget -= 1
                except Exception as e:
                    print(f"[learn-daemon] ingest err: {type(e).__name__}: {e}", flush=True)
            backlog[path] = pend
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
        # (4) OPTIONAL per-cycle charge adapters (substrate.region_wiring): an explicit plug-point for
        #     genuinely-PERIODIC faculties. Empty by default -> a cheap no-op. NOTE (Policy 3): this is NOT
        #     how a dormant faculty is made live -- pinging a faculty here without routing its output into
        #     behaviour is theater. The honest path is to WIRE the faculty into the tick (so it charges
        #     behaviour) and let substrate.dormancy_gate enforce no-silent-dormancy. Authoritative coverage
        #     = region_wiring.live_coverage() (delegates to the gate), logged below.
        try:
            from snn_data_lake.substrate import region_wiring

            region_wiring.charge_all(b, {"cycle": n})
        except Exception as e:
            print(f"[learn-daemon] wiring err: {type(e).__name__}: {e}", flush=True)
        n += 1
        if n % LOG_EVERY == 0:
            # `region_wiring` is imported OUT of the try above deliberately: it used to be bound only
            # inside that block, so an import failure left this reporter raising NameError into a
            # message-less `except: pass` -- the Policy-3 coverage number would simply stop appearing,
            # with nothing in the log saying why. That is the exact silent-degradation shape this
            # project keeps burning on: a stage fails, swallows it, and the absence looks like calm.
            from snn_data_lake.substrate import region_wiring as _rw

            try:
                st = b.state()
                cov = _rw.coverage()
                lc = _rw.live_coverage()  # Policy-3 authoritative coverage (dormancy gate)
                print(
                    f"[learn-daemon] cycle {n}: nights={st.get('nights_elapsed')} "
                    f"grown_units={st.get('grown_units')} reasoning_acc={st.get('reasoning_accuracy')} "
                    f"chat_remembered={remembered} | wired={lc['reachable']}/{lc['total']} "
                    f"exempt={lc['exempt']} dormant={lc['dormant']} adapters={cov['n_wired_faculties']}",
                    flush=True,
                )
            except Exception as e:  # typed, like the file's five other handlers
                print(f"[learn-daemon] coverage-report err: {type(e).__name__}: {e}", flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
