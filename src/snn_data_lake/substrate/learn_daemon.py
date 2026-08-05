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
      shared home brain_lake.bin) -- but ONLY the REAL turns: the noise filter drops tool output,
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

import atexit
import glob
import json
import os
import subprocess
import sys
import time

from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.substrate.brain_telemetry import Recorder
from snn_data_lake.substrate.chat_memory import DurableChatMemory
from snn_data_lake.substrate.cockpit import Acts, Turns, serve
from snn_data_lake.substrate.self_mod_harness import _repo_root

# PROMOTED into the governed package 2026-08-04 under P4c (the daemon is the brain's DRIVE, not a
# launcher beside it). It lived at the repo root as `learn_daemon.py`, outside every P3/P4 gate, with
# no self-check and not one assert -- the most load-bearing process in the project and the least
# verified file in it. `learn_daemon.py` at the root is now a THIN SHIM so the documented command,
# learn_daemon.bat, and the PyInstaller build all keep working unchanged.
#
# THREE MODULE-SCOPE SIDE EFFECTS WERE REMOVED IN THE MOVE, and they were latent bugs the moment this
# became importable. `run_selfchecks.py` runs part of the suite IN ONE PROCESS via runpy, so anything
# this module did at import time would have applied to every module after it:
#   * `os.chdir(os.path.dirname(os.path.abspath(__file__)))` -- would now chdir into substrate/ (the
#     path is relative to THIS file, which moved), and under the in-process runner it would have
#     changed the working directory of the entire gate run.
#   * `sys.path.insert(0, "src")` -- unnecessary for an installed package module, and a CWD-relative
#     path entry is meaningless once the CWD is not the repo root.
#   * `os.environ.setdefault("NORD_GPU", "1")` -- would have forced GPU preference on every module in
#     the gate process, silently changing what the suite measures.
# All three now happen inside main(), which is the only context where they are correct: the daemon
# process. The repo root is taken from the ANCHORED derivation rather than counted from __file__.

# Default transcript glob. NOT a P1 threshold (it arbitrates nothing -- it is a data source location),
# but the old default hardcoded one developer's absolute home directory, so on any other machine the
# daemon silently ingested zero chat and reported success. Derived from the user's home now, with the
# env var still the override.
_DEFAULT_TRANSCRIPTS = os.path.join(
    os.path.expanduser("~"), ".claude", "projects", "D--AI-project", "*.jsonl"
)
TRANSCRIPTS = os.environ.get("SNDL_CHAT_TRANSCRIPT", _DEFAULT_TRANSCRIPTS)
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


def _lock_one(path):
    """Take an OS-level exclusive lock on `path`, or return None. The OS releases it on exit OR crash,
    so there are no stale PID files to reap."""
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _acquire_singleton():
    """ONE daemon per MACHINE. Returns True if this process may run.

    WHY MACHINE-WIDE AND NOT PER-HOME, corrected 2026-08-02. This used to lock only
    <home>/learn_daemon.lock, so "one daemon" meant one daemon PER NORD_HOME -- and two were found
    running at once: the .venv-cu128 one the launcher starts, and a second on a bare uv cpython-3.11
    with no torch installed at all. Two live Brains against one 414 MB append-only brain_lake.bin is
    a data race nobody designed for, and it is also a P4 violation ("one brain"). The per-home lock
    could not see it, because a different home is a different lock file.

    So the machine-wide lock is the one that decides, and the per-home lock is kept as well: it still
    catches the same-home case earlier and its file is where an operator looks.

    ESCAPE HATCH, deliberately narrow: NORD_DAEMON_INSTANCE=<name> namespaces the machine-wide lock,
    so a probe daemon on a scratch home can run beside the real one when someone means it. Unset --
    the normal case -- there is exactly one."""
    global _LOCK_FH
    import tempfile

    from snn_data_lake.substrate.home import home

    instance = (os.environ.get("NORD_DAEMON_INSTANCE") or "").strip() or "default"
    machine_lock = os.path.join(tempfile.gettempdir(), f"sndl_learn_daemon.{instance}.lock")
    held = []
    for path in (machine_lock, os.path.join(home(), "learn_daemon.lock")):
        fh = _lock_one(path)
        if fh is None:
            for open_fh in held:
                open_fh.close()
            return False
        held.append(fh)
    _LOCK_FH = held  # both handles stay open for the process lifetime
    return True


def main():
    from snn_data_lake.substrate.home import knowledge_path

    # The three former module-scope side effects, now scoped to the daemon process (see the header).
    os.chdir(_repo_root())
    if "src" not in sys.path:
        sys.path.insert(0, "src")
    # GPU-PRIORITIZED, ALWAYS-ON: the live brain prefers CUDA whenever it is available. The
    # [train]/perception tier -- neurogenesis growth, the e-prop/surrogate perception encoders, the
    # growable/laminar readouts, the develop curriculum -- runs on the GPU; the light stdlib
    # narrator/bookkeeping stays on CPU (forcing scalar ops onto CUDA only adds transfer overhead).
    # Launch under .venv-cu128 (learn_daemon.bat) for a real GPU; in the torch-CPU stdlib venv this
    # resolves to CPU gracefully (byte-identical). Off: NORD_GPU=0.
    os.environ.setdefault("NORD_GPU", "1")

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
    )  # a VIEW over the ONE durable brain lake (home brain_lake.bin), noise-filtered
    # Telemetry over the ONE brain: a loopback dashboard + an append-only JSONL trace. This does not
    # construct a brain (Policy 4) -- it is a window onto the one above. Loopback-only by design; it is
    # observability for a local process, not a network service.
    tele_port = int(os.environ.get("NORD_TELEMETRY_PORT", "8137"))
    turns = Turns()  # chat typed into the cockpit; drained BELOW, on this thread. See pump() note.
    # OWNER-AUTHORISED ACTUATION (2026-08-03): the cockpit may queue an allow-listed arm command.
    # Drained on THIS thread beside the chat queue, for the reason Acts' docstring gives -- mj_forward
    # from an HTTP thread while this one runs mj_step on the same mjData is undefined behaviour.
    acts = Acts()
    try:
        serve(b, port=tele_port, turns=turns, acts=acts, block=False)
    except (
        OSError
    ) as e:  # port taken (e.g. a second tool already watching) -- never kill the daemon
        print(f"[learn-daemon] cockpit not started: {e}", flush=True)
    # every=1, NOT TICKS_PER_CYCLE. Recorder.every counts CALLS to write(), and write() is called once
    # per cycle below -- so every=TICKS_PER_CYCLE meant one snapshot per 20 cycles (400 ticks), 20x
    # sparser than the "one snapshot per cycle" the call site claims. The two disagreed silently.
    # _repo_root(), not dirname(__file__): this file now lives in substrate/, so the old expression
    # would have written telemetry to src/snn_data_lake/substrate/data/telemetry.jsonl.
    rec = Recorder(os.path.join(_repo_root(), "data", "telemetry.jsonl"), every=1)

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
            # (1c) ARM VIEW: draw the arm HERE, on the tick thread, for the same reason chat is
            # pumped here. A MuJoCo Renderer owns a GL context bound to its creating thread;
            # rendering from the HTTP thread does not raise, it returns garbage, and a lock does
            # not help. Demand-gated inside render_if_wanted(), so a cockpit nobody has open
            # costs one boolean per tick.
            try:
                _arm = b.brain.arm.arm
                _arm.view.render_if_wanted(_arm.model, _arm.data)
            except Exception:  # a picture must never be able to stop the brain
                pass
            try:
                turns.pump(b, budget=1)
                # bounded: one queued command per TICK. (This said "per cycle" and was wrong by
                # TICKS_PER_CYCLE=20 -- both pumps sit inside the tick loop, which is the whole
                # point of the paragraph above.)
                acts.pump(b, budget=1)
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


def _selfcheck():
    """The daemon's first self-check, added with its P4c promotion. It had NONE and not one assert.

    It must never start the drive loop: `run_selfchecks.py` executes this module's __main__, and a
    daemon that runs forever would hang the gate. So it verifies the things that can be checked
    without driving a brain -- and each assert below has a red that was CONSTRUCTED AND RUN, not
    assumed (P8). What it does NOT check is stated at the end rather than left implied.

    That claim was FALSE for checks (e) and (g) until 2026-08-04: both were substring tests over
    this file's own source whose search literals appeared in the assert lines themselves, so no
    edit to the mechanism could redden them. Both are now AST checks and both reds were rebuilt
    and run. Recorded here rather than quietly fixed, because "each assert has a constructed red"
    is exactly the kind of claim that has to stay true to be worth anything."""
    root = _repo_root()

    # (a) IMPORTING THIS MODULE HAS NO SIDE EFFECTS -- measured in a CHILD process, because the only
    #     honest way to ask "what does importing this do?" is to import it somewhere it has not
    #     already been imported. (Reloading it here would re-execute __main__ recursively.) This is
    #     the assert that would have caught the three module-scope statements the promotion removed.
    #     RED, constructed and run: restoring any of the os.chdir / sys.path.insert /
    #     os.environ.setdefault lines to module scope makes the child report a different cwd or a
    #     set NORD_GPU, and this fires.
    probe = (
        "import os,sys,json;"
        "cwd0=os.getcwd();g0=os.environ.get('NORD_GPU');"
        "import snn_data_lake.substrate.learn_daemon as d;"
        "print(json.dumps({'cwd_same':os.getcwd()==cwd0,'gpu_same':os.environ.get('NORD_GPU')==g0}))"
    )
    env = {**os.environ, "PYTHONPATH": os.path.join(root, "src"), "PYTHONUTF8": "1"}
    env.pop("NORD_GPU", None)
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=300,
    )
    assert out.returncode == 0, (
        f"importing the daemon in a clean process failed:\n{out.stderr[-2000:]}"
    )
    side = json.loads(out.stdout.strip().splitlines()[-1])
    assert side["cwd_same"], (
        "importing the daemon must not change the working directory; run_selfchecks runs part of the"
        " suite in ONE process, so an import-time chdir moves every module after this one"
    )
    assert side["gpu_same"], (
        "importing the daemon must not set NORD_GPU; that would change what the whole gate measures"
    )

    # (b) THE REPO ROOT RESOLVES, and resolves to a real repo rather than to substrate/. RED: swap
    #     _repo_root() for os.path.dirname(__file__) and both of these fail.
    assert os.path.isdir(root), f"repo root does not resolve: {root!r}"
    assert os.path.isfile(os.path.join(root, "run_selfchecks.py")), (
        f"repo root {root!r} holds no run_selfchecks.py -- the anchored derivation is wrong, which"
        " would send telemetry and the singleton lock somewhere arbitrary"
    )
    assert os.path.basename(root) != "substrate", (
        "repo root resolved to this module's own directory"
    )

    # (c) POLICY 4: exactly ONE brain construction, through the sanctioned path.
    #
    #     COUNTED OVER THE AST, NOT THE TEXT. The first version of this assert was
    #     `src.count("make_brain(") == 1` and it failed immediately at 8 -- because a substring scan
    #     also counts the import line, every docstring mention, and the assert's own message. That is
    #     not a near-miss worth hiding: a checker that counts its own error text is measuring the
    #     wrong thing, and had the file happened to contain exactly one mention it would have passed
    #     while proving nothing. Walking Call nodes counts CONSTRUCTIONS.
    #     RED, constructed and run: adding a second make_brain() call makes this fire (2 != 1), and
    #     adding a bare Brain(...) call makes the next one fire.
    import ast

    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)

    def _calls_named(name):
        n = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f_ = node.func
                if isinstance(f_, ast.Name) and f_.id == name:
                    n += 1
                elif isinstance(f_, ast.Attribute) and f_.attr == name:
                    n += 1
        return n

    n_make = _calls_named("make_brain")
    assert n_make == 1, (
        f"the daemon must construct the ONE brain exactly once via make_brain() (AST call sites: {n_make})"
    )
    n_direct = _calls_named("Brain")
    assert n_direct == 0, (
        f"the daemon must never construct a Brain directly ({n_direct} site(s)) -- Policy 4 says"
        " make_brain() is the one construction path"
    )

    # (d) P4b: the daemon keeps NO durable store of its own. Its only writes are the ONE lake (via
    #     the brain and DurableChatMemory), the telemetry trace, and the singleton lock. Also AST-
    #     counted, for the same reason. RED: add a json.dump(...) call and this fires.
    n_dump = _calls_named("dump") + _calls_named("dumps")
    assert n_dump == 0, (
        f"the daemon must not serialise its own state ({n_dump} json.dump/dumps call(s)) -- what the"
        " brain learns goes in brain_lake.bin under its own record tag (P4b)"
    )

    # (e) The cockpit gets its ACTS QUEUE. This is the C10 regression, asserted at the daemon end:
    #     serve() silently dropped `acts`, so every arm control including the E-STOP was dead.
    #
    #     COUNTED OVER THE AST, NOT THE TEXT -- corrected 2026-08-04. This was
    #     `assert "acts=acts" in src`, and it was TRUE BY CONSTRUCTION: the literal "acts=acts"
    #     appears in the assert's own line and in the comment above it, so `src` contains it whether
    #     or not the serve( call does. MEASURED: deleting `acts=acts` from the serve( call left the
    #     old assert PASSING. That is the same mistake check (c) records making and fixing three
    #     blocks up ("a checker that counts its own error text is measuring the wrong thing") --
    #     repeated here, on the guard protecting the E-STOP, which is the worst place in this file
    #     for a check that cannot go red.
    #     RED, constructed and run: removing the `acts=` keyword from the serve( call makes this fire.
    serve_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "serve")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "serve")
        )
    ]
    assert len(serve_calls) == 1, (
        f"the daemon must serve the cockpit exactly once (AST call sites: {len(serve_calls)})"
    )
    assert any(kw.arg == "acts" for kw in serve_calls[0].keywords), (
        "the daemon must pass its Acts queue to serve() -- without it every cockpit arm control,"
        " including the E-STOP, is inert (specs/010 C10)"
    )

    # (f) ASCII-only output, checked over the whole file (Windows cp1252 crashes on non-ASCII).
    try:
        src.encode("ascii")
    except UnicodeEncodeError as e:
        raise AssertionError(f"daemon source must be pure ASCII: {e}") from None

    # (g) The singleton lock is per-MACHINE as well as per-home. Same correction as (e): this was
    #     `assert "gettempdir()" in src and "home()" in src`, whose own line contains both literals,
    #     so it was TRUE BY CONSTRUCTION -- MEASURED: replacing the tempdir lock path with a
    #     hardcoded string left the old assert PASSING. Scoped to _acquire_singleton's own AST and
    #     asserted on the CALLS, so the check fails when the mechanism goes rather than when the
    #     spelling does.
    #     RED, constructed and run: drop the gettempdir() lock (leaving only the per-NORD_HOME one,
    #     which is what let two daemons run at once) and this fires.
    singleton_fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_acquire_singleton"
        ),
        None,
    )
    assert singleton_fn is not None, "_acquire_singleton is gone; the one-daemon guarantee with it"
    lock_calls = {
        n.func.attr
        for n in ast.walk(singleton_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    } | {
        n.func.id
        for n in ast.walk(singleton_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "gettempdir" in lock_calls and "home" in lock_calls, (
        "the singleton lock must be both per-machine (tempfile.gettempdir) and per-home (home());"
        f" one daemon per NORD_HOME is not one daemon. Calls found: {sorted(lock_calls)}"
    )

    print("=" * 96)
    print("LEARN DAEMON -- the brain's DRIVE (P4c). Self-check; does NOT start the drive loop.")
    print("=" * 96)
    print(f"repo root (anchored)      : {root}")
    print(f"transcripts glob          : {TRANSCRIPTS}")
    print(f"telemetry port (default)  : {os.environ.get('NORD_TELEMETRY_PORT', '8137')}")
    print("import side effects        : none (cwd unchanged, NORD_GPU unset by import)")
    print("brain construction         : exactly one make_brain(), no direct Brain()")
    print("durable state of its own   : none (P4b -- the one lake only)")
    print("cockpit acts queue         : forwarded to serve() (C10 regression asserted)")
    print("-" * 96)
    print(
        "NOT CHECKED HERE, stated rather than implied: this self-check never drives a tick, never"
    )
    print("opens the lake, never binds the port and never ingests a transcript. It verifies the")
    print("daemon's CONTRACT, not its behaviour under load. Driving the real loop is what")
    print("`python learn_daemon.py` does, and no gate does that.")
    print("=" * 96)
    print(
        "self-check OK: daemon is import-side-effect-free, anchors its own root, constructs the ONE"
        " brain once, keeps no second durable store, and wires the cockpit's act queue"
    )


if __name__ == "__main__":
    # The module __main__ is the SELF-CHECK, matching every other module in the package -- that is
    # what run_selfchecks.py executes. Starting a forever-loop here would hang the gate. The daemon
    # itself is started by the root shim `learn_daemon.py`, or explicitly with --serve.
    if "--serve" in sys.argv:
        main()
    else:
        _selfcheck()
