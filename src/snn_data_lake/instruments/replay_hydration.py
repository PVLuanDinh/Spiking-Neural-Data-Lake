"""R2 / T103 campaign instrument, rebuilt on main: does a restarted brain's interaction learner --
restored from its `r2.v1` checkpoint (cursor + canonical-prefix binding) and hydrated from the lake's
typed learner events (teach, turn, sleep, sleep_bout) -- equal an uninterrupted twin through the next
night's sleep? (specs/022 evidence/r2-typed-replay-hydration.md 2.2-2.5, 2.11, 2.12 and the dated
amendments; ported from claude/r2 56031ab9 and adapted to main's mechanism: `Brain._r2_boot`,
`SpikingInteractionLearner.hydrate`, `scan_learner_events`, `Brain._r2_sleep_step`.)

Every process is a separate interpreter on its OWN fresh NORD_HOME (a restoring process gets a COPY of
the killed process's home), so a kill is a real `os._exit(0)` and a restart is a real fresh process.
Nothing here writes outside those homes; every fault is constructed inside one process (a patch or a
planted lake record) and production has no switch for any of them.

Kill points: K0 close; K1 the end-of-night boundary tick then `os._exit`; K2 as K1 plus M=6
post-boundary turns; K3 as K2 plus J=3 sleep ticks of the next night (P2, redo re-executes them); K4
`os._exit` between the night's last `sleep` event and the checkpoint write (the first flush); B (kind
`sleep_bout`, owner #2106) a staged bout after the boundary, killed before any checkpoint covers it.
Twins run the same schedule and are never killed. After the kill point every restoring process and its
twin tick until S=6 sleep ticks have run, recording the LIVE objects (`_g0c_walk`, never the codec).
Sleep replay is retired on main (owner1039): the sleep tick is the structural pass, so the
must-differ control (R2-c4) is the structural-off twin and the rewirer weight flip of the O_rw red.

The incumbent arms I_K1 / I_K2 (2.11 D1) run THIS FILE at the base head e1b3ae3c with
PYTHONPATH=<base checkout>/src (`--head base`); each compares only to its own twin at that head.
The FAILWRITE arm (the #1037 latch schedule A3, `fw_*` roles) and R2-c8 (the three bout crash
points, planted bout slots, one-frame check, FAILWRITE with B = sleep_bout) run in every seed.
The percept arm (P3, K1g / R2-c7) is another lane's.

  PYTHONUTF8=1 PYTHONPATH=src python src/snn_data_lake/instruments/replay_hydration.py --selfcheck
  ... --seed SEED OUT_DIR [--head base --base-src PATH] [--roles a,b]   one seed's processes
  ... --campaign OUT_DIR [--seeds a,b,...] [--base-src PATH] [--workers N]  the seeds (dev block,
      sequential by default: each seed's processes are heavy)
"""

import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import random
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

DEV_SEEDS = tuple(range(15000, 15008))
S_SLEEP = 6  # sleep ticks recorded after the kill point
M_POST = 6  # K2/K3's post-boundary turns
J_NIGHT = 3  # K3: sleep ticks of the next night before the kill
#: per-process fields, never continuation state (a restart legitimately changes them)
_RUNTIME = (
    "_redo_count",
    "_hydrated",
    "_hydration_count",
    "_hydration_refusals",
    "_unjournaled_refused",
    "_effect_depth",
)
STAGES = [
    ("nrem", {"presentations": 8, "structural_candidates": None}),
    ("rem", {"presentations": 4, "structural_candidates": None}),
]

#: (role, parent pre-process whose home it restores from, or None for a fresh home)
R2_ROLES = (
    ("pre_K0", None),
    ("pre_K1", None),
    ("pre_K2", None),
    ("pre_K3", None),
    ("pre_K4", None),
    ("pre_B", None),  # c8 crash point 2: killed after the bout's effect, before any checkpoint
    ("pre_Bpre", None),  # c8 crash point 1: killed after the event append, before the effect
    ("pre_Bpost", None),  # c8 crash point 3: killed after the checkpoint covering the bout
    ("pre_rw", None),  # K1 schedule with one explicit replay BEFORE the boundary (O_rng)
    ("twin_K1", None),  # K0/K1: the complete boundary tick, never closed or killed
    ("twin_K4", None),  # K4: sample before the checkpoint, before the boundary's wake percept
    ("twin_K2", None),
    ("twin_K3", None),
    ("twin_B", None),
    ("twin_rw", None),
    ("twin_off", None),
    ("twin_rwv", None),  # O_rng's vacuous control: the K1 schedule, replay after the boundary only
    ("res_K0", "pre_K0"),
    ("res_K1", "pre_K1"),
    ("res_K2", "pre_K2"),
    ("res_K3", "pre_K3"),
    ("res_K4", "pre_K4"),
    ("res_B", "pre_B"),
    ("res_Bpre", "pre_Bpre"),
    ("res_Bpost", "pre_Bpost"),
    ("res_rw", "pre_rw"),
    ("O_rw", "pre_rw"),
    ("O_wm", "pre_K2"),
    ("O_hyd", "pre_K2"),
    ("O_rng", "pre_rw"),
    ("O_rngv", "pre_K1"),  # the lone after-restore replay: must NOT fire (a control, not a red)
    ("O_w", "pre_K2"),
    ("O_stack", "pre_K2"),
    ("O_sleepskip", "pre_K3"),
    ("O_cursor", "pre_K1"),
    ("O_bout", "pre_B"),
    ("DUP_on", "pre_K2"),
    ("DUP_off", "pre_K2"),
    ("REORDER_on", "pre_K2"),
    ("REORDER_off", "pre_K2"),
    ("HELDOUT_on", "pre_K2"),
    ("HELDOUT_off", "pre_K2"),
    ("MISSING_on", "pre_K2"),
    ("MISSING_off", "pre_K2"),
    ("CTXREAD_on", "pre_K1"),
    ("CTXREAD_off", "pre_K1"),
    ("SLEEPDIGEST_on", "pre_K3"),
    ("SLEEPDIGEST_off", "pre_K3"),
    ("BOUTDIGEST_on", "pre_B"),
    ("BOUTDIGEST_off", "pre_B"),
    ("BOUTPROGRESS_on", "pre_B"),
    ("BOUTPROGRESS_off", "pre_B"),
    ("CHRONO", "pre_K2"),
    ("MOTOR", "pre_K2"),  # the actuator check's red: the restore reaches So101Arm.apply() once
) + (
    # FAILWRITE (A3, owner #1037): never-faulted references, the faulting survivors, their fresh-OS-
    # process restarts, the constructed reds, and the four #1048 crash points
    ("fw_ref_A", None),
    ("fw_ref_AB", None),
    ("fw_ref_ABb", None),
    ("fw_t_absent", None),
    ("fw_t_landed", None),
    ("fw_t_torn", None),
    ("fw_b_absent", None),
    ("fw_b_landed", None),
    ("fw_b_torn", None),
    ("fw_rs_t_absent", "fw_t_absent"),
    ("fw_rs_t_landed", "fw_t_landed"),
    ("fw_rs_t_torn", "fw_t_torn"),
    ("fw_rs_b_absent", "fw_b_absent"),
    ("fw_rs_b_landed", "fw_b_landed"),
    ("fw_rs_b_torn", "fw_b_torn"),
    ("fw_red_kind_t", None),
    ("fw_red_kind_b", None),
    ("fw_red_retry", None),
    ("fw_red_count", None),
    ("fw_red_ckpt", None),
    ("fw_red_posed", None),
    ("fw_p_t_absent", None),  # FW-PARTIAL on all six arms (A3), armed at C(iii)
    ("fw_p_t_landed", None),
    ("fw_p_t_torn", None),
    ("fw_p_b_absent", None),
    ("fw_p_b_landed", None),
    ("fw_p_b_torn", None),
    ("fw_gate_ctl", "fw_t_absent"),
    ("fw_red_gate", "fw_t_absent"),
    ("fw_k_evt", None),
    ("fw_k_ckpt", None),
    ("fw_k_post", None),
    ("fw_rs_k_evt", "fw_k_evt"),
    ("fw_rs_k_ckpt", "fw_k_ckpt"),
    ("fw_rs_k_post", "fw_k_post"),
    ("fw_rs2_k_post", "fw_rs_k_post"),
)
BOUT_CRASH_POINTS = ("Bpre", "B", "Bpost")
BASE_ROLES = (
    ("pre_K1", None),
    ("pre_K2", None),
    ("twin_K1", None),
    ("twin_K2", None),
    ("res_K1", "pre_K1"),
    ("res_K2", "pre_K2"),
)
#: planted-fault arms: kind -> the reason the guard must report. The first four are READER faults (the
#: strict scan counts them); the last two reach the brain, which must refuse them visibly in `_r2_fault`.
PLANTED = {
    "DUP": "duplicate_id",
    "REORDER": "seq_regression",
    "HELDOUT": "heldout",
    "MISSING": "missing_slot",
    "CTXREAD": "unbound effect: ctxread",
    "SLEEPDIGEST": "sleep redo digest/progress mismatch",
    "BOUTDIGEST": "sleep_bout redo digest/progress mismatch",
    "BOUTPROGRESS": "sleep_bout redo digest/progress mismatch",
}
_READER_KINDS = ("DUP", "REORDER", "HELDOUT", "MISSING")
#: the twin each planted kind's arms are compared with
_PLANT_TWIN = {
    "DUP": "twin_K2",
    "REORDER": "twin_K2",
    "HELDOUT": "twin_K2",
    "MISSING": "twin_K2",
    "CTXREAD": "twin_K1",
    "SLEEPDIGEST": "twin_K3",
    "BOUTDIGEST": "twin_B",
    "BOUTPROGRESS": "twin_B",
}
#: red role -> the twin it must differ from
_RED_TWIN = {
    "O_wm": "twin_K2",
    "O_hyd": "twin_K2",
    "O_rng": "twin_rw",
    "O_w": "twin_K2",
    "O_stack": "twin_K2",
    "O_rw": "twin_rw",
    "O_sleepskip": "twin_K3",
    "O_cursor": "twin_K1",
    "O_bout": "twin_B",
}
_COMPARED_BEFORE = ("walk", "ring", "ring_ids", "rng", "cursor", "replay_counts", "rewirer")
_COMPARED_ROW = ("weights", "drawn", "drawn_ids", "rng", "replay_counts", "rewirer")
_BASE_SKIP = (
    "rewirer",
    "ring_ids",
    "drawn_ids",
    "replay_counts",
    "cursor",
)  # absent at the base head


def check_seeds(seeds):
    """The licence: only the dev block runs here. A confirmatory or reserve seed is read ONCE, by the
    integrator's go, and reading one destroys the pre-data licence -- so it needs R2_CONFIRMATORY_GO=1."""
    stray = [s for s in seeds if s not in DEV_SEEDS]
    if stray and os.environ.get("R2_CONFIRMATORY_GO") != "1":
        raise ValueError(
            f"seeds {stray} are outside the dev block 15000-15007: no confirmatory licence"
        )


# ---- the schedule (identical in every process of a seed; uses only calls both heads have) -------


def _lines():
    from snn_data_lake.readout.interaction_learning import _CONV_TRAIN, _conv_flat

    return [u for u, _ in _conv_flat(_CONV_TRAIN)]


def _tick_to_boundary(brain):
    """Tick until a night completes (`_enter_wake` ran and flushed the checkpoint)."""
    nights, ticks = brain._nights, 0
    while brain._nights == nights:
        brain.tick()
        ticks += 1
        if ticks > 4 * brain.day_period:
            raise RuntimeError("no completed night within four periods")
    return ticks


def _tick_sleeps(brain, n, observe=None):
    """Tick until `n` sleep ticks have run (the learner is touched by sleep ticks only)."""
    done, ticks = 0, 0
    while done < n:
        asleep = brain.tick() == "asleep"
        ticks += 1
        if ticks > 8 * brain.day_period:
            raise RuntimeError("no sleep within eight periods")
        if asleep:
            done += 1
            if observe is not None:
                observe()
    return ticks


def _schedule(brain, seed, replay_first=False):
    """A seeded shuffle of the conversation set through `respond(human=False)` (self-labelled), two
    corrections, then ticks to the end of a night."""
    order = _lines()
    random.Random(seed).shuffle(order)
    for u in order:
        brain.respond(u, human=False)
    brain.correct(order[0], "MEMORY")
    brain.correct(order[3], "GOAL")
    if replay_first:  # the learner RNG is drawn before the checkpoint, so its carrier is exercised
        _explicit_replay(brain)
    _tick_to_boundary(brain)
    return order


def _post_boundary(brain, order):
    """K2's M=6 post-boundary turns: 4 respond, 1 correct, 1 re-correction of a ring utterance."""
    for u in order[4:8]:
        brain.respond(u, human=False)
    brain.correct(order[5], "GOAL")
    brain.correct(order[0], "GOAL")


def _raw():
    from snn_data_lake.substrate import home

    return Path(home.knowledge_path()).read_bytes()


def _raw_state(raw, name):
    """The newest `name` blob in `raw`, decoded read-only from bytes already in hand. Never builds a
    BinLog or calls `home.load_state`: that path repairs a torn tail (`_repair_torn_tail` truncates), so an
    observer using it would shorten the lake between two C calls and the survivor would be blamed."""
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import _LEN

    got = None
    for a, b in _frames(raw)[0]:
        payload = raw[a + _LEN.size : b]
        if payload[:1] == home._T_STATE:
            st = home._unpack_state(payload)
            if st is not None and st[0].decode("ascii", "replace") == name:
                got = st[2]
    return got


def _raw_json(raw, name):
    """`home.load_state`'s value (JSON of the newest blob), decoded read-only from `raw`."""
    blob = _raw_state(raw, name)
    try:
        return None if blob is None else json.loads(blob.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return None


def _raw_rec(data):
    return {"len": len(data), "sha": hashlib.sha256(data).hexdigest()}


def _prefix_rec(data):
    """The complete-frame prefix of the lake (a torn tail is not part of it)."""
    return _raw_rec(data[: _frames(data)[1]])


def _frames(raw, start=None):
    """(start, end) of every COMPLETE length-prefixed frame from `start` (default: after the magic),
    and the end of the last one: the complete prefix a torn tail is repaired to."""
    from snn_data_lake.substrate.binlog import _LEN, _MAGIC

    i, spans = len(_MAGIC) if start is None else start, []
    while i + _LEN.size <= len(raw):
        (n,) = _LEN.unpack_from(raw, i)
        if i + _LEN.size + n > len(raw):
            break
        spans.append((i, i + _LEN.size + n))
        i += _LEN.size + n
    return spans, i


def _episode_kinds(raw, start):
    """Kinds of the complete frames appended after `start` ('tag:..' for a non-episode frame)."""
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import _LEN, unpack_episode

    tag, kinds = home._T_EPISODE, []
    for a, b in _frames(raw, start)[0]:
        payload = raw[a + _LEN.size : b]
        if payload[: len(tag)] == tag:
            kinds.append(unpack_episode(payload[len(tag) :]).get("kind"))
        else:
            kinds.append(f"tag:{payload[:1].hex()}")
    return kinds


def _bout_probe(brain):
    lrn = brain.brain._ixn
    return {
        "raw": _raw(),
        "cache": lrn._snapshot()["cache"],
        "counts": dict(sorted(getattr(lrn, "_replay_counts", {}).items())),
    }


def _bout_check(brain, probe, effect_applied):
    """R2-c8 (c): exactly one `sleep_bout` frame was appended, nothing before it moved, and the ring,
    authority and replay counts are what they were before the bout."""
    now, lrn = _raw(), brain.brain._ixn
    kinds = _episode_kinds(now, len(probe["raw"]))
    return {
        "frames_added": len(kinds),
        "kinds_added": kinds,
        "prefix_preserved": now[: len(probe["raw"])] == probe["raw"],
        "cache_equal": lrn._snapshot()["cache"] == probe["cache"],
        "replay_counts_equal": dict(sorted(getattr(lrn, "_replay_counts", {}).items()))
        == probe["counts"],
        "effect_applied": effect_applied,
    }


def _bout(brain, seed, result=None, out=None, point="B"):
    """The staged bout. `point` Bpre kills the process after the event is appended and before the
    effect runs (the preview learner's own `_run_bout` call is never the kill)."""
    import snn_data_lake.instruments.staged_sleep_bout as staged
    from snn_data_lake.instruments.staged_sleep_bout import run_bout

    probe = _bout_probe(brain)
    if point == "Bpre":
        real = staged._run_bout

        def kill_before_effect(subject, *a, **kw):
            if not hasattr(subject, "_r2_fault"):  # the preview proxy: run it
                return real(subject, *a, **kw)
            result["bout_check"] = _bout_check(brain, probe, effect_applied=False)
            _write(out, result)
            os._exit(0)

        staged._run_bout = kill_before_effect
    run_bout(brain, STAGES, rng_seed=seed)
    if result is not None:
        result["bout_check"] = _bout_check(brain, probe, effect_applied=True)


# ---- records (read off the live objects) --------------------------------------------------------


def _walk(learner):
    from snn_data_lake.readout.interaction_learning import _g0c_walk

    tree = _g0c_walk(learner)
    for name in _RUNTIME:
        tree["fields"].pop(name, None)
    return tree


def _plain(v):
    if isinstance(v, random.Random):
        return repr(v.getstate())
    if isinstance(v, dict):
        return sorted(((repr(k), _plain(x)) for k, x in v.items()), key=lambda p: p[0])
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(repr(x) for x in v)
    return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)


def _rewirer(brain):
    rw = getattr(brain, "_rewirer", None)
    return None if rw is None else _plain(vars(rw))


def _chrono(brain):
    neuro = getattr(brain, "neuro", None)
    return {
        "step": brain._step,
        "nights": brain._nights,
        "observed": getattr(neuro, "observed", None),
        "live_n": getattr(neuro, "live_n", None),
    }


def _before(brain):
    lrn = brain.brain._ixn
    return {
        "walk": _walk(lrn),
        "ring": [[label, key] for _, label, key in lrn._replay],
        "ring_ids": list(getattr(lrn, "_replay_ids", [])),
        "rng": repr(lrn.rng.getstate()),
        "cursor": list(getattr(lrn, "_cursor", None) or []) or None,
        "replay_counts": dict(sorted(getattr(lrn, "_replay_counts", {}).items())),
        "rewirer": _rewirer(brain),
    }


class _MotorCount:
    """Counts calls reaching `So101Arm.apply()` in THIS process by wrapping the class attribute in
    memory. No envelope file is opened for writing or changed; the wrapper always delegates."""

    def __init__(self):
        self.n = 0
        try:
            from snn_data_lake.brainstem.so101_arm import So101Arm
        except Exception:  # noqa: BLE001 - no arm module: nothing can actuate, the count stays 0
            return
        real, counter = So101Arm.apply, self

        def counted(arm, *a, **kw):
            counter.n += 1
            return real(arm, *a, **kw)

        So101Arm.apply = counted


def _watch(brain, motor, windows):
    """Record chronology + motor calls around every learner-affecting sleep step (main) or replay
    (base), and each replay draw's (key, label). Class-level, so the live instance carries no extra
    attribute: the walker and the brain's own checkpoint flush see exactly the learner production has."""
    from snn_data_lake.engine.brain import Brain

    cls = type(brain.brain._ixn)
    draws, real_interact = [], cls.interact

    def interact(self, stimulus, feedback, _record=True, _key=None, **kw):
        if not _record:
            draws.append([_key, feedback])
        return real_interact(self, stimulus, feedback, _record=_record, _key=_key, **kw)

    def around(real):
        def run(*a, **kw):
            before, m0 = _chrono(brain), motor.n
            out = real(*a, **kw)
            windows.append({"before": before, "after": _chrono(brain), "motor": motor.n - m0})
            return out

        return run

    cls.interact = interact
    if hasattr(Brain, "_r2_sleep_learning"):
        real = Brain._r2_sleep_learning
        Brain._r2_sleep_learning = lambda self: around(real)(self)
    else:
        real_replay = cls.replay
        cls.replay = lambda self, k=8: around(real_replay)(self, k=k)
    return draws


def _sleep_rows(brain, draws):
    rows, lrn = [], brain.brain._ixn

    def observe():
        rows.append(
            {
                "weights": lrn.weights(),
                "drawn": list(draws),
                "drawn_ids": list(getattr(lrn, "_last_drawn", [])),
                "rng": repr(lrn.rng.getstate()),
                "replay_counts": dict(sorted(getattr(lrn, "_replay_counts", {}).items())),
                "rewirer": _rewirer(brain),
            }
        )
        del draws[:]

    _tick_sleeps(brain, S_SLEEP, observe)
    return rows


# ---- planted faults (written into a restoring process's COPIED home before its brain is built) --


def _plant(kind):
    """Append one constructed record to the process's lake; returns its event_id."""
    from snn_data_lake.substrate.home import save_episode
    from snn_data_lake.substrate.learner_events import KIND_SPECS, read_learner_events

    if kind in ("BOUTDIGEST", "BOUTPROGRESS"):
        return _plant_bout(kind)
    # P3 percepts share this identity sequence and may follow the last sleep event.
    events, _ = read_learner_events(tuple(KIND_SPECS))
    wake = [e for e in events if e["kind"] in ("teach", "turn")]
    last, base = events[-1], wake[-1]
    session, seq = int(last["session"]), int(last["seq"])
    nxt = {"event_id": f"{session}-{seq + 1}", "seq": str(seq + 1), "session": str(session)}
    if kind == "DUP":  # a byte-distinct record repeating a ring event's id
        e = dict(wake[-3])
        e["target"] = "GREET" if e["target"] != "GREET" else "GOAL"
    elif kind == "REORDER":  # an unused id whose (session, seq) regresses
        e = dict(base, event_id=f"0-{base['seq']}", session="0", source="instrument reorder line")
    elif kind == "HELDOUT":  # eligible-looking, after everything, but held out
        e = dict(base, **nxt, split="heldout", source="instrument heldout line")
    elif kind == "MISSING":  # no target slot
        e = dict(base, **nxt, source="instrument missing line")
        del e["target"]
    elif kind == "CTXREAD":  # a well-formed context read: no producer, no reconstruct_state test
        e = {
            "kind": "ctxread",
            **nxt,
            "stream": "chat",
            "schema": "r2.v1",
            "valid": "1",
            "split": "train",
            "episode_id": "1",
            "turn_index": "0",
            "grown_version": "0",
            "encoded_ranks_digest": "0" * 64,
            "encoder": "text_features n_in=52",
        }
    elif kind == "SLEEPDIGEST":  # a well-formed sleep tick that no re-execution can reproduce
        sleeps = [ev for ev in events if ev["kind"] == "sleep"]
        e = dict(sleeps[-1], **nxt, tick=str(int(sleeps[-1]["tick"]) + 1), weights_digest="0" * 64)
    else:
        raise ValueError(f"unknown planted kind {kind}")
    if not save_episode(e):
        raise OSError("instrument: planted record was not appended")
    return e["event_id"]


_HOME_MARK = ".replay-hydration-owner"
_throwaway_home = None  # bound only after validating the instrument-created root and its token
_throwaway_token = None


def _create_private_home(parent, prefix="arm-"):
    """Parent launch: allocate a fresh root and its independent per-launch nonce."""
    parent = Path(parent).absolute()
    _refuse_production(parent)
    if parent.resolve() != parent:
        raise ValueError("instrument: private home may not alias another directory")
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    token = secrets.token_hex(32)
    with (path / _HOME_MARK).open("x", encoding="ascii") as mark:
        json.dump({"root": str(path), "token": token}, mark)
    return path, token


def _owned_private_home(path, token):
    """Validate the out-of-band token and root binding before any lake IO."""
    if not path or not token:
        raise ValueError("instrument: requires its marked private home")
    path = Path(path).absolute()
    _refuse_production(path)
    if path.resolve() != path:
        raise ValueError("instrument: private home may not alias another directory")
    mark = path / _HOME_MARK
    try:
        info = mark.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("instrument: invalid private home ownership mark")
        owned = json.loads(mark.read_text(encoding="ascii"))
    except (OSError, ValueError) as exc:
        raise ValueError("instrument: missing or invalid private home ownership mark") from exc
    if owned != {"root": str(path), "token": token}:
        raise ValueError("instrument: private home ownership mismatch")
    return path


def _private_lake():
    """An exact, unaliased, single-link lake in the still-owned active home."""
    from snn_data_lake.substrate import home

    root = _owned_private_home(_throwaway_home, _throwaway_token)
    if Path(os.environ.get("NORD_HOME", "")).absolute() != root:
        raise ValueError("instrument: bout plant requires its bound private home")
    path = Path(home.knowledge_path()).absolute()
    if path != root / "brain_lake.bin" or path.resolve() != path:
        raise ValueError("instrument: lake is outside its exact private home path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("instrument: private home lake must have one regular-file link")
    return path, info


def _plant_bout(kind):
    """R2-c8 (b): the lake's own `sleep_bout` frame is replaced by itself with ONE slot altered (a
    digest or a progress slot), so redo's verification is the only thing that can tell."""
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import _LEN, unpack_episode

    path, before = _private_lake()
    home.assert_not_live_home(_throwaway_home)
    raw, tag, found = _raw(), home._T_EPISODE, None
    for a, b in _frames(raw)[0]:
        payload = raw[a + _LEN.size : b]
        if payload[: len(tag)] == tag:
            ep = unpack_episode(payload[len(tag) :])
            if ep.get("kind") == "sleep_bout":
                found = (a, ep)
    if found is None:
        raise ValueError("instrument: no sleep_bout frame to alter")
    start, ep = found
    if kind == "BOUTDIGEST":
        ep["weights_digest"] = "0" * 64
    else:
        ep["presentations_spent"] = str(int(ep["presentations_spent"]) + 1)
    checked, info = _private_lake()  # re-check after scanning and before destructive access
    if checked != path or (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
        raise ValueError("instrument: private home lake changed during the bout scan")
    with open(path, "r+b") as stream:
        opened = os.fstat(stream.fileno())
        if opened.st_nlink != 1 or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("instrument: private home lake changed before destructive access")
        stream.truncate(start)  # nothing has opened this private home's log yet
    if not home.save_episode(ep):
        raise OSError("instrument: altered bout record was not appended")
    return ep["event_id"]


def _neuter(kind):
    """The guard-OFF arm of a planted fault: the reader or the brain admits what its guard refuses."""
    import snn_data_lake.substrate.learner_events as le

    if kind in ("DUP", "REORDER"):
        le._identity_refusal = lambda event, seen, last: None
    elif kind in ("HELDOUT", "MISSING"):
        real = le._refusal

        def faulted(event, spec, allow_instrument):
            if kind == "MISSING" and "target" not in event:
                event["target"] = "GREET"  # the reader pads the missing label (the fault)
            if kind == "HELDOUT" and event.get("split") == "heldout":
                event["split"] = "train"  # the split guard neutered
            return real(event, spec, allow_instrument)

        le._refusal = faulted
    elif kind in (
        "SLEEPDIGEST",
        "BOUTDIGEST",
        "BOUTPROGRESS",
    ):  # redo verifies nothing: `hydrate` reads these at call time
        le.SLEEP_DIGESTS = ()
        le.SLEEP_PROGRESS = ()
    elif kind == "CTXREAD":
        _filtered_hydrate(lambda learner, e: e["kind"] == "ctxread")


def _filtered_hydrate(drop):
    """Patch `hydrate` so it never sees the events `drop(learner, event)` selects (receipt kept
    consistent), i.e. a recovery that silently skips them."""
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner as Learner

    real = Learner.hydrate

    def hydrate(learner, events, *, receipt, **kw):
        kept = [e for e in events if not drop(learner, e)]
        proof = json.loads(receipt)
        proof["eligible_ids"] = [e["event_id"] for e in kept]
        new = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return real(learner, kept, receipt=new, **kw)

    Learner.hydrate = hydrate


class RedoMismatch(ValueError):
    """Instrument-only failure raised at the learner's digest/progress comparison."""


def _attributed_hydrate(real, probe):
    """Bind the two actual comparison raises to a type, without editing the live learner.

    Match syntax, never exception text. Refuse changed comparison/call structure rather than
    silently losing attribution. All other raises and callbacks keep their original types.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(real)))
    comparison = ast.dump(
        ast.parse("any(actual[k] != event[k] for k in SLEEP_DIGESTS + SLEEP_PROGRESS)")
        .body[0]
        .value
    )
    step = ast.dump(ast.parse('sleep_step["step"]').body[0].value)
    comparisons = calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.dump(node.test) == comparison:
            (failure,) = node.body
            if not isinstance(failure, ast.Raise) or not isinstance(failure.exc, ast.Call):
                raise RuntimeError("instrument: redo comparison structure changed")
            failure.exc.func = ast.Name(id="_RedoMismatch", ctx=ast.Load())
            comparisons += 1
        if isinstance(node, ast.Call) and (
            ast.dump(node.func) == step
            or isinstance(node.func, ast.Name)
            and node.func.id == "learner_bout"
        ):
            node.args.insert(0, node.func)
            node.func = ast.Name(id="_redo_call", ctx=ast.Load())
            calls += 1
    if (comparisons, calls) != (2, 2):
        raise RuntimeError("instrument: redo comparison/call sites changed")

    def completed(call, *args, **kw):
        probe["redo_attempts"] += 1
        result = call(*args, **kw)
        probe["redo_steps"] += 1
        return result

    namespace = dict(real.__globals__, _RedoMismatch=RedoMismatch, _redo_call=completed)
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, real.__code__.co_firstlineno - 1)
    exec(compile(tree, real.__code__.co_filename, "exec"), namespace)
    return namespace[real.__name__]


def _patch_restore(role):
    """Omission arms on the restoring side (2.4): each removes ONE carrier from the restore."""
    from snn_data_lake.readout import interaction_learning as ixn

    Learner = ixn.SpikingInteractionLearner
    real_load, real_hydrate = Learner.load_state_dict, Learner.hydrate
    omission = role in _RED_TWIN or role == "O_rngv"
    probe = {
        "binding_valid": False,
        "redo_attempts": 0,
        "redo_steps": 0,
        "foreign_exception": None,
        "mismatch_verified": False,
    }
    if omission:
        real_hydrate = _attributed_hydrate(real_hydrate, probe)
        Learner.hydrate = real_hydrate

    if role in ("O_rng", "O_rngv", "O_w", "O_stack"):

        def omitted(learner, state):
            state = copy.deepcopy(state)
            attrs, fresh = state["tree"]["attrs"], learner.state_dict()["tree"]["attrs"]
            if role == "O_w":
                for neuron, cold in zip(attrs["neurons"], fresh["neurons"], strict=True):
                    neuron["attrs"]["w"] = cold["attrs"]["w"]
            else:
                carrier = {"O_rng": "rng", "O_rngv": "rng", "O_stack": "stack"}[role]
                attrs[carrier] = fresh[carrier]
            return real_load(learner, state)

        Learner.load_state_dict = omitted
    elif role == "O_hyd":

        def no_hydration(learner, events, *, receipt, sleep_step, **kw):
            from snn_data_lake.substrate.learner_events import EMPTY_PREFIX_DIGEST

            learner._hydrated = True  # ring left empty, redo off, the clean report returned
            learner._hydration_count += 1
            return {
                "complete": True,
                "processed_ids": [],
                "eligible_ids": [],
                "cursor": None,
                "redone": 0,
                "rewirer": sleep_step["rewirer"],
                "observations": None,
                "receipt_sha": "",
                "prefix_digest": EMPTY_PREFIX_DIGEST,
            }

        Learner.hydrate = no_hydration
    elif role == "O_wm":

        def no_watermark(learner, events, **kw):
            learner._cursor = None  # the watermark dropped: every event redone on top of the state
            return real_hydrate(learner, events, **{**kw, "checkpoint_prefix_digest": None})

        Learner.hydrate = no_watermark
    elif role == "O_cursor":

        def wake_cursor(learner, events, **kw):
            from snn_data_lake.substrate.learner_events import prefix_digest

            # recovery ignores the cursor's sleep component and resumes from the last wake event
            cur = learner._cursor
            wake = [
                ixn._event_order(e["event_id"])
                for e in events
                if e["kind"] in ("teach", "turn")
                and (cur is None or ixn._event_order(e["event_id"]) <= cur)
            ]
            learner._cursor = max(wake) if wake else None
            binding = prefix_digest(events, learner._cursor)
            probe.update(saved_cursor=cur, mutated_cursor=learner._cursor, binding_valid=True)
            return real_hydrate(learner, events, **{**kw, "checkpoint_prefix_digest": binding})

        Learner.hydrate = wake_cursor
    elif role == "O_sleepskip":
        _filtered_hydrate(
            lambda learner, e: (
                e["kind"] == "sleep"
                and (learner._cursor is None or ixn._event_order(e["event_id"]) > learner._cursor)
            )
        )
    elif role == "O_bout":
        _filtered_hydrate(
            lambda learner, e: (
                e["kind"] == "sleep_bout"
                and (learner._cursor is None or ixn._event_order(e["event_id"]) > learner._cursor)
            )
        )
    elif role == "MOTOR":
        from snn_data_lake.brainstem.so101_arm import So101Arm
        from snn_data_lake.engine.brain import Brain

        real_boot = Brain._r2_boot

        def reaching(brain):
            real_boot(brain)
            try:  # counted by _MotorCount's wrapper before the dummy arm refuses it
                So101Arm.apply(object(), "HOME")
            except Exception:  # noqa: BLE001 - a dummy arm; the count is the red
                pass

        Brain._r2_boot = reaching
    elif role == "O_rw":
        from snn_data_lake.sleep.structural_rewiring import StructuralRewiring

        StructuralRewiring.load_state_dict = lambda self, d: None  # rewirer left fresh

    if omission:
        patched = Learner.hydrate

        def observed(learner, events, **kw):
            try:
                return patched(learner, events, **kw)
            except Exception as exc:
                if isinstance(exc, RedoMismatch) and probe["redo_steps"] > 0:
                    probe["mismatch_verified"] = True
                else:
                    probe["foreign_exception"] = f"{type(exc).__name__}: {exc}"
                raise

        Learner.hydrate = observed
        return probe


def _flip_one_weight(brain):
    """The O_rw construction (2.4): drive one live weight through zero identically in the twin and the
    restoring processes after the boundary, so the structural pass's sign identity is exercised."""
    w = brain.brain._ixn.neurons[0].w
    w[0] = -w[0] if w[0] != 0.0 else -1e-3


def _explicit_replay(brain, k=8):
    """The driven schedule's ONE explicit `replay(k)` step (integrator #2738/#2739): sleep replay is
    retired, so the learner RNG is drawn only here. Identical in the restoring process and its twin;
    it writes no lake event, so O_rng tests the restored RNG carrier, not a durable effect."""
    lrn = brain.brain._ixn
    lrn._effect_depth += 1
    try:
        lrn.replay(k=k)
    finally:
        lrn._effect_depth -= 1


# ---- one process ------------------------------------------------------------------------------


def _refuse_production(home):
    live = (Path.home() / ".snn_data_lake").resolve()
    root = Path(home).resolve()
    if root.is_relative_to(live) or live.is_relative_to(root):
        raise ValueError("instrument: private home overlaps the production home")


def _reader_view():
    """What the strict same-pass reader makes of this process's lake BEFORE the brain boots."""
    try:
        from snn_data_lake.substrate.learner_events import KIND_SPECS, scan_learner_events

        events, refusals, _, receipt = scan_learner_events(
            tuple(KIND_SPECS), strict=True, return_receipt=True
        )
        return {
            "refusals": refusals,
            "eligible": len(events),
            "complete": json.loads(receipt)["complete"],
        }
    except Exception as exc:  # noqa: BLE001 - reported, never raised: it is evidence
        return {"error": f"{type(exc).__name__}: {exc}"}


def _drive(brain, role, seed, result=None, out=None):
    """The pre/twin schedule for `role`; returns after the kill point (or the twin's equivalent)."""
    kp = role.split("_", 1)[1]
    order = _schedule(brain, seed, replay_first=kp in ("rw", "off"))
    if kp == "K0" and role.startswith("pre_"):
        brain.close()
    if kp in ("K2", "K3"):
        _post_boundary(brain, order)
    if kp == "K3":
        _tick_sleeps(brain, J_NIGHT)
    if kp in BOUT_CRASH_POINTS:
        _bout(brain, seed, result, out, point=kp)
        if kp == "Bpost":  # the checkpoint that covers the bout, then the kill
            result["checkpoint_ok"] = bool(brain._flush_interaction_learner())
    if kp == "off":
        brain.neuter_sleep("structural")
    if kp in ("rw", "off", "rwv") and not role.startswith("pre_"):
        _flip_one_weight(brain)
        _explicit_replay(brain)


def run_process(role, seed, home, out, head="r2", *, nonce=None):
    global _throwaway_home, _throwaway_token

    check_seeds([seed])  # `--proc` is an entry point of its own, not only run_seed's child
    _throwaway_home, _throwaway_token = None, None
    private = _owned_private_home(home, nonce)
    _throwaway_home, _throwaway_token = private, nonce
    os.environ["NORD_HOME"] = str(private)
    if role.startswith("fw_"):
        return run_failwrite(role, seed, out)
    result = {"role": role, "seed": seed, "head": head}
    motor = _MotorCount()
    restoring = not (role.startswith("pre_") or role.startswith("twin_"))
    if head == "r2":
        probe = _patch_restore(role)
        if probe is not None:
            result["omission_probe"] = probe
        if role == "O_cursor":
            result["cursor_probe"] = probe
        for kind in PLANTED:
            if role.startswith(kind + "_"):
                result["planted"] = _plant(kind)
                if role.endswith("_off"):
                    _neuter(kind)
        if restoring:
            result["reader"] = _reader_view()
    from snn_data_lake.engine.brain import Brain, make_brain

    k4_sample = {}
    if role == "pre_K4":  # the first flush is the night's `_enter_wake`: kill just before it

        def kill_before_checkpoint(self):
            _write(out, result)
            os._exit(0)

        Brain._flush_interaction_learner = kill_before_checkpoint
    elif role == "twin_K4":
        real_flush = Brain._flush_interaction_learner

        def sample_before_checkpoint(self):
            # A10: the K1 twin runs one more wake percept after this hook. Sample K4's
            # exact cut, then let the never-killed twin flush and continue normally.
            first = not k4_sample
            if first:
                k4_sample["before"] = _before(self)
            flushed = real_flush(self)
            if first:
                k4_sample["checkpoint_ok"] = bool(flushed)
            return flushed

        Brain._flush_interaction_learner = sample_before_checkpoint
    brain = make_brain(seed=seed, persist=True)
    result["construct_motor"] = motor.n
    result["construct_chrono"] = _chrono(brain)
    result["stack_present"] = getattr(brain.brain._ixn, "stack", None) is not None
    result["fault"] = getattr(brain, "_r2_fault", None)
    result["redo"] = getattr(brain.brain._ixn, "_redo_count", None)
    if not restoring:
        _drive(brain, role, seed, result, out)
        if role.startswith("pre_"):
            _write(out, result)
            os._exit(0)
    if role == "CHRONO":
        brain.tick()  # the red: "hydration" ticks the brain once
        result["construct_chrono"] = _chrono(brain)
    if role in ("res_rw", "O_rw", "O_rng", "O_rngv"):
        _flip_one_weight(brain)
        _explicit_replay(brain)
    result["before"] = _before(brain)  # BEFORE the wrappers exist: the walk sees the live learner
    if role == "twin_K4":
        result["post_boundary_cursor"] = result["before"]["cursor"]
        result["before"] = k4_sample["before"]
        result["k4_checkpoint_ok"] = k4_sample["checkpoint_ok"]
    windows = []
    draws = _watch(brain, motor, windows)
    result["rows"] = _sleep_rows(brain, draws)
    result["replay_windows"] = windows
    result["motor_total"] = motor.n
    _write(out, result)
    os._exit(0)


def _write(out, result):
    Path(out).write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")


# ---- FAILWRITE (A3, owner #1037): the latch after an uncertain failed write ----------------------

FW_K1 = ("hello private r2", "GREET")  # A: a successful teach, then the checkpoint
FW_K2 = ("set a private goal", "GOAL")  # B: the teach (or staged bout) whose append fails
FW_CALLS = ("i", "ii", "iii", "iv", "v", "vi", "vii")
_FW_RED_OF = {
    "fw_red_kind_t": "KIND",
    "fw_red_kind_b": "KIND",
    "fw_red_retry": "RETRY",
    "fw_red_count": "COUNT",
    "fw_red_ckpt": "CKPT",
    "fw_red_posed": "POSED",
}


def _sem(brain):
    """main's `_semantic`: the full learner snapshot with exactly `_redo_count` and
    `_unjournaled_refused` popped (each asserted on its own)."""
    snap = brain._r2_snapshot(brain.brain._ixn)
    attrs = snap["learner"]["state"]["tree"]["attrs"]
    for name in ("_redo_count", "_unjournaled_refused"):
        attrs.pop(name)
    return json.loads(json.dumps(snap, sort_keys=True, allow_nan=False))


def _digest(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _posed_value(ixn):
    """A `bind` source that passes `posed_source` and, with the latch open, is appended."""
    labels = list(ixn.responses)
    factory = {
        "problem_id": "ii.xor",
        "labels": labels,
        "seed": 0,
        "arm": "RSTDP_BRAIN",
        "population_origin": "constructed",
        "units": [],
        "birth_rows": [],
        "w_feed": [[] for _ in range(256)],
        "t_post_g": 15.0,
        "source_signs": dict.fromkeys(labels, 1),
        "attachments": [],
        "out_eligible": [],
    }
    return {"op": "bind", "problem_id": "ii.xor", "factory": factory}


def _fw_a(brain, result):
    """A: a successful `teach` of K1, then the `_enter_wake` checkpoint."""
    if brain._r2_fault is not None:
        raise RuntimeError(f"instrument: the fresh brain booted latched: {brain._r2_fault}")
    brain.brain.teach(*FW_K1)
    brain._enter_wake()
    raw = _raw()
    cp = _raw_json(raw, "interaction_learner.json")
    if cp is None:
        raise RuntimeError("instrument: A's checkpoint was not written")
    sem = _sem(brain)
    ixn = brain.brain._ixn
    result["a"] = {
        "seq": brain._r2_seq,
        "hash": _digest(sem),
        "redo": ixn._redo_count,
        "refused": ixn._unjournaled_refused,
        "raw": _raw_rec(raw),
        "cp": _digest(cp),
    }
    return sem


def _fw_reference(role, seed, result):
    """A never-faulted persisting process on its own home. AB references also read n_i: the number
    of `turn` events the same `respond` call appends in a never-faulted brain."""
    from snn_data_lake.engine.brain import make_brain

    brain = make_brain(seed=seed, persist=True)
    _fw_a(brain, result)
    if role == "fw_ref_AB":
        brain.brain.teach(*FW_K2)
    elif role == "fw_ref_ABb":
        brain._r2_run_bout(STAGES, rng_seed=seed)
    result["sem"] = _sem(brain)
    result["seq"] = brain._r2_seq
    result["fault"] = brain._r2_fault
    result["cursor"] = repr(brain.brain._ixn._cursor)
    if role != "fw_ref_A":
        before = _raw()
        brain.respond(FW_K2[0], human=False)
        kinds = _episode_kinds(_raw(), len(before))
        result["n_i"] = kinds.count("turn")
        result["respond_kinds"] = kinds


def _fw_injector(arm, brain, seen):
    """The fault at `substrate.home.save_episode`, the writer's one append. absent: nothing written;
    landed: the complete frame is written, then the call raises; torn: a header and a payload one
    byte short, then False. It stays installed through the first C pass."""
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import _LEN, pack_episode

    real = home.save_episode

    def faulted(event, path=None):
        seen.append(
            {
                "kind": event.get("kind"),
                "id": event.get("event_id"),
                "pre": _sem(brain) if arm == "landed" and not seen else None,
            }
        )
        if arm == "landed":
            if not real(event, path=path):
                raise OSError("instrument: the landed frame was a duplicate")
            raise OSError("instrument: landed append failure")
        if arm == "torn":
            payload = home._T_EPISODE + pack_episode(event)
            with open(home.knowledge_path(), "ab") as stream:
                stream.write(_LEN.pack(len(payload)) + payload[:-1])
                stream.flush()
                os.fsync(stream.fileno())
        return False

    home.save_episode = faulted
    return real


def _fw_wrap_sink(brain, make):
    """Install a wrapped `Brain._record_learner_event` and re-bind the learner's event sink to it."""
    from snn_data_lake.engine.brain import Brain

    Brain._record_learner_event = make(Brain._record_learner_event)
    brain.brain._ixn.event_sink = brain._record_learner_event


def _fw_red_after_b(brain, red, kind):
    from snn_data_lake.engine.brain import Brain

    if red == "KIND":  # the latch reopened, then only B's own kind is gated (no relatch)
        brain._r2_fault = None
        if kind == "teach":

            def make(real):
                def kind_only(self, k, key, target, stimulus):
                    if k == "teach":
                        ixn = self.brain._ixn
                        ixn._unjournaled_refused += 1
                        return ixn.respond(stimulus)
                    return real(self, k, key, target, stimulus)

                return kind_only

            _fw_wrap_sink(brain, make)
        else:

            def refuse_bout(self, *a, **kw):
                raise RuntimeError("R2 sleep_bout refused by latch until restart")

            Brain._r2_run_bout = refuse_bout
    elif red == "PARTIAL":  # while latched: a length header and an incomplete learner payload

        def make(real):
            def partial(self, k, key, target, stimulus):
                if self._r2_fault is not None and getattr(self, "_partial_armed", False):
                    from snn_data_lake.substrate import home
                    from snn_data_lake.substrate.binlog import _LEN

                    with open(home.knowledge_path(), "ab") as fh:
                        fh.write(_LEN.pack(1 << 20) + b"\x04" * 9)
                return real(self, k, key, target, stimulus)

            return partial

        _fw_wrap_sink(brain, make)
    elif red == "COUNT":  # a build that does not count the refused teach/turn calls

        def make(real):
            def uncounted(self, k, key, target, stimulus):
                if self._r2_fault is not None:
                    return self.brain._ixn.respond(stimulus)
                return real(self, k, key, target, stimulus)

            return uncounted

        _fw_wrap_sink(brain, make)


def _fw_survivor(role, seed, result, out):
    """B fails; C(i)-(vii) run in the latched survivor; the fault is removed; C repeats; kill."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    kind = "bout" if role.startswith(("fw_b_", "fw_p_b_")) or role == "fw_red_kind_b" else "teach"
    arm = role.rsplit("_", 1)[1] if role.startswith(("fw_t_", "fw_b_", "fw_p_")) else "absent"
    red = "PARTIAL" if role.startswith("fw_p_") else _FW_RED_OF.get(role)
    result.update(kind=kind, arm=arm, red=red)
    brain = make_brain(seed=seed, persist=True)
    ixn = brain.brain._ixn
    _fw_a(brain, result)
    a = result["a"]
    seen = []
    real_save = _fw_injector(arm, brain, seen)
    exc = None
    try:
        if kind == "teach":
            brain.brain.teach(*FW_K2)
        else:
            brain._r2_run_bout(STAGES, rng_seed=seed)
    except Exception as e:  # noqa: BLE001 - recorded: a bout re-raises its refusal
        exc = f"{type(e).__name__}: {e}"
    raw_b = _raw()
    after_b = _sem(brain)
    result["b"] = {
        "witnessed": len(seen),
        "phase": (brain._r2_fault or {}).get("phase"),
        "seq": brain._r2_seq,
        "sem_equal": _digest(after_b) == a["hash"],
        "refused": ixn._unjournaled_refused,
        "raw": _raw_rec(raw_b),
        "prefix": _prefix_rec(raw_b),
        "exc": exc,
    }
    spans, end = _frames(raw_b)
    result["b"]["tail"] = len(raw_b) - end
    result["precursor"] = seen[0]["pre"] if seen else None
    _fw_red_after_b(brain, red, kind)
    refused_b = ixn._unjournaled_refused
    posed = _posed_value(ixn)
    calls = {
        "i": lambda: brain.respond(FW_K2[0], human=False),
        "ii": lambda: brain.brain.teach("blocked private c", "GREET"),
        "iii": lambda: ixn.learn_turn("blocked context c", "GREET"),
        "iv": lambda: brain._r2_sleep_learning(),
        "v": lambda: brain._r2_run_bout(STAGES, rng_seed=seed),
        "vi": lambda: brain._flush_interaction_learner(),
        "vii": lambda: brain._g9_apply_source(posed),
    }
    records = []

    def one(pass_, name):
        reopen = (
            red in ("CKPT", "POSED") and pass_ == 1 and name == {"CKPT": "vi", "POSED": "vii"}[red]
        )
        saved = brain._r2_fault
        if reopen:  # the latch missing from THIS path only
            brain._r2_fault = None
        err, ret, caught = io.StringIO(), None, None
        try:
            with contextlib.redirect_stderr(err):
                ret = calls[name]()
        except Exception as e:  # noqa: BLE001 - the refusal is the observable
            caught = f"{type(e).__name__}: {e}"
        if reopen:
            brain._r2_fault = saved
        now = _raw()  # ONE snapshot: raw, prefix and checkpoint all decode from these bytes
        records.append(
            {
                "pass": pass_,
                "call": name,
                "ret": None if ret is None else str(ret)[:40],
                "exc": caught,
                "err": [ln for ln in err.getvalue().splitlines() if ln.strip()],
                "seq": brain._r2_seq,
                "raw": _raw_rec(now),
                "prefix": _prefix_rec(now),
                "cp": _digest(_raw_json(now, "interaction_learner.json")),
                "sem_equal": _digest(_sem(brain)) == a["hash"],
                "refused": ixn._unjournaled_refused,
                "redo": ixn._redo_count,
                "phase": (brain._r2_fault or {}).get("phase"),
            }
        )

    for name in FW_CALLS:
        if name == "iii" and red == "PARTIAL":
            brain._partial_armed = True  # installed at C(iii), as A3 registers, not from C(i)
        one(1, name)
    home.save_episode = real_save  # the fault is removed
    if red == "RETRY":  # the latch cleared in-process by the injector's removal
        brain._r2_fault = None
    for name in FW_CALLS:
        one(2, name)
    result["calls"] = records
    result["refused_after_b"] = refused_b
    result["posed"] = "VERIFIED"
    _write(out, result)
    os._exit(0)


def _fw_kill_point(role, seed, result, out):
    """The four #1048 crash points: after a successful event before the checkpoint; after a complete
    checkpoint frame before success is reported; after a successful checkpoint before exit."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    brain = make_brain(seed=seed, persist=True)
    if role == "fw_k_ckpt":
        real = home.save_state

        def write_then_die(name, obj):
            ok = real(name, obj)
            if name == "interaction_learner.json":
                _write(out, result)
                os._exit(0)
            return ok

        home.save_state = write_then_die
    if role == "fw_k_evt":
        if brain._r2_fault is not None:
            raise RuntimeError("instrument: booted latched")
        brain.brain.teach(*FW_K1)
    else:
        _fw_a(brain, result)
    result["killed_at"] = role
    _write(out, result)
    os._exit(0)


def _fw_restart(role, seed, result):
    """A fresh OS process on a copy of the survivor's home, no fault: record, then (teach/bout arms)
    one separately injected failure to show the latch re-arms. fw_gate_* wrap `hydrate` to report
    `complete` False; fw_red_gate also neuters `Brain._r2_may_rearm` to True (RED-GATE)."""
    from snn_data_lake.engine.brain import Brain, make_brain
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner as Learner
    from snn_data_lake.substrate import home

    gate = {"fw_gate_ctl": "ctl", "fw_red_gate": "red"}.get(role)
    if gate:
        real_hydrate = Learner.hydrate

        def incomplete(learner, events, **kw):
            return dict(real_hydrate(learner, events, **kw), complete=False)

        Learner.hydrate = incomplete
        if gate == "red":
            Brain._r2_may_rearm = staticmethod(lambda report: True)
    raw0 = _raw()
    spans, end = _frames(raw0)
    result["raw_before"] = {
        **_raw_rec(raw0),
        "prefix_len": end,
        "prefix_sha": hashlib.sha256(raw0[:end]).hexdigest(),
        "tail": len(raw0) - end,
    }
    brain = make_brain(seed=seed, persist=True)
    ixn = brain.brain._ixn
    result["fault"] = brain._r2_fault
    result["hydrations"] = ixn._hydration_count
    result["sem"] = _sem(brain)
    result["cursor"] = repr(ixn._cursor)
    result["raw_after"] = _raw_rec(_raw())
    if gate:  # the next real write: a latched build must not append
        before = _raw()
        brain.brain.teach("gate probe line", "GREET")
        result["probe_grew"] = _raw() != before
        return
    if role.startswith(("fw_rs_t_", "fw_rs_b_")):
        real = home.save_episode
        home.save_episode = lambda event, path=None: False
        try:
            brain.brain.teach("rearm probe line", "GREET")
        finally:
            home.save_episode = real
        fault = brain._r2_fault or {}
        result["rearm"] = {"latched": brain._r2_fault is not None, "phase": fault.get("phase")}


def run_failwrite(role, seed, out):
    check_seeds([seed])  # outside the try: a refusal is raised, never recorded as an INVALID arm
    result = {"role": role, "seed": seed, "head": "r2"}
    try:
        if role.startswith("fw_ref_"):
            _fw_reference(role, seed, result)
        elif role.startswith(("fw_rs_", "fw_rs2_", "fw_gate_", "fw_red_gate")):
            _fw_restart(role, seed, result)
        elif role.startswith("fw_k_"):
            _fw_kill_point(role, seed, result, out)
        else:
            _fw_survivor(role, seed, result, out)
    except BaseException:  # noqa: BLE001 - an arm that raised is INVALID, never a fire
        result = {
            "role": role,
            "seed": seed,
            "error": traceback.format_exc()[-2400:],
        }
    _write(out, result)
    os._exit(0)


# ---- comparisons and the verdict --------------------------------------------------------------


def records_equal(a, b, skip=()):
    """Exact equality on every compared record; a missing record is never equal, and a latched
    restore (`fault`) never equals a clean twin. `skip` names records left out -- only the base-head I
    arms use it (G0(c) never carried the rewirer, the ring ids, the cursor or the counts)."""
    if not a or not b or "before" not in a or "before" not in b:
        return False
    if a.get("fault") != b.get("fault"):
        return False
    before = [k for k in _COMPARED_BEFORE if k not in skip]
    row = [k for k in _COMPARED_ROW if k not in skip]
    if any(a["before"].get(k) != b["before"].get(k) for k in before):
        return False
    if len(a["rows"]) != len(b["rows"]):
        return False
    return all(ra.get(k) == rb.get(k) for ra, rb in zip(a["rows"], b["rows"]) for k in row)


def chrono_clean(r):
    """Hydration and redo tick nothing (construction leaves a fresh clock) and a sleep step advances
    no chronology and reaches no actuator (R2-c5's instrument-level half)."""
    c = r.get("construct_chrono") or {}
    windows = r.get("replay_windows") or []
    return (
        r.get("construct_motor") == 0
        and c.get("step") == 0
        and c.get("nights") == 0
        and all(w["before"] == w["after"] and w["motor"] == 0 for w in windows)
    )


def comparator_controls(record):
    """R2-c4's comparator half, on a real record: a planted 1-ulp weight change and one swapped ring
    entry must each be flagged; the unmodified copy must compare equal."""
    if not record or "rows" not in record or len(record["before"]["ring"]) < 2:
        return False
    same = records_equal(record, copy.deepcopy(record))
    ulp = copy.deepcopy(record)
    w = ulp["rows"][0]["weights"][0]
    w[0] = math.nextafter(w[0], math.inf)
    ring = copy.deepcopy(record)
    r = ring["before"]["ring"]
    r[0], r[1] = r[1], r[0]
    return same and not records_equal(record, ulp) and not records_equal(record, ring)


def _planted_entry(kind, arms):
    on, off = arms.get(f"{kind}_on", {}), arms.get(f"{kind}_off", {})
    twin, reason = arms.get(_PLANT_TWIN[kind]), PLANTED[kind]
    if not (_complete(on) and _complete(off) and _complete(twin)):
        return {"guard_on_ok": False, "guard_on_refused": False, "guard_off_flagged": False}
    if kind in _READER_KINDS:
        refusals = (on.get("reader") or {}).get("refusals") or {}
        off_refusals = (off.get("reader") or {}).get("refusals") or {}
        refused = refusals.get(reason, 0) >= 1
        flagged = off_refusals.get(reason, 0) == 0 and not records_equal(off, twin)
        if kind == "HELDOUT":  # a held-out id must be in the ring of the neutered arm (explicit)
            flagged = flagged and off.get("planted") in (off.get("before") or {}).get(
                "ring_ids", []
            )
            ok = records_equal(on, twin)  # non-critical: refused AND recovery completes
        else:
            ok = on.get("fault") is not None  # critical: the boot is refused visibly, never silent
    else:
        refused = reason in ((on.get("fault") or {}).get("cause") or "")
        ok = on.get("fault") is not None
        flagged = (
            off.get("fault") is None
            if kind in ("CTXREAD", "BOUTDIGEST", "BOUTPROGRESS")  # admitting the lie IS the red
            else off.get("fault") is None and not records_equal(off, twin)
        )
    return {"guard_on_ok": ok, "guard_on_refused": refused, "guard_off_flagged": flagged}


def _complete(rec):
    """A record counts only if its process finished: no 'error' and the measured state is there."""
    return bool(rec) and "error" not in rec and "before" in rec


def _fw_ok(rec):
    return bool(rec) and "error" not in rec


def _loc(failure):
    bar, _, where = failure.partition("@")
    return bar, (int(where[0]) if where else None), where[1:]


def fw_survivor_failures(rec, n_i, raw_key="raw"):
    """Every A3 bar of one survivor record; [] means all met. Names are `bar` or `bar@<pass><call>`.
    After every C call (the repeats included): `_r2_seq` unchanged, raw bytes equal the after-B
    bytes (`raw_key="prefix"`: the complete-frame prefix, the reading under which a torn tail's
    repair is not a change), the checkpoint equals CP_A, the snapshot equals A's, `_unjournaled_refused` equals b plus
    one per call that reaches `_record_learner_event` (C(i) n_i, C(ii) 1, C(iii) 1), `_redo_count`
    unchanged; C(iv) returns False, C(v) raises RuntimeError naming the latch, C(vi) returns False
    with ONE 'R2 checkpoint refused by latch' line, C(vii) raises PosedRefused."""
    if not _fw_ok(rec) or "calls" not in rec:
        return ["incomplete"]
    f, a, b = [], rec["a"], rec["b"]
    teach = rec["kind"] == "teach"
    if b["witnessed"] < 1:
        f.append("b_fault_not_reached")
    if b["phase"] != ("append" if teach else "bout_append_or_preview"):
        f.append("b_phase")
    if not b["sem_equal"]:
        f.append("b_state")
    if b["seq"] != a["seq"] + 1:
        f.append("b_seq")
    run = 1 if teach else 0
    if b["refused"] != run:
        f.append("b_refused")
    if len(rec["calls"]) != 2 * len(FW_CALLS):
        f.append("calls")
    for c in rec["calls"]:
        run += {"i": n_i, "ii": 1, "iii": 1}.get(c["call"], 0)
        at = f"@{c['pass']}{c['call']}"
        if c["seq"] != b["seq"]:
            f.append("seq" + at)
        if c[raw_key] != b[raw_key]:
            f.append("raw" + at)
        if c["cp"] != a["cp"]:
            f.append("cp" + at)
        if not c["sem_equal"]:
            f.append("state" + at)
        if c["refused"] != run:
            f.append("count" + at)
        if c["redo"] != a["redo"]:
            f.append("redo" + at)
        exc = c.get("exc") or ""
        if c["call"] == "iv" and c["ret"] != "False":
            f.append("ret" + at)
        if c["call"] == "v" and not ("RuntimeError" in exc and "latch" in exc):
            f.append("ret" + at)
        if c["call"] == "vi":
            if c["ret"] != "False":
                f.append("ret" + at)
            if sum("R2 checkpoint refused by latch" in ln for ln in c["err"]) != 1:
                f.append("line" + at)
        if c["call"] == "vii" and not (
            "PosedRefused" in exc and "common all-learning latch is closed" in exc
        ):
            f.append("ret" + at)
    return sorted(set(f))


def fw_red_fired(name, rec, n_i):
    """A constructed FAILWRITE red fires only when the bar it targets fails AND B's own bars held
    (the mutation is installed after B, so B is not what is being tested)."""
    failed = fw_survivor_failures(rec, n_i)
    if not failed or "incomplete" in failed or any(x.startswith("b_") for x in failed):
        return False
    hits = [_loc(x) for x in failed if "@" in x]
    if name == "KIND":  # B's own kind stays refused; the next kind gets through
        skip = "ii" if rec["kind"] == "teach" else "v"
        return any(call != skip for _, _, call in hits)
    if name == "RETRY":
        return any(p == 2 for _, p, _ in hits) and not any(p == 1 for _, p, _ in hits)
    if name == "COUNT":
        return any(bar == "count" for bar, _, _ in hits) and not any(
            bar in ("raw", "seq") for bar, _, _ in hits
        )
    if name == "CKPT":  # must fail on the True return AND the missing line (bytes may move too)
        at = {(bar, p, call) for bar, p, call in hits}
        return ("ret", 1, "vi") in at and ("line", 1, "vi") in at
    if name == "POSED":
        return any(bar in ("raw", "seq") and p == 1 and call == "vii" for bar, p, call in hits)
    if (
        name == "PARTIAL"
    ):  # the strict raw bar fires at C(iii), never before; prefix is reported apart
        raws = {call for bar, p, call in hits if bar == "raw" and p == 1}
        return "iii" in raws and not raws & {"i", "ii"}
    raise ValueError(f"unknown FAILWRITE red {name}")


def fw_oracle_fields(restart_sem, other_sem):
    """The named learner fields in which two snapshots differ (`_cursor`, the ring, its ids, the
    authority, and every other tree attribute); empty means the oracle cannot tell them apart."""
    names = set()
    ta, tb = (s["learner"]["state"]["tree"]["attrs"] for s in (restart_sem, other_sem))
    names |= {k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k)}
    ca, cb = restart_sem["learner"]["cache"], other_sem["learner"]["cache"]
    if isinstance(ca, dict) and isinstance(cb, dict) and set(ca) == {"ring", "ids", "authority"}:
        names |= {f"cache.{k}" for k in ca if ca[k] != cb.get(k)}
    elif ca != cb:
        names.add("cache")
    return names


def fw_gate_fired(control, red):
    """RED-GATE: with `hydrate` reporting complete=False, the correct build latches `reconcile` and
    its next write appends nothing; the build whose gate is forced True rearms and the write lands."""
    if not (_fw_ok(control) and _fw_ok(red)):
        return False
    latched = (control.get("fault") or {}).get("phase") == "reconcile"
    return bool(
        latched
        and control.get("probe_grew") is False
        and red.get("fault") is None
        and red.get("probe_grew") is True
    )


def fw_restart_failures(rec, ref_same, ref_other, torn=False, rearm=True):
    """Restart bars: fault None, one hydration, the learner snapshot equals the INDEPENDENT reference,
    differs from the other arm's reference (identical sides read NOT MET), the latch re-arms, and a
    torn tail is repaired to the last complete frame (an untorn lake is left byte-for-byte)."""
    if not _fw_ok(rec) or "sem" not in rec:
        return ["incomplete"]
    f = []
    if rec.get("fault") is not None:
        f.append("fault")
    if rec.get("hydrations") != 1:
        f.append("hydrations")
    if rec["sem"] != (ref_same or {}).get("sem"):
        f.append("oracle")
    if ref_other is not None:
        if ref_other.get("sem") == (ref_same or {}).get("sem"):
            f.append("oracle_sides_identical")
        if rec["sem"] == ref_other.get("sem"):
            f.append("other_arm_equal")
    if rearm:
        r = rec.get("rearm") or {}
        if not (r.get("latched") and r.get("phase") == "append"):
            f.append("rearm")
    rb, ra = rec["raw_before"], rec["raw_after"]
    if torn:
        if rb["tail"] <= 0:
            f.append("torn_tail_absent")
        if ra["len"] != rb["prefix_len"] or ra["sha"] != rb["prefix_sha"]:
            f.append("not_repaired_to_last_frame")
    elif (ra["len"], ra["sha"]) != (rb["len"], rb["sha"]) or rb["tail"] != 0:
        f.append("lake_changed")
    return f


def fw_seed(arms):
    """The FAILWRITE arm's per-seed reading (None when the arm was not run)."""
    names = [r for r, _ in R2_ROLES if r.startswith("fw_")]
    if not any(r in arms for r in names):
        return None
    incomplete = [r for r in names if not _fw_ok(arms.get(r))]
    out = {"incomplete": incomplete, "arms": {}, "reds": {}, "phases": {}}
    ref = {k: arms.get(f"fw_ref_{k}") or {} for k in ("A", "AB", "ABb")}
    n_i = {"t": ref["AB"].get("n_i"), "b": ref["ABb"].get("n_i")}
    both = {"t": ref["AB"], "b": ref["ABb"]}
    for kind in ("t", "b"):
        for arm in ("absent", "landed", "torn"):
            same = both[kind] if arm == "landed" else ref["A"]
            other = ref["A"] if arm == "landed" else both[kind]
            surv, rs = arms.get(f"fw_{kind}_{arm}"), arms.get(f"fw_rs_{kind}_{arm}")
            if _fw_ok(surv) and "calls" in surv and n_i[kind] is not None:
                out.setdefault("prefix_reading", {})[f"{kind}_{arm}"] = fw_survivor_failures(
                    surv, n_i[kind], raw_key="prefix"
                )
            if _fw_ok(surv) and "b" in surv:  # the true torn-tail size, per arm
                out.setdefault("torn_tail_bytes", {})[f"{kind}_{arm}"] = surv["b"].get("tail")
            out["arms"][f"{kind}_{arm}"] = {
                "survivor": fw_survivor_failures(surv, n_i[kind])
                if n_i[kind] is not None
                else ["n_i"],
                "restart": fw_restart_failures(rs, same, other, torn=arm == "torn"),
            }
            if _fw_ok(surv) and "calls" in surv:
                out["phases"][f"{kind}_{arm}"] = {c["call"]: c["phase"] for c in surv["calls"]}
    for point in ("evt", "ckpt", "post"):
        out["arms"][f"k_{point}"] = {
            "restart": fw_restart_failures(
                arms.get(f"fw_rs_k_{point}"), ref["A"], None, rearm=False
            )
        }
    partial = {}
    for kind in ("t", "b"):
        for arm in ("absent", "landed", "torn"):
            rec = arms.get(f"fw_p_{kind}_{arm}")
            nk = n_i[kind]
            partial[f"{kind}_{arm}"] = {
                "fired": bool(nk is not None and fw_red_fired("PARTIAL", rec or {}, nk)),
                "prefix_moves": bool(
                    nk is not None
                    and _fw_ok(rec)
                    and "calls" in rec
                    and any(
                        x.startswith("raw@")
                        for x in fw_survivor_failures(rec, nk, raw_key="prefix")
                    )
                ),
            }
    out["partial"] = partial
    out["survivor_raw_changed"] = {
        k: any(x.startswith("raw@") for x in v["survivor"])
        for k, v in out["arms"].items()
        if "survivor" in v
    }
    out["arms"]["k_rs2"] = {
        "restart": fw_restart_failures(arms.get("fw_rs2_k_post"), ref["A"], None, rearm=False)
    }
    fired = lambda role, name, kind: (  # noqa: E731
        n_i[kind] is not None and fw_red_fired(name, arms.get(role) or {}, n_i[kind])
    )
    out["reds"] = {
        "RED-KIND-teach": fired("fw_red_kind_t", "KIND", "t"),
        "RED-KIND-bout": fired("fw_red_kind_b", "KIND", "b"),
        "RED-RETRY": fired("fw_red_retry", "RETRY", "t"),
        "RED-COUNT": fired("fw_red_count", "COUNT", "t"),
        "RED-CKPT": fired("fw_red_ckpt", "CKPT", "t"),
        "RED-POSED": fired("fw_red_posed", "POSED", "t"),
        "RED-GATE": fw_gate_fired(arms.get("fw_gate_ctl"), arms.get("fw_red_gate")),
    }
    out["reds"]["RED-PARTIAL"] = all(v["fired"] for v in partial.values())
    landed, rs = arms.get("fw_rs_t_landed") or {}, arms.get("fw_rs_t_landed")
    pre = (arms.get("fw_t_landed") or {}).get("precursor")
    have = _fw_ok(rs) and "sem" in landed and "sem" in ref["A"] and pre is not None
    named = lambda other: {  # noqa: E731
        k
        for k in fw_oracle_fields(landed["sem"], other)
        if k in ("_cursor", "cache.ring", "cache.ids", "cache.authority")
    }
    out["reds"]["RED-ORACLE"] = bool(have and named(ref["A"]["sem"]) and named(pre))
    posed = [
        c
        for r in names
        if _fw_ok(arms.get(r))
        for c in arms[r].get("calls", [])
        if c["call"] == "vii"
    ]
    out["posed"] = (
        "VERIFIED"
        if posed and all("PosedRefused" in (c["exc"] or "") for c in posed[:1])
        else "UNVERIFIED"
    )
    bars_ok = all(not v for a in out["arms"].values() for v in a.values())
    out["ok"] = not incomplete and bars_ok and out["posed"] == "VERIFIED"
    out["bout_ok"] = not incomplete and all(
        not out["arms"][f"b_{arm}"][k]
        for arm in ("absent", "landed", "torn")
        for k in ("survivor", "restart")
    )
    return out


def c8_seed(arms):
    """R2-c8 (sleep_bout, owner #2106): (a) restart equals the never-killed twin at each of the three
    crash points; (b) a planted bout with one altered digest/progress slot is refused by name and
    latched, its red a redo that skips the verification; (c) exactly one `sleep_bout` frame, nothing
    else written, ring/authority/replay counts unchanged; (d) the FAILWRITE arm with B = sleep_bout."""
    twin = arms.get("twin_B")
    a = {pt: records_equal(arms.get(f"res_{pt}"), twin) for pt in BOUT_CRASH_POINTS}
    planted = {k: _planted_entry(k, arms) for k in ("BOUTDIGEST", "BOUTPROGRESS")}
    b = all(all(v.values()) for v in planted.values())

    def one(rec, applied):
        chk = (rec or {}).get("bout_check") or {}
        return bool(
            chk
            and chk.get("frames_added") == 1
            and chk.get("kinds_added") == ["sleep_bout"]
            and chk.get("prefix_preserved") is True
            and chk.get("cache_equal") is True
            and chk.get("replay_counts_equal") is True
            and chk.get("effect_applied") is applied
            and "error" not in rec
        )

    c = (
        one(twin, True)
        and one(arms.get("pre_B"), True)
        and one(arms.get("pre_Bpost"), True)
        and one(arms.get("pre_Bpre"), False)
        and (arms.get("pre_Bpost") or {}).get("checkpoint_ok") is True
    )
    fw = fw_seed(arms)
    return {"a": all(a.values()), "b": b, "c": c, "d": bool(fw and fw["bout_ok"])}


def _c4_moves(off, rw):
    """R2-c4's must-differ control (amendment A2): the structural-OFF twin's post-night WEIGHT rows
    differ from the structural-ON twin's, with the stack present. A rewirer-only or None-vs-object
    difference never passes; equal rows or a missing stack read NOT MET (ruling 9)."""
    if not (_complete(off) and _complete(rw)):
        return False
    if not (off.get("stack_present") and rw.get("stack_present")):
        return False
    pairs = list(zip(off["rows"], rw["rows"], strict=False))
    return bool(pairs) and any(a.get("weights") != b.get("weights") for a, b in pairs)


def seed_comparisons(arms, base=None):
    """Every per-seed reading of 2.3, from the R2-head arms (and the base-head arms when given). A red
    FIRES only if its arm and its twin are COMPLETE records; an incomplete arm makes the seed
    incomplete (never a fire)."""
    eq, get = records_equal, arms.get
    required = [r for r, _ in R2_ROLES if not r.startswith(("fw_", "pre_"))]
    incomplete = [r for r in required if not _complete(get(r))]
    incomplete += [
        r for r, _ in R2_ROLES if r.startswith("pre_") and (not get(r) or "error" in get(r))
    ]
    fired = lambda r, twin: (  # noqa: E731
        _complete(get(r)) and _complete(get(twin)) and not eq(get(r), get(twin))
    )
    c = {
        "c1": {
            "K0": eq(get("res_K0"), get("twin_K1")),
            "K1": eq(get("res_K1"), get("twin_K1")),
            "K2": eq(get("res_K2"), get("twin_K2")),
            "rw": eq(get("res_rw"), get("twin_rw")),
        },
        "c6": {
            "K3": eq(get("res_K3"), get("twin_K3")),
            "K4": eq(get("res_K4"), get("twin_K4")),
            "B": eq(get("res_B"), get("twin_B")),
        },
        "chrono": {
            r: chrono_clean(get(r, {}))
            for r in ("res_K0", "res_K1", "res_K2", "res_K3", "res_K4", "res_B", "res_rw")
        },
        "reds": {r: fired(r, twin) for r, twin in _RED_TWIN.items()},
        "planted": {kind: _planted_entry(kind, arms) for kind in PLANTED},
        "c3_controls": {
            "O_rng_lone_replay_equal": _complete(get("O_rngv"))
            and eq(get("O_rngv"), get("twin_rwv"))
        },
        "c4_structural_moves": _c4_moves(get("twin_off"), get("twin_rw")),
        "c4_rewirer_reported": any(
            a.get("rewirer") != b.get("rewirer")
            for a, b in zip(
                (get("twin_off") or {}).get("rows", []),
                (get("twin_rw") or {}).get("rows", []),
                strict=False,
            )
        ),
        "c4_comparator": comparator_controls(get("twin_K2")),
        "c8": c8_seed(arms),
    }
    for role in (*_RED_TWIN, "O_rngv"):
        arm = get(role) or {}
        probe = arm.get("omission_probe") or arm.get("cursor_probe") or {}
        attributed = probe.get("mismatch_verified") is True and probe.get("redo_steps", 0) > 0
        foreign = bool(probe.get("foreign_exception")) or (
            arm.get("fault") is not None and not attributed
        )
        if foreign and role not in incomplete:
            incomplete.append(role)
        if role in c["reds"]:
            c["reds"][role] &= not foreign
        if role == "O_cursor":
            c["reds"][role] &= probe.get("binding_valid") is True and probe.get("redo_steps", 0) > 0
    c["reds"]["CHRONO"] = _complete(get("CHRONO")) and not chrono_clean(get("CHRONO"))
    c["reds"]["MOTOR"] = _complete(get("MOTOR")) and not chrono_clean(get("MOTOR"))
    fw = fw_seed(arms)
    c["failwrite"] = fw
    if fw is not None:
        incomplete += fw["incomplete"]
        c["reds"].update({f"FW:{k}": v for k, v in fw["reds"].items()})
    if base is not None:
        need = ("twin_K1", "twin_K2", "res_K1", "res_K2")
        incomplete += [f"base:{r}" for r in need if not _complete(base.get(r))]
        whole = all(_complete(base.get(r)) for r in need)
        c["c2"] = {
            "I_K1_equal": whole and eq(base.get("res_K1"), base.get("twin_K1"), _BASE_SKIP),
            "I_K2_differs": whole and not eq(base.get("res_K2"), base.get("twin_K2"), _BASE_SKIP),
        }
    c["incomplete"] = incomplete
    c["arms_complete"] = not incomplete
    return c


def codex14_status():
    """Codex-14's gate `memory_unit_unreconstructible` is ABSENT on main: reported UNVERIFIED, never
    pass. A source that names it is reported as present-but-unexercised, still not a pass."""
    root, name = Path(__file__).resolve().parents[1], "memory_unit_unreconstructible"
    named = [
        p
        for p in (
            "engine/brain.py",
            "readout/interaction_learning.py",
            "substrate/learner_events.py",
        )
        if name in (root / p).read_text(encoding="utf-8", errors="replace")
    ]
    return {name: "UNVERIFIED", "named_in": named, "ctxread_producer": "absent on main"}


def verdict(per_seed, seeds):
    """2.3 + 2.11 D4 + #2797 over the seeds: c1 on EVERY valid seed, c2 load-bearing (an I_K2 that
    equals its twin makes that seed INVALID; R2-c3 still counts it and reads NOT MET), R2-c3 = EVERY red fires on EVERY counted
    seed (a red that fires on no seed halts the arc), and a seed with an INCOMPLETE arm record is
    INVALID for c2; an INVALID seed (any reason) is never excluded from R2-c3's count: it reads NOT MET. Controls and c6 on
    every seed. Equality has no variance: no statistic, no majority. A must-differ bar whose two
    sides are identical reads NOT MET (ruling 9). Scoring a seed reads it, so the licence holds here."""
    check_seeds(seeds)
    incomplete = [s for s in seeds if not per_seed[s].get("arms_complete", True)]
    invalid = [
        s for s in seeds if s in incomplete or not per_seed[s].get("c2", {}).get("I_K2_differs")
    ]
    valid = [s for s in seeds if s not in invalid]
    counted = list(seeds)  # R2-c3 counts EVERY seed: an INVALID one is never dropped or replaced
    every = lambda f, over=None: (
        bool(valid if over is None else over)
        and all(  # noqa: E731
            f(per_seed[s]) for s in (valid if over is None else over)
        )
    )
    c1 = every(lambda c: all(c["c1"].values()) and all(c["chrono"].values()))
    c2 = every(lambda c: c["c2"]["I_K1_equal"])
    first = per_seed[next(iter(per_seed))]
    reds = {r: every(lambda c, r=r: c["reds"].get(r) is True, counted) for r in first["reds"]}
    fired_somewhere = {
        r: any(per_seed[s]["reds"].get(r) is True for s in counted) for r in first["reds"]
    }
    planted = {
        k: every(
            lambda c, k=k: (
                c["planted"][k]["guard_off_flagged"]
                and c["planted"][k]["guard_on_ok"]
                and c["planted"][k]["guard_on_refused"]
            ),
            counted,
        )
        for k in first["planted"]
    }
    controls = every(lambda c: all(c.get("c3_controls", {"x": True}).values()), counted)
    c3 = all(reds.values()) and all(planted.values()) and controls and not invalid
    c4 = every(lambda c: c["c4_structural_moves"] and c["c4_comparator"])
    c6 = every(lambda c: all(c["c6"].values()))
    failwrite = every(lambda c: bool(c.get("failwrite")) and c["failwrite"]["ok"], counted)
    c8 = every(lambda c: all(c["c8"].values()))
    never = [r for r, hit in fired_somewhere.items() if not hit]
    halted = len(invalid) > 2 or bool(never) or not counted
    ok = c1 and c2 and c3 and c4 and c6 and failwrite and c8 and not halted and not invalid
    verdict_of = lambda b: "PASS" if b and not halted else "RED"  # noqa: E731
    return {
        "seeds": list(seeds),
        "invalid_seeds": invalid,
        "incomplete_seeds": incomplete,
        "halted": halted,
        "red_fires_on_no_seed": never,
        "R2-c1": verdict_of(c1),
        "R2-c2": verdict_of(c2),
        "R2-c3": "NOT MET" if invalid else verdict_of(c3),
        "reds": reds,
        "planted": planted,
        "R2-c4": verdict_of(c4),
        "R2-c6": verdict_of(c6),
        "R2-c8": verdict_of(c8),
        "FAILWRITE": verdict_of(failwrite),
        "verdict": "PASS" if ok else ("NOT MET" if invalid else "RED"),
        "codex14": codex14_status(),
        "not_built": {"P3": "another lane (claude-sonnet-r2p3-1015): K1g / R2-c7, kind percept"},
        "scope": "same-code restarts at K0-K4 and the staged bout; the interaction learner and the "
        "sleep carriers that touch it; the #1037 latch (FAILWRITE); R2-c5 is pytest; I arms at base "
        "e1b3ae3c compare only to their own twins; sleep replay is retired on main (owner1039)",
    }


# ---- orchestration ----------------------------------------------------------------------------


def run_seed(seed, out_dir, head="r2", src=None, roles=None):
    """All of one seed's processes for one head, each on its own home; restoring arms get a copy of
    their parent's home. Returns {role: record}. `roles` selects a subset (parents included)."""
    check_seeds([seed])
    here = Path(__file__).resolve()
    src = Path(src) if src else here.parents[2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = R2_ROLES if head == "r2" else BASE_ROLES
    arms, homes = {}, {}
    with tempfile.TemporaryDirectory(prefix=f"r2h-{head}-{seed}-", dir=out_dir) as scratch:
        scratch = Path(scratch)
        for role, parent in table:
            if roles is not None and role not in roles:
                continue
            home, nonce = _create_private_home(scratch, prefix=f"{role}-")
            homes[role] = home, nonce
            if parent is not None:
                source = _owned_private_home(*homes[parent])
                shutil.copytree(
                    source, home, dirs_exist_ok=True, ignore=shutil.ignore_patterns(_HOME_MARK)
                )
            out = out_dir / f"{head}-{seed}-{role}.json"
            out.unlink(missing_ok=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(here),
                    "--proc",
                    role,
                    str(seed),
                    str(home),
                    str(out),
                    head,
                    nonce,
                ],
                env=dict(
                    os.environ,
                    NORD_HOME=str(home),
                    PYTHONPATH=str(src),
                    PYTHONUTF8="1",
                ),
                capture_output=True,
                text=True,
                timeout=900,
            )
            if proc.returncode != 0:
                arms[role] = {"role": role, "error": f"exit {proc.returncode}"}
            elif not out.exists():
                arms[role] = {"role": role, "error": "no output"}
            else:
                try:
                    arms[role] = json.loads(out.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    arms[role] = {"role": role, "error": "unparsable output"}
            if proc.stderr:  # kept even on success: refusal reports are evidence
                (out_dir / f"{head}-{seed}-{role}.stderr.txt").write_text(
                    proc.stderr, encoding="utf-8"
                )
                if "error" in arms[role]:
                    arms[role]["stderr"] = proc.stderr[-3000:]
            print(
                f"R2 {head} seed {seed} {role}: {'ok' if 'error' not in arms[role] else 'FAILED'}",
                flush=True,
            )
    return arms


def _selfcheck():
    """The scorer's own reds, on synthetic records (no process runs): equal records compare equal,
    a 1-ulp weight and a swapped ring entry are flagged, a dirty chronology is flagged, and a
    verdict with an I_K2 equal to its twin marks the seed INVALID."""
    rec = {
        "before": {k: [1, 2] for k in _COMPARED_BEFORE},
        "rows": [{k: [[0.5, 0.25]] if k == "weights" else [1] for k in _COMPARED_ROW}],
        "construct_motor": 0,
        "construct_chrono": {"step": 0, "nights": 0},
        "replay_windows": [{"before": {"step": 3}, "after": {"step": 3}, "motor": 0}],
    }
    rec["before"]["ring"] = [["A", "a"], ["B", "b"]]
    assert records_equal(rec, copy.deepcopy(rec))
    assert comparator_controls(rec)
    dirty = copy.deepcopy(rec)
    dirty["replay_windows"][0]["after"]["step"] = 4
    assert chrono_clean(rec) and not chrono_clean(dirty)
    latched = dict(copy.deepcopy(rec), fault={"phase": "reconcile", "cause": "x"})
    assert not records_equal(rec, latched)
    reds = dict.fromkeys(list(_RED_TWIN) + ["CHRONO", "MOTOR"], True)
    assert not _complete({"role": "O_rng", "error": "exit 1"}), "a crashed arm is never complete"
    entry = {"guard_on_ok": True, "guard_on_refused": True, "guard_off_flagged": True}
    one = {
        "c1": {"K": True},
        "c6": {"K": True},
        "chrono": {"r": True},
        "reds": reds,
        "planted": dict.fromkeys(PLANTED, entry),
        "c4_structural_moves": True,
        "c4_comparator": True,
        "c2": {"I_K1_equal": True, "I_K2_differs": True},
        "arms_complete": True,
        "failwrite": {"ok": True},
        "c8": {"a": True, "b": True, "c": True, "d": True},
    }
    s1, s2 = DEV_SEEDS[:2]
    per = {s1: one, s2: copy.deepcopy(one)}
    per[s2]["c2"]["I_K2_differs"] = False
    v = verdict(per, [s1, s2])
    assert v["verdict"] != "PASS" and v["invalid_seeds"] == [s2], v
    assert v["R2-c3"] == "NOT MET", "an INVALID seed makes R2-c3 NOT MET, never dropped"
    per[s2]["c2"]["I_K2_differs"] = True
    per[s1]["reds"]["O_w"] = False
    assert verdict(per, [s1, s2])["R2-c3"] == "RED"
    assert codex14_status()["memory_unit_unreconstructible"] == "UNVERIFIED"
    print(
        "replay_hydration selfcheck: OK -- equal/ulp/swap/latched/chronology/INVALID-seed reds fire"
    )


def run_campaign(out_dir, seeds, base_src=None, workers=1):
    """The seed block: per seed, the R2-head arms and (with `base_src`) the base-head I arms; seeds in
    `workers` threads, each seed's processes serial. Writes per-seed comparisons and the verdict."""
    from concurrent.futures import ThreadPoolExecutor

    check_seeds(seeds)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(seed):
        arms = run_seed(seed, out_dir / "arms")
        base = run_seed(seed, out_dir / "arms", head="base", src=base_src) if base_src else None
        return seed, seed_comparisons(arms, base)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_seed = dict(pool.map(one, seeds))
    (out_dir / "per_seed.json").write_text(
        json.dumps(per_seed, indent=1, sort_keys=True), encoding="utf-8"
    )
    v = verdict(per_seed, list(seeds))
    (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print(f"R2 campaign: {v['verdict']}")
    return v


def main(argv):
    here = Path(__file__).resolve().parent  # a script's own directory must not shadow the stdlib
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    if argv[:1] == ["--proc"]:
        role, seed, home, out, head = argv[1], int(argv[2]), argv[3], argv[4], argv[5]
        run_process(role, seed, home, out, head, nonce=argv[6] if len(argv) > 6 else None)
        return
    opt = lambda name: argv[argv.index(name) + 1] if name in argv else None  # noqa: E731
    if argv[:1] == ["--seed"]:
        roles = opt("--roles")
        arms = run_seed(
            int(argv[1]),
            argv[2],
            head=opt("--head") or "r2",
            src=opt("--base-src"),
            roles=roles.split(",") if roles else None,
        )
        print(json.dumps({r: "error" in a for r, a in arms.items()}))
        return
    if argv[:1] == ["--campaign"]:
        seeds = tuple(int(s) for s in opt("--seeds").split(",")) if opt("--seeds") else DEV_SEEDS
        run_campaign(argv[1], seeds, base_src=opt("--base-src"), workers=int(opt("--workers") or 1))
        return
    _selfcheck()


if __name__ == "__main__":
    main(sys.argv[1:])
