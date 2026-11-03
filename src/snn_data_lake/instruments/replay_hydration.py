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
K3 records "the rest of the night, then 6" (2.2): twin_K3 measures the sleep ticks its cut night had
left, and every K3 restoring arm records that count plus 6 (integrator ruling, option (i)). The
harness carries circadian phase across the restart; the brain does not (finding iii): a restoring
process starts a fresh clock, so the K3 arms are coupled to twin_K3's record; c6's `K3_phase` checks
that the restore adopted the clock record of the cut night's onset (twin_K3's `onset_step`, D-79),
read only where pre_K3 flushed a circadian record (else not-applicable, `K3_PHASE_REFUSED`). The two sides differ at
row R: twin_K3 runs R rest-of-night rows, then a wake day and a sleep entry (`_enter_wake`,
`_enter_sleep` move learner state), then 6 rows; a K3 restore runs its wake period first, then R+6
sleep rows with no wake/sleep transition between row R and R+1. So res_K3 vs twin_K3 can differ from the
harness alone: `_k3_split` reads rows 1..R and R+1..R+6 apart (`c6_k3_split`), and a dev-seed K3 difference
is attributed to its half before any P2 seed.
Sleep replay is retired on main (owner1039): the sleep tick is the structural pass, so the
must-differ control (R2-c4) is the structural-off twin and the rewirer weight flip of the O_rw red.

The incumbent arms I_K1 / I_K2 (2.11 D1) run THIS FILE at the base head e1b3ae3c with
PYTHONPATH=<base checkout>/src (`--head base`); each compares only to its own twin at that head.
The FAILWRITE arm (the #1037 latch schedule A3, `fw_*` roles) and R2-c8 (the three bout crash
points, planted bout slots, one-frame check, FAILWRITE with B = sleep_bout) run in every seed.
This instrument is the binding reading of R2-c7 and its only one: tests/test_r2_p3_percept_transitions.py
runs these roles (run_seed) for its restart bar and holds no second copy of the procedure. Its "next
refit" is the grower's `ticks % refit_every == 0` sleep step, its boundary is a refit of THIS session
after a clean session 1 (`pre_K1g0`), and it kills only inside `_flush_growth` (`kill_growth`); the
retired pytest copy differed on all three and also killed inside `_flush_prototypes`. The
registration's amendment sentence naming this reading is the integrator's.
The percept arms (P3, K1g / R2-c7, 2.10, `P3_ROLES`) run on CPU in every seed: a session closed
cleanly, then a session killed inside the night's `_flush_growth` call; the restoring process
and the uninterrupted twin run ONLY the grower's sleep step to its next refit.

  PYTHONUTF8=1 PYTHONPATH=src python src/snn_data_lake/instruments/replay_hydration.py --selfcheck
  ... --seed SEED OUT_DIR [--head base --base-src PATH] [--roles a,b]   one seed's processes
  ... --campaign OUT_DIR [--seeds a,b,...] [--base-src PATH] [--workers N]  the seeds (dev block,
      sequential by default: each seed's processes are heavy; a confirmatory block needs
      R2_CONFIRMATORY_GO=P1, =P2 or =P3, which licenses that part's block and reserve only)
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

#: Gradient-claim declaration (gates/gradient_claim_gate.py, FR-002d): module-scope, literal, never imported.
GRADIENT_CLAIM = {
    "_next_refit": (
        "NOT_A_RULE",
        "steps the grower (neuro.tick) to its next refit and reads the model for the R2 bar; the "
        "instrument computes no update of its own, so there is no rule to classify (ARC 6 integrator "
        "declaration, 2026-11-01)",
    ),
}

DEV_SEEDS = tuple(range(15000, 15008))
#: each part's confirmatory block plus its reserve (2.6, 2.10 Seeds)
PART_SEEDS = {
    "P1": tuple(range(15010, 15018)) + tuple(range(15020, 15028)),
    "P2": tuple(range(15030, 15038)) + tuple(range(15040, 15048)),
    "P3": tuple(range(15050, 15058)) + tuple(range(15060, 15068)),
}
#: the verdict lines each part's own verdict joins (2.3 Disposition: R2-c6 is P2's own; A3 runs
#: FAILWRITE on "the P1 seeds"). R2-c3 is read per part (`_RED_PART`); R2-c8 joins BOTH parts until
#: the integrator rules which part the bout check belongs to (OWED, r2-instr-fix REPORT): stricter.
PART_BARS = {
    "P1": ("R2-c1", "R2-c2", "R2-c3", "R2-c4", "R2-c8", "FAILWRITE"),
    "P2": ("R2-c3", "R2-c6", "R2-c8"),
    "P3": ("R2-c7",),  # 2.3 Disposition: "R2-c7 (2.10) is P3's"
}
#: R2-c3's reds that the registration assigns to ONE part: 2.4's table (2.3: R2-c1..c5 are P1's) and
#: FAILWRITE's `FW:*` reds (A3: "then the P1 seeds of 2.6") are P1's; the 2026-10-06 P2 amendment's
#: "Reds" names O_sleepskip and O_cursor (K4). A red no line assigns joins both parts (OWED).
_RED_PART = {
    **dict.fromkeys(("O_wm", "O_hyd", "O_rng", "O_w", "O_stack", "O_rw", "CHRONO", "MOTOR"), "P1"),
    **dict.fromkeys(("DUP", "REORDER", "HELDOUT", "MISSING"), "P1"),  # 2.4's planted rows
    "O_rng_lone_replay_equal": "P1",  # O_rng's vacuous control
    "O_sleepskip": "P2",
    "O_cursor": "P2",
}


# distinct-from: instruments/g5_part_d_reading.py::self_check, gates/manifest_gate.py -- both fire
#   their own constructed reds; neither assigns a red to a part's verdict (their docstrings).
def _joins(red, part):
    """True when `red` is read in `part`'s R2-c3 (the dev block reads every red)."""
    owner = "P1" if red.startswith("FW:") else _RED_PART.get(red)
    return part == "dev" or owner in (None, part)


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
BASE_HEAD = "e1b3ae3c"  # the incumbent I arms' head (2.11 D1)
BASE_ROLES = (
    ("pre_K1", None),
    ("pre_K2", None),
    ("twin_K1", None),
    ("twin_K2", None),
    ("res_K1", "pre_K1"),
    ("res_K2", "pre_K2"),
)
#: R2-c7's planted percept reds (2.10: "DUP, REORDER, HELDOUT, MISSING (as 2.4, on `percept` events);
#: WIDTH ...; VOCAB ...; INSTRUMENT ...") -> the reason the shared reader or the grower's hydration counts
PERCEPT_PLANTED = {
    "DUP": "duplicate_id",
    "REORDER": "seq_regression",
    "HELDOUT": "heldout",
    "MISSING": "missing_slot",
    "WIDTH": "width",
    "VOCAB": "vocab",
    "INSTRUMENT": "instrument",
}
#: K1g (2.10), (role, parent): session 1 closes cleanly so the lake holds the session boundary CROSS
#: needs; session 2 is killed inside the night's `_flush_growth` call (pre_K1g) or runs on
#: uninterrupted (twin_K1g, and K1g_POS: the twin with its buffer shuffled across sessions); every
#: other arm restores a copy of the killed home
P3_ROLES = (
    ("pre_K1g0", None),
    ("pre_K1g", "pre_K1g0"),
    ("twin_K1g", "pre_K1g0"),
    ("K1g_POS", "pre_K1g0"),
    ("res_K1g", "pre_K1g"),
    ("K1g_CROSS", "pre_K1g"),
    ("O_buf", "pre_K1g"),
) + tuple((f"K1g_{k}_{side}", "pre_K1g") for k in PERCEPT_PLANTED for side in ("on", "off"))
_PERCEPT_BUFFER = ("buf_x", "buf_y", "cats")
_PERCEPT_MODEL = ("W", "b", "beta", "units", "traj", "ticks")
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
    integrator's go, and reading one destroys the pre-data licence -- so it needs R2_CONFIRMATORY_GO
    set to ONE part (P1, P2 or P3) with every seed in that part's block or reserve (2.6, 2.10).
    Returns the part the seeds are read for: "dev", "P1", "P2" or "P3"."""
    seeds = list(seeds)
    if all(s in DEV_SEEDS for s in seeds):
        return "dev"
    part = os.environ.get("R2_CONFIRMATORY_GO")
    if part not in PART_SEEDS or any(s not in PART_SEEDS[part] for s in seeds):
        stray = [s for s in seeds if s not in DEV_SEEDS]
        raise ValueError(
            f"seeds {stray} are outside the dev block 15000-15007 and not all in the block that "
            f"R2_CONFIRMATORY_GO={part!r} names (P1, P2 or P3): no confirmatory licence"
        )
    return part


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
    for u in order[4 : 2 + M_POST]:
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


def _track_clock(role, result):
    """Harness-side clock facts, read from the process under test and never from its restore. A pre_
    role notes the step of its last `_flush_circadian` call that WROTE (a truthy return) in
    `result["flushed_step"]` (None when this head has no circadian flush, or none wrote); a call that
    returned False wrote nothing, so a failed write never reads as a restore that dropped the record
    (#7204). Two causes return False (circadian brain.py:5412-5439): a raised write or a failed keep
    bumps `_circadian_flush_failed` and goes to `result["flush_failed_steps"]`; persist off or an
    unchanged record bumps nothing and goes to `result["flush_unwritten_steps"]` (#7255 NIT-1). A head
    with no such counter reads every False as failed. twin_K3 notes the step of each sleep onset, returned as
    `seen["onset"]` (tick order: `_step += 1`, then `_enter_sleep`, whose persist block flushes the
    clock first)."""
    from snn_data_lake.engine.brain import Brain

    seen = {"onset": None}
    if role.startswith("pre_"):
        result["flushed_step"] = None
        result["flush_failed_steps"] = []
        result["flush_unwritten_steps"] = []
        flush = getattr(Brain, "_flush_circadian", None)
        if flush is not None:

            def flushed(self, *a, **kw):
                failures = getattr(self, "_circadian_flush_failed", None)
                done = flush(self, *a, **kw)
                if done:
                    result["flushed_step"] = self._step
                elif failures is not None and failures == self._circadian_flush_failed:
                    result["flush_unwritten_steps"].append(self._step)
                else:
                    result["flush_failed_steps"].append(self._step)
                return done

            Brain._flush_circadian = flushed
    if role == "twin_K3":
        enter = Brain._enter_sleep

        def onset(self, *a, **kw):
            seen["onset"] = self._step
            return enter(self, *a, **kw)

        Brain._enter_sleep = onset
    return seen


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


def _sleep_rows(brain, draws, n=S_SLEEP, until_wake=False):
    """The 2.2 row of each of `n` sleep ticks. With `until_wake` (twin_K3) the rest of the night the
    kill cut is recorded first, so the row count tells the K3 restores how much night was left. The
    harness carries circadian phase across the restart; the brain does not (finding iii)."""
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

    if until_wake:
        nights, ticks = brain._nights, 0
        while brain._nights == nights:
            if brain.tick() == "asleep":
                observe()
            ticks += 1
            if ticks > 4 * brain.day_period:
                raise RuntimeError("the cut night did not end within four periods")
    _tick_sleeps(brain, n, observe)
    return rows


# ---- planted faults (written into a restoring process's COPIED home before its brain is built) --


def _plant(kind, percept=False):
    """Append one constructed record to the process's lake; returns its event_id. With `percept` it
    is R2-c7's planted `percept` event (`_percept_red`)."""
    from snn_data_lake.substrate.home import save_episode
    from snn_data_lake.substrate.learner_events import KIND_SPECS, read_learner_events

    if kind in ("BOUTDIGEST", "BOUTPROGRESS"):
        return _plant_bout(kind)
    # P3 percepts share this identity sequence and may follow the last sleep event.
    events, _ = read_learner_events(tuple(KIND_SPECS))
    if percept:
        e = _percept_red(kind, events)
        if not save_episode(e):
            raise OSError("instrument: planted percept was not appended")
        return e["event_id"]
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


def _percept_red(kind, events):
    """R2-c7's planted `percept` event (2.10), built from the lake's last percept as one more link of
    its session's chain (`pseq` + 1): a guard-OFF arm that admits it forms ONE pair the twin never
    had. DUP repeats an earlier id of that session, REORDER regresses `(session, seq)` to an unused
    id, and each other kind breaks the one slot its guard checks."""
    from snn_data_lake.growth.neurogenesis_engine import pack_percept, unpack_percept

    percepts = [e for e in events if e["kind"] == "percept"]
    last = percepts[-1]
    session = last["session"]
    seq = 1 + max(int(e["seq"]) for e in events if e["session"] == session)
    e = dict(last, event_id=f"{session}-{seq}", seq=str(seq), pseq=str(int(last["pseq"]) + 1))
    if kind == "DUP":
        earlier = [p for p in percepts if p["session"] == session][-2]
        e.update(event_id=earlier["event_id"], seq=earlier["seq"])
    elif kind == "REORDER":
        e.update(event_id=f"{session}-0", seq="0")
    elif kind == "HELDOUT":
        e["split"] = "heldout"
    elif kind == "MISSING":
        del e["category"]
    elif kind == "WIDTH":  # 63 dims under the registered `width=64` spelling
        e["x"] = pack_percept(unpack_percept(last["x"])[:-1])
    elif kind == "VOCAB":  # its own flushed count covers it, so only the vocab check refuses it
        category = max(int(p["n_protos"]) for p in percepts) + 8
        e.update(category=str(category), proto_flushed=str(category + 1))
    elif kind == "INSTRUMENT":
        e["provenance"] = "instrument"
    else:
        raise ValueError(f"unknown planted percept kind {kind}")
    return e


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


def _neuter(kind, percept=False):
    """The guard-OFF arm of a planted fault: the reader or the brain admits what its guard refuses.
    With `percept` (R2-c7, 2.10) the same guard is off on the `percept` path: the shared reader AND
    the grower's hydration (`_percept_refusal`) admit the planted percept."""
    import snn_data_lake.substrate.learner_events as le

    def admit(event):  # in place, as the reader's pads below: the one slot the guard checks
        if kind == "HELDOUT" and event.get("split") == "heldout":
            event["split"] = "train"
        elif kind == "MISSING" and "category" not in event:
            event["category"] = "0"
        elif kind == "INSTRUMENT" and event.get("provenance") == "instrument":
            event["provenance"] = "self"

    if kind in ("DUP", "REORDER"):
        le._identity_refusal = lambda event, seen, last: None
    elif kind in ("HELDOUT", "MISSING", "INSTRUMENT"):
        real = le._refusal

        def faulted(event, spec, allow_instrument):
            if kind == "MISSING" and "target" not in event:
                event["target"] = "GREET"  # the reader pads the missing label (the fault)
            if kind == "HELDOUT" and event.get("split") == "heldout":
                event["split"] = "train"  # the split guard neutered
            if percept and event.get("kind") == "percept":
                admit(event)
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
    if percept:
        import snn_data_lake.growth.neurogenesis_engine as ne

        real_percept = ne._percept_refusal

        def admitted(event, width, n_units, seen, last):
            if kind in ("DUP", "REORDER"):  # the identity guard, as the reader's above
                seen, last = set(), None
            elif kind == "WIDTH":  # the width guard: the event's own length taken as declared
                n = len(ne.unpack_percept(str(event.get("x", ""))) or [])
                event, width = dict(event, encoder=f"arm.percept gained width={n}"), None
            elif kind == "VOCAB":
                n_units = None
            else:
                admit(event)
            return real_percept(event, width, n_units, seen, last)

        ne._percept_refusal = admitted


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


# distinct-from: _attributed_hydrate above -- the same AST technique on another function: that one
#   re-types the learner's redo comparison raises for attribution; this removes one percept guard.
def _cross_pairing():
    """CROSS (2.10): the grower's pairing without its session/adjacency guard, so it joins the last
    percept of one session to the first of the next. Matched by syntax, never by text; a changed
    guard is refused rather than silently left in place."""
    import snn_data_lake.growth.neurogenesis_engine as ne

    real = ne.hydrate_transitions
    tree = ast.parse(textwrap.dedent(inspect.getsource(real)))
    guard = "prev is not None and prev[0] == session and prev[1] + 1 == pseq"
    guard = ast.dump(ast.parse(guard).body[0].value)
    hits = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.dump(node.test) == guard:
            node.test = ast.parse("prev is not None").body[0].value
            hits += 1
    if hits != 1:
        raise RuntimeError("instrument: the percept pairing guard changed")
    namespace = dict(real.__globals__)
    ast.fix_missing_locations(tree)
    exec(compile(tree, real.__code__.co_filename, "exec"), namespace)
    ne.hydrate_transitions = namespace[real.__name__]


def _patch_restore(role):
    """Omission arms on the restoring side (2.4, and 2.10's O_buf and CROSS): each removes ONE
    carrier or guard from the restore."""
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
    elif role == "O_buf":  # a real restart whose grower gets NO transition hydration
        from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

        def no_transitions(engine, events, **kw):
            engine._transitions_hydrated = True  # still construction-only; the buffer stays empty
            engine.transition_refusals = {}
            return 0

        NeurogenesisSelfImprove.restore_transitions = no_transitions
    elif role == "K1g_CROSS":
        _cross_pairing()

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


def run_process(role, seed, home, out, head="r2", *, nonce=None, k3_rest=None):
    global _throwaway_home, _throwaway_token

    check_seeds([seed])  # `--proc` is an entry point of its own, not only run_seed's child
    k3 = dict(R2_ROLES).get(role) == "pre_K3"
    if k3 and k3_rest is None:  # the harness carries circadian phase; the brain does not (iii)
        raise ValueError(
            f"{role}: a K3 restore records twin_K3's measured rest of the night plus "
            f"{S_SLEEP}, and none was given"
        )
    _throwaway_home, _throwaway_token = None, None
    private = _owned_private_home(home, nonce)
    _throwaway_home, _throwaway_token = private, nonce
    os.environ["NORD_HOME"] = str(private)
    if role.startswith("fw_"):
        return run_failwrite(role, seed, out)
    if role in dict(P3_ROLES):
        return run_percept(role, seed, out)
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
    seen = _track_clock(role, result)
    brain = make_brain(seed=seed, persist=True)
    result["construct_motor"] = motor.n
    result["construct_chrono"] = _chrono(brain)
    # the record the construction adopted (circadian_restored, brain.py:4270 of the circadian fix); read
    # BEFORE the CHRONO red ticks, because chrono_clean compares the clock with it, never with 0
    result["construct_circadian_restored"] = getattr(brain, "_circadian_restored", None)
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
    if role == "twin_K3":  # the rest of the night, then 6: its length is the K3 restores' count
        result["onset_step"] = seen["onset"]  # the cut night's onset record: c6's K3_phase reads it
        result["rows"] = _sleep_rows(brain, draws, until_wake=True)
        result["k3_rest"] = len(result["rows"]) - S_SLEEP
    else:
        result["rows"] = _sleep_rows(brain, draws, S_SLEEP + (k3_rest if k3 else 0))
        if k3:
            result["k3_rest"] = k3_rest
    result["replay_windows"] = windows
    result["motor_total"] = motor.n
    _write(out, result)
    os._exit(0)


def _write(out, result):
    Path(out).write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")


# ---- K1g (2.10, P3 / R2-c7): the grower's percept-transition restart ------------------------


def _percept_snap(brain):
    """The hydrated (or, at the boundary, live) transition buffer and the chronology hydration must
    not touch."""
    neuro = brain.neuro
    return {
        "buf_x": _plain(neuro._buf_x),
        "buf_y": _plain(neuro._buf_y),
        "cats": sorted(neuro._categories),
        "refusals": dict(neuro.transition_refusals),
        "prev_none": neuro._prev is None,
        "observed": neuro.observed,
        "n_units": brain.brain.arm.learner.n_units,
    }


def _next_refit(neuro):
    """ONLY the grower's sleep step, through the next refit (`neuro.tick()` until a tick with
    `ticks % refit_every == 0`; no new percepts), then the model the bar reads: W, b, beta, units,
    trajectory, ticks. A refit that raises is recorded, and such a record is never equal."""
    rows = len(neuro.trajectory)
    try:
        neuro.tick()
        while neuro.ticks % neuro.refit_every:
            neuro.tick()
    except Exception as exc:  # noqa: BLE001 - evidence, never a crash of the arm
        return {"error": f"{type(exc).__name__}: {exc}"}
    m = neuro._model
    return {
        **{k: None if m is None else getattr(m, k).tolist() for k in ("W", "b", "beta")},
        "units": neuro._n_units,
        "traj": _plain(neuro.trajectory),
        "ticks": neuro.ticks,
        "refit_row": len(neuro.trajectory) > rows,
    }


def run_percept(role, seed, out):
    """One K1g process (2.10), CPU (CUDA refits are not claimed deterministic). `pre_K1g0` runs
    session 1 to a completed night and closes. `pre_K1g` runs session 2 past a refit of its own and
    dies inside the night's `_flush_growth` call (`kill_growth`: the wrapper exits when the call
    returns, before `_flush_prototypes` and the coverage flush, brain.py:3798-3800); `twin_K1g` runs the same
    schedule, snapshots its buffer there and runs only the grower's sleep step to the next refit;
    `K1g_POS` is that twin with its buffer shuffled across sessions. Every other arm restores a copy
    of the killed home (with its patch or planted percept) and runs the same grower step."""
    from snn_data_lake.engine.brain import make_brain

    result = {"role": role, "seed": seed}
    _patch_restore(role)
    kind = next((k for k in PERCEPT_PLANTED if role.startswith(f"K1g_{k}_")), None)
    if kind is not None:
        result["planted"] = _plant(kind, percept=True)
        if role.endswith("_off"):
            _neuter(kind, percept=True)
    restoring = dict(P3_ROLES)[role] == "pre_K1g"
    if restoring:
        result["reader"] = _reader_view()
    _track_clock(role, result)
    brain = make_brain(seed=seed, persist=True, device="cpu")
    neuro = brain.neuro
    result["construct_chrono"] = _chrono(brain)
    result["construct_circadian_restored"] = getattr(brain, "_circadian_restored", None)
    result["fault"] = getattr(brain, "_r2_fault", None)
    if role == "pre_K1g0":
        _tick_to_boundary(brain)
        brain.close()
        _write(out, result)
        os._exit(0)
    if not restoring:
        rows, real_flush = len(neuro.trajectory), brain._flush_growth

        def boundary(*a, **kw):
            flushed = real_flush(*a, **kw)
            if len(neuro.trajectory) > rows:  # past at least one refit of this session
                result["before"] = _percept_snap(brain)
                if role == "K1g_POS":
                    half = len(neuro._buf_y) // 2
                    neuro._buf_y = neuro._buf_y[half:] + neuro._buf_y[:half]
                if role != "pre_K1g":
                    result["model"] = _next_refit(neuro)
                _write(out, result)
                os._exit(0)
            return flushed

        brain._flush_growth = boundary
        for _ in range(600):
            brain.tick()
        raise RuntimeError("K1g: no refit within 600 ticks")
    result["before"] = _percept_snap(brain)
    result["model"] = _next_refit(neuro)
    _write(out, result)
    os._exit(0)


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


def _k3_split(res, twin, rest):
    """R2-c6's K3 comparison read in its two halves: rows 1..R (the rest of the night, where the
    restore has no wake/sleep transition the twin had) and rows R+1..R+6. The halves are not
    independent: both compare the same 'before' fields, and rows R+1..R+6 inherit any rest-half
    divergence, so 'night' False attributes to the six rows only when 'rest' is True."""
    half = (
        lambda rec, lo, hi: (  # noqa: E731
            rec and "rows" in rec and dict(rec, rows=rec["rows"][lo:hi])
        )
    )
    return {
        "rest": records_equal(half(res, None, rest), half(twin, None, rest)),
        "night": records_equal(half(res, rest, None), half(twin, rest, None)),
    }


def _k3_phase(res, twin):
    """D-79's cross-check on the K3 count (MAJOR-3 (i)): the harness carries the cut night's remaining
    ticks, so the brain must have resumed that night's phase too. Tick order: `_step += 1`, then
    `_enter_sleep`, whose persist block flushes the clock first; the schedule ends at a night's
    boundary, so pre_K3's sleep ticks are the NEXT night's first ones and the last flush is that
    night's onset. The restore adopts that record, so its step EQUALS the onset step twin_K3
    recorded; a step off by any gap reads False. A brain that carries no phase (the record is
    absent) reads False: that is finding (iii), not a harness fault. The caller refuses this read
    when pre_K3 wrote no record; that refusal does not hide an isolation defect: `chrono_clean`
    for res_K3 still reads False when a record is restored and the parent's flushed_step is not an
    int (`_clock_clean`)."""
    step = ((res or {}).get("construct_circadian_restored") or {}).get("step")
    onset = (twin or {}).get("onset_step")
    return type(step) is int and type(onset) is int and step == onset


def _parent_flush_failed(parent):
    """The parent pre_ process called `_flush_circadian`, none wrote and at least one failed
    (a raised write or a failed keep), so it wrote no record (harness-side, `_track_clock`)."""
    parent = parent or {}
    return parent.get("flushed_step") is None and bool(parent.get("flush_failed_steps"))


def _clock_clean(r, parent):
    """The restored clock at construction. `parent` is the pre_ arm's result: the step its last
    `_flush_circadian` call flushed, recorded by the harness. With a restored record the clock, the
    record and the parent's flush must agree (a restore that adopted the wrong record agrees with
    itself, not with the parent); with none, the parent must have flushed none (a restore that
    DROPPED a flushed record is not a head without circadian) and the clock starts at 0. A parent
    whose every flush failed (`flush_failed_steps` set, `flushed_step` None) wrote no record to carry,
    so it is a named FAIL for EVERY restoring arm (`_parent_flush_failed`), never a head without
    circadian (#7255 MINOR-1). `nights` is not restored, so it stays 0."""
    c = r.get("construct_chrono") or {}
    restored = r.get("construct_circadian_restored")
    flushed = (parent or {}).get("flushed_step")
    if restored is None:
        step_ok = flushed is None and c.get("step") == 0 and not _parent_flush_failed(parent)
    else:
        step_ok = type(flushed) is int and c.get("step") == restored.get("step") == flushed
    return step_ok and c.get("nights") == 0


def chrono_clean(r, parent=None):
    """Hydration and redo tick nothing (construction leaves the clock where the parent's flushed
    record put it, see `_clock_clean`) and a sleep step advances no chronology and reaches no
    actuator (R2-c5's instrument-level half)."""
    windows = r.get("replay_windows") or []
    return (
        r.get("construct_motor") == 0
        and _clock_clean(r, parent)
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


# distinct-from: plasticity/rule_composer.py -- rule_charge drives leave-one-out RULE arms on a live
#   trajectory; this only lists which recorded 2.2 records two finished arms differ on. Site read.
# distinct-from: plasticity/r_stdp.py -- _FanOut is a barrier thread pool; a keyword overlap only.
def moved_records(a, b):
    """Per-carrier attribution (2.4 last paragraph): the compared records on which two complete arms
    differ, as `before.<key>` / `rows.<key>`. Reported, never a bar; None when either is incomplete."""
    if not (_complete(a) and _complete(b)):
        return None
    keys = [f"before.{k}" for k in _COMPARED_BEFORE if a["before"].get(k) != b["before"].get(k)]
    rows = list(zip(a.get("rows", []), b.get("rows", [])))
    keys += [f"rows.{k}" for k in _COMPARED_ROW if any(x.get(k) != y.get(k) for x, y in rows)]
    return keys


def _planted_entry(kind, arms):
    on, off = arms.get(f"{kind}_on", {}), arms.get(f"{kind}_off", {})
    twin, reason = arms.get(_PLANT_TWIN[kind]), PLANTED[kind]
    if not (_complete(on) and _complete(off) and _complete(twin)):
        return {"guard_on_ok": False, "guard_on_refused": False, "guard_off_flagged": False}
    if kind in _READER_KINDS:
        refusals = (on.get("reader") or {}).get("refusals") or {}
        off_refusals = (off.get("reader") or {}).get("refusals") or {}
        refused = refusals.get(reason, 0) >= 1
        # 2.4: these reds must fire on the ring and the draws; a latched off arm is INVALID
        moved = set(moved_records(off, twin)) & {
            "before.ring",
            "before.ring_ids",
            "rows.drawn",
            "rows.drawn_ids",
        }
        flagged = off_refusals.get(reason, 0) == 0 and off.get("fault") is None and bool(moved)
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
    survivors = [arms.get(f"fw_{k}_{a}") for k in ("t", "b") for a in ("absent", "landed", "torn")]
    whole = all(_fw_ok(r) and "calls" in r for r in survivors)
    posed = [c for r in survivors if whole for c in r["calls"] if c["call"] == "vii"]
    out["posed"] = (  # A3: C(vii) reads UNVERIFIED unless EVERY survivor's vii call was refused
        "VERIFIED"
        if posed and all("PosedRefused" in (c["exc"] or "") for c in posed)
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


def percept_equal(a, b):
    """R2-c7's equality (2.10, no tolerance): the hydrated `_buf_x`/`_buf_y`/`_categories` and the
    model after the next refit. An incomplete, differently latched or refit-failed record is never
    equal."""
    if not (_complete(a) and _complete(b)) or a.get("fault") != b.get("fault"):
        return False
    ma, mb = a.get("model") or {}, b.get("model") or {}
    if not ma or not mb or "error" in ma or "error" in mb:
        return False
    return all(a["before"].get(k) == b["before"].get(k) for k in _PERCEPT_BUFFER) and all(
        ma.get(k) == mb.get(k) for k in _PERCEPT_MODEL
    )


def _percept_planted_entry(kind, arms):
    """A planted percept red's three readings, as `_planted_entry`'s: the guard-ON arm counts the
    refusal (the shared reader's tally or the grower hydration's) and is either latched visibly or
    restores exactly the twin; the guard-OFF arm is unlatched, counts no refusal, and moves the
    buffer (2.10 "each must fire")."""
    on, off = arms.get(f"K1g_{kind}_on"), arms.get(f"K1g_{kind}_off")
    twin, reason = arms.get("twin_K1g"), PERCEPT_PLANTED[kind]
    if not (_complete(on) and _complete(off) and _complete(twin)):
        return {"guard_on_ok": False, "guard_on_refused": False, "guard_off_flagged": False}

    def counted(rec):
        reader = ((rec.get("reader") or {}).get("refusals") or {}).get(reason, 0)
        return reader + (rec["before"].get("refusals") or {}).get(reason, 0)

    moved = any(off["before"].get(k) != twin["before"].get(k) for k in _PERCEPT_BUFFER)
    return {
        "guard_on_ok": on.get("fault") is not None or percept_equal(on, twin),
        "guard_on_refused": counted(on) >= 1,
        "guard_off_flagged": off.get("fault") is None and counted(off) == 0 and moved,
    }


#: #6952 A-1: on a head without the circadian fix there is no clock record to resume, so the K1g
#: chronology has nothing to be compared with. The read is keyed on the code under test (pre_K1g's own
#: flush, harness-side), never on what the restore reports (#7142).
K1G_CHRONO_REFUSED = (
    "K1g chronology not read: pre_K1g flushed no circadian record, so this head has no clock to "
    "resume (lifted by claude/circadian-p4b-1029; circadian-p4b #6952 A-1)"
)
#: The fourth cell (#7183 (C)): the parent flushed none, the restore carries a record anyway -- an
#: isolation defect (a stale or shared home, or the live lake), a FAIL and never a refusal.
K1G_CHRONO_NO_LINEAGE = (
    "K1g chronology FAIL: a restored record with no flushed lineage (pre_K1g flushed no circadian "
    "record; a stale or shared home, or the live lake)"
)
#: The other False returns (persist off; the record already holds that step) bumped no failure
#: counter: nothing new was written, which is not a failed write (#7255 NIT-1).
K1G_CHRONO_UNWRITTEN = (
    "K1g chronology not read: pre_K1g's _flush_circadian returned False without a failure (persist "
    "off, or the record unchanged), so no record was written to resume"
)
K3_PHASE_UNWRITTEN = (
    "K3_phase not applicable: pre_K3's _flush_circadian returned False without a failure (persist "
    "off, or the record unchanged), so no record was written to resume"
)
CHRONO_FLUSH_FAILED = (
    "chrono FAIL: the parent pre_ process's _flush_circadian wrote nothing (none wrote, and at least "
    "one returned False after a raised write or a failed keep), so it wrote no record for the "
    "restore to carry"
)
#: A pre_ flush that returned False wrote nothing (engine/brain.py:5394-5403): its own cause, never
#: 'the restore dropped the record' (#7204 MINOR-1).
K1G_CHRONO_FLUSH_FAILED = (
    "K1g chronology FAIL: pre_K1g's _flush_circadian returned False and wrote no record (the write "
    "raised or the keep failed), so there is no record to compare the restore with"
)
K3_PHASE_FLUSH_FAILED = (
    "K3_phase FAIL: pre_K3's _flush_circadian returned False and wrote no record (the write raised "
    "or the keep failed), so there is no phase to resume"
)
#: c6's `K3_phase` is keyed on the same harness-side bit as the K1g chronology (#7183 (C) NIT).
K3_PHASE_REFUSED = (
    "K3_phase not applicable: pre_K3 flushed no circadian record, so this head has no clock to "
    "resume (lifted by claude/circadian-p4b-1029; circadian-p4b #6952 A-1)"
)


def c7_seed(arms):
    """R2-c7 (2.10, part P3) on one seed: K1g's restore equals the uninterrupted twin (the buffer and
    the next refit) and leaves the chronology untouched, read as a four-cell table on (pre_K1g
    WROTE a circadian record (a truthy `_flush_circadian` return, never a mere call), the restore
    carries one): a pre_K1g whose every flush returned False is its own FAIL
    (`K1G_CHRONO_FLUSH_FAILED`), whatever the restore carries; (yes, yes) scored; (yes, no) FAIL;
    (no, yes) FAIL by name (`K1G_CHRONO_NO_LINEAGE`); (no, no) refused by name; CROSS and O_buf each move it; each planted
    percept red reads as `_percept_planted_entry`; the shuffled-buffer positive control's refit
    differs. Its own incomplete list: a P1/P2 arm never makes a P3 seed INVALID (2.3, A6)."""
    get, twin = arms.get, arms.get("twin_K1g")
    incomplete = [r for r, _ in P3_ROLES if not r.startswith("pre_") and not _complete(get(r))]
    incomplete += [
        r for r, _ in P3_ROLES if r.startswith("pre_") and (not get(r) or "error" in get(r))
    ]
    twin_model = (twin or {}).get("model") or {}
    if _complete(twin) and not twin_model.get("refit_row"):
        incomplete.append("twin_K1g")  # no refit after the boundary: the bar would compare nothing
    res = get("res_K1g") or {}
    hydrated = res.get("before") or {}
    k1g = (
        percept_equal(res, twin)
        and res.get("fault") is None
        and hydrated.get("prev_none") is True
        and hydrated.get("observed") == 0
        and not hydrated.get("refusals")
    )

    def fired(role):
        arm = get(role)
        return (
            _complete(arm)
            and _complete(twin)
            and "error" not in twin_model
            and arm.get("fault") is None
            and not percept_equal(arm, twin)
        )

    planted = {kind: _percept_planted_entry(kind, arms) for kind in PERCEPT_PLANTED}
    reds = {"CROSS": fired("K1g_CROSS"), "O_buf": fired("O_buf")}
    reds.update({kind: entry["guard_off_flagged"] for kind, entry in planted.items()})
    for kind in PERCEPT_PLANTED:  # A11.2: a neutered guard cannot latch; another guard did
        role = f"K1g_{kind}_off"
        if (get(role) or {}).get("fault") is not None and role not in incomplete:
            incomplete.append(role)
    pre = get("pre_K1g") or {}
    if pre.get("flushed_step") is None and pre.get("flush_failed_steps"):
        chrono, chrono_ok = {"scored": True, "clean": False, "fail": K1G_CHRONO_FLUSH_FAILED}, False
    elif pre.get("flushed_step") is None:  # the code under test wrote no clock: nothing to resume
        if res.get("construct_circadian_restored") is None:
            refused = (
                K1G_CHRONO_UNWRITTEN if pre.get("flush_unwritten_steps") else K1G_CHRONO_REFUSED
            )
            chrono, chrono_ok = {"scored": False, "refused": refused}, True
        else:  # but the restore carries a record: it came from outside this lineage
            chrono, chrono_ok = (
                {"scored": True, "clean": False, "fail": K1G_CHRONO_NO_LINEAGE},
                False,
            )
    else:  # it did, so the restore must carry it: a dropped record is a FAIL, never a refusal
        chrono_ok = _clock_clean(res, pre)
        chrono = {"scored": True, "clean": chrono_ok}
    pos = get("K1g_POS") or {}
    pos_model = pos.get("model") or {}
    control = (
        _complete(pos)
        and _complete(twin)
        and bool(pos_model)
        and "error" not in pos_model
        and any(pos_model.get(k) != twin_model.get(k) for k in ("W", "beta"))
    )
    ok = (
        k1g
        and all(reds.values())
        and all(all(entry.values()) for entry in planted.values())
        and control
        and chrono_ok
        and not incomplete
    )
    return {
        "K1g": k1g,
        "reds": reds,
        "planted": planted,
        "positive_control": control,
        "chrono": chrono,
        "incomplete": incomplete,
        "arms_complete": not incomplete,
        "ok": ok,
    }


def seed_comparisons(arms, base=None):
    """Every per-seed reading of 2.3, from the R2-head arms (and the base-head arms when given). A red
    FIRES only if its arm and its twin are COMPLETE records; an incomplete arm makes the seed
    incomplete (never a fire). c6's `K3_phase` (D-79's cross-check) is a conjunct of its all() only
    where pre_K3 WROTE a circadian record (a flush that returned False is `K3_PHASE_FLUSH_FAILED`,
    a False conjunct under its own name, never a dropped-record reading): a red there (the restore dropped or mis-adopted the
    phase) reads NOT MET. On a head without circadian (pre_K3 flushed none, harness-side) it is
    not-applicable: `K3_phase` is left out of c6 and `k3_phase_refused` names `K3_PHASE_REFUSED`
    (never a NOT MET; closes #7103 NIT-2)."""
    eq, get = records_equal, arms.get
    required = [r for r, _ in R2_ROLES if not r.startswith(("fw_", "pre_"))]
    incomplete = [r for r in required if not _complete(get(r))]
    incomplete += [
        r for r, _ in R2_ROLES if r.startswith("pre_") and (not get(r) or "error" in get(r))
    ]
    fired = lambda r, twin: (  # noqa: E731
        _complete(get(r)) and _complete(get(twin)) and not eq(get(r), get(twin))
    )
    clean = lambda r, twin: eq(get(r), get(twin)) and get(r).get("fault") is None  # noqa: E731
    c = {
        "c1": {  # two records latched by the same fault are equal, but neither restored
            "K0": clean("res_K0", "twin_K1"),
            "K1": clean("res_K1", "twin_K1"),
            "K2": clean("res_K2", "twin_K2"),
            "rw": clean("res_rw", "twin_rw"),
        },
        "c6": {
            "K3": eq(get("res_K3"), get("twin_K3")),
            "K4": eq(get("res_K4"), get("twin_K4")),
            "B": eq(get("res_B"), get("twin_B")),
        },
        "c6_k3_split": _k3_split(
            get("res_K3"), get("twin_K3"), (get("twin_K3") or {}).get("k3_rest", 0)
        ),
        "chrono": {
            r: chrono_clean(get(r, {}), get(dict(R2_ROLES)[r]))
            for r in (
                "res_K0",
                "res_K1",
                "res_K2",
                "res_K3",
                "res_K4",
                "res_B",
                "res_Bpre",
                "res_Bpost",
                "res_rw",
            )
        },
        "reds": {r: fired(r, twin) for r, twin in _RED_TWIN.items()},
        "attribution": {r: moved_records(get(r), get(twin)) for r, twin in _RED_TWIN.items()},
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
        "c7": c7_seed(arms),
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
    for kind in _READER_KINDS:  # A11.2: a neutered reader guard cannot latch; another guard did
        role = f"{kind}_off"
        if (get(role) or {}).get("fault") is not None and role not in incomplete:
            incomplete.append(role)
    pre_k3 = get("pre_K3") or {}
    if pre_k3.get("flushed_step") is None and pre_k3.get("flush_failed_steps"):
        c["c6"]["K3_phase"] = False  # a flush that returned False wrote nothing: a named FAIL
        c["k3_phase_flush_failed"] = K3_PHASE_FLUSH_FAILED
    elif pre_k3.get("flushed_step") is None:  # no clock written: nothing to resume
        unwritten = pre_k3.get("flush_unwritten_steps")
        c["k3_phase_refused"] = K3_PHASE_UNWRITTEN if unwritten else K3_PHASE_REFUSED
    else:
        c["c6"]["K3_phase"] = _k3_phase(get("res_K3"), get("twin_K3"))
    named = {
        role: CHRONO_FLUSH_FAILED
        for role in c["chrono"]
        if _parent_flush_failed(get(dict(R2_ROLES)[role]))
    }
    if named:
        c["chrono_flush_failed"] = named
    c["reds"]["CHRONO"] = _complete(get("CHRONO")) and not chrono_clean(
        get("CHRONO"), get(dict(R2_ROLES)["CHRONO"])
    )
    c["reds"]["MOTOR"] = _complete(get("MOTOR")) and not chrono_clean(
        get("MOTOR"), get(dict(R2_ROLES)["MOTOR"])
    )
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
    sides are identical reads NOT MET (ruling 9). Scoring a seed reads it, so the licence holds here,
    and it names the part: a P1, P2 or P3 verdict joins only that part's lines (PART_BARS); the dev
    block joins them all. A P3 seed is INVALID only by its own K1g arms (2.3: "a P3 red reverts P3,
    never P1"; A11.4); which P3 arm is D4's load-bearing one is OWED, so none replaces a seed yet."""
    part = check_seeds(seeds)
    c7_of = lambda s: per_seed[s].get("c7") or {}  # noqa: E731
    if part == "P3":
        incomplete = [s for s in seeds if not c7_of(s).get("arms_complete")]
        invalid = list(incomplete)
    else:
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
    c2 = every(lambda c: c.get("c2", {}).get("I_K1_equal") is True)
    # every key any counted seed reports: a red missing from one seed's record never fires there
    red_keys = dict.fromkeys(r for s in counted for r in per_seed[s]["reds"] if _joins(r, part))
    plant_keys = dict.fromkeys(
        k for s in counted for k in per_seed[s]["planted"] if _joins(k, part)
    )
    reds = {r: every(lambda c, r=r: c["reds"].get(r) is True, counted) for r in red_keys}
    fired_somewhere = {
        r: any(per_seed[s]["reds"].get(r) is True for s in counted) for r in red_keys
    }
    planted = {
        k: every(
            lambda c, k=k: all(
                (c["planted"].get(k) or {}).get(x)
                for x in ("guard_off_flagged", "guard_on_ok", "guard_on_refused")
            ),
            counted,
        )
        for k in plant_keys
    }
    controls = every(
        lambda c: all(v for k, v in c.get("c3_controls", {}).items() if _joins(k, part)),
        counted,
    )
    c3 = all(reds.values()) and all(planted.values()) and controls and not invalid
    # 2.3: the controls and R2-c6 hold "on EVERY seed", an INVALID one included
    c4 = every(lambda c: c["c4_structural_moves"] and c["c4_comparator"], counted)
    c6 = every(lambda c: all(c["c6"].values()), counted)
    failwrite = every(lambda c: bool(c.get("failwrite")) and c["failwrite"]["ok"], counted)
    c8 = every(lambda c: all(c["c8"].values()))
    # 2.10, on EVERY seed, no tolerance: an INVALID seed is counted and never dropped
    c7 = every(lambda c: (c.get("c7") or {}).get("ok") is True, counted)
    never = [r for r, hit in fired_somewhere.items() if not hit] if part != "P3" else []
    if part in ("dev", "P3"):  # a K1g red that fires on no seed halts P3 (2.3 R2-c3's rule, P8)
        k1g_reds = dict.fromkeys(r for s in counted for r in c7_of(s).get("reds", {}))
        never += [
            f"K1g:{r}"
            for r in k1g_reds
            if not any(c7_of(s).get("reds", {}).get(r) is True for s in counted)
        ]
    halted = len(invalid) > 2 or bool(never) or not counted
    bars = {
        "R2-c1": c1,
        "R2-c2": c2,
        "R2-c3": c3,
        "R2-c4": c4,
        "R2-c6": c6,
        "R2-c7": c7,
        "R2-c8": c8,
        "FAILWRITE": failwrite,
    }
    own = PART_BARS.get(part, tuple(bars))
    ok = all(bars[b] for b in own) and not halted and not invalid
    verdict_of = lambda b: "PASS" if b and not halted else "RED"  # noqa: E731
    lines = {b: verdict_of(v) if b in own else f"not {part}'s verdict" for b, v in bars.items()}
    if invalid and "R2-c3" in own:
        lines["R2-c3"] = "NOT MET"
    if invalid and part == "P3":
        lines["R2-c7"] = "NOT MET"
    return {
        "seeds": list(seeds),
        "part": part,
        "invalid_seeds": invalid,
        "incomplete_seeds": incomplete,
        "halted": halted,
        "red_fires_on_no_seed": never,
        **lines,
        "reds": reds,
        "planted": planted,
        "verdict": "PASS" if ok else ("NOT MET" if invalid else "RED"),
        "codex14": codex14_status(),
        "scope": "same-code restarts at K0-K4 and the staged bout; the interaction learner and the "
        "sleep carriers that touch it; the #1037 latch (FAILWRITE); K1g (R2-c7), the grower's percept "
        "transitions on CPU after a clean session 1; R2-c5 is pytest; I arms at base "
        "e1b3ae3c compare only to their own twins; sleep replay is retired on main (owner1039); "
        "the CTXREAD off arm filters its event out of `hydrate`, so it shows the guard is the only "
        "refuser, not a reachable fault effect (A10)",
    }


# ---- orchestration ----------------------------------------------------------------------------


# distinct-from: engine/learn_daemon.py, faculties/agency_monitor.py -- those (like the pin helpers
#   of report_order_probe.py, binding_instrument.py and recursive_tom.py) read the CWD repo's HEAD
#   and take no path; this reads a named tree's (the base checkout), so the I arms' head is on record.
def _head_of(path):
    """`git rev-parse HEAD` of the tree at `path`, or "" when it is not a git checkout."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _check_base(src):
    """The I arms run at the registered base head (2.11 D1) or not at all; returns that head."""
    head = _head_of(src)
    if not head.startswith(BASE_HEAD):
        raise ValueError(
            f"--base-src {src} is at {head or 'no git head'}, not the registered base {BASE_HEAD}"
        )
    return head


def run_seed(seed, out_dir, head="r2", src=None, roles=None):
    """All of one seed's processes for one head, each on its own home; restoring arms get a copy of
    their parent's home. Returns {role: record}. `roles` selects a subset (parents included)."""
    part = check_seeds([seed])
    if head == "base":  # the I arms run the base tree, never this one relabelled
        if not src:
            raise ValueError("--head base needs --base-src: the I arms run at e1b3ae3c")
        _check_base(src)
    here = Path(__file__).resolve()
    src = Path(src) if src else here.parents[2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # a part reads only its own observables (once-read block, heavy CPU): P3 the K1g arms, P1/P2
    # the R2 arms, the dev block both
    table = BASE_ROLES
    if head == "r2":
        table = {"P3": P3_ROLES, "P1": R2_ROLES, "P2": R2_ROLES}.get(part, R2_ROLES + P3_ROLES)
    if roles is not None:
        parent_of, roles = dict(table), set(roles)
        for role in list(roles):
            if parent_of.get(role) == "pre_K3":  # a K3 restore reads twin_K3's count
                roles.add("twin_K3")
            while parent_of.get(role):
                role = parent_of[role]
                roles.add(role)
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
            args = [sys.executable, str(here), "--proc", role, str(seed), str(home), str(out)]
            args += [head, nonce]
            # option (i): the harness carries circadian phase across the restart; the brain does
            # not (finding iii). A K3 restore gets twin_K3's measured count, or refuses without it.
            if parent == "pre_K3" and "k3_rest" in arms.get("twin_K3", {}):
                args.append(str(arms["twin_K3"]["k3_rest"]))
            try:
                proc = subprocess.run(
                    args,
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
            except subprocess.TimeoutExpired:  # A11.4: a hung arm is INVALID, never an abort
                proc = None
            if proc is None:
                arms[role] = {"role": role, "error": "timeout"}
            elif proc.returncode != 0:
                arms[role] = {"role": role, "error": f"exit {proc.returncode}"}
            elif not out.exists():
                arms[role] = {"role": role, "error": "no output"}
            else:
                try:
                    arms[role] = json.loads(out.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    arms[role] = {"role": role, "error": "unparsable output"}
            if proc is not None and proc.stderr:  # kept even on success: refusals are evidence
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

    part = check_seeds(seeds)
    base_src = None if part == "P3" else base_src  # P3 reads only the K1g arms (once-read block)
    here = Path(__file__).resolve()
    provenance = {  # which heads ran (2.11 D1); a base tree off e1b3ae3c is refused before any seed
        "src": _head_of(here.parents[2]),
        "base_src": _check_base(base_src) if base_src else None,
        "instrument_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
    }
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
    v = dict(verdict(per_seed, list(seeds)), provenance=provenance)
    (out_dir / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print(f"R2 campaign: {v['verdict']}")
    return v


def main(argv):
    here = Path(__file__).resolve().parent  # a script's own directory must not shadow the stdlib
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    if argv[:1] == ["--proc"]:
        role, seed, home, out, head = argv[1], int(argv[2]), argv[3], argv[4], argv[5]
        nonce = argv[6] if len(argv) > 6 else None
        k3_rest = int(argv[7]) if len(argv) > 7 else None  # twin_K3's count (finding iii)
        run_process(role, seed, home, out, head, nonce=nonce, k3_rest=k3_rest)
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
