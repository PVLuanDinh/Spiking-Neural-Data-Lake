"""Brain telemetry: the ONE snapshot contract, its provenance table, and an append-only recorder.

    This is the DATA layer only. It used to carry a second HTTP server and a second dashboard of its
    own, on the same default port as the cockpit -- two frontends racing for :8137, and a docstring in
    substrate/cockpit.py that called this one "retired" while it was still live. `substrate/cockpit.py`
    is the single frontend; everything here is what that frontend is allowed to read.

WHY THIS EXISTS
    `Brain.state()` has carried the docstring "Telemetry for /api/state" since it was written, and
    `research/brain_web.html` has been fetching `/api/state` -- but nothing in this repo ever served
    it. `LiveBrain.spike_snapshot()` documents itself as "the interface contract the live web tracker
    reads" and had zero consumers. Both halves of a dashboard existed; the wire between them did not.

    This module is that wire, and nothing more. It computes no new brain state. It reads what the
    canonical Brain already produces every tick.

THE HONESTY CONTRACT (the whole point -- read before adding a panel)
    A telemetry surface is the easiest place in a project to lie, because a plausible number is
    indistinguishable from a real one once it is rendered as a bar. Three rules hold the line:

    1. `snapshot()` is the ONLY source a panel may read. If a value is not in here, the panel does
       not get built. No reaching into private attributes from the frontend.
    2. A value that is absent renders as `None` -> the UI draws an em-dash, NEVER a zero bar. A zero
       bar is a claim ("this signal is live and reads zero"); an em-dash is the truth ("no signal").
    3. `verify_contract()` fails loudly if a key the dashboard references stops existing. That is the
       no-orphan-panel gate: it makes a silently-dead panel a build error instead of a lie on screen.

    Corollary, stated because it is tempting: several SNDL signals are computed-but-not-routed
    (`_motor_gain`, `_dev_fm_gain`, `_comp_ema` -- their own source comments say "NEVER applied").
    Those are exposed here under `gated:` with `applied: False`. A panel showing them MUST say so.
    A gain that never reaches a motor is not a control signal; it is a number.

    4. `PROVENANCE` (below) names every key whose value does NOT mean what a live bar implies -- frozen
       at construction, refreshed only in sleep, a hardcoded literal, a duplicate of another key, or an
       input stream that does not exist. A dashboard MUST render the tag next to the number.

       This exists because rule 2 was not enough. An audit of this file's own dashboard found four
       signals being drawn as live bars that are nothing of the kind: `integration_competence` is
       assigned only inside `Brain.__init__` (brain.py:394, and its except-branch sibling at :403)
       and never again; `pci` is computed only inside the sleep tick and only when `nights % 4 == 0`
       (`Brain._sleep_tick`, brain.py:2519); `spikes.midbrain` reduces to a constant 0.05 because its
       `novelty` term is written only under a `_salience_on` flag that is never enabled
       (`AutonomousArm.__init__` sets it from the `salience_novelty=False` kwarg at
       autonomous_loop.py:242, and `cortex/brain.py:241` records deliberately NOT passing it; it is
       read at :1170 and :1325 -- the previous note said ":1170" was "the only consumer", which is
       one read short and was corrected 2026-08-05 while re-resolving the rest of this paragraph);
       and `spikes.substrate` is the SAME expression as `spikes.hippocampus`
       (live_brain.py:255,:276), so the two bars are byte-identical every tick and only one of them
       is a region.

       EVERY LINE CITE IN THIS FILE WAS RE-RESOLVED 2026-08-04 AND MOST HAD DRIFTED -- the four
       above pointed at `try:`, `except Exception:`, `return bias` and a `svec` list element
       respectively, in code with nothing to do with the claim. The CLAIMS all re-verified against
       the tree; only the coordinates were wrong, which is the failure mode that makes a stale cite
       worse than none: it reads as evidence and resolves to something. `tools/citation_guard.py`
       does not catch this class (it gates untracked citations, not drifted line numbers), so
       cites here now name the enclosing SYMBOL wherever one exists.

       A number that has not changed since construction, drawn as a live bar, is a lie told slowly.
       Being in the snapshot was never the same as being alive, and this table is where the
       difference is written down.

NOT A SECOND BRAIN (Policy 4)
    This module never constructs a Brain. `snapshot()` is handed one that already exists, so there is
    exactly one brain and this is a window onto it.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

from snn_data_lake.brainstem import arm_config as _arm_config

# Keys the bundled dashboard reads. verify_contract() asserts every one is producible, so deleting a
# brain signal breaks the build instead of silently blanking a panel.
DASHBOARD_KEYS = (
    "circadian",
    "phase",
    "wake_ticks",
    "sleep_ticks",
    "nights_elapsed",
    "neuromod_serotonin",
    "neuromod_noradrenaline",
    "neuromod_acetylcholine",
    "cortisol",
    "curiosity_mag",
    "da_level",
    "rpe",
    "emotion",
    "dominant_drive",
    "access_index",
    "criticality",
    "region_coherence",
    "integration_competence",
    "grown_units",
    "predict_accuracy",
    "reasoning_accuracy",
    "reliability",
    "pc_surprise",
    "attended_modality",
    "map_position",
    # derived-section keys (verify_contract checks cognition | derived, so these belong here too --
    # they were omitted once and the dashboard silently drew em-dashes for affect until a human
    # noticed. That is exactly the regression this tuple exists to turn into a build failure.)
    "valence",
    "arousal",
    # the closed-loop panel: arm competence signals the cockpit draws as its headline progress row
    "cells_covered",
    "recoveries",
    "ground_ticks",
    "cpg_weight",
    "displacement",
    "td_flipped",
    "ignitions",
)

# Keys outside `cognition | derived` that the cockpit reads. Split out because they live in their own
# snapshot sections and verify_contract has to look for them there -- the previous version checked only
# cognition|derived, which meant a panel built on spikes.* or body.* had NO gate behind it at all.
DASHBOARD_SECTION_KEYS = {
    "spikes": ("cortex", "hippocampus", "amygdala", "basal_ganglia", "thalamus", "brainstem"),
    "body": ("theta_deg", "ee_mm", "reach_mm", "effector", "collision", "deviation"),
    # The revision counter is part of the CONTRACT, not decoration. A still picture and a live
    # one are pixel-identical to a viewer; without a counter that visibly advances, a frozen
    # renderer looks exactly like a stationary arm. This is the failure PROVENANCE exists for.
    "arm_view": ("arm_view_revision", "arm_view_available", "arm_view_reason"),
    "hardware": ("built", "joints", "underpowered", "sim_dof", "supply_ok"),
}

# key -> why its value is not what a live bar would imply. The cockpit renders the tag beside the value.
# Deleting an entry is how a future build declares a signal fixed; adding one is how it stays honest.
# Every claim here was verified against the cited source line, not inferred.
PROVENANCE = {
    # assigned once inside Brain.__init__ (brain.py:394, except-branch sibling :403) and never
    # touched by any tick -- read out by Brain.state() at :2930
    "integration_competence": "frozen at construction",
    "integration_gain": "frozen at construction",
    "integration_retention": "frozen at construction",
    "integration_ok": "frozen at construction",
    # written only in the sleep tick; pci additionally only when nights % 4 == 0
    # (pci brain.py:2519; criticality brain.py:2650 -- it is ALSO set once at birth, :367)
    "pci": "sleep-only, refreshes every 4th night",
    "criticality": "sleep-only",
    # string/bool literals in state(), not readings (goal brain.py:2892; reflex_armed :3055)
    "goal": "hardcoded literal",
    "reflex_armed": "hardcoded literal",
    # spike-section rows that are not independent live regions
    "spikes.olfactory": "no input stream -- constant 0.0",
    "spikes.midbrain": (
        "constant: its novelty term is never enabled -- and MEASURED inert, not merely unwired. "
        "Enabling salience_novelty takes this row from 1 to 32 distinct values over 60 ticks, then "
        "changes 0/200 workspace winners, 0/200 act-hold decisions and 0/200 trajectories, at "
        "+8.4% tick cost. Left off deliberately (v4.78)"
    ),
    "spikes.substrate": "duplicate of hippocampus (same expression)",
    # STRUCTURAL CENSUS, not a live reading: these count MODULES ON DISK via the static import
    # graph, computed once per process and never recomputed. They are constant for the entire life
    # of the daemon, and the cockpit drew all three untagged (specs/010, C30). They are honest and
    # load-bearing as a census -- what they must not read as is a live coverage level that could
    # rise while you watch it.
    "coverage_reachable": "structural census of modules on disk, computed once per process",
    "coverage_total": "structural census of modules on disk, computed once per process",
    "coverage_dormant": "structural census of modules on disk, computed once per process",
    # STRUCTURAL, not live: reverberation() resets the router, drives a FIXED seed at a fixed 1.0 for a
    # fixed count and restores -- it never reads the live drive, so it is a pure function of the
    # connectome topology. Proven identical (0.25) across five different live states; it was being
    # recomputed every 5 ticks at 0.43 ms a call to produce the same number. Now computed once
    # (the `self._connectome_router.reverberation(seed, 6)` call at brain.py:2215, stored as
    # `_connectome_reverb` and read by state() at :3023). The graded live counterpart is
    # `connectome_propagation` (v4.79).
    "connectome_reverberation": "structural: topology only, computed once, cannot vary",
}


def _num(v, nd=4):
    """Round a value if it is a finite number; otherwise hand it back untouched (incl. None)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    return None if not math.isfinite(v) else round(float(v), nd)


_COVERAGE = None  # process-lifetime cache; see _coverage_census


def _coverage_census():
    """The Policy-3 module census, computed ONCE PER PROCESS.

    `region_wiring.live_coverage()` walks the package's static import graph. Measured cost: ~1.03 s,
    which was 100% of a snapshot. Calling it per snapshot meant the cockpit's 1 Hz poll forced a full
    AST scan every second INSIDE the daemon process -- starving the tick loop it exists to observe, and
    making responses slow enough that clients abandoned them mid-write.

    Caching is safe for a structural reason rather than a hopeful one: this is a census of the code on
    disk, and the daemon EXITS when git HEAD changes (learn_daemon's auto-update loop), so a cached
    value cannot outlive the code it describes. A process that somehow did outlive its code would
    report a stale census -- which is why this is a census of MODULES, never of live behaviour.
    """
    global _COVERAGE
    if _COVERAGE is None:
        try:
            from snn_data_lake.substrate import region_wiring

            _COVERAGE = {f"coverage_{k}": v for k, v in region_wiring.live_coverage().items()}
        except Exception:
            _COVERAGE = {}
    return _COVERAGE


def snapshot(brain):
    """The single telemetry contract: everything a panel may read, and nothing invented.

    Sections are kept separate so a reader can tell WHERE a number came from, which is most of what
    makes telemetry trustworthy:
      cognition  -- Brain.state(), the canonical ~105-key view
      spikes     -- LiveBrain.spike_snapshot(), 17 REAL per-region rates in [0,1] (region-resolved,
                    NOT cell-type-resolved; do not label these as neuron populations)
      body       -- ArmSim.state(). Carries `moves: False` because only `effector` currently changes.
      derived    -- arithmetic over the above, computed here so state() does not carry two
                    representations of the same fact
      gated      -- real variables their own source marks as never applied. `applied` is False.
    """
    out = {
        "t": time.time(),
        "cognition": {},
        "spikes": {},
        "body": {},
        "derived": {},
        "gated": {},
        "hardware": {},
    }

    try:
        st = brain.state()
        out["cognition"] = {k: _num(v) for k, v in st.items()}
    except Exception as e:  # telemetry must never take the brain down with it
        out["cognition"] = {"_error": str(e)}
        st = {}

    try:
        out["spikes"] = {k: _num(v) for k, v in brain.brain.spike_snapshot().items()}
    except Exception:
        out["spikes"] = {}

    arm = None
    try:
        arm = brain.brain.arm.arm.state()
        out["body"] = {k: _num(v) for k, v in arm.items()}
        # Stated, not hidden: the joints do not move on the canonical tick. A dashboard that let a
        # viewer assume otherwise would be the exact failure this module exists to prevent.
        # The body's own max horizontal radius, so a panel can scale a position without
        # hardcoding a reach that belongs to one particular arm.
        out["body"]["reach_mm"] = float(_arm_config.MAX_REACH_MM)
        out["body"]["moves"] = bool(
            any(abs(float(a)) > 1e-9 for a in arm.get("theta_deg", []) or [])
        )
    except Exception:
        out["body"] = {}

    # The viewer's own honesty row. `arm_view_revision` is the point of the section: a frozen
    # renderer and a stationary arm produce the same picture, so the counter is what tells them
    # apart. Reported even when unavailable, with the reason, rather than an absent panel.
    try:
        out["arm_view"] = brain.brain.arm.arm.view.status()
    except Exception as e:
        out["arm_view"] = {
            "arm_view_revision": 0,
            "arm_view_available": False,
            "arm_view_reason": f"no arm view: {type(e).__name__}",
        }

    # ---- derived (arithmetic only; no new state) ----
    pos = st.get("map_position") or []
    can = st.get("grid_can_xy") or []
    try:
        if len(pos) >= 2:
            out["derived"]["distance_from_origin"] = _num(math.hypot(float(pos[0]), float(pos[1])))
        if len(pos) >= 2 and len(can) >= 2:
            out["derived"]["place_error"] = _num(
                math.hypot(float(can[0]) - float(pos[0]), float(can[1]) - float(pos[1]))
            )
    except Exception:
        pass

    # affect arrives as the pair `affective_appraisal: [valence, arousal]`; the circumplex wants the
    # two axes separately. Unpack rather than teach the frontend the tuple's index order.
    app = st.get("affective_appraisal") or []
    if len(app) >= 2:
        out["derived"]["valence"] = _num(app[0])
        out["derived"]["arousal"] = _num(app[1])

    # ---- the closed loop: what the arm has actually achieved ----------------------------------
    # These live on AutonomousArm and are written by coverage_reward, the canonical wake reward_fn.
    # They are the only signals in this whole snapshot that answer "is it getting better", so they
    # are read straight off the object rather than inferred.
    arm_obj = getattr(getattr(brain, "brain", None), "arm", None)
    if arm_obj is not None:
        cov = getattr(arm_obj, "_covered", None)
        if cov is not None:
            # NOT also emitting `_cover_new`: it is incremented in the same `if fresh:` block that
            # adds to `_covered` and is never reset anywhere in src/, so the two are the same number.
            # Two panels reading one counter would imply an independent measurement that is not there.
            out["derived"]["cells_covered"] = len(cov)
        for key, attr in (
            ("recoveries", "_recoveries"),
            ("ground_ticks", "_ground_ticks"),
            ("cpg_weight", "_cpg_w"),
            ("cpg_spread_max", "_cpg_spread_max"),
            ("td_flipped", "_td_flipped"),
            ("ignitions", "_ignitions"),
            # Displacement, deliberately NOT called speed: ArmSim has no dt, no mass and no dynamics
            # (arm_hardware.SIM_GAP.sim_has_dynamics is False), so dividing by a timestep would invent
            # a velocity the simulator cannot support. This is the chord length of the last scored
            # tick, in sim units.
            #
            # Read from `_last_disp`, NOT recomputed here from `_last_ee`. The first version did the
            # latter and reported 0.000 forever: coverage_reward overwrites `_last_ee` with the
            # CURRENT pose before returning, so by snapshot time the two points are the same one and
            # the distance between them is necessarily zero. It rendered as a plausible live readout.
            ("displacement", "_last_disp"),
        ):
            if hasattr(arm_obj, attr):
                out["derived"][key] = _num(getattr(arm_obj, attr))

        wm = getattr(arm_obj, "lake_wm", None)
        if wm is not None:
            try:
                out["derived"]["lake_events"] = _num(wm.n_events())
            except Exception:
                pass

    # ---- Policy-3 census. `reachable` is REACHABILITY, not proof a faculty charges behaviour; the
    # policy is explicit that the two are different, so the key keeps the gate's own word for it.
    out["derived"].update(_coverage_census())

    # ---- hardware TARGET: a plan, not a measurement. `built` is False and the UI must say so. ----
    # Kept in its own section precisely so it can never be mistaken for body telemetry: `body` above
    # is the live 2-DOF sim, this is the 6-DOF arm that does not exist yet.
    try:
        from snn_data_lake.brainstem.arm_hardware import state as _hw

        out["hardware"] = _hw()
    except Exception:
        out["hardware"] = {}

    # ---- gated: NOT routed to any action -- and, it turns out, not computed either. ----
    # These were labelled "computed but not applied" until an audit checked the write sites: all three
    # are assigned only inside blocks guarded by `_dev_on` (autonomous_loop.py:1714) and `_gain_on`
    # (:1751), which are set from the `develop_env` / `gain_adapt` kwargs at :404 and :457.
    #
    # The evidence that Brain never enables them is an ABSENCE, so it is stated as one rather than
    # cited to a line: re-derived 2026-08-04, `cortex/brain.py` contains NO occurrence of either
    # `gain_adapt` or `develop_env`, tree-wide the only `develop_env=True` sites are inside
    # `cortex/live_brain.py`'s own `main()` (:809, :838), and `gain_adapt` is never passed True
    # anywhere. (The old cite here, `brain.py:197-209`, points at the NORD_PERSIST block and never
    # mentioned either kwarg -- an absence cannot have a line number, which is presumably how it
    # acquired the wrong one.) So they hold their init 0.0 forever. `computed` records that, because
    # "a gain that is never applied" and "a gain that was never calculated" are different failures
    # and the second one is worse.
    a = arm_obj
    if a is not None:
        for key, attr, flag in (
            ("motor_gain", "_motor_gain", "_gain_on"),
            ("fm_gain", "_dev_fm_gain", "_dev_on"),
            ("competence_ema", "_comp_ema", "_dev_on"),
        ):
            if hasattr(a, attr):
                out["gated"][key] = {
                    "value": _num(getattr(a, attr, 0.0)),
                    "applied": False,
                    "computed": bool(getattr(a, flag, False)),
                }

    out["provenance"] = dict(PROVENANCE)
    return out


def verify_contract(snap):
    """Return every dashboard key the snapshot cannot supply. Empty list == no orphan panels.

    Widened deliberately. The first version built `have` from cognition|derived only, so a panel drawn
    from spikes.*, body.* or hardware.* had no gate behind it whatsoever -- exactly the silently-dead
    panel this function exists to prevent, in the three sections most likely to move.
    """
    have = set(snap.get("cognition", {})) | set(snap.get("derived", {}))
    missing = [k for k in DASHBOARD_KEYS if k not in have]
    for section, keys in DASHBOARD_SECTION_KEYS.items():
        present = set(snap.get(section, {}))
        missing += [f"{section}.{k}" for k in keys if k not in present]
    return missing


# ---- Recorder rotation thresholds: OPS parameters, not behaviour-arbitrating values. -------------
# data/telemetry.jsonl grew unbounded (69 -> 176 MB in one session) because Recorder only ever
# appended. These three constants bound it. P1's no-hardcode rule does not forbid them: they
# arbitrate DISK USAGE only -- no learning rule, gate, or measurement reads them -- so they are named
# ops knobs (like a log-rotation size), with the reasoning written down instead of a bare number.
#   ROTATE_BYTES: one full-fidelity segment. At the observed ~8.7 KB/snapshot and one snapshot per
#     ~5 s daemon cycle, 32 MiB is ~3,800 consecutive snapshots (~5 h). The recent window the
#     Recorder guarantees at full fidelity is active+prev = one to two segments.
#   TAIL_KEEP_1_IN: decimation stride. 1-in-10 keeps long-baseline statistics readable (the
#     20,023-snapshot valence==0.0 finding survives a 1-in-10 sample) at a tenth of the bytes.
#   TAIL_MAX_BYTES: when the decimated tail itself exceeds this, it is re-decimated in place by the
#     same stride -- history gets 10x coarser per pass, but is never discarded outright.
# Worst-case steady-state disk: active(~ROTATE) + prev(~ROTATE) + tail(<TAIL_MAX + one decimated
# segment) ~= 131 MiB, vs unbounded before.
TELEMETRY_ROTATE_BYTES = 32 * 2**20
TELEMETRY_TAIL_KEEP_1_IN = 10
TELEMETRY_TAIL_MAX_BYTES = 64 * 2**20

_META_PREFIX = '{"_telemetry_meta"'  # decimation bookkeeping lines; always survive re-decimation


class Recorder:
    """Append-only JSONL of one snapshot per call, BOUNDED by rotation. Record first, render later.

    Deliberately dependency-free and deliberately not on the hot path by default -- the caller
    decides when to `write()`, so a daemon can record every tick and a chat app can record none.

    WHY BOUNDED, NOT RETIRED. This file has no reader in the tree, but it is read by ANALYSIS: the
    strongest measurements this project has (derived.valence == 0.0 on 20,023/20,023 snapshots;
    metacog_hold true on 74/20,023) came from exactly this file's long consecutive baseline. So the
    longitudinal record is kept, and only its byte count is bounded:
      <path>        ACTIVE segment: the newest snapshots, full fidelity
      <path>.prev   PREVIOUS segment: full fidelity, held until the next rotation consumes it
      <path>.tail   decimated history: 1-in-keep_1_in of every retired segment; re-decimated in
                    place when it exceeds tail_max_bytes (coarser each pass, never discarded)
    Decimation is deterministic (line index modulo stride, no RNG) and operates on raw lines (no
    JSON parse). Every pass appends a `_telemetry_meta` line recording stride and counts, and meta
    lines survive later passes, so the effective sampling stride of any stretch of tail stays
    reconstructible by an analysis.

    COST DISCIPLINE. The steady-state write path adds one integer add and one comparison. A rotation
    on the tick thread is close + os.replace + reopen (~ms). Segment decimation runs on a background
    daemon thread that only touches <path>.prev (which the writer never appends to again) and
    <path>.tail (which only the compactor touches) -- no lock, no writer race, by construction.
    A crash mid-compaction leaves <path>.prev behind; the next construction heals it. A crash after
    the tail append but before the .prev delete duplicates one decimated segment in the tail (meta
    lines make that visible); duplication is the failure chosen over data loss.
    """

    def __init__(
        self,
        path,
        every=1,
        rotate_bytes=TELEMETRY_ROTATE_BYTES,
        tail_max_bytes=TELEMETRY_TAIL_MAX_BYTES,
        keep_1_in=TELEMETRY_TAIL_KEEP_1_IN,
        threaded=True,
    ):
        self.path = str(path)
        self.every = max(1, int(every))
        self.n = 0
        self._fh = None
        self.rotate_bytes = int(rotate_bytes)
        self.tail_max_bytes = int(tail_max_bytes)
        self.keep_1_in = max(2, int(keep_1_in))
        self.threaded = bool(threaded)  # False: compaction runs inline (deterministic self-checks)
        self._compactor = None
        self._prev = self.path + ".prev"
        self._tail = self.path + ".tail"
        try:
            self._bytes = os.path.getsize(self.path)
        except OSError:
            self._bytes = 0
        # Construction runs at daemon startup, never on the tick path, so healing and legacy
        # adoption belong here rather than in write().
        if os.path.exists(self._prev):
            self._start_compaction()  # heal a crash that left a segment un-decimated
        elif self._bytes >= 2 * self.rotate_bytes:
            self._adopt_oversized()  # one-time adoption of a legacy unbounded file

    def write(self, brain):
        self.n += 1
        if self.n % self.every:
            return None
        snap = snapshot(brain)
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        line = json.dumps(snap, ensure_ascii=True) + "\n"
        self._fh.write(line)
        self._fh.flush()
        self._bytes += len(line)  # ensure_ascii=True -> 1 char == 1 byte; no encode() just to count
        if self._bytes >= self.rotate_bytes:
            self._maybe_rotate()
        return snap

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        c = self._compactor
        if c is not None and c.is_alive():
            # a clean shutdown finishes the decimation; an unclean one heals at next construction
            c.join(timeout=10)

    # ---- rotation (tick-thread side: cheap, never blocks on compaction) ----
    def _maybe_rotate(self):
        if self._compactor is not None:
            if self._compactor.is_alive():
                return  # previous segment still being decimated; retry on a later write
            self._compactor = None
        if os.path.exists(self._prev):
            # first trigger (or healed crash): decimate the old segment first; the rename happens
            # on a later write, once the compactor has consumed <path>.prev
            self._start_compaction()
            return
        try:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            os.replace(self.path, self._prev)
            self._bytes = 0
        except OSError as e:
            print(f"[telemetry] rotate skipped: {type(e).__name__}: {e}", flush=True)

    def _start_compaction(self):
        if self.threaded:
            self._compactor = threading.Thread(
                target=self._compact, name="telemetry-compact", daemon=True
            )
            self._compactor.start()
        else:
            self._compact()

    # ---- compaction (background side: touches only files the writer never writes) ----
    def _compact(self):
        """Decimate <path>.prev 1-in-keep_1_in into <path>.tail, then bound the tail. Errors are
        printed, never raised -- telemetry must never take the brain down with it."""
        try:
            kept = total = 0
            with (
                open(self._prev, encoding="utf-8") as src,
                open(self._tail, "a", encoding="utf-8") as dst,
            ):
                for i, ln in enumerate(src):
                    total += 1
                    if i % self.keep_1_in == 0:
                        dst.write(ln)
                        kept += 1
                dst.write(self._meta_line("decimate", total, kept))
            os.remove(self._prev)
            if os.path.getsize(self._tail) >= self.tail_max_bytes:
                self._redecimate_tail()
        except Exception as e:
            print(f"[telemetry] compaction err: {type(e).__name__}: {e}", flush=True)

    def _redecimate_tail(self):
        """The tail outgrew its bound: thin it in place by the same stride. Meta lines always
        survive, so every pass stays on the record and cumulative strides are reconstructible."""
        tmp = self._tail + ".tmp"
        kept = total = data_i = 0
        with (
            open(self._tail, encoding="utf-8") as src,
            open(tmp, "w", encoding="utf-8") as dst,
        ):
            for ln in src:
                total += 1
                if ln.startswith(_META_PREFIX):
                    dst.write(ln)
                    kept += 1
                    continue
                if data_i % self.keep_1_in == 0:
                    dst.write(ln)
                    kept += 1
                data_i += 1
            dst.write(self._meta_line("redecimate", total, kept))
        os.replace(tmp, self._tail)

    def _meta_line(self, event, src_lines, kept):
        rec = {
            "_telemetry_meta": {
                "event": event,
                "keep_1_in": self.keep_1_in,
                "src_lines": src_lines,
                "kept": kept,
                "t": time.time(),
            }
        }
        return json.dumps(rec, ensure_ascii=True) + "\n"

    def _adopt_oversized(self):
        """One-time adoption of a legacy unbounded file (it reached 176 MB before rotation existed):
        split it so the NEWEST rotate_bytes stay in the active segment at FULL fidelity and
        everything older becomes a previous segment, which the normal compactor then decimates into
        the tail. Runs at construction (daemon startup), never on the tick path."""
        try:
            cut = (
                self._bytes - self.rotate_bytes
            )  # legacy lines are ASCII: len(chars) == len(bytes)
            tmp = self.path + ".tmp"
            pos = 0
            with (
                open(self.path, encoding="utf-8") as src,
                open(self._prev, "w", encoding="utf-8") as old,
                open(tmp, "w", encoding="utf-8") as new,
            ):
                for ln in src:
                    (old if pos < cut else new).write(ln)
                    pos += len(ln)
            os.replace(tmp, self.path)
            self._bytes = os.path.getsize(self.path)
            self._start_compaction()
        except Exception as e:
            print(f"[telemetry] adopt skipped: {type(e).__name__}: {e}", flush=True)


def main():
    """Self-check: the contract holds against a real brain, the recorder round-trips, HTTP answers."""
    import tempfile

    from snn_data_lake.cortex.brain import make_brain

    # NORD_TORCH=0 dropped 2026-07-29 with the zero-dependency rule: this self-check now builds the
    # brain in the configuration users actually get.
    brain = make_brain(seed=0)
    for _ in range(12):
        brain.tick()

    snap = snapshot(brain)
    for sec in ("cognition", "spikes", "body", "derived", "gated"):
        assert sec in snap, f"snapshot missing section {sec}"
    assert len(snap["spikes"]) == 17, f"expected 17 regions, got {len(snap['spikes'])}"

    missing = verify_contract(snap)
    assert not missing, (
        f"no-orphan-panel gate: dashboard reads keys the brain cannot supply: {missing}"
    )

    # every gated readout must declare itself unapplied -- this is what stops it becoming theatre
    for k, v in snap["gated"].items():
        assert v["applied"] is False, (
            f"gated readout {k} claims applied=True; route it or relabel it"
        )
        # if a future build enables the flag, `computed` flips and the value stops being a frozen 0.0.
        # Until then this asserts the honest reading: not applied AND not calculated.
        if not v["computed"]:
            assert v["value"] in (0.0, None), (
                f"gated readout {k} is not computed but carries a non-zero value {v['value']!r} -- "
                "either it IS being written somewhere and `computed` is wrong, or the value is stale"
            )

    # the closed-loop signals must be present: they are the only ones that answer "is it improving"
    for k in ("cells_covered", "recoveries", "ground_ticks", "cpg_weight", "displacement"):
        assert k in snap["derived"], (
            f"derived.{k} missing -- the arm competence panel would be blank"
        )
    assert snap["derived"]["cells_covered"] >= 0, "coverage cell count went negative"

    # PRESENT IS NOT THE SAME AS ALIVE. The first `displacement` was computed from `_last_ee` AFTER
    # coverage_reward had already overwritten it with the current pose, so it was structurally pinned
    # at 0.0 -- in the contract, shaped like a reading, incapable of ever being one. A signal that can
    # only be zero has to fail here rather than render on the dashboard as a plausible live number.
    moved_once = False
    for _ in range(40):
        brain.tick()
        if (snapshot(brain)["derived"].get("displacement") or 0.0) > 0.0:
            moved_once = True
            break
    assert moved_once, (
        "derived.displacement never left 0.0 across 40 ticks -- either the arm genuinely stopped "
        "moving, or the signal is being read from a source already overwritten by the writer"
    )

    # provenance must ship, and must still flag the four signals the audit caught being drawn as live
    prov = snap.get("provenance") or {}
    for k in ("integration_competence", "pci", "spikes.midbrain", "spikes.substrate"):
        assert k in prov, (
            f"provenance lost its tag for {k}; it would render as a live reading again"
        )

    # recorder round-trips as JSONL. mkstemp, NOT a fixed name: Recorder APPENDS, so a shared path
    # meant two concurrent self-checks interleaved into one file and both failed the line count --
    # the same shared-resource bug as the hardcoded port, one layer down. A self-check must not
    # depend on being the only thing running.
    fd, tmp = tempfile.mkstemp(prefix="sndl_telemetry_selfcheck_", suffix=".jsonl")
    os.close(fd)
    rec = Recorder(tmp, every=1)
    for _ in range(3):
        brain.tick()
        rec.write(brain)
    rec.close()
    with open(tmp, encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    assert len(lines) == 3, f"recorder wrote {len(lines)} lines, expected 3"
    assert lines[0]["spikes"], "recorded snapshot has no spike section"
    os.remove(tmp)

    # ---- rotation gate: the recorder BOUNDS its file. Tiny thresholds, threaded=False so the
    # compaction runs inline and the check is deterministic. Three paths exercised: (a) legacy
    # ADOPTION (a fresh Recorder over an oversized file splits it, newest window kept full
    # fidelity), (b) steady-state ROTATE -> decimate-to-tail, (c) tail RE-decimation.
    fd, rpath = tempfile.mkstemp(prefix="sndl_telemetry_rotate_", suffix=".jsonl")
    os.close(fd)
    line_b = len(json.dumps(snapshot(brain), ensure_ascii=True)) + 1  # ~one snapshot line, bytes
    big = Recorder(rpath, every=1)  # defaults: rotate_bytes is huge here, so no rotation yet
    stamps = []
    for _ in range(10):
        brain.tick()
        stamps.append(big.write(brain)["t"])
    big.close()
    assert os.path.getsize(rpath) > 2 * (3 * line_b), "oversized fixture too small to adopt"
    # (a) adoption: reopen with a small segment size -> split at construction, old part decimated
    rec2 = Recorder(
        rpath,
        every=1,
        rotate_bytes=3 * line_b,
        tail_max_bytes=2 * line_b,
        keep_1_in=2,
        threaded=False,
    )
    tail_path, prev_path = rpath + ".tail", rpath + ".prev"
    assert not os.path.exists(prev_path), "adoption must leave no un-decimated .prev behind"
    assert os.path.exists(tail_path), "adoption must produce a decimated tail"
    with open(rpath, encoding="utf-8") as fh:
        active = [json.loads(x) for x in fh if x.strip()]
    assert active and active[-1]["t"] == stamps[-1], (
        "adoption must keep the NEWEST snapshots in the active segment at full fidelity"
    )
    # (b)+(c) steady state: keep writing; rotations rename, compactions decimate, tail re-decimates
    for _ in range(12):
        brain.tick()
        stamps.append(rec2.write(brain)["t"])
    rec2.close()
    sizes = {p: os.path.getsize(p) for p in (rpath, prev_path, tail_path) if os.path.exists(p)}
    assert sizes[rpath] <= 3 * line_b + 2 * line_b, f"active segment unbounded: {sizes}"
    data_lines, metas = [], []
    with open(tail_path, encoding="utf-8") as fh:
        for ln in fh:
            rec = json.loads(ln)
            (metas if "_telemetry_meta" in rec else data_lines).append(rec)
    events = [m["_telemetry_meta"]["event"] for m in metas]
    assert "decimate" in events, "no decimation pass ran"
    assert "redecimate" in events, "tail exceeded its bound but was never re-decimated"
    assert data_lines and len(data_lines) < len(stamps), "tail must be a strict sample, not a copy"
    assert all("spikes" in r for r in data_lines), "tail lines must be intact snapshots"
    for p in (rpath, prev_path, tail_path, rpath + ".tmp", tail_path + ".tmp"):
        if os.path.exists(p):
            os.remove(p)

    # No HTTP here any more -- serving is substrate/cockpit.py's job, and it self-checks its own
    # endpoints. This module is the data layer; a second server on the same default port was the
    # duplication that made "which dashboard am I looking at?" a real question.

    print(
        f"brain_telemetry OK: {len(snap['cognition'])} cognition keys, {len(snap['spikes'])} regions, "
        f"{len(snap['gated'])} gated readouts, {len(prov)} provenance tags, contract clean, "
        f"recorder round-trips"
    )


if __name__ == "__main__":
    main()
