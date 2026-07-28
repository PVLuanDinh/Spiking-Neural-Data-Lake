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
       assigned only inside `Brain.__init__` (brain.py:352,361) and never again; `pci` is computed
       only inside the sleep tick and only when `nights % 4 == 0` (brain.py:2275); `spikes.midbrain`
       reduces to a constant 0.05 because its `novelty` term is written only under a `_salience_on`
       flag that is never enabled (autonomous_loop.py:1045); and `spikes.substrate` is the SAME
       expression as `spikes.hippocampus` (live_brain.py:244,265), so the two bars are byte-identical
       every tick and only one of them is a region.

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
import time

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
    "body": ("theta_deg", "ee", "effector", "collision", "deviation"),
    "hardware": ("built", "joints", "underpowered", "sim_dof", "supply_ok"),
}

# key -> why its value is not what a live bar would imply. The cockpit renders the tag beside the value.
# Deleting an entry is how a future build declares a signal fixed; adding one is how it stays honest.
# Every claim here was verified against the cited source line, not inferred.
PROVENANCE = {
    # assigned once inside Brain.__init__ and never touched by any tick (brain.py:352,361 -> :2606)
    "integration_competence": "frozen at construction",
    "integration_gain": "frozen at construction",
    "integration_retention": "frozen at construction",
    "integration_ok": "frozen at construction",
    # written only in the sleep tick; pci additionally only when nights % 4 == 0 (brain.py:2275,:2349)
    "pci": "sleep-only, refreshes every 4th night",
    "criticality": "sleep-only",
    # string/bool literals in state(), not readings (brain.py:2572,:2701)
    "goal": "hardcoded literal",
    "reflex_armed": "hardcoded literal",
    # spike-section rows that are not independent live regions
    "spikes.olfactory": "no input stream -- constant 0.0",
    "spikes.midbrain": "constant: its novelty term is never enabled",
    "spikes.substrate": "duplicate of hippocampus (same expression)",
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
        out["body"]["moves"] = bool(
            any(abs(float(a)) > 1e-9 for a in arm.get("theta_deg", []) or [])
        )
    except Exception:
        out["body"] = {}

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
    # are assigned only inside blocks guarded by `_gain_on` / `_dev_on` (autonomous_loop.py:1412,:1451),
    # and those flags come from LiveBrain kwargs that Brain never passes (brain.py:197-209). So they
    # hold their init 0.0 forever. `computed` records that, because "a gain that is never applied" and
    # "a gain that was never calculated" are different failures and the second one is worse.
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


class Recorder:
    """Append-only JSONL of one snapshot per call. The FLOG equivalent: record first, render later.

    Deliberately dependency-free and deliberately not on the hot path by default -- the caller
    decides when to `write()`, so a daemon can record every tick and a chat app can record none.
    """

    def __init__(self, path, every=1):
        self.path = str(path)
        self.every = max(1, int(every))
        self.n = 0
        self._fh = None

    def write(self, brain):
        self.n += 1
        if self.n % self.every:
            return None
        snap = snapshot(brain)
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(snap, ensure_ascii=True) + "\n")
        self._fh.flush()
        return snap

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def main():
    """Self-check: the contract holds against a real brain, the recorder round-trips, HTTP answers."""
    import os
    import tempfile

    from snn_data_lake.cortex.brain import make_brain

    os.environ.setdefault("NORD_TORCH", "0")
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
