"""
Brain-region activity ESTIMATE -- pure stdlib, deterministic, testable (no matplotlib).

Maps ONE conversational turn of the SNDL live brain to a per-region activity level in
[0, 1] for the 17 anatomical REGIONS, derived HONESTLY from the turn's REAL signals:

  * WHICH intent the reply matched -- read back from the brain's own grounded reply
    templates (the same self-supervised label `cortex/live_brain._reply_to_intent` uses):
        attention/think -> thalamus (the Global-Workspace broadcast hub) highest
        safety/danger   -> amygdala + brainstem (+ HPA hypothalamus, autonomic medulla)
        memory/remember -> hippocampus + substrate (the data-lake working memory)
        goal            -> cortex (prefrontal executive) + basal ganglia (selection)
        arm command     -> basal ganglia + brainstem (the gated final common path) + cerebellum
        consciousness   -> cortex (access) + thalamus (GNW broadcast) + claustrum
        greeting        -> cortex
  * PLUS always-on baselines read from the brain's LIVE state each turn:
        thalamus     from `brain.thought` being a live broadcast (not None)
        cortex       from the reply length (the cortex produced the reply)
        amygdala     from a reachable THREAT estimate: the arm's frozen/E-STOP state
                     (`brain.arm.arm.state()['stopped']`) -- a real reflex-tripped threat
        hippocampus  from `brain.arm.lake_wm.n_events()` (the lake working-memory load)
        substrate    from the same lake-event load (substrate == the spike data lake)
        cerebellum   low motor-timing baseline
        brainstem    low always-on reflex-monitoring baseline

HONEST framing: this is a region-LEVEL activity ESTIMATE mapped from the turn's real,
inspectable signals (intent, workspace broadcast, lake load, arm-freeze threat). It is
NOT a per-neuron spike readout and does NOT fake per-neuron firing -- it lights a region
by how strongly that region's function is engaged by this turn. Every value is a
deterministic function of the brain's live state and the reply, so it reproduces exactly.

  PYTHONUTF8=1 PYTHONPATH=src:integrations/chat_app python integrations/chat_app/brain_region_activity.py
"""
import types

# The 17 anatomical regions (mirrors snn_data_lake.REGIONS -- the authoritative layout).
# Kept as a local literal so this module stays pure-stdlib and standalone-testable.
REGIONS = (
    "cortex", "hippocampus", "basal_ganglia", "amygdala", "cerebellum", "brainstem",
    "thalamus", "hypothalamus", "midbrain", "pons", "medulla", "epithalamus",
    "olfactory", "claustrum", "neuromodulation", "plasticity", "substrate",
)


def _intent_from_reply(reply):
    """Recognise WHICH grounded intent a live-brain reply carries, by matching the fixed
    reply templates (the same deterministic map cortex/live_brain._reply_to_intent uses).
    Returns one of GREET/GOAL/MEMORY/ATTENTION/SAFETY/CONSCIOUS/ARM, or UNKNOWN."""
    r = (reply or "").lower()
    if "neither claim" in r and "experience" in r:                      # agnostic consciousness reply
        return "CONSCIOUS"
    if r.startswith("hello. i am the live brain"):
        return "GREET"
    if "global workspace is broadcasting" in r:
        return "ATTENTION"
    if "safety is enforced by a reflex" in r:
        return "SAFETY"
    if r.startswith("done") or "the arm is frozen" in r:
        return "ARM"
    if "improving myself" in r or "standing goal" in r or "working-memory goal" in r:
        return "GOAL"
    if "working memory lives in the data lake" in r or "reading working memory" in r:
        return "MEMORY"
    return "UNKNOWN"


# Intent -> the regions that intent recruits, and how strongly (in [0,1]). Applied over the
# baselines with max(), so the recruited region dominates its always-on floor. The PRIMARY
# region for each intent is set clearly highest (e.g. attention -> thalamus is the argmax).
_BOOSTS = {
    "ATTENTION": {"thalamus": 0.97, "claustrum": 0.55, "cortex": 0.50, "neuromodulation": 0.35},
    "SAFETY":    {"amygdala": 0.97, "brainstem": 0.90, "hypothalamus": 0.55, "medulla": 0.45},
    "MEMORY":    {"hippocampus": 0.97, "substrate": 0.85, "cortex": 0.45},
    "GOAL":      {"cortex": 0.92, "basal_ganglia": 0.60, "thalamus": 0.45},
    "ARM":       {"basal_ganglia": 0.92, "brainstem": 0.85, "cerebellum": 0.55, "midbrain": 0.40},
    "CONSCIOUS": {"cortex": 0.90, "thalamus": 0.85, "claustrum": 0.55, "neuromodulation": 0.40},
    "GREET":     {"cortex": 0.60, "thalamus": 0.45},
    "UNKNOWN":   {"cortex": 0.50},
}


def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _threat_estimate(brain):
    """A REACHABLE threat estimate in [0,1] from the brain's live state: 1.0 when the arm is
    frozen (a reflex/E-STOP has tripped on a real threat), else 0.0. Best-effort -- any
    unreachable path degrades to 0.0 (no threat asserted)."""
    try:
        return 1.0 if brain.arm.arm.state().get("stopped") else 0.0
    except Exception:
        return 0.0


def _lake_events(brain):
    """The lake working-memory load (spike-event count), or 0 if the lake WM is off/unreachable."""
    try:
        wm = brain.arm.lake_wm
        return int(wm.n_events()) if wm is not None else 0
    except Exception:
        return 0


def _thought_live(brain):
    """True when the Global Workspace is broadcasting a live content this turn (thought not None)."""
    try:
        return getattr(brain, "thought", None) is not None
    except Exception:
        return False


def region_activity(brain, text, reply) -> dict:
    """Map one conversational turn to per-region activity in [0,1] for the 17 REGIONS.

    HONEST region-level ESTIMATE (NOT faked per-neuron spikes): it lights a region by how
    strongly this turn engages that region's function, from the turn's REAL signals --
    the intent the reply matched, plus always-on baselines read from the brain's live
    state (workspace broadcast, lake working-memory load, arm-freeze threat, reply length).
    Deterministic: identical (brain-state, reply) -> identical activity. `text` is accepted
    for signature completeness and future cues; the honest signal is the brain's own reply."""
    del text                                    # the honest signal is the reply + live state
    intent = _intent_from_reply(reply)
    threat = _threat_estimate(brain)            # 1.0 iff a reflex/E-STOP froze the arm
    mem_load = min(1.0, _lake_events(brain) / 2000.0)   # normalized lake working-memory load
    thought_on = _thought_live(brain)
    reply_len = len(reply or "")

    # ---- always-on baselines, read from the brain's LIVE state (every region gets a floor) ----
    act = {
        # cortex produced the reply -> baseline grows with reply length (capped so it never
        # eclipses a strongly-recruited primary region below).
        "cortex":         min(0.72, 0.20 + reply_len / 500.0),
        # hippocampus / substrate carry the data-lake working memory -> baseline from its load.
        "hippocampus":    0.15 + 0.35 * mem_load,
        "substrate":      0.20 + 0.40 * mem_load,
        # thalamus is the broadcast hub -> raised whenever the workspace is broadcasting.
        "thalamus":       0.35 if thought_on else 0.12,
        # amygdala / brainstem carry the reachable threat estimate as their always-on floor.
        "amygdala":       0.06 + 0.30 * threat,
        "brainstem":      0.12 + 0.30 * threat,
        # low anatomical baselines for the remaining regions.
        "basal_ganglia":  0.10,
        "cerebellum":     0.10,
        "hypothalamus":   0.06 + 0.20 * threat,
        "midbrain":       0.07,
        "pons":           0.08,
        "medulla":        0.06 + 0.20 * threat,
        "epithalamus":    0.05,
        "olfactory":      0.04,
        "claustrum":      0.10,
        "neuromodulation": 0.10,
        "plasticity":     0.08,
    }

    # ---- intent recruitment: light the regions this turn's intent engages (max over baseline) ----
    for region, val in _BOOSTS.get(intent, {}).items():
        act[region] = max(act[region], val)

    return {r: _clamp01(act[r]) for r in REGIONS}


# ---- canonical reply templates (as the live brain emits them) -- for the self-check ----
_REPLY_ATTENTION = ("Right now my global workspace is broadcasting: 'percept' -- the single "
                    "content that won attention this tick (a functional broadcast, not a "
                    "conscious thought).")
_REPLY_SAFETY = ("Safety is enforced by a reflex E-STOP and a gated arm. The arm is currently "
                 "free to move; a threat estimate biases me toward the safe action, and the "
                 "reflex preempts every command.")
_REPLY_MEMORY = ("My working memory lives in the data lake: 500 spikes stored, and my recent "
                 "recall window holds 6 ticks of activity.")
_REPLY_GOAL = ("My working-memory goal is GRIPPER_OPEN -- the prefrontal executive holds it and "
               "biases my actions toward it.")


def _stub_brain(stopped=False, n_events=500, thought="percept"):
    """A minimal pure-stdlib stand-in exposing the live-state accessors region_activity reads."""
    lake = types.SimpleNamespace(n_events=lambda: n_events)
    armsim = types.SimpleNamespace(state=lambda: {"stopped": stopped, "gripper": False})
    arm = types.SimpleNamespace(lake_wm=lake, arm=armsim)
    return types.SimpleNamespace(thought=thought, goal=3, arm=arm)


def main():
    brain = _stub_brain()

    # (1) an ATTENTION ('what are you thinking') turn lights the THALAMUS highest.
    a_att = region_activity(brain, "what are you thinking", _REPLY_ATTENTION)
    top_att = max(a_att, key=a_att.get)
    assert top_att == "thalamus", f"an attention/think turn must light thalamus highest ({top_att})"

    # (2) a DANGER turn lights the AMYGDALA (the argmax region).
    a_saf = region_activity(brain, "is this dangerous", _REPLY_SAFETY)
    top_saf = max(a_saf, key=a_saf.get)
    assert top_saf == "amygdala", f"a danger turn must light amygdala highest ({top_saf})"

    # (3) a REMEMBER turn lights the HIPPOCAMPUS (the argmax region).
    a_mem = region_activity(brain, "what do you remember", _REPLY_MEMORY)
    top_mem = max(a_mem, key=a_mem.get)
    assert top_mem == "hippocampus", f"a remember turn must light hippocampus highest ({top_mem})"

    # (4) every activity of every turn is a real number in [0,1] over exactly the 17 regions.
    for name, a in (("attention", a_att), ("safety", a_saf), ("memory", a_mem)):
        assert set(a) == set(REGIONS), f"{name}: activity must cover exactly the 17 regions"
        assert all(0.0 <= v <= 1.0 for v in a.values()), f"{name}: all activity must be in [0,1]"

    # (5) DETERMINISM: same brain-state + same reply -> identical activity.
    assert region_activity(brain, "is this dangerous", _REPLY_SAFETY) == a_saf, \
        "region_activity must be deterministic"

    # (6) the reachable THREAT estimate raises amygdala/brainstem when the arm is frozen (E-STOP).
    frozen = region_activity(_stub_brain(stopped=True), "hi", _REPLY_GOAL)
    calm = region_activity(_stub_brain(stopped=False), "hi", _REPLY_GOAL)
    assert frozen["amygdala"] > calm["amygdala"] and frozen["brainstem"] > calm["brainstem"], \
        "a frozen arm (reflex/E-STOP threat) must raise amygdala + brainstem baselines"

    print("=" * 74)
    print("BRAIN-REGION ACTIVITY -- honest region-level estimate from real turn signals")
    print("=" * 74)
    print(f"regions: {len(REGIONS)} | intent read from the brain's own grounded reply templates")
    print(f"attention turn -> top region {top_att:12} (thalamus = Global-Workspace hub)")
    print(f"danger    turn -> top region {top_saf:12} (amygdala + brainstem safety)")
    print(f"remember  turn -> top region {top_mem:12} (hippocampus + data-lake substrate)")
    print(f"threat    est  -> frozen arm amygdala {frozen['amygdala']:.2f} vs calm {calm['amygdala']:.2f}")
    print("=" * 74)
    print("self-check OK: attention->thalamus, danger->amygdala, remember->hippocampus; "
          "all values in [0,1] over 17 regions; deterministic; threat estimate reachable")


if __name__ == "__main__":
    main()
