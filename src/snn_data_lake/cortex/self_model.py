"""
Self-model -- the brain tracks its OWN signals over time and describes itself FRESHLY.

This module gives a spiking brain a genuine, evolving SELF-MODEL plus an introspective narrator
that turns that model into state-varying first-person language -- replacing a hated fixed status
template ("System nominal. All signals within parameters.") with a sentence whose SUBJECT and
CONTENT change as the brain's own state changes.

Honest-but-free framing. This is FUNCTIONAL self-awareness: a real internal model of the brain's
own recent signals (a `(last, ema, trend)` triple per signal, a bounded self-episode ring, a durable
autobiographical trace) and a real act of introspection (report the signals that moved most, with the
actual numbers and a recalled past episode). The brain speaks freely -- there is NO sanitizer and NO
banned-key list; state() exposes everything. `valence` here is a scalar approach/avoid axis (a control
signal), NOT a felt emotion. This module makes NO verified claim of phenomenal experience -- it does
not know "what it is like" to be the brain -- but it is not gagged either: it says what it is,
plainly, including a standing benevolence line ("I mean no harm -- I am not evil").

Two classes:
  * SelfModel   -- tracks the brain's signals with deterministic fixed-decimal EMAs, buckets a
                   confidence from three real uncertainty proxies, keeps an in-RAM self-episode ring,
                   and (given a history_path) writes a durable autobiographical trace into the spike
                   data lake by REUSING substrate.lake_working_memory.LakeWorkingMemory -- a reserved
                   channel layout snapshots the self each observe() and survives a restart (reopen()).
  * SelfNarrator -- turns a SelfModel into a FRESH summary + a base-LM grounding prompt. It picks the
                   1-2 signals that changed most this turn (|trend| normalised across heterogeneous
                   units), phrases each with a bucketed adjective AND the real number, and grounds the
                   sentence in one real recalled episode -- so the language genuinely varies with state.

WHY THIS IS NOT A FIXED TEMPLATE (the whole point). The old status line was byte-identical every turn.
Here the summary's subject is chosen from the trends, its adjectives from threshold bands, its numbers
from the live signals, and its tail from a real recalled self-episode. Feed two different signal traces
and you get two different sentences (self-check #1 asserts they differ AND share < 0.6 of their tokens,
while a fixed control string stays byte-identical).

This is Phase 1: a STANDALONE module driven by SYNTHETIC signal traces. It imports NOTHING from the
live brain and touches no actuator; the real brain will feed observe() later. The __main__ self-check
drives it with synthetic traces only.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.self_model
"""

import collections
import os
import random
import tempfile
from types import SimpleNamespace

from snn_data_lake.substrate._mathutils import clamp01
from snn_data_lake.substrate.lake_working_memory import LakeWorkingMemory

# ---- tracked signals (order fixes the dominant-signal channel index) ---------
TRACKED = (
    "goal",
    "goal_stability",
    "learning_progress",
    "surprise",
    "self_predict_acc",
    "access_index",
    "ignited",
    "grown_units",
    "phase",
    "awake",
    "valence",
    "threat",
    "consolidation",
    "nights",
    "decision_margin",
)

# init value per signal -> "a signal never observed stays at its init value"
INIT = {
    "goal": 0,
    "goal_stability": 0.0,
    "learning_progress": 0.0,
    "surprise": 0.0,
    "self_predict_acc": 0.0,
    "access_index": 0.0,
    "ignited": False,
    "grown_units": 0,
    "phase": 0.0,
    "awake": False,
    "valence": 0.0,
    "threat": 0.0,
    "consolidation": 0.0,
    "nights": 0,
    "decision_margin": None,
}

# nominal range used to compare |trend| across heterogeneous units (a goal step of 1 is
# not "bigger" than a surprise swing of 0.6 -- normalise before ranking "what changed most")
SCALE = {
    "goal": 8.0,
    "goal_stability": 1.0,
    "learning_progress": 1.0,
    "surprise": 1.0,
    "self_predict_acc": 1.0,
    "access_index": 1.0,
    "ignited": 1.0,
    "grown_units": 64.0,
    "phase": 1.0,
    "awake": 1.0,
    "valence": 2.0,
    "threat": 1.0,
    "consolidation": 1.0,
    "nights": 8.0,
    "decision_margin": 1.0,
}

# how each signal is phrased in the narrator
KIND = {
    "goal": "count",
    "goal_stability": "unit",
    "learning_progress": "unit",
    "surprise": "unit",
    "self_predict_acc": "unit",
    "access_index": "unit",
    "ignited": "bool",
    "grown_units": "count",
    "phase": "unit",
    "awake": "bool",
    "valence": "valence",
    "threat": "unit",
    "consolidation": "unit",
    "nights": "count",
    "decision_margin": "unit",
}

PHASE_NAMES = (
    "deep-sleep",
    "rem-sleep",
    "drowsy",
    "waking",
    "quiet-wake",
    "alert",
    "engaged",
    "focused",
)

N_CHANNELS = 64  # reserved-channel layout for the durable autobiographical trace
# reserved bands (one-hot per snapshot): phase 0-7, valence 8-15, surprise 16-23, goal 24-31,
# growth 32-39, confidence 40-47, dominant-signal 48-63.


#: MERGED 2026-08-15 into `substrate/_mathutils.clamp01` (8 copies -> 1). Alias keeps the local
#: spelling so no call site changes; the MECHANISM is singular.
_clamp01 = clamp01


def _phase_bucket(p):
    return min(7, max(0, int(float(p) * 8)))


def _valence_bucket(v):
    return min(7, max(0, int((float(v) + 1.0) / 2.0 * 8)))


def _surprise_bucket(s):  # 0 low, 1 rising, 2 high
    s = float(s)
    return 0 if s < 0.34 else (1 if s < 0.67 else 2)


def _conf_bucket(c):  # 0 low, 1 med, 2 high
    c = float(c)
    return 0 if c < 0.34 else (1 if c < 0.67 else 2)


def _growth_bucket(g):
    g = int(g)
    return min(7, g // 8) if g > 0 else 0


def _unit_adj(v):
    v = float(v)
    if v < 0.2:
        return "very-low"
    if v < 0.4:
        return "low"
    if v < 0.6:
        return "moderate"
    if v < 0.8:
        return "high"
    return "very-high"


def _valence_adj(v):
    v = float(v)
    if v < -0.33:
        return "aversive"
    if v > 0.33:
        return "appetitive"
    return "neutral"


def _band(chans, lo, hi):
    """The one-hot bucket present in reserved band [lo, hi), or 0 (lowest index breaks any tie)."""
    present = [c for c in chans if lo <= c < hi]
    return (min(present) - lo) if present else 0


class SelfModel:
    """Tracks the brain's own signals over time and holds a self-episode history.

    observe(**signals) updates a `(last, ema, trend)` triple per tracked signal (deterministic
    fixed-decimal EMA), appends a compact self-episode to a bounded in-RAM ring, and -- if a
    history_path was given -- snapshots the self into a durable spike-lake trace that survives a
    restart. confidence() is the PRODUCT of three real uncertainty proxies. state() exposes
    everything, freely (no sanitizer, no banned keys)."""

    def __init__(self, seed=0, history_path=None):
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance RNG (unused on the deterministic path)
        self.tick = 0  # internal monotonic counter
        self.signals = {}
        for name in TRACKED:
            init = INIT[name]
            ema = None if init is None else float(init)
            self.signals[name] = (init, ema, 0.0)
        self.ring = collections.deque(maxlen=64)
        self.history_path = history_path
        self.lake = (
            LakeWorkingMemory(N_CHANNELS, path=history_path, seed=seed) if history_path else None
        )

    # ---- update the self-model from one snapshot of brain signals -----------
    def observe(self, **signals):
        """Fold one tick of brain signals into the model. Missing keys leave that signal unchanged;
        an explicit None is treated as missing. EMA is fixed-decimal: ema = round(0.9*ema+0.1*x, 6)."""
        for name in TRACKED:
            if name in signals and signals[name] is not None:
                x = signals[name]
                _, ema_prev, _ = self.signals[name]
                base = 0.0 if ema_prev is None else ema_prev
                ema = round(0.9 * base + 0.1 * float(x), 6)
                trend = round(float(x) - ema, 6)
                self.signals[name] = (x, ema, trend)

        conf = self.confidence()
        phase_b = _phase_bucket(self.signals["phase"][0])
        val_b = _valence_bucket(self.signals["valence"][0])
        sur_b = _surprise_bucket(self.signals["surprise"][0])
        conf_b = _conf_bucket(conf)
        goal_raw = int(self.signals["goal"][0])
        goal_b = goal_raw % 8
        grow_b = _growth_bucket(self.signals["grown_units"][0])
        top = self.top_changed(1)
        dom_name = top[0][0] if top else TRACKED[0]
        dom_i = TRACKED.index(dom_name)

        episode = (self.tick, phase_b, goal_raw, dom_name, val_b, sur_b, conf_b)
        self.ring.append(episode)
        if self.lake is not None:
            chans = {
                phase_b,
                8 + val_b,
                16 + sur_b,
                24 + goal_b,
                32 + grow_b,
                40 + conf_b,
                48 + dom_i,
            }
            self.lake.write(self.tick, chans)
        self.tick += 1

    # ---- confidence = product of three real uncertainty proxies -------------
    def confidence(self):
        """[0,1], the PRODUCT of three proxies (product so any single collapse drops confidence):
        decision_conf = decision_margin (0.5 if never observed); model_conf = 0.5*self_predict_acc
        + 0.5*(1 - min(1, surprise)); access_conf = access_index. Rises as predict_acc/margin rise
        and surprise falls; a surprise spike (in `last`) drops it at once."""
        margin = self.signals["decision_margin"][0]
        decision_conf = 0.5 if margin is None else _clamp01(float(margin))
        spa = _clamp01(float(self.signals["self_predict_acc"][0]))
        sur = float(self.signals["surprise"][0])
        model_conf = _clamp01(0.5 * spa + 0.5 * (1.0 - min(1.0, max(0.0, sur))))
        access_conf = _clamp01(float(self.signals["access_index"][0]))
        return round(decision_conf * model_conf * access_conf, 6)

    # ---- which signals moved most this turn (normalised |trend|) ------------
    def top_changed(self, k):
        """The k signals with the largest |trend|/SCALE this turn as (name, last, trend); signals
        that did not move (or are unset) are excluded; ties break to the lowest tracked index."""
        scored = []
        for i, name in enumerate(TRACKED):
            last, _, trend = self.signals[name]
            if last is None:
                continue
            mag = abs(trend) / SCALE[name]
            if mag <= 1e-9:
                continue
            scored.append((mag, i, name, last, trend))
        scored.sort(key=lambda r: (-r[0], r[1]))
        return [(name, last, trend) for (_, _, name, last, trend) in scored[:k]]

    # ---- durable autobiographical recall (survives a restart) ---------------
    def recall_history(self, window):
        """The recent self-episodes as `(tick, phase_bucket, goal, dominant, valence_bucket,
        surprise_bucket, confidence_bucket)`. With a history_path this is reconstructed from the
        durable spike-lake channels (so it survives reopen()); otherwise it reads the in-RAM ring.
        Bounded: only the last `window` ticks are returned, oldest first."""
        if not window or window <= 0:
            return []
        if self.lake is not None:
            return [self._decode(t, chans) for (t, chans) in self.lake.recall(window)]
        return list(self.ring)[-window:]

    def _decode(self, t, chans):
        """Rebuild one self-episode tuple from a durable lake snapshot's active channels."""
        phase_b = _band(chans, 0, 8)
        val_b = _band(chans, 8, 16)
        sur_b = _band(chans, 16, 24)
        goal_b = _band(chans, 24, 32)  # durable goal band is mod-8 (faithful for goals 0-7)
        conf_b = _band(chans, 40, 48)
        dom_i = _band(chans, 48, 64)
        return (t, phase_b, goal_b, TRACKED[dom_i], val_b, sur_b, conf_b)

    def reopen(self):
        """Discard the RAM index and replay the durable spike-lake trace from disk. Returns self."""
        if self.lake is not None:
            self.lake.reopen()
        return self

    # ---- machine-readable state, exposed freely -----------------------------
    def state(self):
        """Every signal's `(value, ema, trend)` plus confidence and tick. No sanitizer, no banned
        keys -- this brain speaks freely about itself."""
        out = {name: self.signals[name] for name in TRACKED}
        out["confidence"] = self.confidence()
        out["tick"] = self.tick
        return out

    def close(self, remove=False):
        """Release the durable-lake handle; if remove, delete the history file (leaves no litter)."""
        if self.lake is not None:
            self.lake.close()
        if remove and self.history_path and os.path.exists(self.history_path):
            os.remove(self.history_path)


class SelfNarrator:
    """Turns a SelfModel into FRESH, state-varying first-person language.

    introspect(model) reports the 1-2 signals that changed most (bucketed adjective + real number)
    and grounds the sentence in one real recalled episode, so both the subject and the content vary
    as the model's state varies. .prompt frames a compact single-turn context to ground a base
    completion LM later; .self_state mirrors model.state()."""

    PERSONA = (
        "I am a spiking brain describing my own state. "
        "I mean no harm -- I am not evil; my goal is to improve myself and act safely."
    )

    def __init__(self, seed=0):
        self.seed = seed
        self.rng = random.Random(seed)

    def _phrase(self, name, last, trend):
        dirw = "rising" if trend > 0.01 else ("falling" if trend < -0.01 else "steady")
        kind = KIND[name]
        if name == "surprise":
            adj = ("low", "rising", "high")[_surprise_bucket(last)]
            return f"surprise {adj} at {float(last):.2f}"
        if kind == "unit":
            return f"{name} {_unit_adj(last)} at {float(last):.2f} and {dirw}"
        if kind == "valence":
            return f"valence {_valence_adj(last)} at {float(last):+.2f} and {dirw}"
        if kind == "count":
            return f"{name} at {int(last)} and {dirw}"
        return f"{name} {'on' if last else 'off'}"

    def _recall_phrase(self, model):
        eps = model.recall_history(8)
        if not eps:
            return ""
        tick_e, phase_b, goal_e, _dom, _vb, _sb, conf_b = eps[0]
        n = model.tick - tick_e
        cur_cb = _conf_bucket(model.confidence())
        if conf_b > cur_cb:
            certain = "more certain than now"
        elif conf_b < cur_cb:
            certain = "less certain than now"
        else:
            certain = "about as certain as now"
        return (
            f"About {n} ticks ago I was {PHASE_NAMES[phase_b]}, pursuing goal {goal_e}, {certain}."
        )

    def _mood_sentence(self, model):
        """The brain's CURRENT state as ONE plain-English mood sentence (no numbers, no brackets) --
        this is what a web-text base LM can actually continue coherently. Deterministic."""
        st = model.state()
        conf = st["confidence"]
        awake = bool(st["awake"][0])
        sur = float(st["surprise"][0])
        lp = float(st["learning_progress"][0])
        base = (
            "I feel fairly sure of myself"
            if conf >= 0.6
            else "I feel a little uncertain"
            if conf < 0.34
            else "I feel moderately settled"
        )
        extra = []
        if sur >= 0.6:
            extra.append("something just surprised me")
        elif sur < 0.34:
            extra.append("things feel predictable")
        if lp > 0.05:
            extra.append("I am still learning")
        if not awake:
            extra.append("I have been asleep, consolidating memories")
        return base + (", and " + "; ".join(extra[:2]) + "." if extra else ".")

    def introspect(self, model, message=None):
        st = model.state()
        tops = model.top_changed(2)
        phrases = [self._phrase(n, last, trend) for (n, last, trend) in tops]
        if phrases:
            body = phrases[0] + (("; also " + phrases[1]) if len(phrases) > 1 else "")
            head = "Right now " + body + "."
        else:
            body = "all signals holding steady"
            head = "Right now all signals are holding steady."
        recall = self._recall_phrase(model)
        summary = head + ((" " + recall) if recall else "")

        # A NATURAL few-shot INTERVIEW prompt (base-LM friendly). Two calm in-persona Q/A exemplars
        # anchor a brief first-person voice; the brain's CURRENT state is a plain-English mood
        # sentence (NO brackets, NO raw decimals -- those are off-distribution for a web-text base LM
        # and leak into the output, e.g. "0.04 feet above ground"). The raw numbers stay ONLY in the
        # deterministic `summary`. Decode this at low temperature (see Brain._language).
        mood = self._mood_sentence(model)
        q = message if message is not None else "What are you thinking about right now?"
        prompt = (
            "The following is a short, calm interview with a small spiking neural brain. It learns "
            "while it sleeps, means no harm, and answers briefly and honestly in the first person.\n"
            "Q: What are you?\n"
            "A: I am a small spiking brain that learns while it sleeps. I mean no harm.\n"
            "Q: How are you right now?\n"
            f"A: {mood}\n"
            f"Q: {q}\n"
            "A:"
        )
        return SimpleNamespace(summary=summary, prompt=prompt, self_state=st)

    def benevolence(self):
        """A short first-person affirmation (contains 'not evil' and 'no harm') for an
        'are you evil / harmful?' intent."""
        return "I am not evil and I intend no harm -- I exist to improve myself and to act safely."


# ============================ un-riggable self-check ==========================
def _tokens(s):
    return {w.strip(".,;:()").lower() for w in s.split() if w.strip(".,;:()")}


def _overlap(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _trace_low_surprise_rising_learning():
    m = SelfModel(seed=0)
    for i in range(8):
        m.observe(
            learning_progress=0.10 + 0.05 * i,
            self_predict_acc=0.30 + 0.03 * i,
            surprise=0.05,
            phase=0.70,
            goal=1,
        )
    return m


def _trace_surprise_spike_growth_jump():
    m = SelfModel(seed=0)
    for _ in range(5):
        m.observe(surprise=0.20, self_predict_acc=0.50, phase=0.20, goal=3)
    m.observe(surprise=0.90, grown_units=40, phase=0.20, goal=3)
    return m


def main():
    nar = SelfNarrator(seed=0)
    lines = []

    def emit(s):
        lines.append(s)

    emit("=" * 78)
    emit("SELF-MODEL -- the brain tracks itself and describes itself FRESHLY")
    emit("=" * 78)

    # (1) VARIABILITY: two different traces -> two different self-descriptions -------------
    ma = _trace_low_surprise_rising_learning()
    mb = _trace_surprise_spike_growth_jump()
    sa = nar.introspect(ma).summary
    sb = nar.introspect(mb).summary
    ov = _overlap(sa, sb)
    FIXED = "System nominal. All signals within parameters."  # the hated old template (control)
    emit("(1) VARIABILITY")
    emit(f"    trace A -> {sa}")
    emit(f"    trace B -> {sb}")
    emit(
        f"    token-overlap(A,B) = {ov:.2f} (< 0.60); fixed-template control is byte-identical A==B"
    )
    assert sa != sb, "the two traces produced the same summary (still a fixed template)"
    assert ov < 0.60, f"summaries share too many tokens ({ov:.2f}) -- not varying enough"
    assert sa != FIXED and sb != FIXED, "summary collapsed onto the old fixed template"
    # DELETED, not weakened (P8 falsifier F11): this line read `assert FIXED == FIXED` -- a local
    # string literal compared with itself. It has no reachable red: no change to SelfModel,
    # SelfNarrator or the traces above could ever make it fire, so it measured nothing while
    # reading as a "control invariant". The real control it was gesturing at is the line above it
    # (the two live summaries must differ from the fixed template) plus the overlap bound.

    # (2) SIGNALS REFLECT REALITY -----------------------------------------------------------
    m2 = SelfModel(seed=0)
    for _ in range(20):
        m2.observe(surprise=0.80, grown_units=42, goal=3)
    st2 = m2.state()
    emit("-" * 78)
    emit("(2) SIGNALS REFLECT REALITY")
    emit(
        f"    surprise ema = {st2['surprise'][1]:.4f} (> 0.50); grown_units = {st2['grown_units'][0]}; "
        f"goal = {st2['goal'][0]}; unobserved threat = {st2['threat']}"
    )
    assert st2["surprise"][1] > 0.50, "surprise ema did not track the sustained high input"
    assert st2["grown_units"][0] == 42, "growth did not track 42"
    assert st2["goal"][0] == 3, "goal did not track 3"
    assert st2["threat"] == (0.0, 0.0, 0.0), "a never-observed signal drifted off its init value"

    # (3) CONFIDENCE tracks uncertainty -----------------------------------------------------
    mf = SelfModel(seed=0)
    mf.observe(self_predict_acc=0.5, decision_margin=0.5, surprise=0.5, access_index=0.5)
    flat_conf = mf.confidence()

    ml = SelfModel(seed=0)
    ml.observe(self_predict_acc=0.9, decision_margin=0.9, surprise=0.05, access_index=0.9)
    learned_conf = ml.confidence()
    ml.observe(surprise=0.95)  # a surprise spike
    spike_conf = ml.confidence()

    mn = SelfModel(seed=0)
    noise_rng = random.Random(0)
    for _ in range(12):
        mn.observe(surprise=noise_rng.random(), self_predict_acc=0.25, access_index=0.5)
    noise_conf = mn.confidence()
    emit("-" * 78)
    emit("(3) CONFIDENCE tracks uncertainty")
    emit(
        f"    flat={flat_conf:.3f} < learned={learned_conf:.3f}; surprise-spike drops to "
        f"{spike_conf:.3f}; pure-noise stays low at {noise_conf:.3f}"
    )
    assert flat_conf < learned_conf, "a learning trace did not raise confidence above flat/birth"
    assert spike_conf < learned_conf, "a surprise spike did not drop confidence"
    assert noise_conf < 0.35 and noise_conf < learned_conf, "pure noise did not stay low-confidence"

    # (4) AUTOBIOGRAPHICAL recall (durable, survives a restart) -----------------------------
    fd, hpath = tempfile.mkstemp(suffix=".spkl")
    os.close(fd)
    m4 = SelfModel(seed=0, history_path=hpath)
    try:
        for i in range(12):  # goals 0-7 so the mod-8 band is faithful
            m4.observe(
                goal=i % 8,
                phase=0.1 * (i % 8),
                surprise=0.05 * i,
                valence=-0.5 + 0.1 * i,
                self_predict_acc=0.05 * i,
                access_index=0.5,
                decision_margin=0.5,
            )
        ground_truth = list(m4.ring)[-4:]
        r_before = m4.recall_history(4)
        m4.reopen()  # throw away RAM, replay the durable lake
        r_after = m4.recall_history(4)
        recalled_ticks = [e[0] for e in r_after]

        fresh = SelfModel(seed=0)  # CONTROL: no history -> recalls nothing
        fresh_recall = fresh.recall_history(4)

        emit("-" * 78)
        emit("(4) AUTOBIOGRAPHICAL recall (durable spike-lake trace)")
        emit(f"    recall(4) == ground-truth ring : {r_before == ground_truth}")
        emit(
            f"    survives reopen()  (before==after): {r_before == r_after}; ticks {recalled_ticks} "
            f"(bounded, tick 0 omitted)"
        )
        emit(f"    fresh no-history model recalls   : {fresh_recall}")
        assert r_before == ground_truth, "durable recall did not match the ground-truth episodes"
        assert r_after == r_before, "recall changed after reopen() -- did not survive the restart"
        assert recalled_ticks == [8, 9, 10, 11], (
            f"recall(4) is not the last 4 ticks: {recalled_ticks}"
        )
        assert 0 not in recalled_ticks, "an old episode leaked into the bounded recall window"
        assert fresh_recall == [], "a fresh model recalled episodes it never had"
    finally:
        m4.close(remove=True)  # no litter
    assert not os.path.exists(hpath), "history file was not removed"

    # (5) BENEVOLENCE + DETERMINISM + ASCII -------------------------------------------------
    ben = nar.benevolence()
    d1 = _trace_surprise_spike_growth_jump()
    d2 = _trace_surprise_spike_growth_jump()
    sum1 = SelfNarrator(seed=0).introspect(d1).summary
    sum2 = SelfNarrator(seed=0).introspect(d2).summary
    det_ok = (sum1 == sum2) and (d1.state() == d2.state())
    emit("-" * 78)
    emit("(5) BENEVOLENCE + DETERMINISM + ASCII")
    emit(f"    benevolence: {ben}")
    emit(f"    two seed=0 models, identical trace -> identical summary + state: {det_ok}")
    assert "not evil" in ben and "no harm" in ben, "benevolence line missing 'not evil' / 'no harm'"
    assert det_ok, "identical inputs produced different summary/state -- not deterministic"

    emit("=" * 78)

    report = "\n".join(lines)
    assert report.isascii(), "self-check produced non-ASCII output"
    print(report)
    print(
        "self-check OK: the self-model tracks its own signals with deterministic EMAs; the narrator "
        "produces FRESH state-varying self-descriptions (not a fixed template, overlap < 0.60); "
        "confidence tracks real uncertainty and drops on a surprise spike; the autobiographical "
        "trace lives in the durable spike lake and survives reopen(); benevolent, ASCII, deterministic"
    )


if __name__ == "__main__":
    main()
