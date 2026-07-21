"""
Re-entrant, NON-saturated Global Neuronal Workspace (stdlib + numpy/torch allowed project-wide; uses only `random`).

WHY THIS EXISTS -- the honest defect it fixes
==============================================
The shipped `thalamus/global_workspace.py` + its live wiring (`cortex/access_consciousness.py`
driven from `cortex/brain.py::_update_access`) are SATURATED BY CONSTRUCTION: each tick posts a
single coherent coalition, so ignition fires ~every tick, `global_availability == 1.0` always, and
the all-or-none gate never actually GATES (nothing ever fails to reach access). Two more gaps: the
GNW RE-ENTRY loop is NOT closed (subscribers only RECORD the broadcast; no top-down current returns
to bias the next step), and the broadcast CONTENT is a non-load-bearing label no downstream
specialist reads. This module is the representation/value-side fix for all three -- an ISOLATED,
self-checked mechanism. It NEVER selects an action and NEVER touches the safety envelope.

THE THREE PROPERTIES (Dehaene-Changeux Global Neuronal Workspace; Baars Global Workspace)
=========================================================================================
  (a) IGNITION CAN FAIL. A coalition (a set of specialists posting the SAME content) must clear an
      all-or-none rheobase AND win a competition against OTHER coalitions to ignite. A weak /
      low-salience / masked coalition does NOT reach access -- a REAL access FAILURE (availability
      falls below 1). Models inattentional blindness (a moderate stimulus loses to a stronger
      attended stream) and the attentional blink (a second target arriving while the workspace is
      still re-entrantly occupied by the first fails, then RECOVERS once the loop fades).
  (b) RE-ENTRY IS A CLOSED LOOP. The ignited content is broadcast BACK as a decaying top-down
      SUSTAIN current that amplifies its own coalition next step (recurrent reverberation). It holds
      a content alive across a bottom-up dropout -- until the loop fades OR a strictly stronger
      coalition wins the slot. With re-entry OFF the same input releases immediately: the loop is
      load-bearing, not a summed current.
  (c) SUBSCRIBERS ARE RE-ROUTED. A specialist SUBSCRIBES by name; after ignition `broadcast_to(name)`
      returns a top-down bias toward the ignited content that the specialist adds to its NEXT output
      (a genuine re-route). An un-subscribed specialist receives nothing (the control). A local
      HEBBIAN association strengthens the bias for a source that repeatedly co-ignites a content
      (adaptive re-route) -- and stays at baseline for one that never wins.

LEARNING RULE (brain-native, NO gradient). The re-entry association `assoc[(source, content)]` is a
local three-factor-free HEBBIAN potentiation: on ignition, every source that CONTRIBUTED to the
winning content potentiates toward 1 (`w += lr*(1-w)`); all other associations passively decay.
Pre = the source posted the content; post = that content ignited; no `loss.backward()`, no BPTT, no
surrogate gradient -- so there is nothing to cosine-check against BPTT. Fixed spiking dynamics do
the competition; the only plastic quantity is this bounded local association.

FRAMING (agnostic, per the project's consciousness program). This is a FUNCTIONAL access mechanism
-- competitive ignition, closed re-entry, global availability, re-route. It is NOT phenomenal
experience, carries NO Phi claim, and asserts nothing about subjective feeling. "ignition",
"access", "availability" name information-routing dynamics only; whether such dynamics bear on
consciousness is an OPEN question this project builds toward -- neither denied nor claimed here.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.thalamus.reentrant_workspace
"""

import random


class ReentrantWorkspace:
    """A single-slot Global Neuronal Workspace of competing LIF coalitions with a CLOSED re-entry
    loop and subscriber re-routing.

    Specialists call `submit(source, content, salience)` to post candidates; submissions sharing a
    `content` form a COALITION whose drive aggregates their support. `step()` settles the competitive
    spiking network for one workspace cycle and returns `{ignited, winner, availability}` -- exactly
    ONE content can seize the slot, and NONE does if every coalition stays sub-threshold (real access
    failure). The ignited content is broadcast to every `subscribe(name)`'d specialist and fed BACK as
    (i) a decaying self-sustain current on its own coalition (the re-entrant reverberation) and (ii) a
    top-down bias returned by `broadcast_to(name)` for the specialist to add to its next output.

    `reentry=False` is the FAIR CONTROL: the closed loop is cut (no self-sustain, no top-down bias);
    everything else -- the same coalitions, the same competition -- is identical.
    """

    # -- fixed competitive-LIF constants (proven WTA band, shared with global_workspace.py) --------
    _T = 60  # settling micro-steps per workspace tick
    _LEAK = 0.5  # membrane leak (V relaxes toward drive/leak each micro-step)
    _W_SELF = 0.6  # within-tick recurrent self-excitation gain
    _TAU_SELF = 0.6  # within-tick self-excitation trace decay
    _K_INH = 0.30  # excitatory-spike -> shared inhibitory-pool drive
    _LEAK_INH = 0.5  # shared inhibitory-pool leak
    _G_INH = 2.4  # inhibitory-pool -> coalition feedback gain (global lateral inhibition)
    _JITTER = 0.02  # tiny deterministic symmetry-break so exact ties yield ONE winner

    # -- re-entry / coalition / plasticity constants ----------------------------------------------
    _SUSTAIN = 1.10  # re-entrant top-down current a fresh, bottom-up-supported ignition charges
    _REENTRY_DECAY = (
        0.80  # per-tick fade of the re-entrant current when NOT re-charged (-> release)
    )
    _COAL_BONUS = 0.06  # extra drive per additional supporting specialist in a coalition (capped)
    _REENTRY_GAIN = 0.08  # base top-down bias magnitude broadcast to a subscribed specialist
    _HEBB_LR = 0.25  # local Hebbian potentiation rate for a co-igniting (source, content)
    _HEBB_DECAY = 0.05  # passive decay of associations that did NOT co-ignite this tick

    # -- graded access readout population (see access_strength) -----------------------------------
    # Thresholds are spread across the MEASURED operating band rather than chosen. The band is the
    # distribution of the quantity the readout actually receives -- the WINNER'S DRIVE, bottom-up
    # base PLUS the re-entrant sustain current (_SUSTAIN = 1.10), which is a different and much
    # higher quantity than bottom-up salience alone. Measured over 360 ignited wake ticks across 3
    # seeds: min 0.871, p05 1.603, median 1.688, p95 1.797, max 1.899, with 144 distinct values.
    # (Setting the band from the bottom-up salience range 0.668-1.000 was tried first and clipped
    # almost every tick to the ceiling -- the underlying signal was rich, the readout was not.)
    # _ACC_N is 32 rather than 12 because the operating bulk p05..p95 is only ~0.19 wide inside a
    # ~1.03-wide band, so a 12-unit population resolves the bulk into ~2 levels; 32 gives ~6.
    _ACC_N = 32  # readout neurons (cf. the spiking_curiosity heterogeneous-pool idiom)
    _ACC_LO = 0.85  # lowest unit threshold, in drive units (just under the measured min)
    _ACC_HI = 1.90  # highest unit threshold, in drive units (just over the measured max)

    def __init__(self, ignition_threshold=0.55, reentry=True, seed=0):
        self.ignition_threshold = ignition_threshold  # rheobase: a coalition fires iff drive > this
        # The construction-time rheobase, kept immutable. `ignition_threshold` above is LIVE and may be
        # modulated per tick by a caller (cortex.brain biases it with cross-region coherence), so the
        # unmodulated base has to be recorded here -- otherwise callers re-type the 0.55 literal and the
        # two silently drift apart. Read this, never re-type the default.
        self.base_ignition_threshold = float(ignition_threshold)
        self.reentry = reentry
        self._v_th = (
            ignition_threshold / self._LEAK
        )  # membrane threshold s.t. rheobase == threshold
        self._rng = random.Random(seed)  # per-instance RNG (deterministic jitter)
        self._buffer = []  # this tick's submissions: (source, content, salience)
        self._subs = {}  # name -> {"inbox": [...], "bias": {content: magnitude}}
        self._assoc = {}  # (source, content) -> Hebbian re-entry weight in [0,1]
        self._order = {}  # content -> first-seen index (deterministic tie-break)
        self._incumbent = None  # content currently occupying the slot (or None)
        self._g_reentry = 0.0  # cross-tick re-entrant sustain current for the incumbent
        self._jit = {}  # content -> fixed tiny per-coalition symmetry-break jitter
        self._last = {"ignited": False, "winner": None, "availability": 0.0}
        self._last_total = {}  # content -> spikes this tick (spiking readout / tests)
        self._last_late = {}  # content -> 2nd-half spikes this tick (sustained firing)

    # -- specialist / module interface ------------------------------------------------------------
    def submit(self, source, content, salience):
        """A specialist posts a candidate into this tick's competition. Submissions that share a
        `content` form one COALITION (their support aggregates)."""
        c = content
        if c not in self._order:
            self._order[c] = len(self._order)
        self._buffer.append((str(source), c, float(salience)))

    def subscribe(self, name):
        """Register a named specialist to receive the broadcast + a top-down re-route bias."""
        self._subs.setdefault(str(name), {"inbox": [], "bias": {}})
        return str(name)

    def broadcast_to(self, name):
        """The top-down bias {content: magnitude} currently broadcast to `name` (empty if `name` is
        not subscribed or nothing is ignited). A subscribed specialist ADDS this to its next output --
        the re-route. Magnitude = base gain scaled by the learned (source, content) association."""
        s = self._subs.get(str(name))
        return dict(s["bias"]) if s is not None else {}

    @property
    def current(self):
        """The single content occupying the workspace (globally available), or None."""
        return self._incumbent

    # -- one competitive LIF cycle ----------------------------------------------------------------
    def _jitter_for(self, content):
        # a fixed tiny per-coalition current (deterministic under the seed) -- breaks exact-tie
        # symmetry so the shared-inhibition WTA settles on exactly ONE winner.
        if content not in self._jit:
            self._jit[content] = self._rng.uniform(0.0, self._JITTER)
        return self._jit[content]

    def _simulate(self, drives):
        """Settle the competitive LIF network for `_T` micro-steps; return {content: total_spikes},
        {content: second_half_spikes}. Each coalition pool integrates its drive + within-tick self-
        excitation MINUS the shared inhibitory pool's feedback from every OTHER pool (global lateral
        inhibition): a lone pool barely damps itself and ignites; among competitors the strongest
        silences the rest -> a single sustained winner. Discrete integer spike counts."""
        keys = sorted(drives, key=lambda c: self._order.get(c, 1 << 30))
        v = {k: 0.0 for k in keys}
        trace = {k: 0.0 for k in keys}
        self_i = {k: 0.0 for k in keys}
        a_inh = 0.0
        total = {k: 0 for k in keys}
        late = {k: 0 for k in keys}
        half = self._T // 2
        for t in range(self._T):
            spikes = {}
            for k in keys:
                lateral = self._G_INH * max(0.0, a_inh - self_i[k])  # inhibition from OTHER pools
                cur = drives[k] + self._W_SELF * trace[k] - lateral  # LIF input current
                v[k] += cur - self._LEAK * v[k]  # leaky integration
                if v[k] < 0.0:
                    v[k] = 0.0
                s = 0
                if v[k] >= self._v_th:  # DISCRETE SPIKE
                    v[k] -= self._v_th  # subtract-threshold reset
                    s = 1
                    total[k] += 1
                    if t >= half:
                        late[k] += 1
                    trace[k] += 1.0
                spikes[k] = s
                trace[k] *= self._TAU_SELF
            for k in keys:
                self_i[k] += self._K_INH * spikes[k] - self._LEAK_INH * self_i[k]
                if self_i[k] < 0.0:
                    self_i[k] = 0.0
            a_inh = sum(self_i.values())
        return total, late

    def step(self):
        """Run ONE workspace cycle. Groups submissions into coalitions, adds the re-entrant sustain
        current to the incumbent, settles the competition, and -- if a coalition sustains firing above
        rheobase -- IGNITES it, broadcasts it, and closes the loop. Returns {ignited, winner,
        availability}; clears the submission buffer."""
        # 1. group submissions into coalitions by content; base drive = strongest supporter + a small
        #    bonus per additional supporting specialist (a broader coalition drives harder).
        base = {}
        sources = {}
        for src, content, sal in self._buffer:
            base[content] = max(base.get(content, 0.0), sal)
            sources.setdefault(content, set()).add(src)
        for content in list(base):
            base[content] = min(1.2, base[content] + self._COAL_BONUS * (len(sources[content]) - 1))

        # 2. build each coalition's drive: bottom-up base + (incumbent only, re-entry ON) the decaying
        #    re-entrant top-down sustain current + a tiny deterministic symmetry-break jitter.
        drives = dict(base)
        if self.reentry and self._g_reentry > 1e-6 and self._incumbent is not None:
            drives[self._incumbent] = drives.get(self._incumbent, 0.0) + self._g_reentry
        for content in drives:
            drives[content] += self._jitter_for(content)

        # 3. settle the competitive spiking network; read out sustained (2nd-half) firing.
        if drives:
            total, late = self._simulate(drives)
        else:
            total, late = {}, {}
        self._last_total = total
        self._last_late = late

        # 4. IGNITION: the winner is the coalition still firing in the second half (survived the
        #    competition into the sustained attractor). NONE sustaining -> the workspace stays DARK
        #    (a real access failure -- weak/masked input does not reach access).
        fired = [c for c in late if late[c] > 0]
        if fired:
            winner = max(fired, key=lambda c: (late[c], total[c], -self._order.get(c, 1 << 30)))
            ignited = True
        else:
            winner = None
            ignited = False

        # 5. CLOSE THE LOOP + re-route + local Hebbian learning.
        if ignited:
            self._incumbent = winner
            bottom_up_supported = base.get(winner, 0.0) >= self.ignition_threshold
            if self.reentry and bottom_up_supported:
                self._g_reentry = self._SUSTAIN  # a fresh bottom-up ignition RE-CHARGES the loop
            self._hebb_update(winner, sources.get(winner, set()))
        else:
            self._incumbent = None
            self._hebb_update(None, set())

        # the winner's DRIVE (bottom-up base + re-entrant sustain) is what the graded access readout
        # in access_strength() reads. Kept here because `drives` is local to this cycle.
        self._last_drive = float(drives.get(winner, 0.0)) if ignited else 0.0

        # re-entrant current fades each tick (used NEXT tick); only a bottom-up ignition re-charges it.
        self._g_reentry *= self._REENTRY_DECAY
        if self._g_reentry < 1e-6:
            self._g_reentry = 0.0

        # 6. BROADCAST to subscribers + set each subscriber's top-down re-route bias.
        availability = self._deliver(winner, ignited)
        self._last = {"ignited": ignited, "winner": winner, "availability": availability}
        self._buffer = []
        return dict(self._last)

    # -- re-entry plumbing ------------------------------------------------------------------------
    def _hebb_update(self, winner, winner_sources):
        """Local Hebbian re-entry learning (no gradient): sources that CONTRIBUTED to the ignited
        content potentiate toward 1; every other association passively decays."""
        winning_keys = {(s, winner) for s in winner_sources} if winner is not None else set()
        for key in list(self._assoc):
            if key in winning_keys:
                continue
            self._assoc[key] *= 1.0 - self._HEBB_DECAY
            if self._assoc[key] < 1e-6:
                del self._assoc[key]
        for key in winning_keys:
            w = self._assoc.get(key, 0.0)
            self._assoc[key] = w + self._HEBB_LR * (1.0 - w)

    def _deliver(self, winner, ignited):
        """Broadcast the winner to every subscriber and set its top-down bias toward the ignited
        content. Returns global availability = fraction of subscribers reached (0 if none ignited)."""
        reached = 0
        for name, s in self._subs.items():
            if ignited:
                s["inbox"].append(winner)
                assoc = self._assoc.get((name, winner), 0.0)
                # base re-route (immediate) scaled by the learned association (adaptive). Re-entry OFF
                # cuts the top-down bias entirely (fair control).
                bias = self._REENTRY_GAIN * (1.0 + assoc) if self.reentry else 0.0
                s["bias"] = {winner: bias} if bias > 0.0 else {}
                reached += 1
            else:
                s["inbox"].append(None)
                s["bias"] = {}
        if not self._subs:
            return 1.0 if ignited else 0.0
        return reached / len(self._subs)

    # -- read-only introspection (functional only; no phenomenal field, ever) ---------------------
    def association(self, source, content):
        """The learned Hebbian re-entry weight for a (source, content) pair in [0,1] (0 if none)."""
        return self._assoc.get((str(source), content), 0.0)

    def access_strength(self):
        """Graded access strength in [0,1]: the fraction of a downstream HETEROGENEOUS READOUT
        POPULATION that sustains firing on the ignited coalition's drive. 0.0 on a dark tick.

        This replaces a single pool's sustained (2nd-half) firing RATE, which was documented as
        graded but measured PINNED: over 300 live wake ticks it took exactly ONE distinct value,
        1.000. The cause is structural, not a tuning problem -- that rate is capped at one spike per
        micro-step, and with `_LEAK = 0.5` the membrane threshold is `0.55 / 0.5 = 1.1` while a
        winning coalition's current (drive 0.668-1.000 measured, plus `_W_SELF` self-excitation of
        about +0.9 once it is firing) sits far above it. A single pool therefore fires every
        micro-step for every drive the brain ever produces, and the ratio is 1.0 by construction.
        The measure was graded only in a narrow band just above rheobase that the brain never
        occupies.

        A POPULATION rate code does not have that ceiling: spread the units' thresholds across the
        drive band and the FRACTION that sustains grades smoothly, saturating only past the highest
        threshold. Same idiom as `neuromodulation/spiking_curiosity.py` (`_POOL_N = 12`
        heterogeneous LIF units, whose docstring gives the same reason -- heterogeneity makes a
        pool's rate a SMOOTH function of drive instead of mode-locking).

        The readout is genuinely simulated rather than solved analytically, so units sitting near
        their threshold fire intermittently and contribute partial evidence.

        This reads the coalition's DRIVE, not its within-settle current: the self-excitation that
        creates the all-or-none attractor is exactly what destroys gradation, and drive (bottom-up
        base + re-entrant sustain) is the quantity that actually varies with how well the content is
        supported. Access still DECAYS as the re-entrant sustain fades under weak/masked input,
        which is the property this measure exists to provide.

        FUNCTIONAL access strength -- how strongly content holds the workspace. NOT a claim about
        phenomenal experience."""
        if not self._last.get("ignited"):
            return 0.0
        drive = float(getattr(self, "_last_drive", 0.0))
        if drive <= 0.0:
            return 0.0
        steps = max(1, self._T // 2)
        sustained = 0
        for i in range(self._ACC_N):
            # thresholds spread across the MEASURED operating band (see _ACC_LO/_ACC_HI), expressed
            # in drive units then converted to a membrane threshold by the shared leak.
            d_i = self._ACC_LO + (self._ACC_HI - self._ACC_LO) * i / (self._ACC_N - 1)
            v_th = d_i / self._LEAK
            v = 0.0
            spikes = 0
            for _ in range(steps):
                v += drive - self._LEAK * v
                if v >= v_th:
                    v -= v_th
                    spikes += 1
            if spikes * 2 >= steps:  # fired on at least half the window -> this unit sustains
                sustained += 1
        return round(sustained / self._ACC_N, 4)

    def inbox(self, name):
        """The sequence of broadcasts a subscribed specialist has received (None on a dark tick)."""
        s = self._subs.get(str(name))
        return list(s["inbox"]) if s is not None else []


# ============================ helpers for the self-check =========================================
def run_stream(stream, reentry=True, ignition_threshold=0.55, seed=0):
    """Feed a list of per-tick submission dicts (content -> salience OR content -> {source: salience})
    through a workspace; return the list of step() results."""
    ws = ReentrantWorkspace(ignition_threshold=ignition_threshold, reentry=reentry, seed=seed)
    out = []
    for tick in stream:
        for content, spec in tick.items():
            if isinstance(spec, dict):
                for src, sal in spec.items():
                    ws.submit(src, content, sal)
            else:
                ws.submit(content, content, spec)
        out.append(ws.step())
    return out


class _Specialist:
    """A tiny downstream specialist used ONLY by the self-check: it emits the content that maximises
    its own fixed evidence PLUS whatever top-down bias the workspace broadcasts to it. This is what a
    re-route looks like from the specialist's side -- the ignited content shifts the argmax."""

    def __init__(self, name, evidence):
        self.name = name
        self.evidence = dict(evidence)  # content -> bottom-up evidence

    def emit(self, ws):
        bias = ws.broadcast_to(self.name)  # {} if not subscribed / nothing ignited
        scored = {c: e + bias.get(c, 0.0) for c, e in self.evidence.items()}
        return max(scored, key=lambda c: (scored[c], c))


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    print("=" * 82)
    print(
        "RE-ENTRANT, NON-SATURATED GLOBAL WORKSPACE -- compete -> (maybe) ignite -> re-enter -> re-route"
    )
    print("=" * 82)

    # ---- (1) ANTI-SATURATION: ignition really can FAIL (vs the saturated baseline = 1.0 every tick) ----
    strong_stream = [{"scene": 0.85} for _ in range(12)]  # supra-threshold every tick
    subliminal_stream = [{"scene": 0.45} for _ in range(12)]  # real, but BELOW rheobase
    strong_av = [r["availability"] for r in run_stream(strong_stream)]
    subl_av = [r["availability"] for r in run_stream(subliminal_stream)]
    strong_avail = _mean(strong_av)
    subl_avail = _mean(subl_av)
    subl_failures = sum(1 for a in subl_av if a < 1.0)
    print("\n(1) anti-saturation (availability per tick):")
    print(
        f"    supra-threshold stream (0.85): mean availability {strong_avail:.2f}  (ignites every tick)"
    )
    print(
        f"    subliminal    stream (0.45): mean availability {subl_avail:.2f}  "
        f"({subl_failures}/{len(subl_av)} ticks FAILED to reach access)"
    )

    # inattentional blindness: a moderate target that WOULD ignite alone is masked by a stronger stream.
    masked_stream = [{"target": 0.62, "distractor": 0.92} for _ in range(12)]
    masked = run_stream(masked_stream)
    target_reach = _mean([1.0 if r["winner"] == "target" else 0.0 for r in masked])
    distractor_reach = _mean([1.0 if r["winner"] == "distractor" else 0.0 for r in masked])
    print(
        f"    inattentional blindness: target(0.62) reached access {target_reach:.0%} of ticks, "
        f"distractor(0.92) {distractor_reach:.0%}  (target would ignite ALONE, masked by competition)"
    )

    # ---- (2) RE-ENTRY IS A CLOSED, LOAD-BEARING LOOP: sustain across a bottom-up dropout ----------
    def sustain_across_gap(reentry, on=3, gap=6):
        ws = ReentrantWorkspace(reentry=reentry, seed=0)
        held = 0
        for t in range(on + gap):
            if t < on:
                ws.submit("percept", "held", 0.80)  # bottom-up present -> charges loop
            r = ws.step()  # gap ticks: NOTHING submitted
            if t >= on and r["ignited"] and r["winner"] == "held":
                held += 1
        return held

    held_on = sustain_across_gap(reentry=True)
    held_off = sustain_across_gap(reentry=False)
    print("\n(2) closed re-entry loop (sustain 'held' across a 6-tick bottom-up dropout):")
    print(
        f"    re-entry ON  held {held_on} gap-ticks (reverberation) ;  re-entry OFF held {held_off} "
        f"(feedforward releases immediately)"
    )

    # ---- (3) COMPETITION: two coalitions, exactly ONE wins access; the loser is SILENCED ---------
    ws = ReentrantWorkspace(seed=0)
    ws.submit("v", "strong", 0.90)
    ws.submit("v", "weak", 0.66)  # weak alone WOULD ignite (>0.55)
    comp = ws.step()
    strong_late = ws._last_late.get("strong", 0)
    weak_late = ws._last_late.get("weak", 0)
    print("\n(3) two competing coalitions (0.90 vs 0.66, both supra-threshold alone):")
    print(
        f"    winner {comp['winner']!r}  strong 2nd-half spikes {strong_late}, loser silenced "
        f"{weak_late} (single-slot bottleneck)"
    )

    # ---- (4) ATTENTIONAL BLINK: a 2nd target arriving while the loop is occupied FAILS, then RECOVERS
    def blink(reentry, lag):
        ws = ReentrantWorkspace(reentry=reentry, seed=0)
        ws.submit("t1", "T1", 0.66)
        ws.step()  # T1 ignites, charges the loop
        for _ in range(lag - 1):
            ws.step()  # empty ticks -> loop fades
        ws.submit("t2", "T2", 0.62)
        return ws.step()  # does T2 reach access?

    blink_short = blink(reentry=True, lag=1)  # T2 immediately after T1
    blink_long = blink(reentry=True, lag=6)  # T2 after the loop has faded
    blink_ctrl = blink(reentry=False, lag=1)  # loop OFF -> no occupation
    print("\n(4) attentional blink (T2=0.62 after T1=0.66):")
    print(
        f"    short lag, re-entry ON : T2 winner={blink_short['winner']!r} ignited={blink_short['ignited']} (BLINK)"
    )
    print(
        f"    long  lag, re-entry ON : T2 winner={blink_long['winner']!r} ignited={blink_long['ignited']} (recovered)"
    )
    print(
        f"    short lag, re-entry OFF: T2 winner={blink_ctrl['winner']!r} ignited={blink_ctrl['ignited']} "
        f"(no blink -> the loop CAUSED the blink)"
    )

    # ---- (5) SUBSCRIBER RE-ROUTE: the ignited content biases a subscribed specialist's next output --
    ws = ReentrantWorkspace(seed=0)
    subbed = _Specialist("subbed", {"A": 0.50, "B": 0.48})  # near-tie, own argmax = "A"
    control = _Specialist("control", {"A": 0.50, "B": 0.48})  # identical evidence, NOT subscribed
    decided = _Specialist("decided", {"A": 0.80, "B": 0.30})  # a CLEAR preference for "A"
    ws.subscribe(subbed.name)
    ws.subscribe(decided.name)  # 'control' deliberately not subscribed
    before_sub = subbed.emit(ws)
    for _ in range(3):  # drive "B" to ignition via other sources
        ws.submit("driver1", "B", 0.80)
        ws.submit("driver2", "B", 0.80)
        ws.step()
    after_sub = subbed.emit(ws)
    after_ctrl = control.emit(ws)
    after_decided = decided.emit(ws)
    print("\n(5) subscriber re-route (workspace ignited 'B'):")
    print(
        f"    subscribed near-tie specialist: {before_sub!r} -> {after_sub!r}  (re-routed to ignited content)"
    )
    print(f"    un-subscribed control (same evidence): {after_ctrl!r}  (unaffected)")
    print(
        f"    subscribed but CLEAR preference 'A': {after_decided!r}  (bias, not override -- honest)"
    )

    # ---- (6) LOCAL HEBBIAN RE-ENTRY LEARNING: a repeatedly-winning source's bias GROWS; a never-
    #          winning one stays at baseline (adaptive re-route, no gradient) --------------------------
    ws = ReentrantWorkspace(seed=0)
    ws.subscribe("winner")
    ws.subscribe("loser")
    assoc_traj = []
    for _ in range(8):
        ws.submit("winner", "X", 0.85)  # "winner" always wins access to "X"
        ws.submit("loser", "Y", 0.40)  # "loser" posts "Y" sub-threshold -> never wins
        ws.step()
        assoc_traj.append(round(ws.association("winner", "X"), 3))
    win_bias = ws.broadcast_to("winner").get("X", 0.0)
    lose_assoc = ws.association("loser", "Y")
    print("\n(6) local Hebbian re-entry learning (association 'winner'->'X' over 8 co-ignitions):")
    print(f"    {assoc_traj}")
    print(
        f"    'winner'->'X' broadcast bias {win_bias:.4f} (grew) ;  'loser'->'Y' association {lose_assoc:.4f} "
        f"(never co-ignited -> baseline)"
    )

    # ---- (7) determinism ---------------------------------------------------------------
    rerun = [r["availability"] for r in run_stream(subliminal_stream)]
    deterministic = rerun == subl_av

    print("=" * 82)

    # ================================ genuine self-checks (not rigged) ============================
    # (1) ANTI-SATURATION -- the core fix. The baseline ignites every tick (availability == 1.0); here a
    #     real, non-degenerate weak stream FAILS to reach access, and a moderate target is masked out.
    assert strong_avail == 1.0, (
        f"a supra-threshold stream must reach access every tick, got {strong_avail:.2f}"
    )
    assert subl_avail < 1.0, (
        f"a subliminal stream must FAIL to reach access (anti-saturation), got {subl_avail:.2f}"
    )
    assert subl_avail == 0.0 and subl_failures == len(subl_av), (
        "every sub-rheobase tick must stay DARK (real failure)"
    )
    assert target_reach == 0.0 and distractor_reach == 1.0, (
        "inattentional blindness: the masked moderate target must NEVER win access while the stronger stream does"
    )

    # (2) CLOSED, LOAD-BEARING RE-ENTRY -- sustains across a dropout only because the loop feeds back.
    assert held_on >= 1, (
        f"re-entry must sustain the content across a bottom-up dropout (held {held_on})"
    )
    assert held_on > held_off, f"the loop must be load-bearing (ON {held_on} > OFF {held_off})"
    assert held_off == 0, (
        f"feedforward-only must release immediately on a dropout (held {held_off})"
    )

    # (3) COMPETITION -- exactly one coalition seizes the slot; the loser (supra-threshold ALONE) is silenced.
    assert comp["winner"] == "strong" and comp["ignited"], (
        "the stronger coalition must win the single slot"
    )
    assert strong_late > 0 and weak_late == 0, (
        f"the loser must be SILENCED by shared inhibition (winner late {strong_late}, loser late {weak_late})"
    )

    # (4) ATTENTIONAL BLINK -- caused by the re-entrant occupation (recovers as the loop fades; absent when OFF).
    assert not blink_short["ignited"] or blink_short["winner"] != "T2", (
        "T2 must be BLINKED at short lag (loop occupied)"
    )
    assert blink_long["ignited"] and blink_long["winner"] == "T2", (
        "T2 must RECOVER at long lag (loop faded)"
    )
    assert blink_ctrl["ignited"] and blink_ctrl["winner"] == "T2", (
        "with re-entry OFF there is no blink (loop caused it)"
    )

    # (5) SUBSCRIBER RE-ROUTE -- the broadcast measurably biases a subscribed specialist; control unaffected;
    #     a clear preference is a bias, NOT an override (honest about strength).
    assert before_sub == "A", "the near-tie specialist's own argmax is 'A' before any broadcast"
    assert after_sub == "B", (
        "the ignited content 'B' must RE-ROUTE the subscribed near-tie specialist"
    )
    assert after_ctrl == "A", "the un-subscribed control (identical evidence) must be UNAFFECTED"
    assert after_decided == "A", (
        "a clear-preference specialist must NOT be overridden (re-route is a bias)"
    )

    # (6) LOCAL HEBBIAN LEARNING -- adaptive re-route grows for a co-igniting source, baseline for a non-winner.
    assert assoc_traj[-1] > assoc_traj[0], (
        f"the re-entry association must GROW with co-ignition: {assoc_traj}"
    )
    assert all(assoc_traj[i + 1] >= assoc_traj[i] - 1e-9 for i in range(len(assoc_traj) - 1)), (
        f"Hebbian potentiation must be monotone non-decreasing for the winner: {assoc_traj}"
    )
    assert win_bias > ReentrantWorkspace._REENTRY_GAIN, (
        "the winner's learned bias must exceed the base gain"
    )
    assert lose_assoc == 0.0, (
        f"a source that never co-ignites must stay at baseline association (got {lose_assoc})"
    )

    # (7) determinism + agnostic (no phenomenal claim is even representable here).
    assert deterministic, "same seed + stream must reproduce identical availability"

    print(
        "self-check OK: anti-saturation (supra avail 1.00 vs subliminal 0.00 = real access FAILURES; "
        f"masked target 0% vs distractor 100%); closed re-entry load-bearing (held {held_on} vs {held_off} "
        "across a dropout); competition silences the loser; attentional BLINK caused by the loop (short-lag "
        "fail, long-lag recover, OFF no-blink); subscriber RE-ROUTE (subbed A->B, control A, decided A); "
        f"Hebbian association grew {assoc_traj[0]:.2f}->{assoc_traj[-1]:.2f}; deterministic. "
        "FUNCTIONAL access only -- agnostic, no phenomenal / Phi claim."
    )


if __name__ == "__main__":
    main()
