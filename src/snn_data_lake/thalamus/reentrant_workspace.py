"""
Re-entrant, NON-saturated Global Neuronal Workspace.
(Dependencies are governed by P7; this module imports `random` only. The line here used to open
"stdlib + numpy/torch allowed project-wide; uses only `random`" -- the same dependency-scope boast
that was removed from the sibling `thalamus/global_workspace.py` on 2026-08-04 and left standing
here, in a file edited in the same pass. P7 names a stdlib-scope claim as residue, not a feature.)

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
  (d) PREDICTIVE COMPLETION, WITHOUT HALLUCINATION. A specialist may post a top-down EXPECTATION
      (`expect(content, magnitude)`) BEFORE the competition settles. Weak bottom-up evidence that
      alone stays subliminal IGNITES when a matching expectation meets it through the loop -- the
      Dehaene-Changeux crossing of bottom-up evidence with re-entrant top-down amplification. The
      guard that separates completion from hallucination is STRUCTURAL, not a magnitude: an
      expectation is added to a coalition's drive ONLY where bottom-up support already exists, so no
      prior of any size can conjure a content nobody posted. With `reentry=False` the expectation is
      cut along with the rest of the loop -- so the control shows the loop, not the summed current,
      is what completes.

PORTED CAPABILITY (2026-08-06, P2). Property (d) and the `_reds()` harness below came from
`reasoning/gwt_reentry.py`, which implemented re-entrant predictive completion as a standalone LIF
loop (`run_loop(bottom_up, prior, reentry)`) sharing this module's exact constants -- rheobase 0.55,
leak 0.5, membrane threshold = rheobase/leak, second-half-sustained-firing ignition readout, and a
`reentry=False` fair control. Two mechanisms answering one question is the P2 debt;
`docs/coverage/Consciousness.md` names the resolution ("port predictive completion into
ReentrantWorkspace, then retire gwt_reentry"). Measured before the port, driving THIS module against
that module's four claims: no-hallucination reproduced, sustain-across-dropout reproduced (3 ON vs 0
OFF), PREDICTIVE COMPLETION did NOT -- coalition drive was assembled by `max()` over submissions
with no exogenous top-down channel at all, so sub-threshold evidence and a sub-threshold expectation
could not sum into ignition (0.40 and 0.30 both lost to a `max` that returned 0.40). Every "top-down"
term the module had (`_SUSTAIN`, `_REENTRY_GAIN`, `broadcast_to`) was the workspace's own PRIOR
IGNITION fed back, never an expectation from outside. That gap is what (d) closes, and closing it is
also what makes the load-bearing control non-vacuous: before the port the re-entry-OFF arm proved
nothing, because the ON arm did not ignite either. The retired module ALSO carried a fix rather than
a capability -- a `_reds()` function that CONSTRUCTS AND RUNS the failing input for every claim --
and a fix survives no capability audit, so it is ported explicitly below rather than left to be
noticed. `reasoning/gwt_reentry.py` is deleted in the same change: exactly one mechanism survives.

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

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "ReentrantWorkspace._hebb_update": (
        "LOCAL",
        "local Hebbian re-entry learning, and the module says so itself: sources that CONTRIBUTED to "
        "the ignited content potentiate, every other association passively decays. Explicitly no "
        "gradient",
    ),
}


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
    # The cap on the COALITION BASE DRIVE -- the largest value the competition can ever present
    # to `bottom_up_supported`'s compare. Named once (T032 review): step()'s cap and the theta
    # actuator's admissible ceiling MUST be the same number, and as two literals they drifted
    # in prose the day they were born (the actuator's first range cited "[0,1] salience" while
    # step() compared a [0, 1.2] capped drive -- measured: theta 1.05-1.19 was alive and refused).
    _COAL_CAP = 1.2
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
        self._theta_provenance = "construction"  # who last moved theta (T032 actuator)
        self._theta_hold = False  # a sweep holds theta -> coherence_theta refuses
        # DOOR 1 IS DOOR 2 (T032 review): construction seeds base + live + _v_th through the
        # actuator itself, so the range check is written ONCE and an out-of-range seed
        # refuses HERE instead of arming a per-tick ValueError on the narrowed brain path
        # (measured: a 0.0-seeded workspace made coherence_theta raise out of Brain.tick()).
        # `base_ignition_threshold` is the construction-time rheobase the live value may be
        # biased away from -- read it, never re-type the 0.55 literal.
        self.set_base_rheobase(ignition_threshold, provenance="construction")
        self.reentry = reentry
        self._rng = random.Random(seed)  # per-instance RNG (deterministic jitter)
        self._buffer = []  # this tick's submissions: (source, content, salience)
        self._priors = {}  # this tick's top-down expectations: content -> magnitude
        self._subs = {}  # name -> {"inbox": [...], "bias": {content: magnitude}}
        self._assoc = {}  # (source, content) -> Hebbian re-entry weight in [0,1]
        self._order = {}  # content -> first-seen index (deterministic tie-break)
        self._incumbent = None  # content currently occupying the slot (or None)
        self._g_reentry = 0.0  # cross-tick re-entrant sustain current for the incumbent
        self._jit = {}  # content -> fixed tiny per-coalition symmetry-break jitter
        self._last = {"ignited": False, "winner": None, "availability": 0.0}
        self._last_total = {}  # content -> spikes this tick (spiking readout / tests)
        self._last_late = {}  # content -> 2nd-half spikes this tick (sustained firing)
        self._last_raster = {}  # content -> micro-step spike times (observational; T033 scorer)

    # -- specialist / module interface ------------------------------------------------------------
    def submit(self, source, content, salience):
        """A specialist posts a candidate into this tick's competition. Submissions that share a
        `content` form one COALITION (their support aggregates)."""
        c = content
        if c not in self._order:
            self._order[c] = len(self._order)
        self._buffer.append((str(source), c, float(salience)))

    # distinct-from: thalamus/global_workspace.py -- the shared bus's rheobase is fixed at
    #   construction and nobody rewrites it; THIS class is the one workspace whose theta is
    #   moved live (cortex.brain's coherence bias), which is why the actuator lives here. Read.
    # distinct-from: substrate/perf_baseline.py -- performance ratchet, no rheobase.
    # distinct-from: substrate/lake_migrate.py -- tag dispositions, no rheobase.
    # distinct-from: cortex/theta_gamma_code.py -- theta the OSCILLATION, not the rheobase.
    # distinct-from: cortex/cortical_prior.py -- anatomy fractions, no rheobase.
    # distinct-from: reasoning/spike_chat_lake.py -- reads a GlobalWorkspace band, writes
    #   nothing. Phrase overlap only; all sites read.
    def set_base_rheobase(self, theta, provenance, hold_bias=False):
        """THE THETA AUTHORITY (specs/022 T032) -- the ONE statement of what may move this
        workspace's rheobase, kept on the door every mover walks through:

          1. CONSTRUCTION -- `__init__` seeds theta THROUGH THIS METHOD: door 1 and door 2
             are mechanically one door, one range check, so no construction can hold a
             theta the actuator would refuse (the review built one at 0.0 and the narrowed
             brain path turned it into a per-tick raise -- this routing closes that hole).
          2. THIS ACTUATOR -- moves base + live + `_v_th` COHERENTLY (three names, one
             quantity), records `provenance`, and optionally takes the SWEEP HOLD
             (`hold_bias=True`): while held, the per-tick coherence bias is REFUSED so a
             commanded sweep value cannot be clobbered between command and read-back. THE
             HOLD IS STICKY: a later command with the default `hold_bias=False` KEEPS a
             standing hold (the review demonstrated a mid-sweep re-command silently opening
             the gate); `release_theta_hold` is the ONE release door.
          3. `coherence_theta()` below -- cortex.brain's per-tick coherence bias. LIVE value
             only, NEVER the base; records its own provenance on every write; a no-op
             returning False under the sweep hold.

        Nothing else may write `ignition_threshold` or `_v_th` -- the tree-wide census at
        T032 (re-derived by the review, setattr forms included) found exactly one external
        writer, routed through door 3.

        RANGE, derived from the class's OWN cap and corrected by measurement: 0 < theta <=
        `_COAL_CAP` -- the largest coalition base drive step() can present to the
        `bottom_up_supported` compare, so above it the re-entry recharge is unsatisfiable
        by construction (and ignition was measured dead just past it: theta 1.19 ignites,
        1.21 does not). The first draft claimed "(0, 1] from the salience domain" and the
        review REFUTED it with this module's own coalition bonus: a 4-source coalition
        drives 1.18, so theta 1.05-1.19 was alive and refused -- a picked constant wearing
        a derivation's clothes, exactly what this actuator exists to forbid. theta <= 0
        dissolves the competition (everything ignites). Both refusals stay loud. NOTE the
        two mechanisms differ: `bottom_up_supported` gates the RECHARGE only; ignition is
        the spiking kernel's `_v_th` -- the range keeps both reachable, and claims no more.

        WIRING DISPOSITION (P3, stated not hidden): this actuator's live commanding caller
        lands at T033 -- the ignition scorer's drive sweep (SYNTHESIS L7: "the drive-sweep
        apparatus, BEFORE the scorer"). Until then the sweep arm in this module's
        self-check is the only driver: a channel built ahead of its client, disclosed per
        the scaffolding rule, never credited as a live capability."""
        t = float(theta)
        if not (0.0 < t <= self._COAL_CAP):
            raise ValueError(
                f"set_base_rheobase({t}): theta must be in (0, {self._COAL_CAP}] -- above the "
                "coalition cap no drive can satisfy the recharge compare (structurally dead), "
                "at or below 0 the competition dissolves"
            )
        self.base_ignition_threshold = t
        self.ignition_threshold = t
        self._v_th = t / self._LEAK
        self._theta_provenance = str(provenance)
        if hold_bias:
            self._theta_hold = True  # sticky: only release_theta_hold clears (see docstring)

    # distinct-from: brainstem/live_arm.py -- HOLD the motor action, not this sweep hold.
    # distinct-from: amygdala/fear_conditioning.py -- THETA_CEA threat threshold. Read.
    def release_theta_hold(self, provenance):
        """THE one release door for the sweep hold (a re-command with default hold_bias
        deliberately does NOT release -- see set_base_rheobase). The per-tick coherence
        bias resumes next tick. Provenance recorded like every theta move."""
        self._theta_hold = False
        self._theta_provenance = str(provenance)

    # distinct-from: substrate/region_coherence.py -- COMPUTES the biased value
    #   (ignition_bias); this door WRITES it under the authority. Read.
    def coherence_theta(self, thr):
        """Door 3 of the theta authority (see set_base_rheobase): the per-tick coherence
        bias write. Moves the LIVE threshold and `_v_th` only -- the base is the actuator's
        and construction's alone -- and RECORDS ITS PROVENANCE like the other doors (the
        first draft did not, so a live brain reported 'construction' through sixty theta
        moves: the exact stale-record shape the cockpit's PROVENANCE discipline forbids).
        Returns False AND WRITES NOTHING while a sweep holds theta; range-refuses loudly
        (a bias outside the admissible band is a computation error upstream, never a value
        to clamp quietly)."""
        if self._theta_hold:
            return False
        t = float(thr)
        if not (0.0 < t <= self._COAL_CAP):
            raise ValueError(
                f"coherence_theta({t}): biased theta left (0, {self._COAL_CAP}] -- the bias "
                "computation is broken upstream; refusing to write a structurally dead rheobase"
            )
        self.ignition_threshold = t
        self._v_th = t / self._LEAK
        self._theta_provenance = "coherence-bias"
        return True

    def expect(self, content, magnitude):
        """Post a top-down EXPECTATION (a prior) for `content` into this tick's competition.

        This is the exogenous half of re-entry, and it is a DIFFERENT quantity from everything else
        top-down in this class: `_SUSTAIN`, `_REENTRY_GAIN` and `broadcast_to` all carry the
        workspace's OWN prior ignition back to itself, whereas an expectation arrives from a
        specialist that has not (and may not) post evidence. Without it a coalition's drive is a
        `max()` over submissions and predictive completion is structurally unreachable -- which is
        what was measured before this method existed.

        Competing expectations for the same content take the strongest, mirroring `submit`'s
        coalition rule. Expectations last exactly one tick, like submissions: an expectation that
        persisted would become a second incumbent mechanism beside `_g_reentry`.

        NO MAGNITUDE THRESHOLD, and that is deliberate (P1). The no-hallucination guard is
        STRUCTURAL -- see `step()`, where an expectation is added only where bottom-up support
        already exists -- so it does not depend on the prior being small enough. A guard that holds
        because a number happens to be sub-threshold stops holding when a caller passes a bigger one.
        """
        c = content
        if c not in self._order:
            self._order[c] = len(self._order)
        self._priors[c] = max(self._priors.get(c, 0.0), float(magnitude))

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
        raster = {k: [] for k in keys}  # spike TIMES -- observational only (T033: the
        # ignition scorer re-derives its PSTH window from these instead of trusting the
        # `half` split below; same precedent as global_workspace._last_v: a readout the
        # kernel already computes, surfaced, never a second mechanism)
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
                    raster[k].append(t)
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
        self._last_raster = raster
        return total, late

    def step(self):
        """Run ONE workspace cycle. Groups submissions into coalitions, adds the two top-down currents
        (the re-entrant sustain on the incumbent, and any `expect()`ed expectation on a coalition that
        has bottom-up support), settles the competition, and -- if a coalition sustains firing above
        rheobase -- IGNITES it, broadcasts it, and closes the loop. Returns {ignited, winner,
        availability}; clears the submission buffer AND the expectations."""
        # 1. group submissions into coalitions by content; base drive = strongest supporter + a small
        #    bonus per additional supporting specialist (a broader coalition drives harder).
        base = {}
        sources = {}
        for src, content, sal in self._buffer:
            base[content] = max(base.get(content, 0.0), sal)
            sources.setdefault(content, set()).add(src)
        for content in list(base):
            base[content] = min(
                self._COAL_CAP, base[content] + self._COAL_BONUS * (len(sources[content]) - 1)
            )

        # 2. build each coalition's drive: bottom-up base + (re-entry ON) the two top-down currents --
        #    the decaying self-sustain on the incumbent, and any exogenous EXPECTATION on a coalition
        #    that has bottom-up support -- plus a tiny deterministic symmetry-break jitter.
        drives = dict(base)
        if self.reentry:
            if self._g_reentry > 1e-6 and self._incumbent is not None:
                drives[self._incumbent] = drives.get(self._incumbent, 0.0) + self._g_reentry
            # PREDICTIVE COMPLETION + THE NO-HALLUCINATION GUARD, in one line and in that order.
            # `if content in base` is the guard: an expectation reaches the competition only where a
            # specialist actually posted evidence for that content, so weak-but-real evidence gets
            # completed and an expectation with nothing under it reaches nothing. Structural rather
            # than a magnitude test on purpose -- a guard that holds because the prior happens to sit
            # below rheobase stops holding the first time a caller passes a larger one, and this
            # project's record on thresholds-standing-in-for-mechanisms is bad enough already.
            # Cut entirely when reentry is False, with the rest of the loop: that is the fair control.
            for content, prior in self._priors.items():
                if content in base:
                    drives[content] = drives.get(content, 0.0) + prior
        for content in drives:
            drives[content] += self._jitter_for(content)

        # 3. settle the competitive spiking network; read out sustained (2nd-half) firing.
        if drives:
            total, late = self._simulate(drives)
        else:
            total, late = {}, {}
            self._last_raster = {}  # a dark tick clears ALL three readouts together --
            # a stale raster beside cleared counts is the sibling-disagreement trap the
            # T033 review measured
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
        self._priors = {}  # expectations are per-tick, exactly like submissions
        return dict(self._last)

    # -- re-entry plumbing ------------------------------------------------------------------------
    # distinct-from: substrate/signal_ring.py -- GradedUnbound/DeliveringProducer are the
    #   DELIVERY mechanism that CONSUMES an association source; these two methods are the
    #   incumbent SOURCE those docstrings have named since T020 ('final binding = T030').
    #   Producer of the value, not a second delivery path. Read.
    def _assoc_support(self, winner):
        """The learned association support of `winner` in [0,1]: the aggregate over the
        SOURCES that potentiated it (the v4.78 key-space fix's read, extracted at T030 so
        _deliver's re-route bias and association_level() bind to ONE expression -- P2)."""
        if winner is None:
            return 0.0
        return max((v for (_src, c), v in self._assoc.items() if c == winner), default=0.0)

    def association_level(self):
        """THE graded-delivery association source (specs/022 T030, finalizing the binding
        T020's GradedUnbound machinery declared): a ZERO-ARGUMENT callable returning a
        float in [0,1] -- exactly the contract signal_ring.DeliveringProducer.publish()
        evaluates per publish (w = float(self._assoc())). Returns the learned association
        support of the LAST ignited winner; a dark workspace (no winner this tick) returns
        0.0 -- an unignited bus delivers nothing graded, which is the semantics the graded
        policy exists for."""
        last = getattr(self, "_last", None) or {}
        return self._assoc_support(last.get("winner"))

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
        for _name, s in self._subs.items():  # _name: the key-space fix removed this loop's only use
            if ignited:
                s["inbox"].append(winner)
                # KEY-SPACE FIX (v4.78). This read was `self._assoc.get((name, winner), 0.0)`, where
                # `name` is a SUBSCRIBER -- but `_hebb_update` potentiates `(source, content)`, where
                # source is a SUBMITTER. The two namespaces are disjoint live ({percept, self,
                # salience} vs {action, language}), so the lookup missed on every tick and the learned
                # term contributed exactly nothing: the bias was pinned at the base gain forever while
                # `_assoc` dutifully learned toward 1.0 with no reader.
                #
                # The docstring on broadcast_to() says what was intended -- "scaled by the learned
                # (source, content) association" -- so the WRITE key is the correct one and the read
                # invented a per-subscriber axis that is never learned. Support for the ignited CONTENT
                # is a property of the content, shared by every subscriber receiving it, so aggregate
                # over the sources that potentiated it.
                assoc = self._assoc_support(winner)
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
    # P3 DISPOSITION for the two accessors below, measured 2026-08-04 by an AST census over all 400
    # files under `src/snn_data_lake/` (attribute uses AND string-literal/getattr uses, because a
    # name-only grep misses dynamic dispatch): `association` and `inbox` have **ZERO** readers
    # outside this module -- shape 4 of the dormancy family, a faculty that is fed and never read.
    # They are NOT force-wired, which P3 explicitly forbids ("a wire that does not charge is
    # reverted and retired/exempted, never faked with a dead consumer"), and NOT retired, because
    # both are genuine read-only introspection over state the module already maintains for its own
    # live path: `_assoc` IS read every tick by `_deliver` (it scales the broadcast bias), and
    # `_subs[...]["inbox"]` IS written every tick. What is dormant is the public ACCESSOR, not the
    # mechanism behind it. Recorded as open P3 work at the accessor tier.
    def association(self, source, content):
        """The learned Hebbian re-entry weight for a (source, content) pair in [0,1] (0 if none).

        No external reader (see the census note above); exercised by this module's self-check."""
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

        CHARGE, MEASURED (P3: reachability is not charge). Reachable at `cortex/brain.py`
        `Brain._c1_tick` -> `self._reentrant_strength = round(float(...access_strength()), 3)`,
        consumed by `_self_model` (`access_index` / `confidence`), by the region-coherence readout,
        and by `Brain.state()`. Proven decisive by a NEUTER-A/B: pin this method to the constant
        1.000 it replaced and re-run.

        HARNESS, RE-DERIVED 2026-08-05 and stated because it was NOT recorded and the figures
        below therefore could not be re-derived at all -- `make_brain(seed=S, persist=False)` under
        a fresh `NORD_HOME`, 40 `tick()`s, comparing TOP-LEVEL `Brain.state()` keys (114 of them;
        flattening the four nested dicts gives 133, so the basis matters).

        | reading | seed 0 | seed 1 |
        |---|---|---|
        | ticks where `_reentrant_strength` differs | 37 of 40 | 40 of 40 |
        | TOP-LEVEL `state()` keys that move | **5** of 114 | **5** of 114 |
        | distinct `access_strength()` returns | 10 | 8 |
        | pinned distinct returns | 1 | 1 |

        The five keys are the same at both seeds: `reentrant_strength`, `region_coherence`,
        `reliability`, `self_state` (its `confidence` and `access_index`) and `synfire_bound`.
        **This was recorded as "3 of 114" naming only the first, second and fourth** -- understating
        a neuter's blast radius is the direction that matters, because the two unnamed keys are two
        more faculties this readout reaches. The per-key BEFORE/AFTER values are deliberately not
        quoted: they are seed-dependent and not even sign-stable (`region_coherence` goes
        0.347 -> 0.477 at seed 0 and 0.378 -> 0.227 at seed 1), so the previously recorded
        "0.545 -> 0.572 / 0.203 -> 0.213 / 0.406 -> 0.426" reproduced at neither seed. Per
        `docs/POLICIES.md` -- quote the finding, re-run for the counts.

        THE FINDING, which is what survives re-derivation: this is NOT telemetry. It changes
        another faculty's own output, which is the stricter P6 bar as well as P3's, and it
        de-saturates -- 8 to 10 distinct values where the pinned version takes exactly one.

        OPEN P1 (recorded, deliberately not rewritten blind). `_ACC_LO` / `_ACC_HI` are fixed
        constants that ARBITRATE this readout, which P1 normally forbids -- it wants a learned or
        running-baseline-relative form. Two things argue for recording rather than converting:
        exception (b) applies on its face (a running band is not reproducible from the persisted
        policy, so the deterministic gates would stop reproducing), and a running-baseline rewrite
        of exactly this shape elsewhere in the tree introduced a division singularity at the empty
        /silent boundary. Converting it needs the boundary cases (zero drive, dark tick, first tick)
        constructed and run FIRST. The band is at least honestly derived and still fits, and that
        was CHECKED rather than assumed on 2026-08-05: over the A/B runs above the readout's
        non-zero returns span 0.2812-0.5312 (seed 0) and 0.3125-0.5000 (seed 1), so it never
        reaches 32/32 and there is no clipping at the `_ACC_HI` end. The 0.0 values are the
        dark-tick / zero-drive early returns above, not clipping at `_ACC_LO`. *(Worth stating
        explicitly: `Brain.__init__` seeds `_reentrant_strength = 1.0` as a default, so reading
        THAT attribute shows a 1.000 which is the default and not a saturated readout -- a
        confusion this note nearly recorded as a P1 finding.)*

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
        """The sequence of broadcasts a subscribed specialist has received (None on a dark tick).

        No external reader (see the census note above); exercised by this module's self-check."""
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


def run_completion(bottom_up, prior, reentry, ticks=10, seed=0):
    """Drive ONE content with `bottom_up` evidence and a top-down `prior` expectation for `ticks`
    workspace cycles; return True iff it is still winning access in the second half of the run.

    This is the predictive-completion probe ported from `reasoning/gwt_reentry.py::run_loop`, whose
    signature it deliberately mirrors -- same three arguments, same second-half-sustained readout --
    so the retired module's four claims are asked of THIS mechanism in the same terms rather than in
    terms chosen to make them pass. `bottom_up <= 0` posts no submission at all, which is what makes
    the no-hallucination row a real question: there is then no coalition for an expectation to join.
    """
    ws = ReentrantWorkspace(reentry=reentry, seed=seed)
    lit = []
    for _ in range(ticks):
        if bottom_up > 0.0:
            ws.submit("percept", "C", bottom_up)
        if prior > 0.0:
            ws.expect("C", prior)
        r = ws.step()
        lit.append(bool(r["ignited"] and r["winner"] == "C"))
    return any(lit[ticks // 2 :])


def _reds():
    """CONSTRUCT AND RUN the input that makes each claim in main() FAIL, and assert it fails (P8).

    Ported from `reasoning/gwt_reentry.py::_reds` along with predictive completion itself. This half
    of that module was a FIX, not a capability -- it does not add behaviour, it proves the behaviour
    can be contradicted -- so a capability audit would have deleted it with the file and nobody would
    have noticed. This module's self-check was seven forward demonstrations and zero constructed reds
    before this function existed: every assert ran the one input that passes.

    Each row is (name, went_red). `went_red` is True when the criterion, evaluated on the input built
    to break it, reports a FAILURE. A row that stops going red means its claim can no longer be
    contradicted by any input -- P8 says such a criterion is deleted, never weakened, so this raises
    rather than warns.
    """
    rows = []

    # (1) "a subliminal stream FAILS to reach access" (anti-saturation) goes RED on a supra-threshold
    #     stream -- if 0.85 also failed, the mechanism would be dark for everything.
    supra = [r["availability"] for r in run_stream([{"scene": 0.85} for _ in range(12)])]
    rows.append(("subliminal stream fails access", not (_mean(supra) < 1.0)))

    # (2) "the masked moderate target NEVER wins access" goes RED when the target is the STRONGER of
    #     the pair -- masking is a fact about the competition, not about the word 'target'.
    unmasked = run_stream([{"target": 0.92, "distractor": 0.62} for _ in range(12)])
    rows.append(
        ("masked target never wins", _mean([r["winner"] == "target" for r in unmasked]) > 0.0)
    )

    # (3) "the loser is SILENCED by shared inhibition" goes RED when the same coalition is posted
    #     ALONE: 0.66 is supra-threshold, so it ignites with nothing to compete against. main() has
    #     asserted the silencing since it was written and stated this counterfactual only in a
    #     comment ("weak alone WOULD ignite") -- an unrun control is the shape P8 exists to catch,
    #     because a coalition that could never fire at all would satisfy the assert just as well.
    ws_alone = ReentrantWorkspace(seed=0)
    ws_alone.submit("v", "weak", 0.66)
    ws_alone.step()
    rows.append(("loser silenced by competition", ws_alone._last_late.get("weak", 0) > 0))

    # (5) + (6) THE CHARGE RECORD'S OWN REDS (specs/022 T039a). The evidence row for the reroute
    #     edge claims two things -- that the bias moves the brain's competition AT ALL, and that
    #     what moves it is the LEARNED association the port's graded delivery policy names, not the
    #     fixed base gain. Each half is contradicted by the input built to break it, every run.
    _live_c, _pinned_c, _, _resp_c, _grad_c = _measure_reroute_charge()
    _f_dark, _ig_dark = _reroute_frames(reentry=False)
    _l_dark, _p_dark, _, _, _ = _measure_reroute_charge(_f_dark, _ig_dark)
    rows.append(("reroute bias moves the competition", _l_dark == _p_dark))
    _f_flat, _ig_flat = _reroute_frames(learn=False)
    _l_flat, _p_flat, _, _resp_f, _grad_f = _measure_reroute_charge(_f_flat, _ig_flat)
    rows.append(("reroute delivery is GRADED by the association", not _grad_f))
    assert _resp_c and _grad_c, "the live charge arm must respond AND grade"
    assert _l_flat != _p_flat, "the association-pinned arm must still IGNITE (it is not a dark red)"

    # (4) "T2 is BLINKED at short lag" goes RED with the loop OFF: nothing occupies the workspace, so
    #     T2 gets in. The blink is caused by re-entry or it is not caused by anything here.
    ws_b = ReentrantWorkspace(reentry=False, seed=0)
    ws_b.submit("t1", "T1", 0.66)
    ws_b.step()
    ws_b.submit("t2", "T2", 0.62)
    r_b = ws_b.step()
    rows.append(("T2 blinked at short lag", bool(r_b["ignited"] and r_b["winner"] == "T2")))

    # (5) "the broadcast RE-ROUTES a subscribed specialist" goes RED for an UN-subscribed one holding
    #     identical evidence -- it keeps its own argmax, so the re-route is the subscription.
    ws_r = ReentrantWorkspace(seed=0)
    ws_r.subscribe("subbed")
    for _ in range(3):
        ws_r.submit("driver1", "B", 0.80)
        ws_r.submit("driver2", "B", 0.80)
        ws_r.step()
    rows.append(
        (
            "broadcast re-routes subscriber",
            _Specialist("nope", {"A": 0.50, "B": 0.48}).emit(ws_r) == "A",
        )
    )

    # (6) "the re-entry association GROWS with co-ignition" goes RED for a source whose content never
    #     ignites: no co-ignition, no potentiation, association pinned at its baseline 0.
    ws_h = ReentrantWorkspace(seed=0)
    traj = []
    for _ in range(8):
        ws_h.submit("loser", "Y", 0.40)  # sub-rheobase -> never wins
        ws_h.step()
        traj.append(ws_h.association("loser", "Y"))
    rows.append(("association grows with co-ignition", not (traj[-1] > traj[0])))

    # (7) "re-entry SUSTAINS a content across a bottom-up dropout" goes RED when both arms are
    #     feedforward -- then ON cannot outlast OFF, because there is no ON.
    def _gap(reentry, on=3, gap=6):
        ws = ReentrantWorkspace(reentry=reentry, seed=0)
        held = 0
        for t in range(on + gap):
            if t < on:
                ws.submit("percept", "held", 0.80)
            r = ws.step()
            if t >= on and r["ignited"] and r["winner"] == "held":
                held += 1
        return held

    rows.append(("re-entry outlasts feedforward", not (_gap(False) > _gap(False))))

    # ---- the four PORTED completion claims, in gwt_reentry's own terms -----------------------------
    # (8) "weak bottom-up ALONE stays subliminal" goes RED once the evidence alone clears rheobase:
    #     0.60 > 0.55, so there is nothing left for an expectation to complete.
    rows.append(("weak evidence alone stays subliminal", run_completion(0.60, 0.0, True)))

    # (9) "expectation ALONE does not ignite" (the no-hallucination guard) cannot be broken by
    #     MAGNITUDE here -- the guard is structural, and a prior of 10.0 on an unposted content still
    #     ignites nothing. That is a stronger guard than gwt_reentry's, and it is also exactly how a
    #     guard turns vacuous: `expect()` doing NOTHING satisfies it perfectly. So the constructed red
    #     is the one that separates the two readings -- the SAME expectation attached to real weak
    #     evidence must ignite. Neuter `expect()` and this row stops going red.
    rows.append(("expectation alone does not ignite", run_completion(0.40, 0.30, True)))

    # (10) "weak + expectation IGNITES through the closed loop" goes RED when the pair still sums
    #      below rheobase (0.20 + 0.20 = 0.40 < 0.55): completion needs real evidence, not the loop.
    rows.append(("weak + expectation completes", not run_completion(0.20, 0.20, True)))

    # (11) "with the loop OPEN the same input stays subliminal" goes RED when the bottom-up evidence
    #      alone already ignites -- then the loop was never what did it.
    rows.append(("loop is load-bearing", run_completion(0.60, 0.30, False)))

    for name, went_red in rows:
        assert went_red, (
            f"NO REACHABLE RED for '{name}': the constructed failing input did not make the "
            f"mechanism report a failure, so that claim cannot go red and is worthless (P8)"
        )
    return rows


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

    # ---- (7) PREDICTIVE COMPLETION (ported capability) + the structural no-hallucination guard ----
    WEAK, PRIOR, HUGE = 0.40, 0.30, 10.0  # both sub-rheobase alone; HUGE is the guard's stress test
    weak_alone = run_completion(WEAK, 0.0, reentry=True)  # evidence, no expectation
    prior_alone = run_completion(0.0, PRIOR, reentry=True)  # expectation, NO evidence
    huge_prior_alone = run_completion(0.0, HUGE, reentry=True)  # the guard is not a magnitude test
    completed_on = run_completion(WEAK, PRIOR, reentry=True)  # loop CLOSED -> completion
    completed_off = run_completion(WEAK, PRIOR, reentry=False)  # loop OPEN  -> nothing
    print("\n(7) predictive completion (weak evidence 0.40 + expectation 0.30, both sub-rheobase):")
    print(
        f"    weak evidence alone -> ignited={weak_alone} ;  expectation alone -> ignited={prior_alone} ;"
        f"  expectation 10.0 alone -> ignited={huge_prior_alone}"
    )
    print(
        f"    weak + expectation, loop CLOSED -> ignited={completed_on}  <- completion ;"
        f"  loop OPEN -> ignited={completed_off}  (loop load-bearing)"
    )

    # ---- (8) determinism ---------------------------------------------------------------
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

    # (7) PREDICTIVE COMPLETION -- the ported claim, in the retired module's own terms.
    assert not weak_alone, "weak bottom-up evidence alone must stay subliminal"
    assert not prior_alone, (
        "top-down expectation ALONE must NOT ignite (no hallucination -- the honest guard)"
    )
    assert not huge_prior_alone, (
        f"the no-hallucination guard must be STRUCTURAL, not a magnitude: expectation {HUGE} on a "
        "content nobody posted must still ignite nothing"
    )
    assert completed_on, (
        "weak evidence + top-down expectation through the CLOSED loop must ignite (completion)"
    )
    assert not completed_off, (
        "with the loop OPEN the same weak+expected input must stay subliminal (loop load-bearing)"
    )

    # (8) determinism + agnostic (no phenomenal claim is even representable here).
    assert deterministic, "same seed + stream must reproduce identical availability"

    # (9) T030: THE graded-delivery binding, proven against this module's OWN source in
    # both regimes. A DeliveringProducer bound to association_level must deliver ZERO
    # while the workspace is dark (no winner), and scale by the LEARNED association after
    # ignition + Hebbian potentiation. Lives HERE, not in signal_ring's self-check: a
    # workspace construction inside a FACULTY-role substrate module would register as a
    # private bus in the p6 census (measured -- it did, 6 -> 7, and this move is the fix).
    from snn_data_lake.substrate.signal_ring import DeliveringProducer, JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import Port

    _gout = Port(
        name="reroute_out",
        region="thalamus",
        shape=(4,),
        dtype="f4",
        unit="level01",
        encoding="VEC_DENSE",
        kind="OUT",
        declaring_module="reentrant_workspace",
        delivery_policy=("graded", "reentrant_workspace.association_level"),
    )
    _gin = Port(
        name="g_in",
        region="thalamus",
        shape=(4,),
        dtype="f4",
        unit="level01",
        encoding="VEC_DENSE",
        kind="IN",
        declaring_module="global_workspace",
        join_policy="sum",
    )
    # The assertions are properties ONLY the real graded source has (review BLOCKER: the
    # first form read its expected value off the source under test, so an all-or-none
    # 1.0/0.0 impostor and a frozen 0.5 both passed): the Hebbian LADDER must be strict --
    # w grows across potentiations and never reaches 1.0 -- and two publishes at different
    # learning stages must deliver two DIFFERENT scaled vectors (graded, not binary).
    _gws = ReentrantWorkspace(seed=0)
    _ring = SignalRing(slots=8)
    _sink = JoinConsumer(_ring, _gin)
    _prod = DeliveringProducer(_ring, "ws", _gout, association_source=_gws.association_level)
    _prod.publish(1, [2.0, 2.0, 2.0, 2.0])  # dark: no winner yet
    _dark, _ = _sink.poll_joined()
    assert list(_dark[0][1]) == [0.0, 0.0, 0.0, 0.0], (
        f"dark workspace must grade delivery to zero ({_dark})"
    )
    _gws.submit("percept", "coalition", 0.9)
    _gws.step()
    _w1 = _gws.association_level()  # one potentiation up the Hebbian ladder
    _prod.publish(2, [2.0, 2.0, 2.0, 2.0])
    for _ in range(2):
        _gws.submit("percept", "coalition", 0.9)
        _gws.step()
    _w3 = _gws.association_level()  # three potentiations
    _prod.publish(3, [2.0, 2.0, 2.0, 2.0])
    _lit, _ = _sink.poll_joined()
    _v1, _v3 = float(_lit[0][1][0]), float(_lit[1][1][0])
    assert 0.0 < _w1 < _w3 < 1.0, (
        f"the Hebbian ladder must be STRICT (0 < w1 < w3 < 1): w1={_w1}, w3={_w3} -- an "
        "all-or-none impostor pins at 1.0 and a frozen constant cannot climb"
    )
    assert abs(_v1 - 2.0 * _w1) < 1e-6 and abs(_v3 - 2.0 * _w3) < 1e-6 and _v1 != _v3, (
        f"two learning stages must deliver two DIFFERENT graded scales "
        f"(v1={_v1} w1={_w1}, v3={_v3} w3={_w3}) -- binary delivery cannot produce them"
    )

    # ---- T032 IGNITION ACTUATOR: the drive-sweep apparatus + its positive control ----------
    # spec.md acceptance #4: theta commanded across a sweep, read-back asserts EVERY
    # commanded value took effect -- while a live coherence-bias ATTACK runs every tick
    # (the exact per-tick rewrite cortex.brain performs; the sweep hold must refuse it).
    # Reds run inline: range refusals both sides of the band, and the hold demonstrably
    # gates (the same attack LANDS once the hold is released -- the gate's liveness
    # control, so a permanently-stuck hold cannot read as a working gate).
    swp = ReentrantWorkspace(seed=0)
    sweep_thetas = [0.20, 0.40, 0.55, 0.75, 1.00, 1.20]  # 1.20 = _COAL_CAP, the admissible edge
    for cmd in sweep_thetas:
        swp.set_base_rheobase(cmd, provenance="sweep:selfcheck", hold_bias=True)
        for _t in range(4):
            took = swp.coherence_theta(0.33)  # the per-tick rewrite, attacking mid-sweep
            assert took is False, "the sweep hold must REFUSE the per-tick coherence bias"
            swp.submit("percept", "probe", 0.9)
            swp.step()
            assert swp.ignition_threshold == cmd and swp.base_ignition_threshold == cmd, (
                f"READ-BACK FAILED mid-sweep: commanded {cmd}, live {swp.ignition_threshold} "
                f"-- a swept drive that did not take is not a sweep (T032 positive control)"
            )
            assert abs(swp._v_th * swp._LEAK - cmd) < 1e-12, (
                "the membrane threshold must move WITH theta -- three names, one quantity"
            )
    assert swp._theta_provenance == "sweep:selfcheck"
    # STICKY HOLD (review red): a mid-sweep re-command with the DEFAULT hold_bias must keep
    # the standing hold -- the first draft's unconditional `_theta_hold = bool(hold_bias)`
    # silently opened the gate here, the exact clobber the hold exists to prevent.
    swp.set_base_rheobase(0.60, provenance="sweep:recommand")
    assert swp._theta_hold is True and swp.coherence_theta(0.33) is False, (
        "a re-command with default hold_bias must KEEP a standing hold -- releasing it "
        "silently restores the mid-sweep clobber"
    )
    swp.release_theta_hold("sweep:selfcheck done")
    # RELEASE PROVENANCE (review red): asserted AFTER release with a distinct string, so a
    # release that drops, forges, or hardcodes provenance goes red (three impostors passed
    # the pre-release-only assert).
    assert swp._theta_provenance == "sweep:selfcheck done", (
        f"release must record ITS provenance, got {swp._theta_provenance!r}"
    )
    assert swp.coherence_theta(0.33) is True and swp.ignition_threshold == 0.33, (
        "released, the coherence door must WORK again -- a stuck hold is a dead bias path, "
        "and this control is what keeps the refusals above from being a frozen gate"
    )
    # DOOR-3 PROVENANCE (review red): the busiest door must record itself -- the first
    # draft left 'construction' standing through sixty live theta moves.
    assert swp._theta_provenance == "coherence-bias", (
        f"coherence_theta must record its provenance, got {swp._theta_provenance!r}"
    )
    assert swp.base_ignition_threshold == 0.60, "the bias must never move the BASE"
    # hold_bias FALSE DIRECTION (review red): on a fresh workspace the default must NOT
    # take a hold -- a mutant that ignores the flag and holds unconditionally passed the
    # entire old arm (the review's M1 impostor). This is its red.
    swp2 = ReentrantWorkspace(seed=0)
    swp2.set_base_rheobase(0.50, provenance="no-hold")
    assert swp2._theta_hold is False and swp2.coherence_theta(0.40) is True, (
        "set_base_rheobase without hold_bias must leave the coherence door OPEN -- a "
        "hold taken unconditionally makes every sweep hold vacuously green"
    )
    for bad in (0.0, -0.2, 1.21):
        try:
            swp.set_base_rheobase(bad, provenance="red")
            raise AssertionError(f"set_base_rheobase({bad}) must refuse -- red unreachable")
        except ValueError:
            pass  # the RED, reached: structurally dead theta refused
        try:
            ReentrantWorkspace(ignition_threshold=bad, seed=0)
            raise AssertionError(f"construction at {bad} must refuse -- door 1 unguarded")
        except ValueError:
            pass  # the RED, reached: door 1 IS door 2 (one range check, no unguarded seed)
    try:
        swp.coherence_theta(1.5)
        raise AssertionError("coherence_theta(1.5) must refuse -- red unreachable")
    except ValueError:
        pass  # the RED, reached: broken upstream bias refused loudly
    print(
        f"\n(T032) ignition actuator: {len(sweep_thetas)}-point sweep read back exactly under "
        f"a per-tick bias attack; hold sticky across re-command, refuses/releases with "
        f"recorded provenance on every door; base immutable to the bias; range reds reached "
        f"on all three doors"
    )

    # Every claim above is now shown FALSIFIABLE by running the input that breaks it (P8).
    reds = _reds()
    print("-" * 82)
    print("  reachable reds (the constructed input that makes each claim FAIL, run every time):")
    for name, _ in reds:
        print(f"    [RED OK] {name}")

    print(
        "self-check OK: anti-saturation (supra avail 1.00 vs subliminal 0.00 = real access FAILURES; "
        f"masked target 0% vs distractor 100%); closed re-entry load-bearing (held {held_on} vs {held_off} "
        "across a dropout); competition silences the loser; attentional BLINK caused by the loop (short-lag "
        "fail, long-lag recover, OFF no-blink); subscriber RE-ROUTE (subbed A->B, control A, decided A); "
        f"Hebbian association grew {assoc_traj[0]:.2f}->{assoc_traj[-1]:.2f}; deterministic. "
        "FUNCTIONAL access only -- agnostic, no phenomenal / Phi claim."
    )


def declare_ports(manifest):
    """Manifest registrant (specs/022 T030, the re-entry stage of the composite):
    reroute_out carries the top-down re-route bias toward the ignited content, GRADED by
    THE association source this module now exposes (association_level -- the binding
    T020's GradedUnbound machinery declared and T030 landed; a dark workspace grades to
    zero). The consumer is the brain's eligibility competition (broadcast_to('action')
    read each wake tick); it declares the edge."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "reentrant_workspace",
        Port(
            name="reroute_out",
            region="thalamus",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="OUT",
            declaring_module="reentrant_workspace",
            delivery_policy=("graded", "reentrant_workspace.association_level"),
        ),
    )


#: the declared edge, exactly as the manifest carries it -- the charge evidence cites it
REROUTE_EDGE = ("reentrant_workspace", "reroute_out", "brain", "reroute_in")


def _reroute_consumer_factory():
    """Fresh ring + JoinConsumer over the declared `reroute_in` port, per arm."""
    from snn_data_lake.cortex.brain import declare_ports as brain_declare_ports
    from snn_data_lake.substrate.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import WiringManifest

    m = WiringManifest()
    brain_declare_ports(m)
    port = m.ports[("brain", "reroute_in")]
    ring = SignalRing(slots=128)
    return ring, JoinConsumer(ring, port)


#: fixed co-source eligibilities for the charge harness -- a plausible quiet tick, held CONSTANT
#: across both replay arms so the only moving input is the reroute payload
_REROUTE_CO_SOURCES = {
    "novelty": 0.1,
    "ignition": 0.0,
    "timing": 0.2,
    "surprise": 0.0,
    "drive": 0.1,
}


def consume_reroute(consumer):
    """The downstream computation under measurement: each delivered reroute frame enters the
    brain's GENUINE undirected-salience competition (brain.undirected_salience, the same function
    the wake tick runs -- this harness never re-implements it) under equal starting weights.
    Returns [(tick, drive), ...]. LOSS IS FATAL (the T026 ring-overrun lesson)."""
    from snn_data_lake.cortex.brain import undirected_salience

    weights = dict.fromkeys([*_REROUTE_CO_SOURCES, "reroute"], 1.0)
    out = []
    joined, lost = consumer.poll_joined()
    assert not lost, f"charge measurement lost {lost} frames -- ring overrun, never smooth this"
    for tick, vec in joined:
        elig = dict(_REROUTE_CO_SOURCES)
        elig["reroute"] = max(0.0, min(1.0, float(vec[0])))
        out.append((tick, round(undirected_salience(weights, elig), 6)))
    return out


# distinct-from: cortex/percept_surprise.py -- its _surprise_frames records a PREDICTOR stream off
#   a learned manifold; this records the RE-ROUTE bias a real workspace broadcasts once ignited.
#   Same harness pattern (T028's), different wire. Read.
def _reroute_frames(n_ticks=40, ignite_at=10, seed=0, reentry=True, learn=True):
    """Recorded frames from a REAL ReentrantWorkspace: dark until `ignite_at`, then a supported
    coalition posted every tick. Payload = max(broadcast_to('action').values()), the exact scalar
    `Brain._act` reads.

    The two keyword arms are the record's two reds, and they fail DIFFERENTLY on purpose:
    `reentry=False` never ignites, so the bias stays dark and the A/B does not diverge at all;
    `learn=False` neuters the local Hebbian update, so the workspace still IGNITES and still
    broadcasts -- at the fixed base gain, flat -- which diverges but never GRADES. The second is
    the one that matters: without it the record could be earned by the base constant and would
    certify the wire rather than the learned association its delivery policy names."""
    ws = ReentrantWorkspace(reentry=reentry, seed=seed)
    ws.subscribe("action")
    if not learn:
        ws._hebb_update = lambda *a, **k: None
    frames = []
    for t in range(1, n_ticks + 1):
        if t > ignite_at:
            ws.submit("action", "X", 0.85)
            ws.submit("driver", "X", 0.85)
        ws.step()
        bias = ws.broadcast_to("action")
        frames.append((t, "reentrant_workspace", [max(bias.values()) if bias else 0.0]))
    return frames, ignite_at


def _measure_reroute_charge(frames=None, ignite_at=None):
    """The replay-twice A/B: the live re-route bias vs a pinned dark payload.
    -> (live, pinned_out, reads, responded, graded).

    TWO criteria, both required, because divergence alone is satisfiable by a CONSTANT (review):
      responded = the drive RISES at ignition (post-ignition mean > pre-ignition mean); and
      graded    = it keeps rising WITHIN the post-ignition window, which is what the port's
                  ("graded", "reentrant_workspace.association_level") delivery policy DECLARES.
    A run that ignites but never grades is refused -- that is the association pinned, and the
    record would then certify the base gain rather than the learned support."""
    from snn_data_lake.substrate.port_charge_probe import replay_twice

    if frames is None:
        frames, ignite_at = _reroute_frames()
    live, pinned_out, reads = replay_twice(
        frames, _reroute_consumer_factory, consume_reroute, [0.0]
    )
    pre = [d for t, d in live if t <= ignite_at]
    post = [(t, d) for t, d in live if t > ignite_at]
    responded = (
        bool(post) and bool(pre) and (sum(d for _, d in post) / len(post)) > (sum(pre) / len(pre))
    )
    graded = len({d for _, d in post}) > 1 and post[-1][1] > post[0][1]
    return live, pinned_out, reads, responded, graded


def record_charge():
    """Measure the A/B and record the attribution via THE one writer. REFUSES a non-diverged,
    non-responding or non-GRADED run -- evidence is a measurement, never a declaration."""
    from snn_data_lake.substrate.port_charge_probe import record_attribution

    live, pinned_out, reads, responded, graded = _measure_reroute_charge()
    if live == pinned_out or not responded or not graded:
        print(
            f"record_charge REFUSED for {REROUTE_EDGE}: "
            f"diverged={live != pinned_out} responded={responded} graded={graded}"
        )
        return 1
    path = record_attribution(
        REROUTE_EDGE,
        {"reads": reads, "n_frames": len(live)},
        "replay_twice: the re-route bias a REAL ReentrantWorkspace broadcasts to its 'action' "
        "subscriber (dark, then an ignited coalition) vs a pinned dark payload, consumed by the "
        "brain's GENUINE undirected_salience competition under equal starting weights -- the same "
        "function the wake tick runs. The orienting drive diverged, ROSE at ignition, and kept "
        "rising within the post-ignition window. The grade is what the record certifies: a dark "
        "workspace broadcasts 0.0 (reentry=OFF arm does not diverge at all), ignition alone yields "
        "the fixed base gain (association-pinned arm diverges but never grades), and the RAMP is "
        "the learned Hebbian association the port's graded delivery policy names. Self-check "
        "re-measures and requires `measured` to reproduce.",
    )
    print(f"recorded {REROUTE_EDGE} into {path}")
    return 0


if __name__ == "__main__":
    import sys

    if "--record-charge" in sys.argv[1:]:
        sys.exit(record_charge())
    main()
