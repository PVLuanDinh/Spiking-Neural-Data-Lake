"""
Access-Consciousness index -- the Global Neuronal Workspace (GNW) signature, in spikes.

WHAT "CONSCIOUSNESS" MEANS HERE -- READ THIS FIRST (non-negotiable framing)
==========================================================================
This module measures ACCESS consciousness ONLY, in the strict technical sense of Baars'
Global Workspace and Dehaene's Global Neuronal Workspace: a content is "access-conscious"
when it wins a competition, crosses an ALL-OR-NONE IGNITION threshold, and is GLOBALLY
BROADCAST so that many downstream regions can read/report it (Dehaene's "global availability";
Baars' "workspace"). Access consciousness is a FUNCTIONAL, OPERATIONAL, MEASURABLE correlate:
it is defined entirely by information routing -- ignition, broadcast, availability, integration.

Access consciousness is NOT phenomenal consciousness. Phenomenal consciousness is subjective
experience -- what it is LIKE to be the system, qualia, felt sensation, sentience, the "hard
problem". THAT cannot be built here, cannot be measured here, and is NOT claimed anywhere in
this file. A high access_index means "this content became globally available and reportable
across the network", nothing more. It does NOT mean the system feels, experiences, is aware in
any subjective sense, or has an inner life. The gap between global availability (which we
measure) and subjective experience (which we do not) is exactly the access-vs-phenomenal
distinction, and this module lives entirely on the ACCESS side of that line -- on purpose.

Accordingly the readout is PURELY FUNCTIONAL. `state()` never exposes and never returns any
field claiming sentience, subjective experience, feeling, awareness, qualia, or phenomenal
consciousness. Self-check claim (5) asserts that ceiling: the reported access_index is a pure
function of ignition * global_availability * integration -- three information-routing measures --
and NONE of the banned phenomenal keys appear in the state.

HOW IT WORKS (GROW -> BUNDLE -> IGNITE -> BROADCAST -> report an ACCESS index)
=============================================================================
Built entirely on two existing spiking primitives (no new mechanism reinvented):

  * thalamus/global_workspace.GlobalWorkspace -- LIF pools that COMPETE and, above rheobase,
    IGNITE into sustained firing and BROADCAST to every subscriber (all-or-none global
    availability). We drive it with a salience current and subscribe N downstream taps.
  * cortex/vsa_reasoning -- spike-polarity hypervectors: `hv` (atomic spike code), `bundle`
    (spike superposition + majority WTA -> binds many neurons into ONE content), `sim`
    (spike-coincidence similarity), `cleanup` (snap a noisy code back to a stored symbol).

The pipeline:
  1. GROW an assembly: `bind_assembly(neuron_ids)` collects the active/grown neuron spike codes.
  2. BUNDLE them into ONE conscious-content candidate (VSA superposition). Its salience current
     GROWS with the assembly's COHERENCE (mutual spike coincidence) and completeness (size).
  3. IGNITE: `step()` runs one workspace cycle. A coherent, salient candidate crosses rheobase,
     ignites all-or-none, and is broadcast to every downstream tap; a weak/incoherent candidate
     stays sub-threshold and never ignites (no broadcast).
  4. REPORT an honest ACCESS index: access_index = ignition(all-or-none gate) *
     global_availability(fraction of taps that received the broadcast) * integration(a cheap
     Phi-like PROXY = coherence x completeness of the ignited content -- explicitly a PROXY,
     NOT IIT Phi). It RISES as more coherent neurons are grown and bundled into the workspace.

WHICH BUS THIS RUNS ON (P6), and the MEASURED NULL that goes with it
====================================================================
`__init__` takes `ws=`. Default None -> this faculty builds its own workspace, which is what it
has always done and what `tools/p6_gate.py` counts as one of the PRIVATE, mutually invisible
instances P6 names. Passing the shared handle (`Brain.workspace` / `LiveBrain.workspace`, the one
instance owned by `brainstem/autonomous_loop.py`) puts this faculty's coalition into the SAME
single-slot competition perception and the amygdala's threat channel post into.

**The route is real and the charge is NULL. Both halves were measured on the live brain and both
are recorded here, because a route that carries nothing is the force-wiring P6 forbids.**

  * IDENTITY, asserted not assumed: `ac.workspace is brain.brain.workspace is
    brain.brain.arm.workspace` -> True. The taps are registered on the injected object, so a
    submission by one holder reaches another holder's taps (self-check claim 6).
  * CHARGE: **0 of 150.** `make_brain(seed=0, refit_every=10, persist=False)`, `clock.phase`
    forced to 0.25, 150/150 wake ticks, with `AutonomousArm._percept_salience` LIVE (150 distinct
    values in [0.568351, 0.667237]) and then NEUTERED to the incumbent literal 0.6 (1 distinct).
    This faculty's OWN output was byte-identical across the pair on every channel: `ignited` 0/150,
    `global_availability` 0/150, `access_index` 0/150, the brain's `_access_ema` 0/150.
  * THE HARNESS IS NOT DEAD -- the positive control that makes the null trustworthy. Pinning the
    same producer ABOVE its live range moves this faculty hard: at 0.72 its own output differs on
    61/120 ticks, at 0.75 on 71/120, at 0.95 on **120/120** (`ignited` collapses from 120/120 True
    to 0/120). The flip point is 0.72, between 0.71 (0/120) and 0.72; the live signal's ceiling is
    0.667. So the null is a property of the RANGE, not of the probe.
  * WHY, and it is not fixable from this file: the bus gives its incumbent `self_boost +
    hysteresis` = 0.25 of defence, while the whole admissible band a specialist may claim is
    `1 - (self_boost + hysteresis) - ignition_threshold` = 0.20 wide. The defence exceeds the band,
    so whichever faculty first takes the slot holds it against every compliant competitor forever.
    `brainstem/autonomous_loop.py::_percept_salience` had already measured the same ceiling from
    the producer side; this is the same constant closing the SLOT-OWNERSHIP route as well as the
    direct-salience one. Both are properties of `thalamus/global_workspace.py`, which this module
    does not own.

Consequence, stated plainly so nobody re-derives it by surprise: repointed onto the shared bus this
faculty takes the single slot on its second step and holds it, so the arm's broadcast content goes
from the percept id to 'GLOBAL' and stays there. (That channel was already 1 distinct / 0 changes
over 150 ticks before the repoint and is 2 distinct / 1 change after, so nothing informative is
lost -- but it IS a change, and it buys this consumer nothing.) `_max_claim` is what keeps the
consequence bounded: a joining faculty may not make itself indisplaceable, and with it a 0.95
proximity threat still seizes the slot where uncapped it cannot (self-check claim 9).

Determinism: per-instance random.Random(seed) for the workspace; neuron spike codes are
crc32-seeded (no module-level / time seeding). ASCII output.

  Run: python -m snn_data_lake.cortex.access_consciousness      # (also run by run_selfchecks.py)
"""

import random
import zlib
from collections import deque

import numpy as np

from snn_data_lake.cortex.vsa_reasoning import D, bundle, cleanup, hv, sim
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

# phenomenal-consciousness vocabulary that MUST NEVER appear in the functional readout (claim 5).
_BANNED = ("conscious", "sentient", "phenomenal", "subjective", "aware", "feels", "qualia")


class AccessConsciousness:
    """A Global-Neuronal-Workspace access-consciousness index over spiking assemblies.

    Owns a `GlobalWorkspace` (ignition + broadcast) and a codebook of per-neuron spike-polarity
    hypervectors. `bind_assembly` GROWS a coalition of neurons and BUNDLES them into one workspace
    candidate whose salience grows with coherence/size; `step()` runs the ignition competition;
    the readout (`access_index` and `state`) is PURELY FUNCTIONAL -- global availability, ignition,
    integration -- and makes NO phenomenal / subjective-experience claim (see module docstring).
    """

    _LABEL = "GLOBAL"  # hashable workspace handle for the (single) conscious-content slot
    _SRC = "coalition"  # workspace source name for the grown assembly coalition

    def __init__(
        self,
        seed=0,
        ignition_threshold=0.55,
        n_taps=4,
        k_sat=5,
        noise=0.05,
        max_members=None,
        ws=None,
    ):
        """`ws` INJECTS the bus this faculty competes on. Default None keeps the standalone
        behaviour (this instance builds its own), which is what the self-check below runs.

        WHY THE INJECTION LIVES HERE AND NOT AT THE CALL SITE. This constructor registers
        `n_taps` subscriber callbacks on the workspace it ends up holding; rebinding `_ws` from
        outside after construction would leave every one of those taps attached to the abandoned
        instance, so `global_availability()` would report reach on a bus nothing broadcasts to.
        The handle therefore has to arrive before the taps are registered -- i.e. as a parameter.

        WHAT THIS CLOSES AND WHAT IT DOES NOT, because the two are easy to run together and the
        second half is the more useful finding. `tools/p6_gate.py` reports this module as a writer
        nobody reads, posting into a workspace it constructed for itself -- one of the private,
        mutually invisible instances the gate counts. This parameter is what makes that instance
        repointable at all, and the identity is asserted rather than assumed (`main()` claim 6).

        It does NOT close the charge. Neutering the one producer that reaches the shared bus
        (`AutonomousArm._percept_salience`, pinned to the incumbent literal 0.6) over 150 live wake
        ticks changed this faculty's OWN output on **0 of 150** ticks -- and that number is the
        same 0 of 150 BEFORE the repoint and AFTER it. Sharing the object was necessary and was not
        sufficient: the reason is the bus constant recorded in the module docstring, not the
        private instance. Claim (8) shows the same route DOES carry when the competitor's claim is
        large enough, so the mechanism works and the live signal is too small to use it.

        `ignition_threshold` is read back OFF the bus once one is injected rather than kept as the
        argument's value: it is the rheobase of the pool this faculty's submissions actually drive,
        and `cortex/brain.py::_update_access` anchors its salience on `ac.ignition_threshold`
        precisely to avoid duplicating that constant. Reading it through keeps that binding true on
        a shared bus instead of silently pointing at a number no pool uses."""
        self.seed = seed
        self.n_taps = max(2, int(n_taps))  # >=2 independent downstream taps (global reach)
        self.k_sat = max(2, int(k_sat))  # neuron count at which completeness saturates
        self.noise = noise  # bit-flip fraction of a concept-member spike code
        self._rng = random.Random(seed)  # (reserved) per-instance RNG, no global seeding
        self._ws = (
            GlobalWorkspace(ignition_threshold=ignition_threshold, seed=seed) if ws is None else ws
        )
        self._own_ws = ws is None  # False == competing on somebody else's bus
        self.ignition_threshold = float(self._ws.ignition_threshold)
        # downstream taps: each independent subscriber records the content it last received.
        self._taps = []
        for _ in range(self.n_taps):
            inbox = []
            self._ws.subscribe(inbox.append)
            self._taps.append(inbox)
        # Coalition grown across bind_assembly calls THIS CYCLE: (neuron_id, spike_code) pairs.
        #
        # "this cycle" is what the design intends and what nothing enforced: no cycle boundary ever
        # cleared this, so on the always-on daemon it grew for the life of the process. `_coherence`
        # is mean PAIRWISE similarity, so the tick cost is O(M^2 * D) with M unbounded -- measured on
        # the live brain at ~2,250 members after 321 wake ticks, where ONE bind_assembly took ~90
        # seconds and the whole brain advanced one tick per 90s.
        #
        # The members past the first handful bought nothing, which is what makes this a leak rather
        # than a trade: `_completeness` saturates at k_sat (5), and coherence over a large coalition
        # converges to the ~0.018 chance similarity of independent spike codes -- an order of
        # magnitude under `ignition_threshold`. So the tail could never change an ignition decision;
        # it only cost quadratic time and slowly diluted the coherence of the members that mattered.
        #
        # A bounded deque restores the intended per-cycle scope without needing a reset call site:
        # the coalition is the RECENT content, oldest members evicted. The default leaves generous
        # headroom over k_sat so completeness still saturates exactly as before.
        self._max_members = max(2 * self.k_sat, int(max_members or 8 * self.k_sat))
        self._members = deque(maxlen=self._max_members)
        self._n_assemblies = 0
        self._content_hv = None  # the bundled conscious-content spike code
        # last-step functional readout.
        self._ignited = False
        self._availability = 0.0
        self._integration = 0.0
        self._n_bound = 0
        self._last_claim = 0.0  # the salience actually posted (post-cap), for the self-check

    # -- neuron spike codes -----------------------------------------------------------------------
    def neuron_hv(self, nid):
        """Deterministic spike-polarity code for a neuron id.

        A plain id ("n0") is an atomic, ~orthogonal code. A concept-member id ("cat#3") is a
        NOISY COPY of the concept prototype hv("cat") with a small fraction of spike lines flipped
        -- so members of one concept are mutually coincident (a COHERENT assembly) while members of
        different concepts are ~orthogonal (an INCOHERENT assembly). crc32-seeded (reproducible)."""
        s = str(nid)
        proto_label, _, tag = s.partition("#")
        proto = hv(proto_label)
        if not tag:
            return proto
        v = list(proto)
        rng = random.Random(zlib.crc32(s.encode()))
        n_flip = int(round(self.noise * len(v)))
        for i in rng.sample(range(len(v)), n_flip):
            v[i] = -v[i]  # flip one spike line (up<->down)
        return v

    # -- coherence / integration proxies ----------------------------------------------------------
    #: How many times `_coherence` has taken the FUSED matrix branch, process-wide.
    #
    # ADDED 2026-08-05 because the self-check's fused-vs-scalar battery could NOT see the branch it
    # was written to validate. It re-derived the branch PREDICATE from the probe data
    # (`all(x in (1, -1) ...)`) and called that "the fused branch was exercised" -- a statement about
    # the INPUTS, never about the code. MEASURED: replacing `_coherence` with a plain delegation to
    # `_coherence_scalar` (the fusion deleted outright) left the whole self-check GREEN, because
    # `coh_drift` then compared the scalar definition against itself and the "took the fast path"
    # count never looked. That is the vacuous-precondition failure P8 guard (4) names, in the assert
    # added to close a P8 incident. The counter is what makes `coh_drift == 0.0` a measurement.
    #
    # It changes no return value, so the bit-exactness argument in `_coherence` is untouched: the
    # method is still a pure function of `hvs` in the only sense that claim uses (same codes -> same
    # float, no dependence on anything else). One int increment per ignition-cycle bind, against the
    # O(M^2 * D) matrix product it sits beside.
    _fused_calls = 0

    @staticmethod
    def _coherence(hvs):
        """Mean pairwise POSITIVE spike-coincidence similarity among member codes, in [0,1].

        High for a mutually coincident (coherent) assembly, ~0 for an orthogonal (scrambled) one.
        Binding is RELATIONAL: fewer than two neurons cannot be a bound assembly -> coherence 0.

        FUSED, bit-exactly (v4.80). This was M*(M-1)/2 scalar `sim` calls and, at the bounded M=40
        coalition, the single largest term in the live wake tick -- 26.3 ms of a measured 56.6 ms.
        For the +-1 spike codes this class produces, sim(a,b) == (2*coincidence(a,b) - D)/D ==
        (a.b)/D, the same integer identity the episodic-recall fusion used, so the whole pairwise
        table is ONE matrix product. It is exact rather than close: |a.b| <= D and D is a power of
        two, so each similarity is a dyadic multiple of 1/D that float64 holds exactly, and so is
        every partial sum of at most M^2/2 of them -- the sum is therefore order-independent and
        the closing division is the same single rounding the scalar loop performed. Codes that are
        not +-1 (no caller produces any) take the scalar path rather than silently relying on an
        identity that requires them.

        THE BIT-EXACTNESS IS NOW ASSERTED, NOT ARGUED (2026-08-04). `docs/POLICIES.md` lists this
        method under P8 as a live incident: the same `(a.b)/D` identity carries its algebra AND a
        live check in `cortex/vsa_reasoning.py` (`coincidence_many` vs `coincidence`), and was
        reused here claiming "FUSED, bit-exactly" with *no assertion anywhere comparing the fused
        branch against the scalar one*. The scalar definition is now `_coherence_scalar` -- ONE
        definition, used both as this method's fallback and as the control the self-check diffs
        against on the real member codes."""
        if len(hvs) < 2:
            return 0.0
        try:
            m = np.asarray(hvs, dtype=np.float64)
            if m.ndim == 2 and m.shape[1] == D and bool(np.all(np.abs(m) == 1.0)):
                g = (m @ m.T) / D  # g[i, j] == sim(hvs[i], hvs[j]), exactly
                pos = np.maximum(g[np.triu_indices(m.shape[0], 1)], 0.0)
                AccessConsciousness._fused_calls += (
                    1  # observed by the self-check, see _fused_calls
                )
                return float(pos.sum()) / int(pos.size)
        except (TypeError, ValueError):
            pass  # ragged / non-numeric codes -> the scalar definition below
        return AccessConsciousness._coherence_scalar(hvs)

    @staticmethod
    def _coherence_scalar(hvs):
        """The SCALAR definition of `_coherence`: mean pairwise positive `sim` over member codes.

        This is the fallback path of `_coherence` AND the control the self-check compares the fused
        matrix branch against. It is one definition rather than two (P2): the fused branch is an
        optimisation of THIS, and the assertion in main() is what makes that a measured claim
        instead of a prose one."""
        if len(hvs) < 2:
            return 0.0
        tot = 0.0
        n = 0
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                tot += max(0.0, sim(hvs[i], hvs[j]))
                n += 1
        return tot / n

    def _completeness(self, n_members):
        return min(1.0, n_members / self.k_sat)

    def _salience(self, coherence, completeness):
        # the workspace drive current: grows with assembly COHERENCE and (weakly) size. A coherent
        # assembly clears rheobase (ignition_threshold) with margin; an incoherent one falls below.
        return max(0.0, min(1.0, coherence * (0.75 + 0.25 * completeness)))

    def _max_claim(self):
        """The strongest claim this faculty may post: `1 - (self_boost + hysteresis)`, READ OFF THE
        BUS rather than chosen here.

        This is the bus's own displaceability invariant and it is not a new number: it is the same
        derivation `brainstem/autonomous_loop.py::_percept_salience` already applies to perception's
        claim, from the same two `GlobalWorkspace` attributes (P2 -- one definition of "the loudest
        a specialist may legitimately be", applied by every specialist, not a second policy). Above
        it, the incumbent an ignition creates carries more recurrent defence than the widest claim
        any competitor can make, so it can never be displaced by anything.

        MEASURED, which is why this exists at all. Repointed onto the shared bus without it, this
        faculty posts the salience `cortex/brain.py::_update_access` computes -- 1.0 on the early
        wake ticks -- and traced over the live bus it seized the single slot on step 2 and held it
        for every step after: perception's broadcast went from the percept id to 'GLOBAL'
        permanently, and a 0.95 proximity threat could no longer take the slot either (drive 1.10
        against 0.95). A faculty joining a shared bus must not be able to silence the bus.

        Degenerate inputs, checked because a P1 rewrite in this tree shipped a `1e12` singularity:
        there is no division here at all. A bus with `self_boost + hysteresis >= 1` leaves no room
        between audible and displaceable, and the floor returns the rheobase -- the same answer
        `_percept_salience` gives in that case -- rather than a cap of 0 that would silence every
        submission including this faculty's own. A ZERO / EMPTY / SILENT / SINGLE-member assembly
        produces salience 0.0 upstream and `min(0.0, cap)` leaves it 0.0, so the cap cannot lift a
        sub-threshold assembly over the rheobase; it only ever lowers."""
        ws = self._ws
        hi = 1.0 - (float(ws.self_boost) + float(ws.hysteresis))
        return max(float(ws.ignition_threshold), min(1.0, hi))

    # -- GROW + BUNDLE + submit -------------------------------------------------------------------
    def bind_assembly(self, neuron_ids, salience=None):
        """GROW the conscious-content coalition with these neurons and BUNDLE it into ONE workspace
        candidate. Salience defaults to the coherence/size-driven drive current (pass `salience` only
        to inject a fixed background distractor). Returns the workspace content handle."""
        for nid in neuron_ids:
            self._members.append((str(nid), self.neuron_hv(nid)))
        self._n_assemblies += 1
        codes = [h for _, h in self._members]
        self._content_hv = bundle(codes)  # spike superposition -> ONE content
        # The coherence/size drive current is the DEFAULT salience only, and it is not a cheap
        # default: `_coherence` is the O(M^2) pairwise table. The live wake tick
        # (cortex/brain.py::_update_access) always passes an explicit novelty-gated salience, so
        # this computed the expensive default on every tick and then discarded it -- a measured
        # 13.1 ms of a 56.6 ms tick. Computing it only when it is the value actually used is
        # bit-exact: `_coherence` is a pure function of `codes` with no other effect.
        if salience is None:
            sal = self._salience(self._coherence(codes), self._completeness(len(codes)))
        else:
            sal = max(0.0, min(1.0, float(salience)))
        sal = min(sal, self._max_claim())  # stay DISPLACEABLE on the bus -- see _max_claim
        self._ws.submit(self._SRC, self._LABEL, sal)  # post the candidate for ignition
        self._last_claim = sal
        return self._LABEL

    def content_vector(self):
        """The bundled spike code of the current conscious-content candidate (or None)."""
        return self._content_hv

    @property
    def workspace(self):
        """The bus this faculty competes on -- THE object, so a caller can check IDENTITY.

        Exists so "is this faculty on the shared bus?" is answerable by `ac.workspace is
        brain.workspace` rather than by reaching into `_ws`. Equality is not the question and
        would answer it wrongly: two freshly built `GlobalWorkspace`es with the same seed are
        indistinguishable by value while connecting nothing, which is exactly the private-instance
        defect P6 counts. `cortex/live_brain.py` and `cortex/brain.py` expose the same handle under
        the same name for the same reason."""
        return self._ws

    # -- one ignition cycle -----------------------------------------------------------------------
    def step(self):
        """Advance the workspace one tick: ignition competition + all-or-none global broadcast.
        Returns whether THIS faculty's content ignited (True/False) and refreshes the readout.

        IGNITION IS THE WINNER'S IDENTITY, NOT MERELY A NON-EMPTY SLOT. `_ws.step()` returns
        whichever content seized the single slot; this faculty is access-conscious of its own
        coalition only when that winner IS its coalition. On a private bus the two tests are the
        same test -- this class submits under exactly one content handle, so the slot holds
        `_LABEL` or nothing -- and `main()` claim (7) asserts that equivalence by counting, rather
        than asserting it in prose. On a SHARED bus they come apart, and that difference is the
        whole point of the injection: another faculty's louder submission takes the slot and this
        faculty's coalition stays local, which is what "competition for a limited-capacity
        workspace" means and what the private instance made structurally impossible.

        The tap scan is unchanged in kind and now names the content it is looking for. A tap that
        last received somebody ELSE's broadcast has not received THIS content, so counting `is not
        None` would report global availability of 1.0 for a coalition that never ignited the moment
        the bus carries anyone else's traffic."""
        content = self._ws.step()  # broadcasts to every subscribed tap
        self._ignited = content is not None and content == self._LABEL
        received = sum(1 for tap in self._taps if tap and tap[-1] == self._LABEL)
        self._availability = (received / self.n_taps) if self._ignited else 0.0
        if self._ignited:
            codes = [h for _, h in self._members]
            self._integration = max(
                0.0, min(1.0, self._coherence(codes) * self._completeness(len(codes)))
            )
            self._n_bound = self._n_assemblies
        else:
            self._integration = 0.0
            self._n_bound = 0
        return self._ignited

    # -- purely functional readout (NO phenomenal / subjective-experience field, ever) ------------
    def ignited(self):
        """True iff the content crossed the all-or-none ignition threshold and was broadcast."""
        return self._ignited

    def global_availability(self):
        """Fraction in [0,1] of subscribed downstream taps that received the broadcast this tick."""
        return self._availability

    def n_bound(self):
        """Number of grown assemblies fused into the currently ignited conscious content (0 if none)."""
        return self._n_bound

    def integration(self):
        """Cheap Phi-like PROXY in [0,1] for how bound/coherent the ignited content is
        (mean pairwise spike coincidence x completeness). This is a PROXY, explicitly NOT IIT Phi."""
        return self._integration

    def access_index(self):
        """Honest ACCESS-consciousness index in [0,1] = ignition(all-or-none gate) *
        global_availability * integration. RISES as more coherent neurons are grown and bundled.
        Measures global availability only -- it is NOT a measure of phenomenal experience."""
        gate = 1.0 if self._ignited else 0.0
        return max(0.0, min(1.0, gate * self._availability * self._integration))

    def state(self):
        """Functional snapshot. NO 'conscious'/'sentient'/'phenomenal'/'subjective'/'aware'/
        'feels'/'qualia' field -- access consciousness only (see module docstring + claim 5)."""
        return {
            "ignited": self._ignited,
            "global_availability": self._availability,
            "n_bound": self._n_bound,
            "integration": self._integration,
            "access_index": self.access_index(),
        }


# ============================== un-riggable self-check (seed=0) ==================================
def _fresh(**kw):
    return AccessConsciousness(seed=0, **kw)


def _coherent(concept, k):
    return [f"{concept}#{i}" for i in range(k)]


def _scrambled(k):
    return [f"c{i}#0" for i in range(k)]  # k DIFFERENT concepts -> ~orthogonal (incoherent)


def _record_slot(ac, store):
    """Record every content the bus `ac` is on puts in the slot, without adding a step."""
    real = ac._ws.step

    def spy():
        c = real()
        store.append(c)
        return c

    ac._ws.step = spy


def _bus(**kw):
    """THE ONE GlobalWorkspace construction site the self-check uses, for the same reason as
    `_post`: `tools/p6_gate.py` censuses instantiation sites per module and reports how many are
    PRIVATE and mutually invisible. Seven scenario-local buses in this self-check would have added
    seven to the very count this work exists to reduce -- a module's test scaffolding must not
    appear in the tree's private-workspace census as if it were live wiring."""
    return GlobalWorkspace(seed=0, **kw)


def _post(ws, posts):
    """THE ONE submit site the self-check uses to stand in for the OTHER faculties on a shared bus.

    Deliberately a single call site rather than a submit per scenario. `tools/p6_gate.py` counts
    `submit` call sites per module as this module's BROADCAST, and it only discounts self-check
    writes inside the two bus modules themselves -- so scattering the simulated competitors' posts
    would have raised this module's write count from 1 to 9 while its real broadcast stayed one
    site. P6's own verdict line is that a rising write count is not progress, and inflating it with
    test scaffolding is the cheapest possible way to look busy in that census."""
    for source, content, salience in posts:
        ws.submit(source, content, salience)


def _shared_run(competitor_salience, n=12, claim=0.75):
    """Replay the LIVE two-step cadence on ONE shared bus: another faculty submits and the bus
    arbitrates (the arm's step), then this faculty binds and steps (Brain._update_access).
    Returns this faculty's OWN per-tick output."""
    ws = _bus()
    ac = AccessConsciousness(seed=0, ws=ws)
    out = []
    for _ in range(n):
        _post(ws, [("perception", "unit0", competitor_salience), ("salience", "threat", 0.0)])
        ws.step()
        ac.bind_assembly(_coherent("cat", 3), salience=claim)
        out.append((ac.step(), round(ac.access_index(), 6), round(ac.global_availability(), 6)))
    return out


def _slot_after_threat(claim, threat=0.95, warm=8, psal=0.6):
    """Warm a shared bus with this faculty holding the slot at `claim`, then fire a proximity
    threat. Returns what the slot holds afterwards -- 'threat' iff the bus is still displaceable."""
    ws = _bus()
    ac = AccessConsciousness(seed=0, ws=ws)
    for _ in range(warm):
        _post(ws, [("perception", "unit0", psal), ("salience", "threat", 0.0)])
        ws.step()
        # posted RAW, bypassing bind_assembly's cap, so capped and uncapped arms are comparable
        _post(ws, [(ac._SRC, ac._LABEL, claim)])
        ac.step()
    _post(ws, [("perception", "unit0", psal), ("salience", "threat", threat)])
    return ws.step()


def main():
    # ---- (1) IGNITION IS ALL-OR-NONE: coherent+salient ignites (reaches >=2 taps); weak/incoherent
    #          does NOT. The control (a scrambled assembly) proves ignition is a real threshold. ----
    strong = _fresh()
    strong.bind_assembly(_coherent("cat", 5))  # 5 coincident neurons -> supra-threshold
    strong.step()
    strong_reach = sum(1 for tap in strong._taps if tap and tap[-1] is not None)

    weak = _fresh()
    weak.bind_assembly(_scrambled(5))  # 5 orthogonal neurons -> sub-threshold
    weak.step()

    # ---- (2) BUNDLING BINDS: every bundled neuron is recoverable from the ONE content; a
    #          non-bundled distractor is NOT (control). Pure VSA superposition property. ----------
    ac = _fresh()
    ids = [f"n{i}" for i in range(5)]  # 5 distinct (orthogonal) neurons
    codes = [ac.neuron_hv(n) for n in ids]
    content = bundle(codes)  # bind them into ONE conscious content
    distractor = ac.neuron_hv("distractor")  # never bundled in
    member_sims = [sim(content, c) for c in codes]
    distractor_sim = sim(content, distractor)
    codebook = dict(zip(ids, codes))
    codebook["distractor"] = distractor
    cleaned, _ = cleanup(content, codebook)  # snaps to a bundled member, not the distractor
    bundling_binds = (
        min(member_sims) > distractor_sim
        and all(m > 0.0 for m in member_sims)
        and distractor_sim < 0.15
        and cleaned in ids
    )

    # ---- (3) GLOBAL AVAILABILITY: after ignition the content is readable from MULTIPLE independent
    #          taps; on a non-igniting input it is not (control). ---------------------------------
    avail_ignited = strong.global_availability()
    avail_nonignited = weak.global_availability()

    # ---- (4) access_index RISES with integration: a coherent assembly beats a SCRAMBLED one of the
    #          same size (control), and the index climbs monotone-ish as coherent neurons accumulate.
    coherent_ac = _fresh()
    coherent_ac.bind_assembly(_coherent("dog", 5))
    coherent_ac.step()
    index_coherent = coherent_ac.access_index()

    scrambled_ac = _fresh()
    scrambled_ac.bind_assembly(_scrambled(5))
    scrambled_ac.step()
    index_scrambled = scrambled_ac.access_index()

    trend = []
    for k in (2, 3, 4, 5):
        grow = _fresh()
        grow.bind_assembly(_coherent("grow", k))
        grow.step()
        trend.append(grow.access_index())

    # ---- multi-assembly coalition: two coherent assemblies fuse into one reportable content ------
    coalition = _fresh()
    coalition.bind_assembly(_coherent("bird", 3))
    coalition.bind_assembly([f"bird#{i}" for i in range(3, 6)])
    coalition.step()

    # ---- (4b) THE FUSED COHERENCE IS BIT-EXACT vs THE SCALAR DEFINITION. --------------------------
    # POLICIES.md P8 names this file: the (a.b)/D fusion identity was reused here with the claim
    # "FUSED, bit-exactly" and NO assertion comparing the two branches. This is that assertion, on
    # the real member codes the live tick produces (coherent, scrambled, mixed, and the boundary
    # sizes 2 and 1), plus the max observed drift so a regression reports a number rather than a
    # bare False.
    coh_probe = []
    for maker in (_coherent("cat", 5), _coherent("dog", 12), _scrambled(9)):
        ac_p = _fresh()
        ac_p.bind_assembly(maker)
        coh_probe.append([h for _, h in ac_p._members])
    mixed = _fresh()
    mixed.bind_assembly(_coherent("bird", 4))
    mixed.bind_assembly(_scrambled(4))
    coh_probe.append([h for _, h in mixed._members])
    coh_probe.append(coh_probe[0][:2])  # the len==2 boundary
    coh_probe.append(coh_probe[0][:1])  # the len<2 early return
    # `fused_eligible` is a property of the PROBE CORPUS (are these codes the +-1 shape the identity
    # needs?); `fused_took_fast_path` is a property of the CODE (did _coherence actually take that
    # branch?). They must be counted separately and compared. Counting only the first -- which this
    # block did until 2026-08-05 -- makes `coh_drift == 0.0` vacuous the moment the fused branch stops
    # being reached, and MEASURED, deleting the fusion outright left the whole self-check green.
    fused_eligible = sum(
        1 for c in coh_probe if len(c) >= 2 and all(x in (1, -1) for h in c for x in h)
    )
    _fused_before = AccessConsciousness._fused_calls
    coh_drift = 0.0
    for codes in coh_probe:
        fused = AccessConsciousness._coherence(codes)
        scalar = AccessConsciousness._coherence_scalar(codes)
        coh_drift = max(coh_drift, abs(fused - scalar))
    fused_took_fast_path = AccessConsciousness._fused_calls - _fused_before

    # ---- (6) THE INJECTED BUS IS THE SAME OBJECT, and the taps follow it. -------------------------
    # P6's finding is that faculties hold PRIVATE, mutually invisible workspaces, so the check that
    # matters is IDENTITY. Equality would pass on two freshly built seed-0 buses that connect
    # nothing -- which is the defect itself. The RED is the default instance built beside them.
    inj_ws = _bus()
    inj_a = AccessConsciousness(seed=0, ws=inj_ws)
    inj_b = AccessConsciousness(seed=1, ws=inj_ws)
    inj_own = _fresh()
    identity_shared = inj_a.workspace is inj_ws and inj_b.workspace is inj_ws
    identity_red = inj_own.workspace is not inj_ws and inj_own.workspace is not None
    own_flags = (inj_a._own_ws, inj_b._own_ws, inj_own._own_ws)
    # sharing must be REAL, not nominal: A's broadcast has to reach B's taps, which only happens if
    # both tap sets were registered on the one object.
    inj_a.bind_assembly(_coherent("cat", 5))
    inj_a.step()
    cross_tap = bool(inj_b._taps[0]) and inj_b._taps[0][-1] == AccessConsciousness._LABEL
    # the rheobase is read OFF the bus, so the anchor cortex/brain.py uses is the pool's own. RED by
    # construction: the constructor argument says 0.55 and the bus says 0.42.
    odd_ws = _bus(ignition_threshold=0.42)
    odd_ac = AccessConsciousness(seed=0, ignition_threshold=0.55, ws=odd_ws)
    threshold_readthrough = odd_ac.ignition_threshold == 0.42

    # ---- (7) THE WINNER-IDENTITY IGNITION TEST IS BEHAVIOUR-NEUTRAL ON A PRIVATE BUS. ------------
    # `content is not None` and `content == _LABEL` are the same test when this faculty is the only
    # submitter, and a DIFFERENT test the moment it is not. Both halves are counted: agreement on
    # the private bus, and at least one DISAGREEMENT on a shared one -- without which the first
    # assert is vacuous (the `_fused_calls` lesson, applied to a second claim in this file).
    priv_seen = []
    priv_ac = _fresh()
    _record_slot(priv_ac, priv_seen)
    for t in range(10):
        priv_ac.bind_assembly(_coherent("cat", 3) if t % 2 == 0 else _scrambled(3))
        priv_ac.step()
    priv_agree = sum(1 for c in priv_seen if (c is not None) == (c == AccessConsciousness._LABEL))
    sh_seen = []
    sh_ws = _bus()
    sh_ac = AccessConsciousness(seed=0, ws=sh_ws)
    _record_slot(sh_ac, sh_seen)
    for _ in range(10):
        _post(sh_ws, [("perception", "unit0", 0.95)])  # a louder faculty takes the one slot
        sh_ws.step()
        sh_ac.bind_assembly(_coherent("cat", 3), salience=0.60)
        sh_ac.step()
    sh_disagree = sum(1 for c in sh_seen if (c is not None) != (c == AccessConsciousness._LABEL))

    # ---- (8) ON A SHARED BUS ANOTHER FACULTY IS DECISIVE FOR THIS ONE'S OWN OUTPUT. --------------
    # The in-module form of the neuter-A/B. `hi_run`'s competitor is loud enough to hold the single
    # slot, `lo_run`'s is not; `null_run` re-runs `lo_run`'s competitor unchanged and MUST be
    # identical, or the difference above is measuring the harness rather than the bus.
    lo_run = _shared_run(0.60)
    hi_run = _shared_run(0.95)
    null_run = _shared_run(0.60)
    shared_diff = sum(1 for a, b in zip(lo_run, hi_run) if a != b)
    shared_null = sum(1 for a, b in zip(lo_run, null_run) if a != b)

    # ---- (9) A FACULTY JOINING A SHARED BUS MUST NOT BE ABLE TO SILENCE IT. ----------------------
    # `_max_claim` caps this faculty's post at the bus's own displaceability bound. The RED is the
    # SAME replay with the cap removed, and it is run, not argued.
    cap = _fresh()._max_claim()
    capped_after_threat = _slot_after_threat(cap)
    uncapped_after_threat = _slot_after_threat(1.0)
    # the cap only ever LOWERS: a scrambled (sub-rheobase) assembly must stay local on any bus,
    # including one whose defence leaves no room and whose cap therefore floors at the rheobase.
    cap_floor_ws = _bus(self_boost=0.5, hysteresis=0.6)
    cap_floor_ac = AccessConsciousness(seed=0, ws=cap_floor_ws)
    cap_floor_ac.bind_assembly(_scrambled(5))
    cap_floor_ignites = cap_floor_ac.step()
    cap_floor = cap_floor_ac._max_claim()
    cap_open = AccessConsciousness(seed=0, ws=_bus(self_boost=0.0, hysteresis=0.0))._max_claim()

    # ---- (5) HONEST CEILING + DETERMINISM. --------------------------------------------------------
    st = strong.state()
    gate = 1.0 if st["ignited"] else 0.0
    reproduced = gate * st["global_availability"] * st["integration"]
    no_phenomenal_key = all(all(tok not in key for tok in _BANNED) for key in st)
    a1 = _fresh()
    a1.bind_assembly(_coherent("cat", 5))
    a1.step()
    a2 = _fresh()
    a2.bind_assembly(_coherent("cat", 5))
    a2.step()
    deterministic = a1.state() == a2.state()

    # -------------------------------------- ASCII report ------------------------------------------
    rows = [
        ("strong-coherent(5)", strong.state()),
        ("weak-incoherent(5)", weak.state()),
        ("coalition bird(3+3)", coalition.state()),
    ]
    print("=" * 78)
    print(
        "ACCESS CONSCIOUSNESS -- Global Neuronal Workspace signature (FUNCTIONAL, not phenomenal)"
    )
    print("=" * 78)
    print(
        "pipeline: GROW neurons -> BUNDLE (VSA) -> IGNITE (all-or-none) -> BROADCAST -> ACCESS index"
    )
    print(
        f"taps={strong.n_taps}  ignition_threshold={strong.ignition_threshold}  "
        f"index = ignition_gate * global_availability * integration\n"
    )
    print(f"{'scenario':<20}{'ignited':>9}{'avail':>8}{'n_bound':>9}{'integ':>8}{'access_idx':>12}")
    print("-" * 78)
    for name, s in rows:
        print(
            f"{name:<20}{str(s['ignited']):>9}{s['global_availability']:>8.2f}"
            f"{s['n_bound']:>9}{s['integration']:>8.3f}{s['access_index']:>12.3f}"
        )
    print("-" * 78)
    print(
        f"bundling recoverability: min member sim {min(member_sims):.3f} > distractor {distractor_sim:.3f}"
        f"  (cleanup -> {cleaned!r})"
    )
    print(
        f"availability: ignited {avail_ignited:.2f} vs non-igniting {avail_nonignited:.2f}"
        f"  (broadcast reached {strong_reach}/{strong.n_taps} taps)"
    )
    print(
        f"index coherent {index_coherent:.3f} > scrambled {index_scrambled:.3f}"
        f"  |  accumulate-coherent-neurons trend (k=2..5): "
        f"{', '.join(f'{x:.3f}' for x in trend)}"
    )
    print(
        f"coherence fusion: fused vs scalar max|d| = {coh_drift:.1e} over {len(coh_probe)} member "
        f"sets ({fused_took_fast_path} of {fused_eligible} eligible OBSERVED taking the fused branch)"
    )
    print(
        f"shared bus   : injected handle is THE object (A and B taps on one instance: "
        f"{cross_tap}); a default instance is NOT it ({identity_red}); rheobase read off the bus "
        f"({odd_ac.ignition_threshold})"
    )
    print(
        f"winner-identity ignition: private bus agrees with 'slot non-empty' on "
        f"{priv_agree}/{len(priv_seen)} steps; on a shared bus they DISAGREE on "
        f"{sh_disagree}/{len(sh_seen)}"
    )
    print(
        f"competitor decisive: own output differs on {shared_diff}/{len(lo_run)} ticks between a "
        f"0.60 and a 0.95 competitor; unchanged competitor -> {shared_null}/{len(lo_run)}"
    )
    print(
        f"displaceable: cap={cap} -> a 0.95 threat takes the slot ({capped_after_threat!r}); "
        f"UNCAPPED at 1.0 it cannot ({uncapped_after_threat!r}); cap floor={cap_floor} "
        f"open={cap_open}"
    )
    print(
        "MEASURES ACCESS (global availability / ignition), NOT phenomenal experience "
        "-- no subjective-experience claim."
    )
    print("=" * 78)

    # ------------------------------------- assertions ---------------------------------------------
    # (1) all-or-none ignition + control.
    assert strong.ignited(), (
        "a strong, coherent bundled assembly must IGNITE (all-or-none broadcast)"
    )
    assert strong_reach >= 2, f"ignition must broadcast to >=2 downstream taps, got {strong_reach}"
    assert not weak.ignited(), (
        "a weak/incoherent assembly must NOT ignite (ignition is a real threshold)"
    )
    # (2) bundling genuinely binds neurons into ONE recoverable content; the distractor stays out.
    assert bundling_binds, (
        "bundling must bind every member recoverably while excluding the distractor"
    )
    # (3) global availability is present after ignition, absent otherwise.
    assert avail_ignited > avail_nonignited, (
        "ignited content must be globally available, non-igniting not"
    )
    assert avail_ignited == 1.0 and avail_nonignited == 0.0, (
        "availability must be all-or-none across taps"
    )
    # (4) access_index rises with integration; coherent beats scrambled; index in [0,1]; trend climbs.
    assert index_coherent > index_scrambled + 0.2, (
        f"coherent index must clearly beat scrambled ({index_coherent:.3f} vs {index_scrambled:.3f})"
    )
    assert 0.0 <= index_coherent <= 1.0 and 0.0 <= index_scrambled <= 1.0, (
        "access_index must be in [0,1]"
    )
    assert trend[-1] > trend[0], f"index must rise as coherent neurons accumulate: {trend}"
    assert all(trend[i + 1] >= trend[i] - 1e-9 for i in range(len(trend) - 1)), (
        f"the accumulation trend must be monotone-ish (non-decreasing): {trend}"
    )
    assert coalition.ignited() and coalition.n_bound() == 2, (
        "two coherent assemblies must fuse (n_bound==2)"
    )
    # (4b) the fused matrix branch must EQUAL the scalar definition, bit for bit.
    #      Two separate preconditions before the equality means anything, and conflating them is what
    #      made the first version of this block unfalsifiable (see `_fused_calls`):
    assert fused_eligible >= 4, (
        f"the probe corpus must contain at least 4 member sets of the +-1 shape the fused identity "
        f"requires, or the comparison below is scalar-vs-scalar (eligible {fused_eligible}/"
        f"{len(coh_probe)})"
    )
    assert fused_took_fast_path == fused_eligible, (
        f"_coherence must ACTUALLY take the fused branch on every eligible probe -- observed "
        f"{fused_took_fast_path} of {fused_eligible}. A fused branch that is never reached makes the "
        f"bit-exactness assert below compare _coherence_scalar against itself"
    )
    assert coh_drift == 0.0, (
        f"_coherence's fused branch must be BIT-EXACT against _coherence_scalar "
        f"(max |fused - scalar| = {coh_drift:.3e} over {len(coh_probe)} member sets)"
    )
    # (6) the injected bus is THE object, not a copy, and the taps live on it.
    assert identity_shared, "an injected workspace must be held by IDENTITY, not copied"
    assert identity_red, (
        "a default-constructed instance must NOT be on the injected bus -- if this passes "
        "trivially the identity assert above is not discriminating anything"
    )
    assert own_flags == (False, False, True), f"_own_ws must track the injection: {own_flags}"
    assert cross_tap, (
        "A's broadcast must reach B's taps -- two instances sharing a bus by NAME while holding "
        "different objects is exactly the private-workspace defect P6 counts"
    )
    assert threshold_readthrough, (
        f"ignition_threshold must be read off the injected bus (0.42), not kept from the "
        f"constructor argument (0.55): got {odd_ac.ignition_threshold}"
    )
    # (7) the ignition test changed meaning without changing behaviour on the private path.
    assert priv_agree == len(priv_seen) and len(priv_seen) == 10, (
        f"on a private bus 'slot non-empty' and 'the slot is MINE' must agree on every step: "
        f"{priv_agree}/{len(priv_seen)}"
    )
    assert sh_disagree >= 1, (
        f"the two tests must actually COME APART on a shared bus, or the equivalence asserted "
        f"above is vacuous and the winner-identity test is untested: {sh_disagree}/{len(sh_seen)}"
    )
    # (8) another faculty's claim changes THIS faculty's own output -- with its own null control.
    assert shared_diff >= len(lo_run) // 2, (
        f"a competitor that holds the single slot must change this faculty's OWN output on at "
        f"least half the ticks, got {shared_diff}/{len(lo_run)}"
    )
    assert shared_null == 0, (
        f"an UNCHANGED competitor must reproduce the run exactly, or the difference above is the "
        f"harness and not the bus: {shared_null}/{len(lo_run)}"
    )
    # (9) joining a shared bus must not silence it -- the uncapped arm is the reachable RED.
    assert capped_after_threat == "threat", (
        f"with the claim capped at the bus's own displaceability bound, a 0.95 proximity threat "
        f"must still seize the slot, got {capped_after_threat!r}"
    )
    assert uncapped_after_threat != "threat", (
        f"the cap must be load-bearing: UNCAPPED at 1.0 the same threat must FAIL to take the "
        f"slot, or claim (9) is asserting something that was true anyway (got "
        f"{uncapped_after_threat!r})"
    )
    assert not cap_floor_ignites and cap_floor == cap_floor_ws.ignition_threshold, (
        f"the cap must only ever LOWER a claim: on a bus with no displaceable room it floors at "
        f"the rheobase ({cap_floor}) and a sub-threshold assembly still stays local "
        f"(ignited={cap_floor_ignites})"
    )
    assert cap_open == 1.0, f"a bus with no incumbent defence caps at 1.0, got {cap_open}"
    # (5) honest ceiling: no phenomenal field; index reproduces from the 3 functional measures alone.
    assert no_phenomenal_key, f"state() must contain NO phenomenal key {_BANNED}: {list(st)}"
    assert abs(st["access_index"] - reproduced) < 1e-9, (
        "access_index must be a pure function of ignition * global_availability * integration"
    )
    assert deterministic, "two seed=0 instances must reach identical state after the same inputs"

    print(
        "self-check OK: "
        f"all-or-none ignition (strong ignites/reaches {strong_reach} taps, weak does not); "
        f"bundling binds {len(ids)} neurons (distractor excluded); global availability "
        f"{avail_ignited:.1f} vs {avail_nonignited:.1f}; access_index coherent {index_coherent:.3f} > "
        f"scrambled {index_scrambled:.3f}, accumulation trend {trend[0]:.3f}->{trend[-1]:.3f}; "
        "functional-only (no phenomenal claim); deterministic"
    )


if __name__ == "__main__":
    main()
