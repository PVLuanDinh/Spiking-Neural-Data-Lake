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

  * faculties/global_workspace.GlobalWorkspace -- LIF pools that COMPETE and, above rheobase,
    IGNITE into sustained firing and BROADCAST to every subscriber (all-or-none global
    availability). We drive it with a salience current and subscribe N downstream taps.
  * faculties/vsa_reasoning -- spike-polarity hypervectors: `hv` (atomic spike code), `bundle`
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

ACCESS IS NOT FOCUS -- the dissociation this module can now state (2026-08-06)
=============================================================================
`faculties/global_workspace.py` holds up to `capacity` contents at once and RANKS them. Those are
two different facts about one workspace and this file reports them as two:

  * `ignited()` / `state()["ignited"]` -- ACCESS. My content holds a workspace slot, so it was
    broadcast to every subscriber. This is Dehaene's global availability and it is what
    `access_index` is gated on.
  * `focused()` / `state()["focused"]` -- FOCUS. My content is the RANKED HEAD: what the brain is
    currently thinking about, the same quantity `engine/live_brain.py::thought` reads.

UNDER ONE SLOT THEY WERE THE SAME SENTENCE, which is exactly why this file got it wrong. When the
workspace went multi-slot, `_ignited` kept reading the head and silently stopped meaning access.
MEASURED then: with one peer faculty on the bus this faculty was in `broadcast_all()` on 12 of 12
ticks while `ignited()` read False on 12 of 12, and renaming that peer moved it to True on 11 of 12
-- because two compliant pools SATURATE their sustained-firing counts, so the arbiter's rank key
falls through to the source string and the head is settled lexicographically. A
consciousness-relevant flag, and the `access_index` behind it (0.000000 <-> 0.839103), were being
decided by a string comparison.

FIXING THE FLAG ALONE WOULD HAVE BEEN THEATRE, and claim (10) runs that failure rather than
describing it. The bus delivered only the head to subscribers, so on precisely the ticks where this
faculty is admitted-but-not-head every tap held somebody else's content, `global_availability` read
0.0, and `access_index = gate * availability * integration` stayed 0.0 -- "ignited" beside no
reach. So the BUS now delivers its admitted SET on the one channel it already had, and this file's
tap scan asks the question that channel can answer.

WHAT CLAIM (10) MEASURES, all run, and RE-DERIVED 2026-08-06 when the arbiter stopped deciding the
head by name. THE LEVER IS THE PEER'S VOLUME: with the peer neutered (0.00) this faculty has ACCESS
and FOCUS on 12 of 12 ticks; against a 0.95 peer it holds a slot on 12 of 12 and the head on 0 of 12
-- ACCESS WITHOUT FOCUS on every tick -- with the `access_index` BIT-IDENTICAL across the entire
volume sweep from 0.00 to 0.95. Renaming that peer to sort after this faculty now moves NOTHING, on
either readout (FOCUS 0 of 12, ACCESS 0 of 12), which is the two-sided form of the string test.
FOUR REDs run beside it -- a ONE-slot bus never separates access from focus (0 of 12), the retired
head-only delivery keeps membership at 12 of 12 while zeroing the index on all of them, the RETIRED
RANK KEY reconstructed on the same bus moves FOCUS on 1 of 12 under that same rename (the number has
tracked every mechanism change and each value is that arrangement's honest tie count: 11 of 12
before the inhibitory rheobase, 12 of 12 on the rheobase under the retired two-competition harness
cadence, and 1 of 12 since the harness converted to the live one-competition clock -- settling
TOGETHER, the 0.95 peer and this 0.75 faculty tie on the clipped counts only on the first tick,
before their defences diverge; one tick is one reachable red, disclosed as thin rather than
padded), and (added 2026-08-06 by an adversarial re-run)
THE HEAD MUST BE THE PEER'S CONTENT, whose red is this faculty admitted BENEATH ITS OWN RETRACTED
CONTENT with no peer on the bus at all -- a two-step window found by sweeping, at claim 0.55, which
is the cap FLOOR exactly: the rheobase left that window zero slack, so if the floor ever moves this
red must be re-swept rather than assumed. Without that conjunct "admitted and not the head" is
equally true of a
faculty out-ranked by its own history, which is hysteresis and not a dissociation; the arm avoided it
only by re-posting ONE content, which was an unasserted property of the harness until now.

THE CRITERION'S GREEN STATE USED TO REQUIRE THE DEFECT, which is why it was re-derived rather than
kept. It asserted `focus_moved >= 1` with the peer RENAME as its lever, so it passed only on an
arbiter that settles a consciousness-relevant flag lexicographically: there is no arbiter that is
both correct and green on it. P8/F11 -- the PROPERTY (access and focus are two facts, and neither
may be decided by a name) is real, the PROXY was pathology-shaped.

IS THE DEFECT FIRING ON THE LIVE BRAIN TODAY? NO -- ARMED AND LATENT, and saying so is the point of
measuring it. A/B'd on the canonical brain against a shim that restores the retired contract exactly
(head-only delivery AND `_ignited` reading the head): un-repointed, 120 ticks / 56 awake, `thought`
names the submitted percept on 56 of 56 and the crc32 of the thought sequence, of the workspace-set
sequence and of the whole legacy `state()` readout are IDENTICAL between the two contracts -- because
`engine/brain.py::_update_access` builds this faculty with no `ws=`, so its bus has ONE submitter and
the workspace is a 1-tuple on every tick. The 2026-08-06 ARBITER repair is byte-identical there too,
re-measured in BOTH content-key regimes. So this fix buys nothing measurable on any bus this tree can
currently build; the REPOINT that would build one is a measured null recorded at
`engine/brain.py::_update_access` -- swept across the whole compliant band, this faculty is admitted
on 0 of 56 ticks below a 0.6825 ceiling and is the HEAD on every tick it is admitted above it, with a
co-existence window ONE TICK of 56 wide. Credit is claimed for the criterion, never for a live gain
that was measured at zero.

WHAT DID MOVE ON THE LIVE BRAIN, and it is the other half of this pass. This faculty posted a FROZEN
content key, so `faculties/global_workspace.py::_defended` could never withdraw its tenancy subsidy
and its pool was PERMANENTLY SELF-DEFENDED -- the same defect that file closed for perception, still
live here. `content_key` keys the pool on the BUNDLE instead. Consequence over 56 live wake ticks:
`ignited` goes 56 of 56 -- ONE distinct value, a flag with no reachable False, read by TEN
consumers -- to 44 of 56, with 8 of the 12 new lapses posting STRICTLY BELOW the rheobase. `thought`
is unchanged to the byte. `_access_ema` moves 0.488958 -> 0.287282, which is a fiat leaving rather
than a signal collapsing, and the per-reader attribution of that move is tabulated at
`engine/brain.py::_update_access` (two of the ten readers are inert in BOTH directions; a prior
version of that table said three of eight, and both figures were wrong -- see there).

WHAT THE BRAIN SAYS ABOUT ITSELF MOVED WITH IT, traced end to end rather than assumed:
`reasoning/consciousness_probe.py::probe_access` (RETIRED at T081, 2026-09-16) puts `brain._access_ema` into its detail string at
three decimals, and `reasoning/spike_chat_lake.py::_consciousness` renders that detail verbatim into
the reply to "are you conscious". Run on the live brain, the reply's own number goes "live access EMA
0.489" -> "0.287". The measurement that moved is `_access_ema` and NOTHING else on that chain: the
same trace under the retired rank key is byte-identical, so the arbiter repair changes nothing the
brain says. The one thing that reaches a human is therefore reporting the content-key change, and it
is reporting a measurement rather than a verdict, which is the only reason that is acceptable.

WHICH BUS THIS RUNS ON (P6), and the MEASURED NULL that goes with it
====================================================================
`__init__` takes `ws=`. Default None -> this faculty builds its own workspace, which is what it
has always done, what the live path still does (`engine/brain.py::_update_access` constructs
`AccessConsciousness(seed=...)` with no `ws`), and what `tools/p6_gate.py` counts as one of the
PRIVATE, mutually invisible instances P6 names. Passing the shared handle (`Brain.workspace` /
`LiveBrain.workspace`, the one instance owned by `engine/autonomous_loop.py`) puts this
faculty's coalition into the SAME limited-capacity competition perception and the amygdala's
threat channel post into.

**The route is real and the charge is NULL. Both halves were measured on the live brain and both
are recorded here, because a route that carries nothing is the force-wiring P6 forbids.**

  * IDENTITY, asserted not assumed: `ac.workspace is brain.brain.workspace is
    brain.brain.arm.workspace` -> True. The taps are registered on the injected object, so a
    submission by one holder reaches another holder's taps (self-check claim 6).
  * CHARGE: **0 of 150, and RE-DERIVED 2026-08-06 on the multi-slot arbiter rather than inherited
    from the single-slot one.** `make_brain(seed=0, refit_every=10, persist=False)`, `clock.phase`
    forced to 0.25, 150/150 wake ticks, repointed onto `brain.brain.arm.workspace`, with
    `AutonomousArm._percept_salience` LIVE (150 distinct values in [0.568351, 0.667237]) and then
    NEUTERED to the incumbent literal 0.6 (1 distinct). This faculty's OWN output is byte-identical
    across the pair on every channel: `ignited` 0/150, `global_availability` 0/150, `access_index`
    0/150, the brain's `_access_ema` 0/150 -- and the workspace SET (`broadcast_all()`) is
    identical on 150/150 too.
  * **THE NUMBER SURVIVED THE ARBITER CHANGE AND ITS CAUSE INVERTED, which is the part worth
    keeping.** Under the retired single-slot bus the null was 0/150 because this faculty was
    EVICTED -- in the workspace 0 of 150 ticks. Under this one it is in the workspace **150 of
    150** and `ignited` on **150 of 150**. Same null, opposite reason: it used to be "the producer
    cannot change an output that never happens", and it is now "the producer cannot change an
    output that happens anyway". A null re-read without re-deriving it would have carried the old
    explanation onto a bus where it is false. **STILL VALID AFTER `_ignited` BECAME MEMBERSHIP**,
    stated rather than assumed: in that repointed replay this faculty held the HEAD on 150 of 150
    ticks -- which is why `thought` read 'GLOBAL' there -- so headship and membership AGREED across
    the whole run and every number in this bullet is untouched by the 2026-08-06 readout fix.
  * THE HARNESS IS NOT DEAD -- the positive control that makes the null trustworthy. **Measured on
    the RETIRED arbiter and NOT re-derived here; it is labelled rather than dropped because it is
    still the evidence that the probe can move at all.** Pinning the same producer ABOVE its live
    range moved this faculty hard: at 0.72 its own output differed on 61/120 ticks, at 0.75 on
    71/120, at 0.95 on 120/120. The in-module positive control that IS current is self-check claim
    (8), which is run on every gate.
  * WHY, and it is still not fixable from this file, but the reason changed completely.
    **RETIRED, and recorded because this module quoted it as a permanent property:** the bus used
    to grant its one incumbent `self_boost + hysteresis` = 0.25 of defence while the whole band a
    compliant specialist may claim is 0.20 wide, so the first tenant held the single slot forever.
    `faculties/global_workspace.py` FIXED exactly that on 2026-08-06 (multi-slot, capacity
    `floor(1 / defence)` = 4, lateral inhibition divided across the capacity), and a compliant
    challenger now wins a slot against any compliant tenant. **The live null therefore no longer
    has a bus defect behind it.** What is behind it now is the producer population: with room for
    four contents and one admissible producer beside this faculty, both are in every tick, so a
    producer that changes its volume inside the band changes nothing about a consumer that had
    access either way. That is `faculties/global_workspace.py`'s own measured finding on this bus
    ("THE CAPACITY IS NOT THE BINDING CONSTRAINT ON THAT BUS TODAY; THE PRODUCER POPULATION IS"),
    reached independently from the consumer side here.

Consequence of repointing, RE-MEASURED 2026-08-06 AFTER the rank repair and the content-key
rotation, because both could have removed it and neither did: this faculty posts a novelty-gated
0.5207..0.7500 (43 distinct) through `engine/brain.py::_update_access` and takes the head on 40 of
56 wake ticks, so `engine/live_brain.py::thought` carries this faculty's content key on 40 of 56
and 28 distinct values where the un-repointed brain names the submitted percept on 56 of 56 with 4.
The percept co-exists beside it on **1 of 56** ticks. Multi-slot did NOT make the repoint free -- it
made it survivable in principle (the arbiter measures the percept staying in on 40/40 against a
faculty pinned at a compliant 0.65) while this faculty's own live claim is high enough in the band
to push it out anyway. THE FULL BAND SWEEP IS AT `engine/brain.py::_update_access` and it is a hard
step rather than a gradient: excluded on 0 of 56 up to a 0.6800 ceiling, HEAD on every admitted tick
from 0.6825. `_max_claim` is what keeps the consequence bounded: a joining faculty may not make
itself indisplaceable, and with it a 0.95 proximity threat still seizes the top slot where uncapped
it cannot (self-check claim 9).

Determinism: per-instance random.Random(seed) for the workspace; neuron spike codes are
crc32-seeded (no module-level / time seeding). ASCII output.

  Run: python -m snn_data_lake.faculties.access_consciousness      # (also run by run_selfchecks.py)
"""

import random
import zlib
from collections import deque

import numpy as np

from snn_data_lake.faculties.global_workspace import GlobalWorkspace, OneSubsidyArbiter
from snn_data_lake.faculties.vsa_reasoning import D, bundle, cleanup, hv, sim

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

    _LABEL = "GLOBAL"  # NAMESPACE tag on this faculty's contents -- see `content_key`
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
        same 0 of 150 BEFORE the repoint and AFTER it, and the same again after the workspace went
        multi-slot (re-derived 2026-08-06; see the module docstring for the inverted cause).
        Sharing the object was necessary and was not sufficient. Claim (8) shows what the route
        DOES carry, and what it carries is no longer what this sentence used to say: it is not that
        a LOUDER competitor takes the slot -- measured, sweeping one competitor across the whole
        compliant band changes this faculty's admission on 0 of 12 ticks -- it is that a POPULATION
        of competitors fills the workspace and excludes it. The mechanism works and the live bus has
        one other admissible producer where it needs two.

        `ignition_threshold` is read back OFF the bus once one is injected rather than kept as the
        argument's value: it is the rheobase of the pool this faculty's submissions actually drive,
        and `engine/brain.py::_update_access` anchors its salience on `ac.ignition_threshold`
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
        self._key = None  # the workspace key of the CURRENT bundle (see `content_key`)
        # last-step functional readout.
        self._ignited = False  # ACCESS: this content held a workspace slot
        self._focus = False  # FOCUS: this content was the RANKED HEAD of the workspace
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
        live check in `faculties/vsa_reasoning.py` (`coincidence_many` vs `coincidence`), and was
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
        derivation `engine/autonomous_loop.py::_percept_salience` already applies to perception's
        claim, from the same two `GlobalWorkspace` attributes (P2 -- one definition of "the loudest
        a specialist may legitimately be", applied by every specialist, not a second policy). Above
        it, the incumbent an ignition creates carries more recurrent defence than the widest claim
        any competitor can make, so it can never be displaced by anything.

        MEASURED, which is why this exists at all. Repointed onto the shared bus without it, this
        faculty posts the salience `engine/brain.py::_update_access` computes -- 1.0 on the early
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
        # (engine/brain.py::_update_access) always passes an explicit novelty-gated salience, so
        # this computed the expensive default on every tick and then discarded it -- a measured
        # 13.1 ms of a 56.6 ms tick. Computing it only when it is the value actually used is
        # bit-exact: `_coherence` is a pure function of `codes` with no other effect.
        if salience is None:
            sal = self._salience(self._coherence(codes), self._completeness(len(codes)))
        else:
            sal = max(0.0, min(1.0, float(salience)))
        sal = min(sal, self._max_claim())  # stay DISPLACEABLE on the bus -- see _max_claim
        # POST THE BUNDLE'S OWN KEY, not this faculty's name (see `content_key`): a source that can
        # never change its content key can never be superseded and is subsidised forever.
        self._key = self.content_key()
        self._ws.submit(self._SRC, self._key, sal)  # post the candidate for ignition
        self._last_claim = sal
        return self._key

    def content_key(self):
        """The workspace key of the CURRENT bundle -- `(_LABEL, crc32 of the content code)`, or None
        before anything has been bound.

        IT IDENTIFIES THE CONTENT, NOT THE FACULTY, AND THAT IS THE WHOLE POINT (2026-08-06). This
        faculty used to post the frozen string `_LABEL` on every tick. `faculties/global_workspace.py`
        keys its pools on `(source, content)` and `_defended` withholds the tenancy subsidy from a
        held pool exactly when its SOURCE is posting a DIFFERENT content -- so a source whose content
        key can never change is PERMANENTLY SELF-DEFENDED. It re-asserts the identical
        `('coalition', 'GLOBAL')` every tick, the withdrawal branch is unreachable, and the pool
        draws `self_boost + hysteresis` for the life of the process no matter how much the bundle
        underneath it has changed.

        THAT IS THE SAME DEFECT `_defended` WAS BUILT TO CLOSE, one file over. Perception's key
        rotates over the percept ids, so its pool is never defended; this faculty's never rotated, so
        its pool always was. MEASURED at the arm step with both on one bus: the coalition is SILENT
        (it does not post every tick) and therefore DEFENDED, drive 0.8875, against perception's
        0.590 -- so perception is EXCLUDED BY THE SUBSIDY, not out-ranked by a louder claim. A
        faculty that cannot retract cannot be displaced.

        crc32 over the +-1 spike code, one byte per line, per the house determinism rule -- the
        builtin `hash()` is salted per process and a namespace that changes between processes is not
        a namespace. The key is a TUPLE so `_LABEL` stays a real namespace: `broadcast_all()` returns
        contents, and a bare string could collide by SUBSTRING with another faculty's content in a
        membership scan.

        RE-BINDING THE SAME ASSEMBLY GIVES THE SAME KEY, which is not a weakness of the rotation but
        the correct branch of `_defended`: a specialist still asserting the identical content is
        re-asserting its tenancy and keeps its grip. What ends is the grip on content it has moved
        on from."""
        if self._content_hv is None:
            return None
        return (self._LABEL, zlib.crc32(bytes(1 if x == 1 else 0 for x in self._content_hv)))

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
        defect P6 counts. `engine/live_brain.py` and `engine/brain.py` expose the same handle under
        the same name for the same reason."""
        return self._ws

    # -- one ignition cycle -----------------------------------------------------------------------
    def step(self):
        """Advance the workspace one tick: ignition competition + all-or-none global broadcast.
        Returns whether THIS faculty's content reached the workspace (True/False) and refreshes the
        readout.

        ACCESS IS MEMBERSHIP, FOCUS IS HEADSHIP, AND THIS METHOD MEASURES BOTH SEPARATELY
        (2026-08-06). `_ignited` is `content_key() in _ws.broadcast_all()` -- "is my content in the
        limited-capacity workspace at all", which is exactly Dehaene's global availability and
        exactly what this module's name, docstring and `access_index` have always said they meant.
        `_focus` is `_ws.step() == content_key()` -- "am I what the brain is currently thinking
        about". Both ask about the key POSTED THIS TICK; `content_key` carries why that matters and
        what a frozen key cost.

        THE DEBT THIS PAYS, and it is recorded rather than deleted because the defect is the whole
        argument for the change. Between the workspace going multi-slot and this fix, `_ignited`
        was `content == _LABEL` against the head, so it silently stopped measuring access. MEASURED
        on the arbiter as it then stood, one peer faculty at identical salience: this faculty was in
        `broadcast_all()` on 12 of 12 ticks while `ignited()` read False on 12 of 12, and renaming
        that peer 'aaa' -> 'mmm' -- changing nothing else -- moved `ignited()` to True on 11 of 12,
        because rank among co-admitted contents is broken lexicographically on the source key. A
        consciousness-relevant flag was being settled by a string comparison, and `access_index`
        moved 0.000000 -> 0.839103 with it. Under ONE slot the two statements were the same
        statement (claim (7) still asserts that equivalence, and it is the RED that keeps the
        dissociation honest); four slots dissociated them.

        THE HALF THAT COULD NOT BE FIXED FROM THIS FILE, AND WAS. Switching this flag alone would
        have been theatre: the bus delivered only the head to subscribers, so on precisely the
        ticks where this faculty is admitted-but-not-head every tap holds somebody else's content,
        `global_availability` reads 0.0, and `access_index = gate * availability * integration`
        stays 0.0 -- a readout that says "ignited" while reporting no reach. So the bus now
        delivers its ADMITTED SET (`faculties/global_workspace.py::_deliver`, one channel, not a
        second one), and the tap scan below asks the question that channel can now answer: did the
        workspace this tap was handed CONTAIN my content. Claim (10) runs the head-only delivery
        back as a shim and shows the index collapsing to 0.0 -- the trap, measured rather than
        described.

        AVAILABILITY IS DERIVED, NOT REDEFINED (P1). It is the same ratio it always was --
        `|taps whose delivered workspace contained this content| / n_taps` -- over the same taps,
        with no new constant and no new scale. Only the meaning of "received" moved, and it moved
        because the delivery did. Corners, all asserted in `main()`: ZERO taps cannot occur
        (`n_taps` floors at 2 in `__init__`, so there is no empty denominator); an EMPTY workspace
        and SILENCE both deliver `()`, membership is False and availability is 0.0; a
        SINGLE-ELEMENT workspace delivers `(mine,)` and reads 1.0, which is bit-for-bit the answer
        the head-only contract gave -- which is why the live brain's numbers do not move at all
        (60/60 ticks byte-identical, its bus having one admissible producer).

        The tap test is a membership test on the delivered TUPLE, and that the delivered value IS
        the workspace tuple is pinned by claim (3) rather than assumed: a bare content delivered in
        its place would make the scan a substring or an identity test rather than a membership one,
        and it would report reach that never happened. Since `content_key` made the key a TUPLE that
        failure is now loud rather than silent -- `key in "some string"` raises -- which is why the
        namespace tag is kept and the key is not a bare int."""
        # WHO CLOCKS THE ARBITER. A faculty that BUILT this bus is its only producer, so it advances
        # it. A faculty handed SOMEBODY ELSE'S bus submits into it and lets the owner run the one
        # competition per cycle -- `_own_ws` has recorded that distinction since the injection
        # parameter landed and, until 2026-08-07, nothing acted on it (its value was asserted and
        # its consequence was nowhere: P3 shape 5, a flag that changes no outcome).
        #
        # THE DEFECT THAT MADE IT LOAD-BEARING. Repointed onto the arm's bus, this faculty called
        # `_ws.step()` here while `engine/autonomous_loop.py::_workspace_broadcast` called it too,
        # so ONE brain tick ran TWO competitions. Instrumented over 40 live ticks, competition 1 saw
        # `('perception', 'salience')` and competition 2 saw `('coalition',)` ALONE, on 40 of 40.
        # The two producers never met: each won an uncontested competition and evicted the other's
        # content by default, which is the "the producers alternate" reading that three sessions
        # attributed to the band, the capacity and the lateral inhibition in turn. A bus with two
        # producers and two clocks is two buses taking turns.
        #
        # THE ONE-CYCLE LAG IS FORCED, NOT A SHORTCUT, and it is the honest part of this. The
        # coalition's claim is derived from THIS tick's percept, which the arm produces by stepping;
        # so the claim cannot exist before the competition it would join. Submitting it into the
        # NEXT cycle is the only ordering the data dependency allows, and a delayed re-entrant
        # projection is what the biology does anyway. The readout below therefore reports on the
        # competition this faculty's PREVIOUS submission entered.
        head = self._ws.step() if self._own_ws else self._ws.current
        admitted = self._ws.broadcast_all()
        # THE KEY POSTED THIS TICK, never the class label: `content_key` rotates with the bundle, so
        # "is my content in the workspace" has to be asked about the content this faculty is
        # currently asserting. Asking about a frozen label would report a STALE pool -- a bundle this
        # faculty has already moved on from -- as its own current global availability.
        mine = self._key
        self._ignited = mine is not None and mine in admitted  # ACCESS -- in the workspace
        self._focus = mine is not None and head == mine  # FOCUS -- on top of it
        received = sum(1 for tap in self._taps if tap and mine is not None and mine in tap[-1])
        # The `if self._ignited` gate is MEASURED INERT under the current delivery contract, and is
        # KEPT rather than deleted. Why inert: `:507` steps the bus, which broadcasts the admitted
        # SET to every tap, and `:508` reads that same set back -- so `tap[-1]` and `admitted` come
        # from one cycle, `received > 0` implies `_LABEL in tap[-1]` implies `_LABEL in admitted`
        # implies `_ignited`, and the false branch cannot be reached. A mutant deleting the gate is
        # GREEN on every criterion in this module and in faculties/global_workspace.py -- found by
        # adversarial review, recorded here rather than left for the next reader to mistake for a
        # live guard. Why kept anyway: it is inert only BECAUSE delivery carries the set. Under the
        # retired head-only contract `tap[-1]` was a bare head, and this gate is what stopped a
        # faculty that had lost the focus from reporting the reach of whoever took it. It is one
        # conditional standing between a changed delivery contract and a consciousness-relevant
        # readout claiming availability it does not have, so P8's delete-a-criterion-with-no-red
        # rule is not the applicable one: this is a control proven insensitive, which is a
        # disclosure and not a licence to drop it.
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
        """ACCESS. True iff the content crossed the all-or-none ignition threshold AND held one of
        the limited-capacity workspace's slots, so it was broadcast to every subscriber.

        This is membership, not rank. A content can be globally available while another content is
        the focus -- that is what a capacity of more than one MEANS -- and `focused()` is the other
        half. Under a one-slot workspace the two collapse into one statement, which is the RED
        claim (10) uses to show the dissociation is real."""
        return self._ignited

    def focused(self):
        """FOCUS. True iff the content is the RANKED HEAD of the workspace -- what the brain is
        currently thinking about, the same quantity `engine/live_brain.py::thought` reads.

        NOT a second competition and NOT a second quantity (P1/P2): it is one boolean over the
        arbiter's existing head, read from the same `step()` return the workspace already gives
        every caller. It is reported SEPARATELY from `ignited()` rather than folded into it,
        because folding them is exactly the defect this file carried: rank among co-admitted
        contents is settled by the arbiter's lexicographic tie-break on the source name, so a
        readout that reports headship where it means access is settled by a string comparison.
        `access_index` is gated on ACCESS, never on this."""
        return self._focus

    def global_availability(self):
        """Fraction in [0,1] of subscribed downstream taps whose delivered workspace CONTAINED this
        content this tick -- global reach, all-or-none across taps.

        Same ratio, same taps, no new constant: what moved is that the bus delivers its admitted
        set, so "received" can now mean "my content was in the broadcast" on a tick where another
        content is the head. See `step()` for the derivation and the ZERO/EMPTY/SILENCE/SINGLE
        corners."""
        return self._availability

    def n_bound(self):
        """Number of grown assemblies fused into the currently ignited conscious content (0 if none)."""
        return self._n_bound

    def integration(self):
        """Cheap Phi-like PROXY in [0,1] for how bound/coherent the ignited content is
        (mean pairwise spike coincidence x completeness). This is a PROXY, explicitly NOT IIT Phi."""
        return self._integration

    def access_index(self):
        """Honest ACCESS-consciousness index in [0,1] = ignition(all-or-none ACCESS gate) *
        global_availability * integration. RISES as more coherent neurons are grown and bundled.
        Measures global availability only -- it is NOT a measure of phenomenal experience.

        The gate is `ignited()` -- workspace MEMBERSHIP -- and deliberately not `focused()`. An
        index called `access_index`, whose own docstring says global availability, gated on being
        the top-ranked content would be measuring focus and reporting access. That is the defect
        this file carried for the life of the multi-slot arbiter and it is the reason claim (10)
        exists."""
        gate = 1.0 if self._ignited else 0.0
        return max(0.0, min(1.0, gate * self._availability * self._integration))

    def state(self):
        """Functional snapshot. NO 'conscious'/'sentient'/'phenomenal'/'subjective'/'aware'/
        'feels'/'qualia' field -- access consciousness only (see module docstring + claim 5).

        `ignited` is ACCESS (in the workspace) and `focused` is FOCUS (on top of it). They are
        reported as two keys because they are two facts; a consumer that wants the old single-slot
        reading of "ignited" wants `focused`, and every consumer that reads this dict for global
        availability wants `ignited`."""
        return {
            "ignited": self._ignited,
            "focused": self._focus,
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


# distinct-from: sleep_replay.py -- `_batch_states` there is a fused array pass over recall
#   trials; this is the SAME harness function it always was, its cadence converted to the live
#   brain's one-competition clock. No new mechanism is introduced by this edit.
def _contended_run(n_peers, ws, n=12, shared=True):
    """Replay the LIVE cadence with `n_peers` OTHER faculties competing on ONE bus: every
    submitter -- the peers and this faculty -- posts into the SAME competition, and the owner
    clocks it ONCE per tick (`engine/brain.py::_update_access` since the one-clock repoint).
    Returns (own, admitted) -- this faculty's OWN per-tick output tuple, and whether its content
    held a workspace slot on that tick.

    THE SWEPT VARIABLE IS THE LOAD, NOT ONE COMPETITOR'S VOLUME, and choosing that is the whole
    re-derivation of claim (8) -- the measurements that forced it are recorded at the claim. A peer
    that is not posting is a NEUTERED competitor, so sweeping `n_peers` IS the neuter-A/B this
    claim has always been the in-module form of: same bus, same cadence, same claim, one faculty's
    worth of competition removed.

    Every peer posts EXACTLY what this faculty is capped at (`_max_claim()`, read off the bus), so
    the load is a population of COMPLIANT specialists and nothing wins by breaking the bus's own
    displaceability bound. It also introduces no number of its own: the claim is derived from the
    bus and MOVES with it, so a workspace reconfigured to hold one content re-derives the peers'
    claim too.

    `shared=False` builds this faculty its own workspace while the peers still post and the bus is
    still stepped -- the identical replay with two objects instead of one, which is the RED for the
    whole claim.
    """
    ac = AccessConsciousness(seed=0, ws=ws if shared else None)
    claim = ac._max_claim()
    own, admitted = [], []
    for _ in range(n):
        # ONE competition per tick -- the peers and this faculty settle TOGETHER, and the owner
        # supplies the single clock. CONVERTED 2026-08-07, deliberately in its OWN pass: the
        # mechanism changes this cadence judges (the subsidy split, the one-clock repoint, the
        # inhibitory rheobase) all landed and were gated first, so re-deriving the harness's
        # criteria here is not the same author writing code and criterion in one pass. The
        # two-competition cadence this replaced had the peers settle and THEN this faculty settle
        # against their defended remains -- a cadence the live brain stopped running at the
        # one-clock repoint, and one whose defence-broken ties made `k_out` look like a clean
        # equality with the capacity (see the doc block at the claim for what the live cadence
        # actually does at exact ties).
        _post(ws, [(f"peer{b}", f"peer{b}", claim) for b in range(n_peers)])
        ac.bind_assembly(_coherent("cat", 3), salience=claim)
        ws.step()
        own.append((ac.step(), round(ac.access_index(), 6), round(ac.global_availability(), 6)))
        # ADMISSION, not head identity: `broadcast_all()` is the multi-slot read, and asking it
        # about THIS faculty's own content is asking whether this faculty got access. Read off
        # `ac.workspace` rather than off `ws` so the private-bus RED arm asks the same question of
        # the bus this faculty is actually on, instead of trivially reporting False.
        admitted.append(ac.content_key() in ac.workspace.broadcast_all())
    return own, admitted


def _access_by_load(bus_kw, limit, shared=True):
    """`_contended_run` at every peer load 0..limit inclusive, each on a FRESH bus."""
    return [_contended_run(k, _bus(**bus_kw), shared=shared) for k in range(limit + 1)]


def _first_denied(runs):
    """The smallest peer load that denies this faculty the workspace on EVERY tick -- SEARCHED
    across the scan, never assumed and never a constant of this file. None when no load inside the
    scan denies it, which is the reading a PRIVATE bus gives, so the search returns it rather than
    raising: 'there is no such load' is a result claim (8) has to be able to state."""
    for k, (_own, admitted) in enumerate(runs):
        if not any(admitted):
            return k
    return None


def _slot_after_threat(claim, threat=0.95, warm=8, psal=0.6, bus=None):
    """Warm a shared bus with this faculty holding the slot at `claim`, then fire a proximity
    threat. Returns what the slot holds afterwards -- 'threat' iff the bus is still displaceable.

    `bus` selects the ARBITER, defaulting to the shipped one. Claim (9)'s red passes
    `faculties/global_workspace.OneSubsidyArbiter` -- see the criterion for why the red moved off
    the uncapped arm and onto the arbiter."""
    ws = _bus() if bus is None else bus(seed=0)
    ac = AccessConsciousness(seed=0, ws=ws)
    ac.bind_assembly(_coherent("cat", 3))  # establish this faculty's real content key ONCE...
    key = ac.content_key()
    for _ in range(warm):
        # ...then post it RAW, bypassing bind_assembly's cap, so capped and uncapped arms are
        # comparable. The key is CONSTANT across the warm loop on purpose: a faculty re-asserting
        # one content is the branch `_defended` subsidises, which is the HARDEST case for
        # displaceability and therefore the right one for this criterion. ONE competition per
        # tick (converted 2026-08-07 with the rest of the harness cadence): perception, the
        # silent threat channel and this faculty all settle together under the owner's one clock.
        _post(
            ws,
            [("perception", "unit0", psal), ("salience", "threat", 0.0), (ac._SRC, key, claim)],
        )
        ws.step()
        ac.step()
    _post(ws, [("perception", "unit0", psal), ("salience", "threat", threat)])
    return ws.step()


def _peer_names():
    """The two peer source names claim (10) renames between, DERIVED from this faculty's own
    `_SRC` rather than written down.

    The arbiter's rank key ends in the full `(source, content)` pair, so among co-admitted contents
    whose sustained-firing counts TIE the head is settled lexicographically on the source string.
    Claim (10) uses that as its lever, so it needs one name that sorts before `_SRC` and one that
    sorts after -- and if they were literals they could silently stop straddling the day `_SRC`
    changed, which would turn the whole claim green-and-meaningless. Stepping one character either
    side of `_SRC`'s first letter moves with it; `main()` asserts the straddle before using it."""
    c = AccessConsciousness._SRC[0]
    return chr(ord(c) - 1) * 3, chr(ord(c) + 1) * 3


def _own_stale_head(lo=0.55, hi=0.75, step=0.0025, warm=6, tenant=0.75):
    """The state claim (10)'s head conjunct forbids, SWEPT rather than hardcoded (P1).

    This faculty warms ONE content, then asserts a DIFFERENT one at a lower claim while the first
    is still charged. Both wear this faculty's own source, so `_defended` withholds the subsidy
    from the superseded pool -- but it is still decaying from a high charge, so there is a band
    where the fresh content wins a SLOT and the faculty's OWN retracted content holds the HEAD.
    That reads `ignited and not focused` on every accessor this module has, with no peer anywhere
    in the workspace. It is not a dissociation between access and focus; it is one content of this
    faculty's out-ranking another.

    Returns `(claim, head, mine)` for the lowest claim in the band that reaches it, or `(None, ...)`
    -- so the criterion's red is a measured window and not a constant somebody chose."""
    s = lo
    while s <= hi + 1e-9:
        ws = _bus()
        ac = AccessConsciousness(seed=0, ws=ws)
        ac.bind_assembly(_coherent("cat", 3))
        old = ac.content_key()
        fresh = (AccessConsciousness._LABEL, 0)  # a content this faculty has never bundled
        for _ in range(warm):
            _post(ws, [(ac._SRC, old, tenant)])
            ws.step()
        _post(ws, [(ac._SRC, fresh, round(s, 4))])
        ac._key = fresh  # what this faculty is asserting NOW
        ws.step()  # the OWNER settles it -- `ac.step()` no longer clocks a bus it does not own
        ac.step()
        if ac.ignited() and not ac.focused() and ws.broadcast() == old:
            return round(s, 4), old, fresh
        s += step
    return None, None, None


# distinct-from: experience_replay.py -- `replay` there consolidates sampled transitions into a
#   value table; this is the SAME dissociation harness it always was, its cadence converted to
#   the live one-competition clock. No new mechanism is introduced by this edit.
def _focus_run(peer_claim, peer_src=None, ws=None, n=12, who=None):
    """Replay the live ONE-competition cadence with ONE peer faculty posting `peer_claim` every
    tick, and record ACCESS, FOCUS, the index and the reach SEPARATELY on every tick.

    Deliberately not a variant of `_contended_run`, because the two claims pull the two different
    levers this bus has. Claim (8) sweeps the peer POPULATION at a compliant claim, and measured
    that the population is what decides ADMISSION (one competitor's volume moves it on 0 of 12
    ticks). This one holds the population at one and changes what decides the HEAD. Using the
    population lever here would measure nothing: at the loads where admission moves, the faculty is
    out, and a faculty that is out has neither access nor focus.

    `peer_claim = 0.0` is the peer NEUTERED -- sub-rheobase, never in the workspace -- so the arms
    of the dissociation are ONE harness with one variable each, not several harnesses.

    THE INDEX AND THE REACH ARE RECORDED AT FULL PRECISION, NOT ROUNDED, and that is a criterion
    strength decision rather than a formatting one. Claim (10)'s load-bearing assert is that the
    ACCESS readout is IDENTICAL across the peer rename; recording `round(x, 6)` would have made
    that assert mean "identical to six decimal places", leaving a blind window in which a
    headship-dependent term could enter the index and every gate stay green. MEASURED rather than
    assumed in both directions: the three arms are BIT-identical (equal `float.hex()` on 12 of 12
    ticks, not merely equal to 6dp), so the tightening costs nothing and is not a threshold; and a
    mutant that subtracts 1e-9 from the index when this faculty is NOT the head passed the rounded
    form and is RED against this one (at 1e-4 both are red -- the rounded form's blind window is
    the whole interval below ~5e-7). Rounding belongs at the PRINT, and is applied there."""
    ws = _bus() if ws is None else ws
    ac = AccessConsciousness(seed=0, ws=ws)
    claim = ac._max_claim()
    src = _peer_names()[0] if peer_src is None else peer_src
    out = []
    for _ in range(n):
        # ONE competition per tick: the peer and this faculty settle together, the owner clocks
        # once (converted 2026-08-07 with the rest of the harness cadence).
        _post(ws, [(src, "peer", peer_claim)])
        ac.bind_assembly(_coherent("cat", 3), salience=claim)
        ws.step()
        ac.step()
        if who is not None:
            # WHAT HOLDS THE HEAD, and this faculty's own key, recorded per tick. Claim (10) asserts
            # only "admitted and NOT the head", and that sentence is also true of a faculty
            # out-ranked by its OWN previous bundle with the peer absent entirely -- which is a
            # different fact and not a dissociation. Nothing in the loop above forbids it: the
            # branch is closed only because this arm re-posts ONE content, and an unasserted
            # invariant is one edit from being untrue. Handed out rather than folded into `out` so
            # the existing 4-tuple unpacks stay put.
            who.append((ws.broadcast(), ac.content_key()))
        out.append(
            (
                ac.ignited(),
                ac.focused(),
                ac.access_index(),  # FULL precision -- see the docstring; rounding here would
                ac.global_availability(),  # narrow claim (10)'s invariance to 6dp
            )
        )
    return out


def _name_ranked(ws):
    """THE RETIRED RANK KEY, re-expressed on the CURRENT bus: `(-late, -total, k)` instead of
    `(-late, -total, -_last_v, k)`.

    `faculties/global_workspace.py::_rank_key` is the single place that arbiter decides who out-ranks
    whom, so restoring the retired ordering is a one-method override and NOT a second arbiter --
    same pools, same drives, same settle, same capacity, same delivery. That is what makes it the
    RED for claim (10)'s string test rather than a different experiment: the ONLY variable is
    whether the clipped spike counts fall through to the SOURCE NAME.

    Why it is the red this claim needs. Two compliant pools SATURATE their sustained-firing counts,
    so under the retired key the head was settled lexicographically -- and this file's FOCUS readout
    read that head. Renaming a peer therefore moved a consciousness-relevant flag while changing no
    claim, no content and no competition. Returns the bus.

    BOTH ORDERINGS ARE RESTORED, and that became load-bearing on 2026-08-07. The arbiter now
    decides ADMISSION with `_rank_key` and the HEAD with `_focus_key` (the hold/grip split -- see
    `faculties/global_workspace.py::grip`), and this file's FOCUS readout reads the head. Overriding
    `_rank_key` alone therefore left the head on the REPAIRED key and the red went green: measured,
    the rename moved FOCUS on 0 of 12 in both arms, which is a control that has quietly stopped
    being one. A red has to reconstruct the retired behaviour on the path the criterion reads."""

    def rank(k):
        return (-ws._last_late[k], -ws._last_total[k], k)

    ws._rank_key = rank
    ws._focus_key = rank
    return ws


def _head_only(ws):
    """THE RETIRED HEAD-ONLY DELIVERY, re-expressed on the CURRENT bus: subscribers are handed
    `contents[:1]` where they are now handed `contents`.

    Same arbiter, same route, same type, same competition -- the ONLY variable is how much of the
    workspace the broadcast carries. That is what makes it the RED for the delivery contract rather
    than a second implementation of it, and it is why the shim slices the tuple instead of
    reconstructing the old `_deliver(self.current)` signature: reverting the SHAPE too would have
    made a passing arm ambiguous between 'the head is all that arrived' and 'the tap could not read
    what arrived'. Returns the bus."""
    real = ws._deliver

    def head_only(contents):
        real(contents[:1])

    ws._deliver = head_only
    return ws


def main():
    # ---- (1) IGNITION IS ALL-OR-NONE: coherent+salient ignites (reaches >=2 taps); weak/incoherent
    #          does NOT. The control (a scrambled assembly) proves ignition is a real threshold. ----
    strong = _fresh()
    strong.bind_assembly(_coherent("cat", 5))  # 5 coincident neurons -> supra-threshold
    strong.step()
    # A tap now holds the DELIVERED WORKSPACE (a ranked tuple), so reach is a membership test --
    # `is not None` would be true on every tick including the empty ones, because `()` is not None.
    strong_reach = sum(1 for tap in strong._taps if tap and strong.content_key() in tap[-1])
    # ...and what a tap receives must BE the workspace tuple, checked rather than assumed: a bare
    # string delivered instead would satisfy the membership scan above by SUBSTRING ('GLOBAL' in
    # 'GLOBAL'), which is a silently wrong reach reading rather than a loud one.
    tap_payload = strong._taps[0][-1]
    tap_is_workspace = (
        type(tap_payload) is tuple and tap_payload == strong.workspace.broadcast_all()
    )

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
    inj_a.step()  # SUBMITS; on a bus it does not own it no longer clocks it (see `step`)
    inj_ws.step()  # ...so the OWNER runs the one competition. This line is the change made visible:
    # it used to be implicit inside `inj_a.step()`, which is exactly the double-clocking that made a
    # shared bus behave as two buses taking turns on the live brain.
    cross_tap = bool(inj_b._taps[0]) and inj_a.content_key() in inj_b._taps[0][-1]
    # the rheobase is read OFF the bus, so the anchor engine/brain.py uses is the pool's own. RED by
    # construction: the constructor argument says 0.55 and the bus says 0.42.
    odd_ws = _bus(ignition_threshold=0.42)
    odd_ac = AccessConsciousness(seed=0, ignition_threshold=0.55, ws=odd_ws)
    threshold_readthrough = odd_ac.ignition_threshold == 0.42

    # ---- (6b) A FACULTY DOES NOT CLOCK A BUS IT DOES NOT OWN. -----------------------------------
    # Sharing the OBJECT is claim (6); this is sharing the CLOCK, and it is a separate failure that
    # claim (6) passes straight through. Measured on the live brain before the fix: repointed onto
    # the arm's bus, this faculty called `_ws.step()` while `engine/autonomous_loop.py` called it
    # too, so ONE wake tick ran TWO competitions -- instrumented over 40 ticks, competition 1 saw
    # `('perception','salience')` and competition 2 saw `('coalition',)` ALONE, 40 of 40. Two
    # producers on one object that never meet is not a shared bus; it is two buses taking turns, and
    # it is why co-occupancy read 0/120 with the identity assert above passing.
    # After: ONE step per tick, and 39 of 40 competitions carry all three submitters together.
    #
    # The RED is `_own_ws` flipped on the SAME instance and the same bus -- one attribute, so what
    # it isolates is the ownership test and not some other difference between two harnesses.
    clock_ws = _bus()
    clock_ac = AccessConsciousness(seed=0, ws=clock_ws)
    clock_ac.bind_assembly(_coherent("cat", 3))
    _clock_n = [0]
    _clock_real = clock_ws.step

    def _counted_step():
        _clock_n[0] += 1
        return _clock_real()

    clock_ws.step = _counted_step
    clock_ac.step()
    guest_clocks = _clock_n[0]  # a GUEST must not advance the arbiter
    _clock_n[0] = 0
    clock_ac._own_ws = True  # THE RED: the same faculty, now claiming to own the same bus
    clock_ac.step()
    owner_clocks = _clock_n[0]  # an OWNER must
    clock_ws.step = _clock_real

    # ---- (7) WHAT MAKES ACCESS AND FOCUS THE SAME QUESTION IS ONE CONTENT, NOT ONE SOURCE. --------
    # RE-DERIVED 2026-08-06, and the re-derivation is the finding. This claim asserted that on a
    # PRIVATE bus "the slot is non-empty" and "the slot is MINE" agree on every step, with a shared
    # bus as the disagreement arm. That was true only because this faculty posted a FROZEN content
    # key, so its bus could never hold more than one of its pools; `content_key` now rotates with the
    # bundle (see there for why a frozen key is permanently self-defended), and the old sentence
    # became false on a private bus: measured, alternating two assemblies, the two tests disagree on
    # 1 of 10 steps because THE FACULTY IS OUT-RANKED BY ITS OWN PREVIOUS BUNDLE, which is still
    # charged and is a different content. Re-derived rather than weakened: the property claim (10)
    # actually leans on is that a workspace holding ONE content cannot tell ACCESS from FOCUS, and
    # "one submitter" was a proxy for "one content" that stopped holding.
    #
    # THREE ARMS, ONE VARIABLE EACH -- how many CONTENTS are in the competition:
    #   * ONE content, re-asserted every tick on a private bus  -> the two tests must AGREE 10/10.
    #   * TWO contents, both this faculty's (a rotating bundle) -> they must COME APART. This arm is
    #     new and it is a disclosure: a specialist can lose the focus to what it was saying a moment
    #     ago. That is the attractor's hysteresis, not a defect -- `_defended` withholds the subsidy
    #     from the retracted bundle, so it decays at `decay` and is out-ranked as soon as the fresh
    #     bundle out-drives it.
    #   * TWO contents, one a louder PEER on a shared bus     -> they must COME APART.
    # Without the second and third arms the first is vacuous (the `_fused_calls` lesson, applied to
    # a second claim in this file).
    priv_seen = []
    priv_ac = _fresh()
    _record_slot(priv_ac, priv_seen)
    priv_keys = []
    for _t in range(10):
        priv_ac.bind_assembly(_coherent("cat", 3))  # ONE content, re-asserted: the key is constant
        priv_keys.append(priv_ac.content_key())
        priv_ac.step()
    priv_agree = sum(1 for c, k in zip(priv_seen, priv_keys) if (c is not None) == (c == k))
    assert len(set(priv_keys)) == 1, (
        f"the one-content arm must actually post ONE content -- if the key moves here the arm is "
        f"the two-content case and proves nothing about the collapse: {len(set(priv_keys))} keys"
    )
    # ...the SAME replay with the bundle alternating, so the faculty competes with its own history.
    own_seen = []
    own_ac = _fresh()
    _record_slot(own_ac, own_seen)
    own_keys = []
    for t in range(10):
        own_ac.bind_assembly(_coherent("cat", 3) if t % 2 == 0 else _scrambled(3))
        own_keys.append(own_ac.content_key())
        own_ac.step()
    own_disagree = sum(1 for c, k in zip(own_seen, own_keys) if (c is not None) != (c == k))
    sh_seen = []
    sh_ws = _bus()
    sh_ac = AccessConsciousness(seed=0, ws=sh_ws)
    _record_slot(sh_ac, sh_seen)
    sh_keys = []
    for _ in range(10):
        # ONE competition per tick (cadence converted 2026-08-07): the louder peer and this
        # faculty settle together under the one clock, so `_record_slot`'s spy fires ONCE per
        # tick and `sh_seen[t]` is exactly the competition `sh_keys[t]` entered. THE CONVERSION
        # ALSO CLOSED A LATENT MISALIGNMENT: the `sh_seen[1::2]` slicing that stood here was
        # written for the double-clock era ("the peer's step, then this faculty's") and had been
        # silently comparing every SECOND TICK against the wrong key index since the one-clock
        # repoint made the guest's `ac.step()` stop clocking -- green throughout, because the
        # assert below needs only one disagreement. Alignment is now positional and exact.
        _post(sh_ws, [("perception", "unit0", 0.95)])  # a louder faculty out-claims this one
        sh_ac.bind_assembly(_coherent("cat", 3), salience=0.60)
        sh_keys.append(sh_ac.content_key())
        sh_ws.step()
        sh_ac.step()
    sh_own = sh_seen
    sh_disagree = sum(1 for c, k in zip(sh_own, sh_keys) if (c is not None) != (c == k))

    # ---- (8) THE OTHER FACULTIES ON A SHARED BUS DECIDE THIS ONE'S OWN ACCESS TO IT. -------------
    # RE-DERIVED 2026-08-06, after `faculties/global_workspace.py` replaced single-slot winner-take-
    # all with a limited-capacity multi-slot workspace. The PROPERTY is unchanged and is the P6 one:
    # this faculty is a competitor in a SHARED bottleneck, so what else is in the workspace decides
    # what it gets. What changed is that the old criterion read that property through a single-slot
    # proxy -- "a louder competitor takes THE slot away from me" -- and scored this faculty being
    # THROWN OUT as influence. Under the new arbiter it is not thrown out -- measured here: in the
    # workspace on 12 of 12 ticks against the same 0.95 competitor, where `thalamus/
    # global_workspace.py` records 0 of 12 for the retired one -- so the old count fell to 1/12.
    # The property survived; the channel it was read through did not.
    #
    # TWO CANDIDATE RE-DERIVATIONS WERE BUILT AND SWEPT. Both lost, and both losses are measurements
    # rather than preferences:
    #
    #   * KEEP THE COMPETITOR'S VOLUME AS THE LEVER, ADD `capacity - 1` COMPETING FACULTIES. It does
    #     reach 11/12 -- and it reaches it BACKWARDS. Traced: with 3 background faculties the 0.60
    #     competitor leaves this faculty OUT on 12/12 and the 0.95 competitor leaves it IN on 12/12,
    #     because a 0.95 pool silences the background through the shared inhibition and CLEARS THE
    #     FIELD. A criterion that passes because the louder rival granted this faculty access is no
    #     longer measuring its own message. Volume also fails on its own terms here: at the shipped
    #     capacity, sweeping one competitor 0.55 -> 0.95 changes this faculty's ADMISSION on 0 of 12
    #     ticks; only a claim of 1.00, above every specialist's `_max_claim`, excludes it.
    #   * SWAP HEAD IDENTITY FOR `broadcast_all()` RANK. Right instinct, wrong observable. At the
    #     shipped load rank buys literally nothing -- with two contents in the workspace rank 0 IS
    #     the head, so it re-measures 1/12. Worse, rank among CO-ADMITTED peers is settled by the
    #     arbiter's lexicographic tie-break on the source key: renaming the peer faculty from 'aaa'
    #     to 'mmm', changing nothing else, moves this faculty's `ignited` from 0/12 to 11/12.
    #     MEMBERSHIP is the half of `broadcast_all()` that is stable -- identical across three name
    #     prefixes, four bus seeds and 6/12/24/40-tick runs.
    #
    # SO THE LEVER IS THE LOAD AND THE OBSERVABLE IS ADMISSION. `_first_denied` SEARCHES for the
    # smallest peer population that denies this faculty the workspace; nothing here is a chosen
    # number. The scan runs to twice the capacity ("STAY denied" was the two-competition cadence's
    # property and retired with it -- on the live clock, loads past the first denial can re-admit
    # under the tie draw, disclosed below), and the two arms
    # of the neuter-A/B are the empty bus and that population. Its NULL re-runs THREE arms and the
    # third one is load-bearing -- see `k_co` below, where the arm-by-arm measurement is recorded.
    shipped_bus = _bus()
    k_scan = 2 * shipped_bus.capacity  # scan well past the capacity -- derived, not picked
    load_runs = _access_by_load({}, k_scan)
    k_out = _first_denied(load_runs)
    k_use = k_scan if k_out is None else k_out
    # The RETIRED single-slot workspace, rebuilt as a CONFIGURATION of the same arbiter rather than
    # as a memory of one: the entire unit hold budget spent on a single content, in the shipped
    # defence proportion, so `capacity` derives 1. This is where the old criterion's content is
    # preserved -- on a one-slot bus a SINGLE other faculty must deny this one access, which is
    # exactly the eviction 12/12 measured before.
    #
    # HOW CAPACITY REACHES THIS THRESHOLD -- RE-MEASURED 2026-08-07 when the bus's shared
    # inhibitory pool acquired its rheobase, because the paragraph that stood here had become a
    # transcript of the removed pathology. It recorded denial-by-silencing: at the then-shipped
    # `k_out` of 2 the workspace held TWO contents with TWO SLOTS FREE, the faculty's pool
    # annihilated by lateral inhibition before the capacity bound was consulted, and its swept
    # table (capacity 1/2/3 -> k_out 1, 4 -> 2, 6 -> 3, 8 -> never denied) was that defect
    # measured five ways -- it even warned that "a criterion asserting `k_out == capacity` would
    # be red on a correct bus", and it cited the since-retired `_G_INH / capacity` as the
    # mechanism. Under the rheobase kernel, ON THE TWO-COMPETITION CADENCE this harness then ran,
    # that equality was exactly what the bus did: swept over the legal defences, capacity
    # 1/2/3/4/6 -> k_out 1/2/3/4/6. THE EQUALITY RETIRED WITH THE CADENCE (2026-08-07, harness
    # converted to the live one-competition clock): it was the DEFENCE breaking ties -- this
    # faculty settled its own competition against already-defended incumbents, so rank was
    # deterministic in the load. On the live cadence every claimant posts EXACTLY `_max_claim()`
    # into ONE competition and ties exactly; below `capacity` denial is impossible (all admitted,
    # deterministically), and at or above it the population is truncated to exactly `capacity`
    # every tick while WHICH exact-tie claimant is cut is `_jitter_for`'s declared symmetry break
    # -- the (c1b) tie floor of `faculties/global_workspace.py`, surfacing at the faculty level.
    # Measured: smallest fully-denying load per capacity 1/2/3/4/6 -> 1/2/3/5/7 on this seed --
    # at or one above the capacity, never a clean equality and never below it.
    _defence = shipped_bus.self_boost + shipped_bus.hysteresis
    one_slot = {
        "self_boost": shipped_bus.self_boost / _defence,
        "hysteresis": shipped_bus.hysteresis / _defence,
    }
    k_out_one = _first_denied(_access_by_load(one_slot, k_scan))
    # THE RED, run and not argued: the identical replay with this faculty on its OWN workspace.
    k_out_priv = _first_denied(_access_by_load({}, k_scan, shared=False))
    alone_own = load_runs[0][0]
    crowd_own = load_runs[k_use][0]
    load_diff = sum(1 for a, b in zip(alone_own, crowd_own) if a != b)
    # THE NULL RE-RUNS THREE ARMS, and WHICH arms is the entire content of this control. Measured
    # per arm rather than argued, because the two-arm form (empty + crowded) shipped here first and
    # was WEAKER than the single-arm null it replaced -- the one direction F11 does not allow.
    #
    # The arms are EMPTY (k=0), CO-OCCUPIED (`k_co` -- the largest load at which this faculty is
    # still admitted, derived from the SEARCHED `k_out` and not chosen) and CROWDED (`k_use`).
    # RE-MEASURED 2026-08-07 FOR THE THIRD TIME IN ONE DAY, and the trajectory is worth keeping
    # whole: at the original k_co=1 (old kernel) the claim defect read 4 at co-occupancy; under
    # the rheobase on the two-competition cadence (k_co=3, everyone below capacity) it went
    # invisible to every arm; and under the LIVE one-competition cadence (k_co=4 -- five
    # exact-tie claimants at the capacity boundary) it is maximally visible again, because a
    # +-0.02 claim jitter moves WHO wins the tie-break. Injections: posted-claim jitter +-0.02
    # per tick; coalition-size jitter 2..5 per tick instead of 3. MAX per-tick disagreement
    # across three re-runs against the first; 0 is a clean null:
    #
    #     defect                  EMPTY   CO-OCCUPIED(4)   CROWDED(5)
    #     posted-claim jitter        0         12             0
    #     coalition-size jitter     12         12             0
    #
    # Only the CO-OCCUPIED arm sees BOTH defects -- the live cadence RESTORES the original
    # three-arm rationale the two-competition interlude had erased: it is the one regime in which
    # this faculty genuinely shares the workspace, and the boundary regime where nondeterminism of
    # any kind has an admission outcome to move. The CROWDED arm stays vacuous and stays kept,
    # labelled: an excluded faculty emits the same (False, 0.0, 0.0) on every tick, so it
    # reproduces itself under almost any perturbation -- a disclosure (P8 guard 4), not a reason
    # to drop a control.
    k_co = max(0, k_use - 1)
    load_null = (
        sum(1 for a, b in zip(alone_own, _contended_run(0, _bus())[0]) if a != b)
        + sum(1 for a, b in zip(load_runs[k_co][0], _contended_run(k_co, _bus())[0]) if a != b)
        + sum(1 for a, b in zip(crowd_own, _contended_run(k_use, _bus())[0]) if a != b)
    )
    priv_diff = sum(
        1
        for a, b in zip(
            _contended_run(0, _bus(), shared=False)[0],
            _contended_run(k_use, _bus(), shared=False)[0],
        )
        if a != b
    )
    # THE MONOTONE-STEP CRITERION IS DELETED WITH THE CADENCE THAT PRODUCED IT (2026-08-07, the
    # item the previous pass left as named work -- closed now, in its own pass, after the
    # mechanisms it judges all landed and gated). It asserted admission in on EVERY tick below
    # `k_out` and out on every tick from it to the scan's end, and that step was a property of
    # the retired two-competition cadence: this faculty settled AGAINST defended incumbents, so
    # the defence broke every tie and the load was a clean lever. On the live one-competition
    # cadence every claimant posts EXACTLY `_max_claim()` and ties exactly, and the profile over
    # 0..8 peers reads `# # # # # . # . .` -- k=6 re-admits after k=5 denies, because
    # `_jitter_for` draws in first-encounter order and adding a peer re-draws this faculty's
    # tie-break. Bit-identical under both kernels (measured), which pins the cause on the
    # tie-break draw, not the inhibition. What replaces the assert is a MEASURED DISCLOSURE, not
    # a weaker assert, because no candidate replacement has a demonstrable red (F11 cuts both
    # ways): per-load the outcome is CONSTANT (every load reads 12/12 or 0/12 -- 9/9 loads
    # all-or-none), and the POPULATION admitted is exactly `min(claimants, capacity)` on every
    # tick of every load -- a bound `NoAllowanceArbiter` also satisfies on this input, because
    # exact ties are symmetric and even annihilation cannot separate them; an assert that is
    # green on the retired kernel too would be a criterion with no reachable red, so it is
    # recorded here instead of asserted. Claim (8)'s decisiveness survives on the asserts that
    # kept their reds: `k_out` exists (private-bus red: None), the one-slot bus denies at 1
    # (capacity moves the denial point), and the load moves this faculty's own output 12/12
    # (private red: 0).

    # ---- (9) A FACULTY JOINING A SHARED BUS MUST NOT BE ABLE TO SILENCE IT. ----------------------
    # `_max_claim` caps this faculty's post at the bus's own displaceability bound. The RED is the
    # SAME replay with the cap removed, and it is run, not argued.
    cap = _fresh()._max_claim()
    capped_after_threat = _slot_after_threat(cap)
    uncapped_after_threat = _slot_after_threat(1.0)
    # THE RED MOVED, 2026-08-07, and where it moved to is the finding. It used to be the UNCAPPED
    # arm: at a claim of 1.0 this faculty drove 1.0 + hold = 1.25 against the threat's 0.95 and the
    # threat was blocked, so the cap was what bought displaceability. `global_workspace` then SPLIT
    # the incumbent's subsidy -- the HOLD still sustains an admitted assembly, but the head is
    # ordered by a far smaller GRIP (`hold / capacity`) -- and the arbiter itself now guarantees
    # what the cap was guaranteeing: measured, the threat takes the head on the uncapped arm too
    # (1.0 decays to 0.85, + grip 0.0625 = 0.9125 < 0.95). So the uncapped arm went GREEN, which
    # reads exactly like a criterion losing its red and is the opposite -- the hazard it was built
    # to detect was removed. Weakening or deleting the criterion would have thrown away the only
    # thing in this module that tests the guarantee, so the red is RE-POINTED at the mechanism that
    # now buys it: the same uncapped replay on the pre-split arbiter, which still blocks the threat.
    uncapped_one_subsidy = _slot_after_threat(1.0, bus=OneSubsidyArbiter)
    # the cap only ever LOWERS: a scrambled (sub-rheobase) assembly must stay local on any bus,
    # including one whose defence leaves no room and whose cap therefore floors at the rheobase.
    cap_floor_ws = _bus(self_boost=0.5, hysteresis=0.6)
    cap_floor_ac = AccessConsciousness(seed=0, ws=cap_floor_ws)
    cap_floor_ac.bind_assembly(_scrambled(5))
    cap_floor_ignites = cap_floor_ac.step()
    cap_floor = cap_floor_ac._max_claim()
    cap_open = AccessConsciousness(seed=0, ws=_bus(self_boost=0.0, hysteresis=0.0))._max_claim()

    # ---- (10) ACCESS vs FOCUS -- THE DISSOCIATION, CONSTRUCTED AND MEASURED. ---------------------
    # `docs/CONSCIOUSNESS_METHODS.md` asks for a dissociation and not a marker, and until this
    # workspace had more than one slot the tree could not state one: "my content is in the
    # workspace" and "my content is what the brain is thinking about" were the SAME sentence. That
    # is exactly what claim (7) counts, which is why it is this claim's control rather than a
    # relic. Two slots or more and they are two sentences. Three arms on ONE bus, ONE variable
    # each, and every number below was RUN:
    #
    #   peer NEUTERED (0.00, sub-rheobase) -> IN the workspace and ON TOP of it   access+focus
    #   peer LOUD (0.95, out-driving this faculty's cap) -> IN, NOT on top        ACCESS, no focus
    #   the SAME loud peer RENAMED to sort after -> IDENTICAL on BOTH readouts    name-blind
    #
    # THE LEVER IS THE PEER'S VOLUME, AND THAT IS A RE-DERIVATION (2026-08-06). It used to be the
    # peer's NAME: among co-admitted contents that both saturate their sustained-firing counts --
    # which two compliant pools on this bus do -- the arbiter's rank key fell through to the source
    # string, so the HEAD was decided by a letter and renaming the peer moved FOCUS on 11 of 12
    # ticks. That criterion asserted `focus_moved >= 1`, so ITS GREEN STATE REQUIRED THE DEFECT: the
    # only arbiter it passed on was one that decides a consciousness-relevant flag by alphabet.
    # `faculties/global_workspace.py::_rank_key` now ranks co-saturating pools on their final
    # MEMBRANE, and the rename moves nothing. Per P8 the criterion is RE-DERIVED rather than
    # weakened -- the PROPERTY (access and focus are two facts, and neither may be settled by a
    # name) is real, the PROXY was pathology-shaped.
    #
    # SO THE STRING TEST IS NOW TWO-SIDED AND STRICTER, not dropped: the rename must move NEITHER
    # readout. Its red is `_name_ranked`, the retired rank key reconstructed on this same bus and
    # RUN below -- under it the rename moves FOCUS, which is what "green only on the defect" looks
    # like when you can finally see it from both directions.
    #
    # THE DISSOCIATION ITSELF MOVED ONTO THE VOLUME LEVER, and the sweep is what justifies 0.95
    # rather than a taste for it. RE-SWEPT 2026-08-07 on the converted one-competition cadence,
    # one peer, 12 ticks each: at 0.00-0.70 this faculty has ACCESS and FOCUS 12/12; from 0.85
    # through 1.00 it holds a slot on 12/12 and the head on 0/12 -- ACCESS WITHOUT FOCUS on every
    # tick. Two rows moved with the cadence and kernel, both in the direction the mechanisms
    # bought: the focus boundary sits between 0.70 and 0.85 (it was 0.55/0.60 on the two-step
    # cadence), and at 1.00 the faculty is NO LONGER excluded -- below capacity nobody is
    # annihilated, so even a non-compliant peer costs it only the head, and the dissociation
    # reaches all the way up. 0.95 stays the lever value: comfortably inside the
    # access-without-focus regime on both cadences. The index is BIT-IDENTICAL between the quiet
    # arm and the 0.95 arm (12/12 by float.hex), which is the property and the reason the lever
    # is legitimate: the peer is changing this faculty's FOCUS and provably not its ACCESS.
    #
    # The load-bearing number is the index. It must be the SAME in the access-without-focus arm as
    # in the access-with-focus arm: it is an ACCESS index, so losing the head to a louder peer may
    # not change it. That single quantity is what BOTH failure modes destroy, in opposite
    # directions, and both are run below as RED arms rather than described -- reading the head
    # zeroes it when a peer out-ranks this faculty, and switching the flag to membership while the
    # BUS still delivers only the head zeroes it through `global_availability` instead.
    peer_lo, peer_hi = _peer_names()
    peer_straddle = peer_lo < AccessConsciousness._SRC < peer_hi
    diss_quiet = _focus_run(0.0, peer_src=peer_lo)
    diss_who = []
    diss_lo = _focus_run(0.95, peer_src=peer_lo, who=diss_who)
    diss_hi = _focus_run(0.95, peer_src=peer_hi)
    # THE RED for the string test: the same rename, on the same bus, under the RETIRED rank key.
    diss_red_lo = _focus_run(0.95, peer_src=peer_lo, ws=_name_ranked(_bus()))
    diss_red_hi = _focus_run(0.95, peer_src=peer_hi, ws=_name_ranked(_bus()))
    red_focus_moved = sum(1 for x, y in zip(diss_red_lo, diss_red_hi) if x[1] != y[1])
    n_both = sum(1 for a, f, _i, _v in diss_quiet if a and f)
    n_access_only = sum(1 for a, f, _i, _v in diss_lo if a and not f)
    # THE THIRD CONJUNCT: on every access-without-focus tick the head must be the PEER'S content,
    # and this arm must post ONE content of its own. Without both, "admitted and not the head" is
    # equally true of a faculty out-ranked by its own retracted bundle with no peer in the
    # workspace at all -- which claim (7)'s middle arm measures at 1 of 10 and which is NOT a
    # dissociation between access and focus. Claim (7) already guards its one-content arm this way;
    # this one did not, and the guard is the difference between a criterion that holds because of
    # the mechanism and one that holds because of how the harness happens to be written.
    own_keys_lo = {k for _h, k in diss_who}
    head_is_peer = sum(
        1
        for (h, k), (a, f, _i, _v) in zip(diss_who, diss_lo)
        if a and not f and h == "peer" and h != k
    )
    # ITS RED, CONSTRUCTED AND RUN AND SWEPT: `_own_stale_head` finds a claim at which this
    # faculty is ADMITTED, is NOT the head, and the head is its OWN retracted content -- with no
    # peer anywhere in the workspace. That is the state the conjunct-free form of the assert scores
    # as a dissociation, and it is a two-step-wide window, which is why it has to be swept rather
    # than named.
    red_claim, red_head, red_mine = _own_stale_head()
    idx_access_only = min(i for _a, _f, i, _v in diss_lo)
    idx_invariant = sum(1 for x, y in zip(diss_quiet, diss_lo) if x[2] == y[2])
    # the rename A/B: ACCESS (flag, index, reach) must not see it; FOCUS must.
    access_same = sum(
        1 for x, y in zip(diss_lo, diss_hi) if (x[0], x[2], x[3]) == (y[0], y[2], y[3])
    )
    focus_moved = sum(1 for x, y in zip(diss_lo, diss_hi) if x[1] != y[1])
    # RED (i) -- CAPACITY. The identical replay on a ONE-slot workspace, rebuilt from the shipped
    # defence proportion exactly as claim (8) rebuilds it. There, ACCESS and FOCUS are one
    # statement and cannot come apart on any tick, which is what makes the dissociation above a
    # property of the multi-slot bottleneck rather than of two accessors that happen to differ.
    diss_one = _focus_run(0.95, peer_src=peer_lo, ws=_bus(**one_slot))
    n_one_apart = sum(1 for a, f, _i, _v in diss_one if a != f)
    # RED (ii) -- DELIVERY. The identical replay with the bus handed back its retired head-only
    # broadcast. Membership is untouched (this faculty is admitted just the same) and the ACCESS
    # readout collapses anyway, because no tap was ever told the workspace contained it. This is
    # the trap in explicit form: switching the flag WITHOUT the delivery change buys a flag that
    # says "ignited" beside a reach of 0.0 and an index of 0.0.
    diss_head_only = _focus_run(0.95, peer_src=peer_lo, ws=_head_only(_bus()))
    n_ho_access = sum(1 for a, _f, _i, _v in diss_head_only if a)
    idx_ho = max(i for _a, _f, i, _v in diss_head_only)
    # THE REPORTING SURFACE, and it is checked rather than assumed because it was policed by
    # NOTHING. Everything above reads the two ACCESSORS; every consumer outside this file reads the
    # `state()` DICT, and `state()["focused"]` is the whole reason the focus question is answerable
    # for them. MEASURED, not reasoned: deleting the `focused` key outright, and separately wiring
    # it to `self._ignited`, each left this module's self-check green AND
    # `reasoning/consciousness_probe.py` (RETIRED at T081, 2026-09-16) green -- so the key the fix was justified by could report
    # the wrong quantity with every gate passing. Both halves are counted, so this cannot pass
    # vacuously: the dict must MIRROR the accessors, and the two keys must actually COME APART.
    st_ws = _bus()
    st_ac = AccessConsciousness(seed=0, ws=st_ws)
    st_claim = st_ac._max_claim()
    st_rows = []
    for _ in range(12):
        # ONE competition per tick (cadence converted 2026-08-07 with the rest of the harness).
        _post(st_ws, [(peer_lo, "peer", 0.95)])
        st_ac.bind_assembly(_coherent("cat", 3), salience=st_claim)
        st_ws.step()
        st_ac.step()
        st_rows.append((st_ac.state(), st_ac.ignited(), st_ac.focused()))
    # `"focused" in s` is tested BEFORE it is read so a DELETED key fails this criterion by its own
    # message rather than by a bare KeyError three frames down -- the red must name the contract it
    # broke, not merely happen.
    state_mirrors = all(
        "focused" in s and s["ignited"] is i and s["focused"] is f for s, i, f in st_rows
    )
    state_two_facts = sum(
        1 for s, _i, _f in st_rows if s["ignited"] != s.get("focused", s["ignited"])
    )

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
        f"({odd_ac.ignition_threshold}); a GUEST clocks it {guest_clocks}x, its OWNER "
        f"{owner_clocks}x -- one competition per cycle, not one per producer"
    )
    print(
        f"one content vs two: ONE content re-asserted agrees with 'slot non-empty' on "
        f"{priv_agree}/{len(priv_seen)} steps; they DISAGREE on {own_disagree}/{len(own_keys)} when "
        f"the SAME faculty rotates its bundle, and on {sh_disagree}/{len(sh_own)} against a peer"
    )
    print(
        f"load decisive: a {shipped_bus.capacity}-slot bus admits this faculty at every load "
        f"below {k_out} peers and first fully denies it at {k_out} (per-load all-or-none; the "
        f"across-load order at exact ties is the jitter's -- see the disclosure at the claim); "
        f"own output differs {load_diff}/{len(alone_own)} ticks (unchanged load, 3 arms incl. "
        f"co-occupancy at {k_co} -> {load_null}); a ONE-slot bus denies at {k_out_one}; on a "
        f"PRIVATE bus no load ever does ({k_out_priv}, own diff {priv_diff})"
    )
    print(
        f"displaceable: cap={cap} -> a 0.95 threat takes the head ({capped_after_threat!r}); "
        f"UNCAPPED at 1.0 it still does ({uncapped_after_threat!r}) because the hold/grip split "
        f"makes that the ARBITER's guarantee; the RED is the pre-split arbiter, where uncapped it "
        f"cannot ({uncapped_one_subsidy!r}); cap floor={cap_floor} open={cap_open}"
    )
    print(
        f"ACCESS vs FOCUS: neutered peer -> access+focus {n_both}/{len(diss_quiet)}; a 0.95 peer "
        f"named {peer_lo!r} -> ACCESS WITHOUT FOCUS {n_access_only}/{len(diss_lo)}, access_index "
        f"unchanged on {idx_invariant}/{len(diss_lo)} (>= {idx_access_only:.6f}); renaming it "
        f"{peer_hi!r} moves FOCUS on {focus_moved}/{len(diss_hi)} and ACCESS on "
        f"{len(diss_hi) - access_same}/{len(diss_hi)} -- the arbiter is name-blind"
    )
    print(
        f"   and the head is the PEER on {head_is_peer}/{n_access_only} of those ticks, with this "
        f"faculty posting {len(own_keys_lo)} distinct content(s); its RED -- NO peer, this faculty "
        f"behind its OWN retracted content at claim {red_claim} -- reaches access-without-focus "
        f"with the head being its own"
    )
    print(
        f"   REDs: a 1-slot bus never separates them ({n_one_apart}/{len(diss_one)} apart); "
        f"head-only delivery keeps access {n_ho_access}/{len(diss_head_only)} and zeroes the index "
        f"(max {idx_ho}); the RETIRED rank key moves FOCUS on {red_focus_moved}/{len(diss_red_lo)} "
        f"under the same rename"
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
    assert tap_is_workspace, (
        f"a subscribed tap must receive the WORKSPACE TUPLE the same cycle's broadcast_all() "
        f"returns -- the reach scan is a membership test, and a bare content delivered in its "
        f"place would satisfy it by SUBSTRING and report reach that never happened. Got "
        f"{type(tap_payload).__name__} {tap_payload!r} against {strong.workspace.broadcast_all()!r}"
    )
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
    assert guest_clocks == 0 and owner_clocks == 1, (
        f"a faculty handed SOMEBODY ELSE'S bus must submit into it and let the owner run the one "
        f"competition per cycle, or one tick runs two competitions with one submitter each and the "
        f"producers never meet (measured live: co-occupancy 0/120 with the identity assert above "
        f"passing). Guest advanced the arbiter {guest_clocks} time(s), owner {owner_clocks}. "
        f"`_own_ws` had recorded this distinction since the injection parameter landed and nothing "
        f"acted on it -- a flag whose value was asserted and whose consequence was nowhere"
    )
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
        f"with ONE content in the competition, 'slot non-empty' and 'the slot is MINE' must agree "
        f"on every step: {priv_agree}/{len(priv_seen)}"
    )
    assert own_disagree >= 1, (
        f"...and TWO contents must break that, EVEN WHEN BOTH ARE THIS FACULTY'S OWN. A rotating "
        f"bundle means the faculty competes with what it was asserting a moment ago, so the head "
        f"can be its own retracted content while its current one holds a slot beneath. If this is "
        f"0 the key has stopped rotating and `content_key`'s whole argument is dead: "
        f"{own_disagree}/{len(own_keys)}"
    )
    assert sh_disagree >= 1, (
        f"...and so must a louder PEER on a shared bus, or the equivalence asserted above is "
        f"vacuous and the winner-identity test is untested: {sh_disagree}/{len(sh_own)}"
    )
    # (8) the other faculties on the shared bus decide this one's ACCESS -- with its own null and
    #     its own RED, both run.
    assert k_out is not None, (
        f"some population of competing faculties must be able to deny this faculty the shared "
        f"workspace: no load from 0 to {k_scan} peers did, so nothing on this bus is decisive for "
        f"it and it is a private workspace wearing a shared one's name"
    )
    # (The monotone-step assert that stood here was DELETED with the two-competition cadence --
    # see the disclosure at the `load_runs` measurement for the profile, the cause, and why no
    # replacement assert earned a place.)
    assert load_diff == len(alone_own), (
        f"filling the shared workspace with other faculties must change this faculty's OWN output "
        f"on EVERY tick against an empty one, got {load_diff}/{len(alone_own)}"
    )
    assert load_null == 0, (
        f"an UNCHANGED load must reproduce ALL THREE arms exactly -- empty, CO-OCCUPIED ({k_co} "
        f"peers, the only regime in which this faculty shares the workspace) and crowded -- or the "
        f"difference above is the harness and not the bus: {load_null}/{3 * len(alone_own)}"
    )
    assert k_out_priv is None and priv_diff == 0, (
        f"THE RED: the identical replay with this faculty on its OWN workspace must be untouched "
        f"by the load -- no peer count may deny it access (got {k_out_priv}) and its own output "
        f"must not move (got {priv_diff}/{len(alone_own)}). A private instance that moves means "
        f"this claim is measuring the harness, which is the defect the injection exists to remove"
    )
    assert k_out_one == 1 and k_out_one < k_out, (
        f"CAPACITY is what this faculty gets from the shared bus, so it must be what MOVES the "
        f"threshold (under the rheobase kernel it EQUALS it -- see above): on a ONE-slot "
        f"workspace a SINGLE other faculty must deny this one access "
        f"(got {k_out_one} -- that IS the eviction the retired single-slot criterion measured), "
        f"and the shipped {shipped_bus.capacity}-slot bus must tolerate STRICTLY more before "
        f"denying it (got {k_out}). Equality of the two means multi-slot bought this faculty "
        f"nothing"
    )
    # (9) joining a shared bus must not silence it -- the uncapped arm is the reachable RED.
    assert capped_after_threat == "threat", (
        f"with the claim capped at the bus's own displaceability bound, a 0.95 proximity threat "
        f"must still seize the slot, got {capped_after_threat!r}"
    )
    assert uncapped_after_threat == "threat", (
        f"...and on the SPLIT arbiter it must do so even UNCAPPED -- head displaceability is the "
        f"arbiter's guarantee now, not the faculty's own restraint, so a faculty that ignores the "
        f"cap entirely still cannot hold the focus against a threat (got "
        f"{uncapped_after_threat!r})"
    )
    assert uncapped_one_subsidy != "threat", (
        f"claim (9) must be reachable-red, and its red is the PRE-SPLIT ARBITER: with the head "
        f"ranked on the HOLD instead of the GRIP, an uncapped faculty drives 1.0 + hold against "
        f"the threat's 0.95 and the threat must FAIL to take the head, or this claim is asserting "
        f"something that was true anyway (got {uncapped_one_subsidy!r})"
    )
    assert not cap_floor_ignites and cap_floor == cap_floor_ws.ignition_threshold, (
        f"the cap must only ever LOWER a claim: on a bus with no displaceable room it floors at "
        f"the rheobase ({cap_floor}) and a sub-threshold assembly still stays local "
        f"(ignited={cap_floor_ignites})"
    )
    assert cap_open == 1.0, f"a bus with no incumbent defence caps at 1.0, got {cap_open}"
    # (10) ACCESS and FOCUS are two facts and this faculty reports them as two -- with BOTH REDs run.
    assert peer_straddle, (
        f"claim (10)'s lever is the rank tie-break on the SOURCE NAME, so its two peer names must "
        f"straddle this faculty's own {AccessConsciousness._SRC!r}: got {peer_lo!r} / {peer_hi!r}"
    )
    assert n_both == len(diss_quiet), (
        f"with the peer NEUTERED this faculty must be BOTH in the workspace and its head, or the "
        f"focus arm is not measuring focus: {n_both}/{len(diss_quiet)}"
    )
    assert n_access_only == len(diss_lo), (
        f"THE DISSOCIATION: against a 0.95 peer that out-ranks it, this faculty must hold a "
        f"workspace slot while NOT being the head, on every tick. Anything else means access and "
        f"focus have re-collapsed and this module is back to reporting headship where it means "
        f"access: {n_access_only}/{len(diss_lo)} (raw {diss_lo[:3]})"
    )
    assert len(own_keys_lo) == 1 and head_is_peer == n_access_only, (
        f"...AND THE HEAD MUST BE THE PEER'S CONTENT (added 2026-08-06 by an adversarial re-run). "
        f"'admitted and not the head' is equally true of a faculty out-ranked by its OWN retracted "
        f"bundle with no peer in the workspace at all -- claim (7)'s middle arm measures exactly "
        f"that at 1 of 10, and it is a fact about hysteresis, not a dissociation between access and "
        f"focus. This arm avoids it only by re-posting ONE content, which was an unasserted "
        f"property of the harness until now: {len(own_keys_lo)} distinct own key(s), head was the "
        f"peer on {head_is_peer} of the {n_access_only} access-without-focus ticks"
    )
    assert red_claim is not None and red_head != red_mine, (
        f"...and that conjunct must be reachable-red, so the forbidden state is CONSTRUCTED, SWEPT "
        f"and RUN: with NO peer on the bus, this faculty asserting a fresh content behind its own "
        f"still-charged retracted one is ADMITTED and NOT the head, and the head is its own. Found "
        f"at claim {red_claim} (head {red_head!r} vs asserting {red_mine!r}). If this is None the "
        f"conjunct is asserting something that was true anyway and P8 says DELETE it, not keep it"
    )
    assert idx_access_only > 0.0 and idx_invariant == len(diss_lo), (
        f"an ACCESS index must not depend on holding the FOCUS -- losing the head to a louder peer "
        f"must leave it exactly where it was: unchanged on {idx_invariant}/{len(diss_lo)} ticks, "
        f"minimum {idx_access_only}"
    )
    assert access_same == len(diss_lo) and focus_moved == 0, (
        f"THE STRING TEST, two-sided since 2026-08-06. Renaming the peer {peer_lo!r} -> {peer_hi!r} "
        f"changes no claim, no content and no competition, so BOTH readouts must be identical on "
        f"every tick: ACCESS (flag, index, reach) {access_same}/{len(diss_lo)}, FOCUS moved on "
        f"{focus_moved}. This asserted `focus_moved >= 1` until the arbiter's rank key stopped "
        f"falling through to the source name -- i.e. its green state REQUIRED a consciousness-"
        f"relevant flag to be decided by a letter. Re-derived, not weakened (P8/F11)"
    )
    assert red_focus_moved >= 1, (
        f"...and that criterion must be reachable-red, so the RETIRED RANK KEY is reconstructed on "
        f"this same bus (`_name_ranked`, a one-method override -- same pools, same drives, same "
        f"settle) and RUN: under `(-late, -total, k)` the identical rename must MOVE this faculty's "
        f"FOCUS on at least one tick, or the name-blindness asserted above was true anyway and this "
        f"claim is measuring nothing. Got {red_focus_moved}/{len(diss_red_lo)}"
    )
    assert n_one_apart == 0, (
        f"RED (i): on a ONE-slot workspace 'in the workspace' and 'is the head' are the SAME "
        f"statement and must never disagree -- if they do, the dissociation above is a property of "
        f"these two accessors rather than of the capacity, and claim (7) is wrong too: "
        f"{n_one_apart}/{len(diss_one)} ticks disagreed"
    )
    # RED (ii) is SPLIT rather than compound, because a single message on a two-condition assert
    # names a cause it has not established. Measured: under an unrelated `gw-reordered` delivery
    # mutant the compound form fired and printed "handed back its retired HEAD-ONLY broadcast ...
    # must collapse to 0.0 (max 0.8660714285714286)" -- no head-only revert had happened, and the
    # printed number contradicted the sentence demanding it. That is the defect this file's own
    # LIMIT 4 records against rpm_solver, committed in the file that names it. Each half now
    # reports only what it measured.
    assert n_ho_access == len(diss_head_only), (
        f"RED (ii-a), MEMBERSHIP: under head-only delivery this faculty is still ADMITTED on every "
        f"tick -- got {n_ho_access}/{len(diss_head_only)}. If admission itself moved, the arm is not "
        f"isolating delivery and the index reading below is about the wrong variable"
    )
    assert idx_ho == 0.0, (
        f"RED (ii-b), REACH: with the bus handed back its retired HEAD-ONLY broadcast, admission is "
        f"unchanged (see ii-a) and access_index must therefore collapse to 0.0 -- got max {idx_ho}. "
        f"If it does not, the delivery change in faculties/global_workspace.py is buying nothing here "
        f"and should be reverted rather than believed. NOTE this asserts the value under the "
        f"HEAD-ONLY shim only; a non-zero here means either the shim did not apply or availability "
        f"stopped depending on the delivered payload -- check `_deliver` before believing either"
    )
    assert state_mirrors and state_two_facts == len(st_rows), (
        f"THE REPORTING SURFACE must carry the dissociation, not just the accessors. `state()` is "
        f"what every consumer outside this file reads, so its 'ignited' and 'focused' keys must "
        f"MIRROR ignited()/focused() on every tick AND must actually disagree on the "
        f"access-without-focus arm -- a 'focused' key that echoes 'ignited' is this claim's "
        f"collapse wearing the readout instead of the flag, and a missing one makes the question "
        f"unanswerable for the eight live consumers of this dict "
        f"(mirrors={state_mirrors}, keys disagreed on {state_two_facts}/{len(st_rows)} ticks)"
    )
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
