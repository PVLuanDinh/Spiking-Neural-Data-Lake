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
Built entirely on two existing zero-dep spiking primitives (no new mechanism reinvented):

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

Determinism: per-instance random.Random(seed) for the workspace; neuron spike codes are
crc32-seeded (no module-level / time seeding). Pure stdlib, zero dependencies. ASCII output.

  Run: python -m snn_data_lake.cortex.access_consciousness      # (also run by run_selfchecks.py)
"""
import random
import sys
import zlib

from snn_data_lake.cortex.vsa_reasoning import bundle, cleanup, hv, sim
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

    _LABEL = "GLOBAL"          # hashable workspace handle for the (single) conscious-content slot
    _SRC = "coalition"         # workspace source name for the grown assembly coalition

    def __init__(self, seed=0, ignition_threshold=0.55, n_taps=4, k_sat=5, noise=0.05):
        self.seed = seed
        self.ignition_threshold = ignition_threshold
        self.n_taps = max(2, int(n_taps))        # >=2 independent downstream taps (global reach)
        self.k_sat = max(2, int(k_sat))          # neuron count at which completeness saturates
        self.noise = noise                       # bit-flip fraction of a concept-member spike code
        self._rng = random.Random(seed)          # (reserved) per-instance RNG, no global seeding
        self._ws = GlobalWorkspace(ignition_threshold=ignition_threshold, seed=seed)
        # downstream taps: each independent subscriber records the content it last received.
        self._taps = []
        for _ in range(self.n_taps):
            inbox = []
            self._ws.subscribe(inbox.append)
            self._taps.append(inbox)
        # coalition grown across bind_assembly calls this cycle: list of (neuron_id, spike_code).
        self._members = []
        self._n_assemblies = 0
        self._content_hv = None                  # the bundled conscious-content spike code
        # last-step functional readout.
        self._ignited = False
        self._availability = 0.0
        self._integration = 0.0
        self._n_bound = 0

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
            v[i] = -v[i]                          # flip one spike line (up<->down)
        return v

    # -- coherence / integration proxies ----------------------------------------------------------
    @staticmethod
    def _coherence(hvs):
        """Mean pairwise POSITIVE spike-coincidence similarity among member codes, in [0,1].

        High for a mutually coincident (coherent) assembly, ~0 for an orthogonal (scrambled) one.
        Binding is RELATIONAL: fewer than two neurons cannot be a bound assembly -> coherence 0."""
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

    # -- GROW + BUNDLE + submit -------------------------------------------------------------------
    def bind_assembly(self, neuron_ids, salience=None):
        """GROW the conscious-content coalition with these neurons and BUNDLE it into ONE workspace
        candidate. Salience defaults to the coherence/size-driven drive current (pass `salience` only
        to inject a fixed background distractor). Returns the workspace content handle."""
        for nid in neuron_ids:
            self._members.append((str(nid), self.neuron_hv(nid)))
        self._n_assemblies += 1
        codes = [h for _, h in self._members]
        self._content_hv = bundle(codes)                          # spike superposition -> ONE content
        coherence = self._coherence(codes)
        completeness = self._completeness(len(codes))
        sal = self._salience(coherence, completeness) if salience is None \
            else max(0.0, min(1.0, float(salience)))
        self._ws.submit(self._SRC, self._LABEL, sal)              # post the candidate for ignition
        return self._LABEL

    def content_vector(self):
        """The bundled spike code of the current conscious-content candidate (or None)."""
        return self._content_hv

    # -- one ignition cycle -----------------------------------------------------------------------
    def step(self):
        """Advance the workspace one tick: ignition competition + all-or-none global broadcast.
        Returns whether IGNITION occurred (True/False) and refreshes the functional readout."""
        content = self._ws.step()                                # broadcasts to every subscribed tap
        self._ignited = content is not None
        received = sum(1 for tap in self._taps if tap and tap[-1] is not None)
        self._availability = received / self.n_taps
        if self._ignited:
            codes = [h for _, h in self._members]
            self._integration = max(0.0, min(1.0,
                                             self._coherence(codes) * self._completeness(len(codes))))
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
    return [f"c{i}#0" for i in range(k)]          # k DIFFERENT concepts -> ~orthogonal (incoherent)


def main():
    # ---- (1) IGNITION IS ALL-OR-NONE: coherent+salient ignites (reaches >=2 taps); weak/incoherent
    #          does NOT. The control (a scrambled assembly) proves ignition is a real threshold. ----
    strong = _fresh()
    strong.bind_assembly(_coherent("cat", 5))            # 5 coincident neurons -> supra-threshold
    strong.step()
    strong_reach = sum(1 for tap in strong._taps if tap and tap[-1] is not None)

    weak = _fresh()
    weak.bind_assembly(_scrambled(5))                    # 5 orthogonal neurons -> sub-threshold
    weak.step()

    # ---- (2) BUNDLING BINDS: every bundled neuron is recoverable from the ONE content; a
    #          non-bundled distractor is NOT (control). Pure VSA superposition property. ----------
    ac = _fresh()
    ids = [f"n{i}" for i in range(5)]                    # 5 distinct (orthogonal) neurons
    codes = [ac.neuron_hv(n) for n in ids]
    content = bundle(codes)                              # bind them into ONE conscious content
    distractor = ac.neuron_hv("distractor")             # never bundled in
    member_sims = [sim(content, c) for c in codes]
    distractor_sim = sim(content, distractor)
    codebook = dict(zip(ids, codes))
    codebook["distractor"] = distractor
    cleaned, _ = cleanup(content, codebook)             # snaps to a bundled member, not the distractor
    bundling_binds = (min(member_sims) > distractor_sim
                      and all(m > 0.0 for m in member_sims)
                      and distractor_sim < 0.15
                      and cleaned in ids)

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

    # ---- (5) HONEST CEILING + DETERMINISM. --------------------------------------------------------
    st = strong.state()
    gate = 1.0 if st["ignited"] else 0.0
    reproduced = gate * st["global_availability"] * st["integration"]
    no_phenomenal_key = all(all(tok not in key for tok in _BANNED) for key in st)
    a1 = _fresh(); a1.bind_assembly(_coherent("cat", 5)); a1.step()
    a2 = _fresh(); a2.bind_assembly(_coherent("cat", 5)); a2.step()
    deterministic = a1.state() == a2.state()
    zero_dep = "numpy" not in sys.modules and "torch" not in sys.modules

    # -------------------------------------- ASCII report ------------------------------------------
    rows = [
        ("strong-coherent(5)", strong.state()),
        ("weak-incoherent(5)", weak.state()),
        ("coalition bird(3+3)", coalition.state()),
    ]
    print("=" * 78)
    print("ACCESS CONSCIOUSNESS -- Global Neuronal Workspace signature (FUNCTIONAL, not phenomenal)")
    print("=" * 78)
    print("pipeline: GROW neurons -> BUNDLE (VSA) -> IGNITE (all-or-none) -> BROADCAST -> ACCESS index")
    print(f"taps={strong.n_taps}  ignition_threshold={strong.ignition_threshold}  "
          f"index = ignition_gate * global_availability * integration\n")
    print(f"{'scenario':<20}{'ignited':>9}{'avail':>8}{'n_bound':>9}{'integ':>8}{'access_idx':>12}")
    print("-" * 78)
    for name, s in rows:
        print(f"{name:<20}{str(s['ignited']):>9}{s['global_availability']:>8.2f}"
              f"{s['n_bound']:>9}{s['integration']:>8.3f}{s['access_index']:>12.3f}")
    print("-" * 78)
    print(f"bundling recoverability: min member sim {min(member_sims):.3f} > distractor {distractor_sim:.3f}"
          f"  (cleanup -> {cleaned!r})")
    print(f"availability: ignited {avail_ignited:.2f} vs non-igniting {avail_nonignited:.2f}"
          f"  (broadcast reached {strong_reach}/{strong.n_taps} taps)")
    print(f"index coherent {index_coherent:.3f} > scrambled {index_scrambled:.3f}"
          f"  |  accumulate-coherent-neurons trend (k=2..5): "
          f"{', '.join(f'{x:.3f}' for x in trend)}")
    print("MEASURES ACCESS (global availability / ignition), NOT phenomenal experience "
          "-- no subjective-experience claim.")
    print("=" * 78)

    # ------------------------------------- assertions ---------------------------------------------
    # (1) all-or-none ignition + control.
    assert strong.ignited(), "a strong, coherent bundled assembly must IGNITE (all-or-none broadcast)"
    assert strong_reach >= 2, f"ignition must broadcast to >=2 downstream taps, got {strong_reach}"
    assert not weak.ignited(), "a weak/incoherent assembly must NOT ignite (ignition is a real threshold)"
    # (2) bundling genuinely binds neurons into ONE recoverable content; the distractor stays out.
    assert bundling_binds, "bundling must bind every member recoverably while excluding the distractor"
    # (3) global availability is present after ignition, absent otherwise.
    assert avail_ignited > avail_nonignited, "ignited content must be globally available, non-igniting not"
    assert avail_ignited == 1.0 and avail_nonignited == 0.0, "availability must be all-or-none across taps"
    # (4) access_index rises with integration; coherent beats scrambled; index in [0,1]; trend climbs.
    assert index_coherent > index_scrambled + 0.2, (
        f"coherent index must clearly beat scrambled ({index_coherent:.3f} vs {index_scrambled:.3f})")
    assert 0.0 <= index_coherent <= 1.0 and 0.0 <= index_scrambled <= 1.0, "access_index must be in [0,1]"
    assert trend[-1] > trend[0], f"index must rise as coherent neurons accumulate: {trend}"
    assert all(trend[i + 1] >= trend[i] - 1e-9 for i in range(len(trend) - 1)), (
        f"the accumulation trend must be monotone-ish (non-decreasing): {trend}")
    assert coalition.ignited() and coalition.n_bound() == 2, "two coherent assemblies must fuse (n_bound==2)"
    # (5) honest ceiling: no phenomenal field; index reproduces from the 3 functional measures alone.
    assert no_phenomenal_key, f"state() must contain NO phenomenal key {_BANNED}: {list(st)}"
    assert abs(st["access_index"] - reproduced) < 1e-9, (
        "access_index must be a pure function of ignition * global_availability * integration")
    assert deterministic, "two seed=0 instances must reach identical state after the same inputs"
    assert zero_dep, "must be pure stdlib (no numpy/torch imported)"

    print("self-check OK: "
          f"all-or-none ignition (strong ignites/reaches {strong_reach} taps, weak does not); "
          f"bundling binds {len(ids)} neurons (distractor excluded); global availability "
          f"{avail_ignited:.1f} vs {avail_nonignited:.1f}; access_index coherent {index_coherent:.3f} > "
          f"scrambled {index_scrambled:.3f}, accumulation trend {trend[0]:.3f}->{trend[-1]:.3f}; "
          "functional-only (no phenomenal claim); deterministic")


if __name__ == "__main__":
    main()
