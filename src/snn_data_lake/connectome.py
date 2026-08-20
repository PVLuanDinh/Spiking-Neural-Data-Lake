"""Mesoscale schematic connectome -- the white-matter WIRING layer of the brain architecture.

`brain_atlas.py` is the region -> module label map (the NODES). This module is the EDGES: a typed,
directed, signed, loop-aware inter-region projection graph, so the project is a queryable
ARCHITECTURE, not just a bag of region modules whose only connections are Python imports (which
carry no direction, sign, weight, or loop-closure).

Nodes are canonical brain sub-structures, each tagged with the project region it belongs to. Edges
are the major long-range pathways, each typed:
  * tract  : the anatomical pathway name (e.g. "corticospinal", "perforant path")
  * sign   : exc | inh | da | ne | 5ht | ach   (excitatory / inhibitory / neuromodulatory class)
  * cls    : relay | feedback | broadcast | descending
  * loop   : which canonical loop the edge participates in (or None)
The graph encodes the canonical loops -- thalamocortical, cortico-basal-ganglia-thalamo-cortical
(CBGTC), the hippocampal trisynaptic loop, cortico-cerebellar (via the pontine relay), Papez
(limbic), the tecto-thalamo-cortical orienting loop, the habenula reward-opponent loop, the
olfactory bulb<->piriform loop, and the claustro-cortical loop. Plus the amygdala threat pathway,
the ascending neuromodulatory broadcast (dopamine, noradrenaline/LC, serotonin/raphe, ACh, ARAS),
and the descending corticospinal path to the arm.

HONEST label: this is a MESOSCALE SCHEMATIC of canonical pathways (like the Allen mesoscale atlas),
NOT a synapse-level or measured connectome. It asserts graph-topological facts (loop closure, the
single gated actuator path), not biological connection strengths.

Safety tie-in: the ONLY path from any cortical node to the arm actuator runs through the brainstem
motor node's gated edge -- a deterministic self-check that reinforces the arm-safety envelope (no
region reaches the actuator except through the reflex/E-STOP-gated final common path).

  python -m snn_data_lake.connectome        # (also run by run_selfchecks.py)
"""

import json
import os

# node_id -> (project region or None for the effector, human-readable structure name)
NODES = {
    "ctx_sensory": ("cortex", "sensory cortex"),
    "ctx_motor": ("cortex", "motor cortex"),
    "ctx_pfc": ("cortex", "prefrontal / association cortex"),
    "thal_relay": ("thalamus", "thalamic relay nucleus"),
    "trn": ("thalamus", "thalamic reticular nucleus (TRN, inhibitory gate)"),
    # specs/022 T021a: the higher-order thalamic nuclei + the insular terminus. pulvinar and
    # md are HIGHER-ORDER relays (driven BY cortex, projecting back to cortex -- the
    # transthalamic cortico-pulvinar-cortical route P6's topology needs); insula is the
    # interoceptive cortical terminus (module: insula/interoceptive_integration).
    "pulvinar": ("thalamus", "pulvinar (higher-order visual/attentional relay)"),
    "md": ("thalamus", "mediodorsal nucleus (MD, prefrontal higher-order relay)"),
    "insula": ("insula", "insular cortex (interoceptive integration)"),
    "striatum": ("basal_ganglia", "striatum (D1 direct / D2 indirect)"),
    "gpe": ("basal_ganglia", "external pallidum (GPe)"),
    "gpi_snr": ("basal_ganglia", "internal pallidum / SNr (BG output)"),
    "stn": ("basal_ganglia", "subthalamic nucleus (STN)"),
    "vta_snc": ("midbrain", "VTA / SNc (dopamine source nuclei)"),
    "ec": ("hippocampus", "entorhinal cortex"),
    "dg": ("hippocampus", "dentate gyrus"),
    "ca3": ("hippocampus", "CA3"),
    "ca1": ("hippocampus", "CA1"),
    "sub": ("hippocampus", "subiculum"),
    "bla": ("amygdala", "basolateral amygdala (BLA)"),
    "cea": ("amygdala", "central amygdala (CeA)"),
    "cb_ctx": ("cerebellum", "cerebellar cortex (Purkinje)"),
    "cb_deep": ("cerebellum", "deep cerebellar nuclei"),
    "hypothal": ("hypothalamus", "hypothalamus (mammillary body / HPA)"),
    "bf_ach": ("neuromodulation", "basal forebrain (acetylcholine)"),
    "scn": ("hypothalamus", "suprachiasmatic nucleus (SCN, circadian clock)"),
    "sc": ("midbrain", "superior colliculus (orienting / saccade map)"),
    "pn": ("pons", "pontine nuclei (cortico-ponto-cerebellar relay)"),
    "prf": ("pons", "pontine reticular formation (ascending reticular activating system)"),
    "lc": ("neuromodulation", "locus coeruleus (noradrenaline, arousal gain)"),
    "raphe": ("neuromodulation", "dorsal raphe (serotonin, patience / mood)"),
    "ob": ("olfactory", "olfactory bulb (glomerular decorrelation)"),
    "pir": ("olfactory", "piriform cortex (odor association)"),
    "lhb": ("epithalamus", "lateral habenula (anti-reward / negative RPE)"),
    "claustrum": ("claustrum", "claustrum (cortical synchrony / salience hub)"),
    "bs_motor": ("brainstem", "brainstem / spinal motor + reflex (the gated final path)"),
    # v3.5 completed-anatomy: medulla autonomic vital nuclei + SC deep motor layers
    "pbc": ("medulla", "pre-Botzinger complex (medullary respiratory rhythm generator)"),
    "nts_rvlm": ("medulla", "NTS / RVLM (medullary baroreflex cardiovascular nuclei)"),
    "sc_deep": ("midbrain", "superior colliculus deep layers (gaze burst generator)"),
    # v3.36 completed-anatomy: the teaching/broadcast/reward hubs the audit found missing
    "io": ("medulla", "inferior olive (climbing-fibre error / teaching signal to cerebellum)"),
    "gwt": ("thalamus", "global workspace / GNW ignition hub (Dehaene-Changeux broadcast)"),
    "nacc": ("basal_ganglia", "nucleus accumbens / ventral striatum (mesolimbic reward)"),
    "actuator": (None, "robot arm actuator"),
    "viscera": (None, "autonomic effector (diaphragm + heart) -- NOT the arm actuator"),
}


def _e(source, target, tract, sign, cls, loop=None, weight=1.0, delay=1):
    """One long-range projection.

    `delay` is CONDUCTION DELAY in ticks, added 2026-08-04. Until then this constructor had seven
    fields and none of them was delay, so every tract in the brain -- thalamocortical radiation,
    corticostriatal, pallidothalamic -- arrived in exactly one tick. In biology these differ by an
    order of magnitude, and conduction delay is what STDP and coincidence detection are FOR.

    The consumer was already here and complete: `thalamus/connectome_router.py:83` reads
    `e.get("delay", default_delay)` and `:101` sizes a ring buffer to the longest delay. It has
    simply never been handed a value -- a consumer waiting on a producer that did not exist, which
    is P6 seen from the structural side.

    DEFAULT IS 1, deliberately: adding the field changes no behaviour on its own. Varied values are
    an EXPERIMENT (see the A/B in `tools/ab_connectome_delay.py`) and are not adopted here. If the
    A/B shows delay variation charges an outcome, then supplying real per-tract values becomes a P1
    obligation and they must come from anatomy (the 2024 multi-scale human cortex model is
    DTI-derived and open source), NOT from numbers chosen to make a demo work.
    """
    return {
        "source": source,
        "target": target,
        "tract": tract,
        "sign": sign,
        "cls": cls,
        "loop": loop,
        "weight": weight,
        "delay": delay,
    }


# the directed, typed edge list (canonical long-range projections)
EDGES = [
    # specs/022 T021a -- the interoceptive tract terminus + transthalamic routes. DELAYS ARE
    # DERIVED, not picked, and the arithmetic is shown so the derivation survives review:
    #   delay_ticks = max(1, round(tract_latency_ms / tick_ms))
    # tract_latency_ms = conduction + one synaptic delay for a mono-synaptic mammalian tract,
    # 2-15 ms across every tract below (myelinated central axons 10-80 m/s over <=15 cm paths
    # + ~1 ms synapse); tick_ms = this substrate's OWN measured wake-tick period, ~107 ms p50
    # (tools/perf_baseline.json, S0-kernel stage) -- so every mono-synaptic tract derives to
    # max(1, round(0.02..0.14)) = 1, and multi-hop routes accumulate latency STRUCTURALLY
    # (the transthalamic route is 2 ticks over its 2 edges, insula ignition 3 over 3). The
    # per-tract VALUE variation for the pre-existing 83 edges remains the recorded open P1
    # (see the module note above _e); these eight discharge it for the NEW edges by deriving.
    _e("nts_rvlm", "thal_relay", "viscerosensory NTS -> thalamus relay", "exc", "relay"),
    _e("thal_relay", "insula", "interoceptive thalamocortical radiation", "exc", "relay"),
    _e("insula", "gwt", "insular ignition afferent (interoceptive salience)", "exc", "relay"),
    _e("insula", "ctx_pfc", "insulo-prefrontal projection", "exc", "relay"),
    _e(
        "ctx_sensory",
        "pulvinar",
        "cortico-pulvinar driver (transthalamic leg 1)",
        "exc",
        "descending",
    ),
    _e("pulvinar", "ctx_pfc", "pulvino-cortical projection (transthalamic leg 2)", "exc", "relay"),
    _e("ctx_pfc", "md", "prefronto-mediodorsal corticothalamic", "exc", "descending"),
    _e("md", "ctx_pfc", "mediodorsal-prefrontal radiation", "exc", "relay"),
    # thalamocortical loop (+ the TRN inhibitory gate)
    _e("thal_relay", "ctx_sensory", "thalamocortical radiation", "exc", "relay", "thalamocortical"),
    _e(
        "ctx_sensory",
        "thal_relay",
        "corticothalamic feedback",
        "exc",
        "feedback",
        "thalamocortical",
    ),
    _e("ctx_sensory", "trn", "corticoreticular", "exc", "feedback", "thalamocortical"),
    _e("trn", "thal_relay", "reticulothalamic (gate)", "inh", "feedback", "thalamocortical"),
    # cortico-basal-ganglia-thalamo-cortical (CBGTC) selection loop
    _e("ctx_motor", "striatum", "corticostriatal", "exc", "relay", "cbgtc"),
    _e("striatum", "gpi_snr", "striatonigral (direct / Go)", "inh", "relay", "cbgtc"),
    _e("striatum", "gpe", "striatopallidal (indirect / NoGo)", "inh", "relay", "cbgtc"),
    _e("gpe", "stn", "pallidosubthalamic", "inh", "relay", "cbgtc"),
    _e("stn", "gpi_snr", "subthalamopallidal", "exc", "relay", "cbgtc"),
    _e("gpe", "gpi_snr", "intrapallidal", "inh", "relay", "cbgtc"),
    _e("gpi_snr", "thal_relay", "pallidothalamic (tonic inhibition)", "inh", "relay", "cbgtc"),
    _e("thal_relay", "ctx_motor", "thalamocortical (motor)", "exc", "relay", "cbgtc"),
    _e("vta_snc", "striatum", "nigrostriatal (dopamine)", "da", "broadcast", "cbgtc"),
    # hippocampal trisynaptic loop
    _e("ec", "dg", "perforant path", "exc", "relay", "hippocampal"),
    _e("dg", "ca3", "mossy fibers", "exc", "relay", "hippocampal"),
    _e("ca3", "ca1", "Schaffer collaterals", "exc", "relay", "hippocampal"),
    _e("ca1", "sub", "CA1-subiculum", "exc", "relay", "hippocampal"),
    _e("sub", "ec", "subiculum-entorhinal (return)", "exc", "feedback", "hippocampal"),
    # cortico-cerebellar loop (via the explicit pontine mossy-fibre relay)
    _e("ctx_motor", "pn", "corticopontine", "exc", "relay", "cortico_cerebellar"),
    _e("pn", "cb_ctx", "pontocerebellar mossy fibers", "exc", "relay", "cortico_cerebellar"),
    _e("cb_ctx", "cb_deep", "Purkinje projection", "inh", "relay", "cortico_cerebellar"),
    _e("cb_deep", "thal_relay", "dentatothalamic", "exc", "relay", "cortico_cerebellar"),
    # Papez / limbic circuit
    _e("sub", "hypothal", "fornix -> mammillary body", "exc", "relay", "papez"),
    _e("hypothal", "thal_relay", "mammillothalamic tract", "exc", "relay", "papez"),
    _e("thal_relay", "ctx_pfc", "thalamo-cingulate", "exc", "relay", "papez"),
    _e("ctx_pfc", "ec", "cingulum -> entorhinal", "exc", "relay", "papez"),
    # amygdala threat pathway
    _e("ctx_sensory", "bla", "sensory -> amygdala (CS)", "exc", "relay", "amygdala"),
    _e("bla", "cea", "intra-amygdala", "exc", "relay", "amygdala"),
    _e("cea", "hypothal", "amygdala -> HPA (stress)", "exc", "descending", "amygdala"),
    _e(
        "cea",
        "bs_motor",
        "amygdala -> PAG/brainstem (freeze/defense)",
        "exc",
        "descending",
        "amygdala",
    ),
    # ascending neuromodulatory broadcast
    _e("vta_snc", "ctx_pfc", "mesocortical dopamine", "da", "broadcast", None),
    _e("bf_ach", "ctx_sensory", "cholinergic broadcast", "ach", "broadcast", None),
    _e("bf_ach", "ec", "septohippocampal cholinergic", "ach", "broadcast", None),
    # descending corticospinal path to the effector -- THE gated final common path
    _e("ctx_motor", "bs_motor", "corticospinal tract", "exc", "descending", "corticospinal"),
    _e(
        "bs_motor",
        "actuator",
        "final common motor path (reflex/E-STOP gated)",
        "exc",
        "descending",
        "corticospinal",
    ),
    # --- completed-anatomy pathways (v3.4): hypothalamus, midbrain, pons/ARAS, olfactory,
    #     epithalamus, claustrum, and the LC/raphe neuromodulatory broadcast ---
    # suprachiasmatic circadian pacing (hypothalamus)
    _e("scn", "hypothal", "SCN -> hypothalamus (circadian HPA pacing)", "exc", "relay"),
    _e("scn", "prf", "SCN -> reticular formation (circadian arousal drive)", "exc", "broadcast"),
    # locus-coeruleus noradrenaline broadcast (the reserved 'ne' class finally has its source node)
    _e("lc", "ctx_sensory", "coeruleocortical noradrenaline", "ne", "broadcast"),
    _e("lc", "ctx_pfc", "LC -> PFC noradrenaline (gain)", "ne", "broadcast"),
    _e("lc", "thal_relay", "LC -> thalamus (arousal gain)", "ne", "broadcast"),
    # dorsal-raphe serotonin broadcast (the reserved '5ht' class)
    _e("raphe", "ctx_pfc", "serotonergic projection (patience / mood)", "5ht", "broadcast"),
    _e("raphe", "striatum", "raphe -> striatum (5-HT, opponent to dopamine)", "5ht", "broadcast"),
    # ascending reticular activating system (ARAS): pontine reticular formation -> thalamic arousal
    _e(
        "prf", "thal_relay", "ascending reticular activating (ARAS) -> thalamus", "exc", "broadcast"
    ),
    _e("prf", "bf_ach", "ARAS -> basal forebrain (cholinergic arousal)", "exc", "broadcast"),
    # superior-colliculus orienting loop (tecto-thalamo-cortical)
    _e("ctx_sensory", "sc", "corticotectal (visual salience)", "exc", "relay", "orienting"),
    _e("sc", "thal_relay", "tectothalamic (orienting relay)", "exc", "relay", "orienting"),
    _e("ctx_pfc", "sc", "frontal eye field -> colliculus (saccade command)", "exc", "descending"),
    # lateral-habenula reward-opponent loop (anti-reward suppresses the dopamine burst via RMTg)
    _e(
        "ctx_pfc",
        "lhb",
        "prefrontal -> lateral habenula (disappointment)",
        "exc",
        "relay",
        "reward_opponent",
    ),
    _e(
        "lhb",
        "vta_snc",
        "habenula -> RMTg -> VTA (dopamine suppression)",
        "inh",
        "feedback",
        "reward_opponent",
    ),
    # olfactory pathway (bulb decorrelation -> piriform association, with centrifugal feedback)
    _e("ob", "pir", "lateral olfactory tract (bulb -> piriform)", "exc", "relay", "olfactory"),
    _e("pir", "ob", "piriform -> bulb centrifugal feedback", "exc", "feedback", "olfactory"),
    _e("pir", "ctx_pfc", "piriform -> orbitofrontal (odor to cortex)", "exc", "relay"),
    _e("pir", "bla", "piriform -> amygdala (odor valence)", "exc", "relay"),
    # claustrum: cortical synchrony / salience broadcast hub
    _e(
        "ctx_sensory",
        "claustrum",
        "corticoclaustral (sensory)",
        "exc",
        "relay",
        "claustro_cortical",
    ),
    _e(
        "claustrum",
        "ctx_sensory",
        "claustrocortical (salience broadcast)",
        "exc",
        "broadcast",
        "claustro_cortical",
    ),
    _e("ctx_pfc", "claustrum", "corticoclaustral (prefrontal)", "exc", "relay"),
    _e("claustrum", "ctx_pfc", "claustrocortical broadcast (prefrontal)", "exc", "broadcast"),
    # --- medulla autonomic vital nuclei (v3.5): hypothalamic set-points descend to the medullary
    #     respiratory (pre-Botzinger) and cardiovascular (NTS/RVLM) rhythm generators, each closing a
    #     homeostatic reflex loop onto an autonomic effector (the viscera) -- never the arm actuator ---
    _e(
        "hypothal",
        "pbc",
        "hypothalamo-medullary drive (respiratory set-point)",
        "exc",
        "descending",
    ),
    _e(
        "hypothal",
        "nts_rvlm",
        "hypothalamo-medullary drive (cardiovascular set-point)",
        "exc",
        "descending",
    ),
    _e(
        "pbc",
        "viscera",
        "phrenic -> diaphragm (inspiratory drive)",
        "exc",
        "descending",
        "respiratory",
    ),
    _e(
        "viscera",
        "pbc",
        "pulmonary stretch / chemoreceptor afferent (via NTS)",
        "exc",
        "feedback",
        "respiratory",
    ),
    _e(
        "nts_rvlm",
        "viscera",
        "sympathetic / vagal cardiovascular efferent",
        "exc",
        "descending",
        "baroreflex",
    ),
    _e("viscera", "nts_rvlm", "baroreceptor afferent (via NTS)", "exc", "feedback", "baroreflex"),
    # --- superior-colliculus deep motor layers (v3.5): the orienting winner becomes a gaze command
    #     that reaches the effector ONLY through the gated brainstem final common path (no new route) ---
    _e("sc", "sc_deep", "superficial -> deep collicular layers (motor error)", "exc", "relay"),
    _e(
        "sc_deep",
        "bs_motor",
        "tecto-reticulospinal gaze command (through the gate)",
        "exc",
        "descending",
    ),
    # --- interoceptive arousal (v3.6): the medullary visceral state ascends (NTS -> LC) to drive
    #     arousal/caution, wired live into autonomous_loop as a caution-only bias (never a motor route) ---
    _e("nts_rvlm", "lc", "NTS -> locus coeruleus (interoceptive arousal drive)", "exc", "relay"),
    # --- prefrontal executive (v3.7): the "centre of thoughts" holds a goal in working memory and
    #     biases motor selection top-down toward it; wired live into autonomous_loop as a caution-safe
    #     preference that reaches the arm only through the gated final common path (never a bypass) ---
    _e(
        "ctx_pfc",
        "ctx_motor",
        "prefrontal top-down bias (goal -> motor selection)",
        "exc",
        "descending",
    ),
    # --- v3.36 completed-anatomy: the region->region edges the brain audit found genuinely missing.
    #     ALL are afferent / broadcast / modulatory -- none is an edge INTO the actuator, so the arm-safety
    #     gate invariant (actuator's sole in-edge is bs_motor; no node reaches it except through the gate) is
    #     preserved (verified by the self-check below). ---
    # hippocampo-cortical systems consolidation: SWR replay writes memories back to neocortex (the missing
    # ca1/sub -> cortex return that experience_replay / sleep_replay / sleep_consolidation implement)
    _e(
        "ca1",
        "ctx_pfc",
        "hippocampo-cortical (systems consolidation / SWR replay)",
        "exc",
        "broadcast",
    ),
    # basal-ganglia hyperdirect pathway: cortex -> STN fast global "stop" (bypasses the striatum)
    _e("ctx_pfc", "stn", "cortico-subthalamic (hyperdirect / fast stop)", "exc", "relay", "cbgtc"),
    # mesolimbic dopamine reward loop: VTA -> ventral striatum (NAcc) -> VTA (incentive salience)
    _e(
        "vta_snc",
        "nacc",
        "mesolimbic dopamine (reward / incentive salience)",
        "da",
        "broadcast",
        "mesolimbic",
    ),
    _e(
        "nacc",
        "vta_snc",
        "ventral-striatal -> VTA (GABAergic feedback)",
        "inh",
        "feedback",
        "mesolimbic",
    ),
    # amygdala LOW ROAD (LeDoux): thalamus -> BLA fast fear pathway, complementing the cortical high road
    _e("thal_relay", "bla", "thalamo-amygdala (low road, fast fear)", "exc", "relay"),
    # amygdala -> medullary autonomic: fear drives cardiovascular/respiratory arousal (onto viscera, NOT the arm)
    _e("cea", "nts_rvlm", "amygdala -> medulla (autonomic fear response)", "exc", "descending"),
    # amygdala -> hippocampus: emotional modulation of memory encoding
    _e("bla", "ca1", "amygdala -> hippocampus (emotional memory modulation)", "exc", "relay"),
    # hippocampus -> ventral striatum (subiculum -> nucleus accumbens): memory/context state
    # informs value-based selection. DECLARED 2026-08-12 when the promoted
    # hippocampus/developmental_loop_brain realized the route in code (its main path feeds
    # competence state into basal_ganglia/brain_decision's LatentQ every cycle) -- the p6
    # gate's undeclared-pair detector caught the missing declaration, which is the detector
    # working: the import existed, the connectome had not recorded it.
    _e(
        "sub",
        "nacc",
        "hippocampus -> ventral striatum (context informs value selection)",
        "exc",
        "relay",
    ),
    # hippocampus -> thalamic broadcast hub: hippocampus/developmental_loop_brain constructs and
    # drives an instance of the thalamic GWT hub CLASS (GlobalWorkspace) for its offline
    # developmental loop -- two lazy imports (main() / _mechanics_check()), the tree's only
    # hippocampus->thalamus reach. DECLARED 2026-08-12 when the WorkspaceBroadcast retirement made
    # dlb construct the bus directly; the p6 undeclared-pair detector fired, the same detector
    # working the same way as the sub->nacc row above (the import existed, the connectome had not
    # recorded it). HONEST SCOPE, corrected at the same day's review pass: this declares a REACH,
    # not live traffic -- the loop's bus is private and offline (p6's unread-writers census lists
    # it), it is NOT the Papez limb (that is the sub->hypothal fornix row + LOOPS["papez"]), and
    # wiring the loop's goal broadcast into THE shared hub remains open P6 work. Structural
    # consequence disclosed rather than implied: EDGES feeds thalamus/connectome_router, so this
    # row adds one exc/relay projection constructed per Brain -- measured on the live brain
    # (60 ticks, seed 0, with vs without the row) the per-tick (broadcast, hold) sequence is
    # IDENTICAL 60/60: the projection exists, nothing live consumes it yet.
    _e(
        "sub",
        "gwt",
        "hippocampus -> thalamic broadcast hub (goal posted into the global workspace)",
        "exc",
        "relay",
    ),
    # olivo-cerebellar climbing-fibre teaching loop: the error signal cerebellar learning is premised on
    _e(
        "io",
        "cb_ctx",
        "inferior olive climbing fibres (error / teaching signal)",
        "exc",
        "relay",
        "olivocerebellar",
    ),
    _e(
        "cb_deep",
        "io",
        "deep-nuclei -> inferior olive (GABAergic feedback)",
        "inh",
        "feedback",
        "olivocerebellar",
    ),
    # global-workspace (GNW) ignition hub: specialists compete for access, the winner is broadcast globally
    _e(
        "ctx_sensory",
        "gwt",
        "specialist -> global workspace (access competition)",
        "exc",
        "relay",
        "global_workspace",
    ),
    _e(
        "ctx_pfc",
        "gwt",
        "prefrontal -> global workspace (access competition)",
        "exc",
        "relay",
        "global_workspace",
    ),
    _e(
        "gwt",
        "ctx_pfc",
        "global workspace ignition broadcast (to prefrontal)",
        "exc",
        "broadcast",
        "global_workspace",
    ),
    _e(
        "gwt", "ctx_sensory", "global workspace ignition broadcast (to sensory)", "exc", "broadcast"
    ),
    # spinal reflex AFFERENT arc: sensory afferent into the gated brainstem/spinal motor node (the missing
    # ascending half of the reflex; it reaches the actuator ONLY through the bs_motor gate, never around it)
    _e(
        "ctx_sensory",
        "bs_motor",
        "dorsal-root sensory afferent (reflex arc, through the gate)",
        "exc",
        "relay",
    ),
]

# canonical loops as CLOSED node cycles (first repeated at the end)
LOOPS = {
    "thalamocortical": ["thal_relay", "ctx_sensory", "thal_relay"],
    "cbgtc": ["ctx_motor", "striatum", "gpi_snr", "thal_relay", "ctx_motor"],
    "hippocampal": ["ec", "dg", "ca3", "ca1", "sub", "ec"],
    "cortico_cerebellar": ["ctx_motor", "pn", "cb_ctx", "cb_deep", "thal_relay", "ctx_motor"],
    "papez": ["sub", "hypothal", "thal_relay", "ctx_pfc", "ec", "dg", "ca3", "ca1", "sub"],
    "orienting": ["sc", "thal_relay", "ctx_sensory", "sc"],
    "reward_opponent": ["vta_snc", "ctx_pfc", "lhb", "vta_snc"],
    "olfactory": ["ob", "pir", "ob"],
    "claustro_cortical": ["claustrum", "ctx_sensory", "claustrum"],
    # v3.5 medullary homeostatic reflex loops (autonomic negative feedback onto the viscera)
    "respiratory": ["pbc", "viscera", "pbc"],
    "baroreflex": ["nts_rvlm", "viscera", "nts_rvlm"],
    # v3.36 completed-anatomy loops
    "mesolimbic": ["vta_snc", "nacc", "vta_snc"],
    "olivocerebellar": ["io", "cb_ctx", "cb_deep", "io"],
    "global_workspace": ["gwt", "ctx_pfc", "gwt"],
}

ACTUATOR = "actuator"
GATE_NODE = "bs_motor"  # the brainstem/spinal gated node -- the sole gateway to the actuator


def _adjacency(block_node=None, block_edge=None):
    adj = {n: [] for n in NODES}
    for e in EDGES:
        if e["source"] == block_node or e["target"] == block_node:
            continue
        if block_edge is not None and (e["source"], e["target"]) == block_edge:
            continue
        adj[e["source"]].append(e["target"])
    return adj


def has_edge(source, target):
    return any(e["source"] == source and e["target"] == target for e in EDGES)


def reachable(source, target, block_node=None, block_edge=None):
    """BFS: can `target` be reached from `source` (optionally with a node/edge removed)?"""
    adj = _adjacency(block_node=block_node, block_edge=block_edge)
    seen, stack = set(), [source]
    while stack:
        n = stack.pop()
        if n == target:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))
    return False


def route(source, target):
    """Return a shortest node path source -> target over the connectome, or None."""
    adj = _adjacency()
    prev, queue, seen = {source: None}, [source], {source}
    while queue:
        n = queue.pop(0)
        if n == target:
            path = []
            while n is not None:
                path.append(n)
                n = prev[n]
            return list(reversed(path))
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                prev[m] = n
                queue.append(m)
    return None


def is_closed_loop(nodes):
    """True if every consecutive pair in `nodes` is a real directed edge (the loop closes)."""
    return (
        all(has_edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
        and nodes[0] == nodes[-1]
    )


def cortical_nodes():
    return [n for n, (region, _) in NODES.items() if region == "cortex"]


def _import_graph_is_untyped():
    """FAIR CONTROL: the graphify EXTRACTED import graph carries no sign/loop/tract typing, so the
    connectome is not redundant with Python imports. Returns (checked, untyped). checked=False if the
    graph file is absent/unreadable (the anchor is then skipped, not failed)."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "graphify-out", "graph.json")
    try:
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
    except (OSError, ValueError):
        return False, None
    edges = g.get("edges") or g.get("links") or []
    if not isinstance(edges, list) or not edges:
        return False, None
    typed_keys = {"sign", "loop", "tract"}
    untyped = not any(isinstance(e, dict) and (set(e) & typed_keys) for e in edges)
    return True, untyped


def main():
    n_edges, n_loops = len(EDGES), len(LOOPS)
    print("=" * 68)
    print("CONNECTOME -- mesoscale schematic wiring layer (canonical pathways)")
    print("=" * 68)
    print(
        f"nodes={len(NODES)} (across {len(set(r for r, _ in NODES.values() if r))} regions) "
        f"edges={n_edges} loops={n_loops}"
    )
    for name, nodes in LOOPS.items():
        print(f"  loop {name:<18} closed={is_closed_loop(nodes)}  ({' -> '.join(nodes)})")
    cs = route("ctx_motor", ACTUATOR)
    print(f"corticospinal route ctx_motor -> actuator: {' -> '.join(cs) if cs else None}")
    print("=" * 68)

    # ---- self-check: graph-topological facts ----
    # every edge is well-typed and connects real nodes
    for e in EDGES:
        assert e["source"] in NODES and e["target"] in NODES, f"edge on unknown node: {e}"
        assert e["sign"] in ("exc", "inh", "da", "ne", "5ht", "ach"), f"bad sign: {e}"
        assert e["cls"] in ("relay", "feedback", "broadcast", "descending"), f"bad class: {e}"

    # (a) every canonical loop is a CLOSED directed cycle (un-riggable topology)
    for name, nodes in LOOPS.items():
        assert is_closed_loop(nodes), f"canonical loop {name} must be a closed cycle"

    # (b) SAFETY: the actuator is reachable ONLY through the gated brainstem node/edge.
    assert reachable("ctx_motor", ACTUATOR), (
        "motor cortex must reach the actuator (via corticospinal)"
    )
    for c in cortical_nodes():
        # removing the brainstem gate node disconnects EVERY cortical node from the actuator
        assert not reachable(c, ACTUATOR, block_node=GATE_NODE), (
            f"{c} must not reach the actuator except through the gated brainstem node"
        )
    # the actuator's only in-edge is the gated final common path
    in_edges = [(e["source"], e["target"]) for e in EDGES if e["target"] == ACTUATOR]
    assert in_edges == [(GATE_NODE, ACTUATOR)], (
        f"actuator must have exactly one (gated) in-edge, got {in_edges}"
    )
    assert not reachable("ctx_motor", ACTUATOR, block_edge=(GATE_NODE, ACTUATOR)), (
        "removing the gated final motor edge must cut cortex off from the actuator"
    )

    # (b2) SAFETY: the SC deep gaze command and the ascending interoceptive drive can influence motor
    #      ONLY through the gate. The autonomic nuclei ascend to arousal (NTS->LC, v3.6), so -- like
    #      every other node -- removing the gate must cut them off from the actuator (no bypass route).
    assert reachable("sc_deep", ACTUATOR), (
        "SC deep gaze command must reach the actuator (via the gate)"
    )
    assert not reachable("sc_deep", ACTUATOR, block_node=GATE_NODE), (
        "SC deep must reach the arm ONLY through the gated brainstem node (no new route to the actuator)"
    )
    for auto in ("pbc", "nts_rvlm", "viscera"):
        assert not reachable(auto, ACTUATOR, block_node=GATE_NODE), (
            f"autonomic vital node {auto!r} must reach the arm only through the gated brainstem node (no bypass)"
        )

    # (c) FAIR CONTROL: imports lack the connectome's typing (not redundant with Python imports)
    checked, untyped = _import_graph_is_untyped()
    typed_here = all(isinstance(e.get("sign"), str) and "cls" in e for e in EDGES)
    assert typed_here, "connectome edges must carry sign + class typing"
    if checked:
        assert untyped, "import graph unexpectedly carries sign/loop typing (control invalid)"

    print(
        f"self-check OK: {n_loops} canonical loops all closed cycles; the actuator is reachable "
        f"ONLY through the gated brainstem edge (safety); {n_edges} edges typed sign+class+loop"
        + (
            " > untyped import graph (fair control)"
            if checked
            else " (import-graph control skipped)"
        )
    )


if __name__ == "__main__":
    main()
