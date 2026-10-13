"""G6 all-eligible-circuit STATIC census -- the adapter for owner answer #1038 (question #94).

THE OWNER DECISION THIS SERVES. #1038: "G6: include every eligible plastic circuit in the
eligible-brain arm, including eligible legacy circuits beside born units; preserve existing
safety/anatomy/validation/retention exclusions." That settles SCOPE. It grants no measurement:
this module establishes no sleep benefit, no memory-retention proof and no G9 contract.

WHAT THIS IS, AND WHAT IT IS NOT. The authoritative source map
(rows/g6-owner-adopted-circuit-mapping-2026-10-11.md, board #1061) enumerates the
populations it found and says its own table is NOT exhaustive, that it has "no complete
instance-level census and no evaluated counts", and that "all-circuit coverage is explicitly
UNRESOLVED / adapter UNBUILT at this source snapshot". #1080 adopted that map and told this
row to use it rather than repeat the census. So this module does NOT re-derive the map's
inventory. It adds the ONE axis the map explicitly disclaims and refuses to fake: a MECHANICAL
discovery pass over the driver classes, so "did we miss a population?" is answered by parsing
source instead of by re-reading a table, and every discovered handle must acquire an explicit
disposition before the census can call itself complete.

It defines no search, statistic, scorer, lake writer, identity allocator or serializer, and it
proposes none. Where a contract already exists (joint_mask_search.mask_digest, the
sleep_mask_apply prepare/write_prepared/write_commit/check_applied transaction, Meter,
run_instance, seed_row) a record NAMES the contract to reuse rather than restating it.

RUNTIME SCOPE, STATED ACCURATELY. The original incident -- a Brain-building test writing
through the LIVE lake via unscoped monkeypatch.undo() -- is no longer pending: the owner
APPROVED a guarded throwaway resume at #1084, the lead accepted its exact limits at #1086, and
#1226 closes the source-bound controls/sentinel hygiene. What remains is not a pending owner
question but an UNBUILT capability: this module delivers the STATIC census only. Runtime
effective eligible-brain mapping and admission, its separate source/control grant, and any
runtime slice beyond the static one all remain UNBUILT and require their own explicit grant.
That approval does NOT make this module a runtime port, and this module needs no runtime at
all: every input is source text. Nothing here imports the engine, constructs a Brain or an
AutonomousArm, instantiates any population, or opens a home.

P8 PLACEMENT. The self-checks live HERE, in src/, not only in tests/. The project's P8 gate
tools/red_reachability.py scans src/ for `_present` / `_verdict` / `probe_*` criteria and is
blind to tests/, so an oracle that lived only in tests/ would not enforce P8 here. Each
criterion below takes its inputs as FORMAL PARAMETERS and reads them, and each has a reachable
red by construction: mutate one record, one citation, or one source line and the verdict moves.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field

FROZEN_BASE = "6cca7a5be0c4fd9b3b7fcfb140cb87c6a50a0b17"

#: The three driver classes. The census's discovery scope is exactly these, and that scope is
#: a stated LIMITATION, not a claim of global exhaustiveness: populations owned by a class the
#: driver merely calls (e.g. NightlySkills' nested domain heads) are reached through their
#: container's CONDITIONAL record, and the runtime port that would witness them is UNRESOLVED.
DRIVERS: tuple[tuple[str, str], ...] = (
    ("Brain", "src/snn_data_lake/engine/brain.py"),
    ("LiveBrain", "src/snn_data_lake/engine/live_brain.py"),
    ("AutonomousArm", "src/snn_data_lake/engine/autonomous_loop.py"),
)

#: Callees that convert a value rather than construct an object. A site calling one of these
#: builds no population, so it is dispositioned NON_POPULATION BY RULE -- and the rule is
#: checked, so a scalar can never be silently indistinguishable from a dropped population.
CONVERSION_CALLEES = frozenset(
    {
        "abs",
        "bool",
        "bytes",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
    }
)

#: Dispositions, kept in the authoritative map's own vocabulary (its section 3 status legend)
#: so a reader can reconcile this census against #1061 line by line.
#:   REQUIRED_UNMAPPED -- source holds learned state the all-eligible arm must account for,
#:                        but no reversible mask adapter exists (the map's required/unmapped).
#:   CONDITIONAL       -- effective instance, eligibility or child closure must be declared
#:                        (the map's conditional/unresolved).
#:   UNRESOLVED        -- THE SOURCE DOES NOT DETERMINE the count, identity or boundary. A
#:                        correct deliverable here. Never a guessed number.
#:   NON_POPULATION    -- a value conversion, a pure reader, or a non-learned scaffold.
DISPOSITIONS = frozenset({"REQUIRED_UNMAPPED", "CONDITIONAL", "UNRESOLVED", "NON_POPULATION"})


@dataclass(frozen=True)
class Site:
    """One construction site for an object handle inside a driver class."""

    owner_class: str
    attribute: str
    callee: str
    defining_module: str
    callee_kind: str  # CLASS | FACTORY | CALL | CONVERSION | OPTIONAL
    file: str
    line: int
    method: str
    conditional: bool = False

    @property
    def key(self) -> str:
        return f"{self.owner_class}.{self.attribute}"

    @property
    def citation(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Record:
    """One declared population row.

    `sites` names the driver handles this row disposes of, so coverage is an explicit
    declaration rather than a token-matching guess. `identity` records the ID's REAL source in
    the tree; where the source mints none it says UNRESOLVED rather than naming an invented ID
    (the map's line 42: "Legacy names are not born IDs").
    """

    owner_path: str
    storage: str
    identity: str
    boundary: str
    disposition: str
    reason: str
    sites: tuple[str, ...]
    file: str
    line: int
    nested_in: str = ""
    lazy: bool = False
    reuse: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"{self.owner_path}: disposition {self.disposition!r} not in {sorted(DISPOSITIONS)}"
            )
        if not self.reason:
            raise ValueError(
                f"{self.owner_path}: a record without a machine-readable reason is not a disposition"
            )
        if not self.sites and not self.nested_in:
            raise ValueError(
                f"{self.owner_path}: a record must dispose of at least one driver handle, or name the container "
                "handle it is nested inside via nested_in -- otherwise it is not a census row"
            )


@dataclass(frozen=True)
class Excluded:
    """A driver handle that is NOT an eligible learned population, with its reason."""

    site: str
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(f"{self.site}: an exclusion without a reason is a silent drop")


# --------------------------------------------------------------------------------------
# THE CENSUS.
#
# Identity column, read before trusting any row. `NATIVE_BORN_ID` appears exactly once, on the
# G3 row, because GrownUnit.unit_id is the only real born identity in this source.
# `mask_digest` stringifies ids while G7 birth unit/batch identities require non-Boolean
# nonnegative integers (map line 42), so a legacy LIST_INDEX or LABEL_INDEX is NOT a born id
# and must never be dressed as one. `no_invented_born_identity` enforces that.
# --------------------------------------------------------------------------------------

RECORDS: tuple[Record, ...] = (
    # ---- the four named minimums (map section 3.1) ----
    Record(
        owner_path="Brain.brain._ixn",
        storage="neurons[label_index].w[input_index] over the immutable ordered responses "
        "list; one RSTDPNeuron per label. ixn.stack is the SAME object, not a second population.",
        identity="LABEL_INDEX + INPUT_BUCKET bound to the frozen _IXN_INTENTS tuple; "
        "n_in=52 is the original bucket code on the default LiveBrain",
        boundary="ixn.state_dict/load_state_dict (interaction_learning.py:658,:662), written "
        "by the Brain home writer as interaction_learner.json. _authority and teach_sink are "
        "deliberately excluded by _state_attrs; authority is rebuilt from its event source.",
        disposition="REQUIRED_UNMAPPED",
        reason="Zeroing w is not permanent pruning: stabilization and structural updates can "
        "reintroduce it, and deleting a row would change label/index semantics, which is not "
        "an existing supported adapter. R3's state_digest covers this continuation plus "
        "rewirer-none ONLY and is NOT all-circuit state.",
        sites=("LiveBrain._ixn",),
        file="src/snn_data_lake/engine/live_brain.py",
        line=160,
        lazy=True,
        reuse="joint_mask_search.mask_digest (the ONE prepared-mask digest); "
        "sleep_mask_apply.prepare/write_prepared/write_commit/check_applied_mask",
        notes=("constructed under `if interaction_learning else None`, so it is OPTIONAL",),
    ),
    Record(
        owner_path="Brain._plasticity",
        storage="the SAME SpikingInteractionLearner reached through brain._ixn. Aliased: one "
        "population under two handles.",
        identity="LABEL_INDEX + INPUT_BUCKET, identical to Brain.brain._ixn",
        boundary="the same interaction_learner.json writer; the alias means ONE declared owner",
        disposition="REQUIRED_UNMAPPED",
        reason="Counting both paths would double count one population. An edit must update the "
        "stack's matching row/slot state, traces, lazy cascades, meta/E-I/homeostatic state and "
        "the rewirer continuation together, and must verify the authority/journal boundary.",
        sites=("Brain._plasticity",),
        file="src/snn_data_lake/engine/brain.py",
        line=331,
        lazy=True,
        notes=("built through self._try(), so absence is possible and must be witnessed",),
    ),
    Record(
        owner_path="Brain.neuro._model",
        storage="NGSNN.W (L,d_in), .b (L,), .beta (L,m); ._placement aligned to survivors; "
        "_batch_beta is a SEPARATE last-batch solve",
        identity="ROW_INDEX over W rows, aligned to neuro._placement; follow the actual "
        "last_keep, never a post-prune row index",
        boundary="growth_state_bytes/restore_growth_state (ng_snn.py:608,:696) with their own "
        "format/shape/accuracy/placement checks. No state_dict on NGSNN. These serialize "
        "substrate, NOT full engine continuation: buffers, counters, online-rule state and RNG "
        "still need explicit ownership.",
        disposition="REQUIRED_UNMAPPED",
        reason="A unit bit must select the SAME rows of W and b, matching beta rows, placement "
        "entries, any batch-beta rows, category/class semantics and pending prediction/evidence. "
        "The staged census (staged_sleep_bout.py:118) reads _model.W only and misses b and beta.",
        sites=("Brain.neuro",),
        file="src/snn_data_lake/growth/ng_snn.py",
        line=267,
        notes=("reached through Brain.neuro = self._make_neuro()",),
    ),
    Record(
        owner_path="Brain.neuro.online_readout",
        storage="the retained beta rows the online delta readout writes between refits",
        identity="ROW_INDEX over the beta rows, following last_keep",
        boundary="the same growth_state_bytes/restore_growth_state boundary as _model",
        disposition="REQUIRED_UNMAPPED",
        reason="Refit beta on TEACH ONLY. Biases are parameters, not an invented extra "
        "neuron/synapse count.",
        sites=(),
        file="src/snn_data_lake/growth/ng_snn.py",
        line=378,
        nested_in="Brain.neuro",
    ),
    Record(
        owner_path="Brain.attractor",
        storage="AttractorNet.W, the actual N x N recurrent matrix; N=24 in this source, "
        "diagonal zero-initialised",
        identity="TWO_AXIS (postsynaptic row i, presynaptic column k) over neurons 0..N-1; "
        "the source exposes no scalar neuron count",
        boundary="clone() copies W for this class ONLY and is not an R3 twin-brain fork or a "
        "full persistence proof. There is no state_dict on this class.",
        disposition="REQUIRED_UNMAPPED",
        reason="Every method loops over the source N, so physically shortening W is not a "
        "supported mask, and W==0 is a different fact from stored allocation. Both axes, "
        "pattern indices, reset/cue/recall semantics and _normalize's incoming budget must stay "
        "consistent. DISTINCT from Brain._episode_consolidator.",
        sites=("Brain.attractor",),
        file="src/snn_data_lake/sleep/sleep_replay.py",
        line=152,
        notes=("owner answer #1039 retires attractor.sleep LEARNING only; store/recall survive",),
    ),
    # ---- further populations the map enumerates (sections 3.2 / 3.3) ----
    Record(
        owner_path="AutonomousArm.learner (brain.brain.arm.learner)",
        storage="GrowingLearner.prototypes; n_units = len(prototypes)",
        identity="LIST_INDEX of the current prototypes list -- a LEGACY index, not a born unit_id",
        boundary="state_dict/load_state_dict; Brain saves arm/vis/aud state_dict in "
        "PROTOTYPES_STATE (brain.py:5161-5170). The source loader REFUSES to overwrite a "
        "nonempty learner, so it is not an in-place candidate rollback API.",
        disposition="REQUIRED_UNMAPPED",
        reason="Removing/reordering prototypes also affects q keys and neuro category IDs, so "
        "labels and consumers must follow a registered identity map. Prototype coordinates must "
        "not be casually counted as axonal synapses without a declared source interpretation.",
        sites=("AutonomousArm.learner",),
        file="src/snn_data_lake/engine/autonomous_loop.py",
        line=279,
    ),
    Record(
        owner_path="Brain._vis_learner / Brain._aud_learner",
        storage="two MORE GrowingLearner instances, built by setattr inside a loop over "
        '(("vis","_vis_learner"),("aud","_aud_learner")) at brain.py:1062-1067',
        identity="LIST_INDEX, same legacy caveat as AutonomousArm.learner",
        boundary="PROTOTYPES_STATE alongside the arm learner; restored via load_state_dict and "
        "a refusal note (_prototypes_restore_note) when any leg fails",
        disposition="CONDITIONAL",
        reason="The attribute name is DATA-DEPENDENT (a setattr over a tuple), so the name is "
        "not statically decidable from the assignment itself. Both legs are initialised to None "
        "and may be absent; absence must be WITNESSED in the declared snapshot, never inferred "
        "from a failed constructor or a missing reader.",
        sites=("Brain.<dynamic>",),
        file="src/snn_data_lake/engine/brain.py",
        line=672,
        lazy=True,
        notes=(
            "reached only through the DATA_DEPENDENT discovery path: the census records the "
            "setattr installation, not a statically resolvable attribute name",
        ),
    ),
    Record(
        owner_path="Brain._ll.learner",
        storage="a GrowingLearner owned by LifelongConsolidator, beside its parallel "
        "labels/utility, memory and RNG",
        identity="LIST_INDEX. The container's labels/utility/RNG/consumer semantics are a "
        "SEPARATE identity domain the prototype index does not cover.",
        boundary="UNRESOLVED -- GrowingLearner's own source explicitly discloses that this "
        "instance is not in the Brain prototype checkpoint",
        disposition="UNRESOLVED",
        reason="The arm-only checkpoint is not an all-GrowingLearners checkpoint. Reusing the "
        "learner codec supplies the learner's bytes, not the owning container's persistence.",
        sites=("Brain._ll",),
        file="src/snn_data_lake/engine/brain.py",
        line=439,
    ),
    Record(
        owner_path="Brain._affect_int._som",
        storage="the SOM GrowingLearner inside AffectIntegrator; emotion-category prototypes",
        identity="LIST_INDEX over the SOM prototypes",
        boundary="UNRESOLVED -- same disclosure as Brain._ll.learner",
        disposition="UNRESOLVED",
        reason="The SOM is learned INDEPENDENTLY of AffectIntegrator's fixed neuromodulatory "
        "pools, so the container and its SOM must not be disposed as one object. Do not treat "
        "the arm-only checkpoint as covering all GrowingLearners.",
        sites=("Brain._affect_int",),
        file="src/snn_data_lake/engine/brain.py",
        line=1096,
    ),
    Record(
        owner_path="Brain._reason",
        storage="ReasoningLearner.codebook and .rank; .buffer is premise memory",
        identity="NATIVE_UID -- real integer UIDs minted by the learner's own _next_uid, with "
        "next_uid itself part of the state",
        boundary="state_dict/load_state_dict; Brain saves reasoning_curriculum.json with "
        "learner state, stage and accuracy",
        disposition="REQUIRED_UNMAPPED",
        reason="UIDs and next_uid must be preserved. Records and the premise buffer are "
        "continuation data, not mask elements. The structural cost and active-membership "
        "definition must be CHECKED, not inferred from the codec name.",
        sites=("Brain._reason",),
        file="src/snn_data_lake/faculties/reasoning_curriculum.py",
        line=220,
    ),
    Record(
        owner_path="LiveBrain._pc (brain.brain._pc)",
        storage="PredictiveCodingNetwork.w1 and .w2, changed by live learn_step after settling",
        identity="TWO_AXIS over dims (d_in,d_r1,d_r2); dims are the source, not a counted unit number",
        boundary="NO state_dict on this class; it exposes learn_step only. UNRESOLVED as an "
        "all-circuit provider.",
        disposition="REQUIRED_UNMAPPED",
        reason="Rows and columns couple both levels; reset carriers, topology and RNG belong in "
        "the snapshot. Count ONE owned matrix allocation rather than multiplying tied uses or "
        "transposed reads. Causes/error-cell/transient arrays are not additional copies.",
        sites=("LiveBrain._pc",),
        file="src/snn_data_lake/engine/live_brain.py",
        line=193,
        lazy=True,
        notes=("lazy-sized to the substrate width on the first predictive-coding tick",),
    ),
    Record(
        owner_path="Brain._percept_readout",
        storage="LaminarLocalReadout.W1/b1, W2/b2, W6/b6, W4/b4, fixed feedback B, _frozen and "
        "auxiliary state",
        identity="HIDDEN_UNIT_MAP coupling W1 rows to b1, W2 columns and B/W6 rows; the source "
        "exposes no scalar neuron identity for the hidden layer",
        boundary="UNRESOLVED -- no full checkpoint provider was established for this class",
        disposition="UNRESOLVED",
        reason="The existing grow path demonstrates these couplings but is APPEND-ONLY and has "
        "no live caller; it is not a shrink/restore adapter. Which other levels are eligible, "
        "fixed or protected must be declared rather than inherited by name.",
        sites=("Brain._percept_readout",),
        file="src/snn_data_lake/engine/brain.py",
        line=1730,
        lazy=True,
    ),
    Record(
        owner_path="Brain._percept_surprise",
        storage="PerceptSurprise._neurons, actual OjaNeuron instances with w vectors, plus the "
        "baseline _mu/_var",
        identity="NEURON_INDEX + INPUT_COORDINATE within the Oja population",
        boundary="UNRESOLVED -- no provider established in this census",
        disposition="UNRESOLVED",
        reason="The source's k/in_dim describe ALLOCATION, not a measured active mask. Learning "
        "and baseline state must both be carried.",
        sites=("Brain._percept_surprise",),
        file="src/snn_data_lake/engine/brain.py",
        line=1758,
        lazy=True,
    ),
    Record(
        owner_path="AutonomousArm.jepa (brain.brain.arm.jepa)",
        storage="JepaWorldModel encoder/predictor We/be/Wz/Wa/b plus target Wt/bt; optional "
        "reward head Wr/br and enc_hidden/tgt_hidden",
        identity="LATENT_UNIT_INDEX coupled across producer and consumer axes and their teacher "
        "counterparts; the source exposes no stable per-latent id",
        boundary="save_weights/load_weights exist but are NOT demonstrated full continuation of "
        "rng/_buf/_prev",
        disposition="REQUIRED_UNMAPPED",
        reason="Latent removal couples producer and consumer axes, teacher/target counterparts, "
        "actions and vocabulary, and any grown hidden blocks. Every child block must be "
        "enumerated; the optional reward and hidden heads must not silently disappear.",
        sites=("AutonomousArm.jepa",),
        file="src/snn_data_lake/engine/autonomous_loop.py",
        line=2034,
        lazy=True,
    ),
    Record(
        owner_path="Brain._affect_appraisal",
        storage="AffectiveAppraisal.w_pos / w_neg / w_aro, changed by Brain wake learn",
        identity="UNRESOLVED -- the source defines dim and three output channels but mints no "
        "durable neuron/synapse identity",
        boundary="UNRESOLVED -- no established provider",
        disposition="UNRESOLVED",
        reason="A vector bank is not automatically a neuron population. Both eligibility and a "
        "reversible masked consumer need declaration.",
        sites=("Brain._affect_appraisal",),
        file="src/snn_data_lake/engine/brain.py",
        line=987,
    ),
    Record(
        owner_path="Brain._gsm",
        storage="GenerativeSelfModel.w, learned by observe",
        identity="MATRIX_AXIS. A matrix coordinate is not proof of a globally admitted unit.",
        boundary="UNRESOLVED -- no established provider",
        disposition="UNRESOLVED",
        reason="Prediction context and RNG must travel with the matrix.",
        sites=("Brain._gsm",),
        file="src/snn_data_lake/engine/brain.py",
        line=989,
    ),
    Record(
        owner_path="Brain._cfsim",
        storage="CounterfactualSim.W[action][output][input] and .b[action][output], learned by "
        "Brain wake transition calls",
        identity="THREE_AXIS (action, output, input); the action index and its semantics must be preserved",
        boundary="UNRESOLVED -- topology, cost and checkpoint are all undeclared",
        disposition="UNRESOLVED",
        reason="All incident coordinates couple to the action index; a flattened mask would "
        "break action semantics.",
        sites=("Brain._cfsim",),
        file="src/snn_data_lake/engine/brain.py",
        line=1097,
    ),
    Record(
        owner_path="Brain._task_set",
        storage="TaskSetLayer._W, learned RSTDP _shifter.w, plus working-memory and "
        "running-confidence state",
        identity="UNRESOLVED at the row level: the _W write sites must be inspected before "
        "every _W entry is called plastic",
        boundary="UNRESOLVED",
        disposition="UNRESOLVED",
        reason="_W may be a FIXED seeded context policy while _shifter is the learned part. "
        "Calling every _W entry plastic would be a guess, so the census refuses to. Native "
        "set/action/input indices and continuation cannot collapse to one scalar counter.",
        sites=("Brain._task_set",),
        file="src/snn_data_lake/engine/brain.py",
        line=988,
    ),
    Record(
        owner_path="Brain._learned_attn",
        storage="LearnedAttention band weights, updated with _elig by real reward learning",
        identity="BAND_INDEX over the fixed input-band association -- a band is not a validated "
        "neuronal unit count",
        boundary="UNRESOLVED -- baseline/eligibility/RNG continuity is undeclared",
        disposition="CONDITIONAL",
        reason="The fixed band association, baseline, eligibility, RNG and band identities must "
        "all survive a mask.",
        sites=("Brain._learned_attn",),
        file="src/snn_data_lake/engine/brain.py",
        line=1123,
    ),
    Record(
        owner_path="Brain._learned_evc + learned policy scalars/tables",
        storage="LearnedEVC._value/_cost; arm.q[prototype][action]; learned intrinsic, planning "
        "and control gains; Brain._drive_w and the learned place/percept/recall/drive value tables",
        identity="UNRESOLVED at the unit grain: these are adaptive PARAMETERS, and a dict key "
        "or a value estimate is not a neuron",
        boundary="UNRESOLVED",
        disposition="CONDITIONAL",
        reason="Actual adaptive parameters must not be auto-disposed as non-learned scaffold. "
        "They must be preserved in continuation even if a reviewed structural domain later "
        "excludes non-neuronal parameters.",
        sites=("Brain._learned_evc",),
        file="src/snn_data_lake/engine/brain.py",
        line=1146,
    ),
    Record(
        owner_path="Brain._episode_consolidator",
        storage="EpisodeConsolidator.W plus optional wiring _mask, codes/patterns/keys/episodes "
        "and the encoding cache",
        identity="CELL_INDEX over 0..dim-1. This is a DISTINCT population from Brain.attractor.",
        boundary="UNRESOLVED -- store and the distinct replay job both change W and neither is "
        "covered by the R3 digest",
        disposition="REQUIRED_UNMAPPED",
        reason="Pruning must preserve episode codes, retained content and wiring. Its episodic "
        "replay is NOT the retired interaction replay cap.",
        sites=("Brain._episode_consolidator",),
        file="src/snn_data_lake/engine/brain.py",
        line=1167,
    ),
    Record(
        owner_path="Brain._time_clock",
        storage="TimeCellClock.thresholds/cells, interval learning and growth, firing state/RNG",
        identity="CELL_INDEX exists in the source, but no prune or inverse-growth contract is "
        "declared for it",
        boundary="the source-owned save/restore for this class; UNRESOLVED as an all-circuit provider",
        disposition="CONDITIONAL",
        reason="Native indices exist but the mask contract does not. Learned temporal identity "
        "must be preserved.",
        sites=("Brain._time_clock",),
        file="src/snn_data_lake/engine/brain.py",
        line=986,
    ),
    Record(
        owner_path="Brain._acq_lm",
        storage="LanguageAcquisition.W updated by local next-character learning; an internal "
        "PredictiveCodingNetwork settles context, with vocabulary/context/counters/RNG",
        identity="CHARACTER_OUTPUT_ID over the fixed output vocabulary, plus context shape",
        boundary="existing save/load and the Brain language_acq.npz owner",
        disposition="CONDITIONAL",
        reason="Lazy: present only when acquired. The inner PC weights must NOT be assumed "
        "trained merely because the outer W is -- the source learning path must decide each "
        "bank. Per-head cost and complete continuation remain missing.",
        sites=("Brain._acq_lm",),
        file="src/snn_data_lake/engine/brain.py",
        line=4968,
        lazy=True,
        notes=("constructed through the load_lm FACTORY, and it also assigns None on failure",),
    ),
    Record(
        owner_path="Brain._skills",
        storage="NightlySkills language/math/physics/world-KB domain heads, SLDA sufficient "
        "statistics and DistMult parameters, plus meta-controller/router/discovery/autotelic state",
        identity="UNRESOLVED -- a RECURSIVE closure: every instantiated domain and nested learner "
        "needs its own eligibility/ID/shape/restore declaration",
        boundary="NightlySkills has its own durable loading logic, called from _enter_wake; the "
        "result is stashed on self._skills_state via .night(...)",
        disposition="CONDITIONAL",
        reason="A persistent lazy provider with a recursive child closure. This is not one neuron "
        "and not merely a stored record. Existing training/data/gradient policy is neither "
        "altered nor approved by this census.",
        sites=("Brain._skills",),
        file="src/snn_data_lake/engine/brain.py",
        line=3770,
        lazy=True,
    ),
    Record(
        owner_path="AutonomousArm.amyg / AutonomousArm._valence / AutonomousArm._gain",
        storage="Amygdala.w_exc/w_inh keyed by US channel; ValenceLearner lazily built in the "
        "dopamine path; GainAdapter.g",
        identity="UNRESOLVED at the anatomy/safety seam: keyed by US channel, not by a neuron id",
        boundary="UNRESOLVED",
        disposition="CONDITIONAL",
        reason="Learned state is not excluded solely for sitting under arm, but the reflex / "
        "actuator / E-STOP paths and the neuromodulation admission exclusions remain, and "
        "changing a protected safety seam is forbidden.",
        sites=("AutonomousArm.amyg", "AutonomousArm._valence", "AutonomousArm._gain"),
        file="src/snn_data_lake/engine/autonomous_loop.py",
        line=1321,
        lazy=True,
    ),
    Record(
        owner_path="AutonomousArm.latent_q / AutonomousArm.mlp_q / AutonomousArm.curiosity / "
        "AutonomousArm._allo / AutonomousArm._affmem / AutonomousArm._contrast",
        storage="per-context curiosity pools and EMAs, allocentric map cells, affective memory, "
        "contrastive episode analysis, latent/MLP Q heads",
        identity="UNRESOLVED per handle; a context key or pool index is not a neuron identity",
        boundary="per handle; latent_q and mlp_q exist only when configured",
        disposition="CONDITIONAL",
        reason="Actual handles requiring a caller/child audit that separates learned parameters "
        "from fixed state. SpikingCuriosityDrive has FIXED membrane parameters beside evolving "
        "EMA/context state, so 'spiking' alone does not make every field a plastic synapse. "
        "Configuration-dependent absence must be witnessed, not assumed.",
        sites=(
            "AutonomousArm.latent_q",
            "AutonomousArm.mlp_q",
            "AutonomousArm.curiosity",
            "AutonomousArm._allo",
            "AutonomousArm._affmem",
            "AutonomousArm._contrast",
        ),
        file="src/snn_data_lake/engine/autonomous_loop.py",
        line=2050,
        lazy=True,
        notes=(
            "curiosity has TWO alternative construction sites (spiking and intrinsic), so "
            "one handle maps to one population only after the configuration is witnessed",
        ),
    ),
    Record(
        owner_path="G3 born units and projections",
        storage="GrownUnit fields w_in, threshold, window, sign, out, active; ledger birth rows "
        "preserving problem/batch/label identity and original weights",
        identity="NATIVE_BORN_ID -- GrownUnit.unit_id, a non-reused integer. This is the ONLY "
        "class in the census whose identity is a real born ID rather than a legacy index.",
        boundary="reuse grown_code admission/journal helpers once the actual owner port exists",
        disposition="UNRESOLVED",
        reason="NO Brain owner exists at this base: GrowthLoop states that no conforming "
        "taught-learner adapter exists in src yet, and SpikingInteractionLearner has no born "
        "segment port here. brain._born, a return matrix and a G3-to-R3 attachment must NOT be "
        "invented, and no dormant fixture object proves attachment to this Brain.",
        sites=(),
        file="src/snn_data_lake/growth/grown_code.py",
        line=40,
        nested_in="NO Brain owner exists at this base, so this row has no container handle to "
        "name; that absence is the record's own finding",
    ),
    Record(
        owner_path="any attached G3/G4/G5 readout, pool or projection object",
        storage="KernelLsmPool fixed compiled topology with live v/s/acc carriers plus every "
        "KernelState array, tick/rings and policy metadata; trained G5 DELTA/PC/SOM adaptors",
        identity="UNRESOLVED -- depends on the actual producer/consumer instance",
        boundary="reuse the ACTUAL fingerprint/carrier policy at its declared valid boundary; its "
        "fingerprint rejects in-flight bundles",
        disposition="CONDITIONAL",
        reason="Full source ownership and attachment depend on the actual producer/consumer, not "
        "a guessed Brain attribute. The default R3 learner digest provides no coverage of these "
        "objects. Do not hash only s or pretend a fixed reservoir state is an eligible learned "
        "weight.",
        sites=(),
        file="src/snn_data_lake/reservoir/recurrent_pool.py",
        line=527,
        nested_in="NO driver handle exists at this base; attachment is the unresolved question "
        "this row records, and an old fixture object is not proof of attachment",
    ),
    # ---- map section 3.3: real handles that still need a final explicit disposition ----
    Record(
        owner_path="Brain association/prior and integrative handles: _second_order, _metacog, "
        "_novelty, _reentrant, _grid_place, _grid_can, _xmodal, _recursive_tom, "
        "_structure_mapper, _critic, _selfbuild_proposer, _episodic, _affect_core, "
        "_connectome_router, _region_coherence, _synfire, _access",
        storage="association/prior state; CrossModalBinder._mem per-modality bundles; grid cells; "
        "nested agent/world state; episodic lookup stores; calibration baselines",
        identity="UNRESOLVED per handle -- some own fixed dynamics, some adaptive "
        "representations, some only records",
        boundary="per handle; no all-circuit provider is asserted here",
        disposition="CONDITIONAL",
        reason="These are ACTUAL handles requiring a final explicit disposition, NOT a blanket "
        "list of eligible neurons. EpisodicRecall uses fixed role/filler codes and append-only "
        "episode/row stores: stored experience is never converted into disposable mask neurons. "
        "No uninspected object may be silently omitted, assigned a zero, or relabelled scaffold.",
        sites=(
            "Brain._second_order",
            "Brain._metacog",
            "Brain._novelty",
            "Brain._reentrant",
            "Brain._grid_place",
            "Brain._grid_can",
            "Brain._xmodal",
            "Brain._recursive_tom",
            "Brain._structure_mapper",
            "Brain._critic",
            "Brain._selfbuild_proposer",
            "Brain._episodic",
            "Brain._affect_core",
            "Brain._connectome_router",
            "Brain._region_coherence",
            "Brain._synfire",
            "Brain._access",
        ),
        file="src/snn_data_lake/engine/brain.py",
        line=397,
    ),
    Record(
        owner_path="AutonomousArm.workspace / .bg / .intero / .pfc and LiveBrain._planner",
        storage="GlobalWorkspace conscious access, BasalGangliaGate selection, "
        "InteroceptiveArousal, PrefrontalExecutive competence planning -- all built under an "
        "`if/else` with a None fallback, so every one is CONDITIONAL on construction",
        identity="UNRESOLVED -- none of these exposes a native unit identity in this census",
        boundary="UNRESOLVED per handle",
        disposition="CONDITIONAL",
        reason="Found by discovery, not remembered: a direct `self.X = Call()` scan misses "
        "these five because their RHS is a conditional expression, so the naive census would "
        "have silently under-reported exactly the map section 3.3 handles. LiveBrain._planner "
        "and AutonomousArm.pfc construct the SAME PrefrontalExecutive class in two places, so "
        "one declared owner must be chosen or the two must be shown non-aliased.",
        sites=(
            "AutonomousArm.workspace",
            "AutonomousArm.bg",
            "AutonomousArm.intero",
            "AutonomousArm.pfc",
            "LiveBrain._planner",
        ),
        file="src/snn_data_lake/engine/autonomous_loop.py",
        line=363,
        lazy=True,
    ),
    Record(
        owner_path="LiveBrain._planner and LiveBrain._nl -- NOT in the authoritative map's tables",
        storage="PrefrontalExecutive competence planning; NLInterface natural-language routing",
        identity="UNRESOLVED -- neither handle appears in map sections 3.1, 3.2 or 3.3",
        boundary="UNRESOLVED",
        disposition="CONDITIONAL",
        reason="POSTED FINDING, not a silent addition: this census's mechanical discovery found "
        "two CLASS construction sites in LiveBrain that the authoritative map does not list. "
        "They are declared CONDITIONAL and flagged rather than quietly folded into a known "
        "bucket, because the map is adopted as authoritative and a disagreement between it and "
        "a source-derived reading is a finding to post, not a difference to paper over.",
        sites=("LiveBrain._nl",),
        file="src/snn_data_lake/engine/live_brain.py",
        line=150,
        lazy=True,
    ),
    Record(
        owner_path="Brain._rose -- NOT in the authoritative map's tables",
        storage="RoseLanguage, built in _rose_serve and set to None when unavailable",
        identity="UNRESOLVED -- absent from map sections 3.1, 3.2 and 3.3",
        boundary="UNRESOLVED",
        disposition="CONDITIONAL",
        reason="POSTED FINDING: a third CLASS construction site the map does not list, found by "
        "discovery rather than remembered. Declared and flagged, not folded away.",
        sites=("Brain._rose",),
        file="src/snn_data_lake/engine/brain.py",
        line=4993,
        lazy=True,
    ),
)

#: Handles that are NOT eligible learned populations, each with its reason. This is the rule
#: half of the coverage obligation: a scalar conversion, a pure reader, a transport ring, or
#: a protected safety seam must be DISPOSED here, never dropped. Section 4 of the map is the
#: authority for the always-excluded set (the five brainstem envelope files, real servo /
#: actuator / DOF / joint-limit / E-STOP / reflex pathways, and protected non-learned scaffolds).
EXCLUDED: tuple[Excluded, ...] = (
    Excluded(
        "AutonomousArm.arm",
        "So101Arm is the SAFETY ENVELOPE itself. Map section 4 "
        "always excludes real servo/actuator/DOF/joint-limit/E-STOP/reflex pathways.",
    ),
    Excluded(
        "AutonomousArm.efferent",
        "EfferentBoundary is the actuator boundary; a protected safety seam, never a mask domain.",
    ),
    Excluded(
        "AutonomousArm._rgate",
        "ReticularGate is fixed gating scaffold, not a learned "
        "population; it sits outside the G6 structural mask domain by name only, so the "
        "actual region/seam declaration must be resolved or refused.",
    ),
    Excluded(
        "AutonomousArm._dnorm",
        "DivisiveNormalization is a fixed normalisation, not learned weights.",
    ),
    Excluded(
        "AutonomousArm._claustrum",
        "Claustrum is region scaffold; the G6 exclusion pins "
        "claustrum to SCAFFOLD and forbids inferring a legacy object's region from its "
        "module filename.",
    ),
    Excluded(
        "AutonomousArm.cort",
        "Cortisol is a neuromodulatory controller scalar, not a "
        "plastic population; neuromodulation admission fails closed.",
    ),
    Excluded("AutonomousArm.cpg", "MatsuokaCPG is a fixed oscillator with no learned weights."),
    Excluded(
        "AutonomousArm._coalition_ring",
        "SignalRing is an in-tick signal transport, not learned storage.",
    ),
    Excluded(
        "AutonomousArm._coalition_journal",
        "LakeJournaller is a writer. The census "
        "deliberately owns no lake writer; this is map section 4's 'not mask elements' "
        "class -- still brain state that must survive and reconcile.",
    ),
    Excluded(
        "AutonomousArm._vitals",
        "StreamSketchPanel is a telemetry sketch, not learned state the eligible arm mutates.",
    ),
    Excluded(
        "AutonomousArm._dev",
        "DevelopmentalEnvelope gates the developmental schedule; it "
        "is an envelope rather than a learned population and changing it is out of scope.",
    ),
    Excluded(
        "AutonomousArm._agency",
        "AgencyMonitor is a monitor with its own separately "
        "recorded P3 charge; recorded here as a non-population handle, not as an "
        "admission.",
    ),
    Excluded(
        "AutonomousArm._dopa",
        "DopamineNuclei is neuromodulatory gating; admission fails "
        "closed at the neuromodulation boundary.",
    ),
    Excluded(
        "AutonomousArm._lhb",
        "LateralHabenula is neuromodulatory gating; same closed admission as the dopamine nuclei.",
    ),
    Excluded(
        "AutonomousArm._naca",
        "NACABroadcast is a neuromodulatory broadcast rule, not a mask-eligible weight bank.",
    ),
    Excluded(
        "AutonomousArm.empower",
        "Empowerment is a scalar information-gain drive with no durable unit identity.",
    ),
    Excluded(
        "AutonomousArm._exec",
        "ExecutiveControl is the background selection controller; "
        "it is a CONTROL path, not learned storage to be masked.",
    ),
    Excluded(
        "AutonomousArm.lake_wm",
        "LakeWorkingMemory is a lake-backed store. Map section 4: "
        "stored episodic content is never converted into disposable mask neurons.",
    ),
    Excluded(
        "AutonomousArm.buffer",
        "A ReplayBuffer storing experience, the arm-side analogue "
        "of Brain.buffer. Map section 4: stored episodic content is not a mask element, "
        "though it is still brain state that must survive and reconcile.",
    ),
    Excluded(
        "Brain.brain",
        "LiveBrain is the CONTAINER whose children carry state, not a "
        "population. It is the parent under which brain._ixn and arm.* paths are declared.",
    ),
    Excluded(
        "LiveBrain.arm",
        "AutonomousArm is the arm CONTAINER under which the arm-side "
        "population paths are declared.",
    ),
    Excluded(
        "Brain.self_model",
        "SelfModel reports a self state; no learned weight bank is constructed at this site.",
    ),
    Excluded("Brain.narrator", "SelfNarrator produces narration text; not learned storage."),
    Excluded("Brain._reason_env", "ReasoningEnv is the curriculum ENVIRONMENT, not a population."),
    Excluded("Brain.clock", "CircadianClock is a fixed schedule clock, not learned neurons."),
    Excluded(
        "Brain._drives",
        "HomeostaticDrives is a drive set; its fields are adaptive "
        "parameters covered by the learned-policy-scalars row, not a separate population.",
    ),
    Excluded("Brain._insula", "InteroceptiveIntegration is a fixed interoceptive integrator."),
    Excluded("Brain._thal_relay", "InteroceptiveRelay is a fixed relay."),
    Excluded(
        "Brain.buffer",
        "ReplayBuffer stores experience. Map section 4: immutable "
        "teaching/validation/future/retention records and stored episodic content are not "
        "mask elements, though they are still brain state that must reconcile.",
    ),
    Excluded(
        "Brain._salience_calib",
        "SignalCalibrator is a calibration baseline, not a "
        "population; it is map section 3.3's 'calibration baselines'.",
    ),
    Excluded("Brain._recall_conf_calib", "SignalCalibrator, same as _salience_calib."),
    Excluded("Brain._append_budget", "AppendBudget is a flood budget, not learned state."),
    Excluded(
        "Brain._growth_stop",
        "GrowthStop is the growth gate's stop decision value, not a learned population.",
    ),
    Excluded("Brain._lp", "LearningProgress is a scalar reading."),
    Excluded(
        "Brain._lake_recall",
        "LakeRecall is a lake reader. The census owns no lake reader and no lake writer.",
    ),
    Excluded(
        "Brain._rewirer",
        "StructuralRewiring is the separately adjudicated structural "
        "mutator; its own budget and admission are governed by its own row, not by this "
        "census, and it is not a mask-eligible weight bank.",
    ),
    Excluded(
        "Brain._perception_enc",
        "A loaded frozen encoder. Map section 4: loaded original "
        "encoders are frozen serving banks, not automatically mutable G6 mask domains.",
    ),
    Excluded("Brain._audio_enc", "A loaded frozen audio encoder; same frozen-serving-bank rule."),
    Excluded(
        "Brain._regions_live",
        "A set of live region NAMES, read from _regions_for; a "
        "provenance record, not learned storage.",
    ),
    Excluded(
        "Brain._ll_tasks", "A task list built by _make_tasks; schedule data, not learned state."
    ),
    Excluded("Brain._schedule", "A light-cycle schedule from light_cycle; fixed, not learned."),
    Excluded(
        "Brain._prev_append_observer",
        "An observer reference set by set_append_observer; writer plumbing, not learned state.",
    ),
    Excluded(
        "Brain._spont_bootstrap_steps",
        "A step budget from bootstrap_percept_rule; a scalar reading, not learned storage.",
    ),
    # ---- method-call handles that READ from an already-declared population ----
    Excluded(
        "Brain._baseline",
        "Reads the circadian clock's baseline from a declared "
        "non-learned handle; a reading, not a construction.",
    ),
    Excluded("Brain._was_asleep", "Reads the consolidation gate; a phase reading."),
    Excluded(
        "Brain._drive_w",
        "A dict built by dict.fromkeys over fixed drive names; an "
        "adaptive-parameter table already covered by the learned-policy-scalars row.",
    ),
    Excluded("Brain._dominant_drive", "Reads the active drive name; a reading."),
    Excluded(
        "Brain._real_percept", "The result of _perceive_real; a percept reading, not a population."
    ),
    Excluded("Brain._audio_percept", "The result of _perceive_audio; a percept reading."),
    Excluded("Brain._place_here", "Reads place_at() from the declared _grid_place handle."),
    Excluded("Brain._task_film", "Reads film() from the declared _task_set handle."),
    Excluded("Brain._emergent_emotion", "Reads label() from the declared _affect_int handle."),
    Excluded("Brain._access_state", "Reads state() from the declared _access handle."),
    Excluded("Brain._access_hold", "Reads get() from the declared _access handle."),
    Excluded("Brain._emotion", "Reads recognize() from the declared _affect_core handle."),
    Excluded("Brain._sleep_replayed", "Reads replay() from the declared _plasticity handle."),
    Excluded(
        "Brain._sleep_consolidate_calls",
        "Reads consolidation_calls() from the declared _plasticity handle.",
    ),
    Excluded("Brain._structural_report", "Reads report() from the declared _rewirer handle."),
    Excluded(
        "Brain._shy",
        "Reads downscale_prune() from the declared attractor handle; the "
        "value is a structural-prune result, not a weight bank.",
    ),
    Excluded("Brain._consolidation_acc", "Reads accuracy() from the declared attractor handle."),
    Excluded("Brain._skills_state", "Reads night() from the declared _skills handle."),
    Excluded(
        "Brain._patterns",
        "A frozen set built by a comprehension; pattern indices, not a population.",
    ),
    Excluded("Brain._rungs", "A dict built by a module-level census() helper; a rung report."),
    Excluded("Brain._xmodal_handle", "Reads get() from the declared _xmodal handle."),
    Excluded(
        "Brain._prototypes_restore_note", "A f-string over the restore-ok dict; a provenance note."
    ),
    Excluded("Brain._time_cells_grow", "Reads get() from a settings dict; a flag."),
    Excluded("Brain._time_cells_restore", "Reads get() from a settings dict; a flag."),
    Excluded("Brain._persist", "Reads get() from environment; a configuration flag."),
    Excluded(
        "Brain._r2_lock",
        "2026-10-13 Round 4: RLock is a synchronization primitive, not a neuron population; "
        "NON_POPULATION, no learned state or mask domain.",
    ),
    Excluded(
        "Brain._r2_prefix_digest",
        "2026-10-13 Round 5 (#2112): canonical-prefix SHA256 hex string; "
        "NON_POPULATION checkpoint integrity bookkeeping, no neurons or learned mask domain.",
    ),
    Excluded(
        "Brain._pc_cortex",
        "Loads the PC-cortex SERVING artifact. Map section 4: PC-cortex "
        "serving artifacts and fixed input code are frozen serving banks, not "
        "automatically mutable G6 mask domains.",
    ),
    Excluded(
        "Brain._ach / Brain._lc / Brain._raphe / Brain._astro / Brain._homeostat",
        "Built through self._try(), which RETURNS None when the optional module is absent. "
        "These are neuromodulatory controllers whose admission fails closed; the "
        "constructing helper is named in METHOD_CALLS_THAT_CONSTRUCT and each handle is "
        "disposed here rather than assumed present.",
    ),
    Excluded(
        "AutonomousArm.rng / AutonomousArm._dev_rng",
        "A random.Random instance: the "
        "census owns no RNG stream and proposes none; RNG continuity belongs to a "
        "snapshot producer.",
    ),
    Excluded(
        "AutonomousArm._estop_self_clear",
        "An E-STOP field. Never a mask domain, never read for a census, and never cleared here.",
    ),
    Excluded("AutonomousArm._last_disp", "A scalar distance readout."),
    Excluded("AutonomousArm._last_intero", "Reads sense() from the declared interoceptive handle."),
    Excluded(
        "AutonomousArm._sal_ema / AutonomousArm._sal_base / AutonomousArm._percept_sal",
        "Salience EMA / baseline readings; adaptive scalars already covered by the "
        "learned-policy-scalars row, not a separate population.",
    ),
    Excluded(
        "AutonomousArm._z_goal / AutonomousArm._z1_goal",
        "A zip() result building a goal tuple; a reading.",
    ),
    Excluded("AutonomousArm._allo_nearest", "Reads nearest() from the declared _allo handle."),
    Excluded(
        "LiveBrain._lakechat_close",
        "A close() result for the lake chat session; writer "
        "plumbing, and the census owns no writer.",
    ),
)

#: Method-call handles. A CALL-kind site (`self.X = obj.method(...)`) MIGHT construct an
#: object or might only read one, and a static callee attribute cannot tell them apart -- so
#: the census does not guess. It names the helpers it believes CONSTRUCT (each of which is
#: declared as its own row below) and lists the rest as READS FROM A DECLARED HANDLE, with
#: the rule stated here. A NEW method-call attribute that is in neither set becomes an
#: undisposed site, which is the exhaustiveness red: this list cannot silently fall behind
#: the source the way an unstated assumption would.
METHOD_CALLS_THAT_CONSTRUCT: frozenset[str] = frozenset({"_try", "_make_neuro", "load"})
METHOD_CALLS_THAT_READ: frozenset[str] = frozenset(
    {
        "accuracy",
        "baseline",
        "close",
        "consolidation_calls",
        "consolidation_gate",
        "dominant_drive",
        "downscale_prune",
        "film",
        "fromkeys",
        "get",
        "hypot",
        "items",
        "label",
        "nearest",
        "next_baseline",
        "place_at",
        "recognition",
        "recognize",
        "replay",
        "report",
        "sense",
        "state",
        "zip",
        "_perceive_audio",
        "_perceive_real",
        "_percept_salience",
        "_regions_for",
        "frozenset",
        "census",
    }
)

#: The four owner-named minimum populations. A census that drops one of these has not
#: answered #1038 even if it is internally consistent.
NAMED_MINIMUMS: tuple[str, ...] = (
    "Brain.brain._ixn",
    "Brain._plasticity",
    "Brain.neuro._model",
    "Brain.attractor",
)


# --------------------------------------------------------------------------------------
# Static discovery. This is the axis the authoritative map disclaims.
# --------------------------------------------------------------------------------------


def _read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        return fh.read()


def _imports(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                out[alias.asname or alias.name] = f"{mod}:{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = alias.name
    return out


def _rel_module(dotted: str) -> str:
    prefix = "snn_data_lake."
    return dotted[len(prefix) :] if dotted.startswith(prefix) else dotted


def _callee_kind(
    callee: str,
    prov: str | None,
    local: set[str],
    root: str,
    symbol_cache: dict[str, tuple[set, set]],
) -> tuple[str, str]:
    """Resolve a callee to CLASS / FACTORY / CALL / CONVERSION / UNRESOLVED and its module.

    The kind is read out of the DEFINING module's own AST, so "is this an object handle or a
    helper call?" is a source fact and not a guess.
    """
    if callee in CONVERSION_CALLEES:
        return "CONVERSION", "-"
    if prov is None:
        return "CALL", "-"
    mod, _, name = prov.rpartition(":")
    mod = _rel_module(mod)
    if mod in ("", "LOCAL"):
        return ("CLASS", "LOCAL") if callee in local else ("UNRESOLVED", "-")
    if mod not in symbol_cache:
        path = os.path.join(root, "src", "snn_data_lake", *mod.split(".")) + ".py"
        classes: set[str] = set()
        facts: set[str] = set()
        if os.path.isfile(path):
            for node in ast.parse(open(path, encoding="utf-8").read()).body:
                if isinstance(node, ast.ClassDef):
                    classes.add(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    facts.add(node.name)
        symbol_cache[mod] = (classes, facts)
    classes, facts = symbol_cache[mod]
    if callee in classes:
        return "CLASS", mod
    if callee in facts:
        return "FACTORY", mod
    return "UNRESOLVED", mod


def _construction_call(node: ast.AST) -> ast.Call | None:
    """The call that CONSTRUCTS the assigned handle, which is the OUTERMOST one.

    Taking the first `Call` found by walking would descend into a nested argument and report
    `self._sal_ema = list(zip(...))` as a `zip` construction. Descending is only correct for
    the conditional form `self._ixn = C(...) if flag else None`, where the constructor sits in
    a branch rather than at the root -- that is how live_brain.py:160 builds the interaction
    learner, and without this the population would be invisible.
    """
    if isinstance(node, ast.Call):
        return node
    if isinstance(node, ast.IfExp):
        for branch in (node.body, node.orelse):
            if isinstance(branch, ast.Call):
                return branch
            found = _construction_call(branch)
            if found is not None:
                return found
        return None
    return None


def _has_call(node: ast.AST) -> bool:
    """True when a call appears SOMEWHERE in the RHS, at any depth.

    A comprehension RHS (`self._patterns = [f(x) for x in ...]`) calls a function but
    constructs no named object, so it is EXPRESSION rather than a construction site -- and
    saying so explicitly is what stops a comprehension's inner call from being reported as an
    undisposed population.
    """
    return any(isinstance(sub, ast.Call) for sub in ast.walk(node))


def discover_sites(root: str) -> tuple[Site, ...]:
    """Every construction site for an object handle in the three driver classes.

    Three shapes are recognised, and each was found by a real omission first:
      * `self.X = Name(...)`           -- the direct form
      * `self.X = C(...) if c else N` -- reached through the conditional (LiveBrain._ixn)
      * `setattr(self, name, C(...))`  -- reached through a data-dependent name (the vis/aud
                                          GrowingLearners, brain.py:1062-1067)
    Value-conversion sites are returned too, marked CONVERSION, so a scalar can never be
    silently indistinguishable from a dropped population.
    """
    cache: dict[str, tuple[set, set]] = {}
    found: list[Site] = []
    for owner_class, rel in DRIVERS:
        tree = ast.parse(_read(root, rel))
        imports = _imports(tree)
        local = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == owner_class
        )
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(method):
                targets: list[tuple[str, ast.AST, bool]] = []
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if (
                            isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "self"
                        ):
                            targets.append((tgt.attr, sub.value, False))
                elif (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "setattr"
                    and sub.args
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id == "self"
                    and len(sub.args) >= 3
                ):
                    name_node = sub.args[1]
                    name = (
                        name_node.value
                        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)
                        else "<dynamic>"
                    )
                    targets.append((name, sub.args[2], True))
                for attr, value, is_setattr in targets:
                    if attr == "<dynamic>":
                        # An object is installed onto self under a name the source does not
                        # decide statically (brain.py:1062-1067 installs the vis/aud
                        # GrowingLearners this way). The census cannot resolve WHICH handle,
                        # so it records the installation itself as a DATA_DEPENDENT site and
                        # refuses to be complete until that fact has an explicit disposition.
                        found.append(
                            Site(
                                owner_class=owner_class,
                                attribute=attr,
                                callee="<setattr>",
                                defining_module="-",
                                callee_kind="DATA_DEPENDENT",
                                file=rel,
                                line=sub.lineno,
                                method=method.name,
                                conditional=True,
                            )
                        )
                        continue
                    call = _construction_call(value)
                    if call is None:
                        # A comprehension or literal RHS builds no named object. Recorded as
                        # EXPRESSION so it is visibly classified rather than skipped.
                        if not _has_call(value):
                            continue
                        found.append(
                            Site(
                                owner_class=owner_class,
                                attribute=attr,
                                callee="<expression>",
                                defining_module="-",
                                callee_kind="EXPRESSION",
                                file=rel,
                                line=sub.lineno,
                                method=method.name,
                                conditional=_has_conditional(value),
                            )
                        )
                        continue
                    if call is None:
                        continue
                    if isinstance(call.func, ast.Name):
                        callee, prov = call.func.id, imports.get(call.func.id)
                    elif isinstance(call.func, ast.Attribute):
                        callee, prov = call.func.attr, None
                    else:
                        continue
                    kind, mod = _callee_kind(callee, prov, local, root, cache)
                    found.append(
                        Site(
                            owner_class=owner_class,
                            attribute=attr,
                            callee=callee,
                            defining_module=mod,
                            callee_kind=kind,
                            file=rel,
                            line=sub.lineno,
                            method=method.name,
                            conditional=is_setattr or _has_conditional(value),
                        )
                    )
    return tuple(sorted(found, key=lambda s: (s.file, s.line, s.attribute)))


def _has_conditional(node: ast.AST) -> bool:
    return any(isinstance(sub, (ast.IfExp, ast.If)) for sub in ast.walk(node))


def object_handle_sites(root: str) -> tuple[Site, ...]:
    """Sites that construct an OBJECT rather than converting a value.

    EXPRESSION sites are excluded here and handled by `every_expression_site_is_recorded`.
    """
    return tuple(
        s for s in discover_sites(root) if s.callee_kind not in ("CONVERSION", "EXPRESSION")
    )


def expression_sites(root: str) -> tuple[Site, ...]:
    return tuple(s for s in discover_sites(root) if s.callee_kind == "EXPRESSION")


def conversion_sites(root: str) -> tuple[Site, ...]:
    return tuple(s for s in discover_sites(root) if s.callee_kind == "CONVERSION")


def _declared_site_keys() -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in RECORDS:
        for key in rec.sites:
            if key in out:
                raise ValueError(
                    f"site {key!r} is disposed twice ({out[key]} and {rec.owner_path})"
                )
            out[key] = rec.owner_path
    for exc in EXCLUDED:
        # An EXCLUDED key may name several handles at once ("Brain._ach / Brain._lc"), because
        # the reason is one fact about a family of call sites rather than a per-site accident.
        for key in (k.strip() for k in exc.site.replace("+", "/").split("/")):
            if not key:
                continue
            if key in out:
                raise ValueError(f"site {key!r} is both a population record and an exclusion")
            out[key] = "EXCLUDED: " + exc.reason.split(".")[0]
    return out


def undisposed_sites(root: str) -> tuple[Site, ...]:
    """Object-handle construction sites with NO disposition.

    Non-empty means a population is unrecorded. That is refused, never auto-dropped. A CALL
    site is only discharged by an explicit record or an explicit exclusion -- never by the
    METHOD_CALLS_THAT_READ list alone, so a new method call cannot slip through on the
    strength of a name the census happens to recognise.
    """
    declared = _declared_site_keys()
    return tuple(s for s in object_handle_sites(root) if s.key not in declared)


# --------------------------------------------------------------------------------------
# P8 CRITERIA. Each reads its formal parameters, so tools/red_reachability.py cannot class
# any of them as invariant-success on literals; each raises (not returns False) on failure, so
# a passing reading is distinguishable from an unrun one.
# --------------------------------------------------------------------------------------


def named_minimums_present(records: tuple[Record, ...] = RECORDS) -> bool:
    """All four owner-named minimum populations must be declared.

    Reachable red: delete the Brain.attractor row and this raises.
    """
    flat = " | ".join(rec.owner_path for rec in records)
    missing = [p for p in NAMED_MINIMUMS if p not in flat]
    if missing:
        raise AssertionError(f"named minimum population(s) absent from the census: {missing}")
    return True


def every_nested_row_names_a_real_container(records: tuple[Record, ...] = RECORDS) -> bool:
    """A nested row must name a container that some other row or exclusion disposes of.

    Reachable red: point nested_in at a handle nothing disposes of, and this raises -- which is
    how a row would otherwise smuggle in a population with no reachable owner.
    """
    disposed = {key for rec in records for key in rec.sites}
    disposed |= {exc.site for exc in EXCLUDED}
    for rec in records:
        if rec.sites or rec.nested_in.startswith("NO "):
            continue  # a declared absence, spelled out in the field itself
        head = rec.nested_in.split(" ")[0]
        if head not in disposed:
            raise AssertionError(
                f"{rec.owner_path}: nested_in={rec.nested_in!r} names no container handle that any record or exclusion disposes of"
            )
    return True


def no_duplicate_disposition(records: tuple[Record, ...] = RECORDS) -> bool:
    """No driver handle may be disposed twice, and no record may dispose of nothing.

    Reachable red: declare AutonomousArm.jepa in two rows and this raises.
    """
    seen: dict[str, str] = {}
    for rec in records:
        for key in rec.sites:
            if key in seen:
                raise AssertionError(
                    f"handle {key!r} disposed twice: {seen[key]} and {rec.owner_path}"
                )
            seen[key] = rec.owner_path
    return True


def every_citation_resolves(root: str) -> bool:
    """Every record's cited file must exist and its cited line must be inside the file.

    Reachable red: point a record at a file that does not exist.
    """
    for rec in RECORDS:
        path = os.path.join(root, *rec.file.split("/"))
        if not os.path.isfile(path):
            raise AssertionError(f"{rec.owner_path}: citation file {rec.file!r} does not exist")
        n_lines = len(open(path, encoding="utf-8").read().splitlines())
        if not 1 <= rec.line <= n_lines:
            raise AssertionError(
                f"{rec.owner_path}: citation {rec.file}:{rec.line} is outside the file "
                f"({n_lines} lines)"
            )
    return True


def every_unresolved_row_names_its_gap(records: tuple[Record, ...] = RECORDS) -> bool:
    """An UNRESOLVED row must name WHAT is unresolved, not merely that something is.

    Reachable red: replace an UNRESOLVED reason with "unknown".
    """
    banned = {"unknown", "n/a", "tbd", "none", "todo", "?"}
    for rec in records:
        if rec.disposition != "UNRESOLVED":
            continue
        text = rec.reason.strip().lower()
        if len(text) < 25 or text in banned:
            raise AssertionError(f"{rec.owner_path}: UNRESOLVED without a machine-readable cause")
    return True


def no_invented_born_identity(records: tuple[Record, ...] = RECORDS) -> bool:
    """Only the G3 born-unit row may claim a native born identity.

    Reachable red: relabel a legacy index population with NATIVE_BORN_ID -- the map's
    "Legacy names are not born IDs" failure, caught mechanically.
    """
    for rec in records:
        claims_born = rec.identity.startswith("NATIVE_BORN_ID")
        is_g3 = rec.owner_path.startswith("G3 born units")
        if claims_born and not is_g3:
            raise AssertionError(
                f"{rec.owner_path}: claims NATIVE_BORN_ID but is not the G3 born-unit population"
            )
        if is_g3 and not claims_born:
            raise AssertionError(
                f"{rec.owner_path}: the born population must record its native unit_id"
            )
    return True


def every_object_handle_is_disposed(root: str) -> bool:
    """EXHAUSTIVENESS. Every discovered object-handle site must carry a disposition.

    Reachable red: add one line to a copy of brain.py that constructs a new learned object on
    a fresh attribute, and this raises with the site named.
    """
    missing = undisposed_sites(root)
    if missing:
        raise AssertionError(
            "construction site(s) with no census disposition: %s"
            % [f"{s.key} <- {s.callee} ({s.callee_kind}) at {s.citation}" for s in missing]
        )
    return True


def no_second_framework() -> bool:
    """The census must define no search, statistic, scorer, writer, allocator or serializer.

    Reachable red: add a function named e.g. allocate_identity or search_mask to this module.
    """
    forbidden = (
        "search",
        "statistic",
        "scorer",
        "allocate",
        "allocator",
        "serialize",
        "serializer",
        "digest_",
        "run_instance",
        "write_lake",
        "lake_writer",
    )
    names = sorted(
        n for n in globals() if callable(globals()[n]) for bad in forbidden if bad in str(n)
    )
    if names:
        raise AssertionError(f"census must not define a second framework: {names}")
    return True


def runtime_port_is_refused() -> bool:
    """The runtime half is UNRESOLVED by construction and must never be reported done.

    Reachable red: this raises unconditionally; it exists so no caller can read a success.
    """
    raise AssertionError(
        "runtime effective eligible-brain mapping and admission remain UNBUILT: the guarded "
        "sequence was approved at #1084/#1086 and hygiene closed at #1226, but this static "
        "module delivers no runtime port and any runtime slice needs its own explicit grant"
    )


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------


def census_report(root: str) -> dict[str, object]:
    sites = discover_sites(root)
    by_kind: dict[str, int] = {}
    for s in sites:
        by_kind[s.callee_kind] = by_kind.get(s.callee_kind, 0) + 1
    by_disp: dict[str, int] = {}
    for rec in RECORDS:
        by_disp[rec.disposition] = by_disp.get(rec.disposition, 0) + 1
    return {
        "frozen_base": FROZEN_BASE,
        "drivers": [f"{c} ({f})" for c, f in DRIVERS],
        "sites_total": len(sites),
        "sites_by_kind": sorted(by_kind.items()),
        "records": len(RECORDS),
        "excluded": len(EXCLUDED),
        "by_disposition": sorted(by_disp.items()),
        "named_minimums": list(NAMED_MINIMUMS),
        "undisposed": [
            f"{s.key} <- {s.callee} ({s.callee_kind}) at {s.citation}"
            for s in undisposed_sites(root)
        ],
        "conditional_sites": [
            f"{s.key} <- {s.callee} at {s.citation}" for s in sites if s.conditional
        ],
        "runtime_port": "UNBUILT / REFUSED: runtime eligible-brain mapping and admission are "
        "not delivered here; guarded sequence approved at #1084/#1086, hygiene closed at #1226",
        "establishes": "scope only: no sleep benefit, no retention proof, no G9 contract",
    }


def _repo_root() -> str:
    """The checkout root, derived from this file: <root>/src/snn_data_lake/instruments/.

    Three levels up from the instruments directory, never a hard-coded path, so the
    self-check runs in any worktree or clone without editing it.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(3):
        here = os.path.dirname(here)
    return here


def main() -> int:
    root = _repo_root()
    rep = census_report(root)
    print("=" * 78)
    print("G6 all-eligible-circuit STATIC census -- source-derived; no Brain, no home, no lake")
    print("=" * 78)
    print("frozen base            : " + rep["frozen_base"])
    print("drivers                : " + "; ".join(rep["drivers"]))
    print(f"construction sites     : {rep['sites_total']}  by kind {rep['sites_by_kind']}")
    print(f"census records         : {rep['records']}")
    print(f"explicit exclusions    : {rep['excluded']}")
    print("by disposition         : " + str(rep["by_disposition"]))
    print(f"named minimums         : {len(NAMED_MINIMUMS)} required")
    print(f"undisposed handle sites: {len(rep['undisposed'])}")
    print(f"data-dependent sites   : {len(rep['conditional_sites'])}")
    print("runtime port           : " + rep["runtime_port"])
    print("establishes            : " + rep["establishes"])
    print()

    named_minimums_present()
    no_duplicate_disposition()
    every_nested_row_names_a_real_container()
    no_second_framework()
    every_citation_resolves(root)
    every_unresolved_row_names_its_gap()
    no_invented_born_identity()
    every_object_handle_is_disposed(root)
    print("  all in-src P8 criteria GREEN")

    try:
        runtime_port_is_refused()
    except AssertionError as exc:
        print(f"  runtime port refused (expected): {str(exc)[:70]}")
    else:  # pragma: no cover -- the refusal is unconditional
        raise AssertionError("runtime port must refuse")

    print()
    print(f"eligible_circuit_census self-check OK: static census consistent at {FROZEN_BASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
