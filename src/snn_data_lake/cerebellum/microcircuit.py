"""
Cerebellar microcircuit TISSUE -- granule expansion + PF bundle + Purkinje + the MLI
opposite-sign pathway (specs/022 T054-i; FR-019's "on built tissue" substrate). This module
computes near BOTH "cb_ctx" (the cerebellar cortex: granule/PF/MLI/Purkinje stages) and
"cb_deep" (the deep-nuclei readout its output stage drives), and it says those node ids
here on purpose: naming them is what retires the atlas-gate's day-one phantom rows.

WHAT IT IS (Marr 1969 / Albus 1971, the expansion-recoding theory; anatomy as spec):
mossy fibres (delivered by faculties/mossy_relay -- the `pn` stage) fan INTO a much larger
granule population; each granule cell carries FOUR dendrites (the cited anatomical fact:
granule cells have 4-5 dendrites, Eccles/Ito/Szentagothai 1967) and fires only on near-
coincidence of its inputs ON A PHASIC STEP FROM REST -- under sustained drive the leaky
membrane accumulates sub-coincident input too (the latency probe's regime; scoped, not
hidden) -- so on phasic input the granule layer SPARSIFIES its mossy patterns;
granule axons form the PARALLEL-FIBRE bundle crossing every Purkinje dendritic tree;
molecular-layer interneurons (MLI: basket/stellate) read the SAME parallel fibres and
inhibit Purkinje -- the OPPOSITE-SIGN pathway, which makes every Purkinje input row
SIGN-STRUCTURED (PF slots carry a fixed + identity, the MLI slot a fixed - identity):
exactly the sign structure structural_rewiring's parked T048 wire was missing (identity
pruning as native vocabulary) -- and NOT sufficient for it: see the refusal record below,
where T054-ii measured the trigger still unreachable because nothing writes these rows.
Purkinje output is INHIBITORY
onto the deep nuclei (the connectome's cb_ctx -> cb_deep edge is declared "inh"), and the
deep stage balances mossy collateral excitation against it.

STRUCTURE AS SPEC vs CHOSEN vs DERIVED, said per value (P1, re-adjudicated by this
module's own panel): k_dendrites=4 is the cited anatomy (granule cells carry ~4 claw
dendrites) and it is HELD by the codon arms (ablating it to k=1 reds the sparsification
arm). _EXPANSION=4 is a CHOSEN POINT in a monotone family, said plainly -- the panel
measured the first draft's separation criterion ANTI-correlated with it (every value
1..64 passed) -- and its criterion is now the CAPACITY arm: the codon code's hard rank
must EXCEED the mossy input dimension (supra-input recoding, the Marr-Albus property;
measured 64 vs 16 at x4), whose reachable red is the x1 control (capped at exactly the
input dimension, asserted in the same command). Larger expansions only strengthen it
(biology runs ~10^2-10^3 x); this is the gate-affordable point. n_purkinje = n_mossy//2
and k_mli = ceil(sqrt(n_granule)) are chosen structural points, undefended by any
measurement and said so. Every THRESHOLD sits at the LIF's normalized fixed point (P1
case (a)); the MLI weight is -1.0 on the normalized scale (the fixed point balancing a
fully-active PF row). The COINCIDENCE WINDOW is pinned to the MEASURED mossy->Purkinje
channel latency (FR-019) -- the measurement is a STRUCTURAL VERIFICATION (this tissue's
depth answers at 2; the self-check drives a WEAKENED twin -- PF drive x0.4 -- and the
probe REFUSES to pin it at all, because the one-step-later feed-forward inhibition gates
the weakened Purkinje shut past the 32-step budget: the probe demonstrably discriminates
tissue from tissue, and the pin is not a constant compared to itself. That refusal is the
tissue's own FFI window doing the gating, which is the mechanism this window names). RETIRED CLAIM, recorded so it is not re-proposed: the first draft's
Jaccard-gain "pattern separation" figure was an ACTIVITY ARTIFACT top to bottom --
normalized by chance overlap at matched activity, granule codes of similar inputs sit
~27x ABOVE chance (they share codons; similar stays similar) while the k=1 mirror sits
at the input's own ~3x; what the codon code measurably buys at this grain is SPARSITY
and DIMENSIONALITY, not pairwise decorrelation, and the arms now assert exactly those.

Stage latency is REAL: granule spikes computed at step t reach Purkinje/MLI at t+1
through one-step buffers (a channel, not a function call), which is what makes the
latency a measurement instead of an artifact of call order.

THE T048 STRUCTURAL WIRE, MEASURED AND REFUSED ON THIS TISSUE TOO (T054-ii, 2026-08-31 --
the second refusal of the same wire, and the reason is not the one T048 hit). T048 refused
the IXN substrate because it is all-excitatory at birth, so DEEP-R's identity-violation
trigger could never fire; this tissue fixes exactly that (PF(+) rows against an MLI(-)
slot, sign identities READ from the weights). But the trigger is still unreachable here,
for a different reason measured before any wire was written: NOTHING WRITES pf_w. Over
2200 driven steps through the pn relay, PF sign crossings = 0 and PF weights CHANGED = 0
of 512 -- the tissue carries no PF plasticity, because the cerebellar LEARNING RULE
(FR-019's PF-Purkinje depression under the inferior-olive teaching signal) is a later
task's build and the T054 row does not carry it. A structural pass over a substrate
nothing moves is a pass whose prune trigger cannot fire, which is P6's named force-wiring
defect. UNDATED, with a CONDITION a later reader can check instead of a task id to
trust: the wire becomes landable when some mechanism WRITES pf_w (the cerebellar rule is
the expected one, but the criterion is the writing, not the task), and the evidence it
owes is a nonzero sign-crossing count measured the way this record was -- drive the
tissue through the pn relay and count crossings against the birth signs. The T048 FIX (a structural write must also reach the state-of-record) is likewise
inapplicable here and said so rather than faked: no consolidation cascade wraps pf_w, so
pf_w IS the state of record; the fix is only meaningful on the cascade-wrapped IXN rows,
which is the pairing T048 refused.

Deterministic (per-instance random.Random(seed)); stdlib + the reused LIFNeuron.
NO live tick caller yet: the brain-side consumer is future work (the cerebellar rule of
FR-019 lands on this tissue in its own task) -- the dormancy row states this plainly
rather than dressing a harness as a wire.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.microcircuit  # (run by run_selfchecks.py)
"""

import math
import random

from snn_data_lake.growth.lif_neuron import LIFNeuron

# distinct-from: reservoir/cortical_microcircuit.py -- Potjans-Diesmann's 8-population
#   CORTICAL reference dynamics (permanent reference model); this is CEREBELLAR tissue
#   with the expansion/PF/MLI architecture, built to carry live rules. Site read.
# distinct-from: cerebellum/forward_model.py + gain_adaptation.py + spiking_pid.py --
#   FUNCTIONAL cerebellar models (efference copy / LMS gain / control laws); this is the
#   ANATOMICAL tissue layer those functions may one day live on. Sites read.
# distinct-from: cerebellum/jepa_world_model.py + spiking_jepa.py -- latent world models
#   filed in cerebellum; no granule/PF/MLI structure. Sites read.
# distinct-from: sensorimotor/lca_sparse_coding.py -- lateral-inhibition sparse CODING of
#   analog inputs; the granule stage sparsifies by COINCIDENCE thresholding on an
#   expansion, no lateral inhibition, no reconstruction objective. Site read.

NODE_IDS = ("cb_ctx", "cb_deep")  # the connectome ids this tissue computes near

_K_DENDRITES = 4  # cited anatomy: granule cells carry 4-5 dendrites (Eccles/Ito 1967)
# _EXPANSION is a CHOSEN point in a monotone family (docstring adjudication): held by the
# capacity arm (code rank must exceed the input dimension; x1 is the reachable red), NOT
# by the retired separation figure, and NOT "the smallest integer" (that wording was the
# panel's blocker -- 1..64 all passed the first draft's asserts)
_EXPANSION = 4


class CerebellarMicrocircuit:
    """Granule expansion -> PF bundle -> {Purkinje, MLI} -> deep-nuclei readout.

    `step(mossy_spikes)` advances one timestep and returns a dict with granule, mli and
    purkinje 0/1 spike lists plus the scalar deep-nuclei drive. Granule spikes reach the
    molecular layer through a one-step buffer (a real channel). `sign_identities()`
    exposes each Purkinje row's sign structure -- the substrate contract a structural
    pass would consume (its wire is undated; see the refusal record above)."""

    def __init__(self, n_mossy, seed=0):
        if not (isinstance(n_mossy, int) and n_mossy >= _K_DENDRITES):
            raise ValueError(f"n_mossy must be an int >= {_K_DENDRITES}, got {n_mossy!r}")
        self.n_mossy = n_mossy
        self.n_granule = _EXPANSION * n_mossy
        self.n_purkinje = max(2, n_mossy // 2)
        self.n_mli = self.n_purkinje  # one molecular-layer pool per Purkinje (basket/stellate)
        self.seed = seed
        rng = random.Random(seed)
        # GRANULE stage: each cell samples k=4 mossy fibres at UNIFORM weight 1/k -- and
        # 1/4 is EXACT in binary, so full coincidence drives exactly 1.0 == the LIF
        # threshold and fires, while 3/4 = 0.75 stays silent ON A PHASIC STEP FROM REST
        # (the regime scoping is in the docstring). The first draft's failure, corrected
        # to the measured facts by its panel: fan-in-normalized random weights land 1 ulp
        # UNDER 1.0 on ~16 percent of cells (not "about half"), and the ORIGINAL latency
        # probe was ONE-SHOT, so no granule ever crossed threshold and the probe refused
        # -- two defects (ulp-under weights + a probe measuring the leak, not the
        # channel), two fixes (exact uniform weights + a sustained-drive probe).
        self.granule_taps = []
        gw = tuple([1.0 / _K_DENDRITES] * _K_DENDRITES)
        for _ in range(self.n_granule):
            idx = rng.sample(range(n_mossy), _K_DENDRITES)
            self.granule_taps.append((tuple(idx), gw))
        self.granule = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(self.n_granule)]
        # PF bundle: every Purkinje reads the WHOLE parallel-fibre bundle (the crossing
        # geometry), seeded positive weights, fan-in normalized per row
        self.pf_w = []
        for _ in range(self.n_purkinje):
            raw = [rng.random() + 0.1 for _ in range(self.n_granule)]
            s = sum(raw)
            self.pf_w.append([w / s for w in raw])
        # MLI pathway: each pool samples sqrt-many parallel fibres (seeded), fires as a
        # LIF, and inhibits its Purkinje at -1.0 on the normalized scale (the fixed point
        # that balances a fully-active PF row -- opposite sign BY IDENTITY)
        self.k_mli = max(2, math.ceil(math.sqrt(self.n_granule)))
        self.mli_taps = []
        for _ in range(self.n_mli):
            idx = rng.sample(range(self.n_granule), self.k_mli)
            raw = [rng.random() + 0.1 for _ in idx]
            s = sum(raw)
            self.mli_taps.append((tuple(idx), tuple(w / s for w in raw)))
        self.mli = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(self.n_mli)]
        self.mli_w = -1.0  # the opposite-sign identity, fixed (see docstring)
        self.purkinje = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(self.n_purkinje)]
        # one-step channel buffers: granule spikes at t reach the molecular layer at t+1
        self._pf_buf = [0] * self.n_granule
        self._mli_buf = [0] * self.n_mli
        # the COINCIDENCE WINDOW: pinned to the tissue's own measured channel latency
        # (FR-019). Measured once at construction on a throwaway twin so the live
        # instance's buffers stay clean; the self-check re-measures and asserts the pin.
        self.coincidence_window = None
        self.coincidence_window = self._measure_channel_latency()

    # distinct-from: plasticity_rule_stack.py -- its L121 maps drive into BCM's band; this
    #   measures a spike-latency, no overlap beyond the word 'drive'. Site read.
    # distinct-from: reasoning/random_boolean_network.py -- seeded NK wiring; unrelated
    #   term overlap on 'seeded inputs per node'. Site read.
    def _measure_channel_latency(self):
        """Drive a SUSTAINED all-fibre mossy input into a FRESH twin (same seed/dims) and
        count steps from onset to the first Purkinje spike -- the mossy->Purkinje channel
        latency (granule codon + one buffered molecular-layer hop + membrane integration)
        the coincidence window is pinned to. Sustained, not one-shot: a leaky membrane's
        answer to a single-step impulse is an artifact of the leak, not the channel. A
        tissue whose Purkinje never answers within 32 steps is broken, and this refuses
        rather than pinning a lie."""
        twin = object.__new__(CerebellarMicrocircuit)
        twin.__dict__.update(
            {
                k: v
                for k, v in self.__dict__.items()
                if k not in ("granule", "mli", "purkinje", "_pf_buf", "_mli_buf")
            }
        )
        # COPY the mutable structure (pf_w rows) -- sharing by reference was a live
        # aliasing hazard for any future in-place pruning of these rows (panel note)
        twin.pf_w = [list(row) for row in self.pf_w]
        twin.granule = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(self.n_granule)]
        twin.mli = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(self.n_mli)]
        twin.purkinje = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(self.n_purkinje)]
        twin._pf_buf = [0] * self.n_granule
        twin._mli_buf = [0] * self.n_mli
        impulse = [1] * self.n_mossy
        for t in range(1, 33):
            out = twin.step(impulse)
            if any(out["purkinje"]):
                return t
        raise ValueError("channel latency unmeasurable: no Purkinje answer within 32 steps")

    def step(self, mossy_spikes):
        """One timestep. mossy_spikes: 0/1 list (len n_mossy). Returns dict with granule/
        mli/purkinje 0/1 lists and the scalar deep-nuclei drive (mossy collateral
        excitation minus Purkinje inhibition -- the cb_deep balance, both on normalized
        scales)."""
        if len(mossy_spikes) != self.n_mossy:
            raise ValueError(
                f"mossy_spikes has {len(mossy_spikes)} fibres, tissue declares {self.n_mossy}"
            )
        # granule stage (reads THIS step's mossy input)
        g_out = []
        for (idx, w), cell in zip(self.granule_taps, self.granule):
            drive = sum(wi for i, wi in zip(idx, w) if mossy_spikes[i])
            g_out.append(cell.step(drive))
        # molecular layer reads LAST step's parallel-fibre activity (the channel buffer)
        pf = self._pf_buf
        m_out = []
        for (idx, w), cell in zip(self.mli_taps, self.mli):
            drive = sum(wi for i, wi in zip(idx, w) if pf[i])
            m_out.append(cell.step(drive))
        mli_prev = self._mli_buf
        p_out = []
        for j, cell in enumerate(self.purkinje):
            exc = sum(wi for wi, s in zip(self.pf_w[j], pf) if s)
            inh = -self.mli_w if mli_prev[j] else 0.0
            p_out.append(cell.step(exc - inh))
        self._pf_buf = list(g_out)
        self._mli_buf = list(m_out)
        # deep-nuclei readout: mossy collateral excitation minus Purkinje inhibition,
        # each on its own normalized (mean-rate) scale -- computes near "cb_deep"
        mossy_rate = sum(1 for s in mossy_spikes if s) / self.n_mossy
        pk_rate = sum(p_out) / self.n_purkinje
        deep = mossy_rate - pk_rate
        return {"granule": g_out, "mli": m_out, "purkinje": p_out, "deep": deep}

    # distinct-from: substrate/signal_calibrator.py -- its L92 maps a named live SIGNAL to
    #   a persisted running quantile; this removes a named STRUCTURAL element from this
    #   tissue for an A/B. Different object, different verb, prose overlap only. Site read.
    def neuter(self, element):
        """Neuter ONE named tissue element IN PLACE (FR-018's per-element attribution
        instrument, T054-ii). Elements and what each removal means:
          "codon"  -- the granule dendritic gate: each granule keeps ONE tap at weight
                      1.0, so it mirrors a single mossy fibre (the k=1 mirror). ONLY the
                      coincidence requirement goes: the cell COUNT is untouched, so the
                      expansion survives -- the expansion's own ablation is arm (4b)'s
                      x1 capacity control, built differently and deliberately
          "mli"    -- the opposite-sign pathway's WEIGHT (the pools still fire; said
                      plainly, the way the self-check's arm labels it)
          "pf"     -- the parallel-fibre drive onto Purkinje (rows zeroed: the bundle
                      is present, carries nothing). DISCLOSED, because it bounds what
                      this arm can claim: with pf_w zeroed a v_threshold=1.0 LIF cannot
                      fire at all, so this arm's Purkinje delta is -base_pk BY
                      CONSTRUCTION and is indistinguishable from deleting the Purkinje
                      population. It measures that PF drive is Purkinje's ONLY
                      excitatory source, which is worth asserting and is not a
                      graded attribution.
        Returns self so an arm reads as one expression. Refuses an unknown element
        rather than silently neutering nothing -- a no-op neuter would read as a
        measured null."""
        if element == "codon":
            self.granule_taps = [((idx[0],), (1.0,)) for idx, _w in self.granule_taps]
        elif element == "mli":
            self.mli_w = 0.0
        elif element == "pf":
            self.pf_w = [[0.0] * self.n_granule for _ in range(self.n_purkinje)]
        else:
            raise ValueError(f"unknown tissue element {element!r} -- neuterable: codon, mli, pf")
        return self

    # distinct-from: arm_config.py -- its L193 reads joint limits from the mujoco model;
    #   prose-level overlap on 'read from'. Site read.
    def sign_identities(self):
        """Per-Purkinje row sign structure, READ from the tissue's own weights (+1 per PF
        slot from pf_w's sign, the MLI slot from mli_w's sign) -- the sign-structured
        substrate contract a structural pass would consume (T054-ii measured that pass
        unlandable here: nothing writes these rows, so its trigger cannot fire; the
        contract stands, its consumer is undated). VOCABULARY LIMIT, disclosed because
        the panel measured it reachable: this reports +1/-1 only, so a ZEROED slot reads
        as -1 (inhibitory) where structural_rewiring's own _sgn calls it 0 (vacant) --
        after neuter("pf") every PF slot reports -1. Any future consumer must treat
        zero-magnitude slots as vacant from the weights, not from this report. Derived,
        never a literal:
        the panel measured the first draft returning a hardcoded row that could not
        disagree with the weights it claimed to describe; a weight whose sign has drifted
        now CHANGES the report (the self-check constructs that red)."""
        mli_sign = 1 if self.mli_w > 0 else -1
        return [
            tuple((1 if w > 0 else -1) for w in self.pf_w[j]) + (mli_sign,)
            for j in range(self.n_purkinje)
        ]


def _overlap(a, b):
    """Jaccard overlap of two 0/1 lists (0.0 when both empty -- disclosed, not hidden)."""
    inter = sum(1 for x, y in zip(a, b) if x and y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    return inter / union if union else 0.0


def main():
    line = "=" * 96
    print(line)
    print(
        "cerebellum/microcircuit -- T054-i TISSUE: granule expansion + PF bundle + Purkinje "
        "+ MLI (cb_ctx), deep readout (cb_deep)"
    )
    print(line)
    n_mossy = 16
    t1 = CerebellarMicrocircuit(n_mossy, seed=9)
    t2 = CerebellarMicrocircuit(n_mossy, seed=9)
    t3 = CerebellarMicrocircuit(n_mossy, seed=10)
    # (1) PROCEDURALLY REGENERABLE FIXED BASIS: same (seed, dims) -> identical structure
    assert (
        t1.granule_taps == t2.granule_taps and t1.pf_w == t2.pf_w and t1.mli_taps == t2.mli_taps
    ), "same seed must regenerate the identical basis"
    assert t1.granule_taps != t3.granule_taps, "a different seed must differ"
    # (2) SIGN STRUCTURE, read from the weights, with its constructed red: the report
    # must CHANGE when a weight's sign drifts (the first draft's literal could not)
    ids = t1.sign_identities()
    assert len(ids) == t1.n_purkinje and all(
        row[:-1] == tuple([+1] * t1.n_granule) and row[-1] == -1 for row in ids
    ), "the opposite-sign pathway must expose +PF/-MLI identities per row"
    red_t = CerebellarMicrocircuit(n_mossy, seed=9)
    red_t.pf_w[0][3] = -red_t.pf_w[0][3]
    assert red_t.sign_identities()[0][3] == -1 and red_t.sign_identities()[1][3] == +1, (
        "a drifted weight sign must change the identity report at exactly its slot"
    )
    # (3) WINDOW IDENTITY PINNED TO MEASURED CHANNEL LATENCY (FR-019), and the probe must
    # DISCRIMINATE: a weakened twin (PF drive scaled down, needing membrane integration)
    # must measure a LONGER latency than the pin -- the constructed red proving the
    # measurement is of THIS tissue, not a constant compared to itself
    lat = t1._measure_channel_latency()
    assert t1.coincidence_window == lat and isinstance(lat, int) and lat >= 2, (
        f"the coincidence window must BE the measured mossy->Purkinje latency "
        f"(pin {t1.coincidence_window}, re-measured {lat}; >=2: one buffered channel hop)"
    )
    weak = CerebellarMicrocircuit(n_mossy, seed=9)
    weak.pf_w = [[w * 0.4 for w in row] for row in weak.pf_w]
    try:
        _wl = weak._measure_channel_latency()
        assert _wl > lat, (
            f"a weakened tissue must measure a LONGER latency ({_wl} vs {lat}) -- else "
            f"the probe cannot discriminate and the pin is theater"
        )
        weak_red = f"longer latency {_wl}"
    except ValueError:
        # the STRONGER red: with PF weakened below the pre-inhibition window, the MLI
        # feed-forward inhibition (arriving one step later) gates Purkinje shut forever
        # and the probe REFUSES to pin a broken tissue -- discrimination demonstrated,
        # and the window the FFI creates is the tissue's own coincidence gating at work
        weak_red = "probe refusal (>32 steps; FFI gates the weakened PF shut)"
    # (4) WHAT THE CODON CODE MEASURABLY BUYS -- sparsity and capacity, each vs the
    # control that ablates exactly the structure said to buy it. The first draft's
    # Jaccard-gain "separation" figure is RETIRED (activity artifact; docstring record).
    # (4a) SPARSIFICATION vs the k=1 mirror: activity is the quantity itself, no proxy
    rng = random.Random(31)
    g_frac, m_frac, k1_frac = [], [], []
    trials = []
    for _trial in range(20):
        base = [1 if rng.random() < 0.5 else 0 for _ in range(n_mossy)]
        trials.append(base)
        ta = CerebellarMicrocircuit(n_mossy, seed=9)
        ga = ta.step(base)["granule"]
        m_frac.append(sum(base) / n_mossy)
        g_frac.append(sum(ga) / ta.n_granule)
        ca = CerebellarMicrocircuit(n_mossy, seed=9)
        ca.granule_taps = [((idx[0],), (1.0,)) for idx, _w in ca.granule_taps]
        k1_frac.append(sum(ca.step(base)["granule"]) / ca.n_granule)
    mf, gf, kf = (sum(x) / len(x) for x in (m_frac, g_frac, k1_frac))
    assert gf < 0.5 * kf and gf < 0.5 * mf, (
        f"the codon gate must SPARSIFY vs both its input and the k=1 mirror "
        f"(granule {gf:.3f} vs mossy {mf:.3f} vs k=1 {kf:.3f})"
    )
    # (4b) CAPACITY (the criterion that holds _EXPANSION, its red = the x1 control): the
    # codon code's hard rank across an ensemble must EXCEED the mossy input dimension --
    # supra-input recoding, impossible for the x1 control (rank capped at n_mossy)
    try:
        import numpy as _np
    except Exception:
        _np = None
    if _np is not None:
        rng_c = random.Random(41)
        ens = [[1 if rng_c.random() < 0.5 else 0 for _ in range(n_mossy)] for _ in range(200)]
        rows4 = [CerebellarMicrocircuit(n_mossy, seed=9).step(p)["granule"] for p in ens]
        m4 = _np.array(rows4, dtype=float)
        sv4 = _np.linalg.svd(m4, compute_uv=False)
        rank4 = int((sv4 > sv4.max() * max(m4.shape) * _np.finfo(float).eps).sum())
        # the x1 control: a granule layer whose cell count equals its input count CANNOT
        # exceed the input dimension (rank <= min(cells, patterns)) -- the reachable red,
        # constructed in-command on the same codon arithmetic, granule stage only (a full
        # truncated tissue crashed on MLI taps indexing the removed cells -- this arm
        # needs the code, not the molecular layer)
        taps1 = t1.granule_taps[:n_mossy]
        rows1 = []
        for p in ens:
            cells1 = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(n_mossy)]
            rows1.append(
                [
                    cell.step(sum(wi for i, wi in zip(idx, w) if p[i]))
                    for (idx, w), cell in zip(taps1, cells1)
                ]
            )
        m1 = _np.array(rows1, dtype=float)
        sv1 = _np.linalg.svd(m1, compute_uv=False)
        rank1 = int((sv1 > sv1.max() * max(m1.shape) * _np.finfo(float).eps).sum())
        assert rank4 > n_mossy >= rank1, (
            f"the expansion must buy SUPRA-INPUT rank (x4 rank {rank4} > dim {n_mossy} "
            f">= x1 control rank {rank1}) -- the Marr-Albus property, held to its red"
        )
    else:
        rank4 = rank1 = None  # numpy is a core dep; its absence is a broken install

    # (5) THE MLI OPPOSITE-SIGN PATHWAY CHARGES: neutering it (weight 0) must RAISE the
    # Purkinje rate under sustained drive -- feed-forward inhibition doing real work
    def _pk_rate(neuter_mli):
        t = CerebellarMicrocircuit(n_mossy, seed=9)
        if neuter_mli:
            t.neuter("mli")  # ONE mli-neutering mechanism (P2: was a hand-rolled copy)
        rng2 = random.Random(77)
        total = 0
        for _ in range(120):
            m = [1 if rng2.random() < 0.6 else 0 for _ in range(n_mossy)]
            total += sum(t.step(m)["purkinje"])
        return total

    pk_live, pk_neut = _pk_rate(False), _pk_rate(True)
    assert pk_neut > pk_live, (
        f"MLI neuter must raise Purkinje firing (live {pk_live} vs neutered {pk_neut})"
    )
    # (6) PURKINJE OUTPUT IS INHIBITORY AT THE DEEP STAGE, held per step to the readout's
    # own declared identity: deep == mossy_rate - purkinje_rate, so every step where
    # Purkinje fired sits strictly BELOW its mossy excitation, and only those steps do.
    t_live = CerebellarMicrocircuit(n_mossy, seed=9)
    rng3 = random.Random(78)
    pk_steps = 0
    for _ in range(120):
        m = [1 if rng3.random() < 0.6 else 0 for _ in range(n_mossy)]
        out = t_live.step(m)
        mossy_rate = sum(1 for s in m if s) / n_mossy
        pk_rate = sum(out["purkinje"]) / t_live.n_purkinje
        assert out["deep"] == mossy_rate - pk_rate, "the deep readout must BE its identity"
        if pk_rate > 0.0:
            pk_steps += 1
            assert out["deep"] < mossy_rate, "a firing Purkinje must inhibit the deep drive"
        else:
            assert out["deep"] == mossy_rate, "a silent Purkinje must leave the deep drive"
    assert pk_steps > 0, "the driven tissue must fire Purkinje at least once in 120 steps"
    # (7) PONS INTEGRATION end-to-end: ctx -> pn relay -> tissue -> deep, deterministic
    from snn_data_lake.faculties.mossy_relay import MossyRelay

    def _e2e():
        relay = MossyRelay(8, seed=3)
        tissue = CerebellarMicrocircuit(relay.n_pn, seed=9)
        rng4 = random.Random(79)
        trace = []
        for _ in range(40):
            ctx = [1.4 if rng4.random() < 0.5 else 0.0 for _ in range(8)]
            trace.append(tissue.step(relay.step(ctx))["deep"])
        return trace

    e1, e2 = _e2e(), _e2e()
    assert e1 == e2, "two same-seed end-to-end runs must be identical (construction, said)"
    # the panel measured the first form green with Purkinje permanently silenced (deep
    # carries the mossy rate alone) -- the chain now asserts CEREBELLAR participation:
    relay_p = MossyRelay(8, seed=3)
    tissue_p = CerebellarMicrocircuit(relay_p.n_pn, seed=9)
    rng5 = random.Random(79)
    g_total = p_total = 0
    for _ in range(40):
        ctx = [1.4 if rng5.random() < 0.5 else 0.0 for _ in range(8)]
        out = tissue_p.step(relay_p.step(ctx))
        g_total += sum(out["granule"])
        p_total += sum(out["purkinje"])
    assert g_total > 0 and p_total > 0, (
        f"the e2e chain must drive granule AND Purkinje activity (granule {g_total}, "
        f"purkinje {p_total}) -- a deep trace the mossy rate alone explains proves nothing"
    )

    # (8) FR-018 NEUTER-A/B ON THE BUILT TISSUE (T054-ii): per-ELEMENT attribution in ONE
    # command -- each named element removed alone, every arm scored on the SAME tissue-level
    # outcome (the deep readout's mean and the Purkinje rate it carries), against the full
    # tissue and against each other. Per-item neuter, never "the total moved" (P8 cl. 3).
    # The reported channels are (purkinje spikes, granule spikes). `deep` is DELIBERATELY
    # not among them: deep = mossy_rate - pk_rate over an identically-seeded drive, so its
    # delta is exactly -dpk/(steps*n_purkinje) -- an affine image carrying zero independent
    # information, which the panel measured at r = -1.000000 and the first draft published
    # as if it were a third number.
    def _tissue_outcome(neutered):
        t = CerebellarMicrocircuit(n_mossy, seed=9)
        if neutered is not None:
            t.neuter(neutered)
        rngn = random.Random(4242)
        pk = gr = 0
        for _ in range(200):
            m = [1 if rngn.random() < 0.6 else 0 for _ in range(n_mossy)]
            o = t.step(m)
            pk += sum(o["purkinje"])
            gr += sum(o["granule"])
        return pk, gr

    base_pk, base_gr = _tissue_outcome(None)
    attribution = {}
    for elem in ("codon", "mli", "pf"):
        p, g = _tissue_outcome(elem)
        attribution[elem] = (p - base_pk, g - base_gr)
    # every element must MOVE the tissue outcome -- an element whose removal changes
    # nothing is dormant structure (P3), and this arm is what would say so
    for elem, (dp, dg) in attribution.items():
        assert dp != 0 or dg != 0, (
            f"neutering {elem} changed NOTHING (purkinje {dp}, granule {dg}) -- a "
            f"structural element with no measurable participation is dormant"
        )
    # PAIRWISE distinctness: no TWO elements may attribute identically. The first draft
    # asserted `>= 2 distinct values`, which the panel showed passes when two of three
    # collapse -- the realistic failure of a per-element instrument (two neuters removing
    # the same thing), not the three-way lump it was written against.
    _pairs = [("codon", "mli"), ("codon", "pf"), ("mli", "pf")]
    for a_e, b_e in _pairs:
        assert attribution[a_e] != attribution[b_e], (
            f"{a_e} and {b_e} attribute IDENTICALLY {attribution[a_e]} -- the instrument "
            f"is reading them as one element, not per-element charge"
        )

    # CONSTRUCTED RED for the MOVE assert, in the same command (arm 8 shipped without one
    # while every sibling arm carries its own): a no-op element must be caught.
    class _NoOpNeuter(CerebellarMicrocircuit):
        def neuter(self, element):
            return self  # removes nothing

    def _noop_outcome():
        t = _NoOpNeuter(n_mossy, seed=9)
        t.neuter("codon")
        rngn = random.Random(4242)
        pk = gr = 0
        for _ in range(200):
            m = [1 if rngn.random() < 0.6 else 0 for _ in range(n_mossy)]
            o = t.step(m)
            pk += sum(o["purkinje"])
            gr += sum(o["granule"])
        return pk - base_pk, gr - base_gr

    assert _noop_outcome() == (0, 0), (
        "the MOVE criterion's red must be reachable: a no-op neuter has to read as zero "
        "movement, or the arm cannot tell participation from a broken instrument"
    )
    try:
        t1.neuter("nonexistent")
        raise AssertionError("an unknown element must refuse")
    except ValueError as e:
        assert "neuterable" in str(e)
    # (9) refusals
    try:
        t1.step([0] * (n_mossy - 1))
        raise AssertionError("wrong-width mossy input must refuse")
    except ValueError as e:
        assert "fibres" in str(e)
    try:
        CerebellarMicrocircuit(2)
        raise AssertionError("a tissue narrower than one dendrite set must refuse")
    except ValueError as e:
        assert "n_mossy" in str(e)
    rank_bit = (
        f"capacity rank {rank4} > dim {n_mossy} >= x1 control {rank1}; "
        if rank4 is not None
        else "capacity arm skipped (numpy absent -- broken install); "
    )
    print(
        f"self-check OK: basis regenerable ({t1.n_mossy} mossy -> {t1.n_granule} granule "
        f"x{_EXPANSION}, k={_K_DENDRITES} dendrites; {t1.n_purkinje} Purkinje + {t1.n_mli} MLI); "
        f"sign rows READ from weights (+PF/-MLI, drift red constructed); window PINNED to "
        f"measured latency {t1.coincidence_window} steps, weakened-twin red = "
        f"{weak_red}; sparsity granule {gf:.3f} vs mossy {mf:.3f} vs k=1 mirror {kf:.3f}; "
        f"{rank_bit}Jaccard-gain separation RETIRED (activity artifact, docstring record); "
        f"MLI WEIGHT neuter raises PK ({pk_live} -> {pk_neut}; the pools still fire -- the "
        f"weight is the neutered element); Purkinje inhibits deep per-step by identity; "
        f"e2e drives granule+Purkinje through the pn relay; FR-018 per-element neuter A/B "
        f"vs base pk {base_pk}/granule {base_gr} (independent channels only -- the deep "
        f"delta is an affine image of dpk and is not reported): "
        + "; ".join(f"{e} dpk{v[0]:+d} dgr{v[1]:+d}" for e, v in attribution.items())
        + " (pf's dpk is -base_pk by construction: PF is Purkinje's only excitation); "
        + "pairwise-distinct, no-op red constructed; refusals by message"
    )


if __name__ == "__main__":
    main()
