"""
Cross-modal binding -- fuse a VISION percept and an AUDIO percept into ONE unified
multimodal object, the way the brain binds a seen digit "3" and a spoken "three" into a
single cross-modal object. Today the lake perceives N-MNIST vision and SHD audio in
PARALLEL but never BINDS them; this module supplies the missing fusion, reusing three
existing spiking primitives (it reimplements none of them):

  * midbrain.multisensory_integration.MultisensoryCell -- the Stein & Meredith deep
    superior-colliculus cell. A congruent cross-modal pair drives a SUPER-ADDITIVE
    response (bound > sum of the two unimodal responses) via genuine LIF membrane
    dynamics. Content CONGRUENCE is mapped onto the collicular SPATIAL RULE: two percepts
    that name the SAME object share a receptive field (coincident, offset 0 -> super-
    additive), while percepts that name DIFFERENT objects fall in disparate fields
    (non-coincident, large offset -> no enhancement, index ~= 1). Uni-modal input (one
    cue absent) also gives index ~= 1. So super-additivity fires ONLY for a congruent
    bimodal pair -- exactly the multisensory-enhancement signature.

  * cortex.vsa_reasoning -- spike-polarity hyperdimensional codes. Each latent is turned
    into a D-dim two-line spike code by a deterministic sign random projection (locality-
    sensitive: cosine-similar latents -> coincident codes). The unified object code is the
    spike-XOR bind of the two modality codes, bound_code = bind(V, A). Because spike-XOR
    is SELF-INVERSE, either modality UNBINDS the other back out: bind(bound_code, V) -> A,
    bind(bound_code, A) -> V -- the object is recoverable from EITHER modality. A bundled
    cross-modal associative memory (role = object handle, filler = modality code) lets the
    object HANDLE be recalled from a single-modality cue by superposition + cleanup, and
    that recall degrades gracefully with the number of stored objects (finite HDC capacity).

  * claustrum.Claustrum -- the coincidence-gated cortical binder. The two modalities are
    fed as two "areas"; when they are present AND congruent they spike coincidently, the
    claustrum fires and binds the assembly {vision, audio}; an incongruent or uni-modal
    pair never reaches the coincidence threshold and binds nothing. This gives the bound
    object a genuine spiking-coincidence provenance (the "claustrum assembly" handle).

INTERFACE (what the brain calls):
  register(handle, vis_latent, aud_latent)         -- teach a known object (its two prototypes)
  bind(vis_latent, aud_latent) -> {bound_code, superadditive, congruent, handle, index,
                                    assembly, sim_vis, sim_aud}
  recall(cue_latent, modality) -> (handle, sim)    -- recover the OBJECT from ONE modality
  recover_complement(bound_code, cue_latent, modality) -> (handle, sim)  -- unbind the other half
  superadditivity_index(vis_latent, aud_latent) -> float

LEARNING RULE: brain-native VSA / HDC (spike-XOR bind, superposition-WTA bundle, coincidence
cleanup) + a fixed LIF collicular cell. There is NO gradient anywhere -- no loss.backward(),
no learned weights -- so no cosine(e-prop, BPTT) check applies. Binding is algebraic over
spike codes; recognition is nearest-code cleanup.

HONEST FRAMING / NEGATIVE: this binds FIXED latent VECTORS -- it is not a learned perceptual
object hierarchy and it does not learn features from N-MNIST / SHD pixels. "Vision" and
"audio" latents are whatever the upstream percepts hand it; the sign random projection is a
fixed locality-sensitive hash, not a trained encoder. Congruence is decided by nearest-
prototype cleanup over registered objects, so an object must be REGISTERED before it can be
recognised or bound. Single-pair unbind is near-lossless by construction (spike-XOR is exact);
the honest degradation shows up only in the bundled multi-object recall (capacity), reported
below. Nothing here is aware of anything.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.crossmodal_binder
"""

import random
import zlib

from snn_data_lake.claustrum.claustrum import Claustrum
from snn_data_lake.cortex.vsa_reasoning import (
    D,
    bind,
    bundle,
    cleanup,
    coincidence,
    hv,
    is_spike_code,
)
from snn_data_lake.midbrain.multisensory_integration import MultisensoryCell

_MODALITIES = ("vision", "audio")


class CrossModalBinder:
    """Bind a vision latent and an audio latent into one unified, recoverable multimodal object.

    Reuses MultisensoryCell (super-additivity), vsa_reasoning (recoverable object code), and
    Claustrum (coincidence-gated assembly). Objects are taught with `register`; `bind` fuses a
    pair; `recall` recovers the object from a single modality; `recover_complement` unbinds the
    other modality out of a bound code. Deterministic (per-instance / crc32 seeding),
    algebra over discrete spike codes, no gradient learning."""

    def __init__(self, cue_strength=0.055, recog_margin=0.15, seed=0):
        # cue_strength: weak collicular drive per modality (weak regime -> super-additive when
        # coincident). recog_margin: min centred spike-coincidence to accept a nearest-prototype
        # match (below it a modality is treated as UNrecognised).
        self.dim = D
        self.cue_strength = cue_strength
        self.recog_margin = recog_margin
        self._msc = MultisensoryCell(seed=seed)  # fixed LIF SC cell (super-additivity)
        self._far = self._msc.dur + 200  # offset that fully violates coincidence
        self.objects = {}  # handle -> {vision_code, audio_code, bound_code, ...}
        self._proj = {}  # (modality, latent_dim) -> projection matrix
        self._mem = {"vision": None, "audio": None}  # bundled cross-modal associative memories

    # -- latent -> spike code (locality-sensitive sign random projection) -------------------------
    def _matrix(self, modality, latent_dim):
        """Deterministic D x latent_dim Gaussian projection for a modality (crc32-seeded, cached).
        Sign random projection is locality-sensitive: cosine-similar latents -> coincident codes."""
        key = (modality, latent_dim)
        R = self._proj.get(key)
        if R is None:
            rng = random.Random(zlib.crc32(f"{modality}:{latent_dim}".encode()))
            R = [[rng.gauss(0.0, 1.0) for _ in range(latent_dim)] for _ in range(self.dim)]
            self._proj[key] = R
        return R

    def _project(self, latent, modality):
        """Map a real latent vector to a D-dim two-line spike code (+1 up line / -1 down line)."""
        if modality not in _MODALITIES:
            raise ValueError(f"modality must be one of {_MODALITIES}, got {modality!r}")
        vec = [float(x) for x in latent]
        R = self._matrix(modality, len(vec))
        return [1 if sum(r * x for r, x in zip(row, vec)) >= 0.0 else -1 for row in R]

    # -- teach a known object ---------------------------------------------------------------------
    def register(self, handle, vis_latent, aud_latent):
        """Teach an object: store its vision + audio spike codes and their bound object code, and
        rebuild the cross-modal associative memories. `handle` is a hashable label."""
        V = self._project(vis_latent, "vision")
        A = self._project(aud_latent, "audio")
        self.objects[str(handle)] = {
            "vision_code": V,
            "audio_code": A,
            "bound_code": bind(V, A),
            "vision_latent": [float(x) for x in vis_latent],
            "audio_latent": [float(x) for x in aud_latent],
        }
        self._rebuild_memory()
        return str(handle)

    def _rebuild_memory(self):
        """Bundle role(object-handle) -> filler(modality code) into ONE memory code per modality.
        Recall probes it with a modality cue by spike XOR, then cleans up to the handle."""
        for m in _MODALITIES:
            pairs = [bind(hv(h), o[m + "_code"]) for h, o in self.objects.items()]
            self._mem[m] = bundle(pairs) if pairs else None

    # -- recognition (nearest registered prototype by spike coincidence) --------------------------
    def _recognize(self, code, modality):
        """Nearest registered object for a modality code -> (handle, centred coincidence). Returns
        (None, sim) when nothing clears recog_margin or nothing is registered."""
        book = {h: o[modality + "_code"] for h, o in self.objects.items()}
        if not book:
            return None, 0.0
        handle, s = cleanup(code, book)
        return (handle if s >= self.recog_margin else None), s

    # -- the fusion op ----------------------------------------------------------------------------
    def bind(self, vis_latent, aud_latent):
        """Fuse a vision latent and an audio latent into a unified multimodal object.

        Returns a dict:
          bound_code    -- VSA spike-XOR object code bind(V, A); unbinds to either modality.
          superadditive -- True iff a congruent bimodal pair drove the SC cell above additive.
          congruent     -- True iff both modalities are present and name the SAME registered object.
          handle        -- the unified object label (congruent) else None.
          index         -- the multisensory super-additivity index (ms / (uv + ua)); None if silent.
          assembly      -- the claustrum-bound modality set (frozenset) or None.
          sim_vis/sim_aud -- each modality's recognition coincidence margin (0.0 if absent).
        """
        vis_present = vis_latent is not None
        aud_present = aud_latent is not None
        V = self._project(vis_latent, "vision") if vis_present else None
        A = self._project(aud_latent, "audio") if aud_present else None

        obj_v, s_v = self._recognize(V, "vision") if vis_present else (None, 0.0)
        obj_a, s_a = self._recognize(A, "audio") if aud_present else (None, 0.0)
        congruent = bool(vis_present and aud_present and obj_v is not None and obj_v == obj_a)

        # (a) super-additivity via the fixed LIF SC cell. Congruence -> shared receptive field
        #     (coincident, offset 0); incongruence -> disparate field (offset self._far, no enhance).
        W = self.cue_strength
        cue_v = W if vis_present else 0.0
        cue_a = W if aud_present else 0.0
        offset = 0 if congruent else self._far
        e = (
            self._msc.enhancement(cue_v, cue_a, offset)
            if (vis_present or aud_present)
            else {"index": None}
        )
        index = e["index"]
        superadditive = index is not None and index > 1.0 + 1e-9

        # (b) unified, recoverable object code (spike-XOR bind of the two modality codes).
        bound_code = bind(V, A) if (vis_present and aud_present) else (V if vis_present else A)

        # (c) claustrum assembly: two areas that spike coincidently ONLY for a congruent bimodal
        #     pair -> the coincidence gate fires and binds {vision, audio}; else binds nothing.
        assembly = self._claustrum_assembly(vis_present, aud_present, congruent)

        return {
            "bound_code": bound_code,
            "superadditive": superadditive,
            "congruent": congruent,
            "handle": obj_v if congruent else None,
            "index": index,
            "assembly": assembly,
            "sim_vis": s_v,
            "sim_aud": s_a,
        }

    def _claustrum_assembly(self, vis_present, aud_present, congruent):
        """Coincidence-gated binding of the two modality 'areas' via the existing Claustrum.
        Congruent bimodal -> coincident spikes -> fire -> bound {vision, audio}; else None."""
        cl = Claustrum(n_areas=2, k=2, window=3, seed=0)
        n_steps = 16
        rasters = [[0] * n_steps for _ in range(2)]
        if vis_present:
            rasters[0][5] = 1
        if aud_present:
            rasters[1][5 if congruent else 12] = 1  # coincident iff congruent
        _, assemblies = cl.detect(rasters)
        bound = cl.bound_areas(assemblies)
        if not bound:
            return None
        names = {0: "vision", 1: "audio"}
        return frozenset(names[a] for a in bound)

    # -- recovery ---------------------------------------------------------------------------------
    def recall(self, cue_latent, modality):
        """Recover the OBJECT HANDLE from a SINGLE modality cue via the cross-modal associative
        memory (spike XOR probe + coincidence cleanup over object handles). Returns (handle, sim);
        (None, 0.0) if nothing is registered. Degrades with the number of stored objects (capacity)."""
        mem = self._mem.get(modality)
        if mem is None or not self.objects:
            return None, 0.0
        cue = self._project(cue_latent, modality)
        probe = bind(mem, cue)  # unbind the object-handle role out
        handles = {h: hv(h) for h in self.objects}
        return cleanup(probe, handles)

    def recover_complement(self, bound_code, cue_latent, modality):
        """Unbind the COMPLEMENT modality out of a bound object code with a single-modality cue,
        then name it by cleanup against the complement codebook. Returns (handle, sim). This is the
        spike-XOR self-inverse: bind(bound_code, V) -> A (and vice-versa)."""
        if modality not in _MODALITIES:
            raise ValueError(f"modality must be one of {_MODALITIES}, got {modality!r}")
        other = "audio" if modality == "vision" else "vision"
        book = {h: o[other + "_code"] for h, o in self.objects.items()}
        if not book:
            return None, 0.0
        cue = self._project(cue_latent, modality)
        return cleanup(bind(bound_code, cue), book)

    def superadditivity_index(self, vis_latent, aud_latent):
        """The multisensory super-additivity index for a pair (convenience wrapper over bind)."""
        return self.bind(vis_latent, aud_latent)["index"]


# -------------------------------------------------------------------------------------------------
def _rng_latent(rng, dim):
    """A random real latent vector (a stand-in for an upstream N-MNIST / SHD percept)."""
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def _noisy(rng, latent, scale):
    """Perceptual jitter around a prototype latent (different samples of the same object differ)."""
    return [x + rng.gauss(0.0, scale) for x in latent]


def main():
    LATENT_DIM = 48
    K = 12  # registered objects (e.g. digits 0-9 + two extras)
    NOISE = 0.35  # perceptual jitter on each presented percept
    TRIALS = 8  # noisy presentations per object
    chance = 1.0 / K

    binder = CrossModalBinder(seed=0)
    proto_rng = random.Random(2024)
    protos = {}
    for i in range(K):
        vp = _rng_latent(proto_rng, LATENT_DIM)
        ap = _rng_latent(proto_rng, LATENT_DIM)
        h = f"obj{i}"
        protos[h] = (vp, ap)
        binder.register(h, vp, ap)

    # ---- (a) SUPER-ADDITIVITY: congruent bimodal vs incongruent-bimodal / uni-only controls -----
    trial_rng = random.Random(7)
    handles = list(protos)
    cong_idx, incong_idx, uni_idx = [], [], []
    for h in handles:
        vp, ap = protos[h]
        v = _noisy(trial_rng, vp, NOISE)
        a = _noisy(trial_rng, ap, NOISE)
        cong = binder.bind(v, a)  # congruent pair
        other = protos[handles[(handles.index(h) + 1) % K]]  # a DIFFERENT object's audio
        a_bad = _noisy(trial_rng, other[1], NOISE)
        incong = binder.bind(v, a_bad)  # incongruent pair
        uni = binder.bind(v, None)  # vision only
        if cong["index"] is not None:
            cong_idx.append(cong["index"])
        if incong["index"] is not None:
            incong_idx.append(incong["index"])
        if uni["index"] is not None:
            uni_idx.append(uni["index"])

    cong_mean = sum(cong_idx) / len(cong_idx)
    incong_mean = sum(incong_idx) / len(incong_idx)
    uni_mean = sum(uni_idx) / len(uni_idx)
    cong_super = all(x > 1.0 for x in cong_idx)
    incong_super = any(x > 1.0 + 1e-9 for x in incong_idx)
    uni_super = any(x > 1.0 + 1e-9 for x in uni_idx)

    # a representative fused object for reporting / assembly / unbind checks
    v0 = _noisy(trial_rng, protos["obj0"][0], NOISE)
    a0 = _noisy(trial_rng, protos["obj0"][1], NOISE)
    fused = binder.bind(v0, a0)

    # ---- (b) RECOVER OBJECT FROM ONE MODALITY (associative memory) vs random-cue control --------
    rec_rng = random.Random(11)
    hit_v = hit_a = rand_hit = total = 0
    for h in handles:
        vp, ap = protos[h]
        for _ in range(TRIALS):
            if binder.recall(_noisy(rec_rng, vp, NOISE), "vision")[0] == h:
                hit_v += 1
            if binder.recall(_noisy(rec_rng, ap, NOISE), "audio")[0] == h:
                hit_a += 1
            if binder.recall(_rng_latent(rec_rng, LATENT_DIM), "vision")[0] == h:
                rand_hit += 1  # random cue -> chance
            total += 1
    acc_v = hit_v / total
    acc_a = hit_a / total
    acc_rand = rand_hit / total

    # ---- (c) UNBIND the complement modality out of a bound code vs random-object-code control ----
    # Fair negative control: unbind a RANDOM full-entropy object code with the TRUE cue -> recovery
    # requires the ACTUAL bound code, so a random code recovers the object only at chance.
    ctrl_rng = random.Random(99)
    unbind_hit = unbind_total = 0
    unbind_rand = unbind_rand_total = 0
    for h in handles:
        vp, _ap = protos[h]
        bound = binder.objects[h]["bound_code"]  # bind(V, A)
        for _ in range(TRIALS):
            cue = _noisy(rec_rng, vp, NOISE)
            if binder.recover_complement(bound, cue, "vision")[0] == h:
                unbind_hit += 1  # vision cue -> recover audio -> h
            unbind_total += 1
            for _ in range(6):  # more control draws -> stable estimate
                rand_bound = [ctrl_rng.choice((-1, 1)) for _ in range(binder.dim)]
                if binder.recover_complement(rand_bound, cue, "vision")[0] == h:
                    unbind_rand += 1  # random object code -> ~chance
                unbind_rand_total += 1
    unbind_acc = unbind_hit / unbind_total
    unbind_rand_acc = unbind_rand / unbind_rand_total

    print("=" * 76)
    print("CROSS-MODAL BINDER -- vision + audio fused into one recoverable multimodal object")
    print("=" * 76)
    print(
        f"D={binder.dim}  registered objects K={K}  latent_dim={LATENT_DIM}  cue={binder.cue_strength} "
        f"noise={NOISE}  chance={chance:.3f}"
    )
    print("-" * 76)
    print("(a) SUPER-ADDITIVITY (Stein-Meredith collicular cell, congruence -> spatial rule):")
    print(
        f"    congruent bimodal  mean index = {cong_mean:.3f}  (> 1: bound EXCEEDS unimodal sum, "
        f"all {len(cong_idx)}/{len(cong_idx)} super-additive)"
    )
    print(
        f"    incongruent control mean index = {incong_mean:.3f}  (any super-additive? {incong_super})"
    )
    print(f"    uni-only control    mean index = {uni_mean:.3f}  (any super-additive? {uni_super})")
    print(
        f"    example fused obj0: superadditive={fused['superadditive']} congruent={fused['congruent']} "
        f"handle={fused['handle']} assembly={sorted(fused['assembly']) if fused['assembly'] else None}"
    )
    print("-" * 76)
    print("(b) RECOVER OBJECT FROM ONE MODALITY (cross-modal associative memory, cleanup):")
    print(
        f"    from VISION cue: {acc_v:.3f}   from AUDIO cue: {acc_a:.3f}   "
        f"random-cue control: {acc_rand:.3f}  (chance {chance:.3f})"
    )
    print("-" * 76)
    print("(c) UNBIND complement modality from bound code (spike-XOR self-inverse):")
    print(
        f"    vision cue -> recover audio -> object: {unbind_acc:.3f}   "
        f"random-object-code control: {unbind_rand_acc:.3f}  (chance {chance:.3f})"
    )
    print("=" * 76)

    # ---- un-riggable self-checks (FAILING controls at ~chance / no enhancement) ------------------
    # (a) congruent bimodal is super-additive; incongruent and uni-only controls are NOT.
    assert cong_idx and cong_super, (
        "every congruent bimodal pair must be super-additive (index > 1)"
    )
    assert cong_mean > 1.0 + 1e-6, (
        f"congruent mean index must exceed 1 (super-additive): {cong_mean:.3f}"
    )
    assert not incong_super, "incongruent bimodal control must NOT be super-additive (index ~= 1)"
    assert not uni_super, "uni-modal control must NOT be super-additive (index ~= 1)"
    assert cong_mean > incong_mean + 0.1 and cong_mean > uni_mean + 0.1, (
        "congruent enhancement must clearly beat the incongruent / uni-only controls"
    )
    assert fused["superadditive"] and fused["congruent"] and fused["handle"] == "obj0", (
        "a congruent fused pair must report superadditive + congruent + its handle"
    )
    assert fused["assembly"] == frozenset(("vision", "audio")), (
        "the claustrum must bind the {vision, audio} assembly for a congruent bimodal pair"
    )

    # (b) object is recoverable from EITHER modality well above chance; random cue is AT chance.
    assert acc_v > 0.6 and acc_a > 0.6, (
        f"object must be recoverable from a single modality (vision {acc_v:.2f}, audio {acc_a:.2f})"
    )
    assert acc_rand < chance * 2.0 + 1e-9, (
        f"a random cue must recall at chance ({acc_rand:.3f} vs chance {chance:.3f})"
    )
    assert acc_v > acc_rand + 0.3 and acc_a > acc_rand + 0.3, (
        "single-modality recall must clearly beat the random-cue control"
    )

    # (c) unbind recovers the object above chance; the random-pair control is AT chance.
    assert unbind_acc > 0.6, (
        f"unbind must recover the complement modality's object ({unbind_acc:.2f})"
    )
    assert unbind_rand_acc < chance * 2.0 + 1e-9, (
        f"the random-pair unbind control must be at chance ({unbind_rand_acc:.3f})"
    )
    # exact self-inverse: a CLEAN pair unbinds to the exact complement code.
    Vc = binder.objects["obj0"]["vision_code"]
    Ac = binder.objects["obj0"]["audio_code"]
    Bc = binder.objects["obj0"]["bound_code"]
    assert bind(Bc, Vc) == Ac and bind(Bc, Ac) == Vc, "spike-XOR bind must be exactly self-inverse"

    # (d) genuinely spiking substrate + deterministic.
    assert is_spike_code(Bc) and is_spike_code(Vc) and is_spike_code(Ac), (
        "bound and modality codes must be discrete two-line spike codes (+1 / -1)"
    )
    assert coincidence(Vc, Vc) == binder.dim, "self-coincidence must be the full dimensionality"
    b2 = CrossModalBinder(seed=0)
    for h in handles:
        b2.register(h, *protos[h])
    assert b2.bind(v0, a0)["bound_code"] == fused["bound_code"], (
        "same construction + inputs must reproduce the identical bound code (deterministic)"
    )

    print(
        f"self-check OK: congruent super-additive {cong_mean:.2f} (>1) vs incongruent {incong_mean:.2f} "
        f"/ uni {uni_mean:.2f} (neither super); object recalled from vision {acc_v:.2f} / audio "
        f"{acc_a:.2f} vs random {acc_rand:.2f} (chance {chance:.2f});"
    )
    print(
        f"  unbind recovers complement {unbind_acc:.2f} vs random-pair {unbind_rand_acc:.2f}; spike-XOR "
        f"self-inverse exact; claustrum binds {{vision,audio}}; spike codes; deterministic"
    )


if __name__ == "__main__":
    main()
