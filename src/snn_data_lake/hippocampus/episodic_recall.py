"""
v3.79 - episodic_recall: RECONSTRUCTIVE / AUTOBIOGRAPHICAL episodic memory over the spiking
VSA substrate.

WHY THIS EXISTS (the honest gap it closes). hippocampus/affective_memory.py stores affect-tagged
episodes and recall_similar returns a whole Episode, but the conversational surface only ever reads
its affect TAG (valence/arousal) -- there is no RECONSTRUCTIVE recall of episode CONTENT: no way to
give a PARTIAL cue ("the fearful one", or a single where-slot) and have the system COMPLETE it back
into the full episode's other slots, and no way to read out the GIST/schema shared across a family
of similar episodes. That is the Tulving distinction: affective_memory does familiarity/somatic-
marker retrieval; this module does episodic RE-CONSTRUCTION (pattern completion) and semantic GIST
extraction. Grounded in Marr (1971) hippocampal auto-association and Tulving (1972) episodic vs
semantic memory: an episode is a bound conjunction of who/what/where/when/affect, and recall is
content-addressable pattern completion from a partial cue, not a lookup by key.

HOW (the mechanism -- genuinely spiking, no gradients). Each episode is encoded the HDC/VSA way,
reusing the verified spiking primitives in cortex/vsa_reasoning (two-line spike-polarity codes;
bind = self-inverse spike-XOR coincidence gate; bundle = spike-superposition majority WTA;
coincidence/sim = spike-coincidence count; cleanup = snap-to-nearest codebook spike code):

  trace(episode) = bundle_over_slots( bind(role[slot], filler_code[slot,value]) )

i.e. each role slot is BOUND to its filler spike code and all slot-bindings are SUPERPOSED into one
composite spike code (a one-shot hetero-associative / Hebbian trace -- bundling IS the associative
increment). Reconstruction is pattern completion:

  1. build a cue_trace from the KNOWN slots of a partial cue (same bind+bundle);
  2. RETRIEVE the stored episode trace most spike-coincident with the cue_trace
     (content-addressable: the cue's bindings are a subset of the right episode's bindings);
  3. UNBIND every role from the RETRIEVED trace (bind is self-inverse) and CLEAN UP each result to
     the nearest filler in that slot's codebook -- recovering the HELD-OUT slots the cue never
     supplied. That third step is the reconstructive move: the missing slots come from the STORED
     trace, not from the cue.

GIST/schema: bundle the traces of a retrieved FAMILY of episodes into a prototype, unbind each role;
slots that are INVARIANT across the family reinforce and clean up with a high spike-coincidence
margin (they enter the schema), slots that VARY destructively interfere and fall below margin (they
are left out). A single episode yields NO gist (a schema needs a population to separate typical from
idiosyncratic) -- returned honestly as an empty schema.

LEARNING RULE. VSA role-filler binding + bundle superposition = a one-shot Hebbian/attractor
associative memory. There is NO gradient anywhere (no loss.backward, no BPTT): storage is a single
superposition write, recall is fixed spike-algebra + nearest-code cleanup. So the e-prop-vs-BPTT
cosine gate does not apply (nothing here learns by gradient).

SAFETY / SCOPE. Representation/value-side ONLY. This module NEVER selects an action, NEVER calls the
arm, NEVER emits a motor command, and imports NOTHING from brainstem/ or any safety-envelope file.
It reuses cortex/vsa_reasoning (pure spike algebra). Additive/opt-in: nothing runs at import; work
happens only under explicit calls or __main__.

HONEST CAVEATS. (1) The role slots (who/what/where/when/affect) are DEVELOPER-DEFINED, not
discovered. (2) Reconstruction is RETRIEVAL over STORED episodes + cleanup, NOT generative
imagination -- it can only complete to an episode it actually holds (or fail honestly with low
confidence). (3) Fillers must be in the slot codebook for cleanup to name them; a novel cue value
retrieves best-effort and reports low confidence. (4) Capacity is finite: per-episode traces
superpose ~S slot-bindings and cross-episode retrieval is a coincidence scan, so recovery degrades
as slots/episodes grow -- exactly as a spiking associative memory should.

Determinism: crc32-seeded atomic spike codes (never builtin hash); per-instance random.Random(seed)
for the noise/self-check only; ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.episodic_recall
"""

import random
import zlib

from snn_data_lake.cortex.vsa_reasoning import bind, bundle, cleanup, coincidence, hv, sim

# The developer-defined role slots (HONEST: these are hand-specified, not induced). An episode dict
# may supply any subset; unknown keys are accepted and get their own role vector on first sight.
DEFAULT_SLOTS = ("who", "what", "where", "when", "affect")


def _flip(code, frac, rng):
    """Corrupt a spike code by flipping `frac` of its spikes to the opposite line (graceful-noise
    probe for the self-check). Per-instance rng -> deterministic; returns a NEW code."""
    out = list(code)
    n = len(out)
    idx = rng.sample(range(n), int(round(frac * n)))
    for i in idx:
        out[i] = -out[i]
    return out


class EpisodicRecall:
    """Reconstructive/autobiographical episodic memory over the spiking VSA.

    Public interface (the brain calls exactly these):
      encode(episode_dict)        -> key         : bind slots as role-filler, superpose into one
                                                    spike-code trace, store it. Grows the codebook.
      complete(partial_cue_dict)  -> result dict : content-addressable pattern completion --
                                                    retrieve the most coincident stored trace and
                                                    UNBIND every role from it, recovering the slots
                                                    the cue never supplied. Keys of the result:
                                                    {"episode", "confidence", "key", "slot_conf"}.
      gist(cue_dict, m=6)         -> result dict : schema shared across the top-m retrieved family
                                                    (INVARIANT slots only). {"schema","margins","n"}.
      narrate(episode_dict)       -> str         : a short autobiographical sentence (template fill
                                                    over the slots -- NOT generative language).

    Deterministic, safety-inert (never selects/executes an action)."""

    def __init__(self, slots=DEFAULT_SLOTS, seed=0, gist_thresh=0.30):
        self.seed = int(seed)
        self.gist_thresh = float(gist_thresh)
        self._role = {}  # slot -> role spike code (fixed, crc32-seeded)
        self._codebook = {}  # slot -> {value_str: filler spike code}
        self._episodes = []  # list of {"key": crc32, "trace": spike code}
        for s in slots:
            self._ensure_slot(s)

    def __len__(self):
        return len(self._episodes)

    # ---- vocabulary -------------------------------------------------------------------
    def _ensure_slot(self, slot):
        """Register a role vector + empty codebook for a slot on first sight (idempotent)."""
        if slot not in self._role:
            self._role[slot] = hv(f"ROLE|{slot}")
            self._codebook[slot] = {}

    @staticmethod
    def _valstr(value):
        """Deterministic ASCII string image of a filler value (repr, so ints/strs/tuples all key)."""
        return value if isinstance(value, str) else repr(value)

    def _filler_code(self, slot, value, register=False):
        """Atomic spike code for one (slot, value) filler. crc32-seeded via hv -> deterministic and
        independent of insertion order. When register=True the value is added to the slot codebook so
        cleanup can later NAME it; cue-building uses register=False so a novel cue value stays out of
        the vocabulary (and honestly retrieves at low confidence)."""
        self._ensure_slot(slot)
        vs = self._valstr(value)
        code = hv(f"FILLER|{slot}|{vs}")
        if register:
            self._codebook[slot][vs] = code
        return code

    # ---- encoding ---------------------------------------------------------------------
    def _trace(self, episode, register):
        """Bind each slot's role to its filler and SUPERPOSE (bundle) into one composite spike code.
        register=True while storing (grows the codebook); False while building a transient cue."""
        parts = [
            bind(self._role[s], self._filler_code(s, v, register))
            for s, v in episode.items()
            if v is not None
        ]
        if not parts:
            return None
        return bundle(parts)

    def encode(self, episode):
        """Encode + store one episode dict {slot: value}. Returns a crc32 key. The trace is a single
        superposed spike code; every filler value enters the slot codebook (semantic vocabulary)."""
        for s in episode:
            self._ensure_slot(s)
        trace = self._trace(episode, register=True)
        if trace is None:
            raise ValueError("cannot encode an empty episode")
        raw = "|".join(f"{s}={self._valstr(episode[s])}" for s in sorted(episode)).encode("ascii")
        key = zlib.crc32(raw)
        self._episodes.append({"key": key, "trace": trace})
        return key

    # ---- retrieval + reconstruction ----------------------------------------------------
    def _retrieve(self, cue_trace, k=1):
        """Rank stored episodes by spike coincidence with the cue trace (content-addressable).
        Returns the top-k [(episode_record, sim)] most coincident, most-similar first."""
        scored = [(coincidence(cue_trace, ep["trace"]), ep) for ep in self._episodes]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(ep, sim(cue_trace, ep["trace"])) for _c, ep in scored[:k]]

    def _reconstruct(self, trace):
        """UNBIND every role from a trace and CLEAN UP to the nearest filler -> full slot dict, plus
        a per-slot spike-coincidence confidence. This is the reconstructive step: held-out slots are
        recovered from the STORED trace, not from the cue."""
        recon, conf = {}, {}
        for slot, role in self._role.items():
            book = self._codebook.get(slot)
            if not book:
                continue  # nothing ever stored in this slot -> cannot name it
            label, s = cleanup(bind(trace, role), book)  # bind is self-inverse -> this unbinds
            recon[slot] = label
            conf[slot] = round(s, 3)
        return recon, conf

    def complete(self, partial_cue, cue_trace=None):
        """Pattern-complete a full episode from a PARTIAL cue. Build a cue trace from the cue's known
        slots (or accept a pre-corrupted cue_trace for the noise probe), retrieve the most coincident
        stored episode, and reconstruct ALL its slots by unbinding. Returns
        {"episode","confidence","key","slot_conf"}; confidence<=0 and key=None if the store is empty."""
        if not self._episodes:
            return {"episode": {}, "confidence": 0.0, "key": None, "slot_conf": {}}
        if cue_trace is None:
            cue_trace = self._trace(partial_cue, register=False)
        if cue_trace is None:
            return {"episode": {}, "confidence": 0.0, "key": None, "slot_conf": {}}
        ep, conf = self._retrieve(cue_trace, k=1)[0]
        recon, slot_conf = self._reconstruct(ep["trace"])
        return {
            "episode": recon,
            "confidence": round(conf, 3),
            "key": ep["key"],
            "slot_conf": slot_conf,
        }

    # ---- gist / schema -----------------------------------------------------------------
    def gist(self, cue, m=6):
        """Extract the schema (INVARIANT slots) shared across the top-m episodes a cue retrieves.
        Bundle the family's traces into a prototype, unbind each role, and keep a slot only when its
        cleanup margin clears gist_thresh (invariant across the family reinforces; variable slots
        destructively interfere and are dropped). gist_thresh (0.30) is a majority-reinforcement
        floor: an invariant slot is present in EVERY family trace so its prototype-unbind coincidence
        is large, while a varying / single-instance slot sits near the noise floor. A family of <2
        has NO gist (empty schema) -- a schema needs a population. Returns {"schema","margins","n"}."""
        cue_trace = self._trace(cue, register=False)
        if cue_trace is None or not self._episodes:
            return {"schema": {}, "margins": {}, "n": 0}
        fam = [ep for ep, _s in self._retrieve(cue_trace, k=min(m, len(self._episodes)))]
        if len(fam) < 2:
            return {"schema": {}, "margins": {}, "n": len(fam)}
        proto = bundle([ep["trace"] for ep in fam])
        schema, margins = {}, {}
        for slot, role in self._role.items():
            book = self._codebook.get(slot)
            if not book:
                continue
            label, s = cleanup(bind(proto, role), book)
            margins[slot] = round(s, 3)
            if s >= self.gist_thresh:  # invariant across the family -> part of the schema
                schema[slot] = label
        return {"schema": schema, "margins": margins, "n": len(fam)}

    # ---- narration ---------------------------------------------------------------------
    @staticmethod
    def narrate(episode):
        """A short ASCII autobiographical sentence from an episode's slots. Template fill over the
        recovered slots -- HONEST: this is not generative language, just a slot-into-frame render."""
        who = episode.get("who", "I")
        what = episode.get("what", "was there")
        where = episode.get("where")
        when = episode.get("when")
        affect = episode.get("affect")
        subj = "I" if who in (None, "self", "I", "me") else str(who)
        verb = "remember" if subj == "I" else "recalls"
        parts = [f"{subj} {verb} {what}"]
        if where is not None:
            parts.append(f"at {where}")
        if when is not None:
            parts.append(f"during {when}")
        sentence = " ".join(parts)
        if affect is not None:
            sentence += f"; it felt {affect}"
        return sentence + "."


# ------------------------------------------------------------------ self-check --------------------
def _completion_check(rng):
    """Store distinct episodes; from a PARTIAL cue recover the HELD-OUT slots. Real cue vs a
    RANDOM-cue control (values never stored -> retrieval near chance). Returns (real, ctrl, n)."""
    er = EpisodicRecall(seed=1)
    # 12 distinct episodes; every slot value is unique per episode so a held-out slot recovered
    # correctly can only come from retrieving the RIGHT episode (no accidental sharing).
    episodes = []
    for i in range(12):
        episodes.append(
            {
                "who": f"agent{i}",
                "what": f"grasp{i}",
                "where": f"room{i}",
                "when": f"t{i}",
                "affect": ("fear" if i == 3 else f"mood{i}"),
            }
        )
        er.encode(episodes[-1])
    # cue = where+when (unique locators); HELD OUT = who/what/affect -> must be reconstructed.
    held = ("who", "what", "affect")
    real_hits = real_tot = 0
    for ep in episodes:
        cue = {"where": ep["where"], "when": ep["when"]}
        recon = er.complete(cue)["episode"]
        for h in held:
            real_tot += 1
            real_hits += recon.get(h) == ep[h]
    # RANDOM-cue control: cue slots hold values that were NEVER stored -> retrieval is arbitrary,
    # so held-out recovery collapses to chance (distinct fillers -> ~0).
    ctrl_hits = ctrl_tot = 0
    for ep in episodes:
        cue = {"where": f"nowhere{rng.randrange(10**6)}", "when": f"never{rng.randrange(10**6)}"}
        recon = er.complete(cue)["episode"]
        for h in held:
            ctrl_tot += 1
            ctrl_hits += recon.get(h) == ep[h]
    # affect-cue demo: "the fearful one" -> reconstruct that episode's other slots.
    fear = er.complete({"affect": "fear"})
    return real_hits / real_tot, ctrl_hits / ctrl_tot, len(episodes), er, fear


def _noise_check(er, rng):
    """Graceful degradation: corrupt the CUE trace by flipping a fraction of spikes and measure
    held-out recovery. Returns [(frac, recovery)] over increasing noise."""
    episodes = [
        {
            "who": f"agent{i}",
            "what": f"grasp{i}",
            "where": f"room{i}",
            "when": f"t{i}",
            "affect": (f"mood{i}"),
        }
        for i in range(12)
    ]
    curve = []
    for frac in (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5):
        hits = tot = 0
        for ep in episodes:
            clean_cue = er._trace({"where": ep["where"], "when": ep["when"]}, register=False)
            noisy = _flip(clean_cue, frac, rng)
            recon = er.complete(None, cue_trace=noisy)["episode"]
            for h in ("who", "what", "affect"):
                tot += 1
                hits += recon.get(h) == ep[h]
        curve.append((frac, hits / tot))
    return curve


def _gist_check():
    """Schema across a FAMILY (invariant slots survive bundling) vs two controls: a RANDOM family
    (no invariant slot) and a SINGLE episode (no gist). Returns the three result dicts + margins."""
    # (schema) 6 'kitchen breakfast' episodes: where + who INVARIANT; what/when/affect all VARY
    # (every varying slot has 6 DISTINCT values, so only where/who can reinforce across the family).
    sch = EpisodicRecall(seed=2)
    foods = ("toast", "coffee", "eggs", "cereal", "fruit", "bagel")
    feels = ("calm", "tired", "happy", "rushed", "sleepy", "content")
    for i, f in enumerate(foods):
        sch.encode(
            {"who": "self", "what": f, "where": "kitchen", "when": f"day{i}", "affect": feels[i]}
        )
    g_schema = sch.gist({"where": "kitchen"}, m=6)
    # (random control) 8 fully-random episodes: NO slot is shared -> empty schema.
    rnd = EpisodicRecall(seed=3)
    for i in range(8):
        rnd.encode(
            {"who": f"w{i}", "what": f"x{i}", "where": f"p{i}", "when": f"q{i}", "affect": f"a{i}"}
        )
    g_random = rnd.gist({"where": "p0"}, m=6)
    # (single-episode control) one episode -> no gist (n<2).
    one = EpisodicRecall(seed=4)
    one.encode(
        {"who": "self", "what": "toast", "where": "kitchen", "when": "day0", "affect": "calm"}
    )
    g_single = one.gist({"where": "kitchen"}, m=6)
    return g_schema, g_random, g_single


def main():
    print("=" * 78)
    print("RECONSTRUCTIVE / AUTOBIOGRAPHICAL EPISODIC MEMORY (VSA pattern completion + gist)")
    print("=" * 78)
    rng = random.Random(0)

    # (a) PATTERN COMPLETION vs RANDOM-CUE CONTROL ------------------------------------------------
    real, ctrl, n, er, fear = _completion_check(rng)
    chance = 1.0 / n
    print("-" * 78)
    print("(a) COMPLETION: reconstruct 3 HELD-OUT slots (who/what/affect) from a where+when cue")
    print(f"    real partial-cue recovery : {real:6.1%}  ({n} episodes)")
    print(f"    random-cue control        : {ctrl:6.1%}  (chance ~ {chance:5.1%})")
    fe = fear["episode"]
    print(
        f"    'the fearful one' -> who={fe.get('who')} what={fe.get('what')} "
        f"where={fe.get('where')} (conf {fear['confidence']})"
    )
    print(f"    narrate: {er.narrate(fe)}")

    # (b) GRACEFUL DEGRADATION WITH NOISE ---------------------------------------------------------
    curve = _noise_check(er, rng)
    print("-" * 78)
    print("(b) NOISE: held-out recovery vs fraction of cue spikes flipped (graceful decline)")
    print("    " + "  ".join(f"f={f:.2f}:{r:5.1%}" for f, r in curve))

    # (c) GIST / SCHEMA vs RANDOM-FAMILY and SINGLE-EPISODE controls -------------------------------
    g_schema, g_random, g_single = _gist_check()
    inv_margins = [g_schema["margins"].get(s, 0.0) for s in ("where", "who")]
    var_margins = [g_schema["margins"].get(s, 0.0) for s in ("what", "when", "affect")]
    print("-" * 78)
    print("(c) GIST: schema across a family (invariant slots) vs random-family / single controls")
    print(f"    schema family   -> schema={g_schema['schema']}  margins={g_schema['margins']}")
    print(f"    random family   -> schema={g_random['schema']} (empty)  n={g_random['n']}")
    print(f"    single episode  -> schema={g_single['schema']} (no gist)  n={g_single['n']}")

    # ---- ASSERTS: real metric + un-riggable FAILING controls ------------------------------------
    assert real >= 0.95, f"pattern completion must recover held-out slots cleanly, got {real:.2%}"
    assert real > ctrl + 0.5, (
        f"real recovery must dominate the random-cue control ({real:.2%} vs {ctrl:.2%})"
    )
    assert ctrl <= chance + 0.05, f"random-cue control must sit near chance, got {ctrl:.2%}"
    assert fe.get("affect") == "fear", "'the fearful one' cue must reconstruct the fear episode"
    # graceful: clean is high, degrades monotonically-ish, still beats chance at moderate noise,
    # and collapses toward chance at heavy noise (a graceful curve, not a cliff at low noise).
    r0, r2, r_hi = curve[0][1], curve[2][1], curve[-1][1]
    assert r0 >= 0.95, f"zero-noise recovery must be clean, got {r0:.2%}"
    assert r2 > chance + 0.1, f"recovery must beat chance at moderate (0.20) noise, got {r2:.2%}"
    assert r0 >= r_hi, "recovery must not increase with noise (graceful degradation)"
    # gist: invariant slots enter the schema, variable slots do not; random family has no schema;
    # a single episode yields no gist. The non-riggable metric is the invariant-vs-variable margin gap.
    assert set(g_schema["schema"]) == {"where", "who"}, (
        f"gist must keep exactly the invariant slots, got {set(g_schema['schema'])}"
    )
    assert min(inv_margins) > max(var_margins) + 0.15, (
        f"invariant margins {inv_margins} must dominate variable margins {var_margins}"
    )
    assert g_random["schema"] == {}, f"a random family must have NO gist, got {g_random['schema']}"
    assert g_single["schema"] == {} and g_single["n"] == 1, "a single episode must yield no gist"
    # determinism
    a = EpisodicRecall(seed=9)
    a.encode({"who": "self", "what": "toast", "where": "kitchen"})
    b = EpisodicRecall(seed=9)
    b.encode({"who": "self", "what": "toast", "where": "kitchen"})
    assert a.complete({"where": "kitchen"}) == b.complete({"where": "kitchen"}), (
        "must be deterministic"
    )

    print("-" * 78)
    print(
        f"self-check OK: pattern completion recovers held-out slots {real:.0%} vs {ctrl:.0%} "
        f"random-cue control (chance {chance:.0%}); degrades gracefully with noise "
        f"({r0:.0%}->{r_hi:.0%}); gist keeps invariant slots {sorted(g_schema['schema'])} "
        f"(margin gap {min(inv_margins) - max(var_margins):.2f}) while random-family and "
        f"single-episode controls have no gist; VSA one-shot associative memory, no gradients; "
        f"safety-inert."
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    main()
