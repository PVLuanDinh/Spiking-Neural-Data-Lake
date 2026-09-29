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

GENERALIZATION (apply_schema -- v3.79 addition). Storing instances and reading their gist is still
only memory ABOUT what was stored. `apply_schema(partial_situation)` closes the transfer gap: it
(1) finds the episode FAMILY a partial, possibly NEVER-STORED situation belongs to, (2) extracts
that family's schema (the invariant slots), and (3) PREDICTS the situation's missing slots from the
schema. The predicted episode need not exist anywhere in the store -- that is the point: the schema,
not the instance, carries the answer (Bartlett 1932 schema; Posner & Keele 1968 prototype
abstraction; McClelland/McNaughton/O'Reilly 1995 complementary-learning-systems semantic extraction).

  family(cue)  = stored episodes agreeing with the MOST of the cue's GROUNDED slots
                 (agreement = unbind that role from the trace, check spike coincidence with the
                  cue's filler code >= family_thresh)
  schema       = invariant slots of that family (the gist machinery above)
  prediction   = schema slots the situation did NOT supply

HONESTY of the generalization path (this is where a memory module is most tempted to fabricate):
  - a value never stored cannot ADDRESS anything, so UNGROUNDED cue slots are ignored entirely; a
    situation with no grounded slot has NO family -> empty prediction at confidence 0.0. We never
    fall back on a nearest-anything.
  - a family smaller than min_family (default 2) cannot support a schema -> 0.0.
  - a family that is too VARIABLE abstracts nothing beyond the cue itself -> empty prediction at
    0.0 rather than a plausible-looking guess.
  - an AMBIGUOUS cue spanning two families with conflicting values in a slot leaves that slot OUT
    (the conflicting bindings destructively interfere below the gist margin) -- it does not pick a
    winner. All four of these are kept as genuinely failing controls in the self-check.
  - `confidence` is the mean prototype-unbind spike-coincidence margin of the PREDICTED slots. It
    is a retrieval margin, NOT a calibrated probability -- do not read it as P(correct).

LEARNING RULE. VSA role-filler binding + bundle superposition = a one-shot Hebbian/attractor
associative memory. There is NO gradient anywhere (no loss.backward, no BPTT): storage is a single
superposition write, recall is fixed spike-algebra + nearest-code cleanup. So the e-prop-vs-BPTT
cosine gate does not apply (nothing here learns by gradient).

SAFETY / SCOPE. Representation/value-side ONLY. This module NEVER selects an action, NEVER calls the
arm, NEVER emits a motor command, and imports NOTHING from brainstem/ or any safety-envelope file.
It reuses cortex/vsa_reasoning (pure spike algebra). Additive/opt-in: nothing runs at import; work
happens only under explicit calls or __main__.

THRESHOLDS ARE DERIVED, NOT FROZEN. Both decision margins used below are COMPUTED from the store's
own geometry at call time -- there is no tuned constant left. Two analytic quantities do all the
work (see _superposition_margin / _prototype_margin, both MEASURED against the substrate):

  E(n) = C(n-1, floor((n-1)/2)) / 2^(n-1)
      the coincidence at which ONE component survives a superposition of n. bundle is a majority
      WTA, so the target decides a dimension only when the other n-1 are exactly tied. E(1)=1.0,
      E(5)=0.375, E(9)=0.273. (An earlier revision of this docstring quoted 0.25 at S=9; the
      formula and the substrate both say 0.273 -- measured 0.272 over 200 trials at D=512.)
  sigma = 1/sqrt(D)
      the standard deviation of the centred spike coincidence between two UNRELATED D-dim codes
      (measured 0.0452 at D=512 vs 1/sqrt(512)=0.0442).

Each threshold is then N_SIGMA sigmas above ITS OWN floor, capped halfway to ITS OWN signal:

  membership (_family_thresh)  floor 0            signal E(S)               -- S = that episode's
                                                                              own binding count
  schema     (_gist_thresh)    floor E(S)*E(m)    signal two-level majority -- m = family size

The two floors differ and that is the whole point: a NON-member's unbind is unrelated to the cue's
filler so it is zero-mean noise, but a slot that merely VARIES across a family still occurs in one
of the m members, so it lands at E(S)*E(m) -- 0.117 at S=5,m=6, nearly three times the raw noise.
This is why the retired frozen pair was 0.15 and 0.30 rather than one number: they were the SAME
3.4-sigma rule measured against two different floors, one of which was never written down. Both
old constants are recovered by the derivation at the size they were tuned for (S=5, m=6, D=512):
0.150 vs 0.15, and 0.267 vs 0.30. Away from that size they diverge, correctly -- see caveat (4).

What that divergence is worth, MEASURED (300 stores per row, D=512): the rate at which a GENUINE
member of an episode is judged a non-member, under the frozen 0.15 versus the derived threshold.

  S       5      9     11     14     18     25
  frozen  0.0%   0.3%   1.5%   8.2%  21.3%  40.4%
  derived 0.0%   0.1%   0.2%   0.6%   1.5%   3.3%

At the tuned size the two are indistinguishable (0.0% / 0.0%) -- this change is not a re-tuning and
buys nothing there. The gap opens exactly where the docstring said it would: by S=14 the frozen
constant sits at 72% of the signal it is supposed to sit safely below, and silently discards one
member in twelve. The derived value tracks the signal down and holds the miss rate near zero. Note
the derived rate is NOT zero and grows too -- that is the substrate's finite capacity (caveat 4)
showing through honestly, not something a threshold can remove.

HONEST CAVEATS. (1) The role slots (who/what/where/when/affect) are DEVELOPER-DEFINED, not
discovered. (2) Reconstruction is RETRIEVAL over STORED episodes + cleanup, NOT generative
imagination -- it can only complete to an episode it actually holds (or fail honestly with low
confidence). (3) Fillers must be in the slot codebook for cleanup to name them; a novel cue value
retrieves best-effort and reports low confidence. (4) Capacity is finite: per-episode traces
superpose ~S slot-bindings and cross-episode retrieval is a coincidence scan, so recovery degrades
as slots/episodes grow -- exactly as a spiking associative memory should. The thresholds now TRACK
that decay instead of ignoring it (E(S) falls as S grows, and the halfway cap pulls the threshold
down with it), but tracking a collapsing signal is not the same as preventing it: when a family's
signal falls to its own floor there is no threshold that separates them and the schema degrades
regardless. The derivation makes the ceiling visible; it does not raise it.

Determinism: crc32-seeded atomic spike codes (never builtin hash); per-instance random.Random(seed)
for the noise/self-check only; ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.episodic_recall
"""

import math
import random
import zlib

import numpy as np

from snn_data_lake.cortex.vsa_reasoning import (
    D,
    bind,
    bundle,
    cleanup,
    coincidence_many,
    hv,
    sim,
)


class _RowStore:
    """Append-only float32 row buffer with capacity doubling. Two users: the episode traces, and
    each slot's filler codebook.

    Both are append-only and never evicted, so rows sit in INSERTION order and `view()` hands back
    the filled prefix in that order -- which is what makes a fused argmax break ties on the
    first-stored row, exactly as `max()` and a stable sort do over the scalar counter.

    It appends rather than rebuilding on purpose. The `when` codebook gains one row per episode
    (`engine/brain.Brain._w8_tick` fills that slot with `str(self._wake_ticks)`, always novel), so
    rebuilding the matrix per write would be O(rows*D) per write -- reintroducing the very per-tick
    cost this replaces. (Cited as `brain.py:1680` until 2026-08-04, which had drifted onto an
    interval-timing comment; named by SYMBOL now.)
    """

    __slots__ = ("_buf", "_n")

    def __init__(self, width, capacity=64):
        self._buf = np.zeros((capacity, width), dtype=np.float32)
        self._n = 0

    def __len__(self):
        return self._n

    def add(self, vec):
        if self._n == self._buf.shape[0]:
            bigger = np.zeros((2 * self._buf.shape[0], self._buf.shape[1]), dtype=np.float32)
            bigger[: self._n] = self._buf
            self._buf = bigger
        self._buf[self._n] = vec
        self._n += 1

    def view(self):
        return self._buf[: self._n]


# The developer-defined role slots (HONEST: these are hand-specified, not induced). An episode dict
# may supply any subset; unknown keys are accepted and get their own role vector on first sight.
DEFAULT_SLOTS = ("who", "what", "where", "when", "affect")

# Detection margin in units of the code's analytic noise sigma, shared by BOTH thresholds. This is
# the only free number left in the module, and it is not a new one: the retired frozen
# family_thresh was 0.15 at D=512, i.e. 0.15*sqrt(512) = 3.39 sigma, and this module's own docstring
# already named that design margin ("0.15 is ~3.4 sigma above noise"). It is INHERITED, not re-tuned
# -- the evidence that it is the right number is that applying the SAME 3.4 to the (previously
# unnamed) schema floor independently reproduces the other frozen constant, 0.267 vs 0.30.
N_SIGMA = 3.4


def _superposition_margin(n):
    """E(n) = C(n-1, floor((n-1)/2)) / 2^(n-1): the expected spike coincidence at which ONE
    component is recovered by unbinding it back out of a superposition of n role-filler bindings.

    Derivation: bundle is a per-dimension majority WTA over the n components. The target binding
    decides a dimension ONLY when the other n-1 are exactly tied (probability C(n-1,(n-1)/2)/2^(n-1)
    ); otherwise they outvote it and the outcome carries no information about the target. n=1 gives
    1.0 (a lone binding unbinds exactly), n=5 gives 0.375, n=9 gives 0.273.

    MEASURED against the substrate (200 trials, D=512): 1.000 / 0.376 / 0.315 / 0.272 / 0.227 at
    n = 1 / 5 / 7 / 9 / 13, versus predicted 1.000 / 0.375 / 0.313 / 0.273 / 0.226."""
    k = max(0, int(n) - 1)
    return math.comb(k, k // 2) / float(2**k)


def _prototype_margin(m, n_slots):
    """Expected coincidence of a filler that is INVARIANT across all m traces of a family, unbound
    out of their bundled prototype. A TWO-level majority: each trace agrees with the invariant
    binding per dimension at p = (1 + E(n_slots))/2, and the prototype is the majority of m such
    votes. bundle breaks an even tie on the UP line, which agrees with the target on half the
    dimensions, so a tie contributes 1/2.

    MEASURED (200 trials, D=512, n_slots=5): 0.374 / 0.533 / 0.640 / 0.713 / 0.814 at m = 2 / 4 /
    6 / 8 / 12, versus predicted 0.375 / 0.533 / 0.641 / 0.713 / 0.814."""
    p = (1.0 + _superposition_margin(n_slots)) / 2.0
    agree = 0.0
    for k in range(int(m) + 1):
        pk = math.comb(int(m), k) * (p**k) * ((1.0 - p) ** (int(m) - k))
        if 2 * k > m:
            agree += pk
        elif 2 * k == m:
            agree += 0.5 * pk
    return 2.0 * agree - 1.0


def _analytic_thresh(floor, signal, d_width, n_sigma=N_SIGMA):
    """A decision threshold n_sigma analytic sigmas above `floor`, never further than HALFWAY from
    `floor` to `signal`. sigma = 1/sqrt(D) is the standard deviation of the centred spike
    coincidence between two unrelated D-dimensional codes (measured 0.0452 at D=512).

    The halfway cap is not a safety blanket, it is load-bearing: without it a 9-slot episode in a
    2-episode family would be judged at 0.287 against a signal of 0.273 and EVERY genuinely
    invariant slot would be silently rejected. With it the threshold stays strictly inside
    (floor, signal) for every (S, m) the store can produce, degrading with the signal instead of
    crossing it."""
    sigma = 1.0 / math.sqrt(max(1, int(d_width)))
    return min(floor + n_sigma * sigma, 0.5 * (floor + signal))


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
      apply_schema(partial_situ)  -> result dict : GENERALIZATION -- find the family a NEW (possibly
                                                    never-stored) situation belongs to, extract its
                                                    schema, and PREDICT the situation's missing
                                                    slots. Honest empty at 0.0 when no coherent
                                                    family / nothing to abstract. Keys of the
                                                    result: {"prediction","schema","confidence",
                                                    "n_family"}.
      narrate(episode_dict)       -> str         : a short autobiographical sentence (template fill
                                                    over the slots -- NOT generative language).

    Deterministic, safety-inert (never selects/executes an action)."""

    def __init__(
        self,
        slots=DEFAULT_SLOTS,
        seed=0,
        schema_consensus=1.0,
    ):
        # NOTE: the former gist_thresh=0.30 / family_thresh=0.15 arguments are GONE, not defaulted.
        # Both are now computed per call from the store's own geometry (slot count, family size,
        # code width) -- see _family_thresh / _gist_thresh. There is nothing left to pass.
        self.seed = int(seed)
        # fraction of family members that must INDIVIDUALLY reconstruct a candidate schema slot to
        # the same filler. 1.0 = unanimous = the literal definition of "invariant across the
        # family". See _schema_of: the prototype margin alone was measured to invent a winner for a
        # 6/6 split, so this gate is a correctness fix, not a knob.
        self.schema_consensus = float(schema_consensus)
        self._role = {}  # slot -> role spike code (fixed, crc32-seeded)
        self._codebook = {}  # slot -> {value_str: filler spike code}
        self._episodes = []  # list of {"key": crc32, "trace": spike code, "S": n bindings}
        # Row-major mirrors of the two things retrieval scans, so a whole scan is one GEMV instead of
        # one Python loop per stored item per wake tick. They are mirrors, never the source of truth:
        # _episodes and _codebook stay authoritative, and each mirror is appended at the SINGLE site
        # that appends to its source (encode for traces, _filler_code for fillers) so they cannot
        # desync.
        self._trace_rows = _RowStore(D)
        self._book_rows = {}  # slot -> _RowStore of filler codes, in codebook insertion order
        self._book_labels = {}  # slot -> [value_str], parallel to _book_rows
        for s in slots:
            self._ensure_slot(s)

    # ---- analytic thresholds (derived per call; nothing frozen, nothing learned) ---------
    def _slots_of(self, ep):
        """S: the number of role-filler bindings superposed in a stored episode's trace.

        BACK-COMPAT (explicit, not a crash and not a silent guess): records written before this
        field existed carry no "S". We substitute len(self._role) -- the number of registered role
        slots -- which is a strict UPPER bound on any episode's binding count, since every slot an
        episode uses is registered by _ensure_slot before its trace is built. An upper bound on S is
        a LOWER bound on the expected signal E(S), so a legacy record is judged by the most
        conservative threshold the derivation can justify rather than an optimistic one. For a store
        using the 5 DEFAULT_SLOTS this reproduces the retired frozen behaviour exactly (S=5 ->
        family threshold 0.150 vs the old 0.15, which admit the identical set of episodes because
        no attainable coincidence value falls between them)."""
        s = ep.get("S")
        return int(s) if s is not None else len(self._role)

    def _family_thresh(self, ep):
        """Membership threshold for ONE stored episode: is this episode's filler for a cue slot the
        one the cue named? The comparison is a sim against a SINGLE named filler code, so a
        non-member is zero-mean noise -- floor 0.0 -- and the signal is that episode's own E(S).
        Per-episode, so a 12-slot episode and a 3-slot episode in the same store are each judged at
        their own capacity."""
        return _analytic_thresh(0.0, _superposition_margin(self._slots_of(ep)), len(ep["trace"]))

    def _gist_thresh(self, fam):
        """Schema threshold for ONE family prototype. The floor here is NOT the raw noise floor: a
        slot that merely VARIES across the family still contributes its filler to one of the m
        traces, so it survives the two-level bundle at E(S)*E(m) -- 0.117 at S=5,m=6, versus 0.044
        for pure noise. Judging a varying slot against pure noise would let it through; judging it
        against its own floor is what the retired 0.30 was implicitly doing.

        S is taken as the MOST loaded member of the family: the threshold has to stay below the
        WEAKEST member's signal, because a member whose signal fell under it would be dropped from
        the schema with no way to recover it. That errs permissive, which is the safe direction here
        -- the per-member consensus gate in _schema_of is what guards the permissive side."""
        m = len(fam)
        n_slots = max(self._slots_of(ep) for ep in fam)
        floor = _superposition_margin(n_slots) * _superposition_margin(m)
        return _analytic_thresh(floor, _prototype_margin(m, n_slots), len(fam[0]["trace"]))

    def __len__(self):
        return len(self._episodes)

    # ---- vocabulary -------------------------------------------------------------------
    def _ensure_slot(self, slot):
        """Register a role vector + empty codebook for a slot on first sight (idempotent)."""
        if slot not in self._role:
            self._role[slot] = hv(f"ROLE|{slot}")
            self._codebook[slot] = {}
            self._book_rows[slot] = _RowStore(D)
            self._book_labels[slot] = []

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
        if register and vs not in self._codebook[slot]:
            # SINGLE codebook append site -> the only place the mirror can fall behind. Guarded on
            # absence because the original assignment was an idempotent overwrite (hv is
            # deterministic, so re-registering a value rewrites the identical code) and appending a
            # duplicate row would break the label<->row correspondence.
            self._codebook[slot][vs] = code
            self._book_rows[slot].add(code)
            self._book_labels[slot].append(vs)
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
        superposed spike code; every filler value enters the slot codebook (semantic vocabulary).

        The record also keeps S, the number of bindings the trace superposes. S is knowable ONLY
        here (the trace itself is a flat spike code that does not reveal how many components went
        into it), and every analytic threshold downstream needs it -- storing it is what lets the
        thresholds track capacity instead of assuming it."""
        for s in episode:
            self._ensure_slot(s)
        trace = self._trace(episode, register=True)
        if trace is None:
            raise ValueError("cannot encode an empty episode")
        # same predicate _trace bundles on -- the count of role-filler bindings actually superposed
        n_slots = sum(1 for v in episode.values() if v is not None)
        raw = "|".join(f"{s}={self._valstr(episode[s])}" for s in sorted(episode)).encode("ascii")
        key = zlib.crc32(raw)
        self._episodes.append({"key": key, "trace": trace, "S": n_slots})
        self._trace_rows.add(trace)  # mirror; single append site, same order as _episodes
        return key

    # ---- retrieval + reconstruction ----------------------------------------------------
    def _retrieve(self, cue_trace, k=1):
        """Rank stored episodes by spike coincidence with the cue trace (content-addressable).
        Returns the top-k [(episode_record, sim)] most coincident, most-similar first."""
        if not self._episodes:
            return []
        # One GEMV over the trace mirror, replacing one Python coincidence loop per stored episode
        # per call. This was measured at 35.3% of the live wake tick on its own.
        # Tie-identical to the sort it replaces: counts are integers held exactly in float32, and
        # argsort(-counts, kind="stable") ranks equal counts by insertion order, which is what a
        # stable sort(key=..., reverse=True) did. NOT argsort(counts)[::-1] -- that reverses ties too.
        counts = coincidence_many(cue_trace, self._trace_rows.view())
        order = np.argsort(-counts, kind="stable")[:k]
        return [(self._episodes[int(i)], (2.0 * float(counts[i]) - D) / D) for i in order]

    def _reconstruct(self, trace):
        """UNBIND every role from a trace and CLEAN UP to the nearest filler -> full slot dict, plus
        a per-slot spike-coincidence confidence. This is the reconstructive step: held-out slots are
        recovered from the STORED trace, not from the cue."""
        recon, conf = {}, {}
        for slot, role in self._role.items():
            book = self._codebook.get(slot)
            if not book:
                continue  # nothing ever stored in this slot -> cannot name it
            # bind is self-inverse -> this unbinds. One GEMV per slot over that slot's codebook
            # mirror, replacing cleanup()'s Python max() over every stored filler: measured 29.9% of
            # the live wake tick. argmax takes the first maximum, exactly as max() does.
            counts = coincidence_many(bind(trace, role), self._book_rows[slot].view())
            i = int(np.argmax(counts))
            recon[slot] = self._book_labels[slot][i]
            conf[slot] = round((2.0 * float(counts[i]) - D) / D, 3)
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
        cleanup margin clears the DERIVED schema threshold (invariant across the family reinforces;
        variable slots destructively interfere and are dropped). That threshold is computed per
        family by _gist_thresh from the family size m and the members' slot load S -- an invariant
        slot is present in EVERY family trace so its prototype-unbind coincidence is large, while a
        varying slot falls back to the E(S)*E(m) one-of-m floor. A family of <2 has NO gist (empty
        schema) -- a schema needs a population. Returns {"schema","margins","n"}."""
        cue_trace = self._trace(cue, register=False)
        if cue_trace is None or not self._episodes:
            return {"schema": {}, "margins": {}, "n": 0}
        fam = [ep for ep, _s in self._retrieve(cue_trace, k=min(m, len(self._episodes)))]
        if len(fam) < 2:
            return {"schema": {}, "margins": {}, "n": len(fam)}
        schema, margins = self._schema_of(fam)
        return {"schema": schema, "margins": margins, "n": len(fam)}

    def _schema_of(self, fam):
        """The INVARIANT slots of an episode family, behind TWO gates.

        Gate 1 (prototype margin): bundle the family's traces into a prototype, unbind each role,
        clean up to the nearest filler, keep the slot only if its spike-coincidence margin clears
        the family's DERIVED schema threshold (_gist_thresh) -- an invariant filler reinforces
        across the superposition, a varying one destructively interferes down to the E(S)*E(m)
        one-of-m floor.

        Gate 2 (per-member CONSENSUS): the candidate label must ALSO be what at least
        schema_consensus of the family members individually reconstruct in that slot. This gate is
        not decoration -- it exists because gate 1 alone was MEASURED to fabricate. On a cue
        spanning two families that split 6/6 on `who` (alice vs bob), the bundle's majority WTA
        (ties -> up line) does not fully cancel, and the prototype cleaned up to `bob` at margin
        0.367 -- clear of the gate-1 threshold whether that is the retired frozen 0.30 or the
        derived 0.235 for this family (m=12, S=5): a confident schema slot invented out of a dead
        tie. No threshold fixes this, which is exactly why the fix is not a threshold. Consensus
        rejects it (6/12 members agree), which is the actual DEFINITION of invariant-across-family.
        Default 1.0 (unanimous): we prefer an honest missing slot over a fabricated one.

        Returns (schema, margins) -- the shared core of gist() and apply_schema()."""
        proto = bundle([ep["trace"] for ep in fam])
        thresh = self._gist_thresh(fam)  # derived from this family's size and slot load
        schema, margins = {}, {}
        for slot, role in self._role.items():
            book = self._codebook.get(slot)
            if not book:
                continue
            label, s = cleanup(bind(proto, role), book)
            margins[slot] = round(s, 3)
            if s < thresh:
                continue  # varies across the family (or absent) -> not part of the schema
            agree = sum(1 for ep in fam if cleanup(bind(ep["trace"], role), book)[0] == label)
            if agree >= self.schema_consensus * len(fam):
                schema[slot] = label
        return schema, margins

    # ---- schema-driven GENERALIZATION (transfer to a NEVER-STORED situation) -------------
    def _family_for(self, situation):
        """Find the episode FAMILY a (possibly NEW) partial situation belongs to.

        Only GROUNDED cue slots participate -- a filler value that was never stored cannot address
        anything in the trace store, so we drop it rather than let it drag retrieval somewhere
        plausible-looking (never-fabricate). An episode AGREES with a grounded cue slot when
        unbinding that role from its trace is spike-coincident with the cue's filler code above
        family_thresh. The family is the set of episodes agreeing on the MAXIMAL number of cue
        slots (so a 2-slot cue prefers episodes matching both over episodes matching one).

        Returns (family_list, n_agreeing_slots, n_grounded_cue_slots); ([], 0, 0) when the situation
        is empty, the store is empty, or NO cue slot is grounded."""
        if not situation or not self._episodes:
            return [], 0, 0
        grounded = [
            (s, self._filler_code(s, v))
            for s, v in situation.items()
            if v is not None and s in self._role and self._valstr(v) in self._codebook.get(s, {})
        ]
        if not grounded:
            return [], 0, 0  # nothing in this situation was ever observed -> no family, honestly
        scored = []
        for ep in self._episodes:
            thresh = self._family_thresh(ep)  # derived from THIS episode's slot load and code width
            agree = sum(
                1 for s, code in grounded if sim(bind(ep["trace"], self._role[s]), code) >= thresh
            )
            scored.append((agree, ep))
        best = max(a for a, _ep in scored)
        if best == 0:
            return [], 0, len(grounded)
        return [ep for a, ep in scored if a == best], best, len(grounded)

    def apply_schema(self, partial_situation, min_family=2):
        """GENERALIZE: predict the missing slots of a NEW situation from its family's SCHEMA.

        This is the transfer step, not a lookup: the situation being completed need never have been
        stored. (1) find the family via _family_for, (2) abstract its invariant slots via _schema_of,
        (3) return the schema slots the situation did NOT supply as the PREDICTION.

        Returns {"prediction","schema","confidence","n_family"}. `confidence` is the mean
        prototype-unbind spike-coincidence margin over the predicted slots -- a retrieval margin,
        NOT a calibrated probability.

        HONEST FAILURE (all four return an EMPTY prediction at confidence 0.0 -- we never guess a
        nearest-anything): no grounded cue slot (n_family 0); a family below min_family; a family
        too variable to abstract anything the cue did not already supply; and, for an ambiguous cue
        spanning two families, the conflicting slot is simply left out of the schema."""
        fam, _agree, _n_grounded = self._family_for(partial_situation)
        if len(fam) < max(2, int(min_family)):
            # too small (or absent) to separate typical from idiosyncratic -> no schema, no guess
            return {"prediction": {}, "schema": {}, "confidence": 0.0, "n_family": len(fam)}
        schema, margins = self._schema_of(fam)
        supplied = {s for s, v in partial_situation.items() if v is not None}
        prediction = {s: label for s, label in schema.items() if s not in supplied}
        if not prediction:
            # a family exists but it abstracts NOTHING beyond what the cue already gave us.
            return {"prediction": {}, "schema": schema, "confidence": 0.0, "n_family": len(fam)}
        conf = sum(max(0.0, min(1.0, margins[s])) for s in prediction) / len(prediction)
        return {
            "prediction": prediction,
            "schema": schema,
            "confidence": round(conf, 3),
            "n_family": len(fam),
        }

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


def _generalization_check():
    """SCHEMA-DRIVEN GENERALIZATION to episodes that were NEVER stored.

    Setup: a coherent family (who/where/when INVARIANT, what/affect VARY) buried among 12 unrelated
    distractor episodes. Each probe is a NEW situation whose `what` value was never stored -- the
    full episode it names is asserted ABSENT from the store -- so a correct prediction of the
    invariant slots can only come from the family SCHEMA, not from retrieving the episode itself.

    Three controls that must genuinely FAIL (kept as-is, not tuned away):
      (i)   NO-INVARIANT family : 8 episodes sharing only the cue slot -> nothing to abstract
                                  beyond the cue -> empty prediction at 0.0.
      (ii)  UNGROUNDED situation: no cue value was ever stored -> no family at all -> 0.0.
      (iii) AMBIGUOUS cue       : the cue spans two families that CONFLICT on `who` (6 alice /
                                  6 bob) -> `who` must be left OUT rather than a winner picked.

    Returns (accuracy, n_probes, first_result, c_noinv, c_nofam, c_amb).
    """
    invariants = {"who": "self", "where": "kitchen", "when": "morning"}
    foods = ("toast", "coffee", "eggs", "cereal", "fruit", "bagel", "porridge", "waffle")
    feels = ("calm", "tired", "happy", "rushed", "sleepy", "content", "warm", "bright")
    gen = EpisodicRecall(seed=5)
    for i in range(8):
        gen.encode({**invariants, "what": foods[i], "affect": feels[i]})
    for i in range(12):  # distractor episodes -- unrelated, share nothing with the family
        gen.encode(
            {"who": f"w{i}", "what": f"x{i}", "where": f"p{i}", "when": f"q{i}", "affect": f"a{i}"}
        )
    stored = {ep["key"] for ep in gen._episodes}
    # probe `what` values that were NEVER stored -> the probed episode is genuinely unseen
    novel = ("pancakes", "omelette", "granola", "muffin", "scone", "crepe", "grits", "congee")
    hits, first = 0, None
    for w in novel:
        target = {**invariants, "what": w}
        raw = "|".join(f"{s}={target[s]}" for s in sorted(target)).encode("ascii")
        assert zlib.crc32(raw) not in stored, "probe episode must NOT be in the store"
        r = gen.apply_schema({"where": "kitchen", "what": w})
        first = first or r
        # PREDICT the two invariant slots the situation withheld; `affect` varies -> must NOT appear
        hits += r["prediction"] == {"who": "self", "when": "morning"}
    # (i) family with NO invariant beyond the cue slot -> no false abstraction
    novar = EpisodicRecall(seed=6)
    for i in range(8):
        novar.encode(
            {"who": f"w{i}", "what": f"x{i}", "where": "lab", "when": f"q{i}", "affect": f"a{i}"}
        )
    c_noinv = novar.apply_schema({"where": "lab", "what": "brandnew"})
    # (ii) a situation whose every value is unknown to the store -> no family at all
    c_nofam = gen.apply_schema({"where": "nowhere_at_all", "what": "unseen_thing"})
    # (iii) ambiguous cue spanning two families that conflict on `who`
    amb = EpisodicRecall(seed=7)
    for i in range(6):
        amb.encode(
            {"who": "alice", "what": f"m{i}", "where": "cafe", "when": "noon", "affect": f"f{i}"}
        )
    for i in range(6):
        amb.encode(
            {"who": "bob", "what": f"n{i}", "where": "cafe", "when": "noon", "affect": f"g{i}"}
        )
    c_amb = amb.apply_schema({"where": "cafe", "what": "novel_dish"})
    return hits / len(novel), len(novel), first, c_noinv, c_nofam, c_amb


def _threshold_check():
    """The DERIVED thresholds must sit strictly between the floor they reject and the signal they
    must not reject, for every (S, m) the store can produce -- that is the property the retired
    frozen 0.15/0.30 could not have, since a constant crosses any signal that falls far enough.

    Returns (rows, n_pairs, worst_headroom, legacy_matches_stored) where rows are printable
    (S, m, floor, thresh, signal) tuples for a few representative sizes."""
    rows = []
    for n_slots, m in ((5, 6), (9, 6), (5, 2), (9, 2), (13, 3)):
        floor = _superposition_margin(n_slots) * _superposition_margin(m)
        signal = _prototype_margin(m, n_slots)
        rows.append((n_slots, m, floor, _analytic_thresh(floor, signal, 512), signal))
    # exhaustive sweep: threshold strictly inside (floor, signal) for both thresholds
    n_pairs, worst = 0, 1.0
    for m in range(2, 33):
        for n_slots in range(1, 41):
            floor = _superposition_margin(n_slots) * _superposition_margin(m)
            signal = _prototype_margin(m, n_slots)
            t = _analytic_thresh(floor, signal, 512)
            assert floor < t < signal, f"gist threshold outside (floor,signal) at S={n_slots} m={m}"
            worst = min(worst, t - floor, signal - t)
            n_pairs += 1
    for n_slots in range(1, 61):
        sig = _superposition_margin(n_slots)
        t = _analytic_thresh(0.0, sig, 512)
        assert 0.0 < t < sig, f"family threshold outside (0,signal) at S={n_slots}"
    # BACK-COMPAT: a legacy record with no "S" must not crash and must fall back to the registered
    # slot count -- for a DEFAULT_SLOTS store that reproduces the stored-S threshold exactly.
    er = EpisodicRecall(seed=11)
    er.encode(
        {"who": "self", "what": "toast", "where": "kitchen", "when": "day0", "affect": "calm"}
    )
    stored_thresh = er._family_thresh(er._episodes[0])
    legacy = {k: v for k, v in er._episodes[0].items() if k != "S"}  # simulate a pre-"S" record
    legacy_thresh = er._family_thresh(legacy)
    assert er._slots_of(legacy) == len(er._role), (
        "legacy fallback must be the registered slot count"
    )
    return rows, n_pairs, worst, (legacy_thresh == stored_thresh)


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

    # (d) SCHEMA-DRIVEN GENERALIZATION to UNSTORED situations -------------------------------------
    gen_acc, n_probe, g_first, c_noinv, c_nofam, c_amb = _generalization_check()
    print("-" * 78)
    print("(d) GENERALIZATION: apply_schema predicts a NEVER-STORED situation's missing slots")
    print(f"    unstored-situation prediction : {gen_acc:6.1%}  ({n_probe} probes, all unseen)")
    print(
        f"    example -> prediction={g_first['prediction']} conf={g_first['confidence']} "
        f"n_family={g_first['n_family']}"
    )
    print(
        f"    ctrl no-invariant family      : prediction={c_noinv['prediction']} "
        f"conf={c_noinv['confidence']} n_family={c_noinv['n_family']} "
        f"(schema={c_noinv['schema']})"
    )
    print(
        f"    ctrl ungrounded situation     : prediction={c_nofam['prediction']} "
        f"conf={c_nofam['confidence']} n_family={c_nofam['n_family']}"
    )
    print(
        f"    ctrl ambiguous cue (6/6 who)  : prediction={c_amb['prediction']} "
        f"conf={c_amb['confidence']} n_family={c_amb['n_family']}"
    )

    # (e) DERIVED THRESHOLDS (nothing frozen: computed from slot count, family size, code width) ---
    rows, n_pairs, worst, legacy_ok = _threshold_check()
    print("-" * 78)
    print("(e) THRESHOLDS: derived per call from S (slot load), m (family size), D (code width)")
    print(
        "    retired frozen pair: family_thresh=0.15  gist_thresh=0.30  (tuned at S=5, m=6, D=512)"
    )
    for n_slots, m, floor, thr, signal in rows:
        print(
            f"    S={n_slots:>2} m={m:>2} D=512 : schema floor={floor:.3f} < thresh={thr:.3f} "
            f"< signal={signal:.3f}   (membership thresh={_analytic_thresh(0.0, _superposition_margin(n_slots), 512):.3f})"
        )
    print(
        f"    swept {n_pairs} (S,m) pairs: threshold strictly inside (floor,signal) in ALL of them, "
        f"min headroom {worst:.3f}"
    )
    print(
        f"    legacy record with no stored S falls back to the slot count, threshold: {legacy_ok}"
    )

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
    # (d) generalization: real TRANSFER to unstored situations + three honest-failure controls.
    assert gen_acc >= 0.95, (
        f"apply_schema must predict the invariant slots of an UNSTORED situation, got {gen_acc:.2%}"
    )
    assert g_first["prediction"] == {"who": "self", "when": "morning"}, (
        f"prediction must be exactly the withheld INVARIANTS, got {g_first['prediction']}"
    )
    assert "affect" not in g_first["prediction"], "a VARYING slot must never be predicted"
    assert g_first["confidence"] > 0.0 and g_first["n_family"] == 8, (
        f"a real family must be found, got conf={g_first['confidence']} n={g_first['n_family']}"
    )
    assert c_noinv["prediction"] == {} and c_noinv["confidence"] == 0.0, (
        f"a family with no invariant beyond the cue must predict NOTHING, got {c_noinv}"
    )
    assert c_noinv["n_family"] >= 2, "the no-invariant control must still FIND its family"
    assert c_nofam["prediction"] == {} and c_nofam["confidence"] == 0.0, (
        f"an ungrounded situation must predict NOTHING, got {c_nofam}"
    )
    assert c_nofam["n_family"] == 0, f"an ungrounded situation has NO family, got {c_nofam}"
    assert "who" not in c_amb["prediction"], (
        f"an ambiguous cue must NOT pick a winner for the conflicting slot, got {c_amb}"
    )
    assert c_amb["prediction"] == {"when": "noon"}, (
        f"the ambiguous cue must still transfer the genuinely shared slot, got {c_amb}"
    )
    # determinism
    a = EpisodicRecall(seed=9)
    a.encode({"who": "self", "what": "toast", "where": "kitchen"})
    b = EpisodicRecall(seed=9)
    b.encode({"who": "self", "what": "toast", "where": "kitchen"})
    assert a.complete({"where": "kitchen"}) == b.complete({"where": "kitchen"}), (
        "must be deterministic"
    )
    assert _generalization_check()[0] == gen_acc, "apply_schema must be deterministic (re-run)"
    det = []
    for _ in range(2):  # two independently built, identically-fed stores must agree exactly
        er2 = EpisodicRecall(seed=9)
        for _f in ("toast", "eggs", "fruit"):
            er2.encode({"who": "self", "what": _f, "where": "kitchen", "when": "morning"})
        det.append(er2.apply_schema({"where": "kitchen", "what": "pancakes"}))
    assert det[0] == det[1], "apply_schema must be deterministic across instances"
    assert det[0]["prediction"] == {"who": "self", "when": "morning"}, (
        f"3-episode family must still transfer its invariants, got {det[0]}"
    )
    # (e) thresholds are DERIVED, not frozen: they must move with S/m/D, stay inside (floor,signal)
    # (asserted exhaustively inside _threshold_check), and survive a record written without an "S".
    assert legacy_ok, "a legacy record with no stored S must reproduce the DEFAULT_SLOTS threshold"
    assert worst > 0.0, f"no headroom between floor/threshold/signal, got {worst}"
    t5, t9, t21 = (_analytic_thresh(0.0, _superposition_margin(s), 512) for s in (5, 9, 21))
    assert t5 > t9 > t21, (
        f"the membership threshold must TIGHTEN as episodes gain slots, got {t5:.3f}/{t9:.3f}/{t21:.3f}"
    )
    assert abs(t5 - 0.15) < 0.002, (
        f"at the size the retired constant was tuned for (S=5,D=512) the derivation must reproduce "
        f"it, got {t5:.4f} vs 0.15"
    )
    assert _analytic_thresh(0.0, _superposition_margin(5), 2048) < t5, (
        "a wider code must lower the noise-driven threshold (sigma = 1/sqrt(D))"
    )

    print("-" * 78)
    print(
        f"self-check OK: pattern completion recovers held-out slots {real:.0%} vs {ctrl:.0%} "
        f"random-cue control (chance {chance:.0%}); degrades gracefully with noise "
        f"({r0:.0%}->{r_hi:.0%}); gist keeps invariant slots {sorted(g_schema['schema'])} "
        f"(margin gap {min(inv_margins) - max(var_margins):.2f}) while random-family and "
        f"single-episode controls have no gist; apply_schema GENERALIZES to NEVER-STORED "
        f"situations {gen_acc:.0%} ({n_probe} unseen probes, conf {g_first['confidence']}) while "
        f"the no-invariant and ungrounded controls honestly predict NOTHING at 0.0 and an "
        f"ambiguous 6/6 cue drops the conflicting slot ({c_amb['prediction']}) instead of picking "
        f"a winner; both decision thresholds are DERIVED per call from slot load / family size / "
        f"code width (no frozen constant, no learning, no adapted state), verified strictly inside "
        f"(floor,signal) over {n_pairs} (S,m) pairs with {worst:.3f} min headroom and reproducing "
        f"the retired 0.15 at the size it was tuned for; VSA one-shot associative memory, no "
        f"gradients; safety-inert."
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    main()
