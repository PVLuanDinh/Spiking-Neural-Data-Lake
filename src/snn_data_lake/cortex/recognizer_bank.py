r"""
RecognizerBank -- a heterogeneous SPIKING recognizer ENSEMBLE behind one contract.

The lake has SEVEN self-checking-but-dark spiking recognizers, each a different mechanism
that decides a label for a feature vector on its OWN terms and then goes dark:
  - cortex/snn_classifier.py         delta-rule spiking readout (MAGNITUDE / class current)
  - cortex/learned_matcher.py        supervised spiking classifier + Van Rossum NOVELTY gate
  - cortex/tempotron.py              Gutig-Sompolinsky spike-TIMING neuron (one-vs-rest)
  - cortex/lca_sparse_coding.py      LCA sparse coding by lateral INHIBITION (dominant atom)
  - reasoning/snn_moe_classifier.py  firing-rate ROUTED mixture-of-experts (top-k sparse)
  - reasoning/spiking_storage_prototype.py  Hopfield ASSOCIATIVE-memory attractor recall
  - cortex/paradigm_b_matcher.py     LIF COINCIDENCE detector (spike-stream channel overlap)

This module is the MERGE: one ensemble, one contract -- recognize(features) -> (label, confidence,
known). Each recognizer is a THIN ADAPTER over ONE existing module's real callable (it imports +
wraps it; it does NOT reimplement it) exposing recognize(features) -> (label, score, applicability).
The bank runs the panel, forms an applicability-weighted VOTE, and gates the decision on MARGIN and
AGREEMENT: a confident, agreeing panel returns known=True; a low-margin / disagreeing / low-evidence
panel returns an HONEST known=False and never fabricates a label. Out-of-distribution REJECTION is the
real payoff -- several members carry their own novelty signal (learned_matcher's calibrated Van Rossum
gate, the tempotron's absolute firing threshold, the LCA dominant-atom magnitude, the associative
recall overlap), so a garbage vector makes the panel abstain rather than agree on a nearest neighbour.

Feature domain: a length-64 vector in [0,1] (FEATURES == spike_preprocessing.N == snn_classifier.N),
the same window the enrolled recognizers already consume. Labels are arbitrary strings supplied by the
caller's library, so this is a TEXT/recognition path -- it returns a label, it never actuates an arm.

Every adapter is PURE STDLIB, deterministic (per-instance random.Random(seed), zlib.crc32 for content
addressing; the one member that reaches a module-global RNG -- the MoE -- seeds+restores it so the
decision is a pure function of the input), ASCII, and OFFLINE. Heavy modules import LAZILY inside each
adapter's __init__, so importing recognizer_bank stays stdlib-cheap.

  Run: python -m snn_data_lake.cortex.recognizer_bank     # headless ensemble self-check
"""
import contextlib
import io
import random
import zlib

FEATURES = 64            # feature-vector length (== spike_preprocessing.N == snn_classifier.N)
_EPS = 1e-9


# ------------------------------------------------------------- helpers --------------------------------


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@contextlib.contextmanager
def _quiet():
    """Swallow a wrapped trainer's ASCII progress prints so the ensemble self-check stays clean."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _prototype(label, n=FEATURES):
    """Deterministic reference vector for a label -- crc32(label) seeds a fresh RNG (never builtin hash),
    same sparse [0,1] distribution the enrolled command prototypes use, so calibrated gates behave alike."""
    r = random.Random(zlib.crc32(("recognizer_bank|" + label).encode()))
    return [r.random() if r.random() < 0.45 else 0.0 for _ in range(n)]


def default_library(labels=("alpha", "beta", "gamma", "delta", "epsilon")):
    """A small deterministic, neutrally-labelled recognition library (no arm semantics)."""
    return {lab: _prototype(lab) for lab in labels}


def _binarize(vec, thr=0.5):
    return [1 if v >= thr else 0 for v in vec]


def _topk_indices(vec, k):
    """Indices of the k largest values; deterministic tie-break by index (matches the storage prototype)."""
    return tuple(sorted(sorted(range(len(vec)), key=lambda i: (-vec[i], i))[:k]))


# ------------------------------------------------------------- recognizers (thin module adapters) -----
# Each: name; recognize(features) -> (label|None, score in [0,1], applicability in [0,1]).
# applicability == 0 (or label None) is an ABSTENTION (this member has no confident, in-distribution vote).


class LearnedMatcherRecognizer:
    """learned_matcher.py -- supervised spiking classifier (reuses snn_classifier's delta-rule readout)
    with a CALIBRATED Van Rossum novelty gate. The gate is the dedicated OOD rejecter: beyond it the
    member abstains rather than snapping the garbage to a nearest command."""

    name, module, source, uses = "learned-matcher", "learned_matcher", "reason", "stdlib"
    gatekeeper = True                                             # calibrated Van Rossum OOD gate

    def __init__(self, library, seed=0):
        from snn_data_lake.cortex import learned_matcher as L
        from snn_data_lake.cortex.spike_preprocessing import encode_latency
        from snn_data_lake.substrate._mathutils import softmax
        self._L, self._softmax = L, softmax
        self.labels = list(library)
        self.nC = len(self.labels)
        self._refs_enc = [encode_latency(v) for v in library.values()]
        with _quiet():
            self._W, _order = L.train_learned(library, seed=seed)     # _order == list(library)
        self._gate = L.calibrate_gate(library, self._refs_enc, seed=7)

    def recognize(self, features):
        idx, dist, _q = self._L.hybrid_match(self._W, self.nC, self._refs_enc, self._gate, features)
        if idx is None:                                               # novelty gate rejected -> abstain
            return (None, 0.0, 0.0)
        currents = self._L.C.forward(self._W, self._L.binarize(features), self.nC)[0]
        p = sorted(self._softmax(currents), reverse=True)
        score = p[0]
        appl = _clamp01(1.0 - dist / (self._gate + _EPS))            # closeness inside the calibrated gate
        return (self.labels[idx], _clamp01(score), appl)


class TempotronRecognizer:
    """tempotron.py -- one Gutig-Sompolinsky spike-TIMING neuron per label (one-vs-rest). Novelty is
    intrinsic: a tempotron is trained to STAY SILENT for non-members, so a novel vector leaves every
    neuron below threshold -> the member abstains."""

    name, module, source, uses = "tempotron", "tempotron", "reason", "stdlib"
    gatekeeper = True                                             # trained SILENT for non-members -> OOD signal

    def __init__(self, library, per_class=6, epochs=8, seed=0):
        from snn_data_lake.cortex import tempotron as TP
        self._TP = TP
        self.labels = list(library)
        with _quiet():
            self._tm = TP.TempotronMatcher(library, per_class=per_class, epochs=epochs, seed=seed)

    def recognize(self, features):
        pat = self._TP._pattern(features)
        peaks = [tp.vmax(pat)[0] for tp in self._tm.tempo]
        order = sorted(range(len(peaks)), key=lambda c: -peaks[c])
        top = order[0]
        top_v = peaks[top]
        second_v = peaks[order[1]] if len(order) > 1 else -1e30
        thr = self._tm.tempo[top].thr
        if top_v < 0.5 * thr:                                        # nothing fired convincingly -> novel
            return (None, 0.0, 0.0)
        sep = top_v - second_v
        appl = _clamp01(sep / (abs(top_v) + _EPS))
        score = _clamp01(top_v / (thr + _EPS))
        return (self.labels[top], score, appl)


class LcaRecognizer:
    """lca_sparse_coding.py -- LCAMatcher: sparse-code the vector over the enrolled references by lateral
    INHIBITION and read the label off the dominant atom. A near-zero dominant coefficient means nothing
    competed -> abstain."""

    name, module, source, uses = "lca-sparse", "lca_sparse_coding", "reason", "stdlib"

    def __init__(self, library, lam=0.05):
        from snn_data_lake.cortex.lca_sparse_coding import LCAMatcher
        self._m = LCAMatcher(library, lam=lam)
        self.labels = self._m.labels

    def recognize(self, features):
        a = self._m.lca.encode(features)
        mags = [abs(x) for x in a]
        order = sorted(range(len(mags)), key=lambda i: -mags[i])
        top = order[0]
        top_m = mags[top]
        second_m = mags[order[1]] if len(order) > 1 else 0.0
        if top_m < 0.05:                                            # no atom survived competition -> novel
            return (None, 0.0, 0.0)
        appl = _clamp01((top_m - second_m) / (top_m + _EPS))
        score = _clamp01(top_m / (top_m + second_m + _EPS))
        return (self.labels[top], score, appl)


class MoeRecognizer:
    """snn_moe_classifier.py -- firing-rate ROUTED mixture-of-experts (top-k sparse). Trains the ported
    Nord MoE on binarised jittered exemplars of the library. sample_spikes reaches the module-global RNG,
    so recognize seeds it from the input (crc32) and RESTORES it -> the decision is a pure function."""

    name, module, source, uses = "moe", "snn_moe_classifier", "reason", "stdlib"

    def __init__(self, library, seed=0, exemplars=12, train_noise=0.2):
        from snn_data_lake.cortex import learned_matcher as L
        from snn_data_lake.reasoning import snn_moe_classifier as M
        self._M, self._L = M, L
        self.labels = list(library)
        self.nC = len(self.labels)
        rng = random.Random(seed)
        data = []
        for li, lab in enumerate(self.labels):
            for _ in range(exemplars):
                data.append((L.binarize(L.jitter(library[lab], train_noise, rng)), li))
        rng.shuffle(data)
        state = random.getstate()
        random.seed(seed)
        try:
            with _quiet():
                self._experts, self._bias, _usage, _synops = M.train(data, self.nC)
        finally:
            random.setstate(state)

    def recognize(self, features):
        sample = self._L.binarize(features)
        state = random.getstate()
        random.seed(zlib.crc32(bytes(sample)) & 0xFFFFFFFF)
        try:
            probs = self._M.forward(self._experts, self._bias, sample, self.nC)[0]
        finally:
            random.setstate(state)
        order = sorted(range(self.nC), key=lambda c: -probs[c])
        top = order[0]
        margin = probs[top] - (probs[order[1]] if len(order) > 1 else 0.0)
        if margin < 0.05:                                          # flat routing -> no confident vote
            return (None, 0.0, 0.0)
        return (self.labels[top], _clamp01(probs[top]), _clamp01(margin / (probs[top] + _EPS)))


class StorageRecognizer:
    """spiking_storage_prototype.py -- Hopfield ASSOCIATIVE memory. Stores one k-hot pattern per label
    (the label prototype's strongest features), then RECALLS from the query's k-hot cue by attractor
    dynamics and votes for the label whose stored pattern the settled state most overlaps. A low best
    overlap means the cue fell into no basin -> abstain."""

    name, module, source, uses = "assoc-storage", "spiking_storage_prototype", "reason", "stdlib"

    def __init__(self, library, k=12, n=FEATURES, steps=8):
        from snn_data_lake.reasoning import spiking_storage_prototype as S
        self._S = S
        self.labels = list(library)
        self.n, self.k, self.steps = n, k, steps
        self._patterns = [_topk_indices(library[lab], k) for lab in self.labels]
        self._memory = S.train(self._patterns, n, k)

    def recognize(self, features):
        cue = _topk_indices(features, self.k)
        out, _used, _syn = self._S.recall(self._memory, set(cue), self.n, self.k, self.steps)
        overlaps = [self._S.overlap(out, p, self.k) for p in self._patterns]
        order = sorted(range(len(overlaps)), key=lambda i: -overlaps[i])
        top = order[0]
        top_o = overlaps[top]
        second_o = overlaps[order[1]] if len(order) > 1 else 0.0
        if top_o < 0.5:                                            # cue settled into no stored basin -> novel
            return (None, 0.0, 0.0)
        return (self.labels[top], _clamp01(top_o), _clamp01(top_o - second_o))


class ParadigmBRecognizer:
    """paradigm_b_matcher.py -- LIF COINCIDENCE detector. Per label, the ON features are the template
    channels; the query's active-in-template channels are streamed through match_stream (k=1, so the
    emitted match count equals the DISTINCT-channel overlap). Votes for the label with the highest
    normalised overlap; a low overlap fraction is an abstention."""

    name, module, source, uses = "paradigm-b", "paradigm_b_matcher", "reason", "stdlib"

    def __init__(self, library, thr=0.5, window=None):
        from snn_data_lake.cortex import paradigm_b_matcher as PB
        from snn_data_lake.cortex.spike_preprocessing import T
        self._PB = PB
        self.labels = list(library)
        self.thr = thr
        self._window = window or T
        self._channels = [[i for i, v in enumerate(library[lab]) if v >= thr] for lab in self.labels]
        self._queries = [PB.compile_query(ch, self._window, 1) if ch else None
                         for ch in self._channels]

    def recognize(self, features):
        active = [i for i, v in enumerate(features) if v >= self.thr]
        overlaps = []
        for ch, q in zip(self._channels, self._queries):
            if not ch or q is None:
                overlaps.append(0.0)
                continue
            chset = set(ch)
            hits = [c for c in active if c in chset]
            events = list(enumerate(hits))                          # (distinct time, channel), ascending
            overlaps.append(len(self._PB.match_stream(events, q)) / len(ch))
        order = sorted(range(len(overlaps)), key=lambda i: -overlaps[i])
        top = order[0]
        top_o = overlaps[top]
        second_o = overlaps[order[1]] if len(order) > 1 else 0.0
        if top_o < 0.34:                                           # too little coincidence to commit -> novel
            return (None, 0.0, 0.0)
        return (self.labels[top], _clamp01(top_o), _clamp01(top_o - second_o))


# ------------------------------------------------------------- the ensemble (vote + margin/agree gate) -


class RecognizerBank:
    """Heterogeneous spiking recognizer ENSEMBLE behind one contract. register(recognizer) adds a thin
    module adapter; recognize(features) runs the panel, forms an applicability-weighted vote, and gates
    on MARGIN and AGREEMENT. A confident, agreeing, well-evidenced panel returns known=True; otherwise an
    HONEST known=False (never a fabricated label). Deterministic: ties break by registration order."""

    def __init__(self, min_recognizers=3, min_agree=0.5, min_margin=0.15, min_gatekeepers=1):
        self._recognizers = []
        self.min_recognizers = min_recognizers
        self.min_agree = min_agree
        self.min_margin = min_margin
        self.min_gatekeepers = min_gatekeepers

    def register(self, recognizer):
        self._recognizers.append(recognizer)
        return self

    def roster(self):
        """(name, module, source, uses) for every registered recognizer -- which studies are LIVE."""
        return [(r.name, r.module, r.source, r.uses) for r in self._recognizers]

    def _run(self, features):
        """[(recognizer, label|None, score, applicability), ...]. A broken member never crashes the turn
        -- it abstains."""
        out = []
        for r in self._recognizers:
            try:
                label, score, appl = r.recognize(features)
            except Exception:                                       # a broken recognizer just abstains
                label, score, appl = None, 0.0, 0.0
            if label is None:
                out.append((r, None, 0.0, 0.0))
            else:
                out.append((r, label, _clamp01(score), _clamp01(appl)))
        return out

    def panel(self, features):
        """Raw per-recognizer votes: [(name, label|None, score, applicability), ...]."""
        return [(r.name, lab, s, a) for (r, lab, s, a) in self._run(features)]

    def recognize(self, features):
        """Return (label, confidence, known). The applicability-weighted vote picks the label; known=True
        requires a confident, agreeing, well-evidenced panel AND corroboration from at least
        min_gatekeepers GATEKEEPER members (those carrying an intrinsic OOD signal -- the calibrated Van
        Rossum gate and the tempotron's silence-for-non-members). Otherwise an HONEST known=False: a
        garbage vector makes the gatekeepers abstain / back a different label, so the ensemble never
        fabricates a confident label for out-of-distribution input."""
        runs = self._run(features)
        voters = [(r, lab, s, a) for (r, lab, s, a) in runs if lab is not None and a > 0.0]
        if not voters:
            return (None, 0.0, False)
        votes = {}
        for _r, lab, s, a in voters:
            votes[lab] = votes.get(lab, 0.0) + a * s
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        top_label, top_w = ranked[0]
        second_w = ranked[1][1] if len(ranked) > 1 else 0.0
        total_w = sum(votes.values())
        n_applied = len(voters)
        n_top = sum(1 for _r, lab, _s, _a in voters if lab == top_label)
        agree = n_top / n_applied
        margin = (top_w - second_w) / (top_w + _EPS)
        gate_backing = sum(1 for r, lab, _s, _a in voters
                           if lab == top_label and getattr(r, "gatekeeper", False))
        confidence = round(top_w / (total_w + _EPS), 3)
        known = (n_applied >= self.min_recognizers
                 and agree >= self.min_agree
                 and margin >= self.min_margin
                 and gate_backing >= self.min_gatekeepers)
        return (top_label, confidence, bool(known))


def build_default_bank(library=None, seed=0):
    """Register the six heterogeneous spiking recognizers over `library` (default: a small neutral
    library). Reuses all seven listed modules -- snn_classifier is reused transitively through the
    learned-matcher adapter. Training happens here (once); recognize() is then cheap + deterministic."""
    if library is None:
        library = default_library()
    bank = RecognizerBank()
    bank.register(LearnedMatcherRecognizer(library, seed=seed))
    bank.register(TempotronRecognizer(library, seed=seed))
    bank.register(LcaRecognizer(library))
    bank.register(MoeRecognizer(library, seed=seed))
    bank.register(StorageRecognizer(library))
    bank.register(ParadigmBRecognizer(library))
    return bank


# ------------------------------------------------------------- reasoning-cortex hook (opt-in) ---------
# NOT registered by default. The MAIN AGENT wires this into reasoning_cortex.build_default_cortex behind
# an opt-in flag so the existing 8-expert cortex stays byte-identical until enabled (see the module docs
# / returned integration-hook spec). Scores 0 on ordinary text queries, so even once registered it never
# perturbs the other experts' routing; the heavy bank builds LAZILY on first applicable use.


class RecognizerBankExpert:
    """reasoning_cortex adapter over RecognizerBank. Fires ONLY on an explicit feature-recognition query
    ('recognize: v1 v2 ... v64' with exactly FEATURES numbers in [0,1]); anything else scores 0 so the
    other experts are untouched. The bank builds lazily on first answer (train-once, reuse)."""

    name, module, source, uses = "recognizer-bank", "recognizer_bank", "reason", "stdlib"

    def __init__(self, seed=0, library=None):
        self.seed = seed
        self._library = library
        self._bank = None                                          # lazy: built on first applicable answer

    @staticmethod
    def _parse(text):
        import re
        low = text.lower()
        if not any(kw in low for kw in ("recognize", "classify", "which pattern")):
            return None
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return [float(x) for x in nums] if nums else None

    def score(self, q):
        vec = self._parse(getattr(q, "text", str(q)))
        if vec is None or len(vec) < 8:
            return 0.0
        return 0.9

    def answer(self, q):
        vec = self._parse(getattr(q, "text", str(q)))
        if vec is None:
            return ("[honest gap] phrase a recognition query as 'recognize: v1 v2 ... vN'.", False)
        if len(vec) != FEATURES:
            return (f"[honest gap] i recognize length-{FEATURES} feature vectors; got {len(vec)} numbers "
                    "-- no padding, no bluff.", False)
        if self._bank is None:
            self._bank = build_default_bank(library=self._library, seed=self.seed)
        features = [_clamp01(v) for v in vec]
        label, confidence, known = self._bank.recognize(features)
        if not known:
            return (f"[honest gap] the spiking recognizer panel did not agree on a label with enough margin "
                    f"(best='{label}', confidence {confidence:.2f}) -- treated as out-of-distribution, "
                    "no fabrication.", False)
        return (f"[recognizer-bank] the spiking ensemble (learned-matcher / tempotron / LCA / MoE / "
                f"associative-storage / paradigm-B) recognized the vector as '{label}' "
                f"(confidence {confidence:.2f}, margin+agreement gate passed).", True)


# ------------------------------------------------------------- headless self-check --------------------


def main():
    library = default_library()
    labels = list(library)
    bank = build_default_bank(library=library, seed=0)
    roster = bank.roster()

    print("=" * 96)
    print("RECOGNIZER BANK -- heterogeneous spiking recognizer ensemble behind one recognize() contract")
    print("=" * 96)
    print(f"panel ({len(roster)} recognizers; each a thin adapter over one dark spiking module):")
    for name, module, source, uses in roster:
        print(f"  {name:<16} <- {module + '.py':<30} source={source:<7} backend={uses}")
    print("-" * 96)

    # in-distribution: two jittered exemplars per label (light noise) -> correct label, known=True
    rng = random.Random(20260719)

    def jitter(vec, noise):
        return [_clamp01(v + rng.uniform(-noise, noise)) for v in vec]

    known_ok, total, distinct_voters = 0, 0, set()
    print("in-distribution recognition (jittered exemplars):")
    for lab in labels:
        for _ in range(2):
            feats = jitter(library[lab], 0.10)
            got, conf, known = bank.recognize(feats)
            for name, vlab, _s, appl in bank.panel(feats):
                if vlab is not None and appl > 0.0:
                    distinct_voters.add(name)
            total += 1
            hit = (got == lab and known)
            known_ok += 1 if hit else 0
            print(f"  {lab:<9} -> label={str(got):<9} conf={conf:.2f} known={known}  {'OK' if hit else 'MISS'}")

    # out-of-distribution: uniform-random garbage vectors -> honest known=False (no fabrication)
    print("-" * 96)
    print("out-of-distribution rejection (uniform-random vectors -> honest known=False):")
    ood_rng = random.Random(424242)
    ood_rejected = 0
    ood_trials = 4
    for t in range(ood_trials):
        garbage = [ood_rng.random() for _ in range(FEATURES)]
        got, conf, known = bank.recognize(garbage)
        ood_rejected += 0 if known else 1
        print(f"  ood#{t}     -> label={str(got):<9} conf={conf:.2f} known={known}  "
              f"{'REJECTED' if not known else 'FABRICATED'}")

    # determinism: recognize is a pure function of the input
    probe = jitter(library[labels[0]], 0.10)
    d1 = bank.recognize(probe)
    for _ in range(5):
        random.random()                                            # perturb the module-global RNG
    d2 = bank.recognize(probe)

    print("-" * 96)
    voters_str = ", ".join(sorted(distinct_voters))
    print(f"coverage: {known_ok}/{total} in-distribution vectors recognized known=True; "
          f"{ood_rejected}/{ood_trials} OOD vectors rejected; "
          f"{len(distinct_voters)} DISTINCT recognizers fired: {voters_str}")
    print("=" * 96)

    # ---- assertions ----------------------------------------------------------------------------------
    assert len(roster) >= 4, f"at least 4 distinct recognizers must be registered (got {len(roster)})"
    assert known_ok >= total - 1, f"in-distribution vectors must recognize to the right label known=True ({known_ok}/{total})"
    assert len(distinct_voters) >= 4, f"at least 4 DISTINCT recognizers must fire on real inputs (got {len(distinct_voters)}: {sorted(distinct_voters)})"
    assert ood_rejected == ood_trials, f"every OOD vector must be an honest known=False (rejected {ood_rejected}/{ood_trials})"
    assert d1 == d2, f"recognize() must be deterministic for identical input (got {d1!r} vs {d2!r})"

    # the reasoning-cortex hook: scores 0 on ordinary text (never perturbs other experts), fires only on
    # an explicit feature-recognition query, and rejects a malformed one honestly.
    class _Q:
        def __init__(self, text):
            self.text = text
    expert = RecognizerBankExpert(seed=0, library=library)
    assert expert.score(_Q("what comes next in 2 4 6 8")) == 0.0, "hook must not claim ordinary text queries"
    assert expert.score(_Q("please water the office plants")) == 0.0, "hook must not claim non-recognition text"
    good_vec = " ".join(f"{v:.3f}" for v in jitter(library[labels[1]], 0.10))
    reply, known = expert.answer(_Q("recognize: " + good_vec))
    assert known and ("'" + labels[1] + "'") in reply, f"hook must recognize a well-formed in-distribution query: {reply}"
    _r2, k2 = expert.answer(_Q("recognize: 0.1 0.2 0.3"))
    assert not k2, "hook must honestly reject a wrong-length vector"

    print(f"self-check OK: {len(roster)} heterogeneous spiking recognizers merged behind one recognize() "
          f"contract; {known_ok}/{total} in-distribution known=True, {len(distinct_voters)} DISTINCT members "
          f"fired, {ood_rejected}/{ood_trials} OOD vectors rejected (no fabrication); deterministic; "
          "reasoning-cortex hook is opt-in + inert on ordinary text.")
    return 0


if __name__ == "__main__":
    main()
