r"""
REASONING CORTEX -- a mixture-of-experts ROUTER that makes the reasoning/ STUDIES go LIVE.

The lake has ~120 reasoning/ modules that each self-check in isolation (`python -m ...`) and then go
dark: nothing calls them on a live conversational turn. The brain's only reasoning path,
LakeChat.converse (reasoning/spike_chat_lake.py), is an ad-hoc if/elif -- one series/analogy branch
(Gf), one structured-KB branch, one retrieval branch, then an honest gap. Every OTHER study -- HDC
binding, Kanerva sparse distributed memory, dense associative memory, spiking self-organizing maps,
triplet STDP -- is research that never runs against a query.

This module is the MERGE: a SkillRegistry / mixture-of-experts ROUTER. Each EXPERT is a THIN ADAPTER
over ONE existing reasoning/ module's real capability (it imports + wraps that module; it does NOT
reimplement it). Given a text query, the router scores every expert by cheap query features
(keyword / structural type match + each expert's self-reported applicability) and dispatches the
SINGLE best-matching expert -- a router, not a wasteful run-all-at-once. A confident expert returns a
real answer; when NO expert is confident, the router returns an HONEST gap (known=False) and never
fabricates.

Wired into LakeChat.converse as an additional dispatch on the turns the grounded/retrieval path could
not answer: ~8 dead reasoning studies now answer real queries live, in the exact return shape
{reply, known, source, ...} that cortex/live_brain._reasoning_reply consumes (source in the accepted
set {"gf", "reason", ...}, known=True on a hit). Default-safe: any failure -> honest gap, so the
existing converse self-check is byte-identical.

Every adapter here is PURE STDLIB (math/random/zlib), deterministic (per-instance random.Random(seed),
zlib.crc32 for content addressing), ASCII, and OFFLINE (no import-time or boot-time network fetch).
The OPTIONAL ninth expert (RelationalCompletionExpert, registered only when the caller passes its OWN
KB triples to build_default_cortex) lazily imports numpy + kb_ranking.DistMult on FIRST USE -- this
module stays stdlib-importable, and the expert trains only on the passed triples (never a dataset fetch).

  Run: python -m snn_data_lake.reasoning.reasoning_cortex     # headless MoE self-check
"""

import re
import zlib

# ------------------------------------------------------------- query features (computed once) --------

_INT_TOK = re.compile(r"-?\d+")
_EQ_PAIR = re.compile(r"\b([a-z][a-z0-9_]*)\s*=\s*([a-z0-9_]+)")
_ARROW_PAIR = re.compile(r"\b([a-z0-9_]+)\s*->\s*([a-z0-9_]+)")
_ANALOGY = re.compile(r"([a-z]+)\s*:\s*([a-z]+)\s*::\s*([a-z]+)\s*:\s*\?")
_QUERY_ARROW = re.compile(r"\b([a-z0-9_]+)\s*->\s*\?")
_WORD_NUM_ANALOGY = re.compile(r"\b([a-z]+)\s*:\s*(\d+)\s*::\s*([a-z]+)\s*:\s*\?")


class Query:
    """Cheap, once-per-turn feature extraction the experts route on (no per-expert re-parsing cost)."""

    def __init__(self, text):
        self.text = text
        self.low = text.lower()
        # contiguous integer RUN (a real series), not scattered ints across a sentence
        toks = re.findall(r"[^\s,;]+", self.low.replace("?", " "))
        runs, cur = [], []
        for t in toks:
            if _INT_TOK.fullmatch(t) and len(t.lstrip("-")) <= 9:
                cur.append(int(t))
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        self.int_run = max(runs, key=len) if runs else []
        self.all_ints = [int(t) for t in toks if _INT_TOK.fullmatch(t) and len(t.lstrip("-")) <= 9]
        self.eq_pairs = list(_EQ_PAIR.findall(self.low))  # [(role, filler), ...]
        self.arrow_pairs = list(_ARROW_PAIR.findall(self.low))  # [(key, value), ...]
        self.analogy = _ANALOGY.search(self.low)
        self.query_arrow = _QUERY_ARROW.search(self.low)  # trailing 'x -> ?' -- the queried input
        self.word_num_analogy = _WORD_NUM_ANALOGY.search(self.low)  # 'word : N :: word2 : ?'
        self.single_letters = re.findall(r"\b[a-z]\b", self.low)
        # a Raven grid: 3 groups (';' or newline), 3 cells each, exactly one '?' blank
        groups = [g.strip() for g in re.split(r"[;\n]", text) if g.strip()]
        self.raven = None
        if len(groups) == 3:
            grid = [re.findall(r"[a-z0-9?]+", g.lower()) for g in groups]
            if all(len(r) == 3 for r in grid) and sum(c == "?" for r in grid for c in r) == 1:
                self.raven = grid

    def has(self, *kw):
        return any(k in self.low for k in kw)


# ------------------------------------------------------------- the experts (thin module adapters) ----
# Each expert wraps ONE reasoning/ module. score(q) -> applicability in [0,1] (cheap features);
# answer(q) -> (reply, known). known=False is an HONEST gap (recognised shape, no confident answer).


class NumberSeriesExpert:
    """gf.py -- fluid-intelligence PROGRAM SYNTHESIS over a number series (polynomial/geometric/
    recurrence/periodic/interleave). Wraps gf.solve_series; the rule is induced at test time."""

    name, module, source, uses = "gf-number-series", "gf", "gf", "stdlib"

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.gf import solve_series

        self._solve = solve_series

    def score(self, q):
        if q.has("cluster", "group", "partition", "hz", "stdp", "plasticity", "synaptic"):
            return 0.0  # another expert's structural query
        if len(q.int_run) >= 3:
            return 0.88
        if len(q.int_run) == 2 and q.has("next", "series", "sequence", "continue"):
            return 0.5
        return 0.0

    def answer(self, q):
        stmt = " ".join(map(str, q.int_run))
        got = self._solve(q.int_run)
        if got is None:
            return (
                "[honest gap] i induce polynomial/geometric/recurrence/periodic/interleave rules but no "
                "program in my DSL fits %s -- no bluff." % stmt,
                False,
            )
        val, rule = got
        return (
            "[gf] induced a rule for %s at test time -> next = %s (%s). not a stored fact."
            % (stmt, val, rule),
            True,
        )


class LetterExpert:
    """gf.py -- letter analogy ('abc:abd :: ijk:?') and letter-series extrapolation. Wraps
    gf.letter_analogy / gf.letter_series; the per-position successor shift is induced + read out."""

    name, module, source, uses = "gf-letter", "gf", "gf", "stdlib"

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.gf import letter_analogy, letter_series

        self._analogy, self._series = letter_analogy, letter_series

    def score(self, q):
        if q.analogy:
            return 0.92
        if len(q.single_letters) >= 3 and q.has("next", "series", "sequence", "letter", "continue"):
            return 0.8
        return 0.0

    def answer(self, q):
        if q.analogy:
            a, b, c = q.analogy.group(1), q.analogy.group(2), q.analogy.group(3)
            got = self._analogy(a, b, c)
            if got is None:
                return (
                    "[honest gap] '%s:%s :: %s:?' is outside my letter-analogy induction (unequal "
                    "lengths or it runs off the alphabet)." % (a, b, c),
                    False,
                )
            return (
                "[gf] letter analogy %s:%s :: %s:? -> %s (induced the per-position shift, applied it to c)."
                % (a, b, c, got),
                True,
            )
        letters = "".join(q.single_letters)
        got = self._series(letters)
        if got is None:
            return (
                "[honest gap] no single successor shift extends the letters %s."
                % " ".join(q.single_letters),
                False,
            )
        return (
            "[gf] letter series %s -> %s (successor shift induced over the a-z spike codebook)."
            % (" ".join(q.single_letters), got),
            True,
        )


class RavenExpert:
    """gf.py -- Raven grid completion. Tries DISTRIBUTION-OF-THREE first (gf.raven_dist3, relational
    integration across rows via the spiking VSA); when that gaps, falls back to a COMPOSED row x
    column arithmetic grid (gf.raven_arith_grid) for single-letter cells, or a two-attribute
    letter+digit Latin-square grid (gf.raven_cell_attrs) for letter-digit cells."""

    name, module, source, uses = "gf-raven", "gf", "gf", "stdlib"

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.gf import raven_arith_grid, raven_cell_attrs, raven_dist3

        self._raven, self._arith, self._cell_attrs = raven_dist3, raven_arith_grid, raven_cell_attrs

    def score(self, q):
        return 0.93 if q.raven else 0.0

    def answer(self, q):
        rows = [[None if c == "?" else c for c in r] for r in q.raven]
        stmt = " / ".join(" ".join(r) for r in q.raven)
        got = self._raven(rows)
        label = "distribution-of-three"
        if got is None:
            if any(len(c) > 1 for r in rows for c in r if c is not None):
                got, label = (
                    self._cell_attrs(rows),
                    "composed letter+digit (Latin square + positional digit)",
                )
            else:
                got, label = self._arith(rows), "composed row x column arithmetic"
        if got is None:
            return (
                "[honest gap] '%s' is not a Raven distribution-of-three, arithmetic-grid, or "
                "cell-attribute item i can solve." % stmt,
                False,
            )
        return (
            "[gf] Raven %s %s -> the blank is '%s' (relational integration across rows/columns via "
            "the spiking VSA)." % (label, stmt, got),
            True,
        )


class PairInductionExpert:
    """gf.py -- example -> rule INDUCTION from explicit (x, y) PAIRS ('if 3 -> 10, 5 -> 26, 7 -> 50,
    then 9 -> ?'), as opposed to a flat number SERIES. Wraps gf.function_from_pairs (numeric grammar)
    and gf.string_from_pairs (composed string-op grammar); both are LEAVE-ONE-OUT guarded over the
    given pairs before being applied to the queried input."""

    name, module, source, uses = "gf-pair-induction", "gf", "gf", "stdlib"

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.gf import function_from_pairs, string_from_pairs

        self._num, self._str = function_from_pairs, string_from_pairs

    def score(self, q):
        return 0.9 if (q.query_arrow and len(q.arrow_pairs) >= 2) else 0.0

    def answer(self, q):
        pairs, query = q.arrow_pairs, q.query_arrow.group(1)
        numeric = all(
            re.fullmatch(r"-?\d+", k) and re.fullmatch(r"-?\d+", v) for k, v in pairs
        ) and re.fullmatch(r"-?\d+", query)
        got = (
            self._num([(int(k), int(v)) for k, v in pairs], int(query))
            if numeric
            else self._str(pairs, query)
        )
        if got is None:
            return (
                "[honest gap] no program (numeric or string) induced from the given pairs "
                "reproduces every example under leave-one-out.",
                False,
            )
        val, label = got
        return (
            "[gf] induced %s from the given pairs at test time (leave-one-out verified) -> %s = %s. "
            "not a stored fact." % (label, query, val),
            True,
        )


class WordRelationExpert:
    """gf.py -- word:number analogy whose relation is some countable ATTRIBUTE of the word (letter
    count, vowel count, ...) rather than a taught KB fact ('one : 3 :: three : ?'). Wraps
    gf.word_number_relation: a small ordered set of candidate word->number attributes is tried,
    smallest/most-primitive first, and the FIRST whose value on the given word matches the given
    number is trusted (Occam), then applied to the queried word."""

    name, module, source, uses = "gf-word-relation", "gf", "gf", "stdlib"

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.gf import word_number_relation

        self._rel = word_number_relation

    def score(self, q):
        return 0.9 if q.word_num_analogy else 0.0

    def answer(self, q):
        a, b, c = (
            q.word_num_analogy.group(1),
            q.word_num_analogy.group(2),
            q.word_num_analogy.group(3),
        )
        got = self._rel(a, int(b), c)
        if got is None:
            return (
                "[honest gap] no candidate word attribute maps '%s'->%s consistently, so i won't "
                "guess '%s'->?." % (a, b, c),
                False,
            )
        return (
            "[gf] induced word relation from %s:%s -> applied to %s:? = %d." % (a, b, c, got),
            True,
        )


class ArcExpert:
    """arc_solver.py -- ARC-domain TEST-TIME PROGRAM SYNTHESIS over 2-D grids: object perception +
    a grid-transformation DSL + iterative-deepening search for the SMALLEST program consistent with
    every training pair (the gf.induce_program idiom on grids). The task is presented as text:
    'arc: <in> -> <out> ; <in> -> <out> ; test <grid>' -- rows separated by '|', cells digits 0-9.
    score() is pure-regex/cheap; the solver imports LAZILY on first answer (build cost unchanged).
    Honest gap when no program in the DSL reproduces all the training pairs -- never a guessed grid."""

    name, module, source, uses = "arc-grid", "arc_solver", "gf", "stdlib"

    def __init__(self, seed=0):
        self._synth = None  # lazy: arc_solver.synthesize on first use

    @staticmethod
    def _grid(txt):
        rows = [[int(v) for v in re.findall(r"[0-9]", part)] for part in txt.split("|")]
        rows = [r for r in rows if r]
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            return None
        return rows

    def _parse(self, q):
        m = re.search(r"\barc\b\s*:?(.*)$", q.low, re.S)
        if not m:
            return None
        pairs, test = [], None
        for seg in m.group(1).split(";"):
            seg = seg.strip()
            if "->" in seg:
                a, _, b = seg.partition("->")
                gi, go = self._grid(a), self._grid(b)
                if gi is None or go is None:
                    return None
                pairs.append((gi, go))
            elif seg.startswith("test"):
                test = self._grid(seg[4:])
        if not pairs or test is None:
            return None
        return pairs, test

    def score(self, q):
        return 0.95 if self._parse(q) else 0.0

    def answer(self, q):
        pairs, test = self._parse(q)
        if self._synth is None:
            from snn_data_lake.reasoning.arc_solver import synthesize

            self._synth = synthesize
        cands, progs = self._synth(pairs, test, attempts=1, node_budget=1500, time_budget=2.0)
        if not cands:
            return (
                "[honest gap] no program in my grid DSL (<=3 steps, including the primitives "
                "minted from this task's own training pairs) reproduces all %d training "
                "pairs of this arc task -- no bluffed grid." % len(pairs),
                False,
            )
        names, closer = progs[0]
        grid_txt = "|".join("".join(str(v) for v in row) for row in cands[0])
        return (
            "[arc] induced a grid program at test time -> %s (program: %s+%s, the smallest "
            "consistent with all %d train pairs). a synthesized program, not a stored fact."
            % (grid_txt, ">".join(names) if names else "id", closer, len(pairs)),
            True,
        )


class BindingExpert:
    """mcr_hdc.py -- Modular Composite Representation (HDC): bind role=filler pairs into ONE
    hypervector, then unbind a queried role to recover its filler. Wraps mcr_hv/bind/bundle/unbind/sim.
    This is the canonical variable-binding task classical connectionism struggles with."""

    name, module, source, uses = "hdc-binding", "mcr_hdc", "reason", "stdlib"

    def __init__(self, seed=0, d=256):
        self.d = d
        from snn_data_lake.reasoning.mcr_hdc import (
            mcr_bind,
            mcr_bundle,
            mcr_hv,
            mcr_sim,
            mcr_unbind,
        )

        self._hv, self._bind, self._bundle, self._unbind, self._sim = (
            mcr_hv,
            mcr_bind,
            mcr_bundle,
            mcr_unbind,
            mcr_sim,
        )

    @staticmethod
    def _role_query(q):
        # which ROLE to unbind. "who is"/"which"/"what's" are accepted too: a natural way to ask for a
        # filler ("who is the agent") used to miss this regex while still scoring 0.9, so the expert was
        # picked and then refused. Optional determiner so "what is THE color" resolves to 'color'.
        m = re.search(
            r"(?:what is|what's|who is|who's|which|unbind|recover|value of|\bget)\s+"
            r"(?:the\s+|a\s+|an\s+)?([a-z][a-z0-9_]*)",
            q.low,
        )
        return m.group(1) if m else None

    def score(self, q):
        # SCORE ONLY WHAT WE CAN ANSWER. answer() needs BOTH >=2 bindings AND a named role to unbind;
        # scoring 0.9 on "bindings + the word bind" (with no role named) made the router hand this
        # expert a turn it would immediately refuse -- confidence the answerer could not honour.
        # Bindings without a named role still route (0.5 clears the threshold) so the caller gets the
        # useful "tell me which role" prompt -- it just no longer claims high confidence for it.
        if len(q.eq_pairs) >= 2 and self._role_query(q):
            return 0.9
        if len(q.eq_pairs) >= 2:
            return 0.5
        return 0.0

    def answer(self, q):
        pairs = dict(q.eq_pairs)  # role -> filler (last binding wins on dup)
        role = self._role_query(q)
        if role is None:
            return ("[honest gap] tell me which role to unbind (e.g. 'what is color').", False)
        if role not in pairs:
            return (
                "[honest gap] i bound the roles %s but '%s' is not among them -- no filler to recover."
                % (sorted(pairs), role),
                False,
            )
        roles = {k: self._hv("ROLE_" + k, self.d) for k in pairs}
        fills = {v: self._hv("FILL_" + v, self.d) for v in pairs.values()}
        record = self._bundle([self._bind(roles[k], fills[pairs[k]]) for k in pairs])
        probe = self._unbind(record, roles[role])
        best = max(fills, key=lambda v: self._sim(probe, fills[v]))
        return (
            "[hdc] bound %d role=filler pairs into one hypervector, unbound '%s' -> '%s' (MCR: "
            "bind=modular-add, bundle=circular-mean, unbind+cleanup recovers the filler)."
            % (len(pairs), role, best),
            True,
        )


class DistributedRecallExpert:
    """sparse_distributed_memory.py -- Kanerva SDM: a content-addressable key->value store spread over
    many random hard locations, recalled by a distributed vote. Wraps the SDM class (write/read)."""

    name, module, source, uses = "sdm-recall", "sparse_distributed_memory", "reason", "stdlib"

    def __init__(self, seed=0, addr_dim=64):
        self.addr_dim = addr_dim
        from snn_data_lake.reasoning.sparse_distributed_memory import SDM

        self._SDM = SDM

    @staticmethod
    def _query_key(q):
        m = re.search(r"(?:lookup|recall|read|\bget)\s+([a-z0-9_]+)", q.low)
        return m.group(1) if m else None

    def _bits(self, s):
        rng = _rng("SDM|" + s)
        return [rng.randint(0, 1) for _ in range(self.addr_dim)]

    def score(self, q):
        if len(q.arrow_pairs) >= 2 and q.has("lookup", "recall", "sdm"):
            return 0.9 if self._query_key(q) else 0.55
        return 0.0

    def answer(self, q):
        kv = list(q.arrow_pairs)
        key = self._query_key(q)
        if key is None:
            return ("[honest gap] give me a key to look up (e.g. 'lookup apple').", False)
        vals = sorted({v for _, v in kv})
        mem = self._SDM(addr_dim=self.addr_dim, data_dim=len(vals), n=800, radius=24, seed=0)
        for k, v in kv:
            onehot = [1 if vals[i] == v else 0 for i in range(len(vals))]
            for _ in range(6):
                mem.write(self._bits(k), onehot)
        votes = mem.read(self._bits(key))
        if not any(votes):
            return (
                "[honest gap] no stored key addresses close enough to '%s' for distributed recall."
                % key,
                False,
            )
        best = vals[max(range(len(vals)), key=lambda i: votes[i])]
        exact = "exact key" if any(k == key for k, _ in kv) else "distributed nearest-address"
        return (
            "[sdm] Kanerva sparse distributed memory recalled '%s' -> '%s' (%s; %d hard locations, "
            "Hamming radius %d, summed vote)." % (key, best, exact, mem.n, mem.r),
            True,
        )


class AssociativeRecallExpert:
    """dense_associative_memory.py -- modern (degree-4) Hopfield attractor. Store candidate words as
    bipolar patterns and COMPLETE a masked/partial cue by settling to the nearest stored pattern.
    Wraps dense_recall; higher-order energy = super-linear capacity above the classical bound."""

    name, module, source, uses = "dense-assoc", "dense_associative_memory", "reason", "stdlib"
    L, B = 6, 16  # 6 char slots x 16 bits = 96-dim pattern

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.dense_associative_memory import dense_recall

        self._recall = dense_recall

    @staticmethod
    def _parse(q):
        cue = None
        m = re.search(r"(?:to|denoise|autocorrect|correct|complete)\s+([a-z0-9?]+)", q.low)
        if m:
            cue = m.group(1)
        cands = []
        m = re.search(r"(?:among|from|options?|:)\s+(.+)$", q.low)
        if m:
            cands = [w for w in re.findall(r"[a-z0-9]+", m.group(1)) if w]
        return cue, cands

    def _pattern(self, word):
        vec = []
        for i in range(self.L):
            ch = word[i] if i < len(word) else " "
            if ch == "?":  # masked slot -> neutral, attractor fills it
                vec.extend([0] * self.B)
            else:
                rng = _rng("DAM|%d|%s" % (i, ch))
                vec.extend([rng.choice((-1, 1)) for _ in range(self.B)])
        return vec

    def score(self, q):
        if not q.has("closest", "nearest", "denoise", "autocorrect", "associative", "complete"):
            return 0.0
        cue, cands = self._parse(q)
        return 0.9 if (cue and len(cands) >= 2) else 0.0

    def answer(self, q):
        cue, cands = self._parse(q)
        if not cue or len(cands) < 2:
            return (
                "[honest gap] give me a cue and >=2 candidates (e.g. 'nearest to c?t among cat car dog').",
                False,
            )
        pats = [self._pattern(w) for w in cands]
        out = self._recall(pats, self._pattern(cue), degree=4, steps=3)
        best = max(range(len(cands)), key=lambda i: sum(x == y for x, y in zip(out, pats[i])))
        return (
            "[dense-am] modern degree-4 Hopfield attractor settled the cue '%s' onto stored word '%s' "
            "out of %d (higher-order energy = capacity above the classical 0.138N bound)."
            % (cue, cands[best], len(cands)),
            True,
        )


class ClusterExpert:
    """spiking_som.py -- Kohonen self-organizing map: lay numbers on a competitive map and group them
    by best-matching unit (topographic clustering). Wraps som_cluster + SOM.bmu."""

    name, module, source, uses = "som-cluster", "spiking_som", "reason", "stdlib"

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.spiking_som import som_cluster

        self._cluster = som_cluster

    def score(self, q):
        return 0.9 if (q.has("cluster", "group", "partition") and len(q.all_ints) >= 4) else 0.0

    def answer(self, q):
        nums = q.all_ints
        if len(nums) < 4:
            return ("[honest gap] give me >=4 numbers to cluster.", False)
        lo, hi = min(nums), max(nums)
        span = (hi - lo) or 1
        vecs = [[(x - lo) / span, ((x - lo) / span) ** 2, 0.0, 0.0] for x in nums]
        som, _groups = self._cluster(vecs, g=3, epochs=15, seed=0)
        bmu = {}
        for x, v in zip(nums, vecs):
            bmu.setdefault(som.bmu(v), []).append(x)
        clusters = sorted(bmu.values(), key=lambda xs: min(xs))
        desc = "; ".join("{" + ",".join(str(x) for x in sorted(xs)) + "}" for xs in clusters)
        return (
            "[spiking-som] Kohonen map grouped %d numbers into %d clusters by best-matching unit: %s."
            % (len(nums), len(clusters), desc),
            True,
        )


class PlasticityExpert:
    """triplet_stdp.py -- Pfister-Gerstner triplet STDP: net synaptic weight change from a pairing
    protocol repeated at a given frequency (captures the frequency dependence pair-STDP gets backwards).
    Wraps TripletSTDP.pairing_dw for the triplet rule and the pair-STDP control."""

    name, module, source, uses = "triplet-stdp", "triplet_stdp", "reason", "stdlib"

    def __init__(self, seed=0):
        from snn_data_lake.reasoning.triplet_stdp import TripletSTDP

        self._T = TripletSTDP

    def score(self, q):
        return (
            0.9
            if (("hz" in q.low or q.has("stdp", "plasticity", "synaptic")) and q.all_ints)
            else 0.0
        )

    def answer(self, q):
        m = re.search(r"(\d+)\s*hz", q.low)
        freq = int(m.group(1)) if m else (q.all_ints[0] if q.all_ints else 0)
        if freq <= 0:
            return ("[honest gap] give me a pairing frequency in Hz (e.g. '40 hz').", False)
        tri = self._T.triplet().pairing_dw(freq)
        pair = self._T.pair().pairing_dw(freq)
        sign = "LTP (potentiation)" if tri > 0 else "LTD (depression)"
        return (
            "[triplet-stdp] at %d Hz the Pfister-Gerstner triplet rule predicts dw=%+.3f (%s); classical "
            "pair-STDP predicts dw=%+.3f -- the triplet term captures the measured rate dependence "
            "pair-STDP gets backwards." % (freq, tri, sign, pair),
            True,
        )


# ------------------------------------------------------------- relational structure-mapping ---------
# query shape: two relational SITUATIONS as rel(a,b,..) propositions, separated by '::'
#   "map attracts(sun,planet) revolves_around(planet,sun) :: attracts(nucleus,electron) ..."
# The parenthesised proposition form is what distinguishes this from the letter analogy
# 'abc:abd :: ijk:?' that LetterExpert already answers correctly (which carries no props).

_PROP = re.compile(r"([a-z_][a-z0-9_]*)\s*\(([^()]*)\)")
_IS_TO_AS = re.compile(r"\bis\s+to\b.*\bas\b.*\bis\s+to\b")


class StructureMappingSolver:
    """vsa_structure_mapping.py -- Gentner (1983) relational STRUCTURE-MAPPING over VSA spike codes.
    Wraps StructureMapper.map/score: given two relational situations it recovers the one-to-one OBJECT
    correspondence that maximises SYSTEMATICITY -- objects match by the configuration of relations they
    participate in (sun<->nucleus), never by surface identity. This is the relations-over-relations
    analogy the letter/number Gf experts cannot do: they induce a shift within ONE sequence, this maps
    a whole relational SYSTEM onto another.

    Honesty gates, both grounded in the held-out benchmark (reasoning/analogy_bench.py, 9/12):
      - systematicity 0 -> honest gap. With no shared predicate there is no gradient to climb, so the
        returned correspondence would be an ungrounded guess (the benchmark's 'renamed_relations' case).
      - more than MAX_OBJ objects a side -> honest gap. The benchmark shows long chains settle into a
        SHIFTED near-miss local optimum instead of the true maximum, so past that width the answer is
        not trustworthy (and the search cost grows fast on a live turn)."""

    name, module, source, uses = "structure-map", "vsa_structure_mapping", "gf", "stdlib"
    DIM, MAX_OBJ = 1024, 6  # MAX_OBJ: measured trust boundary, see analogy_bench.py

    def __init__(self, seed=0):
        self.seed = seed
        self._Mapper = None  # lazy: StructureMapper on first answer (score() stays pure regex)

    @staticmethod
    def _props(chunk):
        """'rel(a, b) rel2(c)' -> [('rel','a','b'), ('rel2','c')]. Args are bare identifiers."""
        out = []
        for rel, argstr in _PROP.findall(chunk):
            args = re.findall(r"[a-z0-9_]+", argstr)
            if args:
                out.append((rel,) + tuple(args))
        return out

    def _parse(self, q):
        """Two proposition SETS separated by '::' -> (source, target); None when it is not that."""
        if "::" not in q.low:
            return None
        head, _, tail = q.low.partition("::")
        src, tgt = self._props(head), self._props(tail)
        if not src or not tgt:
            return None
        return src, tgt

    @staticmethod
    def _objects(props):
        seen = []
        for prop in props:
            for a in prop[1:]:
                if a not in seen:
                    seen.append(a)
        return seen

    def score(self, q):
        if q.analogy:  # 'abc:abd :: ijk:?' is LetterExpert's -- never hijack it
            return 0.0
        if self._parse(q):
            return 0.62  # a real relational analogy: competes, but under the single-purpose experts
        if _IS_TO_AS.search(q.low):
            return (
                0.2  # plausible relational analogy, but no propositions to ground it -> won't fire
            )
        return 0.0

    def answer(self, q):
        parsed = self._parse(q)
        if parsed is None:
            return (
                "[honest gap] give me two relational situations as propositions separated by '::' "
                "(e.g. 'attracts(sun,planet) revolves_around(planet,sun) :: "
                "attracts(nucleus,electron) revolves_around(electron,nucleus)').",
                False,
            )
        src, tgt = parsed
        wide = max(len(self._objects(src)), len(self._objects(tgt)))
        if wide > self.MAX_OBJ:
            return (
                "[honest gap] %d objects a side is past my measured search budget (%d) -- on wider "
                "systems the mapping settles into a shifted near-miss, so i will not claim one."
                % (wide, self.MAX_OBJ),
                False,
            )
        if self._Mapper is None:
            from snn_data_lake.cortex.vsa_structure_mapping import StructureMapper

            self._Mapper = StructureMapper
        mapper = self._Mapper(dim=self.DIM, seed=self.seed)  # FRESH: map() advances its own rng
        mapping = mapper.map(src, tgt)
        if not mapping:
            return ("[honest gap] i could not put the two situations into correspondence.", False)
        systematicity = mapper.score(mapping, src, tgt)
        if systematicity <= 0.0:
            return (
                "[honest gap] the two situations share no relational structure (systematicity 0.00) "
                "-- any correspondence i returned would be a guess, so i return none.",
                False,
            )
        corr = ", ".join("%s->%s" % (b, t) for b, t in sorted(mapping.items()))
        return (
            "[structure-map] relational structure-mapping (Gentner) over the spiking VSA: %s "
            "(systematicity %.2f -- %d of %d base relations survive the mapping). objects matched by "
            "their RELATIONAL role, not by surface similarity."
            % (corr, systematicity, round(systematicity * len(src)), len(src)),
            True,
        )


def _rng(tag):
    """Deterministic per-content RNG: crc32 of the tag seeds a fresh random.Random (never builtin hash)."""
    import random

    return random.Random(zlib.crc32(tag.encode()))


# ------------------------------------------------------------- relational completion (learned KB) ----
# query shapes: "<head> <relation> ?" | "what is the <relation> of <head>" | "<head>'s <relation>"
# | "<relation> of <head>". Cheap stdlib regex features, same style as the other experts.

_REL_HRQ = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+([a-z][a-z0-9_]*)\s*\?\s*$")
_REL_WHAT = re.compile(r"what(?:\s+is|'s)?\s+(?:the\s+)?([a-z][a-z0-9_]*)\s+of\s+([a-z][a-z0-9_]*)")
_REL_POSS = re.compile(r"([a-z][a-z0-9_]*)'s\s+([a-z][a-z0-9_]*)")
_REL_OF = re.compile(r"\b([a-z][a-z0-9_]*)\s+of\s+([a-z][a-z0-9_]*)")


class RelationalCompletionExpert:
    """kb_ranking.py -- LEARNED link prediction over the brain's OWN stored KB triples. A DistMult
    scorer (score(h,r,t) = <e_h, w_r, e_t>, the class from kb_ranking -- reused, not rewritten) is
    trained on the passed triples and ranks every entity as the tail of (head, relation): GENERALIZED
    relational inference -- e.g. a symmetric edge queried in the direction that was never stored --
    which the exact-lookup _kb_reply cannot do. OFFLINE by construction: trains ONLY on the passed
    triples; kb_ranking.load_dataset()/_fetch_triples() are never called. numpy + DistMult import
    LAZILY inside _ensure() on first answer, so this module stays stdlib-importable; the trained
    deterministic seed-ensemble is CACHED (train once, reuse across turns).
    Honesty gate: an out-of-vocab head/relation, or an ensemble that does not AGREE on one tail with
    a clear averaged-softmax margin, returns an honest gap (known=False) -- never a fabricated tail."""

    name, module, source, uses = "kb-completion", "kb_ranking", "reason", "numpy(lazy)"
    K, DIM, EPOCHS, LR = 5, 24, 20, 0.2  # bounded training (K models x <=30 epochs)
    MIN_AGREE, MIN_MARGIN = 0.6, 0.05  # honesty gate on the deterministic ensemble

    def __init__(self, seed=0, triples=None):
        self.seed = seed
        self.triples = sorted(
            {(str(h).lower(), str(r).lower(), str(t).lower()) for h, r, t in (triples or [])}
        )
        self._ents = sorted({h for h, _r, _t in self.triples} | {t for _h, _r, t in self.triples})
        self._rels = sorted({r for _h, r, _t in self.triples})
        self._ent_set, self._rel_set = set(self._ents), set(self._rels)
        self._exact = set(self.triples)
        self._model = None  # lazy: (np, models, e2i, r2i) on first use

    def _parse(self, q):
        m = _REL_HRQ.match(q.low)
        if m:
            return m.group(1), m.group(2)  # "<head> <relation> ?"
        m = _REL_WHAT.search(q.low)
        if m:
            return m.group(2), m.group(1)  # "what is the <relation> of <head>"
        m = _REL_POSS.search(q.low)
        if m:
            return m.group(1), m.group(2)  # "<head>'s <relation>"
        m = _REL_OF.search(q.low)
        if m:
            return m.group(2), m.group(1)  # "<relation> of <head>"
        return None

    def score(self, q):
        if not self.triples:
            return 0.0
        got = self._parse(q)
        if got is None:
            return 0.0
        head, rel = got
        in_e, in_r = head in self._ent_set, rel in self._rel_set
        if in_e and in_r:
            return 0.9  # a completion over the trained vocab
        if in_e or in_r:
            return 0.55  # recognised shape, partial vocab -> honest gap
        return 0.0

    def _ensure(self):
        """LAZY first-use build: import numpy + kb_ranking.DistMult and train the seed-ensemble on the
        passed triples ONLY (no dataset fetch, no network). Cached -> later turns just rank."""
        if self._model is None:
            import numpy as np

            from snn_data_lake.cortex.kb_ranking import (
                DistMult,  # the class ONLY, never its dataset fns
            )

            e2i = {e: i for i, e in enumerate(self._ents)}
            r2i = {r: i for i, r in enumerate(self._rels)}
            batch = max(8, min(256, len(self.triples) // 4))
            models = []
            for k in range(self.K):  # deterministic seed-ensemble (RandomState(seed+k))
                tr = np.array(
                    [[e2i[h], r2i[r], e2i[t]] for h, r, t in self.triples], dtype=np.int64
                )
                m = DistMult(len(self._ents), len(self._rels), dim=self.DIM, seed=self.seed + k)
                for _ in range(self.EPOCHS):  # bounded epochs, ONLY the passed triples
                    m.train_epoch(tr, self.LR, batch)
                models.append(m)
            self._model = (np, models, e2i, r2i)
        return self._model

    def answer(self, q):
        got = self._parse(q)
        if got is None:
            return (
                "[honest gap] phrase a completion as '<head> <relation> ?' or "
                "'what is the <relation> of <head>'.",
                False,
            )
        head, rel = got
        in_e, in_r = head in self._ent_set, rel in self._rel_set
        if not (in_e and in_r):
            miss = ([] if in_e else ["entity '%s'" % head]) + (
                [] if in_r else ["relation '%s'" % rel]
            )
            return (
                "[honest gap] %s not in the KB i trained on (%d entities, %d relations) -- "
                "no fabricated tail." % (" and ".join(miss), len(self._ents), len(self._rels)),
                False,
            )
        if len(self._ents) < 3:
            return ("[honest gap] the KB is too small to rank candidate tails.", False)
        np, models, e2i, r2i = self._ensure()
        votes = {}
        pavg = np.zeros(len(self._ents))
        for m in models:
            scores = m.rank_tails(e2i[head], r2i[rel]).astype(np.float64).copy()
            scores[e2i[head]] = -1e30  # an entity is not its own tail
            z = scores - scores.max()
            p = np.exp(np.clip(z, -30.0, 0.0))
            pavg += p / p.sum()
            top = int(np.argsort(-scores, kind="stable")[0])
            votes[top] = votes.get(top, 0) + 1
        pavg /= len(models)
        order = np.argsort(-pavg, kind="stable")
        best, second = int(order[0]), int(order[1])
        agree = votes.get(best, 0) / float(len(models))
        margin = float(pavg[best] - pavg[second])
        if agree < self.MIN_AGREE or margin < self.MIN_MARGIN:
            return (
                "[honest gap] the learned embeddings do not single out a tail for (%s, %s) "
                "(ensemble agreement %.2f, margin %.2f) -- no bluff." % (head, rel, agree, margin),
                False,
            )
        tail = self._ents[best]
        how = (
            "confirms a stored triple"
            if (head, rel, tail) in self._exact
            else "INFERRED -- not a stored triple; implied by the learned embeddings"
        )
        return (
            "[kb-completion] DistMult link prediction over %d stored triples: (%s, %s, ?) -> '%s' "
            "(ensemble %d/%d agree, margin %.2f; %s)."
            % (len(self.triples), head, rel, tail, votes.get(best, 0), len(models), margin, how),
            True,
        )


# ------------------------------------------------------------- the router (SkillRegistry / MoE) ------


# The expert scorers key on NOTATION, not on prose: they read structured query features (an
# `a:b :: c:?` analogy, `x->y` pairs, `role=filler` bindings, a Raven grid, a run of integers). That is
# a deliberate design -- the studies are symbolic -- but it is NOT discoverable: a query phrased as
# ordinary English scores 0.0 everywhere and lands in the honest gap even when the matching study is
# live and would have answered. Measured: a 16-query natural-prose spread routed to only 2 of 14
# experts, while the same intents in the notation below answered 6/8. So the gap message TEACHES the
# notation instead of only naming the studies.
_FORMATS = (
    ("gf-number-series", "what comes next in 2 4 6 8"),
    ("gf-letter", "abc:abd :: ijk:?"),
    ("gf-pair-induction", "2->4 3->9 4->16 5->?"),
    ("gf-raven / rpm-raven", "a b c / d e f / g h ?"),
    ("hdc-binding", "bind agent=john theme=ball ; what is the agent"),
    ("sdm-recall", "recall lookup a->1 b->2 c->3 : b"),
    ("dense-assoc", "nearest stored pattern to 10110"),
    ("som-cluster", "cluster 1 2 3 40 41 42"),
    ("triplet-stdp", "stdp 10 hz 20 hz plasticity"),
    ("structure-map", "solar_system: sun->planet :: atom: nucleus->?"),
)


def format_hint(limit=4, closest=None):
    """A compact 'how to phrase it' cheatsheet for the honest gap. `closest` (an expert name that
    scored above zero but under threshold) is listed FIRST -- a near miss is the most useful hint."""
    rows = list(_FORMATS)
    if closest:
        rows.sort(key=lambda r: 0 if r[0].startswith(closest) else 1)
    return "; ".join(f"{n}: '{ex}'" for n, ex in rows[:limit])


class ReasoningCortex:
    """Mixture-of-experts ROUTER over the reasoning studies. register(expert) adds a thin module
    adapter; route(query) scores every expert on cheap query features and dispatches the SINGLE best
    match (a router, not run-all). No confident expert -> an HONEST gap (known=False, never fabricated).
    Deterministic: ties break by registration order."""

    def __init__(self, threshold=0.5):
        self.experts = []
        # P1 NOTE -- this constant is a documented EXCEPTION, deliberately not baseline-relative.
        # Expert scores are hand-assigned BOUNDED confidences on [0,1] (measured vocabulary: 0.0 x11,
        # then 0.2/0.5/0.55/0.62 for weak-or-partial matches, then 0.8-0.95 for confident ones), so 0.5
        # is a dimensionless "at least half-confident" cut on an already-normalized scale -- the same
        # class as a `p >= 0.5` gate, not a magic number on a raw signal. Making it a running quantile
        # would be actively WORSE: it would drift with query MIX, so a run of confident queries raises
        # the bar and starts rejecting valid matches (the habituation failure mode that a contrastive
        # baseline exists to avoid). Revisit only if scores ever become learned/unbounded.
        self.threshold = threshold

    def register(self, expert):
        self.experts.append(expert)
        return self

    def roster(self):
        """(name, module, source, uses) for every registered expert -- which studies are LIVE."""
        return [(e.name, e.module, e.source, e.uses) for e in self.experts]

    def route(self, query):
        """Return {reply, source, confidence, known, expert}. known=True only on a confident hit, with
        source in the set live_brain._reasoning_reply accepts ('gf' | 'reason' | ...)."""
        q = Query(query)
        scored = sorted(
            ((e.score(q), i, e) for i, e in enumerate(self.experts)), key=lambda t: (-t[0], t[1])
        )  # highest score, then registration order
        best_score, _, best = scored[0] if scored else (0.0, 0, None)
        if best is None or best_score < self.threshold:
            # DISCOVERABILITY: the scorers key on notation, so an in-scope question asked in prose lands
            # here even though its study is live. Name the near miss (if any scored above zero) and show
            # how to phrase it -- still an honest gap, never a bluff, just an actionable one.
            near = best.name if (best is not None and best_score > 0.0) else None
            lead = (
                f"[honest gap] no reasoning expert is confident about this query (closest: "
                f"{near} at {best_score:.2f}, under the {self.threshold} bar)."
                if near
                else "[honest gap] no reasoning expert is confident about this query."
            )
            return {
                "known": False,
                "source": None,
                "confidence": round(best_score, 2),
                "expert": None,
                "reply": (
                    f"{lead} the live studies read STRUCTURED queries, not prose -- try, e.g.: "
                    f"{format_hint(closest=near)}. no bluff."
                ),
            }
        try:
            reply, known = best.answer(q)
        except Exception as exc:  # a broken expert never crashes the turn
            return {
                "known": False,
                "source": None,
                "confidence": round(best_score, 2),
                "expert": best.name,
                "reply": "[honest gap] the %s expert could not answer (%s)."
                % (best.name, type(exc).__name__),
            }
        return {
            "known": bool(known),
            "source": best.source if known else None,
            "confidence": round(best_score, 2),
            "expert": best.name,
            "reply": reply,
        }


def build_default_cortex(
    seed=0, triples=None, enable_recognizer_bank=False, enable_abduction=False, enable_rpm=False
):
    """Register the twelve live reasoning experts (each a thin adapter over one reasoning/ study --
    incl. the ARC grid expert, inert unless the query is an explicit 'arc:' grid task, the pair- and
    word-relation inducers, inert unless the query carries explicit example pairs/analogies, and the
    relational structure-mapping solver, inert unless the query carries two 'rel(a,b) :: rel(c,d)'
    proposition sets).
    With `triples` (the caller's OWN stored (head, relation, tail) KB facts), also register the
    RelationalCompletionExpert: LEARNED link prediction (kb_ranking.DistMult) over those triples --
    generalized relational inference past exact lookup. triples=None/empty -> the 12-expert cortex,
    byte-identical to before (the extra expert is skipped, numpy is never touched)."""
    cortex = ReasoningCortex()
    for expert_cls in (
        NumberSeriesExpert,
        LetterExpert,
        RavenExpert,
        PairInductionExpert,
        WordRelationExpert,
        BindingExpert,
        DistributedRecallExpert,
        AssociativeRecallExpert,
        ClusterExpert,
        PlasticityExpert,
        ArcExpert,
        StructureMappingSolver,
    ):
        cortex.register(expert_cls(seed=seed))
    if triples:
        cortex.register(RelationalCompletionExpert(seed=seed, triples=triples))
    if (
        triples and enable_abduction
    ):  # opt-in ABDUCTION: min-free-energy cause selection over causal triples
        from snn_data_lake.reasoning.pc_abduction import AbductionExpert

        cortex.register(
            AbductionExpert(seed=seed, triples=triples)
        )  # inert on non-causal queries; honest gap on ambiguity
    if enable_recognizer_bank:  # opt-in heterogeneous spiking recognizer ensemble
        from snn_data_lake.cortex.recognizer_bank import RecognizerBankExpert

        cortex.register(
            RecognizerBankExpert(seed=seed)
        )  # inert on ordinary text (needs a 'recognize: <64 nums>' query)
    if enable_rpm:  # opt-in spiking NVSA Raven's-matrix solver (Gf induction)
        from snn_data_lake.reasoning.rpm_solver import RpmExpert

        cortex.register(
            RpmExpert(seed=seed)
        )  # inert on ordinary text (needs a 'raven: <cells> | <candidates>' query)
    return cortex


# ------------------------------------------------------------- headless self-check -------------------


def main():
    cortex = build_default_cortex(seed=0)
    roster = cortex.roster()

    print("=" * 96)
    print(
        "REASONING CORTEX -- mixture-of-experts router over the reasoning/ studies (each now LIVE)"
    )
    print("=" * 96)
    print(
        "expert roster (%d experts; each a thin adapter over one reasoning/ module):" % len(roster)
    )
    for name, module, source, uses in roster:
        print(
            "  %-16s <- reasoning/%-27s source=%-7s backend=%s"
            % (name, module + ".py", source, uses)
        )
    print("-" * 96)

    # a spread of queries across the expert types; last two are genuinely out of domain (honest gap).
    probes = [
        ("what comes next in 2 4 6 8", "gf-number-series", "10"),
        ("continue the sequence 2 4 8 16", "gf-number-series", "32"),
        ("abc:abd :: ijk:?", "gf-letter", "ijl"),
        ("letter series a b c d then next", "gf-letter", "e"),
        ("a b c; b c a; c a ?", "gf-raven", "b"),
        ("bind color=red shape=circle size=big ; what is color", "hdc-binding", "red"),
        ("store apple->fruit banana->fruit carrot->veg ; lookup apple", "sdm-recall", "fruit"),
        ("nearest to c?t among cat car dog bird", "dense-assoc", "cat"),
        ("cluster these numbers 1 2 3 20 21 22 40 41 42", "som-cluster", None),
        ("triplet stdp weight change at 40 hz", "triplet-stdp", None),
        (
            "what comes next in 3 1 4 1 5 9 2 6",
            "gf-number-series",
            "GAP",
        ),  # in-domain but honest gap
        ("what is the capital of atlantis", None, None),  # out of domain -> gap
        ("tell me about your weekend plans", None, None),  # out of domain -> gap
    ]

    results = [(q, want_expert, want_ans, cortex.route(q)) for q, want_expert, want_ans in probes]

    print("example routes (query -> expert -> answer):")
    for q, want_expert, _want_ans, out in results:
        tag = "KNOWN" if out["known"] else "GAP  "
        exp = out["expert"] or "-"
        print("  Q: %s" % q)
        print(
            "     -> [%s conf=%.2f expert=%s src=%s] %s"
            % (tag, out["confidence"], exp, out["source"], out["reply"])
        )

    # ---- assertions ----------------------------------------------------------------------------------
    # (a) the clean in-domain queries route to the intended expert and answer plausibly
    clean = results[:10]
    routed_ok = sum(1 for _q, we, _wa, o in clean if o["expert"] == we)
    known_ok = sum(1 for _q, _we, _wa, o in clean if o["known"])
    answer_ok = sum(
        1 for _q, _we, wa, o in clean if o["known"] and (wa is None or wa in o["reply"].lower())
    )
    assert routed_ok == len(clean), (
        "every clean query must route to its intended expert (%d/%d)" % (routed_ok, len(clean))
    )
    assert known_ok >= 9, (
        "a good fraction of clean queries must answer known=True (got %d/10)" % known_ok
    )
    assert answer_ok >= 9, (
        "answered clean queries must carry the expected answer (got %d/10)" % answer_ok
    )

    # (b) an in-domain-but-unsolvable series is an HONEST gap (routed to gf, known=False, no fabrication)
    gapper = results[10][3]
    assert gapper["expert"] == "gf-number-series" and not gapper["known"], (
        "a patternless series must route to the Gf expert yet return an honest gap (no bluffed number)"
    )
    # ...and a genuinely out-of-domain query has NO confident expert (known=False, source None)
    for _q, _we, _wa, o in results[11:]:
        assert not o["known"] and o["source"] is None and o["expert"] is None, (
            "an out-of-domain query must be an honest gap with no expert claiming it"
        )

    # (c) >= 6 DISTINCT experts actually fired with a real answer (the studies are live)
    live = sorted({o["expert"] for _q, _we, _wa, o in results if o["known"]})
    assert len(live) >= 6, "at least 6 distinct experts must go live (got %d: %s)" % (
        len(live),
        live,
    )

    # (d) deterministic: routing every probe twice yields identical dicts
    for q, _we, _wa, o in results:
        assert cortex.route(q) == o, "routing must be deterministic for %r" % q

    # ---- ARC grid tasks: test-time program synthesis over 2-D grids (arc_solver) ---------------------
    arcq = "arc: 12|30 -> 03|21 ; 56|70 -> 07|65 ; test 89|10"
    arc_hit = cortex.route(arcq)
    assert arc_hit["expert"] == "arc-grid" and arc_hit["known"] and arc_hit["source"] == "gf", (
        arc_hit
    )
    assert "01|98" in arc_hit["reply"], arc_hit["reply"]  # rot180 of the test grid, induced
    arc_gap = cortex.route("arc: 11|11 -> 12|34 ; 22|22 -> 56|78 ; test 33|33")
    assert arc_gap["expert"] == "arc-grid" and not arc_gap["known"], arc_gap
    assert cortex.route(arcq) == arc_hit, "arc routing must be deterministic"
    print("arc grid expert (arc_solver test-time program synthesis over 2-D grids):")
    print("  induced     : %s" % arc_hit["reply"])
    print("  honest gap  : %s" % arc_gap["reply"])

    # ---- relational completion: LEARNED link prediction over the brain's OWN triples -----------------
    import sys

    assert "snn_data_lake.cortex.kb_ranking" not in sys.modules, (
        "kb_ranking (numpy) must not load before the relational expert's first use (lazy contract)"
    )
    kb_triples = [
        ("alice", "partner", "bob"),
        ("carol", "partner", "dan"),
        ("erin", "partner", "frank"),
        ("gina", "partner", "hank"),
        ("alice", "lives_in", "rome"),
        ("bob", "lives_in", "rome"),
        ("carol", "lives_in", "oslo"),
        ("dan", "lives_in", "oslo"),
        ("erin", "lives_in", "lima"),
        ("frank", "lives_in", "lima"),
        ("gina", "lives_in", "cairo"),
        ("hank", "lives_in", "cairo"),
    ]
    kbx = build_default_cortex(seed=0, triples=kb_triples)
    rc = kbx.experts[-1]
    assert rc.name == "kb-completion" and len(kbx.roster()) == len(roster) + 1
    assert "snn_data_lake.cortex.kb_ranking" not in sys.modules, (
        "building the cortex must not import kb_ranking/numpy -- training is lazy"
    )
    # (e) no regression: with the extra (relational) expert registered, the base experts still take
    # their queries...
    for pq, we, _wa in probes[:10]:
        assert kbx.route(pq)["expert"] == we, "expert %s must still take %r" % (we, pq)
    # ...and routing those non-relational queries never touched numpy (score() is pure stdlib)
    assert "snn_data_lake.cortex.kb_ranking" not in sys.modules, (
        "routing non-relational queries must never touch numpy"
    )
    # (offline) the expert may use kb_ranking.DistMult ONLY -- its dataset/network fns must never run:
    # stub them to raise BEFORE the first relational answer, so any call would fail the self-check.
    from snn_data_lake.cortex import kb_ranking as _kbr

    def _forbidden(*_a, **_k):
        raise AssertionError("kb_ranking dataset/network path must NEVER run (offline contract)")

    _kbr.load_dataset = _forbidden
    _kbr._fetch_triples = _forbidden
    # (a) GENERALIZED link prediction: (frank, partner, erin) is NOT a stored triple -- only the
    # reverse direction (erin, partner, frank) was ever stored -- so exact lookup CANNOT answer it;
    # the learned embeddings infer it (DistMult's trilinear score is symmetric in head/tail).
    assert ("frank", "partner", "erin") not in set(kb_triples)
    gen = kbx.route("frank partner ?")
    assert gen["expert"] == "kb-completion" and gen["known"] and gen["source"] == "reason", gen
    assert "'erin'" in gen["reply"] and "INFERRED" in gen["reply"], gen["reply"]
    gen2 = kbx.route("what is the partner of hank")
    assert gen2["known"] and "'gina'" in gen2["reply"] and "INFERRED" in gen2["reply"], gen2
    # (b) honest gaps: an out-of-vocab head engages the expert yet returns known=False; a patternless
    # query is nobody's (router-level gap) -- never a fabricated tail either way.
    gap = kbx.route("zeus partner ?")
    assert gap["expert"] == "kb-completion" and not gap["known"] and gap["source"] is None, gap
    non = kbx.route("please water the office plants")
    assert not non["known"] and non["expert"] is None, non
    # (c) deterministic: identical dicts on a second route (cached trained ensemble, stable ranking)
    assert kbx.route("frank partner ?") == gen and kbx.route("zeus partner ?") == gap

    print("-" * 96)
    print(
        "relational completion (kb_ranking.DistMult trained on the brain's OWN %d triples):"
        % len(kb_triples)
    )
    print(
        "  trained vocab: %d entities / %d relations; ensemble K=%d dim=%d epochs=%d lr=%.1f seed=0;"
        % (len(rc._ents), len(rc._rels), rc.K, rc.DIM, rc.EPOCHS, rc.LR)
    )
    print(
        "  offline: kb_ranking.load_dataset/_fetch_triples stubbed to raise -- never called; numpy lazy."
    )
    print("  generalized : 'frank partner ?' -> %s" % gen["reply"])
    print("  honest gap  : 'zeus partner ?'  -> %s" % gap["reply"])

    print("-" * 96)
    print(
        "coverage: %d/%d clean queries answered known=True; %d DISTINCT experts fired live: %s"
        % (known_ok, len(clean), len(live), ", ".join(live))
    )
    print(
        "honest gaps: 1 in-domain unsolvable series + %d out-of-domain queries returned known=False "
        "(no fabrication); routing deterministic." % (len(results) - 11)
    )
    print("=" * 96)
    print(
        "self-check OK: the MoE router made %d dead reasoning studies LIVE -- it dispatches a text query "
        "to the best-matching expert (Gf induction, HDC binding, Kanerva SDM, dense associative memory, "
        "spiking SOM, triplet-STDP), answers from the real module, and returns an HONEST gap when no "
        "expert is confident. Deterministic, offline." % len(live)
    )
    return 0


if __name__ == "__main__":
    main()
