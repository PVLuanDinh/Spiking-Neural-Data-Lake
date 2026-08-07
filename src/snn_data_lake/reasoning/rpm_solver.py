r"""Raven's Progressive Matrices (RPM) solver -- a spiking neuro-vector-symbolic (NVSA) Gf lift.

The brain's general FLUID INTELLIGENCE is near the floor (ARC ~2.75%). RPM is the tractable slice
where the grounded method has a real number: NVSA (Hersche et al. 2023, arXiv:2203.04571) reaches
~88% on the RAVEN benchmark by inducing the puzzle's rule in a vector-symbolic algebra and executing
it over role-filler object codes. This module runs NVSA's backend over the brain's OWN spiking VSA.

WHY THIS RUNS WITH ZERO NUMERICAL CHANGE. cortex/vsa_reasoning.py is a two-line spike-polarity code
whose docstring states it is "numerically lossless w.r.t. the dense HD algebra" -- i.e. the balanced
+/-1 spike code IS NVSA's bipolar +/-1 codebook, and bind/bundle/permute/sim/cleanup are exactly
NVSA's Hadamard-product / superposition / fractional-power / cosine / nearest-neighbour ops. So NVSA
runs here with the same margins as the dense version; the genuine change is the substrate (discrete
spikes) + the spike-domain operators, not the numbers. Every op below is a spike op.

DESIGN (all over vsa_reasoning primitives):
  * ROLE-FILLER OBJECT / PANEL. An object = fold-bind(POS_role, type, size, color, ...) -- a 4-way
    Hadamard (spike-XOR) bind, NVSA's product object vector (position is the ROLE, attributes are
    FILLERS). A panel = bundle(objects) -- spike superposition; quasi-orthogonal codes keep the
    objects separately recoverable, so the superposition catastrophe is avoided. (Single-object
    panels reduce bundle to the object vector; the machinery generalises to multi-object panels.)
  * EXAMPLE -> RULE INDUCTION per attribute -- the core Gf move -- testing NVSA's rule set as VSA ops:
      (a) DISTRIBUTE-THREE: r_row = fold-bind of a complete row's three cell codes. Because spike-XOR
          bind is COMMUTATIVE it is multiset-invariant, so two rows drawn from the same 3-value set
          give sim(r1, r2) ~ 1 -- that FIRES the rule. EXECUTE the missing cell by unbinding the two
          knowns out of the row-invariant: cleanup(unbind(r_row, bind(cell31, cell32))).
      (b) PROGRESSION / CONSTANT: reuse reasoning/gf.py's shift induction. code(n) = permute(ANCHOR,n)
          is fractional-power-encoding-by-permutation, so a constant permute-shift s across a row is a
          progression (s != 0) or constant (s == 0). gf.induce_shift returns that s or None; EXECUTE
          the missing cell as permute(cell31_code, s), read out over the ordinal codebook.
      (c) ARITHMETIC: number/size laid on the SAME permute-power ordinal line, so a + b is a
          permute-compose: code(a + b) = permute(code(a), b). EXECUTE the missing cell by composing
          the two known permute-powers, read out over the ordinal codebook.
  * SPIKING SELECTION (not a bare argmax). The induced rule per attribute is chosen by routing the
    fitting rule types' evidence through the spiking basal-ganglia Go/NoGo gate
    (basal_ganglia.action_gating.BasalGangliaGate.select -- disinhibition WTA). The winning ANSWER
    among the 8 candidates is chosen by the spiking thalamic global workspace
    (thalamus.global_workspace.GlobalWorkspace) -- each candidate's rule-match salience competes,
    and a candidate that crosses the all-or-none IGNITION threshold and ranks inside the workspace's
    capacity is ADMITTED to a slot; the highest-ranked admitted candidate of ours is the answer.
  * SOLVE. RPM is 8-choice: score every candidate by the induced per-attribute predictions
    (spike sim + cleanup), ignite the winner, and return {answer_idx, confidence, rule, known}.

HONEST CEILING (stated, not hidden). This is BOUNDED to the NVSA rule set -- distribute-three,
progression, constant, arithmetic -- over CLEAN attribute codes (one value per attribute per cell).
It is NOT full ARC, NOT open-domain abstraction, and NOT the perception front-end (it consumes
already-parsed attribute cells, as NVSA's neuro-symbolic backend does). A grid whose attributes obey
none of these four rules returns an HONEST gap (known=False), never a bluffed choice.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.reasoning.rpm_solver
"""

# ruff: noqa: UP031  -- %-format is the hand-aligned house style of every sibling in reasoning/
# (which the repo excludes from `ruff check .`); kept for consistency, not reflowed to f-strings.
import random
import re
import zlib

from snn_data_lake.basal_ganglia.action_gating import BasalGangliaGate
from snn_data_lake.cortex.vsa_reasoning import bind, bundle, cleanup, hv, permute, sim
from snn_data_lake.reasoning.gf import code, induce_shift
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

# NVSA rule set, in Occam order (simplest first). Rule selection routes evidence for the FITTING
# types through the spiking basal-ganglia gate; the priority below is the per-channel drive so the
# gate's disinhibition WTA releases the simplest fitting rule.
RULE_ORDER = ("constant", "progression", "arithmetic", "distribute3")
RULE_PRIORITY = {"constant": 4.0, "progression": 3.0, "arithmetic": 2.0, "distribute3": 1.0}

_ANSWER = "rpm_answer"  # namespace tag on the contents this solver posts into the (shared) slots
_POS = hv("RPM_POS0")  # the single object-position ROLE (fillers = the attribute codes)
_POOL = tuple(range(1, 9))  # candidate value pool for the self-check's synthetic distractors
_ROLE_CACHE = {}  # attribute -> its ROLE spike code (crc32-seeded, deterministic)


def _solve_tag(context_cells, candidates):
    """The per-QUESTION token this solve stamps on every content it posts into the workspace.

    WHY A BARE `_ANSWER` TAG IS NOT ENOUGH ON A SHARED BUS, measured rather than argued. The tag
    alone namespaces this solver against OTHER FACULTIES (a percept posting a small int cannot be
    mistaken for a candidate index). It does not namespace this solve against ANOTHER SOLVE, and
    the workspace holds a content's attractor charge ACROSS ticks -- so a previous matrix's winner
    is still in the bus when the next matrix is posed, wearing the identical `('rpm_answer', i)`
    shape. Measured on two ordinary consecutive `solve()` calls on one shared bus, no adversary
    involved: matrix A's winner held the rank-1 slot at charge 0.85 and matrix B, whose own answer
    is #7, returned #0 -- A's answer, read back as B's. That is a foreign content returned as the
    RPM answer, which is exactly what this solver's shared-bus criterion forbids.

    The token is a crc32 over the QUESTION (its context cells and its candidates, attribute-sorted
    so dict order cannot reach it), so it identifies the competition rather than the caller: two
    solves of the SAME matrix legitimately share pools and their attractor persistence is
    meaningful, while two DIFFERENT matrices get disjoint pools because they are different
    competitions. crc32 rather than the builtin `hash()` per the house determinism rule -- a
    namespace that changes between processes is not a namespace.

    TWO HONEST LIMITS, disclosed here rather than left to be discovered later.

    (1) IT IS A NAMESPACE, NOT A SIGNATURE, and is not claimed as one. The content is unforgeable
    only against a poster that does not know the matrix; anything that DOES know the matrix is
    posting this matrix's own answer, which is not the contamination this guards. What it buys is
    provenance against the contents that actually share this bus -- other faculties, and this
    solver's own earlier questions.

    (2) IT COSTS POOLS ON A SHARED BUS, measured against the retired untagged shape rather than
    estimated. Distinct matrices no longer reuse one set of 8 pools, and a pool survives ~85
    workspace STEPS (`decay` 0.85 down to the 1e-6 prune). WORST CASE -- this solver is the only
    thing stepping the bus and every solve is a different matrix -- 160 back-to-back solves plateau
    at 519 live pools and 17.2 ms/solve against 8 pools and 11.5 ms untagged: 1.5x, and it
    PLATEAUS rather than growing, because the prune keeps pace from ~80 solves on. It drains as
    soon as anything else ticks the bus, which on the live brain is every wake tick: the same 160
    solves peak at 279 pools with 1 brain tick between them, 102 with 5, and 33 with 20. The cost
    is therefore paid in full only by a bench harness and is near-zero in production. It is the
    direct price of not conflating two different questions in one competition, which is what the
    untagged shape did and what the measurement above shows it getting wrong."""
    parts = [
        ",".join("%s=%r" % (a, cell[a]) for a in sorted(cell))
        for cell in list(context_cells) + list(candidates)
    ]
    return zlib.crc32("|".join(parts).encode("utf-8"))


def _role(attr):
    """The spike ROLE code an attribute binds its value FILLER to in a panel record (deterministic)."""
    if attr not in _ROLE_CACHE:
        _ROLE_CACHE[attr] = hv("RPM_ROLE|%s" % attr)
    return _ROLE_CACHE[attr]


def _fold_bind(vecs):
    """Fold spike-XOR bind over a list -> the NVSA product object/row vector (bind is associative)."""
    out = vecs[0]
    for v in vecs[1:]:
        out = bind(out, v)
    return out


def _readout(code_vec, book):
    """Snap a predicted spike code to the codebook, accepting ONLY an EXACT match (the induced op must
    land on a real symbol, not a near miss). Returns the label or None -- an honest gap, never a bluff."""
    if not book:
        return None
    label, _ = cleanup(code_vec, book)
    return label if book[label] == code_vec else None


class RpmSolver:
    """Spiking NVSA solver for 3x3 Raven's Progressive Matrices (8-choice).

    solve(context_cells, candidates) -> {answer_idx, confidence, rule, known}
      context_cells : the 8 known cells row-major -- (0,0)(0,1)(0,2)(1,0)(1,1)(1,2)(2,0)(2,1);
                      each cell a dict {attribute: value} (int values enable progression/arithmetic).
      candidates    : the 8 candidate cells (same dict shape); exactly one completes cell (2,2).
    induce(rows) -> a rule descriptor {rules, predicted, n_ruled} for the matrix rows.

    Deterministic: random.Random(seed) per instance; codes are crc32-seeded (vsa_reasoning.hv /
    gf.code); no wall-clock, no global RNG. Guarded: any failure returns an honest gap, never raises.
    """

    def __init__(self, seed=0, workspace=None):
        """`workspace` ADOPTS the one shared GlobalWorkspace instead of building a private one per
        solve. Hand it `Brain.workspace` / `LiveBrain.workspace` -- the single instance
        `brainstem/autonomous_loop.py` constructs -- and the candidate answers compete for the SAME
        limited-capacity slots the brain's perception and salience channels post into. None keeps
        the per-call private instance this class always built, so every existing caller is
        byte-identical.

        The handle is held on the INSTANCE and read inside `_solve`, because the workspace there is
        function-local and re-created on every call: a constructor that only built a private object
        once would still hand each solve a fresh private bus."""
        self.seed = seed
        self.rng = random.Random(seed)
        self.workspace = workspace
        self.bg = BasalGangliaGate(seed=seed)  # spiking Go/NoGo rule selection (stateless select)

    # -- codebooks ------------------------------------------------------------------------------
    def _books(self, cells):
        """Build per-attribute spike codebooks over every value seen (context + candidates).
        cat  : categorical value -> spike code (for constant / distribute-three / candidate scoring).
        ordb : integer value -> fractional-power ordinal code (for progression / arithmetic readout).
        Returns (attrs, int_attrs, cat, ordb)."""
        attrs = sorted({a for c in cells for a in c})
        cat = {}
        seen = {a: set() for a in attrs}
        for c in cells:
            for a, v in c.items():
                seen[a].add(v)
                cat.setdefault((a, v), hv("RPM|%s|%r" % (a, v)))
        int_attrs = sorted(a for a in attrs if seen[a] and all(isinstance(v, int) for v in seen[a]))
        hi = 0
        for a in int_attrs:
            hi = max(hi, max(seen[a]))
        ordb = {n: code(n) for n in range(0, 2 * hi + 6)}  # covers sums / progression overshoot
        return attrs, int_attrs, cat, ordb

    # -- per-attribute rule induction (example -> rule, the core Gf move) ------------------------
    def _induce_attr(self, attr, rc, complete, int_attrs, cat, ordb):
        """Induce ONE NVSA rule for `attr` from the complete rows and predict the missing cell (2,2).
        Returns (rule_name, predicted_value, evidence_weight) or None (this attribute has no rule)."""

        # values by (row, col); complete = rows with all 3 cells; (2,0),(2,1) are the row-2 knowns.
        def val(r, c):
            return rc.get((r, c))

        if (2, 0) not in rc or (2, 1) not in rc:
            return None

        # (a) CONSTANT: every complete row holds one repeated value; row 2's two knowns agree too.
        if all(val(r, 0) == val(r, 1) == val(r, 2) for r in complete) and val(2, 0) == val(2, 1):
            return ("constant", val(2, 0), RULE_PRIORITY["constant"])

        is_int = attr in int_attrs
        # (b) PROGRESSION: a CONSTANT permute-shift across each row (gf.induce_shift over ordinal codes).
        if is_int:
            shifts = []
            for r in complete:
                s = induce_shift([code(val(r, 0)), code(val(r, 1)), code(val(r, 2))])
                shifts.append(s)
            s0 = shifts[0] if shifts else None
            if (
                s0 is not None
                and s0 != 0
                and all(s == s0 for s in shifts)
                and val(2, 1) - val(2, 0) == s0
            ):
                pred = _readout(
                    permute(code(val(2, 1)), s0), ordb
                )  # EXECUTE: permute the last known
                if pred is not None:
                    return ("progression", pred, RULE_PRIORITY["progression"])

            # (c) ARITHMETIC: col3 = col1 (+/-) col2 on the shared permute-power line (permute-compose).
            for op, sign in (("add", 1), ("sub", -1)):
                ok = all((val(r, 0) + sign * val(r, 1)) == val(r, 2) for r in complete)
                if ok:
                    pred_val = val(2, 0) + sign * val(2, 1)
                    if pred_val < 0:
                        continue
                    pred = _readout(
                        permute(code(val(2, 0)), sign * val(2, 1)), ordb
                    )  # compose permutes
                    if pred is not None and pred == pred_val:
                        return ("arithmetic:%s" % op, pred, RULE_PRIORITY["arithmetic"])

        # (d) DISTRIBUTE-THREE: each complete row is a permutation of the SAME 3-value set. Commutative
        #     spike-XOR bind is multiset-invariant, so the row binds coincide (sim ~ 1) -> the rule
        #     FIRES; execute the missing cell by unbinding the two knowns out of the row invariant.
        rowsets = [frozenset((val(r, 0), val(r, 1), val(r, 2))) for r in complete]
        if complete and all(len(s) == 3 for s in rowsets) and all(s == rowsets[0] for s in rowsets):
            triple = rowsets[0]
            knowns = (val(2, 0), val(2, 1))
            if len(set(knowns)) == 2 and set(knowns) < set(triple):
                r_ref = _fold_bind(
                    [
                        cat[(attr, v)]
                        for v in (val(complete[0], 0), val(complete[0], 1), val(complete[0], 2))
                    ]
                )
                # spike-XOR is self-inverse, so unbind == bind: r_ref XOR (know0 XOR know1) = the missing.
                predicted_code = bind(r_ref, bind(cat[(attr, knowns[0])], cat[(attr, knowns[1])]))
                subbook = {v: cat[(attr, v)] for v in triple}
                pred = _readout(predicted_code, subbook)
                if pred is not None:
                    return ("distribute3", pred, RULE_PRIORITY["distribute3"])
        return None

    def _select_rule(self, fits):
        """Route the fitting rule types' evidence through the spiking basal-ganglia Go/NoGo gate so
        the induced rule is DISINHIBITED, not argmax-ed. `fits` maps rule_name -> (value, weight).
        Returns the released rule name (highest-priority fitting rule) or None (no rule -> gap)."""
        drive = [fits[r][1] if r in fits else 0.0 for r in RULE_ORDER]
        if not any(d > 0 for d in drive):
            return None
        res = self.bg.select(drive, da=0.80)  # intact dopamine -> release the winner
        win = res["winner"]
        if win is None:  # tonic GPi never overcome -> fall back
            win = res["disinhibited"]
        return RULE_ORDER[win]

    # -- public induction -----------------------------------------------------------------------
    def induce(self, rows):
        """Induce the per-attribute rules for a matrix. `rows` = [[c00,c01,c02],[c10,c11,c12],[c20,c21]]
        (row 2 has the two knowns; (2,2) is missing). Returns {rules, predicted, n_ruled}."""
        cells = [c for row in rows for c in row]
        attrs, int_attrs, cat, ordb = self._books(cells)
        return self._induce_all(rows, attrs, int_attrs, cat, ordb)

    def _induce_all(self, rows, attrs, int_attrs, cat, ordb):
        rc = {}
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                rc[(r, c)] = cell
        complete = [r for r in range(len(rows)) if len(rows[r]) == 3]
        rules, predicted = {}, {}
        for a in attrs:
            arc = {(r, c): cell.get(a) for (r, c), cell in rc.items()}
            fits = {}
            got = self._induce_attr(a, arc, complete, int_attrs, cat, ordb)
            if got is not None:
                rule, value, weight = got
                base = rule.split(":", 1)[0]
                fits[base] = (value, weight)
            if not fits:
                continue
            chosen = self._select_rule(fits)  # spiking basal-ganglia selection
            if chosen is None or chosen not in fits:
                continue
            rules[a] = (rule if rule.split(":", 1)[0] == chosen else chosen, fits[chosen][0])
            predicted[a] = fits[chosen][0]
        return {"rules": rules, "predicted": predicted, "n_ruled": len(predicted)}

    # -- solve ----------------------------------------------------------------------------------
    def solve(self, context_cells, candidates):
        """Solve one 8-choice RPM. Returns {answer_idx, confidence, rule, known}. Guarded: on any
        failure returns an honest gap (known=False, answer_idx=0), never raises."""
        try:
            return self._solve(context_cells, candidates)
        except Exception as exc:  # honest gap, never a crash
            return {
                "answer_idx": 0,
                "confidence": 0.0,
                "known": False,
                "rule": "gap:%s" % type(exc).__name__,
            }

    def _solve(self, context_cells, candidates):
        if len(context_cells) != 8 or len(candidates) < 2:
            return {"answer_idx": 0, "confidence": 0.0, "known": False, "rule": "gap:malformed"}
        cells = list(context_cells) + list(candidates)
        attrs, int_attrs, cat, ordb = self._books(cells)
        rows = [context_cells[0:3], context_cells[3:6], context_cells[6:8]]
        induced = self._induce_all(rows, attrs, int_attrs, cat, ordb)
        predicted, n_ruled = induced["predicted"], induced["n_ruled"]

        if n_ruled == 0:  # no NVSA rule fits any attribute -> gap
            return {"answer_idx": 0, "confidence": 0.0, "known": False, "rule": "gap:no-rule"}

        # NVSA PANEL VECTORS: each cell -> bundle of role-filler bindings (attribute ROLE bound to its
        # value FILLER, superposed). The predicted panel is the induced completion; a candidate's spike
        # salience is its coincidence with it. Superposition keeps the fillers quasi-orthogonal, so the
        # panels grade cleanly by how many role-filler pairs match. An exact per-attribute match count
        # (also a spike-sim test) drives the robust decision; the panel salience drives the ignition.
        def _fill(a, v):
            return cat.get((a, v)) or hv("RPM|%s|%r" % (a, v))

        pred_panel = bundle([bind(_role(a), _fill(a, pv)) for a, pv in predicted.items()])

        int_match, saliences = [], []
        for cand in candidates:
            m = sum(
                1
                for a, pv in predicted.items()
                if cand.get(a) is not None and sim(_fill(a, cand[a]), _fill(a, pv)) > 0.99
            )
            int_match.append(m)
            pairs = [
                bind(_role(a), _fill(a, cand[a])) for a in predicted if cand.get(a) is not None
            ]
            saliences.append(max(0.0, sim(bundle(pairs), pred_panel)) if pairs else 0.0)

        best_m = max(int_match)
        winners = [i for i, m in enumerate(int_match) if m == best_m]
        argmax = winners[0]

        # SPIKING candidate selection: each candidate's panel salience competes in the thalamic global
        # workspace; a candidate that crosses the all-or-none ignition threshold and ranks inside the
        # capacity is ADMITTED to a slot, and the highest-ranked admitted candidate of OURS is the
        # answer. ADOPTS the one shared bus when this solver was handed it (P6), else builds the
        # private per-call instance it always built. On the shared bus the competition also contains
        # whatever the rest of the brain is holding, so a candidate must out-drive the live tenants'
        # cross-tick attractors to ignite -- which is the consumption, not an added reader.
        ws = GlobalWorkspace(seed=self.seed) if self.workspace is None else self.workspace
        tag = _solve_tag(context_cells, candidates)
        for i, s in enumerate(saliences):
            ws.submit("c%d" % i, (_ANSWER, tag, i), s)
        ws.step()  # settle the competition; the workspace now holds whatever was admitted
        # READ ACCESS, NOT FOCUS -- and read only contents THIS SOLVE POSTED.
        #
        # `step()` returns the ranked HEAD, and on a multi-slot bus the head is a different question
        # from access: a compliant specialist can always win a SLOT and cannot always become the
        # FOCUS (`thalamus/global_workspace.py`, limit 3). Reading the head therefore reports an
        # honest gap whenever any louder faculty is also in the workspace, which is the single-slot
        # "a competitor took the one slot away from me" pathology and not a fact about this solver.
        # `broadcast_all()` is the arbiter's full ranked output and is the read that matches the
        # bottleneck we are actually competing in.
        #
        # The token match is the provenance half and it is doing the safety work: `broadcast_all()`
        # reports CONTENTS, not sources, so a content that merely LOOKS like one of ours -- another
        # faculty's, or this solver's own answer to a previous matrix still holding its charge --
        # is indistinguishable from a candidate by shape alone. Matching the per-question token
        # means `won` can only be a content this solve posted, so `answer_idx` and `known` below
        # are statements about OUR candidate winning access and never about who else is on the bus.
        # BEHAVIOUR-NEUTRAL ON THE PRIVATE PATH, which is every existing caller (`workspace=None`),
        # and that is MEASURED rather than argued: A/B'd against a copy of this file carrying the
        # exact retired read, 250 fresh-solver matrices across all five regimes and 200 more on a
        # REUSED instance gave 0 differing result dicts. The reason it holds is that the drives,
        # the jitter draw order and the sorted pool order are all keyed on the source label "c%d",
        # which is unchanged, and a private workspace holds nothing but this solve's own contents
        # so its ranked head IS its top candidate.
        mine = [
            c
            for c in ws.broadcast_all()
            if isinstance(c, tuple) and len(c) == 3 and c[:2] == (_ANSWER, tag)
        ]
        won = mine[0] if mine else None  # our highest-RANKED admitted candidate, or None
        answer_idx = won[2] if won is not None else argmax

        srt = sorted(saliences, reverse=True)
        confidence = round(srt[0] - (srt[1] if len(srt) > 1 else 0.0), 3)
        known = len(winners) == 1 and best_m == n_ruled and won is not None and answer_idx == argmax

        rule_txt = ", ".join(
            "%s:%s" % (a, rules_a[0]) for a, rules_a in sorted(induced["rules"].items())
        )
        return {
            "answer_idx": answer_idx,
            "confidence": confidence,
            "known": bool(known),
            "rule": rule_txt if known else ("low-confidence (%s)" % rule_txt),
        }


# ---------------------------------------------------------------- reasoning_cortex expert adapter ----
# A thin route(text)-compatible expert (name/module/source/uses + score/answer), mirroring
# reasoning_cortex's RelationalCompletionExpert/AbductionExpert. Inert unless the query is an explicit
# 'raven:' RPM grid, so it never collides with the existing RavenExpert (which fires on the simpler
# 3x3 distribution-of-three symbol grid). Register it in build_default_cortex to make the study live.

_KV = re.compile(r"([a-z][a-z0-9_]*)\s*=\s*(-?\d+|[a-z0-9_]+)")


def _parse_raven(text):
    """Parse a 'raven:' RPM query into (context_cells, candidates), or None on any malformed input.
    Format: raven: <c0> ; ... ; <c7> | <cand0> ; ... ; <candN>   with each cell 'k=v,k=v,...'."""
    low = text.strip()
    if not low.lower().startswith("raven:") or "|" not in low:
        return None
    body = low.split(":", 1)[1]
    ctx_part, cand_part = body.split("|", 1)

    def cells(part):
        out = []
        for chunk in part.split(";"):
            d = {}
            for k, v in _KV.findall(chunk.lower()):
                d[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
            if d:
                out.append(d)
        return out

    context, candidates = cells(ctx_part), cells(cand_part)
    if len(context) != 8 or len(candidates) < 2:
        return None
    return context, candidates


class RpmExpert:
    """rpm_solver.py -- Raven's Progressive Matrices via the spiking NVSA solver (role-filler object
    codes, example->rule induction over the NVSA rule set, basal-ganglia rule gate + global-workspace
    answer ignition). Inert unless the query is an explicit 'raven:' RPM grid. Honest gap when no NVSA
    rule fits the matrix -- never a bluffed choice."""

    name, module, source, uses = "rpm-raven", "rpm_solver", "gf", "stdlib"

    def __init__(self, seed=0, workspace=None):
        # `workspace` is threaded straight through to the solver so an expert built from a live
        # brain answers on the ONE bus rather than on a private one (P6). None = unchanged.
        self._solver = RpmSolver(seed=seed, workspace=workspace)

    def score(self, q):
        t = getattr(q, "low", "").strip()
        return 0.94 if (t.startswith("raven:") and "|" in t) else 0.0

    def answer(self, q):
        parsed = _parse_raven(getattr(q, "text", ""))
        if parsed is None:
            return (
                "[honest gap] i expected an 8-cell 'raven:' RPM grid plus candidates "
                "('raven: <8 cells> | <candidates>'), each cell 'type=..,size=..,color=..'.",
                False,
            )
        context, candidates = parsed
        res = self._solver.solve(context, candidates)
        if not res["known"]:
            try:  # OUT-OF-GRAMMAR fallback: synthesize a NOVEL rule (no fixed menu)
                from snn_data_lake.reasoning.vsa_synthesize import VsaSynthesizer

                if getattr(self, "_synth", None) is None:
                    self._synth = VsaSynthesizer(seed=0)
                syn = self._synth.solve(context, candidates)
            except Exception:
                syn = {"known": False}
            if syn.get("known"):
                return (
                    "[rpm/synth] the fixed NVSA rules gapped, so i SYNTHESIZED a novel rule as a VSA spike-op "
                    "program (%s) -> answer is candidate #%d (confidence %.2f). out-of-grammar, no fixed menu."
                    % (syn.get("program"), syn["answer_idx"], syn["confidence"]),
                    True,
                )
            return (
                "[honest gap] neither the fixed NVSA rules nor out-of-grammar VSA synthesis completes this "
                "matrix over its attribute codes -- no bluffed choice.",
                False,
            )
        return (
            "[rpm] induced %s at test time -> answer is candidate #%d (confidence %.2f), chosen by "
            "spiking role-filler NVSA over the brain's VSA. bounded to the NVSA rule set, not full ARC."
            % (res["rule"], res["answer_idx"], res["confidence"]),
            True,
        )


# ---------------------------------------------------------------- self-check --------------------------


def _rotate(seq, k):
    k %= len(seq)
    return seq[k:] + seq[:k]


def _candidates(answer, attrs, rng, pool=_POOL, n=8):
    """Build n distinct candidate cells including the true answer; distractors perturb 1..len(attrs)
    attributes to different in-pool values. Returns (candidates, answer_idx). Deterministic under rng."""
    cands = [dict(answer)]
    seen = {tuple(answer[a] for a in attrs)}
    guard = 0
    while len(cands) < n and guard < 2000:
        guard += 1
        c = dict(answer)
        for a in rng.sample(attrs, rng.randint(1, len(attrs))):
            c[a] = rng.choice([x for x in pool if x != answer[a]])
        key = tuple(c[a] for a in attrs)
        if key not in seen:
            seen.add(key)
            cands.append(c)
    rng.shuffle(cands)
    idx = next(i for i, c in enumerate(cands) if all(c[a] == answer[a] for a in attrs))
    return cands, idx


def _build_task(kind, rng):
    """Hand-build one 3x3 RPM (attrs type/size/color) under a named rule regime; return (context,
    candidates, answer_idx, answer). Row 2 col 2 is the held-out cell."""
    attrs = ["color", "size", "type"]
    grid = {}  # attr -> [[r0c0,r0c1,r0c2],[...],[...]]
    if kind == "distribute3":
        base = {"type": [1, 2, 3], "size": [3, 1, 2], "color": [2, 3, 1]}
        for a in attrs:
            grid[a] = [_rotate(base[a], r) for r in range(3)]
    elif kind == "progression":
        starts = {"type": [1, 2, 3], "size": [1, 1, 1], "color": [2, 3, 4]}
        steps = {"type": 1, "size": 2, "color": 1}
        for a in attrs:
            grid[a] = [[starts[a][r] + steps[a] * c for c in range(3)] for r in range(3)]
    elif kind == "constant":
        vals = {"type": [1, 2, 3], "size": [2, 2, 2], "color": [3, 1, 2]}
        for a in attrs:
            grid[a] = [[vals[a][r]] * 3 for r in range(3)]
    elif kind == "arithmetic":
        cols = {
            "type": ([1, 2, 1], [2, 1, 3]),
            "size": ([1, 1, 2], [1, 3, 2]),
            "color": ([2, 1, 1], [1, 2, 3]),
        }
        for a in attrs:
            c0, c1 = cols[a]
            grid[a] = [[c0[r], c1[r], c0[r] + c1[r]] for r in range(3)]
    elif kind == "norule":
        # each row a DIFFERENT value set, no constant/progression/arithmetic -> no NVSA rule fires.
        ragged = {
            "type": [[1, 2, 4], [3, 5, 6], [7, 8]],
            "size": [[2, 5, 3], [6, 1, 4], [7, 2]],
            "color": [[3, 6, 2], [1, 4, 7], [5, 3]],
        }
        context = []
        for r in range(3):
            for c in range(len(ragged["type"][r])):
                context.append({a: ragged[a][r][c] for a in attrs})
        candidates = [
            {"type": v, "size": (v % 5) + 1, "color": ((v + 2) % 5) + 1} for v in range(1, 9)
        ]
        return context, candidates, 0, None

    context = []
    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                continue
            context.append({a: grid[a][r][c] for a in attrs})
    answer = {a: grid[a][2][2] for a in attrs}
    candidates, idx = _candidates(answer, attrs, rng)
    return context, candidates, idx, answer


def main():
    print("=" * 78)
    print(
        "SPIKING NVSA RAVEN'S PROGRESSIVE MATRICES SOLVER (role-filler VSA over the brain's spikes)"
    )
    print("=" * 78)
    solver = RpmSolver(seed=0)
    rng = random.Random(0)

    passed = []
    for kind in ("distribute3", "progression", "constant", "arithmetic"):
        context, candidates, idx, _answer = _build_task(kind, rng)
        res = solver.solve(context, candidates)
        ok = res["known"] and res["answer_idx"] == idx
        passed.append(ok)
        print(
            "  [%s] %-12s -> answer #%d (truth #%d) known=%s conf=%.2f rule='%s'"
            % (
                "PASS" if ok else "FAIL",
                kind,
                res["answer_idx"],
                idx,
                res["known"],
                res["confidence"],
                res["rule"],
            )
        )

    # role-filler object binding is genuine: bind POS+type+size+color, then unbind three factors to
    # recover the fourth by spike cleanup (the NVSA product-object recovery, in spikes).
    tv, sv, cv = hv("RPM|type|1"), hv("RPM|size|2"), hv("RPM|color|3")
    obj = _fold_bind([_POS, tv, sv, cv])
    book = {"type1": tv, "typeX": hv("RPM|type|9"), "typeY": hv("RPM|type|7")}
    recovered = _readout(bind(bind(bind(obj, _POS), sv), cv), book)
    bind_ok = recovered == "type1"
    print(
        "  [%s] role-filler   -> unbound the object's 'type' filler back to %r (spike cleanup)"
        % ("PASS" if bind_ok else "FAIL", recovered)
    )

    # honest gap: a grid whose attributes obey NO NVSA rule must return known=False (no bluff).
    ctx_nr, cand_nr, _idx, _ans = _build_task("norule", rng)
    gap = solver.solve(ctx_nr, cand_nr)
    gap_ok = not gap["known"]
    print(
        "  [%s] no-rule grid  -> known=%s conf=%.2f rule='%s' (must be an honest gap)"
        % ("PASS" if gap_ok else "FAIL", gap["known"], gap["confidence"], gap["rule"])
    )

    # determinism: identical result on a second solve of the same task.
    ctx_d, cand_d, _i, _a = _build_task("distribute3", random.Random(0))
    det_ok = solver.solve(ctx_d, cand_d) == RpmSolver(seed=0).solve(ctx_d, cand_d)
    print(
        "  [%s] determinism   -> same task + seed reproduces the identical result dict"
        % ("PASS" if det_ok else "FAIL")
    )

    # ---- P6: the ADOPTED workspace, and what a SHARED bus can and cannot do to this solver ------
    # A handed-in bus must survive every solve (the workspace here is function-local and rebuilt
    # per call, so a constructor that merely stored a private object would still hand each solve a
    # fresh private one), and an EMPTY adopted bus must reproduce the private answer exactly.
    #
    # RE-DERIVED 2026-08-06 FOR THE MULTI-SLOT ARBITER. The property being measured is unchanged
    # and is the one this check has always been about:
    #
    #     THIS SOLVER'S ANSWER AND ITS `known` FLAG MUST BE PRODUCED BY ITS OWN CANDIDATE WINNING
    #     ACCESS ON THIS SOLVE -- never by a foreign content that happens to be on the bus.
    #
    # Only the MEASUREMENT changed, because the old one measured that property through a proxy the
    # arbiter no longer supports. Under the retired SINGLE-SLOT workspace, "a foreign content holds
    # the slot" and "nothing of mine is in the workspace" were THE SAME FACT, so `not known` was a
    # sound reading of own-provenance. A limited-capacity workspace separates them, and it does so
    # in BOTH directions -- so the proxy is now wrong in both, which is why it is replaced rather
    # than relaxed:
    #
    #   * ours can be in the workspace WITH a foreign content -- the old check then demanded an
    #     honest gap that is no longer honest, which is the FALSE FAIL that prompted this pass;
    #   * a foreign content can be in the workspace with NOTHING of ours -- and that case is where
    #     the hazard actually lives. Measured before this criterion was written: with the capacity
    #     full of foreign contents, ZERO slots held by this solver's own candidate pools, and one
    #     foreign content merely SHAPED like a candidate, the solver reported known=True. The
    #     contamination the old criterion existed to catch is real, reachable, and was NOT what the
    #     old criterion was catching.
    #
    # So the discriminating variable is OUR OWN PRESENCE IN THE WORKSPACE, and the three cases below
    # drive it in both directions against a foreign content that is present in all three.
    #
    # PER-CONJUNCT REACHABLE RED, mutated on a copy of this file and RUN rather than argued (P8
    # burdens 1 and 3 -- which conjunct catches what, by mutation, not by the total going red).
    #
    # RE-DERIVED 2026-08-06 by running THIS `main()` on each mutant AFTER `thalamus/
    # global_workspace.py::_rank_key` stopped falling through to the source name. FOUR rows of the
    # previous table (M4, M7, M8, M10) do not reproduce, and an adversarial re-run corrected the
    # ATTRIBUTION: the arbiter explains NONE of the four. Re-measured against the genuinely retired
    # code -- the committed `global_workspace.py` at 4c71742a loaded and bound to these same mutants
    # -- all four already read differently THERE. They were transcription errors in the old table,
    # not consequences of the repair, and "the previous table was measured on the name-decided
    # arbiter" is not a defence any of them needs. The `(ii)` column is dropped because the case
    # itself was deleted (see its block); under the SHIPPED arbiter it read RED on every row
    # INCLUDING the unmutated file.
    #
    #     mutation of _solve's workspace read        (i)coex (iii)excl (iv)impostor
    #     M1 shape-only match, no token               ok      ok        RED
    #     M2 head-identity read, token kept           RED     ok        ok
    #     M3 the retired form: head + shape           RED     ok        RED
    #     M4 untagged contents                        ok      ok        RED
    #     M5 claim only if the bus is EXCLUSIVELY     RED     ok        ok
    #        ours -- the LAZY FIX, single-slot
    #        exclusion re-imposed under the new
    #        arbiter
    #     M6 step() called, its output DISCARDED,     ok      RED       RED
    #        `known` from the rule match alone
    #     M7 the RETIRED c2 DEFECT, in the only       ok      RED       RED
    #        shape that reaches a live path
    #     M8 shape guard dropped from the read,       ok      ok        ok      <- EQUIVALENT
    #        token kept
    #     M10 token present but CONSTANT              ok      ok        RED
    #     M11 `mine[-1]` instead of `mine[0]`         ok      ok        ok      <- EQUIVALENT
    #
    # Every surviving conjunct has a reachable red, and four readings are owed honestly rather than
    # left for the next reader to find.
    #
    # (a) M5 is the mutation this whole pass exists to refuse -- it keeps the token AND the access
    #     read and still throws away what the multi-slot arbiter bought. ONLY (i) sees it now, where
    #     (i) and (ii) saw it before, so (i) went from redundant to uniquely load-bearing for it.
    # (b) M7 is why (iii)'s tenants are BARE INTS and is the correction this criterion needed most.
    #     As first written this case filled the capacity with `("percept", j)` TUPLES, and against
    #     that shape M7 -- which returns the live percept as the RPM answer whenever nothing of
    #     ours is admitted, i.e. verbatim the thing the RETIRED assert message forbade -- PASSED
    #     ALL FOUR CASES. With the live shape it is RED on (iii).
    # (c) TWO cases are now uniquely load-bearing rather than one, and that is a consequence of the
    #     arbiter repair rather than of anything done here: deleting (i) loses M2 and M5, deleting
    #     (iv) loses M1, M4 and M10, and deleting (iii) loses nothing that (iv) does not also catch.
    #     (iii) is KEPT anyway, for per-conjunct ATTRIBUTION (P8 burden 3 -- which half of the read
    #     failed, not merely that one did) and because it is the only case that drives the exclusion
    #     branch at all; it is not credited as adding a catch.
    # (d) M8 AND M11 ARE EQUIVALENT MUTANTS, not gaps. `M8` drops `len(c) == 3` while keeping
    #     `c[:2] == (_ANSWER, tag)`, so it differs only on a 2-tuple wearing this solve's own token,
    #     which nothing posts; `M11` reads `mine[-1]`, and `|mine|` is 1 on every reachable input
    #     (private bus and shared bus alike). Both are the same program as the shipped read, and the
    #     equivalence is ARBITER-INDEPENDENT: run against the committed retired code they are green
    #     on all four cases there too, case (ii) included. An earlier version of this note said each
    #     "was caught by case (ii) under the retired arbiter" and that NO mutant's sole red was
    #     anything else; both are false. Under the retired arbiter (ii)'s reds were M1, M2, M3, M4,
    #     M5 and M10, every one of which ALSO reddened another case -- so no mutant ever had (ii) as
    #     its sole red, and M8/M11 had no red at all. The conclusion below is unaffected and is the
    #     one that was actually checked; the reasoning offered for it was not.
    shared = GlobalWorkspace(seed=0)
    ctx_s, cand_s, idx_s, _a = _build_task("progression", random.Random(7))
    shared_solver = RpmSolver(seed=0, workspace=shared)
    adopt_ok = shared_solver.workspace is shared
    clean = shared_solver.solve(ctx_s, cand_s)  # empty bus: must match the private answer exactly
    adopt_ok = adopt_ok and shared_solver.workspace is shared
    private_same = RpmSolver(seed=0).solve(ctx_s, cand_s)
    ours = (_ANSWER, _solve_tag(ctx_s, cand_s), clean["answer_idx"])

    # (i) CO-EXISTENCE. A percept-shaped int is warmed into the bus and is still posting in the
    #     SAME tick as the solve, so it holds the HEAD and our candidate wins access BENEATH it.
    #     Our answer must be unchanged and it must still be claimed as KNOWN: we won access, and a
    #     louder faculty being present is not our gap. THIS IS THE CASE THE RETIRED PROXY GOT
    #     BACKWARDS -- it read the foreign content's mere presence as proof that nothing of ours
    #     had ignited.
    #
    #     THE BUS IS FRESH, and that is load-bearing rather than tidiness. This case was first
    #     written to warm the foreign into `shared`, the bus `clean` had already solved on -- and
    #     measured on that bus OUR content was STILL RESIDENT from that earlier solve after all
    #     three warm ticks (`broadcast_all()` = `(2, ('rpm_answer', <tag>, 3))`). The token is
    #     per-QUESTION, so a re-solve of the same matrix posts the identical content: `ours in
    #     broadcast_all()` was therefore satisfiable by the PREVIOUS solve's residue and proved
    #     nothing about this one, which is a weaker reading of own-provenance than the case claims
    #     to make. On a bus this matrix has never been posted to it proves exactly what it says.
    foreign = idx_s ^ 1  # a live-percept-shaped int that is NOT this matrix's answer
    ws_co = GlobalWorkspace(seed=0)
    co_solver = RpmSolver(seed=0, workspace=ws_co)
    for _ in range(3):
        ws_co.submit("perception", foreign, 1.0)
        ws_co.step()  # a foreign content takes a slot and warms its attractor
    ws_co.submit("perception", foreign, 1.0)  # ... and is still posting when we solve
    coexist = co_solver.solve(ctx_s, cand_s)
    coexist_ok = (
        ws_co.broadcast() == foreign  # the foreign content really does hold the HEAD
        and ours in ws_co.broadcast_all()  # ... ours won access beneath it, on THIS solve
        and coexist == private_same  # ... and it cost us nothing, `known` included
    )

    # (ii) DELETED 2026-08-06 UNDER F11 -- "a criterion with no reachable red is deleted, not
    #      weakened." It read: two ordinary consecutive solves on one bus, the PREVIOUS matrix's
    #      winner still charged and wearing the identical content shape, RANKING ABOVE this
    #      matrix's candidate, and it asserted `ws_seq.broadcast() == stale` as its own non-vacuity
    #      check. That precondition was manufactured by `thalamus/global_workspace.py`'s rank key
    #      falling through to the SOURCE STRING: measured on the exact configuration this case
    #      built, the stale pool held the head on `_g` 0.850000 / final membrane 0.613777 against
    #      this solve's own candidate at 1.000000 / 0.916325 -- IDENTICAL MEMBRANES UNDER BOTH
    #      ARBITERS, only the order differing, because 'c0' sorts before 'c3'.
    #
    #      Three measurements, and each was run:
    #
    #      1. THE PRECONDITION IS UNREACHABLE, not merely unmet. Swept all 2,304 ordinary
    #         two-solve configurations (4 regimes x 12 seeds, squared, keeping the pairs whose
    #         answers differ): the stale answer holds the head on ZERO of them under a name-blind
    #         arbiter. The fresher candidate out-drives its own decayed predecessor by construction
    #         -- `_defended` withholds the tenancy subsidy from the superseded content and it
    #         decays at `decay` -- so there is no pair of ordinary solves that reaches the state
    #         this case is about.
    #      2. UNDER THE SHIPPED ARBITER IT HAS NO GREEN STATE: re-running this `main()` against ten
    #         mutants of `_solve`'s workspace read, case (ii) is RED on the correct code AND on all
    #         ten. A column that is red everywhere discriminates nothing. STATED PRECISELY, because
    #         an earlier version of this line read "AS IT STOOD IT HAD NO GREEN STATE AT ALL" and
    #         that is not what was measured: run against the genuinely retired code, case (ii) was
    #         GREEN on the correct file and RED on six of the ten (M1, M2, M3, M4, M5, M10). It was
    #         a working criterion on the arbiter it was written for. What the repair removed is its
    #         PRECONDITION, and a criterion whose precondition became unreachable is a criterion
    #         with no green state FROM NOW ON -- which is the F11 case, and it is a smaller and
    #         truer claim than "it never worked".
    #      3. DELETING IT LOSES NO MUTANT, and this is the conjunct that had to be checked rather
    #         than argued, because (2) alone would permit losing every catch (ii) used to have.
    #         Run: all six mutants (ii) reddened under the retired arbiter are reddened under the
    #         shipped one by a SURVIVING case -- M1 by (iv), M2 by (i), M3 by (i) and (iv), M4 by
    #         (iv), M5 by (i), M10 by (iv). The re-run table is below; M8 and M11 are EQUIVALENT
    #         PROGRAMS on every reachable input and were never caught by anything, (ii) included.
    #
    #      The PROPERTY it named is not gone and is not orphaned: "a foreign content must not
    #      become this solve's answer" is asserted by (i) for a co-admitted foreign content and by
    #      (iv) for a content wearing this solver's OWN namespace with a DIFFERENT question token,
    #      which is exactly a previous matrix's answer. What is gone is a precondition the arbiter
    #      can no longer produce.
    ctx_o, cand_o, idx_o, _a = _build_task("distribute3", random.Random(5))

    # (iii) EXCLUSION -- the honest gap, now forced by the mechanism that actually forces it. The
    #       capacity is FULL of foreign contents, so no candidate of ours is admitted. With nothing
    #       of ours in the workspace the answer falls back to the rule-match argmax, must NOT be
    #       claimed as known, and must not be one of the foreign contents.
    #
    #       THE TENANTS ARE BARE INTS, which is the ONE thing the retired criterion tested that a
    #       tuple-shaped tenant does not. `broadcast_all` documents perception as the live producer
    #       on this bus and it posts SMALL INTS -- the retired assert message says so in as many
    #       words ("the brain's percept contents are small ints and so are candidate indices") --
    #       and its `occupied["answer_idx"] != held` conjunct existed precisely to forbid returning
    #       one. Measured with `("percept", j)` TUPLES here instead, a mutation that reads a bare
    #       int head as the answer passed all four cases while returning the PERCEPT as the RPM
    #       answer under exclusion (answer #0 against a true #3); with bare ints it is red on this
    #       case and on (i). The warm count is 5 and is NOT arbitrating -- 1 through 8 give the
    #       identical outcome on every conjunct of (iii) and (iv), measured, so it is a settling
    #       margin rather than a tuned number.
    ws_full = GlobalWorkspace(seed=0)
    full_solver = RpmSolver(seed=0, workspace=ws_full)
    for _ in range(5):
        for j in range(ws_full.capacity):
            ws_full.submit("percept%d" % j, j, 1.0)
        ws_full.step()
    excluded = full_solver.solve(ctx_s, cand_s)
    exclude_ok = (
        not any(  # nothing of OURS got in
            isinstance(c, tuple) and c[:1] == (_ANSWER,) for c in ws_full.broadcast_all()
        )
        and excluded["answer_idx"] == clean["answer_idx"]
        and not excluded["known"]
    )

    # (iv) THE DECISIVE CASE: IMPERSONATION *UNDER* EXCLUSION. One of the foreign tenants is shaped
    #      exactly like one of our candidates AND names our own answer index, and the capacity is
    #      full so nothing of ours is admitted. Every observable a shape-matching reader has says
    #      "my answer ignited"; nothing of ours did. `known` must be False here, and this is the
    #      condition (i) and the retired single-slot proxy both call the same thing and cannot tell
    #      apart. Measured red: with the shape test in place of the provenance test, this reads
    #      known=True.
    ws_imp = GlobalWorkspace(seed=0)
    imp_solver = RpmSolver(seed=0, workspace=ws_imp)
    impostor = (_ANSWER, _solve_tag(ctx_o, cand_o), clean["answer_idx"])
    tenants = [impostor] + list(range(ws_imp.capacity - 1))  # look-alike + live-shaped percepts
    for _ in range(5):
        for j, c in enumerate(tenants):
            ws_imp.submit("foreign%d" % j, c, 1.0)
        ws_imp.step()
    impersonated = imp_solver.solve(ctx_s, cand_s)
    impostor_ok = (
        impostor in ws_imp.broadcast_all()  # the look-alike really is holding a slot
        and ours not in ws_imp.broadcast_all()  # and nothing of ours is
        and impersonated["answer_idx"] == clean["answer_idx"]  # no tenant became the answer
        and not impersonated["known"]  # and nothing may be claimed as known
    )

    collide_ok = coexist_ok and exclude_ok and impostor_ok
    print(
        "  [%s] shared bus    -> adopted=%s; empty bus reproduces the private answer #%d; a "
        "foreign percept %r co-resident leaves known=%s answer #%d; capacity full of foreign -> "
        "known=%s; a look-alike wearing our namespace and naming #%d with nothing of ours admitted "
        "-> known=%s"
        % (
            "PASS" if (adopt_ok and collide_ok) else "FAIL",
            adopt_ok,
            clean["answer_idx"],
            foreign,
            coexist["known"],
            coexist["answer_idx"],
            excluded["known"],
            impostor[2],
            impersonated["known"],
        )
    )

    n_pass = sum(passed)
    print("-" * 78)
    print(
        "solved %d/4 rule regimes correctly (chance on 8 candidates = 12.5%%); role-filler binding, "
        "honest gap, and determinism all hold." % n_pass
    )
    print(
        "ceiling: bounded to the NVSA rule set (distribute-three / progression / constant / "
        "arithmetic) over clean attribute codes; NOT full ARC, NOT open-domain abstraction."
    )

    assert n_pass == 4, "all four NVSA rule regimes must solve correctly (got %d/4)" % n_pass
    assert bind_ok, "role-filler object binding must recover a filler by spike cleanup"
    assert gap_ok, "a grid with no NVSA rule must return an HONEST gap (known=False), never a bluff"
    assert det_ok, "the solver must be deterministic under a fixed seed"
    assert adopt_ok, "a handed-in workspace must be THE object every solve competes in (identity)"
    assert clean == private_same, (
        "an adopted but EMPTY workspace must reproduce the private-bus result exactly -- if it "
        "does not, adoption changed the answer rather than only changing who else can see it"
    )
    assert collide_ok, (
        "this solver's answer and its `known` flag must be produced by ITS OWN candidate winning "
        "access on THIS solve, never by a foreign content sharing the bus. Three cases, and the "
        "message names the condition each one actually drives, because the retired single-slot "
        "form named conditions that could not fail: (i) co-existence -- a percept-shaped int holds "
        "the HEAD on a bus we have never posted to and our candidate wins a slot beneath it, which "
        "must not turn our won access into a gap; (iii) exclusion -- with the capacity full of "
        "live-shaped percepts and nothing of ours admitted, the answer must be the private one and "
        "`known` must be False; (iv) impersonation UNDER exclusion -- a foreign content wearing "
        "this solver's namespace with a DIFFERENT question token, naming our own index, must not "
        "produce `known` nor the answer while nothing of ours is in the workspace. A fourth case "
        "(ii) was DELETED under F11 -- see its block for the 2,304-configuration sweep showing its "
        "precondition unreachable and the mutation table showing the deletion loses no mutant. "
        "coexist=%s exclude=%s impostor=%s" % (coexist_ok, exclude_ok, impostor_ok)
    )

    print("=" * 78)
    print(
        "self-check OK: spiking NVSA RPM solver -- role-filler object codes, example->rule induction "
        "(distribute-three / progression / constant / arithmetic) executed as spike-XOR/permute/"
        "cleanup ops, basal-ganglia rule gate + global-workspace answer ignition; 4/4 regimes solved "
        "(>> 12.5%% chance), honest gap on a no-rule grid, deterministic; a handed-in workspace is "
        "adopted by identity and reproduces the private answer, and on the SHARED multi-slot bus "
        "this solver's answer and its `known` flag come from its own candidate winning access -- "
        "co-existing under a louder foreign head still counts, and a foreign content is refused "
        "whether it is percept-shaped, is a previous matrix's answer, or is shaped like one of our "
        "own candidates while nothing of ours is admitted."
    )
    return 0


if __name__ == "__main__":
    main()
