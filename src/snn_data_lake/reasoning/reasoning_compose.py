"""Compose the single-shot mixture-of-experts router into a MULTI-STEP solver.

``reasoning/reasoning_cortex.py::build_default_cortex(...)`` returns a
``ReasoningCortex`` whose ``.route(text) -> dict`` dispatches the SINGLE best
expert for a query (Gf number/letter series, Raven, HDC binding, SDM recall,
dense-associative recall, SOM clustering, triplet-STDP plasticity, and -- when
the caller supplies stored triples -- learned KB-completion). Each route is
one confident expert or an HONEST GAP; nothing is bluffed.

``CompositionalSolver`` turns that one-shot grammar into a bounded CHAIN. It
decomposes a query into <= 3 sub-steps -- splitting on explicit connectives
(";", "then", "and then") and/or threading a RESULT-SUBSTITUTION scratchpad
(the answer of step 1 is bound into step 2's text via a marker such as "that" /
"it" / "result", etc.) -- dispatches EACH sub-step through the existing
``cortex.route`` UNCHANGED (still one best expert per step), and returns the
final answer with its shown work.

Honest ceiling: this is COMPOSITION of the bounded expert grammar
(series / analogy / binding / recall / KB-completion), NOT open-domain
novel-operator synthesis. If no confident chain forms -- fewer than two steps,
or any step is an honest gap -- ``solve`` returns ``None`` so the caller's honest
gap stands. It never fabricates an answer and it never invents an operator the
experts do not already have.

DEPENDENCIES (P7, 2026-08-04). This paragraph used to read "Pure-stdlib at the module
level (``random`` + ``re``) ... No torch" -- a purity boast left behind by the
zero-dependency rule abolished project-wide on 2026-07-20, and exactly the residue P7
names as a defect. The import set is unchanged and it is the RIGHT one, but for a
measured reason rather than a purity one: this module does string decomposition and
dispatch only -- ``_SPLIT.split``, one regex substitution per step, and a dict handoff
to ``cortex.route`` -- so there is no array or tensor arithmetic here for numpy or torch
to accelerate. The numeric stack is pulled by ``build_default_cortex`` when the caller
constructs the cortex, which is where the arithmetic actually lives. No brainstem
import, no actuator route (that is a safety-envelope statement, not a dependency one).
"""

import re  # `import random` went with the dead `self._rng` it was the only user of

# tokens that, inside a later sub-step, refer back to the previous step's answer (the scratchpad
# binding). Ordered most-specific first so "<prev>" / "result" win before the bare pronouns.
_MARKERS = (r"<prev>", r"\bresult\b", r"\bthat\b", r"\bit\b", r"(?<!\w)_(?!\w)")

# the explicit connectives a query is decomposed on (";", "then", "and then").
_SPLIT = re.compile(r"\s*;\s*|\s+and\s+then\s+|\s+then\s+", re.IGNORECASE)


def _substitute(text, answer):
    """Bind `answer` into `text` at the FIRST back-reference marker, or return `text` unchanged."""
    for mk in _MARKERS:
        new = re.sub(mk, str(answer), text, count=1, flags=re.IGNORECASE)
        if new != text:
            return new
    return text


def _extract_answer(reply):
    """Pull the salient answer token out of an expert reply, or ``None`` if none is legible.

    Handles the expert reply grammars that carry a concrete result: number/letter series
    ("... -> next = 10 (...)"), analogy / series continuation ("... -> ijl (...)"). An honest-gap
    reply carries no such token and yields ``None`` (the step is then treated as a miss).
    """
    if not reply:
        return None
    m = re.search(r"next\s*=\s*([^\s()]+)", reply)
    if m:
        return m.group(1).strip(".,")
    m = re.search(r"->\s*([^\s()]+)", reply)
    if m and m.group(1).lower() != "next":
        return m.group(1).strip(".,")
    return None


class CompositionalSolver:
    """Multi-step reasoning by COMPOSING the single-shot cortex router (each step still one best
    expert). Deterministic: no wall-clock, no RNG AT ALL -- this class has no stochastic step
    (the docstring claimed ``random.Random(seed)`` per instance, describing an attribute that was
    stored and never read; both are gone as of 2026-08-04, see __init__);
    the chain order and the cortex routing are the only state. Honest gap (``None``) when no
    confident chain forms -- this composes the bounded expert grammar, it does not synthesise new
    operators.
    """

    def __init__(self, cortex, records=None, seed=0):
        self.cortex = cortex  # from build_default_cortex(...)
        # `records` is stored and NEVER READ (re-verified tree-wide 2026-08-04). It is kept rather
        # than deleted because it has a LIVE caller that believes otherwise:
        # `LakeChat._compose_reply` in reasoning/spike_chat_lake.py passes `records=self.records`,
        # i.e. the chat hands this
        # solver its knowledge base expecting the chain to consult it. It does not -- routing goes
        # through `self.cortex` alone. Deleting the parameter would break that call site and hide
        # the mismatch; the honest state is that the caller's expectation is unmet. Either wire
        # `records` into _route()'s miss path or drop the argument at the call site -- but do not
        # leave a third reader believing a KB is in play.
        #
        # STALE LINE CITATION, corrected 2026-08-04: this comment cited
        # "reasoning/spike_chat_lake.py:1087". That line is a COMMENT about `route_all` coverage
        # thresholds and has nothing to do with this class; the real construction site is :1217.
        # Naming the SYMBOL instead of the line, because a line number in a 1200-line file that is
        # edited far more often than this one is a citation that rots -- which is exactly what
        # happened here, and `tools/citation_guard.py` does not catch it (it resolves FILES, never
        # line numbers).
        #
        # SYMBOL CORRECTED 2026-08-05: the replacement symbol was `LakeChat._compose`, and there is
        # NO SUCH METHOD -- `grep "def _compose\b"` over spike_chat_lake.py returns zero matches.
        # The method is `LakeChat._compose_reply` (`def` at :1203; the construction is at :1217,
        # which does resolve). Swapping a rotting line number for a symbol only helps if the symbol
        # is checked against the tree, and this one was not: the fix for a citation defect shipped
        # the same defect in the more durable form, which is the worse outcome of the two.
        self.records = list(records or [])  # STORED, NOT READ -- see above
        # `self._rng = random.Random(seed)` stood here marked "STORED, NOT READ". Unlike `records`
        # it had no caller expectation to preserve: the ONLY two construction sites tree-wide are
        # this module's own self-check and `LakeChat._compose_reply`, and NEITHER passes `seed`. There is
        # no stochastic step anywhere in this class, so the object was dead in every sense. Deleted
        # 2026-08-04 (P3: retire, do not keep a dead attribute that suggests a randomised path).
        # `seed` stays in the signature -- it costs nothing, both callers omit it, and removing a
        # public parameter is a breaking change for a property this class does not need.

    def _route(self, text):
        """Dispatch ONE sub-step through the existing cortex router, guarded so a broken expert or a
        cortex without ``.route`` never crashes the caller (returns an honest miss instead)."""
        route = getattr(self.cortex, "route", None)
        if not callable(route):
            return None
        try:
            out = route(text)
        except Exception:  # a broken expert never crashes the turn
            return None
        return out if isinstance(out, dict) else None

    def solve(self, text):
        """Decompose `text` into <= 3 sub-steps, dispatch each through the cortex, thread the results,
        and return the composed answer with its shown work -- or ``None`` for an honest gap.

        Returns ``{"known": True, "source": "compose", "reply": "<final answer> (steps: ...)"}`` on a
        confident chain, else ``None`` (fewer than two steps to compose, or a step was an honest gap).
        """
        if not text or not str(text).strip():
            return None
        parts = _SPLIT.split(str(text).strip())
        steps = [p.strip() for p in parts if p.strip()][:3]
        if len(steps) < 2:
            return None  # nothing to compose; single-shot is the router's job

        scratch = []  # [(resolved_step_text, answer), ...]
        prev = None
        for step in steps:
            stext = _substitute(step, prev) if prev is not None else step
            out = self._route(stext)
            if out is None or not out.get("known"):
                return None  # a step is an honest gap -> the whole chain is
            ans = _extract_answer(out.get("reply", ""))
            if ans is None:
                return None  # no legible result to thread -> honest gap
            scratch.append((stext, ans))
            prev = ans

        final = scratch[-1][1]
        shown = "; ".join(f"[{i + 1}] {s} -> {a}" for i, (s, a) in enumerate(scratch))
        return {
            "known": True,
            "coverage": 1.0,
            "ignited": True,
            "source": "compose",
            "reply": f"{final} (steps: {shown})",
        }


# ------------------------------------------------------------- headless self-check -------------------


def _selfcheck():
    from snn_data_lake.reasoning.reasoning_cortex import build_default_cortex

    # stdlib-only cortex path: triples=None so the numpy-backed KB-completion expert is NOT registered.
    cortex = build_default_cortex(seed=0)
    solver = CompositionalSolver(cortex)

    print("=" * 88)
    print("REASONING COMPOSE -- multi-step composition over the single-shot expert router")
    print("=" * 88)

    # a genuine 2-step chain: series step 1's result (10) is THREADED into series step 2 ("that 20 30").
    q = "what comes next in 2 4 6 8 ; what comes next in that 20 30"
    got = solver.solve(q)
    assert got is not None, "expected a compose dict, got None"
    assert got["known"] and got["source"] == "compose", got
    assert "[1]" in got["reply"] and "[2]" in got["reply"], (
        f"reply must show 2 steps: {got['reply']!r}"
    )
    assert got["reply"].startswith("40"), f"final answer should thread to 40: {got['reply']!r}"
    print(f"(1) compose : {got['reply']}")

    # out-of-grammar / no chain -> honest gap (None), so the caller's honest gap stands.
    none1 = solver.solve("tell me about your weekend plans")  # single step, no connective
    assert none1 is None, f"single out-of-grammar query must be an honest gap (None), got {none1!r}"
    none2 = solver.solve(
        "what comes next in 2 4 6 8 ; tell me about your weekend"
    )  # step 2 is a gap
    assert none2 is None, f"a chain with a gap step must return None, got {none2!r}"
    print(
        "(2) gap     : out-of-grammar refused (None) -- composition, not novel-operator synthesis"
    )

    # determinism: identical inputs -> identical outputs.
    assert solver.solve(q) == solver.solve(q)
    print("(3) determinism: stable across repeated calls")

    print(
        "self-check OK: compositional multi-step over the bounded expert grammar (series/analogy/"
        "binding/recall/KB), honest gap otherwise"
    )


if __name__ == "__main__":
    _selfcheck()
