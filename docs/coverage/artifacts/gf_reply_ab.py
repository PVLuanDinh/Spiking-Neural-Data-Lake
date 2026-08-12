"""gf_reply_ab -- neuter-A/B for the Gf (fluid-intelligence) faculties, on the LIVE chat route.

WHY A SECOND PROBE EXISTS. tools/charge_probe.py neuters a named entry in Brain._drive_w and watches
the arm's act/hold decision and trajectory. Every Gf faculty in this tree is unreachable that way:
none is a drive source, none touches the motor route, and a 60-tick autonomous run never poses a
puzzle. charge_probe answers "unknown drive source 'letter_analogy'" and exits 2. So the category
could not be scored above band 40 for a reason that says nothing about the brain -- the only
instrument in the tree could not address it. This probe addresses it, one layer out, same method.

WHERE IT LIVES, AND WHY THAT IS WRONG. It belongs in tools/. It is here because the audit that wrote
it had write access to docs/coverage/ only (two other agents were editing tools/** concurrently). It
is at least TRACKED, which is the property `coverage_score.py --rerun` needs: tools/arc_score.py's
own docstring records why an untracked probe is not a reproduction path. Promoting it to tools/
changes the recorded command in every artifact below -- re-record them in the same commit. Note also
that docs/ is NOT in ruff's extend-exclude (pyproject.toml:138) and gets no per-file-ignores, so
anything dropped here is linted under the full E,F,I,UP,B ruleset with no E501 relief. This file is
clean under `ruff check` and `ruff format --check`; several sibling probes from this sweep are not.

METHOD, mirroring charge_probe. Two arms at seed 0 on the canonical construction path
(cortex.brain.make_brain, Policy 4). A = wired. B = the named faculty's OWN entry points replaced,
in the module the live route imports them from, by a stub returning that faculty's honest-gap value.
The observable is BEHAVIOURAL for a conversational faculty and nothing weaker: the reply string that
Brain.respond -- the method the daemon pumps cockpit chat into (learn_daemon.py:245) -- returns.
Telemetry is never consulted.

WHAT A PASS BUYS, AND WHAT IT DOES NOT. Exit 0 means the faculty's output reaches the brain's reply:
rubric band 41-60 ("changes an internal read ... a reply"), and that is this probe's CEILING. It is
not band 61. No decision, action or weight is downstream of any Gf faculty here, because the route
is read-only by construction (live_brain.py:443: "this path is read-only reasoning, no motor
route"). A band-61 Gf claim needs a wire this brain does not have.

THE NEUTER IS AT THE IMPORT SOURCE, not the call site, deliberately. The chat route reaches these
functions two ways: spike_chat_lake._gf_request imports them per call (so a module-level patch is
seen every turn), and reasoning_cortex's experts capture them in __init__ (so the patch must land
before the cortex is built -- it does, because the cortex is lazy and arm B patches before its first
turn). Patching one call site would leave the other live and bias the verdict toward NOT-DECISIVE,
the direction that makes a live faculty look dead. Same failure mode charge_probe's _PinnedWeights
exists to fix.

ELIGIBILITY, BEFORE ANY VERDICT. Arm A's reply must actually carry the faculty's marker. It often
does not: converse() consults the reasoning cortex at spike_chat_lake.py:1047, AFTER the
recall/ignition competition, so any lexical or semantic recall hit preempts the expert. Measured
here on the ARC and structure-mapping items. A preempted faculty reports NOT-ELIGIBLE (exit 2), not
a charge verdict: "its output does not reach the reply" and "it was never asked" are different
findings, and reporting the second as the first is how a roster talks itself into a number.

  python docs/coverage/artifacts/gf_reply_ab.py                 # list the faculties
  python docs/coverage/artifacts/gf_reply_ab.py letter_analogy
"""

import os
import re
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# SAFETY, set before anything can construct a persisting brain: every arm of every faculty runs
# against an ISOLATED durable directory. substrate/home.py:60 reads NORD_HOME on each call, so this
# guarantees the probe can never write into the operator's real ~/.snn_data_lake -- which matters
# because the sleep_schema_induction faculty below deliberately runs with persist=True.
_SANDBOX = tempfile.mkdtemp(prefix="gf_reply_ab_")
os.environ["NORD_HOME"] = _SANDBOX

from snn_data_lake.cortex.brain import make_brain  # noqa: E402


def _none(*_a, **_k):
    return None


def _no_grid(*_a, **_k):
    return [], []


def _unknown(*_a, **_k):
    return {"known": False}


def _patch(modname, names, stub):
    mod = __import__(modname, fromlist=["_"])
    for n in names:
        setattr(mod, n, stub)


def _patch_method(modname, cls, meth, stub):
    mod = __import__(modname, fromlist=["_"])
    setattr(getattr(mod, cls), meth, stub)


_GF = "snn_data_lake.reasoning.gf"
_ARC = "snn_data_lake.reasoning.arc_solver"
_SMAP = "snn_data_lake.cortex.vsa_structure_mapping"  # retargeted at the 2026-08-12 promotion (string-name dispatch: no import-graph tool sees this site)
_RPM = "snn_data_lake.reasoning.rpm_solver"
_VSYN = "snn_data_lake.reasoning.vsa_synthesize"
_CHAT = "snn_data_lake.reasoning.spike_chat_lake"


def _neuter_number_series():
    _patch(
        _GF,
        (
            "solve_series",
            "_poly_next",
            "_geometric_next",
            "_recurrence_next",
            "_periodic_additive_next",
            "_interleave_next",
            "consistent_hypotheses",
            "consistent_programs",
        ),
        _none,
    )
    _patch_method(_GF, "OpenGf", "solve", _none)
    _patch_method(_GF, "OpenGf", "solve_unbounded", _none)


def _neuter_rpm():
    _patch_method(_RPM, "RpmSolver", "solve", _unknown)
    _patch_method(_VSYN, "VsaSynthesizer", "solve", _unknown)


def _neuter_experiment_resume():
    """Kill the RESUME half of the active-learning loop only.

    The ask half cannot be neutered without also neutering number_series (the ask is raised inside
    _number_reply when the hypotheses tie), and one artifact must not carry two faculties' evidence.
    So this scopes to _resume_experiment, whose single call site is converse():920. What the
    artifact then shows is the ask surviving unchanged in BOTH arms -- correct, it is upstream --
    and the resume turn diverging, which is the claim being tested.
    """
    _patch_method(_CHAT, "LakeChat", "_resume_experiment", _none)


# faculty -> (queries, marker the WIRED reply must carry, neuter callable)
#
# The markers match the ANSWER form, never the honest-gap form. Both carry the faculty's name --
# "[induced] ... letter analogy" and "[honest gap] i recognise a letter analogy" differ only in the
# prefix -- so a loose marker would call a run eligible on the strength of the brain REFUSING to
# answer it, which is the opposite of what eligibility is for.
FACULTIES = {
    "number_series": (
        ["2 4 6 8 what is next", "1 1 2 3 5 what is next", "3 6 12 24 what is next"],
        r"\[induced\][^|]*at test time|\[gf\] induced a rule",
        _neuter_number_series,
    ),
    "letter_analogy": (
        ["abc:abd :: ijk:?", "rst:rsu :: abc:?"],
        r"\[induced\][^\n]*letter analogy|\[gf\] letter analogy",
        lambda: _patch(_GF, ("letter_analogy",), _none),
    ),
    "novel_alphabet_analogy": (
        ["alphabet q w e r t then q:w :: e:?", "alphabet z y x w then z:y :: x:?"],
        r"\[induced\][^\n]*novel-alphabet analogy",
        lambda: _patch(_GF, ("analogy_over",), _none),
    ),
    "raven_dist3": (
        ["a b c; b c a; c a ?", "x y z; y z x; z x ?"],
        r"\[induced\][^\n]*Raven grid|\[gf\] Raven ",
        lambda: _patch(_GF, ("raven_dist3", "raven_arith_grid", "raven_cell_attrs"), _none),
    ),
    "pair_induction": (
        [
            "if 3 -> 10, 5 -> 26, 7 -> 50, then 9 -> ?",
            "if 2 -> 4, 3 -> 9, 4 -> 16, then 6 -> ?",
            "if 11 -> 23, 12 -> 25, 13 -> 27, then 14 -> ?",
        ],
        r"\[gf\] induced .* from the given pairs",
        lambda: _patch(_GF, ("function_from_pairs", "string_from_pairs"), _none),
    ),
    "arc_grid_synthesis": (
        [
            "arc: 11|22 -> 22|11 ; test 34|56",
            "arc: 12|34 -> 21|43 ; 56|78 -> 65|87 ; test 13|24",
            "arc: 00|11 -> 11|00 ; 22|33 -> 33|22 ; test 44|55",
        ],
        r"\[arc\] induced a grid program",
        lambda: _patch(_ARC, ("synthesize",), _no_grid),
    ),
    "relational_structure_mapping": (
        [
            "zormp(glib,traxel) blenk(traxel,glib) :: zormp(vurn,quilth) blenk(quilth,vurn)",
            "carries(truck,crate) heavier(truck,crate) :: carries(ant,crumb) heavier(ant,crumb)",
            "orbits(planet,sun) attracts(sun,planet) :: "
            "orbits(electron,nucleus) attracts(nucleus,electron)",
        ],
        r"\[structure-map\]",
        lambda: _patch_method(_SMAP, "StructureMapper", "map", _none),
    ),
    # A TWO-TURN sequence, and the only faculty here whose items are not independent: turn 0 must be
    # answered with an experiment REQUEST (the MDL guard refuses to commit on 5 primes), turn 1 is
    # the bare discriminating term the resume folds back in. The marker therefore hits at most 1/2
    # by construction -- that is the design, not a preemption.
    "active_experiment_loop": (
        ["2 3 5 7 11 what is next", "13"],
        r"\[experiment\]",
        _neuter_experiment_resume,
    ),
    "rpm_nvsa": (
        [
            "raven: type=1,size=1 ; type=2,size=1 ; type=3,size=1 ; type=1,size=2 ; "
            "type=2,size=2 ; type=3,size=2 ; type=1,size=3 ; type=2,size=3 | "
            "type=3,size=3 ; type=1,size=1 ; type=2,size=2"
        ],
        r"\[rpm\] induced|\[rpm/synth\]",
        _neuter_rpm,
    ),
}


def run(queries):
    """One arm: a fresh canonical brain, the queries put through the daemon's own reply method.

    Only the REPLY TEXT is kept. respond() returns a dict carrying route labels and live state
    beside the reply, and comparing whole dicts would count an unrelated state field ticking over
    as a Gf flip -- an inflation the verdict would then report as charge.
    """
    b = make_brain(seed=0, persist=False)
    out = []
    for q in queries:
        try:
            r = b.respond(q)
        except Exception as exc:
            r = f"RAISED {type(exc).__name__}"
        out.append(str(r.get("reply", r)) if isinstance(r, dict) else str(r))
    b.close()
    return out


# ------------------------------------------------------------- the one AUTONOMOUS-TICK faculty ----
# Every faculty above is human-triggered: it fires only because somebody typed a puzzle. This one is
# not, and it is the only Gf-adjacent rule induction on the brain's own clock -- so its observable
# is a DURABLE MEMORY WRITE, not a reply. It cost four failed harnesses to measure, and each failure
# was the wrong conclusion waiting to be written down:
#   1. persist=False (the reply harness's setting) -> the branch's first gate is `self._persist`.
#   2. forcing clock.phase before each tick -> tick() calls clock.step(), which overwrites it.
#   3. a brain BORN asleep (brain.py:478 seeds _was_asleep from the gate) never sees a wake->sleep
#      EDGE, and the branch lives in _enter_sleep, the edge handler -- not in _sleep_tick.
#   4. 80 ticks at day_period=20 reaches night 3. _nights increments in _enter_wake, so the first
#      _enter_sleep with `_nights % 4 == 0` is night 4, which needs nine transitions, not seven.
# At attempt 4 the honest-looking readout was "SchemaInducer.consolidate: 0 calls" -- which reads
# exactly like the dormancy this audit was sent to find, and would have been wrong.
_SLEEP_FACTS = (
    "remember: a dog is a mammal",
    "remember: a cat is a mammal",
    "remember: a mammal is an animal",
    "remember: a sparrow is a bird",
    "remember: a bird is an animal",
    "remember: an animal is a living thing",
)


def _run_sleep(neuter, ticks=260):
    """One arm of the schema-induction A/B: teach six taxonomy facts, then let the brain sleep long
    enough to cross the `_nights % 4 == 0` gate, and count what it wrote to the DURABLE lake."""
    work = tempfile.mkdtemp(prefix="gf_schema_")
    os.environ["NORD_HOME"] = work  # isolated: never the operator's ~/.snn_data_lake
    try:
        from snn_data_lake.substrate.home import KnowledgeLog

        if neuter:
            _patch_method(
                "snn_data_lake.reasoning.schema_induction",
                "SchemaInducer",
                "consolidate",
                lambda *_a, **_k: {"derived": [], "kept": 0, "confidence_gate_ran": True},
            )
        b = make_brain(seed=0, persist=True, day_period=20)
        for f in _SLEEP_FACTS:
            b.respond(f)
        taught = sum(1 for r in KnowledgeLog().replay() if isinstance(r.get("triple"), list))
        for _ in range(ticks):
            b.tick()
        rows = [r for r in KnowledgeLog().replay() if r.get("src") == "schema"]
        nights = int(getattr(b, "_nights", 0))
        b.close()
        return taught, nights, [tuple(r.get("triple") or ()) for r in rows]
    finally:
        shutil.rmtree(work, ignore_errors=True)
        os.environ["NORD_HOME"] = _SANDBOX


def _sleep_schema_induction():
    ta, na, wa = _run_sleep(False)
    tb, nb, wb = _run_sleep(True)
    print("faculty                                       : sleep_schema_induction")
    print("observable                                    : DURABLE WRITE, not a reply")
    print(f"taught triples in the durable log             : {ta} (A) / {tb} (B)")
    print(f"subjective nights completed                   : {na} (A) / {nb} (B)")
    print(f"derived src='schema' records written  A(wired): {len(wa)}  {sorted(wa)}")
    print(f"derived src='schema' records written  B(neut) : {len(wb)}  {sorted(wb)}")
    if not wa:
        print(
            "NOT-ELIGIBLE: the wired arm wrote no schema at all -- the branch's gate chain "
            "(persist AND _nights % 4 == 0 AND >= 4 triples AND confidence_gate_ran) did not open "
            "in this run, so there is no charge verdict to give"
        )
        return 2
    if sorted(wa) != sorted(wb):
        print(
            "MEMORY-WRITE CHARGE OK: neutering the inducer changes what the brain durably stores "
            "and can later retrieve (rubric band 41-60, 'a memory write'). NOT band 61 -- no "
            "decision, action or weight was shown to read it back"
        )
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE: the same records are written with the inducer neutered")
    return 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sleep_schema_induction":
        return _sleep_schema_induction()
    if len(sys.argv) < 2 or sys.argv[1] not in FACULTIES:
        print("usage: python docs/coverage/artifacts/gf_reply_ab.py <faculty>")
        print("faculties: sleep_schema_induction, " + ", ".join(sorted(FACULTIES)))
        return 2
    name = sys.argv[1]
    queries, marker, neuter = FACULTIES[name]

    a = run(queries)  # A: wired
    neuter()  # B: the faculty's entry points return their honest-gap value
    b = run(queries)

    hits = sum(1 for r in a if re.search(marker, r))
    flips = sum(1 for x, y in zip(a, b, strict=True) if x != y)
    n = len(queries)
    print(f"faculty                                       : {name}")
    print(f"queries                                       : {n}")
    print(f"wired replies carrying the faculty marker     : {hits}/{n}")
    print(f"reply changed by neutering the faculty        : {flips}/{n} turns")
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        print(f"  [{i}] Q: {queries[i][:88]}")
        print("      A(wired)   : " + x[:150].replace("\n", " "))
        print("      B(neutered): " + y[:150].replace("\n", " "))

    if not hits:
        print(
            "NOT-ELIGIBLE: no wired reply carries this faculty's marker -- another dispatch branch "
            "answered every turn first (recall/ignition preempts the reasoning cortex at "
            "spike_chat_lake.py:1047), so the faculty never got a chance to reach the reply and "
            "there is no charge verdict to give"
        )
        return 2
    if flips > 0:
        print(
            "REPLY CHARGE OK: the faculty's output reaches the brain's reply (rubric band 41-60). "
            "NOT band 61 -- no decision, action or weight is downstream; the route is read-only by "
            "construction (live_brain.py:443)"
        )
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE: the faculty answered, and neutering it changed no reply")
    return 1


if __name__ == "__main__":
    sys.exit(main())
