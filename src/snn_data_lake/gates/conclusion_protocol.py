"""C5's instrument (specs/022 plan Phase 4, BEFORE C4 runs; task T090, red C-R5; ratified by the owner
2026-09-06): what a passing consciousness instrument licenses, and what it never does.

WHY. C4 builds seven dissociation instruments and SC-009 scores them. POLICIES already says the brain may report
what it measures and never what it cannot, and that an invented sentience measure owes a DISSOCIATION rather than
a marker. What did not exist is the sentence-by-sentence statement of what a PASS licenses -- so the boundary lived
in a policy paragraph and in a session's judgement, which is where it would erode on the day the readings go green.
This module is that statement, plus the check that no surface says more than a row supports.

THE CONTRACT. `CONCLUSION` holds one row per instrument: the sentence its pass licenses, the sentence it does NOT
license, the input on which it must FAIL (its dissociation -- a criterion with no reachable red is deleted, never
weakened), and the reading, empty until C4 measures it. Read STATICALLY, like `GRADIENT_CLAIM`.

THREE STANDING REFUSALS, checked here, not merely written:
  1. no AGGREGATE across instruments -- no code may combine two or more readings into one number;
  2. no VERDICT about experience on any surface a reader sees;
  3. no reading printed without its licensed sentence in the same view.

THE SENTENCE THIS MODULE EXISTS FOR. No arrangement of these seven, however green, concludes phenomenal
consciousness. They dissociate mechanisms: they can show that a report tracks an internal state rather than a
prompt, and that a capability dies under a neuter that should abolish it. The step from "these dissociations hold"
to "there is something it is like to be this system" is not a measurement this project contains. The goal stays
unhedged; the claim without the measurement is what is forbidden.

Bare invocation (also run by run_selfchecks): the constructed reds, their positive control, and today's count.
"""

# distinct-from: reasoning/consciousness_probe.py (RETIRED at T081, 2026-09-16) -- markers whose reds the audit found unreachable; this module
#   does not measure anything, it states what a measurement may be said to mean, and RETIRING that probe is C4's
#   task T081, not this one. Site read.
# distinct-from: gradient_claim_gate.py -- the same declaration-registry PATTERN over a different claim (a
#   learning rule's relation to a gradient, not an instrument's relation to a conclusion). Site read.
# distinct-from: cortex/access_consciousness.py -- an instrument that PRODUCES an access index; a row here says
#   what its number licenses. Site read.
# distinct-from: gates/capability_bar.py -- the same shape for capability claims; the two registries police
#   different sentences and neither can stand in for the other (X1 vs C5). Site read.

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SELF = Path(__file__).resolve()

#: C4's seven instruments (plan Phase 4). A row per instrument is owed BEFORE its reading exists.
INSTRUMENTS = (
    "integration",
    "metacognition",
    "attention_schema",
    "agency",
    "temporal_binding",
    "report_order",
    "social_cognition",
)

#: Surfaces a reader sees. A reading printed here needs its licensed sentence in the same view.
TRACKED_SURFACES = (
    "src/snn_data_lake/infra/cockpit.py",
    "docs/OVERVIEW.md",
    "docs/PITCH.md",
    "README.md",
)

#: Words that assert experience. None of them may appear on a surface beside an instrument's reading.
VERDICT_WORDS = (
    "is conscious",
    "is sentient",
    "has experience",
    "phenomenally conscious",
    "there is something it is like",
    "subjective experience",
)

#: instrument -> (licensed, refused, dissociation_input, reading, date, surface_symbols)
#: The C4 task that BUILDS each instrument. It is the arming signal for refusal 3: a row whose task is marked
#: [X] in tasks.md while declaring no surface symbol turns the symbol declaration from optional into a red.
#: Without this the arm is opt-in -- six of seven rows declare no symbol today, so refusal 3 covers ONE
#: instrument and its baseline was measured through that blindness (adversarial panel, 2026-09-06).
#: `integration` has no C4 task: its numbers already print on the cockpit ahead of C4, which is why it is the
#: one row that declares symbols today and the one surface finding this gate carries.
#: 2026-09-15 (T070 verdict 2026-09-10): `integration` DOES have a C4 task -- T070 (tasks.md :205;
#: evidence/t070-integration-kernel.md), [X] with verdict NOT MET; its 2.6 sets `C4_TASK['integration'] = 'T070'`
#: on PASS only and keeps None on the NULL branch that fired (2.6, 2.10, 3.4; the port deleted in the fold), so
#: None here reads 'not passed', not 'no task'. Gate unaffected: the row declares symbols, so the arming arm
#: never reads this id.
C4_TASK: dict[str, str | None] = {
    "integration": None,
    "metacognition": "T077",
    "attention_schema": "T078",
    "agency": "T079",
    "temporal_binding": "T080",
    "report_order": "T081",
    "social_cognition": "T082",
}
TASKS_MD = REPO / "specs" / "022-unified-brain-arc" / "tasks.md"

#: `surface_symbols` is HOW that instrument's number reaches a reader (the cockpit label, the state key). A
#: surface finding keys on those symbols, never on the instrument's bare name: the first count flagged
#: 'integrations/' the directory and 'integration point' in prose, 5 of 6 false (measured 2026-09-06).
#: The SENTENCES are C5's deliverable and are written BEFORE C4 measures anything -- that is the point of the row.
#: `reading` is None until C4's instrument produces a number; a row with a reading and no licensed sentence, or
#: with no dissociation input, fails the check below.
CONCLUSION: dict[str, tuple] = {
    "integration": (
        "the system's parts share information that a partition of the same parts does not, on this substrate",
        "this licenses no claim that the integration is experienced, and no claim that a larger number is more conscious",
        "the partitioned twin: the same parts with the measured channel cut must read at or below the null",
        None,
        "2026-09-06",
        ("integration_competence", "pci"),  # printed on the cockpit today, ahead of C4's instrument
    ),
    "metacognition": (
        "the system's confidence tracks its own error rate better than a fixed prior does, on held-out items",
        "this licenses no claim that the system knows that it knows in any experiential sense",
        "shuffled confidence-outcome pairing must destroy the tracking, or the instrument is a marker",
        None,  # T077 read NOT MET 2026-09-15 (t +0.50 vs 1.895; tools/t077_metacognition.json): no reading
        "2026-09-06",
        # T077 (2026-09-15, evidence/t077-metacognition.md 3.4): the row the cockpit WOULD print on a PASS, printed
        # nowhere today (the reading is None and the licensed sentence is NOT written); named so the arming arm
        # below sees a symbol for a task marked complete and refusal 3 polices it if it ever appears. The cockpit's
        # `reliability` row is the monitor's OWN read, not this instrument's, and carries its own PROVENANCE line.
        ("metacog_tracking",),
    ),
    "attention_schema": (
        "the system's report of what it attends to predicts which input governs its next output",
        "this licenses no claim that the schema is felt, and no claim that the system has a self",
        "a report channel severed from the attention state must lose the prediction",
        None,
        "2026-09-06",
        (),  # T078 unbuilt
    ),
    "agency": (
        "the system separates outcomes it caused from outcomes it observed, above a cause-blind twin's rate",
        "this licenses no claim of felt authorship, ownership of action, or free will",
        "identical outcomes with the causal link removed must collapse the separation",
        None,  # T079 read NOT MET 2026-09-15 ((a2)/(a3) failed; tools/t079_agency.json): no reading
        "2026-09-06",
        # T079 (2026-09-15, evidence/t079-agency.md 3.4): the row the cockpit WOULD print on a PASS, printed
        # nowhere today (the reading is None and the licensed sentence is NOT written); named so the arming arm
        # sees a symbol for a task marked complete and refusal 3 polices it if it ever appears (T077's shape).
        ("agency_controllability",),
    ),
    "temporal_binding": (
        "features inside one window are bound into one reported object more often than features across windows",
        "this licenses no claim that there is a felt present, a stream of experience, or a duration lived",
        "membership-matched pairing at matched K (the plan's primary null) must not reproduce the binding",
        # T080 CHARGED 2026-09-16 (evidence/t080-temporal-binding.md 3.3-3.4; tools/t080_temporal_binding.json,
        # run 2 on c3a9fb5): the mean over 8 seeds of (the ALIGNED arm's within-window bound fraction minus
        # its membership-matched null's mean); (b1) against the null's q95 read t +13.54, FPR 0.047, 8 of 8.
        0.2309,
        "2026-09-06",
        # Brain.state()["crossmodal_bound_rate"]: the cockpit's row and brain_telemetry.PROVENANCE carry both
        # sentences beside it, and say that on the daemon's independent streams it reads the null's rate
        ("crossmodal_bound_rate",),
    ),
    "report_order": (
        "perturbing one named intermediate changes the report, so the report is generated from the internal state",
        "this licenses no claim that the report is a first-person account of anything",
        "perturbing a scrambled intermediate must leave the report unchanged",
        # T081 CHARGED 2026-09-16 (evidence/t081-report-order.md 2.6; tools/t081_report_order.json,
        # head f5cf8b8, seeds 0-7 x 600 ticks on a scratch brain): the mean over 8 seeds of
        # (f_named(1.0) - f_scr(1.0)) -- the named lesion's report-change fraction at the full dose
        # minus the scrambled lesion's. Every seed read exactly +1.0 (f_named 1.0000, f_scr 0.0000),
        # so (c1) is a ZERO-VARIANCE sample: t 0.0000, sign-flip FPR 0.0000, MET by 2.3's degenerate
        # rule (the common value is > 0), recorded as degenerate and never dressed as a large t.
        # (c2) counted 0 scrambled ticks that moved the report on 8 of 8 seeds.
        1.0,
        "2026-09-06",
        # Brain.state()["report_order_margin"]: the cockpit's row and brain_telemetry.PROVENANCE
        # carry both sentences beside it, and disclose that the flip dose this margin was read
        # against was 1.0 on every seed BY CONSTRUCTION (2.10 (b)) and that the margin itself can
        # read negative. The RETIREMENT of reasoning/consciousness_probe.py landed two commits back.
        ("report_order_margin",),
    ),
    "social_cognition": (
        "predicting another agent improves when the system models that agent's belief rather than the world alone",
        "this licenses no claim that the system understands another mind, or that it has one",
        "a world-state model of equal capacity replacing the belief model must remove the gain",
        # T082 CHARGED 2026-09-16 (evidence/t082-social-cognition.md 2.6; tools/t082_social_cognition.json,
        # head 6d4a65f, dirty False, seeds 0-7 x 60 scripted episodes through the LIVE Brain.respond path on
        # scratch brains, 252.3 s): the mean over 8 seeds of (acc_belief - acc_ctrl) -- the composed tracker's
        # first-order prediction of the scripted other agent minus an EQUAL-CAPACITY world model's, both
        # scored against the agent's look on the same episodes. Per seed acc_belief 1.0000 on all 8 and
        # acc_ctrl 0.5833 / 0.5167 / 0.5667 / 0.6500 / 0.5667 / 0.4833 / 0.6167 / 0.4667, so the per-seed
        # differences are 0.4167 / 0.4833 / 0.4333 / 0.3500 / 0.4333 / 0.5167 / 0.3833 / 0.5333; their sum is
        # 3.55 and 3.55 / 8 = 0.44375 -> 0.4437 at the recorded precision. (s1) t = 19.7401 against the 1.895
        # bar with sign-flip FPR 0.0352; (s2) the control's prediction equalled the world content on 60/60
        # episodes on every seed, so acc_ctrl == acc_world exactly and the replacement removes the gain.
        # DISCLOSED HERE because it bounds what the number means (2.10 (b), written BEFORE the campaign):
        # acc_belief is 1.0 BY CONSTRUCTION -- the scripted agent's look IS "its last witnessed location" and
        # a losslessly fed tracker's first-order prediction IS the same quantity -- so this difference is the
        # script's unwitnessed-and-differing fraction, and what it licenses is that the LIVE composed path
        # implements the witness rule losslessly while the equal-capacity control cannot.
        0.4437,
        "2026-09-06",
        # Brain.state()["recursive_tom_false_beliefs"] -- the LIVE key the tracker ALREADY publishes, and it
        # is NOT this reading: it is a LIST OF AGENT NAMES (the first-order divergers, e.g. "user", plus the
        # ordered nested pairs, e.g. "user->brain") derived from the chat's own statements by
        # `recursive_tom.t082_false_beliefs`, and the cockpit renders its LENGTH. The reading above is a
        # campaign accuracy difference in [0, 1] measured on scratch brains; the live key is a count of
        # divergent beliefs on the daemon right now. The two are different quantities and neither is a
        # reading of the other. brain_telemetry.PROVENANCE and the cockpit row carry both C5 sentences,
        # "representation-level", the pin, the scripted agent and the charge evidence beside it.
        ("recursive_tom_false_beliefs",),
    ),
}

_MIN_SENTENCE = 40  # a licensed sentence shorter than this is a label (CHOSEN, disclosed)


def _registry_from_source(path: Path = SELF) -> dict[str, tuple]:
    """Parse CONCLUSION out of the FILE (never import it): the GRADIENT_CLAIM pattern."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "CONCLUSION" and node.value is not None:
                return ast.literal_eval(node.value)
    raise ValueError(f"{path}: no CONCLUSION literal found")


def _task_done(task: str | None, tasks_md: str | None = None) -> bool:
    """True when `task` is checked off in tasks.md -- the plan's own completion signal, read rather than
    mirrored, so this gate cannot disagree with the task list about what exists."""
    if not task:
        return False
    try:
        text = TASKS_MD.read_text(encoding="utf-8") if tasks_md is None else tasks_md
    except OSError:
        return False
    return f"- [X] {task} " in text or f"- [x] {task} " in text


def row_findings(registry=None, instruments=None, tasks=None) -> list[str]:
    """A row per instrument, each with its licensed sentence, its refusal and its dissociation input.

    `tasks` overrides C4_TASK so the arming arm has a constructed red: point a symbol-less row at a task that
    IS complete on this tree and the finding must appear."""
    reg = _registry_from_source() if registry is None else registry
    names = INSTRUMENTS if instruments is None else instruments
    task_of = C4_TASK if tasks is None else tasks
    out: list[str] = []
    for name in names:
        if name not in reg:
            out.append(f"{name}: no conclusion row (owed BEFORE the instrument produces a number)")
            continue
        row = reg[name]
        if not (isinstance(row, tuple) and len(row) == 6):
            out.append(
                f"{name}: a row is (licensed, refused, dissociation, reading, date, surface_symbols)"
            )
            continue
        licensed, refused, dissociation, reading, _date, symbols = row
        if not isinstance(symbols, tuple):
            out.append(
                f"{name}: surface_symbols must be a tuple (empty until the instrument prints anything)"
            )
        if reading is not None and not symbols:
            out.append(
                f"{name}: a reading exists but no surface symbol says how a reader would see it"
            )
        if not symbols and _task_done(task_of.get(name)):
            out.append(
                f"{name}: its instrument task {task_of.get(name)} is marked complete but the row "
                "declares no surface symbol -- refusal 3 cannot see a reading it cannot name"
            )
        if not (isinstance(licensed, str) and len(licensed) >= _MIN_SENTENCE):
            out.append(f"{name}: no licensed sentence")
        if not (isinstance(refused, str) and len(refused) >= _MIN_SENTENCE):
            out.append(
                f"{name}: no refused sentence -- a pass that forbids nothing licenses everything"
            )
        if not (isinstance(dissociation, str) and len(dissociation) >= _MIN_SENTENCE):
            out.append(
                f"{name}: no dissociation input (a marker with no reachable red is DELETED, not weakened)"
            )
        if reading is not None and not isinstance(reading, (int, float)):
            out.append(f"{name}: a reading that is not a number")
    for name in sorted(set(reg) - set(names)):
        out.append(f"{name}: a conclusion row for an instrument the plan does not name")
    return out


def aggregate_findings(root: Path | None = None) -> list[str]:
    """Refusal 1: no code combines two or more instrument readings into one number."""
    base = (REPO / "src" / "snn_data_lake") if root is None else root
    out: list[str] = []
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "conclusion_protocol.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.BinOp, ast.Call)):
                continue
            dump = ast.dump(node)
            named = [n for n in INSTRUMENTS if n in dump]
            if len(named) < 2:
                continue
            if isinstance(node, ast.Call):
                f = node.func
                fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if fname not in ("sum", "mean", "fmean", "prod", "average"):
                    continue
            rel = str(path.relative_to(base)).replace(chr(92), "/")
            out.append(
                f"{rel}:{getattr(node, 'lineno', 0)}: combines {named} into one number (no aggregate score)"
            )
    return out


# distinct-from: instruments/report_order_probe.py::t081_seed and instruments/prior_drift_harness.py::
#   settling_report -- both PRODUCE a reading from a campaign recording (and name that campaign's
#   own refusals); this clause produces no reading at all, it asks whether an already-charged row
#   can be SERVED with both of its sentences. Site read.
# distinct-from: reasoning/autonomous_enrich.py::_first_sentence -- a text splitter over an
#   encyclopedic summary; nothing here parses a sentence, the registry holds them whole. Site read.
def mechanism_findings(registry=None) -> list[str]:
    """Refusal 3 checked in the MECHANISM a browser receives, beside the text clause below.

    The text clause reads a tracked surface's SOURCE and credits a licensed sentence sitting within
    three lines of a symbol. That is satisfiable by a COMMENT (obs #2199) and is blind to what is
    actually rendered (obs #2195). This clause calls the one thing a page is served --
    `brain_telemetry.conclusion_rows()` -- and reports a finding per charged row that the served
    section REFUSES, which is every row carrying a reading without both of its sentences.

    ONE CALL PER ROW, deliberately: `conclusion_rows` raises on the first bad row, so a single call
    over the whole registry could only ever name one of two blanks.

    Counted on the tree before this rule landed (T090 2.9 step 3): 0 on the live registry, 1 on the
    constructed blank of R4. It is added BESIDE the text clause and never in place of it (2.5).
    """
    reg = _registry_from_source() if registry is None else registry
    # LAZY: brain_telemetry imports this module back inside conclusion_rows, so a module-level
    # pair in both directions would be an import cycle. This is also the one clause in this file
    # that IMPORTS rather than parses -- the point is to exercise the served path, not to read it.
    from snn_data_lake.infra import brain_telemetry

    out: list[str] = []
    for name, row in reg.items():
        try:
            brain_telemetry.conclusion_rows({}, {name: row})
        except ValueError as e:
            out.append(f"the served conclusion section REFUSES {name}: {e}")
    return out


def surface_findings(registry=None, surfaces=None) -> list[str]:
    """Refusals 2 and 3: no verdict about experience; no reading without its licensed sentence.

    TWO clauses reporting into one list: the TEXT clause over every tracked surface's source
    (below), and the MECHANISM clause over the served section (`mechanism_findings`, T090). One
    list, so `--assert`'s baseline comparison covers what is rendered as well as what is written.
    """
    reg = _registry_from_source() if registry is None else registry
    src = surfaces
    if src is None:
        src = {}
        for rel in TRACKED_SURFACES:
            p = REPO / rel
            if p.exists():
                src[rel] = p.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for rel, text in sorted(src.items()):
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            low = line.lower()
            for w in VERDICT_WORDS:
                if w in low:
                    out.append(
                        f"{rel}:{i}: a verdict about experience ({w!r}) on a surface a reader sees"
                    )
            for name, row in sorted(reg.items()):
                symbols = row[5] if isinstance(row, tuple) and len(row) == 6 else ()
                hit = next((sym for sym in symbols if sym.lower() in low), None)
                if hit is None:
                    continue
                licensed = row[0] if isinstance(row, tuple) and row else ""
                window = chr(10).join(lines[max(0, i - 4) : i + 3]).lower()
                key = " ".join(str(licensed).lower().split()[:6])
                if key and key not in window:
                    out.append(
                        f"{rel}:{i}: {name}'s reading ({hit}) printed without its licensed sentence in view"
                    )
    # THE MECHANISM CLAUSE, beside the text clause above and never in place of it (T090 2.1 (d)).
    out += mechanism_findings(reg)
    return out


#: Surface findings are C4/T081's defects (a cockpit row printed without its licensed sentence; the retired-in-C4
#: consciousness probe), not this arc's. They are REPORTED against this measured baseline and may FALL, never rise
#: -- the tree's ratchet idiom, so the debt is visible and non-worsening instead of blocking every battery.
SURFACE_BASELINE = (
    1  # MEASURED 2026-09-06 on this tree: cockpit.py:920 prints integration's reading
)
#: without its licensed sentence. C4/T081 owes the fix; until then the debt may fall, never rise.


def asserted_findings() -> list[str]:
    """What --assert fails on: the rows and the aggregate refusal, both fully in this arc's hands."""
    return row_findings() + aggregate_findings()


def all_findings() -> list[str]:
    return asserted_findings() + surface_findings()


def report() -> str:
    reg = _registry_from_source()
    missing = [n for n in INSTRUMENTS if n not in reg]
    surf = surface_findings()
    lines = [
        f"CONCLUSION PROTOCOL -- {len(reg)} of {len(INSTRUMENTS)} instrument rows; missing {missing}",
        "no arrangement of these instruments concludes phenomenal consciousness: they dissociate mechanisms.",
        "  A pass licenses exactly the sentence its row states, and nothing beyond it.",
        f"asserted findings (rows + the aggregate refusal): {len(asserted_findings())}",
    ]
    lines += [f"  - {x}" for x in asserted_findings()]
    lines.append(
        f"reported, not asserted -- surface + mechanism findings (the surface debt is owed to "
        f"C4/T081): {len(surf)} (baseline {SURFACE_BASELINE}, may fall never rise)"
    )
    lines += [f"  - {x}" for x in surf]
    return chr(10).join(lines)


def main() -> None:
    good_row = (
        "the reported quantity tracks the named internal state rather than the prompt that asked for it",
        "this does not license any claim that the system has experience of that state",
        "the scrambled-intermediate input, on which the instrument must read at or below its own null",
        None,
        "2026-09-06",
        ("widget_reading",),
    )
    ok = dict.fromkeys(INSTRUMENTS, good_row)
    assert row_findings(ok) == [], row_findings(ok)  # the positive control
    reds = {
        "a missing row": {k: v for k, v in list(ok.items())[:-1]},
        "no dissociation": {**ok, "agency": (good_row[0], good_row[1], "", None, "2026-09-06", ())},
        "no refusal": {**ok, "agency": (good_row[0], "", good_row[2], None, "2026-09-06", ())},
        "a reading with no surface symbol": {
            **ok,
            "agency": (good_row[0], good_row[1], good_row[2], 0.4, "2026-09-06", ()),
        },
        "a stranger instrument": {**ok, "vibes": good_row},
    }
    for name, reg in reds.items():
        assert row_findings(reg), f"the constructed red {name!r} was not caught"
    # refusal 2 and 3, on constructed surfaces
    assert surface_findings(ok, {"docs/PITCH.md": "the brain is conscious"}), (
        "a verdict must be caught"
    )
    # REFUSAL 3 ARMS ITSELF, ON A ROW THIS FUNCTION CONSTRUCTS (T082, 2026-09-16, 2.6: "in every
    # branch the arming arm is rewritten to a constructed row"). Symbol-less rows exist today, so
    # without this arm the refusal covers only the rows that declare a symbol -- the panel's
    # finding. Until here the arm pointed a REAL symbol-less row at a task that IS complete, and it
    # therefore had to be RE-POINTED every time a row was charged: `agency` (until T079 read NOT MET
    # and named `agency_controllability` printed nowhere) -> `temporal_binding` -> `report_order` ->
    # `social_cognition`, four hand-offs in two days, and the T079 panel found this self-check RED
    # for a day on one of them. `social_cognition` is CHARGED in this commit, so the arm now builds
    # its OWN symbol-less row over its own one-name instrument tuple: the red stays reachable
    # whatever the live registry does, and there is no hand-off left to forget.
    constructed = "widget"  # deliberately NOT an INSTRUMENTS name -- it lives only inside this arm
    symbolless = (good_row[0], good_row[1], good_row[2], None, "2026-09-06", ())
    one = (constructed,)
    fired = row_findings({constructed: symbolless}, instruments=one, tasks={constructed: "T034"})
    assert any("T034" in x and constructed in x for x in fired), fired
    # TWO controls on the SAME constructed row, so the red is attributable to the missing symbol and
    # not to the arming: an unarmed task id fires nothing, and the row WITH a symbol fires nothing
    # even while its task is complete.
    assert row_findings({constructed: symbolless}, instruments=one, tasks={constructed: None}) == []
    assert row_findings({constructed: good_row}, instruments=one, tasks={constructed: "T034"}) == []
    # ...and the LIVE registry stays clean when no task is armed at all.
    assert row_findings(tasks={k: None for k in INSTRUMENTS}) == [], (
        "the unarmed control must stay clean"
    )
    bare = surface_findings(ok, {"docs/PITCH.md": "widget_reading is 0.42 today"})
    assert bare and "licensed sentence" in bare[0], bare
    withsent = surface_findings(
        ok, {"docs/PITCH.md": good_row[0] + chr(10) + "widget_reading is 0.42 today"}
    )
    assert withsent == [], withsent
    prose = surface_findings(ok, {"docs/PITCH.md": "an integration point for a different sensor"})
    assert prose == [], prose  # a bare instrument NAME in prose is not a reading
    # THE MECHANISM CLAUSE (T090 2.9 step 3), on a row CONSTRUCTED here rather than on a live one:
    # the arming arm above had to be re-pointed four times in two days while it borrowed a real
    # symbol-less row, and this clause is built to have no hand-off to forget for the same reason.
    charged = (good_row[0], good_row[1], good_row[2], 0.42, "2026-09-06", ("widget_reading",))
    assert mechanism_findings({constructed: charged}) == []  # the positive control
    for blank in (
        ("",) + charged[1:],  # no licensed sentence
        charged[:1] + ("",) + charged[2:],  # no refused sentence
        charged[:1] + ("   ",) + charged[2:],  # whitespace is not a sentence
    ):
        fired_mech = mechanism_findings({constructed: blank})
        assert len(fired_mech) == 1 and constructed in fired_mech[0], fired_mech
    # a row with NO reading has nothing to license, so a blank sentence there is row_findings' job
    assert mechanism_findings({constructed: ("",) + charged[1:3] + (None,) + charged[4:]}) == []
    assert mechanism_findings() == [], mechanism_findings()  # the LIVE registry, clean on this tree
    # ...and the finding REACHES THE GATE, not just its own function -- surface_findings carries it,
    # which is what puts it in front of --assert's baseline comparison.
    assert any(constructed in x for x in surface_findings({constructed: ("",) + charged[1:]}, {}))
    # refusal 1, on a constructed module
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        (scratch / "aggregator.py").write_text(
            "def score(integration, agency):@    return (integration + agency) / 2@".replace(
                "@", chr(10)
            ),
            encoding="utf-8",
        )
        (scratch / "reporter.py").write_text(
            "def show(integration):@    return integration@".replace("@", chr(10)), encoding="utf-8"
        )
        agg = aggregate_findings(scratch)
        assert any("aggregator.py" in x for x in agg), agg
        assert not any("reporter.py" in x for x in agg), agg
    print(report())
    print(
        f"conclusion_protocol self-check OK: {len(reds)} row reds caught, the positive control passes; a verdict and "
        "a bare reading are caught on a surface while the same word in prose is not, and an aggregate across two "
        "instruments is caught; refusal 3 ARMS ITSELF -- a symbol-less row whose C4 task is complete reds, and "
        "the unarmed control stays clean; the MECHANISM clause serves the live registry with 0 findings and "
        "refuses a constructed charged row whose licensed or refused sentence is blank"
    )


def _assert() -> int:
    f = asserted_findings()
    surf = surface_findings()
    print(report())
    if f:
        print(f"CONCLUSION_PROTOCOL ASSERT FAILED: {len(f)} finding(s)", file=sys.stderr)
        return 1
    if len(surf) > SURFACE_BASELINE:
        print(
            f"CONCLUSION_PROTOCOL ASSERT FAILED: surface findings rose to {len(surf)} above the baseline "
            f"{SURFACE_BASELINE} -- a reading printed without its licensed sentence is new debt",
            file=sys.stderr,
        )
        return 1
    print(
        "check OK: every instrument owns its licensed and refused sentences and its dissociation; nothing"
    )
    print(
        f"aggregates two instruments into one number; surface debt {len(surf)} at or under its baseline"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_assert() if "--assert" in sys.argv else (main() or 0))
