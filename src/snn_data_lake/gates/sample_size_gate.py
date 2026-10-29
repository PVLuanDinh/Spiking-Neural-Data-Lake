"""P1 sample-size gate -- every pre-registration carries a "Sample size (P1)" section with its fields.

Rule enforced: docs/POLICIES.md, "P1 amendment -- sample size is measured, and a campaign runs
sequentially" (landed in fb7b6581, 2026-10-24). Its Gate specification, verbatim:

    "a check that every pre-registration carries the section's fields, read from the file as it stands
    at HEAD (measured SD, effect, target, looks or the one-shot reason, spending function, futility,
    cap), with a planted-missing red; the pre-registrations frozen before this amendment landed are a
    CLOSED list, grandfathered by name when the gate is built (ancestry is not the trigger: a branch
    merged after the landing would slip past it). Count it before this marker retires."

# distinct-from: p5b_gate.py -- that one checks that a native Rust module's header cites a pre-registration
# recording the six P5b criteria; this one reads the evidence markdown itself for the P1 section. Same word
# "pre-registration", different corpus and job. null_scale_gate.py is the shape precedent AND the reuse
# source: its fence-aware heading/section reader and `Red` are imported, not copied (P2).

A pre-registration is any `specs/**/evidence/*.md` file that contains `pre-?regist` or `prereg` (the
reproducible rules R1 and R2 of specs/022-unified-brain-arc/evidence/p1-sample-size-grandfather.md, which
this gate's lists are built from; R1 is a subset of R2). Each one owes a heading named "Sample size (P1)"
(any level, outside a code fence); the text of all such sections of the file is read together and must carry
each field by its word:

    measured SD  `SD` | `standard deviation`      effect    `effect` | `mde`        target  `target`
    looks        `look(s)`                          spending  `spending` | O'Brien | Pocock
    futility     `futility`                         cap       `cap`

A section that says `one-shot` is a fixed count (the amendment's item 3): it owes the measured SD, the effect
and the target, plus the one-shot reason itself; looks, spending, futility and cap are not owed.

Three closed lists, all by file name (never by ancestry):
  GRANDFATHERED          frozen before the landing; exempt whole, so a later edit to a grandfathered file is
                         not read by this gate (the list file says a section added later is not grandfathered;
                         that stays review-enforced).
  OWES_AT_FREEZE         DRAFT pre-registrations that have not frozen; deferred and printed on every run
                         while every pre-registration heading of the file says DRAFT, red once it does not.
  NOT_PREREGISTRATIONS   files the word matches that say of themselves they are not a pre-registration.
A name in any list whose file is gone (or, for GRANDFATHERED, no longer mentions pre-registration) is
R-STALE-GRANDFATHER, so the lists only shrink.

What it reds: R-NO-SECTION, R-FIELD, R-STALE-GRANDFATHER.
What it cannot do, and says so on every run: judge whether a number is right, a simulation was run, or a
boundary is sound (the policy-enforcer and the panel do); it checks that each field is WRITTEN.

  python -m snn_data_lake.gates.sample_size_gate [--assert | --self-check | --census]

Bare invocation is --assert (the run_selfchecks.py convention).
"""

from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from snn_data_lake.gates.null_scale_gate import Red, _lines, _rels, _section, _structure

REPO = Path(__file__).resolve().parents[3]

EVIDENCE_GLOB = "specs/**/evidence/*.md"
LANDING = "fb7b6581"  # the commit that added "### P1 amendment" to docs/POLICIES.md
PREREG_RE = re.compile(r"pre-?regist|prereg", re.I)
SECTION_RE = re.compile(r"sample size \(p1\)", re.I)
ONE_SHOT_RE = re.compile(r"one-shot", re.I)
FIELDS = {
    "measured SD": r"\bSD\b|standard deviation",
    "effect": r"\beffect\b|\bmde\b",
    "target": r"\btarget\b",
    "looks": r"\blooks?\b",
    "spending": r"spending|O'Brien|Pocock",
    "futility": r"futility",
    "cap": r"\bcap\b",
}
ONE_SHOT_FIELDS = ("measured SD", "effect", "target")

_EV = "specs/022-unified-brain-arc/evidence/"

# The pre-registrations frozen before the landing, by name (73): the 76 of p1-sample-size-grandfather.md at
# c9d0c097 less its section 4 (g2, g3, g5: DRAFT, see OWES_AT_FREEZE). Census at c4c10b28: none of them was first
# added after the landing (author time). May shrink only by deleting an entry in the same commit as its file
# (R-STALE-GRANDFATHER). Regenerate the candidates with `--census`.
GRANDFATHERED: frozenset[str] = frozenset(
    {
        "docs/AGENT_COLD_START.md",
        "specs/011-h01-neurogenesis-prior/decisions.md",
        "specs/022-unified-brain-arc/quickstart.md",
        *(
            _EV + name + ".md"
            for name in (
                "c0-everyday-battery",
                "docs-only-battery",
                "encoder-path-root",
                "engine-structure-2026-09-25",
                "g0a-chat-correction",
                "g0b-reply-consumes-route",
                "g0c-learner-persists",
                "g1a-causal-contract",
                "g1b-pool-selection",
                "g1c-carried-state",
                "g3-t0-baseline",
                "g4-wiring-prior",
                "g5-handoff-repair-2026-10-11",
                "g6-sleep-durable-memory",
                "g7-organizing-unit",
                "g7-unit-label-gate",
                "g8-reorganization",
                "gates-for-the-ratified-rows",
                "insula-level-arms-production",
                "insula-p8-engine-reach",
                "jepa-collapse",
                "jepa-live-floor",
                "l-exit",
                "l-parallel-arms",
                "l1b-synaptic-online",
                "l1c-online-readout",
                "l2-grow-on-plateau",
                "l4-resume-and-the-append-ceiling",
                "l4b-append-ceiling-nonstationary",
                "l4c-ignition-is-coalition-size",
                "l6-immortal-vs-consolidation",
                "learning-rules-1019",
                "null-scale-triage",
                "og1-organoid-stream",
                "p8-null-scale-gate",
                "q2-surprise-context",
                "r1-bounded-recall",
                "r2-typed-replay-hydration",
                "r3-replay-growth-composition",
                "s1-genome-fanin-ab",
                "s1-placement-null",
                "s1-unit-blocks",
                "s2-exit",
                "s2-real-percept-stream",
                "s2b-growing-learner-lake-record",
                "s2c-rotating-stream-refused",
                "s3-neuron-pager",
                "s4-lake-compactor",
                "s5-paging-null",
                "s6-kernel-wire",
                "sc-012a-width",
                "scoring-path-abstention",
                "t062-h01-ingest",
                "t063-prior-drift-harness",
                "t069-cerebellar-rule",
                "t070-integration-kernel",
                "t077-metacognition",
                "t078-attention-schema",
                "t079-agency",
                "t080-temporal-binding",
                "t081-report-order",
                "t082-social-cognition",
                "t090-conclusion-plate",
                "t101-brian2-contract",
                "trainer-retirement",
                "trajectory-audit-2026-09-01",
                "w-exit",
                "w1-goal-curriculum",
                "w2-context-latents",
                "w3-world-baselines",
            )
        ),
    }
)
# DRAFT pre-registrations that have not frozen (p1-sample-size-grandfather.md section 4): each owes the section
# the day its DRAFT marker goes.
OWES_AT_FREEZE: frozenset[str] = frozenset(
    _EV + name + ".md" for name in ("g2-taught-problems", "g3-growth-loop", "g5-readout-variants")
)
# Matched by the word, but say of themselves they are not a pre-registration (or are the list of them).
NOT_PREREGISTRATIONS: frozenset[str] = frozenset(
    _EV + name + ".md"
    for name in ("p1-sample-size-grandfather", "p5b-pfc-rng-record", "p5b-retroactive-record")
)


@dataclass
class Report:
    reds: list[Red] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    deferred: list[str] = field(default_factory=list)

    def codes(self) -> set[str]:
        return {r.code for r in self.reds}


def _text(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def _draft(lines) -> bool:
    heads = [lines[j].text for j, _ in _structure(lines)[0] if PREREG_RE.search(lines[j].text)]
    return bool(heads) and all("DRAFT" in h for h in heads)


def _problem(lines) -> Red | None:
    """None when the file carries a complete section; else the Red (without its path, set by the caller)."""
    heads = _structure(lines)[0]
    body = "\n".join(
        ln.text for j, _ in heads if SECTION_RE.search(lines[j].text) for ln in _section(lines, j)
    )
    if not body:
        return Red("R-NO-SECTION", "", 'no heading "Sample size (P1)" outside a code fence')
    owed = ONE_SHOT_FIELDS if ONE_SHOT_RE.search(body) else tuple(FIELDS)
    missing = [f for f in owed if not re.search(FIELDS[f], body, re.I)]
    if missing:
        return Red(
            "R-FIELD", "", f"the Sample size (P1) section does not write: {', '.join(missing)}"
        )
    return None


def check(
    root: Path = REPO,
    grandfathered: frozenset[str] | None = None,
    owes_at_freeze: frozenset[str] | None = None,
    not_prereg: frozenset[str] | None = None,
) -> Report:
    grandfathered = GRANDFATHERED if grandfathered is None else grandfathered
    owes_at_freeze = OWES_AT_FREEZE if owes_at_freeze is None else owes_at_freeze
    not_prereg = NOT_PREREGISTRATIONS if not_prereg is None else not_prereg
    report = Report()
    found = []
    for rel in _rels(root, EVIDENCE_GLOB):
        if rel not in not_prereg and PREREG_RE.search(_text(root, rel)):
            found.append(rel)
    owing = set()
    for rel in found:
        if rel in grandfathered:
            continue
        lines = _lines(root, rel)
        if rel in owes_at_freeze and _draft(lines):
            report.deferred.append(rel)
        elif red := _problem(lines):
            owing.add(rel)
            report.reds.append(Red(red.code, rel, red.message))
    gone = 0
    for names, must_mention in (
        (grandfathered, True),
        (owes_at_freeze, False),
        (not_prereg, False),
    ):
        for rel in sorted(names):
            path = root / rel
            if not path.is_file() or (must_mention and not PREREG_RE.search(_text(root, rel))):
                gone += must_mention
                msg = f"list entry {rel!r} matches no current pre-registration file"
                report.reds.append(Red("R-STALE-GRANDFATHER", rel, msg))
    report.counts = {
        "pre-registrations found": len(found),
        "grandfathered": len(grandfathered) - gone,
        "deferred (owes at freeze)": len(report.deferred),
        "declared not a pre-registration": len(not_prereg),
        "owing the section now": len(owing),
    }
    return report


def census(root: Path = REPO) -> list[tuple[str, str]]:
    """(path, code) of every pre-registration that would red with no list at all: the candidates for
    GRANDFATHERED, OWES_AT_FREEZE and NOT_PREREGISTRATIONS. A human sorts them by their frozen date."""
    report = check(root, frozenset(), frozenset(), frozenset())
    return [(r.where, r.code) for r in report.reds]


def render(report: Report) -> str:
    out = [
        "P1 SAMPLE-SIZE GATE -- every pre-registration derives its seed count (docs/POLICIES.md P1 amendment)"
    ]
    out += [f"  {name}: {n}" for name, n in report.counts.items()]
    out += [f"  DEFERRED (owes the section at freeze): {rel}" for rel in report.deferred]
    out.append(
        "  LIMIT: the gate checks that each field is WRITTEN (by its word, over all of a file's Sample size (P1) "
        "sections together); it cannot judge that a number is right, that a simulation ran, or that a boundary "
        "is sound, and a grandfathered file is exempt whole. Those stay review-enforced."
    )
    out += [f"  RED {r.code} {r.where}: {r.message}" for r in report.reds]
    return "\n".join(out)


_EVT = "specs/s/evidence/"
_SECTION = "## Sample size (P1)\n" + "\n".join(
    [
        "Measured SD 0.3, claimed effect 0.1, fixed-sample target 50 seeds.",
        "Looks: 3 blocks. Alpha spending: O'Brien-Fleming. Futility: non-binding. Hard cap: 60.",
    ]
)


def _planted() -> dict[str, tuple[dict[str, str], dict]]:
    """code -> (tree, check kwargs) for one input that must red with that code, in a temporary tree."""
    head = "# t -- pre-registration\n\n"
    return {
        "R-NO-SECTION": ({f"{_EVT}a.md": head + "seeds 8.\n"}, {}),
        "R-FIELD": ({f"{_EVT}a.md": head + _SECTION.replace("Futility", "Fut")}, {}),
        "R-STALE-GRANDFATHER": ({}, {"grandfathered": frozenset({f"{_EVT}a.md"})}),
    }


def self_check() -> int:
    """Plant each red in a temporary tree; fail unless the gate reds with exactly that code, and stays green
    on the clean tree and on a grandfathered file. A gate seen only green cannot be told from one that is blind."""
    clean = {f"{_EVT}a.md": "# t -- pre-registration\n\n" + _SECTION + "\n"}
    old = {f"{_EVT}a.md": "# t -- pre-registration\n\nseeds 8.\n"}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            ("clean", clean, {}),
            ("grandfathered", old, {"grandfathered": frozenset({f"{_EVT}a.md"})}),
            *((name, tree, kw) for name, (tree, kw) in _planted().items()),
        ]
        for name, tree, kwargs in cases:
            root = Path(tmp) / name
            for rel, text in tree.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text, encoding="utf-8")
            kw = {
                "grandfathered": frozenset(),
                "owes_at_freeze": frozenset(),
                "not_prereg": frozenset(),
            }
            codes = check(root, **{**kw, **kwargs}).codes()
            if name in ("clean", "grandfathered") and codes:
                failures.append(f"{name} tree reds: {sorted(codes)}")
            elif name not in ("clean", "grandfathered") and name not in codes:
                failures.append(f"planted {name} did not red (got {sorted(codes)})")
    for failure in failures:
        print(f"sample_size_gate self-check FAILED: {failure}")
    if not failures:
        print(
            "sample_size_gate self-check OK: every planted red fired, the clean tree stayed green"
        )
    return 1 if failures else 0


def main(argv: list[str] | None = None, root: Path = REPO) -> int:
    ap = argparse.ArgumentParser(description="P1 sample-size gate")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ap.add_argument("--self-check", dest="self_check", action="store_true")
    ap.add_argument(
        "--census", action="store_true", help="print the files that would red with no list at all"
    )
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()
    if args.census:
        for path, code in census(root):
            print(f'    "{path}",  # {code}')
        return 0
    report = check(root)
    print(render(report))
    if report.reds:
        print(f"sample_size_gate ASSERT FAILED: {len(report.reds)} red(s)")
        return 1
    print("sample_size_gate assert OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
