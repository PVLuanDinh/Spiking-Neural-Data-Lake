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

A pre-registration is a markdown file the repo marks as one by either of two reproducible rules (those of
specs/022-unified-brain-arc/evidence/p1-sample-size-grandfather.md, whose lists this gate's are built from):

    R1  a heading, anywhere in the repo (`*.md`, outside code fences), that names a pre-registration
    R2  an evidence file (`specs/**/evidence/*.md`) that says `pre-?regist` or `prereg` anywhere in its text

The gate checks R1 UNION R2. R2 reaches evidence files whose body, not heading, declares the pre-registration; R1
reaches files outside the evidence directories (a session record whose heading restates a design). Neither is a
subset of the other. Vendored and generated trees (dot-directories, node_modules, target, graphify-out) are not scanned.
Each pre-registration owes a heading named "Sample size (P1)" (any level, outside a code fence); the text of all such
sections of the file is read together and must carry each field by its word:

    measured SD  `SD` | `standard deviation`      effect    `effect` | `mde`        target  `target`
    looks        `look(s)`                          spending  `spending` | O'Brien | Pocock
    futility     `futility`                         cap       `cap`

A section with a one-shot line is a fixed count (the amendment's item 3): it owes the measured SD, the effect and the
target. The line must state a reason (six words or more) and must not be negated ("not a one-shot design"); a bare or
negated one-shot claim waives nothing and the sequential fields are owed.

What "frozen" means (the freeze marker rule): a file is frozen only when its OWN TEXT says so, with a line that
carries a freeze marker (`Freeze commit <sha>`, `Status: FROZEN`, `frozen at/in <sha>`). Never a commit subject: an
author controls a subject exactly as they control the word DRAFT, so this gate never reads one and a commit
subject saying "freeze" with no marker in the text grandfathers nothing and ends no deferral. Only a file frozen
before the landing, shown by execution (a dated read in its own text), is grandfathered, and only by name: a file
executed before the landing qualifies; a commit reference alone is not evidence of a freeze, and a commit that
froze it is history, not a freeze.

Three closed lists, all by file name (never by ancestry):
  GRANDFATHERED          frozen before the landing, shown by execution (a dated read in its own text); exempt whole, so a later edit to a grandfathered file is
                         not read by this gate (the list file says a section added later is not grandfathered;
                         that stays review-enforced). Evidence files only: every entry is a file the scan reads.
  OWES_AT_FREEZE         drafts that have not frozen (or not run) by the landing: name -> ISO expiry date. Deferred and printed on
                         every run until the expiry (R-DEFERRAL-EXPIRED) or until a line of the file says it froze
                         (R-FROZEN-DEFERRED); the author-maintained word DRAFT neither buys nor ends a deferral.
                         An expiry is a forcing date, renewable only by a dated integrator commit giving the reason
                         (each renewal an enforcer-reviewed build, D-74).
  NOT_PREREGISTRATIONS   files R1 or R2 matches that say of themselves they are not a pre-registration (references,
                         policy text, retroactive records); each carries its reason in a comment.
A name in any list whose file is gone (or, for GRANDFATHERED, is no longer an R1/R2 match) is R-STALE-GRANDFATHER,
so the lists only shrink.

The list file's section 4 names the files it keeps owing the section (integrator ruling 2026-10-24, #5196); the gate
reads it and reds R-LIST-CONFLICT when GRANDFATHERED names one, or names a file that sections 3 and 4b do not list,
so a lane adding a name to GRANDFATHERED fails unless the same commit edits the integrator-ruled list file, which
review sees (a missing list file is not red: a temporary test tree has none).
Files are the ones git tracks (`git ls-files`), so the verdict does not depend on untracked working-tree litter; a
tree that is not a git checkout (a temporary test tree) is walked instead. Every run prints the scan mode
("scan: git ls-files", "scan: walk", or "scan: walk, git failed: <err>"), so a fallback is never silent.

What it reds: R-NO-SECTION, R-FIELD, R-STALE-GRANDFATHER, R-DEFERRAL-EXPIRED, R-FROZEN-DEFERRED, R-LIST-CONFLICT.
What it cannot do, and says so on every run: judge whether a number is right, a simulation was run, or a
boundary is sound (the policy-enforcer and the panel do), or see who moved an OWES_AT_FREEZE date (it cannot see
who moved a date; review enforces that); it checks that each field is WRITTEN. render()'s LIMIT line states all of
these.

  python -m snn_data_lake.gates.sample_size_gate [--assert | --self-check | --census]

Bare invocation is --assert (the run_selfchecks.py convention).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from snn_data_lake.gates.docs_only_battery import _git
from snn_data_lake.gates.null_scale_gate import Red, _lines, _section, _structure

REPO = Path(__file__).resolve().parents[3]

# the landing commit is fb7b6581 (it added "### P1 amendment" to docs/POLICIES.md); GRANDFATHERED is by name, not ancestry
LIST_FILE = "specs/022-unified-brain-arc/evidence/p1-sample-size-grandfather.md"
SKIP_DIRS = frozenset({"node_modules", "target", "graphify-out", "__pycache__"})
PREREG_RE = re.compile(r"pre-?regist|prereg", re.I)
HEADING_RE = re.compile(
    r"pre-?registration|prereg", re.I
)  # R1's exact heading rule (the list file)
SECTION_RE = re.compile(r"sample size \(p1\)", re.I)
ONE_SHOT_RE = re.compile(r"one-shot", re.I)
NEGATED_RE = re.compile(
    r"\b(?:not|never|isn't|non|no)\b[\s\w,'-]{0,25}?one-shot"
    r"|one-shot\b[\s\w,'-]{0,15}?\b(?:not|never)\s+(?:a|one)\b",
    re.I,
)
ONE_SHOT_MIN_WORDS = 6
# a line that says a deferred file froze: its freeze commit, a FROZEN status, "frozen at/in <sha>"
FREEZE_RE = re.compile(
    r"freeze commit|status:?\W*frozen|\bfrozen\s+(?:at|in|by commit)\W{0,3}[0-9a-f]{7,}", re.I
)
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

# The pre-registrations frozen before the landing, shown by execution (a dated read in its own text), by name
# (71 evidence files): a commit reference alone is not evidence of a freeze. The 76 of
# p1-sample-size-grandfather.md at c9d0c097 less its three R1 reference files outside the evidence directories
# (NOT_PREREGISTRATIONS) and less g3 and g5 (OWES_AT_FREEZE: never run, or a post-landing Part E). g2 is here: the
# list calls it DRAFT, but it was executed before the landing (its section 5.1 is a confirmatory read dated
# 2026-10-07; 6bc97430, 2026-10-03, is history, not a freeze); list section 4 no longer keeps it owing.
# May shrink only by deleting an entry in the same commit as its file (R-STALE-GRANDFATHER). Regenerate the
# candidates with `--census`.
GRANDFATHERED: frozenset[str] = frozenset(
    {
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
                "g2-taught-problems",
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
# Drafts that have not frozen, name -> ISO date the deferral lapses (lane-set, integrator to ratify in the promotion
# commit: past it the file reds R-DEFERRAL-EXPIRED until it carries the section). A forcing date,
# renewable only by a dated integrator commit giving the reason (each renewal an enforcer-reviewed build, D-74);
# the gate cannot see who moved a date, review does. g3-growth-loop never ran (every
# amendment is "PRE-DATA", its A4 says the campaign instrument is ABSENT) and the list file's section 4 keeps it
# owing (integrator ruling, D-73): a commit subject saying "freeze" is not a freeze. g5-readout-variants was
# frozen before the landing, shown by execution (a dated read in its own text); Parts A-D ran
# 2026-10-07, but its Part E (drift-surviving relational readout) was added as a DRAFT on 2026-10-27, so the file as a whole is not grandfathered and Part E owes the section when it freezes.
OWES_AT_FREEZE: Mapping[str, str] = {
    _EV + "g3-growth-loop.md": "2026-11-05",
    _EV + "g5-readout-variants.md": "2026-11-05",
}
# Matched by R1 or R2, but say of themselves they are not a pre-registration. Reasons:
NOT_PREREGISTRATIONS: frozenset[str] = frozenset(
    {
        # the P1 list and two retroactive records
        _EV + "p1-sample-size-grandfather.md",
        _EV + "p5b-pfc-rng-record.md",
        _EV + "p5b-retroactive-record.md",
        # R1 headings that cite or define pre-registration, not register one (the old list's R1 reference files)
        "docs/AGENT_COLD_START.md",  # 8.8: what an invented method owes (policy text)
        "specs/011-h01-neurogenesis-prior/decisions.md",  # RE-DECISION citing T063's pre-registration 2.12
        "specs/022-unified-brain-arc/quickstart.md",  # S2 note citing a panel's pre-registration-honesty lens
        "research/cloud-replies/REPLY-2-FROM-CLOUD-SESSION-2026-10-28.md",  # item 4 names a branch "...-prereg-..."
        # a session record restating d1's design; the registration is d1-readout-staleness.md, which carries the section
        "research/representational-drift/SESSION-RECORD-2026-10-27.md",
    }
)


@dataclass
class Report:
    reds: list[Red] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    deferred: list[str] = field(default_factory=list)
    scan: str = ""

    def codes(self) -> set[str]:
        return {r.code for r in self.reds}


def _markdown(root: Path) -> tuple[list[str], str]:
    """(repo-relative `*.md` paths under root outside dot-directories and SKIP_DIRS, the scan mode): the files git
    tracks when root is a git checkout (untracked litter must not change the verdict), else a directory walk."""
    mode = "walk"
    if (root / ".git").exists():
        try:
            out = _git("ls-files", "-z", "--", "*.md", root=root)
        except (subprocess.CalledProcessError, OSError) as err:
            detail = (getattr(err, "stderr", None) or str(err)).strip().splitlines()
            mode = f"walk, git failed: {detail[-1] if detail else type(err).__name__}"
        else:
            kept = []
            for rel in sorted(out.split("\0")):
                parts = rel.split("/")
                skip = any(p.startswith(".") or p in SKIP_DIRS for p in parts[:-1])
                if rel and not skip and (root / rel).is_file():
                    kept.append(rel)
            return kept, "git ls-files"
    walked = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".md"):
                walked.append((Path(dirpath) / name).relative_to(root).as_posix())
    return walked, mode


def _rows_under(text: str, heads: tuple[str, ...], rows_only: bool) -> frozenset[str]:
    """The `*.md` paths of the list file between a heading that starts with one of heads and the next heading
    (`rows_only`: only table rows, so prose naming a file lists nothing)."""
    names, inside = set(), False
    for line in text.splitlines():
        if line.startswith(("# ", "## ", "### ")):
            inside = line.startswith(heads)
        elif inside and (not rows_only or line.startswith("|")):
            names.update(re.findall(r"`([^`]+\.md)`", line))
    return frozenset(names)


def _owed_by_list(text: str) -> frozenset[str]:
    """The files section 4 of the list file keeps owing the section (4b and later are not section 4's owing rows)."""
    return _rows_under(text, ("## 4.",), rows_only=False)


def _listed_by_list(text: str) -> frozenset[str]:
    """The files the list file grandfathers: the table rows of its section 3 and of section 4b."""
    return _rows_under(text, ("## 3.", "### 3", "### 4b."), rows_only=True)


def _preregistrations(root: Path, paths: list[str]) -> dict[str, list]:
    """rel -> its lines, for every file R1 or R2 marks as a pre-registration (before any list is applied)."""
    found = {}
    for rel in paths:
        lines = _lines(root, rel)
        if rel.startswith("specs/") and rel.split("/")[-2] == "evidence":  # R2: the word anywhere
            hit = any(PREREG_RE.search(ln.text) for ln in lines)
        else:  # R1: a heading that names one
            hit = any(HEADING_RE.search(lines[j].text) for j, _ in _structure(lines)[0])
        if hit:
            found[rel] = lines
    return found


def _one_shot(body: str) -> str:
    """'ok' when a line claims one-shot with a stated, un-negated reason; 'bad' when one-shot is named but no
    line does; 'none' when it is not named."""
    named = [ln for ln in body.splitlines() if ONE_SHOT_RE.search(ln)]
    if not named:
        return "none"
    ok = any(not NEGATED_RE.search(ln) and len(ln.split()) >= ONE_SHOT_MIN_WORDS for ln in named)
    return "ok" if ok else "bad"


def _problem(lines) -> Red | None:
    """None when the file carries a complete section; else the Red (without its path, set by the caller)."""
    heads = _structure(lines)[0]
    body = "\n".join(
        ln.text for j, _ in heads if SECTION_RE.search(lines[j].text) for ln in _section(lines, j)
    )
    if not body:
        return Red("R-NO-SECTION", "", 'no heading "Sample size (P1)" outside a code fence')
    shot = _one_shot(body)
    owed = ONE_SHOT_FIELDS if shot == "ok" else tuple(FIELDS)
    missing = [f for f in owed if not re.search(FIELDS[f], body, re.I)]
    if missing:
        note = (
            " (the one-shot claim has no un-negated one-shot reason, so it waives nothing)"
            if shot == "bad"
            else ""
        )
        return Red(
            "R-FIELD",
            "",
            f"the Sample size (P1) section does not write: {', '.join(missing)}{note}",
        )
    return None


def check(
    root: Path = REPO,
    grandfathered: frozenset[str] | None = None,
    owes_at_freeze: Mapping[str, str] | None = None,
    not_prereg: frozenset[str] | None = None,
    today: date | None = None,
) -> Report:
    grandfathered = GRANDFATHERED if grandfathered is None else grandfathered
    owes_at_freeze = OWES_AT_FREEZE if owes_at_freeze is None else owes_at_freeze
    not_prereg = NOT_PREREGISTRATIONS if not_prereg is None else not_prereg
    today = date.today() if today is None else today
    report = Report()
    paths, report.scan = _markdown(root)
    marked = _preregistrations(root, paths)
    found = [rel for rel in marked if rel not in not_prereg]
    owing, carry = set(), 0
    list_file = root / LIST_FILE
    if list_file.is_file():
        text = list_file.read_text(encoding="utf-8")
        owed = _owed_by_list(text) & grandfathered
        for rel in sorted(owed):
            msg = f"{LIST_FILE} section 4 keeps this file owing the section; it cannot also be grandfathered"
            report.reds.append(Red("R-LIST-CONFLICT", rel, msg))
        for rel in sorted(grandfathered - owed - _listed_by_list(text)):
            msg = f"{LIST_FILE} sections 3 and 4b do not list this file; a lane cannot grandfather it by editing the gate"
            report.reds.append(Red("R-LIST-CONFLICT", rel, msg))
    for rel in found:
        if rel in grandfathered:
            continue
        lines = marked[rel]
        problem = _problem(lines)
        carry += problem is None
        if rel in owes_at_freeze:
            expiry = date.fromisoformat(owes_at_freeze[rel])
            if any(FREEZE_RE.search(ln.text) for ln in lines):
                msg = "a line says this file froze: delete its OWES_AT_FREEZE entry and write the section"
                code = "R-FROZEN-DEFERRED"
            elif expiry < today:
                todo = "write the section" if problem else "delete the OWES_AT_FREEZE entry"
                msg = f"the deferral lapsed on {expiry}: {todo}, or the integrator re-dates it"
                code = "R-DEFERRAL-EXPIRED"
            else:
                if problem:
                    report.deferred.append(rel)
                continue
            owing.add(rel)
            report.reds.append(Red(code, rel, msg))
        elif problem:
            owing.add(rel)
            report.reds.append(Red(problem.code, rel, problem.message))
    gone = 0
    for names, in_scan in ((grandfathered, True), (owes_at_freeze, False), (not_prereg, False)):
        for rel in sorted(names):
            if (rel not in marked) if in_scan else not (root / rel).is_file():
                gone += in_scan
                msg = f"list entry {rel!r} matches no current pre-registration file"
                report.reds.append(Red("R-STALE-GRANDFATHER", rel, msg))
    report.counts = {
        "pre-registrations found": len(found),
        "grandfathered": len(grandfathered) - gone,
        "deferred (owes at freeze)": len(report.deferred),
        "declared not a pre-registration": len(not_prereg),
        "owing the section now": len(owing),
        "carry the section": carry,
    }
    return report


def census(root: Path = REPO) -> list[tuple[str, str]]:
    """(path, code) of every pre-registration that would red with no list at all: the candidates for
    GRANDFATHERED, OWES_AT_FREEZE and NOT_PREREGISTRATIONS. A human sorts them by their frozen date."""
    report = check(root, frozenset(), {}, frozenset())
    return [(r.where, r.code) for r in report.reds]


def render(report: Report) -> str:
    out = [
        "P1 SAMPLE-SIZE GATE -- every pre-registration derives its seed count (docs/POLICIES.md P1 amendment)"
    ]
    out.append(f"  scan: {report.scan}")
    out += [f"  {name}: {n}" for name, n in report.counts.items()]
    out += [f"  DEFERRED (owes the section at freeze): {rel}" for rel in report.deferred]
    out.append(
        "  LIMIT: the gate checks that each field is WRITTEN (by its word, over all of a file's Sample size (P1) "
        "sections together); it cannot judge that a number is right, that a simulation ran, or that a boundary "
        "is sound, it cannot see who moved an OWES_AT_FREEZE date, and a grandfathered file is exempt whole. "
        "Those stay review-enforced."
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
        "R-DEFERRAL-EXPIRED": (
            {f"{_EVT}a.md": head + "seeds 8.\n"},
            {"owes_at_freeze": {f"{_EVT}a.md": "2000-01-01"}},
        ),
        "R-FROZEN-DEFERRED": (
            {f"{_EVT}a.md": head + "DRAFT; the freeze commit 6bc97430.\n"},
            {"owes_at_freeze": {f"{_EVT}a.md": "2999-01-01"}},
        ),
        "R-LIST-CONFLICT section-4": (
            {
                LIST_FILE: f"## 3. Listed\n| 1 | `{_EVT}a.md` |\n## 4. Owes\n| 2 | `{_EVT}a.md` |\n",
                f"{_EVT}a.md": head + "seeds 8.\n",
            },
            {"grandfathered": frozenset({f"{_EVT}a.md"})},
        ),
        "R-LIST-CONFLICT unlisted": (
            {
                LIST_FILE: f"## 3. Listed\n| 1 | `{_EVT}other.md` |\n",
                f"{_EVT}a.md": head + "seeds 8.\n",
            },
            {"grandfathered": frozenset({f"{_EVT}a.md"})},
        ),
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
                "owes_at_freeze": {},
                "not_prereg": frozenset(),
            }
            codes = check(root, **{**kw, **kwargs}).codes()
            if name in ("clean", "grandfathered") and codes:
                failures.append(f"{name} tree reds: {sorted(codes)}")
            elif name not in ("clean", "grandfathered") and name.split()[0] not in codes:
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
