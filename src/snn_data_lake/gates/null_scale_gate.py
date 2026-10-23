"""P8 null-scale gate -- a null names the scale it was measured at, and deletion waits for a 10x re-test.

Pre-registration (frozen before this gate was built): specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md.
Rule enforced: docs/POLICIES.md P8 amendment (draft until the landing commit). Per the cross-cutting
invariant "build the gate and COUNT its output first", this module lands before the rule text does.

# distinct-from: docs_only_battery.py -- hashes `git ls-files -s` lines to fingerprint a docs-only tree; this
# reads evidence-file TEXT for NULL-RECORD / RETEST-RECORD lines and delete-on-null statements. Same word
# "gate", different corpus and job. gates/policy_manifest.py is the template for the CLI shape only.

Records (specs/**/evidence/*.md and docs/*.md; lines inside a fenced code block are examples, never records):

    NULL-RECORD: <method> | units=<int|n/a> | ticks=<int|n/a> | seeds=<int> | mde=<number> | retest=<R>
    R := REF:<repo path>#<method> | PENDING since <40-hex sha> | STRUCTURAL: <reason, >= 20 chars>
         | NOT-NEEDED: harm <reason, >= 20 chars>
    RETEST-RECORD: <method> | axis=<units|ticks> | <axis>=<int> | result=<null|charges> | of=<repo path>

What it reds: R-FIELD, R-UNRECORDED, R-RETEST-REF, R-RETEST-SCALE, R-RETEST-CHARGED, R-EXPIRED,
R-STALE-GRANDFATHER here; R-EXEMPT through `exempt_problems`, the ONE check `dormancy_gate` imports for its
"scale-retest" disposition.
What it cannot do, and says so on every run: judge a STRUCTURAL argument (the reason is printed for review),
or decide whether a method's null was measured on the RIGHT axis (the pre-registration author decides).

  python -m snn_data_lake.gates.null_scale_gate [--assert | --self-check | --census]

Bare invocation is --assert (the run_selfchecks.py convention).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[3]

EVIDENCE_GLOB = "specs/**/evidence/*.md"
DOCS_GLOB = "docs/*.md"
SCOPE = (EVIDENCE_GLOB, DOCS_GLOB)
PREREG = "specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md"
TRIAGE = "specs/022-unified-brain-arc/evidence/null-scale-triage.md"
# Addendum 3: both files quote or list delete-on-null wording without making a statement. R-UNRECORDED skips
# exactly these two BY NAME; R-FIELD and the record checks still read them. Printed on every --assert run.
UNRECORDED_EXCLUDED = (PREREG, TRIAGE)
DORMANCY_GATE = "src/snn_data_lake/gates/dormancy_gate.py"
CERT = "tools/policy_certification.json"  # the battery's execution record: a commit that modifies it closes an arc
# Owner answer 1, verbatim: "Scale factor: 10x minimum on the scaling axis."
SCALE_FACTOR = 10
AXES = ("units", "ticks")

STATEMENT_RE = re.compile(
    r"(delet(ed|ion|e|ing|es)|retir(ed|e|ing|es)|remov(ed|e|es|ing|al)) on (its|a|the|their) (own )?null",
    re.I,
)
RECORD_RE = re.compile(r"^\s*(?:[-*]\s+)?(NULL-RECORD|RETEST-RECORD):\s*(.*?)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s")
SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
QUOTE_RE = re.compile(r"^(?:> ?)+")
EMPHASIS_RE = re.compile(r"[*_`]")
CLOSING_RE = re.compile(
    r"\bclosing\b", re.I
)  # a closing record's subject names it; "disclosing" does not
_INT = r"[1-9][0-9]*"
_NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE]-?[0-9]+)?"
_SHA = r"[0-9a-f]{40}"
_HARM_KINDS = "harm|at-scale|restatement"

# One entry per delete-on-null statement that existed before the rule: (repo path, first 12 hex of the sha256 of
# the stripped line). A moved line keeps its key, a reworded one does not. The set may shrink only by deleting
# an entry in the same commit as its line (R-STALE-GRANDFATHER). Regenerate with `--census`.
GRANDFATHERED: frozenset[tuple[str, str]] = frozenset(
    {
        ("specs/022-unified-brain-arc/evidence/c0-everyday-battery.md", "2b7b457e999a"),
        ("specs/022-unified-brain-arc/evidence/g5-readout-variants.md", "972ae33ba3a3"),
        ("specs/022-unified-brain-arc/evidence/g5-readout-variants.md", "6a37684ba1a2"),
        ("specs/022-unified-brain-arc/evidence/g5-readout-variants.md", "f775c9b65d2e"),
        ("specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md", "5b5293c96aa6"),
        ("specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md", "c52cb3e92600"),
        ("specs/022-unified-brain-arc/evidence/insula-level-arms-production.md", "ab9c69434dc7"),
        ("specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md", "ea385a5d26d5"),
        ("specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md", "f3ffa203b1d7"),
        ("specs/022-unified-brain-arc/evidence/jepa-live-floor.md", "aec6060c978b"),
        ("specs/022-unified-brain-arc/evidence/l-exit.md", "c466ec73c3b6"),
        ("specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md", "c2d13723a832"),
        ("specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md", "8a7ac8989836"),
        ("specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md", "a6b0e7a61c88"),
        ("specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md", "9afba4db4f5a"),
        ("specs/022-unified-brain-arc/evidence/r1-bounded-recall.md", "9c1189e1fd1c"),
        ("specs/022-unified-brain-arc/evidence/r3-replay-growth-composition.md", "2d4c0d656797"),
        ("specs/022-unified-brain-arc/evidence/s0-exit.md", "accb2e772aa0"),
        ("specs/022-unified-brain-arc/evidence/s0-exit.md", "921dcde40836"),
        ("specs/022-unified-brain-arc/evidence/s3-neuron-pager.md", "ff26b6b3cdaf"),
        ("specs/022-unified-brain-arc/evidence/s4-lake-compactor.md", "5b5edb284961"),
        ("specs/022-unified-brain-arc/evidence/s4-lake-compactor.md", "0f636bff2344"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "01cb45021e8a"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "b9f3a73eb473"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "71dae1592a93"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "a9a3aa842cde"),
        ("specs/022-unified-brain-arc/evidence/sc-012a-width.md", "114c3fddb61a"),
        ("specs/022-unified-brain-arc/evidence/t069-cerebellar-rule.md", "948a3b350b92"),
        ("specs/022-unified-brain-arc/evidence/trainer-retirement.md", "8361c2027d5d"),
        ("specs/022-unified-brain-arc/evidence/trainer-retirement.md", "7c121e85fd83"),
        ("specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md", "58d3a54f653f"),
        ("specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md", "f1104c223cfa"),
        ("specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md", "b7994c2b2f50"),
    }
)


@dataclass(frozen=True)
class Line:
    path: str
    no: int
    text: str
    fenced: bool


@dataclass(frozen=True)
class NullRecord:
    path: str
    no: int
    method: str
    units: str
    ticks: str
    seeds: str
    mde: str
    retest: str


@dataclass(frozen=True)
class RetestRecord:
    path: str
    no: int
    method: str
    axis: str
    value: int
    result: str
    of: str


@dataclass(frozen=True)
class Red:
    code: str
    where: str
    message: str


@dataclass
class Report:
    reds: list[Red] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    structural: list[str] = field(default_factory=list)
    harm: list[str] = field(default_factory=list)

    def codes(self) -> set[str]:
        return {r.code for r in self.reds}


def line_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


class History(NamedTuple):
    """What `git_history` reads for one `since=<S>`. `closed`: the closing records in S..HEAD that a non-merge
    commit descends from. `nulls`: every NULL-RECORD line (path, line no, text) of the in-scope files in S's
    tree. `dormancy`: dormancy_gate.py as of S. `passed`, `passed_nulls`, `passed_dormancy`: the newest closing
    record HEAD is past (whatever S says) and the same two reads of its tree."""

    closed: list[str]
    nulls: tuple[tuple[str, int, str], ...] = ()
    dormancy: str = ""
    passed: str = ""
    passed_nulls: tuple[tuple[str, int, str], ...] = ()
    passed_dormancy: str = ""


def git_history(sha: str, repo: Path = REPO, head: str = "HEAD") -> History | None:
    """The History of `sha`, judged at `head`. None when the history cannot answer: unknown sha, `sha` not an
    ancestor of `head`, shallow clone, git missing. The one place git is read."""

    def git(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *argv], cwd=repo, capture_output=True, text=True, timeout=60)

    try:
        if git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}").returncode:
            return None
        if git("merge-base", "--is-ancestor", sha, head).returncode:
            return None
        log = git(
            "log",
            "--no-merges",
            "--full-history",
            "--format=%H%x1f%s",
            f"{sha}..{head}",
            "--",
            CERT,
        )

        # distinct-from: infra/region_wiring.py -- that derives same-tick connectome dependencies; these two
        # helpers read git history (a commit HEAD descends from; the NULL-RECORD lines and EXEMPT rows of a tree)
        def past(commit: str) -> bool | None:
            later = git("rev-list", "--ancestry-path", "--no-merges", f"{commit}..{head}")
            return None if later.returncode else bool(later.stdout.strip())

        def reads(commit: str):
            grep = git(
                "grep", "-F", "-n", "NULL-RECORD:", commit, "--", *(f":(glob){g}" for g in SCOPE)
            )
            if grep.returncode > 1:
                return None
            rows = (row.split(":", 3)[1:] for row in grep.stdout.splitlines())
            show = git("show", f"{commit}:{DORMANCY_GATE}")
            return tuple(
                (p, int(no), t) for p, no, t in rows
            ), show.stdout if not show.returncode else ""

        closed = []
        for row in log.stdout.splitlines():
            commit, _, subject = row.partition("\x1f")
            if CLOSING_RE.search(subject) and (is_past := past(commit)) is not False:
                if is_past is None:
                    return None
                closed.append(commit)
        # re-panel 2 #4627 M1: a record already PENDING at the newest closing record HEAD is past has ridden an
        # arc, whatever `since` now says (a since laundered through a commit that parked the record is expired)
        passed = ""
        everyone = git("log", "--no-merges", "--format=%H%x1f%s", head, "--", CERT)
        for row in everyone.stdout.splitlines():
            commit, _, subject = row.partition("\x1f")
            if CLOSING_RE.search(subject) and (is_past := past(commit)) is not False:
                if is_past is None:
                    return None
                passed = commit
                break
        at_since = reads(sha)
        at_passed = reads(passed) if passed else ((), "")
    except (OSError, subprocess.SubprocessError):
        return None
    if log.returncode or everyone.returncode or at_since is None or at_passed is None:
        return None
    return History(closed, *at_since, passed, *at_passed)


def _scale_retest_modules(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and isinstance(value, ast.Tuple)
        and value.elts
        and isinstance(value.elts[0], ast.Constant)
        and value.elts[0].value == "scale-retest"
    }


def _pending_methods(lines) -> set[str]:
    parsed = (
        (path, no, m.group(2))
        for path, no, text in lines
        if (m := RECORD_RE.match(text)) and m.group(1) == "NULL-RECORD"
    )
    return {
        n.method
        for path, no, body in parsed
        if isinstance(n := parse_null(path, no, body, triage=path == TRIAGE), NullRecord)
        and n.retest.startswith("PENDING since")
    }


def expiry_problem(
    sha: str, history=git_history, method: str | None = None, module: str | None = None
) -> str | None:
    """R-EXPIRED for a `PENDING since <sha>` entry of `method` (or a dormancy "scale-retest" row of `module`).
    It rides in the arc that created it, up to and including the arc's closing record; a closing record is a
    commit that modifies the execution record and says "closing", and it expires the entry only when a
    non-merge commit descends from it on the way to HEAD. A `since` whose own tree already holds the PENDING
    record (or the row) was moved forward past it; a record (or row) already PENDING at the newest closing record
    HEAD is past has ridden an arc, whatever `since` says. No creation-commit lookup: a moved or renamed record is
    read wherever it sits in scope (a record whose METHOD is respelled is a new record to these reads)."""
    found = history(sha)
    if found is None:
        return f"history unavailable for {sha} (unknown sha, not an ancestor of HEAD, or shallow clone)"
    if method in _pending_methods(found.nulls) or module in _scale_retest_modules(found.dormancy):
        return f"expired: {sha[:12]} postdates its own PENDING record (a forward edit)"
    if method in _pending_methods(found.passed_nulls) or module in _scale_retest_modules(
        found.passed_dormancy
    ):
        return f"expired: already PENDING at closing record {found.passed[:12]}, which HEAD is past"
    if found.closed:
        return f"expired: {sha[:12]} predates closing record(s) {[c[:12] for c in found.closed]} and HEAD is past them"
    return None


def exempt_problems(module: str, reason: str, history=git_history, root: Path = REPO) -> list[str]:
    """R-EXEMPT: what is wrong with a dormancy_gate EXEMPT entry whose disposition is "scale-retest"."""
    problems = []
    named = re.search(r"NULL-RECORD:([\w./-]+\.md)", reason)
    since = re.search(rf"since=({_SHA})(?![0-9a-f])", reason)
    pending = f"PENDING since {since.group(1)}" if since else "PENDING since"
    if not named:
        problems.append(f"{module}: reason names no `NULL-RECORD:<repo path>`")
    elif not any(n.retest == pending for n in _scan(root).nulls.get(named.group(1), [])):
        problems.append(
            f"{module}: {named.group(1)} is not an in-scope file holding a NULL-RECORD whose retest is `{pending}`"
        )
    if not since:
        problems.append(f"{module}: reason names no `since=<full 40-hex sha>`")
    elif (expired := expiry_problem(since.group(1), history, module=module)) is not None:
        problems.append(f"{module}: {expired}")
    return problems


def _keyed(parts: list[str], required: tuple[str, ...], optional: tuple[str, ...] = ()):
    got: dict[str, str] = {}
    for part in parts:
        key, eq, value = part.partition("=")
        key = key.strip()
        if not eq or key not in required + optional:
            return None, f"unknown key or malformed field {part!r}"
        if key in got:
            return None, f"duplicate key {key!r}"
        got[key] = value.strip()
    missing = [k for k in required if k not in got]
    return (None, f"missing field(s) {missing}") if missing else (got, None)


def _count_ok(value: str, extra: tuple[str, ...]) -> bool:
    return re.fullmatch(_INT, value) is not None or value in extra


def _retest_ok(value: str, triage: bool) -> bool:
    if re.fullmatch(r"REF:[^\s#]+#.+", value) or re.fullmatch(rf"PENDING since {_SHA}", value):
        return True
    # A reason may not carry the field separator: "STRUCTURAL: ... | note=x" would hide an unknown key (review #4483 m2).
    if value.startswith("STRUCTURAL:"):
        reason = value[len("STRUCTURAL:") :].strip()
        return len(reason) >= 20 and " | " not in reason
    if triage and value == "CANDIDATE":
        return True
    # Owner answer 3: "a measured harm is enough to delete; no scale re-test required" -- harm is allowed in
    # every in-scope file; at-scale and restatement stay triage-only.
    not_needed = re.fullmatch(rf"NOT-NEEDED: ({_HARM_KINDS}) (.+)", value)
    reason = not_needed.group(2).strip() if not_needed else ""
    allowed = bool(not_needed) and (triage or not_needed.group(1) == "harm")
    return allowed and len(reason) >= 20 and " | " not in reason


def parse_null(path: str, no: int, body: str, triage: bool = False) -> NullRecord | str:
    """A NullRecord, or the R-FIELD message. `triage` admits the one-time retroactive list's extra values
    (addenda 1 and 2): `unknown` for every scale field and mde, retest CANDIDATE / NOT-NEEDED."""
    parts = body.split(" | ", 5)
    if not parts[0].strip() or "=" in parts[0]:
        return "first field must be the method name"
    got, err = _keyed(parts[1:], ("units", "ticks", "seeds", "mde", "retest"))
    if err:
        return err
    unknown = ("unknown",) if triage else ()
    for key, extra in (
        ("units", ("n/a",) + unknown),
        ("ticks", ("n/a",) + unknown),
        ("seeds", unknown),
    ):
        if not _count_ok(got[key], extra):
            return f"{key}={got[key]!r} is not an integer >= 1" + (
                " or n/a" if "n/a" in extra else ""
            )
    # mde: the minimum detectable effect at one-sided alpha .05, power .8 (P8 amendment item 1; section 0 of
    # the pre-registration defines the field). The gate checks that it is a number, not how it was derived.
    if re.fullmatch(_NUM, got["mde"]) is None and got["mde"] not in unknown:
        return f"mde={got['mde']!r} is not a number"
    if not _retest_ok(got["retest"], triage):
        return (
            f"retest={got['retest']!r} is not REF:<path>#<method>, PENDING since <sha>, "
            "STRUCTURAL: <20+ chars>, or NOT-NEEDED: harm <20+ chars>"
        )
    return NullRecord(
        path,
        no,
        parts[0].strip(),
        got["units"],
        got["ticks"],
        got["seeds"],
        got["mde"],
        got["retest"],
    )


def parse_retest(path: str, no: int, body: str) -> RetestRecord | str:
    parts = body.split(" | ", 4)
    if not parts[0].strip() or "=" in parts[0]:
        return "first field must be the method name"
    got, err = _keyed(parts[1:], ("axis", "result", "of"), AXES)
    if err:
        return err
    axis = got["axis"]
    if axis not in AXES:
        return f"axis={axis!r} is not one of {AXES}"
    if axis not in got or set(got) - {"axis", "result", "of", axis}:
        return f"the scale field must be exactly `{axis}=<int>`"
    if re.fullmatch(_INT, got[axis]) is None:
        return f"{axis}={got[axis]!r} is not an integer >= 1"
    if got["result"] not in ("null", "charges"):
        return f"result={got['result']!r} is not null or charges"
    if not got["of"] or " " in got["of"]:
        return "of= must be one repo path"
    return RetestRecord(path, no, parts[0].strip(), axis, int(got[axis]), got["result"], got["of"])


def _rels(root: Path, *patterns: str) -> list[str]:
    return sorted(
        {p.relative_to(root).as_posix() for pat in patterns for p in root.glob(pat) if p.is_file()}
    )


def _lines(root: Path, rel: str) -> list[Line]:
    out, fenced = [], False
    for no, text in enumerate((root / rel).read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_RE.match(text):
            fenced = not fenced
            out.append(Line(rel, no, text, True))
        else:
            out.append(Line(rel, no, text, fenced))
    return out


def _structure(lines: list[Line]) -> tuple[list[tuple[int, int]], set[int]]:
    """(start index, level) of every heading outside a fence, and the indexes of the lines that make them up.
    A paragraph underlined by === or --- is a setext heading (level 1 or 2); the same underline after a blank
    line is a thematic break, not a heading."""
    heads: list[tuple[int, int]] = []
    owned: set[int] = set()
    run = None
    for j, ln in enumerate(lines):
        plain = bool(ln.text.strip()) and not ln.fenced
        if plain and (m := HEADING_RE.match(ln.text)):
            heads.append((j, len(m.group(1))))
            owned.add(j)
            run = None
        elif plain and run is not None and (u := SETEXT_RE.match(ln.text)):
            heads.append((run, 1 if u.group(1)[0] == "=" else 2))
            owned.update(range(run, j + 1))
            run = None
        else:
            run = (j if run is None else run) if plain else None
    return heads, owned


def _section(lines: list[Line], i: int) -> list[Line]:
    """Lines from the nearest heading at or above `i` to the next heading of the same or higher level."""
    heads = _structure(lines)[0]
    above = [(j, lvl) for j, lvl in heads if j <= i]
    start, level = above[-1] if above else (0, 7)
    end = next((j for j, lvl in heads if j > i and lvl <= level), len(lines))
    return lines[start:end]


@dataclass
class _Scan:
    root: Path
    evidence: list[str]
    lines: dict[str, list[Line]] = field(default_factory=dict)
    nulls: dict[str, list[NullRecord]] = field(default_factory=dict)
    retests: dict[str, list[RetestRecord]] = field(default_factory=dict)
    reds: list[Red] = field(default_factory=list)


def _scan(root: Path) -> _Scan:
    evidence = _rels(root, EVIDENCE_GLOB)
    scan = _Scan(root, evidence)
    for rel in sorted(set(evidence) | set(_rels(root, DOCS_GLOB))):
        scan.lines[rel] = lines = _lines(root, rel)
        for ln in lines:
            m = None if ln.fenced else RECORD_RE.match(ln.text)
            if not m:
                continue
            kind, body = m.groups()
            if kind == "NULL-RECORD":
                rec = parse_null(rel, ln.no, body, triage=rel == TRIAGE)
            else:
                rec = parse_retest(rel, ln.no, body)
            if isinstance(rec, str):
                scan.reds.append(Red("R-FIELD", f"{rel}:{ln.no}", f"{kind}: {rec}"))
            elif isinstance(rec, NullRecord):
                scan.nulls.setdefault(rel, []).append(rec)
            else:
                scan.retests.setdefault(rel, []).append(rec)
    return scan


def _paragraphs(lines: list[Line]) -> list[Line]:
    """Each paragraph (consecutive non-blank lines that are not headings or fence lines) joined with single
    spaces, as one Line at the paragraph's first line: a statement wrapped across lines is still one statement.
    A block quote's leading `>` markers are not part of the paragraph.
    Heading and fenced lines stay one Line each, as the line census read them (a heading can record a deletion).
    ponytail: a heading or fenced statement wrapped across lines is still missed, as it was before the join."""
    out: list[Line] = []
    run: list[Line] = []
    owned = _structure(lines)[1]
    for j, ln in enumerate([*lines, None]):
        if ln is not None and ln.text.strip() and not ln.fenced and j not in owned:
            run.append(ln)
            continue
        if run:
            joined = " ".join(QUOTE_RE.sub("", r.text.strip()) for r in run)
            out.append(Line(run[0].path, run[0].no, joined, False))
            run = []
        if ln is not None and ln.text.strip():
            out.append(ln)
    return out


def statements(scan: _Scan) -> list[Line]:
    return [
        para
        for rel in scan.evidence
        for para in _paragraphs(scan.lines[rel])
        if STATEMENT_RE.search(EMPHASIS_RE.sub("", para.text))
    ]


def _check_unrecorded(scan: _Scan, grandfathered) -> list[Red]:
    reds = []
    used = set()  # one grandfather key exempts ONE statement: a verbatim copy elsewhere is checked as new (review #4483/#4484)
    for st in statements(scan):
        if st.path in UNRECORDED_EXCLUDED:
            continue
        key = (st.path, line_key(st.text))
        if key in grandfathered and key not in used:
            used.add(key)
            continue
        lines = scan.lines[st.path]
        section = _section(lines, st.no - 1)
        if not any(
            not ln.fenced and (m := RECORD_RE.match(ln.text)) and m.group(1) == "NULL-RECORD"
            for ln in section
        ):
            reds.append(
                Red(
                    "R-UNRECORDED",
                    f"{st.path}:{st.no}",
                    "delete-on-null statement with no NULL-RECORD in its section",
                )
            )
    return reds


def _check_charged(scan: _Scan) -> list[Red]:
    """R-RETEST-CHARGED: a statement whose section holds a NULL-RECORD that ANY in-scope RETEST-RECORD charged,
    whatever the null's own retest says. Grandfathered statements are read too: a charge is a measurement,
    not a dated omission."""
    reds = []
    for st in statements(scan):
        if st.path in UNRECORDED_EXCLUDED:
            continue
        section = {ln.no for ln in _section(scan.lines[st.path], st.no - 1)}
        for n in (n for n in scan.nulls.get(st.path, []) if n.no in section):
            if any(
                r.method == n.method and r.of == n.path and r.result == "charges"
                for recs in scan.retests.values()
                for r in recs
            ):
                reds.append(
                    Red(
                        "R-RETEST-CHARGED",
                        f"{st.path}:{st.no}",
                        "re-test charged: a scale-dependent finding; deletion not licensed",
                    )
                )
                break
    return reds


def _check_stale(scan: _Scan, grandfathered) -> list[Red]:
    live = {(st.path, line_key(st.text)) for st in statements(scan)}
    return [
        Red("R-STALE-GRANDFATHER", f"{path}", f"GRANDFATHERED entry {key} matches no current line")
        for path, key in sorted(grandfathered - live)
    ]


def _check_retest_ref(scan: _Scan) -> list[Red]:
    reds = []
    for recs in scan.nulls.values():
        for n in recs:
            ref = re.fullmatch(r"REF:([^\s#]+)#(.+)", n.retest)
            if not ref:
                continue
            target, method = ref.groups()
            if method != n.method:
                reds.append(
                    Red(
                        "R-RETEST-REF",
                        f"{n.path}:{n.no}",
                        f"REF names method {method!r}, record is {n.method!r}",
                    )
                )
            elif not any(
                r.method == n.method and r.of == n.path for r in scan.retests.get(target, [])
            ):
                reds.append(
                    Red(
                        "R-RETEST-REF",
                        f"{n.path}:{n.no}",
                        f"{target} holds no RETEST-RECORD for {n.method!r} with of={n.path}",
                    )
                )
    for recs in scan.retests.values():
        for r in recs:
            if not any(n.method == r.method for n in scan.nulls.get(r.of, [])):
                reds.append(
                    Red(
                        "R-RETEST-REF",
                        f"{r.path}:{r.no}",
                        f"of={r.of} holds no NULL-RECORD for {r.method!r}",
                    )
                )
    return reds


def _check_retest_scale(scan: _Scan) -> list[Red]:
    reds = []
    for recs in scan.retests.values():
        for r in recs:
            for n in (n for n in scan.nulls.get(r.of, []) if n.method == r.method):
                original = getattr(n, r.axis)
                where = f"{r.path}:{r.no}"
                if original == "n/a":
                    reds.append(
                        Red(
                            "R-RETEST-SCALE",
                            where,
                            f"original has {r.axis}=n/a, so it cannot be the re-test axis",
                        )
                    )
                elif not original.isdigit():
                    reds.append(
                        Red(
                            "R-RETEST-SCALE",
                            where,
                            f"original {r.axis}={original!r} gives no scale to compare",
                        )
                    )
                elif r.value < SCALE_FACTOR * int(original):
                    reds.append(
                        Red(
                            "R-RETEST-SCALE",
                            where,
                            f"{r.axis}={r.value} is under {SCALE_FACTOR}x the original {original}",
                        )
                    )
    return reds


def _check_expired(scan: _Scan, history) -> list[Red]:
    reds = []
    for recs in scan.nulls.values():
        for n in recs:
            pending = re.fullmatch(rf"PENDING since ({_SHA})", n.retest)
            if pending and (problem := expiry_problem(pending.group(1), history, n.method)):
                reds.append(Red("R-EXPIRED", f"{n.path}:{n.no}", problem))
    return reds


def _printed(scan: _Scan, prefix: str) -> list[str]:
    return [
        f"{n.path}:{n.no} {n.method}: {n.retest}"
        for recs in scan.nulls.values()
        for n in recs
        if n.retest.startswith(prefix)
    ]


def check(root: Path = REPO, grandfathered=GRANDFATHERED, history=git_history) -> Report:
    scan = _scan(root)
    reds = [
        *scan.reds,
        *_check_unrecorded(scan, grandfathered),
        *_check_charged(scan),
        *_check_retest_ref(scan),
        *_check_retest_scale(scan),
        *_check_expired(scan, history),
        *_check_stale(scan, grandfathered),
    ]
    found = statements(scan)
    return Report(
        reds=reds,
        counts={
            "files scanned": len(scan.lines),
            "statements": len(found),
            "grandfathered": len({(s.path, line_key(s.text)) for s in found} & set(grandfathered)),
            "NULL-RECORDs": sum(map(len, scan.nulls.values())),
            "RETEST-RECORDs": sum(map(len, scan.retests.values())),
        },
        structural=_printed(scan, "STRUCTURAL:"),
        harm=_printed(scan, "NOT-NEEDED: harm"),
    )


def census(root: Path = REPO) -> list[tuple[str, int, str]]:
    scan = _scan(root)
    return [(st.path, st.no, line_key(st.text)) for st in statements(scan)]


def render(report: Report) -> str:
    out = [
        "P8 NULL-SCALE GATE -- every null names its scale; deletion on a null waits for a re-test"
    ]
    out += [f"  {name}: {n}" for name, n in report.counts.items()]
    out.append(f"  R-UNRECORDED skips by name: {', '.join(UNRECORDED_EXCLUDED)}")
    out += [f"  STRUCTURAL (printed for review): {s}" for s in report.structural]
    out += [f"  HARM (printed for review): {s}" for s in report.harm]
    out.append(
        "  LIMIT: the gate cannot judge a STRUCTURAL argument, whether a NOT-NEEDED: harm claim passed its "
        "pre-registered bar, or whether a re-test used the right axis; all three stay review-enforced. "
        "Scoping is spec-literal: a section runs from its heading (# or setext "
        "=== / ---) to the next heading of the same or higher level, so one NULL-RECORD of any method "
        "covers every statement in its section, a record under a sub-heading included."
    )
    out += [f"  RED {r.code} {r.where}: {r.message}" for r in report.reds]
    return "\n".join(out)


_REC = "NULL-RECORD: m | units=4 | ticks=600 | seeds=8 | mde=0.5 | retest="
_STRUCT = _REC + "STRUCTURAL: zero call sites, never executed"
_EV = "specs/s/evidence/"


def _planted() -> dict[str, tuple[dict[str, str], dict]]:
    """code -> (tree, check kwargs) for one input that must red with that code, in a temporary tree."""
    ref_b = f"{_EV}b.md"
    return {
        "R-FIELD": ({f"{_EV}a.md": "# t\nNULL-RECORD: m | units=4\n"}, {}),
        "R-UNRECORDED": ({f"{_EV}a.md": "# t\nThe method was deleted on its own null.\n"}, {}),
        "R-RETEST-REF": ({f"{_EV}a.md": f"# t\n{_REC}REF:{ref_b}#m\n", ref_b: "# r\n"}, {}),
        "R-RETEST-CHARGED": (
            {
                f"{_EV}a.md": f"# t\nThe method was deleted on its own null.\n\n{_REC}REF:{ref_b}#m\n",
                ref_b: f"# r\nRETEST-RECORD: m | axis=units | units=40 | result=charges | of={_EV}a.md\n",
            },
            {},
        ),
        "R-RETEST-SCALE": (
            {
                f"{_EV}a.md": f"# t\n{_REC}REF:{ref_b}#m\n",
                ref_b: f"# r\nRETEST-RECORD: m | axis=units | units=39 | result=null | of={_EV}a.md\n",
            },
            {},
        ),
        "R-EXPIRED": (
            {f"{_EV}a.md": f"# t\n{_REC}PENDING since {'a' * 40}\n"},
            {"history": lambda sha: None},
        ),
        "R-STALE-GRANDFATHER": (
            {f"{_EV}a.md": "# t\n"},
            {"grandfathered": frozenset({(f"{_EV}a.md", "0" * 12)})},
        ),
    }


def self_check() -> int:
    """Plant each red in a temporary tree; fail unless the gate reds with exactly that code, and stays green
    on the clean tree. A gate seen only green is indistinguishable from one that cannot see."""
    clean = {f"{_EV}a.md": f"# t\nThe method was deleted on its own null.\n{_STRUCT}\n"}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (tree, kwargs) in [
            ("clean", (clean, {"grandfathered": frozenset()})),
            *_planted().items(),
        ]:
            root = Path(tmp) / name
            for rel, text in tree.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text, encoding="utf-8")
            codes = check(root, **{"grandfathered": frozenset(), **kwargs}).codes()
            if name == "clean" and codes:
                failures.append(f"clean tree reds: {sorted(codes)}")
            elif name != "clean" and name not in codes:
                failures.append(f"planted {name} did not red (got {sorted(codes)})")
    if not exempt_problems("m", "no record, no sha"):
        failures.append("planted R-EXEMPT did not red")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / _EV).mkdir(parents=True)
        (Path(tmp) / _EV / "a.md").write_text(
            f"# t\n{_REC}PENDING since {'a' * 40}\n", encoding="utf-8"
        )
        entry = f"NULL-RECORD:{_EV}a.md since={'a' * 40}"
        history = lambda sha: History([])  # noqa: E731
        if exempt_problems("m", entry, history=history, root=Path(tmp)):
            failures.append("clean R-EXEMPT entry reds")
        if not exempt_problems(
            "m", entry.replace("a.md", "gone.md", 1), history=history, root=Path(tmp)
        ):
            failures.append("planted R-EXEMPT phantom NULL-RECORD path did not red")
    for failure in failures:
        print(f"null_scale_gate self-check FAILED: {failure}")
    if not failures:
        print("null_scale_gate self-check OK: every planted red fired, the clean tree stayed green")
    return 1 if failures else 0


def main(argv: list[str] | None = None, root: Path = REPO) -> int:
    ap = argparse.ArgumentParser(description="P8 null-scale gate")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ap.add_argument("--self-check", dest="self_check", action="store_true")
    ap.add_argument(
        "--census", action="store_true", help="print the GRANDFATHERED entries for this tree"
    )
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()
    if args.census:
        for path, no, key in census(root):
            print(f'    ("{path}", "{key}"),  # line {no}')
        return 0
    report = check(root)
    print(render(report))
    if report.reds:
        print(f"null_scale_gate ASSERT FAILED: {len(report.reds)} red(s)")
        return 1
    print("null_scale_gate assert OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
