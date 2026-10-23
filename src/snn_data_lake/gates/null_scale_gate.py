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
    RETEST-RECORD: <method> | axis=<units|ticks|seeds> | <axis>=<int> | result=<null|charges> | of=<repo path>

What it reds: R-FIELD, R-UNRECORDED, R-RETEST-REF, R-RETEST-SCALE, R-EXPIRED, R-STALE-GRANDFATHER here;
R-EXEMPT through `exempt_problems`, the ONE check `dormancy_gate` imports for its "scale-retest" disposition.
What it cannot do, and says so on every run: judge a STRUCTURAL argument (the reason is printed for review),
or decide whether a method's null was measured on the RIGHT axis (the pre-registration author decides).

  python -m snn_data_lake.gates.null_scale_gate [--assert | --self-check | --census]

Bare invocation is --assert (the run_selfchecks.py convention).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

EVIDENCE_GLOB = "specs/**/evidence/*.md"
DOCS_GLOB = "docs/*.md"
PREREG = "specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md"
TRIAGE = "specs/022-unified-brain-arc/evidence/null-scale-triage.md"
# Addendum 3: both files quote or list delete-on-null wording without making a statement. R-UNRECORDED skips
# exactly these two BY NAME; R-FIELD and the record checks still read them. Printed on every --assert run.
UNRECORDED_EXCLUDED = (PREREG, TRIAGE)
CLOSING_SUBJECT = "chore(record): closing execution record"
SCALE_FACTOR = 10
AXES = ("units", "ticks", "seeds")

STATEMENT_RE = re.compile(
    r"(delet(ed|ion|e)|retir(ed|e)|removed) on (its|a|the|their) (own )?null", re.I
)
RECORD_RE = re.compile(r"^\s*(?:[-*]\s+)?(NULL-RECORD|RETEST-RECORD):\s*(.*?)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
_INT = r"[1-9][0-9]*"
_NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE]-?[0-9]+)?"
_SHA = r"[0-9a-f]{40}"
_HARM_KINDS = "harm|at-scale|restatement"

# One entry per delete-on-null statement that existed before the rule: (repo path, first 12 hex of the sha256 of
# the stripped line). A moved line keeps its key, a reworded one does not. The set may shrink only by deleting
# an entry in the same commit as its line (R-STALE-GRANDFATHER). Regenerate with `--census`.
GRANDFATHERED: frozenset[tuple[str, str]] = frozenset(
    {
        ("specs/022-unified-brain-arc/evidence/c0-everyday-battery.md", "f823f1fef028"),
        ("specs/022-unified-brain-arc/evidence/g5-readout-variants.md", "8b94a63cd542"),
        ("specs/022-unified-brain-arc/evidence/g5-readout-variants.md", "4fe8589dc323"),
        ("specs/022-unified-brain-arc/evidence/g5-readout-variants.md", "3057ca1110fe"),
        ("specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md", "b2b6465f1f03"),
        ("specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md", "622b1d1a619a"),
        ("specs/022-unified-brain-arc/evidence/insula-level-arms-production.md", "2fbda974ff4c"),
        ("specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md", "b6838817cbf7"),
        ("specs/022-unified-brain-arc/evidence/insula-p8-engine-reach.md", "fd339a7f9e38"),
        ("specs/022-unified-brain-arc/evidence/jepa-live-floor.md", "ceb93e50d738"),
        ("specs/022-unified-brain-arc/evidence/l-exit.md", "6a99ed3ae91e"),
        ("specs/022-unified-brain-arc/evidence/l-exit.md", "7e97ad3d8373"),
        ("specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md", "3a870391338a"),
        ("specs/022-unified-brain-arc/evidence/l1b-synaptic-online.md", "8a7ac8989836"),
        ("specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md", "9202bf9204a9"),
        ("specs/022-unified-brain-arc/evidence/l2-grow-on-plateau.md", "719e8f98fb86"),
        ("specs/022-unified-brain-arc/evidence/r1-bounded-recall.md", "9297ddb1d509"),
        ("specs/022-unified-brain-arc/evidence/r3-replay-growth-composition.md", "0edb6ea1a6fe"),
        ("specs/022-unified-brain-arc/evidence/s0-exit.md", "d4914f4163e5"),
        ("specs/022-unified-brain-arc/evidence/s0-exit.md", "c326a826a698"),
        ("specs/022-unified-brain-arc/evidence/s3-neuron-pager.md", "6ef10f902e5c"),
        ("specs/022-unified-brain-arc/evidence/s4-lake-compactor.md", "0cc96a9e8077"),
        ("specs/022-unified-brain-arc/evidence/s4-lake-compactor.md", "b198632d46ad"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "a1ad953913a7"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "a977fd8e7138"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "8de96c3725db"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "bf906c5f281f"),
        ("specs/022-unified-brain-arc/evidence/s5-paging-null.md", "c95b15ab106e"),
        ("specs/022-unified-brain-arc/evidence/sc-012a-width.md", "b4005869fac4"),
        ("specs/022-unified-brain-arc/evidence/t069-cerebellar-rule.md", "948a3b350b92"),
        ("specs/022-unified-brain-arc/evidence/trainer-retirement.md", "7edd5f76fa51"),
        ("specs/022-unified-brain-arc/evidence/trainer-retirement.md", "2d523a869ab2"),
        ("specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md", "d7287e22638d"),
        ("specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md", "3c47b4420774"),
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

    def codes(self) -> set[str]:
        return {r.code for r in self.reds}


def line_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


def git_history(sha: str, repo: Path = REPO) -> list[tuple[str, str]] | None:
    """(sha, subject) of each commit in `sha..HEAD`, oldest first, HEAD last. None when the history cannot
    answer: unknown sha, `sha` not an ancestor of HEAD, shallow clone, git missing. The one place git is read."""

    def git(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *argv], cwd=repo, capture_output=True, text=True, timeout=60)

    try:
        if git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}").returncode:
            return None
        if git("merge-base", "--is-ancestor", sha, "HEAD").returncode:
            return None
        log = git("log", "--reverse", "--format=%H%x1f%s", f"{sha}..HEAD")
    except (OSError, subprocess.SubprocessError):
        return None
    if log.returncode:
        return None
    return [tuple(row.split("\x1f", 1)) for row in log.stdout.splitlines() if row]


def expiry_problem(sha: str, history=git_history) -> str | None:
    """R-EXPIRED: a PENDING entry rides in the arc that created it, up to and including that arc's closing
    record; any commit after a closing record that is not HEAD means the next arc is being built."""
    commits = history(sha)
    if commits is None:
        return f"history unavailable for {sha} (unknown sha, not an ancestor of HEAD, or shallow clone)"
    closing = [c[:12] for c, subject in commits[:-1] if subject.startswith(CLOSING_SUBJECT)]
    if closing:
        return f"expired: {sha[:12]} predates closing record(s) {closing} and HEAD is past them"
    return None


def exempt_problems(module: str, reason: str, history=git_history) -> list[str]:
    """R-EXEMPT: what is wrong with a dormancy_gate EXEMPT entry whose disposition is "scale-retest"."""
    problems = []
    if not re.search(r"NULL-RECORD:[\w./-]+\.md", reason):
        problems.append(f"{module}: reason names no `NULL-RECORD:<repo path>`")
    since = re.search(rf"since=({_SHA})(?![0-9a-f])", reason)
    if not since:
        problems.append(f"{module}: reason names no `since=<full 40-hex sha>`")
    elif (expired := expiry_problem(since.group(1), history)) is not None:
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
    not_needed = re.fullmatch(rf"NOT-NEEDED: ({_HARM_KINDS}) (.+)", value)
    reason = not_needed.group(2).strip() if not_needed else ""
    return bool(triage and not_needed and len(reason) >= 20 and " | " not in reason)


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
    if re.fullmatch(_NUM, got["mde"]) is None and got["mde"] not in unknown:
        return f"mde={got['mde']!r} is not a number"
    if not _retest_ok(got["retest"], triage):
        return f"retest={got['retest']!r} is not REF:<path>#<method>, PENDING since <sha>, or STRUCTURAL: <20+ chars>"
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


def _section(lines: list[Line], i: int) -> list[Line]:
    """Lines from the nearest heading at or above `i` to the next heading of the same or higher level."""
    heads = [
        (j, len(m.group(1)))
        for j, ln in enumerate(lines)
        if not ln.fenced and (m := HEADING_RE.match(ln.text))
    ]
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


def statements(scan: _Scan) -> list[Line]:
    return [ln for rel in scan.evidence for ln in scan.lines[rel] if STATEMENT_RE.search(ln.text)]


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
            if pending and (problem := expiry_problem(pending.group(1), history)) is not None:
                reds.append(Red("R-EXPIRED", f"{n.path}:{n.no}", problem))
    return reds


def check(root: Path = REPO, grandfathered=GRANDFATHERED, history=git_history) -> Report:
    scan = _scan(root)
    reds = [
        *scan.reds,
        *_check_unrecorded(scan, grandfathered),
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
        structural=[
            f"{n.path}:{n.no} {n.method}: {n.retest}"
            for recs in scan.nulls.values()
            for n in recs
            if n.retest.startswith("STRUCTURAL:")
        ],
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
    out.append(
        "  LIMIT: the gate cannot judge a STRUCTURAL argument or whether a re-test used the right axis; "
        "both stay review-enforced."
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
    if exempt_problems(
        "m", f"NULL-RECORD:{_EV}a.md since={'a' * 40}", history=lambda sha: [(sha, "x")]
    ):
        failures.append("clean R-EXEMPT entry reds")
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
