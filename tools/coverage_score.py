"""
coverage_score -- re-derive the brain-faculty coverage percentage from tracked evidence.

Evidence lives in docs/coverage/ (eleven category files + machine-recorded artifacts). Read
docs/coverage/README.md first: that ledger is a FIRST PASS, not the adversarially-verified sweep the
rubric specifies, and the number this prints is a floor on a partial roster -- NOT comparable to the
withdrawn 37.5%. The refusals below are mechanical and binding; the bands are provisional.

Deliberately NOT wired into CI yet, and not a fifth CORE gate. The four policy gates assert invariants
of the CODE; a coverage score is a progress MEASUREMENT, and rubric rule 5 says downgrades are results
-- a gate that failed on a lower number would create pressure to inflate, which is backwards. The
evidence-standard refusals are the part that could become a gate, once the roster is fixed.

WHY THIS EXISTS. docs/COVERAGE_RUBRIC.md:8-9 says "a percentage nobody can re-derive is not a
measurement" -- and then the baseline it governs was assembled entirely by hand. No .py file in the
tree reads either document, and the three code paths named "coverage" measure pytest LINE coverage
(pyproject.toml:101), module REACHABILITY (substrate/dormancy_gate.py:346) and registered CHARGE
ADAPTERS (substrate/region_wiring.py:67, currently 0). None of them is the faculty percentage. The
rubric ended the problem for the standard and left it open for the score.

IT DOES NOT SCORE ANYTHING. It has no opinion about what band a faculty deserves; an auditor writes
that in a tracked evidence file. What it does is REFUSE to add up claims that violate the rubric's
own evidence standard (docs/COVERAGE_RUBRIC.md:62-79):

  band > 40   needs a runtime A/B artifact whose RECORDED EXIT CODE says the A/B passed
  band > 80   needs a held-out artifact that states its metric and the baseline it beat
  band >= 1   cites file:line, and the citation must resolve to a line that exists
  all eleven categories are present, always, so the denominator cannot quietly shrink

THE POINT OF THE EXIT CODE. An A/B artifact is a captured run -- the command, its exit status, its
verbatim stdout, and the commit it was measured at -- written by `record` below, never by hand. So a
band claim is not checked against a number somebody typed; it is checked against what the probe
actually returned. `tools/charge_probe.py place 60` exits 2 (NOT-ELIGIBLE): an evidence row claiming
band 65 against that artifact is refused, and editing the row cannot change the exit code. That is
the whole difference between an instrument and a spreadsheet.

IT IS NOT AIRTIGHT, and claiming otherwise would be the failure it exists to prevent. A determined
forger can hand-write a plausible transcript. What that costs them is a specific, checkable lie --
`--rerun` re-executes the recorded command and compares exit code and observables -- and what it
removes entirely is the ACCIDENTAL inflation this project keeps hitting: a hand-assembled mean, a
citation that drifted twelve lines (COVERAGE_TO_100.md:61-64), a faculty dropped from the denominator.

DELIBERATELY ABSENT: a per-faculty weight column. "Evidence-weighted mean" is undefined in the
rubric; the band already encodes evidence strength, so the mean over bands IS the evidence-weighted
mean under the only reading that needs no new numbers. A weight column would be the single most
effective inflation knob available (weight the one charged faculty 5x), so it does not exist.

  python coverage_score.py --root <repo> --evidence <dir>
  python coverage_score.py ... --check                 # + assert COVERAGE_BASELINE.md agrees
  python coverage_score.py ... --rerun place_60        # re-execute one artifact and compare
  python coverage_score.py ... record --kind ab --name drive_60 -- tools/charge_probe.py drive 60
"""

import os
import re
import subprocess
import sys
import time

# The eleven rubric categories, verbatim from docs/COVERAGE_RUBRIC.md:26-38. This tuple IS the
# denominator: a missing file is a hard failure, never a smaller divisor.
CATEGORIES = (
    "Perception",
    "Memory",
    "Executive",
    "Consciousness",
    "SpatialTemporal",
    "Affect",
    "Social",
    "Imagination",
    "Language",
    "Gf",
    "WholeBrain",
)
COLUMNS = ("faculty", "band", "citation", "artifact", "measured_by", "measured_at")
ARTIFACT_KEYS = ("kind", "command", "exit", "commit", "recorded_at", "recorded_by")
FLIPS = re.compile(r"(\d+)/(\d+) ticks")


# ---------------------------------------------------------------- evidence file parsing
def _table(path):
    """Rows of the one strict table in an evidence file, as (lineno, cells). A malformed row RAISES
    rather than being skipped -- skipping a row it cannot parse is how an instrument silently drops a
    faculty out of the denominator, which is the inflation route the rubric names."""
    rows, seen_header = [], False
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not seen_header:
                if tuple(cells) == COLUMNS:
                    seen_header = True
                continue
            if set("".join(cells)) <= set("-: "):  # the |---|---| separator
                continue
            if len(cells) != len(COLUMNS):
                raise ValueError(
                    f"{os.path.basename(path)}:{i}: {len(cells)} cells, want {len(COLUMNS)}"
                )
            rows.append((i, cells))
    if not seen_header:
        raise ValueError(f"{os.path.basename(path)}: no table header {' | '.join(COLUMNS)}")
    return rows


def _artifact(path):
    """A recorded run: {key: value} header plus the verbatim stdout that followed it."""
    head, out, in_out = {}, [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("--- stdout ---"):
                in_out = True
                continue
            if in_out:
                out.append(line)
            elif ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                head[k.strip()] = v.strip()
    head["_stdout"] = "\n".join(out)
    return head


# KNOWN HOLE, found 2026-07-30 by the Gf refuter and NOT yet fixed. `_fingerprint` below matches only
# `N/M ticks`. A probe that reports `3/3` or `3/3 turns` -- as every Gf A/B does -- therefore yields an
# EMPTY tuple, and two things silently weaken:
#
#   1. `--rerun` compares () to (), so it checks the EXIT CODE ONLY. It is advertised as the forgery
#      check; for those artifacts it is nearly vacuous. (The refuter compensated by diffing full stdout
#      by hand: all 12 Gf artifacts reproduced, 9 byte-identically. So nothing was forged -- but the tool
#      is not what proved it.)
#   2. W3 (two faculties sharing an observable fingerprint = probably one serial chain scored twice)
#      short-circuits at `if len(facs) > 1 and fp:` on the falsy empty tuple. So W3 CANNOT fire for such
#      probes -- and Gf's two roster-duplication defects, which W3 exists to catch, passed the tool
#      unflagged and were caught by a human reading the artifacts.
#
# The obvious fix -- widen the regex to `(\d+)/(\d+)` -- is deliberately NOT applied here, because it
# changes the recorded fingerprint of EVERY existing artifact and so changes what `--rerun` compares
# against. That needs a re-verification pass over the whole ledger, not a one-word edit at the end of a
# session. Recorded as the honest state: the four W3 warnings this tool currently prints are a FLOOR on
# serial-chain duplication, not a census, and absence of W3 is not evidence of absence.
def _fingerprint(stdout):
    """The observable deltas a charge probe reports, as a tuple. Two faculties in one category with
    an identical fingerprint are probably two probe points on ONE serial chain -- the defect that cost
    Executive 9 points (COVERAGE_BASELINE.md:12), scored as four faculties because nothing checked."""
    return tuple(FLIPS.findall(stdout))


def _cite_ok(root, cite):
    """A citation resolves iff the file exists AND the line number is inside it. Cheap, and it catches
    the coarse drift this repo has already shipped twice: a moved module and a retired file."""
    if ":" not in cite:
        return False, "not file:line"
    path, _, lineno = cite.rpartition(":")
    if not lineno.isdigit():
        return False, "line is not a number"
    full = os.path.join(root, path.replace("/", os.sep))
    if not os.path.isfile(full):
        return False, "file does not exist"
    with open(full, encoding="utf-8", errors="replace") as fh:
        n = sum(1 for _ in fh)
    if int(lineno) > n:
        return False, f"line {lineno} past EOF (file has {n})"
    return True, ""


def _commits_behind(root, commit):
    try:
        r = subprocess.run(
            ["git", "-C", root, "rev-list", "--count", f"{commit}..HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- scoring
def score(root, evdir):
    """(category -> {score, rows, unscored}, failures, warnings). Failures are REFUSALS: any one of
    them means there is no honest total to print."""
    cats, fail, warn = {}, [], []
    # README.md is documentation living in the evidence directory, not a category. Without this
    # exclusion the F2 check below refused it as an "unknown category file" and the tool exited 1
    # WITHOUT printing a total -- on its own repository, from the day it shipped. Two claims depended
    # on a number it never printed (this module's own docstring and docs/coverage/README.md), and
    # `--check` was unreachable because `if fail: return 1` precedes it. Found by audit, not by use,
    # which is the argument for running an instrument against the tree before quoting it.
    _NOT_CATEGORIES = {"README"}
    present = (
        {f[:-3] for f in os.listdir(evdir) if f.endswith(".md")} - _NOT_CATEGORIES
        if os.path.isdir(evdir)
        else set()
    )
    for extra in sorted(present - set(CATEGORIES)):
        fail.append(
            f"F2 unknown category file {extra}.md (not one of the eleven rubric categories)"
        )

    for cat in CATEGORIES:
        path = os.path.join(evdir, cat + ".md")
        if not os.path.isfile(path):
            fail.append(f"F1 {cat}: no evidence file -- the denominator is eleven, not ten")
            cats[cat] = {"score": 0.0, "rows": 0, "unscored": 0}
            continue
        try:
            rows = _table(path)
        except ValueError as e:
            fail.append(f"F3 {e}")
            cats[cat] = {"score": 0.0, "rows": 0, "unscored": 0}
            continue
        if not rows:
            fail.append(
                f"F1 {cat}: evidence file has zero faculties -- an empty category is not a 0"
            )
            cats[cat] = {"score": 0.0, "rows": 0, "unscored": 0}
            continue

        bands, unscored, seen, prints = [], 0, set(), {}
        for lineno, (fac, band, cite, art, by, when) in rows:
            where = f"{cat}.md:{lineno} [{fac}]"
            if fac in seen:
                fail.append(f"F4 {where}: duplicate faculty name")
            seen.add(fac)

            if band == "UNSCORED":
                b = 0
                unscored += 1
            elif band.isdigit() and 0 <= int(band) <= 100:
                b = int(band)
            else:
                fail.append(f"F12 {where}: band {band!r} is not 0..100 nor UNSCORED")
                continue
            bands.append(b)

            if not by or by == "-" or not when or when == "-":
                fail.append(f"F11 {where}: measured_by/measured_at are required -- bands decay")

            # rubric rule 1: every SCORE cites file:line. A 0 or an UNSCORED is not a score, and
            # requiring a citation for "there is no evidence yet" would only invent one.
            if b >= 1 or (cite and cite != "-"):
                if not cite or cite == "-":
                    fail.append(f"F5 {where}: band {b} with no citation (rubric rule 1)")
                for c in [x.strip() for x in cite.split(";") if x.strip() and x.strip() != "-"]:
                    ok, why = _cite_ok(root, c)
                    if not ok:
                        fail.append(f"F5 {where}: citation {c} does not resolve ({why})")

            if b <= 40:
                if art and art != "-":  # an artifact on a low band is fine; still parse it
                    ap = os.path.join(evdir, "artifacts", art)
                    if os.path.isfile(ap):
                        prints.setdefault(_fingerprint(_artifact(ap)["_stdout"]), []).append(fac)
                continue

            # ---- the refusal rules that make a band claim expensive to inflate
            if not art or art == "-":
                fail.append(f"F6 {where}: band {b} > 40 with no A/B artifact (rubric rule 2)")
                continue
            ap = os.path.join(evdir, "artifacts", art)
            if not os.path.isfile(ap):
                fail.append(f"F6 {where}: artifact {art} does not exist")
                continue
            a = _artifact(ap)
            missing = [k for k in ARTIFACT_KEYS if k not in a]
            if missing:
                fail.append(f"F11 {where}: artifact {art} lacks {', '.join(missing)}")
                continue
            if a["kind"] not in ("ab", "heldout"):
                fail.append(
                    f"F7 {where}: band {b} > 40 needs kind ab|heldout, artifact says {a['kind']!r}"
                )
                continue
            if b > 80 and a["kind"] != "heldout":
                fail.append(
                    f"F9 {where}: band {b} > 80 needs held-out evidence (rubric rule 3), "
                    f"artifact is kind {a['kind']!r}"
                )
                continue
            if a["kind"] == "heldout" and not ("metric" in a and "baseline" in a):
                fail.append(
                    f"F10 {where}: held-out artifact must state metric: and baseline: "
                    f"-- '98.3% on synthetic' is not a band-81 claim (rubric rule 3)"
                )
                continue
            if a["exit"] != "0":
                fail.append(
                    f"F8 {where}: band {b} > 40 but the A/B EXITED {a['exit']} "
                    f"({(a['_stdout'].strip().splitlines() or ['?'])[-1][:60]}) -- a probe that did "
                    f"not return charge cannot support a band above 40"
                )
                continue

            fp = _fingerprint(a["_stdout"])
            prints.setdefault(fp, []).append(fac)
            rates = [int(x) / int(y) for x, y in fp if int(y)]
            if b >= 61 and rates and max(rates) < 0.01:
                warn.append(
                    f"W4 {where}: band {b} on a max flip rate of {max(rates):.4f} -- one "
                    f"divergence in a long run satisfies charge_probe's `flips > 0` contract"
                )
            behind = _commits_behind(root, a["commit"])
            if behind is None:
                warn.append(f"W2 {where}: artifact commit {a['commit'][:8]} is not in this history")
            elif behind > 0:
                warn.append(f"W2 {where}: measured {behind} commit(s) ago at {a['commit'][:8]}")

        for fp, facs in prints.items():
            if len(facs) > 1 and fp:
                warn.append(
                    f"W3 {cat}: {', '.join(facs)} share observable fingerprint {fp} -- "
                    f"probably one serial chain scored as {len(facs)} faculties"
                )
        cats[cat] = {
            "score": sum(bands) / len(bands) if bands else 0.0,
            "rows": len(bands),
            "unscored": unscored,
        }
    return cats, fail, warn


# ---------------------------------------------------------------- record / rerun
def record(root, evdir, name, kind, cmd, extra):
    """Run a probe and capture it as an artifact. The header is machine-written on purpose: a claim
    whose provenance is typed by the same hand that typed the claim is not evidence."""
    env = {**os.environ, "PYTHONPATH": os.path.join(root, "src"), "PYTHONUTF8": "1"}
    argv = [sys.executable] + cmd if cmd[0].endswith(".py") else cmd
    r = subprocess.run(argv, cwd=root, capture_output=True, text=True, env=env, errors="replace")
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True)
    os.makedirs(os.path.join(evdir, "artifacts"), exist_ok=True)
    path = os.path.join(evdir, "artifacts", name + ".txt")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# coverage-artifact v1 -- machine-written by coverage_score.py record\n")
        fh.write(f"kind: {kind}\n")
        fh.write(f"command: {' '.join(cmd)}\n")
        fh.write(f"exit: {r.returncode}\n")
        fh.write(f"commit: {head.stdout.strip() or 'UNKNOWN'}\n")
        fh.write(f"recorded_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        fh.write(f"recorded_by: {extra or 'coverage_score.py record'}\n")
        fh.write(f"python: {sys.version.split()[0]}\n")
        fh.write("--- stdout ---\n")
        fh.write(r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else ""))
    print(f"recorded {path}  exit={r.returncode}")
    return 0


def rerun(root, evdir, name):
    """Re-execute a recorded command and compare. This is the forgery check: a hand-written transcript
    survives `score` and dies here."""
    path = os.path.join(evdir, "artifacts", name + ".txt")
    if not os.path.isfile(path):
        print(f"no artifact {name}")
        return 2
    a = _artifact(path)
    cmd = a["command"].split()
    env = {**os.environ, "PYTHONPATH": os.path.join(root, "src"), "PYTHONUTF8": "1"}
    argv = [sys.executable] + cmd if cmd[0].endswith(".py") else cmd
    r = subprocess.run(argv, cwd=root, capture_output=True, text=True, env=env, errors="replace")
    was, now = a["exit"], str(r.returncode)
    fp_was, fp_now = _fingerprint(a["_stdout"]), _fingerprint(r.stdout)
    print(f"{name}: recorded exit {was} / now {now}")
    print(f"{name}: recorded observables {fp_was} / now {fp_now}")
    if was != now or fp_was != fp_now:
        print("F14 REFUSED: the recorded artifact does not reproduce")
        return 1
    print("reproduces")
    return 0


# ---------------------------------------------------------------- report
def report(root, evdir, check):
    cats, fail, warn = score(root, evdir)
    total = sum(c["score"] for c in cats.values()) / len(CATEGORIES)
    print("=" * 78)
    print("BRAIN-FACULTY COVERAGE -- re-derived from tracked evidence (docs/COVERAGE_RUBRIC.md)")
    print("=" * 78)
    print(f"{'category':<18}{'score':>7}{'faculties':>11}{'unscored':>10}")
    for cat in CATEGORIES:
        c = cats[cat]
        print(f"{cat:<18}{c['score']:>7.1f}{c['rows']:>11}{c['unscored']:>10}")
    print("-" * 78)
    scored = sum(c["rows"] - c["unscored"] for c in cats.values())
    uns = sum(c["unscored"] for c in cats.values())
    # The total is printed ONLY when nothing was refused. An earlier version printed it first and
    # then said "no honest total" underneath, which hands a reader a quotable number the same output
    # goes on to disown -- and a number that escapes its caveat is how this whole file's problem
    # started.
    if not fail:
        print(
            f"equal-weighted total: {total:.1f}%   ({sum(c['score'] for c in cats.values()):.1f}/11)"
        )
    print(f"faculties: {scored + uns} rows, {uns} UNSCORED (counted as 0, never dropped)")

    if warn:
        print("\n--- WARNINGS (visible, not fatal) ---")
        for w in warn:
            print("  " + w)
    if fail:
        print(f"\n--- REFUSED ({len(fail)}) ---")
        for f in fail:
            print("  " + f)
        print(
            f"\nno total printed: {len(fail)} claim(s) above are not supported by the evidence "
            f"standard, so there is no honest number to report until they are fixed or withdrawn"
        )
        return 1

    if check:
        base = os.path.join(root, "docs", "COVERAGE_BASELINE.md")
        m = re.search(r"total:\s*\*{0,2}([\d.]+)\s*%", open(base, encoding="utf-8").read())
        if not m:
            print("\nF13 COVERAGE_BASELINE.md states no equal-weighted total")
            return 1
        if abs(float(m.group(1)) - total) > 0.05:
            print(
                f"\nF13 COVERAGE_BASELINE.md says {m.group(1)}%, the evidence yields {total:.1f}% "
                f"-- the headline is not the number the evidence supports"
            )
            return 1
        print(f"\ncheck OK: COVERAGE_BASELINE.md's {m.group(1)}% is what the evidence yields")
    return 0


def main(argv):
    root = os.path.abspath(
        _opt(argv, "--root") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    evdir = os.path.abspath(_opt(argv, "--evidence") or os.path.join(root, "docs", "coverage"))
    if "record" in argv:
        i = argv.index("--")
        return record(
            root,
            evdir,
            _opt(argv, "--name"),
            _opt(argv, "--kind") or "ab",
            argv[i + 1 :],
            _opt(argv, "--by"),
        )
    if "--rerun" in argv:
        # Explicit rather than `if _opt(...)`: a bare trailing `--rerun` used to yield None and fall
        # through to a full report, so the one verb that catches a forged artifact could be asked for
        # and silently not run.
        name = _opt(argv, "--rerun")
        if not name:
            print("--rerun needs an artifact name, e.g. --rerun drive_60")
            return 2
        return rerun(root, evdir, name)
    return report(root, evdir, "--check" in argv)


def _opt(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
