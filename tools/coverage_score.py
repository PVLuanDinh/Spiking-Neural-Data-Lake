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
import tempfile
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
# Any bare `N/M` counter, with lookarounds that exclude the three shapes that would poison a
# fingerprint: a date or path (`2026/07/30`, `logs/12/34` -- a `/` on either side), a decimal
# (`1.5/2`, `12/400.5`), and a digit-glued suffix (the `\b` is what stops `\d+` from backtracking
# into the middle of a date). Probes report their observables as `N/M ticks`, `N/M turns`, `N/M
# calls`, `N/M` bare, and `(856/1675)`; all five are real counters and all five now count.
FLIPS = re.compile(r"(?<![\d./\\])(\d+)/(\d+)\b(?![/\\])(?!\.\d)")


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


# THE `ticks`-ONLY HOLE IS FIXED, 2026-07-30, after the whole-ledger re-verification it was waiting on.
# It used to match only `N/M ticks`, so a probe reporting `3/3 turns` -- as every Gf A/B does -- yielded
# an EMPTY tuple, `--rerun` compared () to () and degraded to an exit-code check, and W3/W5 could not
# fire at all for those artifacts (both short-circuit on the falsy empty tuple). `FLIPS` above now takes
# any bare `N/M`. A bare `(\d+)/(\d+)` was the wrong widening -- it eats dates, paths and decimals -- so
# the lookarounds reject those three shapes; `tests/test_coverage_score_fingerprint.py` pins them.
#
# MEASURED EFFECT on the 82 tracked artifacts: 13 went from `()` to a real fingerprint, 19 already had
# one and got a longer one, 50 are unchanged, and 2 are still `()` honestly (`gf_sleep_schema_induction`
# reports `2 (A) / 0 (B)` counts and `lang_no_ab_source` is a one-line NOT-ELIGIBLE message -- neither
# prints a ratio, so `--rerun` is still exit-code-only for those two, and that is the true state rather
# than a hole). W3 went from 4 warnings to 6 and W5 from 1 group to 2. No band and no total moved:
# fingerprints feed warnings only.
#
# WHAT SURVIVES, and it is why W3/W5 are STILL A FLOOR rather than a census. Widening only ever makes a
# fingerprint LONGER, and the dominant remaining defect is that a longer fingerprint has more places to
# differ:
#
#   1. The nuisance-counter defect (`docs/coverage/Executive.md:143-154`). `exec_conflict_60` and
#      `exec_evc_60` are the same experiment on all three behavioural observables and still do not
#      collide, because their eligibility line reads 56/60 in one and 55/60 in the other. Widening adds
#      such counters to nineteen more artifacts, so it makes this failure mode MORE available, not less.
#      The real fix is probe-side -- fingerprint NAMED observables, not every ratio in the stream --
#      which needs six probe files to agree on a name and was out of scope for this pass.
#   2. A fingerprint is only as discriminating as what the probe prints. `gf_reply_ab.py` prints exactly
#      `hits/n` and `flips/n`, and `flips == hits` in all nine of its artifacts, so its fingerprint
#      encodes the ITEM COUNT and nothing about the mechanism. Two of the three new Gf W3 groups are
#      that coincidence (adjudicated individually in `docs/coverage/Gf.md`), and the one genuine Gf
#      duplicate W3 still cannot see -- `active_experiment_loop` vs `number_series` -- is invisible
#      because those two posed 2 and 3 items, so they fingerprint `(1/2, 1/2)` and `(3/3, 3/3)`.
#
# So the honest statement is narrower than before but the same in kind: absence of W3/W5 is not evidence
# of absence, and a W3/W5 warning is a prompt to adjudicate, not a finding. The scoreboard for this fix:
# it mechanically RETRACTED one false positive that three human passes had already retracted by hand
# (Imagination's three all-zero negatives), mechanically CONFIRMED one real within-category duplicate
# (Gf's letter/novel analogy pair) and one real cross-category one (Executive/Imagination's plan pair),
# and added two coincidences. One of the two duplicates W3 was built to catch, caught.
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
    # W5's accumulator. `prints` below is re-bound inside the per-category loop, which is what makes
    # W3 per-category; this one outlives the loop so the same fingerprint can be seen in two files.
    gprints = {}
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
                        lowfp = _fingerprint(_artifact(ap)["_stdout"])
                        prints.setdefault(lowfp, []).append(fac)
                        gprints.setdefault(lowfp, []).append((cat, fac, b))
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
            gprints.setdefault(fp, []).append((cat, fac, b))
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

    # W5, added 2026-07-30 by the adjudication of Imagination's `latent_world_model` against
    # Executive's `acc_conflict_monitoring_evc`. W3 above is INSIDE the per-category loop -- `prints`
    # is re-bound per category -- so it is structurally incapable of seeing one mechanism scored in two
    # DIFFERENT categories. A cross-file duplicate is the worse of the two: no single auditor's write
    # scope covers both halves, so neither is obliged to notice, and the withdrawn baseline's 9-point
    # Executive defect was exactly this shape (four probe points on one serial chain).
    #
    # HONEST LIMIT, and it is the reason this check is a floor rather than a census: the collision that
    # PROMPTED it is STILL not one this check can see. `imag_jepa_action_60` fingerprints as a 9-tuple
    # and `exec_conflict_60` as a 4-tuple, so they never collide under `_fingerprint`; their shared
    # `(0/60, 47/60, 23/60)` is three HUMAN-named observables, and the two probes disagree on how many
    # OTHER ratios they print (an eligibility counter, a value-table count, a scramble arm).
    #
    # This paragraph used to blame the `ticks`-only regex and defer the widening. The widening happened
    # on 2026-07-30 (see above `_fingerprint`) and the deferral's premise turned out to be wrong on the
    # merits: BOTH of these artifacts already had non-empty fingerprints, the widening left them a
    # 9-tuple and a 4-tuple, and they still do not meet. The blocker was never the regex -- it is that a
    # fingerprint is a positional tuple of EVERY ratio in the stream, so one extra counter in one probe
    # is enough to hide a real collision. Fixing THAT means fingerprinting named observables, which is a
    # probe-side change across six files.
    #
    # What the widening did buy W5: a second group, `Executive.multi_step_plan_sequence` against
    # `Imagination.model_based_plan_rollout` (`docs/coverage/Imagination.md:112-119`), which is a real
    # one-chain pair -- the first genuine duplication W5 has caught rather than inherited from a human.
    # So W5 catches EXACT-fingerprint cross-category collisions and nothing weaker -- absence of W5 is
    # not evidence of absence, the same caveat W3 already carries.
    #
    # The count of colliding rows banded ABOVE 40 is reported because that is the discriminator
    # between noise and inflation: F8 demands an exit-0 A/B only above 40, so a band at or below 40
    # does not rest on its artifact even when it names one, and a group of all-zero honest negatives
    # sharing `(0/60, 0/60)` is the expected signature of independent nulls rather than of one chain.
    for fp, hits in gprints.items():
        hit_cats = sorted({c for c, _, _ in hits})
        if not fp or len(hit_cats) < 2:
            continue
        hot = [f"{c}.{f}" for c, f, b in hits if b > 40]
        rests = (
            f"{len(hot)} of {len(hits)} rows are banded >40 ({', '.join(hot)}), so their band RESTS "
            f"on this artifact"
            if hot
            else f"none of the {len(hits)} rows is banded >40, so no band rests on this artifact"
        )
        warn.append(
            f"W5 cross-category: {', '.join(f'{c}.{f}' for c, f, _ in hits)} share observable "
            f"fingerprint {fp} across {len(hit_cats)} categories ({', '.join(hit_cats)}) -- "
            f"probably one serial chain scored in more than one category; {rests}"
        )
    return cats, fail, warn


# ---------------------------------------------------------------- record / rerun
# Probes run against an ISOLATED lake, decided by measurement on 2026-07-30 rather than by preference.
# Before this, `record` and `rerun` passed only PYTHONPATH/PYTHONUTF8, so a probe read whatever
# `~/.snn_data_lake` happened to hold -- and with the always-on daemon writing that lake continuously,
# the ledger was unreproducible through no fault of the evidence. Ambient: 70 of 82 artifacts
# reproduce. Under a fresh empty NORD_HOME: 80 of 82.
#
# The risk this had to clear first was the opposite of the obvious one. A cold lake could have turned
# real evidence into null evidence -- a probe that legitimately reads accumulated state going quietly
# flat, taking the floor out from under whatever band rested on it. That did NOT happen: not one
# artifact that reproduced ambiently broke under isolation, so no probe's observables depend on the
# ambient lake, and nothing was re-recorded to reach the 80.
#
# The 2 that fail under BOTH conditions are unreproducible in principle, not stale: they report a
# metacog reliability counter that reads mutable durable state (48/120 recorded vs 50/120 live; 3/60
# vs 5/60 vs 22/60). They are documented in docs/coverage/README.md and deliberately NOT re-recorded --
# re-recording a moving number just freezes one more snapshot of it and calls the snapshot evidence.
#
# An explicit NORD_HOME in the caller's environment WINS, so pointing the ledger at a frozen lake
# snapshot stays possible without touching this file.
def _probe_env(root, home):
    return {
        **os.environ,
        "PYTHONPATH": os.path.join(root, "src"),
        "PYTHONUTF8": "1",
        "NORD_HOME": home,
    }


def record(root, evdir, name, kind, cmd, extra):
    """Run a probe and capture it as an artifact. The header is machine-written on purpose: a claim
    whose provenance is typed by the same hand that typed the claim is not evidence."""
    argv = [sys.executable] + cmd if cmd[0].endswith(".py") else cmd
    with tempfile.TemporaryDirectory(prefix="coverage_home_") as _home:
        env = _probe_env(root, os.environ.get("NORD_HOME") or _home)
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
    argv = [sys.executable] + cmd if cmd[0].endswith(".py") else cmd
    with tempfile.TemporaryDirectory(prefix="coverage_home_") as _home:
        env = _probe_env(root, os.environ.get("NORD_HOME") or _home)
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
