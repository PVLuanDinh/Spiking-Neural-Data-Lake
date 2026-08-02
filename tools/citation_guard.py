"""citation_guard -- a TRACKED file may not cite an UNTRACKED .py instrument.

VERDICT: THE BROAD RULE IS DEAD, THE NARROW ONE IS ADOPTED AND NOW ASSERTS.

  STRICT (cited as a runnable reproduction path):    0 violations + 1 exemption  <- ASSERTING GATE
  BROAD  (cited anywhere in a tracked file)      : ~452 across 127 files, 2026-07-30  <- report only

The BROAD figure is a DATED OBSERVATION, not an invariant -- it moves with every changelog entry and
every doc edit, and nothing asserts on it. Do not "fix" it when it drifts; re-date it if you re-measure.
The STRICT line is the one that must stay at 0-unexempted, and the gate below enforces that.

The strict scope is what the broad scope's own output argued for, and counting both is what made the
difference visible. 21 was adoptable because it named five documents, each with a specific broken
command, and the work list was finite. **All 21 were cleared on 2026-07-30 and the strict scope now
ASSERTS**; the broad count stays a printed report and stays non-asserting, for the two reasons recorded
further down (a CHANGELOG citing a retired file is CORRECT, and prose naming a missing file in order to
say it is missing is also correct -- neither is a defect, and gating on them buys a rubber-stamp
allowlist of the exact shape that killed two earlier policy proposals here).

HOW THE 21 WERE CLEARED, because the dispositions differed and lumping them would have hidden that:

  * 9 of 10 were STALE PATHS, not missing capabilities. The top-level `research/` tree became
    `src/snn_data_lake/reasoning/` in v3.78 (commit 627b2e2b, 2026-07-19) and five documents were never
    updated, so `docs/RUNNING.md` (x5), `docs/JEPA.md` (x3) and `docs/PERFORMANCE.md` (x1) were telling
    readers to run files that had merely MOVED. Corrected to `python -m snn_data_lake.reasoning.<mod>`,
    and the three replacements RUNNING.md names were run to a passing self-check first.
  * 1 was a genuine RETIREMENT: `research/spike_knowledge_graph.py` was merged with two siblings into
    `spike_kg.py` in commit 17067032 (2026-07-08) and deleted. The command names `spike_kg` now.
  * 10 of 11 in `docs/DEPTH_CEILINGS.md` were real evidence for published numbers living in gitignored
    `scratchpad/`. Eight scripts were PROMOTED into tools/ with their inputs parameterised as explicit
    paths and a clean SKIP when an input is absent. `tools/lang_scale.py 1e3 1e4` reproduces the first
    two rows of the doc's own scale table exactly, which is the strongest evidence available that a
    promotion preserved the instrument rather than merely relocating a file.
  * 1 was DELETED rather than promoted: `scratchpad/ar_lm_untrained_control.py` does not exist on disk
    at all, so there was nothing to promote, and the canonical section BUILDS the untrained twin inside
    the tracked `reasoning.blimp_bench` anyway. Its command is gone and the doc says the 49.5% is a
    recorded result reached through the bench.
  * 1 was EXEMPTED, and it is the only one. See EXEMPT below.

Promoting an instrument means three checkable things, not just `git add`: the input is an EXPLICIT path
argument rather than a hardcoded local one, an absent input prints SKIP and exits 0 rather than
reporting a number computed from no data, and the script was RUN. Shipping an unverified script would
have traded one false claim for another, which is the trade this whole exercise exists to refuse.

Original verdict on the broad rule, kept because the reasoning is the finding. Built and counted under the
standing rule "implement the gate before adopting the rule ... its output counted, *before* the policy
lands". Counted on this tree: **474 violations across 126 files, 195 of them in CHANGELOG.md alone**,
out of 1,431 .py citations in 574 tracked text files. That is the allowlist shape this project kills
gates for -- the precedent is a proposal that died at "58 failures where 1 was claimed", and this is
eight times worse. So the rule does NOT land as a CI gate, and this file lives in tools/ as a manual
REPORTING instrument, like capacity_curve.py and charge_probe.py.

Two reasons the count is legitimately that high, both of which the naive rule cannot tell from a defect:

  * CHANGELOG.md is a HISTORICAL RECORD. Citing `research/deep_stdp_readout.py` in a v2.x entry is
    correct even after that file was retired -- the entry describes a tree that existed. A gate that
    reddens on history would force either a lie or an allowlist.
  * `reasoning/` cites sibling `reasoning/` scripts, and that whole tier is ruff-excluded and outside
    the coverage denominator by design. Those are research notes pointing at research code.

WHAT SURVIVES, and it is narrower than the rule I set out to build: the real defect in all three real
incidents was an untracked script presented as a REPRODUCTION PATH -- inside a fenced command block or
under a "Reproduce" heading -- not a filename mentioned in prose. `docs/DEPTH_CEILINGS.md` (20 hits here)
is exactly that shape. **That hypothesis has since been built and counted -- see the STRICT verdict at
the top of this docstring, 21 across 5. This paragraph predicted it and is kept as the prediction.**

Kept rather than deleted because the count is the result. "We built the gate and its output killed it"
is a complete finding, and this file is the evidence so the next person does not rebuild it from the
same intuition.


WHY THIS EXISTS. Three times now, a tracked file has pointed at a script that does not survive a clone:

  1. `docs/POLICIES.md` cited `scratchpad/smoke_drive_charge.py` as the neuter-A/B probe. That file does
     not exist at all. `tools/charge_probe.py` is tracked ON PURPOSE because of it.
  2. `substrate/dormancy_gate.py` and the repo's CLAUDE.md carried the same dead citation.
  3. `docs/DEPTH_CEILINGS.md`'s AR appendix cites nine `scratchpad/ar_lm_*.py` scripts as the evidence
     for its BLiMP numbers -- added HOURS AFTER (1) and (2) were corrected, in the same session.

The failure is not that the scripts are missing; `scratchpad/` is gitignored on purpose. It is that a
tracked document presents them as the reproduction path, so on any other machine the numbers are
recorded results wearing the clothes of instructions. That is the "unreproducible measurement" this
project's coverage rubric exists to forbid, one level out from the code.

SCOPE, and it is the difference between a gate and noise. Only `.py` citations count. Citing untracked
`models/*.pt`, `data/`, `brain_data/` or `assets/` is DELIBERATE -- `.gitignore` calls them "Local-only
-- kept on disk, not published to the public repo", and `docs/TRAINING.md` documents their regeneration.
Data is data; an instrument is a claim about how a number was produced. Widening this to all paths would
report every dataset reference in the tree and earn the rubber-stamp allowlist that killed two earlier
policy proposals.

WHAT IT DOES NOT PROVE. That a tracked script RUNS, or that it reproduces the number beside it. It
proves the weaker, checkable thing: the instrument a reader is pointed at exists in the repository they
cloned. Deliberately weaker than the claim, because that is what a static pass can support.

  PYTHONUTF8=1 python tools/citation_guard.py
"""

import os
import re
import subprocess

# A path-like token ending in .py, optionally followed by :LINE. Requires a directory separator so bare
# module names ("foo.py" in prose) do not match -- a citation worth checking names where the file lives.
_CITE = re.compile(r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+\.py)(?::\d+)?")

# Files that may legitimately name an untracked .py:
#   - .gitignore itself, which is where the ignoring is declared
#   - this module, whose docstring quotes the offenders as its own motivating cases
_EXEMPT_FILES = {".gitignore", "tools/citation_guard.py"}

# STRICT-scope survivors: "path:line" -> why this citation legitimately stays. The pattern is
# substrate/dormancy_gate.EXEMPT's -- every survivor named with its reason IN THE SOURCE, so an
# exemption is something a reader can audit rather than a count quietly tolerated.
#
# Line-keyed exemptions go stale when a document is edited, and a stale exemption that silently blesses
# nothing is worse than none: it reads like coverage. So main() also asserts that every key here still
# matches a real strict violation. An exemption must keep earning its place or the gate says so.
EXEMPT = {
    # RETIRED 2026-08-02, on the terms the entry set for itself: specs/003 landed, so
    # tools/timing_heldout.py exists, runs, and measures something. The two quickstart citations
    # now resolve on their own and need no exemption. Recorded as a comment rather than deleted
    # silently, because "retire this entry the moment specs/003 implements or is abandoned" was a
    # promise, and a promise kept is worth as much evidence as one broken.
    "docs/HARDWARE_SETUP.md:144": (
        "FALSE POSITIVE, and the tokenizer is why. The cited file IS tracked -- at "
        "`integrations/robot arm/sensor_reading.py`. That directory name contains a SPACE, so _CITE's "
        "path token cannot span it and captures only the tail `arm/sensor_reading.py`, which no tracked "
        "path ends with on a `/` boundary. Fixing it in _resolves() (accepting a boundary of ' ' as well "
        "as '/') was considered and REJECTED: it would also resolve the STALE `robot arm/...` form this "
        "same line carried until 2026-07-30, when the sketch tree had already moved under integrations/ "
        "-- so the tokenizer fix would have hidden the live defect instead of reporting it. The false "
        "positive is exempted HERE, where the reason is readable, rather than dissolved into the "
        "resolver where it would blind the gate to a whole class. Verified by hand: "
        "`git ls-files | grep sensor_reading` returns the integrations/ path."
    ),
}

# Prefixes that are not repo paths at all (upstream URLs, site-packages, other checkouts).
_NOT_REPO = ("http", "www.", "site-packages/", "venv/", ".venv")


def _git(args, root):
    r = subprocess.run(
        ["git", "-C", root] + args, capture_output=True, text=True, errors="replace", timeout=60
    )
    return r.stdout.splitlines() if r.returncode == 0 else []


def repo_root():
    # This file lives in tools/, so the root is ONE level up. It was three while the module sat in
    # src/snn_data_lake/substrate/, and moving it without fixing this resolved to a directory outside
    # the repo -- where `git ls-files` returns nothing, `scan()` returns None, and main() prints
    # "SKIP: not a git checkout" and exits 0. A clean exit and no findings, which reads exactly like
    # success. Caught by noticing the violation count had vanished from the output, not by the exit code:
    # the same silent-degradation shape this session found six other instances of.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def scan(root):
    """(violations, n_files_scanned, n_citations). A violation is (citing_file, lineno, cited_path)."""
    tracked = set(_git(["ls-files"], root))
    if not tracked:
        return (
            None,
            0,
            0,
        )  # not a git checkout (or git absent) -- caller decides; never a silent pass

    # Only text files worth citing from. Sources of truth for humans and for CI.
    scanned = [
        f
        for f in sorted(tracked)
        if f.endswith((".md", ".py", ".toml", ".yml", ".yaml", ".cfg", ".bat", ".sh", ".ps1"))
        and f not in _EXEMPT_FILES
    ]

    suffix_index = _suffix_index(tracked)
    viol, n_cites = [], 0
    for rel in scanned:
        full = os.path.join(root, rel.replace("/", os.sep))
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        runnable = _runnable_lines(lines) if rel.endswith(".md") else set()
        for i, line in enumerate(lines, 1):
            for m in _CITE.finditer(line):
                cited = m.group(1)
                if cited.startswith(_NOT_REPO) or any(p in cited for p in _NOT_REPO):
                    continue
                n_cites += 1
                if not _resolves(cited, tracked, suffix_index):
                    viol.append((rel, i, cited, i in runnable))
    return viol, len(scanned), n_cites


# Fenced-block languages that mean "this is a command a reader will paste". A bare ``` fence is
# EXCLUDED on purpose: in this repo those hold captured OUTPUT as often as commands, and counting them
# put noise back into the number the strict scan exists to keep small.
_SHELL_LANGS = {"bash", "sh", "shell", "console", "powershell", "ps", "ps1", "bat", "cmd"}

# A heading that promises the reader can re-run what follows.
_RUNNABLE_HEADING = re.compile(r"reproduc|^\s*#+\s*run\b|how to run|usage", re.I)


def _runnable_lines(lines):
    """Indices (1-based) of lines presented as RUNNABLE: inside a shell-fenced block, or under a
    heading that promises reproduction.

    This is the narrow scope the broad scan's own output argued for. The broad rule counted every
    `.py` path anywhere in a tracked file and reported 475 violations across 127 files -- because a
    CHANGELOG citing a retired file is correct, and because prose naming a missing file in order to
    say it is missing is also correct. Neither is a defect. What IS a defect is an untracked script
    sitting in a command block under a "Reproduce" heading, because that tells a reader to run
    something their clone does not contain.
    """
    runnable, fence_lang, heading_ok = set(), None, False
    for i, line in enumerate(lines, 1):
        s = line.lstrip()
        if s.startswith("```"):
            if fence_lang is None:
                fence_lang = s[3:].strip().split()[0].lower() if s[3:].strip() else ""
            else:
                fence_lang = None
            continue
        if fence_lang is None and s.startswith("#"):
            heading_ok = bool(_RUNNABLE_HEADING.search(line))
            continue
        if (fence_lang in _SHELL_LANGS) or heading_ok:
            runnable.add(i)
    return runnable


def _suffix_index(tracked):
    """basename -> set of tracked paths with that basename. Built once; the resolution below is a
    suffix test against it rather than a scan of all 600+ paths per citation."""
    idx = {}
    for p in tracked:
        idx.setdefault(p.rsplit("/", 1)[-1], set()).add(p)
    return idx


def _resolves(cited, tracked, suffix_index):
    """True if this citation names a tracked file.

    SUFFIX resolution, and it is the difference between a gate and a false-positive machine. The naive
    version -- `cited in tracked` -- reported 1,431 citations as violations on the first run, because
    this tree cites modules RELATIVELY: `substrate/ingest_guard.py` is really
    `src/snn_data_lake/substrate/ingest_guard.py`, and `research/fast_features.py` sits under a package
    root too. Every one of those files exists. Counting the naive output first is what caught it -- the
    same lesson `method_dormancy_gate` records, where matching bare call names scored 0 failures and 47
    UNCERTAIN and could not catch its own motivating case.

    So a citation resolves iff SOME tracked path ends with it on a path boundary. That is deliberately
    permissive: it cannot tell `a/b/foo.py` from `x/a/b/foo.py`. The alternative -- demanding
    repo-root-exact citations -- would fail the entire existing corpus of correct relative references,
    which is an allowlist-shaped result, and this project kills gates that produce those.
    """
    if cited in tracked:
        return True
    for cand in suffix_index.get(cited.rsplit("/", 1)[-1], ()):
        if cand == cited or cand.endswith("/" + cited):
            return True
    return False


def main():
    root = repo_root()
    viol, n_files, n_cites = scan(root)

    print("=" * 78)
    print("citation_guard -- a tracked file may not cite an untracked .py instrument")
    print("=" * 78)
    if viol is None:
        print("SKIP: not a git checkout (or git unavailable) -- nothing to compare against")
        return 0

    print(f"scanned {n_files} tracked text files, {n_cites} .py citations")

    strict = [v for v in viol if v[3]]
    broad_files = {v[0] for v in viol}
    strict_files = {}
    for rel, ln, cited, _ in strict:
        strict_files.setdefault(rel, []).append((ln, cited))

    print(
        "\n--- STRICT scope: cited as a RUNNABLE reproduction path (shell fence or Reproduce/Usage"
    )
    print("    heading). This is the scope the broad count argued for. ---")
    for rel in sorted(strict_files):
        print(f"\n  {rel}")
        for ln, cited in strict_files[rel]:
            print(f"    :{ln}  ->  {cited}")
    if not strict_files:
        print("  (none)")
    strict_keys = {f"{rel}:{ln}" for rel, ln, _c, _r in strict}
    unexempt = [v for v in strict if f"{v[0]}:{v[1]}" not in EXEMPT]
    stale = sorted(set(EXEMPT) - strict_keys)

    print(f"\nSTRICT: {len(unexempt)} unexempted violation(s), {len(EXEMPT)} exemption(s)")
    for key, why in sorted(EXEMPT.items()):
        print(f"  EXEMPT {key}")
        print(f"         {why[:96]}...")
    print(f"BROAD : {len(viol)} violation(s) across {len(broad_files)} file(s)  [report only]")

    # These two came first and stay: they prove the INSTRUMENT is live before its findings are trusted.
    # A broken walk resolves to zero findings, which reads exactly like success -- the silent-degradation
    # shape repo_root()'s comment records catching the hard way.
    assert n_files > 100, (
        f"scanned only {n_files} tracked files -- the walk is broken, not the tree"
    )
    assert n_cites > 50, f"found only {n_cites} .py citations -- the regex is broken, not the tree"

    # THE GATE. Asserting since 2026-07-30, after all 21 findings of the counted run were cleared --
    # per the standing rule in docs/POLICIES.md, the gate was built and its output COUNTED before the
    # rule landed. The BROAD count above deliberately does NOT assert; see the module docstring.
    assert not stale, (
        "stale EXEMPT entr(ies) -- these no longer match any strict violation, so they bless nothing "
        "and only make the gate look more covered than it is. Delete them, or fix the line number if "
        f"the document moved: {stale}"
    )
    assert not unexempt, (
        "a tracked file presents an UNTRACKED .py as a runnable reproduction path, so a reader who "
        "clones this repo is told to run a file they do not have. Fix the path if it moved, delete the "
        "command if the capability is gone, or promote the script into tools/ if it is real evidence "
        "for a real number -- and only add it to EXEMPT with a reason if it legitimately stays:\n"
        + "\n".join(f"  {rel}:{ln} -> {cited}" for rel, ln, cited, _ in unexempt)
    )
    print("\ngate OK: no tracked file cites an untracked .py as a runnable reproduction path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
