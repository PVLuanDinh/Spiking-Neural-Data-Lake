"""citation_guard -- a TRACKED file may not cite an UNTRACKED .py instrument.

VERDICT: THE BROAD RULE IS DEAD, THE NARROW ONE IS ADOPTABLE, AND BOTH COUNTS ARE BELOW.

  STRICT (cited as a runnable reproduction path):  21 violations across  5 files  <- a work list
  BROAD  (cited anywhere in a tracked file)      : 475 violations across 127 files <- an allowlist

The strict scope is what the broad scope's own output argued for, and counting both is what made the
difference visible. 21 is adoptable: it names five documents, each with a specific broken command.
`docs/RUNNING.md` is the sharpest -- the project's how-to-run page has FIVE commands pointing at files
absent from a clone (`research/spiking_storage_prototype.py`, `spike_knowledge_graph.py`,
`nmnist_ingest.py` x2, `eth_mnist_bindsnet.py`), which is worse than the incident that motivated all
this, because it is the first page a new reader executes.

NOT YET AN ASSERTING GATE, and that is a work item rather than a policy question. Promoting it means
clearing or exempting those 21 first; asserting today would just redden CI. The pattern to follow is
`dormancy_gate.EXEMPT` -- each survivor listed with a reason -- and the honest reasons here differ:
DEPTH_CEILINGS' scratchpad citations need the scripts promoted into tools/ (they are real evidence for
real numbers), while RUNNING.md / JEPA.md / PERFORMANCE.md cite research files that appear to have been
retired, so those commands need correcting or deleting, not exempting.

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
    print(f"\nSTRICT: {len(strict)} violation(s) across {len(strict_files)} file(s)")
    print(f"BROAD : {len(viol)} violation(s) across {len(broad_files)} file(s)  [not adoptable]")

    # The gate ASSERTS nothing yet. Per the standing rule in docs/POLICIES.md -- "implement the gate
    # before adopting the rule ... its output counted, before the policy lands" -- this pass exists to
    # be COUNTED first. Promoting it to an assert is a separate, deliberate change, and if the count
    # above is allowlist-shaped then the honest outcome is that this gate dies instead.
    assert n_files > 100, (
        f"scanned only {n_files} tracked files -- the walk is broken, not the tree"
    )
    assert n_cites > 50, f"found only {n_cites} .py citations -- the regex is broken, not the tree"
    print("\nself-check OK: walk and regex both live (counts above are the reportable result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
