"""Derive the policy manifest from docs/POLICIES.md, and audit the tree against it.

# distinct-from: gradient_claim_gate.py -- that gate censuses GRADIENT CLAIMS in src
# code (AST walk over modules); this derives the POLICY MANIFEST from the governing
# document's own tables and audits gate commands. Same wiring pattern (substrate
# module, --assert in CI, bare == assert), different job on a different corpus.
# distinct-from: serial_arm.py, nef_compute.py, pc_active_inference.py,
# godel_machine.py -- lexical matches on "command"/"policy"/"derive" only.

What this module DERIVES (never hand-maintains): one row per policy declared in
docs/POLICIES.md -- id, title, enforcement kind (absolute / ratchet / review-only),
and runnable gate commands or NONE -- from three sources inside that one document:
the policy headings, the enforcement table, and the gate-output table. What it
NEVER does: edit any policy, restate any policy, or decide a review-only policy.
A policy no machine decides is reported REVIEW_ONLY -- honestly, never green.

Derivation failures (empty walk, reconciliation mismatch) exit nonzero BEFORE any
row prints: a dead walk and a clean tree must never look alike.

Bare invocation (what run_selfchecks.py executes, passing no argv) is `--assert`,
the sibling-gate convention: a registered module whose bare path cannot fail is a
PASS that can never fail.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
POLICIES = REPO / "docs" / "POLICIES.md"

# e-prop is TABLE-ONLY: it has an enforcement-table row and NO heading anywhere in
# POLICIES.md (the invariant lives under "## Cross-cutting invariants"). It is a
# first-class row kind, not an error; a future table-only invariant EXTENDS this
# set. The reconciliation identity is:
#   parent_headings + len(TABLE_ONLY_IDS) == table_policy_rows
TABLE_ONLY_IDS = frozenset({"e-prop"})

# Parent policy headings: "## P0 - ...". Sub-clauses are DISCOVERED, never
# enumerated (an enumerated list is itself the hand-maintained list FR-002
# forbids): P4b, P4c, "P4c amendment", P5a, "P3 amendment", any future P4d.
PARENT_RE = re.compile(r"^## (P\d)\s*[-—]\s*(.+)$", re.M)
SUBCLAUSE_RE = re.compile(r"^### (P\d(?:[a-z]+)?(?: amendment)?)\s*[-—]\s*(.+)$", re.M)

# A backticked token is a command IFF it matches this. Everything else in a table
# cell -- `main()`, `.backward()`, `mlp_q`, line-cites like `dormancy_gate.py:71`
# -- is a citation, not a command.
CMD_RE = re.compile(r"^(tools|substrate)/[a-z0-9_]+\.py( --[a-z-]+)*$")
BACKTICK_RE = re.compile(r"`([^`]+)`")


@dataclass
class GateCommand:
    argv: list[str]
    source_token: str
    in_ci: bool = False
    in_selfchecks: bool = False


@dataclass
class Policy:
    id: str
    title: str
    kind: str  # absolute | ratchet | review-only | partial (e-prop's honest split)
    commands: list[GateCommand] = field(default_factory=list)
    subclause_of: str | None = None  # set on sub-clause OUTPUT rows
    table_only: bool = False


@dataclass
class AuditResult:
    policy_id: str
    verdict: str  # PASS | FAIL | CANNOT_RUN | NO_GATE | REVIEW_ONLY
    detail: str = ""


class DerivationError(RuntimeError):
    pass


def token_to_argv(token: str) -> list[str]:
    parts = token.split()
    path, flags = parts[0], parts[1:]
    if path.startswith("substrate/"):
        mod = "snn_data_lake.substrate." + path.split("/", 1)[1].removesuffix(".py")
        return [sys.executable, "-m", mod, *flags]
    return [sys.executable, path, *flags]


def extract_commands(text: str) -> list[GateCommand]:
    out = []
    for tok in BACKTICK_RE.findall(text):
        if CMD_RE.match(tok):
            out.append(GateCommand(argv=token_to_argv(tok), source_token=tok))
    return out


def _split_rows(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and not set(cells[0]) <= {"-", " ", ":"}:
            rows.append(cells)
    return rows


def parse(text: str) -> list[Policy]:
    """Derive the full policy set. Raises DerivationError before any output."""
    parents = PARENT_RE.findall(text)
    subclauses = SUBCLAUSE_RE.findall(text)
    if not parents:
        raise DerivationError(
            "0 policies parsed -- refusing: a dead walk and a clean tree look identical"
        )

    # --- the two head-matter tables, located by STRUCTURAL anchor, never line numbers ---
    tables: list[list[str]] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            cur.append(line)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    enforcement = next((t for t in tables if "mechanized in CI" in t[0]), None)
    gate_output = next(
        (t for t in tables if t[0].replace(" ", "").startswith("|gate|output|")), None
    )
    if enforcement is None:
        raise DerivationError("enforcement table not found (header 'mechanized in CI')")

    # --- enforcement rows -> {id: mech_cell} ---
    table_rows: dict[str, str] = {}
    for cells in _split_rows(enforcement[2:]):
        head = cells[0].replace("*", "").strip()
        if "not a policy" in head:
            continue  # hygiene rows exclude themselves by their own label
        m = re.match(r"(P\d)\b", head)
        pid = m.group(1) if m else ("e-prop" if head.startswith("e-prop") else None)
        if pid is None:
            continue
        table_rows[pid] = cells[1] if len(cells) > 1 else ""

    # --- reconciliation identity (C1): parents + table_only == table rows ---
    n_par, n_tab = len(parents), len(table_rows)
    if n_par + len(TABLE_ONLY_IDS) != n_tab:
        raise DerivationError(
            f"reconciliation failed: parents={n_par} table_only={len(TABLE_ONLY_IDS)} "
            f"table_rows={n_tab} -- {n_par} + {len(TABLE_ONLY_IDS)} != {n_tab}"
        )
    for tid in TABLE_ONLY_IDS:
        if tid in {pid for pid, _ in parents}:
            raise DerivationError(
                f"reconciliation failed: table-only id {tid!r} now HAS a heading (double count)"
            )
        if tid not in table_rows:
            raise DerivationError(f"reconciliation failed: table-only id {tid!r} has no table row")

    # --- section bodies; a parent's OWN text stops at its first ### sub-clause ---
    def section(pid: str) -> str:
        m = re.search(rf"^## {pid}\b.*?$", text, re.M)
        if not m:
            return ""
        start = m.end()
        nxt = re.search(r"^## ", text[start:], re.M)
        body = text[start : start + nxt.start()] if nxt else text[start:]
        sub = re.search(r"^### ", body, re.M)
        return body[: sub.start()] if sub else body

    def kind_of(mech: str, body: str) -> str:
        if mech.strip() in {"", "-", "—"}:
            return "review-only"
        if re.search(r"\bratchet", mech + " " + body, re.I):
            return "ratchet"
        return "absolute"

    policies: list[Policy] = []
    for pid, title in parents:
        mech = table_rows[pid]
        policies.append(
            Policy(
                id=pid,
                title=title.strip(),
                kind=kind_of(mech, section(pid)),
                commands=extract_commands(mech),
            )
        )
    for tid in sorted(TABLE_ONLY_IDS):
        mech = table_rows[tid]
        # e-prop's "PARTIALLY" is an honest split: its gate commands are mechanized,
        # the remainder review-only. Kind "partial" says exactly that; it is never
        # averaged into absolute or ratchet.
        kind = "partial" if re.search(r"\bPARTIAL", mech) else kind_of(mech, "")
        policies.append(
            Policy(
                id=tid,
                title="e-prop not backprop",
                kind=kind,
                commands=extract_commands(mech),
                table_only=True,
            )
        )

    # --- sub-clauses: first-class OUTPUT rows with their OWN kind, default
    # review-only, NEVER inheriting the parent's command (P4b must render
    # NOT MACHINE-DECIDED). Excluded from the reconciliation identity above.
    # Commands come from the GATE-OUTPUT TABLE ONLY, never from section bodies:
    # measured on the first live run, body prose file-mentions satisfy CMD_RE
    # (`substrate/cockpit.py`, `substrate/learn_daemon.py` false-attached to P4c)
    # -- prose is citation, not a command source. An argv row attaches to the
    # sub-clause whose lowered id prefixes the gate basename (p4c_ratchet -> P4c);
    # "amendment" rows never attach (p4c_ratchet asserts P4c's count, not the
    # amendment's reach-main rule). ---
    for sid, title in subclauses:
        cmds: list[GateCommand] = []
        ratchet_hint = False
        if gate_output is not None and "amendment" not in sid:
            prefix = sid.lower() + "_"
            for cells in _split_rows(gate_output[2:]):
                for cmd in extract_commands(cells[0]):
                    base = Path(cmd.source_token.split()[0]).name
                    if base.startswith(prefix):
                        # A bare path token runs the gate's REPORT mode -- a PASS
                        # that cannot fail. When the row itself names --assert,
                        # the audit must run the asserting form.
                        if len(cmd.argv) == 2 and "--assert" in " ".join(cells):
                            cmd.argv.append("--assert")
                            cmd.source_token += " --assert"
                        if cmd.source_token not in {c.source_token for c in cmds}:
                            cmds.append(cmd)
                        if "ratchet" in base or re.search(r"\bratchet", " ".join(cells), re.I):
                            ratchet_hint = True
        kind = ("ratchet" if ratchet_hint else "absolute") if cmds else "review-only"
        policies.append(
            Policy(
                id=sid,
                title=title.strip(),
                kind=kind,
                commands=cmds,
                subclause_of=sid.split()[0][:2],
            )
        )
    return policies


def derive() -> list[Policy]:
    if not POLICIES.exists():
        raise DerivationError(f"{POLICIES} missing")
    return parse(open(POLICIES, encoding="utf-8").read())


# --------------------------------------------------------------------------- #
# Liveness cross-check + unclaimed-gate census (R2)                           #
# distinct-from: nef_compute.py -- that derives its OWN selfcheck command      #
# string; this READS run_selfchecks.py's registration list as a census        #
# surface. distinct-from: self_mod_harness.py, neuron_zoo.py -- they RUN the  #
# suite / manage its executor; nothing here executes run_selfchecks.          #
# --------------------------------------------------------------------------- #

CI_YML = REPO / ".github" / "workflows" / "ci.yml"
RUN_SELFCHECKS = REPO / "run_selfchecks.py"

# The gate predicate, stated once and tested: run_selfchecks registers ~50
# substrate modules, most of them NOT gates. A registered module counts as a
# runnable GATE iff its name carries a gate-shaped word. tools/*.py appearing in
# a ci.yml `--assert` step are gates by construction.
GATE_NAME_RE = re.compile(r"(gate|guard|envelope|manifest)")

SELF_NAME = "policy_manifest"
SELF_DISPOSITION = (
    "self-disposition: policy_manifest (this instrument) is EXCLUDED from its own "
    "unclaimed-gate census -- it is the census taker, not a policy gate; its "
    "liveness is asserted by the CI step and run_selfchecks entry that run it. "
    "Printed on every report so the exclusion is never silent."
)


def _selfcheck_modules() -> set[str]:
    if not RUN_SELFCHECKS.exists():
        return set()
    return set(re.findall(r'"substrate\.([a-z0-9_]+)"', RUN_SELFCHECKS.read_text(encoding="utf-8")))


def cross_check(policies: list[Policy]) -> list[Policy]:
    """Fill in_ci / in_selfchecks on every derived command. The selfcheck runner
    is itself a CI step in every matrix job, so a registered module counts as
    CI-reachable."""
    ci = CI_YML.read_text(encoding="utf-8") if CI_YML.exists() else ""
    mods = _selfcheck_modules()
    for p in policies:
        for c in p.commands:
            path = c.source_token.split()[0]
            if path.startswith("substrate/"):
                stem = Path(path).stem
                c.in_selfchecks = stem in mods
                c.in_ci = stem in ci or c.in_selfchecks
            else:
                c.in_ci = path in ci
    return policies


def census(policies: list[Policy]) -> dict:
    """Drift findings: UNCLAIMED_GATE (a runnable gate no policy claims -- claimed
    means the gate's name appears anywhere in POLICIES.md, the measured
    semantics) and NOT_IN_CI (a claimed mechanized command nothing in CI runs)."""
    text = POLICIES.read_text(encoding="utf-8")
    ci = CI_YML.read_text(encoding="utf-8") if CI_YML.exists() else ""
    runnable = {m for m in _selfcheck_modules() if GATE_NAME_RE.search(m)}
    for line in ci.splitlines():
        if "--assert" in line:
            runnable.update(Path(t).stem for t in re.findall(r"tools/[a-z0-9_]+\.py", line))
    unclaimed = sorted(g for g in runnable if g != SELF_NAME and g not in text)
    not_in_ci = sorted(
        c.source_token
        for p in policies
        if p.kind in {"absolute", "ratchet", "partial"}
        for c in p.commands
        if not c.in_ci
    )
    return {
        "unclaimed_gates": unclaimed,
        "not_in_ci": not_in_ci,
        "self_disposition": SELF_DISPOSITION,
    }


def render(policies: list[Policy]) -> str:
    lines = [
        "POLICY MANIFEST -- derived from docs/POLICIES.md (regeneration is the only edit path)",
        "",
    ]
    lines.append(f"{'id':<16} {'kind':<22} command(s)")
    for p in policies:
        if p.subclause_of:
            continue
        kind = "NOT MACHINE-DECIDED" if p.kind == "review-only" else p.kind
        cmds = "; ".join(c.source_token for c in p.commands) or "NONE"
        lines.append(f"{p.id:<16} {kind:<22} {cmds}")
        for s in policies:
            if s.subclause_of == p.id:
                skind = "NOT MACHINE-DECIDED" if s.kind == "review-only" else s.kind
                scmds = "; ".join(c.source_token for c in s.commands) or "NONE"
                lines.append(f"  {s.id:<14} {skind:<22} {scmds}")
    n_top = sum(1 for p in policies if not p.subclause_of)
    n_sub = sum(1 for p in policies if p.subclause_of)
    lines.append("")
    lines.append(
        f"rows: {n_top} policies ({n_sub} sub-clauses nested) -- reconciled: parents + table_only == table rows"
    )
    return "\n".join(lines)


# The committed review copy (spec assumption: an uncommitted artifact cannot be
# reviewed). Derivation stays the source of truth: --assert byte-compares this
# file against a fresh rendering, and --manifest --write is the ONLY edit path.
MANIFEST_JSON = REPO / "tools" / "policy_manifest.json"


def committed_payload(policies: list[Policy], drift: dict) -> str:
    return (
        json.dumps(
            {
                "_derived_from": "docs/POLICIES.md -- regenerate with: python -m snn_data_lake.substrate.policy_manifest --manifest --write",
                "policies": json.loads(to_json(policies)),
                "drift": drift,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def to_json(policies: list[Policy]) -> str:
    return (
        json.dumps(
            [
                {
                    "id": p.id,
                    "title": p.title,
                    "kind": p.kind,
                    "commands": [
                        {"argv": c.argv[1:], "source": c.source_token} for c in p.commands
                    ],
                    "subclause_of": p.subclause_of,
                    "table_only": p.table_only,
                }
                for p in policies
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# --------------------------------------------------------------------------- #
# Audit + certification (US2)                                                 #
# distinct-from: gradient_claim_gate.py -- lexical docstring overlap only; that gate censuses gradient claims, this executes policy-gate commands and certifies
# distinct-from: self_improve.py -- returns-(bool,reason) shape only, no relation
# distinct-from: metacognitive_monitor.py -- comparison-runner shape only, no relation
# --------------------------------------------------------------------------- #

CERTIFICATION_JSON = REPO / "tools" / "policy_certification.json"

# Drift-ratchet baseline for --assert: the MEASURED set from the first counted
# run (T012, recorded verbatim in specs/021-policy-manifest-audit/
# manifest-baseline.md, 2026-08-18) -- never a number from spec prose. US4
# (T028) claims all three in POLICIES.md's enforcement table and ratchets this
# to frozenset(). The set may SHRINK, never grow.
# Ratcheted to EMPTY 2026-08-18 (T028): all three counted members claimed in the
# enforcement table's hygiene rows. The set may only stay empty or shrink further.
BASELINE_UNCLAIMED = frozenset()


def execute(argv: list[str]) -> tuple[str, str, float]:
    """Run one gate command. Returns (verdict, detail, seconds). CANNOT_RUN
    (missing/unlaunchable) is DISTINCT from FAIL (ran and exited nonzero) --
    collapsing those is a failure this project has already had."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=600)
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as e:
        return "CANNOT_RUN", f"{' '.join(argv)} -- {type(e).__name__}: {e}", time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return "CANNOT_RUN", f"{' '.join(argv)} -- timeout after 600s", time.monotonic() - t0
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        last = tail[-1] if tail else "<no output>"
        return "FAIL", f"{' '.join(argv)} -- exit {proc.returncode}: {last}", dt
    return "PASS", "", dt


def run_audit(policies: list[Policy]) -> tuple[list[AuditResult], list[dict]]:
    """Every row's verdict, per data-model.md. REVIEW_ONLY is neutral-listed
    (no machine decides it); NO_GATE is an absolute/ratchet/partial row with no
    command (FR-004's predicate). Subprocess execution only -- never import."""
    results: list[AuditResult] = []
    commands_run: list[dict] = []
    for p in policies:
        if p.kind == "review-only":
            results.append(AuditResult(p.id, "REVIEW_ONLY", "no machine decides this policy"))
            continue
        if not p.commands:
            results.append(AuditResult(p.id, "NO_GATE", f"kind {p.kind} but no runnable command"))
            continue
        worst = "PASS"
        detail = ""
        for c in p.commands:
            verdict, vdetail, secs = execute(c.argv)
            commands_run.append({"argv": c.argv[1:], "verdict": verdict, "seconds": round(secs, 2)})
            if verdict != "PASS" and worst == "PASS":
                worst, detail = verdict, vdetail
            elif verdict == "CANNOT_RUN" and worst == "FAIL":
                worst, detail = verdict, vdetail
        results.append(AuditResult(p.id, worst, detail))
    return results, commands_run


def render_audit(results: list[AuditResult], drift: dict) -> str:
    lines = ["POLICY AUDIT -- every manifest row's command executed (subprocess, never import)", ""]
    for r in results:
        lines.append(f"{r.policy_id:<16} {r.verdict:<12} {r.detail}".rstrip())
    lines.append("")
    lines.append(f"UNCLAIMED_GATE: {', '.join(drift['unclaimed_gates']) or 'none'}")
    lines.append(f"NOT_IN_CI:      {', '.join(drift['not_in_ci']) or 'none'}")
    lines.append(drift["self_disposition"])
    return "\n".join(lines)


def decide_certified(results: list[AuditResult], drift: dict) -> tuple[bool, list[str]]:
    """Contract: certified iff zero FAIL / CANNOT_RUN / NO_GATE, zero
    unclaimed-gate drift, derivation reconciled (already true if we got here).
    REVIEW_ONLY never blocks -- it is disclosed. Returns (certified, reasons)."""
    reasons = [
        f"{r.policy_id}: {r.verdict} -- {r.detail}"
        for r in results
        if r.verdict in {"FAIL", "CANNOT_RUN", "NO_GATE"}
    ]
    reasons += [f"UNCLAIMED_GATE: {g}" for g in drift["unclaimed_gates"]]
    return (not reasons), reasons


def _tree_state() -> dict:
    def _git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60
            )
            return out.stdout.strip()
        except OSError:
            return "<git unavailable>"

    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def certify(policies: list[Policy], full: bool) -> int:
    drift = census(policies)
    results, commands_run = run_audit(policies)
    if full:
        for argv in (
            [sys.executable, "run_selfchecks.py"],
            [sys.executable, "-m", "pytest", "-q"],
            [sys.executable, "-m", "ruff", "check", "."],
        ):
            verdict, detail, secs = execute(argv)
            commands_run.append({"argv": argv[1:], "verdict": verdict, "seconds": round(secs, 2)})
            if verdict != "PASS":
                results.append(AuditResult("suite:" + argv[-1], verdict, detail))
    certified, reasons = decide_certified(results, drift)
    record = {
        "tree": _tree_state(),
        "certified": certified,
        "full": full,
        "results": [
            {"policy": r.policy_id, "verdict": r.verdict, "detail": r.detail} for r in results
        ],
        "review_only": [r.policy_id for r in results if r.verdict == "REVIEW_ONLY"],
        "commands_run": commands_run,
        "refusal_reasons": reasons,
    }
    CERTIFICATION_JSON.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(render_audit(results, drift))
    print()
    if certified:
        print(f"CERTIFIED (full={full}) -- record: {_rel(CERTIFICATION_JSON)}")
        print(f"not machine-decided (disclosed, never green): {', '.join(record['review_only'])}")
        return 0
    print("REFUSED TO CERTIFY:")
    for r in reasons:
        print(f"  {r}")
    print(f"record written with certified: false -- {_rel(CERTIFICATION_JSON)}")
    return 1


def run_assert() -> int:
    """CI mode. Asserts the MAP is honest, not the territory green: derivation
    reconciles, the committed copy matches a fresh derivation byte-for-byte,
    and drift never worsens vs the counted baseline. Executes no gate command
    (CI already runs them; running them twice buys no information)."""
    try:
        policies = cross_check(derive())
    except DerivationError as e:
        print(f"POLICY_MANIFEST ASSERT FAILED (derivation): {e}")
        return 1
    drift = census(policies)
    fresh = committed_payload(policies, drift)
    if not MANIFEST_JSON.exists():
        print(
            f"POLICY_MANIFEST ASSERT FAILED: {MANIFEST_JSON.relative_to(REPO)} missing -- run --manifest --write"
        )
        return 1
    if MANIFEST_JSON.read_text(encoding="utf-8") != fresh:
        print(
            "POLICY_MANIFEST ASSERT FAILED: committed manifest != fresh derivation -- "
            "run --manifest --write and commit"
        )
        return 1
    grown = set(drift["unclaimed_gates"]) - BASELINE_UNCLAIMED
    if grown:
        print(
            f"POLICY_MANIFEST ASSERT FAILED: NEW unclaimed gate(s) {sorted(grown)} -- "
            "claim in POLICIES.md or baseline with a stated reason"
        )
        return 1
    if drift["not_in_ci"]:
        print(
            f"POLICY_MANIFEST ASSERT FAILED: claimed command(s) not CI-reachable: {drift['not_in_ci']}"
        )
        return 1
    print(
        f"policy_manifest assert OK -- rows reconciled, committed copy in sync, "
        f"unclaimed {len(drift['unclaimed_gates'])} (baseline {len(BASELINE_UNCLAIMED)}, may shrink never grow)"
    )
    print(drift["self_disposition"])
    return 0


def self_check() -> int:
    """Verify-the-verifier (SI 5): every verdict's red is CONSTRUCTED and must
    fire. Exit 0 only when every red is reachable."""
    failures = []

    verdict, _, _ = execute([sys.executable, "-c", "raise SystemExit(1)"])
    if verdict != "FAIL":
        failures.append(f"constructed failing command -> {verdict}, wanted FAIL")

    verdict, _, _ = execute(["definitely_missing_executable_xyz_021"])
    if verdict != "CANNOT_RUN":
        failures.append(f"constructed missing executable -> {verdict}, wanted CANNOT_RUN")

    ghost = Policy(id="PX", title="fixture", kind="absolute")
    results, _ = run_audit([ghost])
    if results[0].verdict != "NO_GATE":
        failures.append(
            f"constructed command-less absolute policy -> {results[0].verdict}, wanted NO_GATE"
        )

    try:
        parse("# empty\n")
        failures.append("empty walk did NOT raise")
    except DerivationError:
        pass

    real = POLICIES.read_text(encoding="utf-8")
    broken = "\n".join(line for line in real.splitlines() if not line.startswith("| **P6"))
    try:
        parse(broken)
        failures.append("reconciliation mismatch did NOT raise")
    except DerivationError:
        pass

    # Certify-level red: a FAIL row must refuse certification and name itself.
    bad = AuditResult("PX", "FAIL", "constructed")
    certified, reasons = decide_certified([bad], {"unclaimed_gates": [], "not_in_ci": []})
    if certified or not any("PX" in r for r in reasons):
        failures.append("certify decision did not refuse/name a constructed FAIL row")

    if failures:
        print("SELF-CHECK FAILED -- unreachable red(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        "policy_manifest self-check OK -- FAIL, CANNOT_RUN, NO_GATE, empty-walk, "
        "reconciliation and certify reds all reachable"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="derive the policy manifest; audit the tree against it"
    )
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ap.add_argument("--self-check", dest="self_check", action="store_true")
    args = ap.parse_args(argv)

    if args.manifest:
        try:
            policies = cross_check(derive())
        except DerivationError as e:
            print(f"DERIVATION FAILED: {e}")
            return 1
        drift = census(policies)
        if args.write:
            MANIFEST_JSON.write_text(
                committed_payload(policies, drift), encoding="utf-8", newline="\n"
            )
            print(f"wrote {MANIFEST_JSON.relative_to(REPO)} (regeneration is the only edit path)")
            return 0
        if args.json:
            print(committed_payload(policies, drift), end="")
        else:
            print(render(policies))
            print()
            print(f"UNCLAIMED_GATE: {', '.join(drift['unclaimed_gates']) or 'none'}")
            print(f"NOT_IN_CI:      {', '.join(drift['not_in_ci']) or 'none'}")
            print(drift["self_disposition"])
        return 0

    if args.audit or args.certify:
        try:
            policies = cross_check(derive())
        except DerivationError as e:
            print(f"DERIVATION FAILED: {e}")
            return 1
        if args.certify:
            return certify(policies, full=args.full)
        drift = census(policies)
        results, _ = run_audit(policies)
        print(render_audit(results, drift))
        # --audit reds on FAIL/CANNOT_RUN only; NO_GATE and REVIEW_ONLY print
        # honest and redden --certify instead (contract).
        return 1 if any(r.verdict in {"FAIL", "CANNOT_RUN"} for r in results) else 0

    if args.self_check:
        return self_check()

    # --assert, and the BARE invocation (what the selfcheck runner executes):
    # bare == --assert, the sibling-gate convention.
    return run_assert()


if __name__ == "__main__":
    raise SystemExit(main())
