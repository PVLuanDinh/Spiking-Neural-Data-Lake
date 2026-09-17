"""X1's bookkeeping (specs/022 plan Phase 3, task T088, red X-R1; ratified by the owner 2026-09-06): the
scale target is a CAPABILITY, not a width -- and no capability claim leaves this project without its protocol and
its provenance.

WHY. T065 derives a width from measured state-per-unit; nothing in the plan said what the brain must be able to DO
at it, so "superhuman" had no referent and no red. This module is the registry that gives it one, plus the check
that refuses a claim without its measurement. It sets no target: a target is a row the owner writes.

THE CONTRACT. `CAPABILITY_BAR` below holds one row per claim. A row is COMPLETE when the brain's number carries a
protocol and a provenance, and the human number carries a provenance that either records a measurement on this host
(with the protocol recorded beside it) or QUOTES the cited protocol's own text. A row whose human number cannot be
obtained under a matching protocol is written with `human=None` and the REFUSED verdict -- and then no comparison
word may be used for that task anywhere a reader can see, which is the honest form of "superhuman is not a claim
this project may make on that task".

READ STATICALLY, like `GRADIENT_CLAIM`: the check parses this file's literal dict with `ast` and never imports it,
so a registry that cannot be imported still cannot hide from its own check.

SCOPE, disclosed. A complete row is a fact about ONE task under ONE protocol, never a statement about general
capability; the word "superhuman" outside a registry row is exactly what this catches.

WHAT IT DOES NOT DO. It does not measure the brain, choose the task, or set the percentage: X1 does that, with the
pre-sized twin as its control. This is the bookkeeping that makes X1's claim checkable.

Bare invocation (`python -m snn_data_lake.substrate.capability_bar`, also run by run_selfchecks): the four
constructed reds, their positive control, and the count of today's registry.
"""

# Checked before writing (graph-exists: "a registry of measured capability numbers with their protocol and
# provenance, a human baseline comparison" -> NO EXISTING IMPLEMENTATION). The duplicate-capability guard's term
# matches were READ; none is a second mechanism for this job:
# distinct-from: online_readout.py -- an online LEARNING RULE on the grown readout; shares the word "readout".
# distinct-from: world_red_probe.py -- W3's world-red instrument PRODUCES numbers from a stream; this registry
#   states what a number may be compared to, and refuses the comparison when it may not. Site read.
# distinct-from: laminar_local_readout.py -- a readout network whose main() constructs its own reds; the shared
#   phrase is P8's red-construction idiom, not a capability claim. Site read.
# distinct-from: r_stdp.py -- a worker-count cap on an uncapped call. Term match only. Site read.
# distinct-from: language_readout_slda.py -- ONE train/test split shared by a readout and its baselines: a
#   within-task control, never a human comparison. Site read.

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SELF = Path(__file__).resolve()

#: Comparison words that may appear beside a task name ONLY when its row is complete.
COMPARISON_WORDS = (
    "superhuman",
    "above parity",
    "at parity",
    "beats a human",
    "human-level",
    "outperforms a human",
)

#: Surfaces a reader sees. A comparison word beside a registry task name here needs a complete row.
TRACKED_SURFACES = (
    "src/snn_data_lake/substrate/cockpit.py",
    "docs/PERFORMANCE.md",
    "docs/OVERVIEW.md",
    "docs/PITCH.md",
    "README.md",
)

#: task -> row. ONE ROW TODAY, and it is a REFUSED one: a measured number with the protocol it was
#: measured under, and no human number, so no capability claim has been earned by it either.
#: Row shape: (protocol, human, human_provenance, brain, brain_provenance, date, verdict)
#:   protocol           what was run, exactly, on both sides -- the same words for both, or it is not a comparison
#:   human              the human number, or None when none could be obtained under a matching protocol
#:   human_provenance   "measured here: <how, when>" or a cited source WITH the protocol's own words quoted
#:   brain              this brain's number, or None before it is measured
#:   brain_provenance   the evidence file and commit the number came from
#:   verdict            "REFUSED: ..." when human is None; else a comparison word or "below"
#
# 2026-09-17, C0 (specs/022, evidence/c0-everyday-battery.md 2.6, MET branch) -- the first row, and it is
# THE SESSION'S CALL under the owner's 2026-09-05 authorization ("every open owner call is the session's,
# disclosed per row"). This module's own text says a TARGET is a row the owner writes: this row sets NO
# target and states no relation between the two sides. It records one measured number with the protocol it
# was measured under, and writes the human side as None with the REFUSED verdict -- so from today the
# registry's own surface check refuses any of COMPARISON_WORDS standing beside this task name anywhere a
# reader can see. That refusal is the point of the row; the number is only what was measured.
CAPABILITY_BAR: dict[str, tuple] = {
    "everyday reasoning, 20 items (C0)": (
        "make_brain(seed=<s>, persist=False) under substrate.home.throwaway_home(), 20 warm-up ticks, "
        "the 4 controls then the 20 items through Brain.respond(q, human=True); the reply is recorded "
        "with any ' (somatic' suffix removed (the L12 audit's and the orientation's transcript rule)",
        None,
        "no human number exists under this protocol and none is claimed: 20 questions put through one "
        "chat lane and scored by a frozen regex table is not a thing any person has been measured on, "
        "here or in any source this project can cite.",
        0.0,
        "tools/c0_everyday_battery.json, campaign head 37135d7 (the pin committed at b167118), "
        "2026-09-17: 0 of 20 CORRECT on each of seeds 0-4; controls 20 of 20; "
        "14 ABSTAIN / 5 UNRELATED_ROW / 1 SELF_REPORT",
        "2026-09-17",
        "REFUSED: no human number under a matching protocol",
    )
}

_MIN_PROV = 24  # a provenance shorter than this is a label, not a provenance (CHOSEN, disclosed)
_QUOTE = chr(34)


def _registry_from_source(path: Path = SELF) -> dict[str, tuple]:
    """Parse CAPABILITY_BAR out of the FILE (never import it): the GRADIENT_CLAIM pattern."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "CAPABILITY_BAR" and node.value is not None:
                return ast.literal_eval(node.value)
    raise ValueError(f"{path}: no CAPABILITY_BAR literal found")


def findings(registry=None, surfaces=None) -> list[str]:
    """Every incompleteness in the registry, plus every surface line claiming more than a row supports."""
    reg = _registry_from_source() if registry is None else registry
    out: list[str] = []
    for task, row in sorted(reg.items()):
        if not (isinstance(row, tuple) and len(row) == 7):
            out.append(
                f"{task}: a row is (protocol, human, human_prov, brain, brain_prov, date, verdict)"
            )
            continue
        protocol, human, human_prov, brain, brain_prov, date, verdict = row
        if not (isinstance(protocol, str) and len(protocol) >= _MIN_PROV):
            out.append(f"{task}: no protocol (a comparison without one compares nothing)")
        if brain is not None and not (isinstance(brain_prov, str) and len(brain_prov) >= _MIN_PROV):
            out.append(f"{task}: a brain number with no provenance")
        if human is not None:
            ok = isinstance(human_prov, str) and (
                (human_prov.startswith("measured here:") and len(human_prov) >= _MIN_PROV)
                or (human_prov.startswith("cited:") and _QUOTE in human_prov)
            )
            if not ok:
                out.append(
                    f"{task}: a human number whose provenance neither records a measurement here "
                    "nor quotes the cited protocol"
                )
        if not isinstance(verdict, str) or not verdict:
            out.append(f"{task}: no verdict")
            continue
        refused = verdict.startswith("REFUSED")
        if human is None and not refused:
            out.append(f"{task}: no human number, so the only honest verdict is REFUSED: ...")
        comparative = any(w in verdict.lower() for w in ("parity", "superhuman", "human"))
        if not refused and comparative and (human is None or brain is None):
            out.append(f"{task}: a comparison verdict with a missing number")
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
            out.append(f"{task}: no date")
    src = surfaces
    if src is None:
        src = {}
        for rel in TRACKED_SURFACES:
            p = REPO / rel
            if p.exists():
                src[rel] = p.read_text(encoding="utf-8", errors="replace")
    for rel, text in sorted(src.items()):
        low = text.lower()
        for task, row in sorted(reg.items()):
            if task.lower() not in low:
                continue
            shaped = isinstance(row, tuple) and len(row) == 7
            complete = shaped and row[1] is not None and row[3] is not None
            refused = shaped and isinstance(row[6], str) and row[6].startswith("REFUSED")
            for line_no, line in enumerate(text.splitlines(), 1):
                ll = line.lower()
                if task.lower() in ll and any(w in ll for w in COMPARISON_WORDS):
                    if refused:
                        out.append(
                            f"{rel}:{line_no}: a comparison word for {task!r}, whose row is REFUSED "
                            "(no human number under a matching protocol)"
                        )
                    elif not complete:
                        out.append(
                            f"{rel}:{line_no}: a comparison word for {task!r} without a complete registry row"
                        )
    return out


def report() -> str:
    reg = _registry_from_source()
    lines = [f"CAPABILITY BAR -- {len(reg)} row(s); tracked surfaces: {len(TRACKED_SURFACES)}"]
    if not reg:
        lines.append(
            "  (empty: no capability claim has been earned. X1 writes the first row with its protocol,"
        )
        lines.append(
            "   its human number's provenance, and the pre-sized twin's comparison beside it.)"
        )
    for task, row in sorted(reg.items()):
        lines.append(f"  {task}: {row}")
    f = findings()
    lines.append(f"findings: {len(f)}")
    lines += [f"  - {x}" for x in f]
    return chr(10).join(lines)


def main() -> None:
    quoted = (
        "cited: a stand-in for this self-check -- " + _QUOTE + "the protocol's own words" + _QUOTE
    )
    complete = {
        "next-category on the live percept stream": (
            "800 wake ticks, the engine's held-out marginal-covering slice, accuracy on that slice",
            0.93,
            quoted,
            0.65,
            "evidence/l1c-online-readout.md section 3.4, commit deadbeef",
            "2026-09-06",
            "below",
        )
    }
    assert findings(complete, {}) == [], findings(complete, {})  # the positive control
    lp = "a protocol long enough to be a protocol"
    mh = "measured here: on this host, protocol recorded 2026-09-06"
    bp = "evidence/x.md, commit abc1234"
    reds = {
        "no protocol": {"t": ("short", 1.0, mh, 0.5, bp, "2026-09-06", "below")},
        "brain number without provenance": {"t": (lp, 1.0, mh, 0.5, "x", "2026-09-06", "below")},
        "human number without a quoted protocol": {
            "t": (lp, 1.0, "cited: some paper", 0.5, bp, "2026-09-06", "below")
        },
        "comparison verdict, missing number": {
            "t": (lp, None, "", 0.5, bp, "2026-09-06", "at parity")
        },
    }
    for name, reg in reds.items():
        assert findings(reg, {}), f"the constructed red {name!r} was not caught"
    refused = {
        "widget task": (
            lp,
            None,
            "",
            0.5,
            bp,
            "2026-09-06",
            "REFUSED: no human number under a matching protocol",
        )
    }
    assert findings(refused, {}) == [], findings(refused, {})
    surf = findings(refused, {"docs/PITCH.md": "the brain is superhuman on the widget task today"})
    assert surf and "REFUSED" in surf[0], surf
    print(report())
    print(
        f"capability_bar self-check OK: {len(reds)} constructed reds caught, the complete row and the REFUSED "
        "row pass, and a comparison word for a REFUSED task is caught on a surface"
    )


def _assert() -> int:
    f = findings()
    print(report())
    if f:
        print(f"CAPABILITY_BAR ASSERT FAILED: {len(f)} finding(s)", file=sys.stderr)
        return 1
    print(
        "check OK: every capability row carries its protocol and its provenance; no surface claims more"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_assert() if "--assert" in sys.argv else (main() or 0))
