"""Retention classes + the IMMORTAL append-budget instrument (specs/022 T010-i).

The retention rule, operational (lands in docs/POLICIES.md P4b in this same commit, per the
gate-before-rule coordination of T004): every record class in the ONE lake is labeled
IMMORTAL or COMPACTABLE at introduction, here, in RETENTION. "Remember everything" means an
appended record is either retained forever or provably reconstructible before any compaction
reclaims it -- COMPACTABLE is a claim about reconstruction, never permission to forget.

IMMORTAL classes carry an append BUDGET: bytes/tick measured against a recorded baseline
(tools/lake_append_baseline.json, the perf_baseline pattern -- record on the pinned host,
ratchet thereafter). The red is a constructed FLOOD: an append rate far above baseline must
trip the assert BEFORE disk pressure becomes the detector. The measured pathology this
instrument exists to prevent: 97.64% of the live lake is repeated whole-substrate _T_GROWTH
snapshots (83,405 records) that nothing ever reclaims.

Legacy-tag labels (keyed to home.py's _T_* constants BY SYMBOL, same census as
lake_migrate.TAG_DISPOSITIONS):
  IMMORTAL     _T_FACT, _T_EPISODE, _T_SELFBUILD, _T_SPIKE, _T_CALIB, _T_STATE
  COMPACTABLE  _T_GROWTH -- *conditionally*: reclaim is FORBIDDEN until the delta-chain
               replacement (T010-ii) proves reconstruction; the label records intent, the
               `reclaimable` flag records proof, and today it is False.

Bare invocation (run_selfchecks): registry completeness vs lake_migrate's disposition table
(every dispositioned tag owns a retention label -- one census, two views, never divergent),
budget wiring on a constructed store, and the FLOOD red with its calm positive control.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from snn_data_lake.substrate.lake_migrate import TAG_DISPOSITIONS

REPO = Path(__file__).resolve().parents[3]
APPEND_BASELINE = REPO / "tools" / "lake_append_baseline.json"
FLOOD_FACTOR = 8.0  # a class appending >8x its recorded per-tick p95 is flooding, not learning

#: class label -> (retention, reclaimable_today, reason)
RETENTION: dict[int, tuple[str, bool, str]] = {
    0x01: ("IMMORTAL", False, "_T_FACT: learned knowledge is never reclaimed"),
    0x02: ("IMMORTAL", False, "_T_SPIKE: AER exemplars are evidence"),
    0x03: (
        "COMPACTABLE",
        False,
        "_T_GROWTH whole-substrate snapshots: 97.64% of the lake, the measured pathology. "
        "Reclaim FORBIDDEN (reclaimable=False) until the T010-ii delta chain PROVES "
        "reconstruction -- the label is intent, the flag is proof",
    ),
    0x04: ("IMMORTAL", False, "_T_EPISODE: lived experience is never reclaimed"),
    0x05: ("IMMORTAL", False, "_T_SELFBUILD: the self-modification archive is the rollback"),
    0x06: ("IMMORTAL", False, "_T_CALIB: calibration provenance"),
    0x07: ("IMMORTAL", False, "_T_STATE: name-keyed config frames (tiny, upserted by replay)"),
}


def retention_of(tag: int) -> tuple[str, bool, str]:
    if tag not in RETENTION:
        raise KeyError(f"tag {tag:#04x} has no retention label -- label it at introduction (P4b)")
    return RETENTION[tag]


def registry_consistent() -> tuple[bool, str]:
    """One census, two views: every dispositioned tag owns a retention label and vice versa."""
    disp, ret = set(TAG_DISPOSITIONS), set(RETENTION)
    if disp != ret:
        return False, f"registry divergence: dispositions {sorted(disp)} vs retention {sorted(ret)}"
    return True, f"{len(ret)} classes labeled, dispositions and retention agree"


class AppendBudget:
    """Per-tag bytes/tick accounting with a flood assert against the recorded baseline."""

    def __init__(self, baseline: dict | None = None):
        if baseline is None:
            baseline = json.loads(APPEND_BASELINE.read_text()) if APPEND_BASELINE.exists() else {}
        self._p95 = {int(k): float(v) for k, v in baseline.get("per_tag_p95_bytes", {}).items()}
        self._tick_bytes: dict[int, int] = {}

    def on_append(self, tag: int, n_bytes: int) -> None:
        self._tick_bytes[tag] = self._tick_bytes.get(tag, 0) + int(n_bytes)

    def end_tick(self) -> list[str]:
        """Close the tick; return violations (empty = green). Unbaselined tags never trip --
        a budget nobody measured is not a budget (they are REPORTED by the caller instead)."""
        violations = []
        for tag, got in self._tick_bytes.items():
            p95 = self._p95.get(tag)
            if p95 is not None and got > p95 * FLOOD_FACTOR:
                retention, _, _ = retention_of(tag)
                violations.append(
                    f"FLOOD: tag {tag:#04x} ({retention}) appended {got} B this tick "
                    f"> {FLOOD_FACTOR}x its recorded p95 ({p95:.0f} B)"
                )
        self._tick_bytes.clear()
        return violations


def main() -> int:
    ok, msg = registry_consistent()
    if not ok:
        print(f"lake_engine self-check FAILED: {msg}")
        return 1
    # Constructed budget: calm control and the FLOOD red share one run.
    budget = AppendBudget(baseline={"per_tag_p95_bytes": {"1": 1000.0}})
    budget.on_append(0x01, 500)
    calm = budget.end_tick()
    budget.on_append(0x01, 9000)
    flood = budget.end_tick()
    if calm:
        print(f"lake_engine self-check FAILED: calm tick tripped the budget ({calm})")
        return 1
    if not flood:
        print("lake_engine self-check FAILED: the constructed flood did not trip")
        return 1
    baseline_state = (
        "recorded"
        if APPEND_BASELINE.exists()
        else "NOT RECORDED (record on the pinned host before the assert binds)"
    )
    print(f"lake_engine self-check OK: {msg}; flood red fired; append baseline {baseline_state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
