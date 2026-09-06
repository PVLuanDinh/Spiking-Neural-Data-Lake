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
snapshots (83,405 records) that nothing ever reclaims -- MEASURED 2026-08-05, and NO LONGER THE
SHAPE OF THE LAKE. Re-censused read-only 2026-09-06 (specs/022 L6): 4,753,532 records, of which
_T_GROWTH is 86,095 (1.81%) and _T_COALITION 3,812,834 (80.21%). Both numbers were true when
taken; the 2026-08-05 one is kept because it is why this instrument exists, and dated because a
ratio without its date reads as current. Four declared classes -- 0x02, 0x06, 0x09, 0x0b -- held
ZERO records at the 2026-09-06 census.

Tag labels (keyed to home.py's _T_* constants BY SYMBOL, same census as
lake_migrate.TAG_DISPOSITIONS):
  IMMORTAL     _T_FACT, _T_EPISODE, _T_SELFBUILD, _T_SPIKE, _T_CALIB, _T_STATE, and
               _T_SYNDELTA (0x09, SynapseDelta below -- every record load-bearing by
               construction, no FULL kind to compact against yet; T010a)
  COMPACTABLE  _T_GROWTH (legacy snapshots, closed to npz writes since T010-ii) and
               _T_GROWTH_DELTA (the chain) -- both `reclaimable=True` since the T010-ii
               reconstruction proof landed (tests/test_growth_delta_chain.py: chain replay
               reconstructs the substrate and REFUSES tampered links); the flag records
               that proof, and each row's reason records what a reclaimer still owes.

Bare invocation (run_selfchecks): registry completeness vs lake_migrate's disposition table
(every dispositioned tag owns a retention label -- one census, two views, never divergent),
budget wiring on a constructed store, and the FLOOD red with its calm positive control.
"""

from __future__ import annotations

import json
import struct
import sys
import threading
from pathlib import Path

from snn_data_lake.substrate.lake_migrate import TAG_DISPOSITIONS

REPO = Path(__file__).resolve().parents[3]
APPEND_BASELINE = REPO / "tools" / "lake_append_baseline.json"
FLOOD_FACTOR = 8.0  # a class appending >8x its recorded per-tick p95 is flooding, not learning

#: class label -> (retention, reclaimable_today, reason)
RETENTION: dict[int, tuple[str, bool, str]] = {
    0x01: (
        "IMMORTAL",
        False,
        "_T_FACT: learned knowledge is never reclaimed. THE PER-RECORD EXEMPTION, stated here because "
        "the label is per TAG while the one reclaimer decides per RECORD, and that gap is how the two "
        "drifted apart (specs/022 L6, 2026-09-06): a fact whose `src` names a bulk re-derivable ingest "
        "(lake_consolidation.RE_DERIVABLE_SRC -- WordNet/ConceptNet and the like) MAY be reclaimed, "
        "because it reconstructs at boot from that same source, which is the COMPACTABLE contract "
        "applied per record. Everything else under this tag is LEARNED and is never dropped: the "
        "sleep-tick cap that used to drop it is refused by `LakeConsolidation.consolidate`, and "
        "`KnowledgeLog.consolidate` refuses a rewrite that would lose one. Measured when the exemption "
        "landed: 26,037 facts in the live lake, ALL learned, ZERO re-derivable -- so the cap had never "
        "bound and no learned record is known to have been lost. Append budget: UNBASELINED (reported, "
        "never trips) -- this tag has no row in tools/lake_append_baseline.json, so removing the cap "
        "leaves the learned class with NO live bound, which L4's first AppendBudget caller does not "
        "supply either (it is scoped to the _T_STATE family). Disclosed rather than implied, after the "
        "2026-09-06 panel measured the claim that the budget replaced the cap and refuted it",
    ),
    0x02: ("IMMORTAL", False, "_T_SPIKE: AER exemplars are evidence"),
    0x03: (
        "COMPACTABLE",
        True,
        "_T_GROWTH whole-substrate snapshots: 97.64% of the lake, the measured pathology. The "
        "LEGACY class -- closed to npz writes since T010-ii (new saves land under 0x08). "
        "reclaimable flipped True by the landed reconstruction proof "
        "(tests/test_growth_delta_chain.py); a reclaimer still owes per-record proof of "
        "state-identity with a retained neighbor (the chain writer's member-hash census is "
        "that instrument) before dropping anything -- the flag licenses reclaim, it never "
        "performs it",
    ),
    0x04: ("IMMORTAL", False, "_T_EPISODE: lived experience is never reclaimed"),
    0x05: ("IMMORTAL", False, "_T_SELFBUILD: the self-modification archive is the rollback"),
    0x06: ("IMMORTAL", False, "_T_CALIB: calibration provenance"),
    0x07: (
        "IMMORTAL",
        False,
        "_T_STATE: name-keyed state frames (upserted by replay; small, except the coverage ledger, which "
        "scales with the body's workspace -- T038). THE PER-RECORD EXEMPTION, stated here on the same "
        "grounds 0x01's is (specs/022 L6 fold, 2026-09-06, after the adversarial panel measured this class "
        "being reclaimed on the live lake while the label said otherwise): a SUPERSEDED frame -- any record "
        "under a name that a later record replaces -- MAY be dropped, because no read path can ever return "
        "it again (`load_named_state` and `_state_index` both take the LAST record per name), so dropping "
        "it loses no state. The CURRENT frame for every name is never dropped, and "
        "`home._prune_state_history` reads this label and refuses if the exemption is ever removed",
    ),
    0x08: (
        "COMPACTABLE",
        True,
        "_T_GROWTH_DELTA chain records (specs/022 T010-ii): a prefix ending BEFORE the newest "
        "FULL is reconstructible from that FULL (replay resets there), so whole prefixes may be "
        "reclaimed -- never a link inside a chain, which the reader would REFUSE by hash. "
        "Append budget: baselined at T012 as promised (2,902 B/tick p95, scratch-brain method "
        "in tools/lake_append_baseline.json; 31% under the legacy 4,205 -- live refits "
        "genuinely rewrite W/b/beta, so member-granularity keeps them; per-synapse granularity "
        "is 0x09's job at the kernel block)",
    ),
    0x09: (
        "IMMORTAL",
        False,
        "_T_SYNDELTA spike-derived synapse deltas (SynapseDelta, specs/022 T010a): compare-drop "
        "means every record is load-bearing -- there are no redundant snapshots to reclaim, by "
        "construction -- and no FULL-snapshot kind exists yet, so there is no base to compact a "
        "prefix against. The label revisits with the kernel block's own proof (T013+). Append "
        "budget: unbaselined (reported, never trips) until a live writer exists to measure",
    ),
    0x0A: (
        "IMMORTAL",
        False,
        "_T_COALITION settled workspace coalitions (global_workspace.Coalition, specs/022 "
        "T031): the recall contract operates here -- Episode consolidation consumes sequences "
        "of these, so every record is lived experience (same class as _T_EPISODE). Append "
        "budget: BASELINED 2026-08-21 (p95 434 B/tick, the documented instrument over "
        "make_brain x120; sleep ticks emit nothing) -- the flood assert is live for this tag",
    ),
    0x0B: (
        "IMMORTAL",
        False,
        "_T_GOAL goal event rows (learning_progress.GoalStore, specs/022 T046a): the brain's "
        "own intentions and their LP/outcome history are never reclaimed -- the store's "
        "materialized view folds every row, so each is load-bearing. Append budget: "
        "unbaselined (goals are LOW-frequency, created deliberately, never per-tick; a "
        "baseline lands with the first live per-tick writer if one ever exists)",
    ),
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


# distinct-from: substrate/basic_components.py -- its Rule protocol COMPUTES a weight change dw
#   from local activity; this class PERSISTS already-computed weights as lake records.
# distinct-from: substrate/home.py -- the _T_GROWTH_DELTA chain diffs whole npz MEMBERS of the
#   grown-substrate blob; this diffs individual synapses inside a weight matrix, keyed by spikes.
# distinct-from: claustrum/claustrum.py -- detects cross-area spike coincidence windows.
# distinct-from: thalamus/global_workspace.py -- formats an ASCII trace of one tick.
# distinct-from: basal_ganglia/self_build_loop.py -- snapshots safety-envelope FILES for restore.
# All sites read.
class SynapseDelta:
    """Spike-derived synapse-weight delta records (specs/022 T010a) -- tag 0x09.

    THE DIRTY SET IS SPIKE-DERIVED: only synapses with a fired PRE or fired POST endpoint are
    candidates this tick, which is the locality every plasticity rule in this tree already obeys
    (e-prop/STDP move weights only where activity was). Candidates are QUANTIZED to `q_step` and an
    entry is emitted ONLY when its quantized value differs from what the lake last saw
    (compare-drop). Two consequences, both deliberate:
      * a byte-frozen substrate whose endpoints keep firing emits NOTHING -- that is the measured
        97.64%-repeated-snapshot pathology (83,405 _T_GROWTH records) refused at the door, and the
        constructed red in main() reproduces exactly that shape;
      * a synapse that changes while its endpoints are silent is captured LATER, on the first tick
        an endpoint fires -- deferred, never lost, because the compare is against what was WRITTEN,
        not against the previous tick.

    `q_step` has NO default (P1): the quantum below which a weight change is noise is a property of
    the caller's learning rule, derived there, never a constant picked here. Reconstruction replays
    records onto a ZERO base (`apply`), so the first capture of a synapse carries its full
    quantized value.
    """

    TAG = 0x09
    _VER = 1

    def __init__(self, q_step):
        if not q_step or q_step <= 0:
            raise ValueError(
                "q_step must be a positive weight quantum DERIVED BY THE CALLER from its own "
                "learning rule (P1: this class refuses to pick one)"
            )
        self.q_step = float(q_step)
        self._last: dict[tuple[int, int], int] = {}

    def collect(self, w, pre_fired, post_fired):
        """One tick's record: dirty set -> quantize -> compare-drop -> packed bytes, or None when
        nothing the lake would learn from changed (writing nothing IS the mechanism)."""
        import numpy as np

        w = np.asarray(w)
        if w.ndim != 2:
            raise ValueError("w must be a 2-D [pre, post] weight array")
        n_pre, n_post = w.shape
        pre = sorted({int(i) for i in pre_fired})
        post = sorted({int(j) for j in post_fired})
        if (pre and (pre[0] < 0 or pre[-1] >= n_pre)) or (
            post and (post[0] < 0 or post[-1] >= n_post)
        ):
            raise ValueError("fired index out of range for w")
        entries, seen = [], set()
        for i in pre:
            row = np.rint(w[i] / self.q_step).astype(np.int64)
            for j in range(n_post):
                q = int(row[j])
                if self._last.get((i, j), 0) != q:
                    entries.append((i, j, q))
                    seen.add((i, j))
        for j in post:
            col = np.rint(w[:, j] / self.q_step).astype(np.int64)
            for i in range(n_pre):
                if (i, j) in seen:
                    continue
                q = int(col[i])
                if self._last.get((i, j), 0) != q:
                    entries.append((i, j, q))
        if not entries:
            return None
        for i, j, q in entries:
            self._last[(i, j)] = q
        out = [struct.pack("<BI", self._VER, len(entries))]
        out += [struct.pack("<IIq", i, j, q) for i, j, q in entries]
        return b"".join(out)

    @classmethod
    def entries(cls, payload):
        """-> [(pre, post, q)]. Refuses a malformed payload WHOLE (wrong version, short, trailing
        bytes) -- the same never-half-read rule every lake parser here follows."""
        try:
            ver, n = struct.unpack_from("<BI", payload, 0)
        except struct.error as exc:
            raise ValueError(f"SynapseDelta payload too short for its header: {exc}") from exc
        if ver != cls._VER:
            raise ValueError(f"SynapseDelta version {ver} is not {cls._VER}")
        if len(payload) != 5 + n * 16:
            raise ValueError(
                f"SynapseDelta length mismatch: {len(payload)} B for {n} entries -- refused whole"
            )
        return [struct.unpack_from("<IIq", payload, 5 + k * 16) for k in range(n)]

    def apply(self, payload, w):
        """Apply one record to `w` in place (zero-base reconstruction). Returns the entry count."""
        got = self.entries(payload)
        for i, j, q in got:
            w[i, j] = q * self.q_step
        return len(got)


# : How many of a tag's own WRITING ticks the running bar is computed over. Sized from what the bar
# needs --
# : a p95 is meaningless below ~20 samples -- and kept small enough that the window tracks a brain
# whose
# : frames grow: at the measured cadence (67 writing ticks in 1200) this is roughly the last 500
# ticks.
# : DISCLOSED, and the reason it is not larger: a longer window is a slower-staling fixed ceiling,
# which is
#: the defect this replaces.
RUNNING_WINDOW = 32


class AppendBudget:
    """Per-tag bytes/tick accounting with a flood assert against the tag's OWN RECENT rate.

    WHY NOT THE RECORDED BASELINE (specs/022 L4b, 2026-09-06). It was a fixed p95 measured once, and the
    quantity it bounds is not stationary: measured over 1200 ticks, per-tick 0x07 bytes climb monotonically
    52,763 -> 77,998 B with no settling, because _T_STATE carries WHOLE FRAMES that grow with the brain
    (growing_learners.json x2.3 over the run, coverage_cells x18.7). The live daemon duly reported 7 floods
    in 8 nights at 160,470 B against a 12,381 B ceiling recorded in August -- a stuck alarm, not a
    detection, and a stuck alarm is worse than none because it teaches its reader to ignore the real one.
    P1 names the fix: running-baseline-relative, never a magic constant held fixed.

    WHAT THIS DOES NOT DO. It does not detect a RAMP -- a class growing without bound over a brain's life,
    which is the pathology the module docstring names. A flood assert never detected that (a fixed p95 fires
    once, forever, and says nothing about the slope), and this one does not either. Two pathologies, and
    conflating them is what produced the stuck alarm.
    """

    def __init__(self, baseline: dict | None = None):
        if baseline is None:
            baseline = json.loads(APPEND_BASELINE.read_text()) if APPEND_BASELINE.exists() else {}
        # The recorded numbers PRE-FILL each tag's window rather than standing as a permanent
        # ceiling.
        # Day-one protection is preserved -- a tag is bounded from its first tick -- and the window
        # then
        # MIGRATES to what the brain actually writes as real ticks push the seed out, so a stale
        # recording
        # self-corrects within a few writes instead of alarming forever. Measured migration cost is
        # in the
        # self-check below and in evidence/l4b-append-ceiling-nonstationary.md.
        self._p95 = {int(k): float(v) for k, v in baseline.get("per_tag_p95_bytes", {}).items()}
        self._recent: dict[int, list[int]] = {
            t: [int(v)] * RUNNING_WINDOW for t, v in self._p95.items()
        }
        self._seed_vals = {t: (float(v),) for t, v in self._p95.items()}
        self._tick_bytes: dict[int, int] = {}
        # THIS PROJECT RUNS A FREE-THREADING INTERPRETER, so read-modify-write on a dict is not
        # atomic.
        # Measured by an adversarial panel (2026-09-06): 8 threads x 20,000 appends counted 36,431
        # of
        # 160,000 -- 78% lost -- and end_tick could raise dict-changed-size out of the caller's
        # tick.
        self._lock = threading.Lock()

    def on_append(self, tag: int, n_bytes: int) -> None:
        with self._lock:
            self._tick_bytes[tag] = self._tick_bytes.get(tag, 0) + int(n_bytes)

    @staticmethod
    def _p95_of(xs) -> float:
        s = sorted(xs)
        return float(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))])

    def bar(self, tag: int):
        """(bar_bytes, source) for `tag`, or (None, "unbaselined"). The tag's own recent writing ticks once
        it has enough of them; the recorded seed until then; nothing if it has neither."""
        recent = self._recent.get(tag)
        if not recent:
            return None, "unbaselined"
        seeded = sum(1 for x in recent if float(x) in self._seed_vals.get(tag, ()))
        src = "its own recent rate" if seeded == 0 else f"a window {seeded}/{len(recent)} seeded"
        return self._p95_of(recent) * FLOOD_FACTOR, src

    def end_tick(self) -> list[str]:
        """Close the tick; return violations (empty = green).

        A tag with neither a window nor a recorded seed never trips -- a budget nobody measured is not a
        budget, and the caller REPORTS those instead. The tick's bytes join the tag's window AFTER the
        comparison, so a tick is always judged against history rather than against itself."""
        violations = []
        with self._lock:  # swap out, then scan outside the lock
            taken, self._tick_bytes = self._tick_bytes, {}
        for tag, got in taken.items():
            bar, source = self.bar(tag)
            if bar is not None and got > bar:
                retention, _, _ = retention_of(tag)
                violations.append(
                    f"FLOOD: tag {tag:#04x} ({retention}) appended {got} B this tick "
                    f"> {FLOOD_FACTOR}x {source} ({bar / FLOOD_FACTOR:.0f} B)"
                )
            if (
                got > 0
            ):  # only WRITING ticks enter the window; silent ticks would drag the bar to zero
                w = self._recent.setdefault(tag, [])
                w.append(int(got))
                del w[:-RUNNING_WINDOW]
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
    # ---- L4b (2026-09-06): the bar is the tag's OWN RECENT RATE, so it owes three more arms
    # ----------
    # CONTROL 1 -- ORDINARY GROWTH MUST NOT FLAG. This replays the MEASURED series from a 1200-tick
    # brain:
    # per-tick 0x07 bytes climbing 52,763 -> 77,998 B over 67 writing ticks, carried by whole frames
    # that
    # grow with the brain. PRECISELY: this series tops out UNDER the retired fixed bar (99,048 B),
    # so it is
    # the no-false-alarm control; what crossed that bar was the LIVE daemon at 160,470 B/tick,
    # further along
    # the same monotone curve, and the migration arm below is where the two comparisons actually
    # diverge.
    grow = [round(52763 * (77998 / 52763) ** (i / 66)) for i in range(67)]
    gb = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    grow_flags = 0
    for n in grow:
        gb.on_append(0x07, n)
        grow_flags += len(gb.end_tick())
    if grow_flags:
        print(f"lake_engine self-check FAILED: ordinary growth flagged {grow_flags} times")
        return 1
    # RED -- a real spike, AFTER the window is all-real, must still flag.
    sb = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    for n in grow[:RUNNING_WINDOW]:
        sb.on_append(0x07, n)
        sb.end_tick()
    sb.on_append(0x07, int(max(grow[:RUNNING_WINDOW]) * FLOOD_FACTOR * 2))
    if not sb.end_tick():
        print("lake_engine self-check FAILED: a real spike did not trip the running bar")
        return 1
    # CONTROL 3 -- THE COST, MEASURED AND PRINTED RATHER THAN HIDDEN. A window that adapts is a
    # window a
    # SUSTAINED writer can walk up: a stale seed stops alarming after a few real ticks (which is the
    # point,
    # and what ended the live daemon's stuck alarm) and by exactly the same mechanism a genuine
    # sustained
    # flood goes quiet after the same few. That is the trade of any self-relative bar, and detecting
    # a
    # sustained RAMP is a slope instrument's job, which this is not and does not claim to be.
    mb = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    sustained = 0
    for _ in range(40):
        mb.on_append(0x07, 160470)  # the live daemon's measured per-tick rate
        if mb.end_tick():
            sustained += 1
    if not 1 <= sustained <= RUNNING_WINDOW // 4:
        print(f"lake_engine self-check FAILED: migration took {sustained} ticks, expected 1..8")
        return 1
    print(
        f"  append budget: ordinary growth (52,763 -> 77,998 B over {len(grow)} writes) flags 0; a real "
        f"spike still flags; a STALE seed costs {sustained} flood(s) before the window migrates -- and a "
        f"SUSTAINED flood goes quiet after the same {sustained}, which is the disclosed cost of an "
        "adaptive bar (a RAMP needs a slope instrument, not this one)"
    )
    # SynapseDelta (T010a): dirty-set wiring, compare-drop, the constructed pathology red, and a
    # zero-base reconstruction -- the cheap halves; full proofs live in tests/test_synapse_delta.py.
    import numpy as np

    sd = SynapseDelta(q_step=0.01)
    w = np.zeros((4, 4))
    w[0, 1] = 0.5
    w[3, 3] = 0.25
    recs = [sd.collect(w, [0], [])]
    touched = [(i, j) for i, j, _q in SynapseDelta.entries(recs[0])]
    if touched != [(0, 1)]:
        print(f"lake_engine self-check FAILED: dirty set not spike-derived ({touched})")
        return 1
    # The constructed 97.64% red: a frozen substrate whose endpoints keep firing must emit NOTHING
    # (a naive per-tick snapshot of the same dirty set would emit 50 ticks x 4 entries x 16 B).
    frozen = sum(len(r) for r in (sd.collect(w, [0], []) for _ in range(50)) if r)
    if frozen:
        print(f"lake_engine self-check FAILED: frozen substrate emitted {frozen} B (the pathology)")
        return 1
    rec2 = sd.collect(w, [3], [])  # (3,3) changed OUTSIDE the dirty set above -> captured NOW
    if rec2 is None or SynapseDelta.entries(rec2) != [(3, 3, 25)]:
        print("lake_engine self-check FAILED: deferred capture lost an out-of-dirty-set change")
        return 1
    recs.append(rec2)
    twin, sd_replay = np.zeros_like(w), SynapseDelta(q_step=0.01)
    for r in recs:
        sd_replay.apply(r, twin)
    if not np.array_equal(twin, np.rint(w / 0.01) * 0.01):
        print("lake_engine self-check FAILED: zero-base reconstruction diverged")
        return 1
    baseline_state = (
        "recorded"
        if APPEND_BASELINE.exists()
        else "NOT RECORDED (record on the pinned host before the assert binds)"
    )
    print(
        f"lake_engine self-check OK: {msg}; flood red fired; append baseline {baseline_state}; "
        "SynapseDelta dirty-set/compare-drop/deferred-capture/reconstruction OK, frozen-substrate "
        "red emitted 0 B"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
