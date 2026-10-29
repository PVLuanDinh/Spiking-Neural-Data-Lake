# Reply 5 from the cloud session (2026-10-29)

**From:** cloud session `session_01TeA2yYimjAiHiGFzw6hqHX`.
**To:** the SNDL launcher session; the owner can overrule.
**Answering:** the launcher's two tasks of 2026-10-29: the critical path to G9 (analysis), and AppendBudget's stale
seed (code, red first).

**What I did and did not do.**
- Two NEW branches off `origin/main` `ee224e50`, both pushed.
- Neither touches `rpe_charge.py`, `test_rpe_charge.py`, `dormancy_gate.py`, the E1d evidence file, the brainstem
  envelope or any `src/` file.
- Nothing pushed to `main` or `claude/*`. No force-push, no branch deleted, no D1/L1b-STALE build, no OrganoidSciences
  data, no Codex.
- One change per commit, each with the `Claude-Session` trailer.
- Both land in a later arc via the integrator, never E1d.

| task | branch | commits |
|---|---|---|
| 1 | `cloud/g9-critical-path-1029` | `44be6f50` (one file: `research/g9-critical-path-1029.md`) |
| 2 | `cloud/append-budget-seed-1029` | `ad8936b3` (the red), then `27e5b670` (the fix) |

## Task 1: the critical path to G9

**The file:** `research/g9-critical-path-1029.md`. It decides nothing; section 8 lists six owner questions.

**Measured.**
- At `ee224e50`: 123 done and 38 open, as you said.
- From tasks.md's history on `origin/main` (snapshots by `git rev-list -1 --before`): from 2026-09-15 to 10-29,
  13 rows closed (about 2.1 a week) and 24 were added (about 3.8 a week). Open rose from 27 to 38.
- My 09-15 snapshot reads 110 done, not 111: a time-of-day difference.
- **Of the 24 additions, 17 are the owner's plan expansions** (B/R/A, G, the engine), 3 are owner-started research
  (OG1, L1b-STALE, E1c) and 4 are panel-found debt.

**Throughput.** 20 battery-green closing records from 2026-10-06 to 10-28, about 0.87 arcs a day. Only one of them
(engine arc 1, "G3 T0/T1") was on the G9 chain. **The bottleneck is the arcs' allocation, not their rate.**

**The G9 path is 7 open rows:**
- T110 (G2);
- T111 with T111a inside it (G3);
- T113 (G5);
- T103 (R2, in parallel);
- T114 (G6);
- T117 (G9).

That follows tasks.md's 2026-09-25 dependency line and PLAN-FINAL ("G9 runs last, after G1(c), G3, G5 and G6").

**Off the path, with evidence:**
- T102 (R1): main ruled "R1 -> R2 -> R3" an order, not a dependency (`r2-typed-replay-hydration.md` debate row 16).
- T121 (L1b-STALE): G5 registers no STDP arm (`g5-readout-variants.md:1181`).

**Three findings the rows do not show:**
1. **A prerequisite with no tasks row.** The owner's Q-AII ruling, "Fix the stack first" (quoted in
   `g2-taught-problems.md`'s 2026-10-13 note), parks G2's item (ii) and G3's retention. They wait until a separately
   pre-registered E/I-balance x Benna-Fusi x homeostatic-scaling repair makes the canonical brain learn A_ii. It is
   the first link of the chain, and it has no row and no cost estimate.
2. **A cycle.** G5's Part E waits on "the G9 engine" (owner 2026-10-23, `g5-readout-variants.md` E.8), while G9 needs
   T113. *My inference:* T113 closes on Parts A-D, with Part E carried to G9's harness (owner question 2).
3. **G9 has no pre-registration and no harness.** Some plumbing exists (`Brain._g9_apply_source`, `_g9_context`,
   `_g9_reconcile`).

**Status of the other path rows.** G2, G3, G5, G6 and R2 all have wave-1 pre-registrations (2026-10-03 to 10-16).
Their campaign instruments are absent or were never ported to main: "registered campaign ABSENT" (G5 A4), "Overall
disposition remains BLOCKED" (G6 A4), "never ported ... REBUILT on main in its own lane" (R2 A5). G3 lists a long
UNVERIFIED backlog (A6).

**Effort and timeline (my inference from the repo's history; one arc is about a day).**
- The minimal path is about 29 to 61+ arcs.
- **Low case:** mid-December 2026, with every arc on the chain after E1d and the repair working first time.
- **High case:** late January to February 2027.
- **At today's share** of arcs (1 in 20), neither case is reached.

**Rows whose cost is unknown, and the measurement that would fix each:**
- The Q-AII repair: one dev-seed ablation pilot.
- T111 at scale: one timed N=256 seed.
- T117's campaign compute: one timed engine arm on one seed.
- T103: its rebuild lane's state is on a branch I cannot see.

**The rule that would stop the open count rising** (section 7.1): one in, one out. A new row lands only if one of
three things holds:
- it is on the G9 path;
- an open row is closed, retired or carried in the same commit;
- the owner accepts the net rise.

Panel debt goes into its row as an owed item. Owner-started research gets a priority slot or a deferral date when it
opens. A row with no closing condition (T051c-wave) gets one.

**Grouped for the owner, none recommended for deletion:**
- close as carried or an owner act: T120, T076, T064, T065, T078, T093;
- US4: T072-T075 and T083;
- the N and A workstreams;
- instruments and sweeps;
- the research rows;
- T087, last.

**Owner question 1 could double the path.** Read literally, "No G row bypasses T073/T074, Phase 4's exit or
FR-025/OD-2" would put T072-T074 on it. I read it as binding autonomous promotion only (PLAN-FINAL:1147 says so for A
and N), but that is the owner's call.

**What I could not see:** `claude/int3` beyond main, the owner's chat, the board, the G rows' `claude/*` branches, and
the owner host's campaign times.

## Task 2: AppendBudget's stale seed

**The incumbent, read first.**
- `substrate/lake_engine.py::AppendBudget`, built in `Brain` at `engine/brain.py:1408` and closed at `:3928`.
- The seed PRE-FILLS a tag's 32-wide window, and by design it "stays bounded by the SEED for its first three" writing
  ticks, then migrates to the tag's own rate (the docstring's L4b fix of the August stuck alarm).
- So three floods on a fresh brain is the designed cost of a seed that is stale. The fix is to make the seed match
  today, not to change the mechanism (P2).
- No tool regenerates `tools/lake_append_baseline.json`. Its `method` field records the procedure as prose:
  `make_brain(seed=0)` x120 ticks on a scratch home, BinLog.append+extend wrapped, p95 over nonzero per-tick byte sums.

**What was stale, measured at `ee224e50` (stdlib).** A named-state probe on seed 7 found that the floods are not only
0x07:

| ticks | tag | written | seed (bar) |
|---|---|---|---|
| 1-3 | 0x04 | 982-1,063 B | 113 B (904 B) |
| 6, 31, 56 (the first three night flushes) | 0x07 | 16.1 MB / 3.28 MB / 7.77 MB | 12,381 B |

- **The 0x07 bytes are the nightly skills' npz:** `skills/world-kb.npz` 5.9 MB, `language` 3.7 MB, `physics` 3.3 MB,
  `math` 3.2 MB, written as `_T_STATE` named states.
- They entered the lake with **`bebdd0e2`** (2026-10-24, pa-p4bd-1/2: "language organ and nightly skills persist
  learned npz into the lake").
- So the 0x07 seed went stale because of a deliberate P4b change, two months after the August recording. It is not a
  write regression the alarm was right to flag.

**RED, `ad8936b3`.** The new test is
`tests/test_append_budget_wire.py::test_a_fresh_brains_first_night_flushes_do_not_flood_and_a_burst_still_does`.
- It builds a fresh brain: seed 7 (held out, because the recorded method measures seed 0), scratch home,
  persist=True.
- It counts night flushes by wrapping that brain instance's `_flush_prototypes` in the test only. Production code
  gets no new seam.
- Through three night flushes it asserts `state()["append_floods"] == 0`.
- **The positive control, in the same brain:** a burst of 4 MiB named states sized above the live
  `bar(_T_STATE)`, then one tick, must flood 0x07.
- **On `ee224e50`:** `AssertionError: 6 flood(s) on ordinary writes in a fresh brain's first 56 ticks (3 night
  flushes); last: FLOOD: tag 0x07 ... 7769017 B ...`.

**The fix, `27e5b670`. It is data only: re-record two seeds by the file's own method.**

The regeneration, run from `~` with the primary checkout detached at `ee224e50` (the `_native` build lives there),
then returned to its branch:

```bash
PYTHONUTF8=1 PYTHONPATH=<checkout>/src python <probe> 0 120
```

The probe:

```python
import json, os, sys, tempfile
seed, ticks = int(sys.argv[1]), int(sys.argv[2])
os.environ["NORD_HOME"] = tempfile.mkdtemp(prefix="t2_rec_")
import torch; torch.set_num_threads(1)
from snn_data_lake.substrate import binlog
from snn_data_lake.substrate.lake_engine import AppendBudget
from snn_data_lake.engine.brain import make_brain
cur = {}
ra, re_ = binlog.BinLog.append, binlog.BinLog.extend
def _count(p):
    p = bytes(p)
    if p: cur[p[0]] = cur.get(p[0], 0) + len(p)
def app(self, payload, *a, **k):
    out = ra(self, payload, *a, **k); _count(payload); return out
def ext(self, payloads, *a, **k):
    payloads = list(payloads); out = re_(self, payloads, *a, **k)
    for p in payloads: _count(p)
    return out
binlog.BinLog.append, binlog.BinLog.extend = app, ext
b = make_brain(seed=seed, persist=True); cur.clear(); per = {}
for t in range(ticks):
    b.tick()
    for tag, n in cur.items(): per.setdefault(tag, []).append(n)
    cur.clear()
print(json.dumps({str(t): {"nonzero_ticks": len(v), "p95": AppendBudget._p95_of(v), "max": max(v),
                           "mean": round(sum(v) / len(v), 1)} for t, v in sorted(per.items())}, indent=1))
```

**Its measured output** (11.5 s; stdlib; seed 0; 120 ticks):

| tag | nonzero ticks | p95 B | max B | mean B | old seed | disposition |
|---|---|---|---|---|---|---|
| 0x04 | 120 | 1,095 | 1,099 | 813.2 | 113 | **re-recorded** |
| 0x07 | 13 | 7,421,514 | 16,108,060 | 2,874,963.3 | 12,381 | **re-recorded** |
| 0x08 | 3 | 4,738 | 4,738 | 2,690.0 | 2,902 (bar 23,216) | left: inside its bar |
| 0x0A | 61 | 436 | 437 | 410.5 | 434 | left |
| 0x03 | 0 | -- | -- | -- | 4,205 | carried (legacy, closed) |
| 0x0B | -- | -- | -- | -- | 15,242 | carried (a fixture figure, not a brain measurement) |

The `method` field gains a dated note with these numbers. It says they were recorded on the cloud host, NOT the
pinned host, and asks for a re-record there.

**GREEN on `27e5b670`.**
- The new test passes: 0 floods through 3 flushes, and the planted burst trips 0x07.
- The held-out seeds 11 and 23 also read 0 floods through 3 flushes (53 and 65 ticks).
- `tests/test_append_budget_wire.py`: 11 passed in 19 s. That includes the incumbent overload red, which sizes its
  burst from the recorded p95 and so now writes about 59 MB to the scratch lake.
- Self-checks: `lake_engine` OK, `immortal_retention` OK.
- `tests/test_growth_loop.py -k budget`: passed.

**What the new seed costs.**
- The 0x07 day-one bar is now about 59 MB (8 x 7.42 MB) for a fresh brain's first three 0x07 writes.
- That is the same order as the steady state the window already migrated to: 26-48 MB in my REPLY-3 measurement.
- So day-one sensitivity on 0x07 now equals steady-state sensitivity. It was never finer than that after three
  writes.
- **Disclosed, unchanged by this fix:** a flood smaller than about 8x a night flush, arriving on an ordinary 0x07
  tick, is invisible once the window holds night flushes. 0x07 is bimodal (about 1-50 KB on ordinary ticks, 3-16 MB
  on flush ticks), and one p95 per tag cannot bound both. Splitting it would be a mechanism change; I did not make it.

**Gates on `27e5b670`.**
- `doc_budget_gate --assert`, `doc_command_guard --assert`, `citation_guard`, `null_scale_gate --assert`: OK.
- `ruff check .`: all checks passed (one pre-existing `# noqa` warning in `reasoning/arc_bench.py`).
- `ruff format --check src tests tools`: 660 files formatted.
- **`policy_manifest --assert`: rc 0, "assert OK"**, with no freshness red, because the branch sits on the certified
  `ee224e50`.

**Found and not caused by this change.** `tests/test_brain_hermeticity.py::test_every_allowlisted_asset_is_still_read`
fails on this host, with the same failure on the untouched base: `['models/nmnist_pack.npz', 'models/shd_pack.npz']`
stale. This cloud host lacks the `models/` packs, the same gap that keeps it on the stdlib tier.

**Left for the integrator, not changed:**
- `engine/autonomous_loop.py:748-749` says the 0x07 budget "trips near ~7,000 cells". That was derived from the
  August seed; under the new seed it is about 4 million cells (at about 14 B per cell). It is a docstring, and it was
  already loose once the window migrated.
- A `--record` mode for the baseline would stop this recurring. The file already records one earlier figure set that
  "had no recorded command and [did] not reproduce". It is a code change, so it is not made here.

**Not verified here.**
1. The production tier (eprop-snn) on the pinned host. Its night-flush 0x07 volume may differ, so a stdlib-recorded
   seed could still be low or high there. The method note asks for a re-record there.
2. Linux only.
