# US1 policy audit — specs/022, owner directive step 1

Audited on `main` at `17dec3ec` (T039l), tree clean, 0 ahead / 0 behind, 2026-08-25. The owner
directive (2026-08-22, recorded at the top of the session-start memory): *audit US1 against
`docs/POLICIES.md` P-by-P for technical debt and defects — a written verdict MET / DEBT / DEFECT
with a measurement behind each; US1 being 5-of-5 with 0 unchecked tasks is NOT the same as clean.*

**Method.** Eight policy lenses ran as independent agents over the US1 surface (T025–T039l), each
required to reproduce a command behind every BLOCKER/MAJOR claim; findings were deduplicated and
then adversarially refuted — every BLOCKER by two independent refuters at full effort, every MAJOR
by one; 57 agents total. Raw yield: 48 findings → 11 BLOCKER / 27 MAJOR / 10 MINOR. After
refutation: **3 BLOCKERs CONFIRMED, 1 CONTESTED, 7 REFUTED; 7 MAJORs CONFIRMED, 20 REFUTED-ONCE.**
Every confirmed finding was then re-verified on this machine before any fix landed. Refuted
findings ship in §5–§6 with the refuter's reasoning — the free eyeball pass the review template
requires, and the rollback trigger if one later proves real.

## 1. Verdict table

Every gate row was re-run on this tree this session (not quoted); the fix column names the commit
that discharged the finding.

| policy | verdict | measurement / command behind it |
|---|---|---|
| P0 autonomous self-improvement | MET (as audited) | No US1 claim credits an unmeasured capability; audited through its guards (P2/P3/verify-first lenses — the enforcement table's own row), not a dedicated lens. Avenue table untouched by US1. |
| P1 no-hardcode | DEBT → discharged | One confirmed finding (M0, the insula `_var` seed at 1.0, 4× the declared unit's maximum) — FIXED (commit 7d608a46). `_G_INH` (B0) refuted 2/2: a total budget consumed differently, not the retired round-trip. |
| P2 adopt-or-merge | MET | Both P2 BLOCKERs refuted 2/2 on reproduction (B1: two DIFFERENT quantities, not two reporters; B2: the production rule HAS a CI red — rc=1 on a nothing-charges mutation). The workspace-duo debt predates US1 and its benchmark condition is unchanged. Relay-vs-router adjudication re-verified: the pinned test exists and fires. |
| P3 no dormancy | DEFECT → fixed | B4 CONFIRMED: `brain_telemetry.snapshot()` called `insula._top_down()`, whose post-condition writes the bus's listener-use term — the cockpit poll and the daemon's own Recorder charged the consumption record US1 certified, and reported "use" on sleep ticks the faculty never observed. FIXED (commit 142c076b) with a polled-vs-unpolled byte-equality control whose red fired on the pre-fix tree. B3 (expired exemption bound) adjudicated as DEBT → discharged (commit 067d9d6b). |
| P4 one brain | MET | `replication_guard` OK (each stack class defined once); `rev-list origin/main..HEAD` = 0. |
| P4b durable lake | MET | Coverage restart red passes (`tests/test_coverage_persistence.py`); ledger writes `brain_lake.bin` record tags, not a sidecar; the ESTOP_LATCH sidecar is inside the owner-approved envelope change (M16 refuted: P4b's clause governs what the BRAIN learns). |
| P4c every build is the brain | DEBT → discharged | All US1 modules registered (REGIONS + run_selfchecks, verified per module); `p4c_ratchet` OK (202, no zone above baseline). But T037a blew SECURITY.md's word ceiling 63.9% and T039f grew an already-over POLICIES.md — no gate said so because `doc_budget_gate`'s battery form exits 0 (B7). Ceilings discharged (commit 518b94c0), gate made able to fail (commit 30fe7182). |
| P5 / P5a GPU | MET (n/a to US1) | US1 shipped no GPU-regime work; wake-tick perf ratchet GREEN alone on a quiet machine: p50 105.843 vs 183.517, p95 221.668 vs 225.768, duty 0.508/0.508. |
| P6 mutual reach + use | DEFECT → fixed | B5 CONFIRMED: four live present-tense sentences (POLICIES.md enforcement row + P6 block; p6_gate BASELINE header + `report_criteria` output) assert faculty-to-faculty = 0 while the gate measures 1 — the gate contradicted itself in its own CI output, and the promised floor conversion never landed (it lives in `tests/test_interoceptive_specialist.py`, which pins count AND identity; the prose now says so). M11/M19 CONFIRMED: the drain census resolved only string literals, printing NEVER DRAINED about the exact subscriber T039h wired. M18 CONFIRMED: a behaviour-identical `import … as` rename flipped two ratchets red. All fixed (commits 518b94c0, 36b89df6, 36b89df6). All five ratchets hold at exactly baseline (zero slack — observed, recorded). |
| P7 any dependency | MET | `dependency_ledger --assert`: UNDECLARED 0, UNUSED 0, MISDESCRIBED 0, EXEMPT 1, ALL_EXCLUDES 8 — no new dependency entered with US1. |
| P8 invent the method | DEFECT → fixed | B7 CONFIRMED (the gate that cannot fail — see P4c row; P8's own line: "a gate that has only ever been seen green is indistinguishable from one that cannot see"). The six US1 inventions each re-verified: reachable reds fire from their own self-checks; B8/B9/B10 refuted on the policy's actual text. |
| e-prop invariant | MET | `eprop_transport_gate` self-check OK (0 governed findings, both constructed reds fire); `gradient_claim_gate --assert` OK (MISSING_DECLARATION 0, UNDETERMINABLE 0). B8 (report_use as undeclared learning rule) refuted: a bounded use-trace outside the census's own scope definition. |
| cross-cutting invariants | DEBT → discharged | Determinism, G5 replay, lake rebuild, envelope hash: all green with their controls this session. But POLICIES.md carried two falsified mechanization counts (M7) and a present-tense P8 sentience claim falsified at HEAD (M26) — corrected (commit 518b94c0). |
| US1 dated hand-offs | MET (recorded) | (1) T028→T046a carries the obligation; T066/T076 orphan-claims refuted (P8 imposes no receiving-row obligation). (2) T034→T081 still right: T081 carries both replacement halves. (3) T037→T069/T079 recorded in-module. (4) esp32 SC-008 milestone: M24 refuted — the proposed re-anchor would have created a stale-exempt red; the prose record stands. |

## 2. Confirmed findings and their fixes

| finding | severity | fix |
|---|---|---|
| B4 — the instrument charges the listener term it reports (`snapshot()` → `_top_down()` → `report_use`; dominant escape = sleep ticks; second site: the T039i chat reply; third: the certification harness itself) | BLOCKER | 142c076b: side-effect-free `last_top_down()` (the `last_salience()` shape); all three reporters converted; PROVENANCE row re-worded; polled-vs-unpolled byte-equality test, red proven on pre-fix tree |
| B5 — falsified faculty-to-faculty prose + undischarged floor promise | BLOCKER | 518b94c0 (POLICIES.md re-derived from gate output) + 36b89df6 (p6_gate `report_criteria` branches on the live value; BASELINE header corrected; the floor's home — the pytest pin — named) |
| B7 — doc_budget_gate red on main, battery form exits 0; SECURITY.md +63.9% (T037a), POLICIES.md +2.4% (T039f), INVENTION_REGISTER +1.8% | BLOCKER | 518b94c0/518b94c0 (ceilings discharged) + 30fe7182 (bare form asserts, like every sibling; the enforcement row's "bare = assert" becomes true) |
| B3 — seven `wire` dormancy exemptions bound to "S0 exit (T039), not movable again"; T039 landed, all seven still unreachable, module gate silent | CONTESTED → DEBT | 067d9d6b: standing-TODO report arm ported to the module gate (the method gate's shape; counted at landing), rows re-adjudicated at the passed bound without a new "not movable" promise — the consumers are ordered US2/US3 tasks (T057/T064/T066) that could not have landed by T039 |
| M0 — insula `_var` seeded at 1.0, 4× the `level01` maximum; still suppressing salience hundreds of ticks after boot | MAJOR | 7d608a46: seed derived from the declared unit's own bound (0.25); settled-variance self-check criterion with the 1.0 seed as its red. (The zero seed was measured WORSE by the refuter — near-maximal ignition subsidy at tick 3 — and refused.) |
| M7 — POLICIES.md P3 carries two falsified mechanization counts (56/16/5 vs measured 66/20/9+3 TODOs; 15/17 vs measured 7/9) | MAJOR | 518b94c0: counts replaced by the re-derive instruction the same file already uses for the region-charge census |
| M10 — Coalition channels (`arousal`, `caution`) computed live, journalled IMMORTAL, read by no faculty, and `submit_coalition`'s docstring claims consumers that do not exist | MAJOR | 6d841483: the false consumer claim corrected; dated CONSUMER DISPOSITION added in the T037/efferent_boundary form |
| M11/M19 — p6_gate prints "NEVER DRAINED 'interoception'" about the one wired drain (census resolves only string literals; the same output counts that drain as the ONE faculty-to-faculty read) | MAJOR | 36b89df6: `_names` resolves module-scope `NAME = "literal"` constants; AND the census was wrong the other direction too (found while fixing, exactly as the refuter predicted): BUS-owned self-check reads counted as drains, so `language` — the deliberate control — read as drained for the life of the census; BUS reads now excluded, census moves 3-of-6 (wrong both ways) → 2-of-6 honest. Self-check gains both reds (interoception-not-undrained, language-still-undrained); s0-exit §8b's misquote corrected |
| M18 — an `import … as` rename flips `cross_module_reads` and `unread_writer_modules` red (the T034 bare-name branch keys on the spelled name) | MAJOR | 36b89df6: ImportFrom-asname resolution in the walk + a pure-rename control arm in `liveness()`; the miscounting "chat adapter" comment corrected |
| M26 — POLICIES.md P8 asserts in the present tense a chat-path defect closed 2026-08-04/05 | MAJOR | 518b94c0: bullet rewritten past-tense with its date and the open T081 pointer |

## 3. What was re-run green this session (the MET evidence)

All commands from the repo root in `.venv-lake`, rc captured:

```
region_charge_gate --assert      8 of 16 CHARGED; floor holds 8 >= 8
p6_gate --assert                 all 5 ratchets hold (each at exactly baseline)
manifest_gate --assert           27 ports, 14 edges, 0 REF; findings=0; warnings=0
p4c_ratchet --assert             202 ungoverned, no zone above baseline
dependency_ledger --assert       0/0/0, EXEMPT 1, ALL_EXCLUDES 8
red_reachability --assert        NO_RED 0, TAUTOLOGICAL 0, READS_SYSTEM 13
envelope_hash_gate --assert      all 5 envelope files match approved hashes
gradient_claim_gate --assert     MISSING_DECLARATION 0, UNDETERMINABLE 0
citation_guard                   STRICT 0 unexempted, 1 exemption
replay_gate                      byte-identical; omit-positions refused; re-seeded control diverged
lake_rebuild_gate                rebuild equality; planted red fired; tracking control moved
determinism_gate --assert        trains BYTE-EQUAL T=1..16 at P=8; f32 control diverged, int64 agreed
eprop_transport_gate             6 constructed cases; 408 modules; 0 governed
dormancy_gate                    266 modules, 213 reachable, 53 exempt, 0 dormant
method_dormancy_gate             66 rules, 0 dormant, 3 standing TODOs (pre-existing)
import_manifest                  7 cross-tier targets import
replication_guard                one canonical Brain/LiveBrain/AutonomousArm
safety_envelope                  SECURITY.md enumerates exactly the five
perf_baseline --assert (ALONE)   p50 105.843 vs 183.517 OK; p95 221.668 vs 225.768; duty 0.508
```

## 4. Observations recorded, not fixed (dispositioned)

- **All five P6 ratchets sit at exactly their baselines with zero slack.** Not a defect — a
  ratchet at its floor is a ratchet doing its job — but the next P6 regression reds immediately;
  budget for it in US2 work that touches workspace wiring.
- **The seven lake/kernel exemptions' consumers are US2/US3 tasks** (T057/T064/T066 et al.); the
  standing-TODO arm now prints them on every gate run so the next expiry cannot be silent.
- **`language` remains subscribed-and-never-drained on purpose** (T039c's negative control) — two
  audit lenses independently re-flagged it; it is deliberate, and the drain census fix keeps
  printing it.

## 5. Refuted BLOCKERs (kept per the review template — the rollback trigger if one proves real)

| id | claim | why it fell |
|---|---|---|
| B0 | `_G_INH = 2.4` is the retired `_W_SELF * capacity` round-trip surviving in reentrant_workspace | The retirement's own tell ("the value only ever appeared divided") measurably does not exist there — it is a total budget consumed differently; predates US1 |
| B1 | Two workspace reporters on one payload disagree 27/30 | The disagreement is the signature of two DIFFERENT quantities (access-hold content vs bus snapshot), not two reporters of one |
| B2 | region_charge_gate's self-check proves a copy, so production has no red | Production HAS a reachable CI red: a nothing-charges mutation returns rc=1 |
| B6 | ignition_instrument's private workspaces evade the ratchet | Harness instances inside an instrument are the exemption the gate's own docs draw; attributable spelling reddening is the disclosed strict-attribution design |
| B8 | `report_use` is an undeclared learning rule invisible to gradient_claim_gate | Outside the census's own scope definition (a bounded use-trace, no error signal, no weight); the "blind spot" is the gate's printed, disclosed limitation |
| B9 | INVENTION_REGISTER misses three US1 laws | The register's own preamble scopes membership; the exit certification adjudicated the cross-link set in writing and register membership is not in it |
| B10 | `_consciousness`'s documented iff cannot be true; the branch is dead | The headline mis-reads the docstring's iff; the reachable-red obligations attach to the probe, whose deletion record already discharged them |

## 6. Refuted-once MAJORs (one refuter, medium effort — eyeballed, kept for the record)

M1 relay bound conditional on horizon (P1 reaches no constant there — running statistics);
M2 IGNITION_GAIN never re-derived for the second input (the input predates T039f; calibration
record covers it); M3 the 200 time base + phantom `horizon` parameter (minor prose defect
survives, folded into 7d608a46's file anyway where cheap); M4 `_last_salience` second store
(divergence does not reproduce on the claimed path); M5 OUT-arm two severities (one
implementation; the pin and the warning are different jobs); M6 report_channel census names a
missing cockpit row (the named consumer exists under another name; fallback lens inverted);
M8 relay burst/tonic unreported (a diagnostic label, not the capability); M9 `tap_errors` zero
readers (planted-failure probe shows the disclosure path fires); M12 "global workspace" docstring
(names the concept, not the class); M13 silent coverage-write failure (disclosed, deliberate);
M14 restart-amnesia on the bare launch path (does not reproduce — restore is not env-gated on
that path); M15 append-budget bound absent in production (the tree's own written, gated
disclosure); M16 ESTOP_LATCH adjudicated only in a lane artifact (P4b's clause governs learning,
not the envelope's latch); M17 cockpit vs chat read different salience attributes (same object;
the raise path already answers honestly); M20 private-method consumption contradicts T039g
(P6 has no encapsulation clause); M21 `_top_down` swallows exceptions into its own null (the red
exists at the live layer); M22 tonic/burst unsayable (the distinguishing capability is the relay
output, which is consumed); M23 orphaned hand-off targets (P8 imposes no receiving-row
obligation); M24 esp32 anchor retirement (the proposed fix reds stale_exempt by the gate's own
semantics); M25 declare-ahead doctrine contradiction (port vs edge elision).

## 7. Minors (listed unverified, per the template)

m0 `_int_tau` shared across seven baselines; m1 EpisodeDriver landed with zero clients (its
EXEMPT row says so); m2 port_charge_probe docstring carries a discharged obligation as pending;
m3 global_workspace port declaration reads as awaiting landed consumers (folded into 6d841483);
m4 two registry rows still call region_charge_gate report-only; m5 `_T_STATE` IMMORTAL vs
physical prune in `_prune_state_history`; m6 P6 blind-spot figures stale both directions;
m7 the declared-anatomy layer lives in region-less `substrate/`; m8 `_adjust`'s "only form"
claim vs `max()`; m9 SYNTHESIS §6 "permanent" vs the same section's retirement record.

## 9. The fix series (each behind a full green battery, pushed, perf ratchet green on every push)

| commit | finding | battery |
|---|---|---|
| `142c076b` | B4 — telemetry poll stops charging the listener term | #69 |
| `7d608a46` | M0 (+M3) — insula variance prior derived from the port | #70 |
| `518b94c0` | B5/M7/M26 prose + B7 ceiling discharge (SECURITY 488, POLICIES under 28083 via byte-exact relocation, REGISTER 950) | #71 |
| `30fe7182` | B7 — doc_budget_gate bare = assert | #72 |
| `36b89df6` | B5/M11/M18/M19 — p6_gate stops contradicting the tree | #73 |
| `067d9d6b` | B3 — standing-TODO arm + seven expired bounds re-adjudicated (+m4) | #74 |
| `6d841483` | M10 — coalition channels stop claiming absent consumers | #75 |

Batteries #69–#75: 333/333 self-checks, pytest rc=0 at 90.25% coverage, ruff clean ×2, 7 gates
green, every run. Perf ratchet on every push (ambient-ratio p50 §binding): all green; the
standalone quiet-machine run recorded in §3.

## 8. What this audit does and does not claim

It claims: every policy walked against the US1 surface with a command behind each verdict; three
confirmed BLOCKERs and seven confirmed MAJORs found, fixed or dispositioned, each fix landing in
its own commit behind a full green battery; seven BLOCKERs and twenty MAJORs killed by
adversarial refutation and preserved above. It does not claim US1 is defect-free — it claims
that what these eight lenses, two refuters per blocker, and this session's re-runs could reach
is now either green, fixed, or recorded. The refuted findings above are the rollback surface:
any of them later proving real reverts its refutation and reopens this audit's verdict row.
