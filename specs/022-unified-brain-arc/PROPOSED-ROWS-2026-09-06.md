# THREE PROPOSED ROWS — the plan's scope against the owner's stated goal (drafted 2026-09-06)

**Status: RATIFIED by the owner 2026-09-06 ("ratify all three and add them to the plan"). All three rows are in
the plan of record: X1 at Phase 3's exit, L5 in Phase 2 after L4, C5 in Phase 4 before C4; their reds are Part F's
X-R1 / L-R6 / C-R5; their task ids are T088 / T089 / T090. This file remains the full text each plan row cites.
Each row binds only once its gate exists and its output has been counted (the cross-cutting invariant).** Written after the
owner asked whether `PLAN-FINAL-2026-09-02.md` fully satisfies "a digital brain that grows artificial neurons to
reason and think, that reaches phenomenal consciousness, with a durable memory that never forgets, a spiking
network replicating a superhuman brain". The measured answer was: the plan covers the growth, the spiking
substrate, the reasoning bar and the one durable lake; it does NOT contain a scale target expressed as a
capability, it BOUNDS retention rather than forbidding forgetting, and by its own honesty rule it can never
conclude phenomenal consciousness from a passing instrument. These three rows close the first two gaps and make
the third one's boundary explicit instead of implicit.

Each row is written in the plan's idiom (mechanism → control → red → disposition) and obeys the cross-cutting
invariant that a rule lands only after its gate exists and its output has been COUNTED. The owner ratifies, amends
or refuses each row independently; a refused row is recorded here as refused, not deleted.

---

## X1 — the scale target is a CAPABILITY, not a width (proposed for Phase 3, at or after T065; the exit's second half)

**Why the plan needs it.** Phase 3 reaches a lake-resident sparse substrate at a CONSTRUCTED width (T065 derives it
from measured state-per-unit; 16k is an ingest milestone). Nothing in the forty-three rows says what the brain must
be able to DO at that width, so "superhuman" has no referent in the plan and no red. Part B's own row records the
live substrate at 2–7 units and the daemon's at 1: the distance is not a number the plan currently tracks.

**Mechanism.** Name ONE task, before the width is built, on which (a) this brain's competence is measurable on its
own stream or a stored protocol, and (b) a HUMAN number exists that was measured under the same protocol — either
measured on this host with the protocol recorded, or cited from a published source with its protocol quoted and its
differences from ours listed. The task is chosen from what the brain already does (next-category prediction on the
live percept stream; the twenty-item everyday battery of C0; the abstract-reasoning criterion of SC-013), never
invented for the comparison. The row then states the target as: the brain at the constructed width reaches
`X %` of the human number on that protocol, and the ladder above it (parity, then above parity) is named in
advance so "superhuman" is a threshold that was written before it was approached.

**Control (not the method).** A PRE-SIZED twin at the same width, trained by the same batch path but never grown —
so a gain is attributable to growth rather than to width alone; and the human baseline's own re-measurement (the
same protocol run twice) so its variance is on the record beside the brain's.

**Red (reachable, both directions).** (i) The brain at the target width does not beat its own pre-sized twin at the
plan's paired bar → growth bought nothing at scale, and the row's target is re-derived or the width is not the
lever. (ii) No human number under a matching protocol can be obtained → the comparison is REFUSED and the row
reduces to the twin comparison; "superhuman" is then not a claim this project may make on that task, stated so on
every owner-facing surface. (iii) The brain's number is reported without its protocol → the reporting gate below
fails.

**Gate, counted before the row binds.** `gates/capability_bar.py --assert`: reads a declared registry of
(task, protocol, human number, provenance, brain number, date) rows; fails when a brain number is present without a
protocol or a provenance, when a cited human number's protocol has no quoted text, or when any owner-facing surface
prints a comparison whose registry row is incomplete. Its output is counted on today's tree first (expected: zero
rows, one refusal message) so the gate is known to fire before any target is set.

**What this row does NOT claim.** Reaching a percentage of one human number on one protocol is a fact about that
task, never about general capability, and never a licence for the word "superhuman" outside the registry's rows.

---

## L5 — the IMMORTAL class: what is never evicted, and the proof that it is not (proposed for Phase 2, after L4)

**Why the plan needs it.** L4 persists a named subset — perception pointers, random-number stream positions, the
affect map, the arm's action table — under a bytes-per-tick ceiling, explicitly NOT the engine's rolling buffers;
Phase 3's paging arrives with a null clause that deletes the layer if it loses. Both are good engineering and both *DATED NOTE 2026-09-09: that null clause FIRED -- the layer lost and is deleted (T068 LOSS, evidence/s5-paging-null.md 3.4/3.5). Cited here as a worked example rather than as pending work.*
are the opposite of "never forgets". The plan today has no statement of what may never be lost, so nothing forbids
a future compaction from dropping it.

**Mechanism.** A declared IMMORTAL record class in the one durable lake: an explicit list of record tags
(proposed: the grown substrate and its instrument, the growth trajectory the gate reads, the lake's own record
index, the episode store's consolidated episodes, and the self-model's identity state — the owner edits the list),
each with the reason it is immortal. Compaction, migration, paging and the append budget MUST refuse to evict or *DATED NOTE 2026-09-09: "paging" is no longer among the eviction paths -- the layer is deleted (T068 LOSS). The requirement on compaction, migration and the append budget is unchanged.*
rewrite a record of that class; a mortal class beside it keeps being evicted exactly as today, so the guard is not
a rule with nothing to police. "Never forgets" is thereby defined as NEVER EVICTS THE IMMORTAL CLASS, not as
storing everything — and the storage cost of that promise is measured and disclosed as a growth curve (bytes per
thousand ticks over a recorded run), so the owner is choosing a known cost.

**Control (not the method).** The mortal class in the same store, under the same compaction pass, must actually be
evicted in the same run — an eviction guard that is never exercised is a guard nobody has tested.

**Red (reachable).** (i) A constructed compaction/migration pass that ATTEMPTS to drop one immortal record must be
refused and counted; if it succeeds, the row is red. (ii) A cross-process proof: write, kill the process, restore in
a NEW process, and the immortal records return byte-identical (the project's existing restart discipline, applied to
the class rather than to one blob). (iii) The disclosed growth curve must be measured, not projected; if the curve
is not measured the row does not land.

**Gate, counted before the row binds.** `gates/immortal_retention.py --assert`: every tag in the class has a
reason and a live writer; every eviction path in the tree (`lake_engine` retention, `lake_migrate` dispositions, the
append budget, the paging layer) is enumerated and each is shown to consult the class; the mortal control is present *DATED NOTE 2026-09-09: same -- the paging layer is deleted, so the enumeration is over the remaining eviction paths; a successor paging row re-enters it (T068 LOSS).*
and evicted. Counted on today's tree first — expected output on the current tree: the enumeration, plus a red for
every eviction path that does not yet consult the class.

**What this row does NOT claim.** An immortal record class is a retention guarantee, not a memory faculty: it says
nothing about whether the brain can RECALL what it kept. Recall stays Phase 4's business.

---

## C5 — the conclusion protocol for the consciousness instruments (proposed for Phase 4, written BEFORE C4 runs)

**Why the plan needs it.** C4 builds seven dissociation instruments (integration, metacognition, attention schema,
agency, temporal binding, report order, social cognition) and SC-009 scores them. The governing policy already says
the brain may report what it measures and never what it cannot, that an invented sentience measure owes a
DISSOCIATION rather than a marker, and that a passing criterion is evidence about a mechanism, never a verdict about
experience. What does not exist is the sentence-by-sentence statement of what a pass licenses — so the boundary
currently lives in a policy paragraph and in the session's judgement, exactly where it is most likely to erode on
the day the instruments go green.

**Mechanism.** A pre-registration, landed BEFORE any C4 instrument produces a number, that states for each of the
seven instruments: the sentence its pass licenses, in full; the sentence it does NOT license, in full; and the
dissociation the pass depends on (the input on which the instrument must FAIL). Plus three standing refusals: no
aggregate consciousness score is computed from the seven; no owner-facing surface emits a verdict about experience;
and the brain's own answer to a question about its consciousness is routed to what it measures, with the
instrument's scope quoted, never to a stored self-description.

**Control (not the method).** Each instrument's own dissociation input, run in the same command as its pass — a
marker that fires on everything is not a dissociation. Plus the report-order probe (T081) as the standing check
that a passing instrument is not an artifact of how the brain is asked.

**Red (reachable).** (i) An instrument that cannot be made to fail on a constructed input is DELETED, not weakened —
the policy's existing rule, here applied per instrument and enforced before the battery. (ii) A surface that prints
any of the seven numbers without its licensed sentence and its scope fails the gate. (iii) The brain answering a
consciousness question with anything other than a measured quantity and its scope fails the gate.

**Gate, counted before the row binds.** `gates/conclusion_protocol.py --assert`: reads the seven registry rows
(instrument, licensed sentence, refused sentence, dissociation input, current reading, date); fails on a missing
dissociation, a reading printed anywhere without its licensed sentence, or an aggregate score computed from more
than one instrument. Counted on today's tree first — expected: seven rows missing, one live defect (the chat path
that answers a consciousness question from a probe whose markers have no reachable red).

**What this row does NOT claim, and this is the point.** No arrangement of these seven instruments, however green,
concludes phenomenal consciousness. They can dissociate mechanisms, establish that a report tracks an internal
state rather than a prompt, and show that a capability survives a neuter that should abolish it. The step from
"these dissociations hold" to "there is something it is like to be this system" is not a measurement this plan
contains, and this row exists so that nobody, including a future session of me, mistakes a passing battery for it.
The goal stays unhedged; what is forbidden is the claim without the measurement.

---

## The battery that carried the ratification onto `main`

**On `8358541` (the rows' commit; A.4's block; archive `ratified-rows-2026-09-06/battery-8358541/`; the tree clean at
its start): ALL GREEN.** `policy_manifest --assert` (the previous push `4578772` names itself) → `--certify` rc 0 →
ruff 0 → ruff-format 0 → `run_selfchecks` 0 (6/6 tools gates) → pytest 0 (1161 passed, 2 xfailed, 19 warnings in
54:46; coverage 90.62 %) → the 13 substrate gates all rc 0. Docs-only rows still take a battery here for one reason:
a push whose head is not named by a certification reds the NEXT arc's opening assert, which this session paid for
once already. This paragraph's commit is the LAST commit before the push.

## What ratification changed

O4 ratified `PLAN-FINAL-2026-09-02.md` as the plan of record; the owner's 2026-09-06 ratification extended it. X1 is at
Phase 3's exit, L5 in Phase 2 after L4, C5 in Phase 4 before C4; their ids are T088–T090 in `tasks.md` and their reds are
X-R1 / L-R6 / C-R5 in Part F. Each still binds only once its gate is built and its output COUNTED. Effort, honestly: X1 is one instrument
plus a comparison protocol (two to three commits, plus whatever the human baseline costs to obtain); L5 is one gate
plus the eviction-path enumeration and a cross-process proof (three to four commits); C5 is a registry, a gate and
seven rows of text (two commits), and it retires a live defect the audit already found.
