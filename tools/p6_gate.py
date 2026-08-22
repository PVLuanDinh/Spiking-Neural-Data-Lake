"""p6_gate -- P6's gate: the workspace write:read topology AND the connectome diff, both ASSERTING.

  set PYTHONUTF8=1 & .venv-lake\\Scripts\\python.exe tools/p6_gate.py
                                                     [--assert] [--self-check] [--root PATH]

  --assert      exit 1 if any ratchet regressed, or if the walk stopped seeing the tree. CI runs this.
  --self-check  prove the walk is alive AND construct+run one RED per criterion. CI runs this too.
  --root PATH   measure a different tree -- a `git archive` extract, or a deliberately worsened copy.
                Without it a gate cannot be SHOWN to go red, only argued to.

WHAT IT MEASURES, AND WHY THESE TWO PROXIES. P6 asks whether a faculty changes ANOTHER FACULTY. P3
asks whether a faculty changes BEHAVIOUR -- the arm, a value, a weight, a reply. The two come apart,
and the gap is the whole point: a faculty wired straight into `_route_percept_to_action` charges the
arm decision and talks to nothing, which is P3-clean and P6-dead. P6 names two conditions, and this
tool now has one half per condition:

  (b) "another faculty CONSUMES its output" -> HALF ONE, the workspace write:read topology. The
      global workspace is the one structure in this brain shaped like a mutual-consumption bus (many
      writers; later computation READS it -- the role a transformer's residual stream plays).
  (a) "reachable through a TYPED DECLARED route"  -> HALF TWO, the connectome diff. `connectome.py`
      declares the region->region wiring; the import graph is what the code actually implements.
      Built 2026-08-06; it was the genuinely-unbuilt half until then.

HOW TO READ THE OUTPUT. A rising READ count against a flat WRITE count is what P6 progress looks
like. A rising WRITE count on its own is NOT progress -- it is more shouting into the same sink, and
this gate is deliberately built so that adding a `submit()` moves the number that does not matter.
NONE of the asserting criteria is a raw read or write count, for exactly that reason.

WHAT WOULD FALSIFY A P6 CLAIM MADE WITH THIS TOOL, stated up front because POLICIES.md requires it.
This walk is STATIC. It proves a read EXISTS; it cannot prove the read CHANGES the reader's output.
The failure this repo has already built, measured and reverted -- see the worked example at
`brainstem/autonomous_loop.py:969` -- is a wire that turns a constant telemetry field into a varying
one and changes nothing downstream (`_wm_spikes` 1 distinct -> 22 distinct, selected action changed
0 of 200 ticks). So a green line here is a NECESSARY, not sufficient, condition. The sufficient
evidence is the neuter-A/B `tools/charge_probe.py` performs, aimed one level over: disable the
WRITER faculty and show the READER faculty's own output changes. If it does not, the wire is
reverted and the faculty is exempted with a reason, exactly as under P3.

--------------------------------------------------------------------------------------------------
RE-MEASURED 2026-08-06 at 4fc5fb1c by running this file. THE PREVIOUS BLOCK'S NUMBERS HAD ROTTED FOR
THE THIRD TIME. It said 46 writes / 9 reads / 31 instances / 16 private (dated 2026-08-04). On the
SAME (method, arity) table it was written with, the tree reads 70 / 21 / 53 / 22; on the corrected
table below it reads 70 / 54 / 53 / 22. Three `P6 phase 1-3` commits landed in between and nobody
re-derived. `docs/POLICIES.md:1011` carried the same stale 16. Re-derive by running this file rather
than quoting it -- and when you quote a read count, say which table produced it, because that number
is not a fact about the tree on its own.

THE SHAPE OF THE FINDING HAS NOT MOVED ACROSS SIX COMMITS, AND THAT IS THE MEASUREMENT THAT DECIDES
WHAT THIS GATE MAY ASSERT. Instrument held constant at THIS file (so every column uses the CORRECTED
API table, including the historical ones), six trees extracted with `git archive` and measured with
`--root`. `*` marks the five quantities this gate asserts:

  metric                  6b33a45d  cd619903  3adb68e2  c7e1b816  acd4627a  4fc5fb1c
  writes (raw)                  46        46        46        55        67        70
  reads (raw)                    9         9        11        19        47        54
  PRIVATE instance SITES        16        16        16        18        22        22   <- RISES TWICE
  PRIVATE instance MODULES *     7         7         7         7         7         7   <- FLAT
  cross-module reads       *     1         1         1         1         1         1   <- FLAT
  FACULTY-to-FACULTY             0         0         0         0         0         0   <- FLAT
  writer modules unread    *     7         7         7         7         7         7   <- FLAT
  undeclared import pairs  *    11        11        11        11        11        11   <- FLAT
  unimplemented declared   *    21        21        21        21        21        21   <- FLAT

Pinning each candidate at commit N-1 and asking whether it reddens commit N:

  PRIVATE instance SITES   -> FIRES on c7e1b816 (16->18) and acd4627a (18->22)   REJECTED
  every asterisked metric  -> CLEAN across all five steps                        ADOPTED

A "private instances must not RISE" gate pinned at the last value recorded in POLICIES.md (16) would
have reddened `c7e1b816` and `acd4627a` -- the two commits that constitute this project's actual P6
work, because a bus self-check that builds throwaway instances is not more private buses. A gate that
fires on the fix is worse than no gate, so the SITE count is REPORTED and NEVER ASSERTED; the MODULE
count is the ratchet, because a module is an object identity and a self-check local is not.

The flat rows are the point twice over. They are what makes a ratchet honest -- five commits of real
work moved none of them, so a rise is a signal and not noise -- and they are also the finding: three
commits of P6 work moved the write count by 24 and the read count by 45 and moved the structure by
ZERO. That is the "a rising write count is not progress" verdict, measured over history rather than
asserted.

THREE INSTRUMENT CHANGES LANDED WITH THIS REVISION. They are reported HERE, and under their own
heading in the output, so that none of them can be read as progress in the tree. Each was measured by
running the OLD and NEW instruments over the SAME tree (4fc5fb1c), varying one change at a time:

  metric              OLD   +broadcast_all   +alias only   NEW (all three)
  reads (raw)          21              54            21                54
  self-reads           15              43            15                43     (see change 2)
  unattributable        5              10             5                10
  cross-module          1               1             1                 1
  faculty-to-faculty    0               0             0                 0
  PRIVATE modules       7               7             7                 7
  unread writers        7               7             7                 7

  1. `broadcast_all` was absent from the (method, arity) table. `GlobalWorkspace.broadcast_all()`
     (`thalamus/global_workspace.py:534`, zero-arg) returns the admitted SET and its own docstring
     calls it the only answer to "what else is available", so every consumer of it was invisible.
     Adding it moves READS 21 -> 54 (+157%) and moves NOTHING structural. That insensitivity is the
     single most useful measurement here: the raw read total is almost entirely an artifact of which
     names the table contains, and every STRUCTURAL number is completely insensitive to it -- which
     is why every assertion below ratchets structure and none of them ratchets a count.
  2. An owner this walk cannot attribute was silently filed as a SELF-read. Five reads were being
     counted as modules talking to themselves when the honest answer was "this walk does not know"
     (`cortex/brain.py:4320` and `reasoning/spike_chat_lake.py:1058` among them; both go through a
     local alias whose token `workspace` is bound in two modules, so the owner is genuinely
     ambiguous to a static walk). They are now reported as UNATTRIBUTABLE. Self-reads 20 -> 15.
  3. Local aliases (`ws = self.arm.workspace` ... `ws.broadcast()`) resolve, and an aliased token is
     now a workspace receiver -- it was not, so such a call was not a site at all.

  CHANGE 3 MOVED NO NUMBER AT HEAD AND WAS STILL A PRECONDITION FOR THE `cross-module reads >= 1`
  FLOOR. Stated plainly because a change with a zero delta looks like a change worth skipping. Its
  value is not a count, it is a closed false-red channel, and that was CONSTRUCTED AND RUN: rewriting
  the one cross-module read as `_w = self.arm.workspace; _w.broadcast()` -- a refactor that removes no
  route and changes no behaviour -- took the OLD instrument's census from 1 to 0 and would have
  reddened the build. Under the new instrument the same refactor holds at 1. A floor a no-op refactor
  can redden is a false red waiting to happen, and it was found before the floor was written.

THE FACULTY-TO-FACULTY FILTER WAS ONE-SIDED AND IS NOW BOTH-ENDPOINTS. It checked only the READER's
role. A new faculty reading the arm's workspace exactly the way `cortex/live_brain.py:313` does --
owner role ACTION -- moved the old number 0 -> 1 while both-endpoints stayed 0. So the reading
"faculty-to-faculty > 0" was satisfiable by wiring a faculty into the one action gate, which P6
verbatim calls "the DEFECT P6 names, not a way to satisfy it". Constructed and run before the filter
was changed. The consequence for the gate: `faculty-to-faculty must not FALL` is NOT asserted, and
not because it is red -- because a count cannot fall below zero, so it has no reachable red at all,
and P8 says such a criterion is DELETED, not weakened. It stays a reported number until the day it
is non-zero, at which point it becomes a floor with a real red.

THREE FURTHER INSTRUMENT CHANGES LANDED 2026-08-06 UNDER ADVERSARIAL REVIEW. All three moved ZERO
numbers at HEAD (70 writes / 54 reads / 1 cross / 10 unattributable / 7 private modules / 7 unread /
11 undeclared / 21 unimplemented, before and after). Each was found by CONSTRUCTING the change on a
worsened `git archive` copy and RUNNING the gate over it, never by reading the source:

  4. `unread_writer_modules` was OWNER-centric -- `m not in read_owners`, where read_owners are the
     modules that OWN a cross-read instance. A module writing into a bus it does NOT own could
     therefore never qualify however many faculties read that bus, so a NEW FACULTY SUBMITTING INTO
     THE ONE SHARED WORKSPACE `cortex/live_brain.py:313` READS went 7 -> 8 AND EXITED 1. The gate
     reddened the single change P6 exists to encourage. This project rejected the private-instance
     SITE count for precisely that property; the same disqualification applied here and was missed
     because the criterion was never run against a fix, only against regressions. Writes are now
     attributed to the instance written INTO, exactly as reads already were.
  5. The alias hop resolved only the LOCAL form (`ws = self.arm.workspace`). Hoisting the same read
     into an ATTRIBUTE (`self._ws = self.arm.workspace` ... `self._ws.broadcast()`) -- a refactor
     that removes no route, changes no behaviour and reaches the same object -- took
     `cross_module_reads` 1 -> 0 and `unread_writer_modules` 7 -> 8, reddening TWO ratchets
     including the floor the local-alias fix was built to protect. Zero such sites exist at HEAD.
  6. `region_imports` matched only `parts[0] == "snn_data_lake"`, so RELATIVE imports were invisible
     to half two. The identical amygdala->cerebellum wire fired `undeclared_import_pairs` when
     spelled absolutely and passed clean when spelled relatively: the ceiling was evadable by import
     style alone. There are zero relative imports under src/snn_data_lake today.

REJECTED BY MEASUREMENT, recorded so it is not re-proposed: resolving an ambiguous receiver by
CLASS (`submit` is GlobalWorkspace's, so the token `workspace` narrows to `autonomous_loop` and not
`developmental_loop_brain`). It would take `cross_module_reads` 1 -> 4 and `unread_writer_modules`
7 -> 6, but three of the four "cross" reads it produces are `ac.workspace` / `strong.workspace` on
LOCAL objects in `cortex/access_consciousness.py`'s self-check, attributed to `autonomous_loop` on
the strength of a naming coincidence. That trades an honest UNATTRIBUTABLE for a confident guess and
would rest the floor on three of them. The ambiguity is real; "this walk does not know" is the
correct answer to it.

THE SELF-CHECK NOW CARRIES GREEN CONTROLS AS WELL AS REDS. A criterion that fires on a regression
AND on the fix has no green state, so its red carries no information. Three pieces of correct work
are constructed and required to stay green: the attribute-hoist refactor, the P6 fix itself, and one
more private-bus SITE inside an already-private module. The first two were measured RED before
changes 4 and 5, which is why they are pinned in the tool rather than described in prose.

WHAT THIS GATE STILL DOES NOT DO. It does not prove consumption changes the consumer (see the
falsification paragraph). It resolves two alias hops and one `self.<attr>` hop -- a workspace held
in a list, or passed as a FUNCTION PARAMETER, is still invisible, and that is the residual FALSE RED
of `unread_writer_modules`: a faculty handed its bus by dependency injection (`def __init__(self,
ws)`) writes through a receiver this walk cannot attribute and is counted as unread. Strict was
chosen over lenient because excluding unknowns lets any new writer evade the ceiling by taking its
bus as a parameter; the count of modules in that position is PRINTED under the unread list so it is
never mistaken for a finding about the tree (1 of the 6 today: `reasoning/spike_chat_lake.py`;
re-derived 2026-08-12 at the WorkspaceBroadcast retirement, which took the unread census 7 -> 6).

IT IS A PURE AST WALK, SO UNREACHABLE SYNTAX COUNTS. Measured: a module that is never imported,
containing an `if False:` block whose class is never instantiated, manufactures a cross-module read,
pulls its target out of the unread set, and CANCELS a genuine new unread writer -- 7 + 1 - 1 = 7,
exit 0, where the genuine writer alone exits 1. Nothing here distinguishes live code from dead code
or from a `__main__` self-check, so these numbers can be laundered by someone willing to write
syntax that never runs. Closing it needs reachability, which this walk does not have.

Its scope is `src/snn_data_lake/` only, so `tests/`, `tools/`, `research/`, `training/`,
`integrations/`, `gcp/` and `infra/` are outside the walk; under P4c ("there is no 'outside the
brain'") that is a scope gap, not a design choice, and it is the next thing to close. Half two is
narrower still: `lakehouse/`, `plasticity/`, `reasoning/`, `substrate/` and `training/` -- 241 of
the 400 .py files -- are directories the connectome declares no region for, so a cross-region wire
routed through any of them is invisible to `undeclared_import_pairs`.
--------------------------------------------------------------------------------------------------
"""

import argparse
import ast
import os
import sys
from collections import defaultdict

# --------------------------------------------------------------------------------------------------
# What counts as "the workspace". Two bus classes. (Until 2026-08-12 this set also carried
# WorkspaceBroadcast, the one wrapper that owned a bus instance and re-exported it under a different
# verb -- omitting it is how a prior reading turned three writes into one read. The wrapper was
# RETIRED that day, cortex/competence_executive's retirement record; its 3-arg `broadcast` arity
# entry left the API table with it.)
# --------------------------------------------------------------------------------------------------
WORKSPACE_CLASSES = {"GlobalWorkspace", "ReentrantWorkspace"}

# (method, total argument count) -> kind. Arity is load-bearing, not decoration: `broadcast` with
# zero arguments is GlobalWorkspace's reader; other arities are other classes' methods that merely
# share a name (Claustrum.broadcast takes two; the cockpit's chat queue has a one-argument submit)
# and are correctly not matched.
#
# THIS TABLE IS A CLOSED ALLOWLIST AND THAT IS ITS KNOWN WEAKNESS. A new accessor is invisible until
# it is added here by hand, which is exactly how `broadcast_all` hid 33 reads. When adding a name,
# print the delta under INSTRUMENT CHANGE and do not let it read as progress.
API = {
    ("submit", 3): "WRITE",
    # INSTRUMENT CHANGE 2026-08-21 (T031): perception's write moved doors -- submit ->
    # submit_coalition (typed channels). Same bus, same WRITE semantics, one site moved;
    # without this row the walk lost sight of it and the autonomous_loop liveness floor
    # correctly fired INSTRUMENT DEAD (2 of 3). Not progress -- a rename made visible.
    ("submit_coalition", 3): "WRITE",
    # INSTRUMENT CHANGE 2026-08-21 (T034): report_channel.snapshot(ws) is the declared
    # READ route to the bus -- live_brain.thought and spike_chat_lake._attended now read
    # through it, and without this row the walk reported cross_module_reads 1 -> 0 while
    # the tree's actual consumption GREW. The resolver attributes snapshot's ARGUMENT
    # the same way it attributes an accessor's receiver.
    ("snapshot", 1): "READ",
    ("broadcast", 0): "READ",
    ("broadcast_all", 0): "READ",  # added 2026-08-06; see the docstring's instrument-change note
    ("broadcast_to", 1): "READ",
    ("inbox", 1): "READ",
    ("subscribe", 1): "REGISTER",
}
READ_ATTR = "current"  # a property on both bus classes; an attribute load, not a call

# REGISTER is reported but never folded into the read count. Registering a callback is not
# consumption -- POLICIES.md's honesty guard says so in as many words ("adding a reader that stores a
# value is not consumption"), and a `subscribe()` that nothing later drains is exactly the telemetry
# bus P6 predicts it will attract.

# --------------------------------------------------------------------------------------------------
# Module roles. Declared, not inferred, so the classification is auditable instead of hidden -- and
# every entry carries the reason it claims, printed in the report. `verify_roles()` re-derives each
# claim from the AST so a stale label goes RED rather than quietly mislabelling a module.
# --------------------------------------------------------------------------------------------------
ROLES = {
    "thalamus/global_workspace.py": (
        "BUS",
        "defines GlobalWorkspace; its writes are its own __main__ self-check, not broadcast",
    ),
    "thalamus/reentrant_workspace.py": (
        "BUS",
        "defines ReentrantWorkspace; likewise self-check writes",
    ),
    "cortex/live_brain.py": (
        "ACTION",
        "the tick driver: owns tick() and constructs the AutonomousArm it ticks",
    ),
    "brainstem/autonomous_loop.py": (
        "ACTION",
        "the arm's wake loop -- step() is the single action gate P6 says faculties all report to",
    ),
}
DEFAULT_ROLE = ("FACULTY", "not the bus and not the action path")

# The self-check's floor. Asserted as a MINIMUM so that adding a write is fine and losing sight of one
# is red. It is a liveness assertion about the INSTRUMENT: a broken walk resolves to zero findings,
# which reads exactly like a clean tree.
#
# These are FLOORS and are deliberately NOT re-pinned to the current measurement. `global_workspace`
# writes 32 today against a floor of 14 -- raising the floor to 32 would make this gate red on a
# legitimate removal of a self-check broadcast, which is the opposite of what a liveness floor is for.
# Raise an entry only when the OLD value can no longer be reached.
#
# CORRECTED 2026-08-06: `KNOWN_READERS` had NO entry for `rpm_solver` or `spike_chat_lake`, so READ
# detection could have stopped matching in both modules entirely and this self-check would have
# stayed green. Both are floored at 1 -- the smallest number that still detects a total loss of
# matching, chosen low for the same reason the write floors are low.
KNOWN_WRITERS = {
    "thalamus/global_workspace.py": 14,
    "thalamus/reentrant_workspace.py": 11,
    "reasoning/train_of_thought.py": 4,
    "brainstem/autonomous_loop.py": 3,
    "cortex/brain.py": 3,
    # cortex/competence_executive.py LEFT this table 2026-08-12: WorkspaceBroadcast was retired
    # (its retirement record in that module) and the module now performs ZERO workspace writes --
    # a floor of 3 would have fired "produced ZERO writes" on a correct deletion, the exact
    # failure mode :258-259 warns a liveness floor must not manufacture.
    "cortex/access_consciousness.py": 1,
    "reasoning/rpm_solver.py": 1,
    "reasoning/spike_chat_lake.py": 1,
    # relpath co-edited 2026-08-12 with the module's promotion (#10 lane 4e) -- the key is a
    # POSIX relpath under src/snn_data_lake/, so the move alone would have made `missing` fire
    # with "known writer module(s) produced ZERO writes".
    "hippocampus/developmental_loop_brain.py": 1,
}
KNOWN_READERS = {
    "thalamus/global_workspace.py": 3,
    "reasoning/train_of_thought.py": 2,
    "thalamus/reentrant_workspace.py": 2,
    "cortex/brain.py": 1,
    "reasoning/rpm_solver.py": 1,
    # INSTRUMENT CHANGE 2026-08-21 (T034): the counted broadcast() reads of BOTH
    # reasoning/spike_chat_lake.py (its _attended became the channel's adapter) and
    # cortex/live_brain.py (its `thought` property routes through the channel) MOVED into
    # substrate/report_channel.py -- the reads did not stop happening, they changed
    # modules onto the ONE declared route. The floor follows the read to its new home;
    # leaving the old rows fired INSTRUMENT DEAD on a correct P2 consolidation, and
    # dropping the floor entirely would blind the walk to a total loss of the tree's
    # cross-module workspace consumption.
    "substrate/report_channel.py": 1,
}

# --------------------------------------------------------------------------------------------------
# THE BASELINE. Every entry is DERIVED -- it is this walk's own output on the tree at 4fc5fb1c,
# recorded on 2026-08-06 with ZERO SLACK. None of these is a cap chosen to fit; a cap with slack is a
# gate that no realistic regression trips, which is the failure mode this project has named three
# times. Each entry says what it asserts, why that shape, and what makes it go red.
#
# WHY A RATCHET AND NOT AN ARRIVAL. faculty-to-faculty reads are 0 today. An assert of the form "> 0"
# is RED on arrival -- it fails the tree rather than gating it, and a gate that must be disabled on
# the day it lands is not a gate. A ratchet asserts the numbers never get WORSE, has a real reachable
# red (make it worse and it fires), and makes P6 monotone, which is the property that actually drives
# the tree to zero.
#
# RAISING A CEILING IS AN ADMISSION, NOT A FIX. If a change must raise one of these, the commit says
# which faculty it added without a consumer and why that was the cheapest honest option.
# --------------------------------------------------------------------------------------------------
BASELINE = {
    # Both ceilings ratcheted DOWN 7 -> 6 on 2026-08-12: WorkspaceBroadcast's retirement removed
    # cortex/competence_executive from both censuses (it no longer constructs or writes into any
    # workspace instance). A ratchet keeps ground it gains -- left at 7, the self-check's
    # one-mutant regression could no longer cross the ceiling, i.e. the criterion would have had
    # no reachable red (measured before this edit: BOTH mutations dead, 6 -> 7 does not cross 7;
    # the three green controls were green at either ceiling -- an earlier 0-of-3-controls reading
    # was caused by the then-undeclared hippocampus<->thalamus import pair, a different red, and
    # is NOT evidence for this ratchet; corrected at the same day's review pass).
    "private_instance_modules": 6,
    # RAISED 6 -> 7 on 2026-08-21 (specs/022 T033), and per this gate's own rule a raise is
    # an ADMISSION, so here it is: thalamus/ignition_instrument.py constructs private
    # ReentrantWorkspace SUBJECTS inside its sweep and ablation protocols and nothing else
    # reads them -- BY DESIGN: it is a measurement instrument (dormancy_gate EXEMPT as
    # such), and an instrument whose subjects were shared would be measuring traffic it
    # does not control. This static walk cannot attribute function-local instances (its
    # own disclosed residual false-red channel), and teaching it a new role machinery for
    # one module would be a bigger honesty risk than this recorded admission. The ceiling
    # still traps the NEXT unread writer: a real faculty submitting into a bus nobody
    # reads lands on 8 and reddens.
    "unread_writer_modules": 7,
    "cross_module_reads": 1,
    "undeclared_import_pairs": 11,
    # RAISED 21 -> 24 on 2026-08-20 (specs/022 T021a), and per this gate's own rule a raise is
    # an ADMISSION, so here it is: the insula tract + transthalamic routes (8 new declared
    # edges: nts_rvlm->thal_relay->insula->{gwt, ctx_pfc}; ctx_sensory->pulvinar->ctx_pfc;
    # ctx_pfc<->md) add 3 region pairs whose code cannot reach each other YET -- the anatomy
    # is deliberately declared AHEAD of its wiring, because the wiring is US1's job (the
    # interoception channel producer at T025+ imports along the tract) and declaring routes
    # first is what lets the manifest and this gate police that wiring when it lands. The
    # cheapest honest alternative -- fake imports to keep the number flat -- is the exact
    # force-wiring this gate's doctrine forbids. The ceiling must fall back to <=21 when US1
    # wires the tract; leaving it at 24 after T039 is a defect, recorded here so the number
    # cannot quietly become the new normal.
    #
    # 24 -> 23 AT T039f: ONE of those three pairs is now wired. `cortex/brain.py` imports the
    # insula terminus to feed it the declared body_in frame, so (cortex, insula) moved from
    # declared-but-unreachable to implemented. TWO REMAIN, and they are named so the remaining
    # debt cannot be read as done: (medulla, thalamus) and (thalamus, insula) -- the tract relays
    # nts_rvlm -> thal_relay -> insula, and `thal_relay` is a connectome NODE with no module of
    # its own, so closing those two is an architectural call inside global_workspace.py and is
    # NOT owed by this commit.
    #
    # RATCHETING THIS IS NOT BOOKKEEPING. Left at 24 the criterion has NO REACHABLE RED: the
    # self-check's constructed mutation raises the count by exactly one, so from 23 it lands ON
    # the ceiling rather than over it and the mutation stops firing. `--self-check` went red for
    # precisely that reason and named this criterion. P8: a criterion with no reachable red is
    # deleted, not weakened -- so the honest move is to lower the ceiling until the red returns,
    # which is what a ratchet is for.
    "unimplemented_declared_pairs": 23,
}

# name -> (kind, human sentence, why this shape and not the obvious neighbour)
CRITERIA = {
    "private_instance_modules": (
        "ceiling",
        "modules holding a PRIVATE workspace instance must not rise",
        "The MODULE count, not the SITE count. Measured across five commits the site count rose "
        "twice -- on the two commits that did this project's actual P6 work, because a bus "
        "self-check that constructs twelve throwaway instances is not twelve more private buses. "
        "The module count stayed flat across all five and rises only when a NEW module builds its "
        "own bus, which is the regression P6 actually names.",
    ),
    "unread_writer_modules": (
        "ceiling",
        "writer modules writing only into instances nobody else reads must not rise",
        "This is P6 condition (b) counted directly: each of these posts into a workspace nothing "
        "else ever reads. A new faculty that submits without acquiring a consumer moves it, and "
        "nothing else does -- adding a write to an ALREADY-unread module leaves it flat, which is "
        "the 'a rising write count is not progress' rule expressed as an assert. Counted on the "
        "instance WRITTEN INTO, not the instance OWNED: the owner-centric form went RED on a new "
        "faculty submitting into the one shared bus live_brain reads, i.e. on the P6 fix itself "
        "(constructed and run; 7 -> 8, exit 1). Zero delta at HEAD.",
    ),
    "cross_module_reads": (
        "floor",
        "reads that reach an instance owned by another module must not fall",
        "The only read shape that can carry one faculty's contribution to another; everything else "
        "is a module talking to itself through an object it built. Floored rather than ceilinged "
        "because losing the last one is the regression. Requires the alias fix: before it, a "
        "no-op local-alias refactor dropped this from 1 to 0.",
    ),
    "undeclared_import_pairs": (
        "ceiling",
        "region pairs that import each other with NO declared connectome route must not rise",
        "P6 condition (a) -- 'reachable through a TYPED DECLARED route'. A new cross-region wire "
        "must either be declared in connectome.py or it is a regression; declaring it costs one "
        "_e(...) line and makes the architecture answerable. Undirected because an import is a "
        "REACH, not a signal direction (measured: 8 of 23 directed mismatches were exact reverses "
        "of a declared edge, so the directed form is dominated by that artifact and is dishonest).",
    ),
    "unimplemented_declared_pairs": (
        "ceiling",
        "declared connectome routes with NO import path in either direction must not rise",
        "The other side of the same diff: a declared pathway that no module can possibly implement, "
        "because no code in either region reaches the other. Necessary-condition only -- an import "
        "does not prove the pathway carries anything, so this finds the declarations that provably "
        "carry NOTHING, never certifies the rest.",
    ),
}


# --------------------------------------------------------------------------------------------------
# Paths. Every one takes an explicit root so the self-check and the history sweep can point this walk
# at an extracted tree; a gate that can only ever look at one tree cannot be shown to go red.
# --------------------------------------------------------------------------------------------------
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_dir(root):
    return os.path.join(root, "src", "snn_data_lake")


def rel(path, root):
    """Repo-relative POSIX path under src/snn_data_lake/, e.g. 'cortex/brain.py'."""
    return os.path.relpath(path, source_dir(root)).replace(os.sep, "/")


def chain(node):
    """The dotted segments of a receiver expression, outermost last.

    `self.arm.workspace` -> ['self', 'arm', 'workspace'].  A non-name root yields None, which is how
    a receiver this walk cannot attribute stays visibly unattributed instead of being guessed at."""
    segs = []
    cur = node
    while isinstance(cur, ast.Attribute):
        segs.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    segs.append(cur.id)
    segs.reverse()
    return segs


def ctor_class(value):
    """The workspace class this expression constructs, or None.

    Walks the whole value subtree so `GlobalWorkspace(seed=s) if on else None` is seen -- the exact
    shape of `autonomous_loop.py`'s construction, the module that owns the only instance anything
    else reads."""
    for sub in ast.walk(value):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else None)
            )
            if name in WORKSPACE_CLASSES:
                return name
    return None


# --------------------------------------------------------------------------------------------------
# Pass 1 -- vocabulary: which receiver names denote a workspace, who constructs one, and which local
# names are ALIASES for a longer receiver chain.
# --------------------------------------------------------------------------------------------------
class _BindVisitor(ast.NodeVisitor):
    """Collect workspace constructions, `self.<attr> = SomeClass(...)` bindings, and local aliases.

    The second kind is what lets `self.arm.workspace` resolve to the module that owns that instance
    rather than to whichever module happens to use the token `workspace`. The third is what stopped
    two real cross-module reads from being filed as self-reads: `ws = self.brain.workspace` followed
    by `ws.broadcast()` reads ANOTHER module's instance, and a walk that sees only `ws` cannot say so."""

    def __init__(self, relpath):
        self.rel = relpath
        self.instances = []  # (token, class_name, lineno)
        self.attr_class = {}  # 'arm' -> 'AutonomousArm'
        self.alias = {}  # 'ws' -> 'self.brain.workspace'

    def _handle(self, targets, value, lineno):
        cls = ctor_class(value)
        for tgt in targets:
            segs = chain(tgt)
            if segs is None:
                continue
            token = segs[-1]
            if cls is not None:
                self.instances.append((token, cls, lineno))
                continue
            if isinstance(value, ast.Call) and segs[0] == "self" and len(segs) == 2:
                fn = value.func
                cname = (
                    fn.id
                    if isinstance(fn, ast.Name)
                    else (fn.attr if isinstance(fn, ast.Attribute) else None)
                )
                if cname and cname[:1].isupper():
                    self.attr_class[token] = cname
                continue
            # ALIAS: `ws = self.brain.workspace`, or `ws = getattr(self.brain, "workspace", None)`,
            # or the ATTRIBUTE form `self._ws = self.arm.workspace`. A dotted source of length >= 2
            # is the thing worth keeping; a bare name adds nothing.
            #
            # THE ATTRIBUTE FORM WAS ADDED 2026-08-06 AFTER IT WAS MEASURED AS A FALSE RED. Only the
            # LOCAL form was resolved, so hoisting `self.arm.workspace` out of `live_brain.thought`
            # into `self._ws` -- a refactor that removes no route, changes no behaviour and reaches
            # the same object -- took `cross_module_reads` from 1 to 0 and `unread_writer_modules`
            # from 7 to 8, reddening TWO ratchets on a no-op. Constructed and run on a worsened tree
            # copy before this branch existed. Zero `self.<attr> = <workspace chain>` sites exist at
            # HEAD, so this moves NO number here; like the local-alias hop before it, its whole value
            # is a closed false-red channel, and a zero-delta change is exactly the kind that looks
            # skippable.
            if not (len(segs) == 1 or (len(segs) == 2 and segs[0] == "self")):
                continue
            src = _alias_source(value)
            if src is not None and len(src) >= 2 and src[-1] != token:
                self.alias.setdefault(token, ".".join(src))

    def visit_Assign(self, node):
        self._handle(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._handle([node.target], node.value, node.lineno)
        self.generic_visit(node)


def _alias_source(value):
    """The dotted chain this expression aliases, or None.

    Handles the two forms the tree actually uses: a bare attribute chain (`self.brain.workspace`) and
    the defensive `getattr(obj, "workspace", default)` that `spike_chat_lake` reaches for."""
    if isinstance(value, ast.Attribute):
        return chain(value)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "getattr"
        and len(value.args) >= 2
        and isinstance(value.args[1], ast.Constant)
        and isinstance(value.args[1].value, str)
    ):
        base = chain(value.args[0])
        if base is not None:
            return base + [value.args[1].value]
    return None


# --------------------------------------------------------------------------------------------------
# Pass 2 -- sites: every workspace read, write and registration
# --------------------------------------------------------------------------------------------------
class _SiteVisitor(ast.NodeVisitor):
    def __init__(self, relpath, vocab):
        self.rel = relpath
        self.vocab = vocab  # receiver tokens known to denote a workspace, tree-wide
        self.sites = []  # (kind, lineno, method, receiver_text, specialist_names)
        self._cls = []
        self._as_func = set()
        self._loop = {}  # loop var -> the constant strings it takes, for `for s in ("a","b")`

    def visit_ClassDef(self, node):
        self._cls.append(node.name)
        self.generic_visit(node)
        self._cls.pop()

    def visit_For(self, node):
        """Bind `for _spec in ("action", "language")` so the subscriber NAMES stay resolvable.

        Without this, `brain.py` registers two specialists under an unreadable loop variable and the
        drained-vs-registered diff below can only shrug -- which is how a subscriber nothing ever
        drains stays invisible."""
        bound = None
        if isinstance(node.target, ast.Name) and isinstance(node.iter, (ast.Tuple, ast.List)):
            vals = [
                e.value
                for e in node.iter.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if vals and len(vals) == len(node.iter.elts):
                bound = node.target.id
                self._loop[bound] = tuple(vals)
        self.generic_visit(node)
        if bound is not None:
            self._loop.pop(bound, None)

    def _names(self, args):
        """The specialist name(s) an API call refers to: a string literal, or a bound loop variable."""
        if not args:
            return ()
        arg = args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return (arg.value,)
        if isinstance(arg, ast.Name) and arg.id in self._loop:
            return self._loop[arg.id]
        return ()

    def _receiver(self, node):
        """(token, text) if `node` is a workspace receiver, else None.

        Inside a workspace class body `self` IS the workspace -- without this the bus modules' own
        reads of `self.current` are invisible, and a walk that cannot see the bus read itself has no
        standing to report that nobody else does."""
        segs = chain(node)
        if segs is None:
            return None
        token = segs[-1]
        if token in self.vocab:
            return token, ".".join(segs)
        if token == "self" and self._cls and self._cls[-1] in WORKSPACE_CLASSES:
            return token, "self"
        return None

    def visit_Call(self, node):
        self._as_func.add(id(node.func))
        if isinstance(node.func, ast.Attribute):
            recv = self._receiver(node.func.value)
            if recv is not None:
                arity = len(node.args) + len(node.keywords)
                kind = API.get((node.func.attr, arity))
                if kind is not None:
                    self.sites.append(
                        (kind, node.lineno, node.func.attr, recv[1], self._names(node.args))
                    )
        elif isinstance(node.func, ast.Name):
            # T034 (with the ("snapshot", 1) API row): a bare-name accessor whose BUS is
            # its first ARGUMENT -- the report channel's read shape. The argument is
            # attributed exactly as a method receiver would be; without this branch the
            # walk reported cross_module_reads 1 -> 0 on a refactor that GREW consumption.
            arity = len(node.args) + len(node.keywords)
            kind = API.get((node.func.id, arity))
            if kind is not None and node.args:
                recv = self._receiver(node.args[0])
                if recv is not None:
                    self.sites.append((kind, node.lineno, node.func.id, recv[1], ()))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if (
            id(node) not in self._as_func
            and node.attr == READ_ATTR
            and isinstance(node.ctx, ast.Load)
        ):
            recv = self._receiver(node.value)
            if recv is not None:
                self.sites.append(("READ", node.lineno, READ_ATTR, recv[1], ()))
        self.generic_visit(node)


# --------------------------------------------------------------------------------------------------
# The walk. DETERMINISM: directory names are sorted as well as file names, and every dict this
# function builds is iterated in sorted key order downstream. os.walk yields directories in whatever
# order the filesystem hands them over, and `class_module` / `resolve_owner` both resolve ties by
# first-seen -- so an unsorted walk is a census that can differ between two machines on one tree.
# --------------------------------------------------------------------------------------------------
def walk(root, overlay=None):
    """Return a dict of the raw AST census.

    `overlay` maps a repo-relative path under src/snn_data_lake (e.g. 'cortex/mutant.py') to source
    text, or to None to delete that file from the walk. It is how `--self-check` constructs a RED
    without touching anything on disk -- the same string-copy discipline
    `tools/dependency_ledger.py --self-check` uses on pyproject."""
    overlay = dict(overlay or {})
    found, trees = [], {}
    for dirpath, dirnames, filenames in os.walk(source_dir(root)):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                found.append(os.path.join(dirpath, fn))

    failures = []
    for path in sorted(found):
        relpath = rel(path, root)
        if relpath in overlay:
            continue  # the overlay decides this file's fate below
        try:
            with open(path, encoding="utf-8") as fh:
                trees[relpath] = ast.parse(fh.read(), filename=path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append((relpath, type(exc).__name__))

    n_found = len(found)
    for relpath in sorted(overlay):
        src = overlay[relpath]
        on_disk = os.path.exists(os.path.join(source_dir(root), relpath.replace("/", os.sep)))
        if src is None:
            if not on_disk:
                raise ValueError(f"overlay deletes {relpath}, which is not in the tree")
            n_found -= 1
            continue
        if not on_disk:
            n_found += 1
        trees[relpath] = ast.parse(src, filename=f"<overlay:{relpath}>")

    class_module = {}
    instances = defaultdict(list)  # relpath -> [(token, class, lineno)]
    attr_class = {}  # relpath -> {attr: ClassName}
    alias = {}  # relpath -> {local: 'self.brain.workspace'}
    vocab = set()
    for relpath in sorted(trees):
        tree = trees[relpath]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_module.setdefault(node.name, relpath)
        bv = _BindVisitor(relpath)
        bv.visit(tree)
        if bv.instances:
            instances[relpath] = bv.instances
            vocab.update(tok for tok, _c, _l in bv.instances)
        attr_class[relpath] = bv.attr_class
        alias[relpath] = bv.alias

    # An ALIAS TOKEN IS A WORKSPACE RECEIVER. Without this the alias map is inert: `_w = self.arm.
    # workspace` records the expansion, but `_w` was never a CONSTRUCTION target so `_w.broadcast()`
    # never became a site at all. Measured before this line existed: rewriting the one cross-module
    # read as a local alias -- a refactor that removes no route -- dropped `cross_module_reads` from
    # 1 to 0 and would have reddened the build. Fixed-point because an alias may name another alias.
    for _ in range(4):
        grew = False
        for relpath in sorted(alias):
            for tok, src in sorted(alias[relpath].items()):
                if tok not in vocab and src.split(".")[-1] in vocab:
                    vocab.add(tok)
                    grew = True
        if not grew:
            break

    sites = defaultdict(list)
    for relpath in sorted(trees):
        sv = _SiteVisitor(relpath, vocab)
        sv.visit(trees[relpath])
        if sv.sites:
            sites[relpath] = sorted(sv.sites, key=lambda s: (s[1], s[0], s[2], s[3]))

    return {
        "sites": sites,
        "instances": instances,
        "attr_class": attr_class,
        "alias": alias,
        "class_module": class_module,
        "n_found": n_found,
        "n_parsed": len(trees),
        "failures": failures,
    }


def role_of(relpath):
    return ROLES.get(relpath, DEFAULT_ROLE)


def resolve_owner(relpath, receiver_text, w):
    """Which module owns the workspace instance this receiver reaches, or None if unattributable.

    Three hops, in order:
      * ALIAS, local form -- `ws` expands to the `self.brain.workspace` it was assigned from.
      * ALIAS, attribute form -- `self._ws` expands to the `self.arm.workspace` it was assigned
        from. Without this a behaviour-preserving hoist of a read into an attribute reddened two
        ratchets; see the note in `_BindVisitor._handle`.
      * `self.<attr> = SomeClass(...)` -- `self.arm.workspace` read in live_brain walks
        `arm` -> AutonomousArm -> brainstem/autonomous_loop.py, and only THEN matches `workspace`
        against that module's own constructions.
    Token matching alone is ambiguous here -- two modules bind the token `workspace` -- and an
    ambiguous answer to "who is read by nobody" is no answer, so ambiguity returns None and the
    caller reports it as UNATTRIBUTABLE rather than folding it into the self-read bucket."""
    segs = receiver_text.split(".")
    aliases = w["alias"].get(relpath, {})
    expanded = aliases.get(segs[0])
    if expanded is not None:
        segs = expanded.split(".") + segs[1:]
    elif len(segs) >= 2 and segs[0] == "self" and aliases.get(segs[1]) is not None:
        segs = aliases[segs[1]].split(".") + segs[2:]
    token = segs[-1]
    here = relpath
    if len(segs) >= 3 and segs[0] == "self":
        cls = w["attr_class"].get(relpath, {}).get(segs[1])
        if cls is not None and cls in w["class_module"]:
            here = w["class_module"][cls]
    if any(tok == token for tok, _c, _l in w["instances"].get(here, [])):
        return here
    owners = sorted(
        m for m, inst in w["instances"].items() if any(t == token for t, _c, _l in inst)
    )
    return owners[0] if len(owners) == 1 else None


# --------------------------------------------------------------------------------------------------
# HALF TWO -- the connectome diff. connectome.py is parsed by AST, never imported: this walk must run
# against an EXTRACTED tree that is not on sys.path, must not drag numpy/torch onto a GPU-less CI
# runner, and must not construct a Brain.
#
# WHAT IT MEASURES, AND THE HONEST LIMIT. `connectome.py` declares signal flow source -> target. An
# import declares that code in region A can REACH code in region B. THESE ARE DIFFERENT RELATIONS,
# and treating one as the other is dishonest in a way that is measurable, not arguable: of the 23
# directed pairs the code imports but the connectome does not declare, EIGHT are the exact reverse of
# a declared edge, because a faculty that pulls (`ws.broadcast()`) imports its upstream and a faculty
# that pushes (`ws.submit()`) imports its downstream. A directed diff is therefore dominated by that
# artifact.
#
# So the diff is UNDIRECTED, and it is a NECESSARY-CONDITION test, stated as such:
#   * FINDING 1 (declared, unimplemented) -- the connectome declares a pathway between two regions
#     whose modules do not reach each other in EITHER direction. No code path exists, so the declared
#     route provably carries nothing. It does NOT follow that the other declarations do carry
#     something; an import is a reach, not a conversation.
#   * FINDING 2 (implemented, undeclared) -- two regions reach each other in code with no declared
#     pathway between them. This is the more interesting half and P6 condition (a) exactly: reachable,
#     but not through a TYPED DECLARED route. Note what dominates it -- 8 of the 11 pairs have
#     `brainstem` as an endpoint, which is `autonomous_loop` (the single action gate) importing
#     faculties to drive them. That is the defect P6 names, arrived at from the structural side and
#     independently of the workspace half.
# --------------------------------------------------------------------------------------------------
def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def parse_connectome(root, source=None):
    """AST-parse connectome.py's NODES map and EDGES list. Returns (node_region, edges, effector)."""
    path = os.path.join(source_dir(root), "connectome.py")
    if source is None:
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
    tree = ast.parse(source, filename=path)
    node_region, edges = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "NODES" in names and isinstance(node.value, ast.Dict):
                for k, v in zip(node.value.keys, node.value.values, strict=True):
                    key = _const_str(k)
                    if key is None or not isinstance(v, ast.Tuple) or not v.elts:
                        continue
                    node_region[key] = _const_str(v.elts[0])  # None for the effector nodes
            if "EDGES" in names and isinstance(node.value, ast.List):
                for elt in node.value.elts:
                    if not (isinstance(elt, ast.Call) and len(elt.args) >= 2):
                        continue
                    s, t = _const_str(elt.args[0]), _const_str(elt.args[1])
                    if s and t:
                        edges.append((s, t))
    effector = [(s, t) for s, t in edges if not node_region.get(s) or not node_region.get(t)]
    return node_region, edges, effector


def region_imports(root, overlay=None):
    """Undirected + directed region->region edges implemented by imports under src/snn_data_lake/."""
    overlay = dict(overlay or {})
    pkg = source_dir(root)
    directed = set()
    paths = []
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                paths.append(os.path.relpath(os.path.join(dirpath, fn), pkg).replace(os.sep, "/"))
    for relpath in sorted(set(paths) | set(overlay)):
        if relpath in overlay and overlay[relpath] is None:
            continue
        if "/" not in relpath:
            continue  # top-level modules (connectome.py, brain_atlas.py) belong to no region
        src_region = relpath.split("/")[0]
        if relpath in overlay:
            text = overlay[relpath]
        else:
            try:
                with open(os.path.join(pkg, relpath.replace("/", os.sep)), encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        here_pkg = relpath.split("/")[:-1]  # package dirs under snn_data_lake, e.g. ['amygdala']
        for node in ast.walk(tree):
            targets = []  # each entry is the dotted path BELOW snn_data_lake
            if isinstance(node, ast.Import):
                targets = [
                    a.name.split(".")[1:]
                    for a in node.names
                    if a.name.split(".")[0] == "snn_data_lake"
                ]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                if node.module and node.module.split(".")[0] == "snn_data_lake":
                    targets = [node.module.split(".")[1:]]
            elif isinstance(node, ast.ImportFrom):
                # RELATIVE IMPORT. `from ..cerebellum import x` inside amygdala/ reaches cerebellum
                # just as surely as the absolute spelling, and was INVISIBLE here until 2026-08-06:
                # the old branch required parts[0] == "snn_data_lake", which a relative import never
                # has. Constructed and run: the identical amygdala->cerebellum wire fired the
                # `undeclared_import_pairs` ceiling when spelled absolutely and passed clean when
                # spelled relatively, so the ceiling was evadable by import style alone. There are
                # ZERO relative imports under src/snn_data_lake today, so this moves NO number --
                # it closes an evasion channel, which is the only thing a zero-delta fix ever buys.
                up = node.level - 1
                if up > len(here_pkg):
                    continue
                base = here_pkg[: len(here_pkg) - up] if up else list(here_pkg)
                tail = node.module.split(".") if node.module else []
                targets = [base + tail] if tail else [base + [a.name] for a in node.names]
            for parts in targets:
                if parts and parts[0] != src_region:
                    directed.add((src_region, parts[0]))
    return directed


def connectome_diff(root, overlay=None, conn_source=None):
    node_region, edges, effector = parse_connectome(root, source=conn_source)
    regions = sorted({r for r in node_region.values() if r})
    declared_dir = set()
    for s, t in edges:
        rs, rt = node_region.get(s), node_region.get(t)
        if rs and rt and rs != rt:
            declared_dir.add((rs, rt))
    declared = {tuple(sorted(p)) for p in declared_dir}
    imported_dir = {
        (a, b) for a, b in region_imports(root, overlay) if a in regions and b in regions
    }
    imported = {tuple(sorted(p)) for p in imported_dir}
    return {
        "regions": regions,
        "n_nodes": len(node_region),
        "n_edges": len(edges),
        "n_effector": len(effector),
        "declared_dir": declared_dir,
        "declared": declared,
        "imported_dir": imported_dir,
        "imported": imported,
        "unimplemented": sorted(declared - imported),  # FINDING 1
        "undeclared": sorted(imported - declared),  # FINDING 2
        "reversed_artifact": sorted(
            (a, b) for a, b in (imported_dir - declared_dir) if (b, a) in declared_dir
        ),
    }


# --------------------------------------------------------------------------------------------------
# Measure -- everything the report and the criteria need, derived once.
# --------------------------------------------------------------------------------------------------
def measure(root, overlay=None, conn_source=None, api=None):
    # dict() on BOTH sides deliberately: `api = API` would alias the global, and the `API.clear()`
    # below would then empty the very table it is about to be restored from -- measured, and the
    # result was 0 writes with the gate still printing OK, which is the silent-degradation shape this
    # repo keeps catching. The liveness self-check caught it; the report did not.
    api = dict(API) if api is None else dict(api)
    saved = dict(API)
    try:
        API.clear()
        API.update(api)
        w = walk(root, overlay)
    finally:
        API.clear()
        API.update(saved)

    per = {}
    for relpath, ss in w["sites"].items():
        per[relpath] = {
            "WRITE": [s for s in ss if s[0] == "WRITE"],
            "READ": [s for s in ss if s[0] == "READ"],
            "REGISTER": [s for s in ss if s[0] == "REGISTER"],
        }
    instances = w["instances"]
    w_mods = sorted(m for m, v in per.items() if v["WRITE"])
    r_mods = sorted(m for m, v in per.items() if v["READ"])

    cross, selfread, unattributable = [], [], []
    for relpath in sorted(per):
        for _k, ln, meth, recv, _n in per[relpath]["READ"]:
            owner = resolve_owner(relpath, recv, w)
            rec = (relpath, ln, meth, recv, owner)
            if owner is None:
                unattributable.append(rec)
            elif owner != relpath:
                cross.append(rec)
            else:
                selfread.append(rec)

    # BOTH endpoints must be FACULTY. A one-sided filter (reader role only) goes green on a faculty
    # reading the ACTION path's workspace, which P6 verbatim calls the defect, not the satisfaction.
    fac_cross = [
        c for c in cross if role_of(c[0])[0] == "FACULTY" and role_of(c[4])[0] == "FACULTY"
    ]
    fac_cross_reader_only = [c for c in cross if role_of(c[0])[0] == "FACULTY"]

    bus_inst_mods = sorted(m for m in instances if role_of(m)[0] == "BUS")
    n_inst = sum(len(v) for v in instances.values())
    priv_mods = sorted(m for m in instances if role_of(m)[0] != "BUS")

    registered, drained = {}, set()
    for relpath in sorted(per):
        for _k, ln, _m, _r, names in per[relpath]["REGISTER"]:
            for nm in names:
                registered.setdefault(nm, []).append(f"{relpath}:{ln}")
        for _k, _ln, meth, _r, names in per[relpath]["READ"]:
            if meth in ("broadcast_to", "inbox"):
                drained.update(names)
    undrained = sorted(n for n in registered if n not in drained)

    # WHICH INSTANCE DOES EACH WRITE GO INTO? Attributed exactly the way reads already are, and for
    # a measured reason. The former test was `m not in read_owners`, where read_owners are the
    # modules that OWN a cross-read instance -- so a module writing into a bus it does NOT own could
    # never qualify, no matter who read that bus. That made the criterion RED on the P6 FIX: a new
    # faculty submitting into the one shared workspace `cortex/live_brain.py:313` reads was counted
    # as "read by nobody". Constructed and run on a worsened tree copy: `unread_writer_modules`
    # 7 -> 8, exit 1, on the single change P6 exists to encourage. This project rejected the private-
    # instance SITE count for exactly that property, and the same disqualification applies here.
    # Attributing the WRITE closes it and moves NOTHING at HEAD (measured: the same 7 modules).
    consumed_by = {}
    for reader, _ln, _meth, _recv, owner in cross:
        consumed_by.setdefault(owner, set()).add(reader)
    write_targets = {
        m: {resolve_owner(m, recv, w) for _k, _ln, _me, recv, _n in per[m]["WRITE"]} for m in w_mods
    }
    unread = [
        m
        for m in w_mods
        if role_of(m)[0] != "BUS"
        and not any(
            o is not None and (consumed_by.get(o, set()) - {m}) for o in write_targets.get(m, ())
        )
    ]
    # An UNATTRIBUTABLE write target resolves to "unread", i.e. strict. That is deliberate and it is
    # the residual false-red channel of this criterion: a faculty handed a workspace by dependency
    # injection (`def __init__(self, ws)`) writes through a receiver this static walk cannot
    # attribute, and is counted here. Reported below so the number is never mistaken for a finding
    # about the tree. The lenient alternative -- excluding unknowns -- was rejected because it lets
    # any new writer evade the ceiling by taking its bus as a parameter.
    unread_only_unattributable = [
        m for m in unread if write_targets.get(m) and all(o is None for o in write_targets[m])
    ]

    conn = connectome_diff(root, overlay, conn_source)

    return {
        "walk": w,
        "per": per,
        "instances": instances,
        "w_mods": w_mods,
        "r_mods": r_mods,
        "writes": sum(len(v["WRITE"]) for v in per.values()),
        "reads": sum(len(v["READ"]) for v in per.values()),
        "regs": sum(len(v["REGISTER"]) for v in per.values()),
        "inst_sites": n_inst,
        "bus_inst_mods": bus_inst_mods,
        "priv_sites": n_inst - sum(len(instances[m]) for m in bus_inst_mods),
        "priv_mods": priv_mods,
        "cross": cross,
        "selfread": selfread,
        "unattributable": unattributable,
        "fac_cross": fac_cross,
        "fac_cross_reader_only": fac_cross_reader_only,
        "registered": registered,
        "undrained": undrained,
        "unread": unread,
        "unread_only_unattributable": unread_only_unattributable,
        "write_targets": write_targets,
        "conn": conn,
        # the five asserted quantities, isolated so a criterion cannot drift from what it reads
        "metrics": {
            "private_instance_modules": len(priv_mods),
            "unread_writer_modules": len(unread),
            "cross_module_reads": len(cross),
            "undeclared_import_pairs": len(conn["undeclared"]),
            "unimplemented_declared_pairs": len(conn["unimplemented"]),
        },
    }


def evaluate(metrics, baseline=None):
    """Return [(name, kind, baseline, actual, ok)] in a fixed order."""
    baseline = BASELINE if baseline is None else baseline
    out = []
    for name in sorted(CRITERIA):
        kind = CRITERIA[name][0]
        base, actual = baseline[name], metrics[name]
        ok = actual <= base if kind == "ceiling" else actual >= base
        out.append((name, kind, base, actual, ok))
    return out


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def report(res, root):
    per, instances = res["per"], res["instances"]
    w = res["walk"]

    print("=" * 98)
    print("P6 GATE -- workspace write:read topology + connectome diff, over src/snn_data_lake/")
    print("=" * 98)
    print(f"root: {root}")
    print(f"scanned {w['n_parsed']} of {w['n_found']} .py files parsed")
    for relpath, why in w["failures"]:
        print(f"  PARSE FAILURE  {relpath}  ({why})")
    print(
        f"workspace classes tracked: {len(WORKSPACE_CLASSES)} "
        f"({', '.join(sorted(WORKSPACE_CLASSES))})"
    )

    print(
        "\n--- WORKSPACE INSTANCES (each construction is a SEPARATE object; a faculty's submit is"
    )
    print("    invisible to any faculty holding a different instance) ---")
    for relpath in sorted(instances):
        role, _why = role_of(relpath)
        toks = ", ".join(f"{t}={c}@:{ln}" for t, c, ln in instances[relpath])
        print(f"  {relpath:<52} [{role:<7}] {toks}")
    print(f"\n  {res['inst_sites']} instantiation sites across {len(instances)} modules")
    print(
        f"  {res['priv_sites']} of {res['inst_sites']} sites, across {len(res['priv_mods'])} "
        "modules, are in modules whose declared ROLE is not BUS -- that many PRIVATE, mutually "
        "invisible buses"
    )
    print(
        "  NOTE the SITE count is reported and NEVER asserted: it rose on the two commits that did "
        "this project's\n  actual P6 work, because a bus self-check building throwaway instances is "
        "not more private buses. The\n  MODULE count is the ratchet."
    )

    print("\n--- PER-MODULE TOPOLOGY ---")
    print(f"  {'module':<52} {'role':<8} {'W':>3} {'R':>3} {'sub':>4}  owns")
    for relpath in sorted(per):
        role, _why = role_of(relpath)
        v = per[relpath]
        owns = "yes" if relpath in instances else "NO (reaches into another module's instance)"
        print(
            f"  {relpath:<52} {role:<8} {len(v['WRITE']):>3} {len(v['READ']):>3} "
            f"{len(v['REGISTER']):>4}  {owns}"
        )

    bus_mods = [m for m in res["w_mods"] if role_of(m)[0] == "BUS"]
    bus_writes = sum(len(per[m]["WRITE"]) for m in bus_mods)
    print("\n" + "=" * 98)
    print("P6 TOPOLOGY -- HALF ONE: does a faculty CONSUME another faculty's output? (condition b)")
    print("=" * 98)
    print(f"  writes  : {res['writes']} across {len(res['w_mods'])} modules")
    print(f"  reads   : {res['reads']} across {len(res['r_mods'])} modules")
    print(
        f"  subscribe registrations: {res['regs']}  (NOT counted as reads -- registering is not consuming)"
    )
    print(
        f"  {bus_writes} of {res['writes']} writes are inside the {len(bus_mods)} bus modules' own "
        f"self-checks, so real broadcast is {res['writes'] - bus_writes} of {res['writes']} writes"
    )
    print(
        f"\n  self-reads     : {len(res['selfread'])} of {res['reads']}  (a module reading an instance it built)"
    )
    print(f"  cross-module   : {len(res['cross'])} of {res['reads']}")
    for reader, ln, meth, recv, owner in res["cross"]:
        role, why = role_of(reader)
        print(f"      {reader}:{ln}  {recv}.{meth}  <- instance owned by {owner}")
        print(f"        reader role {role}: {why}")
    print(
        f"  unattributable : {len(res['unattributable'])} of {res['reads']}  (receiver token bound in "
        "more than one module; reported, never counted as a self-read)"
    )
    for reader, ln, meth, recv, _o in res["unattributable"]:
        print(f"      {reader}:{ln}  {recv}.{meth}")

    print(
        f"\n  named subscribers registered: {len(res['registered'])}; drained by a broadcast_to/inbox "
        f"read: {len(res['registered']) - len(res['undrained'])} of {len(res['registered'])}"
    )
    for nm in res["undrained"]:
        where = ", ".join(res["registered"][nm])
        roles = {role_of(loc.rsplit(":", 1)[0])[0] for loc in res["registered"][nm]}
        note = " (bus self-check control)" if roles == {"BUS"} else ""
        print(f"      NEVER DRAINED  {nm!r}  registered at {where}{note}")

    print(f"\n  FACULTY-to-FACULTY reads (BOTH endpoints FACULTY): {len(res['fac_cross'])}")
    print(
        f"  ... under the OLD one-sided filter (reader role only): {len(res['fac_cross_reader_only'])}"
        "  -- the difference is\n      a faculty reading the ACTION path's bus, which P6 calls the "
        "DEFECT, not the satisfaction."
    )
    if not res["fac_cross"]:
        print("      Zero-way at the faculty level, which is the condition P6 forbids.")

    print("\n" + "-" * 98)
    print(f"  WRITERS READ BY NOBODY: {len(res['unread'])} of {len(res['w_mods'])} writer modules")
    print("-" * 98)
    for m in res["unread"]:
        role, _why = role_of(m)
        lines = ", ".join(f":{ln}" for _k, ln, _meth, _r, _n in per[m]["WRITE"])
        tgt = res["write_targets"].get(m, set())
        where = ", ".join(sorted(o or "UNATTRIBUTABLE" for o in tgt))
        print(f"    {m:<52} [{role:<7}] {len(per[m]['WRITE'])} write(s) at {lines}")
        print(f"      writes into: {where}")
    if res["unread_only_unattributable"]:
        print(
            f"\n    {len(res['unread_only_unattributable'])} of these are counted ONLY because this "
            "static walk cannot attribute\n    the instance they write into -- the residual false-red "
            "channel of this ceiling, not a finding\n    about the tree: "
            + ", ".join(res["unread_only_unattributable"])
        )

    # ---- HALF TWO ---------------------------------------------------------------------------------
    c = res["conn"]
    print("\n" + "=" * 98)
    print(
        "P6 TOPOLOGY -- HALF TWO: is the route TYPED AND DECLARED? (condition a) -- connectome diff"
    )
    print("=" * 98)
    print(
        f"  connectome.py: {c['n_nodes']} nodes / {c['n_edges']} edges across {len(c['regions'])} "
        f"regions ({c['n_effector']} effector edges have no region and are dropped)"
    )
    print(
        f"  declared: {len(c['declared_dir'])} directed region pairs -> {len(c['declared'])} undirected"
    )
    print(
        f"  imported: {len(c['imported_dir'])} directed region pairs -> {len(c['imported'])} undirected"
    )
    print(
        f"  UNDIRECTED because an import is a REACH, not a signal direction: {len(c['reversed_artifact'])} "
        "of the directed\n  mismatches are the exact REVERSE of a declared edge, so a directed diff "
        "measures that artifact."
    )
    print(
        f"\n  FINDING 1 -- declared route, NO import path in either direction: "
        f"{len(c['unimplemented'])} of {len(c['declared'])}"
    )
    print("    (no code path exists, so the declaration provably carries nothing)")
    for a, b in c["unimplemented"]:
        print(f"      {a} <-> {b}")
    print(
        f"\n  FINDING 2 -- regions that reach each other in code with NO declared route: "
        f"{len(c['undeclared'])} of {len(c['imported'])}"
    )
    print("    (P6 condition (a) exactly: reachable, but not through a TYPED DECLARED route)")
    for a, b in c["undeclared"]:
        d = [f"{x}->{y}" for x, y in ((a, b), (b, a)) if (x, y) in c["imported_dir"]]
        print(f"      {a} <-> {b}   (imports: {', '.join(d)})")
    bs = [p for p in c["undeclared"] if "brainstem" in p]
    if bs:
        print(
            f"\n    {len(bs)} of {len(c['undeclared'])} have `brainstem` as an endpoint -- the single "
            "action gate reaching\n    faculties directly. That is the defect P6 names, found from "
            "the structural side."
        )
    return res


def report_criteria(res, baseline=None):
    print("\n" + "=" * 98)
    print("P6 CRITERIA -- ratchets. Baseline DERIVED at 4fc5fb1c on 2026-08-06, zero slack.")
    print("=" * 98)
    rows = evaluate(res["metrics"], baseline)
    for name, kind, base, actual, ok in rows:
        op = "<=" if kind == "ceiling" else ">="
        print(
            f"  [{'OK ' if ok else 'RED'}] {name:<30} {actual:>3} {op} {base:<3}  {CRITERIA[name][1]}"
        )
    failed = [r for r in rows if not r[4]]
    print(
        f"\n  NOT ASSERTED, deliberately: faculty-to-faculty reads = {len(res['fac_cross'])}. "
        "A 'must not FALL'\n  ratchet on a count that is already 0 has NO reachable red -- nothing "
        "can drive it below zero. P8:\n  a criterion with no reachable red is DELETED, not weakened. "
        "It becomes a floor the day it is non-zero."
    )
    print(
        f"  NOT ASSERTED: private instance SITES = {res['priv_sites']}, raw reads = {res['reads']}, "
        f"raw writes = {res['writes']}.\n  The site count fires on the FIX (measured: it rises on 2 "
        "of the 5 steps across the 6 commits swept in\n  this file's header, both of them P6 work). "
        "The raw read count is +157% an artifact of one name added\n  to the API table. Neither is a "
        "quantity a ratchet may hold."
    )
    return failed


# --------------------------------------------------------------------------------------------------
# Self-check. Two halves, because a gate owes both.
#   LIVENESS -- is the walk still matching? A broken walk resolves to zero findings, which reads
#     exactly like a clean tree, and that is the failure this half guards.
#   REACHABLE RED -- can each asserting criterion actually FIRE? Every criterion is mutated against an
#     in-memory OVERLAY of the tree (nothing on disk is touched, the same string-copy discipline
#     `tools/dependency_ledger.py --self-check` uses on pyproject) and is required to go red. A
#     mutation that changes nothing fails loudly, so a rotted anchor cannot present as a vacuous green.
# --------------------------------------------------------------------------------------------------
def verify_roles(root):
    """Re-derive each declared ROLE from the source so a stale label goes red."""
    problems = []
    for relpath, (role, _why) in sorted(ROLES.items()):
        full = os.path.join(source_dir(root), relpath.replace("/", os.sep))
        if not os.path.exists(full):
            problems.append(f"{relpath}: declared {role} but the file does not exist")
            continue
        with open(full, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=full)
        names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        if role == "BUS" and not (names & WORKSPACE_CLASSES):
            problems.append(f"{relpath}: declared BUS but defines no workspace class")
        if role == "ACTION" and not ({"tick", "step"} & names):
            problems.append(f"{relpath}: declared ACTION but defines neither tick() nor step()")
    return problems


# A module that builds its OWN bus and posts to it, read by nobody. One new FACULTY module, so it
# moves private-instance MODULES and unread-writer MODULES by exactly one each.
_MUTANT_PRIVATE = '''"""overlay mutant -- a new faculty that builds its own private bus and shouts into it."""

from snn_data_lake.thalamus.global_workspace import GlobalWorkspace


class MutantFaculty:
    def __init__(self):
        self._p6mut_ws = GlobalWorkspace()

    def think(self, percept):
        self._p6mut_ws.submit("mutant", percept, 0.5)
'''

# A module in one declared region importing another declared region with no pathway between them.
# amygdala <-> cerebellum is in neither the declared set nor the imported set at 4fc5fb1c.
_MUTANT_UNDECLARED = '''"""overlay mutant -- a cross-region wire with no declared connectome pathway."""

from snn_data_lake.cerebellum import jepa_world_model


def use(x):
    return jepa_world_model is not None and x
'''

# The same wire spelled RELATIVELY. Until 2026-08-06 this passed clean while the absolute spelling
# above went red, so the connectome ceiling was evadable by import style alone.
_MUTANT_UNDECLARED_REL = '''"""overlay mutant -- the same cross-region wire, spelled as a relative import."""

from ..cerebellum import jepa_world_model


def use(x):
    return jepa_world_model is not None and x
'''

# --------------------------------------------------------------------------------------------------
# GREEN CONTROLS. A criterion that fires on a real regression AND on correct work has no
# discriminating power -- it has no green state, so its red carries no information. Each mutation
# above proves a criterion CAN fire; each control below proves it does NOT fire on work that is
# right. Both directions were measured, and the first two controls were RED before the fixes landed
# in this revision, which is exactly why they are pinned here rather than described in prose.
# --------------------------------------------------------------------------------------------------

# The P6 FIX itself: a new faculty whose output IS consumed, because it submits into the one shared
# workspace `cortex/live_brain.py:313` reads. Under the former owner-centric `unread_writer_modules`
# this went 7 -> 8 and exited 1 -- the gate reddening the single change P6 exists to encourage.
_CONTROL_SHARED_BUS = '''"""overlay control -- a new faculty that submits into the SHARED bus live_brain reads."""

from snn_data_lake.brainstem.autonomous_loop import AutonomousArm


class ControlSharedBusFaculty:
    def __init__(self, seed=0):
        self.arm = AutonomousArm(seed=seed, workspace=True)

    def think(self, percept):
        self.arm.workspace.submit("control_shared", percept, 0.7)
'''


def _mutate_text(text, old, new, what):
    if old not in text:
        raise AssertionError(
            f"self-check mutation for {what} did not apply: the anchor {old!r} is not in the source. "
            "A mutation that changes nothing cannot prove a red -- fix the anchor, do not skip it."
        )
    return text.replace(old, new, 1)


def liveness(root, res):
    """Return (lines, problems). A problem means this walk has stopped SEEING, not that the tree is
    clean -- the two are indistinguishable in the output, which is the whole reason this exists.

    RUN IN `--assert` MODE TOO, not only under `--self-check`. That is not symmetry for its own sake:
    it was measured. A worsened tree copy written with a UTF-8 BOM made two injected regression
    modules fail to parse; the walk dropped them, `scanned 400 of 402` scrolled past, and the two
    criteria they were built to trip stayed GREEN. A file that will not parse makes the tree look
    CLEANER, so a parse failure has to be a RED and not a log line."""
    per, instances, w = res["per"], res["instances"], res["walk"]
    lines, problems = [], []

    if w["failures"]:
        problems.append(
            "a source file failed to parse, so its sites are silently missing from every count: "
            + ", ".join(f"{r} ({x})" for r, x in w["failures"])
        )
    if w["n_parsed"] != w["n_found"]:
        problems.append(
            f"parsed {w['n_parsed']} of {w['n_found']} files -- the walk is dropping files"
        )
    if w["n_found"] <= len(KNOWN_WRITERS):
        problems.append(
            f"found only {w['n_found']} .py files under src/snn_data_lake -- the walk lost the tree"
        )
    lines.append(
        f"  files        : {w['n_parsed']} of {w['n_found']} parsed, {len(w['failures'])} failures"
    )

    missing, shrunk = [], []
    for relpath, floor in sorted(KNOWN_WRITERS.items()):
        got = len(per.get(relpath, {}).get("WRITE", []))
        if got == 0:
            missing.append(relpath)
        elif got < floor:
            shrunk.append(f"{relpath}: {got} of {floor} expected writes")
    if missing:
        problems.append(
            "known writer module(s) produced ZERO writes -- the receiver vocabulary or the API table "
            "has stopped matching, and a broken walk reports a clean tree: " + ", ".join(missing)
        )
    if shrunk:
        problems.append(
            "known writer module(s) lost writes. If the writes were genuinely removed, lower the "
            "floor in KNOWN_WRITERS and say so; do not leave a walk that half-matches: "
            + "; ".join(shrunk)
        )
    lines.append(
        f"  writers seen : {len(KNOWN_WRITERS)} of {len(KNOWN_WRITERS)} known writer modules"
    )

    missing_r = [
        relpath
        for relpath, floor in sorted(KNOWN_READERS.items())
        if len(per.get(relpath, {}).get("READ", [])) < floor
    ]
    if missing_r:
        problems.append(
            "known reader module(s) lost reads -- READ detection (arity-0 broadcast/broadcast_all, "
            "broadcast_to, inbox, the `current` property) has stopped matching: "
            + ", ".join(missing_r)
        )
    lines.append(
        f"  readers seen : {len(KNOWN_READERS)} of {len(KNOWN_READERS)} known reader modules"
    )

    built = {c for inst in instances.values() for _t, c, _l in inst}
    if built != WORKSPACE_CLASSES:
        problems.append(
            f"constructed workspace classes {sorted(built)} != tracked {sorted(WORKSPACE_CLASSES)} "
            "-- a class was renamed, retired, or is no longer being detected"
        )
    lines.append(f"  classes built: {len(built)} of {len(WORKSPACE_CLASSES)} tracked classes")

    # (The dedicated 3-arg-broadcast-still-WRITE criterion left with WorkspaceBroadcast's
    # retirement 2026-08-12: its subject class no longer exists, and the arity split it guarded
    # now has one broadcast arity. The KNOWN_WRITERS floors above remain the liveness check on
    # submit-detection.)
    lines.append("  arity split  : 0-arg broadcast still READ (3-arg WRITE retired with its class)")

    role_problems = verify_roles(root)
    if role_problems:
        problems.append("declared ROLES no longer match the source: " + "; ".join(role_problems))
    lines.append(
        f"  roles        : {len(ROLES)} of {len(ROLES)} declared roles re-derived from AST"
    )

    c = res["conn"]
    if not (c["n_edges"] > 0 and c["declared"]):
        problems.append(
            "connectome.py parsed to ZERO declared region pairs -- the NODES/EDGES literal shape "
            "changed and half two is silently measuring nothing"
        )
    if not c["imported"]:
        problems.append("the region import graph is EMPTY -- half two is measuring nothing")
    lines.append(
        f"  connectome   : {c['n_edges']} edges -> {len(c['declared'])} declared pairs, "
        f"{len(c['imported'])} imported pairs"
    )
    return lines, problems


def self_check(root):
    """Return (liveness_lines, red_rows). Raises AssertionError on a liveness failure."""
    res = measure(root)
    lines, problems = liveness(root, res)
    assert not problems, "; ".join(problems)

    # determinism: the same tree twice must give identical metrics
    again = measure(root)
    assert again["metrics"] == res["metrics"], (
        f"NON-DETERMINISTIC: {res['metrics']} != {again['metrics']} on the same tree"
    )
    lines.append("  determinism  : two walks of the same tree gave identical metrics")

    # ---- reachable red, one constructed mutation per asserting criterion -------------------------
    live_src = os.path.join(source_dir(root), "cortex", "live_brain.py")
    with open(live_src, encoding="utf-8") as fh:
        live_text = fh.read()
    with open(
        os.path.join(source_dir(root), "reasoning", "spike_chat_lake.py"), encoding="utf-8"
    ) as fh:
        chat_text = fh.read()  # T034: the second counted read the mutation must delete
    with open(os.path.join(source_dir(root), "connectome.py"), encoding="utf-8") as fh:
        conn_text = fh.read()

    mutations = [
        (
            "private_instance_modules",
            "add a new FACULTY module that constructs its own GlobalWorkspace",
            {"cerebellum/_p6_selfcheck_mutant.py": _MUTANT_PRIVATE},
            None,
        ),
        (
            "unread_writer_modules",
            "the same new module submits to its own bus and nobody reads it",
            {"cerebellum/_p6_selfcheck_mutant.py": _MUTANT_PRIVATE},
            None,
        ),
        (
            "cross_module_reads",
            "delete BOTH cross-module reads (live_brain.thought + the chat adapter)",
            {
                # T034: three counted reads exist now (thought + the attending-branch
                # composition + the chat adapter, all through the report channel); the
                # red needs the count BELOW 1, so the mutation deletes every one -- one
                # anchor was enough only while one read was.
                "cortex/live_brain.py": _mutate_text(
                    _mutate_text(
                        live_text,
                        'return snapshot(self.arm.workspace)["content"]',
                        "return None",
                        "cross_module_reads (thought)",
                    ),
                    "rec = snapshot(self.arm.workspace)",
                    'rec = {"content": None, "availability": 0.0, "strength": 0.0}',
                    "cross_module_reads (attending branch)",
                ),
                "reasoning/spike_chat_lake.py": _mutate_text(
                    chat_text,
                    "rec = snapshot(ws)",
                    'rec = {"content": None, "strength": 0.0}',
                    "cross_module_reads (chat half)",
                ),
            },
            None,
        ),
        (
            "undeclared_import_pairs",
            "add an amygdala module importing cerebellum (no declared pathway between them)",
            {"amygdala/_p6_selfcheck_mutant.py": _MUTANT_UNDECLARED},
            None,
        ),
        (
            "undeclared_import_pairs",
            "the SAME wire spelled as a RELATIVE import (was invisible before 2026-08-06)",
            {"amygdala/_p6_selfcheck_mutant.py": _MUTANT_UNDECLARED_REL},
            None,
        ),
        (
            "unimplemented_declared_pairs",
            "declare an amygdala->cerebellum edge in connectome.py that no import implements",
            None,
            _mutate_text(
                conn_text,
                "EDGES = [",
                'EDGES = [\n    _e("bla", "cb_ctx", "p6 self-check mutant", "exc", "relay"),',
                "unimplemented_declared_pairs",
            ),
        ),
    ]

    with open(os.path.join(source_dir(root), "cortex", "brain.py"), encoding="utf-8") as fh:
        brain_text = fh.read()

    # Each control must leave EVERY criterion green. A criterion with no green state has no
    # discriminating power, and the first two of these were measured RED before this revision.
    controls = [
        (
            "no-op refactor: hoist the cross-module read into an attribute",
            "same object, same call, same route -- `self._gw = self.arm.workspace` then "
            "`snapshot(self._gw)`",
            {
                "cortex/live_brain.py": _mutate_text(
                    live_text,
                    'return snapshot(self.arm.workspace)["content"]',
                    'self._gw = self.arm.workspace\n        return snapshot(self._gw)["content"]',
                    "control: attribute-hoist refactor",
                )
            },
        ),
        (
            "THE P6 FIX: a new faculty submits into the SHARED bus live_brain reads",
            "its output IS consumed by another module; a gate that reddens this is worse than no "
            "gate",
            {"hippocampus/_p6_selfcheck_control.py": _CONTROL_SHARED_BUS},
        ),
        (
            "one MORE private bus SITE inside an ALREADY-private module",
            "the site count rose on the two commits that did this project's real P6 work; the "
            "MODULE count is the ratchet precisely so this stays green",
            {
                "cortex/brain.py": _mutate_text(
                    brain_text,
                    "            self._reentrant = ReentrantWorkspace(",
                    "            self._p6_control_bus = GlobalWorkspace()\n"
                    "            self._reentrant = ReentrantWorkspace(",
                    "control: extra private site",
                )
            },
        ),
    ]

    rows = []
    for name, how, overlay, conn_source in mutations:
        mres = measure(root, overlay=overlay, conn_source=conn_source)
        before, after = res["metrics"][name], mres["metrics"][name]
        moved = after != before
        fired = not dict((r[0], r[4]) for r in evaluate(mres["metrics"]))[name]
        rows.append((name, how, before, after, moved, fired))

    control_rows = []
    for what, why, overlay in controls:
        cres = measure(root, overlay=overlay)
        reds = [r[0] for r in evaluate(cres["metrics"]) if not r[4]]
        control_rows.append((what, why, reds))
    return lines, rows, control_rows


# --------------------------------------------------------------------------------------------------
# ---- T023 (specs/022): the manifest-sourced walk, REPORT-ONLY beside the AST walk ----------
# distinct-from: the AST walk above -- same QUANTITY class (faculty-to-faculty routes), sourced
#   from the DECLARED anatomy (substrate/wiring_manifest's committed artifact) instead of parsed
#   code. Two-step retirement per the task: this lands report-only with a parity table; the AST
#   walk retires and baselines re-record in a SEPARATE commit only after faculties migrate onto
#   declared ports and the two views converge. The ratchet verdict below NEVER reads this.
def manifest_parity_lines(res):
    """The parity table: declared (manifest) vs measured (AST/census), day-one divergence
    EXPECTED and recorded as the migration baseline -- never a red."""
    try:
        from snn_data_lake.substrate.wiring_manifest import ARTIFACT, WiringManifest

        m = WiringManifest.load_artifact(ARTIFACT, strict=False)
    except Exception as exc:  # noqa: BLE001 - availability is the finding, whatever the cause
        return [
            f"  manifest-sourced walk UNAVAILABLE ({type(exc).__name__}: {exc}) -- "
            "the AST walk remains authoritative"
        ]
    declared_edges, declared_ports = len(m.edges), len(m.ports)
    measured_fac = len(res.get("fac_cross", ()))
    lines = [
        "  MANIFEST-SOURCED WALK (T023, report-only -- the ratchet verdict never reads this):",
        f"    declared port-to-port edges : {declared_edges:>4}   vs measured faculty-to-faculty reads : {measured_fac}",
        f"    declared typed ports        : {declared_ports:>4}   (no AST analogue -- the declared surface itself)",
        f"    declared REF escape hatches : {len(m.refs()):>4}   (audited census)",
    ]
    if declared_edges == 0 and declared_ports == 0:
        lines.append(
            "    parity: EXPECTED-DIVERGENT (manifest EMPTY until T025's first registrant; this "
            "table is the migration baseline, and the AST walk retires only after convergence)"
        )
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="P6 gate: workspace topology + connectome diff")
    ap.add_argument("--assert", dest="do_assert", action="store_true", help="exit 1 on any RED")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="instrument liveness + a constructed RED per criterion",
    )
    ap.add_argument("--root", default=repo_root(), help="repo root to measure (default: this repo)")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)

    if args.self_check:
        lines, rows, control_rows = self_check(root)
        # T023: the manifest-sourced walk's own liveness -- a CONSTRUCTED in-memory manifest
        # with one edge must move its count 0 -> 1 (the "constructed regression moves BOTH"
        # arm; the AST side's movement is the mutation rows below). Report-only, so it joins
        # the liveness print, never the criteria verdict.
        try:
            from snn_data_lake.substrate.wiring_manifest import WiringManifest

            probe = WiringManifest()
            n0 = len(probe.edges)
            probe.declare_edge("a", "out", "b", "in")
            lines.append(
                f"  manifest-sourced walk ALIVE: constructed edge moved the declared count "
                f"{n0} -> {len(probe.edges)} (report-only beside the AST walk, T023)"
            )
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  manifest-sourced walk DEAD: {type(exc).__name__}: {exc}")
        print("=" * 98)
        print("P6 GATE SELF-CHECK -- is the walk alive, and can every criterion FIRE?")
        print("=" * 98)
        print("LIVENESS")
        for line in lines:
            print(line)
        print("\nREACHABLE RED -- one constructed mutation per asserting criterion")
        print("  (in-memory overlay; nothing on disk is touched)")
        bad = []
        for name, how, before, after, moved, fired in rows:
            tag = "RED OK " if (moved and fired) else ("NO MOVE" if not moved else "NO RED ")
            print(f"  [{tag}] {name:<30} {before} -> {after}   {how}")
            if not (moved and fired):
                bad.append(name)
        n_ok = len(rows) - len(bad)
        print(
            f"\n  {n_ok} of {len(rows)} constructed mutations fired, covering "
            f"{len({r[0] for r in rows})} of {len(CRITERIA)} asserting criteria."
        )

        print("\nGREEN CONTROLS -- correct work that must NOT fire")
        print("  (a criterion red on a regression AND red on the fix has no discriminating power)")
        broken = []
        for what, why, reds in control_rows:
            print(f"  [{'GREEN  ' if not reds else 'FALSE RED'}] {what}")
            print(f"              {why}")
            if reds:
                print(f"              FIRED: {', '.join(reds)}")
                broken.append(what)
        print(
            f"\n  {len(control_rows) - len(broken)} of {len(control_rows)} controls stayed green."
        )

        if bad:
            print(
                "  A criterion whose mutation did not make it fire has NO reachable red. P8: it is "
                "DELETED, not weakened: " + ", ".join(bad)
            )
        if broken:
            print(
                "  A criterion that fires on correct work is worse than no gate -- this project "
                "rejected the\n  private-instance SITE count for exactly that. Fix the criterion, "
                "do not raise the baseline: " + "; ".join(broken)
            )
        return 1 if (bad or broken) else 0

    res = measure(root)
    report(res, root)
    for line in manifest_parity_lines(res):
        print(line)
    failed = report_criteria(res)
    if args.do_assert:
        # LIVENESS FIRST. A dropped file makes every count SMALLER, so a broken walk presents as a
        # cleaner tree and every ceiling above it goes green for the wrong reason.
        _lines, problems = liveness(root, res)
        for p in problems:
            print(f"\n  INSTRUMENT DEAD: {p}")
        if failed or problems:
            if failed:
                print(
                    "\n  P6 GATE FAIL: "
                    + "; ".join(f"{n} {a} vs baseline {b}" for n, _k, b, a, _o in failed)
                )
            if problems:
                print(
                    "\n  P6 GATE FAIL: the walk is not seeing the tree, so no green above is "
                    "meaningful."
                )
            print(
                "  Raising a baseline is an ADMISSION, not a fix: say which faculty was added "
                "without a consumer\n  and why that was the cheapest honest option."
            )
            return 1
        print("\n  P6 GATE OK: all 5 ratchets hold.")
        return 0
    print(
        "\n  (reporting only -- pass --assert to make this exit 1 on a RED, which is how CI runs it)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
