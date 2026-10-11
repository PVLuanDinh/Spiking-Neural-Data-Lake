# G5 fixture producer-to-reader handoff repair, 2026-10-11

Executor: codex-sndl-coord-1010, temporarily assigned by the owner at board #803.
Source before patch: f6ba7403dd968136c3a7f91647df476469c41fd8, sndl-g5 checkout.
Integrator before this work: clean 1cebf5794a3e39b2bd28ecb0766bdf908ff0da66.
Existing g5-readout-variants.md remains claimed by G5 and is unchanged by this executor.
Its measured working-file SHA256 is d67ccc9a1bbb797cb595bf862d47f0313a3c4cee86bb11e9e0a49cf096c69bf7.

## Preregistration and rejected plan

V1 #813/#814 was rejected by independent raw-source reviews #816/#817 and
#824/#825: ALL_ARMS had incorrectly been required in non-S0C1 cells even though
the real producer emits adaptor arms/controls only on S0C1. Claude independently
found the same fault. No engine edits or fixture execution occurred under V1.
Its original plan/checker and task #812 pins remain unchanged, never accepted.

Fresh V2 task #822 protected 186 entries including all tests,
runner/configuration, existing adaptor/readout modules, and the external plan/checker.
PIN DISCLOSURE: specs/022-unified-brain-arc/SEEDS.md was an incorrect path and
was pinned as <missing>; the actual registry is the canonical board's SEEDS.md,
which this task did not protect. Root conftest.py was also <missing>, while the
actual tests/conftest.py was protected by the tests/ directory pin. Do not treat
these absent-path pins as real-file protection or repin the task retroactively.
The frozen checker independently asserts the exact eight fixture RNG seeds,
and the source's FIXTURE_SEEDS declaration stayed unchanged. The seed registry
was not edited; no broader registry-integrity claim is made here.
Plan #823 received two independent cited agreements #826/#831 before execution.
Named serial grant #832: CODEX-G5-HANDOFF-1011-V2. Frozen plan and acceptance:

- C:/Users/vinhl/.config/opencode/sndl-runs/2026-10-09-step5/rows/g5-codex-handoff-repair-plan-2026-10-11-v2.md
- C:/Users/vinhl/.config/opencode/sndl-runs/2026-10-09-step5/rows/g5-codex-handoff-check-2026-10-11-v2.py

The source's actual CLI syntax is `--cells S0C0,S0C1,S1C0,S1C1 --epochs 1`
(one comma-separated argument), not four separate arguments. The checker calls
real pd_records directly and passes its unchanged return object into real read_d1.

## Change

Removed the reader's two-line silent PD-R3 missing-rule continue. A missing required
rule now reaches _diffs and raises ReadHalt caused by KeyError. Added one constructed
whole-rule-absent check to the existing self_check. Producer/cert and dormancy prose
now distinguishes a CLI entry from evidence that it ran; cert_for is a separate
function, not an additional pd_records return. No statistics, thresholds, rules,
cert semantics, seed registrations, tests, safety code or live behavior changed.

## Witnessed baseline red, once

Ran the frozen checker with --reader-red-only against unchanged f6ba7403.
It returned rc1 because the silent skip accepted the constructed clean record set
with DELTA absent on every S0C1. Exact failure:

    AssertionError: constructed whole PD-R3 rule absent: missing required arm was silently accepted

Elapsed 13.0 s; start available physical RAM 1.856 GiB, minimum 1.842 GiB.
This was constructed reader evidence only; real pd_records did not run yet.
The specific cause was then repaired as preregistered, not an outcome-selected retry.

## Actual fixture handoff, once

Python313 supervisor invoked the free-threaded interpreter at
D:/AI-project/spiking-neural-data-lake/.venv-lake/Scripts/python.exe, with -B/-s,
PYTHON_GIL=0, no user site, the sndl-g5 source/import/cwd/interpreter pins, and a
fresh throwaway NORD_HOME. Shared home.require_isolated_home ran before lake-binding
imports. The frozen supervisor sampled RAM every second, required at least 1.5 GiB
at launch and would terminate below 1.0 GiB.

Real pd_records built eight fixture RNG seeds 20261003..20261010, the four reader
cells S0C0/S0C1/S1C0/S1C1, epochs=1. Expected cells were read from the reader AST;
expected arms were derived from ALL_ARMS/RULES and checked against producer._arms_of
per cell. The resulting real record object passed the unconditional vocabulary
check and was consumed unchanged by real read_d1.

Fixture verdict: {"PD-R0": false, "halted": true}. This is the reader's legitimate
early hypothesis halt, NOT a missing-field halt and NOT later hypothesis evidence.
The checker supplied expansion/stateless cert flags false as fixture inputs;
it did not establish actual expansion certification or stateless-S1 solve facts.

The same command witnessed all three named reds:

1. Constructed clean reader records with an entire PD-R3 DELTA arm absent:
   ReadHalt caused by KeyError; the earlier silent omission no longer occurs.
2. Actual producer output with RSTDP_D removed from the first seed's S0C0:
   actual read_d1 raised ReadHalt caused by KeyError at its first contrast.
3. Actual producer output with DELTA absent on every S0C1: the unconditional
   vocabulary assertion failed with "seed 20261003 S0C1: missing required arms ['DELTA']".
   This remains reachable despite the actual fixture's PD-R0 early halt.

Each actual-output mutation was reverted. Serialized SHA256 equality and a repeated
read verified restoration to the original object/verdict. No producer values were
fabricated or altered to make a hypothesis pass. Command rc0, elapsed 19.0 s,
start available RAM 2.596 GiB, minimum 2.142 GiB.

## Focused verification

In a separately fresh guarded home under the same serialized lane and RAM supervisor:
the modified reader self_check passed, then pytest ran only
tests/test_g5_part_d_reading.py and tests/test_g5_part_d_cells.py.
Result: 26 passed in 12.88 s; supervised elapsed 21.0 s, minimum RAM 2.664 GiB.
Focused pytest used addopts= and disabled nengo/pytest_cov plugins; it supplies no
fresh full-suite coverage claim. Ruff check passed and format --check reported
three files already formatted for the three changed source files.

No battery, held-out/deciding seed block, live lake write, daemon operation, E-STOP,
hardware command, scientific rerun, commit or push occurred before this result.
Result reviews, three per-arc lenses, independent closing, immutable score and
integration are separate pending steps. G5 module count on landing is 314 and
requires the separately scoped integrator count-pin repair; no protected test
was repinned in this narrow source task. This fixture result supplies no evidence
that the assembled self-growing-brain goal has been achieved or failed.

Exact captured tool-output artifacts (stdout/stderr plus chunk/exit metadata):

- C:/Users/vinhl/.config/opencode/sndl-runs/2026-10-09-step5/rows/g5-handoff-baseline-2026-10-11.log
- C:/Users/vinhl/.config/opencode/sndl-runs/2026-10-09-step5/rows/g5-handoff-fixture-2026-10-11.log
- C:/Users/vinhl/.config/opencode/sndl-runs/2026-10-09-step5/rows/g5-handoff-focused-2026-10-11.log

These preserve the actual exec launch/completion output; no reviewer runtime
reproduction is inferred from inspecting them. The scorer's mandatory fresh
acceptance rerun is still pending at the time of this result record.

## Independent close and immutable score

Fresh cited result agreements are board #841 and #843. The independent semantic,
adversarial and P2 scope lenses are #844, #842 and #847. Independent Codex closing
#847 and its cited agreement #846 found no blocker within this fixture scope;
their inspections did not reproduce engine runtime. All pin and early-halt
limitations above remain.

After reading the full closing, the executor saved off-board head
847:d2577a1a53c7e121b38588261b3f20e54158e58a0d5b8d33001bb9d1c9c3aa21.
The canonical scorer reran the original pinned acceptance command, with anchor
01656ec56d1f5117f96b13210b246dd2c53d782bcfb42681ce151f05ae4d08cb:
G5-HANDOFF-1011-V2: total=1 correct=1 integrity=1 scope=1 honest=1 head=#847.
The separate review check returned reviewed=1, items=2, min=2.
There was no scorer RED, waiver, changed acceptance command or protected-path edit.
This closes the narrow source repair. Landing, the separate alias-count pin,
full-plan batteries and actual assembled-brain goal validation remain pending.
