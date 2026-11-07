# D9 copy-and-rescale init for grown units: STRUCTURAL null, row deleted (D-389, 2026-11-06)

Row D9 (D-352, ARC7, g5-builder-2). PREREG `lanes/tickets/DESIGN-ROWS-1106/D9/PREREG.md` sha 75c3cc8cfbc1 (A6 holds
the argument). No D9 source or test was ever written; this file is the record of why, so the null is visible to
`gates/null_scale_gate.py` (P8 amendment item 4: a structural null is deleted without a re-test, and the record states
the argument, which owes its own check).

## The argument
D-379 fixed the copy gain as c = v_th / A_q with A_q taken over ALL buffer rows (residual-selected rows refused as a
move toward output preservation, D-362(c)). That gain is IDEMPOTENT: a copy u1 = c0 * u of a source u has
A_q(u1) = v_th by construction, so copying u1 again gives c1 = v_th / A_q(u1) = 1, i.e. an exact duplicate. Under the
engine default of one new unit per block and the copy-only rule (D-379 Q2), every unit after the first block is a
duplicate of an earlier unit's rate column. Any gain that places the copy against its threshold has this property;
a scale-invariant gain avoids it but leaves every unit on the first unit's direction, so no honest alternative was
proposed (D-389 item 2).

## The check the argument owes (run, with a known positive)
Probe `lanes/tickets/DESIGN-ROWS-1106/D9/idem_probe.py` (sha 9a456fdcb7e4), log `idem_probe.log` (sha 4d0b6c1c23ff),
LIGHTRUN #8939/#8940, rc 0, worktree src at int3 2f5a9c22, no seeds:
- float64: copy-of-copy gain c1 == 1.0 exactly; the copied row is byte-identical.
- float32 (engine dtype): c1 = 1 + 2 eps; the row differs by 7.2e-7, but the RATE column is byte-identical.
- Known positive: a planted non-idempotent gain 1.5 gives a distinct row AND a distinct rate column (fired).
- A grower walk admits 6 units with only 2 distinct rate columns (births: random, copy, then 4 copy-duplicates);
  duplicates keep a small positive ridge score, so the grower does not stop at ng_snn.py:548.

Record fields follow the PREREG (A6.2, D-239; D-402 item 4): seeds=1 is the one constructed unit (generator 9000), not
an RNG seed count; mde=n/a because a structural null has no re-test whose power a detectable effect could size.
NULL-RECORD: copy-and-rescale init for grown units (D9) | units=6 | ticks=n/a | seeds=1 | mde=n/a | retest=STRUCTURAL: the threshold-placing copy gain is idempotent, so every copy after the first block duplicates an earlier unit's rate column

Disposition: the D9 row is deleted (nothing was built, so nothing is removed from src). A mixed-pool row
(copies + random candidates) is NOT triggered: D-379 tied it to a confirmed cosine collapse in a campaign that never ran.
