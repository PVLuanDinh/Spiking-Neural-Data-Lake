r"""arc_census -- INSTRUMENT OF RECORD for the fix-combinator trajectory census
(specs/006-fixed-point-gf, FR-016; measured 2026-08-02, RETIRED 2026-08-03).

This tool produced the published trajectory census in
`specs/006-fixed-point-gf/census.md` (raw transcript `census_run.log`): on the full
evaluation split at bench budgets, with a `fix:` twin for every pooled entry,

  * 400 of 400 tasks completed; 500918 of 500918 trajectories classified
    (converged / cycle / diverged / budget -- zero unclassified, zero non-termination)
  * solved 12 of 400 -- the byte-identical solved-ID set as the no-combinator baseline
  * 0 of 12 winning programs contained a fix step; mean winning-fix depth: NONE
  * non-trivial convergence EXISTS (5699 of 214169 converged trajectories at depth 3..19)
    but closes nothing: the unsolved tasks lack VOCABULARY, not depth

The combinator was therefore RETIRED with that null (verdict.md; FR-010 forbids a
flag-parked survivor), and the telemetry hooks this tool drove (`synthesize(stats=...)`)
were removed with it. To REPRODUCE the census, check out the feature-measurement commit
(the one carrying census.md and this tool's driving version -- `git log --follow
tools/arc_census.py`) and run:

  python tools/arc_census.py <training|evaluation> [limit] [--data DIR]

At the current HEAD this tool intentionally only prints the record above (SKIP-not-zero:
a retired mechanism yields no fresh figure, and this tool will not fake one).
"""

import sys

RECORD = __doc__


def main(argv=None):
    print("SKIP arc_census: the fix combinator was retired with its null verdict")
    print("  (specs/006-fixed-point-gf/verdict.md; census record in census.md).")
    print("  Reproduce at the feature-measurement commit -- see this tool's docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
