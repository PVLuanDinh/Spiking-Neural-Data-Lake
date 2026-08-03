r"""arc_fixdepth_control -- INSTRUMENT OF RECORD for the fixed-K control arm of the
fix-combinator A/B (specs/006-fixed-point-gf, contract C18, SC-005; 2026-08-03).

The control arm's contract: re-run the split with every `fix:` entry replaced by an
exactly-K-applications entry, K = round(mean converged depth of WINNING fix programs in
the adaptive record run) -- K derived from run telemetry, never typed (P1).

MEASURED OUTCOME: the census (`specs/006-fixed-point-gf/census.md`) records 0 of 0
winning fix trajectories on the evaluation split, so K was UNDERIVABLE and this arm is
VACUOUS BY MEASUREMENT -- the strongest possible control result: there was no adaptivity
claim left to test, because adaptivity never earned a single solve. The word "adaptive"
therefore appears in no claim (SC-005 satisfied by absence), and the combinator was
retired with the null (verdict.md, FR-010).

At the current HEAD this tool prints that record and exits (SKIP-not-zero). To re-derive
it, check out the feature-measurement commit (`git log --follow
tools/arc_fixdepth_control.py`) and run:

  python tools/arc_fixdepth_control.py <training|evaluation> [on|off] [limit] [--data DIR]
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
CENSUS = os.path.join(_ROOT, "specs", "006-fixed-point-gf", "census.md")


def k_from_census(path=CENSUS):
    """K = round(mean winning-fix depth) read from the published census. None if the census is
    absent or records no winning fix trajectories (the measured outcome)."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"mean winning-fix depth: ([0-9][0-9.]*)", txt)
    if not m:
        return None
    return int(round(float(m.group(1))))


def main(argv=None):
    k = k_from_census()
    if k is None:
        print("SKIP arc_fixdepth_control: the census records 0 of 0 winning fix trajectories")
        print("  -- K is underivable and the control arm is vacuous by measurement (C18).")
        print("  The combinator was retired with the null (specs/006-fixed-point-gf/verdict.md).")
        return 0
    print(f"SKIP arc_fixdepth_control: census-derived K={k}, but the fix combinator was")
    print("  RETIRED with its null verdict; re-run at the feature-measurement commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
