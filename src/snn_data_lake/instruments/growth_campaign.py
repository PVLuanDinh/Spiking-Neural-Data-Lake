"""specs/022 T111 (G3) P4b: the growth campaign's learner-independent instrument (pre-registration 2.8, 2.12, 2.12a).

BUILT here: the seed families and `preflight` (a confirmatory seed is refused unless it is the exact registered
block AND the lead's written go is entered), the arm table, the restricted solve/failure score R(arm, s) with the
size-grid failure penalty (2.8, Codex b1 G3 #6), the superiority contrasts D_sel / D_rand under ruling 9 (a
degenerate sample reads NOT MET), the "fewer neurons" licence (both arms solve on >= 6 of 8 seeds), the
non-inferiority reading under ruling 9a (a degenerate sample reads by its common value), and 2.12a's campaign
isolation: one fresh home per (seed, arm), the live lake refused, one frozen ledger snapshot digest per seed that
every paired arm must equal, and a child launched without its home that refuses to start.

NOT built (blocked, stated plainly): the arm RUNNERS. They need the taught learner's grown-segment port on
`readout/interaction_learning.py`, which waits on R2's r2.v1 checkpoint step (owner question #2), so no arm of
2.8 has been run and no C16 reading exists. This module reads scores; it does not produce them.

The statistic is `paired_contrast`'s and the live-home guard `substrate/home.py`'s (P2: no second copy).
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from snn_data_lake.instruments import paired_contrast as pc
from snn_data_lake.substrate import home as home_mod

DEV_SEEDS = tuple(
    range(9000, 9100)
)  # exposed: design measurements, smoke runs (9040-9047 the T1 pilot)
CONFIRMATORY_SEEDS = tuple(range(9100, 9108))  # S = 8, the primary family, read once
RESERVE_SEEDS = tuple(range(9108, 9116))
FEED_SEEDS = tuple(range(9200, 9208))
FEED_RESERVE_SEEDS = tuple(range(9208, 9216))
T1_SEEDS = tuple(range(9300, 9308))  # T1 + Astra, PRESIZED included
T1_RESERVE_SEEDS = tuple(range(9308, 9316))
RED_SEEDS = tuple(range(9900, 10000))  # constructed reds and fixtures
FAMILIES = (
    CONFIRMATORY_SEEDS,
    RESERVE_SEEDS,
    FEED_SEEDS,
    FEED_RESERVE_SEEDS,
    T1_SEEDS,
    T1_RESERVE_SEEDS,
)
BLOCK = (9000, 10000)

#: the lead's written go for a confirmatory read; None until entered by a dated amendment, never by hand here
GO = None

ARMS = (
    "INCUMBENT",  # no growth: must stay unsolved
    "PRESIZED_FROZEN",  # lifecycle random source at the final size, never selected
    "PRESIZED_SELECTED",  # the targeted arm's candidate source, equal evaluations, chosen all at once
    "TARGETED",  # certificate-gated loop with trace-trained candidates
    "RANDOM",  # same schedule and batch, untrained, unselected units
    "SHAM",  # identical bookkeeping, output synapses held at zero
)
_SEPARATORS = ("/", "\\", ":")  # a family is one path component: no separator, no drive
MIN_SOLVING_SEEDS = (
    6  # "fewer neurons" is licensed only when both arms of a contrast solve on >= 6 of 8
)


# distinct-from: instruments/taught_campaign.py::preflight -- G2's guard is hard-wired to seeds 8000-8999,
#   G2's blocks and a repo-clean check; this one is G3's block 9000-9999 and its six registered families.
# distinct-from: instruments/carried_state_campaign.py::validate_seed_set -- G1(c)'s live/reserve bundles.
#   Each row owns its pre-registered seed families; the shared pieces (statistic, live-home guard) are imported.
def preflight(seeds, *, go=None):
    """Refuses a seed outside G3's block, a duplicate, a confirmatory family that is not the exact registered
    block, and a confirmatory read without the written go."""
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise PermissionError(f"seeds must be non-empty and distinct: {seeds}")
    outside = [s for s in seeds if not BLOCK[0] <= s < BLOCK[1]]
    if outside:
        raise PermissionError(f"seeds outside G3's block 9000-9999: {outside}")
    for family in FAMILIES:
        if set(seeds) & set(family):
            if sorted(seeds) != list(family):
                raise PermissionError(
                    f"seeds touch {family[0]}-{family[-1]} but are not that exact block"
                )
            if (go if go is not None else GO) is None:
                raise PermissionError(
                    "confirmatory read refused: the lead's written go is not entered"
                )


def failure_penalty(budget, batch):
    """The next admissible size-grid value above the budget (54 for batch 9, 53 for batch 1 at 52): an assigned
    score, never an observed size."""
    if budget < 1 or batch < 1:
        raise ValueError(f"budget and batch must be positive, got {budget!r}, {batch!r}")
    return (budget // batch + 1) * batch


def _solved(o):
    return bool(o["solved"] and o["future_solved"])


def restricted_score(outcome, budget, batch):
    """R(arm, s): the hidden-unit count at which an attempt first SOLVED AND the following future read solved;
    anything else (unsolved within budget, a no-growth terminal state, a validation solve whose future read
    fails) scores the failure penalty. outcome = {"solved": bool, "future_solved": bool, "size": int | None}."""
    if _solved(outcome):
        size = outcome["size"]
        if size is None or size < 0:
            raise ValueError(f"a solved outcome needs a size >= 0, got {size!r}")
        return size
    return failure_penalty(budget, batch)


def superiority(name, diffs):
    """Ruling 9: a degenerate (zero-variance) paired sample reads NOT MET, stricter than `contrast`."""
    c = pc.contrast(name, diffs)
    return c._replace(met=False) if c.degenerate else c


def non_inferiority(name, diffs):
    """Ruling 9a: a non-inferiority bar reads a degenerate sample by its common value (> 0 MET)."""
    return pc.contrast(name, diffs)


def primary(outcomes, budget, batch):
    """G-R3 clause 1 over paired seeds. `outcomes[arm]` is the per-seed list for TARGETED, PRESIZED_SELECTED and
    RANDOM. PASS needs t > 1.895 on BOTH D_sel and D_rand AND mean D >= one batch (L4: the grid's resolution).
    'fewer neurons' is licensed only when both arms of a contrast solve on >= 6 seeds and the paired size
    difference on the seeds where both solve is itself >= the floor; otherwise the verdict is a solve-rate
    difference, never a neuron saving."""
    arms = ("TARGETED", "PRESIZED_SELECTED", "RANDOM")
    scores = {a: [restricted_score(o, budget, batch) for o in outcomes[a]] for a in arms}
    n = len(scores["TARGETED"])
    if any(len(outcomes[a]) != n for a in arms):
        raise ValueError("every arm needs the same paired seeds")
    out = {"penalty": failure_penalty(budget, batch), "floor": batch}
    passed = fewer = True
    for name, ctl in (("D_sel", "PRESIZED_SELECTED"), ("D_rand", "RANDOM")):
        diffs = [c - t for c, t in zip(scores[ctl], scores["TARGETED"], strict=True)]
        c = superiority(name, diffs)
        mean = sum(diffs) / n
        met = c.met and mean >= batch
        both = [
            co["size"] - to["size"]
            for co, to in zip(outcomes[ctl], outcomes["TARGETED"], strict=True)
            if _solved(co) and _solved(to)
        ]
        licensed = (
            sum(map(_solved, outcomes[ctl])) >= MIN_SOLVING_SEEDS
            and sum(map(_solved, outcomes["TARGETED"])) >= MIN_SOLVING_SEEDS
            and bool(both)
            and sum(both) / len(both) >= batch
        )
        out[name] = {"contrast": c, "mean": mean, "met": met, "fewer_neurons": licensed}
        passed, fewer = passed and met, fewer and licensed
    out["pass"], out["fewer_neurons"] = passed, passed and fewer
    out["solve_rates"] = {a: sum(map(_solved, outcomes[a])) for a in arms}
    return out


# ---- campaign isolation (2.12a) ------------------------------------------------------------------------------
def arm_home(root, family, seed, arm):
    """A fresh home for one (seed, arm) under the campaign's scratch root. NOTHING is created before every
    check has passed: the root must already exist and pass the live-home guard (equal / inside / containing
    the live lake is refused, a missing root is refused, parents are never made), the family must be a plain
    name, the seed inside G3's block and the arm registered. Only then is the single leaf made, which cannot
    be the live lake because its parent is not it, inside it or a link to it."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if not BLOCK[0] <= seed < BLOCK[1]:
        raise PermissionError(f"seed {seed!r} is outside G3's block 9000-9999")
    if not family or family in (".", "..") or any(c in family for c in _SEPARATORS):
        raise ValueError(f"family must be a plain name, got {family!r}")
    parent = Path(home_mod.assert_not_live_home(str(root)))  # strict: a missing root is refused
    path = parent / f"{family}-{seed}-{arm}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def snapshot_digest(path):
    """sha256 of one frozen ledger snapshot file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def pair_snapshots(digests):
    """Paired arms start from the SAME frozen ledger state: two different digests refuse to pair."""
    digests = list(digests)
    if not digests or len(set(digests)) != 1:
        raise PermissionError(
            f"arms of one seed carry different ledger snapshots: {sorted(set(digests))}"
        )
    return digests[0]


def child_home(argv, env=None):
    """The home a child process was launched with: `--home` AND NORD_HOME must both name it. A child that was
    not handed its home refuses to start rather than falling back to a default (2.12a); the live lake raises."""
    env = os.environ if env is None else env
    if "--home" not in argv or argv.index("--home") + 1 >= len(argv):
        raise PermissionError("child refused: no --home argument (no fallback to a default home)")
    arg = argv[argv.index("--home") + 1]
    if not env.get("NORD_HOME"):
        raise PermissionError("child refused: NORD_HOME is not set")
    if Path(arg).resolve() != Path(env["NORD_HOME"]).resolve():
        raise PermissionError(
            f"child refused: --home {arg!r} and NORD_HOME {env['NORD_HOME']!r} differ"
        )
    return Path(home_mod.assert_not_live_home(arg))


def launch(argv_prefix, home, **extra):
    """Run one arm in its own process with the home passed as environment AND argument; returns the process."""
    env = dict(os.environ, NORD_HOME=str(home))
    args = [*argv_prefix, "--home", str(home)]
    for k, v in extra.items():
        args += [f"--{k}", str(v)]
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)


def main():
    """Self-check: the penalty grid, the restricted score, ruling 9 vs 9a on a degenerate sample, the verdict
    on a planted pass, preflight's refusals and the child's refusal without a home."""
    import tempfile

    assert failure_penalty(52, 9) == 54 and failure_penalty(52, 1) == 53
    ok = {"solved": True, "future_solved": True, "size": 18}
    assert restricted_score(ok, 52, 9) == 18
    assert restricted_score({**ok, "future_solved": False}, 52, 9) == 54
    assert not superiority("d", [9.0] * 8).met and non_inferiority("e", [9.0] * 8).met
    fail = {"solved": False, "future_solved": False, "size": None}
    sizes = (18, 27, 18, 27, 18, 27, 18, 27)

    def arm(extra):
        return [{**ok, "size": s + extra * (i % 3)} for i, s in enumerate(sizes)]

    got = primary({"TARGETED": arm(0), "PRESIZED_SELECTED": arm(18), "RANDOM": [fail] * 8}, 52, 9)
    assert got["pass"] and not got["fewer_neurons"], (
        got
    )  # RANDOM never solves: a solve-rate result only
    for bad in ([9100], [9100, 9101], [123]):
        try:
            preflight(bad)
            raise AssertionError(f"red: preflight must refuse {bad}")
        except PermissionError:
            pass
    preflight(list(DEV_SEEDS[:2]))
    try:
        child_home([sys.executable], {})
        raise AssertionError("red: a child without --home must refuse")
    except PermissionError:
        pass
    with tempfile.TemporaryDirectory() as d:
        home = Path(d) / "h"
        home.mkdir()
        assert child_home(["x", "--home", str(home)], {"NORD_HOME": str(home)}) == home.resolve()
    print(
        "growth_campaign: penalty 54/53, ruling 9 vs 9a, verdict, preflight and child refusals fire"
    )


if __name__ == "__main__":
    main()
