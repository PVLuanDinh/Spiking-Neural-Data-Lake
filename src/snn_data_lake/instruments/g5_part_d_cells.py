"""G5 / T113 Part D D1's cell runner (evidence/g5-readout-variants.md D.1, D.2, D.4, D.5).

Fills the record shape the committed reader `g5_part_d_reading` consumes:

    records[seed][cell][arm] -> held-out accuracy on the FUTURE split

for every arm of D.2, on the cells of D.4 (G2's (i) 2x2 plus (ii)/A_ii). The separate `cert_for` function
supplies the two facts that decide whether PD-R2 reads. An available fixture CLI entry is not evidence
that a fixture smoke ran. Everything goes through the public surfaces of D.6's port-free split, so
NOTHING here needs G3's port (asks 1, 2, 4, 5), a labeled grown population, or the return path (PD-R4 and
the feed neuter are D2's).

THE FEATURE PATH (D.1), one deterministic function of (recording seed, pool mode, cell):
  * the substrate is `KernelLsmPool` at G1(b)'s tuned g1 LSM (carried_state_campaign.TUNED) through
    `g5_readout_cells.pd_pool` + `pool_features` -- the SAME path the registered clean PD-P1 read used, so
    this runner adds no second encoder;
  * every turn of an episode is driven through the pool and the PROBE turn's vector is the stimulus the
    readout sees (G2's Reader reads `turns[-1].text`, and the S1 cells need the cue turns to have carried
    before the probe);
  * CARRY carries within an episode and resets at the episode boundary (the episode tuple is the boundary
    unit, record 10 row 7); RESET is the reset form.
    MEASURED on this build and pinned in tests/test_g5_part_d_cells.py (fixture recording seed 20261003):
    on the reset form the probe turn's vector is bit-identical to a fresh pool's (stateless per turn), on
    the carried form it is not, and `reset_carriers()` at the boundary restores a fresh pool's probe
    vector. Two independently built pools on the same (seed, mode, text sequence) give identical features
    at every step -- the property the same-rows check rests on.

THE PATH IS COMPUTED ONCE per (seed, cell, mode) and every arm consumes that precomputed path, because the
pool is stateful: walking an episode twice would give two different feature vectors and silently change
the registered stimulus. No arm can perturb another arm's path.

THE ARMS (D.2), each one rule or one control against the incumbent R-STDP of the posed-problem job:
  RSTDP_D              label neurons, `interact(stimulus, label)` -- the teacher's external d
  RSTDP_NOTUNE         the same learners at their seeded weights, no deposit at all
  RSTDP_BRAIN          the same learners tuned by BrainProducer's scalar, `interact(stimulus, m)`
  RSTDP_BRAIN_PRODNEUTER  note (f)'s ablation: r still reaches the nucleus, its output is pinned
  RSTDP_BYPASS         DIAGNOSTIC (PD-R1b's other side): `interact(stimulus, r)`, r = +-1
  RSTDP_BRAIN_SHUF     the tuning signal time-shuffled by the 2.3 circular shift of the recorded
                       teaching_sequence, with the vacuity floor REUSED from `g5_readout_cells`
  RSTDP_D@reset / @neuter_carry / @neuter_reset   the incumbent on the other pool modes
  EXPANSION_H          the label neurons fed G2's presized_expansion(problem, 256, seed).features(text)
                       (section 6's reused control, read beside its certified-H solve)
  EXPANSION_H_TEXTS    the texts= non-adapted twin, REPORTED beside EXPANSION_H
  DELTA / PC / SOM     the read-out-only adaptors of `g5_part_d_adaptors`; their controls are DELTA_CTRL
                       (the shuffled twin, built here exactly like RSTDP_BRAIN_SHUF), PC_CTRL (the
                       neutered coupling) and SOM0 (the neighbourhood width at zero)

TWO NAMES FOR ONE ARM, disclosed: the committed reader's ALL_ARMS names SOM's control SOM_CTRL while D.2 /
D.4 register it as SOM0. The runner writes the same value under both keys so the committed read is
satisfied either way; the naming divergence is an item for the lead and is not resolved here.

WHAT THIS RUNNER DOES NOT DO
  * it never runs a registered seed block: 11410-11417 (D1 confirmatory), 11418-11425 (reserve),
    11430-11455 (D2) and 11464-11471 (planted) are all held, and 11500-11507 was spent by the clean
    PD-P1. `main` refuses a non-fixture block without --lead-go; the default is fixture RNG seeds
    (SEEDS.md's 20261003 row: RNG seeds of fixtures, not recording seeds).
  * it never touches a Brain, the live lake, the daemon, the cockpit or the arm. Brain-free, and
    `home.require_isolated_home()` is called before any lake-binding import (record 2.6 PA9).
  * it tunes nothing: PD_EPOCHS and the adaptor hyper-parameters are build-time facts, to be frozen by the
    D1 dev-freeze amendment, and no number produced here is read against a registered set.
"""

from snn_data_lake.instruments.g5_readout_cells import PD_POOL_MODES

PD_H = 256  # section 6: the expansion control's H is set to the reservoir's unit count
PD_EPOCHS = 30  # BUILD DEFAULT, not a frozen campaign value -- the D1 dev-freeze amendment sets it

#: cell name -> G2's Problem (taught_problems.PROBLEMS; the (i) cells carry delay_turns = 1)
PD_CELLS = {
    "S0C0": "i.S0C0",
    "S0C1": "i.S0C1",
    "S1C0": "i.S1C0",
    "S1C1": "i.S1C1",
    "AII": "aii",
}

#: arm -> (feature source, family). 'expansion' is G2's frozen front-end; the rest are pd_pool modes.
PD_MODES = {
    "RSTDP_D": "carry",
    "RSTDP_NOTUNE": "carry",
    "RSTDP_BRAIN": "carry",
    "RSTDP_BRAIN_PRODNEUTER": "carry",
    "RSTDP_BYPASS": "carry",
    "RSTDP_BRAIN_SHUF": "carry",
    "RSTDP_D@reset": "reset",
    "RSTDP_D@neuter_carry": "neuter_carry",
    "RSTDP_D@neuter_reset": "neuter_reset",
    "EXPANSION_H": "expansion",
    "EXPANSION_H_TEXTS": "expansion_texts",
    "DELTA": "carry",
    "DELTA_CTRL": "carry",
    "PC": "carry",
    "PC_CTRL": "carry",
    "SOM": "carry",
    "SOM0": "carry",
    "SOM_CTRL": "carry",
}

#: the label-neuron arms (SpikingInteractionLearner); everything else in PD_MODES is an adaptor arm
LABEL_ARMS = (
    "RSTDP_D",
    "RSTDP_NOTUNE",
    "RSTDP_BRAIN",
    "RSTDP_BRAIN_PRODNEUTER",
    "RSTDP_BYPASS",
    "RSTDP_BRAIN_SHUF",
    "RSTDP_D@reset",
    "RSTDP_D@neuter_carry",
    "RSTDP_D@neuter_reset",
    "EXPANSION_H",
    "EXPANSION_H_TEXTS",
)

#: adaptor arm -> (kind, neutered coupling, shuffled-teaching control)
ADAPTOR_ARMS = {
    "DELTA": ("delta", False, False),
    "DELTA_CTRL": ("delta", False, True),
    "PC": ("pc", False, False),
    "PC_CTRL": ("pc", True, False),
    "SOM": ("som", False, False),
    "SOM0": ("som", False, False),  # the width at zero: g5_part_d_adaptors.SOM0_WIDTH
    "SOM_CTRL": ("som", False, False),
}
SOM_CTRL_KEYS = ("SOM0", "SOM_CTRL")  # two names, one value (see the module docstring)
#: the arms whose reads decide whether a PROVEN-STATELESS reader solved a plain S1 cell (note (e))
STATELESS_ARMS = ("RSTDP_D@neuter_reset", "EXPANSION_H")


class CellHalt(RuntimeError):
    """The instrument, not the hypothesis, is broken: no record is produced for this cell."""


def _arms_of(cell):
    """Every arm of D.2 this cell reads. The adaptor rules and their controls are read on S0C1 only
    (D.4 PD-R3); EXPANSION_H_TEXTS is reported beside EXPANSION_H."""
    base = [
        "RSTDP_D",
        "RSTDP_NOTUNE",
        "RSTDP_BRAIN",
        "RSTDP_BRAIN_PRODNEUTER",
        "RSTDP_BYPASS",
        "RSTDP_BRAIN_SHUF",
        "RSTDP_D@reset",
        "RSTDP_D@neuter_carry",
        "RSTDP_D@neuter_reset",
        "EXPANSION_H",
        "EXPANSION_H_TEXTS",
    ]
    if cell == "S0C1":
        base += ["DELTA", "DELTA_CTRL", "PC", "PC_CTRL", "SOM", "SOM0", "SOM_CTRL"]
    return tuple(base)


# --------------------------------------------------------------------------------------------------
# The deterministic feature path, walked ONCE per (seed, cell, mode)
# --------------------------------------------------------------------------------------------------
class Turn:
    """One driven turn: the stimulus vector, its text, and whether it is the episode's probe."""

    __slots__ = ("text", "x", "probe", "label")

    def __init__(self, text, x, probe, label):
        self.text, self.x, self.probe, self.label = text, x, probe, label


class _Drive:
    """The stateful pool walk for one (recording seed, mode). `stateless` is G2's frozen front-end, whose
    features never depend on the history -- the S1 controls' other side."""

    def __init__(self, seed, mode, problem):
        self.mode = mode
        if mode not in (*PD_POOL_MODES, "expansion", "expansion_texts"):
            raise CellHalt(f"unknown feature source {mode!r}")
        self._front = None
        self._pool = None
        if mode in ("expansion", "expansion_texts"):
            from snn_data_lake.instruments.taught_campaign import presized_expansion

            texts = None
            if mode == "expansion_texts":
                texts = [ep[-1].text for epoch in problem.teach_epochs(seed, 1) for ep in epoch]
            self._front = presized_expansion(problem, PD_H, seed, texts=texts).features
        else:
            from snn_data_lake.instruments.g5_readout_cells import pd_pool

            self._pool = pd_pool(seed, mode)

    def walk(self, episode):
        """Every turn of one episode in order; the pool resets at the boundary afterwards (D.1)."""
        last_i = len(episode) - 1
        if self._pool is None:
            turns = [
                Turn(t.text, [float(v) for v in self._front(t.text)], i == last_i, None)
                for i, t in enumerate(episode)
            ]
        else:
            from snn_data_lake.instruments.g5_readout_cells import pool_features

            turns = [
                Turn(
                    t.text,
                    [float(v) for v in pool_features(self._pool, t.text)],
                    i == last_i,
                    None,
                )
                for i, t in enumerate(episode)
            ]
            self._pool.reset_carriers()
        if not turns or not turns[-1].probe:
            raise CellHalt("an episode's last turn must be its probe")
        turns[-1].label = episode[
            -1
        ].label  # the label rides on the probe only (record: taught_problems)
        return turns


def _path(seed, mode, problem, teach, held):
    """The precomputed path: one walked list of Turn lists per stream, so no arm re-walks an episode."""
    drive = _Drive(seed, mode, problem)
    return [drive.walk(ep) for ep in teach], [drive.walk(ep) for ep in held]


def _teach_stream(problem, seed, epochs):
    """The registered teaching stream (teaching_sequence's shape): E epochs, each a fresh sample order."""
    return [ep for epoch in problem.teach_epochs(seed, epochs) for ep in epoch]


def _n_in(paths):
    n = len(paths[0][-1].x)
    if n != PD_H:
        raise CellHalt(f"the feature path is {n}-wide, D.1 registers {PD_H}")
    return n


def _same_rows(paths_by_mode, teach):
    """The 2.3 same-rows check, carried into Part D: every mode's path must carry the same probe text and
    label sequence. A mode that saw different rows would make the arm comparison a comparison of streams."""
    want = [(ep[-1].text, ep[-1].label) for ep in teach]
    for mode, (t_path, _h) in paths_by_mode.items():
        got = [(turn.text, turn.label) for turn in (t[-1] for t in t_path)]
        if got != want:
            raise CellHalt(f"mode {mode} saw different rows than the cell's teach stream")
    return len(want)


# --------------------------------------------------------------------------------------------------
# One arm over a precomputed path
# --------------------------------------------------------------------------------------------------
def _learner(labels, n_in, seed):
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    return SpikingInteractionLearner(list(labels), n_in, seed=seed)


def _run_label_arm(seed, arm, paths, labels, shifted):
    """One label-neuron arm: the incumbent's label channel, the producer's scalar channel, or none."""
    from snn_data_lake.instruments.g5_readout_cells import PD_OUTCOME, BrainProducer

    t_path, h_path = paths
    learner = _learner(labels, len(t_path[0][-1].x), seed)
    brain = None
    if arm in ("RSTDP_BRAIN", "RSTDP_BRAIN_SHUF"):
        brain = BrainProducer(seed, neutered=False)
    elif arm == "RSTDP_BRAIN_PRODNEUTER":
        brain = BrainProducer(seed, neutered=True)
    reads = []
    for epoch, turns in enumerate(t_path):
        probe = turns[-1]
        label = shifted[epoch] if shifted is not None else probe.label
        if arm == "RSTDP_NOTUNE":
            continue
        if brain is not None:
            r = PD_OUTCOME[0] if learner.respond(probe.x) == label else PD_OUTCOME[1]
            learner.interact(probe.x, float(brain.factor(probe.text, r)))
        elif arm == "RSTDP_BYPASS":
            r = PD_OUTCOME[0] if learner.respond(probe.x) == label else PD_OUTCOME[1]
            learner.interact(probe.x, float(r))
        else:
            learner.interact(probe.x, label)
    for turns in h_path:
        probe = turns[-1]
        reads.append((learner.respond(probe.x), probe.label))
    return reads


def _run_adaptor_arm(seed, arm, paths, labels, shifted):
    """One read-out-only adaptor arm: the same CARRY features and the same labels, one adopted rule."""
    from snn_data_lake.instruments.g5_part_d_adaptors import delta, pc, som

    kind, neutered, is_ctrl = ADAPTOR_ARMS[arm]
    t_path, h_path = paths
    n_in = len(t_path[0][-1].x)
    if kind == "delta":
        unit = delta(labels, n_in)
    elif kind == "pc":
        unit = pc(labels, n_in, seed=seed, neutered=neutered)
    else:
        unit = som(labels, n_in, seed=seed, control=arm in SOM_CTRL_KEYS)
    reads = []
    for epoch, turns in enumerate(t_path):
        probe = turns[-1]
        label = shifted[epoch] if (is_ctrl and shifted is not None) else probe.label
        unit.learn(probe.x, label)
    for turns in h_path:
        probe = turns[-1]
        reads.append((unit.respond(probe.x), probe.label))
    return reads


# --------------------------------------------------------------------------------------------------
# The record for one (seed, cell)
# --------------------------------------------------------------------------------------------------
def pd_cell(seed, cell, epochs=PD_EPOCHS, log=print):
    """One (seed, cell) record: arm -> held-out accuracy over the FUTURE split, plus `cert` and the
    shift's vacuity print. The pool is walked once per mode and shared by every arm (see _same_rows)."""
    import hashlib

    from snn_data_lake.instruments import taught_problems as tp
    from snn_data_lake.instruments.g5_readout_cells import (
        VACUITY_FLOOR,
        circular_shift,
        shift_fraction,
    )

    problem = tp.PROBLEMS[PD_CELLS[cell]]
    teach = _teach_stream(problem, seed, epochs)
    held = list(problem.episodes(seed, "future"))
    if not held:
        raise CellHalt(f"{cell} seed {seed}: the future split is empty")
    modes = sorted({PD_MODES[a] for a in _arms_of(cell)})
    paths = {m: _path(seed, m, problem, teach, held) for m in modes}
    _same_rows(paths, teach)
    labels = sorted({ep[-1].label for ep in teach})
    shifted = circular_shift([ep[-1].label for ep in teach])
    f = shift_fraction([ep[-1].label for ep in teach], shifted)
    log(
        f"  {cell} seed {seed}: {len(teach)} teach episodes, {len(held)} held-out, labels {labels}, "
        f"shift f {f:.3f}{'  VACUOUS' if f < VACUITY_FLOOR else ''}"
    )
    out, reads = {}, {}
    for arm in _arms_of(cell):
        p = paths[PD_MODES[arm]]
        sh = shifted if arm in ("RSTDP_BRAIN_SHUF", "DELTA_CTRL") else None
        r = (
            _run_label_arm(seed, arm, p, labels, sh)
            if arm in LABEL_ARMS
            else _run_adaptor_arm(seed, arm, p, labels, sh)
        )
        reads[arm] = r
        out[arm] = sum(a == b for a, b in r) / len(r)
    # two names, one control: the committed reader reads SOM_CTRL, D.2/D.4 register SOM0
    if "SOM0" in out and out["SOM0"] != out["SOM_CTRL"]:
        raise CellHalt("SOM0 and SOM_CTRL are one control and must carry one value")
    # a second, independent path must reproduce the first bit for bit: the runner is deterministic
    replay = _path(seed, "carry", problem, teach, held)
    for a, b in zip(paths["carry"][1], replay[1], strict=True):
        if a[-1].x != b[-1].x:
            raise CellHalt(f"{cell} seed {seed}: the carry path is not reproducible")
    out["_same_rows"] = hashlib.sha256(
        ",".join(f"{ep[-1].text}|{ep[-1].label}" for ep in teach).encode()
    ).hexdigest()[:16]
    out["_shift_f"] = f
    out["_reads"] = {a: r for a, r in reads.items() if a in STATELESS_ARMS}
    return out


def pd_records(seeds, cells=None, epochs=PD_EPOCHS, log=print, stamp=False):
    """records[seed][cell][arm]. One arm per subprocess is the caller's split; this is the serial
    reference available to the explicit fixture entry; defining the entry does not record a run.

    D.1: `home.require_isolated_home()` runs before any lake-binding import (record 2.6 PA9, the shared
    guard R1 built -- one mechanism, P2, never a second one here). `stamp=True` adds the live-lake
    `lake_snapshot` before/after the stage; it costs about 13 s against the live home (the 3.3 GB lake
    takes ~6.4 s per stamp, measured by G6 at 80d9387c), so it is the campaign's choice, not the fixture
    smoke's."""
    cells = tuple(PD_CELLS) if cells is None else tuple(cells)
    from snn_data_lake.substrate import home

    home.require_isolated_home()
    before = home.lake_snapshot(home.home()) if stamp else None
    rec = {}
    for s in seeds:
        rec[s] = {cell: pd_cell(s, cell, epochs=epochs, log=log) for cell in cells}
    if stamp and not home.live_lake_untouched(home.home(), before):
        raise CellHalt("the live lake moved during the stage")
    return rec


# --------------------------------------------------------------------------------------------------
# `cert`: the two facts that decide whether PD-R2 reads
# --------------------------------------------------------------------------------------------------
def expansion_solves(problem, seed, h, epochs=PD_EPOCHS):
    """Does G2's expansion solve this cell at h, read on the FUTURE split through G2's own reader and
    G2's own solved() predicate? Part D cites this only for the certified H (note (c))."""
    from snn_data_lake.instruments import taught_problems as tp
    from snn_data_lake.instruments.taught_campaign import presized_expansion

    reader = presized_expansion(problem, h, seed)
    for ep in _teach_stream(problem, seed, epochs):
        reader.learn_turn(ep[-1].text, None, feedback=ep[-1].label)
    held = list(problem.episodes(seed, "future"))
    reads = [(reader.route_text(ep[-1].text), ep[-1].label) for ep in held]
    return tp.solved(problem, seed, "future", reads)


def cert_for(cell, seed, epochs=PD_EPOCHS, reads=None):
    """`cert` for read_d1: the expansion's certified-H solve (S0C1 only), and whether a PROVEN-STATELESS
    reader solved a plain S1 cell. Note (e): neuter_reset is stateless per turn at T_in = 1 (it equals the
    reset form), so the HALT clause applies to that arm and to EXPANSION_H, and to nothing else."""
    from snn_data_lake.instruments import taught_problems as tp

    problem = tp.PROBLEMS[PD_CELLS[cell]]
    stateless_solved_s1 = False
    expansion_ok = None
    if cell == "S0C1":
        expansion_ok = expansion_solves(problem, seed, 64, epochs=epochs)
    if cell in ("S1C0", "S1C1") and reads is not None:
        for arm in STATELESS_ARMS:
            if tp.solved(problem, seed, "future", reads.get(arm, [])):
                stateless_solved_s1 = True
    return {"expansion_solves_s0c1_at_64": expansion_ok, "stateless_solved_s1": stateless_solved_s1}


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the runner's constructed reds (brain-free, fixture RNG seeds only, cheap)
# --------------------------------------------------------------------------------------------------
FIXTURE_SEEDS = range(20261003, 20261011)  # SEEDS.md: RNG seeds of fixtures (not recording seeds)


def self_check():
    """The runner's reds at gate cost: a wrong-width path halts, the same-rows check halts a perturbed
    stream, a cell is deterministic, PRODNEUTER IS NOTUNE through the real cell path (note (f)), and a
    non-fixture seed block is refused without the lead's go. The fuller pin (the S1 halt clause, SOM0
    vs SOM, the incumbent's teaching, the shuffle's reach, the reader integration) is
    tests/test_g5_part_d_cells.py. Returns True; raises on the first failure."""
    from snn_data_lake.instruments import taught_problems as tp

    try:
        _n_in([[Turn("t", [0.0] * 8, True, "GREET")]])
    except CellHalt:
        pass
    else:
        raise AssertionError("a wrong-width feature path did not halt")
    problem = tp.PROBLEMS["i.S0C1"]
    teach = _teach_stream(problem, FIXTURE_SEEDS[0], 1)
    paths = {"carry": _path(FIXTURE_SEEDS[0], "carry", problem, teach, [])}
    if _same_rows(paths, teach) != len(teach):
        raise AssertionError("the same-rows check counted the wrong stream")
    try:
        _same_rows(paths, teach[:-1])
    except CellHalt:
        pass
    else:
        raise AssertionError("a perturbed teach stream did not halt the same-rows check")
    rec = pd_cell(FIXTURE_SEEDS[0], "S0C1", epochs=2, log=lambda *_: None)
    body = {k: v for k, v in rec.items() if not k.startswith("_")}
    if body != {
        k: v
        for k, v in pd_cell(FIXTURE_SEEDS[0], "S0C1", epochs=2, log=lambda *_: None).items()
        if not k.startswith("_")
    }:
        raise AssertionError("a cell is not deterministic across two walks")
    if set(body) != set(_arms_of("S0C1")):
        raise AssertionError("the cell does not carry every registered arm")
    sanity = pd_cell(FIXTURE_SEEDS[0], "S0C0", epochs=4, log=lambda *_: None)
    if sanity["RSTDP_BRAIN_PRODNEUTER"] != sanity["RSTDP_NOTUNE"]:
        raise AssertionError("PRODNEUTER != NOTUNE: a deposit with d = 0 moved a weight (note f)")
    if main(["--seeds", "11410", "11417", "--cells", "S0C0", "--epochs", "1"]) != 2:
        raise AssertionError("a non-fixture seed block was not refused")
    print(
        "g5_part_d_cells self-check OK: a wrong-width path and a perturbed stream halt; an S0C1 cell "
        "is deterministic across two walks and carries every registered arm; PRODNEUTER == NOTUNE "
        "through the real cell path; a non-fixture seed block is refused without the lead's go"
    )
    return True


# --------------------------------------------------------------------------------------------------
# CLI: fixtures only. A registered seed block needs the lead's written go.
# --------------------------------------------------------------------------------------------------
def main(argv=None):
    import sys

    from snn_data_lake.instruments.g5_part_d_adaptors import code_hash

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        self_check()  # carried_state_driver's pattern: no arguments runs the self-check
        return 0
    seeds = FIXTURE_SEEDS
    if "--seeds" in argv:
        i = argv.index("--seeds")
        seeds = range(int(argv[i + 1]), int(argv[i + 2]) + 1)
        if not all(FIXTURE_SEEDS.start <= s < FIXTURE_SEEDS.stop for s in seeds) and (
            "--lead-go" not in argv
        ):
            print("REFUSED: a non-fixture seed block needs the lead's written go")
            return 2
    cells = tuple(argv[argv.index("--cells") + 1].split(",")) if "--cells" in argv else ("S0C1",)
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else PD_EPOCHS
    lo, hi = seeds[0], seeds[-1]
    print(f"g5_part_d_cells: fixtures {lo}..{hi}, cells {cells}, epochs {epochs}")
    print(f"adaptor code hash {code_hash()}")
    rec = pd_records(seeds, cells, epochs=epochs)
    for cell in cells:
        for arm in _arms_of(cell):
            vals = [rec[s][cell][arm] for s in seeds]
            print(f"  {cell} {arm:<22} " + " ".join(f"{v:.3f}" for v in vals))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
