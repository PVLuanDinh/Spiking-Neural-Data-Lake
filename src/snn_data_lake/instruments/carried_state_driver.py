r"""G1(c) / T109 part (c): the campaign driver -- section 2.13's stages over one OUT directory (parts 1 and 2).

The frozen text is specs/022-unified-brain-arc/evidence/g1c-carried-state.md; sections 2.4, 2.6, 2.11, 2.12
and 2.13 bind this module. `carried_state_campaign` holds the frozen constants, the pure decisions and the
transforms; this module is the DRIVER. It imports `pool_selection`'s step, job and table machinery (section
2.14, POLICIES P2: one mechanism) and adds the stages. Every stage runs through
`pool_selection.run_step(stage, out, steps=STAGES, function=...)`, so the single-head, dirty-tree, order and
no-retry refusals and the exit codes are pool_selection's own: 0 COMPLETED, 1 ABORTED (a job FAILed, or the
head moved), 2 HALTED (the shared instrument failed; the summary names the check), 3 refused to start (no
summary written). A stage writes OUT/<stage>.summary.json with `head` and `dirty`.

  record      seed 100 ALONE and timed, then R-b8's digest against the G1(a) pilot's 100-stream (a differing
              digest HALTS, 2.6); the cost projection is printed; then ONE batch for 203, the bundle's gate
              seeds and its live set. ONE complete seed bundle per campaign (carried_state_campaign.
              SEED_BUNDLES): live 300-330 + gate 204-207 + synthetic 5005-5008 + R-b6 6000, or, with --seeds
              reserve (2.11), reserve 331-361 + gate 208-211 + synthetic 5009-5012 + R-b6 6001; migrate,
              instrument and controls read it from the record summary. --seeds reserve needs OUT/REPAIR.txt
              naming an ORIGINAL live campaign (never a reserve one) and CLAIMS that campaign's single
              reserve attempt exclusively before recording; a second claim halts. Any recording
              that is not OK halts. 2.6's half-moved rule on the K = 4 drawn permutations is asserted before the
              first recording (over the pre-registered block sizes) and again on each live recording's own block
              sizes before any replay.
  mechanism   pool_selection's step_mechanism, then one "configuration" job for TUNED; any failure halts (the
              pool is the object under test: there is no runner-up).
  instrument  pool_selection's step_instrument on the bundle's R-b6 seed (6000; 6001 reserve) with R-b6 as a READING (a blind R-b6 is
              G-R1's REFUSED, 2.4 (i), recorded, not a halt; R-b2, R-b7's reds and R-b8's two halves still halt),
              then the shuffle's POOL-FREE positive control and its A/A twin on the same stream (2.6).
  references  (a), (f64), (e1), (e1-0) on the live seeds through pool_selection's _phase_table with no pool
              cell: 2.4 (ii)'s oracle through the decoder (a not-met oracle is REFUSED, recorded, not a halt)
              and 2.6's stream-history row. Writes `inspected: True` (and OUT/INSPECTED.txt before the first
              replay): from here a repair runs on the reserve seeds (2.11).
  construct   G-R2's construction check (inseparable_mapping.construction_check) as a "call" job; the stage
              COMPLETES whatever the verdict (2.8: a red makes item (3) NOT RUN; G1(c) continues).

Part 2 (P5c) adds the other seven stages, in three modules this one imports (they never import it):
`carried_state_stages` (migrate, causality, primary and the helpers every pool stage shares),
`carried_state_controls` (controls, transport, separation) and `carried_state_report` (the report stage and
the golden check of 2.11). `golden [OUT] [--against FILE]` is the CLI's thirteenth command, beside the
twelve stages: it runs the golden check in-process and prints its digest, or with --against compares it with
a saved one (exit 0 equal, 1 moved).

Summary keys part 2 reads: record `live_seeds`, `seed_set` (and `bundle`), `recordings`, `identity`, `shuffle`; instrument
`rb6_ok`, `rb6_clauses`, `shuffle_control`, `shuffle_aa`; references `inspected`, `oracle`, `oracle_verdict`,
`stream_history`, `accuracy`, `references`; construct `run`, `verdict`, `item_3`, `result`.

The CLI (no argument runs the self-check; the preflight is the CLI's alone):

  python -m snn_data_lake.instruments.carried_state_driver <stage> OUT [--pilot DIR] [--seeds live|reserve]
  python -m snn_data_lake.instruments.carried_state_driver golden [OUT] [--against FILE]

launched from the repository root with OUT outside it. The preflight refuses (exit 3, nothing written) a
working directory that is not the repository root, an OUT inside the repository, any of SNN_EPROP_CKPT,
SNN_PERCEPTION_PACK, SNN_EPROP_AUDIO_CKPT, SNN_SHD_PACK, NORD_DEVICE or NORD_GPU set, and an asset that is
missing or fails its hash. OUT/REPAIR.txt for --seeds reserve: its first line is the path of the first OUT (a
directory holding record.summary.json of an ORIGINAL live campaign that has not given its one reserve
attempt: OUT/RESERVE_CLAIM.txt there), its further lines the repair.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.carried_state_driver

Deterministic; ASCII-only prints.
"""

import argparse
import contextlib
import functools
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from snn_data_lake.instruments import carried_state_campaign as cs
from snn_data_lake.instruments import carried_state_controls as ctl
from snn_data_lake.instruments import carried_state_report as rep
from snn_data_lake.instruments import carried_state_stages as st
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments import pool_selection as ps

# Sites the duplicate-capability hook flagged on this module's docstring (read; none drives a campaign's stages):
# distinct-from: instruments/memory_diagnostic.py::check_r_b0x_readout -- hand fixtures of the reference readout, which the mechanism stage RUNS through pool_selection.step_mechanism, never reimplemented here
# distinct-from: readout/snn_moe_classifier.py::_new_rng -- seeds a numpy Generator from the global random stream; this module draws nothing
# distinct-from: engine/brain.py::Brain.recall_episode_durable -- a brain's two-store episode recall; this driver builds no Brain
# distinct-from: instruments/pool_selection.py::_child -- the job child, REUSED through ps.run_jobs / ps.run_step, not reimplemented
# distinct-from: plasticity/competitive_learning.py::CompetitiveLearner.step -- advances a competitive learner by one timestep; this driver has no timestep
STAGES = (
    "record",
    "mechanism",
    "instrument",
    "references",
    "construct",
    "migrate",
    "causality",
    "primary",
    "controls",
    "transport",
    "separation",
    "report",
)
POOL = st.POOL  # the one candidate (ps.SELECTABLE): the pool G1(b) selected
REFERENCE_ARMS = ("a", "f64", "e1", "e1-0")  # 2.13's references stage
# 2.6's blocks of a live recording, in segment order: warm-up 256, repeated-canonical 463 = 256 + 207,
# unseen-order 256, then repeated-permuted, one partial block of 207-214 events (209 on the rehearsal's
# seed 100). The last size varies with the recording, so the pre-recording assertion covers its whole range.
PREREGISTERED_BLOCKS = (256, 256, 207, 256)
PERMUTED_BLOCKS = tuple(range(207, 215))
ENV_REFUSED = (
    *md.ASSET_ENV,
    "NORD_DEVICE",
    "NORD_GPU",
)  # an override the recorder, or the policy, reads
REPAIR_FILE = "REPAIR.txt"
RESERVE_CLAIM_FILE = (
    "RESERVE_CLAIM.txt"  # in the ORIGINAL OUT: its one reserve attempt (2.11), made with mode "x"
)
INSPECTED_FILE = "INSPECTED.txt"
CONSTRUCTION_TARGET = "snn_data_lake.instruments.inseparable_mapping:construction_check"


# ---- the preflight and the start refusals (the CLI's alone; never in the self-check) -------------


def repo_root():
    """The repository this module lives in (the root memory_diagnostic's `_revision` and `_assets` anchor on)."""
    return Path(md.__file__).resolve().parents[3]


def _same_path(first, second):
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _inside(path, root):
    resolved, root = Path(path).resolve(), Path(root).resolve()
    return resolved == root or root in resolved.parents


def preflight(out, *, cwd=None, root=None, env=None, assets=None):
    """What forbids starting a campaign, as a list of reasons (empty: clear). The working directory must be
    the repository root (brain.py reads models/ relative to it); OUT must lie outside it (a file written
    inside would dirty the head); no asset or device override may be set; the four assets must be present
    and hash. `cwd`, `root`, `env` and `assets` stand in for the real ones in a test."""
    root = repo_root() if root is None else Path(root)
    cwd = Path.cwd() if cwd is None else Path(cwd)
    env = os.environ if env is None else env
    assets = md._assets if assets is None else assets
    problems = []
    if not _same_path(cwd, root):
        problems.append(f"the working directory {cwd} is not the repository root {root}")
    if _inside(out, root):
        problems.append(f"OUT {out} lies inside the repository {root}")
    overridden = [name for name in ENV_REFUSED if env.get(name) is not None]
    if overridden:
        problems.append(f"an asset or device override is set: {overridden}")
    try:
        assets()
    except ValueError as error:
        problems.append(str(error))
    return problems


def _pilot_problems(pilot, arms):
    if pilot is None:
        return ["--pilot DIR is required"]
    root, missing = Path(pilot), []
    if not (root / f"{ps.PILOT_SEED}-stream.json.gz").exists():
        missing.append(f"{ps.PILOT_SEED}-stream.json.gz")
    for arm in arms:
        name = f"{ps.PILOT_SEED}-{arm}-causal.json"
        if not ((root / name).exists() or (root / f"{name}.gz").exists()):
            missing.append(name)
    return [f"the pilot directory {root} lacks {name}" for name in missing]


def _repair_problems(out):
    path = Path(out) / REPAIR_FILE
    if not path.is_file():
        return [
            f"--seeds reserve needs {path}: its first line names the first OUT, the rest the repair"
        ]
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        return [
            f"{path} must name the first OUT on its first line and the repair on the lines after"
        ]
    first = Path(lines[0])
    if not (first / "record.summary.json").is_file():
        return [f"{path} line 1 must name the first OUT (a directory holding record.summary.json)"]
    if _same_path(first, out):
        return [f"{path} names this very OUT as the first OUT"]
    # 2.11: a repair gets ONE reserve attempt, and only the ORIGINAL (live) campaign is repaired
    try:
        seed_set = md._load(first / "record.summary.json").get("seed_set")
    except (OSError, ValueError, AttributeError) as error:
        return [f"{path} line 1: the first OUT's record summary cannot be read ({error!r})"]
    if seed_set != "live":
        return [
            f"{path} line 1 names a campaign whose record summary reads seed_set {seed_set!r}: only "
            "the ORIGINAL live campaign is repaired, and a repair is never repaired again (a second "
            "repair halts G1(c) and goes to the owner, 2.11)"
        ]
    claim = first / RESERVE_CLAIM_FILE
    if claim.exists():
        return [
            f"{first} already gave its one reserve attempt ({claim}): a second repair of the same "
            "OUT halts G1(c) and goes to the owner (2.11)"
        ]
    return []


# 2.11: a repair's reserve attempt is CLAIMED exclusively (open mode "x" on RESERVE_CLAIM.txt in the
# original OUT) before anything is recorded; a claim that appeared since the start check, a second
# repair racing the first, halts. Returns the claim file's path.
def claim_reserve(out):
    first = Path(Path(out, REPAIR_FILE).read_text(encoding="utf-8").splitlines()[0].strip())
    claim = first / RESERVE_CLAIM_FILE
    try:
        with open(claim, "x", encoding="utf-8") as file:
            file.write(f"{Path(out)}\n")
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError as error:
        text = claim.read_text(encoding="utf-8").strip()
        raise ps.Halt(
            "record: the reserve attempt is already claimed", {"original": str(first), "by": text}
        ) from error
    return str(claim)


def start_problems(stage, out, pilot=None, seeds="live"):
    """The reasons a stage may not start, before OUT is touched: a missing pilot (record needs the
    100-stream, instrument also the eight causal replays), --seeds reserve without a REPAIR.txt that names
    the first OUT (2.11), an unknown seed set."""
    problems = []
    if stage == "record":
        problems += _pilot_problems(pilot, ())
        if seeds not in ("live", "reserve"):
            problems.append(f"--seeds {seeds!r} is neither live nor reserve")
        elif seeds == "reserve":
            problems += _repair_problems(out)
    elif stage == "instrument":
        problems += _pilot_problems(pilot, md.SCORED_ARMS)
    return problems


# ---- the seeds and 2.6's half-moved rule ---------------------------------------------------------


def live_seeds(which="live"):
    """The primary's 31 live seeds (or, after a repair, the reserve's), validated (2.12, 2.11)."""
    reserve = which == "reserve"
    seeds = cs.validate_seed_set(
        cs.RESERVE_SEEDS if reserve else cs.LIVE_SEEDS, allow_reserve=reserve
    )
    if len(seeds) != cs.N_LIVE:
        raise ValueError(f"{len(seeds)} live seeds, S = {cs.N_LIVE} required (g1c 2.12)")
    return seeds


def check_half_moved(seed, size_lists):
    """2.6: for one recording seed, every K = 4 drawn permutation of each block-size list moves at least
    half of every block of two or more events. Asserted, never re-drawn: a failure is a Halt."""
    for sizes in size_lists:
        if not sizes:
            raise ps.Halt(
                "shuffle half-moved", f"seed {seed}: a recording with no block to shuffle"
            )
        for k in range(cs.SHUFFLE_K):
            try:
                cs.block_permutations(seed, k, list(sizes))
            except AssertionError as error:
                raise ps.Halt("shuffle half-moved", str(error)) from error


def preregistered_sizes():
    """Every block-size list a live recording is pre-registered to have (the permuted block varies)."""
    return [(*PREREGISTERED_BLOCKS, permuted) for permuted in PERMUTED_BLOCKS]


def _recorded_blocks(out, live):
    """Each live recording's own block sizes, half-moved asserted on them before any replay."""
    sizes = {}
    for seed in live:
        sizes[str(seed)] = [len(group) for group in cs.blocks(md._load(ps._recording(out, seed)))]
        check_half_moved(seed, [sizes[str(seed)]])
    return sizes


# ---- the stages ----------------------------------------------------------------------------------


def stage_record(out, pilot, seeds="live"):
    """Seed 100 alone (timed) and R-b8's digest, then ONE batch for 203, 204-207 and the live seeds."""
    live, bundle = live_seeds(seeds), cs.seed_bundle(seeds)  # ONE complete bundle for the campaign
    for seed in live:  # before the first recording (2.6): the pre-registered sizes
        check_half_moved(seed, preregistered_sizes())
    claim = claim_reserve(out) if seeds == "reserve" else None  # 2.11: before ANYTHING is recorded
    started = time.monotonic()
    first = ps._record_seeds(out, (ps.PILOT_SEED,))
    seconds = time.monotonic() - started
    row = first["recordings"][str(ps.PILOT_SEED)]
    if row["status"] != "OK":
        raise ps.Halt("record", {"seed": ps.PILOT_SEED, "status": row["status"]})
    pinned = md._source_digest(md._load(Path(pilot) / f"{ps.PILOT_SEED}-stream.json.gz"))
    identity = {
        "seed": ps.PILOT_SEED,
        "seconds": seconds,
        "new": row["source_digest"],
        "pilot": pinned,
    }
    if row["source_digest"] != pinned:  # 2.6: the stream changed: a shared-instrument failure
        raise ps.Halt("R-b8 stream identity", identity)
    rest = (cs.CAUSALITY_SEED, *bundle.gate, *live)
    rounds = math.ceil(len(rest) / ps.WORKERS)
    projected = seconds * rounds
    print(
        f"record: seed {ps.PILOT_SEED} alone took {seconds:.1f} s; the other {len(rest)} recordings "
        f"project to {projected:.0f} s ({projected / 60:.1f} min) in {rounds} rounds of {ps.WORKERS}"
    )
    started = time.monotonic()
    batch = ps._record_seeds(out, rest)
    recordings = {**first["recordings"], **batch["recordings"]}
    late = {seed: row["status"] for seed, row in recordings.items() if row["status"] != "OK"}
    if late:  # never fewer than 31 live recordings: not one TIMEOUT is averaged over
        raise ps.Halt("record", {"not OK": late})
    blocks = _recorded_blocks(out, live)
    return {
        "recordings": recordings,
        "seed_set": seeds,
        "bundle": {  # the complete bundle: migrate, instrument and controls read their seeds from it
            "which": bundle.which,
            "live": list(bundle.live),
            "gate": list(bundle.gate),
            "gate_synthetic": list(bundle.gate_synthetic),
            "rb6": bundle.rb6,
        },
        "reserve_claim": claim,
        "live_seeds": list(live),
        "identity": {**identity, "equal": True},
        "projection": {"seconds": projected, "rounds": rounds, "workers": ps.WORKERS},
        "batch_seconds": time.monotonic() - started,
        "shuffle": {
            "k": cs.SHUFFLE_K,
            "block": cs.SHUFFLE_BLOCK,
            "tag": cs.SHUFFLE_TAG,
            "preregistered_blocks": [list(sizes) for sizes in preregistered_sizes()],
            "recorded_blocks": blocks,
        },
        "repair": (Path(out) / REPAIR_FILE).read_text(encoding="utf-8")
        if seeds == "reserve"
        else None,
    }


def stage_mechanism(out):
    """pool_selection's mechanism step, plus one "configuration" job for TUNED; a failure halts."""
    body = ps.step_mechanism(out)
    if body["refused"]:
        raise ps.Halt("mechanism", body["refused"])
    job = {
        "kind": "configuration",
        "candidate": POOL,
        "config": cs.TUNED,
        "out": str(ps._folder(out, "mechanism") / "tuned-configuration.json"),
    }
    (status,) = ps.run_jobs([job])
    if status["status"] != "OK":
        raise ps.Halt("mechanism: TUNED configuration", status)
    result = md._load(job["out"])
    failed = [name for name in ps.CONFIGURATION_GREENS if not result[name]["verdict"]]
    if failed:
        raise ps.Halt("mechanism: TUNED configuration", {"failed": failed})
    body["configuration"] = {
        "config": cs.TUNED,
        "verdicts": {name: result[name]["verdict"] for name in ps.CONFIGURATION_CHECKS},
        "R-b3 settings": result["R-b3"]["numbers"]["settings"],
    }
    return body


def shuffle_control(stream):
    """2.6's POOL-FREE positive control on one rb6 stream: the reference readout's (e1) - (e1-0) gap, as
    the mean over the K shuffles, unshuffled >= RB6_BAR and shuffled < RB6_BAR; `sighted` says both hold
    (a blind result is a reading, never a halt)."""
    rows = {arm: functools.partial(cs._arm_rows, arm=arm) for arm in ("e1", "e1-0")}
    gaps = {
        name: cs._decoder_gaps([stream], rows["e1"], rows["e1-0"], shuffles=count)[0]
        for name, count in (("unshuffled", 0), ("shuffled", cs.SHUFFLE_K))
    }
    return {
        **gaps,
        "shuffles": cs.SHUFFLE_K,
        "bar": md.RB6_BAR,
        "sighted": bool(gaps["unshuffled"] >= md.RB6_BAR and gaps["shuffled"] < md.RB6_BAR),
    }


# Site the duplicate-capability hook flagged on the name stage_instrument (read; it reads a pilot-pin file, no replay, no stage):
# distinct-from: instruments/prior_drift_harness.py::_pin_stages -- every pilot stage ever written, earliest first, from one pin file
def stage_instrument(out, pilot, rb6_seed=None):
    """R-b2, R-b7's reds, R-b6 on `rb6_seed` (the seed bundle's: 6000 live, 6001 reserve; only a test names another)
    as a reading, R-b8's halves; then the shuffle's pool-free positive control and its A/A twin."""
    rb6_seed = st.bundle_of(out).rb6 if rb6_seed is None else rb6_seed
    body = ps.step_instrument(out, pilot, rb6_seed=rb6_seed, rb6_halts=False)
    families = ps._rb6_families(ps._folder(out, "instrument"), rb6_seed)
    body["rb6_ok"] = all(body["rb6_clauses"].values())
    body["shuffle_control"] = shuffle_control(md._load(families["rb6"]))
    twin = shuffle_control(md._load(families["rb6red"]))  # A/A: no history to lose, shuffled or not
    body["shuffle_aa"] = {
        **twin,
        "held": bool(max(twin["unshuffled"], twin["shuffled"]) < md.RB6_BAR),
    }
    return body


def _accuracies(table, live, learner):
    return {
        seed: {
            arm: ps.recording_accuracy(
                table[(arm, None, None, None, seed, False)]["digest"][learner]
            )
            for arm in REFERENCE_ARMS
        }
        for seed in live
    }


def stage_references(out):
    """(a), (f64), (e1), (e1-0) on the live seeds: 2.4 (ii)'s oracle and 2.6's stream-history row."""
    record = ps._summary(out, "record")
    live, reserve = tuple(record["live_seeds"]), record["seed_set"] == "reserve"
    folder = ps._folder(out, "references")
    # 2.11: inspection starts at the first replay of a live seed; the marker is written before it
    (Path(out) / INSPECTED_FILE).write_text(
        f"inspection began at {folder} (g1c 2.11)\n", encoding="utf-8"
    )
    paths = {seed: ps._recording(out, seed) for seed in live}
    table = ps._phase_table(out, folder, {}, live, (), (), REFERENCE_ARMS, paths)
    late = sorted((key[0], key[4]) for key, cell in table.items() if cell["status"] != "OK")
    if late:  # 2.10: a missing or TIMEOUT cell halts, never read over what completed
        raise ps.Halt("references", {"not OK": late})
    accuracy = {learner: _accuracies(table, live, learner) for learner in ("readout", "grower")}
    try:
        oracle = cs.oracle_check(
            accuracy["readout"], accuracy["grower"], seeds=live, allow_reserve=reserve
        )
    except ValueError as error:
        raise ps.Halt("references: oracle", str(error)) from error
    history = {
        learner: cs.contrast_row(
            f"stream history (e1) - (e1-0) through the {learner}",
            [accuracy[learner][seed]["e1"] - accuracy[learner][seed]["e1-0"] for seed in live],
        )
        for learner in ("readout", "grower")
    }
    return {
        "inspected": True,
        "live_seeds": list(live),
        "accuracy": accuracy,
        "references": {arm: ps._reference(table, arm, live) for arm in REFERENCE_ARMS},
        "oracle": oracle,
        "oracle_verdict": "MET" if oracle["met"] else "REFUSED (blind harness)",
        "stream_history": history,
    }


def stage_construct(out):
    """G-R2's construction check as a "call" job; the stage COMPLETES whatever the verdict (2.8)."""
    job = {
        "kind": "call",
        "target": CONSTRUCTION_TARGET,
        "args": {},
        "out": str(ps._folder(out, "construct") / "construction-check.json"),
    }
    (status,) = ps.run_jobs([job])
    if status["status"] != "OK":  # a TIMEOUT: not a verdict (a FAIL aborted in run_jobs)
        return {"run": False, "verdict": None, "item_3": "NOT RUN", "reason": status["status"]}
    result = md._load(job["out"])
    verdict = bool(result["verdict"])
    return {
        "run": True,
        "verdict": verdict,
        "item_3": "RUN" if verdict else "NOT RUN",
        "reason": None if verdict else "a clause of the construction check failed (g1c 2.8)",
        "result": result,
    }


STAGE_FUNCTIONS = {
    "record": stage_record,
    "mechanism": stage_mechanism,
    "instrument": stage_instrument,
    "references": stage_references,
    "construct": stage_construct,
    **st.STAGE_FUNCTIONS,  # migrate, causality, primary
    **ctl.STAGE_FUNCTIONS,  # controls, transport, separation
    "report": rep.stage_report,
}


# Sites the duplicate-capability hook flagged on the name `run_stage` (read; none runs a campaign stage):
# distinct-from: instruments/pool_selection.py::run_step -- the step protocol this CALLS (steps=STAGES); run_stage adds the start refusals and nothing else
# distinct-from: gates/policy_manifest.py::_enabled_ci_text -- strips the comment lines of ci.yml
# distinct-from: instruments/prior_drift_harness.py::run_arm -- one arm and one seed of another harness's per-trial protocol
# distinct-from: instruments/report_order_probe.py::run_t081_dry_run -- one seed's apparatus dry run of another probe
# distinct-from: plasticity/homeostatic_plasticity_stack.py::_run_stream -- drives deposits onto one neuron to check a stabilizer
def run_stage(stage, out, *, revision=None, function=None, **kwargs):
    """Run one stage under pool_selection's step protocol with STAGES as its order; a stage that may
    not start raises pool_selection's Refused before OUT is touched. `function` stands in for the
    stage's own in a test."""
    if stage not in STAGES:
        raise ps.Refused(f"unknown stage {stage!r}")
    problems = start_problems(stage, out, **kwargs)
    if problems:
        raise ps.Refused(f"{stage}: " + "; ".join(problems))
    return ps.run_step(
        stage,
        out,
        revision=revision,
        function=function or STAGE_FUNCTIONS[stage],
        steps=STAGES,
        **kwargs,
    )


# ---- the self-check (cwd-free, no recording, no pool, no Brain) ----------------------------------


def _clean():
    return "h1", False


def _refused(stage, out, revision, **kwargs):
    try:
        run_stage(stage, out, revision=revision, function=lambda _out, **_kw: {}, **kwargs)
    except ps.Refused:
        return True
    return False


def self_check():
    assert STAGES == tuple(STAGE_FUNCTIONS) and len(set(STAGES)) == 12
    # the preflight: the control passes; each refusal is the only reason it names
    with tempfile.TemporaryDirectory() as scratch:
        root, elsewhere = Path(scratch) / "repo", Path(scratch) / "elsewhere"
        root.mkdir()
        elsewhere.mkdir()
        good = {"cwd": root, "root": root, "env": {}, "assets": lambda: {}}

        def broken():
            raise ValueError("REFUSED: asset hash eprop_nmnist.pt: 00")

        assert preflight(elsewhere / "out", **good) == []
        assert len(preflight(elsewhere / "out", **{**good, "cwd": elsewhere})) == 1
        assert len(preflight(root / "out", **good)) == 1 and len(preflight(root, **good)) == 1
        for name in ENV_REFUSED:
            (problem,) = preflight(elsewhere / "out", **{**good, "env": {name: "x"}})
            assert name in problem
        assert len(preflight(elsewhere / "out", **{**good, "assets": broken})) == 1
        # the stage chain on a scratch OUT at stand-in revisions: order, head, dirty tree, moved head
        out, pilot = elsewhere / "chain", elsewhere / "pilot"
        pilot.mkdir()
        (pilot / f"{ps.PILOT_SEED}-stream.json.gz").touch()
        moves = iter([("h1", False), ("h2", False)])
        with contextlib.redirect_stdout(io.StringIO()):
            assert _refused("mechanism", out, _clean)  # no record summary
            assert _refused("record", out, lambda: ("h1", True), pilot=str(pilot))  # dirty tree
            moved = run_stage(
                "record",
                out,
                revision=lambda: next(moves),
                function=lambda _o, **_k: {},
                pilot=str(pilot),
            )
            assert moved == ps.EXIT_ABORT and ps._summary(out, "record")["status"] == "ABORTED"
            assert _refused("mechanism", out, _clean)  # the predecessor ABORTED
            fresh = elsewhere / "fresh"
            ok = run_stage(
                "record", fresh, revision=_clean, function=lambda _o, **_k: {}, pilot=str(pilot)
            )
            assert ok == ps.EXIT_OK
            assert _refused(
                "mechanism", fresh, lambda: ("h2", False)
            )  # the predecessor at another head
            assert _refused("record", fresh, _clean, pilot=str(pilot))  # no retries
        # every later stage refuses to start before its predecessor and writes nothing; --seeds reserve
        # needs a REPAIR.txt naming the first OUT
        for stage in STAGES[5:]:
            assert _refused(stage, elsewhere / "later", _clean)
        assert not (elsewhere / "later" / "migrate.summary.json").exists()
        assert start_problems("record", fresh, str(pilot), "live") == []
        assert len(start_problems("record", fresh, str(pilot), "reserve")) == 1
        # 2.11: only an ORIGINAL live campaign is repaired, once (P5 panel F5)
        original, earlier = elsewhere / "original", elsewhere / "earlier-reserve"
        for folder, seed_set in ((original, "live"), (earlier, "reserve")):
            folder.mkdir()
            (folder / "record.summary.json").write_text(
                json.dumps({"seed_set": seed_set, "status": "COMPLETED"}), encoding="utf-8"
            )
        (fresh / REPAIR_FILE).write_text(f"{original}\nthe repair\n", encoding="utf-8")
        assert start_problems("record", fresh, str(pilot), "reserve") == []
        (fresh / REPAIR_FILE).write_text(f"{earlier}\nthe repair\n", encoding="utf-8")
        assert "never repaired again" in start_problems("record", fresh, str(pilot), "reserve")[0]
        (fresh / REPAIR_FILE).write_text(f"{original}\nthe repair\n", encoding="utf-8")
        assert claim_reserve(fresh) == str(original / RESERVE_CLAIM_FILE)
        assert (
            "already gave its one reserve attempt"
            in start_problems("record", fresh, str(pilot), "reserve")[0]
        )
        try:
            claim_reserve(fresh)
        except ps.Halt as halt:
            assert halt.check == "record: the reserve attempt is already claimed"
        else:
            raise AssertionError("a second claim of one original OUT was not refused")
        assert len(start_problems("record", fresh, None, "live")) == 1  # no pilot
        assert len(start_problems("instrument", fresh, str(pilot), "live")) == len(md.SCORED_ARMS)
    # 2.6's half-moved rule: holds on every pre-registered size for the live and the reserve seeds, and a
    # block list whose draws move under half halts
    for seed in (*cs.LIVE_SEEDS, *cs.RESERVE_SEEDS):
        check_half_moved(seed, preregistered_sizes())
    try:
        check_half_moved(cs.LIVE_SEEDS[0], [[2] * 12])
    except ps.Halt as halt:
        assert halt.check == "shuffle half-moved"
    else:
        raise AssertionError("the half-moved red did not fire")
    print(
        "carried_state_driver self-check OK: preflight (control clear; wrong cwd, OUT inside the repo, "
        f"{len(ENV_REFUSED)} overrides, a bad asset each refused), stage chain (order, head, dirty tree, "
        "moved head, no retries), the seven later stages refuse before their predecessor, --seeds reserve "
        "needs REPAIR.txt naming an original live campaign that has not spent its one reserve attempt "
        "(a reserve chain and a second claim refused), half-moved "
        f"rule holds on {len(cs.LIVE_SEEDS) + len(cs.RESERVE_SEEDS)} seeds x {len(PERMUTED_BLOCKS)} sizes "
        "and fires on a constructed red"
    )


def _parser():
    parser = argparse.ArgumentParser(
        prog="carried_state_driver",
        description="G1(c)'s campaign driver (g1c-carried-state.md section 2.13); no arguments runs the "
        "self-check",
    )
    commands = parser.add_subparsers(dest="stage")
    for stage in STAGES:
        sub = commands.add_parser(stage, help=STAGE_FUNCTIONS[stage].__doc__.split("\n")[0])
        sub.add_argument("out", help="the campaign directory (outside the repository)")
        if stage in ("record", "instrument"):
            sub.add_argument("--pilot", help="G1(a)'s pilot3 directory (100-stream.json.gz, ...)")
        if stage == "record":
            sub.add_argument("--seeds", choices=("live", "reserve"), default="live")
    gold = commands.add_parser("golden", help="the golden check (g1c 2.11), in-process; no stage")
    gold.add_argument("out", nargs="?", help="a campaign directory whose summaries are recomputed")
    gold.add_argument("--against", help="a saved golden digest (JSON): exit 1 and name what moved")
    return parser


# Sites the duplicate-capability hook flagged on the golden command (read; none runs or compares a golden check):
# distinct-from: engine/learn_daemon.py::_selfcheck -- a daemon's self-check that must never start its drive loop
# distinct-from: gates/gradient_claim_gate.py::_write_measure_artifact -- records one gate run so its age is checkable
# distinct-from: engine/autonomous_loop.py::AutonomousArm._percept_novelty -- a percept's habituating novelty signal
# distinct-from: engine/brain.py::Brain._restore_episodes -- refills the attractor from the lake
# distinct-from: faculties/agency_monitor.py::_t079_reds_line -- prints six booleans beside what each says
def run_golden(out, against):
    """The golden check (2.11) from the CLI: print its digest as JSON, or compare it with a saved one
    (exit 0 equal, 1 moved, 3 an OUT whose summaries cannot be assembled)."""
    try:
        found = rep.golden(out)
    except ValueError as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return ps.EXIT_REFUSED
    if against is None:
        print(json.dumps(found, indent=1, sort_keys=True))
        return ps.EXIT_OK
    moved = rep.differences(json.loads(Path(against).read_text(encoding="utf-8")), found)
    print("golden: " + ("equal" if not moved else "MOVED at " + ", ".join(moved)))
    return ps.EXIT_ABORT if moved else ps.EXIT_OK


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.stage is None:
        self_check()
        return ps.EXIT_OK
    problems = preflight(args.out or tempfile.gettempdir())
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return ps.EXIT_REFUSED
    if args.stage == "golden":
        return run_golden(args.out, args.against)
    extra = {}
    if args.stage in ("record", "instrument"):
        extra["pilot"] = args.pilot
    if args.stage == "record":
        extra["seeds"] = args.seeds
    try:
        return run_stage(args.stage, args.out, **extra)
    except ps.Refused as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return ps.EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
