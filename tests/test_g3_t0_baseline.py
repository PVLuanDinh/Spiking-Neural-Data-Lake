"""specs/022 G3 T0: the baseline capture T1 (the one-grower merge) diffs against.

Pre-registration: evidence/g3-t0-baseline.md (frozen before any brain ran). `capture()` is the
brain-running driver (7 arms x seeds 0-2 x 300 ticks, plus the bare-engine pins); invoke it
explicitly (`python -c "import tests.test_g3_t0_baseline as m; m.capture()"`); it writes
`evidence/g3-t0-baseline.json`. No test calls it: a battery that runs this file must not rewrite
the baseline. The tests recompute the H0a-H0e verdicts from the COMMITTED json, so they are fast
and brain-free; T1's R2 imports `capture`, `ARMS`, `SEEDS` and `verdicts` from here.
"""

from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import pytest

from snn_data_lake.substrate.home import throwaway_home
from tests.test_placement_charge_null import TICKS, _diff, _flatten

EVIDENCE = Path(__file__).resolve().parents[1] / "specs/022-unified-brain-arc/evidence"
BASELINE = EVIDENCE / "g3-t0-baseline.json"
SEEDS = (0, 1, 2)
SEED_SHIFT = {0: 1, 1: 2, 2: 0}  # never a no-op
ARMS = (
    "live_a",
    "live_b",
    "growth_off",
    "engine_seed_shift",
    "engine_admission_shift",
    "neuter_place",
    "reverse_tiebreak",
)
WALL_CLOCK = {"snap.t"}


def _plain(x):
    """JSON-safe form: NaN/inf become strings (NaN != NaN would read as a difference)."""
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    return x


def _view(brain):
    from snn_data_lake.infra import brain_telemetry

    flat = {
        **_flatten(brain.state(), "state."),
        **_flatten(brain_telemetry.snapshot(brain), "snap."),
    }
    return {k: _plain(json.loads(json.dumps(v, default=repr))) for k, v in flat.items()}


def _run_arm(arm, seed):
    """One Brain for one arm under a throwaway home; returns the section 4 record."""
    import snn_data_lake.growth.placement as placement
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.growth import candidate_lifecycle
    from snn_data_lake.growth.neurogenesis_engine import growth_fingerprint

    real_place, real_select = placement.place, candidate_lifecycle.select_block
    t0 = time.time()
    try:
        if arm == "neuter_place":
            placement.place = lambda n_units, seed=0, region="cortex": (
                [(0.5, "Layer 2", 1)] * n_units
            )
        if arm == "reverse_tiebreak":
            candidate_lifecycle.select_block = lambda *a, **k: real_select(
                *a, **{**k, "reverse": True}
            )
        with throwaway_home():
            b = make_brain(seed=seed, persist=False)
            tier0 = (b.state()["perception_tier"], b.state()["perceiving_audio"])
            if arm == "growth_off":
                b.neuro.tick = lambda: None
            if arm == "engine_seed_shift":
                b.neuro.seed = SEED_SHIFT[seed]
            if arm == "engine_admission_shift":
                b.neuro._eps_tol = 0.05
            for _ in range(TICKS):
                b.tick()
            st = b.state()
            rec = {
                "arm": arm,
                "seed": seed,
                "view": _view(b),
                "fingerprint": growth_fingerprint(b.neuro) if b.neuro.grown_units else None,
                "trajectory_crc": zlib.crc32(json.dumps(b.neuro.trajectory, default=str).encode()),
                "grown_units": int(b.neuro.grown_units),
                "predict_accuracy": _plain(float(b.neuro.predict_accuracy)),
                "placement": json.loads(json.dumps(list(b.neuro.placement), default=repr)),
                "perception_tier": [tier0[0], st["perception_tier"]],
                "perceiving_audio": [tier0[1], st["perceiving_audio"]],
                "device": str(b.neuro.device),
                "engine_seed": int(b.neuro.seed),
                "ticks": TICKS,
                "wall_s": round(time.time() - t0, 1),
                "engine_state": json.loads(json.dumps(b.neuro.state(), default=repr)),
            }
            try:
                b.close()
            except Exception:
                pass
            return rec
    finally:
        placement.place, candidate_lifecycle.select_block = real_place, real_select


def _bare_pins():
    """The test_candidate_lifecycle.py pins, re-measured on this tree (bare engines)."""
    import torch

    from snn_data_lake.growth.neurogenesis_engine import growth_fingerprint
    from snn_data_lake.readout.online_readout import ClosingGate, OnlineReadout, _run
    from tests.test_candidate_lifecycle import _stream, _traj

    out = {}
    for seed, ticks in sorted([(s, t) for s in (9000, 9001, 9002, 9003) for t in (200, 600)]):
        e = _stream(seed, ticks)
        out[f"stream:{seed}:{ticks}"] = [growth_fingerprint(e), int(e.grown_units), _traj(e)]
    for seed, name in sorted([(s, n) for s in (0, 1) for n in ("batch", "delta", "delta_open")]):
        torch.set_num_threads(1)
        e = _run(
            seed,
            None if name == "batch" else OnlineReadout("delta"),
            None if name == "delta_open" else ClosingGate(3),
        )
        out[f"online:{seed}:{name}"] = [growth_fingerprint(e), int(e.grown_units), _traj(e)]
    return out


def capture(path=BASELINE):
    """Run all 21 brains and the bare-engine pins; write the baseline json (stdlib tier stops)."""
    import platform

    import torch

    arms = []
    for seed in SEEDS:
        for arm in ARMS:
            rec = _run_arm(arm, seed)
            print(
                f"seed {seed} {arm}: grown={rec['grown_units']} "
                f"tier={rec['perception_tier']} {rec['wall_s']}s",
                file=sys.stderr,
                flush=True,
            )
            if "stdlib" in rec["perception_tier"]:
                raise SystemExit(
                    f"STDLIB perception tier at seed {seed} arm {arm}: launch fault, stop"
                )
            arms.append(rec)
    from snn_data_lake.instruments.prior_drift_harness import _git_head

    head = _git_head()  # cwd=REPO, timeout; recorded only (R2 anchors on the capture commit)
    data = {
        "base_commit": head,
        "python": sys.version,
        "torch": torch.__version__,
        "host": platform.node(),
        "ticks": TICKS,
        "arms": arms,
        "bare_pins": _bare_pins(),
    }
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    return data


def _by(data):
    return {(a["seed"], a["arm"]): a for a in data["arms"]}


def verdicts(data):
    """H0a-H0e per the pre-registration, from a capture's json: {name: (ok, detail)}."""
    by = _by(data)
    seeds = sorted({a["seed"] for a in data["arms"]})
    base, moved = {}, {}
    for s in seeds:
        live = by[(s, "live_a")]
        base[s] = set(_diff(live["view"], by[(s, "live_b")]["view"], set()))
        for arm in ARMS[2:]:
            moved[(s, arm)] = _diff(live["view"], by[(s, arm)]["view"], base[s])
    h0a = {s: sorted(base[s] - WALL_CLOCK) for s in seeds}
    tiers = {
        s: by[(s, "live_a")]["perception_tier"] == by[(s, "live_b")]["perception_tier"]
        for s in seeds
    }
    h0b = {
        s: by[(s, "growth_off")]["grown_units"] == 0
        and "state.grown_units" in moved[(s, "growth_off")]
        for s in seeds
    }
    h0c_place = {
        s: by[(s, "engine_seed_shift")]["placement"] != by[(s, "live_a")]["placement"]
        for s in seeds
    }

    def field_moved(s, arm):
        a, live = by[(s, arm)], by[(s, "live_a")]
        return a["fingerprint"] != live["fingerprint"] or any(
            k in moved[(s, arm)] for k in ("state.grown_units", "state.predict_accuracy")
        )

    h0d = {
        s: any(
            k in moved[(s, "engine_admission_shift")]
            for k in ("state.predict_accuracy", "state.grown_units")
        )
        for s in seeds
    }
    h0e = {
        s: by[(s, "reverse_tiebreak")]["fingerprint"] != by[(s, "live_a")]["fingerprint"]
        for s in seeds
    }
    return {
        "H0a": (
            all(not v for v in h0a.values()) and all(tiers.values()),
            {"extra_baseline_keys": h0a, "tier_stable": tiers},
        ),
        "H0b": (all(h0b.values()), h0b),
        "H0c": (
            all(h0c_place.values()) and any(field_moved(s, "engine_seed_shift") for s in seeds),
            {
                "placement_moved": h0c_place,
                "field_moved": {s: field_moved(s, "engine_seed_shift") for s in seeds},
            },
        ),
        "H0d": (all(h0d.values()), h0d),
        "H0e": (any(h0e.values()), h0e),
    }


def _data():
    assert BASELINE.exists(), f"{BASELINE} is not committed: run capture() (explicit, never a test)"
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_differencer_names_a_one_ulp_edit():
    # R-T0-6: the differencer is not a no-op
    view = {"state.a": 1.0, "state.b": 0.1, "snap.t": 5.0}
    edited = dict(view, **{"state.b": math.nextafter(0.1, 1.0)})
    assert _diff(view, dict(view), set()) == []
    assert _diff(view, edited, set()) == ["state.b"]


def test_capture_covers_every_seed_and_arm_on_a_perceiving_tier():
    data = _data()
    assert {(a["seed"], a["arm"]) for a in data["arms"]} == {(s, m) for s in SEEDS for m in ARMS}
    for a in data["arms"]:
        assert "stdlib" not in a["perception_tier"], (a["seed"], a["arm"], a["perception_tier"])
        assert a["ticks"] == TICKS


@pytest.mark.parametrize("name", ["H0a", "H0b", "H0c", "H0d", "H0e"])
def test_harness_verdicts_hold_on_the_committed_capture(name):
    ok, detail = verdicts(_data())[name]
    assert ok, f"FAILED CONTROL {name}: {detail}"


def test_baseline_nondeterminism_is_only_the_wall_clock():
    data = _data()
    by = _by(data)
    for s in SEEDS:
        assert set(_diff(by[(s, "live_a")]["view"], by[(s, "live_b")]["view"], set())) <= WALL_CLOCK


def test_bare_engine_pins_match_the_committed_values():
    from tests.test_candidate_lifecycle import ONLINE, STREAM, TRAJ

    pins = _data()["bare_pins"]
    for (seed, ticks), (fp, grown) in STREAM.items():
        assert pins[f"stream:{seed}:{ticks}"] == [fp, grown, TRAJ[(seed, ticks)]]
    for (seed, name), (fp, traj, grown) in ONLINE.items():
        assert pins[f"online:{seed}:{name}"] == [fp, grown, traj]
