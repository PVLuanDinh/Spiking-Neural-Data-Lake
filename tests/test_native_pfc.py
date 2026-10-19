"""Phase 2 of the Rust port: `PrefrontalExecutive._decide_spiking` runs in `sndl-core`
(`prefrontal_executive::decide_spiking`) through `_native.pfc_decide`, on the instance's own
CPython MT19937 stream (the state round-trips through getstate/setstate).
Oracle: `native/sndl-core/tests/fixtures_pfc.json`
recorded from the deleted Python body at c612c244 (the 5 Phase 0 `cnt220` full cases + three chains,
460 calls; lane pfc-p2 REPORT.md 1.2 C2/C2'/R1). Each assert here goes through the PYTHON method,
so it judges the wrapper's marshalling as well as the Rust race.
"""

import inspect
import json
import pathlib
import struct
import zlib

from snn_data_lake import _native
from snn_data_lake.faculties.prefrontal_executive import PrefrontalExecutive

FIX = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "native"
        / "sndl-core"
        / "tests"
        / "fixtures_pfc.json"
    ).read_text()
)


def _crc(state):
    return zlib.crc32(struct.pack("<625I", *state[1]))


def _executive(chain):
    p = PrefrontalExecutive(**chain["ctor"])
    for k, v in chain["params"].items():  # the instance constants must be the oracle's
        assert getattr(p, "_" + k) == v, k
    p.rng.setstate((3, tuple(chain["state0"]), None))
    return p


def test_the_five_phase0_full_cases_reproduce_from_their_pre_state():
    p = PrefrontalExecutive(13, seed=0)
    for f in FIX["full"]:
        for k, v in f["params"].items():
            setattr(p, k, v)
        p.rng.setstate((f["pre"][0], tuple(f["pre"][1]), f["pre"][2]))
        assert list(p._decide_spiking(list(f["evidence"]))) == f["out"]
        assert p.rng.getstate() == (f["post"][0], tuple(f["post"][1]), f["post"][2])


def test_the_oracle_chains_replay_call_for_call_through_the_python_method():
    calls = zero = timeouts = 0
    for ch in FIX["chains"]:
        p = _executive(ch)
        for i, c in enumerate(ch["calls"]):
            out = p._decide_spiking(list(c["e"]))
            assert list(out) == c["out"], (ch["name"], i)
            assert _crc(p.rng.getstate()) == c["crc"], (ch["name"], i)
            calls += 1
            zero += max(c["e"]) <= 0.0
            timeouts += out[1] == p._t_max
        assert p.rng.getstate()[1] == tuple(ch["state_end"]), ch["name"]
    assert (calls, zero, timeouts) == (
        FIX["counts"]["calls"],
        FIX["counts"]["zero_path"],
        FIX["counts"]["timeouts"],
    )
    assert (calls, zero, timeouts) == (460, 23, 206)


def test_red_one_extra_draw_before_the_race_changes_the_post_state():
    """The Phase 0 constructed red: a stream offset of one draw is mostly invisible to (winner, rt)
    and always visible in the post-state: that comparison above is the load-bearing one."""
    (ch,) = [c for c in FIX["chains"] if c["name"] == "A"]
    p = _executive(ch)
    differs = zero_differs = out_changed = 0
    for c in ch["calls"]:
        pre = p.rng.getstate()
        p.rng.random()  # the red
        out = p._decide_spiking(list(c["e"]))
        d = _crc(p.rng.getstate()) != c["crc"]
        differs += d
        zero_differs += d and max(c["e"]) <= 0.0
        out_changed += list(out) != c["out"]
        p.rng.setstate(pre)
        assert list(p._decide_spiking(list(c["e"]))) == c["out"]  # back on the oracle trajectory
    # the pre-call draw is itself a state change, so EVERY call differs (zero-path ones included:
    # how Phase 0's 20/20 fired, on zero-evidence calls; the Rust twin draws inside the race)
    n_zero = sum(1 for c in ch["calls"] if max(c["e"]) <= 0.0)
    assert (differs, zero_differs, n_zero) == (len(ch["calls"]), n_zero, 15), (
        differs,
        zero_differs,
        n_zero,
    )
    print(
        f"red: post-state differed {differs}/{len(ch['calls'])}, (winner, rt) changed {out_changed}"
    )


def test_binding_round_trips_the_internal_state():
    """`_native.pfc_decide` takes and returns `getstate()[1]` (624 words + position)."""
    (ch,) = [c for c in FIX["chains"] if c["name"] == "A"]
    c = next(c for c in ch["calls"] if max(c["e"]) > 0.0)
    q = ch["params"]
    w, rt, st = _native.pfc_decide(
        list(c["e"]),
        (q["n_dec"], q["n_inh"], q["t_max"]),
        (
            q["md"],
            q["g_ext"],
            q["b0"],
            q["w_plus"],
            q["w_inh"],
            q["w_ie"],
            q["w_cross"],
            q["trace_decay"],
            q["theta"],
            q["noise_amp"],
            q["init_v"],
        ),
        q["dec_bias"],
        tuple(ch["state0"]),
    )
    assert isinstance(st, tuple) and len(st) == 625 and [w, rt] == c["out"]
    assert zlib.crc32(struct.pack("<625I", *st)) == c["crc"]


def test_the_python_race_body_is_gone_and_nothing_falls_back_to_it():
    """P2: one mechanism. The method IS the native call: no flag, no `except`, no Python loop."""
    src = inspect.getsource(PrefrontalExecutive._decide_spiking)
    assert "_native.pfc_decide" in src
    for leftover in ("rnd()", "for t in range(1, self._t_max", "except", "fallback"):
        assert leftover not in src, leftover
