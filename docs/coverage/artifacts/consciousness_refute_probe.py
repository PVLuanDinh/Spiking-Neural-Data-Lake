"""consciousness_refute_probe -- adversarial re-measurement of the two band-41+ Consciousness rows.

The rows under attack are `access_strength_graded` (55) and `second_order_access` (50), both
resting on `docs/coverage/artifacts/consciousness_ab_probe.py`. Both of that probe's artifacts
DO reproduce under `coverage_score.py --rerun`, so this is not a forgery check. It is an
ATTRIBUTION check: what else does the neuter change, and which row owns the divergence.

THREE MODES.

`strength_paths N` -- the auditor's exact neuter (`ReentrantWorkspace.access_strength -> 1.0`),
    but logging every other live consumer of `_reentrant_strength` as well. There are three, not
    one:
      cortex/brain.py:2137  self-model access_index  (the claimed consumer; getattr default 1.0)
      cortex/brain.py:2040  the region-coherence "access" raster -> synfire binding gate ->
                            `_reentrant.ignition_threshold`  (getattr default 0.0)
      cortex/brain.py:2843  `SecondOrderAccess.observe(availability, STRENGTH, ...)`
                            (getattr default 0.0)
    So "1.0 is the documented no-workspace default" holds at ONE of the three sites and is the
    OPPOSITE extreme at the other two. If the coherence raster / binding flag / ignition
    threshold diverge, arm B is not "the pre-de-saturation brain": it is a brain whose access
    raster is pinned at maximum, feeding back into the ignition of the workspace being probed.

`strength_iso N` -- the counterfactual the row's own text claims to be testing, and only that:
    `access_strength()` is left ALONE (so :2040 and :2843 see the true graded value) and the
    self-model's `access_index` is overridden to `_access_ema`, i.e. min() as the identity, which
    is exactly what brain.py:2134-2135 documents as the no-workspace fallback. Whatever survives
    here is what the graded readout charges THROUGH THE SELF-MODEL, with no other path attached.

`second_thin N` -- the auditor's `hold() -> False` neuter, plus the one thing their probe did not
    print: for every diverging turn, whether arm A's reply is arm B's reply with the canned
    hedge prefix bolted on. If the tails are identical the faculty's whole behavioural footprint
    is a 37-character constant, which is band-floor evidence, not mid-band.

`second_c1 N` / `second_ws N` -- WHOSE information does the second-order abstain decision carry?
    `SecondOrderAccess.observe(availability, access_strength, ignited, first_order_conf)` is fed
    three workspace quantities and one EXECUTIVE quantity (`_metacog_rel`, brain.py:2845). These
    two modes replace one side's inputs with their own RUNNING MEAN -- magnitude kept, information
    removed, which is the neuter form P1 asks for -- and leave the other side live:
      `second_c1` removes the Executive input.   `second_ws` removes the workspace inputs.
    If the hedges survive `second_ws` but collapse under `second_c1`, the monitor is a LIF
    re-transduction of Executive's reliability and its charge is not a Consciousness charge.
    `second_av` / `second_st` split the workspace side further: availability alone, strength alone.
    Availability is provably the ignited bit in this wiring (Consciousness.md's own
    `global_availability` finding), so a hold that rides on availability alone is the same cause
    as the `ignition` and `global_availability` rows, not a third one.

METHOD is charge_probe's: two runs, same seed, same tick/message sequence, one signal removed in
arm B. Arm A is run twice first as a determinism guard. Exit codes are charge_probe's, so
`coverage_score.py` reads them the same way:
  exit 0  BEHAVIOURAL CHARGE OK      -- neutering changed an observable
  exit 1  CONSUMED-BUT-NOT-DECISIVE  -- consumed, no divergence
  exit 2  NOT-ELIGIBLE / UNKNOWN     -- the signal never got a chance to differ, or a noisy harness

  PYTHONPATH=src python docs/coverage/artifacts/consciousness_refute_probe.py strength_paths 120
  PYTHONPATH=src python docs/coverage/artifacts/consciousness_refute_probe.py strength_iso 120
  PYTHONPATH=src python docs/coverage/artifacts/consciousness_refute_probe.py second_thin 120
"""

import os
import sys

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cortex.brain import make_brain  # noqa: E402
from snn_data_lake.cortex.self_model import _conf_bucket  # noqa: E402

_MSG = "what is on your mind"
_CHAT_EVERY = 4
_HEDGE = "I am not fully certain right now, but "
_STRENGTH_NEUTER = 1.0
MODES = (
    "strength_paths",
    "strength_iso",
    "second_thin",
    "second_c1",
    "second_ws",
    "second_av",
    "second_st",
)
_INPUT_MODES = ("second_c1", "second_ws", "second_av", "second_st")


def _snap(b):
    """One tick's observables. Behavioural / internal-read quantities only."""
    rw = getattr(b, "_reentrant", None)
    return {
        "conf": round(float(b.self_model.confidence()), 6),
        "conf_bucket": int(_conf_bucket(b.self_model.confidence())),
        "metacog_rel": round(float(getattr(b, "_metacog_rel", 0.0) or 0.0), 4),
        "metacog_hold": bool(getattr(b, "_metacog_hold", False)),
        "learn_gate_open": int(getattr(b, "_ll_i", 0) or 0),
        # --- the contamination channels: other live consumers of _reentrant_strength ---
        "coherence": round(float(getattr(b, "_region_coherence_val", 0.0) or 0.0), 4),
        "bound": bool(getattr(b, "_bound", False)),
        "ign_thresh": round(float(getattr(rw, "ignition_threshold", 0.0) or 0.0), 6),
        "availability": round(float(getattr(b, "_reentrant_availability", 0.0) or 0.0), 4),
        "second_hold": bool(getattr(b, "_second_order_hold", False)),
        "pos": tuple(b.state().get("map_position") or ()),
    }


def run(mode, n, neuter):
    """One arm. Returns (brain, [per-tick snapshot], [reply per chat turn])."""
    b = make_brain(seed=0, persist=False)  # Policy 4: the ONE sanctioned construction path
    if neuter:
        if mode == "strength_paths" and b._reentrant is not None:
            # the auditor's neuter, verbatim: patch the producer so _w8_tick writes 1.0 itself
            b._reentrant.access_strength = lambda: _STRENGTH_NEUTER
            b._reentrant_strength = _STRENGTH_NEUTER
        elif mode == "strength_iso":
            # ONLY the self-model consumer. access_strength() stays live, so the coherence raster
            # and the second-order monitor still see the real graded value.
            _orig = b.self_model.observe

            def _iso(**kw):
                if "access_index" in kw:
                    kw["access_index"] = round(float(getattr(b, "_access_ema", 0.0)), 6)
                return _orig(**kw)

            b.self_model.observe = _iso
        elif mode == "second_thin" and b._second_order is not None:
            b._second_order.hold = lambda: False  # the monitor never abstains
        elif mode in _INPUT_MODES and b._second_order is not None:
            # replace one side's inputs by their RUNNING MEAN: same scale, no information.
            _obs = b._second_order.observe
            _run = {"n": 0, "c1": 0.0, "av": 0.0, "st": 0.0}

            def _mean(key, x):
                _run[key] += (float(x) - _run[key]) / _run["n"]
                return _run[key]

            def _patched(availability, access_strength, ignited, first_order_conf):
                _run["n"] += 1
                c1 = _mean("c1", first_order_conf)
                av = _mean("av", availability)
                st = _mean("st", access_strength)
                if mode == "second_c1":
                    return _obs(availability, access_strength, ignited, c1)
                if mode == "second_av":
                    return _obs(av, access_strength, ignited, first_order_conf)
                if mode == "second_st":
                    return _obs(availability, st, ignited, first_order_conf)
                return _obs(av, st, ignited, first_order_conf)

            b._second_order.observe = _patched
    snaps, replies = [], []
    for i in range(n):
        b.tick()
        snaps.append(_snap(b))
        if i % _CHAT_EVERY == 0:
            if neuter and mode == "second_thin" and b._second_order is not None:
                b._second_order.hold = lambda: False  # re-pin: respond() re-observes first
            try:
                r = b.respond(_MSG)
            except Exception as exc:  # noqa: BLE001
                r = f"ERR {exc}"
            replies.append(str(r.get("reply") if isinstance(r, dict) else r))
    return b, snaps, replies


def _flips(a, bb, key):
    return sum(1 for x, y in zip(a, bb, strict=True) if x[key] != y[key])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "strength_paths"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    if mode not in MODES:
        print(f"unknown mode {mode!r}; known: {', '.join(MODES)}")
        return 2

    b0, s0, r0 = run(mode, n, False)
    b0.close()
    bA, sA, rA = run(mode, n, False)
    if (s0, r0) != (sA, rA):
        bA.close()
        print("UNKNOWN: arm A is not reproducible at seed=0 -- no verdict from a noisy harness")
        return 2
    bB, sB, rB = run(mode, n, True)
    bA.close()
    bB.close()

    reply_flips = sum(1 for x, y in zip(rA, rB, strict=True) if x != y)
    hedge_a = sum(1 for x in rA if x.startswith(_HEDGE))
    hedge_b = sum(1 for x in rB if x.startswith(_HEDGE))
    f = {k: _flips(sA, sB, k) for k in sA[0]}
    print(f"mode                                          : {mode}")
    print(f"ticks / chat turns                            : {n} / {len(rA)}")
    print("--- the row's CLAIMED consumers -------------------------------------------")
    print(f"self_model.confidence() changed               : {f['conf']}/{n} ticks")
    print(f"  its BUCKET (what reaches the episode ring)  : {f['conf_bucket']}/{n} ticks")
    print(f"metacog reliability changed                   : {f['metacog_rel']}/{n} ticks")
    print(f"metacog HOLD gate changed                     : {f['metacog_hold']}/{n} ticks")
    print(f"lifelong learn-gate admissions changed        : {f['learn_gate_open']}/{n} ticks")
    print(f"arm trajectory changed                        : {f['pos']}/{n} ticks")
    print(f"REPLY TEXT changed                            : {reply_flips}/{len(rA)} turns")
    print(f"hedged replies  A(wired) vs B(neutered)       : {hedge_a} vs {hedge_b}")
    print("--- OTHER live consumers of the same neuter (attribution) -----------------")
    print(f"region-coherence value changed (brain.py:2040): {f['coherence']}/{n} ticks")
    print(f"synfire BINDING flag changed                  : {f['bound']}/{n} ticks")
    print(f"workspace IGNITION THRESHOLD changed          : {f['ign_thresh']}/{n} ticks")
    print(f"workspace availability changed                : {f['availability']}/{n} ticks")
    print(f"second-order HOLD changed (brain.py:2843)     : {f['second_hold']}/{n} ticks")

    if mode == "second_thin":
        # is the whole footprint the canned prefix?
        prefix_only = same_tail = other = 0
        for x, y in zip(rA, rB, strict=True):
            if x == y:
                continue
            if x == _HEDGE + y:
                prefix_only += 1
            elif x.startswith(_HEDGE) and x[len(_HEDGE) :] == y:
                same_tail += 1
            else:
                other += 1
        tails_equal = prefix_only + same_tail
        print("--- is the footprint a constant prefix? -----------------------------------")
        print(f"diverging turns where A == HEDGE + B          : {prefix_only}/{len(rA)} turns")
        print(f"diverging turns, hedge stripped, tails equal  : {tails_equal}/{len(rA)} turns")
        print(f"diverging turns, genuinely different tail     : {other}/{len(rA)} turns")

    distinct = len({s["conf"] for s in sA}) if mode.startswith("strength") else len(set(rA))
    print(f"distinct observable values in arm A           : {distinct}")
    if distinct <= 1:
        print("NOT-ELIGIBLE: the observable took one value in arm A -- no charge verdict")
        return 2
    moved = any(
        _flips(sA, sB, k)
        for k in ("conf", "conf_bucket", "metacog_rel", "metacog_hold", "learn_gate_open", "pos")
    )
    if moved or reply_flips:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
