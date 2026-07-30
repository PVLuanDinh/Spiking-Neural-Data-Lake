"""consciousness_ab_probe -- runtime neuter-A/B for the two Consciousness faculties that
`tools/charge_probe.py` cannot reach.

WHY IT EXISTS. `charge_probe.py` neuters a key of `Brain._drive_w`, so it can only probe
faculties wired into the arm's drive vector. Two Consciousness faculties are consumed
somewhere else entirely and were therefore un-A/B-able:

  * `strength`  -- the graded 32-unit access readout `ReentrantWorkspace.access_strength()`.
                   Consumed at cortex/brain.py:2136 as `min(_access_ema, _reentrant_strength)`
                   -> the self-model's `access_index` -> a MULTIPLICATIVE factor of
                   `self_model.confidence()` (cortex/self_model.py:274). Nothing in the arm.
  * `second`    -- `SecondOrderAccess.hold()`, cached as `_second_order_hold` and branched on at
                   cortex/brain.py:2949 to choose the hedged reply. Chat path only; a bare
                   tick() never evaluates it.

METHOD, and it mirrors charge_probe's: two runs at the same seed over the same tick/message
sequence. A = wired. B = the named signal neutered every tick. Any divergence in the
observable is attributable to that signal alone.

THE NEUTER IS THE DOCUMENTED FALLBACK, NOT A ZERO. For `strength` the neutered value is 1.0,
because brain.py:2134-2135 states that when no workspace exists the attribute defaults to 1.0
and `min()` becomes the identity -- so arm B is exactly the pre-de-saturation behaviour, which
is the honest counterfactual. Pinning it to 0.0 would instead collapse confidence to 0 and
"prove" a charge trivially. For `second` the neuter is `hold() -> False`, i.e. the monitor
never abstains.

AIRTIGHT, per charge_probe's `_PinnedWeights` lesson (a post-hoc zero leaks). Both neuters patch
the PRODUCING METHOD, not the cached attribute, so the live loop cannot restore the real value
mid-tick. THE FIRST VERSION OF THIS PROBE GOT THAT WRONG AND IT MATTERED: pinning
`Brain._reentrant_strength` before `tick()` is useless, because `_w8_tick` (brain.py:1671)
recomputes the attribute from the workspace INSIDE the tick, before `_observe_self_wake`
(brain.py:1821) consumes it. That leaky version reported `confidence() changed 0/120 ticks`,
i.e. it made a live faculty look dead -- the direction that gets a mechanism retired. Patching
`ReentrantWorkspace.access_strength` instead means `_w8_tick` itself writes the neutered value
and every downstream consumer sees it.

OBSERVABLES ARE BEHAVIOURAL, never telemetry:
  strength -> `self_model.confidence()` per tick and `_metacog_hold` per tick (an internal read
              and a gate), plus the arm's `map_position` trajectory.
  second   -> the REPLY TEXT per chat turn (the branch actually taken).

VERDICT, exit codes preserved from charge_probe so the coverage scorer reads them the same way:
  exit 0  BEHAVIOURAL CHARGE OK              -- neutering changed an observable
  exit 1  CONSUMED-BUT-NOT-DECISIVE          -- consumed, no divergence
  exit 2  NOT-ELIGIBLE / UNKNOWN / bad usage -- the signal never got a chance to fire

A DETERMINISM GUARD RUNS FIRST. Arm A is executed twice and the observables compared; if the
two disagree the probe exits 2 rather than reporting noise as charge.

  PYTHONPATH=src python docs/coverage/artifacts/consciousness_ab_probe.py strength 120
  PYTHONPATH=src python docs/coverage/artifacts/consciousness_ab_probe.py second 120
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from snn_data_lake.cortex.brain import make_brain  # noqa: E402

_MSG = "what is on your mind"
_CHAT_EVERY = 4
# the value brain.py:2134-2135 documents as the no-workspace default, i.e. min() as identity
_STRENGTH_NEUTER = 1.0


def run(source, n, neuter):
    """One arm. Returns (confidences, metacog_holds, positions, replies)."""
    b = make_brain(seed=0, persist=False)  # Policy 4: the ONE sanctioned construction path
    if neuter and source == "second" and b._second_order is not None:
        b._second_order.hold = lambda: False  # the monitor never abstains
    if neuter and source == "strength" and b._reentrant is not None:
        # patch the PRODUCER: _w8_tick now writes the neutered value itself, so nothing downstream
        # can read the graded one. Pinning the cached attribute instead leaks (see the docstring).
        b._reentrant.access_strength = lambda: _STRENGTH_NEUTER
        b._reentrant_strength = _STRENGTH_NEUTER
    conf, holds, pos, replies = [], [], [], []
    for i in range(n):
        b.tick()
        st = b.state()
        conf.append(round(float(b.self_model.confidence()), 6))
        holds.append(bool(getattr(b, "_metacog_hold", False)))
        pos.append(tuple(st.get("map_position") or ()))
        if i % _CHAT_EVERY == 0:
            if neuter and source == "second" and b._second_order is not None:
                b._second_order.hold = lambda: False  # re-pin: respond() re-observes first
            try:
                r = b.respond(_MSG)
            except Exception as exc:  # noqa: BLE001
                r = f"ERR {exc}"
            replies.append(str(r.get("reply") if isinstance(r, dict) else r))
    return b, conf, holds, pos, replies


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "strength"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    if source not in ("strength", "second"):
        print(f"unknown source {source!r}; known sources: second, strength")
        return 2

    # ---- determinism guard: arm A twice. Noise reported as charge is the failure mode here. ----
    b0, c0, h0, p0, r0 = run(source, n, False)
    b0.close()
    bA, cA, hA, pA, rA = run(source, n, False)
    if (c0, h0, p0, r0) != (cA, hA, pA, rA):
        bA.close()
        print("UNKNOWN: arm A is not reproducible at seed=0 -- no A/B verdict from a noisy harness")
        return 2

    bB, cB, hB, pB, rB = run(source, n, True)

    # ---- eligibility: did the signal ever actually vary in arm A? -------------------------------
    # `elig_vals` (a sorted set of rounded cA values) was computed here and never read -- caught by
    # F841 once docs/coverage/artifacts/ was linted for real defects rather than excluded. Removed
    # rather than silenced: the eligibility question it looked like it answered is actually carried by
    # `n_distinct` below, which IS read, so no measurement depended on it.
    if source == "strength":
        n_distinct = len(sorted({v for v in cA}))
    else:
        n_distinct = len(sorted(set(rA)))
    conf_flips = sum(1 for x, y in zip(cA, cB, strict=True) if x != y)
    hold_flips = sum(1 for x, y in zip(hA, hB, strict=True) if x != y)
    pos_flips = sum(1 for x, y in zip(pA, pB, strict=True) if x != y)
    reply_flips = sum(1 for x, y in zip(rA, rB, strict=True) if x != y)
    hedge_a = sum(1 for x in rA if x.startswith("I am not fully certain right now"))
    hedge_b = sum(1 for x in rB if x.startswith("I am not fully certain right now"))
    bA.close()
    bB.close()

    print(f"source                                        : {source}")
    print(f"neuter                                        : "
          f"{'_reentrant_strength pinned to 1.0 (the documented no-workspace default)' if source == 'strength' else 'SecondOrderAccess.hold() pinned False'}")
    print(f"ticks / chat turns                            : {n} / {len(rA)}")
    print(f"distinct observable values in arm A           : {n_distinct}")
    print(f"self_model.confidence() changed by the signal  : {conf_flips}/{n} ticks")
    print(f"metacog HOLD gate changed by the signal        : {hold_flips}/{n} ticks")
    print(f"arm trajectory changed by the signal          : {pos_flips}/{n} ticks")
    print(f"REPLY TEXT changed by the signal              : {reply_flips}/{len(rA)} turns")
    print(f"hedged replies  A(wired) vs B(neutered)       : {hedge_a} vs {hedge_b}")

    if n_distinct <= 1:
        print(
            "NOT-ELIGIBLE: the observable took one value in arm A, so the signal never got a chance "
            "to differ -- no charge verdict"
        )
        return 2
    if conf_flips or hold_flips or pos_flips or reply_flips:
        print("BEHAVIOURAL CHARGE OK")
        return 0
    print("CONSUMED-BUT-NOT-DECISIVE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
