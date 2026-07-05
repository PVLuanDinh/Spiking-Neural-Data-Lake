"""
v0.34 — reflex fast-path: instinctive STOP / WITHDRAW on danger signals  (#2).

The biological reflex arc: a few high-priority "nociceptor" channels (collision force,
over/under current, over-temperature) feed a fast LIF that fires a protective action
BEFORE the recognition pipeline (encode -> lake -> match) runs. It is:
  - fast       : crosses threshold in one step for a severe breach (no classification wait)
  - unlearned  : hardwired limits, not trained (that's the learned counterpart, valence_stdp.py)
  - sign-aware : operates on RAW signed samples, not the min-max-normalized window, so a
                 reverse-direction overload (negative) is caught as well as a positive one.

The arm stops on a collision NOW and classifies later.

  python reflex.py
"""
import math
import sys

# channel -> (lo, hi, action): a raw sample outside [lo, hi] is dangerous on that channel.
# Example for the robot arm (raw ADC-ish). lo < 0 catches reverse-direction danger.
DEFAULT_RULES = {
    6: (-50.0, 800.0, "STOP"),        # ch6 motor current: collision / stall spike
    7: (-300.0, 300.0, "WITHDRAW"),   # ch7 force/torque: over-range in either direction
}


class Reflex:
    """One fast LIF. Each violated priority channel injects current proportional to how
    far past its limit (as a fraction of the channel's range); when the membrane crosses
    threshold the reflex fires the most-severe channel's action, then resets (refractory).
    Severe breaches fire in a single step; a marginal breach whose current stays below the
    membrane's leaky steady state still fires once it PERSISTS `persist` consecutive samples
    (a dead-band-free latch — a sustained moderate overload can never be ignored). A rule
    channel that is absent from a frame is reported (once) instead of silently skipped."""

    def __init__(self, rules=None, tau=2.0, threshold=0.10, persist=10):
        self.rules = DEFAULT_RULES if rules is None else rules
        self.decay = math.exp(-1.0 / tau)
        self.threshold = threshold
        self.persist = persist
        self.v = 0.0
        self._breach = 0            # consecutive-breach counter (the persistence latch)
        self.missing = set()        # rule channels seen absent from a frame (observability + test hook)

    def step(self, raw):
        """Feed one raw sample (list of channel values). Return an action string the
        instant the reflex fires, else None."""
        current, worst, worst_sev = 0.0, None, 0.0
        for ch, (lo, hi, action) in self.rules.items():
            if ch >= len(raw):
                if ch not in self.missing:           # surface a disabled safety channel, don't hide it
                    self.missing.add(ch)
                    sys.stderr.write(f"reflex: safety channel {ch} absent from a {len(raw)}-wide "
                                     f"frame — its '{action}' rule is INACTIVE\n")
                continue
            x = raw[ch]
            over = (lo - x) if x < lo else (x - hi if x > hi else 0.0)
            if over > 0.0:
                sev = over / max(1e-9, hi - lo)      # scale-free severity
                current += sev
                if sev > worst_sev:
                    worst_sev, worst = sev, action
        self._breach = self._breach + 1 if worst is not None else 0
        self.v = self.v * self.decay + current
        # fire on a fast severe crossing OR a sustained (persistent) breach the leaky term would miss
        if worst is not None and (self.v >= self.threshold or self._breach >= self.persist):
            self.v = 0.0                             # fire + refractory reset
            self._breach = 0
            return worst
        return None


def reflex_guard(rows, reflex, on_reflex):
    """Side-channel monitor over a RAW sample stream: pass every row through unchanged,
    but the instant a sample trips the reflex, call on_reflex(action, seq). The reflex
    does not consume windows — it runs ahead of, and in parallel with, the matcher."""
    seq = 0
    for row in rows:
        seq += 1
        action = reflex.step(row)
        if action:
            on_reflex(action, seq)
        yield row


def main():
    # 1. safe signal never fires
    r = Reflex()
    safe = [500, 500, 500, 500, 500, 500, 400, 0]
    assert all(r.step(safe) is None for _ in range(50)), "reflex fired on a safe signal"

    # 2. collision: ch6 current spikes past 800 -> STOP
    r2, a = Reflex(), None
    for _ in range(3):
        a = r2.step([500, 500, 500, 500, 500, 500, 950, 0])
        if a:
            break
    assert a == "STOP", f"collision did not trigger STOP: {a}"

    # 3. sign-aware: ch7 force = -500 (below -300) -> WITHDRAW (negative danger caught)
    r3, a3 = Reflex(), None
    for _ in range(3):
        a3 = r3.step([500, 500, 500, 500, 500, 500, 400, -500])
        if a3:
            break
    assert a3 == "WITHDRAW", f"reverse-force did not WITHDRAW: {a3}"

    # 4. a severe breach fires in ONE step (instinct = fast)
    assert Reflex().step([500, 500, 500, 500, 500, 500, 5000, 0]) == "STOP", "severe not instant"

    # 5. guard emits actions as a side channel over a raw stream
    fired = []
    rows = [[500, 500, 500, 500, 500, 500, 400, 0]] * 3 + [[0, 0, 0, 0, 0, 0, 1500, 0]]
    list(reflex_guard(rows, Reflex(), lambda act, seq: fired.append((seq, act))))
    assert fired and fired[0][1] == "STOP", f"guard did not surface the reflex: {fired}"

    # 6. persistence: a MARGINAL sustained breach (below the leaky steady state) still fires -- no dead-band
    r6 = Reflex(persist=10)
    marginal = [500, 500, 500, 500, 500, 500, 815, 0]     # ch6 = 815: over=15, sev~0.018 -> membrane never crosses
    steps = [r6.step(marginal) for _ in range(15)]
    fire_at = next((i for i, a in enumerate(steps) if a), None)
    assert fire_at is not None and steps[fire_at] == "STOP", f"a sustained marginal breach never fired (dead-band): {steps}"
    assert fire_at >= 5, f"marginal breach should need to PERSIST, not fire instantly ({fire_at})"

    # 7. a rule channel absent from a short frame is reported, and present channels still protect
    r7 = Reflex()
    a7 = r7.step([0, 0, 0, 0, 0, 0, 950])                 # 7-wide: ch6 present (breach->STOP), ch7 absent
    assert a7 == "STOP", f"present safety channel must still fire on a short frame ({a7})"
    assert 7 in r7.missing, "an absent safety channel must be recorded, not silently skipped"

    print("=" * 56)
    print("REFLEX FAST-PATH (instinctive withdraw / stop)")
    print("=" * 56)
    print(f"rules: {DEFAULT_RULES}")
    print("safe -> no fire | collision -> STOP | reverse-force -> WITHDRAW | severe -> 1 step")
    print(f"stream guard fired at sample {fired[0][0]}: {fired[0][1]}")
    print("self-check OK: safe ignored, breaches fire correct action, sign-aware, fast")


if __name__ == "__main__":
    main()
