"""
Reticular formation / ascending reticular activating system (ARAS) -- the sleep-wake arousal switch.

The brainstem reticular formation sets the brain's global AROUSAL state: how strongly the cortex is
driven, from deep sleep to alert waking. Saper's model casts this as a FLIP-FLOP -- a wake-promoting
population and a sleep-promoting population locked in MUTUAL INHIBITION. Mutual inhibition plus self-
excitation makes the pair BISTABLE: the network snaps to "mostly wake" or "mostly sleep" and resists
the middle ground, which is why real sleep-wake transitions are sharp rather than a slow dimmer. The
wake side is pushed by ascending sensory/reticular DRIVE; the sleep side is pushed by a HOMEOSTATIC
sleep pressure that builds with time awake. The readout is a single arousal level in [0, 1] -- a
cortical GAIN state.

Mechanism (rate-level flip-flop, deterministic given the seed):
  w <- (1-tau)*w + tau * sigmoid(gain * ( drive + SELF*w - MUTUAL*s - BIAS + noise ))
  s <- (1-tau)*s + tau * sigmoid(gain * ( homeo + SELF*s - MUTUAL*w - BIAS + noise ))
  arousal = w   (wake-population activity, bounded in [0, 1])
  - ascending DRIVE excites the wake population; HOMEOSTATIC pressure excites the sleep population;
  - each population self-excites (SELF) and inhibits the other (MUTUAL) -> a bistable toggle;
  - a small per-step noise (seeded) models reticular jitter and probes attractor robustness.

Fair control (un-riggable -- the switch must be genuinely BISTABLE, not a graded dial):
  (a) TWO ATTRACTORS: with NO ascending drive the network settles to the SLEEP attractor (low
      arousal) whatever its start; with STRONG drive it settles to the WAKE attractor (high arousal).
  (b) HYSTERESIS: sweep drive UP from sleep, then DOWN from wake. The UP switch threshold (asleep ->
      awake) is strictly ABOVE the DOWN threshold (awake -> asleep) -- the loop has width.
  (c) HISTORY DEPENDENCE: at one intermediate drive inside the loop, TWO distinct stable states
      coexist -- starting asleep stays low, starting awake stays high, on identical input.
  A graded (non-bistable) unit would give one arousal per drive with no loop and no coexisting states;
  the asserts below would fail. That is what makes the bistability claim un-riggable.

Honest framing: this is an arousal-STATE function -- it sets a global cortical GAIN level, nothing
more. It is NOT consciousness or awareness, it commands no motion, and it is not in the arm control
loop. "Wake" here means the wake-population attractor of a two-variable flip-flop, not being awake.
Credit: Moruzzi & Magoun 1949 (ARAS); Saper, Chou & Scammell 2001 (sleep-wake flip-flop);
Wilson & Cowan 1972 (mutual-inhibition bistability).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.pons.reticular_formation
"""

import math
import random

SELF_EXC = 2.0  # self-excitation of each population
MUTUAL = 4.0  # mutual inhibition between wake and sleep (drives bistability)
BIAS = 2.2  # activation threshold (> SELF_EXC: a lone pop needs external drive to sustain)
GAIN = 6.0  # sigmoid steepness (sharp switch)
TAU = 0.5  # relaxation rate per step
NOISE = 0.02  # seeded per-step reticular jitter
ITERS = 80  # settling steps
HOMEO = 0.6  # homeostatic sleep pressure onto the sleep population (fixed, moderate)
DRIVE_MAX = 2.0  # strongest ascending drive in the sweep
DRIVE_STEP = 0.05  # sweep resolution
DRIVE_MID = 1.0  # intermediate drive used for the coexisting-states test


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class ReticularFormation:
    """Wake vs sleep mutual-inhibition flip-flop -> a bistable arousal level in [0, 1]."""

    def __init__(
        self,
        self_exc=SELF_EXC,
        mutual=MUTUAL,
        bias=BIAS,
        gain=GAIN,
        tau=TAU,
        noise=NOISE,
        homeo=HOMEO,
        iters=ITERS,
        seed=0,
    ):
        self.self_exc = self_exc
        self.mutual = mutual
        self.bias = bias
        self.gain = gain
        self.tau = tau
        self.noise = noise
        self.homeo = homeo
        self.iters = iters
        self.rng = random.Random(seed)

    def _start(self, start):
        """Initial (w, s): 'sleep' = sleep-dominant, 'wake' = wake-dominant, or an explicit pair."""
        if start == "sleep":
            return 0.0, 1.0
        if start == "wake":
            return 1.0, 0.0
        return start

    def settle(self, drive, start="sleep", homeo=None):
        """Relax the flip-flop to its attractor for a fixed drive; return (w, s)."""
        homeo = self.homeo if homeo is None else homeo
        w, s = self._start(start)
        for _ in range(self.iters):
            nw = self.rng.uniform(-self.noise, self.noise)
            ns = self.rng.uniform(-self.noise, self.noise)
            w_t = _sigmoid(
                self.gain * (drive + self.self_exc * w - self.mutual * s - self.bias + nw)
            )
            s_t = _sigmoid(
                self.gain * (homeo + self.self_exc * s - self.mutual * w - self.bias + ns)
            )
            w = (1.0 - self.tau) * w + self.tau * w_t
            s = (1.0 - self.tau) * s + self.tau * s_t
        return w, s

    def arousal(self, drive, start="sleep", homeo=None):
        """Global arousal level in [0, 1] (the settled wake-population activity)."""
        return self.settle(drive, start=start, homeo=homeo)[0]

    def sweep(self, drives, start, homeo=None):
        """Continuation sweep: carry the settled state from one drive to the next (history matters).
        Returns the arousal level at each drive in order."""
        homeo = self.homeo if homeo is None else homeo
        state = self._start(start)
        out = []
        for dr in drives:
            state = self.settle(dr, start=state, homeo=homeo)
            out.append(state[0])
        return out


def _switch_on(drives, arousals, level=0.5):
    """Up-sweep: first drive at which arousal RISES to `level` (asleep -> awake)."""
    for dr, a in zip(drives, arousals):
        if a >= level:
            return dr
    return None


def _switch_off(drives, arousals, level=0.5):
    """Down-sweep (drives descending): first drive at which arousal DROPS below `level`
    (awake -> asleep). This is the switch-off drive, the lower edge of the hysteresis loop."""
    for dr, a in zip(drives, arousals):
        if a < level:
            return dr
    return None


def _frange(lo, hi, step):
    """Inclusive float range (deterministic, integer-counted to avoid drift)."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


def main():
    rf = ReticularFormation(seed=0)

    up_drives = _frange(0.0, DRIVE_MAX, DRIVE_STEP)
    down_drives = list(reversed(up_drives))
    up_arousal = rf.sweep(up_drives, start="sleep")  # asleep -> raise drive
    down_arousal = rf.sweep(down_drives, start="wake")  # awake  -> lower drive

    up_thresh = _switch_on(up_drives, up_arousal)  # asleep -> awake switch point
    down_thresh = _switch_off(down_drives, down_arousal)  # awake  -> asleep switch point

    # two attractors at the extremes (settle from BOTH starts)
    sleep_from_sleep = rf.arousal(0.0, start="sleep")
    sleep_from_wake = rf.arousal(0.0, start="wake")
    wake_from_sleep = rf.arousal(DRIVE_MAX, start="sleep")
    wake_from_wake = rf.arousal(DRIVE_MAX, start="wake")

    # coexisting states at one intermediate drive (history dependence)
    mid_low = rf.arousal(DRIVE_MID, start="sleep")
    mid_high = rf.arousal(DRIVE_MID, start="wake")

    print("=" * 68)
    print("RETICULAR FORMATION (ARAS) -- bistable sleep-wake arousal switch")
    print("=" * 68)
    print(
        f"flip-flop: self-exc {SELF_EXC}, mutual-inhib {MUTUAL}, gain {GAIN}, "
        f"homeostatic pressure {HOMEO}"
    )
    print(f"arousal in [0, 1] read from the wake population; drive swept 0.0 -> {DRIVE_MAX}")
    print("-" * 68)
    print("(a) two attractors (settle from either start):")
    print(
        f"    NO drive   -> arousal {sleep_from_sleep:.2f} (from sleep) / "
        f"{sleep_from_wake:.2f} (from wake)   [SLEEP]"
    )
    print(
        f"    FULL drive -> arousal {wake_from_sleep:.2f} (from sleep) / "
        f"{wake_from_wake:.2f} (from wake)   [WAKE]"
    )
    print("(b) hysteresis loop (sharp bistable transitions):")
    print(f"    UP   switch (asleep -> awake) at drive {up_thresh:.2f}")
    print(f"    DOWN switch (awake -> asleep) at drive {down_thresh:.2f}")
    print(f"    loop width = {up_thresh - down_thresh:.2f}  (up threshold > down threshold)")
    print(
        f"(c) intermediate drive {DRIVE_MID:.2f}: arousal {mid_low:.2f} (from sleep) vs "
        f"{mid_high:.2f} (from wake)  [two stable states]"
    )
    print("=" * 68)

    # ---- self-check: deterministic asserts, ASCII-only ----
    # (a) two attractors: no drive -> SLEEP from any start; full drive -> WAKE from any start.
    assert sleep_from_sleep < 0.3 and sleep_from_wake < 0.3, (
        f"no drive must settle to the sleep attractor ({sleep_from_sleep:.2f}, {sleep_from_wake:.2f})"
    )
    assert wake_from_sleep > 0.7 and wake_from_wake > 0.7, (
        f"full drive must settle to the wake attractor ({wake_from_sleep:.2f}, {wake_from_wake:.2f})"
    )
    # (b) hysteresis: the up switch threshold is strictly ABOVE the down switch threshold.
    assert up_thresh is not None and down_thresh is not None, (
        "both switches must occur in the sweep"
    )
    assert up_thresh > down_thresh, (
        f"hysteresis requires up threshold > down threshold ({up_thresh:.2f} vs {down_thresh:.2f})"
    )
    assert up_thresh - down_thresh > 0.2, (
        f"hysteresis loop must have width ({up_thresh - down_thresh:.2f})"
    )
    # (c) history dependence: two distinct stable states coexist at the intermediate drive.
    assert mid_high - mid_low > 0.5, (
        f"intermediate drive must have two stable states ({mid_low:.2f} vs {mid_high:.2f})"
    )
    assert mid_low < 0.3 < 0.7 < mid_high, "the coexisting states must straddle the switch"
    # determinism: two fresh same-seed instances reproduce identical trajectories op-for-op.
    rf2 = ReticularFormation(seed=0)
    assert rf2.sweep(up_drives, start="sleep") == up_arousal, "same seed -> identical up-sweep"
    a, b = ReticularFormation(seed=0), ReticularFormation(seed=0)
    assert a.arousal(DRIVE_MID, start="wake") == b.arousal(DRIVE_MID, start="wake"), (
        "same seed -> identical settled arousal"
    )

    print(
        f"self-check OK: sleep attractor {sleep_from_sleep:.2f} <-> wake attractor "
        f"{wake_from_wake:.2f}; hysteresis up {up_thresh:.2f} > down {down_thresh:.2f} "
        f"(loop {up_thresh - down_thresh:.2f}); intermediate drive holds two states "
        f"{mid_low:.2f}/{mid_high:.2f}; deterministic"
    )


if __name__ == "__main__":
    main()
