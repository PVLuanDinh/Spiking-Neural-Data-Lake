"""
Allocentric landmark map with path-integration drift correction.

WHAT THE BRAIN DOES. A hippocampal cognitive map is ALLOCENTRIC: it stores where
things are in a world-centred frame (place fields anchored to landmarks), not just
where they are relative to the body (O'Keefe & Nadel, "The Hippocampus as a
Cognitive Map", 1978). Between landmarks the animal navigates by PATH INTEGRATION /
dead reckoning -- summing self-motion to keep a running position estimate (Mittelstaedt
& Mittelstaedt 1980; McNaughton, Battaglia, Jensen, Moser & Moser, Nat Rev Neurosci
7:663, 2006). Dead reckoning DRIFTS: motion-signal error accumulates without bound with
path length. The brain corrects that drift by RE-SIGHTING a known landmark: a familiar
place fixes the true position and RESETS the path integrator (Etienne & Jeffery, Hippocampus
14:180, 2004; Hardcastle, Ganguli & Giocomo, Neuron 86:827, 2015 -- grid code resets at
boundaries; Evans, Bicanski, Bush & Burgess, J Physiol 594:6535, 2016 -- landmark reset of
the path-integration estimate). This is spatial loop-closure: re-observing a mapped landmark
snaps the accumulated drift back out.

THE GAP THIS FILLS (verified against the repo's spatial substrate). grid_cells.py is a
metric 2D path integrator whose own docstring states it does "NO phase drift or drift
correction" -- it integrates a KNOWN velocity exactly and never resets. ring_attractor.py
is a 1D head-direction integrator (heading angle, not place). Neither BINDS named objects to
allocentric places, and neither CORRECTS dead-reckoning drift when a known landmark is
re-sighted. This module adds exactly that missing piece: an allocentric landmark map that
(a) registers objects at world positions, (b) drifts as it dead-reckons self-motion, and
(c) resets that drift on landmark re-sighting -- brain-general spatial cognition, not tied to
any body or sensor.

DISTINCTNESS.
  - grid_cells.py: exact metric path integration, no landmarks, no drift, no correction.
  - ring_attractor.py: 1D heading only, no place, no correction.
  - hippocampus.py: DG/CA3 pattern separation/completion of discrete memories, not a metric map.
This module stays ZERO-COUPLING: it keeps its own internal [x, y] estimate rather than driving
grid_cells (whose step() integrates a clean velocity with no drift hook), so the drift and its
correction are explicit and testable. Conceptually the estimate is the read-out of the entorhinal
metric that this allocentric map is anchored onto.

Deterministic (random.Random(seed)). ASCII output.

  python -m snn_data_lake.hippocampus.allocentric_map     # (also run by run_selfchecks.py)
"""

import math
import random

# landmark record layout: name -> [x, y, valence, confidence, last_step]
_X, _Y, _VAL, _CONF, _STEP = 0, 1, 2, 3, 4


class AllocentricMap:
    """World-centred landmark map that dead-reckons self-motion and resets drift on re-sighting.

    Usage:
        m = AllocentricMap(seed=0)
        m.observe("door", 0.0, 0.0)      # register the door at the current estimate
        m.move(1.0, 0.0)                 # walk east; the estimate drifts a little
        ...                              # walk a loop back toward the door
        m.observe("door", rel_x, rel_y)  # re-sight it -> estimate snaps back to truth
        m.nearest(valence_pref=1.0)      # recall the best remembered place (approach good ones)
    """

    def __init__(
        self,
        seed=0,
        drift_rate=0.06,
        drift_bias=(0.7, 0.5),
        drift_jitter=0.4,
        correct_gain=0.8,
        conf_gain=0.25,
    ):
        """
        Args:
            seed: RNG seed -> reproducible drift.
            drift_rate: dead-reckoning error per unit distance travelled.
            drift_bias: systematic error direction (like a gyro/odometry bias).
            drift_jitter: random component added to the bias each step.
            correct_gain: fraction of the re-sighting discrepancy applied to the
                estimate (in (0, 1]; >0.5 makes a re-sight more than halve the drift).
            conf_gain: confidence increment per re-sighting.
        """
        self.rng = random.Random(seed)
        self.drift_rate = float(drift_rate)
        self.drift_bias = (float(drift_bias[0]), float(drift_bias[1]))
        self.drift_jitter = float(drift_jitter)
        self.correct_gain = float(correct_gain)
        self.conf_gain = float(conf_gain)

        self.est = [0.0, 0.0]  # allocentric position estimate (drifts)
        self.drift = [0.0, 0.0]  # accumulated dead-reckoning error (introspection only)
        self._step = 0  # internal time counter
        self.landmarks = {}  # name -> [x, y, valence, confidence, last_step]

    @property
    def position(self):
        """Current allocentric position estimate as a tuple."""
        return (self.est[0], self.est[1])

    def move(self, dx, dy):
        """Integrate a self-motion step (dx, dy) into the estimate, accumulating drift.

        Dead reckoning: the commanded motion is added to the estimate together with a
        small error that grows with the distance travelled -- so the raw estimate wanders
        away from the true position, exactly as an unaided path integrator does.
        """
        self._step += 1
        dist = math.hypot(dx, dy)
        noise = self.rng.uniform(-self.drift_jitter, self.drift_jitter)
        gx = self.drift_rate * (self.drift_bias[0] + noise) * dist
        gy = self.drift_rate * (self.drift_bias[1] + noise) * dist
        self.drift[0] += gx
        self.drift[1] += gy
        self.est[0] += dx + gx
        self.est[1] += dy + gy
        return self.position

    def observe(self, name, rel_x, rel_y, valence=0.0):
        """See a landmark at relative offset (rel_x, rel_y) from the agent.

        NEW landmark  -> register its allocentric position = estimate + offset.
        KNOWN landmark -> the offset points from the (drifted) estimate to where the
        landmark is EXPECTED; the stored landmark says where it TRULY is. The gap between
        (estimate + offset) and the stored position is our accumulated drift. We know our
        true position must be (stored_landmark - offset), so we CORRECT the estimate toward
        it by correct_gain (path-integration reset / loop-closure) and raise the landmark's
        confidence. The stored landmark stays fixed -- it is the trusted anchor.
        """
        self._step += 1
        if name not in self.landmarks:  # register a new place
            self.landmarks[name] = [
                self.est[0] + rel_x,
                self.est[1] + rel_y,
                float(valence),
                0.5,
                self._step,
            ]
            return self.position

        lm = self.landmarks[name]  # re-sight a known place -> drift reset
        target_x = lm[_X] - rel_x  # where the agent must actually be
        target_y = lm[_Y] - rel_y
        err_x = target_x - self.est[0]
        err_y = target_y - self.est[1]
        self.est[0] += self.correct_gain * err_x
        self.est[1] += self.correct_gain * err_y
        self.drift[0] *= 1.0 - self.correct_gain  # drift is largely reset
        self.drift[1] *= 1.0 - self.correct_gain
        lm[_CONF] = min(1.0, lm[_CONF] + self.conf_gain)  # familiarity grows on re-sighting
        lm[_STEP] = self._step
        if valence != 0.0:
            lm[_VAL] = 0.7 * float(valence) + 0.3 * lm[_VAL]
        return self.position

    def nearest(self, valence_pref=0.0):
        """Recall the highest-confidence landmark, optionally biased by valence.

        Score = confidence + valence_pref * valence. With valence_pref = 0 this returns the
        best-remembered place; with valence_pref > 0 it prefers positive (approach) places,
        with valence_pref < 0 it prefers to recall aversive places (avoidance). Returns the
        landmark name, or None if the map is empty.
        """
        best_name, best_score = None, None
        for name, lm in self.landmarks.items():
            score = lm[_CONF] + valence_pref * lm[_VAL]
            if best_score is None or score > best_score:
                best_score, best_name = score, name
        return best_name


def main():
    # ---- (a) walk a closed loop with per-step drift; raw dead reckoning ends far off ----
    m = AllocentricMap(seed=0)
    truth = [0.0, 0.0]
    home_true = (0.0, 0.0)
    m.observe("home", 0.0, 0.0)  # anchor home while est == truth
    conf_before = m.landmarks["home"][_CONF]

    step = 0.5
    legs = [(step, 0.0), (0.0, step), (-step, 0.0), (0.0, -step)]  # square loop, side 3.0
    for vx, vy in legs:
        for _ in range(6):
            truth[0] += vx
            truth[1] += vy
            m.move(vx, vy)
    uncorrected_error = math.hypot(m.est[0] - truth[0], m.est[1] - truth[1])

    # ---- (b) re-sight the known 'home' landmark -> path-integration reset corrects drift ----
    rel_x = home_true[0] - truth[0]  # where the agent sees home, from truth
    rel_y = home_true[1] - truth[1]
    m.observe("home", rel_x, rel_y)
    corrected_error = math.hypot(m.est[0] - truth[0], m.est[1] - truth[1])
    conf_after = m.landmarks["home"][_CONF]

    # ---- (d) determinism: same seed + same ops -> identical estimate and landmarks ----
    def run_loop(seed):
        mm = AllocentricMap(seed=seed)
        mm.observe("home", 0.0, 0.0)
        for vx, vy in legs:
            for _ in range(6):
                mm.move(vx, vy)
        mm.observe("home", 0.0, 0.0)
        return mm

    ra, rb = run_loop(3), run_loop(3)
    deterministic = ra.est == rb.est and ra.landmarks == rb.landmarks

    # ---- nearest(): recall by confidence, and biased by valence (approach good places) ----
    r = AllocentricMap(seed=1)
    r.observe("berries", 1.0, 0.0, valence=1.0)  # a good place
    r.observe("cliff", -1.0, 0.0, valence=-1.0)  # a bad place
    r.observe("cliff", -1.0, 0.0, valence=-1.0)  # re-sighted -> more confident
    recall_conf = r.nearest(valence_pref=0.0)  # most-confident place
    recall_good = r.nearest(valence_pref=2.0)  # approach the good place

    print("=" * 68)
    print("ALLOCENTRIC MAP -- landmark binding + path-integration drift reset")
    print("=" * 68)
    print(
        f"loop truth returns to ({truth[0]:.2f},{truth[1]:.2f}); "
        f"raw estimate ({m.est[0]:.2f},{m.est[1]:.2f}) before correction"
    )
    print(f"dead-reckoning drift (uncorrected error) : {uncorrected_error:.3f}")
    print(
        f"after re-sighting 'home' (corrected error): {corrected_error:.3f} "
        f"(ratio {corrected_error / uncorrected_error:.2f})"
    )
    print(f"'home' confidence {conf_before:.2f} -> {conf_after:.2f} on re-sighting")
    print(
        f"nearest by confidence -> '{recall_conf}' ; nearest with approach bias -> '{recall_good}'"
    )
    print(f"determinism (seed 3): identical estimate and landmarks {deterministic}")
    print("=" * 68)

    # ---- self-check: deterministic asserts, un-riggable ----
    assert uncorrected_error > 0.3, (
        f"dead reckoning must actually drift far from truth ({uncorrected_error:.3f})"
    )
    assert corrected_error < 0.5 * uncorrected_error, (
        f"re-sighting must more than halve the drift ({corrected_error:.3f} vs {uncorrected_error:.3f})"
    )
    assert conf_after > conf_before, (
        f"a landmark's confidence must rise on re-sighting ({conf_before:.2f} -> {conf_after:.2f})"
    )
    assert recall_conf == "cliff", (
        f"nearest by confidence must recall the most-confident place ({recall_conf})"
    )
    assert recall_good == "berries", (
        f"approach bias must recall the positive-valence place ({recall_good})"
    )
    assert deterministic, (
        "same seed + same ops must give identical estimate and landmarks (deterministic)"
    )
    print(
        f"self-check OK: dead-reckoning drift {uncorrected_error:.3f} -> {corrected_error:.3f} on "
        f"landmark re-sighting (path-integration reset); confidence rises; nearest recall by "
        f"confidence/valence; deterministic"
    )


if __name__ == "__main__":
    main()
