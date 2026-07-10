"""
Superior colliculus DEEP layers -- turn the selected orienting winner into a gaze motor command.

The SUPERFICIAL colliculus (superior_colliculus.SuperiorColliculus) holds a retinotopic salience
map and, by lateral-inhibition competition + build-to-threshold WTA, commits to ONE target
location. That is a SELECTION, not yet a movement. The DEEP collicular layers convert that
selected location (a "motor error" on the retinotopic map) into a calibrated, gated gaze
displacement via the classic saccadic burst generator + neural integrator local feedback loop.

Mechanism (rate/kinematic level, deterministic):
  1. Motor error: the selected target's location minus current fixation -> how far gaze must move.
  2. Omnipause gate (OPN): while a fixation/pause signal is HIGH, the omnipause neurons clamp the
     excitatory burst neurons -- NO burst forms, so NO saccade. A gaze command forms ONLY once
     fixation releases. This is the go/no-go gate on the movement, distinct from target selection.
  3. Burst generator (EBN): released from OPN, the burst neurons fire a pulse whose rate tracks the
     remaining motor error against an efference-copy reference (a resettable local feedback loop):
        ref[t] = ref[t-1] + k*(error - ref[t-1]);  burst[t] = k*(error - ref[t-1]).
     The pulse is front-loaded: peak burst = k*error at onset, decaying as the error is cancelled.
  4. Neural integrator (NI): a resettable LEAKY integrator sums the burst into eye displacement:
        e[t] = e[t-1]*(1 - leak) + gain*burst[t].
     The time-INTEGRAL of the burst is the gaze displacement. A leaky integrator loses early burst
     mass, so the integrator gain is CALIBRATED once (gain = 1 / unit-burst response) to make the
     emitted saccade amplitude land exactly on the commanded motor error (accurate saccade). An
     uncalibrated unit-gain leaky integrator UNDERSHOOTS.
  5. Main sequence: since peak burst = k*error, peak eye velocity rises with saccade amplitude
     (larger saccades are faster), the pulse-vs-amplitude signature of real saccades.

Fair control (un-riggable):
  - OMNIPAUSE gate: with the fixation/pause signal held HIGH, no burst and zero displacement are
    produced no matter how large the motor error; releasing it yields the saccade.
  - CALIBRATION: the calibrated gain lands the amplitude on the commanded motor error to floating
    precision, while the SAME burst through an uncalibrated (unit-gain) leaky integrator undershoots.
  - MAIN SEQUENCE: sweeping amplitude upward yields strictly increasing peak velocity.
  - SELECTION coupling: on a two-target map SuperiorColliculus picks the WTA winner and this deep
    layer aims the gaze command at THAT winner's location; a flat/zero map selects nothing, so
    there is nothing to move to -- NO gaze command is emitted.

Honest framing: a reduced rate/kinematic model of the collicular deep-layer burst generator +
neural integrator local feedback loop and the main-sequence relationship -- NOT brainstem
saccade-circuit biophysics (EBN/IBN/OPN membrane dynamics, motoneuron pools). It models the
FUNCTION (selected target -> calibrated, gated gaze command) and explicitly DOES NOT ACTUATE:
the command is returned as DATA. In the full architecture a gaze command reaches an effector ONLY
through the gated brainstem final common path; this module imports nothing from brainstem/ and
drives no hardware.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.midbrain.sc_deep_layers
"""
import random

from snn_data_lake.midbrain.superior_colliculus import SuperiorColliculus


class SCDeepLayers:
    """Collicular deep-layer saccadic burst generator + calibrated leaky neural integrator.

    command(motor_error) turns a commanded motor error into a gated, calibrated gaze displacement
    plus a velocity profile; gaze_from_map(salience) chains the SUPERFICIAL WTA selection into a
    deep-layer gaze command. The command is returned as DATA -- this module never actuates."""

    def __init__(self, n=24, k=0.4, leak=0.03, rel_frac=1e-3, vel_thresh=0.05, noise=0.0, seed=0):
        self.n = n
        self.k = k                    # EBN local-feedback gain (fraction of motor error per step)
        self.leak = leak              # neural-integrator membrane leak (>0 => undershoot if unfixed)
        self.vel_thresh = vel_thresh  # velocity floor for the reported main-sequence "duration"
        self.noise = noise            # optional motor jitter on the burst (0 == deterministic)
        self.rng = random.Random(seed)
        self.sc = SuperiorColliculus(n=n, seed=seed)   # the superficial WTA selection layer (reused)

        # burst length: run the pulse until the relative motor error decays below rel_frac. Because
        # burst[t]/peak = (1-k)^(t-1) is independent of the amplitude, this length is CONSTANT across
        # saccades, which keeps the leaky-integrator response exactly linear in amplitude.
        self.n_steps = 1
        while (1.0 - self.k) ** self.n_steps >= rel_frac:
            self.n_steps += 1

        # calibrate the integrator gain: emit a unit burst, measure the leaky-integrated response,
        # and set gain so the calibrated amplitude equals the commanded motor error (accurate saccade).
        raw, _ = self._integrate(self._burst(1.0, False, noisy=False), 1.0)
        self.gain = 1.0 / raw if raw else 1.0

    def _burst(self, motor_error, fixation_high, noisy=True):
        """EBN burst: a pulse whose rate tracks the remaining motor error via a resettable
        efference-copy reference. Omnipause (fixation_high) clamps the burst to zero -- no saccade."""
        burst = []
        ref = 0.0
        for _ in range(self.n_steps):
            if fixation_high or motor_error == 0.0:
                burst.append(0.0)                     # OPN gate open / nothing to move to -> no burst
                continue
            drive = self.k * (motor_error - ref)      # proportional burst against remaining error
            ref += drive                              # clean efference-copy reference (resettable)
            sample = drive
            if noisy and self.noise:
                sample += self.rng.uniform(-self.noise, self.noise)
            burst.append(sample)
        return burst

    def _integrate(self, burst, gain):
        """Resettable LEAKY neural integrator: sum the burst into eye displacement. Returns the final
        displacement and the per-step gaze velocity profile (increments of the integrator output)."""
        e = 0.0
        vel = []
        for b in burst:
            e_new = e * (1.0 - self.leak) + gain * b
            vel.append(e_new - e)
            e = e_new
        return e, vel

    def command(self, motor_error, fixation_high=False):
        """Convert a commanded motor error into a gated, calibrated gaze command (returned as DATA).

        Returns the emitted displacement (== motor_error when the OPN gate is open and gain is
        calibrated; 0 while fixation is held), the velocity profile, its peak, and a main-sequence
        duration. Nothing here is sent to any effector."""
        burst = self._burst(float(motor_error), fixation_high)
        displacement, vel = self._integrate(burst, self.gain)
        peak = max((abs(v) for v in vel), default=0.0)
        duration = sum(1 for v in vel if abs(v) >= self.vel_thresh)
        return {
            "motor_error": float(motor_error),
            "displacement": displacement,
            "velocity": vel,
            "peak_velocity": peak,
            "duration": duration,
            "gated": bool(fixation_high),
        }

    def gaze_from_map(self, salience, fixation=0, fixation_high=False):
        """Chain SUPERFICIAL selection -> DEEP gaze command. SuperiorColliculus picks the WTA winner;
        the deep layers aim the gaze at THAT location. A flat/zero map selects nothing -> NO command
        (nothing to move to). The returned gaze command is DATA and is not actuated."""
        sel = self.sc.saccade(salience)
        target = sel["target"]
        if target is None:
            return {
                "target": None,
                "motor_error": None,
                "displacement": 0.0,
                "landing": float(fixation),
                "velocity": [],
                "peak_velocity": 0.0,
                "duration": 0,
                "gated": bool(fixation_high),
            }
        motor_error = target - fixation
        cmd = self.command(motor_error, fixation_high=fixation_high)
        return {
            "target": target,
            "motor_error": float(motor_error),
            "displacement": cmd["displacement"],
            "landing": fixation + cmd["displacement"],
            "velocity": cmd["velocity"],
            "peak_velocity": cmd["peak_velocity"],
            "duration": cmd["duration"],
            "gated": cmd["gated"],
        }


def main():
    dl = SCDeepLayers(seed=0)

    # omnipause gate: same motor error, fixation held vs released
    gated = dl.command(8.0, fixation_high=True)
    released = dl.command(8.0, fixation_high=False)

    # integrator accuracy across amplitudes (calibrated gain lands on target)
    accur = [(a, dl.command(a)["displacement"]) for a in (3.0, 8.0, 15.0)]
    # the SAME 8.0 burst through an uncalibrated unit-gain leaky integrator undershoots
    raw_undershoot, _ = dl._integrate(dl._burst(8.0, False, noisy=False), 1.0)

    # main sequence: peak velocity vs amplitude
    amps = [2.0, 5.0, 9.0, 14.0, 20.0]
    peaks = [dl.command(a)["peak_velocity"] for a in amps]

    # selection coupling: two-target map -> WTA winner -> gaze aims there; flat map -> no command
    two = dl.sc.two_target_map(loc_strong=4, s_strong=0.9, loc_weak=17, s_weak=0.6)
    g_two = dl.gaze_from_map(two, fixation=0)
    flat = [0.5] * dl.sc.n
    g_flat = dl.gaze_from_map(flat, fixation=0)

    print("=" * 70)
    print("SUPERIOR COLLICULUS DEEP LAYERS -- selected target -> gated gaze command")
    print("=" * 70)
    print(f"burst steps: {dl.n_steps} | feedback k: {dl.k} | integrator leak: {dl.leak} | "
          f"calibrated gain: {dl.gain:.3f}")
    print("-" * 70)
    print("omnipause gate (motor error 8.0):")
    print(f"  fixation HELD    -> displacement {gated['displacement']:.3f}  "
          f"(OPN clamps burst -> no saccade)")
    print(f"  fixation RELEASED-> displacement {released['displacement']:.3f}  "
          f"peak vel {released['peak_velocity']:.3f}")
    print("-" * 70)
    print("integrator accuracy (calibrated -> amplitude == commanded motor error):")
    for a, d in accur:
        print(f"  commanded {a:5.2f} -> emitted {d:.4f}")
    print(f"  uncalibrated unit-gain leaky integrator on 8.0 -> {raw_undershoot:.4f} (UNDERSHOOT)")
    print("-" * 70)
    print("main sequence (peak velocity rises with amplitude):")
    for a, p in zip(amps, peaks):
        print(f"  amplitude {a:5.2f} -> peak velocity {p:.3f}")
    print("-" * 70)
    print("selection coupling (superficial WTA -> deep gaze command):")
    print(f"  two targets (strong@4, weak@17): winner loc {g_two['target']} -> "
          f"gaze lands at {g_two['landing']:.3f}")
    print(f"  flat map (all 0.50)            : winner {g_flat['target']} -> "
          f"displacement {g_flat['displacement']:.3f} (nothing selected -> no command)")
    print("=" * 70)

    # ---- un-riggable self-checks (deterministic asserts) --------------------
    tol = 1e-6
    # (1) omnipause gate: held fixation emits NO saccade; release emits it.
    assert gated["displacement"] == 0.0 and gated["peak_velocity"] == 0.0, \
        "omnipause must suppress the saccade while fixation is held"
    assert abs(released["displacement"] - 8.0) < tol, \
        "releasing fixation must produce the calibrated saccade"
    # (2) integrator accuracy: calibrated amplitude lands on the commanded motor error; a
    #     wrong-gain (unit-gain) leaky integrator undershoots.
    for a, d in accur:
        assert abs(d - a) < tol, "calibrated gain must land the saccade on the commanded motor error"
    assert raw_undershoot < 8.0 - 0.1, \
        "an uncalibrated unit-gain leaky integrator must undershoot the target"
    # (3) main sequence: peak velocity strictly increases with amplitude.
    assert all(b > a for a, b in zip(peaks, peaks[1:])), \
        "peak velocity must increase monotonically with saccade amplitude"
    # (4) selection coupling: gaze aims at the WTA winner; flat map yields no gaze command.
    assert g_two["target"] == 4, "the deep layers must aim at the superficial WTA winner"
    assert round(g_two["landing"]) == 4, "the gaze command must point to the winner's location"
    assert g_flat["target"] is None and g_flat["displacement"] == 0.0, \
        "a flat map selects nothing -> no gaze command is emitted"
    # Determinism: same seed -> identical noisy gaze command.
    p = SCDeepLayers(seed=0, noise=0.3)
    q = SCDeepLayers(seed=0, noise=0.3)
    assert p.command(6.0) == q.command(6.0), "same seed must reproduce an identical gaze command"

    print(f"self-check OK: omnipause gate holds ({gated['displacement']:.1f}) then releases "
          f"({released['displacement']:.1f}); calibrated amplitude == motor error "
          f"(unit-gain undershoots to {raw_undershoot:.2f});")
    print(f"  peak velocity {peaks[0]:.2f}->{peaks[-1]:.2f} strictly rises with amplitude "
          f"{amps[0]:.0f}->{amps[-1]:.0f}; gaze aims at WTA winner loc {g_two['target']}, "
          "flat map -> no command; deterministic")


if __name__ == "__main__":
    main()
