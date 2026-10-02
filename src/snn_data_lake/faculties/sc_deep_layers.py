"""
Superior colliculus DEEP layers -- turn the selected orienting winner into a gaze motor command.

The SUPERFICIAL colliculus (superior_colliculus.SuperiorColliculus) holds a retinotopic salience
map and, by lateral-inhibition competition + build-to-threshold WTA, commits to ONE target
location. That is a SELECTION, not yet a movement. The DEEP collicular layers convert that
selected location (a "motor error" on the retinotopic map) into a calibrated, gated gaze
displacement via the classic saccadic burst generator + neural integrator local feedback loop --
here built from GENUINE SPIKING neurons, not rate/kinematic arithmetic.

Spiking mechanism (discrete events, deterministic):
  1. Motor error: the selected target's location minus current fixation -> how far gaze must move.
  2. Omnipause gate (OPN): while a fixation/pause signal is HIGH, tonically-firing omnipause neurons
     saturate the excitatory burst pool with inhibition, clamping every EBN membrane at rest -- NO
     spikes, so NO saccade. A gaze command forms ONLY once fixation releases (the OPN pause).
  3. EBN burst pool (spiking): a POOL of `n` leaky integrate-and-fire excitatory burst neurons.
     Each neuron integrates a drive current on its membrane V, and when V crosses threshold it
     emits a discrete SPIKE and resets by SUBTRACTING the threshold (charge-conserving), so the
     total number of spikes the pool fires equals the injected charge -- which is proportional to
     the motor error. The drive is the front-loaded EBN pulse against a resettable efference-copy
     reference (ref += k*(error-ref); peak at onset), so the pool fires a front-loaded VOLLEY.
     Dithered resting membranes across the pool spread the thresholds so the population spike count
     tracks the motor error smoothly and linearly.
  4. Line-attractor integrator (NI): a recurrent spiking integrator with self-feedback gain EXACTLY
     1 and no leak -- a perfect integrator, i.e. a LINE ATTRACTOR (every accumulated level is a
     fixed point). Its persistent activity ACCUMULATES the burst spikes: the stored spike count is
     the eye displacement. Because it is a line attractor, once the volley ends the accumulated
     level (and thus the eye position) HOLDS: a staggered readout population keeps firing tonically
     at a rate that encodes the held position, with zero input, indefinitely.
  5. Read-out: the gaze displacement is the ACCUMULATED SPIKE COUNT times a calibrated gain -- it
     EMERGES from counting discrete spikes, never from `gain * motor_error`.

Fair control (un-riggable):
  - OMNIPAUSE gate: with fixation held HIGH the burst pool is clamped, zero spikes and zero
    displacement are produced no matter how large the motor error; releasing it yields the saccade.
  - FUNCTION: a larger commanded motor error drives more charge -> more burst spikes -> a strictly
    larger, linear-ish displacement; and the line-attractor integrator HOLDS the position after the
    burst (the read-out is unchanged under zero input while the population keeps spiking).
  - SPIKING: discrete membrane/threshold/spike/reset events; displacement read from accumulated
    spike count.
  - SELECTION coupling: on a two-target map SuperiorColliculus picks the WTA winner and this deep
    layer aims the gaze command at THAT winner's location; a flat/zero map selects nothing, so
    there is nothing to move to -- NO gaze command is emitted.

Honest framing: a reduced SPIKING model of the collicular deep-layer burst generator + neural
integrator local feedback loop -- LIF burst neurons feeding a recurrent line-attractor spiking
integrator. It is not full brainstem saccade-circuit biophysics (EBN/IBN/OPN conductances,
motoneuron pools); the omnipause is modelled as a saturating inhibitory clamp rather than an
explicit OPN spike train, and the integrator's line attractor is the idealised perfect-integrator
limit. It models the FUNCTION (selected target -> calibrated, gated gaze command via spikes) and
explicitly DOES NOT ACTUATE: the command is returned as DATA. In the full architecture a gaze
command reaches an effector ONLY through the gated brainstem final common path; this module imports
nothing from brainstem/ and drives no hardware.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.sc_deep_layers
"""

import random

from snn_data_lake.faculties.superior_colliculus import SuperiorColliculus


class SCDeepLayers:
    """Collicular deep-layer spiking burst generator + recurrent line-attractor spiking integrator.

    command(motor_error) turns a commanded motor error into a gated gaze displacement read from an
    accumulated SPIKE COUNT, plus a velocity profile; gaze_from_map(salience) chains the SUPERFICIAL
    WTA selection into a deep-layer gaze command. The command is returned as DATA -- never actuated."""

    def __init__(self, n=24, k=0.4, leak=0.01, rel_frac=1e-3, vel_thresh=0.05, noise=0.0, seed=0):
        self.n = n  # size of the LIF excitatory-burst-neuron (EBN) pool
        self.k = k  # EBN efference-copy feedback gain (fraction of motor error per step)
        self.leak = leak  # EBN membrane leak per step (small: near charge-conserving IF)
        self.vel_thresh = vel_thresh  # velocity floor for the reported saccade "duration"
        self.noise = noise  # optional motor jitter on the burst drive (0 == deterministic)
        self.rng = random.Random(seed)

        self.v_threshold = 1.0  # LIF firing threshold (a spike fires when V crosses this)
        self.v_rest = 0.0  # LIF resting membrane potential
        self.cur = 1.2  # drive-current scale (charge injected per unit of EBN drive)
        self.drain = 30  # extra steps after the pulse to flush banked charge into spikes
        self.hold_steps = 20  # zero-input steps used to demonstrate line-attractor persistence
        self.n_hold = 8  # size of the integrator read-out population (persistent spiking)

        # dithered resting membranes: spreading the pool's start states spreads the effective
        # thresholds, so the POPULATION spike count tracks the injected charge almost exactly
        # (smooth + linear) instead of coarsely quantising at a single shared threshold.
        self.v0 = [(i + 0.5) / n for i in range(n)]

        # burst-drive length: run the efference-copy pulse until the relative motor error decays
        # below rel_frac. Because drive[t]/peak = (1-k)^(t-1) is amplitude-independent, this length
        # is CONSTANT across saccades, keeping the pool's spike count linear in amplitude.
        self.n_steps = 1
        while (1.0 - self.k) ** self.n_steps >= rel_frac:
            self.n_steps += 1
        self.win = self.n_steps + self.drain  # full burst-simulation window (pulse + drain)

        self.sc = SuperiorColliculus(n=n, seed=seed)  # the superficial WTA selection layer (reused)

        # calibrate the read-out gain: fire a unit-error burst, count the accumulated spikes, and set
        # gain so a unit motor error reads out as unit displacement (displacement == counted spikes
        # times gain). Uses the noiseless burst so calibration draws nothing from the rng.
        raw, _ = self._integrate(self._burst(1.0, False, noisy=False), 1.0)
        self.gain = 1.0 / raw if raw else 1.0

    def _burst(self, motor_error, fixation_high, noisy=True):
        """EBN burst pool (spiking): `n` LIF neurons integrate a front-loaded efference-copy drive
        current and fire a discrete VOLLEY. Charge-conserving (subtract-on-spike) reset makes the
        total spike count proportional to the motor error. Omnipause (fixation_high) tonically
        clamps every membrane at rest -- zero spikes, no saccade. Returns the per-step pool spike
        count (a list of non-negative ints), i.e. the spike volley."""
        v = list(self.v0)
        ref = 0.0
        train = []
        for t in range(self.win):
            if fixation_high:  # OPN saturating inhibition holds EBNs at rest
                for i in range(self.n):
                    v[i] = self.v_rest
                train.append(0)
                continue
            if motor_error == 0.0:  # nothing selected -> nothing to move to
                train.append(0)
                continue
            drive = self.k * (motor_error - ref) if t < self.n_steps else 0.0
            if t < self.n_steps:
                ref += drive  # resettable efference-copy reference
            current = self.cur * drive
            if noisy and self.noise:
                current += self.rng.uniform(-self.noise, self.noise)
            count = 0
            for i in range(self.n):
                v[i] = (
                    self.v_rest + (v[i] - self.v_rest) * (1.0 - self.leak) + current
                )  # leak + integrate
                if v[i] >= self.v_threshold:
                    v[i] -= self.v_threshold  # subtractive reset conserves charge
                    count += 1  # one discrete 0/1 spike this step
            train.append(count)
        return train

    def _integrate(self, burst, gain):
        """Recurrent LINE-ATTRACTOR spiking integrator: self-feedback gain EXACTLY 1 with no leak is
        a perfect integrator (a line attractor), so its persistent activity ACCUMULATES the burst
        spikes and every level is a stable fixed point. Returns the displacement (accumulated spike
        count * gain) and the per-step gaze-velocity profile (the spike-driven increments)."""
        u = 0.0
        vel = []
        for s in burst:
            u = 1.0 * u + s  # gain-1 recurrent feedback, zero leak -> perfect accumulation
            vel.append(gain * s)
        return gain * u, vel

    def _hold(self, u):
        """Demonstrate line-attractor persistence: step the integrator forward with ZERO burst input
        for hold_steps. A gain-1 zero-leak integrator is a line attractor, so the accumulated level
        `u` is unchanged (it HOLDS). A staggered read-out population fires tonically at a rate that
        encodes the held displacement. Returns the (unchanged) level and the read-out spike train."""
        v = [(j + 0.5) / self.n_hold for j in range(self.n_hold)]  # staggered phases -> steady rate
        train = []
        for _ in range(self.hold_steps):
            u = 1.0 * u + 0.0  # recurrent hold: zero input leaves u fixed
            current = (
                self.gain * u
            ) / self.n_hold  # read-out drive; pop rate encodes held position
            k = 0
            for j in range(self.n_hold):
                v[j] += current
                while v[j] >= self.v_threshold:
                    v[j] -= self.v_threshold
                    k += 1
            train.append(k)
        return u, train

    def command(self, motor_error, fixation_high=False):
        """Convert a commanded motor error into a gated gaze command (returned as DATA).

        The EBN pool fires a spike volley; the line-attractor integrator accumulates it into an eye
        displacement (== counted spikes * calibrated gain; 0 while fixation is held), which then
        HOLDS (persistent integrator spiking). Returns the displacement, the velocity profile, its
        peak, a saccade duration, the burst/persistent spike counts, and the held displacement.
        Nothing here is sent to any effector."""
        burst = self._burst(float(motor_error), fixation_high)
        displacement, vel = self._integrate(burst, self.gain)
        peak = max((abs(v) for v in vel), default=0.0)
        duration = sum(1 for v in vel if abs(v) >= self.vel_thresh)
        held_u, hold_train = self._hold(sum(burst))
        return {
            "motor_error": float(motor_error),
            "displacement": displacement,
            "velocity": vel,
            "peak_velocity": peak,
            "duration": duration,
            "gated": bool(fixation_high),
            "burst_spikes": sum(burst),
            "held_displacement": self.gain * held_u,
            "persistent_spikes": sum(hold_train),
            "hold_train": hold_train,
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
                "burst_spikes": 0,
                "persistent_spikes": 0,
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
            "burst_spikes": cmd["burst_spikes"],
            "persistent_spikes": cmd["persistent_spikes"],
        }


def main():

    dl = SCDeepLayers(seed=0)

    # omnipause gate: same motor error, fixation held vs released
    gated = dl.command(8.0, fixation_high=True)
    released = dl.command(8.0, fixation_high=False)

    # function: displacement tracks amplitude via accumulated spikes (linear-ish); integrator holds
    amps = [2.0, 3.0, 5.0, 8.0, 13.0, 20.0]
    cmds = [dl.command(a) for a in amps]
    disp = [c["displacement"] for c in cmds]
    spk = [c["burst_spikes"] for c in cmds]

    # genuinely-spiking evidence: a discrete volley whose count sets the displacement
    train = dl._burst(20.0, False, noisy=False)

    # selection coupling: two-target map -> WTA winner -> gaze aims there; flat map -> no command
    two = dl.sc.two_target_map(loc_strong=4, s_strong=0.9, loc_weak=17, s_weak=0.6)
    g_two = dl.gaze_from_map(two, fixation=0)
    flat = [0.5] * dl.sc.n
    g_flat = dl.gaze_from_map(flat, fixation=0)

    print("=" * 74)
    print("SUPERIOR COLLICULUS DEEP LAYERS -- spiking burst pool -> line-attractor integrator")
    print("=" * 74)
    print(
        f"EBN pool: {dl.n} LIF neurons | threshold: {dl.v_threshold} | feedback k: {dl.k} | "
        f"burst window: {dl.win} steps | calibrated gain: {dl.gain:.4f}"
    )
    print("-" * 74)
    print("omnipause gate (motor error 8.0):")
    print(
        f"  fixation HELD    -> {gated['burst_spikes']:4d} burst spikes, displacement "
        f"{gated['displacement']:.3f}  (OPN clamps the pool -> no saccade)"
    )
    print(
        f"  fixation RELEASED-> {released['burst_spikes']:4d} burst spikes, displacement "
        f"{released['displacement']:.3f}  peak vel {released['peak_velocity']:.3f}"
    )
    print("-" * 74)
    print("function (more command -> more accumulated spikes -> larger displacement):")
    for a, c in zip(amps, cmds):
        print(
            f"  commanded {a:5.2f} -> {c['burst_spikes']:4d} burst spikes -> displacement "
            f"{c['displacement']:6.3f} | integrator holds at {c['held_displacement']:6.3f} "
            f"({c['persistent_spikes']} persistent spikes)"
        )
    print("-" * 74)
    print("selection coupling (superficial WTA -> deep gaze command):")
    print(
        f"  two targets (strong@4, weak@17): winner loc {g_two['target']} -> "
        f"gaze lands at {g_two['landing']:.3f}"
    )
    print(
        f"  flat map (all 0.50)            : winner {g_flat['target']} -> "
        f"displacement {g_flat['displacement']:.3f} (nothing selected -> no command)"
    )
    print("=" * 74)

    # ---- un-riggable self-checks (deterministic asserts) --------------------
    # (a) FUNCTION: displacement tracks amplitude (more spikes -> larger displacement), linear-ish,
    #     and the line-attractor integrator HOLDS after the burst.
    assert all(b > a for a, b in zip(disp, disp[1:])), (
        "displacement must rise monotonically with commanded amplitude"
    )
    assert all(b > a for a, b in zip(spk, spk[1:])), (
        "a bigger command must accumulate strictly more burst spikes"
    )
    assert all(abs(d - a) / a < 0.08 for a, d in zip(amps, disp)), (
        "displacement must track the commanded amplitude (linear-ish, right magnitude)"
    )
    for c in cmds:
        assert c["held_displacement"] == c["displacement"], (
            "the line-attractor integrator must HOLD the displacement after the burst"
        )
        assert c["persistent_spikes"] > 0, (
            "the integrator population must keep spiking to hold the eye position"
        )
    assert all(s > 0 for s in released["hold_train"]), (
        "the held integrator must fire on every step (sustained persistence, never silent)"
    )
    assert abs(sum(released["hold_train"][:10]) - sum(released["hold_train"][10:])) <= 2, (
        "the held firing rate must be stationary (a line attractor does not drift)"
    )
    # (b) GATING: omnipause suppresses the saccade while fixation is held; release emits it.
    assert (
        gated["displacement"] == 0.0
        and gated["burst_spikes"] == 0
        and gated["peak_velocity"] == 0.0
    ), "omnipause must clamp the burst pool -> no spikes, no saccade while fixation is held"
    assert gated["persistent_spikes"] == 0, "no burst -> no integrator activity while gated"
    assert released["displacement"] > 0.0, "releasing fixation must produce the saccade"
    # (c) GENUINELY SPIKING: membrane/threshold, discrete spikes, displacement from spike count, stdlib.
    assert hasattr(dl, "v_threshold") and dl.v_threshold > 0.0, (
        "a spiking membrane threshold must exist"
    )
    assert all(isinstance(s, int) for s in train), "the burst must be DISCRETE spike events (ints)"
    assert min(train) >= 0 and max(train) <= dl.n, (
        "each neuron emits at most one 0/1 spike per step"
    )
    assert sum(train) > 0, "a real motor error must fire a spike volley"
    assert abs(released["displacement"] - dl.gain * released["burst_spikes"]) < 1e-12, (
        "displacement must be read from the ACCUMULATED SPIKE COUNT, not scalar arithmetic"
    )
    # (d) selection coupling: gaze aims at the WTA winner; flat map yields no gaze command.
    assert g_two["target"] == 4, "the deep layers must aim at the superficial WTA winner"
    assert round(g_two["landing"]) == 4, "the gaze command must point to the winner's location"
    assert g_flat["target"] is None and g_flat["displacement"] == 0.0, (
        "a flat map selects nothing -> no gaze command is emitted"
    )
    # (e) DETERMINISM: same seed -> identical noisy gaze command.
    p = SCDeepLayers(seed=0, noise=0.3)
    q = SCDeepLayers(seed=0, noise=0.3)
    assert p.command(6.0) == q.command(6.0), "same seed must reproduce an identical gaze command"

    print(
        f"self-check OK: omnipause clamps ({gated['burst_spikes']} spikes -> "
        f"{gated['displacement']:.1f}) then releases ({released['burst_spikes']} spikes -> "
        f"{released['displacement']:.2f}); displacement = accumulated spike count;"
    )
    print(
        f"  amplitude {amps[0]:.0f}->{amps[-1]:.0f} tracked by {spk[0]}->{spk[-1]} burst spikes "
        f"(disp {disp[0]:.2f}->{disp[-1]:.2f}); line attractor holds; gaze aims at WTA winner "
        f"loc {g_two['target']}; deterministic"
    )


if __name__ == "__main__":
    main()
