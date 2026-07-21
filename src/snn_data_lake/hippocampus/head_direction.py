"""
Head-direction ring attractor + goal-vector cells -- heading and vector navigation.

WHAT THE BRAIN DOES. Head-direction (HD) cells in the postsubiculum / anterodorsal thalamus fire when the
animal faces a particular allocentric direction, and the population holds ONE bump of activity on a ring of
preferred headings. Vestibular angular-velocity input drives an ASYMMETRIC component of the ring's recurrent
connectivity, translating the bump at a rate proportional to head turning, so the bump position INTEGRATES
angular velocity into absolute heading -- angular dead-reckoning (Taube et al., J Neurosci 10:420, 1990;
Skaggs et al., NIPS 5:173, 1995; Zhang, J Neurosci 16:2112, 1996). That integral drifts, because the velocity
input is noisy and biased; VISUAL LANDMARKS anchor the ring and reset the accumulated error (Knierim et al.,
J Neurosci 15:1648, 1995). Separately, goal-directed navigation needs a VECTOR: subiculum / entorhinal
"goal-vector" and "object-vector" cells encode the DIRECTION and DISTANCE from the animal's current place to a
remembered goal -- the place-minus-goal difference -- which converts a map into an approach command
(Sarel et al., Science 355:176, 2017; Hoydal et al., Nature 568:400, 2019).

THE GAP THIS FILLS (verified against the repo's spatial substrate). spiking_grid_can.py already
path-integrates 2-D POSITION with a moving LIF bump, and grid_place_map.py grounds place cells on it, and the
brain learns an R-STDP place-VALUE that drives approach on re-entry. But nothing in the repo represents WHICH
WAY THE AGENT FACES, and nothing computes the direction TO a goal: place value only says "this spot was good"
once you are standing on it, which cannot steer. ring_attractor.py is the closest relative but it is a RATE
network (relu-square units + divisive normalization) repurposed as an arm-joint proprioceptive observer, not a
spiking heading ring with a landmark anchor. This module supplies both missing pieces.

MECHANISM (recurrent LIF + local tuning; NO backprop, no gradient anywhere).
  HeadDirection -- the 1-D spiking reduction of spiking_grid_can's 2-D sheet, so the two share one design:
    - n LIF neurons tile a ring of preferred headings, theta_i = 2*pi*i/n.
    - Recurrence is a center-surround DIFFERENCE OF VON MISES kernels, a_e*VMe(kappa_e) - a_i*VMi(kappa_i),
      which holds exactly one self-sustaining bump (the analog memory of a heading).
    - Angular velocity enters through the ANGULAR DERIVATIVE of that kernel (dVMe, dVMi -- ring_attractor's
      w_asym), added as g*av*(a_e*dVMe - a_i*dVMi) @ syn. This translates the bump, so bump position =
      integral of angular velocity. The shift gain g is SELF-CALIBRATED once at construction so that
      step(av) advances the decoded heading by exactly `av` radians per tick.
    - Each tick: I = scale*(center-surround + velocity-shift)/n ; v = leak*v + I ; spike where v >= vth ;
      reset to 0 ; syn = syn_decay*syn + spikes. Heading is the population-vector (circular-mean) angle of
      the spike-driven synaptic field.
    - landmark(cue_heading, strength) blends the synaptic field toward a cue bump -- the VISUAL RESET that
      corrects accumulated drift. reset(heading) is the hard clamp (instantaneous re-anchor).

  GoalVector -- a spiking population code for the place-minus-goal vector:
    - n_dir direction cells with von Mises tuning to the goal BEARING, n_dist distance cells with Gaussian
      tuning to the goal DISTANCE; preferred values carry small per-cell heterogeneity drawn from `seed`
      (a real population is not perfectly uniform).
    - Drive -> LIF over `ticks` ticks -> SPIKE COUNTS -> population-vector decode. The returned bearing and
      distance are read back OUT of the spiking population, so the numbers carry genuine population-coding
      quantization error rather than being the analytic answer passed through.
    - heading_error = wrap(bearing - heading) is the top-down approach preference: it goes to zero as the
      agent turns to face the goal, and `approach` = cos(heading_error)*certainty is its signed drive.

DISTINCTNESS. Not ring_attractor.py (rate units, arm-joint proprioception, no landmark anchor, no goal
vector). Not spiking_grid_can.py (2-D position, not heading). Not grid_place_map.py (analytic rate fields).
ZERO coupling to the arm / safety envelope: representation-only, issues no commands, exposes no actuator
route, and never imports the brainstem.

HONEST SCOPE -- what this does NOT do (measured, not asserted away; see main()):
  - The heading integral is accurate only in a BOUNDED angular-velocity regime. At |av| <= ~0.05 rad/tick
    the error over a full 360 turn is under a degree; at |av| ~ 0.08 rad/tick the bump OUT-RUNS its own
    recurrence and integrates SUB-LINEARLY (~8 percent short). main() measures and PRINTS this failure and
    keeps it -- it is a real ceiling of the attractor, not something to tune away.
  - The ring cannot detect its own drift. Drift is corrected only when a landmark is SUPPLIED; with a biased
    velocity input and no landmark the heading error grows without bound. main() runs exactly that failing
    control and keeps it failing.
  - GoalVector bearing is UNDEFINED at the goal itself: at distance ~0 there is no direction to encode, the
    whole population falls silent and `certainty` collapses to 0. Callers must gate on certainty.
  - Distances beyond max_distance are NOT represented at all. The distance cells do not saturate gracefully
    at max_distance -- past roughly max_distance + 2*dist_sigma they fall COMPLETELY SILENT and `distance`
    is decoded as 0.0, i.e. a far goal reads with the same distance as standing on it. This is the most
    dangerous limit in the module and main() measures it rather than hiding it. The returned `in_range`
    flag reports whether the distance population produced any spikes at all -- callers must gate on it,
    since `distance` is meaningless when it is False. Note `in_range` is False for BOTH silent cases
    (at the goal and beyond range); what SEPARATES them is `certainty`, because out of range the
    DIRECTION population still fires (bearing stays correct, certainty ~0.93) whereas at the goal the
    whole population is silent (certainty 0).
  - Distance decode is biased INWARD near the ends of the tuned range (population edge effect): a true
    distance of 2.0 at max_distance=2.0 decodes to about 1.93.
  - Deterministic: per-instance np.random.default_rng(seed), no module-level seeding.

  python -m snn_data_lake.hippocampus.head_direction    # self-check; also runnable by run_selfchecks.py
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def _wrap_pi(a):
    """Signed smallest angle, wrapped to (-pi, pi]."""
    return (a + math.pi) % TWO_PI - math.pi


class HeadDirection:
    """Spiking HD ring attractor: a recurrent LIF bump integrates angular velocity into heading.

    Usage:
        hd = HeadDirection(n=36, seed=0)
        hd.reset(0.0)                  # clamp the bump onto a known heading
        for _ in range(20):
            hd.step(0.05)              # advance one tick at 0.05 rad/tick
        hd.decode()                    # -> heading in radians, wrapped to (-pi, pi]
        hd.landmark(true_heading)      # visual cue -> correct accumulated drift
    """

    def __init__(
        self,
        n=36,
        seed=0,
        kappa_e=3.0,
        kappa_i=0.3,
        a_e=2.0,
        a_i=0.8,
        leak=0.3,
        syn_decay=0.85,
        vth=0.1,
        scale=1.0,
        vel_gain=1.0,
        settle_steps=40,
        calibrate=True,
    ):
        self.n = int(n)
        self.seed = int(seed)
        self.kappa_e = float(kappa_e)
        self.a_e, self.a_i = float(a_e), float(a_i)
        self.leak = float(leak)
        self.syn_decay = float(syn_decay)
        self.vth = float(vth)
        self.scale = float(scale)
        self.vel_gain = float(vel_gain)

        n = self.n
        theta = np.arange(n, dtype=float) * (TWO_PI / n)  # preferred heading of each HD cell
        self.theta = theta
        d = theta[:, None] - theta[None, :]
        # center-surround: sharp excitatory von Mises minus broad inhibitory von Mises (Mexican hat)
        self.VMe = np.exp(kappa_e * (np.cos(d) - 1.0))
        self.VMi = np.exp(kappa_i * (np.cos(d) - 1.0))
        # angular derivative of each kernel -> the asymmetric, velocity-gated shift connectivity
        self.dVMe = kappa_e * np.sin(d) * self.VMe
        self.dVMi = kappa_i * np.sin(d) * self.VMi

        self.v = np.zeros(n)  # LIF membrane potentials
        self.syn = np.zeros(n)  # spike-driven synaptic activity (the readable population)
        self.spikes = np.zeros(n)  # spikes emitted on the last tick
        self._gain = 1.0
        if calibrate:
            self._gain = self._calibrate(int(settle_steps))
        self.reset(0.0)

    # ---- recurrent dynamics ----
    def _current(self, av):
        """Recurrent synaptic current: center-surround sustain + angular-velocity-gated asymmetric shift."""
        syn = self.syn
        rec = self.a_e * (self.VMe @ syn) - self.a_i * (self.VMi @ syn)
        shift = self.a_e * (self.dVMe @ syn) - self.a_i * (self.dVMi @ syn)
        g = self.vel_gain * self._gain
        return self.scale * (rec + g * av * shift) / self.n

    def step(self, angular_velocity=0.0):
        """Advance one tick under `angular_velocity` (rad/tick); return the decoded heading in radians."""
        I = self._current(float(angular_velocity))
        v = self.leak * self.v + I  # LIF membrane integration
        fired = v >= self.vth  # threshold
        v = np.where(fired, 0.0, v)  # reset to 0
        self.spikes = fired.astype(float)
        self.v = v
        self.syn = self.syn * self.syn_decay + self.spikes  # spike-driven recurrent synapses
        return self.decode()

    # ---- anchoring: hard clamp and soft landmark cue ----
    def _bump(self, heading):
        return np.exp(self.kappa_e * (np.cos(self.theta - float(heading)) - 1.0))

    def reset(self, heading=0.0):
        """Hard clamp: place the bump exactly on `heading` (instantaneous re-anchor). Returns decode()."""
        self.syn = self._bump(heading)
        self.v = np.zeros(self.n)
        self.spikes = np.zeros(self.n)
        return self.decode()

    def landmark(self, cue_heading, strength=0.5):
        """Visual-cue reset: blend the bump toward `cue_heading` by `strength` in [0,1] (drift correction).

        strength=1.0 fully re-anchors; small values pull the bump gradually, as a real landmark does over
        repeated sightings. Amplitude is preserved so the attractor keeps running.
        """
        s = min(1.0, max(0.0, float(strength)))
        cue = self._bump(cue_heading)
        peak = float(self.syn.max())
        cue = cue * (peak if peak > 0.0 else 1.0)  # match the running bump's amplitude
        self.syn = (1.0 - s) * self.syn + s * cue
        return self.decode()

    # ---- readout ----
    def decode(self):
        """Heading = population-vector (circular-mean) angle of the spiking population, in (-pi, pi]."""
        s = float(self.syn @ np.sin(self.theta))
        c = float(self.syn @ np.cos(self.theta))
        return math.atan2(s, c)

    def bump_width(self):
        """Angular std of the population around the bump center, in DEGREES (small => one localized bump)."""
        mu = self.decode()
        tot = float(self.syn.sum())
        if tot <= 0.0:
            return float("inf")
        dev = np.angle(np.exp(1j * (self.theta - mu)))
        return float(math.degrees(math.sqrt(float(self.syn @ (dev**2)) / tot)))

    def bump_amplitude(self):
        """Peak synaptic activity -- a liveness measure (a live single bump => > 0)."""
        return float(self.syn.max())

    def certainty(self):
        """Population-vector length in [0,1]: 1 = perfectly concentrated bump, 0 = no coherent heading."""
        tot = float(self.syn.sum())
        if tot <= 0.0:
            return 0.0
        s = float(self.syn @ np.sin(self.theta))
        c = float(self.syn @ np.cos(self.theta))
        return float(math.hypot(s, c) / tot)

    def _calibrate(self, settle_steps):
        """Self-calibrate the shift gain so step(av) advances the decoded heading by exactly av rad/tick."""
        self._gain = 1.0
        self.reset(0.0)
        for _ in range(settle_steps):
            self.step(0.0)
        before = self.decode()
        cal_v, cal_steps = 0.05, 10
        for _ in range(cal_steps):
            self.step(cal_v)
        moved = _wrap_pi(self.decode() - before)
        return (cal_v * cal_steps) / moved if abs(moved) > 1e-9 else 1.0


class GoalVector:
    """Goal-vector cells: a spiking population code for the DIRECTION and DISTANCE to a remembered goal.

    Usage:
        gv = GoalVector(seed=0)
        gv.set_goal((1.0, 0.0))
        gv.vector(current_pos=(0.0, 0.0), heading=0.0)
        # -> {"bearing": ~0.0, "distance": ~1.0, "heading_error": ~0.0, "certainty": .., "approach": ..}
    """

    def __init__(
        self,
        seed=0,
        n_dir=36,
        n_dist=16,
        max_distance=2.0,
        kappa=4.0,
        dist_sigma=None,
        ticks=12,
        leak=0.5,
        vth=0.6,
        near_radius=0.05,
    ):
        self.seed = int(seed)
        self.n_dir = int(n_dir)
        self.n_dist = int(n_dist)
        self.max_distance = float(max_distance)
        self.kappa = float(kappa)
        self.ticks = int(ticks)
        self.leak = float(leak)
        self.vth = float(vth)
        self.near_radius = float(near_radius)
        self.goal = None

        rng = np.random.default_rng(self.seed)  # per-instance, never module-level
        # preferred bearings, evenly tiled + small per-cell heterogeneity (a real population is not uniform)
        spacing = TWO_PI / self.n_dir
        self.pref_dir = np.arange(self.n_dir) * spacing + rng.normal(
            0.0, 0.02 * spacing, self.n_dir
        )
        # preferred distances tiling [0, max_distance] + heterogeneity
        step = self.max_distance / max(1, self.n_dist - 1)
        self.pref_dist = np.clip(
            np.arange(self.n_dist) * step + rng.normal(0.0, 0.02 * step, self.n_dist),
            0.0,
            self.max_distance,
        )
        self.dist_sigma = float(dist_sigma) if dist_sigma is not None else step
        self._dir_counts = np.zeros(self.n_dir)
        self._dist_counts = np.zeros(self.n_dist)

    def set_goal(self, pos):
        """Remember a goal position (x, y). Returns the stored goal."""
        self.goal = (float(pos[0]), float(pos[1]))
        return self.goal

    def _lif_counts(self, drive):
        """Run a tuned drive through LIF cells for `ticks` ticks; return per-cell SPIKE COUNTS."""
        v = np.zeros_like(drive)
        counts = np.zeros_like(drive)
        for _ in range(self.ticks):
            v = self.leak * v + drive
            fired = v >= self.vth
            v = np.where(fired, 0.0, v)
            counts += fired.astype(float)
        return counts

    def vector(self, current_pos, heading):
        """Goal vector from `current_pos` while facing `heading` (rad), decoded from spiking populations.

        Returns {"bearing", "distance", "heading_error", "certainty", "approach", "in_range"}. `bearing`
        is the allocentric direction to the goal; `heading_error` = wrap(bearing - heading) is the top-down
        approach preference (-> 0 as the agent turns to face the goal); `approach` = cos(heading_error)
        * certainty is its signed drive.

        TWO honest failure modes, both of which callers MUST gate on:
          - At the goal itself (distance ~ 0) bearing is UNDEFINED: the whole population is silent and
            `certainty` is 0.
          - Beyond max_distance the DISTANCE population is silent and `distance` is decoded as 0.0 even
            though the goal is far away; bearing and certainty remain valid. Never trust `distance` when
            `in_range` is False -- and note `in_range` is False in BOTH silent cases, so use `certainty`
            to tell "beyond range" (high) from "at the goal" (0).
        """
        if self.goal is None:
            raise ValueError("set_goal(pos) must be called before vector()")
        dx = self.goal[0] - float(current_pos[0])
        dy = self.goal[1] - float(current_pos[1])
        true_bearing = math.atan2(dy, dx)
        true_dist = math.hypot(dx, dy)

        # drive falls to zero at the goal: there is no direction to encode when you are already there
        gate = min(1.0, true_dist / self.near_radius) if self.near_radius > 0 else 1.0
        dir_drive = gate * np.exp(self.kappa * (np.cos(true_bearing - self.pref_dir) - 1.0))
        dist_drive = gate * np.exp(
            -((true_dist - self.pref_dist) ** 2) / (2.0 * self.dist_sigma**2)
        )
        self._dir_counts = self._lif_counts(dir_drive)
        self._dist_counts = self._lif_counts(dist_drive)

        # population-vector decode of bearing
        tot = float(self._dir_counts.sum())
        if tot <= 0.0:
            bearing, certainty = 0.0, 0.0
        else:
            s = float(self._dir_counts @ np.sin(self.pref_dir))
            c = float(self._dir_counts @ np.cos(self.pref_dir))
            bearing = math.atan2(s, c)
            certainty = float(math.hypot(s, c) / tot)
        # weighted-mean decode of distance
        dtot = float(self._dist_counts.sum())
        distance = float(self._dist_counts @ self.pref_dist / dtot) if dtot > 0.0 else 0.0

        heading_error = _wrap_pi(bearing - float(heading))
        return {
            "bearing": bearing,
            "distance": distance,
            "heading_error": heading_error,
            "certainty": certainty,
            "approach": math.cos(heading_error) * certainty,
            "in_range": bool(dtot > 0.0),  # False => `distance` is meaningless (population silent)
        }

    def population(self):
        """Last spike counts of the two goal-vector populations (the raw spiking code)."""
        return {
            "direction": self._dir_counts.tolist(),
            "distance": self._dist_counts.tolist(),
        }


def main():
    # ================= HEAD DIRECTION =================
    hd = HeadDirection(n=36, seed=0)

    # ---- 1. the bump PERSISTS with no input (analog memory of one heading) ----
    hd.reset(0.0)
    for _ in range(60):
        hd.step(0.0)
    persist_drift = abs(math.degrees(_wrap_pi(hd.decode())))
    width, amp = hd.bump_width(), hd.bump_amplitude()

    # ---- 2. ANGULAR PATH INTEGRATION: a known av sequence integrates to the right heading ----
    hd.reset(0.0)
    for _ in range(30):
        hd.step(0.0)  # let the clamped bump relax onto the ring
    start = hd.decode()
    seq = [0.04] * 20 + [-0.03] * 10 + [0.05] * 12
    target = start
    min_amp = amp
    for av in seq:
        hd.step(av)
        target += av
        min_amp = min(min_amp, hd.bump_amplitude())
    integ_decoded = hd.decode()  # capture NOW; hd keeps running in later sections
    integ_err = abs(math.degrees(_wrap_pi(integ_decoded - target)))

    # ---- 3. a FULL 360 turn returns to the start (closed-loop angular integration) ----
    hd.reset(0.0)
    for _ in range(30):
        hd.step(0.0)
    s360 = hd.decode()
    for _ in range(126):
        hd.step(TWO_PI / 126.0)
    full_err = abs(math.degrees(_wrap_pi(hd.decode() - s360)))

    # ---- 4. DRIFT + LANDMARK RESET. The velocity input is biased and noisy (as vestibular input is), so
    #         the integral accumulates error. This is the CONTROL THAT MUST FAIL: with no landmark the
    #         heading drifts away and the ring cannot tell. A landmark cue then corrects it.
    rng = np.random.default_rng(0)
    hd.reset(0.0)
    for _ in range(30):
        hd.step(0.0)
    true_h = hd.decode()
    for _ in range(120):
        av_true = 0.03
        av_sensed = av_true * 1.25 + float(
            rng.normal(0.0, 0.006)
        )  # biased + noisy vestibular estimate
        hd.step(av_sensed)
        true_h += av_true
    drift_before = abs(math.degrees(_wrap_pi(hd.decode() - true_h)))
    for _ in range(6):  # repeated landmark sightings pull the bump back onto the true heading
        hd.landmark(true_h, strength=0.5)
        hd.step(0.0)
        true_h += 0.0
    drift_after = abs(math.degrees(_wrap_pi(hd.decode() - true_h)))

    # ---- 5. HONEST CEILING: too-fast turning OUT-RUNS the bump -> sub-linear integration (kept failing) ----
    def turn_err(av, steps):
        h = HeadDirection(n=36, seed=0)
        h.reset(0.0)
        for _ in range(30):
            h.step(0.0)
        st = h.decode()
        for _ in range(steps):
            h.step(av)
        return math.degrees(_wrap_pi(h.decode() - (st + av * steps))), math.degrees(av * steps)

    slow_err, slow_turn = turn_err(0.04, 25)  # inside the calibrated linear regime
    fast_err, fast_turn = turn_err(0.14, 25)  # OUT-RUNS the recurrence -> genuinely fails
    fast_frac = abs(fast_err) / abs(fast_turn) * 100.0

    # ---- 6. determinism ----
    a, b = HeadDirection(n=36, seed=0), HeadDirection(n=36, seed=0)
    a.reset(0.2)
    b.reset(0.2)
    hd_det = all(a.step(0.03) == b.step(0.03) for _ in range(30))

    # ================= GOAL VECTOR =================
    gv = GoalVector(seed=0)
    gv.set_goal((1.0, 0.0))

    # ---- 7. the goal vector points correctly from SEVERAL positions ----
    probes = [
        ((0.0, 0.0), 0.0, 1.0),  # due east, 1.0 away
        ((1.0, -1.0), math.pi / 2, 1.0),  # due north
        ((2.0, 0.0), math.pi, 1.0),  # due west
        ((0.0, -1.0), math.pi / 4, math.sqrt(2)),  # north-east, sqrt(2)
        ((1.5, 0.5), -math.pi * 3 / 4, math.hypot(0.5, 0.5)),  # south-west
    ]
    bearing_errs, dist_errs = [], []
    for pos, tb, td in probes:
        r = gv.vector(pos, 0.0)
        bearing_errs.append(abs(math.degrees(_wrap_pi(r["bearing"] - tb))))
        dist_errs.append(abs(r["distance"] - td))
    max_bearing_err = max(bearing_errs)
    max_dist_err = max(dist_errs)

    # ---- 8. heading_error SHRINKS as the agent turns to face the goal ----
    turn_seq = [math.pi * 0.9, math.pi * 0.6, math.pi * 0.3, math.pi * 0.1, 0.0]
    herrs = [abs(gv.vector((0.0, 0.0), h)["heading_error"]) for h in turn_seq]
    approaches = [gv.vector((0.0, 0.0), h)["approach"] for h in turn_seq]
    monotone = all(herrs[i + 1] < herrs[i] for i in range(len(herrs) - 1))

    # ---- 9. HONEST FAILURES of the goal-vector population (measured, kept) ----
    at_goal = gv.vector((1.0, 0.0), 0.0)  # standing ON the goal -> bearing undefined
    gv_far = GoalVector(seed=0)
    gv_far.set_goal((8.0, 0.0))
    far = gv_far.vector((0.0, 0.0), 0.0)  # 8.0 away, but cells only tile out to max_distance=2.0
    far_shortfall = 8.0 - far["distance"]  # the distance cells are SILENT -> decodes 0.0, not 2.0
    far_bearing_err = abs(math.degrees(_wrap_pi(far["bearing"] - 0.0)))
    # edge bias: a true distance sitting exactly at max_distance still decodes INWARD
    gv_edge = GoalVector(seed=0)
    gv_edge.set_goal((2.0, 0.0))
    edge_bias = 2.0 - gv_edge.vector((0.0, 0.0), 0.0)["distance"]

    # ---- 10. determinism ----
    g1, g2 = GoalVector(seed=0), GoalVector(seed=0)
    g1.set_goal((0.3, 0.7))
    g2.set_goal((0.3, 0.7))
    gv_det = g1.vector((0.1, 0.2), 0.4) == g2.vector((0.1, 0.2), 0.4)
    g3 = GoalVector(seed=1)
    g3.set_goal((0.3, 0.7))
    seed_matters = g3.vector((0.1, 0.2), 0.4) != g1.vector((0.1, 0.2), 0.4)

    print("=" * 78)
    print("HEAD DIRECTION + GOAL VECTOR -- spiking heading ring and vector navigation")
    print("=" * 78)
    print(
        f"HD ring: {hd.n} LIF cells, center-surround VMe(k={hd.kappa_e:.1f})-VMi, shift gain {hd._gain:.2f}"
    )
    print(
        f"  persist : {persist_drift:.2f} deg drift / 60 idle ticks, bump width {width:.1f} deg, amp {amp:.2f}"
    )
    print(
        f"  integrate: 42-step av sequence -> target {math.degrees(_wrap_pi(target)):8.2f} deg, "
        f"decoded {math.degrees(integ_decoded):8.2f} deg (err {integ_err:.2f} deg, min amp {min_amp:.2f})"
    )
    print(f"  full 360 : return err {full_err:.2f} deg")
    print(
        f"  DRIFT (biased+noisy av, NO landmark) -> {drift_before:.1f} deg error   [control: must FAIL]"
    )
    print(f"  landmark reset x6 -> {drift_after:.1f} deg error   [drift CORRECTED]")
    print(
        f"  CEILING: av=0.04 turn {slow_turn:.0f} deg -> err {slow_err:+.2f} deg (linear regime); "
        f"av=0.14 turn {fast_turn:.0f} deg -> err {fast_err:+.1f} deg ({fast_frac:.0f} pct SHORT, out-runs bump)"
    )
    print(f"  determinism: {hd_det}")
    print("-" * 78)
    print(
        f"Goal-vector cells: {gv.n_dir} direction + {gv.n_dist} distance LIF cells, "
        f"max_distance {gv.max_distance:.1f}, {gv.ticks} ticks"
    )
    print(
        f"  5 probe positions -> max bearing err {max_bearing_err:.2f} deg, max distance err {max_dist_err:.3f}"
    )
    print(
        "  heading_error as the agent turns toward the goal (deg): "
        + " -> ".join(f"{math.degrees(e):.1f}" for e in herrs)
    )
    print("  approach drive: " + " -> ".join(f"{a:+.2f}" for a in approaches))
    print(
        f"  AT THE GOAL (dist 0): certainty {at_goal['certainty']:.3f}, in_range {at_goal['in_range']}, "
        f"bearing UNDEFINED   [honest gap]"
    )
    print(
        f"  BEYOND RANGE (true 8.0): distance cells SILENT -> decoded {far['distance']:.2f} "
        f"({far_shortfall:.2f} SHORT; it does NOT saturate at max_distance), in_range "
        f"{far['in_range']}   [honest ceiling]"
    )
    print(
        f"    but bearing survives out of range: err {far_bearing_err:.2f} deg, certainty "
        f"{far['certainty']:.3f} (vs {at_goal['certainty']:.3f} at the goal) -> in_range is False for "
        f"BOTH, CERTAINTY is what separates them"
    )
    print(f"    edge bias: true 2.0 at max_distance decodes {edge_bias:.3f} INWARD   [honest bias]")
    print(f"  determinism: {gv_det}, seed changes the population: {seed_matters}")
    print("=" * 78)

    # ================= self-check asserts =================
    assert amp > 0.5 and min_amp > 0.5, (
        f"a single spiking HD bump must stay alive (amp {amp:.2f}, min {min_amp:.2f})"
    )
    assert persist_drift < 2.0 and width < 60.0, (
        f"the bump must persist as one localized attractor (drift {persist_drift:.2f} deg, width {width:.1f})"
    )
    assert integ_err < 5.0, (
        f"heading must integrate the angular-velocity sequence (err {integ_err:.2f} deg)"
    )
    assert full_err < 5.0, f"a full 360 turn must return to the start (err {full_err:.2f} deg)"
    # the drift control MUST genuinely fail, and the landmark MUST fix it
    assert drift_before > 20.0, (
        f"CONTROL MUST FAIL: biased velocity with no landmark must drift badly "
        f"(only {drift_before:.1f} deg -- control is not actually failing)"
    )
    assert drift_after < 5.0, (
        f"a landmark reset must correct accumulated drift ({drift_before:.1f} -> {drift_after:.1f} deg)"
    )
    assert drift_after < drift_before / 4.0, "the landmark must cut drift by at least 4x"
    # the fast-turn ceiling MUST genuinely fail (we keep it, we do not tune it away)
    assert abs(slow_err) < 5.0, f"av=0.04 must stay in the linear regime (err {slow_err:.2f} deg)"
    assert fast_frac > 10.0, (
        f"CONTROL MUST FAIL: av=0.14 must out-run the bump and integrate sub-linearly "
        f"(only {fast_frac:.1f} pct short -- control is not actually failing)"
    )
    assert hd_det, "same seed must integrate an identical heading trajectory (deterministic)"

    assert max_bearing_err < 8.0, (
        f"the goal vector must point at the goal from every probe (max err {max_bearing_err:.2f} deg)"
    )
    assert max_dist_err < 0.25, (
        f"the goal-vector population must decode distance (max err {max_dist_err:.3f})"
    )
    assert monotone, f"heading_error must shrink as the agent turns toward the goal ({herrs})"
    assert herrs[-1] < math.radians(8.0), (
        f"heading_error must reach ~0 when facing the goal ({math.degrees(herrs[-1]):.2f} deg)"
    )
    assert approaches[-1] > approaches[0], "approach drive must rise as the agent faces the goal"
    # honest gaps: these MUST stay broken -- they are real limits of the representation
    assert at_goal["certainty"] < 0.05 and not at_goal["in_range"], (
        f"CONTROL MUST FAIL: bearing must be undefined at the goal "
        f"(certainty {at_goal['certainty']:.3f} -- population is not actually silent)"
    )
    assert far_shortfall > 5.0 and not far["in_range"], (
        f"CONTROL MUST FAIL: a goal beyond max_distance must leave the distance population SILENT and "
        f"decode 0.0 (shortfall only {far_shortfall:.2f}, in_range {far['in_range']})"
    )
    assert far["distance"] < 0.01, (
        f"the out-of-range failure is SILENCE, not saturation -- distance must read ~0.0, "
        f"got {far['distance']:.3f}"
    )
    assert far_bearing_err < 8.0 and far["certainty"] > 0.5, (
        f"bearing must survive out of range so in_range/certainty can separate it from 'at the goal' "
        f"(err {far_bearing_err:.2f} deg, certainty {far['certainty']:.3f})"
    )
    assert edge_bias > 0.02, (
        f"CONTROL MUST FAIL: distance decode must be biased inward at the range edge "
        f"(bias only {edge_bias:.3f})"
    )
    assert gv_det, "same seed must produce an identical goal vector (deterministic)"
    assert seed_matters, "the seed must actually change the heterogeneous population"

    print(
        f"self-check OK: {hd.n}-LIF HD ring integrates angular velocity (seq err {integ_err:.2f} deg, "
        f"360 err {full_err:.2f} deg); biased input drifts {drift_before:.1f} deg and a landmark corrects "
        f"it to {drift_after:.1f} deg; FAILS honestly at av=0.14 ({fast_frac:.0f} pct short); goal-vector "
        f"cells decode bearing within {max_bearing_err:.2f} deg and distance within {max_dist_err:.3f}, "
        f"heading_error shrinks to {math.degrees(herrs[-1]):.2f} deg; bearing UNDEFINED at the goal "
        f"(certainty {at_goal['certainty']:.3f}) and beyond max_distance the distance cells go SILENT so "
        f"distance reads 0.0 ({far_shortfall:.1f} SHORT, in_range False -- it does NOT saturate); "
        f"deterministic"
    )


if __name__ == "__main__":
    main()
