"""
Superior colliculus -- a retinotopic salience map that orients a saccade by winner-take-all.

The superior colliculus holds a topographic map of visual space. Salient stimuli drive local
activity; lateral (surround) inhibition makes the locations COMPETE, and a build-to-threshold
race picks a single winner -> a saccade toward that location. Because activity integrates to a
fixed threshold, a MORE salient target reaches threshold sooner: latency falls with salience,
which is the express-vs-regular-saccade distinction.

Mechanism (rate-level, deterministic):
  1. A salience map (one value per retinotopic location) arrives from upstream feature/attention
     maps -- this module does NOT decide what is salient; it takes the map as input.
  2. Lateral inhibition: each location is divisively penalized by its surround (the mean salience
     of every OTHER location, scaled by `inhib`), then rectified:
        drive_i = max(0, salience_i - inhib * mean(salience_{j != i})).
     A location survives only if it EXCEEDS its context -- a flat map cancels to zero drive.
  3. Build-to-threshold: every location integrates its drive (a_i += drive_i, mild membrane leak).
     The FIRST location to cross the collicular threshold fires the saccade; its step is the
     latency. Larger drive -> earlier crossing -> shorter latency (express saccade).
  4. Winner-take-all: only the first crossing is emitted (one saccade). A small inlined k-WTA
     (deterministic tie-break by lowest index, matching basal_ganglia/action_gating._kwta) names
     the map's argmax so the self-check can prove the winner IS the most salient location.

Fair control (un-riggable):
  - a FLAT or ZERO salience map generates NO saccade: with nothing exceeding its surround, every
    drive rectifies to zero and no location ever reaches threshold (no spurious target). A flat
    map can carry large total energy and still produce nothing -- it is CONTRAST, not energy,
    that drives a saccade.
  - two competing targets -> the STRONGER one wins: the selected location equals the argmax of
    the salience map (winner-take-all), not a weaker competitor.
  - LATENCY decreases as target salience increases (build-to-threshold): sweeping a single
    target's salience upward yields strictly shorter saccade latencies.

Honest framing: a reduced, rate-level model of collicular target SELECTION and saccade latency
on a synthetic salience map. It models the orienting/selection FUNCTION -- which location the map
commits to and how fast -- not vision, perception, or awareness. A "saccade" here is a
threshold-crossing event in a toy accumulator array, nothing more.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.midbrain.superior_colliculus
"""
import random


class SuperiorColliculus:
    """Retinotopic salience map: lateral-inhibition competition + build-to-threshold WTA saccade.

    saccade(salience) returns which location (if any) wins and its latency in steps. No location
    ever wins on a flat/zero map; on a peaked map the argmax wins, and higher salience -> lower
    latency."""

    def __init__(self, n=24, inhib=1.0, leak=0.0, theta=3.0, max_steps=200, seed=0):
        self.n = n
        self.inhib = inhib          # surround-inhibition strength (>=1 cancels a flat map to zero)
        self.leak = leak            # accumulator membrane leak (0 == pure build-to-threshold)
        self.theta = theta          # collicular firing threshold (activity builds up to this)
        self.max_steps = max_steps  # give up (no saccade) if nothing crosses in this many steps
        self.rng = random.Random(seed)

    @staticmethod
    def _kwta(values, k):
        """Indices of the k largest values (winner-take-all), tie-break by lowest index.

        Matches basal_ganglia/action_gating._kwta so the two competition circuits agree."""
        order = sorted(range(len(values)), key=lambda i: (values[i], -i), reverse=True)
        return set(order[:k])

    def _lateral(self, salience):
        """Salience -> rectified competitive drive via surround (lateral) inhibition. A location
        that does not exceed the mean of its surround is suppressed to zero drive."""
        n = len(salience)
        total = sum(salience)
        drives = []
        for i in range(n):
            surround = (total - salience[i]) / (n - 1) if n > 1 else 0.0
            drives.append(max(0.0, salience[i] - self.inhib * surround))
        return drives

    def saccade(self, salience):
        """Run the competition + build-to-threshold race. Returns a dict with the winning target
        (or None if nothing crosses -- no saccade), the latency in steps, the per-location drives,
        and the salience argmax for cross-checking."""
        drives = self._lateral(salience)
        act = [0.0] * len(salience)
        target, latency = None, None
        for t in range(1, self.max_steps + 1):
            for i in range(len(salience)):
                act[i] = act[i] * (1.0 - self.leak) + drives[i]
            crossed = [i for i in range(len(act)) if act[i] >= self.theta]
            if crossed:
                target = min(crossed, key=lambda i: (-act[i], i))   # strongest, tie -> lowest index
                latency = t
                break
        argmax = next(iter(self._kwta(salience, 1)))
        return {"target": target, "latency": latency, "drives": drives, "argmax": argmax}

    def single_map(self, loc, salience, background=0.05, jitter=0.0):
        """A map with one target at `loc` over a low uniform background (optional deterministic
        jitter drawn from this instance's rng)."""
        m = [background] * self.n
        m[loc] = salience
        if jitter:
            m = [max(0.0, x + self.rng.uniform(-jitter, jitter)) for x in m]
        return m

    def two_target_map(self, loc_strong, s_strong, loc_weak, s_weak, background=0.05):
        """A map with a strong and a weak target competing over a low uniform background."""
        m = [background] * self.n
        m[loc_strong] = s_strong
        m[loc_weak] = s_weak
        return m


def main():
    sc = SuperiorColliculus(seed=0)

    # single salient target -> a saccade to it
    one = sc.single_map(loc=7, salience=0.8)
    res_one = sc.saccade(one)

    # flat and zero maps -> NO saccade (no spurious target)
    flat = [0.5] * sc.n
    zero = [0.0] * sc.n
    res_flat = sc.saccade(flat)
    res_zero = sc.saccade(zero)

    # two competing targets -> the stronger wins (WTA == argmax)
    two = sc.two_target_map(loc_strong=4, s_strong=0.9, loc_weak=17, s_weak=0.6)
    res_two = sc.saccade(two)

    # latency vs salience sweep (build-to-threshold: more salient -> faster)
    sweep = [0.3, 0.5, 0.7, 0.9]
    sweep_res = [sc.saccade(sc.single_map(loc=11, salience=s)) for s in sweep]
    latencies = [r["latency"] for r in sweep_res]

    print("=" * 68)
    print("SUPERIOR COLLICULUS -- retinotopic salience map -> WTA saccade")
    print("=" * 68)
    print(f"locations: {sc.n} | threshold: {sc.theta} | surround inhib: {sc.inhib} | "
          f"max steps: {sc.max_steps}")
    print(f"single target @ loc 7 (salience 0.80): saccade -> loc {res_one['target']} "
          f"at step {res_one['latency']}")
    print(f"flat map (all 0.50, high energy)     : saccade -> {res_flat['target']} "
          "(no contrast -> no target)")
    print(f"zero map (all 0.00)                  : saccade -> {res_zero['target']} "
          "(no drive -> no target)")
    print("-" * 68)
    print("two targets: strong @ loc 4 (0.90) vs weak @ loc 17 (0.60)")
    print(f"  -> winner loc {res_two['target']} at step {res_two['latency']}  "
          f"(map argmax = loc {res_two['argmax']})")
    print("-" * 68)
    print("latency falls as salience rises (express vs regular saccade):")
    for s, lat in zip(sweep, latencies):
        print(f"  salience {s:.2f} -> latency {lat} steps")
    print("=" * 68)

    # ---- un-riggable self-checks (deterministic asserts) --------------------
    # a real target produces a saccade to itself.
    assert res_one["target"] == 7, "a single salient target must be selected"
    # (1) flat / zero maps generate NO saccade (no spurious target).
    assert res_flat["target"] is None, "a flat salience map must generate no saccade"
    assert res_zero["target"] is None, "a zero salience map must generate no saccade"
    # (2) two competing targets -> the stronger (argmax) wins (winner-take-all).
    assert res_two["target"] == 4, "the stronger target must win the saccade"
    assert res_two["target"] == res_two["argmax"], "the winner must be the salience argmax (WTA)"
    # (3) latency strictly decreases as salience increases (build-to-threshold).
    assert all(lat is not None for lat in latencies), "each real target must produce a saccade"
    assert all(a > b for a, b in zip(latencies, latencies[1:])), \
        "saccade latency must strictly decrease as salience increases"
    # Determinism: same seed -> identical jittered maps and identical saccades.
    a = SuperiorColliculus(seed=0)
    b = SuperiorColliculus(seed=0)
    ma = a.single_map(loc=7, salience=0.8, jitter=0.03)
    mb = b.single_map(loc=7, salience=0.8, jitter=0.03)
    assert ma == mb, "same seed must reproduce identical jittered maps"
    assert a.saccade(ma) == b.saccade(mb), "same seed must reproduce identical saccades"

    print(f"self-check OK: single target -> loc {res_one['target']}; flat/zero maps -> no saccade; "
          f"stronger of two wins (loc {res_two['target']} == argmax);")
    print(f"  latency {latencies[0]}->{latencies[-1]} steps strictly falls as salience "
          f"{sweep[0]:.1f}->{sweep[-1]:.1f} rises; deterministic")


if __name__ == "__main__":
    main()
