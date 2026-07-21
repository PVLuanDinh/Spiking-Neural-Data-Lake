"""
Claustrum -- a coincidence-gated cortical SALIENCE broadcaster.

The claustrum is a thin sheet of grey matter reciprocally connected with most of the
cortical mantle. Its dense, near-symmetric wiring makes it well placed to REGISTER when
many cortical areas are active AT THE SAME MOMENT and to route a signal back to exactly
that co-active set. This module models that one job: a COINCIDENCE detector feeding a
BROADCAST gain.

Mechanism (rate/spike level, deterministic):
  1. INPUTS  : several cortical "areas" each emit a spike train (a binary raster over time).
  2. DETECT  : the claustrum integrates over a short trailing WINDOW. At each tick it counts
               how many DISTINCT areas spiked inside that window (their coincidence). When
               that count reaches the threshold k (>= k areas co-active), the claustrum FIRES.
  3. BIND    : the co-active areas at a firing tick are the bound ASSEMBLY.
  4. BROADCAST: a salience GAIN is applied back onto the bound assembly only -- amplifying the
               areas that were mutually synchronous, leaving the rest at unit gain. Binding
               only redistributes emphasis; unbound channels are never boosted.

THE UN-RIGGABLE ANCHOR is that the trigger is COINCIDENCE, not firing rate, tested against a
FAIR CONTROL that holds the total activity fixed and varies ONLY the timing:
  (a) SYNCHRONOUS : >= k areas spike together -> the claustrum fires and binds the assembly.
  (b) ASYNC       : the SAME per-area spike budget (identical mean firing rate) re-scattered so
                    no two areas ever fall inside one window -> NO firing. Equal rate, zero
                    coincidence: the difference is timing alone, which is what is measured.
  (c) SINGLE AREA : one area driven LOUDER than the whole synchronous budget, all others silent
                    -> NO bind. Multi-area coincidence is required; sheer rate from one channel
                    is not coincidence.

HONEST FRAMING: this models a coincidence / salience-BINDING FUNCTION only. It takes NO position
on -- and gives no support to -- the claustrum-as-seat-of-consciousness hypothesis. "Fires",
"binds" and "broadcast" name information-routing dynamics; nothing here is aware of anything.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.claustrum.claustrum
"""

import random

N_AREAS = 6  # number of reciprocally-connected cortical areas feeding the claustrum
K = 3  # coincidence threshold: fire when >= K distinct areas are co-active
WINDOW = 3  # trailing coincidence-integration window, in time steps
GAIN = 1.8  # broadcast salience gain applied to the bound assembly
N_STEPS = 256  # length of every spike raster (identical across conditions -> equal-rate fair test)
N_BURSTS = 8  # synchronous coincident events in the SYNCHRONOUS stimulus


def _mean(xs):
    """Mean of a sequence, 0.0 when empty (keeps salience math total-free of exceptions)."""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _round_robin(counts):
    """Order spikes as area labels, cycling areas so same-area spikes are spread far apart.

    `counts[a]` is how many spikes area `a` owns; returns a flat list of length sum(counts)
    that preserves each area's count exactly (the async control keeps the SAME mean rate)."""
    remaining = list(counts)
    order = []
    while any(remaining):
        for a, r in enumerate(remaining):
            if r > 0:
                order.append(a)
                remaining[a] -= 1
    return order


class Claustrum:
    """Coincidence-gated salience broadcaster: >= k co-active areas within a window -> fire + bind.

    `detect(trains)` runs the coincidence gate; `broadcast(rates, bound)` applies the salience
    gain to the bound assembly only. The stimulus builders (`synchronous_stream`, `scatter_async`,
    `single_area_stream`) generate the fair-control conditions. The per-instance RNG seeds the
    synchronous burst schedule, so same seed -> identical stimulus and identical firing.
    """

    def __init__(self, n_areas=N_AREAS, k=K, window=WINDOW, gain=GAIN, seed=0):
        self.n_areas = n_areas
        self.k = k  # coincidence threshold (distinct co-active areas)
        self.window = window  # trailing integration window (time steps)
        self.gain = gain  # broadcast salience gain for the bound assembly
        self.rng = random.Random(seed)  # per-instance RNG -> deterministic burst schedule

    # -- coincidence detection ------------------------------------------------------------------
    def _active_areas(self, trains, t):
        """The set of DISTINCT areas that spiked inside the trailing window ending at tick t."""
        lo = max(0, t - self.window + 1)
        return frozenset(a for a in range(self.n_areas) if any(trains[a][lo : t + 1]))

    def detect(self, trains):
        """Run the coincidence gate over the rasters.

        Returns (fire_times, assemblies): a firing tick for every t at which >= k areas are
        co-active within the window, and the co-active (bound) assembly at each such tick."""
        n_steps = len(trains[0])
        fire_times, assemblies = [], []
        for t in range(n_steps):
            active = self._active_areas(trains, t)
            if len(active) >= self.k:
                fire_times.append(t)
                assemblies.append(active)
        return fire_times, assemblies

    def peak_coactivity(self, trains):
        """Largest number of areas ever co-active within one window (the peak coincidence)."""
        n_steps = len(trains[0])
        return max((len(self._active_areas(trains, t)) for t in range(n_steps)), default=0)

    @staticmethod
    def bound_areas(assemblies):
        """Union of every firing assembly -- the areas the claustrum bound at least once."""
        bound = set()
        for a in assemblies:
            bound |= a
        return bound

    # -- broadcast ------------------------------------------------------------------------------
    def broadcast(self, rates, bound):
        """Apply the salience gain to the bound assembly ONLY; unbound areas keep unit gain.

        Broadcasting redistributes emphasis onto the mutually-synchronous set -- it never boosts
        a channel that was not part of a coincidence."""
        return [r * self.gain if i in bound else r for i, r in enumerate(rates)]

    @staticmethod
    def rates(trains):
        """Per-area mean firing rate (spikes per time step)."""
        n_steps = len(trains[0])
        return [sum(row) / n_steps for row in trains]

    # -- fair-control stimulus builders ---------------------------------------------------------
    def synchronous_stream(self, n_steps=N_STEPS, n_bursts=N_BURSTS):
        """SYNCHRONOUS stimulus: n_bursts coincident events at RNG-scheduled ticks; each event
        fires an RNG-chosen set of (>= k) areas together, so every event is a true coincidence."""
        trains = [[0] * n_steps for _ in range(self.n_areas)]
        step = n_steps // (n_bursts + 1)
        for i in range(n_bursts):
            bt = step * (i + 1) + self.rng.randint(-2, 2)
            bt = min(max(bt, 0), n_steps - 1)
            m = self.rng.randint(self.k, self.n_areas)
            for a in self.rng.sample(range(self.n_areas), m):
                trains[a][bt] = 1
        return trains

    def scatter_async(self, trains):
        """ASYNC control: re-place every spike of `trains` so each area keeps its EXACT spike
        count (identical mean firing rate) but consecutive spikes sit `window+1` ticks apart --
        so no window ever holds two areas. Equal rate, zero coincidence; only timing changed."""
        n_steps = len(trains[0])
        counts = [sum(row) for row in trains]
        spacing = self.window + 1  # > window => no two spikes share a window
        out = [[0] * n_steps for _ in range(self.n_areas)]
        for slot, a in enumerate(_round_robin(counts)):
            t = slot * spacing
            if t >= n_steps:
                raise ValueError("async horizon too short to scatter the equal-rate budget")
            out[a][t] = 1
        return out

    def single_area_stream(self, n_spikes, area=0, n_steps=N_STEPS):
        """SINGLE-AREA control: drive ONE area with n_spikes (louder than the whole synchronous
        budget), every other area silent. High rate, but coincidence never exceeds one area."""
        out = [[0] * n_steps for _ in range(self.n_areas)]
        if n_spikes > 0:
            stride = max(1, n_steps // n_spikes)
            for j in range(n_spikes):
                out[area][min(j * stride, n_steps - 1)] = 1
        return out


def main():
    c = Claustrum(seed=0)

    sync = c.synchronous_stream()
    async_ = c.scatter_async(sync)  # SAME per-area budget, spread out
    total_sync = sum(sum(r) for r in sync)
    single = c.single_area_stream(total_sync + 4, area=0)  # louder than the whole sync budget

    conditions = [
        ("SYNCHRONOUS", sync),
        ("ASYNC (same spikes)", async_),
        ("SINGLE AREA (louder)", single),
    ]

    rows = []
    for label, trains in conditions:
        total = sum(sum(r) for r in trains)
        rate = total / (c.n_areas * len(trains[0]))
        peak = c.peak_coactivity(trains)
        fires, assemblies = c.detect(trains)
        bound = c.bound_areas(assemblies)
        base = c.rates(trains)
        amp = c.broadcast(base, bound)
        before = _mean(base[a] for a in bound)
        after = _mean(amp[a] for a in bound)
        amplify = (after / before) if before > 0 else 1.0
        rows.append((label, total, rate, peak, len(fires), len(bound), amplify))

    print("=" * 70)
    print("CLAUSTRUM -- coincidence-gated salience broadcaster (binds, not aware)")
    print("=" * 70)
    print(
        f"areas={c.n_areas}  coincidence: >= {c.k} co-active within window={c.window} steps  "
        f"broadcast gain=x{c.gain:.2f}"
    )
    print("-" * 70)
    print(
        f"{'condition':<22}{'spikes':>7}{'rate/area':>11}{'peak':>6}"
        f"{'fires':>7}{'bound':>7}{'amplify':>9}"
    )
    print("-" * 70)
    for label, total, rate, peak, nfires, nbound, amplify in rows:
        print(
            f"{label:<22}{total:>7}{rate:>11.4f}{peak:>6}"
            f"{nfires:>7}{nbound:>7}{('x' + format(amplify, '.2f')):>9}"
        )
    print("-" * 70)
    print("equal rate/area for SYNCHRONOUS vs ASYNC -> the trigger is COINCIDENCE, not rate.")
    print("=" * 70)

    (_, total_s, _, peak_s, fires_s, bound_s, amp_s) = rows[0]
    (_, total_a, _, peak_a, fires_a, bound_a, _amp_a) = rows[1]
    (_, total_1, _, peak_1, fires_1, bound_1, _amp_1) = rows[2]

    counts_sync = [sum(r) for r in sync]
    counts_async = [sum(r) for r in async_]

    # ---- (a) SYNCHRONY: co-active areas make the claustrum fire and bind the assembly ----------
    assert fires_s > 0, "synchronous co-activation must make the claustrum fire"
    assert peak_s >= c.k, f"synchronous peak coincidence must reach threshold ({peak_s} >= {c.k})"
    assert bound_s >= c.k, f"a fired assembly must bind >= k areas (bound {bound_s})"
    assert amp_s > 1.0 and abs(amp_s - c.gain) < 1e-9, (
        f"the bound assembly must be amplified by the broadcast gain (x{amp_s:.2f} vs x{c.gain:.2f})"
    )

    # ---- (b) FAIR CONTROL: SAME rate, spread in time -> no coincidence, no firing --------------
    assert counts_async == counts_sync, "async control must preserve every area's spike count"
    assert total_a == total_s, "async control must carry the identical total spike budget"
    assert peak_a < c.k, f"spread-out spikes must never reach k co-active areas (peak {peak_a})"
    assert fires_a == 0, "equal mean rate delivered async (never coincident) must NOT fire"
    assert bound_a == 0, "no coincidence -> no assembly is bound"

    # ---- (c) SINGLE AREA: louder than the whole sync budget, yet no multi-area coincidence -----
    assert total_1 >= total_s, "single-area control must be at least as loud as the sync budget"
    assert peak_1 <= 1, f"a lone area can never form a multi-area coincidence (peak {peak_1})"
    assert fires_1 == 0, "strong drive from a SINGLE area must NOT fire the claustrum"
    assert bound_1 == 0, "a single loud area binds nothing (multi-area coincidence required)"

    # ---- DETERMINISM: same seed -> identical stimulus and identical coincidence firing ---------
    a1 = Claustrum(seed=0)
    s1 = a1.synchronous_stream()
    f1, as1 = a1.detect(s1)
    a2 = Claustrum(seed=0)
    s2 = a2.synchronous_stream()
    f2, as2 = a2.detect(s2)
    assert s1 == s2 and f1 == f2 and as1 == as2, (
        "same seed must reproduce identical stimulus and identical coincidence firing"
    )

    print(
        "self-check OK: "
        f"synchrony fires {fires_s}x (peak coact {peak_s} >= k={c.k}), binds {bound_s} areas "
        f"x{amp_s:.2f}; SAME {total_s}-spike budget spread async -> {fires_a} fires "
        f"(equal rate, no coincidence); single loud area ({total_1} spikes) -> {bound_1} bound; "
        "deterministic"
    )


if __name__ == "__main__":
    main()
