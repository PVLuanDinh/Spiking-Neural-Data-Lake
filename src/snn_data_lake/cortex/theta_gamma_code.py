"""
Theta-gamma phase code -- ordered working memory by a SPIKING oscillation.

Working-memory items are multiplexed IN TIME by a genuine spiking network (Lisman & Idiart 1995;
Lisman & Jensen 2013). One slow THETA cycle is divided into K faster GAMMA sub-cycles = K ordered memory
slots. Each held item is a POPULATION of leaky integrate-and-fire (LIF) neurons that is driven ONLY during
its own gamma window, so it INTEGRATES its input current and FIRES a discrete spike at that item's gamma
phase within the theta cycle -- spike TIMING is which slot. Item i fires in gamma slot i, at theta phase
2*pi*i/K. Recall RE-RUNS the oscillation and reads the items back by the gamma window (spike PHASE) in which
each population fires: the ordered set is decoded from spike timing, not from a Python list. The number of
gamma cycles that fit in one theta cycle (steps_per_theta / steps_per_gamma) sets the working-memory CAPACITY
(~7 +/- 2); a K+1-th item evicts the OLDEST. The populations re-fire every theta cycle -> active maintenance.

DISTINCT from existing core modules (read them):
- ring_attractor.py: a continuous attractor network holds ONE self-sustaining bump and integrates angular
  velocity into a single SPATIAL/head-direction angle. This is a DISCRETE buffer multiplexing SEVERAL items
  across gamma slots and ordering them by oscillatory PHASE -- not a spatial bump.
- htm_sequence_memory.py: high-order next-SDR PREDICTION from context via distal segments. This is NOT
  prediction -- it holds several items SIMULTANEOUSLY in a phase-ordered short-term buffer.
- hippocampus.py / hopfield attractor recall: pattern completion of ONE stored attractor. This multiplexes
  SEVERAL items by phase and reads back their ORDER from LIF spike timing.
- lif_neuron.py: the single-neuron LIF primitive. This composes item-specific LIF POPULATIONS driven by a
  nested theta/gamma oscillation into a phase-coded working-memory buffer.

Genuinely spiking: each item is a population of LIF neurons (membrane leak + current integration + threshold
spike + reset + refractory); items fire discrete spikes at item-specific gamma phases; recall reads spike
timing; capacity is the theta/gamma ratio. Additive / opt-in: imports nothing but math + random, changes no
existing behaviour. Deterministic (seeded via random.Random per instance; no module-level seed; no builtin
hash()). Credit Lisman & Idiart 1995 / Lisman & Jensen 2013.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.theta_gamma_code
"""

import math
import random


def _stable_key(item):
    """Deterministic integer key for any item (no builtin hash): char-sum of its repr, stable across runs."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(repr(item)))


class _LIFNeuron:
    """Minimal discrete-time leaky integrate-and-fire unit: V leaks toward 0, integrates current, spikes+resets.

    Inlined (not imported) so this module stays self-contained. step(current) -> 1 if it fired this step else 0."""

    __slots__ = ("decay", "vth", "vreset", "refr_len", "v", "_refr")

    def __init__(self, tau_m, v_threshold, v_reset, refractory):
        self.decay = math.exp(-1.0 / tau_m)  # membrane leak per timestep
        self.vth = v_threshold
        self.vreset = v_reset
        self.refr_len = refractory
        self.v = 0.0
        self._refr = 0

    def step(self, current):
        """Advance one timestep with input `current`; return 1 if the neuron spiked else 0."""
        if self._refr > 0:  # refractory: clamped, integrates nothing
            self._refr -= 1
            self.v = self.vreset
            return 0
        self.v = self.v * self.decay + current  # leak toward rest + integrate the drive current
        if self.v >= self.vth:  # threshold crossing -> discrete spike
            self.v = self.vreset
            self._refr = self.refr_len
            return 1
        return 0


class ThetaGammaCode:
    """Phase-multiplexed working memory: up to K items, one LIF population per gamma slot, ordered by theta phase.

    Item at gamma slot i (0-based, in current order) is a population of LIF neurons driven only in gamma window i,
    where it integrates its input and fires at theta phase 2*pi*i/K -- spike timing encodes position. encode()
    appends into the next gamma slot; a K+1-th item evicts the OLDEST (the ~7 +/- 2 capacity bound = the
    theta/gamma ratio). readout() re-runs the oscillation and returns the held items in phase order, decoded from
    which population spikes in which gamma window (== input order for a fitting run)."""

    def __init__(self, capacity=7, n_neurons=256, ensemble_size=8, seed=0):
        self.k = capacity  # gamma cycles per theta = working-memory capacity (7 +/- 2)
        self.n = n_neurons  # neurons available for an item's gamma-window population
        self.w = ensemble_size  # LIF neurons per population (sparse code)
        self.seed = seed
        self._items = []  # current items; list index = gamma slot = theta-phase rank
        self.evicted = []  # items dropped by the capacity bound, oldest-first (for inspection)
        # --- oscillation geometry: one theta cycle nests K gamma windows of steps_per_gamma steps each ---
        self.steps_per_gamma = (
            6  # timesteps per gamma sub-cycle (a population integrates over these)
        )
        self.steps_per_theta = (
            self.k * self.steps_per_gamma
        )  # capacity K == steps_per_theta / steps_per_gamma
        # --- LIF parameters: driven for ~2 steps before crossing threshold, one spike per gamma window ---
        self.tau_m = 8.0  # membrane time constant
        self.i_drive = 0.8  # suprathreshold drive during an item's own gamma window (0 otherwise)
        self.refractory = self.steps_per_gamma  # one spike per neuron per gamma window

    def _ensemble(self, item):
        """The sparse LIF population (which cells fire) for an item in its gamma sub-cycle -- deterministic, seeded."""
        rng = random.Random(self.seed * 1_000_003 + _stable_key(item))
        return frozenset(rng.sample(range(self.n), self.w))

    def _new_population(self, item):
        """Build this item's population: w LIF neurons with deterministic, seeded threshold heterogeneity."""
        rng = random.Random(self.seed * 2_000_003 + _stable_key(item))
        neurons = []
        for _ in range(self.w):
            vth = (
                0.9 + 0.2 * rng.random()
            )  # heterogeneous threshold in [0.9, 1.1); still fires by step 2
            neurons.append(_LIFNeuron(self.tau_m, vth, 0.0, self.refractory))
        return neurons

    def _simulate(self, n_cycles=1):
        """Run n_cycles theta cycles of every held population; return discrete spikes.

        Each timestep, the population in the CURRENT gamma window gets i_drive (all others 0), so exactly one
        population spikes per gamma window. Returns [(cycle, gamma_window, slot, neuron_id), ...]."""
        ens = [sorted(self._ensemble(it)) for it in self._items]  # neuron ids per slot
        pops = [self._new_population(it) for it in self._items]  # LIF neurons per slot
        spt, spg = self.steps_per_theta, self.steps_per_gamma
        spikes = []
        for cyc in range(n_cycles):
            for local_t in range(spt):
                window = local_t // spg  # active gamma sub-cycle this step
                for slot, neurons in enumerate(pops):
                    drive = (
                        self.i_drive if slot == window else 0.0
                    )  # gamma inhibition: only own window drives
                    for j, neuron in enumerate(neurons):
                        if neuron.step(drive):
                            spikes.append((cyc, window, slot, ens[slot][j]))
        return spikes

    def _decode_cycle(self, spikes, cyc):
        """Recover ordered (gamma_window, item) for one theta cycle by reading SPIKE PHASE, not the buffer.

        For each gamma window that fired (ascending == ascending theta phase), match the set of spiking cells to
        the stored item whose population overlaps it most -> a readout decoded purely from spike timing+identity."""
        fired = {}
        for c, window, _slot, nid in spikes:
            if c == cyc:
                fired.setdefault(window, set()).add(nid)
        out = []
        for window in sorted(fired):
            firing = fired[window]
            best, best_ov = None, -1
            for it in self._items:
                ov = len(self._ensemble(it) & firing)
                if ov > best_ov:
                    best, best_ov = it, ov
            out.append((window, best))
        return out

    def encode(self, item):
        """Place one item into the next gamma slot; evict the OLDEST if the theta cycle is already full (> K)."""
        self._items.append(item)
        if len(self._items) > self.k:
            self.evicted.append(self._items.pop(0))  # capacity limit: oldest theta-phase drops out
        return self.slot_of(item)

    def encode_sequence(self, items):
        """Push a sequence in order; returns the resulting gamma slot of each surviving push."""
        return [self.encode(it) for it in items]

    def readout(self):
        """Recover the stored items IN ORDER by re-running the oscillation and decoding spike phase (low phase first)."""
        if not self._items:
            return []
        decoded = self._decode_cycle(self._simulate(n_cycles=1), 0)
        return [it for _window, it in decoded]

    def __len__(self):
        return len(self._items)

    def slot_phase(self, i):
        """Theta phase (radians, 0..2*pi) of gamma slot i -- monotonic with position, distinct per slot (i < K)."""
        return 2.0 * math.pi * i / self.k

    def phases(self):
        """Per-item theta phase, in current slot order (position -> phase, strictly increasing)."""
        return [self.slot_phase(i) for i in range(len(self._items))]

    def slot_of(self, item):
        """Gamma slot currently occupied by item (0-based), or None if it is not held (never placed / evicted)."""
        for i, it in enumerate(self._items):
            if it == item:
                return i
        return None

    def phase_of(self, item):
        """Theta phase of item, or None if not held."""
        i = self.slot_of(item)
        return None if i is None else self.slot_phase(i)

    def spike_events(self, n_cycles=1):
        """Discrete LIF spike events over n_cycles theta cycles: [(theta_cycle, gamma_window, theta_phase, neuron_id), ...]."""
        return [
            (cyc, window, self.slot_phase(window), nid)
            for cyc, window, _slot, nid in self._simulate(n_cycles)
        ]

    def spike_raster(self):
        """One theta cycle as firing events: [(gamma_window, item), ...] -- decoded from LIF spike phase, in order."""
        if not self._items:
            return []
        return self._decode_cycle(self._simulate(n_cycles=1), 0)

    def raster_with_ensembles(self):
        """Spiking view: [(gamma_window, item, population_SDR), ...] -- which LIF cells fire in each gamma sub-cycle."""
        return [(window, it, self._ensemble(it)) for window, it in self.spike_raster()]


def _fmt_phases(phs):
    """ASCII list of phases in radians, 3dp."""
    return "[" + ", ".join(f"{p:.3f}" for p in phs) + "]"


def main():
    # ---- 1. ORDER BY SPIKE PHASE: a fitting sequence reads back in the SAME order, decoded from LIF spike timing ----
    seq = ["RED", "GREEN", "BLUE", "YELLOW", "CYAN"]  # 5 items into a K=7 theta cycle (fits)
    tgc = ThetaGammaCode(capacity=7)
    slots = tgc.encode_sequence(seq)
    recovered = tgc.readout()  # re-runs the oscillation; reads back by gamma phase

    # ---- 2. PHASE -> POSITION: item i sits in gamma slot i; phases strictly increasing and distinct ----
    phs = tgc.phases()
    slot_ok = all(tgc.slot_of(seq[i]) == i for i in range(len(seq)))
    monotonic = all(phs[i] < phs[i + 1] for i in range(len(phs) - 1))
    distinct_phase = len(set(phs)) == len(phs)

    # ---- 3. CAPACITY (theta/gamma ratio, 7 +/- 2): overflow evicts the OLDEST, most-recent K survive in order ----
    over = ThetaGammaCode(capacity=7)
    over.encode_sequence(list(range(10)))  # push 10 into K=7 -> drop 0,1,2
    survivors = over.readout()
    dropped = list(over.evicted)
    ratio_cap = (
        tgc.steps_per_theta // tgc.steps_per_gamma
    )  # capacity EMERGES from the theta/gamma ratio

    # ---- 4. GENUINELY SPIKING: LIF populations fire discrete spikes at discrete gamma phases, only in their window ----
    events = tgc.spike_events(n_cycles=1)  # (cycle, gamma_window, theta_phase, neuron_id)
    n_spikes = len(events)
    windows_fired = sorted({g for _c, g, _p, _n in events})
    raster = tgc.spike_raster()  # [(gamma_window, item)] decoded from spikes
    slot_reads_own = all(w == tgc.slot_of(it) and recovered[w] == it for w, it in raster)

    # ---- 5. MAINTENANCE: every population re-fires in EVERY theta cycle (persistent active holding) ----
    maint = tgc.spike_events(n_cycles=3)
    per_cycle = {}
    for c, g, _p, _n in maint:
        per_cycle.setdefault(c, set()).add(g)
    refire = all(per_cycle.get(c) == set(range(len(seq))) for c in range(3))

    # ---- 6. SEPARATION: each gamma window reads back exactly its own item; populations are distinct ----
    ens = [e for _w, _it, e in tgc.raster_with_ensembles()]
    ens_distinct = len(set(ens)) == len(ens)
    max_overlap = max((len(a & b) for i, a in enumerate(ens) for b in ens[i + 1 :]), default=0)

    # ---- 7. DETERMINISM: two fresh instances, same input -> identical readout / phases / raster / spikes ----
    a = ThetaGammaCode(capacity=7)
    a.encode_sequence(seq)
    b = ThetaGammaCode(capacity=7)
    b.encode_sequence(seq)
    deterministic = (
        a.readout() == b.readout()
        and a.phases() == b.phases()
        and a.raster_with_ensembles() == b.raster_with_ensembles()
        and a.spike_events(2) == b.spike_events(2)
    )

    print("=" * 72)
    print("THETA-GAMMA CODE -- spiking working memory by LIF gamma-phase multiplexing")
    print("=" * 72)
    print(
        f"capacity K (gamma cycles per theta) : {tgc.k}   (== steps_per_theta/steps_per_gamma = {ratio_cap})"
    )
    print(
        f"theta cycle geometry                : {tgc.steps_per_theta} steps = K x {tgc.steps_per_gamma} gamma steps"
    )
    print(f"encoded sequence                    : {seq}")
    print(f"gamma slots assigned                : {slots}")
    print(f"readout (decoded from spike phase)  : {recovered}")
    print(f"per-item theta phase (radians)      : {_fmt_phases(phs)}")
    print(
        f"LIF spikes / theta cycle            : {n_spikes} (={len(seq)} pops x w={tgc.w}), in windows {windows_fired}"
    )
    print(
        f"maintenance: windows firing / cycle : {[sorted(per_cycle[c]) for c in range(3)]} (re-fires each theta)"
    )
    print(
        f"separation: max population overlap  : {max_overlap} of w={tgc.w} (distinct={ens_distinct})"
    )
    print(f"overflow: push 0..9 into K=7        : survivors {survivors}, evicted oldest {dropped}")
    print("=" * 72)

    # ---- self-check: deterministic asserts, ASCII-only ----
    # FUNCTION -- ordered store/recall by phase + capacity
    assert recovered == seq, (
        f"order by phase: readout must equal input order ({recovered} != {seq})"
    )
    assert slots == list(range(len(seq))), (
        f"a fitting sequence fills gamma slots 0..n-1 in order ({slots})"
    )
    assert slot_ok, "phase->position: item i must occupy gamma slot i"
    assert monotonic, f"theta phase must increase with position (monotonic): {_fmt_phases(phs)}"
    assert distinct_phase, f"each item must sit at a distinct theta phase: {_fmt_phases(phs)}"
    assert len(survivors) == 7, (
        f"capacity: buffer must hold exactly K=7 after overflow (held {len(survivors)})"
    )
    assert survivors == [3, 4, 5, 6, 7, 8, 9], (
        f"overflow keeps the most-recent K in order ({survivors})"
    )
    assert dropped == [0, 1, 2], f"overflow evicts the OLDEST items first ({dropped})"
    assert ratio_cap == tgc.k, f"capacity must equal the theta/gamma ratio ({ratio_cap} != {tgc.k})"
    assert slot_reads_own, (
        "separation: each occupied gamma window must read back exactly the item placed there"
    )
    # GENUINELY SPIKING -- LIF populations, discrete spikes at discrete phases, recall reads spike timing
    assert n_spikes == len(seq) * tgc.w, (
        f"each population must emit w discrete spikes / theta cycle ({n_spikes})"
    )
    assert windows_fired == list(range(len(seq))), (
        f"spikes must land at discrete gamma phases 0..n-1 ({windows_fired})"
    )
    assert refire, "maintenance: every population must re-fire in every theta cycle"
    assert ens_distinct and max_overlap < tgc.w, (
        f"distinct items must use distinct populations (overlap {max_overlap})"
    )
    # DETERMINISM
    assert deterministic, (
        "same input must give identical readout/phases/raster/spikes across two fresh instances"
    )
    print(
        f"self-check OK: {len(seq)} items held as LIF populations firing one-per-gamma, recall decoded from spike "
        f"phase {_fmt_phases(phs)}; {n_spikes} discrete spikes/theta, re-firing each cycle; capacity K={tgc.k} "
        f"(push 0..9 -> keep {survivors}, evict {dropped}); separation clean; deterministic"
    )


if __name__ == "__main__":
    main()
