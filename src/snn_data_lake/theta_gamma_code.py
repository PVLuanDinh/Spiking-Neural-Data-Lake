"""
Theta-gamma phase code -- ordered working memory by oscillatory multiplexing (pure stdlib).

Several working-memory items are multiplexed IN TIME: each item is the cell ensemble that fires in a distinct
GAMMA sub-cycle, and one slow THETA cycle nests K gamma cycles -> the items are ordered SEQUENTIALLY by their
theta PHASE (item i fires in the i-th gamma slot, at phase 2*pi*i/K). The number of gamma cycles per theta cycle
sets the working-memory CAPACITY (~7 +/- 2): pushing a K+1-th item evicts the OLDEST. Gamma inhibition separates
successive ensembles so items do not interfere. Plainly: several items held at once, one ensemble per gamma
cycle, ordered by theta phase, capacity = gamma-cycles-per-theta. (Lisman & Idiart 1995; Lisman & Jensen 2013.)

DISTINCT from existing core modules (read them):
- ring_attractor.py: a continuous attractor network holds ONE self-sustaining bump and integrates angular
  velocity into a single SPATIAL/head-direction angle. This is a DISCRETE buffer multiplexing SEVERAL items
  across gamma slots and ordering them by oscillatory PHASE -- not a spatial bump.
- htm_sequence_memory.py: high-order next-SDR PREDICTION from context via distal segments. This is NOT
  prediction -- it holds several items SIMULTANEOUSLY in a phase-ordered short-term buffer.
- hippocampus.py / hopfield attractor recall: pattern completion of ONE stored attractor. This multiplexes
  SEVERAL items by phase and reads back their ORDER.

ponytail: ceiling -- a discrete phase-slot buffer, a reduced model of the theta-gamma code on synthetic items
(integer slots + deterministic SDR ensembles; NO biophysical oscillators, no LFP, no real inhibition dynamics).
Additive / opt-in: a new memory/coding primitive; imports nothing and changes no existing behaviour. Deterministic
(seeded via random.Random; no module-level seed; no builtin hash()). Credit Lisman & Idiart 1995 / Lisman & Jensen 2013.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.theta_gamma_code
"""
import math
import random


def _stable_key(item):
    """Deterministic integer key for any item (no builtin hash): char-sum of its repr, stable across runs."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(repr(item)))


class ThetaGammaCode:
    """Phase-multiplexed working memory: up to K items, one per gamma slot, ordered by theta phase.

    Item at gamma slot i (0-based, in current order) fires at theta phase 2*pi*i/K -- phase encodes position.
    encode() appends into the next gamma slot; a K+1-th item evicts the OLDEST (the ~7 +/- 2 capacity bound set
    by the gamma/theta ratio). readout() returns the held items in phase order (== input order for a fitting run)."""

    def __init__(self, capacity=7, n_neurons=256, ensemble_size=8, seed=0):
        self.k = capacity                 # gamma cycles per theta = working-memory capacity (7 +/- 2)
        self.n = n_neurons                # neurons available for an item's gamma-window ensemble
        self.w = ensemble_size            # active neurons per ensemble (sparse code)
        self.seed = seed
        self._items = []                  # current items; list index = gamma slot = theta-phase rank
        self.evicted = []                 # items dropped by the capacity bound, oldest-first (for inspection)

    def _ensemble(self, item):
        """The sparse cell ensemble (SDR) that fires for an item in its gamma sub-cycle -- deterministic, seeded."""
        rng = random.Random(self.seed * 1_000_003 + _stable_key(item))
        return frozenset(rng.sample(range(self.n), self.w))

    def encode(self, item):
        """Place one item into the next gamma slot; evict the OLDEST if the theta cycle is already full (> K)."""
        self._items.append(item)
        if len(self._items) > self.k:
            self.evicted.append(self._items.pop(0))     # capacity limit: oldest theta-phase drops out
        return self.slot_of(item)

    def encode_sequence(self, items):
        """Push a sequence in order; returns the resulting gamma slot of each surviving push."""
        return [self.encode(it) for it in items]

    def readout(self):
        """Recover the stored items IN ORDER by scanning gamma slots across the theta phase (low phase first)."""
        return list(self._items)

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

    def spike_raster(self):
        """One theta cycle as firing events: [(gamma_slot, item), ...] -- one ensemble per gamma window, in order."""
        return [(i, it) for i, it in enumerate(self._items)]

    def raster_with_ensembles(self):
        """Spiking view: [(gamma_slot, item, ensemble_SDR), ...] -- which cells fire in each gamma sub-cycle."""
        return [(i, it, self._ensemble(it)) for i, it in enumerate(self._items)]


def _fmt_phases(phs):
    """ASCII list of phases in radians, 3dp."""
    return "[" + ", ".join(f"{p:.3f}" for p in phs) + "]"


def main():
    # ---- 1. ORDER BY PHASE: a fitting sequence reads back in the SAME order (theta phase encodes position) ----
    seq = ["RED", "GREEN", "BLUE", "YELLOW", "CYAN"]        # 5 items into a K=7 theta cycle (fits)
    tgc = ThetaGammaCode(capacity=7)
    slots = tgc.encode_sequence(seq)
    recovered = tgc.readout()

    # ---- 2. PHASE -> POSITION: item i sits in gamma slot i; phases strictly increasing and distinct ----
    phs = tgc.phases()
    slot_ok = all(tgc.slot_of(seq[i]) == i for i in range(len(seq)))
    monotonic = all(phs[i] < phs[i + 1] for i in range(len(phs) - 1))
    distinct_phase = len(set(phs)) == len(phs)

    # ---- 3. CAPACITY (7 +/- 2): overflow evicts the OLDEST, most-recent K survive in order ----
    over = ThetaGammaCode(capacity=7)
    over.encode_sequence(list(range(10)))                   # push 10 into K=7 -> drop 0,1,2
    survivors = over.readout()
    dropped = list(over.evicted)

    # ---- 4. SEPARATION: each gamma slot reads back exactly its own item; ensembles are distinct ----
    raster = tgc.spike_raster()
    slot_reads_own = all(tgc.readout()[i] == it for i, it in raster)
    ens = [e for _i, _it, e in tgc.raster_with_ensembles()]
    ens_distinct = len(set(ens)) == len(ens)
    max_overlap = max((len(a & b) for i, a in enumerate(ens) for b in ens[i + 1:]), default=0)

    # ---- 5. DETERMINISM: two fresh instances, same input -> identical buffer / readout / phases / raster ----
    a = ThetaGammaCode(capacity=7)
    a.encode_sequence(seq)
    b = ThetaGammaCode(capacity=7)
    b.encode_sequence(seq)
    deterministic = (a.readout() == b.readout() and a.phases() == b.phases()
                     and a.raster_with_ensembles() == b.raster_with_ensembles())

    print("=" * 72)
    print("THETA-GAMMA CODE -- ordered working memory by oscillatory phase multiplexing")
    print("=" * 72)
    print(f"capacity K (gamma cycles per theta) : {tgc.k}   (working-memory bound ~7 +/- 2)")
    print(f"encoded sequence                    : {seq}")
    print(f"gamma slots assigned                : {slots}")
    print(f"readout (by theta phase, in order)  : {recovered}")
    print(f"per-item theta phase (radians)      : {_fmt_phases(phs)}")
    print(f"separation: max ensemble overlap    : {max_overlap} of w={tgc.w} (distinct={ens_distinct})")
    print(f"overflow: push 0..9 into K=7        : survivors {survivors}, evicted oldest {dropped}")
    print("=" * 72)

    # ---- self-check (ponytail: deterministic asserts, ASCII-only) ----
    assert recovered == seq, f"order by phase: readout must equal input order ({recovered} != {seq})"
    assert slots == list(range(len(seq))), f"a fitting sequence fills gamma slots 0..n-1 in order ({slots})"
    assert slot_ok, "phase->position: item i must occupy gamma slot i"
    assert monotonic, f"theta phase must increase with position (monotonic): {_fmt_phases(phs)}"
    assert distinct_phase, f"each item must sit at a distinct theta phase: {_fmt_phases(phs)}"
    assert len(survivors) == 7, f"capacity: buffer must hold exactly K=7 after overflow (held {len(survivors)})"
    assert survivors == [3, 4, 5, 6, 7, 8, 9], f"overflow keeps the most-recent K in order ({survivors})"
    assert dropped == [0, 1, 2], f"overflow evicts the OLDEST items first ({dropped})"
    assert slot_reads_own, "separation: each occupied gamma slot must read back exactly the item placed there"
    assert ens_distinct and max_overlap < tgc.w, f"distinct items must use distinct ensembles (overlap {max_overlap})"
    assert deterministic, "same input must give identical buffer/readout/phases across two fresh instances"
    print(f"self-check OK: {len(seq)} items multiplexed one-per-gamma, ordered by theta phase {_fmt_phases(phs)}; "
          f"capacity K={tgc.k} (push 0..9 -> keep {survivors}, evict {dropped}); separation clean; deterministic")


if __name__ == "__main__":
    main()
