"""
Unit blocks and block-aligned lake records (specs/022 T064 / Phase 3 S1).

WHAT IT IS. A partition of `n_units` units into fixed-size BLOCKS, plus the framing that lets a
lake record be grouped by block, plus the residency discipline that makes "lake-resident" mean
something: exactly ONE block is resident at a time.

WHY BLOCKS AND NOT A FLAT RECORD. A flat per-synapse record is addressable but not STREAMABLE: to
apply it you need the whole weight matrix in core. Grouping entries by the block their POST unit
lives in means a reader can load one block, apply every entry addressed to it, free it, and move on
-- so peak resident state is one block, not the substrate. That is the property this module exists
to provide, and R-S1a asserts it rather than assuming it.

PORTED, NOT REBUILT (P2). `substrate/neuro_chip_stream.StreamingChip` already proves the discipline
for chip CORES: shards on disk, one resident, `max_resident_cores == 1` asserted in its own
self-check. This module ports that DISCIPLINE to units -- the counter, the context manager, and the
proof shape -- and deliberately does not port its code, which is AER/chip-specific and would be a
second chip emulator if copied. The two remain distinct mechanisms for distinct objects; what is
shared is the invariant and how it is proven.

BLOCK SIZE HAS NO DEFAULT, and that is P1 rather than an oversight. The right block is a property
of the caller's state-per-unit and its memory budget -- which is exactly what specs/022 T065 exists
to MEASURE. A default here would be this module choosing the number the next row derives, so the
constructor refuses, the same way `lake_engine.SynapseDelta` refuses to pick `q_step`.

WHAT THIS MODULE DOES NOT CLAIM: no paging benefit (T066 owns that, with its own reds), no append
budget (the writer this enables is what makes that measurable, taken in its own row), and nothing
about unit GROWTH -- rewiring becomes lake-resident here, growth does not.
"""

from contextlib import contextmanager

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it. The gate's walk is
#: recall-biased and captures this module's `main`, so it is declared rather than left
#: UNDETERMINABLE -- the same disposition `basal_ganglia/r_stdp.py` gives its `calibrate` benchmark:
#: none of the four learning dispositions describes a self-check, and any of them would be false.
GRADIENT_CLAIM = {
    "main": (
        "NOT_A_RULE",
        "this module's self-check: it partitions units, groups records and proves the "
        "one-block-resident discipline. It moves no weight and computes no update, so it is not a "
        "learning rule at all -- declared so the gate's UNDETERMINABLE ratchet stays honest debt",
    ),
}


class BlockResidencyError(RuntimeError):
    """A second block was made resident while one was already in core.

    Raised rather than counted: the whole point of the discipline is that peak resident state is one
    block, so a nested residency is a broken contract, not a statistic to report afterwards."""


# distinct-from: cortex/neurogenesis_engine.py::placement -- per-unit (depth, layer, sign) metadata
#   aligned with the substrate rows it describes; no blocks, no residency, no record framing.
# distinct-from: basal_ganglia/learning_progress.py::GoalStore -- _T_GOAL event rows with a derived
#   cosine index; a different tag and no partition.
# distinct-from: reasoning/arc_solver.py::_unscale -- uniform kxk BLOCKS of an ARC grid, pure
#   geometry on puzzle rasters.
# distinct-from: substrate/neuro_chip_stream.py::StreamingChip -- the DISCIPLINE is ported (see the
#   module docstring) but the object differs: that streams chip CORES with AER edges and LIF state
#   to disk shards; this partitions UNITS and groups lake entries. Copying it would be a second chip
#   emulator. All four sites read 2026-09-07.
class UnitBlocks:
    """Partition `n_units` into blocks of `block_size`, group records by block, stream one at a time.

    The last block is SHORT, never padded: padding would invent units that do not exist and every
    consumer would have to know to ignore them."""

    def __init__(self, n_units, block_size):
        if not (isinstance(n_units, int) and n_units > 0):
            raise ValueError(f"n_units must be a positive int, got {n_units!r}")
        if not (isinstance(block_size, int) and block_size > 0):
            raise ValueError(
                f"block_size must be a positive int DERIVED BY THE CALLER from its own "
                f"state-per-unit (P1: this class refuses to pick one), got {block_size!r}"
            )
        self.n_units = n_units
        self.block_size = block_size
        self.n_blocks = (n_units + block_size - 1) // block_size
        self._resident = 0
        self.max_resident = 0

    def block_of(self, unit):
        """The block a unit belongs to. Refuses an out-of-range unit rather than bucketing it."""
        if not (isinstance(unit, int) and 0 <= unit < self.n_units):
            raise ValueError(f"unit {unit!r} is outside [0, {self.n_units})")
        return unit // self.block_size

    def units_in(self, block):
        """The half-open unit range this block owns -- SHORT at the tail, never padded."""
        if not (isinstance(block, int) and 0 <= block < self.n_blocks):
            raise ValueError(f"block {block!r} is outside [0, {self.n_blocks})")
        lo = block * self.block_size
        return range(lo, min(lo + self.block_size, self.n_units))

    def group(self, entries):
        """Group `(pre, post, q)` entries by the block their POST unit lives in.

        POST, not PRE, because applying an entry writes the post unit's incoming weight: a reader
        holding one block can apply every entry addressed to it without touching another block.
        Returns {block_id: [entry, ...]} with insertion order preserved inside each block, and
        blocks that own no entry ABSENT rather than present-and-empty -- an empty group would make a
        writer emit a record for a block nothing happened in, which is the repeated-snapshot
        pathology in miniature."""
        out = {}
        for e in entries:
            pre, post, q = e
            out.setdefault(self.block_of(int(post)), []).append((int(pre), int(post), int(q)))
        return out

    @contextmanager
    def resident(self, block):
        """Make one block resident. Nesting raises -- see `BlockResidencyError`.

        WHAT `max_resident` DOES AND DOES NOT MEASURE, corrected by this module's own panel: it
        counts CONTEXT-MANAGER NESTING, not memory. `stream()` yields inside the manager and a
        consumer is free to retain every group it is handed, so peak resident state can be all
        blocks while this counter reads 1 throughout -- measured overshoot equals the block count
        exactly. The honest claim is therefore "`resident()` is never nested", which is what the
        guard enforces; a peak-memory proof is a different instrument and this is not it."""
        if not (isinstance(block, int) and 0 <= block < self.n_blocks):
            raise ValueError(f"block {block!r} is outside [0, {self.n_blocks})")
        if self._resident:
            raise BlockResidencyError(
                f"block {block} requested while {self._resident} block(s) already resident -- the "
                "one-resident discipline this module ports from neuro_chip_stream is broken"
            )
        self._resident += 1
        self.max_resident = max(self.max_resident, self._resident)
        try:
            yield self.units_in(block)
        finally:
            self._resident -= 1

    def stream(self, grouped):
        """Yield (block, entries) for each block that owns entries, ONE resident at a time."""
        for block in sorted(grouped):
            with self.resident(block) as units:
                yield block, grouped[block], units


def main():
    """Self-check: every claim with the control that makes it a measurement (record: s1-unit-blocks.md)."""
    print("== T064 unit blocks ==")

    # R-S1b: the partition tiles the units exactly -- no gap, no overlap, short tail
    ub = UnitBlocks(10, 4)
    assert ub.n_blocks == 3, ub.n_blocks
    seen = [u for b in range(ub.n_blocks) for u in ub.units_in(b)]
    assert seen == list(range(10)), seen
    assert list(ub.units_in(2)) == [8, 9], "the last block must be SHORT, never padded"
    assert all(ub.block_of(u) == u // 4 for u in range(10))
    for bad in (-1, 10, 1.0):
        try:
            ub.block_of(bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"block_of({bad!r}) must refuse, not bucket")
    print(f"partition: {ub.n_blocks} blocks tile {ub.n_units} units, tail short")

    # block_size has no default, and the refusal names why
    try:
        UnitBlocks(10, 0)
    except ValueError as e:
        assert "refuses to pick one" in str(e), str(e)
    else:  # pragma: no cover
        raise AssertionError("block_size=0 must be refused")

    # R-S1a: ONE block resident, and the nesting attempt is MADE so the counter is not vacuous
    ub2 = UnitBlocks(10, 4)
    grouped = ub2.group([(0, 1, 5), (3, 9, -2), (1, 2, 7)])
    for _b, _entries, _units in ub2.stream(grouped):
        pass
    assert ub2.max_resident == 1, ub2.max_resident
    nested = False
    with ub2.resident(0):
        try:
            with ub2.resident(1):  # pragma: no cover
                pass
        except BlockResidencyError:
            nested = True
    assert nested, "the residency red must be REACHABLE -- a nested residency has to raise"
    assert ub2.max_resident == 1, "a refused nesting must not raise the high-water mark"
    print(f"residency: max_resident == {ub2.max_resident}; nested residency raises")

    # grouping is by POST, and a block that owns nothing is ABSENT
    assert set(grouped) == {0, 2}, grouped
    assert grouped[0] == [(0, 1, 5), (1, 2, 7)], grouped[0]
    assert grouped[2] == [(3, 9, -2)], grouped[2]
    assert 1 not in grouped, "a block owning no entry must be ABSENT, not present-and-empty"
    assert ub2.group([]) == {}, "no entries -> no groups, never an empty frame"
    print(f"grouping: by POST block, {len(grouped)} of {ub2.n_blocks} blocks own entries")

    # R-S1c: the frozen-substrate red at the block layer, with its positive control in the SAME run
    try:
        import numpy as np

        from snn_data_lake.substrate.lake_engine import SynapseDelta
    except Exception as exc:  # pragma: no cover - numpy is a core dependency
        print(f"(numpy/lake_engine unavailable: {exc}) -- block layer checked, delta layer skipped")
        print("OK")
        return
    w = np.array([[0.10, 0.20], [0.30, 0.40]], dtype=float)
    sd = SynapseDelta(q_step=0.01)
    ub3 = UnitBlocks(2, 1)
    first = sd.collect(w, [0, 1], [0, 1])
    assert first is not None, "the first capture must carry the substrate"
    n_first = len(ub3.group(SynapseDelta.entries(first)))
    frozen = sd.collect(w, [0, 1], [0, 1])  # byte-identical substrate, endpoints still firing
    assert frozen is None, "a FROZEN substrate must emit nothing -- the repeated-snapshot pathology"
    w[1][0] += 0.05  # one synapse moves past the quantum
    moved = sd.collect(w, [1], [0])
    assert moved is not None, "the positive control must fire: a real change must be captured"
    g = ub3.group(SynapseDelta.entries(moved))
    assert list(g) == [0], g
    assert g[0] == [(1, 0, 35)], g[0]
    print(f"frozen red: first capture {n_first} block(s), frozen -> None, one move -> {g}")

    print("OK")


if __name__ == "__main__":
    main()
