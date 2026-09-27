"""G5 replay gate (specs/022 T017, FR-007, gates.md 'G5 replay gate' row): a snapshot record
class that captures per-instance RNG stream POSITIONS, a restore that ASSERTS stream-position
equality and refuses a snapshot without positions, and the replay criterion -- the continuation
from a restored snapshot is byte-identical to the original run's continuation.

WHY POSITIONS AND NOT SEEDS: a seed names a STREAM, a position names a POINT on it. A restore
that re-seeds replays the stream from the beginning -- every random draw after the snapshot
lands differently than the original run's continuation, so the restored brain quietly walks a
different trajectory while looking perfectly healthy. The omit-positions red is therefore not
decoration: restore REFUSES a snapshot that lacks them, and the self-check demonstrates the
divergence the refusal prevents (a fresh-seeded restore's trains MUST differ -- the control
that keeps the equality assert from being vacuous).

WHAT A SNAPSHOT HOLDS at S0: the KernelState frame (reused whole -- one serialization
mechanism, P2), the tick runtime (tick_index, per-partition delay-ring slot contents,
per-partition integer accumulators), the kernel parameters (asserted against the caller's
kernel at restore -- a snapshot applied to a differently-parameterized kernel is a different
brain wearing the same state), and every registered RNG stream's bit-generator state dict
(numpy Generator) VERBATIM. Synapse topology is deliberately NOT in the snapshot: the lake
owns it (0x09 SynapseDelta at the kernel cutover); the caller supplies the same table and the
parameter assert catches a mismatch in shape.

L7's whole battery consumes THIS gate (SYNTHESIS L1): every perturbation instrument replays
from a snapshot rather than re-running from t=0, so replay-equality -- not just thread-count
invariance -- is the determinism claim the tree actually leans on.
"""

# distinct-from: gates/determinism_gate.py -- thread-count invariance at fixed P; this is
#   trajectory-resume equality (the two are the two halves of D6).
# distinct-from: substrate/kernel_state.py -- the state FRAME this snapshot embeds whole.
# distinct-from: growth/neurogenesis_engine.py growth blobs and substrate/home.py chain --
#   durable substrate persistence in the lake; a snapshot is a REPLAY instrument.
# distinct-from: hippocampus/experience_replay.py and sleep_replay.py -- learned-content
#   replay (memories), not execution-state resume. All sites read.

from __future__ import annotations

import io
import json
import sys

SNAPSHOT_VERSION = 1


class SnapshotRefused(RuntimeError):
    """A snapshot restore that cannot prove it resumes THE SAME trajectory refuses whole."""


#: THE PAGED-KERNEL REFUSAL IS DELETED (2026-09-09, T068 LOSS). It guarded a `KernelTick` whose
#: `resident` mask was not None -- a state that can no longer be constructed: the paging layer was
#: measured at a 35.2x median tick cost against a 3.0x bound fixed before the run and deleted on
#: its null (specs/022 evidence/s5-paging-null.md 3.4/3.5), and `KernelTick` no longer takes a
#: `resident` parameter or allocates `pending`. The criterion therefore has NO REACHABLE RED,
#: which P8 clause 1 (F11) settles: "any success criterion that turns out to have no reachable
#: red is DELETED, not weakened." Deleted rather than left standing, because a guard that cannot
#: fire still reads as live protection to the next reader -- and measured by the panel, leaving it
#: in place keeps `replay_gate --assert` green, so no gate would ever have reported it. What the
#: guard once caught is preserved as a fact a successor row inherits, not as dead code: 262,144
#: fixed-point units sitting in `pending` at capture read back as ZERO after restore, and after an
#: identical page-in the live kernel fired [8, 9] while the restored one fired nothing. A
#: successor paging row owes a snapshot format that carries residency and pending, with
#: SNAPSHOT_VERSION bumped and the synapse table's provenance.


def capture(kt, rng_streams):
    """One snapshot blob from a live KernelTick + its named RNG streams
    ({name: np.random.Generator}). Positions are the bit-generator state dicts, verbatim."""
    import numpy as np

    rings = [[[list(key) for key in slot] for slot in part_rings] for part_rings in kt._rings]
    buf = io.BytesIO()
    arrays = {f"acc_{p}": kt._acc[p] for p in range(kt.p_parts)}
    np.savez(
        buf,
        version=np.array(SNAPSHOT_VERSION),
        state_blob=np.frombuffer(kt.state.to_blob(), dtype=np.uint8),
        tick_index=np.array(int(kt.tick_index)),
        params=np.array(
            json.dumps(
                {
                    "p_parts": kt.p_parts,
                    "frac_bits": kt.syn.frac_bits,
                    "max_delay": kt.syn.max_delay,
                    "decay": kt.decay,
                    "v_reset": kt.v_reset,
                    "refrac_ticks": kt.refrac_ticks,
                    "n_units": kt.state.n_units,
                }
            )
        ),
        rings=np.array(json.dumps(rings)),
        rng_positions=np.array(
            json.dumps({name: gen.bit_generator.state for name, gen in rng_streams.items()})
        ),
        **arrays,
    )
    return buf.getvalue()


def restore(blob, kt):
    """Resume `kt` (a freshly built kernel over the SAME synapse table and parameters) from a
    snapshot. Returns {name: np.random.Generator} rebuilt AT the captured positions.

    THE ASSERTS ARE THE GATE (FR-007): a snapshot without rng_positions REFUSES; a parameter
    mismatch REFUSES (same state on a different kernel is a different brain); and after seating
    each stream, its bit-generator state is compared to the captured dict and any inequality
    REFUSES -- stream-position equality is checked, never assumed."""
    import numpy as np

    from snn_data_lake.substrate.kernel_state import KernelState

    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            if int(z["version"]) != SNAPSHOT_VERSION:
                raise SnapshotRefused(
                    f"snapshot version {int(z['version'])} is not {SNAPSHOT_VERSION}"
                )
            names = set(z.files)
            if "rng_positions" not in names:
                raise SnapshotRefused(
                    "snapshot carries NO rng stream positions -- restoring it would re-seed, "
                    "and a re-seeded continuation silently walks a different trajectory (G5's "
                    "whole point). Refused whole."
                )
            params = json.loads(str(z["params"]))
            state_blob = z["state_blob"].tobytes()
            tick_index = int(z["tick_index"])
            rings = json.loads(str(z["rings"]))
            positions = json.loads(str(z["rng_positions"]))
            acc = {p: z[f"acc_{p}"].copy() for p in range(params["p_parts"])}
    except SnapshotRefused:
        raise
    except Exception as exc:
        raise SnapshotRefused(f"snapshot unreadable: {exc}") from exc

    have = {
        "p_parts": kt.p_parts,
        "frac_bits": kt.syn.frac_bits,
        "max_delay": kt.syn.max_delay,
        "decay": kt.decay,
        "v_reset": kt.v_reset,
        "refrac_ticks": kt.refrac_ticks,
        "n_units": kt.state.n_units,
    }
    if have != params:
        raise SnapshotRefused(
            f"kernel/snapshot parameter mismatch: snapshot {params} vs kernel {have} -- the "
            "same state on a differently-parameterized kernel is a different brain"
        )
    state = KernelState.from_blob(state_blob)
    if state is None:
        raise SnapshotRefused("embedded KernelState frame refused (see kernel_state)")
    for field, arr in state.arrays.items():
        kt.state.arrays[field][:] = arr
    kt.tick_index = tick_index
    for p in range(kt.p_parts):
        kt._acc[p][:] = acc[p]
        for slot_ix, slot in enumerate(rings[p]):
            kt._rings[p][slot_ix] = [tuple(key) for key in slot]

    streams = {}
    for name, pos in positions.items():
        gen = np.random.default_rng()
        gen.bit_generator.state = pos
        if gen.bit_generator.state != pos:
            raise SnapshotRefused(
                f"rng stream {name!r} did not seat at its captured position -- "
                "stream-position equality is the assert, not the hope"
            )
        streams[name] = gen
    return streams


# ---- the gate's own battery (bare = cheap wiring; --assert/--self-check = full) -------------
def _driven_run(kt, rng, n_ticks, n_units, frac_bits):
    """n_ticks of an RNG-driven stimulus -- the randomness G5 exists to resume correctly."""
    import numpy as np

    blobs = []
    for _ in range(n_ticks):
        ext = np.zeros(n_units, dtype=np.int64)
        ix = rng.integers(0, n_units, size=6)
        ext[ix] = int(1.4 * (1 << frac_bits))
        blobs.append(kt.tick(ext).tobytes())
    return b"".join(blobs)


def _build(seed=23, n=48, edges=384, p_parts=4, frac_bits=12):
    import numpy as np

    from snn_data_lake.substrate.kernel_state import KernelState
    from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses

    top = np.random.default_rng(seed)
    state = KernelState([("net", n)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses(
        top.integers(0, n, edges),
        top.integers(0, n, edges),
        top.normal(0.4, 0.2, edges),
        top.integers(1, 6, edges),
        n,
        p_parts,
        frac_bits,
    )
    return state, syn, KernelTick(state, syn, decay=0.8, v_reset=0.0, refrac_ticks=2)


def run_gate(pre_ticks=15, post_ticks=25):
    """-> dict of the four arms' outcomes. The criterion is arm 1; arms 2-4 are its reds."""
    import numpy as np

    from snn_data_lake.substrate.kernel_tick import KernelTick

    n, frac_bits = 48, 12
    _state, syn, kt = _build()
    drive = np.random.default_rng(99)
    _warm = _driven_run(kt, drive, pre_ticks, n, frac_bits)
    snap = capture(kt, {"drive": drive})
    original_tail = _driven_run(kt, drive, post_ticks, n, frac_bits)

    # arm 1: restore into a FRESH kernel -> positions seat equal, continuation byte-identical
    state2, _syn2, kt2 = _build()
    del state2
    streams = restore(snap, kt2)
    replay_tail = _driven_run(kt2, streams["drive"], post_ticks, n, frac_bits)
    replay_equal = replay_tail == original_tail

    # arm 2: omit-positions red -- a snapshot stripped of rng_positions REFUSES
    with io.BytesIO(snap) as f:
        z = np.load(f, allow_pickle=False)
        kept = {k: z[k] for k in z.files if k != "rng_positions"}
    buf = io.BytesIO()
    np.savez(buf, **kept)
    try:
        _state3, _syn3, kt3 = _build()
        restore(buf.getvalue(), kt3)
        omit_refused = False
    except SnapshotRefused:
        omit_refused = True

    # arm 3: the divergence the refusal prevents -- re-SEEDED (not positioned) drive differs
    _state4, _syn4, kt4 = _build()
    restore(snap, kt4)  # state seated correctly; the drive stream is the variable
    fresh_tail = _driven_run(kt4, np.random.default_rng(99), post_ticks, n, frac_bits)
    reseed_diverged = fresh_tail != original_tail

    # arm 4: parameter-mismatch red -- same snapshot, differently-parameterized kernel
    _state5, syn5, _kt5 = _build()
    try:
        restore(snap, KernelTick(_state5, syn5, decay=0.7, v_reset=0.0, refrac_ticks=2))
        mismatch_refused = False
    except SnapshotRefused:
        mismatch_refused = True

    return {
        "replay_equal": replay_equal,
        "omit_refused": omit_refused,
        "reseed_diverged": reseed_diverged,
        "mismatch_refused": mismatch_refused,
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    if mode in ("--assert", "--self-check"):
        got = run_gate()
        ok = all(got.values())
        print(
            f"G5 replay gate [{mode[2:]}] "
            f"continuation {'BYTE-IDENTICAL' if got['replay_equal'] else 'DIVERGED'}; "
            f"omit-positions {'REFUSED' if got['omit_refused'] else 'ACCEPTED (defect)'}; "
            f"re-seeded control {'diverged as it must' if got['reseed_diverged'] else 'FAILED TO DIVERGE (vacuous equality)'}; "
            f"parameter mismatch {'REFUSED' if got['mismatch_refused'] else 'ACCEPTED (defect)'}"
        )
        return 0 if ok else 1
    # bare (run_selfchecks): the same four arms at reduced tick counts -- cheap but never
    # criterion-free (a wiring check that skips the reds is a wiring check for a different gate)
    got = run_gate(pre_ticks=6, post_ticks=8)
    if not all(got.values()):
        print(f"replay_gate wiring-check FAILED: {got}")
        return 1
    print(
        "replay_gate wiring-check OK: snapshot->restore continuation byte-identical; "
        "omit-positions refused; re-seeded control diverged; parameter mismatch refused"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
