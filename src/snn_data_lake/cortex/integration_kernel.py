"""The integration instrument on the KERNEL's port (specs/022 Phase 4 / C4, tasks.md T070;
pre-registration `specs/022-unified-brain-arc/evidence/t070-integration-kernel.md`, sections 1-2
frozen at f09adc4).

WHAT THIS IS. `cortex/integration_measure.py` computes a Perturbational-Complexity-style number -- the
causal footprint of a brief pulse (perturbed raster XOR same-noise baseline raster over the post-pulse
window), serialized time-major and scored by normalized LZ76 -- over a PRIVATE twelve-unit LIF net its
own class builds, and its docstring says so: THIS CLASS HAS NO PORT through which an external system's
state can enter, and a caller wanting to say something about the brain owes the port first and a
contrast second. This module is that port. Nothing about the SCORE changes: the footprint goes through
the incumbent's own `_serialize`, `normalized_lz` and `gzip_complexity`. Only the NETWORK that produces
the two rasters changes, and it is the tree's kernel -- `substrate/kernel_tick.KernelTick` over a
`substrate/kernel_state.KernelState` and a `Synapses` table that `cortex/h01_ingest.population` builds
from an artifact in H01's format (the harness fixture `prior_drift_harness.synthetic_artifact` for the
self-check; the H01 c3 population for the campaign).

THE TWO RUNS ARE TWO RESTORES FROM ONE SNAPSHOT (2.1, Codex objection (c) adopted). The baseline and
the perturbed run both resume from ONE `substrate/replay_gate.capture` taken after a warm-up on the
ORIGINAL run, so both carry the same in-flight spikes (delay rings, partition accumulators, tick_index)
and the same drive-stream POSITION -- a seed names a stream, a position names a point on it. A
frame-only restore that drops the tick runtime DIVERGES on the fixture (R3 (ii)), a positions-stripped
snapshot is REFUSED (R3 (iii)), and the restored baseline is byte-identical to the uninterrupted original
(R3 (i)). ONE snapshot mechanism, ONE PCI computation, ONE paired t (the harness's `contrast`): T070 adds
none of the three (P2).

THE FRAMING IS THE INCUMBENT'S AND DOES NOT RELAX: never Phi, never a consciousness threshold, never a
sentence about experience, never an aggregate with any other reading. ONE PCI NUMBER IS NOT REPORTABLE;
only CONTRASTS between regimes on ONE substrate are, and the number that reaches any surface is the
contrast INTEGRATED - PARTITIONED with its rate-matched control beside it, never an arm's PCI alone.
The conclusion registry (`substrate/conclusion_protocol.CONCLUSION["integration"]`) states what a pass
licenses and what it never does.

"LIVE" IS NOT DELIVERED, said plainly (pre-registration 2.7): no live population runs on the kernel --
the kernel wire was DECLINED at the deployed width (S6, `evidence/s6-kernel-wire.md`) and the kernel
rows are EXEMPT until the first tick population at or above N*. This instrument is live-CAPABLE (it
takes any KernelState/Synapses pair) and is read on the H01 fixture population OFF the tick. Nothing
here is a claim about the live brain's capability.

WHERE THE PORT LIVES, AND WHY NOT IN `integration_measure.py` (a DEVIATION from 2.1's file, dated
2026-09-10, the mechanism unchanged). The port was first written into `integration_measure.py`, and the
dormancy gate read 5 stale EXEMPT entries on the spot: `integration_measure` is reachable from a live
root (`substrate/region_coherence` imports its `normalized_lz` for the brain), the gate walks every
import unconditionally, and the port's imports would have made `kernel_tick`, `kernel_state`,
`replay_gate`, `h01_ingest` and `determinism_gate` "reachable" from the brain. Those rows are the record
that the kernel is NOT on the live tick -- `kernel_tick`'s row says in its own words "T070 is an
instrument" -- and retiring them to green the gate would have been the lie fact 7 of the
pre-registration refuses. The tree's idiom for an instrument that ticks a SCRATCH kernel population is
`cortex/prior_drift_harness`: a module of its own, EXEMPT with the reason stated, whose imports stale no
kernel row. This module takes that shape. The scorers stay where they are, live, in
`integration_measure`; the network that feeds them is here.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.integration_kernel   # (run_selfchecks.py)
"""

# distinct-from: substrate/replay_gate.py::run_gate -- G5's own four-arm battery on its own 48-unit
#   bench; this module CONSUMES capture/restore and never re-implements a snapshot.
# distinct-from: cortex/prior_drift_harness.py::run_trial -- drives a kernel from a per-trial RESET and
#   counts spikes for a readout; this resumes from a snapshot and differences two rasters.
# distinct-from: cortex/h01_ingest.py::drive_train -- a ONE-tick 1 % drive for the liveness gate; the
#   background here is a per-tick per-unit Gaussian current on every unit.
# distinct-from: cortex/integration_measure.py::PerturbationalComplexity._run -- the incumbent's
#   PRIVATE LIF net; this port is the mechanism that supersedes it (pre-registration 2.6: retired in its
#   own commit after the kernel self-check reproduces its orderings).
# distinct-from: substrate/region_coherence.py::RegionCoherence._integration -- a co-activation proxy
#   over LIVE region activity on the tick; no perturbation, no snapshot, no kernel.
#   All five sites read 2026-09-10 before this module was written.

from __future__ import annotations

import io
import json
import sys
from typing import NamedTuple

from snn_data_lake.cortex.integration_measure import (
    PerturbationalComplexity,
    gzip_complexity,
    normalized_lz,
)

DRIVE_STREAM = "drive"  # the ONE named RNG stream the snapshot carries (positions, not seeds)

# ---- CONSTANTS chosen BEFORE the pilot and disclosed as P1 debt (2.2) ---------------------------
WARM = 40  # warm-up ticks on the ORIGINAL run before the snapshot
STEPS = 80  # post-restore ticks
PULSE_T = 0  # the first post-restore tick
PULSE_LEN = 3  # ticks the pulse lasts
PULSE_CURRENT = 1.2  # x threshold -- the amplitude h01_ingest's own liveness driver uses
TRIALS = 8  # perturbation sets per seed
N_PERTURB = 3  # units per perturbation set

#: THE FIXTURE'S BACKGROUND DRIVE AMPLITUDE (x threshold), CHOSEN BY A DISCLOSED SWEEP (obs #2165)
#: on `synthetic_artifact(64, out_degree=8)` on 2026-09-10 and RE-MEASURED on the campaign population
#: (its rates and in-flight counts are pinned beside every number). The drive is a per-unit per-tick
#: Gaussian current, amp x thr x N(0, 1), rounded to the table's fixed point -- the incumbent's noise
#: shape on the kernel's port, and the one background whose firing rate is a SMOOTH function of the
#: amplitude (the rate-matched control bisects on it). Rule: the SMALLEST swept amplitude at which
#: (a) every self-check seed carries >= 1 in-flight spike at the capture tick, (b) the frame-only
#: restore DIVERGES on every self-check seed (R3 (ii)'s content, not just its precondition), and (c)
#: the INTEGRATED - PARTITIONED contrast clears the bar over the four self-check seeds.
#:   Coarse pass, seeds 0-1, one perturbation set: mean baseline rate per tick 0.000 at amp 0.10,
#:   0.006 at 0.20, 0.022 at 0.30, 0.037 at 0.40, 0.052 at 0.50, 0.063 at 0.60, 0.084 at 0.80,
#:   0.101 at 1.00, 0.115 at 1.20, 0.130 at 1.50, 0.148 at 2.00; the INTEGRATED footprint is sparse
#:   at every amplitude (1-51 ones of 4,928 bits) -- the population's fan-in-normalized weights make
#:   its coupling weak, a fact about the wiring and not a knob.
#:   The row's statistic, seeds 0-3, 8 perturbation sets, t(INTEGRATED - PARTITIONED) with the
#:   in-flight spike count at tick 40 per seed: amp 0.30 t +18.5 in-flight [0, 3, 2, 0] (fails a);
#:   0.40 t +5.78 [1, 5, 1, 1]; 0.50 t +4.92 [2, 6, 2, 1]; 0.60 t +4.65 [1, 7, 3, 3]; 0.70 t +6.52
#:   [3, 7, 9, 4].
#:   R3 (ii) at 0.40: the frame-only restore diverged on 4/4 seeds; at 0.70 seed 3 carried FOUR
#:   in-flight spikes and did NOT diverge -- a lost sub-threshold delivery washed out without
#:   flipping a crossing. So 2.4's "the fixture has no in-flight spikes" is not the only way (ii)
#:   fails to diverge, and rule (b) reads the divergence itself. 0.40 is the smallest that meets all
#:   three.
FIXTURE_AMP = 0.4


class Pulse(NamedTuple):
    """The perturbation: `units` (population indices) receive `current` x threshold for `length`
    ticks starting at post-restore tick `at`."""

    units: tuple
    at: int
    length: int
    current: float


class Capture(NamedTuple):
    """A snapshot after the warm-up, with the LIVE original kernel and stream so a caller can
    continue the ORIGINAL run (R3 (i)'s reference), and the spike count at the capture tick
    (R3 (ii)'s precondition, asserted rather than assumed)."""

    blob: bytes
    kt: object
    stream: object
    inflight: int


class Response(NamedTuple):
    """`footprint` = base XOR pert over the post-pulse window, both rasters beside it."""

    footprint: object
    base: object
    pert: object


def kernel_port(art, threads=1):
    """(Synapses, params) for a population in `h01_ingest`'s artifact format -- built ONCE per arm
    through the harness fixture's own constructor. `params` is what a FRESH kernel needs to be
    seated by `replay_gate.restore`: the faculty layout, the LIF constants, the threshold, the sign
    field (immutable at birth; the tick never reads it, the frame carries it) and the thread count.
    """
    from snn_data_lake.cortex.h01_ingest import population

    state, syn, kt = population(art, threads=threads)
    params = {
        "faculties": [(name, hi - lo) for name, (lo, hi) in state.ranges.items()],
        "decay": kt.decay,
        "v_reset": kt.v_reset,
        "refrac_ticks": kt.refrac_ticks,
        "thr": float(state.arrays["thr"].max()),
        "sign": state.arrays["sign"].copy(),
        "threads": int(threads),
    }
    return syn, params


def _fresh_kernel(synapses, params):
    from snn_data_lake.substrate.kernel_state import KernelState
    from snn_data_lake.substrate.kernel_tick import KernelTick

    state = KernelState(params["faculties"])
    state.arrays["thr"][:] = params["thr"]
    state.arrays["sign"][:] = params["sign"]
    return KernelTick(
        state,
        synapses,
        decay=params["decay"],
        v_reset=params["v_reset"],
        refrac_ticks=params["refrac_ticks"],
        threads=params.get("threads", 1),
    )


def _amp_q(synapses, params, amp):
    """amp x threshold in the table's fixed point (P1: quantized with the table's own frac_bits)."""
    return float(amp) * float(params["thr"]) * float(1 << int(synapses.frac_bits))


def background_drive(stream, n_units, amp_q):
    """THE ONE drive-drawing function, used by the warm-up and by both restores: per-unit per-tick
    Gaussian current amp_q x N(0, 1), rounded to fixed-point int64 over the index space."""
    import numpy as np

    return np.rint(stream.standard_normal(n_units) * amp_q).astype(np.int64)


def warm_capture(synapses, params, seed, amp, warm=WARM):
    """The ORIGINAL run: a fresh kernel, `warm` ticks of background drive from the named stream
    seeded by `seed`, then ONE `replay_gate.capture` of the kernel with the stream's POSITION."""
    import numpy as np

    from snn_data_lake.substrate.replay_gate import capture

    kt = _fresh_kernel(synapses, params)
    stream = np.random.default_rng(int(seed))
    amp_q = _amp_q(synapses, params, amp)
    fired = np.zeros(0, dtype=np.int64)
    for _ in range(int(warm)):
        fired = kt.tick(background_drive(stream, kt.state.n_units, amp_q))
    return Capture(capture(kt, {DRIVE_STREAM: stream}), kt, stream, int(fired.size))


def _continue(kt, stream, amp_q, steps, pulse=None, pulse_q=0):
    """`steps` ticks from wherever `kt` stands, drive drawn from `stream`; the pulse, when given,
    lands on its units during [at, at + length). Returns the (steps, n_units) int8 raster."""
    import numpy as np

    n = kt.state.n_units
    rows = np.zeros((int(steps), n), dtype=np.int8)
    units = np.asarray(pulse.units, dtype=np.int64) if pulse is not None else None
    for t in range(int(steps)):
        ext = background_drive(stream, n, amp_q)
        if units is not None and pulse.at <= t < pulse.at + pulse.length:
            ext[units] += int(pulse_q)
        rows[t, kt.tick(ext)] = 1
    return rows


def kernel_causal_response(blob, synapses, params, drive, pulse, steps=STEPS):
    """The CAUSAL footprint of one perturbation on the kernel (2.1): TWO restores from ONE snapshot,
    BASELINE consuming the captured drive stream as captured, PERTURBED consuming the same stream
    plus `pulse` (quantized with the table's frac_bits); footprint = base XOR pert over
    [pulse.at + pulse.length, steps). `drive` is the background amplitude (x threshold). A snapshot
    the gate refuses (no positions, a parameter mismatch) raises `SnapshotRefused` -- the port
    never runs on a restore it cannot prove."""
    import numpy as np

    from snn_data_lake.substrate.replay_gate import restore

    amp_q = _amp_q(synapses, params, drive)
    pulse_q = int(round(_amp_q(synapses, params, pulse.current)))
    base_kt = _fresh_kernel(synapses, params)
    base = _continue(base_kt, restore(blob, base_kt)[DRIVE_STREAM], amp_q, steps)
    pert_kt = _fresh_kernel(synapses, params)
    pert = _continue(pert_kt, restore(blob, pert_kt)[DRIVE_STREAM], amp_q, steps, pulse, pulse_q)
    w0 = int(pulse.at + pulse.length)
    footprint = (base[w0:] != pert[w0:]).astype(np.int8)
    return Response(footprint, base, pert)


def frame_only_continuation(blob, synapses, params, drive, steps=STEPS):
    """R3 (ii)'s CONTROL, and not the method: seat ONLY the KernelState frame from the snapshot into
    a fresh kernel -- rings, accumulators and tick_index DROPPED -- with the drive stream at its
    captured position, and continue. This must DIVERGE from the original run; if it does not, the
    snapshot chain was a reset on that fixture and the fixture is wrong, not the chain."""
    import numpy as np

    from snn_data_lake.substrate.kernel_state import KernelState
    from snn_data_lake.substrate.kernel_tick import KernelTick
    from snn_data_lake.substrate.replay_gate import SnapshotRefused

    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        state = KernelState.from_blob(z["state_blob"].tobytes())
        position = json.loads(str(z["rng_positions"]))[DRIVE_STREAM]
    if state is None:
        raise SnapshotRefused("embedded KernelState frame refused (see kernel_state)")
    kt = KernelTick(
        state,
        synapses,
        decay=params["decay"],
        v_reset=params["v_reset"],
        refrac_ticks=params["refrac_ticks"],
        threads=params.get("threads", 1),
    )
    stream = np.random.default_rng()
    stream.bit_generator.state = position
    return _continue(kt, stream, _amp_q(synapses, params, drive), steps)


def score_footprint(footprint):
    """(normalized LZ76, gzip bytes, ones) of a footprint through the incumbent's own serializer
    and scorers -- the score is unchanged by the port."""
    s = PerturbationalComplexity._serialize(footprint.tolist())
    return normalized_lz(s), gzip_complexity(s), s.count("1")


# =============================================================================================
# The self-check (run_selfchecks CORE): R3 (i)(ii)(iii) and the pulse-less identity, in one command
# =============================================================================================
def _strip_positions(blob):
    import numpy as np

    with io.BytesIO(blob) as f:
        z = np.load(f, allow_pickle=False)
        kept = {k: z[k] for k in z.files if k != "rng_positions"}
    buf = io.BytesIO()
    np.savez(buf, **kept)
    return buf.getvalue()


def chain_check(art, seeds=(0, 1, 2, 3), amp=FIXTURE_AMP):
    """R3 on the fixture, per seed: (i) the restored baseline equals the original continuation;
    (ii) the frame-only restore diverges, with the in-flight count beside it; (iii) a
    positions-stripped snapshot is refused; and 2.5(c), the pulse-less identity -- with the pulse
    current 0 the two restores' rasters are byte-identical and the footprint is empty."""
    from snn_data_lake.substrate.replay_gate import SnapshotRefused

    syn, params = kernel_port(art)
    n = syn.n_units
    out = []
    for seed in seeds:
        cap = warm_capture(syn, params, seed, amp)
        original = _continue(cap.kt, cap.stream, _amp_q(syn, params, amp), STEPS)
        pulse = Pulse((0, 1, 2), PULSE_T, PULSE_LEN, PULSE_CURRENT)
        resp = kernel_causal_response(cap.blob, syn, params, amp, pulse)
        frame = frame_only_continuation(cap.blob, syn, params, amp, STEPS)
        try:
            kernel_causal_response(_strip_positions(cap.blob), syn, params, amp, pulse)
            refused = False
        except SnapshotRefused:
            refused = True
        none = kernel_causal_response(
            cap.blob, syn, params, amp, Pulse((0, 1, 2), PULSE_T, PULSE_LEN, 0.0)
        )
        out.append(
            {
                "seed": int(seed),
                "inflight": cap.inflight,
                "identical": bool(resp.base.tobytes() == original.tobytes()),
                "frame_only_diverged": bool(frame.tobytes() != original.tobytes()),
                "omit_refused": bool(refused),
                "pulseless_identity": bool(
                    none.base.tobytes() == none.pert.tobytes() and int(none.footprint.sum()) == 0
                ),
                "n_units": int(n),
            }
        )
    return out


def main(argv=None) -> int:
    from snn_data_lake.cortex.prior_drift_harness import synthetic_artifact
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        rows = chain_check(synthetic_artifact(64, out_degree=8))
    bad = []
    for r in rows:
        line = (
            f"R3 seed {r['seed']}: in-flight {r['inflight']:2d}; restored baseline "
            f"{'BYTE-IDENTICAL' if r['identical'] else 'DIVERGED (defect)'}; frame-only restore "
            f"{'diverged as it must' if r['frame_only_diverged'] else 'DID NOT DIVERGE (the fixture is wrong, not the chain)'}; "
            f"omit-positions {'REFUSED' if r['omit_refused'] else 'ACCEPTED (defect)'}; "
            f"pulse-less identity {'holds' if r['pulseless_identity'] else 'BROKEN (drive leaked)'}"
        )
        print(line)
        if not (
            r["inflight"] > 0
            and r["identical"]
            and r["frame_only_diverged"]
            and r["omit_refused"]
            and r["pulseless_identity"]
        ):
            bad.append(r["seed"])
    if bad:
        print(f"integration_kernel self-check FAILED on seeds {bad}")
        return 1
    print(
        f"integration_kernel self-check OK: G5 chain through the port on {len(rows)} seeds of the "
        f"{rows[0]['n_units']}-unit fixture -- restored baseline byte-identical, frame-only restore "
        "diverged, positions-stripped snapshot refused, pulse-less identity holds. NOT consciousness, "
        "NOT IIT Phi; no live population runs on the kernel."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
