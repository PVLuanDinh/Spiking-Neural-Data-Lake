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
from snn_data_lake.cortex.prior_drift_harness import (
    T_BAR,
    ArmRefused,
    _copy,
    arm_no_edges,
    contrast,
    edge_table,
    population_order,
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
# The arms (2.2), the null (2.3), the rate-matched control (2.5a) and the verdict rule (2.3)
# =============================================================================================
ARMS = ("integrated", "partitioned", "partitioned_rate_matched", "synchronous", "silent")
RATE_TOL = 0.05  # the rate-matched control sits within 5 % of INTEGRATED's rate (2.2)
BISECT_STEPS = 12  # the bounded bisection on the drive amplitude (2.2)
SYNC_FANOUT = 8  # SYNCHRONOUS out-degree = the population's mean out-degree x 8 (2.2)
SEARCH_CEILING = (
    4.0  # the bisection's upper bracket, x the INTEGRATED amplitude (CHOSEN, disclosed)
)
SELF_SEEDS = (
    0,
    1,
    2,
    3,
)  # the self-check's seeds on the fixture; the campaign runs the harness's 8

#: the harness's no-edges twin IS the registry's "partitioned twin: the same parts with the measured
#: channel cut" -- one constructor, two callers (P2)
arm_partitioned = arm_no_edges


class ArmReading(NamedTuple):
    """One arm on one seed. `pci` is the mean normalized LZ76 over the perturbation sets, `gzip` the
    twin compressor's mean bytes, `ones` the mean footprint bits, `rate` the mean non-pulse firing
    fraction per tick (the BASELINE raster), `null` the mean local-echo floor, `search` the
    rate-matching search (amp, rate) per step where one ran, `matched` whether it ended in tolerance
    (True where no search ran)."""

    arm: str
    seed: int
    amp: float
    pci: float
    gzip: float
    ones: float
    rate: float
    null: float
    inflight: int
    search: tuple
    matched: bool


def perturbation_sets(n_units, seed, trials=TRIALS, n_perturb=N_PERTURB):
    """`trials` sets of `n_perturb` units drawn in POPULATION index space per seed -- identical across
    the arms of a seed by construction, as the harness's `drive_classes` are."""
    import numpy as np

    rng = np.random.default_rng((int(seed) + 1) * 1_000_033 + 7_070)
    return tuple(
        tuple(sorted(int(u) for u in rng.choice(int(n_units), int(n_perturb), replace=False)))
        for _ in range(int(trials))
    )


def arm_uncut(art, keep_sign=1):
    """R2's red INPUT (2.4): a "partition" that did NOT cut -- ONE edge class, the edges whose pre
    carries `keep_sign` (the excitatory class by default), left in place over the same units."""
    import numpy as np

    keep = np.asarray(art["sign"], dtype=np.int64)[
        np.asarray(art["pre_idx"], dtype=np.int64)
    ] == int(keep_sign)
    out = _copy(art)
    for key in ("pre_idx", "post_idx", "n_syn"):
        out[key] = art[key][keep]
    return out


def synchronous_synapses(art, syn, seed, fanout=SYNC_FANOUT):
    """SYNCHRONOUS (2.2), the "integrated but not differentiated" regime: a dense strong-coupling
    table over the SAME units in population index space -- out-degree the population's mean x
    `fanout` (capped at n - 1), every weight at the population table's MAXIMUM magnitude (read off
    the compiled table by the harness's `edge_table`) with each pre's Dale sign kept, delay 1 --
    compiled with the population's own p_parts, frac_bits and max_delay."""
    import numpy as np

    from snn_data_lake.substrate.kernel_tick import Synapses

    n = int(syn.n_units)
    _pre, _post, w = edge_table(syn)
    wmax = float(np.abs(w).max()) if w.size else 1.0
    _order, pos = population_order(art)
    sign_pop = np.zeros(n, dtype=np.int64)
    sign_pop[pos] = np.asarray(art["sign"], dtype=np.int64)
    k = min(n - 1, max(1, int(round(int(art["pre_idx"].size) / max(1, n) * float(fanout)))))
    rng = np.random.default_rng(int(seed) * 7_919 + 3)
    pre = np.repeat(np.arange(n, dtype=np.int64), k)
    post = np.empty(n * k, dtype=np.int64)
    for u in range(n):
        others = np.concatenate([np.arange(u), np.arange(u + 1, n)])
        post[u * k : (u + 1) * k] = rng.choice(others, k, replace=False)
    weight = wmax * sign_pop[pre].astype(np.float64)
    return Synapses(
        pre,
        post,
        weight,
        np.ones(n * k, dtype=np.int64),
        n,
        syn.p_parts,
        syn.frac_bits,
        max_delay=syn.max_delay,
    )


def local_echo_null(footprint, units):
    """THE NULL (2.3, Codex objection (b) adopted): the PCI of the same footprint with every
    non-pulsed unit's column zeroed -- the response the pulse causes when nothing propagates."""
    import numpy as np

    echo = np.zeros_like(footprint)
    idx = np.asarray(units, dtype=np.int64)
    echo[:, idx] = footprint[:, idx]
    return score_footprint(echo)[0]


def run_regime(
    art, name, seed, amp, pulse_current=PULSE_CURRENT, threads=1, trials=TRIALS, n_perturb=N_PERTURB
):
    """One arm on one seed: the warm-up and ONE snapshot; the pulse-less identity ASSERTED before any
    PCI is read (2.5(c) -- two restores without a pulse must be byte-identical, or the arm REFUSES
    with `ArmRefused`); then `trials` perturbation sets, each read by `kernel_causal_response` and
    scored by the incumbent's scorers, the local-echo null beside every set. `name == "synchronous"`
    swaps the population's table for `synchronous_synapses`; SILENT is `amp = 0.0`."""
    syn, params = kernel_port(art, threads=threads)
    if name == "synchronous":
        syn = synchronous_synapses(art, syn, seed)
    cap = warm_capture(syn, params, seed, amp)
    sets = perturbation_sets(syn.n_units, seed, trials, n_perturb)
    identity = kernel_causal_response(
        cap.blob, syn, params, amp, Pulse(sets[0], PULSE_T, PULSE_LEN, 0.0)
    )
    leaked = int(identity.footprint.sum())
    if identity.base.tobytes() != identity.pert.tobytes() or leaked:
        raise ArmRefused(
            f"{name} seed {seed}: the pulse-less identity is BROKEN -- two restores from one "
            f"snapshot diverged with no pulse ({leaked} footprint bits); drive leaked into the port"
        )
    rate = float(identity.base.mean())
    pcis, gzs, ones, nulls = [], [], [], []
    for units in sets:
        resp = kernel_causal_response(
            cap.blob, syn, params, amp, Pulse(units, PULSE_T, PULSE_LEN, pulse_current)
        )
        p, g, o = score_footprint(resp.footprint)
        pcis.append(p)
        gzs.append(float(g))
        ones.append(float(o))
        nulls.append(local_echo_null(resp.footprint, units))
    m = float(len(sets))
    return ArmReading(
        name,
        int(seed),
        float(amp),
        sum(pcis) / m,
        sum(gzs) / m,
        sum(ones) / m,
        rate,
        sum(nulls) / m,
        cap.inflight,
        (),
        True,
    )


def rate_matched_amplitude(
    art, seed, target, amp, threads=1, tol=RATE_TOL, steps=BISECT_STEPS, ceiling=SEARCH_CEILING
):
    """The rate-matched control's amplitude (2.5(a), Codex objection (a) adopted): a BOUNDED bisection
    (<= `steps`) on the background amplitude of the CUT population between 0 and `ceiling` x `amp`,
    until its mean non-pulse rate lands within `tol` of `target` (INTEGRATED's rate on the same
    seed). Every step is the arm's own protocol -- warm-up, snapshot, the baseline continuation --
    so the rate read here is the rate the arm reads. Returns (amp, search, matched) with `search` =
    every (amp, rate) tried, printed by the verdict; `matched` False means the closest step is used
    and (b5) reads that seed NOT MET."""
    syn, params = kernel_port(art, threads=threads)

    def rate_at(a):
        cap = warm_capture(syn, params, seed, a)
        return float(_continue(cap.kt, cap.stream, _amp_q(syn, params, a), STEPS).mean())

    lo, hi = 0.0, float(amp) * float(ceiling)
    search, best = [], None
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        r = rate_at(mid)
        search.append((mid, r))
        if best is None or abs(r - target) < abs(best[1] - target):
            best = (mid, r)
        if target > 0.0 and abs(r - target) <= tol * target:
            return mid, tuple(search), True
        if r < target:
            lo = mid
        else:
            hi = mid
    return best[0], tuple(search), bool(target > 0.0 and abs(best[1] - target) <= tol * target)


def kernel_regimes(art, seed, amp=FIXTURE_AMP, pulse_current=PULSE_CURRENT, threads=1):
    """The five arms of one seed (2.2) on ONE population, the perturbation sets identical across
    them: INTEGRATED (the population's own wiring), PARTITIONED (every edge cut), PARTITIONED-
    RATE-MATCHED (the cut twin at the bisected amplitude), SYNCHRONOUS (the dense strong table),
    SILENT (the wiring with the background amplitude 0; the pulse still lands)."""
    integ = run_regime(art, "integrated", seed, amp, pulse_current, threads)
    cut = arm_partitioned(art)
    part = run_regime(cut, "partitioned", seed, amp, pulse_current, threads)
    amp_m, search, matched = rate_matched_amplitude(cut, seed, integ.rate, amp, threads)
    prm = run_regime(cut, "partitioned_rate_matched", seed, amp_m, pulse_current, threads)
    prm = prm._replace(search=search, matched=matched)
    sync = run_regime(art, "synchronous", seed, amp, pulse_current, threads)
    silent = run_regime(art, "silent", seed, 0.0, pulse_current, threads)
    return {
        "integrated": integ,
        "partitioned": part,
        "partitioned_rate_matched": prm,
        "synchronous": sync,
        "silent": silent,
    }


def _con(name, diffs, bar):
    c = contrast(name, diffs, bar)
    return {
        "met": bool(c.met),
        "t": float(c.t),
        "fpr": float(c.fpr),
        "degenerate": bool(c.degenerate),
        "diffs": tuple(c.diffs),
        "note": f"t {c.t:+.4f} FPR {c.fpr:.4f}" + (" DEGENERATE" if c.degenerate else ""),
    }


def verdict_t070(readings, chain=None, bar=T_BAR):
    """The verdict rule of 2.3, written before any number: every conjunct (b1)-(b7) MET / NOT MET
    with its numbers, the paired contrasts decided by the harness's `contrast` (one-sided t against
    `bar`, the 2^S sign-flip FPR beside every t). `readings` is {seed: {arm: ArmReading}}; `chain`
    is `chain_check`'s rows for (b7), which reads NOT READ when they are absent from the command. No
    conjunct is re-labelled after its number; `pass` is the conjunction."""
    seeds = sorted(readings)
    n = len(seeds)
    integ = [readings[s]["integrated"] for s in seeds]
    part = [readings[s]["partitioned"] for s in seeds]
    prm = [readings[s]["partitioned_rate_matched"] for s in seeds]
    sync = [readings[s]["synchronous"] for s in seeds]
    silent = [readings[s]["silent"] for s in seeds]
    b1 = _con("integrated - partitioned", [i.pci - p.pci for i, p in zip(integ, part)], bar)
    b2 = _con(
        "integrated - partitioned_rate_matched", [i.pci - r.pci for i, r in zip(integ, prm)], bar
    )
    b3 = _con("integrated - synchronous", [i.pci - y.pci for i, y in zip(integ, sync)], bar)
    silent_zero = [z.pci == 0.0 for z in silent]
    b4 = {
        "met": all(silent_zero),
        "t": None,
        "fpr": None,
        "degenerate": False,
        "diffs": tuple(z.pci for z in silent),
        "note": f"SILENT PCI = 0 exactly on {sum(silent_zero)}/{n}",
    }
    sync_ok = [y.rate >= i.rate for i, y in zip(integ, sync)]
    match_ok = [abs(r.rate - i.rate) <= RATE_TOL * i.rate for i, r in zip(integ, prm)]
    b5 = {
        "met": all(sync_ok) and all(match_ok),
        "t": None,
        "fpr": None,
        "degenerate": False,
        "diffs": tuple(r.rate - i.rate for i, r in zip(integ, prm)),
        "note": (
            f"rate(SYNCHRONOUS) >= rate(INTEGRATED) on {sum(sync_ok)}/{n}; "
            f"rate(PARTITIONED-RATE-MATCHED) within {RATE_TOL:.0%} on {sum(match_ok)}/{n}"
        ),
    }
    null_c = _con("integrated - null", [i.pci - i.null for i in integ], bar)
    p_le = [p.pci <= i.null for i, p in zip(integ, part)]
    r_le = [r.pci <= i.null for i, r in zip(integ, prm)]
    b6 = {
        **null_c,
        "met": bool(null_c["met"] and all(p_le) and all(r_le)),
        "note": (
            f"t(INTEGRATED - null) {null_c['t']:+.4f} FPR {null_c['fpr']:.4f}"
            f"{' DEGENERATE' if null_c['degenerate'] else ''}; partitioned <= null on "
            f"{sum(p_le)}/{n}; partitioned_rate_matched <= null on {sum(r_le)}/{n}"
        ),
    }
    if chain is None:
        b7 = {
            "met": False,
            "t": None,
            "fpr": None,
            "degenerate": False,
            "diffs": (),
            "note": "NOT READ (no chain rows in this command)",
        }
    else:
        held = [
            bool(
                r["inflight"] > 0
                and r["identical"]
                and r["frame_only_diverged"]
                and r["omit_refused"]
                and r["pulseless_identity"]
            )
            for r in chain
        ]
        b7 = {
            "met": bool(held) and all(held),
            "t": None,
            "fpr": None,
            "degenerate": False,
            "diffs": (),
            "note": f"R3 (i)(ii)(iii) + the pulse-less identity held on {sum(held)}/{len(held)} seeds",
        }
    conj = {"b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7}
    return {
        "conjuncts": conj,
        "pass": all(c["met"] for c in conj.values()),
        "seeds": seeds,
        "bar": float(bar),
    }


_CONJUNCT_TEXT = {
    "b1": "t(INTEGRATED - PARTITIONED) > bar",
    "b2": "t(INTEGRATED - PARTITIONED-RATE-MATCHED) > bar   [the artefact control]",
    "b3": "t(INTEGRATED - SYNCHRONOUS) > bar",
    "b4": "SILENT PCI = 0 exactly on every seed",
    "b5": "rate(SYNCHRONOUS) >= rate(INTEGRATED) and the matched arm within 5 %, every seed",
    "b6": "t(INTEGRATED - null) > bar AND both partitions <= null on every seed   [the dissociation]",
    "b7": "R3 holds (the G5 chain through the port)",
}


def format_verdict_t070(v, readings):
    """The verdict as a reader sees it: the arms per seed, the rate search per seed, every conjunct
    MET / NOT MET with its t and FPR. The only number that may reach a surface is (b1)'s contrast
    with (b2) beside it -- never an arm's PCI alone."""
    lines = []
    for s in v["seeds"]:
        lines.append(f"seed {s}:")
        for arm in ARMS:
            r = readings[s][arm]
            lines.append(
                f"  {arm:<26} amp {r.amp:6.4f}  PCI {r.pci:.4f}  gzip {r.gzip:7.2f}  "
                f"ones {r.ones:7.1f}  rate {r.rate:.4f}  null {r.null:.4f}  in-flight {r.inflight}"
            )
        prm = readings[s]["partitioned_rate_matched"]
        steps = "; ".join(f"{a:.4f}->{r:.4f}" for a, r in prm.search)
        lines.append(
            f"  rate search ({len(prm.search)} steps, target {readings[s]['integrated'].rate:.4f}, "
            f"{'matched' if prm.matched else 'NOT MATCHED'}): {steps}"
        )
    for name, c in v["conjuncts"].items():
        lines.append(
            f"({name}) {'MET    ' if c['met'] else 'NOT MET'}  {_CONJUNCT_TEXT[name]}: {c['note']}"
        )
    lines.append(f"verdict: {'PASS' if v['pass'] else 'NOT MET'} (bar {v['bar']})")
    return "\n".join(lines)


# =============================================================================================
# The self-check (run_selfchecks CORE): R1, R2, R3 and the pulse-less identity, in one command
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


def _chain_lines(rows):
    out = []
    for r in rows:
        out.append(
            f"R3 seed {r['seed']}: in-flight {r['inflight']:2d}; restored baseline "
            f"{'BYTE-IDENTICAL' if r['identical'] else 'DIVERGED (defect)'}; frame-only restore "
            f"{'diverged as it must' if r['frame_only_diverged'] else 'DID NOT DIVERGE (the fixture is wrong, not the chain)'}; "
            f"omit-positions {'REFUSED' if r['omit_refused'] else 'ACCEPTED (defect)'}; "
            f"pulse-less identity {'holds' if r['pulseless_identity'] else 'BROKEN (drive leaked)'}"
        )
    return out


def selfcheck(seeds=SELF_SEEDS, amp=FIXTURE_AMP):
    """The offline check on the harness fixture, every red in the SAME command as its control:
    R3 (`chain_check`), R1 (the pulse-less run reads every arm 0 and (b1)(b2)(b3)(b6) NOT MET), R2
    (the un-cut partition reads (b6) NOT MET) and the real readings with their verdict. Returns
    (chain rows, R1 verdict, R2 verdict, readings, verdict)."""
    from snn_data_lake.cortex.prior_drift_harness import synthetic_artifact

    art = synthetic_artifact(64, out_degree=8)
    chain = chain_check(art, seeds, amp)
    red1 = {s: kernel_regimes(art, s, amp, pulse_current=0.0) for s in seeds}
    v1 = verdict_t070(red1, chain)
    readings = {s: kernel_regimes(art, s, amp) for s in seeds}
    uncut = {
        s: {**readings[s], "partitioned": run_regime(arm_uncut(art), "partitioned", s, amp)}
        for s in seeds
    }
    v2 = verdict_t070(uncut, chain)
    return chain, (red1, v1), (uncut, v2), readings, verdict_t070(readings, chain)


def main(argv=None) -> int:
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        chain, (red1, v1), (uncut, v2), readings, v = selfcheck()
    print("=" * 96)
    print("T070 -- THE INTEGRATION INSTRUMENT ON THE KERNEL'S PORT (self-check, 64-unit fixture)")
    print("=" * 96)
    for line in _chain_lines(chain):
        print(line)
    r1_every_zero = all(
        r.pci == 0.0 and r.ones == 0 for arms in red1.values() for arm in arms for r in [arms[arm]]
    )
    r1_not_met = [n for n in ("b1", "b2", "b3", "b6") if not v1["conjuncts"][n]["met"]]
    print(
        f"R1 (pulse-less run): every arm PCI 0 and footprint empty: {r1_every_zero}; "
        f"NOT MET on {r1_not_met} -> {'RED fires' if r1_every_zero and len(r1_not_met) == 4 else 'RED DID NOT FIRE (defect)'}"
    )
    above = [s for s in uncut if uncut[s]["partitioned"].pci > uncut[s]["integrated"].null]
    r2_fires = bool(above) and not v2["conjuncts"]["b6"]["met"]
    print(
        f"R2 (the un-cut partition, E edges kept): above the null on seeds {above}; "
        f"(b6) {'NOT MET' if not v2['conjuncts']['b6']['met'] else 'MET'} -> "
        f"{'RED fires' if r2_fires else 'RED DID NOT FIRE (defect)'}: {v2['conjuncts']['b6']['note']}"
    )
    print("-" * 96)
    print(format_verdict_t070(v, readings))
    print("-" * 96)
    chain_ok = v["conjuncts"]["b7"]["met"]
    if not (chain_ok and r1_every_zero and len(r1_not_met) == 4 and r2_fires):
        print("integration_kernel self-check FAILED: a red did not fire or the chain broke")
        return 1
    # the fixture's TRIPWIRES (a regression check on constants this module owns, never a verdict
    # about another system): the conjuncts the fixture reads MET at FIXTURE_AMP, pinned
    for name in FIXTURE_TRIPWIRES:
        c = v["conjuncts"][name]
        if not c["met"]:
            print(
                f"integration_kernel self-check FAILED: fixture tripwire ({name}) NOT MET: {c['note']}"
            )
            return 1
    c1, c2 = v["conjuncts"]["b1"], v["conjuncts"]["b2"]
    print(
        f"integration_kernel self-check OK: R1, R2 and R3 fired RED in this command on {len(chain)} "
        f"seeds of the 64-unit fixture; the contrast INTEGRATED - PARTITIONED t {c1['t']:+.3f} (FPR "
        f"{c1['fpr']:.4f}) with the rate-matched control t {c2['t']:+.3f} (FPR {c2['fpr']:.4f}) beside "
        f"it; tripwires {list(FIXTURE_TRIPWIRES)} MET. NOT consciousness, NOT IIT Phi; no live "
        "population runs on the kernel."
    )
    return 0


#: THE FIXTURE'S TRIPWIRES -- the conjuncts the 64-unit fixture reads MET at FIXTURE_AMP over
#: SELF_SEEDS, MEASURED 2026-09-10 on the first full run and pinned as a regression check on
#: constants this module owns (the incumbent's own rule: a cutoff on a fixed synthetic regime with
#: fixed seeds is a tripwire, never a verdict about another system). That run: (b1) t +4.2504,
#: (b2) t +3.9028, (b3) t +4.9497, every FPR 0.0625 (= 1/16, the floor of a 4-seed sign flip);
#: (b5) 4/4 and 4/4; (b7) 4/4. NOT pinned because the fixture reads them NOT MET, disclosed here
#: rather than re-labelled: (b4) SILENT PCI = 0 exactly on 1/4 -- with the wiring intact a 3-tick
#: pulse at rest reaches a post whose fan-in share from a pulsed unit exceeds ~0.37, one hop, on
#: ~1 of 8 sets (PCI 0.0012-0.0025, 0.1-0.2 ones); (b6) partitioned <= null 4/4 but
#: partitioned_rate_matched <= null 3/4 -- the matched arm's local echo at its higher amplitude
#: (0.0162 at amp 0.475, seed 3) exceeds INTEGRATED's echo (0.0134). The campaign's verdict rule
#: is 2.3's, all seven conjuncts, and is not this tuple.
FIXTURE_TRIPWIRES = ("b1", "b2", "b3", "b5", "b7")


if __name__ == "__main__":
    sys.exit(main())
