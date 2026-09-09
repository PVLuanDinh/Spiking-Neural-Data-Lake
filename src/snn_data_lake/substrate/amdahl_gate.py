"""Gate Zero -- the Amdahl ceiling of a delivery-path conversion on the LIVE tick.

specs/008-vectorize-scalar-paths/plan.md orders this instrument FIRST: "Build Gate Zero
(neuter-and-measure the Amdahl ceiling) plus the T0-T3 ladder. Without it, no conversion has
evidence." specs/022 S6 (evidence/s6-kernel-wire.md 2.2) is its pre-registration. The `tools/`
path specs/008 names is refused by the p4c ratchet since T001, so the instrument lives here, in
the governed tree, as P4c requires.

WHY IT EXISTS. A conversion's microbenchmark measures the converted site; it never measures what
the site is WORTH. A delivery path made infinitely fast still cannot move the tick by more than the
share it holds, and `perf_baseline`'s ratchet cannot even see a gain below TOLERANCE (0.15). So the
question that must be answered BEFORE a conversion is not "is the new mechanism faster" but "is
there anything here to make faster" -- and the honest way to answer it is to remove the path's cost
outright and re-time the tick.

WHAT IT MEASURES, in one command:
  (i)   PROFILE. W warm wake ticks of make_brain(seed, persist=False) in a scratch NORD_HOME, with
        tottime attributed to the PER-OBJECT NEURON PATH (cortex/lif_neuron.py, cortex/neuron_zoo.py,
        substrate/basic_components.py -- the set a flat (post, weight) scatter would replace) versus
        the rest, and the I/O share reported separately.
  (ii)  NEUTER. LIFNeuron.step is replaced by the cheapest correct-shape stand-in and the wake p50 is
        re-timed over W PAIRED ticks in the SAME command. Ceiling = 1 - p50(neutered)/p50(live), with
        a 1,000-resample bootstrap band over the paired per-tick differences. Every tick sample is
        paired with an adjacent crc32 reference sample and BOTH raw and reference-corrected readings
        are printed -- T068 (evidence/s5-paging-null.md 3.6 item 3) measured the reference to be NOT
        arm-independent for a light process, so raw decides when the two disagree.
  (iii) WIDTH TRIGGER N*. The unit count at which the object path's share reaches TOLERANCE with the
        rest of the tick held at its measured cost. Derived TWICE from instruments that fail
        differently -- the profile's attribution and the neuter's profiler-free wall clock -- and
        both printed, because one number from one instrument is an opinion.

WHAT IT DOES NOT DO. It does not decide. The pre-registered rule's verdict WORD is printed as a
REPORT line; the exit code carries only whether the instrument's OWN controls held. A gate that both
measures and rules on its measurement has no reviewer.

THE STAND-IN, and why it is the conversion's LOWER BOUND on cost. `_neutered_step` returns the
neuron's last returned output -- 0, initialised once at class scope and never updated, because a
neutered neuron integrates nothing and so can never spike -- at the cost of ONE attribute read and no
arithmetic. No conversion of this path can cost less than producing one output per unit per tick, so
the neutered arm is a floor on any converted arm and the measured ceiling is an UPPER bound on any
achievable gain. Two disclosures that both push the SAME way (the ceiling is over-stated, never
under-stated, so a DECLINE cannot be manufactured by them): the file-level attribution charges ALL of
basic_components.py to the object path, and the neuter also removes whatever downstream work the
suppressed spikes would have caused.

Modes
  (bare)         the reds, then the live reading; exits 1 only if the instrument's controls fail
  --self-check   the three pre-registered reds with their broken arms, PASS/FAIL each, no brain
  --seed N       read at seed N (default 0);  --ticks N  paired wake ticks per arm (default 150)

  python -m snn_data_lake.substrate.amdahl_gate --self-check
"""

# distinct-from: substrate/perf_baseline.py -- the pinned-host RATCHET, which asks whether the tick
#   got SLOWER than a recorded baseline. This gate asks how much of the tick a delivery-path
#   conversion could remove at all, and it REUSES perf_baseline's _ref_ms and TOLERANCE rather than
#   re-implementing either: one reference sample and one tolerance survive in the tree (P2).
# distinct-from: tools/charge_probe.py -- the neuter-A/B for CHARGE (does a wire change an OUTCOME).
#   This one neuters for COST; its verdict is a ceiling and never a charge claim.
# distinct-from: substrate/determinism_gate.py and substrate/replay_gate.py -- criteria over
#   kernel_tick's mechanism; this gate never constructs a KernelTick and makes no kernel claim.
# distinct-from: cortex/live_brain.py::LiveBrain.run, cortex/heterogeneous_microcircuit.py::run_circuit
#   and cortex/brain.py::Brain._perceive_real -- the three graph-exists near-matches. Each DRIVES a
#   tick; none measures a share, a ceiling or a width trigger. All sites read.

from __future__ import annotations

import gc
import random
import statistics
import sys
import time

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate.basic_components import Connection
from snn_data_lake.substrate.perf_baseline import TOLERANCE, _ref_ms

#: the pre-registered reading (s6-kernel-wire.md 2.4): W warm wake ticks per arm, after a warm-up
W_TICKS = 150
WARMUP = 20
RESAMPLES = 1000

#: the reds' constructed rig -- fast, hermetic, no brain, no lake
RED_TICKS = 140
RED_TICKS_QUICK = 40
RED_RESAMPLES = 400
HARNESS_UNITS = 165  # the deployed width measured for this row (design.md F6/F9)
HARNESS_REST = 12000  # integer work per harness tick that the object-path neuter cannot remove
PAD_ITERS = 600  # the positive control's KNOWN extra cost per unit
PADDED_MIN_CEILING = 0.5  # a padded object path must read at least this, or the instrument is blind
SHAM_NOISE_BAND = 0.05  # a stand-in on a zero-share path must read within this of zero

RED_NAMES = ("padded-share", "repeatability", "sham-neuter")

#: files whose tottime is charged to the per-object neuron path (2.2 i). basic_components.py is
#: charged WHOLE rather than Connection-only: cProfile keys carry no class, and the over-attribution
#: can only INFLATE the ceiling, which is the safe direction for a DECLINE.
_OBJECT_PATH = ("/cortex/lif_neuron.py", "/cortex/neuron_zoo.py", "/substrate/basic_components.py")
_IO_TOKENS = ("fsync", "flush", "nt.write", "nt.read", "nt.open", "nt.close")

_MISSING = object()


# --------------------------------------------------------------------------- the stand-ins


def _neutered_step(self, current):
    """THE NEUTER. Return the neuron's last returned output; integrate nothing.

    It is 0: initialised once at class scope and never written, because a neuron that integrates
    nothing can never cross threshold and so can never return anything else. One attribute read, no
    arithmetic. This is the LOWER BOUND on any conversion of this path -- a vectorized, kernelised or
    CUDA delivery still has to produce one output per unit per tick, and cannot do it for less than
    an attribute read -- so 1 - p50(neutered)/p50(live) is an UPPER bound on any achievable gain.
    """
    return self._neutered_last


def _padded_step(pad_iters):
    """THE POSITIVE CONTROL's stand-in: the real step plus a KNOWN extra cost per unit.

    The real step is taken from _REAL_STEP, captured at import BEFORE any patch can be installed --
    reading LIFNeuron.step at call time would capture whatever stand-in happened to be installed and
    silently turn the control into a copy of the arm it is controlling.
    """

    def padded(self, current):
        out = _REAL_STEP(self, current)
        acc = 0
        for i in range(pad_iters):
            acc += i * i
        self._pad_acc = acc  # keep the padded work observable, never elidable
        return out

    return padded


# distinct-from: training/spiking_lm.py::verify_transfer -- matched only on the substring "ZERO-D";
#   that function loads an exported LM into the zero-dep core and generates text. It installs
#   nothing, times nothing and has no control role. Site read.
def _sham_propagate(self, pre):
    """THE SHAM: installed on Connection.propagate, a path carrying ZERO share. Removes no work.

    MEASURED, not assumed -- and the first choice was WRONG. `LIFNeuron.run` was the obvious sham,
    since the harness never calls it, and the control caught it: on the live brain the same stand-in
    logged 11 calls in one tick, so it was not a zero-share path and the reading it controlled was
    not controlled. `Connection.propagate` is the tree's generic dense propagate and the live brain
    holds ZERO Connection instances (design.md F6), so nothing calls it in either substrate. The
    call count printed beside every reading is the PROOF; a sham whose share is asserted rather than
    counted is the same mistake with better luck.
    """
    return []


SHAM_PATCH = (Connection, "propagate", _sham_propagate)

NEUTER_PATCH = (LIFNeuron, "step", _neutered_step)
_REAL_STEP = LIFNeuron.step  # captured before any stand-in can be installed
LIFNeuron._neutered_last = 0  # the stand-in's one attribute, defined once, never written


# --------------------------------------------------------------------------- the measurement core


def _install(obj, name, fn):
    """Install `fn` as obj.name; return the token that restores the previous binding."""
    return (obj, name, obj.__dict__.get(name, _MISSING), fn)


def _restore(token):
    obj, name, orig, _fn = token
    if orig is _MISSING:
        delattr(obj, name)
    else:
        setattr(obj, name, orig)


def _tick_arm(target, patch):
    """One tick of one arm, with `patch` installed for THAT tick only.

    Returns (phase, dt_ms, ref_ms). The reference sample is taken ADJACENT to the tick so an ambient
    burst hits the pair together (perf_baseline's method), and the install/restore happens OUTSIDE
    the timed region so the patch never charges itself to the arm.
    """
    ref = _ref_ms()
    token = None
    if patch is not None:
        obj, name, fn = patch
        token = _install(obj, name, fn)
        setattr(obj, name, fn)
    try:
        t0 = time.perf_counter()
        phase = target.tick()
        dt = (time.perf_counter() - t0) * 1000.0
    finally:
        if token is not None:
            _restore(token)
    return phase, dt, ref


def _probe_install(target, patch):
    """Prove the patch INSTALLS and count how many times it is CALLED, over one UNTIMED tick.

    Presence is not participation: a swap that silently fails to install measures the unmodified code
    and reports it as a pass (specs/008 plan.md invariant 5). The count is taken on its own tick so
    the counter's cost never enters a timed sample -- an instrumented arm would read SLOWER, which
    would SHRINK the ceiling and make a DECLINE easier to reach.

    The PHASE of that tick is returned with the count, because the count depends on it: measured over
    80 live ticks the neuter's count is 4,324-5,124 awake against 1,264-5,872 asleep, while every
    paired sample behind the ceiling is an AWAKE tick. An unlabelled `calls=1264` beside an
    awake-only ceiling understates by ~4x what the neuter removes on the measured ticks. The probe
    does NOT choose its phase -- that would redefine a control after its readings are on file, and it
    would buy nothing: no tick of either phase reached zero calls.
    """
    obj, name, fn = patch
    calls = [0]

    def counting(*a, **k):
        calls[0] += 1
        return fn(*a, **k)

    token = _install(obj, name, counting)
    setattr(obj, name, counting)
    installed = getattr(obj, name) is counting
    try:
        phase = target.tick()
    finally:
        _restore(token)
    return installed, calls[0], phase


def _p50(vals):
    return statistics.median(vals)


def _bootstrap_ceiling(live, neut, resamples, seed):
    """95% band on the ceiling, resampling the PAIRED indices (the pairing is the statistic)."""
    rng = random.Random(seed)
    n = len(live)
    out = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = _p50([live[i] for i in idx])
        b = _p50([neut[i] for i in idx])
        out.append(1.0 - b / a if a > 0 else 0.0)
    out.sort()
    return (out[int(0.025 * resamples)], out[min(resamples - 1, int(0.975 * resamples))])


def paired_ceiling(live_ms, neut_ms, live_ref, neut_ref, resamples=RESAMPLES, seed=0):
    """The ceiling, RAW and reference-corrected, with its paired bootstrap band on each basis."""
    n = len(live_ms)
    if n == 0 or not (len(neut_ms) == len(live_ref) == len(neut_ref) == n):
        raise ValueError(f"paired arms must be non-empty and equal-length (got {n})")
    if min(live_ms) <= 0.0:
        raise ValueError("a live arm sample at or below zero is not a measurement, it is a bug")
    lc = [d / r for d, r in zip(live_ms, live_ref, strict=True) if r > 0]
    nc = [d / r for d, r in zip(neut_ms, neut_ref, strict=True) if r > 0]
    raw = 1.0 - _p50(neut_ms) / _p50(live_ms)
    cor = 1.0 - _p50(nc) / _p50(lc) if lc and nc else None
    return {
        "n": n,
        "live_p50_raw_ms": round(_p50(live_ms), 4),
        "neut_p50_raw_ms": round(_p50(neut_ms), 4),
        "live_p50_corrected": round(_p50(lc), 4) if lc else None,
        "neut_p50_corrected": round(_p50(nc), 4) if nc else None,
        "ref_p50_live_ms": round(_p50(live_ref), 4),
        "ref_p50_neut_ms": round(_p50(neut_ref), 4),
        "ceiling_raw": round(raw, 5),
        "band_raw": tuple(
            round(v, 5) for v in _bootstrap_ceiling(live_ms, neut_ms, resamples, seed)
        ),
        "ceiling_corrected": round(cor, 5) if cor is not None else None,
        "band_corrected": (
            tuple(round(v, 5) for v in _bootstrap_ceiling(lc, nc, resamples, seed)) if lc else None
        ),
    }


# --------------------------------------------------------------------------- the constructed rig


class _TickHarness:
    """A constructed tick: `n_units` LIFNeuron.step calls plus a fixed REST the neuter cannot touch.

    It exists so the reds are hermetic and cost under a second. DISCLOSED: it shares the measurement
    CORE with the live reading and NOT its substrate, so the positive control proves the core can see
    a ceiling -- never that it can see one on the live brain. The live reading carries its own
    same-substrate sham arm for that.
    """

    def __init__(self, n_units=HARNESS_UNITS, rest_units=HARNESS_REST, seed=0):
        rng = random.Random(seed)
        self.neurons = [LIFNeuron(tau_m=10.0) for _ in range(n_units)]
        self.drive = [0.15 + 0.5 * rng.random() for _ in range(n_units)]
        self.rest_units = rest_units

    def tick(self):
        for nrn, cur in zip(self.neurons, self.drive, strict=True):
            nrn.step(cur)
        acc = 0
        for i in range(self.rest_units):
            acc += i * i
        return "awake" if acc >= 0 else "asleep"


def _harness_pair(patch_a, patch_b, ticks, resamples, seed):
    """Two harness arms, interleaved tick by tick so both see the same ambient, paired by index.

    COUNTERBALANCED: the within-index ORDER alternates (A,B then B,A). Measured before this was
    added, a fixed order gave the second arm a systematic ~0.6% advantage -- small, but the sham
    control resolved it well enough that its bootstrap band EXCLUDED zero, i.e. the instrument was
    reporting a real position artefact as a real ceiling. Alternating removes the artefact by
    design instead of disclosing it in prose.
    """
    a, b = _TickHarness(seed=seed), _TickHarness(seed=seed)
    la, ra, lb, rb = [], [], [], []
    for i in range(ticks):
        order = ((a, patch_a, la, ra), (b, patch_b, lb, rb))
        for tgt, pat, dst, refs in order if i % 2 == 0 else order[::-1]:
            _p, d, r = _tick_arm(tgt, pat)
            dst.append(d)
            refs.append(r)
    return paired_ceiling(la, lb, ra, rb, resamples, seed)


def red_padded_share(pad_iters=PAD_ITERS, ticks=RED_TICKS, resamples=RED_RESAMPLES, seed=0):
    """RED (a), the POSITIVE CONTROL: a padded object path must read a LARGE ceiling."""
    pad = _padded_step(pad_iters)
    installed, calls, phase = _probe_install(_TickHarness(seed=seed), (LIFNeuron, "step", pad))
    d = _harness_pair((LIFNeuron, "step", pad), NEUTER_PATCH, ticks, resamples, seed)
    d.update(pad_installed=installed, pad_calls=calls, pad_iters=pad_iters, pad_phase=phase)
    ok = bool(installed) and calls > 0 and d["ceiling_raw"] >= PADDED_MIN_CEILING
    return (
        ok,
        (
            f"padded-share: ceiling {d['ceiling_raw']:.4f} band {d['band_raw']} "
            f"(bar >= {PADDED_MIN_CEILING}); pad {pad_iters} iters/unit installed={installed} "
            f"calls={calls}"
        ),
        d,
    )


def red_repeatability(ticks=RED_TICKS, resamples=RED_RESAMPLES, seed=0):
    """RED (b): two neutered readings of the same rig, judged by their bands OVERLAPPING.

    The band, not a hand-picked epsilon: a noisy block widens its own band, so the rule tightens
    exactly when the data is clean and cannot be passed by being sloppy.
    """
    neut = NEUTER_PATCH
    a = _harness_pair(None, neut, ticks, resamples, seed)
    b = _harness_pair(None, neut, ticks, resamples, seed + 1)
    (lo1, hi1), (lo2, hi2) = a["band_raw"], b["band_raw"]
    ok = lo1 <= hi2 and lo2 <= hi1
    d = {
        "ceiling_a": a["ceiling_raw"],
        "ceiling_b": b["ceiling_raw"],
        "band_a": a["band_raw"],
        "band_b": b["band_raw"],
    }
    return (
        ok,
        (
            f"repeatability: ceilings {a['ceiling_raw']:.4f} / {b['ceiling_raw']:.4f}; "
            f"bands {a['band_raw']} vs {b['band_raw']} -> {'overlap' if ok else 'DISJOINT'}"
        ),
        d,
    )


def red_sham_neuter(ticks=RED_TICKS, resamples=RED_RESAMPLES, seed=0):
    """RED (c), THE CONTROL THAT IS NOT THE METHOD: a stand-in on a ZERO-share path reads ~0.

    THE BAR IS MAGNITUDE, AND THE BAND IS REPORTED -- a criterion choice made AFTER seeing data, so
    it is recorded here rather than left in a commit message. The pre-registration
    (s6-kernel-wire.md 2.2 iv / 2.5) says the sham must read "within noise of zero", and the reds
    committed before this module existed assert exactly that: |ceiling| <= SHAM_NOISE_BAND. During
    the build an EXTRA conjunct was tried -- the bootstrap band must also CONTAIN zero -- and it was
    removed for cause, with the cause measured:

      fixed arm order, 2 harness instances : -0.0059  (band excluded zero)
      counterbalanced, 2 instances         : +0.0007  -0.0001  -0.0090  -0.0016  (1 of 4 excluded)
      counterbalanced, 1 SHARED instance   : -0.0040  +0.0016  -0.0003  -0.0049  +0.0054  (1 of 5)

    Counterbalancing the within-index arm order fixed a real artefact and is KEPT. What survives is
    ordinary timing drift inside a 0.3 s block, and a 140-sample bootstrap resolves it to about
    +/-0.005 -- so "the band contains zero" is a SIGNIFICANCE test roughly thirty times finer than
    the 0.15 decision it supports, and it fired on about one clean run in five. A red that fires
    without a cause behind it cannot distinguish, and a flaky gate trains overrides (the lesson
    perf_baseline's own header records). The band is still computed and printed as
    `band_excludes_zero`, so nothing is hidden -- it is reported, not asserted.
    """
    sham = SHAM_PATCH
    installed, calls, phase = _probe_install(_TickHarness(seed=seed), sham)
    d = _harness_pair(None, sham, ticks, resamples, seed)
    lo, hi = d["band_raw"]
    d.update(
        sham_installed=installed,
        sham_calls=calls,
        sham_phase=phase,
        band_excludes_zero=not (lo <= 0 <= hi),
    )
    ok = bool(installed) and calls == 0 and abs(d["ceiling_raw"]) <= SHAM_NOISE_BAND
    return (
        ok,
        (
            f"sham-neuter: ceiling {d['ceiling_raw']:+.4f} band {d['band_raw']} "
            f"(bar |c| <= {SHAM_NOISE_BAND}); band excludes zero: {d['band_excludes_zero']} "
            f"(reported, not asserted -- see the docstring); installed={installed} "
            f"calls={calls} (0 is the PROOF the path carries zero share)"
        ),
        d,
    )


def self_check_reds(pad_iters=PAD_ITERS, quick=False):
    """The three pre-registered reds, in their pre-registered order, each with its broken arm."""
    ticks = RED_TICKS_QUICK if quick else RED_TICKS
    res = 150 if quick else RED_RESAMPLES
    rows = [
        ("padded-share", *red_padded_share(pad_iters=pad_iters, ticks=ticks, resamples=res)),
        ("repeatability", *red_repeatability(ticks=ticks, resamples=res)),
        ("sham-neuter", *red_sham_neuter(ticks=ticks, resamples=res)),
    ]
    return rows


# --------------------------------------------------------------------------- the live reading


# distinct-from: brainstem/arm_config.py -- ::ground_free_band sweeps a joint's ground-free range.
# distinct-from: brainstem/cpg_oscillator.py -- ::_reversals counts direction changes in a trace.
# distinct-from: cortex/theory_of_mind.py -- ::believed_location is a spiking WTA belief readout.
# distinct-from: brainstem/arm_bridge.py -- ::send_outcome fires one UDP datagram.
# distinct-from: brainstem/arm_hardware.py -- ::peak_current_amps sums a servo supply budget.
# All five matched on the word "one" and all five sites were read; none counts live instances of a
# class. graph-exists "count live instances of a class by walking the brain object graph" returns
# NO EXISTING IMPLEMENTATION, so this is the first, not the second.
def _count_lif_process():
    """Every LIFNeuron alive in this PROCESS. With three brains up it is a total, not a width."""
    gc.collect()
    return sum(1 for o in gc.get_objects() if type(o) is LIFNeuron)


def _count_lif_in(root, max_nodes=400000):
    """Every LIFNeuron reachable from ONE brain -- the per-brain width, with three brains alive.

    A bounded attribute walk that never follows classes or modules, so it cannot wander out of this
    brain into a sibling through a global. It is CROSS-CHECKED against the process-wide count at the
    single moment the two must agree -- right after the first brain is built and before the second
    exists -- because a census with no second instrument is a number nobody has audited.
    """
    seen, stack, n_lif, walked = set(), [root], 0, 0
    while stack and walked < max_nodes:
        o = stack.pop()
        i = id(o)
        if i in seen:
            continue
        seen.add(i)
        walked += 1
        if type(o) is LIFNeuron:
            n_lif += 1
        if isinstance(o, (str, bytes, bytearray, int, float, bool, type(None), type)):
            continue
        if isinstance(o, dict):
            stack.extend(list(o.values())[:5000])
            continue
        if isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(list(o)[:5000])
            continue
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            stack.extend(list(d.values())[:5000])
    return n_lif, walked


def _profile_shares(brain, wake_ticks):
    """(i) Attribute tottime over `wake_ticks` WAKE ticks to the object path, the I/O and the rest.

    Each tick is profiled by its OWN cProfile.Profile and merged only when the tick was awake -- one
    profiler spanning the block would fold the sleep ticks into the denominator and quietly shrink
    every share it reports.
    """
    import cProfile  # noqa: PLC0415
    import pstats  # noqa: PLC0415

    stats, seen, attempts = None, 0, 0
    while seen < wake_ticks and attempts < wake_ticks * 12:
        attempts += 1
        pr = cProfile.Profile()
        pr.enable()
        phase = brain.tick()
        pr.disable()
        if phase != "awake":
            continue
        seen += 1
        stats = pstats.Stats(pr) if stats is None else stats.add(pr)
    total = obj = io = 0.0
    io_rows = []
    for (fn, _ln, name), (_cc, _nc, tt, _ct, _cs) in stats.stats.items():
        total += tt
        f = str(fn).replace("\\", "/")
        if any(p in f for p in _OBJECT_PATH):
            obj += tt
        if any(tok in name for tok in _IO_TOKENS):
            io += tt
            io_rows.append((round(tt, 4), name))
    io_rows.sort(reverse=True)
    return {
        "wake_ticks_profiled": seen,
        "ticks_attempted": attempts,
        "total_tottime_s": round(total, 4),
        "object_path_tottime_s": round(obj, 4),
        "object_path_share": round(obj / total, 5) if total else None,
        "io_tottime_s": round(io, 4),
        "io_share": round(io / total, 5) if total else None,
        "io_top": io_rows[:4],
        "rest_tottime_s": round(total - obj, 4),
    }


def _n_star_from_ceiling(c, n_lif, tol):
    """N*_neuter = K*((1-c)/c)*N, K = tol/(1-tol) -- the identity the neuter branch computes.

    None when c <= 0: the map is monotone DECREASING in c and has no finite value there.
    """
    return int(tol * n_lif * (1.0 - c) / ((1.0 - tol) * c)) if c > 0 else None


def width_trigger(
    prof,
    n_lif,
    live_p50_ms,
    neut_p50_ms,
    tol=TOLERANCE,
    n_lif_paired_live=None,
    n_lif_neuter_arm=None,
    band_raw=None,
):
    """(iii) N*: the width at which the object path's share reaches `tol`, from TWO instruments.

    share(N) = c_u*N / (c_u*N + R) = tol   ->   N* = tol*R / ((1 - tol)*c_u)

    c_u is the object path's cost per unit per tick and R the rest-of-tick cost per tick. The two
    derivations fail DIFFERENTLY -- one reads the profiler's attribution, the other reads the
    neuter's profiler-free wall clock -- and both are printed, because a single number from a single
    instrument is an opinion.

    NEITHER N* IS A SECOND INSTRUMENT. Each is its upstream statistic restated in width units, and
    the identities are exact (verified to max abs diff 0.0 over 2,000 random inputs):

        N*_profile = K*((1 - s)/s)*N   with s the profiler's object-path share
        N*_neuter  = K*((1 - c)/c)*N   with c the Amdahl ceiling,   K = tol/(1 - tol)

    So `agree_ratio` is the share-vs-ceiling disagreement re-expressed, not two instruments
    disagreeing about a width. The consequence the record owes: the neuter arm INHERITS the ceiling's
    95% band, and that band is mapped here as `n_star_band` with its endpoints SWAPPED (the map is
    monotone decreasing) and "unbounded" wherever an endpoint's c <= 0 -- which is every determinate
    reading on file, so no N* here is a confidence bound. The profile arm's `n_star_band` is None on
    purpose: no bootstrap exists over profiler tottime, so that number has no interval at all, and
    printing the asymmetry is the point.

    CAVEAT, pre-registered: N* is a LOWER bound twice over. The rest of the
    tick is held at its measured cost and in truth grows with the brain, and cProfile's per-call
    overhead inflates c_u more than R because the object path is millions of tiny calls.
    CAVEAT, added 2026-09-09 (NOT pre-registered): and the object path's cost is taken as
    proportional to the LIF census N. c_u is measured at ONE width; its width-invariance is untested,
    so the LOWER-bound direction holds only while the path grows at most linearly in N. Measured on
    the live brain, an awake tick enters the path ~5,124 times over ~71-74 DISTINCT stepped units of
    a 165-unit census -- one unit carrying ~30% of the calls -- so "cost per unit" is not a
    mechanism-level quantity and extrapolating it 15-40x assumes growth preserves that composition.

    THE WIDTH THE NEUTER ARM'S c_u IS DIVIDED BY. `n_lif` is the post-profile LIVE width, and the
    paired block that produced the p50 delta ran at a possibly narrower one. Given
    `n_lif_paired_live`, the neuter arm additionally reports `c_u_paired_block` and
    `n_star_paired_block` on that width -- printed BESIDE the published `n_star`, never in place of
    it, because the published number is on the record. `n_lif_neuter_arm` is reported for the same
    reason: the two arms are not always the same width (measured seed 1 live 177 / neut 165,
    seed 2 189 / 177, seed 0 165 / 165), and the missing units make the neutered arm faster for a
    reason other than the neutering -- a THIRD attribution bias, pushing the ceiling UP and N* DOWN,
    the same direction as the two the record already names.
    """
    out = {
        "tol": tol,
        "n_lif": n_lif,
        "formula": "N* = tol*R / ((1 - tol)*c_u)",
        "identity_profile": "N*_profile = K*((1-s)/s)*N, s = the profiler's object-path share",
        "identity_neuter": "N*_neuter = K*((1-c)/c)*N, c = the Amdahl ceiling",
        "identity_K": "K = tol/(1 - tol)",
        "identity_note": (
            "each N* is its upstream statistic restated in width units, NOT a third measurement; "
            "agree_ratio is the share-vs-ceiling disagreement re-expressed"
        ),
    }
    w = prof["wake_ticks_profiled"]
    if n_lif and w:
        c_prof = prof["object_path_tottime_s"] * 1000.0 / (n_lif * w)
        r_prof = prof["rest_tottime_s"] * 1000.0 / w
        out["profile"] = {
            "c_u_ms_per_unit_tick": round(c_prof, 8),
            "rest_ms_per_tick": round(r_prof, 4),
            "n_star": int(tol * r_prof / ((1 - tol) * c_prof)) if c_prof > 0 else None,
            "n_star_band": None,
            "n_star_band_reason": (
                "no bootstrap exists over profiler tottime -- this number has no interval"
            ),
        }
    delta = live_p50_ms - neut_p50_ms
    if n_lif and delta > 0:
        c_neut = delta / n_lif
        out["neuter"] = {
            "c_u_ms_per_unit_tick": round(c_neut, 8),
            "rest_ms_per_tick": round(neut_p50_ms, 4),
            "n_star": int(tol * neut_p50_ms / ((1 - tol) * c_neut)),
            "n_lif_used": n_lif,
            "n_lif_neuter_arm": n_lif_neuter_arm,
        }
        if n_lif_paired_live:
            c_pb = delta / n_lif_paired_live
            out["neuter"]["n_lif_paired_block_live"] = n_lif_paired_live
            out["neuter"]["c_u_paired_block"] = round(c_pb, 8)
            out["neuter"]["n_star_paired_block"] = int(tol * neut_p50_ms / ((1 - tol) * c_pb))
        if band_raw:
            b_lo, b_hi = band_raw
            lo_end = _n_star_from_ceiling(b_hi, n_lif, tol)  # high c -> low N*
            hi_end = _n_star_from_ceiling(b_lo, n_lif, tol)  # low c -> high N*
            out["neuter"]["n_star_band"] = (
                lo_end if lo_end is not None else "unbounded",
                hi_end if hi_end is not None else "unbounded",
            )
            out["neuter"]["n_star_band_note"] = (
                "the ceiling's own 95% band mapped through the SAME identity, endpoints SWAPPED "
                "(the map is monotone decreasing in c); an endpoint at c <= 0 is 'unbounded' -- "
                "there is no finite width there, so this is not a confidence bound on N*"
            )
    else:
        out["neuter"] = {
            "n_star": None,
            "reason": f"live-neutered p50 delta {delta:.4f} ms not > 0",
        }
    a = (out.get("profile") or {}).get("n_star")
    b = (out.get("neuter") or {}).get("n_star")
    out["agree_ratio"] = round(max(a, b) / min(a, b), 2) if a and b else None
    return out


def _verdict(band, tol=TOLERANCE):
    lo, hi = band
    if hi < tol:
        return "DECLINE"
    if lo >= tol:
        return "PROCEED"
    return "INDETERMINATE"


def live_reading(seed=0, wake_ticks=W_TICKS, warmup=WARMUP, resamples=RESAMPLES):
    """(ii) The paired reading on three seed-identical scratch brains: live, neutered, sham.

    THREE arms, interleaved tick by tick and paired by TRAJECTORY INDEX. Separate brains keep the
    neuter from contaminating the live arm's state; identical seeds keep index i the same logical
    tick in all three; interleaving keeps them in the same ambient. NORD_HOME is a throwaway for the
    whole run and persist is False -- a probe never touches the live lake.
    """
    from snn_data_lake.substrate.home import throwaway_home  # noqa: PLC0415

    neuter = NEUTER_PATCH
    sham = SHAM_PATCH
    with throwaway_home() as home:
        from snn_data_lake.cortex.brain import make_brain  # noqa: PLC0415

        t0 = time.perf_counter()
        # A WALL-CLOCK STAMP beside the duration. `elapsed_s` alone cannot locate a reading on the
        # commit timeline, so a host condition asserted for a reading ("the daemon up and
        # listening") is uncheckable after the fact -- which is exactly what happened to the five
        # readings on file (evidence/s6-kernel-wire.md 3.2). No host or port is probed: this gate
        # runs on a scratch home and must read the same on a clean checkout.
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        live = make_brain(seed=seed, persist=False)
        # TWO INSTRUMENTS ON THE WIDTH, cross-checked at the one moment they must agree: the
        # process-wide gc census and the reachable-from-this-brain walk, with exactly one brain up.
        n_proc0 = _count_lif_process()
        n_walk0, walked0 = _count_lif_in(live)
        census_agrees = n_proc0 == n_walk0
        neut = make_brain(seed=seed, persist=False)
        shm = make_brain(seed=seed, persist=False)
        for b, p in ((live, None), (neut, neuter), (shm, sham)):
            for _ in range(warmup):
                _tick_arm(b, p)
        rows, misaligned, attempts = [], 0, 0
        arms = [("live", live, None), ("neut", neut, neuter), ("sham", shm, sham)]
        while len(rows) < wake_ticks and attempts < wake_ticks * 12:
            # COUNTERBALANCED by a rotating Latin square: no arm sits in the same within-index
            # position twice running, so a position advantage cannot accumulate into one arm and
            # be read as that arm's ceiling. Measured with a fixed order the sham arm -- which
            # removes NO work -- read a ceiling of +0.018 against the real neuter's +0.021.
            k = attempts % 3
            got = {}
            for name, tgt, pat in arms[k:] + arms[:k]:
                got[name] = _tick_arm(tgt, pat)
            attempts += 1
            pl, pn, ps = got["live"][0], got["neut"][0], got["sham"][0]
            if pl == pn == ps == "awake":
                rows.append(
                    (
                        got["live"][1],
                        got["live"][2],
                        got["neut"][1],
                        got["neut"][2],
                        got["sham"][1],
                        got["sham"][2],
                    )
                )
            elif not pl == pn == ps:
                misaligned += 1
        # untimed probes: the neuter reached the path; the sham path carries zero share. The tick's
        # PHASE comes back with the count because the count depends on it (see _probe_install).
        n_ok, n_calls, n_phase = _probe_install(neut, neuter)
        s_ok, s_calls, s_phase = _probe_install(shm, sham)
        # PER-ARM WIDTH AT THE END OF THE PAIRED BLOCK -- the widths the ceiling was actually timed
        # at. It has to be read HERE and not after the profile: the profile block ticks the LIVE arm
        # ALONE, so a post-profile walk reports an arm gap that did not exist during the timing
        # (measured on seed 0, twice: 165/165/165 at block end against 177/165/165 after the
        # profile). The neutered brain spikes less and so GROWS LESS -- not "does not grow".
        n_live_pb, _wl = _count_lif_in(live)
        n_neut_pb, _wn = _count_lif_in(neut)
        n_sham_pb, _ws = _count_lif_in(shm)
        prof = _profile_shares(live, wake_ticks)
        # the width DURING the profiled ticks, per brain, from the audited walk -- not a third of a
        # process total: the neutered brain never spikes and so need not grow like the live one
        n_lif, _walked1 = _count_lif_in(live)
        n_lif_all_after = _count_lif_process()  # a process TOTAL, deliberately NOT divided by three
        out = {
            "seed": seed,
            "home": str(home),
            "n_lif_per_brain_at_build": n_walk0,
            "n_lif_paired_block": {"live": n_live_pb, "neut": n_neut_pb, "sham": n_sham_pb},
            # kept with its key and its semantics -- the record cites it -- but it is the
            # POST-PROFILE LIVE width, not the width the paired block ran at
            "n_lif_per_brain_profiled": n_lif,
            "n_lif_all_three_after": n_lif_all_after,
            "census_cross_check": {
                "process_at_build": n_proc0,
                "walk_at_build": n_walk0,
                "agree": census_agrees,
                "objects_walked": walked0,
            },
            "paired_wake_ticks": len(rows),
            "ticks_attempted": attempts,
            "phase_misaligned": misaligned,
            "neuter_installed": n_ok,
            "neuter_calls_one_tick": n_calls,
            "neuter_probe_phase": n_phase,
            "sham_installed": s_ok,
            "sham_calls_one_tick": s_calls,
            "sham_probe_phase": s_phase,
            "profile": prof,
            "started_utc": started_utc,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    live_ms = [r[0] for r in rows]
    out["neuter_arm"] = paired_ceiling(
        live_ms, [r[2] for r in rows], [r[1] for r in rows], [r[3] for r in rows], resamples, seed
    )
    out["sham_arm"] = paired_ceiling(
        live_ms, [r[4] for r in rows], [r[1] for r in rows], [r[5] for r in rows], resamples, seed
    )
    na = out["neuter_arm"]
    out["ceiling_raw"] = na["ceiling_raw"]
    out["band_raw"] = na["band_raw"]
    io = prof["io_share"]
    out["ceiling_raw_ex_io"] = (
        round(na["ceiling_raw"] / (1.0 - io), 5) if io is not None and io < 1.0 else None
    )
    # THE DISCLOSURE THIS READING OWES (P8 clause 4). The sham arm removes NO work, so whatever it
    # reads is the instrument's own floor. When the neuter's ceiling is the same size as the sham's,
    # the measured "removable cost" is inside the noise floor and the reading's honest content is
    # "below what this instrument can resolve", not a number. Reported, never asserted on, and never
    # substituted for the pre-registered statistic.
    out["sham_ceiling_raw"] = out["sham_arm"]["ceiling_raw"]
    out["ceiling_minus_sham"] = round(na["ceiling_raw"] - out["sham_arm"]["ceiling_raw"], 5)
    out["ceiling_exceeds_noise_floor"] = abs(na["ceiling_raw"]) > 2.0 * abs(
        out["sham_arm"]["ceiling_raw"]
    )
    out["width_trigger"] = width_trigger(
        prof,
        n_lif,
        na["live_p50_raw_ms"],
        na["neut_p50_raw_ms"],
        n_lif_paired_live=n_live_pb,
        n_lif_neuter_arm=n_neut_pb,
        band_raw=na["band_raw"],
    )
    out["verdict"] = _verdict(na["band_raw"])
    out["verdict_corrected"] = _verdict(na["band_corrected"]) if na["band_corrected"] else None
    out["verdict_basis"] = "raw decides (T068 3.6 item 3); both printed"
    return out


def run_reading(
    seed=0, wake_ticks=W_TICKS, warmup=WARMUP, pad_iters=PAD_ITERS, resamples=RESAMPLES
):
    """The reds FIRST (2.3), then -- only if the instrument can see -- the live reading."""
    reds = self_check_reds(pad_iters=pad_iters)
    for name, ok, msg, _d in reds:
        print(f"  red {name:<14} {'PASS' if ok else 'FAIL'}  {msg}")
    if not {r[0]: r[1] for r in reds}["padded-share"]:
        return 1, {
            "refused": (
                "INSTRUMENT BLIND (2.6 row 3): the padded-share positive control did not fire, so "
                "a ceiling read here would measure nothing. No reading; fix it first."
            ),
            "reds": [(r[0], r[1], r[2]) for r in reds],
        }
    rep = live_reading(seed=seed, wake_ticks=wake_ticks, warmup=warmup, resamples=resamples)
    rep["reds"] = [(r[0], r[1], r[2]) for r in reds]
    # THE ROSTER IS BUILT ONCE, HERE, where the criterion lives -- never re-expressed in the
    # printer. Two copies of one criterion can disagree, and a printed "all held" beside rc 1 is
    # worse than silence; conjunct 7 is not reconstructible from `rep` at all, because it compares
    # against the REQUESTED tick count and only this scope holds it. Measured cause: a real
    # `controls FAILED` run printed every conjunct it does print as holding, because the two terms
    # that can fail silently -- the neuter's install and its call count -- reach no line of output.
    terms = [
        ("reds", all(r[1] for r in reds)),
        ("neuter_installed", bool(rep["neuter_installed"])),
        ("neuter_calls_one_tick", rep["neuter_calls_one_tick"] > 0),
        ("sham_installed", bool(rep["sham_installed"])),
        ("sham_calls_zero", rep["sham_calls_one_tick"] == 0),
        ("sham_within_noise_band", abs(rep["sham_arm"]["ceiling_raw"]) <= SHAM_NOISE_BAND),
        ("paired_wake_ticks", rep["paired_wake_ticks"] == wake_ticks),
        ("census_agrees", bool(rep["census_cross_check"]["agree"])),
    ]
    rep["controls"] = dict(terms)
    controls_ok = all(v for _, v in terms)
    rep["controls_ok"] = controls_ok
    return (0 if controls_ok else 1), rep


def _print_reading(rep):
    na, sa, p, wt = rep["neuter_arm"], rep["sham_arm"], rep["profile"], rep["width_trigger"]
    print(
        f"  width: {rep['n_lif_per_brain_profiled']} live LIFNeuron per brain during the profile "
        f"({rep['n_lif_per_brain_at_build']} at build; census cross-check "
        f"{rep['census_cross_check']}; all three brains after the run: "
        f"{rep['n_lif_all_three_after']}); paired wake ticks "
        f"{rep['paired_wake_ticks']}/{rep['ticks_attempted']} attempted, "
        f"phase-misaligned {rep['phase_misaligned']}"
    )
    pb = rep["n_lif_paired_block"]
    print(
        f"  per-arm width at the END of the paired timing block (live/neut/sham): "
        f"{pb['live']}/{pb['neut']}/{pb['sham']} -- the width the ceiling was TIMED at. The profile "
        f"block then ticks the LIVE arm alone, {pb['live']} -> {rep['n_lif_per_brain_profiled']}, "
        f"so the number above is the POST-PROFILE LIVE width. A narrower neuter arm makes it faster "
        f"for a reason other than the neutering: a THIRD attribution bias, ceiling UP and N* DOWN"
    )
    print(
        f"  shares over {p['wake_ticks_profiled']} wake ticks ({p['total_tottime_s']} s tottime): "
        f"object path {p['object_path_share']}, I/O {p['io_share']} {p['io_top']}, "
        f"rest {round(1 - (p['object_path_share'] or 0), 5)}"
    )
    print(
        f"  RAW       p50 live {na['live_p50_raw_ms']} ms / neutered {na['neut_p50_raw_ms']} ms "
        f"-> ceiling {na['ceiling_raw']} band {na['band_raw']}"
    )
    print(
        f"  CORRECTED p50 live {na['live_p50_corrected']} / neutered {na['neut_p50_corrected']} "
        f"(ref {na['ref_p50_live_ms']} / {na['ref_p50_neut_ms']} ms) -> ceiling "
        f"{na['ceiling_corrected']} band {na['band_corrected']}"
    )
    print(
        f"  ceiling WITHOUT the I/O share: {rep['ceiling_raw_ex_io']} "
        f"(= ceiling / (1 - io_share); the neuter removes no I/O, so the divisor is the whole "
        f"correction. persist=False, so the fsync share is the live daemon's -- disclosed)"
    )
    print(
        f"  MEASUREMENT arm liveness (the arm the ceiling is computed FROM): "
        f"installed={rep['neuter_installed']} calls={rep['neuter_calls_one_tick']} "
        f"probe phase={rep['neuter_probe_phase']} -- the count is from ONE untimed tick of that "
        f"phase, while every paired sample behind the ceiling is AWAKE, so an asleep-phase count is "
        f"roughly a quarter of what the neuter removes on the measured ticks"
    )
    print(
        f"  CONTROL sham arm (zero-share path): ceiling {sa['ceiling_raw']:+} band "
        f"{sa['band_raw']}; installed={rep['sham_installed']} calls={rep['sham_calls_one_tick']} "
        f"probe phase={rep['sham_probe_phase']}"
    )
    print(
        f"  NOISE FLOOR (disclosure, not the statistic): the sham removes no work, so its "
        f"{sa['ceiling_raw']:+} is this instrument's floor. neuter - sham = "
        f"{rep['ceiling_minus_sham']:+}; ceiling exceeds 2x the floor: "
        f"{rep['ceiling_exceeds_noise_floor']}. A ceiling inside the floor means 'below what this "
        "instrument resolves', never a measured gain."
    )
    print(f"  N* {wt['formula']} tol={wt['tol']} n_lif={wt['n_lif']}")
    print(
        f"     IDENTITIES: {wt['identity_profile']}; {wt['identity_neuter']}; {wt['identity_K']} "
        f"-- {wt['identity_note']}"
    )
    print(f"     from the PROFILE: {wt.get('profile')}")
    print(f"     from the NEUTER : {wt.get('neuter')}   agree-ratio {wt['agree_ratio']}")
    _nb = wt.get("neuter") or {}
    if _nb.get("n_star_paired_block") is not None:
        print(
            f"     the neuter's n_star is published on n_lif={_nb.get('n_lif_used')} (post-profile "
            f"live). On the width the paired block actually ran at "
            f"({_nb.get('n_lif_paired_block_live')} live, {_nb.get('n_lif_neuter_arm')} neut) it is "
            f"{_nb['n_star_paired_block']} -- printed beside the published number, never over it"
        )
    print(
        "     CAVEAT (pre-registered): N* is a LOWER bound -- the rest of the tick is held at its "
        "measured cost and in truth grows with the brain, and cProfile inflates c_u more than R."
    )
    print(
        "     CAVEAT (added 2026-09-09, NOT pre-registered): and the object path's cost is taken as "
        "proportional to the LIF census N. c_u is measured at ONE width; its width-invariance is "
        "untested, so the LOWER-bound direction holds only while the path grows at most linearly "
        "in N. Measured: an awake tick enters the path ~5,124 times over ~71-74 DISTINCT stepped "
        "units of a 165-unit census, one unit carrying ~30% of the calls."
    )
    print(
        f"  REPORT verdict={rep['verdict']} (raw) / {rep['verdict_corrected']} (corrected); "
        f"{rep['verdict_basis']}; bar TOLERANCE={TOLERANCE}. The DECISION is the orchestrator's: "
        "this word is a report line and never the exit code."
    )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-check" in argv:
        print("AMDAHL GATE (Gate Zero) self-check -- the three pre-registered reds")
        rows = self_check_reds()
        for name, ok, msg, _d in rows:
            print(f"  {name:<14} {'PASS' if ok else 'FAIL'}  {msg}")
        bad = [r[0] for r in rows if not r[1]]
        print("self-check OK: 3 reds fired their broken arms" if not bad else f"FAILED: {bad}")
        return 1 if bad else 0

    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    ticks = int(argv[argv.index("--ticks") + 1]) if "--ticks" in argv else W_TICKS
    print(f"AMDAHL GATE (Gate Zero) reading -- seed {seed}, W {ticks} paired wake ticks per arm")
    rc, rep = run_reading(seed=seed, wake_ticks=ticks)
    if rep.get("refused"):
        print(f"  {rep['refused']}")
        return rc
    _print_reading(rep)
    # the roster LAST, so run_selfchecks' 2000-char stdout tail cannot truncate the failing term
    false_terms = [k for k, v in rep["controls"].items() if not v]
    print("  controls: " + "; ".join(f"{k}={v}" for k, v in rep["controls"].items()))
    print(
        f"amdahl gate: controls {'HELD' if rep['controls_ok'] else 'FAILED'}"
        + (f": {', '.join(false_terms)}" if false_terms else "")
        + f" (started {rep['started_utc']}, {rep['elapsed_s']} s)"
        " -- exit code carries the CONTROLS, never the verdict"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
