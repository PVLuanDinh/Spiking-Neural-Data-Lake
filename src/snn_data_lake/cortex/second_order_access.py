r"""
Second-order ACCESS metacognition -- a spiking read on "did that content actually
reach my workspace, and how reliable is that read?".

Honest framing (read this first). The brain already has a FIRST-order access layer:
thalamus/reentrant_workspace grades how strongly content holds the workspace
(`access_strength()`), cortex/access_consciousness reports all-or-none ignition +
global availability, and cortex/brain.py folds both into a self-model access
confidence. What was MISSING is the SECOND order: nothing read the ACCESS STATE
ITSELF. The brain could be told "content X is available" but had no estimate of how
trustworthy that report was, so a masked, weakly-sustained, or outright FAILED
ignition was consumed with the same standing as a clean one. This module is that
missing read: a small LIF reliability pool that takes the first-order access signals
as input current and emits a SECOND-ORDER access confidence -- the system's estimate
of its own access reliability, "knowing THAT it knows".

What this is and is NOT. This is FUNCTIONAL metacognition OF INFORMATION ACCESS:
a monitor over routing signals (availability, sustained access strength, ignition
success/failure, first-order decision confidence). It is NOT a claim of phenomenal
experience -- there is no felt state, no qualia, no introspective report, nothing it
is "like" to be this pool. Its confidence is a windowed spike-count firing rate, and
`state()` never exposes any phenomenal-vocabulary key (asserted by claim 5). The
access-vs-phenomenal line drawn in cortex/access_consciousness.py holds here too:
this module lives entirely on the ACCESS side, one level up.

The spiking mechanism (why a LIF pool, not a formula). `observe()` drives a
population of LIFNeurons with a current EXCITED by global availability, by the graded
sustained access strength, and by the first-order decision confidence, and INHIBITED
hard by an IGNITION FAILURE (content never crossed threshold -> the first-order report
is about nothing). The confidence is read from the EMITTED SPIKES --
spikes / (pop_size * steps) over the window -- not the subthreshold membrane: a pool
charged to just under threshold emits nothing, and such a silent "almost" must not
read as reliable access. Fixed per-neuron bias heterogeneity (per-instance
random.Random(seed)) turns the all-or-none LIF into a graded population rate.

API:
  observe(availability, access_strength, ignited, first_order_conf) -> conf in [0,1]
  hold()        -> True when the second-order confidence is too low to act on the content
  calibration() -> type-2 discrimination (auroc2) of the second-order confidence against
                   whether the content was ACTUALLY globally available (ignited AND
                   availability >= 0.5), over the running history, read by THE ONE scorer
                   `metacognitive_monitor.track_error` (T077, 2026-09-15; P2 -- the thresholded-
                   call agreement it returned before is retired; 0.5 chance, 0.0 no evidence)

MEASURED LIMITS -- KEPT, NOT TUNED AWAY (the self-check prints the true numbers).
  * The monitor errs on MASKED content: a tick that ignites but reaches only a minority
    of taps sits right at the hold threshold, so some masked trials are called
    "available" when they were not. Those errors are left in the stream and are the
    only reason calibration is under 1.0.
  * Available-but-weakly-sustained content (ignited, fully broadcast, but sustain ~0
    and a low first-order read) clears the hold threshold only BARELY -- its mean
    confidence sits just above c_hold, so the margin there is thin, not comfortable.
  * The single strongest cue is the ignition-failure term, so much of the work is
    "was there an ignition at all". The self-check therefore runs an IGNITION-FLAG-ECHO
    control (predict availability = the ignited bit); the monitor must beat it, or it
    would be adding nothing over reading that one bit.

Reuses (does NOT reimplement): cortex/lif_neuron.LIFNeuron. Mirrors the style of
cortex/metacognitive_monitor.py (which monitors LEARNING reliability; this one
monitors ACCESS reliability). Deterministic, ASCII-only, per-instance RNG.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.second_order_access
"""

import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate._mathutils import clamp01

# phenomenal vocabulary that MUST NEVER appear in the functional readout (claim 5).
_BANNED = ("conscious", "sentient", "phenomenal", "subjective", "aware", "feels", "qualia")


#: MERGED 2026-08-15 into `substrate/_mathutils.clamp01` (8 copies -> 1). This file's copy WAS the
#: canonical form -- it coerced up front, and that is the behaviour that survived. Alias kept so no
#: call site changes; the MECHANISM is singular.
_clamp01 = clamp01


class SecondOrderAccess:
    """A spiking SECOND-ORDER monitor of the brain's own access state.

    observe(availability, access_strength, ignited, first_order_conf) drives a LIF pool
    and reads a second-order access confidence from its EMITTED spikes; hold() abstains
    when that confidence is too low to act on the content; calibration() is the running
    agreement between the monitor's own call and whether the content was actually
    globally available. Functional only -- no phenomenal claim. Deterministic."""

    def __init__(
        self,
        pop_size=24,
        steps=16,
        tau_m=10.0,
        w_avail=0.35,  # global availability (fraction of taps reached) -> excitation
        w_strength=0.45,  # graded sustained access strength -> excitation
        w_conf=0.35,  # first-order decision confidence -> excitation
        w_fail=0.90,  # IGNITION FAILURE -> hard inhibition (the report is about nothing)
        base=0.0,
        bias_spread=0.35,
        c_hold=0.30,  # confidence below this -> HOLD (too unreliable to act on)
        avail_majority=0.5,  # availability at/above which content counts as truly available
        seed=0,
    ):
        self.rng = random.Random(seed)  # per-instance rng (fixes bias heterogeneity)
        self.pop_size = int(pop_size)
        self.steps = int(steps)
        self.w_avail = float(w_avail)
        self.w_strength = float(w_strength)
        self.w_conf = float(w_conf)
        self.w_fail = float(w_fail)
        self.base = float(base)
        self.c_hold = float(c_hold)
        self.avail_majority = float(avail_majority)
        self.pool = [LIFNeuron(tau_m=tau_m) for _ in range(self.pop_size)]
        self.bias = [self.rng.uniform(-bias_spread, bias_spread) for _ in range(self.pop_size)]
        # last-observation cache
        self._conf2 = 0.0  # second-order confidence (windowed spike-count firing rate)
        self._spikes = 0
        self._avail = 0.0
        self._strength = 0.0
        self._ignited = False
        self._conf1 = 0.0
        # running calibration bookkeeping
        self._n = 0
        self._agree = 0
        self._hist = []  # (second_order_conf, called_available, truly_available) per observation

    # -- the spiking read ------------------------------------------------------------------------
    def observe(self, availability, access_strength, ignited, first_order_conf):
        """Read the first-order ACCESS state and return a second-order confidence in [0,1].

        The drive is excited by global availability + the graded sustained access strength
        + the first-order decision confidence, and inhibited by an IGNITION FAILURE. The
        returned confidence is the windowed SPIKE-COUNT firing rate of the pool (emitted
        spikes, never the subthreshold membrane)."""
        a = _clamp01(availability)
        s = _clamp01(access_strength)
        ig = bool(ignited)
        c1 = _clamp01(first_order_conf)
        self._avail, self._strength, self._ignited, self._conf1 = a, s, ig, c1
        drive = (
            self.base
            + self.w_avail * a
            + self.w_strength * s
            + self.w_conf * c1
            - (
                0.0 if ig else self.w_fail
            )  # failed ignition -> the first-order read is about nothing
        )
        spikes = 0
        for neuron, b in zip(self.pool, self.bias):
            neuron.reset_state()  # fresh integration window per observation (no cross-leak)
            current = drive + b
            for _ in range(self.steps):
                spikes += neuron.step(current)
        self._spikes = spikes
        self._conf2 = spikes / float(self.pop_size * self.steps)
        # calibration: did the monitor's own call match actual global availability?
        called = self._conf2 >= self.c_hold
        truly = ig and a >= self.avail_majority
        self._n += 1
        if called == truly:
            self._agree += 1
        self._hist.append((self._conf2, called, truly))
        return self._conf2

    # -- readout ---------------------------------------------------------------------------------
    def second_order_confidence(self):
        """The last second-order access confidence in [0,1] -- the pool's spike-count rate."""
        return self._conf2

    def hold(self):
        """True when the second-order confidence is too low to act on the content: the brain's
        own estimate says the content did not reliably reach access, so do not commit to it."""
        return self._conf2 < self.c_hold

    def calibration(self):
        """Type-2 discrimination (auroc2) of the second-order confidence against whether the content
        was ACTUALLY globally available (ignited AND availability >= majority), over the running history,
        read by THE ONE calibration scorer of the tree (T077, 2026-09-15: `metacognitive_monitor.track_error`
        -- P2, the thresholded-call agreement this method returned until then is retired). 0.5 is chance
        (the fixed prior's exact value). Returns 0.0 before any observation, or while the history cannot
        resolve tracking on its own permutation null -- an HONEST 'no evidence yet', not a score. The call
        itself (`hold`) and its per-trial errors are unchanged; only the score's statistic changed."""
        if self._n == 0:
            return 0.0
        from snn_data_lake.cortex.metacognitive_monitor import track_error

        r = track_error([h[0] for h in self._hist], [1 if h[2] else 0 for h in self._hist], seed=0)
        return float(r["auroc2"]) if r["readable"] else 0.0

    def n_observed(self):
        """Number of observations folded into the running calibration."""
        return self._n

    def state(self):
        """Functional snapshot for the brain's per-tick state(). NO phenomenal key, ever."""
        return {
            "second_order_confidence": round(self._conf2, 6),
            "second_order_hold": self.hold(),
            "second_order_calibration": round(self.calibration(), 6),
            "second_order_n": self._n,
        }


# ============================== un-riggable self-check (seed=0) ==================================
# A mixed access stream with GENUINELY FAILING controls. Every trial's ground truth (was the
# content actually globally available?) is known to the harness and NEVER fed to the monitor as
# a label -- it only ever sees the four first-order signals.

N_STRONG = 30  # clean ignition, sustained, confident -> truly available
N_FAILED = 25  # ignition FAILED -> not available, first-order read is about nothing
N_MASKED = 15  # ignited but reached a minority of taps (masked) -> not available
N_FALSE_ALARM = 15  # ignition failed yet the FIRST-ORDER read is loud -> not available
N_WEAK_OK = 15  # ignited + fully broadcast but sustain ~0 and first-order low -> IS available


def _stream(seed=0):
    """Build the mixed access stream: (kind, availability, access_strength, ignited, conf1, truth)."""
    rng = random.Random(seed)
    trials = []
    for _ in range(N_STRONG):
        trials.append(("strong", 1.0, rng.uniform(0.70, 0.95), True, rng.uniform(0.70, 0.95), True))
    for _ in range(N_FAILED):
        trials.append(("failed", 0.0, 0.0, False, rng.uniform(0.00, 0.35), False))
    for _ in range(N_MASKED):
        trials.append(
            ("masked", 0.25, rng.uniform(0.05, 0.25), True, rng.uniform(0.15, 0.40), False)
        )
    for _ in range(N_FALSE_ALARM):
        trials.append(("false_alarm", 0.0, 0.0, False, rng.uniform(0.75, 0.95), False))
    for _ in range(N_WEAK_OK):
        trials.append(
            ("weak_ok", 1.0, rng.uniform(0.02, 0.15), True, rng.uniform(0.05, 0.25), True)
        )
    rng.shuffle(trials)
    return trials


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _experiment(seed=0):
    """Run the mixed stream + the failing controls. Returns pure numbers (comparable across runs)."""
    trials = _stream(seed)

    mon = SecondOrderAccess(seed=seed)
    confs, truths, calls, kinds, conf1s, igs = [], [], [], [], [], []
    for kind, avail, strength, ig, c1, truth in trials:
        c2 = mon.observe(avail, strength, ig, c1)
        confs.append(c2)
        truths.append(truth)
        calls.append(not mon.hold())
        kinds.append(kind)
        conf1s.append(c1)
        igs.append(ig)

    # T077 (2026-09-15): calibration and its four controls are read on THE ONE scorer, auroc2 via
    # metacognitive_monitor.track_error, each control being the SAME statistic of a rival score on
    # the SAME trials (the thresholded-call accuracies they were until then are retired, P2).
    from snn_data_lake.cortex.metacognitive_monitor import track_error

    y = [1 if t else 0 for t in truths]
    scored = track_error(confs, y, seed=seed)
    cal = mon.calibration()

    # ---- CONTROL 1 (must FAIL): the SAME confidences paired with SHUFFLED truths -> chance
    #      (the median of the scorer's own permutation draws; its q95 is printed beside). --------
    cal_shuffled = scored["shuffled_auroc2"]
    cal_q95 = scored["perm_q95"]

    # ---- CONTROL 2 (must FAIL): the FIRST-ORDER read alone as the score. It is fooled by the
    #      loud-but-never-ignited false alarms -- exactly what the second order adds. -----------
    cal_first_order = track_error(conf1s, y, seed=seed)["auroc2"]

    # ---- CONTROL 3 (must FAIL): the fixed prior -- a constant score reads exactly 0.5. ---------
    cal_majority = track_error([0.5] * len(y), y, seed=seed)["auroc2"]

    # ---- CONTROL 4 (must FAIL): echo the IGNITION FLAG alone as the score. This is the cheap
    #      strategy the monitor must beat -- otherwise the second order re-reads one bit. -------
    cal_ignition_echo = track_error([1.0 if ig else 0.0 for ig in igs], y, seed=seed)["auroc2"]

    # ---- per-kind confidence means AND per-kind ERROR counts (the honest breakdown) ------------
    by_kind, err_kind = {}, {}
    for kind in ("strong", "failed", "masked", "false_alarm", "weak_ok"):
        by_kind[kind] = _mean([c for c, k in zip(confs, kinds) if k == kind])
        err_kind[kind] = sum(1 for c, k, t in zip(calls, kinds, truths) if k == kind and c != t)

    # ---- clean streams for claims (a) and (b) --------------------------------------------------
    hi = SecondOrderAccess(seed=seed)
    hi_confs = [
        hi.observe(1.0, 0.85, True, 0.85) for _ in range(20)
    ]  # strong, reliably ignited access
    hi_hold = hi.hold()

    lo = SecondOrderAccess(seed=seed)
    lrng = random.Random(31 + seed)
    lo_confs = []
    lo_holds = []
    for i in range(20):  # weak / masked / outright FAILED access
        if i % 3 == 0:
            lo_confs.append(lo.observe(0.0, 0.0, False, lrng.uniform(0.0, 0.9)))  # failed ignition
        elif i % 3 == 1:
            lo_confs.append(lo.observe(0.25, 0.10, True, 0.2))  # masked (minority of taps)
        else:
            lo_confs.append(lo.observe(0.0, 0.0, False, 0.1))  # dark tick
        lo_holds.append(lo.hold())

    # ---- structural: the confidence is a SPIKE COUNT, and a silent sub-threshold pool reads 0 --
    probe = SecondOrderAccess(seed=seed)
    probe.observe(1.0, 0.9, True, 0.9)
    spike_count_is_integer = abs(probe._conf2 * probe.pop_size * probe.steps - probe._spikes) < 1e-9
    probe.observe(0.0, 0.0, False, 0.9)
    silent_spikes = probe._spikes

    # ---- empty-monitor honesty: no observations -> no evidence -------------------------------
    empty_cal = SecondOrderAccess(seed=seed).calibration()

    st = hi.state()
    no_phenomenal_key = all(all(tok not in key for tok in _BANNED) for key in st)

    return {
        "cal": cal,
        "cal_q95": cal_q95,
        "cal_shuffled": cal_shuffled,
        "cal_first_order": cal_first_order,
        "cal_majority": cal_majority,
        "cal_ignition_echo": cal_ignition_echo,
        "by_kind": by_kind,
        "err_kind": err_kind,
        "errors": sum(err_kind.values()),
        "hi_mean": _mean(hi_confs),
        "hi_hold": hi_hold,
        "lo_mean": _mean(lo_confs),
        "lo_all_hold": all(lo_holds),
        "n": mon.n_observed(),
        "spike_count_is_integer": spike_count_is_integer,
        "silent_spikes": silent_spikes,
        "empty_cal": empty_cal,
        "no_phenomenal_key": no_phenomenal_key,
        "c_hold": mon.c_hold,
        "state": st,
    }


def _p(lines, s):
    """Append an ASCII-only line (Windows cp1252 safety)."""
    if any(ord(ch) > 127 for ch in s):
        raise AssertionError("non-ASCII output")
    lines.append(s)


def main():
    r = _experiment(0)
    lines = []
    _p(lines, "=" * 78)
    _p(lines, "SECOND-ORDER ACCESS -- spiking metacognition OF the workspace access state")
    _p(lines, "=" * 78)
    _p(
        lines,
        "knowing THAT it knows: a LIF pool reads availability / sustained strength / ignition-",
    )
    _p(lines, "failure / first-order confidence and emits a SECOND-ORDER access confidence.")
    _p(lines, "-" * 78)
    _p(lines, "trial kind          mean 2nd-order conf   truly available?   wrong calls")
    for kind, truth, n in (
        ("strong", "yes", N_STRONG),
        ("weak_ok", "yes", N_WEAK_OK),
        ("masked", "no ", N_MASKED),
        ("failed", "no ", N_FAILED),
        ("false_alarm", "no ", N_FALSE_ALARM),
    ):
        _p(
            lines,
            f"  {kind:<18}{r['by_kind'][kind]:>13.3f}          {truth}"
            f"              {r['err_kind'][kind]:>2}/{n}",
        )
    _p(lines, "-" * 78)
    _p(
        lines,
        f"(a) strong reliable access : mean confidence {r['hi_mean']:.3f}, hold={r['hi_hold']}",
    )
    _p(
        lines,
        f"(b) weak/masked/FAILED     : mean confidence {r['lo_mean']:.3f}, hold on every tick={r['lo_all_hold']}",
    )
    _p(
        lines,
        f"(c) calibration over {r['n']} mixed trials : {r['cal']:.3f}  (auroc2 on THE ONE scorer, T077; "
        f"chance 0.5; its permutation q95 {r['cal_q95']:.3f})",
    )
    _p(lines, f"    CONTROL shuffled-truth (chance)   : {r['cal_shuffled']:.3f}  <- must FAIL")
    _p(lines, f"    CONTROL first-order-echo only     : {r['cal_first_order']:.3f}  <- must FAIL")
    _p(lines, f"    CONTROL majority-class baseline   : {r['cal_majority']:.3f}  <- must FAIL")
    _p(lines, f"    CONTROL ignition-flag echo only   : {r['cal_ignition_echo']:.3f}  <- must FAIL")
    _p(
        lines,
        f"    HONEST LIMIT: {r['errors']} wrong calls remain (see the per-kind column); the "
        f"weak-but-available trials clear hold by only {r['by_kind']['weak_ok'] - r['c_hold']:+.3f}",
    )
    _p(lines, "(d) determinism                       : two seed=0 runs identical")
    _p(
        lines,
        f"    spike-derived: conf*pop*steps integer={r['spike_count_is_integer']}, "
        f"failed-ignition spikes={r['silent_spikes']}",
    )
    _p(
        lines,
        "FUNCTIONAL metacognition of information ACCESS only -- NOT a claim of phenomenal experience.",
    )
    _p(lines, "=" * 78)
    for line in lines:
        print(line)

    # ---- (a) strong, reliably ignited access -> HIGH second-order confidence -------------------
    assert r["hi_mean"] > 0.60, (
        f"strong reliably-ignited access must yield HIGH second-order confidence, got {r['hi_mean']:.3f}"
    )
    assert not r["hi_hold"], "the monitor must NOT hold on strong, reliably ignited access"
    assert r["by_kind"]["strong"] > 0.60, (
        f"the strong trials in the mixed stream must read high, got {r['by_kind']['strong']:.3f}"
    )

    # ---- (b) weak / masked / failed access -> LOW confidence and hold() ------------------------
    assert r["lo_mean"] < 0.15, (
        f"weak/masked/failed access must yield LOW second-order confidence, got {r['lo_mean']:.3f}"
    )
    assert r["lo_all_hold"], "the monitor must HOLD on every weak/masked/failed access tick"
    assert r["by_kind"]["failed"] == 0.0 and r["by_kind"]["false_alarm"] == 0.0, (
        "a FAILED ignition must silence the pool even when the first-order read is loud"
    )
    assert r["by_kind"]["masked"] < r["by_kind"]["strong"], (
        "masked access must read below clean access"
    )

    # ---- (c) it DISCRIMINATES: calibration clearly beats the failing controls ------------------
    assert r["cal"] > r["cal_shuffled"] + 0.20, (
        f"calibration {r['cal']:.3f} must clearly beat the shuffled-truth control {r['cal_shuffled']:.3f}"
    )
    assert r["cal"] > r["cal_first_order"] + 0.05, (
        f"calibration {r['cal']:.3f} must beat the first-order-echo control {r['cal_first_order']:.3f} "
        "-- otherwise the second order adds nothing"
    )
    assert r["cal"] > r["cal_majority"] + 0.10, (
        f"calibration {r['cal']:.3f} must beat the majority-class baseline {r['cal_majority']:.3f}"
    )
    assert r["cal"] > r["cal_ignition_echo"] + 0.05, (
        f"calibration {r['cal']:.3f} must beat the ignition-flag-echo control {r['cal_ignition_echo']:.3f} "
        "-- otherwise the monitor only re-reads the ignited bit"
    )
    # T077 (2026-09-15): the honest limit is the CALLS, which still err on the masked trials; the
    # rank-based calibration on this constructed stream reads 1.000 (every truly-available trial
    # outranks every unavailable one) and is NOT asserted below 1.0 any more -- the constructed
    # stream is not the live path (T077 2.2 arm D measured the live hold saturated).
    assert r["errors"] > 0, (
        f"the calls must NOT be perfect -- {r['errors']} real wrong calls are kept, not tuned away"
    )

    # ---- structural + honesty ------------------------------------------------------------------
    assert r["spike_count_is_integer"], (
        "confidence * pop_size * steps must be an exact SPIKE COUNT (the read is spike-derived)"
    )
    assert r["silent_spikes"] == 0, "a failed ignition must emit NO spikes -> confidence exactly 0"
    assert r["empty_cal"] == 0.0, (
        "calibration before any observation must be an honest 0.0 (no evidence)"
    )
    assert r["no_phenomenal_key"], (
        f"state() must contain NO phenomenal key {_BANNED}: {list(r['state'])}"
    )

    # ---- (d) determinism -----------------------------------------------------------------------
    assert _experiment(0) == r, "two seed=0 runs must be byte-for-byte identical"

    print(
        f"self-check OK: 2nd-order access confidence {r['hi_mean']:.3f} on strong reliable ignition vs "
        f"{r['lo_mean']:.3f} on weak/masked/FAILED (holds on all); calibration {r['cal']:.3f} beats "
        f"shuffled {r['cal_shuffled']:.3f} / first-order-echo {r['cal_first_order']:.3f} / majority "
        f"{r['cal_majority']:.3f} / ignition-echo {r['cal_ignition_echo']:.3f}; HONEST LIMIT {r['errors']} "
        f"wrong calls kept (masked {r['err_kind']['masked']}/{N_MASKED}); functional access metacognition "
        "only, NOT phenomenal; deterministic"
    )


if __name__ == "__main__":
    main()
