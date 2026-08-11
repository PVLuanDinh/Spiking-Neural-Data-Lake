r"""
GWT DEEP -- the tree's honest PHASE-BINDING demonstration: theta-gamma multiplexed binding
with the SPIKING-vs-RATE control that the consciousness-concepts review named the single
most important test for this project.

THE IGNITION / RE-ENTRY HALF IS RETIRED (2026-08-11, P2 -- one mechanism per job). This
module carried its own all-or-none workspace ignition (an IGNITE_TH gate on every
broadcast) and re-entrant sustain (extra theta cycles at `_REENTRY * drive` reverberation)
BESIDE `thalamus/reentrant_workspace`, which is THE ignition/re-entry mechanism -- it
already absorbed `reasoning/gwt_reentry.py`'s predictive-completion (2026-08-06, ported to
`::expect`, neuter-A/B verified). The duplicated half is deleted: the gate, the cycles,
their two demos and their two self-check asserts. What remains is the BINDING half,
CREDITED 2026-08-07 on the exact benchmark that killed the workspace `bind_*` family
(Jaccard, 120 trials/K, all-fired-pairs strawman; ledger row + CHANGELOG at `66b6168a`),
and `tests/instruments/gwt_deep_split_ab.py` re-runs that form on BOTH sides of this
surgery: spiking Jaccard 1.0000 at every K, strawman 0.5000/0.2500 at K=2/4 -- unchanged.
The sub-threshold-ignition red in `_reds()` became a sub-RHEOBASE red (drive 0.2 <
LEAK*V_TH = 0.25): the same two criteria go red deterministically with no ignition gate.

  SCORING (read this before the claim below). Pair recovery is scored by F1 -- precision AND recall.
  It was recall-only until 2026-08-04, which made the spiking side unfalsifiable (a reader emitting
  the whole KxK cross product scores 1.000 under recall), and the broadcast was in fact superposing:
  every object was driven from t=0, so "phase multiplexing" never ran. Both are fixed; see
  `spiking_broadcast` and `_pair_accuracy` for the measured before/after.

  THE SPIKING-vs-RATE CONTROL (the honest question: what do spikes BUY here?):
    The workspace must BIND K objects, each a (color, shape) pair, and BROADCAST the pairings. The
    SPIKING workspace phase-multiplexes them (Lisman-Jensen theta-gamma: each object in its own gamma
    slot, its color+shape firing SYNCHRONOUSLY) so a downstream reader recovers the pairings from
    synchrony. The RATE workspace -- SAME topology, SAME hysteresis, continuous activation, no spike
    timing -- broadcasts a SUPERPOSED activation vector: all K colors and K shapes are "on", but which
    color binds which shape is GONE (the classic superposition / binding problem).

  HONEST FRAMING (per the consciousness-pursuit directive: pursue it, do not fake it):
    * K=1 (one object): spiking and rate TIE -- no binding is needed, so spikes are NOT load-bearing.
      Reported, not hidden. Single-item selection does not justify spikes.
    * K>=2: spiking BINDS via synchrony where rate superposes and loses the pairing -> spikes are
      load-bearing for MULTI-ITEM binding. This is the honest, specific answer to "why spikes."
    * Caveat (contested): binding-by-synchrony works here GIVEN a synchrony-reading reader (Singer/Gray;
      critiqued by Shadlen-Movshon). It is ONE solution, not the only one -- a conjunctive/tensor-product
      rate code could also bind. We do not claim synchrony is how brains bind, only that it works and
      that it is what distinguishes the spiking workspace from the rate one on this task.

  This is a step of the consciousness program (GWT re-entry + binding), built honestly, not a claim of
  phenomenal consciousness. LIF loops, deterministic. PROMOTED reasoning/ -> thalamus/ 2026-08-11
  (P4c, the split's recorded follow-on): registered in REGIONS + run_selfchecks, dormancy-EXEMPT
  "permanent" as a benchmark demonstration -- nothing on the live tick binds K objects; the
  consumer is `tests/instruments/gwt_deep_split_ab.py` plus this __main__.

  PYTHONPATH=src python -m snn_data_lake.thalamus.gwt_deep
"""

import random

T_SLOT = 6  # time steps per gamma slot (one bound object)
LEAK = 0.25  # LIF leak per step
V_TH = 1.0  # LIF threshold


class LIF:
    """A leaky integrate-and-fire neuron: v <- v*(1-leak) + I; spike + reset when v >= threshold."""

    __slots__ = ("v",)

    def __init__(self):
        self.v = 0.0

    def step(self, i):
        self.v = self.v * (1.0 - LEAK) + i
        if self.v >= V_TH:
            self.v = 0.0
            return 1
        return 0


def _objects(n_color, n_shape, k, rng):
    """K objects, each a (color, shape) pair with DISTINCT colors and DISTINCT shapes (so the pairing
    is the only thing to recover -- identity of the parts is unambiguous, only the BINDING is at risk)."""
    cols = rng.sample(range(n_color), k)
    shps = rng.sample(range(n_shape), k)
    return list(zip(cols, shps))


# ----------------------------------------------------------------------------------------------------
# SPIKING workspace: theta-gamma phase-multiplexed broadcast (binds by synchrony)
# distinct-from: cortex/theta_gamma_code.py -- that is a working-memory BUFFER (store/recall API,
#   its own dormancy lane, P4C_BACKLOG row on theta_gamma_code); this is the measured
#   spiking-vs-rate CONTROL demonstration. Their merge question was examined at the bind_*
#   retirement and the three-way merge REFUTED on a pre-registered null (P4C_BACKLOG); this
#   module's binding half PASSED the same-form benchmark. One demo, one buffer, no shared job.
# ----------------------------------------------------------------------------------------------------
def spiking_broadcast(objs, n_color, n_shape, drive=1.0):
    """Broadcast the K objects PHASE-MULTIPLEXED on the theta-gamma clock (Lisman-Jensen): one THETA
    cycle holds K gamma slots of T_SLOT ticks, object k owns gamma slot k. Within a slot the object's
    color and shape LIFs see the SAME current, so they spike together (bound); across slots no two
    objects are ever driven at once, which is what "each object in its own gamma slot" has to mean.

    Returns raster, raster[t] = (fired_colors, fired_shapes). (Until 2026-08-11 this also carried an
    all-or-none ignition gate and re-entrant sustain cycles -- retired, see the module docstring;
    `thalamus/reentrant_workspace` is the one ignition/re-entry mechanism.)

    CORRECTED 2026-08-04, and the correction is the point. The previous version put EVERY object into
    an `active` list at t=0 and gave the out-of-slot ones `drive*0.55`, which is 2.2x the LIF rheobase
    (LEAK*V_TH = 0.25) -- so every object fired on essentially every tick from the start and the
    "phase multiplexing" never happened. Measured on the old code at K=3: the reader recovered all 9
    cross-product pairs for 3 true objects, precision 0.333. It scored 1.000 anyway because
    _pair_accuracy was recall-only; see the note there. Both halves had to be fixed -- a decoder that
    emits everything and a metric that only counts hits are the same defect seen from two sides."""
    col = [LIF() for _ in range(n_color)]
    shp = [LIF() for _ in range(n_shape)]
    raster = []
    k_obj = len(objs)
    for t in range(k_obj * T_SLOT):
        slot = t // T_SLOT  # which object owns this gamma slot
        c, s = objs[slot]
        fc, fs = [], []
        if col[c].step(drive):
            fc.append(c)
        if shp[s].step(drive):
            fs.append(s)
        raster.append((fc, fs))
    return raster


def spiking_decode(raster):
    """Reader: bind color+shape that fired SYNCHRONOUSLY (same time step) into a pair. Synchrony IS the
    binding cue. Returns the set of recovered (color, shape) pairs."""
    pairs = set()
    for fc, fs in raster:
        for c in fc:
            for s in fs:
                pairs.add((c, s))  # co-firing -> bound
    return pairs


# ----------------------------------------------------------------------------------------------------
# RATE workspace: SAME topology, but continuous activation, NO spike timing
# ----------------------------------------------------------------------------------------------------
def rate_broadcast(objs, n_color, n_shape, drive=1.0):
    """Rate control: broadcast a SUPERPOSED activation vector -- every object's color and shape
    channel is 'on' with the same magnitude. No timing, so which color binds which shape is not
    represented. Returns (color_act, shape_act). (Its ignition gate retired 2026-08-11 with the
    spiking arm's, keeping the two arms topology-matched -- the control isolates SPIKE TIMING.)"""
    color_act = [0.0] * n_color
    shape_act = [0.0] * n_shape
    for c, s in objs:  # superposition: all channels summed, no pairing
        color_act[c] += drive
        shape_act[s] += drive
    return color_act, shape_act


def rate_decode(color_act, shape_act, k, rng):
    """Reader for the rate broadcast: recover the K active colors and K active shapes (easy -- magnitudes
    say which are on), but the PAIRING is absent, so it must GUESS a bijection. Fair best effort: pair
    them in a fixed canonical order (sorted), which is right only by luck. Returns recovered pairs."""
    cols = sorted(range(len(color_act)), key=lambda i: -color_act[i])[:k]
    shps = sorted(range(len(shape_act)), key=lambda i: -shape_act[i])[:k]
    # no binding info -> canonical (sorted) pairing; a random bijection would do equally well in expectation
    cols_sorted = sorted(cols)
    shps_sorted = sorted(shps)
    return set(zip(cols_sorted, shps_sorted))


def _pair_accuracy(true_pairs, got_pairs):
    """F1 over the recovered (color, shape) pair set: the harmonic mean of PRECISION and RECALL.

    This was `len(true & got) / len(true)` -- RECALL ONLY -- until 2026-08-04, and that made the
    headline binding claim unfalsifiable on the spiking side: a reader that emits the entire KxK
    cross product scores a perfect 1.000 under recall, because it necessarily contains every true
    pair. Measured on the pre-fix broadcast at K=3: 9 pairs returned for 3 true objects, recall
    1.000, PRECISION 0.333. Superposing and being scored as binding is exactly the failure the rate
    control is supposed to demonstrate, so the metric was crediting the spiking workspace for the
    defect it exists to contrast against.

    F1 punishes the cross-product reader (K=2 -> 0.667, K=3 -> 0.500, K=4 -> 0.400) while leaving a
    genuinely multiplexed reader at 1.000, so the K>=2 assert below now has a reachable red -- and
    the pre-fix broadcast IS that red -- for the exact-set criterion and the K=4 floor. That is no
    longer a docstring claim: `_reds()` below RECONSTRUCTS the pre-fix broadcast and RUNS it against
    those two every time this self-check runs, per P8/F11 and the "if a gate lands green, ship the
    reconstruction of the defect it was built for" rule. It is NOT the red for the K=2 margin --
    see `_reds` for the measurement that says why."""
    t, g = set(true_pairs), set(got_pairs)
    # Both empty-set guards come BEFORE the divisions. `if not g` alone left the symmetric case
    # (empty TRUE set, non-empty GOT) raising ZeroDivisionError on `hit / len(t)`, because that
    # tuple-assignment evaluates both quotients before the `hit == 0` return below can fire
    # (measured 2026-08-05: `_pair_accuracy([], [(1, 2)])` -> ZeroDivisionError).
    hit = len(t & g)
    if hit == 0:  # covers empty t, empty g, and disjoint sets -- no overlap means no F1
        return 0.0
    precision, recall = hit / len(g), hit / len(t)
    return 2.0 * precision * recall / (precision + recall)


def _superposing_broadcast(objs, n_color, n_shape, drive=1.0, sustain_ticks=4):
    """The PRE-FIX broadcast, reconstructed verbatim, so the binding criteria below can be shown to
    go RED against it on every run instead of in a docstring (P8/F11; the same discipline
    `tests/test_red_reachability.py` uses -- a gate that lands green ships the reconstruction of the
    defect it was built for).

    Every object is placed in `active` at t=0 and the out-of-slot ones get `drive*0.55`, which is
    2.2x the LIF rheobase (LEAK*V_TH = 0.25), so all K objects fire on nearly every tick and the
    reader returns the whole KxK cross product. This is NOT a second broadcast mechanism competing
    with `spiking_broadcast` (P2): nothing outside `_reds()` may call it, it is the negative
    control, and it is what makes the positive one falsifiable. (Its ignition-gate line left with
    the ignition retirement 2026-08-11; the reconstructed DEFECT -- the superposing drive pattern
    below -- is byte-unchanged.)"""
    col = [LIF() for _ in range(n_color)]
    shp = [LIF() for _ in range(n_shape)]
    raster = []
    active = [[c, s, T_SLOT + sustain_ticks] for c, s in objs]
    for t in range(len(objs) * T_SLOT + sustain_ticks):
        slot = t // T_SLOT
        fc, fs = [], []
        for a in active:
            c, s, left = a
            if left <= 0:
                continue
            cur = drive if objs.index((c, s)) == slot else (drive * 0.55)
            if col[c].step(cur):
                fc.append(c)
            if shp[s].step(cur):
                fs.append(s)
            a[2] -= 1
        raster.append((fc, fs))
    return raster


def _reds(n_color, n_shape, trials):
    """CONSTRUCT AND RUN the input that makes each binding criterion in main() FAIL, and assert it
    fails. Returns [(name, went_red)] for the banner.

    WHICH CRITERION CATCHES WHICH DEFECT -- measured 2026-08-04 over 40 seeds x 200 trials, and the
    result changes what belongs in which row. Against the SUPERPOSING reader the spiking F1 is
    exactly 0.6667 at K=2 (2 of 4 emitted pairs correct) and exactly 0.4000 at K=4 (4 of 16), while
    the rate arm ranges 0.435-0.590 at K=2:

        K=4 floor  (`> 0.9`)                 went RED in 40/40 seeds  -- deterministic
        K=2 margin (`> rate + 0.15`)         went RED in 12/40 seeds  -- a coin flip

    So the K=2 margin does NOT reliably separate a superposing reader from a binding one: 0.6667
    sits only ~0.017 above the 0.15 bar over a rate arm at ~0.50, inside the sampling noise of 200
    trials. That is recorded rather than repaired, because the repair would be re-tuning 0.15 until
    it says what we want, which is the failure P1 and P8 both name. The K=4 floor is the criterion
    that actually distinguishes the two readers, and it is the one driven by the superposing
    control here. The K=2 row is instead driven by a sub-RHEOBASE `drive` -- a plain input to
    `spiking_broadcast` -- which reddens it deterministically (spiking 0.000 vs rate ~0.495).
    (Until the 2026-08-11 ignition retirement that input was sub-IGNITION-threshold; drive 0.2 <
    LEAK*V_TH = 0.25 pins every membrane at a 0.8 fixed point, so the raster is empty the same
    deterministic way and the same two criteria go red.)"""
    rows = []
    rng = random.Random(99)

    # (1) EXACT-SET: the superposing reader returns the KxK cross product, not the presented pairs.
    objs = _objects(n_color, n_shape, 3, rng)
    r = _superposing_broadcast(objs, n_color, n_shape, drive=1.0)
    rows.append(("exact-set recovery (vs superposing reader)", spiking_decode(r) != set(objs)))

    # (2) K=4 binding floor, driven by the superposing reader and scored the way main() scores it.
    sp = 0.0
    for _ in range(trials):
        o = _objects(n_color, n_shape, 4, rng)
        rr = _superposing_broadcast(o, n_color, n_shape, drive=1.0)
        sp += _pair_accuracy(o, spiking_decode(rr))
    rows.append(("K=4 binding floor (vs superposing reader)", not (sp / trials > 0.9)))

    # (3)+(4) a sub-rheobase drive leaves the spiking arm with an empty raster (F1 0.0) while the
    #         rate arm still reads its superposition, so both the K=2 margin and the K=1 tie break.
    sub = {}
    for k in (1, 2):
        sp = rt = 0.0
        for _ in range(trials):
            o = _objects(n_color, n_shape, k, rng)
            rr = spiking_broadcast(o, n_color, n_shape, drive=0.2)
            sp += _pair_accuracy(o, spiking_decode(rr))
            ca, sa = rate_broadcast(o, n_color, n_shape, drive=1.0)
            rt += _pair_accuracy(o, rate_decode(ca, sa, k, rng))
        sub[k] = (sp / trials, rt / trials)
    rows.append(("K=2 spiking-beats-rate margin", not (sub[2][0] > sub[2][1] + 0.15)))
    rows.append(("K=1 spiking-rate TIE", not (abs(sub[1][0] - sub[1][1]) < 0.05)))

    for name, went_red in rows:
        assert went_red, (
            f"NO REACHABLE RED for '{name}': the constructed failing input did not make the "
            f"criterion fail, so it cannot go red and is worthless (P8/F11 -- delete it, do not "
            f"weaken it)"
        )
    return rows


def main():
    rng = random.Random(0)
    N_COLOR, N_SHAPE = 8, 8
    TRIALS = 200
    print("=" * 90)
    print(
        "GWT DEEP -- phase-multiplexed BINDING; the SPIKING-vs-RATE control (what do spikes buy?)"
    )
    print("=" * 90)
    print(
        "  (ignition + re-entry demos retired 2026-08-11 -- thalamus/reentrant_workspace is the one"
    )
    print("   mechanism for both; this module is the tree's phase-binding demonstration)")

    # ---- BINDING: spiking vs rate across K, over many trials ----
    print("-" * 90)
    print(
        f"{'K objects':>9} | {'spiking pair-F1':>16} | {'rate pair-F1':>13} | spikes load-bearing?"
    )
    print("-" * 90)
    results = {}
    for k in (1, 2, 3, 4):
        sp_acc = rt_acc = 0.0
        for _ in range(TRIALS):
            objs = _objects(N_COLOR, N_SHAPE, k, rng)
            r = spiking_broadcast(objs, N_COLOR, N_SHAPE, drive=1.0)
            sp_acc += _pair_accuracy(objs, spiking_decode(r))
            ca, sa = rate_broadcast(objs, N_COLOR, N_SHAPE, drive=1.0)
            rt_acc += _pair_accuracy(objs, rate_decode(ca, sa, k, rng))
        sp_acc /= TRIALS
        rt_acc /= TRIALS
        results[k] = (sp_acc, rt_acc)
        verdict = (
            "TIE (spikes not needed)"
            if abs(sp_acc - rt_acc) < 0.05
            else f"YES (+{(sp_acc - rt_acc) * 100:.0f} pts)"
        )
        print(f"{k:>9} | {sp_acc:>15.1%} | {rt_acc:>12.1%} | {verdict}")
    print("-" * 90)
    print("  HONEST: at K=1 spiking and rate TIE -- single-item selection does NOT justify spikes.")
    print("  At K>=2 the spiking workspace BINDS via synchrony (theta-gamma slots) where the rate")
    print(
        "  workspace superposes and loses the pairing -- THAT is what spikes buy here (multi-item"
    )
    print(
        "  binding). Contested (Shadlen-Movshon) + reader-dependent + not the only solution; reported honestly."
    )
    print("=" * 90)

    # ---- self-checks. "Un-riggable" was a CLAIM, not a property, until 2026-08-04: the two binding
    # asserts below rode on a recall-only score that a cross-product reader satisfies perfectly, so
    # neither could go red for a binding failure. They can now -- see _pair_accuracy -- and the red
    # is the pre-fix broadcast itself, which `_reds()` rebuilds and RUNS at the end of this
    # function rather than asserting in prose. (The ignition-all-or-none and re-entry-sustains
    # asserts that stood here retired 2026-08-11 with their mechanism -- those properties are
    # asserted where the one mechanism lives, thalamus/reentrant_workspace.) ----
    # PRECISION is the half the old metric could not see: a multiplexed reader must not emit pairs
    # that were never presented. Without this, K>=2 below is satisfiable by emitting everything.
    _objs_p = _objects(N_COLOR, N_SHAPE, 3, random.Random(99))
    _r_p = spiking_broadcast(_objs_p, N_COLOR, N_SHAPE, drive=1.0)
    _got_p = spiking_decode(_r_p)
    assert _got_p == set(_objs_p), (
        f"the spiking reader must recover EXACTLY the presented pairs, not a superposition: "
        f"got {sorted(_got_p)} for {sorted(_objs_p)}"
    )
    # TWO-SIDED, corrected 2026-08-04. This read `results[1][0] - results[1][1] < 0.05`, which is a
    # TAUTOLOGY: at K=1 `rate_decode` returns the single presented pair for every input (measured:
    # F1 exactly 1.000 on 20,000 trials), and F1 is bounded above by 1.0, so the one-sided
    # difference is <= 0 for EVERY possible spiking result -- including a spiking arm that emits
    # nothing. It could not go red, which is the shape F11 says to delete rather than keep.
    # `abs(...)` is what "TIE" already means in the verdict column above and is strictly STRONGER
    # (it adds the lower bound the one-sided form was missing); its red is row 4 of _reds().
    assert abs(results[1][0] - results[1][1]) < 0.05, (
        f"at K=1 spiking and rate must TIE (spikes not load-bearing for single-item): "
        f"{results[1][0]:.2f} vs {results[1][1]:.2f}"
    )
    assert results[2][0] > results[2][1] + 0.15, (
        f"at K=2 spiking must beat rate on binding (the load-bearing case): "
        f"{results[2][0]:.2f} vs {results[2][1]:.2f}"
    )
    assert results[4][0] > 0.9, f"spiking binding should stay high at K=4 ({results[4][0]:.2f})"
    # Every binding criterion above is now SHOWN falsifiable by running the input that breaks it.
    reds = _reds(N_COLOR, N_SHAPE, TRIALS)
    print("-" * 90)
    print(
        "  reachable reds (the constructed input that makes each criterion FAIL, run every time):"
    )
    for name, _ in reds:
        print(f"    [RED OK] {name}")
    print(
        f"self-check OK: spiking-vs-rate control is HONEST -- ties at K=1 (spikes not needed), "
        f"spiking binds {results[2][0]:.0%} vs rate {results[2][1]:.0%} at K=2 (spikes load-bearing "
        f"for multi-item binding via synchrony)."
    )
    return 0


if __name__ == "__main__":
    main()
