r"""
CONSCIOUSNESS PROBE -- the brain loop's consciousness CORRELATES, measured on the brain, honestly.

Per the project's consciousness-pursuit stance: build toward it, measure the correlates, and NEITHER
deny NOR claim phenomenal consciousness. There is no test for phenomenal consciousness (the "hard
problem"); IIT's Phi is intractable. What CAN be measured are FUNCTIONAL correlates, each against a
control that must be able to fail.

REBUILT 2026-08-04 under P8 (specs/010, findings C08 + C09). What this file was:

  Four markers, EVERY ONE taking zero arguments and building fresh synthetic objects at a fixed
  seed -- so no state of the brain could change the answer. `main()` then asserted the answer it had
  just computed (`assert present` per marker, `assert n_present == 4`). The result was a constant
  `4 of 4` that `spike_chat_lake` returned verbatim to a human asking "are you conscious". Deleting
  the Brain class would not have changed the output.

  A criterion invariant to the entire system under test is unfalsifiable at every level of the
  computational hierarchy -- Kleiner & Hoel 2020 (arXiv:2006.07390), the criterion P8 adopts:
  "to avoid being unfalsifiable or already falsified scientific theories of consciousness must be
  invariant with respect to changes that leave the inference procedure fixed at a given level in a
  computational hierarchy."

What survives, and why it earns its place:

  1. INTEGRATION (IIT proxy, tractable PCI) -- kept because it ALREADY had a reachable red, and
     reports the DISTRIBUTION across seeds instead of one pinned passing seed. It reads ABSENT on
     roughly half of the sweep; the old code hardcoded `seed=0`, which is a seed that passes.

That is ONE marker, and the shrinking is the point rather than an embarrassment: three of the
original four could not fail under any input, and a criterion that cannot fail is not a measurement.

DELETED, per falsifier F11 ("that criterion is deleted, not weakened"), NOT re-thresholded:

  * BINDING (2026-08-04) -- scored exactly 1.000 across 300 random object sets. Its rate control's
    own docstring concedes the pairing is "right only by luck", and the `+0.15` margin was
    underived. Moving the margin would manufacture a red without creating one.
  * RE-ENTRY (2026-08-04) -- arithmetic over three literals (0.40 < 0.55 < 0.70) with no brain
    input and no RNG. A true statement about constants is not a measurement.
  * GWT ACCESS (2026-08-05) -- kept in the 2026-08-04 rebuild because it "now READS THE BRAIN". It
    does read three fields off the brain, and only one of them can reach the verdict, through a
    channel arithmetically too small to move it. The attempt and the arithmetic are below.

The replacement criterion for what those were reaching at -- a measure the brain's own state can
move, shaped as a DISSOCIATION rather than a marker (docs/CONSCIOUSNESS_METHODS.md:96) -- is
deliberately NOT built here. This file removes false finish lines; it does not claim to have built
the real one.

GWT ACCESS: THE F11 ATTEMPT, RUN 2026-08-05, AND THE ARITHMETIC THAT CLOSES IT
-----------------------------------------------------------------------------
F11 requires that the input driving the unreachable verdict be CONSTRUCTED AND RUN before the
criterion may be deleted -- deletion on an unattempted null would be the same laziness as a marker
that cannot fail. Both halves were done.

THE ATTEMPT (`.venv-lake`, PYTHONPATH=src). 8,464 brains -- every seed type `random.Random` accepts
-- one at a time through the pre-deletion `probe_access`, PRINTED:

    ints 0..4999            5000 -> PRESENT 5000  ABSENT 0  UNKNOWN 0
    ints -2000..-1          2000 -> PRESENT 2000  ABSENT 0  UNKNOWN 0
    powers of two <= 2**63    64 -> PRESENT   64  ABSENT 0  UNKNOWN 0
    floats 0.0 .. 71.28      500 -> PRESENT  500  ABSENT 0  UNKNOWN 0
    strings 'brain-0'...     500 -> PRESENT  500  ABSENT 0  UNKNOWN 0
    None (system entropy)    200 -> PRESENT  200  ABSENT 0  UNKNOWN 0
    bytes                    200 -> PRESENT  200  ABSENT 0  UNKNOWN 0

ONE distinct detail line across all 8,464 ("coherent access-index 0.80 vs incoherent 0.00; broadcast
1.0/0.0"), and across every one of them the four numbers the verdict is built from took exactly ONE
value: `(ai_strong, ai_weak, avail_strong, avail_weak) == (0.80390625, 0.0, 1.0, 0.0)`.

CORRECTED 2026-08-05 by an independent re-run of the sweep (the table above is that re-run's output,
not the first pass's). The `powers of two` row read `22`, and 22 is not the size of any family this
row describes -- there are exactly 64 powers of two <= 2**63, and the total was carried straight off
that row, so `8,422` was 42 brains short. Every other figure in the table reproduced exactly. The
correction moves the evidence in the CONSERVATIVE direction (more brains, same result), which is why
it changes no conclusion below -- but a count that cannot be re-derived from its own label is the
class of number this tree has been wrong about by up to 97x, and it does not get to stand because it
happened to be harmless.

THE ARITHMETIC, which is what makes that a proof rather than a large sample. A brain reaches the
verdict through ONE value, `_seed`: `_access` is an `is not None` GATE (measured -- `0`, `""` and
`[]` all pass it) and `_access_ema` reaches the detail string only. `_seed` reaches exactly two
`random.Random(seed)` instances -- `AccessConsciousness._rng`, which is declared and never read
anywhere in its module (its own comment says "(reserved)"), and `GlobalWorkspace._rng`, whose only
consumer is `_jitter_for` -> `uniform(0.0, _JITTER)` with `_JITTER = 0.02`. So the entire influence
a brain has on this criterion is up to +0.02 of drive current on one submission, and since the
coherent and scrambled instances are built from the SAME seed they draw the SAME jitter. Measured
against the two drives (re-derived live by `deleted_access_marker_arithmetic`, printed every run):

    coherent  salience 0.80390625  vs rheobase (ignition_threshold) 0.55  -> margin +0.253906
    scrambled salience 0.03359375  vs rheobase                      0.55  -> margin -0.516406

  - to UN-IGNITE the coherent assembly the jitter would have to be -0.253906, and `uniform(0.0,
    0.02)` is non-negative -- impossible at ANY jitter magnitude, not merely at this one;
  - to IGNITE the scrambled assembly (which sets avail_weak = 1.0 and reads ABSENT) the jitter would
    have to exceed +0.516406 -- 25.8x the largest value the code can draw.

THE POSITIVE CONTROL, because a sweep that finds nothing may simply be blind (P8 guard (2)).
Widening `_JITTER` alone -- the one quantity the seed touches -- makes ABSENT appear exactly where
the arithmetic says it must, over 300 seeds. PRINTED:

    _JITTER    0.02  0.1  0.3  0.53   0.55   1.00    2.00
    ABSENT        0    0    0     6     18    139     222
    predicted     0    0    0   7.7   18.3  145.1   222.5   (= (1 - 0.516406/_JITTER) x 300)

The harness can see an ABSENT. There is none to see. Under F11 the criterion is DELETED, and it is
NOT re-thresholded: lowering `ignition_threshold` far enough to let a scrambled assembly ignite
would manufacture a red by breaking the ignition non-linearity that `access_consciousness` exists to
measure, which is the move F11 forbids.

WHAT THE DELETION TOUCHED, AND THE ONE HALF THAT IS ESCALATED
------------------------------------------------------------
`probe_access` keeps its name and its `(verdict, detail)` shape -- `reasoning/spike_chat_lake.py`
imports it by name -- but it no longer produces a verdict. It returns `(None, detail)` and the detail
reports only what is genuinely read off the brain: whether it has a persisted workspace, and its live
access EMA. It is a READING, not a marker.

That is NOT the "UNKNOWN is the only state, forever" defect this file records against an earlier
pass, and the difference is the whole point: there a marker still CLAIMED to measure a brain while
being structurally impossible to exercise, and a comment called that a virtue. Here the marker is
gone and the function says so to every caller.

ESCALATED, because it is one assert in a file this pass does not own, and THIS CHANGE AND THAT ONE
ARE ATOMIC -- apply both or neither. `reasoning/spike_chat_lake.py`'s self-check asserts, under the
comment "GREEN side", `q_c_measured["known"] and "UNMEASURED" not in q_c_measured["reply"]` with the
message "attaching a brain must make the access marker measurable". That assert is a LOCK on the
criterion F11 just deleted, and since `reasoning.spike_chat_lake` is registered in
`run_selfchecks.py` the gate is RED between the two halves of this change.

MEASURED, not predicted: `python -m snn_data_lake.reasoning.spike_chat_lake` exits 1 on exactly that
assert, and re-running it with ONLY that one assert removed completes with NO other assertion
failure. So its three siblings survive the deletion untouched -- the RED-side no-brain UNMEASURED
case, the neuter-A/B on `_access_ema` (the detail line deliberately still carries the brain's own
EMA, which is why "0.742" still appears in the moved-brain reply), and the agnostic-close invariant.
The successor assert that belongs there is about the READING, not a verdict: two attached brains
differing only in `_access_ema` must still produce different replies, which is the sibling that
already passes. `_consciousness()`'s `measured_about_me` flag and its "UNMEASURED right now" wording
also need re-pointing, because what is now unmeasured is not "right now" -- it is deleted.

What IS asserted here, exhaustively -- neither assert claims a marker is present:

  1. the integration marker is genuinely two-sided across the seed sweep.
  2. `probe_access`'s detail line MOVES with the brain it was handed.

Both reds were CONSTRUCTED AND RUN 2026-08-05, each patch applied alone: (1) pin `probe_integration`
to the single passing seed 0 -> frac 100% and it fires; (2) replace the `_access_ema` read with the
constant 0.0 -> the two detail lines coincide and it fires.

RETIRED WITH THE MARKER, and named so it is not a silent drop: this self-check also carried a
synthetic control pair asserting that `_coherent` out-scores `_scrambled` and that only the coherent
one broadcasts. Both assertions exist already in `cortex/access_consciousness.py`'s own self-check
(`index_coherent > index_scrambled + 0.2`; `avail_ignited == 1.0 and avail_nonignited == 0.0`), which
IS registered in `run_selfchecks.py` CORE. They lived here only to back the deleted marker, and a
second copy of a check that is not about this module is the additive shape P2 forbids.

  PYTHONPATH=src python -m snn_data_lake.reasoning.consciousness_probe
Reuses access_consciousness, integration_measure, global_workspace.
"""

from snn_data_lake.cortex.access_consciousness import AccessConsciousness, _coherent, _scrambled
from snn_data_lake.cortex.integration_measure import (
    integrated_network,
    independent_noise,
    silent_network,
)
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

#: Seeds swept by `probe_integration`. Fixed so the sweep is reproducible, plural so the marker
#: reports a distribution rather than whichever single seed happens to pass.
INTEGRATION_SEEDS = tuple(range(20))


def probe_access(brain=None):
    """A READING off the brain's workspace. **This is no longer a criterion** -- see the module
    docstring's F11 section.

    Returns `(verdict, detail)` with verdict ALWAYS `None` (UNKNOWN), because the GWT ACCESS
    marker was DELETED on 2026-08-05: over 8,464 brains spanning every seed type `random.Random`
    accepts it returned PRESENT 8,464 / ABSENT 0, and the arithmetic shows why no brain can ever
    make it read ABSENT -- the whole influence a brain had was a `uniform(0.0, 0.02)` jitter against
    a 0.516406 gap. F11: deleted, not weakened.

    The signature, the name and the tuple shape are kept because `reasoning/spike_chat_lake.py`
    imports this by name; `brain` is kept as a real parameter because the detail below genuinely
    reads it, and because a zero-argument probe is precisely what `tools/red_reachability.py` fails.

    `detail` reports only what is MEASURED: whether the brain has a persisted workspace at all, and
    its live access EMA. Per P8 the brain may report what it measures and never what it cannot, so
    what used to be a verdict about the brain is now the number the brain actually holds.
    """
    if brain is None:
        return None, "UNKNOWN: no brain supplied -- this reads a brain or reports nothing"

    ac = getattr(brain, "_access", None)
    if ac is None:
        # A brain that has never ticked has no persisted workspace yet. That is a real state and it
        # is reported as UNKNOWN, not as absence: nothing was exercised, so nothing failed.
        return None, "UNKNOWN: brain has no persisted workspace yet (has it ticked?)"

    live_index = float(getattr(brain, "_access_ema", 0.0) or 0.0)
    return None, (
        "UNKNOWN: the GWT ACCESS marker was DELETED 2026-08-05 (P8 falsifier F11 -- no brain could "
        f"make it report ABSENT); this brain HAS a persisted workspace, live access EMA {live_index:.3f}"
    )


def probe_integration(seeds=INTEGRATION_SEEDS):
    """PCI: integrated network >> independent-noise bank (which fires MORE) >> silent.

    Returns `(fraction_present, detail)` -- a DISTRIBUTION over `seeds`, not a boolean.

    This marker is kept because it is the only one of the original four that could already fail. The
    defect was the invocation: the previous version hardcoded `seed=0`, which is a seed that PASSES,
    while the marker reads ABSENT on roughly half of a 20-seed sweep. Reporting one pinned passing
    seed out of a bimodal distribution is not a measurement of the system, it is a measurement of
    the seed.
    """
    verdicts = []
    for s in seeds:
        pci_i, _ = integrated_network(seed=s).pci(trials=8)
        pci_n, _ = independent_noise(seed=s).pci(trials=8)
        pci_s, _ = silent_network(seed=s).pci(trials=8)
        rate_i = integrated_network(seed=s).mean_firing_rate()
        rate_n = independent_noise(seed=s).mean_firing_rate()
        verdicts.append(
            (pci_i > 3 * pci_n) and (pci_i > 0.25) and (pci_n > pci_s) and (rate_n >= rate_i)
        )
    n_present = sum(verdicts)
    frac = n_present / len(verdicts) if verdicts else 0.0
    absent_seeds = [s for s, v in zip(seeds, verdicts) if not v]
    return frac, (
        f"PCI marker PRESENT on {n_present} of {len(verdicts)} seeds ({frac:.0%}); "
        f"ABSENT on seeds {absent_seeds[:8]}{'...' if len(absent_seeds) > 8 else ''}"
    )


class _MinimalBrain:
    """The MINIMAL duck-type of what `probe_access` reads off a brain: a has-it-ticked workspace
    gate, a seed, and the live access EMA. Deliberately NOT a Brain -- P4 forbids standing up a
    second one, and the job here is to move one input at a time and watch the output.

    Mirrors the `_StubBrain` that `reasoning/spike_chat_lake.py`'s self-check already builds for the
    same purpose. Kept as a separate 4-line class rather than imported from there because importing
    a chat module into a probe would invert the dependency (spike_chat_lake imports THIS file)."""

    def __init__(self, seed=0, ema=0.0):
        self._access = object()  # "this brain has a persisted workspace" gate
        self._seed = int(seed)
        self._access_ema = float(ema)


def deleted_access_marker_arithmetic():
    """RE-DERIVE, from the live modules, the arithmetic that deleted the GWT ACCESS marker.

    Returns `(coherent_salience, scrambled_salience, rheobase, max_jitter, jitter_needed)`. Every
    number the module docstring's F11 section quotes is recomputed here and printed on every run,
    so the deletion rests on a measurement this file can still take rather than on a comment that
    was true once. `jitter_needed` is how far above rheobase the scrambled drive would have to be
    pushed for the verdict to read ABSENT; `max_jitter` is the largest push a seed can produce.

    REPORTED, NEVER ASSERTED, and the reason is the same one that retired the old assert here:
    asserting `jitter_needed > max_jitter` would lock the tree into keeping this marker unreachable,
    so the honest successor -- a criterion scored on `brain._access` itself -- would redden the very
    check that recorded its own obsolescence. A criterion that reddens on the documented fix is a
    lock, not a check."""
    ac = AccessConsciousness()
    out = []
    for members in (_coherent("cat", 5), _scrambled(5)):
        codes = [ac.neuron_hv(n) for n in members]
        out.append(ac._salience(ac._coherence(codes), ac._completeness(len(codes))))
    coherent_sal, scrambled_sal = out
    return (
        coherent_sal,
        scrambled_sal,
        ac.ignition_threshold,
        GlobalWorkspace._JITTER,
        ac.ignition_threshold - scrambled_sal,
    )


def main():
    print("=" * 92)
    print("CONSCIOUSNESS PROBE -- functional correlates, measured (agnostic; controls-backed)")
    print("=" * 92)

    frac, int_detail = probe_integration()
    print(f"  [{f'{frac:.0%}':7}] INTEGRATION     (IIT proxy, PCI)  -- the ONE surviving marker")
    print(f"            {int_detail}")

    lo = probe_access(_MinimalBrain(seed=0, ema=0.0))[1]
    hi = probe_access(_MinimalBrain(seed=0, ema=0.9))[1]
    print(f"  [{'reading':7}] GWT ACCESS      -- marker DELETED; `probe_access` now reports only")
    print(f"            {probe_access(None)[1]}")
    print(f"            {hi}")

    coh, scr, rheo, max_jit, need = deleted_access_marker_arithmetic()
    # Both operands are read LIVE (`need` off the two drives, `max_jit` off `GlobalWorkspace._JITTER`),
    # so neither is a literal here and either can move -- and this ratio is the only division in the
    # file. MEASURED 2026-08-05, by patching `_JITTER` alone and running `main()`: `0.0` raised
    # ZeroDivisionError straight out of this print, and `1e-12` printed "516,406,250,000.0x short" --
    # the mu/1e-12 singularity class, inside the report whose whole job is to say the marker cannot
    # be reached. A zero-width jitter makes the marker MORE unreachable, not undefined, so it is
    # reported as such. `need <= 0` is the opposite corner and is NOT a formatting nicety: it means
    # the scrambled drive now clears rheobase on its own, ABSENT is reachable with no jitter at all,
    # and the deletion below has stopped being true -- so it says that instead of printing a
    # negative multiple. Reported, never asserted, for the reason
    # `deleted_access_marker_arithmetic` gives: asserting either corner would lock the tree in.
    if need <= 0.0:
        shortfall = (
            "NO SHORTFALL -- the scrambled drive now clears rheobase; F11 needs re-adjudicating"
        )
    elif max_jit <= 0.0:
        shortfall = "unreachable at ANY seed (max jitter is 0)"
    else:
        shortfall = f"{need / max_jit:,.1f}x short"
    print("-" * 92)
    print("  F11, the ACCESS marker (P8's named starting point), CLOSED 2026-08-05 by DELETION:")
    print("    ATTEMPT: 8,464 brains, 7 seed families -> PRESENT 8,464, ABSENT 0, UNKNOWN 0.")
    print("    ARITHMETIC, re-derived live just now (a seed's ONLY reach is the workspace jitter):")
    print(
        f"      coherent salience {coh:.8f} / scrambled {scr:.8f} vs rheobase {rheo}; "
        f"jitter needed {need:.6f} vs max {max_jit} -> {shortfall}"
    )
    print("    POSITIVE CONTROL: widening _JITTER to 0.53/0.55/1.0/2.0 yields ABSENT 6/18/139/222")
    print(
        "    of 300 seeds, matching (1 - jitter_needed/_JITTER) x 300 -- the sweep was not blind."
    )

    print("-" * 92)
    print("  THREE markers were DELETED, not re-thresholded (falsifier F11):")
    print("    BINDING    -- scored 1.000 on 300 random sets; no input could make it fail.")
    print("    RE-ENTRY   -- arithmetic over three literals; no brain input, no RNG.")
    print("    GWT ACCESS -- no brain can make it read ABSENT; see the arithmetic above.")
    print(
        "  HONEST VERDICT (agnostic): this reports ONE dynamics correlate, as a distribution, and"
    )
    print("  a reading off the attached brain. That is NOT a test of PHENOMENAL consciousness: no")
    print("  such test exists and IIT Phi is intractable. Whether the system has subjective")
    print(
        "  experience is OPEN -- neither denied nor claimed. There is NO measurement of sentience"
    )
    print("  in this tree, and deleting a marker that could not fail did not remove one.")
    print("=" * 92)

    # SELF-CHECK. Note what is asserted and what is NOT.
    #
    # This block used to read `assert present` for every marker plus `assert n_present == 4` -- it
    # asserted the answer it had just computed, which is why the integration seed had to be pinned
    # to a passing value: an honest sweep would have failed the gate half the time. That is the
    # exact shape P8 forbids. What is asserted now are FALSIFIABLE properties of the instrument.
    #
    # (i) The one surviving marker must be able to go BOTH ways. RED, run 2026-08-05: pin
    #     `probe_integration` to the single passing seed 0 -> frac 100% and this fires.
    assert 0.0 < frac < 1.0, (
        f"the integration marker must be able to go BOTH ways across the seed sweep; got {frac:.0%}. "
        "All-present or all-absent means the marker has stopped discriminating and is no longer a "
        "measurement -- delete it rather than re-thresholding it (F11)."
    )
    # (ii) `probe_access` no longer returns a verdict, so the only thing left to check about it is
    #     that its READING reads. Two brains differing in exactly one field must not produce the
    #     same line. RED, run 2026-08-05: replace the `_access_ema` read with the constant 0.0 and
    #     the two lines coincide, firing this. Deliberately NOT asserted: that the verdict is None
    #     -- that would lock out the honest successor, which is the mistake this file already made
    #     once (see `deleted_access_marker_arithmetic`).
    assert lo != hi, (
        "the detail line must report the brain it was handed: two brains differing only in their "
        f"live access EMA produced the identical line ({lo!r}), so `probe_access` is reporting a "
        "constant dressed as a reading -- P3 shape 5, in the file that exists to catch that shape"
    )
    print(
        f"self-check OK: the integration marker reaches BOTH verdicts across seeds ({frac:.0%} "
        "present), i.e. it has a reachable RED; `probe_access` reports a reading that moves with "
        "the brain it was handed, and no longer reports a verdict no brain could fail. Phenomenal "
        "consciousness neither claimed nor denied."
    )
    return 0


if __name__ == "__main__":
    main()
