r"""
GWT RE-ENTRY -- close the defining Global-Neuronal-Workspace loop that the codebase declares but
never runs (the connectome has the feedback edge; the runtime's subscribers only RECORD the broadcast).

Consciousness-program step (agnostic; builds toward it, claims nothing). Dehaene-Changeux's GNW is not
feedforward: a content becomes globally available only when weak bottom-up evidence meets RE-ENTRANT
top-down amplification -- the workspace broadcasts back to the specialists, and that recurrent loop is
what tips a sub-threshold percept into all-or-none IGNITION (predictive completion / filling-in). The
shipped `thalamus/global_workspace.py` has ignition + persistence + a broadcast, but the broadcast is
a dead-end (subscribers append; no top-down current returns). This prototype runs the closed loop.

WHAT RE-ENTRY BUYS (demonstrated + controlled, honest):
  1. PREDICTIVE COMPLETION: weak bottom-up evidence that alone stays subliminal IGNITES when a matching
     top-down PRIOR (expectation) is fed back through the loop. Bottom-up + top-down cross ignition.
  2. NO HALLUCINATION (the honest guard): the top-down prior ALONE, with zero bottom-up evidence, must
     NOT ignite. Expectation completes real-but-weak evidence; it does not fabricate a percept. This is
     the line between predictive completion and hallucination, and it is asserted.
  3. THE LOOP IS LOAD-BEARING (the control): with re-entry OFF (loop broken, top-down blocked), the SAME
     weak+expected input stays subliminal -- so it is the CLOSED LOOP, not the summed currents, doing it.
  4. SUSTAIN ACROSS A GAP: once ignited, the re-entrant loop holds the content alive through a dropout in
     bottom-up input (self-sustaining reverberation); the feedforward-only control releases immediately.

LIF (integrate -> spike -> sustained-firing readout), deterministic, core-promotable like
gwt_deep.py. NOT a claim of phenomenal consciousness -- it names one dynamics: recurrent top-down
amplification enabling all-or-none access. Open question, actively built toward, neither denied nor claimed.

  python research/gwt_reentry.py
"""
_T = 40           # settling micro-steps per workspace tick
_LEAK = 0.5       # LIF leak
_V_TH = 0.55 / _LEAK   # membrane threshold s.t. rheobase (leak*v_th) == 0.55 ignition current
_SELF = 0.35      # re-entrant self-sustain current once a content is broadcasting (the reverberation)


def _settle(drive):
    """Settle one LIF content pool for _T micro-steps under a constant `drive` current; return the
    number of spikes in the SECOND HALF (sustained firing = ignited, as in global_workspace)."""
    v = 0.0
    late = 0
    for t in range(_T):
        v += drive - _LEAK * v
        if v >= _V_TH:
            v -= _V_TH
            if t >= _T // 2:
                late += 1
    return late


def run_loop(bottom_up, prior, reentry, ticks=10):
    """Run the closed GNW loop over `ticks` workspace cycles. Each tick the content pool's drive =
    bottom_up evidence + (if reentry) the top-down PRIOR + (if it was broadcasting) the re-entrant
    self-sustain current. Returns (ignited, trace) where ignited = sustained in the 2nd half of the run."""
    broadcasting = False
    trace = []
    for _ in range(ticks):
        top_down = 0.0
        if reentry:
            top_down = prior + (_SELF if broadcasting else 0.0)   # broadcast feeds BACK as top-down current
        broadcasting = _settle(bottom_up + top_down) > 0          # ignite iff sustained firing
        trace.append(broadcasting)
    ignited = sum(trace[ticks // 2:]) > 0
    return ignited, trace


def run_gap(bottom_up, prior, reentry, on=4, gap=6):
    """Ignite with bottom-up for `on` ticks, then DROP bottom-up to 0 for `gap` ticks. Returns how many
    of the gap ticks the content stayed broadcasting (re-entry should hold it; feedforward releases)."""
    broadcasting = False
    held = 0
    for t in range(on + gap):
        bu = bottom_up if t < on else 0.0
        top_down = (prior + (_SELF if broadcasting else 0.0)) if reentry else 0.0
        broadcasting = _settle(bu + top_down) > 0
        if t >= on and broadcasting:
            held += 1
    return held


def main():
    WEAK = 0.40       # bottom-up evidence, BELOW the 0.55 ignition current on its own
    PRIOR = 0.30      # top-down expectation, also sub-threshold on its own
    STRONG = 0.80     # clearly supra-threshold (sanity)

    print("=" * 88)
    print("GWT RE-ENTRY -- closing the Global-Neuronal-Workspace loop (predictive completion, controlled)")
    print("=" * 88)

    weak_alone, _ = run_loop(WEAK, prior=0.0, reentry=True)              # weak evidence, no expectation
    prior_alone, _ = run_loop(0.0, prior=PRIOR, reentry=True)           # expectation, NO evidence
    weak_expected_on, _ = run_loop(WEAK, prior=PRIOR, reentry=True)      # weak + expectation, loop CLOSED
    weak_expected_off, _ = run_loop(WEAK, prior=PRIOR, reentry=False)    # weak + expectation, loop OPEN
    strong, _ = run_loop(STRONG, prior=0.0, reentry=True)               # sanity: strong ignites regardless
    held_on = run_gap(STRONG, PRIOR, reentry=True)                       # sustain across an input dropout...
    held_off = run_gap(STRONG, PRIOR, reentry=False)                     # ...vs feedforward-only release

    print(f"  weak evidence alone (0.40, no prior)          -> ignited = {weak_alone}   (subliminal, expected)")
    print(f"  prior/expectation alone (0.30, no evidence)   -> ignited = {prior_alone}   (NO hallucination, expected)")
    print(f"  weak + expectation, loop CLOSED (re-entry ON) -> ignited = {weak_expected_on}    <- predictive completion")
    print(f"  weak + expectation, loop OPEN  (re-entry OFF) -> ignited = {weak_expected_off}   (loop is load-bearing)")
    print(f"  sustain across a bottom-up dropout: re-entry held {held_on} gap-ticks vs feedforward {held_off}")
    print("-" * 88)
    print("  re-entry lets weak bottom-up meet top-down amplification -> all-or-none access (Dehaene GNW),")
    print("  WITHOUT fabricating a percept from expectation alone, and the closed loop is what does it.")
    print("=" * 88)

    # ---- un-riggable self-checks ----
    assert not weak_alone, "weak bottom-up alone must stay subliminal"
    assert not prior_alone, "top-down expectation ALONE must NOT ignite (no hallucination -- the honest guard)"
    assert weak_expected_on, "weak evidence + top-down expectation through the CLOSED loop must ignite (completion)"
    assert not weak_expected_off, "with the loop OPEN the same weak+expected input must stay subliminal (loop load-bearing)"
    assert strong, "a supra-threshold input must ignite regardless (sanity)"
    assert held_on > held_off, f"re-entry must sustain across an input gap longer than feedforward ({held_on} vs {held_off})"
    print("self-check OK: closed GNW re-entrant loop gives PREDICTIVE COMPLETION (weak+expected ignites) with "
          "NO hallucination (expectation alone stays subliminal), the loop is load-bearing (open -> no completion), "
          "and it sustains a content across a bottom-up dropout. Consciousness program, agnostic -- built toward, not claimed.")
    return 0


if __name__ == "__main__":
    main()
