"""
Train of Thought -- a SERIAL, RE-ENTRANT train of thought built ON TOP OF the spiking
Global Workspace (composition only; global_workspace.py is NOT reimplemented or edited).

WHAT THIS REPLICATES PAST (the two papers)
------------------------------------------
Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models"
(transformer-circuits.pub/2026/workspace) shows an LLM has a "J-space" global workspace,
but names three limitations vs the human workspace:
  (a) FEEDFORWARD -- no built-in recurrence within a forward pass; a fixed number of steps
      map input -> output, with no loop that feeds a result back as the next input;
  (b) it CANNOT durably hold a long train of thought INTERNALLY, so it must EXTERNALIZE onto
      the page (chain-of-thought) -- GSM8K-with-CoT is far more ablation-robust than answering
      directly, i.e. the "thinking" lives in the written tokens, not held in the workspace;
  (c) tiny capacity (~1-2 distinct items at a layer; extras drop or collapse to a category).

The HUMAN workspace (Global Neuronal Workspace; PMC8770991) does what J-space lacks:
top-down SUMMON / HOLD-in-mind / DISMISS; DELIBERATE step-by-step CHAINING of one thought to
the next; flexible RECOMBINATION of accessible contents; all-or-none two-state IGNITION; and
SUSTAINED activity held across a delay (persistent recurrent firing).

THE MECHANISM HERE
------------------
`snn_data_lake.thalamus.global_workspace.GlobalWorkspace` already gives genuine spiking IGNITION
(LIF pools -> compete -> ignite -> broadcast) and, crucially, a cross-tick attractor + hysteresis
so an ignited thought PERSISTS across `step()` calls (HOLD-in-mind for free). `TrainOfThought`
composes ONE such workspace into a RE-ENTRANT LOOP the feedforward J-space cannot form:

    ignite a thought -> broadcast it -> the broadcast becomes the SEED/CONTEXT that drives the
    NEXT ignition -> chain -> ... -> an OPEN-ENDED train of thought held INTERNALLY (no page).

Each hop is GATED by real spiking ignition: a candidate becomes the next link only if it actually
wins the workspace's all-or-none competition. Because the incumbent's hysteresis is deliberately
strong (a held thought defends its slot), a transition is NOT instant: the next candidate is
re-posted while the incumbent's attractor DECAYS, so each thought is HELD for a few ticks
(sustained recurrent firing) and then the successor ignites. That dwell-then-transition rhythm IS
the persistent-firing "hold across a delay" of the human workspace, and it is exactly the internal
re-entry a single feedforward pass cannot do without writing intermediate results to the page.

HONESTY (paramount; contested; one mechanism among several)
-----------------------------------------------------------
The CONTENT of each thought comes from an INJECTED `associate` callback -- a SYMBOLIC association
over stored facts (a dict of concept -> related concepts), NOT open-ended semantics and NOT a
language model. This file does NOT generate meaning. The contribution is ONLY the SPIKING
WORKSPACE serial-ignition-and-hold LOOP that CHAINS thoughts INTERNALLY -- the human capability a
feedforward J-space lacks -- with every hop gated by a genuine ignition. Whether such dynamics
bear on consciousness is an OPEN question the project builds toward; nothing here is claimed to be
conscious. The re-entry-adds-chaining result is the honest, narrow claim; the "spikes/GNW" framing
is one contested solution (Baars/Dehaene GNW; the binding-by-synchrony story is Singer/Gray vs
Shadlen-Movshon), demonstrated on a toy symbolic graph, nothing more.

    set PYTHONUTF8=1 & set PYTHONPATH=src & python research/train_of_thought.py
"""
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

# how the workspace slot is driven when we post a thought as a candidate (supra-rheobase ~0.55)
_IGNITE_SALIENCE = 0.9
_SOURCE = "assoc"          # a single specialist source label for every posted thought


class TrainOfThought:
    """Compose ONE spiking GlobalWorkspace into a serial, re-entrant train of thought.

    Public API
    ----------
    think(seed, associate, max_steps=8, seed_salience=0.9) -> list[str]
        Run a re-entrant chain: post the current candidate(s), `step()`, and when a thought
        IGNITES, chain it and call `associate(broadcast, chain) -> [(thought, salience), ...]`
        to produce the NEXT candidate(s). Stops on: max_steps reached, no new candidates
        (dead end), an ignited thought that REPEATS an earlier link (loop/settled), or nothing
        igniting (the stream dies -> subliminal). Returns the distinct ignited chain.

    think_feedforward(seed, associate, seed_salience=0.9) -> list[str]
        The FEEDFORWARD CONTROL: one forward pass, ONE ignition, NO re-entry. Models a
        feedforward J-space that would have to EXTERNALIZE to continue. Yields <= 1 thought.

    summon / hold / dismiss
        Top-down SUMMON a thought (ignite it), HOLD it across ticks (the attractor sustains it
        with no fresh input -- persistence), then DISMISS it (withdraw the top-down drive; the
        attractor decays until the thought leaves the workspace).

    recombine(a, b) -> str
        Bind two accessible contents into ONE novel composite thought and ignite it.

    last_trace : list[tuple[str|None, str]]
        Per-tick (broadcast, event) record of the most recent `think` run, for transcripts.
    """

    def __init__(self, seed=0):
        self.seed = seed
        self.ws = GlobalWorkspace(seed=seed)   # composes ONE workspace (one thought-stream)
        self.last_trace = []

    def _reset(self):
        """Begin a fresh thought-stream (a fresh single-slot workspace)."""
        self.ws = GlobalWorkspace(seed=self.seed)

    # -- the re-entrant train of thought --------------------------------------------------------
    def think(self, seed, associate, max_steps=8, seed_salience=_IGNITE_SALIENCE):
        """Serial re-entrant chain of ignited thoughts held INTERNALLY (no externalization)."""
        self._reset()
        self.last_trace = []
        chain = []                                   # distinct ignited thoughts, in order
        candidates = [(seed, float(seed_salience))]  # (thought, salience) posted next tick
        for _ in range(max_steps):
            for thought, sal in candidates:
                self.ws.submit(_SOURCE, thought, sal)
            broadcast = self.ws.step()               # one spiking ignition cycle

            if broadcast is None:                    # nothing sustained firing -> subliminal
                self.last_trace.append((None, "die"))
                break
            if chain and broadcast == chain[-1]:     # incumbent still holds: sustained firing
                self.last_trace.append((broadcast, "hold"))
                continue                             # keep re-posting; the attractor will decay
            if broadcast in chain:                   # returned to an earlier link -> settled/loop
                self.last_trace.append((broadcast, "loop"))
                break
            chain.append(broadcast)                  # a NEW thought ignited: extend the chain
            self.last_trace.append((broadcast, "ignite"))
            candidates = associate(broadcast, list(chain))  # re-entry: broadcast seeds the next
            if not candidates:                       # dead end: no further association
                self.last_trace.append((broadcast, "end"))
                break
        return chain

    # -- the honest feedforward contrast --------------------------------------------------------
    def think_feedforward(self, seed, associate, seed_salience=_IGNITE_SALIENCE):
        """FEEDFORWARD CONTROL: a single forward pass with NO re-entry -- ignites at most ONE
        thought and structurally cannot chain (it never feeds a broadcast back as the next seed).
        This models the J-space limitation: to continue it would have to write to the page."""
        self._reset()
        self.ws.submit(_SOURCE, seed, float(seed_salience))
        broadcast = self.ws.step()                   # exactly one ignition; then stop (no loop)
        return [] if broadcast is None else [broadcast]

    # -- top-down summon / hold / dismiss (persistence) -----------------------------------------
    def summon(self, thought, salience=_IGNITE_SALIENCE):
        """Top-down: post a thought and ignite it into the workspace slot (a fresh stream)."""
        self._reset()
        self.ws.submit(_SOURCE, thought, float(salience))
        return self.ws.step()

    def hold(self, ticks=2):
        """Hold the current thought across `ticks` steps with NO fresh input; the cross-tick
        attractor SUSTAINS it (persistent recurrent firing). Returns the per-tick broadcasts."""
        return [self.ws.step() for _ in range(ticks)]

    def dismiss(self, max_wait=16):
        """Top-down release: withdraw the drive and step until the attractor decays below
        threshold and the thought LEAVES the workspace. Returns the number of ticks to release.

        Honest note: the workspace has no instant clear; dismissal is the natural GNW release
        (stop refreshing -> the sustained-firing attractor fades), so it takes a few ticks."""
        ticks = 0
        while self.ws.current is not None and ticks < max_wait:
            self.ws.step()                           # no submission -> unrefreshed decay
            ticks += 1
        return ticks

    # -- flexible recombination of accessible contents ------------------------------------------
    def recombine(self, a, b, salience=0.95):
        """Bind two accessible contents into ONE novel composite thought and ignite it, showing
        the composite is itself a single accessible workspace content. The binding operation is
        symbolic (a string join); the workspace supplies the ignition that makes it accessible."""
        composite = "(%s + %s)" % (a, b)
        self._reset()
        self.ws.submit(_SOURCE, composite, float(salience))
        self.ws.step()                               # ignite the composite
        return composite


# ==============================================================================================
# DETERMINISTIC OFFLINE SELF-CHECK (no network, no torch, no numpy; ASCII-only output).
# ==============================================================================================
def _fact_graph():
    """A small fact-association graph: concept -> list of (related_concept, salience).

    Linear astronomy chain with ONE branch (earth_rotates has a strong and a weak associate) so
    the workspace's spiking WTA must SELECT the stronger link and the weak one never ignites."""
    return {
        "sun": [("rises_east", 0.9)],
        "rises_east": [("earth_rotates", 0.9)],
        "earth_rotates": [("day_night", 0.9), ("coriolis", 0.6)],   # strong vs weak: WTA selects
        "day_night": [("seasons", 0.9)],
        "seasons": [("axial_tilt", 0.9)],
        "axial_tilt": [("climate", 0.9)],
        # "climate" and "coriolis" are dead ends (no outgoing association)
    }


def _make_associate(graph):
    """Return an `associate(broadcast, chain) -> [(thought, salience), ...]` over the graph."""
    def associate(broadcast, chain):
        return list(graph.get(broadcast, []))
    return associate


def _print_transcript(trace):
    """ASCII transcript of one train of thought: ignitions marked '*', holds marked '-'."""
    print("train-of-thought transcript (seed -> t1 -> t2 -> ...; '*'=ignite, '-'=hold):")
    step = 0
    for broadcast, event in trace:
        step += 1
        if event == "ignite":
            print("   tick %2d  * IGNITE  %s" % (step, broadcast))
        elif event == "hold":
            print("   tick %2d  -   hold  %s  (sustained recurrent firing / held in mind)" % (step, broadcast))
        elif event == "die":
            print("   tick %2d  .   (nothing ignited -> subliminal, stream ends)" % step)
        elif event == "loop":
            print("   tick %2d  o   loop  %s  (returned to an earlier thought -> resolves)" % (step, broadcast))
        elif event == "end":
            print("   tick %2d  |   (no further association -> train ends)" % step)


def main():
    import sys

    tot = TrainOfThought(seed=0)
    graph = _fact_graph()
    associate = _make_associate(graph)

    # ---- (1) RE-ENTRANT train of thought: >=3 distinct ignited thoughts from one seed ----------
    chain = tot.think("sun", associate, max_steps=20)
    trace = tot.last_trace

    # ---- (2) FEEDFORWARD control on the SAME seed + associate: at most ONE thought --------------
    ff = tot.think_feedforward("sun", associate)

    # ---- (3) HOLD across a delay, then DISMISS (top-down summon / hold / dismiss) ---------------
    summoned = tot.summon("holdme")
    held = tot.hold(ticks=3)                        # no fresh input -> attractor sustains it
    held_ok = summoned == "holdme" and all(h == "holdme" for h in held)
    release_ticks = tot.dismiss()                  # withdraw drive -> decays out of the workspace
    dismissed_ok = tot.ws.current is None

    # ---- (4a) WEAK seed: sub-threshold drive ignites NOTHING (no hallucinated chain) ------------
    weak = tot.think("sun", associate, max_steps=8, seed_salience=0.4)   # 0.4 < rheobase 0.55
    # ---- (4b) DEAD-END seed: ignites but has no association -> terminates at length 1 -----------
    dead = tot.think("climate", associate, max_steps=8)

    # ---- (4c) genuine LOOP: a cyclic graph settles (returns to an earlier thought, stops) -------
    cyclic = {"a": [("b", 0.9)], "b": [("c", 0.9)], "c": [("a", 0.9)]}
    loop_chain = tot.think("a", _make_associate(cyclic), max_steps=30)

    # ---- (5) RECOMBINE two accessible contents into one novel composite ------------------------
    composite = tot.recombine("day_night", "seasons")

    # ---- DETERMINISM: identical seed reproduces the identical chain -----------------------------
    tot2 = TrainOfThought(seed=0)
    chain2 = tot2.think("sun", _make_associate(_fact_graph()), max_steps=20)

    # ============================== transcript + framing =======================================
    print("=" * 84)
    print("TRAIN OF THOUGHT -- serial re-entrant ignition+hold over the spiking Global Workspace")
    print("=" * 84)
    _print_transcript(trace)
    print()
    print("re-entrant train of thought (held INTERNALLY, no externalization):")
    print("   " + " -> ".join(chain))
    print("   %d distinct thoughts, chained internally over %d workspace ticks" % (len(chain), len(trace)))
    print()
    print("FEEDFORWARD control (single pass, no re-entry -- the J-space limitation):")
    print("   %s  -> %d thought(s); cannot chain without writing to the page" % (ff, len(ff)))
    print()
    print("summon/hold/dismiss: summoned %r, held %d ticks %r, dismissed after %d ticks (released=%r)"
          % (summoned, len(held), held, release_ticks, dismissed_ok))
    print("recombine('day_night','seasons') -> %s" % composite)
    print("weak sub-threshold seed -> %r ; dead-end seed 'climate' -> %r ; cyclic graph -> %s"
          % (weak, dead, loop_chain))
    print()
    print("J-space limitations this replicates PAST (Anthropic workspace paper):")
    print("   (a) feedforward, no in-pass recurrence  -> here a re-entrant loop feeds each")
    print("       broadcast back as the next seed;")
    print("   (b) cannot hold a train of thought internally, must externalize (CoT on the page)")
    print("       -> here the chain of %d thoughts is held INTERNALLY across ticks, never written;" % len(chain))
    print("   (c) the human workspace's summon/hold/dismiss + sustained firing + chaining +")
    print("       recombination are all exercised above via genuine spiking ignition.")

    # ============================== genuine assertions (not rigged) ============================
    # (1) a real internal chain of >= 3 DISTINCT ignited thoughts from one seed.
    assert len(chain) >= 3, "think() must chain >= 3 distinct ignited thoughts, got %r" % (chain,)
    assert len(set(chain)) == len(chain), "chained thoughts must be distinct, got %r" % (chain,)
    assert chain[0] == "sun", "the chain must start at the seed, got %r" % (chain,)
    # the chain followed the association graph hop by hop (each link is a real graph edge).
    for prev, nxt in zip(chain, chain[1:]):
        edges = [c for c, _ in graph.get(prev, [])]
        assert nxt in edges, "hop %r -> %r must be a real association edge" % (prev, nxt)
    # the spiking WTA SELECTED the strong branch: the weak associate never ignited into the chain.
    assert "coriolis" not in chain, "the weak (0.6) associate must lose the WTA and NOT ignite"
    # the trace actually contains HOLD ticks -> transitions are re-entrant, not instantaneous.
    assert any(ev == "hold" for _, ev in trace), "thoughts must be HELD (sustained firing) between hops"

    # (2) the FEEDFORWARD control cannot chain: at most one thought.
    assert len(ff) <= 1, "the feedforward control must yield <= 1 thought (no re-entry), got %r" % (ff,)
    assert len(chain) >= 3 > len(ff), "re-entry must chain N>=3 while feedforward yields <=1"

    # (3) HOLD persists across >= 2 ticks; DISMISS releases the thought.
    assert held_ok, "a summoned thought must persist (be re-broadcast) across the held ticks"
    assert len(held) >= 2, "hold must span >= 2 ticks"
    assert dismissed_ok, "dismiss must release the thought (workspace current back to None)"
    assert release_ticks >= 1, "dismiss must take at least one tick of decay"

    # (4) honest termination: a weak seed ignites nothing; a dead-end seed stops at length 1;
    #     a cyclic graph settles (does not run forever).
    assert weak == [], "a sub-threshold (0.40) seed must ignite NOTHING (no hallucinated chain)"
    assert dead == ["climate"], "a dead-end seed must ignite once then terminate, got %r" % (dead,)
    assert 0 < len(loop_chain) <= 4, "a cyclic graph must settle quickly, not loop forever, got %r" % (loop_chain,)

    # (5) recombination binds both inputs into one novel composite content.
    assert "day_night" in composite and "seasons" in composite, "composite must contain both inputs"
    assert composite not in graph, "the composite must be a NOVEL thought, not a stored fact"

    # determinism.
    assert chain2 == chain, "same seed must reproduce the identical train of thought"
    # zero-dep: no numerical framework imported to run the spiking workspace.
    assert "numpy" not in sys.modules and "torch" not in sys.modules, "must be pure stdlib"

    print()
    print("self-check OK: "
          "re-entrant train of thought chained %d distinct ignited thoughts internally "
          "(%s) over %d ticks with %d hold ticks; feedforward control yielded %d (cannot chain); "
          "hold persisted %d ticks then dismiss released in %d; weak seed ignited nothing; "
          "dead-end stopped at 1; cyclic graph settled to %d; recombine -> %s; deterministic; "
          "pure-stdlib. Replicates PAST J-space feedforward/must-externalize limits (content is a "
          "symbolic associate, the workspace supplies serial ignition+hold; contested, one mechanism)."
          % (len(chain), " -> ".join(chain), len(trace),
             sum(1 for _, ev in trace if ev == "hold"), len(ff),
             len(held), release_ticks, len(loop_chain), composite))


if __name__ == "__main__":
    main()
