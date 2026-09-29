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

    set PYTHONPATH=src & python -m snn_data_lake.reasoning.train_of_thought
"""

from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

# how the workspace slot is driven when we post a thought as a candidate (supra-rheobase ~0.55)
_IGNITE_SALIENCE = 0.9
_SOURCE = "assoc"  # a single specialist source label for every posted thought


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

    def __init__(self, seed=0, workspace=None):
        """`workspace` ADOPTS the one shared GlobalWorkspace instead of building a private one.

        Hand it `Brain.workspace` / `LiveBrain.workspace` -- the single instance
        `engine/autonomous_loop.py` constructs -- and this train of thought competes for the
        SAME slot the brain's perception and salience channels post into. Passing None keeps the
        private instance this class always built, so every existing caller is byte-identical.

        WHY THE FLAG AND NOT A None-TEST AT EACH USE. `_reset()` below rebuilds the workspace to
        start a fresh thought-stream; a rebuild on a handed-in instance would silently
        RE-PRIVATISE it one call into `think()`, which is the exact defect this parameter exists
        to remove and it would look like it worked. `_owns_ws` is what makes the adoption survive
        a reset."""
        self.seed = seed
        self._owns_ws = workspace is None
        # composes ONE workspace (one thought-stream), or adopts the one it was handed
        self.ws = GlobalWorkspace(seed=seed) if workspace is None else workspace
        self.last_trace = []

    def _reset(self):
        """Begin a fresh thought-stream (a fresh single-slot workspace).

        An ADOPTED workspace is NOT rebuilt: it belongs to the brain, other faculties are posting
        into it, and replacing it here would hand this object a private bus again."""
        if self._owns_ws:
            self.ws = GlobalWorkspace(seed=self.seed)

    # -- the re-entrant train of thought --------------------------------------------------------
    def think(self, seed, associate, max_steps=8, seed_salience=_IGNITE_SALIENCE):
        """Serial re-entrant chain of ignited thoughts held INTERNALLY (no externalization)."""
        self._reset()
        self.last_trace = []
        chain = []  # distinct ignited thoughts, in order
        mine = set()  # every content THIS stream has posted -- see the "yield" branch below
        candidates = [(seed, float(seed_salience))]  # (thought, salience) posted next tick
        for _ in range(max_steps):
            for thought, sal in candidates:
                self.ws.submit(_SOURCE, thought, sal)
                mine.add(thought)
            broadcast = self.ws.step()  # one spiking ignition cycle

            if broadcast is None:  # nothing sustained firing -> subliminal
                self.last_trace.append((None, "die"))
                break
            if broadcast not in mine:
                # ANOTHER FACULTY HOLDS THE SLOT. Only reachable on an ADOPTED workspace: on a
                # private one every pool in the competition was posted by this object, so this
                # branch cannot fire and the private path is unchanged. On the shared bus the
                # incumbent is the brain's own percept, defended by its cross-tick attractor, and
                # a thought that cannot out-drive it is NOT ignited -- so it is not a link in the
                # train and must not be chained. The stream re-posts and waits, exactly as it
                # waits out its own incumbent in the `hold` branch below.
                self.last_trace.append((broadcast, "yield"))
                continue
            if chain and broadcast == chain[-1]:  # incumbent still holds: sustained firing
                self.last_trace.append((broadcast, "hold"))
                continue  # keep re-posting; the attractor will decay
            if broadcast in chain:  # returned to an earlier link -> settled/loop
                self.last_trace.append((broadcast, "loop"))
                break
            chain.append(broadcast)  # a NEW thought ignited: extend the chain
            self.last_trace.append((broadcast, "ignite"))
            candidates = associate(broadcast, list(chain))  # re-entry: broadcast seeds the next
            if not candidates:  # dead end: no further association
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
        broadcast = self.ws.step()  # exactly one ignition; then stop (no loop)
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
            self.ws.step()  # no submission -> unrefreshed decay
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
        self.ws.step()  # ignite the composite
        return composite


# ==============================================================================================
# DETERMINISTIC OFFLINE SELF-CHECK (no network; ASCII-only output).
# ==============================================================================================
def _fact_graph():
    """A small fact-association graph: concept -> list of (related_concept, salience).

    Linear astronomy chain with ONE branch (earth_rotates has a strong and a weak associate) so
    the workspace's spiking WTA must SELECT the stronger link and the weak one never ignites."""
    return {
        "sun": [("rises_east", 0.9)],
        "rises_east": [("earth_rotates", 0.9)],
        "earth_rotates": [("day_night", 0.9), ("coriolis", 0.6)],  # strong vs weak: WTA selects
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
            print(
                "   tick %2d  -   hold  %s  (sustained recurrent firing / held in mind)"
                % (step, broadcast)
            )
        elif event == "die":
            print("   tick %2d  .   (nothing ignited -> subliminal, stream ends)" % step)
        elif event == "loop":
            print(
                "   tick %2d  o   loop  %s  (returned to an earlier thought -> resolves)"
                % (step, broadcast)
            )
        elif event == "yield":
            print(
                "   tick %2d  ~  yield  %s  (another faculty holds the shared slot; re-posting)"
                % (step, broadcast)
            )
        elif event == "end":
            print("   tick %2d  |   (no further association -> train ends)" % step)


def main():

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
    held = tot.hold(ticks=3)  # no fresh input -> attractor sustains it
    held_ok = summoned == "holdme" and all(h == "holdme" for h in held)
    release_ticks = tot.dismiss()  # withdraw drive -> decays out of the workspace
    dismissed_ok = tot.ws.current is None

    # ---- (4a) WEAK seed: sub-threshold drive ignites NOTHING (no hallucinated chain) ------------
    weak = tot.think("sun", associate, max_steps=8, seed_salience=0.4)  # 0.4 < rheobase 0.55
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

    # ---- (6) P6: ADOPT a workspace, and SURVIVE the reset ---------------------------------------
    # THE TRAP, stated because an adoption that covers only __init__ passes an identity assertion
    # written immediately after construction and then silently re-privatises on the first think():
    # `_reset()` rebuilds the workspace at the top of think / think_feedforward / summon /
    # recombine. Identity is therefore re-checked AFTER each of those, not once.
    shared = GlobalWorkspace(seed=0)
    adopt = TrainOfThought(seed=0, workspace=shared)
    adopt_at_init = adopt.ws is shared
    adopt_chain = adopt.think("sun", associate, max_steps=20)
    adopt.think_feedforward("sun", associate)
    adopt.summon("holdme")
    adopt.recombine("day_night", "seasons")
    adopt_survives = adopt.ws is shared
    # ...and the private default still rebuilds per stream, which is what makes each stream fresh.
    priv = TrainOfThought(seed=0)
    priv_first = priv.ws
    priv.think("sun", associate, max_steps=20)
    priv_rebuilds = priv.ws is not priv_first

    # ---- (7) P6: another faculty HOLDING the shared slot is not chained as a thought -------------
    # The reachable RED for the `yield` branch. On a shared bus the incumbent belongs to somebody
    # else; without the branch it is appended to the chain as if this module had thought it.
    # Deliberately run on the SAME `shared` object rather than a second one: `tools/p6_gate.py`
    # counts every construction in a FACULTY module as one more private bus (it exempts only the
    # bus modules' own self-checks), so a check that demonstrates sharing by building two more
    # workspaces makes the number it is trying to improve worse.
    #
    # RE-DERIVED 2026-08-07 when the bus's shared inhibitory pool acquired its rheobase. This arm
    # used to warm the foreign content three ticks and then fall SILENT, and the yields it
    # measured were the annihilation holding this stream's thoughts under ignition while a STALE
    # incumbent decayed. Under the rheobase a fresh thought takes the head from a silent
    # incumbent on the first tick -- which is not a regression of this criterion, it is the
    # attention-capture property the subsidy split exists to buy, reached from the consumer side.
    # "Holding the slot" therefore has to mean what it means on the live brain: a faculty that is
    # STILL POSTING. The foreign faculty is modelled live through the one channel a real one
    # uses -- the subscribe/deliver cycle -- re-asserting its content on the next three
    # deliveries; the yield ticks occur exactly while it posts, and the chain resumes once it
    # stops. Both halves of the criterion are now measured against a live incumbent, where the
    # old arm conflated "holding" with "posted once, long ago".
    for _ in range(3):
        shared.submit("other_faculty", "NOT_A_THOUGHT", 1.0)
        shared.step()  # a foreign content now holds the slot with a maximal attractor
    _foreign_posts = [3]  # the foreign faculty stays LIVE for three more deliveries, then stops

    def _foreign_faculty(_admitted):
        if _foreign_posts[0] > 0:
            _foreign_posts[0] -= 1
            shared.submit("other_faculty", "NOT_A_THOUGHT", 1.0)

    shared.subscribe(_foreign_faculty)
    busy_chain = adopt.think("sun", associate, max_steps=20)
    busy_yields = sum(1 for _b, ev in adopt.last_trace if ev == "yield")
    busy_ok = busy_yields > 0 and "NOT_A_THOUGHT" not in busy_chain and adopt.ws is shared

    # ============================== transcript + framing =======================================
    print("=" * 84)
    print("TRAIN OF THOUGHT -- serial re-entrant ignition+hold over the spiking Global Workspace")
    print("=" * 84)
    _print_transcript(trace)
    print()
    print("re-entrant train of thought (held INTERNALLY, no externalization):")
    print("   " + " -> ".join(chain))
    print(
        "   %d distinct thoughts, chained internally over %d workspace ticks"
        % (len(chain), len(trace))
    )
    print()
    print("FEEDFORWARD control (single pass, no re-entry -- the J-space limitation):")
    print("   %s  -> %d thought(s); cannot chain without writing to the page" % (ff, len(ff)))
    print()
    print(
        "summon/hold/dismiss: summoned %r, held %d ticks %r, dismissed after %d ticks (released=%r)"
        % (summoned, len(held), held, release_ticks, dismissed_ok)
    )
    print("recombine('day_night','seasons') -> %s" % composite)
    print(
        "weak sub-threshold seed -> %r ; dead-end seed 'climate' -> %r ; cyclic graph -> %s"
        % (weak, dead, loop_chain)
    )
    print()
    print("J-space limitations this replicates PAST (Anthropic workspace paper):")
    print("   (a) feedforward, no in-pass recurrence  -> here a re-entrant loop feeds each")
    print("       broadcast back as the next seed;")
    print("   (b) cannot hold a train of thought internally, must externalize (CoT on the page)")
    print(
        "       -> here the chain of %d thoughts is held INTERNALLY across ticks, never written;"
        % len(chain)
    )
    print("   (c) the human workspace's summon/hold/dismiss + sustained firing + chaining +")
    print("       recombination are all exercised above via genuine spiking ignition.")
    print()
    print("P6 -- ONE shared workspace instead of a private one:")
    print(
        "   adopted by identity=%s; still the same object after think/feedforward/summon/"
        "recombine=%s; private default still rebuilds per stream=%s"
        % (adopt_at_init, adopt_survives, priv_rebuilds)
    )
    print(
        "   a foreign faculty holding the slot -> %d yield tick(s), and %r was NOT chained as a "
        "thought (chain %s)" % (busy_yields, "NOT_A_THOUGHT", " -> ".join(busy_chain))
    )

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
    # DELETED 2026-08-06, NOT WEAKENED (P8 falsifier F11), and this is the only edit this pass makes
    # outside `thalamus/global_workspace.py`. The line here was:
    #
    #     assert any(ev == "hold" for _, ev in trace), \
    #         "thoughts must be HELD (sustained firing) between hops"
    #
    # A `hold` tick is one where the workspace kept broadcasting the PREVIOUS thought while this
    # stream was already posting its successor. That was never sustained firing -- it was
    # `thalamus/global_workspace.py` paying `self_boost + hysteresis` to a content THIS STREAM HAD
    # ALREADY RETRACTED, against this stream's own fresh claim. MEASURED on that arbiter, one source
    # posting 'A' at 0.95 then 'B' at 0.95: the two pools carry `g` A=0.8075 / B=0.9500 and BOTH
    # saturate at late=30, and the head was 'A' -- the retracted content, holding the workspace with
    # LESS attractor charge than the live one. `GlobalWorkspace._defended` closed that (a tenancy is
    # held by a SOURCE, so a source posting a different content has withdrawn the old one) and the
    # same probe now reads head='B' with the `g` values UNCHANGED.
    #
    # It is deleted rather than re-derived because the property its message NAMES is already asserted
    # eleven lines below, on the input where "sustained firing" is what is actually happening:
    # `held_ok` summons a thought, stops posting entirely, and requires it to survive. That is
    # UNAFFECTED -- `hold(ticks=3)` returns ['holdme', 'holdme', 'holdme'] under both arbiters,
    # byte-identical. Restating it here would be a second copy of one criterion, not a second
    # criterion. And the chain got FASTER on the module's own headline claim, not shorter: on this
    # module's `max_steps=20` call the same seed and graph reach the same 7 distinct ignited
    # thoughts in a trace of 8 ticks that the subsidised arbiter took 13 ticks to reach, spending 5
    # of them holding a content this stream had already retracted; on an 8-tick budget instead it
    # gets 5 thoughts against 7. The ticks it spent lagging its own source are now hops.
    # (CORRECTED 2026-08-06 by an adversarial re-run -- this read "7 in 8 ticks where the subsidised
    # arbiter managed 6", which is neither the same-budget nor the same-chain comparison.)

    # (2) the FEEDFORWARD control cannot chain: at most one thought.
    assert len(ff) <= 1, "the feedforward control must yield <= 1 thought (no re-entry), got %r" % (
        ff,
    )
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
    assert 0 < len(loop_chain) <= 4, (
        "a cyclic graph must settle quickly, not loop forever, got %r" % (loop_chain,)
    )

    # (5) recombination binds both inputs into one novel composite content.
    assert "day_night" in composite and "seasons" in composite, "composite must contain both inputs"
    assert composite not in graph, "the composite must be a NOVEL thought, not a stored fact"

    # determinism.
    assert chain2 == chain, "same seed must reproduce the identical train of thought"

    # (6) adoption of the ONE shared workspace, and the reset that must not undo it.
    assert adopt_at_init, "a handed-in workspace must be adopted by IDENTITY, not copied"
    assert adopt_survives, (
        "_reset() RE-PRIVATISED an adopted workspace -- think/think_feedforward/summon/recombine "
        "each call it, so an adoption that only covers __init__ dies on the first call"
    )
    assert adopt_chain == chain, (
        "adopting an EMPTY workspace must reproduce the private chain exactly -- adoption changes "
        "who else can see the slot, not what this module thinks"
    )
    assert priv_rebuilds, "the private default must still start each stream on a fresh workspace"

    # (7) a foreign incumbent on the SHARED slot is yielded to, never chained as a thought.
    assert busy_ok, (
        "a content another faculty is holding must produce a `yield`, not a chain link (got "
        "yields=%d chain=%r)" % (busy_yields, busy_chain)
    )

    print()
    print(
        "self-check OK: "
        "re-entrant train of thought chained %d distinct ignited thoughts internally "
        "(%s) over %d ticks with %d hold ticks; feedforward control yielded %d (cannot chain); "
        "hold persisted %d ticks then dismiss released in %d; weak seed ignited nothing; "
        "dead-end stopped at 1; cyclic graph settled to %d; recombine -> %s; deterministic. "
        "Replicates PAST J-space feedforward/must-externalize limits (content is a "
        "symbolic associate, the workspace supplies serial ignition+hold; contested, one mechanism)."
        % (
            len(chain),
            " -> ".join(chain),
            len(trace),
            sum(1 for _, ev in trace if ev == "hold"),
            len(ff),
            len(held),
            release_ticks,
            len(loop_chain),
            composite,
        )
    )


if __name__ == "__main__":
    main()
