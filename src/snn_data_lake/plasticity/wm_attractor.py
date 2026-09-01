"""
Synaptic working memory -- items held as FACILITATION on Tsodyks-Markram synapses, with
selective populations and a competitive readout (specs/022 T055; the mechanism family of
Mongillo, Barak & Tsodyks, "Synaptic Theory of Working Memory", Science 319:1543, 2008).

WHAT IS ACTUALLY BUILT, and what is NOT -- said first because this module's own review
panel measured the first draft's framing as a caricature dressed as the theory. Mongillo's
model is a RECURRENT spiking network in which a facilitated population emits a population
spike under nonspecific drive. Here there is NO recurrent connectivity: `n_pop` populations
of unconnected LIF cells sit over one lumped T-M synapse each, the competition is
ARITHMETIC (a scalar `inhibition` gain, not an inhibitory pool), and the LIF cells are a
THRESHOLD READOUT rather than a network. What this build does carry from the theory is the
part T056 needs and the part the palette wanted: the memory is a synaptic state variable
(facilitation u) that outlives the activity that wrote it, decays on tau_f, is refreshed by
reactivation, and competes for a limited readout. Calling it "the Mongillo mechanism"
without this paragraph would be the overclaim the panel named.

THE MEMORY IS WRITTEN BY SPIKES, not by the tick clock. `load()` facilitates a synapse only
on steps where its population actually FIRED (the first draft called `spike()` once per
tick regardless, so a load with zero output still wrote a full trace while the docstring
claimed otherwise -- measured and corrected: at drive 0.0 the population emits 0 spikes,
takes 0 facilitations, and the trace stays exactly 0.0).

P2 ADJUDICATION -- SETTLED BY MEASUREMENT, and the measurement went AGAINST this module's
first two claims in turn. Claim one was that `cortex/prefrontal_executive` holds working
memory as persistent SPIKING while this holds it in a silent synapse: REFUTED by this
module's own panel (the incumbent holds a goal across EIGHT consecutive zero-spike ticks;
its memory is `_stack[-1][1]`, a per-tick-decayed scalar). Claim two was that the question
was open pending a benchmark. THE BENCHMARK NOW EXISTS AND HAS RUN -- `delayed_match()`
below, the multi-item delayed-match-with-distractors that paragraph named, both stores
behind one adapter, with three controls -- a store that forgets instantly, this store with
its memory variable ablated, and a stub that reports EVERY slot -- so that a real store is
distinguishable both from one that holds nothing and from one that holds everything.

ITS VERDICT, in arm (8): THE TASK CANNOT SEPARATE THEM, and getting there cost two more wrong
verdicts, both recorded so neither is re-derived. Wrong verdict A: "this store wins" --
true only against the incumbent at its CONSTRUCTOR DEFAULT, which is a parameter result.
Wrong verdict B: "the incumbent's best form wins, so retire this one" -- that steel-manned
the RIVAL ONLY. `tau_f` here is as unbounded a constructor argument as `recurrence` is
there; symmetrised, tau_f=8000 holds exactly what recurrence=0.9999 holds. The delay axis
is a PARAMETER RACE whoever tunes last wins, and it settles nothing.

A THIRD wrong verdict came from the repair itself, and it is recorded because it is the
subtlest: the arm was rewritten to score `max(0,(hits-fa)/n_items)` and reported "every
contender scores 0.50" as the P2 finding. That number MANUFACTURED the tie. Its 0.50 level
set contains (2,0), (3,1) and (4,2) -- full distractor suppression, partial, and none --
so three different behaviours read as one result, and on the raw hit counts those same runs
SEPARATE. The composite is deleted: `delayed_match()` returns the (hits, false_alarms) pair
and the caller compares vectors.

What the corrected benchmark says, both sides swept over their own retention knob, at ONE
delay, on the CLEAN task: tuned, the two stores are IDENTICAL -- (4 hits, 0 false alarms)
each, against empty nulls and a report-everything stub at (4,4) that the retired floored
score could not tell from a null. Add distractors and both give (4,2): NEITHER SUPPRESSES A
DISTRACTOR, a shared failure rather than a shared success. The clean task is primary because
the distractor axis is CONFOUNDED for the incumbent -- `maintain()` decays only the top
stack frame, so a distractor pushed on top FREEZES the targets beneath and its hit count
RISES with distractor count (1 -> 2 -> 3). Interference that protects is not interference,
and moving the false-alarm set did not fix that; it only stopped distractors being free to
report.

WHAT IS TRUE ABOUT R, after a claim in each direction was measured wrong. It is NOT "still
consumed" (that arm measured facilitation saturation), and it is NOT flatly "inert" (that
one measured only drive=1.2, where every cell fires regardless and `eff` cannot change an
outcome). Measured both ways: at a MARGINAL drive of 0.9 the resource neuter moves a load
from 40 spikes to 24, so R is genuinely load-bearing -- while at this module's own DEFAULT
drive the same neuter leaves the whole production surface bit-identical. The defect is
therefore not a dead variable but a DEAD OPERATING POINT: every other arm here loads at 1.2,
so nothing this module does ever exercises its own second variable. Disclosed under P8(4).

DISPOSITION: same job, same failure, no measured advantage on the task that was named to
settle it. Exactly one mechanism survives (P2) and this is not the one with a working claim
-- its dormancy row reads RETIRE, not "wire". It is NOT wired and NOT a second workspace,
and it outlives this commit only for P2's honesty guard: T056 is where R acquires its real
job (resources bounding the refresh cycle, which is what makes a capacity FALL OUT of time
constants instead of being declared as `max_goals`) AND where the default drive stops
hiding it -- or this file is deleted. If T056's port measures no better than the incumbent's
declared capacity, P8(5) applies to the port too.

REUSE, NOT REBUILD (P2): the synapse is `short_term_plasticity.DynamicSynapse` -- the
incumbent T-M u/R implementation, imported and driven, never re-implemented. One honest
split: DynamicSynapse decays only inside `spike(dt)`, so a SILENT delay never reaches it;
the elapsed-time bookkeeping is therefore this module's (one last-touch stamp per synapse,
every read decaying from it) while the arithmetic stays the incumbent's.

VALUES, per kind (P1): U / tau_f / tau_d are the synapse's parameters, and the defaults put
tau_f > tau_d because the mechanism REQUIRES facilitation to outlive depression -- the
constructor refuses the other regime rather than running a store that cannot remember.
n_pop, pop_size, the load drive, and the probe's (drive, gain) are CHOSEN points, not
derivations: the readout pair is chosen from a measured margin structure and the module
says where the choice sits relative to the cliff rather than claiming a two-sided
derivation it does not have.

DISCLOSED LIMITS, each measured by the panel that found them: (1) the probe's per-population
answer is 0 or pop_size -- one tick over cells sharing one input, so it is a COMPARATOR, not
a graded strength; no arm here may read it as a magnitude. (2) The competition's one-winner
regime is bounded: at the shipped gain it holds for n_pop <= 5 and degrades to no-winner
above that, which is a property of a fixed readout gain against a shrinking margin, not of
the memory. (3) There is no stochasticity anywhere in this module, so `seed` is accepted
for interface symmetry with its siblings and is NOT a source of variation -- determinism
here is by absence, not by seeding, and the constructor says so rather than advertising a
seed that carries nothing.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.wm_attractor  # (run by run_selfchecks.py)
"""

import math

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.plasticity.short_term_plasticity import DynamicSynapse

# distinct-from: cortex/prefrontal_executive.py -- NOT distinct on the job, and no longer
#   claimed to be. It is a goal-stack executive whose memory is a per-tick-decayed scalar;
#   it survives silent ticks too; and on the corrected delayed-match benchmark in arm (8),
#   on the CLEAN task with BOTH sides tuned it produces the IDENTICAL (hits, false_alarms)
#   vector this store produces -- (4, 0) each -- and with distractors both give (4, 2),
#   neither suppressing. The distinction this comment previously claimed (decoupled
#   retention/release from a two-variable synapse) is not demonstrated in this build: R acts
#   only at marginal drives, and this module always loads at a drive that saturates it away
#   (arm 7). Retirement of THIS file is dated T056; see the docstring's P2 ADJUDICATION.
#   Site read.
# distinct-from: cortex/theta_gamma_code.py -- ORDERED WM by oscillatory phase (which item
#   is first); this holds items in synaptic state with no phase code. Site read.
# distinct-from: substrate/lake_working_memory.py -- the DURABLE lake as short-term store
#   (records on disk); this is a synaptic trace in RAM on the fast clock. Site read.
# distinct-from: plasticity/short_term_plasticity.py -- that module OWNS the synapse and its
#   novelty_filter application; this composes those synapses into competing populations.
#   Imported, never re-implemented. Site read.


class SynapticWorkingMemory:
    """n_pop selective populations of LIF cells, one lumped Tsodyks-Markram synapse each,
    under a competitive readout. load(k) drives population k and facilitates its synapse ON
    THE STEPS IT FIRES; delay(steps) advances with no drive; probe() reactivates
    nonspecifically and reports which populations answer. The memory is `u`."""

    def __init__(self, n_pop=4, pop_size=8, U=0.2, tau_f=1500.0, tau_d=200.0, seed=0):
        if tau_f <= tau_d:
            raise ValueError(
                f"synaptic WM requires tau_f > tau_d (facilitation must outlive depression, "
                f"or nothing holds the memory); got tau_f={tau_f} tau_d={tau_d}"
            )
        if not (isinstance(n_pop, int) and n_pop >= 2):
            raise ValueError(f"n_pop must be an int >= 2, got {n_pop!r}")
        self.n_pop = n_pop
        self.pop_size = pop_size
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.seed = seed  # interface symmetry only: this module has no stochasticity
        self.syn = [DynamicSynapse(U, tau_f, tau_d) for _ in range(n_pop)]
        self.cells = [
            [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(pop_size)] for _ in range(n_pop)
        ]
        self._last_touch = [0.0] * n_pop  # elapsed-time bookkeeping (see module docstring)
        self._occupancy = []  # ticks each load actually spent -- what capacity_terms reads
        self.inhibition = 1.0  # competition gain (neuterable)
        self._resources_live = True  # R's only route to an output; see neuter("resources")
        self._t = 0.0

    # distinct-from: cerebellum/microcircuit.py::neuter (tissue elements -- codon/mli/pf) and
    #   plasticity/rule_composer.py::neutered (a rule dropped from a composition). Same IDIOM,
    #   different substrate: a neuter has to know its own internals, which is why this tree
    #   keeps one per module rather than sharing one. cortex/percept_surprise.py::record_charge
    #   and substrate/port_charge_probe.record_attribution RECORD an attribution for a port
    #   edge -- they do not remove anything and cannot express "cut R's route to the drive".
    #   Verdict vocabulary below is deliberately rule_composer's. All sites read.
    def neuter(self, element):
        """Neuter ONE named element IN PLACE -- this module's per-element attribution arm.

          "resources" -- cuts R's ONLY route to any output of this module: the `eff` term
                         letting a synapse's remaining resources amplify the next step's
                         drive inside load(). The synapse still updates; only its influence
                         on an output is removed. Arm (7) drives the whole production
                         surface through this and asserts NOTHING MOVES -- which makes
                         "R is INERT here" (rule_composer's sense: present, computed, never
                         moved anything, P3's shape) a measurement rather than an opinion.

        Returns self so an arm can chain. Refuses an unknown element rather than silently
        neutering nothing -- a no-op neuter reads as a passing control."""
        if element != "resources":
            raise ValueError(
                f"unknown element {element!r}: this store's only neuterable element is "
                f"'resources' (the competition is neutered by setting .inhibition = 0.0)"
            )
        self._resources_live = False
        return self

    # ---- state reads ------------------------------------------------------------------------
    def _decayed(self, k, at):
        """(u, R) for population k decayed to time `at` -- a pure read, no state written."""
        s = self.syn[k]
        gap = at - self._last_touch[k]
        if gap <= 0:
            return s.u, s.R
        u = s.u * math.exp(-gap / self.tau_f)
        r = 1.0 - (1.0 - s.R) * math.exp(-gap / self.tau_d)
        return u, r

    def trace(self):
        """Each population's facilitation u decayed to now -- the memory itself, read
        without disturbing it (a read that refreshed the trace would be the instrument
        writing what it reports)."""
        at = self._t
        return [self._decayed(k, at)[0] for k in range(self.n_pop)]

    # ---- the three operations ---------------------------------------------------------------
    def load(self, k, drive=1.2, steps=6, dt=1.0):
        """Drive population k for `steps` ticks. Its synapse facilitates ONLY on the steps
        the population actually fires, so a load that emits nothing loads nothing (measured:
        at drive 0.0 -> 0 spikes, 0 facilitations, trace stays 0.0). Returns the spike
        count."""
        if not (0 <= int(k) < self.n_pop):
            raise ValueError(f"population {k!r} out of range (n_pop={self.n_pop})")
        k = int(k)
        spikes, facilitations, eff = 0, 0, 0.0
        for _ in range(int(steps)):
            self._t += dt
            gain = (1.0 + eff) if self._resources_live else 1.0  # neuter("resources") cuts this
            fired = sum(cell.step(drive * gain) for cell in self.cells[k])
            spikes += fired
            if fired:  # PRESYNAPTIC activity is what facilitates a T-M synapse
                gap = self._t - self._last_touch[k]
                eff = self.syn[k].spike(max(gap, dt))
                self._last_touch[k] = self._t
                facilitations += 1
        if facilitations:
            # the ticks the load ACTUALLY spent on the synapse, not the `steps` it asked for.
            # Appending float(steps) here was the second form of this module's oldest defect:
            # the first draft returned a literal 6.0, the fix tracked the ARGUMENT, and the
            # docstring meanwhile claimed a measurement. A sub-threshold load that fires on
            # one tick of six recorded 6.0 -- a 6x overstatement handed to T056's divisor.
            self._occupancy.append(float(facilitations))
        return spikes

    def delay(self, steps, dt=1.0):
        """Advance `steps` ticks with no external drive. Returns the total spike count.

        STRUCTURAL, not measured, and said so rather than asserted as a property: these
        populations have no recurrent connectivity, so an undriven LIF cannot fire and this
        count is zero by construction. The falsifiable property is not the silence -- it is
        that the TRACE survives the silence and the readout still finds it, which is what
        the self-check's arms measure."""
        spikes = 0
        for _ in range(int(steps)):
            self._t += dt
            for pop in self.cells:
                for cell in pop:
                    spikes += cell.step(0.0)
        return spikes

    def probe(self, drive=0.7, gain=12.0, dt=1.0):
        """Nonspecific reactivation: every population receives the SAME weak drive, so which
        ones answer is decided by their facilitation alone. IDEMPOTENT by construction --
        the readout cells are reset before reading, because the first draft carried membrane
        state between calls and a second consecutive probe answered [8,8,8,8], every
        population, held or not (0.7 alone is period-2 on this LIF). A readout whose answer
        depends on how many times it has been read is not a readout.

        THE READOUT RANKS ON u -- THE MEMORY -- NOT ON THE EFFICACY u*R, and that is this
        module's second correction of the same defect rather than a preference. Measured:
        a freshly refreshed population has the STRONGEST memory (u 0.81) and the FEWEST
        resources (R spent by the very spikes that wrote it), so an efficacy-ranked readout
        returned the OLDEST item instead -- wrong on 3 of 4 refresh choices at a 40-tick
        delay, and `trace()` ranked that winner LAST. The first repair only REFUSED to read
        inside that transient, which left the store unreadable exactly when it had just
        been written and blocked its own P2 benchmark. Ranking on u dissolves the transient
        instead of guarding it: the refreshed item wins at 40 ticks AND at 1200, 8 of 8
        (arm 4 runs the retired u*R scoring on the identical state as its reachable red).
        This DOES orphan R, and the first draft of that claim went the other way and was
        wrong: it said R stayed consumed in `load()`. Arm (7) now measures the truth --
        cutting R's only route to an output changes nothing at all. See the module
        docstring's R IS INERT paragraph; do not restore the reassuring version.

        Returns per-population spike counts, each 0 or pop_size (a comparator, see the
        module's disclosed limits)."""
        self._t += dt
        at = self._t
        effs = [self._decayed(k, at)[0] for k in range(self.n_pop)]
        out = []
        for k in range(self.n_pop):
            rivals = [effs[j] for j in range(self.n_pop) if j != k]
            mx = max(rivals) if rivals else 0.0
            if effs[k] <= 0.0:
                comp = 0.0
            elif mx <= 0.0 or self.inhibition == 0.0:
                comp = effs[k]
            else:
                # RELATIVE margin against the strongest rival. The absolute differences
                # between co-held items are tiny (0.025 at four items) while their ratio is
                # stable, so a difference-scored competition suppressed everything --
                # measured, and the reason the mean-scored first form held nothing.
                comp = effs[k] * (effs[k] / mx - 1.0) * self.inhibition
            fired = 0
            for cell in self.cells[k]:
                cell.reset_state()  # idempotence: the readout carries no state between probes
                fired += cell.step(drive + gain * max(0.0, comp))
            out.append(fired)
        return out

    def capacity_terms(self):
        """The quantities T056's K = T_max/t_s derivation reads from this build. `tau_f` is
        the trace's own lifetime constant; `load_occupancy_ticks` is MEASURED -- the mean
        ticks the loads actually spent (the first draft returned a hardcoded 6.0 that did
        not track `load(steps=...)` at all, so T056 would have divided by a literal while
        being told it read a measurement). Reported, never asserted: T056 does the
        derivation and owes its own guards."""
        occ = (sum(self._occupancy) / len(self._occupancy)) if self._occupancy else None
        return {
            "tau_f": self.tau_f,
            "load_occupancy_ticks": occ,  # None until something has actually been loaded
            "loads_measured": len(self._occupancy),
            "n_pop": self.n_pop,
        }


def delayed_match(write, wait, held, n_items, delay, n_distractors, n_slots):
    """THE P2 SETTLING BENCHMARK: multi-item delayed-match-to-sample with distractors.

    Writes `n_items` targets, waits, presents `n_distractors` NEW items mid-delay, waits
    again, then reads the store. RETURNS THE PAIR (hits, false_alarms) -- deliberately not a
    composite. This function shipped once returning max(0,(hits-fa)/n_items), and that single
    number MANUFACTURED the equivalence it was used to prove: its 0.50 level set contains
    (2,0), (3,1) and (4,2) alike, so three different behaviours -- full distractor
    suppression, partial, and none -- all read as the same result, while the raw hit counts
    separated. A scalar that conflates recall with discrimination cannot support "these two
    stores are the same"; the caller compares vectors.

    Store-agnostic on purpose -- it takes three callables, so the same task scores this
    module and `cortex/prefrontal_executive` without either knowing about the other. That
    is what makes it a P2 instrument rather than a self-report.

    A DISTRACTOR IS A NON-TARGET AND REPORTING ONE IS AN ERROR. The first version of this
    function scored false alarms over `range(n_items + n_distractors, n_slots)` -- slots
    that NEITHER store is ever asked to write -- so the FA term was a structural zero for
    both arms, reporting a distractor was free, and "with distractors" was decorative on a
    metric that was really a pure hit count. Worse, it ran BACKWARDS on the incumbent:
    `maintain()` decays only the top stack frame, so distractors pushed on top FREEZE the
    targets beneath them, and its headline score rose with distractor count (0.25 / 0.50 /
    0.75 at 0 / 1 / 2 distractors) -- interference that protected. The FA set now starts at
    `n_items`, so every non-target slot counts, the term has a reachable red for both
    stores, and both promptly hit it."""
    targets = set(range(n_items))
    for t in sorted(targets):
        write(t)
        wait(5)
    wait(delay // 2)
    for d in range(n_items, n_items + n_distractors):
        write(d)
        wait(5)
    wait(delay // 2)
    got = held()
    return len(targets & got), len(set(range(n_items, n_slots)) & got)


def main():
    line = "=" * 92
    print(line)
    print("plasticity/wm_attractor -- T055: items held as facilitation on a T-M synapse")
    print(line)

    # (1) THE MEMORY IS SPIKE-WRITTEN: a load that fires writes a trace; one that does not
    # fire writes nothing. This is the arm the first draft's parenthetical claimed and did
    # not have -- its facilitation ran off the tick clock.
    wm = SynapticWorkingMemory()
    loaded = wm.load(1)
    tr = wm.trace()
    assert loaded > 0 and tr[1] > 0.0, f"a firing load must write a trace ({loaded}, {tr[1]})"
    silent = SynapticWorkingMemory()
    silent_spikes = silent.load(1, drive=0.0)
    assert silent_spikes == 0 and silent.trace()[1] == 0.0, (
        f"a load that emits NOTHING must load nothing (spikes {silent_spikes}, trace "
        f"{silent.trace()[1]}) -- the memory is written by spikes, not by ticks"
    )
    # SELECTIVITY, with its constructed red: only the loaded population facilitates, and
    # loading a SECOND population must move that second trace and no other (the first
    # draft compared 0.0 against 0.0, which no code path could falsify).
    wm.load(2)
    tr2 = wm.trace()
    assert tr2[2] > 0.0 and tr2[1] > 0.0, "both loaded populations must hold traces"
    assert tr2[0] == 0.0 and tr2[3] == 0.0, f"unloaded populations must stay at zero ({tr2})"
    assert abs(tr2[1] - tr[1]) < 0.01, (
        f"loading population 2 must not rewrite population 1's trace ({tr[1]} -> {tr2[1]})"
    )

    # (2) THE TRACE SURVIVES SILENCE AND THE READOUT FINDS IT. (The delay's zero spike count
    # is structural -- unconnected LIFs cannot fire undriven -- so the falsifiable claim is
    # the one asserted here: the trace persists and the probe still discriminates.)
    def _hold_and_probe(ablate, delay_steps=40):
        w = SynapticWorkingMemory()
        w.load(1)
        if ablate:  # ABLATE THE MEMORY VARIABLE on the population that holds it
            w.syn[1].u = 0.0
        w.delay(delay_steps)
        return w.trace()[1], w.probe()

    held_trace, answers = _hold_and_probe(ablate=False)
    assert held_trace > 0.5, f"the trace must survive the silent delay ({held_trace})"
    assert answers[1] > 0 and all(answers[k] == 0 for k in (0, 2, 3)), (
        f"only the held item may answer a nonspecific probe ({answers})"
    )
    abl_trace, abl_answers = _hold_and_probe(ablate=True)
    assert abl_trace == 0.0 and max(abl_answers) == 0, (
        f"erasing the facilitation must lose the item -- the memory IS u ({abl_trace}, "
        f"{abl_answers})"
    )

    # (3) IDEMPOTENCE: a readout must answer the same on consecutive reads. The first draft
    # answered [8,8,8,8] on the second probe (every population, held or not).
    w_idem = SynapticWorkingMemory()
    w_idem.load(1)
    w_idem.delay(40)
    first, second, third = w_idem.probe(), w_idem.probe(), w_idem.probe()
    assert first == second == third, (
        f"probe must be idempotent ({first} then {second} then {third})"
    )

    # (4) COMPETITION RANKS ON THE MEMORY, AT EVERY DELAY -- and the scoring this replaced
    # is kept HERE as its reachable red rather than deleted quietly. The first build ranked
    # on the efficacy u*R, which made a just-refreshed item (strongest memory, spent
    # resources) LOSE to the oldest one; the repair after that only REFUSED to read inside
    # the transient, which left the store unreadable exactly when freshly written and
    # blocked the P2 benchmark in arm (8). Ranking on u dissolves it: the refreshed item
    # wins INSIDE the old transient (40 ticks) and past it (1200) alike, while the retired
    # u*R ordering -- recomputed from the identical state, in this same command -- picks
    # the wrong item at 40. That contrast is the control that is not the method itself.
    def _multi(inhib, post=1200, refresh=None, n_items=4):
        w = SynapticWorkingMemory()
        for k in range(n_items):
            w.load(k)
            w.delay(5)
        if refresh is None:
            w.delay(post)
        else:
            w.delay(post // 2)
            w.load(refresh, steps=2)
            w.delay(post // 2)
        w.inhibition = inhib
        return w, w.probe()

    reds = []
    for post in (40, 1200):
        w_m, ans = _multi(1.0, post=post, refresh=3)
        assert sum(1 for a in ans if a > 0) == 1 and ans[3] > 0, (
            f"the REFRESHED item must win the competition at delay {post}, inside the old "
            f"recovery transient as well as past it ({ans})"
        )
        st = [w_m._decayed(k, w_m._t) for k in range(w_m.n_pop)]
        by_u = max(range(w_m.n_pop), key=lambda k: st[k][0])
        by_eff = max(range(w_m.n_pop), key=lambda k: st[k][0] * st[k][1])
        assert by_u == 3, f"the memory u must rank the refreshed item first at {post} ({st})"
        reds.append((post, by_eff))
    assert any(k != 3 for _, k in reds), (
        f"the RETIRED u*R scoring must still get it wrong somewhere, or there was no defect "
        f"to fix and this arm is decorative ({reds})"
    )
    _, maint_on = _multi(1.0, refresh=3)
    _, maint_off = _multi(0.0, refresh=3)
    n_on = sum(1 for a in maint_on if a > 0)
    n_off = sum(1 for a in maint_off if a > 0)
    assert n_on == 1 and maint_on[3] > 0, (
        f"the REFRESHED item must be the one that survives competition ({maint_on})"
    )
    assert n_off > n_on, (
        f"neutering the competition must stop it discriminating (on {maint_on}, off {maint_off})"
    )
    _, flat = _multi(1.0)
    assert sum(1 for a in flat if a > 0) == 0, (
        f"four equally-aged traces must produce NO winner, not an arbitrary one ({flat})"
    )

    # (5) REFRESH OUTLIVES DECAY at a MEASURED horizon (at 600 ticks both still answer --
    # tau_f is 1500 and the decay has barely started; the dissociation is total at 8000).
    def _survives(refresh_every, total=8000):
        w = SynapticWorkingMemory()
        w.load(1)
        step = refresh_every or total
        done = 0
        while done < total:
            chunk = min(step, total - done)
            w.delay(chunk)
            done += chunk
            if refresh_every and done < total:
                w.load(1, steps=2)
        return w.probe()[1], w.trace()[1]

    refreshed, tr_ref = _survives(refresh_every=300)
    unrefreshed, tr_unref = _survives(refresh_every=None)
    assert refreshed > 0 and unrefreshed == 0, (
        f"a refreshed item must outlive an unrefreshed one across 8000 ticks (refreshed "
        f"{refreshed}/{tr_ref:.4f}, unrefreshed {unrefreshed}/{tr_unref:.4f})"
    )

    # (6) CAPACITY TERMS ARE MEASURED, not literals: the reported occupancy must track the
    # loads that actually happened (the first draft returned 6.0 whatever was loaded).
    w_cap = SynapticWorkingMemory()
    assert w_cap.capacity_terms()["load_occupancy_ticks"] is None, (
        "an unloaded store must report no measured occupancy rather than a default"
    )
    w_cap.load(0, steps=4)
    w_cap.delay(300)
    w_cap.load(1, steps=8)
    terms = w_cap.capacity_terms()
    assert terms["load_occupancy_ticks"] == 6.0 and terms["loads_measured"] == 2, (
        f"occupancy must be the measured mean of real loads ({terms})"
    )
    w_cap2 = SynapticWorkingMemory()
    w_cap2.load(0, steps=2)
    assert w_cap2.capacity_terms()["load_occupancy_ticks"] == 2.0, (
        f"a different load must report a different occupancy ({w_cap2.capacity_terms()})"
    )
    # THE RED THAT WAS MISSING, and it fired when a review lens constructed it: every arm
    # above drives at drive=1.2, where every step fires and requested ticks == firing ticks,
    # so tracking the ARGUMENT and measuring the SYNAPSE were indistinguishable. A weak load
    # separates them -- it must report the ticks it actually spent, not the six it asked for.
    w_cap3 = SynapticWorkingMemory()
    weak_spikes = w_cap3.load(0, drive=0.30, steps=6)
    weak_occ = w_cap3.capacity_terms()["load_occupancy_ticks"]
    assert 0.0 < weak_occ < 6.0, (
        f"a load that fires on only some of its steps must report the FIRING ticks, not the "
        f"requested ones ({weak_spikes} spikes over 6 requested steps -> occupancy {weak_occ})"
    )

    # (7) R IS INERT, AND THIS ARM PINS THAT RATHER THAN DENYING IT. Arm (4) took resources
    # out of the ranking, so the honest question was whether R still reaches any output.
    # The arm that stood here CLAIMED it did -- a rapid re-load writing less than a spaced
    # one (0.2505 vs 0.3285). Three review lenses independently showed that separation is
    # FACILITATION SATURATION, not depression: DynamicSynapse.spike updates u without ever
    # reading R (u *= exp(-dt/tau_f); u += U*(1-u)), so after 2 ticks u is still ~0.73 and
    # three more spikes add little, while after 600 ticks u has decayed and the same three
    # add more. R's only channel to any output is `eff` amplifying the next step's drive in
    # load(), and at drive=1.2 every cell fires on every step in BOTH conditions, so that
    # channel is saturated shut. Deleting it leaves this entire self-check BIT-IDENTICAL --
    # verified below, in this command, which is the neuter the old arm should have been.
    #
    # So this build is a ONE-variable trace wearing a two-variable synapse, which is exactly
    # the P3 "computed but not consumed" shape, and it is disclosed rather than argued away
    # (P8 cl.4). T056 is where R must acquire its real job -- resources bounding the refresh
    # cycle, which is what makes a capacity fall out of time constants -- or the module is
    # deleted. THIS ARM MUST THEN FLIP: it asserts inertness today so that making R
    # load-bearing cannot pass unnoticed.
    # distinct-from: cortex/access_consciousness.py::workspace (identity of a shared bus) and
    #   substrate/doc_budget_gate.py::_selfcheck (constructs a red per reportable status).
    #   Neither snapshots a store's outputs for an A/B. Sites read.
    def _surface(neutered):
        w = SynapticWorkingMemory()
        if neutered:
            w.neuter("resources")
        out = []
        for k in (1, 2):
            out.append(w.load(k))
            out.append([round(x, 12) for x in w.trace()])
        w.delay(40)
        out.append(w.probe())
        w.load(1, steps=2)  # a rapid re-load: the case where spent resources SHOULD matter
        w.delay(600)
        out.append(w.probe())
        out.append([round(x, 12) for x in w.trace()])
        out.append(sorted(w.capacity_terms().items()))
        return repr(out)

    live_surface, neut_surface = _surface(False), _surface(True)
    assert len(live_surface) > 200, (
        f"the R neuter must be compared over REAL outputs, not an empty surface ({live_surface!r})"
    )
    assert live_surface == neut_surface, (
        f"R was expected to be saturated-inert AT THE DEFAULT DRIVE and is not: cutting its "
        f"only route to an output moved something, so this module's P3 disclosure and T056's "
        f"obligation are STALE and must be re-derived (that is the good outcome -- update "
        f"the claims).\n  live    : {live_surface[:200]}\n  neutered: {neut_surface[:200]}"
    )
    # ...AND THE OTHER HALF, without which the assert above is a claim measured only where
    # the mechanism cannot act -- the shape that has caught this module repeatedly. That
    # bit-identity holds ONLY at drive=1.2, where every cell fires on every step and `eff`
    # cannot change an outcome. Where firing is MARGINAL, R is plainly load-bearing. So the
    # flat "R is INERT" this arm once asserted was FALSE AS STATED; what is true is more
    # specific and worse -- this module's own default operating point saturates its second
    # variable away, so no other arm here ever exercises it. (Written inline rather than as
    # a helper: the tree's spike-count-at-a-drive helpers -- spiking_pid.advance/run,
    # habenula.lhb_spikes, drive_reduction.spikes, spiking_curiosity._calibrate,
    # mossy_relay, claustrum -- were all read, and none can drive THIS store through THIS
    # module's load() path, which is what a neuter A/B on `eff` requires.)
    R_DRIVE = 0.9
    r_counts = []
    for _neutered in (False, True):
        w_r = SynapticWorkingMemory()
        if _neutered:
            w_r.neuter("resources")
        r_counts.append(w_r.load(1, drive=R_DRIVE))
    r_live, r_neut = r_counts
    assert r_live != r_neut, (
        f"R must be CHARGED where firing is marginal -- if neutering resources changes "
        f"nothing even at drive {R_DRIVE}, then R reaches no output ANYWHERE and the "
        f"docstring's 'charged, but saturated at the default' is wrong ({r_live} vs "
        f"{r_neut} spikes)"
    )
    try:
        SynapticWorkingMemory().neuter("nonsense")
        raise AssertionError("an unknown neuter element must refuse, not no-op")
    except ValueError as e:
        assert "unknown element" in str(e)

    # (8) THE P2 SETTLING BENCHMARK -- the measurement the earlier draft only NAMED. Both
    # stores through delayed_match(), two controls that must score 0.00 (or the instrument
    # cannot tell a win from a null), and the incumbent in BOTH configurations, because a
    # win over a default-configured rival is a parameter result and P2 measures best forms.
    slots = 8

    def _synaptic_store(ablate=False, tau_f=1500.0):
        w = SynapticWorkingMemory(n_pop=slots, tau_f=tau_f)

        def write(k):
            w.load(k)
            if ablate:
                w.syn[k].u = 0.0  # the memory variable erased -- must score a flat null

        def held():
            w.inhibition = 0.0  # per-item recognition, not winner-take-all
            ans = w.probe()
            w.inhibition = 1.0
            return {k for k, a in enumerate(ans) if a > 0}

        return write, w.delay, held

    def _executive_store(recurrence):
        p = PrefrontalExecutive(n_options=slots, max_goals=slots, recurrence=recurrence)

        def write(k):
            ev = [0.0] * slots
            ev[k] = 1.0
            p.push_subgoal(ev)

        def wait(steps):
            for _ in range(int(steps)):
                p.maintain()

        return write, wait, (lambda: set(p.goals))

    def _score(make, n_items, delay, n_dist=2):
        return delayed_match(*make(), n_items, delay, n_dist, slots)

    # READ AS (hits, false_alarms) VECTORS, AT ONE DELAY, ON THE CLEAN TASK. Each of those
    # three is a repair of a defect this arm shipped with, and all three were found after it
    # had already passed a full green battery:
    #
    #  * A COMPOSITE SCORE MANUFACTURED A TIE. The arm scored max(0,(hits-fa)/n_items) and
    #    reported "every contender scores 0.50" as the P2 finding. That level set contains
    #    (2,0), (3,1) and (4,2) -- full distractor suppression, partial, and none -- so the
    #    "tie" spanned three different behaviours, and one shipped contender really was
    #    (3,1) while the others were (4,2). Worse, the tie was CREATED by the subtraction:
    #    on raw hits those same runs separate 4 vs 3. A number that conflates recall with
    #    discrimination cannot carry an equivalence claim, so it is deleted rather than
    #    caveated -- delayed_match returns the pair.
    #  * THE CONTENDERS WERE SCORED AT DIFFERENT DELAYS (one at 600, the rest at 20000). At
    #    a common 20000 that contender holds NOTHING; its place in the table was bought by
    #    scoring it in an easier cell.
    #  * THE DISTRACTOR AXIS IS CONFOUNDED FOR THE INCUMBENT, and moving the false-alarm set
    #    did not fix it. maintain() decays only the top stack frame, so a distractor pushed
    #    on top FREEZES the targets beneath: its hit count RISES with distractor count
    #    (1 -> 2 -> 3 at rec=0.99968). Interference that protects is not interference, so
    #    the primary comparison is the CLEAN task and the confound is asserted below rather
    #    than left for the next reviewer to find.
    DELAY = 20000
    nulls = {
        "ablated memory": _score(lambda: _synaptic_store(ablate=True), 4, DELAY, 0),
        "forgets everything": delayed_match(
            (lambda k: None), (lambda s: None), set, 4, DELAY, 0, slots
        ),
    }
    # ...and a control the previous version could not see. Under the floored score a stub
    # that reported EVERY slot scored 0.00 -- identical to one that reported nothing -- so
    # the adjacent "cannot distinguish a store from a stub" justification was false. On
    # vectors the two separate, and this arm now requires that.
    blurt = delayed_match(
        (lambda k: None), (lambda s: None), (lambda: set(range(slots))), 4, DELAY, 2, slots
    )
    assert all(v == (0, 0) for v in nulls.values()), f"both nulls must be empty ({nulls})"
    assert blurt != (0, 0) and blurt[1] > 0, (
        f"a stub that reports EVERY slot must be DISTINGUISHABLE from one that reports "
        f"nothing -- under the retired floored score both read 0.00 ({blurt} vs {nulls})"
    )

    # THE CLEAN TASK (no distractors), both sides swept over their own retention knob. This
    # is the comparison that settles P2, and it is symmetric: tau_f here is as unbounded a
    # constructor argument as recurrence is there.
    syn_clean = {
        tf: _score(lambda tf=tf: _synaptic_store(tau_f=tf), 4, DELAY, 0)
        for tf in (1500.0, 8000.0, 30000.0)
    }
    inc_clean = {
        r: _score(lambda r=r: _executive_store(r), 4, DELAY, 0) for r in (0.98, 0.9999, 0.99999)
    }
    assert syn_clean[1500.0] == (0, 0) and inc_clean[0.98] == (0, 0), (
        f"at their DEFAULTS both stores must fail this delay -- if either now holds, the "
        f"'both untuned arms are nulls' framing is stale ({syn_clean}, {inc_clean})"
    )
    assert syn_clean[8000.0] == (4, 0) and inc_clean[0.99999] == (4, 0), (
        f"THE P2 FINDING: tuned, BOTH stores must reach a perfect (4 hits, 0 false alarms) "
        f"on the clean task -- that identity, not a composite score, is what makes them "
        f"indistinguishable here. If they have separated, the docstring's P2 adjudication is "
        f"stale and must be re-derived ({syn_clean}, {inc_clean})"
    )
    assert syn_clean[8000.0] == inc_clean[0.99999], "the tuned pair must match as a VECTOR"

    # WITH DISTRACTORS, tuned, neither suppresses: both report every distractor they were
    # shown. A shared FAILURE, and the honest content of "with distractors".
    syn_d = _score(lambda: _synaptic_store(tau_f=8000.0), 4, DELAY, 2)
    inc_d = _score(lambda: _executive_store(0.99999), 4, DELAY, 2)
    assert syn_d == inc_d == (4, 2), (
        f"tuned, both stores are expected to hold every target AND every distractor "
        f"(synaptic {syn_d}, incumbent {inc_d})"
    )
    # ...and the incumbent's confound, pinned so it cannot rot back into a "result":
    inc_by_nd = [_score(lambda: _executive_store(0.99968), 4, DELAY, nd)[0] for nd in (0, 1, 2)]
    assert inc_by_nd[0] < inc_by_nd[-1], (
        f"the incumbent's hit count is expected to RISE with distractor count (its stack "
        f"freezes frames beneath a distractor), which is why the clean task is the primary "
        f"comparison -- if this stops holding, re-derive the confound ({inc_by_nd})"
    )

    # (9) REFUSALS: the regime guard is the one that matters -- tau_f <= tau_d cannot hold a
    # memory, and this refuses rather than running a store that silently forgets.
    try:
        SynapticWorkingMemory(tau_f=100.0, tau_d=200.0)
        raise AssertionError("tau_f <= tau_d must refuse")
    except ValueError as e:
        assert "tau_f > tau_d" in str(e)
    try:
        wm.load(99)
        raise AssertionError("an out-of-range population must refuse")
    except ValueError as e:
        assert "out of range" in str(e)

    print(
        f"self-check OK: the memory is SPIKE-written (firing load -> trace {tr[1]:.4f}; a "
        f"0-drive load emits {silent_spikes} spikes and writes 0.0) and selective (a second "
        f"load moves only its own trace); it SURVIVES silence ({held_trace:.4f} after 40 "
        f"quiet ticks, probe {answers}) and erasing u loses the item {abl_answers}; probe is "
        f"IDEMPOTENT ({first} == {second} == {third}); competition ranks on the MEMORY at "
        f"both delays (refreshed item wins {maint_on} at 40 and at 1200 where the retired "
        f"u*R scoring picks {reds}, its neuter leaves {n_off} answering, equally-aged "
        f"traces leave no winner {flat}); refresh outlives decay at 8000 ticks ({refreshed} "
        f"vs {unrefreshed}); capacity occupancy MEASURED ({terms['load_occupancy_ticks']} "
        f"over {terms['loads_measured']} loads, 2.0 for a 2-tick load, and a weak load "
        f"reports {weak_occ} of 6 requested); refusals by message.\n"
        f"R IS CHARGED BUT NOT AT THIS MODULE'S OWN OPERATING POINT (measured both ways, "
        f"and the earlier flat 'R is inert' was measured only where it cannot act): "
        f"neuter('resources') at drive {R_DRIVE} moves the load {r_live} -> {r_neut} spikes, "
        f"while at the default drive 1.2 every cell fires regardless and the same neuter "
        f"leaves the production surface BIT-IDENTICAL. Every other arm here loads at 1.2, "
        f"so the module never exercises the regime its own second variable acts in -- "
        f"disclosed under P8(4), dated T056.\n"
        f"P2, ON VECTORS AND ON THE CLEAN TASK: tuned, the two stores are IDENTICAL -- "
        f"synaptic tau_f=8000 {syn_clean[8000.0]} == incumbent rec=0.99999 "
        f"{inc_clean[0.99999]} (hits, false_alarms) at delay {DELAY}, both nulls empty and a "
        f"report-everything stub {blurt} distinguishable from them. With distractors both "
        f"give {syn_d}: neither suppresses. The retired composite score would have called "
        f"(2,0), (3,1) and (4,2) the same result, and the incumbent's hit count still RISES "
        f"with distractor count {inc_by_nd} because its stack freezes frames beneath one, "
        f"which is why the clean task decides. Same job, same failure, no measured "
        f"advantage: exactly one survives, its dormancy row reads RETIRE, and T056 decides "
        f"what is ported first."
    )


if __name__ == "__main__":
    main()
