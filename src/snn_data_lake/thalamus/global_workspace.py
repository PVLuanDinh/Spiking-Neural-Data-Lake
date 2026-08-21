"""
Global Workspace -- a SPIKING Global-Workspace / global-broadcast attention mechanism.
(Dependencies are governed by P7; this module imports `random` only. The line here used to say
"uses only `math` + `random`", and `math` has never been imported.)

Global Workspace Theory (Baars 1988; Dehaene & Changeux's Global Neuronal Workspace,
Neuron 70:200, 2011): many specialist modules run in parallel and post candidate "contents";
they COMPETE for a LIMITED-CAPACITY workspace; a candidate that crosses an all-or-none
IGNITION threshold takes a slot and is BROADCAST back to every module (global availability);
the content is SUSTAINED by recurrent self-amplification and DECAYS once it is no longer refreshed.
This is the "conscious access = global broadcast" bottleneck.

Here that bottleneck is implemented with GENUINE SPIKING neurons -- not scalar thresholds:

  * Each candidate content/specialist is a LEAKY INTEGRATE-AND-FIRE (LIF) POOL: a membrane
    potential V integrates input current, emits a DISCRETE SPIKE when V crosses v_th, and
    resets. No spike = no activity; the sub-threshold / supra-threshold boundary IS the
    ignition non-linearity (a pool driven below rheobase never fires and stays purely local).
  * RECURRENT SELF-EXCITATION: an active pool feeds its own spikes back as excitatory current
    (a within-tick synaptic trace) and carries a cross-tick attractor memory `g` -- this is the
    attractor that SUSTAINS an ignited content and gives a slot occupant its hysteresis.
  * A SHARED INHIBITORY POOL WITH A RHEOBASE: every excitatory spike drives one global
    inhibitory pool whose output subtracts from EVERY content pool (global lateral inhibition)
    -- but only PAST an allowance of `capacity` slots' worth of activity (`_inh_free`). At or
    below capacity the pool is silent and nothing is silenced; above it, the competition binds
    and strong pools keep firing while the excess is suppressed.
  * IGNITION = SUSTAINED FIRING: a content ignites (is eligible for a slot) only if it is
    still spiking in the SECOND HALF of the settling window -- i.e. it survived the competition
    and is in the high-firing attractor. A pool that fires a few transient spikes and is then
    inhibited to silence does NOT ignite; it stays local.
  * CAPACITY: the ignited set is admitted in competition rank order up to `capacity` slots.
    Everything past the capacity is ignited-but-not-accessed: it keeps its attractor charge and
    competes again next tick, but it is not broadcast and it earns no hold.

So: weak input -> the pool never crosses v_th (or, past the allowance, is inhibited to silence)
-> stays local, no broadcast. Strong input -> the pool ignites into sustained firing, recruits its
self-excitation, and -- if it ranks inside the capacity -- is broadcast globally, suppressing
what the workspace has no room for through the shared inhibitory pool. That all-or-none ignition is the un-riggable anchor, checked against a
FAIR CONTROL (`use_ignition=False`) that is a pure per-tick argmax with no spiking, no attractor,
no inhibition: the control FLICKERS on noise; the spiking workspace is STABLE and only a
genuinely stronger, supra-threshold signal takes a slot.

=====================================================================================================
THE SHARED INHIBITORY POOL HAS A RHEOBASE (2026-08-07b): inhibition begins PAST `capacity` slots.
=====================================================================================================
Every excitatory pool in this network is a spiking LIF with a threshold; the shared inhibitory pool
was the ONLY unit without one -- `a_inh` drove lateral inhibition linearly from its first spike --
and the measured consequence was that the derived, Cowan-corroborated `capacity` never bound on the
live path. Over 120 wake ticks the occupancy histogram was {1: 117, 2: 3}: one content held with
THREE SLOTS FREE while both producers posted every tick, because one saturated incumbent subtracts
`_W_SELF * _si_sat = 0.36` from a rival whose claim sits in [0.51, 0.75] against a v_th of 1.1 --
annihilated before `fired` is built, so `fired[:cap]` was a no-op. TWO mechanisms limited the held
count; P2 says exactly one survives, and the one that survives is the derived one.

THE CHANGE IS ONE KERNEL TERM AND TWO DERIVED PROPERTIES (see `_si_sat` / `_inh_free`): the pool
now ignores `(capacity - 1) * _si_sat` of activity -- what `capacity` saturated slots produce among
themselves -- and inhibits only the excess. At capacity 1 the allowance is EXACTLY 0.0, so the
single-slot control is the retired kernel byte-for-byte. The reachable red throughout is
`NoAllowanceArbiter` (the retired kernel, module scope). Re-measured on this module's own
harnesses, same seeds:

    churn on the 48-tick stream         9 -> 4        (argmax control: 54; retention 47/47 both)
    head switches, same stream          12 -> 21      <- the disclosed cost: rank churn among
                                                         co-admitted contents; retention says no
                                                         content is dropped by it
    flood (40 identical at 0.90)        40 ignited -> 4 admitted, unchanged
    win a SLOT (5 tenant claims)        [0.72 x5] -> [0.55 x5], flat at the band floor
    become HEAD, same sweep             [0.72, 0.72, 0.725, 0.765, 0.815]
                                        -> [0.615, 0.645, 0.725, 0.765, 0.815]  (3 of 5 in-band
                                        under both; the two floor tenants got cheaper to displace)
    consumer k_out (access_consciousness)  2 -> 4, and k_out == capacity at 1/2/3/4/6, where the
                                        old kernel read 1/1/1/2/3 -- denial is now the capacity
                                        bound, not the silencing. (Measured on the consumer's
                                        then-current two-competition harness; when that harness
                                        later converted to the live one-competition clock the
                                        equality gave way to the exact-tie draw regime -- smallest
                                        denying load 1/2/3/5/7 -- see the claim's own doc block.)

The flat SLOT row is the winnability criterion going TRIVIALLY green -- any supra-rheobase claim
joins a below-capacity workspace -- so that criterion and its `over_band` red RETIRED into the
full-workspace pair (`full_compliant` None / `full_strong` 0.905), which is strictly stronger on
the same axis. What replaced them is the P2 close-out criterion: capacity decides the admitted set
(min(capacity, load) at 1/2/3/4/6, sets nesting), the gain does not (set identity pinned across a
12x gain sweep), and the gain is not dead (sigma-total 349 -> 215 across that sweep). Inhibition no
longer decides the SET; it shapes the RATE.

TWO RECORDS BELOW THIS BANNER WERE CORRECTED BY THE RE-MEASUREMENT, disclosed rather than patched
silently: the split block's head row quoted `None` at the ceiling where LIMIT 3 quoted 0.815 for
the same arbiter -- re-run on the byte-exact retired kernel it is 0.815, LIMIT 3 was right, and the
row below is fixed. And the (e2) entrenchment red went green-for-the-wrong-reason: unconditional
tenancy alone no longer entrenches ANYTHING (the enforcement was always the annihilation), so that
red now runs on `NoAllowanceArbiter`. Every other `churn 9` / `[0.72 x5]` / `k_out 2` figure in the
dated blocks below is that block's own pre-rheobase record and is left standing as history; LIMIT
(1)'s one-step co-existence window on the live arm bus (evict at 0.66 with three slots free) was
the annihilation itself and is expected to have dissolved -- the live-bus re-measurement belongs to
the occupancy harness, not to this header.

=====================================================================================================
THE INCUMBENT'S SUBSIDY IS SPLIT IN TWO (2026-08-07): a HOLD on the slot, a GRIP on the head.
=====================================================================================================
The block below diagnoses the single-slot arbiter with the sentence "THE DEFENCE EXCEEDED THE WHOLE
BAND" and then fixes the SLOT half of it. The FOCUS half survived that repair and is fixed here:
`self_boost + hysteresis` was doing two jobs at once -- sustaining an admitted assembly and deciding
which assembly is broadcast -- and at 0.25 against a 0.20-wide band no compliant tenant anywhere in
the band could be displaced from the head by any compliant rival. `live_brain.thought` reads that
head, so the first tenant to arrive held what the brain was attending to, permanently.

The block below already says why no parameter fixes this ("no amount of tuning fixes it, because
`hi` is DEFINED as `1 - defence`"), and that was re-measured the long way before the split was
built: the admissible defence window is `(0.2000, 0.2250)`, and EVERY value in it costs churn
9 -> 15 and retention 47/47 -> 45/47, while `(1 - lo)/3 = 0.15` reddens five criteria across three
modules by taking the capacity to 6 and stopping the bottleneck binding. Churn-clean and
head-displaceable are DISJOINT on that axis. The rheobase axis has no admissible value at all.

So the parameter was never the problem. ADMISSION still ranks on the full hold (`_rank_key`),
which is why churn, retention, the capacity, the lateral gain and the whole SLOT row are
BYTE-UNCHANGED; the head is ordered over the admitted set by `_focus_key`, carrying only
`grip = hold / capacity`. Measured, same five tenants, same 0.75 ceiling:

    become HEAD at    0.805  0.835  0.910  0.955  None      <- one subsidy: outside the band, 0 of 5
    become HEAD at    0.720  0.720  0.725  0.765  0.815     <- split: inside the band, 3 of 5
    (CORRECTED 2026-08-07b: the last cell was recorded here as `None`, contradicting LIMIT 3's
    0.815 for the same arbiter. Re-measured on the byte-exact retired kernel: 0.815 -- the ceiling
    tenant is displaceable at `hi + grip` + one sweep step, which is above the band but finite.)

Its reachable red is `OneSubsidyArbiter`, the pre-split arbiter kept as a running control. The
biology it answers to: recurrent excitation is what sustains a cortical assembly, but which
assembly DOMINATES is set by afferent drive under biased competition -- an arbiter where the sitting
tenant's reverberation also decides what is attended to is one whose attention cannot be captured.

Retired in the same pass: `_G_INH = 2.4`, the "TOTAL lateral-inhibition budget", which was exactly
`_W_SELF * capacity` -- the per-slot E/I balance multiplied back up by the capacity it was about to
be divided by again. See `_lateral_gain`; the shipped workspace is byte-identical without it.

=====================================================================================================
WHY THIS IS MULTI-SLOT, AND WHAT IT REPLACED (2026-08-06). All numbers below were RUN, not deduced.
=====================================================================================================
This module was SINGLE-SLOT winner-take-all, and the slot was measurably UNWINNABLE. `step()`
computed the ignited SET (`late[k] > 0`, one flag per pool) and then threw all but one member away
with an argmax. Meanwhile the one incumbent was granted `self_boost + hysteresis = 0.25` of extra
drive, while the entire band a compliant specialist may claim -- `[ignition_threshold, 1 -
(self_boost + hysteresis)]` = [0.55, 0.75], the derivation `brainstem/autonomous_loop.py::
_percept_salience` and `cortex/access_consciousness.py::_max_claim` both apply -- is 0.20 WIDE.
THE DEFENCE EXCEEDED THE WHOLE BAND. Measured on the old arbiter, incumbent warmed 5 ticks and
still posting, challenger swept 0.55..1.00 in steps of 0.002:

    incumbent claim  0.550  0.600  0.667  0.700  0.750
    challenger needs 0.734  0.734  0.892  0.948  0.952      <- compliant ceiling is 0.75

so a compliant challenger could take the slot only from an incumbent at the very bottom of its
range, and never from one at 0.667 or above. The live producer range is 0.5797..0.6573, which
straddles that boundary: the bus was winnable on some ticks and structurally frozen on others,
and no amount of tuning fixes it, because `hi` is DEFINED as `1 - defence`, so an incumbent that
itself claimed `hi` carries `hi + defence == 1.0` -- exactly the maximum any challenger can post.

THE FIX IS NOT A NUMBER, IT IS THE COUPLING (P1's corollary: when a hardcode resists replacement
the blocking constraint is the bug). Two changes, both inside the one arbiter:

  1. `step()` KEEPS the ignited set instead of argmaxing it. This required no new mechanism --
     `late[k] > 0` was already computed per pool and discarded.
  2. The shared lateral inhibition is DIVIDED ACROSS THE CAPACITY -- written `_G_INH / capacity`
     at the time; the typed `_G_INH` was later retired as `_W_SELF * capacity` exactly (see
     `_lateral_gain`), and the 2026-08-07b banner above gave the pool its rheobase. The old
     dynamics gave the incumbent's defence a second, hidden effect -- more drive meant more
     spikes, which meant more drive into the SHARED inhibitory pool, which suppressed everyone
     else. Stability and winnability were therefore coupled through `_G_INH` and were provably
     DISJOINT: swept over defence 0.12..0.30 there was no setting that was both stable (the
     module's own `switches` criterion) and winnable inside the band. A workspace with C slots
     applies 1/C of its lateral budget to each, so holding a slot costs a SLOT, not the silence
     of every competitor.

MEASURED CONSEQUENCE, same sweep on the new arbiter:

    tenant claim     0.550  0.5797  0.6573  0.700  0.750     (all inside the compliant band)
    challenger wins  0.720   0.720   0.720  0.720  0.720     <- flat, and inside the 0.75 ceiling

The defence no longer compounds into unwinnability. A compliant specialist wins a slot against
ANY compliant tenant, which is the entire point of this change.

A NULL WORTH RECORDING (P8 guard (d)). The obvious alternative fix -- decouple the defence from
the pool's inhibitory EXPORT, so a held pool contributes to the shared inhibitory pool only in
proportion to its own claim -- was BUILT and swept alongside the above. It buys winnability at
some gains and it costs stability: at the gain this module ships (0.60) it moved the workspace's
switch count 3 -> 7 while the divided-budget form alone already reached 0.72/0.72. Two mechanisms
where one suffices is the additive failure P2 forbids, so the export decoupling is NOT here.

WHAT DID NOT CHANGE, deliberately. `ignition_threshold`, `decay`, `hysteresis` and `self_boost`
keep their exact defaults. `brainstem/autonomous_loop.py::_percept_salience` and
`cortex/access_consciousness.py::_max_claim` READ those three attributes off this object at
runtime and size their own admissible band from them, and `autonomous_loop`'s self-check ASSERTS
its live saliences land inside it. Moving any of them here would move a producer's range in
another file and could redden a self-check this module does not own.

=====================================================================================================
THE TENANCY IS SCOPED TO THE SOURCE (2026-08-06). The defect above, one level up, and it was LIVE.
=====================================================================================================
The fix above stopped a defence exceeding the whole band for equal compliant PEERS. It did not stop
the same subsidy being paid to a content THE PRODUCER ITSELF HAD ALREADY RETRACTED, and on the live
bus that is what actually held the workspace shut. `_g` is keyed by `(source, content)`, and
perception's content key ROTATES over the percept ids -- so each fresh percept opened a NEW pool at
its own salience while the entrenched pool drew `self_boost + hysteresis` and decayed only `decay`
per tick between refreshes. FIRST TENANT, FOREVER.

MEASURED on `make_brain(seed=0, refit_every=10, persist=False)`, `clock.phase = 0.25`, 120 live
ticks. `cortex/live_brain.py::thought` -- THE TREE'S ONE GENUINE CROSS-MODULE WORKSPACE READ --
carried the int 0 on 120 of 120 ticks while perception submitted FOUR distinct contents at 56
DISTINCT saliences, and workspace occupancy was 1 of 4 on 120 of 120 with three slots never used.
The proof that this was ENTRENCHMENT and not a quiet bus is in the final pool state:

    pool                    _g          late spikes
    ('perception', 3)   0.621394             0        <- the LIVE percept
    ('perception', 0)   0.544709            30        <- the tenant, holding the workspace

The fresh content held MORE attractor charge than the one occupying the workspace and still fired
nothing, because +0.25 of tenancy took the tenant to a drive the challenger could not survive the
shared inhibition against. `GlobalWorkspace._defended` carries the fix and the full derivation: a
tenancy is held by a SOURCE, so a source posting a DIFFERENT content has WITHDRAWN the old one and
stops paying for it. Same run after the change: `thought` takes 4 distinct values, changes 55 times,
and NAMES THE CONTENT SUBMITTED THIS TICK on 56 of 56 posting ticks (holding the last percept on all
64 silent ticks). The pre-fix arbiter, reconstructed and re-run as the RED: 1 distinct, 0 changes,
19 of 56.

NOTHING PUBLISHED BY THIS MODULE MOVED, re-run rather than argued: winning a SLOT is still a flat
[0.72, 0.72, 0.72, 0.72, 0.72] across the same five tenant claims; a 40-content flood still admits 4;
a silent source still holds 7 ticks then releases; churn is still 9 against the argmax control's 54.
Every one of those harnesses has its tenant RE-POSTING THE SAME CONTENT, which is the branch this
change leaves untouched -- that is why they are unchanged, and it is the reason the change is safe
rather than a coincidence. (The HEAD row read [0.72, 0.825 x4] here and moved to [0.805, 0.835,
0.910, 0.955, None] when the rank key was repaired below; it was not `_defended` that moved it.)

WHAT IT COSTS, disclosed rather than found later. The workspace no longer damps a source's own
content changes -- it damps competition BETWEEN sources, which is what `count_churn` measures and
what is unchanged. One consumer relied on the intra-source damping: `reasoning/train_of_thought.py`
asserted its `think()` trace contains HOLD ticks, and a HOLD tick was the workspace lagging behind
the very stream that was driving it. Measured, one source posting 'A' at 0.95 then 'B' at 0.95: `g`
reads A=0.8075 / B=0.9500, both pools saturate at late=30, and the old head was 'A' -- the retracted
content, on less charge. That assertion was DELETED at the site per F11 (its message named
"sustained firing", which is measured eleven lines below it by summoning a thought and then posting
NOTHING -- `hold(3)` returns the same thought under both arbiters, byte-identical). On the module's
own `max_steps=20` call the same seed and graph now chain the same 7 distinct ignited thoughts in a
trace of 8 ticks that the subsidised bus took 13 ticks to reach, spending 5 of them holding a
content this stream had already retracted; given an 8-tick budget instead, the subsidised bus gets 5
thoughts against 7. (CORRECTED 2026-08-06 by an adversarial re-run -- this read "managed 6", which
is neither the same-budget figure nor the same-chain figure.)

THE RANK TIE-BREAK: THE NAME IS OUT OF THE ARBITER (SHIPPED 2026-08-06). Every number was RUN.
`(-late[k], -total[k], k)` fell through to the source string when two pools tied on both spike
counts. The census is re-derived rather than quoted -- exhaustive two-producer grid, warmups
{0,1,2,3} x 51 x 51 salience pairs on a 0.01 grid over [0.50, 1.00], scoring the tick AFTER the
warmup: 3,576 co-admissions of which 1,941 -- 54.28% -- were settled lexicographically. `_defended`
did not move that census at all, because it changes which pools are subsidised and not how ties
among them are broken.

IT WAS NOT CONFINED TO THE HEAD, and that is the half every prior pass of this header got wrong.
`step()` admits `fired[:cap]`, and `_defended` then grants every ADMITTED occupant `self_boost +
hysteresis` on the NEXT tick -- so whatever settles the rank on tick t settles who is SUBSIDISED on
tick t+1, and a subsidised pool silences an unsubsidised one through the shared inhibition. The
string therefore decided ACCESS, not merely order. MEASURED, and the construction is spelled out
because the METRIC has a trap in it: six co-saturating producers into a capacity of 4 over 20
ticks, arm A naming them in loudness order and arm B in reverse, comparing the set of admitted
CLAIMS -- never the set of admitted NAMES, which differs between the arms whatever the arbiter does
and so reports a finding that is not there. Swept over 21 top claims x 7 spacings, keeping the 96
loads whose weakest claim stays >= 0.55 (COUNTS ARE PROPERTIES OF THE GRID, the phenomenon is the
finding -- a prior pass quoted 123/24/13 off a different grid and it is not re-derivable from this
one):

    loads where the NAME changes the admitted CLAIM set   retired   -`_g`   -`_last_v` (SHIPPED)
    all 96                                                  15        0         0
    of those fully compliant (top claim <= 0.75)            15        0         0
    control: the same six producers spread at 0.04           0        0         0

At top 0.650 / spacing 0.005 -- every claim compliant -- the LOUDEST producer held a slot on 20 of
20 ticks under one naming and 0 of 20 under the other, having changed nothing but its name. That
was the entrenchment shape `_defended` closed for retracted content, one level up, the alphabet
choosing who is subsidised. It is 0 of 96 under the shipped key.

THE PRECONDITION, stated so the finding is not overstated: it needs MORE co-saturating pools than
slots -- at least `capacity + 1` producers whose claims sit within roughly 0.005..0.02 of each
other. Compliant producers SPREAD across the band are separated by the competition itself: six of
them at 0.04 spacing leave ONE content in the workspace under both namings. And none of it reaches
the live arm bus, checked rather than assumed: occupancy there is 1 of 4 on 120 of 120 ticks, so no
tick ranks two contents at all and `thought` is decided by the ignition dynamics.

THE THREE DIRECTIONS, MEASURED, so none of them has to be re-run. The test is two-sided -- a rename
must not move the head, AND a pool that genuinely out-drives another must take it. A rule that
ignores names by ignoring everything passes the first half and fails the second, so both halves are
counted on the same 3,576-pair grid.

  (b) MORE MICRO-STEPS -- DEAD, and the premise under it is false. `late` is not three-valued
      because the window is short; it SATURATES, because a pool in the high-firing attractor fires
      on EVERY micro-step of the second half. On the SAME full grid as the census: _T=60 and _T=120
      each read exactly three values ({0, _T/4, _T/2}) and give the SAME 1,941 of 3,576 -- 54.28% --
      so DOUBLING the micro-steps moves the tie count by ZERO, not by approximately zero. On an
      11x11 subsample the value set stays {0, _T/4, _T/2} and the rate stays flat through _T=960.
      _T=30 is the lone exception and it goes the WRONG way: seven values ({0,1,2,7,8,14,15}, the
      four extras appearing 21 times in 20,808 readings) and still 1,962 of 3,604 -- 54.44%, worse
      than _T=60. The cost is linear: 0.1415 -> 1.7808 ms per 5-pool step, 12.6x (measured 2026-08-06
      and not re-derived since). Raising _T buys resolution only for a readout that is not already
      clipped. (CORRECTED 2026-08-06 by an adversarial re-run: this read "the values are always
      {0, _T/4, _T/2}", which is false at _T=30, and quoted a rate row "54.8 / 54.4 / 54.4 ..."
      whose _T=60 entry contradicts this header's own census three paragraphs up. The NULL survives
      unchanged; the numbers under it did not.)
  (c) EARLIEST IGNITION / LONGEST SUSTAINED -- DEAD for the same reason. Recording the first and the
      last spike micro-step per pool and ranking on either separates 0 of the 1,941 tied pairs and
      leaves the name-flip count at 1,941: co-saturating pools ignite on the same micro-step and
      fire to the same last one. Principled, and constant across exactly the population that needs
      breaking.
  (a) A FINER READOUT -- the only direction that works, and one term is enough. Half 2 is scored as
      "the head went to the pool with the higher `drives`", stated because an earlier scoring
      counted the 176 EXACTLY-EQUAL-claim configs as MISSES, where neither pool out-drives the other
      and the half has no content to test:
        -`_g[k]`      (attractor charge) 1,941 -> 176 name-flips (4.92%), out-drive-head
                      2,514/3,400 (73.9%) -> 3,400/3,400 (100%), ACCESS arm 15 -> 0 loads.
                      PARTIAL, and it is the LOSER: `_g` still ties on 9.1% of the tied set
                      (176 of 1,941, separating 1,765), and those 176 are exactly the equal-claim
                      configs.
        -`_last_v[k]` (final membrane) 1,941 -> 0 name-flips, separates 1,941 of 1,941,
                      out-drive-head 100%, ACCESS arm 0. THIS IS THE SHIPPED KEY.
      Ranking on the raw drive scores identically (0 name-flips / 100%) and is REJECTED: `drives` is
      the INPUT to the competition and carries `_jitter_for`, so it never ties because the
      SYMMETRY-BREAK NOISE never ties. That swaps one arbitrary tie-break for another and stops the
      settle deciding anything. `_last_v` is the network's OUTPUT, after the inhibition and the
      self-excitation have run.

WHY `-_g` LOST, weighed rather than asserted, because its case is real and it is the one direction
that leaves this module's own head table alone. Three measurements decided it:

  * `-_g` IS A STRICT SUBSET OF `-_last_v`, not an alternative to it. Under `_JITTER = 0` the two
    produce the IDENTICAL flip SET -- 184 configs each of 3,160 co-admissions, compared as sets and
    equal, not merely equal in count. So there is no input on which `-_g` separates a pair
    `-_last_v` does not, and choosing `-_g` can never buy discrimination `-_last_v` lacks. What
    `-_last_v` adds on top is exactly the jitter-amplified 176.
  * `-_last_v` IS THE MORE STABLE FOCUS, on this module's own 48-tick stream: HEAD switches
    17 (retired) -> 14 (`-_g`) -> 10 (`-_last_v`), with churn 9, retention 47/47, the 40-content
    flood admitting 4 and the silent-source hold releasing after 7 IDENTICAL across all three. Head
    hysteresis is the property this module exists to have over the flickering argmax control, and
    `-_g` gives the rank away to whichever pool currently holds the most charge regardless of
    tenancy.
  * IT DOES NOT BUY A CHEAPER MIGRATION. Run against every consumer self-check in the tree, `-_g`
    and `-_last_v` redden the SAME TWO criteria and nothing else, so the "leaves the head table
    byte-identical" credit does not translate into one fewer file to re-derive.

WHAT IT RESTS ON, disclosed rather than found later: 9.1% OF THE SEPARATION IS JITTER, NOT PHYSICS.
Two pools claiming exactly 0.70 settle to membranes 3.458698e-03 apart with the jitter live and
EXACTLY 0.0 apart under `_JITTER = 0` -- the same figure at 0.60 and 0.80, so it is the jitter draw
and not the claim. On the 176 equal-claim configs `-_last_v` is therefore the symmetry-break noise
laundered through the LIF, and its "never ties" is inherited there rather than earned. Its case
rests on the other 1,765 of 1,941, where the claims genuinely differ and the settle has something to
amplify. That is a WEAKER objection than the one that kills `-drives`, and the difference is
measurable rather than rhetorical: `-drives` puts the jitter AHEAD of the settle, so it never ties on
ANY of the 3,576 and the network decides nothing; `-_last_v` puts it after, so the settle decides
90.9% of the tied set on its own and the noise surfaces only where there is no fact of the matter --
two pools asserting the identical claim, which neither key can rank on merit because neither
out-drives the other.

WHAT THE REPAIR COSTS. The tick cost is one dict lookup inside a sort key. Determinism holds. The
live brain is BYTE-IDENTICAL on `thought` and `_access_ema`; see the live block below for the
numbers and for the honest reading of that (it is worth ZERO on today's bus, because occupancy there
is 1 of 4). Slot winnability stays a flat [0.72 x5], the capacity still binds (40 at 0.90 -> 4), the
silent-source hold still releases after 7, `_defended` is untouched. THREE things moved:

  1. `cortex/access_consciousness.py` claim (10): `focus_moved >= 1` went to 0. Its FOCUS lever WAS
     a peer rename, so its green state REQUIRED the head to be name-decided and no arbiter is both
     correct and green on it. RE-DERIVED at the claim onto the peer VOLUME lever -- the property is
     real, the proxy was pathology-shaped. Its ACCESS half was untouched (`access_same` 12 of 12
     under both keys) and claim (8) reads identically under both.
  2. `reasoning/rpm_solver.py` `collide_ok` case (ii): `ws_seq.broadcast() == stale` went False. The
     precondition -- "the previous matrix's answer holds the HEAD" -- was itself manufactured by the
     string compare: measured, the stale pool held the head on `_g` 0.850000 / membrane 0.613777
     against this solve's own candidate at 1.000000 / 0.916325, purely because source 'c0' sorts
     before 'c3'. Under this key the fresher pool correctly takes the head. The case was DELETED at
     the claim under F11 -- not re-derived -- because its precondition is UNREACHABLE (0 of the 1,856
     answer-differing two-solve configurations on a 4-regime x 12-seed grid squared); its
     LOAD-BEARING half (answer #3, known=True) is unchanged under both keys.
     (CORRECTED 2026-08-06 by an adversarial re-run. This item read "'c7'", "0.904735", "answer #7"
     and "RE-DERIVED". None of the three numbers reproduces, all three contradicted `rpm_solver.py`'s
     own block inside the same change, and the disposition was the wrong one -- which is exactly the
     unre-derivable-figure failure this header removed a bare crc32 for two paragraphs above.)
  3. THIS MODULE'S OWN head-capture table, [0.72, 0.825, 0.825, 0.825, 0.825] -> [0.805, 0.835,
     0.910, 0.955, None]. That is not a regression the repair introduces; it is the repair
     WITHDRAWING a credit this header was taking for the alphabet. The flat 0.825 was the challenger
     'chal' sorting before the tenant 't0'. See LIMIT 3 for why the new row is defended rather than
     apologised for, and for the criterion that now pins it (a RENAME must not move the sweep --
     which the retired key fails, run as the red).

=====================================================================================================
WHAT THIS BUYS ON THE LIVE BRAIN, AND WHAT IT DOES NOT. Measured, 2026-08-06, on the canonical
`LiveBrain` composition -- 60 wake ticks, `.venv-lake`.
=====================================================================================================
THE 2026-08-06 RANK REPAIR IS WORTH EXACTLY ZERO HERE, and it is measured rather than assumed. On
`make_brain(seed=0, refit_every=10, persist=False)` with `clock.phase = 0.25` over 120 ticks (56
awake), the live brain is BYTE-IDENTICAL under the retired key and the shipped one: `thought` crc32
0f0938d6, 4 distinct values, 55 changes, naming the submitted percept on 56 of 56 posting ticks;
`_access_state["ignited"]` identical; `_access_ema` 0.488958 to 6 dp under both. Re-run in BOTH
`cortex/access_consciousness.py` content-key regimes (frozen and rotating) so the null is not an
artefact of the other change landing in the same pass -- 0.488958/0.488958 frozen, 0.287282/0.287282
rotating, the rank key moving nothing in either. THE REASON IS OCCUPANCY: 1 of 4 on 56 of 56 ticks,
so no tick ranks two contents and there is nothing for a rank key to decide. What the repair buys is
that a name-decided ACCESS path is gone and the dissociation is reachable from the consumer side; it
is not that today's broadcast changed. (A prior pass's figure for `_access_ema` on this run was
quoted as 0.492479 in one place and 0.488958 in another. Re-measured: 0.488958.)

WITH THE TREE AS IT STANDS: NOTHING, and that is the honest result. The arm bus has exactly ONE
admissible producer. Spied over 60 live ticks, `perception` posts in [0.55, 0.65] every tick and
`salience`/threat posts a structural 0.000000 every tick (the amygdala is never paired on the wake
path; `autonomous_loop` documents and proves that null in both directions). K slots over one
admissible candidate broadcast one content: `broadcast_all()` returned a 1-tuple on 60 of 60 ticks
and `cortex/live_brain.py::thought` named the percept on 60 of 60 -- unchanged, which is the
requirement, not the achievement. THE CAPACITY IS NOT THE BINDING CONSTRAINT ON THAT BUS TODAY;
THE PRODUCER POPULATION IS, and fixing that is not this module's work.

CORRECTION, 2026-08-06, and it is the sharpest lesson in this header. The sentence above --
"`thought` named the percept on 60 of 60" -- is the phrasing THREE separate commits used as the
safety property they had preserved. It was true and it was nearly empty: over those 60 ticks
`thought` carried ONE distinct value, so "named the percept" meant "named the percept that ignited
on tick 0 and was never displaced". A preserved constant reads exactly like a preserved invariant,
and 60-of-60 reads like strong evidence when the quantity behind it cannot vary. The block above
diagnoses why it could not, and `_defended` fixes it; what is worth carrying forward is the
INSTRUMENT rule rather than the incident: an agreement count over a sequence is worth nothing until
the DISTINCT-VALUE count of that sequence is printed beside it, which is why every table in this
header now carries one.

WHAT IT BUYS THE MOMENT A SECOND FACULTY JOINS, which is the whole point and is measured rather
than asserted. `cortex/access_consciousness.AccessConsciousness` was put on the arm's OWN bus
object (identity-checked, not a copy) and posted at a compliant 0.65:

    arbiter              workspace         percept still in it   producer B in it
    retired single-slot  ('GLOBAL',)             0 / 40 ticks       40 / 40 ticks
    this one             ('GLOBAL', 0)          40 / 40 ticks       40 / 40 ticks

The retired arbiter DESTROYED the percept -- which is precisely why a prior pass declined to ship
that repoint, having watched the arm's broadcast go '0' -> 'GLOBAL' permanently. Under this one the
percept is still in the workspace on every tick. That is the capability the change exists to buy.

THE HARNESS WAS PROVEN ABLE TO GO RED BEFORE ANY OF THIS WAS BELIEVED, because the first run of it
returned a null on BOTH arbiters and the null was an artefact: AccessConsciousness's own computed
salience on this brain is 0.007..0.140, entirely below the 0.55 rheobase, so producer B never made
an audible claim and no arbiter could have differed. Swept across the band the harness reads 0/40
consumer-output changes at claims 0.55 and 0.60 and 40/40 at 0.65 and above -- a red and a green
from the same instrument.

FOUR LIMITS, disclosed rather than discovered later. (1) CO-EXISTENCE ON THE LIVE BUS IS ONE STEP
WIDE, re-measured 2026-08-06 by sweeping the claim in 0.01 rather than reading the band ends.
Against a perception claiming 0.5500-0.6500, producer B co-exists on 40 of 40 ticks at a claim of
0.65 and EVICTS the percept on 40 of 40 at 0.66 -- and at 0.67, 0.68, 0.69, 0.70, 0.72 and 0.75,
all of which read percept 0/40. The retired arbiter evicts at 0.65 as well, so the window this
change opens on that bus is real and is exactly one 0.01 step of the 0.20-wide band. The eviction
is the shared inhibition silencing the weaker pool with THREE SLOTS STILL FREE -- competition
working rather than the capacity binding -- but this limit previously said the loser only loses at
the band CEILING, which was a single point read as a range. The sweep is the honest form. (2)
`tools/p6_gate.py` matches an explicit `(method, arity)` table that does not contain
`broadcast_all`, so the tree's P6 READ count does not move because this accessor exists. Adding it
there is work in a file this pass does not own. (3) and (4) are the two that matter and they get
their own blocks below.

=====================================================================================================
LIMIT 3 -- THE SLOT IS WINNABLE; THE *HEAD* IS NOT. Do not read one as the other.
=====================================================================================================
**SUPERSEDED 2026-08-07, AND KEPT BECAUSE THE REASONING BELOW IS WHY THE FIX EXISTS.** Everything in
this block was correct about the arbiter it described, and the four grounds it defends the `None` row
on are still sound as far as they go. What it did not see is that it was defending a DEFECT. The head
was unwinnable because ONE subsidy was doing two jobs: `self_boost + hysteresis` sustained an
admitted assembly AND decided which assembly was broadcast. Those jobs want opposite magnitudes, so
the head row sat outside the band at every point, and the ground-1 argument ("it is not a cost the
repair introduced, it is a credit this header was TAKING for the alphabet") explains why the row
moved without asking whether where it landed was right.

The subsidies are now SPLIT -- see `grip`. Admission still ranks on the full hold, so every number
this block cites for the SLOT is unchanged; the head is ordered by `hold / capacity` and the row
below now reads `[0.72, 0.72, 0.725, 0.765, 0.815]` against the same 0.75 ceiling, i.e. inside the
band for three of the five tenants instead of none of them. Ground 3 ("THE FOCUS IS MEASURABLY MORE
STABLE FOR IT") survives at a measured cost of two: head switches on the 48-tick stream are 12 where
they were 10, with churn 9 and retention 47/47 both unchanged. Ground 4 is the one that inverted and
it is the important one: `access_consciousness` claim (9) no longer NEEDS the faculty's own claim cap
to let a 0.95 proximity threat take the head -- measured, the threat takes it even from an UNCAPPED
faculty, and the claim's reachable red is now the pre-split arbiter (`OneSubsidyArbiter`) rather than
the uncapped arm. The safety-shaped property got stronger, not weaker.

The text that follows is the pre-split record. Read it as the argument that a `None` at the ceiling
is correct -- which it still is, since `hi + hold == 1.0` identically -- and NOT as a statement that
the whole band is undisplaceable, which is what it says and which stopped being true.

`broadcast()` / `current` / `live_brain.thought` return the RANKED HEAD, and the head is a
different question from access. Swept the same way as the winnability table above:

    tenant claim      0.550   0.5797   0.6573   0.700   0.750     compliant ceiling 0.75
    win a SLOT at     0.720    0.720    0.720   0.720   0.720     <- inside the band, every tenant
    become HEAD at    0.805    0.835    0.910   0.955    None     <- ABOVE the band at every point
    1-slot control    0.805    0.835    0.910   0.955    None     <- slot and head are one thing
    (name-decided)    0.720    0.825    0.825   0.825   0.825     <- what the alphabet was paying

READ THE THIRD ROW CAREFULLY, because it is a re-derivation and not a drift. `_SingleSlotControl`
reconstructs the RETIRED CAPACITY as a configuration of THIS arbiter, so it now carries the repaired
rank key too and its numbers moved with it: run on THIS sweep the genuinely retired CODE measures
[0.735, 0.735, 0.895, 0.950, 0.955]. (The [0.734, 0.734, 0.892, 0.948, 0.952] at the top of this
header is a DIFFERENT grid -- step 0.002, incumbent claims 0.550/0.600/0.667/0.700/0.750, warmed 5 --
and is kept as the historical record of the deleted mechanism, not as this sweep's control. Quoting
it here as though it were was corrected 2026-08-06; the same-grid figure above is re-derivable by
checking out the retired file and running its own `_lowest_win`.) The row above is the one-variable
control -- capacity 1, everything else
shipped -- and it is now IDENTICAL to the multi-slot HEAD row at all five points. That is the honest
statement of what multi-slot bought and it is narrower than this header used to imply: THE GAIN IS
ENTIRELY IN ACCESS (a flat 0.72 against 0.805..None), and NOT ONE POINT OF IT IS IN FOCUS. The
apparent focus gain was the alphabet.

So a compliant specialist can always get ACCESS and can never become the FOCUS by volume. THE HEAD
ROW IS AT OR ABOVE THE RETIRED SINGLE-SLOT ROW AT ALL FIVE POINTS, and the `None` at the ceiling
means a tenant claiming the maximum a compliant specialist may claim cannot be out-ranked by
ANYTHING up to 1.00. That is disclosed and DEFENDED rather than apologised for, on four measured
grounds:

  1. It is not a cost the repair introduced; it is a credit this header was TAKING for the alphabet.
     The flat 0.825 row is the challenger source 'chal' sorting before the tenant 't0' at the rank
     line, and it is printed above so the withdrawal is visible rather than quietly absent.
  2. THE SLOT ROW DID NOT MOVE. What this workspace asserts, in `win_new` below, is that a compliant
     specialist wins ACCESS against every compliant tenant -- a flat [0.72 x5], unchanged. Being
     admitted is being broadcast to every subscriber and being visible in `broadcast_all()`; the
     head is one further question. Making the FOCUS cheap is not what a bottleneck is for.
  3. THE FOCUS IS MEASURABLY MORE STABLE FOR IT: head switches on this module's 48-tick stream go
     17 -> 10, with churn, retention, the flood and the silent-source hold all unchanged. A focus
     that flips whenever a marginally louder specialist posts is the flickering argmax control this
     module exists to beat.
  4. NOTHING SAFETY-SHAPED IN THE TREE READS THE HEAD THROUGH THIS ROW, checked rather than assumed:
     the 0.95 proximity threat still ignites and still takes the head on the live contract, on the
     stream's threat burst (which now holds the head 7 ticks rather than 5), and on
     `access_consciousness` claim (9)'s displaceability criterion -- all three run green.

WHAT THE `None` IS NOT, and this qualifies ground 1 rather than replacing it (added 2026-08-06 by an
adversarial re-derivation of this row). It is NOT a structural ceiling. At the band ceiling the
tenant is DEFENDED, so its drive is `0.75 + 0.25 + jitter` and a challenger at the `_g` clamp of 1.00
has `1.00 + 0 + jitter`: the two are equal to within ONE JITTER DRAW. Measured at seed 0, tenant
1.016888 against challenger 1.015159 -- a margin of 1.7e-03 -- and the membranes 0.913777 against
0.910318, which is the same 3.46e-03 gap (c1b) reports at exactly-equal claims, because it IS that
gap. So the last cell of this row is decided by the jitter, not by the physics, and the honest
statement is that the shipped key declines to hand the tie to the ALPHABET rather than that the
tenant genuinely out-drives. Swept over 12 seeds the cell is `None` on 6 and a finite 0.990..1.000 on
the other 6, and the sign of `jit(tenant) - jit(challenger)` predicts which on 12 of 12. Two things
follow and both are load-bearing for a reader: (a) "cannot be out-ranked by ANYTHING up to 1.00" is a
seed-0 fact about this harness, not a property of the arbiter -- the other four cells of the row are
genuine and move only +-0.015 across those seeds, and the SLOT row is 0.715..0.735 across all of
them, inside the band on every seed; (b) it is nevertheless a real narrowing against the deleted
single-slot CODE, which took the head at 0.955 here. The criterion this module asserts is the SLOT
row, which did not move; nothing asserts the last head cell in either direction, and it should not
be quoted as though something did.

THE ROW IS NOW PINNED BY A CRITERION AND NOT ONLY PRINTED. The old assert was `any(h is None or h >
band_hi)`, which BOTH tables satisfy, so it could not see the alphabet at all. The criterion added
below is the one the repair actually owes: RENAMING THE CHALLENGER MUST NOT MOVE THE SWEEP. Its red
is the retired key, constructed and run -- under it the same rename moves the row from
[0.720, 0.825, 0.825, 0.825, 0.825] (challenger 'aaa', sorting BEFORE the tenant 't0') to
[0.825, None, None, None, None] (challenger 'zzz', sorting after it). Same claim, same content, same
competition; four of the five points become unwinnable at any salience because of a letter.

The head is still outside the band, so `broadcast()` alone remains a poor place to look for
faculty-to-faculty influence -- `broadcast_all()` is where the change lives, and since 2026-08-06 so
does the SUBSCRIBE channel (see the delivery section below; a consumer no longer has to reach for a
second accessor to ask whether it got in).

=====================================================================================================
WHAT SUBSCRIBERS RECEIVE: THE SET, not the head (2026-08-06). The design question multi-slot CREATED.
=====================================================================================================
Multi-slot gave this workspace two different things -- CONTENTS (the admitted set) and a HEAD (the
top-ranked one) -- and left ONE push channel carrying only the second. That is wrong on the theory
and it was wrong in the tree. GWT's claim is that the workspace BROADCASTS ITS CONTENTS, and that
broadcast IS global availability; a bottleneck that admits four contents and announces one has a
capacity of four and an availability of one.

WHAT IT COST, measured on this bus rather than reasoned about, AND MEASURED ON THE ARBITER AS IT
THEN STOOD. Two contents at identical salience, capacity 4, varying NOTHING but the peer's source
name: the second content is in `broadcast_all()` on 12 of 12 ticks under every name, while a
head-only subscriber saw it on 0 of 12 against a peer named 'aaa' and 12 of 12 against 'mmm'. Two
compliant pools both SATURATE their sustained-firing counts, so the rank key FELL THROUGH `-late`
and `-total` to the source string and the head was settled lexicographically -- the defect the
tie-break block above closes at the rank line. `cortex/access_consciousness.py` read its own
ignition off that head, so a consciousness-relevant flag -- and the `access_index` behind it -- were
being decided by a string comparison. TWO INDEPENDENT FIXES, and keeping both is not additive: the
delivery change removes the channel by which a subscriber has to infer membership from rank at all,
and the rank repair removes the name from the rank. Either alone leaves one of the two halves.

THE CHANGE IS ONE LINE OF MECHANISM: `_deliver` is handed `broadcast_all()` instead of
`self.current`, on BOTH the ignition path and the argmax control, so there is one delivery contract
and a subscriber never needs to know which arbiter ran. NO second `subscribe` channel exists and
none may be added (P2): `current` / `broadcast()` / `broadcast_hold` are untouched projections of
the same `_slot()` load, `broadcast()` is still zero-arg and still returns ONE content, and
`cortex/live_brain.py::thought` is byte-identical on the live brain. A/B'd against a shim that
restores the retired contract exactly, 120 wake ticks: `thought` names the percept on 120/120 and
the crc32 of the thought sequence, of the workspace-set sequence and of the consumer's whole legacy
`state()` readout are the SAME under both contracts (`_access_ema` = 0.488958 in both), because
that bus has one admissible producer and head == set on every tick. With the consumer REPOINTED
onto it -- the only two-producer configuration this tree can build -- 60 ticks give the same null
again: that faculty is in the workspace 60/60 AND the head 60/60. HONEST CONSEQUENCE: this change
buys nothing measurable on any bus that exists today. It is shipped because the dissociation
appears the moment two producers co-exist with one out-ranking the other, and doing it before the
bus is shared is doing it in the only window where it is free.

BLAST RADIUS, derived and not estimated: `GlobalWorkspace.subscribe` has FIVE call sites tree-wide
-- `cortex/access_consciousness.py` and four inside this module's own self-check. ONE production
consumer. (`cortex/brain.py:799` subscribes to a `ReentrantWorkspace`, a different class.)

DETERMINISM is preserved by construction and re-measured: `broadcast_all()` builds a FRESH tuple of
immutables from one `_slot()` load, so a subscriber gets no handle on `_slots` and cannot observe a
later rebinding. That is strictly safer than the alternative of exposing the slot list, and it is
asserted (the delivered object must hash) rather than argued. 24 same-seed runs of a stream harness
that digests head + set + hold + EVERY DELIVERY give 1 distinct digest; so do 24 runs of the
consumer on a shared bus and 24 on a private one, and 6 of the live brain.

=====================================================================================================
LIMIT 4 -- THE TWO CONSUMER CRITERIA THAT WENT RED ARE BOTH RE-DERIVED AND GREEN. Re-run, not read.
=====================================================================================================
This block previously said `cortex/access_consciousness.py` claim (8) and `reasoning/rpm_solver.py`'s
`collide_ok` were red and were required companion work. Both were done, and both are re-run here
rather than taken from the note that scheduled them:

  * `access_consciousness` claim (8) was re-derived onto the LOAD (a population of compliant peers)
    instead of one competitor's volume, after the volume lever measured 0 of 12 and the
    `broadcast_all()`-RANK lever measured 1 of 12 and moved to 11 of 12 on a peer RENAME. Its
    observable is now ADMISSION. Green.
  * `rpm_solver` `collide_ok` was re-derived onto `broadcast_all()` membership -- its own comment
    reads "READ ACCESS, NOT FOCUS" -- and now carries four conditions including exclusion at full
    capacity and an impostor case. Green.

DISCLOSED, because the delivery change is what made it true: claim (8)'s two observables now COME
FROM ONE READ. Its `admitted` list and the `ignited` element of its own-output tuple are both
`_LABEL in broadcast_all()`, where before this pass one was membership and the other was headship.
That is a real loss of independence and it is recorded rather than left for the next reader to
find. It does not make the criterion vacuous -- the threshold is SEARCHED, the step is asserted
monotone across the whole scan, the private-bus arm is a run RED and the null re-runs three loads
-- and it is the direct rather than the proxied form of the property. But a future pass looking for
a second, independent channel on that claim should know the old one was spent here.

Verified UNAFFECTED, RE-RUN 2026-08-06 after the delivery change rather than carried over:
`cortex/live_brain.py`, `cortex/brain.py`, `brainstem/autonomous_loop.py`,
`thalamus/reentrant_workspace.py`, `cortex/self_model.py`, `cortex/executive_control.py`,
`reasoning/rpm_solver.py`, `reasoning/spike_chat_lake.py`, `reasoning/consciousness_probe.py`,
`reasoning/train_of_thought.py`, `reasoning/concept_reflection.py`,
`cortex/competence_executive.py` (verified at its pre-promotion path
`reasoning/brain_adapters/executive_control.py`; PROMOTED+RENAMED 2026-08-12, same file),
`thalamus/gwt_deep.py` (verified at its pre-promotion
path `reasoning/gwt_deep.py`; PROMOTED 2026-08-11, same file). (`reasoning/gwt_reentry.py`
stood here and was RETIRED 2026-08-06 under P2 -- its predictive-completion capability ported into
`thalamus/reentrant_workspace.py::expect` first, verified by a neuter-A/B that flips the ported claim
and leaves the other four rows unmoved. Corrected here rather than left dangling: a prose list naming
a path that no longer exists reads as coverage nobody has, and no gate reads this list, so nothing
would have caught it.)

FRAMING (agnostic, per the project's consciousness program): this is a functional spiking
attention/broadcast bottleneck (integrate -> spike -> compete -> ignite -> broadcast -> decay), a
concrete step toward the Global Neuronal Workspace substrate of ACCESS consciousness. Whether such
dynamics contribute to consciousness is an OPEN question the project actively builds toward; it is
neither denied nor claimed here. "ignition" and "broadcast" name the information-routing dynamics.

THE BINDING DEEPENING IS RETIRED (2026-08-07, P8 burden 5 -- deletion on a null). This header
carried "spikes WIN at K>=2 (multi-item binding via synchrony) -- the specific, honest answer to
'what do spikes buy'" for as long as the family existed, and the claim died in two steps the same
day. First its margin criteria were deleted (F11): `_bind_pair_acc` was recall with no precision
term, `bind_decode_spiking` returned K^2 pairs for K true, and an all-pairs strawman decoder scored
1.0000 on the same scorer -- the SECOND deletion of that criterion, `reasoning/consciousness_probe.py`
having deleted its copy 2026-08-04, after which this one read as independent corroboration. Then
the honest precision-bearing benchmark was run and returned the pre-registered NULL, sharper than
predicted: the spiking decoder's Jaccard EQUALS the all-fired-pairs strawman at every K (1.0000 /
0.5000 / 0.2500 at K = 1/2/4), at K=2 the rate control is nominally BETTER, and no margin at any K
clears the control's own per-trial spread. Phase multiplexing recovered membership, never pairing.
The functions, the `_BIND_*` constants, the undeclared 0.55 sustain that produced the cross
product, the harness and the ignition assert are all gone; the measurement is preserved in the
retirement record above `main()`, and the three-way P2 merge that hung on this evidence
(GlobalWorkspace binding / bind_* / ThetaGammaCode) is REFUTED rather than deferred. What this
module still claims spikes buy is what its remaining criteria measure: the all-or-none ignition
bottleneck, sustained-firing admission, and the stability-vs-argmax margin -- nothing about
feature binding.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.thalamus.global_workspace
"""

import itertools
import json
import random
import threading
import time

REGIONS = ("perception", "memory", "salience", "language", "motor")


def _fold_parts(parts):
    """The shared inhibitory pool: `sum()` over the per-chunk contribution lists CONCATENATED in
    ascending chunk order, which is exactly the flat sequence the unpartitioned code summed.

    This is the ONE definition both the serial and the parallel settling paths call, and the choice
    of `sum` is load-bearing rather than idiomatic. Two ways to get this wrong, both measured on this
    module rather than reasoned about:

      * summing each chunk and adding the SUBTOTALS -- the obvious way to parallelise a reduction --
        reassociates the additions and gives a different float. On 240 pools it differs at some
        partition counts and agrees at others, so it is not even consistently wrong; it makes the
        result depend on the thread count.
      * a naive `acc += x` left fold is ALSO not equivalent, which is the trap here. CPython's
        builtin `sum()` applies Neumaier compensated summation to floats, so it is strictly more
        accurate than an accumulator loop and disagrees with one in the last bits. A hand-rolled
        fold "in serial order" therefore silently changes the numbers; this lane wrote that fold
        first, and the integer spike counts absorbed the difference and reported byte-identical at
        ten widths. It was caught only by comparing MEMBRANE FLOATS against a frozen reference.

    Concatenating and calling the same builtin avoids both: identical elements, identical order,
    identical algorithm."""
    return sum(itertools.chain.from_iterable(parts))


# distinct-from: substrate/signal_ring.py -- the ring frames TRANSPORT records; this is
#   the workspace's COMPETITION record (the ring carries its bytes). distinct-from:
#   cortex/self_model.py -- self-episode history; this is per-tick coalition identity.
#   distinct-from: substrate/kernel_tick.py -- the neuron-substrate tick frame. Read.
class Coalition:
    """The COALITION RECORD (specs/022 T031, data-model 'Episode / Coalition'):
    {coalition_id, tick, proposer_ids[], content_ref, base_level, current, channels{...}}.

    Replaces the buffer's (source, content, salience) 3-tuple as the unit of competition
    WITHOUT touching the kernel: `as_tuple()` is the exact projection the kernel body
    consumes (same fields, same floats, same iteration order), while the record carries
    what the tuple could not -- identity (coalition_id, a LAKE-CONTINUED ordinal assigned
    at settle -- unique across restarts because it continues from the destination lake's
    own count, never from a process counter), provenance (proposer_ids), TYPED channels
    from the declared producers (salience always; the arm feeds arousal/caution from its
    live medullary interoception), and the post-settle `current` the tuple threw away.
    `tick` and `current` are written back by step(); an unsettled record has tick=None,
    current=None -- honest, never defaulted. `tick` is the workspace's WITHIN-LIFE settle
    clock (restarts at 1 with the instance); cross-life ordering is the id ordinal /
    lake append order, stated here so no consumer joins on tick across sessions.

    Persistence: to_bytes()/from_bytes() frame the record as the lake's _T_COALITION
    payload (tag registered in substrate/home.py; an IMMORTAL class per
    contracts/lake-records.md). JSON body -- symbolic content today (T030's disclosed
    provisional typing); the round-trip is asserted byte-exact in the self-check."""

    __slots__ = (
        "coalition_id",
        "tick",
        "proposer_ids",
        "content_ref",
        "base_level",
        "current",
        "channels",
    )

    def __init__(
        self,
        proposer_ids,
        content_ref,
        base_level,
        channels,
        tick=None,
        current=None,
        coalition_id=None,
    ):
        self.proposer_ids = list(proposer_ids)
        self.content_ref = content_ref
        self.base_level = float(base_level)
        self.channels = dict(channels)
        self.tick = tick
        self.current = current
        self.coalition_id = coalition_id

    def as_tuple(self):
        """The kernel's view -- exactly the old 3-tuple, nothing more."""
        return (self.proposer_ids[0], self.content_ref, self.base_level)

    # The record QUACKS like the 3-tuple so every kernel consumer -- `for src, content,
    # sal in buffer`, `c[2]`, `c[0]` -- reads bit-for-bit what it always read. The kernel
    # body is untouched by T031; only the record type changed underneath it.
    def __iter__(self):
        return iter(self.as_tuple())

    def __getitem__(self, i):
        return self.as_tuple()[i]

    def __len__(self):
        return 3

    # distinct-from: substrate/minhash.py -- Jaccard estimation over iterables; no record.
    # distinct-from: reasoning/arc_agi/play_brain.py -- SimHash occupancy features. Dunder-
    #   phrase overlap only; these two lines are the Coalition record's tuple-projection
    #   equality, nothing else defines equality for this record. Both sites read.
    # Equality is BY PROJECTION, same as everything else the kernel sees: a record equals
    # the 3-tuple it projects (and another record projecting it). Channels/identity do NOT
    # enter -- lake identity is to_bytes(), never ==. Hash matches, so either form keys a
    # dict/set interchangeably.
    def __eq__(self, other):
        if isinstance(other, (Coalition, tuple)):
            return self.as_tuple() == tuple(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.as_tuple())

    # distinct-from: substrate/wiring_manifest.py -- hdf5 artifact codec, not this record.
    # distinct-from: substrate/envelope_hash_gate.py -- envelope baseline writer.
    # distinct-from: cortex/neurogenesis_engine.py -- growth-state npz restore.
    # distinct-from: basal_ganglia/self_build_loop.py -- firewalled-file byte restore.
    # distinct-from: cortex/trained_brain.py -- crc32 seed derivation. Serializer-phrase
    #   overlap only; THIS pair is the _T_COALITION lake payload codec and nothing else
    #   encodes that record. All sites read.
    def to_bytes(self):
        # JSON-native content rides AS-IS (str/int/float/bool/None -- today's live tokens),
        # so serialize(deserialize(b)) == b holds and the self-check asserts it. Anything
        # else falls to repr() ONE-WAY -- disclosed provisional (T030's symbolic-payload
        # note), never re-reprred on a round trip because the fallback only fires on
        # non-native types and a deserialized record's content is always native.
        content = self.content_ref
        if not isinstance(content, (str, int, float, bool, type(None))):
            content = repr(content)
        payload = {
            "id": self.coalition_id,
            "tick": self.tick,
            "proposers": self.proposer_ids,
            "content": content,
            "base": self.base_level,
            "current": self.current,
            "channels": self.channels,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw):
        d = json.loads(raw.decode("utf-8"))
        return cls(
            proposer_ids=d["proposers"],
            content_ref=d["content"],
            base_level=d["base"],
            channels=d["channels"],
            tick=d["tick"],
            current=d["current"],
            coalition_id=d["id"],
        )


class GlobalWorkspace:
    """A LIMITED-CAPACITY workspace of competing LIF pools with all-or-none ignition.

    Capacity is a small number of contents at a time -- that bound IS the bottleneck, and it is
    what distinguishes a workspace from a message bus. Specialists call `submit(source, content,
    salience)` to post candidates for the current tick; `step()` runs one settle-to-ignition cycle
    of the spiking network and returns the TOP broadcast content (or None). `broadcast_all()`
    returns everything in the workspace. `subscribe(fn)` registers a module callback that receives
    every broadcast (global reach).

    The salience of a submission is the EXTERNAL CURRENT driving that content's LIF pool. A pool
    fires only above rheobase (`leak * v_th == ignition_threshold`), so a sub-threshold salience
    never ignites; a supra-threshold one integrates to a spike, recruits recurrent self-excitation,
    and -- if it survives the shared-inhibition competition and ranks inside the capacity --
    reaches SUSTAINED firing and is broadcast.

    `use_ignition=False` is the FAIR CONTROL: the spiking dynamics are switched off and `step()`
    becomes a pure per-tick argmax over the submissions (no LIF, no attractor, no inhibition, one
    content). Everything else -- the same submissions, the same candidates -- is identical.

    THE DECLARED READ SURFACE IS THREE QUESTIONS ABOUT ONE WORKSPACE, not three buses. `current` /
    `broadcast()` answer WHAT is at the top; `broadcast_all()` answers WHAT ELSE is available;
    `broadcast_hold` answers HOW STRONGLY the top content is still held. All three go through the
    single `_slot()` load of the one slot tuple -- see `broadcast_hold` for why a second question
    needed a route of its own and why these are projections of one arbiter's output rather than
    additional mechanisms (P2).

    THE PUSH SIDE IS ONE CHANNEL AND IT CARRIES THE SET. `subscribe(fn)` is registered once and
    `fn` is handed the whole ranked admitted tuple on every cycle -- the same value `broadcast_all()`
    returns -- because that is what "globally broadcast" means and because a head-only push made a
    subscriber's own membership unanswerable (`_deliver` carries the measurement). There is exactly
    one delivery; nothing here pushes the head separately.
    """

    # -- fixed LIF network constants (per-instance; deterministic, ASCII) --------------
    _T = 60  # settling micro-steps per workspace tick
    _LEAK = 0.5  # membrane leak (V decays toward input/leak each micro-step)
    _W_SELF = 0.6  # recurrent self-excitation gain (within-tick attractor)
    _TAU_SELF = 0.6  # within-tick self-excitation synaptic-trace decay
    _K_INH = 0.30  # excitatory-spike -> shared inhibitory-pool drive
    _LEAK_INH = 0.5  # shared inhibitory-pool leak
    _JITTER = 0.02  # tiny deterministic symmetry-break so exact ties resolve to a fixed order
    # `_G_INH = 2.4  # TOTAL lateral-inhibition budget` was RETIRED here: it was exactly
    # `_W_SELF * capacity`, i.e. the per-slot E/I balance multiplied back up by the capacity it was
    # about to be divided by. See `_lateral_gain`, which is byte-identical at the shipped capacity.

    def __init__(
        self,
        ignition_threshold=0.55,
        decay=0.85,
        hysteresis=0.20,
        self_boost=0.05,
        use_ignition=True,
        seed=0,
    ):
        self.ignition_threshold = (
            ignition_threshold  # rheobase current: a pool fires iff drive > this
        )
        self.decay = decay  # cross-tick attractor decay (persistence fade)
        self.hysteresis = hysteresis  # a slot occupant's extra recurrent defence current...
        self.self_boost = self_boost  # ...(self_boost + hysteresis) holds the attractor
        self.use_ignition = use_ignition
        self._v_th = (
            ignition_threshold / self._LEAK
        )  # membrane threshold s.t. rheobase == ignition_threshold
        self._rng = random.Random(seed)  # per-instance RNG (deterministic symmetry-break jitter)
        self._slots = ()  # ranked tuple of (source, content, hold) occupying the workspace
        self._g_lat = self._lateral_gain  # per-slot inhibition, E/I balanced (see `_lateral_gain`)
        self._g_free = self._inh_free  # the shared inhibitory pool's rheobase (see `_inh_free`)
        self._last_focus = {}  # (source, content) -> this tick's FOCUS drive (claim + grip)
        self._buffer = []  # this tick's submitted candidates
        self._subscribers = []  # module callbacks receiving each broadcast
        self._g = {}  # (source, content) -> cross-tick attractor memory in [0,1]
        self._jit = {}  # (source, content) -> fixed tiny jitter current
        self._last_total = {}  # (source, content) -> spikes this tick (spiking self-check)
        self._last_late = {}  # (source, content) -> 2nd-half spikes this tick (sustained)
        self._sim_per_pool_s = 0.0  # running-EMA cost of settling ONE pool, seconds (P1 baseline)
        self._sim_par_eff = 1.0  # running-EMA fraction of IDEAL parallel savings actually achieved
        self._slot_buf = []  # per-submitter staging buffers for CONCURRENT specialists
        self._last_v = {}  # (source, content) -> final membrane float (determinism self-check)
        self._cycle = 0  # settles run so far THIS LIFE -- the Coalition record's tick field (T031)
        self._out_ring = None  # SignalRing this workspace publishes settled records onto
        self._settled = ()  # last settle's Coalition records, focus order (T031 read surface)
        self._ordinal = 0  # next coalition_id ordinal; attach_ring() rebases it on LAKE state

    # -- the two subsidies an occupant carries, and why they are two ------------------------------
    @property
    def _lateral_gain(self):
        """Per-slot lateral inhibition. EXCITATION AND INHIBITION ARE BALANCED PER SLOT, so this is
        `_W_SELF` -- the same gain the pool's own recurrent self-excitation carries.

        THIS RETIRED A TYPED CONSTANT (P1). It read `_G_INH / capacity` with
        `_G_INH = 2.4  # TOTAL lateral-inhibition budget`, and 2.4 was never independent: measured,
        `2.4 == _W_SELF * capacity` exactly at the shipped capacity, and `2.4 / 4 == 0.6 == _W_SELF`
        is an EXACT float identity, not a near-miss. So the "total budget" was the per-slot balance
        multiplied back up by the capacity it was about to be divided by again, and the 2.4 was
        that round trip written down as a number. The shipped workspace is BYTE-IDENTICAL under
        this form; what changes is that a workspace of any other capacity stays E/I balanced
        instead of inheriting one capacity's arithmetic.

        The division-across-capacity property the multi-slot repair was built on is unchanged and
        is now structural rather than arithmetic: each slot gets one unit of inhibition against one
        unit of self-excitation, so holding a slot costs A SLOT and not the silence of every
        competitor.

        A CAPACITY-DERIVED QUANTITY IS BACK IN THE KERNEL, and the reconciliation is worth being
        exact about because this docstring used to read as if removing one was the achievement.
        What was removed here was a capacity dependence in the GAIN's arithmetic (the `_G_INH`
        round trip). What `_inh_free` adds is a capacity-derived RHEOBASE on the shared inhibitory
        pool -- a property of the WORKSPACE, exactly like `capacity` itself, and NOT of the load,
        which is why the refusal recorded in `capacity`'s docstring (making the gain track the
        competitor count destroyed winnability, 0.72 -> None) is preserved, not contradicted: the
        gain still never moves with how many pools happen to be posting."""
        return self._W_SELF

    # -- the shared inhibitory pool's rheobase ---------------------------------------------------
    # distinct-from: episode_consolidation.py -- that `_kwta` is a hard top-K cardinality cut over
    #   hippocampal attractor units; THIS workspace's top-K is `fired[:cap]`, which already exists
    #   and is untouched here. What is added is a firing threshold on a dynamical shared inhibitory
    #   pool, a mechanism neither module has.
    # distinct-from: inhibitory_stdp.py -- per-synapse PLASTIC inhibitory weights onto one neuron
    #   (Vogels 2011); this is ONE shared diffuse pool with a DERIVED, non-learned threshold.
    @property
    def _si_sat(self):
        """A SATURATED pool's contribution to the shared inhibitory pool: the fixed point of the
        `si += _K_INH * sp - _LEAK_INH * si` update under sustained every-micro-step firing, i.e.
        `_K_INH / _LEAK_INH`. DERIVED, never written as a literal: typing the 0.6 with this
        derivation in a comment beside it is the exact failure `_G_INH = 2.4` was retired for --
        a constant that only ever agrees with its own derivation cannot be caught drifting. The
        self-check asserts this MOVES when `_K_INH` or `_LEAK_INH` move."""
        return self._K_INH / self._LEAK_INH

    @property
    def _inh_free(self):
        """The shared inhibitory pool's RHEOBASE: the activity it ignores before it starts to
        inhibit, set at what `capacity` saturated slots produce.

        Every excitatory pool in this network is a spiking LIF with a threshold (`_v_th`); until
        this property existed the shared inhibitory pool was the ONLY unit with none -- `a_inh`
        drove lateral inhibition linearly from its first spike. The consequence, measured live
        over 120 wake ticks, was a 4-slot workspace running 1-slot: occupancy `{1: 117, 2: 3}`,
        because a saturated incumbent's `si` contribution alone subtracted `_W_SELF * _si_sat =
        0.36` from every rival's drive, and live claims in [0.51, 0.75] against `_v_th = 1.1`
        cannot survive that. The weaker pool was annihilated BEFORE `fired` was built, so
        `fired[:cap]` -- the explicit, derived, Cowan-corroborated bound -- was a no-op on the
        production path. Two mechanisms limited the held count; P2 says exactly one survives, and
        the one that survives is the one that is derived.

        THE VALUE. Pool i feels `a_inh - si[i]`; among N mutually-saturated pools that is
        `(N - 1) * _si_sat`. Silent through N == capacity therefore means a threshold of
        `(capacity - 1) * _si_sat` -- 1.8 at the shipped capacity 4, and EXACTLY 0.0 at capacity
        1, which is why the single-slot control (`_SingleSlotControl`, capacity pinned at 1) is
        byte-identical to the pre-rheobase kernel and every criterion built on it is untouched.

        OWNED, NOT SMOOTHED: at or below capacity the lateral term is identically 0.0. Inhibition
        no longer decides the SET -- `fired[:cap]` does -- it shapes the RATE above the allowance
        (measured: sigma-total 348 -> 198 across a 50x gain sweep on which the admitted set is
        pinned; the new criterion's third conjunct asserts the gain stays load-bearing so this
        cannot silently become a dead parameter)."""
        return (self.capacity - 1) * self._si_sat

    @property
    def grip(self):
        """An incumbent's advantage in the contest for the HEAD -- and it is NOT its hold on the
        slot. `hold / capacity`. DERIVED; it introduces no constant.

        THE DEFECT THIS SPLIT REPAIRS (#16). One subsidy, `self_boost + hysteresis`, used to do two
        jobs: sustain an admitted assembly (retention) AND decide which assembly is broadcast
        (dominance). Those jobs want opposite magnitudes, and with one number they cannot both be
        served -- which is provable rather than a matter of taste:

          * A defended incumbent claiming `c` ranked at `c + hold`; the loudest a COMPLIANT rival
            may claim is `hi = 1 - hold`. It out-ranks the incumbent only where `c < 1 - 2*hold`,
            which intersects the band at all only when `hold < (1 - lo)/2`. At the shipped
            `hold = 0.25` against `lo = 0.55` that bound is 0.225, so NO tenant anywhere in the
            band was displaceable from the head by any compliant rival. Measured on this module's
            own sweep, the head row read `[0.805, 0.835, 0.910, 0.955, None]` against a 0.75
            ceiling -- outside the band at all five points. The first tenant held the focus
            permanently, and `live_brain.thought` reads the head.
          * Lowering `hold` to satisfy that bound was BUILT AND MEASURED AND REJECTED. The hold is
            also what `capacity` divides the unit budget by, so lowering it widens the workspace,
            and past capacity 4 the bottleneck stops binding: at capacity 5 an above-band tenant
            became joinable by a compliant challenger (the winnability red of that era lost its
            reach; a FULL workspace's exclusion of a compliant challenger carries the red today)
            and at capacity 6 a FULL workspace admits one. The candidate `(1 - lo)/3 = 0.15` --
            the value making exactly half the band displaceable -- reddens five criteria across
            three modules. Inside the window that leaves, `(0.2000, 0.2250)`, EVERY value costs
            workspace churn 9 -> 15 on this module's own 48-tick stream (its criterion is <= 12)
            and retention 47/47 -> 45/47, because entrenchment is what buys stability and
            entrenchment is precisely what the defect is made of. Churn-clean and #16-clean are
            DISJOINT on that axis: 0.2500 fails #16, 0.2250 fails both, everything below fails
            churn. Swept, and the rheobase axis has no admissible value at all.

        So the parameter was not the problem; ONE subsidy doing two jobs was. Splitting them costs
        nothing on either side: admission still ranks on the full hold, so retention, churn and the
        capacity are untouched, and only the ordering of the already-admitted set changes.

        WHY `hold / capacity` AND NOT ANOTHER FRACTION. It is this module's own division principle,
        reused rather than re-invented: the lateral budget is what one slot's share of a
        workspace-wide entitlement looks like, and the grip is the same shape -- a workspace-wide
        hold, divided by the number of occupants entitled to it. A wider workspace grips its focus
        more weakly, which is the same sentence as "a wider workspace competes less fiercely per
        slot" and falls out of the same division. It also lands, unforced, well inside the bound
        the first bullet derives: `lo + grip` is 0.6125 against a 0.75 ceiling.

        THE BIOLOGY IT ANSWERS TO, since this is the part a single subsidy got wrong. Recurrent
        excitation is what sustains a cortical assembly -- that is the hold, and it is large. Which
        assembly DOMINATES is set by afferent drive under biased competition, not by how long the
        incumbent has been there. An arbiter in which the sitting tenant's reverberation also
        decides what is attended to is one where attention cannot be captured, and a brain whose
        focus cannot be captured by a stronger signal is not a brain that can be startled."""
        return (float(self.self_boost) + float(self.hysteresis)) / self.capacity

    # -- the capacity of the bottleneck ----------------------------------------------------------
    @property
    def capacity(self):
        """How many contents this workspace can hold at once. DERIVED, not picked.

        THE DERIVATION. `self_boost + hysteresis` is what it COSTS to hold a slot: it is the extra
        recurrent drive a slot occupant is granted, on this module's normalized [0,1] drive scale
        (the same scale `_g` is clamped to, `broadcast_hold` contracts to, and `1 - defence` is the
        band ceiling of). A workspace whose total recurrent hold budget is one unit of that scale
        can therefore hold `floor(1 / defence)` contents. At the defaults that is
        `floor(1 / 0.25)` = 4.

        WHY THAT IS A DERIVATION AND NOT A HAND-PICKED 4 (P1). It introduces NO new constant: both
        the budget (1.0, the module's normalized ceiling -- P1 honesty guard case (a), a fixed
        point on a normalized scale) and the cost (`self_boost + hysteresis`) already exist and
        already arbitrate. It MOVES when they move, which is the falsifier and is asserted in the
        self-check: defence 0.25 -> 4, 0.33 -> 3, 0.50 -> 2, 1.00 -> 1. It is also the parameter
        this module's own defect was about -- the defence was previously free to the incumbent and
        is now what a slot costs -- so capacity and winnability are derived from the same quantity
        rather than tuned against each other.

        The number it lands on is corroborated rather than justified by Cowan (2001, Behav Brain
        Sci 24:87, "the magical number 4") and Luck & Vogel (1997, Nature 390:279): the capacity of
        conscious access measures ~4 items. GNWT itself is explicit that the workspace is
        capacity-LIMITED and never states a number -- Dehaene & Changeux (PLoS Biol 3(5):e141,
        2005) describe "a single limited-capacity serial workspace" -- so the literature fixes the
        SHAPE of this quantity and not its value, which is exactly why it has to be derived here.

        WHAT WOULD FALSIFY IT: a measurement showing this brain's specialists carry more (or
        fewer) simultaneously usable contents than the defence budget affords. Nothing in this tree
        measures that today -- there is no consensus benchmark (P2's standing note on this very
        module) -- so the derivation stands on its own terms and the number is not credited beyond
        them.

        DEGENERATE INPUTS, checked because a P1 rewrite in this tree shipped a `mu/1e-12 = 1e12`
        singularity onto a live path. There is no unbounded denominator here: the divisor is a
        MAXIMUM, so it cannot approach zero. ZERO defence -> nothing is held at all, so there is no
        multi-tick workspace, only a per-tick competition, and this returns 1 -- the maximally
        selective answer, which is the safe direction for a bottleneck. A defence BELOW the
        symmetry-break jitter cannot hold a content against this workspace's own tie-break noise,
        so `_JITTER` floors the cost and bounds the count at 49; a workspace configured there is
        not a bottleneck and this module does not pretend otherwise. EMPTY / SILENCE / SINGLE do
        not enter: capacity is a property of the workspace, never of the load -- a workspace does
        not become more selective because less is competing, and making the lateral gain depend on
        the competitor count was measured to destroy winnability (0.72 -> None) and is why this is
        a property rather than a per-tick computation.

        THE DIVISOR IS THE FLOAT DEFENCE, NOT THE DECIMAL ONE, and the gap shows up at exactly the
        values a reader is most likely to try. `1.0 // d` floors the quotient of the two DOUBLES,
        so a defence whose decimal reciprocal is a whole number but which is not exactly
        representable in binary lands one BELOW the decimal reading: measured, 0.20 -> 4 (not 5),
        0.10 -> 9 (not 10), 0.05 -> 19, 0.04 -> 24, 0.02 -> 49 (not 50). Dyadic defences agree
        exactly (0.25 -> 4, 0.125 -> 8, 0.50 -> 2), which is why the SHIPPED default is unaffected.
        This is the correct floor for the defence the workspace ACTUALLY holds -- `fl(0.20)` is
        0.2000000000000000111, and five slots of it overspend the unit budget -- so it is
        documented rather than rounded away into a capacity that would spend more than there is.
        The rounding MODE is falsifiable rather than assumed: the table in the self-check carries a
        defence of 0.15, where floor gives 6 and round gives 7. Every other row in it (0.25, 0.33,
        0.50, 1.00) is a value at which floor and round agree, so before that row existed a
        mutation replacing `//` with `round` passed the entire self-check -- run, not argued."""
        defence = self.self_boost + self.hysteresis
        if defence <= 0.0:
            return 1
        return max(1, int(1.0 // max(defence, self._JITTER)))

    # -- specialist / module interface ----------------------------------------------------------
    # distinct-from: thalamus/reentrant_workspace.py -- the re-entry stage's own submit
    #   (different kernel, its own record shape is NOT changed here); this is the shared
    #   bus's buffer becoming Coalition records. Read.
    def submit(self, source, content, salience):
        """A specialist posts a candidate into this tick's competition buffer.

        SINGLE-THREADED ONLY. This appends to one shared list, and buffer ORDER is observable: the
        `_g` dict is built by iterating the buffer, and `_jitter_for` draws each pool's tie-break
        current from the instance RNG in first-encounter order. Two specialists calling this
        concurrently would therefore make the symmetry-break jitter depend on which thread won a
        race, which is exactly the nondeterminism this module's determinism self-check forbids.
        Concurrent specialists use `open_slots()` + `submit_from()` below instead.

        T031: the buffer entry is a COALITION RECORD (the 3-tuple is its projection --
        the kernel's iteration order, keys and float values are bit-for-bit what they
        always were; the record adds identity, provenance and typed channels without
        touching the competition). This 3-arg form seeds channels={'salience': s};
        submit_coalition() below carries the full typed set."""
        self._buffer.append(
            Coalition(
                proposer_ids=[str(source)],
                content_ref=content,
                base_level=float(salience),
                channels={"salience": float(salience)},
            )
        )

    def submit_coalition(self, source, content, channels):
        """T031: post a candidate with TYPED CHANNELS (data-model: channels{salience,
        valence, arousal, interoception, ...}). 'salience' is MANDATORY -- it is the
        competition drive; every other channel rides as provenance the record persists
        and downstream consumers (Episode consolidation, the report channel) read."""
        ch = {k: float(v) for k, v in dict(channels).items()}
        if "salience" not in ch:
            raise ValueError("submit_coalition requires a 'salience' channel (the drive)")
        self._buffer.append(
            Coalition(
                proposer_ids=[str(source)],
                content_ref=content,
                base_level=ch["salience"],
                channels=ch,
            )
        )

    # distinct-from: substrate/signal_ring.py -- the ring/journaller TRANSPORT this attaches;
    #   this method only binds the producer side and rebases identity. Read.
    def attach_ring(self, ring, base_ordinal=0):
        """T031: hand this workspace a SignalRing to publish settled Coalition records onto.

        The persistence chain is the EXISTING one, reused whole (P2): workspace frames
        `_T_COALITION + record.to_bytes()` (framing belongs to the producer -- the ring's own
        contract), SignalRing transports, and whoever owns the lake drains a LakeJournaller
        over this ring into the ONE BinLog -- overrun fatal there, per P4b. No ring attached
        (the default) means records still settle onto `self.settled` for in-process readers;
        nothing is silently persisted anywhere else.

        `base_ordinal` is the LAKE-DERIVED identity base: the caller passes how many
        _T_COALITION records the destination lake already holds, and ids continue from
        there. This is what makes coalition_id an identity in the DURABLE store rather than
        in one process's memory -- the review that forced it measured the instance-derived
        scheme losing 16 of 24 restart records to the lake's byte-identical dedup and
        naming 7-of-10 pairs of DIFFERENT lived records with the same id. Identity from
        lake state, never from process state. Two workspaces journaling one lake would need
        disjoint bases; today exactly one ringed workspace exists (the arm bus -- measured,
        live_brain.py's is the only workspace=True construction) and the single-writer ring
        binds that down to one thread besides."""
        self._out_ring = ring
        self._ordinal = int(base_ordinal)

    @property
    def settled(self):
        """Last settle's Coalition records in focus order -- the SUBMISSION-SETTLE stream,
        NOT the occupancy view. A record exists only for a post that settled THIS tick; a
        defended incumbent that did not re-post is still in `_slots`/`broadcast_all()` and
        deliberately absent here, so `len(settled) <= len(broadcast_all())` and the lake
        stream records when coalitions were POSTED-AND-ADMITTED, never how long they were
        held. Duration/occupancy is `_slots`' job; a consumer needing release semantics
        (T035 Episode consolidation) derives them from its own reads of the occupancy view,
        not from gaps in this stream. (An earlier docstring claimed equivalence to
        broadcast_all(); the review measured the held-over case and the claim was false --
        corrected rather than the stream padded with synthesized rows, because a record's
        provenance is the proposer's ACTUAL post.)"""
        return self._settled

    def _record_settle(self, admitted_keys):
        """Close the settle for the record layer: stamp identity onto the buffer records the
        kernel admitted, retain them on `self.settled`, publish framed bytes to the attached
        ring. Runs BEFORE the buffer clears, on both kernel paths.

        Held-over occupants that did not re-post this tick have no buffer record and get none
        -- a record per SUBMISSION-THAT-SETTLED, never a synthesized row (the record's
        provenance is the proposer's actual post; occupancy across ticks is `_slots` /
        `broadcast_all()`, a DIFFERENT stream -- see `settled`). Where one (source, content)
        was posted twice, the max-base_level record wins -- the same dedup rule `submitted`
        applies; keyed on repr(content) so the fair-control path, which never required
        hashable content, still never does. `current` is the level the SLOT carries: the
        attractor `_g` on the ignition path, and the control path's clamped hold where `_g`
        is empty by construction -- one meaning per accessor, both paths.

        IDENTITY: `coalition_id = c<ordinal>` where the ordinal CONTINUES FROM THE LAKE
        (attach_ring's base_ordinal) and increments per stamped record. Unique in the
        durable store across restarts; `tick` is deliberately NOT that -- it is this LIFE'S
        settle clock, restarting at 1 each construction, honest as a within-life ordering
        only. The lake's append order (equivalently the id ordinal) is the cross-life
        sequence. Deterministic under G5 replay: same lake state + same inputs -> same ids
        (no wall clock, no global RNG)."""
        self._cycle += 1
        best = {}
        for rec in self._buffer:
            key = (rec.proposer_ids[0], repr(rec.content_ref))
            if key not in best or rec.base_level > best[key].base_level:
                best[key] = rec
        out = []
        for key in admitted_keys:
            rec = best.get((key[0], repr(key[1])))
            if rec is None:
                continue  # defended incumbent, no fresh post -- no record to stamp
            rec.tick = self._cycle
            g = self._g.get(key)
            rec.current = float(g) if g is not None else min(1.0, max(0.0, rec.base_level))
            rec.coalition_id = f"c{self._ordinal}"
            self._ordinal += 1
            out.append(rec)
        self._settled = tuple(out)
        if self._out_ring is not None:
            from snn_data_lake.substrate.home import _T_COALITION

            for rec in out:
                self._out_ring.publish(_T_COALITION + rec.to_bytes())

    def open_slots(self, n_slots):
        """Allocate `n_slots` DISJOINT staging buffers, one per concurrent specialist, and return the
        count. Call this from the single thread that is about to fan out, BEFORE starting any worker.

        This is the collect-then-apply answer to 'the workspace is a shared mutable structure'. It
        needs no lock because it removes the sharing rather than arbitrating it: each specialist owns
        buffer `slot` outright, the outer list is only read once the workers exist, and `step()`
        splices the buffers back in ASCENDING SLOT ORDER. Thread completion order therefore cannot
        reach the competition, and the result is a pure function of (which slot submitted what, in
        what program order) -- identical whether the specialists ran concurrently or one at a time."""
        n = max(0, int(n_slots))
        self._slot_buf = [[] for _ in range(n)]
        return n

    def submit_from(self, slot, source, content, salience):
        """A CONCURRENT specialist posts a candidate into its own staging buffer. Safe to call from
        `slot`'s own thread with no lock: no other thread touches `self._slot_buf[slot]`.

        T031: stages the same Coalition record submit() builds -- the buffer holds ONE
        type whichever door a candidate came through."""
        self._slot_buf[int(slot)].append(
            Coalition(
                proposer_ids=[str(source)],
                content_ref=content,
                base_level=float(salience),
                channels={"salience": float(salience)},
            )
        )

    def _drain_slots(self):
        """Splice the staged concurrent submissions into the competition buffer, ascending slot order
        then program order within a slot. Deterministic by construction; a no-op when nothing fanned
        out, so the single-threaded path is bit-for-bit what it always was."""
        if not self._slot_buf:
            return
        for buf in self._slot_buf:
            if buf:
                self._buffer.extend(buf)
                buf.clear()

    def subscribe(self, fn):
        """Register a module callback `fn(contents)` that receives every broadcast.

        `contents` IS THE ADMITTED SET -- the identical ranked tuple `broadcast_all()` returns for
        that same cycle, never the head alone. A subscriber that wants the FOCUS takes
        `contents[0]`; a subscriber that wants to know whether ITS content reached the workspace
        asks `mine in contents`. Those are two different questions and this channel can now answer
        both; see `_deliver` for why it delivers the set and what the head-only contract cost."""
        self._subscribers.append(fn)
        return fn

    # -- the DECLARED read surface: WHAT is in the workspace, and HOW STRONGLY it is held --------
    def _slot(self):
        """ONE load of `self._slots` -- the single point at which every declared read observes the
        workspace, and the reason each of them is race-free on the free-threaded build.

        CONCURRENCY, stated as the argument rather than as a hope (this is cp314t, Py_GIL_DISABLED).
        THIS ARGUMENT IS UNCHANGED BY THE MOVE TO MULTIPLE SLOTS, and keeping it unchanged is why
        the slots are a TUPLE rather than a list or a dict. `_slots` is only ever REBOUND
        (`self._slots = (...)`), never mutated in place, and it holds tuples of immutables all the
        way down. A load therefore sees either the whole previous workspace or the whole new one --
        there is no torn read, no partially-filled workspace is ever observable, and no lock is
        needed to take that snapshot. A list of slots mutated in place would have made every one of
        those sentences false, which is the thing multi-slot most easily gets wrong: more slots is
        a strictly larger shared-mutable surface unless the whole set is swapped at once.

        What DOES need care is loading it TWICE. `self._slots[0][1] if self._slots else None`
        evaluates its test on one load and its subscript on ANOTHER, so a concurrent `step()` that
        empties the workspace between them raises `IndexError: tuple index out of range`. The
        self-check constructs exactly that interleaving deterministically and shows the two-load
        form raising on it.

        Cross-ACCESSOR coherence is a separate and weaker guarantee, stated rather than implied: two
        accessor calls are two loads, so a reader that needs `current` and `broadcast_hold` from the
        SAME tick must take them from the same tick -- which the live wake tick, single-threaded
        through `step()`, already does. This snapshot removes the race INSIDE a read; it does not
        make the workspace thread-safe and does not claim to (`submit()` says the same)."""
        return self._slots

    @property
    def current(self):
        """The TOP content in the workspace (the highest-ranked broadcast), or None if empty."""
        slots = self._slot()
        return slots[0][1] if slots else None

    def broadcast(self):
        """The top workspace content -- what every module can see right now.

        ZERO-ARG AND STAYING THAT WAY. This is the tree's ONE genuine cross-module workspace read
        (`cortex/live_brain.py::thought` -> `self.arm.workspace.broadcast()`), and `tools/p6_gate.py`
        classifies the zero-arg `broadcast` as THE reader. (Until 2026-08-12 the gate also
        classified a 3-arg `broadcast` as a WRITE -- the retired WorkspaceBroadcast's submit
        wearing a reader's name; that arity entry left the gate's API table with the class, so a
        future 3-arg method named `broadcast` would be INVISIBLE to the gate until added to its
        closed allowlist by hand.) Its return type is a single content, and every `is not None`
        test in the tree is written against that. Widening it to return the whole workspace would
        break both."""
        return self.current

    def broadcast_all(self):
        """EVERY content in the workspace, ranked, as a tuple -- the multi-slot read.

        `broadcast()` answers "what is the brain thinking about"; this answers "what ELSE is
        available at the same time", which is the question a single slot could not be asked. It is
        the arbiter's full output; `broadcast()` and `broadcast_hold` are projections of that same
        one result (P2 -- one arbiter, three questions, no second mechanism and no second
        competition). Empty workspace -> `()`, never None, so a caller may iterate unconditionally.

        HONEST LIMIT ON WHAT THIS BUYS TODAY, so a reader is not misled by the accessor's
        existence. Two things it does NOT do. (1) It is invisible to `tools/p6_gate.py`: that gate
        matches an explicit `(method, arity)` table which has `("broadcast", 0)` and does not have
        this name, so the tree's P6 read count does NOT move because this exists. Adding it there
        is work in a file this change does not own. (2) On the live arm bus it currently returns
        one content, because only one producer is admissible: measured over 60 live `AutonomousArm`
        ticks, `perception` posts in [0.5798, 0.6573] every tick and `salience`/threat posts a
        structural 0.0 (the amygdala is never paired on the wake path -- `autonomous_loop`
        documents and proves that null in both directions). K slots over one admissible candidate
        broadcast one content. The capacity is not the binding constraint on that bus today; the
        producer population is, and fixing that is not this module's work."""
        return tuple(entry[1] for entry in self._slot())

    def band_claim(self, signal, baseline):
        """THE ONE derivation of a compliant claim from a producer's own running signal.

        WHY THIS LIVES HERE AND IS SHARED. The admissible band is a property of THIS BUS -- its
        floor is the rheobase below which nothing is audible, and its ceiling is `1 - (self_boost +
        hysteresis)`, above which a submission is no longer displaceable by a compliant rival. Two
        producers that map their signals into that band by DIFFERENT rules are not competing in the
        same coordinate system, and the louder rule wins every tick regardless of what either
        producer actually measured. That is not competition; it is a fixed ordering wearing
        competition's clothes.

        MEASURED, and it is why this exists (2026-08-06, 120 live ticks). `brainstem/
        autonomous_loop.py::_percept_salience` mapped deviance through `lo + (hi-lo)*ratio` and
        occupied [0.568, 0.667]. `cortex/brain.py::_update_access` mapped novelty through
        `rheobase + 0.5*(nov - nov_base) + 0.15*real` clamped to [0,1] -- TWO HAND-PICKED GAINS, and
        NOT BOUNDED BY THE BAND AT ALL, so it rode its separate cap. The faculty therefore sat at or
        above perception's maximum on essentially every tick: repointed onto the shared bus it held
        the workspace 80 of 120 ticks and pushed perception out of it on 79, taking
        `live_brain.thought` from naming the percept 120/120 down to 41/120. Occupancy stayed 1 of 4
        because one producer was structurally always louder, not because the capacity was too small.

        THE FORM. `lo + (hi - lo) * signal / (signal + baseline)`, where `baseline` is the
        producer's OWN running mean of its OWN signal. It introduces NO constant (P1): the floor and
        ceiling are read off this bus, and the ratio is self-normalising, so a producer whose signal
        sits at its running baseline claims the MIDPOINT of the band and one at twice its baseline
        claims two thirds of the way up. What a claim means is now the same sentence for every
        producer, which is the property, and the only thing a producer supplies is a signal and its
        own history.

        THIS IS NOT A CLAIM THAT THE BAND IS OCCUPIED EVENLY. A right-skewed signal sits below its
        own running mean most ticks, so a producer's claims concentrate in the LOWER half. That is a
        property of the signal, not a defect of this map, and it is the same for every producer --
        which is exactly what makes them comparable. Widening a single producer's spread to "use the
        whole band" would restore the asymmetry this removes.

        DEGENERATE INPUTS, checked, because a P1 rewrite in this tree already shipped a
        `mu/1e-12 = 1e12` singularity onto a live path. Both non-positive (an unvarying stream from
        the first tick) -> the denominator is not formed at all and the RHEOBASE is returned, which
        is the honest reading: no signal, no claim above the floor. SIGNAL zero with a positive
        baseline -> ratio 0 -> exactly `lo`, audible but minimal. BASELINE zero with a positive
        signal -> ratio 1 -> exactly `hi`, the loudest compliant claim, never above it. A bus
        configured with no room between audible and displaceable (`hi <= lo`) -> `lo`.

        THE CEILING IS STRUCTURAL, and that is the half the previous faculty form did not have: the
        return is bounded ABOVE by `hi` by construction, so a producer using this cannot make an
        undisplaceable claim even if its signal explodes. A `min()` after the fact would be a second
        mechanism; this is one."""
        lo, hi = self._band()
        if hi <= lo:
            return lo
        r = self._band_ratio(signal, baseline)
        if r is None:
            return lo
        return lo + (hi - lo) * r

    def _band(self):
        """The admissible band, read off THIS bus. `lo` is the rheobase below which nothing is
        audible; `hi` is `1 - (self_boost + hysteresis)`, above which a submission is no longer
        displaceable by a compliant rival. ONE definition, shared by every mapping below."""
        return float(self.ignition_threshold), 1.0 - (
            float(self.self_boost) + float(self.hysteresis)
        )

    @staticmethod
    def next_baseline(previous, sample, n):
        """THE ONE rule for "a producer's own running baseline": the running MEAN, `prev + (x -
        prev)/n`. Parameter-free.

        WHY THIS IS SHARED AND WHY IT IS THE FINISH OF THE BAND WORK. Mapping two producers into one
        band through one derivation is NOT enough to make them comparable, and this tree measured
        that the expensive way: after both were routed through the same map, the repoint outcome was
        BYTE-IDENTICAL (perception in the workspace 41/120, coalition 80/120, co-occupancy 1/120).
        The ratio each producer feeds the map is `signal / (signal + baseline)`, so IF THEY COMPUTE
        `baseline` DIFFERENTLY THE RATIOS ARE NOT ON ONE SCALE and the shared map inherits the
        difference intact.

        They did. `brainstem/autonomous_loop.py::_percept_salience` used a true running mean
        (`base += (dev - base)/n`); `cortex/brain.py`'s novelty baseline is an EMA with a fixed
        `_base_tau`. AN EMA LAGS A RISING SIGNAL, so its ratio sits persistently ABOVE 0.5; a
        converging running mean sits BELOW 0.5 by whatever the signal's skew is. One producer was
        therefore structurally louder than the other no matter what map they shared.

        WHY THE RUNNING MEAN AND NOT THE EMA: the EMA carries a time constant, and a time constant is
        a hand-picked number that arbitrates (P1). The running mean has none -- it is the
        arithmetic mean of everything seen, computed incrementally, and its only input is the count.

        THIS DOES NOT REPLACE EVERY BASELINE IN THE TREE, and that restraint is deliberate. The
        novelty EMA at `cortex/brain.py:1777` has a SECOND consumer -- the explore/fidget decision --
        which asks "is this more novel than RECENTLY?", a question a lagging estimator answers
        correctly. Access asks "is this more novel than TYPICALLY?". Those are different questions
        and they legitimately want different estimators; what is NOT legitimate is answering the
        second with the first's estimator by accident, which is what was happening. So the EMA stays
        where it belongs and access grows its own baseline under this rule.

        DEGENERATE INPUTS: n <= 0 returns the sample itself (the first observation IS its own
        baseline, with no division). n == 1 returns the sample exactly, so a single-sample baseline
        gives ratio 0.5 and a claim at the band's midpoint -- the honest reading of one observation.
        """
        k = int(n)
        if k <= 1:
            return float(sample)
        p = float(previous)
        return p + (float(sample) - p) / float(k)

    @staticmethod
    def _band_ratio(signal, baseline):
        """`signal / (signal + baseline)` in [0,1], or None when neither is positive. Self-
        normalising, so it needs no constant and no scale: at its own running baseline a signal
        gives exactly 0.5."""
        s = float(signal)
        b = float(baseline)
        total = s + b
        if total <= 0.0:
            return None
        return s / total

    def band_bid(self, signal, baseline):
        """A claim from a producer WHOSE ACCESS MAY FAIL -- centred on the rheobase, not floored at
        it. Same band, same ratio, different question, and the difference is the whole point.

        WHY THIS IS SEPARATE FROM `band_claim`, and why it is NOT the flag P2 forbids. These map two
        genuinely different quantities. A PERCEPT is always present: perception posts on every wake
        tick and the only question is how loud, so flooring it at the rheobase is correct and it
        should never fall silent. An ACCESS BID is a request that may be REFUSED: a stimulus no more
        novel than the brain has habituated to SHOULD fail to reach access, which is repetition
        suppression and is the mechanism `_update_access` exists to implement. One band, two
        mappings, each named for the quantity it maps -- not one mapping behind a switch.

        MEASURED, and this method exists because the first attempt got it wrong. Mapping the access
        bid through `band_claim` (floored) took the faculty from `ignited 96/120` to `120/120` and
        `_access_ema` from 0.287282 back to 0.488958 -- IT COULD NO LONGER LAPSE AT ALL, because a
        floored map is bounded below by the rheobase by construction. "Access can fail" had been
        reported as a gain one commit earlier and a floored map silently deletes it.

        THE FORM. `lo + (hi - lo) * (2r - 1)`, r the same self-normalising ratio. At the running
        baseline (r = 0.5) it returns EXACTLY the rheobase, which is the anchor `_update_access`'s
        own comment has always described: at baseline novelty, ignition reduces to the gate boundary.
        Above baseline it rises toward `hi` and is bounded there -- so a bid cannot become
        undisplaceable however large the signal. Below baseline it falls BELOW the rheobase and the
        bid LAPSES, which is the property. It is deliberately NOT clamped at the bottom: clamping
        would restore the defect this method was written to fix.

        DEGENERATE INPUTS: neither positive -> the rheobase (no signal, no bid above the floor, and
        no lapse either -- an unvarying stream is not evidence of habituation). Signal zero with a
        positive baseline -> `lo - (hi - lo)`, the deepest lapse. Baseline zero with a positive
        signal -> exactly `hi`. `hi <= lo` -> `lo`."""
        lo, hi = self._band()
        if hi <= lo:
            return lo
        r = self._band_ratio(signal, baseline)
        if r is None:
            return lo
        return lo + (hi - lo) * (2.0 * r - 1.0)

    @property
    def broadcast_hold(self):
        """HOW STRONGLY the workspace is still holding the content `broadcast()` names: that
        pool's cross-tick attractor charge, in [0,1]. 0.0 when the workspace is empty.

        WHAT THIS VALUE IS, named for its meaning rather than for where it sat. It is `self._g` for
        the winning pool -- the same attractor memory that builds that pool's drive current on the
        next tick. It RISES to the salience of a fresh submission of that content and DECAYS by
        `decay` on every tick the content is not re-posted, so it is the workspace's own running,
        refresh-relative measure of how loudly and how RECENTLY the broadcast content was asserted.
        It is not a parameter and not a constant: it is a live trace of live submissions, which is
        why exposing it needs no threshold, no scale factor and no hand-picked number of its own
        (P1 -- the accessor introduces no arbitrating value, and reads one that is already a running
        form).

        WHY IT NEEDED A ROUTE. Measured on the LIVE brain over 150 wake ticks, before this accessor
        existed: `broadcast()` returned '0' on EVERY tick -- 1 distinct value, 0 changes -- while
        this quantity cycled 0.6 -> 0.51 -> 0.4335 -- 3 distinct values, 149 changes -- because the
        two submitters post literal constant saliences and the incumbent's hysteresis holds the same
        content while its charge decays and refreshes underneath. Same slot, same tick: WHAT was
        pinned and HOW STRONGLY was the only thing on the bus that moved, and it was private with no
        declared accessor, so no faculty could consume it.

        WHY THIS IS NOT A SECOND READ PATH (P2, one mechanism). There is one bus, one arbiter, one
        `_slot()` load behind all four accessors, and no second copy of the content route:
        `current` / `broadcast()` remain THE answer to WHAT, `broadcast_all()` is the ONLY answer
        to WHAT ELSE, and this is the ONLY answer to HOW STRONGLY. Folding them into one return
        value instead -- `broadcast()` yielding a (content, hold) pair -- was rejected because it
        changes the return type of the tree's one cross-module workspace read
        (`cortex/live_brain.py::thought`) and of every `is not None` test written against it, which
        buys no signal and breaks the consumers this exists to serve.

        IT REPORTS THE TOP SLOT, and that is a deliberate narrowing rather than an oversight. A
        per-slot hold vector would be a fourth accessor answering a question nothing in this tree
        asks; `_slot()` already exposes the triples for anything that later needs them.

        HONEST LIMIT, MEASURED rather than inherited, so a consumer is not misled by its own read.
        On the live brain this spans 0.4335..0.6. Sweeping `cortex/access_consciousness.py` at
        constant salience 0.00..1.00 in steps of 0.01, its `ignited` flips exactly once, at 0.54 --
        which the live range straddles. It still cannot move it: fed the live cycle
        0.6/0.51/0.4335 against a pinned 0.6 for 150 ticks, that consumer's outputs are IDENTICAL
        (`ignited` [True] / 0 changes in both; `access_index` 46 distinct / 149 changes in both, and
        equal element-wise -- its variation comes from its own coalition dynamics, not from the
        salience), because once a content has ignited the incumbent's `self_boost + hysteresis`
        defence keeps it ignited whether the drive sits above or below rheobase. So: this quantity
        varies, it is now readable, and straddling a consumer's threshold is NOT the same as moving
        it. Making it decisive somewhere is that consumer's work, not this accessor's, and claiming
        it here would be the force-wire P6 forbids.

        In the `use_ignition=False` control there is no attractor at all, so the slot's third
        element is the winning SALIENCE, clamped in `step()` to this same [0,1] -- the same question,
        and the only answer that path can give."""
        slots = self._slot()
        return slots[0][2] if slots else 0.0

    # -- one workspace cycle --------------------------------------------------------------------
    def _deliver(self, contents):
        """BROADCAST: hand every subscriber the WHOLE admitted set, ranked, as one tuple.

        WHY THE SET AND NOT THE HEAD (2026-08-06, and this is a CONTRACT CHANGE, not a widening).
        This method used to pass `self.current`. Under one slot that was the same object as the
        workspace, so nothing distinguished "what is in the workspace" from "what is on top of it".
        Multi-slot separated them and the delivery channel kept carrying only the top, which made
        every subscriber's membership question unanswerable: a faculty admitted beside a louder peer
        received that peer's content and could only conclude that its own had not got in. MEASURED
        on this bus AND ON THE ARBITER AS IT THEN STOOD, two contents at identical salience 0.65 and
        capacity 4, varying NOTHING but the peer's source name: the second content is in
        `broadcast_all()` on 12 of 12 ticks under every name, while a head-only subscriber saw it on
        0 of 12 with a peer named 'aaa' and 12 of 12 with a peer named 'mmm' -- because a
        `late`/`total` tie WAS broken lexicographically on the source key at the rank line in
        `step()`. A subscriber's view of its own global availability was being settled by a string
        comparison. That rank-line defect is now closed too (see the header's tie-break block); this
        change is the independent half, and it is the half that stops a subscriber having to infer
        membership from a rank it cannot see at all.

        It is also what the theory says. GWT's claim is that the workspace BROADCASTS ITS CONTENTS,
        and global availability IS that broadcast; a bottleneck that admits four contents and
        announces one has a capacity of four and an availability of one.

        ONE MECHANISM, NOT TWO (P2). No second `subscribe` channel exists and none may be added:
        there is exactly one delivery, carrying exactly the arbiter's full output. `current` /
        `broadcast()` / `broadcast_hold` are unchanged and remain projections of that same
        `_slot()` load -- `broadcast()` is still the tree's one cross-module read
        (`cortex/live_brain.py::thought`), still zero-arg, still returning ONE content, and
        `tools/p6_gate.py` still classifies it by arity. The head did not lose a route; the
        subscribers gained the rest of the workspace on the route they already had.

        DETERMINISM IS PRESERVED BY CONSTRUCTION, and this is the half that had to be checked
        rather than assumed. `broadcast_all()` builds a FRESH tuple of the contents out of one
        `_slot()` load, so a subscriber receives no handle on `_slots` at all -- it cannot observe a
        later rebinding and cannot mutate what it was given. That is strictly safer than the old
        contract in one respect nobody had to think about before: handing out a mutable container
        (a list of slots) would have made `_slot()`'s "either the whole previous workspace or the
        whole new one" false from outside the class. The self-check asserts the delivered object is
        hashable-all-the-way-down and equal to `broadcast_all()` on the same cycle.

        EMPTY / SILENCE deliver `()`, never None and never a stale set -- the same contract
        `broadcast_all()` already documents, so a subscriber may iterate or use `in`
        unconditionally on every tick including the ones where the workspace released."""
        for fn in self._subscribers:
            fn(contents)

    def _top_candidate(self):
        # Highest salience wins; deterministic tie-break by source name (lexicographically first).
        if not self._buffer:
            return None
        return min(self._buffer, key=lambda c: (-c[2], c[0]))

    def _jitter_for(self, key):
        # A fixed, tiny per-pool current drawn once (deterministic under the seed) -- breaks the
        # symmetry of exactly-equal drives so the shared-inhibition WTA settles on ONE winner.
        if key not in self._jit:
            self._jit[key] = self._rng.uniform(0.0, self._JITTER)
        return self._jit[key]

    def _settle_chunk(self, mkeys, drives, sync):
        """Integrate ONE contiguous chunk of content pools across the whole settling window.

        This is the ONLY definition of the pool arithmetic -- the serial and the parallel paths both
        run exactly this function, differing only in how many chunks there are and in what `sync`
        does (P2: one mechanism, not two implementations to keep in agreement).

        The chunk owns `v/trace/si/total/late/sp` EXCLUSIVELY: no other chunk reads or writes them,
        so a thread running this touches no container another thread touches. The only shared value
        is the scalar `a_inh`, and it is read-only within a micro-step -- which is the whole reason a
        timestep of an SNN parallelises. `sync(si, t)` publishes this chunk's inhibitory
        contributions and returns the shared inhibitory-pool activity for the next micro-step.

        Returns (mkeys, total, late, v) -- integer spike counts and the final float membranes,
        positionally aligned. `v` is returned because the integer counts are TOO COARSE to police
        this module's determinism: a reassociated reduction changes `a_inh` in the last bits and the
        spike counts absorb it, so the self-check compares the float trajectory instead."""
        m = len(mkeys)
        dr = [drives[k] for k in mkeys]
        v = [0.0] * m  # membrane potentials
        trace = [0.0] * m  # within-tick self-excitation traces
        si = [0.0] * m  # each pool's own contribution to the inh pool
        sp = [0] * m  # this micro-step's spikes
        total = [0] * m
        late = [0] * m
        half = self._T // 2
        glat = self._g_lat  # this workspace's PER-SLOT share of the lateral budget (see `capacity`)
        free = self._g_free  # the inhibitory pool's rheobase (see `_inh_free`); hoisted like _g_lat
        a_inh = sync(si, -1)  # register this chunk; shared pool starts silent
        for t in range(self._T):
            for i in range(m):
                lateral = glat * max(
                    0.0, (a_inh - si[i]) - free
                )  # OTHER pools' drive past the inh rheobase
                cur = dr[i] + self._W_SELF * trace[i] - lateral  # LIF input current
                v[i] = v[i] + (cur - self._LEAK * v[i])  # leaky integration
                if v[i] < 0.0:
                    v[i] = 0.0
                s = 0
                if v[i] >= self._v_th:  # DISCRETE SPIKE
                    v[i] -= self._v_th  # subtract-threshold reset
                    s = 1
                    total[i] += 1
                    if t >= half:
                        late[i] += 1
                    trace[i] += 1.0  # self-excitation kick
                sp[i] = s
                trace[i] *= self._TAU_SELF
            for i in range(m):  # integrate the inhibitory pools
                si[i] += self._K_INH * sp[i] - self._LEAK_INH * si[i]
                if si[i] < 0.0:
                    si[i] = 0.0
            a_inh = sync(si, t)  # shared pool == sum of contributions
        return mkeys, total, late, v

    def _simulate(self, drives):
        """Settle the LIF network for `_T` micro-steps and return the sustained-firing readout.

        Each content pool k has a membrane V_k that integrates: its external drive current, its
        own recurrent self-excitation (within-tick trace), MINUS the shared inhibitory pool's
        feedback. The shared inhibitory pool `a_inh` sums EVERY excitatory spike; each content pool
        `self_k` tracks only its OWN contribution to it, so the inhibition actually applied to pool
        k is `g_lat * max(0, (a_inh - self_k) - _inh_free)` -- the drive from every OTHER pool,
        PAST the pool's rheobase allowance of `capacity` slots' worth of saturation. That is the
        global lateral inhibition (each pool inhibits the others through the shared pool): at or
        below `capacity` competitors the pool is silent and every supra-rheobase claimant ignites,
        while past it the strongest pools' spikes suppress the excess to silence.

        `g_lat` is `_lateral_gain` -- one unit of inhibition per slot against one unit of recurrent
        self-excitation, E/I balanced. It read `_G_INH / capacity` until that typed total was shown
        to be `_W_SELF * capacity` exactly and retired; the per-slot value is unchanged, so a wider
        workspace still competes less fiercely in TOTAL while each slot stays balanced. See the
        `_lateral_gain` docstring for the identity, and the module header
        for the measurement, and note that it is a property of the WORKSPACE, not of the load. It
        was also measured the other way: making the divisor the number of competitors instead made
        the gain depend on how many pools happened to be posting, and winnability went 0.72 -> None
        because the same workspace competed at gain 2.4 while warming one tenant and 1.2 on the
        tick a challenger arrived.

        PARALLEL ABOVE A MEASURED CROSSOVER. The pools partition into contiguous chunks of the sorted
        key order; each chunk is a `_settle_chunk` on its own thread. The shared inhibitory pool is a
        genuine global reduction every micro-step, so it is NOT parallelised -- one designated thread
        folds the per-chunk contributions in ascending chunk order, which visits exactly the elements
        `sum(self_i.values())` visited in exactly that order. Float addition is not associative and
        that is why the reduction stays serial: a tree-reduction here would change the numbers.

        The crossover is not written down anywhere. `region_wiring.pick_thread_count` compares this
        instance's running-EMA cost per pool against this machine's measured barrier and dispatch
        floors and the efficiency this kernel has actually achieved. Measured on a 28-core
        free-threaded 3.14.6 box: 5 and 50 pools stay serial, 500 pools 1.16-1.22x on 2 threads,
        2,000 pools 2.0x on 4, 10,000 pools 2.1-2.4x on 8, 50,000 pools 2.1-3.0x on 16. Spike counts
        AND membrane floats identical at every width and at every thread count.

        200 pools is the honest edge case and this row used to read "0.98x (gate: serial)", which is
        true only of a WARM gate. Re-measured: the FIRST call at 200 pools picks 2 threads and runs
        0.81-0.87x -- a real loss -- after which `realized_efficiency` records ~0.69 and every
        later call at that width returns to serial and stays there. So the crossover is not a clean
        step: within roughly a factor of two below it the gate pays for exactly one mispriced call
        before its own feedback corrects it. That is the designed behaviour of a measured gate
        rather than a defect, but "gate: serial" stated flatly hid a cost that is real and
        reproducible, and a cold process pays it on every fresh instance.

        The LIVE brain runs about 5 candidate pools, ~100x below the crossover, so the gate keeps
        production on the serial path today -- correctly. What would move it is P6: this tree holds
        dozens of workspace instantiation sites, most of them private and mutually invisible, and
        consolidating them onto one shared bus is what puts hundreds of candidates in one
        competition. Re-derive that census with `tools/p6_gate.py` rather than quoting it -- the
        figures this paragraph used to carry (26 sites, 11 private) were already stale when they
        were written.

        Returns {key: total_spikes}, {key: second_half_spikes} -- discrete integer spike counts."""
        keys = sorted(drives)  # deterministic evaluation order
        n = len(keys)
        if n == 0:
            return {}, {}

        from snn_data_lake.substrate import region_wiring as _rw

        nt = 1
        if n >= 2 and self._sim_per_pool_s > 0.0:
            # dedicated threads, not the shared pool: this kernel synchronises on a barrier every
            # micro-step, and a barrier across pool tasks deadlocks if the pool is saturated.
            # `sync_rounds` is the real barrier count -- 2 per micro-step (publish, then reduce),
            # plus the registration round before the loop. Passing it is what stops the gate from
            # over-threading a lock-step kernel; see pick_thread_count for the measured consequence.
            nt = min(
                _rw.pick_thread_count(
                    n * self._sim_per_pool_s,
                    mode="threads",
                    sync_rounds=2 * (self._T + 1),
                    efficiency=self._sim_par_eff,
                ),
                n,
            )

        if nt < 2:
            t0 = time.perf_counter()

            def sync_serial(si, _t):
                return _fold_parts((si,))  # the identical left fold the parallel lead performs

            _, total, late, v = self._settle_chunk(keys, drives, sync_serial)
            dt = (time.perf_counter() - t0) / n
            self._sim_per_pool_s = (
                dt
                if self._sim_per_pool_s <= 0.0
                else 0.75 * self._sim_per_pool_s + 0.25 * dt  # running baseline (P1)
            )
            self._last_v = dict(zip(keys, v))
            return dict(zip(keys, total)), dict(zip(keys, late))

        est = n * self._sim_per_pool_s
        t0 = time.perf_counter()
        out = self._simulate_parallel(keys, drives, nt, _rw)
        # Feed the ACHIEVED scaling back into the gate. A parallel run that underperformed lowers the
        # efficiency estimate and the next call at this width picks fewer threads -- or none.
        eff = _rw.realized_efficiency(
            est, time.perf_counter() - t0, nt, mode="threads", sync_rounds=2 * (self._T + 1)
        )
        self._sim_par_eff = 0.5 * self._sim_par_eff + 0.5 * eff
        return out

    def _simulate_parallel(self, keys, drives, nt, _rw):
        """Run `_settle_chunk` on `nt` dedicated threads over a contiguous partition of `keys`.

        DETERMINISM, stated as the argument rather than as a hope: (1) every chunk's mutable state is
        its own list, so no container is written by two threads; (2) the only cross-chunk value is the
        scalar `a_inh`, published behind a barrier and therefore never read while it is being written;
        (3) the reduction that produces it is a LEFT FOLD over the chunks in ascending index order,
        and the partition is contiguous, so the addition order is exactly the serial one; (4) results
        are collected into per-chunk slots and merged in ascending chunk order, so no dict is built
        from completion order. Nothing here depends on which thread finishes first.

        FAILURE IS NOT AN OPTION THIS KERNEL CAN IGNORE. Every chunk meets every other at two
        barriers per micro-step, so a chunk that raises stops arriving and its partners wait on a
        barrier that can NEVER again reach `nb` parties. Measured before the guard below existed: a
        single non-numeric drive value hung `_simulate_parallel` FOREVER -- not a slowdown, a dead
        tick, and because the worker threads are non-daemon the interpreter could not exit either.
        Each chunk therefore runs under a handler that records its exception and ABORTS both
        barriers, which converts every partner's permanent wait into a `BrokenBarrierError`; the
        caller then re-raises the ROOT cause. The serial path cannot express this failure at all,
        so it is a defect the parallel path introduced and must clean up after itself."""
        bounds = _rw.blocks(len(keys), nt)
        nb = len(bounds)
        parts = [None] * nb  # chunk index -> that chunk's si list (each written by ONE thread)
        out = [None] * nb
        shared = [0.0]
        b_reduced = threading.Barrier(nb)
        b_published = threading.Barrier(nb)

        def make_sync(tid, lead):
            def sync(si, _t):
                parts[tid] = si  # same list object every round; publish once, mutate in place
                b_published.wait()  # every chunk's si is now written for this micro-step
                if lead:
                    shared[0] = _fold_parts(
                        parts
                    )  # ONE fold definition, shared with the serial path
                b_reduced.wait()  # the fold is complete; a_inh is safe to read
                return shared[0]

            return sync

        errors = [None] * nb

        def work(tid, a, b):
            try:
                out[tid] = self._settle_chunk(keys[a:b], drives, make_sync(tid, tid == 0))
            except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised by the caller
                errors[tid] = exc
                # Release every partner stuck at a barrier this chunk will never reach again.
                # `abort()` is idempotent, so several failing chunks racing here is harmless.
                b_published.abort()
                b_reduced.abort()

        threads = [
            threading.Thread(target=work, args=(i, a, b), name=f"gw-pool-{i}")
            for i, (a, b) in enumerate(bounds)
            if i > 0
        ]
        for th in threads:
            th.start()
        work(0, bounds[0][0], bounds[0][1])  # the caller runs chunk 0 and leads the reduction
        for th in threads:
            th.join()

        # Re-raise the ROOT cause in preference to the BrokenBarrierError it induced in the others,
        # so the traceback names the real defect rather than the unblocking mechanism. Scanning in
        # ascending chunk order keeps the choice deterministic when several chunks fail.
        root = None
        for exc in errors:
            if exc is not None and not isinstance(exc, threading.BrokenBarrierError):
                root = exc
                break
        if root is None:
            for exc in errors:
                if exc is not None:
                    root = exc
                    break
        if root is not None:
            raise root

        total, late, last_v = {}, {}, {}
        for mkeys, tot, lat, v in out:  # ascending chunk order -> deterministic key order
            for i, k in enumerate(mkeys):
                total[k] = tot[i]
                late[k] = lat[i]
                last_v[k] = v[i]
        self._last_v = last_v
        return total, late

    def _rank_key(self, k):
        """THE competition rank order, and the ONLY place this workspace decides who out-ranks whom.

        Read after `_simulate` has run, so every term is that settle's own output: sustained
        (second-half) spikes first, total spikes next, then the FINAL MEMBRANE, then the pool key.

        WHY THE MEMBRANE IS IN HERE, and it is the whole reason this method exists as a method. The
        two spike counts SATURATE -- a pool in the high-firing attractor fires on every micro-step of
        the second half -- so co-saturating pools tie on both and the key used to fall straight
        through to `k`, which is a lexicographic compare of the SOURCE NAME. That settled 1,941 of
        3,576 co-admissions on the exhaustive two-producer grid, and because `step()` admits
        `fired[:cap]` and `_defended` subsidises every admitted occupant on the NEXT tick, the name
        was deciding ACCESS and not merely display order. `_last_v` is the same settle's residual
        membrane: after the subtract-threshold reset a pool firing every micro-step carries
        `2 * (cur - v_th)`, so the float holds LINEARLY the drive excess the clipped integer count
        threw away. Name-flips over that grid go 1,941 -> 0 and the head goes to the higher-drive
        pool on 3,400 of 3,400 configs where one exists.

        IT IS A METHOD SO THE RETIRED KEY CAN BE RECONSTRUCTED AS A CONTROL, not so the key can be
        configured (P2 -- there is exactly one rank order on every live path, no flag and no second
        setting). `main()` subclasses this to restore `(-late, -total, k)` and RUNS it as the red for
        the rename criterion, exactly as `_SingleSlotControl` restores the retired capacity.

        `k` stays as the final term: it costs nothing, it is unreachable whenever the membranes
        differ, and it keeps the order total if a future readout ever ties on all three.

        THIS ORDERS ADMISSION ONLY. The head is ordered by `_focus_key` over the set this one
        admits -- see there, and see `grip`, for the defect that split them."""
        return (-self._last_late[k], -self._last_total[k], -self._last_v[k], k)

    def _focus_key(self, k):
        """THE order of the ADMITTED SET -- who is the head, i.e. what the brain is attending to.

        `_rank_key` above decides ACCESS and carries the full HOLD; this decides FOCUS and carries
        only the `grip`. Everything about why is in `grip`'s docstring; what matters here is that
        the two are separate methods over the same settle, so there is one competition and two
        readouts of it (P2 -- no second arbiter, no second simulation).

        The terms, in order. The pool's own AFFERENT CLAIM plus its grip if it holds a slot, which
        is the quantity this key exists to rank on. Then the settled membrane `-_last_v`, which
        carries the HOLD and so breaks an exact claim tie in the incumbent's favour -- hysteresis
        where the afferent evidence is genuinely equal, which is the one place it belongs. Then
        `k`, unreachable whenever anything above it differs, kept so the order stays total.

        NAME-BLINDNESS IS INHERITED, NOT RE-EARNED, and that is worth stating because this key is
        new and the name-decided arbiter is the defect this module most recently removed: `k` sits
        below two continuous floats here exactly as it does above, and `main()` runs the rename
        sweep through the head accessor, which reads THIS key."""
        return (-self._last_focus[k], -self._last_v[k], k)

    def _defended(self, submitted):
        """The slot occupants that still carry the tenancy defence this tick.

        A TENANCY IS HELD BY A SOURCE, NOT BY A (source, content) STRING PAIR -- and that one
        distinction is the whole of this method. `self_boost + hysteresis` is a SPECIALIST'S grip on
        a slot: it exists so that a content which has already ignited is not knocked out by a
        marginally louder RIVAL, which is the flicker the argmax control demonstrates and this
        workspace exists to beat. It was never meant to defend a content against ITS OWN AUTHOR. A
        specialist that posts a DIFFERENT content this tick has WITHDRAWN the one it was posting;
        the old content is not a rival's claim being out-shouted, it is a RETRACTED claim, and
        subsidising it against the fresh claim from the same source is the workspace propping up
        something nobody is asserting any more.

        So the defence is withheld from a held `(src, content)` exactly when `src` is posting this
        tick and is NOT posting that content. Three cases, and the two that are not the defect are
        deliberately untouched:

          * `(src, content)` IS submitted -> the source re-asserts its own tenancy -> DEFENDED.
            Every winnability figure this module publishes is measured on a tenant that re-posts the
            same content every tick, so this branch is the one they all run through and none of them
            moves. Re-run after the change: the challenger still wins a slot at a flat 0.720 against
            tenants at 0.550/0.5797/0.6573/0.700/0.750, which is the row the module header prints.
            (The HEAD row read 0.720/0.825/0.825/0.825/0.825 across that change too; it moved later,
            when the RANK KEY was repaired, and that is a different mechanism -- see `_rank_key`.)
          * `src` submitted NOTHING this tick -> silent, not superseding -> DEFENDED. This is the
            hysteresis-across-a-quiet-tick that "ignited content is HELD across ticks, not rebuilt
            from scratch each tick" names, and the persist/release criterion measures it (7 held
            ticks then release). An EMPTY tick supersedes nothing at all, so a workspace with no
            submissions behaves exactly as before.
          * `src` submitted something ELSE -> superseded -> NOT DEFENDED. The retracted content
            keeps its attractor charge `_g` and keeps competing on it; it loses only the subsidy.
            It is still free to win on its own charge, which is what makes this a narrowing of who
            pays for a slot rather than an eviction rule.

        THE MEASUREMENT THAT FORCED IT, on `make_brain(seed=0, refit_every=10, persist=False)` with
        `clock.phase = 0.25`, 120 live ticks. `cortex/live_brain.py::thought` -- the tree's ONE
        genuine cross-module workspace read -- carried the int 0 on 120 of 120 ticks while
        `perception` submitted FOUR distinct contents at 56 DISTINCT saliences. The final pool state
        is the proof that this was entrenchment and not a quiet bus: `_g[('perception', 3)]` was
        0.621394 against the incumbent `_g[('perception', 0)]` at 0.544709 -- THE FRESH CONTENT HELD
        MORE ATTRACTOR CHARGE THAN THE ONE OCCUPYING THE WORKSPACE -- and the fresh pool still fired
        0 sustained spikes against the incumbent's 30, because +0.25 of tenancy defence took the
        incumbent to a drive the challenger could not survive the shared inhibition against.
        Workspace occupancy was 1 of 4 on 120 of 120 ticks with the other three slots never used.

        Note the shape rather than the numbers, because the numbers are the perishable half: the
        content key ROTATES, so a fresh percept always starts a NEW pool at its own salience while
        the entrenched pool draws a defence and decays only `decay` per tick between refreshes. This
        is the SAME defect class as the one the defence-exceeds-band fix closed for equal compliant
        PEERS, one level up -- there the winner was decided by a subsidy no challenger could match,
        here by a subsidy paid to a claim that had already been withdrawn.

        DEGENERATE INPUTS. ZERO/EMPTY submissions -> `active` is empty, no source supersedes, every
        occupant keeps its defence (unchanged behaviour). SILENCE (a source posting only
        sub-rheobase claims) still counts as posting: an audible-or-not submission is still the
        source stating what it is currently asserting, and `_g` already decides audibility. SINGLE
        occupant re-posting itself -> defended, as above. An EMPTY workspace -> the comprehension is
        over `self._slots`, so it returns an empty set without touching `submitted`."""
        active = {src for src, _content in submitted}
        return {
            (src, content)
            for src, content, _hold in self._slots
            if (src, content) in submitted or src not in active
        }

    def step(self):
        """Run ONE workspace cycle. Returns the TOP broadcast content (or None), clears the buffer.

        `broadcast_all()` returns everything admitted this cycle; this returns its head, because
        that is the signature every existing consumer is written against. SUBSCRIBERS RECEIVE THE
        SET, not this return value -- one delivery, carrying the arbiter's full output (`_deliver`).
        The two are deliberately different: a caller holding the workspace asks it what the FOCUS
        is, a subscriber is told what the workspace CONTAINS."""
        self._drain_slots()  # concurrent specialists join the competition in a FIXED slot order
        if not self.use_ignition:
            # FAIR CONTROL: pure per-tick argmax. Whatever is loudest THIS tick wins outright, and
            # exactly one content is held -- no spiking, no attractor, no inhibition, so there is
            # no competition for a second slot to arbitrate. That is what makes it the control.
            top = self._top_candidate()
            # A slot's third element is what `broadcast_hold` reads, and on the ignition path it is
            # `_g` -- already clamped to [0,1]. `submit()` accepts any float, so the control path
            # would otherwise hand a reader a raw 5.0 under a [0,1] contract. Clamping HERE keeps
            # ONE meaning for one accessor instead of a caveat per path, and it is behaviour-neutral:
            # nothing else in this module reads a slot's third element (the drives are built from
            # `self._g`), which the self-check checks rather than asserts.
            #
            # An empty tick EMPTIES the workspace on this path too. It previously left the previous
            # winner in place, so the two paths had different emptying semantics behind one
            # accessor; nothing depended on that (the control's stream posts every region every
            # tick, and `sw_without` is unchanged at 27), and one meaning per accessor is the point.
            self._slots = () if top is None else ((top[0], top[1], min(1.0, max(0.0, top[2]))),)
            content = self.current
            self._record_settle([] if top is None else [(top[0], top[1])])  # T031, pre-clear
            self._buffer = []
            self._deliver(self.broadcast_all())  # ONE delivery contract, both paths (see _deliver)
            return content

        # --- SPIKING Global-Workspace ignition dynamics ---
        # 1. Refresh / decay each pool's cross-tick attractor memory `g`. A submission this tick is
        #    a bottom-up refresh (g rises to the salience); an unrefreshed pool DECAYS (persistence
        #    fade) and, once negligible, is dropped.
        submitted = {}
        for src, content, sal in self._buffer:
            key = (src, content)
            if key not in submitted or sal > submitted[key]:
                submitted[key] = sal
        for key in list(self._g):
            if key in submitted:
                self._g[key] = min(1.0, max(self._g[key] * self.decay, submitted[key]))
            else:
                self._g[key] = self._g[key] * self.decay
        for key, sal in submitted.items():
            if key not in self._g:
                self._g[key] = min(1.0, max(0.0, sal))
        self._g = {k: v for k, v in self._g.items() if v > 1e-6}

        # 2. Set the competition's fierceness from the workspace's own capacity, then build each
        #    pool's external drive current: attractor memory + (for a pool ALREADY HOLDING A SLOT)
        #    its recurrent self-excitation defence (self_boost + hysteresis, the attractor hold) +
        #    a tiny deterministic symmetry-break jitter.
        #
        #    EVERY slot occupant carries the defence, not one privileged incumbent -- that is what
        #    makes this a multi-slot workspace rather than one slot with spectators, and it is why
        #    the capacity has to be paid for: the defence is the per-slot cost `capacity` divides
        #    the unit budget by, so granting it to C occupants is exactly the budget being spent.
        cap = self.capacity
        self._g_lat = self._lateral_gain  # read by _settle_chunk; set before any thread fans out
        self._g_free = self._inh_free  # read by _settle_chunk; set before any thread fans out
        held = self._defended(submitted)
        grip = self.grip
        drives = {}
        focus = {}
        for key, g in self._g.items():
            d = g
            r = g
            if key in held:
                d += self.self_boost + self.hysteresis  # the HOLD -- sustains the assembly
                r += grip  # the GRIP -- a far smaller advantage in the contest for DOMINANCE
            j = self._jitter_for(key)
            drives[key] = d + j
            focus[key] = r + j
        self._last_focus = focus

        # 3. Settle the spiking network; read out sustained (second-half) firing.
        total, late = self._simulate(drives)
        self._last_total = total
        self._last_late = late

        # 4. IGNITION -> ADMISSION. Every pool still firing in the second half has ignited (it
        #    survived the competition into the high-firing attractor); the workspace then admits
        #    them in competition rank order up to `capacity`. Nothing sustaining -> the workspace
        #    empties (release).
        #
        #    THE RANK KEY CARRIES A FINER READOUT THAN THE CLIPPED COUNTS (2026-08-06). `late` and
        #    `total` SATURATE -- a pool in the high-firing attractor fires on every micro-step of the
        #    second half -- so two co-saturating pools tie on both and the key used to fall through to
        #    `k`, A LEXICOGRAPHIC COMPARE OF THE SOURCE NAME. That decided ACCESS and not merely
        #    order, because `fired[:cap]` below picks who `_defended` subsidises on the NEXT tick.
        #    `_last_v` is the SAME settle's final membrane, already stored and already policed by the
        #    determinism check: a pool firing every micro-step settles to a residual of
        #    `2 * (cur - v_th)` after the subtract-threshold reset, so the membrane carries LINEARLY
        #    the drive excess the clipped integer count discards. It is read BEFORE `k`, so the counts
        #    still decide wherever they can and the name decides nothing that the physics can.
        #    Re-derive the census with the header's grid rather than quoting it.
        #
        #    TWO ORDERINGS, NOT ONE, and the split is the whole of defect #16. `_rank_key` decides
        #    ADMISSION and carries the full HOLD, so a sustained assembly keeps its slot exactly as
        #    before. `_focus_key` then orders the ADMITTED SET for the head and carries only the
        #    GRIP. See `grip` for why one subsidy cannot do both jobs.
        fired = sorted((k for k in late if late[k] > 0), key=self._rank_key)
        admitted = sorted(fired[:cap], key=self._focus_key)
        self._slots = tuple((k[0], k[1], self._g[k]) for k in admitted)

        content = self.current
        self._record_settle(admitted)  # T031: stamp + retain + publish, before the buffer clears
        self._buffer = []
        self._deliver(self.broadcast_all())  # BROADCAST the admitted SET to all subscribers
        return content


class OneSubsidyArbiter(GlobalWorkspace):
    """THE PRE-#16 ARBITER, kept as a running control: one subsidy doing both jobs.

    `_focus_key` is restored to `_rank_key`, so the HOLD that sustains an assembly also decides
    which assembly is broadcast. Nothing else changes -- same hold, same band, same capacity, same
    lateral gain -- so an A/B against this isolates the SPLIT and not the parameters.

    IT IS MODULE-SCOPE AND EXPORTED BECAUSE TWO SELF-CHECKS NEED IT (P2 -- one definition of "the
    retired arbiter", not one per consumer). `main()` here runs it as the reachable red for the
    focus-capture criterion; `cortex/access_consciousness.py` runs it as the red for claim (9),
    whose displaceability guarantee moved from the faculty's own claim cap to this split. A control
    that two criteria describe differently is two controls, which is how an A/B stops comparing
    what it says it compares."""

    def _focus_key(self, k):
        return self._rank_key(k)


# distinct-from: action_gating.py -- `kwta` there is the canonical top-K selection primitive for
#   the brain's selection circuits; this class selects nothing. It is an A/B CONTROL: the shipped
#   arbiter with its inhibitory rheobase pinned back at zero, i.e. the retired kernel.
class NoAllowanceArbiter(GlobalWorkspace):
    """THE PRE-RHEOBASE KERNEL, kept as a running control: lateral inhibition with no allowance.

    `_inh_free` is pinned at 0.0, so the shared inhibitory pool inhibits linearly from its first
    spike -- the retired kernel exactly, byte for byte. Nothing else changes (same gain, same
    band, same capacity, same hold), so an A/B against this isolates the rheobase and not the
    parameters.

    MODULE-SCOPE FOR THE SAME REASON AS `OneSubsidyArbiter` (P2 -- one definition of "the retired
    kernel", not one per consumer): the re-pointed spiking-competition criterion and the
    capacity-decides criterion in `main()` both run it as their reachable red, and a control that
    each consumer reconstructs differently is two controls."""

    @property
    def _inh_free(self):
        return 0.0


def build_stream(n_ticks, seed=0):
    """A fixed per-tick submission stream: a list of {region: salience} dicts.

    'perception' is a persistently strong (mostly supra-threshold) signal; the other regions are
    noisy near-threshold distractors that heavily overlap it -- so a pure argmax flickers among
    them. A scripted THREAT burst drives 'salience' (an amygdala-like threat detector) far above
    threshold for a short window, injecting a genuinely stronger signal mid-stream. Deterministic
    under `seed`.
    """
    rng = random.Random(seed)
    threat_window = set(range(30, 36))
    means = {"perception": 0.66, "memory": 0.54, "salience": 0.54, "language": 0.53, "motor": 0.52}
    sd = 0.11
    stream = []
    for t in range(n_ticks):
        tick = {}
        for r in REGIONS:
            s = rng.gauss(means[r], sd)
            if r == "salience" and t in threat_window:
                s = 0.98  # supra-threshold threat event
            tick[r] = min(1.0, max(0.0, s))
        stream.append(tick)
    return stream


def run_stream(stream, use_ignition, **params):
    """Feed a stream through a workspace tick by tick.

    Returns (heads, sets): the top broadcast content per tick, and the FULL workspace per tick.
    Both are needed because they answer different questions -- see `count_churn`."""
    ws = GlobalWorkspace(use_ignition=use_ignition, **params)
    heads, sets = [], []
    for tick in stream:
        for region, salience in tick.items():
            ws.submit(region, region, salience)  # content == region label
        heads.append(ws.step())
        sets.append(ws.broadcast_all())
    return heads, sets


def count_switches(seq):
    """Number of ticks whose value differs from the previous tick."""
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def count_churn(sets):
    """FLICKER, for a workspace that holds a SET: contents ENTERING or LEAVING, summed over ticks.

    THIS REPLACES `count_switches` ON THE HEAD AS THE STABILITY MEASURE, and the replacement is
    forced rather than convenient. Under one slot the two coincide exactly -- the head changing IS
    a content leaving. Under multiple slots they come apart, and measured on this module's own
    48-tick stream they come apart completely: the head changed on 10 ticks and the previous head
    was STILL IN THE WORKSPACE on 47 of 47 of them. Not one of those 10 was a content losing
    access; every one was a rank swap inside a stable set. So `count_switches(heads)` no longer
    measures flicker on this arbiter, and keeping it as the stability criterion would have meant
    loosening its bound from 6 to something that admits 10 -- which is the weakening P8 forbids
    ("if a criterion turns out not to measure the right thing it is deleted, not weakened").
    (That count read 17 while the head was name-decided; repairing `_rank_key` took it to 10 with
    churn, retention and the flood all unchanged, which is the same finding from the other side.)

    Symmetric difference, not a boolean "did the set change": a workspace that swaps one of four
    contents has churned less than one that empties, and the criterion should be able to say so."""
    return sum(len(set(a) ^ set(b)) for a, b in zip(sets, sets[1:]))


def count_retained(heads, sets):
    """Ticks where the PREVIOUS head is still somewhere in the workspace -- what a single slot
    could not do, and the evidence that `count_switches(heads)` is rank churn rather than flicker.
    The argmax control is its reachable red: there `broadcast_all()` is the head, so this can only
    equal the no-switch count."""
    return sum(1 for a, b in zip(heads, sets[1:]) if a is not None and a in b)


def _fmt(seq):
    """Compact ASCII trace: initials of the broadcast region per tick, '.' for an empty workspace."""
    return "".join((c[0].upper() if c is not None else ".") for c in seq)


def _fmt_set(slots):
    """Compact ASCII trace of ONE tick's whole workspace: sorted initials, '.' when empty."""
    return "".join(sorted(c[0].upper() for c in slots)) or "."


# ======================================================================================================
# THE BINDING FAMILY IS RETIRED (2026-08-07, P8 burden 5: deletion on a null). What stood here:
# `bind_objects` / `bind_broadcast_spiking` / `bind_decode_spiking` / `bind_broadcast_rate` /
# `bind_decode_rate` / `_bind_pair_acc` / `_bind_step` and the four `_BIND_*` constants -- the
# "multi-item BINDING + spiking-vs-rate control" deepening, self-check only, zero external callers
# (swept at retirement: src/tests/tools/research/integrations/training, plus importlib/string
# dispatch; `reasoning/consciousness_probe.py`'s call died 2026-08-04 with its F11 deletion).
#
# THE MEASUREMENT THAT DECIDED IT, run against the family the day it was retired -- the honest
# benchmark its recall-only criterion had been standing in for. Jaccard |true & got| / |true | got|
# (F1 agreed directionally), 120 trials/K, seed 0, baseline-relative margin = the rate control's
# own per-trial spread:
#
#     K   spiking   rate     margin    control sd    all-FIRED-pairs strawman
#     1   1.0000    1.0000   +0.0000     0.0000      1.0000
#     2   0.5000    0.5083   -0.0083     0.5020      0.5000
#     4   0.2500    0.1829   +0.0671     0.2175      0.2500
#
# The pre-registered NULL (from docs/GAP_CLOSURE.md's own K=2 Jaccard numbers) CONFIRMED, and
# sharper than predicted: the spiking decoder's score EQUALS the all-fired-pairs strawman at every
# K -- it returned the full cross product, so phase multiplexing recovered NOTHING beyond
# membership -- and at K=2 the rate control is nominally BETTER. No K shows a margin outside the
# control's own spread. "Synchrony binds" was a property of the scorer; with an honest scorer there
# is no effect to guard, so the family is DELETED rather than the benchmark installed over a null
# (a permanent criterion needs an effect, and the refutation is this record). The bare 0.55
# re-entry sustain -- the undeclared constant that produced the cross product -- goes with it.
# THE THREE-WAY P2 MERGE (GlobalWorkspace binding / bind_* / ThetaGammaCode) IS REFUTED, not
# deferred: there is no measured binding capability here to merge. See docs/P4C_BACKLOG.md.
# ======================================================================================================


def main():

    # ---- FAIR CONTROL: same stream, spiking ignition ON vs OFF (pure per-tick argmax) ----------
    n_ticks = 48
    stream = build_stream(n_ticks, seed=0)
    with_ig, set_ig = run_stream(stream, use_ignition=True, seed=0)
    without, set_no = run_stream(stream, use_ignition=False, seed=0)
    churn_with = count_churn(set_ig)
    churn_without = count_churn(set_no)
    kept_with = count_retained(with_ig, set_ig)
    kept_without = count_retained(without, set_no)
    sw_with = count_switches(with_ig)  # HEAD switches -- rank churn, NOT flicker; see count_churn
    # ONE default-parameter workspace read by every section below that needs the shipped constants.
    # `tools/p6_gate.py` censuses instantiation sites per module, and four separate constructions of
    # an object built only to read three attributes off it would add four to the very count this
    # work exists to reduce -- the same reason `cortex/access_consciousness.py::_bus` exists.
    _defaults = GlobalWorkspace()
    cap = _defaults.capacity
    band_lo = _defaults.ignition_threshold
    band_hi = 1.0 - (_defaults.self_boost + _defaults.hysteresis)

    print("=" * 74)
    print("MULTI-SLOT GLOBAL WORKSPACE -- LIF pools -> compete -> ignite -> admit -> decay")
    print("=" * 74)
    print("regions: perception(P) memory(M) salience/threat(S) language(L) motor(O); '.' = empty")
    print(f"capacity = floor(1 / (self_boost + hysteresis)) = {cap} slots")
    print("threat burst (region 'salience') at ticks 30-35\n")
    print(f"WITH ignition  (spiking LIF + inhibition): {_fmt(with_ig)}  churn={churn_with}")
    print(f"WITHOUT (pure per-tick argmax control)   : {_fmt(without)}  churn={churn_without}")
    print("workspace sets  : " + " ".join(_fmt_set(s) for s in set_ig))
    print(
        f"\n-> the spiking ignition non-linearity cut workspace CHURN from {churn_without} (flicker) "
        f"to {churn_with} (stable). The HEAD changed on {sw_with} ticks and the previous head was "
        f"still in the workspace on {kept_with}/{len(with_ig) - 1} of them (control: "
        f"{kept_without}), so head switches are RANK churn here and not flicker at all"
    )

    # threat must actually break through: a genuinely stronger supra-threshold signal ignites.
    threat_broadcast = any(c == "salience" for c in with_ig[30:36])

    # ---- ANCHOR (i) BOTTLENECK: many supra-threshold candidates -> at most `capacity` admitted --
    # This is the criterion that stops "multi-slot" from becoming "everything broadcasts". The
    # candidates are IDENTICAL, so the spiking competition cannot separate them and every one of
    # them ignites -- which is exactly the case where only the capacity can be the bottleneck.
    seen = []
    b_ws = GlobalWorkspace(seed=0)
    b_ws.subscribe(seen.append)
    n_flood = 40
    for i in range(n_flood):
        b_ws.submit(f"src{i:02d}", f"cand{i:02d}", 0.90)  # 40 simultaneous supra-threshold posts
    b_content = b_ws.step()
    b_ignited = sum(1 for v in b_ws._last_late.values() if v > 0)
    b_admitted = len(b_ws.broadcast_all())
    bottleneck_ok = (
        b_content is not None
        and b_ignited == n_flood  # every candidate ignited...
        and b_admitted == cap  # ...and the workspace admitted only `capacity` of them
    )
    # The DELIVERY conditions are asserted separately below rather than folded in here, so a
    # regression in the broadcast contract reports the broadcast contract instead of surfacing as a
    # bottleneck failure -- measured: with the delivery reverted, both of the compound form's arms
    # went red with a message about capacity.
    delivery_ok = (
        len(seen) == 1  # ONE delivery per step (not one per admitted content)
        and seen[0] == b_ws.broadcast_all()  # carrying the whole admitted SET
        and seen[0][0] == b_content  # whose head is what step() returned
    )
    # THE DELIVERY CONTRACT, measured on the one tick where head and set are furthest apart. A
    # subscriber receives the workspace, so it must receive contents it did NOT top -- which is the
    # entire capability `_deliver` was changed to provide, and the only thing that lets a consumer
    # ask about its own membership instead of inferring it from a rank it cannot see.
    b_beyond_head = len(seen[0]) - 1
    b_immutable = True
    try:
        hash(seen[0])  # a delivered MUTABLE container would hand out a live handle on the workspace
    except TypeError:
        b_immutable = False
    print(
        f"\nbottleneck   : {n_flood} identical supra-threshold candidates -> {b_ignited} ignited, "
        f"{b_admitted} admitted (capacity {cap}); {len(seen)} delivery of {len(seen[0])} content(s) "
        f"-- {b_beyond_head} beyond the head"
    )

    # ---- SPIKING COMPETITION under the inhibitory rheobase. This block used to assert the
    #      OPPOSITE -- "a free slot is NOT a free pass: the spiking competition still excludes
    #      them", three candidates in four slots with ONE admitted -- and that comment stated the
    #      P2 violation as design intent: two mechanisms limited the held count, and the implicit
    #      one (lateral inhibition) overrode the derived one (`capacity`) on every live tick.
    #      RE-POINTED, not weakened: BELOW capacity nothing is excluded (3 candidates, 4 slots ->
    #      all three admitted, ordered by strength); ABOVE capacity the competition binds and the
    #      weakest is excluded (5 candidates -> exactly `capacity` admitted). The red is
    #      `NoAllowanceArbiter`, the retired kernel, admitting ('strong',) ALONE on the identical
    #      under-capacity load -- inhibition deciding the set with three slots free.
    g_ws = GlobalWorkspace(seed=0)
    g_ws.submit("strong", "strong", 0.90)
    g_ws.submit("mid", "mid", 0.68)
    g_ws.submit("weak", "weak", 0.62)
    g_content = g_ws.step()
    wta_winner_late = g_ws._last_late["strong", "strong"]
    wta_losers_late = g_ws._last_late["mid", "mid"] + g_ws._last_late["weak", "weak"]
    wta_ok = (
        g_content == "strong"
        and wta_winner_late > 0
        and wta_losers_late > 0  # the sub-capacity rivals now SUSTAIN -- nobody is annihilated
        and g_ws.broadcast_all() == ("strong", "mid", "weak")  # ALL admitted, strength order
    )
    over_ws = GlobalWorkspace(seed=0)
    for _src, _sal in (
        ("strong", 0.90),
        ("mid2", 0.80),
        ("weak2", 0.72),
        ("mid", 0.68),
        ("weak", 0.62),
    ):
        over_ws.submit(_src, _src, _sal)
    over_ws.step()
    wta_red_ws = NoAllowanceArbiter(seed=0)
    wta_red_ws.submit("strong", "strong", 0.90)
    wta_red_ws.submit("mid", "mid", 0.68)
    wta_red_ws.submit("weak", "weak", 0.62)
    wta_red_ws.step()

    # ---- ANCHOR (ii) ALL-OR-NONE IGNITION: sub-threshold stays local, supra-threshold ignites --
    w_ws = GlobalWorkspace(seed=0)
    w_ws.submit("perception", "weak", 0.40)  # below rheobase -> never fires
    sub = w_ws.step()
    weak_spikes = sum(w_ws._last_total.values())  # a weak submission emits NO spikes
    s_ws = GlobalWorkspace(seed=0)
    s_ws.submit("perception", "strong", 0.80)  # above rheobase -> ignites
    supra = s_ws.step()
    strong_spikes = sum(s_ws._last_total.values())  # a strong submission DOES spike
    # not linear averaging: two sub-threshold posts do NOT sum into an ignition.
    n_ws = GlobalWorkspace(seed=0)
    n_ws.submit("memory", "half_a", 0.50)
    n_ws.submit("language", "half_b", 0.50)
    no_sum = n_ws.step()
    no_sum_spikes = sum(n_ws._last_total.values())

    # ---- ANCHOR (iii) GLOBAL REACH: every subscriber gets the identical broadcast --------------
    r_ws = GlobalWorkspace(seed=0)
    inbox_a, inbox_b, inbox_c = [], [], []
    r_ws.subscribe(inbox_a.append)
    r_ws.subscribe(inbox_b.append)
    r_ws.subscribe(inbox_c.append)
    for sal in (0.70, 0.30, 0.72, 0.20):
        r_ws.submit("perception", "scene", sal)
        r_ws.step()
    reach_ok = inbox_a == inbox_b == inbox_c and len(inbox_a) == 4

    # ---- ANCHOR (iv) PERSISTENCE + DECAY: ignite, then no input -> attractor holds then fades ---
    d_ws = GlobalWorkspace(seed=0)
    d_ws.submit("memory", "trace", 0.95)
    d_ws.step()  # ignite
    life = []
    for _ in range(8):
        life.append(d_ws.step())  # no new submissions -> attractor decays
    persisted = sum(1 for c in life if c == "trace")
    released = life[-1] is None
    print("\npersistence/decay (ignite 'memory' at 0.95, then silence):")
    print(
        f"   post-ignition life trace: {_fmt(life)}  "
        f"(held 'trace' {persisted} ticks, released={released})"
    )

    # (The multi-item BINDING harness that ran here is retired with its family -- see the
    # retirement record above `main()` for the honest-scorer measurement that decided it.)

    # ---- LIVE CONTRACT: the exact submit/step sequences autonomous_loop + live_brain rely on ----
    c_ws = GlobalWorkspace(seed=0)
    c_ws.submit("perception", "unit0", 0.6)  # safe tick: perception holds the slot...
    c_ws.submit("salience", "threat", 0.2)  # ...vs a low background threat
    contract_safe = c_ws.step()  # -> broadcasts the current percept
    c_ws.submit("salience", "threat", 0.95)  # danger tick: only the threat is posted
    contract_danger = c_ws.step()  # -> the threat ignites + is broadcast

    # ---- DETERMINISM: identical stream+seed -> byte-identical over MANY runs -------------------
    # 24 runs, not 2. The arbiter now publishes a SET, so the digest covers the whole workspace per
    # tick (head + full set + hold) rather than the head alone -- a reordering inside the set would
    # be invisible to a head-only comparison, and reordering is the failure mode multiple slots
    # introduce. Byte-identical means one distinct repr across every run.
    # THE MEMBRANE FLOATS ARE IN THE DIGEST, and since 2026-08-06 that is a requirement rather than
    # thoroughness: `_rank_key` reads `_last_v`, so a float that moves in its last bits can now
    # change ADMISSION and not merely a printed number. A digest over contents alone would report
    # byte-identical while the arbiter was ranking on a different quantity.
    def _det_run():
        """One full stream, digested as (head, admitted set, EVERY pool's final membrane) per tick."""
        ws = GlobalWorkspace(use_ignition=True, seed=0)
        rows, heads = [], []
        for tick in build_stream(n_ticks, seed=0):
            for region, salience in tick.items():
                ws.submit(region, region, salience)
            heads.append(ws.step())
            rows.append((heads[-1], ws.broadcast_all(), tuple(sorted(ws._last_v.items()))))
        return repr(rows), heads

    det_digests, det_heads = zip(*(_det_run() for _ in range(24)))
    # NON-VACUITY: the digest has to CONTAIN floats that vary, or "24 identical digests" is a
    # statement about a constant. Counted rather than assumed.
    det_probe = GlobalWorkspace(seed=0)
    det_probe.submit("a", "A", 0.90)
    det_probe.submit("b", "B", 0.72)
    det_probe.step()
    det_membranes = sorted(set(det_probe._last_v.values()))
    deterministic = (
        len(set(det_digests)) == 1
        and list(det_heads[0]) == with_ig
        and count_switches(with_ig) == sw_with
        and len(det_membranes) >= 2
        and all(v not in (0.0, 1.0) for v in det_membranes)
    )

    print(
        f"\nlive contract : safe tick broadcasts {contract_safe!r}; "
        f"proximity threat ignites {contract_danger!r}"
    )
    print(
        f"spiking comp  : competitors 0.90/0.68/0.62 (below capacity) -> head {g_content!r} "
        f"(late spikes {wta_winner_late}), rivals SUSTAIN (late spikes {wta_losers_late}), all "
        f"admitted {g_ws.broadcast_all()}; 5 candidates (above capacity) -> "
        f"{len(over_ws.broadcast_all())} admitted, weakest excluded; retired-kernel red "
        f"{wta_red_ws.broadcast_all()}"
    )
    print(
        f"spiking readout: weak@0.40 spikes={weak_spikes}, strong@0.80 spikes={strong_spikes} "
        f"(discrete integer spike events)"
    )

    # ============================ genuine self-checks (not rigged) ==============================
    # (a) FUNCTION -----------------------------------------------------------------------------
    # the spiking ignition stabilises the workspace vs the pure-argmax control, by a real margin.
    # Measured on CHURN (contents entering/leaving), which is what flicker means for a set -- see
    # `count_churn` for why the head-switch count was retired rather than re-thresholded.
    assert churn_without > churn_with, (
        f"argmax control must flicker more ({churn_without} vs {churn_with})"
    )
    assert churn_without >= 3 * max(churn_with, 1), (
        f"stabilisation must be large, not marginal: with={churn_with} without={churn_without}"
    )
    assert churn_with <= 12, (
        f"spiking workspace must be stable WITH ignition (little churn), got {churn_with}"
    )
    # ...and the head-switch count is NOT flicker on this arbiter, which is the measurement that
    # forced the criterion above. Every tick whose head changed still had the previous head in the
    # workspace. Its RED is the argmax control, where one slot makes the two measures identical.
    assert kept_with == len(with_ig) - 1, (
        f"a multi-slot workspace must not DROP the previous broadcast when its rank changes: "
        f"retained {kept_with} of {len(with_ig) - 1}"
    )
    assert kept_without < len(without) - 1, (
        "the retention criterion must be reachable-red -- the single-content argmax control MUST "
        f"fail it, and it retained {kept_without} of {len(without) - 1}"
    )
    # bottleneck: 40 identical supra-threshold candidates all ignite, and only `capacity` are
    # admitted. This is the criterion that stops multi-slot becoming "everything broadcasts".
    assert bottleneck_ok, (
        f"exactly `capacity` contents must be admitted through the bottleneck: ignited {b_ignited} "
        f"of {n_flood}, admitted {b_admitted}, capacity {cap}"
    )
    assert b_admitted < b_ignited, (
        f"the capacity must actually BIND -- if every ignited candidate is admitted the bottleneck "
        f"is gone ({b_admitted} admitted of {b_ignited} ignited)"
    )
    assert isinstance(b_content, str), "the top broadcast content must be one region, not a set"
    # DELIVERY: subscribers are handed the WHOLE admitted set, and the point of that is the part
    # they could not previously see -- the contents they did not top. A subscriber's own membership
    # is unanswerable without it, which is how `cortex/access_consciousness.py` came to settle a
    # consciousness-relevant flag on the arbiter's lexicographic tie-break.
    assert delivery_ok, (
        f"ONE broadcast per cycle, carrying EXACTLY the admitted set, head first: {len(seen)} "
        f"delivery/deliveries of {seen[0]!r} against broadcast_all() {b_ws.broadcast_all()!r} and "
        f"a step() head of {b_content!r}"
    )
    assert b_beyond_head == cap - 1 and b_beyond_head >= 1, (
        f"a broadcast must carry every admitted content, not the head alone: the delivery held "
        f"{b_beyond_head} content(s) beyond the head against a capacity of {cap}"
    )
    assert b_immutable and type(seen[0]) is tuple, (
        f"the delivered workspace must be an immutable TUPLE all the way down -- handing a "
        f"subscriber a mutable container gives it a live handle on workspace state and makes "
        f"`_slot()`'s whole-snapshot argument false from outside this class (got "
        f"{type(seen[0]).__name__}, hashable={b_immutable})"
    )
    # ...and THAT criterion is reachable-red by a configuration this module accepts rather than by
    # breaking the accessor: a workspace whose defence consumes the entire hold budget has capacity
    # 1 -- the retired single-slot arbiter -- so its delivery can never carry anything beyond the
    # head, and the assert above goes red on it. Run, not argued.
    one_seen = []
    one_slot_ws = GlobalWorkspace(seed=0, self_boost=0.10, hysteresis=0.90)
    one_slot_ws.subscribe(one_seen.append)
    for i in range(n_flood):
        one_slot_ws.submit(f"src{i:02d}", f"cand{i:02d}", 0.90)
    one_slot_ws.step()
    assert one_slot_ws.capacity == 1 and len(one_seen) == 1 and len(one_seen[0]) - 1 == 0, (
        f"the delivery criterion must be reachable-red: on a ONE-slot workspace the broadcast "
        f"cannot carry anything past the head (capacity {one_slot_ws.capacity}, delivered "
        f"{one_seen[0] if one_seen else None})"
    )
    # ...and THAT criterion is falsifiable by an input this module accepts rather than only by
    # breaking the accessor: a workspace whose defence is below the tie-break noise floor has a
    # capacity wider than this flood, so nothing is excluded and the bottleneck assert goes red.
    # Run, not argued. (Under the inhibitory rheobase this arm is unchanged for a second reason
    # worth recording: at capacity 49 the allowance is 48 saturated slots' worth of activity,
    # which exceeds any `a_inh` this flood can produce, so the lateral term is identically zero
    # here with or without the excess clause.)
    nocap = GlobalWorkspace(seed=0, self_boost=0.0, hysteresis=1e-9)
    for i in range(n_flood):
        nocap.submit(f"src{i:02d}", f"cand{i:02d}", 0.90)
    nocap.step()
    assert nocap.capacity > n_flood and len(nocap.broadcast_all()) == n_flood, (
        f"the bottleneck criterion must be reachable-red: at a sub-jitter defence the capacity is "
        f"{nocap.capacity} and all {len(nocap.broadcast_all())} candidates broadcast"
    )
    # genuine spiking competition, on the side of the allowance where it belongs: BELOW capacity
    # every distinct-strength pool sustains and is admitted in strength order; the competition is
    # allowed to bind only where the capacity says the workspace is full.
    assert wta_ok, (
        f"below capacity NOTHING may be excluded and the head order must follow strength: "
        f"expected all of strong/mid/weak admitted with 'strong' the head, got "
        f"{g_ws.broadcast_all()} (late winner {wta_winner_late}, late rivals {wta_losers_late})"
    )
    assert len(over_ws.broadcast_all()) == cap and "weak" not in over_ws.broadcast_all(), (
        f"ABOVE capacity the competition must bind at exactly `capacity` and exclude the weakest: "
        f"5 distinct-strength candidates into {cap} slots gave {over_ws.broadcast_all()}"
    )
    assert wta_red_ws.broadcast_all() == ("strong",), (
        f"the sub-capacity criterion must be reachable-red, and its red is the RETIRED KERNEL "
        f"(`NoAllowanceArbiter`): with no inhibitory rheobase the same three candidates collapse "
        f"to the strongest alone -- inhibition deciding the set with {cap - 1} slots free. Got "
        f"{wta_red_ws.broadcast_all()}"
    )
    # all-or-none ignition non-linearity: sub-threshold excluded, supra admitted, no summation.
    assert sub is None, f"a sub-threshold candidate must NOT ignite the workspace (got {sub!r})"
    assert supra == "strong", f"a supra-threshold candidate must ignite (got {supra!r})"
    assert no_sum is None, (
        "two sub-threshold candidates must NOT sum into an ignition (all-or-none)"
    )
    # global reach: every subscriber received the identical broadcast sequence.
    assert reach_ok, "every subscriber must receive the identical broadcast (global reach)"
    # persistence + decay: the content is sustained several ticks, then fades below threshold.
    assert persisted >= 2, f"ignited content must persist for several ticks (held {persisted})"
    assert released, "unrefreshed content must eventually decay below threshold and be released"
    # a genuinely stronger, supra-threshold signal DID take over (workspace is not frozen).
    assert threat_broadcast, "the supra-threshold threat burst must break through and be broadcast"
    # the LIVE contract used by autonomous_loop (sec 8) + live_brain must hold exactly.
    assert contract_safe is not None, (
        "a safe tick must broadcast the current percept (ws_safe is not None)"
    )
    assert contract_danger == "threat", (
        "a proximity threat must ignite + be broadcast (ws_danger == 'threat')"
    )

    # (a2) WINNABILITY -- the defect this arbiter replaced, and the A/B that retires the old one --
    # THE OLD SINGLE-SLOT ARBITER, kept ONLY here as the control (P2: the superseded behaviour is
    # reachable for comparison in the self-check, never on the live path). It is the current
    # arbiter with its capacity pinned at one, which is exactly what the old code did: one slot,
    # one incumbent carrying the whole defence, and the undivided `_G_INH` lateral gain. Nothing
    # about the old dynamics is reconstructed by hand -- pinning the capacity IS the old dynamics,
    # which is itself the evidence that the capacity is the thing that changed.
    class _SingleSlotControl(GlobalWorkspace):
        """The retired winner-take-all arbiter: capacity 1, undivided lateral gain."""

        @property
        def capacity(self):
            return 1

    def _in_slot(w):
        return "CHAL" in w.broadcast_all()

    def _is_head(w):
        return w.current == "CHAL"

    def _lowest_win(cls, tenant_claim, ceiling, warm=6, tenants=1, test=None, chal_src="chal"):
        """Lowest salience a challenger must post to WIN A SLOT beside warmed tenant(s).

        Returns None if no salience at or below `ceiling` gets in. Sweeps in steps of 0.005; the
        tenants keep posting on the challenge tick, so this is a live incumbent, not a stale one.

        `chal_src` renames the CHALLENGER'S SOURCE and changes nothing else -- same content, same
        claim, same competition. It exists for the rename criterion below: a correct arbiter must
        return the identical sweep whatever the challenger is called."""
        s = band_lo
        while s <= ceiling + 1e-9:
            w = cls(seed=0)
            for _ in range(warm):
                for i in range(tenants):
                    w.submit(f"t{i}", f"T{i}", tenant_claim)
                w.step()
            for i in range(tenants):
                w.submit(f"t{i}", f"T{i}", tenant_claim)
            w.submit(chal_src, "CHAL", s)
            w.step()
            if (test or _in_slot)(w):
                return round(s, 3)
            s += 0.005
        return None

    # The live producer range, measured on 60 real AutonomousArm ticks, plus the band extremes.
    win_claims = (band_lo, 0.5797, 0.6573, 0.700, band_hi)
    win_new = [_lowest_win(GlobalWorkspace, c, 1.0) for c in win_claims]
    win_old = [_lowest_win(_SingleSlotControl, c, 1.0) for c in win_claims]
    print("\nwinnability  : lowest salience a compliant challenger needs to WIN A SLOT")
    print(
        f"   compliant band = [ignition_threshold, 1 - (self_boost + hysteresis)] = "
        f"[{band_lo:.2f}, {band_hi:.2f}]  (defence {band_hi and 1.0 - band_hi:.2f})"
    )
    print(f"   {'tenant claim':>13} | {'OLD 1-slot':>10} | {'NEW multi':>9}")
    for c, o, nw in zip(win_claims, win_old, win_new):
        print(
            f"   {c:>13.4f} | {str(o):>10} | {str(nw):>9}"
            f"{'' if (nw is not None and nw <= band_hi) else '   <- OUTSIDE BAND'}"
        )

    # THE SWEEP'S TWO ASSERTS AND ITS `over_band` RED ARE RETIRED, and the retirement is the
    # inhibitory rheobase working rather than a loss of coverage. Under the rheobase any
    # supra-rheobase challenger joins a below-capacity workspace, so `win_new` collapses flat to
    # the band floor and "a compliant specialist can win a slot" is TRIVIALLY green -- a criterion
    # that cannot fail is deleted, not kept for its message (P8 burden 1). Its red, `over_band`
    # (an above-band tenant must not be joinable), is likewise unreachable-green now: joinability
    # below capacity no longer depends on the tenant at all, which was confirmed unavoidable at 15
    # points across 3 mechanism families before retiring it. WHERE THE QUESTION STILL BITES is a
    # FULL workspace, and the `full_compliant` / `full_strong` pair below inherits it there --
    # strictly stronger on `over_band`'s own stated axis, because its tenant count IS the capacity,
    # so widening the workspace cannot cheat it. The sweep itself is still RUN and PRINTED: it
    # feeds the head-dissociation criteria below, and the flat row is the disclosure of what the
    # rheobase did to slot winnability.

    # WINNING A SLOT MUST NOT MEAN EVICTING THE TENANT -- that would be single-slot with extra
    # steps, and it is the specific regression a prior pass declined to ship: repointing a second
    # faculty onto the shared bus made the arm's broadcast go '0' -> 'GLOBAL' PERMANENTLY and the
    # percept vanished from the workspace entirely. Measured on the LIVE brain with the real
    # `cortex/access_consciousness` faculty sharing the arm's own bus object at a compliant 0.65
    # claim: the retired arbiter kept the percept in the workspace on 0 of 40 ticks, this one on
    # 40 of 40. The mechanism-level form of that measurement is asserted here.
    def _coexist(cls, tenant_claim, challenger):
        w = cls(seed=0)
        for _ in range(6):
            w.submit("t0", "T0", tenant_claim)
            w.step()
        w.submit("t0", "T0", tenant_claim)
        w.submit("chal", "CHAL", challenger)
        w.step()
        return w.broadcast_all()

    # WINNING A SLOT IS NOT BECOMING THE HEAD, and the header's LIMIT 3 rests on that dissociation.
    # Asserted so the limit cannot go quietly stale in either direction: if the head ever becomes
    # winnable inside the band this fires and the header must be re-derived rather than left
    # standing, which is the exemption-rot discipline `tools/citation_guard.py` uses. Its RED is the
    # retired arbiter, where one slot makes "in the workspace" and "is the head" the SAME question.
    win_head = [_lowest_win(GlobalWorkspace, c, 1.0, test=_is_head) for c in win_claims]
    head_old = [_lowest_win(_SingleSlotControl, c, 1.0, test=_is_head) for c in win_claims]
    print(f"   winning a SLOT {win_new}  vs becoming the HEAD {win_head}")
    assert any(h is None or h > band_hi + 1e-9 for h in win_head), (
        f"LIMIT 3 in the module header states the HEAD is not winnable inside the band while a "
        f"SLOT is. If that has stopped being true the header is stale and must be re-derived, not "
        f"left standing: head {win_head} against ceiling {band_hi:.2f}"
    )
    assert head_old == win_old, (
        f"the dissociation must be reachable-red -- on the retired single-slot arbiter 'in the "
        f"workspace' and 'is the head' are the SAME question and must give the SAME sweep: "
        f"slot {win_old} vs head {head_old}"
    )

    # (a2c) DEFECT #16 -- THE FOCUS MUST BE CAPTURABLE, and the assert above cannot see it. --------
    # That one is an `any()` over the sweep, and the point at the band CEILING satisfies it by
    # construction at every setting this module can take: displacing the head costs the tenant's
    # claim plus the GRIP, so a ceiling tenant is displaceable only above `hi + grip` -- outside
    # the band, and measured at 0.815 against the 0.75 ceiling. (Under the pre-split arbiter the
    # bound was `hi + hold == 1.0` identically, i.e. unwinnable outright; the split narrowed it to
    # merely non-compliant, and the assert is satisfied either way.) It was therefore GREEN
    # throughout the period when NO point of the band was winnable, and stays green now that three
    # of five are: it can only ever police the top of the band. This is its complement and the
    # pair is the honest statement -- the ceiling is undisplaceable by any COMPLIANT challenger,
    # the rest of the band must not be.
    #
    # THE RED IS ONE SUBSIDY DOING BOTH JOBS, which is the pre-split arbiter exactly: restore
    # `_focus_key` to `_rank_key` and the head is ranked on `claim + hold` again. Nothing else about
    # the module changes, so this isolates the split rather than the parameters -- the hold, the
    # band, the capacity and the lateral gain are all the shipped ones in both arms.
    red_head = [_lowest_win(OneSubsidyArbiter, c, 1.0, test=_is_head) for c in win_claims]
    in_band = [h for h in win_head if h is not None and h <= band_hi + 1e-9]
    red_in_band = [h for h in red_head if h is not None and h <= band_hi + 1e-9]
    print(
        f"   focus capture: hold {_defaults.self_boost + _defaults.hysteresis:.4f} holds the SLOT, "
        f"grip {_defaults.grip:.4f} contests the HEAD -> {len(in_band)} of {len(win_claims)} "
        f"compliant tenants displaceable; ONE-SUBSIDY red {red_head} -> {len(red_in_band)}"
    )
    assert in_band, (
        f"defect #16: a compliant challenger must be able to take the HEAD from a compliant tenant "
        f"somewhere inside the band, or the first tenant to arrive holds the focus permanently and "
        f"a stronger signal cannot capture attention. grip {_defaults.grip:.4f}, head {win_head} "
        f"against ceiling {band_hi:.4f}"
    )
    assert not red_in_band, (
        f"the focus-capture criterion must be reachable-red, and its red is ONE SUBSIDY DOING BOTH "
        f"JOBS: with the head ranked on `claim + hold` instead of `claim + grip`, an incumbent at "
        f"the band FLOOR drives {band_lo + _defaults.self_boost + _defaults.hysteresis:.4f} against "
        f"a ceiling of {band_hi:.4f} and NO compliant tenant may be displaceable, or this criterion "
        f"is asserting something that was true anyway. Got {red_head}"
    )
    # ...and the arithmetic the split answers, stated as the inequality rather than only its symptom.
    _hold = float(_defaults.self_boost) + float(_defaults.hysteresis)
    assert band_lo + _defaults.grip < band_hi, (
        f"the GRIP must fit inside the range a compliant challenger can move in: lo {band_lo:.4f} "
        f"+ grip {_defaults.grip:.4f} = {band_lo + _defaults.grip:.4f} must be < ceiling "
        f"{band_hi:.4f}"
    )
    assert band_lo + _hold >= band_hi, (
        f"...and the HOLD must NOT, or the split this criterion guards bought nothing and should be "
        f"reverted (P2): lo {band_lo:.4f} + hold {_hold:.4f} = {band_lo + _hold:.4f} against "
        f"ceiling {band_hi:.4f}. If the hold ever fits, `grip` is redundant -- delete it"
    )
    # THE GRIP IS DERIVED, NOT TYPED (P1): it MOVES with the hold and the capacity it comes from.
    _grips = [
        round(GlobalWorkspace(seed=0, self_boost=0.0, hysteresis=h).grip, 4)
        for h in (0.20, 0.33, 0.5)
    ]
    assert len(set(_grips)) == len(_grips), (
        f"the grip derivation must move with the hold it is derived from -- if it does not it is a "
        f"typed constant wearing a derivation's clothes and should be deleted. Got {_grips}"
    )
    # THE INHIBITORY ALLOWANCE IS DERIVED, NOT TYPED (P1), same discipline as the grip: `_si_sat`
    # must move when the LIF constants it is derived from move -- that is what stops it decaying
    # back into a hardcoded 0.6 with its derivation written beside it, the exact failure
    # `_G_INH = 2.4` was retired for -- and `_inh_free` must move with the capacity it multiplies.
    _sat_ws = GlobalWorkspace(seed=0)
    _sats = [_sat_ws._si_sat]
    _sat_ws._K_INH = _sat_ws._K_INH * 1.5
    _sats.append(_sat_ws._si_sat)
    _sat_ws._LEAK_INH = _sat_ws._LEAK_INH * 0.5
    _sats.append(_sat_ws._si_sat)
    assert len(set(_sats)) == len(_sats), (
        f"the saturation derivation must move with `_K_INH` and `_LEAK_INH` -- if it does not it "
        f"is a typed constant wearing a derivation's clothes and should be deleted. Got {_sats}"
    )
    _frees = [
        GlobalWorkspace(seed=0, self_boost=0.0, hysteresis=h)._inh_free
        for h in (0.20, 0.33, 0.5, 1.0)
    ]
    assert len(set(_frees)) == len(_frees) and _frees[-1] == 0.0, (
        f"the allowance must move with the capacity it is derived from, and at capacity 1 it must "
        f"be EXACTLY 0.0 -- the single-slot workspace IS the retired kernel -- got {_frees}"
    )

    # (a2b) THE ARBITER IS NAME-BLIND -- the criterion the rank repair owes, with its red run. ------
    # The assert above (`any(h is None or h > band_hi)`) is satisfied by BOTH the repaired sweep and
    # the name-decided one it replaced, so it could never see the defect. This is the one that can:
    # RENAMING THE CHALLENGER'S SOURCE, changing nothing else, must leave both sweeps identical.
    #
    # THE RED IS THE RETIRED RANK KEY, reconstructed as a CONFIGURATION of this arbiter rather than
    # as a memory of one -- `_rank_key` is the single place the order is decided, so restoring
    # `(-late, -total, k)` IS the retired behaviour and nothing else about the module changes. Under
    # it two co-saturating pools tie on both clipped counts and the SOURCE STRING settles the head,
    # so a challenger that sorts AFTER the tenant needs a strictly higher claim than one that sorts
    # before it -- which is a specialist's global availability being decided by what it is called.
    class _NameRanked(GlobalWorkspace):
        """The retired rank key: the clipped spike counts, then the source name.

        BOTH orderings are restored, and that is required rather than tidy. The retired arbiter had
        ONE key deciding admission and head alike; restoring only `_rank_key` leaves the head being
        decided by the repaired `_focus_key`, so the red goes GREEN and the criterion silently
        stops testing anything -- measured, the two rename sweeps came back byte-identical. A
        control has to reconstruct the retired behaviour on the path the criterion actually reads."""

        def _rank_key(self, k):
            return (-self._last_late[k], -self._last_total[k], k)

        def _focus_key(self, k):
            return (-self._last_late[k], -self._last_total[k], k)

    lo_src, hi_src = "aaa", "zzz"  # sort BEFORE and AFTER the tenants' "t0"
    head_lo = [
        _lowest_win(GlobalWorkspace, c, 1.0, test=_is_head, chal_src=lo_src) for c in win_claims
    ]
    head_hi = [
        _lowest_win(GlobalWorkspace, c, 1.0, test=_is_head, chal_src=hi_src) for c in win_claims
    ]
    red_lo = [_lowest_win(_NameRanked, c, 1.0, test=_is_head, chal_src=lo_src) for c in win_claims]
    red_hi = [_lowest_win(_NameRanked, c, 1.0, test=_is_head, chal_src=hi_src) for c in win_claims]
    slot_lo = [_lowest_win(GlobalWorkspace, c, 1.0, chal_src=lo_src) for c in win_claims]
    slot_hi = [_lowest_win(GlobalWorkspace, c, 1.0, chal_src=hi_src) for c in win_claims]
    print(f"   name-blind   : HEAD {lo_src!r} {head_lo} == {hi_src!r} {head_hi}")
    print(f"                  retired rank key {red_lo} vs {red_hi}  <- the RED")
    assert head_lo == head_hi and slot_lo == slot_hi, (
        f"THE ARBITER MUST BE NAME-BLIND. Renaming the challenger's SOURCE changes no claim, no "
        f"content and no competition, so both sweeps must be identical: HEAD {head_lo} vs "
        f"{head_hi}, SLOT {slot_lo} vs {slot_hi}. A difference means the rank key has fallen back "
        f"through to `k` and a specialist's access is decided by what it is called"
    )
    assert red_lo != red_hi, (
        f"the name-blindness criterion must be reachable-red, and its red is the RETIRED RANK KEY "
        f"reconstructed here rather than described: under `(-late, -total, k)` the same rename must "
        f"MOVE the head sweep, or this criterion is asserting something that was true anyway. Got "
        f"{red_lo} vs {red_hi}"
    )
    # ...and the repaired sweep is the one printed in LIMIT 3. Pinned so the header cannot go stale.
    assert win_head == head_lo, (
        f"the head sweep must not depend on the challenger's default name either: {win_head} vs "
        f"{head_lo}"
    )
    # A FIXED in-band claim, no longer `max(win_new)`: that anchor was the top of a sweep that
    # meant something when winnability was graded, and under the rheobase the sweep sits flat at
    # the band floor -- an anchor that silently drifted to the weakest legal challenger. 0.72 is
    # the historical anchor value, kept as a fixed test input so the co-existence arms stay on the
    # claim they have always been measured at.
    coex_claim = 0.72
    coex_new = _coexist(GlobalWorkspace, 0.6573, coex_claim)
    coex_old = _coexist(_SingleSlotControl, 0.6573, coex_claim)
    print(f"   co-existence at {coex_claim}: NEW {coex_new} | retired arbiter {coex_old}")
    assert "T0" in coex_new and "CHAL" in coex_new, (
        f"a challenger winning a slot must not EVICT the tenant -- both must stay available, got "
        f"{coex_new}"
    )
    assert len(coex_old) == 1, (
        f"the co-existence criterion must be reachable-red -- the retired single-slot arbiter "
        f"cannot hold two contents and MUST fail it, got {coex_old}"
    )
    # A FULL workspace is FULL -- capacity is a real constraint, not a formality. With every slot
    # held by a compliant tenant, a compliant challenger cannot get in; admission then costs a
    # genuinely exceptional claim, which is the 0.95 proximity threat the live contract posts.
    # THIS PAIR NOW CARRIES THE WINNABILITY RED retired above: `full_compliant is None` is the
    # exclusion `over_band` asserted, restated where it is still true, and it is strictly stronger
    # on that criterion's own axis because the tenant count here IS the capacity -- widening the
    # workspace recruits more tenants instead of opening a side door.
    full_compliant = _lowest_win(GlobalWorkspace, 0.6573, band_hi, tenants=cap)
    full_strong = _lowest_win(GlobalWorkspace, 0.6573, 1.0, tenants=cap)
    print(
        f"   full workspace ({cap} compliant tenants at 0.6573): compliant challenger "
        f"{full_compliant}, admitted at {full_strong}"
    )
    assert full_compliant is None and full_strong is not None, (
        f"a FULL workspace must exclude a compliant challenger and admit an exceptional one: "
        f"compliant {full_compliant}, exceptional {full_strong}"
    )

    # (a2d) THE P2 CLOSE-OUT FOR THE INHIBITORY RHEOBASE: capacity decides the set, the gain does
    # not, AND the gain is not dead. Before the rheobase, two mechanisms limited the held count --
    # lateral inhibition (implicit, growing with the competitor count) and `capacity` (explicit,
    # derived, Cowan-corroborated) -- and the implicit one won: measured live over 120 wake ticks,
    # occupancy {1: 117, 2: 3} with three slots free, `fired[:cap]` a no-op on the production
    # path. Exactly one mechanism may survive (P2). Three conjuncts on one 8-producer load at
    # live-shaped claims, each producer verified independently ignitable first, with the retired
    # kernel (`NoAllowanceArbiter`) run as the red on the identical load.
    p2_claims = (0.5797, 0.60, 0.62, 0.6573, 0.68, 0.70, 0.72, 0.75)

    def _p2_load(cls, **kw):
        w = cls(seed=0, **kw)
        for i, c in enumerate(p2_claims):
            w.submit(f"p{i}", f"C{i}", c)
        w.step()
        return w

    # PRE-GUARD: every producer ignites ALONE, so the counts below are about the capacity and
    # never about a claimant that could not have ignited anyway.
    for _i, _c in enumerate(p2_claims):
        _w_alone = GlobalWorkspace(seed=0)
        _w_alone.submit(f"p{_i}", f"C{_i}", _c)
        _w_alone.step()
        assert len(_w_alone.broadcast_all()) == 1, (
            f"pre-guard: producer p{_i} at {_c} must be independently ignitable, got "
            f"{_w_alone.broadcast_all()}"
        )

    # (1) CAPACITY DECIDES. The same defences the capacity table sweeps; the admitted count is
    # exactly `min(capacity, load)` -- `min`, never bare `capacity`, or this asserts something
    # about the load -- and the admitted sets NEST as the workspace widens.
    p2_sets = {}
    for _sb, _hy in ((0.10, 0.90), (0.05, 0.45), (0.05, 0.28), (0.05, 0.20), (0.05, 0.10)):
        _w = _p2_load(GlobalWorkspace, self_boost=_sb, hysteresis=_hy)
        assert len(_w.broadcast_all()) == min(_w.capacity, len(p2_claims)), (
            f"CAPACITY must decide the admitted count: capacity {_w.capacity} on {len(p2_claims)} "
            f"ignitable producers admitted {len(_w.broadcast_all())}, want "
            f"{min(_w.capacity, len(p2_claims))}"
        )
        p2_sets[_w.capacity] = set(_w.broadcast_all())
    p2_caps = sorted(p2_sets)
    assert len(p2_caps) >= 3, (
        f"the capacity conjunct must be non-vacuous -- fewer than three distinct occupancies "
        f"cannot show a bound moving (got capacities {p2_caps})"
    )
    assert all(p2_sets[a] < p2_sets[b] for a, b in zip(p2_caps, p2_caps[1:])), (
        f"widening the workspace must ADD contents, never swap them -- the admitted sets must "
        f"nest strictly as capacity grows: {[sorted(p2_sets[c]) for c in p2_caps]}"
    )

    # (2) THE GAIN DOES NOT. Across a gain sweep from a quarter of the shipped `_W_SELF` balance
    # to three times it, the admitted SET -- its identity, not its cardinality, which is the
    # weaker claim jitter could satisfy by swapping members -- is pinned.
    def _gained(gain, base=GlobalWorkspace):
        class _Gained(base):
            @property
            def _lateral_gain(self):
                return gain

        return _Gained

    p2_gains = [GlobalWorkspace._W_SELF * (0.25 + 2.75 * i / 9) for i in range(10)]
    p2_rows = []
    for _g in p2_gains:
        _w = _p2_load(_gained(_g))
        p2_rows.append(
            (
                frozenset(_w.broadcast_all()),
                sum(_w._last_total.values()),
                sum(_w._last_v.values()),
            )
        )
    assert len({r[0] for r in p2_rows}) == 1, (
        f"the GAIN must not decide the admitted set: across gains "
        f"{[round(g, 3) for g in p2_gains]} the sets were "
        f"{sorted(len(s) for s, _t, _v in p2_rows)} -- identity must be constant"
    )

    # (3) AND THE GAIN IS NOT DEAD. Without this conjunct, (1) and (2) are green on `_g_lat = 0`
    # -- a deleted mechanism satisfies "does not decide the set" perfectly. The gain must still
    # shape the RATE: the summed spike counts and summed final membranes move across the sweep.
    assert len({t for _s, t, _v in p2_rows}) >= 2 and len({v for _s, _t, v in p2_rows}) >= 2, (
        f"the lateral gain must remain load-bearing on the RATE while the set is pinned: "
        f"sigma-totals {sorted({t for _s, t, _v in p2_rows})}, distinct membrane sums "
        f"{len({v for _s, _t, v in p2_rows})}"
    )

    # THE RED, run on the identical load: the retired kernel fails (1) -- occupancy stops
    # following the capacity -- and fails (2) -- the admitted set moves with the gain.
    p2_red_follow = []
    for _sb, _hy in ((0.05, 0.28), (0.05, 0.20), (0.05, 0.10)):
        _w = _p2_load(NoAllowanceArbiter, self_boost=_sb, hysteresis=_hy)
        p2_red_follow.append(len(_w.broadcast_all()) == min(_w.capacity, len(p2_claims)))
    assert not any(p2_red_follow), (
        f"the close-out must be reachable-red, and its red is the RETIRED KERNEL: with no "
        f"allowance the occupancy must stop following the capacity at 3/4/6, or this criterion "
        f"asserts nothing the old kernel lacked (followed: {p2_red_follow})"
    )
    p2_red_sets = {
        frozenset(_p2_load(_gained(_g, NoAllowanceArbiter)).broadcast_all()) for _g in p2_gains
    }
    assert len(p2_red_sets) > 1, (
        f"...and on the retired kernel the GAIN must decide the set -- the defect itself, run: "
        f"{sorted(len(s) for s in p2_red_sets)} distinct admitted sets expected > 1"
    )
    print(
        f"   P2 close-out : capacity {p2_caps} -> occupancy "
        f"{[len(p2_sets[c]) for c in p2_caps]} (nested), gain x12 sweep set-pinned with "
        f"sigma-total {min(t for _s, t, _v in p2_rows)}..{max(t for _s, t, _v in p2_rows)}; "
        f"retired kernel: occupancy stuck, {len(p2_red_sets)} gain-decided sets"
    )

    # (a3) THE CAPACITY IS DERIVED, NOT PICKED (P1) ----------------------------------------------
    # If this number cannot move it is a hardcode wearing a derivation's clothes. It is computed
    # from `self_boost + hysteresis`, so changing them MUST change it -- run, not asserted in prose.
    cap_rows = [
        (0.05, 0.20, 4),  # the shipped defaults
        (0.05, 0.28, 3),
        (0.05, 0.45, 2),
        (0.10, 0.90, 1),  # a defence that consumes the whole budget -> the single-slot workspace
        (0.00, 0.00, 1),  # ZERO: nothing is held, so there is no multi-tick workspace at all
        # The ROUNDING MODE is a free parameter of this derivation and the four rows above cannot
        # see it: floor and round agree at every one of 0.25 / 0.33 / 0.50 / 1.00, and a mutation
        # swapping `//` for `round` passed this whole self-check. 0.15 separates them (floor 6,
        # round 7), so the mode is now asserted rather than assumed.
        (0.05, 0.10, 6),
        # ...and the divisor is the FLOAT defence, so a non-dyadic value whose decimal reciprocal
        # is a whole number lands one below the decimal reading. 0.20 -> 4, NOT the 5 a reader
        # computes by hand; documented in `capacity` and pinned here so it cannot drift silently.
        (0.00, 0.20, 4),
    ]
    print("\ncapacity     : derived from the defence, not chosen (P1)")
    cap_measured = []
    for sb, hy, want in cap_rows:
        k = GlobalWorkspace(seed=0, self_boost=sb, hysteresis=hy)
        cap_measured.append(k.capacity)
        print(
            f"   self_boost {sb:.2f} + hysteresis {hy:.2f} = defence {sb + hy:.2f} -> "
            f"capacity {k.capacity}, lateral gain {k._lateral_gain:.2f} (E/I balanced), "
            f"grip {k.grip:.4f}"
        )
        assert k.capacity == want, (
            f"capacity must be floor(1/defence): defence {sb + hy:.2f} gave {k.capacity}, want {want}"
        )
    # MEASURED capacities, not the wanted column. A set built from the table's own literals is
    # >= 3 whatever the code does, so the previous form of this guard could not fail at all.
    assert len(set(cap_measured)) >= 3, (
        f"the capacity criterion must be non-vacuous -- if every defence gives the same capacity "
        f"this is a constant with a derivation written beside it and should be deleted. Got "
        f"{sorted(set(cap_measured))} from {len(cap_rows)} defences"
    )
    # SINGLE and SILENCE for the capacity: it is a property of the WORKSPACE, never of the load,
    # so it does not move when one pool competes, when none does, or when the workspace is empty.
    quiet = GlobalWorkspace(seed=0)
    assert quiet.capacity == cap, "capacity must not depend on an empty workspace"
    quiet.submit("solo", "solo", 0.9)
    quiet.step()
    assert quiet.capacity == cap and len(quiet.broadcast_all()) == 1, (
        "capacity must not depend on the number of competitors (a single pool fills one slot)"
    )
    for _ in range(12):
        quiet.step()
    assert quiet.capacity == cap and quiet.broadcast_all() == (), (
        "capacity must survive SILENCE unchanged while the workspace itself empties"
    )

    # (b) GENUINELY SPIKING --------------------------------------------------------------------
    # discrete spikes: a sub-threshold pool emits ZERO spikes; a supra-threshold pool DOES spike.
    assert all(isinstance(n, int) for n in s_ws._last_total.values()), (
        "spikes must be discrete integer events"
    )
    assert weak_spikes == 0, (
        f"a sub-threshold (0.40) pool must emit NO spikes (stays local), got {weak_spikes}"
    )
    assert no_sum_spikes == 0, (
        f"two 0.50 pools must both stay sub-threshold (no spikes), got {no_sum_spikes}"
    )
    assert strong_spikes > 0, (
        f"a supra-threshold (0.80) pool must emit discrete spikes, got {strong_spikes}"
    )
    # winner by SUSTAINED firing: the ignited pool is still spiking in the second half of the
    # window. (The duplicate "sustained firing must belong to ONE pool (WTA)" assert that stood
    # here was DELETED with the inhibitory rheobase: below capacity rivals now sustain by design,
    # and the re-pointed competition criterion in (a) owns both sides of that. Section (b)'s
    # spiking evidence stands on its remaining asserts.)
    assert s_ws._last_late["perception", "strong"] > 0, (
        "the ignited winner must show sustained (2nd-half) firing"
    )

    # (c) DETERMINISM --------------------------------------------------------------------------
    assert deterministic, (
        f"same seed must reproduce identical broadcasts, switch counts AND membrane floats over 24 "
        f"runs: {len(set(det_digests))} distinct digests, {len(det_membranes)} distinct membranes "
        f"in the non-vacuity probe ({det_membranes})"
    )

    # (c1) THE PARALLEL SETTLE THROUGH ITS OWN GATE -- forced by the EMA, and ASSERTED to have run.
    # (c2) below drives `_simulate_parallel` DIRECTLY, which proves the kernel agrees but bypasses
    # the gate that decides whether it is ever reached. That gap became load-bearing the day
    # `_rank_key` started reading `_last_v`: a membrane that differs between the paths in its last
    # bits no longer merely fails a float comparison, it can change WHICH CONTENTS ARE ADMITTED. So
    # the gate is driven to the parallel branch by the one thing that legitimately drives it -- a
    # populated per-pool cost EMA -- and the branch is OBSERVED rather than hoped for.
    def _ranked(ws, tot, late, v):
        """The admission order `step()` would produce from this settle -- one `_rank_key`, no copy."""
        ws._last_total, ws._last_late, ws._last_v = tot, late, v
        return sorted((k for k in late if late[k] > 0), key=ws._rank_key)

    gate_calls = []
    g_ws = GlobalWorkspace(seed=0)
    g_real = g_ws._simulate_parallel

    def _spy(keys, drives, nt, rw):
        gate_calls.append(nt)
        return g_real(keys, drives, nt, rw)

    g_ws._simulate_parallel = _spy
    grng = random.Random(99)
    g_drives = {(f"s{i:03d}", f"c{i:03d}"): grng.uniform(0.6, 1.1) for i in range(240)}
    g_ws._sim_per_pool_s = 1e-3  # a POPULATED EMA: 240 pools x 1 ms is far past the crossover
    g_par_tot, g_par_late = g_ws._simulate(g_drives)
    g_par_v = dict(g_ws._last_v)
    g_ranked_par = _ranked(g_ws, g_par_tot, g_par_late, g_par_v)
    g_ws._simulate_parallel = g_real
    g_ws._sim_per_pool_s = 0.0  # cold EMA -> the gate must refuse to guess and run SERIAL
    g_ser_tot, g_ser_late = g_ws._simulate(g_drives)
    g_ser_v = dict(g_ws._last_v)
    g_ranked_ser = _ranked(g_ws, g_ser_tot, g_ser_late, g_ser_v)
    # The thread COUNT is not printed and that is deliberate: `pick_thread_count` reads this
    # machine's measured barrier and dispatch floors, so it legitimately differs between processes
    # (16 and 8 observed on the same box, same seed). Printing it would make this module's own
    # output non-reproducible across runs while proving nothing the boolean does not. The count is
    # in the assert message, where it is only read on a failure.
    print(
        f"parallel gate: a populated cost EMA sent 240 pools down the PARALLEL branch "
        f"({len(gate_calls)} dispatch); the ranked admission order matches the serial settle on "
        f"{len(g_ranked_ser)} ignited pools"
    )
    assert gate_calls and gate_calls[0] >= 2, (
        f"the EMA-driven gate must ACTUALLY have taken the parallel branch -- a determinism claim "
        f"about a path that never ran is vacuous, and this is the failure mode the direct call in "
        f"(c2) cannot see. `pick_thread_count` returned {gate_calls}"
    )
    assert g_par_v == g_ser_v and (g_par_tot, g_par_late) == (g_ser_tot, g_ser_late), (
        "the gate-selected parallel settle must reproduce the serial MEMBRANE FLOATS bit for bit"
    )
    assert g_ranked_par == g_ranked_ser and len(g_ranked_ser) > cap, (
        f"...and therefore the RANK must be identical, which is what the membranes now decide: "
        f"{len(g_ranked_ser)} ignited pools ranked, first {cap} admitted"
    )

    # (c1b) WHAT `_rank_key` RESTS ON AT EXACTLY-EQUAL CLAIMS -- disclosed and RUN, not argued.
    # `_last_v` separates two co-saturating pools because the settle amplifies the difference in
    # their drives. When the drives are EXACTLY equal the settle has nothing to amplify and the only
    # thing left is `_JITTER`, this module's declared symmetry break. So the honest statement of the
    # repair is: it is name-blind at every claim, and at IDENTICAL claims its answer is the jitter's,
    # not the physics'. Both halves are measured here so neither can be over-read.
    #
    # THE FLOOR MATTERS AS MUCH AS THE HEADLINE. At `_JITTER = 0` two identical claims settle to
    # EXACTLY the same membrane, so `_rank_key` falls through to `k` and the name decides again --
    # on precisely the configs where nothing else can decide anything. That is a property of the
    # INPUT (two specialists asserting the identical claim), not a residual defect: no key over this
    # settle's outputs can separate two pools whose every output is bit-identical.
    def _equal_claim_gap(jitter, claim=0.70):
        w = GlobalWorkspace(seed=0)
        w._JITTER = jitter
        w.submit("aaa", "A", claim)
        w.submit("zzz", "B", claim)
        w.step()
        vs = list(w._last_v.values())
        gs = list(w._g.values())
        return abs(vs[0] - vs[1]), abs(gs[0] - gs[1]), w.current

    jit_dv, jit_dg, jit_head = _equal_claim_gap(GlobalWorkspace._JITTER)
    nojit_dv, nojit_dg, nojit_head = _equal_claim_gap(0.0)
    print(
        f"tie floor    : two pools at an IDENTICAL 0.70 settle {jit_dv:.3e} apart with the jitter "
        f"live (head {jit_head!r}) and {nojit_dv:.3e} apart at _JITTER=0 (head {nojit_head!r}); "
        f"their attractor charges are {jit_dg:.3e} apart in both"
    )
    assert jit_dv > 0.0 and jit_dg == 0.0, (
        f"the membrane must separate identical claims where the attractor charge cannot -- that is "
        f"the whole reason `_rank_key` reads `_last_v` and not `_g` (dv {jit_dv:.3e}, dg {jit_dg:.3e})"
    )
    assert nojit_dv == 0.0 and nojit_head is not None, (
        f"...and it must be DISCLOSED that at _JITTER=0 it cannot: identical claims settle to "
        f"identical membranes ({nojit_dv:.3e}), the key falls through to `k`, and the workspace "
        f"still admits a content rather than raising. If this ever stops being 0.0 the header's "
        f"'9.1% of the separation is jitter' paragraph is wrong and must be re-derived"
    )

    # (c1c) THE TWO BAND MAPPINGS -- one band, two questions, and the DIFFERENCE is the criterion.
    # `band_claim` is for a producer that is always present (a percept): FLOORED at the rheobase,
    # so it can never fall silent. `band_bid` is for a request that may be REFUSED: CENTRED on the
    # rheobase, so below-baseline signal LAPSES. Shipping the bid through the floored map is exactly
    # what this pass did first, and it took the faculty from `ignited` 96/120 to 120/120 -- a floored
    # map DELETES the lapse. That is the red this criterion exists to catch.
    bw = GlobalWorkspace(seed=0)
    b_lo, b_hi = bw._band()
    assert b_lo < b_hi, "the band must have room between audible and displaceable"
    # at its own running baseline: claim sits MID-band, bid sits EXACTLY on the rheobase
    assert abs(bw.band_claim(1.0, 1.0) - (b_lo + 0.5 * (b_hi - b_lo))) < 1e-12, (
        "a signal AT its own baseline must claim the MIDPOINT of the band"
    )
    assert abs(bw.band_bid(1.0, 1.0) - b_lo) < 1e-12, (
        "a bid AT its own baseline must land EXACTLY on the rheobase -- the anchor "
        "cortex/brain.py::_update_access has always described"
    )
    # THE PROPERTY: the bid can lapse, the claim cannot. Below-baseline signal, both maps.
    lapse_bid = bw.band_bid(0.2, 1.0)
    floor_claim = bw.band_claim(0.2, 1.0)
    assert lapse_bid < b_lo, (
        f"a BELOW-baseline bid must fall UNDER the rheobase and lapse (repetition suppression), "
        f"got {lapse_bid:.6f} against a rheobase of {b_lo:.6f}. If this ever passes it means the "
        f"bid was floored, which is the defect measured at ignited 96/120 -> 120/120"
    )
    assert floor_claim >= b_lo, (
        f"a percept CLAIM must stay audible however quiet: {floor_claim:.6f} < {b_lo:.6f}"
    )
    # BOTH are bounded ABOVE by the displaceable ceiling -- structurally, not by a later clamp.
    # This is the half the retired `rheobase + 0.5*dnov + 0.15*real` form did not have: it clamped
    # to [0,1], so a large signal could make an UNDISPLACEABLE claim.
    for huge in (10.0, 1e6):
        assert bw.band_claim(huge, 1e-9) <= b_hi + 1e-12, "a claim must never exceed the ceiling"
        assert bw.band_bid(huge, 1e-9) <= b_hi + 1e-12, "a bid must never exceed the ceiling"
    # DEGENERATE INPUTS (P1: a rewrite in this tree already shipped a mu/1e-12 singularity live).
    assert bw.band_claim(0.0, 0.0) == b_lo and bw.band_bid(0.0, 0.0) == b_lo, (
        "neither positive is NOT evidence of habituation -- both map to the rheobase, and the bid "
        "must NOT lapse there"
    )
    assert abs(bw.band_claim(0.0, 1.0) - b_lo) < 1e-12, "zero signal claims exactly the floor"
    assert abs(bw.band_claim(1.0, 0.0) - b_hi) < 1e-12, "zero baseline claims exactly the ceiling"
    # A bus with no room between audible and displaceable collapses both maps onto the floor.
    flat = GlobalWorkspace(seed=0, self_boost=0.30, hysteresis=0.30)
    f_lo, f_hi = flat._band()
    assert f_hi <= f_lo and flat.band_claim(5.0, 1.0) == f_lo and flat.band_bid(5.0, 1.0) == f_lo, (
        "a bus configured with no admissible band must not invent one"
    )
    print(
        f"  (c1c) band [{b_lo:.4f}, {b_hi:.4f}]: at-baseline claim={bw.band_claim(1.0, 1.0):.4f} "
        f"(mid) bid={bw.band_bid(1.0, 1.0):.4f} (rheobase); below-baseline bid={lapse_bid:.4f} "
        f"LAPSES while claim={floor_claim:.4f} stays audible; both capped at {b_hi:.4f}"
    )

    # (c2) PARALLEL SETTLING is BYTE-IDENTICAL to serial settling ------------------------------
    # The gate is measured, so force both paths explicitly here rather than hoping to land on one.
    p_ws = GlobalWorkspace(seed=0)
    prng = random.Random(4242)
    # The drive ceiling is 1.6, not the 1.2 it was before the inhibitory rheobase, and the reason
    # is the NON-VACUITY guard below: 240 near-identical pools under the rheobase burst and crush
    # in near-synchrony, and at a 1.2 ceiling micro-step 60 happens to land on the crushed phase
    # with EVERY final membrane floored at 0.0 -- which would make the float comparison this
    # section exists for compare nothing. Measured: ceiling 1.2 -> 0/240 nonzero finals,
    # 1.6 -> 102/240 (89 pools sustained). The comparison itself is range-agnostic.
    p_drives = {(f"s{i}", f"c{i}"): prng.uniform(0.3, 1.6) for i in range(240)}
    p_ws._sim_per_pool_s = 0.0  # cold EMA -> the gate must refuse to guess and run SERIAL
    ser_tot, ser_late = p_ws._simulate(p_drives)
    assert p_ws._sim_per_pool_s > 0.0, "a serial settle must record its own per-pool cost (P1)"
    from snn_data_lake.substrate import region_wiring as _rw

    assert _rw.pick_thread_count(0.0, mode="threads") == 1, (
        "with no measured cost the gate must return 1 (serial), never a guess"
    )
    ser_v = dict(p_ws._last_v)
    par_tot, par_late = p_ws._simulate_parallel(sorted(p_drives), p_drives, 4, _rw)
    par_v = dict(p_ws._last_v)
    assert par_tot == ser_tot and par_late == ser_late, (
        "parallel settling must reproduce the serial spike counts EXACTLY"
    )
    # The FLOAT trajectory, not the integer counts, is what polices the reduction. Counts are far
    # too coarse: a reassociated `a_inh` moves the membranes in the last bits and the spike counts
    # absorb it silently -- measured, which is why this assert exists in this form.
    assert par_v == ser_v, (
        "parallel settling must reproduce the serial MEMBRANE FLOATS bit for bit -- this is what "
        "fails the moment the shared inhibitory reduction is reassociated"
    )
    assert list(par_tot) == list(ser_tot), (
        "the parallel path must return keys in sorted-chunk order, not thread-completion order"
    )
    # ...and the reassociation this guards against is REAL, not hypothetical: on this very data the
    # per-chunk-subtotal form gives a different float. If this ever stops differing, the fold above
    # is protecting nothing and the assert on `par_v` is decoration.
    adversarial = [[1e16, 1.0], [-1e16, 1.0]]
    assert _fold_parts(adversarial) == 2.0 and sum(sum(c) for c in adversarial) == 0.0, (
        _fold_parts(adversarial),
        sum(sum(c) for c in adversarial),
    )
    # The reduction must equal the FLAT sum for every partitioning -- that is the whole determinism
    # claim -- and must differ from both wrong forms, or the claim is vacuous.
    rrng = random.Random(31)
    flat = [rrng.uniform(-2.0, 2.0) for _ in range(240)]
    naive = 0.0
    for x in flat:
        naive += x
    for k in (1, 2, 3, 7, 16):
        step = max(1, len(flat) // k)
        chunks = [flat[i : i + step] for i in range(0, len(flat), step)]
        assert _fold_parts(chunks) == sum(flat), (
            f"the reduction must be independent of the partition ({k} chunks disagreed)"
        )
    assert naive != sum(flat), (
        "an `acc += x` fold must actually differ from sum() here -- if it stops differing, this "
        "module's determinism argument has lost the case it was written to catch"
    )
    assert _fold_parts(()) == 0 and _fold_parts(([],)) == 0, "empty fold must be zero"

    # A FROZEN REFERENCE of the pre-parallel dict-based settle -- an independent second
    # implementation, so `_settle_chunk`'s arithmetic is checked against something other than
    # itself. Serial-vs-parallel equality can never catch a change made to the SHARED kernel; this
    # can.
    def _reference_settle(ws, drives):
        keys = sorted(drives)
        v = {k: 0.0 for k in keys}
        trace = {k: 0.0 for k in keys}
        self_i = {k: 0.0 for k in keys}
        a_inh = 0.0
        total = {k: 0 for k in keys}
        late = {k: 0 for k in keys}
        half = ws._T // 2
        for t in range(ws._T):
            spikes = {}
            for k in keys:
                # mirrored LONGHAND, deliberately not pasted from `_settle_chunk`: this reference
                # is the only check that can see a change to the shared kernel, and a copy-paste
                # would retire it. `if/else` here vs `max()` there -- verified bit-identical on
                # +0.0, -0.0 and negatives.
                others = a_inh - self_i[k]
                excess = others - ws._inh_free
                lateral = ws._g_lat * excess if excess > 0.0 else 0.0
                cur = drives[k] + ws._W_SELF * trace[k] - lateral
                v[k] += cur - ws._LEAK * v[k]
                if v[k] < 0.0:
                    v[k] = 0.0
                s = 0
                if v[k] >= ws._v_th:
                    v[k] -= ws._v_th
                    s = 1
                    total[k] += 1
                    if t >= half:
                        late[k] += 1
                    trace[k] += 1.0
                spikes[k] = s
                trace[k] *= ws._TAU_SELF
            for k in keys:
                self_i[k] += ws._K_INH * spikes[k] - ws._LEAK_INH * self_i[k]
                if self_i[k] < 0.0:
                    self_i[k] = 0.0
            a_inh = sum(self_i.values())
        return total, late, v

    ref_tot, ref_late, ref_v = _reference_settle(p_ws, p_drives)
    assert (ref_tot, ref_late) == (ser_tot, ser_late), "settle must match the reference counts"
    assert ref_v == ser_v, (
        "the partitioned settle kernel must reproduce the reference dict implementation's MEMBRANE "
        "FLOATS exactly -- any re-association inside the LIF update fails here"
    )
    assert any(x != 0.0 for x in ser_v.values()), (
        "the membrane comparison must be non-vacuous (all-zero membranes would compare equal to "
        "anything that also zeroes them)"
    )
    # zero / one: the partition boundaries the gate must survive
    assert p_ws._simulate({}) == ({}, {}), "an empty workspace must settle to empty, not raise"
    one = {("solo", "solo"): 0.9}
    assert p_ws._simulate_parallel(sorted(one), one, 1, _rw) == p_ws._simulate(one), (
        "a single pool must give the same answer however it is partitioned"
    )

    # REACHABLE RED for the worker-failure path. Every chunk meets every other at two barriers per
    # micro-step, so a chunk that raises stops arriving and its partners wait on a barrier that can
    # never again reach `nb` parties. MEASURED before the abort/re-raise guard existed: this exact
    # call hung FOREVER, and because the workers are non-daemon the interpreter could not exit
    # either. Reaching this assert at all is the proof it no longer can.
    poisoned = {(f"s{i:02d}", f"c{i:02d}"): 0.9 for i in range(16)}
    poisoned[("s09", "c09")] = "not-a-float"  # index 9 -> chunk 2 of 4, i.e. a WORKER, not the lead
    assert _rw.blocks(16, 4)[2] == (8, 12), _rw.blocks(16, 4)
    try:
        GlobalWorkspace(seed=0)._simulate_parallel(sorted(poisoned), poisoned, 4, _rw)
        raise AssertionError("a chunk that raises must PROPAGATE -- never hang, never return")
    except TypeError as exc:
        # the ROOT cause, not the BrokenBarrierError the abort induced in the three live chunks
        assert "str" in str(exc), f"the root cause must survive, got {exc!r}"

    # (c3) CONCURRENT SUBMISSION is DETERMINISTIC ----------------------------------------------
    # RED for the ordering claim: buffer order IS observable, so the slot ordering is load-bearing.
    o1, o2 = GlobalWorkspace(seed=0), GlobalWorkspace(seed=0)
    o1.submit("alpha", "A", 0.8)
    o1.submit("beta", "B", 0.8)
    o2.submit("beta", "B", 0.8)  # same two candidates, opposite arrival order
    o2.submit("alpha", "A", 0.8)
    o1.step()
    o2.step()
    assert o1._jit != o2._jit, (
        "arrival order must be OBSERVABLE (it seeds the symmetry-break jitter) -- if this ever "
        "passes trivially, the slot ordering below is protecting nothing and should be deleted"
    )

    def _slot_trial(threaded):
        w = GlobalWorkspace(seed=0)
        n_slots = 8
        w.open_slots(n_slots)
        r = random.Random(11)
        payload = [
            [(f"s{t}_{j}", f"c{t}_{j}", r.uniform(0.3, 1.1)) for j in range(5)]
            for t in range(n_slots)
        ]
        if threaded:
            gate = threading.Barrier(n_slots)

            def sub(t):
                gate.wait()  # release all submitters together to maximise real interleaving
                for src, con, sal in payload[t]:
                    w.submit_from(t, src, con, sal)

            ths = [threading.Thread(target=sub, args=(t,)) for t in range(n_slots)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
        else:
            for t in range(n_slots):
                for src, con, sal in payload[t]:
                    w.submit_from(t, src, con, sal)
        return w.step(), tuple(sorted(w._jit.items()))

    # The merge ORDER is the contract, and it is asserted directly -- 'both paths agree' would still
    # hold if the drain sorted or reversed the buffers, so agreement alone does not pin the order.
    ord_ws = GlobalWorkspace(seed=0)
    ord_ws.open_slots(3)
    ord_ws.submit_from(2, "c", "c1", 0.9)  # slot 2 submits FIRST in real time...
    ord_ws.submit_from(0, "a", "a1", 0.1)
    ord_ws.submit_from(2, "c", "c2", 0.8)
    ord_ws.submit_from(1, "b", "b1", 0.5)
    ord_ws._drain_slots()
    assert ord_ws._buffer == [
        ("a", "a1", 0.1),  # ...but slot 0 lands first: ascending SLOT, then program order
        ("b", "b1", 0.5),
        ("c", "c1", 0.9),
        ("c", "c2", 0.8),
    ], f"slot drain must be ascending-slot then program order, got {ord_ws._buffer}"
    assert ord_ws._slot_buf == [[], [], []], "drained buffers must be cleared, not re-delivered"
    assert GlobalWorkspace(seed=0).open_slots(0) == 0, "zero slots is legal and drains to nothing"

    ref_trial = _slot_trial(False)
    n_trials = 40
    all_same = all(_slot_trial(True) == ref_trial for _ in range(n_trials))
    assert all_same, (
        "concurrent submission through the slot buffers must equal the one-at-a-time result on "
        "EVERY trial -- a single mismatch means thread order reached the competition"
    )
    print(
        f"\nparallel     : settling 240 pools on 4 threads reproduces the serial spike counts "
        f"exactly; {n_trials} threaded submission trials all matched the serial order"
    )

    # (d) THE BINDING FAMILY IS GONE. Its margin criteria were deleted first (P8/F11 -- the
    # recall-only scorer on which an all-pairs strawman scores 1.0000; the SECOND deletion of the
    # same criterion, `reasoning/consciousness_probe.py` having deleted its copy 2026-08-04), and
    # the honest precision-bearing benchmark then returned the pre-registered NULL, so the whole
    # family -- functions, constants, harness, and the ignition assert that rode on them --
    # retired under P8 burden 5. The measurement and the disposition are the retirement record
    # above `main()`.

    # (e) THE DECLARED READ SURFACE: `current` says WHAT, `broadcast_hold` says HOW STRONGLY -----
    # RE-DERIVED 2026-08-06, ONTO A DIFFERENT INPUT, AND THE RE-DERIVATION IS THE POINT. This
    # section used to drive the dissociation with CYCLING CONTENT at a constant salience and assert
    # that `current` reported ONE distinct value over 120 ticks. That assertion was true, and what
    # made it true was the ENTRENCHMENT `_defended` now closes -- a superseded content held the
    # workspace against its own author's fresh claim. So the criterion was measuring the defect and
    # scoring it as the feature, and its own failure message said what to do about that ("re-derive
    # the whole section rather than keeping it"); it fired on exactly the input it named. It is
    # re-derived here, NOT weakened: the assertion `current` reports 1 distinct value / 0 changes is
    # kept VERBATIM and moved onto the input where it is a statement about the read surface --
    # ONE content re-asserted at a VARYING claim, which is what a specialist that keeps attending
    # the same thing actually does. That input also makes the A/B one-variable for the first time:
    # the RED below is the same content at a CONSTANT claim, so green and red now differ ONLY in
    # whether the claim moves, where before they differed in the content cycling as well.
    e_ws = GlobalWorkspace(seed=0)
    assert e_ws.current is None and e_ws.broadcast_hold == 0.0 and e_ws.broadcast_all() == (), (
        "an EMPTY workspace must read as (None, 0.0, ()) -- never raise, never a stale number, and "
        "the multi-slot accessor must give an empty TUPLE rather than None so a caller may iterate"
    )
    e_lo = e_ws.ignition_threshold  # the claim band this module is written against, read off itself
    e_hi = 1.0 - (e_ws.self_boost + e_ws.hysteresis)
    hold_seq, content_seq = [], []
    for t in range(120):
        # ONE content, a claim that MOVES inside the compliant band: WHAT is pinned, HOW STRONGLY
        # is not. Re-asserting the same content is the branch `_defended` leaves untouched.
        e_ws.submit("perception", "same", e_lo + (e_hi - e_lo) * ((t % 7) / 6.0))
        e_ws.step()
        content_seq.append(e_ws.current)
        hold_seq.append(e_ws.broadcast_hold)
    hold_changes = count_switches(hold_seq)
    content_changes = count_switches(content_seq)
    assert content_changes == 0 and len(set(content_seq)) == 1, (
        "this criterion is only about the DISSOCIATION -- if the content route has started varying "
        f"on this input, re-derive the whole section rather than keeping it (got {content_changes} "
        f"changes over {sorted(set(content_seq))})"
    )
    assert hold_changes >= 100 and len(set(hold_seq)) >= 3, (
        f"the slot's hold must be readable AND moving where the content route is frozen: "
        f"{hold_changes} changes, {len(set(hold_seq))} distinct in 120 ticks"
    )
    assert all(0.0 <= h <= 1.0 for h in hold_seq), (
        "broadcast_hold must stay inside its [0,1] contract"
    )

    # REACHABLE RED for that criterion: the same 120 ticks with the claim held CONSTANT. `g` is then
    # pinned at the submitted salience and the hold cannot move -- so the assertion above is
    # falsifiable by an input this module accepts, not merely by breaking the accessor.
    red_ws = GlobalWorkspace(seed=0)
    red_hold = []
    for _ in range(120):
        red_ws.submit("perception", "same", 0.6)  # constant claim -> no decay, no movement
        red_ws.step()
        red_hold.append(red_ws.broadcast_hold)
    assert count_switches(red_hold) == 0 and len(set(red_hold)) == 1, (
        f"the RED must actually be red: a never-decaying hold must report 0 changes, got "
        f"{count_switches(red_hold)} over {sorted(set(red_hold))}"
    )

    # (e2) THE CONTENT ROUTE TRACKS A SUPERSEDING SOURCE -- the criterion `_defended` owes, on the
    # input the block above was retired from. A source that posts a DIFFERENT content is retracting
    # the one it was posting, so the workspace must follow it rather than subsidise the retracted
    # claim. Asserted as CORRESPONDENCE and not as variety: `current` must BE the content submitted
    # this tick, on every tick. A route that merely CHANGES would pass a workspace cycling through
    # stale contents in some other order, which is the thing this claim must be able to reject.
    e2_seq = []
    e2_ws = GlobalWorkspace(seed=0)
    for t in range(120):
        e2_ws.submit("perception", t % 3, 0.6)  # constant claim, CYCLING content: the live shape
        e2_ws.step()
        e2_seq.append(e2_ws.current)
    assert e2_seq == [t % 3 for t in range(120)], (
        f"a superseded content must not hold the workspace against its own author: `current` must "
        f"name the content submitted this tick on all 120, got {count_switches(e2_seq)} changes "
        f"over {sorted(set(e2_seq))}"
    )
    # ITS RED, CONSTRUCTED AND RUN: the same 120 ticks with the pre-fix tenancy restored, where
    # EVERY occupant is defended unconditionally -- ON THE PRE-RHEOBASE KERNEL, and the kernel is
    # not decoration. Measured when the rheobase landed: unconditional tenancy on the SHIPPED
    # kernel no longer entrenches anything (119 changes -- the subsidy is symmetric across
    # occupants, so g-recency still decides the head), and neither does it on the one-subsidy
    # arbiter (118). The entrenchment's ENFORCEMENT was always the annihilation: the defended
    # incumbent's inhibition held the fresh claim below ignition, and only `NoAllowanceArbiter`
    # restores that route. So the red is the arbiter that actually produced the live measurement
    # in `_defended`'s docstring -- retired tenancy on the retired kernel -- and a red that
    # reconstructs the retired behaviour on only ONE of the two paths it ran through is green for
    # the wrong reason (the lesson `_NameRanked` records, met again from the other side).
    e2_red_ws = NoAllowanceArbiter(seed=0)
    e2_red_ws._defended = lambda submitted: {(s, c) for s, c, _h in e2_red_ws._slots}
    e2_red = []
    for t in range(120):
        e2_red_ws.submit("perception", t % 3, 0.6)
        e2_red_ws.step()
        e2_red.append(e2_red_ws.current)
    assert count_switches(e2_red) == 0 and len(set(e2_red)) == 1, (
        f"the RED must actually be red: under unconditional tenancy the first content must hold all "
        f"120 ticks, got {count_switches(e2_red)} changes over {sorted(set(e2_red))}"
    )

    # (e3) THE DEFENCE FOLLOWS THE SOURCE ON EVERY SLOT, NOT JUST ON THE HEAD -- the half of
    # `_defended`'s rule that (e2) structurally cannot reach. (e2) drives ONE source, so its
    # workspace never holds more than one content and nothing above this line ever presents
    # `_defended` with an occupant that is not the head. MEASURED gap this closes, by mutating and
    # RUNNING rather than by reading: a tenancy that withholds the defence from the head occupant
    # only and leaves every lower-ranked occupant subsidised passed the ENTIRE rest of this
    # self-check, while producing a different workspace on 20 of 40 ticks of a five-source
    # superseding load (SHIPPED admitted ('a2',) where the mutant admitted ('b1', 'a2') -- 'b1'
    # being a content its own source had already retracted). `_defended` is a pure function of
    # (`_slots`, `submitted`), so the rule is asserted here DIRECTLY on a four-slot state carrying
    # all three documented cases at once, with a superseded occupant BELOW the head.
    e3_slots = (
        ("a", "old_a", 0.9),  # slot 0, the HEAD: its source posts something else -> superseded
        ("b", "keep_b", 0.8),  # slot 1: its source re-asserts this content       -> defended
        ("c", "old_c", 0.7),  # slot 2: superseded, and NOT the head             -> superseded
        ("d", "quiet_d", 0.6),  # slot 3: its source is silent this tick           -> defended
    )
    e3_sub = {("a", "new_a"): 0.9, ("b", "keep_b"): 0.8, ("c", "new_c"): 0.7}
    e3_ws = GlobalWorkspace(seed=0)
    e3_ws._slots = e3_slots
    assert e3_ws._defended(e3_sub) == {("b", "keep_b"), ("d", "quiet_d")}, (
        f"the tenancy defence must follow the SOURCE on every slot -- a re-asserted or silent "
        f"occupant keeps it and a superseded one loses it wherever it sits in the rank; got "
        f"{sorted(map(repr, e3_ws._defended(e3_sub)))}"
    )
    # ITS RED, CONSTRUCTED AND RUN: the same state under a tenancy that withholds the defence from
    # the head occupant ONLY. That is the mutant this criterion exists to reject, and until this
    # line it had no reachable red anywhere in this file.
    e3_red_ws = GlobalWorkspace(seed=0)
    e3_red_ws._slots = e3_slots
    e3_red_ws._defended = lambda submitted: {
        (s, c)
        for i, (s, c, _h) in enumerate(e3_red_ws._slots)
        if i > 0 or (s, c) in submitted or s not in {src for src, _content in submitted}
    }
    assert ("c", "old_c") in e3_red_ws._defended(e3_sub), (
        "the RED must actually be red: a head-only tenancy must still subsidise the superseded "
        "content sitting BELOW the head"
    )

    # BOUNDARIES -- zero, silence, single element, and the control path's out-of-range salience.
    # The DELIVERY is walked over the same four, because a subscriber sees only what it is handed:
    # if the empty tick delivered None (or nothing at all) every consumer would need a per-tick
    # special case for `mine in contents`, and the one that forgot would read a stale set.
    d_seen = []
    d_ws = GlobalWorkspace(seed=0)
    d_ws.subscribe(d_seen.append)
    d_ws.step()  # EMPTY: nothing submitted at all
    d_ws.submit("perception", "nothing", 0.0)
    d_ws.step()  # ZERO: a submission that cannot ignite
    d_ws.submit("perception", "one", 0.9)
    d_ws.step()  # SINGLE: exactly one admitted content
    for _ in range(10):
        d_ws.step()  # SILENCE: decay, then release
    assert d_seen[0] == () and d_seen[1] == () and d_seen[2] == ("one",), (
        f"delivery corners: EMPTY and ZERO must broadcast the empty TUPLE (never None), SINGLE "
        f"must broadcast a 1-tuple. Got {d_seen[:3]}"
    )
    assert d_seen[-1] == () and all(type(c) is tuple for c in d_seen), (
        f"SILENCE must broadcast the empty tuple once the workspace releases, and every delivery "
        f"must be a tuple: got {d_seen[-1]!r} from {len(d_seen)} deliveries"
    )
    assert len(d_seen) == 13, f"one delivery per step(), got {len(d_seen)} over 13 steps"
    z_ws = GlobalWorkspace(seed=0)
    z_ws.submit("perception", "nothing", 0.0)
    z_ws.step()
    assert z_ws.current is None and z_ws.broadcast_hold == 0.0 and z_ws.broadcast_all() == (), (
        "a zero-salience submission must leave the workspace empty and the hold at exactly 0.0"
    )
    assert GlobalWorkspace(seed=0).step() is None, "an EMPTY tick must broadcast nothing, not raise"
    s_ws2 = GlobalWorkspace(seed=0)
    s_ws2.submit("solo", "solo", 0.9)
    s_ws2.step()
    assert s_ws2.broadcast_hold == 0.9 and s_ws2.broadcast_all() == ("solo",), (
        f"a single ignited pool fills ONE of {cap} slots and its hold is its own attractor charge, "
        f"got {s_ws2.broadcast_all()} / {s_ws2.broadcast_hold}"
    )
    sil = []
    for _ in range(10):
        s_ws2.step()  # SILENCE: nothing refreshes it -> decay, then release
        sil.append((s_ws2.current, s_ws2.broadcast_hold))
    assert sil[-1] == (None, 0.0) and s_ws2.broadcast_all() == (), (
        f"a released workspace must read as (None, 0.0, ()), not a stale charge, got {sil[-1]}"
    )
    assert [h for _c, h in sil][0] > [h for _c, h in sil][6] > 0.0, (
        "the hold must DECAY monotonically through silence before the release"
    )
    # ...and THAT criterion is falsifiable by an input this module accepts rather than only by
    # breaking the accessor: at decay=1.0 the attractor never fades, so the hold is flat through
    # silence and the slot is never released. Run, not argued.
    nd_ws = GlobalWorkspace(seed=0, decay=1.0)
    nd_ws.submit("memory", "trace", 0.95)
    nd_ws.step()
    nd_hold = [nd_ws.broadcast_hold]
    for _ in range(10):
        nd_ws.step()
        nd_hold.append(nd_ws.broadcast_hold)
    assert nd_hold[0] == nd_hold[6] and nd_ws.current == "trace", (
        f"the decay criterion must be reachable-red: at decay=1.0 the hold must stay flat and the "
        f"slot must never release, got {nd_hold[:7]} / {nd_ws.current!r}"
    )
    ctl = GlobalWorkspace(seed=0, use_ignition=False)
    ctl_seen = []
    ctl.subscribe(ctl_seen.append)
    ctl.submit("perception", "big", 5.0)
    assert ctl._top_candidate()[2] == 5.0, (
        "the clamp in step() must have something to clamp -- if submissions can no longer exceed "
        "1.0 this guard is decoration and should be deleted"
    )
    ctl.step()
    assert ctl.current == "big" and ctl.broadcast_hold == 1.0 and ctl.broadcast_all() == ("big",), (
        f"the control path must honour the same [0,1] hold contract and hold ONE content, got "
        f"{ctl.broadcast_all()} / {ctl.broadcast_hold}"
    )
    assert ctl.step() is None and ctl.broadcast_all() == (), (
        "an empty tick must EMPTY the control path too -- one emptying semantics per accessor, "
        "not one per path (this path used to leave the previous winner in place)"
    )
    # WHAT THIS CRITERION CAN AND CANNOT CATCH, stated because the difference is not obvious and a
    # reader will otherwise credit it with more than it does. Its red is the delivery SHAPE and
    # that red is REACHABLE and was run: delivering the bare `content` here gives ['big', None] and
    # delivering `()` gives [(), ()], both red. What it CANNOT catch is a head-only regression,
    # because on the argmax path `_slots` is rebound to at most ONE slot (see `step()`), so
    # `broadcast_all()` is always a 0- or 1-tuple and head-vs-set is structurally unreachable here.
    # MEASURED: a mutant reverting this line to `(content,)` is GREEN on the whole module. The
    # set-carrying half of the contract is covered by `delivery_ok` and `b_beyond_head` on the
    # IGNITION path, which is the only path that can hold more than one content.
    assert ctl_seen == [("big",), ()], (
        f"ONE delivery contract, BOTH paths: the argmax control must broadcast the same tuple "
        f"SHAPE the ignition path does -- a subscriber may not need to know which arbiter ran. "
        f"Got {ctl_seen}"
    )

    # RACE: the workspace REBOUND to empty between two loads -- what a concurrent step() does on
    # cp314t. The red is run, not argued: a two-load read raises IndexError on this exact
    # interleaving and the single-load form cannot. Multiple slots make this MORE important, not
    # less: `_slots` is a tuple swapped whole, so a reader still sees one consistent workspace.
    class _RacingSlot(GlobalWorkspace):
        """`_slots` is emptied by a 'concurrent step()' immediately after the FIRST load."""

        def __init__(self, **kw):
            self._race = ()
            super().__init__(**kw)

        @property
        def _slots(self):
            v = self._race
            self._race = ()  # the racing step() lands here, between the caller's two loads
            return v

        @_slots.setter
        def _slots(self, v):
            self._race = v

    def _two_load_current(ws):
        """A two-load read of the workspace -- kept ONLY as the control."""
        return ws._slots[0][1] if ws._slots else None

    def _two_load_all(ws):
        """The same mistake on the multi-slot accessor."""
        return tuple(e[1] for e in ws._slots) if ws._slots else ()

    race = _RacingSlot(seed=0)
    race._slots = (("perception", "scene", 0.7), ("memory", "trace", 0.4))
    assert race.current == "scene", "a single-load read must survive a workspace emptied under it"
    race._slots = (("perception", "scene", 0.7), ("memory", "trace", 0.4))
    assert race.broadcast_hold == 0.7, "broadcast_hold must take the same single snapshot"
    race._slots = (("perception", "scene", 0.7), ("memory", "trace", 0.4))
    assert race.broadcast_all() == ("scene", "trace"), (
        "broadcast_all must take ONE snapshot of the whole workspace -- a partially-observed set "
        "is the failure mode multiple slots introduce"
    )
    race._slots = (("perception", "scene", 0.7),)
    try:
        _two_load_current(race)
        raise AssertionError(
            "the two-load read must FAIL on this interleaving -- if it no longer can, `_slot()` is "
            "protecting nothing and its concurrency argument should be deleted"
        )
    except IndexError:
        pass  # the RED, reached: tuple index out of range
    race._slots = (("perception", "scene", 0.7),)
    assert _two_load_all(race) == (), (
        "the two-load multi-slot read must silently report an EMPTY workspace where the single-load "
        "form reports the real one -- a wrong answer rather than a raise, which is why it is worse"
    )
    # THE SLOT SET MUST BE IMMUTABLE, and this criterion was ADDED because a mutation harness found
    # that nothing here could see it: rebinding `_slots` as a LIST passed every other assert in this
    # file while destroying the entire concurrency argument above. On cp314t a list is mutated in
    # place, so a reader can observe a half-filled workspace and `_slot()`'s "either the whole
    # previous set or the whole new one" becomes false -- and multiple slots is precisely where that
    # goes wrong, because there is now a set to be half-observed. `hash()` is the check rather than
    # `isinstance`: it proves immutability all the way DOWN, so a tuple of lists fails it too.
    imm = GlobalWorkspace(seed=0)
    for sal in (0.90, 0.80, 0.70, 0.60):
        imm.submit(f"s{sal}", f"c{sal}", sal)
    imm.step()
    assert len(imm._slots) > 1, (
        "the immutability criterion must be non-vacuous -- a one-element or empty workspace would "
        f"satisfy it trivially, got {imm.broadcast_all()}"
    )
    hash(imm._slots)  # RED: TypeError the moment any level of the slot set becomes mutable
    assert type(imm._slots) is tuple and all(type(e) is tuple for e in imm._slots), (
        f"the workspace must be a TUPLE of TUPLES rebound whole, never a container mutated in "
        f"place -- got {type(imm._slots).__name__} of "
        f"{sorted({type(e).__name__ for e in imm._slots})}"
    )
    try:
        imm._slots[0] = ("x", "x", 0.0)
        raise AssertionError(
            "the workspace must REFUSE in-place assignment -- if it accepts one, `_slot()`'s "
            "lock-free snapshot argument is false and should be deleted rather than believed"
        )
    except TypeError:
        pass  # the RED, reached: 'tuple' object does not support item assignment
    print(
        f"\nread surface : over 120 ticks of ONE content at a MOVING claim the content route "
        f"reports {len(set(content_seq))} distinct value(s) / {content_changes} changes while "
        f"broadcast_hold reports {len(set(hold_seq))} / {hold_changes}; constant-claim RED "
        f"reports {len(set(red_hold))} / {count_switches(red_hold)}"
    )
    print(
        f"tenancy      : on CYCLING content the route now TRACKS its source -- "
        f"{sum(1 for t, c in enumerate(e2_seq) if c == t % 3)}/120 ticks name the content just "
        f"submitted ({len(set(e2_seq))} distinct / {count_switches(e2_seq)} changes); the pre-fix "
        f"unconditional-tenancy RED reports {len(set(e2_red))} / {count_switches(e2_red)}"
    )
    print(
        f"             : and it follows the SOURCE on EVERY slot -- a 4-slot state of "
        f"superseded/re-asserted/superseded/silent occupants defends "
        f"{len(e3_ws._defended(e3_sub))} of 4, where the head-only RED defends "
        f"{len(e3_red_ws._defended(e3_sub))}"
    )

    # ---- T031 COALITION RECORDS: the record layer over the untouched kernel --------------------
    # distinct-from: thalamus/reentrant_workspace.py -- its step() groups submissions into
    #   coalitions for the RE-ENTRY kernel; this block is a SELF-CHECK harness driving THIS
    #   module's record layer end-to-end. distinct-from: substrate/port_charge_probe.py -- the
    #   replay-twice charge A/B instrument; nothing here measures charge. distinct-from:
    #   cortex/pc_cortex_serve.py -- its settle is the predictive-coding hierarchy relaxation.
    #   distinct-from: cerebellum/spiking_jepa.py and reasoning/arc_agi/brain_player.py --
    #   phrase overlap only (their drives run retinas/games, not lake framing). All sites read.
    # Four claims, each with its red reachable IN THIS BLOCK: (a) the record IS the 3-tuple to
    # every kernel consumer (projection identity -- mutate as_tuple and every assert here goes
    # red); (b) lake round-trip is byte-exact; (c) ring -> journaller -> lake -> a FRESH BinLog
    # reader recovers the settled records (P4b: persistence proven on a NEW reader, never the
    # writer's own memory); (d) refusals: a channel set without 'salience' raises, and identity
    # stamping is deterministic (two same-seed runs journal identical bytes).
    import tempfile
    from pathlib import Path

    from snn_data_lake.substrate.binlog import BinLog
    from snn_data_lake.substrate.home import _T_COALITION
    from snn_data_lake.substrate.signal_ring import LakeJournaller, SignalRing

    c_rec = Coalition(["perception"], "scene", 0.7, {"salience": 0.7, "arousal": 0.2})
    assert tuple(c_rec) == c_rec.as_tuple() == ("perception", "scene", 0.7) and len(c_rec) == 3, (
        "the record must project EXACTLY the legacy 3-tuple -- the kernel body is unchanged by "
        "T031 and consumes records only through this projection"
    )
    assert c_rec.tick is None and c_rec.current is None and c_rec.coalition_id is None, (
        "an unsettled record must be honestly unstamped -- None, never a defaulted 0"
    )
    c_rt = Coalition.from_bytes(c_rec.to_bytes())
    assert c_rt.to_bytes() == c_rec.to_bytes(), "lake round-trip must be byte-exact"
    assert c_rt.channels == {"salience": 0.7, "arousal": 0.2}, (
        "typed channels must survive the lake"
    )

    # distinct-from: cortex/access_consciousness.py and cortex/live_brain.py -- their "one
    #   workspace" is the LIVE composite; this closure is a self-check harness driving a
    #   scratch lake through restarts. Phrase overlap only; sites read.
    def _restart_life(base=None):
        """One workspace LIFETIME against the shared scratch lake: fresh instance, base
        ordinal derived from the lake exactly as the arm derives it (replay count of the
        tag), unless the caller FORCES a base to reach the retired scheme's red below."""
        log = BinLog(str(c_work / "lake.bin"))
        if base is None:
            base = sum(1 for p in log.replay() if p[:1] == _T_COALITION)
        ws = GlobalWorkspace(seed=0)
        ring = SignalRing(slots=32)
        ws.attach_ring(ring, base_ordinal=base)
        journal = LakeJournaller(ring, log)
        ws.submit("perception", "scene", 0.90)
        ws.submit_coalition("salience", "threat", {"salience": 0.85, "valence": -0.6})
        ws.step()
        return ws, journal.drain()

    c_work = Path(tempfile.mkdtemp(prefix="gw_coalition_"))
    c_ws, c_wrote = _restart_life()
    assert c_wrote == 2 and len(c_ws.settled) == 2 and all(r.tick == 1 for r in c_ws.settled), (
        f"both admitted posts must settle AND persist, got settled={len(c_ws.settled)} wrote={c_wrote}"
    )
    assert {r.channels.get("valence") for r in c_ws.settled} == {None, -0.6}, (
        "submit_coalition's typed channels must reach the settled record"
    )
    c_back = [p for p in BinLog(str(c_work / "lake.bin")).replay() if p[:1] == _T_COALITION]
    assert len(c_back) == 2, f"a FRESH reader must recover 2 journaled records, got {len(c_back)}"
    c_rec2 = [Coalition.from_bytes(p[1:]) for p in c_back]
    assert {r.content_ref for r in c_rec2} == {"scene", "threat"} and {
        r.coalition_id for r in c_rec2
    } == {r.coalition_id for r in c_ws.settled}, (
        "the lake's records must be the settle's records -- same identities, same contents"
    )
    # RESTART against the SAME lake: a second life's records must ALL land, with identities
    # the lake has never held. This arm replaced a "two same-seed runs journal identical
    # bytes" assert that CERTIFIED the defect it should have caught: identical bytes into a
    # dedup=True lake is precisely what made a restarted life persist NOTHING (review,
    # measured: 16 of 24 restart records silently dropped under the instance-derived scheme).
    c_ws_b, c_wrote_b = _restart_life()
    c_all = [
        Coalition.from_bytes(p[1:])
        for p in BinLog(str(c_work / "lake.bin")).replay()
        if p[:1] == _T_COALITION
    ]
    assert c_wrote_b == 2 and len(c_all) == 4, (
        f"RESTART AMNESIA: the second life wrote {c_wrote_b}, lake holds {len(c_all)} of 4 -- "
        "a restart is not amnesia (P4b)"
    )
    assert [r.coalition_id for r in c_all] == [f"c{i}" for i in range(4)], (
        f"identity must be the lake-continued ordinal, unique across lives, got "
        f"{[r.coalition_id for r in c_all]}"
    )
    # THE RETIRED SCHEME'S RED, kept reachable: force the process-local base (always 0) for
    # a third life and the lake drops EVERY record it re-mints -- the measured defect the
    # lake-derived base retires. If this ever starts writing, the dedup semantics changed
    # and the identity scheme must be re-derived, loudly.
    _c_ws_c, c_wrote_c = _restart_life(base=0)
    assert c_wrote_c == 0, (
        f"the instance-derived-identity red is no longer reachable (wrote {c_wrote_c}); "
        "re-derive the identity scheme against the lake's actual dedup semantics"
    )
    try:
        c_ws_b.submit_coalition("x", "y", {"arousal": 1.0})
        raise AssertionError("submit_coalition without 'salience' must refuse -- red unreachable")
    except ValueError:
        pass  # the RED, reached: no drive channel, no competition entry

    print(
        f"\ncoalition    : record==tuple projection holds; round-trip byte-exact; restart "
        f"against one lake persists 4/4 with lake-continued ids {[r.coalition_id for r in c_all]}; "
        f"the retired process-local scheme demonstrably loses its life (wrote 0); salience-less "
        f"refusal reached"
    )

    print(
        "\nself-check OK: "
        f"capacity {cap} (derived from defence {band_hi and 1.0 - band_hi:.2f}); churn {churn_with} "
        f"vs argmax {churn_without} (>=3x stabler), previous broadcast retained {kept_with}/"
        f"{len(with_ig) - 1} vs {kept_without}; bottleneck {b_admitted} of {b_ignited} ignited; "
        f"WINNABLE at {max(w for w in win_new):.3f} <= band ceiling {band_hi:.2f} where the retired "
        f"single-slot arbiter needed {win_old}; competition still excludes losers with slots free; "
        f"all-or-none (sub spikes={weak_spikes}, supra spikes={strong_spikes}); global-reach; "
        f"persist {persisted}->release; threat ignited; live-contract holds; deterministic over 24 "
        f"runs; BINDING family RETIRED on the honest benchmark's null (spiking Jaccard == the "
        f"all-fired-pairs strawman at every K; see the retirement record above main())"
    )


def declare_ports(manifest):
    """Manifest registrant (specs/022 T030, the workspace instance resolution): THE
    declared composite. The resolution is a composite WITH ROLES, not one object -- the
    measured record forbids naive unification (spike_chat_lake's own recorded experiment:
    repointing its per-turn arena at this bus returned the arm's unit index as a chat
    reply and destroyed the producer's broadcast):
      * THE SHARED BUS is the arm-owned instance (autonomous_loop constructs it;
        Brain.workspace / LiveBrain.workspace are ADOPTING accessors -- views, never
        second constructions; access_consciousness receives it injected).
      * the chat's per-turn arena is a declared PRIVATE-ARENA VIEW (its isolation is
        load-bearing by experiment); it reads THIS bus through broadcast()/broadcast_hold.
      * the re-entry stage is thalamus/reentrant_workspace (its own registrant).
    Declared surfaces carry today's REAL traffic: coalition_in receives the arm's per-tick
    submits under the bus's own ARBITRATE semantics (the P6-fixed rank key -- ranked by
    final membrane, never by name -- lives in this module's kernel and its self-check);
    broadcast_out is the one genuine cross-module read (LiveBrain.thought / the chat's
    _attended sentence), declared edge-less until the coalition channels (T031) and the
    report channel (T034) land their consumers.

    TYPE HONESTY (review, T030; T031 DISCHARGED the record half): the buffer entry is now
    the typed Coalition record -- identity, provenance, typed channels{salience, ...} --
    persisted under _T_COALITION by the arm's ring journaller, so 'the payload is an
    untyped 3-tuple' is no longer true anywhere. What REMAINS provisional, deliberately:
    the manifest's type row still cannot carry symbolic structure, so the (1,) f4 row on
    coalition ports describes the SALIENCE FIELD ONLY, the EVENT encoding marks the
    content symbolic (a token: an int unit index, the string 'threat'), and
    broadcast_out's unit is 'workspace_content' -- the winning token, NOT a bid. Stating
    it here keeps the artifact from certifying a description the five checks cannot
    themselves falsify (they compare declarations to declarations, never to runtime)."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "global_workspace",
        Port(
            name="coalition_in",
            region="thalamus",
            shape=(1,),
            dtype="f4",
            unit="coalition_bid",
            encoding="EVENT",
            kind="IN",
            declaring_module="global_workspace",
            join_policy="arbitrate",
        ),
    )
    manifest.declare_port(
        "global_workspace",
        Port(
            name="broadcast_out",
            region="thalamus",
            shape=(1,),
            dtype="f4",
            unit="workspace_content",
            encoding="EVENT",
            kind="OUT",
            declaring_module="global_workspace",
            delivery_policy="all-or-none",
        ),
    )
    # the consumer declares its subscription: the arm's per-tick coalition bids
    manifest.declare_edge("autonomous_loop", "coalition_out", "global_workspace", "coalition_in")


if __name__ == "__main__":
    main()
