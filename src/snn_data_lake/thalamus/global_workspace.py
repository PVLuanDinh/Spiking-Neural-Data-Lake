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
  * A SHARED INHIBITORY POOL: every excitatory spike drives one global inhibitory pool whose
    output subtracts from EVERY content pool (global lateral inhibition). This is the
    competition: strong pools keep firing while they silence the rest.
  * IGNITION = SUSTAINED FIRING: a content ignites (is eligible for a slot) only if it is
    still spiking in the SECOND HALF of the settling window -- i.e. it survived the competition
    and is in the high-firing attractor. A pool that fires a few transient spikes and is then
    inhibited to silence does NOT ignite; it stays local.
  * CAPACITY: the ignited set is admitted in competition rank order up to `capacity` slots.
    Everything past the capacity is ignited-but-not-accessed: it keeps its attractor charge and
    competes again next tick, but it is not broadcast and it earns no hold.

So: weak input -> the pool never crosses v_th (or is inhibited to silence) -> stays local, no
broadcast. Strong input -> the pool ignites into sustained firing, recruits its self-excitation,
suppresses competitors through the shared inhibitory pool, and -- if it ranks inside the capacity
-- is broadcast globally. That all-or-none ignition is the un-riggable anchor, checked against a
FAIR CONTROL (`use_ignition=False`) that is a pure per-tick argmax with no spiking, no attractor,
no inhibition: the control FLICKERS on noise; the spiking workspace is STABLE and only a
genuinely stronger, supra-threshold signal takes a slot.

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
  2. The shared lateral inhibition is DIVIDED ACROSS THE CAPACITY: `_G_INH / capacity`. The old
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
[0.72, 0.72, 0.72, 0.72, 0.72] and becoming the HEAD still [0.72, 0.825, 0.825, 0.825, 0.825] across
the same five tenant claims; a 40-content flood still admits 4; a silent source still holds 7 ticks
then releases; churn is still 9 against the argmax control's 54. Every one of those harnesses has
its tenant RE-POSTING THE SAME CONTENT, which is the branch this change leaves untouched -- that is
why they are unchanged, and it is the reason the change is safe rather than a coincidence.

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

THE RANK TIE-BREAK: MEASURED IN FULL, REPAIR FOUND, AND NOT SHIPPED. Every number below was RUN.
`(-late[k], -total[k], k)` falls through to the source string when two pools tie on both spike
counts. The census is re-derived rather than quoted -- exhaustive two-producer grid, warmups
{0,1,2,3} x 51 x 51 salience pairs on a 0.01 grid over [0.50, 1.00], scoring the tick AFTER the
warmup: 3,576 co-admissions of which 1,941 -- 54.3% -- are settled lexicographically. `_defended`
did not move this census at all, because it changes which pools are subsidised and not how ties
among them are broken.

IT IS NOT CONFINED TO THE HEAD, and that is the half every prior pass of this header got wrong.
`step()` admits `fired[:cap]`, and `_defended` then grants every ADMITTED occupant `self_boost +
hysteresis` on the NEXT tick -- so whatever settles the rank on tick t settles who is SUBSIDISED on
tick t+1, and a subsidised pool silences an unsubsidised one through the shared inhibition. The
string therefore decides ACCESS, not merely order. MEASURED, and the construction is spelled out
because the METRIC has a trap in it: six co-saturating producers into a capacity of 4 over 20
ticks, arm A naming them in loudness order and arm B in reverse, comparing the set of admitted
CLAIMS -- never the set of admitted NAMES, which differs between the arms whatever the arbiter does
and so reports a finding that is not there. Swept over 21 top claims x 7 spacings, keeping the 123
loads whose weakest claim stays >= 0.55:

    loads where the NAME changes the admitted CLAIM set   SHIPPED   -`_g`   -`_last_v`
    all 123                                                  24        0         0
    of those fully compliant (top claim <= 0.75)             13        0         0
    control: the same six producers spread at 0.04            0        0         0

At top 0.650 / spacing 0.005 -- every claim compliant -- the LOUDEST producer holds a slot on 20 of
20 ticks under one naming and 0 of 20 under the other, having changed nothing but its name. This is
the entrenchment shape `_defended` closed for retracted content, one level up, the alphabet
choosing who is subsidised. (CORRECTED 2026-08-06 by an adversarial re-run: this block first
carried a two-row table of name TUPLES and the counts "28 loads / SEVEN compliant" with no grid
stated. The phenomenon reproduces and is the finding; those counts do not, because they are
properties of a sweep grid rather than of the arbiter, and a name-tuple comparison cannot tell the
finding apart from the rename itself.)

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
        -`_g[k]`      (attractor charge) 1,941 -> 176 name-flips (4.9%), out-drive-head 75.2% ->
                      100%, ACCESS arm 24 -> 0 loads. PARTIAL: a previous version of this header
                      called it "the repair", and it is not -- `_g` still ties on 9.1% of the tied
                      set (176 of 1,941), and those 176 are exactly the equal-claim configs. It
                      leaves this module's head-capture TABLE byte-identical; it does NOT leave the
                      module's figures byte-identical, and the earlier wording generalised a true
                      narrow claim into a false broad one -- the 48-tick head trace moves and "the
                      HEAD changed on 17 ticks" becomes 14.
        -`_last_v[k]` (final membrane) 1,941 -> 0 name-flips, out-drive-head 100%, ACCESS arm 0.
                      Already stored, already policed by the determinism check, and the right
                      quantity on the physics: a pool firing every micro-step settles to a residual
                      of `2 * (cur - v_th)` after the subtract-threshold reset, so the membrane
                      carries LINEARLY the drive excess that the clipped integer count discards.
      Ranking on the raw drive scores identically (0 name-flips / 100%) and is REJECTED: `drives` is
      the INPUT to the competition and carries `_jitter_for`, so it never ties because the
      SYMMETRY-BREAK NOISE never ties. That swaps one arbitrary tie-break for another and stops the
      settle deciding anything. `_last_v` is the network's OUTPUT, after the inhibition and the
      self-excitation have run -- BUT ONLY PARTLY, and the limit is measured rather than argued:
      two pools claiming exactly 0.70 settle to membranes 3.46e-03 apart with the jitter live and
      EXACTLY 0.0 apart under `_JITTER = 0`. So on the 176 equal-claim configs `-_last_v` is the
      same symmetry-break noise laundered through the LIF, and its "never ties" is inherited rather
      than earned. It beats `-drives` on the other 3,400, where the claims genuinely differ and the
      settle has something to amplify -- which is the whole of the case for it.

WHAT THE REPAIR WOULD COST, which is why it is recorded here instead of shipped. The tick cost is
nil (0.1340 -> 0.1321 ms per 5-pool step -- one dict lookup inside a sort key), determinism holds (1
digest over 24 same-seed runs hashing the membrane FLOATS), and the live brain is BYTE-IDENTICAL:
`thought` 4 distinct values / 55 changes / names the submitted percept on 56 of 56 posting ticks
under both arbiters, on `make_brain(seed=0, refit_every=10, persist=False)` with `clock.phase =
0.25` over 120 ticks, `_access_state` and `_access_ema` (0.488958) identical to 6 dp. (A bare
"crc32 eb70d8f5" stood here until 2026-08-06 and was REMOVED rather than re-quoted: it never said
what had been hashed, so it could not be re-derived, and an unre-derivable digest is decoration.)
Slot winnability stays [0.72 x5], the capacity still binds (40
at 0.70 -> 4), the silent-source hold still releases after 7, `_defended` is untouched. THREE things
do move, and all three are recorded with the exact assert that fires:

  1. `cortex/access_consciousness.py` claim (10): `focus_moved >= 1` goes to 0. Its FOCUS lever IS a
     peer rename, so its green state REQUIRES the head to be name-decided and there is no arbiter
     that is both correct and green on it. Its ACCESS half is untouched (`access_same` 12 of 12
     under both), and claim (8) reads identically under both. Re-derive (10) onto the peer VOLUME
     lever it already runs in its own quiet-peer arm.
  2. `reasoning/rpm_solver.py` `collide_ok` case (ii): `ws_seq.broadcast() == stale` goes False.
     THIS BLOCKER WAS NOT PREVIOUSLY RECORDED. The precondition -- "the previous matrix's answer
     holds the HEAD" -- is itself manufactured by the string compare: measured, the stale pool holds
     the head on `_g` 0.850000 and membrane 0.613777 against this solve's own candidate at 1.000000
     and 0.904735, purely because source 'c0' sorts before 'c7'. Under the repair the fresher pool
     correctly takes the head and the case's own non-vacuity check fails. Its LOAD-BEARING half is
     unchanged (answer #7, known=True under both), so this is a precondition to reconstruct, not a
     behaviour to restore.
  3. THIS MODULE'S OWN head-capture table, [0.72, 0.825, 0.825, 0.825, 0.825] -> [0.805, 0.835,
     0.910, 0.955, None]. That is not a regression the repair introduces; it is the repair
     WITHDRAWING a credit this header was taking for the alphabet. The flat 0.825 is the challenger
     'chal' sorting before the tenant 't0'. Once the head is dynamic, becoming the FOCUS costs what
     out-driving a defended tenant actually costs, and against a tenant at the band ceiling (drive
     1.00) it is impossible -- LIMIT 3's ceiling, one level deeper. The self-check cannot see this:
     it asserts only `any(h is None or h > band_hi)`, which BOTH tables satisfy, so that table is
     printed evidence and only weakly asserted.

NOT SHIPPED, deliberately and not silently. The repair buys nothing measurable on any bus that
exists today -- live occupancy is 1 of 4 and `thought` is byte-identical -- and it reddens two
self-checks in files this pass does not own, one of them under a concurrent lane. Both of those
criteria have their green state only on the defect, which P8 answers by DELETING and re-deriving
them, never by loosening a bound; doing that is work for their owners. What is NOT yet measured, and
is the honest gap in this block: whether any repair exists that leaves the head-capture table where
it is, since item 3 above is a consequence of correctness rather than of the particular key chosen.

=====================================================================================================
WHAT THIS BUYS ON THE LIVE BRAIN, AND WHAT IT DOES NOT. Measured, 2026-08-06, on the canonical
`LiveBrain` composition -- 60 wake ticks, `.venv-lake`.
=====================================================================================================
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
`broadcast()` / `current` / `live_brain.thought` return the RANKED HEAD, and the head is a
different question from access. Swept the same way as the winnability table above:

    tenant claim      0.550   0.5797   0.6573   0.700   0.750     compliant ceiling 0.75
    win a SLOT at     0.720    0.720    0.720   0.720   0.720     <- inside the band, every tenant
    become HEAD at    0.720    0.825    0.825   0.825   0.825     <- ABOVE the band from 0.5797 up
    (retired arbiter) 0.735    0.735    0.895   0.950   0.955     <- slot and head were one thing

So a compliant specialist can always get ACCESS and cannot always become the FOCUS. Two things
follow and both are honest rather than convenient. The head row LOOKS like an improvement too
(0.895-0.955 down to a flat 0.825, flat instead of compounding with the tenant's claim) -- but that
credit is WITHDRAWN as of 2026-08-06 and the reason is measured, not argued: the flat 0.825 is the
challenger source 'chal' sorting before the tenant 't0' at the rank line, not the challenger
out-driving it. On a name-blind arbiter the same sweep reads [0.805, 0.835, 0.910, 0.955, None] --
AT OR ABOVE the retired arbiter's head row [0.735, 0.735, 0.895, 0.950, 0.955] at every one of the
five points, so the multi-slot head is not cheaper than the one it replaced once the alphabet stops
paying for it. See the tie-break block above; the number to quote about the head is that it is
currently decided by a string. It is
still outside the band, so `broadcast()` alone remains a poor place to look for faculty-to-faculty
influence -- `broadcast_all()` is where the change lives, and since 2026-08-06 so does the
SUBSCRIBE channel (see the delivery section below; a consumer no longer has to reach for a second
accessor to ask whether it got in). It is also arguably the CORRECT behaviour
rather than residual defect: a focus that flips whenever a marginally louder specialist posts is
the flickering argmax control this module exists to beat. Recorded as a limit and not defended as a
feature, because the next reader will otherwise assume "winnable" covered both.

=====================================================================================================
WHAT SUBSCRIBERS RECEIVE: THE SET, not the head (2026-08-06). The design question multi-slot CREATED.
=====================================================================================================
Multi-slot gave this workspace two different things -- CONTENTS (the admitted set) and a HEAD (the
top-ranked one) -- and left ONE push channel carrying only the second. That is wrong on the theory
and it was wrong in the tree. GWT's claim is that the workspace BROADCASTS ITS CONTENTS, and that
broadcast IS global availability; a bottleneck that admits four contents and announces one has a
capacity of four and an availability of one.

WHAT IT COST, measured on this bus rather than reasoned about. Two contents at identical salience,
capacity 4, varying NOTHING but the peer's source name: the second content is in `broadcast_all()`
on 12 of 12 ticks under every name, while a head-only subscriber saw it on 0 of 12 against a peer
named 'aaa' and 12 of 12 against 'mmm'. Two compliant pools both SATURATE their sustained-firing
counts, so the rank key falls through `-late` and `-total` to the source string, and the head is
settled lexicographically. `cortex/access_consciousness.py` read its own ignition off that head, so
a consciousness-relevant flag -- and the `access_index` behind it -- were being decided by a string
comparison. Delivering the set removes that channel; it does not soften it.

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
`reasoning/brain_adapters/executive_control.py`, `reasoning/gwt_deep.py`. (`reasoning/gwt_reentry.py`
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

DEEPENING + the SPIKING-vs-RATE CONTROL (functions + self-check below): the argmax control above
stabilises the workspace but strips spikes AND cross-tick memory AND inhibition together, so it does
NOT isolate whether SPIKES are load-bearing. A binding test does. The workspace must BIND K objects,
each a (color, shape) pair, and broadcast the pairings. The spiking path phase-multiplexes them
(Lisman-Jensen theta-gamma: each object in its own gamma slot, its color+shape firing SYNCHRONOUSLY
so a reader recovers the pairing from synchrony); a RATE path with the SAME hysteresis superposes the
activations and loses which color binds which shape (the classic binding problem). Honest result,
asserted below: they TIE at K=1 (spikes are NOT needed for single-item selection) and spikes WIN at
K>=2 (multi-item binding via synchrony) -- the specific, honest answer to "what do spikes buy." This
is contested (Singer/Gray vs Shadlen-Movshon), reader-dependent, and one solution among several; it
distinguishes the spiking workspace from the rate one on this task, nothing more.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.thalamus.global_workspace
"""

import itertools
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
    _G_INH = 2.4  # TOTAL lateral-inhibition budget; each slot gets _G_INH/capacity (see `capacity`)
    _JITTER = 0.02  # tiny deterministic symmetry-break so exact ties resolve to a fixed order

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
        self._g_lat = self._G_INH / self.capacity  # per-slot share of the lateral budget
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
    def submit(self, source, content, salience):
        """A specialist posts a candidate into this tick's competition buffer.

        SINGLE-THREADED ONLY. This appends to one shared list, and buffer ORDER is observable: the
        `_g` dict is built by iterating the buffer, and `_jitter_for` draws each pool's tie-break
        current from the instance RNG in first-encounter order. Two specialists calling this
        concurrently would therefore make the symmetry-break jitter depend on which thread won a
        race, which is exactly the nondeterminism this module's determinism self-check forbids.
        Concurrent specialists use `open_slots()` + `submit_from()` below instead."""
        self._buffer.append((str(source), content, float(salience)))

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
        `slot`'s own thread with no lock: no other thread touches `self._slot_buf[slot]`."""
        self._slot_buf[int(slot)].append((str(source), content, float(salience)))

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
        classifies `broadcast` BY ARITY -- 0 arguments is this reader, 3 is a submit wearing a
        reader's name. Its return type is a single content, and every `is not None` test in the tree
        is written against that. Widening it to return the whole workspace would break both."""
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
        on this bus, two contents at identical salience 0.65 and capacity 4, varying NOTHING but the
        peer's source name: the second content is in `broadcast_all()` on 12 of 12 ticks under every
        name, while a head-only subscriber sees it on 0 of 12 with a peer named 'aaa' and 12 of 12
        with a peer named 'mmm' -- because a `late`/`total` tie is broken lexicographically on the
        source key at the rank line in `step()`. A subscriber's view of its own global availability
        was being settled by a string comparison. Delivering the set removes that channel entirely;
        it does not soften it.

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
        a_inh = sync(si, -1)  # register this chunk; shared pool starts silent
        for t in range(self._T):
            for i in range(m):
                lateral = glat * max(0.0, a_inh - si[i])  # inhibition from OTHER pools
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
        k is `g_lat * (a_inh - self_k)` -- the drive from every OTHER pool. That is the global
        lateral inhibition (each pool inhibits the others through the shared pool): a lone pool
        inhibits only itself so it is barely damped and ignites, but among competitors the strongest
        pools' spikes suppress the rest to silence.

        `g_lat` is `_G_INH / capacity` -- the workspace divides ONE lateral budget across the slots
        it can hold, so a wider workspace competes less fiercely per slot and a capacity-1 workspace
        reduces to `_G_INH` exactly. That division is what un-froze this bus; see the module header
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
            tenants at 0.550/0.5797/0.6573/0.700/0.750, and still becomes the head at 0.720/0.825/
            0.825/0.825/0.825. Byte-identical to the table in the module header.
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
        self._g_lat = self._G_INH / cap  # read by _settle_chunk; set before any thread fans out
        held = self._defended(submitted)
        drives = {}
        for key, g in self._g.items():
            d = g
            if key in held:
                d += self.self_boost + self.hysteresis
            d += self._jitter_for(key)
            drives[key] = d

        # 3. Settle the spiking network; read out sustained (second-half) firing.
        total, late = self._simulate(drives)
        self._last_total = total
        self._last_late = late

        # 4. IGNITION -> ADMISSION. Every pool still firing in the second half has ignited (it
        #    survived the competition into the high-firing attractor); the workspace then admits
        #    them in competition rank order up to `capacity`. Nothing sustaining -> the workspace
        #    empties (release).
        #
        #    The rank key is the one this module always used, with the full key as its final
        #    tie-break instead of the source alone. THE FALL-THROUGH TO `k` IS A LIVE DEFECT and the
        #    module header carries the whole measurement: `late` saturates, so 1,941 of 3,576
        #    co-admissions are settled by the SOURCE STRING, and because `fired[:cap]` decides who
        #    `_defended` subsidises next tick, that string decides ACCESS and not just order.
        #    Ranking on `-self._last_v[k]` before `k` takes both counts to 0 at no tick cost; it is
        #    not applied here because it reddens two self-checks in files this pass does not own.
        #    HONEST STATUS of the `k`-vs-source part alone, measured rather than argued:
        #    that is hardening with NO reachable red, and the "determinism hole" it was written to
        #    close does not exist. `late` is keyed in `sorted(drives)` order on BOTH settle paths
        #    and `sorted` is stable, so the source-only key already resolved every tie identically
        #    -- 0 disagreements over four tie-heavy loads (one source with 4, 8 and 40 identical
        #    contents; two sources x 4) x six seeds, and a mutation reverting it passes this whole
        #    self-check. Kept because it costs nothing and does not depend on the settle paths
        #    continuing to hand back sorted keys; recorded as criterion-less per P8 rather than
        #    credited as a fix.
        fired = sorted((k for k in late if late[k] > 0), key=lambda k: (-late[k], -total[k], k))
        self._slots = tuple((k[0], k[1], self._g[k]) for k in fired[:cap])

        content = self.current
        self._buffer = []
        self._deliver(self.broadcast_all())  # BROADCAST the admitted SET to all subscribers
        return content


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
    48-tick stream they come apart completely: the head changed on 17 ticks and the previous head
    was STILL IN THE WORKSPACE on 47 of 47 of them. Not one of those 17 was a content losing
    access; every one was a rank swap inside a stable set. So `count_switches(heads)` no longer
    measures flicker on this arbiter, and keeping it as the stability criterion would have meant
    loosening its bound from 6 to something that admits 17 -- which is the weakening P8 forbids
    ("if a criterion turns out not to measure the right thing it is deleted, not weakened").

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
# DEEPENING: multi-item BINDING + the SPIKING-vs-RATE control (does synchrony buy anything over a rate
# integrator with the same hysteresis?). Self-contained LIF; used only by the self-check.
# ======================================================================================================
_BIND_TSLOT = 6  # micro-steps per gamma slot (one bound object)
_BIND_LEAK = 0.25  # binding-LIF leak per step
_BIND_VTH = 1.0  # binding-LIF threshold
_BIND_IGNITE = 0.6  # all-or-none ignition threshold for the binding broadcast


def _bind_step(v, i):
    """One leaky integrate-and-fire step for the binding demo. Returns (new_v, spike)."""
    v = v * (1.0 - _BIND_LEAK) + i
    if v >= _BIND_VTH:
        return 0.0, 1  # subtract-threshold reset -> discrete spike
    return (v if v > 0.0 else 0.0), 0


def bind_objects(n_color, n_shape, k, rng):
    """K objects, each a (color, shape) pair with DISTINCT colors and DISTINCT shapes -- so the part
    IDENTITIES are unambiguous and only the BINDING (which color goes with which shape) is at risk."""
    return list(zip(rng.sample(range(n_color), k), rng.sample(range(n_shape), k)))


def bind_broadcast_spiking(objs, n_color, n_shape, drive=1.0, sustain=4):
    """Ignite (if drive clears the threshold), then broadcast the objects PHASE-MULTIPLEXED: object k
    drives its color and shape LIF with the SAME current in gamma slot k, so they spike SYNCHRONOUSLY
    (bound); re-entry sustains each after its slot. Returns (ignited, raster[t]=(fired_colors,fired_shapes))."""
    if drive < _BIND_IGNITE:
        return False, []  # subliminal: no global broadcast
    vc = [0.0] * n_color
    vs = [0.0] * n_shape
    raster = []
    T = len(objs) * _BIND_TSLOT + sustain
    for t in range(T):
        slot = t // _BIND_TSLOT
        fc, fs = [], []
        for k, (c, s) in enumerate(objs):
            in_slot = k == slot
            cur = (
                drive if in_slot else drive * 0.55
            )  # re-entry current sustains (weaker), does not re-ignite
            vc[c], sc = _bind_step(vc[c], cur)
            vs[s], ss = _bind_step(vs[s], cur)
            if sc:
                fc.append(c)
            if ss:
                fs.append(s)
        raster.append((fc, fs))
    return True, raster


def bind_decode_spiking(raster):
    """Reader: color+shape that fire SYNCHRONOUSLY (same step) are bound. Returns the recovered pairs."""
    pairs = set()
    for fc, fs in raster:
        for c in fc:
            for s in fs:
                pairs.add((c, s))
    return pairs


def bind_broadcast_rate(objs, n_color, n_shape, drive=1.0):
    """Rate control: same all-or-none ignition, but a SUPERPOSED activation vector -- every object's
    color and shape channel is on with equal magnitude, NO timing, so the pairing is not represented."""
    if drive < _BIND_IGNITE:
        return False, [], []
    color_act = [0.0] * n_color
    shape_act = [0.0] * n_shape
    for c, s in objs:
        color_act[c] += drive
        shape_act[s] += drive
    return True, color_act, shape_act


def bind_decode_rate(color_act, shape_act, k):
    """Reader for the rate broadcast: recover the K active colors and K active shapes (magnitudes say
    which are on), but the PAIRING is absent -> pair them canonically (sorted); right only by luck."""
    cols = sorted(sorted(range(len(color_act)), key=lambda i: -color_act[i])[:k])
    shps = sorted(sorted(range(len(shape_act)), key=lambda i: -shape_act[i])[:k])
    return set(zip(cols, shps))


def _bind_pair_acc(true_pairs, got_pairs):
    return len(set(true_pairs) & set(got_pairs)) / len(true_pairs)


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

    # ---- SPIKING COMPETITION: distinct-strength competitors -> the shared inhibitory pool
    #      SILENCES the weaker ones (0 sustained spikes) and only the strongest ignites. Note this
    #      is UNDER the capacity (3 candidates, 4 slots), so a free slot is NOT a free pass: the
    #      spiking competition still excludes them. Capacity bounds the workspace; competition
    #      fills it. Both have to be real or the mechanism is gone.
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
        and wta_losers_late == 0
        and g_ws.broadcast_all() == ("strong",)  # 3 candidates, 4 slots, ONE admitted
    )

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

    # ---- DEEPENING: multi-item BINDING, spiking-vs-rate control (isolates whether SPIKES matter) --
    brng = random.Random(0)
    bind = {}
    for k in (1, 2, 4):
        sp = rt = 0.0
        trials = 120
        for _ in range(trials):
            objs = bind_objects(8, 8, k, brng)
            _, r = bind_broadcast_spiking(objs, 8, 8)
            sp += _bind_pair_acc(objs, bind_decode_spiking(r))
            _, ca, sa = bind_broadcast_rate(objs, 8, 8)
            rt += _bind_pair_acc(objs, bind_decode_rate(ca, sa, k))
        bind[k] = (sp / trials, rt / trials)
    # all-or-none: the binding broadcast ignites on strong drive, stays subliminal below threshold.
    bind_ig_strong, _ = bind_broadcast_spiking(bind_objects(8, 8, 2, brng), 8, 8, drive=1.0)
    bind_ig_weak, _ = bind_broadcast_spiking(bind_objects(8, 8, 2, brng), 8, 8, drive=0.4)
    print("\nbinding (spiking vs rate control, pair-accuracy):")
    for k in (1, 2, 4):
        s, r = bind[k]
        tag = "tie (spikes not needed)" if abs(s - r) < 0.05 else f"spikes +{(s - r) * 100:.0f} pts"
        print(f"   K={k}: spiking {s:.0%}  rate {r:.0%}  -> {tag}")

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
    det_runs = []
    for _ in range(24):
        h2, s2 = run_stream(build_stream(n_ticks, seed=0), use_ignition=True, seed=0)
        det_runs.append(repr([(a, b) for a, b in zip(h2, s2)]))
    deterministic = (
        len(set(det_runs)) == 1
        and det_runs[0] == repr([(a, b) for a, b in zip(with_ig, set_ig)])
        and count_switches(with_ig) == sw_with
    )

    print(
        f"\nlive contract : safe tick broadcasts {contract_safe!r}; "
        f"proximity threat ignites {contract_danger!r}"
    )
    print(
        f"spiking WTA   : competitors 0.90/0.68/0.62 -> winner {g_content!r} keeps firing "
        f"(late spikes {wta_winner_late}); losers silenced (late spikes {wta_losers_late})"
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
    # Run, not argued.
    nocap = GlobalWorkspace(seed=0, self_boost=0.0, hysteresis=1e-9)
    for i in range(n_flood):
        nocap.submit(f"src{i:02d}", f"cand{i:02d}", 0.90)
    nocap.step()
    assert nocap.capacity > n_flood and len(nocap.broadcast_all()) == n_flood, (
        f"the bottleneck criterion must be reachable-red: at a sub-jitter defence the capacity is "
        f"{nocap.capacity} and all {len(nocap.broadcast_all())} candidates broadcast"
    )
    # genuine spiking competition: among distinct-strength pools, the shared inhibitory pool leaves
    # only the strongest in sustained firing; the weaker pools are silenced (not merely out-ranked)
    # and are excluded even though THREE free slots remain.
    assert wta_ok, (
        "the shared inhibitory pool must silence weaker pools so only the strongest ignites, even "
        f"with {cap - 1} slots free (got {g_ws.broadcast_all()})"
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

    def _lowest_win(cls, tenant_claim, ceiling, warm=6, tenants=1, test=None):
        """Lowest salience a challenger must post to WIN A SLOT beside warmed tenant(s).

        Returns None if no salience at or below `ceiling` gets in. Sweeps in steps of 0.005; the
        tenants keep posting on the challenge tick, so this is a live incumbent, not a stale one."""
        s = band_lo
        while s <= ceiling + 1e-9:
            w = cls(seed=0)
            for _ in range(warm):
                for i in range(tenants):
                    w.submit(f"t{i}", f"T{i}", tenant_claim)
                w.step()
            for i in range(tenants):
                w.submit(f"t{i}", f"T{i}", tenant_claim)
            w.submit("chal", "CHAL", s)
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

    # THE CRITERION: a compliant specialist must be able to win a slot against ANY compliant
    # tenant. `<= band_hi` is not a number chosen here -- it is the ceiling every producer in this
    # tree already derives from these same two attributes (`autonomous_loop::_percept_salience`,
    # `access_consciousness::_max_claim`), so a challenger above it is not a compliant specialist.
    assert all(w is not None and w <= band_hi + 1e-9 for w in win_new), (
        f"a compliant specialist must win a slot against every compliant tenant, inside the band "
        f"ceiling {band_hi:.2f}: needed {win_new} for tenants {[round(c, 4) for c in win_claims]}"
    )
    # ...and the OLD arbiter must FAIL it, or this change bought nothing and should be reverted.
    # This is the A/B P2 demands before retiring a mechanism, run rather than argued.
    assert any(w is None or w > band_hi + 1e-9 for w in win_old), (
        f"the retired single-slot arbiter must be measurably WORSE on this criterion -- if it "
        f"passes, the multi-slot replacement is unjustified. Got {win_old}"
    )
    # REACHABLE RED for the winnability criterion itself, by an input this module accepts rather
    # than by breaking the accessor: a tenant claiming ABOVE the band monopolises the workspace,
    # exactly as `access_consciousness::_max_claim` says it would. That is what the band is FOR,
    # and it is why this criterion cannot be satisfied by simply widening the capacity.
    over_band = _lowest_win(GlobalWorkspace, 1.0, band_hi)
    assert over_band is None, (
        f"the winnability criterion must be reachable-red: a tenant claiming 1.00 (above the "
        f"{band_hi:.2f} ceiling) must NOT be joinable by a compliant challenger, got {over_band}"
    )

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
    coex_claim = max(win_new)
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
            f"capacity {k.capacity}, lateral gain {k._G_INH}/{k.capacity} = "
            f"{k._G_INH / k.capacity:.2f}"
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
    # winner by SUSTAINED firing: the ignited pool is still spiking in the second half of the window,
    # having silenced its competitors through the shared inhibitory pool.
    assert s_ws._last_late["perception", "strong"] > 0, (
        "the ignited winner must show sustained (2nd-half) firing"
    )
    assert wta_winner_late > 0 and wta_losers_late == 0, (
        "sustained firing must belong to ONE pool (WTA)"
    )

    # (c) DETERMINISM --------------------------------------------------------------------------
    assert deterministic, "same seed must reproduce identical broadcasts and switch counts"

    # (c2) PARALLEL SETTLING is BYTE-IDENTICAL to serial settling ------------------------------
    # The gate is measured, so force both paths explicitly here rather than hoping to land on one.
    p_ws = GlobalWorkspace(seed=0)
    prng = random.Random(4242)
    p_drives = {(f"s{i}", f"c{i}"): prng.uniform(0.3, 1.2) for i in range(240)}
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
                lateral = ws._g_lat * max(0.0, a_inh - self_i[k])
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

    # (d) DEEPENING: the SPIKING-vs-RATE control is a PERMANENT GUARD -- honest about where spikes help.
    assert bind_ig_strong and not bind_ig_weak, "binding broadcast ignition must be all-or-none"
    assert bind[1][0] - bind[1][1] < 0.05, (
        f"at K=1 spikes must NOT beat rate (single-item binding needs no spikes): "
        f"{bind[1][0]:.2f} vs {bind[1][1]:.2f}"
    )
    assert bind[2][0] > bind[2][1] + 0.15, (
        f"at K=2 the spiking workspace must bind better than the rate control (synchrony is load-bearing): "
        f"{bind[2][0]:.2f} vs {bind[2][1]:.2f}"
    )
    assert bind[4][0] > 0.9, f"spiking multi-item binding must stay high at K=4 ({bind[4][0]:.2f})"

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
    # EVERY occupant is defended unconditionally. This is the arbiter that produced the live
    # measurement in `_defended`'s docstring, so the red here is the defect itself and not a
    # sabotage of the accessor.
    e2_red_ws = GlobalWorkspace(seed=0)
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

    print(
        "\nself-check OK: "
        f"capacity {cap} (derived from defence {band_hi and 1.0 - band_hi:.2f}); churn {churn_with} "
        f"vs argmax {churn_without} (>=3x stabler), previous broadcast retained {kept_with}/"
        f"{len(with_ig) - 1} vs {kept_without}; bottleneck {b_admitted} of {b_ignited} ignited; "
        f"WINNABLE at {max(w for w in win_new):.3f} <= band ceiling {band_hi:.2f} where the retired "
        f"single-slot arbiter needed {win_old}; competition still excludes losers with slots free; "
        f"all-or-none (sub spikes={weak_spikes}, supra spikes={strong_spikes}); global-reach; "
        f"persist {persisted}->release; threat ignited; live-contract holds; deterministic over 24 "
        f"runs; BINDING spiking-vs-rate control honest "
        f"(K1 tie {bind[1][0]:.0%}/{bind[1][1]:.0%}, K2 spikes {bind[2][0]:.0%}>{bind[2][1]:.0%})"
    )


if __name__ == "__main__":
    main()
