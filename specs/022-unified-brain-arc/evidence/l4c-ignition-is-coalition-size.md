# L4c — the ignition asymmetry is COALITION SIZE, not the relaunch

*Measured 2026-09-06. This RETRACTS L4 §8.2 and replaces the finding it recorded. It also closes the first
of the two openings L4 left, by refuting it.*

## 1. The claim being tested

L4 §8.2 recorded, and the plan and tasks repeated: *a first life ignites its workspace on 4 of 5 seeds and a
second life on the same lake ignites on none — a real behavioural loss across the relaunch.* On that basis
L4's persistence half was re-opened with ignition named as its target.

**It is wrong.** Ignition has nothing to do with the relaunch.

## 2. First: the counter, for the fourth time

Four instruments have now been wrong on this one quantity, and the corrections went in both directions:

1. `state()["ignited"]` — read the wrong field; reported 0 everywhere, and I concluded there was no
   regression at all.
2. `_access_state["ignited"]` read after EVERY tick — `_update_access` runs only on awake ticks and never
   clears the flag, so one ignition is re-counted on every following sleep tick. An adversarial panel
   measured a naive counter reading 293 over 50 seeds against a true 126, and manufacturing a fake
   invariant out of a wake index that cannot advance while the brain sleeps.
3. The corrected wake-gated counter — the one now committed.
4. A verifier's independent counter, which reported seed 0 igniting twice and seed 1 twice, and **does not
   reproduce here.**

Rather than add a fifth opinion, the numbers below come from **two independent instruments in the same run,
required to agree before anything is reported**: one reads the flag on wake ticks from outside, the other
wraps `AccessConsciousness.step` and reads `_ignited` at its source the instant it is set. Staleness cannot
fool the second; a missed call site cannot fool the first.

**Agreement: 10 arms of 10, zero disagreements.** Arm A ignites once, at wake tick 13, on 4 of 5 seeds; arm B
ignites zero times on all five. The committed probe's numbers were right.

## 3. The three arms that settle the cause

The coalition size enters the wake path in exactly one place — the number of members handed to
`bind_assembly`:

```
k = min(2 + (len(arm.q) + grown_units + real_units) // _percepts + round(2*sustained), 8)
```

| arm | what it does | result |
|---|---|---|
| **1** | a FIRST life at 480 ticks — is ignition ongoing? | seeds 0/2/3: ignitions at **[13]** only, over **236–240 wake ticks**. `k` climbs 3→8 over the first ~13 binds and then sits at 8 for 223+ binds with **zero** ignitions. |
| **2** | a SECOND life with the bound assembly pinned SMALL | seeds 0/2/3: **[10, 11, 12]** — it ignites THREE times, more than the first life ever does. Plain arm B: `k` is 8 on every one of its 61 binds. |
| **3** | **THE DISCRIMINATOR** — a FIRST life with the assembly pinned LARGE from tick 0 | seeds 0/2/3: **no ignitions at all.** |

Arm 3 is decisive. A first life that never sees a small coalition behaves exactly like a second life. Arm 2
is its mirror: a second life that is given a small coalition ignites freely.

### 3.1 CORRECTION: which restored quantity caps `k`

This section first named `grown_units` as the restored quantity that puts `k` at its cap. **That is wrong,
and a panel measured it in both directions:** zeroing `grown_units` in a second life leaves `k` at 8 on all
61 binds and still gives 0 ignitions, while zeroing `_real_units` — the restored prototype count — drops `k`
to 3–5 and brings ignitions back. At the first bind of a second life the terms are
`len(arm.q)=1, grown_units=4, _real_units=20, _percepts=3`, so `k = min(2 + 25//3, 8) = 8`: **`_real_units`
alone already saturates it.**

The arms above are unaffected, because they intervened on the member list directly rather than on any
upstream term — which is why the conclusion (coalition size decides) stands while the attribution of which
restored number drives it did not. Recorded as a correction rather than silently amended: naming the wrong
carrier would have sent the next reader to the wrong line.

**Ignition is governed by coalition size. It is not governed by persistence.**

## 4. What is actually true, and it is worse than a resume bug

A first life ignites only because it passes briefly through a small-`k` window while its substrate is still
growing. A relaunched brain boots past that window because its restored `_real_units` (see §3.1) puts `k` at its cap
immediately. But arm 1 shows the first life never returns either: **once `k` saturates at 8, ignition does
not occur again in 223 consecutive binds.**

So the honest statement is not "a relaunched brain lost a faculty". It is:

> **Global-workspace ignition is reachable only while the coalition is small, and every brain permanently
> outgrows it.** A matured brain of any lineage — first life or hundredth — never ignites.

That makes `access_index` on the live cockpit a number that reads ~0 forever on any brain past its first
few dozen wake ticks, and `ignited` a flag that is structurally unreachable rather than merely rare.

## 5. What this row does NOT claim

- **It does not claim a mechanism inside the workspace.** A bisect proposed that ignition requires two
  consecutive identical content keys and that a saturated `k` makes that impossible. That is plausible and
  it was NOT verified here; what is measured is only that the assembly SIZE decides the outcome, in both
  directions, in a first life and a second.
- **It does not fix it.** The access path has a documented history of experiments reverted on their nulls,
  and changing an ignition criterion at the end of a long session without a pre-registered protocol is
  exactly the move that history warns against. This row measures and hands over.
- **It does not touch C4's instruments**, which are unbuilt. C5's conclusion protocol already governs what
  may be said about this surface, and this finding is a fact about its reachability, not a reading of it.

## 6. Consequences for L4

- **L4 §8.2 is RETRACTED.** Ignition is not evidence of a resume defect, and the plan and tasks rows are
  corrected.
- **L4's persistence half stays re-opened, but on ONE leg instead of two.** What survives is the state
  comparison: of 93 numeric `state()` keys only 52 move, and 17 of those 52 return identical in a second
  life. That is real and unaffected by any of this. Ignition is struck from its target list.
- **The committed probe's verdict line is corrected.** It printed "a second life that ignites less than its
  first is a RESUME DEFECT". It is not, and a probe that asserts a false conclusion is worse than no probe.

## 7. Handed over, named

**For the C-rows (the access/consciousness family):** ignition is unreachable at saturated coalition size,
measured in both directions by pinning the assembly. If a C4 instrument is ever to read this surface, this is
the precondition it has to deal with first — and C5's row already forbids reporting a number from an
instrument whose gate cannot open without saying so.
