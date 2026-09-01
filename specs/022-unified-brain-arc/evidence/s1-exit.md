# S1 (US2) exit certification — specs/022 T061

Certified on `main` at the commit that lands this file (parent `6d85c096`), on the pinned
measurement host (`LUAN_AND_ANNIE`, `.venv-lake`). Every number below was produced by the
command printed beside it and is re-derivable by running that command again. **This file
certifies what was measured, including what did not pass** — an exit certification that
certifies only its greens certifies nothing (the s0-exit precedent, kept).

## Verdict

| # | Criterion (quickstart §S1 / SC-004 / SC-005) | Measured | Status |
|---|---|---|---|
| 1 | Seed→multi-region growth: coverage over the pinned growable set rises from 0 | **1 of 10** growable regions receive grown units from a live path (cortex-only placement) | **UNMET at the multi-region grain — see §1** |
| 1a | Boundary-width beats the gradient-only control | sharp 0 vs graded 15 boundary positions (0.117 of the axis), 7 seeds, production labelers | GREEN |
| 1b | Scrambled-field spatial null scores worse, same command | real → 1 boundary (closed form) vs scrambled → 64 | GREEN |
| 1c | External control (Allen/prosomeric) | elementwise 16/16 regions asserted; ARI 1.000 vs permutation null +0.008 (200 shuffles) | GREEN |
| 2 | Stopping criterion trips on overgrowth; \|LP\| gate executive-independent | overgrowth red trips; 6-seed real two-arm A/B stops 6/6, armed null did not trip | GREEN (PATIENCE=3 picked, disclosed) |
| 3 | Palette attribution: per-rule neuter isolates contributions; composition red fires | LOO attribution kernel + rule_charge CHARGED on the tick-live composed rule; naive-sum red at the measured boundary | **PARTIAL — the census ratchet is the carrier, see §3** |
| 4 | Derived K == held ±1; tau_f sweep monotone; frozen-STP refuses (SC-005) | K 96+1 = **97 derived vs 97 held** of 99 pushed; sweep 96/257/482; both frozen twins refuse | GREEN |
| 5 | Sleep A/B separates per-capability; phase aggregate is the control | replay/Benna-Fusi/structural each neuterable with its own observable; phase neuter leaves the substrate byte-identical | GREEN |
| 6 | Arm quiesced through sleep; E-STOP latch persists across phase boundaries | by construction + the battery's own sidecar machinery — no dedicated S1 A/B ran | **CERTIFIED-BY-CONSTRUCTION, said plainly — see §6** |

## §1 — Growth: the instruments are green, the coverage is red

    PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.growth_coverage_gate --assert

Output (the load-bearing lines, verbatim):

    COVERAGE RED (reported every run, asserted only as a ratchet + refusals): 1 of 10
    growable regions receive grown units from a LIVE path -- **BELOW FLOOR: cortex-only
    placement** (identity-pinned baseline ['cortex']; the set may only grow)
    gate OK -- and OK means the RATCHET held ... the coverage red above is STILL RED
    (1 of 10). Green never means coverage is adequate.

**What this means for the exit:** SC-004's first clause — a single-unit seed grows a
MULTI-region micro-brain — is NOT demonstrated. Grown units place into `cortex` only
(`neurogenesis_engine.py:511/:271`, the gate's own census). The measurement INSTRUMENTS the
criterion needs are all green and discriminating (boundary sharpness beats the gradient-only
control 0-vs-15; the scrambled-field null separates 1-vs-64; the Allen external control
agrees 16/16 with ARI 1.000 against a +0.008 permutation null), and the coverage ratchet
pins `['cortex']` so regression is impossible — but the multi-region placement itself is
open work. The growth machinery (arealization fields, prosomeric labels, locality-metered
program, per-region priors, g-network initial weights, spontaneous bootstrap: T040–T045) is
built and gated; what has not happened is a live path placing grown units into a second
growable region. **US2 exits with this named as its open half, not with this smoothed.**

## §2 — Stopping

    PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.growth_stop

    green  REAL two-arm A/B (6 seeds x 1600 ticks): unit deltas [4, 7, -7, -12, -11, -8]
    (sum -27), acc deltas median -0.0167, stops on 6/6 seeds -- the NULL is armed and did
    not trip

The overgrowth red trips; the |LP|-derived gate carries its executive-independent arm; the
armed null (deletion licence) did not trip. Standing disclosures, carried not hidden:
PATIENCE=3 is PICKED (a derivation from the trajectory's own autocorrelation is owed);
grow-one/prune-one churn at flat net units is invisible in the (units, acc) vocabulary.

## §3 — Palette attribution

    PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.rule_composer

The attribution KERNEL is the instrument: typed read/write sets, measured LOO attribution,
`rule_charge` as its first reader (verdict CHARGED on the tick-live composed rule — the
InhibitorySTDP production step applies through the kernel since T053-ii), and the
composition-arbitration red fires at its MEASURED boundary (naive sum wrong past
|1 + Σc| ≥ 1; the T048 conflict pair at component grain: the sum sign-violates at an
excitatory prune while sequential reproduces the component step to the bit).

**The honest scope:** SC-004's "every palette rule shows nonzero isolated attribution on at
least one outcome" is carried by the RATCHETED census (`tools/rule_census.json`, day-one 99
rows / 56 UNDISPOSITIONED / 43 dispositioned; the artifact's own `ratchet_baseline` is the
live figure, never this file). The T051c-wave advances by dispositioning rows and the
ratchet forbids lag/stale/rise — attribution is proven per-rule as rules pass through the
kernel, not universally today. PARTIAL, by design of the wave; the ratchet is the carrier.

## §4 — Working memory: K derived, never picked

    PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.wm_capacity

    K DERIVED from the store's own constants and MATCHED behaviorally -- tau_f=300.0:
    t_s=4 (dwell 3 resource-gated ticks + handoff), T_max=385, K=96+1 active = 97 derived
    vs 97 held of 99 pushed (+/-1; over-capacity frames died by starvation); monotone
    sweep K=[96, 257, 482] over tau_f=(300, 800, 1500) is the positive control; frozen
    STP REFUSES both ways

SC-005's first clause, met exactly: derived == held (97 == 97, tolerance ±1 unused), the
sweep is the positive control, both frozen twins land in refusals, and the derivation
REFUSES outside its measured-valid regimes (cycle ≥ 5·tau_d; default membrane; seed-u
viability) rather than averaging over them. `wm_attractor` retired per P2 (T055→T056).

## §5 — Sleep ownership: per-capability separation

    python -m pytest tests/test_sleep_capabilities.py -q --no-cov -p no:randomly   (4/4)

Each sleep-owned capability separates under ITS neuter with the others' observables intact:
replay-neutered → consolidate() reached 0 times while wake-ensured cascades still rest;
Benna-Fusi-neutered → deposits survive, nothing settles; structural-neutered → a
constructed sign-flip survives that the live slow clock prunes (budget conserved); the
phase aggregate leaves the interaction substrate byte-identical with every counter flat
(counters are per-tick: the stale-carry red asserts a neutered tick can never inherit the
previous tick's success). The cascade-adoption charge kills its deletion mutant. Disclosed:
the replay ring is process-local; a restarted daemon replays nothing until a genuinely new
conversational turn arrives (refeed-from-lake is named open work in T058's record).

## §6 — Sleep embodiment: certified by construction, not by a dedicated A/B

The arm is idle through sleep BY CONSTRUCTION (the daemon's ASLEEP phase runs no brain.tick
and issues no motor command — `cortex/brain.py`'s own contract), and the E-STOP latch is a
durable sidecar that persists across process and phase boundaries (its machinery fires in
every battery run: the GATE-HOME sidecar line attributing latch_in_window). **No dedicated
S1 arm drove a sleep transition with a latched E-STOP and asserted persistence end-to-end.**
That is a certification by construction plus the sidecar's own battery coverage — said
plainly rather than dressed as a measurement this file does not hold.

## What this certification does NOT cover (the misses, named)

1. **Multi-region growth placement** (§1) — the S1 headline criterion, open. The ratchet
   pins today's `['cortex']`; the next growable-region placement moves it.
2. **Universal per-rule attribution** (§3) — carried by the T051c census ratchet, advancing
   by wave, not universal today.
3. **A dedicated sleep-embodiment A/B** (§6).
4. **Deep-history similarity recall** — T057's window tier is calibrated and live; indexed
   margin-scored recall over the full multi-GB lake awaits the segment-engine lane (T007).
5. **The replay ring across restarts** (§5, T058's named open work).
