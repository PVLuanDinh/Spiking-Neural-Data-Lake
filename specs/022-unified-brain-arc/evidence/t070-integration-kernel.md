# T070 -- the integration instrument on the kernel's port (specs/022 Phase 4 / C4; plan row :150; SC-009's first half)

Sections 1-2 are the PRE-REGISTRATION, written 2026-09-10 BEFORE a line of this row's code existed and frozen at the
commit that lands them; amendments are dated appended sections; section 3 is written after the runs.

## 1. Territory, measured before the design (each fact with its source line, re-derived before citing)

1. The incumbent `cortex/integration_measure.py` (511 lines) computes a Perturbational-Complexity-style number: the
   CAUSAL footprint of a brief pulse (the perturbed raster XOR the same-noise unperturbed raster over the post-pulse
   window, `causal_response` :265), serialized time-major (`_serialize` :280) and scored by normalized LZ76
   (`normalized_lz` :161; `source_entropy` :148 is the density CEILING, reported and never divided by). Its
   docstring (:1-60) binds the framing: NEVER Phi, NEVER a consciousness threshold, ONE number is not reportable --
   only CONTRASTS between regimes on one substrate are (`main()` asserts orderings; its six absolute cutoffs are
   tripwires on its own fixture). Its network is PRIVATE: `PerturbationalComplexity._run` (:231) builds its own LIF
   net from its own weight matrix and pre-draws its own noise; the module says "THIS CLASS HAS NO PORT through
   which an external system's state can enter ... a future caller wanting to say something about the brain owes
   the port first, and a contrast second". The brain's `pci` producer was DELETED 2026-08-05 for that reason
   (brain.py :3363, :4180).
2. The kernel's port: `KernelTick.tick(ext_q)` (kernel_tick.py :274) takes external drive as fixed-point int64 over
   the index space and returns the fired indices; deterministic (determinism_gate); `KernelState.to_blob/from_blob`
   (kernel_state.py :108-149) is the whole-state frame that refuses a half restore. KernelTick holds runtime OUTSIDE
   the frame: `_rings` (delay rings), `_acc` (per-partition accumulators), `tick_index` (:198-203), all reset at
   construction (:192).
3. G5 (T017, tasks.md :40) ALREADY snapshots that runtime: `substrate/replay_gate.capture(kt, rng_streams)` embeds
   the KernelState frame whole PLUS tick_index, the ring slot contents and the accumulators, with the named RNG
   streams' POSITIONS; `restore(blob, kt)` seats a freshly built kernel over the SAME synapse table at that point and
   REFUSES (`SnapshotRefused`) a snapshot without positions, a version mismatch or a parameter mismatch (:100-118).
   Its docstring: "every perturbation instrument replays from a snapshot rather than re-running from t=0".
4. The T063 harness drives a KernelState built from the H01 artifact with pre-drawn drive and reads spike counts
   (`prior_drift_harness.run_arm` :1039, `fired = kt.tick(ext)` :1026); it owns the population builders
   (`arm_seeded` :509, `band_balanced_subgraph` :810, `drive_classes` :849, the no-edges twin :2028), the paired
   one-sided t with its 2^S sign-flip FPR (`contrast` :417, bar `T_BAR` = 1.895 at 7 d.f.), the per-arm
   above-chance reading (`above_chance`), the 64-unit fixture (`synthetic_artifact` :1281) and the pinned campaign
   constants (`_pin_stages()[-1]`: K = 8, D = 4, W = 1.0). Population 14,215 units / 70,001 edges at W = 1.0.
5. The conclusion registry (`substrate/conclusion_protocol.py`): `CONCLUSION["integration"]` = (licensed "the system's
   parts share information that a partition of the same parts does not, on this substrate"; refused "this licenses
   no claim that the integration is experienced, and no claim that a larger number is more conscious";
   dissociation "the partitioned twin: the same parts with the measured channel cut must read at or below the
   null"; reading None; date 2026-09-06; surface symbols ("integration_competence", "pci")); `C4_TASK["integration"]
   = None` (:84). The gate carries ONE surface debt: cockpit.py:920 prints integration's reading without its
   licensed sentence in view (baseline 1, may fall never rise).
6. `brain_telemetry.PROVENANCE["pci"]` (:189) reads "sleep-only, refreshes every 4th night" -- STALE since the
   producer's deletion 2026-08-05; the cockpit renders `row('PCI', s.pci)` (:920) and `s.pci` is absent from
   `Brain.state()`.
7. The live brain runs on per-object neurons; the kernel wire was DECLINED at the deployed width (S6,
   evidence/s6-kernel-wire.md) and the `kernel_tick`/`kernel_state` wire rows are EXEMPT until the first tick
   population at or above N*. No live population runs on the kernel today.
8. The arc's one Codex read (`scratchpad/t070/codex_defects.md`, 2026-09-10) raised three objections, each
   reproduced against its cited line and adopted below: an activity-floor artefact on the cut-edges arm (2.5a);
   the registry's "at or below the null" untested as drafted (2.4 R2); the G5 chain reduced to a reset (2.1, item 3).

## 2. Pre-registration

### 2.1 The mechanism (P2: one PCI computation, one snapshot mechanism, one paired t -- T070 adds none of the three)
A kernel port in `cortex/integration_measure.py`: `kernel_causal_response(blob, synapses, params, drive, pulse)`:
  (i) build a KernelTick over `synapses` with `params` (decay, v_reset, refrac_ticks) and `restore(blob, kt)` from a
      snapshot `capture`d AFTER a warm-up of `warm` ticks on the ORIGINAL run (the snapshot carries in-flight spikes:
      rings, accumulators, tick_index and the drive stream's position);
  (ii) run TWO restores from the same blob: BASELINE consumes the drive stream as captured; PERTURBED consumes the
      same stream and additionally adds `pulse_current` (quantized with the table's frac_bits, P1) to the `perturb`
      units for `pulse_len` ticks starting at the first post-restore tick `pulse_t`;
  (iii) the footprint = raster_PERTURBED XOR raster_BASELINE over [pulse_t + pulse_len, steps), serialized by the
      incumbent `_serialize` and scored by the incumbent `normalized_lz` (gzip twin reported beside). Nothing about the
      score changes; only the network that produces the rasters changes, and it is the tree's kernel.
The drive stream is a NAMED numpy Generator captured by `capture` (positions, not seeds), so the two restores draw
identical drive; there is no RNG inside `tick`.

### 2.2 The arms (four regimes on ONE population, built by the harness's own constructors)
  INTEGRATED    the population's own wiring (`arm_seeded(art)`).
  PARTITIONED   the same units with every edge cut (the harness's no-edges twin) -- the registry's literal
                "partitioned twin: the same parts with the measured channel cut".
  PARTITIONED-RATE-MATCHED (control, 2.5a): PARTITIONED with its background drive amplitude re-calibrated by a
                bounded bisection (<= 12 steps, disclosed) so its mean non-pulse firing rate is within 5 % of
                INTEGRATED's on the same seed.
  SYNCHRONOUS   a dense strong-coupling table over the same units (out-degree the population's mean x 8, every weight
                at the table's maximum magnitude, signs kept) -- the "integrated but not differentiated" regime.
  SILENT        INTEGRATED with the background drive amplitude 0 (the pulse still lands).
Populations: the harness fixture `synthetic_artifact(64, out_degree=8)` for the self-check; the H01 population at
W = 1.0 (`band_balanced_subgraph` at the pinned fraction; 14,215 units / 70,001 edges) for the campaign. S = 8 seeds
(0..7); per seed: `trials` = 8 perturbation sets of `n_perturb` = 3 units drawn in population index space (identical
across arms, as `drive_classes` does), `warm` = 40 ticks, `pulse_t` = the first post-restore tick, `pulse_len` = 3,
`steps` = 80 post-restore ticks, `pulse_current` = 1.2 x threshold (the amplitude h01_ingest's own liveness driver
uses). These constants are chosen BEFORE the pilot and disclosed as P1 debt.

### 2.3 The statistic and the verdict rule (written first)
Per arm, per seed: PCI = the mean normalized LZ76 over the 8 perturbation sets. Contrasts are paired over the 8 seeds
with the harness's `contrast` (one-sided t, 7 d.f., bar 1.895, the 2^8 sign-flip FPR printed beside every t).
The NULL (2.4 R2), per seed, on the INTEGRATED run: the local-echo floor = the PCI of the same footprint with every
non-pulsed unit's row zeroed (the response the pulse causes when nothing propagates).
PASS iff ALL of:
  (b1) t(INTEGRATED - PARTITIONED) > 1.895;
  (b2) t(INTEGRATED - PARTITIONED-RATE-MATCHED) > 1.895   [the artefact control];
  (b3) t(INTEGRATED - SYNCHRONOUS) > 1.895;
  (b4) SILENT PCI = 0 exactly on 8/8 seeds;
  (b5) rate(SYNCHRONOUS) >= rate(INTEGRATED) on 8/8 and rate(PARTITIONED-RATE-MATCHED) within 5 % of INTEGRATED on 8/8;
  (b6) t(INTEGRATED - null) > 1.895 AND PARTITIONED <= null on 8/8 AND PARTITIONED-RATE-MATCHED <= null on 8/8
       [the registry's dissociation, R2];
  (b7) R3 holds (2.4).
Any conjunct NOT MET -> the row reads NOT MET on that conjunct, with the numbers; no conjunct is re-labelled after
its number. The number that reaches any surface is the CONTRAST (b1) with (b2) beside it, never an arm's PCI alone.

### 2.4 The reds (RUN RED before the mechanism, in-command with their controls thereafter)
R1 (the row's red, "four regime arms must separate"): the instrument with the pulse REMOVED reads every arm's PCI
   exactly 0 -> (b1)(b2)(b3)(b6) all NOT MET -> RED. Reachable by construction; run first with `pulse_current = 0`.
R2 (the registry's dissociation, C-R5): a PARTITIONED arm that reads ABOVE the null fires -- constructed by leaving
   ONE edge class uncut (a "partition" that did not cut): its footprint propagates and its PCI exceeds the echo
   floor; the instrument must report (b6) NOT MET for it. The un-cut arm is the red's input; the cut arm is the
   measurement.
R3 (the G5 chain): (i) the restored BASELINE continuation is byte-identical to the uninterrupted original run over the
   same ticks (replay_gate's own criterion); (ii) a state-frame-only "restore" (`KernelState.from_blob` into a fresh
   KernelTick, rings dropped) DIVERGES from the original on the fixture with non-zero delays -- if it does NOT
   diverge the fixture has no in-flight spikes at the capture tick and the fixture is wrong, not the chain; (iii)
   `restore` REFUSES a blob with the positions stripped (`SnapshotRefused`).

### 2.5 The controls that are not the method
(a) PARTITIONED-RATE-MATCHED (Codex (a), accepted): a PCI gap that vanishes under rate matching is the activity
    floor (`source_entropy`'s density ceiling), not integration; (b2) is the conjunct that carries it.
(b) The rate control of the incumbent, kept: SYNCHRONOUS at least as active as INTEGRATED (b5).
(c) The pulse-less identity: with the pulse removed the two restores' rasters are byte-identical (footprint empty);
    asserted on every run before any PCI is read (it is R1's mechanism and the port's own liveness check).
(d) SILENT = 0 exactly: a non-zero SILENT PCI means drive leaked into the footprint (b4).

### 2.6 Disposition on a null (fixed in advance)
PASS: `C4_TASK["integration"] = "T070"` and `CONCLUSION["integration"].reading` = the campaign's (b1) t, set in the
SAME commit that pins the campaign (`tools/t070_pci.json`: head, harness sha256, artifact sha256, every arm's
per-seed PCI and rate, every contrast with its FPR, the null per seed, the host windows); `PROVENANCE["pci"]`
rewritten to "off-tick instrument on the H01 fixture population; a CONTRAST (integrated vs partitioned, rate-matched
control beside it) pinned in tools/t070_pci.json; NOT the live brain"; the cockpit's PCI row prints the contrast
WITH the registry's licensed sentence in view (the conclusion gate's surface debt 1 -> 0); the incumbent's private
LIF net (`_run`, the four toy factories) RETIRED in its own commit AFTER the kernel self-check reproduces the
incumbent's orderings on the fixture (P2: the unique capability -- four regimes -- ported first; ONE network
implementation survives).
NULL on (b1) or (b2) or (b6): the port is DELETED (P8 clause 5), the toy instrument stays as it is, the registry's
reading stays None, `C4_TASK` stays None, the record says what was measured and on which conjunct.
NULL on R3 (the chain does not resume byte-identically): a G5 defect, not this row's finding -- the row STOPS and
reports it to T017's gate; nothing is read.

### 2.7 What this row does NOT claim
Never Phi; never a threshold; never a sentence about experience; never an aggregate with any other reading (the
three standing refusals). "Live" is NOT delivered: no live population runs on the kernel (fact 7); the instrument
is live-CAPABLE (any KernelState/Synapses pair) and is read on the H01 fixture population off the tick; the record
names what delivers "live" -- the first tick population at or above N*. Nothing here is a claim about the live
brain's capability (capability_bar: no row earned).

### 2.8 Widths, cost, host
Self-check: the 64-unit fixture, 4 seeds, ~2 s. Campaign: 5 arms x 8 seeds x (40 warm + 8 x 83 ticks) at ~3 ms per
tick on 14,215 units -> ~1.5 min per arm-seed set, ~10 min total; the host clause read as T063 2.12(c) reads it (five
90 s windows, no restart, the daemon's PID per window) -- the daemon exits on every new commit (T063 panel F9), so
the campaign runs in a commit-free window.

### 2.9 Commit order (one change each; reds first)
1 this pre-registration · 2 R3 RED (the tests; the port skeleton that refuses) · 3 the port green (R3 (i)(ii)(iii)) ·
4 R1/R2 RED (pulse-less run, the un-cut partition) · 5 the four arms + rate matching + the null + the contrasts;
self-check green with every red in the same command · 6 the campaign pin · 7 the disposition (registry, PROVENANCE,
cockpit row, and -- on PASS -- the toy net's retirement) · 8 section 3 · panel · fold. Census gates after every
symbol-adding commit (rule_composer included; `above_chance`'s sibling readings carry no learning verb, verified by
running the gate). No battery until the stack's rows are in (owner 2026-09-10).

### 2.10 Amendment, dated 2026-09-10 03:55 local, written BEFORE any campaign number was read (the campaign was
### running with its log block-buffered; the only numbers seen are the 64-unit fixture's, quoted here)

**What the fixture showed (the builder's self-check on `synthetic_artifact(64, out_degree=8)`, 4 seeds; commit `551ba84`):**
(b4) "SILENT PCI = 0 exactly on every seed" read NOT MET -- SILENT PCI = 0 exactly on 1 of 4 seeds; on the others the
pulse propagates ONE HOP from rest through the intact wiring on roughly 1 of 8 pulse sets (PCI 0.0012-0.0025). (b6)'s
rate-matched clause read NOT MET on 1 of 4 seeds: the rate-matched arm's local echo at its higher drive amplitude
exceeded INTEGRATED's echo on seed 3 (0.0162 vs 0.0134). Every other conjunct read MET on the fixture ((b1) t +4.2504,
(b2) +3.9028, (b3) +4.9497, (b5) 4/4, (b7) 4/4); R1, R2 and R3 fired RED in the same command.

**The defect is in this pre-registration, not in the instrument.** 2.2 defined SILENT as "INTEGRATED with the background
drive amplitude 0 (the pulse still lands)" and 2.5(d) read a non-zero SILENT PCI as "drive leaked into the footprint".
That reading assumed the pulse alone cannot fire a post-synaptic unit from rest. On this kernel it can: a 1.2x-threshold
pulse on three units fires them, and their out-edges carry enough charge to fire a neighbour in some pulse sets, so the
footprint contains propagation with NO background drive present. A non-zero SILENT PCI on this arm is therefore NOT drive
leakage; it is integration of the pulse through the resting wiring -- the very quantity the instrument exists to read.
The incumbent's `silent_network` read exactly 0 because its silent regime had neither coupling nor noise (a network that
cannot propagate anything), which 2.2's SILENT does not reproduce. The drive-leak control 2.5(d) wanted already exists
and is asserted on every arm before any PCI is read: the pulse-less identity (2.5(c)) -- with the pulse removed the two
restores' rasters are byte-identical, so nothing in any footprint comes from anywhere but the pulse.

**What this amendment does and does NOT do.** It does NOT remove (b4) from the verdict rule and does NOT relabel it: 2.3's
PASS requires all seven conjuncts as written, and the campaign's (b4) reading is recorded exactly as the rule reads it. It
records, before the number, WHAT a NOT MET (b4) means on this arm (pulse propagation at rest, reported beside the
local-echo null), so that the reading is not later explained after the fact. The same holds for (b6)'s rate-matched
clause: a rate-matched arm at a higher drive amplitude has a larger local echo by construction (more pulsed-unit spikes
in the window), so "rate-matched <= null" compares an echo at one amplitude against a null taken at another; where it
reads NOT MET the record says so and names this cause, and the conjunct stays in the rule.

**Disposition, fixed now for the branch 2.6 did not name.** 2.6 prescribes PASS (all seven) and NULL on (b1), (b2) or (b6).
If the campaign reads (b1), (b2) and (b6)'s first two clauses MET but (b4) and/or (b6)'s rate-matched clause NOT MET, the
row reads NOT MET on those conjuncts with their numbers; neither 2.6 branch fires: the port is RETAINED as an instrument
(it separates the arms it was built to separate and its reds are reachable), the registry `reading` stays None and
`C4_TASK["integration"]` stays None (no PASS, no C4 number), the toy net is NOT retired (its retirement was conditional on
PASS), the cockpit PCI row is NOT filled, and `PROVENANCE["pci"]`'s stale row is corrected as a map/territory fix only
(it is false regardless of this row's outcome). A successor pre-registration re-specifies (b4) as a reported quantity
(SILENT propagation from rest, beside the null) and (b6)'s rate-matched clause against a null taken at the matched
amplitude, and re-reads the row -- never this row re-reading itself.

**Numbers seen before this amendment:** only the fixture readings quoted above. No campaign number had been read; the
campaign's log had flushed nothing when this section was written.
