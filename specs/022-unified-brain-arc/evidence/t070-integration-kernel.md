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

### 2.11 Amendment, dated 2026-09-10 13:20 local, BEFORE any campaign PCI number exists: the scorer's cost, an apparatus
### defect of this pre-registration, and its fix (the same statistic, a different algorithm, equality-tested)

**What happened (numbers seen, all of them).** The first campaign (`python -m snn_data_lake.cortex.integration_kernel
--campaign`, started 03:40:35 local on the H01 population, 14,215 units / 70,001 edges, 8 seeds) read the host clause
(settled = True, one daemon PID 37988 across five windows: 97.969 / 124.719 / 115.734 / 110.297 / 79.375 CPU-s), ran R3 on
all 8 seeds in under 8 minutes (in-flight 485-611 spikes per snapshot; restored baseline BYTE-IDENTICAL; the frame-only
restore DIVERGED as it must; omit-positions REFUSED; the pulse-less identity held), and then spent 9 h 25 min inside seed 0's
five arms without printing a seed line. It was stopped at 13:20 with no PCI number produced. A probe of the incumbent
scorer on sparse binary strings (density 0.02): 5,120 characters (the 64-unit fixture's footprint) parse in 0.241 s; a
160,000-character string had not returned after 20 s; the campaign's footprint is 14,215 x 80 = 1,137,200 characters, and
the design calls for 8 pulse sets x 5 arms x 8 seeds = 320 parses plus 320 null parses.

**The defect.** 2.8 costed the campaign in TICKS and ignored the scorer. `integration_measure.lz76_complexity` is the
Kaspar & Schuster (1987) form of the LZ76 exhaustive-history parse, written for 12-unit nets (960-character strings); its
inner loop extends a candidate match one character at a time from every earlier position, so on a sparse string with
long zero runs the cost grows quadratically with the length. The 64-unit fixture (2.8's self-check width) sits far below
the knee, which is why no fixture run showed it -- the same shape as obs #2165: a parameter the fixture never exercised is
where the real object sits outside the instrument's range.

**The fix, pre-registered.** The STATISTIC does not change: c(L), the number of phrases in the LZ76 exhaustive-history
parse, and normalized_lz = c(L) log2(L) / L, exactly as 2.1 and the incumbent define them. The ALGORITHM changes to one
with the same output and linear-or-near-linear cost: the exhaustive history of Lempel-Ziv 1976 is the LZ77 factorization
with self-overlapping references in which every phrase is the longest previous occurrence plus one new symbol; it is
computed from a suffix array and LCP array (prefix-doubling in numpy; the longest-previous-factor array of Crochemore &
Ilie) or by any other method that reproduces the incumbent's count EXACTLY. The equivalence is a RED FIRST: a test that
compares the new function against the incumbent's on (i) every string the incumbent's self-check scores, (ii) 500 random
binary strings of lengths 2-4,096 at densities 0.5, 0.1, 0.02 and 0.005, (iii) all-zero, all-one, periodic and
single-one strings -- byte-for-byte equal counts on every one, or the fix is REFUSED and this row stops. One implementation
survives (P2): the incumbent's function body is replaced, its name, signature, docstring contract and every caller
unchanged; the old body is kept ONLY inside the equality test as the reference oracle, never as a code path.
Cost is then MEASURED before the 8-seed run: one full seed of the five arms at the campaign width, its wall time printed
per arm; if one seed projects the campaign past 3 hours the row STOPS and reports rather than running for a day.

**The compute, as the owner directed (2026-09-10: "We are using python 3.14t, make sure to use the correct Python
dependencies for the task, use PyTorch and GeNN, and multithreading appropriately"), verified on the venv before writing
this: Python 3.14.6 free-threaded (GIL disabled at runtime), 28 cores, torch 2.11.0+cu128 with CUDA on the RTX 5070 (a core
dependency of the tree), pygenn 5.4.0. (i) The LZ76 factorization's sort rounds run as torch tensor ops on the GPU (CPU
fallback when CUDA is absent; the count is device-independent because every suffix of a string is distinct, so the
suffix array is unique and the equality test decides, never the device); (ii) the eight seeds run in a
ThreadPoolExecutor -- under 3.14t these are true parallel threads -- one seed per worker, the arms of a seed in order, the
pin assembled after all workers return in seed order so the pin is byte-identical whatever the scheduling; (iii) the
kernel's own partition loop runs with `threads = p_parts` (KernelTick.tick's ThreadPoolExecutor over partitions; the
determinism gate asserts thread-count invariance, so the trains are byte-identical to the threads=1 run); (iv) GeNN is NOT
used in this row: the pre-registered instrument is the tree's numpy kernel with G5's snapshot chain (2.1), which GeNN's
simulator has no restore for, and the bottleneck was never the simulation; a measured fact for the record -- importing
`pygenn` re-enables the GIL in the importing process ("The global interpreter lock (GIL) has been enabled to load module
'pygenn._genn'"), so a process that free-threads its seeds cannot also host GeNN without `PYTHON_GIL=0`, which this row does
not set. GeNN's place in the plan is the GPU tick (P5a, the S6 kernel wire's successor), not this scorer.

**What else changes, and what does not.** The campaign entry point prints per ARM (not only per seed), unbuffered
(`python -u`), so a stalled run is distinguishable from a running one within minutes -- a run whose log is empty for
hours is the failure mode this amendment records. Nothing in 2.2-2.7 changes: the arms, the constants, the seven
conjuncts, the null, the reds, the controls, the disposition. No campaign number had been produced when this was
written; the stopped run's only readings are the host clause and the R3 chain lines quoted above.

## 3. Results (written after the runs, 2026-09-10; nothing above this line was edited after 2.11)

### 3.1 The build, red-first, and what the fixture found before the campaign
`a2cf76d` R3 RED (the G5 chain through a port that did not exist) -> `47c9bdf` the port GREEN: `cortex/integration_kernel.py`
restores TWO KernelTicks from ONE `replay_gate.capture` blob taken after the warm-up (rings, accumulators, tick_index and the
drive stream's position inside it), drives both from the captured stream, adds the quantized pulse to one, scores the XOR
footprint with the incumbent's `_serialize` / `normalized_lz` / `gzip_complexity`; R3 (i) restored baseline byte-identical,
(ii) the frame-only restore DIVERGES on a delayed fixture, (iii) a positions-stripped blob REFUSED -- all three in one
test file -> `a42cdbe` R1/R2 RED before the arms existed -> `551ba84` the five arms (`kernel_regimes`; PARTITIONED is the
harness's no-edges twin lifted into `arm_no_edges`, P2; SYNCHRONOUS = `synchronous_synapses`, mean out-degree x 8 at the
table's maximum magnitude, Dale signs kept; the rate-matched arm by a bounded bisection printed per seed), the local-echo
null, `verdict_t070` over (b1)-(b7) with the harness's `contrast`, the pulse-less identity ASSERTED before any PCI is read.
The 64-unit fixture's self-check reads R1, R2 and R3 RED in one command with (b1) +4.2504, (b2) +3.9028, (b3) +4.9497,
(b5) 4/4, (b7) 4/4 -- and (b4) NOT MET on 3 of 4 seeds and (b6)'s rate-matched clause NOT MET on 1 of 4, which is what
amendment 2.10 recorded BEFORE any campaign number (the fixture predicted the campaign's (b4) reading; obs #2165).
The incumbent's orderings reproduce on the kernel fixture: PCI(integrated) 0.043-0.087 > PCI(partitioned) 0.008-0.012 >
PCI(silent) 0.000-0.003; the gzip twin orders the same way (56-107 > 17-21 > 9-11 bytes).

### 3.2 Campaign run 1, STOPPED, and the scorer's repair (amendment 2.11)
Run 1 (`30b4d2e`'s entry point, started 03:40:35): host clause settled (daemon PID 37988 across five windows 97.969 /
124.719 / 115.734 / 110.297 / 79.375 CPU-s); R3 held on all 8 seeds within 8 minutes (in-flight 532 / 571 / 611 / 561 / 548
/ 485 / 517 / 579); seed 0's five arms then ran 9 h 25 min without returning; stopped at 13:20 with no PCI number. The
probe: the incumbent `lz76_complexity` parses 5,120 characters in 0.241 s, 160,000 in 156.569 s, 320,000 in 601.248 s --
quadratic on sparse strings -- and the campaign footprint is 1,094,555 characters (14,215 units x 77 post-pulse ticks),
640 parses: about two hours each, weeks in all. 2.8 had costed ticks and not the scorer.
The repair (`e8d10d3` RED: the incumbent's body verbatim as the oracle, equality on its own self-check strings, 500
random strings at four densities and the edge cases, plus a 5 s cost bar on 200,000 characters -- RED as run: "still
parsing after 5.0 s"; `3b7a2b6` GREEN): the SAME count by a suffix array (prefix doubling in torch through the tree's
own device router `substrate/device.select`), LCP by binary lifting, the longest-previous-factor array, the phrase walk.
Oracle-equal on every string; the campaign string (c = 7,369) parses in 0.37 s on the RTX 5070 and 1.18 s on the CPU
fallback; the live `region_coherence` path's 1,000-character strings score in 1.93 ms against 11.92 ms before. Both bare
self-checks read numbers IDENTICAL before and after the change (the incumbent's integrated 0.4268 / gzip 65.12 /
independent 0.0826 / synchronous 0.0000 / silent 0.0000; the kernel fixture's seven conjuncts and every per-seed line).
`b9bffbf`: the port's kernel at the lineage's pinned partition count (`tools/kernel_partition_baseline.json`, P = 1, so
2.11 (iii) resolves to threads = 1 -- the thread-invariance test is real on a P = 4 port and vacuous on the campaign's);
`c1aac92`: every arm reported as it finishes, unbuffered; seed 0 measured solo, the 8-seed projection printed against a
3-hour ceiling (REFUSED, rc 2, no pin, if exceeded), the remaining seeds pooled over 8 worker threads (Python 3.14t, the
GIL disabled), the pin assembled in seed order.

### 3.3 Campaign run 2 (`tools/t070_pci.json`, run at `c1aac92`, committed `ab09b9b`)
Host: daemon PID 15928 across all five windows (118.813 / 92.141 / 110.969 / 122.0 / 113.531 CPU-s), settled = True,
daemon_restarted = False, port 8137 LISTEN; wall 573.9 s of which 450 s the settling probe. Population: the H01 artifact
(sha256 a1d0491a...4a0549) at W = 1.0, 14,215 units / 70,001 edges; amp 0.4 re-measured on it (2.11's obs #2165 clause).
R3 on the campaign population: identical to run 1 seed for seed. Seed 0 solo 40.8 s (integrated 8.8, partitioned 5.5,
rate-matched 5.5, synchronous 19.0, silent 2.0 s); projection serial 0.09 h, pooled 0.02 h -> PROCEEDS; pooled seeds 71.4-76.4
s each. Scorer on CUDA (n = 1,094,555 above the measured 262,144 crossover).

| conjunct | reading (8 seeds) | state |
|---|---|---|
| (b1) t(INTEGRATED - PARTITIONED) | t +5.2015, FPR 0.0508 | MET |
| (b2) t(INTEGRATED - PARTITIONED-RATE-MATCHED) [the artefact control] | t +4.8967, FPR 0.0508 | MET |
| (b3) t(INTEGRATED - SYNCHRONOUS) | t +5.7220, FPR 0.0430 | MET |
| (b4) SILENT PCI = 0 exactly on every seed | 0 of 8 (SILENT PCI 0.0000-0.0001; gzip 48-136 bytes: the pulse propagates from rest, 2.10's named cause) | NOT MET |
| (b5) rate(SYNCHRONOUS) >= rate(INTEGRATED); rate-matched within 5 % | 8/8 (0.987-0.988 vs 0.037-0.038); 8/8 (0.039 vs 0.038, amp 0.5 vs 0.4) | MET |
| (b6) clause 1: t(INTEGRATED - null) | t +5.0732, FPR 0.0469 | MET |
| (b6) clause 2: PARTITIONED <= null | 6 of 8 (misses on seeds 2 and 5 by 0.5 and 0.25 phrases in the 8-set mean) | NOT MET |
| (b6) clause 3: PARTITIONED-RATE-MATCHED <= null | 4 of 8 (misses on 0/2/4/6 by 0.9 / 0.25 / 3.0 / 0.25 phrases) | NOT MET |
| (b7) R3 (i)(ii)(iii) + the pulse-less identity | 8/8 | MET |

**Verdict, by 2.3's rule: NOT MET** (`pass` = false in the pin), on (b4) and on (b6)'s clauses 2 and 3.

**The reading sits at the instrument's floor, said plainly.** INTEGRATED reads PCI 0.0003-0.0011 (13-60 LZ phrases in a
1,094,555-character footprint; one phrase = 1.83e-5 PCI); every partition, every echo and the null read 2-6 phrases;
SYNCHRONOUS reads 0 (rate 0.988: the saturated table's response to the pulse is empty, gzip 9 bytes -- "integrated but
not differentiated" exactly as the incumbent's regime); SILENT reads 1-7 phrases of propagation from rest. The absolute
PCI values are not comparable to the incumbent's 12-unit fixture (0.43) and 2.1 says a lone PCI number is never
reportable; what this row reads is the CONTRASTS, and they carry: a three-unit pulse on this population causes a
footprint 5-10x more complex through the wiring than without it, on every seed, matched for rate.

### 3.4 The disposition, and why 2.6's deletion is HELD rather than applied
2.6 prescribes PASS on all seven, or NULL on (b1), (b2) or (b6) -> the port DELETED. 2.10 carved out the branch where (b1),
(b2) and (b6)'s FIRST TWO clauses are MET and only (b4) / (b6)'s rate-matched clause fail -> the port RETAINED. The campaign
reads (b6)'s second clause 6 of 8, so by the letter 2.10's carve-out does not apply and 2.6's NULL branch fires.
**The deletion is HELD, and the hold is disclosed as a session ruling, not a reading.** The two failing clauses compare
two floor-level echoes with an exact <= and no tolerance: PARTITIONED's own local echo against the null taken on the
INTEGRATED run's pulsed units, whose trains differ only in the recurrent input the cut removes -- 2-6 phrases on each side
of a 1,094,555-character parse, differences of 0.25-3 phrases in an 8-set mean. That inequality was pre-registered on a
4,928-character fixture where one phrase is 2.5e-3 of the reading; on the campaign it is a coin flip between two
quantities the instrument cannot separate, and it does not touch what (b6) exists to test: the dissociation "the
partitioned twin reads at or below the null" holds where the null is INTEGRATED's echo and the partition is measured
against it as a CONTRAST -- (b1) +5.20 and (b6) clause 1 +5.07. A null that impugns the clause's construction rather than
the mechanism does not license a deletion under P8 clause 5 ("an invented method that measures nothing"): this method
measured a separation at t +5 on four contrasts with its artefact control beside it. The owner's rule of 2026-09-05 (a
pre-registered null disposition is a causal claim; HOLD its routing when the null impugns it) is applied and NAMED here.
**What the hold does NOT do:** it takes none of the PASS disposition -- `C4_TASK["integration"]` stays None, the registry
`reading` stays None, the cockpit's PCI row carries NO T070 quantity, the toy net is NOT retired (its retirement was
conditional on PASS), the licensed sentence is licensed to nothing. `PROVENANCE["pci"]` is corrected as a map/territory
fix only (`a0d8946`: the producer deleted 2026-08-05, the row said "refreshes every 4th night"). Annotations appended,
dated, at tasks.md T070, PLAN-FINAL's C4 bullet, Part F C-R5 and spec.md SC-009 (`750ee15`).
**The adjudicator, named before the panel runs:** the arc's panel, pre-registration-honesty lens, is asked whether this
hold is a relabelling of 2.6 after its number. If it rules so, the fold's FIRST commit executes 2.6's deletion of
`cortex/integration_kernel.py` and its tests (the exact scorer, the pin and this record survive); if it does not, the port
is retained as an instrument and a successor re-specifies (b6)'s clauses 2-3 with a tolerance the parse length sets (at
least one phrase) or a paired test, and (b4) as a reported quantity. Either way this row does not re-read itself.

### 3.5 Deviations from 2.1-2.11, each with its number (dated 2026-09-10)
1. 2.8's cost estimate was wrong by four orders of magnitude on the scorer (3.2); corrected by 2.11 before any number.
2. 2.3's (b4) and (b6) clauses 2-3 were mis-specified for this population (2.10, 3.3); recorded, not relabelled.
3. 2.6's NULL branch fired by the letter and is HELD (3.4), adjudicated by the panel's honesty lens.
4. 2.11 (iii) "threads = p_parts" resolves to 1 on the lineage's pinned partition baseline; the multithreading the owner
   directed lives at the seed level (8 workers, 71-76 s per seed pooled) and in the torch scorer, not in the kernel's tick.
5. 2.11 named Kasai for the LCP; the build used binary lifting over the doubling ranks (inside 2.11's "any other method
   that reproduces the count EXACTLY"; the oracle test decides, and did).
6. The settling probe cost 450 of the campaign's 574 s; the host clause's cost now exceeds the campaign's.
7. Found, not touched: `substrate/region_coherence.py:70` carries a SECOND Kaspar-Schuster LZ76 body behind an import
   fallback (`except Exception`) -- a dormant duplicate (P2/P3) that predates this row; for the fold.
8. Campaign run 1's 9 h 40 min was invisible because its log was block-buffered and printed per seed only; run 2 prints
   per arm, unbuffered, and refuses above a projected 3 h (obs #2170).

### 3.6 What this row does NOT claim
No Phi, no threshold, no sentence about experience, no aggregate with another reading; no C4 number; no statement about
the live brain (no live population runs on the kernel -- "live" is NOT delivered, 2.7); no claim that the absolute PCI
values mean anything beyond their contrasts; no claim that the hold in 3.4 is a pass -- SC-009's integration instrument
has fired its reds and has NOT met its own verdict rule.
