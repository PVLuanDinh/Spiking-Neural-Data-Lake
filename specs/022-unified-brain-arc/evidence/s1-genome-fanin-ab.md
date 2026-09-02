# FR-033 A/B: map-drawn initial fan-in vs the random candidate pool — a measured NULL (2026-09-02)

Measured on `main` after `c4a64613` (audit §8 rank 1b: "wire `genomic_bottleneck` and run its A/B" —
owner directive the same day), pinned host, `.venv-lake`, CPU, the one live grower
(`cortex/neurogenesis_engine.NeurogenesisSelfImprove`) run standalone — no brain, no daemon, no lake
write (the `--lake` arm READS the live lake's growth record once). Reproduce:

    PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.genomic_bottleneck --ab --seeds 24 --warm 400 --cont 400

This command exists only at the commit that landed this file and the wire — **`847c5f6a`** — because
the following commit removed the module and its `--ab` runner per the disposition below; reproduce by
checking out `847c5f6a`.

## What was wired

`cortex/genomic_bottleneck.py`'s docstring pre-registered the wire and the verdict rule: swap the random
candidate pool inside `hippocampus/ng_snn.py::NGSNN.fit` for map-drawn fan-in, reached through the
engine's refit; "if that A/B shows map-drawn fan-in does not beat the random pool on the real earned
substrate, this module is DELETED, not parked." Built exactly that:

- `NGSNN.fit(..., candidates=None)`: each scale's pool is drawn from the seeded stream exactly as
  before (the stream advances identically with or without the hook, so arms differ only in the rows
  the hook replaces); the hook may substitute rows and returns per-row provenance; the admitted units'
  provenance lands in `last_provenance`. `candidates=None` is byte-identical to HEAD (pinned by
  `tests/test_genome_candidates.py`: same `W`, same `b`). **Disclosure (P8 clause 4): the engine's
  DEFAULT path no longer passes `None` — it passes a genome-free `CandidatePool`, and its placement
  bookkeeping changed.** Weights on the default path are byte-identical to HEAD (runtime A/A, HEAD
  extract vs tree, seed 2, 140 ticks: same `W` sha, same `b` sha, same accuracy 0.7174), but the
  placement STREAM is not (placement sha differs; first unit `Layer 3` at HEAD vs `Layer 2` in the
  tree): placements are now drawn pre-fit per candidate under a seed folding tick, step and scale,
  where HEAD drew them post-hoc under `seed*1_000_003 + n_before`. That stream is persisted in the
  lake blob and has no reader, so no charge moves — which is exactly why the pinning test cannot see
  the change and this sentence has to.
- `genomic_bottleneck.CandidatePool`: draws the candidates' placements BEFORE the fit (through the
  engine's own `place()` call site, so the coverage census keeps its site and the placement that
  generated a unit's fan-in is the one the admitted unit keeps); with a fitted genome it replaces a
  candidate's random row with `init_fanin(feature_tags, unit_tag(placement))` wherever the tag lies in
  the fitted span, rescaled to the pool's envelope at that scale (`lam·sqrt(d/3)`) so the genome
  supplies DIRECTION only and the multi-scale search stays the pool's; out-of-span tags keep their
  random row (`init_fanin`'s refusal, counted).
- `NeurogenesisSelfImprove(init_pool=...)`: `random` (the incumbent pool; placement bookkeeping
  changed as disclosed above), `genome` (the map refitted every refit on the earned substrate — `GenomicMap.fit`, the
  declared `DIRECT_SOLVE`), `genome_shuffled` (the same map fitted on tag-SHUFFLED units: the control
  that keeps the map's magnitude and span and destroys its tag→weight association). Default `random`.
  The pool census (`drawn / genome / fallback / degenerate / admitted / admitted_genome`, the fit's
  rank and RMSE) rides on `state()["pool_stats"]`.

## Protocol (pre-registered before the first run)

The verdict rule was written into the A/B runner's docstring before any run; the runner file that
carried that docstring's birth time was folded into `genomic_bottleneck.py --ab` before commit, so the
evidence of pre-registration is the FIRST run's own output: `fr033_ab.txt` (archived beside this
session's panel record) prints `VERDICT: genome WINS -- majority 5 of 8 needed on BOTH comparisons
(pre-registered)` at 11:28, before any 24-seed run existed.

Paired by seed. Warm: grow a substrate with the incumbent pool for 400 ticks (the EARNED substrate).
Arms: restore it into three fresh engines with the same seed (identical continuation stimulus) and
run 400 more ticks. Statistic: continuation-mean held-out next-mode accuracy; verdict = genome WINS
iff it beats random on a majority of seeds AND beats the shuffled control on a majority. `--lake`:
the live brain's own substrate from the lake as one more warm arm.

## Result

| run | genome vs random (wins–losses, mean Δ) | genome vs shuffled | shuffled vs random | verdict |
|---|---|---|---|---|
| 8 seeds (first run) | 5–3, +0.0066 (SE 0.011) | 5–3, +0.011 (SE 0.011) | 4–4, −0.005 | "WIN" under the rule — one-sided sign-test p=0.36 (two-sided 0.73) |
| **24 seeds** | **13–11, +0.0020 (SE 0.0054)** | **12–12, +0.0016 (SE 0.0054)** | 12–12, +0.0004 | **NULL** |
| lake substrate (5 units, one sample) | 0.666 vs 0.628 | 0.666 vs 0.642 | — | one sample; not a verdict |

Standard errors are population-sd (ddof=0) over the 24 paired deltas; sample-sd gives 0.0055. Read the
rule, not the t-statistic: genome-vs-random 13–11 MET the majority-13 bar exactly; the NULL is carried
by the second half, 12 < 13 against the shuffled control. The p-values are one-sided because the rule
is directional (a win is "beats").

**The 24-seed run was decided after the 8-seed WIN, not before** (P8 clause 4). It is a strict superset
under the unchanged rule — seeds 0–7 of the 24-seed run are byte-equal to the 8-seed run (`acc_mean`,
`units_end`) — so it is a power increase, not a re-roll; but the direction of that choice ran against
the method and is stated here for that reason. The 16 added seeds alone: genome vs random 8–8, mean
−0.0003.

Per-arm means over 24 seeds (continuation-mean acc / final acc / units at end / admitted):

| arm | mean acc | final acc | units | admitted | vs random | admitted from genome rows |
|---|---|---|---|---|---|---|
| random | 0.6000 (sd 0.043) | 0.711 | 13.7 | 1,527 | — | 0 |
| genome | 0.6019 (sd 0.037) | 0.707 | 17.6 | 2,263 | +48% | 1,395 (genome rows 593,443 of 681,480 drawn; 84,557 out-of-span fallbacks) |
| genome_shuffled | 0.6003 (sd 0.035) | 0.701 | 16.7 | 2,216 | +45% | 1,255 |

The 8-seed "win" did not survive threefold power: the effect is +0.002 with a standard error of 0.005 —
zero within error — and it is indistinguishable from the shuffled control, which is the second half of
the rule. The genome arm admits **48% more units** for the same accuracy, and the per-item attribution
(P8 clause 3) says this is NOT the map's rows passing the bar more often — it is the opposite:

| per drawn row | random arm | genome arm | shuffled arm |
|---|---|---|---|
| admissions per drawn row | 0.419% | 0.332% | 0.329% |
| rows drawn per refit | 760 | 1,420 | 1,404 |
| inside the genome arm: map-drawn rows admitted | — | 1,395 / 593,443 = **0.235%** | |
| inside the genome arm: random-sourced (fallback) rows admitted | — | 868 / 88,037 = **0.986%** | |

A map-drawn row is admitted about 4x LESS often per draw than a random row in the same arm. The extra
admissions come from the fit loop drawing 1.87x more candidate rows (24 of 24 seeds; it runs longer
before its stopping rule), and the tag-shuffled control does the same (+85% rows, +45% admissions), so
the effect is a property of map-shaped rows' magnitude/span, not of tag identity — and the extra units
buy no held-out accuracy. P0's line, "growth is not improvement", measured.

## What the review panel found about the instrument (P8 clause 4, said here rather than smoothed)

The three-lens panel on this landing measured the A/B itself, not just its numbers. Every finding
below reproduces from the archived outputs (scripts and raw outputs archived with the session's panel
record, outside the repo because the module they drive is deleted in the next commit).

1. **The pre-registered majority rule is a weak detector.** Permuting arm labels within seed on the
   archived 24-seed data, "majority on BOTH comparisons" fires **23.1%** of the time under
   exchangeability (18.3% at 8 seeds — the 8-seed "WIN" was one of those). At 24 seeds the verdict
   was one seed-flip from "WIN" (12 vs the 13 needed against shuffled). The NULL therefore rests on
   the effect ± SE (+0.002 ± 0.0055, t = 0.36) with a working positive control — under the same
   protocol a mode-centre oracle pool wins 8–0 at +0.074 ± 0.010 and the warm substrate's own rows
   win 6–2 at +0.030 ± 0.012, so the differencer is not blind — and not on the rule as written.
   A rule with a significance level (sign ≥ 18/24, or a paired test at α = 0.05) is what a future
   A/B on this grower should pre-register.
2. **The "continuation" is a cold restart.** Each arm restores the warm blob into a FRESH engine
   (empty buffer, fresh stimulus chain), so the arms replay the warm stream from the start of the
   buffer: warm 12.4 units fall to 5.3 at the first continuation refit (57% pruned), 19 of 20
   continuation refits sit below the warm accuracy, and the statistic is a ramp from 0.47 to 0.69 that
   is identical across arms. The panel re-ran the three arms CARRYING buffer, chain state and tick
   count (a true continuation, 24 seeds, 400/400): **genome vs random 4–20, −0.0154 ± 0.0030
   (t ≈ −5)**; shuffled vs random 6–18, −0.0117 ± 0.0028; genome vs shuffled 10–14, −0.0038 ± 0.0030.
   On a true continuation the map-drawn pool is measurably WORSE than the incumbent.
3. **The shuffled half of the rule is null by construction on this grower.** A candidate's placement
   is drawn from the H01 prior under a seed of (engine seed, tick, step, scale) — independent of the
   row's weights — so there is no tag→weight association in the earned substrate for the shuffle to
   destroy: on 8 warm substrates the true-tag fit RMSE sits at a uniform percentile of its own
   unit-tag permutation null (0.13–0.86), it is exactly permutation-invariant when units ≤ rank
   (both arms report `genome_rmse` 2.42937 at 4 fit units), and the feature-tag span (bias + 6 index
   bits) holds only 8–17% of earned fan-in energy. Both map arms are fits of noise, hence 12–12.
   FR-033's premise is refuted structurally for this grower, which is a stronger reason to delete than
   the measured tie.
4. **The incumbent "random" pool re-offers the same candidate rows at every refit** (`_refit`
   constructs `NGSNN(seed=self.seed)` each time — a HEAD property, but it is this A/B's comparator);
   the map arms' rows change every refit. Against a novelty-matched control (fresh uniform rows per
   refit and step, no tags): genome vs random_fresh 7–17, **−0.0140 ± 0.0070**; shuffled vs
   random_fresh 5–19, −0.0157 ± 0.0066; and random_fresh vs random 14–10, +0.0160 ± 0.0075. The
   last number is OPEN work on the grower, not on this module: a fresh candidate pool per refit beat
   the incumbent at t ≈ 2.1 and deserves its own pre-registered A/B.
5. **The unit surplus is a measured cost, not a description.** `units_end` genome vs random 16–7,
   +3.96 ± 1.42 (t = 2.78); accuracy per unit 8–16, −0.011 ± 0.004 (t = −2.93). `acc_final` is a
   60-row slice with 1/60 granularity (11 ties of 24) and is not usable as a sign statistic.
6. The raw-magnitude variant (no envelope rescale) was also run: genome_raw vs random 14–10,
   +0.0037 ± 0.0048 — null as well; the rescale is hygiene (raw/envelope norm ratio median 9.6 at
   λ = 0.1, max 6.4e4), not a neutering.
7. The lake arm shows the wire restores and runs on a real lake blob and nothing about the verdict;
   the live lake now holds 22 units (the archived arm's 5 is stale), and it starts from the same
   restart.

## Disposition

**NULL — and on the corrected protocol and the novelty-matched control, a measured LOSS.** The
module's own pre-registered disposition applies: `cortex/genomic_bottleneck.py` is DELETED,
not parked** (P8 clause 5; the deletion is its own commit so this measurement stays reproducible at the
commit that carries it, and the two commits ship in ONE push — commit A alone would be the
parked-behind-a-flag state P8 forbids). What was true of the mechanism stays true — its algebra,
refusals and permutation nulls all held in its self-check — and what it claimed about the real
substrate is now measured: on this grower the tag→weight rule carries nothing the random pool lacks.

**What survives (P2: one mechanism):** nothing of the seam. The pre-fit candidate placement, the
`candidates=` hook in `NGSNN.fit`, `last_provenance`, `CandidatePool` and the engine's `init_pool` /
`pool_stats` / `_candidates` / `_place_candidates` existed to feed the map; with the map deleted their
only effect would be P×scales `place()` calls per greedy step on the default path plus a changed
persisted placement stream with no reader — the shape the 007 wire was reverted for. The commit after
`847c5f6a` restored the pre-wire post-hoc placement bookkeeping and removed the hook (the engine and
`ng_snn` are byte-identical to `c4a64613` again); FR-033's map clause in `spec.md` is marked
withdrawn on measurement (a MUST is withdrawn; its premise is what was refuted).

FR-033 as written ("grown units MUST receive initial weights from the genomic-bottleneck map") is
therefore refuted on measurement for this grower; the requirement's honest residue — "no grown unit
starts from an UNDECLARED initializer" — is met by the declared random pool it always had. The
spontaneous-activity half of FR-033 (T045) is untouched by this result.

## Caveats, said before anyone asks

- The task is the engine's own: a finite 5-mode Markov chain over 64 multimodal features (the audit's
  §3 item 1). A bounded task with a greedy, residual-driven admission rule is exactly where an
  initialization prior has the least to add; a richer stimulus stream (audit §8 rank 2) could change
  this, and if it does the measurement is the same tool with a different engine behind it. That is a
  reason to re-run after rank 2, not a reason to keep the module meanwhile.
- The genome supplied direction only (rows rescaled to the pool's envelope). The shuffled control shares
  that design, so the comparison isolates tag identity; a magnitude claim was never made.
- The feature tag is a bias + 7-bit positional code of the feature index — the module's docstring itself
  named this geometry as "a variable, not a given". A different feature tagging is a different method
  and would owe its own pre-registration.
- 24 paired seeds bound an effect at roughly ±0.01 accuracy; a smaller true effect is not excluded and is
  not worth a module.

## What this closes and what stays open

Closes: rank 1b. `placement` still has no live reader — the wire that would have made it one measured
a null — so `tests/test_placement_charge_null.py` stays a pinned null and the coverage gate's disclosure
stays true. The two dormancy exemptions that named this wire retire with it. Open: giving placement a
consumer remains a build, not a wire, and the audit's ranks 2–4 stand as ordered.
