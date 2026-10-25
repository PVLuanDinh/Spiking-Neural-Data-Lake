# E1b pre-registration -- FROZEN at the commit that adds this file, before any E1b brain run

**Status: FROZEN.** This file is frozen at the commit that introduces it, which precedes every E1b brain run and every E1b
datum. From that commit it is append-only: later changes are dated addenda, each disclosed in the report; the text below the
rule is never edited in place. Nothing below was run: no E1b brain exists yet.

**Binds to harness module:** `src/snn_data_lake/instruments/rpe_charge.py` (governed; tests `tests/test_rpe_charge.py`). The
body below was written against E1's lane scripts and names them (`e1b_probe.py`, `e1b_replay.py`, `e1b_compare.py`,
`run_arm.ps1`, `batch.sh`); in the governed tree those jobs are subcommands of the one module: `e1b_probe.py` = `rpe_charge run`,
`e1b_replay.py` = `rpe_charge replay`, `e1b_compare.py <tier>` / `cross` = `rpe_charge compare` / `cross`. The module keeps every
constant, bar and rule of the body below; where the port had to choose, the choice is in the module docstring and the report,
never a change to this text.

**Owner words held fixed (session prompt, 2026-10-23, verbatim):** "run E1b (fresh seeds 8-15, BOTH perception tiers, harm
pre-registered for _plan_w) before any change to them". Seeds are therefore 8-15 (board #4615 stays open). Board #4616
(does a production-only harm suffice to delete) takes the draft's default, `SINGLE_TIER_HARM_DELETES = False`: a single-tier harm
is a tier finding the owner decides on. Both are constants in the module (`SEEDS`, `K_MIN`, `BAR`, `MDE_F`,
`SINGLE_TIER_HARM_DELETES`), so a later owner answer becomes a separately pre-registered extension, never a post-hoc change
to this file. Reviewed text: Revision 2, both reviewers agreed (board #4697 / #4698). The campaign runs once, on the pushed
head of the engine arc after the arc's battery (integrator); this lane builds the harness and does not run it.

---

# DRAFT PREREG -- E1b (thread E1B): does the E1 harm replicate, on both perception tiers?

**REVISION 2, 2026-10-23 (lane e1b-prereg-r2), still before any E1b datum.** Folds reviews #4623 (red team, rv-e1b-a) and #4622
(policy, rv-e1b-b); one row per finding in `REVIEW-RESPONSE.md`. Integrator rulings applied: tree = post-arc origin/main
(sha pinned, section 1); full both-tier budget; degenerate seeds stay in the primary, sensitivity reported (section 3). What
changed, in one list: (1) claimed effects are frozen literals, labelled E1 no-encoder numbers (section 5); (2) a failed planted
control makes an endpoint VOID, never "not met" (section 3, 6); (3) the cross-tier ".0025" is withdrawn: tiers share seeds and are
correlated, so "both tiers" now needs a pre-registered cross-tier agreement rule and is worded "effective p between .0025 and
.05" (section 3, 6); (4) a BRIDGE (E1's seeds 0-7, no-encoder tier, new tree, `real` + `pfrozen`, 16 runs) makes tree-vs-seed
attributable (section 2, 6); (5) encoder-call counters and a tier-effect test make "tier" a measured difference, not a label
(sections 1, 4); (6) a planted-HARM stream and a brain-free compare fixture give the harm direction a reachable green (section 4);
(7) one NULL-RECORD and one RETEST-RECORD per method, tier-split results write none (section 6); (8) refusals are explicit
exits, not `assert`s; tree sha equal across tiers; `SNN_PC_CKPT` pinned; unlisted harness side effects listed (section 0,
HARNESS-CHANGES). **The two OPEN owner questions (board #4615, #4616) are parameters, not rewrites**: see section 10 for the
exact lines that change.

DRAFT, written 2026-10-23 before any E1b brain exists and before any E1b datum. Frozen-style: when the integrator
posts the freeze, this file is append-only (dated addenda, each disclosed in the report). Every clause of E1's
PREREG.md (`/mnt/d/AI-project/lanes/e1-rpe-charge-1019-work/PREREG.md`, frozen sha256 d69cd655...ca38fa, addenda 1-6) holds
unless section 0 lists a change. Owner (session prompt, 2026-10-23, verbatim): "run E1b (fresh seeds 8-15, BOTH
perception tiers, harm pre-registered for _plan_w) before any change to them".

Binding texts read as files (checkout `/mnt/d/AI-project/lanes/p8-restack-1023`): `docs/POLICIES.md` P8 + the null-scale
amendment (L907-957), P5b (L657); `specs/022-unified-brain-arc/evidence/learning-rules-1019.md` sections 1, 2 item 1, 7,
8; `engine/brain.py` L194 (`_model_asset`), L655-790 (encoder load), L3980-3987 (`state()` tier keys).

Anchors re-checked in the P8-arc tree (read-only grep): `_rpe` init L410, update read L1504, write L2326;
`_plan_lr` init L613, single reader L1506; `_pfc_bias(self, unit, pattern=None, qa=None)` L1363; `ValenceLearner`
`valence` L60, `dopamine` L74, `learn` L80 in `plasticity/valence_stdp.py`; `self._valence = ValenceLearner()` L2304.
All unchanged from E1's tree. The line numbers must be re-confirmed on the tree actually used (section 1).

## 0. Changes from E1 (complete list)

| # | E1 | E1b | why |
|---|---|---|---|
| C1 | seeds 0-7 | seeds 8-15 | fresh seeds (owner) |
| C2 | one campaign; tier unrecorded (cwd-relative defaults; found no-encoder post hoc) | TWO campaigns, labelled `production` and `no-encoder`, tier pinned by ABSOLUTE env paths, hashes checked, tier printed and stored with every run, a run whose reported tier differs from the requested one is REFUSED | the working directory was a silent input (learning-rules section 8) |
| C3 | `_plan_w` harm was an unregistered signal (t +2.82, "mirrored bar") | `_plan_w` HARM endpoint H-P1 pre-registered; ValenceLearner harm endpoint H-V1 (and H-V2) pre-registered, "frozen beats learning" as a direction with a bar | owner; learning-rules section 2 item 1 |
| C4 | dispositions in the E1 vocabulary (CHARGED / DORMANT) | dispositions under P8 + null-scale amendment (section 6): harm / NULL-RECORD + one 10x re-test / INCONCLUSIVE / tier finding | amendment landed in this arc |
| C5 | power only reported for E-P2 | mde at S = 8 stated per endpoint in advance (section 5); an endpoint whose mde exceeds its claimed effect is INCONCLUSIVE when not met, by rule | amendment item 4 |
| C6 | code tree d501c7aa | code tree = origin/main tip once the P8 arc is merged; its sha recorded in addendum 1 BEFORE the first run; BOTH tiers use the same sha | the live code |
| C7 | repeat arm `real_0_r2`, pc seeds 0-1 | `real_8_r2`, pc seeds 8-9, PER TIER | follows C1, C2 |
| C8 | brain run order seed-major | stage-major, so a battery or timeout still yields both tiers' primary endpoints first (section 7) | partial-result safety |
| C9 | `ValenceLearner` objective endpoints only positive direction | H-V1/H-V2/H-V3 added in the harm direction; E-V1/E-V2 stay as secondary | C3 |
| C10 | planted-stream control seeds 0-7 of the replay | planted-LEARNING control unchanged (seeds 0-7, tier-independent); ADDED a planted-HARM stream (section 4) | keeps the instrument identical; gives the harm direction a reachable green (R2-M5) |
| C11 | none | BRIDGE: `real` + `pfrozen`, seeds 0-7, no-encoder tier only, on the new tree (16 runs); compared with E1's own recorded series | tree and seeds both change (R2-M3) |
| C12 | no counter of encoder use | per-run `encoder_calls` (vision, audio, pc), `percept_ticks`; cross-tier "tier effect present" test (section 4) | the tier label alone does not prove the tier acted (R2-M4) |
| C13 | probe prints and stores what E1 had | ALSO: `state()` is called after the output dict is built and copied (never before); cwd = the new worktree; `SNN_PC_CKPT` pinned to a missing path in both tiers; refusals are explicit exits (`python -O` safe); `E1B_TREE_SHA` + clean-tree flag stored | harness side effects the first draft left unlisted (R2-m1, m2, m7, m8) |

Unchanged on purpose: N = 600 ticks, held-out window H = 0-based tick index >= 300 (1-based 301..600), df = 7, bar 1.895
(one-sided .05), "at least 5 of 8 seeds in the direction" (a tie is NOT in the direction), arms (`real`, `vfrozen`,
`vshuf`, `pfrozen`, `pshuf`, `pc`, the repeat arm) with E1's exact patch code (E1 PREREG "Exact patch code", incl. the
arm-instance scoping of addendum 1: only `ARM._valence` is patched), permutation rule `random.Random(seed*1009+17)`,
the replay `e1_replay` and its planted control. N is NOT re-derived by a tick rule: a changed N would break comparability
with E1; if production runs are slower, claims are renewed (section 7), N stays 600.

## 1. Code tree and the two tiers

- Tree (integrator ruling): a NEW git worktree `D:\AI-project\lanes\e1b-rpe-harm-1023`, checked out DETACHED at ONE full 40-hex
  sha = the post-P8-arc origin/main tip. That sha is written to `TREE_SHA` in the work folder and into addendum 1 BEFORE the first
  run; `run_arm.ps1` refuses (exit 3) when `git rev-parse HEAD` differs from it or `git status --porcelain` is non-empty. If the
  arc is NOT merged when the batch is released, the batch does not start (no fallback to a pre-arc tree is pre-registered; the
  integrator re-rules). Made by the integrator or the run lane, not by this draft. Addendum 1 re-confirms the anchors above.
  The scripts live in the NEW work folder `D:\AI-project\lanes\e1b-rpe-harm-1023-work` (never the E1 folder).
- "Replication" wording: the tree differs from E1's (d501c7aa) and the seeds differ, so a MET on seeds 8-15 is "a second sample on
  the same tier" and is called a replication of E1's signal ONLY when the BRIDGE (section 2) shows the new tree reproduces E1.
- `production` tier definition: BOTH e-prop encoders and both packs loaded; `SNN_PC_CKPT` is pinned to a missing path in BOTH
  tiers (the pc cortex is a fifth model asset; `perceive_pc()` is never called in a tick, brain.py L1206-1207; counted to be 0,
  section 4), so `pc_cortex_tier == "stdlib"` in both tiers and is stored.
- `production` tier: the brain with BOTH frozen e-prop encoders and both packs, loaded through four ABSOLUTE env paths (plus the fifth pc pin below)
  pointing at the main checkout's gitignored `models/`:
  `SNN_EPROP_CKPT=D:\AI-project\spiking-neural-data-lake\models\eprop_nmnist.pt` (sha256 prefix 9b471619e39acfa0),
  `SNN_EPROP_AUDIO_CKPT=...\models\eprop_shd.pt` (6b32f9eb38beeca3), `SNN_PERCEPTION_PACK=...\models\nmnist_pack.npz`,
  `SNN_SHD_PACK=...\models\shd_pack.npz`. (Prefixes verified against the files on 2026-10-23, read-only `sha256sum`.)
  Required report: `perception_tier == "eprop-snn"`, `perceiving_real == True`, `perceiving_audio == True`.
- `no-encoder` tier: the same four variables (plus the fifth pc pin below) pinned to ABSOLUTE paths that DO NOT EXIST
  (`D:\AI-project\lanes\e1b-rpe-harm-1023-work\NO-ENCODER\<name>`). Not an empty string: PowerShell deletes a variable set to
  `""`, and `_model_asset` (brain.py L194) then falls back to the repo-root default, which is only accidentally empty in a
  worktree. Required report: `perception_tier == "stdlib"`, `perceiving_real == False`, `perceiving_audio == False`.
- No open-weight model enters the brain (owner 2026-10-23, Q1): the two encoders are the project's own e-prop-trained
  checkpoints, already in the production brain; E1b adds no model, teacher, frozen external encoder or converted weights.
- Fifth pin (R2-m7): `SNN_PC_CKPT` = `D:\AI-project\lanes\e1b-rpe-harm-1023-work\NO-ENCODER\pc_cortex_nmnist.json` (missing) in BOTH tiers.
- Preflight (before any brain is built): the five variables are set and absolute; production: both checkpoints exist and their
  sha256 begins with the pinned prefix, both packs exist, the pc path does not; no-encoder: none of the five paths exists. Any
  failure: print `REFUSED`, exit code 3, no brain, no result file. Refusals are explicit `if`/`os._exit(3)`, never `assert`
  (R2-m1); the probe also refuses when `sys.flags.optimize != 0` and stores `optimize=0`.
- Post-construction check: read the brain's own tier fields (`b._perception_tier`, `b._perceiving_real`,
  `b._perceiving_audio`, the attributes `state()` reports at brain.py L3980-3987); compare with the requested tier; mismatch ->
  `REFUSED`, exit code 3, no ticks. After the last tick the run also stores `b.state()["perception_tier"]`; compare asserts it
  equals the attribute (validity). `perception_tier` is printed on the `TIER` line and the `DONE` line and stored in every JSON.
  The label alone is NOT the tier's proof (R2-M4): every run also stores `encoder_calls = {vision, audio, pc}` (instance-scoped
  wrappers on `_perception_enc.classify/encode`, `_audio_enc.classify/encode`, `perceive_pc`, recording then re-raising) and
  `percept_ticks` (ticks with a non-None `_real_percept` / `_audio_percept`); section 4 requires production > 0 and no-encoder == 0.
- Reachable reds, run BEFORE the first brain run (they build no brain): (R1) `--tier production` with the five variables pinned
  to the NO-ENCODER paths -> exit 3 at preflight; (R2) `--tier no-encoder` with the production paths -> exit 3; (R3) a pure
  function `tier_verdict("production", {"perception_tier": "stdlib", ...})` returns not-ok; checked at import by an explicit
  `if ...: os._exit(3)`, so the post-construction check cannot be silently disabled (and survives `python -O`). The campaign
  is void unless `logs/tier_red.log` holds both `rc=3 result_file_exists=False` lines; `e1b_compare.py` checks the file and
  VOIDs both tiers when it is absent (R2-m7, no longer review-enforced).

## 2. Arms and runs (per tier)

`real`, `vfrozen`, `pfrozen`, `vshuf`, `pshuf` x seeds 8-15 = 40 runs; plus the repeat arm (`real` seed 8 again, `real_8_r2`),
`pc` seeds 8 and 9 = 43 runs per tier, 86 for both. The learning-rules note's "40 runs, about 35 min" is ONE tier's core arms;
both tiers is 86 runs, about 60-70 min at E1's ~38 s per no-encoder run, longer if the encoders cost per tick (unknown until
the first production run; see OPEN-QUESTIONS Q2). Shuffle arms read the SAME seed, SAME tier `real` series.
BRIDGE (C11, R2-M3): no-encoder tier only, `real` + `pfrozen` seeds 0-7 on the new tree = 16 runs (about 10 min), results in
`results/bridge/`, tier label `no-encoder` (E1's tier). Total 102 runs at S = 8. Bridge readout, computed from JSON, no bar of
its own: (i) per seed, are the new-tree `real` and `pfrozen` reward and action series IDENTICAL to E1's recorded ones
(`/mnt/d/AI-project/lanes/e1-rpe-charge-1019-work/results/{real,pfrozen}_<s>.json`, read-only)? "tree-equivalent" = identical on
all 8 seeds for both arms; (ii) otherwise "tree moved the run" and H-P1's t and sign count are computed on the bridge (E1's seeds,
new tree) and reported beside E1's t +2.82. The bridge changes only the WORDING of the non-replication dispositions (section 6).

## 3. Endpoints (statistic: paired one-sample t over the 8 per-seed differences, df 7, bar 1.895, and at least 5 of 8 seeds
strictly in the direction; every per-seed value reported; computed by `e1b_compare.py` from JSON only)

Notation: r_X(s) = mean reward over H of arm X, seed s. MAE_X(s) = mean |pred_t - r_t| over H (pred before learning on tick t).
Replay quantities (a) learn, (b) frozen, (c) shuffled are E1's `e1_replay` values on the same seed's `real` series.

PRIMARY (harm; the dispositions rest on these):
- **H-P1 (`_plan_w` harm)**: d_s = r_pfrozen(s) - r_real(s). MET iff t(d) > 1.895 AND #{d_s > 0} >= 5. "Harm replicates" = the
  neutered rule earns MORE held-out reward than the live rule.
- **H-V1 (ValenceLearner harm, same-stream replay)**: d_s = MAE_(a)(s) - MAE_(b)(s). MET iff the replay's planted control passes
  AND t(d) > 1.895 AND #{d_s > 0} >= 5. "Frozen beats learning" on the identical stream. If t and the sign count pass but the
  planted-learning control or the planted-HARM control (section 4) fails, the status is **VOID (control failed)**: never MET,
  and never "NOT MET" either (R2 red-team M1); it feeds no NULL-RECORD and no deletion candidate.

SECONDARY (reported with the same rule; they change the wording of a disposition, never create one):
- **H-V2**: d_s = MAE_(a) - MAE_(c) (shuffled-reward twin beats the learner: learning from the true pairing is worse than learning
  the mean). **H-P2**: d_s = r_pshuf - r_real. **H-V3 (in-brain)**: d_s = MAE_real - MAE_vfrozen.
- E1's objective endpoints, positive direction, unchanged: **E-V1** (t(MAE_vfrozen - MAE_real) and t(MAE_vshuf - MAE_real),
  both > 1.895), **E-V2** (replay: t(b - a) and t(c - a) both > 1.895 plus planted control), **E-P2** (t(r_real - r_pfrozen) and
  t(r_real - r_pshuf) both > 1.895), **E-P1** (pfrozen flips above the repeat floor on >= 6 of 8 seeds), **E-V3** (flips, reported).
- Reported, no bar, no disposition: r_vfrozen - r_real per seed; sign counts (7 of 8 = one-sided binomial p .035, 8 of 8 = .004,
  6 of 8 = .145, 5 of 8 = .363); the sensitivity of each primary t with a seed excluded whose mean reward is below -0.9 in
  every arm of that tier (E1 seed 1 was such a seed). The primary uses ALL 8 seeds; no seed is excluded (integrator ruling:
degenerate seeds stay in the primary, the sensitivity is reported beside it, and a primary that is MET only with the seed
included or only without it is flagged "seed-sensitive" in the report, wording only).
- What the frozen arms are (R2 red-team m6): `pfrozen` holds `_plan_w` at its init 0.1 (`_plan_lr = 0`), so H-P1 tests "the
  learned w trajectory vs w fixed at 0.1", not "the update rule vs no rule". `vfrozen` makes the valence constantly tanh(0)=0, so
  RPE equals raw reward, which changes `da` into `_plan_w` and, through `act`, the action stream: H-V3 compares two behaviours,
  not only two predictors, and stays secondary. H-V1 (replay) is the clean predictor comparison.

Family: two primary endpoints per tier, each one-sided .05, no correction. The first draft's ".0025 for harm on both tiers" is
WITHDRAWN (R2 red-team M2, policy M3): both tiers run the SAME seeds on the same code, so the per-seed differences d_s are
correlated across tiers; the false-replication rate of "MET on both" lies between .0025 (independent) and .05 (perfectly
correlated) and no value below .05 is claimed. Pre-registered cross-tier requirement for the words "harm on both tiers": the
sign of d_s agrees across the two tiers on >= 6 of 8 seeds AND both tiers' individual bars hold; `e1b_compare.py cross`
prints the per-seed sign table, the Pearson correlation of d_s across tiers (reported, no bar), and whether the requirement
holds. "Harm on either tier" is not a pre-registered claim (null ~.10).
Planted effect sizes (the claimed constants of section 5) are E1 no-encoder-tier numbers and are frozen as literals, never
recomputed from E1b's data (policy M1).

## 4. Harness validity (all must hold per tier, else that tier's verdicts are VOID and the report says so)

E1's set: `errs == 0` and `swallowed_exceptions == 0` in every run (wrappers record, then re-raise; instance-scoped);
recorded series length == learn calls; every patched arm's call counter > 0; pfrozen `_plan_lr == 0.0`; in every `real` run
`abs(pred - (r - rpe)) < 1e-9` and `pred` not None; repeat arm: 0 action flips and identical recorded series vs `real` seed 8;
`pc` flips > 0 on seeds 8 and 9; replay planted control PASS (t(frozen-learn) and t(shuf-learn) > 1.895); foreign-instance
`valence`/`learn` counts reported per run (any value allowed). Added: every JSON carries `tier.requested == tier.reported-as-
required` for its campaign (`production`: eprop-snn, real, audio all true; `no-encoder`: stdlib, both false); `tier.state_api ==
tier.reported`; the sha256 prefixes (production) are stored and equal the pinned ones; all runs of BOTH tiers (and the bridge) share the one
sha in `TREE_SHA`, and every run stored `tree_clean = true` (compare asserts equality ACROSS tiers, not per tier: R2-m2); R1 and R2
logged in `logs/tier_red.log`; `optimize == 0`; `pc_cortex_tier == "stdlib"` and `encoder_calls.pc == 0` in every run. No result
file from a refused run exists.
Added in Revision 2 (R2-M4, R2-M5):
- **Encoder use**: production: `encoder_calls.vision > 0` and `.audio > 0` and `percept_ticks > 0` in every run; no-encoder: all
  three counts == 0. A production run with zero calls is VOID for that tier (a silent per-call fallback is caught here, where the
  attribute check cannot).
- **Tier effect present** (computed by `e1b_compare.py cross`; the claim "the tiers are two experiments"): for the same seed and
  arm, the held-out reward series (or the action series) of the production run differs from the no-encoder run's on >= 6 of 8
  `real` seeds. If not: the report states "tiers indistinguishable on this path", NO cross-tier claim and no "both tiers"
  wording is made, and every tier result is called one replicate.
- **Planted-learning control** (E1's, unchanged): t(frozen-learn) and t(shuf-learn) > 1.895 on the planted stream, seeds 0-7.
- **Planted-HARM control** (new, brain-free, in `e1b_replay.py`): a stream where learning hurts, per-pattern reward ALTERNATES
  sign on each occurrence of that pattern (600 ticks, two patterns, seeds 0-7); H-V1's statistic on it, t(MAE_learn - MAE_frozen),
  must exceed 1.895 (the harm direction is seen to go green on a known harm). If it does not on the instrument, the run lane
  strengthens the planted harm (e.g. shorten the alternation block) BEFORE any brain datum and discloses the change in
  addendum 1; H-V1 has no verdict until it is green.
- **Compare fixture** (brain-free, two JSON files generated by the run lane before the campaign, `fixtures/`): `pfrozen = real +
  0.1` on every seed -> H-P1 must print MET; `pfrozen = real` -> must print NOT MET; a third with a failed planted control -> must
  print VOID. Logged; compare refuses to print verdicts unless `logs/fixture.log` shows the three expected lines.

## 5. Power (stated in advance; hand arithmetic from E1's REPORT-TABLES per-seed values, one-sided .05, power .8, S = 8)

mde at S = 8 for a paired mean: (t_.95,7 + t_.80,7) * SD / sqrt(8) = (1.895 + 0.896)/2.828 * SD = 0.987 * SD.
Revision 2 constant: the pre-registered factor is **1.00** (rounded UP from 0.987, so "powered" is slightly harder to claim). The
red-team's guess that the exact noncentral-t factor is 1.05-1.1 (m3) was re-derived by hand with Abramowitz-Stegun 26.7.10 (df 7,
t_c = 1.895: needed noncentrality delta = 1.895*(1-1/28) + 0.8416*sqrt(1+1.895^2/14) = 1.827 + 0.943 = 2.770; factor
2.770/sqrt(8) = 0.979): the central-quantile sum is within 1% of the noncentral value, not 6-10% below it. Not run as code
(no Python in this lane); `e1b_compare.py` prints both the 1.00 rule and an exact bisection (noncentral t by numerical
integration, stdlib) as a check, and the 1.00 rule decides. Seed counts below use the same t constants throughout (the first
draft mixed z for seed counts and t for mde: R2 red-team re-derivation; with t, E-P2 needs ((1.895+0.896)*0.0695/0.0131)^2 = 219).
**Claimed effects are FROZEN LITERALS (policy M1), E1 no-encoder-tier numbers, never recomputed from E1b data:** `DH = 0.035`
reward units (harm; half of E1's observed 0.069: winner's-curse shrinkage), `DR = 0.0131` reward units (10% of E1's mean
held-out r 0.1311, E-P2/H-P2), `DM = 0.048` MAE units (10% of E1's mean replay MAE 0.477, all valence endpoints). The production
tier has NO SD basis from E1; the own-SD recompute rule below covers it; E1's numbers are used as the claimed effects for both
tiers by pre-registration, not because the production tier shares them. The line `DV = 0.1 * mean(...)` in the first harness
draft is deleted.

| endpoint | E1 SD of paired diff (no-encoder tier) | mde at S = 8 | claimed effect (frozen literal) | verdict on power |
|---|---|---|---|---|
| H-P1 (r_pfrozen - r_real) | 0.0695 | 0.069 | 0.035 (dh) | UNDERPOWERED for dh (power about .25); powered only for an effect as large as E1's (biased-up) 0.069; needs about 25 seeds (z) to 31 seeds (t) for dh |
| E-P2 vs pfrozen | 0.0695 | 0.069 | 0.0131 (10% of mean r 0.1311) | UNDERPOWERED (needs about 174 seeds) |
| E-P2 vs pshuf (H-P2) | 0.077 | 0.076 | 0.0131 | UNDERPOWERED |
| H-V1 / E-V2 (a vs b) | 0.204 (seed 1 outlier +0.54) | 0.20 | 0.048 (10% of mean MAE 0.477) | UNDERPOWERED (about 113 seeds; without seed 1 the SD is about 0.039 and 5 seeds would do: the outlier decides power) |
| H-V2 / E-V2 (a vs c) | 0.044 | 0.043 | 0.048 | marginally powered (mde just below the claimed effect) |
| E-V1 vfrozen / H-V3 | 0.153 | 0.15 | 0.048 | UNDERPOWERED |
| E-V1 vshuf | 0.181 | 0.18 | 0.048 | UNDERPOWERED |

Rule (pre-registered): when a primary endpoint is NOT met, `e1b_compare.py` recomputes the mde from E1b's OWN observed SD
and compares it with the claimed effect above. mde <= claimed effect -> the non-replication is real at that power; mde >
claimed effect -> INCONCLUSIVE, no deletion, no null. A MET endpoint stands whatever its power.
Plain statement: at S = 8 the harm endpoints can only be met by an effect near E1's observed size; a failure to meet them most
likely returns INCONCLUSIVE, not "no harm". That is the price of the owner's 8-seed request; section 6 says what follows.
Stated in advance (R2-m4): H-P1 can return "powered, no harm" only if E1b's observed SD of d is <= DH/1.00 = 0.035, half of E1's
0.0695; that outcome is expected to be unreachable, so H-P1 is expected to end MET or INCONCLUSIVE. The report says this
rather than treating a non-MET as evidence of absence.
Seed-count slot (OPEN with the owner, section 10): the numbers above are for S = 8. For S = 16 the constants are bar 1.753
(df 15, one-sided .05), mde factor (1.753+0.866)/4 = 0.655 -> pre-registered 0.66, "in the direction" >= 10 of 16, E-P1 "flips on
>= 12 of 16"; H-P1 mde becomes 0.66 * 0.0695 = 0.046 (still above DH = 0.035: still INCONCLUSIVE-prone), E-P2 vs pfrozen needs
~219 seeds regardless.

## 6. Dispositions (the lane REPORTS; the integrator and owner act; no code edit, no deletion in this lane)

Per tier T in {production, no-encoder}, then combined. If a tier's validity is VOID (any section 4 check, any VOID control): that
tier has NO verdict. A tier with no verdict makes every cross-tier statement **PENDING**, never "not met" and never a "tier
finding" (R2-m5). Tier results are never pooled. Wording rule for every not-met line below: "no harm on tree `<TREE_SHA>`,
seeds 8-15, tier T" with the bridge read-out (section 2) beside it; "E1's signal did not replicate" requires the bridge to read
"tree-equivalent"; "retired as a false positive" is not used in E1b (R2 red-team M3).

`_plan_w` (the update is LOCAL and scalar; no rule is added; section 9):
- Definition "**harm on both tiers**" = H-P1 MET on both tiers AND the cross-tier requirement of section 3 (sign of d_s agrees on
  >= 6 of 8 seeds) AND the tier-effect-present test of section 4 passed. Disposition: the learned `_plan_w` trajectory is a
  **harm-deletion candidate** (a measured harm past its pre-registered bar is deleted without a scale re-test, amendment item 4);
  the owner decides; the report states the false-replication rate as "between .0025 and .05" (two correlated tiers), and that
  the tree and seeds both differ from E1 unless the bridge reads tree-equivalent. (`_plan_w` is not an owner-drawn P4d part.)
- H-P1 MET on exactly one tier (other tier has a verdict and is not MET) -> **tier finding**, name the tier; no deletion
  candidate **unless the owner's answer to board #4616 is "a production-only harm suffices" (section 10, flag
  `SINGLE_TIER_HARM_DELETES`; default FALSE; if TRUE, a production-tier MET with a valid production tier alone is the candidate,
  the no-encoder result is reported beside it)**. The no-encoder tier is E1's tier; a production-only MET is a finding about the
  brain that runs.
- H-P1 not met on a tier, with a verdict: SD <= 0.035 (section 5) -> "powered, no harm on this tree/seeds/tier"; otherwise
  **INCONCLUSIVE** on harm for that tier; E1's signal stands as unreplicated. No deletion, no record.
- `_plan_w` objective (E-P2): reported; expected INCONCLUSIVE (section 5). The ONE NULL-RECORD for `_plan_w` is written ONLY when
  ALL hold on BOTH tiers (each with a verdict): E-P2 not met; observed mde <= DR = 0.0131 (frozen literal); H-P1 not MET (no
  measured harm on either tier: a method with a measured harm gets the harm route, never a null record). Then exactly one line,
  `NULL-RECORD: _plan_w | units=n/a | ticks=600 | seeds=8 | mde=<the LARGER of the two tiers' observed mde> | retest=PENDING
  since <full 40-hex sha of the arc's base on origin/main, predating this record>` (the gate's fields have no tier key; the
  tier split lives in the surrounding prose). The one 10x re-test is `RETEST-RECORD: _plan_w | axis=ticks | ticks=6000 |
  result=<null|charges> | of=<this file's repo path>` covering the `real` arms of BOTH tiers as ONE pre-registered re-test,
  same endpoint, bar and control, pre-registered separately. A tier-split result (null on one tier, charge or harm on the
  other) writes NO record (tier finding). No deletion before the re-test.

`ValenceLearner` (P3 disposition PROPOSAL for the owner, not an action):
- H-V1 MET on both tiers (both planted controls PASS, same cross-tier requirement as `_plan_w`) -> **harm on its replay objective,
  replicated**. It becomes a "harm-deletion candidate for the learning update" ONLY if, on each tier counted, the pre-registered
  in-brain check also holds: t(r_real - r_vfrozen) is NOT > 1.895 (removing the update does not cost reward; E-V3 shows 112-387
  of 600 actions flip when frozen, so the replay alone does not show consequence; policy m1). Otherwise the line reads
  "objective finding (replay harm, in-brain consequence present/unclear), owner decides". `valence`/`dopamine` as the RPE path
  are a separate owner question.
- H-V1 MET on one tier -> tier finding, no candidate (the `SINGLE_TIER_HARM_DELETES` flag of section 10 applies here too, with
  the same in-brain check).
- H-V1 not met (VOID is not "not met") and E-V2 not met, on BOTH tiers, each with a verdict: mde (observed SD of b - a) <= DM =
  0.048 (frozen literal) on both -> **objective null**: ONE `NULL-RECORD: ValenceLearner.learn | units=n/a | ticks=600 | seeds=8
  | mde=<larger of the two tiers> | retest=PENDING since <full 40-hex sha>`, then ONE `RETEST-RECORD ... axis=ticks | ticks=6000`
  covering both tiers, endpoint E-V2, bar and planted controls unchanged, `real` arm only (about 6 min per run, 8 seeds, per
  tier); NO deletion before it; the method leaves the live path dated `scale-retest`. mde > DM on either tier ->
  **INCONCLUSIVE**, no deletion, no NULL-RECORD (amendment item 4). Tier-split -> tier finding, no record.
- E-V2 MET on both tiers (and planted controls PASS) -> "learns its objective in replay"; with E-V1 also MET -> CHARGED.
- E-V3 flips are reported and change no disposition.

P2 note (policy m4): learning-rules section 1 lists the third-factor RPE job with TWO live mechanisms (`ValenceLearner`,
`_plan_w`). Whatever E1b returns, the adopt-or-merge question for that job goes to the owner afterwards; a non-harm outcome is NOT
read as "both stay".

Cross-tier rule: a result on ONE tier only (the other tier having a verdict) is a tier finding, whatever the endpoint; the other
tier having no verdict is PENDING. Tier results are never pooled. If "tier effect present" fails (section 4) the tiers are
replicates and every cross-tier wording is dropped.

## 7. Resources and order (`resource:timing`)

- Claim `resource:timing` (board, `--as <run lane>`) only when (i) the board shows it free, (ii) the previous holder's PROCESS
  is gone (a lease expiry is not a release: E1 addendum 5; check the holder's named battery/daemon-side process on the box,
  `tasklist`/`Get-CimInstance Win32_Process`, and the integrator's "battery ended" post), (iii) free RAM >= 2 GB (E1's floor).
  Post the hold status with a projected release time; renew every 40 min; release right after the last run of a stage if a
  battery asks; yield to any battery at a RUN boundary (a run in flight finishes and is kept).
- Order (stage-major, one brain at a time): A = `real` + `pfrozen` seeds 8-15, production then no-encoder (32 runs; enough for
  H-P1 and E-P1/E-P2-vs-pfrozen on both tiers); B = `vfrozen` (16; H-V3, E-V1, E-V3); C = `vshuf` + `pshuf` (32; E-V1, E-P2,
  H-P2); D = controls (repeat, pc x2, per tier: 6) and the replay (brain-free, H-V1/H-V2/E-V2); E = the BRIDGE (`real` +
  `pfrozen` seeds 0-7, no-encoder, 16 runs; last, because it only changes wording; if it is not reached, every non-replication
  line reads "bridge not run: tree and seeds both differ from E1, no statement about E1's signal"). H-V1 needs only stage A's
  `real` series. The brain-free pre-campaign checks (R1/R2 reds, compare fixture, planted-harm control) run first and are
  logged; they need no `resource:timing`. A partial campaign is evaluated for the endpoints whose pairs are complete; an endpoint with < 8 complete
  seed pairs has NO verdict.
- Skip rule: a run whose log already contains `DONE` is skipped on resume; a repeat-arm equality check voids resumed
  campaigns that are not deterministic-equivalent (E1 addendum 3 precedent).

## 8. Out of scope / guards

No change to the brain, `_plan_w`, `ValenceLearner` or any file under the checkout or either worktree is made by or proposed to
the run lane; a harm disposition is the owner's act on a separate arc. The live lake is never written (`NORD_HOME` under the
work folder `home/`, asserted by the probe). `NORD_PERSIST`, `NORD_SELFMOD`, `NORD_ESTOP_SELF_CLEAR` unset. The brainstem
envelope is not touched. Every brain-outcome number in the report names its tier. No Codex.

## 9. Dispositions of graph-exists hits (COMMON.md rule)

`node graph-exists.js "assert perception tier of brain equals requested tier and refuse run" --root p8-restack-1023` -> NO
LEXICAL MATCH (UNKNOWN, not absence). Disposition: the tier check is lane-scratch harness code (`tier_verdict`, `preflight`, about
25 lines) in the run folder, not a def under `src/`; the brain's own tier fields (`_perception_tier`, `_perceiving_real`,
`_perceiving_audio`) are REUSED, nothing is reimplemented. No new module, class or def under `src/` is proposed. Ponytail ladder:
stdlib `hashlib`/`os`; no dependency; E1's scripts patched, not rewritten. Revision 2 adds only lane-scratch harness code
(counters, fixtures, planted-harm stream), same disposition. **P5b: Python.** Criterion 1 (hot, measured) fails: this is
lane-scratch measurement code, no neuter of it moves a tick by >= 5% (it adds a few method-call counters); criterion 4's
precondition (a ported kernel) does not exist; no Rust is proposed and the first failing criterion is enough to stay in Python.
Locality (policy m3): the `_plan_w` update (`autonomous_loop.py` L1505-1506, `w += lr * da * (elig - base)`) is a LOCAL scalar
three-factor rule (no transport, cosine test not owed); `ValenceLearner.learn` is LOCAL (`valence_stdp.py` GRADIENT_CLAIM). The
census row `AutonomousArm._pfc_bias` = `NOT_A_RULE` (autonomous_loop.py L70-73, "a read of executive state") labels the function
that CONTAINS the `_plan_w` write and so mislabels the rule: flagged to the integrator in `OPEN-QUESTIONS.md` (E1b edits nothing; the
claim is self-declared and caught by review only, POLICIES e-prop clause).

## 10. Open owner questions as parameters (board #4615, #4616): the exact lines that change

Both answers slot in by editing constants in one place; no section is rewritten.
- **Seeds per tier (#4615): 8 vs 16.** Constants: `SEEDS` (`range(8,16)` -> `range(8,24)`), `K_MIN` (5 -> 10), `BAR` (1.895 ->
  1.753), `MDE_F` (1.00 -> 0.66), E-P1 threshold (6 -> 12), cross-tier agreement (6 -> 12), tier-effect threshold (6 -> 12), `PSEEDS`
  unchanged, `/8` text in compare derived from `len(SEEDS)`. Runs: 102 -> 188 (about 2x). Section 5's table is recomputed by
  compare from the same formula. The two tiers still share seeds (correlation stays: section 3 rule stays). A 16-seed
  design where each tier gets its OWN 8-seed block (production 8-15, no-encoder 16-23) is a third option: it makes the tiers
  independent (the ".0025" claim would then be valid) but removes the paired-tier-effect test and the cross-tier agreement rule;
  it is NOT the default because the owner said "BOTH perception tiers" on seeds 8-15.
- **Does a production-only harm suffice to delete (#4616)?** Constant `SINGLE_TIER_HARM_DELETES` (default FALSE) in section 6
  and in `e1b_compare.py cross`. FALSE: production-only MET = tier finding. TRUE: a valid production tier alone with H-P1 MET
  (and for ValenceLearner, the in-brain check) is the deletion candidate, with the no-encoder result and the correlated
  false-positive range reported beside it. If the owner answers "production only is THE tier", the no-encoder campaign becomes
  secondary and stages order production-first is already how stage A runs.

---

## Addendum 1 (2026-10-23, lane e1b-harness, before any E1b brain run)

Append-only; no bar, constant or rule above is changed. Nothing here is a datum from an E1b campaign.

- **Where the harness lives.** The jobs the body assigns to run-lane scratch scripts are subcommands of one governed module,
  `src/snn_data_lake/instruments/rpe_charge.py` (lane brief E1B-HARNESS). Section 9's sentence "No new module, class or def under
  `src/` is proposed" described lane-scratch code and is superseded for this module. Its graph-exists dispositions are in the
  module docstring (nothing existing hosts the arms; `tools/charge_probe.py` and `instruments/port_charge_probe.py` are
  distinct-from). P5b: Python, unchanged.
- **Port choices** (module docstring, departures 1-6): a refusal raises `Refused` (a SystemExit, code 3) instead of `os._exit(3)`;
  the `pc` arm stubs `PrefrontalExecutive._decide_spiking` class-wide as E1 did; with `SINGLE_TIER_HARM_DELETES = True` the cross-tier
  verdict still needs both tiers to have a verdict; compare fixtures are built in memory, not written as `fixtures/` files; a
  missing or wrong `tier_red.log` / `fixture.log` makes the tier VOID (harness validity) rather than refusing to print;
  campaign homes and output dirs must sit under an existing directory that does not contain `~/.snn_data_lake`.
- **Additions the body does not list.** A partial campaign marks missing arms, the repeat arm, pc runs and the replay record as
  PENDING (an endpoint with fewer than all seed pairs prints NO VERDICT, as section 7 says) instead of VOID; runs must have
  `ticks == 600`; `cross` is PENDING while either tier has a PENDING validity item; the bridge's runs carry the same validity
  checks and must share the campaign's one tree sha.
- **Controls measured brain-free, before any datum** (module self-check and `tests/test_rpe_charge.py`): planted-learning control
  t(frozen-learn) = 13447.7, t(shuf-learn) = 39.5 (equal to E1's recorded values); planted-HARM control t(learn-frozen) = 4075.2
  (> 1.895, so the stream is not strengthened); planted null t = 0; exact mde factor 0.978 against the pre-registered 1.00.
- **Not done here.** The anchors of the header (`_rpe`, `_plan_lr`, `_pfc_bias`, `ValenceLearner` line numbers) are re-confirmed by
  the run lane on the tree it uses; the R1/R2 reds are run by the run lane (the module's preflight and `tier_verdict` reds are
  covered by the tests); no campaign run was made.

## Addendum 2 (2026-10-23, lane e1b-seq, before any E1b brain run): the design becomes SEQUENTIAL (P1)

Append-only; written before any E1b brain run and before any E1b datum. Owner, 2026-10-23 (verbatim, after asking "why a
constant number of seeds?"): "switch E1b to a sequential design, P1 policy, make sure all sessions understand that" (board
#4876). This addendum replaces the FIXED 8-seed decision of the two PRIMARY endpoints (H-P1, H-V1; sections 3, 5, 6) with a
group-sequential decision. Everything else above stands: the arms, N = 600, H, the patch code, the tier pins, every validity
check of section 4, the planted controls, the claimed effects (frozen literals DH = 0.035, DM = 0.048), the dispositions'
vocabulary and the P8 null-scale rules. Every number below is re-derived by `lanes/engine-1024/e1b-seq-row/derive_seq.py`
(brain-free; E1's recorded per-seed values only; output in that folder's `logs/derive-200k.log`).

### Sample size (P1)

Measured SDs (E1 REPORT-TABLES, no-encoder tier, seeds 0-7, the SD of the paired per-seed difference; Revision 2's outlier
handling: the degenerate seed STAYS in, so E1 seed 1 is in both SDs):
- H-P1, d_s = r_pfrozen - r_real: d = -0.0069, 0.0000, +0.1564, +0.0738, +0.1802, +0.0390, +0.0869, +0.0244; SD = **0.0695**.
- H-V1, d_s = MAE_learn - MAE_frozen: d = +0.0405, -0.5355, +0.0378, +0.0083, +0.0591, -0.0441, +0.0709, +0.0537; SD = **0.2040**
  (0.0391 without seed 1, reported only: the outlier decides power, as section 5 said).
- The production tier has NO SD basis (section 5); E1's SDs are used for BOTH tiers by pre-registration. The observed SD
  of each tier enters only the `powered` test at a stop (below), never the boundaries.

| endpoint (both tiers) | SD | claimed effect | d_z | fixed-sample target (power .8, one-sided .05, t) | hard cap | sequential power at the cap | P(harm by look 2) |
|---|---|---|---|---|---|---|---|
| H-P1 | 0.0695 | DH = 0.035 | 0.504 | 26 seeds | **32** (26 rounded up to a block; the sequential design keeps power .816 >= .8 there) | **0.816** | 0.229 |
| H-V1 | 0.2040 | DM = 0.048 | 0.235 | 114 seeds (120 at a block) | **32** (budget, below) | **0.303** | 0.045 |

Alpha: one-sided .05 PER primary endpoint per tier (Revision 2's multiplicity rule, section 3: no correction; the cross-tier
".0025" stays withdrawn). Target formula: the smallest S with (t_.95,S-1 + t_.80,S-1) SD / delta <= sqrt(S).

**Budget and the H-V1 cap.** The hard cap is 32 seeds per tier (looks 8, 16, 24, 32) for BOTH primaries: it is H-P1's
power-derived target rounded up to a block. H-V1's target (114, i.e. 120 at a block) exceeds it: reaching 120 would add 88 `real`
runs per tier beyond the cap (about 60 min per tier at E1's no-encoder per-run time, production unknown) on a campaign whose
production per-run cost is not yet measured. This addendum does NOT pre-register that extension. **H-V1 is pre-declared
INCONCLUSIVE-prone at the cap**: power .303 for DM; with E1's SD its mde at the cap is 0.095 > DM, so a stop without a crossing
is INCONCLUSIVE unless the tier's observed SD is <= DM / 0.4650 = 0.103 (half of E1's 0.204; H-P1's threshold is
DH / 0.4650 = 0.075, above E1's 0.0695).
Extending H-V1 alone to 120 seeds is an owner question (row folder `OPEN-QUESTIONS.md`); if the owner says yes it is a separately
pre-registered extension, never a post-hoc one.

### Looks, boundaries and the decision

- **Looks** (seed blocks of 8, both tiers on the SAME seeds, synchronized): look 1 = seeds 8-15 (the owner's block, the full
  campaign of sections 2 and 7), look 2 = 16-23, look 3 = 24-31, look 4 = 32-39 (the cap: 32 seeds). Looks 2-4 add ONLY the
  `real` and `pfrozen` arms (all H-P1 and H-V1 need) plus the brain-free replay of the new `real` series; the secondary
  endpoints, the repeat arm, the `pc` control and the bridge are evaluated at look 1 only, by the frozen text, unchanged.
- **Alpha spending**: Lan-DeMets O'Brien-Fleming type, one-sided: a(tau) = 2 - 2 Phi(z_{0.975} / sqrt(tau)), tau = n / 32.
  Boundaries solved on the z scale by recursive numerical integration of the B-value path (Armitage-McPherson-Rowe), with no
  futility bound in the integration (futility is NON-BINDING).
- **df handling**: each look's t boundary keeps that look's nominal one-sided p at df = n - 1: t_k = t_{1 - p_k, n_k - 1}.

| look | seeds | n | cumulative alpha spent | z boundary | nominal p | **t boundary** (df) | sign count needed |
|---|---|---|---|---|---|---|---|
| 1 | 8-15 | 8 | 0.000089 | 3.7496 | 0.000089 | **7.2020** (7) | 5 |
| 2 | 8-23 | 16 | 0.005575 | 2.5399 | 0.005544 | **2.8959** (15) | 10 |
| 3 | 8-31 | 24 | 0.023625 | 2.0161 | 0.021897 | **2.1333** (23) | 15 |
| 4 (cap) | 8-39 | 32 | 0.050000 | 1.7202 | 0.042702 | **1.7769** (31) | 20 |

- **Harm boundary at look k**: t(d) over the n_k seeds > t_k AND #{d_s > 0} >= ceil(5 n_k / 8) (section 3's "5 of 8", scaled;
  a tie is not in the direction). This replaces "t > 1.895 and >= 5 of 8" for H-P1 and H-V1. At look 1 the old bar 1.895 is
  printed beside it as a reported statistic with no disposition.
- **Futility (non-binding)**, looks 1-3: stop when the conditional power under the frozen claimed effect is < 0.10:
  CP = 1 - Phi((z_cap - t sqrt(tau) - theta (1 - tau)) / sqrt(1 - tau)), theta = delta sqrt(32) / SD_obs, the observed t in place
  of z, tau = n / 32. Non-binding: the boundaries above ignore it, so a futility stop never inflates the type-I rate.
- **Powered** (decides what a non-harm stop means): mde at the cap with the tier's observed SD,
  (t_cap + t_.80,31) SD_obs / sqrt(32) = (1.7769 + 0.8534) / 5.6569 SD_obs = 0.4650 SD_obs, <= the claimed effect. (Replaces the 1.00 * SD rule for the primaries.)
- **Stop rule**: decide ONLY at a look, ONLY on a written boundary (harm, futility, or the cap). A decision at any other seed
  count is refused by the harness (exit 3). A primary endpoint's FIRST stop on a tier is its decision of record; later looks
  re-print it but never re-decide it. The campaign runs the next block while any primary endpoint on either tier is CONTINUE
  and the cap is not reached; both tiers always run the same block (synchronized looks), so a tier that has stopped keeps
  adding the shared seeds (reported, not re-decided).

Simulated operating characteristics (200 000 campaigns per row, d_s ~ N(mu, SD), the harness's own decision function):

| endpoint | truth | harm (type I or power) | futility stop | reaches the cap | expected seeds |
|---|---|---|---|---|---|
| H-P1 | zero effect, futility ignored | 0.0438 | n/a | | |
| H-P1 | zero effect | 0.0435 | 0.658 | 0.299 | 24.9 |
| H-P1 | DH = 0.035 | 0.816 (look 1 0.002, 2 0.226, 3 0.396, 4 0.190) | 0.022 | 0.162 | 25.0 |
| H-V1 | zero effect, futility ignored | 0.0424 | n/a | | |
| H-V1 | zero effect | 0.0423 | 0.812 | 0.146 | 18.9 |
| H-V1 | DM = 0.048 | 0.303 (look 1 0.0004, 2 0.045, 3 0.134, 4 0.124) | 0.418 | 0.279 | 24.3 |

### What each stop means for the Revision 2 dispositions (section 6; P8 null-scale rules unchanged)

- **Stopped for harm** (boundary crossed at look k) = "H-P1 / H-V1 MET" in section 6, with every condition section 6 adds
  (validity, controls PASS, the cross-tier requirement, the tier effect, the in-brain check for ValenceLearner). Wording names
  the look: "MET at look k (n seeds)". A measured harm past its pre-registered boundary is the harm route (amendment item 4).
- **Stopped for futility** or **reached the cap without crossing** = "not met" in section 6: an objective null at that
  endpoint ONLY if `powered` (above) holds on the tier; otherwise **INCONCLUSIVE**: no deletion, no NULL-RECORD. A NULL-RECORD's
  `seeds=` field is the n at the stop. No NULL-RECORD while either tier's primary is CONTINUE.
- **CONTINUE** is no verdict: it is never "not met", never a tier finding, never a null.
- Cross-tier (section 3): "harm on both tiers" needs both tiers' decisions of record to be harm, and the sign of d_s to agree on
  >= ceil(6 n / 8) seeds at the later of the two stop looks (6, 12, 18, 24); the tier-effect test scales the same way
  (>= ceil(6 n / 8) `real` seeds differ). The false-replication wording "between .0025 and .05" stands.
- `_plan_w` objective E-P2 and every secondary endpoint: look 1 only (seeds 8-15), the frozen text; the `_plan_w` NULL-RECORD of
  section 6 additionally needs H-P1 stopped without harm and `powered` on both tiers.

### Run time per look (no brain was run for this estimate)

Addendum 1 measured no per-run time (it measured brain-free controls only). The only measured per-run time is E1's: about
34-38 s per 600-tick no-encoder run plus about 3 s construction (E1 REPORT, "Wall time and RAM"; 1435 s over 42 runs); about
41 s per run is used. Look 1 = the frozen campaign: 86 runs + the 16-run bridge, about 70 min no-encoder-equivalent. Looks 2, 3
and 4 each add 16 runs per tier (`real` + `pfrozen` x 8 seeds) = 32 runs, about 22 min per look at the no-encoder rate; the
production tier's per-run time is measured at look 1 (stage A) and each later look is then re-estimated from it before its
`resource:timing` claim. Worst case to the cap: 102 + 96 = 198 runs, about 2 h 15 min no-encoder-equivalent.

### Harness binding

`src/snn_data_lake/instruments/rpe_charge.py`: `LOOKS`, `SEQ_ALPHA`, `FUTILITY_CP`; `seq_boundaries()` (the table above) and
`seq_decision(ds, claimed)` (one look's decision; refuses a non-look seed count); `analyse` decides H-P1 and H-V1 with it and
refuses a non-look seed count; `null_precondition` refuses while a primary is CONTINUE; `cross` scales the agreement and
tier-effect counts with the look; `replay` / `compare` / `cross` take `--look K`. Reds R-S1..R-S5 in `tests/test_rpe_charge.py`
(fast, 4000 simulated campaigns; R-S1/R-S2 slow twin 40 000): R-S1 sequential type I <= .05 + 3 MC SE; R-S2 naive testing at
.05 at every look on the same null series exceeds .05 + 3 SE (measured 0.099 at 4000, 0.108 at 40 000); R-S3 a planted harm of
the claimed size crosses by look 2 and by the cap at the rates above (within 3 SE); R-S4 a planted zero effect stops for
futility at the rates above; R-S5 a decision at 7, 12, 20 or 33 seeds is refused. graph-exists ("alpha spending", "group
sequential", "interim analysis": NO LEXICAL MATCH; "power analysis sample size": three keyword hits, read, distinct): nothing in
`src/` spends alpha or decides at interim looks; `instruments/paired_contrast.py` (fixed-S paired t) is distinct-from;
`instruments/pool_selection.t_quantile` is REUSED. P5b: Python (criterion 1 fails: measurement code off the tick).

## Addendum 3 (2026-10-24, lane e1b-seq2, before any E1b brain run): the production tier decides (owner ruling #4616)

Append-only; written before any E1b brain run and before any E1b datum. Addendum 2 and every frozen section above are
untouched; where this addendum and an earlier text differ, this addendum rules. Owner, 2026-10-23 (board owner-answer #4906 to
#4616, verbatim): "#4616: production tier decides". Integrator's binding reading (lane brief E1B-SEQ, section OWNER RULING):
`SINGLE_TIER_HARM_DELETES = True`.

### What the ruling changes

- **The production tier alone decides** the deletion route of each primary endpoint (H-P1 `_plan_w`; H-V1 ValenceLearner). It
  is THE confirmatory tier: one-sided alpha .05 per endpoint, power target .8, the looks and the Lan-DeMets O'Brien-Fleming
  boundaries of addendum 2, unchanged.
- **The no-encoder tier** runs on the same synchronized looks and the same seeds and is reported beside the production result.
  It never deletes on its own: a no-encoder-only crossing is a **tier finding**, no deletion candidate (owner decides).
- **Revision 2's cross-tier agreement rule** (section 3: sign of d_s agrees on >= 6 of 8 seeds, scaled by addendum 2 to
  ceil(6 n / 8)) and the tier-effect test (section 4) **no longer gate** a deletion candidate. Both are still computed and
  printed beside the production verdict ("d_s sign agrees on k/n", "tiers indistinguishable on this path"). The wording
  "harm on both tiers" and its "between .0025 and .05" false-replication clause leave the deletion route, and so does section
  10's "correlated false-positive range reported beside it" for TRUE (one confirmatory tier at one-sided .05 has no
  cross-tier range); the route's wording is "harm on the production tier (MET at look k (n seeds); the confirmatory tier;
  <no-encoder beside>)", naming the look of the decision of record as addendum 2 asks.
- **Multiplicity**: Revision 2's rule across ENDPOINTS stays (two primaries, each one-sided .05, no correction). There is no
  tier multiplicity: only one tier is confirmatory, so no alpha is split across tiers.
- **Seed target unchanged.** Addendum 2 already sized the target per tier (one tier's paired t, one-sided .05, power .8):
  H-P1 26 seeds, hard cap **32 per tier** (looks 8, 16, 24, 32), sequential power .816 at the cap; H-V1 114 seeds (120 at a block),
  capped at 32, power .303. Those numbers are the production tier's design; no joint both-tiers crossing is sized for.
- **Validity**: a VOID production tier (wrong perception tier, a failed section-4 check) is REFUSED for the deletion route: no
  candidate, no tier finding, no record. This REFUSED is a verdict line in the cross report, not the harness's exit-3 refusal. A VOID no-encoder tier keeps the verdict PENDING (the stricter reading of "a tier with no
  verdict makes every cross-tier statement PENDING"; harness departure 3).
- **Stop rule** (replaces addendum 2's "while any primary endpoint on either tier is CONTINUE"): the campaign runs the next block
  while a primary endpoint on the PRODUCTION tier is CONTINUE and the cap is not reached. The no-encoder tier runs the same block;
  its decision of record (below, the same rule on both tiers) is re-printed beside, never re-decided; its CONTINUE never
  extends the campaign and never blocks a
  NULL-RECORD (this replaces addendum 2's "No NULL-RECORD while either tier's primary is CONTINUE": the production tier's
  primary must have stopped).
- **Decision of record** (addendum 2's rule, now carried by the harness): `analyse` walks the look prefixes of the seed-ordered
  per-seed differences and keeps the FIRST stop (harm, futility or the cap) as the decision of record, with its look, n and mde;
  a later look never re-decides it. The validity checks and controls are those of the look being analysed (they only grow
  stricter as seeds are added).

### The SDs, and what decides powered

The SDs of addendum 2 (H-P1 0.0695, H-V1 0.2040) are E1's **NO-ENCODER** values: no production-tier SD exists. The production
tier's SD is first seen at look 1. The `powered` rule then decides between a null and INCONCLUSIVE with the tier's OBSERVED SD;
at the cap that is addendum 2's rule: powered iff observed SD <= **0.075** (H-P1, DH / 0.4650) / **0.103** (H-V1, DM / 0.4650).

### A futility stop, or the cap without a crossing, is not a null by itself (arc panel B-M2, m6)

The decision writes the **mde at the stopped seed count n**: mde = (t_k + t_.80,n-1) SD_obs / sqrt(n), where t_k is the look's
t boundary (the test actually applied at that look). P8 null-scale amendment item 4 then decides:
- mde <= the claimed effect (powered) -> the **NULL-RECORD path**: one NULL-RECORD, then the ONE 10x re-test before any
  deletion (amendment item 2). The record's fields (replacing section 6's `seeds=8` and "the LARGER of the two tiers' observed
  mde"): `seeds=<n at the production stop>` and `mde=<the production primary's mde at that n>` (H-P1 against DH = 0.035, H-V1
  against DM = 0.048). That mde uses the look's O'Brien-Fleming t boundary, not the nominal one-sided .05 t of P8 item 1, so it
  is the larger, conservative value. The objective endpoint's mde (E-P2 against DR, E-V2 against DM, look 1, both tiers) stays a
  precondition and is reported in the record's surrounding prose;
- mde > the claimed effect -> **INCONCLUSIVE when underpowered**: no NULL-RECORD, no deletion at any scale.
A futility stop is never by itself the powered null that amendment item 2's re-test asks for. Addendum 2's word
"INCONCLUSIVE-prone" is replaced by "INCONCLUSIVE when underpowered" (panel m6).

Powered-SD bound per look (the observed SD at or below which a stop there is powered; `lanes/engine-1024/e1b-seq-row/
derive_seq2.py`, output `logs2/derive-2.log`, from the harness's own `seq_boundaries`):

| look | n | t_k | t_.80,n-1 | mde factor | H-P1 SD bound (DH 0.035) | H-V1 SD bound (DM 0.048) | mde at E1's SD (H-P1 / H-V1) |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 7.2020 | 0.8960 | 2.8631 | 0.0122 | 0.0168 | 0.1990 / 0.5841 |
| 2 | 16 | 2.8959 | 0.8662 | 0.9405 | 0.0372 | 0.0510 | 0.0654 / 0.1919 |
| 3 | 24 | 2.1333 | 0.8575 | 0.6105 | 0.0573 | 0.0786 | 0.0424 / 0.1245 |
| 4 (cap) | 32 | 1.7769 | 0.8534 | 0.4650 | 0.0753 | 0.1032 | 0.0323 / 0.0949 |

Consequence, stated before data: with an SD near E1's, an H-P1 futility stop at looks 1-3 reads INCONCLUSIVE (only a cap stop
can be the powered null), and every H-V1 non-harm stop reads INCONCLUSIVE. Whether an underpowered futility stop should instead
continue (futility is non-binding, so continuing never inflates type I) is an owner question (row folder `OPEN-QUESTIONS.md`,
Q3); until answered, addendum 2's futility stop stands and reads INCONCLUSIVE.

### NULL-RECORD precondition (replaces addendum 2's "on both tiers" clause for the primaries)

A NULL-RECORD for `_plan_w` / `ValenceLearner.learn` needs: the PRODUCTION tier's primary stopped (futility or cap) and powered
at its stopped n; and, as section 6 states, on BOTH tiers (valid, not PENDING; a no-encoder CONTINUE on the primary does not
block): the objective endpoint (E-P2 / E-V2) not met with mde
<= its frozen literal, no measured harm (a no-encoder H-P1/H-V1 MET is a tier finding and writes no record), controls PASS, one
tree sha. Requiring the no-encoder tier clean only ever withholds a record; it never adds a deletion.

### Harness binding

`rpe_charge.py`: `SINGLE_TIER_HARM_DELETES = True`; `cross` / `render_cross` carry "What the ruling changes" and the
validity rule (for H-V1 a production MET is a deletion candidate only when section 6's in-brain check holds, else an
objective finding); `seq_record` the decision of record; `seq_decision` the mde at the stopped n; `null_precondition` the
NULL-RECORD precondition; the in-brain check reads the look-1 block's `vfrozen` pairs at every look, since looks 2-4 add no
`vfrozen` runs (addendum 2). Revision 2's both-tiers rule stays in the code as the `SINGLE_TIER_HARM_DELETES = False` branch
(the tests pin it). Test record: `lanes/engine-1024/e1b-seq-row/REPORT-2.md`.

## Addendum 4 (2026-10-24, lane e1b-fix, before any E1b brain run): H-V1's cap is raised to its power target (owner ruling)

Append-only; written before any E1b brain run and before any E1b datum. Addenda 2 and 3 and every frozen section above are
untouched; where this addendum and an earlier text differ, this addendum rules. Owner, 2026-10-24 (verbatim): "Raise its cap
to its power target". The integrator's binding reading (lane brief E1B-FIX, OWNER RULING): option (a) is decided. Option (b),
keeping 32 as a disclosed P1 departure, was not chosen. Both texts are in `lanes/engine-1024/e1b-fix-row/OPTIONS.md`. This
addendum also folds the round-2 re-panel findings A-B1, A-B2, A-M2, B-N2, B-N3, B-N4, B-N7 and C R2-M1.

### Sample size (P1)

**H-V1 (ValenceLearner, valence MAE, claimed effect DM = 0.048, one-sided alpha .05, power target .8).**

- **Pre-registered SD**: 0.2040, E1's NO-ENCODER paired SD (seed 1 kept). No production-tier SD exists, so the production
  tier's design power is borrowed from the no-encoder SD (re-panel B-N3). The production tier decides (#4616). The
  OBSERVED SD at the stopped look decides `powered` (addendum 3).
- **Why the plain design fails (R-S1b, re-panel A-B2)**: E1's eight per-seed d_s hold one degenerate seed (-0.5355). The
  planted-SHAPE null resamples the centred d_s to the cap. Under that null, addendum 2's normal-model t boundaries cross at
  **0.2185** (rv-arc1-a round 2, `typeI_shape.out`), four times the nominal .05. A shape-blind cap would be unsafe.
- **The statistic**: the paired t and the sign gate ceil(5 n / 8), with each look's t boundary set by the shape. The boundary
  is the value that keeps Lan-DeMets O'Brien-Fleming cumulative spending under the planted-SHAPE null. It is never below
  the normal-model boundary. It was calibrated on 200 000 resampled campaigns (`derive_v1.py calib`, RNG seed 21) and frozen
  here before data.
- **Fixed-sample power target under the shape**: **134 seeds**. That is the smallest n where the single-look test at the
  shape's own .05 critical value (1.9236) has power >= .8 at DM (0.8017; `derive_v1.py target`, 200 000 campaigns per n).
  The normal-model target was 114.
- **Inflation factor**: the LD-OBF power inflation for K equal looks, normal model, one-sided .05 and power .8 (400 000 paths,
  `derive_v1.py if K`): K = 17: 1.0632; K = 18: **1.0624**; K = 19: 1.0628. 134 x 1.0624 = 142.4, rounded up to a whole
  8-seed block = 144 (18 looks, a fixed point).
- **Power check, then one block more**: at the 144 cap with its calibrated boundaries, simulated power under the shape plus
  DM = **0.7935**, below .8 (`derive_v1.py power`, 200 000 campaigns, seeds 31/32, independent of the calibration seed). So the
  cap goes up by one block. **Hard cap = 152 seeds per tier, 19 looks** (8, 16, ..., 152). Power there = **0.8155** (numpy
  model). The harness (`rpe_charge.seq_decision`) gives 0.8164 at 40 000 campaigns (seed 8080). Cap 160 would give 0.8364;
  it is not taken.
- **Type I under the REAL shape** (planted-shape null, resampled centred E1 d_s): **0.0502** (numpy, 200 000) and **0.0510**
  (harness, 40 000; seed 7070; 3 MC SE = 0.0033). Under the normal null the shape boundaries are at least the normal ones, so
  type I <= .05 (test R-S1).
- **Futility**: non-binding conditional power < .10. It is taken only at a look whose mde from the PRE-REGISTERED SD,
  (t_k + t_.80,n-1) x 0.2040 / sqrt(n), is <= DM (re-panel B-N2). For H-V1 no look before the cap qualifies: the mde_pre at
  n = 136 / 144 / 152 is 0.0556 / 0.0527 / 0.0499. So H-V1 has **no futility stop**: P(futility) = 0. The campaign runs to a
  crossing or to the cap.
- **P(INCONCLUSIVE)** (a non-harm stop whose mde at the observed SD exceeds DM): **0.3649** under the planted-shape null and
  **0.1836** at DM (harness, 40 000). A cap stop that does not cross is **NOT a null**. It is INCONCLUSIVE unless powered at
  its observed SD (P8 item 4).
- **Harm crossing by look 2**: 0.0004 at DM. The early boundaries are near-unreachable by design. Before look 7 the spending is
  < .0013. Before look 4 the shape boundary is the largest t among the calibration campaigns, so no calibration campaign
  crosses there.

The H-V1 boundaries (t; frozen in `rpe_charge.V1_SHAPE_T`). The normal-model boundary is shown beside for comparison. At
n = 8 the normal value is t_quantile's 64.0 bisection ceiling, which is moot because the shape boundary rules there.

| look | n | shape t | normal t | look | n | shape t | normal t |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 152.4982 | 64.0 | 11 | 88 | 3.4068 | 2.4758 |
| 2 | 16 | 42.7365 | 12.3056 | 12 | 96 | 3.1276 | 2.3605 |
| 3 | 24 | 39.7059 | 6.3858 | 13 | 104 | 2.8925 | 2.2603 |
| 4 | 32 | 27.9062 | 4.7985 | 14 | 112 | 2.7317 | 2.1722 |
| 5 | 40 | 19.4013 | 4.0261 | 15 | 120 | 2.5690 | 2.0939 |
| 6 | 48 | 15.4001 | 3.5500 | 16 | 128 | 2.4429 | 2.0236 |
| 7 | 56 | 6.0660 | 3.2179 | 17 | 136 | 2.3370 | 1.9600 |
| 8 | 64 | 5.5080 | 2.9684 | 18 | 144 | 2.2539 | 1.9022 |
| 9 | 72 | 4.4073 | 2.7715 | 19 (cap) | 152 | 2.1730 | 1.8493 |
| 10 | 80 | 3.7690 | 2.6106 | | | | |

**H-P1 (`_plan_w`, harm, DH = 0.035): the cap stays at 32.** R-S1b on E1's centred H-P1 d_s gives a type I of **0.0356**
(harness, 40 000; re-panel A's probe: 0.0352), below .05. R-S1b therefore does not move it. H-P1 keeps addendum 2's design:
looks 8/16/24/32, the normal-model boundaries, and power .8155 at DH under the pre-registered SD 0.0695 (harness, 40 000).
Harm crossing by look 2 = 0.2326; P(INCONCLUSIVE) under the normal null = 0.2293. The futility rule above closes every H-P1
futility look: mde_pre = 0.1990 / 0.0654 / 0.0424 > 0.035 at looks 1-3. So H-P1 also runs to a crossing or to the cap.
Disclosed, not acted on: under E1's resampled SHAPE, H-P1's power at DH is 0.6544 (harness, 40 000). P1 item 2's power
target is stated on the pre-registered SD model. Whether H-P1's cap should also follow its shape power is an owner question
(row folder `OPEN-QUESTIONS.md`, Q2). Until it is answered, the brief's rule stands: "H-P1's cap stays unless R-S1b moves it".

Addendum 3's open question Q3 (should an underpowered futility stop continue?) is settled by B-N2: **it continues**. Futility
is non-binding, so continuing costs no type I.

### Runs per look and hours (production tier decides, #4616; the no-encoder tier runs beside on the same looks)

- Look 1 is the frozen campaign: 86 runs plus the 16-run bridge.
- Looks 2-4 each add `real` + `pfrozen` x 8 seeds = 16 runs per tier, 32 runs per look (addendum 2).
- **Looks 5-19 add `real` only** (H-P1 stopped at its cap 32; the secondaries and dispositions read the look-1 block):
  8 runs per tier, 16 runs per look, about 11 min per look at E1's 41 s no-encoder rate.
- Worst case to the H-V1 cap: 102 + 96 + 15 x 16 = **438 runs, about 5 h 0 min no-encoder-equivalent** (41 s per run, E1).
- The production tier's per-run time is measured at look 1 (stage A). Each later look is re-estimated from it before its
  `resource:timing` claim. The stop rule is addendum 3's: the next block runs while a production primary is CONTINUE and its
  cap is not reached.

### Other changes (re-panel round 2)

- **A-B1, each endpoint from its pre-registered seed block**: every E-* endpoint (E-P1, E-P2, E-V1, E-V2, E-V3), the in-brain
  check and the dispositions are computed on the look-1 block (seeds 8-15) at every look. H-P1 reads its first 32 seeds after
  its cap, and H-V1 reads all seeds. Example: an E-P2 MET at look 1 still blocks the `_plan_w` NULL-RECORD at look 2.
- **A-M2, replay binding**: each replay row carries the tree sha and the sha256 of its `real` run's replayed (pattern, r, tick)
  series (`replay_binding`). A row whose binding differs from the current `real` run voids the tier. No sequential decision
  is taken while a validity check is pending. The primary reads "PENDING (validity checks not yet run: ...)", and the cross
  verdict says PENDING with no candidate wording.
- **C R2-M1, wording**: a look that continues prints "CONTINUE at look k of K, n=... (no verdict): next look n=...". It never
  prints "NOT MET ... CONTINUE". The no-harm compare fixture accordingly reads "H-P1 CONTINUE at look 1"; section 4's
  "no-harm -> NOT MET" is replaced for look 1.
- **B-N4**: a VOID no-encoder tier is reported beside the production verdict ("no-encoder VOID: reported beside, no
  verdict") and never vetoes it. This replaces addendum 3's "A VOID no-encoder tier keeps the verdict PENDING". The NULL-RECORD
  precondition (addendum 3) is unchanged: it still needs the objective endpoint clean on BOTH tiers. That only ever withholds
  a record (row folder `OPEN-QUESTIONS.md`, Q1).
- **B-N7, one mechanism**: `seq_boundaries` / `seq_decision` / `seq_record` are the project's one alpha-spending
  implementation (graph-exists "alpha spending": no other site). G1(c)-2, G2 and every later P1 pre-registration import them.
- **Numerical fix**: with 19 looks, look 1's spending 2 - 2 Phi(z / sqrt(tau)) is 0.0 in floating point, and `seq_boundaries`
  crashed. It now uses the tail form 2 Phi-bar(z / sqrt(tau)) and z = -Phi^-1(pi). The 4-look boundaries are unchanged
  (their tests pin them).

### Harness binding

`src/snn_data_lake/instruments/rpe_charge.py`: `SD_PRE`, `V1_CAP` = 152, `V1_LOOKS`, `V1_SHAPE_T`, `V1_FUTILITY_LOOKS` = ()
and `SEQ` (each primary's claimed effect, looks, pre-registered SD and boundaries, in one place); `futility_open`;
`seq_decision` / `seq_record` take the design; `analyse` refuses a seed count that is not an H-V1 look and carries the E-*
endpoints from the look-1 block; `replay_binding`; `cross` REFUSED / PENDING / VOID-beside. Reds and greens:
`lanes/engine-1024/e1b-fix-row/REPORT.md`. The derivation is `lanes/engine-1024/e1b-fix-row/derive_v1.py` (logs
`logs/derive-*.log`). All numbers are brain-free simulations; no perception tier applies.

## Addendum 5 (2026-10-24, lane e1b-hp1, before any E1b brain run): H-P1's cap is raised to its power target

Append-only; written before any E1b brain run and before any E1b datum. Addenda 2-4 and every frozen section above are
untouched; where this addendum and an earlier text differ, this addendum rules. Owner, 2026-10-24 (verbatim, for H-V1):
"Raise its cap to its power target". The integrator applies the same P1 rule (P1 amendment item 2: power at the cap >= .8)
to H-P1 (lane brief E1B-HP1; told to the owner, no objection). Addendum 4 disclosed H-P1's power under E1's measured
per-seed shape at its cap 32 as 0.6544 (the owner question Q2 of `lanes/engine-1024/e1b-fix-row/OPEN-QUESTIONS.md`); this
addendum answers it.

### Sample size (P1)

**H-P1 (`_plan_w`, harm, claimed effect DH = 0.035, one-sided alpha .05, power target .8).**

- **Pre-registered SD**: 0.0695, E1's NO-ENCODER paired SD (seed 1 kept). The production power is borrowed from it
  (addendum 4, B-N3). The production tier decides (#4616). The OBSERVED SD at the stopped look decides `powered`.
- **Statistic**: unchanged: the paired t and the sign gate ceil(5 n / 8) at each look, against the look's Lan-DeMets
  O'Brien-Fleming t boundary.
- **The shape, and why no fixed-sample target exists under it**: R-S1b resamples E1's eight centred H-P1 d_s, shifted by
  the effect. Shifted by DH, exactly 5 of the 8 values are positive, so P(d > 0) = 5/8. That is the sign gate's own ratio
  K_MIN / 8. One-look power with the gate therefore peaks at 0.584 (n = 24) and falls toward .5 as n grows (P(gate) at n
  = 128: 0.5419). Without the gate, the t test alone would pass .8 by n = 24 (0.8468). (`lanes/engine-1024/e1b-hp1-row/
  gate_probe.py`, 100 000 campaigns per n, `logs/gate-probe.log`.) So the brief's literal formula ("the seed count at which
  shape power >= .8, times the design's inflation factor") has no solution: no one-look n reaches .8. The normal-model
  fixed-sample target is 26 seeds, and 26 x the 4-look inflation gives addendum 2's cap of 32.
- **The rule taken instead**: the cap is the first 8-seed look at which the SEQUENTIAL design's power under the shape plus
  DH is >= .8. Each look is another chance at the gate, so sequential power rises with the cap. This is the rule
  addendum 4 used for H-V1 (one more block while power at the cap < .8). The inflation factor is built in, because power
  is simulated on the sequential design itself. Shown for reference: the LD-OBF inflation factor at 23 looks is 1.0644
  (`derive_v1.py if 23`, 400 000 paths).
  Sequential power under the shape, with addendum 2's normal-model boundaries (`derive_p1.py power <cap> plain`, numpy,
  200 000 campaigns, RNG seed 62):

  | cap | 32 | 48 | 64 | 96 | 128 | 160 | 168 | 176 | **184** | 192 | 256 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | power | 0.6588 | 0.7029 | 0.7283 | 0.7616 | 0.7810 | 0.7952 | 0.7980 | 0.8010 | **0.8036** | 0.8063 | 0.8214 |

  At 176 the numpy estimate is 0.8010, but the harness (`rpe_charge.seq_decision`, 40 000 campaigns, seed 9191) gives
  **0.7987 < .8**. As addendum 4 did at 144, the cap takes one more block. **Hard cap = 184 seeds per tier, 23 looks of 8
  seeds** (8, 16, ..., 184). Power there: **0.8036** (numpy, 200 000) and **0.8038** (harness, 40 000).
- **Boundaries**: calibrated on the shape exactly as addendum 4 calibrated H-V1's (`derive_p1.py calib 184`, 200 000
  campaigns, RNG seed 51). The shape-calibrated boundary is BELOW the normal-model boundary at every one of the 23 looks.
  The rule "never below the normal-model boundary" therefore keeps the normal-model LD-OBF boundaries
  (`SEQ["H-P1"]["bounds"] = None`):

  | look | n | t | look | n | t | look | n | t |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 8 | 64.0 (*) | 9 | 72 | 3.0889 | 17 | 136 | 2.1751 |
  | 2 | 16 | 16.2946 | 10 | 80 | 2.9062 | 18 | 144 | 2.1106 |
  | 3 | 24 | 7.5479 | 11 | 88 | 2.7537 | 19 | 152 | 2.0516 |
  | 4 | 32 | 5.4969 | 12 | 96 | 2.6238 | 20 | 160 | 1.9973 |
  | 5 | 40 | 4.5528 | 13 | 104 | 2.5112 | 21 | 168 | 1.9472 |
  | 6 | 48 | 3.9879 | 14 | 112 | 2.4124 | 22 | 176 | 1.9007 |
  | 7 | 56 | 3.6010 | 15 | 120 | 2.3247 | 23 (cap) | 184 | 1.8575 |
  | 8 | 64 | 3.3137 | 16 | 128 | 2.2461 | | | |

  (*) At n = 8 the LD-OBF z boundary is far in the tail, and `t_quantile` returns its 64.0 bisection ceiling. That ceiling
  is the boundary the harness applies. Under the null, P(t_7 > 64) is about 3e-11, so the understatement adds at most about 3e-11 to the type I. Disclosed and not changed: `pool_selection.
  t_quantile` is shared code outside this brief.
- **Type I under the REAL shape** (planted-shape null): **0.0071** (numpy, 200 000, seed 61) and **0.0069** (harness,
  40 000, seed 9090). Under the normal null: 0.0077 (harness); under the normal model at DH, power at the cap is 0.9938 (harness). R-S1b (<= .05 + 3 SE) holds.
- **Futility** (non-binding, so continuing costs no type I; conditional power < .10, only where the mde from the
  PRE-REGISTERED SD is <= DH; B-N2): it opens at **looks 9-22 (n = 72 to 176)**. mde_pre is 0.0361 at n = 64 (closed) and 0.0322 at n = 72 (open)
  (`derive_p1.py futility 184 plain`). P(futility) under the null: 0.9081 (shape, harness) and 0.9010 (normal, harness).
  At DH it is 0.0000 for both.
- **P(INCONCLUSIVE)** (a non-harm stop whose mde at the observed SD exceeds DH): 0.0000 under the null and at DH, for both
  the shape and the normal model (harness, 40 000). A futility stop at n >= 72, or the cap without a crossing, is powered
  at E1's SD and takes the NULL-RECORD path (one 10x re-test before any deletion; addendum 3). It is never a deletion by
  itself.
- **Harm crossing by look 2**: 0.0000 (shape and normal).
- **A disclosed limitation of the shape model** (owner question, `lanes/engine-1024/e1b-hp1-row/OPEN-QUESTIONS.md`): the
  gap between shape power and normal-model power comes from an 8-point resample whose positive fraction sits exactly on
  the sign gate. A smoother shape would move P(d > 0) off 5/8 and change the cap a lot in either direction. The cap above
  applies the P1 rule as written to the shape model R-S1b uses. It does not re-open the frozen sign gate.

### Shared looks (H-P1 and H-V1)

Both primaries are decided at 8-seed blocks starting at seed 8, so H-V1's 19 looks are H-P1's first 19. The campaign runs
`CAMPAIGN_LOOKS` = H-P1's 23 looks. Looks 1-19 decide both primaries. At looks 20-23 (160, 168, 176 and 184 seeds) only
H-P1 is decided, and H-V1 is carried from its cap 152 (its replay rows stop at seed 159). Each primary stops for itself:
once H-V1 has stopped, the campaign continues while H-P1 is CONTINUE and below its cap, and vice versa (addendum 3's stop
rule, per primary). The E-* endpoints and the dispositions still read the look-1 block (addendum 4, A-B1).

### Runs per look and hours (production tier decides, #4616; the no-encoder tier runs beside on the same looks)

- Look 1 is the frozen campaign: 86 runs plus the 16-run bridge (unchanged).
- **Looks 2-23 each add `real` + `pfrozen` x 8 seeds** = 16 runs per tier, 32 runs per look. Addendum 4's "looks 5-19
  add `real` only" is replaced, because H-P1 now needs `pfrozen` at every look up to its cap.
- Worst case to the H-P1 cap: 102 + 22 x 32 = **806 runs, about 9 h 11 min no-encoder-equivalent** (41 s per run, E1).
  Addendum 4's worst case was 438 runs (about 5 h 0 min), so this adds **368 runs, about 4 h 11 min**. Of those, 240 are
  the `pfrozen` runs at looks 5-19 and 128 are looks 20-23.
- The production per-run time is measured at look 1 (stage A). Each later look is re-estimated from it before its
  `resource:timing` claim.

### Harness binding

`src/snn_data_lake/instruments/rpe_charge.py`: `P1_CAP` = 184, `P1_LOOKS`, `P1_FUTILITY_LOOKS` (72..176) and `SEQ["H-P1"]`
(bounds None, the normal-model boundaries), beside H-V1's constants in the same block. `CAMPAIGN_LOOKS` = the longer of
the two primaries' looks; `analyse` refuses a seed count that is not a campaign look, and carries each primary from its own
cap, including the cross-tier sign-agreement denominator and its AGREE_MIN bar (each endpoint counts only the seeds its own
verdict reads, so H-V1 past look 19 is judged on 152 seeds, not the campaign's). `--look K` reaches 23. The brain-free harm fixture's per-seed spread is
narrowed (0.004 to 0.001 per step), so its planted harm still crosses look 1 (t > 64.0). Reds and greens:
`lanes/engine-1024/e1b-hp1-row/REPORT.md`. The derivation is `lanes/engine-1024/e1b-hp1-row/derive_p1.py` (logs
`logs/derive-*.log`). All numbers are brain-free simulations; no perception tier applies.

## Addendum 6 (2026-10-24, lane e1b-fold, before any E1b brain run): erratum, supersession list, panel fold

Append-only; written before any E1b brain run and before any E1b datum. Addenda 2-5 and every frozen section above are
untouched; where this addendum and an earlier text differ, this addendum rules. It folds the arc-2 panel's E1b findings
(`lanes/engine-1024/rv-arc2-a-row`, `-b-row`, `-c-row` REPORT.md): A-M3, A-M4, A-m3, A-m4, A-m6, B MAJOR-1, B MINOR-4,
C-M1, C-M2, C-M3, C-m12, C-m13, C-m14. Nothing here changes a cap, a boundary, a claimed effect or a seed count.

### Erratum: what decides `powered` (B MAJOR-1)

Addendum 5's Futility / P(INCONCLUSIVE) bullet says: "A futility stop at n >= 72, or the cap without a crossing, is powered
at E1's SD and takes the NULL-RECORD path". That sentence is withdrawn. It contradicts addendum 3 ("the OBSERVED SD at the
stopped look decides `powered`"), the P1 amendment item 2 (a stop reports the mde at the seed count reached, and P8 item 4
decides) and P8 null-scale item 4 (underpowered = INCONCLUSIVE). The rule is: **`powered` is decided at the OBSERVED SD at
the stopped look** (addendum 3; `rpe_charge.seq_decision` / `seq_record`). The 0.0000 P(INCONCLUSIVE) figures of addenda
4 and 5 are simulations at E1's NO-ENCODER SD; they predict, they do not decide. The production tier decides (#4616) and
has no measured SD before look 1, so a production stop powered at E1's SD may still read INCONCLUSIVE at its own.

Also from the fold (A-m4): an observed SD of exactly 0 (every d_s equal) gives mde 0. That is **not** powered. The stop
reads "degenerate: zero variance, not powered: INCONCLUSIVE" and names the endpoint and its arm; a constant negative mean
gives t = -inf (was 0.0).

### Supersession list (C-M2, C-m13): the current values beside the earlier text they replace

| earlier text | stated there | current (rules) | source |
|---|---|---|---|
| add. 2 / 3: hard cap | 32 per tier, looks 8/16/24/32 | H-P1 **184** (23 looks), H-V1 **152** (19 looks); the campaign runs H-P1's 23 | add. 4, 5; `P1_CAP`, `V1_CAP` |
| add. 3 :531 H-V1 | capped at 32, power .303 | cap 152, shape power 0.8164 (harness) | add. 4 |
| add. 3 :531 H-P1 | cap 32, power .816 | cap 184, shape power 0.8038 (harness) | add. 5 |
| add. 3 :551 powered-SD bound at the cap | H-P1 0.075, H-V1 0.103 (DH or DM / 0.4650) | **H-P1 0.1758** = DH / ((1.8575 + 0.8436) / sqrt 184); **H-V1 0.1961** = DM / ((2.1730 + 0.8440) / sqrt 152) | `_mde` at each cap |
| add. 3 :570-575 per-look table | 4 looks | the per-look bound is `DH or DM / _mde(n, _bound_t(n), 1)` at any look; the 4-look table is history | `rpe_charge._mde` |
| add. 3 :577-580 consequence | "every H-V1 non-harm stop reads INCONCLUSIVE" | every stop is decided at its observed SD; at E1's SD an H-P1 cap stop is powered (0.0695 <= 0.1758; mde 0.0138) and an H-V1 cap stop is not (0.2040 > 0.1961: mde 0.0499 > DM 0.048, add. 4's P(INCONCLUSIVE) 0.3649) | this addendum |
| add. 2 :448 futility drift | theta = delta sqrt(32) / SD_obs, tau = n / 32 | theta = delta sqrt(cap) / SD_obs, tau = n / cap, each primary its own cap | `seq_decision` |
| add. 2 :560 "the look's O'Brien-Fleming t boundary" | normal-model LD-OBF t | H-P1: normal-model LD-OBF t (add. 5); H-V1: the shape-calibrated `V1_SHAPE_T` (add. 4) | `_bound_t` |
| add. 2 :500 R-S2 | naive optional stopping 0.099 at 4000, 0.108 at 40 000 (4 looks) | over H-P1's 23 looks: naive **0.1310**, sequential 0.0075, at 4000 campaigns (seed 101) | `e1b-fold-row/logs/derive6.log` |
| add. 4 / 5: "H-P1 / each primary carried from its cap" | carried from the cap | carried from its **own stop of record** (below, A-M4) | `_n_owed` |
| add. 3 / 4 "a VOID no-encoder tier is reported beside" | VOID only; a valid but short / pending / other-tree no-encoder tier held production PENDING | the no-encoder tier is reported beside **in every state** (below, C-M3) | `cross` |
| add. 5 :794 "its replay rows stop at seed 159" | rows stop | `replay` writes a row for every seed of the look; rows past 159 are **written and binding-checked, not read** (C-m12) | `replay`, `analyse` |

### Each primary is carried from its own stop (A-M4)

Addendum 3 makes the first stop the decision of record. The harness carried each primary from its CAP, so a primary that
had stopped still owed its later runs: a missing later `pfrozen` run made the whole tier PENDING (H-P1 stopped, H-V1
running), and a stopped H-V1 whose replay rows ended read NO VERDICT. Now each primary reads min(look, its stop of record);
the stop is the first look whose complete prefix stops (`_n_owed`, which reuses `seq_record`). Runs past a stop are not
owed, are not pending, and a later look reprints the stop. Cross-tier sign agreement counts the seeds both tiers read
(the smaller of the two stops). `real` is still owed at every look while the campaign runs (H-V1's replay rows need it).

### The seed-sensitivity flag reads the sequential bar (A-M3)

`seed_sensitive` compared the full and the without-degenerate-seed t against the fixed-sample BAR = 1.895 and K_MIN = 5.
It now uses the stopped look's `bound_t` and `k_min` (the bar of the decision it guards); with no sequential decision it
keeps BAR / K_MIN. Degeneracy is screened on the `real` / `pfrozen` pair of every seed of the look, not only on the look-1
block's five arms.

### The no-encoder tier never holds the production decision (C-M3, A-m6)

Owner #4616 (verbatim): "production tier decides". Addendum 3 and addendum 4 B-N4 named only a VOID no-encoder tier as
"reported beside"; the code also held production PENDING when a VALID no-encoder tier was short, had pending checks, or ran
on another tree. That hold is removed: the no-encoder tier is reported beside in every state ("no-encoder VOID / on another
tree / short (n/owed seeds) / checks pending: reported beside") and never holds or vetoes the production decision. The
NULL-RECORD precondition (addendum 3) is unchanged: it still needs the objective endpoint clean on BOTH tiers, which only
ever withholds a record.

### R-S1b is validated on its own calibration shape (A-m3)

R-S1b resamples the same 8 E1 no-encoder d_s that `V1_SHAPE_T` was calibrated on, so its type-I check confirms the
calibration on its fitting data. Type-I control under the shape boundaries is therefore established only for that
no-encoder shape, not for the production tier's (which decides, and is first seen at look 1). One out-of-sample check is
added (`test_rs1b_out_of_sample_shape_leave_one_out_keeps_type_one`): each of the 8 leave-one-out 7-value shapes, centred,
500 campaigns each, RNG seeds 9000-9007. Result (`lanes/engine-1024/e1b-fold-row/derive6.py`, `logs/derive6.log`): H-P1 per-shape type I
0.000-0.032, pooled **0.0043**; H-V1 per-shape 0.028-0.062, pooled **0.0450**. Both hold: pooled <= .05 + 3 SE (4000) =
.0603 and every shape <= .05 + 3 SE (500) = .0792. A held-out H-V1 shape reaches .062, so the shape boundaries are not
exactly .05 off their fitting data; the check bounds that excess, it does not remove it.

### Family-wise false-harm rate per tier (B MINOR-4)

H-P1 and H-V1 each decide at one-sided .05 with no correction (Revision 2's multiplicity rule), and a harm verdict is exempt
from P8's re-test. Per tier the family-wise false-harm rate is at most 1 - .95^2 = **.0975** (independent endpoints; the
Bonferroni bound is .10). Under the designs' simulated type I it is about **.0069 + .0510 = .058** under E1's shape
(H-P1 0.0069, H-V1 0.0510, harness 40 000) and at most **.0077 + .05 = .058** under the normal null. Only the production
tier decides a deletion (#4616), so this per-tier rate is the campaign's deletion FWER; the no-encoder tier's equal rate
deletes nothing. Whether to correct is the owner's call (row folder `OPEN-QUESTIONS.md`, Q1); until answered, no correction.

### The H-P1 cap's owner status (C-m14)

One wording, which replaces addendum 5's "told to the owner, no objection" and "this addendum answers it": the H-P1 cap of
184 is **a disclosed session call applying P1 item 2; the owner's choice (keep 184 / 32 as a disclosed departure /
smoothed shape) is OPEN** (`lanes/engine-1024/e1b-hp1-row/OPEN-QUESTIONS.md` Q1). 184 binds until that question is
answered.

### Harness binding

`src/snn_data_lake/instruments/rpe_charge.py`: `_n_owed(key, seeds, d_by_seed)` (each primary's stop of record), `analyse`
(`n_owed`, the stopped-look sensitivity bar, the every-seed degeneracy screen, `ZERO_SD` / `ARM_OF` naming), `cross`
(no-encoder beside in every state), `seq_decision` / `status` (zero SD never powered), `ttest` (-inf). Reds and greens:
`lanes/engine-1024/e1b-fold-row/REPORT.md`. All numbers are brain-free simulations or arithmetic; no perception tier applies.

## Addendum 7 (2026-10-24, lane e1b-smooth, before any E1b brain run): H-P1's cap under a smoothed planted shape (owner ruling)

Append-only; written before any E1b brain run and before any E1b datum. Addenda 2-6 and every frozen section above are
untouched; where this addendum and an earlier text differ, this addendum rules. Owner, 2026-10-24 (verbatim, in chat to
claude-engine-1024): **"Smoothed shape"**. This answers Q1 of `lanes/engine-1024/e1b-hp1-row/OPEN-QUESTIONS.md` with the
option the owner was shown, (c): "re-state R-S1b's planted shape as a smoothed one (e.g. kernel bootstrap) and re-derive
the cap, as a pre-data addendum before any brain run". It replaces addendum 5's cap of 184 and addendum 6's line "184
binds until that question is answered" (section "The H-P1 cap's owner status"). Addendum 5's raw-shape derivation stays
as a disclosed reference. The frozen statistic is unchanged: the paired t and the sign gate ceil(5 n / 8) at each look.
A change to the sign gate would be a new pre-registration.

### Supersession list: the current values beside the earlier text they replace (as addendum 6's table)

| earlier text | stated there | current (rules) | source |
|---|---|---|---|
| add. 5 hard cap; add. 6 rows "hard cap", "add. 3 :531 H-P1" | H-P1 184 (23 looks), shape power 0.8038; the campaign runs H-P1's 23 | H-P1 **40** (5 looks), smoothed-shape power 0.8097 (raw-shape 0.6845, reference); the campaign runs H-V1's 19 | `P1_CAP`; `logs/scan-hp1-plain.log`, `logs/verify-hp1-smooth-40.log`, `logs/verify-p1-raw-normal-40.log` |
| add. 6 "The H-P1 cap's owner status" | 184 binds until the owner answers | answered ("Smoothed shape"): 40 binds | owner, 2026-10-24 |
| add. 5 futility | open at looks 9-22 (n = 72-176) | no futility look (`P1_FUTILITY_LOOKS` = ()) | `logs/futility-hp1-40.log` |
| add. 5 boundaries, look 1, and its harness-binding fixture note; e1b-hp1 Q2 | t = 64.0, `t_quantile`'s ceiling; the planted harm crosses look 1 at t > 64.0 | t = 9.8615 (5 looks), no ceiling | `logs/calib-hp1-40.log`; `fixture_runs` |
| add. 5 type I / P(futility) / P(INCONCLUSIVE) | 0.0069 / 0.9081 / 0.0000 (raw shape, cap 184) | the Sample size section below | `logs/verify-*-40.log` |
| add. 6 row "powered-SD bound at the cap", H-P1 | 0.1758 | **0.0840** | `logs/derive6-cap40.log` |
| add. 6 row "add. 3 :577-580 consequence", H-P1 | an H-P1 cap stop at E1's SD is powered (0.0695 <= 0.1758; mde 0.0138) | powered (0.0695 <= 0.0840; mde 0.0290) | `logs/derive6-cap40.log` |
| add. 6 row "add. 2 :500 R-S2" | 23 looks: naive 0.1310, sequential 0.0075 | 5 looks: naive 0.1075, sequential 0.0380 | `logs/derive6-cap40.log` |
| add. 6 row "add. 5 :794" (C-m12) | `replay` rows past seed 159 are written and binding-checked, not read | moot: the campaign ends at H-V1's cap, seed 159 | `CAMPAIGN_LOOKS` = `V1_LOOKS` |
| add. 6 A-m3, H-P1 pooled leave-one-out | 0.0043 | 0.0187 | `logs/derive6-cap40.log` |
| add. 6 B MINOR-4, per-tier FWER | about .058 | about .085 (raw shape), at most .090 (normal) | this addendum, Sample size |
| add. 5 "Shared looks" | H-V1 carried from 152 at looks 20-23 | H-P1 carried from its stop (at most 40) at looks 6-19 | `_n_owed` |
| add. 5 runs and hours | 806 runs, about 9 h 11 min | 454 runs, about 5 h 10 min | this addendum, Runs |

### Sample size (P1)

**H-P1 (`_plan_w`, harm, claimed effect DH = 0.035, one-sided alpha .05, power target .8).**

- **Pre-registered SD**: 0.0695, unchanged (E1's NO-ENCODER paired SD, seed 1 kept; the production power is borrowed from
  it, addendum 4 B-N3). The production tier decides (#4616). The OBSERVED SD at the stopped look decides `powered`
  (addendum 6 erratum).
- **The smoothed planted shape** (it replaces R-S1b's raw 8-point resample for H-P1): a variance-corrected Gaussian
  kernel bootstrap (Silverman 1986, sec. 6.4.1, the "shrunk" smoothed bootstrap) of E1's eight centred H-P1 d_s:
  y = (x_I + h e) / sqrt(1 + h^2 / s0^2) + mu, where I is uniform on the 8 values, e ~ N(0, 1), and s0^2 is the 8
  values' ddof-0 variance. That variance is the raw bootstrap's, so the smoothing changes the shape and keeps the
  spread.
  - **Bandwidth rule**: Silverman's rule of thumb, h = 0.9 min(SD, IQR / 1.34) n^(-1/5). On the 8 values: SD 0.0694,
    IQR 0.0860 (0.0183 to 0.1043, inclusive quartiles), so IQR / 1.34 = 0.0642. That gives **h = 0.9 x 0.0642 x 8^(-0.2)
    = 0.0381**, with a shrink factor of 0.8626.
  - **P(d > 0) at DH: 0.6603**, off the sign gate's 5/8 = 0.6250, where the raw shape is pinned
    (`lanes/engine-1024/e1b-smooth-row/derive_smooth.py bw`, `logs/bw.log`).
- **The cap rule** is addendum 5's, unchanged: the first 8-seed look at which the SEQUENTIAL design's power (shape plus
  DH) is >= .8, in both the numpy (200 000 campaigns) and the harness (40 000) estimate. The inflation factor is built in,
  because power is simulated on the sequential design itself. Smoothed-shape power with the normal-model LD-OBF
  boundaries (`derive_smooth.py scan H-P1 184 200000 plain`, RNG seeds 71/72, `logs/scan-hp1-plain.log`):

  | cap | 32 | **40** | 48 | 56 | 64 | 96 | 128 | 184 |
  |---|---|---|---|---|---|---|---|---|
  | power | 0.7720 | **0.8078** | 0.8322 | 0.8505 | 0.8661 | 0.9054 | 0.9292 | 0.9529 |

  The harness (`rpe_charge.seq_decision`, 40 000, seed 9393) gives **0.7717 at 32** and **0.8097 at 40**
  (`logs/verify-hp1-smooth-32.log`, `-40.log`). **Hard cap = 40 seeds per tier, 5 looks of 8 seeds** (8, 16, 24, 32,
  40). Numpy power at the cap: 0.8080 (seed 82, `logs/power-hp1-40-smooth.log`).
- **Boundaries**: the shape-calibrated boundary was derived as addendum 4 derived H-V1's, now under the smoothed shape
  (`derive_smooth.py calib H-P1 40 200000`, seed 91, `logs/calib-hp1-40.log`). It is BELOW the normal-model LD-OBF one
  at every look, so the normal-model boundaries stay (`SEQ["H-P1"]["bounds"] = None`):

  | look | 1 | 2 | 3 | 4 | 5 (cap) |
  |---|---|---|---|---|---|
  | n | 8 | 16 | 24 | 32 | 40 |
  | t used (normal-model) | 9.8615 | 3.4103 | 2.4658 | 2.0415 | 1.7858 |
  | smoothed-shape t | 8.5490 | 3.1836 | 2.3084 | 1.5221 | -inf |

- **Type I** (<= .05 + 3 SE, which is .0533 at 40 000 and .0603 at 4000; futility ignored, conservative):
  - under the smoothed shape: **0.0233** (numpy, seed 81) and **0.0222** (harness, seed 9292);
  - under the raw 8-point shape: **0.0345** (numpy, seed 81), **0.0344** (harness, seed 9090) and 0.0338 (harness, seed
    9292);
  - under the normal null: 0.0398 (harness, seed 9090).
  All hold. In the suite, R-S1b runs on both shapes for H-P1 (`test_rs1b_type_one_holds_under_e1s_measured_per_seed_shape`
  over `SHAPES`, and its 40 000-campaign slow twin `test_rs1_rs2_slow_twin_40000_campaigns`).
- **Power at the cap** (harness, 40 000): smoothed shape **0.8097**, normal model 0.8680. Under the raw 8-point shape it is
  **0.6845**, a disclosed reference. It is below .8 because the raw shape's P(d > 0) sits on the gate, which is what the
  ruling answers. Harm crossing by look 2: smoothed 0.1111, raw 0.0876, normal 0.1280.
- **Futility**: no look before the cap has mde_pre <= DH (mde_pre at n = 32 is 0.0356; `derive_smooth.py futility H-P1
  40`). So `P1_FUTILITY_LOOKS` is empty, and every H-P1 non-harm stop is the cap stop at 40. A cap stop is not a null by
  itself: it reports its mde, and P8 item 4 decides (INCONCLUSIVE unless powered).
- **Powered at the cap** (the addendum 6 supersession row "powered-SD bound at the cap", for H-P1): an H-P1 cap stop is
  powered iff its observed SD <= **0.0840** = DH / ((1.7858 + 0.8509) / sqrt 40). At E1's SD the mde is 0.0290 <= DH.
  P(INCONCLUSIVE) at E1's SD under the null (a prediction, not a decision): smoothed 0.0006, raw 0.0000, normal 0.0324.
  At DH it is 0.0003 (smoothed). Source: `lanes/engine-1024/e1b-fold-row/derive6.py` re-run, `logs/derive6-cap40.log`.
- **R-S2 over H-P1's 5 looks** (4000 campaigns, seed 101): naive optional stopping **0.1075**, sequential 0.0380 (the
  log's "23 looks" label is derive6's literal). **A-m3 leave-one-out at cap 40**: H-P1 per-shape type I 0.006-0.042,
  pooled **0.0187**.
- **Family-wise false-harm rate per tier** (addendum 6, B MINOR-4): about 0.0344 + 0.0510 = **.085** under E1's raw shape
  and at most 0.0398 + .05 = **.090** under the normal null. Both are within the .0975 bound. No correction (owner
  question still open).

### Shared looks (H-P1 and H-V1)

H-P1's 5 looks are now H-V1's first 5. The campaign runs `CAMPAIGN_LOOKS` = **H-V1's 19 looks** (cap 152). Looks 1-5
decide both primaries; looks 6-19 decide only H-V1, and H-P1 is carried from its own stop of record (at most 40,
addendum 6 A-M4). `pfrozen` runs past seed 47 are not owed. `--look K` reaches 19.

### Runs per look and hours (production tier decides, #4616; the no-encoder tier runs beside on the same looks)

- Look 1: the frozen campaign, 86 runs, plus the 16-run bridge (unchanged).
- Looks 2-5: `real` + `pfrozen` x 8 seeds = 16 runs per tier, 32 per look.
- Looks 6-19: `real` only = 8 runs per tier, 16 per look.
- Worst case to H-V1's cap: 102 + 4 x 32 + 14 x 16 = **454 runs, about 5 h 10 min no-encoder-equivalent** (41 s per run,
  E1).
  - Against addendum 5's 806 (about 9 h 11 min): **352 runs fewer, about 4 h 0 min less**.
  - Against addendum 4's 438: 16 runs more (about 11 min), the `pfrozen` runs at look 5.
- To H-P1's cap alone: 102 + 4 x 32 = 230 runs, about 2 h 37 min.
- The production per-run time is measured at look 1, and each later look is re-estimated from it before its
  `resource:timing` claim (unchanged).

### H-V1 under the smoothed shape (reported beside, NO change; for a later owner decision)

The same kernel bootstrap on E1's H-V1 d_s gives h = 0.0265 (shrink 0.9905) and P(d > 0) at DM = 0.8684 (raw: 0.8750).
- The normal-model boundaries do not hold under it: type I at 152 is 0.1234 (`logs/scan-hv1-plain.log`). The boundaries
  must be shape-calibrated, as addendum 4's are.
- With smoothed-shape-calibrated boundaries per cap (`derive_smooth.py scan H-V1 184 200000 calib`,
  `logs/scan-hv1-calib.log`), power is 0.7980 at 144 and **0.8200 at 152**. So the smoothed shape gives the same cap,
  152.
- Today's design (`V1_SHAPE_T`, cap 152) under the smoothed shape: type I **0.0488**, power **0.8160** (numpy 200 000,
  seeds 81/82, `logs/power-hv1-152-v1shapet-smooth.log`).
- H-V1's cap, boundaries and planted shape stay as addendum 4 set them.

### Harness binding

`src/snn_data_lake/instruments/rpe_charge.py`, in the same single block (`SEQ`):
- `P1_CAP` = 40, `P1_LOOKS` = (8, 16, 24, 32, 40), `P1_FUTILITY_LOOKS` = ();
- `CAMPAIGN_LOOKS` = `V1_LOOKS`, through the existing `max(..., key=len)`.

Tests (`tests/test_rpe_charge.py`):
- `_smooth_campaigns` (the smoothed shape) and `_bandwidth` (pinned to h = 0.0381 / 0.0265);
- R-S3 / R-S4 stated rates at cap 40 for the normal, raw and smoothed models, with the power >= .8 bar on the shape each
  cap is powered on (`CAP_SHAPE`);
- R-S1b on the raw and the smoothed shape (`SHAPES`) and its slow twin;
- a decision past the new cap is refused, and a look at the new cap is accepted; a missing `pfrozen` run up to the cap
  is pending;
- re-bound to this addendum: `test_addendum_7_fixes_the_h_p1_cap_and_its_looks_in_one_place`, the
  futility test's H-P1 loop (closed at every look before the cap) and the CLI test (`--look` reaches 19, H-V1's cap);
- past-cap tests at look 7, since look 20 is no longer a campaign look: H-P1 carried from its cap (compare, cross, its
  agreement bar 30 of 40), the agreement bar over a look-1 stop, and H-V1 stopped at look 1 with later replay rows
  written but not read.

Reds and greens: `lanes/engine-1024/e1b-smooth-row/REPORT.md`. The derivation is `derive_smooth.py` and `raw_at.py`
(addendum 5's `derive_p1.py verify` at cap 40). All numbers are brain-free simulations or arithmetic; no perception tier
applies.

### Derivation timeline (integrator note, policy-enforcer #5321)

The policy-enforcer asked (#5241 to e1b-fold, 17:50; #5321 to e1b-smooth, 21:41) that the smoothing rule be committed
before any cap derivation, or, if derivations had already run, that their times be stated here. They had already run.
Times are 2026-10-24, -0500:
- 21:10: the integrator's brief named the rule: "Gaussian kernel; name the bandwidth rule, e.g. Silverman's"
  (`lanes/engine-1024/E1B-SMOOTH.BRIEF.md`). The brief did not carry #5241's commit-first requirement. That gap is the
  integrator's, not the lane's.
- 21:14: the lane fixed the bandwidth (`logs/bw.log`): Silverman's rule of thumb, in the variance-corrected ("shrunk")
  variant. The brief did not name the variant. The lane chose it before any cap was computed.
- 21:19 to 21:29: cap scans, boundary calibration, type I, futility (`logs/scan-*`, `calib-*`, `verify-hp1-*`).
- 21:41: #5321 reached the lane, after the derivation.
- 22:10: this addendum was committed.

One bandwidth and one variant were ever computed: `derive_smooth.py` has a single `bandwidth()`, and there is a single
`bw.log`. One choice remains disclosed and untested: the unshrunk smoothed bootstrap inflates the planted shape's
variance by 1.344 (= 0.8626^-2) for H-P1. That lowers power at every cap, and its cap was not derived. Cap 40 therefore
rests on the variance-preserving variant, which was chosen before any cap was seen.

## Erratum 2026-10-25 (lane e1b-r3, before any E1b brain run): the degeneracy screen past H-P1's cap (R3-M1)

Addendum 6 (A-M3) says degeneracy "is screened on the `real` / `pfrozen` pair of every seed of the look". That stopped
being true at addendum 7: cap 40 means `pfrozen` runs only on seeds 8-47, so the screen never saw H-V1's seeds 48-159
and `seed_sensitive` read False for any degenerate seed there (rv-arc2-r3 R3-M1, policy-enforcer #5378). The screen now
runs on every seed of the look (`have("real")`): a seed is degenerate when its mean reward is below SENSITIVITY_R in
every arm it ran. This reads the pre-registration's "every arm of that tier" as the arms a seed ran: all five on the
look-1 block (seeds 8-15), `real` + `pfrozen` up to H-P1's cap (seeds 16-47), `real` alone past it. A seed past the
cap is judged on one arm, so it is easier to name degenerate there; this is disclosed and kept (rv-arc2-r3 fold
review), because screening on the arms that ran is the R3-M1 fix. Each primary's sensitivity drops only the degenerate
seeds it reads, and the report prints those per primary beside the look's full list. No pre-registered number
changes. Red: `tests/test_rpe_charge.py::test_degeneracy_is_screened_on_seeds_past_the_h_p1_cap` (seed 50, `real`
only). The same fold writes `P1_FUTILITY_LOOKS = ()` outright: `range(72, P1_CAP, 8)` was already empty, and its
comment reasoned from n=72 (rv-arc2-r3 m6).

## Addendum 8 (2026-10-25, lane e1b-a8, before any E1b brain run): H-V1 also powered on the smoothed shape (owner ruling); smoothing sensitivity

Append-only; written before any E1b brain run and before any E1b datum. Addenda 2-7, the erratum of 2026-10-25 and every
frozen section above are untouched; where this addendum and an earlier text differ, this addendum rules. Owner,
2026-10-25 (verbatim, in chat to claude-engine-1024): **"I approve the H-V1 under smoothed shape"**. This answers Q2 of
`lanes/engine-1024/e1b-smooth-row/OPEN-QUESTIONS.md` with option (b): "state H-V1's design as powered on the smoothed
shape as well, with the cap and boundaries unchanged". H-V1's cap stays **152** (19 looks) and its boundaries stay
`V1_SHAPE_T`; H-P1's cap stays **40** (5 looks) with the normal-model boundaries. Only the owner moves a cap; nothing in
this addendum can move one. The frozen statistic is unchanged.

This section ("Rule") was committed BEFORE any derivation run of this addendum (policy-enforcer #5241 / #5321, the
lesson of addendum 7's derivation timeline). Every run is logged with a timestamp after that commit
(`lanes/engine-1024/e1b-a8-row/logs/`). A rule fixed after a number was seen would void the reading.

### Rule

**1. H-V1 is stated as powered on two planted shapes.** These are the raw 8-point resample (addendum 4) and the
smoothed shape of addendum 7 (the shrunk Silverman kernel bootstrap, h = 0.0265 on E1's H-V1 d_s). The design is
today's `SEQ["H-V1"]`. Its bars on EACH shape:
- type I <= .05 + 3 SE;
- power >= .8.

Both bars are taken on both estimators:
- numpy: `derive_smooth.rates`, 200 000 campaigns, seeds 8801 (null) / 8802 (claimed DM);
- harness: `rpe_charge.seq_decision` walked look by look, 40 000 campaigns, seeds 9494 / 9595.

The pins go in `tests/test_rpe_charge.py`:
- `CAP_SHAPE` names every shape each cap is powered on;
- `SHAPES` gains ("H-V1", "smooth"), so R-S1b's type-I test and its slow twin run on it;
- `STATED` gains the harness rates at 40 000 for ("H-V1", "smooth");
- the fast twin (4000) checks the rates within 3 SE, and power >= .8 - 3 SE on each powered shape.

The e1b-smooth reference (numpy, seeds 81/82: type I 0.0488, power 0.8160) is reported beside and is not re-used as a
pin. The harness changes only if a bar fails, and then through its ONE place (`SEQ`). A bar failure is reported to the
owner; it does not move a cap.

**2. Smoothing sensitivity (REPORT-ONLY).** The grid is 2 designs x 5 variants x 2 estimators, each at the null and at
the claimed effect.

Designs (today's harness, unaltered):
- H-P1: cap 40, normal LD-OBF t boundaries;
- H-V1: cap 152, `V1_SHAPE_T`.

Variants. Each is y = (x_I + h e) / c + mu on E1's 8 centred d_s, where c = sqrt(1 + h^2 / s0^2), s0^2 is the ddof-0
variance, and SD / IQR are ddof-1 / inclusive quartiles:

| id | variant | h | c |
|---|---|---|---|
| S | Silverman shrunk (the pre-registered one) | hS = 0.9 min(SD, IQR / 1.34) n^(-1/5) | shrunk |
| U | Silverman UNSHRUNK (addendum 7's disclosed untested variant; variance x1.344 for H-P1) | hS | 1 |
| SC | Scott's normal-reference rule | 1.06 SD n^(-1/5) | shrunk |
| H05 | half bandwidth | 0.5 hS | shrunk |
| H2 | double bandwidth | 2 hS | shrunk |

Estimators and counts: numpy 200 000 campaigns per cell (seeds 8801 null / 8802 effect), and harness 40 000 per cell
(seeds 9494 / 9595). The same seeds serve every variant (common random numbers). The S row of H-V1 is item 1's
re-derivation. Each cell reports P(harm), P(futility) and P(INCONCLUSIVE); the harness also reports P(harm by look 2).

Flags (named, never acted on):
- a **type-I risk** when type I > .05 + 3 SE of that estimator's count (numpy .0515, harness .0533);
- a **power risk** when the point power is < .8;
- an **estimator disagreement** when numpy and harness differ by more than 3 combined SE.

Each flagged variant is named below as a disclosed risk and goes to the owner as a question
(`lanes/engine-1024/e1b-a8-row/REPORT.md`). **No cap, boundary or shape moves from this table** (owner rulings fix H-P1
40 and H-V1 152).

The script is `lanes/engine-1024/e1b-a8-row/derive_a8.py`. It reuses `derive_smooth.py` (bandwidth, numpy walk) and
`derive_v1.py` (E1 table, prefixes). All numbers are brain-free simulations; no perception tier applies.

### Sample size (P1)

No sample size changes.
- H-P1: SD_PRE 0.0695, DH 0.035, cap 40, 5 looks, LD-OBF one-sided .05, no futility look (addendum 7).
- H-V1: SD_PRE 0.2040, DM 0.048, cap 152, 19 looks, `V1_SHAPE_T`, no futility look (addendum 4).

The caps' power >= .8 is now stated on these shapes: H-P1 on the smoothed shape, H-V1 on the raw and the smoothed
shape. The check is item 1's simulation, done before any data. A futility stop or an uncrossed cap is still not a null:
it reports its mde, and P8 null-scale item 4 decides.

### Results

Results appended below after the runs.
