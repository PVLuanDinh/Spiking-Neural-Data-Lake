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
