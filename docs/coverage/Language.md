# Coverage evidence -- Language

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| generation_acquired_lm | 40 | src/snn_data_lake/cortex/brain.py:3480#_language; src/snn_data_lake/cortex/brain.py:3016#respond; src/snn_data_lake/cortex/language_acquisition.py:188#generate | lang_no_ab_source.txt | opus-5 language-auditor / live respond() route census, 40-tick brain, seed 0 | 2026-07-30 |
| comprehension_intent_firewall | 40 | src/snn_data_lake/cortex/brain.py:2907#respond | - | opus-5 language-auditor / live respond() route census, seed 0 | 2026-07-30 |
| generation_narrator | 35 | src/snn_data_lake/cortex/self_model.py:411#introspect; src/snn_data_lake/cortex/brain.py:2986#respond | - | opus-5 language-auditor / live respond() route census + benevolence turn, seed 0 | 2026-07-30 |
| pragmatics_implicature | 32 | src/snn_data_lake/reasoning/spike_chat_lake.py:1000#converse; src/snn_data_lake/reasoning/pragmatics.py:356#implicature | - | opus-5 language-auditor / live neuter-A/B on LakeChat._pragmatics, seed 0 | 2026-07-30 |
| constituency_parse_rose | 28 | src/snn_data_lake/cortex/brain.py:2952#respond; src/snn_data_lake/reasoning/rose_language.py:222#parse | - | opus-5 language-auditor / live respond('parse ...') + shuffled-input control | 2026-07-30 |
| nightly_language_skill | 25 | src/snn_data_lake/reasoning/nightly_skills.py:299#_ensure_domains; src/snn_data_lake/cortex/brain.py:2526#_enter_sleep | - | opus-5 language-auditor / static read of the sleep-tick consumer, NOT runtime | 2026-07-30 |
| comprehension_learned_intent_router | 22 | src/snn_data_lake/cortex/live_brain.py:512#chat; src/snn_data_lake/cortex/live_brain.py:495#route | - | opus-5 language-auditor / caller search for the route() readout, static | 2026-07-30 |
| grammar_competence_blimp | 12 | src/snn_data_lake/reasoning/blimp_bench.py:170#module; src/snn_data_lake/reasoning/ar_spiking_lm_serve.py:356#logprob | lang_blimp_bench_67x25.txt | opus-5 language-auditor / blimp_bench.py self-check, 67x25 pairs | 2026-07-30 |
| core_spiking_lm_serve | 5 | src/snn_data_lake/cortex/spiking_lm_serve.py:96#generate | - | opus-5 language-auditor / weight-file search + caller search | 2026-07-30 |
| rose_grounded_comprehension | 5 | src/snn_data_lake/reasoning/rose_language.py:443#comprehend | - | opus-5 language-auditor / live comprehend() on 3 ordinary sentences | 2026-07-30 |
| discourse_model | 0 | - | - | opus-5 language-auditor / tree-wide search, nothing implements it | 2026-07-30 |

**Mean over the eleven rows: 244 / 11 = 22.2.** Zero rows are `UNSCORED`. Zero rows are above 40, so
this file makes no claim the scorer's F6/F8 refusals apply to.

---

## The number moved UP from 0.0, and that is a correction, not a gain

The withdrawn ledger printed Language **0.0** off two rows: an `ar_spiking_lm_serve` row scored 0 under
rule 4 ("the trainer exports to `models/spiking_lm.npz`, the server loads `models/ar_spiking_lm.npz`, so
the load precondition never fires") and one `UNSCORED` placeholder. **That rule-4 claim is refuted on
runtime evidence.** The path split was fixed in `bcd53611` and both halves now agree; on this machine
`SpikingLMServe()` reports `available True`, returns `logprob('the cat sat on the mat') = -38.150`, and
generates `'most of the much sturned'`. The precondition fires.

So nothing was built between the two files. A stale precondition claim was retired, and a roster of two
rows was replaced by a roster of eleven. **Neither 0.0 nor 22.2 measures a change in the brain**, and
comparing them as progress would be the exact error `COVERAGE_RUBRIC.md:14-16` warns about.

Two things push the other way and are recorded as downgrades:

- The refuted row's replacement, `grammar_competence_blimp`, still scores only **12**, because the fixed
  path bought reachability from a *research-tier bench*, not from the brain. See below.
- **`models/` is gitignored.** On a fresh clone `available` is `False` and `grammar_competence_blimp`
  *is* a rule-4 zero. The band above is what runs on this machine; it is not reproducible from a
  checkout, and `docs/DEPTH_CEILINGS.md:448-465` already says so about every AR-LM number.

---

## What I verified at runtime, and how

All runs: `.venv-lake/Scripts/python.exe`, `PYTHONPATH=src`, `PYTHONUTF8=1`,
`make_brain(seed=0, persist=False)` -- the Policy-4 construction path `tools/charge_probe.py` uses.

**R1 -- the live route census, and it reversed my own first reading.** Fifteen free-conversation turns
on a *freshly constructed* brain produced **0/15** replies from the acquired language organ: 12 took
`Gf (second-order access hold: abstain)` (the narrator), 2 `Gc (crystallized)`, 1
`act (gated arm/lake command)`. Read alone that is a rule-4 zero for generation. It is not. Ticking the
same brain 40 times clears the hold (`_second_order_hold True -> ...`, `_metacog_rel 0.4193`), and the
census flips to **9/10 `Gf (fluid: acquired language center)`**. The generated text *is* the reply in a
running brain. A census on an unticked brain would have produced a confident false negative -- the same
shape of error that once flagged PFC and amygdala as dormant.

**R2 -- the generated text is character-level babble, and the coherence guard passes it.** Forced
directly, `Brain._language()` returned non-empty text on 15/15 narrator prompts, e.g.
`'dht to leemecthe da the ko the ts ine to ste s at ste in the saf'`; the raw organ on `'the sky is'`
gave `'wnde the ba ins to safhe an it'` and `Brain._coherent()` returned **True** on it.
`brain.py:3390-3391` says the guard "keeps a too-babbly generation from replacing the narrator". It
does not: it rejects empty, short and digit-heavy strings only. So in the steady state the brain's
free-conversation output is babble that no guard catches.

**R3 -- exposure really does write.** `LanguageAcquisition.W` is `(34, 88)`; one `expose()` call on a
sentence changed it, `max|dW| = 0.1578`. Comprehension-as-learning is a real weight update. Per the
project's own "movement is not activation" lesson, that is recorded as a *write*, not as evidence the
faculty pays off; 2,992 parameters and the ~57% architectural ceiling are in `DEPTH_CEILINGS.md:198-206`.

**R4 -- ROSE parse is live and consumed verbatim; it is positional, not syntactic.**
`respond('parse the cat sat on the mat')` returns route `language (ROSE constituency parse)` and puts
`[NP SPEC:the MOD:cat MOD:sat MOD:on MOD:the HEAD:mat]` into the reply text.
`respond('diagram the robot moved the block')` likewise. The control is the finding: the shuffled
string `'cat the mat sat on the'` parses to `[NP MOD:cat MOD:the MOD:mat MOD:sat MOD:on HEAD:the]` --
**HEAD is whatever token comes last**, and every other content word is `MOD`. It is a positional
template with VSA role-filler bindings behind it, not a constituency analysis.

**R5 -- ROSE `comprehend` is ungrounded on ordinary English and has no live caller.**
`comprehend` returned `{'known': False, 'answer': None, 'proposition': None, 'reason': 'ungrounded'}`
for 3/3 of `'the cat sat on the mat'`, `'cat the mat sat on the'`, `'the robot moved the block'` --
reproducing `DEPTH_CEILINGS.md:135-138`. The only callers of `.comprehend(` in the tree are
`rose_language.py:492` (inside its own `generate`, itself self-check-only) and its self-check at
`:566-568`. `Brain` calls `rose.parse` and never `rose.comprehend`.

**R6 -- pragmatics is live-wired and I neuter-A/B'd it. The stale claim was the "zero callers" one.**
`LiveBrain._reasoning_bridge` is `True` (the default at `live_brain.py:81`; the docstring at `:441`
saying "Off by default (reasoning=False)" is stale), `_lakechat` was built after 15 turns, and
`LakeChat._pragmatics` is a live `Pragmatics`. Setting `_pragmatics = None` and re-running the same
utterance changed the reply on **2 of 4** textbook cases:

```
'can you pass the salt?'        ON  source='reason'  "[pragmatics] literally a question, but you mean: request to pass the salt..."
                               OFF  source=None     "[honest gap] nothing reached conscious access..."      -> CHANGED
'some of the students passed'   ON  source='reason'  "[pragmatics] ... implicates 'not all' -- scalar implicature..."
                               OFF  source=None     "[honest gap] ..."                                      -> CHANGED
'do you have the time?'         implicature -> None                                                          -> identical
"it's cold in here"             implicature -> None                                                          -> identical
```

The two nulls are not noise. `Pragmatics.implicature(utterance, context=None)`
(`pragmatics.py:356`) needs `context` for the environmental-relevance maxim, and the live caller
(`spike_chat_lake.py:1000`) calls `implicature(text)` with **no context** -- so the module's own
headline example, "it's cold in here" as an indirect request, **cannot fire on the live path**. That is
a rule-4 zero inside an otherwise-charged faculty, which is why the row is 32 and not 40.
Also live-dead: `Pragmatics.act()` (speech-act classification) and `is_indirect_request()` have no
caller outside the self-check, and `PragmaticsLearner` -- the brain-native VSA/Hebbian half that is the
module's evidence the map is *acquirable* -- runs only in `__main__`.

**R7 -- the learned intent router is write-only.** `LiveBrain.chat` calls
`self._ixn.learn_turn(...)` on every turn (`live_brain.py:512`), so the R-STDP spiking intent router
learns from live conversation. Its readout, `_ixn.respond(...)`, is reachable only through
`LiveBrain.route()` (`:495`), and `route()` has **no caller anywhere except `live_brain.py`'s own
self-check** (`:847,:851,:863,:915,:919`). `Brain.respond` classifies intent with the keyword firewall
(`brain.py:2853`), never with the learned router. The faculty that "GENERALIZES to novel paraphrases
the fixed keyword router misses" is trained every turn and read by nothing.

**R8 -- no Language faculty is eligible for the tracked A/B instrument, and this is recorded.**
`tools/charge_probe.py language 60` exits 2: *"unknown drive source 'language'; known sources: drive,
help, ignition, novelty, percept, place, recall, reroute, surprise, taskset, timing"*
(`artifacts/lang_no_ab_source.txt`). The probe neuters a key of `Brain._drive_w`
(`brain.py:633-648`), and that dictionary contains **no language source at all** -- its observables are
`arm_stopped` and `map_position`. So the instrument that the rubric makes the sole gate for band > 40
cannot express any Language claim, in either direction.

**R9 -- `nightly_language_skill` is the one row I did not exercise, and its band says so.** The
sleep-tick language `Domain` is behind `if self._persist:` (`brain.py:2479`, comment: *"skipped under
`NORD_PERSIST=0` -> gates byte-identical"*), and every run above used `persist=False`, so I never made
it fire. The band is **25** on a static read of the consumer chain plus the fact that its output lands
in `_skills_state` and from there in `state()` (`brain.py:2805`) -- i.e. telemetry, which is the
definition of 21-40. The always-on daemon does run with persistence, so this is *not* a claim that the
guard never fires; it is a claim that I did not watch it fire and did not credit it as though I had.

**R10 -- the grammar instrument, machine-recorded** (`artifacts/lang_blimp_bench_67x25.txt`, exit 0,
67 x 25 = 1675 pairs). This is the only recorded run behind any row here, and it reproduces
`DEPTH_CEILINGS.md`'s table to the decimal:

```
  control random  :  51.1%      control constant:  0.0%
  control length  :  29.4%      oracle          : 100.0%
  model           sum-logprob  per-char   delta
  AR spik-RWKV          47.7%     49.0%    +1.3        10,339,425 params
  e-prop ALIF-LM        46.9%     51.9%    +5.0           812,256 params
  e-prop flat LM        48.3%     50.4%    +2.1           112,475 params
```

Chance is 50.0 and the char-level bar is 53.0 (a count-based 5-gram). Every row is at or below chance
under one scoring and at or barely above it under the other, the three controls behave correctly, and
the length-only control at 29.4% is why neither column can be quoted alone. **This artifact is evidence
of an instrument that works and a faculty that is absent.** It supports band 12 and nothing higher: the
serves load, they score, and no brain path reads them.

---

## Why no row exceeds 40, stated as a limit of the instrument and not of the evidence

Three rows have runtime evidence that meets the *band 41-60* definition -- "the output demonstrably
alters something the brain itself later reads -- a reply" (`COVERAGE_RUBRIC.md:53`):

- `generation_acquired_lm` -- alters the reply text; 9/10 replies after 40 ticks.
- `comprehension_intent_firewall` -- alters the reply route and sends the turn down the gated arm/lake
  path; 1/10 turns rerouted.
- `pragmatics_implicature` -- alters the reply text and the `source` field; 2/4 utterances, neuter-A/B.

All three are scored at or below **40** anyway, because the rubric permits nothing above 40 without a
recorded A/B artifact and `tools/charge_probe.py` cannot produce one for a language signal (R8). The
honest statement is therefore **evidence-capped, not evidence-absent**: I measured the charge with a
one-off script, and a one-off script is not the tracked, `--rerun`-able artifact the standard demands.
I did not manufacture the artifact by other means, for two reasons worth writing down:

- A `python -c` command survives `record` but dies on `--rerun`, because `rerun` does
  `a["command"].split()` -- so it would be an artifact that cannot be re-executed, i.e. the forgery the
  exit-code mechanism exists to catch, dressed as compliance.
- A probe promoted into `tools/` would be the right fix, and `tools/**` is being edited by another
  agent this session. Closing this properly means **adding a language-eligible source or a
  reply-level observable to `charge_probe.py`**, and that is a change to the instrument, not an audit
  finding to smuggle into an evidence file.

**A second, mechanical hole found while working:** `record` writes only
`kind/command/exit/commit/recorded_at/recorded_by/python`, but F10 refuses any `kind: heldout`
artifact that does not also carry `metric:` and `baseline:`. So a machine-recorded held-out artifact
**always** fails F10, and **band > 80 is currently unreachable through the sanctioned path** -- the
only way to satisfy F10 is to hand-edit an artifact header, which is precisely what
`coverage_score.py:311` says must never happen. Language is nowhere near 80, so this cost me nothing;
it will cost Perception or Gf something.

---

## Capability inventory (Policy 2: what provides what, before anything is retired)

Format: `file` -- unique capability -- **live consumer**.

- `cortex/language_acquisition.py` -- online spiking predictive-coding char LM; `expose()` learns from
  live turns, `generate()` produces the reply. **`Brain._lm_serve` / `_language`
  (`brain.py:3384,3426`).**
- `cortex/self_model.py::SelfNarrator` -- deterministic state-grounded sentence generation; the only
  always-coherent generator in the stack. **`brain.py:2932,2951,2961`.**
- `reasoning/rose_language.py` -- VSA role-filler parse string, KB-grounded `comprehend`, SLDA
  competence head. **`parse` only, via `brain.py:2898`.**
- `reasoning/pragmatics.py` -- speech-act taxonomy, Gricean implicature, indirect-request detection,
  VSA-Hebbian learner. **`implicature` only, via `spike_chat_lake.py:1000`.**
- `reasoning/ar_spiking_lm_serve.py` -- order-sensitive full-sentence AR logprob over a frozen
  10,339,425-param Spiking-RWKV. **`blimp_bench.py:150` only.**
- `reasoning/eprop_lm_serve.py`, `reasoning/eprop_lm_alif_serve.py` -- e-prop recurrent-spiking char
  LMs, 112,475 / 812,256 params. **`blimp_bench.py:151-152` only.**
- `cortex/spiking_lm_serve.py` -- pure-Python `snn_lm/1` forward; the CORE-tier serve. **No live
  consumer; `src/snn_data_lake/training/spiking_lm.py:407` transfer check and its own self-check.**
- `reasoning/blimp_bench.py` -- the grammaticality instrument, with four controls that must fail.
  **Research-tier self-check.**
- `reasoning/language_readout_slda.py`, `reasoning/language_self_build.py` -- SLDA-RFF and R-STDP
  next-char readouts. **`nightly_skills.py` language Domain, on the sleep tick.**
- `cortex/interaction_learning.py` -- R-STDP spiking intent router. **`learn_turn` written every turn;
  readout unread (R7).**

**Nothing here should be retired on this audit's evidence.** Two notes for whoever does:
`cortex/spiking_lm_serve.py` is the *only* torch-free `snn_lm/1` reader, so deleting it orphans
`src/snn_data_lake/training/spiking_lm.py`'s export format; and `models/spikegpt/SpikeGPT-216M.pth` (861,716,013 bytes)
sits on disk with **no `.py` in the tree referencing it** -- `run_selfchecks.py:343` records SpikeGPT as
scrapped in v3.77, so that file is 862 MB of dead weight, not a faculty.

---

## Gaps, ranked by distance_to_100 x feasibility

1. **Nothing can be proven above 40 (11 rows x up to 60 points).** Add a language-eligible source or a
   reply-level observable to `tools/charge_probe.py`. Three rows already have the runtime evidence;
   they need a *recordable* channel. Cheapest item on this list by a wide margin.
2. **`discourse_model`: 0 -> anything (100 points on one row).** There is no coreference, anaphora,
   topic tracking, common-ground update or turn-taking model in the tree; `pragmatics.py:32` says so in
   its own honest framing. `chat_memory` recall of past turns is bag-of-words retrieval scored under
   Memory, and ToM belief tracking is scored under Social; neither is a discourse structure. This is
   the one genuinely *absent* sub-area the category claims.
3. **The learned intent router's readout (22 -> 61-80).** `route()` exists, learns every turn, and is
   read by nothing. Consuming it in `Brain.respond` alongside the keyword firewall is a small wire with
   a real behavioural consequence -- and it is the brain-native half of comprehension.
4. **`pragmatics.implicature` context (32 -> ~40, and one dead maxim revived).** Pass a context to the
   live call so the relevance maxim can fire; wire `act()` so speech-act force reaches the router.
5. **Grammar (12 -> ?) is the architectural ceiling and is NOT a wiring job.** `DEPTH_CEILINGS.md`
   eliminated corpus scale (4,119x moved BLiMP -1.6), capacity (3,400x scored *worse*), the unit,
   the context window and the learning rule by ablation. The best number in the whole arc is a char
   5-gram at **53.0%** against a 50.0% floor; the trained 10.3M AR model is **-1.8 / +0.7** points
   against its own untrained twin at SE ~1.2, and *the sign flips with the scorer*. Wiring any of this
   into `_route_percept_to_action` raises nothing -- `COVERAGE_RUBRIC.md:99-106` pre-declares Language
   unreachable by wiring, and this audit found no reason to argue with that.
6. **Babble reaches the user (quality, not band).** `_coherent()` accepts
   `'wnde the ba ins to safhe an it'`. Either the guard should test something (n-gram plausibility
   against the exposure) or the docstring at `brain.py:3390-3391` should stop claiming it filters
   babble. This moves no band and is the most visible defect in the category.

## Reproduce

```bash
cd D:/AI-project/spiking-neural-data-lake
# the recorded artifacts
python tools/coverage_score.py --rerun lang_no_ab_source          # exit 2, no language drive source
python tools/coverage_score.py --rerun lang_blimp_bench_67x25     # needs data/lang/blimp + models/ (gitignored)

# the runtime readings above, in order (R1-R7); all read-only, none writes state
PYTHONPATH=src PYTHONUTF8=1 .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.rose_language
PYTHONPATH=src PYTHONUTF8=1 .venv-lake/Scripts/python.exe -m snn_data_lake.reasoning.pragmatics
PYTHONPATH=src PYTHONUTF8=1 .venv-lake/Scripts/python.exe -m snn_data_lake.cortex.spiking_lm_serve
```

The route census, the `expose()` weight delta and the `_pragmatics` neuter-A/B were run from a
throwaway script against `make_brain(seed=0, persist=False)` and are **not** re-runnable from the tree.
Saying that is the point of gap 1: `tools/` is where a cited instrument has to live
(`DEPTH_CEILINGS.md:459-462`), and the `lang_*.py` / `ar_lm_*.py` scripts that document says were never
promoted have since appeared there -- so the pattern is already being fixed for the LM scripts and not
yet for the live-brain ones.
