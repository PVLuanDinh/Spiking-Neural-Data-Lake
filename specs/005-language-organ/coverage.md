# lang_coverage record -- machine-written by tools/lang_coverage.py
# specs/005-language-organ, contract A. Frozen BEFORE any arm is scored (FR-023).

commit: 95c3f806f19fe3c1dc704336581ade75541578de
blimp_root: data\lang\blimp
limit_per_paradigm: ALL

rule (verbatim):
  A pair is IN the restricted set when every word of BOTH its sentences is in the exposure lexicon. Membership is a function of the rule and the lexicon ONLY; no arm's score is an input.

lexicon_source:
  reasoning/world_kb_stream.py::load_kb() -- every (head, relation, tail) triple rendered into the word sequence the organ hears, split on whitespace, '_' and '-', lowercased, stripped of surrounding punctuation. NOT read off a trained organ.
exposure_triples: 605
exposure_lexicon_size: 508 words

paradigms with at least one covered pair: 0 of 67
covered pairs: 0 of 67000
floors: >= 8 of 67 paradigms, >= 200 pairs

admission controls, on the RESTRICTED set, this invocation (A6):
controls: NOT COMPUTABLE -- the restricted set is empty.
  An empty set cannot demonstrate that the instrument can go red on it, which is itself sufficient to refuse a verdict.

instrument check on the FULL universe -- NOT the verdict set, decides nothing:
  random   :  49.9%  (33425 of 67000 correct)  [expected near 50.0]
  constant :   0.0%  (0 of 67000 correct)  [expected 0.0]
  length   :  30.7%  (20591 of 67000 correct)  [leakage; docs/DEPTH_CEILINGS.md records 29.4% at 25 pairs per paradigm]

diagnostics (decide nothing):
  BLiMP word types in the exposure lexicon: 39 of 2637
  best per-pair word coverage reached      : 50.0%
  (diagnostic only -- the rule is 100% coverage and it is not relaxed here. research.md R1 permits a derived coverage fraction only if it is derived, committed and justified BEFORE scoring, never tried until something passes.)

outcome: NOT-P2-ELIGIBLE
  reason: paradigm floor missed: 0 of 67 covered, floor is 8
  reason: pair floor missed: 0 covered pairs, floor is 200
  reason: an admission control passed where it must fail, or was not computable
pair_ids: NONE -- no set was frozen, so no arm may be scored

This is the tool doing its job, not failing. research.md R1: widening the exposure corpus in response to this exit is choosing the pair set to get a verdict and is forbidden. The permitted outcome is: the swap is not P2-eligible, recorded, feature stops.
