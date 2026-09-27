r"""Freeze the BLiMP subset the language-organ A/B may be scored on, and run the admission controls.

specs/005-language-organ, contract A (specs/005-language-organ/contracts/lang_ab.md).

WHY THIS RUNS BEFORE ANY ORGAN EXISTS. The restriction rule is a property of the pair set and of the
EXPOSURE PLAN, never of any arm's score (A1, A2, FR-023). A lexicon read off a trained organ is a
lexicon that could have been tuned, so the lexicon here is computed from
reasoning/world_kb_stream.py::load_kb() rendered into sentences -- the words the organ is PLANNED to
hear, before it has heard any of them. This is the only order in which "fixed before either arm is
scored" is achievable.

THE RULE, and it is printed verbatim in the output:

    A pair is IN the restricted set when every word of BOTH its sentences is in the exposure lexicon.

FLOORS (research.md R1, stated before the measurement so they cannot be moved after it):

    >= 8 of 67 paradigms with at least one covered pair
    >= 200 covered pairs

Missing either exits 2 with NOT-P2-ELIGIBLE. Exit 2 is not a failure of this tool; it is the tool
doing its job. Widening the exposure corpus in response to a 2 is choosing the pair set to get a
verdict, and research.md R1 forbids it by name.

Usage:
  python tools/lang_coverage.py [--blimp DIR] [--out specs/005-language-organ/coverage.md]
                                [--limit N]

  --blimp DIR   BLiMP *.jsonl directory (default <repo>/data/lang/blimp). Absent data SKIPs (exit 0).
                Download: https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip
  --out PATH    where to write the frozen record (default: stdout only).
  --limit N     pairs per paradigm to consider. Default: ALL. This bounds the UNIVERSE, not the rule;
                it is reported in the record so a shrunk denominator is visible. It cannot turn a
                floor miss into a pass -- shrinking the universe can only shrink the covered count.

Exit codes: 0 frozen-and-admissible, or a clean SKIP | 2 NOT-P2-ELIGIBLE | 1 tool error.
"""

import os
import re
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# A10: the always-on learn_daemon writes ~/.snn_data_lake continuously. Ten of this ledger's
# artifacts were once irreproducible for exactly this reason, so every invocation gets its own home.
_ISOLATED_HOME = tempfile.mkdtemp(prefix="lang_coverage_home_")
os.environ["NORD_HOME"] = _ISOLATED_HOME
os.environ.setdefault("NORD_PERSIST", "0")

from snn_data_lake.reasoning import blimp_bench as BB  # noqa: E402
from snn_data_lake.sensorimotor.world_kb_stream import load_kb  # noqa: E402

# The floors. Stated in research.md R1 BEFORE the measurement; duplicated here as the enforced form.
FLOOR_PARADIGMS = 8
FLOOR_PAIRS = 200
N_PARADIGMS_TOTAL = 67

RULE = (
    "A pair is IN the restricted set when every word of BOTH its sentences is in the exposure "
    "lexicon. Membership is a function of the rule and the lexicon ONLY; no arm's score is an input."
)

LEXICON_SOURCE = (
    "reasoning/world_kb_stream.py::load_kb() -- every (head, relation, tail) triple rendered into "
    "the word sequence the organ hears, split on whitespace, '_' and '-', lowercased, stripped of "
    "surrounding punctuation. NOT read off a trained organ."
)

_WORD = re.compile(r"[a-z']+|[0-9]+")


def sentence_words(sentence):
    """Tokenise a BLiMP sentence the same way the exposure lexicon is tokenised.

    One tokeniser for both sides is the point: a lexicon built with a looser tokeniser than the one
    that tests membership would report coverage the organ does not have."""
    return _WORD.findall(sentence.lower())


def render_triple(triple):
    """One exposure item's word sequence. The renderer is stated because it decides the lexicon.

    A triple IS a two-argument scene (research.md R6), so the rendered sentence is head, relation,
    tail in that order. The renderer introduces NO words of its own -- no determiners, no copula.
    That is a property of the exposure plan and it is reported, not corrected here: correcting it
    would be widening the corpus, which is what the floors exist to refuse."""
    head, rel, tail = triple
    joined = " ".join(str(x) for x in (head, rel, tail)).replace("_", " ").replace("-", " ")
    return _WORD.findall(joined.lower())


def exposure_lexicon(triples):
    """The word inventory of the exposure plan (A1). Sorted so the record is byte-reproducible."""
    lex = set()
    for t in triples:
        lex.update(render_triple(t))
    lex.discard("")
    return lex


def restrict(lexicon, pairs):
    """Return the ids of the pairs every word of which is in `lexicon`.

    Deliberately takes no scorer, no arm, no score and no threshold (A2). The absence of those
    parameters is what makes the membership untunable, and tests/test_lang_coverage.py asserts the
    signature stays that way rather than trusting this comment."""
    ids = []
    for pid, good, bad in pairs:
        if all(w in lexicon for w in sentence_words(good)) and all(
            w in lexicon for w in sentence_words(bad)
        ):
            ids.append(pid)
    return ids


def load_universe(root=None, limit=None):
    """Every candidate pair as (pair_id, good, bad), in a deterministic order.

    pair_id is "<paradigm>:<line index>", which is stable across runs because load_pairs reads the
    file in order and never shuffles."""
    names = BB.paradigms(root)
    out = []
    for name in names:
        for i, (good, bad) in enumerate(BB.load_pairs(name, root=root, limit=limit)):
            out.append((f"{name}:{i}", good, bad))
    return names, out


def score_pairs(scorer, pairs):
    """Score a scorer over an explicit pair list, with blimp_bench.score's exact semantics.

    blimp_bench.score() takes whole paradigms, and the restricted set is a SUBSET of pairs inside
    paradigms, so the comparison loop is repeated here rather than the instrument being edited to
    accept a pair list -- an instrument edited by the party being measured is not an instrument
    (C5.5). The semantics copied are: strict `>` so a TIE COUNTS AS WRONG, and an exception on a
    sentence counts as wrong rather than propagating. tests/test_lang_coverage.py asserts this loop
    agrees with blimp_bench.score on a whole paradigm, so it cannot drift away from the instrument
    unnoticed."""
    correct = 0
    for _pid, good, bad in pairs:
        try:
            if scorer(good) > scorer(bad):
                correct += 1
        except Exception:
            pass
    return correct, len(pairs)


def _pct(correct, total):
    return (100.0 * correct / total) if total else 0.0


def run_controls(pairs, out):
    """A6: the three data-only admission controls, on the restricted set, in this invocation.

    Returns (results, disqualified). `disqualified` is True when a control PASSED where it must
    FAIL -- which means the restriction introduced a leak and the set cannot carry a verdict (A7).

    THE FOURTH CONTROL (an unexposed organ) IS NOT BUILT. This said it "runs in
    tools/lang_organ_ab.py"; there is no such file, and there never has been -- a capability
    reported before it exists, which is exactly what P0's honesty guard forbids and what
    citation_guard cannot catch, because it only resolves .py names that appear as reproduction
    paths in MARKDOWN. It needs an organ, and specs/005-language-organ has not produced one, so it
    is OPEN WORK named here rather than a file name that reads like a location."""
    results = {}
    disqualified = False
    if not pairs:
        out.append("controls: NOT COMPUTABLE -- the restricted set is empty.")
        out.append(
            "  An empty set cannot demonstrate that the instrument can go red on it, which is "
            "itself sufficient to refuse a verdict."
        )
        return results, True

    c, n = score_pairs(BB.random_scorer, pairs)
    results["random"] = _pct(c, n)
    ok_random = abs(results["random"] - 50.0) <= 3.0
    out.append(
        f"  random   : {results['random']:5.1f}%  ({c} of {n} correct)  "
        f"[must be within 3 points of 50.0 -- {'ok' if ok_random else 'LEAK'}]"
    )

    c, n = score_pairs(BB.constant_scorer, pairs)
    results["constant"] = _pct(c, n)
    ok_const = results["constant"] == 0.0
    out.append(
        f"  constant : {results['constant']:5.1f}%  ({c} of {n} correct)  "
        f"[must be 0.0 -- ties count as wrong -- {'ok' if ok_const else 'LEAK'}]"
    )

    # A6: length leakage is PRINTED, never thresholded. Its value IS the size of the leak, and a
    # threshold would hide a growing one. FR-026: it is named in the ADOPT criteria for this reason.
    c, n = score_pairs(BB.length_scorer, pairs)
    results["length"] = _pct(c, n)
    out.append(
        f"  length   : {results['length']:5.1f}%  ({c} of {n} correct)  "
        "[leakage, PRINTED not thresholded -- this number IS the size of the length leak]"
    )

    if not (ok_random and ok_const):
        disqualified = True
        out.append(
            "  A7: a control passed where it must fail. The restriction introduced a leak and the "
            "set is DISQUALIFIED."
        )
    return results, disqualified


def instrument_check(pairs, out):
    """Run the same three controls on the FULL universe, labelled as NOT the verdict set.

    Why this exists. When the restricted set is empty the controls above are not computable on it,
    and a record that only says "not computable" leaves the reader unable to tell whether the tool
    would have caught a leak had there been a set to catch it in. This block demonstrates the
    control machinery can go red, on a set that is explicitly not the verdict's. It decides nothing:
    A6's obligation is controls on the RESTRICTED set, and this is not a substitute for it."""
    out.append("instrument check on the FULL universe -- NOT the verdict set, decides nothing:")
    c, n = score_pairs(BB.random_scorer, pairs)
    out.append(f"  random   : {_pct(c, n):5.1f}%  ({c} of {n} correct)  [expected near 50.0]")
    c, n = score_pairs(BB.constant_scorer, pairs)
    out.append(f"  constant : {_pct(c, n):5.1f}%  ({c} of {n} correct)  [expected 0.0]")
    c, n = score_pairs(BB.length_scorer, pairs)
    out.append(
        f"  length   : {_pct(c, n):5.1f}%  ({c} of {n} correct)  "
        "[leakage; docs/DEPTH_CEILINGS.md records 29.4% at 25 pairs per paradigm]"
    )


def diagnostics(lexicon, pairs, out):
    """How far from the floor, computed from the data rather than guessed.

    Reported so a reader of a NOT-P2-ELIGIBLE record knows whether the miss was marginal or
    structural. These numbers decide nothing; the rule above decides."""
    types = set()
    best = 0.0
    covered_types = set()
    for _pid, good, bad in pairs:
        ws = sentence_words(good) + sentence_words(bad)
        if not ws:
            continue
        types.update(ws)
        inlex = [w for w in ws if w in lexicon]
        covered_types.update(inlex)
        frac = len(inlex) / len(ws)
        if frac > best:
            best = frac
    out.append(f"  BLiMP word types in the exposure lexicon: {len(covered_types)} of {len(types)}")
    out.append(f"  best per-pair word coverage reached      : {100.0 * best:.1f}%")
    out.append(
        "  (diagnostic only -- the rule is 100% coverage and it is not relaxed here. "
        "research.md R1 permits a derived coverage fraction only if it is derived, committed and "
        "justified BEFORE scoring, never tried until something passes.)"
    )


def pop_opt(argv, name, default=None):
    """Remove `--name VALUE` from argv in place and return VALUE, else default."""
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def _write(out_path, lines):
    if not out_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _commit():
    import subprocess

    r = subprocess.run(
        ["git", "-C", _ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, errors="replace"
    )
    return r.stdout.strip() or "UNKNOWN"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    blimp = pop_opt(argv, "--blimp")
    out_path = pop_opt(argv, "--out")
    limit = pop_opt(argv, "--limit")
    limit = int(limit) if limit else None

    names, pairs = load_universe(root=blimp, limit=limit)
    if not names:
        where = blimp or os.path.join(_ROOT, "data", "lang", "blimp")
        print(f"SKIP lang_coverage: no BLiMP paradigms under {where}")
        print(
            "  download: https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip"
            " -> data/lang/blimp/"
        )
        # A8: the skip is recorded such that no downstream consumer can read it as a passed A/B.
        _write(
            out_path,
            [
                "# lang_coverage record -- machine-written by tools/lang_coverage.py",
                "status: SKIP",
                "reason: no BLiMP data on disk",
                "pair_ids: NONE",
                "NOT A FROZEN PAIR SET. No arm may be scored against this file, and this file is",
                "not an A/B result. A skip is never a pass.",
            ],
        )
        return 0

    triples, _bundled = load_kb()
    lexicon = exposure_lexicon(triples)
    ids = restrict(lexicon, pairs)
    in_par = sorted({pid.rsplit(":", 1)[0] for pid in ids})

    out = []
    out.append("# lang_coverage record -- machine-written by tools/lang_coverage.py")
    out.append("# specs/005-language-organ, contract A. Frozen BEFORE any arm is scored (FR-023).")
    out.append("")
    out.append(f"commit: {_commit()}")
    out.append(f"blimp_root: {blimp or os.path.join('data', 'lang', 'blimp')}")
    out.append(f"limit_per_paradigm: {limit if limit else 'ALL'}")
    out.append("")
    out.append("rule (verbatim):")
    out.append("  " + RULE)
    out.append("")
    out.append("lexicon_source:")
    out.append("  " + LEXICON_SOURCE)
    out.append(f"exposure_triples: {len(triples)}")
    out.append(f"exposure_lexicon_size: {len(lexicon)} words")
    out.append("")
    out.append(f"paradigms with at least one covered pair: {len(in_par)} of {N_PARADIGMS_TOTAL}")
    out.append(f"covered pairs: {len(ids)} of {len(pairs)}")
    out.append(
        f"floors: >= {FLOOR_PARADIGMS} of {N_PARADIGMS_TOTAL} paradigms, >= {FLOOR_PAIRS} pairs"
    )
    out.append("")
    out.append("admission controls, on the RESTRICTED set, this invocation (A6):")
    restricted_pairs = [p for p in pairs if p[0] in set(ids)]
    controls, disqualified = run_controls(restricted_pairs, out)
    out.append("")
    instrument_check(pairs, out)
    out.append("")
    out.append("diagnostics (decide nothing):")
    diagnostics(lexicon, pairs, out)
    out.append("")

    floors_ok = len(in_par) >= FLOOR_PARADIGMS and len(ids) >= FLOOR_PAIRS
    if not floors_ok or disqualified:
        reasons = []
        if len(in_par) < FLOOR_PARADIGMS:
            reasons.append(
                f"paradigm floor missed: {len(in_par)} of {N_PARADIGMS_TOTAL} covered, "
                f"floor is {FLOOR_PARADIGMS}"
            )
        if len(ids) < FLOOR_PAIRS:
            reasons.append(f"pair floor missed: {len(ids)} covered pairs, floor is {FLOOR_PAIRS}")
        if disqualified:
            reasons.append("an admission control passed where it must fail, or was not computable")
        out.append("outcome: NOT-P2-ELIGIBLE")
        for r in reasons:
            out.append(f"  reason: {r}")
        out.append("pair_ids: NONE -- no set was frozen, so no arm may be scored")
        out.append("")
        out.append(
            "This is the tool doing its job, not failing. research.md R1: widening the exposure "
            "corpus in response to this exit is choosing the pair set to get a verdict and is "
            "forbidden. The permitted outcome is: the swap is not P2-eligible, recorded, feature "
            "stops."
        )
        print("\n".join(out))
        _write(out_path, out)
        return 2

    out.append("outcome: FROZEN")
    out.append(f"pair_ids: {len(ids)} entries follow, one per line, sorted")
    out.extend("  " + pid for pid in sorted(ids))
    _ = controls
    print("\n".join(out))
    _write(out_path, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
