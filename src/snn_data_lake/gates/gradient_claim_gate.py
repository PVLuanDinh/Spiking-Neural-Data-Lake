"""Gradient-claim census -- DOES THIS RULE OWE A COSINE AT ALL?

The e-prop invariant carries an admission bar (`cosine > 0.5` against an independent BPTT gradient).
That bar is a PIVOT rule: it governs work that was about to use backprop. It is NOT a requirement on
every learning rule -- roughly 22 local rules ship here (R-STDP, BCM, Oja/Sanger, Vogels iSTDP,
Benna-Fusi, ReSuMe, tempotron, ...) and none carries a gradient cosine. None is in violation. Until
this module existed, nobody could answer "does this rule owe a cosine?" without re-reading the module.

# distinct-from: gates/eprop_transport_gate.py (merge: it ADOPTS this module's registry).
#   Same TECHNIQUE (an AST pass over src/), different QUESTION -- it asks "is there a rule NOBODY
#   CHECKED?" (a transposed weight reaching a weight update). This asks "does this rule OWE a
#   cosine?". The tree already accepted that test for its own siblings at eprop_transport_gate.py:63-77.
# distinct-from: gates/dormancy_gate.py -- asks "is this MODULE reachable from a live path?".
#   Reachability is not a claim about learning; a fully reachable module may owe nothing.
# distinct-from: gates/method_dormancy_gate.py -- asks "does this learning METHOD have a live
#   call site?". A method can be called every tick and still make no gradient claim.

TWO HALVES, SPLIT BY COST.

  * The STATIC half asserts in CI. It is `ast`-only: it never imports the module it classifies,
    because ~25.7% of the ungoverned subtree does not import in the gate environment and a
    classifier that skips what it cannot import silently UNDER-COUNTS.
  * The NUMERIC half (`--measure`) never asserts in CI. CI and the local gate run different torch
    builds, and this project has already recorded that a build-sensitive numeric assertion can be
    green in one and red in the other.

CENSUS SCOPE AND ASSERT SCOPE ARE DIFFERENT, AND THE SPLIT IS DELIBERATE (spec FR-001a).
The census walks the WHOLE package. The rule-quality bounds assert on the GOVERNED tier only;
ungoverned rules are counted and REPORTED. That is the verdict contract `eprop_transport_gate`
already ships, reused rather than reinvented: `reasoning/` alone is ~108 modules against ~22
governed local rules, so asserting over both would gate this work on remediating a subtree it does
not own. TWO stated deviations (2026-08-27, amended into FR-001a): PHANTOM_DECLARATION and the
unclassified bound assert ALL-scope -- census-INTEGRITY bounds, and every bearing row, governed
or not, flows into tools/rule_census.json (see evaluate()).

THE DECLARATION IS AUTHORITATIVE; THE INFERENCE IS A RECALL NET.
Structural inference measured ~31% error in BOTH directions, with false negatives on exactly the
load-bearing cases (`mlp_q`, `jepa_world_model`, `gf.MLP`, `predictive_coding`), because this tree
hand-rolls backprop (no token to key on), keeps P8 controls beside the rules they control, and
computes the gradient in a different function from the one that applies it. So `infer()` may answer
only CLAIM or UNKNOWN -- it is structurally incapable of returning NO-CLAIM. A confident negative from
a signal with measured 31% error is the dangerous verdict Standing Invariant 3 forbids.

BASELINE HOME, decided by measurement rather than by precedent-quoting: `tools/` is at ZERO p4c
ratchet headroom (30/30) for *.py files, but the p4c corpus is
`git ls-files --cached --others --exclude-standard -- *.py` (p4c_ratchet.py:114, pathspec :171), so a
.json there costs the zone nothing -- proven by `tools/p4c_ungoverned_baseline.json` sitting in that
zone today while it reads 30/30 green. KNOWN LIMIT, stated rather than discovered: under an installed
wheel there is no repo root above the package, so `_repo_root()` returns None and the baseline is a
source-tree artifact only -- the walk then reports an EMPTY census, which the LIVENESS bound turns
into a loud failure rather than a silent pass.

THE DELETE-AND-RE-DERIVE SEAM, and what closing it can and cannot buy.
`--update-baseline` refuses a rise by comparing against the recorded counts -- so DELETING
`tools/gradient_claim_baseline.json` made that refusal vacuous: `load_baseline()` returned None and
nothing was compared. `tools/p4c_ratchet.py` still has that hole and names it a deliberate escape
hatch ("a hand-edit of the JSON inside a reviewed commit"), i.e. a SOCIAL guard.

Here it is mechanized instead: **creating a baseline where none exists REFUSES unless
`--bootstrap "<reason>"` is given, and the reason is written into the file.** Nothing short of signing
can PREVENT a determined bypass, and this does not claim to -- what it removes is the SILENT one. A
reader seeing the counts jump now finds either a stated cause or a conspicuously missing one, which
turns an invisible act into an attributable one. The reason also SURVIVES later ratchet-downs, so it
does not evaporate on the next legitimate write.

Exercised once on purpose, 2026-08-17: 41/100 -> 48/102, because the CENSUS WIDENED -- the enumeration
had lost `plasticity/eprop_kernel.py` entirely, all 17 of its defs invisible -- not because the tree
regressed. That reason is in the file today.

Run:
    python -m snn_data_lake.gates.gradient_claim_gate                  # report + ASSERT (the run_selfchecks path)
    python -m snn_data_lake.gates.gradient_claim_gate --assert         # the CI gate
    python -m snn_data_lake.gates.gradient_claim_gate --self-check     # verify the verifier
    python -m snn_data_lake.gates.gradient_claim_gate --update-baseline
"""

import ast
import json
import math
import os
import random
import sys
import time

# THE safety envelope. Imported rather than retyped -- see the `_SAFETY_ENVELOPE` note below.
from snn_data_lake.gates.safety_envelope import ENVELOPE as _SAFETY_ENVELOPE

_HERE = os.path.abspath(__file__)
_PKG_DIR = os.path.dirname(_HERE)
_PKG = "snn_data_lake"


def _repo_root():
    """REUSES `substrate/self_mod_harness._repo_root` -- no fourth resolver is defined here.

    This file first hand-rolled a four-level counted climb copied from `doc_budget_gate.py:49-50`.
    That was the wrong pattern to copy, and `readout/laminar_local_readout._repo_root` already records
    why: a counted depth "encodes the module's then-current nesting", so moving the file one level
    silently resolves to the WRONG directory -- there it made a gate go green while testing synthetic
    data instead of the real pack. A gate that stops seeing what it checks and still exits 0 is the
    fail-green shape this project treats as a defect.

    The rewrite then hand-rolled an ANCHORED climb instead, which was still a second mechanism for
    one job (P2). Rejected on a MEASUREMENT rather than on the dependency-count argument P7 forbids:
    the objection raised against reuse was "it drags heavy imports into a gate that must run in the
    stdlib tier", and that was measured FALSE -- `self_mod_harness` imports in 47 ms / 111 modules
    with `torch` NOT loaded, against `dormancy_gate`'s 25 ms / 58. It does not break the stdlib tier.

    Imported LAZILY, matching how `census()` defers its `dormancy_gate` import, so module-top stays
    light. Returns None when the markers are absent (an installed wheel has no `run_selfchecks.py`),
    which callers MUST treat as "no repo" rather than as a path.
    """
    try:
        from snn_data_lake.infra.self_mod_harness import _repo_root as _shared

        root = _shared()
    except Exception:
        return None
    return root if root and os.path.exists(os.path.join(root, "src", _PKG)) else None


BASELINE_NAME = os.path.join("tools", "gradient_claim_baseline.json")

#: The declaration a rule module carries. Module-scope, `ast.literal_eval`-able, NEVER imported.
#: Keyed per QUALIFIED symbol because `plasticity/eprop_kernel.py` holds a rule, its BPTT control and a
#: surrogate autograd.Function in ONE file -- a per-module verdict cannot describe it.
DECL_NAME = "GRADIENT_CLAIM"  # FR-002d: module-scope, not a method, no LEARN_VERBS substring

#: FIVE dispositions. Two are insufficient: `reasoning/gf.py::MLP` is an owner-EXEMPT CONTROL, and a
#: two-value classifier condemns it. FOUR were insufficient too, and that was found by USE rather
#: than by design -- see NOT_A_RULE below.
LOCAL = "LOCAL"
APPROXIMATES_GRADIENT = "APPROXIMATES_GRADIENT"
EXACT_GRADIENT = "EXACT_GRADIENT"
CONTROL = "CONTROL"
#: ADDED after the first real declaration pass, because five enumerated symbols could not be
#: described truthfully by any of the four above -- none of them is a learning rule at all:
#:   plasticity/eprop_kernel::_first_layer_grad   a DIAGNOSTIC returning summed |grad|
#:   reasoning/gf::OpenGf.solve_unbounded     iterative-deepening SEARCH
#:   reasoning/gf::OpenGf._learn              promotes a solved rule to a named primitive
#:   reasoning/gf::ActiveLearner.learn        grows a SYMBOLIC GRAMMAR; no weight is touched
#:   reasoning/eprop::_batch_grad             a COMBINATOR summing whatever grad_fn it is handed,
#:                                            so any fixed claim would be false half the time
#: The enumeration deliberately over-captures for recall, so this is a predictable consequence of
#: that choice rather than a surprise. Without this value the only options were to label a search
#: "LOCAL" (false) or leave it UNDETERMINABLE (true, but it conflates "I cannot decide" with "there
#: is nothing here to decide", and UNDETERMINABLE is ratcheted DOWN as debt it is not).
#:
#: TWO GUARDS, because a fifth value is a drawer waiting to happen -- which is exactly what FR-016a
#: ratchets UNDETERMINABLE for:
#:   1. Reachable ONLY by an explicit declaration carrying a reason. `infer()` cannot produce it and
#:      no default resolves to it, so the classifier can sweep nothing in here.
#:   2. Its count is REPORTED as a precision meter for the enumeration. A rising NOT_A_RULE does not
#:      mean the tree got worse -- it means the heuristic is capturing more non-rules and wants
#:      tightening. Stated in the report so the number cannot be quietly re-purposed.
NOT_A_RULE = "NOT_A_RULE"

#: ADDED for a class of ONE, and named with `docs/POLICIES.md`'s own word rather than an invented
#: synonym. `growth/ng_snn.py::fit` is a GENUINE learning rule that owes no cosine, and `LOCAL`
#: is literally the wrong label for it: POLICIES.md adjudicates it as *"not a violation. Greedy
#: matching pursuit with an analytical beta per block is a DIRECT SOLVE: no gradient is propagated
#: through a network and no weight is transported. It IS NON-LOCAL, and the policy should say so
#: DISTINCTLY RATHER THAN LUMPING IT IN -- 'non-local' and 'backprop' are different claims, and only
#: the second is what this rule forbids."* Declaring it LOCAL would lump in exactly the way that
#: adjudication forbids; leaving it undeclared would report a settled question as unresolved.
#:
#: SCOPED DELIBERATELY NARROW. It is NOT a general "non-local" bucket. `readout/predictive_coding` was
#: checked against this value and stays LOCAL: POLICIES.md:1766-1768 records that its weight update is
#: *"a local Hebbian outer product ... No transposed weight appears in the learning rule at all"* --
#: its `w1.T` is INFERENCE settling an activation, not the rule. The "same disposition as ng_snn::fit"
#: line there means "owes no cosine", not "non-local".
DIRECT_SOLVE = "DIRECT_SOLVE"

DISPOSITIONS = (LOCAL, APPROXIMATES_GRADIENT, EXACT_GRADIENT, CONTROL, NOT_A_RULE, DIRECT_SOLVE)

#: Resolved states. `UNDETERMINABLE` COUNTS AS CLASSIFIED (FR-016): classified must not be read as
#: decided. Without it, "zero unclassified, always" forces an undecidable rule out as a confident
#: NO-CLAIM -- measurably wrong on at least five governed modules, all load-bearing.
CLASSIFIED_LOCAL = "CLASSIFIED_LOCAL"
CLASSIFIED_CLAIMANT = "CLASSIFIED_CLAIMANT"
CLASSIFIED_REFERENCE = "CLASSIFIED_REFERENCE"
CLASSIFIED_CONTROL = "CLASSIFIED_CONTROL"
CLASSIFIED_NOT_A_RULE = "CLASSIFIED_NOT_A_RULE"
CLASSIFIED_DIRECT_SOLVE = "CLASSIFIED_DIRECT_SOLVE"
MISSING_DECLARATION = "MISSING_DECLARATION"
UNDETERMINABLE = "UNDETERMINABLE"
#: A declared qualname NOTHING in the parsed tree provides (owner task, 2026-08-27; the
#: T050 incident: a GRADIENT_CLAIM keyed "EligibilityAlgebra.evaluate" -- the method lives on
#: AuthoredRule -- produced a governed CLASSIFIED_NOT_A_RULE row for a symbol with no code, while the
#: real method carried no row; a typo'd declaration was indistinguishable from coverage.
#: ABSOLUTE bound 0, asserted ALL-scope (the one stated FR-001a deviation -- see evaluate()),
#: not a ratchet: zero live phantoms existed when this landed (measured at commit: 172
#: declarations across 419 trees, every key resolving against _symbol_surface) and a phantom
#: is always a defect, never legacy debt. NO envelope carve-out (a phantom can only enter the
#: envelope through an already-approved edit; see the census() comment). TWO DISCLOSED
#: REMAINDERS, not closed here, both the same defect family and both chipped rather than
#: silently owned: (1) a declaration whose DISPOSITION is typo'd ("LOCAl") is dropped whole by
#: read_declaration, so a phantom KEY under a typo'd disposition yields NO row at all; (2) a
#: NON-LITERAL declaration (e.g. GRADIENT_CLAIM = dict(...) or item assignment) fails
#: ast.literal_eval and silently drops the module's ENTIRE declaration surface -- panel-measured
#: taking a real module's genuine rows with it.
PHANTOM_DECLARATION = "PHANTOM_DECLARATION"

_STATE_OF = {
    LOCAL: CLASSIFIED_LOCAL,
    APPROXIMATES_GRADIENT: CLASSIFIED_CLAIMANT,
    EXACT_GRADIENT: CLASSIFIED_REFERENCE,
    CONTROL: CLASSIFIED_CONTROL,
    NOT_A_RULE: CLASSIFIED_NOT_A_RULE,
    DIRECT_SOLVE: CLASSIFIED_DIRECT_SOLVE,
}

#: Every state a census row may legally carry. `unclassified` is counted against THIS set,
#: never against the accumulating counts dict -- the counting loop inserts every SEEN state
#: as a key, so an `in out` membership test is vacuously true and the ABSOLUTE bound could
#: never fire (measured 2026-08-27: a poisoned live census read unclassified=0, evaluate()=[],
#: zero delta vs control). Derived from the same table resolve() indexes, so the bound it
#: feeds is a source-edit tripwire, never a tree bound. Spec-side record of this
#: enumeration: specs/020-eprop-cosine-scope/data-model.md "ClassificationState".
_DECLARED_STATES = frozenset(_STATE_OF.values()) | {
    MISSING_DECLARATION,
    UNDETERMINABLE,
    PHANTOM_DECLARATION,
}

#: The five brainstem files. They are ENUMERATED and CLASSIFIED like everything else, but a rule in
#: them is NEVER reported as MISSING_DECLARATION, because that finding's only remedy is to edit the
#: file -- and the safety envelope is modified only with explicit approval (FR-010, Standing
#: Invariant 7). Found by measurement, not by review: the first census enumerated
#: `brainstem/reflex.py::Reflex.step` as a learning rule, so a gate demanding declarations would have
#: demanded one inside the E-STOP path.
#:
#: IMPORTED at the top of this module, not retyped here (P2, 2026-08-18) -- this was the THIRD
#: hand-maintained copy, and the P2 NOTE further down recorded it as latent debt this feature's own
#: FR-010 made it the wrong lane to fix. The name is kept as an alias of
#: `gates/safety_envelope.ENVELOPE`, so both use sites here and the two tests naming it are
#: unchanged. This gate adds NOTHING to the envelope: it protects the five and only the five.


#: ADDED to the borrowed learn-verbs, not a competing copy of them. `method_dormancy_gate` asks "does
#: this learning METHOD have a live caller?", so its list keys on names that MUTATE. This gate asks
#: "does this rule OWE a cosine?", and the answer lives partly in functions that COMPUTE a gradient
#: and return it -- which mutate nothing and carry no learn verb.
#:
#: MEASURED, and the measurement is the whole reason this constant exists: dropping `grad` to cure an
#: over-count made `plasticity/eprop_kernel.py` -- the e-prop REFERENCE GRADIENT, the module FR-011
#: reconciles -- vanish from the census entirely. All 17 of its defs failed both signals, because
#: `bptt_grads` / `_TinyNet.eprop_grads` / `_first_layer_grad` / `_apply_grads` RETURN gradients
#: rather than augmented-assigning a weight. Cost of restoring it, from the same sweep that condemned
#: `step`: `grad` is +9 governed rules, `step` was +68. Precision was tuned without measuring recall,
#: and the most load-bearing module in the feature is what it cost.
#: Verbs EXCLUDED from the borrowed LEARN_VERBS, on measured precision against the 149 hand-reviewed
#: declarations -- the labelled sample research.md R1 asked for and recorded as not done. Scored over
#: 157 labelled symbols, treating "declared NOT_A_RULE" as a false positive of the walk:
#:
#:     verb          true-rule   NOT_A_RULE   precision
#:     train              36           2         95%
#:     fit                 4           1         80%
#:     grad                8           4         67%
#:     learn              20          14         59%
#:     update              7          11         39%
#:     consolidate         3           6         33%
#:     adapt               1           2         33%
#:     calibrate           0           5          0%   <- dropped
#:     plast               0           3          0%   <- dropped
#:
#: Only the two that caught NOTHING real are dropped. `update` and `consolidate` score badly but each
#: catches real rules, and a missed rule is invisible forever while a false positive costs one
#: declaration -- so the walk stays recall-biased everywhere the evidence is not unanimous.
#:
#: HONEST SIZE OF THE WIN: this removes exactly ONE symbol today, because every other symbol those
#: verbs caught is now DECLARED and a declaration always enumerates. The benefit is PROSPECTIVE --
#: future code named *calibrate* / *plast* will not be enumerated. Verified before landing: 0 real
#: rules lost, and the recall control in tests/test_gradient_claim_gate.py still passes.
#:
#: NOT done by editing method_dormancy_gate.LEARN_VERBS: that list is the sibling gate own definition
#: of a learning method and mutating it would change ITS behaviour. This is a local filter over a
#: reused list, which keeps one definition of the verbs (P2) and one owner of it.
_DROPPED_VERBS = ("calibrate", "plast")

_GRADIENT_VERBS = ("grad",)


def _rule_verbs():
    """REUSES `method_dormancy_gate.LEARN_VERBS` -- this tree's existing definition of a learning verb.

    A parallel verb list was written here first and is retired: two lists deciding "is this a learning
    method?" is the additive shape P2 forbids, and they would drift the first time one was tuned.
    The retired copy also carried `step`, `grad` and `calibrate`, which MEASURED badly -- `step`
    alone pulled the governed count 174 -> 242 by enumerating every neuron's membrane advance,
    including `brainstem/reflex.py::Reflex.step`.
    """
    try:
        from snn_data_lake.gates.method_dormancy_gate import LEARN_VERBS

        base = tuple(LEARN_VERBS)
    except Exception:  # pragma: no cover - the gate must still report without the sibling
        base = ("learn", "update", "train", "adapt", "consolidate", "reinforce", "fit", "plast")
    return tuple(v for v in base if v not in _DROPPED_VERBS) + _GRADIENT_VERBS


#: Tokens that make the INFERENCE answer CLAIM. Never NO-CLAIM -- see the module docstring.
_CLAIM_TOKENS = (
    "gradient",
    "backprop",
    "bptt",
    "descent",
    "surrogate",
    "eprop",
    "e-prop",
    "eligibility",
)


def source_dir(root):
    """The package directory under `root`. Mirrors `tools/p6_gate.py:384`."""
    return os.path.join(root, "src", _PKG)


def rel(path, root):
    """Repo-relative POSIX path under src/snn_data_lake/, e.g. 'engine/brain.py'."""
    return os.path.relpath(path, source_dir(root)).replace(os.sep, "/")


def modname(relpath):
    """'engine/brain.py' -> 'snn_data_lake.engine.brain'."""
    return f"{_PKG}." + relpath[: -len(".py")].replace("/", ".")


def walk(root=None, overlay=None):
    """(trees, parse_failures, n_found) -- the raw AST census.

    `overlay` maps a repo-relative path under src/snn_data_lake ('cortex/mutant.py') to source text,
    or to None to DELETE that file from the walk. It is how `--self-check` constructs a RED against
    the REAL tree while touching nothing on disk. Ported from `tools/p6_gate.py:612`, the only gate
    whose walk takes one -- which is exactly what "prove the criterion can discriminate before it
    binds" needs.
    """
    root = _repo_root() if root is None else root
    if root is None:
        # No repo (an installed wheel). Report an empty walk rather than guessing a path -- the
        # LIVENESS bound below is what turns that into a loud failure instead of a silent pass.
        return {}, [], 0
    overlay = dict(overlay or {})
    found, trees, failures = [], {}, []

    for dirpath, dirnames, filenames in os.walk(source_dir(root)):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                found.append(os.path.join(dirpath, fn))

    for path in sorted(found):
        relpath = rel(path, root)
        if relpath in overlay:
            continue  # the overlay decides this file's fate below
        try:
            with open(path, encoding="utf-8") as fh:
                trees[relpath] = ast.parse(fh.read(), filename=path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append((relpath, type(exc).__name__))

    n_found = len(found)
    for relpath in sorted(overlay):
        src = overlay[relpath]
        on_disk = os.path.exists(os.path.join(source_dir(root), relpath.replace("/", os.sep)))
        if src is None:
            if on_disk:
                n_found -= 1  # a deletion the walk must SEE, so the liveness floor can fire
            continue
        if not on_disk:
            n_found += 1
        try:
            trees[relpath] = ast.parse(src, filename=relpath)
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append((relpath, type(exc).__name__))

    return trees, sorted(failures), n_found


def read_declaration(tree):
    """The module's GRADIENT_CLAIM dict, or {} -- read by `ast.literal_eval`, never imported.

    Implements FR-002c: a module-scope constant readable WITHOUT importing the module.

    Returns {qualified_symbol: (disposition, reason)}. A malformed entry is DROPPED rather than
    guessed at, so a typo surfaces as a missing declaration (a reportable finding) instead of as a
    silently wrong disposition.
    """
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == DECL_NAME for t in node.targets):
            continue
        try:
            raw = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return {}
        if not isinstance(raw, dict):
            return {}
        for sym, val in raw.items():
            if not isinstance(sym, str) or not isinstance(val, (tuple, list)) or len(val) != 2:
                continue
            disp, reason = val
            if disp in DISPOSITIONS:
                out[sym] = (disp, str(reason))
    return out


def _qualnames(tree):
    """[(qualname, node)] for every def in the module, methods qualified by their class.

    Free functions matter: `reasoning/eprop.py::_batch_grad` is a rule and is not on a class, which
    is why a class-attribute declaration scheme lost during Phase 0.
    """
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((f"{node.name}.{sub.name}", sub))
    return out


def _mutates_weightlike(node):
    """True if the body performs an in-place update to a weight-shaped target.

    `self.W[i] -= lr * dW` and `self.w1 -= lr * gw1` are the shapes that make a function a RULE
    rather than a reader. Keyed on the AugAssign, not on the name, so a rule whose weights are called
    something unexpected is still enumerated.
    """
    for sub in ast.walk(node):
        if not isinstance(sub, ast.AugAssign):
            continue
        tgt = sub.target
        while isinstance(tgt, ast.Subscript):
            tgt = tgt.value
        name = tgt.attr if isinstance(tgt, ast.Attribute) else getattr(tgt, "id", "")
        if name and _is_weight_name(name):
            return True
    return False


#: Weight-ish names, matched WHOLE rather than by prefix. A prefix match on ("w", "b", ...) was
#: MEASURED catching `bound_areas`, `_spectral_bound`, `buf` and `best` -- 50 governed symbols whose
#: only qualification was starting with the letter b. `W2`/`Wrec`/`Win`/`Wout` are included
#: deliberately: those are `reasoning/eprop.py`'s actual weights, and the sibling transport gate
#: records that missing them was its own most expensive defect.
_WEIGHT_NAMES = frozenset(
    ("w", "b", "weight", "weights", "bias", "biases", "theta", "syn", "synapse", "synapses")
)


def _is_weight_name(name):
    """Whether an assignment target denotes a weight/bias tensor, not merely a name starting with w or b.

    # distinct-from: brainstem/so101_arm.py::trajectory_breach -- a PER-JOINT DROOP ENVELOPE
    # comparison on arm pose; matched only on the shared docstring opener "True when ...". Read
    # before adding this marker: it mentions droop 13 times and weights/bias once, incidentally, and
    # decides nothing about names. (That file is safety envelope and was READ ONLY, never edited.)
    """
    stem = name.lower().lstrip("_")
    if stem in _WEIGHT_NAMES:
        return True
    # W2 / Wrec / Win / Wout / w1 / b2 -- a weight letter followed only by digits or a short suffix.
    head = stem[0] if stem else ""
    tail = stem[1:]
    return (
        head in ("w", "b")
        and tail != ""
        and (tail.isdigit() or tail in ("rec", "in", "out", "hh", "ih"))
    )


def enumerate_rules(tree, declared):
    """Qualnames in this module that are candidate learning rules.

    HEURISTIC, and its recall is UNPROVEN -- a hand-named control of 32 canonical rules hit 28
    during Phase 0. Recorded here rather than in a commit message because every count this module
    prints is a lower bound under one definition and an upper bound under another. A declared symbol
    is ALWAYS enumerated, so a rule this heuristic misses can be pulled in by declaring it.
    """
    names = set(declared)
    verbs = _rule_verbs()
    for qual, node in _qualnames(tree):
        short = qual.split(".")[-1].lower()
        if short.startswith("__"):
            continue
        if any(v in short for v in verbs) or _mutates_weightlike(node):
            names.add(qual)
    return sorted(names)


def infer(tree, qual):
    """CLAIM or UNKNOWN. NEVER NO-CLAIM -- see FR-002b and the module docstring.

    Its only job is to catch a rule carrying no declaration. It may not overrule one that has it.
    The intra-module call graph is the highest-value signal: in every claimant found during Phase 0
    the gradient is computed in one function and APPLIED in another (`grad_eprop`/`grad_bptt` ->
    `Net.apply`, `eprop_grads` -> `_apply_grads`), so neither a mutation-site scan nor a claim-site
    scan alone sees a rule.
    """
    nodes = dict(_qualnames(tree))
    node = nodes.get(qual)
    if node is None:
        return "UNKNOWN"

    def _claims(n):
        doc = (ast.get_docstring(n) or "").lower()
        if any(t in doc for t in _CLAIM_TOKENS):
            return True
        if any(t in n.name.lower() for t in _CLAIM_TOKENS):
            return True
        for sub in ast.walk(n):
            if isinstance(sub, ast.Attribute) and sub.attr in ("backward", "T"):
                return True
            if isinstance(sub, ast.Call):
                fn = sub.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if nm and any(t in nm.lower() for t in _CLAIM_TOKENS):
                    return True
        return False

    if _claims(node):
        return "CLAIM"

    # One level of the intra-module call graph: a rule that APPLIES what a claiming function computed
    # is itself a claimant, and this is the case a mutation-site scan cannot see.
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        callee = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        for other_qual, other in nodes.items():
            if other_qual.split(".")[-1] == callee and other is not node and _claims(other):
                return "CLAIM"
    return "UNKNOWN"


def resolve(declared_disp, inferred):  # FR-002a: TWO signals -- declaration + inference
    """The single ClassificationState for one rule. Declaration AUTHORITATIVE."""
    if declared_disp is not None:
        return _STATE_OF[declared_disp]
    if inferred == "CLAIM":
        return MISSING_DECLARATION  # a finding, never a default classification (FR-002b)
    return UNDETERMINABLE  # classified, NOT decided (FR-016)


# distinct-from: gates/selfcheck_cache.py -- its AST walk HASHES module structure for
#   carry-forward verification; this one collects a NAME set for existence checks. Read.
# distinct-from: substrate/home.py (sidecar census) -- a census of MODULES that can write
#   the T037a sidecar; unrelated to symbol names. Read.
# distinct-from: cortex/metacognitive_monitor.py -- reliability signals; term overlap on
#   'ground truth' only. Read.
def _symbol_surface(tree):
    """Every name the module PROVIDES, at any nesting -- the phantom check's ground truth.

    _qualnames stays the ENUMERATION scan (module-body and direct class-body defs -- the
    grain rules live at); this is the broader EXISTENCE surface. What registers:
      (a) every def/async def at any depth, qualified by its enclosing CLASS path.
          Control-flow and function scopes are TRANSPARENT for DEFS ONLY (a def under
          `if`/`try`/`with`, or a factory closure, registers) -- panel-measured refactor
          shapes with zero false-red cost;
      (b) ASSIGNMENT targets at MODULE level and CLASS body only, and only when the
          assigned value is callable-shaped (a lambda, a call like staticmethod(_impl)/
          partial(...), or a bare-name alias). NEVER function-local assignments -- the
          first draft collected those transparently and the honesty panel reproduced the
          T050 incident SLIPPING BACK THROUGH: a method-local `evaluate = x + 1` inside
          the WRONG class validated the wrong-class key. A local variable is not a
          provided symbol. Literals are excluded for the same reason a float is not a
          learning rule.
    Calibration, measured at landing: every one of the 172 live declarations resolves
    against the STRICT shallow _qualnames set, so this surface's extra breadth has zero
    live cost and exists purely to keep plausible refactors (conditional defs, nested
    classes, callable rebinds) from red-flagging an honest declaration. DISCLOSED
    RESIDUAL: runtime member injection (setattr(Cls, name, fn), decorator-registered
    methods) is invisible to any static walk -- a declaration keying one would false-red;
    no live declaration does, and the remedy if one appears is declaring at the
    statically-achievable grain, said here rather than discovered."""
    surface = set()

    def _callable_ish(v):
        return isinstance(v, (ast.Lambda, ast.Call)) or isinstance(v, ast.Name)

    def visit(node, class_path, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                surface.add(".".join(class_path + [child.name]))
                visit(child, class_path + [child.name], in_function)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                surface.add(".".join(class_path + [child.name]))
                visit(child, class_path, True)  # defs transparent; assigns below are NOT
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                value = child.value
                if not in_function and value is not None and _callable_ish(value):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for t in targets:
                        if isinstance(t, ast.Name):
                            surface.add(".".join(class_path + [t.id]))
                        elif isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                            surface.add(f"{t.value.id}.{t.attr}")
                visit(child, class_path, in_function)
            else:
                visit(child, class_path, in_function)

    visit(tree, [], False)
    return surface


def census(root=None, overlay=None, brain_modules=None):
    """(rules, failures, n_found) where each rule is a dict.

    Walks the WHOLE package; `governed` marks which half the bounds may assert on (FR-001a). The
    governed predicate is `dormancy_gate.BRAIN_MODULES` -- the same set `eprop_transport_gate` uses,
    imported lazily for the same reason, so there is exactly one definition of "governed" in the tree.
    """
    if brain_modules is None:
        try:
            from snn_data_lake.gates.dormancy_gate import BRAIN_MODULES

            brain_modules = BRAIN_MODULES
        except Exception:  # pragma: no cover - the gate must still report without the package
            brain_modules = set()

    trees, failures, n_found = walk(root, overlay)
    rules = []
    for relpath in sorted(trees):
        tree = trees[relpath]
        declared = read_declaration(tree)
        governed = modname(relpath) in brain_modules
        surface = _symbol_surface(tree)
        for qual in enumerate_rules(tree, declared):
            disp = declared.get(qual, (None, ""))[0]
            inferred = infer(tree, qual)
            state = resolve(disp, inferred)
            # PHANTOM: a declaration naming a symbol NOTHING in the parsed tree provides.
            # Checked AFTER resolve (a declaration is authoritative about DISPOSITION, but
            # only for a symbol that exists). The check runs against _symbol_surface, NOT
            # _qualnames: the panel measured ten truthful-declaration shapes the shallow
            # scan red-flagged (nested classes, conditional/try/with defs, closures,
            # lambda/staticmethod assignments) -- and a FALSE red on an honest declaration
            # is the direction that forces someone to weaken an ABSOLUTE bound. The surface
            # is deliberately GENEROUS: a phantom slipping (false negative) costs one
            # unverified row; a truthful declaration red-flagged costs the bound itself.
            # `main` gets no exemption: a real `def main` is on the surface; only a typo'd
            # key lands here.
            if disp is not None and qual not in surface:
                state = PHANTOM_DECLARATION
            # The safety envelope is classified like everything else but never yields a finding whose
            # ONLY remedy is editing an approval-gated file (FR-010, Standing Invariant 7). It falls
            # to UNDETERMINABLE -- honest about not being decided, and still counted, so the envelope
            # cannot vanish from the census. PHANTOM deliberately gets NO carve-out (the panel
            # measured the downgrade landing on the UNDETERMINABLE ratchet anyway -- renaming the
            # finding, not removing the obligation): the envelope carries zero declarations, so a
            # phantom can only ENTER it through an already-approved envelope edit, and fixing the
            # typo is part of that same approved act.
            if relpath in _SAFETY_ENVELOPE and state == MISSING_DECLARATION:
                state = UNDETERMINABLE
            rules.append(
                {
                    "qualname": f"{relpath}::{qual}",
                    "module_rel": relpath,
                    "governed": governed,
                    "declared": disp,
                    "inferred": inferred,
                    "state": state,
                }
            )
    return rules, failures, n_found


def counts(rules, governed_only=True):
    """Per-state counts. `governed_only` is the ASSERT scope; report mode calls both ways."""
    sel = [r for r in rules if r["governed"]] if governed_only else list(rules)
    # sorted() so key order is deterministic across processes (frozenset iteration order
    # varies with PYTHONHASHSEED; nothing serializes this dict today, so determinism of
    # the key order itself is the whole claim).
    out = {s: 0 for s in sorted(_DECLARED_STATES)}
    for r in sel:
        out[r["state"]] = out.get(r["state"], 0) + 1
    out["rules_seen"] = len(sel)
    # UNDETERMINABLE counts as CLASSIFIED, so unclassified is 0 by construction unless a state
    # escapes the declared table. Tested against _DECLARED_STATES, NEVER against `out` -- the
    # loop above inserts every seen state as a key, so an `in out` test can never fire (the
    # 2026-08-27 measured hole: poisoned census, unclassified=0, zero delta vs control).
    out["unclassified"] = sum(1 for r in sel if r["state"] not in _DECLARED_STATES)
    return out


def _baseline_path(root=None):
    """Absolute path of the ratchet baseline file, or None when there is no repo to write into.

    # distinct-from: growth/neurogenesis_engine.py::growth_state_bytes -- returns a GROWTH BLOB.
    # distinct-from: infra/device.py::crossover -- returns a CUDA crossover WIDTH.
    # distinct-from: thalamus/global_workspace.py::_band_ratio -- returns a RATIO in [0,1].
    # All three were flagged on the shared docstring phrase "or None when there is no ...", not on the
    # capability; each was READ before this marker was added, and none of the three calls os.path.join
    # even once (measured 0/0/0 against this file's 6). None resolves a filesystem path.
    """
    root = _repo_root() if root is None else root
    return None if root is None else os.path.join(root, BASELINE_NAME)


def load_baseline(root=None):
    """The committed baseline, or None.

    `utf-8-sig` is deliberate, not decoration: the documented escape hatch is a hand-edit in a
    reviewed commit, and a PowerShell hand-edit writes a BOM that would otherwise crash the gate.
    Same reasoning as `tools/p4c_ratchet.py:255-268`.
    """
    path = _baseline_path(root)
    if path is None or not os.path.exists(path):
        return None
    try:
        return json.loads(open(path, encoding="utf-8-sig").read())
    except (OSError, ValueError):
        return None


def ratchet_state(live, allowed):
    """SATISFIED when the bound rests at its floor, GREEN when it is above it, RED when it rose.

    A ratchet at zero and a ratchet nobody ever exercised both print green today, and a success state
    indistinguishable from a dead check is what Standing Invariant 5 calls worse than no gate. The
    floor is 0: a ratchet resting there can never fire again, and that is the state worth naming.
    """
    if live > allowed:
        return "RED"
    return "SATISFIED" if live == 0 else "GREEN"


def evaluate(c, baseline, n_found, c_all=None):
    """[] when every bound holds, else a list of human-readable failures. PURE -- no I/O.

    `c_all` (all-scope counts) widens exactly TWO bounds past FR-001a's governed scope, and
    both deviations are deliberate and stated: PHANTOM_DECLARATION and unclassified are
    CENSUS-INTEGRITY bounds, not rule-quality bounds -- declarations and states are
    tree-wide artifacts and every bearing row, governed or not, flows into
    tools/rule_census.json, so an ungoverned phantom or escaped state poisons a governed
    artifact. The panel measured the governed-only forms leaving 82 ungoverned rows (33%)
    outside both bounds and outside the printed number, while the self-check advertised
    the red as proven. When c_all is None each check falls back to c (the self-check's
    pure-composition arms pass both)."""
    base = (baseline or {}).get("counts", {})
    floor = (baseline or {}).get("rules_seen_floor", 0)
    out = []
    _unc = (c_all if c_all is not None else c)["unclassified"]
    if _unc != 0:
        out.append(
            f"unclassified={_unc} (ABSOLUTE bound is 0, ALL-scope -- a state outside the "
            "declared table is a census hole wherever it sits; extend _DECLARED_STATES "
            "only together with the state that joins the table)"
        )
    _ph = (c_all if c_all is not None else c).get(PHANTOM_DECLARATION, 0)
    if _ph != 0:
        out.append(
            f"{PHANTOM_DECLARATION}={_ph} (ABSOLUTE bound is 0, ALL-scope -- a declared "
            "qualname with no code is a typo wearing a disposition; fix the declaration's "
            "key, never weaken this bound)"
        )
    for key in (MISSING_DECLARATION, UNDETERMINABLE):
        allowed = base.get(key)
        if allowed is not None and c[key] > allowed:
            out.append(f"{key}={c[key]} > baseline {allowed} (RATCHET may only fall)")
    if c["rules_seen"] < floor:
        out.append(
            f"rules_seen={c['rules_seen']} < floor {floor} (LIVENESS -- a dead walk reports zero of everything)"
        )
    if n_found <= 0:
        out.append("the walk found no files at all (LIVENESS)")
    return out


def _report(rules, failures, n_found, baseline, elapsed):
    gov = counts(rules, governed_only=True)
    allr = counts(rules, governed_only=False)
    base = (baseline or {}).get("counts", {})
    print("GRADIENT-CLAIM CENSUS")
    print(f"  files parsed         : {n_found}")
    print(f"  rules seen  GOVERNED : {gov['rules_seen']}        <- the ASSERT scope")
    print(
        f"  rules seen  ungov.   : {allr['rules_seen'] - gov['rules_seen']}        <- reported, never asserted"
    )
    print(f"  rules seen  TOTAL    : {allr['rules_seen']}")
    for key, label in (
        (CLASSIFIED_LOCAL, "classified local"),
        (CLASSIFIED_CLAIMANT, "classified claimant"),
        (CLASSIFIED_REFERENCE, "classified reference"),
        (CLASSIFIED_CONTROL, "classified control"),
        (CLASSIFIED_NOT_A_RULE, "classified NOT-A-RULE"),
        (CLASSIFIED_DIRECT_SOLVE, "classified DIRECT-SOLVE"),
    ):
        print(f"  {label:<21}: {gov[key]:<5} governed   {allr[key]:<5} all")
    if gov[CLASSIFIED_NOT_A_RULE] or allr[CLASSIFIED_NOT_A_RULE]:
        print("      ^ NOT-A-RULE measures ENUMERATION IMPRECISION, not tree health: it counts")
        print("        symbols the recall-biased walk captured that are not learning rules at all.")
        print(
            "        A RISING count means the heuristic wants tightening, never that code got worse."
        )
    for key, label in (
        (MISSING_DECLARATION, "MISSING DECLARATION"),
        (UNDETERMINABLE, "UNDETERMINABLE"),
    ):
        allowed = base.get(key)
        state = ratchet_state(gov[key], allowed) if allowed is not None else "no baseline"
        shown = "-" if allowed is None else allowed
        print(
            f"  {label:<21}: {gov[key]:<5} governed   {allr[key]:<5} all   (ratchet, baseline {shown}, {state})"
        )
    print(
        f"  PHANTOM DECLARATION  : {gov[PHANTOM_DECLARATION]:<5} governed   "
        f"{allr[PHANTOM_DECLARATION]:<5} all   (ABSOLUTE 0, asserted ALL-scope -- a "
        f"declared key with no code)"
    )
    print(
        f"  unclassified         : {gov['unclassified']:<5} governed   "
        f"{allr['unclassified']:<5} all   (ABSOLUTE 0, asserted ALL-scope -- a state "
        f"outside the declared table; must be 0)"
    )
    print(f"  parse failures       : {len(failures)}")
    for relpath, kind in failures:
        print(f"      {relpath}  ({kind})")
    # FR-013a: the cost is MEASURED and PRINTED, and no assertion reads it. Wall-clock on shared CI
    # runners is not deterministic, and a fixed second-count arbitrating the build is a P1 constant.
    print(f"  wall-clock           : {elapsed:.3f}s   (reported, NEVER asserted -- FR-013a)")
    print("  a green ratchet means NO WORSE. It never means the scope is correct.")
    return gov, allr


def _selfcheck():
    """Construct each RED against the REAL tree via the overlay. Verify the verifier.

    A self-check that only runs the happy path verifies nothing -- this repo has four gates that pass
    that way. Every red below is CONSTRUCTED and then confirmed to fire, and red 5 is a GREEN control
    proving the repaired shape is NOT flagged (else the checker cannot tell fixed from broken).
    """
    ok = True
    base_rules, base_fail, base_n = census()
    base_gov = counts(base_rules)
    print(
        f"[control ] unmutated tree -> {base_gov['rules_seen']} governed rules, {len(base_fail)} parse failures"
    )
    if base_gov["rules_seen"] <= 0:
        print("  FAIL   the control census is empty; every red below would be vacuous")
        return False

    # RED 1 -- a rule with no declaration whose inference says CLAIM becomes MISSING_DECLARATION.
    mutant = (
        "class Rule:\n"
        "    def learn(self, x):\n"
        '        """Approximates the gradient by backprop."""\n'
        "        self.W -= 0.1 * x\n"
    )
    r1, _, _ = census(overlay={"cortex/zz_probe_rule.py": mutant})
    hits = [r for r in r1 if r["module_rel"] == "cortex/zz_probe_rule.py"]
    fired = any(r["state"] == MISSING_DECLARATION for r in hits)
    print(
        f"  {'RED  ' if fired else 'FAIL '}  undeclared claimant -> MISSING_DECLARATION ({len(hits)} rule(s) seen)"
    )
    ok = ok and fired

    # RED 2 -- removing a declaration must NOT silently become LOCAL.
    declared = (
        f'{DECL_NAME} = {{"Rule.learn": ("{APPROXIMATES_GRADIENT}", "reason")}}\n'
        "class Rule:\n"
        "    def learn(self, x):\n"
        '        """Approximates the gradient."""\n'
        "        self.W -= 0.1 * x\n"
    )
    r2, _, _ = census(overlay={"cortex/zz_probe_rule.py": declared})
    got = [r["state"] for r in r2 if r["module_rel"] == "cortex/zz_probe_rule.py"]
    good = CLASSIFIED_CLAIMANT in got and CLASSIFIED_LOCAL not in got
    print(
        f"  {'RED  ' if good else 'FAIL '}  declared claimant stays CLAIMANT, never LOCAL -> {sorted(set(got))}"
    )
    ok = ok and good

    # RED 3 -- a deleted file must LOWER rules_seen so the liveness floor can fire. A dead walk
    # reports exactly zero unclassified and looks perfect; four gates in this tree fail that way.
    victim = max(
        ({r["module_rel"] for r in base_rules if r["governed"]} or {""}),
        key=lambda m: sum(1 for r in base_rules if r["module_rel"] == m),
    )
    r3, _, n3 = census(overlay={victim: None})
    fell = counts(r3)["rules_seen"] < base_gov["rules_seen"] and n3 < base_n
    print(f"  {'RED  ' if fell else 'FAIL '}  deleting {victim} lowers rules_seen and n_found")
    ok = ok and fell
    lively = evaluate(counts(r3), {"counts": {}, "rules_seen_floor": base_gov["rules_seen"]}, n3)
    print(f"  {'RED  ' if lively else 'FAIL '}  the liveness floor fires on that deletion")
    ok = ok and bool(lively)

    # RED 4 -- an unparseable module raises parse_failures, an ABSOLUTE bound.
    r4, f4, _ = census(overlay={"cortex/zz_probe_rule.py": "def broken(:\n"})
    rose = len(f4) > len(base_fail)
    print(
        f"  {'RED  ' if rose else 'FAIL '}  an unparseable module raises parse_failures {len(base_fail)} -> {len(f4)}"
    )
    ok = ok and rose

    # RED 5 -- GREEN CONTROL. The repaired shape must NOT be flagged, or the checker cannot tell a
    # fix from a defect. This is the half that fails when a criterion only ever ran on regressions.
    repaired = (
        f'{DECL_NAME} = {{"Rule.learn": ("{LOCAL}", "purely local, no gradient claimed")}}\n'
        "class Rule:\n"
        "    def learn(self, x):\n"
        "        self.W -= 0.1 * x\n"
    )
    r5, _, _ = census(overlay={"cortex/zz_probe_rule.py": repaired})
    st5 = [r["state"] for r in r5 if r["module_rel"] == "cortex/zz_probe_rule.py"]
    clean = st5 and all(s == CLASSIFIED_LOCAL for s in st5)
    print(
        f"  {'green' if clean else 'FAIL '}  CONTROL: the repaired shape is NOT flagged -> {sorted(set(st5))}"
    )
    ok = ok and bool(clean)

    # RED 6 -- determinism. Two walks of the same tree must agree exactly (SC-009).
    a, _, _ = census()
    b, _, _ = census()
    same = [r["qualname"] for r in a] == [r["qualname"] for r in b] and counts(a) == counts(b)
    print(f"  {'green' if same else 'FAIL '}  two runs over the same tree are identical")
    ok = ok and same

    # RED 7 -- the baseline writer must REFUSE a rise. A ceiling raised together with the regression
    # it excuses passes both --assert and --self-check on the existing p6_gate: laundered clean.
    worse = dict(base_gov)
    worse[MISSING_DECLARATION] = worse[MISSING_DECLARATION] + 1
    refused = _refuses_rise(worse, {"counts": {MISSING_DECLARATION: base_gov[MISSING_DECLARATION]}})
    print(
        f"  {'RED  ' if refused else 'FAIL '}  --update-baseline refuses a rise ({MISSING_DECLARATION} +1 offered)"
    )
    ok = ok and refused

    # RED 8 -- PHANTOM DECLARATION (owner task 2026-08-27; the T050 incident: a typo'd
    # qualname produced a governed row for a symbol with no code, indistinguishable from
    # coverage). Both arms run THE EXACT COMPOSITION main()'s --assert path runs --
    # evaluate(governed_counts, ..., c_all=all_counts) -- because this red's first draft
    # proved itself only under evaluate(all_counts, ...), a composition main() never uses
    # (the panel's finding 1: an advertised red the assert path could not fire). The
    # overlay keeps the construction pure (no module-object patching, so the -m
    # double-import trap never arises).
    phantom_src = (
        f'{DECL_NAME} = {{"Ghost.no_such_method": ("{LOCAL}", "keys a symbol with no code")}}\n'
        "class Rule:\n"
        "    def learn(self, x):\n"
        "        self.W -= 0.1 * x\n"
    )

    def _main_composition_reds(rules_x, n_x, needle=PHANTOM_DECLARATION):
        gov_x = counts(rules_x, governed_only=True)
        return [
            f
            for f in evaluate(gov_x, None, n_x, c_all=counts(rules_x, governed_only=False))
            if needle in f
        ]

    # (a) GOVERNED phantom: overlay a GOVERNED module's relpath (readout/tempotron.py is in
    # BRAIN_MODULES; a zz_probe filename would be ungoverned by construction).
    r8a, _, n8a = census(overlay={"readout/tempotron.py": phantom_src})
    gov_hit = any(
        r["state"] == PHANTOM_DECLARATION and r["governed"]
        for r in r8a
        if r["module_rel"] == "readout/tempotron.py"
    )
    gov_reds = _main_composition_reds(r8a, n8a)
    print(
        f"  {'RED  ' if gov_hit and gov_reds else 'FAIL '}  GOVERNED phantom -> "
        f"{PHANTOM_DECLARATION} and the ASSERT composition reds"
    )
    ok = ok and gov_hit and bool(gov_reds)
    # (b) UNGOVERNED phantom: the ALL-scope bound must catch it through the SAME composition
    # (declarations are tree-wide artifacts; every bearing row enters rule_census.json).
    r8b, _, n8b = census(overlay={"cortex/zz_probe_rule.py": phantom_src})
    ungov_reds = _main_composition_reds(r8b, n8b)
    print(
        f"  {'RED  ' if ungov_reds else 'FAIL '}  UNGOVERNED phantom -> the SAME assert "
        f"composition reds (ALL-scope bound)"
    )
    ok = ok and bool(ungov_reds)
    # (c) GREEN control: the same key WITH its symbol present is not flagged, else fixed
    # and broken are indistinguishable. Plus the truthful-shape controls the panel measured
    # the first draft red-flagging: a conditional def and a nested-class method must NOT
    # be phantoms (the _symbol_surface fix's own reachable green).
    ghost_present = (
        f'{DECL_NAME} = {{"Ghost.no_such_method": ("{LOCAL}", "now the symbol exists")}}\n'
        "class Ghost:\n"
        "    def no_such_method(self, x):\n"
        "        return x\n"
    )
    truthful_shapes = (
        f'{DECL_NAME} = {{"Outer.Inner.learn": ("{LOCAL}", "nested class method"), '
        f'"cond_learn": ("{LOCAL}", "conditional def"), '
        f'"Rule.learn": ("{LOCAL}", "staticmethod rebind")}}\n'
        "class Outer:\n"
        "    class Inner:\n"
        "        def learn(self, x):\n"
        "            return x\n"
        "if True:\n"
        "    def cond_learn(x):\n"
        "        return x\n"
        "def _impl(x):\n"
        "    return x\n"
        "class Rule:\n"
        "    pass\n"
        "Rule.learn = staticmethod(_impl)\n"
    )
    g_ok = True
    for src8 in (ghost_present, truthful_shapes):
        r8c, _, n8c = census(overlay={"cortex/zz_probe_rule.py": src8})
        if _main_composition_reds(r8c, n8c):
            g_ok = False
    print(
        f"  {'green' if g_ok else 'FAIL '}  CONTROL: present-symbol + truthful shapes "
        f"(nested class, conditional def, staticmethod rebind) are NOT flagged"
    )
    ok = ok and g_ok
    # (d) THE INCIDENT'S OWN SHAPE, which the first _symbol_surface draft let SLIP BACK
    # THROUGH (honesty panel: function scopes transparent for ASSIGNMENTS meant a
    # method-local variable in the WRONG class validated the wrong-class key). A key
    # naming Wrong.evaluate, with `evaluate` existing only as a local inside a Wrong
    # method and as a def on Right, MUST red.
    incident_shape = (
        f'{DECL_NAME} = {{"Wrong.evaluate": ("{LOCAL}", "the T050 wrong-class shape")}}\n'
        "class Wrong:\n"
        "    def author(self, x):\n"
        "        evaluate = x + 1\n"
        "        return evaluate\n"
        "class Right:\n"
        "    def evaluate(self, x):\n"
        "        return x\n"
    )
    r8d, _, n8d = census(overlay={"cortex/zz_probe_rule.py": incident_shape})
    inc_reds = _main_composition_reds(r8d, n8d)
    print(
        f"  {'RED  ' if inc_reds else 'FAIL '}  the T050 wrong-class shape (local var in the "
        f"wrong class) REDS -- a local variable is not a provided symbol"
    )
    ok = ok and bool(inc_reds)

    # RED 9 -- the unclassified ABSOLUTE bound (owner task 2026-08-27; the counting-loop
    # incident: counts() tested membership against the dict its own loop had just inserted
    # every SEEN state into, so a state escaping the declared table could NEVER count as
    # unclassified -- a poisoned live census measured unclassified=0, evaluate()=[], zero
    # delta vs control). Honest scope of what the fix buys: _DECLARED_STATES is derived
    # from the same table resolve() indexes, so NO TREE INPUT can produce an escape -- the
    # bound is a SOURCE-EDIT tripwire for a future state constant added to census()/
    # resolve() without extending the set, and both arms construct the row synthetically
    # because no walk can produce one. No additional census walk (base_rules is reused).
    # Both arms run main()'s assert composition (RED 8's lesson, twelve lines up), and the
    # bound asserts ALL-scope like PHANTOM, so the ungoverned arm is load-bearing.
    i9 = next((i for i, r in enumerate(base_rules) if r["governed"]), None)
    j9 = next((i for i, r in enumerate(base_rules) if not r["governed"]), None)
    if i9 is None or j9 is None:
        print("  FAIL   RED 9 needs one governed and one ungoverned row in the live census")
        ok = False
    else:
        for label9, k9 in (("GOVERNED", i9), ("UNGOVERNED", j9)):
            poisoned = [dict(r) for r in base_rules]
            poisoned[k9]["state"] = "ZZ_NOT_A_DECLARED_STATE"
            reds9 = _main_composition_reds(poisoned, base_n, needle="unclassified")
            fired9 = counts(poisoned, governed_only=False)["unclassified"] == 1 and bool(reds9)
            print(
                f"  {'RED  ' if fired9 else 'FAIL '}  a state escaping the declared table on a "
                f"{label9} row counts as unclassified and REDS the ABSOLUTE bound (ALL-scope)"
            )
            ok = ok and fired9
        # GREEN control: the same composition over the UNPOISONED census raises no
        # unclassified red -- else fixed and broken are indistinguishable.
        clean9 = not _main_composition_reds(base_rules, base_n, needle="unclassified")
        print(
            f"  {'green' if clean9 else 'FAIL '}  CONTROL: the unpoisoned census raises no "
            f"unclassified red through the same composition"
        )
        ok = ok and clean9

    print("self-check: " + ("OK" if ok else "FAILED"))
    return ok


def _refuses_rise(c, baseline):
    """True if this baseline write must be REFUSED. PURE, so the self-check can exercise it."""
    base = (baseline or {}).get("counts", {})
    for key in (MISSING_DECLARATION, UNDETERMINABLE):
        allowed = base.get(key)
        if allowed is not None and c.get(key, 0) > allowed:
            return True
    return False


# ---------------------------------------------------------------------------------------------
# THE ON-DEMAND HALF (`--measure`). Never asserts in CI (FR-014).
#
# It answers a question the census cannot: BEFORE the bar binds on a rule, can the criterion
# discriminate for it at all? A bar applied to a rule the instrument could never move is not a
# test, it is a verdict with a number attached.
#
# FR-013b -- THIS MODULE IMPORTS NO RULE-CARRYING MODULE, IN ANY MODE. Not a preference: 89 src
# modules carry `Read statically by gates/gradient_claim_gate.py; never imported by it.`, and
# an in-process measure would make that sentence false on the seven it touches. Gradients come
# from SUBPROCESSES, which also makes UNAVAILABLE a natural outcome rather than a special case --
# a fixture whose subprocess cannot import is exactly what "unavailable" means.
#
# P2 -- the cosine here is REUSED, not re-authored. Ten cosine definitions already exist in src/;
# six sit in rule-carrying modules FR-013b forbids importing, and of the four that remain
# `cortex/wholistic_loop::_cosine` and `growth/growing_learner::_cosine` both document
# NON-NEGATIVE inputs ([0,1]^N) and are wrong for a signed gradient. That leaves exactly one.
#
# distinct-from: cortex/access_consciousness.py -- its `_coherence_scalar` is mean PAIRWISE
#   similarity over coalition member codes, a between-vectors quantity. `coherence` here is
#   |mean|/rms WITHIN one vector, and answers whether a permutation null is a null at all.
# distinct-from: growth/neurogenesis_engine.py, growth/neuron_zoo.py, cortex/rose_language.py,
#   readout/growable_readout.py -- matched on the words "frozen"/"derived"/"zero"; none
#   computes a cosine, a null, or a sign-coherence statistic.
# ---------------------------------------------------------------------------------------------

RULE_JUDGED = "RULE_JUDGED"
INSTRUMENT_FALSIFIED = "INSTRUMENT_FALSIFIED"
UNAVAILABLE = "UNAVAILABLE"
VERDICTS = (RULE_JUDGED, INSTRUMENT_FALSIFIED, UNAVAILABLE)

#: Pinned so the FR-015 artifact is reproducible on an unchanged tree (FR-009e). Unpinned, `0.5`
#: sits anywhere from +2.97 to +3.27 sd across seeds and the artifact's staleness signal is noise.
MEASURE_DRAWS = 400
MEASURE_SEED = 1

#: The bar lives in exactly ONE place. It is READ STATICALLY rather than imported, because the
#: module holding it is itself rule-carrying and FR-013b forbids importing those. A second copy
#: here would be the P2 defect this whole feature exists to police.
_BAR_MODULE = "plasticity/eprop_kernel.py"
_BAR_NAME = "EPROP_BPTT_COSINE_FLOOR"


def read_bar(root=None):
    """The frozen bar, parsed from its single definition. RAISES if it is gone.

    A silent fallback to a literal 0.5 is the failure worth refusing: the constant could be renamed
    or deleted and every measurement here would keep reporting against a number no longer connected
    to the thing it claims to mirror.
    """
    root = _repo_root() if root is None else root
    if root is None:
        raise RuntimeError("no repo root; cannot read the bar")
    path = os.path.join(root, "src", "snn_data_lake", *_BAR_MODULE.split("/"))
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == _BAR_NAME for t in node.targets
        ):
            return float(ast.literal_eval(node.value))
    raise RuntimeError(f"{_BAR_NAME} is not defined at module scope in {_BAR_MODULE}")


def _cosine(a, b):
    """REUSED, never re-authored: `reasoning/weight_perturbation::cosine` (P2).

    Imported LAZILY so the CI half's import cost stays at zero -- measured 0.010 s, no torch, and
    it is the only general signed-vector cosine in the tree that FR-013b permits importing.
    """
    from snn_data_lake.reasoning.weight_perturbation import cosine

    return cosine(a, b)


def coherence(v):
    """`|mean| / rms` -- 0 when zero-mean, 1 when every entry is the same value.

    This is the quantity deciding whether the mandated null is a null at all (FR-009d).
    """
    if not v:
        return 0.0
    rms = math.sqrt(math.fsum(x * x for x in v) / len(v))
    return abs(math.fsum(v) / len(v)) / rms if rms > 0 else 0.0


def coherence_ceiling(root=None):
    """DERIVED from the frozen bar, never chosen -- P1 forbids a magic constant here.

    A within-matrix permutation destroys pairing but PRESERVES each vector's mean, so the expected
    null cosine is the PRODUCT of the two coherences. MEASURED across eight magnitudes, predicted
    vs measured null mean agrees to <=0.010 (typically <0.003), and the crossing is exact: at
    coherence 0.7069 x 0.7191 the predicted 0.5083 met a measured 0.5103. The permutation therefore
    stops being a null precisely when that product reaches the bar, making the symmetric per-vector
    ceiling sqrt(bar) -- a consequence of the bar rather than a second constant beside it.
    """
    return math.sqrt(read_bar(root))


def verdict_for(aligned, coherence, root=None):
    """FR-004 + FR-009d. The INSTRUMENT is judged before the RULE is.

    `INSTRUMENT_FALSIFIED` is not a softer `RULE_JUDGED`: the rule is left UNJUDGED, because at high
    coherence the separation statistic INVERTS -- measured on a constructed pair, null mean +0.980,
    P(null > bar) 1.0000, aligned 1.75 sd BELOW the null. Reporting a separation there would bless a
    shuffled rule and condemn an aligned one.
    """
    c_a, c_b = coherence
    if aligned is None or c_a is None or c_b is None:
        return UNAVAILABLE
    return INSTRUMENT_FALSIFIED if c_a * c_b >= read_bar(root) else RULE_JUDGED


class DiscriminationResult:
    """Per rule the bar is applied to. ALWAYS a pair -- no rule is judged on one number (FR-003).

    The pair is enforced with an explicit `raise`, NOT `assert`: `python -O` strips asserts, and a
    structural guarantee that evaporates under an optimisation flag is not structural.
    """

    __slots__ = ("qualname", "aligned", "null_sample", "coherence", "verdict", "reason")

    def __init__(self, qualname, aligned, null_sample, coherence, verdict, reason=""):
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {VERDICTS}")
        if verdict == RULE_JUDGED and not null_sample:
            raise ValueError(
                f"{qualname}: RULE_JUDGED with an EMPTY null -- FR-003 requires a PAIR, and a "
                "verdict resting on the aligned number alone is the shape this feature forbids"
            )
        if verdict != RULE_JUDGED and not reason:
            raise ValueError(f"{qualname}: {verdict} without a reason is a shrug, not a verdict")
        self.qualname = qualname
        self.aligned = aligned
        self.null_sample = list(null_sample)
        self.coherence = tuple(coherence)
        self.verdict = verdict
        self.reason = reason


class SeparationReport:
    """Where the bar SITS relative to the null. Records placement; never sets it (FR-009b)."""

    __slots__ = (
        "null_kind",
        "bar_sd_above_null",
        "aligned_sd_above_null",
        "p_null_exceeds_bar",
        "bar",
        "draws",
        "seed",
        "reported_only",
    )

    def __init__(self, null_sample, aligned, bar, draws, seed, reported_only=None):
        n = len(null_sample)
        mu = math.fsum(null_sample) / n if n else 0.0
        sd = math.sqrt(math.fsum((x - mu) ** 2 for x in null_sample) / n) if n else 0.0
        self.bar = bar
        self.null_kind = "within_matrix_permutation"
        # TWO fields, not one. On the anchor fixture the BAR sits at +3.06 sd while the ALIGNED
        # cosine sits at +5.93 sd; one field feeding both readings is right in the printed line and
        # wrong in the JSON, with nothing edited in between to show which.
        self.bar_sd_above_null = (bar - mu) / sd if sd > 0 else float("inf")
        self.aligned_sd_above_null = (aligned - mu) / sd if sd > 0 and aligned is not None else None
        self.p_null_exceeds_bar = sum(1 for c in null_sample if c > bar) / n if n else 0.0
        self.draws = draws
        self.seed = seed
        self.reported_only = reported_only or {}


def within_matrix_null(families, ref, draws=MEASURE_DRAWS, seed=MEASURE_SEED):
    """FR-009a EXACTLY as worded: permute each coordinate FAMILY among itself (W_in among W_in,
    off-diagonal W_rec among W_rec), then cosine against the UNCHANGED reference.

    `families` is a list of lists, one per family. Permuting ACROSS families is a different null
    with a different answer -- the under-specification FR-009a was amended to close.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        flat = []
        for fam in families:
            shuffled = list(fam)
            rng.shuffle(shuffled)
            flat.extend(shuffled)
        out.append(_cosine(flat, ref))
    return sorted(out)


def reported_nulls(
    families, ref, row_widths=None, ref_relabelled=None, draws=MEASURE_DRAWS, seed=MEASURE_SEED
):
    """FR-009c: three further nulls, REPORTED and asserted on by NOTHING.

    Each answers a different question, and one of them reverses the verdict -- publishing them
    pre-empts the reader who recomputes the bar mentally and gets a different answer.

    The UNTRAINED TWIN is REJECTED as a control and that rejection is recorded here rather than left
    implicit: it moves the net, the forward path and the reference all at once, so a movement in the
    statistic cannot be attributed to any one of them. A control that changes several things is not
    a control.
    """
    rng = random.Random(seed + 977)  # offset: never the same stream as the asserted null
    cand = [x for fam in families for x in fam]
    n = len(cand)
    norm = math.sqrt(math.fsum(x * x for x in cand))

    iso = []
    for _ in range(draws):
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        s = math.sqrt(math.fsum(x * x for x in v)) or 1.0
        iso.append(_cosine([x * norm / s for x in v], ref))

    # WITHIN-UNIT: permute inside each POST-SYNAPTIC UNIT's row -- a strictly FINER partition than
    # the asserted null, which permutes across a whole matrix. Deliberately a different partition,
    # not a second spelling of the same one: if it agreed by construction there would be nothing to
    # report. `row_widths` comes from the fixture, which knows the geometry; deriving it here by
    # solving H*(H-1) = len(family) would be a guess that silently mis-blocks on a shape change.
    unit = []
    if row_widths:
        for _ in range(draws):
            flat = []
            for fam, w in zip(families, row_widths):
                for i in range(0, len(fam), w):
                    chunk = fam[i : i + w]
                    rng.shuffle(chunk)
                    flat.extend(chunk)
            unit.append(_cosine(flat, ref))
    unit = sorted(unit)

    out = {
        "isotropic_matched_norm": sorted(iso),
        "within_unit_permutation": unit,
        "_untrained_twin": "REJECTED as a control: moves net, forward path and reference at once",
    }
    if ref_relabelled is not None:
        # RANDOM RELABEL: the reference recomputed against PERMUTED targets. This is the null
        # research.md R4 measured as REVERSING the verdict, so it is reported as an actual re-run
        # rather than approximated by a shuffle of the vector -- those are not the same object.
        out["random_relabel"] = [_cosine(cand, ref_relabelled)]
    else:
        out["random_relabel"] = "UNAVAILABLE: the fixture did not emit a relabelled reference"
    return out


def render_discrimination(result, report=None):
    """The DISCRIMINATING POWER block from contracts/cli.md."""

    def _f(x):
        return "--" if x is None else f"{x:.4f}"

    lines = [f"  {result.qualname}"]
    if result.aligned is None:
        lines.append("    aligned            : --")
    else:
        sd = ""
        if report is not None and report.aligned_sd_above_null is not None:
            sd = f"     ({report.aligned_sd_above_null:+.2f} sd above null)"
        lines.append(f"    aligned            : {result.aligned:+.4f}{sd}")
    lines.append(f"    coherence cand/ref : {_f(result.coherence[0])} / {_f(result.coherence[1])}")
    if result.null_sample:
        n = result.null_sample
        lines.append(
            f"    null p50 / p95     : {n[len(n) // 2]:+.4f} / {n[int(0.95 * len(n))]:+.4f}"
        )
    if report is not None:
        lines.append(
            f"    bar sits           : {report.bar_sd_above_null:+.2f} sd above null,  "
            f"P(null > bar) = {report.p_null_exceeds_bar:.4f}"
        )
    lines.append(f"    verdict            : {result.verdict}")
    if result.reason:
        lines.append(f"    reason             : {result.reason}")
    if report is not None and report.reported_only:
        lines.append("    REPORTED ONLY (FR-009c -- these assert NOTHING):")
        for key in sorted(report.reported_only):
            val = report.reported_only[key]
            if isinstance(val, list) and val and isinstance(val[0], float):
                mu = math.fsum(val) / len(val)
                p = sum(1 for c in val if c > report.bar) / len(val)
                # n= printed per null: a 1-element list rendered like a 400-draw one invites
                # reading P(> bar) as a per-draw property when it is 1 of 1 (round-8 L02).
                tag = "n=1 (single re-run)" if len(val) == 1 else f"n={len(val)}"
                lines.append(f"      {key:<24} mean {mu:+.4f}   P(> bar) {p:.4f}   {tag}")
            else:
                lines.append(f"      {key:<24} {val}")
    return "\n".join(lines)


#: The subprocess probe for the stdlib anchor. It runs in a CLEAN interpreter (FR-013b) and prints
#: ONE tagged JSON line; anything else it emits is diagnostic noise the parent ignores. The two
#: coordinate FAMILIES are kept separate because FR-009a's null permutes within each, not across.
_EPROP_PROBE = """
import json
import random

from snn_data_lake.reasoning.eprop import N_IN, N_REC, N_TRAIN, Net, _batch_grad, make_dataset

rng = random.Random(0)
train = make_dataset(N_TRAIN, rng)
net = Net(0)
y = [c for _, c in train]
e_in, e_rec, _, _ = _batch_grad(net, train, net.grad_eprop)
b_in, b_rec, _, _ = _batch_grad(net, train, net.grad_bptt)


def fams(gin, grec):
    a = [gin[j][i] for j in range(N_REC) for i in range(N_IN)]
    b = [grec[j][k] for j in range(N_REC) for k in range(N_REC) if k != j]
    return [a, b]


# The RELABEL null (FR-009c, reported only): the reference recomputed against PERMUTED
# targets. research.md R4 measured this one as REVERSING the verdict, so it is produced by an
# actual re-run rather than approximated by shuffling a vector -- those are not the same object.
y_shuf = list(y)
random.Random(4).shuffle(y_shuf)
r_in, r_rec, _, _ = _batch_grad(net, [(xs, c) for (xs, _), c in zip(train, y_shuf)], net.grad_bptt)
print(
    "GRADJSON"
    + json.dumps(
        {
            "cand": fams(e_in, e_rec),
            "ref": fams(b_in, b_rec),
            "ref_relabelled": fams(r_in, r_rec),
            "row_widths": [N_IN, N_REC - 1],
        }
    )
)
"""

#: qualname -> (probe source, what the reference IS). The reference must be INDEPENDENT of the
#: candidate; a fixture whose "reference" is the candidate recomputed measures an identity, which is
#: the shape P8 clause 2 forbids as a control.
_FIXTURES = {
    "reasoning/eprop.py::Net.grad_eprop": (
        _EPROP_PROBE,
        "reasoning/eprop.py::Net.grad_bptt (hand-rolled BPTT, independent of the e-prop path)",
    ),
}


def _run_fixture(probe, timeout=600):
    """Run one fixture OUT OF PROCESS. Returns (payload, None) or (None, reason).

    Every failure mode here is an UNAVAILABLE, not a crash and not a low cosine: an import error, a
    missing dependency and a timeout all mean the same thing -- no measurement was obtained. That is
    FR-005's distinction, and subprocess isolation is what makes it fall out naturally.
    """
    import subprocess

    try:
        p = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=timeout
        )
    except Exception as exc:  # timeout, OSError, ...
        return None, f"fixture subprocess did not complete: {type(exc).__name__}"
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        return None, f"fixture exited {p.returncode}: {tail[-1] if tail else 'no stderr'}"
    for line in p.stdout.splitlines():
        if line.startswith("GRADJSON"):
            try:
                return json.loads(line[len("GRADJSON") :]), None
            except ValueError as exc:
                return None, f"fixture emitted unparseable payload: {exc}"
    return None, "fixture produced no GRADJSON line"


def measure_all(root=None, draws=MEASURE_DRAWS, seed=MEASURE_SEED):
    """Every registered fixture, measured. Returns [(DiscriminationResult, SeparationReport|None)].

    Rules with no registered fixture are NOT silently dropped -- silence would be
    indistinguishable from a clean measurement, which is the FR-015 failure in miniature.
    """
    try:
        bar = read_bar(root)
    except RuntimeError as exc:
        return [
            (
                DiscriminationResult(q, None, [], (None, None), UNAVAILABLE, str(exc)),
                None,
            )
            for q in _FIXTURES
        ]

    out = []
    for qualname, (probe, _ref_desc) in sorted(_FIXTURES.items()):
        payload, why = _run_fixture(probe)
        if payload is None:
            out.append(
                (DiscriminationResult(qualname, None, [], (None, None), UNAVAILABLE, why), None)
            )
            continue
        fams = payload["cand"]
        cand = [x for fam in fams for x in fam]
        ref = [x for fam in payload["ref"] for x in fam]
        aligned = _cosine(cand, ref)
        c = (coherence(cand), coherence(ref))
        verdict = verdict_for(aligned, c, root)
        if verdict == INSTRUMENT_FALSIFIED:
            out.append(
                (
                    DiscriminationResult(
                        qualname,
                        aligned,
                        [],
                        c,
                        INSTRUMENT_FALSIFIED,
                        f"coherence product {c[0] * c[1]:.4f} >= bar {bar}: a within-matrix "
                        "permutation preserves the mean, so it is not a null here and the "
                        "separation statistic inverts. The rule is left UNJUDGED",
                    ),
                    None,
                )
            )
            continue
        null = within_matrix_null(fams, ref, draws, seed)
        relab = payload.get("ref_relabelled")
        extra = reported_nulls(
            fams,
            ref,
            payload.get("row_widths"),
            None if relab is None else [x for fam in relab for x in fam],
            draws,
            seed,
        )
        out.append(
            (
                DiscriminationResult(qualname, aligned, null, c, RULE_JUDGED),
                SeparationReport(null, aligned, bar, draws, seed, extra),
            )
        )
    return out


MEASURE_ARTIFACT = os.path.join("tools", "gradient_claim_measure.json")

# distinct-from: infra/self_build_loop.py -- `_crc_map` is the closest thing in the tree and
#   is NOT the same mechanism: it crc32s the BYTES of each safety-envelope file, one entry per file,
#   so a self-modification can be detected and rolled back. `tree_digest` below crc32s the PARSED
#   DECLARATIONS (qualname=disposition), one digest for the whole census, to answer whether a
#   recorded measurement still describes the current tree. Different input, different granularity,
#   different question -- and neither can serve the other's purpose.
# distinct-from: infra/self_mod_harness.py (runs selfchecks in a sandbox),
#   reasoning/make_results_plot.py (writes an SVG), cortex/time_cells.py (returns a max),
#   cortex/quality_diversity.py, reasoning/godel_machine.py -- matched on "returns/records"; none
#   computes a fingerprint of anything.


def tree_digest(rules=None):
    """A stable fingerprint of the DECLARATIONS a measurement was taken against.

    This is what makes staleness VISIBLE (FR-015). A timestamp cannot: an artifact dated yesterday
    against a tree edited this morning still reads as fresh, leaving "nobody ran it lately"
    indistinguishable from "it passed" -- the exact confusion FR-015 exists to remove. Comparing
    this digest answers the question that actually matters: was this measured against the
    declarations in the tree RIGHT NOW?

    `zlib.crc32`, never the builtin `hash()`, which is salted per process and would make the
    artifact differ on every run for no reason (project determinism rule).
    """
    import zlib

    if rules is None:
        rules, _, _ = census()
    payload = ";".join(
        f"{r['qualname']}={r['declared']}" for r in sorted(rules, key=lambda r: r["qualname"])
    )
    return f"crc32:{zlib.crc32(payload.encode('utf-8')):08x}"


def _write_measure_artifact(results, root=None):
    """Record the run so its age is CHECKABLE. Returns the path written, or None."""
    root = _repo_root() if root is None else root
    if root is None:
        return None
    path = os.path.join(root, MEASURE_ARTIFACT)
    payload = {
        "_generated_by": "python -m snn_data_lake.gates.gradient_claim_gate --measure",
        "_staleness": (
            "Compare tree_digest with a fresh run. A MISMATCH means this file predates the current "
            "declarations and its verdicts describe a tree that no longer exists."
        ),
        "_asserts": "NOTHING. This mode never gates CI (FR-014).",
        "null_kind": "within_matrix_permutation",
        "draws": MEASURE_DRAWS,
        "seed": MEASURE_SEED,
        "tree_digest": tree_digest(),
        "freshness": "CURRENT",  # recomputed on every write; read it back with is_stale()
        "results": [
            {
                "qualname": r.qualname,
                "aligned": r.aligned,
                "coherence": list(r.coherence),
                "verdict": r.verdict,
                "reason": r.reason,
                "bar_sd_above_null": None if rep is None else rep.bar_sd_above_null,
                "aligned_sd_above_null": None if rep is None else rep.aligned_sd_above_null,
                "p_null_exceeds_bar": None if rep is None else rep.p_null_exceeds_bar,
            }
            for r, rep in results
        ],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


# distinct-from: reservoir/criticality_regulator.py -- its `sigma`/`measure` run a live measurement
#   battery on a spiking pool to estimate a branching ratio. This reads a RECORDED FILE and compares
#   a content digest. Nothing is measured here and no pool is involved; the shared word is "measure".
#: FR-010's SINGLE named exception: the one rule body this feature may change, plus the two files
#: carrying recorded numbers that the change necessarily moves. Anything else touching a
#: learning-rule body is out of scope by construction rather than by judgement.
FR010_ALLOWED_RULE_EDITS = (
    "src/snn_data_lake/plasticity/eprop_kernel.py",  # _TinyNet.eprop_grads + its recorded numbers
    "src/snn_data_lake/basal_ganglia/mlp_q.py",  # the SECOND copy of the share figure only
)

# distinct-from: src/snn_data_lake/instruments/integration_measure.py -- its `_run` executes one
#   perturbed subnet simulation against a pre-drawn noise realization. It inspects no change set,
#   reads no path, and has nothing to do with which files a diff touched.
# distinct-from: src/snn_data_lake/engine/learn_daemon.py -- its `_self_check` asserts the
#   daemon does not start its drive loop under run_selfchecks. Also a self-check, also returns a
#   list; it examines the daemon's own behaviour, never a set of changed paths.
#
# P2 NOTE -- CLOSED 2026-08-18, in its own change, which is what it asked for. `_SAFETY_ENVELOPE`
# above WAS the THIRD hand-maintained copy of the envelope path list in src/, alongside
# `reasoning/godel_machine.py` and `reasoning/self_improve.py`, in two different path spellings. All
# three agreed when measured, so this was a latent defect rather than an active one. The note said
# consolidating it here would be the wrong lane, because this feature's FR-010 forbids touching the
# envelope; that reasoning stands and is why it was done separately instead of not at all. There is
# now ONE definition, `gates/safety_envelope.ENVELOPE`, which all three import, and it carries a
# self-check asserting the list still matches SECURITY.md's table -- so code cannot drift from the
# document that authorises it either. Nothing in the envelope itself was touched.


def fr010_findings(changed_paths, approved=()):
    """Paths in a change set that FR-010 forbids: the safety envelope, first and absolutely.

    Returns a list of (path, why). PURE -- it is handed paths and reads no git, so a test can plant
    a violation without needing a repository in a particular state. That matters here more than
    usual: a check never shown to produce a finding cannot evidence the absence of one (FR-010a),
    and certifying an absence is this function's entire job.

    `approved` (specs/022 T037a, the first envelope byte change since this check landed): the
    envelope paths whose modification IS the explicit approval -- computed by `fr010_approved`,
    never assumed -- are not findings. Handed in explicitly so the constructed red stays reachable:
    a planted path with no approval still fires.
    """
    approved = {a.replace("\\", "/").lstrip("./") for a in approved}
    out = []
    for p in changed_paths:
        norm = p.replace("\\", "/").lstrip("./")
        if norm in approved:
            continue
        for env in _SAFETY_ENVELOPE:
            if norm.endswith(env):
                out.append((p, f"SAFETY ENVELOPE: {env} is modified only with explicit approval"))
                break
    return out


def fr010_approved(changed_paths):
    """The envelope paths in a change set whose modification is APPROVED -- the project's approval
    act mechanized by specs/022 T021b: `tools/envelope_hashes.json` lands in the SAME change set
    (so the baseline update is reviewable in the same diff) AND `envelope_hash_gate.check()` holds,
    i.e. the bytes on disk ARE the approved bytes. Either condition missing approves nothing: a
    tampered envelope file with a stale baseline, or a baseline edited for a file that does not
    match it, stays a finding. Returns normalized paths."""
    norm = {p.replace("\\", "/").lstrip("./") for p in changed_paths}
    if "tools/envelope_hashes.json" not in norm:
        return set()
    from snn_data_lake.gates import envelope_hash_gate

    ok, _findings = envelope_hash_gate.check()
    if not ok:
        return set()
    return {p for p in norm if any(p.endswith(env) for env in _SAFETY_ENVELOPE)}


def fr010_changed_paths(root=None):
    """The tracked+untracked change set, or None when git is unavailable.

    None rather than an empty list on failure. An empty list reads as "nothing was touched", which
    is exactly the false clean bill this check exists to avoid handing out.
    """
    import subprocess

    root = _repo_root() if root is None else root
    if root is None:
        return None
    try:
        p = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=120
        )
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return [ln[3:].strip().strip('"') for ln in p.stdout.splitlines() if ln.strip()]


def measure_freshness(root=None):
    """FR-015 / T058: is the recorded measurement STALE, CURRENT, or ABSENT?

    Returns (verdict, detail). The three are structurally different answers and collapsing any two
    is the failure this requirement exists to prevent:

      ABSENT  -- nobody has ever run it. Must NOT read as a pass.
      STALE   -- it ran, against declarations that have since changed. Its verdicts describe a tree
                 that no longer exists.
      CURRENT -- it ran against exactly these declarations.

    The digest is RECOMPUTED HERE, from the tree, and compared with the recorded one. A freshness
    field merely written and read back would report CURRENT forever -- the never-recomputed-digest
    defect SC-017 names, and one that reads identical to a digest that always matches.
    """
    root = _repo_root() if root is None else root
    if root is None:
        return "ABSENT", "no repo root; the artifact cannot be located"
    path = os.path.join(root, MEASURE_ARTIFACT)
    if not os.path.exists(path):
        return "ABSENT", f"{MEASURE_ARTIFACT} has never been written"
    try:
        with open(path, encoding="utf-8-sig") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        return "ABSENT", f"{MEASURE_ARTIFACT} is unreadable: {exc}"
    recorded = payload.get("tree_digest")
    live = tree_digest()
    if recorded == live:
        return "CURRENT", f"digest {live} matches the tree"
    return "STALE", f"recorded {recorded} but the tree is {live}"


def _measure(root=None):
    """`--measure`. Prints the DISCRIMINATING POWER block and returns its exit code.

    Exit 0 measured; exit 2 when NOTHING could be measured (FR-005). Exit 2 is deliberately not
    "any rule was unavailable" -- a single torch-tier fixture missing on a stdlib box would then
    make the exit code carry no information about the rest.
    """
    results = measure_all(root)
    # `coherence_ceiling()` is PRINTED here, not merely defined. It had no production caller at all
    # (`verdict_for` re-expresses the rule inline as the exact product test), which made the function
    # documenting the derivation dormant -- P3, found by the specs/020 audit. Surfacing it is also
    # the more useful half: a reader sees WHERE the instrument refuses, not just that it can.
    try:
        ceiling = f"{coherence_ceiling(root):.4f}"
    except RuntimeError:
        ceiling = "--"
    print(
        f"DISCRIMINATING POWER   null=within_matrix_permutation  "
        f"draws={MEASURE_DRAWS}  seed={MEASURE_SEED}  coherence ceiling={ceiling}"
    )
    for result, report in results:
        print(render_discrimination(result, report))
    n_unavail = sum(1 for r, _ in results if r.verdict == UNAVAILABLE)
    print(
        f"  {len(results)} fixture(s): "
        f"{sum(1 for r, _ in results if r.verdict == RULE_JUDGED)} judged, "
        f"{sum(1 for r, _ in results if r.verdict == INSTRUMENT_FALSIFIED)} instrument-falsified, "
        f"{n_unavail} unavailable"
    )
    print("  this mode NEVER asserts in CI (FR-014); a verdict here gates nothing")
    # T061: stated where the numbers are, not in a document a reader of this output may never open.
    print("  EVERY separation above is measured AT INITIALISATION. e-prop truncation error grows")
    print("  as recurrence strengthens, so a TRAINED net can read materially lower -- on the torch")
    print(
        "  fixture, raising recurrent gain alone moves cosine_hidden 0.9993 -> -0.1782. These are"
    )
    print("  a floor on the INSTRUMENT, never a claim about a trained network.")
    written = _write_measure_artifact(results, root)
    if written:
        fresh, why = measure_freshness(root)
        print(
            f"  recorded to {os.path.relpath(written, _repo_root() or '.')} (FR-015): {fresh} -- {why}"
        )
    if results and n_unavail == len(results):
        print("NOT-ELIGIBLE: nothing could be measured in this environment.")
        return 2
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.perf_counter()
    rules, failures, n_found = census()
    elapsed = time.perf_counter() - t0
    baseline = load_baseline()

    if "--self-check" in argv:
        return 0 if _selfcheck() else 1

    if "--measure" in argv:
        return _measure()

    gov, allr = _report(rules, failures, n_found, baseline, elapsed)

    if "--update-baseline" in argv:
        if _refuses_rise(gov, baseline):
            print("REFUSED: --update-baseline may only ratchet DOWN.")
            print("Fix the finding or declare the rule; do not absorb it into the baseline.")
            return 1

        # CLOSING THE SEAM the refusal above cannot reach. Deleting the baseline makes
        # `_refuses_rise` vacuous -- there is nothing to compare against -- so a silent
        # delete-and-re-derive laundered a rise past it. Nothing short of signing can PREVENT that,
        # but a bootstrap can be made impossible to perform SILENTLY: creating a baseline where none
        # exists requires a written reason, and that reason is stored in the file. A reader seeing
        # the counts jump then finds either a stated cause or a conspicuously absent one.
        reason = ""
        for i, a in enumerate(
            argv
        ):  # `--bootstrap X` and `--bootstrap=X`, as the siblings read flags
            if a.startswith("--bootstrap="):
                reason = a.split("=", 1)[1].strip()
            elif a == "--bootstrap" and i + 1 < len(argv):
                reason = argv[i + 1].strip()
        if baseline is None and not reason:
            print(
                "REFUSED: no baseline exists, so this would CREATE one rather than ratchet it down."
            )
            print(
                "A fresh baseline records whatever the tree currently is, including a regression."
            )
            print(
                'Re-run with:  --update-baseline --bootstrap "<why the counts are what they are>"'
            )
            print(
                "The reason is written into the file, so the next reader can tell a widened census"
            )
            print(
                "from a laundered rise. Deleting the baseline to avoid this is the thing it detects."
            )
            return 1

        # LOWERING THE LIVENESS FLOOR IS THE OTHER WAY TO LAUNDER A DEAD WALK, and it was unguarded
        # until 2026-08-17: `_refuses_rise` checks the two ratchets and never looked at rules_seen, so
        # a walk that stopped seeing half the tree could be blessed by re-running this flag. The floor
        # is the ONLY bound that separates a complete census from a broken one -- `unclassified == 0`
        # is TRUE of a walk that found nothing -- so it gets the same treatment as bootstrapping: a
        # fall must carry a written reason, which is then stored beside the number.
        prev_floor = (baseline or {}).get("rules_seen_floor")
        if prev_floor is not None and gov["rules_seen"] < prev_floor and not reason:
            print(
                f"REFUSED: rules_seen {gov['rules_seen']} is BELOW the recorded floor {prev_floor}."
            )
            print(
                "The liveness floor is the only bound that can tell a narrowed walk from a broken"
            )
            print(
                "one, and this would lower it silently. If the walk was deliberately narrowed, say so:"
            )
            print('  --update-baseline --bootstrap "<why the census now sees fewer rules>"')
            return 1

        payload = {
            "_generated_by": "python -m snn_data_lake.gates.gradient_claim_gate --update-baseline",
            "_contract": "counts may only go DOWN; rules_seen_floor is a LIVENESS bound, not a target",
            "counts": {k: gov[k] for k in (MISSING_DECLARATION, UNDETERMINABLE)},
            "rules_seen_floor": gov["rules_seen"],
        }
        if reason:
            payload["_bootstrap_reason"] = reason
        elif baseline is not None and "_bootstrap_reason" in baseline:
            payload["_bootstrap_reason"] = baseline["_bootstrap_reason"]  # survives a ratchet DOWN
        with open(_baseline_path(), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"baseline written: {BASELINE_NAME}")
        return 0

    # BARE `__main__` ASSERTS, exactly as every sibling gate does. `run_selfchecks.py` invokes each
    # registered module through `runpy.run_module(..., run_name="__main__")` with NO argv, so a module
    # whose default path exits 0 regardless of findings contributes a PASS that can never fail --
    # "a gate wired in BARE is a log line wearing a gate's clothes", the lesson ci.yml records three
    # times. This module shipped that way for one commit: registered in run_selfchecks (T003) while
    # its default path was pure report.
    #
    # MEASURED 2026-08-17, which is why this changed rather than the claim being defended:
    # `dormancy_gate` and `method_dormancy_gate` carry NO --assert flag and simply assert in main;
    # `eprop_transport_gate` and `doc_budget_gate` have the flag AND assert in main. All four exit 0
    # only because the tree is clean -- `dormancy_gate` bare exited 1 earlier today when this very
    # module was dormant. contracts/cli.md claimed report-mode-never-fails was "consistent with the
    # sibling gates"; that was false, and the contract is corrected rather than the outlier kept.
    _MODE_FLAGS = ("--self-check", "--update-baseline", "--measure")
    if "--assert" in argv or not any(f in argv for f in _MODE_FLAGS):
        if baseline is None:
            print(f"GATE FAIL: no baseline at {BASELINE_NAME}.")
            print(
                "Generate it with: python -m snn_data_lake.gates.gradient_claim_gate --update-baseline"
            )
            return 1
        findings = evaluate(gov, baseline, n_found, c_all=counts(rules, governed_only=False))
        if len(failures) != 0:
            findings.append(f"parse_failures={len(failures)} (ABSOLUTE bound is 0)")
        if findings:
            print("gate FAIL:")
            for f in findings:
                print(f"  {f}")
            return 1
        print(
            "gate OK: governed tier within every bound, PHANTOM asserted ALL-scope "
            "(NO WORSE -- not 'the scope is correct')"
        )
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
