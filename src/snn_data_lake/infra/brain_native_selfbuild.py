r"""
BRAIN-NATIVE SELF-BUILD -- open-ended, source-pluggable, three-factor-learned, gate-firewalled.

WHAT THIS IS (honest framing -- read before believing any "self-improvement" claim):
  An OPEN-ENDED, SOURCE-PLUGGABLE self-build framework. Candidate self-modifications from one or more
  SelfBuildSource plugins compete in a reward-modulated competitive bandit that reuses the BRAIN'S OWN
  three-factor rule (plasticity/r_stdp.RSTDPNeuron: w[k] += lr*reward*(elig_k - mean_elig)) to learn
  WHICH candidate to sandbox-test next -- behind the existing byte-exact-rollback + protected-crc firewall
  (infra/self_build_loop + infra/self_mod_harness, UNCHANGED -- both paths corrected
  2026-08-04, they read `reasoning/...` here after the region-reorganisation wave moved the files and
  nothing re-read the sentence). It replaces (retires) the old
  reasoning/improvement_proposer, a fixed-heuristic NUMERIC-CONSTANT TUNER whose own docstring admitted
  "NOT a learner ... no plasticity rule" -- exactly what the brain-native/no-hardcode policy forbids.

  Policy compliance:
    (1) brain-native / no-hardcode -- selection is a LEARNED three-factor weight (persisted, so it
        autonomously updates across restarts); ranking/exploration are RELATIVE to the running weight
        spread (no fixed epsilon/temperature). The CHANGES it proposes are DE-HARDCODINGS: replace a
        fixed ABSOLUTE behaviour-arbitrating threshold with a parameter-free RUNNING-BASELINE form
        (mean + sample-std, one canonical sigma, no tuned knob). "Never keep a constant fixed."
    (2) adopt-better + retire the inferior -- the tuner is DELETED from the path, not run beside this.

  Reward = the GATE OUTCOME (kept +1 / rolled-back -1), NEVER the census/self._rungs. (A census reward is
  guaranteed-null against a near-constant pass-count, or compute-GAMED by any metric monotone in its own
  budget knob -- adversarial-audit finding.) The gate outcome is binary, budget-invariant, non-gameable.

WHAT THIS IS NOT (do not overclaim):
  * It INVENTS NOTHING. The source catalogue and each source's target constants are developer-AUTHORED;
    the learner only re-orders an authored menu. It converges to honest-null as the gate-SAFE
    de-hardcodings are exhausted -- and honest-null is the GOAL state (a less-hardcoded brain), REPORTED,
    never faked and never patched over with a smuggled cold-start constant.
  * Learning from raw real-world / internet data is a DESIGNED EXTENSION POINT (DataIngestSource, a
    documented NotImplemented stub reusing sensorimotor/web_ingest + the chat DurableChatMemory), NOT built
    here. This is an honest, extensible substrate for it -- not "unbounded self-improvement" today.

HONEST EMPIRICAL FINDING (why the de-hardcode source honest-nulls on many seeds): a fixed absolute
threshold like locus_coeruleus.SALIENCE_TH=0.50 encodes GLOBAL calibration knowledge (it "knows" a
drive of 0.2 is small and 1.0 large). A PER-INSTANCE running baseline built from a short self-check
cannot reconstruct that global point -- so the naive swap changes behaviour and the module's own
self-check (the gate) REJECTS it. That is the "if a hardcode resists replacement, the blocking
constraint is the bug" corollary made concrete: the blocking constraint (a globally-calibrated operating
point pinned by the self-check) is surfaced, and the honest fix is learned recalibration from data -- i.e.
a future DataIngestSource -- not a heuristic swap. The DehardcodeSource is proven gate-safe on a target
whose self-check DOES exercise the signal across its range (see the self-check's toy module).

Determinism (per CLAUDE.md): random.Random(seed) only; zlib.crc32 for tie-break jitter (never builtin
hash); no wall-clock in any returned value; ASCII-only prints.

DEPENDENCIES (P7 -- judged on the merits, not assumed from a purity rule). All arithmetic here is a
Welford recurrence over ONE scalar at a time plus a sample-std over the learned weight vector, whose
length is len(_ALL_IDS): 0 today, 2 in the self-check, a handful even with a fully authored registry.
MEASURED on .venv-lake: numpy is 14.89x SLOWER than stdlib at n=2 and 6.77x at n=8, breaks even near
n=128, and only wins at n=1024 (0.45x); a sequential Welford recurrence does not vectorize at all. So
there is no zero-dependency compromise to restore in this module -- stdlib is correct AT THIS WIDTH on
evidence, and the module makes no zero-dep claim that needs deleting. If a registry ever reaches
hundreds of candidates, re-measure and switch: n~128 is the documented trigger, not a preference.

Run the self-check with `python -m <this module's dotted path>`; the exact path is DERIVED and printed
in the self-check banner, so relocating this file cannot leave a stale invocation line behind.
"""

import json
import math
import os
import re
import zlib

from snn_data_lake.infra.self_mod_harness import _PROTECTED
from snn_data_lake.plasticity.r_stdp import RSTDPNeuron
from snn_data_lake.substrate.home import load_selfbuild, load_state, save_selfbuild

# This module's own dotted name -- correct both when imported and when run as `python -m ...` (runpy
# sets __spec__ to the real name even though __name__ becomes "__main__"). The self-check's toy patches
# generate a `from <pkg> import RunningBaseline` line from this, so a MOVE cannot stale it.
_MODNAME = __spec__.name if __spec__ is not None else __name__


def _repo_root():
    """Repo root derived from THIS file: <root>/src/snn_data_lake/<region>/<this>.py -> three levels up.
    Derived locally rather than imported from self_mod_harness, so that relocating either module cannot
    make the other one's answer wrong, and so the firewall's self-path below has a root to be relative
    to without depending on another module's private helper. Verified equal to the harness's own value
    at the pre-move location (both -> the repo root); the depth is identical from any region directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))


# SELF-PATH, DERIVED -- never a literal. The firewall entry that stops the learner from editing THIS
# file used to be the hardcoded string "src/snn_data_lake/reasoning/brain_native_selfbuild.py".
# MEASURED before this change: _forbidden("src/snn_data_lake/infra/brain_native_selfbuild.py")
# returned False, so relocating this file would have made the proposer a legal target of its own
# self-build -- while the G5 self-check went on asserting the OLD string and passing GREEN. A safety
# firewall that goes stale on a file move and still reports success is what this derivation removes.
_SELF_REL = os.path.relpath(os.path.abspath(__file__), _repo_root()).replace(os.sep, "/")

# --- learning-rule constants (BENIGN: a learning RATE + trace init, NOT behaviour-arbitrating thresholds;
#     these are the brain's own `Brain._drive_lr` idiom). No DECISION threshold is a fixed constant.
#     Cited as "brain.py L568" until 2026-08-04; that line is a CUDA-load comment. The assignment is
#     `self._drive_lr = 0.02` in `Brain.__init__` -- named by SYMBOL now, because a line number in a
#     4,000-line file is a citation with a half-life.
_SELECT_LR = 0.02  # three-factor learning rate (matches Brain._drive_lr) -- a step size, not a gate
_W0 = 0.1  # uniform initial preference (matches Brain._drive_w init) -- pre-learning tie
# P1 exception (a), fixed point on a NORMALIZED output: the gate outcome is binary, so the reward is a
# dimensionless polarity on [-1,1], not a magnitude anyone could tune. There is no scale to learn.
_REWARD_KEPT = 1.0
_REWARD_ROLLED = -1.0
# CIRCUIT-BREAKER backstop: after this many consecutive futile cycles (candidates proposed, none kept)
# on the SAME candidate set, skip the expensive gate until the set changes.
# P1, STATED PLAINLY RATHER THAN CLAIMED AWAY: this fits NONE of P1's four documented exceptions, and
# it is live (engine/brain.py calls should_skip/note_cycle). It is declared here with its reason rather
# than carried silently. Reason: it is a flake-tolerance RETRY BUDGET for an EXTERNAL multi-minute gate
# (a subprocess run of run_selfchecks.py), not a cognitive threshold -- it arbitrates SCHEDULING only.
# There is no running baseline to learn it from: on a fixed candidate set a non-flaky gate is
# deterministic, so the mechanism-derived value would be 1, and the only thing >1 buys is tolerance of
# gate flakiness, which this module does not observe and therefore cannot calibrate. Worst case of a
# wrong value is a delayed or slightly-repeated retry; rollback stays byte-exact either way, so it can
# never cause a wrong EDIT. Open review debt, not a silent hardcode.
_FUTILE_LIMIT = 3

# --- EXTENDED FIREWALL (guard G5): the monotonic correctness ratchet must NOT be relaxable by any
#     whitelisted self-edit. On top of self_mod_harness._PROTECTED (5 brainstem safety files +
#     run_selfchecks + godel_machine + self_improve + self_mod_harness), the self-build's OWN machinery
#     and its scorer/accept modules are NEVER legal targets -- else the learner could learn to loosen its
#     own gate.
#     MATCHED BY BASENAME, not by full path (see _forbidden). Every entry here and in _PROTECTED is a
#     safety-relevant module that a region-reorganisation wave may relocate, and a full-path match means
#     the protection disappears the instant the file moves -- silently, because nothing re-checks it.
#     Basename matching is deliberately OVER-BROAD (it would also refuse a same-named file elsewhere),
#     which is the correct direction to be wrong in for a firewall. MEASURED blast radius today: zero
#     -- every one of these basenames resolves to exactly one file in src/snn_data_lake/.
#     `improvement_proposer.py` is the RETIRED numeric tuner; the file is already deleted, and the entry
#     is kept so that re-creating it anywhere would still not be a legal self-edit target.
_SELFBUILD_OFF_LIMITS = frozenset(
    {
        "self_build_loop.py",  # the gated-apply / accept path
        os.path.basename(_SELF_REL),  # this proposer itself -- DERIVED, so a move cannot stale it
        "improvement_proposer.py",  # the retired tuner (never a target, even if re-created)
        "self_improvement_suite.py",  # the census scorer
    }
)
_PROTECTED_NAMES = frozenset(os.path.basename(p) for p in _PROTECTED)


def _forbidden(rel):
    """True if `rel` is not a legal self-build target: protected/off-limits, under brainstem/, escaping
    the repo, or not a .py under src/snn_data_lake/. The four-layer firewall's outermost gate.

    Protected and off-limits files are matched by BASENAME as well as by full path, so RELOCATING one of
    them cannot silently un-protect it. Over-broad on purpose: refusing a legal target is a missed
    de-hardcode, allowing an illegal one is a self-build that can edit its own gate."""
    rel = str(rel).replace("\\", "/")
    parts = rel.split("/")
    base = parts[-1]
    if rel in _PROTECTED or base in _PROTECTED_NAMES or base in _SELFBUILD_OFF_LIMITS:
        return True
    if "brainstem" in parts:
        return True
    if rel.startswith("/") or ":" in rel:  # absolute path / drive letter -> escapes the repo
        return True
    if any(p in ("", ".", "..") for p in parts):
        return True
    if not rel.endswith(".py") or parts[:2] != ["src", "snn_data_lake"]:
        return True
    return False


class RunningBaseline:
    """Parameter-free running mean + sample-std of a scalar stream (Welford, numerically stable).

    threshold() = mean + sample_std  -- one canonical sigma above the running mean; there is NO tuned k.
    BOUNDARY-HONEST (guard G6): with < 2 samples it returns None ("not enough data to gate on the running
    baseline yet"). The caller MUST treat None as 'do not fire on this baseline yet' -- never fall back to
    a hidden constant (that would re-introduce the magic number through the back door). This is the honest
    substrate a DehardcodeSource replacement imports and uses in place of an absolute literal.
    """

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def observe(self, x):
        x = float(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def ready(self):
        return self._n >= 2

    def mean(self):
        return self._mean if self._n >= 1 else None

    def threshold(self):
        if self._n < 2:
            return None
        var = self._m2 / (self._n - 1)
        return self._mean + math.sqrt(var)


# ================================================================================================
# SOURCES -- the open-ended seam. A source yields gate-verifiable candidate self-modifications.
# ================================================================================================
class SelfBuildSource:
    """Base class: a plug-in that proposes candidate self-modifications. Subclass and implement
    candidates(repo) -> list of dicts {id, source, target, kind, rationale, edit_fn, is_applied}.
    edit_fn(root) applies the edit under `root` (a sandbox copy first, then -- if the gate is green --
    the real tree, via SelfBuildLoop.promote). is_applied(repo) reports whether the edit is already in
    the real tree (so an applied change is not re-proposed -> the source drains to honest-null)."""

    name = "base"

    def candidates(self, repo):  # pragma: no cover - abstract
        raise NotImplementedError


def _patch_edit_fn(rel, patches, marker):
    """Build a structural edit closure that applies each (anchor_regex, replacement, count) patch to
    <root>/rel, asserting each matches exactly its `count` (else raise -> the loop rolls back byte-exact).
    Re-checks the firewall at apply time (a forged/tampered target raises before any write)."""

    compiled = [(re.compile(a, re.M), r, c) for (a, r, c) in patches]

    def edit_fn(root):
        if _forbidden(rel):
            raise RuntimeError(
                f"edit_fn refused at apply time: {rel} is not a legal self-build target"
            )
        path = os.path.join(root, *rel.split("/"))
        with open(path, encoding="utf-8", newline="") as f:
            src = f.read()
        if marker in src:
            raise RuntimeError(
                f"edit_fn: {rel} already carries the de-hardcode marker (double-apply)"
            )
        for pat, repl, count in compiled:
            src, n = pat.subn(repl, src)
            if n != count:
                raise RuntimeError(
                    f"edit_fn: patch {pat.pattern!r} matched {n} times, expected {count} in {rel}"
                )
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)

    return edit_fn


def _is_applied_fn(rel, marker):
    def is_applied(repo):
        path = os.path.join(repo, *rel.split("/"))
        try:
            with open(path, encoding="utf-8", newline="") as f:
                return marker in f.read()
        except OSError:
            return False

    return is_applied


class DehardcodeSource(SelfBuildSource):
    """SHIPPED source #1: replace a fixed ABSOLUTE behaviour-arbitrating threshold with a parameter-free
    RUNNING-BASELINE form. Each transform is a developer-authored structural patch (it makes the module
    track a RunningBaseline of the gating signal and compare against threshold() instead of the literal).
    Only transforms whose module self-check + run_selfchecks stay GREEN survive the gate; a transform
    whose absolute cut encodes global calibration a short running history cannot reconstruct simply rolls
    back and the learner deprioritises it (honest-null). The registry is authored (invents nothing)."""

    name = "dehardcode"

    def __init__(self, transforms=None):
        # transforms: tuple of specs {id, target, const, current, rationale, patches, marker}.
        self._specs = tuple(transforms) if transforms is not None else DEHARDCODE_TRANSFORMS
        for s in self._specs:
            if _forbidden(s["target"]):
                raise ValueError(
                    f"FIREWALL: transform {s['id']} targets an illegal file {s['target']}"
                )

    def candidates(self, repo):
        out = []
        for s in self._specs:
            is_applied = _is_applied_fn(s["target"], s["marker"])
            if is_applied(repo):
                continue  # already de-hardcoded -> not a candidate (source drains toward honest-null)
            out.append(
                {
                    "id": s["id"],
                    "source": self.name,
                    "target": s["target"],
                    "kind": "dehardcode",
                    "rationale": s["rationale"],
                    "edit_fn": _patch_edit_fn(s["target"], s["patches"], s["marker"]),
                    "is_applied": is_applied,
                }
            )
        return out


class DataIngestSource(SelfBuildSource):
    """EXTENSION POINT (NOT built): the open-ended direction -- ingest raw data from any source (the real
    world / the internet) and propose a gate-verified LEARNED change (e.g. a consolidated learned-weight
    recalibration of a threshold the DehardcodeSource could not de-hardcode heuristically). It would reuse
    existing ingestion primitives -- sensorimotor/web_ingest, the chat daemon's substrate/chat_memory
    DurableChatMemory, sensor/file ingest. Shipped as an honest stub so the framework is provably
    open-ended WITHOUT overclaiming that internet/real-world learning works today."""

    name = "data_ingest"

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)

    def candidates(self, repo):
        if not self.enabled:
            return []  # honest: not implemented -> proposes nothing (never a fake change)
        raise NotImplementedError(
            "DataIngestSource is a designed extension point, not implemented: a real ingestion+learning "
            "source (web_ingest / chat_memory / sensors) must be built before it may propose changes."
        )


# ------------------------------------------------------- data-recalibration targets (the real ingest) --
# Each target: a band-gated threshold whose GATING SIGNAL the brain records live (substrate/signal_calibrator,
# Step-0) -> recalibrate the hand-guessed literal to a percentile of the RECORDED distribution. `signal` must
# match the SignalCalibrator name recorded in brain.py. NEVER a brainstem/protected/scorer file.
#
# ADOPT-BETTER + RETIRE: the former target `recalib:salience_th` was a nightly SOURCE-EDIT that FROZE
# locus_coeruleus.SALIENCE_TH to the recorded P75 as a fixed literal. That is superseded -- brain.py's
# _w8_tick now CONSUMES the live running quantile straight into the LC gate every tick (self._lc.salience_th
# = calibrator.result()), so the threshold is genuinely baseline-relative and never a constant kept fixed.
# Freezing a literal is the inferior mechanism and was retired, not left additive beside the live path.
# RECALIB_TARGETS therefore drains to empty (an honest-null extension point, exactly like DEHARDCODE_TRANSFORMS
# below): the source-edit mechanism stays available for a FUTURE recorded-signal threshold that has no live
# consumer wired, but it holds no live target today.
RECALIB_TARGETS = ()


class ThresholdRecalibrationSource(SelfBuildSource):
    """A DataIngestSource: recalibrate a band-gated threshold from the brain's OWN recorded signal
    distribution (a persisted P2 quantile, substrate/signal_calibrator Step-0), emitting a
    DehardcodeSource-style source patch; the existing SelfModHarness + SelfBuildLoop gate KEEPS it iff it
    lands in the module's behavioural-band self-check, else rolls back byte-exact.

    SUPERSEDED FOR ITS ORIGINAL TARGET: this source distils a FIXED data-calibrated literal into the
    source, which is INFERIOR to consuming the running quantile LIVE at runtime (a frozen literal is still
    a constant kept fixed). Its one real target, locus_coeruleus.SALIENCE_TH, has been retired here and is
    now consumed live in brain.py _w8_tick (self._lc.salience_th = calibrator.result() every tick). So
    RECALIB_TARGETS drains to empty and this class remains a general, honest-null EXTENSION POINT for a
    future recorded-signal threshold that has no live consumer wired -- not a live production mechanism.

    HONEST CEILING (when it does hold a target): the gate certifies COMPATIBILITY (the value stays
    consistent with the module's own self-check), NOT correctness -- reward=kept != "the value is more
    right". It does NOT learn from the internet; it recalibrates the brain's OWN operating point from its
    OWN recorded data. Honest-null until the log has >= MIN_SAMPLES real samples -- NO smuggled seed
    constant. Converges to honest-null once the threshold is data-calibrated or the recorded value resists
    (its self-check band rejects it)."""

    name = "data_recalibrate"
    # P1, DECLARED not claimed away: a DATA-SUFFICIENCY floor, not a behaviour-arbitrating threshold --
    # it gates whether there is enough recorded signal to compute a quantile at all, and it never
    # compares against the signal's VALUE. It fits none of P1's four exceptions verbatim, so it is
    # stated here: below this the proposer emits nothing (honest-null) rather than a fabricated value,
    # which is the fail-safe direction, and the source is target-less today (RECALIB_TARGETS == ()), so
    # no live decision depends on it. If a real target lands, this becomes live P1 debt to re-open.
    MIN_SAMPLES = 50

    # P4b, 2026-08-05 -- `calib_path` is GONE, and it was a SECOND wrong location, not merely a
    # redundant one. It defaulted to `<repo>/data/signal_calib.json`: REPO-relative (so it ignored
    # NORD_HOME, exactly as `data/selfbuild_state.json` did before `_T_SELFBUILD` closed it) and NOT
    # the file `engine/brain.py` wrote, which was `<home>/signal_calib.json`. So this source read a
    # path the brain never wrote, and the honest-nulls below were honest for the wrong reason.
    # `lake_path=None` now means THE ONE LAKE; a path is an ISOLATED LAKE, which is what the
    # self-check uses so a gate run cannot write probe records into the operator's real lake.
    def __init__(self, recalibs=None, lake_path=None):
        self._recalibs = tuple(recalibs) if recalibs is not None else RECALIB_TARGETS
        self._lake_path = lake_path
        for r in self._recalibs:
            if _forbidden(r["target"]):
                raise ValueError(
                    f"FIREWALL: recalib {r['id']} targets an illegal file {r['target']}"
                )

    def candidates(self, repo):
        from snn_data_lake.substrate.signal_calibrator import SignalCalibrator

        out = []
        for r in self._recalibs:
            is_applied = _is_applied_fn(r["target"], r["marker"])
            if is_applied(repo):
                continue  # already data-calibrated -> not a candidate (drains toward honest-null)
            res = SignalCalibrator.load_result(r["signal"], self._lake_path)
            if not res or res["value"] is None or res["n"] < self.MIN_SAMPLES:
                continue  # HONEST-NULL: no / too little recorded data -> propose NOTHING (no fabricated value)
            val = round(max(r["lo"], min(r["hi"], float(res["value"]))), 4)
            pct = int(round(res["p"] * 100))
            new_line = f"{r['const']} = {val!r}  {r['marker']}@{res['n']} (P{pct} of recorded {r['signal']}, was {r['orig']})"
            patches = ((rf"^{re.escape(r['const'])} = [^\n#]*(?:#.*)?$", new_line, 1),)
            out.append(
                {
                    "id": r["id"],
                    "source": self.name,
                    "target": r["target"],
                    "symbol": r[
                        "const"
                    ],  # for impact-analysis blast radius (harness LSP principle)
                    "kind": "recalibrate",
                    "rationale": (
                        f"recalibrate {r['const']} {r['orig']} -> data P{pct} {val} (n={res['n']}) of the "
                        f"brain's OWN recorded {r['signal']} signal; gate certifies compatibility not correctness"
                    ),
                    "edit_fn": _patch_edit_fn(r["target"], patches, r["marker"]),
                    "is_applied": is_applied,
                }
            )
        return out


# ------------------------------------------------------------------- authored transform registry --
# Developer-authored de-hardcode transforms. Kept SMALL + honest: each targets a real absolute
# behaviour-arbitrating threshold; the LIVE gate is the arbiter of which are safe (a transform whose
# absolute cut encodes un-reconstructable global calibration rolls back and is learned-avoided). The
# stable ID order below fixes the RSTDPNeuron synapse index of each transform across restarts.
# ONE authored transform today. It is deliberately NOT pre-applied to the target: DehardcodeSource
# drains any transform whose marker is already present, so applying it by hand would empty the
# registry again and the proposer would go back to proposing zero. The BRAIN applies this one.
# Seven sibling candidates were authored and measured alongside it and every one was REJECTED by
# its own module self-check -- that is the honest-null this source was designed to produce, not a
# shortfall. The rejections are recorded in specs/019 [B2.03].
DEHARDCODE_TRANSFORMS = (
    {
        "id": "metacog_r_explore",
        "target": "src/snn_data_lake/faculties/metacognitive_monitor.py",
        "const": "r_explore",
        "current": 0.30,
        "rationale": (
            "retire the absolute `r_explore = 0.30` reliability cut for a parameter-free RunningBaseline of "
            "this pool's OWN windowed rate. Measured before authoring: at the live consumer "
            "faculties/brain_decision the low-side branch fired 100/100 (5 seeds x 20 cycles, reliability "
            "mean 0.1238, max 0.1849) -- the pool's rate NEVER clears 0.30 on that workload, so the exploit "
            "branch was structurally unreachable and 0.30 arbitrated nothing. Under the baseline it fires "
            "10/100. Precedent is in the same file: `r_hold` was already retired this way for `_rel_floor`."
        ),
        "marker": "selfbuild:dehardcode:metacog_r_explore",
        "patches": (
            (
                "\\ \\ \\*\\ explore\\(\\)\\ \\ \\ \\ \\ \\ \\->\\ bool:\\ explore\\ when\\ reliability\\ is\\ LOW\\ or\\ learnable\\-novelty\\ is\\\r\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ high;\\ exploit\\ when\\ reliability\\ is\\ high\\ and\\ the\\ context\\ familiar\\.\\\r\\\n",
                "  * explore()      -> bool: explore when reliability is UNUSUALLY LOW for this pool (one\r\n                      sample std below its OWN running mean) or learnable-novelty is high;\r\n                      exploit when reliability is ordinary and the context familiar. The bar\r\n                      is a running baseline, not a value: see __init__ for the measurement\r\n                      that retired the previous absolute `r_explore = 0.30` cut.\r\n",
                1,
            ),
            (
                "import\\ math\\\r\\\n\\\r\\\n",
                "import math\r\n\r\nfrom snn_data_lake.infra.brain_native_selfbuild import RunningBaseline\r\n",
                1,
            ),
            ("\\ \\ \\ \\ \\ \\ \\ \\ r_explore=0\\.30,\\\r\\\n", "", 1),
            (
                "\\ \\ \\ \\ \\ \\ \\ \\ self\\.r_explore\\ =\\ float\\(r_explore\\)\\ \\ \\#\\ reliability\\ below\\ this\\ \\->\\ explore\\\r\\\n",
                "        # (r_explore retired: the EXPLORE bar is now `_rel_lo`, a running baseline of this pool's rate)\r\n",
                1,
            ),
            (
                '\\ \\ \\ \\ \\ \\ \\ \\ self\\._rel_q\\ =\\ SignalCalibrator\\("metacog_reliability",\\ p=self\\.hold_p\\)\\\r\\\n',
                '        self._rel_q = SignalCalibrator("metacog_reliability", p=self.hold_p)\r\n        # DERIVED EXPLORE BAR -- retires the absolute `r_explore = 0.30` cut, which arbitrated on the\r\n        # VERY SAME reliability scale the HOLD bar above already abandoned, for the same reason and with\r\n        # a second, independent measurement behind it: `faculties/brain_decision.py` records that\r\n        # `monitor.explore()` returned True on EVERY measured cycle (20/20 x 5 seeds) because this pool\'s\r\n        # windowed rate never clears 0.30 on that workload. The exploit branch was UNREACHABLE and 0.30\r\n        # was arbitrating nothing there -- the same structural-saturation failure the no-evidence HOLD\r\n        # floor had, one sign flipped.\r\n        #\r\n        # `infra/brain_native_selfbuild.RunningBaseline` is parameter-free by construction:\r\n        # threshold() = running mean + ONE sample std, with no tuned k to smuggle a constant back in. It\r\n        # is an UPPER bar, so the low-side cut is taken by observing the NEGATED rate: -rel above\r\n        # threshold(-rel) is exactly rel more than one sample-std BELOW its own running mean, i.e.\r\n        # "unusually unreliable FOR THIS POOL". Scale-free like the HOLD quantile -- rescaling upstream\r\n        # confidence moves the samples and the bar together. threshold() is None below two samples and\r\n        # that is honoured as "do not fire on this baseline yet"; there is NO fallback to the retired\r\n        # literal, because a fallback would re-introduce the magic number through the back door.\r\n        self._rel_lo = RunningBaseline()\r\n',
                1,
            ),
            (
                "\\ \\ \\ \\ \\ \\ \\ \\ self\\._rel_q\\.observe\\(self\\._rel\\)\\ \\ \\#\\ fold\\ into\\ the\\ pool's\\ own\\ rate\\ distribution\\ \\(the\\ HOLD\\ bar\\)\\\r\\\n",
                "        self._rel_q.observe(self._rel)  # fold into the pool's own rate distribution (the HOLD bar)\r\n        self._rel_lo.observe(-self._rel)  # NEGATED rate -> the low-side EXPLORE bar (see __init__)\r\n",
                1,
            ),
            (
                '\\ \\ \\ \\ \\ \\ \\ \\ """Explore\\ when\\ reliability\\ is\\ LOW\\ or\\ learnable\\-novelty\\ is\\ high;\\ exploit\\ \\(False\\)\\\r\\\n\\ \\ \\ \\ \\ \\ \\ \\ when\\ reliability\\ is\\ high\\ and\\ the\\ context\\ is\\ familiar\\.\\ lesion\\ \\->\\ False\\ \\(ungated\\)\\."""\\\r\\\n',
                '        """Explore when reliability is UNUSUALLY LOW **for this pool** -- more than one sample\r\n        std below its own running mean -- or when learnable-novelty is high; exploit (False)\r\n        when reliability is ordinary and the context is familiar. The bar is `_rel_lo`, a\r\n        parameter-free RunningBaseline over the negated rate (see __init__ for why the absolute\r\n        0.30 cut it replaces was measured unreachable-in-one-direction). Below two samples the\r\n        baseline returns None and this branch does NOT fire -- never a fallback to the retired\r\n        literal. lesion -> False (ungated)."""\r\n',
                1,
            ),
            (
                "\\ \\ \\ \\ \\ \\ \\ \\ if\\ self\\._rel\\ <\\ self\\.r_explore:\\\r\\\n",
                "        lo = self._rel_lo.threshold()  # selfbuild:dehardcode:metacog_r_explore\r\n        if lo is not None and -self._rel > lo:\r\n",
                1,
            ),
        ),
    },
)

# All candidate IDs the learner has a synapse for (stable order == synapse index). Extend as the registry
# grows; existing persisted weights keep their index. A tiny fixed capacity avoids resizing the bank.
_ALL_IDS = tuple(s["id"] for s in DEHARDCODE_TRANSFORMS) + tuple(r["id"] for r in RECALIB_TARGETS)

# The pre-P4b home sidecar the learner's state is imported from once (see `_load`).
_LEGACY_STATE = "selfbuild_state.json"


# ================================================================================================
# THE PROPOSER -- one three-factor learner over all sources' candidates + the gated apply.
# ================================================================================================
class BrainNativeSelfBuildProposer:
    """Open-ended, three-factor-learned self-build proposer (replaces ImprovementProposer).

    propose(budget) collects applicable candidates from every source, ranks them by a LEARNED
    three-factor weight (RSTDPNeuron, the brain's own rule) with a running-spread-relative tie-break, and
    returns the top `budget` as {target, edit_fn, id, param, rationale} for SelfBuildLoop.run(). After the
    gated loop reports outcomes, observe(outcomes) feeds the GATE result back three-factor (kept +1 /
    rolled -1). Weights + tally persist to a durable JSON so learning survives restarts. Honest-null when
    no candidate is applicable -> propose() returns [] (never a forced/seeded change)."""

    def __init__(self, seed=0, repo=None, sources=None, lake_path=None, ids=None):
        self._repo = os.path.abspath(repo) if repo else _repo_root()
        self._seed = int(seed)
        self._sources = (
            list(sources)
            if sources is not None
            else [DehardcodeSource(), ThresholdRecalibrationSource(), DataIngestSource()]
        )
        self._ids = tuple(ids) if ids is not None else _ALL_IDS
        self._idx = {tid: i for i, tid in enumerate(self._ids)}
        n = max(1, len(self._ids))
        self._cell = RSTDPNeuron(n_syn=n, lr=_SELECT_LR, w0=_W0, seed=self._seed)
        self._cell.w = [
            _W0
        ] * n  # exact uniform init (drop RSTDPNeuron's +-0.05 jitter: symmetric start)
        # None -> THE ONE LAKE at its default location. A path is an ISOLATED lake, which is what the
        # self-check below uses per case; it is not a second store, it is the same store elsewhere.
        self._lake_path = lake_path
        self._tally = {"proposed": 0, "kept": 0, "rolled_back": 0}
        # CIRCUIT BREAKER (harness robustness principle): a candidate set that always rolls back is
        # re-proposed forever (byte-exact rollback -> is_applied stays False) and re-runs the FULL
        # multi-minute gate every cycle. Track consecutive futile cycles (proposed>0, kept==0) for the
        # SAME candidate set; should_skip() then lets the caller skip the expensive gate until the set
        # changes (a NEW commit / a newly-applicable candidate resets it -- so it is a cooldown on a
        # DETERMINISTICALLY-doomed set, never a permanent abandonment of transient-recoverable work).
        self._futile = 0
        self._last_cand_key = []
        self.log = []
        self._load()

    # -------- durable state (guard: weights autonomously update across restarts) --------
    #
    # P4b, 2026-08-04 -- THE ONE LAKE, not a sidecar. This learner's RSTDP weights are LEARNED
    # state: they are what the brain knows about which candidate sources are worth proposing, and
    # P4b says what the brain learns goes IN brain_lake.bin under its own record tag. Until this
    # change they went to `<repo>/data/selfbuild_state.json` (139 B on disk), which was wrong in
    # two ways that were not merely stylistic: the path was REPO-relative, so it ignored NORD_HOME
    # and two brains attached to different lakes silently shared one weight file; and `consolidate()`
    # could not see it, so sleep compaction reasoned about a lake missing part of what was learned.
    # Reads now come FROM the lake, which is the other half of P4b -- the in-memory `_cell.w` is a
    # CACHE of the newest _T_SELFBUILD record, never the source of truth.
    #
    # T120, 2026-10-25 -- ONE MIGRATION (P2). The pre-P4b file is a HOME sidecar now,
    # `<home>/selfbuild_state.json`, read through `home.load_state` (-> `_migrate_sidecar`, which
    # retires it `.imported`). The repo
    # copy `<repo>/data/selfbuild_state.json` is NEVER read: every NORD_HOME shared it, so every
    # temp-home brain imported it and none could retire it. The operator moves it into the home
    # once, before the first boot of this build (docs/RUNNING.md); a home whose lake holds no
    # _T_SELFBUILD record otherwise cold-starts. It holds real learned weights, so it is migrated,
    # never deleted. An isolated lake (`lake_path`) pairs with no home sidecar, mirroring
    # `load_named_state(path=...)`.
    def _load(self):
        st = blob = None
        read_failed = False
        try:
            blob = load_selfbuild(self._lake_path)
        except Exception:  # a broken or absent record must never block construction
            read_failed = True
        try:
            st = json.loads(bytes(blob).decode("ascii")) if blob else None
        except (ValueError, UnicodeDecodeError):
            pass
        if self._lake_path is None:
            # Asked on every default-lake boot, not only on a miss: the read is what retires a
            # sidecar moved into a home whose lake already holds a _T_SELFBUILD record.
            try:
                legacy = load_state(_LEGACY_STATE)
            except Exception:
                legacy = None
            # Only a lake with NO _T_SELFBUILD record takes it: the import leaves a frozen _T_STATE
            # copy nothing supersedes, and a corrupt or unreadable newest record cold-starts rather
            # than restoring those Aug-2026 weights over the learned history.
            if not blob and not read_failed and isinstance(legacy, dict):
                st = legacy
                try:
                    save_selfbuild(json.dumps(st, sort_keys=True).encode("ascii"))
                except Exception:  # best-effort; the in-memory restore below still happens
                    pass
        if not isinstance(st, dict):
            return
        for tid, w in (st.get("weights") or {}).items():
            if tid in self._idx and isinstance(w, (int, float)) and math.isfinite(w):
                self._cell.w[self._idx[tid]] = float(w)
        t = st.get("tally") or {}
        for k in self._tally:
            if isinstance(t.get(k), int):
                self._tally[k] = t[k]
        if isinstance(st.get("futile"), int):
            self._futile = st["futile"]
        if isinstance(st.get("last_cand_key"), list):
            self._last_cand_key = [str(x) for x in st["last_cand_key"]]

    def _save(self):
        weights = {tid: self._cell.w[self._idx[tid]] for tid in self._ids}
        payload = {
            "weights": weights,
            "tally": self._tally,
            "futile": self._futile,
            "last_cand_key": self._last_cand_key,
        }
        try:
            save_selfbuild(json.dumps(payload, sort_keys=True).encode("ascii"), self._lake_path)
        except Exception:  # persistence must never take a self-build cycle down
            pass

    # -------- introspection --------
    def weights(self):
        return {tid: self._cell.w[self._idx[tid]] for tid in self._ids}

    def state(self):
        return {
            "sources": [s.name for s in self._sources],
            "n_transforms": len(self._ids),
            "weights": self.weights(),
            "tally": dict(self._tally),
            "note": (
                "OPEN-ENDED three-factor-learned self-build; reward = gate kept(+1)/rolled(-1), never "
                "census; changes are running-baseline de-hardcodings; INVENTS NOTHING (authored menu); "
                "honest-null when no candidate applies. DataIngestSource is a stub (not implemented)."
            ),
        }

    # -------- the generator --------
    def _candidates(self):
        seen, cands = set(), []
        for src in self._sources:
            try:
                for c in src.candidates(self._repo):
                    if c["id"] in seen or _forbidden(c["target"]):
                        continue
                    seen.add(c["id"])
                    cands.append(c)
            except NotImplementedError as e:
                self.log.append((src.name, "not_implemented", str(e)[:80]))
        return cands

    def propose(self, budget=3):
        """Up to `budget` candidates, ranked by the LEARNED three-factor weight (desc), tie-broken by a
        deterministic crc jitter that is scaled to the RUNNING SPREAD of the weights (no fixed epsilon:
        while weights are ~uniform the jitter round-robins; once they diverge the learned order dominates).
        Returns [] (honest-null) when nothing is applicable. Pure fn of (sources, repo, learned weights)."""
        cands = self._candidates()
        if not cands:
            self.log.append(
                ("*", "honest_null", "no applicable candidate (nothing left to de-hardcode)")
            )
            return []
        w = self.weights()
        spread = _std([w[tid] for tid in self._ids]) if len(self._ids) > 1 else 0.0
        # jitter amplitude tracks the running weight spread; when spread==0 (cold start) it is the ONLY
        # differentiator -> deterministic round-robin (honestly reported), never a fixed exploration knob.
        jitter_amp = spread if spread > 1e-12 else 1.0
        # IMPACT ANALYSIS (the harness LSP principle, native): each candidate's PRE-MUTATION blast radius
        # (reasoning/impact_analysis -- graphify dep edges + ast symbol refs) makes the self-build PREFER
        # SURGICAL changes (fewer dependents affected) as a SECONDARY key after the learned weight, relative
        # to the candidate set (no fixed threshold). Advisory: the full gate + byte-exact rollback stay the
        # safety backstop; this just consumes graphify's cross-file edges the self-build was ignoring.
        from snn_data_lake.reasoning.impact_analysis import blast_radius

        for c in cands:
            try:
                c["blast_radius"] = blast_radius(
                    repo_root=self._repo, target_rel=c["target"], symbol=c.get("symbol")
                )["radius"]
            except Exception:
                c["blast_radius"] = 0
        ranked = sorted(
            cands,
            key=lambda c: (
                -w.get(c["id"], _W0),
                c.get(
                    "blast_radius", 0
                ),  # surgical preference: fewer dependents affected ranks first
                -(zlib.crc32(f"{self._seed}|{c['id']}".encode("ascii", "replace")) % 100003)
                * 1e-9
                * jitter_amp,
            ),
        )
        out = ranked[: max(int(budget), 0)]
        self._tally["proposed"] += len(out)
        for c in out:
            c["param"] = c["id"]  # SelfBuildLoop.run summary key
        self.log.append(
            (
                "*",
                "proposed",
                f"{len(out)}/{len(cands)} (spread {spread:.4g}, blast {[c.get('blast_radius') for c in out]})",
            )
        )
        self._save()
        return out

    def observe(self, outcomes):
        """Three-factor update from GATE outcomes. outcomes: list of {id/param, status}. reward = +1 kept,
        -1 rolled_back (or refused/gate_error); other statuses = 0 (no learning signal). Reuses RSTDPNeuron:
        set the eligibility to a competitive one-hot (elig_k - mean_elig) and gate the reward into w, then
        clip to [0,1]. Weights persist -> the learner autonomously updates across restarts."""
        n = len(self._ids)
        for o in outcomes or ():
            tid = o.get("id") or o.get("param")
            if tid not in self._idx:
                continue
            status = o.get("status")
            if status == "kept":
                reward = _REWARD_KEPT
                self._tally["kept"] += 1
            elif status in ("rolled_back", "refused", "gate_error", "sandbox_rejected"):
                reward = _REWARD_ROLLED
                self._tally["rolled_back"] += 1
            else:
                continue  # disabled/skipped -> no learning signal
            k = self._idx[tid]
            elig = [1.0 if j == k else 0.0 for j in range(n)]
            mean_e = sum(elig) / n
            self._cell.e = [
                elig[j] - mean_e for j in range(n)
            ]  # competitive mean-eligibility baseline
            self._cell.reward(reward)  # w[j] += lr*reward*(elig_j - mean_e)
            self._cell.w = [min(1.0, max(0.0, wj)) for wj in self._cell.w]  # clip to [0,1]
            self.log.append((tid, status, f"reward {reward:+.0f} -> w {self._cell.w[k]:.4f}"))
        self._save()

    # -------- circuit breaker (harness robustness): skip the expensive gate on a doomed candidate set --
    def should_skip(self, proposed):
        """True if this EXACT candidate set has already been proposed-and-all-rolled-back _FUTILE_LIMIT
        consecutive times -> skip the multi-minute gate (cooldown). Reset when the set changes (a new
        commit / newly-applicable candidate), so a deterministically-doomed set is paused, never a
        transient-recoverable one permanently abandoned. Cheap: compares the (already-computed) proposal
        ids, runs no gate."""
        key = sorted(p.get("id") for p in (proposed or ()))
        return bool(key) and key == self._last_cand_key and self._futile >= _FUTILE_LIMIT

    def note_cycle(self, proposed, outcomes):
        """Update the futile-cycle counter AFTER a gated cycle. A NEW candidate set resets the counter
        (the situation changed -> worth retrying); the SAME set with zero kept increments it; any kept
        resets it. Persisted, so the cooldown survives daemon restarts."""
        key = sorted(p.get("id") for p in (proposed or ()))
        if key != self._last_cand_key:
            self._futile = 0
            self._last_cand_key = key
        attempted = len(proposed or ())
        kept = sum(1 for o in (outcomes or ()) if o.get("status") == "kept")
        if kept > 0:
            self._futile = 0
        elif attempted > 0:
            self._futile += 1
        self.log.append(("*", "cycle", f"attempted {attempted} kept {kept} futile {self._futile}"))
        self._save()

    # -------- guard G3: net-hardcode ledger --------
    def ledger(self):
        """NET absolute-threshold ledger: how many of the registry's target constants are still fixed
        absolutes vs de-hardcoded (marker present in the real tree). net_removed must be >= 0; a de-hardcode
        that merely moves a constant elsewhere would not increase it."""
        remaining = dehardcoded = 0
        for src in self._sources:
            remaining += len(src.candidates(self._repo))  # applicable == still absolute
            if isinstance(src, DehardcodeSource):
                for s in src._specs:
                    if _is_applied_fn(s["target"], s["marker"])(self._repo):
                        dehardcoded += 1
        total = remaining + dehardcoded
        return {
            "absolute_remaining": remaining,
            "dehardcoded": dehardcoded,
            "net_removed": dehardcoded,
            "total_registry": total,
        }


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ================================================================================================
# SELF-CHECK -- proves the mechanism end-to-end on a TOY repo with a FAKE gate (no real module touched,
# no multi-minute gate). Exercises: gate-safe de-hardcode KEPT + gate-breaking rolled-back; three-factor
# weight DIVERGENCE from uniform (G4); net-ledger drop (G3); RunningBaseline boundaries (G6); honest-null
# (G2); firewall rejects protected/off-limits/self-referential targets (G5).
# ================================================================================================
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def main():  # noqa: C901 - a single linear self-check kept in one place on purpose
    import shutil
    import tempfile

    width = 100
    _emit("=" * width)
    _emit(
        "BRAIN-NATIVE SELF-BUILD -- open-ended, three-factor-learned, gate-firewalled (self-check)"
    )
    _emit("=" * width)
    _emit(
        "HONEST: reward = GATE outcome (never census); changes are running-baseline de-hardcodings;"
    )
    _emit(
        "INVENTS NOTHING (authored menu, learner re-orders); honest-null is the goal state, reported."
    )
    # DERIVED, never a literal: the invocation line and the self-path the firewall protects both come
    # from __file__/__spec__, so relocating this module updates them instead of staling them.
    _emit(f"  module: {_MODNAME}   self-path (firewall-protected): {_SELF_REL}")
    _emit(f"  run:    PYTHONUTF8=1 PYTHONPATH=src python -m {_MODNAME}")
    _emit("-" * width)

    # ---- guard G6: RunningBaseline boundary semantics ------------------------------------------------
    rb = RunningBaseline()
    assert rb.threshold() is None and not rb.ready(), (
        "empty history must gate NOTHING (None), not a constant"
    )
    rb.observe(5.0)
    assert rb.threshold() is None, (
        "single sample must still be None (variance undefined) -- no smuggled seed"
    )
    rb2 = RunningBaseline()
    for _ in range(4):
        rb2.observe(3.0)  # zero-variance stream
    assert rb2.threshold() == 3.0, "zero-variance -> threshold == mean (std 0); finite, no NaN"
    rb3 = RunningBaseline()
    for x in (0.0, 0.0, 10.0, 10.0):
        rb3.observe(x)
    thr = rb3.threshold()
    assert thr is not None and 5.0 < thr < 12.0, (
        f"mean+std must sit above the mean on a spread stream ({thr})"
    )
    _emit(
        "  [OK] G6 RunningBaseline boundaries: empty/single -> None (no smuggled constant); "
        "zero-var -> mean; spread -> mean+std finite"
    )

    # ---- build a TOY repo: a module whose ABSOLUTE threshold CAN be de-hardcoded gate-safe -----------
    # The toy's self-check exercises the gating signal ACROSS ITS RANGE (0,1,2 quiet; 8,9,10 salient), so a
    # running-baseline (mean+std over the observed stream) reproduces "the salient ones fire" -- proving a
    # gate-safe de-hardcode IS possible when the operating range is exercised (unlike a globally-calibrated
    # absolute cut probed at a single point -- see the module docstring's locus_coeruleus finding).
    # PROMOTION HAZARD, fixed here: once promoted this self-check runs IN-PROCESS under pytest, so
    # setting NORD_SELFMOD=1 and then os.environ.pop()-ing it in `finally` would leave an operator's
    # PRE-EXISTING NORD_SELFMOD unset for every later test in the same process. Capture the prior value
    # BEFORE the try (so the finally can never NameError) and restore exactly it.
    _prev_selfmod = os.environ.get("NORD_SELFMOD")
    work = tempfile.mkdtemp(prefix="brain_native_selfbuild_selfcheck_")
    try:
        fake = os.path.join(work, "fakerepo")
        rel_good = "src/snn_data_lake/faculties/_toy_gate.py"
        rel_bad = "src/snn_data_lake/faculties/_toy_pinned.py"
        for rel, body in ((rel_good, _TOY_GOOD_SRC), (rel_bad, _TOY_BAD_SRC)):
            p = os.path.join(fake, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(body)

        transforms = (
            {
                "id": "toy_gate.THRESH",
                "target": rel_good,
                "const": "THRESH",
                "current": "5.0",
                "rationale": "toy: replace absolute THRESH=5.0 with a running-baseline (mean+std) gate",
                "marker": "# DEHARDCODED:toy_gate",
                "patches": _TOY_GOOD_PATCHES,
            },
            {
                "id": "toy_pinned.CUT",
                "target": rel_bad,
                "const": "CUT",
                "current": "0.5",
                "rationale": "toy: CUT=0.5 encodes a GLOBAL point a per-call baseline cannot reconstruct",
                "marker": "# DEHARDCODED:toy_pinned",
                "patches": _TOY_BAD_PATCHES,
            },
        )
        ids = tuple(t["id"] for t in transforms)
        src = DehardcodeSource(transforms=transforms)
        lake_path = os.path.join(work, "state_lake.bin")

        def gate_ok(root):
            """FAKE gate: exec the (possibly patched) toy modules under `root` and run their check().
            The patched toys import the REAL RunningBaseline from this package (already imported)."""
            ok = True
            for rel in (rel_good, rel_bad):
                path = os.path.join(root, *rel.split("/"))
                with open(path, encoding="utf-8", newline="") as f:
                    code = f.read()
                ns = {"__name__": "toy", "__file__": path}
                try:
                    exec(compile(code, path, "exec"), ns)  # noqa: S102 - sandboxed toy in a temp dir
                    ns["check"]()
                except Exception:  # noqa: BLE001 - any toy failure == a red gate
                    ok = False
            return ok

        # ---- drive promote() directly (propose -> apply-to-real-toy -> gate_fn -> keep/rollback ->
        # observe): the harness sandbox pre-verify (SelfBuildLoop.run) needs a real run_selfchecks in the
        # tree, which it has its own self-check for; here we exercise the PROPOSER + promote + learning. ----
        from snn_data_lake.infra.self_build_loop import SelfBuildLoop

        os.environ["NORD_SELFMOD"] = "1"
        w_trajectory = []
        prop = BrainNativeSelfBuildProposer(
            seed=0, repo=fake, sources=[src], lake_path=lake_path, ids=ids
        )
        loop = SelfBuildLoop(repo_root=fake, enable=True, gate_fn=lambda: gate_ok(fake))
        w_trajectory.append(dict(prop.weights()))
        for _round in range(4):
            props = prop.propose(budget=2)
            if not props:
                break
            outcomes = [{"id": c["id"], **loop.promote(c["edit_fn"], c["target"])} for c in props]
            prop.observe(outcomes)
            w_trajectory.append(dict(prop.weights()))

        # ---- guard checks on the run --------------------------------------------------------------------
        w_final = prop.weights()
        # G3 net ledger: the gate-safe toy got de-hardcoded (marker in real tree) -> net_removed >= 1
        led = prop.ledger()
        assert led["net_removed"] >= 1, f"a gate-safe de-hardcode must be applied (ledger {led})"
        good_applied = _is_applied_fn(rel_good, "# DEHARDCODED:toy_gate")(fake)
        bad_applied = _is_applied_fn(rel_bad, "# DEHARDCODED:toy_pinned")(fake)
        assert good_applied, "the gate-SAFE toy de-hardcode must be KEPT (its self-check survives)"
        assert not bad_applied, (
            "the gate-BREAKING toy de-hardcode must be ROLLED BACK (self-check fails)"
        )
        # G4 divergence: the safe transform's weight must RISE above the broken one's, and away from uniform
        assert w_final["toy_gate.THRESH"] > w_final["toy_pinned.CUT"], (
            f"three-factor must prefer the gate-safe transform ({w_final})"
        )
        assert w_final["toy_gate.THRESH"] > _W0 > w_final["toy_pinned.CUT"], (
            "weights must DIVERGE from the 0.1 uniform init (kept rises, rolled falls) -- not round-robin"
        )
        assert prop._tally["kept"] >= 1 and prop._tally["rolled_back"] >= 1, (
            "both outcomes must be exercised"
        )
        _emit(
            f"  [OK] G1/G4 three-factor learned from the GATE: kept={prop._tally['kept']} "
            f"rolled={prop._tally['rolled_back']}; w toy_gate.THRESH {w_final['toy_gate.THRESH']:.3f} "
            f"> 0.1 > toy_pinned.CUT {w_final['toy_pinned.CUT']:.3f} (diverged, not round-robin)"
        )
        _emit(
            f"  [OK] G3 ledger: net_removed={led['net_removed']} (gate-safe de-hardcode applied; "
            f"gate-breaking one rolled back byte-exact)"
        )
        # weight trajectory (evidence the weights MOVE, not stay uniform)
        traj = " -> ".join(
            f"({r['toy_gate.THRESH']:.2f},{r['toy_pinned.CUT']:.2f})" for r in w_trajectory
        )
        _emit(f"       weight trajectory (toy_gate,toy_pinned): {traj}")

        # ---- G2 honest-null: once the safe one is applied + the broken one keeps failing, drain to null --
        # apply the safe transform is already done; force the broken to be 'applied' to simulate exhaustion
        for s in transforms:
            _patch_edit_fn(
                s["target"], s["patches"], s["marker"]
            )  # (only builds closures; no write)
        # mark the broken as applied by hand to simulate 'nothing left'
        bp = os.path.join(fake, *rel_bad.split("/"))
        with open(bp, encoding="utf-8", newline="") as f:
            bad_src = f.read()
        with open(bp, "w", encoding="utf-8", newline="") as f:
            f.write("# DEHARDCODED:toy_pinned\n" + bad_src)
        prop2 = BrainNativeSelfBuildProposer(
            seed=0,
            repo=fake,
            sources=[DehardcodeSource(transforms=transforms)],
            lake_path=lake_path,
            ids=ids,
        )
        assert prop2.propose(budget=3) == [], (
            "honest-null: no applicable candidate -> propose() returns []"
        )
        assert any(s == "honest_null" for (_a, s, _b) in prop2.log), (
            "honest-null must be logged, not faked"
        )
        _emit(
            "  [OK] G2 honest-null: when nothing is left to de-hardcode, propose() returns [] (logged), "
            "no seed/warm-up constant injected to force a change"
        )

        # ---- G5 firewall: protected / off-limits / self-referential targets are REJECTED ----------------
        # The self-entry is _SELF_REL (DERIVED from __file__), not a literal path. Every case tagged
        # NOT-AT-THAT-PATH below names a file that does NOT exist where it is written -- they are the
        # reachable RED for the basename widening in _forbidden: revert it to full-path matching and
        # they go legal, which is precisely the "firewall goes stale on a move and still passes green"
        # failure being closed. (Two of them, the `reasoning/` pair, were the REAL paths when this list
        # was written and became relocated copies when the region wave moved the files -- which is the
        # same lesson arriving by itself, and is why the tags are stated rather than assumed.)
        bad_targets = (
            next(iter(_PROTECTED)),  # a brainstem safety file
            "run_selfchecks.py",
            "src/snn_data_lake/reasoning/self_mod_harness.py",  # harness, NOT-AT-THAT-PATH
            "src/snn_data_lake/reasoning/self_build_loop.py",  # accept path, NOT-AT-THAT-PATH
            _SELF_REL,  # itself, at wherever it actually lives right now
            "src/snn_data_lake/faculties/self_improvement_suite.py",  # the census scorer
            "src/../../evil.py",
            "src/snn_data_lake/brainstem/reflex.py",
            "src/snn_data_lake/reasoning/brain_native_selfbuild.py",  # this file, NOT-AT-THAT-PATH
            "src/snn_data_lake/faculties/self_build_loop.py",  # accept path, NOT-AT-THAT-PATH
            "src/snn_data_lake/infra/self_mod_harness.py",  # the harness at its REAL path
        )
        rejected = 0
        for bad_target in bad_targets:
            assert _forbidden(bad_target), f"{bad_target} must be forbidden"
            try:
                DehardcodeSource(
                    transforms=(
                        {
                            "id": "x",
                            "target": bad_target,
                            "const": "C",
                            "current": "1",
                            "rationale": "",
                            "marker": "# m",
                            "patches": (),
                        },
                    )
                )
            except ValueError:
                rejected += 1
        assert rejected == len(bad_targets), (
            f"every illegal target must be rejected at construction ({rejected}/{len(bad_targets)})"
        )
        # and the apply-time closure firewall fires even on a forged edit_fn:
        forged = _patch_edit_fn("src/snn_data_lake/brainstem/reflex.py", (), "# m")
        fired = False
        try:
            forged(fake)
        except RuntimeError:
            fired = True
        assert fired, "a forged edit_fn onto a brainstem file must raise BEFORE any write"
        _emit(
            f"  [OK] G5 firewall: {rejected}/{len(bad_targets)} illegal targets rejected at construction "
            "(protected, scorer, accept-path, self via DERIVED _SELF_REL, run_selfchecks, and FOUR "
            "targets naming a protected module at a path it does not live at -- the reachable red for "
            "basename matching); forged brainstem edit_fn raised at apply time"
        )

        # ---- determinism ------------------------------------------------------------------------------
        # P8, NO REACHABLE RED (fixed, not weakened): by this point BOTH toy transforms are applied, so
        # the previous form of this check compared propose(3) == propose(3) with both sides structurally
        # []. It would have passed with the entire ranking deleted. Re-marking the transforms makes them
        # applicable again, so the comparison is now between two REAL orderings -- and the length assert
        # below is the guard that stops it from silently going vacuous again.
        det_transforms = tuple({**t, "marker": t["marker"] + ":det"} for t in transforms)
        pa = BrainNativeSelfBuildProposer(
            seed=0,
            repo=fake,
            sources=[DehardcodeSource(transforms=det_transforms)],
            lake_path=os.path.join(work, "s2_lake.bin"),
            ids=ids,
        )
        pb = BrainNativeSelfBuildProposer(
            seed=0,
            repo=fake,
            sources=[DehardcodeSource(transforms=det_transforms)],
            lake_path=os.path.join(work, "s3_lake.bin"),
            ids=ids,
        )
        order_a = [c["id"] for c in pa.propose(3)]
        order_b = [c["id"] for c in pb.propose(3)]
        assert len(order_a) == len(det_transforms), (
            f"determinism must be checked on a NON-EMPTY candidate set, got {order_a}"
        )
        assert order_a == order_b, f"same seed -> same order ({order_a} vs {order_b})"
        _emit(
            f"  [OK] deterministic: same seed reproduces the ranked order over {len(order_a)} live "
            f"candidates {order_a} (checked on a NON-EMPTY set: [] == [] proves nothing)"
        )

        # ---- ThresholdRecalibrationSource: honest-null + propose + firewall (SYNTHETIC target) ----
        # RECALIB_TARGETS now ships EMPTY (salience_th moved to live consumption in brain.py), so the
        # mechanism is exercised against a SYNTHETIC recorded-signal threshold via the recalibs= override.
        from snn_data_lake.substrate.signal_calibrator import SignalCalibrator

        # Reachable red: author any entry into RECALIB_TARGETS and this fails, which is what forces the
        # synthetic-target block below to be re-read rather than silently bypassed.
        assert RECALIB_TARGETS == (), (
            "default recalib targets drain to empty (salience_th is live-consumed)"
        )
        # P8, DELETED (not weakened): an
        #     assert ThresholdRecalibrationSource().candidates(fake) == []
        # stood here. With RECALIB_TARGETS == () asserted on the line above, candidates() iterates an
        # empty tuple and returns [] unconditionally -- no constructible input makes it red. The one
        # thing it did exercise, that substrate/signal_calibrator still imports, is covered with a real
        # red by rc.candidates(fake) a few lines down, which runs the same import on a live target.
        synth = {
            "id": "recalib:probe_th",
            "signal": "probe_th",
            "target": "src/snn_data_lake/reasoning/_selfbuild_recalib_probe.py",  # fake, non-protected
            "const": "PROBE_TH",
            "orig": "0.50",
            "marker": "# DATA-CALIBRATED:probe",
            "lo": 0.02,
            "hi": 0.98,
        }
        # An ISOLATED LAKE, not a temp sidecar (P4b, 2026-08-05). The isolation is not optional and it
        # is not cosmetic: with `lake_path=None` this block would write `probe_th` records INTO the
        # operator's real brain_lake.bin on every gate run, and the honest-null asserts below would be
        # contaminable by a `probe_th` left there by a previous run -- i.e. they would go green for a
        # reason that has nothing to do with the mechanism under test.
        calib_lake = os.path.join(work, "calib_lake.bin")
        rc = ThresholdRecalibrationSource(recalibs=(synth,), lake_path=calib_lake)
        assert rc.candidates(fake) == [], (
            "recalibration must honest-null with NO recorded data (empty lake)"
        )
        sc = SignalCalibrator("probe_th", p=0.75)
        for _ in range(10):
            sc.observe(0.4)  # below MIN_SAMPLES
        SignalCalibrator.save_all([sc], calib_lake)
        assert rc.candidates(fake) == [], (
            "must honest-null below MIN_SAMPLES (no smuggled seed constant)"
        )
        # >= MIN_SAMPLES of a real distribution -> it proposes a data-calibrated recalibration
        sc2 = SignalCalibrator("probe_th", p=0.75)
        for i in range(200):
            sc2.observe(0.30 + 0.40 * ((i % 5) / 4.0))  # spread ~[0.3,0.7], P75 ~0.6
        SignalCalibrator.save_all([sc2], calib_lake)
        rel = synth["target"]
        dst = os.path.join(fake, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write("PROBE_TH = 0.50  # drive >= this is salient\n")
        cands = rc.candidates(fake)
        assert len(cands) == 1 and cands[0]["id"] == "recalib:probe_th", cands
        cands[0]["edit_fn"](fake)  # apply to the fake target tree
        with open(dst, encoding="utf-8", newline="") as f:
            newlc = f.read()
        assert "# DATA-CALIBRATED:probe" in newlc and "= 0.50 " not in newlc, newlc
        assert _is_applied_fn(rel, "# DATA-CALIBRATED:probe")(fake), (
            "the marker must mark it applied"
        )
        assert rc.candidates(fake) == [], (
            "an APPLIED recalibration must not be re-proposed (drains to null)"
        )
        rej = 0
        for bad in ("src/snn_data_lake/brainstem/reflex.py", "run_selfchecks.py"):
            try:
                ThresholdRecalibrationSource(
                    recalibs=(
                        {
                            "id": "x",
                            "signal": "s",
                            "target": bad,
                            "const": "C",
                            "orig": "1",
                            "marker": "# m",
                            "lo": 0.0,
                            "hi": 1.0,
                        },
                    )
                )
            except ValueError:
                rej += 1
        assert rej == 2, (
            "recalib targeting a protected/brainstem file must be rejected at construction"
        )
        _emit(
            "  [OK] ThresholdRecalibrationSource: default targets empty (salience_th now live-consumed);"
        )
        _emit(
            "       on a SYNTHETIC target it honest-nulls with no/insufficient data (no seed), then once"
        )
        _emit(
            "       >= MIN_SAMPLES proposes a data-calibrated literal + marker; applied -> not re-proposed;"
        )
        _emit("       firewall rejects brainstem/protected targets")

        # ---- CIRCUIT BREAKER (harness robustness): a doomed set skips the gate after _FUTILE_LIMIT ------
        # (should_skip/note_cycle read only p["id"], so a synthetic candidate set exercises the logic
        # directly -- the toy transforms are already applied by the tests above, so propose() is empty.)
        cb = BrainNativeSelfBuildProposer(
            seed=0, repo=fake, sources=[], lake_path=os.path.join(work, "cb_lake.bin"), ids=ids
        )
        doomed = [{"id": ids[0]}, {"id": ids[1]}]
        assert not cb.should_skip(doomed), "a fresh candidate set must NOT skip the gate"
        rolled = [{"id": c["id"], "status": "rolled_back"} for c in doomed]
        for _ in range(_FUTILE_LIMIT):
            cb.note_cycle(doomed, rolled)  # all rolled back, same set
        assert cb.should_skip(doomed), (
            f"a set rolled back {_FUTILE_LIMIT}x in a row must skip the gate"
        )
        cb.note_cycle(doomed, [{"id": doomed[0]["id"], "status": "kept"}])
        assert not cb.should_skip(doomed), "a KEPT outcome must reset the breaker"
        for _ in range(_FUTILE_LIMIT):
            cb.note_cycle(doomed, rolled)
        assert cb.should_skip(doomed), "re-armed after another run of futile cycles"
        # (a candidate id that is deliberately NOT in `doomed`; the previous literal here named
        # "recalib:salience_th", a target retired when salience_th moved to live consumption)
        assert not cb.should_skip([{"id": "a-different-candidate-id"}]), (
            "a CHANGED candidate set must NOT skip (the situation changed -> worth retrying)"
        )
        _emit(
            "  [OK] circuit breaker: a doomed set skips the multi-minute gate after _FUTILE_LIMIT futile"
        )
        _emit(
            "       cycles; a KEPT outcome or a CHANGED candidate set resets it (cooldown, not permanent)"
        )

    finally:
        if _prev_selfmod is None:
            os.environ.pop("NORD_SELFMOD", None)
        else:
            os.environ["NORD_SELFMOD"] = _prev_selfmod
        shutil.rmtree(work, ignore_errors=True)

    _emit("-" * width)
    _emit(
        "self-check OK: open-ended source-pluggable self-build; three-factor learner (brain's own rule)"
    )
    _emit(
        "learns from the GATE (kept+1/rolled-1, never census) which authored de-hardcode to apply;"
    )
    _emit(
        "gate-safe de-hardcode KEPT, gate-breaking rolled back byte-exact; weights diverge from uniform;"
    )
    _emit(
        "net absolute-threshold ledger drops; honest-null when exhausted; 4-layer firewall holds."
    )
    _emit(
        "HONEST: invents nothing (authored menu); DataIngestSource (real-world/internet learning) is a"
    )
    _emit(
        "documented NotImplemented extension point, not built. An honest substrate, not unbounded AGI."
    )
    return 0


# ---- toy modules for the self-check (kept as module-level source strings for readability) ------------
# gate-SAFE: THRESH gates on a signal whose BASELINE stream (0,1,2) is exercised in check(); replacing it
# with a RunningBaseline over that baseline stream (mean+std = 1+1 = 2) reproduces "baseline not salient,
# high salient" -> the module self-check survives -> KEPT. (Proves a gate-safe de-hardcode is possible.)
_TOY_GOOD_SRC = (
    "THRESH = 5.0  # absolute gating threshold; the baseline stream 0,1,2 is exercised in check()\n"
    "def salient(x):\n"
    "    return x > THRESH\n"
    "def check():\n"
    "    baseline = [0.0, 1.0, 2.0]\n"
    "    high = [8.0, 9.0, 10.0]\n"
    "    for x in baseline:\n"
    "        assert not salient(x), 'baseline must not be salient'\n"
    "    for x in high:\n"
    "        assert salient(x), 'high must be salient'\n"
)
_TOY_GOOD_PATCHES = (
    (
        r"^THRESH = 5\.0  # absolute gating threshold; the baseline stream 0,1,2 is exercised in check\(\)$",
        # DERIVED import path (_MODNAME): the toy the fake gate execs imports RunningBaseline back out
        # of THIS module, so a hardcoded dotted path here breaks the toy gate the moment the file moves.
        "# DEHARDCODED:toy_gate  (absolute THRESH -> running-baseline mean+std)\n"
        f"from {_MODNAME} import RunningBaseline as _RB\n"
        "_BASE = _RB()",
        1,
    ),
    (
        r"^def salient\(x\):\n    return x > THRESH$",
        "def salient(x):\n    return _BASE.threshold() is not None and x > _BASE.threshold()",
        1,
    ),
    (
        r"^    baseline = \[0\.0, 1\.0, 2\.0\]\n    high = \[8\.0, 9\.0, 10\.0\]$",
        "    baseline = [0.0, 1.0, 2.0]\n"
        "    high = [8.0, 9.0, 10.0]\n"
        "    for x in baseline:\n"
        "        _BASE.observe(x)",
        1,
    ),
)
# gate-BREAKING de-hardcode: CUT encodes a GLOBAL point; the check probes a single instance that only ever
# sees small values, so a per-call baseline cannot reconstruct the 0.5 cut -> the swap changes the decision
# -> check() fails -> rolled back. (Models the locus_coeruleus.SALIENCE_TH finding.)
_TOY_BAD_SRC = (
    "CUT = 0.5  # a GLOBALLY-calibrated cut: 0.2 is 'small', 1.0 is 'large' (global knowledge)\n"
    "def act(v):\n"
    "    return v >= CUT\n"
    "def check():\n"
    "    assert not act(0.2), 'a small value must not act'\n"
    "    assert act(1.0), 'a large value must act'\n"
)
_TOY_BAD_PATCHES = (
    (
        r"^CUT = 0\.5  # a GLOBALLY-calibrated cut: 0\.2 is 'small', 1\.0 is 'large' \(global knowledge\)$",
        "# DEHARDCODED:toy_pinned\n"
        f"from {_MODNAME} import RunningBaseline as _RB\n"  # DERIVED, see _TOY_GOOD_PATCHES
        "_B = _RB()",
        1,
    ),
    (
        r"^def act\(v\):\n    return v >= CUT$",
        "def act(v):\n"
        "    _B.observe(v)\n"
        "    return _B.threshold() is not None and v >= _B.threshold()",
        1,
    ),
)


if __name__ == "__main__":
    import sys

    sys.exit(main())
