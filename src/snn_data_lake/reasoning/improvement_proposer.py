r"""
AUTONOMOUS IMPROVEMENT PROPOSER -- the 'what to try next' generator for the self-mod harness.

WHAT THIS IS (honest framing, read before believing any "autonomous self-improvement" claim):
  The existing self_mod_harness.SelfModHarness is a DISABLED-by-default, sandboxed, verifier-gated
  TESTER of candidate edits -- but the candidate edit itself (the edit_fn) has always been
  CALLER-SUPPLIED: the generative "what should I try?" intelligence was human. This module closes
  exactly that gap and NOTHING more: given a metrics snapshot (e.g. the brain's
  self_improvement_suite.census() dict) and a developer-authored whitelist of TUNABLE numeric
  parameters, it autonomously generates RANKED candidate improvement proposals. Each proposal is
    {rationale, target_metric, target (file), param, current, proposed, bounds, direction,
     score, edit_fn}
  where edit_fn(sandbox_root) rewrites ONLY the one whitelisted module-level numeric assignment,
  to a value clamped inside its developer-set safe bounds -- shaped exactly for
  SelfModHarness.propose(target, edit_fn, utility_fn) to sandbox-test.

WHAT THIS IS NOT (do not overclaim):
  * NOT open-ended code invention. The proposal space is parameter tweaks over a finite,
    developer-whitelisted set of numeric knobs with hard bounds. No new code, no new logic,
    no structural edits. The rationale strings are heuristic HYPOTHESES (direction hints),
    not causal knowledge -- the harness's sandbox gate is the only judge.
  * NOT an applier, NOT an enabler. This module NEVER writes to the real tree (the only write
    path is the returned edit_fn, which the HARNESS invokes inside ITS temp sandbox copy),
    NEVER constructs or enables a SelfModHarness, and NEVER touches the harness's protected set.
  * NOT a learner. Fully deterministic stdlib heuristics; no gradient, no plasticity rule.

SAFETY (the firewall this module itself enforces, on top of the harness's own):
  * The tunable whitelist is validated at CONSTRUCTION: any spec whose file is one of the
    harness's 9 PROTECTED files (5 brainstem arm-safety files + run_selfchecks.py +
    godel_machine.py + self_improve.py + self_mod_harness.py), or lives under brainstem/, or
    escapes the repo (absolute / '..' / drive-letter paths), raises ValueError immediately.
  * propose() RE-validates every spec per call (defense against post-construction tampering
    with the whitelist) and silently refuses -- with an audit-log entry -- anything illegal.
  * Every emitted edit_fn re-checks the firewall AT APPLY TIME inside the closure: if the
    captured path is protected or under brainstem/, it raises RuntimeError before touching
    any file. Three independent layers; the harness's crc32 firewall is a fourth.
  * This module imports NOTHING from brainstem/ (asserted by the self-check, which scans this
    file's own import lines). The _PROTECTED constant is imported from self_mod_harness
    (a reasoning/ sibling with side-effect-free imports) so the two firewalls
    can never diverge.

Determinism & style invariants (per CLAUDE.md): no import-time side effects; tie-break
jitter via zlib.crc32 (never builtin hash); no wall-clock in any returned value; ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.improvement_proposer
"""
import math
import os
import re
import zlib

from snn_data_lake.reasoning.self_mod_harness import _PROTECTED

# Census failure sentinel (self_improvement_suite.SENTINEL). A metric at/below this is a crashed
# rung, not a real measurement -- never a proposal target.
_SENTINEL_FLOOR = -998.0

# Candidate value ladder per direction hint: multiplicative factors applied to the CURRENT value,
# then clamped into the spec's [lo, hi]. "up" = the developer believes more usually helps the
# target metric (typical for training-budget knobs); "down" the reverse; "both" = unknown a priori.
_FACTORS = {
    "up":   (2.0, 1.5, 1.25),
    "down": (0.5, 0.75, 0.8),
    "both": (2.0, 0.5, 1.5, 0.75),
}
_MAX_PER_PARAM = 3        # at most this many distinct candidate values per parameter
_HEADROOM_CAP = 10.0      # sanity cap on normalized headroom (keeps scores finite/sane)

_NUM_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

# ----------------------------------------------------------------------- default tunable space ----
# The developer-authored whitelist: the reduced-scale census budget knobs of
# cortex/self_improvement_suite.py (each maps 1:1 to a census metric) plus two readout-training
# knobs of cortex/autotelic_goals.py. All are plain module-level numeric assignments in NON-safety
# files. growth_exact_diff is deliberately ABSENT: it is an exact invariant (must stay 0.0), not a
# metric with headroom -- nothing may be tuned "toward" it.
_SUITE = "src/snn_data_lake/cortex/self_improvement_suite.py"
_AUTOTELIC = "src/snn_data_lake/cortex/autotelic_goals.py"

DEFAULT_TUNABLES = (
    {"file": _SUITE, "name": "_GOAL_EPISODES", "kind": "int", "lo": 2, "hi": 48,
     "target_metric": "goal_reach", "goal": "max", "goal_value": 1.0, "scale": 1.0, "hint": "up"},
    {"file": _SUITE, "name": "_QD_BUDGET", "kind": "int", "lo": 20, "hi": 240,
     "target_metric": "qd_coverage", "goal": "max", "goal_value": 30.0, "scale": 30.0, "hint": "up"},
    {"file": _SUITE, "name": "_QD_INIT", "kind": "int", "lo": 5, "hi": 60,
     "target_metric": "qd_coverage", "goal": "max", "goal_value": 30.0, "scale": 30.0, "hint": "up"},
    {"file": _SUITE, "name": "_BF_A", "kind": "int", "lo": 50, "hi": 600,
     "target_metric": "consolidation_retention", "goal": "max", "goal_value": 1.0, "scale": 1.0, "hint": "up"},
    {"file": _SUITE, "name": "_BF_RELAX", "kind": "int", "lo": 10, "hi": 300,
     "target_metric": "consolidation_retention", "goal": "max", "goal_value": 1.0, "scale": 1.0, "hint": "both"},
    {"file": _SUITE, "name": "_DR_EPISODES", "kind": "int", "lo": 1, "hi": 24,
     "target_metric": "drive_regulation", "goal": "min", "goal_value": 0.0, "scale": 6.0, "hint": "up"},
    {"file": _SUITE, "name": "_DR_EPSTEPS", "kind": "int", "lo": 4, "hi": 32,
     "target_metric": "drive_regulation", "goal": "min", "goal_value": 0.0, "scale": 6.0, "hint": "up"},
    {"file": _SUITE, "name": "_MP_EPOCHS", "kind": "int", "lo": 1, "hi": 8,
     "target_metric": "metaplast_gain", "goal": "max", "goal_value": 0.5, "scale": 0.5, "hint": "up"},
    {"file": _SUITE, "name": "_MP_WIN", "kind": "int", "lo": 4, "hi": 32,
     "target_metric": "metaplast_gain", "goal": "max", "goal_value": 0.5, "scale": 0.5, "hint": "both"},
    {"file": _AUTOTELIC, "name": "_EPOCHS", "kind": "int", "lo": 8, "hi": 64,
     "target_metric": "goal_reach", "goal": "max", "goal_value": 1.0, "scale": 1.0, "hint": "up"},
    {"file": _AUTOTELIC, "name": "_LR", "kind": "float", "lo": 0.005, "hi": 0.08,
     "target_metric": "goal_reach", "goal": "max", "goal_value": 1.0, "scale": 1.0, "hint": "both"},
)

_SPEC_KEYS = ("file", "name", "kind", "lo", "hi", "target_metric", "goal", "goal_value", "scale", "hint")


def _repo_root():
    # improvement_proposer.py -> reasoning -> snn_data_lake -> src -> REPO ROOT
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _check_path_legal(relpath, protected):
    """The path firewall. Raises ValueError unless `relpath` is a clean repo-relative POSIX path
    that is NOT protected and NOT under brainstem/. Shared by construction-time validation,
    per-call re-validation, and (as RuntimeError) the apply-time closure check."""
    if not isinstance(relpath, str) or not relpath:
        raise ValueError("file must be a non-empty repo-relative POSIX path string")
    if "\\" in relpath or relpath.startswith("/") or ":" in relpath:
        raise ValueError(f"file must be repo-relative POSIX (no abs path / drive / backslash): {relpath!r}")
    parts = relpath.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"file must not contain empty/./..' segments: {relpath!r}")
    if relpath in protected:
        raise ValueError(f"FIREWALL: {relpath} is a PROTECTED safety/verifier file -- never tunable")
    if "brainstem" in parts:
        raise ValueError(f"FIREWALL: {relpath} is under brainstem/ (safety envelope) -- never tunable")
    if not relpath.endswith(".py") or parts[:2] != ["src", "snn_data_lake"]:
        raise ValueError(f"tunable file must be a .py under src/snn_data_lake/: {relpath!r}")


def _validate_spec(spec, protected):
    """Full validation of one tunable-parameter spec. Raises ValueError on ANY problem."""
    if not isinstance(spec, dict):
        raise ValueError("tunable spec must be a dict")
    missing = [k for k in _SPEC_KEYS if k not in spec]
    if missing:
        raise ValueError(f"tunable spec missing keys: {missing}")
    _check_path_legal(spec["file"], protected)
    name = spec["name"]
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"param name must be a Python identifier: {name!r}")
    if spec["kind"] not in ("int", "float"):
        raise ValueError(f"kind must be 'int' or 'float': {spec['kind']!r}")
    lo, hi = spec["lo"], spec["hi"]
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))
            and math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"bounds must be finite with lo < hi: lo={lo!r} hi={hi!r}")
    if not isinstance(spec["target_metric"], str) or not spec["target_metric"]:
        raise ValueError("target_metric must be a non-empty string")
    if spec["goal"] not in ("max", "min"):
        raise ValueError(f"goal must be 'max' or 'min': {spec['goal']!r}")
    if not (isinstance(spec["goal_value"], (int, float)) and math.isfinite(spec["goal_value"])):
        raise ValueError(f"goal_value must be a finite number: {spec['goal_value']!r}")
    if not (isinstance(spec["scale"], (int, float)) and math.isfinite(spec["scale"]) and spec["scale"] > 0):
        raise ValueError(f"scale must be a finite positive number: {spec['scale']!r}")
    if spec["hint"] not in _FACTORS:
        raise ValueError(f"hint must be one of {sorted(_FACTORS)}: {spec['hint']!r}")


def _assign_pattern(name):
    """Regex matching exactly the module-level `NAME = <number>` line (optional trailing comment,
    CR-safe). The name is anchored by the following `\\s*=` so `_MP_EPOCH` can never match
    `_MP_EPOCHS = 2` and vice versa."""
    return re.compile(rf"^({re.escape(name)}\s*=\s*)({_NUM_RE})((?:\s*(?:#.*)?)?)$", re.M)


def _read_current(repo, relpath, name, kind):
    """Read (read-only) the current numeric value of `name` in the REAL repo file. Returns the
    parsed value, or raises ValueError if the file is missing or the assignment does not occur
    exactly once (an ambiguous target is never proposed)."""
    path = os.path.join(repo, *relpath.split("/"))
    if not os.path.isfile(path):
        raise ValueError(f"tunable file not found: {relpath}")
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    hits = _assign_pattern(name).findall(src)
    if len(hits) != 1:
        raise ValueError(f"assignment '{name} = <number>' must occur exactly once in {relpath} (found {len(hits)})")
    text = hits[0][1]
    return int(text) if kind == "int" else float(text)


def _fmt_value(value, kind):
    """Deterministic source text for the new value."""
    if kind == "int":
        return str(int(value))
    return repr(float(f"{float(value):.6g}"))   # trim float noise (0.015000000000000001 -> 0.015)


def _make_edit_fn(relpath, name, new_text, protected):
    """Build the sandbox-only edit closure. It rewrites ONLY the single `name = <number>`
    assignment in `relpath` under the root the HARNESS passes in (its temp sandbox copy).
    APPLY-TIME FIREWALL: re-checks the captured path against the protected set / brainstem
    before touching anything -- even a tampered whitelist cannot route a write into the
    safety envelope through this closure."""
    pattern = _assign_pattern(name)

    def edit_fn(sandbox_root):
        try:
            _check_path_legal(relpath, protected)
        except ValueError as e:                       # promote: an illegal apply is a hard failure
            raise RuntimeError(f"edit_fn refused at apply time: {e}") from e
        path = os.path.join(sandbox_root, *relpath.split("/"))
        with open(path, encoding="utf-8", newline="") as f:
            src = f.read()
        out, n = pattern.subn(lambda m: m.group(1) + new_text + m.group(3), src, count=2)
        if n != 1:
            raise RuntimeError(f"edit_fn: '{name} = <number>' not unique in sandbox copy of {relpath} (n={n})")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(out)

    return edit_fn


class ImprovementProposer:
    """Autonomous 'what to try next' generator for SelfModHarness -- STRICTLY A PROPOSER.

    Given a metrics snapshot and a developer-whitelisted set of tunable numeric parameters,
    propose(metrics, budget) returns ranked candidate proposals; each carries a sandbox-only
    edit_fn shaped for SelfModHarness.propose(candidate['target'], candidate['edit_fn'], ...).
    It never writes to the real tree, never constructs/enables a harness, and refuses any
    parameter in a protected/brainstem file at three layers (construction, per-call, apply).
    Deterministic: same tunables + same metrics + same seed -> byte-identical proposal list.
    """

    def __init__(self, tunables=None, seed=0, repo=None):
        self._repo = os.path.abspath(repo) if repo else _repo_root()
        self._protected = frozenset(_PROTECTED)
        self._seed = int(seed)
        specs = tuple(tunables) if tunables is not None else DEFAULT_TUNABLES
        for spec in specs:                            # construction-time firewall: ValueError on ANY bad spec
            _validate_spec(spec, self._protected)
        self._tunables = specs
        self.log = []                                 # (file, param, status, reason) -- the audit trail

    # ------------------------------------------------------------------ introspection ----

    def state(self):
        return {
            "tunables": [(s["file"], s["name"], s["target_metric"]) for s in self._tunables],
            "protected": sorted(self._protected),
            "note": ("PROPOSER ONLY: generates ranked parameter-tweak candidates for the DISABLED-by-"
                     "default SelfModHarness to sandbox-test. Never writes the real tree, never enables "
                     "the harness, never targets a protected/brainstem file."),
        }

    # ------------------------------------------------------------------ the generator ----

    def _headroom(self, spec, metrics):
        """Normalized distance-to-goal of the spec's target metric, or None if the metric is
        absent / non-finite / the census failure sentinel / already at goal."""
        m = metrics.get(spec["target_metric"])
        if not isinstance(m, (int, float)) or not math.isfinite(m) or m <= _SENTINEL_FLOOR:
            return None
        if spec["goal"] == "max":
            h = (spec["goal_value"] - m) / spec["scale"]
        else:
            h = (m - spec["goal_value"]) / spec["scale"]
        if h <= 1e-12:
            return None                               # goal met/beyond -> nothing to improve here
        return min(h, _HEADROOM_CAP)

    def _candidate_values(self, spec, current):
        """The bounded candidate ladder for one parameter: hint-ordered factors on the current
        value, clamped into [lo, hi], deduplicated, never equal to the current value."""
        lo, hi, kind = spec["lo"], spec["hi"], spec["kind"]
        out = []
        for factor in _FACTORS[spec["hint"]]:
            v = current * factor
            v = lo if v < lo else (hi if v > hi else v)
            if kind == "int":
                v = int(round(v))
                v = int(lo) if v < lo else (int(hi) if v > hi else v)
            else:
                v = float(f"{float(v):.6g}")
            if v != current and v not in out:
                out.append(v)
            if len(out) >= _MAX_PER_PARAM:
                break
        return out

    def propose(self, metrics, budget=8):
        """Generate up to `budget` ranked candidate improvement proposals from `metrics`.

        Each candidate = {rationale, target_metric, target, param, current, proposed, bounds,
        direction, score, edit_fn}; edit_fn(sandbox_root) is sandbox-only and firewalled.
        Ranking: normalized headroom of the target metric (worst metric first), then the hint
        order, then a deterministic crc32 jitter (stable tie-break; never builtin hash).
        Pure function of (tunables, metrics, seed, current repo param values) -- deterministic.
        """
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dict of name -> number")
        budget = int(budget)
        candidates = []
        for idx, spec in enumerate(self._tunables):
            try:                                      # per-call re-validation (anti-tamper, layer 2)
                _validate_spec(spec, self._protected)
            except ValueError as e:
                self.log.append((str(spec.get("file")), str(spec.get("name")), "refused", str(e)))
                continue
            h = self._headroom(spec, metrics)
            if h is None:
                self.log.append((spec["file"], spec["name"], "skipped",
                                 f"no headroom on {spec['target_metric']} (absent/at-goal/sentinel)"))
                continue
            try:
                current = _read_current(self._repo, spec["file"], spec["name"], spec["kind"])
            except ValueError as e:
                self.log.append((spec["file"], spec["name"], "skipped", str(e)))
                continue
            for rank, proposed in enumerate(self._candidate_values(spec, current)):
                key = f"{self._seed}|{spec['file']}|{spec['name']}|{proposed!r}"
                jitter = (zlib.crc32(key.encode("ascii", "replace")) % 997) * 1e-9
                score = h + (len(_FACTORS[spec["hint"]]) - rank) * 1e-4 + jitter
                direction = "raise" if proposed > current else "lower"
                m = metrics[spec["target_metric"]]
                rationale = (
                    f"{spec['target_metric']}={m:.4g} vs goal {spec['goal']}~{spec['goal_value']:g} "
                    f"(headroom {h:.3g}): {direction} {spec['name']} {current!r} -> {proposed!r} in "
                    f"{spec['file']} (safe bounds [{spec['lo']:g}, {spec['hi']:g}]); hypothesis only -- "
                    f"the sandbox harness gate decides."
                )
                candidates.append({
                    "rationale": rationale,
                    "target_metric": spec["target_metric"],
                    "target": spec["file"],
                    "param": spec["name"],
                    "current": current,
                    "proposed": proposed,
                    "bounds": (spec["lo"], spec["hi"]),
                    "direction": direction,
                    "score": score,
                    "_order": (idx, rank),            # stable secondary key (deterministic sort)
                    "edit_fn": _make_edit_fn(spec["file"], spec["name"],
                                             _fmt_value(proposed, spec["kind"]), self._protected),
                })
        candidates.sort(key=lambda c: (-c["score"], c["_order"]))
        for c in candidates:
            del c["_order"]
        out = candidates[:max(budget, 0)]
        self.log.append(("*", "*", "proposed", f"{len(out)}/{len(candidates)} candidates (budget {budget})"))
        return out


# ------------------------------------------------------------------- the self-check ---------------

def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def _parse_all_nums(path):
    """(name -> literal text) of every module-level numeric assignment in a file (self-check helper)."""
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    return dict(re.findall(rf"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*({_NUM_RE})\s*(?:#.*)?$", src, re.M))


def _crc_file(path):
    with open(path, "rb") as f:
        return zlib.crc32(f.read())


def _summary(cands):
    """Closure-free deterministic summary of a proposal list (for equality checks)."""
    return [(c["target"], c["param"], c["proposed"], c["target_metric"], round(c["score"], 12)) for c in cands]


def main():
    import random
    import shutil
    import tempfile

    width = 100
    _emit("=" * width)
    _emit("IMPROVEMENT PROPOSER -- autonomous 'what to try next' generator for the self-mod harness")
    _emit("=" * width)
    _emit("HONEST: parameter tweaks over a developer-whitelisted numeric space with hard bounds --")
    _emit("NOT open-ended code invention. Proposals are HYPOTHESES; only the DISABLED-by-default")
    _emit("sandbox harness (never enabled here) can test them. Nothing below writes the real tree.")
    _emit("-" * width)

    work = tempfile.mkdtemp(prefix="improvement_proposer_selfcheck_")
    try:
        # ---- build a throwaway fake repo with one tunable toy module -------------------------------
        fake = os.path.join(work, "fakerepo")
        toy_rel = "src/snn_data_lake/reasoning/toy_knobs.py"
        toy_path = os.path.join(fake, *toy_rel.split("/"))
        os.makedirs(os.path.dirname(toy_path))
        toy_src = (
            "_P0 = 10        # knob 0\n_P1 = 10   # knob 1\n_P2 = 10\n_P3 = 10\n_P4 = 10\n"
            "_ALPHA = 10     # int knob\n_BETA = 0.5  # float knob\n_GAMMA = 3\nOTHER = 99\n"
        )
        with open(toy_path, "w", encoding="utf-8", newline="") as f:
            f.write(toy_src)

        def spec(name, metric, kind="int", lo=2, hi=40, hint="up", goal="max", gv=1.0, scale=1.0):
            return {"file": toy_rel, "name": name, "kind": kind, "lo": lo, "hi": hi,
                    "target_metric": metric, "goal": goal, "goal_value": gv, "scale": scale, "hint": hint}

        tunables = (
            spec("_ALPHA", "m_alpha"),
            spec("_BETA", "m_beta", kind="float", lo=0.05, hi=2.0, hint="both"),
            spec("_GAMMA", "m_absent"),               # its metric is never in the snapshot
        )
        metrics = {"m_alpha": 0.2, "m_beta": 0.9, "m_other": 0.0}
        prop = ImprovementProposer(tunables=tunables, seed=0, repo=fake)
        cands = prop.propose(metrics, budget=8)

        # ---- (1) BOUNDS + REAL-METRIC targeting -----------------------------------------------------
        assert cands, "proposer must generate candidates when metrics have headroom"
        for c in cands:
            lo, hi = c["bounds"]
            assert lo <= c["proposed"] <= hi, f"proposal out of safe bounds: {c['param']}={c['proposed']}"
            assert c["proposed"] != c["current"], "a proposal must actually change the parameter"
            assert c["target_metric"] in metrics, f"proposal targets a metric absent from the snapshot: {c}"
            assert c["param"] != "_GAMMA", "_GAMMA's metric is absent -> it must generate NO proposal"
        assert cands[0]["target_metric"] == "m_alpha", \
            "ranking must put the max-headroom metric first (m_alpha headroom 0.8 > m_beta 0.1)"
        _emit(f"  [OK] {len(cands)} proposals: all inside safe bounds, all target REAL snapshot metrics,")
        _emit("       absent-metric param proposed nothing, top-ranked targets the worst metric (m_alpha)")

        # ---- (2) edit_fn changes ONLY the intended param, in a throwaway copy only ------------------
        sandbox = os.path.join(work, "sandbox")
        shutil.copytree(fake, sandbox)
        crc_real_before = _crc_file(toy_path)
        before = _parse_all_nums(os.path.join(sandbox, *toy_rel.split("/")))
        top = cands[0]
        top["edit_fn"](sandbox)                        # <- the ONLY write path, and only into the copy
        after = _parse_all_nums(os.path.join(sandbox, *toy_rel.split("/")))
        changed = sorted(k for k in before if before[k] != after.get(k))
        assert changed == [top["param"]], f"edit_fn must change ONLY {top['param']} (changed: {changed})"
        assert float(after[top["param"]]) == float(top["proposed"]), "edit_fn must write exactly the proposed value"
        assert _crc_file(toy_path) == crc_real_before, "the real (fake-repo) tree must be byte-unchanged"
        # float param path too:
        cb = next(c for c in cands if c["param"] == "_BETA")
        cb["edit_fn"](sandbox)
        after2 = _parse_all_nums(os.path.join(sandbox, *toy_rel.split("/")))
        assert float(after2["_BETA"]) == float(cb["proposed"]) and after2["OTHER"] == "99"
        _emit(f"  [OK] edit_fn on a throwaway copy changed ONLY {top['param']} "
              f"({before[top['param']]} -> {after[top['param']]}); float path OK; real tree crc unchanged")

        # ---- (3) FAILING CONTROLS: safety/non-whitelisted params are REJECTED -----------------------
        rejected = 0
        bad_specs = (
            spec("_X", "m")                                            | {"file": "src/snn_data_lake/brainstem/reflex.py"},
            spec("_X", "m")                                            | {"file": "run_selfchecks.py"},
            spec("_X", "m")                                            | {"file": "src/snn_data_lake/reasoning/self_mod_harness.py"},
            spec("_X", "m")                                            | {"file": "src/snn_data_lake/reasoning/self_improve.py"},
            spec("_X", "m")                                            | {"file": "src/../../evil.py"},
            spec("_X", "m")                                            | {"file": "/abs/path.py"},
            spec("_ALPHA", "m_alpha", lo=40, hi=2),                    # inverted bounds
            spec("_ALPHA", "m_alpha")                                  | {"goal": "sideways"},
        )
        for bad in bad_specs:
            try:
                ImprovementProposer(tunables=(bad,), repo=fake)
            except ValueError:
                rejected += 1
        assert rejected == len(bad_specs), f"every illegal spec must be REJECTED ({rejected}/{len(bad_specs)})"
        # layer 2: post-construction whitelist tamper -> propose() refuses, emits nothing illegal
        tampered = ImprovementProposer(tunables=(spec("_ALPHA", "m_alpha"),), repo=fake)
        tampered._tunables = (spec("_X", "m_alpha") | {"file": "src/snn_data_lake/brainstem/reflex.py"},)
        t_cands = tampered.propose({"m_alpha": 0.1}, budget=8)
        assert t_cands == [], "a tampered-in brainstem spec must be refused at propose() time"
        assert any(s == "refused" for (_f, _n, s, _r) in tampered.log), "the refusal must be audit-logged"
        # layer 3: apply-time closure firewall -- even a directly-forged edit_fn refuses to write
        forged = _make_edit_fn("src/snn_data_lake/brainstem/reflex.py", "_X", "1", frozenset(_PROTECTED))
        fired = False
        try:
            forged(sandbox)
        except RuntimeError:
            fired = True
        assert fired, "a forged edit_fn onto a brainstem file must raise BEFORE touching anything"
        assert not os.path.exists(os.path.join(sandbox, "src", "snn_data_lake", "brainstem", "reflex.py")), \
            "the forged edit_fn must not have created/written the safety file"
        _emit(f"  [OK] CONTROLS: {rejected}/{len(bad_specs)} illegal specs rejected at construction;")
        _emit("       tampered whitelist refused at propose() (audit-logged); forged edit_fn onto")
        _emit("       brainstem/reflex.py raised at APPLY time without writing (3-layer firewall)")

        # ---- (4) DETERMINISM: same inputs -> byte-identical proposal list ---------------------------
        s1 = _summary(ImprovementProposer(tunables=tunables, seed=0, repo=fake).propose(metrics, 8))
        s2 = _summary(ImprovementProposer(tunables=tunables, seed=0, repo=fake).propose(metrics, 8))
        assert s1 == s2 == _summary(cands), "same tunables+metrics+seed must reproduce proposals exactly"
        _emit("  [OK] deterministic: two fresh instances reproduce the identical ranked list")

        # ---- (5) RANKING-QUALITY metric vs RANDOM + SPEC-ORDER controls (~chance) -------------------
        n_trials, n_metrics = 20, 5
        five = tuple(spec(f"_P{i}", f"m{i}") for i in range(n_metrics))
        hits_real = hits_rand = hits_order = 0
        for t in range(n_trials):
            rng = random.Random(t)                     # per-trial seeded RNG (control machinery only)
            worst = rng.randrange(n_metrics)
            snap = {f"m{i}": (0.05 if i == worst else rng.uniform(0.80, 0.95)) for i in range(n_metrics)}
            ranked = ImprovementProposer(tunables=five, seed=t, repo=fake).propose(snap, budget=32)
            hits_real += 1 if ranked and ranked[0]["target_metric"] == f"m{worst}" else 0
            shuffled = list(ranked)
            random.Random(1000 + t).shuffle(shuffled)  # RANDOM control: metric-blind ranking
            hits_rand += 1 if shuffled and shuffled[0]["target_metric"] == f"m{worst}" else 0
            in_spec_order = sorted(ranked, key=lambda c: c["param"])   # IDENTITY control: fixed order
            hits_order += 1 if in_spec_order and in_spec_order[0]["target_metric"] == f"m{worst}" else 0
        r_real, r_rand, r_order = (hits_real / n_trials, hits_rand / n_trials, hits_order / n_trials)
        _emit(f"  [METRIC] top-1 targets the max-headroom metric: proposer {r_real:.2f} "
              f"({hits_real}/{n_trials})  vs controls: random-rank {r_rand:.2f}, "
              f"spec-order {r_order:.2f} (chance ~= {1 / n_metrics:.2f})")
        assert r_real >= 0.95, f"the proposer must rank the needy metric first ({r_real:.2f})"
        assert r_rand <= 0.5 and r_order <= 0.5, \
            f"controls must sit near chance ({r_rand:.2f}, {r_order:.2f}) -- else the metric is riggable"
        assert r_real > r_rand + 0.4 and r_real > r_order + 0.4, "the proposer must clearly beat both controls"

        # ---- (6) REAL REPO + REAL census metrics (integration smoke, read-only) --------------------
        from snn_data_lake.cortex.self_improvement_suite import census
        repo = _repo_root()
        watched = [os.path.join(repo, *p.split("/")) for p in
                   ("src/snn_data_lake/cortex/self_improvement_suite.py",
                    "src/snn_data_lake/cortex/autotelic_goals.py",
                    "src/snn_data_lake/brainstem/reflex.py", "run_selfchecks.py")]
        crc0 = {p: _crc_file(p) for p in watched if os.path.isfile(p)}
        snap = census(0)                               # the brain's REAL self-improvement metric dict
        real = ImprovementProposer(seed=0)             # DEFAULT_TUNABLES against the real repo
        real_cands = real.propose(snap, budget=6)
        assert real_cands, "the real census(0) snapshot must yield at least one proposal (metrics below goal)"
        for c in real_cands:
            lo, hi = c["bounds"]
            assert lo <= c["proposed"] <= hi and c["target_metric"] in snap
            assert c["target"] not in _PROTECTED and "brainstem" not in c["target"]
        for p, v in crc0.items():
            assert _crc_file(p) == v, f"proposing must be READ-ONLY on the real tree: {p}"
        _emit(f"  [OK] REAL census(0) snapshot -> {len(real_cands)} ranked proposals (budget 6), all bounded,")
        _emit("       all target real census metrics; real tree crc-verified READ-ONLY. Top 3:")
        for c in real_cands[:3]:
            _emit(f"       - {c['rationale']}")

        # ---- (7) NO-ACTUATOR: this module imports nothing from the safety envelope ------------------
        with open(__file__, encoding="utf-8") as fh:
            import_lines = [ln for ln in fh.read().splitlines()
                            if ln.startswith("import ") or ln.startswith("from ")]
        forbidden = ("brainstem", "serial_arm", "arm_sim", "arm_config", "interpreter")
        hits = [tok for tok in forbidden if any(tok in ln for ln in import_lines)]
        assert hits == [], f"proposer must import NOTHING from brainstem/arm (found {hits})"
        _emit(f"  [OK] NO-ACTUATOR: forbidden-import hits={hits}; harness never constructed or enabled here")

    finally:
        shutil.rmtree(work, ignore_errors=True)

    _emit("-" * width)
    _emit("self-check OK: autonomous PROPOSER generates ranked, bounded, deterministic parameter-tweak")
    _emit("candidates targeting real metrics (top-1 alignment 1.00 vs ~chance controls); 3-layer firewall")
    _emit("rejects every safety/non-whitelisted target; the only write path is the sandbox-only edit_fn.")
    _emit("HONEST: a whitelist-bounded tuner, not open-ended self-improvement; rationales are hypotheses;")
    _emit("the DISABLED-by-default harness (never enabled here) remains the sole judge and applier-of-nothing.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
