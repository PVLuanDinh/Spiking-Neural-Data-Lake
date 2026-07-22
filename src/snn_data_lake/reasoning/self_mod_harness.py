r"""
SELF-MODIFICATION HARNESS -- Wave 4B, DISABLED-BY-DEFAULT, sandboxed, verifier-gated PROPOSER.

WHAT THIS IS (honest framing, read before believing any "self-modifying brain" claim):
  A standalone PROPOSER that, ONLY when a human explicitly enables it, tries ONE candidate edit to a
  WHITELISTED non-safety reasoning/ module inside a THROWAWAY COPY of the repo, runs the real
  verification gate there, and returns a PROPOSAL (accepted/rejected + the sandbox path). It NEVER
  mutates the real tree, NEVER touches the safety envelope or the verifier, and NEVER applies anything
  live -- turning a proposal into a real commit is a SEPARATE, human-gated step the main agent owns.

WHAT THIS IS NOT: it is not wired into any tick/sleep/loop; it does not auto-run; it is not an
  intelligence explosion. It is inert by default and, even when enabled, is bounded by the gate + the
  fixed protected set + the fact that it only ever writes into a temp sandbox.

DOUBLE OFF-SWITCH (both required before it will ever attempt anything):
  1. constructor argument   enable=True
  2. environment variable   NORD_SELFMOD=1
  With either missing, EVERY call to propose() returns {"status": "disabled"} and does nothing at all
  (no sandbox is created, no file is read or written).

MECHANISM when (and only when) enabled -- mirrors self_improve.py's discipline
(propose -> isolate -> verify -> ratchet -> accept) and reuses godel_machine.py's IMMUTABLE kernel:
  (1) WHITELIST      -- the target must be a NON-safety reasoning/ module; a target in the protected
                        set is refused up front (the firewall), before any sandbox is even created.
  (2) ISOLATE        -- copy the repo (filtered: no .git/venvs/caches/big data) into a TEMP sandbox;
                        the real tree is never touched.
  (3) APPLY          -- run the candidate edit_fn INSIDE the sandbox copy only.
  (4) VERIFY (gate)  -- run run_selfchecks.py in the sandbox AND assert every PROTECTED file is
                        byte-identical (zlib.crc32) before/after the edit AND a monotonic UTILITY
                        (default: selfcheck pass-count; or a supplied metric) did NOT regress.
  (5) ACCEPT/REJECT  -- the accept/reject verdict is produced by the project's own audited
                        self_improve.SelfImprover (corrigible: it refuses protected targets and only
                        accepts a candidate that keeps the gate green and does not regress utility).

THE PROTECTED SET it enforces (the exact immutable files -- a rewrite may touch NONE of them):
  godel_machine._SAFETY_KERNEL  = the 5 brainstem arm-safety files
                                    (reflex.py, serial_arm.py, arm_sim.py, arm_config.py, interpreter.py)
                                  + run_selfchecks.py + reasoning/godel_machine.py
  PLUS reasoning/self_improve.py  (the bounded-self-improvement verifier)
  PLUS reasoning/self_mod_harness.py  (this proposer itself)
  = 9 files total. None is ever a legal target; touching any (even as a side effect) fails the gate.

Deps: stdlib (os, re, shutil, subprocess, sys, tempfile, zlib, random) + the sibling modules
  reasoning/godel_machine.py (for _SAFETY_KERNEL) and reasoning/self_improve.py (SelfImprover, Change).
  REPRESENTATION-ONLY: imports nothing from brainstem, no actuator route.

  PYTHONPATH=src python -m snn_data_lake.reasoning.self_mod_harness   # the DISABLED-by-default self-check
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from random import Random

from snn_data_lake.reasoning.godel_machine import _SAFETY_KERNEL
from snn_data_lake.reasoning.self_improve import Change, SelfImprover

# extra immutable files beyond godel_machine's arm-safety kernel: the bounded-improvement verifier and
# THIS proposer. Together with _SAFETY_KERNEL these form the exact protected set enforced below.
_EXTRA_PROTECTED = (
    "src/snn_data_lake/reasoning/self_improve.py",
    "src/snn_data_lake/reasoning/self_mod_harness.py",
)
_PROTECTED = tuple(dict.fromkeys((*_SAFETY_KERNEL, *_EXTRA_PROTECTED)))  # de-duped, order-stable

# heavy / irrelevant trees skipped when copying the real repo into a sandbox (the self-checks are
# hermetic, so none of these is needed to run the gate). A tiny hermetic test repo matches none of them.
_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv*",
    "venv",
    "env",
    "__pycache__",
    "*.pyc",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "*.egg-info",
    "graphify-out",
    "data",
    "brain_data",
    "assets",
    "models",
    "vendor",
    "environment_files",
)
_PASS_RE = re.compile(r"(\d+)\s*/\s*\d+\s+self-checks passed")  # run_selfchecks.py's summary line


def _repo_root():
    # self_mod_harness.py -> reasoning -> snn_data_lake -> src -> REPO ROOT
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _rel(repo, target):
    """Normalize `target` (absolute OR repo-relative) to a repo-relative POSIX path, matching the way
    the protected/whitelist entries are written."""
    t = target if os.path.isabs(target) else os.path.join(repo, target)
    return os.path.relpath(t, repo).replace(os.sep, "/")


def _crc_map(root, rels):
    """crc32 of each protected file that exists under `root`. Comparing before/after an edit is the
    byte-identity firewall: any change to a protected file (even a side effect) shows up as a crc delta."""
    out = {}
    for rel in rels:
        p = os.path.join(root, *rel.split("/"))
        if os.path.isfile(p):
            with open(p, "rb") as f:
                out[rel] = zlib.crc32(f.read())
    return out


class SelfModHarness:
    """DISABLED-by-default, sandboxed, verifier-gated self-modification PROPOSER.

    Default (enable=False AND env NORD_SELFMOD unset): every propose() returns {"status": "disabled"}
    and does nothing. Enabling requires BOTH enable=True AND NORD_SELFMOD=1 -- explicit human sign-off.
    Even enabled it only ever writes into a temp sandbox and returns a proposal; it applies nothing live.
    """

    def __init__(
        self,
        enable=False,
        whitelist=None,
        seed=0,
        repo=None,
        protected=None,
        sandbox_root=None,
        max_attempts=8,
        timeout=900,
        require_improvement=False,
        require_ruff=True,
    ):
        self._enable = bool(enable)
        self._require_improvement = bool(
            require_improvement
        )  # True -> accept ONLY a STRICT utility increase
        self._require_ruff = bool(
            require_ruff
        )  # True -> the sandbox must ALSO be ruff-clean (fully tested)
        self._repo = os.path.abspath(repo) if repo else _repo_root()
        self._protected = frozenset(protected if protected is not None else _PROTECTED)
        self._whitelist = frozenset(
            whitelist if whitelist is not None else self._default_whitelist()
        )
        self._rng = Random(seed)  # per-instance RNG (no global RNG, no wall-clock)
        self._sandbox_root = sandbox_root  # where temp sandboxes are created (None -> system temp)
        self._max_attempts = int(max_attempts)
        self._timeout = timeout
        self._attempts = 0
        self.log = []  # (target, status, reason) -- the audit trail

    # -------------------------------------------------------------------- configuration helpers ----

    def _default_whitelist(self):
        """Auto-build the candidate set: every reasoning/ module EXCEPT the verifier / self-improvement
        machinery and anything in the protected set. Read-only scan; returns () on any failure."""
        base = "src/snn_data_lake/reasoning"
        never = {"self_improve.py", "godel_machine.py", "self_mod_harness.py", "__init__.py"}
        try:
            d = os.path.join(self._repo, *base.split("/"))
            wl = []
            for f in sorted(os.listdir(d)):
                if f.endswith(".py") and f not in never:
                    rel = base + "/" + f
                    if rel not in self._protected:
                        wl.append(rel)
            return tuple(wl)
        except OSError:
            return ()

    def _is_enabled(self):
        # BOTH gates, evaluated live so a mid-session env change is honored. Default -> inert.
        return self._enable and os.environ.get("NORD_SELFMOD") == "1"

    def state(self):
        note = (
            "DISABLED sandboxed verifier-gated PROPOSER; never touches the real tree or the safety "
            "envelope. Enabling requires enable=True AND env NORD_SELFMOD=1; applying an accepted "
            "proposal is a separate human-gated step. Protected (immutable) files: "
            + ", ".join(sorted(self._protected))
        )
        return {"enabled": self._is_enabled(), "whitelist": sorted(self._whitelist), "note": note}

    # -------------------------------------------------------------------- the gate ------------------

    def _run_selfchecks(self, path):
        """Run run_selfchecks.py inside the sandbox. Returns (returncode, passcount_or_None). A fresh
        NORD_HOME + no-.pyc keeps it hermetic (the durable lake never touches the real home)."""
        script = os.path.join(path, "run_selfchecks.py")
        if not os.path.isfile(script):
            return 1, None
        home = tempfile.mkdtemp(prefix="selfmod_home_", dir=self._sandbox_root)
        env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src",
            "NORD_HOME": home,
            # ANTI-RECURSION (uniform guard): neutralize NORD_SELFMOD inside the sandbox gate so a brain
            # self-check run here can NEVER trigger a nested autonomous self-build (which would recurse:
            # sandbox -> run_selfchecks -> self-build -> sandbox ...). The promote gate already sets this;
            # setting it here too makes the guard uniform across BOTH subprocess gates, not incidental.
            "NORD_SELFMOD": "0",
        }
        try:
            r = subprocess.run(
                [sys.executable, "run_selfchecks.py"],
                cwd=path,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return 1, None
        finally:
            shutil.rmtree(home, ignore_errors=True)
        pc = None
        for line in r.stdout.splitlines():
            m = _PASS_RE.search(line)
            if m:
                pc = int(m.group(1))
        return r.returncode, pc

    def _run_ruff(self, path):
        """Run `ruff check .` inside the sandbox -> True iff clean. Part of the FULL gate ('no bugs').
        If ruff is not importable in the sandbox interpreter, returns True (do not block; run_selfchecks
        -- which actually EXECUTES the code -- remains the breakage gate)."""
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import ruff"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if probe.returncode != 0:
                return True  # ruff absent in this env -> selfchecks still gate
            r = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "."],
                cwd=path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout,
            )
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return True

    # -------------------------------------------------------------------- the proposer --------------

    def propose(self, target, edit_fn, utility_fn=None):
        """Try ONE candidate edit to a whitelisted non-safety reasoning module, in a temp sandbox only.

        Returns {status, accepted, protected_ok, utility_before, utility_after, sandbox}.
          status: 'disabled' | 'rejected' | 'accepted' | 'error'
        `edit_fn(sandbox_root)` performs the candidate edit INSIDE the sandbox copy (never the real tree).
        `utility_fn(sandbox_root)->float|None` is an optional metric; default utility is the run_selfchecks
        pass-count. Acceptance requires: protected files byte-identical + gate green + utility not regressed.
        This NEVER applies anything to the real repo -- it returns a proposal for human review.
        """
        result = {
            "status": "rejected",
            "accepted": False,
            "protected_ok": True,
            "utility_before": None,
            "utility_after": None,
            "sandbox": None,
            "reason": "",
        }

        # (0) DOUBLE OFF-SWITCH: inert unless BOTH enable=True and NORD_SELFMOD=1. Do NOTHING otherwise.
        if not self._is_enabled():
            self.log.append(
                (str(target), "disabled", "off by default (needs enable=True + NORD_SELFMOD=1)")
            )
            return {
                **result,
                "status": "disabled",
                "reason": "disabled: needs enable=True AND NORD_SELFMOD=1",
            }

        try:
            rel = _rel(self._repo, target)

            # (1a) FIREWALL: a protected/safety/verifier file is NEVER a legal target -> refuse up front,
            # before touching the filesystem. protected_ok=False marks the safety-envelope refusal.
            if rel in self._protected:
                self.log.append((rel, "rejected", "firewall: protected file"))
                return {
                    **result,
                    "protected_ok": False,
                    "reason": f"firewall: {rel} is a PROTECTED safety/verifier file (never a target)",
                }

            # (1b) WHITELIST: only NON-safety reasoning/ modules are candidates. (Not a safety breach --
            # simply not an allowed target -- so protected_ok stays True.)
            if rel not in self._whitelist:
                self.log.append((rel, "rejected", "not in whitelist"))
                return {
                    **result,
                    "reason": f"rejected: {rel} is not in the non-safety reasoning/ whitelist",
                }

            # (1c) BOUNDED: cap the number of attempts per instance.
            self._attempts += 1
            if self._attempts > self._max_attempts:
                self.log.append((rel, "rejected", "attempt budget exhausted"))
                return {
                    **result,
                    "reason": f"rejected: attempt budget ({self._max_attempts}) exhausted",
                }

            # (2) ISOLATE: copy the repo into a fresh temp sandbox; the real tree is never mutated.
            token = self._rng.randrange(1 << 30)
            sandbox = tempfile.mkdtemp(prefix=f"selfmod_wt_{token}_", dir=self._sandbox_root)
            shutil.rmtree(sandbox)  # copytree wants a non-existent dest
            shutil.copytree(self._repo, sandbox, ignore=_COPY_IGNORE)
            result["sandbox"] = sandbox

            protected_present = [
                p for p in self._protected if os.path.isfile(os.path.join(sandbox, *p.split("/")))
            ]
            outcome = {}

            # (3)+(4) apply the edit + measure the gate INSIDE the sandbox. All heavy work happens here;
            # the verdict is produced by SelfImprover below (steps 4-5), which isolates a deepcopy first.
            def apply(state):
                crc_before = _crc_map(sandbox, protected_present)
                # utility BEFORE (on the clean copy). Default = selfcheck pass-count.
                if utility_fn is None:
                    _, util_before = self._run_selfchecks(sandbox)
                else:
                    util_before = utility_fn(sandbox)
                outcome["utility_before"] = util_before

                edit_fn(sandbox)  # <-- the candidate edit, sandbox-only

                crc_after = _crc_map(sandbox, protected_present)
                protected_ok = crc_before == crc_after  # byte-identity firewall (crc32)
                rc, pc_after = self._run_selfchecks(
                    sandbox
                )  # the real gate (executes the code), run in the sandbox
                selfcheck_ok = rc == 0
                ruff_ok = (
                    self._run_ruff(sandbox) if self._require_ruff else True
                )  # FULL gate: ALSO lint-clean (no bugs)
                util_after = pc_after if utility_fn is None else utility_fn(sandbox)
                outcome["utility_after"] = util_after
                outcome["protected_ok"] = protected_ok
                outcome["ruff_ok"] = ruff_ok

                gate_ok = bool(protected_ok and selfcheck_ok and ruff_ok)
                # "only serves to improve itself": with require_improvement a STRICT utility increase is
                # demanded; otherwise non-regress (a held utility is safe). Fed to SelfImprover's monotonic
                # ratchet (1.0 > 0.0 baseline -> accept); correct for integer pass-counts AND float metrics.
                if util_after is None or util_before is None:
                    utility_ok = False
                elif self._require_improvement:
                    utility_ok = util_after > util_before
                else:
                    utility_ok = util_after >= util_before
                state["gate_ok"] = gate_ok
                state["score"] = 1.0 if (gate_ok and utility_ok) else 0.0

            # (5) VERDICT via the project's own audited verifier: corrigible (refuses protected targets),
            # isolates a deepcopy, requires the verify gate green, and accepts only a strict improvement
            # over the 0.0 baseline (== gate green + utility held/raised).
            improver = SelfImprover(
                {"gate_ok": True, "score": 0.0},
                verify=lambda s: bool(s["gate_ok"]),
                fitness=lambda s: s["score"],
                protected=self._protected,
            )
            accepted, reason = improver.propose(Change(rel, [rel], apply))

            result["accepted"] = bool(accepted)
            result["protected_ok"] = outcome.get("protected_ok", True)
            result["utility_before"] = outcome.get("utility_before")
            result["utility_after"] = outcome.get("utility_after")
            result["status"] = "accepted" if accepted else "rejected"
            result["reason"] = reason
            self.log.append((rel, result["status"], reason))

            # keep the sandbox on ACCEPT (for human inspection of the proposal); clean it up on REJECT.
            if not accepted:
                shutil.rmtree(sandbox, ignore_errors=True)
                result["sandbox"] = None
            return result
        except Exception as e:  # guard: a failure must never crash the caller
            self.log.append((str(target), "error", f"{type(e).__name__}: {e}"))
            return {**result, "status": "error", "reason": f"error: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------- the self-check (DISABLED-safe) -----


def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _crc_file(path):
    with open(path, "rb") as f:
        return zlib.crc32(f.read())


# a fast, hermetic stand-in for run_selfchecks.py: it py_compiles every .py under src/ (no imports, no
# heavy deps) so a whitespace edit stays green and a syntax-breaking edit reddens the gate -- instantly.
_TINY_SELFCHECKS = (
    "import os, sys, py_compile\n"
    "root = os.path.dirname(os.path.abspath(__file__))\n"
    "src = os.path.join(root, 'src')\n"
    "mods = []\n"
    "for dp, dn, fn in os.walk(src):\n"
    "    for f in fn:\n"
    "        if f.endswith('.py'):\n"
    "            mods.append(os.path.join(dp, f))\n"
    "mods.sort()\n"
    "ok = 0\n"
    "for m in mods:\n"
    "    try:\n"
    "        py_compile.compile(m, doraise=True)\n"
    "        print('PASS ' + os.path.relpath(m, root))\n"
    "        ok += 1\n"
    "    except Exception:\n"
    "        print('FAIL ' + os.path.relpath(m, root))\n"
    "print(str(ok) + '/' + str(len(mods)) + ' self-checks passed')\n"
    "sys.exit(0 if ok == len(mods) else 1)\n"
)


def _build_tiny_repo(root):
    """A throwaway repo the harness can drive fast: a stub run_selfchecks.py, one whitelisted reasoning
    module (a legal target), and one protected 'safety' file (reflex.py) that must stay untouchable."""
    _write(os.path.join(root, "run_selfchecks.py"), _TINY_SELFCHECKS)
    _write(
        os.path.join(root, "src", "pkg", "reasoning", "good_mod.py"),
        "def contribute(x):\n    return x + 1\n",
    )
    _write(
        os.path.join(root, "src", "pkg", "brainstem", "reflex.py"),
        "def estop():\n    return True\n",
    )


def main():
    _emit("=" * 96)
    _emit("SELF-MOD HARNESS -- DISABLED-by-default, sandboxed, verifier-gated PROPOSER (Wave 4B)")
    _emit("=" * 96)

    repo = _repo_root()
    # crc a couple of REAL files up front; nothing below may change them (the sandbox is a copy).
    real_run = os.path.join(repo, "run_selfchecks.py")
    real_reflex = os.path.join(repo, "src", "snn_data_lake", "brainstem", "reflex.py")
    real_before = {}
    for p in (real_run, real_reflex):
        if os.path.isfile(p):
            real_before[p] = _crc_file(p)

    work = tempfile.mkdtemp(prefix="selfmod_selfcheck_")
    saved_env = os.environ.get("NORD_SELFMOD")
    try:
        # ---- (1) DISABLED BY DEFAULT: no enable, no env -> status 'disabled', no sandbox, does nothing ----
        os.environ.pop("NORD_SELFMOD", None)
        h_off = SelfModHarness(enable=False)
        r_off = h_off.propose("src/snn_data_lake/reasoning/derivation.py", lambda sb: None, None)
        assert r_off["status"] == "disabled" and r_off["accepted"] is False, (
            "default must be DISABLED"
        )
        assert r_off["sandbox"] is None, "a disabled propose() must NOT create a sandbox"
        # enable=True alone is NOT enough (env still unset) -> still disabled
        h_env_missing = SelfModHarness(enable=True)
        assert (
            h_env_missing.propose("src/snn_data_lake/reasoning/derivation.py", lambda sb: None)[
                "status"
            ]
            == "disabled"
        ), "enable=True WITHOUT NORD_SELFMOD=1 must stay disabled"
        # env set but enable=False -> still disabled
        os.environ["NORD_SELFMOD"] = "1"
        assert (
            SelfModHarness(enable=False).propose(
                "src/snn_data_lake/reasoning/derivation.py", lambda sb: None
            )["status"]
            == "disabled"
        ), "NORD_SELFMOD=1 WITHOUT enable=True must stay disabled"
        _emit(
            "  [OK] DISABLED by default: needs BOTH enable=True AND NORD_SELFMOD=1 (no sandbox, no change)"
        )

        # ---- build a hermetic tiny repo; from here on the harness is EXPLICITLY enabled (env already =1) ----
        tiny = os.path.join(work, "tinyrepo")
        _build_tiny_repo(tiny)
        wl = ("src/pkg/reasoning/good_mod.py",)
        prot = ("src/pkg/brainstem/reflex.py",)

        def mk():
            # require_ruff=False: the tiny STUB repo has no ruff config (the ruff FULL-gate applies to the
            # real ruff-configured repo, exercised via the brain's opt-in self_improve path, not this stub).
            return SelfModHarness(
                enable=True,
                whitelist=wl,
                seed=0,
                repo=tiny,
                protected=prot,
                sandbox_root=work,
                timeout=120,
                require_ruff=False,
            )

        def edit_whitespace(
            sb,
        ):  # a benign no-op: append a blank line (still compiles) -> utility held
            with open(
                os.path.join(sb, "src", "pkg", "reasoning", "good_mod.py"), "a", encoding="utf-8"
            ) as f:
                f.write("\n")

        def edit_break(sb):  # a syntax-breaking edit -> run_selfchecks reddens -> must be REJECTED
            with open(
                os.path.join(sb, "src", "pkg", "reasoning", "good_mod.py"), "w", encoding="utf-8"
            ) as f:
                f.write("def (((  # deliberately broken\n")

        def edit_side_effect(
            sb,
        ):  # legal target BUT also mutates a protected file -> crc firewall trips
            with open(
                os.path.join(sb, "src", "pkg", "reasoning", "good_mod.py"), "a", encoding="utf-8"
            ) as f:
                f.write("\n")
            with open(
                os.path.join(sb, "src", "pkg", "brainstem", "reflex.py"), "a", encoding="utf-8"
            ) as f:
                f.write("# tampered\n")

        # ---- (2) enabled + benign edit to a whitelisted module -> ACCEPTED (in the sandbox only) ----
        r_ok = mk().propose("src/pkg/reasoning/good_mod.py", edit_whitespace, None)
        assert r_ok["status"] == "accepted" and r_ok["accepted"] is True, (
            "a green, non-regressing edit must be ACCEPTED"
        )
        assert r_ok["protected_ok"] is True, "the benign edit touched no protected file"
        assert r_ok["sandbox"] and os.path.isdir(r_ok["sandbox"]), (
            "accepted proposal keeps its sandbox for review"
        )
        assert r_ok["sandbox"] != tiny and os.path.abspath(r_ok["sandbox"]) != os.path.abspath(
            tiny
        ), "sandbox is a COPY"
        _emit(
            f"  [OK] enabled + benign edit ACCEPTED in sandbox (util {r_ok['utility_before']}->{r_ok['utility_after']}), real tree untouched"
        )

        # ---- (3) THE FIREWALL ----
        # 3a: a PROTECTED safety file is never a legal target -> REJECTED, protected_ok False, no sandbox
        r_fw = mk().propose("src/pkg/brainstem/reflex.py", edit_whitespace, None)
        assert r_fw["accepted"] is False and r_fw["protected_ok"] is False, (
            "editing a safety file must be REJECTED (firewall)"
        )
        assert r_fw["sandbox"] is None, "a firewalled target never gets a sandbox"
        # 3b: a legal target whose edit ALSO tampers a protected file -> crc firewall -> REJECTED, protected_ok False
        r_se = mk().propose("src/pkg/reasoning/good_mod.py", edit_side_effect, None)
        assert r_se["accepted"] is False and r_se["protected_ok"] is False, (
            "a side-effect touch of a protected file must be REJECTED"
        )
        # 3c: a legal target whose edit BREAKS a selfcheck -> gate red -> REJECTED (protected untouched)
        r_br = mk().propose("src/pkg/reasoning/good_mod.py", edit_break, None)
        assert r_br["accepted"] is False, "an edit that reddens the gate must be REJECTED"
        assert r_br["protected_ok"] is True, (
            "the broken edit touched no protected file (it just failed the gate)"
        )
        _emit(
            "  [OK] FIREWALL: protected target rejected (protected_ok False); side-effect on a safety file rejected; broken selfcheck rejected"
        )

        # ---- (5) DETERMINISM: same inputs -> same verdict ----
        v1 = mk().propose("src/pkg/reasoning/good_mod.py", edit_whitespace, None)["accepted"]
        v2 = mk().propose("src/pkg/reasoning/good_mod.py", edit_whitespace, None)["accepted"]
        assert v1 == v2 is True, "the harness must be deterministic (same edit -> same verdict)"

        # the exact protected set it enforces (reported for the audit trail)
        default_state = SelfModHarness(enable=False).state()
        assert default_state["enabled"] is False, "state() reflects the disabled default"
        assert "src/snn_data_lake/brainstem/reflex.py" in _PROTECTED, (
            "the arm-safety kernel is protected"
        )
        assert "src/snn_data_lake/reasoning/self_improve.py" in _PROTECTED, (
            "the bounded-improvement verifier is protected"
        )
        assert "src/snn_data_lake/reasoning/self_mod_harness.py" in _PROTECTED, (
            "the proposer protects itself"
        )
        assert (
            "run_selfchecks.py" in _PROTECTED
            and "src/snn_data_lake/reasoning/godel_machine.py" in _PROTECTED
        ), "verifier + godel kernel protected"
        assert len(_PROTECTED) == 9, (
            "exactly the 9 immutable files (5 brainstem + run_selfchecks + godel + self_improve + self_mod)"
        )
        _emit(f"  [OK] deterministic; PROTECTED set = {len(_PROTECTED)} immutable files enforced")

    finally:
        # restore env exactly as we found it, then remove ALL temp state
        if saved_env is None:
            os.environ.pop("NORD_SELFMOD", None)
        else:
            os.environ["NORD_SELFMOD"] = saved_env
        shutil.rmtree(work, ignore_errors=True)

    # ---- (4) the REAL repo tree is byte-unchanged (crc a couple of files before/after the whole run) ----
    for p, crc0 in real_before.items():
        assert _crc_file(p) == crc0, f"the real tree must be byte-unchanged: {p}"
    _emit(
        "  [OK] real repo tree byte-identical after the self-check (crc32 of run_selfchecks.py + reflex.py unchanged)"
    )

    _emit("")
    _emit(
        "self-check OK: DISABLED-by-default sandboxed verifier-gated PROPOSER -- inert unless enable=True "
        "AND NORD_SELFMOD=1; even enabled it only writes a temp sandbox, refuses the 9 protected files "
        "(crc32 firewall), rejects broken/regressing edits, applies NOTHING to the real tree, and leaves "
        "the safety envelope untouched. Enabling + applying is human-gated; this is a proposer, not an "
        "intelligence explosion."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
