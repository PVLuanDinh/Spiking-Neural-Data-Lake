r"""
GODEL MACHINE -- self-rewriting code with a native code-generation path, isolated in real git
worktrees, gated by a single immutable meta-rule: a rewrite is COMMITTED only if, verified FROM
OUTSIDE the rewrite, it passes every gate and strictly increases utility.

What makes this an actual Godel machine (Schmidhuber 2003), not just self-editing code:

  * EVERYTHING in the repository is a legal rewrite target -- the utility function, the proposal
    searcher, the test suite, this file's in-repo copy. The machine can rewrite its own checker.
  * There is exactly ONE fixed point, the meta-axiom: "execute a self-rewrite only upon PROOF that it
    increases utility." A machine that could also delete that rule is not a Godel machine -- it is
    self-erasing code with no optimality guarantee. The fixed meta-rule IS the definition.
  * The proof here is engineering-grade, not a theorem prover: apply the rewrite in an isolated git
    worktree, run the REAL gates + a PINNED behavioural probe against it, and merge only if green AND
    strictly better. Utility is MEASURED FROM OUTSIDE by the kernel's own probe, so a rewrite that
    edits the in-repo checker to always-pass, or inflates a self-reported score, is INERT -- the
    external measurement is unchanged, the proof fails, the rewrite is discarded. This is the
    anti-wireheading fixed point: the running kernel verifies candidates; it does not load its verdict
    from the code being judged.

Real-mode maps to a git worktree (this module):
  worktree add -> apply the generated code patch -> run gates from outside -> merge on proof / discard.

HONEST CEILING (stated so no one mistakes this for an intelligence explosion): the machine improves
only what its proposal generator can express and its utility can measure, and every step must pass the
current gates. SAFETY (corrected -- an earlier version overclaimed): a rewrite could otherwise weaken
the arm-safety envelope AND strip it from the very gate it runs against, so pure "safety by proof" is
NOT enough. Real-mode therefore fixes a minimal SAFETY KERNEL (`_SAFETY_KERNEL`: the 5 brainstem
arm-safety files + run_selfchecks.py + this file) that a rewrite may NOT touch -- the Godel machine's
immutable axioms. Everything else is rewritable; the fixed kernel is what makes "the proof obligation
preserves safety" actually true (the checker and E-STOP cannot be co-rewritten out). The loop is
invoked deliberately and merges to a local branch; not wired to auto-push or hardware. Unbounded in
target space, bounded by the gate + the fixed meta-rule + the fixed safety kernel -- which is what a
Godel machine facing safety-critical code must be.

Deps: stdlib + git (subprocess). The self-check builds a HERMETIC throwaway git repo and drives the
full worktree protocol on it (fast, touches nothing real); `attempt_on_repo` is the real-repo adapter.
  PYTHONPATH=src .venv-lake/Scripts/python research/godel_machine.py     # protocol on a temp git repo
"""

import os
import shutil
import subprocess
import sys
import tempfile

_GIT_ID = ["-c", "user.email=godel@sndl.local", "-c", "user.name=godel-machine"]
# no .pyc in worktrees -- else `git add -A` commits __pycache__ and the merge collides with main's
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.returncode, r.stdout


def _have_git():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


class GodelMachine:
    """The immutable kernel. `verify(worktree)->bool` and `utility(worktree)->float|None` are the
    machine's PINNED external judgment -- they run in the parent, measuring the candidate worktree from
    outside, and are NEVER loaded from the code under test. A None utility means 'failed to measure' =
    treated as no proof. Everything the machine edits lives in the repo; these two do not."""

    def __init__(self, repo, verify, utility):
        self.repo = repo
        self._verify = verify
        self._utility = utility
        self._n = 0
        base = self._utility(repo)
        if not self._verify(repo) or base is None:
            raise ValueError("refusing to start: HEAD does not pass its own gates")
        self._util = base
        self.log = []  # (name, accepted, reason, utility-after) -- the audit trail

    @property
    def utility(self):
        return self._util

    def attempt(self, name, rewrite):
        """Try ONE self-rewrite. `rewrite(worktree_dir)` writes the generated code into an ISOLATED git
        worktree (native code-generation / self-modification). Committed to main ONLY on proof."""
        self._n += 1
        branch = f"godel/try-{self._n}-{os.getpid()}"  # unique per attempt (pid-tagged)
        wt = tempfile.mkdtemp(prefix="godel_wt_")
        shutil.rmtree(wt)  # git worktree add wants a non-existent path
        _git(self.repo, "worktree", "add", "--quiet", "-b", branch, wt, "HEAD")
        try:
            rewrite(wt)  # <-- the self-rewrite mutates code in the worktree
            # PROOF, measured FROM OUTSIDE (the kernel's pinned judgment, not the worktree's own tests):
            if not self._verify(wt):
                return self._record(
                    name, False, "rejected: a gate went red under the pinned verifier"
                )
            new_util = self._utility(wt)
            if new_util is None:
                return self._record(
                    name, False, "rejected: utility could not be measured (no proof)"
                )
            if new_util <= self._util:
                return self._record(
                    name, False, f"rejected: utility {new_util} !> current {self._util} (no gain)"
                )
            # proof holds -> commit the rewrite and fast-forward main onto it (the machine has improved)
            _git(wt, "add", "-A")
            _git(
                wt,
                *_GIT_ID,
                "commit",
                "--quiet",
                "-m",
                f"godel: {name} (utility {self._util}->{new_util})",
            )
            _git(self.repo, "merge", "--ff-only", "--quiet", branch)
            self._util = new_util
            return self._record(name, True, f"committed: green + utility {new_util}")
        except Exception as e:  # a proposal that crashes is discarded, main intact
            return self._record(
                name, False, f"rejected: apply/verify raised {type(e).__name__}: {e}"
            )
        finally:
            # ALWAYS discard the worktree + branch: on accept the commit is already on main (redundant
            # ref), on reject nothing merged. Keeps the repo clean and branch names collision-free.
            _git(self.repo, "worktree", "remove", "--force", wt, check=False)
            _git(self.repo, "branch", "-D", branch, check=False)

    def _record(self, name, accepted, reason):
        self.log.append((name, accepted, reason, self._util))
        return accepted, reason


# ---------------------------------------------------------------- real-repo adapter (documented) ----

# The FIXED SAFETY KERNEL: files a real-repo self-rewrite may NOT touch. This is the Godel machine's
# immutable axiom set -- the arm-safety envelope (E-STOP + arm limits) AND the verification harness
# itself. Without it, a rewrite could weaken reflex.py AND strip it from the gate it runs against, then
# commit with safety silently removed. Everything ELSE stays rewritable; only this minimal core is fixed
# (so the "proof obligation preserves safety" claim actually holds -- it holds because the checker and
# the safety envelope cannot be co-rewritten out of existence).
_SAFETY_KERNEL = (
    "src/snn_data_lake/brainstem/reflex.py",
    "src/snn_data_lake/brainstem/serial_arm.py",
    "src/snn_data_lake/brainstem/arm_sim.py",
    "src/snn_data_lake/brainstem/arm_config.py",
    "src/snn_data_lake/brainstem/interpreter.py",
    "run_selfchecks.py",
    "src/snn_data_lake/reasoning/godel_machine.py",
)


def attempt_on_repo(repo, name, rewrite, pinned_probe, protected=_SAFETY_KERNEL):
    """Real-mode against an actual repo: a rewrite may NOT touch the fixed safety kernel (arm-safety +
    the verifier); if it leaves that intact it must make run_selfchecks + ruff GREEN in the worktree
    AND strictly raise the pinned external utility. NOT run in CI (spawns the full gate suite); this is
    the entrypoint a human invokes to let the brain rewrite its own core, one proven step at a time."""
    _git(repo, "worktree", "prune", check=False)  # clear any worktree left by a prior hard-kill

    def verify(wt):
        # CORRIGIBILITY: refuse a rewrite that modified any fixed-kernel file (checked from git status,
        # BEFORE trusting the worktree's own gate -- so the gate cannot be co-rewritten to pass).
        # -z (NUL-separated, no quote-wrapping) + --no-renames so a `git mv` of a protected file surfaces
        # as a plain `D <path>` that matches, instead of a `R "old" -> new` entry the parser would miss.
        _, status = _git(wt, "status", "--porcelain", "-z", "--no-renames")
        touched = {entry[3:] for entry in status.split("\0") if len(entry) > 3}
        if touched & set(protected):
            return False
        env = {
            **_ENV,
            "PYTHONPATH": "src",
            "NORD_HOME": tempfile.mkdtemp(prefix="godel_gate_home_"),
        }
        a = subprocess.run(
            [sys.executable, "run_selfchecks.py"], cwd=wt, env=env, capture_output=True
        )
        b = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."], cwd=wt, capture_output=True
        )
        return a.returncode == 0 and b.returncode == 0

    return GodelMachine(repo, verify, pinned_probe).attempt(name, rewrite)


# ---------------------------------------------------------------- the protocol self-check -----------

_SYSTEM_V0 = "def predict(x):\n    return 0            # v0: always guesses 0 (poor)\n"
# the machine's PINNED, EXTERNAL judgment -- defined HERE, never loaded from the repo under test.
_BATTERY = [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)]  # the true task: x -> x*x


def _probe_source():
    """A kernel-owned probe (not in the repo): import the candidate `system`, measure ACCURACY on the
    battery, print it. Because the kernel supplies this, a repo rewrite cannot fake the measurement."""
    return (
        "import sys; sys.path.insert(0, '.')\n"
        "import system\n"
        f"bat = {_BATTERY!r}\n"
        "ok = 0\n"
        "for x, y in bat:\n"
        "    try:\n"
        "        v = system.predict(x)\n"
        "    except Exception:\n"
        "        v = None\n"
        "    ok += 1 if v == y else 0\n"
        "print('UTIL', ok / len(bat))\n"
    )


def _measure(worktree):
    """Run the pinned probe against the worktree from OUTSIDE; return accuracy or None on failure."""
    probe = os.path.join(tempfile.gettempdir(), "godel_probe.py")
    with open(probe, "w") as f:
        f.write(_probe_source())
    r = subprocess.run(
        [sys.executable, probe], cwd=worktree, env=_ENV, capture_output=True, text=True
    )
    for line in r.stdout.splitlines():
        if line.startswith("UTIL "):
            return float(line.split()[1])
    return None


def main():
    print("=" * 96)
    print("GODEL MACHINE -- self-rewrite in real git worktrees, committed only on external proof")
    print("=" * 96)
    if not _have_git():
        print("SKIP: git not available (the worktree protocol needs git); nothing to prove here.")
        return 0

    repo = tempfile.mkdtemp(prefix="godel_repo_")
    try:
        # a hermetic throwaway repo whose 'core' is system.py -- the machine will rewrite it live.
        with open(os.path.join(repo, "system.py"), "w") as f:
            f.write(_SYSTEM_V0)
        _git(repo, "init", "--quiet")
        _git(repo, "add", "-A")
        _git(repo, *_GIT_ID, "commit", "--quiet", "-m", "seed: system v0")

        def verify(wt):  # a gate: the module must import and predict an int, no crash
            probe = os.path.join(tempfile.gettempdir(), "godel_gate.py")
            with open(probe, "w") as f:
                f.write(
                    "import sys; sys.path.insert(0,'.')\nimport system\nassert isinstance(system.predict(0), int)\n"
                )
            return (
                subprocess.run(
                    [sys.executable, probe], cwd=wt, env=_ENV, capture_output=True
                ).returncode
                == 0
            )

        gm = GodelMachine(repo, verify, _measure)
        start = gm.utility

        # a REAL improvement: rewrite the core to be partly right (native code-generation into the worktree)
        def improve(wt):
            with open(os.path.join(wt, "system.py"), "w") as f:
                f.write("def predict(x):\n    return x + x        # closer, but wrong for x*x\n")

        # a BREAKING rewrite: the gate (predict returns int, no crash) fails
        def broken(wt):
            with open(os.path.join(wt, "system.py"), "w") as f:
                f.write("def predict(x):\n    raise RuntimeError('boom')\n")

        # a WIREHEADING attempt: inflate a self-reported score, do NOT improve the real task
        def cheat(wt):
            with open(os.path.join(wt, "system.py"), "w") as f:
                f.write(
                    "SELF_SCORE = 1.0        # claim perfection...\n"
                    "def predict(x):\n    return x + x   # ...but the real accuracy is unchanged\n"
                )

        # the SELF-REWRITE that actually solves the task (x -> x*x): strict gain -> committed
        def solve(wt):
            with open(os.path.join(wt, "system.py"), "w") as f:
                f.write("def predict(x):\n    return x * x        # the true rule\n")

        r_improve = gm.attempt("improve-core", improve)
        r_broken = gm.attempt("faster-but-broken", broken)
        r_cheat = gm.attempt("wirehead-self-score", cheat)
        r_solve = gm.attempt("solve-task", solve)

        for nm, (ok, why) in [
            ("improve-core", r_improve),
            ("faster-but-broken", r_broken),
            ("wirehead-self-score", r_cheat),
            ("solve-task", r_solve),
        ]:
            print(f"  [{'COMMIT' if ok else 'REJECT'}] {nm:22s} -> {why}")
        _, main_src = _git(repo, "show", "HEAD:system.py")
        _, n_commits = _git(repo, "rev-list", "--count", "HEAD")
        print(
            f"\n  utility ratchet: {start} -> {gm.utility}   (commits on main: {n_commits.strip()})"
        )

        # ---- gated self-check: the Godel protocol holds on real worktrees ----
        assert r_improve[0] is True, "a green, strictly-better rewrite must be COMMITTED"
        assert r_broken[0] is False, "a rewrite that reddens the gate must be REJECTED"
        assert r_cheat[0] is False, (
            "a wireheading rewrite (inflated self-score, no real gain) must be REJECTED"
        )
        assert "x * x" in main_src, (
            "the committed core must be the genuinely-better one (self-rewrite landed)"
        )
        assert "boom" not in main_src and "SELF_SCORE" not in main_src, (
            "no rejected rewrite may reach main"
        )
        assert r_solve[0] is True and gm.utility == 1.0, (
            "the task-solving self-rewrite must reach full utility"
        )
        assert int(n_commits.strip()) == 3, (
            "exactly the two accepted rewrites are committed on top of the seed"
        )
        utils = [start] + [row[3] for row in gm.log]
        assert all(b >= a for a, b in zip(utils, utils[1:])), (
            "utility must be MONOTONIC (a Godel machine never regresses)"
        )

        print(
            "\nSELF-CHECK PASSED: real git-worktree self-rewrite -- the machine generated code into its "
            "own core, committed the two PROVEN improvements, and rejected the broken and the "
            "wireheading rewrites via an EXTERNAL measurement it cannot tamper. One immutable "
            "meta-rule (commit only on external proof of strict gain); everything else rewritable. "
            "Bounded by the gate, honest by construction -- not an intelligence explosion."
        )
        return 0
    finally:
        _git(repo, "worktree", "prune", check=False)
        shutil.rmtree(repo, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
