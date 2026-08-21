"""Self-check carry-forward cache (owner-authorized build, 2026-08-21): a battery may
CARRY a module's PASS verdict when the module's entire static import closure is
byte-identical to the closure that produced that verdict on THIS host -- a hash-verified
re-statement of "these exact bytes already passed here", never a skip.

WHY: the sequential battery re-runs all ~326 self-checks per commit while a typical commit
touches a handful of files. Parallel battery forms are REFUTED on this host (the pilot,
twice; sequential is binding), so the honest accelerator shrinks WORK instead of
overlapping it: a verdict carries only when nothing that could have changed the outcome --
the module's source, every in-package source it can statically reach, the shared data
packs, the runner, the NORD_* environment, or the interpreter -- has changed a byte.

PRIOR ART, read before building (testmon's determining-affected-tests + hidden-test-
dependencies; Bazel's hermeticity rule): coverage-block selection is FINER but traces
nothing across subprocess boundaries -- and every self-check here IS a subprocess -- so
the import-closure key is the structurally right granularity for this battery: coarser,
over-invalidating, never carrying wrongly for statically-reachable code. Bazel's lesson
is the envelope: an UNDECLARED input is the classic cache poison, so everything a check
could see that is not per-module goes into one envelope hash and any envelope change
invalidates every entry.

THE KEY (all of it, or no carry):
  * the module's own file sha256 + the sha256 of EVERY file in its static import closure
    (dormancy_gate.import_graph() -- the ONE walk, reused; ast-level, so function-local
    lazy imports are included);
  * the ENVELOPE: run_selfchecks.py's hash, the tracked model packs (nmnist/shd -- data
    several perception checks read), every NORD_* env var of the invoking process, the
    python version tag (free-threaded marked), the cache format version, and (v2, the
    review's BLOCKER class) the GOVERNANCE SURFACE -- every package __init__.py (executed
    on every import, present in no static closure), the five brainstem/ safety-envelope
    files + SECURITY.md (data the envelope_hash_gate and safety_envelope READ; an
    unapproved brainstem edit must never carry the stale PASS of the gate built to catch
    it), docs/POLICIES.md and the tools/ .json/.nir artifacts the policy gates read.
Only PASS verdicts are cached; a FAIL always re-runs; record() re-hashes the module's
closure FRESH at record time (the memo-staleness fix -- a PASS is attributed to the bytes
that ran, not to hashes read minutes earlier).

The store lives at the REAL user home (~/.snn_data_lake/selfcheck_cache.json),
DELIBERATELY not substrate.home(): the battery exports NORD_HOME to a throwaway temp dir,
and a cache living there would die with every run.

DISCLOSED BLIND SPOTS (banner() prints them at every carrying run):
  1. STRING-DISPATCH imports (importlib with a computed name) are invisible to the static
     closure -- the dormancy gate's own known limit, inherited knowingly.
  2. SHARED-HOME SEQUENCE COUPLING: all self-check subprocesses share one persistent
     NORD_HOME for the run, so a carried module's ABSENCE changes what home-state a
     downstream check sees this run (the hidden-test-dependency shape). Bounded -- that
     home already persists across batteries -- but real.
  3. Host state outside the envelope (installed package versions, GPU state) is unhashed.
THE PROTOCOL those blind spots buy: carries are a LOCAL-battery accelerator only (CI,
act containers and exit-certification batteries run FULL -- --carry is opt-in and nothing
in .github/ passes it), and ANY FAIL in a carrying battery is re-run FULL before it is
diagnosed -- a carried-run red is not evidence until the full run confirms it.

  python -m snn_data_lake.substrate.selfcheck_cache   # self-check (the P8 reds)
"""

# distinct-from: substrate/dormancy_gate.py -- owns THE import graph; this REUSES it via
#   import_graph() to hash closures, never re-deriving reachability.
# distinct-from: tools/p4c_ratchet.py -- counts modules per zone; this hashes them.
# distinct-from: substrate/policy_manifest.py -- certifies POLICY text; this caches
#   self-check verdicts. All sites read.

from __future__ import annotations

import hashlib
import json
import os
import sys

_CACHE_VERSION = 2  # v2 (review BLOCKERs): the envelope grew the governance surface --
# every package __init__.py (they execute on EVERY import yet appear in NO static
# closure; REGIONS lives in the top one), the five brainstem/ safety-envelope files +
# SECURITY.md (envelope_hash_gate and safety_envelope read them as DATA, invisible to
# import closures -- an unapproved brainstem edit must never carry the stale PASS of the
# gate built to catch it), docs/POLICIES.md and the tools/ governance artifacts (the
# policy/manifest/charge gates read them as data too). Over-invalidation by design: a
# governance-surface change is exactly when a full battery is wanted.
_PKG_PREFIX = "snn_data_lake."


def _repo_root():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def _cache_path():
    return os.path.join(os.path.expanduser("~"), ".snn_data_lake", "selfcheck_cache.json")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_file(dotted):
    """snn_data_lake.<region>.<mod> -> source file path (package dirs -> __init__.py)."""
    rel = dotted[len(_PKG_PREFIX) :] if dotted.startswith(_PKG_PREFIX) else dotted
    base = os.path.join(_repo_root(), "src", "snn_data_lake", rel.replace(".", os.sep))
    if os.path.isdir(base):
        return os.path.join(base, "__init__.py")
    return base + ".py"


class SelfcheckCache:
    """consult(mod) -> cached row | None; record(mod, when) after a live PASS."""

    def __init__(self, file_hash_overrides=None, env=None):
        # `file_hash_overrides` ({path: fake_hash}) and `env` are the TESTING SEAMS for
        # the self-check's reds: they prove a dependency change or an environment change
        # flips the key WITHOUT mutating the tree or the process env. Never set otherwise.
        self._overrides = dict(file_hash_overrides or {})
        self._env = dict(os.environ if env is None else env)
        self._graph = None
        self._file_hashes = {}
        self._closure_keys = {}
        self.envelope = self._compute_envelope()
        self._store = self._load()

    # -- hashing ---------------------------------------------------------------------
    def _hash_file(self, path):
        if path in self._overrides:
            return self._overrides[path]
        if path not in self._file_hashes:
            self._file_hashes[path] = _sha256_file(path) if os.path.exists(path) else "ABSENT"
        return self._file_hashes[path]

    # distinct-from: substrate/envelope_hash_gate.py -- that gate VERIFIES the five
    #   brainstem files against an approved baseline; this merely lists governance-surface
    #   paths (reusing safety_envelope.repo_relative(), the ONE path list) so a change to
    #   any of them invalidates the whole cache. Consumer of the list, not a second
    #   verifier. Read.
    def _governance_files(self):
        """The envelope's data-input surface: files gates read as DATA (invisible to any
        import closure) plus every package __init__ (executed on every import, in no
        closure). Derived by walk + the safety envelope's ONE path list, never curated
        by hand here."""
        root = _repo_root()
        out = [os.path.join(root, "SECURITY.md"), os.path.join(root, "docs", "POLICIES.md")]
        from snn_data_lake.substrate.safety_envelope import repo_relative

        out += [os.path.join(root, p) for p in repo_relative()]
        pkg = os.path.join(root, "src", "snn_data_lake")
        for r, _d, files in os.walk(pkg):
            if "__init__.py" in files:
                out.append(os.path.join(r, "__init__.py"))
        tools = os.path.join(root, "tools")
        if os.path.isdir(tools):
            for f in sorted(os.listdir(tools)):
                if f.endswith((".json", ".nir")):
                    out.append(os.path.join(tools, f))
        return sorted(out)

    def _compute_envelope(self):
        root = _repo_root()
        gil = getattr(sys, "_is_gil_enabled", lambda: True)()
        nord = "|".join(f"{k}={v}" for k, v in sorted(self._env.items()) if k.startswith("NORD_"))
        gov = self._governance_files()
        self.absent_inputs = [p for p in gov if not os.path.exists(p)]
        parts = [
            f"cache_v{_CACHE_VERSION}",
            f"py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            + ("" if gil else "t"),
            "runner:" + self._hash_file(os.path.join(root, "run_selfchecks.py")),
            "nmnist:" + self._hash_file(os.path.join(root, "models", "nmnist_pack.npz")),
            "shd:" + self._hash_file(os.path.join(root, "models", "shd_pack.npz")),
            "nord:" + nord,
        ] + [os.path.relpath(p, root) + ":" + self._hash_file(p) for p in gov]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _import_graph(self):
        if self._graph is None:
            from snn_data_lake.substrate.dormancy_gate import import_graph

            self._graph = import_graph()
        return self._graph

    def closure_key(self, mod):
        """The carry key for one MODULES entry ('cortex.percept_surprise'): sha256 over
        the sorted (dotted, file-hash) pairs of its static import closure + envelope."""
        if mod in self._closure_keys:
            return self._closure_keys[mod]
        g = self._import_graph()
        start = _PKG_PREFIX + mod
        seen, frontier = {start}, [start]
        while frontier:
            nxt = []
            for node in frontier:
                for dep in g.get(node, ()):
                    if dep not in seen:
                        seen.add(dep)
                        nxt.append(dep)
            frontier = nxt
        rows = sorted((d, self._hash_file(_module_file(d))) for d in seen)
        key = hashlib.sha256(
            (self.envelope + "||" + "||".join(f"{d}={h}" for d, h in rows)).encode()
        ).hexdigest()
        self._closure_keys[mod] = key
        return key

    # -- store ------------------------------------------------------------------------
    def _load(self):
        try:
            with open(_cache_path(), encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self):
        path = _cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=0)

    def consult(self, mod):
        """The carried row iff a PASS is stored under the CURRENT closure key, else None.
        A mismatched key, a FAIL verdict, or no row all mean: run live."""
        row = self._store.get(mod)
        if not row or row.get("verdict") != "PASS":
            return None
        if row.get("key") != self.closure_key(mod):
            return None
        return row

    def record(self, mod, when):
        """Record a live PASS under a FRESHLY-hashed key (review MAJOR: the memoised
        hashes could be a whole battery old, so a mid-run edit let record() attribute a
        PASS to bytes that never ran -- and that row becomes a wrong carry the moment the
        tree returns to those bytes. The memos for this module's closure are dropped and
        recomputed at record time; the residual race window shrinks from the battery's
        duration to milliseconds, with the frozen-tree rule as the primary defence and
        this as the mechanism that does not trust it)."""
        self._closure_keys.pop(mod, None)
        g = self._import_graph()
        start = _PKG_PREFIX + mod
        seen, frontier = {start}, [start]
        while frontier:
            nxt = []
            for node in frontier:
                for dep in g.get(node, ()):
                    if dep not in seen:
                        seen.add(dep)
                        nxt.append(dep)
            frontier = nxt
        for d in seen:
            self._file_hashes.pop(_module_file(d), None)
        self._store[mod] = {"key": self.closure_key(mod), "verdict": "PASS", "when": when}
        self._save()

    def banner(self):
        absent = getattr(self, "absent_inputs", [])
        absent_note = (
            " ENVELOPE INPUTS ABSENT (each hashed as the literal ABSENT -- loud, not "
            f"silent): {', '.join(os.path.basename(p) for p in absent)}."
            if absent
            else ""
        )
        return (
            f"CARRY-FORWARD ON (envelope v{_CACHE_VERSION} {self.envelope[:12]}..., "
            f"{len(self._store)} cached verdicts): a PASS carries only when the module's "
            "ENTIRE static import closure + runner + model packs + NORD_* env + the "
            "governance surface (every __init__, the brainstem safety files, SECURITY.md, "
            "POLICIES.md, tools/ artifacts) are byte-identical. BLIND to string-dispatch "
            "imports, shared-home sequence coupling, and unhashed host state -- local "
            "batteries only; CI/act/exit certifications run FULL, and any FAIL here is "
            f"re-run FULL before diagnosis.{absent_note}"
        )


def self_check() -> int:
    """The P8 reds, each constructed and fired:
    (a) DEPENDENCY INVALIDATION -- faking one closure-dependency's file hash must flip the
        module's key (proves the key covers deps, not just the module file);
    (b) POISON CONTROL -- a stored PASS under a STALE key must NOT carry (the control that
        is not the method itself: consult() compared against an independently recomputed
        key), while the same row under the current key must carry;
    (c) ENVELOPE INVALIDATION -- a changed NORD_* env must flip every key;
    (d) FAIL never carries, even under a matching key."""
    probe = "substrate.signal_ring"  # a real module with a real multi-file closure

    base = SelfcheckCache(env={"NORD_GPU": "1"})
    k0 = base.closure_key(probe)

    # (a) dependency invalidation: override a DEP's hash (wiring_manifest is in
    # signal_ring's closure via its Port import), key must move
    dep = _module_file("snn_data_lake.substrate.wiring_manifest")
    forked = SelfcheckCache(file_hash_overrides={dep: "0" * 64}, env={"NORD_GPU": "1"})
    if forked.closure_key(probe) == k0:
        print("selfcheck_cache self-check FAILED: a changed DEPENDENCY did not flip the key")
        return 1

    # (b) poison control, both directions -- consult() against an in-memory store
    base._store = {probe: {"key": "STALE" + k0[5:], "verdict": "PASS", "when": "x"}}
    if base.consult(probe) is not None:
        print("selfcheck_cache self-check FAILED: a stale-key PASS was carried (poison)")
        return 1
    base._store = {probe: {"key": k0, "verdict": "PASS", "when": "x"}}
    if base.consult(probe) is None:
        print("selfcheck_cache self-check FAILED: a current-key PASS did not carry")
        return 1

    # (c) envelope invalidation: different NORD_* env -> different key
    other_env = SelfcheckCache(env={"NORD_GPU": "0"})
    if other_env.closure_key(probe) == k0:
        print("selfcheck_cache self-check FAILED: a changed NORD_* env did not flip the key")
        return 1

    # (d) FAIL never carries
    base._store = {probe: {"key": k0, "verdict": "FAIL", "when": "x"}}
    if base.consult(probe) is not None:
        print("selfcheck_cache self-check FAILED: a FAIL verdict was carried")
        return 1

    # (e) THE SAFETY-SURFACE RED (review BLOCKER, v2): a byte change to a brainstem
    # envelope file -- data the envelope_hash_gate reads, invisible to every import
    # closure -- must flip the ENVELOPE and with it every key. Same for the package
    # __init__ (in zero closures, executed by every import).
    from snn_data_lake.substrate.safety_envelope import repo_relative

    brainstem_file = os.path.join(_repo_root(), sorted(repo_relative())[0])
    tampered = SelfcheckCache(file_hash_overrides={brainstem_file: "f" * 64}, env={"NORD_GPU": "1"})
    if tampered.envelope == base.envelope or tampered.closure_key(probe) == k0:
        print(
            "selfcheck_cache self-check FAILED: a brainstem safety-file change did not "
            "invalidate the cache -- the stale envelope_hash_gate PASS would carry"
        )
        return 1
    pkg_init = os.path.join(_repo_root(), "src", "snn_data_lake", "__init__.py")
    init_fork = SelfcheckCache(file_hash_overrides={pkg_init: "e" * 64}, env={"NORD_GPU": "1"})
    if init_fork.closure_key(probe) == k0:
        print(
            "selfcheck_cache self-check FAILED: a package __init__ change (REGIONS lives "
            "there; it executes on every import yet sits in no closure) did not flip keys"
        )
        return 1

    n_closure = len(base._closure_keys) and len(
        [1 for _ in [probe]]
    )  # attribution: one probe, printed below
    print(
        f"selfcheck_cache self-check OK: dependency-change flipped the key; stale-key "
        f"poison refused and current-key carry accepted (the control is the recomputed "
        f"key, not the cache's own claim); NORD_* envelope change invalidated; FAIL never "
        f"carries; a brainstem SAFETY-FILE change and a package __init__ change each "
        f"invalidate everything (the v2 governance surface -- data inputs no import "
        f"closure can see). Probe closure: {probe} ({n_closure} module probed, envelope "
        f"v{_CACHE_VERSION} {base.envelope[:12]}...). Blind spots: {banner_blind_spots()}"
    )
    return 0


def banner_blind_spots():
    return (
        "string-dispatch imports; shared-home sequence coupling; unhashed host state "
        "(carries = local batteries only; any carried-run FAIL reruns FULL first)"
    )


if __name__ == "__main__":
    sys.exit(self_check())
