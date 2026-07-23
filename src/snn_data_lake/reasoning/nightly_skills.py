r"""
NIGHTLY SKILLS -- one bounded night of autonomous cross-domain skill formation for the sleep loop.

The autonomous skill-building machinery already exists in this repo (meta_skill_loop's standing
language/math/physics/world-kb loop, skill_builder's proximity-routed SkillRegistry, open_world_loop's
unsupervised domain discovery, autotelic_goals' goal-babbling) -- but only as standalone `__main__`
studies that never tick. This module wraps that machinery into a PERSISTENT, BOUNDED "one night of
skill formation" step the canonical brain's circadian sleep loop calls once per night:

  night() = pick ONE frontier domain (coverage-first, staleness-guarded, else spiking-curiosity
            learning-progress -- exactly meta_skill_loop.main()'s selection), study ONE small
            env-budgeted slice of it into that domain's persistent readout, register/refresh the
            skill in the durable proximity router, then run one guarded bounded open-world
            discovery sub-step and one guarded bounded autotelic goal-babbling sub-step, and
            persist EVERYTHING under substrate.home() so a reboot resumes instead of restarting.

Persistence (real memory only, bounded, deduped -- never noise):
  <home>/skills/<domain>.npz     the learned readout state per domain:
                                   SLDA-RFF head -> W, b, sum_c, cnt_c, ZZt (the order-invariant
                                   sufficient statistics: restoring them restores competence
                                   EXACTLY -- continual == joint, so reload never resets to 0);
                                   DistMult head -> E, R factor matrices + both MT19937 rng states;
                                 plus stream position, competence, visits, baseline and the <=150
                                 signature exemplars that index the skill in the router lake.
  <home>/skills/open_world.npz   discovered-domain centroids + counts + stream position (the
                                 novelty buffer and per-domain item lists are ephemeral/bounded).
  <home>/skills.spkl             a per-boot SCRATCH proximity-router index (LakeMemory / StreamingHub),
                                 NOT a durable store -- deleted fresh and rebuilt from the persisted
                                 exemplars WITHOUT re-appending, so the lake never bloats.
  <home>/nightly_skills.json     curiosity + selection bookkeeping, autotelic labels, summary.

HONEST SCOPE (the ceiling, stated plainly): this is bounded nightly skill FORMATION with
per-domain competence curves -- NOT general intelligence, and it does not "solve" any domain.
Competence rides the project's hybrid: a spiking/positional representation under a strong
SLDA-RFF (or DistMult ranking) readout, and the readout IS the ceiling (the recorded next levers
are encoder gains, delays/adLIF). A "discovered" open-world domain is a region of crc32-signature
space, not a clean human concept. Autotelic goal mastery is near-tabular: it interpolates within
the babbled span and does NOT transfer to novel goals. Representation/cognition ONLY: no actuator
route, no motor command, brainstem is never imported.

Budget: env NORD_NIGHT_BUDGET pairs per night (default 300; target well under ~2s CPU per
steady-state night -- the one-time lazy domain construction on the first night costs more).
Knobs: NORD_NIGHT_BUDGET(300) NORD_NIGHT_RFF(600) NORD_NIGHT_CTX(3) NORD_NIGHT_LANG(40000)
       NORD_NIGHT_MATHN(4000) NORD_NIGHT_TRAJ(400) NORD_NIGHT_KBDIM(50) NORD_NIGHT_KBEVAL(250)
       NORD_NIGHT_KBTRIPLES(0=all) NORD_NIGHT_OW(120) NORD_NIGHT_GOALS(6)

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.nightly_skills
Deps: numpy (+ stdlib). Reuses meta_skill_loop, skill_builder, corpus_stream,
language_readout_slda (via Domain), kb_ranking, lake_memory, spiking_curiosity,
open_world_loop, autotelic_goals, substrate.home. Deterministic: per-instance
random.Random(seed) / np.random.RandomState(seed); no wall-clock; crc32 signatures only.
"""

import os
import random

import numpy as np

from snn_data_lake.cortex.autotelic_goals import (
    EVAL_STARTS,
    HELDOUT_GOALS,
    REACH_HI,
    REACH_LO,
    GoalConditionedReadout,
    H,
    _reach_rate,
    collect_labels,
)
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive
from snn_data_lake.reasoning.brain_adapters.lake_memory import LakeMemory
from snn_data_lake.reasoning.corpus_stream import ARPABET, load_cmudict, load_text, phoneme_stream
from snn_data_lake.reasoning.kb_ranking import DistMult, load_dataset
from snn_data_lake.reasoning.meta_skill_loop import Domain, physics_stream
from snn_data_lake.reasoning.open_world_loop import OpenWorldController, dna_stream
from snn_data_lake.reasoning.skill_builder import MATH_VOCAB, SkillRegistry, math_stream, signature
from snn_data_lake.substrate.home import home, lake_path, load_state, save_state

_SKILL_IDS = {"language": 0, "math": 1, "physics": 2, "world-kb": 3}
_ID_NAMES = {v: k for k, v in _SKILL_IDS.items()}
_STATE_JSON = "nightly_skills.json"
_CURIO_LOG_CAP = 240  # bounded chronological (domain, error) log replayed into the drive on reload
_EXEMPLAR_CAP = 150  # signature exemplars per skill (the existing signature_exemplars size)
_OW_ITEM_CAP = 150  # bounded per-discovered-domain item lists (never noise, never unbounded)

_SCALE_NOTE = (
    "bounded nightly skill FORMATION: one small env-budgeted slice per night into per-domain "
    "SLDA-RFF / DistMult readouts (the project's hybrid: spiking/positional representation + "
    "strong decoder -- the readout IS the ceiling). Per-domain competence curves vs each domain's "
    "own baseline; NOT general intelligence, no domain is 'solved'; a discovered open-world "
    "domain is a region of crc32-signature space, not a human concept; autotelic mastery is "
    "near-tabular with no transfer beyond the babbled span."
)


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


def _rng_state_arrays(rng):
    """(keys[624] uint32, pos, has_gauss, cached_gaussian) of a np.random.RandomState -- npz-storable."""
    st = rng.get_state()
    return np.asarray(st[1], dtype=np.uint32), int(st[2]), int(st[3]), float(st[4])


def _set_rng_state(rng, keys, pos, has_gauss, cached):
    rng.set_state(
        ("MT19937", np.asarray(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached))
    )


class _NightKB:
    """Bounded world-KB ranking domain: the same DistMult head + filtered-Hits@10 competence as
    meta_skill_loop.RankingDomain (duck-typed: name/vocab/baseline/competence/visits/train/test/
    study/signature_exemplars), with two night-budget changes: (a) an optional cap on train triples
    so the lazy construction stays bounded, (b) study() trains a budget-sized ROLLING slice instead
    of a full epoch -- one bounded night, accumulation across nights."""

    def __init__(self, name, dim=50, seed=0, eval_n=250, max_triples=0):
        tr_s, te_s, self.offline = load_dataset()
        if max_triples and len(tr_s) > int(max_triples):
            tr_s = tr_s[: int(max_triples)]
        ent, rel = {}, {}
        eid = lambda e: ent.setdefault(e, len(ent))  # noqa: E731
        rid = lambda r: rel.setdefault(r, len(rel))  # noqa: E731
        self.tr = np.array([[eid(h), rid(r), eid(t)] for h, r, t in tr_s], dtype=np.int64)
        self.te = np.array(
            [[eid(h), rid(r), eid(t)] for h, r, t in te_s if h in ent and r in rel and t in ent],
            dtype=np.int64,
        )
        if len(self.te) < 40 and len(self.tr) > 200:  # capped train left too few evaluable test
            k = max(40, len(self.tr) // 10)  # triples -> carve a held-out tail from train
            self.te = self.tr[-k:]
            self.tr = self.tr[:-k]
        self.ient = {v: k for k, v in ent.items()}
        self.irel = {v: k for k, v in rel.items()}
        self.n_ent = len(ent)
        self.known = {}
        for h, r, t in np.vstack([self.tr, self.te]):
            self.known.setdefault((int(h), int(r)), set()).add(int(t))
        self.model = DistMult(len(ent), len(rel), dim=dim, seed=seed)
        self.name = name
        self.vocab = ["<distmult-ranking-head>"]
        self.eval_n = int(eval_n)
        self.baseline = 10.0 / max(1, self.n_ent)  # random-ranking Hits@10
        self.competence = 0.0
        self.visits = 0
        self.registered = False
        self.train = self.tr  # duck-type alias
        self.rng = np.random.RandomState(0)
        self.tr_pos = 0
        self.test = [
            ([self.irel[int(r)], self.ient[int(h)]], self.ient[int(t)]) for h, r, t in self.te[:500]
        ]

    def study(self, slice_n):
        """Train the NEXT budget-sized rolling slice of the accumulated KB (not a full epoch), then
        filtered Hits@10 on an eval sample. The DistMult factors persist across nights."""
        if not len(self.tr):
            return 0.0
        n = max(1, min(len(self.tr), int(slice_n)))
        start = self.tr_pos % len(self.tr)
        sl = np.array([self.tr[(start + i) % len(self.tr)] for i in range(n)], dtype=np.int64)
        self.tr_pos = start + n
        self.model.train_epoch(sl, lr=0.1, batch=256)
        self.visits += 1
        idx = self.rng.permutation(len(self.te))[: min(self.eval_n, len(self.te))]
        hits = 0
        for j in idx:
            h, r, t = int(self.te[j, 0]), int(self.te[j, 1]), int(self.te[j, 2])
            sc = self.model.rank_tails(h, r)
            gold = sc[t]
            for o in self.known.get((h, r), ()):
                if o != t:
                    sc[o] = -1e30
            hits += (int(np.sum(sc > gold)) + 1) <= 10
        self.competence = hits / max(1, len(idx))
        return self.competence

    def signature_exemplars(self, n=150):
        src = self.te if len(self.te) else self.tr
        idx = self.rng.permutation(len(src))[:n]
        return [signature([self.irel[int(src[j, 1])], self.ient[int(src[j, 0])]]) for j in idx]


class NightlySkills:
    """The sleep loop's persistent, bounded nightly skill-formation step.

    One instance owns four lazily-built domains (language / math / physics / world-kb -- the
    meta_skill_loop set), a SpikingCuriosityDrive meta-controller keyed on domain name, a rebuildable
    proximity router (SkillRegistry over a SCRATCH LakeMemory at <home>/skills.spkl, rebuilt from the
    npz STATE on load), a persistent
    open-world discovery controller and an accumulated autotelic goal-label store. Construction
    RELOADS any persisted state under home() so a reboot resumes with competence intact
    (order-invariant SLDA == continual, so restoring the sufficient statistics restores the head
    exactly). night() never raises: every sub-step is guarded and failures degrade to a partial
    result dict. Honest scope: see _SCALE_NOTE / the module docstring."""

    def __init__(self, seed=0, budget_pairs=None, backend="slda"):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.budget_pairs = int(budget_pairs) if budget_pairs else _envi("NORD_NIGHT_BUDGET", 300)
        self.requested_backend = backend
        self.backend = "slda"
        if backend != "slda":
            # torch training is a FUTURE hook: accept the param, fall back honestly. No torch import.
            print(f"[nightly-skills] backend '{backend}' not implemented; falling back to 'slda'")
        # sizes (env-tuned; the self-check shrinks these)
        self.ctx = _envi("NORD_NIGHT_CTX", 3)
        self._rff = _envi("NORD_NIGHT_RFF", 600)
        self._gamma = _envf("NORD_NIGHT_GAMMA", 0.15)
        self._lang_limit = _envi("NORD_NIGHT_LANG", 40000)
        self._mathn = _envi("NORD_NIGHT_MATHN", 4000)
        self._traj = _envi("NORD_NIGHT_TRAJ", 400)
        self._kbdim = _envi("NORD_NIGHT_KBDIM", 50)
        self._kbeval = _envi("NORD_NIGHT_KBEVAL", 250)
        self._kbcap = _envi("NORD_NIGHT_KBTRIPLES", 0)
        self._ow_batch = _envi("NORD_NIGHT_OW", 120)
        self._goal_eps = _envi("NORD_NIGHT_GOALS", 6)
        # state
        self._domains = None  # name -> Domain/_NightKB (lazy, built ONCE per instance)
        self._lake = None
        self._registry = None
        self._exemplars = {}  # name -> [<=150 signature vectors] (the router build set)
        self._router_order = []  # consolidation order (band alignment across reloads)
        self._pending = {}  # name -> raw npz dict, applied when the domain is built
        self.meta = SpikingCuriosityDrive(seed=self.seed)
        self._curio_log = []  # bounded chronological [(name, error)] -> drive replay
        self._last = {}  # name -> last cycle studied (staleness guard)
        self._cycle = 0
        self._nights = 0
        self._ow = None
        self._ow_pool = None
        self._ow_pos = 0
        self._pending_ow = None
        self._auto_labels = {}  # (state, goal) -> direction (deduped by construction)
        self._auto_visited = set()
        self._auto_reach = None
        self._summary = {}
        try:
            self.load()
        except Exception as exc:  # a corrupt home must never block a boot
            print(f"[nightly-skills] load skipped ({exc!r}); starting fresh")

    # ------------------------------------------------------------------ lazy builds
    def _skills_dir(self):
        d = os.path.join(home(), "skills")
        os.makedirs(d, exist_ok=True)
        return d

    def _ensure_lake(self):
        if self._lake is None:
            # skills.spkl is a per-boot SCRATCH proximity index -- NOT a durable store. The durable skill
            # memory is the per-domain <home>/skills/*.npz STATE (exemplars + readout weights); load()
            # rebuilds this index from it. Deleted fresh at first build so it is never a second durable
            # copy of the exemplars -- one durable lake (brain_lake.bin), no more no less.
            sp = lake_path("skills.spkl")
            try:
                os.remove(sp)
            except FileNotFoundError:
                pass
            self._lake = LakeMemory(path=sp, seed=self.seed)
            self._registry = SkillRegistry(self._lake)
            for n in self._router_order:
                self._registry.names[_SKILL_IDS[n]] = n
        return self._lake

    def _ensure_ow(self):
        if self._ow is None:
            self._ow = OpenWorldController(
                lam=_envf("NORD_NIGHT_LAM", 1.7),
                harvest=_envi("NORD_NIGHT_HARVEST", 60),
                min_ev=_envi("NORD_NIGHT_MINEV", 12),
                cap=_envi("NORD_NIGHT_CAP", 8),
            )
            if self._pending_ow is not None:
                z = self._pending_ow
                for i, did in enumerate(z["dids"].tolist()):
                    self._ow.cents[int(did)] = np.asarray(z["cents"][i], dtype=np.float64)
                    self._ow.counts[int(did)] = int(z["counts"][i])
                    self._ow.items[int(did)] = []
                self._ow_pos = int(z["pos"])
                self._pending_ow = None
        return self._ow

    def _ensure_domains(self):
        """Build the four domains ONCE per instance (the corpus load is the heavy step), apply any
        persisted readout state, and prepare the bounded open-world discovery pool."""
        if self._domains is not None:
            return self._domains
        cmudict, _ = load_cmudict()
        words, _ = load_text()
        doms = {}
        doms["language"] = Domain(
            "language",
            list(phoneme_stream(words, cmudict, k=self.ctx, limit=self._lang_limit)),
            ARPABET,
            self.ctx,
            self._rff,
            self._gamma,
        )
        doms["math"] = Domain(
            "math",
            math_stream(n_expr=self._mathn, ctx=self.ctx, seed=0),
            MATH_VOCAB,
            self.ctx,
            self._rff,
            self._gamma,
        )
        doms["physics"] = Domain(
            "physics",
            physics_stream(n_traj=self._traj, ctx=self.ctx, seed=0),
            [str(b) for b in range(20)] + [" "],
            self.ctx,
            self._rff,
            self._gamma,
        )
        try:
            doms["world-kb"] = _NightKB(
                "world-kb", dim=self._kbdim, seed=0, eval_n=self._kbeval, max_triples=self._kbcap
            )
        except Exception as exc:  # KB dataset unavailable -> a 3-domain night, not a crash
            print(f"[nightly-skills] world-kb domain unavailable ({exc!r}); 3-domain night")
        for name, d in doms.items():
            z = self._pending.pop(name, None)
            if z is not None:
                try:
                    self._restore_domain(name, d, z)
                except Exception as exc:
                    print(f"[nightly-skills] restore of '{name}' skipped ({exc!r})")
        self._domains = doms
        self._build_ow_pool()
        return doms

    def _build_ow_pool(self):
        """A bounded, deterministic mixed unlabeled pool for the nightly discovery sub-step: slices
        of every built domain plus a dna stream as a genuinely-new source. Labels ride along ONLY
        for the summary; the controller never sees them as supervision."""
        pool = []

        def take(pairs, lbl, k=350):
            step = max(1, len(pairs) // k)
            for c, nx in pairs[::step][:k]:
                pool.append((c, nx, lbl))

        for name in ("language", "math", "physics"):
            if name in self._domains:
                take(self._domains[name].train, name)
        if "world-kb" in self._domains:
            for c, nx in self._domains["world-kb"].test[:200]:
                pool.append((c, nx, "world-kb"))
        take(dna_stream(n=800, ctx=self.ctx, seed=self.seed), "dna")
        random.Random(self.seed).shuffle(pool)
        self._ow_pool = pool

    # ------------------------------------------------------------------ persistence
    def _restore_domain(self, name, d, z):
        """Overwrite a freshly-built domain's readout with the persisted learned state (shape-guarded:
        a drifted corpus/vocab means the persisted head no longer fits and is skipped, honestly)."""
        if isinstance(d, _NightKB):
            if z["E"].shape != d.model.E.shape or z["R"].shape != d.model.R.shape:
                raise ValueError(f"world-kb factor shape drift {z['E'].shape} vs {d.model.E.shape}")
            d.model.E = np.asarray(z["E"], dtype=np.float64)
            d.model.R = np.asarray(z["R"], dtype=np.float64)
            _set_rng_state(d.model.rng, z["mkeys"], z["mpos"], z["mg"], z["mgv"])
            _set_rng_state(d.rng, z["ekeys"], z["epos"], z["eg"], z["egv"])
            d.tr_pos = int(z["tr_pos"])
        else:
            s = d.slda
            if z["W"].shape != s.W.shape or z["ZZt"].shape != s.ZZt.shape or int(z["V"]) != d.V:
                raise ValueError(f"'{name}' SLDA shape drift {z['W'].shape} vs {s.W.shape}")
            s.W = np.asarray(z["W"], dtype=np.float64)
            s.b = np.asarray(z["b"], dtype=np.float64)
            s.sum_c = np.asarray(z["sum_c"], dtype=np.float64)
            s.cnt_c = np.asarray(z["cnt_c"], dtype=np.float64)
            s.ZZt = np.asarray(z["ZZt"], dtype=np.float64)
            d.pos = int(z["pos"])
        d.competence = float(z["competence"])
        d.visits = int(z["visits"])
        d.baseline = float(z["baseline"])
        d.registered = bool(int(z["registered"]))

    def save(self):
        """Persist everything to home(): one npz per built domain (learned arrays + bookkeeping +
        exemplars), the open-world controller's centroids, and the json summary. The router lake
        (<home>/skills.spkl) is a rebuildable SCRATCH index (rebuilt from these npz exemplars on load),
        NOT a durable store -- the durable skill memory is the npz STATE. Atomic writes."""
        sk = self._skills_dir()
        for name, d in (self._domains or {}).items():
            path = os.path.join(sk, name + ".npz")
            tmp = path + ".tmp.npz"
            ex = np.asarray(self._exemplars.get(name, []), dtype=np.float64)
            if ex.size == 0:
                ex = np.zeros((0, 64), dtype=np.float64)
            if isinstance(d, _NightKB):
                mk, mp, mg, mgv = _rng_state_arrays(d.model.rng)
                ek, ep, eg, egv = _rng_state_arrays(d.rng)
                np.savez_compressed(
                    tmp,
                    E=d.model.E,
                    R=d.model.R,
                    mkeys=mk,
                    mpos=mp,
                    mg=mg,
                    mgv=mgv,
                    ekeys=ek,
                    epos=ep,
                    eg=eg,
                    egv=egv,
                    tr_pos=d.tr_pos,
                    competence=d.competence,
                    visits=d.visits,
                    baseline=d.baseline,
                    registered=int(d.registered),
                    exemplars=ex,
                )
            else:
                s = d.slda
                np.savez_compressed(
                    tmp,
                    W=s.W,
                    b=s.b,
                    sum_c=s.sum_c,
                    cnt_c=s.cnt_c,
                    ZZt=s.ZZt,
                    pos=d.pos,
                    competence=d.competence,
                    visits=d.visits,
                    baseline=d.baseline,
                    registered=int(d.registered),
                    inV=d.inV,
                    V=d.V,
                    ctx=d.ctx,
                    exemplars=ex,
                )
            os.replace(tmp, path)
        if self._ow is not None and self._ow.cents:
            dids = sorted(self._ow.cents)
            path = os.path.join(sk, "open_world.npz")
            tmp = path + ".tmp.npz"
            np.savez_compressed(
                tmp,
                cents=np.array([self._ow.cents[k] for k in dids], dtype=np.float64),
                dids=np.array(dids, dtype=np.int64),
                counts=np.array([self._ow.counts.get(k, 0) for k in dids], dtype=np.int64),
                pos=self._ow_pos,
            )
            os.replace(tmp, path)
        doms = dict(self._summary.get("domains", {}))
        if self._domains:
            for n, d in self._domains.items():
                doms[n] = {
                    "competence": float(d.competence),
                    "visits": int(d.visits),
                    "baseline": float(d.baseline),
                }
        self._curio_log = self._curio_log[-_CURIO_LOG_CAP:]
        summary = {
            "nights": int(self._nights),
            "cycle": int(self._cycle),
            "backend": self.backend,
            "last": {k: int(v) for k, v in self._last.items()},
            "curio_log": [[n, float(e)] for n, e in self._curio_log],
            "router_order": list(self._router_order),
            "domains": doms,
            "autotelic": {
                "labels": {f"{s},{g}": int(a) for (s, g), a in self._auto_labels.items()},
                "visited": sorted(int(v) for v in self._auto_visited),
                "heldout_reach": (None if self._auto_reach is None else float(self._auto_reach)),
            },
            "open_world": {
                "pos": int(self._ow_pos),
                "n_domains": (
                    len(self._ow.cents)
                    if self._ow is not None
                    else int(self._summary.get("open_world", {}).get("n_domains", 0))
                ),
            },
        }
        self._summary = summary
        save_state(_STATE_JSON, summary)

    def load(self):
        """Restore from home(); a no-op when nothing was saved. Rebuilds the proximity router's
        in-RAM index from the persisted exemplars WITHOUT re-appending to the durable lake (the
        appended spike events are still on disk; only the RAM-side build set is reconstructed), so
        repeated reboots never bloat the lake. Replays the bounded curiosity log into a fresh
        drive so selection resumes deterministically."""
        st = load_state(_STATE_JSON)
        if st is None:
            return
        self._summary = st
        self._nights = int(st.get("nights", 0))
        self._cycle = int(st.get("cycle", 0))
        self._last = {k: int(v) for k, v in st.get("last", {}).items()}
        self._curio_log = [[n, float(e)] for n, e in st.get("curio_log", [])][-_CURIO_LOG_CAP:]
        self.meta = SpikingCuriosityDrive(seed=self.seed)
        for n, e in self._curio_log:
            self.meta.observe(n, float(e))
        auto = st.get("autotelic", {})
        self._auto_labels = {}
        for k, a in auto.get("labels", {}).items():
            s, g = k.split(",")
            self._auto_labels[(int(s), int(g))] = int(a)
        self._auto_visited = set(int(v) for v in auto.get("visited", []))
        self._auto_reach = auto.get("heldout_reach")
        self._ow_pos = int(st.get("open_world", {}).get("pos", 0))
        sk = os.path.join(home(), "skills")
        self._pending = {}
        for name in _SKILL_IDS:
            p = os.path.join(sk, name + ".npz")
            if os.path.exists(p):
                with np.load(p, allow_pickle=False) as z:
                    self._pending[name] = {k: z[k] for k in z.files}
                ex = self._pending[name].get("exemplars")
                if ex is not None and ex.size:
                    self._exemplars[name] = [[float(x) for x in row] for row in ex]
        owp = os.path.join(sk, "open_world.npz")
        if os.path.exists(owp):
            with np.load(owp, allow_pickle=False) as z:
                self._pending_ow = {k: z[k] for k in z.files}
        order = [n for n in st.get("router_order", []) if self._exemplars.get(n)]
        if order:
            self._router_order = order
            lake = self._ensure_lake()
            pats = []
            for n in order:
                for v in self._exemplars[n]:
                    pats.append(lake._encode(v))
            expected = sum(len(p) for p in pats)
            if lake.hub.n_events() == 0:  # lake file was lost -> rewrite it
                for n in order:
                    lake.consolidate(
                        self._exemplars[n], payloads=[_SKILL_IDS[n]] * len(self._exemplars[n])
                    )
            elif lake.hub.n_events() == expected:  # durable events intact -> RAM-only rebuild
                i = 0
                for n in order:
                    for v in self._exemplars[n]:
                        lake._patterns.append(pats[i])
                        lake._payloads.append(_SKILL_IDS[n])
                        lake._features.append([float(x) for x in v])
                        i += 1
                lake._dirty = True
            else:  # stale/foreign lake -> rebuild losslessly
                lake.close()
                os.remove(lake.path)
                self._lake = None
                self._registry = None
                lake = self._ensure_lake()
                for n in order:
                    lake.consolidate(
                        self._exemplars[n], payloads=[_SKILL_IDS[n]] * len(self._exemplars[n])
                    )
            for n in order:
                self._registry.names[_SKILL_IDS[n]] = n
                self._registry.experts[_SKILL_IDS[n]] = {"ctx": self.ctx}

    # ------------------------------------------------------------------ the night
    def _register(self, name, d):
        ex = d.signature_exemplars(_EXEMPLAR_CAP)
        self._ensure_lake()
        self._registry.register(
            _SKILL_IDS[name], name, ex, {"vocab": list(d.vocab), "ctx": self.ctx}
        )
        d.registered = True
        self._exemplars[name] = [[float(x) for x in v] for v in ex]
        if name not in self._router_order:
            self._router_order.append(name)

    def _select(self, names):
        """meta_skill_loop.main()'s selection, verbatim in spirit: coverage first (unvisited domains
        go first), a staleness guard so no domain starves, else the spiking-curiosity frontier."""
        unstudied = [n for n in names if self._domains[n].visits == 0]
        if unstudied:
            return min(unstudied)
        stale = [n for n in names if self._cycle - self._last.get(n, 0) >= len(names)]
        if stale:
            return min(stale, key=lambda x: self._last.get(x, 0))
        return self.meta.most_curious(sorted(names))

    def _ow_step(self):
        """One bounded open-world discovery sub-step: feed the next batch of the mixed unlabeled
        pool to the persistent controller; new coherent signature clusters SPAWN discovered domains."""
        if self._ow_batch <= 0 or not self._ow_pool:
            return None
        ow = self._ensure_ow()
        before = len(ow.cents)
        n = min(self._ow_batch, len(self._ow_pool))
        for i in range(n):
            c, nx, lbl = self._ow_pool[(self._ow_pos + i) % len(self._ow_pool)]
            ow.observe(c, nx, lbl, self._ow_pos + i)
        self._ow_pos += n
        for did in list(ow.items):  # bounded item lists (never unbounded RAM)
            ow.items[did] = ow.items[did][-_OW_ITEM_CAP:]
        return {
            "domains": len(ow.cents),
            "new_domains": len(ow.cents) - before,
            "seen": int(self._ow_pos),
        }

    def _autotelic_step(self):
        """One bounded autotelic goal-babbling sub-step: a few hindsight-relabeled episodes merged
        into the accumulated (state, goal) -> direction label store; the goal-conditioned spiking
        readout is retrained only when the store grew, and scored on the FROZEN held-out goals (the
        one metric hindsight cannot inflate). Near-tabular; no transfer claim."""
        if self._goal_eps <= 0:
            return None
        sd = (self.seed * 1000003 + self._nights) % (2**31 - 1)
        examples, _buf = collect_labels(self._goal_eps, hindsight=True, seed=sd)
        before = len(self._auto_labels)
        for s, g, a in examples:
            self._auto_labels[(s, g)] = a
            self._auto_visited.add(int(s))
            self._auto_visited.add(int(g))
        new = len(self._auto_labels) - before
        if new > 0 or self._auto_reach is None:
            ro = GoalConditionedReadout(seed=self.seed)
            ro.train(sorted((s, g, a) for (s, g), a in self._auto_labels.items()))
            self._auto_reach = _reach_rate(ro, EVAL_STARTS, HELDOUT_GOALS, H, REACH_LO, REACH_HI)
        return {
            "labels": len(self._auto_labels),
            "new_labels": int(new),
            "heldout_reach": float(self._auto_reach),
            "visited": len(self._auto_visited),
        }

    def night(self):
        """ONE bounded night of skill formation. Never raises: every stage is guarded so a sleep
        tick cannot crash the brain; failures degrade to a partial result dict."""
        out = {
            "domain": None,
            "competence": 0.0,
            "progress": 0.0,
            "visits": 0,
            "discovered": None,
            "autotelic": None,
        }
        errors = []
        self._nights += 1
        try:
            self._ensure_domains()
        except Exception as exc:
            errors.append(f"domains: {exc!r}")
            self._domains = self._domains or {}
        try:
            names = sorted(self._domains)
            if names:
                self._cycle += 1
                name = self._select(names)
                self._last[name] = self._cycle
                d = self._domains[name]
                before = d.competence
                comp = float(d.study(self.budget_pairs))
                prog = max(0.0, comp - before)
                err = 1.0 - comp
                self.meta.observe(name, err)
                self._curio_log.append([name, err])
                if not d.registered:
                    self._register(name, d)
                out.update(domain=name, competence=comp, progress=float(prog), visits=int(d.visits))
        except Exception as exc:
            errors.append(f"study: {exc!r}")
        try:
            out["discovered"] = self._ow_step()
        except Exception as exc:
            errors.append(f"open_world: {exc!r}")
        try:
            out["autotelic"] = self._autotelic_step()
        except Exception as exc:
            errors.append(f"autotelic: {exc!r}")
        try:
            self.save()
        except Exception as exc:
            errors.append(f"save: {exc!r}")
        if errors:
            out["errors"] = errors
        return out

    # ------------------------------------------------------------------ queries
    def route(self, tokens):
        """Proximity-route a token context to a learned skill (content-addressable, no domain id).
        Returns the skill NAME when the routed payload is a known skill, else the raw payload / -1."""
        try:
            if self._registry is None:
                self._ensure_lake()
            got = self._registry.route(list(tokens))
            return _ID_NAMES.get(got, got)
        except Exception:
            return -1

    def state(self):
        """{"domains": {name: {"competence","visits","baseline"}}, "_scale": <honest note>} -- live
        values when the domains are built this session, else the persisted summary."""
        doms = {}
        if self._domains:
            for n, d in self._domains.items():
                doms[n] = {
                    "competence": float(d.competence),
                    "visits": int(d.visits),
                    "baseline": float(d.baseline),
                }
        else:
            for n, rec in self._summary.get("domains", {}).items():
                doms[n] = {
                    "competence": float(rec.get("competence", 0.0)),
                    "visits": int(rec.get("visits", 0)),
                    "baseline": float(rec.get("baseline", 0.0)),
                }
        return {"domains": doms, "_scale": _SCALE_NOTE}

    def close(self):
        """Release the durable lake handle (so another instance may attach to the same home)."""
        if self._lake is not None:
            try:
                self._lake.close()
            except Exception:
                pass
            self._lake = None
            self._registry = None


# ---------------------------------------------------------------------- self-check
def main():
    import shutil
    import tempfile

    small = {
        "NORD_NIGHT_LANG": "2500",
        "NORD_NIGHT_MATHN": "400",
        "NORD_NIGHT_TRAJ": "40",
        "NORD_NIGHT_KBDIM": "24",
        "NORD_NIGHT_KBEVAL": "60",
        "NORD_NIGHT_KBTRIPLES": "1500",
        "NORD_NIGHT_RFF": "300",
        "NORD_NIGHT_OW": "90",
        "NORD_NIGHT_GOALS": "4",
        "NORD_NIGHT_BUDGET": "300",
    }
    saved_env = {k: os.environ.get(k) for k in list(small) + ["NORD_HOME"]}
    work = tempfile.mkdtemp(prefix="sndl_night_")
    work2 = tempfile.mkdtemp(prefix="sndl_night2_")
    try:
        os.environ.update(small)
        os.environ["NORD_HOME"] = work

        print("=" * 96)
        print("NIGHTLY SKILLS -- persistent bounded nightly skill formation for the sleep loop")
        print(
            "(coverage/staleness/curiosity pick ONE frontier domain per night; everything persists under home())"
        )
        print("=" * 96)

        # ---- instance A: two nights, then persist ------------------------------------------
        a = NightlySkills(seed=0)
        r1 = a.night()
        r2 = a.night()
        print(
            f"  night 1: {r1['domain']:9s} competence {r1['competence']:.1%} (visit {r1['visits']}) "
            f"discovered={r1['discovered']} autotelic={r1['autotelic']}"
        )
        print(
            f"  night 2: {r2['domain']:9s} competence {r2['competence']:.1%} (visit {r2['visits']})"
        )
        assert "errors" not in r1, f"night 1 had guarded errors: {r1.get('errors')}"
        assert "errors" not in r2, f"night 2 had guarded errors: {r2.get('errors')}"
        assert r1["domain"] == "language" and r2["domain"] == "math", "coverage-first order broken"
        comp_a = {n: rec["competence"] for n, rec in a.state()["domains"].items()}
        lang_test = a._domains["language"].test[:3000]
        a.save()
        a.close()

        # ---- instance B: a FRESH process-equivalent reload from the same home --------------
        b = NightlySkills(seed=0)
        st_b = b.state()["domains"]
        for n in ("language", "math"):
            assert st_b[n]["competence"] >= comp_a[n] - 1e-9 and st_b[n]["competence"] > 0.0, (
                f"'{n}' competence NOT restored ({st_b[n]['competence']} vs {comp_a[n]})"
            )
            assert st_b[n]["visits"] == 1, f"'{n}' visits not restored"
        # the restored READOUT itself (not just the stored number) reproduces competence exactly
        b._ensure_domains()
        dl = b._domains["language"]
        yte = np.array([dl.ridx[nx] for _, nx in lang_test])
        re_eval = float((dl.slda.predict(dl._X(lang_test)) == yte).mean())
        assert abs(re_eval - comp_a["language"]) < 1e-9, (
            f"restored SLDA head re-evaluates {re_eval} vs saved {comp_a['language']}"
        )
        # router index survived: durable lake + rebuilt RAM index recall the right skill
        lake = b._ensure_lake()
        got = lake.retrieve(b._exemplars["language"][0], k=1)
        assert got and int(got[0]) == _SKILL_IDS["language"], f"router lost 'language' ({got})"
        dur = lake.durability_check()
        assert dur["ok"], f"router lake failed the durability round-trip: {dur}"
        routed = b.route(["3", "+", "4"])
        assert routed in _SKILL_IDS, f"route() returned no known skill ({routed})"
        print(
            f"  reload  : competence restored (language re-eval {re_eval:.1%} == saved "
            f"{comp_a['language']:.1%}); router durable={dur['ok']}; route(['3','+','4']) -> {routed}"
        )

        # ---- instance B: keep forming skills; some domain must clear its own baseline ------
        r3 = b.night()
        assert "errors" not in r3, f"night 3 had guarded errors: {r3.get('errors')}"
        assert r3["domain"] == "physics", "coverage-first should reach physics on night 3"
        extra = 0
        while extra < 8:
            doms = b._domains
            if any(d.visits > 0 and d.competence >= d.baseline for d in doms.values()):
                break
            rx = b.night()
            assert "errors" not in rx, f"extra night had guarded errors: {rx.get('errors')}"
            extra += 1
        cleared = [n for n, d in b._domains.items() if d.visits > 0 and d.competence >= d.baseline]
        assert cleared, "no domain reached its own baseline after bounded nightly accumulation"
        st = b.state()
        print(
            f"  nights  : {b._nights} total; baseline cleared by {cleared}; "
            f"open-world domains {len(b._ow.cents) if b._ow else 0}; "
            f"autotelic heldout reach {b._auto_reach}"
        )
        assert b._ow is not None and len(b._ow.cents) >= 1, (
            "open-world discovery never spawned a domain"
        )
        assert b._auto_reach is not None and b._auto_reach >= 0.0, (
            "autotelic sub-step never measured"
        )
        assert st["_scale"], "honest _scale note missing"
        b.close()

        # ---- determinism: a fresh home + same seed reproduces night 1 exactly --------------
        os.environ["NORD_HOME"] = work2
        c = NightlySkills(seed=0)
        rc = c.night()
        c.close()
        assert rc["domain"] == r1["domain"] and abs(rc["competence"] - r1["competence"]) < 1e-9, (
            f"nondeterministic night 1: {rc['domain']}/{rc['competence']} vs {r1['domain']}/{r1['competence']}"
        )

        ok = (
            f"self-check OK: {b._nights} bounded nights formed+persisted skills across "
            f"{len(st['domains'])} domains (baseline cleared by {','.join(sorted(cleared))}); reload "
            f"restored competence EXACTLY (SLDA sufficient statistics) and the proximity router "
            f"survived the restart; open-world discovered {len(b._ow.cents)} signature domains; "
            f"autotelic heldout reach {b._auto_reach:.2f}; deterministic night 1 across fresh homes. "
            f"Honest scope: {_SCALE_NOTE.split(';')[0]}."
        )
        ok.encode("ascii")  # ASCII-only guarantee (Windows cp1252)
        print("=" * 96)
        print(ok)
        return 0
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(work2, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
