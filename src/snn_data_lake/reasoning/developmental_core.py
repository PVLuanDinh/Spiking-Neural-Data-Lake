r"""
DEVELOPMENTAL CORE -- the enclosed digit-curriculum loop, extracted as ONE reusable function.

The loop body previously lived inline in research/autonomous_brain_recovered.py's main(). It is
extracted here VERBATIM so two callers share ONE source of truth:

  * research/autonomous_brain_recovered.py -- the standalone recovery benchmark (a thin wrapper that
    runs the same banner + the same regression asserts: online > 90%, sleep-refit > 90%).
  * research/brain_loop.py -- the unified always-on Brain, which injects its OWN durable SSD lake +
    reflector and receives every newly-learned concept through `emit`, so the perceptual concepts the
    loop learns become facts the chat can converse from.

  perceive (conv1 latency-STDP, unsupervised + frozen) -> hold objective in prefrontal WM ->
  LEARN (SLDA-RFF, order-invariant => no forgetting) -> GROW the RFF basis (neurogenesis) ->
  CONSOLIDATE exemplars to the durable lake -> REFLECT (train of thought over its own concepts) ->
  repeat -> sleep-refit (RBF over the lake).

TORCH TIER: this module imports torch at module load, so it is imported LAZILY by brain_loop.py
(never at its top level) -- that is what lets the Brain boot + converse under the torch-free
.venv-lake while the perceptual develop runs under .venv-cu128 against the SAME SSD home.

  Standalone: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/autonomous_brain_recovered.py
Deps: numpy torch torchvision snntorch scikit-learn
"""

import os

import torch

from snn_data_lake.reasoning.torch_device import resolve_device, enable
from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.reasoning.fast_features import batched_features_and_keys
from snn_data_lake.reasoning._loop_common import envi, envf
from snn_data_lake.reasoning.brain_adapters.lake_memory import LakeMemory
from snn_data_lake.reasoning.developmental_loop_slda import SLDA_RFF, _acc
from snn_data_lake.reasoning.developmental_loop_lake99 import _rbf_refit_eval
from snn_data_lake.reasoning.concept_reflection import ConceptReflector
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.cortex.metacognitive_monitor import MetacognitiveMonitor
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus


def cfg_from_env():
    """The NORD_* knobs, exactly as the standalone loop read them."""
    return {
        "cycles": envi("NORD_CYCLES", 18),
        "c1": envi("NORD_C1", 30),
        "t_lat": envi("NORD_T", 15),
        "l1_epochs": envi("NORD_L1_EPOCHS", 3),
        "per": envi("NORD_PER", 400),
        "kreplay": envi("NORD_KREPLAY", 150),
        "store": envi("NORD_STORE", 400),
        "rff0": envi("NORD_RFF", 2000),
        "grow_rff": envi("NORD_GROW_RFF", 200),
        "gamma": envf("NORD_GAMMA", 0.04),
        "shrink": envf("NORD_SHRINK", 0.1),
        "maintain": envi("NORD_MAINTAIN", 20),
        "train": envi("NORD_TRAIN", 12000),
        "test": envi("NORD_TEST", 2000),
        "reflect_thr": envf("NORD_REFLECT_THR", 0.55),
    }


def _evidence(option, n=10, strong=1.0, weak=0.05):
    e = [weak] * n
    e[option] = strong
    return e


def _fresh_slda(D, rff, gamma, shrink, obs_F, obs_y):
    """NEUROGENESIS via order-invariant re-accumulation: build an SLDA_RFF with a WIDER random-Fourier
    basis (rff projection neurons) and SLEEP-REPLAY every accumulated (feature, true-label) batch back
    into its statistics. Order-invariance guarantees this equals the joint fit over the wider basis, so
    growing neurons rebuilds -- never forgets. Deterministic (seed fixed)."""
    s = SLDA_RFF(D, rff=rff, gamma=gamma, shrink=shrink, seed=0)
    for F, y in zip(obs_F, obs_y):
        s.observe(F, y)
    return s


def develop_cycles(cfg, *, lake=None, reflector=None, emit=None, quiet=False):
    """Run the enclosed developmental curriculum. Returns a metrics dict (no asserts, no banner).

    lake=None      -> a temp LakeMemory is created + closed here (the standalone behaviour).
    lake given     -> consolidate into the CALLER's durable lake (the Brain's SSD .spkl); not closed here.
    reflector=None -> a fresh ConceptReflector.
    emit(concept, prototype, chain) -> called once per NEWLY-learned digit with its learned feature
                    PROTOTYPE and the reflection train-of-thought CHAIN, so a caller (the Brain) can
                    narrate them into facts and persist to durable storage. (Narration lives in the
                    caller so it stays torch-free -- this module is the torch tier.)
    """
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    say = (lambda *a, **k: None) if quiet else print

    own_lake = lake is None
    if own_lake:
        lake_path = os.path.join(os.environ.get("TEMP", "."), "sndl_autonomous_recovered.spkl")
        if os.path.exists(lake_path):
            os.remove(lake_path)
        lake = LakeMemory(path=lake_path, seed=0)
    if reflector is None:
        reflector = ConceptReflector(threshold=cfg["reflect_thr"], seed=0)
    if emit is None:

        def emit(_c, _p, _f):
            return None

    Xtr, Ytr, Xte, Yte = load_mnist(cfg["train"], cfg["test"])

    # ---- PERCEPTION: unsupervised spiking features (frozen), batched extraction + <=64-dim lake keys ----
    torch.manual_seed(0)
    say(
        f"\n[perception] conv1 latency-STDP C1={cfg['c1']} (unsupervised, {cfg['l1_epochs']} epochs)..."
    )
    conv = ConvLatencyStdp(cfg["c1"], cfg["t_lat"])
    for ep in range(cfg["l1_epochs"]):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{cfg['l1_epochs']}")
    Ftr, Gtr = batched_features_and_keys(conv, Xtr)
    Fte, _ = batched_features_and_keys(conv, Xte)
    say(f"[perception] frozen; feat_dim={Ftr.shape[1]}; diversity={conv.kernel_diversity():.3f}")
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    yte_np = Yte.numpy()
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    # ---- the faculties (all real, verified components) ----
    rff = cfg["rff0"]
    slda = SLDA_RFF(D, rff=rff, gamma=cfg["gamma"], shrink=cfg["shrink"], seed=0)
    wm = PrefrontalExecutive(n_options=10, seed=0)
    meta = MetacognitiveMonitor(seed=0)
    curiosity = SpikingCuriosityDrive(seed=0)
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
    lc = LocusCoeruleus()

    learned, competence, mem_rows, curve = [], {d: 0.0 for d in range(10)}, {}, []
    obs_F, obs_y = [], []
    last_visit = {}
    grow_events, wm_hits = 0, 0
    for cyc in range(1, cfg["cycles"] + 1):
        # ---- DECIDE: curiosity proposes an unseen digit; else a ROUND-ROBIN revisit scheduler ----
        unseen = [d for d in range(10) if d not in learned]
        if unseen:
            proposed = curiosity.most_curious(unseen)
        else:
            proposed = min(learned, key=lambda d: last_visit.get(d, 0))
        wm.update(_evidence(proposed))
        for _ in range(cfg["maintain"]):
            wm.maintain()
        objective = wm.goal if wm.goal is not None else proposed
        wm_hits += int(objective == proposed)
        is_new = objective not in learned
        last_visit[objective] = cyc
        gain = 1.0 + 0.5 * lc.step(1.0 if is_new else (1.0 - competence[objective]))

        # ---- CONSOLIDATE a held-out slice of a NEW digit TO THE DURABLE LAKE ----
        if is_new:
            pool = tr_by[objective][cfg["per"] : cfg["per"] + cfg["store"]]
            lake.consolidate([Gtr[r].tolist() for r in pool], payloads=pool)
            mem_rows[objective] = pool
            learned.append(objective)

        # ---- LEARN: the objective's own exemplars + LIFELONG replay PULLED BY PROXIMITY ----
        obj_idx = torch.tensor(tr_by[objective][: cfg["per"]], dtype=torch.long, device=dev)
        F_parts, y_parts = [Ftr[obj_idx]], [ytr[obj_idx]]
        replay_n = 0
        if objective in mem_rows and lake.n_stored > 0:
            q = (
                Gtr[torch.tensor(mem_rows[objective], dtype=torch.long, device=dev)]
                .mean(0)
                .tolist()
            )
            rows = [int(r) for r in lake.retrieve(q, k=cfg["kreplay"])]
            if rows:
                ridx = torch.tensor(rows, dtype=torch.long, device=dev)
                F_parts.append(Ftr[ridx])
                y_parts.append(ytr[ridx])
                replay_n = len(rows)
        Fb, yb = torch.cat(F_parts, 0), torch.cat(y_parts, 0)
        slda.observe(Fb, yb)
        obs_F.append(Fb)
        obs_y.append(yb)

        # ---- NEUROGENESIS: for each NEW concept, GROW the RFF basis + sleep-replay (order-invariant) ----
        grew = ""
        if is_new and len(learned) >= 2:
            rff += cfg["grow_rff"]
            slda = _fresh_slda(D, rff, cfg["gamma"], cfg["shrink"], obs_F, obs_y)
            grow_events += 1
            grew = f"  [+neurogenesis: RFF {rff} neurons]"

        # ---- REFLECT: a train of thought over the brain's OWN concepts; wire what it thought together.
        #      EMIT the concept + its narrated facts so a caller can persist them durably. ----
        reflected = ""
        if is_new:
            proto = Ftr[obj_idx].mean(0).detach().cpu().tolist()
            reflector.register(objective, proto)
            chain = reflector.reflect(str(objective))
            if len(chain) >= 2:
                reflected = f"  [reflect: {' -> '.join(chain)}]"
            emit(objective, proto, chain)  # caller narrates + persists (keeps narration torch-free)

        # ---- MEASURE competence (the understanding), feed self-direction ----
        for d in learned:
            competence[d] = _acc(slda, Fte, yte, te_by[d], learned)
        overall = _acc(slda, Fte, yte, [r for d in learned for r in te_by[d]], learned)
        curve.append(overall)
        curiosity.observe(objective, 1.0 - competence[objective])
        lp = curiosity.learning_progress(objective)
        curriculum.observe(lp)
        meta.update(
            confidence=competence[objective],
            surprise=abs(lp),
            learning_progress=lp,
            novelty=curiosity.novelty(objective),
        )
        say(
            f"CYCLE {cyc:2d} propose {proposed}->WM {objective} ({'new' if is_new else 'rev'})  "
            f"understand={overall:.1%}  lake={lake.n_stored}  replay={replay_n}  gain={gain:.2f}{grew}{reflected}"
        )

    online = curve[-1] if curve else 0.0

    # ---- SLEEP CONSOLIDATION: refit an RBF-SVM on ALL lake-consolidated exemplars (readout upgrade) ----
    all_rows = [r for d in learned for r in mem_rows[d]]
    pooled_te = [r for d in learned for r in te_by[d]]
    say(
        f"\n[sleep-refit] RBF-SVM on {len(all_rows)} lake-consolidated exemplars (offline consolidation)..."
    )
    rbf_acc, n_used = _rbf_refit_eval(Ftr, Ytr, all_rows, Fte, yte_np, pooled_te)
    durable = lake.durability_check()
    res = {
        "online": online,
        "rbf": rbf_acc,
        "n_used": n_used,
        "learned": learned,
        "competence": competence,
        "curve": curve,
        "grow_events": grow_events,
        "wm_hits": wm_hits,
        "rff": rff,
        "links": reflector.association_graph(),
        "strongest": reflector.strongest_link(),
        "durable": bool(durable.get("ok")),
        "n_stored": lake.n_stored,
        "cycles": cfg["cycles"],
        "conv_diversity": conv.kernel_diversity(),
        "curriculum_mastered": curriculum.mastered,
    }
    if own_lake:
        lake.close()
    return res
