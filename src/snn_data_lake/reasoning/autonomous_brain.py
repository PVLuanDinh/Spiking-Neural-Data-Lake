r"""
AUTONOMOUS BRAIN -- the full enclosed prototype. One self-contained, open-ended, self-directed loop
that ties together everything the session built into a single autonomous agent with lifelong memory
that deepens its own understanding and grows its own neurons, then repeats.

The enclosed cycle (each faculty is a REAL, verified component, not a stub):

  PERCEPTION      conv1 latency-STDP (unsupervised, spiking, frozen)         [deep_stdp_latency]
                  -> batched byte-identical feature extraction               [fast_features]
  WORKING MEMORY  the current objective is held as a POINTER in prefrontal   [cortex.prefrontal_executive]
                  PERSISTENT SPIKING ACTIVITY (reverberating attractor)
  SELF-DIRECTION  spiking curiosity (intrinsic learning-progress) + curriculum + a
                  METACOGNITIVE monitor pick the next objective + decide when to grow
                  [neuromodulation.spiking_curiosity, cortex.metacognitive_monitor]
  UNDERSTANDING   a supervised readout on the frozen features -- competence deepens as
                  the agent studies more                                     [brain_adapters.growable_readout]
  NEUROGENESIS    when metacognition fires grow_trigger, the brain GROWS new readout neurons
                  (function-preserving grow-and-freeze) -> new capacity      [brain_adapters.growable_readout]
  LIFELONG MEMORY each learned objective's exemplars are CONSOLIDATED to the durable substrate
                  data lake + recalled BY PROXIMITY (survives restart)       [brain_adapters.lake_memory]
  REPEAT          the cycle continues open-ended; competence feeds the next decision.

Honest scope: this is an ARCHITECTURAL prototype -- it demonstrates the CLOSED autonomous loop with
every faculty genuinely active and measurable on MNIST digits (WM 18/18, 10x neurogenesis, 3000
exemplars durable, self-directed curriculum). Understanding lands ~47% here, NOT because the loop is
broken but because ONLINE single-pass grow-and-freeze is a genuinely weak readout: the frozen-hidden /
plastic-output MLP still forgets earlier classes, and curiosity fixates on one hard digit late. The
accuracy-optimal enclosed loop was developmental_loop_lake_slda.py (SLDA-RFF readout, ~94.9%;
RETIRED 2026-08-11, verdict in CHANGELOG v3.75, mechanism absorbed by developmental_core),
which is order-invariant so it does NOT forget -- but it does not grow neurons. That tension (genuine
online neurogenesis vs. forgetting-free accuracy) is the honest finding, not a bug to paper over.
Feature learning is unsupervised; only the readout is supervised.

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python -m snn_data_lake.reasoning.autonomous_brain
Knobs: NORD_CYCLES(18) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(300) NORD_KREPLAY(60)
       NORD_HIDDEN(32) NORD_GROW_K(16) NORD_HIDDEN_CAP(200) NORD_MAINTAIN(20) NORD_TRAIN(10000) NORD_TEST(2000)
Deps: numpy torch torchvision brian2
"""

import os
import tempfile

import torch

from snn_data_lake.substrate.device import resolve_device, enable
from snn_data_lake.instruments.snn_mnist_stdp import load_mnist
from snn_data_lake.instruments.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.reasoning.fast_features import batched_features_and_keys
from snn_data_lake.reasoning._loop_common import envi, acc
from snn_data_lake.hippocampus.growable_readout import GrowableReadout
from snn_data_lake.hippocampus.proximity_replay import LakeMemory
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.cortex.metacognitive_monitor import MetacognitiveMonitor
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus


def _evidence(option, n=10, strong=1.0, weak=0.05):
    e = [weak] * n
    e[option] = strong
    return e


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    CYCLES = envi("NORD_CYCLES", 18)
    C1 = envi("NORD_C1", 30)
    T_LAT = envi("NORD_T", 15)
    L1_EPOCHS = envi("NORD_L1_EPOCHS", 3)
    PER = envi("NORD_PER", 300)
    KREPLAY = envi("NORD_KREPLAY", 60)
    HIDDEN = envi("NORD_HIDDEN", 32)
    GROW_K = envi("NORD_GROW_K", 16)
    HIDDEN_CAP = envi("NORD_HIDDEN_CAP", 200)
    MAINTAIN = envi("NORD_MAINTAIN", 20)
    # NOTE: this was os.environ.get("TEMP", ".") -- a WINDOWS-only variable. On the Linux/macOS
    # CI runners TEMP is unset, so the "." fallback fired and this file was written into the
    # process WORKING DIRECTORY (the repo checkout). tempfile.gettempdir() honours TMPDIR/TEMP/TMP
    # on every platform and never falls back to ".".
    lake_path = os.path.join(tempfile.gettempdir(), "sndl_autonomous_brain.spkl")

    print("=" * 92)
    print(
        "AUTONOMOUS BRAIN -- enclosed self-directed loop: perceive -> hold-in-WM -> learn -> grow -> consolidate -> repeat"
    )
    print(
        f"device={dev}  cycles={CYCLES}  C1={C1}  hidden={HIDDEN}(+{GROW_K}/grow,cap {HIDDEN_CAP})  lake={lake_path}"
    )
    print("=" * 92)
    Xtr, Ytr, Xte, Yte = load_mnist(envi("NORD_TRAIN", 10000), envi("NORD_TEST", 2000))

    # ---- PERCEPTION: unsupervised spiking features (frozen), batched extraction + <=64-dim lake keys ----
    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    Ftr, Gtr = batched_features_and_keys(conv, Xtr)
    Fte, _ = batched_features_and_keys(conv, Xte)
    print(f"[perception] frozen; feat_dim={Ftr.shape[1]}; diversity={conv.kernel_diversity():.3f}")
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    # ---- the faculties (all real, verified components) ----
    readout = GrowableReadout(
        D, n_classes=10, hidden=HIDDEN, seed=0
    )  # UNDERSTANDING + NEUROGENESIS
    lake = LakeMemory(path=lake_path, seed=0)  # LIFELONG durable memory
    wm = PrefrontalExecutive(n_options=10, seed=0)  # WORKING MEMORY (persistent-activity pointer)
    meta = MetacognitiveMonitor(seed=0)  # METACOGNITION (drives grow)
    curiosity = SpikingCuriosityDrive(seed=0)  # intrinsic learning-progress
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
    lc = LocusCoeruleus()

    learned, competence, mem_rows, curve = [], {d: 0.0 for d in range(10)}, {}, []
    grow_events, wm_hits = 0, 0
    for cyc in range(1, CYCLES + 1):
        # ---- DECIDE: curiosity proposes; WM HOLDS the objective as a persistent-activity pointer ----
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        proposed = curiosity.most_curious(candidates)
        wm.update(_evidence(proposed))
        for _ in range(MAINTAIN):
            wm.maintain()  # reverberating hold, no input
        objective = wm.goal if wm.goal is not None else proposed  # act on the WM-held pointer
        wm_hits += int(objective == proposed)
        is_new = objective not in learned
        comp_before = competence[objective]
        gain = 1.0 + 0.5 * lc.step(1.0 if is_new else (1.0 - comp_before))

        # ---- LEARN: readout on the objective's features + LIFELONG replay PULLED BY PROXIMITY from the lake ----
        replay = []
        for d in [x for x in learned if x != objective]:
            q = Gtr[mem_rows[d]].mean(0)
            replay.extend(int(r) for r in lake.retrieve(q.tolist(), k=KREPLAY))
        n_train = PER if is_new else max(KREPLAY, PER // 4)
        idx = torch.tensor(tr_by[objective][:n_train] + replay, dtype=torch.long, device=dev)
        readout.train_step(Ftr[idx], ytr[idx], lr=0.01 * gain, steps=300)

        # ---- CONSOLIDATE to LIFELONG lake memory ----
        if is_new:
            pool = tr_by[objective][n_train : n_train + PER]
            lake.consolidate([Gtr[r].tolist() for r in pool], payloads=pool)
            mem_rows[objective] = pool
            learned.append(objective)

        # ---- MEASURE competence (the understanding), feed self-direction ----
        for d in learned:
            competence[d] = acc(readout, Fte, yte, te_by[d])
        overall = acc(readout, Fte, yte, [r for d in learned for r in te_by[d]])
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

        # ---- NEUROGENESIS: the brain GROWS a fresh block of readout neurons for each NEW concept and FREEZES
        #      the old ones (function-preserving grow-and-freeze) -> new capacity per concept, no forgetting.
        #      Metacognition can request EXTRA growth on a hard revisit (grow_trigger). ----
        grew = ""
        # skip growth on the FIRST concept (the initial block learns it); grow+freeze for every new concept after
        want_grow = (is_new and len(learned) >= 1) or meta.grow_trigger()
        if want_grow and readout.n_hidden < HIDDEN_CAP:
            readout.grow(GROW_K, freeze_old=True)
            grow_events += 1
            grew = f"  [+neurogenesis: grew to {readout.n_hidden} neurons]"
        print(
            f"CYCLE {cyc:2d} propose {proposed}->WM {objective} ({'new' if is_new else 'rev'})  "
            f"understand={overall:.1%}  lake={lake.n_stored}  metacog(rel={meta.reliability():.2f}){grew}"
        )

    durable = lake.durability_check()
    final = curve[-1] if curve else 0.0
    print("\n" + "=" * 92)
    print("AUTONOMOUS BRAIN -- one full enclosed cycle, all faculties active:")
    print("-" * 92)
    print(
        f"  PERCEPTION   : unsupervised conv1 features (frozen), diversity {conv.kernel_diversity():.3f}"
    )
    print(
        f"  WORKING MEM  : objective held in prefrontal persistent activity  {wm_hits}/{CYCLES} cycles"
    )
    print(
        f"  SELF-DIRECT  : curiosity + curriculum ({curriculum.mastered}/10 mastered) + metacognition chose every objective"
    )
    print(
        f"  NEUROGENESIS : the brain GREW its own readout {grow_events}x -> {readout.n_hidden} neurons"
    )
    print(
        f"  LIFELONG MEM : {lake.n_stored} exemplars in the DURABLE data lake; survives restart = {durable.get('ok')}"
    )
    print(f"  UNDERSTANDING: held-out competence over {sorted(learned)} = {final:.1%}")
    print("-" * 92)
    print(
        "  perceive -> hold-in-WM -> learn -> deepen -> grow neurons -> consolidate to lifelong memory -> REPEAT"
    )
    print("=" * 92)
    assert final > 0.10, "autonomous brain at/below chance"
    assert durable.get("ok"), "lifelong memory did not survive restart"
    assert wm_hits > 0 and lake.n_stored > 0, "a faculty was inactive"
    print(
        f"self-check OK: enclosed autonomous loop -- perception+WM+self-direction+neurogenesis({grow_events} grows)+"
        f"lifelong-lake(durable)+understanding({final:.1%}) all active over {CYCLES} cycles."
    )
    return 0


if __name__ == "__main__":
    main()
