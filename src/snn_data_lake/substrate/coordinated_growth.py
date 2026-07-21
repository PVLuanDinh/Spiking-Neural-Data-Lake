"""
Capstone -- connected brain regions GROW TOGETHER through the shared growth substrate.

Wires two REAL regions to one GrowthCoordinator seam:
  * the growable multimodal encoder      (cortex.multimodal_encoder.MultiModalEncoder) -- the
    sensory surface, whose width is the number of feature channels across all modalities;
  * the growing spiking learner          (hippocampus.growing_learner.GrowingLearner) -- an
    unbounded, neurogenesis-driven memory that lives in that feature space.

When the encoder gains a NEW sensor modality its output width grows; the coordinator ripples that
growth to the learner, which widens its prototype space function-preservingly. The brain extends its
sensory surface and its memory follows automatically -- shape-consistent, with no relearning and no
forgetting. This is the "unbounded + working together" property on real components, not toy layers.

Proves:
  1. before growth, encoder and learner agree at the default width (N=64);
  2. adding a 5th modality grows the encoder to 80 and AUTO-grows the learner to 80 via the
     coordinator (one growth event, rippled across the connectome);
  3. every pattern learned at width 64 is still recognised after the coordinated widening
     (function-preserving -- the new modality is absent, so its dimensions are 0);
  4. the learner keeps recruiting units UNBOUNDEDLY on the wider substrate;
  5. the safety envelope is untouched -- no arm / action / E-STOP seam exists in the growth network.

Honest scope: "unbounded" = no fixed architectural ceiling, grows on demand. It does NOT mean the
brain can compute anything -- the fundamental wall (uncomputable functions, undecidable rule
identification from finite data) is untouched. And the effector / action / reflex envelope is fixed
BY DESIGN, structurally excluded by GrowthCoordinator (see substrate.growable._SAFETY_SEAMS).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.coordinated_growth
"""

from snn_data_lake.cortex.multimodal_encoder import MultiModalEncoder
from snn_data_lake.cortex.spike_preprocessing import N
from snn_data_lake.hippocampus.growing_learner import GrowingLearner
from snn_data_lake.substrate.growable import GrowthCoordinator

# 5 distinct multimodal signatures (visual, temperature, audio, distance levels).
_LEVELS = [
    (0.1, 0.1, 0.1, 0.1),
    (0.9, 0.9, 0.9, 0.9),
    (0.1, 0.9, 0.1, 0.9),
    (0.9, 0.1, 0.9, 0.1),
    (0.5, 0.5, 0.5, 0.5),
]


def main():
    enc = MultiModalEncoder()
    learner = GrowingLearner(vigilance=0.85, lr=0.2, dim=enc.width)

    # ONE seam: the sensory width. The learner is wired as a consumer, so it grows when the
    # encoder does. (register_seam would REFUSE a safety name -- the arm/action/E-STOP stay out.)
    coord = GrowthCoordinator()
    coord.register_seam("sensory_width", enc.width)
    coord.connect("sensory_width", learner.grow_dim)

    def enc4(lv):
        v, t, a, d = lv
        return enc.encode(visual=[v], temperature=t, audio=[a], distance=d)

    print("=" * 70)
    print("COORDINATED GROWTH -- encoder + learner grow together across the connectome")
    print("=" * 70)

    # 1. learn 5 signatures at width 64
    for _ in range(6):
        for lv in _LEVELS:
            learner.observe(enc4(lv))
    assert enc.width == N and learner.dim == N, f"both regions start at the default width {N}"
    assert learner.n_units == 5, f"5 distinct signatures -> 5 units, got {learner.n_units}"
    known = enc4(_LEVELS[2])
    known_id = learner.classify(known)
    print(
        f"1. learn @ W={N:<3d}        : {learner.n_units} units recruited; encoder.width={enc.width} "
        f"== learner.dim={learner.dim}"
    )

    # 2. add a 5th modality -> encoder grows -> coordinator ripples -> learner grows (one event)
    enc.add_modality("tactile", 16)
    rippled = coord.grow("sensory_width", enc.width)
    assert enc.width == N + 16, "adding a modality must grow the encoder width"
    assert learner.dim == N + 16 and rippled == 1, (
        "growth must RIPPLE through the coordinator to the learner (memory follows the sensory surface)"
    )
    print(
        f"2. +tactile -> W={enc.width}     : one growth event rippled to {rippled} consumer; "
        f"learner.dim auto-grew to {learner.dim}"
    )

    # 3. function-preserving: a pattern learned at 64, re-encoded with the new modality ABSENT
    #    (its band is zeros), still recognises to the SAME unit -- no relearning, no forgetting.
    known_wide = enc4(_LEVELS[2])  # tactile unspecified -> zeros in [64:80]
    assert len(known_wide) == N + 16 and known_wide[:N] == known, (
        "old bands unchanged; new band zero"
    )
    assert learner.classify(known_wide) == known_id, (
        "a pattern known before growth must still recognise after coordinated widening"
    )
    print(
        f"3. function-preserving   : pre-growth pattern still recognises to unit {known_id} "
        f"(bit-identical old bands)"
    )

    # 4. unbounded growth continues on the wider substrate: a brand-new signature recruits a new unit
    before = learner.n_units
    novel = enc.encode(visual=[0.7], temperature=0.3, audio=[0.7], distance=0.3, tactile=[0.95])
    uid, grew = learner.observe(novel)
    assert grew and learner.n_units == before + 1, (
        "the learner must keep recruiting unboundedly at the grown width"
    )
    print(
        f"4. still unbounded       : a new signature grew unit {uid} at W={enc.width} "
        f"-> {learner.n_units} units (no ceiling)"
    )

    # 5. the safety envelope stays out of the growth network: exact AND near-miss effector/reflex
    #    names are refused registration (the tripwire; the real bound is the brainstem ArmSim gate).
    unsafe = ("arm_dof", "ARM_DOF", "motor_actions", "e-stop", "effector_servo")
    refused = 0
    for seam in unsafe:
        try:
            coord.register_seam(seam, 2)
        except ValueError:
            refused += 1
    assert refused == len(unsafe), (
        f"the coordinator must refuse effector/action/E-STOP-looking seams ({refused}/{len(unsafe)})"
    )
    print(
        f"5. safety intact         : growable={sorted(coord.seams())}; "
        f"{refused}/{len(unsafe)} effector/reflex-looking seams refused"
    )
    print("=" * 70)
    print(
        "self-check OK: a new sensor modality grew the encoder AND, through the coordinator, the "
        "learner -- function-preserving, still unbounded, safety envelope untouched"
    )


if __name__ == "__main__":
    main()
