"""Hippocampus region: memory formation, spatial coding, replay.

The same-named `hippocampus.hippocampus` module was RETIRED in the Policy-3 wave: its three roles are
all already live AND charged elsewhere -- completion-from-partial-cue as EpisodicRecall.complete,
nearest-stored-item as GrowingLearner, sparse generalizing hashing as GeneralizingNovelty, and the
attractor store as sleep_replay.AttractorNet. Its own self-check measured exact parity (1.000 vs 1.000)
against the live path, i.e. zero gain. The region keeps its live members -- import them directly, e.g.
`from snn_data_lake.substrate.episodic_recall import EpisodicRecall`.
"""
