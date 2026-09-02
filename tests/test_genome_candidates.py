"""FR-033's seam: the candidate hook in NGSNN.fit, the CandidatePool, and the engine's pool modes.

Pins: (1) the default path is byte-identical -- fit() without the hook and fit() with the
incumbent pool (genome=None) produce the same weights, because the hook never touches the seeded
stream; (2) provenance covers exactly the admitted units; (3) the genome replaces rows only where
the candidate's tag is in the fitted span, rescaled to the pool's envelope, and counts every
refusal; (4) the engine runs in all three modes with its placement list aligned through growth
and prune, and reports the pool census on its state.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from snn_data_lake.cortex.genomic_bottleneck import (
    CandidatePool,
    GenomicMap,
    feature_tags,
    unit_tags,
)
from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove
from snn_data_lake.cortex.placement import place
from snn_data_lake.hippocampus.ng_snn import NGSNN


def _task(seed=0, n=120, d=16, m=3):
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(m, d, generator=g)
    y = torch.randint(0, m, (n,), generator=g)
    x = (centers[y] + 0.15 * torch.rand(n, d, generator=g)).clamp(0, 1)
    return x, y


def _placer(n_cand, step, lam):
    return place(n_cand, seed=1000 * step + int(lam * 10))


def test_default_path_is_byte_identical_with_the_incumbent_pool():
    x, y = _task()
    a = NGSNN(device="cpu", seed=3, T=5).fit(x, y)
    b = NGSNN(device="cpu", seed=3, T=5).fit(x, y, candidates=CandidatePool(_placer, "cortex"))
    assert a.n_units > 0
    assert torch.equal(a.W, b.W) and torch.equal(a.b, b.b)
    assert len(b.last_provenance) == b.n_units - b.n_seeded
    assert all(src == "random" for _p, src in b.last_provenance)
    assert a.last_provenance == []


def test_genome_replaces_rows_in_span_only_and_counts_refusals():
    d = 16
    placed = place(6, seed=5)  # the "earned" units' placements
    W = np.random.default_rng(0).normal(size=(6, d))
    genome = GenomicMap()
    genome.fit(W, feature_tags(d), unit_tags(placed, "cortex"))
    pool = CandidatePool(_placer, "cortex", genome=genome, ftags=feature_tags(d))
    rows = np.random.default_rng(1).uniform(-1, 1, size=(40, d))
    out, prov = pool.init(rows, 1.0, 0)
    assert out.shape == rows.shape and len(prov) == 40
    assert pool.stats["drawn"] == 40
    assert pool.stats["genome"] + pool.stats["fallback"] + pool.stats["degenerate"] == 40
    envelope = 1.0 * math.sqrt(d / 3.0)
    for i, (_p, src) in enumerate(prov):
        if src == "genome":
            assert not np.allclose(out[i], rows[i])
            assert abs(np.linalg.norm(out[i]) - envelope) < 1e-9
        else:
            assert np.array_equal(out[i], rows[i])
    # a genome fitted on ONE tag family refuses every other family: the refusal is reachable
    one = GenomicMap()
    one.fit(W[:2], feature_tags(d), unit_tags([placed[0], placed[0]], "cortex"))
    strict = CandidatePool(_placer, "cortex", genome=one, ftags=feature_tags(d))
    strict.init(rows, 1.0, 1)
    assert strict.stats["fallback"] > 0


def test_engine_modes_run_and_placement_stays_aligned():
    seen_genome_fit = False
    for mode in ("random", "genome", "genome_shuffled"):
        eng = NeurogenesisSelfImprove(seed=2, device="cpu", refit_every=20, init_pool=mode)
        for _ in range(140):
            eng.tick()
        assert eng.grown_units > 0
        assert len(eng.placement) == eng.grown_units, (mode, len(eng.placement), eng.grown_units)
        st = eng.state()
        assert st["init_pool"] == mode and st["pool_stats"]["mode"] == mode
        assert st["pool_stats"]["drawn"] > 0
        if mode == "random":
            assert st["pool_stats"]["genome"] == 0 and st["pool_stats"]["genome_fit"] is False
        else:
            seen_genome_fit |= bool(st["pool_stats"]["genome_fit"])
        for _d, layer, sign in eng.placement:  # Dale survives the new path
            assert not (layer == "Layer 1" and sign == 1)
    assert seen_genome_fit, "neither genome mode ever fitted a map on the earned units"
