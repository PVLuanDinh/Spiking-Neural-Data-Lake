"""specs/009 -- connectivity may be thinned only if the module still does its job.

The feature's whole risk is that thinning is a BEHAVIOURAL change wearing a storage change's
clothes: every memory-and-fanout number improves monotonically as connections are deleted, so a
suite that checked only those would be maximised by destroying the brain. Function preservation is
therefore asserted FIRST here, and the savings are asserted second.

The tolerance is pre-registered rather than fitted: a sparse store must match the dense store's own
`retained()` at the same width, load and seed. specs/009 research fixed it before any sweep ran.

Run: pytest tests/test_sparse_connectivity.py
"""

import numpy as np
import pytest

from snn_data_lake.sleep.episode_consolidation import EpisodeConsolidator

#: Pre-registered in specs/009 research.md BEFORE the sweep that measured the knee.
TOLERANCE = 0.02

#: Measured F* is 160 at N=1,024 and 128 at N=16,384 at fixed load; 128 passes at both widths
#: tested here. NOT a default in the module -- specs/009 forbids one, because the knee tracks
#: stored LOAD rather than width, so any baked-in number would be a magic constant.
FANOUT = 128


def build(dim, fanout, load=40, seed=0):
    e = EpisodeConsolidator(dim=dim, capacity=64, seed=seed, fanout=fanout)
    for i in range(load):
        ep = {"who": f"a{i}", "what": f"w{i}", "where": f"p{i % 7}", "when": str(i)}
        e.store(e.encode(ep), ep)
    return e


def nnz(e):
    return int((np.asarray(e.W) != 0).sum())


@pytest.mark.parametrize("dim", [256, 1024])
def test_function_survives_thinning(dim):
    """FR-001: the module's OWN task metric, within the pre-registered tolerance of dense."""
    dense = build(dim, None)
    sparse = build(dim, FANOUT)
    base = dense.retained()
    assert base > 0.0, "the dense baseline must actually work, or the comparison is vacuous"
    got = sparse.retained()
    assert abs(got - base) <= TOLERANCE, (
        f"dim={dim}: retained {got:.3f} vs dense {base:.3f}, outside the "
        f"pre-registered tolerance {TOLERANCE}"
    )


def test_thinning_actually_removes_connections():
    """The saving is real -- and asserted SECOND, because on its own it is maximised by deleting
    everything. It only means anything alongside the function test above."""
    dim = 1024
    dense, sparse = build(dim, None), build(dim, FANOUT)
    assert nnz(sparse) < nnz(dense) / 3, (
        f"expected a large reduction, got {nnz(sparse):,} vs dense {nnz(dense):,}"
    )
    mean_deg, _ = sparse.degree()
    assert mean_deg < dim / 3, f"mean degree {mean_deg:.1f} is not meaningfully below dim {dim}"


def test_wiring_is_deterministic():
    """FR-010: identical seeds must produce identical wiring. A random topology that is not
    reproducible would take the gates' seeded-structure property with it."""
    a, b = build(512, FANOUT, seed=7), build(512, FANOUT, seed=7)
    assert np.array_equal(a._mask, b._mask), "same seed produced different wiring"
    c = build(512, FANOUT, seed=8)
    assert not np.array_equal(a._mask, c._mask), "different seeds produced identical wiring"


def test_mask_is_symmetric_with_zero_diagonal():
    """The covariance rule is symmetric, so a one-way connection would not be a thinned version of
    the dense dynamics -- it would be a different system."""
    e = build(256, FANOUT)
    assert np.array_equal(e._mask, e._mask.T), "asymmetric wiring under a symmetric learning rule"
    assert not e._mask.diagonal().any(), "a unit must not connect to itself"


def test_weights_never_escape_the_mask():
    """Every update routes through _zero_offmask(). Storing, replaying and evicting must all leave
    absent connections at exactly zero -- otherwise the store is sparse only on paper."""
    e = build(256, FANOUT)
    e.replay(steps=20)
    for i in range(70):  # past capacity=64, to drive ring eviction as well
        ep = {"who": f"z{i}", "what": "x", "where": "p", "when": str(i)}
        e.store(e.encode(ep), ep)
    off = np.asarray(e.W)[~e._mask]
    assert not off.any(), f"{int((off != 0).sum())} weights outside the wiring mask"


def test_a_fixed_fanout_collapses_as_the_store_grows():
    """THE REASON `fanout` HAS NO DEFAULT, asserted rather than described.

    A fanout that holds at one capacity destroys memory at a larger one. Measured:

        capacity 128, dim  512, F=192 -> retained 0.992   (fine)
        capacity 512, dim 2048, F=192 -> retained 0.000   (total loss)

    So a constant F wired into brain.py would pass every test today and silently annihilate
    episodic memory the moment the store grew -- which is precisely what specs/007 exists to make
    happen. F must be re-measured whenever capacity changes; this test fails if anyone ever decides
    otherwise.
    """
    fixed_f = 192

    def at_capacity(cap):
        # capacity drives BOTH the store size and dim (derived), which is the whole point: the
        # generic build() helper pins capacity at 64, so using it here would just exercise ring
        # eviction and prove nothing about load.
        e = EpisodeConsolidator(capacity=cap, seed=0, fanout=fixed_f)
        for i in range(cap):
            ep = {"who": f"a{i}", "what": f"w{i}", "where": f"p{i % 7}", "when": str(i)}
            e.store(e.encode(ep), ep)
        return e.retained()

    ok = at_capacity(128)  # dim 512
    assert ok > 0.9, (
        f"F={fixed_f} should hold at capacity 128; got {ok:.3f}. If this changed, the measured "
        "point behind this test moved and the comment above is stale."
    )
    grown = at_capacity(512)  # dim 2048, same F, 4x the load
    assert grown < 0.1, (
        f"F={fixed_f} unexpectedly survived a 4x larger store ({grown:.3f}). That would be good "
        "news, and it would mean the no-default rule needs re-deriving rather than trusting."
    )


def test_dense_is_unchanged_by_default():
    """fanout=None must be byte-identical to the historical all-to-all path: this feature may not
    silently alter a brain that never asked for it."""
    e = build(256, None)
    assert e._mask is None
    assert e.degree() == (255.0, 255)
    w = np.asarray(e.W)
    assert not w.diagonal().any()
    assert (w != 0).sum() > 0.9 * 256 * 255, "dense wiring should stay dense"
