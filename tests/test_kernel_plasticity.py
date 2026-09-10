"""The kernel-side driver of the ONE R-STDP rule (specs/022 T063 2.1) -- and its reachable reds.

Nothing in this tree could write a weight into a `Synapses` bundle table before this module: the
table is compiled once and quantized once, and `KernelTick.tick` writes only v/refrac/last_spike.
T063 cannot ask "does learning move the brain off the seed" without a learner, so the learner comes
with the row -- and with the invariants that make it safe to attribute anything to it.

Committed RED, before `substrate/kernel_plasticity.py` exists.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_kernel_plasticity.py -q --no-cov
"""

import pytest

np = pytest.importorskip("numpy")

from snn_data_lake.basal_ganglia import r_stdp  # noqa: E402

N_UNITS = 12


def _plast(sign_preserving=True, lr=r_stdp.LR, frac_bits=16, weights=None):
    from snn_data_lake.substrate.kernel_plasticity import KernelPlasticity
    from snn_data_lake.substrate.kernel_state import KernelState
    from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses

    #: two out-edges per unit on a ring; units 8..11 are the inhibitory sources, so the sign
    #: clamp has both polarities to protect
    pre = [u for u in range(N_UNITS) for _ in (0, 1)]
    post = [(u + 1 + 4 * k) % N_UNITS for u in range(N_UNITS) for k in (0, 1)]
    if weights is None:
        weights = [(0.20 if u < 8 else -0.20) for u in range(N_UNITS) for _ in (0, 1)]
    state = KernelState([("kp", N_UNITS)])
    syn = Synapses(
        pre=pre,
        post=post,
        weight=weights,
        delay=[1] * len(pre),
        n_units=N_UNITS,
        p_parts=1,
        frac_bits=frac_bits,
        max_delay=1,
    )
    kt = KernelTick(state, syn, decay=0.9, v_reset=0.0, refrac_ticks=0)
    return KernelPlasticity(kt, lr=lr, sign_preserving=sign_preserving)


def _drive(plast, rewards, ticks=12):
    """A scripted spike sequence -- every unit fires on its own phase, so causal and anti-causal
    pairs both form on the ring."""
    for t in range(1, ticks + 1):
        fired = np.asarray([u for u in range(N_UNITS) if (u + t) % 3 == 0], dtype=np.int64)
        plast.observe(fired, float(t))
        plast.decay()
        if t in rewards:
            plast.reward(rewards[t])
    return plast


def test_the_bundle_table_is_actually_written():
    plast = _plast()
    before = plast.qflat.copy()
    _drive(plast, {4: +1.0, 8: -1.0})
    assert not np.array_equal(before, plast.qflat), "the learner must move the kernel's weights"
    #: the bundles the tick reads must see the change -- a learner writing a private copy is a null
    key = next(iter(plast.kt.syn.bundles))
    posts, q = plast.kt.syn.bundles[key]
    assert q.base is not None or q.flags["OWNDATA"] is False, (
        "each bundle's qweight must be a VIEW into the learner's flat array, or the tick reads "
        "the pre-learning weights"
    )


# ---- SIGN PRESERVATION (SC-004 as a checked invariant) and its reachable red -----------------
def test_sign_is_preserved_through_learning():
    plast = _plast(lr=40.0)
    sign_before = np.sign(plast.qflat.copy())
    _drive(plast, {2: +1.0, 4: +1.0, 6: -1.0, 8: -1.0, 10: +1.0, 12: -1.0})
    sign_after = np.sign(plast.qflat)
    crossed = int(np.count_nonzero((sign_before * sign_after) < 0))
    assert crossed == 0, f"{crossed} weights crossed zero -- Dale is an invariant, not a tunable"
    assert plast.clamped > 0, (
        "at this learning rate the clamp must have DONE something -- a criterion that holds only "
        "because nothing reached zero is vacuous"
    )


def test_removing_the_clamp_flips_a_sign_the_red():
    """The RED for the invariant above: the same sequence with the clamp removed must produce a
    flip. If it cannot, the clamp is untested decoration."""
    plast = _plast(lr=40.0, sign_preserving=False)
    sign_before = np.sign(plast.qflat.copy())
    _drive(plast, {2: +1.0, 4: +1.0, 6: -1.0, 8: -1.0, 10: +1.0, 12: -1.0})
    sign_after = np.sign(plast.qflat)
    crossed = int(np.count_nonzero((sign_before * sign_after) < 0))
    assert crossed > 0, "no flip occurred with the clamp removed -- the red is unreachable"


def test_a_weight_may_reach_zero_but_never_cross():
    plast = _plast(lr=40.0)
    _drive(plast, {2: +1.0, 4: -1.0, 6: -1.0, 8: -1.0, 10: -1.0, 12: -1.0})
    started_pos = plast.q_start > 0
    started_neg = plast.q_start < 0
    assert np.all(plast.qflat[started_pos] >= 0)
    assert np.all(plast.qflat[started_neg] <= 0)


# ---- DETERMINISM ----------------------------------------------------------------------------
def test_same_seed_same_rewards_gives_byte_equal_weights():
    import hashlib

    def go():
        p = _plast()
        _drive(p, {3: +1.0, 6: -1.0, 9: +1.0})
        return hashlib.sha256(p.weight_bytes()).hexdigest()

    a, b = go(), go()
    assert a == b, f"the learner is not deterministic: {a} vs {b}"


def test_a_different_reward_sequence_gives_different_weights():
    import hashlib

    def go(rewards):
        p = _plast()
        _drive(p, rewards)
        return hashlib.sha256(p.weight_bytes()).hexdigest()

    assert go({3: +1.0, 6: -1.0, 9: +1.0}) != go({3: -1.0, 6: +1.0, 9: -1.0})


# ---- THE FOUR DIAGNOSTIC LINES (2.5) --------------------------------------------------------
def test_the_four_diagnostics_are_reported_separately():
    plast = _plast()
    _drive(plast, {4: +1.0, 8: -1.0})
    d = plast.diagnostics()
    for k in ("executed", "attempted", "effective_count", "effective_l1"):
        assert k in d, f"diagnostic {k} missing -- a zero must be readable from four lines"
    assert d["executed"] == 2, "one execution per reward delivered"
    assert d["attempted"] > 0
    assert d["effective_count"] > 0
    assert d["effective_l1"] > 0.0


def test_zero_lr_executes_and_attempts_but_changes_nothing():
    """The FROZEN arm's mechanism: the pin the harness itself applies. Executed and attempted must
    both be positive, so red 1 can attribute the zero to the VERIFIED pin and to nothing else."""
    plast = _plast(lr=0.0)
    before = plast.qflat.copy()
    _drive(plast, {4: +1.0, 8: -1.0})
    d = plast.diagnostics()
    assert d["executed"] > 0 and d["attempted"] > 0
    assert d["effective_count"] == 0 and d["effective_l1"] == 0.0
    assert np.array_equal(before, plast.qflat)


def test_an_update_below_quantization_is_not_reported_as_pinning():
    """attempted > 0 with effective == 0 is 'cancelled or below quantization', never 'pinned'."""
    plast = _plast(lr=1e-12)
    _drive(plast, {4: +1.0})
    d = plast.diagnostics()
    assert d["attempted"] > 0 and d["effective_count"] == 0


def test_reset_clears_eligibility_and_spike_memory_but_not_weights():
    plast = _plast()
    _drive(plast, {4: +1.0})
    w = plast.qflat.copy()
    plast.reset()
    assert float(np.abs(plast.elig).sum()) == 0.0
    assert np.array_equal(w, plast.qflat)


# ---- THE GRADIENT CLAIM ---------------------------------------------------------------------
def test_gradient_claim_is_local_with_a_reason():
    import ast
    import pathlib

    src = pathlib.Path("src/snn_data_lake/substrate/kernel_plasticity.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    decl = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "GRADIENT_CLAIM" for t in node.targets
        ):
            decl = ast.literal_eval(node.value)
    assert isinstance(decl, dict) and decl, "GRADIENT_CLAIM must be a module-scope literal dict"
    assert any(v[0] == "LOCAL" for v in decl.values()), "the reward gate's claim is LOCAL"
    assert all(len(v[1]) > 40 for v in decl.values()), "every claim carries a reason"
