"""The P2 MERGE of R-STDP: ONE authoritative implementation, two substrates (specs/022 T063 2.1).

`basal_ganglia/r_stdp.py` owned the three-factor rule as per-object list arithmetic inside
`RSTDPNeuron`. T063 needs the same rule over a kernel bundle table. P2 forbids a second
implementation, so the rule's operations are lifted to module-level array functions that BOTH forms
call, and this file is the behavioural verification P2 demands: identical input, identical weights.

These tests are committed RED, before the merge exists.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_r_stdp_kernel_equivalence.py -q --no-cov
"""

import math

import pytest

np = pytest.importorskip("numpy")

from snn_data_lake.basal_ganglia import r_stdp  # noqa: E402

#: The incumbent's pinned trajectory, measured at 851b026 BEFORE the merge. The merge is only a
#: merge if the surviving implementation reproduces it exactly -- a "merge" that moves the
#: incumbent's numbers is a rewrite wearing the word.
RUN7 = [0.5587212259170016, 0.5453367971886981, 0.4783055031868412]
E_AFTER = [0.8598476986592055, 0.903933032885864, -0.8607079764250578]
W_AFTER = [0.5122901694889702, 0.5241786989260729, 0.5295193565565697]


def test_the_four_operations_are_module_level():
    """The rule's operations live ONCE, at module scope, callable over any indexable container."""
    for name in (
        "accumulate_causal",
        "accumulate_anticausal",
        "decay_eligibility",
        "reward_deltas",
    ):
        assert callable(getattr(r_stdp, name)), f"r_stdp.{name} must be the authoritative form"


def test_incumbent_trajectory_is_byte_identical_after_the_merge():
    assert r_stdp._run(7) == RUN7
    n = r_stdp.RSTDPNeuron(3, seed=5)
    n.pre_spike(0, 1.0)
    n.pre_spike(1, 2.0)
    n.post_spike(4.0)
    n.step()
    n.pre_spike(2, 7.0)
    assert n.e == E_AFTER
    assert n.w == W_AFTER


def test_the_neuron_calls_the_module_functions():
    """Not "there exist functions that agree" -- the per-object form must GO THROUGH them, else
    there are still two implementations and one of them is merely tested."""
    calls = []
    real = {
        k: getattr(r_stdp, k)
        for k in (
            "accumulate_causal",
            "accumulate_anticausal",
            "decay_eligibility",
            "reward_deltas",
        )
    }

    def spy(name):
        def go(*a, **k):
            calls.append(name)
            return real[name](*a, **k)

        return go

    for k in real:
        setattr(r_stdp, k, spy(k))
    try:
        n = r_stdp.RSTDPNeuron(2, seed=0)
        n.pre_spike(0, 1.0)
        n.post_spike(3.0)
        n.pre_spike(1, 5.0)
        n.step()
        n.reward(0.5)
    finally:
        for k, v in real.items():
            setattr(r_stdp, k, v)
    assert set(calls) == set(real), f"the per-object form bypassed {set(real) - set(calls)}"


# --------------------------------------------------------------------------------------------
# The equivalence test proper: one spike/reward sequence through both forms.
# --------------------------------------------------------------------------------------------
N_POST = 4
N_SYN = 3
#: unit ids: 0..N_POST-1 are the post units, the rest are pre units
PRE0 = N_POST
N_UNITS = N_POST + N_POST * N_SYN

#: (tick, fired unit ids) -- scripted, so the comparison is of the RULE and not of two dynamics
SCRIPT = [
    (1, [PRE0 + 0, PRE0 + 1, PRE0 + 5]),
    (2, [0, 1]),
    (3, [PRE0 + 3, PRE0 + 7]),
    (4, [1, 2]),
    (5, [PRE0 + 2, PRE0 + 9, PRE0 + 11]),
    (6, [0, 3]),
    (7, [PRE0 + 4, PRE0 + 6, PRE0 + 8, PRE0 + 10]),
    (8, [2, 3]),
]
#: tick -> reward delivered AFTER that tick's pairing and decay
REWARDS = {4: +1.0, 8: -1.0}


def _edges():
    """post unit q <- pre units PRE0 + q*N_SYN + i, i = 0..N_SYN-1. Weights well away from zero so
    the kernel's sign clamp cannot fire and the comparison is of the shared rule alone."""
    pre, post, w = [], [], []
    for q in range(N_POST):
        for i in range(N_SYN):
            pre.append(PRE0 + q * N_SYN + i)
            post.append(q)
            w.append(0.40 + 0.03 * (q * N_SYN + i))
    return pre, post, w


def _kernel_form(frac_bits=16, sign_preserving=True):
    from snn_data_lake.substrate.kernel_plasticity import KernelPlasticity
    from snn_data_lake.substrate.kernel_state import KernelState
    from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses

    pre, post, w = _edges()
    state = KernelState([("equiv", N_UNITS)])
    syn = Synapses(
        pre=pre,
        post=post,
        weight=w,
        delay=[1] * len(pre),
        n_units=N_UNITS,
        p_parts=1,
        frac_bits=frac_bits,
        max_delay=1,
    )
    kt = KernelTick(state, syn, decay=0.9, v_reset=0.0, refrac_ticks=0)
    return KernelPlasticity(kt, sign_preserving=sign_preserving)


def _run_kernel(plast):
    for t, fired in SCRIPT:
        plast.observe(np.asarray(sorted(fired), dtype=np.int64), float(t))
        plast.decay()
        if t in REWARDS:
            plast.reward(REWARDS[t])
    return plast


def _run_neurons(w_start):
    """The per-object mirror, in the SAME intra-tick order the kernel form freezes: every fired
    post unit pairs first (against pre times from before this tick), then every fired pre unit."""
    cells = [r_stdp.RSTDPNeuron(N_SYN, seed=q) for q in range(N_POST)]
    for q, cell in enumerate(cells):
        for i in range(N_SYN):
            cell.w[i] = w_start[(PRE0 + q * N_SYN + i, q)]
    for t, fired in SCRIPT:
        for u in sorted(fired):
            if u < N_POST:
                cells[u].post_spike(float(t))
        for u in sorted(fired):
            if u >= N_POST:
                q, i = divmod(u - PRE0, N_SYN)
                cells[q].pre_spike(i, float(t))
        for cell in cells:
            cell.step()
        if t in REWARDS:
            for cell in cells:
                cell.reward(REWARDS[t])
    return cells


def test_kernel_and_per_object_agree_within_one_quantization_step():
    plast = _kernel_form()
    scale = float(1 << plast.frac_bits)
    start = {
        (int(a), int(b)): float(q) / scale
        for a, b, q in zip(plast.edge_pre, plast.edge_post, plast.qflat, strict=True)
    }
    _run_kernel(plast)
    cells = _run_neurons(start)

    step = 1.0 / scale
    worst = 0.0
    for j in range(plast.n_edges):
        a, b = int(plast.edge_pre[j]), int(plast.edge_post[j])
        q, i = b, (a - PRE0) - b * N_SYN
        worst = max(worst, abs(float(plast.qflat[j]) / scale - cells[q].w[i]))
    assert plast.clamped == 0, "the sign clamp must not fire in the equivalence sequence"
    assert worst <= step, f"kernel vs per-object max |dw| = {worst:.3e} exceeds one step {step:.3e}"


def test_equivalence_is_not_vacuous():
    """A red for the equivalence test itself: perturb the kernel form's rule constants and the
    comparison must FAIL. Otherwise the test above would pass for two forms that never learn."""
    plast = _kernel_form()
    plast.a_plus = plast.a_plus * 2.0
    scale = float(1 << plast.frac_bits)
    start = {
        (int(a), int(b)): float(q) / scale
        for a, b, q in zip(plast.edge_pre, plast.edge_post, plast.qflat, strict=True)
    }
    _run_kernel(plast)
    cells = _run_neurons(start)
    worst = 0.0
    for j in range(plast.n_edges):
        a, b = int(plast.edge_pre[j]), int(plast.edge_post[j])
        q, i = b, (a - PRE0) - b * N_SYN
        worst = max(worst, abs(float(plast.qflat[j]) / scale - cells[q].w[i]))
    assert worst > 1.0 / scale, (
        "a doubled A_plus must break the equivalence -- else it proves nothing"
    )


def test_module_functions_accept_a_numpy_container():
    e = np.zeros(4, dtype=np.float64)
    r_stdp.accumulate_causal(e, [(0, 5.0), (2, 20.0)])
    assert e[0] == pytest.approx(r_stdp.A_PLUS * math.exp(-5.0 / r_stdp.TAU_PLUS))
    assert e[1] == 0.0
    r_stdp.accumulate_anticausal(e, [(1, -5.0)])
    assert e[1] < 0.0
    before = e.copy()
    r_stdp.decay_eligibility(e, 1.0, r_stdp.TAU_E)
    assert np.all(np.abs(e) <= np.abs(before))
    d = r_stdp.reward_deltas(e, 1.0, 0.1)
    assert len(d) == 4
