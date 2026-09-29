"""The P2 MERGE of R-STDP: ONE authoritative implementation (specs/022 T063 2.1).

`plasticity/r_stdp.py` owned the three-factor rule as per-object list arithmetic inside
`RSTDPNeuron`. T063 needed the same rule over a kernel bundle table. P2 forbids a second
implementation, so the rule's operations were lifted to module-level functions that BOTH forms
call, and this file was the behavioural verification P2 demands: identical input, identical
weights. Those tests were committed RED, before the merge existed.

DISPOSITION 2026-09-09. THE SECOND SUBSTRATE IS GONE: `substrate/kernel_plasticity` was DELETED on
its own retention criterion (T063's campaign read t(d_f) = -13.0154, the frozen twin ahead of the
learning arm on every one of the 8 seeds; 2.9 -- a learner that moves weights without an
attributed benefit is a null). What that does and does not change here:

  KEPT, because their subject is the MERGE and the merge stands -- the rule's operations live once
  at module scope, `RSTDPNeuron` GOES THROUGH them rather than merely agreeing with them, and the
  incumbent's trajectory is byte-identical to its pre-merge pin at 851b026. A merge is a claim
  about the surviving implementation, and it survives.
  ADDED: the P2 pin that exactly ONE substrate now exists -- the deleted module must not come back
  by import.
  DELETED, because their subject was the deleted kernel form: the two-substrate equivalence
  comparison, its not-vacuous red (a doubled A_plus must break the comparison), and the numpy-
  container test -- the last together with the container branch itself, which had no caller left.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_r_stdp_kernel_equivalence.py -q --no-cov
"""

import importlib

import pytest

pytest.importorskip("numpy")

from snn_data_lake.plasticity import r_stdp  # noqa: E402

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
# P2 after the deletion: ONE substrate, and the deleted one must not come back.
# --------------------------------------------------------------------------------------------
def test_the_kernel_form_is_deleted_and_the_rule_has_ONE_substrate():
    """P2 asks that exactly one mechanism survive. It does -- `RSTDPNeuron` over the module-level
    functions -- and the kernel form that motivated lifting them is GONE, deleted on its own
    retention criterion rather than kept because it was already written. This is the pin: an
    import of it must FAIL, so a later change cannot quietly restore a second substrate."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("snn_data_lake.substrate.kernel_plasticity")
