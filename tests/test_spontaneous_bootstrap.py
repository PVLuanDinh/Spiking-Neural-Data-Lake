"""specs/022 T045: the pre-experience spontaneous bootstrap fires at the REAL wire.

The wire sits inside `Brain._perceive_real`'s PerceptSurprise lazy-init, which is guarded by a
bare `except Exception: pass` -- a swallowed failure there would read as a completed bootstrap
to any observer but this one. These tests drive the PRODUCTION method with a stub encoder (the
perception encoder is duck-typed; no torch needed -- which also means they pin the CODE PATH,
not the deployment: the real wire fires only where the gitignored eprop checkpoint exists) and
assert:

  1. the bootstrap RAN at the wire and DELIVERED its steps (`_spont_bootstrap_steps` is
     initialised to None in `Brain.__init__`, so "never ran", "ran and delivered 0" and
     "completed 1600" are three distinguishable states -- the T045 panel measured the earlier
     getattr-default-0 reading conflating the first two);
  2. it installed STRUCTURE, not merely motion: the Oja subspace aligns to the declared modes
     (the panel measured a constant-DC mutant drive passing a weights-differ assert at
     alignment 0.19 while the real bootstrap sits at 0.93 -- the alignment bar is the
     discriminating red);
  3. the A/A holds (two same-seed bootstrapped brains, bit-identical weights), so the A/B
     cannot be harness noise; and
  4. neutering the bootstrap leaves the cold seed (charge, not reachability).

The module's own main() carries the stationarity / alignment-curve / null / refusal
measurements; this file pins only the WIRE, deliberately."""

import numpy as np

from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.substrate.spontaneous_activity import _cosine_modes, _principal_alignment


class _StubEnc:
    """Duck-typed stand-in for the torch perception encoder: a flat 64-dim latent."""

    def encode(self, frame):
        return [0.5] * 64

    def classify(self, frame):
        return 0


def _perceiving_brain():
    b = make_brain(seed=0)
    b._perceiving_real = True
    b._perception_enc = _StubEnc()
    b._real_frames = [object()]
    b._real_labels = [0]
    return b


def test_bootstrap_fires_and_installs_structure():
    b = _perceiving_brain()
    percept = b._perceive_real()
    assert percept is not None, "the stubbed production path did not produce a percept"
    assert b._percept_surprise is not None
    assert b._spont_bootstrap_steps == 1600, (
        f"the pre-experience bootstrap did not complete at the production wire "
        f"(recorded {b._spont_bootstrap_steps!r}; None = never ran, 0 = delivered nothing)"
    )
    # STRUCTURE, not motion: the panel measured a constant-DC drive passing a weights-differ
    # assert at alignment 0.19; the real bootstrap installs 0.93 at this configuration.
    w = np.asarray([n.w for n in b._percept_surprise._neurons])
    align = _principal_alignment(w, _cosine_modes(64, 3))
    assert align > 0.6, f"the wire bootstrap did not install the declared structure ({align:.3f})"
    # The canonical composition leaves the surprise baseline at the state ONE real observe
    # gives it (the bootstrap itself resets to cold; the percept above then seeds it) -- the
    # baseline must not carry 1600 steps of synthetic statistics.
    assert b._percept_surprise._var == 0.0, (
        "the baseline carries bootstrap statistics -- the canonical reset did not run"
    )


def test_wire_a_a_is_bit_identical():
    b1, b2 = _perceiving_brain(), _perceiving_brain()
    b1._perceive_real()
    b2._perceive_real()
    w1 = np.asarray([n.w for n in b1._percept_surprise._neurons])
    w2 = np.asarray([n.w for n in b2._percept_surprise._neurons])
    assert np.array_equal(w1, w2), "A/A differs -- the wire has run nondeterminism"


def test_neutered_bootstrap_leaves_the_cold_seed(monkeypatch):
    """Charge, not reachability: the bootstrap MOVES the rule's weights at the wire."""
    b1 = _perceiving_brain()
    b1._perceive_real()
    w_boot = np.asarray([n.w for n in b1._percept_surprise._neurons])

    from snn_data_lake.substrate import spontaneous_activity as sa

    monkeypatch.setattr(sa, "bootstrap_percept_rule", lambda ps, **kw: 0)
    b2 = _perceiving_brain()
    b2._perceive_real()
    w_cold = np.asarray([n.w for n in b2._percept_surprise._neurons])

    assert b2._spont_bootstrap_steps == 0
    assert not np.array_equal(w_boot, w_cold), (
        "the bootstrap changed nothing at the wire -- presence is not participation"
    )
