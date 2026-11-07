"""D10 (DESIGN-ROWS-1106, PREREG 4fc3ff89c2e9): apical-compartment entry of the readout's return.

Section-6 unit reds on `growth/grown_code.py`. The return afferents form an APICAL potential
A(t) = sum w * psp(t - arrival, tau); the soma drive is D(t) = B(t) * (1 + tanh(A(t) / theta)),
where B(t) is the unchanged frozen-code (basal) drive. B = 0 => D = 0: the return alone never fires
a unit (isolation), which the additive entry it replaces (B + A) did. Constructed fixtures only.
"""

import math
from types import SimpleNamespace

import pytest

from snn_data_lake.growth import grown_code as gc
from snn_data_lake.growth.grown_code import GrownUnit

TAU = 8.0
WINDOW = 12
PEAK = math.exp(-1.0 / TAU)  # psp one step after a rank-0 input


def _base(unit, ranks, t):
    return sum(w * gc._psp(t - r, TAU) for w, r in zip(unit.w_in, ranks, strict=False) if w != 0.0)


def _apical(unit, aff, t):
    return sum(w * gc._psp(t - arrival, TAU) for arrival, w in aff.get(unit.unit_id, ()))


def _ref_latency(unit, ranks, aff, mode):
    """Reference crossing: mode 'additive' is the 2f5a9c22 entry, 'apical' the D10 entry."""
    for t in range(1, unit.window + 1):
        b, a = _base(unit, ranks, t), _apical(unit, aff, t)
        d = b + a if mode == "additive" else b * (1.0 + math.tanh(a / unit.threshold))
        if d >= unit.threshold:
            return t
    return None


def _unit(uid=0, w=(0.0,), theta=1.0, sign=1, window=WINDOW):
    return GrownUnit(uid, tuple(w), theta, window, sign, (0.0,))


# R1 isolation fixture: zero basal drive, one return arrival large enough to cross additively.
R1_UNIT = _unit(w=(0.0,))
R1_AFF = {0: [(0.0, 5.0)]}

# R2 unit-level fixture: basal peak = 0.8 theta (sub-threshold alone), positive return arrival.
SUB_THETA = PEAK / 0.8
SUB_UNIT = _unit(w=(1.0,), theta=SUB_THETA)
SUB_AFF = {0: [(0.0, 0.5)]}


def test_apical_alone_never_fires():
    """R1 (decides the row): the return alone must not fire a unit with zero basal drive."""
    assert gc.latency(R1_UNIT, [0], TAU, afferents=R1_AFF) is None
    assert gc.latencies((R1_UNIT,), [0], TAU, afferents=R1_AFF) == [None]
    assert gc.segment_psp((R1_UNIT,), [0], TAU, WINDOW + 3.0, afferents=R1_AFF) == [0.0]


def test_additive_fires_on_isolation_fixture():
    """R4 (control arm): the R1 fixture DOES fire under the additive entry, so R1 discriminates."""
    assert _ref_latency(R1_UNIT, [0], R1_AFF, "additive") == 1
    assert _ref_latency(R1_UNIT, [0], R1_AFF, "apical") is None


def _byte(xs):
    return repr([None if x is None else float(x).hex() for x in xs])


def test_zero_apical_is_byte_identical():
    """A = 0 (no pending arrival, or neutered) leaves every crossing byte-identical to the
    no-return path."""
    units = (
        _unit(0, w=(0.7, 0.4), theta=0.9),
        _unit(1, w=(1.3, 0.0), theta=1.0),
        _unit(2, w=(0.2, 0.2), theta=1.5, sign=-1),
    )
    ranks = [0, 3]
    empty = {u.unit_id: [] for u in units}
    for aff in (empty, {}):
        assert _byte(gc.latencies(units, ranks, TAU, afferents=aff)) == _byte(
            gc.latencies(units, ranks, TAU)
        )
        assert _byte(gc.segment_psp(units, ranks, TAU, 15.0, afferents=aff)) == _byte(
            gc.segment_psp(units, ranks, TAU, 15.0)
        )
        assert gc.column_latencies(units, ranks, TAU, 0.5, afferents=aff) == gc.column_latencies(
            units, ranks, TAU, 0.5
        )


def test_no_afferents_path_byte_identical():
    """A2.C1: afferents=None skips the apical helper entirely (batch, column_batch, plain latency);
    the pre-D10 base-only crossing holds byte-for-byte, even for a theta <= 0 unit (no return,
    no refusal)."""
    units = (
        _unit(0, w=(0.7, 0.4), theta=0.9),
        _unit(1, w=(0.0, 0.0), theta=-1.0),
        _unit(2, w=(0.5, 0.0), theta=0.0),
    )
    ranks = [0, 3]
    expected = [_ref_latency(u, ranks, {}, "additive") for u in units]
    assert gc.latencies(units, ranks, TAU) == expected
    for u in units:
        for t in range(1, WINDOW + 1):
            assert repr(gc._drive(u, ranks, TAU, t)) == repr(_base(u, ranks, t))


def test_apical_gains_subthreshold_basal():
    """R2 at unit level: a sub-threshold basal drive crosses only with the return; the drive is
    the multiplicative gate, not the sum."""
    peak = max(_base(SUB_UNIT, [0], t) for t in range(1, WINDOW + 1))
    assert 0.5 <= peak / SUB_THETA < 1.0
    assert gc.latency(SUB_UNIT, [0], TAU, afferents={0: []}) is None
    assert gc.latency(SUB_UNIT, [0], TAU, afferents=SUB_AFF) is not None
    for t in range(1, WINDOW + 1):
        b, a = _base(SUB_UNIT, [0], t), _apical(SUB_UNIT, SUB_AFF, t)
        assert gc._drive(SUB_UNIT, [0], TAU, t, afferents=SUB_AFF) == b * (
            1.0 + math.tanh(a / SUB_THETA)
        )


def test_inhibitory_return_shunts():
    """B > 0 crosses alone; an inhibitory return SHUNTS it below threshold (divisive), where the
    additive entry (B + A, A < 0) would still cross."""
    theta = PEAK / 1.5  # basal peak = 1.5 theta
    unit = _unit(w=(1.0,), theta=theta)
    aff = {0: [(0.0, -0.4 * theta / PEAK)]}  # A(1) = -0.4 theta
    assert gc.latency(unit, [0], TAU) == 1
    assert _ref_latency(unit, [0], aff, "additive") == 1
    assert gc.latency(unit, [0], TAU, afferents=aff) is None


def _charges():
    return gc.latency(SUB_UNIT, [0], TAU, afferents=SUB_AFF) != gc.latency(
        SUB_UNIT, [0], TAU, afferents={0: []}
    )


def test_planted_disconnected_apical_is_caught(monkeypatch):
    """R3: both disconnected-apical plants must make the R2 charge disappear."""
    assert _charges()
    with monkeypatch.context() as m:
        # Plant 1: A is computed but the gain is forced to 1 (D = B).
        m.setattr(gc, "_apical_gain", lambda apical, theta: 1.0, raising=False)
        assert not _charges(), "gain=1 plant was not caught"
    with monkeypatch.context() as m:
        # Plant 2: afferents reach the call but _drive reads an empty apical dict.
        original = gc._drive

        def blind(unit, ranks, tau, t, *, afferents=None):
            return original(unit, ranks, tau, t, afferents=None if afferents is None else {})

        m.setattr(gc, "_drive", blind)
        assert not _charges(), "empty-afferent plant was not caught"
    assert _charges()


def test_column_latencies_uses_apical():
    """COL path: an E unit with zero basal stays silent under the return; an I unit with zero
    basal still fires from its E partner's spike (the brain's own additive E->I, not the return)."""
    e_silent = GrownUnit(0, (0.0,), 1.0, WINDOW, 1, (0.0,))
    inh = GrownUnit(1, (0.0,), 1.0, WINDOW, -1, (0.0,))
    aff = {0: [(0.0, 5.0)], 1: [(0.0, 5.0)]}
    assert gc.column_latencies((e_silent, inh), [0], TAU, 0.5, afferents=aff) == {0: None, 1: None}
    e_fires = GrownUnit(2, (2.0,), 1.0, WINDOW, 1, (0.0,))
    out = gc.column_latencies((e_fires, inh), [0], TAU, 2.0, afferents={2: [], 1: [(0.0, 5.0)]})
    assert out[2] == 1 and out[1] is not None


def _port(theta):
    units = (GrownUnit(0, (0.0,), theta, WINDOW, 1, (0.0,)),)
    birth = {
        0: {
            "origin": "constructed",
            "unit_id": 0,
            "sign": 1,
            "w_in": [0.0],
            "threshold": theta,
            "window": WINDOW,
        }
    }
    rule = {k: 1.0 for k in ("lr", "a_plus", "a_minus", "tau_plus", "tau_minus", "tau_e")}
    attach = [
        {"source": "A", "target": 0, "sign": 1, "weight": 0.0, "delay": 0, "receipt": "a" * 64}
    ]
    return gc.GrownReturnPort(units, birth, ("A",), {"A": 1}, attach, rule)


@pytest.mark.parametrize("theta", [0.0, -1e-18, -1.0])
def test_nonpositive_target_threshold_refused(theta):
    """A1.B1/A2.C1: a return target with theta <= 0 is refused when the port is built; the helper's
    own raise is a backstop. Reachability: without the refusal, B = 0 gives D = 0 >= theta."""
    assert 0.0 * (1.0 + math.tanh(1.0)) >= theta  # the red the refusal prevents is reachable
    _port(1.0)  # a positive target builds
    with pytest.raises(ValueError, match="positive target threshold"):
        _port(theta)
    with pytest.raises(ValueError, match="positive target threshold"):
        gc._apical_gain(0.5, theta)


def test_chat_context_refuses_nonpositive_threshold_before_any_drive(monkeypatch):
    """A2.C1: the chat52 return context refuses a theta <= 0 target at CONTEXT BUILD, before
    route_text computes any latency; the helper backstop is never the path the refusal takes."""
    from snn_data_lake.readout import _IXN_INTENTS
    from snn_data_lake.readout import posed_problem as pp

    n = len(_IXN_INTENTS)
    unit = GrownUnit(0, (0.0,) * 52, 0.0, WINDOW, 1, (0.0,) * n)
    ixn = SimpleNamespace(
        n_in=52,
        responses=tuple(_IXN_INTENTS),
        tau=TAU,
        _g9_context={
            "problem_id": "ii.xor",
            "units": pp.unit_rows((unit,)),
            "afferents": [[0, [[0.0, 1.0]]]],
            "t_post_g": float(WINDOW + 3),
            "out_eligible": [[True] * n],
        },
    )
    calls = []
    monkeypatch.setattr(
        gc, "_apical_gain", lambda a, th: calls.append((a, th)) or 1.0, raising=False
    )
    with pytest.raises(pp.PosedRefused, match="positive target threshold"):
        pp.chat_grown_drives(ixn, [0] * 52)
    assert calls == []
