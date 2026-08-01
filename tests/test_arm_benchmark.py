"""The P2 benchmark must report both arms honestly, including what each CANNOT do.

The failure mode this guards is not a wrong number -- it is a benchmark that quietly drops the
arm that looks bad, or reports a torque figure for a body that has no concept of torque. Either
would manufacture the licence that P2 requires the benchmark to grant or withhold.

  pytest tests/test_arm_benchmark.py
"""

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("mujoco")

from tools import arm_benchmark as ab  # noqa: E402


def test_benchmark_reports_both_arms():
    """Neither arm may be silently omitted."""
    incumbent = ab.probe_armsim()
    assert incumbent["dof"] == 2, "the incumbent is a 2-DOF planar arm"
    assert incumbent["units"].startswith("dimensionless")
    m = ab._model()
    assert int(m.nu) == 6, "the candidate is 6-DOF"


def test_incumbent_torque_is_reported_as_unavailable_not_zero():
    """`None` and `0.0` are different claims.

    A zero would assert the arm needs no torque to hold a pose, which is a statement about
    physics. `None` asserts the quantity does not exist in that body. Conflating them is how a
    kinematic toy comes to look like it passed a torque check.
    """
    assert ab.probe_armsim()["torque_nm"] is None


def test_candidate_reports_per_joint_torque_in_newton_metres():
    m = ab._model()
    import mujoco

    d = mujoco.MjData(m)
    tau = ab.hold_torque(m, d, 0.0)
    assert len(tau) == m.nu, "one torque per actuated joint"
    # gravity loads the pitch joints and not the vertical axis -- a sanity check that this is
    # a physical reading and not an array of zeros
    assert abs(tau[1]) > 0.1, "shoulder_lift must carry a gravity moment at home"
    assert abs(tau[0]) < 1e-6, "shoulder_pan is a vertical axis: no gravity moment"


def test_payload_increases_torque_and_leaves_the_model_untouched():
    """The payload is applied analytically; the vendored asset must not be mutated."""
    m = ab._model()
    import mujoco

    d = mujoco.MjData(m)
    bare = ab.hold_torque(m, d, 0.0)
    loaded = ab.hold_torque(m, d, 0.5)
    assert abs(loaded[1]) > abs(bare[1]), "500 g at the TCP must raise shoulder torque"
    # same model object, same body count and mass: nothing was added to the scene
    assert m.nbody == ab._model().nbody
    assert abs(float(m.body_mass.sum()) - float(ab._model().body_mass.sum())) < 1e-12


def test_worst_case_exceeds_home_case():
    """Home is a folded pose; reporting it alone understates the build's actual question."""
    m = ab._model()
    import mujoco

    d = mujoco.MjData(m)
    home = abs(ab.hold_torque(m, d, 0.5)[1])
    worst, _pose = ab.worst_case_torque(m, 0.5, samples=2000)
    assert worst[1] >= home, "the sampled worst case cannot be below the home pose"
