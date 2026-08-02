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


def test_benchmark_no_longer_probes_the_retired_arm():
    """The A/B that licensed the swap is RECORDED, not re-run: the incumbent is deleted.

    Re-printing its numbers as literals would turn a measurement into a transcription no run
    could refute, which is the shape of fake measurement this project treats as a defect. The
    comparison lives in specs/004-mujoco-so101-arm/benchmark.md; what remains here measures
    the live arm only.
    """
    import ast
    import pathlib

    src = pathlib.Path(ab.__file__).read_text(encoding="utf-8")
    names = {n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
    assert "probe_armsim" not in names, "the benchmark still probes a deleted body"
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("arm_sim" in n for n in imported), f"still imports the retired arm: {imported}"

    assert "probe_mujoco" in names, "the live half of the benchmark went with it"


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
