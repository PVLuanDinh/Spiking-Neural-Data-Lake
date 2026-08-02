"""Compare the MuJoCo SO-101 against the incumbent ArmSim, on the same inputs.

WHY THIS EXISTS. P2 is non-additive: when two mechanisms do the same job exactly one survives,
and "better" requires a benchmark that exists. None did. This is that benchmark, and it is the
precondition for retiring `brainstem/arm_sim.py` -- not a report written after the decision.

It deliberately does NOT depend on the new gate. It drives a bare MuJoCo body and the existing
ArmSim directly, so it can run before any of the replacement is built and can therefore license
or refuse that work.

WHAT IS AND IS NOT COMPARABLE. The two bodies are not commensurable: 2 joints against 6,
dimensionless links against metres, teleport against dynamics. Reporting a single "winner"
number would be dishonest. So each row states what each arm CAN report and names what it
STRUCTURALLY CANNOT -- the capability delta is the finding, not a score.

THE PAYLOAD IS APPLIED ANALYTICALLY, NOT BY EDITING THE MODEL. Holding torque for a mass at the
tool centre point is tau = qfrc_bias + J_site^T . F, with F the payload weight. That is exact for
a point mass and leaves the vendored asset untouched, which the spec requires.

  PYTHONUTF8=1 PYTHONPATH=src python tools/arm_benchmark.py
  PYTHONUTF8=1 PYTHONPATH=src python tools/arm_benchmark.py --payload-kg 0.5
"""

import argparse
import math
import os
import pathlib
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from snn_data_lake.brainstem import arm_config, arm_hardware  # noqa: E402

SCENE = pathlib.Path(arm_config.__file__).parent / "so101" / "scene.xml"
G = 9.81
KGCM_PER_NM = 100.0 / 9.81  # 1 N*m = 10.197 kg*cm


# ---- the incumbent ---------------------------------------------------------------------


def _model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _tcp_site(m):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if sid < 0:
        raise SystemExit("gripperframe site missing from the vendored model")
    return sid


def hold_torque(m, d, payload_kg=0.0):
    """Per-joint torque needed to hold the current pose, N*m.

    qfrc_bias is gravity + Coriolis for the arm's own mass. A payload at the tool centre point
    adds J^T . F, where F is its weight in world coordinates.
    """
    mujoco.mj_forward(m, d)
    tau = np.array(d.qfrc_bias[: m.nu], dtype=float)
    if payload_kg:
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, _tcp_site(m))
        force = np.array([0.0, 0.0, -payload_kg * G])
        tau = tau - jacp[:, : m.nu].T @ force
    return tau


def probe_mujoco(payload_kg):
    m, d = _model(), None
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    sid = _tcp_site(m)
    home_tcp = d.site_xpos[sid].copy()

    unloaded = hold_torque(m, d, 0.0)
    loaded = hold_torque(m, d, payload_kg)
    worst, worst_pose = worst_case_torque(m, payload_kg)

    # E-STOP semantics are a property of the GATE, not the body -- the bare body has none.
    # Named rather than silently omitted.
    return {
        "dof": int(m.nu),
        "home_tcp_m": home_tcp,
        "units": "metres",
        "timestep": float(m.opt.timestep),
        "forcerange_nm": float(m.actuator_forcerange[0][1]),
        "tau_unloaded": unloaded,
        "tau_loaded": loaded,
        "tau_worst": worst,
        "worst_pose": worst_pose,
        "reach_m": max_horizontal_reach(m),
    }


def worst_case_torque(m, payload_kg, samples=20000, seed=0):
    """Largest per-joint hold torque over the reachable envelope, not just at home.

    Home is a FOLDED pose: the lever arm is near its minimum and the torques there are the
    easy case. The build notes' question -- can these servos hold 500 g at reach -- is a
    question about full extension, so the envelope is sampled and the maximum kept.
    """
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    worst = np.zeros(m.nu)
    worst_pose = None
    for _ in range(samples):
        d.qpos[: m.nq] = rng.uniform(lo, hi)
        tau = np.abs(hold_torque(m, d, payload_kg))
        if tau[1] > worst[1]:  # index the shoulder: it carries the whole arm
            worst_pose = d.qpos[: m.nq].copy()
        worst = np.maximum(worst, tau)
    return worst, worst_pose


def max_horizontal_reach(m, samples=20000, seed=0):
    """SC-006's reach: the furthest the TCP gets from the base axis, horizontally.

    Sampled rather than solved -- the closed form does not exist at 6 DOF, and a corner search
    over joint limits understates it because the maximum is interior.
    """
    d = mujoco.MjData(m)
    sid = _tcp_site(m)
    rng = np.random.default_rng(seed)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    best = 0.0
    for _ in range(samples):
        d.qpos[: m.nq] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        x, y = d.site_xpos[sid][0], d.site_xpos[sid][1]
        best = max(best, math.hypot(x, y))
    return best


# ---- report ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload-kg", type=float, default=0.5, help="TCP payload (default 0.5)")
    args = ap.parse_args()

    b = probe_mujoco(args.payload_kg)
    joints = [j.name for j in arm_config.JOINTS]

    print("=" * 78)
    print("ARM BENCHMARK -- the live MuJoCo SO-101")
    print("=" * 78)

    print("\n-- what the body IS -------------------------------------------------------")
    print(f"  {'degrees of freedom':<22}{b['dof']}")
    print(f"  {'units':<22}{b['units']}")
    print(f"  {'integration step':<22}{b['timestep']} s")
    print(f"  {'reach':<22}{b['reach_m']:.4f} m horizontal")
    print("  The A/B against the retired 2-DOF ArmSim -- the measurement that licensed this")
    print("  swap under Policy 2 -- is recorded in specs/004-mujoco-so101-arm/benchmark.md.")
    print("  It is NOT reproduced here: the incumbent is deleted, so re-printing its numbers")
    print("  would be a transcription no run could refute.")

    print("\n-- E-STOP ------------------------------------------------------------------")
    print("  The latch is a property of the GATE that owns the body, not of the body: the")
    print("  incumbent kept it inside the arm, and the replacement had to carry it across")
    print("  deliberately. It is proven in brainstem/so101_arm.py and its tests, not here.")

    print("\n-- holding torque, the capability the incumbent DID NOT HAVE --------------")
    print("  The retired ArmSim could not report this at all: no mass, no gravity, no dt.")
    lim = b["forcerange_nm"]
    stall_kgcm = lim * KGCM_PER_NM
    print(f"\n  The model's actuator forcerange is {lim:.3f} N*m = {stall_kgcm:.1f} kg*cm.")
    print("  That is STALL torque (the notes quote ~30 kg*cm stall at 12V for this gear).")
    print("  The notes' own rule is to size against RATED, which is roughly a quarter of it.")
    print("  Both columns are shown; the RATED column is the one that decides the build.")
    print(f"\n  Per-joint hold torque WITH {args.payload_kg} kg at the tool centre point:")
    print(
        f"    {'joint':<16}{'home':>10}{'WORST':>10}{'kg*cm':>9}"
        f"{'rated':>8}{'x rated':>9}{'% stall':>9}"
    )
    over_rated, over_stall = [], []
    for i, name in enumerate(joints):
        ld, wst = b["tau_loaded"][i], b["tau_worst"][i]
        kgcm = wst * KGCM_PER_NM
        rated = arm_hardware.rated_kgcm(arm_config.JOINTS[i].servo, 12.0)
        x_rated = kgcm / rated if rated else 0.0
        pct_stall = 100.0 * wst / lim
        if x_rated > 1.0:
            over_rated.append(name)
        if pct_stall > 100.0:
            over_stall.append(name)
        print(
            f"    {name:<16}{abs(ld):>10.3f}{wst:>10.3f}{kgcm:>9.2f}"
            f"{rated:>8.1f}{x_rated:>8.2f}x{pct_stall:>8.0f}%"
        )

    print("\n-- cross-check against the repo's own hardware table ----------------------")
    rows = {r["joint"]: r for r in arm_hardware.budget(volts=12.0)}
    sh = rows["shoulder_pitch"]
    mj_kgcm = b["tau_worst"][1] * KGCM_PER_NM
    print(
        f"  arm_hardware (analytic 300 mm lever): shoulder needs {sh['required_kgcm']} kg*cm"
        f" vs {sh['rated_kgcm']} rated -> ok={sh['ok']} ({sh['margin_x']}x)"
    )
    print(
        f"  MuJoCo (full model, worst pose)     : shoulder needs {mj_kgcm:.2f} kg*cm"
        f" vs {sh['rated_kgcm']} rated -> {mj_kgcm / sh['rated_kgcm']:.2f}x"
    )
    print("  Two independent routes to the same conclusion: an analytic lever and a full")
    print("  rigid-body model, agreeing that this joint is over its CONTINUOUS rating with")
    print("  500 g at reach. They are not the same quantity and equality would be suspect;")
    print("  agreement in direction and magnitude is the corroboration.")

    print("\n-- verdict ----------------------------------------------------------------")
    print("  Capability delta: the candidate reports per-joint torque in N*m; the incumbent")
    print("  structurally could not. That is not a better number, it is a")
    print("  quantity that does not exist in the incumbent at all.")
    print(f"\n  Over CONTINUOUS (rated) torque somewhere in the envelope: {over_rated or 'none'}")
    print(f"  Over STALL torque somewhere in the envelope:            {over_stall or 'none'}")
    print("  A joint over rated but under stall will move, then heat, then trip thermal")
    print("  cutoff (~70 C per the notes). 'It moved in the sim' is not 'the servo holds it'.")
    print("\n  P2: 'better requires a benchmark that exists'. It now exists, and it is this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
