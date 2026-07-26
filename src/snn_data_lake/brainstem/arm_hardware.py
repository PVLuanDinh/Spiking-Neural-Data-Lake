"""Hardware TARGET for the physical arm: spec, static torque budget, and the sim-to-real gap.

WHAT THIS IS, AND IS NOT
    This is the build spec for an arm that DOES NOT EXIST YET. Nothing here reads a sensor and
    nothing here is telemetry. It is the plan, expressed as numbers a program can check, so the
    dashboard can show the gap between the body we simulate and the body we intend to build without
    either one pretending to be the other.

    Source: robot_arm_build_notes.md (planning conversation, 2026-07). Every figure below is from
    those notes; where a number is DERIVED it says so and shows the arithmetic.

THE FINDING THAT MOTIVATED WRITING THIS AS CODE
    The notes state the rule -- "size servos to rated torque, not stall torque; rated is roughly a
    quarter of stall" -- and separately state the target: 500 g payload at ~300 mm reach on an
    ST3215 1:345 at 12 V. Applying the rule to the target:

        payload-only shoulder torque = 0.5 kg * 30 cm            = 15.0 kg.cm
        ST3215 1:345 @12V, stall 30 kg.cm -> rated = 30/4         =  7.5 kg.cm
        margin                                                    =  0.50x   UNDERPOWERED

    The shoulder is short by 2x on the payload ALONE, before any arm mass. This is not a criticism
    of the notes -- it is the notes' own rule applied to the notes' own numbers, and it is exactly
    the check that is easy to skip when a parts list looks complete. It is asserted in main() so it
    cannot be quietly forgotten between now and a purchase.

    Ways out, all in the notes' own vocabulary: shorten the reach, drop the payload target, gear the
    shoulder down further, use two servos in parallel on the shoulder axis, or counterbalance the
    upper arm. Choosing among them is a design decision, not something this file should presume.
"""

from __future__ import annotations

# ---- servos ---------------------------------------------------------------------------------
# Feetech ST3215 family. Stall figures from the notes; RATED is derived as stall/4 per the notes'
# own sizing rule, because a continuous-duty number is what a joint must actually be sized against.
_RATED_FRACTION_OF_STALL = 0.25

SERVOS = {
    "ST3215-1:345": {"stall_kgcm_12v": 30.0, "stall_kgcm_7v4": 19.5, "ratio": 345, "role": "base, shoulder"},
    "ST3215-1:191": {"stall_kgcm_12v": 16.6, "stall_kgcm_7v4": 10.8, "ratio": 191, "role": "elbow, wrist pitch"},
    "ST3215-1:147": {"stall_kgcm_12v": 12.8, "stall_kgcm_7v4": 8.3, "ratio": 147, "role": "wrist roll"},
    "SC-series": {"stall_kgcm_12v": 8.0, "stall_kgcm_7v4": 5.0, "ratio": 0, "role": "gripper"},
}

# The 1:191 and 1:147 stall figures are INTERPOLATED from the 1:345 datum by gear ratio -- the notes
# give ratios and relative behaviour ("faster, less torque") but not absolute torque for these two.
# Marked so nobody mistakes them for datasheet values.
_INTERPOLATED = ("ST3215-1:191", "ST3215-1:147", "SC-series")

GRAVITY = 9.81  # m/s^2 (only used for documentation; kg.cm math below is in kg-force units)


def rated_kgcm(servo, volts=12.0):
    """Continuous-duty torque. The notes are explicit that stall is NOT the number to size against."""
    s = SERVOS[servo]
    stall = s["stall_kgcm_12v"] if volts >= 10.0 else s["stall_kgcm_7v4"]
    return stall * _RATED_FRACTION_OF_STALL


# ---- the intended arm -----------------------------------------------------------------------
# 6 DOF, ~250-350 mm reach, 500 g payload target.
TARGET = {
    "dof": 6,
    "reach_mm": 300.0,  # mid of the 250-350 range in the notes
    "payload_kg": 0.5,
    "bus_volts": 12.0,
    "supply_amps": 10.0,  # notes: 8-10A+, several servos stalling at 2.7A each
    "stall_amps_per_servo": 2.7,
}

# joint -> (servo, distance from that joint to the payload in mm). The distance is what converts a
# payload mass into a torque at that joint; wrist roll and the gripper carry no bending moment from
# the payload's weight, so their lever arm is 0.
JOINTS = (
    ("base_yaw", "ST3215-1:345", 0.0),  # vertical axis: carries no gravity moment, only inertia
    ("shoulder_pitch", "ST3215-1:345", 300.0),
    ("elbow_pitch", "ST3215-1:191", 180.0),
    ("wrist_pitch", "ST3215-1:191", 60.0),
    ("wrist_roll", "ST3215-1:147", 0.0),
    ("gripper", "SC-series", 0.0),
)


def holding_torque_kgcm(lever_mm, payload_kg=None, arm_mass_kg=0.0, arm_cog_mm=None):
    """Static gravity torque at a joint with the link held HORIZONTAL -- the worst case.

    kg.cm is kilogram-force-centimetre, the unit servo datasheets use, so mass_kg * distance_cm is
    the torque directly with no g factor. Horizontal is the worst case because the moment arm is the
    full length; any other pose is smaller by cos(elevation).

    This is STATIC only: no acceleration, no dynamics, no friction. A real arm needs headroom above
    this, not equality with it.
    """
    payload_kg = TARGET["payload_kg"] if payload_kg is None else payload_kg
    arm_cog_mm = (lever_mm / 2.0) if arm_cog_mm is None else arm_cog_mm
    return payload_kg * (lever_mm / 10.0) + arm_mass_kg * (arm_cog_mm / 10.0)


def budget(arm_mass_kg=0.0, volts=None):
    """Per-joint required vs rated torque. Returns rows; `ok` False means underpowered.

    arm_mass_kg is the mass of everything OUTBOARD of the joint. Default 0.0 gives the
    payload-only floor -- the most generous case, and the shoulder already fails it.
    """
    volts = TARGET["bus_volts"] if volts is None else volts
    rows = []
    for name, servo, lever in JOINTS:
        need = holding_torque_kgcm(lever, arm_mass_kg=arm_mass_kg) if lever > 0 else 0.0
        have = rated_kgcm(servo, volts)
        rows.append(
            {
                "joint": name,
                "servo": servo,
                "lever_mm": lever,
                "required_kgcm": round(need, 2),
                "rated_kgcm": round(have, 2),
                "margin_x": round(have / need, 2) if need > 0 else None,
                "ok": (need <= have) if need > 0 else True,
                "estimated": servo in _INTERPOLATED,
            }
        )
    return rows


def peak_current_amps(n_simultaneous=3):
    """Supply sizing. The notes size for several servos stalling at once, not one."""
    return n_simultaneous * TARGET["stall_amps_per_servo"]


# ---- the sim-to-real gap --------------------------------------------------------------------
# What we currently simulate is NOT this arm, and the dashboard must not imply otherwise.
# ArmSim is a 2-link planar arm in DIMENSIONLESS units (l1 = l2 = 1.0, reach 2.0), with no mass, no
# gravity, no torque and no dt -- JOINT_x_ROTATE teleports the joint. It is a kinematic toy.
SIM_GAP = {
    "sim_dof": 2,
    "target_dof": 6,
    "missing_dof": ("wrist_pitch", "wrist_roll", "gripper_articulation", "base_yaw"),
    "sim_units": "dimensionless (l1=l2=1.0, reach 2.0)",
    "target_units": "mm (reach ~300)",
    "mm_per_sim_unit": TARGET["reach_mm"] / 2.0,  # 150 mm per unit IF the toy is ever scaled
    "sim_has_dynamics": False,
    "sim_has_gravity": False,
    "bridge": (
        "MuJoCo + SO-ARM100 (TheRobotStudio) ships MJCF with motor properties matched to the real "
        "STS3215 -- a genuine digital twin. That is the path from this kinematic toy to a body whose "
        "torque limits are real, and it is the same engine mhflocke drives its demo with."
    ),
}


def state():
    """The hardware section for telemetry. Explicitly a TARGET; `built` is False."""
    rows = budget()
    return {
        "built": False,  # nothing here is measured; this is a plan
        "target_dof": TARGET["dof"],
        "reach_mm": TARGET["reach_mm"],
        "payload_kg": TARGET["payload_kg"],
        "bus_volts": TARGET["bus_volts"],
        "peak_amps_3_stall": round(peak_current_amps(3), 1),
        "supply_amps": TARGET["supply_amps"],
        "supply_ok": peak_current_amps(3) <= TARGET["supply_amps"],
        "joints": rows,
        "underpowered": [r["joint"] for r in rows if not r["ok"]],
        "sim_dof": SIM_GAP["sim_dof"],
        "sim_units": SIM_GAP["sim_units"],
        "sim_has_dynamics": SIM_GAP["sim_has_dynamics"],
    }


def main():
    """Self-check: the torque budget is arithmetic, and the shoulder shortfall must stay visible."""
    st = state()
    assert st["built"] is False, "this module must never claim the arm exists"

    rows = {r["joint"]: r for r in st["joints"]}

    # the headline finding, asserted so a future edit cannot quietly erase it
    sh = rows["shoulder_pitch"]
    assert sh["required_kgcm"] == 15.0, f"payload-only shoulder torque changed: {sh['required_kgcm']}"
    assert sh["rated_kgcm"] == 7.5, f"ST3215 1:345 rated changed: {sh['rated_kgcm']}"
    assert not sh["ok"], "shoulder must read UNDERPOWERED for 500g at 300mm"
    assert sh["margin_x"] == 0.5, f"shoulder margin changed: {sh['margin_x']}"

    # a vertical axis carries no gravity moment
    assert rows["base_yaw"]["required_kgcm"] == 0.0, "base yaw should carry no gravity moment"

    # rated must be a quarter of stall, per the notes' own rule
    assert abs(rated_kgcm("ST3215-1:345", 12.0) - 7.5) < 1e-9, "rated/stall rule broken"
    assert rated_kgcm("ST3215-1:345", 7.4) < rated_kgcm("ST3215-1:345", 12.0), "12V must beat 7.4V"

    # adding arm mass can only make it worse, never better
    worse = {r["joint"]: r for r in budget(arm_mass_kg=0.4)}["shoulder_pitch"]
    assert worse["required_kgcm"] > sh["required_kgcm"], "arm mass must increase required torque"

    # supply sizing: 3 servos stalling at 2.7A must fit the specified supply
    assert st["supply_ok"], "12V supply undersized for 3 simultaneous stalls"

    # the sim/target gap must stay stated, not smoothed over
    assert SIM_GAP["sim_dof"] < SIM_GAP["target_dof"], "sim/target DOF gap vanished"
    assert SIM_GAP["sim_has_dynamics"] is False, "ArmSim has no dynamics; do not claim otherwise"

    under = ", ".join(st["underpowered"]) or "none"
    print(
        f"arm_hardware OK: target {st['target_dof']}-DOF, {st['reach_mm']:.0f}mm, "
        f"{st['payload_kg']}kg payload | shoulder needs {sh['required_kgcm']} kg.cm vs "
        f"{sh['rated_kgcm']} rated ({sh['margin_x']}x) | UNDERPOWERED: {under} | "
        f"sim is {st['sim_dof']}-DOF {st['sim_units']}, dynamics={st['sim_has_dynamics']}"
    )


if __name__ == "__main__":
    main()
