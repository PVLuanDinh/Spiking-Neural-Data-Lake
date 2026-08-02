"""
Boundary tests for arm_config: dbin/present/bins_from_capture cut points and the
uncalibrated-IR gate. These are the recognition-calibration edges a rig re-mount hits, so
an off-by-one here silently mis-bins distance into the wrong Paradigm-B channel.

Run: pytest tests/test_arm_config.py
"""

from snn_data_lake import arm_config as ac


def test_dbin_boundaries():
    assert ac.dbin(0.0) == ac.NEAR
    assert ac.dbin(ac.NEAR_MAX - 1e-9) == ac.NEAR
    assert ac.dbin(ac.NEAR_MAX) == ac.MID  # d < NEAR_MAX is strict -> 0.30 is MID
    assert ac.dbin(ac.MID_MAX - 1e-9) == ac.MID
    assert ac.dbin(ac.MID_MAX) == ac.FAR  # 0.60 is FAR
    assert ac.dbin(ac.IN_RANGE_MAX - 1e-9) == ac.FAR
    assert ac.dbin(ac.IN_RANGE_MAX) is None  # >= in-range -> idle (no object)
    assert ac.dbin(99.0) is None


def test_present_distance_and_ir_gate():
    assert ac.present(ac.IN_RANGE_MAX - 1e-9) is True
    assert ac.present(ac.IN_RANGE_MAX) is False  # boundary: not present at exactly in-range
    # IR disabled by default -> the ir value is ignored either way
    assert ac.present(0.5, ir=0) is True
    assert ac.present(0.5, ir=99999) is True


def test_present_ir_confirms_only_when_calibrated(monkeypatch):
    monkeypatch.setattr(ac, "IR_PRESENT", (">", 500))
    assert ac.present(0.5, ir=600) is True  # in range AND IR agrees
    assert ac.present(0.5, ir=400) is False  # in range but IR rejects (distance ghost)
    assert ac.present(2.5, ir=600) is False  # out of range regardless of IR
    monkeypatch.setattr(ac, "IR_PRESENT", ("<", 500))
    assert ac.present(0.5, ir=400) is True  # "<" polarity honoured


def test_bins_from_capture_defaults_when_too_few():
    assert ac.bins_from_capture([0.1, 0.2]) == (ac.NEAR_MAX, ac.MID_MAX)


def test_bins_from_capture_percentiles_and_range_filter():
    caps = [0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 5.0]  # 5.0 is out of range -> must be dropped
    near, mid = ac.bins_from_capture(caps)
    assert 0.0 < near <= mid < ac.IN_RANGE_MAX  # monotone, in range, derived from real data


# ---- SO-101 joint table (specs/004) -------------------------------------------------------
# The limits are read from the vendored model rather than written here. These tests exist to
# make that provenance enforceable: if someone "helpfully" transcribes the numbers, or swaps the
# asset, the derived values stop matching the model and this fails.

import hashlib  # noqa: E402
import pathlib  # noqa: E402

_SO101 = pathlib.Path(ac.__file__).parent / "so101"

# sha256 of the model DEFINITION files (not the 16 MB of meshes -- those do not define behaviour).
# Pinned so an asset swap is loud. If a model bump is intentional, update these in the same commit
# and say why; that is the point of pinning, not an obstacle to it.
#
# REPINNED 2026-08-02, and the reason matters more than the values. The originals were computed
# on a Windows checkout, where git's default eol handling had inflated these two files by 23 and
# 165 bytes of CRLF over the stored blob. So the pin described one platform's checkout rather
# than the asset, and CI failed against it on Linux -- correctly. `.gitattributes` now marks the
# model `-text`, so every platform checks out the blob verbatim and these digests are the same
# everywhere. Verified: working-copy size == blob size on both files after the change.
_EXPECTED_MODEL_SHA = {
    "scene.xml": "3b79a253a742f55ff0b16682173609229560d97744be472a238ea2e0a6a31ef6",
    "so101_new_calib.xml": "d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580",
}


def test_vendored_model_fingerprint():
    """A silent asset swap changes the physics under every torque and limit assertion."""
    for name, want in _EXPECTED_MODEL_SHA.items():
        got = hashlib.sha256((_SO101 / name).read_bytes()).hexdigest()
        assert got == want, (
            f"{name} changed: {got[:16]}... != pinned {want[:16]}... "
            "If the model bump is intentional, update the pin in the same commit."
        )


def test_joint_table_shape():
    assert len(ac.JOINTS) == 6, "SO-101 is 6-DOF"
    assert [j.qpos_index for j in ac.JOINTS] == list(range(6)), "table must be in qpos order"
    assert len({j.name for j in ac.JOINTS}) == 6, "MJCF joint names must be unique"
    assert len(ac.HOME_QPOS) == 6


def test_joint_names_resolve_in_the_model():
    """The MJCF name is the canonical identity -- it must actually exist in the asset."""
    import mujoco

    m = ac._model()
    for j in ac.JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)
        assert jid == j.qpos_index, f"{j.name}: table says qpos {j.qpos_index}, model says {jid}"


def test_limits_are_read_from_the_model_not_transcribed():
    """Every limit must equal the model's own value exactly."""
    m = ac._model()
    lim = ac.JOINT_LIMITS
    for j in ac.JOINTS:
        r = lim[j.name]
        assert r["ctrl_lo"] == float(m.actuator_ctrlrange[j.qpos_index][0])
        assert r["ctrl_hi"] == float(m.actuator_ctrlrange[j.qpos_index][1])
        assert r["stop_lo"] == float(m.jnt_range[j.qpos_index][0])
        assert r["stop_hi"] == float(m.jnt_range[j.qpos_index][1])


def test_ctrl_and_stop_ranges_are_distinct():
    """They are MEASURED unequal on this model, and conflating them is a defect.

    Clamping to the physical stop would let a command through that the actuator then silently
    pins at the rail -- the silent acceptance the spec forbids. Pinned so the distinction is not
    "simplified away" by someone who assumes they are the same table.
    """
    lim = ac.JOINT_LIMITS
    differing = [
        j.name
        for j in ac.JOINTS
        if (lim[j.name]["ctrl_hi"], lim[j.name]["ctrl_lo"])
        != (lim[j.name]["stop_hi"], lim[j.name]["stop_lo"])
    ]
    assert differing, (
        "ctrl and stop ranges are now identical on every joint -- if the model really changed, "
        "the two-table distinction still holds conceptually, but re-check the gate's clamp source."
    )


def test_home_pose_is_commandable():
    lim = ac.JOINT_LIMITS
    for j in ac.JOINTS:
        r = lim[j.name]
        assert r["ctrl_lo"] <= ac.HOME_QPOS[j.qpos_index] <= r["ctrl_hi"], (
            f"{j.name}: HOME is outside the commandable range"
        )


def test_arm_hardware_rows_are_derived_from_this_table():
    """arm_hardware.JOINTS must be a view of arm_config.JOINTS, not a second literal."""
    from snn_data_lake.brainstem import arm_hardware

    assert arm_hardware.JOINTS == tuple((j.alias, j.servo, j.lever_mm) for j in ac.JOINTS), (
        "arm_hardware.JOINTS drifted from arm_config.JOINTS"
    )
