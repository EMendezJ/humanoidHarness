"""G1 (29 DoF, base fixed as on the gantry) with BrainCo Revo2 hands, as a MuJoCo spec.

Starts from xr_teleoperate's g1_body29_hand14.xml, removes its Unitree Dex3 hands (3 fingers) and
attaches the Revo2 URDFs at the mount pose from hand_model. Used by the simulator and by the
self-collision checker, so both see the same hands the real robot has.
"""
import os
import numpy as np
import mujoco

from hand_model import revo2, urdf_path, BRAINCO_DIR

MJCF = os.path.expanduser(os.environ.get(
    "FABLE_PICK_MJCF", "~/xr_teleoperate/assets/g1/g1_body29_hand14.xml"))
HAND_PREFIX = "rv_"                     # names of the attached hand elements: rv_<urdf name>
HAND_RGBA = [0.18, 0.18, 0.2, 1.0]


def hand_joint(side, urdf_joint):
    return f"{HAND_PREFIX}{urdf_joint}"


def g1_revo2_spec():
    spec = mujoco.MjSpec.from_file(MJCF)
    spec.meshdir = os.path.join(os.path.dirname(MJCF), "meshes")
    spec.delete(spec.joint("floating_base_joint"))
    for side in ("left", "right"):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        for b in list(wrist.bodies):
            if b.name.startswith(f"{side}_hand_"):
                spec.delete(b)
        for g in list(wrist.geoms):
            if g.meshname == f"{side}_hand_palm_link":
                spec.delete(g)

        hand = mujoco.MjSpec.from_file(urdf_path(side))
        for m in hand.meshes:
            m.file = os.path.join(BRAINCO_DIR, m.file)
        for g in hand.geoms:
            g.rgba = HAND_RGBA
        T = revo2(side).mount
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T.rotation.flatten())
        base = hand.body(f"{side}_base_link")
        base.pos, base.quat = [0, 0, 0], [1, 0, 0, 0]   # drop the URDF's fixed base_link -> {side}_base_link offset
        frame = wrist.add_frame(pos=T.translation, quat=quat)
        frame.attach_body(base, HAND_PREFIX, "")
    return spec


def hand_qpos_targets(model, side, cmd6):
    """[(qpos address, angle)] for a 6-value Revo2 command in a compiled g1_revo2 model."""
    h = revo2(side)
    from hand_model import joint_targets
    return [(model.jnt_qposadr[model.joint(hand_joint(side, n)).id], v)
            for n, v in joint_targets(side, cmd6, h.limits).items()]
