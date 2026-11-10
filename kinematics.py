"""G1 arm kinematics for pick-and-place: per-arm IK, head-camera pose, gravity feed-forward.

Every pose here is in the "pelvis0" frame: the pelvis with the waist locked at zero. The head
camera and both arms hang off torso_link, so this frame stays consistent with the camera even if
the real waist sits slightly off zero.
"""
import os
import numpy as np
import pinocchio as pin

from hand_model import revo2

URDF = os.path.expanduser(os.environ.get(
    "FABLE_PICK_URDF", "~/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"))

ARM_JOINTS = {
    side: [f"{side}_{j}_joint" for j in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw")]
    for side in ("left", "right")
}
# Order used by G1_29_ArmController.ctrl_dual_arm: left 7 then right 7.
DUAL_ARM_JOINTS = ARM_JOINTS["left"] + ARM_JOINTS["right"]

# Tool center point = where the object's center should sit at grasp time, in the wrist_yaw frame.
# At the G1 zero pose the forearm points forward (+x) and the palms face the body midline, so
# +x runs along the fingers and the palm normal is +y for the right hand, -y for the left.
# Derived from the BrainCo Revo2 model and its mount (hand_model); CALIBRATE on the real hand.
TCP_OFFSET = {side: revo2(side).tcp for side in ("left", "right")}

# Null-space posture for IK: shoulders abducted so the elbow stays out, away from the torso.
# Abduction is +shoulder_roll on the left arm and -shoulder_roll on the right.
POSTURE = {"left": np.array([0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0]),
           "right": np.array([0.0, -0.35, 0.0, 0.0, 0.0, 0.0, 0.0])}

# Optical frame (x right, y down, z forward) expressed in d435_link (x forward, y left, z up).
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


class G1Kinematics:
    def __init__(self, urdf=URDF):
        full = pin.buildModelFromUrdf(urdf)
        self.full = full
        q0 = pin.neutral(full)

        # Camera pose at waist = 0.
        data = full.createData()
        pin.framesForwardKinematics(full, data, q0)
        T_link = data.oMf[full.getFrameId("d435_link")]
        self.T_cam = pin.SE3(T_link.rotation @ R_LINK_OPTICAL, T_link.translation.copy())

        self.arm = {}
        for side in ("left", "right"):
            keep = set(ARM_JOINTS[side])
            lock = [full.getJointId(n) for n in full.names[1:] if n not in keep]
            model = pin.buildReducedModel(full, lock, q0)
            jid = model.getJointId(f"{side}_wrist_yaw_joint")
            fid = model.addFrame(pin.Frame(f"{side}_tcp", jid, pin.SE3(np.eye(3), TCP_OFFSET[side]),
                                           pin.FrameType.OP_FRAME))
            assert [model.names[i] for i in range(1, model.njoints)] == ARM_JOINTS[side]
            self.arm[side] = dict(model=model, data=model.createData(), fid=fid,
                                  lo=model.lowerPositionLimit + 0.05,
                                  hi=model.upperPositionLimit - 0.05)

        # Both arms, for gravity compensation of the dual-arm command.
        keep = set(DUAL_ARM_JOINTS)
        lock = [full.getJointId(n) for n in full.names[1:] if n not in keep]
        self.dual = pin.buildReducedModel(full, lock, q0)
        self.dual_data = self.dual.createData()
        names = [self.dual.names[i] for i in range(1, self.dual.njoints)]
        self.dual_perm = [names.index(n) for n in DUAL_ARM_JOINTS]   # controller order -> model order

    # ---------------------------------------------------------------- forward kinematics
    def fk(self, side, q7):
        a = self.arm[side]
        pin.framesForwardKinematics(a["model"], a["data"], np.asarray(q7, float))
        return a["data"].oMf[a["fid"]].copy()

    def gravity_tau(self, q14):
        q = np.zeros(self.dual.nq)
        q[self.dual_perm] = q14
        tau = pin.computeGeneralizedGravity(self.dual, self.dual_data, q)
        return tau[self.dual_perm]

    # ---------------------------------------------------------------- inverse kinematics
    def ik(self, side, target, seeds, w_rot=0.3, iters=150, pos_tol=0.004, rot_tol=0.05, k_post=0.05):
        """Damped least squares from several seeds. Returns (q7, pos_err_m, rot_err_rad).

        Orientation is weighted below position: a few degrees off is fine for a power grasp,
        a few centimetres is not. The arm's spare degree of freedom (elbow swivel) is pulled toward
        POSTURE in the null space. The caller decides what error is acceptable.
        """
        a = self.arm[side]
        model, data, fid = a["model"], a["data"], a["fid"]
        W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
        best = None
        for seed in seeds:
            q = np.clip(np.asarray(seed, float), a["lo"], a["hi"])
            for _ in range(iters):
                pin.framesForwardKinematics(model, data, q)
                err = pin.log6(data.oMf[fid].actInv(target)).vector
                if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                    break
                J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)
                JW = J.T @ W
                J_pinv = np.linalg.solve(JW @ J + 1e-3 * np.eye(7), JW)
                dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (k_post * (POSTURE[side] - q))
                q = np.clip(q + np.clip(dq, -0.2, 0.2), a["lo"], a["hi"])
            T = self.fk(side, q)
            pos_err = np.linalg.norm(T.translation - target.translation)
            rot_err = np.linalg.norm(pin.log3(T.rotation.T @ target.rotation))
            # Prefer accurate solutions, then the one closest to the first seed (current pose).
            cost = pos_err + 0.05 * rot_err + 0.002 * np.linalg.norm(q - seeds[0])
            if best is None or cost < best[0]:
                best = (cost, q, pos_err, rot_err)
        return best[1], best[2], best[3]

    # ---------------------------------------------------------------- camera
    def cam_to_pelvis(self, p_cam):
        return self.T_cam.act(np.asarray(p_cam, float))

    def pelvis_to_cam(self, p):
        return self.T_cam.actInv(np.asarray(p, float))


def deproject(u, v, depth_m, K):
    return np.array([(u - K[0, 2]) / K[0, 0] * depth_m, (v - K[1, 2]) / K[1, 1] * depth_m, depth_m])


def project(p_cam, K):
    return np.array([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2], K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
