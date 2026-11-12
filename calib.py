"""Head-camera calibration from an ArUco marker stuck on the back of a hand.

The arm visits several poses; at each one the camera measures the marker's pose (PnP, printed size
known) and forward kinematics gives the wrist's pose. A least-squares fit then solves at once for
  - the correction of the camera pose relative to the URDF d435 mount (6 DoF), and
  - where the marker sits on the hand (6 DoF, so it can be stuck anywhere without measuring).
The correction uses HomeBody's convention (assets/calib/g1_head.json): R = Rodrigues(rot_vec) @ R_urdf
(rotation about the camera centre) and t = t_urdf + translation, both in torso (pelvis0) axes.
"""
import json
import numpy as np
import cv2
import pinocchio as pin
from scipy.optimize import least_squares

ROT_WEIGHT = 0.05      # 1 rad of rotation residual counts like 5 cm of position residual


def marker_pose_pnp(corners, size, K):
    """Marker -> optical frame (pin.SE3) from its 4 image corners, or None."""
    h = size / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return pin.SE3(cv2.Rodrigues(rvec)[0], tvec.ravel())


def corrected(T_cam_nominal, rot_vec, translation):
    return pin.SE3(cv2.Rodrigues(np.asarray(rot_vec, float))[0] @ T_cam_nominal.rotation,
                   T_cam_nominal.translation + np.asarray(translation, float))


def _se3(v):
    return pin.SE3(cv2.Rodrigues(np.asarray(v[:3], float))[0], np.asarray(v[3:6], float))


def _residuals(params, T_cam_nominal, wrists, measured, fit_camera=True):
    cam = corrected(T_cam_nominal, params[:3], params[3:6]) if fit_camera else T_cam_nominal
    X = _se3(params[6:12])
    out = []
    for T_w, T_cm in zip(wrists, measured):
        pred = cam.actInv(T_w * X)                      # marker in the optical frame, predicted
        err = pred.actInv(T_cm)                          # 0 when prediction == measurement
        out.extend(err.translation)
        out.extend(ROT_WEIGHT * pin.log3(err.rotation))
    return np.array(out)


def solve(T_cam_nominal, wrists, measured):
    """wrists: [pin.SE3 wrist_yaw in pelvis0], measured: [pin.SE3 marker in optical frame].
    Returns a dict with the correction, the marker-on-hand pose and fit statistics."""
    # Initial marker-on-hand from the first view with the nominal camera.
    X0 = wrists[0].actInv(T_cam_nominal * measured[0])
    x0 = np.r_[np.zeros(6), cv2.Rodrigues(X0.rotation)[0].ravel(), X0.translation]

    def pos_rms(params, fit_camera):
        r = _residuals(params, T_cam_nominal, wrists, measured, fit_camera).reshape(-1, 6)
        return float(np.sqrt(np.mean(np.sum(r[:, :3] ** 2, 1))))

    # Baseline: URDF camera, only the marker placement fitted.
    base = least_squares(lambda p: _residuals(np.r_[np.zeros(6), p], T_cam_nominal, wrists, measured, False),
                         x0[6:], loss="soft_l1", f_scale=0.01)
    full = least_squares(lambda p: _residuals(p, T_cam_nominal, wrists, measured), x0,
                         loss="soft_l1", f_scale=0.01)
    per_pose = np.linalg.norm(_residuals(full.x, T_cam_nominal, wrists, measured).reshape(-1, 6)[:, :3], axis=1)
    return dict(rot_vec_deg=np.degrees(full.x[:3]).tolist(), translation_m=full.x[3:6].tolist(),
                marker_in_wrist=dict(rot_vec=full.x[6:9].tolist(), translation_m=full.x[9:12].tolist()),
                rms_urdf_m=pos_rms(np.r_[np.zeros(6), base.x], False), rms_calibrated_m=pos_rms(full.x, True),
                per_pose_error_m=per_pose.tolist(), poses=len(wrists))


def save(path, result, extra=None):
    out = dict(schema="humanoidHarness.head_camera_calibration.v1",
               convention="R = Rodrigues(rot_vec_deg) @ R_urdf; t = t_urdf + translation_m, torso (pelvis0) axes",
               **result, **(extra or {}))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load(path):
    with open(path) as f:
        d = json.load(f)
    return np.radians(d["rot_vec_deg"]), np.array(d["translation_m"])
