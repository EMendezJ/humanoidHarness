"""ArUco/AprilTag markers on the table: detection (dictionary found automatically) and 3D pose in the
pelvis frame, both from depth and, when the printed size is known, from the image alone (PnP).
Comparing the two checks the camera intrinsics and the colour-depth alignment."""
import numpy as np
import cv2

from kinematics import deproject

DICTS = ("DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_7X7_250", "DICT_ARUCO_ORIGINAL",
         "DICT_APRILTAG_36h11")


def detect(rgb):
    """Markers of the dictionary that finds the most: [(dict_name, id, corners 4x2 in pixels)]."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    best = []
    for name in DICTS:
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)),
                                      cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
        found = [] if ids is None else [(name, int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]
        if len(found) > len(best):
            best = found
    return best


def _depth_at(depth, u, v, r=3):
    d = depth[max(0, int(v) - r):int(v) + r + 1, max(0, int(u) - r):int(u) + r + 1]
    d = d[(d > 0.1) & (d < 2.5)]
    return float(np.median(d)) if len(d) else None


def marker_pose(kin, corners, depth, K, size=None):
    """Center, corners, side length, normal and tilt of a flat marker, in the pelvis frame.
    Corner depths are sampled slightly inside the marker (edges give flying pixels)."""
    c = corners.mean(0)
    inner = c + 0.8 * (corners - c)
    pts = []
    for u, v in inner:
        d = _depth_at(depth, u, v)
        if d is None:
            return None
        pts.append(kin.cam_to_pelvis(deproject(u, v, d, K)))
    pts = np.array(pts)
    center = pts.mean(0)
    corners3 = center + (pts - center) / 0.8            # back out to the real corners
    side = np.mean([np.linalg.norm(corners3[i] - corners3[(i + 1) % 4]) for i in range(4)])
    normal = np.cross(corners3[2] - corners3[0], corners3[3] - corners3[1])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    tilt = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))
    out = dict(center=center, corners=corners3, side=side, normal=normal, tilt_deg=tilt,
               pitch_deg=np.degrees(np.arctan2(normal[0], normal[2])),   # + = surface rises away from the robot
               roll_deg=np.degrees(np.arctan2(normal[1], normal[2])))    # + = surface rises to the robot's right
    if size:
        half = size / 2.0
        obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            out["center_pnp"] = kin.cam_to_pelvis(tvec.ravel())
    return out


def draw(rgb, markers, poses):
    img = rgb.copy()
    for (name, mid, corners), pose in zip(markers, poses):
        cv2.polylines(img, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        u, v = corners.mean(0).astype(int)
        label = f"id {mid}" + (f" z={pose['center'][2]:+.3f}" if pose else " sin depth")
        cv2.putText(img, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img
