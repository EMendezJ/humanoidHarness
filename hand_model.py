"""BrainCo Revo2 (5 fingers, 6 motors) on the G1 wrists, derived from xr_teleoperate's hand URDFs.

Provides the mount pose on wrist_yaw_link, the 6-command -> 11-joint mapping, the tool point (where
the centre of a grasped object sits) and a cloud of points on the hand's surface for clearance checks.

Mount convention, same as the Dex3 it replaces: at the G1 zero pose the fingers point forward, the
palm faces the body midline and the index/thumb side is up. The hand's own axes (finger direction,
closing direction, index side) are measured from its kinematics, so no sign is hand-written.
"""
import os
import numpy as np
import pinocchio as pin

BRAINCO_DIR = os.path.expanduser(os.environ.get(
    "FABLE_PICK_BRAINCO", "~/xr_teleoperate/assets/brainco_hand"))
# Wrist face of the Revo2 along wrist_yaw x: the flange where the Dex3 palm mounts in the G1 model.
# CALIBRATE: add the thickness of the real adapter plate.
MOUNT_X = 0.0415
FINGERS = ("index", "middle", "ring", "pinky")
OBJECT_RADIUS = 0.035     # typical can/cup: the tool point sits this far in front of the palm
TCP_ALONG = 0.06          # ... and this far from the wrist face along the fingers (palm middle)

# Command order of rt/brainco/*/cmd: [thumb, thumb_aux, index, middle, ring, pinky], 0 open .. 1 closed.
# The hand comes down with the thumb up in the palm plane (aux 0): opposed, the open thumb sticks out
# ~9 cm in front of the palm and would land on the object's top. It swings into opposition on closing.
OPEN = np.zeros(6)
PRESHAPE = OPEN
CLOSED = np.array([0.75, 1.0, 0.85, 0.85, 0.85, 0.85])   # CALIBRATE grip on the real hand


def urdf_path(side):
    return os.path.join(BRAINCO_DIR, f"brainco_{side}.urdf")


def joint_targets(side, cmd6, limits):
    """Joint angles for a 6-value command. limits: {joint_name: upper}. Distal joints follow the
    URDF mimic ratios (thumb 1.0, fingers 1.155), clipped to their own limits."""
    c = np.clip(np.asarray(cmd6, float), 0.0, 1.0)
    q = {f"{side}_thumb_metacarpal_joint": c[1] * limits[f"{side}_thumb_metacarpal_joint"],
         f"{side}_thumb_proximal_joint": c[0] * limits[f"{side}_thumb_proximal_joint"]}
    q[f"{side}_thumb_distal_joint"] = min(q[f"{side}_thumb_proximal_joint"], limits[f"{side}_thumb_distal_joint"])
    for k, f in enumerate(FINGERS):
        prox = c[2 + k] * limits[f"{side}_{f}_proximal_joint"]
        q[f"{side}_{f}_proximal_joint"] = prox
        q[f"{side}_{f}_distal_joint"] = min(1.155 * prox, limits[f"{side}_{f}_distal_joint"])
    return q


def _read_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = int(np.frombuffer(data[80:84], np.uint32)[0])
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(rec["v"].reshape(-1, 3).astype(float).round(4), axis=0)


class Revo2:
    def __init__(self, side):
        self.side = side
        self.model = pin.buildModelFromUrdf(urdf_path(side))
        self.data = self.model.createData()
        m = self.model
        self.limits = {m.names[i]: m.upperPositionLimit[m.joints[i].idx_q] for i in range(1, m.njoints)}
        self.base = m.getFrameId(f"{side}_base_link", pin.FrameType.BODY)

        # Hand axes in its base frame, from the kinematics: knuckles give the long axis (fingertips
        # already curl toward the palm when open), finger closing gives the palm normal.
        open_tips = self._tips(np.zeros(6))
        closed_tips = self._tips(np.array([0, 0, 0.8, 0.8, 0.8, 0.8]))
        pin.framesForwardKinematics(m, self.data, pin.neutral(m))
        B = self.data.oMf[self.base]
        fingers = np.mean([B.actInv(self.data.oMi[m.getJointId(f"{side}_{f}_proximal_joint")].translation)
                           for f in FINGERS], 0)
        fingers /= np.linalg.norm(fingers)
        close = np.mean([closed_tips[f] - open_tips[f] for f in FINGERS], 0)
        palm = close - close.dot(fingers) * fingers
        palm /= np.linalg.norm(palm)
        index_side = np.cross(fingers, palm) if side == "right" else -np.cross(fingers, palm)
        assert index_side.dot(open_tips["index"] - open_tips["pinky"]) > 0, "unexpected hand geometry"

        # wrist_yaw axes: fingers -> +x, palm -> body midline (+y right hand, -y left), index -> +z.
        medial = 1.0 if side == "right" else -1.0
        A_hand = np.stack([fingers, palm, index_side], 1)
        A_wrist = np.diag([1.0, medial, 1.0])
        R = A_wrist @ A_hand.T
        U, _, Vt = np.linalg.svd(R)                  # snap to the nearest rotation
        R = U @ Vt
        assert np.linalg.det(R) > 0.99, "mount is not a proper rotation"
        self.mount = pin.SE3(R, np.array([MOUNT_X, 0.0, 0.0]))     # wrist_yaw_link -> {side}_base_link

        # Mesh points per link (link frames, URDF mesh origins are identity).
        mesh_dir = os.path.join(BRAINCO_DIR, "meshes")
        self.link_points = {}
        for fr in m.frames:
            if fr.type != pin.FrameType.BODY or not fr.name.startswith(side):
                continue
            for cand in (f"{fr.name}.STL", f"{fr.name.replace('_link', '_Link')}.STL", f"{fr.name}_Link.STL"):
                path = os.path.join(mesh_dir, cand)
                if os.path.exists(path):
                    pts = _read_stl(path)
                    step = max(1, len(pts) // 400)
                    self.link_points[fr.name] = pts[::step]
                    break

        palm_pts = self.link_points[f"{side}_base_link"]
        z = palm_pts @ fingers
        band = palm_pts[(z > TCP_ALONG - 0.02) & (z < TCP_ALONG + 0.02)]
        palm_surface = (band @ palm).max()
        tcp_hand = fingers * TCP_ALONG + palm * (palm_surface + OBJECT_RADIUS)
        self.tcp = self.mount.act(tcp_hand)          # in wrist_yaw_link

    def q(self, cmd6):
        q = pin.neutral(self.model)
        for name, val in joint_targets(self.side, cmd6, self.limits).items():
            q[self.model.joints[self.model.getJointId(name)].idx_q] = val
        return q

    def _tips(self, cmd6):
        m, d = self.model, self.data
        pin.framesForwardKinematics(m, d, self.q(cmd6))
        B = d.oMf[self.base]
        return {f: B.actInv(d.oMf[m.getFrameId(f"{self.side}_{f}_tip", pin.FrameType.BODY)].translation)
                for f in ("thumb",) + FINGERS}

    def clearance_points(self, n=300):
        """Hand surface while descending (open) and while carrying (closed), wrist_yaw_link frame."""
        return np.concatenate([self.surface_points(OPEN, n // 2), self.surface_points(CLOSED, n // 2)])

    def surface_points(self, cmd6, n=250):
        """Points on the hand surface in the wrist_yaw_link frame, for a command."""
        m, d = self.model, self.data
        pin.framesForwardKinematics(m, d, self.q(cmd6))
        B = d.oMf[self.base]
        pts = []
        for name, p in self.link_points.items():
            T = B.actInv(d.oMf[m.getFrameId(name, pin.FrameType.BODY)])
            pts.append((T.rotation @ p.T).T + T.translation)
        pts = np.concatenate(pts)
        pts = (self.mount.rotation @ pts.T).T + self.mount.translation
        idx = np.random.default_rng(0).choice(len(pts), min(n, len(pts)), replace=False)
        return pts[idx]


_HANDS = {}


def revo2(side):
    if side not in _HANDS:
        _HANDS[side] = Revo2(side)
    return _HANDS[side]
