"""Deterministic skills behind Fable's tools: pixel -> 3D -> grasp plan -> joint trajectory.

Fable chooses WHAT to grab and WHERE (a pixel in the head-camera image). Everything physical is
decided here, and every plan is fully solved (all IK waypoints) before the arm moves.
"""
import time
import numpy as np
import cv2
import pinocchio as pin

from kinematics import deproject, project, TCP_OFFSET
from collision import SelfCollision
from hand_model import revo2, OPEN as HAND_OPEN, PRESHAPE as HAND_PRESHAPE, CLOSED as HAND_CLOSE

# Hand orientations tried in order: (finger tilt below horizontal, wrist yaw toward the body midline).
# Turning the fingers toward the midline lets the elbow stay out when reaching across.
ORIENTATIONS = [(p, y) for y in (0, 20, 40) for p in (35, 20, 10, 50, 0, 65)]
MIN_GRASP_H = 0.04   # lowest TCP height tried above the support surface
MAX_GRASP_H = 0.07   # preferred upper bound; raised up to the object's top if the hand needs room
HAND_CLEARANCE = 0.01  # hand and carried object must stay this far above the table and other objects
TRAVEL_EXTRA = (0.10, 0.15, 0.20)  # approach/lift heights tried, so a carried object clears tall neighbours
CARRIED_RADIUS = 0.03  # carried object modelled as an upright cylinder hanging below the tool point
COLLISION_STEP = 0.03  # rad, joint-space sampling of every planned motion for the self-collision check
PLACE_CLEARANCE = 0.015
POS_TOL = 0.006      # max IK position error accepted in a plan
STEP = 0.02          # Cartesian step for straight-line segments
WORKSPACE = dict(x=(0.12, 0.45), y=(-0.35, 0.35), z=(-0.15, 0.35))   # TCP box, pelvis frame
READY_TCP = {"left": (0.14, 0.22, 0.22), "right": (0.14, -0.22, 0.22)}  # hands up, outside the view
CTRL_HZ = 50.0


class SkillError(Exception):
    pass


class HeightMap:
    """Highest point per 1 cm cell of the table area (pelvis frame), dilated by one cell.
    Cells without depth data count as free."""
    RES = 0.01
    X, Y = (0.05, 0.65), (-0.45, 0.45)

    def __init__(self, pts=(), flat=None):
        self.nx = int(round((self.X[1] - self.X[0]) / self.RES))
        self.ny = int(round((self.Y[1] - self.Y[0]) / self.RES))
        h = np.full((self.nx, self.ny), -np.inf if flat is None else float(flat))
        pts = np.asarray(pts, float).reshape(-1, 3)
        if len(pts):
            ix, iy = self._index(pts)
            m = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            np.maximum.at(h, (ix[m], iy[m]), pts[m, 2])
        p = np.pad(h, 1, constant_values=-np.inf)
        self.h = np.max([p[i:i + self.nx, j:j + self.ny] for i in range(3) for j in range(3)], axis=0)

    def _index(self, pts):
        return (np.floor((pts[:, 0] - self.X[0]) / self.RES).astype(int),
                np.floor((pts[:, 1] - self.Y[0]) / self.RES).astype(int))

    def margin(self, pts):
        """(smallest height of pts above the map, the point where it happens)."""
        ix, iy = self._index(pts)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ground = np.full(len(pts), -np.inf)
        ground[inside] = self.h[ix[inside], iy[inside]]
        gap = pts[:, 2] - ground
        i = int(np.argmin(gap))
        return gap[i], pts[i]


def _jpeg(img_rgb, quality=88):
    ok, buf = cv2.imencode(".jpg", img_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def annotate(rgb, marks=()):
    """Light coordinate grid so Fable can name pixels precisely; marks = [(u, v, label), ...]."""
    img = rgb.copy()
    h, w = img.shape[:2]
    over = img.copy()
    for x in range(80, w, 80):
        cv2.line(over, (x, 0), (x, h - 1), (255, 255, 255), 1)
    for y in range(80, h, 80):
        cv2.line(over, (0, y), (w - 1, y), (255, 255, 255), 1)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    for x in range(0, w, 80):
        cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(80, h, 80):
        cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    for u, v, label in marks:
        u, v = int(round(u)), int(round(v))
        cv2.drawMarker(img, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        if label:
            cv2.putText(img, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    return img


class Skills:
    def __init__(self, robot, kin, table_z=None, max_qd=0.5, confirm=None, log=print):
        self.robot, self.kin = robot, kin
        self.table_z = table_z          # fallback support height when there is no depth
        self.max_qd = max_qd            # rad/s, joint-space speed limit of every motion
        self.confirm = confirm or (lambda text, img: (True, ""))
        self.log = log
        self.held = {"left": None, "right": None}
        self.collision = SelfCollision()
        # Revo2 surface points (open and closed) in the TCP frame, for clearance checks.
        self.hand_pts = {side: revo2(side).clearance_points() - TCP_OFFSET[side] for side in ("left", "right")}
        self.ready_q = {}
        for side, p in READY_TCP.items():
            q, err, _ = kin.ik(side, pin.SE3(np.eye(3), np.array(p)), [np.zeros(7)])
            if err > 0.02:
                raise SkillError(f"ready pose unreachable for {side} ({err:.3f} m)")
            self.ready_q[side] = q
        self.seed_bank = {side: self._build_seed_bank(side) for side in ("left", "right")}

    def _build_seed_bank(self, side):
        """IK solutions over the workspace; the nearest one seeds each new solve (avoids local minima)."""
        rng = np.random.default_rng(0)
        a = self.kin.arm[side]
        rand = [rng.uniform(a["lo"], a["hi"]) for _ in range(4)]
        sign = 1 if side == "left" else -1
        R = pin.rpy.rpyToMatrix(0, np.radians(35), 0)
        pts, qs = [], []
        for x in np.arange(0.22, 0.43, 0.04):
            for y in np.arange(-0.15, 0.27, 0.06):
                for z in np.arange(-0.02, 0.25, 0.06):
                    p = np.array([x, sign * y, z])
                    q, err, _ = self.kin.ik(side, pin.SE3(R, p), [np.zeros(7), self.ready_q[side]] + rand, iters=100)
                    if err < 0.005:
                        pts.append(p)
                        qs.append(q)
        return np.array(pts), qs

    def _seeds(self, side, q_prev, p):
        pts, qs = self.seed_bank[side]
        near = [qs[i] for i in np.argsort(np.linalg.norm(pts - p, axis=1))[:2]]
        return [q_prev] + near + [self.ready_q[side]]

    # ================================================================ perception
    def _frame(self):
        rgb, depth, K = self.robot.rgbd()
        return rgb, depth, K

    def _points(self, depth, K, u, v, r_in, r_out):
        """3D points (pelvis frame) of valid depth pixels in an annulus around (u, v)."""
        h, w = depth.shape
        u0, u1 = max(0, int(u - r_out)), min(w, int(u + r_out + 1))
        v0, v1 = max(0, int(v - r_out)), min(h, int(v + r_out + 1))
        vv, uu = np.mgrid[v0:v1, u0:u1]
        d = depth[v0:v1, u0:u1]
        rr = np.hypot(uu - u, vv - v)
        m = (d > 0.1) & (d < 2.5) & (rr >= r_in) & (rr <= r_out)
        if m.sum() < 5:
            return np.zeros((0, 3))
        z = d[m]
        pc = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * z, (vv[m] - K[1, 2]) / K[1, 1] * z, z], 1)
        return (self.kin.T_cam.rotation @ pc.T).T + self.kin.T_cam.translation

    def locate(self, u, v, depth, K):
        """Surface point under pixel (u, v), the support surface around it, and the object top."""
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            raise SkillError(f"pixel ({u}, {v}) is outside the {depth.shape[1]}x{depth.shape[0]} image")
        core = self._points(depth, K, u, v, 0, 4)
        if len(core):
            p = self.kin.cam_to_pelvis(deproject(u, v, self._median_depth(depth, u, v), K))
            # Support = the lowest cluster of points around the object. A tall object can fill most of
            # the ring, so take the bottom few percent and average the points near it.
            ring = self._points(depth, K, u, v, 45, 90)
            if len(ring) > 20:
                z_low = np.percentile(ring[:, 2], 5)
                support = float(np.median(ring[ring[:, 2] < z_low + 0.015, 2]))
            else:
                support = p[2]
            near = self._points(depth, K, u, v, 0, 40)
            near = near[np.hypot(near[:, 0] - p[0], near[:, 1] - p[1]) < 0.05]
            top = max(np.percentile(near[:, 2], 95), p[2]) if len(near) else p[2]
            return p, support, top, "depth"
        if self.table_z is None:
            raise SkillError("no depth at that pixel and no --table-z fallback configured")
        # No depth: intersect the pixel ray with the table plane.
        ray = self.kin.T_cam.rotation @ deproject(u, v, 1.0, K)
        o = self.kin.T_cam.translation
        s = (self.table_z - o[2]) / ray[2]
        p = o + s * ray
        return p, self.table_z, self.table_z + 0.06, "table plane (no depth)"

    @staticmethod
    def _median_depth(depth, u, v, r=4):
        d = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        d = d[(d > 0.1) & (d < 2.5)]
        return float(np.median(d))

    def _cloud(self, depth, K, step=3):
        d = depth[::step, ::step]
        vv, uu = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
        m = (d > 0.1) & (d < 2.5)
        z = d[m]
        pc = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * z, (vv[m] - K[1, 2]) / K[1, 1] * z, z], 1)
        return (self.kin.T_cam.rotation @ pc.T).T + self.kin.T_cam.translation

    def _arm_samples(self):
        """Points along both forearms and hands (elbow to fingertips) at the current pose."""
        q14 = self.robot.arm_q()
        chain = []
        for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
            T = self.kin.fk(side, q14[sl])
            joints = [self.kin.arm[side]["data"].oMi[j].translation.copy() for j in range(4, 8)]
            pts = joints + [T.translation, T.act(np.array([0.08, 0.0, 0.0]))]
            for a, b in zip(pts[:-1], pts[1:]):
                chain += [a + t * (b - a) for t in np.linspace(0, 1, 5)]
        return np.array(chain)

    def _scene(self, depth, K, target=None, support=None, r=0.06):
        """Obstacle height maps from the depth image, without the robot's own arms and hands.
        hand: everything else, keeping the table under the target. obj: also without the target
        (for the object the hand is lifting)."""
        if depth is None:
            flat = HeightMap(flat=self.table_z)
            return dict(hand=flat, obj=flat)
        pts = self._cloud(depth, K)
        keep = np.ones(len(pts), bool)
        for c in self._arm_samples():
            keep &= ((pts - c) ** 2).sum(1) > 0.09 ** 2
        pts = pts[keep]
        if target is None:
            hm = HeightMap(pts)
            return dict(hand=hm, obj=hm)
        column = np.hypot(pts[:, 0] - target[0], pts[:, 1] - target[1]) < r
        return dict(hand=HeightMap(pts[~(column & (pts[:, 2] > support + 0.01))]), obj=HeightMap(pts[~column]))

    # ================================================================ planning
    def _in_workspace(self, p):
        return all(WORKSPACE[a][0] <= p[i] <= WORKSPACE[a][1] for i, a in enumerate("xyz"))

    def _solve_path(self, side, q_start, cart_points, R):
        """IK along a polyline of TCP positions (subdivided). Returns joint waypoints or raises."""
        qs, q = [], q_start
        prev = self.kin.fk(side, q_start).translation
        for p in cart_points:
            if not self._in_workspace(p):
                raise SkillError(f"target {np.round(p, 3).tolist()} is outside the safety box")
            n = max(1, int(np.ceil(np.linalg.norm(p - prev) / STEP)))
            for s in np.linspace(1.0 / n, 1.0, n):
                pt = prev + s * (p - prev)
                q, err, _ = self.kin.ik(side, pin.SE3(R, pt), self._seeds(side, q, pt))
                if err > POS_TOL:
                    raise SkillError(f"{side} hand cannot reach {np.round(pt, 3).tolist()} (IK error {err*100:.1f} cm)")
                qs.append(q)
            prev = p
        return qs

    def _hand_points(self, side, q7):
        T = self.kin.fk(side, q7)
        return (T.rotation @ self.hand_pts[side].T).T + T.translation

    def _check(self, side, q14_start, segs, scene, carry=False, below=0.0, static=False):
        """Sample the joint-space motion; raise if the robot hits itself, if the hand gets within
        HAND_CLEARANCE of the table or another object, or (carry=True) if the object hanging `below`
        the tool point does. static=True checks only the configuration q14_start."""
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        q14 = q14_start.copy()
        if static:
            segs = [[q14[sl].copy()]]
        ring = np.array([[CARRIED_RADIUS * np.cos(a), CARRIED_RADIUS * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
        for seg in segs:
            for q_next in seg:
                q_from = q14[sl].copy()
                n = max(1, int(np.ceil(np.max(np.abs(q_next - q_from)) / COLLISION_STEP)))
                for s in np.linspace(1.0 / n, 1.0, n):
                    q14[sl] = q_from + s * (q_next - q_from)
                    hits = self.collision.check(q14)
                    if hits:
                        a, b, d = hits[0]
                        raise SkillError(f"el brazo {side} chocaría consigo mismo ({a} contra {b}, {-d*100:.1f} cm)")
                    if scene is None:
                        continue
                    gap, at = scene["hand"].margin(self._hand_points(side, q14[sl]))
                    if gap < HAND_CLEARANCE:
                        raise SkillError(f"la mano {side} tocaría la mesa u otro objeto en "
                                         f"x={at[0]:.2f} y={at[1]:+.2f} (margen {gap*100:+.1f} cm)")
                    if carry:
                        tcp = self.kin.fk(side, q14[sl]).translation
                        obj = np.concatenate([np.c_[tcp[:2] + ring, np.full(8, tcp[2] - dz)] for dz in (below, below / 2)]
                                             + [[[tcp[0], tcp[1], tcp[2] - below]]])
                        gap, at = scene["obj"].margin(obj)
                        if gap < HAND_CLEARANCE:
                            raise SkillError(f"el objeto en la mano {side} chocaría con algo en "
                                             f"x={at[0]:.2f} y={at[1]:+.2f} (margen {gap*100:+.1f} cm)")

    def _grasp_rotation(self, side, pitch, yaw):
        yaw_signed = np.radians(yaw) * (1.0 if side == "right" else -1.0)
        return pin.rpy.rpyToMatrix(0, 0, yaw_signed) @ pin.rpy.rpyToMatrix(0, np.radians(pitch), 0)

    def _safe_free(self, side, q14, q_goal, scene, carry, below):
        try:
            self._check(side, q14, [[q_goal]], scene, carry, below)
            return [q_goal]
        except SkillError:
            via = [self.ready_q[side], q_goal]
            self._check(side, q14, [via], scene, carry, below)
            return via

    def _plan(self, side, waypoints, scene=None, carry=None, below=0.0, end_ready=False):
        """waypoints: list of (kind, xyz) with kind 'free' (joint move) or 'line' (straight).
        carry[i]: the hand holds the object while moving to waypoint i.

        Tries each hand orientation until one gives a plan that reaches every waypoint, never
        self-collides and keeps hand and object clear of the scene. Waypoints are checked on their
        own first (cheap), then the whole sampled motion. end_ready appends the return to the ready
        pose. Returns (segments, (pitch, yaw)).
        """
        carry = carry or [False] * len(waypoints)
        q14_now = self.robot.arm_q()
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        errors = []
        for pitch, yaw in ORIENTATIONS:
            R = self._grasp_rotation(side, pitch, yaw)
            try:
                q, keys = q14_now[sl], []
                for (_, p), c in zip(waypoints, carry):
                    p = np.asarray(p, float)
                    qn, err, _ = self.kin.ik(side, pin.SE3(R, p), self._seeds(side, q, p))
                    if err > POS_TOL or not self._in_workspace(p):
                        raise SkillError(f"la mano {side} no alcanza {np.round(p, 3).tolist()} "
                                         f"(le faltan {err*100:.1f} cm)")
                    q14 = q14_now.copy()
                    q14[sl] = qn
                    self._check(side, q14, [], scene, c, below, static=True)
                    keys.append(qn)
                    q = qn
                # One group of joint targets per waypoint. A free move whose straight joint-space
                # path would hit something goes through the ready pose (hands up and out) instead.
                segs, q14 = [], q14_now.copy()
                for (kind, p), qk, c in zip(waypoints, keys, carry):
                    if kind == "free":
                        group = self._safe_free(side, q14, qk, scene, c, below)
                    else:
                        group = self._solve_path(side, q14[sl], [np.asarray(p, float)], R)
                        self._check(side, q14, [group], scene, c, below)
                    segs.append(group)
                    q14[sl] = group[-1]
                if end_ready:
                    segs.append(self._safe_free(side, q14, self.ready_q[side], scene, False, 0.0))
                return segs, (pitch, yaw)
            except SkillError as e:
                errors.append(str(e))
        # Report the most specific failure: collisions beat reach errors.
        errors.sort(key=lambda e: 0 if ("chocaría" in e or "tocaría" in e) else 1)
        raise SkillError(errors[0])

    def reach_report(self, p):
        out = []
        for side in ("left", "right"):
            q, err, _ = self.kin.ik(side, pin.SE3(pin.rpy.rpyToMatrix(0, np.radians(35), 0), p),
                                    self._seeds(side, self.ready_q[side], p))
            out.append(f"{'izquierda' if side == 'left' else 'derecha'}: "
                       f"{'sí' if err < POS_TOL else f'no (le faltan {err*100:.0f} cm)'}")
        return ", ".join(out)

    # ================================================================ execution
    def _arm_q14(self):
        return self.robot.arm_q()

    def _play(self, side, segs):
        q14 = self._arm_q14()
        sl = slice(0, 7) if side == "left" else slice(7, 14)
        dt = 1.0 / CTRL_HZ
        for seg in segs:
            for q_next in seg:
                q_from = q14[sl].copy()
                n = max(1, int(np.ceil(np.max(np.abs(q_next - q_from)) / (self.max_qd * dt))))
                for s in np.linspace(1.0 / n, 1.0, n):
                    q14[sl] = q_from + s * (q_next - q_from)
                    self.robot.command_arm(q14, self.kin.gravity_tau(q14))
                    self.robot.sleep(dt)
        self.robot.sleep(0.3)

    def _hand(self, side, q6, wait=0.8):
        self.robot.set_hand(side, q6)
        self.robot.sleep(wait)

    def _pick_side(self, hand, p):
        if hand in ("left", "right"):
            return hand
        return "right" if p[1] < 0 else "left"

    # ================================================================ tools
    def look(self):
        rgb, _, _ = self._frame()
        return rgb

    def inspect_point(self, u, v):
        rgb, depth, K = self._frame()
        p, support, top, src = self.locate(u, v, depth, K)
        h, w = rgb.shape[:2]
        r = 60
        u0, v0 = int(np.clip(u - r, 0, w - 2 * r)), int(np.clip(v - r, 0, h - 2 * r))
        crop = cv2.resize(rgb[v0:v0 + 2 * r, u0:u0 + 2 * r], (6 * r, 6 * r), interpolation=cv2.INTER_CUBIC)
        cc = (int((u - u0) * 3), int((v - v0) * 3))
        cv2.drawMarker(crop, cc, (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
        text = (f"Pixel ({u}, {v}) -> punto 3D [{src}]: adelante {p[0]:.3f} m, lateral {p[1]:+.3f} m "
                f"(+ = izquierda del robot), altura {p[2]:+.3f} m (0 = pelvis). "
                f"Superficie de apoyo estimada {support:+.3f} m; altura del objeto ~{(top - support)*100:.0f} cm. "
                f"Alcanzable con mano {self.reach_report(np.array([p[0], p[1], max(p[2], support + MIN_GRASP_H)]))}. "
                f"El recorte ampliado x3 está centrado en la cruz magenta.")
        return text, [crop]

    def pick(self, u, v, hand="auto", name="objeto"):
        rgb, depth, K = self._frame()
        p, support, top, src = self.locate(u, v, depth, K)
        side = self._pick_side(hand, p)
        if self.held[side]:
            raise SkillError(f"la mano {side} ya sostiene '{self.held[side]['name']}'")
        # The clicked point is on the side facing the camera: nudge toward the object's centre.
        view = p - self.kin.T_cam.translation
        view[2] = 0
        c = p + 0.015 * view / max(np.linalg.norm(view), 1e-6)
        h_obj = max(top - support, 0.02)
        gz0 = support + float(np.clip(0.5 * h_obj, MIN_GRASP_H, MAX_GRASP_H))
        heights = np.arange(gz0, max(gz0, top - 0.015) + 1e-6, 0.01)
        scene = self._scene(depth, K, target=c, support=support)
        first_err, plan = None, None
        for gz in heights:
            for extra in TRAVEL_EXTRA:
                grasp = np.array([c[0], c[1], gz])
                try:
                    plan = self._plan(side, [("free", grasp + [0, 0, extra]), ("line", grasp),
                                             ("line", grasp + [0, 0, extra])],
                                      scene=scene, carry=[False, False, True], below=gz - support)
                    break
                except SkillError as e:
                    first_err = first_err or e
            if plan:
                break
        if plan is None:
            if "tocaría" in str(first_err) and h_obj < 0.08:
                raise SkillError(f"{first_err}: el objeto (~{h_obj*100:.0f} cm de alto) es muy bajo para el "
                                 f"agarre lateral; se necesitan ~8 cm o más")
            raise first_err
        segs, (pitch, yaw) = plan
        summary = (f"PICK '{name}' con mano {side}: pixel ({u},{v}) -> agarre en "
                   f"x={grasp[0]:.3f} y={grasp[1]:+.3f} z={grasp[2]:+.3f} m (pelvis), "
                   f"dedos {pitch}° abajo, {yaw}° hacia el centro, objeto ~{h_obj*100:.0f} cm [{src}]")
        ok, note = self.confirm(summary, annotate(rgb, [(u, v, name)]))
        if not ok:
            raise SkillError(f"el operador rechazó el movimiento{': ' + note if note else ''}")
        self.log(summary)
        self._hand(side, HAND_PRESHAPE, 0.5)
        self._play(side, segs[:2])
        self._hand(side, HAND_CLOSE, 1.0)
        state = self.robot.hand_state(side)
        self._play(side, segs[2:])
        fingers = state[2:].mean()
        contact = fingers < 0.9 * HAND_CLOSE[2:].mean()
        self.held[side] = dict(name=name, below=gz - support)
        msg = (f"Mano {side} cerrada en x={grasp[0]:.3f} y={grasp[1]:+.3f} z={grasp[2]:+.3f} y levantada {extra*100:.0f} cm. "
               f"Dedos en {fingers:.2f} (comando {HAND_CLOSE[2:].mean():.2f}): "
               + ("hay contacto, probablemente lo sostiene." if contact else
                  "los dedos cerraron casi por completo: probablemente NO agarró nada."))
        return msg, [self.look()]

    def place(self, u, v, hand="auto"):
        sides = [s for s in ("left", "right") if self.held[s]] if hand == "auto" else [hand]
        if not sides or not self.held[sides[0]]:
            raise SkillError("ninguna mano tiene un objeto que soltar" if hand == "auto"
                             else f"la mano {hand} no sostiene nada")
        side = sides[0]
        rgb, depth, K = self._frame()
        p, support, top, src = self.locate(u, v, depth, K)
        # Lowest release height that gives a valid plan; the object drops at most 2 cm extra
        # (higher drops tip cans and cups over).
        below = self.held[side]["below"]
        rz0 = p[2] + below + PLACE_CLEARANCE
        scene = self._scene(depth, K)
        first_err, plan = None, None
        for rz in rz0 + np.arange(0.0, 0.021, 0.01):
            for extra in TRAVEL_EXTRA:
                target = np.array([p[0], p[1], rz])
                try:
                    plan = self._plan(side, [("free", target + [0, 0, extra]), ("line", target),
                                             ("line", target + [0, 0, extra])],
                                      scene=scene, carry=[True, True, False], below=below, end_ready=True)
                    break
                except SkillError as e:
                    first_err = first_err or e
            if plan:
                break
        if plan is None:
            raise first_err
        segs, _ = plan
        name = self.held[side]["name"]
        summary = (f"PLACE '{name}' con mano {side}: pixel ({u},{v}) -> soltar en "
                   f"x={target[0]:.3f} y={target[1]:+.3f} z={target[2]:+.3f} m (superficie {p[2]:+.3f}) [{src}]")
        ok, note = self.confirm(summary, annotate(rgb, [(u, v, "soltar")]))
        if not ok:
            raise SkillError(f"el operador rechazó el movimiento{': ' + note if note else ''}")
        self.log(summary)
        self._play(side, segs[:2])
        self._hand(side, HAND_OPEN, 0.8)
        self.held[side] = None
        self._play(side, segs[2:])
        return f"Soltó '{name}' en x={target[0]:.3f} y={target[1]:+.3f} y volvió a la pose de espera.", [self.look()]

    def home(self):
        ok, note = self.confirm("HOME: ambos brazos a la pose de espera (manos arriba, fuera de la vista)",
                                annotate(self.look()))
        if not ok:
            raise SkillError(f"el operador rechazó el movimiento{': ' + note if note else ''}")
        for side in ("left", "right"):
            self._check(side, self.robot.arm_q(), [[self.ready_q[side]]], None)
            self._play(side, [[self.ready_q[side]]])
            if not self.held[side]:
                self._hand(side, HAND_OPEN, 0.3)
        return "Brazos en la pose de espera.", [self.look()]

    def status(self):
        def s(side):
            return f"sostiene '{self.held[side]['name']}'" if self.held[side] else "vacía"
        return f"Mano izquierda: {s('left')}. Mano derecha: {s('right')}."
