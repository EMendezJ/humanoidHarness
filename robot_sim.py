"""MuJoCo stand-in for the hanging G1: same interface as robot_real.RealRobot.

The robot (G1 with BrainCo Revo2 hands, see g1_model) is kinematic: base fixed like on the gantry,
arms and fingers follow commands exactly, and it does not collide with the objects. Closing a hand
next to an object welds the object to the tool point. This tests the perception -> 3D -> IK ->
motion chain and Fable's decisions, not grasp physics.
"""
import os
import subprocess
import unicodedata
import time
import cv2
import numpy as np
import mujoco
import pinocchio as pin

from kinematics import DUAL_ARM_JOINTS, R_LINK_OPTICAL
from g1_model import g1_revo2_spec, hand_qpos_targets

PELVIS_Z = 0.80          # world height of the pelvis while hanging
TABLE_TOP = 0.0          # table surface relative to the pelvis (see README: table at pelvis height)
W, H, FOVY = 640, 480, 42.6   # ~ D435i color at 640x480 (fx = fy ~ 615)

# Objects: name -> (shape, size, rgba, xy relative to pelvis). The Revo2 goes on the outer side of
# the object with its fingers forward, so it needs ~8-10 cm free there: the reachable strip holds the
# plate in the middle and one object on each side.
LATA = ("cylinder", (0.033, 0.05), (0.1, 0.7, 0.2, 1), (0.35, -0.175))
SCENES = {
    "lata_vaso": {"lata_verde": LATA,
                  "vaso_rojo": ("cylinder", (0.035, 0.045), (0.85, 0.1, 0.1, 1), (0.33, 0.165))},
    "lata_caja": {"lata_verde": LATA,
                  "caja_amarilla": ("box", (0.02, 0.03, 0.06), (0.95, 0.85, 0.1, 1), (0.35, 0.16))},
}
TRAY_XY, TRAY_HALF, TRAY_WALL = (0.34, -0.03), (0.07, 0.085), 0.015  # half sizes x, y; low rim (a plate)


class SimRobot:
    def __init__(self, kin, realtime=False, viewer=False, seed=None, scene="lata_vaso"):
        self.kin = kin
        self.objects = SCENES[scene]
        self.realtime = realtime
        spec = g1_revo2_spec()                  # G1 with BrainCo Revo2 hands, base fixed
        spec.body("pelvis").pos = [0, 0, PELVIS_Z]

        # Head camera at the URDF d435 mount. MuJoCo cameras look down -z with +y up.
        R_link = pin.rpy.rpyToMatrix(0, 0.8307767239493009, 0)
        R_cam = R_link @ R_LINK_OPTICAL @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_cam.flatten())
        spec.body("torso_link").add_camera(name="head", pos=[0.0576235, 0.01753, 0.42987],
                                           quat=quat, fovy=FOVY)

        wb = spec.worldbody
        # External camera for recordings: front-left of the robot, looking at the hands and the table.
        pos, target = np.array([1.05, 0.70, 1.30]), np.array([0.22, 0.0, 0.88])
        f = (target - pos) / np.linalg.norm(target - pos)
        x = np.cross(f, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        R_ext = np.stack([x, np.cross(x, f), -f], axis=1)
        q_ext = np.zeros(4)
        mujoco.mju_mat2Quat(q_ext, R_ext.flatten())
        wb.add_camera(name="external", pos=pos, quat=q_ext, fovy=45)
        wb.add_light(pos=[0.3, 0, 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        wb.add_light(pos=[1.5, -1.0, 2.0], dir=[-0.6, 0.4, -0.7], diffuse=[0.4, 0.4, 0.4])
        grp = dict(contype=2, conaffinity=2)        # objects collide with table/floor, never the robot
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.55, 0.55, 0.6, 1], **grp)
        top = PELVIS_Z + TABLE_TOP
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.45, top / 2],
                    pos=[0.38, 0, top / 2], rgba=[0.62, 0.47, 0.33, 1], **grp)
        # Blue open tray made of five boxes.
        tx, ty = TRAY_XY
        tray = wb.add_body(name="tray", pos=[tx, ty, top])
        hx, hy = TRAY_HALF
        tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.003],
                      pos=[0, 0, 0.003], rgba=[0.15, 0.3, 0.85, 1], **grp)
        for dx, dy, sx, sy in ((1, 0, 0.004, hy), (-1, 0, 0.004, hy),
                               (0, 1, hx, 0.004), (0, -1, hx, 0.004)):
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, TRAY_WALL / 2],
                          pos=[dx * hx, dy * hy, TRAY_WALL / 2],
                          rgba=[0.15, 0.3, 0.85, 1], **grp)
        rng = np.random.default_rng(seed)
        shapes = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                  "box": mujoco.mjtGeom.mjGEOM_BOX}
        for name, (shape, size, rgba, (x, y)) in self.objects.items():
            if seed is not None:
                x, y = x + rng.uniform(-0.008, 0.008), y + rng.uniform(-0.008, 0.008)
            half_h = size[1] if shape == "cylinder" else size[-1] if shape == "box" else size[0]
            b = wb.add_body(name=name, pos=[x, y, top + half_h])
            b.add_freejoint()
            # condim 6 enables rolling friction, otherwise the ball keeps rolling forever
            b.add_geom(type=shapes[shape], size=list(size) + [0] * (3 - len(size)), rgba=rgba,
                       mass=0.1, condim=6, friction=[1.0, 0.01, 0.005], **grp)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in DUAL_ARM_JOINTS])
        self.robot_adr = np.array([m.jnt_qposadr[j] for j in range(m.njnt)
                                   if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE])
        self.robot_dof = np.array([m.jnt_dofadr[j] for j in range(m.njnt)
                                   if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE])
        self.obj_adr = {n: m.jnt_qposadr[m.body(n).jntadr[0]] for n in self.objects}
        self.obj_dof = {n: m.jnt_dofadr[m.body(n).jntadr[0]] for n in self.objects}

        self.q_robot = np.zeros(len(self.robot_adr))
        self.arm_target = np.zeros(14)
        self.hand_cmd = {"left": np.zeros(6), "right": np.zeros(6)}
        self.attached = {"left": None, "right": None}     # (object name, SE3 tcp->object)
        self.renderer = mujoco.Renderer(m, H, W)
        fy = (H / 2) / np.tan(np.radians(FOVY) / 2)
        self.K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
        self.caption = ""          # planner summary, top bar of recordings
        self.subtitle = ""         # what Fable said last, bottom of recordings
        self._rec = None
        self.viewer = None
        if viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(m, self.data)
        self._apply_robot()
        mujoco.mj_forward(m, self.data)
        self.sleep(0.5)       # let objects settle on the table

    # ---------------------------------------------------------------- robot interface
    def arm_q(self):
        return self.data.qpos[self.arm_adr].copy()

    def command_arm(self, q14, tau14=None):
        self.arm_target = np.asarray(q14, float).copy()

    def set_hand(self, side, q6):
        q6 = np.clip(np.asarray(q6, float), 0, 1)
        closing = q6[2:].mean() > self.hand_cmd[side][2:].mean() + 0.2
        opening = q6[2:].mean() < self.hand_cmd[side][2:].mean() - 0.2
        self.hand_cmd[side] = q6
        if closing and self.attached[side] is None:
            self._try_grasp(side)
        if opening and self.attached[side] is not None:
            self.attached[side] = None

    def hand_state(self, side):
        q = self.hand_cmd[side].copy()
        if self.attached[side] is not None:     # fingers stop on the object
            q[[0, 2, 3, 4, 5]] = np.minimum(q[[0, 2, 3, 4, 5]], 0.55)
        return q

    def rgbd(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="head")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="head")
        depth = self.renderer.render().astype(np.float32).copy()
        self.renderer.disable_depth_rendering()
        depth[depth > 3.0] = 0.0          # like RealSense: no return -> 0
        return rgb, depth, self.K.copy()

    def sleep(self, dt):
        n = max(1, int(round(dt / self.model.opt.timestep)))
        t0 = time.time()
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._apply_robot()
            if self._rec and self.data.time >= self._rec["next"]:
                self._write_frame()
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        if self.realtime:
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def close(self):
        self.stop_recording()
        self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()

    # ---------------------------------------------------------------- recording
    def start_recording(self, path, fps=30):
        """MP4 (H.264) at sim time: external view | head camera, with self.caption on top."""
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{2 * W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", path]
        self._rec = dict(proc=subprocess.Popen(cmd, stdin=subprocess.PIPE), dt=1.0 / fps,
                         next=self.data.time, path=path)

    def stop_recording(self):
        if self._rec:
            self._rec["proc"].stdin.close()
            self._rec["proc"].wait()
            self._rec = None

    @staticmethod
    def _ascii(text):
        """OpenCV's Hershey fonts only draw ASCII: '°' and accents would show up as '??'."""
        text = text.replace("°", " grados")
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

    def _write_frame(self):
        mujoco.mj_forward(self.model, self.data)
        views = []
        for cam in ("external", "head"):
            self.renderer.update_scene(self.data, camera=cam)
            views.append(self.renderer.render())
        frame = np.ascontiguousarray(np.hstack(views))
        cv2.rectangle(frame, (0, 0), (2 * W, 26), (0, 0, 0), -1)
        cv2.putText(frame, self._ascii(f"t={self.data.time:5.1f}s  {self.caption}"), (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "camara de la cabeza", (W + 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if self.subtitle:
            sub = self._ascii(self.subtitle)
            lines = [sub[i:i + 70] for i in range(0, min(len(sub), 210), 70)]
            y0 = H - 12 - 20 * (len(lines) - 1)
            cv2.rectangle(frame, (0, y0 - 18), (W, H), (0, 0, 0), -1)
            for k, line in enumerate(lines):
                cv2.putText(frame, line, (8, y0 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 230, 255), 1,
                            cv2.LINE_AA)
        self._rec["proc"].stdin.write(frame.tobytes())
        self._rec["next"] += self._rec["dt"]

    # ---------------------------------------------------------------- sim internals
    def object_positions(self):
        """Ground truth for the self-test, relative to the pelvis."""
        return {n: self.data.qpos[a:a + 3] - np.array([0, 0, PELVIS_Z]) for n, a in self.obj_adr.items()}

    def _tcp_world(self, side):
        q7 = self.data.qpos[self.arm_adr[:7] if side == "left" else self.arm_adr[7:]]
        T = self.kin.fk(side, q7)
        return pin.SE3(T.rotation, T.translation + np.array([0, 0, PELVIS_Z]))

    def _try_grasp(self, side):
        tcp = self._tcp_world(side)
        best = None
        for n, a in self.obj_adr.items():
            if any(att and att[0] == n for att in self.attached.values()):
                continue
            d = np.linalg.norm(self.data.qpos[a:a + 3] - tcp.translation)
            if d < 0.045 and (best is None or d < best[1]):
                best = (n, d)
        if best:
            a = self.obj_adr[best[0]]
            T_obj = pin.SE3(pin.Quaternion(*self.data.qpos[a + 3:a + 7]).toRotationMatrix(),
                            self.data.qpos[a:a + 3].copy())
            self.attached[side] = (best[0], tcp.actInv(T_obj))

    def _apply_robot(self):
        d = self.data
        d.qpos[self.robot_adr] = 0.0
        d.qpos[self.arm_adr] = self.arm_target
        for side in ("left", "right"):
            for adr, angle in hand_qpos_targets(self.model, side, self.hand_state(side)):
                d.qpos[adr] = angle
        d.qvel[self.robot_dof] = 0.0
        for side, att in self.attached.items():
            if att is None:
                continue
            name, T_rel = att
            T = self._tcp_world(side) * T_rel
            a, v = self.obj_adr[name], self.obj_dof[name]
            d.qpos[a:a + 3] = T.translation
            d.qpos[a + 3:a + 7] = pin.Quaternion(T.rotation).coeffs()[[3, 0, 1, 2]]
            d.qvel[v:v + 6] = 0.0
