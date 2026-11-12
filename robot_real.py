"""Real G1 Edu U7 hanging on the gantry in debug mode (rt/lowcmd), BrainCo Revo2 hands, head D435i.

Same interface as robot_sim.SimRobot. Not built on xr_teleoperate's G1_29_ArmController on purpose:
that class starts publishing with q_target = zeros(14) at a 20 rad/s limit, so the arms snap to the
zero pose the moment it is created. This controller starts by holding the measured pose, and its
250 Hz publisher rate-limits every arm target on top of the skills' own speed limit.
"""
import json
import threading
import time
import numpy as np
import cv2
import zmq

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.utils.crc import CRC

NUM_BODY = 29
ARM_IDS = list(range(15, 29))            # left 15-21, right 22-28 (same order as kinematics.DUAL_ARM_JOINTS)
WRIST_IDS = {19, 20, 21, 26, 27, 28}
WEAK_IDS = {4, 10, 15, 16, 17, 18, 22, 23, 24, 25}   # ankle pitch + shoulders/elbows, as in xr_teleoperate
KP_HIGH, KD_HIGH, KP_LOW, KD_LOW, KP_WRIST, KD_WRIST = 300.0, 3.0, 80.0, 3.0, 40.0, 1.5
PUBLISH_HZ = 250.0
MAX_TARGET_SPEED = 1.0                   # rad/s, hard cap inside the publisher


class RealRobot:
    def __init__(self, iface, cam_host, cam_port=55600, hands=True, arms=("left", "right")):
        """arms: sides allowed to move. A locked arm keeps the pose it had at start-up (same target,
        no torque change) and a locked hand never receives a command, whatever the layers above do."""
        ChannelFactoryInitialize(0, iface)
        self._low = None
        self._lock = threading.Lock()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self._on_lowstate, 10)
        t0 = time.time()
        while self._low is None:
            if time.time() - t0 > 5.0:
                raise RuntimeError(f"no rt/lowstate on {iface}: check the cable/interface and that the robot is on")
            time.sleep(0.05)

        # Lock every body joint where it is now; arms included, until the skills send targets.
        low = self._low
        q_now = np.array([low.motor_state[i].q for i in range(NUM_BODY)])
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = low.mode_machine
        for i in range(NUM_BODY):
            kp, kd = ((KP_WRIST, KD_WRIST) if i in WRIST_IDS else
                      (KP_LOW, KD_LOW) if i in WEAK_IDS else (KP_HIGH, KD_HIGH))
            c = self.msg.motor_cmd[i]
            c.mode, c.q, c.dq, c.tau, c.kp, c.kd = 1, float(q_now[i]), 0.0, 0.0, kp, kd
        self.target = q_now[ARM_IDS].copy()
        self.tau = np.zeros(14)
        self.arms = tuple(arms)
        self.locked = np.zeros(14, bool)
        for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
            if side not in self.arms:
                self.locked[sl] = True
        self.locked_target = self.target.copy()
        self.sent = self.target.copy()
        self.crc = CRC()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self._running = True
        threading.Thread(target=self._publish_loop, daemon=True, name="lowcmd").start()

        self.hands = hands
        self.hand_cmd = {"left": np.zeros(6), "right": np.zeros(6)}
        self.hand_q = {"left": None, "right": None}
        if hands:
            self.hand_pub, self.hand_sub = {}, {}
            for side in self.arms:          # locked hands get no publisher at all
                self.hand_pub[side] = ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
                self.hand_pub[side].Init()
                self.hand_sub[side] = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
                self.hand_sub[side].Init(lambda m, s=side: self._on_hand(s, m), 10)
            threading.Thread(target=self._hand_loop, daemon=True, name="brainco").start()

        self.cam = CameraClient(cam_host, cam_port)

    # ---------------------------------------------------------------- DDS callbacks / loops
    def _on_lowstate(self, msg):
        self._low = msg

    def _on_hand(self, side, msg):
        self.hand_q[side] = np.array([msg.states[i].q for i in range(6)])

    def _publish_loop(self):
        dt = 1.0 / PUBLISH_HZ
        while self._running:
            t0 = time.time()
            with self._lock:
                target, tau = self.target.copy(), self.tau.copy()
            step = MAX_TARGET_SPEED * dt
            self.sent = self.sent + np.clip(target - self.sent, -step, step)
            for k, i in enumerate(ARM_IDS):
                c = self.msg.motor_cmd[i]
                c.q, c.dq, c.tau = float(self.sent[k]), 0.0, float(tau[k])
            self.msg.crc = self.crc.Crc(self.msg)
            self.pub.Write(self.msg)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def _hand_loop(self):
        msgs = {}
        for side in self.hand_pub:
            m = MotorCmds_()
            m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
            msgs[side] = m
        while self._running:
            for side, m in msgs.items():
                for i in range(6):
                    m.cmds[i].q, m.cmds[i].dq = float(self.hand_cmd[side][i]), 1.0
                self.hand_pub[side].Write(m)
            time.sleep(0.02)

    # ---------------------------------------------------------------- robot interface
    def arm_q(self):
        low = self._low
        return np.array([low.motor_state[i].q for i in ARM_IDS])

    def arm_cmd(self):
        with self._lock:
            return self.target.copy()

    def command_arm(self, q14, tau14=None):
        q = np.asarray(q14, float).copy()
        tau = np.zeros(14) if tau14 is None else np.asarray(tau14, float).copy()
        q[self.locked] = self.locked_target[self.locked]
        tau[self.locked] = 0.0
        with self._lock:
            self.target, self.tau = q, tau

    def set_hand(self, side, q6):
        if not self.hands or side not in self.arms:
            return
        self.hand_cmd[side] = np.clip(np.asarray(q6, float), 0.0, 1.0)

    def hand_state(self, side):
        q = self.hand_q[side]
        return self.hand_cmd[side].copy() if q is None else q.copy()

    def rgbd(self):
        return self.cam.get()

    def sleep(self, dt):
        time.sleep(dt)

    def close(self):
        # Leave the publisher running until the process exits: stopping it would drop the hold.
        self.cam.close()


class CameraClient:
    """Client for cam_server.py: color JPEG + aligned depth (uint16 mm PNG) + intrinsics."""

    def __init__(self, host, port, timeout_ms=3000):
        self.addr, self.timeout_ms = f"tcp://{host}:{port}", timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def get(self):
        try:
            self.sock.send(b"get")
            meta, jpg, png = self.sock.recv_multipart()
        except zmq.Again:
            self.sock.close()          # a REQ socket is stuck after a timeout: rebuild it
            self._connect()
            raise RuntimeError(f"no frame from camera server {self.addr}")
        meta = json.loads(meta)
        rgb = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        depth = None
        if png:
            depth = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
        K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1.0]])
        self.meta = meta
        return rgb, depth, K

    def close(self):
        self.sock.close()
