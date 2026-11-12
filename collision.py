"""Self-collision check for planned arm motions, on the G1 + Revo2 MuJoCo model (robot only, base fixed).

Collision geometry is the convex hull of each link mesh, which is conservative: a configuration
collides when two hulls penetrate. Pairs that already penetrate at the neutral pose (neighbouring
links), or at a pose the real robot is actually in (an arm hanging against the hip), only count when
they get deeper. Finger-finger contacts within a hand are ignored.
"""
import os
import numpy as np
import mujoco

from kinematics import DUAL_ARM_JOINTS
from g1_model import g1_revo2_spec, HAND_PREFIX

MARGIN = 0.01        # report anything closer than 1 cm
EXTRA = 0.005        # pairs already penetrating at the neutral pose: allowed this much deeper


class SelfCollision:
    def __init__(self):
        spec = g1_revo2_spec()
        for g in spec.geoms:
            if g.contype or g.conaffinity:
                g.margin = MARGIN
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in DUAL_ARM_JOINTS])
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        self.body = names
        # Bodies that move with each arm (shoulder_yaw onward, hand included).
        self.hand_bodies = {side: {b for b, n in enumerate(names) if n and n.startswith(f"{HAND_PREFIX}{side}_")}
                            for side in ("left", "right")}
        self.arm_bodies = {side: {b for b, n in enumerate(names) if n and n.startswith(side) and
                                  any(k in n for k in ("shoulder_yaw", "elbow", "wrist"))} | self.hand_bodies[side]
                           for side in ("left", "right")}
        self.baseline = self._pairs(np.zeros(14))

    def _pairs(self, q14):
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.arm_adr] = q14
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = sorted((self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]))
            out[(b1, b2)] = min(out.get((b1, b2), 1.0), c.dist)
        return out

    def _arm_pair(self, b1, b2):
        """Pair that involves a moving arm and is not two parts of the same hand."""
        if not any(b1 in s or b2 in s for s in self.arm_bodies.values()):
            return False
        return not any(b1 in h and b2 in h for h in self.hand_bodies.values())

    def contacts(self, q14):
        """{(body1, body2): depth} of hull pairs already penetrating at q14 (a pose the robot really is in)."""
        return {p: d for p, d in self._pairs(np.asarray(q14, float)).items() if d < 0 and self._arm_pair(*p)}

    def check(self, q14, tolerate=None):
        """Returns [(body_a, body_b, distance_m)] of contacts involving a moving arm that are new or
        deeper than allowed. tolerate: {pair: depth} measured at real poses (contacts() of where the
        robot actually is); those pairs only count if they get deeper than that."""
        hits = []
        for (b1, b2), dist in self._pairs(np.asarray(q14, float)).items():
            limit = min(0.0, self.baseline.get((b1, b2), 1.0) - EXTRA)
            if tolerate and (b1, b2) in tolerate:
                limit = min(limit, tolerate[(b1, b2)] - EXTRA)
            if dist >= limit or not self._arm_pair(b1, b2):
                continue
            hits.append((self.body[b1], self.body[b2], dist))
        return hits
