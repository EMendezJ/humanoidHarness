"""Pipeline test without the API: an oracle points at each sim object and the tray, using the same
pick/place tools Fable calls. Checks that every object ends up inside the tray.

    MUJOCO_GL=egl python selftest_sim.py [--seeds 5] [--save-dir DIR]
"""
import argparse
import os
import numpy as np
import cv2

from kinematics import G1Kinematics, project
from robot_sim import SimRobot, SCENES, TRAY_XY, TRAY_HALF
from skills import Skills, SkillError, annotate


def run(seed, scene, save_dir=None, video=None):
    kin = G1Kinematics()
    robot = SimRobot(kin, seed=seed, scene=scene)
    if video:
        robot.start_recording(video)
    sk = Skills(robot, kin, max_qd=0.5, log=lambda s: setattr(robot, "caption", s))
    sent = []
    command = robot.command_arm
    robot.command_arm = lambda q, tau=None: (sent.append(np.array(q)), command(q, tau))
    sk.home()
    # A free spot in the tray for each object (the oracle never stacks them).
    spots = [(-0.025, -0.045), (0.022, 0.03)]
    results = []
    for (name, p), (dx, dy) in zip(robot.object_positions().items(), spots):
        u, v = project(kin.pelvis_to_cam(p), robot.K)
        tu, tv = project(kin.pelvis_to_cam(np.array([TRAY_XY[0] + dx, TRAY_XY[1] + dy, 0.0])), robot.K)
        try:
            sk.pick(int(round(u)), int(round(v)), "auto", name)
            sk.place(int(round(tu)), int(round(tv)), "auto")
            robot.sleep(1.0)
            q = robot.object_positions()[name]
            inside = abs(q[0] - TRAY_XY[0]) < TRAY_HALF[0] and abs(q[1] - TRAY_XY[1]) < TRAY_HALF[1]
            results.append((name, inside, f"final {np.round(q, 3).tolist()}"))
        except SkillError as e:
            results.append((name, False, f"SkillError: {e}"))
            robot.caption = f"RECHAZADO: {e}"
            robot.sleep(2.0)
            sk.home()
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f"{scene}_seed{seed}_{name}.png"), annotate(sk.look())[:, :, ::-1])
    robot.close()
    hits = sum(bool(sk.collision.check(q)) for q in sent)
    results.append(("(auditoría)", hits == 0, f"{hits} de {len(sent)} configuraciones enviadas con autocolisión"))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--save-dir")
    ap.add_argument("--video", help="graba la primera semilla de cada escena: <video>_<escena>.mp4")
    a = ap.parse_args()
    total = ok = 0
    for scene in SCENES:
        for seed in range(a.seeds):
            video = f"{os.path.splitext(a.video)[0]}_{scene}.mp4" if a.video and seed == 0 else None
            for name, inside, info in run(seed, scene, a.save_dir, video):
                total += 1
                ok += inside
                print(f"{scene} seed {seed} {name:16s} {'OK ' if inside else 'FAIL'} {info}")
    print(f"\n{ok}/{total} OK (objetos en la charola + auditorías sin autocolisión)")
