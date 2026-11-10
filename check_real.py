"""Step-by-step bring-up on the real G1, before letting Fable drive it.

    python check_real.py cam   [--cam-host IP]            # read-only: camera, depth, table height
    python check_real.py hand  --side right [--iface IF]  # open / preshape / close / open one hand
    python check_real.py arm   --side right [--iface IF]  # ready pose, hover over the table, TCP overlay

Every motion waits for Enter. Robot hanging on the gantry, in debug mode. Images go to runs/check_*.
"""
import argparse
import datetime
import os
import numpy as np
import cv2
import pinocchio as pin

from kinematics import G1Kinematics, project
from skills import Skills, HeightMap, HAND_OPEN, HAND_CLOSE, annotate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                   "check_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


def save(name, rgb):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, rgb[:, :, ::-1])
    print(f"  imagen: {path}")


def go(msg):
    return input(f"\n{msg} [Enter = sí, q = salir] ").strip().lower() != "q"


def check_cam(a, kin):
    from robot_real import CameraClient
    rgb, depth, K = CameraClient(a.cam_host, a.cam_port).get()
    print(f"color {rgb.shape}, intrínsecos fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    if depth is None:
        print("SIN depth: el agente necesitará --table-z")
        save("cam.png", annotate(rgb))
        return

    sk = Skills.__new__(Skills)      # only locate() is used: no robot, no motion
    sk.kin, sk.table_z = kin, None
    valid = (depth > 0.1).mean()
    print(f"depth válido en {valid*100:.0f}% de la imagen")
    marks = []
    for u, v in ((320, 300), (200, 360), (440, 360), (320, 420)):
        try:
            p, support, top, _ = sk.locate(u, v, depth, K)
            print(f"  pixel ({u},{v}) -> x={p[0]:.3f} y={p[1]:+.3f} z={p[2]:+.3f} m (pelvis)   apoyo {support:+.3f}")
            marks.append((u, v, f"z={p[2]:+.3f}"))
        except Exception as e:
            print(f"  pixel ({u},{v}): {e}")
    print("Con la mesa vacía, z debe salir igual en todos los puntos (= altura de la mesa vs pelvis).\n"
          "Si cambia mucho de cerca a lejos, la inclinación de la cámara del URDF no coincide con la real.\n"
          "Meta: mesa entre -0.03 y +0.03 m.")
    save("cam.png", annotate(rgb, marks))


def make_robot(a):
    from robot_real import RealRobot
    return RealRobot(a.iface, a.cam_host, a.cam_port, hands=True)


def check_hand(a, kin):
    robot = make_robot(a)
    for name, q in (("abrir (pulgar arriba, como al bajar sobre el objeto)", HAND_OPEN),
                    ("cerrar (el pulgar gira a oposición)", HAND_CLOSE), ("abrir", HAND_OPEN)):
        if not go(f"Mano {a.side}: {name} {np.round(q, 2).tolist()}?"):
            return
        robot.set_hand(a.side, q)
        robot.sleep(1.5)
        print(f"  estado: {np.round(robot.hand_state(a.side), 2).tolist()}")


def check_arm(a, kin):
    robot = make_robot(a)
    sk = Skills(robot, kin, max_qd=a.speed, log=print)
    side = a.side
    sl = slice(0, 7) if side == "left" else slice(7, 14)
    q = robot.arm_q()
    print(f"q brazos ahora: {np.round(q, 2).tolist()}")
    print(f"TCP {side} ahora (pelvis): {np.round(kin.fk(side, q[sl]).translation, 3).tolist()}")
    if not go(f"Llevar el brazo {side} a la pose de espera a {a.speed} rad/s?"):
        return
    sk._check(side, robot.arm_q(), [[sk.ready_q[side]]], None)
    sk._play(side, [[sk.ready_q[side]]])
    y = -0.12 if side == "right" else 0.12
    hover = np.array([0.33, y, a.table_z + 0.15])
    flat = HeightMap(flat=a.table_z)
    segs, (pitch, yaw) = sk._plan(side, [("free", hover)], scene=dict(hand=flat, obj=flat))
    if not go(f"Bajar a flotar 15 cm sobre la mesa en {hover.round(3).tolist()} (dedos {pitch}°, giro {yaw}°)?"):
        return
    sk._play(side, segs)
    rgb, _, K = robot.rgbd()
    T = kin.fk(side, robot.arm_q()[sl])
    u, v = project(kin.pelvis_to_cam(T.translation), K)
    print(f"TCP medido por cinemática: {np.round(T.translation, 3).tolist()} -> pixel ({u:.0f}, {v:.0f})")
    print("La cruz magenta debe caer donde quedaría el centro de un objeto agarrado (frente a la palma).\n"
          "Si cae corrida, ajusta TCP_OFFSET en kinematics.py (o la pose de la cámara).")
    save("tcp_overlay.png", annotate(rgb, [(u, v, "TCP")]))
    if go("Regresar a la pose de espera?"):
        sk._play(side, [[sk.ready_q[side]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cam", "hand", "arm"])
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--iface", default="enx0c37969f434c")
    ap.add_argument("--cam-host", default="192.168.123.164")
    ap.add_argument("--cam-port", type=int, default=55600)
    ap.add_argument("--table-z", type=float, default=0.0, help="altura de la mesa vs pelvis (de 'cam')")
    ap.add_argument("--speed", type=float, default=0.3, help="rad/s")
    a = ap.parse_args()
    kin = G1Kinematics()
    dict(cam=check_cam, hand=check_hand, arm=check_arm)[a.what](a, kin)


if __name__ == "__main__":
    main()
