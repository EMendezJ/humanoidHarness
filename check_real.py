"""Step-by-step bring-up on the real G1, before letting Fable drive it.

    python check_real.py status [--iface eth0]                    # read-only: who commands the motors, arm/waist angles
    python check_real.py cam   [--cam-host IP]                    # read-only: camera, depth, table height
    python check_real.py aruco [--marker-size 0.05]               # read-only: marker pose, table tilt, PnP vs depth
    python check_real.py fxcal [--marker-size 0.05]               # read-only: focal-scale correction from the marker
    python check_real.py guide [--side left]                      # read-only, live: where to slide the marker
    python check_real.py calib --side left --aruco 10 --marker-size 0.05   # camera calibration from the hand marker
    python check_real.py hand  --side left [--iface IF]           # open / close / open one hand
    python check_real.py arm   --side left [--aruco ID]           # ready pose, hover (over a marker), TCP overlay

Only --side moves: the other arm and hand stay locked where they are, and the waist and legs are
held at their start-up angles. Every motion waits for Enter. Robot hanging on the gantry, in debug
mode. Images go to runs/check_*.
"""
import argparse
import datetime
import os
import numpy as np
import cv2
import pinocchio as pin

from kinematics import G1Kinematics, project
from skills import Skills, SkillError, HeightMap, HAND_OPEN, HAND_CLOSE, annotate
import aruco
import calib

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


def check_status(a, kin):
    """Read-only: is anything else commanding the motors (Unitree's locomotion publishes rt/lowcmd at
    hundreds of Hz)? If so our controller would fight it: switch the robot to debug mode first."""
    import time as _t
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    ChannelFactoryInitialize(0, a.iface)
    seen = dict(state=None, cmd=0)
    ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: seen.__setitem__("state", m), 10)
    ChannelSubscriber("rt/lowcmd", LowCmd_).Init(lambda m: seen.__setitem__("cmd", seen["cmd"] + 1), 10)
    _t.sleep(2.0)
    st = seen["state"]
    if st is None:
        print(f"no llega rt/lowstate por {a.iface}: ¿interfaz correcta? ¿robot encendido?")
        return
    q = np.array([st.motor_state[i].q for i in range(29)])
    rate = seen["cmd"] / 2.0
    rpy = np.degrees(np.array(st.imu_state.rpy))
    print(f"mode_machine {st.mode_machine} | rt/lowcmd de otro proceso: {rate:.0f} Hz")
    print(f"cintura yaw/roll/pitch: {np.degrees(q[12:15]).round(1).tolist()} grados | IMU roll/pitch: {rpy[:2].round(1).tolist()}")
    print(f"brazo izquierdo (rad): {q[15:22].round(2).tolist()}")
    print(f"brazo derecho   (rad): {q[22:29].round(2).tolist()}")
    for side, sl in (("izquierdo", slice(0, 7)), ("derecho", slice(7, 14))):
        print(f"TCP {side} (pelvis): {np.round(kin.fk('left' if side == 'izquierdo' else 'right', q[15:29][sl]).translation, 3).tolist()}")
    if rate > 5:
        print("\n*** Alguien está comandando los motores (¿modo de caminar de Unitree?). NO arranques hand/arm ni el\n"
              "    agente: pon el robot en modo debug (colgado; L2+R2 o mode_switch/dev_mode_on.py) y repite status.")
    else:
        print("\nNadie más comanda los motores: se puede seguir con cam / aruco / hand / arm.")


MARKER_TARGET = (0.33, 0.12)      # marker centre for the arm checks: x forward, |y| to the arm's side (m)
MARKER_TOL = 0.025


def placement_advice(center, side):
    """How to move the marker so its centre lands on MARKER_TARGET; None when it is there."""
    tx, ty = MARKER_TARGET[0], MARKER_TARGET[1] * (1 if side == "left" else -1)
    dx, dy = center[0] - tx, center[1] - ty
    if abs(dx) <= MARKER_TOL and abs(dy) <= MARKER_TOL:
        return None
    out = []
    if abs(dx) > MARKER_TOL:
        out.append(f"{'acércalo' if dx > 0 else 'aléjalo'} {abs(dx) * 100:.0f} cm {'al' if dx > 0 else 'del'} robot")
    if abs(dy) > MARKER_TOL:
        out.append(f"muévelo {abs(dy) * 100:.0f} cm a la {'derecha' if dy > 0 else 'izquierda'} del robot")
    return ", ".join(out)


def check_guide(a, kin):
    """Read-only, live: where the marker is and how to move it onto the arm check's target spot."""
    import time as _t
    from robot_real import CameraClient
    cam, ok_since = CameraClient(a.cam_host, a.cam_port), None
    print(f"objetivo: centro del marcador {MARKER_TARGET[0]*100:.0f} cm al frente del pelvis y "
          f"{MARKER_TARGET[1]*100:.0f} cm a la {'izquierda' if a.side == 'left' else 'derecha'}. Ctrl+C para salir.")
    while True:
        rgb, depth, K = cam.get()
        markers, poses = find_markers(rgb, depth, K, kin)
        match = [p for (_, mid, _), p in zip(markers, poses) if p is not None and (a.aruco is None or mid == a.aruco)]
        if not match:
            print("  no veo el marcador (completo y plano en la mesa)")
            ok_since = None
        else:
            c = match[0]["center"]
            advice = placement_advice(c, a.side)
            print(f"  centro x={c[0]*100:5.1f} cm  y={c[1]*100:+5.1f} cm  ->  {advice or 'EN SU LUGAR'}")
            ok_since = (ok_since or _t.time()) if advice is None else None
            if ok_since and _t.time() - ok_since > 2.0:
                print("listo: ya puedes correr el paso del brazo")
                return
        _t.sleep(1.0)


def find_markers(rgb, depth, K, kin, size=None):
    markers = aruco.detect(rgb)
    poses = [aruco.marker_pose(kin, c, depth, K, size) if depth is not None else None for _, _, c in markers]
    return markers, poses


def check_aruco(a, kin):
    from robot_real import CameraClient
    cam = CameraClient(a.cam_host, a.cam_port)
    # Average 5 frames: single-frame depth at the corners is noisy (~1 cm at 0.8 m).
    runs = []
    for _ in range(5):
        rgb, depth, K = cam.get()
        runs.append(find_markers(rgb, depth, K, kin, a.marker_size))
    markers, poses = runs[-1]
    for k, (_, mid, _) in enumerate(markers):
        same = [p for ms, ps in runs for (_, i, _), p in zip(ms, ps) if i == mid and p is not None]
        if same and poses[k] is not None:
            for key in ("center", "side", "tilt_deg", "pitch_deg", "roll_deg", "center_pnp"):
                if all(key in p for p in same):
                    poses[k][key] = np.mean([p[key] for p in same], axis=0)
    if not markers:
        print("No se detectó ningún marcador (probé 4x4, 5x5, 6x6, 7x7, original y AprilTag 36h11).\n"
              "Revisa que esté plano, completo en la imagen, bien iluminado y con borde blanco.")
        save("aruco.png", annotate(rgb))
        return
    print(f"diccionario {markers[0][0]}, {len(markers)} marcador(es)")
    for (name, mid, corners), pose in zip(markers, poses):
        if pose is None:
            print(f"  id {mid}: sin profundidad en sus esquinas")
            continue
        x, y, z = pose["center"]
        print(f"  id {mid}: centro x={x:.3f} y={y:+.3f} z={z:+.3f} m (pelvis) | lado medido {pose['side']*100:.1f} cm"
              f" | inclinación {pose['tilt_deg']:.1f}° (pitch {pose['pitch_deg']:+.1f}°, roll {pose['roll_deg']:+.1f}°)")
        if "center_pnp" in pose:
            d = np.linalg.norm(pose["center_pnp"] - pose["center"])
            cam = kin.T_cam.translation
            r_pnp, r_depth = np.linalg.norm(pose["center_pnp"] - cam), np.linalg.norm(pose["center"] - cam)
            print(f"         PnP (solo imagen, lado {a.marker_size*100:.1f} cm) vs profundidad: {d*1000:.0f} mm "
                  f"(distancia a la cámara {r_pnp:.3f} vs {r_depth:.3f} m)")
            print(f"         si el lado impreso es exacto, el fx de color debería ser ~{K[0, 0] * pose['side'] / a.marker_size:.0f}"
                  f" (ahora {K[0, 0]:.1f})")
        if pose["center"][0] > 0.45:
            print(f"         el marcador está a {pose['center'][0]*100:.0f} cm al frente: fuera del alcance del brazo (~40 cm)")
    print("\nCómo leerlo:\n"
          "- 'lado medido' debe coincidir con el lado impreso (±3 mm): si no, la profundidad está mal escalada.\n"
          "- PnP vs profundidad < 10 mm: intrínsecos y alineación color-profundidad bien.\n"
          "- z = altura de la mesa respecto al pelvis; meta entre -0.03 y +0.03.\n"
          "- inclinación > 2-3°: el torso no está nivelado respecto a la mesa (cintura/arnés) o la cámara\n"
          "  difiere del URDF; las alturas se equivocan ~tan(inclinación) x distancia.")
    save("aruco.png", aruco.draw(rgb, markers, poses))


def check_fxcal(a, kin):
    """Read-only. With the printed side known (ruler), the marker gives the absolute focal scale twice:
    side measured with depth / printed side, and depth distance / image-only (PnP) distance. Both are
    1.0 when the colour intrinsics are right. Use it with the marker near the robot (~0.5 m)."""
    from robot_real import CameraClient
    if not a.marker_size:
        print("falta --marker-size (lado del cuadro negro, medido con regla)")
        return
    cam = CameraClient(a.cam_host, a.cam_port)
    k_side, k_dist, dist = [], [], []
    for _ in range(15):
        rgb, depth, K = cam.get()
        for (_, mid, corners), pose in zip(*find_markers(rgb, depth, K, kin, a.marker_size)):
            if pose is None or "center_pnp" not in pose:
                continue
            c = kin.T_cam.translation
            r_depth, r_pnp = np.linalg.norm(pose["center"] - c), np.linalg.norm(pose["center_pnp"] - c)
            k_side.append(pose["side"] / a.marker_size)
            k_dist.append(r_depth / r_pnp)
            dist.append(r_depth)
    if not k_side:
        print("no veo el marcador con profundidad")
        return
    scale_now = cam.meta.get("fx_scale", 1.0)
    ks, kd = np.median(k_side), np.median(k_dist)
    k = (ks + kd) / 2
    print(f"marcador a {np.median(dist):.2f} m de la cámara, {len(k_side)} mediciones")
    print(f"  lado por profundidad / regla: {ks:.3f} (±{np.std(k_side):.3f})")
    print(f"  distancia profundidad / PnP:  {kd:.3f} (±{np.std(k_dist):.3f})")
    print(f"  fx ahora {K[0, 0]:.1f} (escala {scale_now:.3f}) -> sugerido {K[0, 0] * k:.1f} (escala {scale_now * k:.3f})")
    if abs(k - 1) < 0.01:
        print("  menos de 1%: la calibración está bien, no hace falta cambiar nada")
    else:
        print(f"\n  para aplicarlo:\n    echo {scale_now * k:.3f} > ~/humanoidHarness/camera_fx_scale && ~/humanoidHarness/start_cam.sh\n"
              f"  y vuelve a correr fxcal: debe dar ~1.000")


def make_robot(a):
    from robot_real import RealRobot
    return RealRobot(a.iface, a.cam_host, a.cam_port, hands=True, arms=(a.side,))


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
    sk = Skills(robot, kin, max_qd=a.speed, log=print, arms=(a.side,))
    side = a.side
    sl = slice(0, 7) if side == "left" else slice(7, 14)
    q = robot.arm_q()
    q_start = q[sl].copy()
    print(f"q brazos ahora: {np.round(q, 2).tolist()}")
    print(f"TCP {side} ahora (pelvis): {np.round(kin.fk(side, q[sl]).translation, 3).tolist()}")
    if not go(f"Llevar el brazo {side} a la pose de espera a {a.speed} rad/s?"):
        return
    try:
        up = sk.route(side, sk.ready_q[side])
    except SkillError as e:
        print(f"\nNo me muevo: {e}")
        return
    try:
        sk._play(side, up)
        try:
            hover_over_marker(a, kin, robot, sk, side, sl)
        except SkillError as e:
            print(f"\nPlan rechazado, el brazo no se movió: {e}")
    finally:
        # When this program exits nothing commands the arm any more: lower it in a controlled way to
        # where it started (hanging), after the table is moved away (the camera can't see its near edge).
        if go("Fin. Aleja la mesa del robot y presiona Enter para bajar el brazo a como estaba al inicio\n"
              "(q = dejarlo donde está)"):
            try:
                sk._play(side, sk.route(side, sk.ready_q[side]))
                sk._play(side, sk.route(side, q_start))
            except SkillError as e:
                print(f"No bajo el brazo automáticamente: {e}")


def calib_poses(kin, side):
    """Hand poses for the calibration: tool point in front of the body, back of the hand turned up
    toward the head camera (rolled about the fingers), varied so the fit is well conditioned."""
    sign = 1 if side == "left" else -1
    spots = [(0.27, 0.12, 0.12), (0.27, 0.20, 0.20), (0.31, 0.15, 0.09), (0.24, 0.18, 0.25),
             (0.30, 0.22, 0.14), (0.33, 0.10, 0.16), (0.26, 0.26, 0.10), (0.29, 0.06, 0.20),
             (0.22, 0.22, 0.18), (0.34, 0.18, 0.22)]
    # Wide rotation spread: with similar hand orientations a camera offset hides in the marker's
    # placement on the hand and the fit is ill-conditioned.
    turns = [(r, p, y) for r in (50, 90, 130) for p in (-40, 0, 20) for y in (-35, 0, 35)]
    out = []
    for x, y, z in spots:
        for roll, pitch, yaw in turns:
            R = pin.rpy.rpyToMatrix(0, 0, np.radians(yaw * sign)) @ pin.rpy.rpyToMatrix(0, np.radians(pitch), 0) \
                @ pin.rpy.rpyToMatrix(np.radians(roll * sign), 0, 0)
            T = pin.SE3(R, np.array([x, y * sign, z]))
            back = R @ np.array([0, sign, 0])                     # back-of-hand normal
            to_cam = kin.T_cam.translation - T.translation
            u, v = project(kin.pelvis_to_cam(T.translation), np.array([[580.0, 0, 322], [0, 580.0, 237], [0, 0, 1]]))
            if back @ to_cam / np.linalg.norm(to_cam) > np.cos(np.radians(60)) and 60 < u < 580 and 60 < v < 420:
                out.append(T)
    # At most MAX_CALIB_POSES, spread over the list (positions x rotations) to keep the variety.
    if len(out) > MAX_CALIB_POSES:
        out = [out[i] for i in np.linspace(0, len(out) - 1, MAX_CALIB_POSES).round().astype(int)]
    return out


MAX_CALIB_POSES = 30


def check_calib(a, kin):
    """Moves only --side's arm through calibration poses (each checked for collisions), measures the
    hand marker at each, fits the camera correction and saves camera_calib.json."""
    if not a.marker_size or a.aruco is None:
        print("hace falta --aruco <id del marcador de la mano> y --marker-size <lado del cuadro negro en m>")
        return
    robot = make_robot(a)
    sk = Skills(robot, kin, max_qd=a.speed, log=print, arms=(a.side,))
    side, sl = a.side, (slice(0, 7) if a.side == "left" else slice(7, 14))
    q_start = robot.arm_q()[sl].copy()
    poses = calib_poses(kin, side)
    print(f"{len(poses)} poses candidatas. Aleja la mesa: la mano se mueve frente al robot entre 4 y 25 cm sobre el pelvis.")
    if not go(f"Empezar? El brazo {side} recorre las poses a {a.speed} rad/s (Ctrl+C lo deja donde esté)"):
        return
    wrists, measured = [], []
    try:
        sk._play(side, sk.route(side, sk.ready_q[side]))
        for k, T in enumerate(poses):
            q14 = robot.arm_cmd()
            q, err, rot = kin.ik(side, T, sk._seeds(side, q14[sl], T.translation))
            if err > 0.01 or rot > np.radians(15):
                continue
            # Free space in front of the robot (table moved away): self-collision only. The depth map is
            # not used here because, before calibrating, the arm's own points may land off its model.
            try:
                group = sk._safe_free(side, q14, q, None, False, 0.0)
            except SkillError as e:
                print(f"  pose {k}: la salto ({e})")
                continue
            sk._play(side, [group])
            robot.sleep(0.8)
            views = []
            for _ in range(3):
                rgb, depth, K = robot.rgbd()
                for name, mid, corners in aruco.detect(rgb):
                    if mid == a.aruco:
                        Tm = calib.marker_pose_pnp(corners, a.marker_size, K)
                        if Tm is not None:
                            views.append(Tm)
            if not views:
                print(f"  pose {k}: no veo el marcador {a.aruco}")
                continue
            # Middle view by translation (robust to one bad frame).
            mid_view = sorted(views, key=lambda T: T.translation[2])[len(views) // 2]
            Tw = kin.fk(side, robot.arm_q()[sl]) * pin.SE3(np.eye(3), -kin_tcp(side))
            wrists.append(Tw)
            measured.append(mid_view)
            print(f"  pose {k}: marcador a {np.linalg.norm(mid_view.translation):.2f} m de la cámara ({len(wrists)} medidas)")
        sk._play(side, [[sk.ready_q[side]]])
    finally:
        if go("Fin del recorrido. Presiona Enter para bajar el brazo a como estaba al inicio (q = dejarlo)"):
            try:
                sk._play(side, sk.route(side, sk.ready_q[side]))
                sk._play(side, sk.route(side, q_start))
            except SkillError as e:
                print(f"No bajo el brazo automáticamente: {e}")
    if len(wrists) < 5:
        print(f"Solo {len(wrists)} medidas: hacen falta al menos 5. Revisa que el marcador se vea en las poses.")
        return
    res = calib.solve(kin.T_cam_nominal, wrists, measured)
    print(f"\n{res['poses']} medidas. Error de posición del marcador: URDF {res['rms_urdf_m']*100:.1f} cm -> "
          f"calibrado {res['rms_calibrated_m']*100:.1f} cm")
    print(f"corrección de la cámara: traslación {np.round(np.array(res['translation_m'])*100, 1).tolist()} cm, "
          f"rotación {np.round(res['rot_vec_deg'], 2).tolist()} grados")
    print(f"error por pose (cm): {np.round(np.array(res['per_pose_error_m'])*100, 1).tolist()}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calib.json")
    if go(f"Guardar en {path}? (hh.sh la usa desde ahí y deja de usar camera_offset)"):
        calib.save(path, res, dict(marker_id=a.aruco, marker_size_m=a.marker_size, side=side))
        print("guardada")


def kin_tcp(side):
    from kinematics import TCP_OFFSET
    return TCP_OFFSET[side]


def hover_over_marker(a, kin, robot, sk, side, sl):
    sign = 1 if side == "left" else -1
    if not go(f"Brazo arriba. Ahora acerca la mesa al robot y pon el marcador plano con su centro ~33 cm al frente\n"
              f"del pelvis y ~{12 * sign:+d} cm a los lados (izquierda = +), sin nada a su lado exterior. ¿Listo?"):
        return
    while True:
        table_z = a.table_z
        rgb, depth, K = robot.rgbd()
        if a.aruco is not None:
            markers, poses = find_markers(rgb, depth, K, kin)
            match = [p for (_, mid, _), p in zip(markers, poses) if mid == a.aruco and p is not None]
            if not match:
                if not go(f"no veo el marcador id {a.aruco} con profundidad (ids visibles: {[m[1] for m in markers]}).\n"
                          f"Acomódalo completo y plano. ¿Volver a medir?"):
                    return
                continue
            target = match[0]["center"]
            table_z = target[2]
            hover = np.array([target[0], target[1], table_z + a.hover])
            print(f"marcador {a.aruco}: centro x={target[0]:.3f} y={target[1]:+.3f} z={target[2]:+.3f} m")
            advice = placement_advice(target, side)
            if advice and not go(f"El marcador no está donde lo alcanza bien el brazo: {advice}.\n"
                                 f"Muévelo (el brazo espera arriba) y presiona Enter para volver a medir"):
                return
            if advice:
                continue
        else:
            hover = np.array([0.33, 0.12 * sign, table_z + a.hover])
        # Obstacles from the depth image (table, anything on it); flat table only if there is no depth.
        scene = sk._scene(depth, K) if depth is not None else dict(hand=HeightMap(flat=table_z), obj=HeightMap(flat=table_z))
        try:
            segs, (pitch, yaw) = sk._plan(side, [("free", hover)], scene=scene)
            break
        except SkillError as e:
            if not go(f"Plan rechazado, el brazo no se movió: {e}\nAcomoda el marcador o lo que estorbe y presiona "
                      f"Enter para volver a intentar"):
                return
    if not go(f"Llevar el centro del agarre a {a.hover*100:.0f} cm sobre la mesa en {hover.round(3).tolist()} "
              f"(dedos {pitch}°, giro {yaw}°)?"):
        return
    sk._play(side, segs)
    rgb, _, K = robot.rgbd()
    T = kin.fk(side, robot.arm_q()[sl])
    u, v = project(kin.pelvis_to_cam(T.translation), K)
    print(f"TCP medido por cinemática: {np.round(T.translation, 3).tolist()} -> pixel ({u:.0f}, {v:.0f})")
    print("Mide con regla, visto desde arriba, cuánto se desvía del centro del marcador el punto ~4.7 cm frente\n"
          "a la palma, a media palma (adelante/atrás y hacia los lados).")
    save("tcp_overlay.png", annotate(rgb, [(u, v, "TCP")]))
    go("Cuando termines de medir, Enter para regresar a la pose de espera")
    sk._play(side, [[sk.ready_q[side]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["status", "cam", "aruco", "fxcal", "guide", "hand", "arm", "calib"])
    ap.add_argument("--side", choices=["left", "right"], default="left", help="el único brazo/mano que se mueve")
    ap.add_argument("--iface", default=os.environ.get("HH_IFACE", "enx0c37969f434c"))
    ap.add_argument("--cam-host", default=os.environ.get("HH_CAM_HOST", "192.168.123.164"))
    ap.add_argument("--cam-port", type=int, default=55600)
    ap.add_argument("--table-z", type=float, default=0.0, help="altura de la mesa vs pelvis (de 'cam')")
    ap.add_argument("--speed", type=float, default=0.3, help="rad/s")
    ap.add_argument("--marker-size", type=float, default=float(os.environ["HH_MARKER_SIZE"]) if os.environ.get("HH_MARKER_SIZE") else None,
                    help="lado del cuadro negro del marcador en m (para PnP)")
    ap.add_argument("--aruco", type=int, help="arm: flotar sobre este id de marcador")
    ap.add_argument("--hover", type=float, default=0.12, help="arm: altura del centro del agarre sobre la mesa (m)")
    a = ap.parse_args()
    kin = G1Kinematics()
    dict(status=check_status, cam=check_cam, aruco=check_aruco, fxcal=check_fxcal, guide=check_guide,
         hand=check_hand, arm=check_arm, calib=check_calib)[a.what](a, kin)


if __name__ == "__main__":
    main()
