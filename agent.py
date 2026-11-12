"""Claude Fable 5.2 drives the G1 + BrainCo Revo2 for table-top pick-and-place.

Fable sees the head camera and calls tools with pixel coordinates; skills.py turns those into
solved, speed-limited arm motions. Each physical action waits for the operator's Enter on the real
robot (--auto skips that).

    # simulation (MuJoCo), no robot needed
    MUJOCO_GL=egl python agent.py --backend sim
    # real robot: G1 hanging in debug mode, cam_server.py running on the Jetson
    python agent.py --backend real --iface enx0c37969f434c --cam-host 192.168.123.164
"""
import argparse
import base64
import datetime
import json
import os
import sys
import time
import traceback

import anthropic
import cv2

from kinematics import G1Kinematics
from skills import Skills, SkillError, annotate

MODEL = "claude-fable-5-2"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens: input, 5-minute cache write, cache read, output. Fast mode doubles all.
PRICE = dict(inp=8.0, cache_write=10.0, cache_read=0.20, out=40.0)

SYSTEM = """Eres el cerebro de un humanoide Unitree G1 con dos manos BrainCo Revo2 de cinco dedos. \
El robot cuelga de un arnés, de frente a una mesa que le queda a la altura de la cadera. Ves a través \
de la cámara RealSense de su cabeza: imágenes de 640x480 con una cuadrícula tenue cada 80 px y etiquetas \
amarillas de coordenadas. Una persona te da instrucciones por la terminal y supervisa al robot.

Tus herramientas mueven el robot de verdad. Tú decides qué tomar y dónde dejarlo, señalando píxeles \
(x hacia la derecha, y hacia abajo) en la imagen completa de 640x480; el robot calcula solo la trayectoria. \
Para agarrar, señala el centro visible del objeto. Para soltar, señala el punto de la superficie donde \
debe quedar el objeto. Cuando no estés seguro de un punto, usa inspect_point antes de mover el brazo.

Límites físicos que conviene tener presentes:
- Los brazos son cortos: solo alcanzan una franja de la mesa cercana al robot, más o menos la mitad \
inferior de la imagen. La mano derecha trabaja del lado derecho de la imagen y la izquierda del lado \
izquierdo; las dos llegan al centro. Si algo queda fuera de alcance, pide a la persona que lo acerque.
- Hay dos agarres (parámetro grasp de pick). "arriba": la mano baja vertical con la palma hacia la mesa y \
envuelve la parte de arriba del objeto; sirve para objetos de unos 3 a 8 cm de ancho y al menos ~4 cm de \
alto. "lado": la palma de lado, como para tomar una lata; rodea el cuerpo del objeto por su lado exterior, \
necesita objetos de al menos ~8 cm de alto y espacio libre de ese lado. Si la persona pide un agarre, usa \
ese; si no, "arriba" para objetos bajos y cualquiera de los dos para los altos. Los muy planos no se pueden \
tomar. Señala el centro del cuerpo del objeto. \
Si el robot se niega a un movimiento porque chocaría con algo, propone otra mano, otro punto o pide ayuda.
- Si ves manos o brazos de una persona cerca del objetivo, no muevas el brazo: avisa y espera.

Después de cada pick o place revisa la imagen que regresa y confirma si salió bien; si falló, reintenta \
una vez ajustando el punto o explica qué pasó. Si el operador rechaza un movimiento, respeta su motivo.

Antes de cada acción física di en una frase corta qué vas a hacer. Habla en español, breve y claro."""

XY = {"x": {"type": "integer", "description": "Columna del píxel, 0-639"},
      "y": {"type": "integer", "description": "Fila del píxel, 0-479"}}
ARMS_NOTE = {
    ("left", "right"): "",
    ("left",): "\n\nEn esta sesión solo se mueve la mano IZQUIERDA; la derecha está bloqueada y no se puede usar. "
               "Trabaja con lo que alcance la mano izquierda (lado izquierdo y centro de la imagen).",
    ("right",): "\n\nEn esta sesión solo se mueve la mano DERECHA; la izquierda está bloqueada y no se puede usar. "
                "Trabaja con lo que alcance la mano derecha (lado derecho y centro de la imagen).",
}


def hand_schema(arms):
    return {"type": "string", "enum": ["auto", *arms],
            "description": "Mano a usar. auto = la del lado del objeto (o la única disponible). Elige una "
                           "explícitamente si el destino queda del otro lado."}


def _tool(name, description, props=None, required=()):
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props or {}, "required": list(required),
                             "additionalProperties": False}}


def make_tools(arms):
    HAND = hand_schema(arms)
    return [
        _tool("look", "Toma una foto nueva de la cámara de la cabeza y reporta qué sostiene cada mano."),
        _tool("inspect_point", "Muestra un recorte ampliado x3 alrededor de un píxel, con una cruz, y dice su "
              "posición 3D, la altura del objeto y qué mano lo alcanza. No mueve el robot.",
              XY, ["x", "y"]),
        _tool("pick", "Agarra el objeto que está en ese píxel: abre la mano, la lleva sobre el objeto, cierra y lo "
              "levanta 10 a 20 cm (más si hay objetos altos cerca). Regresa una foto nueva y si los dedos hicieron contacto.",
              {**XY, "hand": HAND, "object": {"type": "string", "description": "Nombre corto del objeto"},
               "grasp": {"type": "string", "enum": ["arriba", "lado"],
                         "description": "arriba: palma hacia la mesa, envuelve la parte de arriba del objeto. "
                                        "lado: palma de lado, rodea el cuerpo del objeto por su lado exterior."}},
              ["x", "y", "hand", "object", "grasp"]),
        _tool("place", "Lleva el objeto sostenido sobre ese píxel, lo baja hasta la superficie, lo suelta y "
              "regresa el brazo a la pose de espera. Regresa una foto nueva.",
              {**XY, "hand": {**HAND, "description": "Mano que sostiene el objeto (auto = la que tenga algo)."}},
              ["x", "y", "hand"]),
        _tool("home", "Lleva los brazos disponibles a la pose de espera (manos arriba, fuera de la vista de la cámara)."),
    ]


class Session:
    def __init__(self, skills, effort, fast, run_dir, max_tokens=32000):
        self.skills, self.effort, self.fast = skills, effort, fast
        # Fixed for the whole session (prompt cache + preserved thinking need an unchanged prefix).
        self.system = SYSTEM + ARMS_NOTE[tuple(skills.arms)]
        self.tools = make_tools(skills.arms)
        self.run_dir = run_dir
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self.messages = []       # append-only: Fable 5.2 rejects edited history with thinking blocks
        self.cost = 0.0
        self.n_img = 0

    # ---------------------------------------------------------------- helpers
    def _image_block(self, rgb, grid=True):
        img = annotate(rgb) if grid else rgb
        self.n_img += 1
        path = os.path.join(self.run_dir, f"img_{self.n_img:03d}.jpg")
        cv2.imwrite(path, img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}

    def _log(self, **kw):
        with open(os.path.join(self.run_dir, "events.jsonl"), "a") as f:
            f.write(json.dumps(dict(t=time.time(), **kw), ensure_ascii=False, default=str) + "\n")

    def _run_tool(self, name, args):
        sk = self.skills
        if name == "look":
            text, images = sk.status(), [sk.look()]
        elif name == "inspect_point":
            text, images = sk.inspect_point(args["x"], args["y"])
        elif name == "pick":
            text, images = sk.pick(args["x"], args["y"], args["hand"], args["object"], args.get("grasp", "arriba"))
            text += " " + sk.status()
        elif name == "place":
            text, images = sk.place(args["x"], args["y"], args["hand"])
            text += " " + sk.status()
        elif name == "home":
            text, images = sk.home()
        else:
            raise SkillError(f"herramienta desconocida {name}")
        content = [{"type": "text", "text": text}]
        for img in images:
            content.append(self._image_block(img, grid=img.shape[:2] == (480, 640)))
        return content

    def _account(self, msg):
        u = msg.usage
        mult = 2.0 if getattr(u, "speed", None) == "fast" else 1.0
        cost = mult * (u.input_tokens * PRICE["inp"] + (u.cache_creation_input_tokens or 0) * PRICE["cache_write"]
                       + (u.cache_read_input_tokens or 0) * PRICE["cache_read"] + u.output_tokens * PRICE["out"]) / 1e6
        self.cost += cost
        served = f" (servido por {msg.model})" if msg.model != MODEL else ""
        print(f"\n\033[2m[{msg.usage.output_tokens} tok salida, {u.cache_read_input_tokens or 0} de caché, "
              f"~${cost:.3f}; sesión ~${self.cost:.2f}]{served}\033[0m")

    # ---------------------------------------------------------------- the loop
    def user_turn(self, text):
        self.messages.append({"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "text", "text": f"Estado actual: {self.skills.status()} Imagen actual de la cámara:"},
            self._image_block(self.skills.look()),
        ]})
        self._log(role="user", text=text)
        self._feed("user", text)
        while True:
            params = dict(model=MODEL, max_tokens=self.max_tokens, system=self.system, tools=self.tools,
                          messages=self.messages, thinking={"type": "adaptive", "display": "summarized"},
                          output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
                          betas=[FALLBACK_BETA], fallbacks="default")
            if self.fast:
                params["speed"] = "fast"
            try:
                msg = self._stream(params)
            except KeyboardInterrupt:
                print("\n[respuesta cancelada]")
                return
            self.messages.append({"role": "assistant", "content": msg.content})
            self._log(role="assistant", stop=msg.stop_reason, model=msg.model,
                      content=[b.model_dump() for b in msg.content if b.type != "thinking"],
                      usage=msg.usage.model_dump())
            if msg.stop_reason == "refusal":
                d = msg.stop_details
                print(f"\n[Fable declinó la petición: {getattr(d, 'category', None)} {getattr(d, 'explanation', '')}]")
                return
            if msg.stop_reason == "max_tokens":
                print("\n[se alcanzó max_tokens; di 'continúa' para seguir]")
            tool_uses = [b for b in msg.content if b.type == "tool_use"]
            if not tool_uses:
                return
            results, stopped = [], False
            for tu in tool_uses:
                if stopped:
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                    "content": "No se ejecutó: el operador detuvo el robot."})
                    continue
                print(f"\033[36m→ {tu.name}({json.dumps(tu.input, ensure_ascii=False)})\033[0m")
                self._feed("tool", f"{tu.name}(" + ", ".join(f"{k}={v}" for k, v in tu.input.items()) + ")", hold=False)
                try:
                    content = self._run_tool(tu.name, tu.input)
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content})
                    print(f"\033[36m  {content[0]['text']}\033[0m")
                except KeyboardInterrupt:
                    # Ctrl+C mid-motion: freeze the arms where they are and hand control back to the operator.
                    robot, kin = self.skills.robot, self.skills.kin
                    q = robot.arm_q()
                    robot.command_arm(q, kin.gravity_tau(q))
                    print("\n\033[31m[detenido: brazos quietos en su posición actual]\033[0m")
                    stopped = True
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                    "content": "El operador detuvo el movimiento con Ctrl+C; los brazos quedaron "
                                               "quietos donde estaban. Espera nuevas instrucciones."})
                except SkillError as e:
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": str(e), "is_error": True})
                    print(f"\033[33m  error: {e}\033[0m")
                except Exception as e:
                    traceback.print_exc()
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": f"falla interna del robot: {e}", "is_error": True})
                r = results[-1]
                self._feed("error" if r.get("is_error") else "result",
                           r["content"] if isinstance(r["content"], str) else r["content"][0]["text"])
                self._log(role="tool", name=tu.name, input=tu.input,
                          result=results[-1]["content"] if isinstance(results[-1]["content"], str)
                          else results[-1]["content"][0]["text"])
            self.messages.append({"role": "user", "content": results})
            if stopped:
                return

    def _feed(self, kind, text, hold=True):
        """Sim recordings: show it in the reasoning panel."""
        robot = self.skills.robot
        if hasattr(robot, "feed_add"):
            robot.feed_add(kind, text, hold)

    def _stream(self, params):
        in_thinking = False
        said, thought = [], []
        with self.client.beta.messages.stream(**params) as stream:
            for ev in stream:
                if ev.type == "content_block_start" and ev.content_block.type == "fallback":
                    print("\n[fallback a otro modelo]")
                elif ev.type == "content_block_delta":
                    if ev.delta.type == "thinking_delta":
                        if not in_thinking:
                            print("\n\033[2m(pensando) ", end="")
                            in_thinking = True
                        print(ev.delta.thinking, end="", flush=True)
                        thought.append(ev.delta.thinking)
                    elif ev.delta.type == "text_delta":
                        if in_thinking:
                            print("\033[0m")
                            in_thinking = False
                        print(ev.delta.text, end="", flush=True)
                        said.append(ev.delta.text)
            if in_thinking:
                print("\033[0m")
            msg = stream.get_final_message()
        self._feed("thinking", "".join(thought))
        self._feed("said", "".join(said))
        self._account(msg)
        return msg


def make_confirm(view):
    def confirm(text, img):
        print(f"\n\033[1;35m{text}\033[0m")
        if view:
            cv2.imshow("fable_pick", img[:, :, ::-1])
            cv2.waitKey(1)
        ans = input("¿Ejecutar? [Enter] = sí, n <motivo> = no: ").strip()
        if ans.lower().startswith("n"):
            return False, ans[1:].strip()
        return True, ""
    return confirm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "real"], default="sim")
    ap.add_argument("--iface", default=os.environ.get("HH_IFACE", "enx0c37969f434c"), help="interfaz cableada al robot (real)")
    ap.add_argument("--cam-host", default=os.environ.get("HH_CAM_HOST", "192.168.123.164"), help="IP de la Jetson con cam_server.py")
    ap.add_argument("--cam-port", type=int, default=55600)
    ap.add_argument("--no-hands", action="store_true", help="no mandar comandos a las BrainCo")
    ap.add_argument("--arms", choices=["left", "right", "both"], default="both",
                    help="brazos que se pueden mover; el otro queda bloqueado en su pose inicial")
    ap.add_argument("--table-z", type=float, help="altura de la mesa vs pelvis (m), si no hay depth")
    ap.add_argument("--speed", type=float, default=None, help="rad/s articular (sim 1.0, real 0.4)")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--fast", action="store_true", help="fast mode: ~2.5x tokens/s, precio x2")
    ap.add_argument("--auto", action="store_true", help="no pedir confirmación antes de mover")
    ap.add_argument("--confirm", action="store_true", help="pedir confirmación también en sim")
    ap.add_argument("--view", action="store_true", help="ventana con la imagen de cada movimiento")
    ap.add_argument("--viewer", action="store_true", help="visor MuJoCo (sim)")
    ap.add_argument("--scene", default="lata_vaso", help="escena de la simulación: recoger_mesa | lata_vaso | lata_caja | cilindro_azul | vacia")
    ap.add_argument("--record", help="sim: graba la sesión en este .mp4")
    ap.add_argument("--seg-host", default=os.environ.get("HH_SEG_HOST"),
                    help="host del servidor SAM2 (segment.py serve); sin esto el pick usa solo la depth del clic")
    ap.add_argument("--seg-port", type=int, default=55610)
    ap.add_argument("--task", help="primera instrucción (si no, se pregunta)")
    a = ap.parse_args()

    kin = G1Kinematics()
    arms = ("left", "right") if a.arms == "both" else (a.arms,)
    if a.backend == "sim":
        from robot_sim import SimRobot
        robot = SimRobot(kin, realtime=a.viewer, viewer=a.viewer, scene=a.scene)
        if a.record:
            robot.start_recording(a.record)
        speed = a.speed or 1.0
        ask = a.confirm
    else:
        from robot_real import RealRobot
        robot = RealRobot(a.iface, a.cam_host, a.cam_port, hands=not a.no_hands, arms=arms)
        speed = a.speed or 0.4
        ask = not a.auto
    confirm = make_confirm(a.view) if ask else None
    def log(s):
        print(f"\033[35m{s}\033[0m")
        robot.caption = s           # shown in sim recordings
    segmenter = None
    if a.seg_host:
        from segment import SegClient
        segmenter = SegClient(a.seg_host, a.seg_port)
        if segmenter.ping():
            print(f"SAM2: tcp://{a.seg_host}:{a.seg_port}")
        else:
            print(f"\033[33mSAM2 no responde en tcp://{a.seg_host}:{a.seg_port} (¿corriste start_seg.sh?); "
                  f"el pick usará solo la depth del clic\033[0m")
            segmenter = None
    skills = Skills(robot, kin, table_z=a.table_z, max_qd=speed, confirm=confirm, log=log, arms=arms,
                    segmenter=segmenter)
    q_start = robot.arm_q()          # real robot: lowered back here on exit
    if a.backend == "real":
        # Raise the working arm(s) first, with the table away (the camera can't see its near edge).
        print("Subiendo el brazo a la pose de espera (mesa lejos)...")
        for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
            if side in arms:
                try:
                    skills._play(side, skills.route(side, skills.ready_q[side]))
                except SkillError as e:
                    print(f"No subo el brazo {side}: {e}")
                    return
        input("Brazo arriba. Acomoda la mesa y el objeto y presiona Enter para empezar: ")
    if a.backend == "sim":
        skills.home()          # start with the hands up and out of the camera's view

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                           datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{a.backend}")
    os.makedirs(run_dir, exist_ok=True)
    session = Session(skills, a.effort, a.fast, run_dir)
    print(f"Fable 5.2 ({a.effort}{', fast' if a.fast else ''}) controlando el G1 [{a.backend}]. "
          f"Registro en {run_dir}. Ctrl+C para salir.")
    try:
        task = a.task
        while True:
            if not task:
                task = input("\n\033[1mtú> \033[0m").strip()
                if not task:
                    continue
            try:
                session.user_turn(task)
            except anthropic.APIStatusError as e:
                print(f"\n[error de la API {e.status_code}: {e.message}]")
            except anthropic.APIConnectionError as e:
                print(f"\n[sin conexión con la API: {e}]")
            task = None
    except (KeyboardInterrupt, EOFError):
        print(f"\nsesión terminada, costo aproximado ${session.cost:.2f}")
    finally:
        if a.backend == "real":
            # Once this process exits nothing commands the arms: lower them in a controlled way, via the
            # ready pose, after the table is moved away (the camera can't see its edge near the robot).
            try:
                if input("\nAleja la mesa y presiona Enter para bajar el brazo a como estaba al inicio "
                         "(q = dejarlo donde está): ").strip().lower() != "q":
                    for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
                        if side in arms:
                            skills._play(side, skills.route(side, skills.ready_q[side]))
                            skills._play(side, skills.route(side, q_start[sl]))
            except (KeyboardInterrupt, EOFError):
                pass
            except SkillError as e:
                print(f"No bajo el brazo automáticamente: {e}")
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
