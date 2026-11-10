# humanoidHarness — Claude Fable 5.2 maneja un Unitree G1 con manos BrainCo Revo2

Arnés para que un LLM (Claude Fable 5.2) haga pick-and-place con un humanoide Unitree G1 Edu U7 y manos
BrainCo Revo2 de 5 dedos, en simulación (MuJoCo) o en el robot real.

Demo en simulación: 


https://github.com/user-attachments/assets/6e996c19-a172-460e-b80d-0ef071b1758a

Instrucción: "Pon la lata verde y el vaso rojo dentro de la charola azul". Fable lo resolvió en 3 llamadas a la API, por ~$0.09.

Fable 5.2 (`claude-fable-5-2`) ve la cámara de la cabeza y decide **qué** tomar y **dónde** dejarlo,
señalando píxeles. El código local decide **cómo** moverse:

```
Fable 5.2 ──herramientas──► look · inspect_point · pick(x,y,mano,objeto) · place(x,y,mano) · home
                                │
   píxel + depth D435i → punto 3D (marco pelvis) → plan IK completo (pinocchio, por brazo)
   → trayectoria articular lenta → rt/lowcmd (modo debug) + rt/brainco/*/cmd (Revo2 de 5 dedos)
```

Cada plan se resuelve completo (todos los puntos de IK) antes de mover el brazo. En el robot real,
cada `pick`, `place` y `home` espera tu Enter.

| archivo | qué hace |
|---|---|
| `agent.py` | ciclo con Fable 5.2: streaming, herramientas `strict`, historial que solo crece, caching, fallback por rechazo, registro en `runs/` |
| `skills.py` | percepción (píxel→3D, superficie de apoyo, altura del objeto), planes de agarre/soltar, ejecución |
| `kinematics.py` | IK por brazo (codo hacia afuera), pose de la cámara (`d435_joint` del URDF), TCP de la mano, gravedad |
| `hand_model.py` | BrainCo Revo2 de 5 dedos desde sus URDF/mallas: montaje en la muñeca, 6 comandos → 11 articulaciones, centro del agarre, puntos de su superficie |
| `g1_model.py` | G1 con las Revo2 en lugar de las Dex3, para la simulación y el detector de choques |
| `collision.py` | autocolisión de cada trayectoria planeada (envolventes de las mallas en MuJoCo) |
| `robot_sim.py` | G1 colgado en MuJoCo con mesa, charola y dos escenas (`lata_vaso`, `lata_caja`); graba MP4 |
| `robot_real.py` | controlador propio en modo debug + BrainCo + cliente de cámara |
| `cam_server.py` | corre en la Jetson: color + depth alineado + intrínsecos por ZMQ |
| `check_real.py` | puesta en marcha paso a paso: cámara, mano, brazo |
| `selftest_sim.py` | prueba sin API: un oráculo señala los objetos y usa las mismas herramientas |

## Instalación

Python 3.10, dependencias en [`requirements.txt`](requirements.txt). Para el robot real también hace falta
[`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2_python).

Los modelos del G1 y de la Revo2 **no vienen en este repo**: se toman de un checkout de
[`xr_teleoperate`](https://github.com/unitreerobotics/xr_teleoperate) (carpeta `assets/`). Por defecto se buscan en
`~/xr_teleoperate/assets`; para otra ruta usa estas variables:

| variable | por defecto |
|---|---|
| `FABLE_PICK_URDF` | `~/xr_teleoperate/assets/g1/g1_body29_hand14.urdf` |
| `FABLE_PICK_MJCF` | `~/xr_teleoperate/assets/g1/g1_body29_hand14.xml` |
| `FABLE_PICK_BRAINCO` | `~/xr_teleoperate/assets/brainco_hand` |

Credenciales de la API: `ant auth login` (CLI de Anthropic) o `export ANTHROPIC_API_KEY=...` (de
[platform.claude.com](https://platform.claude.com/settings/keys)).

En la laptop del laboratorio el entorno es un venv encima del env conda `tv` (pinocchio, SDK de Unitree):

```bash
~/miniconda3/envs/tv/bin/python -m venv --system-site-packages .venv
PYTHONNOUSERSITE=1 .venv/bin/pip install anthropic mujoco
```

Ahí usa siempre `PYTHONNOUSERSITE=1` (un numpy de `~/.local` rompe pinocchio).

## Simulación

```bash
MUJOCO_GL=egl PYTHONNOUSERSITE=1 .venv/bin/python selftest_sim.py --seeds 5 --video runs/prueba.mp4   # sin API
PYTHONNOUSERSITE=1 .venv/bin/python agent.py --backend sim --viewer --scene lata_caja --record runs/fable.mp4
```

En la simulación el robot es cinemático y no choca con los objetos; al cerrar la mano cerca de un objeto, este
se pega a la mano. Sirve para probar percepción → IK → movimiento y las decisiones de Fable, no la física del
agarre. Las revisiones de seguridad sí son las mismas del robot real: cada plan se revisa contra autocolisión
(brazo contra torso, etc.) y contra un mapa de alturas sacado de la profundidad (mesa, otros objetos), para la
mano y para el objeto que carga; si algo estorba prueba trayectos más altos o se niega a moverse.

## Robot real

### 1. La mesa (lo más importante)

Los brazos del G1 son cortos y la D435i de la cabeza ve hacia abajo a 47.6° con 42.6° de campo vertical.
La zona donde el robot **ve y alcanza a la vez** es pequeña:

- **Superficie de la mesa a la altura del pelvis (±3 cm).** Con la mesa 9 cm más baja, la zona se reduce a la
  quinta parte.
- Objetos entre **28 y 40 cm delante del pelvis** y a menos de **±20 cm** a los lados: una franja de unos
  12 × 40 cm.
- Objetos de 4 a 9 cm de ancho y **8 cm de alto o más** (lata, vaso, caja de jugo): el agarre es lateral y la
  Revo2 mide ~9 cm de canto. Recipiente de borde bajo (≤1.5 cm, tipo charola) al centro, porque las dos manos
  llegan ahí.
- La mano va del lado exterior del objeto con los dedos hacia adelante: deja **8-10 cm libres** a ese lado.
  En la franja caben la charola y un objeto por lado.
- La mano derecha trabaja del lado derecho de la imagen y la izquierda del izquierdo.

### 2. Jetson

```bash
# la cámara la tienen ocupada teleimager y el video_hub de Unitree
sudo systemctl stop teleimager
conda activate tv && python ~/svc_switch.py video_hub_pc4 off    # no persiste tras reiniciar
pip install pyrealsense2                                          # hay wheel aarch64 cp310
python cam_server.py                                              # copia el archivo con scp
# manos: brainco_hand.service corriendo; reinícialo cuando las dos manos estén en verde fijo
```

### 3. Robot en modo debug, colgado

Igual que para el teleop sin `--motion`: robot en el arnés y en modo debug. Al conectarse,
`robot_real.py` sostiene **todas** las articulaciones en la posición en que estén (piernas con kp 300). No usa
`G1_29_ArmController` del teleop, porque esa clase manda los brazos de golpe a la pose cero al crearse.

### 4. Puesta en marcha, en este orden

```bash
P="PYTHONNOUSERSITE=1 .venv/bin/python"
$P check_real.py cam  --cam-host <IP Jetson>                 # solo lectura: mesa vacía => z igual en todos los puntos
$P check_real.py hand --side right                           # abrir (pulgar arriba) / cerrar / abrir
$P check_real.py arm  --side right --table-z <z de 'cam'>    # pose de espera, flotar sobre la mesa, cruz del TCP
```

`check_real.py arm` guarda una foto con una cruz donde la cinemática cree que está el centro del agarre. Si no
cae frente a la palma, ajusta en `hand_model.py` el montaje (`MOUNT_X`, grosor del adaptador) o el centro del
agarre (`TCP_ALONG`, `OBJECT_RADIUS`). `CLOSED` (fuerza del cierre) también se afina con la mano real.

### 5. Fable

```bash
$P agent.py --backend real --iface enx0c37969f434c --cam-host <IP Jetson> --view
```

- Velocidad articular por defecto de 0.4 rad/s (`--speed`). El hilo que publica limita además a 1.0 rad/s.
- Caja de seguridad para el TCP en `skills.WORKSPACE`.
- **Ctrl+C durante un movimiento** deja los brazos quietos donde están y le avisa a Fable, sin salir del
  programa. Otro Ctrl+C en el prompt sí sale. Al salir se deja de publicar `rt/lowcmd`: deja el robot en
  `home` y ten el control a la mano.
- Notas de hardware: según las notas de BFM-Zero, el roll del hombro derecho llegaba al tope de torque. Si la
  mano derecha no llega, usa la izquierda.

## Costos y opciones

Fable 5.2 cuesta $8 / $40 por MTok (caché $0.20). Cada foto de 640×480 son unos 400 tokens. El agente
imprime el costo de cada llamada y el total de la sesión.

- `--effort` (por defecto `high`): `medium` contesta más rápido y más barato; `xhigh` para tareas largas.
- `--fast`: genera tokens unas 2.5 veces más rápido a precio doble.
- Fallback por rechazo activado (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`): si un
  clasificador declina, la API reintenta con otro modelo y el agente avisa. Quítalo en `agent.py` si no lo
  quieres.

## Pendientes

- Medir en el robot real: inclinación real de la cámara (`check_real.py cam`), `TCP_OFFSET` y cierre de la Revo2.
- Inclinar la cintura hacia adelante (waist pitch) ampliaría mucho la zona útil. La cinemática ya trabaja en el
  marco del torso; faltaría usar el ángulo de cintura medido para "arriba" y para la superficie de apoyo.
- Ajedrez (v2): tablero con AprilTags y homografía en vez de que Fable señale casillas, más python-chess para
  validar jugadas.
