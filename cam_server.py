"""Head D435i server for humanoidHarness: color + depth aligned to color + intrinsics over ZMQ REQ/REP.
Runs on the G1 Jetson. Reply to b"get": [meta json, color JPEG, depth uint16 PNG in mm].

Two sources:
  --source realsense  pyrealsense2 owns the camera (stop teleimager / Unitree's video_hub first).
  --source rpc        the robot's rpc-camera-guard owns the camera and serves colour on ZMQ 55555 ("mono"
                      mode). Depth is read raw from the depth node with v4l2-ctl, listen-only, like
                      rpc_bundle/teleop_sonic/depth_rec.py, and registered to colour with that bundle's
                      calibration (d435i_color_from_depth.npy). Nothing of theirs is stopped or changed.
                      Only valid while the guard runs in "mono" mode (its "color" mode uses the depth node).

    python cam_server.py --source rpc          # this lab's G1
    python cam_server.py --source realsense    # pip install pyrealsense2 (aarch64 cp310 wheel on PyPI)
"""
import argparse
import glob
import json
import os
import subprocess
import threading
import time
import cv2
import numpy as np
import zmq

RPC_CALIB = os.path.expanduser("~/rpc_bundle/teleop_sonic/d435i_color_from_depth.npy")


def find_depth_node():
    """/dev/videoN of the head D435i depth stream (USB interface :1.0, node index 0); numbers move after resets."""
    for v in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            dev = os.path.basename(os.path.realpath(os.path.join(v, "device")))
            idx = int(open(os.path.join(v, "index")).read())
            name = open(os.path.join(v, "name")).read()
        except (OSError, ValueError):
            continue
        if "RealSense" in name and dev.endswith(":1.0") and idx == 0:
            return "/dev/" + os.path.basename(v)
    return "/dev/video0"


# Depth -> colour offset along x (m). The bundle's fit gives -28.5 mm: on its nearly flat calibration scene tx
# trades off against cx, and on a 0.8 m object that put the depth 31 px (~4 cm) left of the colour. The D435i RGB
# sensor sits ~15 mm from the left imager; fitting a standing cylinder's depth to its colour silhouette gave
# +15..+20 mm (2026-11-12). None = the bundle's value.
DEPTH_TX = 0.016


class RpcSource:
    """Colour from the guard's ZMQ stream, raw Z16 depth registered into the colour image."""

    def __init__(self, zmq_port=55555, fps=15, fx_scale=1.0, tx=DEPTH_TX):
        fxc, cxc, cyc, tx_bundle, ty, fxd = [float(v) for v in np.load(RPC_CALIB)]
        tx = tx_bundle if tx is None else tx
        # The bundle's fit fixes the colour/depth focal *ratio* well; fx_scale corrects the absolute scale of
        # both (measured with a ruler-checked marker: check_real.py fxcal). 1.0 = their calibration as is.
        fxc, fxd = fxc * fx_scale, fxd * fx_scale
        self.meta = dict(w=640, h=480, fx=fxc, fy=fxc, cx=cxc, cy=cyc, source="rpc", fx_scale=fx_scale, depth_tx=tx)
        v, u = np.mgrid[0:480, 0:640].astype(np.float32)
        self._xn, self._yn = (u - 320) / fxd, (v - 240) / fxd     # depth pixel -> normalised ray (depth camera)
        self._fxc, self._cxc, self._cyc, self._tx, self._ty = fxc, cxc, cyc, tx, ty
        self.sub = zmq.Context.instance().socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.connect(f"tcp://127.0.0.1:{zmq_port}")
        self.fps, self.depth_mm, self.lock = fps, None, threading.Lock()
        threading.Thread(target=self._depth_loop, daemon=True).start()

    def _depth_loop(self):
        n = 640 * 480 * 2
        while True:
            node = find_depth_node()
            proc = subprocess.Popen(["v4l2-ctl", "-d", node, "--set-fmt-video=width=640,height=480,pixelformat=Z16 ",
                                     f"--set-parm={self.fps}", "--stream-mmap", "--stream-count=0", "--stream-to=-"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            # v4l2-ctl writes "Frame rate set to 15.000 fps\n" to stdout before the first frame: skip it,
            # or every frame is read 29 bytes off (an odd offset that scrambles the 16-bit depth).
            if proc.stdout.peek(16).startswith(b"Frame rate"):
                proc.stdout.readline()
            buf = bytearray()
            while True:
                chunk = proc.stdout.read(n - len(buf))
                if not chunk:
                    break
                buf += chunk
                if len(buf) == n:
                    raw = np.frombuffer(bytes(buf), np.uint16).reshape(480, 640)
                    with self.lock:
                        self.depth_mm = raw
                    buf = bytearray()
            proc.kill()
            print("depth stream ended, reopening")
            time.sleep(1.0)

    def _register(self, raw):
        """Depth (mm, depth camera) -> depth in the colour image (mm), nearest surface wins."""
        z = raw.astype(np.float32) / 1000.0
        ok = (z > 0.1) & (z < 3.0)
        Z = z[ok]
        uc = np.round(self._fxc * (self._xn[ok] * Z + self._tx) / Z + self._cxc).astype(np.int32)
        vc = np.round(self._fxc * (self._yn[ok] * Z + self._ty) / Z + self._cyc).astype(np.int32)
        inside = (uc >= 0) & (uc < 640) & (vc >= 0) & (vc < 480)
        out = np.full(480 * 640, np.inf, np.float32)
        np.minimum.at(out, vc[inside] * 640 + uc[inside], Z[inside])
        out = out.reshape(480, 640)
        out[np.isinf(out)] = 0.0
        # Close the 1-px gaps left by forward projection (depth is coarser than colour).
        filled = cv2.dilate(out, np.ones((3, 3), np.uint8))
        out = np.where(out > 0, out, filled)
        return (out * 1000).astype(np.uint16)

    def get(self):
        jpg = self.sub.recv()
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        with self.lock:
            raw = self.depth_mm
        depth = self._register(raw) if raw is not None else None
        return bgr, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=55600)
    ap.add_argument("--source", choices=["realsense", "rpc"], default="realsense")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth-tx", default=str(DEPTH_TX), help="rpc: offset x depth->color en m, o 'bundle'")
    ap.add_argument("--fx-scale", type=float, default=1.0, help="rpc: corrección de escala focal (check_real.py fxcal)")
    a = ap.parse_args()
    if a.source == "rpc":
        serve_rpc(a)
        return
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, a.fps)
    if not a.no_depth:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, a.fps)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    meta = dict(w=intr.width, h=intr.height, fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                model=str(intr.model), coeffs=list(intr.coeffs))
    scale = 0.0
    if not a.no_depth:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    temporal = rs.temporal_filter()
    print(f"color intrinsics {meta}  depth scale {scale}")

    latest = {}
    lock = threading.Lock()

    def grab():
        while True:
            frames = pipe.wait_for_frames()
            if not a.no_depth:
                frames = align.process(frames)
                d = frames.get_depth_frame()
                d = temporal.process(d)
                depth_mm = (np.asanyarray(d.get_data()).astype(np.float32) * scale * 1000).astype(np.uint16)
            else:
                depth_mm = None
            color = np.asanyarray(frames.get_color_frame().get_data()).copy()
            with lock:
                latest.update(color=color, depth=depth_mm, t=time.time())

    threading.Thread(target=grab, daemon=True).start()
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{a.port}")
    print(f"serving on tcp://0.0.0.0:{a.port}")
    while True:
        sock.recv()
        while "color" not in latest:
            time.sleep(0.01)
        with lock:
            color, depth, t = latest["color"], latest["depth"], latest["t"]
        jpg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        png = cv2.imencode(".png", depth)[1].tobytes() if depth is not None else b""
        sock.send_multipart([json.dumps(dict(meta, t=t)).encode(), jpg, png])


def serve_rpc(a):
    src = RpcSource(fps=15, fx_scale=a.fx_scale, tx=None if a.depth_tx == "bundle" else float(a.depth_tx))
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{a.port}")
    print(f"rpc source (colour ZMQ 55555 + raw depth {find_depth_node()}, fx_scale {a.fx_scale}), "
          f"serving on tcp://0.0.0.0:{a.port}")
    while True:
        sock.recv()
        bgr, depth = src.get()
        jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        png = cv2.imencode(".png", depth)[1].tobytes() if depth is not None else b""
        sock.send_multipart([json.dumps(dict(src.meta, t=time.time())).encode(), jpg, png])


if __name__ == "__main__":
    main()
