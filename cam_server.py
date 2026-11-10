"""Head D435i server for fable_pick: color + depth aligned to color + intrinsics over ZMQ REQ/REP.

Runs on the G1 Jetson. teleimager.service and Unitree's video_hub_pc4 hold the camera, so stop them
first (see README). Reply to b"get": [meta json, color JPEG, depth uint16 PNG in mm].

    pip install pyrealsense2 pyzmq opencv-python   # aarch64 cp310 wheels exist on PyPI
    python cam_server.py [--port 55600] [--no-depth]
"""
import argparse
import json
import threading
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=55600)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

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


if __name__ == "__main__":
    main()
