"""SAM2 point-prompt segmentation as a small service, so the model stays loaded and runs in its own
process (on the G1's Jetson: CPU, pinned away from the camera guard's cores and the control loop).

    python segment.py serve [--device cpu|cuda] [--port 55610]      # needs torch + sam2
    SegClient("127.0.0.1").mask(rgb, u, v) -> (bool mask HxW, score) or (None, 0.0)

The client side only needs numpy, OpenCV and pyzmq.
"""
import argparse
import json
import os
import time
import numpy as np
import cv2
import zmq

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "models", "sam2.1_hiera_tiny.pt")
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
PORT = 55610


class SegClient:
    def __init__(self, host="127.0.0.1", port=PORT, timeout_ms=15000):
        self.addr, self.timeout_ms = f"tcp://{host}:{port}", timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def ping(self, timeout_ms=2000):
        """True if a server answers (it replies at once, without running the model)."""
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.addr)
        try:
            sock.send_multipart([b'{"ping": true}', b""])
            sock.recv_multipart()
            return True
        except zmq.Again:
            return False
        finally:
            sock.close()

    def mask(self, rgb, u, v):
        """Mask of the object under pixel (u, v); (None, 0.0) if the server does not answer."""
        jpg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        try:
            self.sock.send_multipart([json.dumps(dict(u=int(u), v=int(v))).encode(), jpg])
            meta, png = self.sock.recv_multipart()
        except zmq.Again:
            self.sock.close()           # a REQ socket is stuck after a timeout: rebuild it
            self._connect()
            return None, 0.0
        meta = json.loads(meta)
        if not png:
            return None, 0.0
        return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE) > 0, float(meta["score"])


def serve(device, port, threads):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if device == "cpu":
        torch.set_num_threads(threads)
    predictor = SAM2ImagePredictor(build_sam2(CONFIG, CHECKPOINT, device=device))
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{port}")
    print(f"SAM2 on {device}, serving on tcp://0.0.0.0:{port}", flush=True)
    while True:
        meta, jpg = sock.recv_multipart()
        req = json.loads(meta)
        if req.get("ping"):
            sock.send_multipart([b'{"score": 0}', b""])
            continue
        rgb = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        t0 = time.time()
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictor.set_image(rgb)
                    masks, scores, _ = predictor.predict(point_coords=np.array([[req["u"], req["v"]]], np.float32),
                                                         point_labels=np.array([1]), multimask_output=True)
            else:
                predictor.set_image(rgb)
                masks, scores, _ = predictor.predict(point_coords=np.array([[req["u"], req["v"]]], np.float32),
                                                     point_labels=np.array([1]), multimask_output=True)
        # Highest-scoring mask that contains the clicked pixel and is not most of the image.
        best, best_score = None, 0.0
        for m, s in sorted(zip(masks, scores), key=lambda x: -x[1]):
            m = m.astype(bool)
            if m[min(req["v"], m.shape[0] - 1), min(req["u"], m.shape[1] - 1)] and m.mean() < 0.4:
                best, best_score = m, float(s)
                break
        png = cv2.imencode(".png", best.astype(np.uint8) * 255)[1].tobytes() if best is not None else b""
        sock.send_multipart([json.dumps(dict(score=best_score, seconds=time.time() - t0)).encode(), png])
        print(f"mask at ({req['u']},{req['v']}): score {best_score:.2f}, {time.time() - t0:.2f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve"])
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--threads", type=int, default=4, help="cpu: torch threads")
    a = ap.parse_args()
    serve(a.device, a.port, a.threads)
