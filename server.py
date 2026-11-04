"""
server.py - Solder joint detection API + web app (lightweight ONNX version)

Runs weights.onnx with onnxruntime (no PyTorch / Ultralytics needed), so it fits
on small hosts like Render's free tier (512 MB RAM) as well as Codespaces or your PC.

Create weights.onnx once on your PC (in the venv where app.py works):
    python app.py --export --imgsz 640
and commit weights.onnx to the repo next to this file.

Endpoints
    GET  /              web app (index.html)
    POST /predict       raw JPEG/PNG in the request body -> JSON detections
                        query params: conf (default 0.4), iou (0.5),
                                      source=esp32 to also store it as the latest ESP32 frame
    GET  /latest        JSON result of the most recent ESP32 frame
    GET  /latest.jpg    the most recent ESP32 frame
    GET  /health        model info

Run
    uvicorn server:app --host 0.0.0.0 --port 8000        (PC / Codespaces)
    uvicorn server:app --host 0.0.0.0 --port $PORT       (Render)

Test from a terminal
    curl -X POST --data-binary @board.jpg -H "Content-Type: image/jpeg" \
         "http://localhost:8000/predict?conf=0.4"

Optional environment variable: WEIGHTS=path/to/model.onnx
"""

import ast
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

BASE = Path(__file__).resolve().parent
WEIGHTS = os.environ.get("WEIGHTS", str(BASE / "weights.onnx"))
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per image

if not Path(WEIGHTS).exists():
    raise SystemExit(f"Model not found: {WEIGHTS}\n"
                     "Create it on your PC with:  python app.py --export --imgsz 640")

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
_opts = ort.SessionOptions()
_opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
session = ort.InferenceSession(WEIGHTS, sess_options=_opts, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
_shape = session.get_inputs()[0].shape            # [1, 3, H, W]
IMG_H = _shape[2] if isinstance(_shape[2], int) else 640
IMG_W = _shape[3] if isinstance(_shape[3], int) else 640

# Ultralytics stores class names in the ONNX metadata, e.g. "{0: 'Bad', 1: 'Good'}"
_meta = session.get_modelmeta().custom_metadata_map
try:
    NAMES = {int(k): str(v) for k, v in ast.literal_eval(_meta.get("names", "{}")).items()}
except (ValueError, SyntaxError):
    NAMES = {}
model_lock = threading.Lock()


def letterbox(img):
    """Resize keeping aspect ratio and pad to (IMG_H, IMG_W) with gray, like Ultralytics."""
    h, w = img.shape[:2]
    r = min(IMG_H / h, IMG_W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (IMG_H - nh) // 2
    left = (IMG_W - nw) // 2
    canvas = np.full((IMG_H, IMG_W, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def run_model(img, conf, iou):
    """Returns list of (x0, y0, x1, y1, class_id, score) in original image pixels."""
    h, w = img.shape[:2]
    lb, r, pad_x, pad_y = letterbox(img)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])          # 1x3xHxW

    with model_lock:
        out = session.run(None, {INPUT_NAME: x})[0]                # 1 x (4+nc) x N
    pred = out[0].T                                                # N x (4+nc)
    if pred.shape[1] < 5:
        return []

    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(pred)), cls_ids]
    keep = scores >= conf
    if not keep.any():
        return []
    pred, cls_ids, scores = pred[keep], cls_ids[keep], scores[keep]

    # cx, cy, w, h (letterboxed pixels) -> x0, y0, x1, y1 (original pixels)
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x0 = np.clip((cx - bw / 2 - pad_x) / r, 0, w)
    y0 = np.clip((cy - bh / 2 - pad_y) / r, 0, h)
    x1 = np.clip((cx + bw / 2 - pad_x) / r, 0, w)
    y1 = np.clip((cy + bh / 2 - pad_y) / r, 0, h)

    # Per-class non-max suppression (shift boxes by class so classes don't suppress each other)
    offset = cls_ids.astype(np.float32) * (max(w, h) + 1)
    rects = np.stack([x0 + offset, y0 + offset, x1 - x0, y1 - y0], axis=1).tolist()
    idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), conf, iou)
    idx = np.array(idx).flatten() if len(idx) else []
    return [(float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i]), int(cls_ids[i]), float(scores[i]))
            for i in idx]


def detect(image_bytes: bytes, conf: float, iou: float) -> dict:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Request body is not a valid JPEG/PNG image.")
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    dets = run_model(img, conf, iou)
    ms = (time.perf_counter() - t0) * 1000

    counts = {name: 0 for name in NAMES.values()}
    boxes = []
    for x0, y0, x1, y1, c, s in sorted(dets, key=lambda d: -d[5]):
        name = NAMES.get(c, f"class_{c}")
        counts[name] = counts.get(name, 0) + 1
        boxes.append({"class": name, "conf": round(s, 3),
                      "x0": round(x0), "y0": round(y0), "x1": round(x1), "y1": round(y1)})

    defects = sum(n for k, n in counts.items() if "good" not in k.lower())
    status = "DEFECTS" if defects else ("OK" if boxes else "NO_JOINTS")
    return {"status": status, "defects": defects, "counts": counts, "boxes": boxes,
            "width": w, "height": h, "inference_ms": round(ms, 1)}


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
latest = {"id": 0, "time": None, "jpeg": None, "result": None}
latest_lock = threading.Lock()

app = FastAPI(title="Solder Joint Inspector")


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "weights": Path(WEIGHTS).name, "classes": list(NAMES.values()),
            "input_size": [IMG_H, IMG_W]}


@app.post("/predict")
async def predict(request: Request,
                  conf: float = Query(0.4, ge=0.01, le=0.99),
                  iou: float = Query(0.5, ge=0.05, le=0.95),
                  source: str = Query("browser")):
    body = await request.body()
    if not body:
        raise HTTPException(400, "Send the image as the raw request body (Content-Type: image/jpeg).")
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")

    result = await run_in_threadpool(detect, body, conf, iou)

    if source == "esp32":
        with latest_lock:
            latest["id"] += 1
            latest["time"] = time.time()
            latest["jpeg"] = body
            latest["result"] = result
            result["id"] = latest["id"]
        print(f"[ESP32 #{result['id']}] {result['status']} {result['counts']} "
              f"({result['inference_ms']} ms)", flush=True)
    return result


@app.get("/latest")
def get_latest():
    with latest_lock:
        if not latest["id"]:
            return {"id": 0, "result": None, "age_s": None}
        return {"id": latest["id"], "result": latest["result"],
                "age_s": round(time.time() - latest["time"], 1)}


@app.get("/latest.jpg")
def get_latest_jpg():
    with latest_lock:
        jpeg = latest["jpeg"]
    if jpeg is None:
        return JSONResponse({"error": "No ESP32 frame received yet."}, status_code=404)
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
