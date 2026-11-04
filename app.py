"""
app.py - Local solder joint detector (YOLOv11 weights from Roboflow)

Runs your trained weights.pt fully offline on your PC: live webcam, a single
image, a folder of images, or a video file. Boxes are drawn per joint with the
class name and confidence, plus a Good/Bad count for the frame.

SETUP (in your Python 3.12 venv)
    pip install -r requirements.txt

RUN
    python app.py                         # webcam 0, weights.pt in this folder
    python app.py --source 1              # second camera (e.g. USB webcam)
    python app.py --source board.jpg      # single image
    python app.py --source photos/        # every image in a folder
    python app.py --source clip.mp4       # video file
    python app.py --conf 0.4 --imgsz 640  # change threshold / input size

KEYS (in the video window)
    q / Esc   quit
    s         save a snapshot (with boxes) to snapshots/
    + / -     raise / lower confidence threshold
    p         pause / resume
    n         next image (folder mode)

EXPORT FOR ESP32 (later)
    python app.py --export --imgsz 320
    -> writes weights.onnx. For ESP32-S3 / ESP32-P4 this ONNX file is the input
       to Espressif's ESP-DL toolchain (esp-ppq), which quantizes it to int8
       and produces the .espdl model the firmware loads. A smaller --imgsz
       (e.g. 320 or 256) is much faster on the ESP32 but should match what you
       test here, so also try:  python app.py --imgsz 320
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
WINDOW = "Solder Joint Inspector"

# BGR colours: names containing "good" are green, everything else red/orange.
GOOD_COLOR = (60, 180, 75)
BAD_COLORS = [(40, 40, 230), (0, 140, 255), (200, 0, 160), (0, 200, 255), (160, 60, 0)]


def parse_args():
    p = argparse.ArgumentParser(description="Solder joint detector")
    p.add_argument("--weights", default="weights.pt", help="Path to the trained .pt file")
    p.add_argument("--source", default="0",
                   help="Webcam index (0, 1, ...), image, folder of images, or video file")
    p.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (0-1)")
    p.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size (training used 640)")
    p.add_argument("--cam-width", type=int, default=1280, help="Requested webcam width")
    p.add_argument("--cam-height", type=int, default=720, help="Requested webcam height")
    p.add_argument("--device", default=None, help="'cpu', '0' for first GPU, or leave empty")
    p.add_argument("--export", action="store_true", help="Export weights to ONNX and exit")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def class_colors(names):
    colors, i = {}, 0
    for idx, name in names.items():
        if "good" in name.lower():
            colors[idx] = GOOD_COLOR
        else:
            colors[idx] = BAD_COLORS[i % len(BAD_COLORS)]
            i += 1
    return colors


def draw(frame, result, names, colors, conf_thr, fps=None):
    h, w = frame.shape[:2]
    scale = max(0.5, min(w, h) / 900)
    thick = max(1, int(round(2 * scale)))
    counts = {n: 0 for n in names.values()}

    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        cf = boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), c, s in zip(xyxy, cls, cf):
            name = names[c]
            counts[name] += 1
            color = colors[c]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, thick)
            label = f"{name} {s:.0%}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, thick)
            ty = max(y0, th + base + 2)
            cv2.rectangle(frame, (x0, ty - th - base - 2), (x0 + tw + 4, ty), color, -1)
            cv2.putText(frame, label, (x0 + 2, ty - base), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale, (255, 255, 255), thick, cv2.LINE_AA)

    # Summary panel
    bad_total = sum(n for k, n in counts.items() if "good" not in k.lower())
    lines = [f"{k}: {n}" for k, n in counts.items()]
    lines.append(f"conf >= {conf_thr:.2f}")
    if fps is not None:
        lines.append(f"{fps:.1f} FPS")
    status = "DEFECTS FOUND" if bad_total else ("OK" if sum(counts.values()) else "no joints")
    status_color = (40, 40, 230) if bad_total else (GOOD_COLOR if sum(counts.values()) else (160, 160, 160))

    fs = 0.6 * scale
    line_h = int(26 * scale)
    panel_w = int(230 * scale)
    panel_h = line_h * (len(lines) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, status, (16, 8 + line_h), cv2.FONT_HERSHEY_SIMPLEX, fs,
                status_color, thick + 1, cv2.LINE_AA)
    for i, text in enumerate(lines, start=2):
        cv2.putText(frame, text, (16, 8 + line_h * i), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return frame, counts


def save_snapshot(frame):
    out = Path("snapshots")
    out.mkdir(exist_ok=True)
    path = out / time.strftime("snap_%Y%m%d_%H%M%S.jpg")
    cv2.imwrite(str(path), frame)
    print(f"Saved {path}")


def handle_key(key, state):
    """Returns 'quit', 'next', or None. Mutates state (conf, paused)."""
    if key in (ord("q"), 27):
        return "quit"
    if key in (ord("+"), ord("=")):
        state["conf"] = min(0.95, round(state["conf"] + 0.05, 2))
    elif key in (ord("-"), ord("_")):
        state["conf"] = max(0.05, round(state["conf"] - 0.05, 2))
    elif key == ord("p"):
        state["paused"] = not state["paused"]
    elif key == ord("s"):
        state["save"] = True
    elif key == ord("n"):
        return "next"
    return None


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def open_camera(index, width, height):
    # CAP_DSHOW opens much faster on Windows; fall back to default elsewhere.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else None
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try --source 1 or close other apps using it.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run_stream(model, cap, args, names, colors):
    state = {"conf": args.conf, "paused": False, "save": False}
    fps, last = None, time.time()
    shown = None
    while True:
        if not state["paused"] or shown is None:
            ok, frame = cap.read()
            if not ok:
                print("End of stream.")
                break
            result = model.predict(frame, conf=state["conf"], iou=args.iou, imgsz=args.imgsz,
                                   device=args.device, verbose=False)[0]
            now = time.time()
            inst = 1.0 / max(now - last, 1e-6)
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            last = now
            shown, _ = draw(frame, result, names, colors, state["conf"], fps)
        cv2.imshow(WINDOW, shown)
        if state["save"]:
            save_snapshot(shown)
            state["save"] = False
        if handle_key(cv2.waitKey(1) & 0xFF, state) == "quit":
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()


def run_images(model, paths, args, names, colors):
    state = {"conf": args.conf, "paused": False, "save": False}
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Skipping unreadable file: {path}")
            continue
        current_conf = None
        while True:
            if current_conf != state["conf"]:  # re-run when threshold changes
                current_conf = state["conf"]
                result = model.predict(img, conf=current_conf, iou=args.iou, imgsz=args.imgsz,
                                       device=args.device, verbose=False)[0]
                shown, counts = draw(img.copy(), result, names, colors, current_conf)
                print(f"{path.name}: {counts}")
                cv2.setWindowTitle(WINDOW, f"{WINDOW} - {path.name}")
            cv2.imshow(WINDOW, shown)
            if state["save"]:
                save_snapshot(shown)
                state["save"] = False
            action = handle_key(cv2.waitKey(30) & 0xFF, state)
            if action == "quit" or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            if action == "next":
                break
    print("No more images. Press any key in the window to close.")
    cv2.waitKey(0)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights.resolve()}\n"
                         "Put weights.pt next to app.py or pass --weights <path>.")

    model = YOLO(str(weights))
    names = model.names  # {0: 'Bad', 1: 'Good'} etc.
    print(f"Loaded {weights.name}  classes: {list(names.values())}")

    if args.export:
        out = model.export(format="onnx", imgsz=args.imgsz, opset=13, simplify=True)
        print(f"\nExported: {out}\nNext step for ESP32: quantize this ONNX with ESP-DL's esp-ppq "
              f"using ~100 of your training images for calibration.")
        return

    colors = class_colors(names)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    src = args.source

    if src.isdigit():
        cap = open_camera(int(src), args.cam_width, args.cam_height)
        print("Webcam running. Keys: q quit | s snapshot | +/- confidence | p pause")
        run_stream(model, cap, args, names, colors)
    else:
        path = Path(src)
        if path.is_dir():
            files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMG_EXTS)
            if not files:
                raise SystemExit(f"No images found in {path}")
            print(f"{len(files)} images. Keys: n next | q quit | s snapshot | +/- confidence")
            run_images(model, files, args, names, colors)
        elif path.suffix.lower() in IMG_EXTS:
            run_images(model, [path], args, names, colors)
        elif path.suffix.lower() in VID_EXTS:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise SystemExit(f"Could not open video: {path}")
            run_stream(model, cap, args, names, colors)
        else:
            raise SystemExit(f"Unrecognised source: {src}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
