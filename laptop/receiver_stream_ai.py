from flask import Flask, request, jsonify, Response
from pathlib import Path
from datetime import datetime
from collections import deque
import threading
import time
import csv
import json

import cv2
import numpy as np

app = Flask(__name__)

SHARED_TOKEN = "sws3009_token"
AI_MODEL_NAME = "OpenCV YuNet ONNX neural face detector"

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx"

RAW_DIR = REPO_ROOT / "stream_received"
ANN_DIR = REPO_ROOT / "stream_annotated"
LOG_DIR = REPO_ROOT / "logs"

RAW_DIR.mkdir(exist_ok=True)
ANN_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

CSV_PATH = LOG_DIR / "stream_detection_log.csv"
SUMMARY_PATH = LOG_DIR / "runtime_summary.json"

SAVE_EVERY_N = 30
ROLLING_WINDOW = 50

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YuNet model not found: {MODEL_PATH}")

if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "server_timestamp",
            "ai_model",
            "device",
            "frame_id",
            "image_size_bytes",
            "width",
            "height",
            "num_faces",
            "server_inference_ms",
            "client_capture_ms",
            "client_encode_ms",
            "fps_est",
            "detections",
            "annotated_file"
        ])

state_lock = threading.Lock()

latest_annotated_jpeg = None
last_frame_time = None
yunet_detector = None
yunet_input_size = None

recent_fps = deque(maxlen=ROLLING_WINDOW)
recent_ai_ms = deque(maxlen=ROLLING_WINDOW)
recent_size = deque(maxlen=ROLLING_WINDOW)
recent_faces = deque(maxlen=ROLLING_WINDOW)

latest_stats = {
    "ok": True,
    "status": "waiting for frames",
    "security": "token_required",
    "ai_model": AI_MODEL_NAME,
    "device": "",
    "frame_id": "",
    "num_faces": 0,
    "server_inference_ms": "",
    "width": 640,
    "height": 360,
    "image_size_bytes": 0,
    "fps_est": 0.0,
    "avg_fps": 0.0,
    "avg_server_ai_ms": 0.0,
    "avg_image_size_kb": 0.0,
    "face_detection_rate": 0.0,
    "last_update": "",
    "detections": []
}


def safe_avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def get_yunet_detector(width, height):
    global yunet_detector, yunet_input_size

    input_size = (int(width), int(height))

    if yunet_detector is None:
        yunet_detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            input_size,
            0.75,
            0.30,
            5000
        )
        yunet_input_size = input_size

    elif yunet_input_size != input_size:
        yunet_detector.setInputSize(input_size)
        yunet_input_size = input_size

    return yunet_detector


def detect_faces_yunet(img_bgr):
    height, width = img_bgr.shape[:2]
    detector = get_yunet_detector(width, height)

    _, faces = detector.detect(img_bgr)

    detections = []

    if faces is None:
        return detections

    for idx, face in enumerate(faces):
        x, y, w, h = face[:4]
        score = face[-1]

        detections.append({
            "id": int(idx + 1),
            "type": "face",
            "x": int(round(x)),
            "y": int(round(y)),
            "w": int(round(w)),
            "h": int(round(h)),
            "score": round(float(score), 3)
        })

    return detections


@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pi-Laptop Edge AI Video Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 24px;
            background: #111;
            color: #eee;
        }

        h1 { margin-bottom: 8px; }

        .subtitle {
            color: #66ff99;
            margin-bottom: 20px;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(640px, 1fr) 520px;
            gap: 24px;
            align-items: start;
        }

        .card {
            background: #1d1d1d;
            padding: 16px;
            border-radius: 12px;
            box-sizing: border-box;
        }

        .video-box {
            width: 100%;
            aspect-ratio: 16 / 9;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #444;
        }

        .video-box img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .stats-card {
            height: auto; min-height: 0; overflow: visible;
        }

        pre {
            font-size: 14px; line-height: 1.45;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .badge {
            color: #66ff99;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Pi-Laptop Edge AI Video Stream</h1>
    <p class="subtitle">Raspberry Pi video stream → Laptop neural-network AI inference → analysis result</p>

    <div class="layout">
        <div class="card">
            <h2>Live stream</h2>
            <div class="video-box">
                <img src="/video" />
            </div>
        </div>

        <div class="card stats-card">
            <h2>Runtime stats</h2>
            <pre id="stats">waiting for frames...</pre>
        </div>
    </div>

<script>
function formatStats(j) {
    const sizeKB = j.image_size_bytes ? (j.image_size_bytes / 1024).toFixed(1) : "0.0";

    let detText = "none";
    if (j.detections && j.detections.length > 0) {
        detText = j.detections.map(d =>
            `Face #${d.id}: score=${d.score}, x=${d.x}, y=${d.y}, w=${d.w}, h=${d.h}`
        ).join("\\n");
    }

    return [
        `status                 : ${j.status}`,
        `security               : ${j.security}`,
        `ai_model               : ${j.ai_model}`,
        `device                 : ${j.device}`,
        `frame_id               : ${j.frame_id}`,
        `resolution             : ${j.width} x ${j.height}`,
        `image_size             : ${sizeKB} KB`,
        `faces                  : ${j.num_faces}`,
        ``,
        `current fps_est        : ${j.fps_est}`,
        `rolling avg_fps        : ${j.avg_fps}`,
        `server_ai_ms           : ${j.server_inference_ms}`,
        `rolling avg_ai_ms      : ${j.avg_server_ai_ms}`,
        `rolling avg_size_kb    : ${j.avg_image_size_kb}`,
        `face_detection_rate    : ${j.face_detection_rate}`,
        ``,
        `client_capture_ms      : ${j.client_capture_ms}`,
        `client_encode_ms       : ${j.client_encode_ms}`,
        `last_update            : ${j.last_update}`,
        ``,
        `detections:`,
        `${detText}`
    ].join("\\n");
}

async function refreshStats() {
    try {
        const r = await fetch('/stats');
        const j = await r.json();
        document.getElementById('stats').textContent = formatStats(j);
    } catch (e) {
        document.getElementById('stats').textContent = 'failed to fetch stats: ' + e;
    }
}

setInterval(refreshStats, 500);
refreshStats();
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "status": "running",
        "ai_model": AI_MODEL_NAME,
        "time": datetime.now().isoformat(timespec="seconds")
    })


@app.get("/stats")
def stats():
    with state_lock:
        return jsonify(latest_stats)


@app.get("/video")
def video():
    def generate():
        while True:
            with state_lock:
                frame = latest_annotated_jpeg

            if frame is None:
                time.sleep(0.1)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/upload_frame")
def upload_frame():
    global latest_annotated_jpeg, latest_stats, last_frame_time

    server_start = time.time()

    token = request.form.get("token", "")
    if token != SHARED_TOKEN:
        return jsonify({
            "ok": False,
            "error": "unauthorized: invalid token"
        }), 403

    if "image" not in request.files:
        return jsonify({
            "ok": False,
            "error": "no image field"
        }), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    device = request.form.get("device", "unknown")
    frame_id = request.form.get("frame_id", "unknown")
    client_capture_ms = request.form.get("capture_ms", "")
    client_encode_ms = request.form.get("encode_ms", "")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    np_buf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({
            "ok": False,
            "error": "failed to decode image"
        }), 400

    height, width = img.shape[:2]

    detections = detect_faces_yunet(img)

    for det in detections:
        x = det["x"]
        y = det["y"]
        w = det["w"]
        h = det["h"]
        face_id = det["id"]
        score = det["score"]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            img,
            f"YuNet Face #{face_id} {score:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2
        )

    ok, ann_buf = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )

    if not ok:
        return jsonify({
            "ok": False,
            "error": "failed to encode annotated image"
        }), 500

    annotated_jpeg = ann_buf.tobytes()

    try:
        frame_id_int = int(frame_id)
    except Exception:
        frame_id_int = -1

    annotated_file = ""

    should_save = (
        frame_id_int < 0 or
        frame_id_int % SAVE_EVERY_N == 0 or
        len(detections) > 0
    )

    if should_save:
        raw_path = RAW_DIR / f"raw_{ts}_f{frame_id}.jpg"
        ann_path = ANN_DIR / f"annotated_yunet_{ts}_f{frame_id}.jpg"
        raw_path.write_bytes(image_bytes)
        ann_path.write_bytes(annotated_jpeg)
        annotated_file = str(ann_path)

    now = time.time()
    if last_frame_time is None:
        fps_est = 0.0
    else:
        dt = now - last_frame_time
        fps_est = round(1.0 / dt, 2) if dt > 0 else 0.0
    last_frame_time = now

    server_inference_ms = round((time.time() - server_start) * 1000, 2)

    recent_fps.append(fps_est)
    recent_ai_ms.append(server_inference_ms)
    recent_size.append(len(image_bytes) / 1024)
    recent_faces.append(1 if len(detections) > 0 else 0)

    avg_fps = safe_avg(recent_fps)
    avg_ai_ms = safe_avg(recent_ai_ms)
    avg_size_kb = safe_avg(recent_size)
    face_detection_rate = safe_avg(recent_faces)

    with state_lock:
        latest_annotated_jpeg = annotated_jpeg
        latest_stats = {
            "ok": True,
            "status": "receiving",
            "security": "token_required",
            "ai_model": AI_MODEL_NAME,
            "device": device,
            "frame_id": frame_id,
            "num_faces": len(detections),
            "detections": detections,
            "server_inference_ms": server_inference_ms,
            "width": width,
            "height": height,
            "image_size_bytes": len(image_bytes),
            "fps_est": fps_est,
            "avg_fps": avg_fps,
            "avg_server_ai_ms": avg_ai_ms,
            "avg_image_size_kb": avg_size_kb,
            "face_detection_rate": face_detection_rate,
            "client_capture_ms": client_capture_ms,
            "client_encode_ms": client_encode_ms,
            "last_update": ts,
            "annotated_file": annotated_file
        }

    SUMMARY_PATH.write_text(
        json.dumps(latest_stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            AI_MODEL_NAME,
            device,
            frame_id,
            len(image_bytes),
            width,
            height,
            len(detections),
            server_inference_ms,
            client_capture_ms,
            client_encode_ms,
            fps_est,
            detections,
            annotated_file
        ])

    print(
        f"[{ts}] frame={frame_id}, "
        f"faces={len(detections)}, "
        f"fps={fps_est}, "
        f"avg_fps={avg_fps}, "
        f"ai={server_inference_ms}ms, "
        f"avg_ai={avg_ai_ms}ms"
    )

    return jsonify({
        "ok": True,
        "timestamp": ts,
        "ai_model": AI_MODEL_NAME,
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "image_size_bytes": len(image_bytes),
        "num_faces": len(detections),
        "detections": detections,
        "server_inference_ms": server_inference_ms,
        "fps_est": fps_est,
        "avg_fps": avg_fps,
        "avg_server_ai_ms": avg_ai_ms,
        "avg_image_size_kb": avg_size_kb,
        "face_detection_rate": face_detection_rate,
        "annotated_file": annotated_file
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
