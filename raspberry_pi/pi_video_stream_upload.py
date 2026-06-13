import time
from datetime import datetime
from pathlib import Path
import csv

import cv2
import requests
from picamera2 import Picamera2

LAPTOP_IP = "10.253.24.73"
UPLOAD_URL = f"http://{LAPTOP_IP}:5000/upload_frame"

SHARED_TOKEN = "sws3009_token"

WIDTH = 640
HEIGHT = 360
JPEG_QUALITY = 70
TARGET_FPS = 5
COLOR_MODE = "direct"
ENABLE_ENHANCEMENT = True
ENHANCEMENT_MODE = "CLAHE_LAB_SHARPEN"

DEVICE_NAME = "raspberry_pi_4b_picamera2"
LOG_PATH = Path("client_stream_log.csv")

if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "client_timestamp",
            "frame_id",
            "size_kb",
            "capture_ms",
            "preprocess_ms",
            "encode_ms",
            "upload_ms",
            "total_ms",
            "http_status",
            "num_faces",
            "server_ai_ms",
            "fps_est",
            "enhancement_mode"
        ])


def write_log(row):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def edge_enhance_bgr(frame_bgr):
    """
    Lightweight edge-side image enhancement on Raspberry Pi A.

    Steps:
    1. Convert BGR to LAB color space.
    2. Apply CLAHE on L channel to improve local contrast.
    3. Apply mild sharpening to make facial edges clearer.
    """

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.25, blurred, -0.25, 0)

    return sharpened


def main():
    print("Pi video stream client started.")
    print(f"Upload URL: {UPLOAD_URL}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"JPEG quality: {JPEG_QUALITY}")
    print(f"Color mode: {COLOR_MODE}")
    print(f"Enhancement enabled: {ENABLE_ENHANCEMENT}")
    print(f"Enhancement mode: {ENHANCEMENT_MODE}")
    print(f"Client log: {LOG_PATH.resolve()}")
    print("Press Ctrl+C to stop.\n")

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={
            "size": (WIDTH, HEIGHT),
            "format": "RGB888"
        },
        controls={
            "FrameRate": TARGET_FPS
        }
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)

    session = requests.Session()

    frame_id = 0
    interval = 1.0 / TARGET_FPS

    try:
        while True:
            loop_start = time.time()
            frame_id += 1

            client_ts = datetime.now().isoformat(timespec="milliseconds")

            try:
                capture_start = time.time()
                frame = picam2.capture_array("main")
                capture_ms = round((time.time() - capture_start) * 1000, 2)

                preprocess_start = time.time()

                if COLOR_MODE == "swap":
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame

                if ENABLE_ENHANCEMENT:
                    frame_bgr = edge_enhance_bgr(frame_bgr)

                preprocess_ms = round((time.time() - preprocess_start) * 1000, 2)

                encode_start = time.time()

                ok, jpeg_buf = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )

                if not ok:
                    print(f"[{frame_id}] ERROR: JPEG encode failed")
                    continue

                jpeg_bytes = jpeg_buf.tobytes()
                encode_ms = round((time.time() - encode_start) * 1000, 2)
                size_kb = round(len(jpeg_bytes) / 1024, 2)

                upload_start = time.time()

                files = {
                    "image": ("frame.jpg", jpeg_bytes, "image/jpeg")
                }

                data = {
                    "token": SHARED_TOKEN,
                    "device": DEVICE_NAME,
                    "frame_id": str(frame_id),
                    "width": str(WIDTH),
                    "height": str(HEIGHT),
                    "quality": str(JPEG_QUALITY),
                    "capture_ms": str(capture_ms),
                    "preprocess_ms": str(preprocess_ms),
                    "encode_ms": str(encode_ms),
                    "enhancement_enabled": str(ENABLE_ENHANCEMENT),
                    "enhancement_mode": ENHANCEMENT_MODE,
                    "client_timestamp": client_ts
                }

                response = session.post(
                    UPLOAD_URL,
                    files=files,
                    data=data,
                    timeout=5
                )

                upload_ms = round((time.time() - upload_start) * 1000, 2)
                total_ms = round((time.time() - loop_start) * 1000, 2)

                num_faces = ""
                server_ai_ms = ""
                fps_est = ""

                if response.status_code == 200:
                    result = response.json()
                    num_faces = result.get("num_faces")
                    server_ai_ms = result.get("server_inference_ms")
                    fps_est = result.get("fps_est")

                    print(
                        f"[{frame_id}] "
                        f"size={size_kb}KB, "
                        f"capture={capture_ms}ms, "
                        f"preprocess={preprocess_ms}ms, "
                        f"encode={encode_ms}ms, "
                        f"upload={upload_ms}ms, "
                        f"server_ai={server_ai_ms}ms, "
                        f"faces={num_faces}, "
                        f"fps_est={fps_est}, "
                        f"total={total_ms}ms"
                    )
                else:
                    print(
                        f"[{frame_id}] HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )

                write_log([
                    client_ts,
                    frame_id,
                    size_kb,
                    capture_ms,
                    preprocess_ms,
                    encode_ms,
                    upload_ms,
                    total_ms,
                    response.status_code,
                    num_faces,
                    server_ai_ms,
                    fps_est,
                    ENHANCEMENT_MODE if ENABLE_ENHANCEMENT else "none"
                ])

            except requests.exceptions.RequestException as e:
                total_ms = round((time.time() - loop_start) * 1000, 2)
                print(f"[{frame_id}] NETWORK ERROR: {e}")

                write_log([
                    client_ts,
                    frame_id,
                    "",
                    "",
                    "",
                    "",
                    "",
                    total_ms,
                    "network_error",
                    "",
                    "",
                    "",
                    ENHANCEMENT_MODE if ENABLE_ENHANCEMENT else "none"
                ])

            except Exception as e:
                total_ms = round((time.time() - loop_start) * 1000, 2)
                print(f"[{frame_id}] ERROR: {e}")

                write_log([
                    client_ts,
                    frame_id,
                    "",
                    "",
                    "",
                    "",
                    "",
                    total_ms,
                    "error",
                    "",
                    "",
                    "",
                    ENHANCEMENT_MODE if ENABLE_ENHANCEMENT else "none"
                ])

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        picam2.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
