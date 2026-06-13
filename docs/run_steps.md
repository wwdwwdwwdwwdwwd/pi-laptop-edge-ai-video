# Run Steps

These steps are for reproducing the sealed two-device demo. Do not include Wi-Fi passwords, private tokens, or other secrets in the repository.

## Laptop Side: Device B

Open PowerShell in the repository root, then run:

```powershell
cd laptop
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe receiver_stream_ai.py
```

Open the dashboard:

```text
http://127.0.0.1:5000/
```

For Raspberry Pi uploads from another device on the same Wi-Fi network, use the laptop's local network IP instead of `127.0.0.1`, for example:

```text
http://10.253.24.73:5000/
```

The laptop script listens on `0.0.0.0:5000` and expects the YuNet model at:

```text
../models/face_detection_yunet_2023mar.onnx
```

## Raspberry Pi Side: Device A

Copy or clone this repository to the Raspberry Pi, then enter the Raspberry Pi script directory:

```bash
git clone https://github.com/wwdwwdwwdwwdwwd/pi-laptop-edge-ai-video.git
cd pi-laptop-edge-ai-video/raspberry_pi
bash setup_pi.sh
python3 pi_video_stream_upload.py
```

Before running the Pi client, update `LAPTOP_IP` in `pi_video_stream_upload.py` to the real laptop IP on the same Wi-Fi network. Example:

```python
LAPTOP_IP = "10.253.24.73"
```

The Pi script posts frames to:

```text
http://YOUR_LAPTOP_IP:5000/upload_frame
```

## Expected Behavior

1. Raspberry Pi captures frames from the IMX708 camera.
2. Raspberry Pi resizes/controls the stream to 640x360 and approximately 5 FPS.
3. Raspberry Pi applies CLAHE and mild sharpening, then JPEG-compresses each frame.
4. Raspberry Pi sends each frame to the laptop with an HTTP POST request over Wi-Fi.
5. Laptop receives frames in Flask, validates the shared token, and runs OpenCV YuNet ONNX neural face detection.
6. Laptop returns JSON with face count, bounding boxes, confidence scores, AI inference time, and FPS estimate.
7. Laptop dashboard displays annotated MJPEG video and runtime stats.

## Notes

- Keep the laptop and Raspberry Pi on the same local network or hotspot.
- If Windows Firewall asks for permission, allow Python/Flask on the private network.
- Do not commit local `.venv/`, generated frame folders, or private `.env` files.
