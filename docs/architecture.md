# System Architecture

This project is a two-device edge-cloud-style video AI demo using a Raspberry Pi as the weaker edge device and a Windows laptop as the stronger AI processing device.

## Data Flow

Camera -> Raspberry Pi Picamera2 -> edge-side enhancement/preprocessing -> JPEG compression -> Wi-Fi HTTP upload -> Laptop Flask receiver -> YuNet neural face detection -> annotated MJPEG dashboard + logs + JSON result.

## Device A: Raspberry Pi

Device A is a Raspberry Pi 4 Model B with an IMX708 Raspberry Pi Camera. It is responsible for the origin of the raw video stream and for lightweight edge-side preprocessing before upload.

The Raspberry Pi client performs:

- continuous frame capture through Picamera2
- frame-rate control with a target of approximately 5 FPS
- 640x360 frame sizing
- CLAHE local contrast enhancement in LAB color space
- mild sharpening
- JPEG compression
- HTTP POST upload over Wi-Fi to the laptop

## Device B: Laptop

Device B is a Windows laptop. It receives the compressed frames, runs the neural-network-based AI algorithm, visualizes the output, and records runtime evidence.

The laptop Flask service performs:

- token check for incoming frame uploads
- JPEG decoding
- OpenCV YuNet ONNX neural face detection
- bounding-box annotation
- MJPEG live stream generation
- JSON response generation for each uploaded frame
- CSV/JSON runtime logging
- browser dashboard for runtime stats

## AI Model

The AI model is the pre-trained OpenCV YuNet ONNX face detector:

`models/face_detection_yunet_2023mar.onnx`

This model is used as a neural-network-based detector. It was not self-trained in this project.

## Runtime Outputs

The system outputs:

- annotated MJPEG stream at `http://YOUR_LAPTOP_IP:5000/`
- runtime stats dashboard in the same Flask page
- JSON response from `/upload_frame`
- CSV log of laptop-side detections
- JSON summary of current runtime state
- Raspberry Pi client CSV upload log

## Repository Evidence

The repository includes the source code, setup files, architecture and run instructions, sealed demo logs, and demo media captured from the real Raspberry Pi and laptop test.
