# Standalone Visual Voice Activity Detection (VvAD)

A CPU-only webcam prototype that stops strictly at **mouth feature extraction**. It has no speech classifier, talking/not-talking output, speech events, WebRTC, or deep-learning speech model.

## Pipeline

`webcam -> multi-face detector -> primary-face tracker -> face mesh -> mouth ROI -> MouthFeatures`

All faces are detected each frame. The tracker keeps one primary ID through short detection losses using overlap matching; when it is lost it reacquires the largest face. Face Mesh runs only on a padded crop of that primary face. The bundled `assets/` directory contains the official MediaPipe Tasks models required by MediaPipe 0.10.35+ (including Python 3.13).

## Setup

Python 3.11+ and a webcam are required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python demo.py --camera 0
```

Press `q` or `Esc` to exit. Select a different webcam with `--camera 1`. If CPU performance is constrained, try `--width 640 --height 480`.

## Output

`MouthFeatureExtractor` produces an immutable `MouthFeatures` object for every valid frame:

- Monotonic capture `timestamp`
- `mouth_width`, `mouth_height`, rectangle `mouth_area`, and `mouth_aspect_ratio` (height/width)
- Lip `centroid` in source-frame pixels
- 96 x 96 BGR `roi`

## Visual output / example screenshots

The **Visual VAD** window shows a green primary-face rectangle and stable face ID, orange lip landmarks, a blue padded mouth rectangle, rolling FPS, and tracking status. The **Mouth ROI (96x96)** window continuously shows the normalized mouth crop. When a face, mesh, or landmarks are unavailable, the main status reports the condition without crashing.

## Configuration and layout

Typed dataclasses in `utils/config.py` control camera properties, detector confidence, tracking IoU/loss duration, face-crop padding, mouth padding, and ROI size.

```text
detector/  webcam capture, detection, tracking, mesh, ROI, feature extraction
models/    Face and MouthFeatures data contracts
utils/     configuration, logging, FPS, visualization
assets/    official MediaPipe face detector and landmarker model files
demo.py    asynchronous interactive entry point
```

Empty frames and disconnections are logged and end capture cleanly. Missing faces, landmarks, or ROI data are safely skipped.
