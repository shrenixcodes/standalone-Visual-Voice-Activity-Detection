# Visual Voice Activity Detection

CPU-only, real-time visual speech activity detection from a webcam. It analyzes one tracked face's lip motion and emits `speech_start` and `speech_end` events; it does not use audio.

## Features

- MediaPipe Tasks face detection and face landmarking, compatible with Python 3.13
- Stable primary-face tracking and boundary-safe 96×96 mouth ROI extraction
- Multi-signal speech scoring: mouth geometry, temporal motion, velocity, and acceleration
- Configurable moving average, exponential moving average, or median filtering
- Debounced event state machine with no duplicate transition events
- Async bounded-queue pipeline and clean public API

## Architecture

```text
Camera -> Face/Tracker -> Face Mesh -> Mouth Features -> Filter/Score -> State Machine -> SpeechEvent
```

See [architecture](docs/architecture.md) and [pipeline](docs/pipeline.md) for implementation detail.

## Installation

Python 3.11+ is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m unittest discover -s tests
```

The checked-in `assets/` directory provides the required official MediaPipe model files.

On Windows the default camera backend is DirectShow. If your camera needs a different backend, set `pipeline.camera.backend` to `auto` or `mediafoundation` in configuration.

## Quick start

```python
import asyncio
from visual_vad import VisualVAD

async def main() -> None:
    async for event in VisualVAD().detect():
        print(event.event_type, event.timestamp, event.confidence)

asyncio.run(main())
```

The same integration is available in `python examples/events.py`.

`demo.py` remains available as the visual feature-extraction diagnostic:

```powershell
python demo.py --camera 0
```

## Configuration

Copy [config.example.yaml](config.example.yaml), adjust its values, then call `load_config(path)` and pass the result to `VisualVAD`. Every speech threshold, filter setting, debounce setting, and pipeline setting is configurable. See [configuration](docs/configuration.md).

## Project structure

```text
visual_vad/  public API, async engine, filters, events, speech state/scoring
detector/    existing capture, detection, tracking, mesh, ROI, feature modules
models/      immutable frame-domain data contracts
utils/       configuration, timing, drawing, logging
assets/      MediaPipe Tasks models
tests/       deterministic component tests
docs/        architecture, pipeline, API, configuration, development notes
```

## Performance and limitations

The system initializes MediaPipe once and uses bounded queues to limit latency. Achievable FPS depends on camera resolution and CPU; use 640×480 when required. VvAD is inherently sensitive to occlusion, profile poses, low light, and non-speech mouth movements. It is a visual signal, not a reliable speech-recognition system.

## Future improvements

Calibration per camera/user, adaptive thresholds, frame-level diagnostics, metrics hooks, and optional persistent event sinks are natural next steps.

## Contributing

Run tests before submitting changes, preserve typed module boundaries, and avoid adding speech-model dependencies without a separate design review.

## License

No license has been selected for this repository yet.
