# Architecture

`VisualVAD` composes the existing camera, face detection, tracking, face mesh, ROI, and feature components with new temporal filtering, scoring, and event state components. Data moves through bounded asyncio queues: capture, feature extraction, and speech publishing. The public API only exposes immutable `SpeechEvent` objects.
