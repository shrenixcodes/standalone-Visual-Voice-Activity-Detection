"""Command-line entry point for the complete visual VAD pipeline."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from visual_vad.config import VisualVADConfig, load_config
from visual_vad.engine import VisualVAD


async def _run(config: VisualVADConfig) -> None:
    async for event in VisualVAD(config).detect():
        logging.getLogger(__name__).info(
            "%s confidence=%.2f", event.event_type, event.confidence
        )


async def _run_visual(config: VisualVADConfig) -> None:
    from demo import run

    await run(config.pipeline, config.speech)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete visual voice activity detection")
    parser.add_argument("--config", type=Path, help="Optional JSON or YAML configuration file")
    parser.add_argument("--camera", type=int, help="Override webcam index")
    parser.add_argument("--visualize", action="store_true", help="Show camera and mouth ROI windows")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(arguments.config) if arguments.config else VisualVADConfig()
    if arguments.camera is not None:
        config.pipeline.camera.index = arguments.camera
    asyncio.run(_run_visual(config) if arguments.visualize else _run(config))


if __name__ == "__main__":
    main()
