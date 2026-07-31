"""Print visual speech transition events from the default webcam."""
from __future__ import annotations

import asyncio
import logging

from visual_vad import VisualVAD


async def main() -> None:
    async for event in VisualVAD().detect():
        logging.getLogger(__name__).info("%s confidence=%.2f", event.event_type, event.confidence)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
