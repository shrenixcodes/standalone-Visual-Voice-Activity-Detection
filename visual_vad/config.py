"""Application configuration loading from JSON or YAML."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from utils.config import AppConfig
from visual_vad.filters import FilterConfig, FilterStrategy
from visual_vad.speech_detector import SpeechDetectorConfig
from visual_vad.state_machine import StateMachineConfig


@dataclass(slots=True)
class VisualVADConfig:
    pipeline: AppConfig = field(default_factory=AppConfig)
    speech: SpeechDetectorConfig = field(default_factory=SpeechDetectorConfig)
    queue_size: int = 3
    visualize: bool = True


def load_config(path: str | Path) -> VisualVADConfig:
    file_path = Path(path)
    if file_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("YAML configuration requires PyYAML; install project dependencies") from error
        raw: dict[str, Any] = yaml.safe_load(file_path.read_text())
    else:
        raw = json.loads(file_path.read_text())
    config = VisualVADConfig()
    _apply(config.pipeline, raw.get("pipeline", {}))
    speech = raw.get("speech", {})
    _apply(config.speech, {key: value for key, value in speech.items() if key not in {"filter", "state_machine"}})
    if "filter" in speech:
        filter_data = dict(speech["filter"])
        if "strategy" in filter_data:
            filter_data["strategy"] = FilterStrategy(filter_data["strategy"])
        config.speech.filter = FilterConfig(**filter_data)
    if "state_machine" in speech:
        config.speech.state_machine = StateMachineConfig(**speech["state_machine"])
    _apply(config, {key: value for key, value in raw.items() if key not in {"pipeline", "speech"}})
    return config


def _apply(target: object, values: dict[str, Any]) -> None:
    allowed = {item.name for item in fields(target)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    for key, value in values.items():
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(target, key, value)
