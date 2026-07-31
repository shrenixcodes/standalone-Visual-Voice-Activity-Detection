from __future__ import annotations

import unittest
import numpy as np

from detector.face_tracker import FaceTracker
from detector.feature_extractor import MouthFeatureExtractor
from detector.mouth_roi import MouthROIExtractor
from models.face import BoundingBox, Face
from utils.config import MouthRoiConfig, TrackerConfig
from visual_vad.events import SpeechEventType
from visual_vad.filters import FilterConfig, FilterStrategy, TemporalFilter
from visual_vad.speech_detector import SpeechDetectorConfig, VisualSpeechDetector
from visual_vad.state_machine import SpeechStateMachine, StateMachineConfig


class CoreTests(unittest.TestCase):
    def test_tracker_keeps_id_and_reacquires_largest(self) -> None:
        tracker = FaceTracker(TrackerConfig(max_missed_frames=0))
        first = tracker.update([Face(0, BoundingBox(1, 1, 10, 10), 1.0)], (100, 100, 3))
        self.assertEqual(first.id, 1)
        second = tracker.update([Face(0, BoundingBox(2, 2, 10, 10), 1.0)], (100, 100, 3))
        self.assertEqual(second.id, 1)
        tracker.update([], (100, 100, 3))
        third = tracker.update([Face(0, BoundingBox(1, 1, 20, 20), 1.0)], (100, 100, 3))
        self.assertEqual(third.id, 2)

    def test_roi_and_features(self) -> None:
        points = [(40, 45), (60, 45), (60, 55), (40, 55)]
        result = MouthROIExtractor(MouthRoiConfig()).extract(np.zeros((100, 100, 3), dtype=np.uint8), points)
        self.assertIsNotNone(result)
        features = MouthFeatureExtractor().extract(1.0, points, result[1])
        self.assertEqual((features.mouth_width, features.mouth_height), (20.0, 10.0))

    def test_temporal_filters(self) -> None:
        moving = TemporalFilter(FilterConfig(FilterStrategy.MOVING_AVERAGE, window_size=3))
        self.assertEqual([moving.update(value) for value in (1, 2, 3)], [1, 1.5, 2])
        median_filter = TemporalFilter(FilterConfig(FilterStrategy.MEDIAN, window_size=3))
        self.assertEqual(median_filter.update(5), 5)
        self.assertEqual(median_filter.update(1), 3)

    def test_state_machine_debounces_events(self) -> None:
        machine = SpeechStateMachine(StateMachineConfig(start_frames=2, end_frames=2))
        self.assertIsNone(machine.update(True, 1, 1))
        self.assertEqual(machine.update(True, 2, 1).event_type, SpeechEventType.SPEECH_START)
        self.assertIsNone(machine.update(True, 3, 1))
        self.assertIsNone(machine.update(False, 4, 0))
        self.assertEqual(machine.update(False, 5, 0).event_type, SpeechEventType.SPEECH_END)

    def test_speech_detector_emits_transitions(self) -> None:
        config = SpeechDetectorConfig(min_confidence=0.1, motion_threshold=0.001, velocity_threshold=0.01, acceleration_threshold=0.01)
        config.state_machine = StateMachineConfig(start_frames=2, end_frames=2)
        detector = VisualSpeechDetector(config)
        roi = np.zeros((96, 96, 3), dtype=np.uint8)
        extractor = MouthFeatureExtractor()
        samples = [extractor.extract(time, [(0, 0), (width, 10)], roi) for time, width in ((0., 10), (.1, 20), (.2, 10), (.3, 10), (.4, 10))]
        events = [detector.update(sample) for sample in samples]
        self.assertIn(SpeechEventType.SPEECH_START, [event.event_type for event in events if event])


if __name__ == "__main__":
    unittest.main()
