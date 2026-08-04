from __future__ import annotations

import unittest
import numpy as np

from detector.face_tracker import FaceTracker
from detector.face_pose import estimate_yaw_degrees, has_usable_lips
from detector.feature_extractor import MouthFeatureExtractor
from detector.mouth_roi import MouthROIExtractor
from models.face import BoundingBox, Face
from utils.config import FaceQualityConfig, MouthRoiConfig, TrackerConfig
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

    def test_tracker_predicts_short_face_loss(self) -> None:
        tracker = FaceTracker(TrackerConfig(max_missed_frames=1))
        tracker.update([Face(0, BoundingBox(0, 0, 20, 20), 1.0)], (100, 100, 3))
        tracker.update([Face(0, BoundingBox(10, 0, 20, 20), 1.0)], (100, 100, 3))
        predicted = tracker.update([], (100, 100, 3))
        self.assertEqual(predicted.bbox.x, 20)

    def test_pose_and_landmark_quality_checks(self) -> None:
        points = [(0, 0)] * 455
        points[234], points[1], points[454] = (10, 20), (50, 20), (90, 20)
        self.assertEqual(estimate_yaw_degrees(points), 0.0)
        points[1] = (80, 20)
        self.assertGreater(abs(estimate_yaw_degrees(points)), 30)
        self.assertTrue(has_usable_lips([(10, 10)] * 16, (100, 100, 3), FaceQualityConfig()))
        self.assertFalse(has_usable_lips([(120, 10)] * 16, (100, 100, 3), FaceQualityConfig()))

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
        machine = SpeechStateMachine(StateMachineConfig(start_confidence=0.6, end_confidence=0.2, start_hold_seconds=0.2, end_hold_seconds=1.0, minimum_talking_seconds=0.5))
        self.assertIsNone(machine.update(0.8, 1.0))
        self.assertEqual(machine.update(0.8, 1.25).event_type, SpeechEventType.SPEECH_START)
        self.assertIsNone(machine.update(0.1, 1.8))
        self.assertIsNone(machine.update(0.1, 2.7))
        self.assertEqual(machine.update(0.1, 2.85).event_type, SpeechEventType.SPEECH_END)

    def test_state_machine_ignores_brief_confidence_drops_while_talking(self) -> None:
        machine = SpeechStateMachine(StateMachineConfig(start_confidence=0.6, end_confidence=0.2, start_hold_seconds=0, end_hold_seconds=1.5, minimum_talking_seconds=0))
        self.assertEqual(machine.update(0.8, 0.0).event_type, SpeechEventType.SPEECH_START)
        for timestamp, confidence in ((0.2, 0.1), (0.7, 0.1), (1.0, 0.7), (1.2, 0.1)):
            self.assertIsNone(machine.update(confidence, timestamp))
        self.assertEqual(machine.state.value, "talking")

    def test_speech_detector_emits_transitions(self) -> None:
        config = SpeechDetectorConfig(motion_threshold=0.001, velocity_threshold=0.01, acceleration_threshold=0.01)
        config.state_machine = StateMachineConfig(start_confidence=0.1, end_confidence=0.01, start_hold_seconds=0, end_hold_seconds=0, minimum_talking_seconds=0)
        detector = VisualSpeechDetector(config)
        roi = np.zeros((96, 96, 3), dtype=np.uint8)
        extractor = MouthFeatureExtractor()
        samples = [extractor.extract(time, [(0, 0), (width, 10)], roi) for time, width in ((0., 10), (.1, 20), (.2, 10), (.3, 10), (.4, 10))]
        events = [detector.update(sample) for sample in samples]
        self.assertIn(SpeechEventType.SPEECH_START, [event.event_type for event in events if event])


if __name__ == "__main__":
    unittest.main()
