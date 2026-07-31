# Pipeline

Frames are captured with monotonic timestamps. The largest reacquired primary face is tracked, then its face crop is landmarked. Lip landmarks create a 96x96 ROI and mouth geometry. The detector derives temporal movement, velocity, and acceleration, filters motion, combines normalized scores, and debounces state transitions.
