# API

```python
from visual_vad import VisualVAD

async for event in VisualVAD().detect():
    print(event.event_type, event.timestamp, event.confidence)
```

Events are emitted only when the speech state changes.
