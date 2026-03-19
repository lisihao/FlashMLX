"""
Profile event logger
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ProfileEvent:
    """Single profiling event"""

    def __init__(
        self,
        event_type: str,
        name: str,
        timestamp: float,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type
        self.name = name
        self.timestamp = timestamp
        self.duration_ms = duration_ms
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "name": self.name,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            **self.metadata
        }


class ProfileLogger:
    """Logs profiling events to file"""

    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.events: List[ProfileEvent] = []
        self.start_time = time.perf_counter()

        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        name: str,
        duration_ms: Optional[float] = None,
        **metadata
    ):
        """Log a profiling event"""
        timestamp = time.perf_counter() - self.start_time
        event = ProfileEvent(
            event_type=event_type,
            name=name,
            timestamp=timestamp,
            duration_ms=duration_ms,
            metadata=metadata
        )
        self.events.append(event)

    def log_function_call(
        self,
        function_name: str,
        duration_ms: float,
        input_shapes: Optional[List] = None,
        memory_mb: Optional[float] = None
    ):
        """Log a function call"""
        self.log_event(
            event_type="function_call",
            name=function_name,
            duration_ms=duration_ms,
            input_shapes=input_shapes,
            memory_mb=memory_mb
        )

    def save(self, metadata: Optional[Dict[str, Any]] = None):
        """Save events to file"""
        data = {
            "metadata": metadata or {},
            "events": [event.to_dict() for event in self.events]
        }

        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def save_jsonl(self):
        """Save as JSON Lines (one event per line)"""
        output_file = self.output_file.with_suffix('.jsonl')
        with open(output_file, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict()) + '\n')

    def get_total_time(self) -> float:
        """Get total profiled time in seconds"""
        return time.perf_counter() - self.start_time

    def get_event_count(self) -> int:
        """Get number of events"""
        return len(self.events)

    def clear(self):
        """Clear all events"""
        self.events.clear()
        self.start_time = time.perf_counter()
