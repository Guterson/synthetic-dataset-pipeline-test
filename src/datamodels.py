from dataclasses import dataclass, field
from typing import List


@dataclass
class NoteEvent:
    pitch: int  # Absolute MIDI note (0-127)
    onset_ticks: int  # Pre-baked start position in a 120 BPM timeline
    duration_ticks: int  # Pre-baked duration in ticks scaled to a 120 BPM timeline
    velocity: int = 64  # Final dynamic velocity (0-127)


@dataclass
class PerformanceScore:
    events: List[NoteEvent] = field(default_factory=list)
