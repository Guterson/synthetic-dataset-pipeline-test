"""Data models for score representation, performance mutations, and dataset generation tracking.

This module provides the central vocabulary used across the score2dataset package.
It defines how musical events are structured in memory, modified through performance
perturbations, and packaged for ingestion into machine learning onset detection models.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NoteEvent:
    """Represents a single parsed and humanized musical note event.

    Unlike raw MIDI wire protocols which split actions into disjointed Note-On and
    Note-Off status messages, this model encapsulates a note's complete structural
    identity. This duration-based layout simplifies random length alterations and
    micro-timing perturbations before baking the data back out to disk.
    """

    pitch: int
    """Absolute MIDI note number ranging from 0 to 127 (e.g., 60 is Middle C)."""

    onset_ticks: int
    """The start position of the note, measured in musical ticks relative to a 
    standardized 480 Ticks Per Quarter Note (TPQN) resolution clock."""

    duration_ticks: int
    """The total sustained duration of the note measured in musical ticks."""

    velocity: int = 80
    """The dynamic striking force / volume layer of the note event, ranging from 
    0 (silent) to 127 (maximum amplitude). Defaults to a standard mezzoforte of 80."""

    score_expected: int = 1
    """Binary auditing flag tracking if this note was written in the original score.
    1 = Yes (Preserved or Omitted), 0 = No (Substitution or Ghost Note)."""

    audio_present: int = 1
    """Binary auditing flag tracking if this note physically manifests in the audio waveform.
    1 = Yes (Preserved, Substituted, or Ghost Note), 0 = No (Omitted)."""


@dataclass
class PerformanceScore:
    """Represents a complete musical performance timeline extracted from a score.

    This object serves as the primary payload passed between the system's subpackages.
    It is populated by the input parsers, structurally mutated by the performance
    modifiers (introducing tempo drift and dynamic shifts), and read by the exporters
    to generate customized raw MIDI sequences.
    """

    source: Path
    """The original file path location (such as a .musicxml score) from which 
    this performance timeline was derived."""

    events: list[NoteEvent] = field(default_factory=list)
    """An ordered array of NoteEvent instances representing the complete 
    polyphonic sequence of the performance."""

    omitted_events: list[NoteEvent] = field(default_factory=list)
    """An isolated array of NoteEvent objects that were chosen to be omitted
    during mutation, retained strictly to audit False Negatives later."""


@dataclass
class DatasetEntry:
    """Encapsulates the final synthesized assets generated for model training.

    This model acts as the terminal receipt produced by the bulk generator loop.
    It couples the physical audio rendering file with its exact ground-truth
    onset validation vectors, ready to be fed straight into a machine learning dataset loader.
    """

    audio_path: Path
    """The absolute path to the completed, rendered audio file on disk (.wav)."""

    ground_truth_path: Path
    """The absolute path to the corresponding text file (.txt, .csv, or .json) 
    containing the precise millisecond markers for every note onset event."""

    id: str = ""
    """A unique identifying hash or string tracking this specific 
    performance variation within the broader generated dataset pool."""


@dataclass
class WedgeEvent:
    """Tracks the continuous boundaries of a dynamic hairpin volume slope (Crescendo/Decrescendo)."""

    wedge_type: str  # "crescendo" or "decrescendo"
    onset_tick: int
    offset_tick: int = -1


@dataclass
class ScoreExpressionMap:
    """Carries structural metadata, textual notation markers, and spanners from a score.

    Acts as the comprehensive metadata exchange layer bridging the gap between
    raw sheet extraction and independent humanizing processors.
    """

    initial_tempo_marking: str = "Andante"
    beats_per_bar: int = 4
    ticks_per_beat: int = 480  # Forced 480 TPQN baseline
    total_ticks: int = 0

    """Broad interpretive changes over time
    e.g., {2400: "ritardando", 9600: "tempo_primo"}
    """
    text_directions: dict[int, str] = field(default_factory=dict)

    """Continuous dynamic volume shapes (Hairpins) """
    dynamic_wedges: list[WedgeEvent] = field(default_factory=list)

    # 3. Individual note articulation overlays
    # Maps absolute tick locations to specific structural modifiers
    # e.g., {1440: {"staccato", "accent"}, 2880: {"tenuto"}}
    local_articulations: dict[int, set[str]] = field(default_factory=dict)

    # 4. Phrasing boundaries for physical legato connection tracking
    # List of tuple brackets tracking slur zones: e.g., [(start_tick, end_tick)]
    slur_phrases: list[tuple[int, int]] = field(default_factory=list)
