"""Data validation and default property structural checks for the core score2dataset data models.

This module verifies that notes, performance timelines, and terminal dataset asset
receipts correctly instantiate fields, preserve data types, and retain custom
asymmetric tracking default values.
"""

from pathlib import Path

from score2dataset.config import DEFAULT_VELOCITY
from score2dataset.datamodels import DatasetEntry, NoteEvent, PerformanceScore


def test_note_event_creation_with_defaults() -> None:
    """Ensure velocity defaults to a mezzo-forte value of 80 and auditing flags initialize to 1."""
    note = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480)

    assert note.velocity == DEFAULT_VELOCITY
    assert note.score_expected == 1
    assert note.audio_present == 1


def test_note_event_integrity() -> None:
    """Verify all explicit assignments map correctly to fields."""
    note = NoteEvent(
        pitch=72,
        onset_ticks=240,
        duration_ticks=120,
        velocity=100,
        score_expected=0,
        audio_present=1,
    )

    assert note.pitch == 72
    assert note.onset_ticks == 240
    assert note.duration_ticks == 120
    assert note.velocity == 100
    assert note.score_expected == 0
    assert note.audio_present == 1


def test_performance_score_initializes_empty_list() -> None:
    """Confirm a newly parsed score defaults to empty primary and omitted event arrays."""
    mock_path = Path("mock_score.musicxml")
    score = PerformanceScore(source=mock_path)

    assert isinstance(score.events, list)
    assert len(score.events) == 0
    assert isinstance(score.omitted_events, list)
    assert len(score.omitted_events) == 0


def test_performance_score_can_hold_events() -> None:
    """Validate appending active events into the score container."""
    score = PerformanceScore(source=Path("mock_score.musicxml"))
    note = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480)

    score.events.append(note)

    assert len(score.events) == 1
    assert score.events[0].pitch == 60


def test_dataset_entry_fields(tmp_path: Path) -> None:
    """Assert final receipt properties match system export targets using safe pytest sandbox paths."""
    audio = tmp_path / "render.wav"
    ground_truth = tmp_path / "labels.json"

    entry = DatasetEntry(audio_path=audio, ground_truth_path=ground_truth)

    assert entry.audio_path == audio
    assert entry.ground_truth_path == ground_truth
    assert entry.id == ""
