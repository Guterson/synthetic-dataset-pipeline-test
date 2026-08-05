"""Functional verification suite targeting timeline serialization, delta-time calculation, and overlap sorting priority inside the MIDI exporter engine."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import mido  # Tyṕe: ignore
import pytest

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.exceptions import ExporterError
from score2dataset.exporters.midi_exporter import MidiExporter


def test_exporter_initialization_defaults() -> None:
    """Verify initialization assigns target TPQN and default BPM settings."""
    exporter = MidiExporter(target_tpqn=960, fixed_bpm=100)
    assert exporter.target_tpqn == 960
    assert exporter.fixed_bpm == 100


@patch("mido.MidiFile.save")
def test_export_score_converts_absolute_ticks_to_deltas(
    mock_save: MagicMock,
) -> None:
    """Verify absolute overlapping timeline notes map into proper delta messages."""
    exporter = MidiExporter(target_tpqn=480, fixed_bpm=120)
    mock_score = PerformanceScore(source=Path("test_input.musicxml"))

    note_one = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480, velocity=64)
    note_two = NoteEvent(pitch=62, onset_ticks=240, duration_ticks=480, velocity=80)
    mock_score.events.extend([note_one, note_two])

    output_path = Path("/tmp/output.mid")

    with patch("score2dataset.exporters.midi_exporter.MidiTrack") as mock_track_class:
        real_track = mido.MidiTrack()
        mock_track_class.return_value = real_track

        exporter.export_score(mock_score, output_path)

        mock_save.assert_called_once_with(filename=str(output_path))

    assert len(real_track) == 6

    msg_1: Any = real_track[2]
    assert msg_1.type == "note_on"
    assert msg_1.note == 60
    assert msg_1.time == 0

    msg_2: Any = real_track[3]
    assert msg_2.type == "note_on"
    assert msg_2.note == 62
    assert msg_2.time == 240

    msg_3: Any = real_track[4]
    assert msg_3.type == "note_off"
    assert msg_3.note == 60
    assert msg_3.time == 240

    msg_4: Any = real_track[5]
    assert msg_4.type == "note_off"
    assert msg_4.note == 62
    assert msg_4.time == 240


@patch("mido.MidiFile.save")
def test_export_score_enforces_note_off_priority_on_identical_ticks(
    mock_save: MagicMock,
) -> None:
    """Verify note_off comes first if a note ends exactly when another begins."""
    exporter = MidiExporter()
    mock_score = PerformanceScore(source=Path("test_input.musicxml"))
    target_path = Path("/tmp/priority.mid")

    note_one = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480)
    note_two = NoteEvent(pitch=60, onset_ticks=480, duration_ticks=480)
    mock_score.events.extend([note_one, note_two])

    with patch("score2dataset.exporters.midi_exporter.MidiTrack") as mock_track_class:
        real_track = mido.MidiTrack()
        mock_track_class.return_value = real_track

        exporter.export_score(performance_score=mock_score, output_path=target_path)

    msg_off: Any = real_track[3]
    assert msg_off.type == "note_off"
    assert msg_off.time == 480

    msg_on: Any = real_track[4]
    assert msg_on.type == "note_on"
    assert msg_on.time == 0

    mock_save.assert_called_once_with(filename=str(target_path))


@patch("mido.MidiFile.save")
def test_export_score_wraps_io_failures_into_exporter_error(
    mock_save: MagicMock,
) -> None:
    """Ensure that system IO errors during save operations are caught and raised as ExporterError."""
    exporter = MidiExporter()
    mock_score = PerformanceScore(source=Path("test_input.musicxml"))
    target_path = Path("/invalid/path/file.mid")

    mock_save.side_effect = OSError("Permission denied")

    with pytest.raises(ExporterError):
        exporter.export_score(performance_score=mock_score, output_path=target_path)
