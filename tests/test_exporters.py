"""Unit tests for performance exporters.

Ensures that absolute timeline coordinates mapping NoteEvent matrices
serialize accurately into sequential delta-time standard MIDI streams.
"""

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import mido  # type: ignore

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.exporters.midi_exporter import MidiExporter


class TestMidiExporter(unittest.TestCase):
    """Encapsulates test scenarios for translation and serialization logic."""

    def test_exporter_initialization_defaults(self) -> None:
        """Verify initialization assigns target TPQN and default BPM settings."""
        exporter = MidiExporter(target_tpqn=960, fixed_bpm=100)
        self.assertEqual(exporter.target_tpqn, 960)
        self.assertEqual(exporter.fixed_bpm, 100)

    @patch("mido.MidiFile.save")
    def test_export_score_converts_absolute_ticks_to_deltas(
        self,
        mock_save: MagicMock,
    ) -> None:
        """Verify absolute overlapping timeline notes map into proper delta messages."""
        exporter = MidiExporter(target_tpqn=480, fixed_bpm=120)
        mock_score = PerformanceScore(source=Path("test_input.musicxml"))

        # Create two overlapping notes:
        # Note 1: Pitch 60, Onset 0, Duration 480 (Ends at 480)
        # Note 2: Pitch 62, Onset 240, Duration 480 (Ends at 720)
        note_one = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480, velocity=64)
        note_two = NoteEvent(pitch=62, onset_ticks=240, duration_ticks=480, velocity=80)
        mock_score.events.extend([note_one, note_two])

        output_path = Path("/tmp/output.mid")

        # Capture the internal midi engine track compilation
        with patch(
            "score2dataset.exporters.midi_exporter.MidiTrack"
        ) as mock_track_class:
            real_track = mido.MidiTrack()
            mock_track_class.return_value = real_track

            exporter.export_score(mock_score, output_path)

            # Ensure mido.MidiFile.save was triggered with our target destination path
            mock_save.assert_called_once_with(filename=str(output_path))

        # The expected timeline of absolute actions is:
        # Tick 0:   note_on pitch 60
        # Tick 240: note_on pitch 62
        # Tick 480: note_off pitch 60
        # Tick 720: note_off pitch 62
        #
        # Converted to Delta Times:
        # Msg 1: note_on  (60) -> delta = 0 (0 - 0)
        # Msg 2: note_on  (62) -> delta = 240 (240 - 0)
        # Msg 3: note_off (60) -> delta = 240 (480 - 240)
        # Msg 4: note_off (62) -> delta = 240 (720 - 480)

        self.assertEqual(len(real_track), 6)  # 4 note actions + 1 program change

        # Validate message sequencing and matching delta offsets
        # i = 0: set_tempo; i = 1: Program Change (default channel)
        msg_1: Any = real_track[2]  # Index 0 is program_change
        self.assertEqual(msg_1.type, "note_on")
        self.assertEqual(msg_1.note, 60)
        self.assertEqual(msg_1.time, 0)

        msg_2: Any = real_track[3]
        self.assertEqual(msg_2.type, "note_on")
        self.assertEqual(msg_2.note, 62)
        self.assertEqual(msg_2.time, 240)

        msg_3: Any = real_track[4]
        self.assertEqual(msg_3.type, "note_off")
        self.assertEqual(msg_3.note, 60)
        self.assertEqual(msg_3.time, 240)

        msg_4: Any = real_track[5]
        self.assertEqual(msg_4.type, "note_off")
        self.assertEqual(msg_4.note, 62)
        self.assertEqual(msg_4.time, 240)

    @patch("mido.MidiFile.save")
    def test_export_score_enforces_note_off_priority_on_identical_ticks(
        self,
        mock_save: MagicMock,
    ) -> None:
        """Verify note_off comes first if a note ends exactly when another begins."""
        exporter = MidiExporter()
        mock_score = PerformanceScore(source=Path("test_input.musicxml"))
        target_path = Path("/tmp/priority.mid")

        note_one = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480)
        note_two = NoteEvent(pitch=60, onset_ticks=480, duration_ticks=480)
        mock_score.events.extend([note_one, note_two])

        with patch(
            "score2dataset.exporters.midi_exporter.MidiTrack"
        ) as mock_track_class:
            real_track = mido.MidiTrack()
            mock_track_class.return_value = real_track

            exporter.export_score(performance_score=mock_score, output_path=target_path)

        # 1. Assertions verifying your custom sorting rules are functioning
        # i = 0: set_tempo; i = 1: Program Change (default channel)

        msg_off: Any = real_track[3]
        self.assertEqual(msg_off.type, "note_off")
        self.assertEqual(msg_off.time, 480)

        msg_on: Any = real_track[4]
        self.assertEqual(msg_on.type, "note_on")
        self.assertEqual(msg_on.time, 0)

        # 2. ACTUALLY USE THE MOCK: Verify that the save loop executed cleanly
        mock_save.assert_called_once_with(filename=str(target_path))


if __name__ == "__main__":
    unittest.main()
