"""Unit tests for the score2dataset data models.

This suite ensures structural integrity, default property behaviors,
and future-proof validations for notes, scores, and dataset entries.
"""

import unittest
from pathlib import Path

from score2dataset.datamodels import DatasetEntry, NoteEvent, PerformanceScore


class TestDataModels(unittest.TestCase):
    """Encapsulates test cases verifying dataclass state and constraints."""

    # =========================================================================
    # NOTEEVENT TESTS
    # =========================================================================

    def test_note_event_creation_with_defaults(self) -> None:
        """Ensure velocity defaults to a mezzo-forte value of 64."""
        # Future property renames only require updating this instantiation
        note = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480)

        self.assertEqual(note.velocity, 64)

    def test_note_event_integrity(self) -> None:
        """Verify all explicit assignments map correctly to fields."""
        note = NoteEvent(
            pitch=72,
            onset_ticks=240,
            duration_ticks=120,
            velocity=100,
        )

        self.assertEqual(note.pitch, 72)
        self.assertEqual(note.onset_ticks, 240)
        self.assertEqual(note.duration_ticks, 120)
        self.assertEqual(note.velocity, 100)

    # =========================================================================
    # PERFORMANCESCORE TESTS
    # =========================================================================

    def test_performance_score_initializes_empty_list(self) -> None:
        """Confirm a newly parsed score defaults to an empty event array."""
        mock_path = Path("mock_score.musicxml")
        score = PerformanceScore(source=mock_path)

        self.assertIsInstance(score.events, list)
        self.assertEqual(len(score.events), 0)

    def test_performance_score_can_hold_events(self) -> None:
        """Validate appending active events into the score container."""
        score = PerformanceScore(source=Path("mock_score.musicxml"))
        note = NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480)

        score.events.append(note)

        self.assertEqual(len(score.events), 1)
        self.assertEqual(score.events[0].pitch, 60)

    # =========================================================================
    # DATASETENTRY TESTS
    # =========================================================================

    def test_dataset_entry_fields(self) -> None:
        """Assert final receipt properties match system export targets."""
        audio = Path("/tmp/render.wav")
        ground_truth = Path("/tmp/labels.json")

        entry = DatasetEntry(audio_path=audio, ground_truth_path=ground_truth)

        self.assertEqual(entry.audio_path, audio)
        self.assertEqual(entry.ground_truth_path, ground_truth)
        self.assertEqual(entry.id, "")


if __name__ == "__main__":
    unittest.main()
