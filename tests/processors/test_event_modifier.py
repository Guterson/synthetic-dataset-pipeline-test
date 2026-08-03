"""Module level test suite for the EventLevelModifier humanizer processor.

Ensures stochastic mutations adhere to state consistency contracts, reproducibility
matrices, and rigorous error boundaries.
"""

from pathlib import Path

import pytest

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.exceptions import ProcessorError
from score2dataset.processors.event_modifier import EventLevelModifier


@pytest.fixture(name="mock_score")
def fixture_mock_score() -> PerformanceScore:
    """Provides a deterministic performance score timeline containing standard notes."""
    score = PerformanceScore(source=Path("test_input.musicxml"))
    score.events = [
        NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480, velocity=64),
        NoteEvent(pitch=64, onset_ticks=480, duration_ticks=480, velocity=64),
        NoteEvent(pitch=67, onset_ticks=960, duration_ticks=960, velocity=80),
    ]
    return score


def test_initialization_valid_defaults() -> None:
    """Verifies default parameter state assignments without relying on volatile magic values."""
    modifier = EventLevelModifier()
    assert hasattr(modifier, "probabilities")
    assert modifier.sigma_p > 0.0


def test_probability_sum_validation() -> None:
    """Ensures a explicit ProcessorError is thrown if categorization distribution weights bypass 1.0."""
    invalid_p_keep = 0.4
    invalid_p_sub = 0.2
    invalid_p_omit = 0.2  # Totals 0.8 instead of 1.0

    with pytest.raises(
        ProcessorError, match="Categorical probabilities must sum to 1.0"
    ):
        EventLevelModifier(
            p_keep=invalid_p_keep, p_sub=invalid_p_sub, p_omit=invalid_p_omit
        )


def test_rng_reproducibility(mock_score: PerformanceScore) -> None:
    """Ensures that distinct instances utilizing identical tracking seeds output identical scores."""
    seed = 54321
    modifier_a = EventLevelModifier(seed=seed)
    modifier_b = EventLevelModifier(seed=seed)

    # Trigger sequential generation runs across isolated pipelines
    score_a: PerformanceScore = modifier_a.perturb_score(canonical_score=mock_score)
    score_b: PerformanceScore = modifier_b.perturb_score(canonical_score=mock_score)

    assert len(score_a.events) == len(score_b.events)
    for note_a, note_b in zip(score_a.events, score_b.events):
        assert note_a.pitch == note_b.pitch
        assert note_a.onset_ticks == note_b.onset_ticks
        assert note_a.velocity == note_b.velocity


def test_rng_independence(mock_score: PerformanceScore) -> None:
    """Ensures that disparate initialization seeds yield non-identical timeline variations."""
    modifier_a = EventLevelModifier(seed=111)
    modifier_b = EventLevelModifier(seed=999)

    score_a: PerformanceScore = modifier_a.perturb_score(canonical_score=mock_score)
    score_b: PerformanceScore = modifier_b.perturb_score(canonical_score=mock_score)

    # A sufficiently deep transformation list must drift with distinct seeds
    serialized_a: list[tuple[int, int, int]] = [
        (n.pitch, n.onset_ticks, n.velocity) for n in score_a.events
    ]
    serialized_b: list[tuple[int, int, int]] = [
        (n.pitch, n.onset_ticks, n.velocity) for n in score_b.events
    ]

    assert serialized_a != serialized_b


def test_mutation_bounds_preservation(mock_score: PerformanceScore) -> None:
    """Validates that stochastic mutations preserve core pitch parameters within valid MIDI ranges."""
    modifier = EventLevelModifier(seed=42)
    mutated_score = modifier.perturb_score(canonical_score=mock_score)

    for note in mutated_score.events:
        assert 0 <= note.pitch <= 127
        assert 0 <= note.velocity <= 127
        assert note.duration_ticks > 0
        assert note.onset_ticks >= 0


def test_extreme_omission_profile(mock_score: PerformanceScore) -> None:
    """Forces the omission threshold to maximum to guarantee notes are dropped correctly."""
    # Force p_omit to high values via instantiation args if your API allows it,
    # or pass specific test parameters to check structural dropping logic safely.
    modifier = EventLevelModifier(p_keep=0.0, p_sub=0.0, p_omit=1.0, seed=123)
    mutated_score: PerformanceScore = modifier.perturb_score(canonical_score=mock_score)

    # Timeline should either be cleared completely or reduced to zero notes
    assert len(mutated_score.events) == 0
