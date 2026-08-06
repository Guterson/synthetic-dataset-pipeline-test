"""Module-level test suite for ArticulationModifier.

Validates intentional alpha scaling ranges, Gaussian biomechanical zeta noise,
and minimum duration tracking constraints across structural scores.
"""

from pathlib import Path

import pytest

from score2dataset.config import STANDARD_TPQN
from score2dataset.datamodels import NoteEvent, PerformanceScore, ScoreExpressionMap
from score2dataset.exceptions import ProcessorError
from score2dataset.processors.articulation_modifier import ArticulationModifier


@pytest.fixture(name="clean_score")
def fixture_clean_score() -> PerformanceScore:
    """Provides a baseline PerformanceScore instance with a standard quarter note."""
    score = PerformanceScore(source=Path("mock_score.musicxml"))
    score.events = [
        NoteEvent(pitch=60, onset_ticks=0, duration_ticks=STANDARD_TPQN, velocity=64),
    ]
    return score


@pytest.fixture(name="empty_expression_map")
def fixture_empty_expression_map() -> ScoreExpressionMap:
    """Provides a blank ScoreExpressionMap with no active marks or spanners."""
    return ScoreExpressionMap(
        initial_tempo_marking="Moderato",
        beats_per_bar=4,
        ticks_per_beat=STANDARD_TPQN,
        total_ticks=STANDARD_TPQN * 4,
    )


def test_initialization_invalid_duration_floor() -> None:
    """Ensure a ProcessorError is raised if d_min is set below or equal to zero."""
    invalid_floor = 0
    with pytest.raises(ProcessorError, match="must be greater than zero"):
        ArticulationModifier(d_min=invalid_floor)


def test_staccato_intentional_shortening(
    clean_score: PerformanceScore, empty_expression_map: ScoreExpressionMap
) -> None:
    """Business Rule: Staccato notes must scale down according to staccato bounds.

    Verifies that explicit local markings trigger the staccato alpha range window.
    """
    # Inject a local staccato marking at the target onset tick (0)
    target_onset = 0
    empty_expression_map.local_articulations = {target_onset: {"staccato"}}

    # Turn off random motor noise (sigma_d=0) to purely test the alpha bounding box
    modifier = ArticulationModifier(sigma_d=0.0, seed=42)
    mutated_score: PerformanceScore = modifier.perturb_articulation(
        score=clean_score, expression_map=empty_expression_map
    )

    original_duration: int = clean_score.events[0].duration_ticks
    realized_duration: int = mutated_score.events[0].duration_ticks

    # With sigma_d=0, the realized duration must be within the defined staccato window bounds
    min_allowed: int = round(original_duration * modifier.range_staccato[0])
    max_allowed: int = round(original_duration * modifier.range_staccato[1])

    assert min_allowed <= realized_duration <= max_allowed
    assert realized_duration < original_duration


def test_legato_spanner_extension(
    clean_score: PerformanceScore, empty_expression_map: ScoreExpressionMap
) -> None:
    """Business Rule: Slur phrases must trigger legato alpha scaling ranges.

    Verifies that notes falling within active spanner windows are lengthened.
    """
    # Inject a continuous legato slur spanning across the note's position
    empty_expression_map.slur_phrases = [(0, STANDARD_TPQN * 2)]

    modifier = ArticulationModifier(sigma_d=0.0, seed=42)
    mutated_score: PerformanceScore = modifier.perturb_articulation(
        score=clean_score, expression_map=empty_expression_map
    )

    original_duration: int = clean_score.events[0].duration_ticks
    realized_duration: int = mutated_score.events[0].duration_ticks

    min_allowed: int = round(original_duration * modifier.range_legato[0])
    max_allowed: int = round(original_duration * modifier.range_legato[1])

    assert min_allowed <= realized_duration <= max_allowed
    assert realized_duration > original_duration


def test_minimum_duration_floor_clamping(
    clean_score: PerformanceScore, empty_expression_map: ScoreExpressionMap
) -> None:
    """Business Rule: Realized duration must never drop below d_min.

    Forces extreme negative motor noise deviations and aggressive staccato
    bounds to verify that max(d_min, calculated) properly shields the timeline.
    """
    configured_d_min = 15
    # Force extreme negative noise offsets and near-zero scaling windows
    modifier = ArticulationModifier(
        sigma_d=500.0,
        d_min=configured_d_min,
        staccato_low=0.001,
        staccato_high=0.002,
        seed=123,
    )

    empty_expression_map.local_articulations = {0: {"staccatissimo"}}
    mutated_score: PerformanceScore = modifier.perturb_articulation(
        score=clean_score, expression_map=empty_expression_map
    )

    for event in mutated_score.events:
        assert event.duration_ticks == configured_d_min


def test_reproducibility_via_random_state(
    clean_score: PerformanceScore, empty_expression_map: ScoreExpressionMap
) -> None:
    """Ensure that identical instantiation seeds yield completely identical sequences."""
    arbitrary_seed = 9999
    proc_a = ArticulationModifier(seed=arbitrary_seed)
    proc_b = ArticulationModifier(seed=arbitrary_seed)

    score_a: PerformanceScore = proc_a.perturb_articulation(
        score=clean_score, expression_map=empty_expression_map
    )
    score_b: PerformanceScore = proc_b.perturb_articulation(
        score=clean_score, expression_map=empty_expression_map
    )

    assert score_a.events[0].duration_ticks == score_b.events[0].duration_ticks
