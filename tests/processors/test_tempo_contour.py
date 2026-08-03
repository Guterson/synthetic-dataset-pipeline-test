"""Module-level test suite for TempoContourGenerator.

Validates C2-continuous spline interpolation curves, biomechanical inertia
boundary constraints, and deterministic anchor reproducibility.
"""

from pathlib import Path

import pytest
from scipy.interpolate import CubicSpline

from score2dataset.datamodels import NoteEvent, PerformanceScore, ScoreExpressionMap
from score2dataset.processors.tempo_contour import TempoContourGenerator


@pytest.fixture(name="expression_map")
def fixture_expression_map() -> ScoreExpressionMap:
    """Provides a standardized score expression map tracking structural markings."""
    expr_map = ScoreExpressionMap(
        initial_tempo_marking="Andante",
        beats_per_bar=4,
        ticks_per_beat=480,
        total_ticks=9600,  # Explicit total length boundary
    )
    # Inject continuous dynamic phrasing anchors/text directions
    expr_map.text_directions = {
        2400: "ritardando",
        4800: "accelerando",
        7200: "tempo_primo",
    }
    return expr_map


@pytest.fixture(name="populated_score")
def fixture_populated_score() -> PerformanceScore:
    """Provides a basic PerformanceScore sequence containing baseline timeline ticks."""
    score = PerformanceScore(source=Path("mock_score.musicxml"))
    score.events = [
        NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480, velocity=64),
        NoteEvent(pitch=62, onset_ticks=2400, duration_ticks=480, velocity=70),
        NoteEvent(pitch=64, onset_ticks=4800, duration_ticks=480, velocity=80),
    ]
    return score


def test_generator_initialization() -> None:
    """Verifies that the generator instantiates cleanly with operational defaults."""
    generator = TempoContourGenerator()
    assert isinstance(generator, TempoContourGenerator)
    assert hasattr(generator, "seed")


def test_spline_continuity_assertion(expression_map: ScoreExpressionMap) -> None:
    """Business Rule: The generated tempo map curve must be C2-continuous.

    Verifies the internal mathematical maps extract functional Scipy CubicSplines.
    """
    generator = TempoContourGenerator()
    tempo_spline: CubicSpline = generator.generate_tempo_map(metadata=expression_map)

    # Validate structural math engine contract
    assert isinstance(tempo_spline, CubicSpline)
    # CubicSpline naturally forces continuous 1st and 2nd derivatives.
    # Check that it can compute derivatives safely without blowing up
    first_derivative: CubicSpline = tempo_spline.derivative(nu=1)
    second_derivative: CubicSpline = tempo_spline.derivative(nu=2)

    assert first_derivative is not None
    assert second_derivative is not None


def test_stochastic_anchor_generation(expression_map: ScoreExpressionMap) -> None:
    """Ensures temporal anchors fall safely within valid performance ranges."""
    generator = TempoContourGenerator(seed=42)
    tempo_spline: CubicSpline = generator.generate_tempo_map(metadata=expression_map)

    # Sample tempo values across the timeline tracking points (ticks 0 to 9600)
    for tick in range(9601):
        tempo_at_tick = float(tempo_spline(tick))
        # Ensure tempo stays within physically playable limits (e.g., 40 to 240 BPM)
        assert 40.0 <= tempo_at_tick <= 240.0


def test_physical_boundary_adherence(
    populated_score: PerformanceScore, expression_map: ScoreExpressionMap
) -> None:
    """Ensures tempo transformations prevent biomechanical physics violations.

    Mutated events cannot shift to negative times or cause overlap reversals.
    """
    generator = TempoContourGenerator()
    tempo_spline: CubicSpline = generator.generate_tempo_map(metadata=expression_map)

    mutated_score: PerformanceScore = generator.apply_phrasing_to_score(
        populated_score, tempo_spline
    )

    previous_onset: int = -1
    for note in mutated_score.events:
        assert note.onset_ticks >= 0
        assert note.duration_ticks > 0
        # Sequential check: humanization must not reverse structural chronological ordering
        assert note.onset_ticks >= previous_onset
        previous_onset = note.onset_ticks


def test_reproducibility_with_seed(expression_map: ScoreExpressionMap) -> None:
    """Ensures identical seeding criteria output completely deterministic tempo points."""
    seed = 999
    gen_a = TempoContourGenerator(seed=seed)
    gen_b = TempoContourGenerator(seed=seed)

    spline_a: CubicSpline = gen_a.generate_tempo_map(metadata=expression_map)
    spline_b: CubicSpline = gen_b.generate_tempo_map(metadata=expression_map)

    # Sample check absolute array coefficients matching exactly
    for tick in range(9601):
        assert spline_a(tick) == spline_b(tick)
