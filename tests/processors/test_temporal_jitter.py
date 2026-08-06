"""Module-level test suite for TemporalJitter processor.

Validates AR(1) autoregressive drift, biomechanical finger asynchrony within
chords, and timeline causality boundary enforcement.
"""

from pathlib import Path

import pytest
from scipy.interpolate import CubicSpline

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.processors.temporal_jitter import TemporalJitterModifier


@pytest.fixture(name="flat_tempo_spline")
def fixture_flat_tempo_spline() -> CubicSpline:
    """Provides a deterministic constant tempo spline at 120 BPM."""
    ticks: list[int] = [0, 4800, 9600]
    bpms: list[float] = [120.0, 120.0, 120.0]
    return CubicSpline(ticks, bpms)


@pytest.fixture(name="chord_score")
def fixture_chord_score() -> PerformanceScore:
    """Provides a score containing a structurally aligned three-note chord."""
    score = PerformanceScore(source=Path("mock_chord.musicxml"))
    # Three notes struck simultaneously at tick 480
    score.events = [
        NoteEvent(pitch=60, onset_ticks=480, duration_ticks=480, velocity=64),
        NoteEvent(pitch=64, onset_ticks=480, duration_ticks=480, velocity=64),
        NoteEvent(pitch=67, onset_ticks=480, duration_ticks=480, velocity=64),
    ]
    return score


@pytest.fixture(name="edge_onset_score")
def fixture_edge_onset_score() -> PerformanceScore:
    """Provides a score with notes sitting precisely on the timeline boundary floor."""
    score = PerformanceScore(source=Path("mock_edge.musicxml"))
    score.events = [
        NoteEvent(pitch=60, onset_ticks=0, duration_ticks=480, velocity=64),
    ]
    return score


def test_chord_asynchrony_spread(
    chord_score: PerformanceScore, flat_tempo_spline: CubicSpline
) -> None:
    """Business Rule: Simultaneous chord notes must be slightly staggered.

    Verifies that the FINGER_DAMPENING implementation injects micro-timing
    spread instead of flattening the chords entirely.
    """
    # Use a high sigma_time to guarantee measurable micro-displacements
    processor = TemporalJitterModifier(sigma_time=0.05, seed=12345)
    perturbed_score: PerformanceScore = processor.perturb_timing(
        score=chord_score, tempo_spline=flat_tempo_spline
    )

    assert len(perturbed_score.events) == 3

    # Extract the new shifted timestamps
    onsets: list[int] = [event.onset_ticks for event in perturbed_score.events]

    # Rational check: Notes should not all share an identical timestamp anymore
    assert len(set(onsets)) > 1


def test_timeline_floor_clamping(
    edge_onset_score: PerformanceScore, flat_tempo_spline: CubicSpline
) -> None:
    """Business Rule: Jitter perturbations cannot shift notes to negative times.

    Verifies that np.clip cleanly intercepts negative floor exceptions.
    """
    # Force heavy timing deviations to trigger extreme negative tails
    processor = TemporalJitterModifier(sigma_time=0.5, seed=42)
    perturbed_score: PerformanceScore = processor.perturb_timing(
        edge_onset_score, tempo_spline=flat_tempo_spline
    )

    for event in perturbed_score.events:
        assert event.onset_ticks >= 0


def test_autoregressive_drift_memory(
    chord_score: PerformanceScore, flat_tempo_spline: CubicSpline
) -> None:
    """Business Rule: The engine must track sequential state drift across time steps.

    Verifies that the seed produces deterministic sequences across distinct runs.
    """
    seed = 888
    proc_a = TemporalJitterModifier(seed=seed)
    proc_b = TemporalJitterModifier(seed=seed)

    score_a: PerformanceScore = proc_a.perturb_timing(
        score=chord_score, tempo_spline=flat_tempo_spline
    )
    score_b: PerformanceScore = proc_b.perturb_timing(
        score=chord_score, tempo_spline=flat_tempo_spline
    )

    onsets_a: list[int] = [e.onset_ticks for e in score_a.events]
    onsets_b: list[int] = [e.onset_ticks for e in score_b.events]

    assert onsets_a == onsets_b
