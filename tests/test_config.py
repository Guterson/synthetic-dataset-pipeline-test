"""Unit tests for system configuration invariants."""

from score2dataset import config


def test_tempo_bounds_mathematical_consistency() -> None:
    """Ensure the minimum BPM remains lower than the maximum BPM."""
    assert config.MIN_BPM < config.MAX_BPM


def test_grid_resolution_constraints() -> None:
    """Ensure the symbolic grid standard matches expectations."""
    # Verifies that changing this value requires intentional architectural review
    assert config.STANDARD_TPQN == 480


def test_velocity_table_coverage() -> None:
    """Verify that the dynamics dictionary maps correct string levels."""
    assert "mf" in config.NOMINAL_VELOCITIES
    assert config.NOMINAL_VELOCITIES["mf"] == config.DEFAULT_VELOCITY
