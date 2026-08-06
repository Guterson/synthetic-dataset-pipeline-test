"""Unit tests for the abstract score parsing interface contracts.

This module verifies that structural base classes strictly enforce system APIs, Ensure
that any custom child classes are blocked from initialization if they fail to implement
the mandatory parse score requirements.
"""

import pytest

from score2dataset.parsers.score_parser import ScoreParser


def test_cannot_instantiate_abstract_base_class() -> None:
    """Ensure the engine blocks initialization of a raw ScoreParser."""
    with pytest.raises(TypeError):
        _ = ScoreParser()  # Type: ignore[abstract]


def test_subclass_must_implement_parse_score() -> None:
    """Ensure any child class is forced to implement parse_score."""

    class BrokenParser(ScoreParser):
        """A parser that forgets to implement the required method."""

    with pytest.raises(TypeError):
        _ = BrokenParser()  # Type: ignore[abstract]
