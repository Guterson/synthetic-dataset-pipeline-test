"""Centralized exception hierarchy for the score2dataset package.

Provides structured, domain-specific error types allowing the dataset
generation pipeline to respond distinctively to configuration, parsing,
and runtime execution faults.
"""

from collections.abc import Callable
from typing import Any


class Score2DatasetError(Exception):
    """Base exception class for all errors raised by this package."""


class AudioEngineError(Score2DatasetError):
    """Tracks backend executable failures and child shell process crashes.

    Attributes:
        returncode: The exit status code returned by the child process shell.
        details: Raw stdout or stderr messages pulled from the execution crash.
    """

    def __init__(self, message: str, returncode: int, details: str = "") -> None:
        super().__init__(message)
        self.returncode: int = returncode
        self.details: str = details

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        """Customizes pickling behavior to preserve attributes across process boundaries."""
        return (
            self.__class__,
            (self.args[0], self.returncode, self.details),
        )


class ExporterError(Score2DatasetError):
    """Raised when serialization to standard protocols (like MIDI) fails."""


class ParserError(Score2DatasetError):
    """Raised when an external score file breaks schema rules or is corrupt."""


class ProcessorError(Score2DatasetError):
    """Raised when an internal audio digital signal processing step fails."""
