"""Abstraction layer and contracts for musical score notation parsers.

This module establishes universal interfaces to convert varying external notation
formats into standardized in-memory PerformanceScore data models.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from score2dataset.datamodels import PerformanceScore


class ScoreParser(ABC):
    """Abstract base class establishing the contract for all score parsers.

    Any file notation reader integrated into the package must subclass this
    template to guarantee compatibility with the dataset generation loop.
    """

    @abstractmethod
    def parse_score(self, file_path: Path) -> PerformanceScore:
        """Parses an external score file and maps it to a PerformanceScore object.

        Args:
            file_path: The absolute path to the target notation file on disk.

        Returns:
            A standard populated PerformanceScore instance.

        Raises:
            ValueError: If the file content is corrupted or breaks schema rules.
        """
