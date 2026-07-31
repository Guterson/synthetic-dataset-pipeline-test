"""Unit tests for the SfzConditioner class using the unittest framework."""

import shutil
import tempfile
import unittest
from pathlib import Path

from score2dataset.processors.sfz_conditioner import SfzConditioner


class TestSfzConditioner(unittest.TestCase):
    """Verifies text extraction and modification integrity of the SfzConditioner."""

    def setUp(self) -> None:
        """Initializes a temporary sandbox directory and generates a mock SFZ structure."""
        self.test_dir: str = tempfile.mkdtemp()
        self.tmp_path: Path = Path(self.test_dir)

        # Construct a realistic mock sample audio asset to satisfy existence checks
        self.mock_sample: Path = self.tmp_path / "piano_c4.wav"
        self.mock_sample.touch()

        # Build raw text content containing both compliant and non-compliant rules
        self.sfz_lines = [
            "<group>\n",
            "ampeg_attack=0.15\n",  # Latency hazard: must be forced to 0.0
            f"sample={self.mock_sample.name}\n",  # Valid path configuration
            "<region>\n",
            "ampeg_attack = 0.0\n",  # Already safe: must remain untouched
            "sample=missing_file.wav\n",  # Invalid path: must trigger dependency fault
        ]

        self.source_sfz: Path = self.tmp_path / "instrument.sfz"
        with open(self.source_sfz, "w", encoding="utf-8") as f:
            f.writelines(self.sfz_lines)

        self.output_sfz: Path = self.tmp_path / "instrument_safe.sfz"

    def tearDown(self) -> None:
        """Cleans up the file system sandbox after test execution completes."""
        shutil.rmtree(self.test_dir)

    def test_conditioner_rewrites_attack_values(self) -> None:
        """Verifies that non-zero envelope attacks are forced to zero while valid tracks pass."""
        # Remove the missing sample line to test the pure text-fixing mechanics safely
        safe_source: Path = self.tmp_path / "safe_instrument.sfz"
        with open(safe_source, "w", encoding="utf-8") as f:
            f.writelines(self.sfz_lines[:-1])  # Exclude the missing file line

        conditioner = SfzConditioner(sfz_path=safe_source)
        success = conditioner.process_and_fix(output_path=self.output_sfz)

        self.assertTrue(success)

        with open(self.output_sfz, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("ampeg_attack=0.0", content)
        self.assertIn("ampeg_attack = 0.0", content)


if __name__ == "__main__":
    unittest.main()
