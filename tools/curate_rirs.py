"""Automated asset curation pipeline for Room Impulse Response files.

This module provides a standalone utility class designed to parse unstructured,
raw acoustic data downloads, isolate valid deconvolved impulse waveforms via
structural peak profiling, and stage them for the rendering pipeline.
"""

from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

from score2dataset.processors.rir_conditioner import RirConditioner


class RirCurator:
    """Filters, validates, and reorganizes raw Room Impulse Response audio files."""

    FORBIDDEN_KEYWORDS: Final[set[str]] = {
        "sweep",
        "excitation",
        "raw",
        "b-format",
        "ambisonic",
        "quad",
    }

    def __init__(self, raw_dir: Path, target_dir: Path) -> None:
        """Initializes the asset processing directory boundaries.

        Args:
            raw_dir: Path to the unstructured directory containing unzipped raw assets.
            target_dir: Destination path where validated pure RIRs will be staged.
        """
        self.raw_dir: Path = raw_dir
        self.target_dir: Path = target_dir
        self.conditioner: RirConditioner = RirConditioner(noise_threshold_db=-60.0)

    def execute_curation(self) -> list[Path]:
        """Scans the source directory and populates the target directory with pure RIR files.

        Returns:
            A list of Path locations pointing to the newly staged, validated assets.
        """
        self.target_dir.mkdir(parents=True, exist_ok=True)
        curated_paths: list[Path] = []

        for wav_path in self.raw_dir.rglob("*.wav"):
            if self._has_invalid_metadata(file_name=wav_path.name):
                continue

            if not self._is_pure_impulse(file_path=wav_path):
                continue

            destination_path: Path = self.target_dir / wav_path.name
            self.conditioner.condition_file(
                input_path=wav_path, output_path=destination_path
            )
            curated_paths.append(destination_path)

        return curated_paths

    def _has_invalid_metadata(self, file_name: str) -> bool:
        """Checks if the file name contains keywords indicating an unprocessed sound trigger."""
        normalized_name: str = file_name.lower()
        return any(keyword in normalized_name for keyword in self.FORBIDDEN_KEYWORDS)

    def _is_pure_impulse(self, file_path: Path) -> bool:
        """Verifies if the acoustic signal exhibits an instantaneous direct arrival spike.

        Validates the file structure against physical Time-of-Flight (ToF) limits.
        A direct path arrival exceeding 0.5 seconds implies a source-to-receiver
        distance greater than 171.5 meters (assuming c = 343 m/s), indicating
        an un-deconvolved excitation sweep rather than a clean impulse response.
        """
        try:
            audio_data, sample_rate = sf.read(file_path)
            absolute_signal: np.ndarray = np.abs(audio_data)
            max_peak_sample: int = int(np.argmax(absolute_signal))

            half_second_threshold: int = int(sample_rate * 0.5)
            return max_peak_sample <= half_second_threshold

        except (RuntimeError, ValueError):
            return False


if __name__ == "__main__":
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    curator = RirCurator(
        raw_dir=PROJECT_ROOT / "data" / "raw_downloads",
        target_dir=PROJECT_ROOT / "data" / "assets" / "rir",
    )

    processed_assets: list[Path] = curator.execute_curation()
    print(f"Asset curation complete. Staged {len(processed_assets)} pure RIR tracks.")
