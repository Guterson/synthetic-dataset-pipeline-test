"""Digital Signal Processing utilities for Room Impulse Response conditioning.

Provides high-performance structural array manipulation tools to eliminate
leading propagation delays and truncate ambient decay floors below noise thresholds
to prepare acoustic signals for zero-latency convolution.
"""

from pathlib import Path

import numpy as np
import soundfile as sf  # Type: ignore(import)

from score2dataset.exceptions import ProcessorError


class RirConditioner:
    """Removes propagation delay windows and truncates decaying noise floors from RIR arrays."""

    def __init__(self, noise_threshold_db: float = -60.0) -> None:
        """Initializes the structural conditioning parameters.

        Args:
            noise_threshold_db: Decibel floor below which acoustic energy is
                considered ambient noise. Defaults to -60.0.
        """
        self.noise_threshold: float = 10 ** (noise_threshold_db / 20.0)

    def condition_file(self, input_path: Path, output_path: Path) -> None:
        """Trims leading propagation delay and trailing noise from an RIR file.

        Args:
            input_path: Source location of the raw, unconditioned RIR file.
            output_path: Destination target to write the zero-latency conditioned file.

        Raises:
            ProcessorError: If the file is blank, corrupted, or parsing fails.
        """
        try:
            data, sample_rate = sf.read(input_path, dtype="float64")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            absolute_signal: np.ndarray = np.abs(data)
            max_peak_value: float = float(np.max(absolute_signal))

            if max_peak_value < 1e-4:
                raise ProcessorError(
                    f"Structural Validation Fault: File is completely blank: {input_path.name}"
                )

            # Locate the direct-path onset index to eliminate time-of-flight latency
            onset_index: int = int(np.argmax(absolute_signal))

            # Scan backwards to truncate trailing noise below the operational decibel floor
            active_mask = absolute_signal > self.noise_threshold
            if not np.any(active_mask):
                end_index: int = len(data)
            else:
                end_index = int(np.max(np.where(active_mask))) + 1

            if onset_index >= end_index:
                end_index = len(data)

            conditioned_data: np.ndarray = data[onset_index:end_index]

            sf.write(
                file=output_path,
                data=conditioned_data,
                samplerate=sample_rate,
                subtype="PCM_16",
            )

        except Exception as error:
            if isinstance(error, ProcessorError):
                raise
            raise ProcessorError(
                f"RIR Conditioning Failure on file {input_path.name}: {error}"
            ) from error
