"""Unit tests for the RirConditioner processing class using the unittest framework."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from score2dataset.exceptions import ProcessorError
from score2dataset.processors.rir_conditioner import RirConditioner


class TestRirConditioner(unittest.TestCase):
    """Verifies architectural and algorithmic integrity of the RirConditioner class."""

    def setUp(self) -> None:
        """Initializes a temporary sandbox directory and generates a raw synthetic RIR file."""
        self.test_dir: str = tempfile.mkdtemp()
        self.tmp_path: Path = Path(self.test_dir)
        self.sample_rate: int = 44100

        # Construct a 0.1-second artificial propagation delay (4410 frames)
        leading_silence: np.ndarray = np.zeros(4410)

        # Construct an instantaneous peak followed by an exponential decay window
        time_axis: np.ndarray = np.linspace(0, 0.5, 22050)
        decay_wave: np.ndarray = np.exp(-10.0 * time_axis)

        # Construct a trailing noise floor falling cleanly below -60dB
        trailing_noise: np.ndarray = np.random.normal(0, 0.0001, 8820)

        # Compile components into an unconditioned composite signal matrix
        self.raw_signal: np.ndarray = np.concatenate(
            [leading_silence, decay_wave, trailing_noise]
        )

        self.synthetic_rir_file: Path = self.tmp_path / "raw_synthetic_rir.wav"
        sf.write(
            file=self.synthetic_rir_file,
            data=self.raw_signal,
            samplerate=self.sample_rate,
            subtype="PCM_16",
        )

        self.output_path: Path = self.tmp_path / "conditioned_output.wav"
        self.conditioner: RirConditioner = RirConditioner(noise_threshold_db=-60.0)

    def tearDown(self) -> None:
        """Cleans up the file system sandbox after test execution completes."""
        shutil.rmtree(self.test_dir)

    def test_conditioner_removes_leading_latency(self) -> None:
        """Verifies that the direct-path onset index becomes the new absolute start frame."""
        self.conditioner.condition_file(
            input_path=self.synthetic_rir_file, output_path=self.output_path
        )

        conditioned_data, _ = sf.read(self.output_path)

        # The maximum absolute peak must reside exactly at index 0
        self.assertEqual(int(np.argmax(np.abs(conditioned_data))), 0)

    def test_conditioner_truncates_trailing_noise(self) -> None:
        """Verifies that samples falling permanently below the noise floor are pruned."""
        self.conditioner.condition_file(
            input_path=self.synthetic_rir_file, output_path=self.output_path
        )

        conditioned_data, _ = sf.read(self.output_path)

        # The processed output array must be structurally shorter than the source file
        self.assertLess(len(conditioned_data), len(self.raw_signal))

    def test_conditioner_handles_blank_file_exception(self) -> None:
        """Verifies that a completely flat silent array securely raises a ProcessorError."""
        blank_path: Path = self.tmp_path / "blank.wav"
        sf.write(
            file=blank_path,
            data=np.zeros(8000),
            samplerate=self.sample_rate,
            subtype="PCM_16",
        )

        with self.assertRaises(ProcessorError):
            self.conditioner.condition_file(
                input_path=blank_path, output_path=self.output_path
            )


if __name__ == "__main__":
    unittest.main()
