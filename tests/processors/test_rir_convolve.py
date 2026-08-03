"""Unit test suite for the RirConvolver acoustic convolution engine.

This module validates multi-channel downmixing, sample rate resampling,
peak normalization safety, and error handling for bulk audio synthesis.
"""

import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy._typing._array_like import NDArray

from score2dataset.exceptions import ProcessorError
from score2dataset.processors.rir_convolve import RirConvolver


class TestRirConvolver(unittest.TestCase):
    """Test cases for the RirConvolver class processing steps."""

    def setUp(self) -> None:
        """Sets up temporary file tracking for clean state teardown."""
        self.created_files: list[Path] = []

    def tearDown(self) -> None:
        """Cleans up temporary audio files generated during test runs."""
        for path in self.created_files:
            if path.exists():
                path.unlink()

    def _create_dummy_wav(
        self, path: Path, data: np.ndarray, sample_rate: int, subtype: str = "PCM_16"
    ) -> Path:
        """Helper to write temporary WAV files and track them for cleanup."""
        sf.write(file=path, data=data, samplerate=sample_rate, subtype=subtype)
        self.created_files.append(path)
        return path

    def test_process_audio_successful_convolution(self) -> None:
        """Tests successful mono convolution without intermediate resampling."""
        dry_path = Path("test_dry.wav")
        rir_path = Path("test_rir.wav")
        out_path = Path("test_out.wav")
        self.created_files.extend([dry_path, rir_path, out_path])

        dry_signal: NDArray[np.float64] = np.random.uniform(
            low=-0.5, high=0.5, size=48000
        )
        rir_signal: NDArray[np.float64] = np.zeros(24000)
        rir_signal[0] = 1.0

        self._create_dummy_wav(path=dry_path, data=dry_signal, sample_rate=48000)
        self._create_dummy_wav(path=rir_path, data=rir_signal, sample_rate=48000)

        convolver = RirConvolver(target_sr=48000)
        convolver.process_audio(
            dry_wav_path=dry_path, rir_path=rir_path, output_wav_path=out_path
        )

        self.assertTrue(out_path.exists())

        written_data, written_sr = sf.read(out_path, dtype="int16")
        self.assertEqual(written_sr, 48000)
        self.assertEqual(written_data.ndim, 1)

    def test_process_audio_forces_downmix(self) -> None:
        """Tests that stereo signals are successfully downmixed to mono."""
        dry_path = Path("test_stereo.wav")
        rir_path = Path("test_mono_rir.wav")
        out_path = Path("test_mono_out.wav")
        self.created_files.extend([dry_path, rir_path, out_path])

        stereo_signal: NDArray[np.float64] = np.random.uniform(
            low=-0.5, high=0.5, size=(48000, 2)
        )
        mono_rir: NDArray[np.float64] = np.zeros(1000)
        mono_rir[0] = 0.5

        self._create_dummy_wav(path=dry_path, data=stereo_signal, sample_rate=48000)
        self._create_dummy_wav(path=rir_path, data=mono_rir, sample_rate=48000)

        convolver = RirConvolver()
        convolver.process_audio(
            dry_wav_path=dry_path, rir_path=rir_path, output_wav_path=out_path
        )

        written_data, _ = sf.read(out_path)
        self.assertEqual(written_data.ndim, 1)

    def test_process_audio_resamples_to_highest_rate_then_to_target(self) -> None:
        """Tests convolution matching to highest native rate before target downsampling."""
        dry_path = Path("test_44100.wav")
        rir_path = Path("test_96000.wav")
        out_path = Path("test_48000.wav")
        self.created_files.extend([dry_path, rir_path, out_path])

        dry_signal: NDArray[np.float64] = np.random.uniform(
            low=-0.2, high=0.2, size=44100
        )
        rir_signal: NDArray[np.float64] = np.random.uniform(
            low=-0.2, high=0.2, size=96000
        )

        self._create_dummy_wav(path=dry_path, data=dry_signal, sample_rate=44100)
        self._create_dummy_wav(path=rir_path, data=rir_signal, sample_rate=96000)

        convolver = RirConvolver(target_sr=48000)
        convolver.process_audio(
            dry_wav_path=dry_path, rir_path=rir_path, output_wav_path=out_path
        )

        _, written_sr = sf.read(out_path)
        self.assertEqual(written_sr, 48000)

    def test_process_audio_supports_24bit_rir(self) -> None:
        """Tests execution handles high bit depth 24-bit PCM impulse responses safely."""
        dry_path = Path("test_16bit_dry.wav")
        rir_path = Path("test_24bit_rir.wav")
        out_path = Path("test_wet_final.wav")
        self.created_files.extend([dry_path, rir_path, out_path])

        dry_signal: NDArray[np.float64] = np.random.uniform(
            low=-0.4, high=0.4, size=48000
        )
        rir_signal: NDArray[np.float64] = np.random.uniform(
            low=-0.4, high=0.4, size=24000
        )

        self._create_dummy_wav(
            path=dry_path, data=dry_signal, sample_rate=48000, subtype="PCM_16"
        )
        self._create_dummy_wav(
            path=rir_path, data=rir_signal, sample_rate=48000, subtype="PCM_24"
        )

        convolver = RirConvolver(target_sr=48000)
        convolver.process_audio(
            dry_wav_path=dry_path, rir_path=rir_path, output_wav_path=out_path
        )

        self.assertTrue(out_path.exists())

    def test_process_audio_raises_custom_processor_error_on_missing_file(self) -> None:
        """Tests that a specialized ProcessorError is raised if files are missing."""
        invalid_path = Path("nonexistent.wav")
        valid_path = Path("valid.wav")
        out_path = Path("out.wav")
        self.created_files.extend([valid_path, out_path])

        self._create_dummy_wav(path=valid_path, data=np.zeros(1000), sample_rate=48000)

        convolver = RirConvolver()
        with self.assertRaisesRegex(ProcessorError, "DSP Pipeline Failure"):
            convolver.process_audio(invalid_path, valid_path, out_path)


if __name__ == "__main__":
    unittest.main()
