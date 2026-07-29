"""Digital Signal Processing (DSP) acoustic convolution and audio formatting engine.

This module provides high-performance frequency-domain convolution algorithms
to apply spatial room acoustic characteristics to dry synthetic audio renders,
enforcing a unified 48kHz, 16-bit mono PCM output constraint optimized for onset detection.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from score2dataset.exceptions import ProcessorError


class RirConvolver:
    """Applies acoustic physical space characteristics and normalizes bit depth maps."""

    def __init__(self, target_sr: int = 48000) -> None:
        """Initializes the audio normalization engine configurations.

        Args:
            target_sr: Final output sample rate resolution forced across all
                processed audio files. Defaults to 48000.
        """
        self.target_sr: int = target_sr

    def process_audio(
        self, dry_wav_path: Path, rir_path: Path, output_wav_path: Path
    ) -> None:
        """Transforms a dry render into a spatially convolved 48kHz, 16-bit mono audio file.

        Executes high-speed mono frequency domain array multiplications, handles high-fidelity
        resampling arrays, and preserves velocity dynamic variations accurately.

        Args:
            dry_wav_path: Path location pointing to the source raw audio render file.
            rir_path: Path to the specific Room Impulse Response WAV file to apply.
            output_wav_path: Target destination path to store the processed wet file.

        Raises:
            ProcessorError: If any DSP matrix operation or array transformation fails.
        """
        try:
            dry_data, dry_sr = sf.read(dry_wav_path, dtype="float64")
            rir_data, rir_sr = sf.read(rir_path, dtype="float64")

            dry_mono: np.ndarray = self._to_mono(dry_data)
            rir_mono: np.ndarray = self._to_mono(rir_data)

            working_sr: int = max(dry_sr, rir_sr)
            dry_matched: np.ndarray = self._resample(dry_mono, dry_sr, working_sr)
            rir_matched: np.ndarray = self._resample(rir_mono, rir_sr, working_sr)

            rir_normalized: np.ndarray = self._normalize_rir_peak(rir_data=rir_matched)

            wet_signal: np.ndarray = fftconvolve(
                dry_matched, rir_normalized, mode="full"
            )

            clipped_signal: np.ndarray = np.clip(wet_signal, -1.0, 1.0)
            pcm16_data: np.ndarray = (clipped_signal * 32767.0).astype(dtype=np.int16)

            sf.write(
                file=output_wav_path,
                data=pcm16_data,
                samplerate=self.target_sr,
                subtype="PCM_16",
            )

        except Exception as error:
            raise ProcessorError(
                f"DSP Pipeline Failure: Convolution aborted. Details: {error}"
            ) from error

    def _to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """Downmixes multi-channel audio arrays to mono via mean averaging."""
        if audio_data.ndim > 1:
            return np.mean(audio_data, axis=1)
        return audio_data

    def _resample(
        self, audio_data: np.ndarray, current_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resamples audio data between two custom sample rates using polyphase filtering."""
        if current_sr == target_sr:
            return audio_data

        gcd: int = np.gcd(target_sr, current_sr)
        up_factor: int = target_sr // gcd
        down_factor: int = current_sr // gcd

        return resample_poly(audio_data, up_factor, down_factor)

    def _normalize_rir_peak(self, rir_data: np.ndarray) -> np.ndarray:
        """Normalizes the Room Impulse Response peak amplitude to exactly 1.0."""
        rir_peak: float = float(np.max(np.abs(rir_data)))
        if rir_peak > 0.0:
            return rir_data / rir_peak
        return rir_data
