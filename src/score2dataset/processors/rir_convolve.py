"""Digital Signal Processing (DSP) acoustic convolution and audio formatting engine.

This module provides high-performance frequency-domain convolution algorithms
to apply spatial room acoustic characteristics to dry synthetic audio renders,
enforcing a unified 48kHz, 16-bit mono PCM output constraint optimized for onset detection.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


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
            RuntimeError: If any DSP matrix operation or array transformation fails.
        """
        try:
            # 1. Load data as 64-bit floats between -1.0 and 1.0
            dry_data, dry_sr = sf.read(str(dry_wav_path))
            rir_data, rir_sr = sf.read(str(rir_path))

            # 2. Force Pure Mono Downmixing Immediately
            if len(dry_data.shape) > 1:
                dry_data = np.mean(dry_data, axis=1)

            if len(rir_data.shape) > 1:
                rir_data = np.mean(rir_data, axis=1)

            # 3. High-Fidelity Resampling to Target 48kHz
            if dry_sr != self.target_sr:
                gcd = np.gcd(self.target_sr, dry_sr)
                dry_data = resample_poly(dry_data, self.target_sr // gcd, dry_sr // gcd)

            if rir_sr != self.target_sr:
                gcd = np.gcd(self.target_sr, rir_sr)
                rir_data = resample_poly(rir_data, self.target_sr // gcd, rir_sr // gcd)

            # 4. Normalize the RIR Only (Preserves original dry audio velocity/volume)
            rir_peak = np.max(np.abs(rir_data))
            if rir_peak > 0:
                rir_data = rir_data / rir_peak

            # 5. Mono Frequency-Domain Overlap-Save Convolution
            wet_data = fftconvolve(dry_data, rir_data, mode="full")

            # 6. Safety Headroom Clip
            # We do NOT normalize to 1.0 here. We only apply a hard ceiling safety clip
            # to protect the system, though files should naturally maintain their MIDI
            # velocity curves.
            wet_data = np.clip(wet_data, -1.0, 1.0)

            # 7. Fixed Bit Depth Scaling (64-bit float -> 16-bit signed Int PCM)
            pcm16_data = (wet_data * 32767.0).astype(np.int16)

            # Write the completed 16-bit mono file directly to disk
            sf.write(str(output_wav_path), pcm16_data, self.target_sr, subtype="PCM_16")

        except Exception as error:
            raise RuntimeError(
                f"DSP Pipeline Failure: Convolution aborted. Details: {error}"
            ) from error
