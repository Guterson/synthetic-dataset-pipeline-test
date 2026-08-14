"""Data curation tool to compute time-aligned log-mel feature spectrograms.

This module acts as an isolated preparation pipeline utility. It ingests raw
high-resolution time-domain waveforms and transforms them into fixed 24ms
logarithmic mel-frequency energy blocks required by the onset detector network models.
"""

import numpy as np
from pathlib import Path

import torch
import torchaudio


def extract_log_mel_features(
    audio_path: Path, n_mels: int = 229, sample_rate: int = 48000
) -> np.ndarray:
    """Transforms a raw audio wave file into a fixed-resolution mel spectrogram.

    Args:
        audio_path: Strongly typed Path leading to the raw wav file.
        n_mels: Number of logarithmic target frequency bins.
        sample_rate: Expected target sampling resolution of the system.

    Returns:
        np.ndarray: Matrix tensor of shape [229, total_frames] containing float32 log values.
    """
    # 1. Read waveform data out from storage disk
    waveform, native_sr = torchaudio.load(str(audio_path))

    # Mix down to single channel mono if file records multi-channel data
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Enforce standard tracking sample rate
    if native_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=native_sr, new_freq=sample_rate
        )
        waveform = resampler(waveform)

    # 2. Configure time-aligned physical framing resolution
    # 0.024s * 48000Hz = 1152 samples hop length
    hop_length = int(0.024 * sample_rate)
    # Use standard 2x window factor to capture clean harmonic overlaps
    n_fft = hop_length * 2

    # 3. Apply the mathematical transform pipelines
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
    )

    # Generate linear mel amplitudes and convert to logarithmic decibel ranges
    mel_spectrogram = mel_transform(waveform)
    log_mel_spec = torch.log(torch.clamp(mel_spectrogram, min=1e-5))

    # Strip batch/channel dimension and export as simple primitive numpy array
    return log_mel_spec.squeeze(0).numpy()


if __name__ == "__main__":
    # Test execution illustrating conversion of a single asset file
    target_audio = Path("data/raw_downloads/1st_baptist_nashville_balcony.wav")
    output_dir = Path("data/assets/rir")

    if target_audio.exists():
        features = extract_log_mel_features(target_audio)
        np.save(output_dir / "nashville_balcony_specs.npy", features)
        print(f"Success! Spectrogram shape generated completely: {features.shape}")
