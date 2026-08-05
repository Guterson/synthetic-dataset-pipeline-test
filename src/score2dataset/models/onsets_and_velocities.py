"""Onsets and Velocities (O&V) modeling infrastructure.

Implements acoustic frame featurization at 24ms resolution, custom asymmetric
binary cross-entropy cost weights, and a local-maxima peak-picking decoder.
"""

import torch
from torch import nn


class OVOnsetClassifier(nn.Module):
    """Minimal O&V style front-end acoustic feature framing classifier.

    Processes log-mel spectrogram features using 2D convolutions and projects
    them directly into discrete, frame-by-frame pitch probabilities.
    """

    def __init__(self, n_mels: int = 229) -> None:
        super().__init__()
        # Input shape expected: [Batch, 1, n_mels, Frames]
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Reduce frequency bins, keep frames raw
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Flatted frequency space calculation: 229 // 4 = 57 bins remaining
        flattened_freqs = n_mels // 4
        self.pitch_projector = nn.Conv1d(
            in_channels=64 * flattened_freqs,
            out_channels=128,  # Predict across all 128 standard MIDI piano keys
            kernel_size=3,
            padding=1,
        )

    def forward(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        """Processes the spectrogram and yields pitch frame logits.

        Args:
            log_mel_spec: Tensor of shape [Batch, Bins, Frames]

        Returns:
            Logits matrix tensor of shape [Batch, Frames, 128]
        """
        # Append explicit channel dimension for 2D convolutions
        x = log_mel_spec.unsqueeze(1)
        x = self.features(x)

        # Reshape to combine channels and frequency bins into a flat time-series
        batch, channels, freqs, frames = x.shape
        x = x.view(batch, channels * freqs, frames)

        # Project across the MIDI spectrum and permute back to standard frame sequence
        logits = self.pitch_projector(x)
        return logits.permute(0, 2, 1)


class AsymmetricBCEWithLogitsLoss(nn.Module):
    """Custom asymmetric loss function penalizing score-memory dependency.

    Evaluates frame predictions by heavily scaling costs where the model
    hallucinates notes that were expected by the score but omitted in the audio.
    """

    def __init__(
        self, hallucination_penalty: float = 5.0, acoustic_penalty: float = 3.0
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.w_hallucination = hallucination_penalty
        self.w_acoustic = acoustic_penalty

    def forward(
        self,
        predictions: torch.Tensor,
        audio_present: torch.Tensor,
        score_expected: torch.Tensor,
    ) -> torch.Tensor:
        """Computes weighted loss.

        Args:
            predictions: Logits matrix of shape [Batch, Frames, 128]
            audio_present: Target matrix of shape [Batch, Frames, 128] (Acoustic reality)
            score_expected: Structure matrix of shape [Batch, Frames, 128] (Score layout)
        """
        # Calculate raw individual cross-entropy losses
        base_loss = self.bce(predictions, audio_present)

        # Initialize dynamic weight mask matching the shape boundaries
        loss_weights = torch.ones_like(base_loss)

        # Mask 1: Hallucination -> Note was in the score, but missing in the audio wave
        hallucination_mask = (score_expected == 1.0) & (audio_present == 0.0)
        loss_weights[hallucination_mask] *= self.w_hallucination

        # Mask 2: Unwritten Acoustic Event -> Note wasn't in the score, but physically played
        acoustic_mask = (score_expected == 0.0) & (audio_present == 1.0)
        loss_weights[acoustic_mask] *= self.w_acoustic

        # Apply the final localized penalty scaling and average over elements
        weighted_loss = base_loss * loss_weights
        return weighted_loss.mean()


class PeakPickingFrameDecoder:
    """Decodes frame probabilities into discrete onset events using local maxima rules."""

    def __init__(self, threshold: float = 0.5, frame_resolution: float = 0.024) -> None:
        self.threshold = threshold
        self.frame_res = frame_resolution

    def decode_predictions(self, raw_logits: torch.Tensor) -> list[tuple[float, int]]:
        """Applies peak-picking filters across each pitch channel sequentially.

        Args:
            raw_logits: Model outputs for a single song file tracking tensor [Frames, 128]

        Returns:
            A list tracking tuples of (absolute_onset_seconds, midi_pitch)
        """
        # Apply sigmoid activation to transition from raw logits to true 0.0-1.0 probabilities
        probabilities = torch.sigmoid(raw_logits).detach().cpu().numpy()
        n_frames, n_pitches = probabilities.shape

        detected_onsets = []

        for pitch in range(n_pitches):
            pitch_curve = probabilities[:, pitch]

            for frame in range(1, n_frames - 1):
                prob_value = pitch_curve[frame]

                # Filter Condition 1: Must cross baseline detection boundary
                if (
                    prob_value > self.threshold
                    and prob_value > pitch_curve[frame - 1]
                    and prob_value > pitch_curve[frame + 1]
                ):

                    # Convert discrete frame coordinate back to absolute elapsed time
                    onset_seconds = float(frame * self.frame_res)
                    detected_onsets.append((onset_seconds, pitch))

        # Ensure final list is sorted chronologically
        detected_onsets.sort(key=lambda x: x[0])
        return detected_onsets
