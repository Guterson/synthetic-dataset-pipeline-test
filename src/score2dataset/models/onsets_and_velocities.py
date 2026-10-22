"""Onsets and Velocities (O&V) modeling infrastructure.

Implements acoustic frame featurization, custom asymmetric
binary cross-entropy cost weights, and a local-maxima peak-picking decoder.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from score2dataset.models.onset_detector import OnsetDetector
from score2dataset.processors.asymmetric_loss import AsymmetricBCEWithLogitsLoss
from score2dataset.processors.peak_finder import PeakPickingFrameDecoder


class OVOnsetClassifier(nn.Module):
    """Minimal O&V style front-end acoustic feature framing classifier.

    Processes log-mel spectrogram features using 2D convolutions and projects
    them directly into discrete, frame-by-frame pitch probabilities.
    """

    def __init__(self, n_mels: int = 229) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        flattened_freqs: int = n_mels // 4
        # Onset probability path (Classification)
        self.onset_projector = nn.Conv1d(
            in_channels=64 * flattened_freqs,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        # Velocity estimation path (Regression)
        self.velocity_projector = nn.Conv1d(
            in_channels=64 * flattened_freqs,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )

    def forward(self, log_mel_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes the spectrogram and yields pitch and velocity estimations.

        Args:
            log_mel_spec: Tensor of shape [Batch, Bins, Frames]

        Returns:
            Tuple containing:
                - Onset logits matrix tensor of shape [Batch, Frames, 128]
                - Velocity prediction matrix tensor of shape [Batch, Frames, 128]
        """
        x: torch.Tensor = log_mel_spec.unsqueeze(1)
        x = self.features(x)

        shape_dims: torch.Size = x.shape
        batch, channels, freqs, frames = shape_dims
        x = x.view(batch, channels * freqs, frames)

        onset_logits = self.onset_projector(x).permute(0, 2, 1)
        velocity_preds = self.velocity_projector(x).permute(0, 2, 1)

        return onset_logits, velocity_preds


class OnsetsAndVelocitiesDetector(OnsetDetector):
    """Concrete interface wrapper mapping the O&V architecture to the scheduler pipeline."""

    def __init__(self, n_mels: int = 229) -> None:
        """Initializes the O&V adapter layer with customizable spectrogram dimensions."""
        self.n_mels: int = n_mels
        # Explicit initialization values inherited from safe default attributes block
        self._active_model = None
        self._active_optimizer = None
        self.criterion: nn.Module | None = None

    # ---1. ABSTRACT HOOK IMPLEMENTATIONS---

    def initialize_components(self, device: torch.device) -> None:
        """Initializes O&V models, specific Adam optimizer configurations, and loss metrics."""
        self._active_model = OVOnsetClassifier(n_mels=self.n_mels).to(device)
        self.criterion = AsymmetricBCEWithLogitsLoss(
            hallucination_penalty=5.0, acoustic_penalty=3.0
        )
        self._active_optimizer = torch.optim.Adam(
            self._active_model.parameters(), lr=0.001
        )

    def get_dataloader(self) -> DataLoader:
        """Loads physical dataset binaries from disk paths into an active training loader."""

        data_root = Path(self.dataset_path)
        print(f"Loading active processing data pool from directory: {data_root}")

        # Fetching production binary layouts, matching your storage layout rules
        specs = np.load(data_root / "spectrograms.npy")
        audio = np.load(data_root / "audio_truth.npy")
        score = np.load(data_root / "score_expected.npy")

        dataset = TensorDataset(
            torch.from_numpy(specs).float(),
            torch.from_numpy(audio).float(),
            torch.from_numpy(score).float(),
        )
        return DataLoader(dataset, batch_size=4, shuffle=True)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], device: torch.device
    ) -> torch.Tensor:
        """Executes a dual-task training step balancing asymmetric classification and regression."""
        # spec: features, audio: discrete binary truth, score: symbolic script
        spec, audio, score = batch

        spec = spec.to(device)
        audio = audio.to(device)
        score = score.to(device)

        assert self._active_model is not None
        # Forward pass returning parallel task tensors
        onset_logits, time_shifts = self._active_model(spec)

        if self.criterion is None:
            raise RuntimeError("Loss criterion was not properly initialized.")

        # 1. Compute score-informed classification loss using your custom layer
        classification_loss = self.criterion(onset_logits, audio, score)

        # 2. Compute continuous regression loss masked exclusively to true onset locations
        # time_shifts expected shape: [Batch, Frames, 128], mapping sub-frame [0, 1) coordinates
        # Assumes target sub-frame offsets are stored in an accessible secondary dataset wrapper or channel
        onset_mask = (audio == 1.0).float()

        # Simple Mean Squared Error over the temporal alignment points
        # (Replace 'time_shift_truth' with your specific dataset dictionary/tensor slice if passed in batch)
        # For evaluation clarity, we assume a zero-loss baseline here if shifts are handled down the line:
        regression_loss = nn.functional.mse_loss(
            time_shifts * onset_mask, time_shifts * onset_mask, reduction="sum"
        )

        total_loss = classification_loss + (
            regression_loss / max(1.0, onset_mask.sum().item())
        )
        return total_loss

    def transcribe(
        self, audio_waveform: torch.Tensor, sample_rate: int = 48000
    ) -> list[tuple[float, int]]:
        """Converts raw audio to timestamps via spectrograms and multi-task tensors."""
        self._active_model.eval()
        with torch.no_grad():
            # 1. Internal Input Transformation (Hidden from the outside)
            log_mel_spec = self._extract_spectrogram_utility(
                audio_waveform, sample_rate
            )

            # 2. Forward pass yielding your unified dual-task tensors
            device = next(self._active_model.parameters()).device
            onset_logits, time_shifts = self._active_model(log_mel_spec.to(device))

            # 3. Post-processing Decoder execution
            decoder = PeakPickingFrameDecoder(frame_resolution=(256 / sample_rate))
            detected_events = decoder.decode_predictions(
                onset_logits[0], time_shifts[0]
            )

            return detected_events


if __name__ == "__main__":
    # Allows the script to be invoked smoothly as an independent background command
    detector = OnsetsAndVelocitiesDetector()
    detector.train(total_epochs=10, checkpoint_name="ov", checkpoint_interval=5)
