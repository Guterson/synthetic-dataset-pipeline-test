"""Onsets and Frames (O&F) multi-task neural network architecture.

Implements independent parallel convolutional streams for onset and frame
activation prediction, utilizing onset masks as explicit frame gating criteria,
wrapped in an automated scheduler-compliant lifecycle engine.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Project-wide infrastructure imports
from score2dataset.models.onset_detector import OnsetDetector
from score2dataset.processors.asymmetric_loss import AsymmetricBCEWithLogitsLoss


class AcousticFrontEndHead(nn.Module):
    """Standardized 2D CNN block used to featurize log-mel spectrogram arrays."""

    def __init__(self, n_mels: int = 229) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        flattened_freqs = n_mels // 4
        self.projector = nn.Conv1d(64 * flattened_freqs, 128, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial acoustic maps from a raw spectrogram tensor."""
        x = x.unsqueeze(1)  # Add channel dim -> [Batch, 1, Bins, Frames]
        x = self.conv_stack(x)

        batch, channels, freqs, frames = x.shape
        x = x.view(batch, channels * freqs, frames)
        return self.projector(x)


class OnsetsAndFramesClassifier(nn.Module):
    """Multi-task network managing parallel detection heads and gating layers."""

    def __init__(self, n_mels: int = 229) -> None:
        super().__init__()
        self.onset_head = AcousticFrontEndHead(n_mels=n_mels)
        self.frame_head = AcousticFrontEndHead(n_mels=n_mels)

        # Sequential Bidirectional LSTM to track temporal dependencies across frames
        self.temporal_recurrent_layer = nn.LSTM(
            input_size=128 + 128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pitch_projector = nn.Linear(256, 128)

    def forward(self, log_mel_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes acoustic inputs and outputs parallel multi-task probability grids.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Onset and frame logit tensors.
        """
        onset_features = self.onset_head(log_mel_spec).permute(0, 2, 1)
        frame_features = self.frame_head(log_mel_spec).permute(0, 2, 1)

        combined_features = torch.cat([frame_features, onset_features], dim=-1)
        lstm_out, _ = self.temporal_recurrent_layer(combined_features)
        frame_logits = self.pitch_projector(lstm_out)

        return onset_features, frame_logits


class OnsetsAndFramesDetector(OnsetDetector):
    """Concrete interface wrapper mapping the O&F architecture to the scheduler pipeline."""

    def __init__(self, n_mels: int = 229) -> None:
        """Initializes the multi-task O&F tracking adapter."""
        self.n_mels = n_mels
        # Explicit base class attribute alignments
        self._active_model = None
        self._active_optimizer = None
        self.criterion: nn.Module | None = None

    @property
    def dataset_path(self) -> str:
        """Returns the targeted dataset file path required for this model.

        Returns:
            str: Hardcoded path string pointing to the dedicated feature dataset.
        """
        return "data/processed/onsets_and_frames_dataset"

    # ---1. ABSTRACT HOOK IMPLEMENTATIONS---

    def initialize_components(self, device: torch.device) -> None:
        """Initializes O&F networks, multi-task criteria, and specific optimization configurations."""
        self._active_model = OnsetsAndFramesClassifier(n_mels=self.n_mels).to(device)
        self.criterion = AsymmetricBCEWithLogitsLoss(
            hallucination_penalty=5.0, acoustic_penalty=3.0
        )
        self._active_optimizer = torch.optim.Adam(
            self._active_model.parameters(), lr=0.0006
        )

    def get_dataloader(self) -> DataLoader:
        """Loads concrete multi-task array matrices from specified disk locations."""
        data_root = Path(self.dataset_path)
        print(f"Loading true audio data pool from directory: {data_root}")

        specs = np.load(data_root / "spectrograms.npy")
        audio = np.load(data_root / "audio_truth.npy")
        score = np.load(data_root / "score_expected.npy")

        dataset = TensorDataset(
            torch.from_numpy(specs).float(),
            torch.from_numpy(audio).float(),
            torch.from_numpy(score).float(),
        )
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], device: torch.device
    ) -> torch.Tensor:
        """Executes a parallel multi-task forward pass and merges loss conditions."""
        spec_batch, audio_batch, score_batch = batch

        spec_batch = spec_batch.to(device)
        audio_batch = audio_batch.to(device)
        score_batch = score_batch.to(device)

        # Satisfy strict linting narrowing boundaries via type assertions
        assert self._active_model is not None
        assert self.criterion is not None

        # Compute parallel multi-task outputs
        onset_preds, frame_preds = self._active_model(spec_batch)

        # Apply asymmetric cost evaluations to both tracking masks
        loss_onset = self.criterion(onset_preds, audio_batch, score_batch)
        loss_frame = self.criterion(frame_preds, audio_batch, score_batch)

        # Return composite loss sum directly back to your engine
        return loss_onset + loss_frame


if __name__ == "__main__":
    detector = OnsetsAndFramesDetector()
    detector.train(total_epochs=50, checkpoint_name="of", checkpoint_interval=5)
