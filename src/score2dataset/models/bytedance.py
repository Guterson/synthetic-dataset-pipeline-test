"""ByteDance high-resolution regression model architecture for piano transcription.

Implements a shared acoustic convolutional front-end that splits into parallel
discrete frame classification and continuous sub-frame time offset regression heads,
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


class ByteDanceRegressionFrontEnd(nn.Module):
    """Shared 2D CNN backbone collapsing frequency bins into a flat time-series."""

    def __init__(self, n_mels: int = 128) -> None:
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
        self.feature_combiner = nn.Conv1d(
            64 * flattened_freqs, 256, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts and collapses spatial maps into a unified time series."""
        x = x.unsqueeze(1)
        x = self.conv_stack(x)

        batch, channels, freqs, frames = x.shape
        x = x.view(batch, channels * freqs, frames)
        return self.relu(self.feature_combiner(x))


class ByteDanceRegressionModel(nn.Module):
    """Multi-head regression network tracking exact sub-frame performance offsets."""

    def __init__(self, n_mels: int = 128) -> None:
        super().__init__()
        self.backbone = ByteDanceRegressionFrontEnd(n_mels=n_mels)
        self.classification_head = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.regression_head = nn.Conv1d(256, 128, kernel_size=3, padding=1)

    def forward(self, log_mel_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes acoustic inputs and outputs aligned classification and regression grids."""
        shared_features = self.backbone(log_mel_spec)

        onset_logits = self.classification_head(shared_features).permute(0, 2, 1)
        offset_raw = self.regression_head(shared_features).permute(0, 2, 1)
        offset_regression = 0.5 * torch.tanh(offset_raw)

        return onset_logits, offset_regression


class ByteDanceDetector(OnsetDetector):
    """Concrete interface wrapper mapping the ByteDance architecture to the scheduler."""

    def __init__(self, n_mels: int = 229) -> None:
        """Initializes the multi-head dual-objective ByteDance adapter."""
        self.n_mels = n_mels
        # Explicit modern class property typing declarations
        self._active_model = None
        self._active_optimizer = None
        self.criterion_cls: torch.nn.Module | None = None
        self.criterion_reg: torch.nn.Module | None = None

    @property
    def dataset_path(self) -> str:
        """Returns the targeted dataset file path required for this model.

        Returns:
            str: Hardcoded path string pointing to the dedicated feature dataset.
        """
        return "data/processed/bytedance_regression_dataset"

    # ---1. ABSTRACT HOOK IMPLEMENTATIONS---

    def initialize_components(self, device: torch.device) -> None:
        """Initializes ByteDance networks, classification/regression criteria, and optimizer states."""
        self._active_model = ByteDanceRegressionModel(n_mels=self.n_mels).to(device)
        self.criterion_cls = AsymmetricBCEWithLogitsLoss(
            hallucination_penalty=5.0, acoustic_penalty=3.0
        )
        self.criterion_reg = nn.MSELoss()
        self._active_optimizer = torch.optim.Adam(
            self._active_model.parameters(), lr=0.0005
        )

    def get_dataloader(self) -> DataLoader:
        """Loads physical 4-tensor dataset matrix binaries from local storage blocks."""
        data_root = Path(self.dataset_path)
        print(f"Loading synthetic audio data pool from directory: {data_root}")

        specs = np.load(data_root / "spectrograms.npy")
        audio = np.load(data_root / "audio_truth.npy")
        score = np.load(data_root / "score_expected.npy")
        subframe_offsets = np.load(data_root / "subframe_offsets.npy")

        dataset = TensorDataset(
            torch.from_numpy(specs).float(),
            torch.from_numpy(audio).float(),
            torch.from_numpy(score).float(),
            torch.from_numpy(subframe_offsets).float(),
        )
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], device: torch.device
    ) -> torch.Tensor:
        """Executes a composite forward pass blending classification and regression margins."""
        # Cleanly unpack the specific 4-tensor variable structure signature
        spec_batch, audio_batch, score_batch, offset_batch = batch

        spec_batch = spec_batch.to(device)
        audio_batch = audio_batch.to(device)
        score_batch = score_batch.to(device)
        offset_batch = offset_batch.to(device)

        # Enforce strict compiler type safety narrowing via assertions
        assert self._active_model is not None
        assert self.criterion_cls is not None
        assert self.criterion_reg is not None

        # Forward pass yields discrete classification logits and continuous regression matrix
        onset_preds, offset_preds = self._active_model(spec_batch)

        # Head 1 Evaluation: Asymmetric categorization cost
        loss_cls = self.criterion_cls(onset_preds, audio_batch, score_batch)

        # Head 2 Evaluation: Continuous time displacement error matrix tracking
        loss_reg = self.criterion_reg(offset_preds, offset_batch)

        # Return blended loss matrix sum directly back to the training container engine
        return loss_cls + (2.0 * loss_reg)


if __name__ == "__main__":
    detector = ByteDanceDetector()
    detector.train(total_epochs=50, checkpoint_name="bd", checkpoint_interval=5))
