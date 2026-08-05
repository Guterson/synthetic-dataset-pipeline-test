"""ByteDance high-resolution regression model architecture for piano transcription.

Implements a shared acoustic convolutional front-end that splits into parallel
discrete frame classification and continuous sub-frame time offset regression heads.
"""

import torch
from torch import nn


class ByteDanceRegressionFrontEnd(nn.Module):
    """Shared 2D CNN backbone collapsing frequency bins into a flat time-series."""

    def __init__(self, n_mels: int = 128) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Pull down frequencies, preserve time
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        # 128 // 4 = 32 frequency bins remaining after pooling layers
        flattened_freqs = n_mels // 4
        self.feature_combiner = nn.Conv1d(
            64 * flattened_freqs, 256, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape expected: [Batch, Bins, Frames]
        x = x.unsqueeze(1)  # Add explicit channel dim -> [Batch, 1, Bins, Frames]
        x = self.conv_stack(x)

        batch, channels, freqs, frames = x.shape
        x = x.view(batch, channels * freqs, frames)

        # Output shape returned: [Batch, 256, Frames]
        return self.relu(self.feature_combiner(x))


class ByteDanceRegressionModel(nn.Module):
    """Multi-head regression network tracking exact sub-frame performance offsets."""

    def __init__(self, n_mels: int = 128) -> None:
        super().__init__()
        self.backbone = ByteDanceRegressionFrontEnd(n_mels=n_mels)

        # Head 1: Standard classification layer tracking IF an onset exists in a frame
        self.classification_head = nn.Conv1d(256, 128, kernel_size=3, padding=1)

        # Head 2: Continuous regression layer predicting the floating-point offset (-0.5 to 0.5)
        self.regression_head = nn.Conv1d(256, 128, kernel_size=3, padding=1)

    def forward(self, log_mel_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes acoustic inputs and outputs aligned classification and regression grids.

        Args:
            log_mel_spec: Tensor matrix tracking [Batch, Bins, Frames]

        Returns:
            onset_logits: Discrete indicator tensor of shape [Batch, Frames, 128]
            offset_regression: Continuous offset tensor of shape [Batch, Frames, 128]
        """
        # Extract unified features from the shared convolutional backbone
        shared_features = self.backbone(log_mel_spec)

        # Compute forward steps over individual tasks and permute output dimensions
        # to standard sequence orientation -> [Batch, Frames, 128 Piano Pitches]
        onset_logits = self.classification_head(shared_features).permute(0, 2, 1)

        # Tanh activation restricts continuous regression outputs between -1.0 and 1.0.
        # We multiply by 0.5 to strictly map offsets within the target -0.5 to +0.5 sub-frame bounds.
        offset_raw = self.regression_head(shared_features).permute(0, 2, 1)
        offset_regression = 0.5 * torch.tanh(offset_raw)

        return onset_logits, offset_regression
