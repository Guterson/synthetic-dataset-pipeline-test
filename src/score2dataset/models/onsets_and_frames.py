"""Onsets and Frames (O&F) multi-task neural network architecture.

Implements independent parallel convolutional streams for onset and frame
activation prediction, utilizing onset masks as explicit frame gating criteria.
"""

import torch
from torch import nn


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
        # 229 // 4 = 57 frequency bins remaining after pooling
        flattened_freqs = n_mels // 4
        self.projector = nn.Conv1d(64 * flattened_freqs, 128, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Bins, Frames]
        x = x.unsqueeze(1)  # Add channel dim -> [Batch, 1, Bins, Frames]
        x = self.conv_stack(x)

        batch, channels, freqs, frames = x.shape
        x = x.view(batch, channels * freqs, frames)

        # Output shape: [Batch, 128, Frames]
        return self.projector(x)


class OnsetsAndFramesClassifier(nn.Module):
    """Multi-task network managing parallel detection heads and gating layers."""

    def __init__(self, n_mels: int = 229) -> None:
        super().__init__()
        # Parallel, structurally isolated feature extractors
        self.onset_head = AcousticFrontEndHead(n_mels=n_mels)
        self.frame_head = AcousticFrontEndHead(n_mels=n_mels)

        # A simple linear projection layer to merge onset memory into frame predictions
        self.fusion_layer = nn.Linear(128 + 128, 128)

    def forward(self, log_mel_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes acoustic inputs and outputs parallel multi-task probability grids.

        Returns:
            onset_logits: Shape [Batch, Frames, 128]
            frame_logits: Shape [Batch, Frames, 128]
        """
        # Execute parallel forward passes over the shared spectrogram timeline
        # Outputs arrive as [Batch, 128, Frames] -> Permute instantly to [Batch, Frames, 128]
        onset_features = self.onset_head(log_mel_spec).permute(0, 2, 1)
        frame_features = self.frame_head(log_mel_spec).permute(0, 2, 1)

        # 💡 The Core O&F Trick: Concatenate features across the pitch dimension
        # We combine raw frame features with the onset features to enforce gating constraints
        combined_features = torch.cat([frame_features, onset_features], dim=-1)

        # Project back down to our canonical 128 standard MIDI piano pitch size
        frame_logits = self.fusion_layer(combined_features)

        return onset_features, frame_logits
