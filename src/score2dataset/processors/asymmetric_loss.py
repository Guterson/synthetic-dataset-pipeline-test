"""Custom domain cost function metrics for training robust onset detectors.

This module provides specialized loss layers designed to evaluate neural network
predictions against multiple ground-truth reference frames. It isolates custom
asymmetric penalties to prevent models from overfitting to symbolic music scores
when training on synthesized audio datasets.
"""

import torch
from torch import nn


class AsymmetricBCEWithLogitsLoss(nn.Module):
    """Custom asymmetric loss function penalizing score-memory dependency.

    Evaluates frame predictions by heavily scaling costs where the model
    hallucinates notes that were expected by the score but omitted in the audio.
    """

    def __init__(
        self, hallucination_penalty: float = 5.0, acoustic_penalty: float = 3.0
    ) -> None:
        """Initializes the asymmetric penalty weights.

        Args:
            hallucination_penalty: Weight multiplier for false positive score hallucination.
            acoustic_penalty: Weight multiplier for unwritten real audio events.
        """
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
        """Computes localized weighted loss over the entire tensor matrix batch.

        Args:
            predictions: Logits matrix of shape [Batch, Frames, 128]
            audio_present: Target matrix of shape [Batch, Frames, 128] (Acoustic reality)
            score_expected: Structure matrix of shape [Batch, Frames, 128] (Score layout)

        Returns:
            torch.Tensor: A single scalar tracking the mean weighted cross-entropy loss.
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
