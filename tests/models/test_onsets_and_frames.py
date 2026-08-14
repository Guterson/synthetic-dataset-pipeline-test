"""Automated verification suite testing the Onsets and Frames multi-task pipeline.

Validates parallel convolutional recurrent neural shape grids, multi-task linear
loss summation steps, and concrete initialization hook compliance contracts.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from score2dataset.models.onsets_and_frames import (
    OnsetsAndFramesClassifier,
    OnsetsAndFramesDetector,
)

# ---1. MULTI-TASK NEURAL SHAPE MATRIX TESTS---


@pytest.mark.parametrize("batch_size, n_mels, frames", [(1, 229, 30), (4, 229, 64)])
def test_multi_task_classifier_yields_parallel_tensor_tuples(
    batch_size, n_mels, frames
):
    """Verifies that the dual front-end heads and LSTM yield correct matching output shapes."""
    model = OnsetsAndFramesClassifier(n_mels=n_mels)
    model.eval()

    # Structural fake input specs matrix array: [Batch, MelBins, Frames]
    fake_specs = torch.randn(batch_size, n_mels, frames)

    with torch.no_grad():
        onset_logits, frame_logits = model(fake_specs)

    # Verify both heads map to [Batch, Frames, 128] independently
    assert onset_logits.shape == (batch_size, frames, 128)
    assert frame_logits.shape == (batch_size, frames, 128)
    assert onset_logits.dtype == torch.float32
    assert frame_logits.dtype == torch.float32


# ---2. MULTI-TASK HOOK STEP LIFECYCLE TESTS---


def test_multi_task_training_step_blends_losses_correctly():
    """Verifies that training_step unpacks the batch and sums parallel head costs."""
    detector = OnsetsAndFramesDetector()
    device = torch.device("cpu")

    # 💡 ACCURATE FIX: Use mocking boundaries via patch.object contexts to satisfy Pyright
    mock_classifier = MagicMock(spec=OnsetsAndFramesClassifier)
    mock_classifier.return_value = (torch.randn(2, 40, 128), torch.randn(2, 40, 128))

    mock_criterion = MagicMock()
    mock_criterion.side_effect = [
        torch.tensor(0.5),
        torch.tensor(0.8),
    ]  # Return distinct loss scalars

    # Inject variables through mocking context frames safely
    with (
        patch.object(detector, "_active_model", mock_classifier),
        patch.object(detector, "criterion", mock_criterion),
    ):

        # Multi-task batch signature shape mapping
        batch = (
            torch.randn(2, 229, 40),  # spec_batch
            torch.zeros(2, 40, 128),  # audio_truth_batch
            torch.ones(2, 40, 128),  # score_expected_batch
        )

        composite_loss = detector.training_step(batch, device)

        assert isinstance(composite_loss, torch.Tensor)
        assert np.isclose(composite_loss.item(), 1.3)  # 0.5 (onset) + 0.8 (frame) = 1.3
        assert mock_classifier.called
        assert mock_criterion.call_count == 2


# ---3. DATA LOADING INFRASTRUCTURE TESTS---


@patch("numpy.load")
def test_multi_task_dataloader_ingests_and_types_matrices(mock_np_load):
    """Verifies that the data hook fetches files from the dedicated O&F directory allocation."""
    detector = OnsetsAndFramesDetector()

    # Mask disk array metrics cleanly
    mock_np_load.side_effect = [
        np.zeros((4, 229, 50)),  # spectrograms.npy
        np.zeros((4, 50, 128)),  # audio_truth.npy
        np.zeros((4, 50, 128)),  # score_expected.npy
    ]

    # Prove that the unique model directory path contract maps perfectly
    assert detector.dataset_path == "data/processed/onsets_and_frames_dataset"

    loader = detector.get_dataloader()
    assert isinstance(loader, DataLoader)

    batch = next(iter(loader))
    assert len(batch) == 3
    # Check that batch configurations utilize batch size 8 as requested by construction rules
    assert loader.batch_size == 8
