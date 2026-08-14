"""Automated verification suite testing the ByteDance high-resolution pipeline.

Validates parallel classification-regression neural shapes, 4-tensor dataset
ingestion structures, and linearly weighted mixed-objective step calculations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from score2dataset.models.bytedance import (
    ByteDanceDetector,
    ByteDanceRegressionModel,
)

# ---1. HYBRID REGRESSION NEURAL GRAPH SHAPE TESTS---


@pytest.mark.parametrize("batch_size, n_mels, frames", [(1, 128, 40), (2, 128, 80)])
def test_regression_model_outputs_aligned_prediction_grids(batch_size, n_mels, frames):
    """Verifies that the shared backbone correctly splits into dual heads with exact shapes."""
    # Instantiating the internal model graph
    model = ByteDanceRegressionModel(n_mels=n_mels)
    model.eval()

    # Generate mock log-mel spectrum array tokens: [Batch, MelBins, Frames]
    fake_specs = torch.randn(batch_size, n_mels, frames)

    with torch.no_grad():
        onset_logits, offset_regression = model(fake_specs)

    # Both heads must align completely to track the subframe time metrics cleanly
    assert onset_logits.shape == (batch_size, frames, 128)
    assert offset_regression.shape == (batch_size, frames, 128)

    # Verify that the continuous regression activation mapping limits stay inside bounds [-0.5, 0.5]
    assert torch.all(offset_regression >= -0.5)
    assert torch.all(offset_regression <= 0.5)


# ---2. LINEARLY WEIGHTED DUAL-LOSS LIFECYCLE TESTS---


def test_dual_objective_training_step_scales_regression_loss_accurately():
    """Verifies that training_step unpacks 4 tensors and weights the MSE regression head by 2.0."""
    detector = ByteDanceDetector()
    device = torch.device("cpu")

    mock_model = MagicMock()
    mock_model.return_value = (torch.randn(2, 30, 128), torch.randn(2, 30, 128))

    mock_criterion_cls = MagicMock(return_value=torch.tensor(0.60))
    mock_criterion_reg = MagicMock(return_value=torch.tensor(0.15))

    # Secure test isolation contexts to bypass MethodType constraints safely
    with (
        patch.object(detector, "_active_model", mock_model),
        patch.object(detector, "criterion_cls", mock_criterion_cls),
        patch.object(detector, "criterion_reg", mock_criterion_reg),
    ):

        # Assemble production-grade 4-tensor dataset loop structure
        batch = (
            torch.randn(2, 128, 30),  # spec_batch
            torch.zeros(2, 30, 128),  # audio_batch
            torch.ones(2, 30, 128),  # score_batch
            torch.randn(2, 30, 128),  # offset_batch (Microtiming regression target)
        )

        blended_loss = detector.training_step(batch, device)

        assert isinstance(blended_loss, torch.Tensor)
        # Mathematical verification: loss_cls + (2.0 * loss_reg) -> 0.60 + (2.0 * 0.15) = 0.90
        assert np.isclose(blended_loss.item(), 0.90)
        assert mock_model.called
        assert mock_criterion_cls.called
        assert mock_criterion_reg.called


# ---3. 4-TENSOR DATA LOADING INFRASTRUCTURE TESTS---


@patch("numpy.load")
def test_regression_dataloader_structures_all_four_binary_matrices(mock_np_load):
    """Verifies that the data factory maps four files cleanly and sets a batch size of 8."""
    detector = ByteDanceDetector()

    # Mock all 4 targets matching your extraction loops
    mock_np_load.side_effect = [
        np.zeros((8, 128, 40)),  # spectrograms.npy
        np.zeros((8, 40, 128)),  # audio_truth.npy
        np.zeros((8, 40, 128)),  # score_expected.npy
        np.zeros((8, 40, 128)),  # subframe_offsets.npy
    ]

    with patch.object(ByteDanceDetector, "dataset_path", "mock/bytedance/paths"):
        loader = detector.get_dataloader()
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 8

        batch = next(iter(loader))
        assert len(batch) == 4  # Confirms completeness of the 4-tensor split structure
        assert (
            batch[3].dtype == torch.float32
        )  # Verifies regression targets cast properly
