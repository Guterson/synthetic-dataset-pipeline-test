"""Automated verification suite testing the Onsets and Velocities pipeline layers.

Validates 2D/1D convolutional feature tracking shape boundaries, peak-picking
local maxima decoding algorithms, and concrete model contract hook integrations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from score2dataset.models.onsets_and_velocities import (
    OnsetsAndVelocitiesDetector,
    OVOnsetClassifier,
    PeakPickingFrameDecoder,
)

# ---1. NEURAL ARCHITECTURE SHAPE MATRIX TESTS---


@pytest.mark.parametrize(
    "batch_size, n_mels, frames", [(1, 229, 50), (4, 229, 100), (2, 229, 12)]
)
def test_classifier_outputs_exact_expected_tensor_shapes(batch_size, n_mels, frames):
    """Verifies that the convolutional layers and pitch projector reshape tensors accurately."""
    model = OVOnsetClassifier(n_mels=n_mels)
    model.eval()

    # Generate pure structural fake input specs: [Batch, MelBins, Frames]
    fake_specs = torch.randn(batch_size, n_mels, frames)

    with torch.no_grad():
        outputs = model(fake_specs)

    assert outputs.shape == (batch_size, frames, 128)
    assert outputs.dtype == torch.float32


# ---2. LOCAL MAXIMA DECODER MATHEMATICAL BOUNDARY TESTS---


def test_peak_picker_extracts_clean_local_maxima_events():
    """Verifies that the peak picker correctly registers peaks above threshold boundaries."""
    decoder = PeakPickingFrameDecoder(threshold=0.5, frame_resolution=0.024)

    # Setup mock logit matrix: [Frames=5, Pitches=1]
    # Frame 2 is a perfect isolated peak above 0.5 (sigmoid context)
    fake_logits = torch.tensor([[-2.0], [-1.0], [2.0], [-1.0], [-2.0]])

    events = decoder.decode_predictions(fake_logits)

    assert len(events) == 1
    assert events[0] == (2 * 0.024, 0)  # Frame index 2 mapped to pitch channel 0


def test_peak_picker_rejects_plateaus_and_low_confidence_nodes():
    """Verifies that uniform values and points below threshold do not trigger fake events."""
    decoder = PeakPickingFrameDecoder(threshold=0.5, frame_resolution=0.024)

    # Pitch 0: Uniform plateau curve values above threshold (strict '>' check should reject)
    # Pitch 1: Peak exists, but its value remains completely below the 0.5 threshold
    fake_logits = torch.tensor([[2.0, -5.0], [2.0, -2.0], [2.0, -5.0]])

    events = decoder.decode_predictions(fake_logits)
    assert len(events) == 0


# ---3. CONCRETE HOOK CONTRACT COMPLIANCE TESTS---


def test_training_step_unpacks_and_computes_cost_correctly():
    """Verifies that the child adapter executes the forward pass and returns scalar losses."""
    detector = OnsetsAndVelocitiesDetector()
    device = torch.device("cpu")

    mock_model = MagicMock()
    mock_model.return_value = torch.randn(2, 50, 128)
    mock_criterion = MagicMock(return_value=torch.tensor(1.23))

    with (
        patch.object(detector, "_active_model", mock_model),
        patch.object(detector, "criterion", mock_criterion),
    ):

        # Build valid 3-tensor signature matching production requirements
        batch = (
            torch.randn(2, 229, 50),  # spec
            torch.zeros(2, 50, 128),  # audio
            torch.ones(2, 50, 128),  # score
        )

        loss = detector.training_step(batch, device)
        assert isinstance(loss, torch.Tensor)

    # Build valid 3-tensor signature matching production requirements
    batch = (
        torch.randn(2, 229, 50),  # spec
        torch.zeros(2, 50, 128),  # audio
        torch.ones(2, 50, 128),  # score
    )

    loss = detector.training_step(batch, device)

    assert isinstance(loss, torch.Tensor)
    assert mock_model.called
    assert mock_criterion.called


def test_training_step_guards_against_uninitialized_loss_layer():
    """Verifies that uninitialized criteria drop explicit runtime handling errors."""
    detector = OnsetsAndVelocitiesDetector()

    # Secure type narrowing context and return a valid structural tracking tensor
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 100, 128)

    with (
        patch.object(detector, "_active_model", mock_model),
        patch.object(detector, "criterion", None),
    ):  # Intentionally leave loss empty

        batch = (
            torch.zeros(1, 229, 100),
            torch.zeros(1, 100, 128),
            torch.zeros(1, 100, 128),
        )

        with pytest.raises(
            RuntimeError, match="Loss criterion was not properly initialized"
        ):
            detector.training_step(batch, torch.device("cpu"))


@patch("numpy.load")
def test_get_dataloader_fetches_and_types_disk_matrices(mock_np_load):
    """Verifies that the data loader hook reads the correct 3 files and structures data."""
    detector = OnsetsAndVelocitiesDetector()

    # Setup mock binary returns to isolate disk reads completely
    mock_np_load.side_effect = [
        np.zeros((2, 229, 20)),  # spectrograms
        np.zeros((2, 20, 128)),  # audio_truth
        np.zeros((2, 20, 128)),  # score_expected
    ]

    with patch.object(OnsetsAndVelocitiesDetector, "dataset_path", "mock/data/root"):
        loader = detector.get_dataloader()

        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert len(batch) == 3
        assert batch[0].dtype == torch.float32
