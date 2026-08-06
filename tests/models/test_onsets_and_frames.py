"""Functional verification suite for the multi-task Onsets and Frames (O&F) architecture.

Validates parallel CNN feature extraction heads, time-series permutation tracking,
dual-head logit matrix dimensions, and gradient flow through the fusion gating layer.
"""

import torch

from score2dataset.models.onsets_and_frames import OnsetsAndFramesClassifier


def test_onsets_and_frames_dual_output_shapes() -> None:
    """Verify the network cleanly returns separate, synchronized onset and frame tracking grids."""
    model = OnsetsAndFramesClassifier(n_mels=229)
    model.eval()

    # Simulate 1 audio window: 229 log-mel bins across exactly 200 temporal frames (32ms resolution)
    mock_spec = torch.randn(1, 229, 200)

    with torch.no_grad():
        onset_logits, frame_logits = model(mock_spec)

    # Verify both heads match the exact same timeline scale bounds [Batch, Frames, Pitches]
    assert onset_logits.shape == (1, 200, 128)
    assert frame_logits.shape == (1, 200, 128)


def test_acoustic_head_pooling_boundaries() -> None:
    """Ensure the underlying pooling math collapses frequency scales while protecting time steps."""
    model = OnsetsAndFramesClassifier(n_mels=229)

    # Verify that changing the number of timeline frames scales the output length linearly
    mock_short_spec = torch.randn(1, 229, 50)
    mock_long_spec = torch.randn(1, 229, 150)

    with torch.no_grad():
        onset_short, _ = model(mock_short_spec)
        onset_long, _ = model(mock_long_spec)

    assert onset_short.shape[1] == 50
    assert onset_long.shape[1] == 150


def test_multi_task_gradient_flow_and_gating() -> None:
    """Business Rule: Frame activations must structurally integrate onset feature weights.

    Verifies that gradients flow uninterrupted through both parallel convolutional
    heads back to the shared spectrogram input during a simulated training step.
    """
    model = OnsetsAndFramesClassifier(n_mels=229)
    model.train()  # Activate training mode to unblock gradient tracking registers

    # Simulate a single input requiring gradient tracking
    mock_spec = torch.randn(1, 229, 100, requires_grad=True)

    onset_logits, frame_logits = model(mock_spec)

    # Create dummy targets matching output dimensions [Batch, Frames, Pitches]
    dummy_onset_targets = torch.zeros_like(onset_logits)
    dummy_frame_targets = torch.zeros_like(frame_logits)

    # Compute a combined mock loss mimicking your L_total = L_onset + L_frame step
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_onset = loss_fn(onset_logits, dummy_onset_targets)
    loss_frame = loss_fn(frame_logits, dummy_frame_targets)
    total_loss = loss_onset + loss_frame

    # Execute backpropagation
    total_loss.backward()

    # 1. Verify that the shared input tensor successfully collected backpropagated gradients
    assert mock_spec.grad is not None
    assert mock_spec.grad.shape == (1, 229, 100)

    # 2. Verify that both underlying independent heads contributed to the gradient step
    for name, param in model.named_parameters():
        assert (
            param.grad is not None
        ), f"Gradient breakdown detected: parameter '{name}' received no updates."
