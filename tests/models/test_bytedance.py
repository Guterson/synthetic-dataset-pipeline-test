"""Functional verification suite for the ByteDance High-Resolution Regression Model.

Validates parallel classification and continuous regression output dimensions,
tanh scaling bounds, and gradient propagation through the shared convolutional backbone.
"""

import torch

from score2dataset.models.bytedance import ByteDanceRegressionModel


def test_bytedance_regression_output_dimensions() -> None:
    """Verify that the network cleanly returns separate, aligned classification and regression tensor grids."""
    # Instantiated at the canonical 128 Mel-bins high-resolution blueprint configuration
    model = ByteDanceRegressionModel(n_mels=128)
    model.eval()

    # Simulate a batch of 2 audio files: 128 log-mel bins across 150 frames (10ms dense resolution)
    mock_spec = torch.randn(2, 128, 150)

    with torch.no_grad():
        onset_logits, offset_regression = model(mock_spec)

    # Both heads must match the exact same timeline scale bounds [Batch, Frames, 128 Piano Pitches]
    assert onset_logits.shape == (2, 150, 128)
    assert offset_regression.shape == (2, 150, 128)


def test_regression_sub_frame_offset_bounds() -> None:
    """Business Rule: Continuous regression offsets must be strictly bounded between -0.5 and +0.5.

    Ensures extreme forward layer activations cannot saturate or break sub-frame
    decoding boundaries during continuous timestamp mapping.
    """
    model = ByteDanceRegressionModel(n_mels=128)
    model.eval()

    # Create extreme input values to try and force the regression head to saturate its outputs
    extreme_spec = torch.clamp(
        torch.randn(1, 128, 50) * 1000.0, min=-5000.0, max=5000.0
    )

    with torch.no_grad():
        _, offset_regression = model(extreme_spec)

    # 1. Assert that every predicted value falls strictly within the sub-frame boundaries
    assert torch.all(offset_regression >= -0.5)
    assert torch.all(offset_regression <= 0.5)

    # 2. Verify that values are not absolute flat integers (retains fractional float data precision)
    assert torch.any((offset_regression > -0.5) & (offset_regression < 0.5))


def test_shared_backbone_gradient_propagation() -> None:
    """Verify that backpropagated gradients flow cleanly through both distinct task heads simultaneously.

    Ensures that optimizing the continuous time offsets modifies the shared front-end
    features without locking or blocking the parallel onset classification graph.
    """
    model = ByteDanceRegressionModel(n_mels=128)
    model.train()  # Activate training registers to unblock autograd hooks

    mock_spec = torch.randn(1, 128, 80, requires_grad=True)
    onset_logits, offset_regression = model(mock_spec)

    # Initialize separate mock targets tracking classification and continuous regression bounds
    classification_targets = torch.zeros_like(onset_logits)
    regression_targets = torch.zeros_like(offset_regression)

    # Instantiate joint multi-task loss components
    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    loss_class = bce_loss(onset_logits, classification_targets)
    loss_reg = mse_loss(offset_regression, regression_targets)

    # Combined backward execution pass
    total_loss = loss_class + loss_reg
    total_loss.backward()

    # 1. Verify that the primary input tracking tensor successfully accumulated gradients
    assert mock_spec.grad is not None
    assert mock_spec.grad.shape == (1, 128, 80)

    # 2. Assert that parameters within the shared backbone collect updates cleanly
    backbone_weight = next(model.backbone.conv_stack.parameters())
    assert backbone_weight.grad is not None
