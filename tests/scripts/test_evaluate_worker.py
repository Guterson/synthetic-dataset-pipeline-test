"""Automated pipeline validation checking metric accuracy and worker mechanics.

Validates timeline metric formulas and unifications without executing
active remote SSH connections or heavy network hardware allocations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Target functional units matching source package structures
from score2dataset.scripts.evaluate_worker import (
    calculate_unified_mir_score,
    run_isolated_inference,
)

# ---1. ACCURACY CALCULATION AND BOUNDARY TESTS---


def test_mir_score_matches_perfect_predictions_completely():
    """Verifies that flawless prediction alignment yields a unified score of 1.0."""
    gt = [(0.50, 64), (1.20, 72)]
    pred = [(0.50, 64), (1.20, 72)]

    metrics = calculate_unified_mir_score(pred, gt, tolerance_ms=25.0, tau_ms=10.0)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mae_ms"] == 0.0
    assert metrics["unified_score"] == 1.0


def test_mir_score_drops_on_pitch_mismatch_anomalies():
    """Verifies that predicted times are ignored completely if pitch attributes diverge."""
    gt = [(1.00, 60)]
    pred = [(1.00, 61)]  # Timing maps identically, but notes do not align

    metrics = calculate_unified_mir_score(pred, gt)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["unified_score"] == 0.0


# ---2. WORKER HOOK UNIFICATION AND DECODING TESTS---


@patch("importlib.import_module")
@patch("pathlib.Path.exists")
def test_worker_inference_handles_convolutional_peak_picking_loops(
    mock_exists, mock_import
):
    """Verifies that worker feeds standard inputs safely through convolutional adapters."""
    mock_exists.return_value = True

    # Assemble mock model layout outputs
    mock_detector = MagicMock()
    mock_dataloader = [
        torch.randn(1, 100, 128)
    ]  # Simulated inference batch tensor stream
    mock_detector.get_dataloader.return_value = mock_dataloader

    # Configure mock network forward responses
    mock_net = MagicMock()
    mock_net.return_value = torch.zeros(1, 100, 128)
    mock_detector.get_active_model.return_value = mock_net

    # Bind the detector mock instance directly to our dynamic loader factory mock
    mock_module = MagicMock()
    mock_module.OnsetsAndVelocitiesDetector = lambda: mock_detector
    mock_import.return_value = mock_module

    with patch(
        "score2dataset.models.onsets_and_velocities.PeakPickingFrameDecoder"
    ) as mock_decoder_cls:
        decoder_instance = MagicMock()
        decoder_instance.decode_predictions.return_value = [(0.24, 60)]
        mock_decoder_cls.return_value = decoder_instance

        results = run_isolated_inference(
            "OnsetsAndVelocities", Path("fake/data"), torch.device("cpu")
        )

        assert len(results) == 1
        assert results[0] == (0.24, 60)
        assert mock_net.called
