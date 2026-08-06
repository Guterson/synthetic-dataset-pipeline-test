"""Functional verification suite for the Onsets and Velocities (O&V) modeling infrastructure.

Validates 2D-to-1D convolutional channel projection shapes, asymmetric gradient
masking bounds for hallucination tracking, and local-maxima peak-picking frame decoding.
"""

import pytest
import torch

from score2dataset.models.onsets_and_velocities import (
    AsymmetricBCEWithLogitsLoss,
    OVOnsetClassifier,
    PeakPickingFrameDecoder,
)


def test_classifier_output_tensor_shapes() -> None:
    """Verify that the 2D CNN successfully collapses frequency bins into a flat time series."""
    # Instantiated at the canonical 229 Mel-bins footprint resolution
    model = OVOnsetClassifier(n_mels=229)
    model.eval()

    # Simulate a batch of 2 featurized tracks: 229 bins across exactly 100 temporal frames
    mock_spec = torch.randn(2, 229, 100)

    with torch.no_grad():
        logits = model(mock_spec)

    # The network must return a 3D grid tracking [Batch, Frames, 128 Piano Keys]
    assert logits.shape == (2, 100, 128)


def test_asymmetric_loss_hallucination_penalty() -> None:
    """Business Rule: Score-memory hallucinations must trigger heavier penalties than baseline errors."""
    loss_fn = AsymmetricBCEWithLogitsLoss(
        hallucination_penalty=5.0, acoustic_penalty=3.0
    )

    # Establish an identical baseline layout space: Batch=1, Frames=1, Pitch=1
    # Logits = 2.0 implies a strong model prediction probability (~88% confidence)
    mock_logits = torch.tensor([[[2.0]]], dtype=torch.float32)

    # Scenario A: Standard Match Error (The note was missing in BOTH score and audio)
    audio_present_a = torch.tensor([[[0.0]]], dtype=torch.float32)
    score_expected_a = torch.tensor([[[0.0]]], dtype=torch.float32)
    loss_standard = loss_fn(mock_logits, audio_present_a, score_expected_a)

    # Scenario B: Score Hallucination (The model predicted it because it memorized the score,
    # but the human pianist actually omitted the note -> score_expected=1, audio_present=0)
    audio_present_b = torch.tensor([[[0.0]]], dtype=torch.float32)
    score_expected_b = torch.tensor([[[1.0]]], dtype=torch.float32)
    loss_hallucination = loss_fn(mock_logits, audio_present_b, score_expected_b)

    # The asymmetric mask must scale the cost function output precisely by 5.0x
    assert torch.isclose(loss_hallucination, loss_standard * 5.0)


def test_peak_picking_decoder_heuristics() -> None:
    """Verify the decoder enforces threshold limits and local maxima boundaries cleanly."""
    # Configured to standard 24ms resolution properties
    decoder = PeakPickingFrameDecoder(threshold=0.5, frame_resolution=0.024)

    # Generate a custom probability curve across a single pitch stream (10 frames, 1 pitch)
    # Target Peak is located at Frame index 5 (0.85 probability)
    mock_probabilities = torch.tensor(
        [
            [0.1],
            [0.2],
            [0.4],
            [0.45],
            [0.49],
            [0.85],  # Frame 5: Crosses threshold and represents a true local maximum
            [0.82],  # Frame 6: Crosses threshold but is falling (not a local maximum)
            [0.3],
            [0.1],
            [0.0],
        ],
        dtype=torch.float32,
    )

    # Convert probabilities backward into raw logits to bypass the decoder's sigmoid activation step
    # Equation inverse: logits = log(p / (1 - p))
    mock_logits = torch.log(mock_probabilities / (1.0 - mock_probabilities))

    # Run the structural parsing heuristic
    detected_events = decoder.decode_predictions(mock_logits)

    # The tracker must capture exactly 1 single onset, completely ignoring the trailing offset frame 6
    assert len(detected_events) == 1

    onset_time, pitch_index = detected_events[0]
    assert pitch_index == 0
    # Absolute calculated time must equal Frame Index 5 * 24ms = 0.120 seconds
    assert pytest.approx(onset_time) == 0.120
