"""Automated verification suite testing the Magenta Seq2Seq Transformer pipeline.

Validates autoregressive attention masking matrices, mixed-type sequence data ingestion,
and dimensional collapsing sequence loss calculations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from score2dataset.models.google_magenta import (
    MagentaTranscriptionTransformer,
    MagentaTransformerDetector,
    PositionalEncoding,
)

# ---1. TRANSFORMER LAYER COMPONENT TESTS---


def test_causal_mask_generator_creates_upper_triangular_bounds():
    """Verifies that the casual mask contains 0.0 on diagonal/lower and -inf on future blocks."""
    model = MagentaTranscriptionTransformer(n_mels=128)
    sequence_length = 4
    device = torch.device("cpu")

    mask = model.generate_square_subsequent_mask(sequence_length, device)

    # Assert structural layout properties
    assert mask.shape == (sequence_length, sequence_length)
    assert mask.dtype == torch.float32

    # Diagonal and lower bounds must allow attention (0.0 logit offset)
    assert mask[0, 0] == 0.0
    assert mask[1, 0] == 0.0

    # Upper bounds (future indices) must block attention (-inf logit offset)
    assert mask[0, 1] == float("-inf")
    assert mask[1, 3] == float("-inf")


def test_positional_encoding_preserves_tensor_shapes():
    """Verifies that sinusoidal positional waves inject data without warping vector sizes."""
    encoding_layer = PositionalEncoding(d_model=256)
    fake_embeddings = torch.randn(2, 50, 256)  # [Batch, SeqLen, EmbeddingDim]

    output = encoding_layer(fake_embeddings)
    assert output.shape == (2, 50, 256)


# ---2. SEQUENCE LOSS COLLAPSING LIFECYCLE TESTS---


def test_sequence_training_step_collapses_dimensions_correctly():
    """Verifies that training_step flattens sequence outputs for cross-entropy evaluation."""
    detector = MagentaTransformerDetector()
    device = torch.device("cpu")

    # Configure mock transformer to return logit sequences matching vocabulary space
    mock_transformer = MagicMock(spec=MagentaTranscriptionTransformer)
    mock_transformer.vocab_size = 358
    mock_transformer.return_value = torch.randn(
        2, 50, 358
    )  # [Batch, SeqLen, VocabSize]

    # Mock cross entropy to expect flattened [Batch * SeqLen] dimensions
    mock_criterion = MagicMock(return_value=torch.tensor(2.45))

    with (
        patch.object(detector, "_active_model", mock_transformer),
        patch.object(detector, "criterion", mock_criterion),
    ):

        # Assemble mixed structural data types: float frames and long tokens
        batch = (
            torch.randn(2, 128, 50),  # spec_b (float)
            torch.randint(0, 358, (2, 50)).long(),  # token_in_b (long)
            torch.randint(0, 358, (2, 50)).long(),  # token_tgt_b (long)
            torch.randn(2, 50, 358),  # bias_b (float)
        )

        loss = detector.training_step(batch, device)

        assert isinstance(loss, torch.Tensor)
        assert np.isclose(loss.item(), 2.45)

        # Verify that view flattening arguments matched expected targets
        called_args = mock_criterion.call_args[0]
        assert called_args[0].shape == (
            100,
            358,
        )  # [2 * 50, 358] collapsed matrix shape
        assert called_args[1].shape == (100,)  # [2 * 50] collapsed target vector shape


# ---3. HETEROGENEOUS DATA LOADING TESTS---


@patch("numpy.load")
def test_transformer_dataloader_enforces_mixed_tensor_types(mock_np_load):
    """Verifies that the dataloader casts token lists to long and audio frames to float."""
    detector = MagentaTransformerDetector()

    # Return 4 target arrays matching production disk extraction sequences
    mock_np_load.side_effect = [
        np.zeros((4, 229, 100)),  # spectrograms.npy
        np.ones((4, 50)),  # tokens_input.npy
        np.ones((4, 50)),  # tokens_target.npy
        np.zeros((4, 50, 358)),  # score_bias_masks.npy
    ]

    with patch.object(MagentaTransformerDetector, "dataset_path", "mock/magenta/paths"):
        loader = detector.get_dataloader()
        assert isinstance(loader, DataLoader)

        batch = next(iter(loader))
        spec, token_in, token_tgt, bias = batch

        # Verify exact datatypes required to prevent embedding layer crashes
        assert spec.dtype == torch.float32
        assert token_in.dtype == torch.long
        assert token_tgt.dtype == torch.long
        assert bias.dtype == torch.float32
