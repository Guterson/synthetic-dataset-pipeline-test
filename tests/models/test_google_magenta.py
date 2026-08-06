"""Functional verification suite for the Sequence-to-Sequence (Seq2Seq) Magenta Transformer.

Validates front-end acoustic frame projections, causal sequence masking bounds,
and logit bias gradient distributions for your score-memory mitigation thesis layers.
"""

import torch

from score2dataset.models.google_magenta import MagentaTranscriptionTransformer


def test_transformer_output_dimensions() -> None:
    """Verify that the encoder-decoder pipeline maps audio features cleanly to vocabulary token spaces."""
    # Instantiated using compact configurations to save local runtime compilation footprints
    model = MagentaTranscriptionTransformer(
        n_mels=128, d_model=256, nhead=8, num_encoder_layers=2, num_decoder_layers=2
    )
    model.eval()

    # Simulate 1 audio window: 128 log-mel bins across exactly 100 temporal frames
    mock_spec = torch.randn(1, 128, 100)

    # Simulate a target sequence of 50 tokens (e.g., NOTE_ON, TIME_SHIFT events)
    mock_targets = torch.randint(0, 358, (1, 50))

    with torch.no_grad():
        logits = model(log_mel_spec=mock_spec, target_tokens=mock_targets)

    # The model output must trace exactly: [Batch, Target_Seq_Len, Vocabulary_Size]
    assert logits.shape == (1, 50, 358)


def test_causal_mask_generation() -> None:
    """Business Rule: The autoregressive decoder must be physically blocked from looking into the future.

    Ensures the generated causal matrix applies -inf boundaries to protect the timeline.
    """
    model = MagentaTranscriptionTransformer()
    device = torch.device("cpu")

    # Generate a mask for a 5-step event sequence
    mask = model.generate_square_subsequent_mask(sz=5, device=device)

    assert mask.shape == (5, 5)
    # The upper triangle (future frames) must be heavily penalized with -inf values
    assert torch.all(mask[0, 1:] == float("-inf"))
    # The lower triangle and diagonal (past/current frames) must retain standard 0.0 scale parameters
    assert torch.all(mask.diagonal() == 0.0)


def test_score_bias_logit_injection() -> None:
    """Verify that custom structural bias masks cleanly modify output token probabilities.

    Ensures that injecting external logit adjustments does not disrupt or clip
    backpropagated gradients flowing through the linear projection heads.
    """
    model = MagentaTranscriptionTransformer(n_mels=128, d_model=256)
    model.train()  # Activate backpropagation register states

    mock_spec = torch.randn(1, 128, 60)
    mock_targets = torch.randint(0, 358, (1, 30))

    # Create a custom structural bias mask matching the output logit block boundaries
    # Setting a -10.0 penalty at a specific coordinate simulates suppressing a score hallucination
    bias_mask = torch.zeros((1, 30, 358), dtype=torch.float32)
    bias_mask[0, 15, 45] = -10.0

    logits_without_bias = model(mock_spec, mock_targets, score_bias_mask=None)
    logits_with_bias = model(mock_spec, mock_targets, score_bias_mask=bias_mask)

    # 1. Assert that the mathematical modification changed the target output token value explicitly
    assert not torch.equal(logits_with_bias, logits_without_bias)
    assert torch.isclose(
        logits_with_bias[0, 15, 45], logits_without_bias[0, 15, 45] - 10.0
    )

    # 2. Execute a backward pass to guarantee gradient paths are left unhindered
    loss_fn = torch.nn.CrossEntropyLoss()
    dummy_labels = torch.randint(0, 358, (1, 30))

    # Flatten the outputs for the cross-entropy function contract
    loss = loss_fn(logits_with_bias.view(-1, 358), dummy_labels.view(-1))
    loss.backward()

    # Assert that optimization updates flow cleanly out to the linear generator head
    assert model.logits_generator.weight.grad is not None
