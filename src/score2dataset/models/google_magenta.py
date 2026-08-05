"""Google Magenta style Sequence-to-Sequence (Seq2Seq) Transformer for piano transcription.

Maps raw log-mel spectrogram frames directly into a sequence of discrete musical event tokens
using self-attention layers modified to prevent structural template memorization.
"""

import math
from typing import cast

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Injects relative temporal awareness tensors into feature embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Seq_Len, d_model]
        # Explicitly narrow the union type down to a pure Tensor to satisfy static type checks
        positional_embedding = cast(torch.Tensor, self.pe)

        seq_len = x.size(1)
        return x + positional_embedding[:, :seq_len]


class MagentaTranscriptionTransformer(nn.Module):
    """Autoregressive transformer network for discrete token sequence synthesis."""

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # 1. Acoustic Front-End Projector: Maps Mel frames cleanly into d_model embedding space
        self.spectrogram_projector = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 2. Vocabulary Definitions: 128 Note_ONs + 128 Note_OFFs + 100 Time_Shifts (10ms to 1s) + SOS + EOS
        # Total Vocab Size = 128 + 128 + 100 + 2 = 358
        self.vocab_size = 358
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)

        # 3. Core Transformer Engine
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=1024,
            batch_first=True,
        )

        # 4. Target Projection Head
        self.logits_generator = nn.Linear(d_model, self.vocab_size)

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        """Generates a causal upper-triangular mask to prevent looking into the future."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, 0.0)
        )

    def forward(
        self,
        log_mel_spec: torch.Tensor,
        target_tokens: torch.Tensor,
        score_bias_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Translates spectrogram images into event sequence predictions.

        Args:
            log_mel_spec: Spectrogram tensor shaped [Batch, Bins, Frames]
            target_tokens: Shifted target token arrays shaped [Batch, Target_Seq_Len]
            score_bias_mask: Structural logit constraints matching vocabulary space

        Returns:
            Logits tensor tracking probabilities of shape [Batch, Target_Seq_Len, Vocab_Size]
        """
        # Re-orient spectrogram inputs to standard sequence layout -> [Batch, Frames, Bins]
        x_src = log_mel_spec.permute(0, 2, 1)
        src_embeddings = self.spectrogram_projector(x_src) * math.sqrt(self.d_model)
        src_embeddings = self.pos_encoder(src_embeddings)

        # Process target token sequences into dense embeddings
        tgt_embeddings = self.token_embedding(target_tokens) * math.sqrt(self.d_model)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)

        # Enforce causality boundaries on the autoregressive decoder pipeline
        tgt_mask = self.generate_square_subsequent_mask(
            target_tokens.size(1), target_tokens.device
        )

        # Execute unified transformer architecture graph execution pass
        output_features = self.transformer(
            src=src_embeddings, tgt=tgt_embeddings, tgt_mask=tgt_mask
        )

        raw_logits = self.logits_generator(output_features)

        # 💡 Inject Your Asymmetric Thesis directly as a dynamic cross-attention logit bias
        if score_bias_mask is not None:
            # Shift network predictions to heavily suppress tokens that rely purely on score-memory
            raw_logits = raw_logits + score_bias_mask

        return raw_logits
