"""Onsets and Velocities (O&V) modeling infrastructure.

Implements acoustic frame featurization at 24ms resolution, custom asymmetric
binary cross-entropy cost weights, and a local-maxima peak-picking decoder.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from score2dataset.models.onset_detector import OnsetDetector
from score2dataset.processors.asymmetric_loss import AsymmetricBCEWithLogitsLoss


class OVOnsetClassifier(nn.Module):
    """Minimal O&V style front-end acoustic feature framing classifier.

    Processes log-mel spectrogram features using 2D convolutions and projects
    them directly into discrete, frame-by-frame pitch probabilities.
    """

    def __init__(self, n_mels: int = 229) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        flattened_freqs: int = n_mels // 4
        self.pitch_projector = nn.Conv1d(
            in_channels=64 * flattened_freqs,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )

    def forward(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        """Processes the spectrogram and yields pitch frame logits.

        Args:
            log_mel_spec: Tensor of shape [Batch, Bins, Frames]

        Returns:
            Logits matrix tensor of shape [Batch, Frames, 128]
        """
        x: torch.Tensor = log_mel_spec.unsqueeze(1)
        x = self.features(x)

        shape_dims: torch.Size = x.shape
        batch, channels, freqs, frames = shape_dims
        x = x.view(batch, channels * freqs, frames)

        logits = self.pitch_projector(x)
        return logits.permute(0, 2, 1)


class PeakPickingFrameDecoder:
    """Decodes frame probabilities into discrete onset events using local maxima rules."""

    def __init__(self, threshold: float = 0.5, frame_resolution: float = 0.024) -> None:
        self.threshold: float = threshold
        self.frame_res: float = frame_resolution

    def decode_predictions(self, raw_logits: torch.Tensor) -> list[tuple[float, int]]:
        """Applies peak-picking filters across each pitch channel sequentially.

        Args:
            raw_logits: Model outputs for a single song file tracking tensor [Frames, 128]

        Returns:
            A list tracking tuples of (absolute_onset_seconds, midi_pitch)
        """
        probabilities = torch.sigmoid(raw_logits).detach().cpu().numpy()
        n_frames, n_pitches = probabilities.shape

        detected_onsets = []

        for pitch in range(n_pitches):
            pitch_curve = probabilities[:, pitch]

            for frame in range(1, n_frames - 1):
                prob_value = pitch_curve[frame]

                if (
                    prob_value > self.threshold
                    and prob_value > pitch_curve[frame - 1]
                    and prob_value > pitch_curve[frame + 1]
                ):
                    onset_seconds = float(frame * self.frame_res)
                    detected_onsets.append((onset_seconds, pitch))

        detected_onsets.sort(key=lambda x: x[0])
        return detected_onsets


class OnsetsAndVelocitiesDetector(OnsetDetector):
    """Concrete interface wrapper mapping the O&V architecture to the scheduler pipeline."""

    def __init__(self, n_mels: int = 229) -> None:
        """Initializes the O&V adapter layer with customizable spectrogram dimensions."""
        self.n_mels: int = n_mels
        # Explicit initialization values inherited from safe default attributes block
        self._active_model = None
        self._active_optimizer = None
        self.criterion: nn.Module | None = None

    # ---1. ABSTRACT HOOK IMPLEMENTATIONS---

    def initialize_components(self, device: torch.device) -> None:
        """Initializes O&V models, specific Adam optimizer configurations, and loss metrics."""
        self._active_model = OVOnsetClassifier(n_mels=self.n_mels).to(device)
        self.criterion = AsymmetricBCEWithLogitsLoss(
            hallucination_penalty=5.0, acoustic_penalty=3.0
        )
        self._active_optimizer = torch.optim.Adam(
            self._active_model.parameters(), lr=0.001
        )

    def get_dataloader(self) -> DataLoader:
        """Loads physical dataset binaries from disk paths into an active training loader."""

        data_root = Path(self.dataset_path)
        print(f"Loading active processing data pool from directory: {data_root}")

        # Fetching production binary layouts, matching your storage layout rules
        specs = np.load(data_root / "spectrograms.npy")
        audio = np.load(data_root / "audio_truth.npy")
        score = np.load(data_root / "score_expected.npy")

        dataset = TensorDataset(
            torch.from_numpy(specs).float(),
            torch.from_numpy(audio).float(),
            torch.from_numpy(score).float(),
        )
        return DataLoader(dataset, batch_size=4, shuffle=True)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], device: torch.device
    ) -> torch.Tensor:
        """Executes a single processing step using O&V signatures."""
        # Unpack the specific 3-tensor signature safely
        spec, audio, score = batch

        spec = spec.to(device)
        audio = audio.to(device)
        score = score.to(device)

        # Forward pass tracking [Batch, Frames, 128]
        assert self._active_model is not None
        predictions = self._active_model(spec)

        # Return the computed loss scalar tensor directly back to the execution engine
        if self.criterion is None:
            raise RuntimeError("Loss criterion was not properly initialized.")

        return self.criterion(predictions, audio, score)


if __name__ == "__main__":
    # Allows the script to be invoked smoothly as an independent background command
    detector = OnsetsAndVelocitiesDetector()
    detector.train(total_epochs=10, checkpoint_name="ov", checkpoint_interval=5)
