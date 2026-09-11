"""Module for decoding peak predictions into high-resolution onset timestamps."""

import torch


class PeakPickingFrameDecoder:
    """Decodes parallel multi-task channels into high-resolution sub-frame timestamps."""

    def __init__(
        self, threshold: float = 0.5, frame_resolution: float = (256 / 48000)
    ) -> None:
        self.threshold: float = threshold
        self.frame_res: float = frame_resolution

    def decode_predictions(
        self, classification_logits: torch.Tensor, regression_shifts: torch.Tensor
    ) -> list[tuple[float, int]]:
        """Finds discrete probability peaks and applies continuous micro-timing adjustments.

        Args:
            classification_logits: Onset probability tracker shaped [Frames, 128]
            regression_shifts: Sub-frame continuous offset estimates shaped [Frames, 128]

        Returns:
            A list tracking tuples of (high_resolution_onset_seconds, midi_pitch)
        """
        probabilities = torch.sigmoid(classification_logits).detach().cpu().numpy()
        # Shifts map to a continuous [0, 1) percentage interval into the frame width
        shifts = torch.sigmoid(regression_shifts).detach().cpu().numpy()

        n_frames, n_pitches = probabilities.shape
        detected_onsets = []

        for pitch in range(n_pitches):
            pitch_curve = probabilities[:, pitch]
            shift_curve = shifts[:, pitch]

            for frame in range(1, n_frames - 1):
                prob_value = pitch_curve[frame]

                # Local maxima verification constraint
                if (
                    prob_value > self.threshold
                    and prob_value > pitch_curve[frame - 1]
                    and prob_value > pitch_curve[frame + 1]
                ):
                    # Combine discrete step placement with continuous sub-frame regression offset
                    fractional_frame = float(frame) + float(shift_curve[frame])
                    high_res_seconds = fractional_frame * self.frame_res

                    detected_onsets.append((high_res_seconds, pitch))

        detected_onsets.sort(key=lambda x: x[0])
        return detected_onsets
