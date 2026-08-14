"""Isolated evaluation worker executing single-model inference and normalization.

This script runs inside a localized remote cluster GPU context to process a song-level
validation dataset split, normalize variable model output dimensions into a standardized
seconds-pitch timeline event structure, and compute asymmetric performance metrics.
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

# Infrastructure contracts
from score2dataset.models.onsets_and_velocities import PeakPickingFrameDecoder

# ---1. ASYMMETRIC METRIC ENGINE---


def calculate_unified_mir_score(
    predicted_events: list[tuple[float, int]],
    ground_truth_events: list[tuple[float, int]],
    tolerance_ms: float = 25.0,
    tau_ms: float = 10.0,
    fp_penalty_weight: float = 2.0,
) -> dict[str, float]:
    """Balances onset event timeline hits and millisecond time drift deviations."""
    tol_sec = tolerance_ms / 1000.0
    tau_sec = tau_ms / 1000.0

    true_positives = 0
    false_positives = 0
    time_deviations = []
    matched_gt_indices = set()

    for pred_time, pred_pitch in predicted_events:
        best_match_idx = None
        smallest_drift = float("inf")

        for idx, (gt_time, gt_pitch) in enumerate(ground_truth_events):
            if idx in matched_gt_indices or pred_pitch != gt_pitch:
                continue

            drift = abs(pred_time - gt_time)
            if drift <= tol_sec and drift < smallest_drift:
                smallest_drift = drift
                best_match_idx = idx

        if best_match_idx is not None:
            true_positives += 1
            matched_gt_indices.add(best_match_idx)
            time_deviations.append(smallest_drift)
        else:
            false_positives += 1

    false_negatives = len(ground_truth_events) - len(matched_gt_indices)

    precision = (
        true_positives / (true_positives + (fp_penalty_weight * false_positives))
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if len(ground_truth_events) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    mean_absolute_error: float = float(
        np.mean(time_deviations) if time_deviations else tol_sec
    )
    temporal_multiplier: float = float(
        np.exp(-mean_absolute_error / tau_sec) if time_deviations else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mae_ms": mean_absolute_error * 1000.0,
        "unified_score": f1_score * temporal_multiplier,
    }


# ---2. OUTPUT NORMALIZATION ROUTINES---


def run_isolated_inference(
    model_name: str, data_directory: Path, device: torch.device
) -> list[tuple[float, int]]:
    """Loads a runtime class model dynamically and enforces output unification."""

    module_path = f"score2dataset.models.{model_name.lower()}"
    model_module = importlib.import_module(module_path)

    detector_class = getattr(model_module, f"{model_name}Detector")
    detector = detector_class()

    # Direct dynamic assignment bypassing magic strings
    detector.set_validation_path(str(data_directory))
    detector.initialize_components(device)

    checkpoint_path = (
        Path("data/outputs/checkpoints")
        / f"{model_name.lower()}_checkpoint_epoch_50.pt"
    )
    detector.load_checkpoint(checkpoint_path, device)

    dataloader = detector.get_dataloader()
    detector.get_active_model().eval()

    normalized_events = []

    with torch.no_grad():
        if model_name == "MagentaTransformer":
            # Autoregressive Token Sequence Decoding Normalizer
            for batch in dataloader:
                spec, tokens_in, _, bias = [t.to(device) for t in batch]
                outputs = detector.get_active_model()(
                    spec, tokens_in, score_bias_mask=bias
                )
                token_indices = torch.argmax(outputs, dim=-1).cpu().numpy()
                for b_idx in range(token_indices.shape[0]):
                    for frame_idx, token in enumerate(token_indices[b_idx]):
                        if 0 < token < 128:
                            normalized_events.append((frame_idx * 0.024, int(token)))
        else:
            # Frame / Multi-Task Matrix Peak-Picking Normalizer
            decoder = PeakPickingFrameDecoder(threshold=0.5, frame_resolution=0.024)
            for batch in dataloader:
                spec = batch[0].to(device)  # Safely reference input feature tensor
                outputs = detector.get_active_model()(spec)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Isolate onset tracking head from frame head

                for b_idx in range(outputs.size(0)):
                    events = decoder.decode_predictions(outputs[b_idx])
                    normalized_events.extend(events)

    normalized_events.sort(key=lambda x: x[0])
    return normalized_events


# ---3. WORKER RUNTIME ENTRYPOINT---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    val_path = Path("data/processed/validation_isolated_songs")
    target_device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    # Execute inference steps across unified tracking interfaces
    computed_timeline = run_isolated_inference(args.model, val_path, target_device)

    # Save the normalized prediction event list to a local temporary cache file
    cache_destination = Path(
        f"data/outputs/evaluation_cache/{args.model.lower()}_events.npy"
    )
    cache_destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_destination, np.array(computed_timeline, dtype=object))

    print(f"Worker process complete. Normalized outcomes saved for model: {args.model}")
    sys.exit(0)
