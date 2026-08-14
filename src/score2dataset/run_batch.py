"""Master orchestration entrypoint for the complete score2dataset training pipeline.

Sequentially drives the entire pipeline framework: handles process-isolated parallel
dataset rendering, executes 24ms time-aligned Log-Mel feature extraction loops over
the generated audio pools, and invokes the low-priority multi-node cluster job
scheduler to drain the workload queue concurrently.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

# Core pipeline infrastructure imports
from score2dataset.exceptions import Score2DatasetError
from score2dataset.generator import DatasetGenerator
from score2dataset.parsers.musicxml_parser import MusicXMLParser
from score2dataset.scripts.scheduler import LowPriorityClusterScheduler, get_ssh_hosts
from score2dataset.scripts.static_parser import ModelStaticParser


def compute_dataset_spectrograms(
    audio_dir: Path, n_mels: int = 229, sample_rate: int = 48000
) -> None:
    """Computes fixed-resolution log-mel arrays for all generated WAV files in disk.

    Args:
        audio_dir: Strongly typed Path pointing to the directory containing rendered wav files.
        n_mels: Number of logarithmic target frequency bins required by the model architectures.
        sample_rate: Expected target sampling resolution of the audio files.
    """
    print(
        f"\n🎵 Starting 24ms Log-Mel Spectrogram extraction loop inside: {audio_dir.name}..."
    )
    hop_length = int(0.024 * sample_rate)  # 0.024s * 48000Hz = 1152 samples hop size
    n_fft = hop_length * 2

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
    )

    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        print(
            "⚠️ Warning: No audio wave files discovered for feature extraction processing."
        )
        return

    for idx, wav_path in enumerate(wav_files, start=1):
        try:
            waveform, native_sr = torchaudio.load(str(wav_path))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if native_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=native_sr, new_freq=sample_rate
                )
                waveform = resampler(waveform)

            # Generate linear mel amplitudes and convert to logarithmic decibel ranges
            mel_spec = mel_transform(waveform)
            log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

            # Strip the temporary channel dimension and save as raw NumPy array
            feature_array = log_mel_spec.squeeze(0).numpy()

            # Save the spec matching the naming token hash used by the audio file
            output_npy_path = wav_path.with_suffix(".npy")
            np.save(output_npy_path, feature_array)

            if idx % 10 == 0 or idx == len(wav_files):
                print(
                    f"    + [{idx}/{len(wav_files)}] Extracted features for asset: {wav_path.name}"
                )
        except Exception as feature_err:
            print(
                f"❌ Failed to extract features for file {wav_path.name}: {feature_err}",
                file=sys.stderr,
            )


def run_full_pipeline() -> int:
    """Discovers repository assets, renders data, extracts features, and runs the scheduler.

    Returns:
        int: System exit code where 0 indicates success and 1 indicates a structural fault.
    """
    print("================================================================")
    print("🚀 Score2Dataset Unified Master Training Pipeline Execution Engine")
    print("================================================================")

    root_dir: Path = Path(__file__).resolve().parent.parent.parent
    inputs_folder: Path = root_dir / "data" / "inputs"
    output_directory: Path = root_dir / "data" / "outputs" / "wav"

    sfz_root: Path = root_dir / "data" / "assets" / "sfz"
    rir_folder: Path = root_dir / "data" / "assets" / "rir"

    # 1. Gather input MusicXML files dynamically
    xml_files: list[Path] = list(inputs_folder.glob("*.musicxml")) + list(
        inputs_folder.glob("*.mxl")
    )
    if not xml_files:
        print(
            f"❌ Critical Fault: No score files found inside inputs directory: {inputs_folder}",
            file=sys.stderr,
        )
        return 1

    try:
        # 2. Parse all symbolic notation scores into memory arrays
        print(" -> Initializing MusicXML Parser engines...")
        parser = MusicXMLParser(target_tpqn=480)
        parsed_score_tuples = []
        for xml_path in xml_files:
            in_memory_score = parser.parse_score(xml_path)
            parsed_score_tuples.append(in_memory_score)

        # 3. Assemble environmental configurations
        safe_sfz_paths = list(sfz_root.rglob("*_OnsetSafe.sfz"))
        engine_configs: list[dict[str, Any]] = [
            {"sfz_path": str(sfz_path)} for sfz_path in safe_sfz_paths
        ]
        batch_rir_paths: list[str] = [str(p) for p in rir_folder.glob("*.wav")]

        if not engine_configs or not batch_rir_paths:
            print(
                "❌ Critical Fault: Missing required instrument or convolution files.",
                file=sys.stderr,
            )
            return 1

        # 4. Fire Process-Isolated Dataset Generation
        print(" -> Instantiating Dataset Generator middleware...")
        generator = DatasetGenerator(
            engine_configs=engine_configs, rir_paths=batch_rir_paths
        )

        total_variations_per_piece = 50
        print(f" -> Handing matrix stream to the isolated worker pool pool context...")
        _ = generator.generate_batch(
            score_tuples=parsed_score_tuples,
            output_dir=str(output_directory),
            variation_count=total_variations_per_piece,
            base_seed=1000,
        )

        # 5. Extract fixed-resolution 24ms features across the output batch
        audio_out_path = output_directory / "audio"
        compute_dataset_spectrograms(
            audio_dir=audio_out_path, n_mels=229, sample_rate=48000
        )

        # 6. DISTRIBUTED CLUSTER EXECUTION BLOCK
        print("\n🌐 Initiating Distributed Cluster Scheduling Layer...")

        # Discover remote worker infrastructure nodes defined inside user profiles
        cluster_hosts: list[str] = get_ssh_hosts()
        if not cluster_hosts:
            print("⚠️ Warning: No remote cluster hosts found inside ~/.ssh/config.")
            print("   Defaulting loop execution to localized CPU/GPU parameters.")
            cluster_hosts = ["localhost"]

        # Statically analyze your package trees to parse concrete OnsetDetector models
        models_package_dir: Path = root_dir / "src" / "score2dataset" / "models"
        parser_agent = ModelStaticParser(models_directory=models_package_dir)
        discovered_work_pool: list[JobTask] = parser_agent.extract_executable_jobs()

        if not discovered_work_pool:
            print(
                "❌ Critical Fault: No concrete OnsetDetector subclasses found inside package tree.",
                file=sys.stderr,
            )
            return 1

        print(
            f" -> Discovered {len(discovered_work_pool)} valid OnsetDetector workload target(s)."
        )
        print(
            f" -> Concurrently targeting {len(cluster_hosts)} remote host context connection(s)."
        )

        # Instantiate the final low-priority cluster scheduler loop
        # monitor_interval=20 sweeps hardware slots every 20 seconds for foreign intruder PIDs
        scheduler = LowPriorityClusterScheduler(
            hosts=cluster_hosts, utilization_limit=70, monitor_interval=20
        )

        print("\n🎬 Launching master training queue event loop...")
        scheduler.execute_orchestration(job_queue=discovered_work_pool)

        print("\n================================================================")
        print("🎉 SUCCESS: Master Execution Complete. Full Pipeline Drained.")
        print("================================================================")
        return 0

    except Score2DatasetError as pipeline_error:
        print(
            f"\n❌ Pipeline Master Orchestrator Intercepted Failure: {pipeline_error}",
            file=sys.stderr,
        )
        return 1
    except Exception as unexpected_err:
        print(
            f"\n❌ Unexpected Crash Intercepted by Master Wrapper: {unexpected_err}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(run_full_pipeline())
