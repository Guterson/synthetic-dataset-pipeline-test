"""Core orchestration engine for memory-constrained dataset generation.

Provides a stateless, stream-oriented parallel rendering framework designed
to eliminate memory and network file system swap accumulation on restricted nodes.
"""

import copy
import multiprocessing
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from score2dataset.audio_engine import SfizzRenderEngine
from score2dataset.datamodels import PerformanceScore
from score2dataset.exceptions import ProcessorError
from score2dataset.exporters.midi_exporter import MidiExporter
from score2dataset.processors.rir_convolve import RirConvolver


def _isolated_render_thunk(
    task_args: tuple[PerformanceScore, dict[str, Any], Path, Path, str],
) -> str:
    """Executes a single audio rendering and convolution task inside an isolated memory node.

    This function operates as a stateless anchor. It allocates all intermediate
    DSP arrays within an ephemeral local directory and forces absolute cleanup
    before yielding control back to the operating system.

    Args:
        task_args: A packed tuple containing the score object, synthesizer configurations,
            target RIR file path, target output directory, and the destination filename.

    Returns:
        A string representation of the successfully written file path destination.
    """
    score, engine_config, rir_path, output_dir, filename = task_args

    midi_exporter = MidiExporter()
    convolver = RirConvolver()
    engine = SfizzRenderEngine(**engine_config)

    final_destination: Path = output_dir / f"{filename}.wav"

    # Use a localized context manager to ensure unmanaged libsndfile handles close cleanly
    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as local_cache:
            cache_path: Path = Path(local_cache)
            temp_midi: Path = cache_path / "scratch.mid"
            temp_dry_wav: Path = cache_path / "scratch_dry.wav"
            temp_wet_wav: Path = cache_path / "scratch_wet.wav"

            # Execute the linear processing pipe entirely within local memory space
            midi_exporter.export_score(score, temp_midi)
            engine.render_audio(midi_path=temp_midi, output_wav_path=temp_dry_wav)

            convolver.process_audio(
                dry_wav_path=temp_dry_wav,
                rir_path=rir_path,
                output_wav_path=temp_wet_wav,
            )

            # Move the final product directly to the target directory
            shutil.copy(temp_wet_wav, final_destination)

    finally:
        # Force destruction of local references to guarantee instantaneous unlinking
        del midi_exporter
        del convolver
        del engine

    return str(final_destination)


class DatasetGenerator:
    """Orchestrates high-throughput parallel dataset synthesis beneath volatile memory caps."""

    def __init__(
        self, engine_configs: list[dict[str, Any]], rir_paths: list[str]
    ) -> None:
        """Initializes the batch engine balance maps with structured validation parameters.

        Args:
            engine_configs: A list of configuration dicts used to instantiate audio layers.
            rir_paths: A list of file path strings pointing to verified RIR audio files.

        Raises:
            ProcessorError: If any input configuration pool is empty.
        """
        if not engine_configs or not rir_paths:
            raise ProcessorError("Asset initialization vectors cannot be empty arrays.")

        self.engine_configs: list[dict[str, Any]] = engine_configs
        self.rir_pool: list[Path] = [Path(p).resolve() for p in rir_paths]

    def generate_batch(
        self,
        base_scores: list[PerformanceScore],
        output_dir: str,
        variation_count: int = 1,
    ) -> list[Path]:
        """Distributes performance tasks across a streaming multiprocessing queue.

        Utilizes an un-ordered stream generator with process-level recycling to keep
        the network file system virtual memory allocation flat over long execution runs.

        Args:
            base_scores: A list containing the singular base performance score container.
            output_dir: String location specifying where output WAV variations are written.
            variation_count: The total number of unique environment mutations to calculate.

        Returns:
            A list of Path locations tracking every successfully generated variant.
        """
        resolved_out_dir: Path = Path(output_dir).resolve()
        resolved_out_dir.mkdir(parents=True, exist_ok=True)

        worker_concurrency: int = self._calculate_conservative_worker_limit()
        task_payload: list[tuple[PerformanceScore, dict[str, Any], Path, Path, str]] = (
            []
        )
        matrix_index: int = 0

        for score in base_scores:
            base_name: str = score.source.stem

            for v_idx in range(variation_count):
                instance_filename: str = f"{base_name}_var_{v_idx}"

                # Create a complete data break from the main loop thread state
                mutated_score: PerformanceScore = copy.deepcopy(score)

                # Balanced extraction from asset parameters pools
                selected_config: dict[str, Any] = self.engine_configs[
                    matrix_index % len(self.engine_configs)
                ]
                selected_rir: Path = self.rir_pool[matrix_index % len(self.rir_pool)]
                matrix_index += 1

                task_payload.append(
                    (
                        mutated_score,
                        selected_config,
                        selected_rir,
                        resolved_out_dir,
                        instance_filename,
                    )
                )

        print(
            f"\n🚀 Launching Parallel Stream Engine [{worker_concurrency} Cores Active]"
        )
        print(f"📦 Total target matrix payload allocation: {len(task_payload)} files")

        completed_records: list[Path] = []

        # Enforce strict single-task process recycling via a controlled spawn pool
        pool_context = multiprocessing.get_context("spawn")
        with pool_context.Pool(
            processes=worker_concurrency, maxtasksperchild=1
        ) as stream_pool:

            # imap_unordered pulls tasks individually, preventing long array queues in memory
            task_stream = stream_pool.imap_unordered(
                _isolated_render_thunk, task_payload
            )

            for idx, result_path_str in enumerate(task_stream, start=1):
                completed_records.append(Path(result_path_str))

                print(
                    f"  ✅ [{idx}/{len(task_payload)}] Generated asset variance: {Path(result_path_str).name}"
                )

                # Clear python cache layers at the end of each stream loop iteration
                import gc

                gc.collect()

        print(
            f"\n🎉 Generation successful. {len(completed_records)} variations secured cleanly.\n"
        )
        return completed_records

    def _calculate_conservative_worker_limit(self) -> int:
        """Calculates strict hardware safety limits based on active host capacities."""
        hardware_cores: int = os.cpu_count() or 1

        # On a highly volatile lab workstation, never consume more than half of the cores
        # to ensure the host process can maintain network sync stability.
        return max(1, int(hardware_cores * 0.5))
