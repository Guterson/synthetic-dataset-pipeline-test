"""Core orchestration engine for synthetic audio dataset generation.

This module provides concurrent, non-cartesian multiprocessing factory loops
to manage performance variations across workstation processor clusters safely.
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


def _parallel_worker_thunk(
    score_data: PerformanceScore,
    engine_config: dict[str, Any],
    rir_path: Path,
    output_dir: Path,
    filename: str,
) -> Path:
    """Isolated multiprocessing anchor executing single rendering tasks.

    Runs inside an independent OS process memory node to insulate parent state.
    """
    midi_exporter = MidiExporter()
    convolver = RirConvolver()
    engine = SfizzRenderEngine(**engine_config)

    with tempfile.TemporaryDirectory() as local_cache_dir:
        ssd_path: Path = Path(local_cache_dir)
        temp_midi: Path = ssd_path / "normalized.mid"
        temp_wav: Path = ssd_path / "dry_render.wav"
        temp_wet_wav: Path = ssd_path / "wet_render.wav"

        midi_exporter.export_score(score_data, temp_midi)
        engine.render_audio(midi_path=temp_midi, output_wav_path=temp_wav)

        convolver.process_audio(
            dry_wav_path=temp_wav, rir_path=rir_path, output_wav_path=temp_wet_wav
        )

        final_destination: Path = output_dir / f"{filename}.wav"
        shutil.copy(temp_wet_wav, final_destination)
        return final_destination


class DatasetGenerator:
    """Manages secure parallel execution routines across local hardware pools."""

    def __init__(
        self, engine_configs: list[dict[str, Any]], rir_paths: list[str]
    ) -> None:
        """Initializes the batch engine with balancing asset pools.

        Args:
            engine_configs: List of configuration maps to spawn AudioEngine instances.
            rir_paths: A list of public string paths pointing to RIR assets (.wav).

        Raises:
            ProcessorError: If either the engine configuration pool or RIR pool is empty.
        """
        if not engine_configs or not rir_paths:
            raise ProcessorError("Asset distribution pools cannot be empty.")

        self.engine_configs: list[dict[str, Any]] = engine_configs
        self.rir_pool: list[Path] = [Path(p).resolve() for p in rir_paths]

    def generate_batch(
        self,
        base_scores: list[PerformanceScore],
        output_dir: str,
        variation_count: int = 1,
    ) -> list[Path]:
        """Executes a non-cartesian parallel processing sweep using CPU safety caps.

        Distributes variation layers across processes while balancing task assignments
        evenly across the available synthesizer and reverb pools.

        Args:
            base_scores: A list of pre-parsed base PerformanceScore memory objects.
            output_dir: Target destination path string where dataset files are saved.
            variation_count: Total unique performance adjustments to generate per score.

        Returns:
            A list of Path destinations tracking every successfully generated file.
        """
        resolved_out_dir: Path = Path(output_dir).resolve()
        resolved_out_dir.mkdir(parents=True, exist_ok=True)

        safe_worker_limit: int = self._calculate_safe_worker_limit()
        processing_tasks: list[
            tuple[PerformanceScore, dict[str, Any], Path, Path, str]
        ] = []
        task_counter: int = 0

        for score in base_scores:
            base_name: str = score.source.stem

            for v_idx in range(variation_count):
                instance_filename: str = f"{base_name}_var_{v_idx}"
                mutated_score: PerformanceScore = copy.deepcopy(score)

                selected_config: dict[str, Any] = self.engine_configs[
                    task_counter % len(self.engine_configs)
                ]
                selected_rir: Path = self.rir_pool[task_counter % len(self.rir_pool)]
                task_counter += 1

                processing_tasks.append(
                    (
                        mutated_score,
                        selected_config,
                        selected_rir,
                        resolved_out_dir,
                        instance_filename,
                    )
                )

        completed_records: list[Path] = []
        with multiprocessing.Pool(processes=safe_worker_limit) as pool:
            results: list[Path] = pool.starmap(_parallel_worker_thunk, processing_tasks)
            completed_records.extend(results)

        return completed_records

    def _calculate_safe_worker_limit(self) -> int:
        """Computes hardware allocation limits based on live system load thresholds."""
        available_cores: int = os.cpu_count() or 1

        try:
            # os.getloadavg() returns (1-min, 5-min, 15-min) system load metrics
            one_min_load: float = os.getloadavg()[0]

            # If the current 1-minute load exceeds 70% of a single core's capacity
            if one_min_load > (available_cores * 0.7):
                # Scale down aggressively to avoid choking an already stressed machine
                return max(1, int(available_cores * 0.3))
        except (AttributeError, OSError):
            # Fallback guard for environments where load averages cannot be requested
            pass

        return max(1, int(available_cores * 0.7))
