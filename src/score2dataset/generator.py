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

from score2dataset.datamodels import PerformanceScore
from score2dataset.exporters.midi_exporter import MidiExporter
from score2dataset.processors.rir_convolve import RirConvolver
from score2dataset.wrapper import AudioEngine


def _parallel_worker_thunk(
    score_data: PerformanceScore,
    engine: AudioEngine,
    rir_path: Path,
    output_dir: Path,
    filename: str,
) -> Path:
    """Isolated multiprocessing anchor executing single rendering tasks.

    Runs inside an independent OS process memory node to insulate parent state.
    """
    midi_exporter = MidiExporter()
    convolver = RirConvolver()

    with tempfile.TemporaryDirectory() as local_cache_dir:
        ssd_path = Path(local_cache_dir)

        temp_midi = ssd_path / "normalized.mid"
        temp_wav = ssd_path / "dry_render.wav"
        temp_wet_wav = ssd_path / "wet_render.wav"

        # 1. Save absolute timeline performance to scratchpad disk
        midi_exporter.export_score(score_data, temp_midi)

        # 2. Render dry PCM blocks using the abstract synthesizer extension
        engine.render_audio(midi_path=temp_midi, output_wav_path=temp_wav)

        # 3. Apply high-fidelity resampling, spatial convolution, and 16-bit mapping
        convolver.process_audio(
            dry_wav_path=temp_wav, rir_path=rir_path, output_wav_path=temp_wet_wav
        )

        # 4. Move the finished 48kHz, 16-bit track back onto persistent space
        final_destination = output_dir / f"{filename}.wav"

        shutil.copy(temp_wet_wav, final_destination)

        return final_destination


class DatasetGenerator:
    """Manages secure parallel execution routines across local hardware pools [1]."""

    def __init__(self, engines: list[AudioEngine], rir_paths: list[str]) -> None:
        """Initializes the batch engine with balancing asset pools.

        Args:
            engines: A list of initialized engine instances subclassing AudioEngine.
            rir_paths: A list of public string paths pointing to RIR assets (.wav).

        Raises:
            ValueError: If either the engine pool or RIR pool is empty.
        """
        if not engines or not rir_paths:
            raise ValueError("Asset distribution pools cannot be empty.")

        self.engines: list[AudioEngine] = engines
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
        resolved_out_dir = Path(output_dir).resolve()
        resolved_out_dir.mkdir(parents=True, exist_ok=True)

        # Compute process allocation boundaries (Clamp usage to max 70% of local cores) [1]
        available_cores = os.cpu_count() or 1
        safe_worker_limit = max(1, int(available_cores * 0.7))

        processing_tasks = []
        task_counter = 0

        # Build balanced non-cartesian task matrices
        for score in base_scores:
            base_name = score.source.stem

            for v_idx in range(variation_count):
                instance_filename = f"{base_name}_var_{v_idx}"

                # Clone the base score container to isolate random mutation states
                mutated_score = copy.deepcopy(score)

                # --- FUTURE ALTERATION CALL WILL INJECT HERE ---
                # alterations.apply_drift_variations(mutated_score, seed=v_idx)

                # Balance assignments round-robin fashion instead of cross-multiplying pools [1]
                selected_engine = self.engines[task_counter % len(self.engines)]
                selected_rir = self.rir_pool[task_counter % len(self.rir_pool)]
                task_counter += 1

                # Queue the clean unpacked parameters into the process payload list
                processing_tasks.append(
                    (
                        mutated_score,
                        selected_engine,
                        selected_rir,
                        resolved_out_dir,
                        instance_filename,
                    )
                )

        # Launch the multiprocessing core pool using the calculated hardware safety cap [1]
        completed_records: list[Path] = []
        with multiprocessing.Pool(processes=safe_worker_limit) as pool:
            # starmap unpacks the task tuples across workers concurrently
            results = pool.starmap(_parallel_worker_thunk, processing_tasks)
            completed_records.extend(results)

        return completed_records
