"""Integrated parallel orchestration engine for synthetic dataset generation.

Applies sequential humanizing perturbations (Event, Tempo, Jitter, Articulation, Intensity)
entirely in memory before executing stateless, local cache audio renders.
"""

import copy
import multiprocessing
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from score2dataset.audio_engine import SfizzRenderEngine
from score2dataset.datamodels import PerformanceScore, ScoreExpressionMap
from score2dataset.exceptions import ProcessorError
from score2dataset.exporters.midi_exporter import MidiExporter
from score2dataset.processors.articulation_modifier import ArticulationModifier

# Import the four newly formalized humanizer modules cleanly
from score2dataset.processors.event_modifier import EventLevelModifier
from score2dataset.processors.intensity_profile import IntensityProfileModifier
from score2dataset.processors.rir_convolve import RirConvolver
from score2dataset.processors.tempo_contour import TempoContourGenerator
from score2dataset.processors.temporal_jitter import TemporalJitterModifier


def _isolated_render_thunk(
    task_args: tuple[
        PerformanceScore, ScoreExpressionMap, dict[str, Any], Path, Path, str, int
    ],
) -> str:
    """Executes humanization, audio rendering, and convolution inside an isolated process.

    Operates as a stateless anchor to prevent memory accumulation on lab nodes.
    """
    (
        base_score,
        expression_map,
        engine_config,
        rir_path,
        output_dir,
        filename,
        variation_seed,
    ) = task_args

    # 1. Instantiate the Humanizer Modifiers using the unique variation seed for strict reproducibility
    event_mod = EventLevelModifier(seed=variation_seed)
    tempo_gen = TempoContourGenerator(seed=variation_seed)
    jitter_mod = TemporalJitterModifier(seed=variation_seed)
    artic_mod = ArticulationModifier(seed=variation_seed)
    intensity_mod = IntensityProfileModifier(seed=variation_seed)

    # 2. Execute Sequential In-Memory Perturbations (The Pipeline Core)
    try:
        # Step A: Structural event modifications (omissions, substitutions, insertions)
        perturbed_score = event_mod.perturb_score(base_score)

        # Step B: Continuous phrasing map derivation and timing conversion
        tempo_spline = tempo_gen.generate_tempo_map(expression_map)
        perturbed_score = tempo_gen.apply_phrasing_to_score(
            perturbed_score, tempo_spline
        )

        # Step C: Microtiming AR(1) motor jitter displacement mapping
        perturbed_score = jitter_mod.perturb_timing(perturbed_score, tempo_spline)

        # Step D: Local articulation duty-cycle adjustments (staccato/legato)
        perturbed_score = artic_mod.perturb_articulation(
            perturbed_score, expression_map
        )

        # Step E: Piecewise intensity and velocity profiling mapping
        perturbed_score = intensity_mod.perturb_intensity(
            perturbed_score, expression_map
        )

    except Exception as perturbation_error:
        # If a Structural Integrity Threshold is broken, wrap it cleanly for the pool tracker
        raise ProcessorError(
            f"Humanizer pipeline failed on {filename}: {perturbation_error}"
        ) from perturbation_error

    # 3. Stateless Audio Rendering Pipe using RAM-isolated local cache storage (/tmp)
    midi_exporter = MidiExporter()
    convolver = RirConvolver()
    engine = SfizzRenderEngine(**engine_config)

    final_destination: Path = output_dir / f"{filename}.wav"

    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as local_cache:
            cache_path: Path = Path(local_cache)
            temp_midi: Path = cache_path / "scratch.mid"
            temp_dry_wav: Path = cache_path / "scratch_dry.wav"
            temp_wet_wav: Path = cache_path / "scratch_wet.wav"

            midi_exporter.export_score(perturbed_score, temp_midi)
            engine.render_audio(midi_path=temp_midi, output_wav_path=temp_dry_wav)

            convolver.process_audio(
                dry_wav_path=temp_dry_wav,
                rir_path=rir_path,
                output_wav_path=temp_wet_wav,
            )

            shutil.copy(temp_wet_wav, final_destination)

    finally:
        # Force instantaneous garbage collection of unmanaged memory vectors
        del midi_exporter
        del convolver
        del engine
        del event_mod
        del tempo_gen
        del jitter_mod
        del artic_mod
        del intensity_mod

    return str(final_destination)


class DatasetGenerator:
    """Orchestrates parallel dataset synthesis beneath volatile university memory caps."""

    def __init__(
        self, engine_configs: list[dict[str, Any]], rir_paths: list[str]
    ) -> None:
        """Initializes the batch generator with balanced asset configurations."""
        if not engine_configs or not rir_paths:
            raise ProcessorError("Asset initialization vectors cannot be empty arrays.")

        self.engine_configs: list[dict[str, Any]] = engine_configs
        self.rir_pool: list[Path] = [Path(p).resolve() for p in rir_paths]

    def generate_batch(
        self,
        score_tuples: list[tuple[PerformanceScore, ScoreExpressionMap]],
        output_dir: str,
        variation_count: int = 1,
        base_seed: int = 42,
    ) -> list[Path]:
        """Distributes multi-environment humanized score variations across a streaming pool.

        Args:
            score_tuples: A list containing tuples of (PerformanceScore, ScoreExpressionMap)
                as returned by your refactored MusicXMLParser.
            output_dir: Target output location where final wet WAV files are written.
            variation_count: The total number of distinct environmental variations to make.
            base_seed: Global anchor seed value to guarantee reproducible random sequences.

        Returns:
            A list of Path locations tracking every successfully generated file variation.
        """
        resolved_out_dir: Path = Path(output_dir).resolve()
        resolved_out_dir.mkdir(parents=True, exist_ok=True)

        worker_concurrency: int = self._calculate_conservative_worker_limit()
        task_payload: list[
            tuple[
                PerformanceScore,
                ScoreExpressionMap,
                dict[str, Any],
                Path,
                Path,
                str,
                int,
            ]
        ] = []
        matrix_index: int = 0

        for score, expression_map in score_tuples:
            base_name: str = score.source.stem

            for v_idx in range(variation_count):
                instance_filename: str = f"{base_name}_var_{v_idx}"

                # Deriving a mathematically unique, predictable variant seed for this specific execution loop
                variant_seed: int = base_seed + matrix_index

                # Decouple the data structure states entirely from the parent thread context loop
                mutated_score: PerformanceScore = copy.deepcopy(score)
                mutated_map: ScoreExpressionMap = copy.deepcopy(expression_map)

                selected_config: dict[str, Any] = self.engine_configs[
                    matrix_index % len(self.engine_configs)
                ]
                selected_rir: Path = self.rir_pool[matrix_index % len(self.rir_pool)]
                matrix_index += 1

                task_payload.append(
                    (
                        mutated_score,
                        mutated_map,
                        selected_config,
                        selected_rir,
                        resolved_out_dir,
                        instance_filename,
                        variant_seed,
                    )
                )

        print(
            f"\n🚀 Launching Integrated Humanizer Stream Engine [{worker_concurrency} Cores Active]"
        )
        print(
            f"📦 Total target dataset variation matrix size: {len(task_payload)} files"
        )

        completed_records: list[Path] = []
        pool_context = multiprocessing.get_context("spawn")

        with pool_context.Pool(
            processes=worker_concurrency, maxtasksperchild=1
        ) as stream_pool:
            task_stream = stream_pool.imap_unordered(
                _isolated_render_thunk, task_payload
            )

            for idx, result_path_str in enumerate(task_stream, start=1):
                completed_records.append(Path(result_path_str))
                print(
                    f"  ✅ [{idx}/{len(task_payload)}] Humanized & Rendered variation: {Path(result_path_str).name}"
                )

                import gc

                gc.collect()

        print(
            f"\n🎉 Generation successful! {len(completed_records)} files compiled smoothly beneath memory caps.\n"
        )
        return completed_records

    def _calculate_conservative_worker_limit(self) -> int:
        """Computes hardware allocation safety limits."""
        hardware_cores: int = os.cpu_count() or 1
        return max(1, int(hardware_cores * 0.5))
