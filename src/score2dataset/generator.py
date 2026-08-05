"""Integrated parallel orchestration engine for synthetic dataset generation.

Applies sequential humanizing perturbations (Event, Tempo, Jitter, Articulation, Intensity)
entirely in memory before executing stateless, local cache audio renders.
"""

import copy
import gc
import glob
import hashlib
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
        PerformanceScore, ScoreExpressionMap, dict[str, Any], Path, Path, Path, int
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
        audio_out_dir,
        annotations_out_dir,
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
            f"Humanizer pipeline failed on {base_score.source.name} (Seed: {variation_seed}): {perturbation_error}"
        ) from perturbation_error

    # 3. Stateless Audio Rendering Pipe using RAM-isolated local cache storage (/tmp)
    midi_exporter = MidiExporter()
    convolver = RirConvolver()
    engine = SfizzRenderEngine(**engine_config)

    # Extract the absolute SFZ path directly from your engine configuration
    sfz_asset_path = Path(engine_config["sfz_path"])

    # Construct the audit payload string using the distinct transformation inputs
    audit_payload = f"{base_score.source.name}_{sfz_asset_path.name}_{rir_path.name}_{variation_seed}"
    hashed_id = hashlib.sha256(audit_payload.encode("utf-8")).hexdigest()[:16]

    # Assign the destination path using the concise 16-character hexadecimal hash identifier
    final_destination: Path = audio_out_dir / f"{hashed_id}.wav"

    # Write a completely isolated, thread-safe micro-manifest for this specific variation
    sidecar_manifest: Path = audio_out_dir / ".." / f"{hashed_id}.manifest"
    with open(sidecar_manifest, mode="w", encoding="utf-8") as f:
        f.write(
            f"{hashed_id},{base_score.source.name},{sfz_asset_path.name},{rir_path.name},{variation_seed}\n"
        )

    csv_truth_path: Path = annotations_out_dir / f"{hashed_id}.csv"
    all_tracked_events = list(perturbed_score.events) + getattr(
        perturbed_score, "omitted_events", []
    )
    all_tracked_events.sort(key=lambda e: e.onset_ticks)

    with open(csv_truth_path, mode="w", encoding="utf-8") as csv_file:
        csv_file.write("onset_time,pitch_midi,score_expected,audio_present\n")
        for event in all_tracked_events:
            onset_seconds: float = float(event.onset_ticks / 960.0)
            csv_file.write(
                f"{onset_seconds:.7f},{event.pitch},{event.score_expected},{event.audio_present}\n"
            )

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

    # Returning both the path and the hash allows the outer pool manager to easily map
    # the structured parameters straight into your global 'manifest.csv' file.
    return f"{hashed_id}|{final_destination}"


class DatasetGenerator:
    """Orchestrates parallel dataset synthesis beneath volatile memory caps."""

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
        output_path: Path = Path(output_dir).resolve()
        audio_out_path: Path = output_path / "audio"
        annotations_out_path: Path = output_path / "annotations"

        audio_out_path.mkdir(parents=True, exist_ok=True)
        annotations_out_path.mkdir(parents=True, exist_ok=True)

        worker_concurrency: int = self._calculate_conservative_worker_limit()
        task_payload: list[
            tuple[
                PerformanceScore,
                ScoreExpressionMap,
                dict[str, Any],
                Path,
                Path,
                Path,
                int,
            ]
        ] = []
        matrix_index: int = 0

        for score, expression_map in score_tuples:

            for _ in range(variation_count):

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
                        audio_out_path,
                        annotations_out_path,
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

                gc.collect()

        # Run a synchronized aggregation loop to merge all isolated thread-safe micro-manifest sidecar files
        manifest_csv_path: Path = output_path / "manifest.csv"
        manifest_search_pattern: str = str(output_path / "*.manifest")

        print("📊 Consolidating centralized dataset auditing ledger...")
        with open(manifest_csv_path, mode="w", encoding="utf-8") as master_manifest:
            # Write the single, canonical column schema header
            master_manifest.write(
                "file_hash,piece,sfz_instrument,rir_convolution,seed\n"
            )

            # Read each sidecar file, append its data row, and delete the temporary footprint
            for sidecar_path_str in glob.glob(manifest_search_pattern):
                sidecar_path = Path(sidecar_path_str)
                master_manifest.write(sidecar_path.read_text(encoding="utf-8"))
                sidecar_path.unlink()

        print(
            f"\n🎉 Generation successful! {len(completed_records)} files compiled smoothly beneath memory caps.\n"
        )
        return completed_records

    def _calculate_conservative_worker_limit(self) -> int:
        """Computes hardware allocation safety limits."""
        hardware_cores: int = os.cpu_count() or 1
        return max(1, int(hardware_cores * 0.7))
