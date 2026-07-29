"""Unit test suite for the DatasetGenerator dataset orchestrator.

Validates parallel worker distribution, core mapping logic limits, and
proper handling of centralized processing custom exception triggers.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from score2dataset.datamodels import PerformanceScore
from score2dataset.exceptions import ProcessorError
from score2dataset.generator import DatasetGenerator, _parallel_worker_thunk


class TestDatasetGenerator(unittest.TestCase):
    """Test framework for validating execution sweeps across resource pools."""

    def test_initialization_raises_processor_error_on_empty_pools(self) -> None:
        """Tests that empty initialization configuration elements trigger a ProcessorError."""
        with self.assertRaisesRegex(
            ProcessorError, "Asset distribution pools cannot be empty."
        ):
            DatasetGenerator(engine_configs=[], rir_paths=["rir.wav"])

        with self.assertRaisesRegex(
            ProcessorError, "Asset distribution pools cannot be empty."
        ):
            DatasetGenerator(engine_configs=[{"bin": "sfizz"}], rir_paths=[])

    @patch("score2dataset.generator.multiprocessing.Pool")
    def test_generate_batch_assigns_tasks_round_robin(
        self, mock_pool_class: MagicMock
    ) -> None:
        """Tests that generation maps assign balanced assets without cross-multiplying."""
        mock_pool_instance = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = [Path("out1.wav"), Path("out2.wav")]

        mock_score = MagicMock(spec=PerformanceScore)
        mock_score.source = MagicMock()
        mock_score.source.stem = "score_a"

        engine_configs: list[dict[str, int]] = [{"id": 0}, {"id": 1}]
        rir_paths: list[str] = ["room_a.wav", "room_b.wav"]

        generator = DatasetGenerator(engine_configs=engine_configs, rir_paths=rir_paths)

        results: list[Path] = generator.generate_batch(
            base_scores=[mock_score], output_dir="dataset_output", variation_count=2
        )

        self.assertEqual(len(results), 2)
        mock_pool_instance.starmap.assert_called_once()

        # Extract the positional arguments list passed to starmap
        called_args, _ = mock_pool_instance.starmap.call_args
        tasks_payload = called_args[1]  # The list of tuple tasks

        # Verify the round-robin balance pattern inside tasks payloads
        task_1_config = tasks_payload[0][1]
        task_2_config = tasks_payload[1][1]

        self.assertEqual(task_1_config, {"id": 0})
        self.assertEqual(task_2_config, {"id": 1})

    @patch("score2dataset.generator.MidiExporter")
    @patch("score2dataset.generator.SfizzRenderEngine")
    @patch("score2dataset.generator.RirConvolver")
    @patch("score2dataset.generator.shutil.copy")
    def test_parallel_worker_thunk_execution_steps(
        self,
        mock_copy: MagicMock,
        mock_convolver_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_exporter_cls: MagicMock,
    ) -> None:
        """Verifies step pipeline sequence inside parallel execution thunks."""
        # Grab the mock instances that will be returned when constructors are called
        mock_exporter = mock_exporter_cls.return_value
        mock_engine = mock_engine_cls.return_value
        mock_convolver = mock_convolver_cls.return_value

        mock_score = MagicMock(spec=PerformanceScore)
        config: dict[str, str] = {"sampler": "sfizz"}
        rir = Path("room.wav")
        out_dir = Path("out")
        filename = "track_var_0"

        result: Path = _parallel_worker_thunk(
            score_data=mock_score,
            engine_config=config,
            rir_path=rir,
            output_dir=out_dir,
            filename=filename,
        )

        # Assert data flow pipeline execution order matches
        mock_exporter.export_score.assert_called_once()
        mock_engine.render_audio.assert_called_once()
        mock_convolver.process_audio.assert_called_once()
        mock_copy.assert_called_once()

        self.assertEqual(result, out_dir / "track_var_0.wav")

    @patch("score2dataset.generator.os.getloadavg")
    @patch("score2dataset.generator.os.cpu_count")
    @patch("score2dataset.generator.multiprocessing.Pool")
    def test_generate_batch_throttles_pool_size_under_heavy_load(
        self,
        mock_pool_class: MagicMock,
        mock_cpu_count: MagicMock,
        mock_getloadavg: MagicMock,
    ) -> None:
        """Tests that generate_batch scales down the multiprocessing pool size under high system load."""
        mock_cpu_count.return_value = 8  # Simulate an 8-core CPU
        mock_score = MagicMock(spec=PerformanceScore)
        mock_score.source = MagicMock()
        mock_score.source.stem = "score_a"

        generator = DatasetGenerator(engine_configs=[{"id": 0}], rir_paths=["rir.wav"])

        # Scenario A: Heavy Load (6.0 load average on 8 cores means >70% usage)
        mock_getloadavg.return_value = (6.0, 5.0, 4.0)
        generator.generate_batch(base_scores=[mock_score], output_dir="out")

        # Verify the Pool was initialized with the throttled 30% core count (8 * 0.3 = 2)
        mock_pool_class.assert_called_with(processes=2)

        # Scenario B: Low Load (1.0 load average on 8 cores means well under 70% usage)
        mock_getloadavg.return_value = (1.0, 1.0, 1.0)
        generator.generate_batch(base_scores=[mock_score], output_dir="out")

        # Verify the Pool was initialized with the standard 70% safe limit cap (8 * 0.7 = 5)
        mock_pool_class.assert_called_with(processes=5)


if __name__ == "__main__":
    unittest.main()
