"""Module-level test suite for DatasetGenerator orchestration engine.

Validates process worker limit limits, batch matrix payload distribution,
and error catching across parallel stream pools via public boundaries.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from score2dataset.datamodels import PerformanceScore, ScoreExpressionMap
from score2dataset.exceptions import ProcessorError
from score2dataset.generator import DatasetGenerator


@pytest.fixture(name="valid_generator")
def fixture_valid_generator() -> DatasetGenerator:
    """Provides a valid generator instance pre-configured with mocked boundaries."""
    engine_configs: list[dict[str, str]] = [
        {"sampler": "sfizz_rhodes"},
        {"sampler": "sfizz_grand"},
    ]
    rir_paths: list[str] = ["/assets/rirs/hall.wav", "/assets/rirs/room.wav"]
    return DatasetGenerator(engine_configs=engine_configs, rir_paths=rir_paths)


@pytest.fixture(name="mock_score_pair")
def fixture_mock_score_pair() -> tuple[MagicMock, MagicMock]:
    """Provides an isolated score data target paired with an expression metadata layout."""
    mock_score = MagicMock(spec=PerformanceScore)
    mock_score.source = Path("/data/scores/sonata_no_1.musicxml")

    mock_map = MagicMock(spec=ScoreExpressionMap)
    return mock_score, mock_map


def test_initialization_empty_pool_errors() -> None:
    """Business Rule: Instantiation vectors cannot be empty arrays.

    Ensures ProcessorError is cleanly triggered with an accurate feedback trace.
    """
    # Validate that an empty engine config pool cleanly triggers the domain exception contract
    with pytest.raises(ProcessorError):
        DatasetGenerator(engine_configs=[], rir_paths=["rir.wav"])

    # Validate that an empty RIR path pool cleanly triggers the identical domain exception contract
    with pytest.raises(ProcessorError):
        DatasetGenerator(engine_configs=[{"id": 1}], rir_paths=[])


@patch("score2dataset.generator.multiprocessing.get_context")
@patch("score2dataset.generator.os.cpu_count")
def test_worker_limit_scales_pool_implicitly_via_public_api(
    mock_cpu_count: MagicMock,
    mock_get_context: MagicMock,
    valid_generator: DatasetGenerator,
    mock_score_pair: tuple[MagicMock, MagicMock],
    tmp_path: Path,
) -> None:
    """Business Rule: Parallel pool processes must target exactly 70% of available CPU cores.

    Verifies hardware allocation safety limits implicitly by checking the arguments
    passed to the Pool constructor during a public batch execution run.
    """
    # Mock out 16 available hardware threads on the processing node
    mock_cpu_count.return_value = 16

    mock_ctx_instance = MagicMock()
    mock_pool_instance = MagicMock()
    mock_get_context.return_value = mock_ctx_instance
    mock_ctx_instance.Pool.return_value.__enter__.return_value = mock_pool_instance
    mock_pool_instance.imap_unordered.return_value = []

    # Run the public API method
    valid_generator.generate_batch(
        score_tuples=[mock_score_pair], output_dir=str(tmp_path), variation_count=1
    )

    # 3. Verify that the system scales allocation to 70% of raw processing capacity (16 cores * 0.7 = 11.2 -> int(11))
    mock_ctx_instance.Pool.assert_called_once_with(processes=11, maxtasksperchild=1)


@patch("score2dataset.generator.multiprocessing.get_context")
def test_generate_batch_payload_distribution(
    mock_get_context: MagicMock,
    valid_generator: DatasetGenerator,
    mock_score_pair: tuple[MagicMock, MagicMock],
    tmp_path: Path,
) -> None:
    """Business Rule: Payloads must map configurations in balanced round-robin steps.

    Validates that the public generate_batch method maps variant seeds, configs,
    and output directories flawlessly into the multiprocessing task matrix.
    """
    # 1. Setup deep multiprocessing pool mock objects natively matching the 'spawn' frame
    mock_ctx_instance = MagicMock()
    mock_pool_instance = MagicMock()

    mock_get_context.return_value = mock_ctx_instance
    mock_ctx_instance.Pool.return_value.__enter__.return_value = mock_pool_instance

    # Simulate the imap_unordered streaming process yielding deterministic hash string paths back
    mock_pool_instance.imap_unordered.return_value = [
        str(tmp_path / "audio" / "5d41402abc4b2a76.wav"),
        str(tmp_path / "audio" / "8f91a34bba4c7b12.wav"),
    ]

    # 2. Execute pipeline matrix run requesting two environmental variations
    base_seed = 100
    results: list[Path] = valid_generator.generate_batch(
        score_tuples=[mock_score_pair],
        output_dir=str(tmp_path),
        variation_count=2,
        base_seed=base_seed,
    )

    # 3. Assert results structures map correctly to our dynamic output hash arrays
    assert len(results) == 2
    assert results[0] == tmp_path / "audio" / "5d41402abc4b2a76.wav"

    # 4. Extract the exact task payloads pushed into the multiprocessing stream channel
    mock_pool_instance.imap_unordered.assert_called_once()
    called_args, _ = mock_pool_instance.imap_unordered.call_args
    task_payload = called_args[1]

    # Verify that variation matrix distribution parameters match step-by-step
    assert len(task_payload) == 2

    # Task 0 (Variation 0) Evaluation
    task_0_args = task_payload[0]
    assert (
        task_0_args[4] == tmp_path / "audio"
    )  # Position 4 is now the explicit audio dir
    assert (
        task_0_args[5] == tmp_path / "annotations"
    )  # Position 5 is now the explicit annotations dir
    assert (
        task_0_args[6] == base_seed + 0
    )  # Position 6 captures your variant seed cleanly
    assert task_0_args[2] == {"sampler": "sfizz_rhodes"}

    # Task 1 (Variation 1) Evaluation
    task_1_args = task_payload[1]
    assert task_1_args[4] == tmp_path / "audio"
    assert task_1_args[5] == tmp_path / "annotations"
    assert task_1_args[6] == base_seed + 1

    assert task_1_args[2] == {
        "sampler": "sfizz_grand"
    }  # Second engine config round-robin loop
