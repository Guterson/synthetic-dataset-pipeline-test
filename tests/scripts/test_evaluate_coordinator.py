"""Automated pipeline validation checking metric accuracy and coordinator mechanics.

Validates distributed concurrent coordinator process scheduling tracking loops,
resource depletion queues, and data extraction bounds without executing real network SSH calls.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from score2dataset.scripts.evaluate_coordinator import main
from score2dataset.scripts.static_parser import JobTask

# ---1. LOGICAL CLUSTER ORCHESTRATION COMPONENT TESTS---


@patch("score2dataset.scripts.evaluate_coordinator.get_ssh_hosts")
@patch("score2dataset.scripts.evaluate_coordinator.ModelStaticParser")
@patch("score2dataset.scripts.evaluate_coordinator.LowPriorityClusterScheduler")
def test_coordinator_exits_gracefully_when_hosts_missing(
    _mock_scheduler, _mock_parser, mock_get_hosts
):
    """Verifies that the main engine drops out with error code 1 if profiles are absent."""
    mock_get_hosts.return_value = (
        []
    )  # Return empty array tracking available connections

    with pytest.raises(SystemExit) as exit_check:
        main()

    assert exit_check.value.code == 1


@patch("score2dataset.scripts.evaluate_coordinator.get_ssh_hosts")
@patch("score2dataset.scripts.evaluate_coordinator.ModelStaticParser")
@patch("score2dataset.scripts.evaluate_coordinator.LowPriorityClusterScheduler")
def test_coordinator_exits_gracefully_when_jobs_missing(
    _mock_scheduler, mock_parser, mock_get_hosts
):
    """Verifies that the main engine drops out with error code 1 if parser fails to locate components."""
    mock_get_hosts.return_value = ["server-node-01"]

    # Configure parser instance mock to return no jobs
    parser_instance = MagicMock()
    parser_instance.extract_executable_jobs.return_value = []
    mock_parser.return_value = parser_instance

    with pytest.raises(SystemExit) as exit_check:
        main()

    assert exit_check.value.code == 1


# ---2. HARDWARE SCHEDULING AND SYNC LOOP MATRIX TESTS---


@patch("score2dataset.scripts.evaluate_coordinator.get_ssh_hosts")
@patch("score2dataset.scripts.evaluate_coordinator.ModelStaticParser")
@patch("score2dataset.scripts.evaluate_coordinator.LowPriorityClusterScheduler")
@patch("score2dataset.scripts.evaluate_coordinator.RemoteClusterNode")
@patch("pathlib.Path.exists")
@patch("numpy.load")
def test_coordinator_handles_successful_parallel_execution_cycle(
    mock_np_load, mock_exists, _mock_node, mock_scheduler, mock_parser, mock_get_hosts
):
    """Verifies that coordinator maps processes, triggers jobs, and aggregates outcomes."""
    mock_get_hosts.return_value = ["node-alpha"]

    # 1. Setup mock syntax tree outputs tracking a single model type
    fake_job = JobTask(
        model_name="OVOnsetClassifier", script_path="models/ov.py", dataset_argument=""
    )
    parser_instance = MagicMock()
    parser_instance.extract_executable_jobs.return_value = [fake_job]
    mock_parser.return_value = parser_instance

    # 2. Setup mock cluster monitoring interactions
    node_mock = MagicMock()
    node_mock.host_name = "node-alpha"
    node_mock.launch_background_job.return_value = (
        99421  # Return a mock OS process PID integer
    )

    scheduler_instance = MagicMock()
    scheduler_instance.get_available_resources.return_value = [(node_mock, 0)]
    mock_scheduler.return_value = scheduler_instance

    # 3. Setup mock monitoring tracking loop completion mechanics
    # First query returns the active PID running, second sweep returns an empty set indicating finish
    node_mock.query_active_gpu_pids.side_effect = [{99421}, set()]

    # 4. Mock disk files to bypass local operational requirements safely
    mock_exists.return_value = (
        True  # Forces all validation cache presence checks to pass
    )
    mock_np_load.side_effect = [
        np.array([(1.0, 60)], dtype=object),  # Expected ground truth timeline entries
        np.array(
            [(1.0, 60)], dtype=object
        ),  # Normalized model outcome timeline entries
    ]

    # Run the comprehensive aggregation routine loop
    with patch(
        "score2dataset.scripts.evaluate_coordinator.calculate_unified_mir_score"
    ) as mock_metric:
        mock_metric.return_value = {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "mae_ms": 0.0,
            "unified_score": 1.0,
        }

        with patch("pathlib.Path.unlink") as mock_unlink:
            main()

            # Assertions tracking structural engine completeness maps
            assert node_mock.launch_background_job.called
            assert node_mock.query_active_gpu_pids.call_count == 2
            assert mock_metric.called
            assert mock_unlink.called
