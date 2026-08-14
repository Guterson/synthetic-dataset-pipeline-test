"""Automated verification suite testing the RemoteClusterNode networking client.

Validates remote nividia-smi parser strings, detached background PID captures,
and network failure exception pathways safely using mock framework transports.
"""

from unittest.mock import MagicMock, patch

import paramiko

from score2dataset.scripts.cluster_nodes import RemoteClusterNode

# ---1. NETWORK DATA INTERFACE PARSER TESTS---


@patch.object(RemoteClusterNode, "_establish_client")
def test_query_gpu_utilization_parses_csv_output_accurately(mock_connect):
    """Verifies that the engine correctly extracts utilization integers from nvidia-smi strings."""
    node = RemoteClusterNode(host_name="test-server")

    # Assemble mock stdout stream simulation data layers
    mock_client = MagicMock()
    mock_stdout = MagicMock()
    # Simulate a 2-GPU server setup returning 12% and 85% utilization metrics
    mock_stdout.read.return_value = b"12\n85\n"

    mock_client.exec_command.return_value = (None, mock_stdout, None)
    mock_connect.return_value = mock_client

    gpu_metrics = node.query_gpu_utilization()

    assert len(gpu_metrics) == 2
    assert gpu_metrics[0] == 12
    assert gpu_metrics[1] == 85
    assert mock_client.close.called


@patch.object(RemoteClusterNode, "_establish_client")
def test_query_active_gpu_pids_unrolls_unique_process_sets(mock_connect):
    """Verifies that compute applications are isolated and converted into an integer set."""
    node = RemoteClusterNode(host_name="test-server")

    mock_client = MagicMock()
    mock_stdout = MagicMock()
    # Simulate multiple processing PIDs running on a targeted hardware layout card
    mock_stdout.read.return_value = b"28841\n31002\n"

    mock_client.exec_command.return_value = (None, mock_stdout, None)
    mock_connect.return_value = mock_client

    active_pids = node.query_active_gpu_pids(gpu_index=0)

    assert isinstance(active_pids, set)
    assert len(active_pids) == 2
    assert 28841 in active_pids
    assert 31002 in active_pids


# ---2. DETACHED JOB LAUNCH & INTERRUPT TESTS---


@patch.object(RemoteClusterNode, "_establish_client")
def test_launch_background_job_captures_issued_process_pid(mock_connect):
    """Verifies that the nohup wrapper successfully returns the trailing background target PID."""
    node = RemoteClusterNode(host_name="test-server")

    mock_client = MagicMock()
    mock_stdout = MagicMock()
    mock_stdout.read.return_value = b"4012\n"  # OS issued system process PID

    mock_client.exec_command.return_value = (None, mock_stdout, None)
    mock_connect.return_value = mock_client

    pid = node.launch_background_job("python3 run_loop.py")

    assert pid == 4012
    # Verify that the bash command string utilizes correct nohup background wrapping tokens
    fired_command = mock_client.exec_command.call_args[0][0]
    assert "nohup" in fired_command
    assert "echo $!" in fired_command


# ---3. ELEGANT EXCEPTION SAFE BOUNDARY TESTS---


@patch.object(RemoteClusterNode, "_establish_client")
def test_query_gpu_utilization_handles_network_dropouts_gracefully(
    mock_connect, capsys
):
    """Verifies that Paramiko connection drops are caught gracefully without crashing loops."""
    node = RemoteClusterNode(host_name="offline-server")

    mock_client = MagicMock()
    # Force the command pipe to raise a strict SSH connection exception dropout
    mock_client.exec_command.side_effect = paramiko.SSHException(
        "Connection timed out reset."
    )
    mock_connect.return_value = mock_client

    gpu_metrics = node.query_gpu_utilization()

    # Crucial assertion: Returns an empty safe fallback dict instead of crashing the thread loop
    assert not gpu_metrics
    assert mock_client.close.called

    # Verify our elegant warning was logged safely to the console tracking streams
    captured = capsys.readouterr()
    assert "Remote cluster network exception" in captured.err
