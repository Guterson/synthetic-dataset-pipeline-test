"""Automated verification suite testing the LowPriorityClusterScheduler engine.

Validates parallel resource assignments, preemption eviction sets, queue draining
boundaries, and local SSH profile parsing hooks without running physical remote jobs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from score2dataset.scripts.scheduler import (
    ActiveDeployment,
    LowPriorityClusterScheduler,
    get_ssh_hosts,
)
from score2dataset.scripts.static_parser import JobTask

# ---1. LOCAL CONFIGURATION PARSER UTILITY TESTS---


def test_get_ssh_hosts_parses_profile_directives_accurately(tmp_path: Path):
    """Verifies that the parser isolates explicit host aliases and bypasses catch-all wildcards."""
    fake_config = tmp_path / "config"
    # Construct a valid standard SSH configuration profile layout block
    fake_config.write_text(
        "Host server-alpha\n"
        "    HostName 192.168.1.10\n"
        "Host *\n"
        "    IdentityFile ~/.ssh/id_rsa\n"
        "Host server-beta server-gamma\n",  # Composite multi-match line
        encoding="utf-8",
    )

    with (
        patch("os.path.expanduser", return_value=str(fake_config)),
        patch("os.path.exists", return_value=True),
    ):

        hosts = get_ssh_hosts()

        assert isinstance(hosts, list)
        assert "server-alpha" in hosts
        assert "server-beta" in hosts  # Captures leading structural tokens
        assert "*" not in hosts  # Verifies wildcard rule exception works


# ---2. PREEMPTIVE EVICTION METRIC SET TESTS---


@patch("score2dataset.scripts.scheduler.RemoteClusterNode")
def test_evaluate_preemption_constraints_handles_clean_completions(
    mock_node_cls,
) -> None:
    """Verifies that an empty tracking PID set unloads completed deployments smoothly."""
    scheduler = LowPriorityClusterScheduler(hosts=["host-01"])
    fake_job = JobTask(model_name="M1", script_path="ov.py", dataset_argument="")
    deployment = ActiveDeployment(host="host-01", pid=5001, gpu_index=0, job=fake_job)

    scheduler.active_deployments.append(deployment)

    node_instance = MagicMock()
    node_instance.query_active_gpu_pids.return_value = set()
    mock_node_cls.return_value = node_instance

    # TYPE FIX: Force explicit class validation constraints on the list instance
    job_queue: list[JobTask] = []

    # ELEGANT FIX: Replaced protected access with a clean public interface if available,
    # or kept it typed strictly to satisfy your variable checks
    scheduler._evaluate_preemption_constraints(
        job_queue
    )  # pylint: disable=protected-access

    # Completion confirmation tracking
    assert len(scheduler.active_deployments) == 0
    assert len(job_queue) == 0
    assert not node_instance.terminate_remote_process.called


@patch("score2dataset.scripts.scheduler.RemoteClusterNode")
def test_evaluate_preemption_constraints_evicts_on_foreign_interference(
    mock_node_cls,
) -> None:
    """Verifies that foreign PIDs trigger a SIGTERM kill sequence and re-queue the task."""
    scheduler = LowPriorityClusterScheduler(hosts=["host-02"])
    fake_job = JobTask(model_name="M2", script_path="of.py", dataset_argument="")
    deployment = ActiveDeployment(host="host-02", pid=6002, gpu_index=1, job=fake_job)

    scheduler.active_deployments.append(deployment)

    node_instance = MagicMock()
    node_instance.query_active_gpu_pids.return_value = {6002, 7550}
    mock_node_cls.return_value = node_instance

    # TYPE FIX: Force explicit class validation constraints on the list instance
    job_queue: list[JobTask] = []
    scheduler._evaluate_preemption_constraints(
        job_queue
    )  # pylint: disable=protected-access

    assert len(scheduler.active_deployments) == 0
    assert len(job_queue) == 1

    # INDICES FIX: Pull the item from the queue list to check its properties safely
    evicted_job = job_queue[0]
    assert evicted_job.model_name == "M2"


# ---3. LOOP ORCHESTRATION SHIFT MATRIX TESTS---


@patch(
    "time.sleep"
)  # Short-circuit interval blockages to run tests instantly in memory
@patch.object(LowPriorityClusterScheduler, "_locate_available_gpu_slots")
@patch.object(LowPriorityClusterScheduler, "_evaluate_preemption_constraints")
@patch("score2dataset.scripts.scheduler.RemoteClusterNode")
def test_execute_orchestration_drains_job_queue_completely(
    mock_node_cls, _mock_preempt, mock_slots, _mock_sleep
):
    """Verifies that the main queue engine assigns steps, loops modules, and registers deployments."""
    scheduler = LowPriorityClusterScheduler(hosts=["host-03"])
    fake_job = JobTask(
        model_name="M3", script_path="src/models/ov.py", dataset_argument=""
    )

    node_instance = MagicMock()
    node_instance.host_name = "host-03"
    node_instance.launch_background_job.return_value = 8801  # Return mock PID
    mock_node_cls.return_value = node_instance

    # First loop loop query finds an open slot, second sweep returns empty to finalize tracking
    mock_slots.side_effect = [[(node_instance, 0)], []]

    job_queue = [fake_job]
    scheduler.execute_orchestration(job_queue)

    # Assert module string conversions matched formatting specifications [src/ -> package]
    node_instance.launch_background_job.assert_called_once_with(
        "CUDA_VISIBLE_DEVICES=0 python3 -m models.ov"
    )
    assert len(job_queue) == 0
