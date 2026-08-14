"""Asynchronous task orchestrator and preemptive resource scheduler.

This module runs the main job distribution loop across independent cluster nodes.
It balances active model deployments concurrently based on ongoing hardware load
profiles and enforces strict low-priority constraints by evicting running tasks
gracefully if foreign, higher-priority workloads are detected.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from score2dataset.scripts.cluster_nodes import RemoteClusterNode
from score2dataset.scripts.static_parser import JobTask, ModelStaticParser


class ActiveDeployment:
    """Tracks state attributes of an active remote training process."""

    def __init__(self, host: str, pid: int, gpu_index: int, job: JobTask) -> None:
        """Initializes structural tracking fields for an active background job.

        Args:
            host: Name alias of the server executing the workload.
            pid: Remote process identifier issued by the operating system.
            gpu_index: Isolated hardware GPU index slot assigned to the task.
            job: The specific metadata tracking token representing the model.
        """
        self.host = host
        self.pid = pid
        self.gpu_index = gpu_index
        self.job = job


class LowPriorityClusterScheduler:
    """Orchestrates job deployment, continuous GPU monitoring, and preemption loops."""

    def __init__(
        self,
        hosts: list[str],
        utilization_limit: int = 70,
        monitor_interval: int = 30,
    ) -> None:
        """Initializes parameters for scheduler execution.

        Args:
            hosts: A list of SSH host profiles parsed matching local definitions.
            utilization_limit: Allowed GPU load ceiling prior to exclusion marks.
            monitor_interval: Sleep duration scale between runtime audit sweeps.
        """
        self.nodes = [RemoteClusterNode(h) for h in hosts]
        self.utilization_limit = utilization_limit
        self.monitor_interval = monitor_interval
        self.active_deployments: list[ActiveDeployment] = []

    def _locate_available_gpu_slots(self) -> list[tuple[RemoteClusterNode, int]]:
        """Concurrently audits cluster nodes to look up unassigned GPU indexes.

        Returns:
            List[Tuple[RemoteClusterNode, int]]: Pairs of nodes and free GPU slots.
        """
        available_slots: list[tuple[RemoteClusterNode, int]] = []

        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            future_to_node = {
                executor.submit(node.query_gpu_utilization): node for node in self.nodes
            }

            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    gpu_metrics = future.result()
                    for gpu_idx, load in gpu_metrics.items():
                        # Protect active allocations from being double-assigned
                        already_assigned = any(
                            d.host == node.host_name and d.gpu_index == gpu_idx
                            for d in self.active_deployments
                        )
                        if load < self.utilization_limit and not already_assigned:
                            available_slots.append((node, gpu_idx))
                except Exception:
                    print(
                        f"⚠️ Error querying resource states on host: {node.host_name}"
                    )

        return available_slots

    def get_available_resources(self) -> list[tuple[RemoteClusterNode, int]]:
        """Public interface providing an audited matrix of unassigned cluster GPU slots.

        Returns:
            list[tuple[RemoteClusterNode, int]]: Pairs of nodes and free GPU slots.
        """
        # Exposes the internal slot lookup safely to external scripts (like evaluation)
        return self._locate_available_gpu_slots()

    def execute_orchestration(self, job_queue: list[JobTask]) -> None:
        """Drains the global dynamic job queue under polite preemption limits.

        Args:
            job_queue: Discovered JobTask models ready for grid scaling executions.
        """
        print(
            f"Beginning scheduler execution pipeline. Pending tasks: {len(job_queue)}"
        )

        while job_queue or self.active_deployments:
            # 1. Audit active worker health profiles and check for preemptions
            self._evaluate_preemption_constraints(job_queue)

            # 2. Assign pending workloads if free cluster targets exist
            if job_queue:
                free_slots = self._locate_available_gpu_slots()
                for node, gpu_idx in free_slots:
                    if not job_queue:
                        break

                    next_job = job_queue.pop(0)

                    # Convert file path to a safe, runnable module notation string
                    # e.g., src/score2dataset/models/bytedance.py -> score2dataset.models.bytedance
                    relative_module = (
                        next_job.script_path.replace("src/", "")
                        .replace(".py", "")
                        .replace("/", ".")
                    )

                    execution_str = (
                        f"CUDA_VISIBLE_DEVICES={gpu_idx} python3 -m {relative_module}"
                    )

                    pid = node.launch_background_job(execution_str)
                    if pid:
                        deployment = ActiveDeployment(
                            host=node.host_name,
                            pid=pid,
                            gpu_index=gpu_idx,
                            job=next_job,
                        )
                        self.active_deployments.append(deployment)
                        print(
                            f"🚀 Job allocated on [{node.host_name}] GPU {gpu_idx} (PID: {pid})"
                        )
                    else:
                        # Re-queue task if remote tracking registration drops out unexpectedly
                        job_queue.insert(0, next_job)

            time.sleep(self.monitor_interval)

        print("🎉 Execution success. All dynamic job workloads handled completely.")

    def _evaluate_preemption_constraints(self, job_queue: list[JobTask]) -> None:
        """Inspects targeted slots to halt tasks if external workloads appear.

        Args:
            job_queue: Global job queue to append evicted tasks to.
        """
        surviving_deployments: list[ActiveDeployment] = []

        for deployment in self.active_deployments:
            node = RemoteClusterNode(deployment.host)
            active_pids = node.query_active_gpu_pids(deployment.gpu_index)

            if not active_pids:
                print(
                    f"✅ Job on [{deployment.host}] GPU {deployment.gpu_index} completed."
                )
                continue

            # Detect foreign processes by ignoring our own tracked task PID
            foreign_pids = active_pids - {deployment.pid}

            if foreign_pids:
                print(
                    f"🚨 Preemption Alert! Foreign workload detected on [{deployment.host}] "
                    f"GPU {deployment.gpu_index}. Evicting low-priority task PID: {deployment.pid}"
                )
                node.terminate_remote_process(deployment.pid)

                # Re-queue step: Safe tracking ensures this exact job loops back to run later
                job_queue.append(deployment.job)
            else:
                surviving_deployments.append(deployment)

        self.active_deployments = surviving_deployments


def get_ssh_hosts() -> list[str]:
    """Retrieves standard host identification entries from local configuration profiles.

    Returns:
        List[str]: Clean array matching user ssh connection tags.
    """

    config_path = os.path.expanduser("~/.ssh/config")
    if not os.path.exists(config_path):
        return []

    hosts: list[str] = []
    with open(config_path, "r", encoding="utf-8") as config_file:
        for line in config_file:
            clean_line = line.strip()
            if clean_line.lower().startswith("host ") and "*" not in clean_line:
                parts = clean_line.split()
                if len(parts) >= 2:
                    hosts.append(parts[1])
    return hosts


if __name__ == "__main__":

    # Discover architecture nodes defined in local client files
    cluster_hosts = get_ssh_hosts()

    # Discover and construct executable parameters from concrete python files
    parser_agent = ModelStaticParser(models_directory=Path("src/score2dataset/models"))
    discovered_work_pool = parser_agent.extract_executable_jobs()

    if not cluster_hosts:
        print("❌ System configurations could not locate available host profiles.")
    elif not discovered_work_pool:
        print("❌ No matching concrete models derived from standard neural classes.")
    else:
        # Instantiate the final orchestrator pipeline loops
        scheduler = LowPriorityClusterScheduler(
            hosts=cluster_hosts, monitor_interval=15
        )
        scheduler.execute_orchestration(job_queue=discovered_work_pool)
