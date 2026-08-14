"""Distributed multi-GPU concurrent model evaluation coordinator.

This script parses the local models directory, discovers active detectors,
allocates available cluster hardware nodes dynamically, launches worker tasks
concurrently, and aggregates the normalized outcomes into a comparison report.
"""

import sys
import time
from pathlib import Path

import numpy as np

from score2dataset.scripts.evaluate_worker import calculate_unified_mir_score
from score2dataset.scripts.scheduler import (
    ActiveDeployment,
    LowPriorityClusterScheduler,
    RemoteClusterNode,
    get_ssh_hosts,
)
from score2dataset.scripts.static_parser import JobTask, ModelStaticParser


def main() -> None:
    """Orchestrates cluster resources to execute model evaluations concurrently."""
    validation_data_path = Path("data/processed/validation_isolated_songs")

    print("👑 Initializing remote cluster concurrent evaluation coordinator...")

    # 1. Discover architecture nodes and model configurations dynamically
    cluster_hosts = get_ssh_hosts()
    parser_agent = ModelStaticParser(models_directory=Path("src/score2dataset/models"))
    discovered_jobs = parser_agent.extract_executable_jobs()

    if not cluster_hosts:
        print(
            "❌ Coordinator Error: Configuration profiles could not locate active host credentials."
        )
        sys.exit(1)

    if not discovered_jobs:
        print(
            "❌ Coordinator Error: No concrete models derived from standard neural classes."
        )
        sys.exit(1)

    # Instantiate the monitoring scheduler to track GPU availability
    scheduler = LowPriorityClusterScheduler(hosts=cluster_hosts)
    active_workers: list[ActiveDeployment] = []

    # 2. Concurrently deploy evaluation script workers across cluster slots
    for job in discovered_jobs:
        free_slots = scheduler.get_available_resources()
        while not free_slots:
            print(
                "⏳ All network compute nodes saturated. Waiting for available hardware context..."
            )
            time.sleep(15)
            free_slots = scheduler.get_available_resources()

        # Select the first available node and GPU channel pair
        node, gpu_idx = free_slots[0]

        # Build command pointing cleanly back to the worker execution script
        cmd = (
            f"python3 score2dataset/scripts/evaluate_worker.py "
            f"--model {job.model_name} --gpu {gpu_idx}"
        )

        pid = node.launch_background_job(cmd)
        if pid:
            eval_job = JobTask(
                model_name=job.model_name, script_path="", dataset_argument=""
            )

            deployment = ActiveDeployment(
                host=node.host_name, pid=pid, gpu_index=gpu_idx, job=eval_job
            )
            active_workers.append(deployment)
            scheduler.active_deployments.append(deployment)

        else:
            print(
                f"⚠️ Critical: Failed to launch detached worker thread context for {job.model_name} on {node.host_name}"
            )

    # 3. Synchronize distributed execution blocks across grid clusters
    print(
        "⏳ Synchronizing distributed execution blocks. Monitoring remote process threads..."
    )
    while active_workers:
        still_running = []
        for worker in active_workers:
            # Instantiate the verified client class using the tracked host name string
            node_client = RemoteClusterNode(worker.host)
            active_pids = node_client.query_active_gpu_pids(worker.gpu_index)

            # Mypy/Ruff are completely satisfied because worker.pid is strictly an int
            if worker.pid in active_pids:
                still_running.append(worker)

        active_workers = still_running
        if active_workers:
            time.sleep(10)

    # 4. Load ground truth tracking matrix arrays (Song-level validation split matching)
    gt_path = validation_data_path / "ground_truth_events.npy"
    if not gt_path.exists():
        print(
            f"❌ Aggregator error: Missing centralized validation ground-truth matrix array at {gt_path}"
        )
        sys.exit(1)

    ground_truth = [tuple(x) for x in np.load(gt_path, allow_pickle=True)]

    print("\n📊 ========================================================")
    print("📈       CONCURRENT PIPELINE EVALUATION METRICS REPORT")
    print("============================================================")

    # 5. Aggregate outcomes, calculate asymmetric ratios, and build comparison report
    for job in discovered_jobs:
        cache_file = Path(
            f"data/outputs/evaluation_cache/{job.model_name.lower()}_events.npy"
        )

        if cache_file.exists():
            # Load timeline predictions parsed out by the worker process
            predictions = [tuple(x) for x in np.load(cache_file, allow_pickle=True)]

            # Run the asymmetric scoring formula (Tolerance=25ms, Tau=10ms, FP_Weight=2.0)
            metrics = calculate_unified_mir_score(predictions, ground_truth)

            print(f"\n🔹 Model Class Summary: {job.model_name}")
            print(f"   ├── Asymmetric Precision : {metrics['precision']:.4f}")
            print(f"   ├── Timeline Event Recall: {metrics['recall']:.4f}")
            print(f"   ├── Structural F1-Score  : {metrics['f1_score']:.4f}")
            print(f"   ├── Mean Absolute Drift  : {metrics['mae_ms']:.2f} ms")
            print(f"   └── UNIFIED ACCURACY SCORE: {metrics['unified_score']:.4f}")

            # Clean up temp cache storage artifacts from the disk
            cache_file.unlink()
        else:
            print(
                f"\n⚠️ Warning: No cached normalization dump discovered for class {job.model_name}."
            )

    print("\n🎉 Distributed evaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()
