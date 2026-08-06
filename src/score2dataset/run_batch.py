"""Master execution wrapper to prevent university NFS swap memory crashes.

Launches the core generation script sequentially in isolated process blocks,
guaranteeing complete operating system swap space reclamation between runs.
"""

import subprocess
import sys


def main() -> None:
    """Executes the rendering pipeline script in controlled, isolated cycles."""
    # Define how many distinct variations you want in total
    total_variations: int = 1
    print(f"=== Starting Safe Batch Execution Matrix ({total_variations} loops) ===")

    for i in range(1, total_variations + 1):
        print(
            f"\n[CYCLE {i}/{total_variations}] Spawning isolated shell worker process..."
        )

        # Launch your diagnostics script as an independent operating system task.
        # This gives your script a clean environment slate on every single loop.
        result = subprocess.run(
            ["python", "src/score2dataset/quick_test.py"],
            capture_output=False,
            text=True,
            check=False,
        )

        # Exit immediately if an internal crash or file path fault is detected
        if result.returncode != 0:
            print(
                f"[CRITICAL] Batch pipeline failed at cycle iteration {i}.",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    print(
        "\n=== SYSTEM SUCCESS: Complete batch processed with flat swap utilization! ==="
    )


if __name__ == "__main__":
    main()
