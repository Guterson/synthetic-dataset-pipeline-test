"""Static source code analysis utilities for parsing package training workloads.

This module provides infrastructure tools to parse Python modules statically using
abstract syntax trees (AST). It inspects target package definitions from the
outside to extract valid onset detection subclasses, safely mapping files to
executable jobs without triggering active runtime side-effects.
"""

import ast
import sys
from pathlib import Path


class JobTask:
    """Encapsulates execution parameters for an orchestration workload."""

    def __init__(
        self, model_name: str, script_path: str, dataset_argument: str
    ) -> None:
        """Initializes structural job task parameters.

        Args:
            model_name: Name of the concrete model class.
            script_path: Absolute or relative disk path to the source file.
            dataset_argument: Evaluated target data argument string.
        """
        self.model_name: str = model_name
        self.script_path: str = script_path
        self.dataset_argument: str = dataset_argument


class ModelStaticParser:
    """Statically analyzes python modules to locate valid detector implementations."""

    def __init__(
        self, models_directory: Path = Path("src/score2dataset/models")
    ) -> None:
        """Initializes the static parser with the models package target path.

        Args:
            models_directory: Strongly typed Path pointing to the models folder.
        """
        self.models_directory: Path = models_directory

    def extract_executable_jobs(self) -> list[JobTask]:
        """Scans the directory and extracts concrete detector definitions via AST.

        Returns:
            list[JobTask]: A list of parsed jobs ready for the cluster scheduler.
        """
        jobs: list[JobTask] = []
        if not self.models_directory.is_dir():
            return jobs

        for entry in self.models_directory.iterdir():
            # Filter for Python files and ignore core configuration/base modules
            if (
                entry.is_file()
                and entry.suffix == ".py"
                and entry.name not in ("__init__.py", "onset_detector.py")
            ):
                model_class: str | None = self._find_concrete_subclass(entry)
                if model_class:
                    # Default placeholder data binding for the job definition
                    jobs.append(
                        JobTask(
                            model_name=model_class,
                            script_path=str(entry),
                            dataset_argument="default_data.csv",
                        )
                    )
        return jobs

    def _find_concrete_subclass(self, file_path: Path) -> str | None:
        """Parses an abstract syntax tree to find classes inheriting from OnsetDetector.

        Args:
            file_path: Strongly typed Path pointing to the target python script.

        Returns:
            str | None: The name of the valid subclass found, or None.
        """
        try:
            node = ast.parse(
                file_path.read_text(encoding="utf-8"), filename=str(file_path)
            )

            for item in node.body:
                if isinstance(item, ast.ClassDef):
                    base_names: list[str] = [
                        b.id for b in item.bases if isinstance(b, ast.Name)
                    ]
                    # Locate files explicitly implementing the OnsetDetector contract
                    if "OnsetDetector" in base_names:
                        return item.name
        except (OSError, SyntaxError) as err:
            print(
                f"⚠️ Static parse warning for {file_path.name}: {err}", file=sys.stderr
            )
            return None
        return None
