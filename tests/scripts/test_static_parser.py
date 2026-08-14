"""Automated verification suite testing the static AST source code parser.

Validates subclass extraction, syntax exception handling, blocklist exclusion filters,
and missing directory boundaries using temporary file configurations.
"""

from pathlib import Path

from score2dataset.scripts.static_parser import ModelStaticParser

# ---1. SYNTAX MATCHING AND PARSING MATRIX TESTS---


def test_parser_extracts_valid_subclasses_successfully(tmp_path: Path):
    """Verifies that python modules implementing the OnsetDetector contract are parsed."""
    # Create an isolated temporary test directory
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Write a mock compliant module implementing the specific subclass contract
    valid_file = models_dir / "valid_model.py"
    valid_file.write_text(
        "class ValidClassifier(OnsetDetector):\n"
        "    def initialize_components(self):\n"
        "        pass\n",
        encoding="utf-8",
    )

    parser = ModelStaticParser(models_directory=models_dir)
    jobs = parser.extract_executable_jobs()

    assert len(jobs) == 1
    assert jobs[0].model_name == "ValidClassifier"
    assert jobs[0].script_path == str(valid_file)
    assert jobs[0].dataset_argument == "default_data.csv"


def test_parser_ignores_non_inheriting_classes_and_syntax_errors(tmp_path: Path):
    """Verifies that non-matching architectures or corrupted files do not leak false jobs."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # File A: A normal class that does not implement the base contract
    invalid_file = models_dir / "standard_class.py"
    invalid_file.write_text("class StandardModel:\n    pass\n", encoding="utf-8")

    # File B: Broken syntax that triggers AST parse handling blocks
    corrupted_file = models_dir / "broken_syntax.py"
    corrupted_file.write_text("class BrokenModel(OnsetDetector:\n", encoding="utf-8")

    parser = ModelStaticParser(models_directory=models_dir)
    jobs = parser.extract_executable_jobs()

    # Both files must be safely bypassed
    assert len(jobs) == 0


# ---2. ARCHITECTURAL BOUNDARY AND BLACKLIST TESTS---


def test_parser_enforces_structural_blacklist_exclusions(tmp_path: Path):
    """Verifies that structural setup core scripts are explicitly ignored by scanning rules."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Write a mocked file using a reserved blocklist name
    base_file = models_dir / "onset_detector.py"
    base_file.write_text("class OnsetDetector(ABC):\n    pass\n", encoding="utf-8")

    init_file = models_dir / "__init__.py"
    init_file.write_text("\n", encoding="utf-8")

    parser = ModelStaticParser(models_directory=models_dir)
    jobs = parser.extract_executable_jobs()

    assert len(jobs) == 0


def test_parser_handles_missing_directory_gracefully(tmp_path: Path):
    """Verifies that an uncreated or missing directory returns an empty list without throwing exceptions."""
    non_existent_dir = tmp_path / "ghost_folder"

    parser = ModelStaticParser(models_directory=non_existent_dir)
    jobs = parser.extract_executable_jobs()

    assert isinstance(jobs, list)
    assert len(jobs) == 0
