"""Conftest.py for pytest: Intercepts directory creation calls to provide detailed lineage diagnostics."""

import os
import pathlib
import traceback

import pytest

# Store the original, un-mutated core system functions
_original_mkdir = os.mkdir
_original_makedirs = os.makedirs
_original_path_mkdir = pathlib.Path.mkdir


def _diagnostic_tracker(func_name, path_arg, *args, **kwargs):
    """Intercepts directory creation calls and prints the exact code trigger lineage."""
    # Convert path to string for evaluation
    path_str = str(path_arg)

    # Ignore standard internal pytest/python system cache folders
    if any(x in path_str for x in [".pytest_cache", "__pycache__", ".ruff_cache"]):
        return

    print(
        f"\n🚨 [DIRECTORY DETECTED] Function '{func_name}' called for path: {path_str}"
    )
    print("---------------- Stack Trace Origin ----------------")
    # Print the execution lineage directly to your terminal screen
    traceback.print_stack()
    print("----------------------------------------------------\n")


# Create our wrapper overrides
def wrapped_mkdir(path, *args, **kwargs):
    _diagnostic_tracker("os.mkdir", path, *args, **kwargs)
    return _original_mkdir(path, *args, **kwargs)


def wrapped_makedirs(name, *args, **kwargs):
    _diagnostic_tracker("os.makedirs", name, *args, **kwargs)
    return _original_makedirs(name, *args, **kwargs)


def wrapped_path_mkdir(self, *args, **kwargs):
    _diagnostic_tracker("Path.mkdir", self, *args, **kwargs)
    return _original_path_mkdir(self, *args, **kwargs)


@pytest.fixture(scope="session", autouse=True)
def intercept_directory_creation():
    """Session-wide fixture that forces runtime interception before any test executes."""
    # Hot-patch the active Python session properties
    os.mkdir = wrapped_mkdir
    os.makedirs = wrapped_makedirs
    pathlib.Path.mkdir = wrapped_path_mkdir

    yield

    # Restore originals cleanly when the test suite concludes
    os.mkdir = _original_mkdir
    os.makedirs = _original_makedirs
    pathlib.Path.mkdir = _original_path_mkdir
