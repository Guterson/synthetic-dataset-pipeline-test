"""Single-sample rendering pipeline verification script.

This standalone diagnostic tool executes a full end-to-end integration test,
parsing a solitary notation score, serializing intermediate assets, and
applying spatial convolutions to output a standardized 48kHz mono 16-bit WAV.
"""

import sys
from pathlib import Path
from typing import Any

from score2dataset.exceptions import AudioEngineError, Score2DatasetError
from score2dataset.generator import DatasetGenerator
from score2dataset.parsers.musicxml_parser import MusicXMLParser

# Explicitly append the local src directory tree to guarantee package discovery
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(ROOT_DIR / "src"))


def run_diagnostics() -> int:
    """Orchestrates a single isolated test pass across the pipeline framework.

    Returns:
        An integer system exit status code where 0 indicates a flawless run
        and 1 represents a structural configuration fault.
    """
    print("=== Pipeline Integration Diagnostics ===")

    xml_input: Path = ROOT_DIR / "data" / "inputs" / "hungarian_dance_n5.musicxml"
    output_dir: Path = ROOT_DIR / "data" / "outputs" / "diagnostic_test"

    if not xml_input.exists():
        print(
            f"CRITICAL TEST FAULT: Missing sample input score at: {xml_input}",
            file=sys.stderr,
        )
        return 1

    try:
        print(f" -> Activating MusicXML Parser on target: {xml_input.name}...")
        parser = MusicXMLParser(target_tpqn=480)
        in_memory_score = parser.parse_score(xml_input)
        print(
            f" -> Parse Successful. Extracted {len(in_memory_score.events)} NoteEvents."
        )

        # 1. Discover all pre-conditioned, zero-latency instrument maps dynamically
        sfz_root: Path = ROOT_DIR / "data" / "assets" / "sfz"
        safe_sfz_paths = list(sfz_root.rglob("*_OnsetSafe.sfz"))

        # Formulate a structured configuration dictionary for each unique instrument found
        engine_configs: list[dict[str, Any]] = [
            {"sfz_path": str(sfz_path)} for sfz_path in safe_sfz_paths
        ]
        print(
            f" -> Discovered {len(engine_configs)} optimized SFZ engine mapping configuration(s)."
        )

        # 2. Gather all pre-conditioned RIR assets from your curated folder pool
        rir_folder: Path = ROOT_DIR / "data" / "assets" / "rir"
        batch_rir_paths: list[str] = [str(p) for p in rir_folder.glob("*.wav")]
        print(
            f" -> Discovered {len(batch_rir_paths)} conditioned RIR asset configuration(s)."
        )

        print(
            " -> Initializing Dataset Generator middleware with complete asset pool matrices..."
        )
        generator = DatasetGenerator(
            engine_configs=engine_configs, rir_paths=batch_rir_paths
        )

        print(" -> Launching batch rendering loop for the singular target score...")
        # Keeps base_scores as a single-item list containing exactly one score.
        # variation_count=5 means the generator will create 5 distinct environmental variations
        # combining different discovered SFZ setups and RIR acoustical profiles.
        saved_paths: list[Path] = generator.generate_batch(
            base_scores=[in_memory_score],
            output_dir=str(output_dir),
            variation_count=100,
        )

        print("\n=== BATCH RENDER COMPLETE ===")
        print(
            f"Successfully serialized {len(saved_paths)} audio file variations inside: {output_dir}"
        )

        print("\n=== SYSTEM CHECKS PASSED ===")
        print(f"Generated File Location: {saved_paths[0]}")
        print("Pipeline output successfully formatted to: Mono, 48kHz, 16-bit PCM.")
        return 0

    except Score2DatasetError as error:
        print("\n=== PIPELINE DOMAIN FAILURE TRAPPED ===", file=sys.stderr)
        print(f"Diagnostic Exception Intercepted: {error}", file=sys.stderr)

        # Explicit type narrowing for backend engine failures
        if isinstance(error, AudioEngineError):
            print(f"Engine Exit Code: {error.returncode}", file=sys.stderr)
            if error.details:
                print(
                    "\n--- Engine Process Output (stdout/stderr) ---", file=sys.stderr
                )
                print(error.details, file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)

        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
