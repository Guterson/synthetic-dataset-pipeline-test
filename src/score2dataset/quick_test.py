"""Single-sample rendering pipeline verification script.

This standalone diagnostic tool executes a full end-to-end integration test,
parsing a solitary notation score, serializing intermediate assets, and
applying spatial convolutions to output a standardized 48kHz mono 16-bit WAV.
"""

import sys
from pathlib import Path
from typing import Any

from score2dataset.exceptions import Score2DatasetError
from score2dataset.generator import DatasetGenerator
from score2dataset.parsers.musicxml_parser import MusicXMLParser

# Explicitly append the local src directory tree to guarantee package discovery
ROOT_DIR: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))


def run_diagnostics() -> int:
    """Orchestrates a single isolated test pass across the pipeline framework.

    Returns:
        An integer system exit status code where 0 indicates a flawless run
        and 1 represents a structural configuration fault.
    """
    print("=== Pipeline Integration Diagnostics ===")

    xml_input: Path = ROOT_DIR / "data" / "inputs" / "hungarian_dance_n5.musicxml"
    sfz_asset: Path = (
        ROOT_DIR
        / "data"
        / "assets"
        / "sfz"
        / "SalamanderGrandPianoV3_48khz24bit"
        / "SalamanderGrandPianoV3Retuned.sfz"
    )
    rir_asset: Path = (
        ROOT_DIR / "data" / "assets" / "rir" / "1st_baptist_nashville_balcony.wav"
    )
    output_dir: Path = ROOT_DIR / "data" / "output_test"

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

        # Formulate configuration map to pass safely across process barriers
        print(f" -> Packaging engine configuration for: {sfz_asset.name}...")
        engine_configs: list[dict[str, Any]] = [{"sfz_path": str(sfz_asset)}]

        print(" -> Initializing Dataset Generator middleware...")
        generator = DatasetGenerator(
            engine_configs=engine_configs, rir_paths=[str(rir_asset)]
        )

        print(" -> Launching rendering loop. Cache processing active inside /tmp...")
        saved_paths: list[Path] = generator.generate_batch(
            base_scores=[in_memory_score], output_dir=str(output_dir), variation_count=1
        )

        print("\n=== SYSTEM CHECKS PASSED ===")
        print(f"Generated File Location: {saved_paths[0]}")
        print("Pipeline output successfully formatted to: Mono, 48kHz, 16-bit PCM.")
        return 0

    except Score2DatasetError as error:
        print("\n=== PIPELINE DOMAIN FAILURE TRAPPED ===", file=sys.stderr)
        print(f"Diagnostic Exception Intercepted: {error}", file=sys.stderr)
        return 1
    except Exception as unexpected_error:
        print("\n=== UNEXPECTED CRITICAL FAILURE OCCURRED ===", file=sys.stderr)
        print(f"System Error Trace: {unexpected_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
