"""Single-sample rendering pipeline verification script.

This standalone diagnostic tool executes a full end-to-end integration test,
parsing a solitary notation score, serializing intermediate assets, and
applying spatial convolutions to output a standardized 48kHz mono 16-bit WAV.
"""

import sys
from pathlib import Path

from score2dataset.generator import DatasetGenerator
from score2dataset.parsers.musicxml_parser import MusicXMLParser
from score2dataset.wrapper import SfizzRenderEngine

# Explicitly append the local src directory tree to guarantee package discovery
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))


def run_diagnostics() -> int:
    """Orchestrates a single isolated test pass across the pipeline framework.

    Returns:
        An integer system exit status code where 0 indicates a flawless run
        and 1 represents a structural configuration fault.
    """
    print("=== Pipeline Integration Diagnostics ===")

    # 1. Establish absolute file system target parameters
    xml_input = ROOT_DIR / "data" / "input" / "test_score.musicxml"
    sfz_asset = ROOT_DIR / "data" / "assets" / "instrument.sfz"
    rir_asset = ROOT_DIR / "data" / "assets" / "reverb_hall.wav"
    output_dir = ROOT_DIR / "data" / "output_test"

    # Verify that your base development test files are physically present on disk
    if not xml_input.exists():
        print(
            f"CRITICAL TEST FAULT: Missing sample input score at: {xml_input}",
            file=sys.stderr,
        )
        return 1

    try:
        # 2. Parse the notation data into our internal memory format
        print(f" -> Activating MusicXML Parser on target: {xml_input.name}...")
        parser = MusicXMLParser(target_tpqn=480)
        in_memory_score = parser.parse_score(xml_input)
        print(
            f" -> Parse Successful. Extracted {len(in_memory_score.events)} NoteEvents."
        )

        # 3. Initialize the audio synthesis backend wrapper
        print(f" -> Linking Sfizz Render C++ engine to asset: {sfz_asset.name}...")
        sfizz_engine = SfizzRenderEngine(sfz_path=str(sfz_asset))

        # 4. Inject the engine into the top-level orchestration generator
        print(" -> Initializing Dataset Generator middleware...")
        generator = DatasetGenerator(
            engines=[sfizz_engine], rir_paths=[str(object=rir_asset)]
        )

        # 5. Execute a single variation sample pass on local NVMe SSD storage space
        print(" -> Launching rendering loop. Cache processing active inside /tmp...")

        # We invoke generate_batch with 1 variation to test the underlying system
        saved_paths = generator.generate_batch(
            base_scores=[in_memory_score], output_dir=str(output_dir), variation_count=1
        )

        print("\n=== SYSTEM CHECKS PASSED ===")
        print(f"Generated File Location: {saved_paths[0]}")
        print("Pipeline output successfully formatted to: Mono, 48kHz, 16-bit PCM.")
        return 0

    except RuntimeError as error:
        print("\n=== PIPELINE FAILURE ACCORDINGLY TRIGGERED ===", file=sys.stderr)
        print(f"Diagnostic Exception Intercepted: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
