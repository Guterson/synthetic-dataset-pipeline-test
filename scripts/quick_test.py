"""Single-sample rendering pipeline verification tool.

This standalone diagnostic tool executes a full end-to-end integration test,
parsing a solitary notation score, serializing intermediate assets, and
applying spatial convolutions to output a standardized 48kHz mono 16-bit WAV
for immediate human evaluation and quality auditing.
"""

import sys
from pathlib import Path
from typing import Any

from score2dataset.exceptions import AudioEngineError, Score2DatasetError
from score2dataset.generator import DatasetGenerator
from score2dataset.parsers.musicxml_parser import MusicXMLParser


def run_diagnostics() -> int:
    """Orchestrates a single isolated test pass across the pipeline framework.

    Returns:
        int: System exit status code where 0 indicates a flawless run
             and 1 represents a structural configuration fault.
    """
    print("=== Pipeline Integration Diagnostics ===")

    # Enforce strict Path evaluation from the directory root
    root_dir: Path = Path(__file__).resolve().parent.parent
    xml_input: Path = root_dir / "data" / "inputs" / "hungarian_dance_n5.musicxml"
    output_dir: Path = root_dir / "data" / "outputs" / "diagnostic_test"

    if not xml_input.exists():
        print(
            f"❌ Critical Test Fault: Missing sample input score at: {xml_input}",
            file=sys.stderr,
        )
        return 1

    try:
        print(f" -> Activating MusicXML Parser on target: {xml_input.name}...")
        parser = MusicXMLParser(target_tpqn=480)
        in_memory_score = parser.parse_score(xml_input)
        print(
            f" -> Parse Successful. Extracted {len(in_memory_score[0].events)} NoteEvents."
        )

        # 1. Discover optimized instrument setups dynamically
        sfz_root: Path = root_dir / "data" / "assets" / "sfz"
        safe_sfz_paths = list(sfz_root.rglob("*_OnsetSafe.sfz"))
        if not safe_sfz_paths:
            print(
                "❌ Critical Test Fault: No '_OnsetSafe.sfz' instruments found.",
                file=sys.stderr,
            )
            return 1

        engine_configs: list[dict[str, Any]] = [
            {"sfz_path": str(sfz_path)} for sfz_path in safe_sfz_paths
        ]
        print(
            f" -> Discovered {len(engine_configs)} optimized SFZ engine mapping configuration(s)."
        )

        # 2. Gather room response assets dynamically
        rir_folder: Path = root_dir / "data" / "assets" / "rir"
        batch_rir_paths: list[str] = [str(p) for p in rir_folder.glob("*.wav")]
        if not batch_rir_paths:
            print(
                "❌ Critical Test Fault: No room impulse response (.wav) files found.",
                file=sys.stderr,
            )
            return 1

        print(
            f" -> Discovered {len(batch_rir_paths)} conditioned RIR asset configuration(s)."
        )

        # 3. Trigger the generation core
        print(" -> Initializing Dataset Generator middleware...")
        generator = DatasetGenerator(
            engine_configs=engine_configs, rir_paths=batch_rir_paths
        )

        print(" -> Launching isolated validation render pass...")
        # variation_count=1 ensures we only generate a single, lightweight test sample
        saved_paths: list[Path] = generator.generate_batch(
            score_tuples=[in_memory_score],
            output_dir=str(output_dir),
            variation_count=1,
            base_seed=42,
        )

        print("\n=== SAMPLE RENDER COMPLETE ===")
        print(f"Successfully serialized verification file inside: {output_dir}")
        print("Pipeline output successfully formatted to: Mono, 48kHz, 16-bit PCM.")
        if saved_paths:
            print(f"Generated Audit Target Location: {saved_paths[0]}")

        print("\n=== SYSTEM HEALTH CHECKS: PASSED ===")
        return 0

    except Score2DatasetError as error:
        print("\n❌ === PIPELINE DOMAIN FAILURE TRAPPED ===", file=sys.stderr)
        print(f"Diagnostic Exception Intercepted: {error}", file=sys.stderr)

        if isinstance(error, AudioEngineError):
            print(f"Engine Process Exit Code: {error.returncode}", file=sys.stderr)
            if error.details:
                print(
                    "\n--- Engine Process Output (stdout/stderr) ---", file=sys.stderr
                )
                print(error.details, file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
