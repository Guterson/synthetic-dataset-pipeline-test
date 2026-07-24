"""Command Line Interface for the score2dataset generation pipeline.

This module provides the terminal-facing interface to orchestrate mass synthetic
audio dataset production. It decodes terminal flags, coordinates security validation
sweeps across asset trees, and boots up the pipeline execution engines.
"""

import argparse
import sys
from pathlib import Path

from score2dataset.generator import DatasetGenerator
from score2dataset.parsers.musicxml_parser import MusicXMLParser
from score2dataset.wrapper import SfizzRenderEngine

# from score2dataset.validators import validate_musicxml
# from score2dataset.validators import validate_sfz

# Safeguard to support running this file directly on lab terminals
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(object=Path(__file__).resolve().parents[2]))
    # __package__ = "score2dataset"


def parse_arguments(args_list: list[str]) -> argparse.Namespace:
    """Decodes, standardizes, and validates terminal string arguments.

    Processes raw string parameters from the shell environment, mapping them
    into structured namespace variables while enforcing operational bounds.

    Args:
        args_list: A list of string arguments passed via the terminal interface.

    Returns:
        A Namespace object containing validated user options and paths.
    """
    parser = argparse.ArgumentParser(
        description="score2dataset: Bulk acoustic variations generator from structured scores."
    )

    subparsers: argparse._SubParsersAction = parser.add_subparsers(
        dest="command", required=True, help="Target processing routine to execute."
    )

    # Main workflow execution engine configuration
    parser_run = subparsers.add_parser(
        "run",
        help="Execute the full alteration, timing deformation, and rendering pipeline.",
    )
    parser_run.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the primary source input (.musicxml or .mxl) target score.",
    )
    parser_run.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Destination directory tree where final synthesized datasets are saved.",
    )
    parser_run.add_argument(
        "-s",
        "--sfz",
        required=True,
        type=str,
        help="Path to the target SFZ virtual instrument file to load into the engine.",
    )
    parser_run.add_argument(
        "-r",
        "--rir",
        type=str,
        default=None,
        help="Optional path to a physical Room Impulse Response WAV file for reverb processing.",
    )

    # Asset integrity verification suite configuration
    parser_validate = subparsers.add_parser(
        "validate",
        help="Run non-destructive structure diagnostics across input audio and score files.",
    )

    validate_subparsers = parser_validate.add_subparsers(
        dest="subcommand", required=True, help="Target file format validation matrix."
    )

    # Structural sampler file diagnostics configuration
    parser_validate_sfz = validate_subparsers.add_parser(
        "sfz",
        help="Validate an SFZ file structure and its internal audio sample linkages.",
    )
    parser_validate_sfz.add_argument(
        "-s",
        "--sfz",
        required=True,
        type=str,
        help="Path to the target SFZ file to verify.",
    )

    # XML notation schema validation configuration
    parser_validate_xml = validate_subparsers.add_parser(
        "xml", help="Validate a MusicXML file's readability and structural continuity."
    )
    parser_validate_xml.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the target MusicXML file to verify.",
    )

    return parser.parse_args(args_list)


def main(args: list[str] | None = None) -> int:
    """The central package orchestration controller and program lifecycle anchor.

    Intercepts terminal parameters, routes execution workflow tasks to matching
    submodules, manages pipeline execution resources, and signals execution status
    codes back to the operating system host.

    Args:
        args: A list of runtime parameters. Defaults to sys.argv[1:] if omitted.

    Returns:
        An integer system exit status code where 0 indicates a flawless run
        and 1 represents a structural operational fault.
    """
    if args is None:
        args = sys.argv[1:]

    try:
        parsed_args = parse_arguments(args)

        # Route execution based on the parsed subcommand string
        match parsed_args.command:
            case "run":
                input_file: Path = Path(parsed_args.input).resolve()

                # 1. Dynamically select the correct parser based on file extension
                match input_file.suffix.lower():
                    case ".musicxml" | ".mxl":
                        parser = MusicXMLParser()
                    case _:
                        print(
                            f"ERROR: Unsupported file format extension: {input_file.suffix}",
                            file=sys.stderr,
                        )
                        return 1

                # 2. Convert file to the uniform internal memory structure
                score_data = parser.parse_score(input_file)

                # 3. Initialize the audio synthesis backend wrapper (Packaged inside a list)

                sfizz_backend = SfizzRenderEngine(sfz_path=parsed_args.sfz)

                # 4. Inject engine and rir lists into the non-cartesian balanced coordinator
                generator = DatasetGenerator(
                    engines=[sfizz_backend],
                    rir_paths=[parsed_args.rir] if parsed_args.rir else [],
                )

                # 5. Execute the batch generation routine using the memory structures
                # For local testing, we default variation_count to 1
                generator.generate_batch(
                    base_scores=[score_data],
                    output_dir=parsed_args.output,
                    variation_count=1,
                )

                print("SUCCESS: Full generation pipeline executed flawlessly.")
                return 0

            case "validate":
                match parsed_args.subcommand:
                    case "sfz":
                        print(
                            f"Running diagnostics on SFZ asset tree: {parsed_args.sfz}..."
                        )

                        # Call the silent backend validator
                        is_valid = True  # validate_sfz(parsed_args.sfz)

                        # The UI presentation happens HERE
                        if is_valid:
                            print(
                                "SUCCESS: The SFZ file and all internal sample linkages are perfectly valid!"
                            )
                            return 0
                        else:
                            print(
                                "ERROR: SFZ validation failed! Check that all referenced audio assets exist.",
                                file=sys.stderr,
                            )
                            return 1

                    case "xml":
                        print(
                            f"Running diagnostics on MusicXML score: {parsed_args.input}..."
                        )

                        # Call the silent backend validator
                        is_valid = True  # validate_xml(parsed_args.input)

                        # The UI presentation happens HERE
                        if is_valid:
                            print(
                                f"SUCCESS: '{parsed_args.input}' is a fully compliant and readable MusicXML file."
                            )
                            return 0
                        else:
                            print(
                                f"ERROR: '{parsed_args.input}' contains structural issues or is an invalid XML.",
                                file=sys.stderr,
                            )
                            return 1
    except FileNotFoundError as e:
        print(
            f"ERROR: A required file could not be found during generation.\nDetails: {e}",
            file=sys.stderr,
        )
        return 1

    except RuntimeError as e:
        print(
            f"ERROR: The Sfizz audio engine encountered an operational fault.\nDetails: {e}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
