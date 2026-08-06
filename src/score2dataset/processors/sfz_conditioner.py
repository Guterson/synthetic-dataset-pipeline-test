"""Digital Signal Processing utilities for SFZ instrument library conditioning.

Provides structural text analysis and parsing utilities to identify, log, and
correct onset latency hazards and missing sample assets within SFZ definition maps.
"""

import re
from pathlib import Path
from typing import Final


class SfzConditioner:
    """Evaluates, details, and modifies SFZ files to guarantee instant note onsets."""

    # re.IGNORECASE forces the regex engine to match any case combination natively
    ATTACK_PATTERN: Final[re.Pattern] = re.compile(
        pattern=r"(ampeg_attack\s*=\s*)([0-9.]+)", flags=re.IGNORECASE
    )
    SAMPLE_PATTERN: Final[re.Pattern] = re.compile(
        pattern=r"sample\s*=\s*([^\s\r\n]+)", flags=re.IGNORECASE
    )

    def __init__(self, sfz_path: Path) -> None:
        """Initializes the SFZ conditioner tracking structures.

        Args:
            sfz_path: Location pointing to the specific target .sfz file.
        """
        self.sfz_path: Path = sfz_path
        self.sfz_dir: Path = sfz_path.parent

    def process_and_fix(self, output_path: Path) -> bool:
        """Detailed diagnostic scan that writes a corrected, zero-latency SFZ file.

        Args:
            output_path: Target destination to store the modified safe SFZ file.

        Returns:
            A boolean indicating True if the asset is completely safe and written,
            or False if structural dependency failures prevent execution.
        ```"""
        if not self.sfz_path.exists():
            print(f"[ERROR] Source file does not exist: {self.sfz_path}")
            return False

        with open(self.sfz_path, "r", encoding="utf-8", errors="ignore") as file:
            lines: list[str] = file.readlines()

        missing_samples: list[tuple[int, str]] = []
        modified_lines_count: int = 0
        corrected_content: list[str] = []

        # print(f"\n=== Running Detailed Diagnostics for: {self.sfz_path.name} ===")

        for idx, line in enumerate(lines, start=1):
            current_line_text = line

            sample_match = self.SAMPLE_PATTERN.search(line)
            if sample_match:
                sample_name = sample_match.group(1).replace("\\", "/")
                sample_path = self.sfz_dir / sample_name
                if not sample_path.exists():
                    missing_samples.append((idx, sample_name))

            attack_match = self.ATTACK_PATTERN.search(line)
            if attack_match:
                attack_val = float(attack_match.group(2))
                if attack_val > 0.001:
                    # print(
                    #    f" -> [FIXED LINE {idx}]: Found ampeg_attack={attack_val}. Forcing to 0.0."
                    # )
                    current_line_text = self.ATTACK_PATTERN.sub(r"\g<1>0.0", line)
                    modified_lines_count += 1

            corrected_content.append(current_line_text)

        if missing_samples:
            # print(
            #    f"\n[CRITICAL WARNING] Found {len(missing_samples)} missing WAV audio file(s):"
            # )
            for line_no, file_name in missing_samples[:10]:
                print(f"   * Line {line_no}: Missing file -> {file_name}")
            if len(missing_samples) > 10:
                print(f"   * ... and {len(missing_samples) - 10} more missing files.")
            return False

        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.writelines(corrected_content)

        print("\n[SUMMARY] Curation completed successfully.")
        print(
            f" -> Forced {modified_lines_count} delayed attack envelopes to instant 0.0 transients."
        )
        print(
            f" -> Output safe, zero-latency instrument mapping saved to: {output_path.name}\n"
        )
        return True
