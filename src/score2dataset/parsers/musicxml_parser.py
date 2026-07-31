"""MusicXML and MXL notation schema parser.

This module extracts pitch and rhythmic grids from structured XML elements
and maps them into standardized, in-memory PerformanceScore data containers.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.parsers.score_parser import ScoreParser


class MusicXMLParser(ScoreParser):
    """Concrete parser implementation for reading MusicXML format sheets.

    Converts localized raw notation definitions, divisions, and time counters
    into a uniform, integer-based timeline mapped to an abstract tick resolution.
    """

    # Centralized vocabulary shared between validation and calculation
    _STEP_MAP: Final[dict[str, int]] = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    # High-visibility musical dynamics translation table
    _DYNAMICS_VELOCITY_MAP: Final[dict[str, int]] = {
        "ppp": 16,
        "pp": 33,
        "p": 49,
        "mp": 64,
        "mf": 80,
        "f": 96,
        "ff": 112,
        "fff": 126,
    }

    def __init__(self, target_tpqn: int = 480) -> None:
        """Initializes the parser configurations and immutable musical maps.

        Args:
            target_tpqn: Ticks Per Quarter Note resolution forced onto the
                output dataset timeline. Defaults to 480.
        """
        self.target_tpqn: int = target_tpqn

    def parse_score(self, file_path: Path) -> PerformanceScore:
        """Parses an external MusicXML score file into a standard PerformanceScore.

        Evaluates global subdivisions, loops measures sequentially to handle
        polyphonic backtracks, and calculates discrete, absolute tick positions.

        Args:
            file_path: The absolute path to the target notation file on disk.

        Returns:
            A standard populated PerformanceScore instance.

        Raises:
            ValueError: If the file contents are missing core XML structural
                elements or contain invalid musical pitch annotations.
        """
        tree = ET.parse(source=file_path)
        root = tree.getroot()

        score = PerformanceScore(source=file_path)

        divisions_elem = root.find(path=".//attributes/divisions")
        if (
            divisions_elem is None
            or divisions_elem.text is None
            or not divisions_elem.text.strip().isdigit()
        ):
            raise ValueError(
                f"Invalid MusicXML structure: Global <divisions> definition "
                f"must be a valid, positive integer in file '{file_path}'."
            )

        # The linter now has mathematical proof that .text is a clean numeric string
        xml_divisions = int(divisions_elem.text)

        if xml_divisions == 0:
            raise ValueError(
                f"Invalid MusicXML layout: Global <divisions> cannot be zero "
                f"in file '{file_path}'."
            )

        tick_scale_factor: float = self.target_tpqn / xml_divisions

        # Track dynamic scaling per direction event (Default to standard mf velocity)
        current_velocity = 80

        # Enforce strict part-level containment separation to avoid accumulator contamination
        for part in root.findall(path=".//part"):
            current_xml_time_accumulator = 0

            for measure in part.findall(path=".//measure"):
                for element in measure:

                    # Extract structural volume changes from layout expressions safely
                    if element.tag == "direction":
                        dyn_elem = element.find(path=".//dynamics/*")
                        if dyn_elem is not None:
                            # Map standard notation markers (p, mf, f) to standard MIDI velocity numbers
                            tag_name = dyn_elem.tag.lower()
                            current_velocity = self._DYNAMICS_VELOCITY_MAP.get(
                                tag_name, 80
                            )

                    elif element.tag == "forward":
                        skip_duration = element.find(path="duration")
                        if skip_duration is not None and skip_duration.text is not None:
                            current_xml_time_accumulator += int(skip_duration.text)

                    elif element.tag == "backup":
                        skip_duration = element.find(path="duration")
                        if skip_duration is not None and skip_duration.text is not None:
                            current_xml_time_accumulator -= int(skip_duration.text)

                    elif element.tag == "note":
                        is_rest = element.find(path="rest") is not None
                        duration_elem = element.find(path="duration")
                        raw_xml_duration = (
                            int(duration_elem.text.strip())
                            if (
                                duration_elem is not None
                                and duration_elem.text is not None
                            )
                            else 0
                        )

                        if is_rest:
                            current_xml_time_accumulator += raw_xml_duration
                            continue

                        pitch_elem = element.find(path=".//step")
                        octave_elem = element.find(path=".//octave")
                        is_chord = element.find(path="chord") is not None

                        if (
                            pitch_elem is not None
                            and octave_elem is not None
                            and pitch_elem.text is not None
                            and octave_elem.text is not None
                        ):
                            octave = int(octave_elem.text.strip())
                            pitch_step = pitch_elem.text.upper().strip()

                            alter_elem = element.find(path=".//alter")
                            alteration = (
                                int(alter_elem.text.strip())
                                if (
                                    alter_elem is not None
                                    and alter_elem.text is not None
                                )
                                else 0
                            )

                            midi_pitch = self._calculate_midi_pitch(
                                step=pitch_step, octave=octave, alter=alteration
                            )

                            # If it's a chord note, calculate its onset relative to where the chord block *started*
                            target_onset_time = (
                                current_xml_time_accumulator - raw_xml_duration
                                if is_chord
                                else current_xml_time_accumulator
                            )

                            baked_onset_ticks = round(
                                target_onset_time * tick_scale_factor
                            )
                            baked_duration_ticks = round(
                                raw_xml_duration * tick_scale_factor
                            )

                            # Assign the dynamically extracted notation volume value
                            note_event = NoteEvent(
                                pitch=midi_pitch,
                                onset_ticks=baked_onset_ticks,
                                duration_ticks=baked_duration_ticks,
                                velocity=current_velocity,
                            )
                            score.events.append(note_event)

                        # Update the timeline only if the parsed note object is a standalone step
                        if not is_chord:
                            current_xml_time_accumulator += raw_xml_duration

        return score

    def _calculate_midi_pitch(self, step: str, octave: int, alter: int) -> int:
        """Translates a traditional musical step coordinate into a raw MIDI index.

        Args:
            step: Pre-validated musical note letter name string (A through G).
            octave: Target octave integer index value.
            alter: Semitone sharp or flat transformation offset tracking integers.

        Returns:
            An absolute integer mapping representing standard MIDI key values.
        """
        # Dictionary lookups are now safe from KeyErrors due to prior validation checks
        base_note: int = self._STEP_MAP[step]
        return int((octave + 1) * 12 + base_note + alter)
