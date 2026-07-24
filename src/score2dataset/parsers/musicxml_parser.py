"""MusicXML and MXL notation schema parser.

This module extracts pitch and rhythmic grids from structured XML elements
and maps them into standardized, in-memory PerformanceScore data containers.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.parsers.score_parser import ScoreParser


class MusicXMLParser(ScoreParser):
    """Concrete parser implementation for reading MusicXML format sheets.

    Converts localized raw notation definitions, divisions, and time counters
    into a uniform, integer-based timeline mapped to an abstract tick resolution.
    """

    def __init__(self, target_tpqn: int = 480) -> None:
        """Initializes the parser configurations and immutable musical maps.

        Args:
            target_tpqn: Ticks Per Quarter Note resolution forced onto the
                output dataset timeline. Defaults to 480.
        """
        self.target_tpqn: int = target_tpqn

        # Centralized pitch vocabulary shared between validation and calculation
        self._step_map: dict[str, int] = {
            "C": 0,
            "D": 2,
            "E": 4,
            "F": 5,
            "G": 7,
            "A": 9,
            "B": 11,
        }

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
        current_xml_time_accumulator = 0

        for element in root.findall(path=".//measure/*"):

            if element.tag == "forward":
                skip_duration = element.find(path="duration")
                if skip_duration is not None:
                    current_xml_time_accumulator += int(str(skip_duration.text))

            elif element.tag == "backup":
                skip_duration = element.find(path="duration")
                if skip_duration is not None:
                    current_xml_time_accumulator -= int(str(skip_duration.text))

            elif element.tag == "note":
                is_rest = element.find(path="rest") is not None
                if is_rest:
                    rest_duration_elem = element.find(path="duration")
                    if rest_duration_elem is not None:
                        current_xml_time_accumulator += int(
                            str(rest_duration_elem.text)
                        )
                    continue

                pitch_elem = element.find(".//step")
                octave_elem = element.find(".//octave")
                duration_elem = element.find("duration")

                if (
                    pitch_elem is not None
                    and octave_elem is not None
                    and duration_elem is not None
                ):
                    # Guard against empty fields returning NoneType attributes
                    if (
                        pitch_elem.text is None
                        or octave_elem.text is None
                        or duration_elem.text is None
                    ):
                        raise ValueError(
                            f"Malformed MusicXML data: Empty notation nodes found "
                            f"in score file '{file_path}'."
                        )

                    raw_octave: str = octave_elem.text.strip()
                    raw_duration: str = duration_elem.text.strip()

                    # Explicitly verify the text strings are structural numbers
                    if (
                        not raw_octave.replace("-", "").isdigit()
                        or not raw_duration.isdigit()
                    ):
                        raise ValueError(
                            f"Malformed MusicXML data: Invalid Octave '{raw_octave}' or "
                            f"Duration '{raw_duration}' layout found in score file '{file_path}'."
                        )

                    octave = int(raw_octave)
                    raw_xml_duration = int(raw_duration)

                    pitch_step = pitch_elem.text.upper().strip()
                    if pitch_step not in self._step_map:
                        raise ValueError(
                            f"Malformed MusicXML asset: Invalid pitch step '{pitch_step}' "
                            f"encountered in score file '{file_path}'."
                        )

                    alter_elem = element.find(path=".//alter")

                    # Apply the same text safety check to the optional alter property
                    if alter_elem is not None and alter_elem.text is not None:
                        raw_alter: str = alter_elem.text.strip()
                        if not raw_alter.replace("-", "").isdigit():
                            raise ValueError(
                                f"Malformed MusicXML data: Invalid Alteration value '{raw_alter}' "
                                f"found in score file '{file_path}'."
                            )
                        alteration = int(raw_alter)
                    else:
                        alteration = 0

                    # Private math helper reads the validated step string safely
                    midi_pitch: int = self._calculate_midi_pitch(
                        step=pitch_step, octave=octave, alter=alteration
                    )

                    baked_onset_ticks: int = round(
                        number=current_xml_time_accumulator * tick_scale_factor
                    )
                    baked_duration_ticks: int = round(
                        number=raw_xml_duration * tick_scale_factor
                    )

                    note_event = NoteEvent(
                        pitch=midi_pitch,
                        onset_ticks=baked_onset_ticks,
                        duration_ticks=baked_duration_ticks,
                        velocity=64,
                    )
                    score.events.append(note_event)

                    is_chord: bool = element.find(path="chord") is not None
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
        base_note: int = self._step_map[step]
        return int((octave + 1) * 12 + base_note + alter)
