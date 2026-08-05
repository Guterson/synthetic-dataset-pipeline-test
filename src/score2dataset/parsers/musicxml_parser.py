"""MusicXML and MXL notation schema parser.

This module extracts pitch and rhythmic grids from structured XML elements,
populating standardized PerformanceScore containers alongside deep Expression Maps.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

from score2dataset.config import DEFAULT_VELOCITY, NOMINAL_VELOCITIES, STANDARD_TPQN
from score2dataset.datamodels import (
    NoteEvent,
    PerformanceScore,
    ScoreExpressionMap,
    WedgeEvent,
)
from score2dataset.exceptions import ParserError
from score2dataset.parsers.score_parser import ScoreParser


class MusicXMLParser(ScoreParser):
    """Concrete parser implementation for reading MusicXML format sheets.

    Converts localized raw notation definitions, divisions, and time counters
    into a uniform, integer-based timeline mapped to an abstract tick resolution.
    """

    _STEP_MAP: Final[dict[str, int]] = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    def __init__(self, target_tpqn: int = STANDARD_TPQN) -> None:
        """Initializes the parser configurations and immutable musical maps."""
        self.target_tpqn: int = target_tpqn

    def parse_score(
        self, file_path: Path
    ) -> tuple[PerformanceScore, ScoreExpressionMap]:
        """Parses an external MusicXML score file into a standard PerformanceScore and ExpressionMap."""
        tree = ET.parse(source=file_path)
        root = tree.getroot()

        score = PerformanceScore(source=file_path)
        expression_map = ScoreExpressionMap(ticks_per_beat=self.target_tpqn)

        # Enforce global scale parameter validations
        divisions_elem = root.find(path=".//attributes/divisions")
        if (
            divisions_elem is None
            or divisions_elem.text is None
            or not divisions_elem.text.strip().isdigit()
        ):
            raise ParserError(
                f"Invalid MusicXML structure: Global <divisions> definition error in '{file_path}'."
            )

        xml_divisions = int(divisions_elem.text)
        if xml_divisions == 0:
            raise ParserError(
                f"Invalid MusicXML layout: Global <divisions> cannot be zero in '{file_path}'."
            )

        tick_scale_factor: float = self.target_tpqn / xml_divisions
        self._parse_global_metadata(root, expression_map)

        current_velocity = DEFAULT_VELOCITY
        active_wedges: dict[str, WedgeEvent] = {}

        for part in root.findall(path=".//part"):
            active_wedges = {}
            active_slurs: dict[str, int] = {}
            current_xml_time_accumulator = 0
            chord_root_onset = 0

            for measure in part.findall(path=".//measure"):
                for element in measure:
                    absolute_tick = round(
                        current_xml_time_accumulator * tick_scale_factor
                    )

                    if element.tag == "direction":
                        current_velocity = self._parse_direction_element(
                            element,
                            absolute_tick,
                            expression_map,
                            active_wedges,
                            current_velocity,
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

                        is_chord = element.find(path="chord") is not None
                        if is_chord:
                            target_onset_time = chord_root_onset
                        else:
                            target_onset_time = current_xml_time_accumulator
                            chord_root_onset = current_xml_time_accumulator
                        baked_onset_ticks = round(target_onset_time * tick_scale_factor)

                        # Extract isolated local metadata layers safely
                        self._parse_note_metadata(
                            element, baked_onset_ticks, expression_map, active_slurs
                        )

                        pitch_elem = element.find(path=".//step")
                        octave_elem = element.find(path=".//octave")

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
                            baked_duration_ticks = round(
                                raw_xml_duration * tick_scale_factor
                            )

                            note_event = NoteEvent(
                                pitch=midi_pitch,
                                onset_ticks=baked_onset_ticks,
                                duration_ticks=baked_duration_ticks,
                                velocity=current_velocity,
                            )
                            score.events.append(note_event)

                        if not is_chord:
                            current_xml_time_accumulator += raw_xml_duration

            final_calculated_tick = round(
                current_xml_time_accumulator * tick_scale_factor
            )
            expression_map.total_ticks = max(
                expression_map.total_ticks, final_calculated_tick
            )

        for wedge in active_wedges.values():
            if wedge.offset_tick == -1:
                wedge.offset_tick = expression_map.total_ticks

        return score, expression_map

    def _parse_global_metadata(
        self, root: ET.Element, expression_map: ScoreExpressionMap
    ) -> None:
        """Extracts primary signature and initial score word tokens."""
        beats_elem = root.find(path=".//attributes/time/beats")
        if (
            beats_elem is not None
            and beats_elem.text is not None
            and beats_elem.text.strip().isdigit()
        ):
            expression_map.beats_per_bar = int(beats_elem.text.strip())

        words_elem = root.find(path=".//direction/direction-type/words")
        if words_elem is not None and words_elem.text is not None:
            expression_map.initial_tempo_marking = words_elem.text.strip()

    def _parse_direction_element(
        self,
        element: ET.Element,
        tick: int,
        expression_map: ScoreExpressionMap,
        active_wedges: dict,
        current_velocity: int,
    ) -> int:
        """Processes spanners, text, and volume changes inside direction tags."""
        words = element.find(path=".//direction-type/words")
        if words is not None and words.text is not None:
            expression_map.text_directions[tick] = words.text.strip()

        wedge_elem = element.find(path=".//direction-type/wedge")
        if wedge_elem is not None:
            w_type = wedge_elem.get("type")
            w_num = wedge_elem.get("number", "1")

            if w_type in ["crescendo", "decrescendo"]:
                new_wedge = WedgeEvent(wedge_type=w_type, onset_tick=tick)
                active_wedges[w_num] = new_wedge
                expression_map.dynamic_wedges.append(new_wedge)
            elif w_type == "stop" and w_num in active_wedges:
                active_wedges[w_num].offset_tick = tick
                del active_wedges[w_num]

        dyn_elem = element.find(path=".//dynamics/*")
        if dyn_elem is not None:
            return NOMINAL_VELOCITIES.get(dyn_elem.tag.lower(), DEFAULT_VELOCITY)

        return current_velocity

    def _parse_note_metadata(
        self,
        element: ET.Element,
        tick: int,
        expression_map: ScoreExpressionMap,
        active_slurs: dict,
    ) -> None:
        """Extracts articulation types and maps active slur spanners."""
        articulation_elements = element.findall(path=".//notations/articulations/*")
        if articulation_elements:
            if tick not in expression_map.local_articulations:
                expression_map.local_articulations[tick] = set()
            for art_node in articulation_elements:
                expression_map.local_articulations[tick].add(art_node.tag.lower())

        slur_elements = element.findall(path=".//notations/slur")
        for slur_node in slur_elements:
            s_type = slur_node.get("type")
            s_num = slur_node.get("number", "1")

            if s_type == "start":
                active_slurs[s_num] = tick
            elif s_type == "stop" and s_num in active_slurs:
                expression_map.slur_phrases.append((active_slurs[s_num], tick))
                del active_slurs[s_num]

    def _calculate_midi_pitch(self, step: str, octave: int, alter: int) -> int:
        """Calculates the absolute MIDI note number from standard music variables."""
        return 12 * (octave + 1) + self._STEP_MAP[step] + alter
