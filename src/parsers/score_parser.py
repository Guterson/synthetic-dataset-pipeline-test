import xml.etree.ElementTree as ET
from src.datamodels import PerformanceScore, NoteEvent


class MusicXMLToScoreParser:
    def __init__(self, fixed_bpm: int = 120, target_tpqn: int = 480):
        """
        Initializes the parser.
        :param target_tpqn: Ticks Per Quarter Note resolution forced onto the output data.
        """
        self.target_tpqn = target_tpqn
        # Output is locked to a constant 120 BPM world coordinate system
        self.fixed_bpm = fixed_bpm

    def parse(self, xml_path: str) -> PerformanceScore:
        """
        Parses a MusicXML file and bakes notes straight into the unified integer tick schema.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        score = PerformanceScore()

        # MusicXML uses internal division integers per quarter note to express fractions.
        # We find this value to calculate a strict scaling factor relative to our target TPQN.
        divisions_elem = root.find(".//attributes/divisions")
        if divisions_elem is None:
            raise ValueError("Invalid MusicXML: Global <divisions> definition missing.")

        xml_divisions = int(divisions_elem.text)
        # Scale factor converts raw XML duration units to your target 480 TPQN layout
        tick_scale_factor = self.target_tpqn / xml_divisions

        # Track absolute elapsed position across the track layout in raw XML units
        current_xml_time_accumulator = 0

        # Iterate through elements sequentially inside the first part/measure stream
        for element in root.findall(".//measure/*"):

            if element.tag == "forward":
                # Advance global timeline marker (e.g., structural rests)
                dist = element.find("duration")
                if dist is not None:
                    current_xml_time_accumulator += int(dist.text)

            elif element.tag == "backup":
                # Move timeline backwards (e.g., managing polyphonic voices/harmonies)
                dist = element.find("duration")
                if dist is not None:
                    current_xml_time_accumulator -= int(dist.text)

            elif element.tag == "note":
                # Process structural note definitions
                is_rest = element.find("rest") is not None
                if is_rest:
                    dur_elem = element.find("duration")
                    if dur_elem is not None:
                        current_xml_time_accumulator += int(dur_elem.text)
                    continue

                # Extract frequency/pitch metadata
                pitch_elem = element.find(".//step")
                octave_elem = element.find(".//octave")
                duration_elem = element.find("duration")

                if (
                    pitch_elem is not None
                    and octave_elem is not None
                    and duration_elem is not None
                ):
                    # Convert musical pitch step string to raw MIDI note number
                    pitch_step = pitch_elem.text
                    octave = int(octave_elem.text)
                    raw_xml_duration = int(duration_elem.text)

                    alter_elem = element.find(".//alter")
                    alteration = int(alter_elem.text) if alter_elem is not None else 0

                    midi_pitch = self._calculate_midi_pitch(
                        pitch_step, octave, alteration
                    )

                    # --- CONVERSION TO TARGET TIMELINE INT TICKS ---
                    # Map the accumulated raw units directly into the absolute 480 TPQN domain
                    baked_onset_ticks = int(
                        round(current_xml_time_accumulator * tick_scale_factor)
                    )
                    baked_duration_ticks = int(
                        round(raw_xml_duration * tick_scale_factor)
                    )

                    # Instantiate the simplified data model with clean integer coordinates
                    note_event = NoteEvent(
                        pitch=midi_pitch,
                        onset_ticks=baked_onset_ticks,
                        duration_ticks=baked_duration_ticks,
                        velocity=64,  # Uniform baseline dynamic layer before modifications
                    )
                    score.events.append(note_event)

                    # Check for chords: A chord note shares the onset with the main note.
                    # It must NOT advance the main time accumulator.
                    is_chord = element.find("chord") is not None
                    if not is_chord:
                        current_xml_time_accumulator += raw_xml_duration

        return score

    def _calculate_midi_pitch(self, step: str, octave: int, alter: int) -> int:
        """Helper to calculate standard MIDI key indices."""
        step_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        base_note = step_map.get(step.upper(), 0)
        return int((octave + 1) * 12 + base_note + alter)
