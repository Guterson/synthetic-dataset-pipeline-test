import xml.etree.ElementTree as ET


def parse_musicxml_to_midi_ticks(xml_path, default_volume=80):

    TPQN = 480

    tree = ET.parse(xml_path)
    root = tree.getroot()

    events = []

    note_base_midi = {"C": 12, "D": 14, "E": 16, "F": 17, "G": 19, "A": 21, "B": 23}

    for part in root.findall("part"):

        current_time_ticks = 0
        divisions = 1
        last_note_duration_ticks = 0

        for measure in part.findall("measure"):

            attributes = measure.find("attributes")
            if attributes is not None:

                div_elem = attributes.find("divisions")
                if div_elem is not None:
                    divisions = int(div_elem.text)

            for element in measure:
                if element.tag == "note":

                    is_chord = element.find("chord") is not None
                    is_rest = element.find("rest") is not None

                    duration_elem = element.find("duration")
                    if duration_elem is not None:
                        xml_duration = float(duration_elem.text)
                        duration_ticks = int(round((xml_duration / divisions) * TPQN))
                    else:
                        duration_ticks = 0

                    if is_chord:
                        start_time = current_time_ticks - last_note_duration_ticks
                    else:
                        start_time = current_time_ticks

                    if not is_rest:

                        pitch_elem = element.find("pitch")
                        if pitch_elem is not None:
                            step = pitch_elem.find("step").text
                            octave = int(pitch_elem.find("octave").text)

                            midi_note = note_base_midi[step] + (octave * 12)

                            alter_elem = pitch_elem.find("alter")
                            if alter_elem is not None:
                                midi_note += int(alter_elem.text)

                            event_obj = {
                                "pitch": midi_note,
                                "start_time": start_time,
                                "duration": duration_ticks,
                                "volume": default_volume,
                            }
                            events.append(event_obj)

                    # Avança o relógio geral se não for um acorde empilhado
                    if not is_chord:
                        current_time_ticks += duration_ticks
                        last_note_duration_ticks = duration_ticks

    return events


"""
# =====================================================================
# EXECUÇÃO DO TESTE
# =====================================================================
if __name__ == "__main__":
    xml_filename = "hungarian_dance_n5.musicxml"

    try:
        musical_events = parse_musicxml_to_midi_ticks(xml_filename)

        print(f"--- LISTA DE EVENTOS MIDI (Resolução: 480 TPQN) ---")
        for idx, ev in enumerate(musical_events[:]):
            print(
                f"Nota {idx+1:02d} -> Pitch: {ev['pitch']} | Início: {ev['start_time']:5d} ticks | Duração: {ev['duration']:4d} ticks | Volume: {ev['volume']}"
            )

    except FileNotFoundError:
        print(
            f"Erro: Certifique-se de ter o arquivo '{xml_filename}' gerado para o teste."
        )
"""
