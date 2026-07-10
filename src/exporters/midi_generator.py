from mido import Message, MetaMessage, MidiFile, MidiTrack
from src.datamodels import PerformanceScore, NoteEvent
from src.parsers.score_parser import MusicXMLToScoreParser


class PerformanceToMIDIConverter:
    def __init__(self, fixed_bpm: int = 120, ticks_per_beat: int = 480):
        """
        Initializes the MIDI generator with a fixed Time Code Resolution (PPQN).
        All inputs are assumed to be pre-scaled to this specific resolution at 120 BPM.
        """
        self.ticks_per_beat = ticks_per_beat
        self.fixed_bpm = fixed_bpm

    def generate(
        self,
        performance_score: PerformanceScore,
        output_path: str = "output.mid",
        instrument_program: int = 0,
    ):
        """
        Serializes pre-baked, fixed-tempo integer ticks directly into a standard MIDI file.
        """
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Write constant 120 BPM metadata to the MIDI header (500,000 microseconds per beat)
        tempo_microsec = int(round(60_000_000 / self.fixed_bpm))
        track.append(MetaMessage(type="set_tempo", tempo=tempo_microsec, time=0))
        track.append(Message("program_change", program=instrument_program, time=0))

        midi_messages_pool = []

        for event in performance_score.events:
            # Enforce compatibility with the MIDI protocol boundaries
            pitch = max(0, min(int(event.pitch), 127))
            velocity = max(0, min(int(event.velocity), 127))

            # Read pre-baked temporal data
            abs_start_tick = max(0, int(event.onset_ticks))
            dur_ticks = max(1, int(event.duration_ticks))
            abs_end_tick = abs_start_tick + dur_ticks

            # Pool Note-On and Note-Off events using absolute timeline ticks
            midi_messages_pool.append(
                {
                    "type": "note_on",
                    "pitch": pitch,
                    "velocity": velocity,
                    "abs_tick": abs_start_tick,
                }
            )  # Verification block with the streamlined attribute names
            midi_messages_pool.append(
                {
                    "type": "note_off",
                    "pitch": pitch,
                    "velocity": 0,
                    "abs_tick": abs_end_tick,
                }
            )

        # Chronological sorting: Note-Off executes first if timestamps are identical
        midi_messages_pool.sort(
            key=lambda x: (x["abs_tick"], 0 if x["type"] == "note_off" else 1)
        )

        # Flatten absolute timeline positions into sequential Delta Times
        last_abs_tick = 0
        for msg in midi_messages_pool:
            delta_tick = msg["abs_tick"] - last_abs_tick
            last_abs_tick = msg["abs_tick"]

            track.append(
                Message(
                    msg["type"],
                    note=msg["pitch"],
                    velocity=msg["velocity"],
                    time=delta_tick,
                )
            )

        mid.save(output_path)
        print(f"[MIDI Generator] Baked file exported successfully to: '{output_path}'")


if __name__ == "__main__":
    score = MusicXMLToScoreParser()
    performance = score.parse("hungarian_dance_n5.musicxml")

    generator = PerformanceToMIDIConverter()
    generator.generate(performance, "baked_tempo_test.mid")
