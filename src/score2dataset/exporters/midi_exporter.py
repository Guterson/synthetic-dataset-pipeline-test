"""MIDI serialization and timeline rendering engine.

This module converts internal, duration-based PerformanceScore data models
back into sequential, delta-time MIDI streams compliant with standard MIDI file formats.
"""

from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack

from score2dataset.datamodels import PerformanceScore


class MidiExporter:
    """Serializes in-memory performance timelines into standard MIDI files.

    Translates absolute onset coordinates and duration vectors into sequential
    delta-time message pools, enforcing protocol value constraints.
    """

    def __init__(self, target_tpqn: int = 480, fixed_bpm: int = 120) -> None:
        """Initializes the MIDI exporter configuration.

        Args:
            target_tpqn: Ticks Per Quarter Note resolution forced onto the
                output MIDI file structure. Defaults to 480.
        """
        self.target_tpqn: int = target_tpqn
        self.fixed_bpm: int = fixed_bpm

    def export_score(
        self,
        performance_score: PerformanceScore,
        output_path: Path,
        instrument_program: int = 0,
    ) -> None:
        """Serializes an absolute integer tick timeline into a flat MIDI file.

        Sorts concurrent events chronologically, resolves polyphonic note spans
        into progressive delta ticks, and saves the file to disk.

        Args:
            performance_score: The mutated PerformanceScore container to serialize.
            output_path: Target file system path destination for the exported file (.mid).
            instrument_program: Standard MIDI patch program index (0-127). Defaults to 0.

        Raises:
            IOError: If writing to the target output path fails due to system constraints.
        """
        mid = MidiFile(ticks_per_beat=self.target_tpqn)
        track = MidiTrack()
        mid.tracks.append(track)

        # Enforce constant 120 BPM metadata parameters (500,000 microseconds per beat)
        tempo_microsec: int = round(number=60_000_000 / self.fixed_bpm)
        track.append(MetaMessage(type="set_tempo", tempo=tempo_microsec, time=0))
        track.append(Message(type="program_change", program=instrument_program, time=0))

        midi_messages_pool: list[dict] = []

        for event in performance_score.events:
            # Enforce numerical clip constraints within protocol boundaries
            pitch: int = max(0, min(int(event.pitch), 127))
            velocity: int = max(0, min(int(event.velocity), 127))

            # Resolve coordinates into absolute start and terminal boundaries
            abs_start_tick = max(0, int(event.onset_ticks))
            dur_ticks = max(1, int(event.duration_ticks))
            abs_end_tick = abs_start_tick + dur_ticks

            # Pool independent edge markers for chronological layout grouping
            midi_messages_pool.append(
                {
                    "type": "note_on",
                    "pitch": pitch,
                    "velocity": velocity,
                    "abs_tick": abs_start_tick,
                }
            )
            midi_messages_pool.append(
                {
                    "type": "note_off",
                    "pitch": pitch,
                    "velocity": 0,
                    "abs_tick": abs_end_tick,
                }
            )

        # Note-Off message priority resolves hanging overlaps on identical timestamps
        midi_messages_pool.sort(
            key=lambda x: (x["abs_tick"], 0 if x["type"] == "note_off" else 1)
        )

        # Deform absolute intervals into progressive delta time offsets
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

        mid.save(filename=str(object=output_path))
