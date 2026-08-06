"""Public API entry point for the score2dataset package.

Exposes core generation pipelines and domain-specific exception types
directly at the root namespace level.
"""

from score2dataset.audio_engine import AudioEngine, SfizzRenderEngine
from score2dataset.datamodels import PerformanceScore
from score2dataset.exceptions import (
    AudioEngineError,
    ExporterError,
    ParserError,
    ProcessorError,
    Score2DatasetError,
)
from score2dataset.exporters.midi_exporter import MidiExporter
from score2dataset.generator import DatasetGenerator
from score2dataset.parsers.musicxml_parser import MusicXMLParser
from score2dataset.parsers.score_parser import ScoreParser

__all__: list[str] = [
    "AudioEngine",
    "AudioEngineError",
    "DatasetGenerator",
    "ExporterError",
    "MidiExporter",
    "MusicXMLParser",
    "ParserError",
    "PerformanceScore",
    "ProcessorError",
    "Score2DatasetError",
    "ScoreParser",
    "SfizzRenderEngine",
]
