"""Functional calculation and data mapping test suite for the MusicXML parser.

This module exercises the concrete MusicXML parsing engine, validating pitch math,
timeline navigation via forward/backward jumps, polyphonic chord stacking, and the
extraction of deep score expressions like dynamic hairpins and slur spanners.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from score2dataset.datamodels import PerformanceScore, ScoreExpressionMap
from score2dataset.parsers.musicxml_parser import MusicXMLParser, ParserError


def test_parser_initialization_with_custom_tpqn() -> None:
    """Verify initialization sets the target TPQN property properly."""
    parser = MusicXMLParser(target_tpqn=960)
    assert parser.target_tpqn == 960


def test_parse_score_raises_value_error_on_missing_divisions() -> None:
    """Ensure parsing fails immediately if the global divisions element is missing."""
    invalid_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <score-partwise version="4.0">
      <part id="P1">
        <measure number="1">
          <attributes>
            <!-- Missing divisions element completely -->
          </attributes>
        </measure>
      </part>
    </score-partwise>
    """
    parser = MusicXMLParser()
    mock_path = Path("fake_score.musicxml")

    with (
        patch("builtins.open", mock_open(read_data=invalid_xml)),
        patch("xml.etree.ElementTree.parse") as mock_et,
    ):
        mock_et.return_value = ET.ElementTree(ET.fromstring(invalid_xml))

        with pytest.raises(ParserError):
            parser.parse_score(file_path=mock_path)


def test_parse_score_raises_value_error_on_zero_divisions() -> None:
    """Ensure the parser safely rejects calculations dividing by zero."""
    zero_div_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <score-partwise version="4.0">
      <part id="P1">
        <measure number="1">
          <attributes>
            <divisions>0</divisions>
          </attributes>
        </measure>
      </part>
    </score-partwise>
    """
    parser = MusicXMLParser()
    mock_path = Path("zero_div.musicxml")

    with (
        patch("builtins.open", mock_open(read_data=zero_div_xml)),
        patch("xml.etree.ElementTree.parse") as mock_et,
    ):
        mock_et.return_value = ET.ElementTree(ET.fromstring(zero_div_xml))

        with pytest.raises(ParserError):
            parser.parse_score(mock_path)


def test_parse_score_successful_mapping() -> None:
    """Verify pitch calculation, forward jumps, chord accumulation, and defaults."""
    valid_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <score-partwise version="4.0">
      <part id="P1">
        <measure number="1">
          <attributes>
            <divisions>1</divisions>
          </attributes>
          <note>
            <pitch>
              <step>C</step>
              <octave>4</octave>
            </pitch>
            <duration>1</duration>
          </note>
          <forward>
            <duration>2</duration>
          </forward>
          <note>
            <pitch>
              <step>F</step>
              <octave>4</octave>
              <alter>1</alter>
            </pitch>
            <duration>1</duration>
          </note>
        </measure>
      </part>
    </score-partwise>
    """
    parser = MusicXMLParser(target_tpqn=480)
    mock_path = Path("valid_test.musicxml")

    with (
        patch("builtins.open", mock_open(read_data=valid_xml)),
        patch("xml.etree.ElementTree.parse") as mock_et,
    ):
        mock_et.return_value = ET.ElementTree(ET.fromstring(valid_xml))
        score, _ = parser.parse_score(file_path=mock_path)

    assert isinstance(score, PerformanceScore)
    assert len(score.events) == 2

    note_one = score.events[0]
    assert note_one.pitch == 60
    assert note_one.onset_ticks == 0
    assert note_one.duration_ticks == 480
    assert note_one.score_expected == 1
    assert note_one.audio_present == 1

    note_two = score.events[1]
    assert note_two.pitch == 66
    assert note_two.onset_ticks == 1440
    assert note_two.score_expected == 1
    assert note_two.audio_present == 1


def test_parse_score_complex_expressions_and_jumps() -> None:
    """Verify backing time jumps, chords, dynamics, and slur spanner tracking maps."""
    complex_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <score-partwise version="4.0">
      <part id="P1">
        <measure number="1">
          <attributes>
            <divisions>1</divisions>
          </attributes>
          <direction>
            <direction-type><words>Adagio</words></direction-type>
            <direction-type><wedge type="crescendo" number="1"/></direction-type>
          </direction>
          <note>
            <pitch>
              <step>E</step>
              <octave>4</octave>
            </pitch>
            <duration>2</duration>
            <notations>
              <slur type="start" number="1"/>
            </notations>
          </note>
          <backup>
            <duration>2</duration>
          </backup>
          <note>
            <chord/>
            <pitch>
              <step>G</step>
              <octave>4</octave>
            </pitch>
            <duration>2</duration>
            <notations>
              <slur type="stop" number="1"/>
            </notations>
          </note>
          <direction>
            <direction-type><wedge type="stop" number="1"/></direction-type>
          </direction>
        </measure>
      </part>
    </score-partwise>
    """
    parser = MusicXMLParser(target_tpqn=480)
    mock_path = Path("complex_test.musicxml")

    with (
        patch("builtins.open", mock_open(read_data=complex_xml)),
        patch("xml.etree.ElementTree.parse") as mock_et,
    ):
        mock_et.return_value = ET.ElementTree(ET.fromstring(complex_xml))
        score, expression_map = parser.parse_score(file_path=mock_path)

    assert isinstance(expression_map, ScoreExpressionMap)
    assert len(score.events) == 2
    assert score.events[0].onset_ticks == 0
    assert score.events[1].onset_ticks == 0

    assert expression_map.initial_tempo_marking == "Adagio"
    assert len(expression_map.dynamic_wedges) == 1
    assert expression_map.dynamic_wedges[0].wedge_type == "crescendo"
    assert len(expression_map.slur_phrases) == 1
    assert expression_map.slur_phrases[0] == (0, 0)
