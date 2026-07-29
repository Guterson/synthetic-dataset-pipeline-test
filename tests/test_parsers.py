"""Unit tests for the score parsing infrastructure.

Ensures that structural abstractions strictly enforce API contracts
and that concrete MusicXML parsers process valid/invalid XML layouts safely.
"""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import mock_open, patch

from score2dataset.datamodels import PerformanceScore
from score2dataset.parsers.musicxml_parser import MusicXMLParser
from score2dataset.parsers.score_parser import ScoreParser


class TestScoreParserContract(unittest.TestCase):
    """Verifies that the ScoreParser interface cannot be misconfigured."""

    def test_cannot_instantiate_abstract_base_class(self) -> None:
        """Ensure the engine blocks initialization of a raw ScoreParser."""
        with self.assertRaises(TypeError):
            _ = ScoreParser()  # type: ignore[abstract]

    def test_subclass_must_implement_parse_score(self) -> None:
        """Ensure any child class is forced to implement parse_score."""

        class BrokenParser(ScoreParser):
            """A parser that forgets to implement the required method."""

        with self.assertRaises(TypeError):
            _ = BrokenParser()


class TestMusicXMLParser(unittest.TestCase):
    """Encapsulates test cases for structural and calculation layout rules."""

    def test_parser_initialization_with_custom_tpqn(self) -> None:
        """Verify initialization sets the target TPQN property properly."""
        parser = MusicXMLParser(target_tpqn=960)
        self.assertEqual(parser.target_tpqn, 960)

    def test_parse_score_raises_value_error_on_missing_divisions(self) -> None:
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

        # Mock standard open/read and ElementTree parsing using our string layout
        with (
            patch("builtins.open", mock_open(read_data=invalid_xml)),
            patch("xml.etree.ElementTree.parse") as mock_et,
        ):

            mock_et.return_value = ET.ElementTree(ET.fromstring(invalid_xml))

            with self.assertRaises(ValueError) as context:
                parser.parse_score(file_path=mock_path)

            self.assertIn("Global <divisions> definition", str(context.exception))

    def test_parse_score_raises_value_error_on_zero_divisions(self) -> None:
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

            with self.assertRaises(ValueError) as context:
                parser.parse_score(mock_path)

            self.assertIn("cannot be zero", str(context.exception))


class TestMusicXMLParserIntegration(unittest.TestCase):
    """Verifies successful data mapping and pitch math for valid XML blocks."""

    def test_parse_score_successful_mapping(self) -> None:
        """Verify pitch calculation, forward jumps, and chord accumulation."""
        # A valid score: C4 (MIDI 60) for 1 duration, a forward skip, then F#4 (MIDI 66)
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

            score: PerformanceScore = parser.parse_score(file_path=mock_path)

        # Assertions to verify global layout outputs
        self.assertIsInstance(score, PerformanceScore)
        self.assertEqual(len(score.events), 2)

        # Note 1 verification: C4 -> (4+1)*12 + 0 + 0 = 60. Onset: 0
        note_one = score.events[0]
        self.assertEqual(note_one.pitch, 60)
        self.assertEqual(note_one.onset_ticks, 0)
        self.assertEqual(note_one.duration_ticks, 480)

        # Note 2 verification: F#4 -> (4+1)*12 + 5 + 1 = 66.
        # Onset calculation: 0 + 1 (duration) + 2 (forward) = 3 xml time units.
        # 3 units * 480 scale factor = 1440 ticks.
        note_two = score.events[1]
        self.assertEqual(note_two.pitch, 66)
        self.assertEqual(note_two.onset_ticks, 1440)


if __name__ == "__main__":
    unittest.main()
