"""Unit tests for the audio rendering engines.

Ensures that the interface contracts are maintained and that subprocess
command matrices for the sfizz compiler are structured accurately.
"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from score2dataset.audio_engine import AudioEngine, SfizzRenderEngine
from score2dataset.exceptions import AudioEngineError


class TestAudioEngineContract(unittest.TestCase):
    """Verifies that the abstract base class enforces its structural rules."""

    def test_cannot_instantiate_abstract_base_class(self) -> None:
        """Ensure the blueprint cannot be run directly without subclasses."""
        with self.assertRaises(TypeError):
            _ = AudioEngine()  # type: ignore[abstract]


class TestSfizzRenderEngine(unittest.TestCase):
    """Encapsulates isolated unit scenarios for the Sfizz binary connector."""

    @patch("score2dataset.audio_engine.Path.exists")
    def test_initialization_with_valid_explicit_binary(
        self, mock_exists: MagicMock
    ) -> None:
        """Verify engine accepts explicit files when they pass physical disk guards."""
        # Force the path validation guard to return True for any file check
        mock_exists.return_value = True

        engine = SfizzRenderEngine(
            sfz_path="/fake/instrument.sfz",
            binary_path="/fake/bin/sfizz_render",
        )

        self.assertEqual(engine.sfz_path, Path("/fake/instrument.sfz").resolve())
        self.assertEqual(engine.binary, Path("/fake/bin/sfizz_render").resolve())

    @patch("score2dataset.audio_engine.Path.exists")
    def test_initialization_raises_file_not_found_for_missing_sfz(
        self, mock_exists: MagicMock
    ) -> None:
        """Ensure initialization terminates immediately if the instrument file drops out."""
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError) as context:
            _ = SfizzRenderEngine(sfz_path="/missing/instrument.sfz")

        self.assertIn("Target SFZ instrument not found", str(context.exception))

    @patch("score2dataset.audio_engine.subprocess.run")
    @patch("score2dataset.audio_engine.Path.exists")
    def test_render_audio_compiles_matrix_correctly(
        self,
        mock_exists: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Verify command arguments assemble strings cleanly for process injection."""
        mock_exists.return_value = True

        engine = SfizzRenderEngine(
            sfz_path="/fake/instrument.sfz",
            binary_path="/fake/bin/sfizz_render",
        )

        midi_path = Path("/fake/input.mid")
        output_wav = Path("/fake/output.wav")

        # Execute the process trigger
        engine.render_audio(
            midi_path=midi_path,
            output_wav_path=output_wav,
            sample_rate=48000,
            block_size=256,
            polyphony=32,
        )

        # Inspect the exact array arguments passed to the sub-shell
        mock_run.assert_called_once()
        called_args = mock_run.call_args[1]["args"]

        expected_matrix: list[str] = [
            str(Path("/fake/bin/sfizz_render").resolve()),
            "--sfz",
            str(Path("/fake/instrument.sfz").resolve()),
            "--midi",
            str(midi_path.resolve()),
            "--wav",
            str(output_wav.resolve()),
            "--mono",
            "--samplerate",
            "48000",
            "--blocksize",
            "256",
            "--voices",
            "32",
        ]
        self.assertEqual(called_args, expected_matrix)

    @patch("score2dataset.audio_engine.subprocess.run")
    @patch("score2dataset.audio_engine.Path.exists")
    def test_render_audio_translates_process_errors_to_runtime_faults(
        self,
        mock_exists: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Ensure failed process exits translate safely into system errors."""
        mock_exists.return_value = True

        # Simulate a crash inside the compiled C++ executable shell
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["sfizz_render"],
            stderr="NFS file lock timeout exception",
        )

        engine = SfizzRenderEngine(
            sfz_path="/fake/instrument.sfz",
            binary_path="/fake/bin/sfizz_render",
        )

        with self.assertRaises(AudioEngineError) as context:
            engine.render_audio(
                midi_path=Path("in.mid"), output_wav_path=Path("out.wav")
            )

        self.assertEqual(context.exception.returncode, 1)
        self.assertEqual(context.exception.details, "NFS file lock timeout exception")


if __name__ == "__main__":
    unittest.main()
