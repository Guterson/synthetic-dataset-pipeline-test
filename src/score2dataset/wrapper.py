"""Abstraction layer and engine interfaces for audio synthesizer plugins.

This module defines the universal execution template for audio rendering backends,
allowing the dataset pipeline to remain agnostic of the underlying synthesizer engine.
"""

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal


class AudioEngine(ABC):
    """Abstract base class establishing the contract for all rendering engines.

    Any audio synthesizer module integrated into the package must subclass
    this template and implement its methods to guarantee pipeline compatibility.
    """

    @abstractmethod
    def render_audio(
        self,
        midi_path: Path,
        output_wav_path: Path,
        sample_rate: int = 48000,
        block_size: int = 512,
        polyphony: int = 64,
    ) -> None:
        """Executes an offline rendering pass to convert MIDI data into audio.

        Args:
            midi_path: Location of the input source MIDI file (.mid).
            output_wav_path: Destination path where the rendered audio is written.
            sample_rate: Frequency resolution profile in Hz.
            block_size: Audio vector block streaming constraints.
            polyphony: Maximum simultaneous voice allocation limit.

        Raises:
            RuntimeError: If the underlying synthesis process encounters a fault.
        """


class SfizzRenderEngine(AudioEngine):
    """Sfizz-specific implementation of the universal audio engine interface.

    Spawns and manages the locally compiled native C++ sfizz_render engine
    as an isolated child process to generate acoustic samples from SFZ libraries.
    """

    def __init__(self, sfz_path: str, binary_path: str | None = None) -> None:
        """Locates the sfizz binary and locks it to a specific instrument layout.

        Args:
            sfz_path: Path to the target SFZ virtual instrument file to load.
            binary_path: Optional explicit string path to the sfizz_render binary.

        Raises:
            FileNotFoundError: If the execution binary or SFZ file cannot be located.
        """
        self.sfz_path: Path = Path(sfz_path).resolve()
        if not self.sfz_path.exists():
            raise FileNotFoundError(f"Target SFZ instrument not found: {self.sfz_path}")

        if binary_path:
            self.binary: Path = Path(binary_path).resolve()
        else:
            self.binary = self._discover_binary()

        if not self.binary.exists():
            raise FileNotFoundError(f"Sfizz rendering binary not found: {self.binary}")

    def _discover_binary(self) -> Path | None:
        """Scans standardized system paths to locate the sfizz_render executable.

        Returns:
            A validated Path object pointing to the binary file destination.
        """

        # Inject your specific lab user-space bin directory into the lookup array
        user_bin_tree: str = os.path.expanduser(path="~/.local/bin")

        # Combine your local directory with the global system PATH string
        search_paths: str = f"{user_bin_tree}{os.pathsep}{os.environ.get('PATH', '')}"

        # Perform a single, exhaustive system search using the combined paths
        lookup: str | None = shutil.which(
            "sfizz_render", path=search_paths
        ) or shutil.which("sfizz-render", path=search_paths)

        if lookup:
            return Path(lookup)

        return None

    def render_audio(
        self,
        midi_path: Path,
        output_wav_path: Path,
        sample_rate: int = 44100,
        block_size: int = 512,
        polyphony: int = 64,
    ) -> None:
        """Executes the sfizz command-line tool matrix to render raw audio blocks."""
        command_matrix: list[str] = [
            str(object=self.binary),
            "--sfz",
            str(object=self.sfz_path),
            "--midi",
            str(object=midi_path.resolve()),
            "--wav",
            str(object=output_wav_path.resolve()),
            "--mono",
            "--samplerate",
            str(object=sample_rate),
            "--blocksize",
            str(object=block_size),
            "--voices",
            str(object=polyphony),
        ]

        try:
            subprocess.run(
                args=command_matrix,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            error_details: str | Literal["Unknown fault."] = (
                error.stderr.strip() if error.stderr else "Unknown fault."
            )
            raise RuntimeError(
                f"Sfizz engine rendering execution failed.\nDetails: {error_details}"
            ) from error
