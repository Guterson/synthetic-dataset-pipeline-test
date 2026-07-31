"""Microtiming temporal jitter and motor noise modification processor.

Implements tempo-proportional variance mapping and first-order autoregressive
AR(1) error curves to simulate human physical timing drift and execution noise.
"""

from typing import Final

import numpy as np
from scipy.interpolate import CubicSpline

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.exceptions import ProcessorError


class TemporalJitterModifier:
    """Applies tempo-calibrated AR(1) human motor jitter to note onset grids."""

    def __init__(
        self,
        sigma_time: float = 0.010,
        phi: float = 0.35,
        target_tpqn: int = 480,
        seed: int | None = None,
    ) -> None:
        """Initializes the microtiming variance parameters and stochastic seed handles.

        Args:
            sigma_time: Target time-domain deviation standard deviation in seconds (e.g., 0.010s).
            phi: Autoregressive correlation coefficient tracking short-term memory trends.
            target_tpqn: Ticks Per Quarter Note timeline resolution constraint. Defaults to 480.
            seed: Fixed stochastic seed to ensure strict scientific transparency.
        """
        if not (0.0 <= phi < 1.0):
            raise ProcessorError(
                f"AR(1) coefficient Phi must reside in the interval [0, 1). Got: {phi}"
            )

        self.sigma_time: Final[float] = sigma_time
        self.phi: Final[float] = phi
        self.tpqn: Final[int] = target_tpqn
        self.rng: Final[np.random.Generator] = np.random.default_rng(seed)

    def perturb_timing(
        self, score: PerformanceScore, tempo_spline: CubicSpline
    ) -> PerformanceScore:
        """Applies dynamic autoregressive jitter to the score onset timeline.

        Groups note elements sharing chord structures to prevent un-synchronized
        arpeggiation errors, tracking temporal drift curves across execution loops.

        Args:
            score: The performance data score structure containing target note events.
            tempo_spline: Pre-calculated continuous cubic spline mapping absolute ticks to BPM.

        Returns:
            A new modified PerformanceScore instance with microtimed onset indices.
        """
        original_events: list[NoteEvent] = score.events
        if not original_events:
            return copy.deepcopy(score)

        # Step 1: Extract and group absolute timelines to identify chord bounds
        # Maps unique onset tick targets to all note instances striking on that beat
        chord_groups: dict[int, list[NoteEvent]] = {}
        for event in original_events:
            chord_groups.setdefault(event.onset_ticks, []).append(event)

        # Ensure we process the performance chronologically from start to finish
        sorted_onsets: list[int] = sorted(chord_groups.keys())

        modified_events: list[NoteEvent] = []
        previous_delta: float = 0.0

        # Step 2: Loop chronologically across the unique time moments
        for onset_tick in sorted_onsets:
            notes_in_chord = chord_groups[onset_tick]

            # Evaluate the instantaneous tempo curve to find the exact local BPM
            instantaneous_bpm: float = float(tempo_spline(onset_tick))

            # Equation: sigma_ticks = sigma_time * (B_ki * TPQN) / 60
            sigma_ticks: float = self.sigma_time * (
                (instantaneous_bpm * self.tpqn) / 60.0
            )

            # Generate the white noise innovation vector element
            epsilon: float = float(self.rng.normal(0.0, sigma_ticks))

            # Equation: delta_i = phi * delta_i-1 + epsilon_i
            current_delta: float = (self.phi * previous_delta) + epsilon
            previous_delta = current_delta

            # Convert floating-point error offset cleanly into discrete grid units
            tick_displacement = int(np.round(current_delta))

            # Step 3: Apply the identical displacement step uniformly to all notes in the chord
            for event in notes_in_chord:
                perturbed_onset = int(
                    np.clip(event.onset_ticks + tick_displacement, 0, None)
                )

                modified_event = NoteEvent(
                    pitch=event.pitch,
                    onset_ticks=perturbed_onset,
                    # Structural release coordinates are preserved as defined in your paper text
                    duration_ticks=event.duration_ticks,
                    velocity=event.velocity,
                )
                modified_events.append(modified_event)

        # Step 4: Re-sort the final timeline to protect structural execution linearity
        modified_events.sort(key=lambda e: e.onset_ticks)

        perturbed_score = PerformanceScore(source=score.source)
        perturbed_score.events = modified_events
        return perturbed_score
