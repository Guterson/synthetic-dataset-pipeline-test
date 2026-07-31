"""Articulatory variance and note duration modification processor.

Implements intentional stylistic envelope scaling (staccato/legato) combined
with Gaussian motor noise residuals to simulate human physical touch variations.
"""

import copy
from typing import Final

import numpy as np

from score2dataset.datamodels import NoteEvent, PerformanceScore, ScoreExpressionMap
from score2dataset.exceptions import ProcessorError


class ArticulationModifier:
    """Modulates symbolic note durations using intentional style maps and motor noise."""

    def __init__(
        self,
        sigma_d: float = 8.0,
        d_min: int = 24,
        staccato_low: float = 0.25,
        staccato_high: float = 0.45,
        neutral_low: float = 0.85,
        neutral_high: float = 0.95,
        legato_low: float = 1.05,
        legato_high: float = 1.25,
        seed: int | None = None,
    ) -> None:
        """Initializes the articulatory boundary ranges and random state handles.

        Args:
            sigma_d: Standard deviation for the Gaussian motor residual noise (zeta).
            d_min: Minimum allowable note duration in ticks to prevent transient cutoff.
            staccato_low: Minimum scaling boundary for staccato markings.
            staccato_high: Maximum scaling boundary for staccato markings.
            neutral_low: Minimum scaling boundary for standard unmarked notes.
            neutral_high: Maximum scaling boundary for standard unmarked notes.
            legato_low: Minimum scaling boundary for legato/slurred passages.
            legato_high: Maximum scaling boundary for legato/slurred passages.
            seed: Fixed stochastic seed to ensure scientific transparency.
        """
        if d_min <= 0:
            raise ProcessorError(
                f"Minimum physical duration d_min must be greater than zero. Got: {d_min}"
            )

        self.sigma_d: Final[float] = sigma_d
        self.d_min: Final[int] = d_min

        # Stochastic range vectors matching your qualitative paper categories
        self.range_staccato: Final[tuple[float, float]] = (staccato_low, staccato_high)
        self.range_neutral: Final[tuple[float, float]] = (neutral_low, neutral_high)
        self.range_legato: Final[tuple[float, float]] = (legato_low, legato_high)

        self.rng: Final[np.random.Generator] = np.random.default_rng(seed)

    def perturb_articulation(
        self, score: PerformanceScore, expression_map: ScoreExpressionMap
    ) -> PerformanceScore:
        """Evaluates localized score markings to structurally scale note durations.

        Args:
            score: The performance score structure containing target note events.
            expression_map: The extracted structural metadata and spanner coordinate tracker.

        Returns:
            A new modified PerformanceScore instance with adjusted duration_ticks.
        """
        original_events: list[NoteEvent] = score.events
        if not original_events:
            return copy.deepcopy(score)

        modified_events: list[NoteEvent] = []

        for event in original_events:
            # 1. Determine the intentional stylistic factor (alpha)
            alpha: float = self._calculate_intentional_alpha(
                event.onset_ticks, expression_map
            )

            # 2. Sample the unintentional biomechanical motor residual error (zeta)
            # Equation: zeta ~ N(0, sigma_d^2)
            zeta: float = float(self.rng.normal(0.0, self.sigma_d))

            # 3. Calculate and clamp the final realized performance length
            # Equation: d' = max(d_min, round(d * alpha + zeta))
            calculated_duration = int(np.round(event.duration_ticks * alpha + zeta))
            final_duration: int = max(self.d_min, calculated_duration)

            modified_event = NoteEvent(
                pitch=event.pitch,
                onset_ticks=event.onset_ticks,
                duration_ticks=final_duration,
                velocity=event.velocity,
            )
            modified_events.append(modified_event)

        perturbed_score = PerformanceScore(source=score.source)
        perturbed_score.events = modified_events
        return perturbed_score

    def _calculate_intentional_alpha(
        self, onset_tick: int, expression_map: ScoreExpressionMap
    ) -> float:
        """Determines the baseline touch scaling factor by analyzing local spanners and marks."""
        # Check explicit note-level technical markings first
        local_marks: set[str] = expression_map.local_articulations.get(
            onset_tick, set()
        )

        if "staccato" in local_marks or "staccatissimo" in local_marks:
            return float(self.rng.uniform(*self.range_staccato))

        if "tenuto" in local_marks:
            return float(self.rng.uniform(0.98, 1.05))

        # Check continuous structural spanners (Slur phrase groups indicating legato connection)
        for start_tick, end_tick in expression_map.slur_phrases:
            if start_tick <= onset_tick <= end_tick:
                return float(self.rng.uniform(*self.range_legato))

        # Default fallback context for unmarked musical events
        return float(self.rng.uniform(*self.range_neutral))
