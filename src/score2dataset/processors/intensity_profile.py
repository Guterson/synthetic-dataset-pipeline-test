"""Intensity profile and note velocity dynamic phrasing processor.

Implements piecewise linear interpolation across dynamic wedges, metric downbeat
accents, and Gaussian execution residuals to simulate human piano phrasing.
"""

import copy
from typing import Final

import numpy as np

from score2dataset.datamodels import NoteEvent, PerformanceScore, ScoreExpressionMap


class IntensityProfileModifier:
    """Modulates note velocities using dynamic curves, metric weight, and motor noise."""

    # High-visibility translation table mapped precisely to your 7-bit MIDI target range
    _NOMINAL_VELOCITIES: Final[dict[str, int]] = {
        "ppp": 16,
        "pp": 33,
        "p": 49,
        "mp": 64,
        "mf": 80,
        "f": 96,
        "ff": 112,
        "fff": 126,
    }

    def __init__(
        self,
        delta_v: int = 4,
        gamma_min: float = 1.2,
        gamma_max: float = 1.6,
        beta: float = 0.12,
        sigma_v: float = 3.5,
        seed: int | None = None,
    ) -> None:
        """Initializes the dynamic variance limits and random state handles.

        Args:
            delta_v: Fixed uniform offset spread around nominal velocity levels.
            gamma_min: Minimum scaling factor for crescendo dynamic gradients.
            gamma_max: Maximum scaling factor for crescendo dynamic gradients.
            beta: Maximum metric accent boost intensity (e.g., 0.12 for up to 12%).
            sigma_v: Standard deviation for the Gaussian human motor noise (xi).
            seed: Fixed stochastic seed to ensure strict scientific transparency.
        """
        self.delta_v: Final[int] = delta_v
        self.gamma_min: Final[float] = gamma_min
        self.gamma_max: Final[float] = gamma_max
        self.beta: Final[float] = beta
        self.sigma_v: Final[float] = sigma_v
        self.rng: Final[np.random.Generator] = np.random.default_rng(seed)

    def perturb_intensity(
        self, score: PerformanceScore, expression_map: ScoreExpressionMap
    ) -> PerformanceScore:
        """Evaluates anchors and spanners to smoothly project humanized velocity profiles.

        Args:
            score: The performance score structure containing target note events.
            expression_map: The extracted structural metadata and spanner coordinate tracker.

        Returns:
            A new modified PerformanceScore instance with humanized note velocities.
        """
        original_events: list[NoteEvent] = score.events
        if not original_events:
            return copy.deepcopy(score)

        ticks_per_bar: int = (
            expression_map.beats_per_bar * expression_map.ticks_per_beat
        )

        # Step 1: Establish absolute dynamic baseline anchors across the score timeline
        anchors: list[tuple[int, float]] = []
        current_baseline_v: float = float(
            self._NOMINAL_VELOCITIES.get(
                expression_map.initial_tempo_marking.lower().strip(), 80
            )
        )

        # Seed the initial boundary context at tick 0
        v_init = float(
            self.rng.uniform(
                current_baseline_v - self.delta_v, current_baseline_v + self.delta_v
            )
        )
        anchors.append((0, v_init))

        # Track explicit textual dynamic marks (p, mf, ff)
        for tick, marking in sorted(expression_map.text_directions.items()):
            norm_mark = marking.lower().strip()
            if norm_mark in self._NOMINAL_VELOCITIES:
                v_nom = self._NOMINAL_VELOCITIES[norm_mark]
                current_baseline_v = float(
                    self.rng.uniform(v_nom - self.delta_v, v_nom + self.delta_v)
                )
                anchors.append((tick, current_baseline_v))

        # Track continuous hairpins (Crescendo/Decrescendo wedges)
        for wedge in expression_map.dynamic_wedges:
            v_start = current_baseline_v
            gamma_v = float(self.rng.uniform(self.gamma_min, self.gamma_max))

            if wedge.wedge_type == "crescendo":
                v_end = v_start * gamma_v
            else:  # decrescendo / diminuendo
                v_end = v_start * (1.0 / gamma_v)

            anchors.append((wedge.onset_tick, v_start))
            anchors.append((wedge.offset_tick, v_end))
            current_baseline_v = v_end

        # Ensure the anchor array has closure at the absolute end of the track timeline
        if expression_map.total_ticks > anchors[-1][0]:
            anchors.append((expression_map.total_ticks, current_baseline_v))

        # Sort the anchor coordinates to prepare for error-free linear lookup
        anchors.sort(key=lambda x: x[0])

        modified_events: list[NoteEvent] = []

        # Step 2: Loop chronologically to project baseline, accents, and residuals
        for event in original_events:
            # Evaluate piecewise linear interpolation for the exact note onset tick
            v_interp = self._interpolate_velocity(event.onset_ticks, anchors)

            # Calculate the metric downbeat indicator
            is_downbeat: bool = (event.onset_ticks % ticks_per_bar) == 0

            # Dynamic Headroom Scale: Mitigates velocity compression as values near 127
            headroom_scale: float = max(0.0, (127.0 - v_interp) / 127.0)

            # Multiplicative metric accent adjusted smoothly by local headroom space
            if is_downbeat:
                v_metric = 1.0 + (self.beta * headroom_scale)
            else:
                v_metric = 1.0

            # Sample the stochastic human micro-dynamic residual error (xi)
            # Equation: xi ~ N(0, sigma_v^2)
            xi: float = float(self.rng.normal(0.0, self.sigma_v))

            # Step 3: Compute, round, and safely clamp the final 7-bit MIDI velocity
            # Equation: v_final = clip(round(V_interp * V_metric + xi), 1, 127)
            calculated_v = int(np.round(v_interp * v_metric + xi))
            final_velocity: int = max(1, min(127, calculated_v))

            modified_event = NoteEvent(
                pitch=event.pitch,
                onset_ticks=event.onset_ticks,
                duration_ticks=event.duration_ticks,
                velocity=final_velocity,
            )
            modified_events.append(modified_event)

        perturbed_score = PerformanceScore(source=score.source)
        perturbed_score.events = modified_events
        return perturbed_score

    def _interpolate_velocity(
        self, tick: int, anchors: list[tuple[int, float]]
    ) -> float:
        """Calculates the exact piecewise linear target between surrounding dynamic coordinates."""
        if tick <= anchors[0][0]:
            return anchors[0][1]
        if tick >= anchors[-1][0]:
            return anchors[-1][1]

        # Locate the exact surrounding anchor bracket bounding this specific note tick
        for i in range(len(anchors) - 1):
            k_a, v_a = anchors[i]
            k_b, v_b = anchors[i + 1]
            if k_a <= tick <= k_b:
                if k_b == k_a:
                    return v_a
                # Equation: V_interp = V_A + (V_B - V_A) * (k_i - k_A) / (k_B - k_A)
                return v_a + (v_b - v_a) * ((tick - k_a) / (k_b - k_a))

        return anchors[0][1]
