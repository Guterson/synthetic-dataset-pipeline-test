"""Continuous tempo map and phrasing contour generation engine.

Implements Natural Cubic Spline interpolation over stochastic behavioral anchors
to simulate biomechanical inertia and human temporal phrasing variations.
"""

from typing import Final

import numpy as np
from scipy.interpolate import CubicSpline

from score2dataset.datamodels import NoteEvent, PerformanceScore, ScoreExpressionMap


class TempoContourGenerator:
    """Derives smooth, C2-continuous tempo maps satisfying physical boundary constraints."""

    # High-visibility translation table derived exactly from Table 1 of the paper
    _STYLE_MAPPINGS: Final[dict[str, tuple[float, float]]] = {
        "grave": (25.0, 45.0),
        "largo": (40.0, 60.0),
        "lento": (45.0, 60.0),
        "adagio": (66.0, 76.0),
        "andante": (76.0, 108.0),
        "moderato": (108.0, 120.0),
        "allegro": (120.0, 168.0),
        "presto": (168.0, 200.0),
        "prestissimo": (200.0, 250.0),
    }

    # Strict physical safety limits derived from human motor noise and grid quantization thresholds
    _B_MIN: Final[float] = 12.5
    _B_MAX: Final[float] = 250.0

    def __init__(self, theta_sigma: float = 0.15, seed: int | None = None) -> None:
        """Initializes the contour generator with randomization bounds and state handles.

        Args:
            theta_sigma: Structural domain randomization boundary. Defaults to 0.15.
            seed: Fixed stochastic seed to ensure scientific transparency.
        """
        self.theta_sigma: Final[float] = theta_sigma
        self.rng: Final[np.random.Generator] = np.random.default_rng(seed)

    def generate_tempo_map(self, metadata: ScoreExpressionMap) -> CubicSpline:
        """Constructs a continuous cubic spline mapping across phrasing anchors.

        Args:
            metadata: Structured expression and structural metadata object.

        Returns:
            A SciPy CubicSpline instance mapping absolute tick inputs to continuous BPM.
        """
        ticks_per_bar: int = metadata.beats_per_bar * metadata.ticks_per_beat

        # 1. Resolve primary/initial tempo anchor speed
        base_style = metadata.initial_tempo_marking.lower().strip()
        low_b, high_b = self._STYLE_MAPPINGS.get(base_style, (76.0, 108.0))
        primary_base_bpm: float = float(self.rng.uniform(low_b, high_b))

        # Skeletal points tracking: keys = absolute ticks, values = target BPM
        anchors: dict[int, float] = {0: primary_base_bpm}

        # Process score timeline milestones sequentially to build the anchors set
        current_running_bpm: float = primary_base_bpm
        sorted_directions = sorted(metadata.text_directions.items())

        for tick, marking in sorted_directions:
            norm_marking = marking.lower().strip()

            if norm_marking in self._STYLE_MAPPINGS:
                low_b, high_b = self._STYLE_MAPPINGS[norm_marking]
                current_running_bpm = float(self.rng.uniform(low_b, high_b))
                anchors[tick] = current_running_bpm

            elif norm_marking in ["accelerando", "accel"]:
                gamma_sigma = float(self.rng.uniform(1.1, 2.0))
                target_bpm = current_running_bpm * gamma_sigma

                # Check if a terminal marker exists ahead, otherwise enforce an interpretative horizon
                horizon_bars = float(self.rng.uniform(2.0, 4.0))
                horizon_ticks = int(horizon_bars * ticks_per_bar)
                end_tick = min(tick + horizon_ticks, metadata.total_ticks)

                current_running_bpm = target_bpm
                anchors[end_tick] = current_running_bpm

            elif norm_marking in ["ritardando", "rit", "rallentando", "rall"]:
                gamma_sigma = float(self.rng.uniform(0.5, 0.9))
                target_bpm = current_running_bpm * gamma_sigma

                horizon_bars = float(self.rng.uniform(2.0, 4.0))
                horizon_ticks = int(horizon_bars * ticks_per_bar)
                end_tick = min(tick + horizon_ticks, metadata.total_ticks)

                current_running_bpm = target_bpm
                anchors[end_tick] = current_running_bpm

            elif norm_marking == "tempo_primo":
                current_running_bpm = primary_base_bpm
                anchors[tick] = current_running_bpm

        # 2. Structural Fallback: Generate bar-line anchors to avoid mechanical tracking rigidity
        for tick in range(0, metadata.total_ticks, ticks_per_bar):
            if tick not in anchors:
                # Equation: B_anchor ~ U(B_score * [1 - theta], B_score * [1 + theta])
                min_scale = 1.0 - self.theta_sigma
                max_scale = 1.0 + self.theta_sigma
                random_scale = float(self.rng.uniform(min_scale, max_scale))
                anchors[tick] = current_running_bpm * random_scale

        # Enforce boundary anchor closure at the absolute end of the score timeline
        if metadata.total_ticks not in anchors:
            anchors[metadata.total_ticks] = current_running_bpm

        # 3. Compile vectors and build the continuous C2-continuous Cubic Spline
        sorted_anchors = sorted(anchors.items())
        knots_x = np.array([item[0] for item in sorted_anchors], dtype=np.float64)
        values_y = np.array([item[1] for item in sorted_anchors], dtype=np.float64)

        # Enforce physical saturation ceilings immediately over knot parameters
        values_y = np.clip(values_y, self._B_MIN, self._B_MAX)

        # Natural boundary configuration forces second derivatives to zero, preventing wild oscillations
        return CubicSpline(knots_x, values_y, bc_type="natural")

    def apply_phrasing_to_score(
        self, score: PerformanceScore, tempo_spline: CubicSpline
    ) -> PerformanceScore:
        """Evaluates the continuous spline map to modify absolute note timing grids.

        Args:
            score: The raw, canonical performance data container.
            tempo_spline: The pre-calculated continuous tempo map spline instance.

        Returns:
            A new modified PerformanceScore instance with realigned time coordinates.
        """
        perturbed_score = PerformanceScore(source=score.source)
        modified_events: list[NoteEvent] = []

        for event in score.events:
            # Evaluate the instantaneous cubic polynomial at the specific note onset tick coordinate
            instantaneous_bpm: float = float(tempo_spline(event.onset_ticks))
            instantaneous_bpm = max(self._B_MIN, min(self._B_MAX, instantaneous_bpm))

            # Calculate a microtiming scaling scalar to adjust performance properties
            # Higher instantaneous BPM means a faster clock speed, shortening relative durations
            speed_ratio: float = (
                120.0 / instantaneous_bpm
            )  # Normalized relative to a neutral 120 BPM midpoint

            modified_event = NoteEvent(
                pitch=event.pitch,
                onset_ticks=event.onset_ticks,  # Leave absolute grid position intact for metadata sequencing
                duration_ticks=int(np.round(event.duration_ticks * speed_ratio)),
                velocity=event.velocity,
            )
            modified_events.append(modified_event)

        perturbed_score.events = modified_events
        return perturbed_score
