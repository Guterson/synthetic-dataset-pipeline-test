"""Event-level stochastic modification processor for symbolic performance data.

Implements categorical, discrete Gaussian, and Poisson distributions to model
human motor deviations, technical execution failures, and additive ghost strikes.
"""

import copy
from typing import Final

import numpy as np

from score2dataset.datamodels import NoteEvent, PerformanceScore
from score2dataset.exceptions import ProcessorError


class EventLevelModifier:
    """Applies probabilistic modifications and verifies structural integrity bounds."""

    def __init__(
        self,
        p_keep: float = 1.0,
        p_sub: float = 0.0,
        p_omit: float = 0.0,
        sigma_p: float = 1.2,
        poisson_lambda: float = 0.0,
        threshold_gamma: float = 0.15,
        w_sub: float = 0.2,
        w_omit: float = 0.5,
        w_ins: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initializes the stochastic deviation matrices and weight boundaries.

        Args:
            p_keep: Probability of note preservation.
            p_sub: Probability of note pitch substitution.
            p_omit: Probability of note omission.
            sigma_p: Standard deviation for the discrete Gaussian pitch shifts.
            poisson_lambda: Expected count (lambda) of random additive insertions.
            threshold_gamma: Maximum allowed cumulative structural noise density.
            w_sub: Penalty weight assigned to substitution events.
            w_omit: Penalty weight assigned to omission events.
            w_ins: Penalty weight assigned to insertion events.
            seed: Fixed stochastic seed to ensure strict scientific transparency.
        """
        # Enforce mathematical validity of the categorical distribution array
        prob_sum: float = p_keep + p_sub + p_omit
        if not np.isclose(prob_sum, 1.0):
            raise ProcessorError(
                f"Categorical probabilities must sum to 1.0. Got: {prob_sum}"
            )

        self.probabilities: Final[list[float]] = [p_keep, p_sub, p_omit]
        self.sigma_p: Final[float] = sigma_p
        self.poisson_lambda: Final[float] = poisson_lambda
        self.gamma: Final[float] = threshold_gamma

        # Weight coefficients mapping functional impact on onset tracking
        self.w_sub: Final[float] = w_sub
        self.w_omit: Final[float] = w_omit
        self.w_ins: Final[float] = w_ins

        # Instantiate the isolated state-safe random generator wrapper
        self.rng: Final[np.random.Generator] = np.random.default_rng(seed)

    def perturb_score(self, canonical_score: PerformanceScore) -> PerformanceScore:
        """Applies categorical mutations and additive ghost noise to a performance.

        Args:
            canonical_score: The pristine target template container to modify.

        Returns:
            A new modified PerformanceScore instance if it passes structural bounds.

        Raises:
            ProcessorError: If the cumulative noise violates the integrity threshold.
        """
        original_events: list[NoteEvent] = canonical_score.events
        n_canonical: int = len(original_events)
        if n_canonical == 0:
            return copy.deepcopy(canonical_score)

        modified_events: list[NoteEvent] = []
        n_sub: int = 0
        n_omit: int = 0

        # Outcome mappings -> 0: Keep, 1: Substitute, 2: Omit
        outcomes = self.rng.choice(3, size=n_canonical, p=self.probabilities)

        for idx, event in enumerate(original_events):
            outcome = outcomes[idx]

            if outcome == 0:  # Keep
                modified_events.append(copy.deepcopy(event))

            elif outcome == 1:  # Substitution
                # Sample from a discrete Gaussian using rounding transforms
                pitch_shift = int(np.round(self.rng.normal(0, self.sigma_p)))
                if pitch_shift == 0:
                    pitch_shift = 1 if self.rng.random() > 0.5 else -1

                mutated_event = NoteEvent(
                    pitch=int(np.clip(event.pitch + pitch_shift, 0, 127)),
                    onset_ticks=event.onset_ticks,
                    duration_ticks=event.duration_ticks,
                    velocity=event.velocity,
                )
                modified_events.append(mutated_event)
                n_sub += 1

            elif outcome == 2:  # Omission
                n_omit += 1
                continue

        # Poisson random noise arrival for additive ghost strikes
        n_ins: int = int(self.rng.poisson(self.poisson_lambda))

        if n_ins > 0 and len(modified_events) > 0:
            # Uniformly select parent nodes across the newly modified sequence
            target_indices = self.rng.integers(0, len(modified_events), size=n_ins)

            for target_idx in target_indices:
                parent_note = modified_events[target_idx]
                ghost_shift = int(np.round(self.rng.normal(0, self.sigma_p)))
                if ghost_shift == 0:
                    ghost_shift = 1 if self.rng.random() > 0.5 else -1

                ghost_event = NoteEvent(
                    pitch=int(np.clip(parent_note.pitch + ghost_shift, 0, 127)),
                    # Ghost note inherits temporal coordinate and velocity completely
                    onset_ticks=parent_note.onset_ticks,
                    duration_ticks=parent_note.duration_ticks,
                    velocity=parent_note.velocity,
                )
                modified_events.append(ghost_event)

        # Structural Integrity Threshold Evaluation (Equation check)
        cumulative_noise_density: float = (
            (self.w_sub * n_sub) + (self.w_omit * n_omit) + (self.w_ins * n_ins)
        ) / n_canonical

        if cumulative_noise_density >= self.gamma:
            raise ProcessorError(
                f"Structural Integrity Violation: Cumulative noise density "
                f"({cumulative_noise_density:.4f}) exceeded threshold Gamma ({self.gamma})."
            )

        # Reassemble and sort the finalized sequence timeline
        # Sorting guarantees that chord notes and ghost notes stay sequentially linear
        modified_events.sort(key=lambda e: e.onset_ticks)

        perturbed_score = PerformanceScore(source=canonical_score.source)
        perturbed_score.events = modified_events
        return perturbed_score
