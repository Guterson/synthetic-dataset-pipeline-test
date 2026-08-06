"""System configuration and mathematical domain invariants for score2dataset."""

from typing import Final

# --- Hardware Specifications ---
STANDARD_TPQN: Final[int] = 480
# Absolute boundaries dictated by external data protocols or specifications.
MIN_MIDI_VELOCITY: Final[int] = 1
MAX_MIDI_VELOCITY: Final[int] = 127


# --- Biomechanical & sampling bounds ---
# Physical performance thresholds derived from human motor limitations.
HUMAN_MOTOR_NOISE_MS: Final[float] = 10.0


# --- Expressive Tempo Mappings ---
# 1 tick = exactly 10ms at 12.5 BPM (preserves human jitter on 480 TPQN grid)
MIN_BPM: Final[float] = 12.5

# Prevents event-order swaps that invalidate authoritative ground-truths
MAX_BPM: Final[float] = 250.0

NOMINAL_TEMPOS: Final[dict[str, tuple[float, float]]] = {
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

DEFAULT_BPM: Final[float] = 120.0

# --- Expressive dynamics mappings ---
# Human performance dynamics matrices and default target calibrations.
NOMINAL_VELOCITIES: Final[dict[str, int]] = {
    "ppp": 16,
    "pp": 33,
    "p": 49,
    "mp": 64,
    "mf": 80,
    "f": 96,
    "ff": 112,
    "fff": 126,
}

DEFAULT_VELOCITY: Final[int] = NOMINAL_VELOCITIES["mf"]
