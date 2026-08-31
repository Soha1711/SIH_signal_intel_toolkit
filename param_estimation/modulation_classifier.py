"""
SIH 2026 - Modulation Type Classifier

Detects the digital modulation type from
constellation (I/Q) data.

Supported modulation types:

    BPSK   - Binary Phase Shift Keying
             2 symbols, 1 bit/symbol
             Points are 180° apart

    QPSK   - Quadrature Phase Shift Keying
             4 symbols, 2 bits/symbol
             Points are 90° apart

    8-PSK  - 8-Phase Shift Keying
             8 symbols, 3 bits/symbol
             Equally spaced on unit circle

    16-QAM - 16-Quadrature Amplitude Modulation
             16 symbols, 4 bits/symbol
             4×4 grid (amplitude + phase)

    64-QAM - 64-Quadrature Amplitude Modulation
             64 symbols, 6 bits/symbol
             8×8 grid (higher data rate)
"""

import numpy as np


# ============================================================
# IDEAL CONSTELLATION GENERATORS
# ============================================================

def generate_bpsk():
    """
    BPSK: 2 signal points on the real axis.
    Points are 180° apart.
    1 bit per symbol.
    """

    return np.array(
        [-1.0 + 0j, 1.0 + 0j]
    )


def generate_qpsk():
    """
    QPSK: 4 signal points at 45°, 135°, 225°, 315°.
    Points are 90° apart.
    2 bits per symbol.
    """

    angles = (
        np.array([1, 3, 5, 7])
        * np.pi / 4
    )

    return np.exp(1j * angles)


def generate_8psk():
    """
    8-PSK: 8 equally spaced points on unit circle.
    3 bits per symbol.
    """

    angles = (
        2 * np.pi
        * np.arange(8)
        / 8
    )

    return np.exp(1j * angles)


def generate_16qam():
    """
    16-QAM: 4×4 rectangular grid.
    Uses both amplitude and phase.
    4 bits per symbol.
    """

    levels = np.array([-3, -1, 1, 3])

    points = []

    for i_val in levels:
        for q_val in levels:
            points.append(
                i_val + 1j * q_val
            )

    pts = np.array(points)

    return pts / np.max(np.abs(pts))


def generate_64qam():
    """
    64-QAM: 8×8 rectangular grid.
    6 bits per symbol.
    Higher data rate but more sensitive to noise.
    """

    levels = np.array(
        [-7, -5, -3, -1, 1, 3, 5, 7]
    )

    points = []

    for i_val in levels:
        for q_val in levels:
            points.append(
                i_val + 1j * q_val
            )

    pts = np.array(points)

    return pts / np.max(np.abs(pts))


# ============================================================
# CONSTELLATION REGISTRY
# ============================================================

MODULATION_TYPES = [
    "BPSK",
    "QPSK",
    "8-PSK",
    "16-QAM",
    "64-QAM",
]


_GENERATORS = {
    "BPSK":   generate_bpsk,
    "QPSK":   generate_qpsk,
    "8-PSK":  generate_8psk,
    "16-QAM": generate_16qam,
    "64-QAM": generate_64qam,
}


MODULATION_INFO = {

    "BPSK": {
        "name": "BPSK",
        "full_name": "Binary Phase Shift Keying",
        "symbols": 2,
        "bits_per_symbol": 1,
        "description": "2 points, 180° apart",
    },

    "QPSK": {
        "name": "QPSK",
        "full_name": "Quadrature Phase Shift Keying",
        "symbols": 4,
        "bits_per_symbol": 2,
        "description": "4 points, 90° apart",
    },

    "8-PSK": {
        "name": "8-PSK",
        "full_name": "8-Phase Shift Keying",
        "symbols": 8,
        "bits_per_symbol": 3,
        "description": "8 points on circle",
    },

    "16-QAM": {
        "name": "16-QAM",
        "full_name": "16-QAM",
        "symbols": 16,
        "bits_per_symbol": 4,
        "description": "4×4 grid",
    },

    "64-QAM": {
        "name": "64-QAM",
        "full_name": "64-QAM",
        "symbols": 64,
        "bits_per_symbol": 6,
        "description": "8×8 grid",
    },
}


# ============================================================
# GET IDEAL CONSTELLATION
# ============================================================

def get_ideal_constellation(mod_type):
    """
    Get ideal constellation points for a modulation type.

    Returns complex numpy array of ideal symbol positions,
    or None if mod_type is not recognized.
    """

    generator = _GENERATORS.get(mod_type)

    if generator is None:
        return None

    return generator()


# ============================================================
# MATCHING SCORE
# ============================================================

def _compute_matching_score(
    signal_norm,
    ideal_points
):
    """
    Compute how well signal points match
    an ideal constellation.

    Returns average minimum distance from each
    signal point to its nearest ideal symbol.

    Lower score = better match.
    """

    if (
        len(signal_norm) == 0
        or len(ideal_points) == 0
    ):
        return float('inf')

    # Subsample for performance
    max_check = min(
        len(signal_norm),
        5000
    )

    if len(signal_norm) > max_check:

        indices = np.random.choice(
            len(signal_norm),
            max_check,
            replace=False
        )

        check_signal = signal_norm[indices]

    else:

        check_signal = signal_norm

    # Distance from each check point
    # to each ideal point
    distances = np.abs(
        check_signal[:, None]
        - ideal_points[None, :]
    )

    # Minimum distance for each signal point
    min_distances = np.min(
        distances,
        axis=1
    )

    return float(np.mean(min_distances))


# ============================================================
# MODULATION DETECTION
# ============================================================

def detect_modulation_type(i_data, q_data):
    """
    Detect the modulation type from I/Q
    constellation data.

    Parameters:
        i_data : array of in-phase values
        q_data : array of quadrature values

    Returns:
        mod_type   : detected modulation type string
        confidence : confidence score (0.0 to 1.0)
    """

    signal = (
        np.asarray(i_data, dtype=float)
        + 1j
        * np.asarray(q_data, dtype=float)
    )

    if len(signal) == 0:
        return "Unknown", 0.0

    # Normalize to unit amplitude
    max_amp = np.max(np.abs(signal))

    if max_amp == 0:
        return "Unknown", 0.0

    signal_norm = signal / max_amp

    # Score each candidate modulation type
    scores = {}

    for mod_type in MODULATION_TYPES:

        ideal = get_ideal_constellation(
            mod_type
        )

        score = _compute_matching_score(
            signal_norm,
            ideal
        )

        scores[mod_type] = score

    # Find best match (lowest score)
    best_type = min(
        scores,
        key=scores.get
    )

    best_score = scores[best_type]

    # Convert score to confidence
    # Score 0.0 → confidence 1.0
    # Score ≥ 0.5 → confidence ≈ 0
    confidence = float(
        max(
            0.0,
            1.0 - 2.0 * best_score
        )
    )

    if confidence < 0.05:
        return "Unknown", confidence

    return best_type, confidence
