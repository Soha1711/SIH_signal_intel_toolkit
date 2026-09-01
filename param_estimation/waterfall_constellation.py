"""
SIH 2026 - Waterfall and Constellation Analysis

Responsibilities:
1. Generate waterfall / spectrogram data using STFT
2. Generate constellation I/Q data
3. Return data in a form that can be plotted by the GUI

Input:
    Complex NumPy samples
    Sampling rate

Output:
    Dictionary containing waterfall and constellation data
"""

import numpy as np
from scipy.signal import stft


# ============================================================
# 1. WATERFALL / SPECTROGRAM
# ============================================================

def compute_waterfall(
    signal,
    sample_rate,
    nperseg=1024,
    noverlap=None
):
    """
    Compute a waterfall / spectrogram representation.

    Parameters
    ----------
    signal : array-like
        Complex IQ samples.

    sample_rate : float
        Sampling frequency in Hz.

    nperseg : int
        Number of samples per STFT segment.

    noverlap : int or None
        Number of overlapping samples.

    Returns
    -------
    frequencies : ndarray
        Frequency axis in Hz.

    times : ndarray
        Time axis in seconds.

    power_db : ndarray
        Signal power in dB.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        raise ValueError(
            "Signal is empty."
        )

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    # Prevent segment size from exceeding signal length
    nperseg = min(
        int(nperseg),
        len(signal)
    )

    if nperseg < 2:
        raise ValueError(
            "Signal is too short for waterfall analysis."
        )

    # Default overlap = 50%
    if noverlap is None:
        noverlap = nperseg // 2

    # Make sure overlap is valid
    noverlap = min(
        int(noverlap),
        nperseg - 1
    )

    # STFT
    frequencies, times, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None
    )

    # Shift zero frequency to center
    frequencies = np.fft.fftshift(
        frequencies
    )

    spectrum = np.fft.fftshift(
        spectrum,
        axes=0
    )

    # Convert magnitude to power
    power = np.abs(spectrum) ** 2

    # Convert power to dB
    power_db = 10 * np.log10(
        power + 1e-12
    )

    return (
        frequencies,
        times,
        power_db
    )


# ============================================================
# 2. CONSTELLATION
# ============================================================

def compute_constellation(
    signal,
    max_points=10000
):
    """
    Prepare I/Q samples for constellation plotting.

    Parameters
    ----------
    signal : array-like
        Complex IQ samples.

    max_points : int
        Maximum number of points returned.

    Returns
    -------
    i_data : ndarray
        In-phase samples.

    q_data : ndarray
        Quadrature samples.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        raise ValueError(
            "Signal is empty."
        )

    if max_points <= 0:
        raise ValueError(
            "max_points must be greater than zero."
        )

    # Limit number of points for GUI performance
    if len(signal) > max_points:

        indices = np.linspace(
            0,
            len(signal) - 1,
            max_points,
            dtype=int
        )

        signal = signal[indices]

    # Separate I and Q
    i_data = np.real(signal)
    q_data = np.imag(signal)

    return (
        i_data,
        q_data
    )


# ============================================================
# 3. COMPLETE WATERFALL + CONSTELLATION ANALYSIS
# ============================================================

def analyze_waterfall_constellation(
    signal,
    sample_rate,
    nperseg=1024,
    noverlap=None,
    max_constellation_points=10000
):
    """
    Perform both waterfall and constellation analysis.

    Returns a dictionary containing all data
    required by the GUI.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        raise ValueError(
            "Signal is empty."
        )

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    # --------------------------------------------------------
    # WATERFALL
    # --------------------------------------------------------

    (
        waterfall_frequency,
        waterfall_time,
        waterfall_power_db
    ) = compute_waterfall(
        signal,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap
    )

    # --------------------------------------------------------
    # CONSTELLATION
    # --------------------------------------------------------

    (
        constellation_i,
        constellation_q
    ) = compute_constellation(
        signal,
        max_points=max_constellation_points
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "waterfall_frequency":
            waterfall_frequency,

        "waterfall_time":
            waterfall_time,

        "waterfall_power_db":
            waterfall_power_db,

        "constellation_i":
            constellation_i,

        "constellation_q":
            constellation_q
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    # Generate a test QPSK-like signal

    sample_rate = 100000

    num_samples = 20000

    rng = np.random.default_rng(42)

    symbols = rng.choice(
        [
            1 + 1j,
            1 - 1j,
            -1 + 1j,
            -1 - 1j
        ],
        size=num_samples
    )

    signal = symbols.astype(
        np.complex64
    )

    result = analyze_waterfall_constellation(
        signal,
        sample_rate
    )

    print(
        "Waterfall frequency bins:",
        len(result["waterfall_frequency"])
    )

    print(
        "Waterfall time bins:",
        len(result["waterfall_time"])
    )

    print(
        "Waterfall matrix shape:",
        result["waterfall_power_db"].shape
    )

    print(
        "Constellation points:",
        len(result["constellation_i"])
    )"""
SIH 2026 - Waterfall and Constellation Analysis

Responsibilities:
1. Generate waterfall / spectrogram data using STFT
2. Generate constellation I/Q data
3. Return data in a form that can be plotted by the GUI

Input:
    Complex NumPy samples
    Sampling rate

Output:
    Dictionary containing waterfall and constellation data
"""

import numpy as np
from scipy.signal import stft


# ============================================================
# 1. WATERFALL / SPECTROGRAM
# ============================================================

def compute_waterfall(
    signal,
    sample_rate,
    nperseg=1024,
    noverlap=None
):
    """
    Compute a waterfall / spectrogram representation.

    Parameters
    ----------
    signal : array-like
        Complex IQ samples.

    sample_rate : float
        Sampling frequency in Hz.

    nperseg : int
        Number of samples per STFT segment.

    noverlap : int or None
        Number of overlapping samples.

    Returns
    -------
    frequencies : ndarray
        Frequency axis in Hz.

    times : ndarray
        Time axis in seconds.

    power_db : ndarray
        Signal power in dB.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        raise ValueError(
            "Signal is empty."
        )

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    # Prevent segment size from exceeding signal length
    nperseg = min(
        int(nperseg),
        len(signal)
    )

    if nperseg < 2:
        raise ValueError(
            "Signal is too short for waterfall analysis."
        )

    # Default overlap = 50%
    if noverlap is None:
        noverlap = nperseg // 2

    # Make sure overlap is valid
    noverlap = min(
        int(noverlap),
        nperseg - 1
    )

    # STFT
    frequencies, times, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None
    )

    # Shift zero frequency to center
    frequencies = np.fft.fftshift(
        frequencies
    )

    spectrum = np.fft.fftshift(
        spectrum,
        axes=0
    )

    # Convert magnitude to power
    power = np.abs(spectrum) ** 2

    # Convert power to dB
    power_db = 10 * np.log10(
        power + 1e-12
    )

    return (
        frequencies,
        times,
        power_db
    )


# ============================================================
# 2. CONSTELLATION
# ============================================================

def compute_constellation(
    signal,
    max_points=10000
):
    """
    Prepare I/Q samples for constellation plotting.

    Parameters
    ----------
    signal : array-like
        Complex IQ samples.

    max_points : int
        Maximum number of points returned.

    Returns
    -------
    i_data : ndarray
        In-phase samples.

    q_data : ndarray
        Quadrature samples.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        raise ValueError(
            "Signal is empty."
        )

    if max_points <= 0:
        raise ValueError(
            "max_points must be greater than zero."
        )

    # Limit number of points for GUI performance
    if len(signal) > max_points:

        indices = np.linspace(
            0,
            len(signal) - 1,
            max_points,
            dtype=int
        )

        signal = signal[indices]

    # Separate I and Q
    i_data = np.real(signal)
    q_data = np.imag(signal)

    return (
        i_data,
        q_data
    )


# ============================================================
# 3. COMPLETE WATERFALL + CONSTELLATION ANALYSIS
# ============================================================

def analyze_waterfall_constellation(
    signal,
    sample_rate,
    nperseg=1024,
    noverlap=None,
    max_constellation_points=10000
):
    """
    Perform both waterfall and constellation analysis.

    Returns a dictionary containing all data
    required by the GUI.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )

    if signal.size == 0:
        raise ValueError(
            "Signal is empty."
        )

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    # --------------------------------------------------------
    # WATERFALL
    # --------------------------------------------------------

    (
        waterfall_frequency,
        waterfall_time,
        waterfall_power_db
    ) = compute_waterfall(
        signal,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap
    )

    # --------------------------------------------------------
    # CONSTELLATION
    # --------------------------------------------------------

    (
        constellation_i,
        constellation_q
    ) = compute_constellation(
        signal,
        max_points=max_constellation_points
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "waterfall_frequency":
            waterfall_frequency,

        "waterfall_time":
            waterfall_time,

        "waterfall_power_db":
            waterfall_power_db,

        "constellation_i":
            constellation_i,

        "constellation_q":
            constellation_q
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    # Generate a test QPSK-like signal

    sample_rate = 100000

    num_samples = 20000

    rng = np.random.default_rng(42)

    symbols = rng.choice(
        [
            1 + 1j,
            1 - 1j,
            -1 + 1j,
            -1 - 1j
        ],
        size=num_samples
    )

    signal = symbols.astype(
        np.complex64
    )

    result = analyze_waterfall_constellation(
        signal,
        sample_rate
    )

    print(
        "Waterfall frequency bins:",
        len(result["waterfall_frequency"])
    )

    print(
        "Waterfall time bins:",
        len(result["waterfall_time"])
    )

    print(
        "Waterfall matrix shape:",
        result["waterfall_power_db"].shape
    )

    print(
        "Constellation points:",
        len(result["constellation_i"])
    )
