"""
SIH 2026 - Signal Analysis Module

Responsibilities:
1. Time-domain I/Q waveform
2. FFT / Magnitude Spectrum
3. Power Spectral Density (PSD)
4. Waterfall / Spectrogram
5. Constellation Diagram

Input:
    result["samples"]
    result["sample_rate"]

Output:
    Original result dictionary + analysis results
"""

import numpy as np
from scipy.signal import welch, stft

from param_estimation.modulation_classifier import (
    detect_modulation_type,
)


# ============================================================
# 1. WAVEFORM
# ============================================================

def compute_waveform(
    signal,
    sample_rate,
    preview_duration=0.001,
    max_points=50000
):
    """
    Generate a readable time-domain I/Q waveform.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    preview_samples = int(
        preview_duration * sample_rate
    )

    preview_samples = max(
        preview_samples,
        1
    )

    preview_samples = min(
        preview_samples,
        len(signal)
    )

    preview = signal[:preview_samples]

    if len(preview) > max_points:

        step = int(
            np.ceil(
                len(preview) / max_points
            )
        )

        preview = preview[::step]

    i_data = np.real(preview)
    q_data = np.imag(preview)

    time = (
        np.arange(len(preview))
        / sample_rate
    )

    return time, i_data, q_data


# ============================================================
# 2. FFT
# ============================================================

def compute_fft(
    signal,
    sample_rate,
    max_samples=262144
):
    """
    Compute normalized FFT magnitude spectrum.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    n = min(
        len(signal),
        max_samples
    )

    fft_signal = signal[:n]

    window = np.hanning(n)

    windowed_signal = (
        fft_signal * window
    )

    spectrum = np.fft.fft(
        windowed_signal
    )

    frequencies = np.fft.fftfreq(
        n,
        d=1.0 / sample_rate
    )

    spectrum = np.fft.fftshift(
        spectrum
    )

    frequencies = np.fft.fftshift(
        frequencies
    )

    magnitude = (
        np.abs(spectrum)
        / np.sum(window)
    )

    magnitude_db = (
        20
        * np.log10(
            magnitude + 1e-12
        )
    )

    return frequencies, magnitude_db


# ============================================================
# 3. PEAK FREQUENCY
# ============================================================

def find_peak_frequency(
    frequencies,
    magnitude_db
):
    """
    Find strongest frequency component.
    """

    if len(frequencies) == 0:
        raise ValueError(
            "Frequency array is empty."
        )

    if len(magnitude_db) == 0:
        raise ValueError(
            "Magnitude array is empty."
        )

    peak_index = np.argmax(
        magnitude_db
    )

    return float(
        frequencies[peak_index]
    )


# ============================================================
# 4. PSD
# ============================================================

def compute_psd(
    signal,
    sample_rate,
    nperseg=4096
):
    """
    Compute Power Spectral Density using Welch's method.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    nperseg = min(
        nperseg,
        len(signal)
    )

    if nperseg < 2:
        raise ValueError(
            "Signal is too short for PSD analysis."
        )

    frequencies, psd = welch(
        signal,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density"
    )

    frequencies = np.fft.fftshift(
        frequencies
    )

    psd = np.fft.fftshift(
        psd
    )

    psd_db = (
        10
        * np.log10(
            psd + 1e-12
        )
    )

    return frequencies, psd_db


# ============================================================
# 5. WATERFALL / SPECTROGRAM
# ============================================================

def compute_waterfall(
    signal,
    sample_rate,
    nperseg=1024,
    noverlap=768
):
    """
    Compute time-frequency waterfall data.

    Returns:
        frequencies : frequency axis in Hz
        times       : time axis in seconds
        power_db    : power in dB
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    nperseg = min(
        nperseg,
        len(signal)
    )

    if nperseg < 2:
        raise ValueError(
            "Signal is too short for waterfall analysis."
        )

    noverlap = min(
        noverlap,
        nperseg - 1
    )

    frequencies, times, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None,
        padded=False
    )

    # Shift zero frequency to center
    frequencies = np.fft.fftshift(
        frequencies
    )

    spectrum = np.fft.fftshift(
        spectrum,
        axes=0
    )

    # Power
    power = np.abs(
        spectrum
    ) ** 2

    # Convert to dB
    power_db = (
        10
        * np.log10(
            power + 1e-12
        )
    )

    return (
        frequencies,
        times,
        power_db
    )


# ============================================================
# 6. CONSTELLATION
# ============================================================

def compute_constellation(
    signal,
    max_points=10000
):
    """
    Extract I/Q samples for constellation diagram.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    # Limit points for GUI performance
    if len(signal) > max_points:

        indices = np.linspace(
            0,
            len(signal) - 1,
            max_points,
            dtype=int
        )

        signal = signal[indices]

    i_data = np.real(signal)
    q_data = np.imag(signal)

    return (
        i_data,
        q_data
    )


# ============================================================
# 7. COMPLETE ANALYSIS
# ============================================================

def analyze_signal(
    result,
    preview_duration=0.001,
    max_waveform_points=50000,
    max_fft_samples=262144,
    psd_nperseg=4096,
    waterfall_nperseg=1024,
    waterfall_noverlap=768,
    max_constellation_points=10000
):
    """
    Perform complete signal analysis.

    Analysis:
        1. Waveform
        2. FFT
        3. Peak frequency
        4. PSD
        5. Waterfall
        6. Constellation
    """

    output = dict(result)

    try:

        # ----------------------------------------------------
        # CHECK INPUT
        # ----------------------------------------------------

        if "samples" not in output:
            raise KeyError(
                "Input result is missing 'samples'."
            )

        if "sample_rate" not in output:
            raise KeyError(
                "Input result is missing 'sample_rate'."
            )

        # ----------------------------------------------------
        # GET SIGNAL
        # ----------------------------------------------------

        signal = np.asarray(
            output["samples"]
        )

        sample_rate = float(
            output["sample_rate"]
        )

        # ----------------------------------------------------
        # WAVEFORM
        # ----------------------------------------------------

        (
            waveform_time,
            waveform_i,
            waveform_q
        ) = compute_waveform(
            signal,
            sample_rate,
            preview_duration=preview_duration,
            max_points=max_waveform_points
        )

        # ----------------------------------------------------
        # FFT
        # ----------------------------------------------------

        (
            fft_frequency,
            fft_magnitude_db
        ) = compute_fft(
            signal,
            sample_rate,
            max_samples=max_fft_samples
        )

        # ----------------------------------------------------
        # PEAK FREQUENCY
        # ----------------------------------------------------

        peak_frequency = find_peak_frequency(
            fft_frequency,
            fft_magnitude_db
        )

        # ----------------------------------------------------
        # PSD
        # ----------------------------------------------------

        (
            psd_frequency,
            psd_db
        ) = compute_psd(
            signal,
            sample_rate,
            nperseg=psd_nperseg
        )

        # ----------------------------------------------------
        # WATERFALL
        # ----------------------------------------------------

        (
            waterfall_frequency,
            waterfall_time,
            waterfall_power_db
        ) = compute_waterfall(
            signal,
            sample_rate,
            nperseg=waterfall_nperseg,
            noverlap=waterfall_noverlap
        )

        # ----------------------------------------------------
        # CONSTELLATION
        # ----------------------------------------------------

        (
            constellation_i,
            constellation_q
        ) = compute_constellation(
            signal,
            max_points=max_constellation_points
        )

        # ----------------------------------------------------
        # MODULATION TYPE DETECTION
        # ----------------------------------------------------

        (
            detected_modulation,
            modulation_confidence
        ) = detect_modulation_type(
            constellation_i,
            constellation_q
        )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        output["waveform_time"] = (
            waveform_time
        )

        output["waveform_i"] = (
            waveform_i
        )

        output["waveform_q"] = (
            waveform_q
        )

        output["fft_frequency"] = (
            fft_frequency
        )

        output["fft_magnitude_db"] = (
            fft_magnitude_db
        )

        output["peak_frequency"] = (
            peak_frequency
        )

        output["psd_frequency"] = (
            psd_frequency
        )

        output["psd_db"] = (
            psd_db
        )

        # ----------------------------------------------------
        # WATERFALL RESULTS
        # ----------------------------------------------------

        output["waterfall_frequency"] = (
            waterfall_frequency
        )

        output["waterfall_time"] = (
            waterfall_time
        )

        output["waterfall_power_db"] = (
            waterfall_power_db
        )

        # ----------------------------------------------------
        # CONSTELLATION RESULTS
        # ----------------------------------------------------

        output["constellation_i"] = (
            constellation_i
        )

        output["constellation_q"] = (
            constellation_q
        )

        # ----------------------------------------------------
        # MODULATION DETECTION RESULTS
        # ----------------------------------------------------

        output["detected_modulation"] = (
            detected_modulation
        )

        output["modulation_confidence"] = (
            modulation_confidence
        )

        output["analysis_status"] = (
            "success"
        )

    except Exception as error:

        output["analysis_status"] = (
            "failed"
        )

        output["error"] = str(error)

    return output
