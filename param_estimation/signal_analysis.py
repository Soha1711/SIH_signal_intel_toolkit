"""
SIH 2026 - Signal Analysis Module

Responsibilities:
1. Time-domain I/Q waveform
2. FFT / Magnitude Spectrum
3. Power Spectral Density (PSD)

Input:
    result["samples"]
    result["sample_rate"]

Output:
    Original result dictionary + analysis results
"""

import numpy as np
from scipy.signal import welch


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

    Only a small preview is selected for plotting so that
    very large recordings do not appear as a dense block.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    # Number of samples to display
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

    # Take only the preview section
    preview = signal[:preview_samples]

    # Reduce points if there are too many
    if len(preview) > max_points:

        step = int(
            np.ceil(
                len(preview) / max_points
            )
        )

        preview = preview[::step]

    # Separate I and Q
    i_data = np.real(preview)

    q_data = np.imag(preview)

    # Create time axis
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

    A Hann window is applied to reduce spectral leakage.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    # Limit FFT size for large recordings
    n = min(
        len(signal),
        max_samples
    )

    fft_signal = signal[:n]

    # Hann window
    window = np.hanning(n)

    windowed_signal = (
        fft_signal * window
    )

    # Calculate FFT
    spectrum = np.fft.fft(
        windowed_signal
    )

    # Frequency axis
    frequencies = np.fft.fftfreq(
        n,
        d=1.0 / sample_rate
    )

    # Put zero frequency in the center
    spectrum = np.fft.fftshift(
        spectrum
    )

    frequencies = np.fft.fftshift(
        frequencies
    )

    # Normalize magnitude
    magnitude = (
        np.abs(spectrum)
        / np.sum(window)
    )

    # Convert to dB
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
    Find the strongest frequency component.
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

    # Do not let segment size exceed signal length
    nperseg = min(
        nperseg,
        len(signal)
    )

    # Welch PSD
    frequencies, psd = welch(
        signal,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density"
    )

    # Center zero frequency
    frequencies = np.fft.fftshift(
        frequencies
    )

    psd = np.fft.fftshift(
        psd
    )

    # Convert to dB/Hz
    psd_db = (
        10
        * np.log10(
            psd + 1e-12
        )
    )

    return frequencies, psd_db


# ============================================================
# 5. COMPLETE ANALYSIS
# ============================================================

def analyze_signal(
    result,
    preview_duration=0.001,
    max_waveform_points=50000,
    max_fft_samples=262144,
    psd_nperseg=4096
):
    """
    Perform waveform, FFT and PSD analysis.

    The input dictionary is preserved and extended.

    Required input keys:

        result["samples"]
        result["sample_rate"]
    """

    # Make a copy so that we don't unexpectedly modify
    # the original dictionary.
    output = dict(result)

    try:

        # ----------------------------------------------------
        # Check required data
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
        # Get signal and sample rate
        # ----------------------------------------------------

        signal = np.asarray(
            output["samples"]
        )

        sample_rate = float(
            output["sample_rate"]
        )

        # ====================================================
        # WAVEFORM
        # ====================================================

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

        # ====================================================
        # FFT
        # ====================================================

        (
            fft_frequency,
            fft_magnitude_db
        ) = compute_fft(
            signal,
            sample_rate,
            max_samples=max_fft_samples
        )

        # ====================================================
        # PEAK FREQUENCY
        # ====================================================

        peak_frequency = find_peak_frequency(
            fft_frequency,
            fft_magnitude_db
        )

        # ====================================================
        # PSD
        # ====================================================

        (
            psd_frequency,
            psd_db
        ) = compute_psd(
            signal,
            sample_rate,
            nperseg=psd_nperseg
        )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        output["waveform_time"] = waveform_time

        output["waveform_i"] = waveform_i

        output["waveform_q"] = waveform_q

        output["fft_frequency"] = fft_frequency

        output["fft_magnitude_db"] = (
            fft_magnitude_db
        )

        output["peak_frequency"] = (
            peak_frequency
        )

        output["psd_frequency"] = (
            psd_frequency
        )

        output["psd_db"] = psd_db

        output["analysis_status"] = "success"

    except Exception as error:

        output["analysis_status"] = "failed"

        output["error"] = str(error)

    return output