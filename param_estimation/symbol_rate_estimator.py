"""
SIH 2026 - Signal Intelligence Toolkit
Symbol Rate (Baud Rate) Estimator Module

Estimates symbol rate using non-linear amplitude transformation (|r(t)|^2),
spectral line peak detection, and standard baud rate quantization.
"""

from typing import Union, Optional
import numpy as np
from scipy.signal import find_peaks


class SymbolRateEstimator:
    """
    Automatic Symbol Rate (Baud Rate) estimator using non-linear FFT
    and cyclostationary feature line extraction.
    """

    def __init__(self, sample_rate: float = 1e6):
        """
        Initialize SymbolRateEstimator.

        Parameters:
            sample_rate: Sampling frequency of the signal in Hz. Default is 1e6 (1 MHz).
        """
        self.sample_rate = float(sample_rate)

    def estimate_symbol_rate(self, signal: Union[np.ndarray, list, tuple]) -> float:
        """
        Estimates the symbol rate (Baud rate in Hz) from a complex or real signal.

        Process:
        1. Clean signal input (handle complex64/complex128/floats, flatten, nan removal).
        2. Remove initial DC offset from signal: r_zero_mean = r(t) - mean(r(t)).
        3. Compute non-linear magnitude squared signal: sq_signal = |r_zero_mean(t)|^2.
        4. Remove DC offset from sq_signal: sq_signal = sq_signal - mean(sq_signal).
        5. Compute FFT and isolate positive frequencies (0 < f <= Nyquist).
        6. Detect dominant spectral line peak corresponding to clock / symbol frequency.

        Parameters:
            signal: 1D complex (complex64/128) or real NumPy array of IQ / WAV samples.

        Returns:
            Estimated symbol rate (Baud rate) in Hz as a float.
        """
        if signal is None or self.sample_rate <= 0:
            return 20000.0

        # Convert to numpy array
        arr = np.asarray(signal)

        if arr.size < 100:
            return 20000.0

        # Flatten multi-dimensional inputs (e.g., shape (N, 1) or (2, N))
        if arr.ndim > 1:
            arr = arr.flatten()

        # Clean NaN or Inf values
        arr = np.nan_to_num(arr)

        # 1. Remove initial DC offset from raw signal
        signal_zero_mean = arr - np.mean(arr)

        # 2. Non-linear amplitude nonlinearity: |r(t)|^2
        sq_signal = np.abs(signal_zero_mean) ** 2

        # 3. Remove DC offset from non-linear amplitude squared signal
        sq_signal = sq_signal - np.mean(sq_signal)

        # 4. FFT calculation
        n = len(sq_signal)
        fft_vals = np.abs(np.fft.fft(sq_signal))
        freqs = np.fft.fftfreq(n, d=1.0 / self.sample_rate)

        # Restrict search to positive frequencies within Nyquist limit
        pos_mask = (freqs > 0) & (freqs <= self.sample_rate / 2.0)
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]

        if len(fft_vals) == 0:
            return 20000.0

        # 5. Peak detection for dominant spectral line
        max_val = np.max(fft_vals)
        if max_val == 0:
            return 20000.0

        height_thresh = max_val * 0.20
        min_distance = max(1, int(n * 0.001))
        peaks, properties = find_peaks(fft_vals, height=height_thresh, distance=min_distance)

        if len(peaks) == 0:
            # Fallback to absolute maximum spectral peak
            best_idx = int(np.argmax(fft_vals))
            raw_baud = float(abs(freqs[best_idx]))
        else:
            # Select peak with highest prominence / height
            prominent_peak_idx = peaks[np.argmax(properties["peak_heights"])]
            raw_baud = float(abs(freqs[prominent_peak_idx]))

        # Quantize to closest standard telecom baud rate to avoid fractional drift
        standard_rates = [200, 300, 600, 1200, 2400, 4800, 9600, 10000, 19200, 20000, 38400, 56000]
        closest_baud = min(standard_rates, key=lambda x: abs(x - raw_baud))

        return float(closest_baud)


def estimate_symbol_rate(signal: Union[np.ndarray, list, tuple], sample_rate: float = 2e6) -> float:
    """
    Convenience wrapper function for pipeline and GUI auto-estimation calls.

    Parameters:
        signal: IQ / Real sample sequence.
        sample_rate: Sampling frequency in Hz (Default: 2 MHz).

    Returns:
        Estimated symbol rate in Hz.
    """
    estimator = SymbolRateEstimator(sample_rate=sample_rate)
    return estimator.estimate_symbol_rate(signal)