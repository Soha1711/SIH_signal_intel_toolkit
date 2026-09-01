import numpy as np
import matplotlib.pyplot as plt

from param_estimation.signal_analysis import analyze_signal


# ============================================================
# 1. TEST SIGNAL SETTINGS
# ============================================================

sample_rate = 2_000_000

duration = 0.02

frequency_1 = 100_000

frequency_2 = 250_000


# ============================================================
# 2. CREATE TIME ARRAY
# ============================================================

time = np.arange(
    0,
    duration,
    1 / sample_rate
)


# ============================================================
# 3. CREATE COMPLEX IQ SIGNAL
# ============================================================

signal = (
    0.8
    * np.exp(
        1j
        * 2
        * np.pi
        * frequency_1
        * time
    )
    +
    0.3
    * np.exp(
        1j
        * 2
        * np.pi
        * frequency_2
        * time
    )
)


# ============================================================
# 4. ADD SMALL COMPLEX NOISE
# ============================================================

noise = (
    0.03
    *
    (
        np.random.randn(len(signal))
        +
        1j
        * np.random.randn(len(signal))
    )
)


signal = signal + noise


# Convert to complex64
signal = signal.astype(
    np.complex64
)


# ============================================================
# 5. CREATE INGESTION-STYLE RESULT
# ============================================================

result = {

    "samples": signal,

    "sample_rate": sample_rate,

    "source_format": "iq",

    "duration_sec": duration,
}


# ============================================================
# 6. RUN YOUR SIGNAL ANALYSIS
# ============================================================

result = analyze_signal(
    result
)


# ============================================================
# 7. CHECK RESULT
# ============================================================

print()
print("=" * 60)
print("SIH 2026 - SIGNAL ANALYSIS TEST")
print("=" * 60)

print(
    "Analysis status:",
    result.get("analysis_status")
)

print(
    "Number of samples:",
    len(result["samples"])
)

print(
    "Sample rate:",
    result["sample_rate"],
    "Hz"
)

print(
    "Waveform preview points:",
    len(result["waveform_time"])
)

print(
    "FFT points:",
    len(result["fft_frequency"])
)

print(
    "PSD points:",
    len(result["psd_frequency"])
)

print(
    "Peak frequency:",
    result["peak_frequency"],
    "Hz"
)

print("=" * 60)


# ============================================================
# 8. STOP IF ANALYSIS FAILED
# ============================================================

if result.get("analysis_status") != "success":

    print(
        "ERROR:",
        result.get("error")
    )

    raise SystemExit


# ============================================================
# 9. GET WAVEFORM DATA
# ============================================================

waveform_time = (
    result["waveform_time"]
)

waveform_i = (
    result["waveform_i"]
)

waveform_q = (
    result["waveform_q"]
)


# Convert seconds to milliseconds

waveform_time_ms = (
    waveform_time * 1000
)


# ============================================================
# 10. SHOW WAVEFORM
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    waveform_time_ms,
    waveform_i,
    label="I (In-phase)"
)

plt.plot(
    waveform_time_ms,
    waveform_q,
    label="Q (Quadrature)"
)

plt.title(
    "Time-Domain I/Q Waveform"
)

plt.xlabel(
    "Time (ms)"
)

plt.ylabel(
    "Amplitude"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


# ============================================================
# 11. GET FFT DATA
# ============================================================

fft_frequency = (
    result["fft_frequency"]
)

fft_magnitude_db = (
    result["fft_magnitude_db"]
)


# Convert Hz to kHz

fft_frequency_khz = (
    fft_frequency / 1000
)


# ============================================================
# 12. SHOW FFT
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    fft_frequency_khz,
    fft_magnitude_db
)

plt.title(
    "FFT / Magnitude Spectrum"
)

plt.xlabel(
    "Frequency (kHz)"
)

plt.ylabel(
    "Magnitude (dB)"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 13. GET PSD DATA
# ============================================================

psd_frequency = (
    result["psd_frequency"]
)

psd_db = (
    result["psd_db"]
)


# Convert Hz to kHz

psd_frequency_khz = (
    psd_frequency / 1000
)


# ============================================================
# 14. SHOW PSD
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    psd_frequency_khz,
    psd_db
)

plt.title(
    "Power Spectral Density"
)

plt.xlabel(
    "Frequency (kHz)"
)

plt.ylabel(
    "PSD (dB/Hz)"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 15. DISPLAY ALL THREE WINDOWS
# ============================================================

plt.show()
