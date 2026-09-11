import numpy as np


class ParameterExtractor:
    """Extracts RF signal parameters like SNR, Bandwidth, and Peak Power

    from raw IQ or WAV samples.
    """

    def __init__(self, signal_samples: np.ndarray, sample_rate: float):
        self.samples = signal_samples
        self.sample_rate = sample_rate

    def compute_snr(self) -> float:
        """Estimates Signal-to-Noise Ratio (SNR) in dB."""
        signal_power = np.mean(np.abs(self.samples) ** 2)
        noise_floor = np.percentile(np.abs(self.samples) ** 2, 10)
        if noise_floor <= 0:
            return 0.0
        snr_db = 10 * np.log10(signal_power / noise_floor)
        return float(round(snr_db, 2))

    def compute_bandwidth(self, threshold_db: float = 3.0) -> float:
        """Estimates -3dB occupied bandwidth using FFT spectrum."""
        fft_data = np.abs(np.fft.fft(self.samples))
        fft_data = np.fft.fftshift(fft_data)
        max_power = np.max(fft_data)

        if max_power == 0:
            return 0.0

        # Find frequencies above threshold
        threshold_linear = max_power * (10 ** (-threshold_db / 10))
        active_bins = np.where(fft_data >= threshold_linear)[0]

        if len(active_bins) == 0:
            return 0.0

        bin_width = self.sample_rate / len(self.samples)
        bandwidth = (active_bins[-1] - active_bins[0]) * bin_width
        return float(round(bandwidth, 2))

    def get_full_extraction_report(
        self, modulation_type: str = "Unknown"
    ) -> dict:
        """Generates a complete dictionary of extracted signal parameters."""
        peak_power = float(np.max(np.abs(self.samples) ** 2))
        avg_power = float(np.mean(np.abs(self.samples) ** 2))

        return {
            "sample_rate": self.sample_rate,
            "total_samples": len(self.samples),
            "snr_db": self.compute_snr(),
            "bandwidth_3db_hz": self.compute_bandwidth(),
            "peak_power_db": round(10 * np.log10(peak_power + 1e-12), 2),
            "avg_power_db": round(10 * np.log10(avg_power + 1e-12), 2),
            "modulation_type": modulation_type,
        }


if __name__ == "__main__":
    # Test Run with synthetic test signal
    fs = 2000000  # 2 MHz
    t = np.linspace(0, 0.001, int(fs * 0.001))
    test_iq = np.exp(1j * 2 * np.pi * 100000 * t) + 0.1 * (
        np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    )

    extractor = ParameterExtractor(test_iq, sample_rate=fs)
    report = extractor.get_full_extraction_report(modulation_type="QPSK")

    print("Parameter Extraction Report:")
    for k, v in report.items():
        print(f" - {k}: {v}")